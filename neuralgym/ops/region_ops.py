"""Region / object mask ops for region-based image generation and editing.

Masks are float tensors in ``[0, 1]`` with shape ``[batch, height, width, 1]``
(NHWC, a single channel broadcast over image channels). A value of ``1``
means "inside the region to generate / manipulate", ``0`` means "keep".

"""
import math

import tensorflow as tf
if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
    tf = tf.compat.v1


def bbox_to_mask(bbox, height, width, batch_size=1, name='bbox_to_mask'):
    """Generate a rectangular region mask from a bounding box.

    Args:
        bbox: Tuple (top, left, box_height, box_width), python ints or
            scalar int tensors.
        height (int): Image height.
        width (int): Image width.
        batch_size (int): Batch size of the returned mask.

    Returns:
        tf.Tensor: float32 mask of shape [batch_size, height, width, 1].

    """
    with tf.variable_scope(name):
        top, left, box_h, box_w = bbox
        rows = tf.range(height)[:, None]
        cols = tf.range(width)[None, :]
        inside = tf.logical_and(
            tf.logical_and(rows >= top, rows < top + box_h),
            tf.logical_and(cols >= left, cols < left + box_w))
        mask = tf.cast(inside, tf.float32)[None, :, :, None]
        return tf.tile(mask, [batch_size, 1, 1, 1])


def object_mask(segmentation, object_ids, name='object_mask'):
    """Generate an object mask from a segmentation / instance map.

    Args:
        segmentation: Int tensor of shape [batch, height, width] or
            [batch, height, width, 1] holding a label id per pixel.
        object_ids: An int or list of ints, labels that belong to the object.

    Returns:
        tf.Tensor: float32 mask of shape [batch, height, width, 1].

    """
    with tf.variable_scope(name):
        segmentation = tf.convert_to_tensor(segmentation)
        if segmentation.get_shape().ndims == 4:
            segmentation = tf.squeeze(segmentation, axis=3)
        if not isinstance(object_ids, (list, tuple)):
            object_ids = [object_ids]
        ids = tf.cast(tf.constant(object_ids), segmentation.dtype)
        hit = tf.reduce_any(
            tf.equal(segmentation[..., None], ids), axis=-1, keepdims=True)
        return tf.cast(hit, tf.float32)


def resize_mask(mask, size, name='resize_mask'):
    """Bilinearly resize a mask, e.g. to match a feature map.

    Args:
        mask: Mask of shape [batch, height, width, 1].
        size: Tuple (height, width).

    Returns:
        tf.Tensor: Resized mask clipped to [0, 1].

    """
    with tf.variable_scope(name):
        mask = tf.image.resize_bilinear(mask, size, align_corners=True)
        return tf.clip_by_value(mask, 0., 1.)


def dilate_mask(mask, radius, name='dilate_mask'):
    """Grow a mask by ``radius`` pixels (max filter)."""
    if radius <= 0:
        return mask
    with tf.variable_scope(name):
        k = 2 * radius + 1
        return tf.nn.max_pool(mask, [1, k, k, 1], [1, 1, 1, 1], 'SAME')


def erode_mask(mask, radius, name='erode_mask'):
    """Shrink a mask by ``radius`` pixels (min filter)."""
    if radius <= 0:
        return mask
    with tf.variable_scope(name):
        return 1. - dilate_mask(1. - mask, radius)


def feather_mask(mask, radius, sigma=None, name='feather_mask'):
    """Soften mask borders with a gaussian blur.

    Feathered masks let edits fade smoothly into the surrounding image
    instead of leaving visible seams.

    Args:
        mask: Mask of shape [batch, height, width, 1].
        radius (int): Kernel radius in pixels, 0 returns mask unchanged.
        sigma (float): Gaussian sigma, defaults to radius / 2.

    Returns:
        tf.Tensor: Feathered mask in [0, 1].

    """
    if radius <= 0:
        return mask
    with tf.variable_scope(name):
        sigma = sigma or max(radius / 2., 1e-3)
        xs = [math.exp(-(i * i) / (2. * sigma * sigma))
              for i in range(-radius, radius + 1)]
        total = sum(xs)
        kernel_1d = tf.constant([x / total for x in xs], dtype=tf.float32)
        k = 2 * radius + 1
        kernel_h = tf.reshape(kernel_1d, [1, k, 1, 1])
        kernel_v = tf.reshape(kernel_1d, [k, 1, 1, 1])
        # reflect padding so borders of the image do not fade to zero
        x = tf.pad(mask, [[0, 0], [radius, radius], [radius, radius], [0, 0]],
                   mode='SYMMETRIC')
        x = tf.nn.conv2d(x, kernel_h, [1, 1, 1, 1], 'VALID')
        x = tf.nn.conv2d(x, kernel_v, [1, 1, 1, 1], 'VALID')
        return tf.clip_by_value(x, 0., 1.)


def union_masks(masks, name='union_masks'):
    """Union of several (soft) masks, i.e. element-wise maximum."""
    with tf.variable_scope(name):
        return tf.reduce_max(tf.stack(masks, axis=0), axis=0)


def composite(generated, original, mask, name='composite'):
    """Paste ``generated`` into ``original`` inside ``mask``.

    Returns ``mask * generated + (1 - mask) * original``.

    """
    with tf.variable_scope(name):
        return mask * generated + (1. - mask) * original


def region_loss(x, y, mask, loss_type='l1', epsilon=1e-6,
                name='region_loss'):
    """Reconstruction loss averaged over the masked area only.

    Use ``mask`` for a loss on the edited region, or ``1 - mask`` to keep
    the rest of the image close to the original.

    Args:
        x: Tensor, e.g. generated image.
        y: Tensor, e.g. target / original image.
        mask: Mask broadcastable to x.
        loss_type (str): 'l1' or 'l2'.

    Returns:
        tf.Tensor: Scalar loss.

    """
    with tf.variable_scope(name):
        if loss_type == 'l1':
            diff = tf.abs(x - y)
        elif loss_type == 'l2':
            diff = tf.square(x - y)
        else:
            raise ValueError('Unknown loss_type: {}.'.format(loss_type))
        weight = mask * tf.ones_like(diff)
        return tf.reduce_sum(diff * weight) / (
            tf.reduce_sum(weight) + epsilon)
