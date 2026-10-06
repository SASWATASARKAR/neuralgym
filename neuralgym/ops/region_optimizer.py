"""Region-aware optimizer for region / object based image generation.

:class:`RegionOptimizer` wraps any tensorflow optimizer and re-weights the
gradients of image-like variables (pixels, latent maps, feature maps) with
spatial masks, so that an optimization only generates or manipulates the
selected regions / objects while the rest of the image is kept intact.

Example::

    image = tf.get_variable('image', initializer=original)  # [1, H, W, 3]
    obj = ng.ops.region_ops.object_mask(segmentation, object_ids=[7])
    obj = ng.ops.region_ops.feather_mask(obj, radius=4)
    sky = ng.ops.region_ops.bbox_to_mask((0, 0, 64, 256), 256, 256)
    optimizer = ng.ops.region_optimizer.RegionOptimizer(
        tf.train.AdamOptimizer(1e-2),
        regions=[Region(obj, scale=1.), Region(sky, scale=0.3)],
        references={image: original})
    train_op = optimizer.minimize(loss, var_list=[image])

"""
import tensorflow as tf
if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
    tf = tf.compat.v1

from .region_ops import resize_mask


class Region(object):
    """A region of an image with its own update strength.

    Args:
        mask: Mask in [0, 1] broadcastable to the variables it applies to,
            typically [batch, height, width, 1]. Tensor or numpy array.
        scale (float): Gradient multiplier inside the region. 1 updates the
            region normally, values in (0, 1) edit it gently, 0 freezes it
            and values > 1 speed it up.
        var_list (list): Variables this region applies to. If None, it
            applies to every variable whose shape the mask fits.
        name (str): Optional name of region.

    """

    def __init__(self, mask, scale=1., var_list=None, name=None):
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        self.scale = scale
        self.var_list = var_list
        self.name = name


class RegionOptimizer(tf.train.Optimizer):
    """Optimizer wrapper that restricts updates to image regions / objects.

    For each variable the regions it applies to are composited in order,
    later regions on top of earlier ones::

        weight = background_scale
        for region in regions:
            weight = (1 - region.mask) * weight + region.mask * region.scale

    and gradients are multiplied by ``weight`` before being passed to the
    wrapped optimizer. Variables no region applies to (e.g. network weights)
    are updated as usual. Because the gradients themselves are masked, slot
    variables of the wrapped optimizer (momentum, Adam moments) never
    accumulate updates outside the regions either.

    Args:
        optimizer: The wrapped tf.train.Optimizer, e.g. AdamOptimizer.
        regions (list): List of :class:`Region`, or of masks (scale 1).
        background_scale (float): Gradient multiplier outside all regions.
            0 (default) keeps everything outside the regions untouched.
        resize_masks (bool): Bilinearly resize 4-D masks to the spatial size
            of 4-D variables, e.g. to edit a low resolution latent or
            feature map with an image resolution mask.
        references (dict): Optional {variable: reference tensor}. After each
            step the variable is reset to the reference wherever its weight
            is 0, which guarantees untouched pixels stay exactly equal to
            the original even with weight decay or other side effects.
        use_locking (bool): Passed to tf.train.Optimizer.
        name (str): Name of optimizer.

    """

    def __init__(self, optimizer, regions, background_scale=0.,
                 resize_masks=True, references=None, use_locking=False,
                 name='RegionOptimizer'):
        super(RegionOptimizer, self).__init__(use_locking, name)
        self._opt = optimizer
        self._regions = [r if isinstance(r, Region) else Region(r)
                         for r in regions]
        self._background_scale = background_scale
        self._resize_masks = resize_masks
        self._references = references or {}

    def _fit_mask(self, mask, var_shape, strict):
        """Fit mask to var_shape, return None if it does not apply."""
        mask_shape = mask.get_shape()
        if (self._resize_masks and mask_shape.ndims == 4 and
                var_shape.ndims == 4):
            size = var_shape.as_list()[1:3]
            if None not in size and mask_shape.as_list()[1:3] != size:
                mask = resize_mask(mask, size)
                mask_shape = mask.get_shape()
        try:
            shape = tf.broadcast_static_shape(mask_shape, var_shape)
            fits = shape.ndims == var_shape.ndims and all(
                v is None or s == v for s, v in zip(
                    shape.as_list(), var_shape.as_list()))
        except ValueError:
            fits = False
        if not fits:
            if strict:
                raise ValueError(
                    'Mask of shape {} can not be applied to variable of '
                    'shape {}.'.format(mask_shape, var_shape))
            return None
        return mask

    def region_weight(self, var):
        """Gradient weight map for var, or None if no region applies.

        Args:
            var: A tf.Variable.

        Returns:
            tf.Tensor or None: Weight broadcastable to var.

        """
        weight = None
        with tf.name_scope(self.get_name() + '/weights'):
            for region in self._regions:
                if region.var_list is not None:
                    names = [v.op.name for v in region.var_list]
                    if var.op.name not in names:
                        continue
                mask = self._fit_mask(
                    region.mask, var.get_shape(),
                    strict=region.var_list is not None)
                if mask is None:
                    continue
                mask = tf.cast(mask, var.dtype.base_dtype)
                if weight is None:
                    weight = self._background_scale * tf.ones_like(mask)
                weight = (1. - mask) * weight + mask * region.scale
        return weight

    def mask_gradients(self, grads_and_vars):
        """Multiply gradients with their region weights.

        Args:
            grads_and_vars: List of (gradient, variable) pairs.

        Returns:
            list: Masked (gradient, variable) pairs.

        """
        masked = []
        for grad, var in grads_and_vars:
            if grad is not None:
                weight = self.region_weight(var)
                if weight is not None:
                    grad = tf.convert_to_tensor(grad) * weight
            masked.append((grad, var))
        return masked

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients with the wrapped optimizer.

        Gradients are returned unmasked (so gradient summaries show the raw
        values), masking happens in :meth:`apply_gradients`.
        """
        return self._opt.compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Mask gradients by regions and apply them with wrapped optimizer.

        Args:
            grads_and_vars: List of (gradient, variable) pairs.
            global_step: Optional variable to increment by one.
            name: Optional name for the returned operation.

        Returns:
            An operation that applies the masked gradients.

        """
        grads_and_vars = list(grads_and_vars)
        apply_op = self._opt.apply_gradients(
            self.mask_gradients(grads_and_vars),
            global_step=global_step, name=name)
        if not self._references:
            return apply_op
        restore_ops = []
        with tf.control_dependencies([apply_op]):
            for _, var in grads_and_vars:
                ref = self._lookup_reference(var)
                if ref is None:
                    continue
                weight = self.region_weight(var)
                if weight is None:
                    continue
                keep = tf.cast(weight > 0, var.dtype.base_dtype)
                value = var.read_value()
                restore_ops.append(tf.assign(
                    var, keep * value + (1. - keep) * tf.cast(
                        ref, var.dtype.base_dtype),
                    use_locking=self._use_locking))
        return tf.group(apply_op, *restore_ops,
                        name=(name or self.get_name()) + '_restore')

    def _lookup_reference(self, var):
        for v, ref in self._references.items():
            if v.op.name == var.op.name:
                return ref
        return None

    def get_slot(self, *args, **kwargs):
        return self._opt.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._opt.get_slot_names(*args, **kwargs)

    def variables(self):
        return self._opt.variables()
