"""Tests for region_ops and region_optimizer.

The modules are loaded directly from their files so the tests do not need
the rest of neuralgym (which depends on tf.contrib). Run with ``pytest``.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import tensorflow as tf
if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()

OPS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'neuralgym', 'ops')


def _load(name):
    for pkg in ['_ng', '_ng.ops']:
        if pkg not in sys.modules:
            module = types.ModuleType(pkg)
            module.__path__ = []
            sys.modules[pkg] = module
    full = '_ng.ops.' + name
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(OPS_DIR, name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


region_ops = _load('region_ops')
region_optimizer = _load('region_optimizer')
Region = region_optimizer.Region
RegionOptimizer = region_optimizer.RegionOptimizer


def _run(fetches, steps=0, train_op=None):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(steps):
            sess.run(train_op)
        return sess.run(fetches)


def setup_function(_):
    tf.reset_default_graph()


def test_bbox_and_object_masks():
    mask = region_ops.bbox_to_mask((1, 2, 2, 3), 5, 6, batch_size=2)
    seg = np.array([[[0, 1], [2, 1]]], dtype=np.int32)
    obj = region_ops.object_mask(seg, [1, 2])
    m, o = _run([mask, obj])
    assert m.shape == (2, 5, 6, 1)
    assert m.sum() == 2 * 6
    assert m[0, 1:3, 2:5, 0].min() == 1.
    np.testing.assert_array_equal(o[0, :, :, 0], [[0, 1], [1, 1]])


def test_mask_morphology_and_feather():
    mask = region_ops.bbox_to_mask((4, 4, 2, 2), 10, 10)
    dil = region_ops.dilate_mask(mask, 1)
    ero = region_ops.erode_mask(dil, 1)
    soft = region_ops.feather_mask(mask, 2)
    d, e, s = _run([dil, ero, soft])
    assert d.sum() == 16
    assert e.sum() == 4
    assert 0. < s.max() < 1. and s.min() >= 0.
    np.testing.assert_allclose(s.sum(), 4., rtol=1e-4)


def test_region_loss_and_composite():
    x = tf.ones([1, 4, 4, 3])
    y = tf.zeros([1, 4, 4, 3])
    mask = region_ops.bbox_to_mask((0, 0, 2, 4), 4, 4)
    inside = region_ops.region_loss(x, y, mask)
    outside = region_ops.region_loss(x, x, 1. - mask)
    comp = region_ops.composite(x, y, mask)
    i, o, c = _run([inside, outside, comp])
    np.testing.assert_allclose(i, 1., rtol=1e-5)
    np.testing.assert_allclose(o, 0., atol=1e-6)
    assert c[0, :2].min() == 1. and c[0, 2:].max() == 0.


def test_only_region_is_updated():
    original = np.zeros([1, 6, 6, 3], np.float32)
    target = np.ones([1, 6, 6, 3], np.float32)
    image = tf.get_variable('image', initializer=original)
    other = tf.get_variable('other', initializer=np.zeros([3], np.float32))
    loss = (tf.reduce_mean(tf.square(image - target)) +
            tf.reduce_mean(tf.square(other - 1.)))
    mask = region_ops.bbox_to_mask((1, 1, 3, 3), 6, 6)
    opt = RegionOptimizer(tf.train.AdamOptimizer(0.1), [mask])
    train_op = opt.minimize(loss)
    img, oth = _run([image, other], steps=50, train_op=train_op)
    assert img[0, 1:4, 1:4].min() > 0.9
    inside = np.zeros([6, 6], bool)
    inside[1:4, 1:4] = True
    assert np.abs(img[0][~inside]).max() == 0.
    # variables the mask does not fit are optimized normally
    assert oth.min() > 0.9


def test_region_scales_and_override_order():
    image = tf.get_variable('image', initializer=np.zeros(
        [1, 4, 4, 1], np.float32))
    whole = np.ones([1, 4, 4, 1], np.float32)
    hole = region_ops.bbox_to_mask((0, 0, 2, 2), 4, 4)
    opt = RegionOptimizer(
        tf.train.GradientDescentOptimizer(1.),
        [Region(whole, scale=1.), Region(hole, scale=0.)],
        background_scale=0.5)
    weight = opt.region_weight(image)
    w = _run(weight)[0, :, :, 0]
    np.testing.assert_allclose(w[:2, :2], 0.)
    np.testing.assert_allclose(w[2:, :], 1.)


def test_mask_resized_to_latent():
    latent = tf.get_variable('latent', initializer=np.zeros(
        [1, 4, 4, 8], np.float32))
    mask = region_ops.bbox_to_mask((0, 0, 8, 16), 16, 16)
    opt = RegionOptimizer(tf.train.GradientDescentOptimizer(1.), [mask])
    w = _run(opt.region_weight(latent))
    assert w.shape == (1, 4, 4, 1)
    assert w[0, 0].min() == 1. and w[0, 3].max() == 0.


def test_strict_var_list_shape_error():
    image = tf.get_variable('image', initializer=np.zeros(
        [3, 5], np.float32))
    opt = RegionOptimizer(
        tf.train.GradientDescentOptimizer(1.),
        [Region(np.ones([4, 1], np.float32), var_list=[image])])
    try:
        opt.region_weight(image)
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_references_restore_outside():
    original = np.full([1, 4, 4, 1], 0.5, np.float32)
    image = tf.get_variable('image', initializer=original)
    mask = region_ops.bbox_to_mask((0, 0, 2, 4), 4, 4)
    # weight decay through the wrapped optimizer would drift every pixel
    base = tf.train.GradientDescentOptimizer(0.1)
    opt = RegionOptimizer(base, [mask], references={image: original})
    loss = tf.reduce_sum(tf.square(image))
    grads = opt.compute_gradients(loss, [image])
    # add an unmasked drift after masking by perturbing the variable
    with tf.control_dependencies([opt.apply_gradients(grads)]):
        drift = tf.assign_add(image, tf.ones_like(image))
    train_op = opt.apply_gradients([(tf.zeros_like(image), image)])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(drift)
        sess.run(train_op)
        img = sess.run(image)
    np.testing.assert_allclose(img[0, 2:], 0.5)
    assert img[0, :2].min() > 1.
