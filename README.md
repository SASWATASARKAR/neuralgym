NeuralGym: Deep Learning Toolkit
=================================

NeuralGym is born for fast prototyping of experimental ideas on deep learning.

[![ReadTheDoc](https://readthedocs.org/projects/neuralgym/badge/?version=latest)](http://neuralgym.readthedocs.io)

Due to rapid/messy development and stability concerns, currently only limited modules are released for reproducing other repositories.

## Document

[ReadTheDocs: NeuralGym](http://neuralgym.readthedocs.io)

## Installation

`pip install git+https://github.com/JiahuiYu/neuralgym`

## Region / Object Based Image Manipulation

`neuralgym.ops.region_optimizer.RegionOptimizer` wraps any optimizer so that
only selected regions or objects of an image (or of a latent / feature map)
are generated or edited, while everything else is preserved.
Masks come from `neuralgym.ops.region_ops` (`bbox_to_mask`, `object_mask`,
`feather_mask`, `dilate_mask`, `erode_mask`, `union_masks`, `composite`,
`region_loss`).

```python
import neuralgym as ng
from neuralgym.ops.region_ops import object_mask, bbox_to_mask, feather_mask
from neuralgym.ops.region_optimizer import Region, RegionOptimizer

image = tf.get_variable('image', initializer=original)        # [1, H, W, 3]
car = feather_mask(object_mask(segmentation, [7]), radius=4)  # edit object
sky = bbox_to_mask((0, 0, 64, 256), 256, 256)                 # gentle edit
optimizer = RegionOptimizer(
    tf.train.AdamOptimizer(1e-2),
    regions=[Region(car, scale=1.), Region(sky, scale=0.3)],
    background_scale=0.,              # keep the rest untouched
    references={image: original})     # hard-reset pixels with weight 0
train_op = optimizer.minimize(loss, var_list=[image])
```

Regions are composited in order (later regions override earlier ones), so a
`Region(mask, scale=0.)` can protect part of an object. Image resolution masks
are automatically resized to match 4-D latent / feature map variables, and
variables a mask does not fit (e.g. network weights) are updated as usual.
It also works as the `optimizer` of `ng.train.Trainer`.
