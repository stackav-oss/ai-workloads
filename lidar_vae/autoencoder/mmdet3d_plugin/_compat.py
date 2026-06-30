"""Compatibility shim: mmcv 1.x / mmdet 2.x API → mmcv 2.x / mmengine / mmdet 3.x.

UniPAD was written for mmcv-full==1.3.8 + mmdet==2.14.0.
Our environment has mmcv==2.1.0 + mmdet==3.3.0 + mmdet3d==1.4.0 + mmengine.

This module re-exports every symbol under its OLD name so that the original
mmdet3d_plugin source files can keep working with minimal sed-level edits
(just change the import source to ``mmdet3d_plugin._compat``).
"""

from __future__ import annotations

import functools
from typing import Any


# --------------------------------------------------------------------------- #
# AttrDict – dict subclass with attribute-style access
# --------------------------------------------------------------------------- #
# mmcv 1.x Config objects allow ``cfg.field`` on plain dicts.  In our setup the
# config dicts are just plain Python dicts, so we wrap them.


class AttrDict(dict):
    """Dict subclass that supports attribute-style access (read & write)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)

    def copy(self):
        return AttrDict(super().copy())


def _to_attrdict(obj):
    """Recursively convert dicts (including nested) to AttrDict."""
    if isinstance(obj, dict) and not isinstance(obj, AttrDict):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    return obj


# --------------------------------------------------------------------------- #
# mmcv.runner  →  mmengine
# --------------------------------------------------------------------------- #
from mmengine.model import BaseModule  # noqa: F401

# force_fp32 / auto_fp16 were AMP helpers in mmcv 1.x.
# In mmcv 2.x they no longer exist; PyTorch-native AMP is used instead.
# We provide transparent no-op wrappers so decorated code keeps running.


def force_fp32(apply_to: Any = None, out_fp32: bool = False, **kwargs):
    """No-op replacement for ``mmcv.runner.force_fp32``."""

    def _decorator(fn):
        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return _wrapper

    if callable(apply_to):  # used without parentheses: @force_fp32
        return apply_to
    return _decorator


def auto_fp16(apply_to: Any = None, out_fp32: bool = False, **kwargs):
    """No-op replacement for ``mmcv.runner.auto_fp16``."""

    def _decorator(fn):
        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return _wrapper

    if callable(apply_to):
        return apply_to
    return _decorator


# --------------------------------------------------------------------------- #
# mmcv.cnn  (symbols that moved to mmengine)
# --------------------------------------------------------------------------- #
from mmengine.model.weight_init import (  # noqa: F401
    xavier_init,
    constant_init,
    bias_init_with_prob,
)

# These still live in mmcv.cnn under v2:
from mmcv.cnn import (  # noqa: F401
    build_conv_layer,
    build_norm_layer,
    build_upsample_layer,
    ConvModule,
    Conv2d,
)

# ``mmcv.cnn.Linear`` was removed in v2 – just use ``torch.nn.Linear``.
from torch.nn import Linear  # noqa: F401


# --------------------------------------------------------------------------- #
# mmcv.cnn.bricks  (transformer / attention registries)
# --------------------------------------------------------------------------- #
# The old bricks registries (ATTENTION, TRANSFORMER_LAYER_SEQUENCE) are gone
# in mmcv 2.x.  We provide tiny dummy registries so @register_module() is a
# no-op.  The actual layers are still importable from mmcv.cnn.bricks.transformer.
class _DummyRegistry:
    """Minimal stand-in for ``mmcv.utils.Registry`` when we only need
    ``@REG.register_module()`` to be a no-op."""

    def __init__(self, name: str = ""):
        self._name = name

    def register_module(self, *args, **kwargs):
        def _wrap(cls_or_fn):
            return cls_or_fn

        if args and callable(args[0]):
            return args[0]
        return _wrap

    def build(self, cfg, *args, **kwargs):
        raise NotImplementedError(
            f"DummyRegistry({self._name}).build() called – "
            "registry-based construction is not supported in compat mode."
        )


ATTENTION = _DummyRegistry("ATTENTION")
TRANSFORMER_LAYER_SEQUENCE = _DummyRegistry("TRANSFORMER_LAYER_SEQUENCE")
TRANSFORMER = _DummyRegistry("TRANSFORMER")

# Registries from mmdet / mmdet3d that we stub out:
BACKBONES = _DummyRegistry("BACKBONES")
HEADS = _DummyRegistry("HEADS")
NECKS = _DummyRegistry("NECKS")
DETECTORS = _DummyRegistry("DETECTORS")
VOXEL_ENCODERS = _DummyRegistry("VOXEL_ENCODERS")
MIDDLE_ENCODERS = _DummyRegistry("MIDDLE_ENCODERS")
DATASETS = _DummyRegistry("DATASETS")
PIPELINES = _DummyRegistry("PIPELINES")
OBJECTSAMPLERS = _DummyRegistry("OBJECTSAMPLERS")
BBOX_ASSIGNERS = _DummyRegistry("BBOX_ASSIGNERS")
BBOX_CODERS = _DummyRegistry("BBOX_CODERS")
IOU_CALCULATORS = _DummyRegistry("IOU_CALCULATORS")
MATCH_COST = _DummyRegistry("MATCH_COST")

# build_from_cfg was in mmcv.utils, now in mmengine
try:
    from mmengine.registry import build_from_cfg  # noqa: F401
except ImportError:

    def build_from_cfg(cfg, registry, default_args=None):
        raise NotImplementedError("build_from_cfg not available")


# --------------------------------------------------------------------------- #
# mmcv.parallel  →  mmengine (DataContainer removed, provide stub)
# --------------------------------------------------------------------------- #
class DataContainer:
    """Minimal stub for ``mmcv.parallel.DataContainer``.

    Only used during dataset pipeline construction which we don't run.
    """

    def __init__(self, data, **kwargs):
        self.data = data


# --------------------------------------------------------------------------- #
# mmdet3d.ops  →  mmdet3d.models.layers  (mmdet3d 1.4)
# --------------------------------------------------------------------------- #
from mmdet3d.models.layers import SparseBasicBlock  # noqa: F401
from mmdet3d.models.layers import make_sparse_convmodule  # noqa: F401

# spconv namespace changed: mmdet3d.ops.spconv → spconv.pytorch
import spconv.pytorch as spconv  # noqa: F401
