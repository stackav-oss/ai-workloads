try:
    from .smooth_sampler import SmoothSampler
except ImportError:
    SmoothSampler = None  # CUDA extension not compiled; fall back to grid_sample_3d

__all__ = ["SmoothSampler"]
