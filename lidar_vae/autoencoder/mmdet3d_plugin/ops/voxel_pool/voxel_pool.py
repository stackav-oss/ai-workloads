import torch

try:
    from . import voxel_pool_ext

    _HAS_CUDA_EXT = True
except ImportError:
    _HAS_CUDA_EXT = False


class QuickCumsumCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, coords, ranks, B, X, Y, Z):
        kept = torch.ones(feats.shape[0], device=feats.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = feats.shape[0] - interval_starts[-1]
        coords = coords.int()

        out = voxel_pool_ext.voxel_pool_forward(
            feats, coords, interval_lengths, interval_starts, B, X, Y, Z
        )

        ctx.save_for_backward(interval_starts, interval_lengths, coords)
        ctx.saved_shapes = B, X, Y, Z
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, coords = ctx.saved_tensors
        B, X, Y, Z = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        feats_grad = voxel_pool_ext.voxel_pool_backward(
            out_grad, coords, interval_lengths, interval_starts, B, X, Y, Z
        )

        return feats_grad, None, None, None, None, None, None


def _voxel_pool_pytorch(feats, coords, B, X, Y, Z):
    """Pure PyTorch fallback for voxel pooling via scatter_add."""
    C = feats.shape[1]
    device = feats.device
    output = feats.new_zeros(B, C, Z, Y, X)
    coords = coords.to(device)
    bs_idx = coords[:, 0].long()
    x_idx = coords[:, 1].long().clamp(0, X - 1)
    y_idx = coords[:, 2].long().clamp(0, Y - 1)
    z_idx = coords[:, 3].long().clamp(0, Z - 1)
    output_flat = output.reshape(B * Z * Y * X, C)
    flat_idx_global = (bs_idx * Z * Y * X + z_idx * Y * X + y_idx * X + x_idx).long()
    output_flat.scatter_add_(0, flat_idx_global.unsqueeze(1).expand(-1, C), feats)
    return output_flat.reshape(B, Z, Y, X, C).permute(0, 4, 1, 2, 3)


def voxel_pool(feats, coords, B, X, Y, Z):
    # coords: [bs_idx, x, y, z]
    assert feats.shape[0] == coords.shape[0]

    ranks = (
        coords[:, 0] * X * Y * Z
        + coords[:, 1] * Y * Z
        + coords[:, 2] * Z
        + coords[:, 3]
    )
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    if _HAS_CUDA_EXT:
        x = QuickCumsumCuda.apply(feats, coords, ranks, B, X, Y, Z)
    else:
        x = _voxel_pool_pytorch(feats, coords, B, X, Y, Z)

    return x
