"""Compatibility wrapper for the common `chamfer_3DDist` import path."""

from __future__ import annotations

import torch
from torch import nn

from . import ChamferDis


class chamfer_3DDist(nn.Module):
    """Return per-point squared Chamfer distances for 2D or batched 3D inputs."""

    def forward(
        self, xyz1: torch.Tensor, xyz2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if xyz1.ndim == 2 and xyz2.ndim == 2:
            return ChamferDis.apply(xyz1.contiguous(), xyz2.contiguous())

        if xyz1.ndim != 3 or xyz2.ndim != 3:
            raise ValueError(
                "chamfer_3DDist expects tensors shaped (N, 3) or (B, N, 3)."
            )
        if xyz1.shape[0] != xyz2.shape[0]:
            raise ValueError("Batched Chamfer inputs must have the same batch size.")

        outputs = [
            ChamferDis.apply(xyz1[i].contiguous(), xyz2[i].contiguous())
            for i in range(xyz1.shape[0])
        ]
        dist1, dist2, idx1, idx2 = zip(*outputs, strict=True)
        return (
            torch.stack(dist1, dim=0),
            torch.stack(dist2, dim=0),
            torch.stack(idx1, dim=0),
            torch.stack(idx2, dim=0),
        )
