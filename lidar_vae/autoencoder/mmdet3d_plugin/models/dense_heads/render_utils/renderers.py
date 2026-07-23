import torch
from torch import nn


class DepthRenderer(nn.Module):
    """Calculate depth along ray."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, ray_samples, weights):
        """Composite samples along ray and calculate depths.

        Args:
            weights: Weights for each sample.
            ray_samples: Set of ray samples.
        Returns:
            Outputs of depth values.
        """
        eps = 1e-10
        steps = ray_samples.frustums.starts
        depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)
        depth = torch.clip(depth, steps.min(), steps.max())
        return depth


class NormalRenderer(nn.Module):
    """Calculate normals along the ray."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, normals, weights):
        """Calculate normals along the ray."""
        n = torch.sum(weights * normals, dim=-2)
        return n
