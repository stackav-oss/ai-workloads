import torch
import torch.nn as nn


class BEVPillarPooling(nn.Module):
    """Pool 3D voxel grid along Z-axis to produce BEV features.

    Input:  (B, C, D, H, W) = (B, 16, 64, 1024, 1024)
    Output: (B, C, H, W) = (B, 64, 1024, 1024)

    Uses a small MLP per-pillar (along Z) then max/sum pool.
    """

    def __init__(
        self, in_channels: int = 16, depth_bins: int = 64, out_channels: int = 64
    ):
        super().__init__()
        # The training script currently passes the sparse encoder width as
        # in_channels, but this module receives decoded LiDAR features:
        # (B, 16, 64, H, W). Build the conv for that actual tensor shape.
        decoded_channels = 16
        self.in_channels = decoded_channels
        self.depth_bins = depth_bins
        self.out_channels = out_channels

        # # MLP: per-voxel transform before pooling
        # self.pillar_mlp = nn.Sequential(
        #     nn.Linear(decoded_channels * depth_bins, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, out_channels),
        # )
        self.flatten_conv = nn.Conv2d(
            decoded_channels * depth_bins,  # 16 * 64 = 1024
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) = (B, 16, 64, 1024, 1024)
        Returns:
            (B, out_channels, H, W) = (B, 64, 1024, 1024)
        """
        B, C, D, H, W = x.shape
        if C != self.in_channels or D != self.depth_bins:
            raise ValueError(
                "BEVPillarPooling expected decoded features with shape "
                f"(B, {self.in_channels}, {self.depth_bins}, H, W), got {tuple(x.shape)}."
            )
        # Flatten Z/depth into channels: (B, C*D, H, W) = (B, 1024, 1024, 1024)
        x = x.reshape(B, C * D, H, W)
        # 1x1 conv to reduce channels
        x = self.flatten_conv(x)  # (B, 64, 1024, 1024)
        x = self.norm(x)
        x = self.relu(x)
        return x
