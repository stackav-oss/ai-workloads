"""LiDAR Encoder: VFE -> MaskSparseEncoder -> SECOND3D Backbone -> FPN.

Uses UniPAD modules (from unipad package) backed by the mmdet3d submodule:
- CustomDynamicSimpleVFE: dynamic voxelization (one point per voxel)
- MaskSparseEncoderHD: sparse 3D encoder with MAE-style masking
- SECOND3D: 3D backbone with strided convolutions
- SECOND3DFPN: 3D feature pyramid network

Architecture matches UniPAD's uvtr_lidar_vs0.075_pretrain.py config.
"""

import torch
import torch.nn as nn
import numpy as np

from mmdet3d_plugin import (
    CustomDynamicSimpleVFE,
    MaskSparseEncoderHD,
    SECOND3D,
    SECOND3DFPN,
)
from mmdet3d_plugin._compat import _to_attrdict
from lidar_decoder import LiDARDecoder


class LiDAREncoder(nn.Module):
    """Complete LiDAR encoding pipeline matching UniPAD.

    VFE (CustomDynamicSimpleVFE) -> MaskSparseEncoderHD -> SECOND3D -> FPN

    Args:
        in_channels: Point feature dimension (default 5: x,y,z,intensity,time).
        output_channels: Final output feature dimension (256 per UniPAD).
        pts_voxel_size: [vx, vy, vz] voxel resolution.
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        sparse_shape: [D, H, W] shape for sparse tensor.
        unified_voxel_shape: [nx, ny, nz] target output spatial shape.
        mae_mask_ratio: MAE masking ratio (0 = no masking).
        mae_downsample_scale: Block size for MAE masking.
        encoder_channels: Sparse encoder channel config.
        encoder_paddings: Sparse encoder padding config.
    """

    def __init__(
        self,
        in_channels: int = 5,
        output_channels: int = 256,
        pts_voxel_size: list[float] = None,
        pc_range: list[float] = None,
        sparse_shape: list[int] = None,
        unified_voxel_shape: list[int] = None,
        mae_mask_ratio: float = 0.8,
        mae_downsample_scale: int = 8,
        encoder_channels: tuple = (
            (16, 16, 32),
            (32, 32, 64),
            (64, 64, 128),
            (128, 128),
        ),
        encoder_paddings: tuple = (
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, (0, 1, 1)),
            (0, 0),
        ),
    ):
        super().__init__()
        if pts_voxel_size is None:
            pts_voxel_size = [0.15625, 0.15625, 0.140625]
        if pc_range is None:
            pc_range = [-80.0, -80.0, -4.5, 80.0, 80.0, 4.5]
        if sparse_shape is None:
            sparse_shape = [64, 1024, 1024]
        if unified_voxel_shape is None:
            unified_voxel_shape = [1024, 1024, 64]

        self.pc_range = pc_range
        self.pts_voxel_size = pts_voxel_size
        self.unified_voxel_shape = unified_voxel_shape

        # VFE: dynamic voxelization (one point per voxel via scatter)
        self.vfe = CustomDynamicSimpleVFE(
            voxel_size=tuple(pts_voxel_size),
            point_cloud_range=tuple(pc_range),
        )

        # MAE config
        mae_cfg = None
        if mae_mask_ratio > 0:
            mae_cfg = _to_attrdict(
                dict(
                    downsample_scale=mae_downsample_scale,
                    mask_ratio=mae_mask_ratio,
                    learnable=False,
                )
            )

        print("sparse shape:", sparse_shape)

        # Sparse middle encoder (matches UniPAD config)
        self.middle_encoder = MaskSparseEncoderHD(
            in_channels=in_channels,
            sparse_shape=sparse_shape,
            output_channels=output_channels,
            encoder_channels=encoder_channels,
            encoder_paddings=encoder_paddings,
            block_type="basicblock",
            keep_depth=True,
            mae_cfg=mae_cfg,
        )

        # 3D Backbone (UniPAD SECOND3D)
        # in_channels=[256,256,256]: non-cascade mode, each stage reads from middle_encoder output
        self.backbone = SECOND3D(
            in_channels=[output_channels, output_channels, output_channels],
            out_channels=[128, 256, 512],
            layer_nums=[5, 5, 5],
            layer_strides=[1, 2, 4],
            is_cascade=False,
            norm_cfg=_to_attrdict(dict(type="BN3d", eps=1e-3, momentum=0.01)),
            conv_cfg=_to_attrdict(dict(type="Conv3d", kernel=(1, 3, 3), bias=False)),
        )

        # FPN (UniPAD SECOND3DFPN)
        self.fpn = SECOND3DFPN(
            in_channels=[128, 256, 512],
            out_channels=[256, 256, 256],
            upsample_strides=[1, 2, 4],
            norm_cfg=_to_attrdict(dict(type="BN3d", eps=1e-3, momentum=0.01)),
            upsample_cfg=_to_attrdict(dict(type="deconv3d", bias=False)),
            use_conv_for_no_stride=True,
        )

        self.decoder = LiDARDecoder()

    def forward(self, points_batch: list[torch.Tensor]) -> torch.Tensor:
        """Encode batch of point clouds.

        Args:
            points_batch: List of (N_i, C) point clouds.

        Returns:
            features: (B, C, D, H, W) 3D voxel features at unified resolution.
        """
        batch_size = len(points_batch)

        # Dynamic voxelization
        voxel_features, coors = self.vfe(points_batch)

        # Sparse encoder (+ MAE masking during training)
        spatial_features, encode_features = self.middle_encoder(
            voxel_features, coors, batch_size, return_encode_features=True
        )

        # 3D Backbone + FPN
        multi_scale = self.backbone(spatial_features)
        features = self.fpn(multi_scale)

        # Resize to unified voxel shape if needed
        target_shape = (
            self.unified_voxel_shape[2],  # D (nz)
            self.unified_voxel_shape[1],  # H (ny)
            self.unified_voxel_shape[0],  # W (nx)
        )
        decoded = self.decoder(features, encode_features, batch_size)
        # decoded: (B, 16, 64, 1024, 1024)

        return decoded
