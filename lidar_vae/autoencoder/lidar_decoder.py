import torch
import torch.nn as nn
from mmdet3d_plugin._compat import (
    SparseBasicBlock,
    make_sparse_convmodule,
    spconv,  # = spconv.pytorch
)
from mmdet3d.models.layers.sparse_block import replace_feature


class LiDARDecoder(nn.Module):
    """Sparse 3D decoder with lateral connections from encoder.

    Uses mmdet3d's make_sparse_convmodule with conv_type='SparseInverseConv3d'
    to upsample, matching the pattern from mmdet3d's SparseUNet.

    (256, 8, 128, 128) → (16, 64, 1024, 1024)
    """

    def __init__(
        self,
        in_channels: int = 256,  # bottleneck channels (from FPN output)
        out_channels: int = 16,  # final output channels
        # Encoder layer output channels at each stage (for lateral connections):
        # [layer4=128, layer3=128, layer2=64, layer1=32]
        encoder_channels: list[int] = [128, 128, 64, 32],
        # Decoder output channels at each stage:
        decoder_channels: tuple[tuple[int]] = (
            (128, 128),  # stage 4→3: lateral_ch, merge_ch (operates at 8,128,128)
            (64, 64),  # stage 3→2: upsample to (17,256,256)
            (32, 32),  # stage 2→1: upsample to (33,512,512)
            (16, 16),  # stage 1→input: upsample to (65,1024,1024)
        ),
        decoder_paddings: tuple[tuple[int]] = ((1, 0), (1, 0), (1, 0), (0, 1)),
        sparse_shape: list[int] = [65, 1024, 1024],
        norm_cfg: dict = dict(type="BN1d", eps=1e-3, momentum=0.01),
    ):
        super().__init__()
        self.sparse_shape = sparse_shape
        self.decoder_channels = decoder_channels
        self.stage_num = len(decoder_channels)
        self.decoder_channels = decoder_channels
        self.decoder_paddings = decoder_paddings

        # Build decoder layers using make_sparse_convmodule (same as SparseUNet)
        self._make_decoder_layers(
            make_sparse_convmodule, norm_cfg, in_channels, encoder_channels
        )

        # Final 1x1 to output channels
        self.conv_out = make_sparse_convmodule(
            decoder_channels[-1][1],
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            padding=0,
            indice_key="subm_dec_out",
            conv_type="SubMConv3d",
        )

    def _make_decoder_layers(self, make_block, norm_cfg, in_channels, encoder_channels):
        """Build decoder layers following mmdet3d SparseUNet pattern.

        Creates lateral_layer, merge_layer, upsample_layer for each stage.
        Uses make_sparse_convmodule with conv_type='SparseInverseConv3d'.
        """
        block_num = len(self.decoder_channels)

        for i, block_channels in enumerate(self.decoder_channels):
            paddings = self.decoder_paddings[i]
            stage_idx = block_num - i  # 4, 3, 2, 1

            # lateral_layer: processes encoder skip connection (SparseBasicBlock)
            setattr(
                self,
                f"lateral_layer{stage_idx}",
                SparseBasicBlock(
                    encoder_channels[i],
                    encoder_channels[i],
                    conv_cfg=dict(
                        type="SubMConv3d",
                        indice_key=f"subm_dec_lat{stage_idx}",
                    ),
                    norm_cfg=norm_cfg,
                ),
            )

            # merge_layer: merges upsampled + lateral features (cat → SubMConv3d)
            setattr(
                self,
                f"merge_layer{stage_idx}",
                make_block(
                    in_channels + encoder_channels[i],  # cat(bottom_up, lateral)
                    block_channels[1],
                    3,
                    norm_cfg=norm_cfg,
                    padding=paddings[0],
                    indice_key=f"subm_dec{stage_idx}",
                    conv_type="SubMConv3d",
                ),
            )

            # Encoder layer 4 does not downsample, so stage 4 only merges at
            # the bottleneck resolution. Stages 3, 2, and 1 use the matching
            # encoder SparseConv3d indice keys for inverse upsampling.
            if stage_idx == 4:
                setattr(self, f"upsample_layer{stage_idx}", nn.Identity())
            else:
                setattr(
                    self,
                    f"upsample_layer{stage_idx}",
                    make_block(
                        block_channels[1],
                        block_channels[1],
                        3,
                        norm_cfg=norm_cfg,
                        indice_key=f"spconv{stage_idx}",
                        conv_type="SparseInverseConv3d",
                    ),
                )

            in_channels = block_channels[1]

    def decoder_layer_forward(
        self,
        x_lateral: "spconv.SparseConvTensor",
        x_bottom: "spconv.SparseConvTensor",
        lateral_layer,
        merge_layer,
        upsample_layer,
    ) -> "spconv.SparseConvTensor":
        """Forward of one decoder stage. Follows mmdet3d SparseUNet pattern.

        1. Process lateral (encoder skip) with lateral_layer
        2. Cat bottom-up features with lateral features
        3. Merge via merge_layer
        4. Upsample via SparseInverseConv3d (upsample_layer)
        """
        x_lat = lateral_layer(x_lateral)
        if not torch.equal(x_bottom.indices, x_lat.indices):
            raise RuntimeError("Decoder bottom and lateral sparse indices must match.")
        # Concatenate features from bottom-up path and lateral path
        x = replace_feature(
            x_lat, torch.cat((x_bottom.features, x_lat.features), dim=1)
        )
        x = merge_layer(x)
        x = upsample_layer(x)
        return x

    def forward(
        self,
        bottleneck_dense: torch.Tensor,  # (B, 256, 8, 128, 128) from FPN
        encode_features: list,  # SparseConvTensors from encoder stages
        batch_size: int,
    ) -> torch.Tensor:
        """
        Args:
            bottleneck_dense: Dense bottleneck features (B, C, D, H, W)
            encode_features: [enc_layer1, enc_layer2, enc_layer3, enc_layer4]
                             SparseConvTensors from MaskSparseEncoderHD
            batch_size: batch size

        Returns:
            Dense tensor (B, 16, 64, 1024, 1024)
        """
        # Convert dense bottleneck → SparseConvTensor using the same active
        # coordinates as encoder layer 4. This keeps the feature rows aligned
        # with the lateral skip tensor and avoids densifying all 8*128*128 cells.
        x = self._dense_to_sparse_like(
            bottleneck_dense, encode_features[-1], batch_size
        )

        # Decode: iterate from deepest to shallowest
        # encode_features = [layer1, layer2, layer3, layer4]
        # Decoder processes: layer4 → layer3 → layer2 → layer1
        for i in range(self.stage_num, 0, -1):
            x = self.decoder_layer_forward(
                x_lateral=encode_features[i - 1],
                x_bottom=x,
                lateral_layer=getattr(self, f"lateral_layer{i}"),
                merge_layer=getattr(self, f"merge_layer{i}"),
                upsample_layer=getattr(self, f"upsample_layer{i}"),
            )

        # Final conv
        x = self.conv_out(x)

        # Densify and crop: (B, 16, 65, 1024, 1024) → (B, 16, 64, 1024, 1024)
        dense_out = x.dense()
        dense_out = dense_out[:, :, :64, :, :]

        return dense_out

    def _dense_to_sparse_like(
        self,
        dense: torch.Tensor,
        reference: "spconv.SparseConvTensor",
        batch_size: int,
    ) -> "spconv.SparseConvTensor":
        """Sample dense features at the active coordinates of ``reference``."""
        indices = reference.indices.long()
        features = dense[
            indices[:, 0],
            :,
            indices[:, 1],
            indices[:, 2],
            indices[:, 3],
        ]
        return spconv.SparseConvTensor(
            features,
            reference.indices.int(),
            reference.spatial_shape,
            batch_size,
        )
