import torch
import torch.nn as nn


class BEV2DFPN(nn.Module):
    """2D encoder-decoder on BEV with dual output heads.

    Downsample: (B, in_ch, 1024, 1024) → (B, bottleneck_ch, 128, 128)  [8× spatial down]
    Upsample:   (B, bottleneck_ch, 128, 128) → (B, shared_ch, 1024, 1024) [8× spatial up]

    Two output heads (both lightweight 1×1 convolutions):
      occ_head: (B, shared_ch, 1024, 1024) → (B, nz, 1024, 1024)        occupancy logits
      nfg_head: (B, shared_ch, 1024, 1024) → (B, C*nz, 1024, 1024)      → reshape → (B, C, nz, 1024, 1024)
    """

    def __init__(
        self,
        in_channels: int = 64,
        bottleneck_channels: int = 32,
        shared_channels: int = 64,
        channels_per_voxel: int = 16,  # per-voxel feature dim for NFG
        height_dim: int = 64,  # height bins
        occupancy_bias: float = -5.0,  # occupancy head bias initialization
        encoding_channels=(64, 128, 256),
        decoding_channels=(32, 32),
        train_vae: bool = True,
        debug_numerics: bool = False,
    ):
        super().__init__()
        if len(encoding_channels) != 3:
            raise ValueError("encoding_channels must contain exactly 3 channel sizes.")
        if len(decoding_channels) != 2:
            raise ValueError("decoding_channels must contain exactly 2 channel sizes.")

        self.occupancy_bias = occupancy_bias
        self.height_dim = height_dim
        self.channels_per_voxel = channels_per_voxel
        self.shared_channels = shared_channels
        self.bottleneck_channels = bottleneck_channels
        self.train_vae = train_vae
        self.debug_numerics = debug_numerics
        self.use_stochastic_sampling = train_vae

        # 8x Downsampling: 1024,64 -> 512,C0 -> 256,C1 -> 128,C2 -> 128,bottleneck
        self.down = nn.Sequential(
            # 1024 -> 512 (2x down)
            nn.Conv2d(
                in_channels, encoding_channels[0], 3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(encoding_channels[0]),
            nn.ReLU(inplace=True),
            # 512 -> 256 (2x down)
            nn.Conv2d(
                encoding_channels[0],
                encoding_channels[1],
                3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(encoding_channels[1]),
            nn.ReLU(inplace=True),
            # 256 -> 128 (2x down)
            nn.Conv2d(
                encoding_channels[1],
                encoding_channels[2],
                3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(encoding_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                encoding_channels[2],
                bottleneck_channels,
                3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        if train_vae:
            self.mu_head = nn.Conv2d(
                bottleneck_channels, bottleneck_channels, kernel_size=1, bias=True
            )
            self.log_var_head = nn.Conv2d(
                bottleneck_channels, bottleneck_channels, kernel_size=1, bias=True
            )
            self._init_vae_heads()
        else:
            self.mu_head = None
            self.log_var_head = None

        # 8x Upsampling: 128,32 -> 256,32 -> 512,32 -> 1024,64
        up_channels = (bottleneck_channels, *decoding_channels, shared_channels)
        up_layers: list[nn.Module] = []
        for in_channel, out_channel in zip(up_channels, up_channels[1:]):
            up_layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channel,
                        out_channel,
                        4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True),
                ]
            )
        self.up = nn.Sequential(*up_layers)
        # Occupancy Head
        # Input: (B, shared_ch, 1024, 1024)
        # Output: (B, nz, 1024, 1024) = (B, 64, 1024, 1024) raw logits
        self.occ_head = nn.Conv2d(
            shared_channels, self.height_dim, kernel_size=1, bias=True
        )
        # Initialize: predict "all empty" at start
        nn.init.zeros_(self.occ_head.weight)
        nn.init.constant_(self.occ_head.bias, self.occupancy_bias)

        # Neural Feature Grid Head
        # Input: (B, shared_ch, 1024, 1024)
        # Output: (B, C * nz, 1024, 1024) → reshape → (B, C, nz, 1024, 1024)
        self.nfg_head = nn.Conv2d(
            shared_channels,
            self.channels_per_voxel * self.height_dim,
            kernel_size=1,
            bias=True,
        )

    def _init_vae_heads(self) -> None:
        if self.mu_head is None or self.log_var_head is None:
            return
        with torch.no_grad():
            self.mu_head.weight.zero_()
            eye = torch.eye(
                self.bottleneck_channels,
                device=self.mu_head.weight.device,
                dtype=self.mu_head.weight.dtype,
            )
            self.mu_head.weight[:, :, 0, 0].copy_(eye)
            self.mu_head.bias.zero_()
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.constant_(self.log_var_head.bias, -4.0)

    def encode(self, bev: torch.Tensor) -> torch.Tensor:
        """Encode BEV to latent space.

        Args:
            bev: (B, in_channels, 1024, 1024)

        Returns:
            z_raw: (B, bottleneck_channels, 128, 128)
        """
        return self.down(bev)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode sampled latent z back to BEV features.

        Args:
            z: (B, bottleneck_channels, 128, 128) — sampled latent
        Returns:
            occ_logits: (B, 64, 1024, 1024)
            nfg: (B, 16, 64, 1024, 1024)
        """
        B = z.shape[0]
        x_shared = self.up(z)  # (B, shared_channels, 1024, 1024)
        occ_logits = self.occ_head(x_shared)  # (B, 64, 1024, 1024)
        nfg_flat = self.nfg_head(x_shared)
        nfg = nfg_flat.view(
            B,
            self.channels_per_voxel,
            self.height_dim,
            nfg_flat.shape[2],
            nfg_flat.shape[3],
        )
        return occ_logits, nfg

    # def forward(self, bev: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args:
    #         bev: (B, 64, 1024, 1024)
    #     Returns:
    #         (B, 16, 64, 1024, 1024) 3D voxel features at full BEV resolution
    #     """
    #     B = bev.shape[0]

    #     # Shared backbone
    #     x_down = self.down(bev)  # (B, bottleneck_channels, 128, 128)
    #     x_shared = self.up(x_down)  # (B, shared_channels, 1024, 1024)

    #     # Branch 1: Occupancy
    #     occ_logits = self.occ_head(x_shared)  # (B, 64, 1024, 1024)

    #     # Branch 2: Neural Feature Grid
    #     nfg_flat = self.nfg_head(
    #         x_shared
    #     )  # (B, channels_per_voxel*height_dim, 1024, 1024)
    #     nfg = nfg_flat.view(
    #         B,
    #         self.channels_per_voxel,
    #         self.height_dim,
    #         nfg_flat.shape[2],
    #         nfg_flat.shape[3],
    #     )  # (B, channels_per_voxel, height_dim, 1024, 1024)

    #     return x_down, occ_logits, nfg

    def set_stochastic_sampling(self, enabled: bool) -> None:
        self.use_stochastic_sampling = enabled and self.train_vae

    def _assert_finite(self, name: str, tensor: torch.Tensor | None) -> None:
        if not self.debug_numerics or tensor is None:
            return
        finite = torch.isfinite(tensor)
        if finite.all():
            return
        finite_count = int(finite.sum().item())
        total_count = tensor.numel()
        finite_values = tensor[finite]
        min_value = finite_values.min().item() if finite_count else "none"
        max_value = finite_values.max().item() if finite_count else "none"
        raise FloatingPointError(
            f"{name} has non-finite values: "
            f"shape={tuple(tensor.shape)}, finite={finite_count}/{total_count}, "
            f"min={min_value}, max={max_value}"
        )

    def sample_z(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.train_vae:
            raise RuntimeError("sample_z() is only valid when train_vae=True.")
        if h.shape[1] != self.bottleneck_channels:
            raise ValueError(
                "VAE adapter expects the encoder to output "
                f"{self.bottleneck_channels} channels, got {h.shape[1]}."
            )
        if self.mu_head is None or self.log_var_head is None:
            raise RuntimeError("VAE heads are not initialized.")

        mu = self.mu_head(h)
        log_var = self.log_var_head(h).clamp(min=-10.0, max=10.0)

        if not self.use_stochastic_sampling:
            return mu, mu, log_var

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z_sampled = mu + std * eps
        return z_sampled, mu, log_var

    def forward(
        self, bev: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Full forward: encode -> optional VAE sample -> decode.

        Returns:
            z_raw: (B, bottleneck_channels, 128, 128) - the shared AE bottleneck
            z_latent: (B, bottleneck_channels, 128, 128) - sampled z for VAE, z_raw for AE
            occ_logits: (B, 64, 1024, 1024)
            nfg: (B, 16, 64, 1024, 1024)
            mu: (B, bottleneck_channels, 128, 128) or None when train_vae=False
            log_var: (B, bottleneck_channels, 128, 128) or None when train_vae=False
        """
        self._assert_finite("bev_2d_fpn.input", bev)
        z_raw = self.encode(bev)
        self._assert_finite("bev_2d_fpn.z_raw", z_raw)

        if self.train_vae:
            z_latent, mu, log_var = self.sample_z(z_raw)
        else:
            z_latent = z_raw
            mu = None
            log_var = None
        self._assert_finite("bev_2d_fpn.mu", mu)
        self._assert_finite("bev_2d_fpn.log_var", log_var)
        self._assert_finite("bev_2d_fpn.z_latent", z_latent)

        occ_logits, nfg = self.decode(z_latent)
        self._assert_finite("bev_2d_fpn.occ_logits", occ_logits)
        self._assert_finite("bev_2d_fpn.nfg", nfg)
        return z_raw, z_latent, occ_logits, nfg, mu, log_var
