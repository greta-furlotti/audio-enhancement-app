"""
UNet for spectrogram-based audio denoising.

Architecture:
    - Input:  magnitude spectrogram  [B, 1, F, T]
    - Output: denoising mask          [B, 1, F, T]  (applied to input via multiplication)

The mask-based approach is standard in speech enhancement:
instead of predicting the clean spectrogram directly, the model
predicts a soft mask in [0, 1] that suppresses noise components.

Usage:
    model = UNet(in_channels=1, base_channels=32, depth=4)
    mask  = model(noisy_spec)          # [B, 1, F, T]
    clean_spec = noisy_spec * mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """ConvBlock followed by 2×2 max-pool (halves spatial dims)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)   # saved for skip connection
        down = self.pool(skip)
        return down, skip


class UpBlock(nn.Module):
    """Bilinear upsample + concatenate skip + ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # after cat with skip, channels = in_ch + out_ch (skip has out_ch channels)
        self.conv = ConvBlock(in_ch + out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── UNet ──────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    UNet for spectrogram masking.

    Args:
        in_channels:   number of input channels (1 for mono magnitude spectrogram)
        base_channels: feature channels at the first encoder level (doubles each level)
        depth:         number of encoder/decoder levels (4 → bottleneck at 16× downsampled)
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        enc_channels = []
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(DownBlock(ch, out_ch))
            enc_channels.append(out_ch)
            ch = out_ch

        # Bottleneck
        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = ConvBlock(ch, bottleneck_ch)
        ch = bottleneck_ch

        # Decoder
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = enc_channels[i]
            out_ch = base_channels * (2 ** i)
            self.decoders.append(UpBlock(ch, skip_ch))
            ch = out_ch

        # Output: sigmoid mask in [0, 1]
        self.output_conv = nn.Conv2d(ch, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: noisy magnitude spectrogram  [B, C, F, T]
        Returns:
            mask: soft mask                 [B, C, F, T]  values in (0, 1)
        """
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return torch.sigmoid(self.output_conv(x))


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = UNet(in_channels=1, base_channels=32, depth=4)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UNet parameters: {total_params:,}")

    dummy = torch.randn(2, 1, 128, 128)   # batch=2, mono, 128 freq bins, 128 frames
    mask  = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Mask:   {mask.shape}")
    assert mask.shape == dummy.shape, "Output shape mismatch"
    assert mask.min() >= 0 and mask.max() <= 1, "Mask out of [0,1]"
    print("Sanity check passed.")