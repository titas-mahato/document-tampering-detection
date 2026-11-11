"""
ResNet34 U-Net Architecture for Pixel-Level Document Tampering Localization
Implements an Encoder-Decoder network with ResNet34 backbone and skip connections.
Corresponds to Course Module 3 (Spatial Vision & Segmentation) and Module 5 (U-Net).
"""

from typing import Optional
import os

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class DoubleConv(nn.Module):
        """Standard double 3x3 convolution block with BatchNorm and ReLU."""
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)


    class DecoderBlock(nn.Module):
        """Upsamples feature maps via bilinear interpolation and concatenates encoder skip connections."""
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
            super().__init__()
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)

        def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            return self.conv(torch.cat([x, skip], dim=1))


    class ResNet34UNet(nn.Module):
        """
        ResNet34 U-Net for Document Tampering Localization.
        - Encoder: Pretrained ResNet34 extracting multi-scale feature hierarchies.
        - Decoder: 4-stage upsampling with skip connections preserving fine character details.
        - Output: 1-channel logit map representing pixel-level tamper probability.
        """
        def __init__(self, pretrained: bool = False):
            super().__init__()
            try:
                weights = models.ResNet34_Weights.DEFAULT if pretrained else None
                backbone = models.resnet34(weights=weights)
            except Exception:
                backbone = models.resnet34(weights=None)

            self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
            self.pool = backbone.maxpool
            self.encoder1 = backbone.layer1
            self.encoder2 = backbone.layer2
            self.encoder3 = backbone.layer3
            self.encoder4 = backbone.layer4

            self.decoder4 = DecoderBlock(512, 256, 256)
            self.decoder3 = DecoderBlock(256, 128, 128)
            self.decoder2 = DecoderBlock(128, 64, 64)
            self.decoder1 = DecoderBlock(64, 64, 64)

            self.final_up = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.output = nn.Conv2d(32, 1, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_size = x.shape[-2:]

            e0 = self.encoder0(x)                 # 1/2
            e1 = self.encoder1(self.pool(e0))     # 1/4
            e2 = self.encoder2(e1)                # 1/8
            e3 = self.encoder3(e2)                # 1/16
            e4 = self.encoder4(e3)                # 1/32

            d4 = self.decoder4(e4, e3)
            d3 = self.decoder3(d4, e2)
            d2 = self.decoder2(d3, e1)
            d1 = self.decoder1(d2, e0)

            x = nn.functional.interpolate(
                d1, size=orig_size, mode="bilinear", align_corners=False
            )
            return self.output(self.final_up(x))


    def build_unet(weights_path: Optional[str] = None, device: str = "cpu") -> ResNet34UNet:
        """Instantiates ResNet34UNet and optionally loads checkpoint weights."""
        model = ResNet34UNet(pretrained=False)
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else (checkpoint["model"] if "model" in checkpoint else checkpoint)
            model.load_state_dict(state_dict)
        model.to(device)
        return model

else:
    class DoubleConv:
        pass

    class DecoderBlock:
        pass

    class ResNet34UNet:
        pass

    def build_unet(*args, **kwargs):
        raise ImportError("PyTorch is required to build ResNet34UNet. Install with `pip install torch torchvision`.")
