"""Model definitions shared by training (Notebook 2) and the prototype.

Two variants were trained and compared:
  - 'unet_scratch' : plain U-Net, all weights trained from random initialisation
  - 'unet_resnet34': U-Net with an ImageNet-pretrained ResNet-34 encoder (segmentation_models_pytorch)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Plain U-Net. Encoder 32-64-128-256 filters, bottleneck 512, symmetric decoder with skip connections."""
    def __init__(self, in_ch=3, n_classes=7, base=32):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        self.downs = nn.ModuleList()
        c = in_ch
        for c_out in ch:
            self.downs.append(DoubleConv(c, c_out))
            c = c_out
        self.bottleneck = DoubleConv(ch[-1], ch[-1] * 2)
        self.ups, self.up_convs = nn.ModuleList(), nn.ModuleList()
        c = ch[-1] * 2
        for c_out in reversed(ch):
            self.ups.append(nn.ConvTranspose2d(c, c_out, 2, stride=2))
            self.up_convs.append(DoubleConv(c_out * 2, c_out))
            c = c_out
        self.head = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = conv(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def build_model(variant, n_classes=7, pretrained_encoder=True):
    if variant == 'unet_scratch':
        return UNet(3, n_classes)
    if variant == 'unet_resnet34':
        import segmentation_models_pytorch as smp
        # When loading our own trained weights, the ImageNet download is not needed.
        return smp.Unet('resnet34', encoder_weights='imagenet' if pretrained_encoder else None,
                        in_channels=3, classes=n_classes)
    raise ValueError(f'Unknown model variant: {variant}')
