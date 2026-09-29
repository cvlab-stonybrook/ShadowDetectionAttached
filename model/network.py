"""Final cast/attached-shadow network used in the ECCV paper."""

from functools import partial

import torch
import torch.nn as nn
import torchvision.models as modelzoo

from model.backbone.PVT.pvt_v2 import OverlapPatchEmbed, PyramidVisionTransformerV2


class NetworkMultiClassNoMask(nn.Module):
    """PVT-B5 shadow detector with a ConvNeXt-S light estimator."""

    def __init__(self, input_resolution=512, light_in=8, pretrained_light=True):
        super().__init__()
        if light_in != 8:
            raise ValueError("The released model expects 8 light-estimator channels")

        self.input_resolution = int(input_resolution)
        self.light_in = int(light_in)
        embed_dims = [64, 128, 320, 512]

        self.encoder = PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=embed_dims,
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        self.layer3_transposed_conv = nn.ConvTranspose2d(512, 320, 2, 2)
        self.cat_32 = self._decoder_block(640, 320)
        self.layer2_transposed_conv = nn.ConvTranspose2d(320, 128, 2, 2)
        self.cat_21 = self._decoder_block(256, 128)
        self.layer1_transposed_conv = nn.ConvTranspose2d(128, 64, 2, 2)
        self.cat_10 = self._decoder_block(128, 64)
        self.output_seg = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.Conv2d(16, 3, 1),
        )

        light_weights = (
            modelzoo.ConvNeXt_Small_Weights.DEFAULT if pretrained_light else None
        )
        self.lightest = modelzoo.convnext_small(weights=light_weights)
        self.lightest.features[0][0] = nn.Conv2d(8, 96, kernel_size=4, stride=4)
        self.lightest.classifier[2] = nn.Linear(768, 3)

    @staticmethod
    def _decoder_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def set_detector_input_channels(self, in_channels=7):
        """Replace the first PVT projection before loading a 7-channel checkpoint."""
        self.encoder.patch_embed1 = OverlapPatchEmbed(
            img_size=224,
            patch_size=7,
            stride=4,
            in_chans=in_channels,
            embed_dim=64,
        )

    def forward(self, image, attached_prior, normal):
        detector_input = torch.cat((image, attached_prior, normal), dim=1)
        layer0, layer1, layer2, layer3 = self.encoder.forward_features(detector_input)

        layer3_up = self.layer3_transposed_conv(layer3)
        layer2_up = self.layer2_transposed_conv(
            self.cat_32(torch.cat((layer2, layer3_up), dim=1))
        )
        layer1_up = self.layer1_transposed_conv(
            self.cat_21(torch.cat((layer1, layer2_up), dim=1))
        )
        layer0_up = self.cat_10(torch.cat((layer0, layer1_up), dim=1))
        segmentation_logits = self.output_seg(layer0_up)

        probabilities = torch.softmax(segmentation_logits, dim=1)
        cast_probability = probabilities[:, 1:2].detach()
        attached_probability = probabilities[:, 2:3].detach()
        light_input = torch.cat(
            (cast_probability, attached_probability, normal, image), dim=1
        )
        light = self.lightest(light_input)
        return segmentation_logits, light

