"""The two ViT-B/16 backbones, built for 512 px input: images [B, 1, 512, 512] -> patch tokens [B, 1024, 768]."""
import timm
from torch import nn


class ViT(nn.Module):
    """timm ViT-B/16 pretrained on ImageNet-21k."""

    def __init__(self, pretrained: bool):
        super().__init__()
        # in_chans=1: timm sums the RGB patch-embedding weights; img_size=512: it resamples the position embedding
        self.vit = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=pretrained,
                                     img_size=512, in_chans=1, num_classes=0)
        self.embed_dim = 768

    @property
    def blocks(self):
        return self.vit.blocks

    def forward(self, x):
        tokens = self.vit.forward_features(x)  # [B, 1 + N, 768], final norm applied
        return tokens[:, self.vit.num_prefix_tokens:]  # drop the CLS token


class TerraMind(nn.Module):
    """TerraMind-1.0-base (IBM/ESA) through its Sentinel-1 GRD input."""

    def __init__(self, pretrained: bool):
        super().__init__()
        # Import TerraTorch only when this backbone is selected. Its registry imports
        # optional packages (including audio dependencies) that ViT runs do not need.
        from terratorch.registry import BACKBONE_REGISTRY

        self.terramind = BACKBONE_REGISTRY.build('terratorch_terramind_v1_base', pretrained=pretrained,
                                                 modalities=['S1GRD'])
        self.embed_dim = 768

    @property
    def blocks(self):
        return self.terramind.encoder

    def forward(self, x):
        per_block = self.terramind({'S1GRD': x.repeat(1, 2, 1, 1)})  # gray image as both VV and VH
        return per_block[-1]  # last block's tokens [B, N, 768]; terratorch already applied encoder_norm


def build_backbone(name, pretrained):
    """Build a supported backbone. Generic ImageNet initialization is requested only at future run time."""
    if name in {'vit', 'terramind'}:
        return {'vit': ViT, 'terramind': TerraMind}[name](pretrained)
    timm_names = {
        'swin_tiny': ('swin_tiny_patch4_window7_224', (0, 1, 2, 3)),
        'resnet50': ('resnet50', (1, 2, 3, 4)),
        'convnext_tiny': ('convnext_tiny', (0, 1, 2, 3)),
    }
    if name not in timm_names:
        raise ValueError(f'Unknown backbone {name!r}; choose one of {sorted(set(timm_names) | {"vit", "terramind"})}')
    model_name, indices = timm_names[name]
    return TimmFeatureBackbone(model_name, indices, pretrained)


class TimmFeatureBackbone(nn.Module):
    """Hierarchical generic-image backbone exposing native multiscale feature maps."""

    def __init__(self, model_name, out_indices, pretrained):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True,
                                       in_chans=1, out_indices=out_indices)
        self.feature_channels = self.model.feature_info.channels()
        self.embed_dim = self.feature_channels[-1]

    @property
    def blocks(self):
        # Kept for interface compatibility; hierarchical models are tuned by freezing stages instead.
        stages = getattr(self.model, 'layers', None) or getattr(self.model, 'stages', None)
        if stages is not None:
            return stages
        named = [(name, module) for name, module in self.model.named_children()
                 if name.startswith(('layer', 'stage'))]
        return nn.ModuleList([module for _, module in named])

    def forward(self, x):
        return self.model(x)
