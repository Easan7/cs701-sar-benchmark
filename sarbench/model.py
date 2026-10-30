"""The multi-task model: one shared ViT backbone feeds a classification head and a Faster R-CNN detector."""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from .adapters import set_task


class SimpleFeaturePyramid(nn.Module):
    """ViTDet neck (Li et al., 2022): the single stride-16 ViT map -> 256-channel maps at strides 4, 8, 16, 32, 64."""

    def __init__(self, dim=768, channels=256):
        super().__init__()
        self.rescale = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 2, stride=2), nn.GroupNorm(32, dim // 2), nn.GELU(),
                          nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2)),  # stride 4
            nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),  # stride 8
            nn.Identity(),  # stride 16
            nn.MaxPool2d(2),  # stride 32
        ])
        self.project = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, channels, 1, bias=False), nn.GroupNorm(32, channels),
                          nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(32, channels))
            for c in (dim // 4, dim // 2, dim, dim))

    def forward(self, x):
        levels = [project(rescale(x)) for rescale, project in zip(self.rescale, self.project)]
        levels.append(F.max_pool2d(levels[-1], 1, stride=2))  # stride 64, as torchvision's FPN LastLevelMaxPool
        return OrderedDict(zip(['0', '1', '2', '3', 'pool'], levels))


class MultiTaskModel(nn.Module):
    """backbone tokens -> classification logits (mean token -> LayerNorm -> Linear) and Faster R-CNN detections."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, raw_pixel_branch=False,
                 anchor_sizes=((8, 11), (16, 23), (32, 45), (64, 91), (128, 181)),
                 aspect_ratios=(0.5, 1.0, 2.0), roi_score_threshold=0.05,
                 roi_nms_threshold=0.5, max_detections=100, roi_hidden_dim=1024,
                 roi_pool_size=7, roi_sampling_ratio=2):
        super().__init__()
        self.backbone = backbone
        self.task_routing = task_routing  # MoE-LoRA: the backbone runs once per task, with that task's routers
        self.cls_head = nn.Sequential(nn.LayerNorm(backbone.embed_dim), nn.Linear(backbone.embed_dim, num_classes))
        self.hierarchical = hasattr(backbone, 'feature_channels')
        self.neck = (FeaturePyramidNetwork(backbone.feature_channels, 256, extra_blocks=LastLevelMaxPool())
                     if self.hierarchical else SimpleFeaturePyramid(backbone.embed_dim))
        self.raw_pixel_branch = RawPixelPyramid() if raw_pixel_branch else None
        # Small objects (median ~16 px at 512 px, 99 % below ~240 px): two anchor scales per octave, from 8 px at
        # stride 4 up to 181 px at stride 64, times 3 aspect ratios = 6 anchors per location.
        if len(anchor_sizes) != 5 or len({len(sizes) * len(aspect_ratios) for sizes in anchor_sizes}) != 1:
            raise ValueError('Faster R-CNN expects five anchor levels with the same anchors-per-location count')
        anchors = AnchorGenerator(tuple(tuple(s) for s in anchor_sizes), (tuple(aspect_ratios),) * 5)
        anchors_per_location = len(anchor_sizes[0]) * len(aspect_ratios)
        # Everything else is torchvision's Faster R-CNN default.
        self.rpn = RegionProposalNetwork(
            anchors, RPNHead(256, anchors_per_location), fg_iou_thresh=0.7, bg_iou_thresh=0.3, batch_size_per_image=256,
            positive_fraction=0.5, pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000), nms_thresh=0.7)
        self.roi_heads = RoIHeads(
            MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=roi_pool_size, sampling_ratio=roi_sampling_ratio),
            TwoMLPHead(256 * roi_pool_size * roi_pool_size, roi_hidden_dim),
            FastRCNNPredictor(roi_hidden_dim, num_classes + 1),  # + 1: background
            fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None, score_thresh=roi_score_threshold, nms_thresh=roi_nms_threshold,
            detections_per_img=max_detections)

    def forward(self, images, targets=None, image_sizes=None):
        """images [B, 1, H, W], targets: per-image {'boxes' XYXY, 'labels' 1..9} (training only) ->
        (logits [B, num_classes], per-image detections (eval mode), the four Faster R-CNN losses (training mode))."""
        set_task(self.backbone, 0)  # 0 = classification, 1 = detection; a no-op without MoE-LoRA
        backbone_output = self.backbone(images)
        if self.hierarchical:
            logits = self.cls_head(backbone_output[-1].mean(dim=(-2, -1)))
        else:
            logits = self.cls_head(backbone_output.mean(dim=1))
        if self.task_routing:
            set_task(self.backbone, 1)
            backbone_output = self.backbone(images)
        if self.hierarchical:
            features = self.neck(OrderedDict((str(i), f) for i, f in enumerate(backbone_output)))
        else:
            h, w = images.shape[-2] // 16, images.shape[-1] // 16
            features = self.neck(backbone_output.transpose(1, 2).unflatten(2, (h, w)))
        if self.raw_pixel_branch is not None:
            raw = self.raw_pixel_branch(images)
            features['0'] = features['0'] + torch.sigmoid(self.raw_pixel_branch.gates[0]) * raw['0']
            features['1'] = features['1'] + torch.sigmoid(self.raw_pixel_branch.gates[1]) * raw['1']
        # Precision rule: the box coder casts anchors to the dtype of the regression output, so in bf16 every
        # coordinate near 512 would snap to a 2 px grid. RPN and RoIHeads therefore run in fp32.
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            if image_sizes is None:
                image_sizes = ([tuple(t.get('input_size', images.shape[-2:])) for t in targets]
                               if targets is not None else [tuple(images.shape[-2:])] * len(images))
            image_list = ImageList(images, [tuple(map(int, size)) for size in image_sizes])
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        return logits, detections, {**rpn_losses, **roi_losses}


class RawPixelPyramid(nn.Module):
    """Low-level stride-4/8 features computed directly from the grayscale input pixels."""

    def __init__(self, channels=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.p2 = nn.Sequential(nn.Conv2d(64, channels, 1, bias=False), nn.GroupNorm(32, channels), nn.GELU())
        self.p3_block = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, channels, 1, bias=False), nn.GroupNorm(32, channels), nn.GELU(),
        )
        # sigmoid(-4) gives a small initial residual while keeping gradients available.
        self.gates = nn.Parameter(torch.full((2,), -4.0))

    def forward(self, image):
        x = self.stem(image)
        return {'0': self.p2(x), '1': self.p3_block(x)}
