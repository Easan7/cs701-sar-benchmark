"""The course data in TA Zichen's folder layout (see README): every image has one class label and >= 1 box.
Loads images and boxes at a fixed input size and maps predicted boxes back to original pixels."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2
import torch.nn.functional as F

# Mean and std of all pixels of the 9,392 train images, read as gray, resized to
# 512 x 512 and scaled to [0, 1] (computed once).
MEAN, STD = 0.2526, 0.2133  # legacy resized-image values; recompute on native pixels before training


class SARMultiTask(Dataset):
    """Course archive split (train/labels.csv + train/instances.json).
    An item is (image [1, size, size], label 0..8, target), target = {'boxes': XYXY on the resized
    image, 'labels': 1..9, 'image_id': COCO id, 'orig_size': (h, w)}."""

    def __init__(self, root, split='train', train=False, size=512, limit=None, indices=None,
                 rotate90_prob=0.0, gain_prob=0.0, gamma_prob=0.0, speckle_prob=0.0,
                 mean=MEAN, std=STD):
        if size != 512:
            raise ValueError('The assignment pipeline uses a fixed 512px padded canvas; source pixels are not resized')
        self.size = size
        self.mean, self.std = float(mean), float(std)
        self.train = train
        self.rotate90_prob, self.gain_prob = rotate90_prob, gain_prob
        self.gamma_prob, self.speckle_prob = gamma_prob, speckle_prob
        self.root = Path(root)
        self.split_root = self.root / split
        with (self.split_root / 'labels.csv').open(newline='') as f:
            self.rows = list(csv.DictReader(f))
        coco = json.loads((self.split_root / 'instances.json').read_text())
        self.coco = coco
        self.coco_images = {img['file_name']: img for img in coco['images']}
        if len(self.coco_images) != len(coco['images']):
            raise ValueError('instances.json contains duplicate file_name records')
        self.annotations = defaultdict(list)  # COCO image id -> its boxes
        for ann in coco['annotations']:
            self.annotations[ann['image_id']].append(ann)
        if indices is not None:
            self.rows = [self.rows[i] for i in indices]
        if limit is not None:
            self.rows = self.rows[:limit]
        self.transform = v2.Compose([v2.ToImage(), v2.ConvertBoundingBoxFormat('XYXY'),
                                     v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        label = int(row['label'])
        if not 0 <= label < 9:
            raise ValueError(f'Image label must be in 0..8, got {label} for {row["file_name"]}')
        if row['file_name'] not in self.coco_images:
            raise ValueError(f'CSV image {row["file_name"]} has no COCO image record')
        info = self.coco_images[row['file_name']]
        anns = self.annotations[info['id']]
        if not anns or any(int(ann['category_id']) != label + 1 for ann in anns):
            raise ValueError(f'Image {info["id"]} must have one or more boxes of category {label + 1}')
        orig_size = (info['height'], info['width'])
        for ann in anns:
            x, y, box_w, box_h = ann['bbox']
            if (box_w <= 0 or box_h <= 0 or x < 0 or y < 0 or
                    x + box_w > info['width'] or y + box_h > info['height']):
                raise ValueError(f'Invalid original-coordinate box for image {info["id"]}: {ann["bbox"]}')
        boxes = tv_tensors.BoundingBoxes([a['bbox'] for a in anns], format='XYWH', canvas_size=orig_size,
                                         dtype=torch.float32)  # integer boxes would be truncated by Resize
        image_path = self.split_root / row['file_name']
        if not image_path.exists():
            candidates = (self.split_root / 'images' / row['file_name'], self.root / row['file_name'])
            image_path = next((p for p in candidates if p.exists()), image_path)
        image = Image.open(image_path).convert('L')
        image, boxes = self.transform(image, boxes)
        h, w = image.shape[-2:]
        if h > self.size or w > self.size:
            raise ValueError(f'Image {info["file_name"]} is {w}x{h}, exceeding max canvas {self.size}')
        if self.train:
            if random.random() < 0.5:
                image, boxes = image.flip(-1), _flip_boxes(boxes, w, horizontal=True)
            if random.random() < 0.5:
                image, boxes = image.flip(-2), _flip_boxes(boxes, h, horizontal=False)
            if random.random() < self.rotate90_prob:
                k = random.randrange(1, 4)
                image = torch.rot90(image, k, (-2, -1))
                boxes = _rot90_boxes(boxes, h, w, k)
                h, w = image.shape[-2:]
            image = _intensity_augment(image, self.gain_prob, self.gamma_prob, self.speckle_prob)
        image = (image - self.mean) / self.std
        image = F.pad(image, (0, self.size - w, 0, self.size - h))
        target = {'boxes': boxes, 'labels': torch.tensor([int(a['category_id']) for a in anns]),
                  'image_id': info['id'], 'orig_size': orig_size, 'input_size': (h, w)}
        return image, label, target


def collate(batch):
    """Stack images and labels; targets stay a list because images have different numbers of boxes."""
    images, labels, targets = zip(*batch)
    return torch.stack(images), torch.tensor(labels), list(targets)


def to_original_xywh(boxes_xyxy, orig_size, size=512):
    """Map top-left-aligned native-resolution canvas boxes to original COCO pixel coordinates."""
    return box_convert(boxes_xyxy, 'xyxy', 'xywh')


def _flip_boxes(boxes, extent, horizontal):
    result = boxes.clone()
    if horizontal:
        result[:, [0, 2]] = extent - boxes[:, [2, 0]]
    else:
        result[:, [1, 3]] = extent - boxes[:, [3, 1]]
    return result


def _rot90_boxes(boxes, height, width, k):
    """Rotate XYXY boxes counter-clockwise by k exact quarter turns."""
    out = boxes.clone()
    for _ in range(k % 4):
        x1, y1, x2, y2 = out.unbind(-1)
        out = torch.stack((y1, width - x2, y2, width - x1), dim=-1)
        height, width = width, height
    return out


def _intensity_augment(image, gain_prob, gamma_prob, speckle_prob):
    if random.random() < gain_prob:
        image = image * random.uniform(0.9, 1.1)
    if random.random() < gamma_prob:
        image = image.clamp(0, 1).pow(random.uniform(0.85, 1.15))
    if random.random() < speckle_prob:
        noise = torch.randn_like(image).mul_(random.uniform(0.05, 0.1)).add_(1)
        image = image * noise
    return image.clamp(0, 1)


def compute_training_statistics(root, rows, indices=None):
    """Compute grayscale pixel mean/std from the selected training rows only, before padding/augmentation."""
    root = Path(root)
    selected = range(len(rows)) if indices is None else indices
    total = total_sq = 0.0
    count = 0
    for index in selected:
        row = rows[index]
        path = root / 'train' / row['file_name']
        if not path.exists():
            alternatives = (root / 'train' / 'images' / row['file_name'], root / row['file_name'])
            path = next((candidate for candidate in alternatives if candidate.exists()), path)
        with Image.open(path) as source:
            pixels = torch.from_numpy(np.array(source.convert('L'), copy=True)).float().div_(255)
        total += pixels.sum().item()
        total_sq += pixels.square().sum().item()
        count += pixels.numel()
    if count == 0:
        raise ValueError('Cannot compute normalization statistics from an empty training split')
    mean = total / count
    variance = max(total_sq / count - mean * mean, 1e-12)
    return mean, variance ** 0.5
