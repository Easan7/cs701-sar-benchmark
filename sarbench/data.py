"""The course data in TA Zichen's folder layout (see README): every image has one class label and >= 1 box.
Loads images and boxes at a fixed input size and maps predicted boxes back to original pixels."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2

# Mean and std of all pixels of the 9,392 train images, read as gray, resized to
# 512 x 512 and scaled to [0, 1] (computed once).
MEAN, STD = 0.2526, 0.2133


class SARMultiTask(Dataset):
    """Course archive split (train/labels.csv + train/instances.json).
    An item is (image [1, size, size], label 0..8, target), target = {'boxes': XYXY on the resized
    image, 'labels': 1..9, 'image_id': COCO id, 'orig_size': (h, w)}."""

    def __init__(self, root, split='train', train=False, size=512, limit=None, indices=None):
        self.root = Path(root)
        self.split_root = self.root / split
        with (self.split_root / 'labels.csv').open(newline='') as f:
            self.rows = list(csv.DictReader(f))
        coco = json.loads((self.split_root / 'instances.json').read_text())
        self.coco = coco
        self.coco_images = {img['file_name']: img for img in coco['images']}
        self.annotations = defaultdict(list)  # COCO image id -> its boxes
        for ann in coco['annotations']:
            self.annotations[ann['image_id']].append(ann)
        if indices is not None:
            self.rows = [self.rows[i] for i in indices]
        if limit is not None:
            self.rows = self.rows[:limit]
        flips = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()] if train else []
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ConvertBoundingBoxFormat('XYXY'),  # COCO [x, y, w, h] -> [x1, y1, x2, y2]
            v2.Resize((size, size)),  # stretches non-square images; boxes are scaled with the pixels
            *flips,  # overhead imagery has no canonical "up"
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([MEAN], [STD]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        info = self.coco_images[row['image']]
        anns = self.annotations[info['id']]
        orig_size = (info['height'], info['width'])
        boxes = tv_tensors.BoundingBoxes([a['bbox'] for a in anns], format='XYWH', canvas_size=orig_size,
                                         dtype=torch.float32)  # integer boxes would be truncated by Resize
        image_path = self.split_root / row['file_name']
        if not image_path.exists():
            candidates = (self.split_root / 'images' / row['file_name'], self.root / row['file_name'])
            image_path = next((p for p in candidates if p.exists()), image_path)
        image = Image.open(image_path).convert('L')
        image, boxes = self.transform(image, boxes)
        target = {'boxes': boxes, 'labels': torch.tensor([a['category_id'] for a in anns]),
                  'image_id': info['id'], 'orig_size': orig_size}
        return image, int(row['label']), target


def collate(batch):
    """Stack images and labels; targets stay a list because images have different numbers of boxes."""
    images, labels, targets = zip(*batch)
    return torch.stack(images), torch.tensor(labels), list(targets)


def to_original_xywh(boxes_xyxy, orig_size, size=512):
    """Undo the resize: XYXY boxes on the size x size input -> COCO [x, y, w, h] in original pixels."""
    h, w = orig_size
    return box_convert(boxes_xyxy * boxes_xyxy.new_tensor([w, h, w, h]) / size, 'xyxy', 'xywh')
