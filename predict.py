"""Run one trained model once per unlabeled image and write a Codabench ZIP.

Example: python predict.py --checkpoint runs/vit_pretrained_hybrid/best.pt --split val
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from sarbench.adapters import add_adapters
from sarbench.backbones import build_backbone
from sarbench.data import MEAN, STD, to_original_xywh
from sarbench.model import MultiTaskModel
from make_submission import write_submission


class UnlabeledSAR(Dataset):
    def __init__(self, root, split):
        self.folder = Path(root) / split
        self.images = json.loads((self.folder / 'images.json').read_text())
        if isinstance(self.images, dict):
            self.images = self.images['images']
        self.transform = v2.Compose([
            v2.ToImage(), v2.Resize((512, 512)), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([MEAN], [STD]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        path = self.folder / info['file_name']
        if not path.exists():
            path = self.folder / 'images' / info['file_name']
        with Image.open(path) as image:
            tensor = self.transform(image.convert('L'))
        return tensor, index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent / 'cs701-sar-course-data')
    parser.add_argument('--split', required=True, choices=['val', 'test'])
    parser.add_argument('--out', type=Path, default=Path('submission.zip'))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    backbone = build_backbone(checkpoint['backbone'], pretrained=False)
    if checkpoint['adapt'] != 'full':
        add_adapters(backbone, checkpoint['adapt'])
    model = MultiTaskModel(backbone, task_routing=checkpoint['adapt'] == 'moelora')
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    dataset = UnlabeledSAR(args.data, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    preds = {}
    with torch.inference_mode():
        for images, indices in loader:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logits, detections, _ = model(images.to(device))
            labels = logits.argmax(dim=1).tolist()
            for index, label, det in zip(indices.tolist(), labels, detections):
                info = dataset.images[index]
                keep = det['labels'] == label + 1 if checkpoint.get('class_gate', False) else torch.ones_like(
                    det['labels'], dtype=torch.bool)
                boxes = to_original_xywh(det['boxes'][keep], (info['height'], info['width'])).tolist()
                preds[info['file_name']] = (label, boxes, det['labels'][keep].tolist(),
                                            det['scores'][keep].tolist())
    # write_submission expects input-space XYXY boxes. Convert the original-space boxes back
    # to that representation so its single output path handles ZIP formatting consistently.
    for info in dataset.images:
        label, boxes, classes, scores = preds[info['file_name']]
        sx, sy = 512 / info['width'], 512 / info['height']
        xyxy = [[x * sx, y * sy, (x + w) * sx, (y + h) * sy] for x, y, w, h in boxes]
        preds[info['file_name']] = (label, xyxy, [c - 1 for c in classes], scores)
    write_submission(dataset.images, preds, out=args.out)
    print(f'Wrote {args.out}: {len(dataset.images)} classifications, '
          f'{sum(len(p[1]) for p in preds.values())} detections')


if __name__ == '__main__':
    main()
