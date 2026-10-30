"""Run one trained model once per unlabeled image and write a Codabench ZIP.

Example: python predict.py --checkpoint runs/vit_pretrained_hybrid/best.pt --split val
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from sarbench.adapters import configure_adaptation
from sarbench.backbones import build_backbone
from sarbench.data import MEAN, STD, to_original_xywh
from sarbench.model import MultiTaskModel
from make_submission import write_submission


class UnlabeledSAR(Dataset):
    def __init__(self, root, split, mean=MEAN, std=STD):
        self.folder = Path(root) / split
        self.mean, self.std = float(mean), float(std)
        self.images = json.loads((self.folder / 'images.json').read_text())
        if isinstance(self.images, dict):
            self.images = self.images['images']
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        path = self.folder / info['file_name']
        if not path.exists():
            path = self.folder / 'images' / info['file_name']
        with Image.open(path) as image:
            tensor = self.transform(image.convert('L'))
        h, w = tensor.shape[-2:]
        if max(h, w) > 512:
            raise ValueError(f'Input {info["file_name"]} exceeds the 512px assignment limit')
        tensor = (tensor - self.mean) / self.std
        tensor = F.pad(tensor, (0, 512 - w, 0, 512 - h))
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
    configure_adaptation(backbone, checkpoint['adapt'], rank=checkpoint.get('lora_rank', 16),
                         alpha=checkpoint.get('lora_alpha', 32),
                         unfreeze_blocks=checkpoint.get('unfreeze_blocks', 2))
    model = MultiTaskModel(backbone, task_routing=checkpoint['adapt'] == 'moelora',
                           raw_pixel_branch=checkpoint.get('raw_pixel_branch', False),
                           **checkpoint.get('detector_config', {}))
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    amp_dtype = (torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported()
                 else torch.float16)
    dataset = UnlabeledSAR(args.data, args.split, checkpoint.get('mean', MEAN), checkpoint.get('std', STD))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    preds = {}
    with torch.inference_mode():
        for images, indices in loader:
            batch_infos = [dataset.images[index] for index in indices.tolist()]
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
                logits, detections, _ = model(images.to(device),
                                              image_sizes=[(info['height'], info['width']) for info in batch_infos])
            labels = logits.argmax(dim=1).tolist()
            for index, label, det in zip(indices.tolist(), labels, detections):
                info = dataset.images[index]
                keep = det['labels'] == label + 1 if checkpoint.get('class_gate', False) else torch.ones_like(
                    det['labels'], dtype=torch.bool)
                boxes = det['boxes'][keep].tolist()
                preds[info['file_name']] = (label, boxes, det['labels'][keep].tolist(),
                                            det['scores'][keep].tolist())
    for info in dataset.images:
        label, boxes, classes, scores = preds[info['file_name']]
        preds[info['file_name']] = (label, boxes, [c - 1 for c in classes], scores)
    write_submission(dataset.images, preds, out=args.out, boxes_are_original=True)
    print(f'Wrote {args.out}: {len(dataset.images)} classifications, '
          f'{sum(len(p[1]) for p in preds.values())} detections')


if __name__ == '__main__':
    main()
