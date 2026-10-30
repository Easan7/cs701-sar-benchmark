"""sarbench/data.py on the real dataset: split contents, the box geometry round trip, and flips."""
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors

from sarbench.data import MEAN, STD, SARMultiTask, collate, to_original_xywh, _flip_boxes, _rot90_boxes
from sarbench.metrics import detection_metrics

ROOT = Path(__file__).resolve().parents[2] / 'dataset' / 'SARFact-Course-20K'


def batches(split):
    return DataLoader(SARMultiTask(ROOT, split), batch_size=64, collate_fn=collate)


@pytest.mark.parametrize('split, size', [('train', 9392), ('val', 1426), ('test', 2065)])
def test_subset_a(split, size):
    loader = batches(split)
    assert len(loader.dataset) == size
    for images, labels, targets in loader:
        assert images.shape[1:] == (1, 512, 512) and images.dtype == torch.float32
        for label, target in zip(labels.tolist(), targets):
            assert len(target['labels']) >= 1
            assert (target['labels'] == label + 1).all()
            assert target['boxes'].min() >= 0 and target['boxes'].max() <= 512


@pytest.mark.parametrize('split', ['val', 'test'])
def test_ground_truth_round_trip_scores_map_1(split):
    predictions, image_ids = [], []
    for _, _, targets in batches(split):
        for t in targets:
            image_ids.append(t['image_id'])
            boxes = to_original_xywh(t['boxes'], t['orig_size']).tolist()
            predictions += [{'image_id': t['image_id'], 'category_id': c, 'bbox': b, 'score': 1.0}
                            for b, c in zip(boxes, t['labels'].tolist())]
    metrics = detection_metrics(ROOT / 'detection' / f'instances_{split}.json', predictions, image_ids)
    assert round(metrics['mAP'], 3) == 1.0
    assert [round(ap, 3) for ap in metrics['per_class_AP'].values()] == [1.0] * 9


def test_exact_geometry_preserves_boxes_without_resizing():
    box = torch.tensor([[30., 50., 120., 90.]])
    assert torch.equal(_flip_boxes(box, 300, horizontal=True), torch.tensor([[180., 50., 270., 90.]]))
    assert torch.equal(_flip_boxes(box, 200, horizontal=False), torch.tensor([[30., 110., 120., 150.]]))
    assert torch.equal(_rot90_boxes(box, 200, 300, 1), torch.tensor([[50., 180., 90., 270.]]))
    assert torch.equal(_rot90_boxes(box, 200, 300, 2), torch.tensor([[180., 110., 270., 150.]]))
    assert torch.equal(to_original_xywh(box, (200, 300)), torch.tensor([[30., 50., 90., 40.]]))
