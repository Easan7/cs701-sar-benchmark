"""Checks for the assignment-eligible adaptation and prediction format."""
import csv
import io
import json
import zipfile

from torch import nn

from make_submission import write_submission
from sarbench.adapters import LoRA, add_adapters


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(8, 24)
        self.attn.proj = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(4))
        self.vit = nn.Module()
        self.vit.norm = nn.LayerNorm(8)


def test_hybrid_trains_only_early_lora_and_last_two_blocks():
    backbone = Backbone()
    add_adapters(backbone, 'hybrid')
    assert isinstance(backbone.blocks[0].attn.qkv, LoRA)
    assert not backbone.blocks[0].mlp.weight.requires_grad
    assert not backbone.blocks[0].attn.qkv.base.weight.requires_grad
    assert backbone.blocks[0].attn.qkv.up.weight.requires_grad
    assert all(p.requires_grad for block in backbone.blocks[-2:] for p in block.parameters())
    assert backbone.vit.norm.weight.requires_grad


def test_submission_has_correct_ids_classes_and_original_coordinates():
    images = [{'id': 17, 'file_name': 'images/a.jpg', 'width': 256, 'height': 512}]
    predictions = {'images/a.jpg': (3, [[20., 40., 60., 100.]], [3], [0.9])}
    output = io.BytesIO()
    write_submission(images, predictions, out=output)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {'classification.csv', 'detection.json'}
        assert list(csv.DictReader(io.StringIO(archive.read('classification.csv').decode()))) == [
            {'image_id': '17', 'label': '3'}]
        assert json.loads(archive.read('detection.json')) == [
            {'image_id': 17, 'category_id': 4, 'bbox': [10., 40., 20., 60.], 'score': 0.9}]
