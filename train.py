"""Train one model jointly for classification and detection, validating on a holdout from labeled train data.
Example: python train.py --backbone vit --init pretrained --adapt lora"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sarbench.adapters import add_adapters
from sarbench.backbones import build_backbone
from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.model import MultiTaskModel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parent / 'cs701-sar-course-data')
    p.add_argument('--backbone', required=True, choices=['vit', 'terramind'])
    p.add_argument('--init', required=True, choices=['pretrained', 'scratch'])
    p.add_argument('--adapt', required=True, choices=['full', 'lora', 'moelora', 'hybrid'])
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone-lr', type=float, help='for the trainable backbone parameters (default: --lr)')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--eval-every', type=int, default=4, help='validate every N epochs, and after the last one')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='use only the first N labeled training images (quick checks)')
    p.add_argument('--val-fraction', type=float, default=0.1, help='fraction of labeled train images held out locally')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path('runs'))
    p.add_argument('--select-class-gate', action='store_true',
                   help='compare detections restricted to the predicted image class on the train holdout')
    p.add_argument('--allow-over-budget-reference', action='store_true',
                   help='run an over-budget scratch/full baseline for research only; not assignment eligible')
    p.add_argument('--train-all', action='store_true',
                   help='fit all labeled training images after selecting an epoch count on a holdout')
    p.add_argument('--class-gate', action='store_true',
                   help='store the selected class gate in a --train-all checkpoint')
    args = p.parse_args()
    if args.init == 'scratch' and args.adapt != 'full':
        p.error('--init scratch is only valid with --adapt full')
    if args.backbone_lr is None:
        args.backbone_lr = args.lr
    return args


def log(message, file):
    """Print to the console and to the run's log.txt."""
    print(message, flush=True)
    print(message, file=file, flush=True)


def make_loader(args, indices, train):
    """Batches from labeled train; only the training subset is shuffled and augmented."""
    dataset = SARMultiTask(args.data, 'train', train=train, limit=args.limit, indices=indices)
    return DataLoader(dataset, args.batch_size, shuffle=train, num_workers=args.workers, collate_fn=collate,
                      persistent_workers=train and args.workers > 0)  # training workers live across epochs


def lr_factor(step, warmup, total):
    """Learning-rate multiplier: linear warmup over `warmup` steps, then cosine decay to 0 at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))  # --epochs 1: no decay phase


def train_one_epoch(model, loader, optimizer, schedule, epoch, log_file):
    """One pass over the training set; the loss is cross-entropy plus the four Faster R-CNN losses."""
    model.train()
    start, seen = time.time(), 0
    for step, (images, labels, targets) in enumerate(loader, 1):
        targets = [{'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda()} for t in targets]
        with torch.autocast('cuda', dtype=torch.bfloat16):  # bf16 has fp32's range: no loss scaling needed
            logits, _, det_losses = model(images.cuda(), targets)
            losses = {'classification': F.cross_entropy(logits, labels.cuda()), **det_losses}
        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        seen += len(images)
        if step % 50 == 0 or step == len(loader):
            parts = ' '.join(f'{name} {value.item():.3f}' for name, value in losses.items())
            log(f'epoch {epoch} iter {step}/{len(loader)} loss {loss.item():.3f} ({parts}) '
                f'lr {optimizer.param_groups[1]["lr"]:.2e} {seen / (time.time() - start):.1f} images/s', log_file)
            start, seen = time.time(), 0


@torch.no_grad()
def evaluate(model, loader, gt_json, class_gate=False):
    """Classification and COCO detection metrics on one split, and the detections in COCO result format."""
    model.eval()
    labels, predicted, detections, image_ids = [], [], [], []
    for images, y, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda())
        labels += y.tolist()
        batch_classes = logits.argmax(dim=1).tolist()
        predicted += batch_classes
        for target, out, image_class in zip(targets, outputs, batch_classes):
            image_ids.append(target['image_id'])
            boxes = to_original_xywh(out['boxes'], target['orig_size']).tolist()
            detections += [{'image_id': target['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                           for b, c, s in zip(boxes, out['labels'].tolist(), out['scores'].tolist())
                           if not class_gate or c == image_class + 1]
    metrics = {**classification_metrics(labels, predicted), **detection_metrics(gt_json, detections, image_ids)}
    return metrics, detections


def headline(metrics):
    return ' '.join(f'{k} {metrics[k]:.4f}' for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50'))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a CUDA GPU; none is available in this environment')
    if not 0 < args.val_fraction < 1:
        raise ValueError('--val-fraction must be between 0 and 1')
    if args.train_all and args.select_class_gate:
        raise ValueError('--select-class-gate needs a holdout; use --class-gate with --train-all')
    torch.manual_seed(args.seed)
    run_dir = args.out / f'{args.backbone}_{args.init}_{args.adapt}{"_all" if args.train_all else ""}'
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / 'log.txt', 'w')
    log(json.dumps(vars(args), default=str), log_file)

    # Course val/test labels are withheld. Make a deterministic, class-stratified holdout from train.
    all_data = SARMultiTask(args.data, 'train', limit=args.limit)
    train_indices, val_indices = [], []
    if args.train_all:
        train_indices = list(range(len(all_data)))
    else:
        by_class = {}
        for i, row in enumerate(all_data.rows):
            by_class.setdefault(int(row['label']), []).append(i)
        rng = random.Random(args.seed)
        for indices in by_class.values():
            rng.shuffle(indices)
            n_val = max(1, round(len(indices) * args.val_fraction)) if len(indices) > 1 else 0
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])
    train_loader = make_loader(args, train_indices, True)
    if not args.train_all:
        val_loader = make_loader(args, val_indices, False)
        val_rows = [all_data.rows[i] for i in val_indices]
        val_ids = {all_data.coco_images[row['file_name']]['id'] for row in val_rows}
        val_gt = {**all_data.coco, 'images': [im for im in all_data.coco['images'] if im['id'] in val_ids],
                  'annotations': [a for a in all_data.coco['annotations'] if a['image_id'] in val_ids]}

    # Model: adapters are created on the CPU, so they are added before .cuda()
    backbone = build_backbone(args.backbone, pretrained=args.init == 'pretrained')
    if args.adapt != 'full':
        add_adapters(backbone, args.adapt)  # also freezes the backbone
    model = MultiTaskModel(backbone, task_routing=args.adapt == 'moelora').cuda()
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'parameters: {parameters:,} total, {trainable:,} trainable', log_file)
    if (parameters > 130_000_000 or trainable > 40_000_000) and not args.allow_over_budget_reference:
        raise ValueError(f'Assignment parameter budget exceeded: {parameters:,} total, {trainable:,} trainable')

    # Optimizer and schedule: trainable backbone parameters at --backbone-lr, all new modules at --lr
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': backbone_params, 'lr': args.backbone_lr}, {'params': head_params}],
                                  lr=args.lr, weight_decay=args.weight_decay)
    warmup, total = len(train_loader), args.epochs * len(train_loader)  # one epoch of warmup
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))

    # Train, validating every --eval-every epochs; the reported model is the last-epoch model
    history, start, best_score = [], time.time(), -float('inf')
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, schedule, epoch, log_file)
        if not args.train_all and (epoch % args.eval_every == 0 or epoch == args.epochs):
            choices = [False, True] if args.select_class_gate else [False]
            candidates = [(*evaluate(model, val_loader, val_gt, class_gate=gate), gate) for gate in choices]
            val, val_detections, gate = max(
                candidates, key=lambda x: 0.5 * (x[0]['macro_f1'] / 0.9331 + x[0]['mAP'] / 0.2613))
            score = 50 * (val['macro_f1'] / 0.9331 + val['mAP'] / 0.2613 - 2)
            history.append({'epoch': epoch, 'delta_m_proxy': score, 'class_gate': gate, **val})
            log(f'epoch {epoch} val: {headline(val)} delta_m_proxy {score:.2f}% class_gate {gate}', log_file)
            if score > best_score:
                best_score = score
                torch.save({'model': model.state_dict(), 'backbone': args.backbone, 'adapt': args.adapt,
                            'class_gate': gate, 'epoch': epoch, 'val': val}, run_dir / 'best.pt')
                (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
    train_hours = (time.time() - start) / 3600

    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log(f'training {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB', log_file)

    # Save
    summary = {'args': vars(args), 'parameters': parameters, 'trainable_parameters': trainable,
               'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb,
               'val_history': history, 'best': max(history, key=lambda x: x['delta_m_proxy']) if history else None}
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    if args.train_all:
        torch.save({'model': model.state_dict(), 'backbone': args.backbone, 'adapt': args.adapt,
                    'class_gate': args.class_gate, 'epoch': args.epochs}, run_dir / 'final.pt')
        log(f'final checkpoint: epoch {args.epochs}', log_file)
    else:
        log(f'best checkpoint: epoch {summary["best"]["epoch"]}, holdout delta_m_proxy '
            f'{summary["best"]["delta_m_proxy"]:.2f}%', log_file)


if __name__ == '__main__':
    main()
