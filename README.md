# CS701 SAR Multi-Task Benchmark

Reference code for the CS701 team assignment. One ViT-B/16 backbone, two tasks on SAR images: **classify** the
image (9 classes) and **detect** its objects.

![One SAR image, one shared backbone, two heads: a class and the boxes](docs/task_pipeline.png)

## Contents

1. [SAR Task Introduction](#1-sar-task-introduction)
   - [1.1 SAR Basics](#11-sar-basics)
   - [1.2 Task](#12-task)
   - [1.3 Metric](#13-metric)
   - [1.4 Small Objects](#14-small-objects)
2. [Quick Start](#2-quick-start)
   - [2.1 Install](#21-install)
   - [2.2 Train Baseline](#22-train-baseline)
   - [2.3 Use Course Data](#23-use-course-data)
   - [2.4 Make Submission](#24-make-submission)
3. [Baseline Results (for reference)](#3-baseline-results-for-reference)
4. [Code Structure](#4-code-structure)
5. [Links](#5-links)

## Run in Google Colab

Colab cannot access files that exist only on your computer. Push the code changes you want to use to your GitHub
repository first, then clone it in Colab (or upload a ZIP of the current project and unzip it under `/content`).

Use a GPU runtime. The model has 35 M trainable parameters, and 512 px detection can use substantial GPU memory; an
L4 or A100 is preferable. On a 16 GB T4, lower `--batch-size` to 4 or 2 if you run out of memory. Enable GPU in
**Runtime → Change runtime type → T4 GPU** (or the strongest GPU available to your account).

Run these cells in a new notebook:

```bash
!git clone https://github.com/doem97/cs701-sar-benchmark.git
%cd /content/cs701-sar-benchmark
!nvidia-smi
```

Install the pinned project dependencies. This installs the CUDA 12.8 PyTorch wheels specified in
`requirements.txt`; restart the Colab runtime if prompted, then confirm `torch.cuda.is_available()` is `True`.

```bash
!pip install -r requirements.txt
!python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Download and unpack the training archive:

```bash
!pip install -U huggingface_hub
!hf download doem1997/cs701-sar-course-data --repo-type dataset --local-dir cs701-sar-course-data
!tar -xzf cs701-sar-course-data/train.tar.gz -C cs701-sar-course-data
```

Start with a smoke run, then launch the hybrid training command. Change `--batch-size 16` to `4` or `2` if needed.

```bash
!python train.py --data cs701-sar-course-data --backbone vit --init pretrained --adapt hybrid \
  --backbone-lr 2e-5 --lr 1e-4 --select-class-gate --limit 64 --epochs 1 --workers 2 --out runs/smoke
!python train.py --data cs701-sar-course-data --backbone vit --init pretrained --adapt hybrid \
  --backbone-lr 2e-5 --lr 1e-4 --select-class-gate --batch-size 16 --workers 2 --out runs
```

Colab storage is temporary. Copy the best checkpoint to Google Drive before the session ends:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!mkdir -p /content/drive/MyDrive/cs701-sar
!cp runs/vit_pretrained_hybrid/best.pt /content/drive/MyDrive/cs701-sar/best.pt
```

After extracting the released validation or test archive into `cs701-sar-course-data`, predict with the saved
checkpoint and copy the ZIP to Drive. For example, validation:

```bash
!tar -xzf cs701-sar-course-data/val.tar.gz -C cs701-sar-course-data
!python predict.py --checkpoint /content/drive/MyDrive/cs701-sar/best.pt \
  --data cs701-sar-course-data --split val --out submission_val.zip
!cp submission_val.zip /content/drive/MyDrive/cs701-sar/submission_val.zip
```

Use `--split test` and `submission_test.zip` during Phase 2. The `--train-all` option can fit a final model on all
labeled training images after selecting an epoch count and class gate from the holdout run; save that `final.pt`
checkpoint to Drive too.

## 1. SAR Task Introduction

### 1.1 SAR Basics

![A radar sends microwave pulses sideways, records the echoes, and combines them along its flight path into one long virtual antenna](docs/sar_explainer.gif)

- **Active radar**: satellite or aircraft sends microwave pulses, records echoes. Works day, night, through clouds.
  Looks sideways.
- **Synthetic aperture**: echoes along the flight path combined into one long virtual antenna, so fine resolution.
- **Pixel = echo strength**: 8-bit grey, no colour. Metal and corners bright, calm water dark, grainy speckle.

<details>
<summary>What SAR images look like</summary>

![Speckle, bright metal and corners, dark smooth surfaces, bright and dark slopes](docs/sar_effects.png)

</details>

### 1.2 Task

- Input: one SAR image. Output: its class + a box, class and score for each object.
- Each image: one class label, 1 or more boxes, all of that class.
- 9 classes: aircraft, airport, bridge, car, harbor, oil tank, playground, ship, wind turbine.
- Data, formats, release dates: [dataset card](https://huggingface.co/datasets/doem1997/cs701-sar-course-data).
  Rules: [Codabench](https://www.codabench.org/competitions/18271).
- Limits: one model; at most 130 M parameters in total (frozen ones count; this code: 106-107 M) and at most 40 M
  trainable (heads included: LoRA 21.0 M; full fine-tuning and from scratch, 106 M, are over); each image in once at
  ≤ 512 × 512 px; no ensembles, no test-time augmentation. Your report states parameters (total, trainable) and
  compute; full list on Codabench (Terms).

### 1.3 Metric

- Classification: macro-F1. Detection: COCO mAP.
- Ranking: **Δm**, mean relative change of macro-F1 and mAP vs a reference model (ViT + LoRA, the example command
  in 2.2; mean of 3 seeds):
  `Δm = 100% × ½ [(F1 − F1_ref) / F1_ref + (mAP − mAP_ref) / mAP_ref]`
- Δm = 0: as good as the reference. Δm > 0: better. ViT full fine-tuning: about +13 (over the trainable budget: an
  upper bound).

### 1.4 Small Objects

Half of all boxes are smaller than one 16 × 16 ViT patch (at 512 × 512 input).

![A close-up with the 16 px patch grid: wind turbines of 6-10 px and the 15.9 px median box are smaller than one patch](docs/patch_grid.png)

## 2. Quick Start

### 2.1 Install

```bash
conda create -n cs701bench python=3.12 -y && conda activate cs701bench
pip install -r requirements.txt   # CUDA 12.8 builds of torch 2.9.0 / torchvision 0.24.0
```

### 2.2 Train Baseline

```bash
python train.py --backbone vit --init pretrained --adapt lora   # the Δm reference; one run, about 50 min
bash run_all.sh                                                  # all eight runs, about 7 h
```

- Options: `--backbone vit|terramind`, `--init pretrained|scratch`, `--adapt full|lora|moelora|hybrid`
  (`python train.py -h`).
- `--adapt full` and `--init scratch` train 106 M parameters: over the 40 M budget, for reference only.
  They require `--allow-over-budget-reference` to run.
- Pretrained weights download from Hugging Face on first use (ViT 0.4 GB, TerraMind 1.5 GB).
- Times for one RTX PRO 6000 Blackwell GPU. Memory: [docs/DETAILS.md](docs/DETAILS.md#time-and-memory).

### 2.3 Use Course Data

Download and unpack the released training archive in the repository root:

```bash
pip install -U huggingface_hub
hf download doem1997/cs701-sar-course-data --repo-type dataset --local-dir cs701-sar-course-data
cd cs701-sar-course-data && tar -xzf train.tar.gz && cd ..
```

The loader reads `cs701-sar-course-data/train/labels.csv` and
`cs701-sar-course-data/train/instances.json`. Training holds out a deterministic,
class-stratified 10% of labeled train data for local validation. Course val and
test labels are withheld, so they are never used for scoring during training.
Override the holdout with `--val-fraction 0.15`, for example. Run one model with:

```bash
python train.py --data cs701-sar-course-data --backbone vit --init pretrained --adapt lora
```

Use `--limit 64 --epochs 1 --workers 0 --out runs/smoke` for a quick data/model
path check. This is only a smoke run; for useful validation, train on the full
training archive. `run_all.sh` still launches eight long reference experiments.

After the val or test release, download again and unpack its archive in the same
folder. Those releases contain prediction images/manifests only; a separate
inference and submission step must use `val/images.json` or `test/images.json`.

### 2.4 Make Submission

`make_submission.py` has `write_submission` (the function in the briefing video): maps your boxes back to
original image pixels, writes `submission.zip` for Codabench. Inference on the
unlabeled course val/test images is a separate step from `train.py`.

### 2.5 Assignment-eligible hybrid run

The `hybrid` method freezes most of the pretrained ViT, applies rank-16 LoRA to attention in its first ten blocks,
and fully trains its last two blocks and final normalization. Together with the shared classification and detection
heads, this has **106,891,801 total parameters and 35,027,737 trainable parameters**, within both assignment limits.
It is a candidate method, not a measured improvement over the reference. It uses one model and one forward pass
per image. The optional class gate removes detections that disagree with the predicted image class; the local
training holdout determines whether to use it.

```bash
python train.py --data cs701-sar-course-data --backbone vit --init pretrained --adapt hybrid \
  --backbone-lr 2e-5 --lr 1e-4 --select-class-gate --out runs
```

The script saves the best local-holdout checkpoint as `runs/vit_pretrained_hybrid/best.pt`, selected by the assignment
metric using validation reference values. The holdout comes only from the labeled training split. After unpacking
the released validation archive, produce the submission with:

```bash
python predict.py --checkpoint runs/vit_pretrained_hybrid/best.pt --split val \
  --out submission_val.zip
```

Use `--split test` after the test images are released. `predict.py` writes one classification per image and detection
boxes in original image pixels. The holdout checkpoint uses only 90% of the labeled training data. Once its best
epoch and class gate are known from `metrics.json`, fit all 9,392 training images for that many epochs and submit
`runs/vit_pretrained_hybrid_all/final.pt` instead:

```bash
python train.py --data cs701-sar-course-data --backbone vit --init pretrained --adapt hybrid \
  --backbone-lr 2e-5 --lr 1e-4 --train-all --epochs BEST_EPOCH --class-gate --out runs
```

Include `--class-gate` only if the selected holdout checkpoint records `class_gate: true`. The full-data checkpoint
is not locally scored; use the holdout run for method selection and the full-data run for final predictions.

## 3. Baseline Results (for reference)

Test split, %, mean ± std over 3 seeds. Δm reference: ViT + LoRA, mean of its 3 seeds (93.99 / 25.41), so its Δm
is 0. Full fine-tuning and from scratch train 106 M parameters with the heads: over the 40 M budget, shown for
reference (full fine-tuning: the upper bound).

| backbone | adaptation | trained backbone params | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|---:|
| ViT (ImageNet-21k) | full fine-tuning | 86.0 M | 95.8 ± 0.3 | 31.8 ± 0.2 | +13.5 ± 0.4 |
| ViT (ImageNet-21k) | LoRA | 0.88 M | 94.0 ± 0.2 | 25.4 ± 0.5 | 0.0 ± 0.9 |
| ViT (ImageNet-21k) | MoE-LoRA | 1.03 M | 93.4 ± 0.5 | 26.8 ± 0.3 | +2.5 ± 0.8 |
| TerraMind (Earth observation) | full fine-tuning | 85.3 M | 95.6 ± 0.2 | 30.8 ± 0.4 | +11.5 ± 0.7 |
| TerraMind (Earth observation) | LoRA | 0.88 M | 92.6 ± 0.4 | 26.5 ± 0.4 | +1.4 ± 1.0 |
| TerraMind (Earth observation) | MoE-LoRA | 1.03 M | 92.4 ± 0.1 | 27.8 ± 0.3 | +3.8 ± 0.7 |
| ViT, from scratch | full training | 86.0 M | 82.8 ± 0.7 | 17.0 ± 0.3 | −22.5 ± 0.9 |
| TerraMind, from scratch | full training | 85.3 M | 86.1 ± 0.2 | 17.8 ± 0.5 | −19.3 ± 0.8 |

<details>
<summary>Val results</summary>

Val split, %, mean ± std over 3 seeds. Δm reference: the same configuration on val (93.31 / 26.13).

| backbone | adaptation | trained backbone params | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|---:|
| ViT (ImageNet-21k) | full fine-tuning | 86.0 M | 94.4 ± 0.3 | 32.0 ± 0.4 | +11.8 ± 0.7 |
| ViT (ImageNet-21k) | LoRA | 0.88 M | 93.3 ± 0.4 | 26.1 ± 0.4 | 0.0 ± 0.9 |
| ViT (ImageNet-21k) | MoE-LoRA | 1.03 M | 92.2 ± 0.8 | 27.1 ± 0.2 | +1.3 ± 0.9 |
| TerraMind (Earth observation) | full fine-tuning | 85.3 M | 94.2 ± 0.3 | 31.5 ± 0.1 | +10.7 ± 0.3 |
| TerraMind (Earth observation) | LoRA | 0.88 M | 92.2 ± 0.4 | 27.0 ± 0.5 | +1.0 ± 1.3 |
| TerraMind (Earth observation) | MoE-LoRA | 1.03 M | 91.9 ± 0.4 | 28.8 ± 0.1 | +4.4 ± 0.3 |
| ViT, from scratch | full training | 86.0 M | 83.5 ± 0.9 | 18.2 ± 0.5 | −20.3 ± 1.3 |
| TerraMind, from scratch | full training | 85.3 M | 86.0 ± 0.9 | 19.2 ± 0.4 | −17.3 ± 1.2 |

</details>

Accuracy and AP50: [docs/DETAILS.md](docs/DETAILS.md#results-in-full).

## 4. Code Structure

```
train.py              entry: train + evaluate one configuration
run_all.sh            the eight reference runs (calls train.py)
make_submission.py    write_submission(): predictions -> submission.zip
sarbench/             the code behind train.py
├── data.py           images + boxes at 512 × 512, flips; boxes back to original pixels
├── backbones.py      ViT (ImageNet-21k), TerraMind-1.0-base
├── adapters.py       LoRA, MoE-LoRA on a frozen backbone
├── model.py          backbone + class head + ViTDet feature pyramid + Faster R-CNN
└── metrics.py        accuracy, macro-F1, COCO box AP
tests/                checks of sarbench/ (python -m pytest tests)
docs/DETAILS.md       model, training protocol, design notes, time and memory
```

More: [docs/DETAILS.md](docs/DETAILS.md).

## 5. Links

- Data: https://huggingface.co/datasets/doem1997/cs701-sar-course-data
- Leaderboard: https://www.codabench.org/competitions/18271
- Questions: TA Zichen, zichen.tian.2023@phdcs.smu.edu.sg
