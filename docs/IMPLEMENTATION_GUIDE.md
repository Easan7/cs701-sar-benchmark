# Implementation guide: compliant SAR multi-task experiments

## Repository audit

The existing project already included a shared ViT-B/16 or TerraMind backbone, scene classification, a ViTDet-style simple feature pyramid, Faster R-CNN with small anchors, LoRA/MoE-LoRA adapters, a stratified local holdout, COCO metrics, and a ZIP submission writer. It also resized all images to 512×512, so native 256px scenes were enlarged. Its stride-4/8 detector features came from upsampled stride-16 tokens, and normalization constants were computed on resized training images.

The implementation now provides native-resolution inputs with bottom/right padding, train-only native-pixel mean/std computation, exact box transforms for flips and quarter-turn rotations, mild optional grayscale gain/gamma/speckle augmentation, a genuine raw-pixel stride-4/8 branch, FPN candidates for hierarchical backbones, fixed or uncertainty-based task loss weighting, parameter-budget checks, configuration files, and original-coordinate submission output. Existing uncommitted changes in `README.md` and `sarbench/backbones.py` were retained.

No model initialization, forward pass, training, inference, evaluation, test, benchmark, download, or dependency installation was performed. Thus there are no measured parameter counts or performance claims in this report.

## Assignment constraints encoded

- One jointly trained multi-task model produces scene logits and object detections. The raw-pixel branch is fused into the same ViT detector and checkpoint.
- Input images must be no larger than 512px in either dimension. Smaller images remain at native resolution and are padded on the bottom and right; predictions are clipped and serialized in original image coordinates.
- `sarbench.compliance.validate_model_budget` counts frozen and trainable model parameters and enforces 130M / 40M limits. The training entry point checks the selected model before optimization begins.
- The final predictor runs one checkpoint, with no TTA, tiling, super-resolution, or model ensemble.
- Training normalization statistics are calculated from the training indices only, excluding the local holdout, then stored with the checkpoint. Training-only augmentation is disabled in validation and inference.
- Course val/test labels are not used by the trainer. The built-in holdout is drawn from labeled train data.

The no-upscaling interpretation needs confirmation against the current course brief/TA. Padding is used to preserve small native scenes on the current fixed 512 canvas. Confirm that padding is accepted before a submission.

## Architectures and rationale

| Configuration | Integrated model | Status and rationale |
|---|---|---|
| `configs/baseline.json` | ImageNet-21k ViT-B/16 + LoRA + existing FPN/Faster R-CNN | Closest to the repository baseline; starts with native geometry. |
| `configs/raw_p2_hybrid.json` | ViT-B/16 + earlier-block LoRA + final-block tuning + raw-pixel P2/P3 gated fusion + shared detector | Primary research hypothesis. Raw pixels provide true stride-4/8 evidence while ViT contributes global context. |
| `configs/terramind_hybrid.json` | TerraMind + MoE-LoRA + raw-pixel fusion | Experimental. Input has one grayscale band duplicated to meet the two-channel S1GRD interface; this does not create VV/VH information. It incurs two backbone passes as currently implemented. |
| `configs/swin_tiny_fpn.json` | Generic pretrained hierarchical Swin-T + FPN + shared classification/detection | Experimental multiscale transformer alternative. Verify padding/window behavior and actual parameter budget. |
| `configs/resnet50_fpn.json` | Generic pretrained ResNet-50 feature pyramid + shared heads | Experimental CNN inductive-bias comparison. Partial tuning is provided because a fully trainable backbone plus heads may exceed 40M. |
| `configs/convnext_tiny_fpn.json` | Generic pretrained ConvNeXt-Tiny feature pyramid + shared heads | Experimental CNN alternative; budget validation is mandatory. |

`configs/best.json` is a *starting hypothesis* based on the research plan, not an empirically selected winner. It uses ViT hybrid raw P2/P3 fusion, quarter-turn rotation, mild gain, and a 0.5:1 classification/detection weight. No local comparison has been run.

Candidate CNN/Swin weights are requested from timm's generic ImageNet model registry only when the user later starts training with `--init pretrained`. This repository contains no downloaded weights. Do not substitute a SAR-trained checkpoint or external SAR data. The TerraMind source and its SAR-domain caveat are described in the research plan.

## Preprocessing and training

Use PNG/JPEG grayscale decoded as `L`, scaled to [0,1], normalized by statistics computed over only the training subset. Native pixels are never resized. All samples are padded to a 512×512 canvas. Exact flips and 90° rotations transform XYXY boxes with the same integer geometry. Gain, gamma, and multiplicative speckle are configurable and disabled by default unless enabled in a config. Filtering, CLAHE, arbitrary affine transforms, random crops, MixUp, and CopyPaste remain off due to risk to tiny targets or annotation validity.

The detector retains the reference small-anchor setup and its FP32 RPN/RoI path. The raw branch adds low-level P2/P3 features through small-initialized sigmoid gates. It does not replace the ViT or create a second independently trained model.

Fixed task weighting is default. `--loss-weighting uncertainty` enables Kendall-style two-task uncertainty weighting (classification and the sum of all detector losses). Individual classification, RPN, and RoI losses remain visible in logs. Do not stack this with GradNorm/PCGrad before diagnosing task-gradient conflict.

## Colab workflow

Before opening Colab, review the local changes, commit and push the intended source/config/docs files to the GitHub repository, and do not include downloaded datasets or run outputs. Colab clones GitHub and cannot see unpushed VS Code changes. In a new notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/Easan7/cs701-sar-benchmark.git
%cd /content/cs701-sar-benchmark
!nvidia-smi
!pip install -r requirements.txt
```

Use the repository README's dataset section to fetch/extract only the permitted course archive under
`/content/cs701-sar-benchmark/cs701-sar-course-data/`. Keep extracted data on Colab's local disk for faster reads and
put checkpoints/logs on mounted Drive. Colab's free GPU type, availability, memory, and session limits can change;
check `nvidia-smi` on every session. The current runtime code selects BF16 where available and otherwise uses FP16
with gradient scaling, while keeping detector box operations in FP32.

For screening, first use 2–3 epochs for each config with the same holdout and seed. Run a full 24-epoch comparison only
for the strongest candidates if free-runtime time allows. `--eval-every 1` writes a best checkpoint every epoch.
Different config names now create separate run directories. Example commands (not executed here):

```bash
python train.py --data cs701-sar-course-data --config configs/baseline.json --epochs 3 --eval-every 1 --batch-size 4 --out /content/drive/MyDrive/cs701-sar/runs
python train.py --data cs701-sar-course-data --config configs/sar_aug.json --epochs 3 --eval-every 1 --batch-size 4 --out /content/drive/MyDrive/cs701-sar/runs
python train.py --data cs701-sar-course-data --config configs/partial_ft.json --epochs 3 --eval-every 1 --batch-size 4 --out /content/drive/MyDrive/cs701-sar/runs
python train.py --data cs701-sar-course-data --config configs/raw_p2_hybrid.json --epochs 3 --eval-every 1 --batch-size 4 --out /content/drive/MyDrive/cs701-sar/runs
python predict.py --data cs701-sar-course-data --checkpoint /content/drive/MyDrive/cs701-sar/runs/raw_p2_hybrid/best.pt --split val --out /content/drive/MyDrive/cs701-sar/submission.zip
python -m pytest tests/test_data.py::test_exact_geometry_preserves_boxes_without_resizing tests/test_competition.py::test_submission_has_correct_ids_classes_and_original_coordinates tests/test_compliance.py tests/test_losses.py
```

These are the quick non-training checks. Avoid running `python -m pytest tests` as a quick suite: the repository also contains a test that performs a 200-step GPU overfit run. Store `runs/` on mounted Drive if checkpoints must persist between Colab sessions. Reduce batch size if GPU memory is insufficient; gradient accumulation is not yet wired into the training CLI. Checkpoint resumption and Drive mounting are user-controlled Colab steps; training currently saves best and final checkpoints locally and does not resume optimizer/scheduler state.

### Free-runtime time estimate

These are planning estimates, **not measurements**. A 15% holdout leaves about 7,983 training images, or about 1,996 steps per epoch at batch size 4. For a 24-epoch run that is roughly 47,900 optimizer steps. Depending on the free GPU and actual batch time, a 1–4 second step implies about **13–53 hours of training per config**, before validation overhead. A 3-epoch screening run is about **1.7–6.7 hours per config** at the same assumed step times; four configs would take about **7–27 hours** plus validation and setup. Time one representative batch/epoch in Colab first and extrapolate with `steps × measured seconds/step`; actual results may fall outside these ranges.

Free Colab sessions can disconnect and available compute varies. Checkpoints are saved to Drive when best validation improves, but optimizer/scheduler resumption is not implemented, so an interrupted run cannot continue exactly where it stopped. The official Colab FAQ could not be reached while preparing this guide; treat free-tier session/usage limits as variable and check the current Colab UI.

## Recommended experiment order

1. Run the baseline config on a fixed local train holdout and confirm native geometry, parameter caps, and submission schema.
2. Compare exact quarter-turn augmentation against baseline, then gain/gamma separately; do not enable all radiometric changes at once.
3. Compare fixed loss weights (1,1) and (0.5,1), then uncertainty weighting as a separate experiment.
4. Compare hybrid partial tuning against LoRA-only with the same split/seed.
5. Compare raw P2/P3 fusion to the best prior ViT candidate; record APsmall, AP75, recall, macro-F1, and the official composite.
6. Only then try TerraMind, Swin-T, ResNet-50, and ConvNeXt-Tiny. Select one eligible checkpoint based on measured local validation and repeated seeds for close differences.

Record each run in `configs/experiment_ledger.csv`: git SHA, split hash, seed, GPU time, total/trainable parameters, F1, AP, AP50/AP75/APsmall, and notes. The current file is a header only.

## Unverified issues to check before training/submission

- Confirm native padding is permitted under the no-upscaling wording. If not, seek course clarification before changing geometry.
- Confirm the pretrained initialization source/license and timm model availability in Colab without SAR-derived weights.
- Confirm TerraTorch's exact input size and feature shape; duplicated grayscale channels are only a compatibility proxy.
- Confirm Swin window padding and feature-map dimensions at 512px, including mixed native sizes after padding.
- Inspect COCO category IDs and CSV-to-image ID consistency in the actual course archive; no data archive was present in this repository inspection.
- Validate bounding-box round trips on non-square, corner, and tiny-box examples before the first GPU run.
- Selected quick checks (geometry, ZIP schema, budget/input limits, and loss weighting) were run directly and passed. Pytest is not installed in the local environment, and no tests were installed or run through pytest. Run the full written suite in Colab, then do one short controlled training/holdout experiment.
- Check actual parameter counts for every configuration. Some full CNN tuning configurations will be rejected by the runtime budget gate; prefer partial tuning.
- Effective-batch gradient accumulation, optimizer/scheduler resume, group-aware near-duplicate splitting, proposal-recall dashboards, and full per-size AP analysis are not yet implemented.

## Research references

- ViTDet simple feature pyramids: [Li et al., ECCV 2022](https://arxiv.org/abs/2203.16527).
- Image-specific dense-prediction adapters and FPN motivation: [ViT-Adapter, ICLR 2023](https://arxiv.org/abs/2205.08534) and [Feature Pyramid Networks, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html).
- Low-rank adaptation: [Hu et al., ICLR 2022](https://arxiv.org/abs/2106.09685); task-specific low-rank adaptation: [MTLoRA, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Agiza_MTLoRA_Low-Rank_Adaptation_Approach_for_Efficient_Multi-Task_Learning_CVPR_2024_paper.html).
- Multi-task uncertainty weighting: [Kendall et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_Uncertainty_CVPR_2018_paper.html).
- Hierarchical Swin backbone: [Liu et al., ICCV 2021](https://arxiv.org/abs/2103.14030).
- SAR filtering tradeoffs: [Argenti et al., IEEE GRSM 2013](https://doi.org/10.1109/MGRS.2013.2277512). Filtering is not enabled by default because it can erase small scattering structures.

These papers motivate code paths and hypotheses; they do not establish results on this course split. TerraMind-specific input details and the prohibition on SARDet-derived data/weights are documented in the supplied research plan and must be observed when selecting checkpoints.
