# Dermoscopic lesion segmentation

Reproducible pipeline for the Imperial College London Summer School project:

1. lesion segmentation;
2. segmentation and presence detection for five dermoscopic attributes;
3. evidence-anchored JSON and short-text findings reports.

The original training data remain outside the repository. Scripts receive the
dataset root explicitly, so no 6 GB data copy is required.

## Environment

Use a native ARM64 Python 3.10-3.12 environment on Apple Silicon:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install .
```

## Dataset audit

```bash
python scripts/audit_dataset.py \
  --data-root "/path/to/train" \
  --output-dir artifacts
```

The command creates:

- `artifacts/manifest.csv`: one aligned row per image;
- `artifacts/dataset_audit.json`: machine-readable quality-control statistics;
- `artifacts/DATASET_AUDIT.md`: concise human-readable audit report.

## Deterministic split

```bash
python scripts/create_split.py \
  --manifest artifacts/manifest.csv \
  --output splits/train_val.csv
```

The default seed is `20260727`. The split is stratified by the complete
five-attribute presence signature and by lesion-size category.

## Training baselines

Task 1:

```bash
python scripts/cache_task1.py \
  --output-dir cache/task1-384-letterbox \
  --image-size 384

python scripts/train.py \
  --task task1 \
  --architecture unetplusplus \
  --image-size 384 \
  --resize-mode letterbox \
  --cache-root cache/task1-384-letterbox \
  --batch-size 8 \
  --task1-loss lovasz_dice \
  --freeze-encoder-epochs 1 \
  --lr-scheduler-patience 1 \
  --lr-scheduler-factor 0.3 \
  --patience 5 \
  --epochs 25 \
  --device auto
```

The Task 1 cache preserves the original aspect ratio, pads with the ImageNet
mean colour, and avoids repeatedly decoding multi-megapixel JPEG files.
`--init-checkpoint checkpoints/task1-baseline-256/best.pt` can be used to
fine-tune a higher-resolution run from the 256-pixel baseline.
Task 1 supports `focal_dice`, `bce_dice`, and `lovasz_dice` through
`--task1-loss`. Encoder and decoder learning rates can be set independently
with `--encoder-learning-rate` and `--learning-rate`.
For ImageNet initialization, `--freeze-encoder-epochs` provides a decoder
warm-up before end-to-end fine-tuning. The plateau scheduler can be tuned with
`--lr-scheduler-patience`, `--lr-scheduler-factor`, and
`--min-learning-rate`; keep the early-stopping `--patience` longer than the
scheduler patience so a reduced learning rate has time to take effect.
`--architecture` supports `unet`, `unetplusplus`, and `deeplabv3plus`.
When changing decoder architecture, `--init-encoder-checkpoint` transfers only
the compatible pretrained encoder from an existing checkpoint.

Task 2 uses five segmentation channels plus an auxiliary five-label presence
head:

```bash
python scripts/train.py \
  --task task2 \
  --image-size 384 \
  --batch-size 4 \
  --device auto
```

Both commands save the best and latest checkpoints, the run configuration, and
one JSON record per epoch under `checkpoints/<task>/`.

Resume an interrupted run with the same task, encoder, and image size:

```bash
python scripts/train.py \
  --task task2 \
  --resume checkpoints/task2/last.pt \
  --epochs 30 \
  --device auto
```

Task 2 uses a class-balanced, positive-case Dice term so rare attributes are
not overwhelmed by empty masks or by the more frequent pigment-network class.

## Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/task1/best.pt \
  --workers 4 \
  --device auto

python scripts/evaluate.py \
  --checkpoint checkpoints/task2/best.pt \
  --workers 4 \
  --device auto
```

Task 1 evaluation includes Dice, IoU, and HD95. Task 2 includes overall and
positive-case segmentation metrics per attribute, plus average precision and
ROC AUC for the auxiliary presence head.

For rigorous Task 1 comparisons, evaluate restored predictions against the
native-resolution masks and inspect lesion-size and border-touching subgroups:

```bash
python scripts/evaluate_task1_advanced.py \
  --checkpoint checkpoints/task1-lovasz-384/best.pt \
  --ensemble-checkpoint checkpoints/task1-unetplusplus-384-lovasz/best.pt \
  --ensemble-weights 1,1 \
  --batch-size 4 \
  --workers 4 \
  --thresholds 0.40,0.45,0.50 \
  --selected-threshold 0.50 \
  --tta d4 \
  --device auto
```

This also writes per-case metrics, a threshold sweep, a bootstrap confidence
interval, and the twenty lowest-Dice cases.
Compatible checkpoints can be averaged by repeating `--ensemble-checkpoint`
and optionally setting `--ensemble-weights`. `--image-size` together with
`--disable-cache` supports controlled inference-resolution ablations.

Create a qualitative Task 1 montage:

```bash
python scripts/visualize_task1_predictions.py \
  --checkpoint checkpoints/task1/best.pt \
  --output artifacts/task1_predictions.jpg \
  --device auto
```

## Findings reports

`lesion_segmentation.reporting` converts model evidence into a deterministic
JSON payload and a short findings sentence. It describes lesion size, border
regularity, and the required five attributes. Probabilities between 0.40 and
0.60 are explicitly reported as uncertain. These outputs are research
prototypes and must not be interpreted as clinical diagnoses.

Run the complete two-model inference pipeline on one image or a directory:

```bash
python scripts/predict_reports.py \
  --images "/path/to/images" \
  --task1-checkpoint checkpoints/task1/best.pt \
  --task2-checkpoint checkpoints/task2/best.pt \
  --output-dir outputs \
  --device auto
```

The command writes lesion masks, five attribute-mask directories,
`reports.jsonl`, and `reports.txt`.
