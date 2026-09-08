# Deep-Sea Coral Detection Pipeline

Instance segmentation of deep-sea corals in ROV imagery using YOLOv11x-seg. The pipeline covers data augmentation, model training, image inference, and automated ROV video analysis with laser-calibrated coral density estimation.

---

## Requirements

- Python >= 3.10
- CUDA-capable GPU (tested on NVIDIA RTX 3090, 24 GB VRAM)
- **Tesseract OCR binary** — required for depth extraction from video overlays:
  - Windows: download and run the [official installer](https://github.com/UB-Mannheim/tesseract/wiki)
  - Linux: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`

---

## Data availability

The trained model weights and an example ROV video are hosted on Zenodo,
because they are too large for GitHub.

- **`Weights/`** — trained weights for the coral segmentation model, ready
  for inference (Steps 3–4). See [`Weights/README.md`](Weights/README.md)
  in this repo for what's in each group's folder (final soup + ingredient
  checkpoints) and the reported test-set metrics.
- **`Video_example/`** — an example ROV video transect
  (`CHR0389_LowerMesophotic.mp4`) used for the video analysis in Step 4.

Download them from Zenodo and place the `Weights/` and `Video_example/`
folders in the repository root, keeping the folder names unchanged.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19703699.svg)](https://doi.org/10.5281/zenodo.19703699)

Zenodo: https://doi.org/10.5281/zenodo.19703699

---

## Installation

**1. Create and activate the conda environment:**

```bash
conda create -n coral-seg python=3.10
conda activate coral-seg
```

**2. Install PyTorch with CUDA support:**

```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

> For a different CUDA version or CPU-only install, use the command selector at the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

**3. Install the remaining dependencies:**

```bash
pip install -r requirements.txt
```

The base model weights (`yolo11x-seg.pt`) are downloaded automatically by Ultralytics on first run — no manual download needed.

---

## Dataset Structure

Datasets are expected to come pre-split (e.g. exported from Roboflow). The augmentation script expects the following layout:

```
/path/to/MyDataset/
├── img_dir/
│   ├── train/   ← JPEG images
│   ├── val/
│   └── test/
└── ann_dir/
    ├── train/   ← PNG segmentation masks (same stem as image, suffix _mask.png)
    ├── val/
    └── test/
```

The augmentation script writes output to `/path/to/MyDataset_Aug/` with the same layout.

---

## Pipeline

### Step 1 — Data Augmentation

Expands the training set using Mixup, CutMix, Mosaic, and Albumentations transforms (15× per image by default).

```bash
python Scripts/data_augmentation.py \
    --dataset_dir /path/to/datasets \
    --datasets MyDataset \
    --splits train val \
    --n_fold 15
```

| Argument | Default | Description |
|---|---|---|
| `--dataset_dir` | required | Parent directory containing dataset folders |
| `--datasets` | required | One or more dataset folder names |
| `--splits` | `train val` | Which splits to augment |
| `--n_fold` | `15` | Augmentation iterations per image |

---

### Step 2 — Training

Fine-tune YOLOv11x-seg on the augmented dataset. After training, automatically runs validation on the val and test splits.

**1. Copy and edit the dataset config:**

```bash
cp configs/data_example_seg.yaml configs/data_mycoral_seg.yaml
```

Edit `configs/data_mycoral_seg.yaml`:

```yaml
path: /path/to/MyDataset_Aug
train: img_dir/train
val:   img_dir/val
test:  img_dir/test
nc: 1
names: ['mycoral']
```

**2. Run training:**

```bash
python Scripts/train.py --config configs/train_config.yaml --data configs/data_mycoral_seg.yaml
```

Override individual hyperparameters at the CLI:

```bash
python Scripts/train.py --config configs/train_config.yaml \
    --data configs/data_mycoral_seg.yaml \
    --epochs 200 --batch 2 --lr0 0.00005
```

Enable Weights & Biases logging:

```bash
python Scripts/train.py --config configs/train_config.yaml \
    --data configs/data_mycoral_seg.yaml \
    --wandb --wandb_project MyProject
```

Best weights are saved to `results/run/weights/best.pt` (configurable via `--project` and `--name`).

> Pre-trained weights are on Zenodo (see [Data availability](#data-availability)).

**A note on train/eval resolution.** `configs/train_config.yaml` trains at `imgsz: 1024` by default — that's a memory/speed choice, not a reporting choice. `train.py` always re-validates the final model on the test split at a *fixed* `imgsz=1920` regardless of the training resolution, because 1920×1080 is the native resolution of the source ROV imagery and it's the resolution every number in the paper and every deployment script in this pipeline is evaluated at. Expect (often large) differences between the metrics your training run logs live and the test-split numbers `train.py` prints at the end — that's the resolution difference, not an inconsistency, and it's the number you should trust. If you're training on imagery with a different native resolution, change the fixed `imgsz=1920` in the final test validation call in `Scripts/train.py` to match, and evaluate downstream (Steps 3 and 4) at that same resolution too.

For a thin/wiry morphology in our data (whip corals), training and evaluating at mismatched resolutions cost several points of every metric — re-sweeping at a resolution closer to the eval resolution recovered them. If a class in your data is much smaller or thinner than the rest, and its metrics lag the others, this mismatch is worth checking before anything else.

**Hyperparameter sweeps.** The results in the paper come from a Bayesian hyperparameter sweep (with hyperband early-termination) over each coral group's training run, using Weights & Biases — not a single `train.py` call. `configs/sweep_config_example.yaml` is a genericized version of the actual sweep configs used for every group in this study — copy it, point its `command` block's `--data` at your dataset config, and edit `project`/`entity`, then:

```bash
wandb sweep configs/sweep_config_example.yaml     # prints a sweep ID
wandb agent <entity>/<project>/<sweep_id>          # launches trials; run several in parallel to speed up the search
```

Each finished (or simply checkpointed) trial becomes a candidate for Step 2.5 below. See the comments inside the config for how sweep-selected hyperparameters actually reach `train.py` (via `wandb.config` after `wandb.init()`, not CLI flags), and the [W&B sweeps docs](https://docs.wandb.ai/guides/sweeps/) for the general config format.

---

### Step 2.5 — Model Soup

A hyperparameter sweep produces many decent checkpoints of varying quality — every model reported in the paper is a *greedy model soup* (Wortsman et al., 2022) built from several of a sweep's checkpoints, not a single training run picked by best validation score. This is the step that actually produces the final model, and empirically it beat every individual candidate on the held-out test set for every coral group in this study — never skip it in favor of just taking the top sweep run.

```bash
python Scripts/model_soup.py \
    --data configs/data_mycoral_seg.yaml \
    --candidates_dir results/sweep_runs \
    --imgsz 1920 \
    --output_dir results/model_soup \
    --output_name mycoral_greedy_soup
```

`--candidates_dir` should contain one subfolder per sweep run, each with `weights/best.pt` (the layout `train.py` produces under `--project`). Alternatively pass explicit checkpoint paths with `--candidates ckpt1.pt ckpt2.pt ...`.

The script re-validates every candidate on the **validation** split at `--imgsz`, ranks them by F1, then greedily tries adding each one (weight-averaged into the running soup) in rank order, keeping it only if validation F1 doesn't drop. The **test** split is touched exactly once, at the very end, on the final soup — see the module docstring in `Scripts/model_soup.py` for the full method and practical notes (including when widening the candidate pool is and isn't worth trying, and why you should rank by re-validating yourself rather than trusting a live sweep dashboard's logged numbers).

Output: `<output_name>.pt` (the soup — load directly with `YOLO(path)`) and `<output_name>_state_dict.pt` (raw weights only) in `--output_dir`.

---

### Step 3 — Image Inference

Runs YOLO inference on a folder of still images, saves annotated copies, and writes per-image coral counts and depth (via OCR) to an Excel file.

```bash
python Scripts/inference_images.py \
    --images Image_example/Whipcoral \
    --model results/run/weights/best.pt
```

To skip training entirely and run inference with the actual models from the manuscript, download the pretrained checkpoints from Zenodo (see [Data availability](#data-availability)) and point `--model` at one of them instead — e.g. `--model Weights/scleractinia/scleractinia_souped_weights.pt`. See [`Weights/README.md`](Weights/README.md) for all four groups' final soups, their ingredient checkpoints, and reported metrics.

| Argument | Default | Description |
|---|---|---|
| `--images` | required | Folder of input images |
| `--model` | required | Path to trained `.pt` weights |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.5` | IoU threshold (NMS) |
| `--imgsz` | `1920` | Inference resolution. Should match the resolution you evaluate/report at — see the note in Step 2. |
| `--area_threshold` | `0.15` | Reject a detection whose mask covers more than this fraction of the frame — filters out oversized/malformed masks |
| `--containment_threshold` | `0.7` | Reject a detection if more than this fraction of its area is already covered by a larger, higher-confidence detection (see note below) |
| `--tesseract_path` | system PATH | Path to `tesseract.exe` (Windows) |
| `--roi_depth` | calibrated | Depth OCR region: `CX CY W H` in relative coords |
| `--output_dir` | parent of `--images` | Output directory |
| `--excel_name` | `results.xlsx` | Output Excel filename |

**Why a containment filter in addition to standard NMS:** standard NMS compares box IoU, which fails when one predicted mask sits entirely inside a much larger one — the union area is dominated by the big box, so IoU stays low even though the two masks obviously overlap. In practice this showed up as small nonsense detections nested inside a correct large detection that NMS never removed no matter how the confidence/IoU thresholds were tuned (we swept IoU from 0.5 down to 0.2 and got identical detection counts). The fix is a second pass, after NMS, that operates on actual mask overlap rather than box IoU: if `containment_threshold` fraction of a smaller detection's mask area is covered by a larger, higher-confidence one, the smaller one is dropped.

---

### Step 4 — ROV Video Analysis

> The example video is on Zenodo (see [Data availability](#data-availability)),
> not in this repo.

Extracts frames at a fixed interval, runs YOLO detection, reads depth and heading from the video overlay via OCR, and computes laser-calibrated coral density (corals/m²). Outputs per-video Excel files, a combined file, and a summary sheet grouped by habitat and transect type.

**1. Copy and edit the video metadata config:**

```bash
cp configs/video_metadata_example.yaml configs/my_survey.yaml
```

Set `model_path` to a trained checkpoint — either your own `results/run/weights/best.pt` or, to reproduce the manuscript's results directly, one of the pretrained soups from Zenodo (see [Data availability](#data-availability) and [`Weights/README.md`](Weights/README.md)).

Edit `configs/my_survey.yaml` — every key is documented with inline comments inside the file, including `conf_threshold` (0.25), `iou_threshold` (0.5), `imgsz` (1920), `area_threshold_percent` (15.0), and `containment_threshold` (0.7). These carry the same values and reasoning as the `Scripts/inference_images.py` defaults above — see that section for why the containment filter exists, and Step 2 for why `imgsz` matters as much as it does.

**2. Run analysis:**

```bash
python Scripts/automatic_detection.py --config configs/my_survey.yaml
```

Output structure:

```
results/rov_output/
├── CHR0389/
│   ├── frames/                  ← annotated frame images
│   └── CHR0389_results.xlsx
├── combined_results.xlsx
└── summary_analysis.xlsx        ← density by habitat, transect, dive
```

---

## Laser Calibration

Coral density is estimated from the visible seafloor area per frame, derived from two parallel laser dots projected onto the substrate. Per-frame label files must be in YOLO bounding box format (one dot per line):

```
0  0.5236  0.4865  0.0024  0.0044
0  0.4761  0.4833  0.0024  0.0044
```

The pixel distance between the two dot centers is converted to a cm/pixel scale using `actual_laser_distance_cm`, then used to compute the visible area (m²) for each frame.

Laser labels for the example video are provided in `Video_example/Laser_CHR0389_LowerMesophotic/train/labels/`. To annotate your own video, export laser dot labels from Roboflow or any YOLO-format annotation tool.

---

## Troubleshooting & Lessons Learned

Practical issues we ran into deploying these models on real ROV footage, and how we diagnosed/fixed each one — useful if you hit similar symptoms adapting this pipeline to your own data.

**Duplicate/nested boxes on the same coral.** Standard NMS (`--iou`) compares bounding-box IoU, which can't catch a small predicted mask sitting entirely inside a much larger one — the union area is dominated by the big box so IoU stays low regardless of how much the masks actually overlap. Lowering `--iou` doesn't fix it (we tested 0.5 → 0.3 → 0.2 and got identical detection counts). Fixed by `remove_contained_duplicates()` in `Scripts/inference_images.py` / `Scripts/automatic_detection.py`, a second pass done on real mask overlap after NMS (`--containment_threshold`, default 0.7).

**Oversized, nonsensical masks.** A single detection that swallows most of the frame is almost always a segmentation failure, not a giant coral. `--area_threshold` (default 0.15, i.e. 15% of frame area) filters these out before the containment pass runs.

**False positives between visually similar coral groups.** Because each group is a separate single-class detector, a detector never sees an example of the other groups labeled *not this class* during training — it has no reason to learn "this looks like Octocoral, not Whip coral," only "this does/doesn't look like Whip coral in isolation." When two groups share coarse visual structure (e.g. thin branching morphology), this shows up as one detector confidently firing on the other group's true examples. We experimented with cross-group hard-negative fine-tuning (continuing training with a low learning rate on images that contain *only* the confusable other group, with zero positive labels for the class being trained, so the model learns to suppress them) and it is a real technique worth trying if you hit this. Two things to watch if you do:
  - **Full fine-tunes can crash the model's recall.** We saw one attempt collapse recall from 0.44 to ~0.13 over 11 epochs (the model learned to predict almost nothing rather than to discriminate) using `lr0=0.0005` over the full network for 15 epochs. Freezing the backbone (`freeze=10` in Ultralytics), dropping `lr0` an order of magnitude (`0.0001`), and cutting to ~8 epochs avoided this — but still monitor the recall curve epoch-by-epoch and stop early if it drops.
  - Weigh whether the fine-tune is worth keeping at all: in this study, hard-negative fine-tuning measurably reduced cross-group confusion for one group but at a cost to other metrics we weren't willing to accept, so the plain model soup (no hard-negative fine-tuning) shipped as the official model instead. Confusion-matrix inspection on the *test* split, not just eyeballing predictions, is what should decide this.

**A group's metrics look weak and you're not sure if it's real.** Before assuming a model is bad, check whether the gap could just be test-set sampling noise — bootstrap-resample the test predictions (sample-with-replacement, recompute the metric, repeat ~1000x) and look at the resulting confidence interval. For our smallest test split (52 images), the 95% CI on some metrics spanned ~15-20 points — wide enough that a single point estimate is not a reliable signal of "this model is worse." If the CI is wide, more data (or a resolution-matched sweep, see Step 2's note) is a better lever to pull than further hyperparameter tuning.

**Numbers from your experiment tracker don't match what you get re-validating a saved checkpoint.** This is almost always the train/eval resolution mismatch described in Step 2 — a tracker logs metrics computed at the training `imgsz`, which is frequently lower than your actual reporting resolution. Always re-validate the checkpoint yourself, at your real reporting resolution, before trusting a number.

---

## Citation

> Paper under review. Citation will be added upon publication.
