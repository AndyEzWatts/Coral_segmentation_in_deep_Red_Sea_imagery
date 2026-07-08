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
  for inference (Steps 2–4).
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

> Pre-trained weights are on Zenodo (see [Data availability]).

---

### Step 3 — Image Inference

Runs YOLO inference on a folder of still images, saves annotated copies, and writes per-image coral counts and depth (via OCR) to an Excel file.

```bash
python Scripts/inference_images.py \
    --images Image_example/Whipcoral \
    --model results/run/weights/best.pt
```

| Argument | Default | Description |
|---|---|---|
| `--images` | required | Folder of input images |
| `--model` | required | Path to trained `.pt` weights |
| `--conf` | `0.1` | Confidence threshold |
| `--iou` | `0.0` | IoU threshold |
| `--area_threshold` | `0.15` | Reject detections covering > this fraction of the frame |
| `--tesseract_path` | system PATH | Path to `tesseract.exe` (Windows) |
| `--roi_depth` | calibrated | Depth OCR region: `CX CY W H` in relative coords |
| `--output_dir` | parent of `--images` | Output directory |
| `--excel_name` | `results.xlsx` | Output Excel filename |

---

### Step 4 — ROV Video Analysis

> The example video is on Zenodo (see [Data availability]),
> not in this repo.

Extracts frames at a fixed interval, runs YOLO detection, reads depth and heading from the video overlay via OCR, and computes laser-calibrated coral density (corals/m²). Outputs per-video Excel files, a combined file, and a summary sheet grouped by habitat and transect type.

**1. Copy and edit the video metadata config:**

```bash
cp configs/video_metadata_example.yaml configs/my_survey.yaml
```

Edit `configs/my_survey.yaml` — every key is documented with inline comments inside the file.

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

## Citation

> Paper under review. Citation will be added upon publication.
