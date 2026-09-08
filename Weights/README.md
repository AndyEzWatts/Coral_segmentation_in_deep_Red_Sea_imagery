# Pretrained Weights

Ready-to-use checkpoints for all four coral groups, so you can run Steps 3
and 4 of the pipeline (image inference / ROV video analysis) without
re-running the full training + sweep + soup pipeline yourself.

For each group:

- `<group>_souped_weights.pt` — **the final model** (a greedy model soup,
  see `Scripts/model_soup.py`). This is the checkpoint used for every
  result reported in the manuscript, and the one you should point
  `--model` / `model_path` at for Steps 3 and 4.
- `<group>_model{1,2,3,4}_weights.pt` — the individual sweep-run
  checkpoints that went into that soup (3 for scleractinia, 4 for the
  other three groups). Provided so the soup-building process in Step 2.5
  is fully reproducible from these files alone, and so you can inspect or
  re-validate any single ingredient on its own. These are not meant to be
  used for inference directly — the soup outperforms every one of them
  individually on the held-out test set.

```python
from ultralytics import YOLO
model = YOLO("Weights/scleractinia/scleractinia_souped_weights.pt")
results = model.predict("path/to/image.jpg", conf=0.25, imgsz=1920)
```

## Ingredient mapping

| Group | Ingredients (in `model1..N`, in this order) |
|---|---|
| scleractinia | light-sweep-113, different-sweep-596, volcanic-sweep-502 |
| octocoral | woven-sweep-88, glorious-sweep-86, frosty-sweep-52, lunar-sweep-21 |
| antipatharia | stellar-sweep-111, sweepy-sweep-235, worthy-sweep-118, fast-sweep-20 |
| whipcoral | wise-sweep-168, vivid-sweep-188, mild-sweep-229, smooth-sweep-230 |

These are the exact W&B sweep run names the checkpoints came from — kept
as a reference so the soup composition can be independently reproduced
with `Scripts/model_soup.py --candidates <model1.pt> <model2.pt> ...` if
you want to verify the greedy-selection outcome yourself, or rebuild the
soup after fine-tuning any one ingredient further.

## Final test-set metrics (souped weights, imgsz=1920)

| Group | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| scleractinia | 72.6% | 65.5% | 72.1% | 46.8% |
| octocoral | 74.7% | 55.7% | 65.7% | 37.3% |
| antipatharia | 65.9% | 61.5% | 60.7% | 32.7% |
| whipcoral | 51.7% | 46.2% | 43.5% | 16.6% |

These match the manuscript's reported numbers (Table 1) on or above every
metric for scleractinia, octocoral, and antipatharia. Whipcoral is the
weakest-performing group in the study — its test split is small (52
images), so its metrics carry more sampling noise than the others', and
its thin/wiry morphology is disproportionately sensitive to train/eval
resolution mismatch (see the note in the main README's Step 2). An
alternative 12-ingredient soup built from a resolution-matched sweep beat
these numbers on all four metrics in internal testing, but is not
included here — the 4-ingredient soup above is kept as the reported model
to match the manuscript's methodology exactly across all four groups. If
you're extending this work specifically on whip-coral-like thin
morphologies, training and evaluating at matched (ideally higher)
resolution is the first thing worth trying.

## Regenerating from scratch

To rebuild any of these from the ingredient checkpoints (or after
retraining ingredients yourself):

```bash
python Scripts/model_soup.py \
    --data configs/data_mycoral_seg.yaml \
    --candidates Weights/scleractinia/scleractinia_model1_weights.pt \
                 Weights/scleractinia/scleractinia_model2_weights.pt \
                 Weights/scleractinia/scleractinia_model3_weights.pt \
    --imgsz 1920 \
    --output_dir Weights/scleractinia \
    --output_name scleractinia_souped_weights
```
