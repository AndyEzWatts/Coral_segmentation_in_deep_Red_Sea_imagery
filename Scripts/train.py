import argparse
import gc
import os

import torch
import yaml
from ultralytics import YOLO


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv11-seg model. YAML config sets defaults; CLI args override individual keys.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to train_config.yaml")

    # Overrides for common keys
    parser.add_argument("--model", help="Model name, e.g. yolo11x-seg")
    parser.add_argument("--data", help="Path to YOLO dataset YAML")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--lrf", type=float)
    parser.add_argument("--optimizer")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--single_cls", action="store_true", default=None)
    parser.add_argument("--rect", action="store_true", default=None)
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", help="W&B project name (overrides config)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides: any non-None CLI value wins over the YAML default
    for key in ["model", "data", "epochs", "batch", "imgsz", "lr0", "lrf",
                "optimizer", "patience", "workers", "fraction", "project", "name", "wandb_project"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if args.single_cls:
        cfg["single_cls"] = True
    if args.rect:
        cfg["rect"] = True

    gc.collect()
    torch.cuda.empty_cache()

    run_name = cfg.get("name", "run")
    wandb_run = None

    if args.wandb:
        import wandb as wb
        wandb_run = wb.init(
            project=cfg.get("wandb_project", "YOLO-Seg"),
            config=cfg,
            resume="allow",
            job_type="train",
        )
        run_name = wandb_run.name

    data_path = cfg["data"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Dataset: {data_path}")
    print(f"Model: {cfg['model']}")

    model = YOLO(f"{cfg['model']}.pt")
    model.info()

    print("Starting training...")
    model.train(
        data=data_path,
        project=cfg.get("project", "results"),
        name=run_name,
        epochs=cfg.get("epochs", 100),
        batch=cfg.get("batch", 4),
        imgsz=cfg.get("imgsz", 1024),
        optimizer=cfg.get("optimizer", "AdamW"),
        lr0=cfg.get("lr0", 0.00001),
        lrf=cfg.get("lrf", 0.001),
        patience=cfg.get("patience", 90),
        cos_lr=cfg.get("cos_lr", True),
        fraction=cfg.get("fraction", 1.0),
        workers=cfg.get("workers", 4),
        pretrained=cfg.get("pretrained", True),
        label_smoothing=cfg.get("label_smoothing", 0.0),
        dropout=cfg.get("dropout", 0.0),
        warmup_epochs=cfg.get("warmup_epochs", 0),
        augment=cfg.get("augment", True),
        mosaic=cfg.get("mosaic", 0.0),
        scale=cfg.get("scale", 0.5),
        translate=cfg.get("translate", 0.1),
        degrees=cfg.get("degrees", 0.0),
        flipud=cfg.get("flipud", 0.0),
        fliplr=cfg.get("fliplr", 0.5),
        mixup=cfg.get("mixup", 0.0),
        hsv_h=cfg.get("hsv_h", 0.01),
        hsv_s=cfg.get("hsv_s", 0.6),
        hsv_v=cfg.get("hsv_v", 0.6),
        single_cls=cfg.get("single_cls", False),
        overlap_mask=cfg.get("overlap_mask", False),
        rect=cfg.get("rect", False),
        device=device,
        verbose=True,
    )
    print("Training complete.")

    best_weights = os.path.join(cfg.get("project", "results"), run_name, "weights", "best.pt")
    if not os.path.exists(best_weights):
        print(f"Warning: best.pt not found at {best_weights}. Skipping val/test runs.")
    else:
        model_best = YOLO(best_weights)

        print("Running validation (val split)...")
        model_best.val(
            data=data_path,
            split="val",
            imgsz=cfg.get("imgsz", 1024),
            batch=cfg.get("batch", 4),
            device=device,
            verbose=True,
        )

        print("Running validation (test split)...")
        model_best.val(
            data=data_path,
            split="test",
            imgsz=cfg.get("imgsz", 1024),
            batch=1,
            device=device,
            verbose=True,
        )

    if wandb_run:
        import wandb as wb
        wb.finish()

    print(f"Done. Best weights: {best_weights}")


if __name__ == "__main__":
    main()
