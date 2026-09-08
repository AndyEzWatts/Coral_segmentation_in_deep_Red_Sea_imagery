"""
Greedy model soup builder (Wortsman et al., 2022) for YOLO-seg checkpoints.

This is the step referenced in the paper as "for each coral group, we
built a model soup" -- it is what actually combines several trained runs
from a hyperparameter sweep into one final model, and it's the reason the
reported models outperform any single training run.

## What this does

A hyperparameter sweep produces many trained checkpoints of varying
quality. Rather than pick the single best one, or blindly average a fixed
set of them, this script:

  1. Re-validates every candidate checkpoint individually on the
     VALIDATION split, and ranks them best-to-worst by F1
     (precision/recall balance).
  2. Starts the "soup" as just the single best candidate.
  3. Goes down the ranked list one candidate at a time: tentatively
     averages its weights (parameter-wise mean, matching layers) into
     the current soup, re-validates the averaged model on VALIDATION,
     and keeps the candidate only if validation F1 does not drop.
     Otherwise the candidate is discarded and the soup is unchanged.
  4. Once every candidate has been considered, the TEST split is
     evaluated exactly once, on the final soup -- this is the only time
     test data is touched, so it can't leak into which candidates got
     selected.

## Usage

    python Scripts/model_soup.py \\
        --data configs/data_mycoral_seg.yaml \\
        --candidates run1/weights/best.pt run2/weights/best.pt run3/weights/best.pt \\
        --imgsz 1920 \\
        --output_dir results/model_soup

Or point at a directory of candidate run folders instead of listing each
checkpoint by hand:

    python Scripts/model_soup.py \\
        --data configs/data_mycoral_seg.yaml \\
        --candidates_dir results/sweep_runs \\
        --imgsz 1920 \\
        --output_dir results/model_soup
"""
import argparse
import os
from collections import OrderedDict

import torch
from ultralytics import YOLO


def load_state_dict(path):
    return YOLO(path).model.cpu().state_dict()


def average_state_dicts(state_dicts_list):
    """Uniform (equal-weight) parameter-wise mean across all given state_dicts."""
    ref = state_dicts_list[0]
    out = OrderedDict()
    for key in ref.keys():
        tensors = [sd[key] for sd in state_dicts_list]
        if ref[key].is_floating_point():
            out[key] = torch.mean(torch.stack([t.float() for t in tensors], dim=0), dim=0).type(ref[key].dtype)
        else:
            # Non-float buffers (e.g. BatchNorm's num_batches_tracked) aren't
            # meaningfully "averaged" -- just keep the reference value.
            out[key] = ref[key].clone()
    return out


def build_model_from_state_dict(sd, arch_ckpt_path, device):
    """arch_ckpt_path just supplies the model architecture/config; its weights are overwritten by sd."""
    m = YOLO(arch_ckpt_path)
    m.model.load_state_dict(sd, strict=True)
    m.to(device)
    return m


def val_f1(model, data_yaml, split, imgsz, batch, device, out_dir, run_tag):
    r = model.val(
        data=data_yaml,
        split=split,
        batch=batch,
        imgsz=imgsz,
        device=device,
        verbose=False,
        plots=False,
        project=os.path.join(out_dir, "val_scratch"),
        name=run_tag,
        exist_ok=True,
    )
    p, rc = r.seg.mp, r.seg.mr
    f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0
    return f1, p, rc, r.seg.map50, r.seg.map


def discover_candidates(candidates_dir):
    """Find <run>/weights/best.pt under a directory of sweep-run folders."""
    found = {}
    for name in sorted(os.listdir(candidates_dir)):
        ckpt = os.path.join(candidates_dir, name, "weights", "best.pt")
        if os.path.isfile(ckpt):
            found[name] = ckpt
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Greedy model soup builder for YOLO-seg checkpoints (Wortsman et al., 2022).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to YOLO dataset YAML")
    parser.add_argument("--candidates", nargs="+", default=None,
                        help="Explicit list of candidate checkpoint (.pt) paths")
    parser.add_argument("--candidates_dir", default=None,
                        help="Directory of run folders, each containing weights/best.pt "
                             "(alternative to --candidates; every run with a usable checkpoint is used)")
    parser.add_argument("--imgsz", type=int, default=1920,
                        help="Resolution for both the ranking/selection validation and the final "
                             "test evaluation. Use your actual reporting resolution here, not the "
                             "training resolution, if the two differ.")
    parser.add_argument("--batch", type=int, default=2, help="Validation batch size")
    parser.add_argument("--output_dir", required=True, help="Where to save the soup checkpoint and logs")
    parser.add_argument("--output_name", default="model_soup", help="Base filename for the saved soup")
    args = parser.parse_args()

    if not args.candidates and not args.candidates_dir:
        parser.error("Provide either --candidates or --candidates_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.candidates_dir:
        candidate_paths = discover_candidates(args.candidates_dir)
        print(f"Discovered {len(candidate_paths)} candidates with a saved checkpoint under {args.candidates_dir}", flush=True)
    else:
        candidate_paths = {os.path.splitext(os.path.basename(p))[0] + f"_{i}": p
                            for i, p in enumerate(args.candidates)}

    if len(candidate_paths) < 2:
        raise SystemExit(f"Need at least 2 usable candidates, found {len(candidate_paths)}.")

    candidate_names = list(candidate_paths.keys())

    print("Loading candidate state_dicts...", flush=True)
    state_dicts = {}
    for name in candidate_names:
        try:
            state_dicts[name] = load_state_dict(candidate_paths[name])
        except Exception as e:
            print(f"WARNING: failed to load {name} ({candidate_paths[name]}): {e} -- skipping", flush=True)
    candidate_names = [n for n in candidate_names if n in state_dicts]
    print(f"Loaded {len(state_dicts)} checkpoints.", flush=True)

    # Step 1: individual val performance, to establish rank order.
    print("\n=== Individual candidate val-split performance ===", flush=True)
    individual_scores = {}
    for name in candidate_names:
        m = build_model_from_state_dict(state_dicts[name], candidate_paths[name], device)
        f1, p, rc, m50, m5095 = val_f1(m, args.data, "val", args.imgsz, args.batch, device,
                                        args.output_dir, f"individual_{name}")
        individual_scores[name] = f1
        print(f"{name:30s} val_F1={f1:.4f}  P={p:.4f} R={rc:.4f} mAP50={m50:.4f} mAP50-95={m5095:.4f}", flush=True)
        del m

    ranked = sorted(candidate_names, key=lambda n: -individual_scores[n])
    print(f"\nRank order (best individual val F1 first): {ranked}", flush=True)

    # Step 2+3: greedy inclusion, gated on validation F1 only.
    soup_names = [ranked[0]]
    soup_sds = [state_dicts[ranked[0]]]
    current_soup_sd = soup_sds[0]
    m = build_model_from_state_dict(current_soup_sd, candidate_paths[ranked[0]], device)
    best_f1, p, rc, _, _ = val_f1(m, args.data, "val", args.imgsz, args.batch, device, args.output_dir, "soup_step0")
    print(f"\nGreedy soup start: [{ranked[0]}]  val_F1={best_f1:.4f} P={p:.4f} R={rc:.4f}", flush=True)
    del m

    for name in ranked[1:]:
        trial_sds = soup_sds + [state_dicts[name]]
        trial_sd = average_state_dicts(trial_sds)
        m = build_model_from_state_dict(trial_sd, candidate_paths[ranked[0]], device)
        trial_f1, p, rc, m50, m5095 = val_f1(m, args.data, "val", args.imgsz, args.batch, device,
                                              args.output_dir, f"soup_trial_{name}")
        print(f"Trying + {name}: val_F1={trial_f1:.4f} P={p:.4f} R={rc:.4f} mAP50={m50:.4f} mAP50-95={m5095:.4f}", flush=True)
        del m
        if trial_f1 >= best_f1:
            soup_names.append(name)
            soup_sds.append(state_dicts[name])
            current_soup_sd = trial_sd
            best_f1 = trial_f1
            print(f"  -> KEPT. Soup is now ({len(soup_names)} ingredients): {soup_names}", flush=True)
        else:
            print(f"  -> REJECTED (val F1 would drop from {best_f1:.4f} to {trial_f1:.4f}).", flush=True)

    print(f"\n=== Final greedy soup: {soup_names} ===", flush=True)

    final_model = build_model_from_state_dict(current_soup_sd, candidate_paths[ranked[0]], device)
    state_dict_path = os.path.join(args.output_dir, f"{args.output_name}_state_dict.pt")
    full_ckpt_path = os.path.join(args.output_dir, f"{args.output_name}.pt")
    torch.save(current_soup_sd, state_dict_path)
    final_model.save(full_ckpt_path)
    print(f"Saved: {full_ckpt_path} (full checkpoint -- use this one, loads directly via YOLO(path))", flush=True)
    print(f"Saved: {state_dict_path} (raw state_dict only -- not directly loadable via YOLO(path))", flush=True)

    print("\nRunning FINAL test-set validation on the greedy soup (touched exactly once)...", flush=True)
    test_results = final_model.val(
        data=args.data,
        split="test",
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        verbose=True,
        project=os.path.join(args.output_dir, "val"),
        name=f"{args.output_name}_test",
    )
    print("\n=== FINAL soup test-set metrics ===", flush=True)
    print(f"Ingredients ({len(soup_names)}): {soup_names}", flush=True)
    print(f"Precision(M): {test_results.seg.mp:.4f}", flush=True)
    print(f"Recall(M):    {test_results.seg.mr:.4f}", flush=True)
    print(f"mAP50(M):     {test_results.seg.map50:.4f}", flush=True)
    print(f"mAP50-95(M):  {test_results.seg.map:.4f}", flush=True)


if __name__ == "__main__":
    main()
