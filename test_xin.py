import argparse
import csv
import glob
import os
import time

import numpy as np
if "bool" not in np.__dict__:
    np.bool = bool
import torch
import torch.nn as nn
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism

from dataset.dataset import get_loader_brats
from light_training.evaluation.metric import dice, hausdorff_distance_95
from train_xin import DEFAULT_DATA_DIR, DiffUNet, ROI_SIZE


DEFAULT_CHECKPOINT = "./logs_xin/fold0/model/best_model.pt"


def resolve_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    search_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    if not search_dir:
        search_dir = "."

    candidates = []
    for pattern in ("best_model*.pt", "final_model*.pt", "*.pt"):
        candidates.extend(glob.glob(os.path.join(search_dir, pattern)))
    candidates = sorted(set(candidates), key=os.path.getmtime, reverse=True)
    if candidates:
        return candidates[0]
    return checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test DiffUNet with Dice, HD95, inference time, params and FLOPs."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-fold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--infer-ddim-steps", type=int, default=10)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--voxel-spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--output-csv", default="test_xin_results.csv")
    parser.add_argument("--skip-flops", action="store_true")
    return parser.parse_args()


def strip_prefix(state_dict, prefix):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def candidate_state_dicts(state_dict):
    module_stripped = strip_prefix(state_dict, "module.")
    candidates = [module_stripped]
    candidates.append(strip_prefix(module_stripped, "model."))
    return candidates


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = resolve_checkpoint(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]

    best = None
    for state_dict in candidate_state_dicts(checkpoint):
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        score = len(missing) + len(unexpected)
        if best is None or score < best[0]:
            best = (score, missing, unexpected, state_dict)

    _, missing, unexpected, state_dict = best
    model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {len(unexpected)}")
    return checkpoint_path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FlopsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image=image, pred_type="ddim_sample")


def format_number(value):
    if value is None or np.isnan(value):
        return "N/A"
    if value >= 1e12:
        return f"{value / 1e12:.3f} T"
    if value >= 1e9:
        return f"{value / 1e9:.3f} G"
    if value >= 1e6:
        return f"{value / 1e6:.3f} M"
    return f"{value:.0f}"


def profile_flops(model, device):
    dummy = torch.randn(1, 4, *ROI_SIZE, device=device)
    wrapper = FlopsWrapper(model).eval()
    try:
        from thop import profile

        with torch.no_grad():
            flops, _ = profile(wrapper, inputs=(dummy,), verbose=False)
        return float(flops)
    except Exception as thop_error:
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.no_grad():
                with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                    wrapper(dummy)
            flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
            return float(flops) if flops > 0 else float("nan")
        except Exception as profiler_error:
            print(f"[Warning] FLOPs profiling failed: {thop_error}; {profiler_error}")
            return float("nan")


def case_metrics(pred, target, voxel_spacing):
    names = {"WT": 1, "TC": 0, "ET": 2}
    values = {}
    for name, channel in names.items():
        pred_mask = pred[channel].astype(np.uint8)
        target_mask = target[channel].astype(np.uint8)
        values[f"{name}_dice"] = dice(pred_mask, target_mask)
        values[f"{name}_hd95"] = hausdorff_distance_95(
            pred_mask,
            target_mask,
            voxel_spacing=voxel_spacing,
        )
    return values


def synchronize(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def main():
    args = parse_args()
    set_determinism(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _, _, test_dataset = get_loader_brats(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        fold=args.fold,
        num_workers=args.num_workers,
        seed=args.seed,
        n_splits=args.n_fold,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DiffUNet(infer_ddim_steps=args.infer_ddim_steps).to(device)
    checkpoint_path = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    window_infer = SlidingWindowInferer(
        roi_size=ROI_SIZE,
        sw_batch_size=1,
        overlap=args.overlap,
    )

    params = count_parameters(model)
    flops = None if args.skip_flops else profile_flops(model, device)

    rows = []
    infer_times = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image = batch["image"].to(device)
            label = batch["label"].float().to(device)

            synchronize(device)
            start = time.perf_counter()
            output = window_infer(image, model, pred_type="ddim_sample")
            synchronize(device)
            elapsed = time.perf_counter() - start

            pred = (torch.sigmoid(output) > args.threshold).float().cpu().numpy()[0]
            target = label.cpu().numpy()[0]
            metrics = case_metrics(pred, target, tuple(args.voxel_spacing))

            row = {"case": idx, "infer_time_sec": elapsed}
            row.update(metrics)
            rows.append(row)
            infer_times.append(elapsed)

            print(
                f"[{idx + 1}/{len(test_loader)}] "
                f"WT Dice={metrics['WT_dice']:.4f}, TC Dice={metrics['TC_dice']:.4f}, "
                f"ET Dice={metrics['ET_dice']:.4f}, "
                f"WT HD95={metrics['WT_hd95']:.4f}, TC HD95={metrics['TC_hd95']:.4f}, "
                f"ET HD95={metrics['ET_hd95']:.4f}, time={elapsed:.4f}s"
            )

    metric_keys = [
        "WT_dice",
        "TC_dice",
        "ET_dice",
        "WT_hd95",
        "TC_hd95",
        "ET_hd95",
    ]
    summary = {key: float(np.nanmean([row[key] for row in rows])) for key in metric_keys}
    summary["mean_dice"] = float(np.nanmean([summary["WT_dice"], summary["TC_dice"], summary["ET_dice"]]))
    summary["mean_hd95"] = float(np.nanmean([summary["WT_hd95"], summary["TC_hd95"], summary["ET_hd95"]]))
    summary["mean_infer_time_sec"] = float(np.mean(infer_times))

    if args.output_csv:
        fieldnames = ["case", "infer_time_sec"] + metric_keys
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("=" * 72)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test cases: {len(rows)}")
    print(
        f"Dice  WT={summary['WT_dice']:.4f}, TC={summary['TC_dice']:.4f}, "
        f"ET={summary['ET_dice']:.4f}, Mean={summary['mean_dice']:.4f}"
    )
    print(
        f"HD95  WT={summary['WT_hd95']:.4f}, TC={summary['TC_hd95']:.4f}, "
        f"ET={summary['ET_hd95']:.4f}, Mean={summary['mean_hd95']:.4f}"
    )
    print(f"Single-sample inference time: {summary['mean_infer_time_sec']:.4f} s")
    print(f"Parameters: {params:,} ({params / 1e6:.3f} M)")
    print(f"FLOPs per {ROI_SIZE} DDIM inference: {format_number(flops)}")
    if args.output_csv:
        print(f"Per-case CSV: {args.output_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()




