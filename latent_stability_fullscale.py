"""
Latent-Stability-Guided Prompt Pruning for SAM — Full-Scale with Calibration
=============================================================================
Builds on the pilot with four calibration strategies:
  1. Overlap filtering  — exclude IoU=0 prompts from means
  2. Stability-weighted — weight IoU by stability score
  3. Adaptive threshold — per-image median instead of fixed 0.7
  4. Near/far breakdown — separate analysis for GT-near vs far prompts

Usage:
    python latent_stability_fullscale.py --datasets busi_malignant
    python latent_stability_fullscale.py --datasets busi_malignant,busi_benign,kvasir
    python latent_stability_fullscale.py --datasets all
"""

import os, argparse, json, warnings, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# DATASET CONFIGS
# ═══════════════════════════════════════════════════════════════

BASE = "/home/mi3dr/SCCS/sccs"

DATASETS = {
    "busi_malignant": {
        "image_dir": f"{BASE}/Dataset_BUSI_with_GT/malignant",
        "mask_dir":  f"{BASE}/Dataset_BUSI_with_GT/malignant",
    },
    "busi_benign": {
        "image_dir": f"{BASE}/Dataset_BUSI_with_GT/benign",
        "mask_dir":  f"{BASE}/Dataset_BUSI_with_GT/benign",
    },
    "kvasir": {
        "image_dir": f"{BASE}/Kvasir-SEG/images",
        "mask_dir":  f"{BASE}/Kvasir-SEG/masks",
    },
}


# ═══════════════════════════════════════════════════════════════
# 1. GT-INFORMED PROMPT GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_grid_points(h: int, w: int, grid_size: int = 4) -> np.ndarray:
    """Uniform grid points (used for 'far' / background prompts)."""
    xs = np.linspace(0, w - 1, grid_size + 2)[1:-1]
    ys = np.linspace(0, h - 1, grid_size + 2)[1:-1]
    return np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)


def jitter_around_center(center, h, w, n=18, spread=0.15):
    """Generate n points scattered around a center (x, y) with Gaussian spread."""
    cx, cy = center
    points = []
    for _ in range(n):
        px = np.clip(cx + np.random.normal(0, spread * w), 0, w - 1)
        py = np.clip(cy + np.random.normal(0, spread * h), 0, h - 1)
        points.append([px, py])
    return np.array(points)


def points_to_boxes(points, h, w, gt_mask=None, scale_range=(0.15, 0.4)):
    """Convert center points to box prompts. If gt_mask is provided,
    scale boxes relative to GT bounding box; otherwise use random scale."""
    if gt_mask is not None and gt_mask.any():
        ys_gt, xs_gt = np.where(gt_mask)
        gt_w = xs_gt.max() - xs_gt.min()
        gt_h = ys_gt.max() - ys_gt.min()
    else:
        gt_w, gt_h = None, None

    boxes = []
    for cx, cy in points:
        if gt_w is not None:
            s = np.random.uniform(0.8, 1.5)
            bw, bh = gt_w * s, gt_h * s
        else:
            s = np.random.uniform(*scale_range)
            bw, bh = w * s, h * s
        boxes.append([
            max(0, cx - bw / 2), max(0, cy - bh / 2),
            min(w - 1, cx + bw / 2), min(h - 1, cy + bh / 2)
        ])
    return np.array(boxes)


def generate_gt_informed_prompts(h, w, gt_mask, n_near=18, n_far_grid=4):
    """Generate prompts: ~50% near GT centroid, ~50% far/grid.
    Returns dict with points, boxes, and a boolean mask 'is_near'."""
    gt_coords = np.argwhere(gt_mask)
    if len(gt_coords) == 0:
        far_pts = generate_grid_points(h, w, n_far_grid)
        boxes = points_to_boxes(far_pts, h, w)
        return {"points": far_pts, "boxes": boxes,
                "is_near": np.zeros(len(far_pts), dtype=bool)}

    gt_center = gt_coords.mean(axis=0)[::-1]  # (x, y)

    near_pts = jitter_around_center(gt_center, h, w, n=n_near, spread=0.15)
    far_pts  = generate_grid_points(h, w, grid_size=n_far_grid)

    all_pts = np.vstack([near_pts, far_pts])
    is_near = np.array([True] * len(near_pts) + [False] * len(far_pts))

    near_boxes = points_to_boxes(near_pts, h, w, gt_mask=gt_mask)
    far_boxes  = points_to_boxes(far_pts, h, w)
    all_boxes  = np.vstack([near_boxes, far_boxes])

    return {"points": all_pts, "boxes": all_boxes, "is_near": is_near}


# ═══════════════════════════════════════════════════════════════
# 2. JITTER (THE STRESS TEST)
# ═══════════════════════════════════════════════════════════════

def apply_prompt_jitter(prompt, jitter_frac=0.10, h=1024, w=1024, is_box=False):
    p = prompt.copy().astype(np.float64)
    if is_box:
        bw, bh = p[2] - p[0], p[3] - p[1]
        jx = np.random.uniform(-jitter_frac, jitter_frac) * bw
        jy = np.random.uniform(-jitter_frac, jitter_frac) * bh
        p[0] = np.clip(p[0]+jx, 0, w-1); p[1] = np.clip(p[1]+jy, 0, h-1)
        p[2] = np.clip(p[2]+jx, 0, w-1); p[3] = np.clip(p[3]+jy, 0, h-1)
    else:
        jx = np.random.uniform(-jitter_frac, jitter_frac) * w
        jy = np.random.uniform(-jitter_frac, jitter_frac) * h
        p[0] = np.clip(p[0]+jx, 0, w-1); p[1] = np.clip(p[1]+jy, 0, h-1)
    return p


class LatentJitterHook:
    """Injects Gaussian noise into ViT encoder's last block output."""
    
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.active = False
        self._handle = None

    def hook_fn(self, module, input, output):
        if self.active:
            return output + torch.randn_like(output) * self.epsilon
        return output

    def register(self, encoder):
        target = getattr(encoder, 'blocks', getattr(encoder, 'layers', None))
        target = target[-1] if target is not None else encoder
        self._handle = target.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle: self._handle.remove()


# ═══════════════════════════════════════════════════════════════
# 3. MASK IoU
# ═══════════════════════════════════════════════════════════════

def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def pairwise_miou(masks):
    if len(masks) < 2: return 1.0
    ious = [mask_iou(masks[i], masks[j])
            for i in range(len(masks)) for j in range(i+1, len(masks))]
    return float(np.mean(ious))


def majority_vote(masks):
    return np.stack(masks).astype(np.float32).mean(0) >= 0.5


# ═══════════════════════════════════════════════════════════════
# 4. STABILITY SCORING PER PROMPT
# ═══════════════════════════════════════════════════════════════

def score_prompt_stability(predictor, image, prompt, is_box, hook, n_trials=5, jitter_frac=0.10):
    h, w = image.shape[:2]
    masks = []
    
    for _ in range(n_trials):
        jittered = apply_prompt_jitter(prompt, jitter_frac, h, w, is_box)
        hook.active = True
        predictor.set_image(image)
        
        if is_box:
            pred, _, _ = predictor.predict(box=jittered[None,:], multimask_output=False)
        else:
            pred, _, _ = predictor.predict(
                point_coords=jittered[None,:], point_labels=np.array([1]),
                multimask_output=False)
        
        hook.active = False
        masks.append(pred[0].astype(bool))
    
    return pairwise_miou(masks), masks


# ═══════════════════════════════════════════════════════════════
# 5. CALIBRATED ANALYSIS — THE KEY IMPROVEMENT
# ═══════════════════════════════════════════════════════════════

def _split_metrics(ious, scores, indices):
    """Helper: compute mean, max, weighted-mean for a subset of prompts."""
    if not indices:
        return {"mean": 0.0, "max": 0.0, "weighted": 0.0, "count": 0}
    sub_ious   = [ious[i] for i in indices]
    sub_scores = [scores[i] for i in indices]
    w_total = sum(sub_scores)
    w_iou   = sum(s * iou for s, iou in zip(sub_scores, sub_ious))
    return {
        "mean": float(np.mean(sub_ious)),
        "max":  float(np.max(sub_ious)),
        "weighted": float(w_iou / w_total) if w_total > 0 else 0.0,
        "count": len(indices),
    }


def analyze_calibrated(scores, all_masks, gt, is_near, fixed_threshold=0.7):
    """Run all four calibration strategies and return a unified result dict.
    
    Calibrations:
      1. uncalibrated      — raw fixed threshold, all prompts
      2. overlap_filtered  — exclude prompts with IoU == 0
      3. weighted          — stability-weighted IoU
      4. adaptive          — per-image median threshold
    Plus near/far breakdown.
    """
    n = len(scores)
    
    # Representative mask and IoU for each prompt
    per_prompt_ious = []
    for masks in all_masks:
        rep = majority_vote(masks)
        per_prompt_ious.append(mask_iou(rep, gt))
    
    scores_arr = np.array(scores)
    ious_arr   = np.array(per_prompt_ious)
    is_near_arr = np.array(is_near)
    
    # ── Thresholds ──
    fixed_stable  = [i for i in range(n) if scores[i] >= fixed_threshold]
    fixed_brittle = [i for i in range(n) if scores[i] <  fixed_threshold]
    
    adaptive_thresh = float(np.median(scores_arr)) if n > 0 else fixed_threshold
    adapt_stable  = [i for i in range(n) if scores[i] >= adaptive_thresh]
    adapt_brittle = [i for i in range(n) if scores[i] <  adaptive_thresh]
    
    # ── Overlap filter: exclude IoU==0 prompts ──
    has_overlap = [i for i in range(n) if per_prompt_ious[i] > 0]
    filt_stable  = [i for i in has_overlap if scores[i] >= fixed_threshold]
    filt_brittle = [i for i in has_overlap if scores[i] <  fixed_threshold]
    
    # ── Near/far split ──
    near_idx = [i for i in range(n) if is_near_arr[i]]
    far_idx  = [i for i in range(n) if not is_near_arr[i]]
    near_stable  = [i for i in near_idx if scores[i] >= fixed_threshold]
    near_brittle = [i for i in near_idx if scores[i] <  fixed_threshold]
    far_stable   = [i for i in far_idx  if scores[i] >= fixed_threshold]
    far_brittle  = [i for i in far_idx  if scores[i] <  fixed_threshold]
    
    # ── Best prompt ──
    best_idx = int(np.argmax(ious_arr))
    best_is_stable = best_idx in fixed_stable
    best_is_near   = bool(is_near_arr[best_idx])
    
    # ── Stability-weighted IoU (global) ──
    w_total = scores_arr.sum()
    global_weighted = float((scores_arr * ious_arr).sum() / w_total) if w_total > 0 else 0.0
    
    return {
        # Raw per-prompt data
        "per_prompt_ious": per_prompt_ious,
        "stability_scores": scores,
        "is_near": is_near.tolist() if hasattr(is_near, 'tolist') else list(is_near),
        
        # Best prompt
        "best_prompt_idx": best_idx,
        "best_prompt_iou": float(ious_arr[best_idx]),
        "best_is_stable": best_is_stable,
        "best_is_near": best_is_near,
        
        # 1. Uncalibrated (fixed threshold, all prompts)
        "uncalibrated": {
            "threshold": fixed_threshold,
            "stable": _split_metrics(per_prompt_ious, scores, fixed_stable),
            "brittle": _split_metrics(per_prompt_ious, scores, fixed_brittle),
            "gap": (_split_metrics(per_prompt_ious, scores, fixed_stable)["mean"] -
                    _split_metrics(per_prompt_ious, scores, fixed_brittle)["mean"]),
        },
        
        # 2. Overlap-filtered (exclude IoU=0)
        "overlap_filtered": {
            "n_with_overlap": len(has_overlap),
            "n_excluded": n - len(has_overlap),
            "stable": _split_metrics(per_prompt_ious, scores, filt_stable),
            "brittle": _split_metrics(per_prompt_ious, scores, filt_brittle),
            "gap": (_split_metrics(per_prompt_ious, scores, filt_stable)["mean"] -
                    _split_metrics(per_prompt_ious, scores, filt_brittle)["mean"]),
        },
        
        # 3. Stability-weighted
        "weighted": {
            "global_weighted_iou": global_weighted,
            "stable_weighted": _split_metrics(per_prompt_ious, scores, fixed_stable)["weighted"],
            "brittle_weighted": _split_metrics(per_prompt_ious, scores, fixed_brittle)["weighted"],
        },
        
        # 4. Adaptive threshold
        "adaptive": {
            "threshold": adaptive_thresh,
            "stable": _split_metrics(per_prompt_ious, scores, adapt_stable),
            "brittle": _split_metrics(per_prompt_ious, scores, adapt_brittle),
            "gap": (_split_metrics(per_prompt_ious, scores, adapt_stable)["mean"] -
                    _split_metrics(per_prompt_ious, scores, adapt_brittle)["mean"]),
        },
        
        # Near/far breakdown
        "near_far": {
            "near": {
                "total": len(near_idx),
                "stable": _split_metrics(per_prompt_ious, scores, near_stable),
                "brittle": _split_metrics(per_prompt_ious, scores, near_brittle),
            },
            "far": {
                "total": len(far_idx),
                "stable": _split_metrics(per_prompt_ious, scores, far_stable),
                "brittle": _split_metrics(per_prompt_ious, scores, far_brittle),
            },
        },
    }


# ═══════════════════════════════════════════════════════════════
# 6. FULL-SCALE RUNNER
# ═══════════════════════════════════════════════════════════════

def find_gt_mask(name, mask_dir, img_path):
    """Find ground truth mask file, preferring _mask variants."""
    candidates = [
        Path(mask_dir) / f"{name}_mask.png",
        Path(mask_dir) / f"{name}_mask.tif",
        Path(mask_dir) / f"{name}.png",
        Path(mask_dir) / f"{name}.jpg",
        Path(mask_dir) / f"{name}.tif",
    ]
    for p in candidates:
        if p.exists() and p != img_path:
            return p
    return None


def run_dataset(predictor, hook, dataset_name, image_dir, mask_dir, output_dir,
                n_near=18, n_far_grid=4, n_trials=5, fixed_threshold=0.7,
                prompt_jitter_frac=0.10, use_boxes=True):
    """Run the full experiment on one dataset."""
    
    image_paths = sorted([p for p in Path(image_dir).glob("*.png") if "_mask" not in p.stem]) + \
                  sorted([p for p in Path(image_dir).glob("*.jpg") if "_mask" not in p.stem])
    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset_name} — {len(image_paths)} images")
    print(f"{'='*70}")
    
    # ── Accumulators ──
    acc = {
        "uncal_gaps": [], "filt_gaps": [], "adapt_gaps": [],
        "uncal_stable_mean": [], "uncal_brittle_mean": [],
        "filt_stable_mean": [], "filt_brittle_mean": [],
        "adapt_stable_mean": [], "adapt_brittle_mean": [],
        "weighted_stable": [], "weighted_brittle": [],
        "max_stable": [], "max_brittle": [],
        "near_stable_mean": [], "near_brittle_mean": [],
        "far_stable_mean": [], "far_brittle_mean": [],
        "best_is_stable": 0, "best_is_near": 0,
    }
    results_per_image = []
    n_processed = 0
    t0 = time.time()
    
    for idx, img_path in enumerate(image_paths):
        name = img_path.stem
        gt_path = find_gt_mask(name, mask_dir, img_path)
        if not gt_path:
            print(f"  [Skip] No GT for {name}"); continue
        
        image = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("L")) > 127
        h, w = image.shape[:2]
        if gt.shape != (h, w):
            gt = np.array(Image.fromarray(gt.astype(np.uint8)*255).resize((w, h))) > 127
        
        if not gt.any():
            print(f"  [Skip] Empty GT for {name}"); continue
        
        prompts = generate_gt_informed_prompts(h, w, gt, n_near=n_near, n_far_grid=n_far_grid)
        prompt_arr = prompts["boxes"] if use_boxes else prompts["points"]
        is_near = prompts["is_near"]
        
        scores, all_masks = [], []
        for p in prompt_arr:
            s, m = score_prompt_stability(predictor, image, p, use_boxes, hook, n_trials, prompt_jitter_frac)
            scores.append(s); all_masks.append(m)
        
        analysis = analyze_calibrated(scores, all_masks, gt, is_near, fixed_threshold)
        
        # Accumulate
        acc["uncal_gaps"].append(analysis["uncalibrated"]["gap"])
        acc["filt_gaps"].append(analysis["overlap_filtered"]["gap"])
        acc["adapt_gaps"].append(analysis["adaptive"]["gap"])
        acc["uncal_stable_mean"].append(analysis["uncalibrated"]["stable"]["mean"])
        acc["uncal_brittle_mean"].append(analysis["uncalibrated"]["brittle"]["mean"])
        acc["filt_stable_mean"].append(analysis["overlap_filtered"]["stable"]["mean"])
        acc["filt_brittle_mean"].append(analysis["overlap_filtered"]["brittle"]["mean"])
        acc["adapt_stable_mean"].append(analysis["adaptive"]["stable"]["mean"])
        acc["adapt_brittle_mean"].append(analysis["adaptive"]["brittle"]["mean"])
        acc["weighted_stable"].append(analysis["weighted"]["stable_weighted"])
        acc["weighted_brittle"].append(analysis["weighted"]["brittle_weighted"])
        acc["max_stable"].append(analysis["uncalibrated"]["stable"]["max"])
        acc["max_brittle"].append(analysis["uncalibrated"]["brittle"]["max"])
        acc["near_stable_mean"].append(analysis["near_far"]["near"]["stable"]["mean"])
        acc["near_brittle_mean"].append(analysis["near_far"]["near"]["brittle"]["mean"])
        acc["far_stable_mean"].append(analysis["near_far"]["far"]["stable"]["mean"])
        acc["far_brittle_mean"].append(analysis["near_far"]["far"]["brittle"]["mean"])
        if analysis["best_is_stable"]: acc["best_is_stable"] += 1
        if analysis["best_is_near"]:   acc["best_is_near"] += 1
        
        n_processed += 1
        
        # Compact per-image record
        results_per_image.append({
            "image": name,
            "best_iou": round(analysis["best_prompt_iou"], 4),
            "best_is_stable": analysis["best_is_stable"],
            "best_is_near": analysis["best_is_near"],
            "uncal_gap": round(analysis["uncalibrated"]["gap"], 4),
            "filt_gap": round(analysis["overlap_filtered"]["gap"], 4),
            "adapt_gap": round(analysis["adaptive"]["gap"], 4),
            "adapt_thresh": round(analysis["adaptive"]["threshold"], 4),
            "n_overlap": analysis["overlap_filtered"]["n_with_overlap"],
            "n_excluded": analysis["overlap_filtered"]["n_excluded"],
            "uncal_stable_mean": round(analysis["uncalibrated"]["stable"]["mean"], 4),
            "uncal_brittle_mean": round(analysis["uncalibrated"]["brittle"]["mean"], 4),
            "filt_stable_mean": round(analysis["overlap_filtered"]["stable"]["mean"], 4),
            "filt_brittle_mean": round(analysis["overlap_filtered"]["brittle"]["mean"], 4),
            "weighted_stable": round(analysis["weighted"]["stable_weighted"], 4),
            "weighted_brittle": round(analysis["weighted"]["brittle_weighted"], 4),
            "near_stable_mean": round(analysis["near_far"]["near"]["stable"]["mean"], 4),
            "near_brittle_mean": round(analysis["near_far"]["near"]["brittle"]["mean"], 4),
            "stability_scores": [round(s, 4) for s in scores],
            "per_prompt_ious": [round(x, 4) for x in analysis["per_prompt_ious"]],
            "is_near_gt": analysis["is_near"],
        })
        
        elapsed = time.time() - t0
        eta = elapsed / n_processed * (len(image_paths) - idx - 1) if n_processed > 0 else 0
        print(f"  [{idx+1}/{len(image_paths)}] {name} — "
              f"gap: uncal={analysis['uncalibrated']['gap']:.3f} "
              f"filt={analysis['overlap_filtered']['gap']:.3f} "
              f"adapt={analysis['adaptive']['gap']:.3f} | "
              f"best={'S' if analysis['best_is_stable'] else 'B'} "
              f"{'N' if analysis['best_is_near'] else 'F'} "
              f"(IoU={analysis['best_prompt_iou']:.3f}) | "
              f"ETA {eta/60:.0f}m")
    
    # ── Dataset summary ──
    if n_processed == 0:
        print("  [!] No images processed.")
        return {"dataset": dataset_name, "n_processed": 0, "per_image": []}
    
    def _mean(lst): return float(np.mean(lst)) if lst else 0.0
    
    bsp = acc["best_is_stable"] / n_processed
    bnp = acc["best_is_near"] / n_processed
    
    summary = {
        "dataset": dataset_name,
        "n_processed": n_processed,
        "time_seconds": round(time.time() - t0, 1),
        "best_is_stable_pct": round(bsp, 4),
        "best_is_near_pct": round(bnp, 4),
        "calibration_comparison": {
            "uncalibrated": {
                "mean_stable": round(_mean(acc["uncal_stable_mean"]), 4),
                "mean_brittle": round(_mean(acc["uncal_brittle_mean"]), 4),
                "mean_gap": round(_mean(acc["uncal_gaps"]), 4),
            },
            "overlap_filtered": {
                "mean_stable": round(_mean(acc["filt_stable_mean"]), 4),
                "mean_brittle": round(_mean(acc["filt_brittle_mean"]), 4),
                "mean_gap": round(_mean(acc["filt_gaps"]), 4),
            },
            "weighted": {
                "mean_stable": round(_mean(acc["weighted_stable"]), 4),
                "mean_brittle": round(_mean(acc["weighted_brittle"]), 4),
                "mean_gap": round(_mean(acc["weighted_stable"]) - _mean(acc["weighted_brittle"]), 4),
            },
            "adaptive": {
                "mean_stable": round(_mean(acc["adapt_stable_mean"]), 4),
                "mean_brittle": round(_mean(acc["adapt_brittle_mean"]), 4),
                "mean_gap": round(_mean(acc["adapt_gaps"]), 4),
            },
        },
        "max_iou_comparison": {
            "stable_max": round(_mean(acc["max_stable"]), 4),
            "brittle_max": round(_mean(acc["max_brittle"]), 4),
        },
        "near_far": {
            "near_stable_mean": round(_mean(acc["near_stable_mean"]), 4),
            "near_brittle_mean": round(_mean(acc["near_brittle_mean"]), 4),
            "far_stable_mean": round(_mean(acc["far_stable_mean"]), 4),
            "far_brittle_mean": round(_mean(acc["far_brittle_mean"]), 4),
        },
        "per_image": results_per_image,
    }
    
    # Print comparison table
    cc = summary["calibration_comparison"]
    mx = summary["max_iou_comparison"]
    nf = summary["near_far"]
    
    print(f"\n{'─'*70}")
    print(f"  {dataset_name.upper()} SUMMARY ({n_processed} images, {summary['time_seconds']}s)")
    print(f"{'─'*70}")
    print(f"  {'':25s} {'Uncalibrated':>14s} {'Overlap-Filt':>14s} {'Weighted':>14s} {'Adaptive':>14s}")
    print(f"  {'Mean Stable IoU':25s} {cc['uncalibrated']['mean_stable']:>14.4f} {cc['overlap_filtered']['mean_stable']:>14.4f} {cc['weighted']['mean_stable']:>14.4f} {cc['adaptive']['mean_stable']:>14.4f}")
    print(f"  {'Mean Brittle IoU':25s} {cc['uncalibrated']['mean_brittle']:>14.4f} {cc['overlap_filtered']['mean_brittle']:>14.4f} {cc['weighted']['mean_brittle']:>14.4f} {cc['adaptive']['mean_brittle']:>14.4f}")
    print(f"  {'Gap (S-B)':25s} {cc['uncalibrated']['mean_gap']:>14.4f} {cc['overlap_filtered']['mean_gap']:>14.4f} {cc['weighted']['mean_gap']:>14.4f} {cc['adaptive']['mean_gap']:>14.4f}")
    print(f"\n  Max IoU — Stable: {mx['stable_max']:.4f}  Brittle: {mx['brittle_max']:.4f}")
    print(f"  Best prompt is stable: {acc['best_is_stable']}/{n_processed} ({bsp:.1%})")
    print(f"  Best prompt is near-GT: {acc['best_is_near']}/{n_processed} ({bnp:.1%})")
    print(f"\n  Near/Far Breakdown:")
    print(f"    Near-GT — Stable: {nf['near_stable_mean']:.4f}  Brittle: {nf['near_brittle_mean']:.4f}")
    print(f"    Far     — Stable: {nf['far_stable_mean']:.4f}  Brittle: {nf['far_brittle_mean']:.4f}")
    print(f"{'─'*70}")
    
    # Save per-dataset JSON
    out_path = os.path.join(output_dir, f"{dataset_name}_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")
    
    return summary


def run_fullscale(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse dataset list
    if args.datasets == "all":
        ds_names = list(DATASETS.keys())
    else:
        ds_names = [d.strip() for d in args.datasets.split(",")]
        for d in ds_names:
            if d not in DATASETS:
                print(f"[Error] Unknown dataset '{d}'. Available: {list(DATASETS.keys())}")
                return
    
    print(f"[FullScale] Loading SAM ({args.model_type})")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    predictor = SamPredictor(sam)
    hook = LatentJitterHook(args.latent_epsilon).register(sam.image_encoder)
    
    all_summaries = []
    
    for ds_name in ds_names:
        cfg = DATASETS[ds_name]
        summary = run_dataset(
            predictor, hook, ds_name,
            cfg["image_dir"], cfg["mask_dir"], args.output_dir,
            n_near=args.n_near, n_far_grid=args.n_far_grid,
            n_trials=args.n_trials, fixed_threshold=args.stability_threshold,
            prompt_jitter_frac=args.prompt_jitter, use_boxes=not args.use_points,
        )
        all_summaries.append(summary)
    
    # ── Aggregate across datasets ──
    if len(all_summaries) > 1:
        print(f"\n{'='*70}")
        print(f"  AGGREGATE ACROSS {len(all_summaries)} DATASETS")
        print(f"{'='*70}")
        for s in all_summaries:
            if s["n_processed"] > 0:
                cc = s["calibration_comparison"]
                print(f"  {s['dataset']:20s} — uncal gap: {cc['uncalibrated']['mean_gap']:.4f}  "
                      f"filt gap: {cc['overlap_filtered']['mean_gap']:.4f}  "
                      f"best-stable: {s['best_is_stable_pct']:.1%}")
    
    # Save combined results
    combined_path = os.path.join(args.output_dir, "fullscale_results.json")
    combined = {
        "config": {
            "model_type": args.model_type,
            "n_near": args.n_near, "n_far_grid": args.n_far_grid,
            "n_trials": args.n_trials,
            "fixed_threshold": args.stability_threshold,
            "latent_epsilon": args.latent_epsilon,
            "prompt_jitter": args.prompt_jitter,
        },
        "datasets": {s["dataset"]: {k: v for k, v in s.items() if k != "per_image"}
                     for s in all_summaries},
    }
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Combined results: {combined_path}")
    
    hook.remove()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Latent Stability Full-Scale Experiment")
    p.add_argument("--checkpoint", default="/home/mi3dr/SCCS/sccs/sam_vit_h_4b8939.pth")
    p.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--datasets", default="busi_malignant",
                   help="Comma-separated dataset names or 'all'")
    p.add_argument("--output-dir", default="/home/mi3dr/SCCS/eccv/fullscale_results")
    p.add_argument("--n-near", type=int, default=18,
                   help="Number of near-GT prompts per image")
    p.add_argument("--n-far-grid", type=int, default=4,
                   help="Grid size for far prompts (n^2 total)")
    p.add_argument("--n-trials", type=int, default=5,
                   help="Number of jitter trials per prompt")
    p.add_argument("--stability-threshold", type=float, default=0.7,
                   help="Fixed threshold for stable/brittle split")
    p.add_argument("--latent-epsilon", type=float, default=0.1)
    p.add_argument("--prompt-jitter", type=float, default=0.10)
    p.add_argument("--use-points", action="store_true",
                   help="Use point prompts instead of box prompts")
    p.add_argument("--device", default="cuda")
    
    args = p.parse_args()
    run_fullscale(args)
