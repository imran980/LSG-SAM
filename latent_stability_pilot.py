"""
Latent-Stability-Guided Prompt Pruning for SAM — Pilot Implementation
=====================================================================
Usage:
    python latent_stability_pilot.py \
        --checkpoint sam_vit_h_4b8939.pth \
        --model-type vit_h \
        --image-dir /path/to/images \
        --mask-dir /path/to/gt_masks \
        --output-dir ./pilot_results
"""

import os, argparse, json, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

warnings.filterwarnings("ignore")


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
    # Compute GT-relative scale if available
    if gt_mask is not None and gt_mask.any():
        ys_gt, xs_gt = np.where(gt_mask)
        gt_w = xs_gt.max() - xs_gt.min()
        gt_h = ys_gt.max() - ys_gt.min()
    else:
        gt_w, gt_h = None, None

    boxes = []
    for cx, cy in points:
        if gt_w is not None:
            # Scale between 0.8x and 1.5x GT bbox size
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
    gt_coords = np.argwhere(gt_mask)  # (row, col)
    if len(gt_coords) == 0:
        # Fallback to pure grid if GT is empty
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
    far_boxes  = points_to_boxes(far_pts, h, w)  # random-scale boxes
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
        predictor.set_image(image)  # re-encode with noise
        
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
# 5. PRUNE & ENSEMBLE
# ═══════════════════════════════════════════════════════════════

def analyze_per_prompt(scores, all_masks, gt, threshold=0.7):
    """Per-prompt analysis: compute representative mask IoU with GT for each prompt,
    then partition into stable vs brittle groups."""
    per_prompt_ious = []
    per_prompt_reps = []
    
    for i, (score, masks) in enumerate(zip(scores, all_masks)):
        rep = majority_vote(masks)  # representative mask for this prompt
        iou = mask_iou(rep, gt)
        per_prompt_ious.append(iou)
        per_prompt_reps.append(rep)
    
    stable_idx  = [i for i, s in enumerate(scores) if s >= threshold]
    brittle_idx = [i for i, s in enumerate(scores) if s < threshold]
    
    stable_ious  = [per_prompt_ious[i] for i in stable_idx]
    brittle_ious = [per_prompt_ious[i] for i in brittle_idx]
    
    return {
        "per_prompt_ious": per_prompt_ious,
        "stable_idx": stable_idx,
        "brittle_idx": brittle_idx,
        "stable_ious": stable_ious,
        "brittle_ious": brittle_ious,
        "mean_stable_iou": float(np.mean(stable_ious)) if stable_ious else 0.0,
        "mean_brittle_iou": float(np.mean(brittle_ious)) if brittle_ious else 0.0,
        "max_stable_iou": float(np.max(stable_ious)) if stable_ious else 0.0,
        "max_brittle_iou": float(np.max(brittle_ious)) if brittle_ious else 0.0,
        "mean_all_iou": float(np.mean(per_prompt_ious)),
        "max_all_iou": float(np.max(per_prompt_ious)),
        "best_prompt_idx": int(np.argmax(per_prompt_ious)),
        "best_prompt_is_stable": int(np.argmax(per_prompt_ious)) in stable_idx,
        "n_stable": len(stable_idx),
        "n_brittle": len(brittle_idx),
    }


# ═══════════════════════════════════════════════════════════════
# 6. PILOT RUNNER — THE CRUCIAL CHECKPOINT
# ═══════════════════════════════════════════════════════════════

def run_pilot(sam_checkpoint, model_type, image_dir, mask_dir, output_dir,
              grid_size=6, n_trials=5, stability_threshold=0.7,
              latent_epsilon=0.1, prompt_jitter_frac=0.10,
              use_boxes=True, device="cuda"):
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Pilot] Loading SAM ({model_type})")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    hook = LatentJitterHook(latent_epsilon).register(sam.image_encoder)
    
    image_paths = sorted([p for p in Path(image_dir).glob("*.png") if "_mask" not in p.stem]) + \
                  sorted([p for p in Path(image_dir).glob("*.jpg") if "_mask" not in p.stem])
    print(f"[Pilot] Found {len(image_paths)} images")
    
    # Accumulators across all images
    img_mean_stable, img_mean_brittle, img_mean_all = [], [], []
    img_max_stable, img_max_brittle, img_max_all = [], [], []
    best_is_stable_count = 0
    results_per_image = []
    
    for img_path in image_paths:
        name = img_path.stem
        gt_path = next((p for p in [Path(mask_dir)/f"{name}_mask.png",
                                     Path(mask_dir)/f"{name}_mask.tif",
                                     Path(mask_dir)/f"{name}.png",
                                     Path(mask_dir)/f"{name}.jpg",
                                     Path(mask_dir)/f"{name}.tif"] if p.exists()), None)
        if gt_path and gt_path == img_path:
            gt_path = None
        if not gt_path:
            print(f"  [Skip] No GT for {name}"); continue
        
        image = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("L")) > 127
        h, w = image.shape[:2]
        if gt.shape != (h, w):
            gt = np.array(Image.fromarray(gt.astype(np.uint8)*255).resize((w,h))) > 127
        
        prompts = generate_gt_informed_prompts(h, w, gt, n_near=18, n_far_grid=4)
        prompt_arr = prompts["boxes"] if use_boxes else prompts["points"]
        is_near = prompts["is_near"]
        n_near = is_near.sum(); n_far = len(is_near) - n_near
        print(f"  [{name}] {len(prompt_arr)} prompts ({n_near} near-GT, {n_far} far)...")
        
        scores, all_masks = [], []
        for p in prompt_arr:
            s, m = score_prompt_stability(predictor, image, p, use_boxes, hook, n_trials, prompt_jitter_frac)
            scores.append(s); all_masks.append(m)
        
        # Per-prompt IoU analysis
        analysis = analyze_per_prompt(scores, all_masks, gt, stability_threshold)
        
        img_mean_stable.append(analysis["mean_stable_iou"])
        img_mean_brittle.append(analysis["mean_brittle_iou"])
        img_mean_all.append(analysis["mean_all_iou"])
        img_max_stable.append(analysis["max_stable_iou"])
        img_max_brittle.append(analysis["max_brittle_iou"])
        img_max_all.append(analysis["max_all_iou"])
        if analysis["best_prompt_is_stable"]:
            best_is_stable_count += 1
        
        results_per_image.append({
            "image": name,
            "n_stable": analysis["n_stable"],
            "n_brittle": analysis["n_brittle"],
            "mean_stable_iou": round(analysis["mean_stable_iou"], 4),
            "mean_brittle_iou": round(analysis["mean_brittle_iou"], 4),
            "max_stable_iou": round(analysis["max_stable_iou"], 4),
            "max_brittle_iou": round(analysis["max_brittle_iou"], 4),
            "mean_all_iou": round(analysis["mean_all_iou"], 4),
            "best_prompt_stable": analysis["best_prompt_is_stable"],
            "stability_scores": [round(s, 4) for s in scores],
            "per_prompt_ious": [round(x, 4) for x in analysis["per_prompt_ious"]],
            "is_near_gt": is_near.tolist(),
        })
        print(f"    Mean IoU — Stable: {analysis['mean_stable_iou']:.4f} | "
              f"Brittle: {analysis['mean_brittle_iou']:.4f} | All: {analysis['mean_all_iou']:.4f}")
        print(f"    Max  IoU — Stable: {analysis['max_stable_iou']:.4f} | "
              f"Brittle: {analysis['max_brittle_iou']:.4f} | "
              f"Best is {'STABLE' if analysis['best_prompt_is_stable'] else 'BRITTLE'}")
    
    # ═══════════════════════════════════════════════════════════
    # THE CRUCIAL CHECKPOINT
    # ═══════════════════════════════════════════════════════════
    n = len(results_per_image)
    ms  = np.mean(img_mean_stable)  if img_mean_stable  else 0
    mb  = np.mean(img_mean_brittle) if img_mean_brittle else 0
    ma  = np.mean(img_mean_all)     if img_mean_all     else 0
    mxs = np.mean(img_max_stable)   if img_max_stable   else 0
    mxb = np.mean(img_max_brittle)  if img_max_brittle  else 0
    mxa = np.mean(img_max_all)      if img_max_all      else 0
    gap = ms - mb
    best_stable_pct = best_is_stable_count / n if n > 0 else 0
    
    print("\n" + "="*70)
    print("PILOT CHECKPOINT — PER-PROMPT IoU ANALYSIS")
    print("="*70)
    print(f"  Images evaluated: {n}")
    print(f"  Mean per-prompt IoU — Stable: {ms:.4f}  Brittle: {mb:.4f}  All: {ma:.4f}")
    print(f"  Max  per-prompt IoU — Stable: {mxs:.4f}  Brittle: {mxb:.4f}  All: {mxa:.4f}")
    print(f"  Gap (mean stable - mean brittle): {gap:.4f}")
    print(f"  Best prompt is stable: {best_is_stable_count}/{n} ({best_stable_pct:.1%})")
    
    if gap > 0.05 and best_stable_pct > 0.6:
        verdict = f"STRONG SIGNAL — Gap {gap:.4f}, best-is-stable {best_stable_pct:.1%}. Worth pursuing."
    elif gap > 0.02 or best_stable_pct > 0.55:
        verdict = f"MODERATE — Gap {gap:.4f}, best-is-stable {best_stable_pct:.1%}. Tune parameters."
    else:
        verdict = f"WEAK — Gap {gap:.4f}, best-is-stable {best_stable_pct:.1%}. Rethink approach."
    
    print(f"  VERDICT: {verdict}")
    print("="*70)
    
    with open(os.path.join(output_dir, "pilot_results.json"), "w") as f:
        json.dump({"config": {"grid_size": grid_size, "n_trials": n_trials,
                               "threshold": stability_threshold, "epsilon": latent_epsilon},
                   "checkpoint": {"mean_stable": round(ms, 4), "mean_brittle": round(mb, 4),
                                  "mean_all": round(ma, 4), "max_stable": round(mxs, 4),
                                  "max_brittle": round(mxb, 4), "gap": round(gap, 4),
                                  "best_is_stable_pct": round(best_stable_pct, 4),
                                  "verdict": verdict},
                   "per_image": results_per_image}, f, indent=2)
    
    hook.remove()
    return verdict


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/mi3dr/SCCS/sccs/sam_vit_h_4b8939.pth")
    p.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    p.add_argument("--image-dir", default="/home/mi3dr/SCCS/sccs/Dataset_BUSI_with_GT/malignant")
    p.add_argument("--mask-dir", default="/home/mi3dr/SCCS/sccs/Dataset_BUSI_with_GT/malignant")
    p.add_argument("--output-dir", default="/home/mi3dr/SCCS/eccv/pilot_results")
    p.add_argument("--grid-size", type=int, default=6)
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--stability-threshold", type=float, default=0.7)
    p.add_argument("--latent-epsilon", type=float, default=0.1)
    p.add_argument("--prompt-jitter", type=float, default=0.10)
    p.add_argument("--use-points", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    
    run_pilot(a.checkpoint, a.model_type, a.image_dir, a.mask_dir, a.output_dir,
              a.grid_size, a.n_trials, a.stability_threshold, a.latent_epsilon,
              a.prompt_jitter, not a.use_points, a.device)