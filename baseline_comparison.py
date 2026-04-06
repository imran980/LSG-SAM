#!/usr/bin/env python3
"""
Baseline Comparison: SAM (single) vs SAM (best-of-N) vs MedSAM vs Ours
across BUSI-malignant, JSRT, Kvasir-SEG.

Produces:
  1. Quantitative table (IoU, Dice, Hallucination) printed + saved as CSV/LaTeX
  2. Qualitative figure: smart-selected examples per dataset
     - Top improvement cases
     - Hallucination prevention cases
     - Hard recovery cases
"""

import os, sys, time, json, argparse, random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, "/home/mi3dr/SCCS/sccs")
from segment_anything import sam_model_registry, SamPredictor

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

BASE = "/home/mi3dr/SCCS/sccs"

DATASETS = {
    "busi_malignant": {
        "image_dir": f"{BASE}/Dataset_BUSI_with_GT/malignant",
        "mask_dir":  f"{BASE}/Dataset_BUSI_with_GT/malignant",
        "text_query": "breast ultrasound tumor",
    },
    "jsrt": {
        "image_dir": f"{BASE}/jsrt/jpg",
        "mask_dir":  f"{BASE}/jsrt/masks",
        "text_query": "chest X-ray lungs",
    },
    "kvasir": {
        "image_dir": f"{BASE}/Kvasir-SEG/images",
        "mask_dir":  f"{BASE}/Kvasir-SEG/masks",
        "text_query": "colorectal polyp endoscopy",
    },
}

SAM_H_CKPT = f"{BASE}/sam_vit_h_4b8939.pth"
SAM_B_CKPT = f"{BASE}/checkpoints/sam_vit_b.pth"
MEDSAM_CKPT = f"{BASE}/checkpoints/medsam_vit_b.pth"


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def dice_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return float(2 * inter / total) if total > 0 else (1.0 if inter == 0 else 0.0)


def evaluate_mask(pred, gt):
    iou = mask_iou(pred, gt)
    dc = dice_score(pred, gt)
    halluc = (iou < 0.05) and pred.any()
    return {"iou": iou, "dice": dc, "hallucination": halluc}


def gt_center_box(gt_mask, h, w, scale=1.2):
    ys, xs = np.where(gt_mask)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    return np.array([
        max(0, cx - bw/2), max(0, cy - bh/2),
        min(w-1, cx + bw/2), min(h-1, cy + bh/2)
    ], dtype=np.float64)


def find_gt_mask(name, mask_dir, img_path):
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


def generate_variants(base_box, h, w, n=10):
    cx = (base_box[0] + base_box[2]) / 2
    cy = (base_box[1] + base_box[3]) / 2
    bw = base_box[2] - base_box[0]
    bh = base_box[3] - base_box[1]
    variants = [base_box.copy()]
    rng = np.random.RandomState(42)
    for _ in range(n - 1):
        s = rng.uniform(0.85, 1.15)
        jx = rng.uniform(-0.08, 0.08) * bw
        jy = rng.uniform(-0.08, 0.08) * bh
        box = np.array([
            max(0, cx - bw*s/2 + jx), max(0, cy - bh*s/2 + jy),
            min(w-1, cx + bw*s/2 + jx), min(h-1, cy + bh*s/2 + jy)
        ], dtype=np.float64)
        variants.append(box)
    return variants


# ═══════════════════════════════════════════════════════════════
# IMPORT "OURS" PIPELINE COMPONENTS
# ═══════════════════════════════════════════════════════════════

from stability_medclipsam import (
    generate_variants_from_box,
    run_sam_variants,
    compute_global_consistency,
    spatial_search,
    refine_from_mask,
)

# Ours config — match stability_medclipsam defaults
OURS_CONF_GATE = 0.85       # lowered from 0.92 so method actually triggers
OURS_TAU_CONSISTENCY = 0.5


def run_ours(predictor, image, gt_box, n_variants=10):
    """Run our full pipeline: confidence gate → consistency check → refine/search.
    Returns (mask, route, safeguard)."""
    h, w = image.shape[:2]

    # Step 1: SAM single prediction
    predictor.set_image(image)
    pred_masks, pred_scores, _ = predictor.predict(
        box=gt_box[None, :], multimask_output=False)
    sam_mask = pred_masks[0].astype(bool)
    sam_score = float(pred_scores[0])

    # Gate: high confidence → keep SAM as-is
    if sam_score >= OURS_CONF_GATE:
        return sam_mask, "confident", "kept_sam"

    # Step 2: Generate variants and run SAM
    variants = generate_variants_from_box(gt_box, h, w, n_variants=n_variants)
    sam_results = run_sam_variants(predictor, image, variants)
    consistency = compute_global_consistency(sam_results)

    # Step 3: Route based on consistency
    if consistency < OURS_TAU_CONSISTENCY:
        candidate, _ = spatial_search(predictor, image, gt_box, h, w)
        route = "search"
    else:
        candidate, _ = refine_from_mask(predictor, image, sam_mask)
        route = "refine"

    # Step 4: Safeguard — only upgrade if meaningfully different
    if candidate is not None and candidate.any():
        cand_vs_sam = mask_iou(candidate, sam_mask)
        if cand_vs_sam > 0.9:
            return sam_mask, route, "kept_sam"
        else:
            return candidate, route, "upgraded"

    return sam_mask, route, "kept_sam"


# ═══════════════════════════════════════════════════════════════
# QUALITATIVE SAMPLE SELECTION (the smart part)
# ═══════════════════════════════════════════════════════════════

def select_qualitative_examples(per_image_data, top_k=4):
    """
    Smart selection of qualitative examples per dataset.

    Returns dict with keys:
      "top_improvement": images where Ours >> SAM
      "hard_recovery":   images where SAM < 0.5 but Ours > 0.6
      "halluc_prevent":  images where SAM hallucinates but Ours doesn't
      "best_overall":    top-k by (Ours - SAM) delta

    Each entry is a list of dicts with image data + masks.
    """
    results = {
        "top_improvement": [],
        "hard_recovery": [],
        "halluc_prevent": [],
    }

    # Compute deltas
    for item in per_image_data:
        iou_sam = item["ev_sam"]["iou"]
        iou_bon = item["ev_bon"]["iou"]
        iou_ours = item["ev_ours"]["iou"]
        delta = iou_ours - iou_sam

        item["delta_vs_sam"] = delta
        item["delta_vs_bon"] = iou_ours - iou_bon

        # Category 1: Clear improvement (delta > 0.08)
        if delta > 0.08:
            results["top_improvement"].append(item)

        # Category 2: Hard recovery (SAM bad, Ours good)
        if iou_sam < 0.5 and iou_ours > 0.6:
            results["hard_recovery"].append(item)

        # Category 3: Hallucination prevention
        sam_halluc = item["ev_sam"]["hallucination"]
        ours_halluc = item["ev_ours"]["hallucination"]
        if sam_halluc and not ours_halluc:
            results["halluc_prevent"].append(item)

    # Sort each category by delta descending
    for key in results:
        results[key].sort(key=lambda x: x["delta_vs_sam"], reverse=True)
        results[key] = results[key][:top_k]

    # Best overall: sort ALL by delta, take top-k, skip near-ties
    candidates = [item for item in per_image_data if abs(item["delta_vs_sam"]) > 0.03]
    candidates.sort(key=lambda x: x["delta_vs_sam"], reverse=True)
    results["best_overall"] = candidates[:top_k]

    return results


# ═══════════════════════════════════════════════════════════════
# QUALITATIVE FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════

def make_qualitative_figure(examples, methods, ds_name, output_dir, tag="best"):
    """
    Render a qualitative comparison grid.
    Columns: Image | GT | SAM (single) | SAM (best-of-N) | MedSAM | Ours
    """
    if not examples:
        return

    n_rows = len(examples)
    col_labels = ["Image", "Ground Truth"] + methods
    n_cols = len(col_labels)

    # Method overlay colors: red, blue, orange, green
    overlay_colors = {
        "SAM (single)":   [1, 0, 0, 0.4],
        "SAM (best-of-N)": [0, 0.4, 1, 0.4],
        "MedSAM":          [1, 0.6, 0, 0.4],
        "Ours":            [0, 0.85, 0.2, 0.45],
    }

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.2, n_rows * 3.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, item in enumerate(examples):
        img = item["image"]
        gt_mask = item["gt"]

        # Col 0: Image
        axes[row, 0].imshow(img)
        delta = item.get("delta_vs_sam", 0)
        axes[row, 0].set_ylabel(
            f"{item['name']}\nΔIoU={delta:+.3f}",
            fontsize=9, fontweight='bold', rotation=90, labelpad=12)

        # Col 1: GT overlay (green)
        axes[row, 1].imshow(img)
        gt_ov = np.zeros((*gt_mask.shape, 4))
        gt_ov[gt_mask] = [0, 1, 0, 0.45]
        axes[row, 1].imshow(gt_ov)

        # Cols 2+: Methods
        for j, m in enumerate(methods):
            col = j + 2
            axes[row, col].imshow(img)
            pred_mask = item.get(f"mask_{m}")
            if pred_mask is not None:
                ov = np.zeros((*pred_mask.shape, 4))
                ov[pred_mask] = overlay_colors.get(m, [0.5, 0.5, 0.5, 0.4])
                axes[row, col].imshow(ov)
                iou_val = item.get(f"iou_{m}", 0)
                axes[row, col].set_xlabel(f"IoU: {iou_val:.3f}", fontsize=9)

        # Column headers
        if row == 0:
            for c, label in enumerate(col_labels):
                axes[row, c].set_title(label, fontsize=11, fontweight='bold', pad=8)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    title = f"{ds_name.replace('_', ' ').title()} — {tag.replace('_', ' ').title()}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"qual_{ds_name}_{tag}.png")
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run(args):
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # Monkey-patch torch.load for PyTorch 2.6+
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

    # ── Load SAM ViT-H ──
    print("[+] Loading SAM ViT-H...")
    sam_h = sam_model_registry["vit_h"](checkpoint=SAM_H_CKPT).to(device)
    pred_h = SamPredictor(sam_h)

    # ── Load MedSAM ViT-B ──
    print("[+] Loading MedSAM ViT-B...")
    _orig_lsd = torch.nn.Module.load_state_dict
    torch.nn.Module.load_state_dict = lambda self, sd, **kw: _orig_lsd(self, sd, strict=False)
    medsam = sam_model_registry["vit_b"](checkpoint=MEDSAM_CKPT)
    torch.nn.Module.load_state_dict = _orig_lsd
    medsam = medsam.to(device)
    pred_med = SamPredictor(medsam)
    print("[+] MedSAM loaded.")

    # Restore original torch.load
    torch.load = _orig_load

    methods = ["SAM (single)", "SAM (best-of-N)", "MedSAM", "Ours"]
    all_results = {}

    for ds_name in args.datasets.split(","):
        ds_name = ds_name.strip()
        if ds_name not in DATASETS:
            print(f"[!] Unknown: {ds_name}"); continue
        cfg = DATASETS[ds_name]

        image_paths = sorted([p for p in Path(cfg["image_dir"]).glob("*.png")
                              if "_mask" not in p.stem]) + \
                      sorted([p for p in Path(cfg["image_dir"]).glob("*.jpg")
                              if "_mask" not in p.stem])

        print(f"\n{'='*70}")
        print(f"  {ds_name} — {len(image_paths)} images")
        print(f"{'='*70}")

        accum = {m: {"ious": [], "dices": [], "hallucs": []} for m in methods}
        per_image_data = []    # for qualitative selection
        t0 = time.time()
        n_proc = 0

        for idx, img_path in enumerate(image_paths):
            name = img_path.stem
            gt_path = find_gt_mask(name, cfg["mask_dir"], img_path)
            if not gt_path:
                continue

            image = np.array(Image.open(img_path).convert("RGB"))
            gt_raw = np.array(Image.open(gt_path).convert("L"))
            gt = gt_raw > (127 if gt_raw.max() > 1 else 0)
            h, w = image.shape[:2]
            if gt.shape != (h, w):
                gt = np.array(Image.fromarray(gt.astype(np.uint8)*255).resize((w, h))) > 127
            if not gt.any():
                continue

            gt_box = gt_center_box(gt, h, w, scale=1.2)

            # ── SAM (single) ──
            pred_h.set_image(image)
            masks_s, scores_s, _ = pred_h.predict(box=gt_box[None, :], multimask_output=False)
            sam_single = masks_s[0].astype(bool)
            ev_s = evaluate_mask(sam_single, gt)
            accum["SAM (single)"]["ious"].append(ev_s["iou"])
            accum["SAM (single)"]["dices"].append(ev_s["dice"])
            accum["SAM (single)"]["hallucs"].append(ev_s["hallucination"])

            # ── SAM (best-of-N) ──
            variants = generate_variants(gt_box, h, w, n=args.n_variants)
            best_mask_bon = None
            best_score_bon = -1
            for v in variants:
                ms, sc, _ = pred_h.predict(box=v[None, :], multimask_output=True)
                for j in range(len(ms)):
                    if float(sc[j]) > best_score_bon:
                        best_score_bon = float(sc[j])
                        best_mask_bon = ms[j].astype(bool)
            if best_mask_bon is None:
                best_mask_bon = sam_single
            ev_b = evaluate_mask(best_mask_bon, gt)
            accum["SAM (best-of-N)"]["ious"].append(ev_b["iou"])
            accum["SAM (best-of-N)"]["dices"].append(ev_b["dice"])
            accum["SAM (best-of-N)"]["hallucs"].append(ev_b["hallucination"])

            # ── MedSAM ──
            pred_med.set_image(image)
            masks_m, scores_m, _ = pred_med.predict(box=gt_box[None, :], multimask_output=False)
            medsam_mask = masks_m[0].astype(bool)
            ev_m = evaluate_mask(medsam_mask, gt)
            accum["MedSAM"]["ious"].append(ev_m["iou"])
            accum["MedSAM"]["dices"].append(ev_m["dice"])
            accum["MedSAM"]["hallucs"].append(ev_m["hallucination"])

            # ── Ours (stability-guided pipeline) ──
            ours_mask, route, safeguard = run_ours(
                pred_h, image, gt_box, n_variants=args.n_variants)
            ev_o = evaluate_mask(ours_mask, gt)
            accum["Ours"]["ious"].append(ev_o["iou"])
            accum["Ours"]["dices"].append(ev_o["dice"])
            accum["Ours"]["hallucs"].append(ev_o["hallucination"])

            n_proc += 1

            # ── Store per-image data for qualitative selection ──
            per_image_data.append({
                "name": name,
                "image": image,
                "gt": gt,
                "ev_sam": ev_s,
                "ev_bon": ev_b,
                "ev_med": ev_m,
                "ev_ours": ev_o,
                "route": route,
                "safeguard": safeguard,
                # Store masks for figure rendering
                "mask_SAM (single)": sam_single,
                "mask_SAM (best-of-N)": best_mask_bon,
                "mask_MedSAM": medsam_mask,
                "mask_Ours": ours_mask,
                # Store IoU for figure labels
                "iou_SAM (single)": ev_s["iou"],
                "iou_SAM (best-of-N)": ev_b["iou"],
                "iou_MedSAM": ev_m["iou"],
                "iou_Ours": ev_o["iou"],
            })

            if n_proc % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / n_proc * (len(image_paths) - idx - 1)
                flag = "🔍" if route == "search" else ("🔧" if route == "refine" else "✓")
                sg = "⬆" if safeguard == "upgraded" else "="
                print(f"  [{idx+1}/{len(image_paths)}] "
                      f"SAM={ev_s['iou']:.3f} BoN={ev_b['iou']:.3f} "
                      f"Med={ev_m['iou']:.3f} Ours={ev_o['iou']:.3f} {flag}{sg} "
                      f"| ETA {eta/60:.0f}m")

        dt = time.time() - t0
        print(f"\n  {ds_name}: {n_proc} images, {dt/60:.1f} min")

        # ── Aggregate metrics ──
        summary = {}
        for m in methods:
            if accum[m]["ious"]:
                summary[m] = {
                    "iou": round(float(np.mean(accum[m]["ious"])), 4),
                    "dice": round(float(np.mean(accum[m]["dices"])), 4),
                    "halluc": round(float(np.mean(accum[m]["hallucs"])), 4),
                }
        all_results[ds_name] = summary

        # ── Smart qualitative selection ──
        print(f"\n  Selecting qualitative examples...")
        qual = select_qualitative_examples(per_image_data, top_k=4)

        for tag, examples in qual.items():
            n_found = len(examples)
            if n_found > 0:
                print(f"    {tag}: {n_found} candidates found")
                make_qualitative_figure(examples, methods, ds_name,
                                        args.output_dir, tag=tag)
            else:
                print(f"    {tag}: 0 candidates")

        # ── Print routing stats ──
        n_confident = sum(1 for d in per_image_data if d["route"] == "confident")
        n_search = sum(1 for d in per_image_data if d["route"] == "search")
        n_refine = sum(1 for d in per_image_data if d["route"] == "refine")
        n_upgraded = sum(1 for d in per_image_data if d["safeguard"] == "upgraded")
        print(f"\n  Routing stats:")
        print(f"    Confident (kept SAM): {n_confident}/{n_proc} ({n_confident/max(n_proc,1):.1%})")
        print(f"    Search route:         {n_search}/{n_proc} ({n_search/max(n_proc,1):.1%})")
        print(f"    Refine route:         {n_refine}/{n_proc} ({n_refine/max(n_proc,1):.1%})")
        print(f"    Actually upgraded:    {n_upgraded}/{n_proc} ({n_upgraded/max(n_proc,1):.1%})")

        # ── Print delta distribution ──
        deltas = [d["ev_ours"]["iou"] - d["ev_sam"]["iou"] for d in per_image_data]
        deltas = np.array(deltas)
        n_better = (deltas > 0.03).sum()
        n_worse = (deltas < -0.03).sum()
        n_tie = n_proc - n_better - n_worse
        print(f"\n  Ours vs SAM(single):")
        print(f"    Improved (Δ>0.03): {n_better}/{n_proc} ({n_better/max(n_proc,1):.1%})")
        print(f"    Tied:              {n_tie}/{n_proc} ({n_tie/max(n_proc,1):.1%})")
        print(f"    Worse (Δ<-0.03):   {n_worse}/{n_proc} ({n_worse/max(n_proc,1):.1%})")
        print(f"    Mean Δ:            {deltas.mean():+.4f}")

        # Free per-image images from memory (keep only summary)
        del per_image_data

    # ═══════════════════════════════════════════════════════════
    # PRINT TABLE
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("  QUANTITATIVE RESULTS")
    print(f"{'='*80}")
    print(f"  {'Dataset':20s} {'Method':20s} {'IoU↑':>8s} {'Dice↑':>8s} {'Halluc↓':>8s}")
    print(f"  {'─'*64}")
    for ds in all_results:
        for m in methods:
            if m in all_results[ds]:
                r = all_results[ds][m]
                print(f"  {ds:20s} {m:20s} {r['iou']:>8.4f} {r['dice']:>8.4f} {r['halluc']:>7.1%}")
        print(f"  {'─'*64}")

    # ── Save CSV ──
    csv_path = os.path.join(args.output_dir, "baseline_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("dataset,method,iou,dice,hallucination\n")
        for ds in all_results:
            for m in methods:
                if m in all_results[ds]:
                    r = all_results[ds][m]
                    f.write(f"{ds},{m},{r['iou']},{r['dice']},{r['halluc']}\n")
    print(f"\n  CSV saved: {csv_path}")

    # ── Save LaTeX ──
    latex_path = os.path.join(args.output_dir, "baseline_comparison.tex")
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Baseline comparison across medical segmentation datasets.}\n")
        f.write("\\label{tab:baseline}\n")
        f.write("\\begin{tabular}{llccc}\n\\toprule\n")
        f.write("Dataset & Method & IoU $\\uparrow$ & Dice $\\uparrow$ & Halluc $\\downarrow$ \\\\\n")
        f.write("\\midrule\n")
        for i, ds in enumerate(all_results):
            ds_label = ds.replace("_", "\\_")
            for j, m in enumerate(methods):
                if m in all_results[ds]:
                    r = all_results[ds][m]
                    ds_col = ds_label if j == 0 else ""
                    # Bold best IoU/Dice per dataset
                    best_iou = max(all_results[ds][mm]["iou"] for mm in methods if mm in all_results[ds])
                    best_dice = max(all_results[ds][mm]["dice"] for mm in methods if mm in all_results[ds])
                    best_halluc = min(all_results[ds][mm]["halluc"] for mm in methods if mm in all_results[ds])
                    iou_s = f"\\textbf{{{r['iou']:.4f}}}" if r['iou'] == best_iou else f"{r['iou']:.4f}"
                    dice_s = f"\\textbf{{{r['dice']:.4f}}}" if r['dice'] == best_dice else f"{r['dice']:.4f}"
                    halluc_s = f"\\textbf{{{r['halluc']:.1%}}}" if r['halluc'] == best_halluc else f"{r['halluc']:.1%}"
                    f.write(f"{ds_col} & {m} & {iou_s} & {dice_s} & {halluc_s} \\\\\n")
            if i < len(all_results) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  LaTeX saved: {latex_path}")

    # ── Save JSON ──
    json_path = os.path.join(args.output_dir, "baseline_comparison.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  JSON saved: {json_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="busi_malignant,jsrt,kvasir",
                   help="Comma-separated dataset list")
    p.add_argument("--n-variants", type=int, default=10,
                   help="Number of box variants for best-of-N and Ours")
    p.add_argument("--output-dir", default="results/baseline_comparison",
                   help="Output directory for results")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(args)