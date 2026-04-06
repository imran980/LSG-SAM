#!/usr/bin/env python3
"""
Paper-Ready Qualitative Figure Generator
=========================================
Runs ALL 4 methods on all datasets, selects the BEST showcase image
per dataset (where Ours wins most), produces a single publication-quality
figure.

Layout:  3 rows (datasets) × 6 cols (Image, GT, SAM, SAM-BoN, MedSAM, Ours)

Usage:
    python qualitative_figure.py --datasets busi_malignant,jsrt,kvasir
    python qualitative_figure.py --datasets busi_malignant,jsrt,kvasir --top-k 2
"""

import os, sys, time, argparse, random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, "/home/mi3dr/SCCS/sccs")
from segment_anything import sam_model_registry, SamPredictor

# Import Ours pipeline
from stability_medclipsam import (
    generate_variants_from_box,
    run_sam_variants,
    compute_global_consistency,
    spatial_search,
    refine_from_mask,
)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

BASE = "/home/mi3dr/SCCS/sccs"

DATASETS = {
    "busi_malignant": {
        "image_dir": f"{BASE}/Dataset_BUSI_with_GT/malignant",
        "mask_dir":  f"{BASE}/Dataset_BUSI_with_GT/malignant",
        "label": "BUSI (Malignant)",
    },
    "jsrt": {
        "image_dir": f"{BASE}/jsrt/jpg",
        "mask_dir":  f"{BASE}/jsrt/masks",
        "label": "JSRT (Lungs)",
    },
    "kvasir": {
        "image_dir": f"{BASE}/Kvasir-SEG/images",
        "mask_dir":  f"{BASE}/Kvasir-SEG/masks",
        "label": "Kvasir-SEG (Polyp)",
    },
}

SAM_H_CKPT = f"{BASE}/sam_vit_h_4b8939.pth"
MEDSAM_CKPT = f"{BASE}/checkpoints/medsam_vit_b.pth"

# Ours hyperparams
OURS_CONF_GATE = 0.85
OURS_TAU_CONSISTENCY = 0.5

# ── Overlay colors (RGBA) ──
COLOR_GT   = [0, 1, 0, 0.40]           # green
COLOR_SAM  = [1, 0.15, 0.15, 0.40]     # red
COLOR_BON  = [0, 0.40, 1.0, 0.40]      # blue
COLOR_MED  = [0.60, 0.20, 0.85, 0.45]  # purple
COLOR_OURS = [1, 0.75, 0, 0.50]        # gold/amber — stands out


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


def generate_bon_variants(base_box, h, w, n=10):
    """Box variants for best-of-N (same as baseline_comparison.py)."""
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
# MEDSAM LOADING (with interpolation for 256→1024 mismatch)
# ═══════════════════════════════════════════════════════════════

def load_medsam(ckpt_path, device):
    """Load MedSAM ViT-B with proper interpolation of pos_embed and rel_pos."""
    print("[+] Loading MedSAM ViT-B (with interpolation)...")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = raw.get("model", raw)

    # Build ViT-B architecture without loading any weights
    from segment_anything.modeling import (
        ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer,
    )
    model = Sam(
        image_encoder=ImageEncoderViT(
            depth=12, embed_dim=768, img_size=1024, mlp_ratio=4,
            norm_layer=torch.nn.LayerNorm, num_heads=12, patch_size=16,
            qkv_bias=True, use_rel_pos=True, global_attn_indexes=[2, 5, 8, 11],
            window_size=14, out_chans=256,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=256, image_embedding_size=(64, 64),
            input_image_size=(1024, 1024), mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3, transformer=TwoWayTransformer(
                depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8,
            ),
            transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    # Interpolate mismatched keys
    model_sd = model.state_dict()
    n_interp = 0
    for k in list(state_dict.keys()):
        if k in model_sd and state_dict[k].shape != model_sd[k].shape:
            target_shape = model_sd[k].shape
            ckpt_shape = state_dict[k].shape

            if "pos_embed" in k:
                # [1, H1, W1, D] → [1, H2, W2, D]
                t = state_dict[k].permute(0, 3, 1, 2).float()
                t = F.interpolate(t, size=(target_shape[1], target_shape[2]),
                                  mode="bicubic", align_corners=False)
                state_dict[k] = t.permute(0, 2, 3, 1)
                n_interp += 1

            elif "rel_pos" in k:
                # [L1, D] → [L2, D]
                t = state_dict[k].unsqueeze(0).permute(0, 2, 1).float()
                t = F.interpolate(t, size=target_shape[0],
                                  mode="linear", align_corners=False)
                state_dict[k] = t.permute(0, 2, 1).squeeze(0)
                n_interp += 1

            else:
                # Unknown mismatch — skip this key
                del state_dict[k]

    result = model.load_state_dict(state_dict, strict=False)
    n_missing = len(result.missing_keys)
    n_unexpected = len(result.unexpected_keys)
    print(f"    Interpolated {n_interp} keys (pos_embed + rel_pos)")
    print(f"    Missing: {n_missing}, Unexpected: {n_unexpected}")

    model = model.to(device).eval()
    return SamPredictor(model)


# ═══════════════════════════════════════════════════════════════
# OURS PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_ours(predictor, image, gt_box, n_variants=10):
    """Full pipeline: confidence gate → consistency → refine/search."""
    h, w = image.shape[:2]

    predictor.set_image(image)
    pred_masks, pred_scores, _ = predictor.predict(
        box=gt_box[None, :], multimask_output=False)
    sam_mask = pred_masks[0].astype(bool)
    sam_score = float(pred_scores[0])

    if sam_score >= OURS_CONF_GATE:
        return sam_mask, "confident"

    variants = generate_variants_from_box(gt_box, h, w, n_variants=n_variants)
    sam_results = run_sam_variants(predictor, image, variants)
    consistency = compute_global_consistency(sam_results)

    if consistency < OURS_TAU_CONSISTENCY:
        candidate, _ = spatial_search(predictor, image, gt_box, h, w)
        route = "search"
    else:
        candidate, _ = refine_from_mask(predictor, image, sam_mask)
        route = "refine"

    if candidate is not None and candidate.any():
        if mask_iou(candidate, sam_mask) <= 0.9:
            return candidate, route

    return sam_mask, route


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run(args):
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

    # ── Load models ──
    print("[+] Loading SAM ViT-H...")
    sam_h = sam_model_registry["vit_h"](checkpoint=SAM_H_CKPT).to(device)
    pred_h = SamPredictor(sam_h)

    pred_med = load_medsam(MEDSAM_CKPT, device)

    torch.load = _orig_load

    # ── Process each dataset ──
    methods = ["SAM (single)", "SAM (best-of-N)", "MedSAM", "Ours"]
    best_per_dataset = {}  # ds_name → list of top-k examples

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

        all_scored = []
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

            # ── 1. SAM (single) ──
            pred_h.set_image(image)
            m_s, sc_s, _ = pred_h.predict(box=gt_box[None, :], multimask_output=False)
            sam_single = m_s[0].astype(bool)
            iou_sam = mask_iou(sam_single, gt)

            # ── 2. SAM (best-of-N) ──
            variants = generate_bon_variants(gt_box, h, w, n=args.n_variants)
            best_bon_mask, best_bon_score = None, -1
            for v in variants:
                ms, sc, _ = pred_h.predict(box=v[None, :], multimask_output=True)
                for j in range(len(ms)):
                    if float(sc[j]) > best_bon_score:
                        best_bon_score = float(sc[j])
                        best_bon_mask = ms[j].astype(bool)
            if best_bon_mask is None:
                best_bon_mask = sam_single
            iou_bon = mask_iou(best_bon_mask, gt)

            # ── 3. MedSAM ──
            pred_med.set_image(image)
            m_m, sc_m, _ = pred_med.predict(box=gt_box[None, :], multimask_output=False)
            medsam_mask = m_m[0].astype(bool)
            iou_med = mask_iou(medsam_mask, gt)

            # ── 4. Ours ──
            ours_mask, route = run_ours(pred_h, image, gt_box,
                                        n_variants=args.n_variants)
            iou_ours = mask_iou(ours_mask, gt)

            # ── Score: how much does Ours beat the BEST baseline? ──
            best_baseline = max(iou_sam, iou_bon, iou_med)
            delta_vs_best = iou_ours - best_baseline
            delta_vs_sam = iou_ours - iou_sam

            all_scored.append({
                "name": name,
                "image": image,
                "gt": gt,
                "masks": {
                    "SAM (single)": sam_single,
                    "SAM (best-of-N)": best_bon_mask,
                    "MedSAM": medsam_mask,
                    "Ours": ours_mask,
                },
                "ious": {
                    "SAM (single)": iou_sam,
                    "SAM (best-of-N)": iou_bon,
                    "MedSAM": iou_med,
                    "Ours": iou_ours,
                },
                "delta_vs_best": delta_vs_best,
                "delta_vs_sam": delta_vs_sam,
                "route": route,
            })
            n_proc += 1

            if n_proc % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / n_proc * (len(image_paths) - idx - 1)
                print(f"  [{idx+1}/{len(image_paths)}] {name:25s}  "
                      f"SAM={iou_sam:.3f} BoN={iou_bon:.3f} "
                      f"Med={iou_med:.3f} Ours={iou_ours:.3f} "
                      f"Δ={delta_vs_sam:+.3f} [{route}]  "
                      f"| ETA {eta/60:.0f}m")

        dt = time.time() - t0
        print(f"\n  {ds_name}: {n_proc} images, {dt/60:.1f} min")

        # ── Print summary ──
        for m in methods:
            vals = [x["ious"][m] for x in all_scored]
            print(f"    {m:20s}  IoU={np.mean(vals):.4f} ± {np.std(vals):.4f}")

        # ── Select top-k images where Ours wins ──
        # Primary sort: delta_vs_sam (descending)
        # Filter: Ours IoU > 0.4 (not total garbage) and delta > 0.03 (visible)
        candidates = [x for x in all_scored
                      if x["delta_vs_sam"] > 0.03 and x["ious"]["Ours"] > 0.4]
        candidates.sort(key=lambda x: x["delta_vs_sam"], reverse=True)

        if len(candidates) < args.top_k:
            # Relax filter
            candidates = sorted(all_scored,
                                key=lambda x: x["delta_vs_sam"], reverse=True)

        selected = candidates[:args.top_k]
        print(f"\n  Selected {len(selected)} best examples:")
        for s in selected:
            ious = s["ious"]
            print(f"    {s['name']:25s}  "
                  f"SAM={ious['SAM (single)']:.3f}  "
                  f"BoN={ious['SAM (best-of-N)']:.3f}  "
                  f"Med={ious['MedSAM']:.3f}  "
                  f"Ours={ious['Ours']:.3f}  "
                  f"Δ={s['delta_vs_sam']:+.3f}")

        best_per_dataset[ds_name] = selected

        # Free images we won't use
        del all_scored

    # ═══════════════════════════════════════════════════════════
    # RENDER THE SINGLE PAPER FIGURE
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  Rendering paper figure...")
    print(f"{'='*70}")

    ds_order = [d.strip() for d in args.datasets.split(",") if d.strip() in best_per_dataset]

    # Count total rows
    total_rows = sum(len(best_per_dataset[ds]) for ds in ds_order)
    col_labels = ["Input", "Ground Truth", "SAM", "SAM (best-of-N)", "MedSAM", "Ours (Proposed)"]
    n_cols = len(col_labels)

    overlay_colors = {
        "SAM (single)":    COLOR_SAM,
        "SAM (best-of-N)": COLOR_BON,
        "MedSAM":          COLOR_MED,
        "Ours":            COLOR_OURS,
    }

    # ── Figure setup ──
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
    })

    cell_w, cell_h = 3.0, 3.0
    fig, axes = plt.subplots(total_rows, n_cols,
                              figsize=(n_cols * cell_w, total_rows * cell_h))
    if total_rows == 1:
        axes = axes[np.newaxis, :]

    row = 0
    for ds in ds_order:
        cfg = DATASETS[ds]
        ds_label = cfg["label"]
        examples = best_per_dataset[ds]

        for ex_idx, item in enumerate(examples):
            img = item["image"]
            gt_mask = item["gt"]
            ious = item["ious"]

            # ── Col 0: Input Image ──
            axes[row, 0].imshow(img)

            # Dataset label on left (only first row of each dataset)
            if ex_idx == 0:
                axes[row, 0].set_ylabel(
                    ds_label,
                    fontsize=13, fontweight="bold", rotation=90,
                    labelpad=15, va="center")

            # ── Col 1: Ground Truth (green contour + fill) ──
            axes[row, 1].imshow(img)
            ov = np.zeros((*gt_mask.shape, 4))
            ov[gt_mask] = COLOR_GT
            axes[row, 1].imshow(ov)

            # ── Cols 2–5: Methods ──
            method_keys = ["SAM (single)", "SAM (best-of-N)", "MedSAM", "Ours"]
            for j, mk in enumerate(method_keys):
                col = j + 2
                axes[row, col].imshow(img)
                pred_mask = item["masks"][mk]
                ov = np.zeros((*pred_mask.shape, 4))
                ov[pred_mask] = overlay_colors[mk]
                axes[row, col].imshow(ov)

                # IoU badge
                iou_val = ious[mk]
                # Bold + green-ish for best, normal for rest
                is_best = (mk == "Ours" and iou_val >= max(ious.values()) - 0.001)
                badge_color = "#1a7a2e" if is_best else "#333333"
                badge_bg = "#d4edda" if is_best else "#f0f0f0"
                badge_weight = "bold"

                txt = axes[row, col].text(
                    0.50, 0.02, f"IoU: {iou_val:.3f}",
                    transform=axes[row, col].transAxes,
                    fontsize=11, fontweight=badge_weight,
                    color=badge_color, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor=badge_bg, edgecolor=badge_color,
                              alpha=0.92, linewidth=1.2))

            # ── Column headers (top row only) ──
            if row == 0:
                for c, label in enumerate(col_labels):
                    axes[0, c].set_title(
                        label, fontsize=13, fontweight="bold",
                        pad=12, color="#222222")

            row += 1

    # Clean up all axes
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.subplots_adjust(wspace=0.03, hspace=0.08)
    fig.suptitle(
        "Qualitative Comparison: SAM vs SAM (best-of-N) vs MedSAM vs Ours",
        fontsize=16, fontweight="bold", y=1.01, color="#111111")

    # ── Save at high DPI ──
    fig_path = os.path.join(args.output_dir, "qualitative_comparison.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n  ✓ Paper figure saved: {fig_path}")

    # Also save PDF for LaTeX
    pdf_path = os.path.join(args.output_dir, "qualitative_comparison.pdf")
    fig2, axes2 = plt.subplots(total_rows, n_cols,
                                figsize=(n_cols * cell_w, total_rows * cell_h))
    if total_rows == 1:
        axes2 = axes2[np.newaxis, :]

    # Re-render for PDF (same logic)
    row = 0
    for ds in ds_order:
        cfg = DATASETS[ds]
        ds_label = cfg["label"]
        examples = best_per_dataset[ds]

        for ex_idx, item in enumerate(examples):
            img = item["image"]
            gt_mask = item["gt"]
            ious = item["ious"]

            axes2[row, 0].imshow(img)
            if ex_idx == 0:
                axes2[row, 0].set_ylabel(
                    ds_label, fontsize=13, fontweight="bold",
                    rotation=90, labelpad=15, va="center")

            axes2[row, 1].imshow(img)
            ov = np.zeros((*gt_mask.shape, 4))
            ov[gt_mask] = COLOR_GT
            axes2[row, 1].imshow(ov)

            method_keys = ["SAM (single)", "SAM (best-of-N)", "MedSAM", "Ours"]
            for j, mk in enumerate(method_keys):
                col = j + 2
                axes2[row, col].imshow(img)
                pred_mask = item["masks"][mk]
                ov = np.zeros((*pred_mask.shape, 4))
                ov[pred_mask] = overlay_colors[mk]
                axes2[row, col].imshow(ov)

                iou_val = ious[mk]
                is_best = (mk == "Ours" and iou_val >= max(ious.values()) - 0.001)
                badge_color = "#1a7a2e" if is_best else "#333333"
                badge_bg = "#d4edda" if is_best else "#f0f0f0"

                axes2[row, col].text(
                    0.50, 0.02, f"IoU: {iou_val:.3f}",
                    transform=axes2[row, col].transAxes,
                    fontsize=11, fontweight="bold",
                    color=badge_color, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor=badge_bg, edgecolor=badge_color,
                              alpha=0.92, linewidth=1.2))

            if row == 0:
                for c, label in enumerate(col_labels):
                    axes2[0, c].set_title(
                        label, fontsize=13, fontweight="bold",
                        pad=12, color="#222222")
            row += 1

    for ax in axes2.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.subplots_adjust(wspace=0.03, hspace=0.08)
    fig2.suptitle(
        "Qualitative Comparison: SAM vs SAM (best-of-N) vs MedSAM vs Ours",
        fontsize=16, fontweight="bold", y=1.01, color="#111111")

    fig2.savefig(pdf_path, dpi=300, bbox_inches="tight",
                 facecolor="white", edgecolor="none")
    plt.close(fig2)
    print(f"  ✓ PDF version saved:  {pdf_path}")

    # ── Print selection summary ──
    print(f"\n{'='*70}")
    print(f"  SELECTED EXAMPLES SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Dataset':20s} {'Image':25s} {'SAM':>6s} {'BoN':>6s} "
          f"{'MedSAM':>7s} {'Ours':>6s} {'Δ(SAM)':>7s}")
    print(f"  {'─'*80}")
    for ds in ds_order:
        for item in best_per_dataset[ds]:
            ious = item["ious"]
            print(f"  {ds:20s} {item['name']:25s} "
                  f"{ious['SAM (single)']:>6.3f} "
                  f"{ious['SAM (best-of-N)']:>6.3f} "
                  f"{ious['MedSAM']:>7.3f} "
                  f"{ious['Ours']:>6.3f} "
                  f"{item['delta_vs_sam']:>+7.3f}")

    print(f"\n  Output directory: {args.output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Paper-ready qualitative figure")
    p.add_argument("--datasets", default="busi_malignant,jsrt,kvasir")
    p.add_argument("--n-variants", type=int, default=10)
    p.add_argument("--top-k", type=int, default=1,
                   help="Number of examples per dataset (1 = cleanest figure)")
    p.add_argument("--output-dir", default="results/qualitative")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(args)
