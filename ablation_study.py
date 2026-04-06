#!/usr/bin/env python3
"""
Ablation Study: Cumulative Component Impact
=============================================
Runs 5 configurations on each dataset to measure the marginal gain
of every pipeline component.

  A0: SAM baseline              (single box → single mask)
  A1: + Confidence gate         (keep SAM if confident, else try harder)
  A2: + Multi-mask search       (generate variants, pick best SAM score)
  A3: + Consistency routing     (use global consistency to decide search vs refine)
  A4: Full pipeline             (+ safeguard: only upgrade if meaningfully different)

Produces:
  1. ablation_results.json      — raw per-image + summary
  2. ablation_table.tex         — paper-ready LaTeX table
  3. ablation_table.csv         — CSV for plotting
  4. fig_ablation.png / .pdf    — grouped bar chart

Usage:
    python ablation_study.py --datasets busi_malignant,jsrt,kvasir
    python ablation_study.py --datasets busi_malignant --device cuda
"""

import os, sys, time, json, argparse, random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, "/home/mi3dr/SCCS/sccs")
from segment_anything import sam_model_registry, SamPredictor

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
        "label": "BUSI",
    },
    "jsrt": {
        "image_dir": f"{BASE}/jsrt/jpg",
        "mask_dir":  f"{BASE}/jsrt/masks",
        "label": "JSRT",
    },
    "kvasir": {
        "image_dir": f"{BASE}/Kvasir-SEG/images",
        "mask_dir":  f"{BASE}/Kvasir-SEG/masks",
        "label": "Kvasir-SEG",
    },
}

SAM_H_CKPT = f"{BASE}/sam_vit_h_4b8939.pth"

# Pipeline hyperparams
CONF_GATE = 0.85
TAU_CONSISTENCY = 0.5
N_VARIANTS = 10


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


# ═══════════════════════════════════════════════════════════════
# ABLATION CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════

CONFIGS = [
    {
        "name": "A0: SAM baseline",
        "short": "SAM baseline",
        "use_conf_gate": False,
        "use_multimask": False,
        "use_consistency": False,
        "use_safeguard": False,
    },
    {
        "name": "A1: + Conf. gate",
        "short": "+ Conf. gate",
        "use_conf_gate": True,
        "use_multimask": False,
        "use_consistency": False,
        "use_safeguard": False,
    },
    {
        "name": "A2: + Multi-mask",
        "short": "+ Multi-mask",
        "use_conf_gate": True,
        "use_multimask": True,
        "use_consistency": False,
        "use_safeguard": False,
    },
    {
        "name": "A3: + Consistency",
        "short": "+ Consistency",
        "use_conf_gate": True,
        "use_multimask": True,
        "use_consistency": True,
        "use_safeguard": False,
    },
    {
        "name": "A4: Full pipeline",
        "short": "Full (Ours)",
        "use_conf_gate": True,
        "use_multimask": True,
        "use_consistency": True,
        "use_safeguard": True,
    },
]


def run_config(cfg, predictor, image, gt, gt_box, n_variants=10):
    """
    Run a single ablation configuration on one image.
    Returns the predicted mask.
    """
    h, w = image.shape[:2]

    # Step 1: Always get SAM single prediction
    predictor.set_image(image)
    pred_masks, pred_scores, _ = predictor.predict(
        box=gt_box[None, :], multimask_output=False)
    sam_mask = pred_masks[0].astype(bool)
    sam_score = float(pred_scores[0])

    # ── A0: SAM baseline — just return single mask ──
    if not cfg["use_conf_gate"]:
        return sam_mask

    # ── A1: Confidence gate — if SAM confident, keep it ──
    if sam_score >= CONF_GATE:
        return sam_mask

    # SAM not confident — what we do next depends on enabled components

    # ── A1 only (no multimask): try multimask from same box, pick best ──
    if not cfg["use_multimask"]:
        # Without multi-mask search, just use multimask_output=True on original box
        masks_multi, scores_multi, _ = predictor.predict(
            box=gt_box[None, :], multimask_output=True)
        best_idx = int(np.argmax(scores_multi))
        return masks_multi[best_idx].astype(bool)

    # ── A2: Multi-mask search — generate variants, run SAM, pick best score ──
    variants = generate_variants_from_box(gt_box, h, w, n_variants=n_variants)
    sam_results = run_sam_variants(predictor, image, variants)

    if not cfg["use_consistency"]:
        # A2: Just pick highest-confidence mask from all variants
        if sam_results:
            best_idx = max(range(len(sam_results)),
                           key=lambda i: sam_results[i]["sam_score"])
            return sam_results[best_idx]["mask"]
        return sam_mask

    # ── A3: Consistency routing — compute global consistency, route to search/refine ──
    consistency = compute_global_consistency(sam_results)

    if consistency < TAU_CONSISTENCY:
        candidate, _ = spatial_search(predictor, image, gt_box, h, w)
        route = "search"
    else:
        candidate, _ = refine_from_mask(predictor, image, sam_mask)
        route = "refine"

    if not cfg["use_safeguard"]:
        # A3: Accept candidate blindly (no safeguard check)
        if candidate is not None and candidate.any():
            return candidate
        return sam_mask

    # ── A4: Full pipeline — safeguard: only upgrade if meaningfully different ──
    if candidate is not None and candidate.any():
        cand_vs_sam = mask_iou(candidate, sam_mask)
        if cand_vs_sam > 0.9:
            return sam_mask  # too similar, keep SAM
        else:
            return candidate  # meaningfully different, upgrade
    return sam_mask


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_ablation_bars(results, output_dir):
    """Grouped bar chart: configs × datasets for IoU, Dice, Halluc."""

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })

    datasets = list(results.keys())
    config_names = [c["short"] for c in CONFIGS]
    n_configs = len(config_names)
    n_ds = len(datasets)

    # Colors for each config stage (grey → blue → orange → teal → green)
    colors = ["#95a5a6", "#3498db", "#e67e22", "#1abc9c", "#27ae60"]

    metrics = ["iou", "dice", "halluc"]
    metric_labels = ["IoU ↑", "Dice ↑", "Hallucination Rate ↓"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    x = np.arange(n_ds)
    bar_w = 0.14

    for mi, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[mi]

        for ci, cname in enumerate(config_names):
            vals = []
            for ds in datasets:
                v = results[ds]["summary"].get(CONFIGS[ci]["name"], {}).get(metric, 0)
                vals.append(v)

            offset = (ci - (n_configs - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, bar_w,
                          label=cname if mi == 0 else "",
                          color=colors[ci], edgecolor="white", linewidth=0.6)

            # Value labels on bars
            for b, v in zip(bars, vals):
                fmt = f"{v:.1%}" if metric == "halluc" else f"{v:.3f}"
                y_off = b.get_height() + (0.005 if metric != "halluc" else 0.003)
                ax.text(b.get_x() + b.get_width() / 2, y_off,
                        fmt, ha="center", va="bottom", fontsize=7,
                        fontweight="bold", rotation=0)

        ds_labels = [DATASETS[d]["label"] for d in datasets]
        ax.set_xticks(x)
        ax.set_xticklabels(ds_labels, fontweight="bold", fontsize=11)
        ax.set_title(mlabel, fontweight="bold", fontsize=13)

        if metric == "halluc":
            max_h = max(results[ds]["summary"].get(c["name"], {}).get("halluc", 0)
                        for ds in datasets for c in CONFIGS)
            ax.set_ylim(0, max(0.25, max_h * 1.4))
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        else:
            all_vals = [results[ds]["summary"].get(c["name"], {}).get(metric, 0)
                        for ds in datasets for c in CONFIGS]
            low = min(all_vals) - 0.04
            ax.set_ylim(max(0, low), 1.02)

    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.95,
                   ncol=1, title="Configuration", title_fontsize=10)

    fig.suptitle("Ablation Study: Cumulative Component Impact",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    png_path = os.path.join(output_dir, "fig_ablation.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Bar chart  → {png_path}")

    pdf_path = os.path.join(output_dir, "fig_ablation.pdf")
    # Re-render for PDF
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5.5))
    x = np.arange(n_ds)
    for mi, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes2[mi]
        for ci, cname in enumerate(config_names):
            vals = [results[ds]["summary"].get(CONFIGS[ci]["name"], {}).get(metric, 0)
                    for ds in datasets]
            offset = (ci - (n_configs - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, bar_w,
                          label=cname if mi == 0 else "",
                          color=colors[ci], edgecolor="white", linewidth=0.6)
            for b, v in zip(bars, vals):
                fmt = f"{v:.1%}" if metric == "halluc" else f"{v:.3f}"
                y_off = b.get_height() + (0.005 if metric != "halluc" else 0.003)
                ax.text(b.get_x() + b.get_width() / 2, y_off,
                        fmt, ha="center", va="bottom", fontsize=7,
                        fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([DATASETS[d]["label"] for d in datasets],
                           fontweight="bold", fontsize=11)
        ax.set_title(mlabel, fontweight="bold", fontsize=13)
        ax.grid(alpha=0.25, linestyle="--")

        if metric == "halluc":
            max_h = max(results[ds]["summary"].get(c["name"], {}).get("halluc", 0)
                        for ds in datasets for c in CONFIGS)
            ax.set_ylim(0, max(0.25, max_h * 1.4))
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        else:
            all_vals = [results[ds]["summary"].get(c["name"], {}).get(metric, 0)
                        for ds in datasets for c in CONFIGS]
            ax.set_ylim(max(0, min(all_vals) - 0.04), 1.02)

    axes2[0].legend(loc="lower left", fontsize=9, framealpha=0.95,
                    ncol=1, title="Configuration", title_fontsize=10)
    fig2.suptitle("Ablation Study: Cumulative Component Impact",
                  fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2.savefig(pdf_path, dpi=300, bbox_inches="tight",
                 facecolor="white", edgecolor="none")
    plt.close(fig2)
    print(f"  ✓ PDF chart  → {pdf_path}")


def save_latex_table(results, output_dir):
    """Paper-ready LaTeX ablation table."""
    datasets = list(results.keys())
    n_ds = len(datasets)

    tex_path = os.path.join(output_dir, "ablation_table.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Ablation study: cumulative impact of each component. "
                "Each row adds one component to the pipeline.}\n")
        f.write("\\label{tab:ablation}\n")

        # Column spec: config name + 3 metrics per dataset
        col_spec = "l" + "ccc" * n_ds
        f.write(f"\\resizebox{{\\linewidth}}{{!}}{{%\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")

        # Header row 1: dataset names
        header1 = "Configuration"
        for ds in datasets:
            label = DATASETS[ds]["label"]
            header1 += f" & \\multicolumn{{3}}{{c}}{{{label}}}"
        f.write(header1 + " \\\\\n")

        # Header row 2: metric names
        cmidrules = ""
        for i in range(n_ds):
            start = 2 + i * 3
            end = start + 2
            cmidrules += f"\\cmidrule(lr){{{start}-{end}}} "
        f.write(cmidrules + "\n")

        header2 = ""
        for _ in datasets:
            header2 += " & IoU$\\uparrow$ & Dice$\\uparrow$ & Halluc$\\downarrow$"
        f.write(header2 + " \\\\\n\\midrule\n")

        # Find best values per dataset for bolding
        best = {}
        for ds in datasets:
            best[ds] = {"iou": 0, "dice": 0, "halluc": 1.0}
            for cfg in CONFIGS:
                s = results[ds]["summary"].get(cfg["name"], {})
                if s.get("iou", 0) > best[ds]["iou"]:
                    best[ds]["iou"] = s["iou"]
                if s.get("dice", 0) > best[ds]["dice"]:
                    best[ds]["dice"] = s["dice"]
                if s.get("halluc", 1) < best[ds]["halluc"]:
                    best[ds]["halluc"] = s["halluc"]

        # Data rows
        for ci, cfg in enumerate(CONFIGS):
            row = cfg["short"]
            for ds in datasets:
                s = results[ds]["summary"].get(cfg["name"], {})
                iou = s.get("iou", 0)
                dice = s.get("dice", 0)
                halluc = s.get("halluc", 0)

                iou_s = f"\\textbf{{{iou:.4f}}}" if abs(iou - best[ds]["iou"]) < 1e-5 else f"{iou:.4f}"
                dice_s = f"\\textbf{{{dice:.4f}}}" if abs(dice - best[ds]["dice"]) < 1e-5 else f"{dice:.4f}"
                halluc_s = f"\\textbf{{{halluc:.1%}}}" if abs(halluc - best[ds]["halluc"]) < 1e-5 else f"{halluc:.1%}"

                row += f" & {iou_s} & {dice_s} & {halluc_s}"

            f.write(row + " \\\\\n")

            # Add midrule after A0 (baseline) to visually separate
            if ci == 0:
                f.write("\\midrule\n")

        f.write("\\bottomrule\n\\end{tabular}}%\n\\end{table}\n")

    print(f"  ✓ LaTeX table → {tex_path}")


def save_csv(results, output_dir):
    """CSV for external plotting."""
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    with open(csv_path, "w") as f:
        f.write("dataset,config,iou,dice,hallucination\n")
        for ds in results:
            for cfg in CONFIGS:
                s = results[ds]["summary"].get(cfg["name"], {})
                f.write(f"{ds},{cfg['name']},{s.get('iou',0):.4f},"
                        f"{s.get('dice',0):.4f},{s.get('halluc',0):.4f}\n")
    print(f"  ✓ CSV         → {csv_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run(args):
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

    # ── Load SAM ViT-H ──
    print("[+] Loading SAM ViT-H...")
    sam_h = sam_model_registry["vit_h"](checkpoint=SAM_H_CKPT).to(device)
    predictor = SamPredictor(sam_h)
    print("[+] SAM loaded.")

    torch.load = _orig_load

    all_results = {}

    for ds_name in args.datasets.split(","):
        ds_name = ds_name.strip()
        if ds_name not in DATASETS:
            print(f"[!] Unknown: {ds_name}"); continue
        cfg_ds = DATASETS[ds_name]

        image_paths = sorted([p for p in Path(cfg_ds["image_dir"]).glob("*.png")
                              if "_mask" not in p.stem]) + \
                      sorted([p for p in Path(cfg_ds["image_dir"]).glob("*.jpg")
                              if "_mask" not in p.stem])

        print(f"\n{'='*70}")
        print(f"  {ds_name} — {len(image_paths)} images × {len(CONFIGS)} configs")
        print(f"{'='*70}")

        accum = {c["name"]: {"ious": [], "dices": [], "hallucs": []}
                 for c in CONFIGS}
        per_image = []
        t0 = time.time()
        n_proc = 0

        for idx, img_path in enumerate(image_paths):
            name = img_path.stem
            gt_path = find_gt_mask(name, cfg_ds["mask_dir"], img_path)
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

            img_row = {"image": name}

            for cfg in CONFIGS:
                pred_mask = run_config(cfg, predictor, image, gt, gt_box,
                                       n_variants=N_VARIANTS)
                ev = evaluate_mask(pred_mask, gt)
                accum[cfg["name"]]["ious"].append(ev["iou"])
                accum[cfg["name"]]["dices"].append(ev["dice"])
                accum[cfg["name"]]["hallucs"].append(ev["hallucination"])
                img_row[cfg["name"]] = {"iou": round(ev["iou"], 4),
                                         "dice": round(ev["dice"], 4)}

            per_image.append(img_row)
            n_proc += 1

            if n_proc % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / n_proc * (len(image_paths) - idx - 1)
                a0 = img_row[CONFIGS[0]["name"]]["iou"]
                a4 = img_row[CONFIGS[-1]["name"]]["iou"]
                print(f"  [{idx+1}/{len(image_paths)}] {name:25s}  "
                      f"A0={a0:.3f}  A4={a4:.3f}  Δ={a4-a0:+.3f}  "
                      f"| ETA {eta/60:.0f}m")

        dt = time.time() - t0
        print(f"\n  {ds_name}: {n_proc} images, {dt/60:.1f} min")

        # ── Summary ──
        summary = {}
        print(f"\n  {'Config':25s} {'IoU↑':>8s} {'Dice↑':>8s} {'Halluc↓':>8s}  {'ΔIoU':>7s}")
        print(f"  {'─'*60}")

        baseline_iou = None
        for cfg in CONFIGS:
            cname = cfg["name"]
            if accum[cname]["ious"]:
                miou = float(np.mean(accum[cname]["ious"]))
                mdice = float(np.mean(accum[cname]["dices"]))
                mhalluc = float(np.mean(accum[cname]["hallucs"]))
                summary[cname] = {
                    "iou": round(miou, 4),
                    "dice": round(mdice, 4),
                    "halluc": round(mhalluc, 4),
                }
                if baseline_iou is None:
                    baseline_iou = miou
                delta = miou - baseline_iou
                print(f"  {cname:25s} {miou:>8.4f} {mdice:>8.4f} {mhalluc:>7.1%}  {delta:>+7.4f}")

        all_results[ds_name] = {
            "n_images": n_proc,
            "time_min": round(dt / 60, 1),
            "summary": summary,
            "per_image": per_image,
        }

    # ═══════════════════════════════════════════════════════════
    # SAVE OUTPUTS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  Saving outputs...")
    print(f"{'='*70}")

    # JSON (full results)
    json_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: bool(x) if isinstance(x, np.bool_) else x)
    print(f"  ✓ JSON        → {json_path}")

    # CSV
    save_csv(all_results, args.output_dir)

    # LaTeX table
    save_latex_table(all_results, args.output_dir)

    # Bar chart
    plot_ablation_bars(all_results, args.output_dir)

    # ── Final summary across datasets ──
    print(f"\n{'='*70}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Dataset':15s}", end="")
    for cfg in CONFIGS:
        print(f"  {cfg['short']:>13s}", end="")
    print()
    print(f"  {'─'*80}")

    for ds in all_results:
        print(f"  {ds:15s}", end="")
        for cfg in CONFIGS:
            iou = all_results[ds]["summary"].get(cfg["name"], {}).get("iou", 0)
            print(f"  {iou:>13.4f}", end="")
        print()

    # Show marginal gains
    print(f"\n  Marginal IoU gains (each component's contribution):")
    for ds in all_results:
        gains = []
        prev = 0
        for i, cfg in enumerate(CONFIGS):
            cur = all_results[ds]["summary"].get(cfg["name"], {}).get("iou", 0)
            if i == 0:
                prev = cur
                continue
            gains.append(f"{cfg['short']}: {cur - prev:+.4f}")
            prev = cur
        print(f"  {ds:15s}  {' | '.join(gains)}")

    print(f"\n  All outputs in: {args.output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ablation study: cumulative component impact")
    p.add_argument("--datasets", default="busi_malignant,jsrt,kvasir")
    p.add_argument("--output-dir", default="results/ablation")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(args)
