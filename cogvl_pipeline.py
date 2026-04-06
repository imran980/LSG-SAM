"""
CogVL Pipeline — Semantic-Stability Mask Selection for Medical Imaging
======================================================================
SAM generates K masks from prompt variants → BiomedCLIP verifies semantic
correctness → stability + semantic agreement selects the trustworthy mask.

Target: CogVL CVPR Workshop

Usage:
    python cogvl_pipeline.py --datasets busi_malignant
    python cogvl_pipeline.py --datasets all
"""

import os, argparse, json, warnings, time, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import open_clip

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

BASE = "/home/mi3dr/SCCS/sccs"

DATASETS = {
    "busi_malignant": {
        "image_dir": f"{BASE}/Dataset_BUSI_with_GT/malignant",
        "mask_dir":  f"{BASE}/Dataset_BUSI_with_GT/malignant",
        "target_prompts": [
            "a breast ultrasound showing a tumor",
            "a malignant mass in ultrasound",
            "hypoechoic lesion with irregular margins",
        ],
        "non_target_prompts": [
            "normal skin tissue",
            "rib bone in x-ray",
            "lung field",
            "gel coupling artifact in ultrasound",
            "a photograph of a house",
        ],
    },
    "busi_benign": {
        "image_dir": f"{BASE}/Dataset_BUSI_with_GT/benign",
        "mask_dir":  f"{BASE}/Dataset_BUSI_with_GT/benign",
        "target_prompts": [
            "a breast ultrasound showing a lesion",
            "a benign cyst in ultrasound",
            "well-circumscribed mass in breast ultrasound",
        ],
        "non_target_prompts": [
            "normal skin tissue",
            "rib bone in x-ray",
            "lung field",
            "gel coupling artifact in ultrasound",
            "a photograph of a house",
        ],
    },
    "kvasir": {
        "image_dir": f"{BASE}/Kvasir-SEG/images",
        "mask_dir":  f"{BASE}/Kvasir-SEG/masks",
        "target_prompts": [
            "a colorectal polyp in endoscopy",
            "an abnormal mucosal growth",
            "a protruding lesion in the colon",
        ],
        "non_target_prompts": [
            "healthy pink mucosa",
            "endoscope light reflection",
            "stool residue",
            "normal colon wall",
            "a photograph of a house",
        ],
    },
}

SAM_CHECKPOINT = f"{BASE}/sam_vit_h_4b8939.pth"
BIOMEDCLIP_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


# ═══════════════════════════════════════════════════════════════
# 1. PROMPT VARIANT GENERATION
# ═══════════════════════════════════════════════════════════════

def gt_bbox(gt_mask):
    ys, xs = np.where(gt_mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def gt_center_box(gt_mask, h, w, scale=1.2):
    bbox = gt_bbox(gt_mask)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    bw, bh = (bbox[2] - bbox[0]) * scale, (bbox[3] - bbox[1]) * scale
    return np.array([max(0, cx - bw/2), max(0, cy - bh/2),
                     min(w-1, cx + bw/2), min(h-1, cy + bh/2)])


def generate_k_variants(gt_mask, h, w, n_variants=10):
    """Generate K box prompt variants with meaningful diversity:
    - Scale variation (0.8x to 1.5x)
    - Position jitter (up to 15% of box dimensions)
    - Aspect ratio perturbation
    Returns list of K boxes."""
    bbox = gt_bbox(gt_mask)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    gt_w, gt_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    variants = []
    # First: the clean 1.2x scale box (matches SAM single baseline)
    variants.append(gt_center_box(gt_mask, h, w, scale=1.2))

    # Remaining: diverse perturbations
    for _ in range(n_variants - 1):
        # Scale: 0.8x to 1.5x
        scale = np.random.uniform(0.8, 1.5)
        # Aspect ratio perturbation: ±20%
        ar = np.random.uniform(0.8, 1.2)
        bw = gt_w * scale * ar
        bh = gt_h * scale / ar
        # Position jitter: up to 15% of box dimensions
        jx = np.random.uniform(-0.15, 0.15) * gt_w
        jy = np.random.uniform(-0.15, 0.15) * gt_h
        ncx, ncy = cx + jx, cy + jy
        box = np.array([
            max(0, ncx - bw / 2), max(0, ncy - bh / 2),
            min(w - 1, ncx + bw / 2), min(h - 1, ncy + bh / 2)
        ])
        variants.append(box)

    return variants


# ═══════════════════════════════════════════════════════════════
# 2. SAM INFERENCE
# ═══════════════════════════════════════════════════════════════

def run_sam_variants(predictor, image, variants):
    """Run SAM for each variant box with multimask_output=True.
    Each prompt yields 3 candidate masks → dramatically expands pool."""
    results = []
    predictor.set_image(image)
    for box in variants:
        masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
        # Add ALL 3 masks as separate candidates
        for i in range(len(masks)):
            results.append({
                "mask": masks[i].astype(bool),
                "sam_score": float(scores[i]),
                "box": box.tolist(),
            })
    return results


# ═══════════════════════════════════════════════════════════════
# 3. STABILITY SCORING
# ═══════════════════════════════════════════════════════════════

def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def dice_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return float(2 * inter / total) if total > 0 else (1.0 if inter == 0 else 0.0)


def compute_stability(sam_results):
    """Stability = mean pairwise IoU among all K masks.
    If masks agree → prediction is structurally stable."""
    masks = [r["mask"] for r in sam_results]
    n = len(masks)
    if n < 2:
        return 1.0
    ious = []
    for i in range(n):
        for j in range(i + 1, n):
            ious.append(mask_iou(masks[i], masks[j]))
    return float(np.mean(ious))


def compute_per_mask_stability(sam_results):
    """Per-mask stability: average IoU of mask_i vs all other masks."""
    masks = [r["mask"] for r in sam_results]
    n = len(masks)
    stabilities = []
    for i in range(n):
        if n < 2:
            stabilities.append(1.0)
            continue
        ious = [mask_iou(masks[i], masks[j]) for j in range(n) if j != i]
        stabilities.append(float(np.mean(ious)))
    return stabilities


# ═══════════════════════════════════════════════════════════════
# 4. BIOMEDCLIP SEMANTIC SCORING
# ═══════════════════════════════════════════════════════════════

def create_masked_crop(image, mask):
    """Masked crop: image * mask (isolate the segmented structure)."""
    crop = image.copy()
    crop[~mask] = 0
    # Crop to bounding box of mask
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    return crop[y1:y2, x1:x2]


def create_context_crop(image, mask, margin_frac=0.15):
    """Context crop: bbox(mask) + 15% margin.
    Keeps enough context without overwhelming with background."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    h, w = image.shape[:2]
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    bh, bw = y2 - y1, x2 - x1
    margin_y = int(bh * margin_frac)
    margin_x = int(bw * margin_frac)
    y1 = max(0, y1 - margin_y)
    y2 = min(h, y2 + margin_y)
    x1 = max(0, x1 - margin_x)
    x2 = min(w, x2 + margin_x)
    return image[y1:y2, x1:x2]


def create_overlay_crop(image, mask, margin_frac=0.15):
    """Overlay crop: mask boundary drawn on context crop.
    Gives BiomedCLIP both structure AND spatial context."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    h, w = image.shape[:2]
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    bh, bw = y2 - y1, x2 - x1
    margin_y = int(bh * margin_frac)
    margin_x = int(bw * margin_frac)
    cy1 = max(0, y1 - margin_y)
    cy2 = min(h, y2 + margin_y)
    cx1 = max(0, x1 - margin_x)
    cx2 = min(w, x2 + margin_x)
    overlay = image[cy1:cy2, cx1:cx2].copy()
    # Dim outside mask region
    local_mask = mask[cy1:cy2, cx1:cx2]
    overlay[~local_mask] = (overlay[~local_mask] * 0.4).astype(np.uint8)
    return overlay


def encode_crop(crop, clip_model, preprocess, device):
    """Encode a crop with BiomedCLIP → normalized feature vector."""
    if crop is None or crop.size == 0:
        return None
    pil = Image.fromarray(crop)
    inp = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(inp)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def compute_semantic_margin(image, mask, clip_model, preprocess, tokenizer,
                            target_prompts, non_target_prompts, device):
    """Compute semantic_margin = max(sim(target)) - max(sim(non_target)).
    Averages scores from masked crop, context crop, and overlay crop."""
    masked_crop = create_masked_crop(image, mask)
    context_crop = create_context_crop(image, mask)
    overlay_crop = create_overlay_crop(image, mask)

    # Encode text prompts
    all_prompts = target_prompts + non_target_prompts
    tokens = tokenizer(all_prompts).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    n_target = len(target_prompts)

    margins = []
    for crop in [masked_crop, context_crop, overlay_crop]:
        feat = encode_crop(crop, clip_model, preprocess, device)
        if feat is None:
            continue
        sims = (feat @ text_feats.T).squeeze(0)
        target_max = sims[:n_target].max().item()
        non_target_max = sims[n_target:].max().item()
        margins.append(target_max - non_target_max)

    if not margins:
        return 0.0
    return float(np.mean(margins))


# ═══════════════════════════════════════════════════════════════
# 5. COGVL SELECTION + RELIABILITY FLAG
# ═══════════════════════════════════════════════════════════════

def cogvl_select(sam_results, semantic_margins, stabilities,
                 alpha=0.6, beta=0.4,
                 tau_semantic=0.1, tau_stability=0.5):
    """
    final_score_i = alpha * semantic_margin_i + beta * stability_i
    best_mask = argmax(final_score_i)
    unreliable if semantic_margin < tau1 OR stability < tau2
    """
    final_scores = []
    for i in range(len(sam_results)):
        fs = alpha * semantic_margins[i] + beta * stabilities[i]
        final_scores.append(fs)

    best_idx = int(np.argmax(final_scores))
    best_semantic = semantic_margins[best_idx]
    best_stability = stabilities[best_idx]
    unreliable = (best_semantic < tau_semantic) or (best_stability < tau_stability)

    return {
        "best_idx": best_idx,
        "best_mask": sam_results[best_idx]["mask"],
        "best_score": final_scores[best_idx],
        "best_semantic": best_semantic,
        "best_stability": best_stability,
        "unreliable": unreliable,
        "all_scores": final_scores,
    }


# ═══════════════════════════════════════════════════════════════
# 6. EVALUATION + BASELINES
# ═══════════════════════════════════════════════════════════════

def evaluate_mask(pred, gt):
    iou = mask_iou(pred, gt)
    dc = dice_score(pred, gt)
    halluc = (iou < 0.05) and pred.any()
    return {"iou": iou, "dice": dc, "hallucination": halluc}


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
# 7. MAIN RUNNER
# ═══════════════════════════════════════════════════════════════

def run_benchmark(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.datasets == "all":
        ds_names = list(DATASETS.keys())
    else:
        ds_names = [d.strip() for d in args.datasets.split(",")]

    # ── Load SAM ──
    print(f"[CogVL] Loading SAM ViT-H...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(args.device)
    predictor = SamPredictor(sam)

    # ── Load BiomedCLIP ──
    print(f"[CogVL] Loading BiomedCLIP...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_NAME)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_NAME)
    clip_model = clip_model.to(args.device)
    clip_model.eval()
    print(f"[CogVL] Models loaded.")

    method_names = ["SAM (single)", "SAM (best conf)", f"Random-{args.top_k}",
                    "CogVL (ours)", "Oracle"]

    all_dataset_results = {}

    for ds_name in ds_names:
        if ds_name not in DATASETS:
            print(f"[!] Unknown dataset: {ds_name}"); continue
        cfg = DATASETS[ds_name]

        image_paths = sorted([p for p in Path(cfg["image_dir"]).glob("*.png")
                              if "_mask" not in p.stem]) + \
                      sorted([p for p in Path(cfg["image_dir"]).glob("*.jpg")
                              if "_mask" not in p.stem])

        print(f"\n{'='*80}")
        print(f"  DATASET: {ds_name} — {len(image_paths)} images")
        print(f"{'='*80}")

        accum = {m: {"ious": [], "dices": [], "hallucs": []} for m in method_names}
        n_unreliable = 0
        n_unreliable_correct = 0  # unreliable flag catches true bad predictions
        per_image_results = []
        t0 = time.time()
        n_processed = 0

        for idx, img_path in enumerate(image_paths):
            name = img_path.stem
            gt_path = find_gt_mask(name, cfg["mask_dir"], img_path)
            if not gt_path:
                continue

            image = np.array(Image.open(img_path).convert("RGB"))
            gt = np.array(Image.open(gt_path).convert("L")) > 127
            h, w = image.shape[:2]
            if gt.shape != (h, w):
                gt = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((w, h))) > 127
            if not gt.any():
                continue

            # ── Step 1: Generate K prompt variants ──
            variants = generate_k_variants(gt, h, w)

            # ── Step 2: Run SAM for each ──
            sam_results = run_sam_variants(predictor, image, variants)

            # ── Step 3: Compute stability ──
            per_mask_stab = compute_per_mask_stability(sam_results)
            global_stab = compute_stability(sam_results)

            # ── Step 4+5: BiomedCLIP semantic scoring per mask ──
            semantic_margins = []
            for r in sam_results:
                if not r["mask"].any():
                    semantic_margins.append(0.0)
                    continue
                margin = compute_semantic_margin(
                    image, r["mask"], clip_model, preprocess, tokenizer,
                    cfg["target_prompts"], cfg["non_target_prompts"], args.device)
                semantic_margins.append(margin)

            # ── Step 6: CogVL selection ──
            cogvl = cogvl_select(
                sam_results, semantic_margins, per_mask_stab,
                alpha=args.alpha, beta=args.beta,
                tau_semantic=args.tau_semantic, tau_stability=args.tau_stability)

            # ── Baselines ──
            img_results = {"image": name}

            def _eval(method_name, pred):
                ev = evaluate_mask(pred, gt)
                img_results[method_name] = ev
                accum[method_name]["ious"].append(ev["iou"])
                accum[method_name]["dices"].append(ev["dice"])
                accum[method_name]["hallucs"].append(ev["hallucination"])
                return ev

            # 1. SAM single (first variant = original box)
            _eval("SAM (single)", sam_results[0]["mask"])

            # 2. SAM best confidence
            best_conf_idx = max(range(len(sam_results)),
                                key=lambda i: sam_results[i]["sam_score"])
            _eval("SAM (best conf)", sam_results[best_conf_idx]["mask"])

            # 3. Random-k
            k = min(args.top_k, len(sam_results))
            rand_subset = random.sample(range(len(sam_results)), k)
            rand_best = max(rand_subset, key=lambda i: sam_results[i]["sam_score"])
            _eval(f"Random-{args.top_k}", sam_results[rand_best]["mask"])

            # 4. CogVL (ours)
            ev_cogvl = _eval("CogVL (ours)", cogvl["best_mask"])

            # 5. Oracle (best GT IoU)
            oracle_ious = [mask_iou(r["mask"], gt) for r in sam_results]
            oracle_idx = int(np.argmax(oracle_ious))
            _eval("Oracle", sam_results[oracle_idx]["mask"])

            # Track reliability
            if cogvl["unreliable"]:
                n_unreliable += 1
                if ev_cogvl["iou"] < 0.3:  # truly bad prediction
                    n_unreliable_correct += 1

            per_image_results.append({
                **img_results,
                "cogvl_semantic": round(cogvl["best_semantic"], 4),
                "cogvl_stability": round(cogvl["best_stability"], 4),
                "cogvl_score": round(cogvl["best_score"], 4),
                "cogvl_unreliable": cogvl["unreliable"],
                "global_stability": round(global_stab, 4),
            })
            n_processed += 1

            # Progress
            elapsed = time.time() - t0
            eta = elapsed / n_processed * (len(image_paths) - idx - 1)
            ci = img_results["CogVL (ours)"]["iou"]
            si = img_results["SAM (single)"]["iou"]
            flag = "⚠" if cogvl["unreliable"] else "✓"
            print(f"  [{idx+1}/{len(image_paths)}] {name} — "
                  f"SAM={si:.3f} CogVL={ci:.3f} {flag} "
                  f"sem={cogvl['best_semantic']:.2f} stab={cogvl['best_stability']:.2f} | "
                  f"ETA {eta/60:.0f}m")

        # ── Dataset Summary ──
        total_time = time.time() - t0
        print(f"\n{'─'*80}")
        print(f"  {ds_name.upper()} — {n_processed} images, {total_time/60:.1f} min")
        print(f"{'─'*80}")
        print(f"  {'Method':25s} {'IoU↑':>8s} {'Dice↑':>8s} {'Halluc↓':>8s}")
        print(f"  {'─'*51}")

        summary = {}
        for m in method_names:
            if not accum[m]["ious"]:
                continue
            miou = float(np.mean(accum[m]["ious"]))
            mdice = float(np.mean(accum[m]["dices"]))
            hrate = float(np.mean(accum[m]["hallucs"]))
            summary[m] = {"iou": round(miou, 4), "dice": round(mdice, 4),
                          "hallucination_rate": round(hrate, 4)}
            print(f"  {m:25s} {miou:>8.4f} {mdice:>8.4f} {hrate:>7.1%}")

        # Reliability report
        print(f"\n  Reliability Flag:")
        print(f"    Flagged as unreliable: {n_unreliable}/{n_processed} "
              f"({n_unreliable/n_processed:.1%})" if n_processed > 0 else "    N/A")
        if n_unreliable > 0:
            prec = n_unreliable_correct / n_unreliable
            print(f"    Flag precision (truly bad): {n_unreliable_correct}/{n_unreliable} "
                  f"({prec:.1%})")
        print(f"{'─'*80}")

        # Save
        ds_result = {
            "dataset": ds_name,
            "n_images": n_processed,
            "time_minutes": round(total_time / 60, 1),
            "config": {"alpha": args.alpha, "beta": args.beta,
                       "tau_semantic": args.tau_semantic, "tau_stability": args.tau_stability,
                       "top_k": args.top_k},
            "summary": summary,
            "reliability": {
                "n_flagged": n_unreliable,
                "flag_precision": round(n_unreliable_correct / n_unreliable, 4) if n_unreliable > 0 else None,
            },
            "per_image": per_image_results,
        }
        out_path = os.path.join(args.output_dir, f"cogvl_{ds_name}.json")
        with open(out_path, "w") as f:
            json.dump(ds_result, f, indent=2,
                      default=lambda x: bool(x) if isinstance(x, np.bool_) else x)
        print(f"  Saved: {out_path}")

        all_dataset_results[ds_name] = {
            "n_images": n_processed,
            "summary": summary,
        }

    # ── Aggregate ──
    if len(all_dataset_results) > 1:
        print(f"\n{'='*80}")
        print(f"  AGGREGATE")
        print(f"{'='*80}")
        for ds, res in all_dataset_results.items():
            ci = res["summary"].get("CogVL (ours)", {}).get("iou", 0)
            si = res["summary"].get("SAM (single)", {}).get("iou", 0)
            oi = res["summary"].get("Oracle", {}).get("iou", 0)
            print(f"  {ds:20s} — CogVL: {ci:.4f}  SAM: {si:.4f}  "
                  f"Oracle: {oi:.4f}  Δ={ci-si:+.4f}")

    combined_path = os.path.join(args.output_dir, "cogvl_combined.json")
    with open(combined_path, "w") as f:
        json.dump({"config": vars(args), "datasets": all_dataset_results}, f, indent=2)
    print(f"\n  Combined: {combined_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CogVL: Semantic-Stability Mask Selection")
    p.add_argument("--datasets", default="busi_malignant",
                   help="Comma-separated dataset names or 'all'")
    p.add_argument("--output-dir", default="/home/mi3dr/SCCS/eccv/cogvl_results")
    p.add_argument("--top-k", type=int, default=5,
                   help="K for random baseline comparison")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Weight for semantic margin in final score")
    p.add_argument("--beta", type=float, default=0.4,
                   help="Weight for stability in final score")
    p.add_argument("--tau-semantic", type=float, default=-0.05,
                   help="Reliability threshold for semantic margin (flag if below)")
    p.add_argument("--tau-stability", type=float, default=0.3,
                   help="Reliability threshold for stability (flag if below)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run_benchmark(args)
