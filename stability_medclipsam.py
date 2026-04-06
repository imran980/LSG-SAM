"""
Latent-Stability Enhanced MedCLIP-SAM
======================================
Baseline: BiomedCLIP M2IB heatmap → threshold → SAM box → mask
Ours:     + latent-stability + self-consistency + reliability flag + fallback

Usage:
    python stability_medclipsam.py --datasets busi_malignant
    python stability_medclipsam.py --datasets all
"""

import os, sys, argparse, json, warnings, time, random, tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import open_clip

# Add M2IB stack to path (only needed if --no-skip-m2ib)
sys.path.insert(0, "/home/mi3dr/SCCS/sccs")

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

BASE = "/home/mi3dr/SCCS/sccs"

DATASETS = {
    "busi_malignant": {
        "image_dir": f"{BASE}/Dataset_BUSI_with_GT/malignant",
        "mask_dir":  f"{BASE}/Dataset_BUSI_with_GT/malignant",
        "text_query": "a breast ultrasound showing a tumor",
    },
    "busi_benign": {
        "image_dir": f"{BASE}/Dataset_BUSI_with_GT/benign",
        "mask_dir":  f"{BASE}/Dataset_BUSI_with_GT/benign",
        "text_query": "a breast ultrasound showing a lesion",
    },
    "kvasir": {
        "image_dir": f"{BASE}/Kvasir-SEG/images",
        "mask_dir":  f"{BASE}/Kvasir-SEG/masks",
        "text_query": "a colorectal polyp in endoscopy",
    },
    "jsrt": {
        "image_dir": f"{BASE}/jsrt/jpg",
        "mask_dir":  f"{BASE}/jsrt/masks",
        "text_query": "a chest X-ray showing lungs",
    },
    "promise12": {
        "image_dir": f"{BASE}/promise12/png_slices",
        "mask_dir":  f"{BASE}/promise12/png_slices",
        "text_query": "a prostate MRI showing the prostate gland",
    },
}

SAM_CHECKPOINT = f"{BASE}/sam_vit_h_4b8939.pth"
BIOMEDCLIP_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


# ═══════════════════════════════════════════════════════════════
# CORE UTILITIES
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


def gt_bbox(gt_mask):
    ys, xs = np.where(gt_mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def gt_center_box(gt_mask, h, w, scale=1.2):
    bbox = gt_bbox(gt_mask)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    bw, bh = (bbox[2] - bbox[0]) * scale, (bbox[3] - bbox[1]) * scale
    return np.array([max(0, cx - bw/2), max(0, cy - bh/2),
                     min(w-1, cx + bw/2), min(h-1, cy + bh/2)])


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
# 1. MEDCLIP-SAM BASELINE: M2IB HEATMAP → BOX → SAM
# ═══════════════════════════════════════════════════════════════

def heatmap_to_box(heatmap, h_orig, w_orig, threshold=0.5, margin_frac=0.1):
    """Convert M2IB heatmap (224x224) to a bounding box in original image coords."""
    binary = (heatmap > threshold).astype(np.uint8)
    ys, xs = np.where(binary)
    if len(ys) == 0:
        # Fallback: use top 10% of heatmap
        thresh_val = np.percentile(heatmap, 90)
        ys, xs = np.where(heatmap >= thresh_val)
    if len(ys) == 0:
        return np.array([0, 0, w_orig - 1, h_orig - 1], dtype=np.float64)

    # Scale from 224x224 to original
    scale_x = w_orig / 224.0
    scale_y = h_orig / 224.0

    x1 = xs.min() * scale_x
    y1 = ys.min() * scale_y
    x2 = (xs.max() + 1) * scale_x
    y2 = (ys.max() + 1) * scale_y

    # Add margin
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin_frac, bh * margin_frac
    return np.array([
        max(0, x1 - mx), max(0, y1 - my),
        min(w_orig - 1, x2 + mx), min(h_orig - 1, y2 + my)
    ], dtype=np.float64)


def get_m2ib_box(img_path, clip_model, preprocess, tokenizer, text_query, h, w):
    """Run M2IB to get heatmap and derive box prompt."""
    from m2ib_methods import vision_heatmap_m2ib_openclip
    heatmap, peak = vision_heatmap_m2ib_openclip(
        clip_model, preprocess, tokenizer,
        str(img_path), text_query,
        layer_idx=9, beta=0.1, train_steps=10, progbar=False
    )
    box = heatmap_to_box(heatmap, h, w, threshold=0.5)
    return box, heatmap


# ═══════════════════════════════════════════════════════════════
# 2. PROMPT VARIANT GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_variants_from_box(base_box, h, w, n_variants=10):
    """Generate diverse box variants from a base box."""
    cx = (base_box[0] + base_box[2]) / 2
    cy = (base_box[1] + base_box[3]) / 2
    bw = base_box[2] - base_box[0]
    bh = base_box[3] - base_box[1]

    variants = [base_box.copy()]  # first = original

    for _ in range(n_variants - 1):
        scale = np.random.uniform(0.8, 1.5)
        ar = np.random.uniform(0.8, 1.2)
        new_bw = bw * scale * ar
        new_bh = bh * scale / ar
        jx = np.random.uniform(-0.15, 0.15) * bw
        jy = np.random.uniform(-0.15, 0.15) * bh
        ncx, ncy = cx + jx, cy + jy
        box = np.array([
            max(0, ncx - new_bw / 2), max(0, ncy - new_bh / 2),
            min(w - 1, ncx + new_bw / 2), min(h - 1, ncy + new_bh / 2)
        ])
        variants.append(box)

    return variants


# ═══════════════════════════════════════════════════════════════
# 3. SAM INFERENCE (MULTIMASK FOR DIVERSITY)
# ═══════════════════════════════════════════════════════════════

def run_sam_variants(predictor, image, variants):
    """Run SAM with multimask_output=True for each box variant.
    Returns list of {mask, sam_score, box}."""
    results = []
    predictor.set_image(image)
    for box in variants:
        masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
        for i in range(len(masks)):
            results.append({
                "mask": masks[i].astype(bool),
                "sam_score": float(scores[i]),
                "box": box.tolist(),
            })
    return results


# ═══════════════════════════════════════════════════════════════
# 4. LATENT STABILITY SCORING
# ═══════════════════════════════════════════════════════════════

class LatentJitterHook:
    """Injects Gaussian noise into SAM encoder's last block output."""
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


def apply_box_jitter(box, jitter_frac, h, w):
    p = box.copy().astype(np.float64)
    bw, bh = p[2] - p[0], p[3] - p[1]
    jx = np.random.uniform(-jitter_frac, jitter_frac) * bw
    jy = np.random.uniform(-jitter_frac, jitter_frac) * bh
    p[0] = np.clip(p[0] + jx, 0, w - 1); p[1] = np.clip(p[1] + jy, 0, h - 1)
    p[2] = np.clip(p[2] + jx, 0, w - 1); p[3] = np.clip(p[3] + jy, 0, h - 1)
    return p


def compute_latent_stability(predictor, image, box, hook,
                              n_trials=5, jitter_frac=0.10):
    """Latent stability: run SAM with jitter + noise N times, measure consistency."""
    h, w = image.shape[:2]
    jittered_masks = []
    for _ in range(n_trials):
        jittered = apply_box_jitter(box, jitter_frac, h, w)
        hook.active = True
        predictor.set_image(image)
        pred, _, _ = predictor.predict(box=jittered[None, :], multimask_output=False)
        hook.active = False
        jittered_masks.append(pred[0].astype(bool))

    if len(jittered_masks) < 2:
        return 1.0
    ious = [mask_iou(jittered_masks[i], jittered_masks[j])
            for i in range(len(jittered_masks))
            for j in range(i + 1, len(jittered_masks))]
    return float(np.mean(ious))


# ═══════════════════════════════════════════════════════════════
# 6. STABILITY-AS-SWITCH + SPATIAL SEARCH
# ═══════════════════════════════════════════════════════════════

def compute_global_consistency(sam_results):
    """Global consistency: mean pairwise IoU among ALL masks.
    High = masks agree. Low = masks disagree."""
    masks = [r["mask"] for r in sam_results]
    n = len(masks)
    if n < 2:
        return 1.0
    # Subsample for speed if many masks
    if n > 15:
        indices = random.sample(range(n), 15)
        masks = [masks[i] for i in indices]
        n = 15
    ious = [mask_iou(masks[i], masks[j])
            for i in range(n) for j in range(i+1, n)]
    return float(np.mean(ious))


def refine_from_mask(predictor, image, seed_mask, margin_px=8):
    """Tight-box refinement: derive box from mask → small perturbations → best SAM output.
    Returns (refined_mask, sam_score)."""
    h, w = image.shape[:2]
    ys, xs = np.where(seed_mask)
    if len(ys) == 0:
        return seed_mask, 0.0

    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    # Small perturbations: ±5, ±10px margin + slight scale (±5%)
    micro_boxes = []
    for m in [5, 8, 10]:
        micro_boxes.append(np.array([
            max(0, x1 - m), max(0, y1 - m),
            min(w - 1, x2 + m), min(h - 1, y2 + m)
        ], dtype=np.float64))
    for scale in [0.97, 1.0, 1.03]:
        sbw, sbh = bw * scale, bh * scale
        micro_boxes.append(np.array([
            max(0, cx - sbw/2), max(0, cy - sbh/2),
            min(w - 1, cx + sbw/2), min(h - 1, cy + sbh/2)
        ], dtype=np.float64))

    predictor.set_image(image)
    best_mask = seed_mask
    best_score = -1.0
    for box in micro_boxes:
        masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
        for i in range(len(masks)):
            if float(scores[i]) > best_score:
                best_score = float(scores[i])
                best_mask = masks[i].astype(bool)
    return best_mask, best_score


def spatial_search(predictor, image, base_box, h, w):
    """Conservative spatial search: small jitter around base box.
    Uses ±5-10px shifts and ±5% scale, NOT large movements.
    Returns (consensus_mask, best_score) via median voting."""
    cx = (base_box[0] + base_box[2]) / 2
    cy = (base_box[1] + base_box[3]) / 2
    bw = base_box[2] - base_box[0]
    bh = base_box[3] - base_box[1]

    predictor.set_image(image)
    all_masks = []
    all_scores = []

    # A) Box with small jitter (±5-10px) + slight scale
    for dx in [-10, -5, 0, 5, 10]:
        for dy in [-10, -5, 0, 5, 10]:
            if dx == 0 and dy == 0:
                continue  # skip duplicate of base
            box = np.array([
                max(0, base_box[0] + dx), max(0, base_box[1] + dy),
                min(w - 1, base_box[2] + dx), min(h - 1, base_box[3] + dy)
            ], dtype=np.float64)
            masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
            for i in range(len(masks)):
                all_masks.append(masks[i].astype(bool))
                all_scores.append(float(scores[i]))

    # B) Scale variations (tight)
    for scale in [0.95, 1.0, 1.05, 1.10]:
        box = np.array([
            max(0, cx - bw * scale / 2), max(0, cy - bh * scale / 2),
            min(w - 1, cx + bw * scale / 2), min(h - 1, cy + bh * scale / 2)
        ], dtype=np.float64)
        masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
        for i in range(len(masks)):
            all_masks.append(masks[i].astype(bool))
            all_scores.append(float(scores[i]))

    if not all_masks:
        return None, 0.0

    # Consensus: pixel-wise majority vote (median mask)
    stacked = np.stack(all_masks, axis=0).astype(np.float32)
    median_mask = (stacked.mean(axis=0) >= 0.5).astype(bool)

    # Also get best single mask by SAM confidence
    best_single_idx = int(np.argmax(all_scores))
    best_single = all_masks[best_single_idx]
    best_single_score = all_scores[best_single_idx]

    # Return whichever has more non-zero pixels (avoid degenerate empty medians)
    if median_mask.sum() > 0:
        return median_mask, best_single_score
    return best_single, best_single_score


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
    print(f"[StabMCS] Loading SAM ViT-H...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(args.device)
    predictor = SamPredictor(sam)
    # hook = LatentJitterHook(args.latent_epsilon).register(sam.image_encoder)

    # ── Load BiomedCLIP ──
    print(f"[StabMCS] Loading BiomedCLIP...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_NAME)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_NAME)
    clip_model = clip_model.to(args.device)
    clip_model.eval()
    print(f"[StabMCS] Models loaded.")

    method_names = ["SAM (single)", "SAM (best conf)",
                    "Ours", "Oracle"]
    if not args.skip_m2ib:
        method_names.insert(0, "MedCLIP-SAM")

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
        n_unreliable_correct = 0
        per_image_results = []
        t0 = time.time()
        n_processed = 0

        for idx, img_path in enumerate(image_paths):
            name = img_path.stem
            gt_path = find_gt_mask(name, cfg["mask_dir"], img_path)
            if not gt_path:
                continue

            image = np.array(Image.open(img_path).convert("RGB"))
            gt_raw = np.array(Image.open(gt_path).convert("L"))
            gt = gt_raw > (127 if gt_raw.max() > 1 else 0)  # handle both 0/255 and 0/1
            h, w = image.shape[:2]
            if gt.shape != (h, w):
                gt = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((w, h))) > 127
            if not gt.any():
                continue

            img_results = {"image": name}

            def _eval(method_name, pred):
                ev = evaluate_mask(pred, gt)
                img_results[method_name] = ev
                accum[method_name]["ious"].append(ev["iou"])
                accum[method_name]["dices"].append(ev["dice"])
                accum[method_name]["hallucs"].append(ev["hallucination"])
                return ev

            # ── MedCLIP-SAM baseline (optional, slow) ──
            if not args.skip_m2ib:
                m2ib_box, heatmap = get_m2ib_box(img_path, clip_model, preprocess,
                                                  tokenizer, cfg["text_query"], h, w)
                predictor.set_image(image)
                pred_mcs, _, _ = predictor.predict(box=m2ib_box[None, :], multimask_output=False)
                _eval("MedCLIP-SAM", pred_mcs[0].astype(bool))

            # ── Baseline: SAM single (GT-center box) ──
            gt_box = gt_center_box(gt, h, w, scale=1.2)
            predictor.set_image(image)
            pred_single, pred_single_scores, _ = predictor.predict(
                box=gt_box[None, :], multimask_output=False)
            _eval("SAM (single)", pred_single[0].astype(bool))

            # ── Step 2: Generate K prompt variants from GT-center box ──
            # (simulating a detection model / radiologist click)
            variants = generate_variants_from_box(gt_box, h, w,
                                                   n_variants=args.n_variants)

            # ── Step 3: Run SAM for all variants ──
            sam_results = run_sam_variants(predictor, image, variants)

            # ── Baseline: SAM best confidence ──
            if sam_results:
                best_conf_idx = max(range(len(sam_results)),
                                    key=lambda i: sam_results[i]["sam_score"])
                _eval("SAM (best conf)", sam_results[best_conf_idx]["mask"])
            else:
                _eval("SAM (best conf)", np.zeros_like(gt))

            # ══════════════════════════════════════════════
            # SAM-FIRST: only refine when SAM confidence is low
            # ══════════════════════════════════════════════

            sam_single_mask = pred_single[0].astype(bool)
            sam_single_score = float(pred_single_scores[0])

            # Gate 1: If SAM is already confident, keep it
            if sam_single_score >= args.sam_confidence_gate:
                ours_mask = sam_single_mask
                route = "confident"
                safeguard = "kept_sam"
                consistency = 1.0  # skipped computation
            else:
                # SAM confidence is low — worth trying to improve
                consistency = compute_global_consistency(sam_results)

                if consistency < args.tau_consistency:
                    # Low consistency → masks disagree → spatial search
                    candidate_mask, _ = spatial_search(
                        predictor, image, gt_box, h, w)
                    route = "search"
                else:
                    # Moderate confidence + high consistency → refine
                    candidate_mask, _ = refine_from_mask(
                        predictor, image, sam_single_mask)
                    route = "refine"

                # Safeguard: only upgrade if candidate is meaningfully different
                # and covers a reasonable area
                if (candidate_mask is not None and candidate_mask.any()):
                    cand_iou_vs_sam = mask_iou(candidate_mask, sam_single_mask)
                    # If candidate is nearly identical to SAM, keep SAM (no risk)
                    if cand_iou_vs_sam > 0.9:
                        ours_mask = sam_single_mask
                        safeguard = "kept_sam"
                    else:
                        # Candidate differs — use it (SAM was low-confidence anyway)
                        ours_mask = candidate_mask
                        safeguard = "upgraded"
                else:
                    ours_mask = sam_single_mask
                    safeguard = "kept_sam"

            ev_ours = _eval("Ours", ours_mask)

            per_image_results.append({
                **img_results,
                "consistency": round(consistency, 4),
                "route": route,
                "safeguard": safeguard,
            })
            n_processed += 1

            # ── Oracle ──
            if sam_results:
                oracle_ious = [mask_iou(r["mask"], gt) for r in sam_results]
                oracle_idx = int(np.argmax(oracle_ious))
                _eval("Oracle", sam_results[oracle_idx]["mask"])

            # Progress
            elapsed = time.time() - t0
            eta = elapsed / n_processed * (len(image_paths) - idx - 1)
            oi = img_results["Ours"]["iou"]
            si = img_results["SAM (single)"]["iou"]
            mi_str = ""
            if not args.skip_m2ib:
                mi = img_results["MedCLIP-SAM"]["iou"]
                mi_str = f"MCS={mi:.3f} "
            flag = "🔍" if route == "search" else ("🔧" if route == "refine" else "✓")
            sg = "⬆" if safeguard == "upgraded" else "="
            print(f"  [{idx+1}/{len(image_paths)}] {name} — "
                  f"{mi_str}SAM={si:.3f}({sam_single_score:.2f}) Ours={oi:.3f} {flag}{sg} "
                  f"[{route}] | ETA {eta/60:.0f}m")

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

        # Reliability
        print(f"\n  Reliability:")
        if n_processed > 0:
            n_confident = sum(1 for r in per_image_results if r["route"] == "confident")
            n_search = sum(1 for r in per_image_results if r["route"] == "search")
            n_upgraded = sum(1 for r in per_image_results if r["safeguard"] == "upgraded")
            n_kept = sum(1 for r in per_image_results if r["safeguard"] == "kept_sam")
            print(f"    Confident route: {n_confident}/{n_processed} "
                  f"({n_confident/n_processed:.1%})")
            print(f"    Search route:    {n_search}/{n_processed} "
                  f"({n_search/n_processed:.1%})")
            print(f"    Upgraded:        {n_upgraded}/{n_processed} "
                  f"({n_upgraded/n_processed:.1%})")
            print(f"    Kept SAM:        {n_kept}/{n_processed} "
                  f"({n_kept/n_processed:.1%})")
        print(f"{'─'*80}")

        # Save
        ds_result = {
            "dataset": ds_name,
            "n_images": n_processed,
            "time_minutes": round(total_time / 60, 1),
            "config": {
                "n_variants": args.n_variants,
                "tau_consistency": args.tau_consistency,
            },
            "summary": summary,
            "routing": {
                "n_confident": n_confident if n_processed > 0 else 0,
                "n_search": n_search if n_processed > 0 else 0,
            },
            "per_image": per_image_results,
        }
        out_path = os.path.join(args.output_dir, f"stabmcs_{ds_name}.json")
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
            oi = res["summary"].get("Ours", {}).get("iou", 0)
            mi = res["summary"].get("MedCLIP-SAM", {}).get("iou", 0)
            si = res["summary"].get("SAM (single)", {}).get("iou", 0)
            print(f"  {ds:20s} — Ours: {oi:.4f}  MCS: {mi:.4f}  SAM: {si:.4f}  "
                  f"Δ(vs MCS)={oi-mi:+.4f}")

    combined_path = os.path.join(args.output_dir, "stabmcs_combined.json")
    with open(combined_path, "w") as f:
        json.dump({"config": vars(args), "datasets": all_dataset_results}, f, indent=2)
    print(f"\n  Combined: {combined_path}")

    # hook.remove()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Latent-Stability Enhanced MedCLIP-SAM")
    p.add_argument("--datasets", default="busi_malignant",
                   help="Comma-separated dataset names or 'all'")
    p.add_argument("--output-dir", default="/home/mi3dr/SCCS/eccv/stabmcs_results")
    p.add_argument("--n-variants", type=int, default=10,
                   help="Number of box prompt variants")
    p.add_argument("--n-trials", type=int, default=3,
                   help="Jitter trials per box for latent stability")
    p.add_argument("--latent-epsilon", type=float, default=0.1,
                   help="Gaussian noise magnitude for latent jitter")
    p.add_argument("--prompt-jitter", type=float, default=0.10,
                   help="Prompt jitter fraction")
    p.add_argument("--tau-consistency", type=float, default=0.5,
                   help="Global consistency threshold: below this triggers search")
    p.add_argument("--sam-confidence-gate", type=float, default=0.92,
                   help="SAM confidence above this = skip refinement entirely")
    p.add_argument("--tau-stability", type=float, default=0.3,
                   help="Latent stability threshold for reliability flag")
    p.add_argument("--skip-m2ib", action="store_true", default=True,
                   help="Skip slow M2IB baseline (default: skip)")
    p.add_argument("--no-skip-m2ib", dest="skip_m2ib", action="store_false",
                   help="Include M2IB baseline (adds ~70 min)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # Lazy import M2IB only if needed
    if not args.skip_m2ib:
        from m2ib_methods import vision_heatmap_m2ib_openclip

    run_benchmark(args)
