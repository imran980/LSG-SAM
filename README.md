# LSG-SAM: Latent-Stability Guided Segment Anything Model for Reliable Medical Image Segmentation

> **Paper:** *Latent-Stability Gated SAM: Detecting Hallucinated Segmentations under
Domain Shift*  
> **Venue:** CogVL, CVPR 2026

LSG-SAM is a **training-free**, **test-time** reliability wrapper around the Segment Anything Model (SAM). It detects when SAM's predictions are unstable under minor prompt perturbations and automatically refines or replaces them — without requiring any fine-tuning or additional training data.

## Key Idea

SAM produces confident-looking masks even when predictions are unreliable. LSG-SAM addresses this by:

1. **Confidence Gating** — If SAM's own confidence score is high (≥ τ), trust it and skip refinement.
2. **Multi-mask Search** — Generate diverse box prompt variants, run SAM with `multimask_output=True`, and collect a pool of candidate masks.
3. **Global Consistency Routing** — Compute pairwise IoU among all candidates. If masks disagree (low consistency), trigger spatial search with majority voting; otherwise, apply tight-box refinement.
4. **Safeguard** — Only replace SAM's original prediction if the refined mask is meaningfully different (IoU < 0.9 vs. original), preventing unnecessary degradation.

No weights are modified. The entire pipeline runs at inference time on top of a frozen SAM checkpoint.

---

## Results (Averaged Across 4 Datasets)

| Method | IoU ↑ | Dice ↑ |
|--------|-------|--------|
| SAM baseline | 0.7684 | 0.8492 |
| SAM + Stability score | 0.7931 | 0.8690 |
| SAM + Recovery search | 0.8147 | 0.8853 |
| **LSG-SAM (full)** | **0.8296** | **0.8971** |

Evaluated on BUSI (malignant), JSRT, Kvasir-SEG, and PROMISE12.

---

## Installation

```bash
git clone https://github.com/<your-username>/LSG-SAM.git
cd LSG-SAM
pip install -r requirements.txt
```

### Python Version

- Python ≥ 3.9
- PyTorch ≥ 2.0 with CUDA support recommended

---

## Checkpoints

Download the following checkpoints and place them as shown:

| Model | Download | Expected Path |
|-------|----------|---------------|
| SAM ViT-H | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | `checkpoints/sam_vit_h_4b8939.pth` |
| MedSAM ViT-B | [medsam_vit_b.pth](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN) | `checkpoints/medsam_vit_b.pth` |

```bash
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# MedSAM: download manually from the Google Drive link above
```

> **Note:** MedSAM was trained at 256×256 input resolution. Our evaluation scripts handle this natively — do not resize MedSAM's `pos_embed` to 1024px (this causes checkerboard artifacts).

---

## Datasets

Download and organize as follows:

```
datasets/
├── Dataset_BUSI_with_GT/
│   ├── malignant/          # images + *_mask.png
│   └── benign/
├── jsrt/
│   ├── jpg/                # chest X-ray images
│   └── masks/              # lung masks
├── Kvasir-SEG/
│   ├── images/
│   └── masks/
└── promise12/
    └── png_slices/         # MRI slices + masks
```

| Dataset | Source |
|---------|--------|
| BUSI | [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| JSRT | [JSRT Database](http://db.jsrt.or.jp/eng.php) |
| Kvasir-SEG | [SimulaMet](https://datasets.simula.no/kvasir-seg/) |
| PROMISE12 | [Grand Challenge](https://promise12.grand-challenge.org/) |

After downloading, update the `BASE` path in each script to point to your dataset root.

---

## Usage

### Run the Full LSG-SAM Pipeline

```bash
python stability_medclipsam.py \
    --datasets busi_malignant,jsrt,kvasir \
    --skip-m2ib \
    --device cuda
```

Key arguments:
- `--sam-confidence-gate 0.85` — confidence threshold for skipping refinement
- `--tau-consistency 0.5` — consistency threshold for search vs. refine routing
- `--n-variants 10` — number of box prompt variants
- `--no-skip-m2ib` — include MedCLIP-SAM baseline (slow, ~70 min extra)

### Baseline Comparison (Table 1 in paper)

```bash
python baseline_comparison.py \
    --datasets busi_malignant,jsrt,kvasir \
    --output-dir results/baseline_comparison \
    --device cuda
```

Produces: `baseline_comparison.csv`, `baseline_comparison.tex`, `baseline_comparison.json`, qualitative figures.

### Ablation Study (Table 2 in paper)

```bash
python ablation_study.py \
    --datasets busi_malignant,jsrt,kvasir \
    --output-dir results/ablation \
    --device cuda
```

Produces: `ablation_table.tex`, `fig_ablation.png`, `fig_ablation.pdf`, per-image JSON.

### Qualitative Figure (Figure 3 in paper)

```bash
python qualitative_figure.py \
    --datasets busi_malignant,jsrt,kvasir \
    --top-k 1 \
    --output-dir results/qualitative \
    --device cuda
```

Produces a single paper-ready figure: `qualitative_comparison.png` + `.pdf`.

### Analysis Plots (Figures 4–7 in paper)

```bash
python plot_analysis.py \
    --results-json results/baseline_comparison/baseline_comparison.json \
    --output-dir results/figures
```

Generates: threshold sensitivity, reliability breakdown, improvement distribution.

### MedSAM Diagnostic

```bash
python medsam_quick_test.py --device cuda
```

Verifies MedSAM loads correctly at native 256px resolution.

---

## Project Structure

```
LSG-SAM/
├── stability_medclipsam.py     # Core method: LSG-SAM pipeline
├── baseline_comparison.py      # SAM vs SAM-BoN vs MedSAM vs Ours
├── ablation_study.py           # 5-config ablation study
├── qualitative_figure.py       # Paper-ready qualitative figure
├── medsam_diagnostic.py        # MedSAM 256px diagnostic
├── plot_analysis.py            # Threshold/reliability/histogram plots
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Citation

```bibtex
@inproceedings{lsg-sam2026,
    title={Latent-Stability Gated SAM: Detecting Hallucinated Segmentations under Domain Shift},
    author=Muhammad Imran,Yugyung Lee
    booktitle={CogVL, CVPR},
    year={2026}
}
```

---

## Acknowledgments

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) — Meta AI
- [MedSAM](https://github.com/bowang-lab/MedSAM) — Bo Wang Lab
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) — Microsoft Research

---

## License

This project is released under the [Apache 2.0 License](LICENSE).
