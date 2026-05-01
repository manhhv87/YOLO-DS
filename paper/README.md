# Conference Paper — Reference-Guided YOLO

IEEE conference paper built around the academic contribution
**Reference-Guided YOLO (RG-YOLO)**: a 4-channel YOLOv8 variant that fuses an
absolute-difference map between the aligned capture and the golden reference.

## Layout

```
paper/
├── main.tex            the IEEE conference paper
├── refs.bib            bibliography (TeXLive-compatible)
├── README.md           this file
├── figures/            figures used by main.tex (paths are relative)
└── thesis/             original BSc thesis (source material for the paper)
    ├── main.tex
    ├── main.pdf
    ├── tex/                 chapter sources
    └── figures/             full set of original figures
```

## Build

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Requires the `IEEEtran` class (TeX Live `texlive-publishers` package).

## Story

- **Problem.** SMT defect detection is a constrained instance of object
  detection because a golden reference image is always available. Standard
  detectors ignore it.
- **Method.** Align the capture to the golden image with ORB+homography,
  compute `|aligned - golden|`, feed YOLOv8 a 4-channel input, retrain.
- **Hypotheses tested in §VI.**
  - **H1**: 4-channel beats RGB-only at equal compute.
  - **H2**: input-level fusion beats feature-level fusion (F-Sub).
  - **H3**: the gap grows as training data shrinks.
- **System.** The detector is deployed on a real cell with Mitsubishi PLC
  + Yaskawa GP7 + Hyundai HH7 over CC-Link/OPC UA.

## What you (the authors) need to fill in

Every `\TODO{...}` macro in [main.tex](main.tex) renders in **red** in the
PDF. Group them by experiment:

### Numbers to obtain by running the code in [`../code/`](../code/)

- **Table II — Main results.** Train `rgb`, `diff-only`, `stack6`,
  `f-sub`, `rg-yolo` (5 runs). Plug Precision / Recall / mAP@0.5 /
  mAP@0.5:0.95 for each.
- **Table III — Per-class.** Take `rgb` and `rg-yolo` runs, fill per-class
  mAP@0.5.
- **Table IV — Latency.** Measure each pipeline stage on the deployment PC.
- **Table V — Robustness.** Run `eval_robustness.py` for `rgb` and `rg-yolo`.
- **Fig. 5 — Data-fraction.** Train `rgb` and `rg-yolo` with
  `--data-fraction 0.25 0.5 1.0`, plot mAP vs fraction.

### Hyperparameters reported in the text

- ORB `n_features` (currently 2000), Lowe's `ratio` (0.75), RANSAC `ransac_thresh` (3.0 px) in §IV-A.
- CLAHE clip limit and tile-grid (only if you decide to keep CLAHE; the
  current draft drops it because the diff map already normalises contrast).
- Best-checkpoint epoch (§VI-B).
- Deployment PC specs (§VI-G).
- Colab GPU type used for training (§VI-B).
- CC-Link bus speed (§V — currently 2.5 Mbps).

### Dataset statistics (Table I)

Already filled in from [`code/dataset/binary_class_counts.json`](../code/dataset/binary_class_counts.json):
$211$ boards, $147/32/32$ split, $5{,}420$ component instances total
($3{,}097$ OK / $2{,}323$ NG). The remaining `\TODO{}` is the image
resolution (peek at any file in [`code/dataset/train/images/`](../code/dataset/train/images/)).

### References to add to `refs.bib`

- 2–3 YOLO-for-PCB papers (§II — paragraph "Deep detectors").
- 1 classical AOI / template-matching baseline (§II — paragraph "Classical AOI").
- 1 Siamese change-detection paper if F-Sub variant is kept (§IV-C).
- 1–2 OPC UA + CC-Link Industry 4.0 integration papers (§II — paragraph "Industrial integration").
- 1–2 template-aware PCB defect detection papers if any exist (§II).

### Optional polish

- Replace the EasyODM market reference with a peer-reviewed source.
- Acknowledgment text (funding, lab equipment).
- Replace [Chapter4/i5.png](../figures/Chapter4/i5.png) and
  [Chapter4/i9.png](../figures/Chapter4/i9.png) with side-by-side
  visualisations (RGB vs RG-YOLO predictions) — the academic story is
  much stronger if Fig. 7 explicitly shows where RG-YOLO catches a
  defect that the baseline misses.

## Suggested venues

- **MAPR** (Int'l Conf. on Multimedia Analysis and Pattern Recognition) — IEEE
- **NICS** (NAFOSTED Conf. on Information and Computer Science) — IEEE
- **ICCAIS** (Int'l Conf. on Control, Automation and Information Sciences) — IEEE
- **VCCA** (Vietnam Conf. on Control and Automation) — Vietnamese-friendly
- **ATC** (Int'l Conf. on Advanced Technologies for Communications) — IEEE
- For higher venues (ICRA workshops, CASE) the data-fraction and
  robustness studies should be tightened and a comparison with anomaly
  baselines (PaDiM/PatchCore) added.

## Files

- [main.tex](main.tex) — paper source.
- [refs.bib](refs.bib) — bibliography.
- [../code/](../code/) — experiment code template (training + evaluation).
- [../figures/](../figures/) — figures (paths in main.tex use `../figures/`).
