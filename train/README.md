# Training Code — Reference-Guided YOLO

All scripts and data needed to train (and re-train) the YOLOv8 detectors
reported in the conference paper. The UI side lives in
[`../ui/`](../ui/) and consumes the resulting `best.pt`.

## Layout

```
train/
├── README.md                this file
├── requirements.txt
│
├── models/
│   ├── __init__.py
│   └── ds_yolo.py           DS-YOLO architecture (CRFM + dual-stream)
│
├── train_ds.py              train DS-YOLO  (recommended for the paper)
├── train.py                 train RGB / RG-YOLO / Stack6 / F-Sub baselines
│
├── align.py                 ORB + homography alignment + diff map
├── model_4ch.py             patch YOLOv8's first conv to N channels (RG-YOLO)
├── build_4ch_dataset.py     pre-compute (RGB + diff) tensors (RG-YOLO)
├── eval_robustness.py       robustness study under H perturbations
├── aggregate_results.py     emit LaTeX rows for the paper tables
├── remap_to_binary.py       8-class -> 2-class binary remap + resplit
│
├── dataset/                  primary in-house dataset (binary OK/NG, with golden)
│   ├── data.yaml               YOLO config: 2 classes (OK, NG)
│   ├── binary_class_counts.json
│   ├── train/{images,labels}/    147 boards
│   ├── val/{images,labels}/      32 boards
│   ├── test/{images,labels}/     32 boards
│   └── golden/                 3 reference images (used by build_4ch_dataset.py)
│
├── dataset/SolDef_AI/        SolDef_A raw drop (Labeled/ + Dataset/CS*/R*/V*)
│   └── ...                     polygon JSON + JPGs (~1.2 GB)
├── dataset/soldef_yolo/      SolDef_A converted to YOLO bbox (binary OK/NG)
├── dataset/soldef_yolo_5class/  SolDef_A converted to YOLO bbox (5 classes)
│
├── convert_soldef_to_yolo.py  polygon JSON -> YOLO bbox + train/val/test split
│
└── runs/                    training output
    └── baseline_8class/     metrics record of the original 8-class run
                             (60 epochs, mAP@0.5 = 0.985). Weights live in
                             ../ui/runs/detect/train/weights/{best,last}.pt
```

## Quick re-train (DS-YOLO, recommended)

```
cd train
pip install -r requirements.txt

python train_ds.py \
    --data   dataset/data.yaml \
    --golden dataset/golden/golden_ok.bmp \
    --pretrained yolov8m.pt \
    --epochs 60 --imgsz 1024 --batch 4 \
    --name ds_yolo
```

Outputs land at `runs/ds_yolo/weights/best.pt` (selected by validation
mAP@0.5) and `runs/ds_yolo/weights/last.pt`. Each checkpoint is a plain
PyTorch dict containing `state_dict`, `num_classes`, `epoch`, `metrics`
and the training args, so the UI can rebuild the model from it.

### Deploy to UI

```
mkdir -p ../ui/runs/ds_yolo/weights
cp runs/ds_yolo/weights/best.pt ../ui/runs/ds_yolo/weights/best.pt
cp runs/ds_yolo/weights/last.pt ../ui/runs/ds_yolo/weights/last.pt

# In a shell that runs the UI:
USE_DS_YOLO=1 python ../ui/main.py
```

The UI's `window_camera.py` swaps the legacy 8-class pipeline for the
DS-YOLO pipeline whenever the environment variable `USE_DS_YOLO=1` is
set.

## Alternative: train the RGB / RG-YOLO baselines (paper ablations)

```
cd train

# RGB baseline (single-stream YOLOv8m on the aligned capture)
python train.py --variant rgb     --data dataset/data.yaml         --name rgb

# RG-YOLO ablation (4-channel input)
python build_4ch_dataset.py --src dataset --dst dataset_rg \
    --golden dataset/golden/golden_ok.bmp
python train.py --variant rg-yolo --data dataset_rg/rg-yolo.yaml --name rg_yolo
```

## Full RG-YOLO experiment (paper Table II–VI)

```
cd train

# 1. Build the 4-channel (RGB + diff) dataset.
python build_4ch_dataset.py \
    --src dataset --dst dataset_rg \
    --golden dataset/golden/golden_ok.bmp

# 2. Train all 5 variants (each ~2 h on a Tesla T4)
python train.py --variant rgb       --data dataset/data.yaml         --name rgb
python train.py --variant diff-only --data dataset/data.yaml         --name diff_only
python train.py --variant stack6    --data dataset_stack6/stack6.yaml --name stack6
python train.py --variant f-sub     --data dataset/data.yaml         --name f_sub
python train.py --variant rg-yolo   --data dataset_rg/rg-yolo.yaml   --name rg_yolo

# 3. Data-fraction study for H3 (rgb and rg-yolo only).
python train.py --variant rgb     --data dataset/data.yaml        --name rgb_25  --data-fraction 0.25
python train.py --variant rgb     --data dataset/data.yaml        --name rgb_50  --data-fraction 0.50
python train.py --variant rg-yolo --data dataset_rg/rg-yolo.yaml  --name rgyolo_25 --data-fraction 0.25
python train.py --variant rg-yolo --data dataset_rg/rg-yolo.yaml  --name rgyolo_50 --data-fraction 0.50

# 4. Robustness study for the alignment-perturbation table.
python eval_robustness.py \
    --weights runs/detect/rgb/weights/best.pt    --variant rgb     \
    --data dataset/data.yaml --golden dataset/golden/golden_ok.bmp \
    --out runs/robustness/rgb.csv

python eval_robustness.py \
    --weights runs/detect/rg_yolo/weights/best.pt --variant rg-yolo \
    --data dataset_rg/rg-yolo.yaml --golden dataset/golden/golden_ok.bmp \
    --out runs/robustness/rg_yolo.csv

# 5. Aggregate all results into LaTeX rows ready to paste into the paper.
python aggregate_results.py main \
    --variants rgb diff-only stack6 f-sub rg-yolo \
    --runs runs/detect

python aggregate_results.py perclass \
    --variants rgb rg-yolo \
    --runs runs/detect \
    --data dataset/data.yaml

python aggregate_results.py latency \
    --weights runs/detect/rg_yolo/weights/best.pt --channels 4
```

## Dataset

`dataset/` is a **binary OK/NG** taxonomy obtained by collapsing the
original 8-class Roboflow export (`cap_OK`, `cap_NG`, ..., `res_NG`)
through [remap_to_binary.py](remap_to_binary.py). It contains:

- $211$ raw boards covering 4 component types (chip resistor, ceramic
  capacitor, NE555 timer, CD4017 decade counter)
- Split $147 / 32 / 32$ (train / val / test) with seed $0$
- $5{,}420$ component-level instances ($3{,}097$ OK / $2{,}323$ NG)

The original 8-class Roboflow export was removed after the binary
remap was finalised; if you need to redo the remap (e.g. with a
different seed or new images), re-export from Roboflow first.

## SolDef_A as a secondary baseline

The public SolDef_A dataset \[Ulger et al.\] is included for an
independent comparison. It is a *solder-joint* inspection dataset
(distinct from our component-placement task), but is useful as a
public reference: it lets a reviewer reproduce a YOLOv8m baseline on a
publicly available SMT-defect dataset without access to our in-house
data.

### Convert SolDef_A polygon JSON to YOLO bbox

```
# Binary OK/NG (good vs anything else)
python convert_soldef_to_yolo.py \
    --src dataset/SolDef_AI/Labeled \
    --dst dataset/soldef_yolo \
    --mode binary --val-frac 0.15 --test-frac 0.15 --seed 0

# Or 5 classes (good / exc_solder / spike / no_good / poor_solder)
python convert_soldef_to_yolo.py \
    --src dataset/SolDef_AI/Labeled \
    --dst dataset/soldef_yolo_5class \
    --mode multi --val-frac 0.15 --test-frac 0.15 --seed 0
```

### Train a baseline YOLOv8m on SolDef_A

```
python train.py --variant rgb \
    --data dataset/soldef_yolo/data.yaml \
    --imgsz 1280 --batch 4 --epochs 60 \
    --name soldef_baseline
```

### Why not DS-YOLO on SolDef_A?

The reference-guided paradigm at the heart of DS-YOLO requires a
*single* golden image per product line. SolDef_A bundles seven
component setups (CS1-CS7) at three resistor sizes (R0603, R0805,
R1206) with multiple variants per setup, so no single board layout
serves as a global reference. We therefore use SolDef_A only for the
RGB baseline; the DS-YOLO contributions are evaluated on the in-house
dataset where the reference paradigm applies.

## Existing baseline

`runs/baseline_8class/` keeps the *metrics* (`results.csv`, plots,
confusion matrices, args) of a YOLOv8m trained for 60 epochs on the
**original 8-class** taxonomy (precision $0.926$, recall $1.000$,
mAP@0.5 $0.985$). The actual `best.pt` / `last.pt` are deployed under
[`../ui/runs/detect/train/weights/`](../ui/runs/detect/train/weights/);
they were removed from `train/runs/` to avoid duplication.

The paper retrains on the binary 2-class taxonomy, so this baseline is
not a paper experiment — only a sanity-check that the dataset is
learnable.
