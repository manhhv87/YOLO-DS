# Reference-Guided YOLO for SMT PCB Defect Detection

Project root for the IEEE conference paper *"Reference-Guided YOLOv8 for
SMT Component Defect Detection on PCB Production Lines"* together with
all training code, runtime UI and the dataset needed to reproduce it.

## Layout

```
PCB/
├── paper/      IEEE conference paper (LaTeX) + figures + the BSc thesis
│               that is the source material for the paper.
│
├── train/      Training code + dataset + YOLO experiment runs.
│               Use this side to (re-)train detectors.
│
└── ui/         PyQt5 UI application + runtime pipeline
                (camera, alignment, YOLO inference, OPC UA to PLC).
                Use this side to run the system on the cell.
```

| Folder | When you touch it |
|---|---|
| `paper/` | Editing the paper, building figures, citing references. |
| `train/` | Re-training models, running ablations, generating LaTeX rows. |
| `ui/`    | Running the inspection cell, deploying a trained model. |

See each folder's `README.md` for build/run instructions.

## Typical workflows

### 1. Re-train and deploy
```
cd train
pip install -r requirements.txt
python train.py --variant rgb --data dataset/data.yaml --name rgb_binary
cp runs/detect/rgb_binary/weights/best.pt ../ui/runs/detect/train/weights/best.pt
cd ../ui
python main.py
```

### 2. Reproduce paper experiments
See [`train/README.md`](train/README.md) for the full sequence
(5 variants × 2 fractions, robustness study, latency, LaTeX
aggregation).

### 3. Edit the paper
```
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## What the paper claims

We extend YOLOv8 with a fourth input channel that carries the per-pixel
absolute difference between the captured frame and a fixed *golden*
reference image, after homography alignment. The resulting detector
(*RG-YOLO*) exploits a property unique to the AVI setting — the golden
image is always available — and is shown to be more accurate than the
RGB baseline on the minority NG class and substantially more
data-efficient on small training budgets.

## Dataset

`train/dataset/` contains a **binary OK/NG** dataset of $211$ labeled
SMT boards covering four component types (chip resistor, ceramic
capacitor, NE555 timer and CD4017 decade counter), totalling $5{,}420$
component-level instances ($3{,}097$ OK / $2{,}323$ NG). Split
$147 / 32 / 32$ with seed $0$.
# YOLO-DS
