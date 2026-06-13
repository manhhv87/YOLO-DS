# Training Code — DS-YOLO for SMT Defect Detection

Toan bo code train va eval cho paper. UI nam o [`../ui/`](../ui/).

Paper so sanh **hai model**: RGB baseline (YOLOv8s mot luong) vs **DS-YOLO**
(Siamese YOLOv8s + Cross-Reference Fusion Module, CRFM). Ket qua phu thuoc
**loai khuyet tat** (defect-dependent), the hien qua hai che do:

- **In-house (reference du thua):** mot board co dinh -> capture-only da bao hoa
  -> DS-YOLO ngang RGB, cong CRFM `alpha ~ 0`.
- **Synthetic (reference can thiet):** layout ngau nhien moi mau, golden rieng
  tung mau -> capture-only khong giai duoc missing/shift -> cong kich hoat.

## Thu muc

```
train/
├── README.md               file nay
├── run_all.py              chay tat ca buoc tu dong (khuyen dung)
├── train.py                train RGB baseline (Ultralytics pipeline)
├── train_ds.py             train DS-YOLO (dual-stream loop + CRFM, per-sample golden)
│
├── models/
│   └── ds_yolo.py          kien truc DS-YOLOv8s: shared backbone + CRFM + neck + head
│
├── data/
│   ├── resplit_inhouse.py        fix data leakage: re-split theo SOURCE board
│   ├── make_synthetic_refguided.py  sinh benchmark reference-dependent (golden rieng/mau)
│   ├── make_fraction_subset.py   tao subset seed-co-dinh dung chung cho RGB & DS (H3)
│   ├── inject_defects.py         tiem khuyet tat (remove/shift/rotate) pha tran mAP
│   ├── convert_soldef_to_yolo.py SolDef_A JSON -> YOLO (da xong)
│   └── remap_to_binary.py        8-class -> 2-class (da xong)
│
├── eval/
│   ├── aggregate_results.py  xuat LaTeX rows (mean+/-std over seeds)
│   ├── verify_evaluator.py   cross-check evaluator tuy chinh == yolo.val (tol 0.002)
│   ├── eval_robustness.py    mAP vs nhieu dich chuyen alignment (sigma_t)
│   ├── plot_data_fraction.py / plot_qual_figures.py  hinh ve
│
├── datasets/
│   ├── inhouse/
│   │   ├── leaky/          Roboflow export goc — KHONG DUNG TRUC TIEP
│   │   └── golden/         golden_ok.bmp, golden2.bmp, golden_ng.png
│   └── soldef/             SolDef_A da convert sang YOLO
│
├── dataset_fixed/          [TU DONG TAO] resplit_inhouse.py output (gitignored)
├── dataset_synth/          [TU DONG TAO] make_synthetic_refguided.py output (gitignored)
└── runs/                   [TU DONG TAO] training outputs (gitignored)
    ├── detect/             RGB baseline (rgb, rgb_synth, rgb_<frac>, rgb_hard, ...)
    ├── ds_yolo/            DS-YOLO (ds_yolo, ds_yolo_synth, ds_yolo_hard)
    └── fraction/           data-fraction study (10/25/50%)
```

Config mac dinh trong `run_all.py`: `SIZE="s"`, `IMGSZ=640`, `EPOCHS=100`,
`BATCH=4`, `SEEDS=[0]` (multi-seed qua `--seeds 0 1 2`).

---

## Chay nhanh — tu dong

```bash
cd train

# Kiem tra lenh (khong chay that)
python run_all.py --dry-run

# Chay mac dinh: resplit -> train_all (rgb) -> train_ds (ds-yolo) -> fraction
#               -> robustness -> soldef_val -> soldef_finetune -> tables -> figures
python run_all.py

# Multi-seed (Bang ket qua mean+/-std)
python run_all.py --steps train_all train_ds fraction --seeds 0 1 2

# Buoc tuy chon (KHONG chay mac dinh):
#   synth          benchmark reference-dependent (kiem tra cong CRFM kich hoat)
#   harden         tiem khuyet tat kho hon len cung 1 board
#   crfm_ablation  ablation thiet ke CRFM (no-gate / fixed-alpha / fuse-scales)
python run_all.py --steps synth --force
```

`--force` ghi de output cu (tao lai dataset + train lai).

---

## Chay tung buoc thu cong

### 1. Chuan bi dataset (in-house)

```bash
# Fix data leakage: re-split theo SOURCE board (53/11/11 sources -> 147/11/11 imgs)
python data/resplit_inhouse.py \
    --src datasets/inhouse/leaky --dst dataset_fixed \
    --val-frac 0.15 --test-frac 0.15 --seed 0
```

### 2. Train hai model (Bang ket qua chinh)

```bash
# RGB baseline -> runs/detect/rgb/
python train.py --variant rgb --size s \
    --data dataset_fixed/data.yaml --epochs 100 --imgsz 640 --batch 4

# DS-YOLO (CRFM, golden co dinh in-house) -> runs/ds_yolo/ds_yolo/
python train_ds.py \
    --data    dataset_fixed/data.yaml \
    --golden  datasets/inhouse/golden/golden_ok.bmp \
    --variant s --fusion crfm \
    --epochs 100 --imgsz 640 --batch 4 \
    --name ds_yolo --save-dir runs/ds_yolo
```

Recipe (SGD@1e-2 + cosine + warmup3 + wd5e-4) **dong nhat** giua `train.py` va
`train_ds.py` de so sanh chi co lap CRFM. DS-YOLO `alpha` khoi tao 0 (identity-
at-init); xem dong `[CRFM alpha]` trong log.

### 3. Data-fraction study (H3)

```bash
# Dung subset seed-co-dinh CHUNG cho ca RGB va DS o moi fraction
python run_all.py --steps fraction --seeds 0 1 2
# -> runs/detect/rgb_<10|25|50>pct[_sN]/  va  runs/fraction/ds_yolo_<...>pct[_sN]/
```

### 4. Synthetic reference-dependent benchmark (cong CRFM kich hoat)

```bash
python run_all.py --steps synth --force
# Sinh dataset_synth (board-disjoint, golden rieng/mau), train rgb_synth + ds_yolo_synth.
# Kiem: [CRFM alpha] tang > 0; doc 2 file test_results.json.
python eval/verify_evaluator.py \
    --weights runs/detect/rgb_synth/weights/best.pt \
    --data dataset_synth/data.yaml --imgsz 640 --split test
```

### 5. Robustness (sigma_t)

```bash
python eval/eval_robustness.py --variant rgb \
    --weights runs/detect/rgb/weights/best.pt \
    --data dataset_fixed/data.yaml \
    --golden datasets/inhouse/golden/golden_ok.bmp \
    --out runs/robustness/rgb.csv --imgsz 640
```

### 6. Tong hop so lieu (LaTeX, mean+/-std)

```bash
python eval/aggregate_results.py mainstats \
    --runs runs/detect --ds-runs runs/ds_yolo \
    --variants rgb ds-yolo --seeds 0 1 2
```

---

## Deploy len UI

```bash
mkdir -p ../ui/runs/ds_yolo/weights
cp runs/ds_yolo/ds_yolo/weights/best.pt ../ui/runs/ds_yolo/weights/best.pt
USE_DS_YOLO=1 python ../ui/main.py
```

---

## Luu y ky thuat

| Dieu | Chi tiet |
|------|----------|
| **Chi co 2 model** | RGB (`train.py`) va DS-YOLO (`train_ds.py --fusion crfm`). Cac variant stack6/diff-only/f-sub/rg-yolo da bo. |
| **imgsz = 640** | Khop checkpoint da train. Khong dung 1024. |
| **golden chi cho in-house** | `golden_ok.bmp` chi dung cho in-house. SolDef_A khong co golden. |
| **synth dung golden rieng/mau** | `train_ds.py --golden-dir dataset_synth` (khong phai `--golden`). |
| **Val/Test khong augment** | Chi train co augmented; val/test 1 anh per source. |
| **Uu tien test_results.json** | `aggregate_results.py` doc `test_results.json` (tren test split) truoc `results.csv`. |
