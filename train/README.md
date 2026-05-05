# Training Code — DS-YOLO for SMT Defect Detection

Toan bo code train va eval cho paper. UI nam o [`../ui/`](../ui/).

## Thu muc

```
train/
├── README.md               file nay
├── run_all.py              chay tat ca buoc tu dong (khuyen dung)
├── train.py                train RGB / Diff-only / Stack6 / RG-YOLO
├── train_ds.py             train F-Sub va DS-YOLO (dual-stream loop)
│
├── models/
│   ├── ds_yolo.py          kien truc DS-YOLOv8m: backbone + CRFM + neck + head
│   └── model_4ch.py        patch first Conv2d cua YOLOv8 sang N kenh (RG-YOLO/Stack6)
│
├── data/
│   ├── align.py            ORB + RANSAC homography alignment
│   ├── resplit_inhouse.py  fix data leakage: re-split theo source image level
│   ├── build_4ch_dataset.py  tao dataset 4-ch (RGB+diff) cho RG-YOLO
│   ├── convert_soldef_to_yolo.py  chuyen SolDef_A JSON -> YOLO (da xong)
│   └── remap_to_binary.py  8-class -> 2-class (da xong, khong can chay lai)
│
├── eval/
│   ├── eval_robustness.py  Table V: eval rgb+rg-yolo duoi alignment perturbation
│   └── aggregate_results.py  xuat LaTeX rows cho Table II/III/IV/V
│
├── datasets/
│   ├── inhouse/
│   │   ├── leaky/          Roboflow export goc — KHONG DUNG TRUC TIEP
│   │   └── golden/         golden_ok.bmp, golden2.bmp, golden_ng.png
│   └── soldef/             SolDef_A da convert sang YOLO (300/64/64)
│
├── dataset_fixed/          [TU DONG TAO] resplit_inhouse.py output
├── dataset_rg/             [TU DONG TAO] build_4ch_dataset.py output
└── runs/                   [TU DONG TAO] training outputs
    ├── detect/             Ultralytics variants (rgb, diff-only, stack6, rg-yolo)
    ├── ds_yolo/            DS-YOLO + F-Sub (train_ds.py)
    ├── fraction/           data-fraction study (25%/50%/100%)
    └── robustness/         robustness CSVs (eval_robustness.py)
```

---

## Chay nhanh — tat ca tu dong

```bash
cd train

# Kiem tra lenh (khong chay that)
python run_all.py --dry-run

# Chay tat ca tu dau (~3-5 ngay)
python run_all.py

# Chi chay mot so buoc (neu dataset da co san)
python run_all.py --steps train_all train_ds fraction robustness soldef_val tables
```

---

## Chay tung buoc thu cong

### Buoc 1: Chuan bi dataset

```bash
cd train

# 1a. Fix data leakage: re-split theo source identity
python data/resplit_inhouse.py \
    --src       datasets/inhouse/leaky \
    --dst       dataset_fixed \
    --val-frac  0.15 \
    --test-frac 0.15 \
    --seed      0
# Ket qua: dataset_fixed/{train,val,test}/{images,labels}/ + data.yaml

# 1b. Tao 4-channel dataset cho RG-YOLO
python data/build_4ch_dataset.py \
    --src    dataset_fixed \
    --dst    dataset_rg \
    --golden datasets/inhouse/golden/golden_ok.bmp
# Ket qua: dataset_rg/{train,val,test}/{images,labels}/ + rg-yolo.yaml
```

### Buoc 2: Train 6 variants (Table II)

**LUU Y**: `f-sub` va `ds-yolo` PHAI dung `train_ds.py`, KHONG dung `train.py`.

```bash
# [2a] RGB baseline -> runs/detect/rgb/
python train.py --variant rgb --size m \
    --data dataset_fixed/data.yaml \
    --epochs 60 --imgsz 640 --batch 4

# [2b] Diff-only -> runs/detect/diff-only/
python train.py --variant diff-only --size m \
    --data dataset_fixed/data.yaml \
    --epochs 60 --imgsz 640 --batch 4

# [2c] Stack6 (6-ch: RGB+Golden) -> runs/detect/stack6/
python train.py --variant stack6 --size m \
    --data dataset_fixed/data.yaml \
    --epochs 60 --imgsz 640 --batch 4

# [2d] RG-YOLO (4-ch: RGB+Diff) -> runs/detect/rg-yolo/
python train.py --variant rg-yolo --size m \
    --data dataset_rg/rg-yolo.yaml \
    --epochs 60 --imgsz 640 --batch 4

# [2e] F-Sub (feature subtraction ablation) -> runs/ds_yolo/f_sub/
#      PHAI dung train_ds.py --fusion sub
python train_ds.py \
    --data     dataset_fixed/data.yaml \
    --golden   datasets/inhouse/golden/golden_ok.bmp \
    --variant  m \
    --fusion   sub \
    --epochs   60 --imgsz 640 --batch 4 \
    --name     f_sub \
    --save-dir runs/ds_yolo

# [2f] DS-YOLO (main contribution: CRFM) -> runs/ds_yolo/ds_yolo/
python train_ds.py \
    --data     dataset_fixed/data.yaml \
    --golden   datasets/inhouse/golden/golden_ok.bmp \
    --variant  m \
    --fusion   crfm \
    --epochs   60 --imgsz 640 --batch 4 \
    --name     ds_yolo \
    --save-dir runs/ds_yolo
```

### Buoc 3: Data-fraction study (Figure H3)

```bash
# RGB tai 25% / 50% -> runs/detect/rgb_25pct/, rgb_50pct/
python train.py --variant rgb --size m \
    --data dataset_fixed/data.yaml --epochs 60 --imgsz 640 --batch 4 \
    --data-fraction 0.25 --name rgb_25pct
python train.py --variant rgb --size m \
    --data dataset_fixed/data.yaml --epochs 60 --imgsz 640 --batch 4 \
    --data-fraction 0.50 --name rgb_50pct

# DS-YOLO tai 25% / 50% -> runs/fraction/ds_yolo_25pct/, ds_yolo_50pct/
python train_ds.py \
    --data dataset_fixed/data.yaml \
    --golden datasets/inhouse/golden/golden_ok.bmp \
    --variant m --fusion crfm --epochs 60 --imgsz 640 --batch 4 \
    --data-fraction 0.25 --name ds_yolo_25pct --save-dir runs/fraction
python train_ds.py \
    --data dataset_fixed/data.yaml \
    --golden datasets/inhouse/golden/golden_ok.bmp \
    --variant m --fusion crfm --epochs 60 --imgsz 640 --batch 4 \
    --data-fraction 0.50 --name ds_yolo_50pct --save-dir runs/fraction
# 100% = Buoc 2f (ds_yolo day du), khong can train lai
```

### Buoc 4: Robustness study (Table V)

```bash
# Can xong Buoc 2a (rgb) va 2d (rg-yolo) truoc

python eval/eval_robustness.py \
    --variant rgb \
    --weights runs/detect/rgb/weights/best.pt \
    --data    dataset_fixed/data.yaml \
    --golden  datasets/inhouse/golden/golden_ok.bmp \
    --out     runs/robustness/rgb.csv \
    --imgsz   640

python eval/eval_robustness.py \
    --variant rg-yolo \
    --weights runs/detect/rg-yolo/weights/best.pt \
    --data    dataset_rg/rg-yolo.yaml \
    --golden  datasets/inhouse/golden/golden_ok.bmp \
    --out     runs/robustness/rg-yolo.csv \
    --imgsz   640
```

### Buoc 5: External validation tren SolDef_A

```bash
python train.py --variant rgb --size m \
    --data   datasets/soldef/data.yaml \
    --epochs 60 --imgsz 640 --batch 4 \
    --name   rgb_soldef
# Ket qua -> runs/detect/rgb_soldef/test_results.json
```

### Buoc 6: Tong hop so lieu (LaTeX)

```bash
# Table II
python eval/aggregate_results.py main \
    --runs    runs/detect \
    --ds-runs runs/ds_yolo \
    --variants rgb diff-only stack6 rg-yolo f-sub ds-yolo

# Table III — per-class OK/NG (can GPU)
python eval/aggregate_results.py perclass \
    --runs     runs/detect \
    --ds-runs  runs/ds_yolo \
    --variants rgb rg-yolo ds-yolo \
    --data     dataset_fixed/data.yaml \
    --golden   datasets/inhouse/golden/golden_ok.bmp \
    --imgsz    640 --split test

# Table IV — latency (RGB va RG-YOLO)
python eval/aggregate_results.py latency \
    --weights runs/detect/rgb/weights/best.pt \
    --imgsz 640 --channels 3 --runs-count 200
python eval/aggregate_results.py latency \
    --weights runs/detect/rg-yolo/weights/best.pt \
    --imgsz 640 --channels 4 --runs-count 200
# DS-YOLO latency: do thu cong (xem PLAN.md Buoc 7c)

# Table V — robustness
python eval/aggregate_results.py robust \
    --csvs   runs/robustness/rgb.csv runs/robustness/rg-yolo.csv \
    --labels "RGB baseline" "RG-YOLO"
```

---

## Deploy lên UI

```bash
mkdir -p ../ui/runs/ds_yolo/weights
cp runs/ds_yolo/ds_yolo/weights/best.pt ../ui/runs/ds_yolo/weights/best.pt

# Chay UI voi DS-YOLO backend
USE_DS_YOLO=1 python ../ui/main.py
```

---

## Luu y ky thuat

| Dieu | Chi tiet |
|------|----------|
| **f-sub phai dung train_ds.py** | `train.py` khong co variant `f-sub`. Phai dung `train_ds.py --fusion sub` |
| **imgsz phai la 640** | Tat ca script dung `--imgsz 640`. Khong dung 1024 (lech voi run_all.py) |
| **golden dung cho in-house** | `golden_ok.bmp` chi dung cho in-house dataset. SolDef_A khong co golden |
| **Val/Test khong augment** | Chi train split co augmented images; val/test chi dung 1 anh per source |
| **Uu tien test_results.json** | aggregate_results.py doc `test_results.json` truoc (do tren test split), roi moi doc `results.csv` |
