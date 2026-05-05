# Ke hoach hoan thien paper: DS-YOLO for SMT Defect Detection
# Updated: 2026-04-30

===============================================================================
MO TA CAC FILE CODE (doc truoc khi chay)
===============================================================================

Tat ca file Python nam trong thu muc  train/  (da duoc to chuc vao cac folder)

--- models/  (thu vien, khong chay truc tiep) ---

  models/ds_yolo.py
    - Kien truc DS-YOLOv8m: Dual-Stream backbone + Cross-Reference Fusion (CRFM)
    - CRFM: F_fused = F_cap + alpha * sigmoid(g(|F_cap - F_gold|)) * F_cap
    - Import boi train_ds.py

  models/model_4ch.py
    - Patch first Conv2d cua YOLOv8 de nhan N kenh thay vi 3 kenh
    - Su dung boi: train.py (khi variant = rg-yolo hoac stack6)
    - 3 kenh dau lay tu pretrained weights; kenh extra init = trung binh RGB

--- data/  (chuan bi va chuyen doi dataset) ---

  data/align.py
    - Ham ORB + RANSAC de can chinh capture vao golden reference
    - Su dung boi: data/build_4ch_dataset.py
    - Xuat: aligned image, homography matrix, inlier ratio

  data/resplit_inhouse.py     [BUOC 1 — bat buoc, chay dau tien]
    - Fix data leakage: re-split in-house dataset theo SOURCE image level
    - Doc tu: datasets/inhouse/leaky/{train,val,test}/
    - Trich xuat ten anh goc (bo phan .rf.<hash>)
    - Split 75 source anh thanh 70/15/15 (seed=0)
    - Train: giu TAT CA ban augmented (~148 anh)
    - Val/Test: chi giu 1 ban moi source (11 anh moi split, khong aug)
    - Xuat: dataset_fixed/ + dataset_fixed/data.yaml + split_info.json
    - Lenh: python data/resplit_inhouse.py

  data/build_4ch_dataset.py   [BUOC 1b — can cho rg-yolo va stack6]
    - Tien xu ly: can chinh anh -> tinh diff map -> luu (H,W,4) .npy
    - Goi align.py cho tung anh, compute grayscale difference map
    - Xuat: dataset_rg/ + dataset_rg/rg-yolo.yaml
    - Lenh: python data/build_4ch_dataset.py --src dataset_fixed
              --dst dataset_rg --golden datasets/inhouse/golden/golden_ok.bmp

  data/convert_soldef_to_yolo.py  [Chi can neu datasets/soldef/ chua ton tai]
    - Chuyen SolDef_A tu LabelMe JSON sang YOLO bbox format
    - Hai che do: binary (OK/NG) hoac multi (5 lop)
    - LUU Y: datasets/soldef/ DA CO SAN, KHONG CAN CHAY LAI

  data/remap_to_binary.py     [Khong can chay — da xu ly]
    - Chuyen dataset 8-class cua Roboflow sang binary OK/NG
    - Chi can neu ban co raw data 8-class chua xu ly

--- eval/  (danh gia va tong hop ket qua) ---

  eval/eval_robustness.py     [BUOC 4 — Table V: robustness]
    - Danh gia model duoi nhieu muc do nhieu alignment (sigma_t = 0..20 px)
    - Tao temp dataset voi perturbation ngau nhien (rigid homography)
    - rgb variant: luu .jpg tam, goi yolo.val()
    - rg-yolo variant: tinh lai 4-ch .npy voi anh bi pertub, goi yolo.val()
    - Xuat: runs/robustness/<variant>.csv
    - Lenh: python eval/eval_robustness.py --variant rgb
               --weights runs/detect/rgb/weights/best.pt
               --data dataset_fixed/data.yaml
               --golden datasets/inhouse/golden/golden_ok.bmp
               --out runs/robustness/rgb.csv

  eval/aggregate_results.py   [Cuoi cung — tao LaTeX table rows]
    - Subcommand `main`:     Table II — so sanh tat ca variant
    - Subcommand `perclass`: Table III — per-class OK/NG (can GPU)
    - Subcommand `latency`:  Table IV — do latency tung buoc
    - Subcommand `robust`:   Table V — bang robustness tu eval_robustness.py
    - Ho tro ca Ultralytics CSV format (train.py) va DS-YOLO CSV (train_ds.py)
    - Lenh: python eval/aggregate_results.py main
               --runs runs/detect
               --variants rgb diff-only stack6 rg-yolo ds-yolo

--- Root scripts (entry points chinh) ---

  train.py                    [BUOC 2 — train cac variant Ultralytics]
    - Train 1 trong 5 variant: rgb | diff-only | rg-yolo | stack6 | f-sub
    - Dung Ultralytics YOLO API, ho tro --data-fraction cho H3 study
    - Goi models/model_4ch.py neu variant can > 3 kenh
    - Xuat: runs/detect/<name>/weights/best.pt + results.csv
    - Vi du: python train.py --variant rgb
               --data dataset_fixed/data.yaml --epochs 60

  train_ds.py                 [BUOC 2 — train DS-YOLO (main contribution)]
    - Train DS-YOLOv8m (Dual-Stream + CRFM) voi manual PyTorch loop
    - Nhan them --golden: anh reference de chay qua Siamese branch
    - Dung Ultralytics data loader + v8DetectionLoss, nhung forward pass rieng
    - Xuat: runs/ds_yolo/<name>/weights/best.pt + last.pt + results.csv
    - Vi du: python train_ds.py
               --data dataset_fixed/data.yaml
               --golden datasets/inhouse/golden/golden_ok.bmp
               --epochs 60 --name ds_yolo

  run_all.py                  [Chay tat ca tu dong]
    - Goi tat ca buoc theo thu tu dung
    - Moi buoc idempotent: bo qua neu output da ton tai
    - Ho tro --dry-run (in lenh khong chay) va --force (chay lai)
    - Ho tro --steps de chon buoc cu the
    - Vi du: python run_all.py --dry-run
             python run_all.py --steps resplit train_ds tables


===============================================================================
CAU TRUC THU MUC SAU KHI SETUP
===============================================================================

  train/
  |-- datasets/
  |   |-- inhouse/
  |   |   |-- leaky/          <- Roboflow export GOC, BI LEAKY, KHONG DUNG TRUC TIEP
  |   |   |   |-- train/val/test/
  |   |   |   `-- data.yaml
  |   |   `-- golden/
  |   |       |-- golden_ok.bmp   <- reference anh chuan (OK board)
  |   |       |-- golden2.bmp
  |   |       `-- golden_ng.png
  |   `-- soldef/             <- DA CONVERT SAN: 300/64/64 train/val/test
  |       |-- train/val/test/
  |       |-- data.yaml
  |       `-- class_counts.json
  |
  |-- dataset_fixed/          <- TU DONG TAO boi data/resplit_inhouse.py
  |-- dataset_rg/             <- TU DONG TAO boi data/build_4ch_dataset.py
  `-- runs/                   <- TU DONG TAO boi qua trinh training


===============================================================================
BUOC THUC HIEN CHI TIET
===============================================================================

---------------------------------------
BUOC 1: FIX DATA LEAKAGE  [~30 phut]
---------------------------------------

  Van de: 46/75 source anh bi leak qua train/val/test vi Roboflow split
  sai o muc augmented file, khong phai source image.

  cd C:\Users\manhh\Desktop\PCB\train

  # 1a. Chay that
  python data/resplit_inhouse.py \
      --src      datasets/inhouse/leaky \
      --dst      dataset_fixed \
      --val-frac 0.15 \
      --test-frac 0.15 \
      --seed     0

  # Ket qua can co:
  #   dataset_fixed/train/images/  ~148 anh (giu tat ca augmented)
  #   dataset_fixed/val/images/    ~11 anh  (1 per source)
  #   dataset_fixed/test/images/   ~11 anh  (1 per source)
  #   dataset_fixed/golden/        sao chep tu datasets/inhouse/golden/
  #   dataset_fixed/data.yaml
  #   dataset_fixed/split_info.json

  # 1b. Build 4-channel dataset cho RG-YOLO va Stack6
  python data/build_4ch_dataset.py \
      --src    dataset_fixed \
      --dst    dataset_rg \
      --golden datasets/inhouse/golden/golden_ok.bmp \
      --splits train val test

  # Ket qua can co:
  #   dataset_rg/images/{train,val,test}/*.npy   (H,W,4) uint8
  #   dataset_rg/labels/{train,val,test}/*.txt   (copy tu dataset_fixed)
  #   dataset_rg/rg-yolo.yaml

---------------------------------------
BUOC 2: TRAIN CAC VARIANTS  [~3-5 ngay]
---------------------------------------

  Chay tat ca 6 variant tren dataset_fixed (yolov8m backbone, 60 epochs):

  LUU Y QUAN TRONG:
    - RGB, Diff-only, RG-YOLO, Stack6  -> train.py (Ultralytics pipeline)
    - F-Sub, DS-YOLO                   -> train_ds.py (custom dual-stream loop)
    - f-sub KHONG the chay qua train.py (khong co trong VARIANT_CHANNELS)

  # [2a] RGB baseline — luu vao runs/detect/rgb/
  python train.py --variant rgb --size m \
      --data dataset_fixed/data.yaml \
      --epochs 60 --imgsz 640 --batch 4

  # [2b] Diff-only — luu vao runs/detect/diff-only/
  python train.py --variant diff-only --size m \
      --data dataset_fixed/data.yaml \
      --epochs 60 --imgsz 640 --batch 4

  # [2c] Stack6 (RGB + Golden, 6 kenh) — luu vao runs/detect/stack6/
  python train.py --variant stack6 --size m \
      --data dataset_fixed/data.yaml \
      --epochs 60 --imgsz 640 --batch 4

  # [2d] RG-YOLO (RGB + diff, 4 kenh) — luu vao runs/detect/rg-yolo/
  python train.py --variant rg-yolo --size m \
      --data dataset_rg/rg-yolo.yaml \
      --epochs 60 --imgsz 640 --batch 4

  # [2e] F-Sub (feature subtraction ablation) — luu vao runs/ds_yolo/f_sub/
  #      Phai dung train_ds.py --fusion sub, KHONG dung train.py
  python train_ds.py \
      --data     dataset_fixed/data.yaml \
      --golden   datasets/inhouse/golden/golden_ok.bmp \
      --variant  m \
      --fusion   sub \
      --epochs   60 --imgsz 640 --batch 4 \
      --name     f_sub \
      --save-dir runs/ds_yolo

  # [2f] DS-YOLO (Dual-Stream + CRFM — main contribution) — luu vao runs/ds_yolo/ds_yolo/
  python train_ds.py \
      --data     dataset_fixed/data.yaml \
      --golden   datasets/inhouse/golden/golden_ok.bmp \
      --variant  m \
      --fusion   crfm \
      --epochs   60 --imgsz 640 --batch 4 \
      --name     ds_yolo \
      --save-dir runs/ds_yolo

  Metrics can co tu buoc nay (Table II):
    - runs/detect/rgb/test_results.json
    - runs/detect/diff-only/test_results.json
    - runs/detect/stack6/test_results.json
    - runs/detect/rg-yolo/test_results.json
    - runs/ds_yolo/f_sub/test_results.json
    - runs/ds_yolo/ds_yolo/test_results.json

---------------------------------------
BUOC 3: DATA FRACTION STUDY  [~1-2 ngay]
---------------------------------------

  Chung minh H3: DS-YOLO co loi the khi data it

  LUU Y: 100% khong can train rieng — dung ket qua tu Buoc 2a (rgb) va 2f (ds_yolo)

  # RGB tai 25% va 50% — luu vao runs/detect/rgb_25pct/, rgb_50pct/
  python train.py --variant rgb --size m \
      --data dataset_fixed/data.yaml \
      --epochs 60 --imgsz 640 --batch 4 \
      --data-fraction 0.25 --name rgb_25pct
  python train.py --variant rgb --size m \
      --data dataset_fixed/data.yaml \
      --epochs 60 --imgsz 640 --batch 4 \
      --data-fraction 0.50 --name rgb_50pct

  # DS-YOLO tai 25% va 50% — luu vao runs/fraction/ds_yolo_25pct/, ds_yolo_50pct/
  python train_ds.py \
      --data dataset_fixed/data.yaml \
      --golden datasets/inhouse/golden/golden_ok.bmp \
      --variant m --fusion crfm \
      --epochs 60 --imgsz 640 --batch 4 \
      --data-fraction 0.25 --name ds_yolo_25pct \
      --save-dir runs/fraction
  python train_ds.py \
      --data dataset_fixed/data.yaml \
      --golden datasets/inhouse/golden/golden_ok.bmp \
      --variant m --fusion crfm \
      --epochs 60 --imgsz 640 --batch 4 \
      --data-fraction 0.50 --name ds_yolo_50pct \
      --save-dir runs/fraction

  Ket qua can co (3 diem moi model):
    - runs/detect/rgb_25pct/test_results.json
    - runs/detect/rgb_50pct/test_results.json
    - runs/detect/rgb/test_results.json          (100% — tu Buoc 2a)
    - runs/fraction/ds_yolo_25pct/test_results.json
    - runs/fraction/ds_yolo_50pct/test_results.json
    - runs/ds_yolo/ds_yolo/test_results.json     (100% — tu Buoc 2f)
  -> Ve Figure: x-axis = fraction (25/50/100%), y-axis = mAP@0.5, 2 duong (RGB vs DS-YOLO)

---------------------------------------
BUOC 4: ROBUSTNESS STUDY  [~2-4 gio]
---------------------------------------

  Muc tieu: Table V trong paper — RGB vs RG-YOLO duoi alignment perturbation.

  Ly do chi can 2 model:
    - rgb     : khong dung golden/diff -> khong bi anh huong alignment (baseline tham chieu)
    - rg-yolo : diff channel tinh tren anh bi pertub -> nhat cam voi alignment error
  -> Paper (Table V) da duoc cap nhat chi bao gom 2 model nay.

  # Can train xong Buoc 2a va 2d truoc.
  # --imgsz PHAI khop voi imgsz dung khi train (640)

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

  Ket qua -> Table V (sigma_t = 0, 2, 5, 10, 20 px vs mAP@0.5)
  Mong doi: rg-yolo > rgb tai sigma_t=0; khoang cach thu hep va dao nguoc khi sigma_t tang

---------------------------------------
BUOC 5: SOLDEF_A EXTERNAL VALIDATION  [~1 ngay]
---------------------------------------

  Muc tieu: Chung minh RGB backbone generalize ra ngoai 1 board.

  python train.py --variant rgb --size m \
      --data   datasets/soldef/data.yaml \
      --epochs 60 --imgsz 640 --batch 4 \
      --name   rgb_soldef

  Metrics: mAP@0.5 tren SolDef_A test set
  -> Bao cao trong paper (1 doan Section VI.A):
     "RGB YOLOv8m dat mAP@0.5 = X.XX tren SolDef_A, chung minh
     backbone co the transfer sang board layout khac."

---------------------------------------
BUOC 6 (OPTIONAL): SOLDEF_A PRE-TRAINING  [~2-3 ngay]
---------------------------------------

  Neu thoi gian cho phep, them 1 ablation row vao Table II:

  # 6a. Pre-train tren SolDef_A
  python train.py --variant rgb --size m \
      --data   datasets/soldef/data.yaml \
      --epochs 60 --imgsz 640 --batch 4 \
      --name   soldef_pretrain

  # 6b. Fine-tune tren in-house
  python train.py --variant rgb --size m \
      --data    dataset_fixed/data.yaml \
      --weights runs/detect/soldef_pretrain/weights/best.pt \
      --epochs  60 --imgsz 640 --batch 4 \
      --name    rgb_soldef_pretrain

  python train_ds.py \
      --data       dataset_fixed/data.yaml \
      --golden     datasets/inhouse/golden/golden_ok.bmp \
      --variant    m \
      --pretrained runs/detect/soldef_pretrain/weights/best.pt \
      --epochs     60 --imgsz 640 --batch 4 \
      --name       ds_yolo_soldef \
      --save-dir   runs/ds_yolo

  Narrative: "Domain-specific pre-training compensates for limited in-house data"

---------------------------------------
BUOC 7: TONG HOP SO LIEU  [~2-4 gio]
---------------------------------------

  # [7a] Table II — so sanh tat ca 6 variant
  #      Doc test_results.json (uu tien) hoac results.csv (du phong)
  python eval/aggregate_results.py main \
      --runs     runs/detect \
      --ds-runs  runs/ds_yolo \
      --variants rgb diff-only stack6 rg-yolo f-sub ds-yolo

  # [7b] Table III — per-class OK/NG mAP@0.5 (can GPU de chay yolo.val())
  python eval/aggregate_results.py perclass \
      --runs     runs/detect \
      --ds-runs  runs/ds_yolo \
      --variants rgb rg-yolo ds-yolo \
      --data     dataset_fixed/data.yaml \
      --golden   datasets/inhouse/golden/golden_ok.bmp \
      --imgsz    640 \
      --split    test

  # [7c] Table IV — latency (3 cot: RGB | RG-YOLO | DS-YOLO)
  #      2 dong dau do qua aggregate_results.py latency:
  python eval/aggregate_results.py latency \
      --weights runs/detect/rgb/weights/best.pt \
      --imgsz 640 --channels 3 --runs-count 200
  python eval/aggregate_results.py latency \
      --weights runs/detect/rg-yolo/weights/best.pt \
      --imgsz 640 --channels 4 --runs-count 200
  #      DS-YOLO: do thu cong bang cach chay 1 epoch evaluate() trong Python:
  #        import torch
  #        from models.ds_yolo import DSYOLOv8m
  #        from train_ds import load_golden, evaluate
  #        import time, yaml
  #        from ultralytics.data.build import build_yolo_dataset
  #        from ultralytics.cfg import get_cfg; from ultralytics.utils import DEFAULT_CFG
  #        from torch.utils.data import DataLoader
  #        ckpt = torch.load("runs/ds_yolo/ds_yolo/weights/best.pt", weights_only=False)
  #        model = DSYOLOv8m(num_classes=2).cuda()
  #        model.load_state_dict(ckpt["state_dict"])
  #        golden = load_golden("datasets/inhouse/golden/golden_ok.bmp", 640, "cuda")
  #        with open("dataset_fixed/data.yaml") as f: dy = yaml.safe_load(f)
  #        cfg = get_cfg(DEFAULT_CFG, overrides=dict(imgsz=640, batch=4, mode="val"))
  #        ds = build_yolo_dataset(cfg, dy["test"], 4, dy, mode="val")
  #        ldr = DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)
  #        t0 = time.perf_counter()
  #        evaluate(model, ldr, golden, "cuda", nc=2)
  #        print(f"DS-YOLO inference: {(time.perf_counter()-t0)/len(ds)*1000:.1f} ms/img")

  # [7d] Table V — robustness (chi can 2 CSV: rgb + rg-yolo)
  python eval/aggregate_results.py robust \
      --csvs   runs/robustness/rgb.csv runs/robustness/rg-yolo.csv \
      --labels "RGB baseline" "RG-YOLO"

  # [7e] Hoac chay tat ca qua run_all.py:
  python run_all.py --steps tables

---------------------------------------
BUOC 8: TAO FIGURES  [~30 phut]
---------------------------------------

  Paper co 4 figures:
    Fig 1  fig:dsyolo    DS-YOLO pipeline (TikZ) — DA CO SAN trong main.tex
    Fig 2  fig:datafrac  mAP vs training fraction (H3) — can: data_fraction.pdf
    Fig 3  fig:qual_diff Golden | Aligned | Diff map — can: qual_diff.pdf
    Fig 4  fig:qual_detect RGB vs DS-YOLO detection — can: qual_detect.pdf

  Tat ca output luu vao paper/figures/

  # [8a] Figure 2 — Data fraction (can xong Buoc 2 + 3)
  python eval/plot_data_fraction.py
  # Output: paper/figures/data_fraction.pdf

  # [8b] Figure 3 — Golden/Aligned/Diff (khong can model, can golden + dataset_fixed)
  python eval/plot_qual_figures.py \
      --data   dataset_fixed/data.yaml \
      --golden datasets/inhouse/golden/golden_ok.bmp \
      --imgsz  640 \
      --fig3-only
  # Output: paper/figures/qual_diff.pdf

  # [8c] Figure 4 — RGB vs DS-YOLO detection (can xong Buoc 2a + 2f)
  python eval/plot_qual_figures.py \
      --data    dataset_fixed/data.yaml \
      --golden  datasets/inhouse/golden/golden_ok.bmp \
      --imgsz   640 \
      --rgb     runs/detect/rgb/weights/best.pt \
      --ds-yolo runs/ds_yolo/ds_yolo/weights/best.pt
  # Output: paper/figures/qual_diff.pdf + paper/figures/qual_detect.pdf

  # [8d] Hoac chay ca 3 figure qua run_all.py:
  python run_all.py --steps figures

  LUU Y:
    - Kiem tra Figure 4 sau khi tao: chon dung anh co cau kien NG ro rang
    - Neu muon dung anh cu the: them --image path/to/test_img.jpg
    - Figure 2 se bao loi neu chua train fraction variants; chay Buoc 3 truoc

---------------------------------------
BUOC 9: HOAN THIEN PAPER
---------------------------------------

  Dien tat ca TODO con lai trong paper/main.tex:

  [ ] Table II: Ket qua tu Buoc 2 (copy paste tu aggregate_results.py main)
  [ ] Table III: Per-class mAP tu aggregate_results.py perclass
  [ ] Table IV: Latency do tren PC thuc te (Buoc 7c)
  [ ] Table V:  Robustness tu Buoc 4 (copy paste tu aggregate_results.py robust)
  [ ] GPU spec: Dien GPU dung de train (thay TODO trong Implementation Details)
  [ ] Email tac gia (2 author blocks)
  [ ] Acknowledgment
  [ ] Gia tri sigma_t crossover trong text robustness (~15 px)
  [ ] delta_t latency trong abstract va conclusion

  Them trich dan con thieu:
  [ ] 2-3 YOLO-for-PCB papers (Section II)
  [ ] 1 classical AOI paper (Section II)
  [ ] SolDef_A paper [cite] (Section VI.A)
  [ ] 1-2 OPC UA + CC-Link papers (Section II)


===============================================================================
CHAY NHANH VOI run_all.py (KHUYEN NGHI)
===============================================================================

  # Kiem tra lenh se chay (khong thuc hien)
  python run_all.py --dry-run

  # Chay tat ca tu dau
  python run_all.py

  # Chi chay buoc cu the
  python run_all.py --steps resplit build_rg train_ds

  # Buoc co san roi, chay tu buoc training tro di
  python run_all.py --steps train_all train_ds fraction robustness soldef_val tables

  # Buoc co san nhung muon chay lai
  python run_all.py --steps resplit --force


===============================================================================
LUU Y KY THUAT QUAN TRONG
===============================================================================

1. Chay resplit_inhouse.py TRUOC TAT CA experiment khac
2. golden_ok.bmp phai la anh cua chinh board do (khong dung SolDef_A lam golden)
3. SolDef_A chi dung cho: external val (Buoc 5) va optional pretrain (Buoc 6)
4. DS-YOLO chi test duoc tren in-house (can golden reference cung board)
5. Val va Test KHONG augment — chi dung anh goc de danh gia trung thuc
6. KHONG thay doi pipeline PLC/Robot — chi thay model va dataset


===============================================================================
TARGET VENUES
===============================================================================

1. NICS 2025  (NAFOSTED Conf. on Information and Computer Science, IEEE)
   -> Phu hop nhat voi scope va level hien tai

2. ATC 2025   (Advanced Technologies for Communications, IEEE)
   -> Backup neu NICS miss deadline

3. MAPR 2025  (Multimedia Analysis and Pattern Recognition, IEEE)
   -> Neu muon cao hon, can Buoc 6 va ket qua tot hon
