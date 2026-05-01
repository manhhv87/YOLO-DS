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

  Chay 5 variant tren dataset_fixed (yolov8m backbone, 60 epochs):

  # RGB baseline
  python train.py --variant rgb --size m \
      --data dataset_fixed/data.yaml \
      --epochs 60 --imgsz 640 --batch 4

  # Diff-only (kenh diff thay vi RGB)
  python train.py --variant diff-only --size m \
      --data dataset_fixed/data.yaml \
      --epochs 60 --imgsz 640 --batch 4

  # RG-YOLO (RGB + diff, 4 kenh)
  python train.py --variant rg-yolo --size m \
      --data dataset_rg/rg-yolo.yaml \
      --epochs 60 --imgsz 640 --batch 4

  # Stack6 (RGB + Golden, 6 kenh)
  python train.py --variant stack6 --size m \
      --data dataset_fixed/data.yaml \
      --epochs 60 --imgsz 640 --batch 4

  # F-Sub (feature-level fusion)
  python train.py --variant f-sub --size m \
      --data dataset_fixed/data.yaml \
      --epochs 60 --imgsz 640 --batch 4

  # DS-YOLO (Dual-Stream + CRFM — main contribution)
  python train_ds.py \
      --data     dataset_fixed/data.yaml \
      --golden   datasets/inhouse/golden/golden_ok.bmp \
      --variant  m \
      --epochs   60 --imgsz 640 --batch 4 \
      --name     ds_yolo \
      --save-dir runs/ds_yolo

  Metrics can co tu buoc nay:
    - Precision, Recall, mAP@0.5, mAP@0.5:0.95 moi variant -> Table II
    - runs/detect/<variant>/results.csv  (train.py variants)
    - runs/ds_yolo/ds_yolo/results.csv   (DS-YOLO)

---------------------------------------
BUOC 3: DATA FRACTION STUDY  [~1-2 ngay]
---------------------------------------

  Chung minh H3: gap lon hon khi data it

  python train.py --variant rgb --size m \
      --data dataset_fixed/data.yaml \
      --epochs 60 --imgsz 640 --batch 4 \
      --data-fraction 0.25 --name rgb_25pct
  python train.py --variant rgb --size m \
      --data dataset_fixed/data.yaml \
      --epochs 60 --imgsz 640 --batch 4 \
      --data-fraction 0.50 --name rgb_50pct
  python train.py --variant rgb --size m \
      --data dataset_fixed/data.yaml \
      --epochs 60 --imgsz 640 --batch 4 \
      --data-fraction 1.0  --name rgb_100pct

  python train_ds.py \
      --data          dataset_fixed/data.yaml \
      --golden        datasets/inhouse/golden/golden_ok.bmp \
      --variant       m \
      --epochs        60 --imgsz 640 --batch 4 \
      --data-fraction 0.25 --name ds_yolo_25pct \
      --save-dir      runs/ds_yolo
  # (tuong tu cho --data-fraction 0.50 --name ds_yolo_50pct)
  # (tuong tu cho --data-fraction 1.0  --name ds_yolo_100pct)

  Ket qua -> Figure: data fraction vs mAP -> ung ho H3

---------------------------------------
BUOC 4: ROBUSTNESS STUDY  [~1-2 ngay]
---------------------------------------

  Chung minh H3b: DS-YOLO ben vung hon RG-YOLO va RGB khi alignment bi loi

  Model can danh gia:
    - rgb      : baseline, khong dung golden -> khong bi anh huong alignment (diem tham chieu)
    - rg-yolo  : 4-ch dung diff map -> nhat cam voi alignment error
    - ds-yolo  : model de xuat, CRFM co the giam thieu anh huong alignment [QUAN TRONG]
    - diff-only, stack6, f-sub: tuy chon (neu co thoi gian)

  LUU Y: eval_robustness.py hien chi ho tro rgb va rg-yolo.
         Can bo sung ds-yolo vao eval_robustness.py truoc khi chay buoc nay.
         (Dung train_ds.evaluate() voi perturbed images + golden co dinh)

  # --imgsz phai khop voi kich thuoc anh dung khi train
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

  # ds-yolo: sau khi bo sung ho tro vao eval_robustness.py
  python eval/eval_robustness.py \
      --variant ds-yolo \
      --weights runs/ds_yolo/ds_yolo/weights/best.pt \
      --data    dataset_fixed/data.yaml \
      --golden  datasets/inhouse/golden/golden_ok.bmp \
      --out     runs/robustness/ds-yolo.csv \
      --imgsz   640

  Ket qua -> Table V (sigma_t=0,2,5,10,20 px vs mAP@0.5)
  Mong doi: rgb giam it, rg-yolo giam nhieu, ds-yolo giam it hon rg-yolo

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
BUOC 7: TONG HOP SO LIEU  [~1 ngay]
---------------------------------------

  # Table II — so sanh tat ca variant
  python eval/aggregate_results.py main \
      --runs     runs/detect \
      --ds-runs  runs/ds_yolo \
      --variants rgb diff-only stack6 f-sub rg-yolo ds-yolo

  # Table III — per-class OK/NG (can GPU)
  python eval/aggregate_results.py perclass \
      --runs     runs/detect \
      --ds-runs  runs/ds_yolo \
      --variants rgb rg-yolo ds-yolo \
      --data     dataset_fixed/data.yaml \
      --golden   datasets/inhouse/golden/golden_ok.bmp \
      --imgsz    640 \
      --split    test

  # Table IV — latency (chay tren moi variant can biet latency)
  python eval/aggregate_results.py latency \
      --weights    runs/detect/rgb/weights/best.pt \
      --imgsz      640 --channels 3 --runs-count 200
  python eval/aggregate_results.py latency \
      --weights    runs/detect/rg-yolo/weights/best.pt \
      --imgsz      640 --channels 4 --runs-count 200
  # DS-YOLO: do bang tay qua train_ds.evaluate() vi can golden arg

  # Table V — robustness (them ds-yolo.csv neu da chay buoc 4 day du)
  python eval/aggregate_results.py robust \
      --csvs   runs/robustness/rgb.csv runs/robustness/rg-yolo.csv runs/robustness/ds-yolo.csv \
      --labels "RGB baseline" "RG-YOLO" "DS-YOLO (ours)"

  Hoac chay tat ca qua run_all.py:
    python run_all.py --steps tables

---------------------------------------
BUOC 8: HOAN THIEN PAPER
---------------------------------------

  Dien tat ca TODO trong paper/main.tex:

  [ ] Table I:  Raw=75, Train/Val/Test = 53/11/11 source, ~148 aug train
  [ ] Table II: Ket qua tu Buoc 2
  [ ] Table III: Per-class mAP tu Buoc 2
  [ ] Table IV: Latency do tren PC thuc te (Buoc 7)
  [ ] Table V:  Robustness tu Buoc 4
  [ ] Figure:   Data fraction plot tu Buoc 3
  [ ] GPU spec: Dien GPU dung de train
  [ ] Email tac gia
  [ ] Acknowledgment

  Sua ngon ngu ve dataset:
    Thay: "211 labeled SMT boards"
    Bang: "75 raw captures of a single SMT board layout, augmented to
           ~148 training images via offline augmentation"

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
