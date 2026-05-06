# Ke hoach hoan thien paper: DS-YOLO for SMT Defect Detection
# Updated: 2026-05-06

===============================================================================
TONG QUAN (doc truoc khi chay)
===============================================================================

Goal: train + benchmark DS-YOLO + cac variant cho paper, tat ca chay duoc tu
`run_all.py` hoac chay tay tung lenh ben duoi.

Cau hinh thuc te (lay tu  runs/<...>/args.yaml  va checkpoint metadata):

  - YOLO size  : `s` (yolov8s.pt — fits Colab T4 / RTX 3080+ for batch=4)
  - epochs     : 100
  - imgsz      : 640
  - batch      : 4
  - golden ref : `datasets/inhouse/golden/golden_inhouse.jpg`

DS-YOLO defaults moi (xem PHU LUC A neu muon hieu why):
  - `--fliplr 0.5`         : synced flip cap + golden + bbox trong loop
  - `--no-grad-golden`     : no_grad cho backbone golden -> ~halves train mem
  - `--ema-tau` auto       : tau = clamp(0.3 * total_updates, 200, 2000)


===============================================================================
FILES (chi liet ke nhung gi THUC SU co trong repo)
===============================================================================

train/
|-- run_all.py             [Master runner; idempotent; --dry-run, --force]
|-- train.py               [Ultralytics variants: rgb, diff-only, stack6]
|-- train_ds.py            [DS-YOLO + F-Sub (custom dual-stream loop)]
|
|-- models/
|   |-- ds_yolo.py         [DSYOLOv8m + CRFM + SubFusion]
|   `-- model_4ch.py       [patch_first_conv() — used by stack6]
|
|-- data/
|   |-- resplit_inhouse.py    [BUOC 1: fix data leakage]
|   |-- convert_soldef_to_yolo.py  [LabelMe -> YOLO; 1 lan, da chay]
|   `-- remap_to_binary.py    [8-class -> 2-class; 1 lan, da chay]
|
|-- eval/
|   |-- eval_robustness.py    [Table V: rgb vs ds-yolo duoi alignment noise]
|   |-- aggregate_results.py  [LaTeX rows cho Table II / III / IV / V]
|   |-- plot_data_fraction.py [Figure 2: H3 study]
|   `-- plot_qual_figures.py  [Figure 3 + Figure 4]
|
|-- datasets/
|   |-- inhouse/
|   |   |-- leaky/        [Roboflow goc, BI LEAKY, KHONG dung truc tiep]
|   |   `-- golden/golden_inhouse.jpg
|   `-- soldef/           [DA CONVERT: 300/64/64 train/val/test]
|
|-- dataset_fixed/        [TU DONG TAO boi data/resplit_inhouse.py]
`-- runs/                 [TU DONG TAO khi training]


===============================================================================
WORKFLOW NHANH (khuyen nghi: chay qua run_all.py)
===============================================================================

  # Tren Windows:
  cd C:\Users\manhh\Desktop\PCB\train

  # Tren Colab (Google Drive):
  # from google.colab import drive; drive.mount('/content/drive')
  # %cd /content/drive/MyDrive/PCB/train

  # In tat ca lenh se chay (khong thuc thi)
  python run_all.py --dry-run

  # Chay het tu dau (~6-9 ngay tren 1 GPU)
  python run_all.py

  # Chi 1 buoc cu the
  python run_all.py --steps train_ds
  python run_all.py --steps fraction
  python run_all.py --steps tables

  # Chay lai du output da co
  python run_all.py --steps train_ds --force

Cac buoc co san trong run_all.py (theo thu tu):
  resplit       Re-split inhouse theo source-image-level (fix leakage)
  train_all     train.py cho rgb, diff-only, stack6
  train_ds      train_ds.py cho ds_yolo (CRFM) va f_sub (sub fusion)
  fraction      H3 study: rgb_25/50pct + ds_yolo_25/50pct
  robustness    eval_robustness.py: rgb va ds-yolo voi sigma_t = 0..20 px
  soldef_val    train.py rgb tren SolDef_A (external generalisation)
  tables        aggregate_results.py main / perclass / latency / robust
  figures       plot_data_fraction.py + plot_qual_figures.py


===============================================================================
CHAY TAY TUNG BUOC (neu khong dung run_all.py)
===============================================================================

---------------------------------------
BUOC 1. FIX DATA LEAKAGE  [~30 phut]
---------------------------------------

  python data/resplit_inhouse.py `
      --src       datasets/inhouse/leaky `
      --dst       dataset_fixed `
      --val-frac  0.15 `
      --test-frac 0.15 `
      --seed      0

  Ket qua: dataset_fixed/{train,val,test}/{images,labels}/ + data.yaml.

---------------------------------------
BUOC 2. TRAIN CAC VARIANTS  [~3-5 ngay]
---------------------------------------

  # 2a. RGB baseline (Ultralytics pipeline)
  python train.py --variant rgb --size s `
      --data dataset_fixed/data.yaml `
      --epochs 100 --imgsz 640 --batch 4

  # 2b. Diff-only va Stack6 (ablation)
  python train.py --variant diff-only --size s `
      --data dataset_fixed/data.yaml `
      --epochs 100 --imgsz 640 --batch 4

  python train.py --variant stack6 --size s `
      --data dataset_fixed/data.yaml `
      --epochs 100 --imgsz 640 --batch 4

  # 2c. DS-YOLO (main contribution)
  python train_ds.py `
      --data    dataset_fixed/data.yaml `
      --golden  datasets/inhouse/golden/golden_inhouse.jpg `
      --variant s --fusion crfm `
      --epochs 100 --imgsz 640 --batch 4 `
      --fliplr 0.5 --no-grad-golden `
      --name ds_yolo --save-dir runs/ds_yolo

  # 2d. F-Sub (raw feature subtraction ablation)
  python train_ds.py `
      --data    dataset_fixed/data.yaml `
      --golden  datasets/inhouse/golden/golden_inhouse.jpg `
      --variant s --fusion sub `
      --epochs 100 --imgsz 640 --batch 4 `
      --fliplr 0.5 --no-grad-golden `
      --name f_sub --save-dir runs/ds_yolo

  Output:
    runs/detect/{rgb,diff-only,stack6}/test_results.json
    runs/ds_yolo/{ds_yolo,f_sub}/test_results.json

  Note: test_results.json cua train_ds.py giu them: params, gflops,
  peak_gpu_mem_gb, no_grad_golden, fliplr, ema -> dung cho Table IV.

---------------------------------------
BUOC 3. DATA-FRACTION STUDY (H3)
---------------------------------------

  # RGB tai 25% va 50% (100% lay tu Buoc 2a)
  python train.py --variant rgb --size s `
      --data dataset_fixed/data.yaml `
      --epochs 100 --imgsz 640 --batch 4 `
      --data-fraction 0.25 --name rgb_25pct
  python train.py --variant rgb --size s `
      --data dataset_fixed/data.yaml `
      --epochs 100 --imgsz 640 --batch 4 `
      --data-fraction 0.50 --name rgb_50pct

  # DS-YOLO tai 25% va 50% (100% lay tu Buoc 2c)
  python train_ds.py `
      --data dataset_fixed/data.yaml `
      --golden datasets/inhouse/golden/golden_inhouse.jpg `
      --variant s --fusion crfm `
      --epochs 100 --imgsz 640 --batch 4 `
      --fliplr 0.5 --no-grad-golden `
      --data-fraction 0.25 --name ds_yolo_25pct --save-dir runs/fraction
  python train_ds.py `
      --data dataset_fixed/data.yaml `
      --golden datasets/inhouse/golden/golden_inhouse.jpg `
      --variant s --fusion crfm `
      --epochs 100 --imgsz 640 --batch 4 `
      --fliplr 0.5 --no-grad-golden `
      --data-fraction 0.50 --name ds_yolo_50pct --save-dir runs/fraction

---------------------------------------
BUOC 4. ROBUSTNESS STUDY  [~2-4 gio]
---------------------------------------

  python eval/eval_robustness.py `
      --variant rgb `
      --weights runs/detect/rgb/weights/best.pt `
      --data    dataset_fixed/data.yaml `
      --imgsz   640 `
      --out     runs/robustness/rgb.csv

  python eval/eval_robustness.py `
      --variant ds-yolo `
      --weights runs/ds_yolo/ds_yolo/weights/best.pt `
      --data    dataset_fixed/data.yaml `
      --golden  datasets/inhouse/golden/golden_inhouse.jpg `
      --imgsz   640 `
      --out     runs/robustness/ds_yolo.csv

  Mong doi: ds-yolo > rgb tai sigma_t=0; gap thu hep khi sigma_t tang.

---------------------------------------
BUOC 5. SOLDEF_A EXTERNAL VALIDATION  [~1 ngay]
---------------------------------------

  Quan trong: --full-aug.  SolDef co nhieu board layouts va KHONG co
  golden reference, nen rang buoc "tat geometric aug + tat mosaic" cua
  in-house KHONG ap dung.  --full-aug bat lai mosaic=1.0, translate=0.1,
  scale=0.5 (Ultralytics defaults) de model generalize tot hon.

  python train.py --variant rgb --size s `
      --data   datasets/soldef/data.yaml `
      --epochs 100 --imgsz 640 --batch 4 `
      --full-aug `
      --name   rgb_soldef

  -> Cau: "RGB YOLOv8s dat mAP@0.5 = X.XX tren SolDef_A, chung minh
     backbone co the transfer sang board layout khac."

  LUU Y: run rgb_soldef hien tai trong  runs/detect/rgb_soldef/  duoc
  train TRUOC khi co --full-aug -> mosaic=0, geometric aug=0.  Neu can
  bao cao chinh thuc, re-train voi co --full-aug:
      python run_all.py --steps soldef_val --force

---------------------------------------
BUOC 6. TONG HOP SO LIEU
---------------------------------------

  # Table II
  python eval/aggregate_results.py main `
      --runs    runs/detect `
      --ds-runs runs/ds_yolo `
      --variants rgb diff-only stack6 f-sub ds-yolo

  # Table III (per-class, can GPU)
  python eval/aggregate_results.py perclass `
      --runs    runs/detect `
      --ds-runs runs/ds_yolo `
      --variants rgb ds-yolo `
      --data    dataset_fixed/data.yaml `
      --golden  datasets/inhouse/golden/golden_inhouse.jpg `
      --imgsz   640 --split test

  # Table IV (latency)
  python eval/aggregate_results.py latency `
      --weights runs/detect/rgb/weights/best.pt `
      --imgsz 640 --channels 3 --runs-count 200

  # Table V (robustness)
  python eval/aggregate_results.py robust `
      --csvs   runs/robustness/rgb.csv runs/robustness/ds_yolo.csv `
      --labels "RGB baseline" "DS-YOLO (ours)"

---------------------------------------
BUOC 7. FIGURES
---------------------------------------

  # Figure 2 — Data fraction (H3)
  python eval/plot_data_fraction.py
  # Output: paper/figures/data_fraction.pdf

  # Figure 3 — Golden / Aligned / Diff
  python eval/plot_qual_figures.py `
      --data    dataset_fixed/data.yaml `
      --golden  datasets/inhouse/golden/golden_inhouse.jpg `
      --imgsz   640 --fig3-only

  # Figure 4 — RGB vs DS-YOLO detection
  python eval/plot_qual_figures.py `
      --data    dataset_fixed/data.yaml `
      --golden  datasets/inhouse/golden/golden_inhouse.jpg `
      --imgsz   640 `
      --rgb     runs/detect/rgb/weights/best.pt `
      --ds-yolo runs/ds_yolo/ds_yolo/weights/best.pt


===============================================================================
PHU LUC A. TAI SAO DS-YOLO YEU HON RGB TREN mAP@0.5:0.95 ?
===============================================================================

So lieu tu  runs/<...>/test_results.json :

  Variant         Backbone  mAP@0.5   mAP@0.5:0.95
  ----------------------------------------------------
  RGB             yolov8s   0.9861    0.8264
  Diff-only       yolov8s   0.9861    0.8264   (= rgb, xem note)
  Stack6          yolov8s   0.9843    0.8192
  F-Sub           ds-s      0.9737    0.7540
  DS-YOLO         ds-s      0.9855    0.7960
  RGB  25%        yolov8s   0.5668    0.4201
  RGB  50%        yolov8s   0.9923    0.8072
  DS-YOLO  25%    ds-s      0.9890    0.7741
  DS-YOLO  50%    ds-s      0.9776    0.7893

Bon nguyen nhan duoc chan doan va da sua trong code:

  (1) BAT DOI XUNG VE AUGMENTATION
      RGB qua YOLO.train() mac dinh fliplr=0.5, erasing=0.4, hsv jitter.
      DS-YOLO truoc day ep cung fliplr=0 (so phai mirror hoa golden) ->
      mat ~50% diversity -> mAP@0.5:0.95 thap.
      [FIX] train_ds.py them --fliplr 0.5: flip dong bo cap+golden+bbox.

  (2) CRFM ALPHA FROZEN: softplus(-6) gradient vanishing
      alpha_raw = -6, alpha = softplus(-6) ≈ 0.0025.
      d(alpha)/d(alpha_raw) = sigmoid(-6) ≈ 0.0025 -> gradient toi alpha_raw
      qua nho de cap nhat: alpha dong bang o 0.0025 suot 100 epoch.
      [FIX] doi sang tanh:  alpha_raw = 0, alpha = tanh(alpha_raw).
        tanh(0) = 0 -> identity tuyet doi tai init.
        d(tanh)/d(alpha_raw)|0 = 1 -> gradient day du tu epoch 1.
        tanh bounded (-1, 1) -> khong blow up.
      backward compat: from_checkpoint tu dong convert softplus-era alpha_raw
      (raw < -2) sang tanh domain qua atanh(softplus(raw)).

  (3) EMA TAU = 2000 QUA LON CHO TRAIN NGAN
      ~225 optimiser updates / 100 epochs => decay reach ~0.10.
      EMA gan nhu giu nguyen weight init -> validation under-report.
      [FIX] tau auto = clamp(0.3 * total_updates, 200, 2000),
      override bang --ema-tau, hoac tat hoan toan bang --no-ema.

  (4) HEAD KHONG DUOC COPY TU PRETRAINED
      Truoc day idx 22 (Detect head) bi loai khoi  load_yolov8_pretrained.
      [FIX] them idx 22 vao mapping.  Shape filter giu lai tat ca conv +
      BN cua  head.cv2  (box) va  head.cv3  (cls featuriser).  Chi 1×1
      projection cuoi cung cua cv3 (out_ch=80 != nc=2) bi loai do shape
      mismatch.  RGB baseline (qua YOLO()) hanh xu y het.

Kich thuoc cu the cua moi sua: xem section C ben duoi.


===============================================================================
PHU LUC B. TAI SAO GPU MEMORY CAO HON DU PARAMS/GFLOPs THAP HON ?
===============================================================================

GPU memory training = parameters + GRADIENT + ACTIVATIONS + optimizer + EMA.
Phan dominate trong YOLO la ACTIVATION memory (~70-80%).

  (a) DUAL BACKBONE FORWARD
      Backbone chay 2 lan (capture & golden).  Du weight chia se,
      ACTIVATION TENSOR cua moi forward la rieng biet va ca hai duoc
      retain cho backprop -> ~2x activation footprint cua YOLOv8s.

  (b) PARAMS / GFLOPs CUA DS-YOLO-s VS YOLOv8m PAPER REFERENCE
      DS-YOLO-s: 11.2M params + 70K CRFM ≈ 11.3M params, ~57 GFLOPs
                  (thop count cho ca 2 forward pass).
      YOLOv8m baseline (paper refers): 25.9M params, 78 GFLOPs.
      Vi the params THAP hon nhung activation thi CAO hon.

  [FIX] --no-grad-golden: chay golden trong torch.no_grad() ->
  KHONG retain activation cho golden -> halves training-time mem.
  Gradient van flow vao shared backbone qua nhanh capture (du).
  Voi imgsz=640 batch=4 : ~5.5 GB -> ~2.8 GB peak reserved.


===============================================================================
PHU LUC C. CAC THAY DOI CODE (commit 2026-05-06)
===============================================================================

models/ds_yolo.py:
  - CrossRefFusion.alpha_raw init = -6.0  -> alpha ≈ 2.5e-3 (identity init)
  - forward(...) them golden_no_grad=True (default ON)
  - load_yolov8_pretrained: them idx 22 -> Detect head box+cls featuriser
    duoc copy tu pretrained.

train_ds.py:
  - Default --variant s, --epochs 100, --imgsz 640 (khop checkpoint thuc te)
  - Them --fliplr (default 0.5) — synced flip cho cap, golden, bbox
  - Them --no-grad-golden / --grad-golden
  - Them --ema-tau auto + --no-ema
  - test_results.json moi: params, gflops, peak_gpu_mem_gb,
    no_grad_golden, fliplr, ema

run_all.py:
  - SIZE = "s", EPOCHS = 100 (khop voi runs/ thuc te)
  - _train_ds(...) tu dong them --fliplr va --no-grad-golden
  - step_fraction dung _train_ds() cho 25/50%
  - GOLDEN = golden_inhouse.jpg


===============================================================================
PHU LUC D. PAPER TODO (main.tex)
===============================================================================

  [ ] Table II: copy paste tu  aggregate_results.py main
  [ ] Table III: per-class mAP tu  aggregate_results.py perclass
  [ ] Table IV: latency tu  aggregate_results.py latency  + train mem
                tu test_results.json["peak_gpu_mem_gb"]
  [ ] Table V: robustness tu  aggregate_results.py robust
  [ ] GPU spec: dien GPU dung de train (Implementation Details)
  [ ] Email tac gia (2 author blocks)
  [ ] Acknowledgment
  [ ] sigma_t crossover trong robustness narrative
  [ ] delta_t latency trong abstract va conclusion

  Trich dan con thieu:
  [ ] 2-3 YOLO-for-PCB papers (Section II)
  [ ] 1 classical AOI paper (Section II)
  [ ] SolDef_A paper [cite] (Section VI.A)
  [ ] 1-2 OPC UA + CC-Link papers (Section II)


===============================================================================
LUU Y KY THUAT
===============================================================================

  1. Chay  resplit_inhouse.py  TRUOC TAT CA cac experiment khac.
  2. golden_inhouse.jpg phai la anh cua chinh board do (khong dung SolDef_A
     lam golden).
  3. SolDef_A chi dung cho external validation (Buoc 5).
  4. DS-YOLO chi test duoc tren in-house dataset (can golden cung board).
  5. Val va Test KHONG augment — chi anh goc, danh gia trung thuc.
  6. Diff-only hien tai cho ket qua trung khit voi RGB (cung dataset 3-ch);
     se y nghia hon neu compute diff map offline va luu thanh dataset rieng,
     nhung hien tai chua co script do — neu paper khong can ablation nay,
     co the bo qua khoi Table II.

===============================================================================
TARGET VENUES
===============================================================================

  1. NICS 2025  (NAFOSTED Conf. on Information and Computer Science, IEEE)
     -> Phu hop nhat voi scope va level hien tai
  2. ATC 2025   (Advanced Technologies for Communications, IEEE)
     -> Backup neu NICS miss deadline
  3. MAPR 2025  (Multimedia Analysis and Pattern Recognition, IEEE)
     -> Neu muon cao hon
