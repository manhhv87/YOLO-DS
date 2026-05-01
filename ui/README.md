# UI — Automated Visual Inspection Application

PyQt5 application that runs the inspection pipeline on the SMT cell:
camera capture → align to golden → YOLO detection → OK/NG decision →
OPC UA write to the PLC.

## Layout

```
ui/
├── main.py                  entry point (`python main.py`)
├── window_camera.py         Qt main window (camera viewer, results, controls)
├── requirements.txt
├── README.md
│
├── core/                    runtime modules
│   ├── algorithm.py           ORB/SIFT/SURF align, CLAHE, PCB cropping, YOLO infer
│   ├── camera.py              Basler GigE Vision interface (pypylon)
│   ├── opc_client.py          OPC UA client (sends OK/NG to the PLC)
│   ├── pipeline.py            legacy 8-class YOLOv8m inspection pipeline
│   ├── pipeline_ds.py         DS-YOLO inspection pipeline (binary OK/NG)
│   ├── ds_inference.py        DS-YOLO checkpoint loader + NMS wrapper
│   └── vision_opc_server.py   OPC UA server-side for triggers
│
├── runs/detect/train/weights/   <-- the deployed YOLO weights live here
│   ├── best.pt                  copy from train/runs/<your-run>/weights/best.pt
│   └── last.pt
│
├── stored_images/               runtime image storage
│   ├── golden/                  reference images
│   │   ├── golden2.bmp            <-- the one used by pipeline.py
│   │   ├── golden_ok.bmp
│   │   └── golden_ng.png
│   ├── raw_images/              every camera frame is saved here
│   ├── final_images/            after CLAHE + align + crop
│   ├── infer_images/            with YOLO bounding boxes drawn
│   └── result/                  final OK/NG visualisations
│
└── golden_points.json           hand-curated component positions on the golden
```

## Run

```
cd ui
pip install -r requirements.txt

# Legacy 8-class detector (default):
python main.py

# DS-YOLO (binary OK/NG, recommended after re-training under ../train/):
USE_DS_YOLO=1 python main.py    # bash / zsh / Git Bash
$env:USE_DS_YOLO=1; python main.py    # PowerShell
```

The `USE_DS_YOLO=1` switch makes `window_camera.py` import
`core.pipeline_ds.main_pipeline_ds` instead of the legacy
`core.pipeline.main_pipeline`. DS-YOLO loads its weights from
`runs/ds_yolo/weights/best.pt` (copy them there after training).

> **Pylon SDK note.** `pypylon` requires the Basler Pylon runtime to be
> installed system-wide; install it from the Basler website before
> `pip install pypylon`.

## Deploying a freshly trained model

After running a training job under [`../train/`](../train/), copy the
resulting checkpoints into the path that `pipeline.py` reads from:

```
cp ../train/runs/detect/<run-name>/weights/best.pt runs/detect/train/weights/best.pt
cp ../train/runs/detect/<run-name>/weights/last.pt runs/detect/train/weights/last.pt
```

`pipeline.py` resolves the model path relative to the `ui/` folder using
`BASE_DIR = os.path.dirname(os.path.dirname(__file__))`, so as long as
the binary is at `ui/runs/detect/train/weights/best.pt` no code change
is needed.

## OPC UA endpoint

`window_camera.py` defaults to:

```python
OPC_URL = "opc.tcp://127.0.0.1:49320"
NODE_REQ    = "ns=2;s=Channel1.Device1.bit_REQ"
NODE_BUSY   = "ns=2;s=Channel1.Device1.bit_BUSY"
NODE_RESULT = "ns=2;s=Channel1.Device1.word_RESULT"
NODE_SENSOR = "ns=2;s=Channel1.Device1.bit_SENSOR"
```

Edit those constants to match your KEPserverEX / PLC tag layout.

## Class taxonomy used at inference

`pipeline.py` was developed against the original 8-class taxonomy
(`cap_OK`, `cap_NG`, `cd4017_OK`, ..., `res_NG`) and the bundled
`runs/detect/train/weights/best.pt` is trained for that taxonomy. It
checks `cls_name.endswith("_OK")` / `endswith("_NG")` to make
inspection decisions, which works for both the 8-class and the
2-class binary models — but if you deploy a **2-class** model trained
under [`../train/`](../train/), update `CLASS_NAMES` in
[`core/pipeline.py`](core/pipeline.py) accordingly (e.g.
`CLASS_NAMES = ["OK", "NG"]`) so the suffix check still triggers.
