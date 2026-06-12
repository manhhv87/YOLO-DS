"""Training script for DS-YOLO (Dual-Stream YOLOv8m with CRFM).

Manual PyTorch training loop using:
  * Ultralytics' YOLO dataset for image+label loading + augmentation,
  * Ultralytics' v8DetectionLoss for the detection loss,
  * a fixed golden reference image broadcast to the batch dimension.

Usage::

    python train_ds.py \\
        --data    dataset_fixed/data.yaml \\
        --golden  inhouse/golden/golden_ok.bmp \\
        --epochs 60 --imgsz 1024 --batch 4 \\
        --pretrained yolov8m.pt \\
        --name ds_yolo

Outputs land at ``runs/ds_yolo/<name>/``: ``best.pt``, ``last.pt`` and
``results.csv``. ``best.pt`` selects the epoch with the best validation
mAP@0.5; ``last.pt`` is the final-epoch state. Both store a plain
``state_dict`` together with the model class name and num_classes so the
UI inference helper can rebuild the model.

For evaluation (mAP@0.5) we use Ultralytics' YOLO validator on a
temporary on-disk dummy detector that wraps the same predictions as
DS-YOLO --- this avoids re-implementing mAP from scratch.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import ap_per_class, box_iou
# xywh2xyxy stays in utils.ops across versions; only NMS moved.
from ultralytics.utils.ops import xywh2xyxy

# CRITICAL FOR EVALUATOR FAIRNESS: use the SAME NMS that yolo.val() uses for the
# RGB/Stack6 baselines, called with multi_label=True. In ultralytics >=8.3
# non_max_suppression moved utils.ops -> utils.nms, so the old
# `from ultralytics.utils.ops import non_max_suppression` ALWAYS failed and the
# previous torchvision fallback ran with single-label (best-class) semantics —
# producing a structurally different prediction set than the multi_label
# baselines and a fake ~0.7pp per-variant offset. Prefer the real ultralytics
# NMS; only fall back to a (now multi_label-aware) torchvision shim if neither
# import exists.
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    try:
        from ultralytics.utils.ops import non_max_suppression
    except ImportError:
        import torchvision

        def non_max_suppression(
            prediction: torch.Tensor,
            conf_thres: float = 0.25,
            iou_thres: float = 0.45,
            max_det: int = 300,
            nc: int = 0,
            multi_label: bool = False,
            agnostic: bool = False,
            **_kwargs,
        ) -> list[torch.Tensor]:
            """torchvision NMS shim (last resort) — replicates Ultralytics'
            multi_label semantics so it stays comparable to yolo.val().
            prediction: [B, 4+nc, N] (channel-first Detect output) or [B, N, 4+nc].
            Returns a list of [K, 6]: x1, y1, x2, y2, conf, cls.
            """
            if prediction.ndim == 3 and prediction.shape[1] < prediction.shape[2]:
                prediction = prediction.transpose(1, 2)
            output: list[torch.Tensor] = []
            for x in prediction:                 # x: [N, 4+nc]
                boxes_xy = xywh2xyxy(x[:, :4])    # cx,cy,w,h -> x1,y1,x2,y2
                cls_scores = x[:, 4:]
                if multi_label and cls_scores.shape[1] > 1:
                    ii, jj = (cls_scores > conf_thres).nonzero(as_tuple=True)
                    boxes, conf, cls_idx = boxes_xy[ii], cls_scores[ii, jj], jj
                else:
                    conf, cls_idx = cls_scores.max(dim=1)
                    m = conf > conf_thres
                    boxes, conf, cls_idx = boxes_xy[m], conf[m], cls_idx[m]
                if boxes.shape[0] == 0:
                    output.append(torch.zeros(0, 6, device=x.device))
                    continue
                idxs = torch.zeros_like(cls_idx) if agnostic else cls_idx
                keep = torchvision.ops.batched_nms(
                    boxes.float(), conf.float(), idxs, iou_thres
                )[:max_det]
                output.append(torch.cat(
                    [boxes[keep], conf[keep, None], cls_idx[keep, None].float()], dim=1))
            return output

from models.ds_yolo import DSYOLOv8m, _LossCompatModel


# ---------------------------------------------------------------------------
# Exponential Moving Average (matches Ultralytics ModelEMA)
# ---------------------------------------------------------------------------
class ModelEMA:
    """EMA copy of model weights for stabler validation metrics.

    Decay ramps from ~0 up to `decay` over the first ~tau updates,
    identical to Ultralytics: d = decay*(1-exp(-updates/tau)).
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, tau: int = 2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_((1.0 - d) * msd[k].detach())


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True, help="Path to YOLO dataset YAML.")
    p.add_argument("--golden",     required=True, help="Path to golden reference image.")
    p.add_argument("--variant",    default="s", choices=["n", "s", "m", "l", "x"],
                   help="YOLOv8 variant (n/s/m/l/x). Default: s "
                        "(matches the YOLOv8s baselines used in the paper's Table II).")
    p.add_argument("--fusion",     default="crfm", choices=["crfm", "sub"],
                   help="Fusion type: crfm=DS-YOLO, sub=F-Sub ablation. Default: crfm.")
    # ---- CRFM design ablations (Section: CRFM design study) ----
    p.add_argument("--fuse-scales", dest="fuse_scales", default="p3,p4,p5",
                   help="Comma list of scales to apply CRFM at (subset of "
                        "p3,p4,p5). Fusion-location ablation, e.g. 'p5' or 'p4,p5'.")
    p.add_argument("--no-gate", dest="no_gate", action="store_true",
                   help="CRFM ablation: additive difference injection WITHOUT the "
                        "learned sigmoid gate.")
    p.add_argument("--fixed-alpha", dest="fixed_alpha", action="store_true",
                   help="CRFM ablation: fixed (non-learned) alpha=1 instead of the "
                        "learnable tanh scalar.")
    p.add_argument("--pretrained", default=None,
                   help="Pretrained weights path. Default: yolov8{variant}.pt")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--imgsz",      type=int, default=640)
    p.add_argument("--batch",      type=int, default=4)
    p.add_argument("--workers",    type=int, default=2)
    p.add_argument("--lr0",        type=float, default=0.01)
    p.add_argument("--momentum",   type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--name",       default="ds_yolo")
    p.add_argument("--save-dir",   default="runs/ds_yolo")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--data-fraction", type=float, default=1.0,
                   help="Fraction of training data to use (for the H3 study).")
    p.add_argument("--mosaic", type=float, default=0.0,
                   help="Mosaic prob in [0,1]. DEFAULT 0.0 because mosaic re-tiles 4 "
                        "captures into 1 frame and breaks the fixed alignment that "
                        "CRFM relies on. Only enable for diverse multi-board datasets.")
    p.add_argument("--fliplr", type=float, default=0.5,
                   help="Synced horizontal-flip probability in [0,1]. Implemented in "
                        "the training loop so capture, golden and bboxes are flipped "
                        "TOGETHER (Ultralytics' YOLODataset can only flip the capture "
                        "and would break CRFM alignment). 0.5 matches the RGB baseline. "
                        "Set to 0.0 to disable.")
    p.add_argument("--no-grad-golden", dest="no_grad_golden", action="store_true",
                   default=True,
                   help="Run the golden backbone pass under torch.no_grad() so its "
                        "activations are not retained for backprop. ~halves training "
                        "GPU memory at no quality cost. Default: ON.")
    p.add_argument("--grad-golden", dest="no_grad_golden", action="store_false",
                   help="Disable --no-grad-golden (debug only).")
    p.add_argument("--ema-tau", type=float, default=None,
                   help="EMA time constant. Default = max(2000, 0.3 * total_iterations). "
                        "Smaller value = faster EMA convergence. Override only for ablation.")
    p.add_argument("--no-ema", action="store_true",
                   help="Disable EMA entirely. Use raw model weights for validation. "
                        "Useful when training is too short for EMA to converge.")
    p.add_argument("--erasing", type=float, default=0.0,
                   help="Random-erasing probability for capture images (0=off). "
                        "DEFAULT 0.0: Ultralytics 'erasing' is classification-only, "
                        "so the detection baselines get NONE — applying it here "
                        "would be an uncontrolled augmentation on the 'ours' "
                        "variants. Only enable if you also add it to the baselines.")
    p.add_argument("--freeze-crfm", dest="freeze_crfm", type=int, default=0,
                   help="Freeze CRFM parameters for the first N epochs (phase-1/2 "
                        "training). Phase 1: backbone+neck+head train like the RGB "
                        "baseline so DFL converges fully. Phase 2: CRFM unfreezes "
                        "and learns on top. Recommended: int(0.7 * epochs). "
                        "Default 0 = no freezing (both phases merged).")
    return p.parse_args()


def random_erase(imgs: torch.Tensor, prob: float = 0.4) -> torch.Tensor:
    """Random erasing on capture images only (Zhong et al., 2020).

    Zeroes out a random rectangle in each image independently.
    Applied to the capture image ONLY so the capture↔golden alignment
    needed by CRFM is preserved.  Matches Ultralytics' erasing=0.4
    default so the data augmentation is equivalent to the RGB baseline.
    """
    if prob <= 0.0:
        return imgs
    B, C, H, W = imgs.shape
    for b in range(B):
        if torch.rand(1).item() > prob:
            continue
        for _ in range(10):  # up to 10 attempts to fit the rectangle
            area = H * W * (0.02 + torch.rand(1).item() * 0.31)  # 2-33% of image
            aspect = torch.empty(1).uniform_(0.3, 1 / 0.3).item()
            eh = int((area / aspect) ** 0.5)
            ew = int((area * aspect) ** 0.5)
            if 0 < eh < H and 0 < ew < W:
                y0 = torch.randint(0, H - eh, (1,)).item()
                x0 = torch.randint(0, W - ew, (1,)).item()
                imgs[b, :, y0:y0 + eh, x0:x0 + ew] = 0.0
                break
    return imgs


def load_golden(path: str, imgsz: int, device: str) -> torch.Tensor:
    """Load + resize + normalise the golden image as a (3, H, W) tensor."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read golden image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t.to(device)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_results(save_dir: Path, csv_path: Path, variant: str = "m") -> None:
    """Plot training curves from results.csv → results.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = []
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                rows.append({k: float(v) for k, v in row.items()})
        if len(rows) < 2:
            return

        epochs = [r["epoch"] for r in rows]
        keys = [
            ("train_loss",    "Train Loss"),
            ("box_loss",      "Box Loss"),
            ("cls_loss",      "Cls Loss"),
            ("dfl_loss",      "DFL Loss"),
            ("val_precision", "Precision"),
            ("val_recall",    "Recall"),
            ("val_map50",     "mAP@0.5"),
            ("val_map",       "mAP@0.5:0.95"),
        ]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, (key, label) in zip(axes.flat, keys):
            ax.plot(epochs, [r[key] for r in rows], linewidth=1.5)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"DS-YOLOv8{variant} Training Results", fontsize=13)
        fig.tight_layout()
        out = save_dir / "results.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {out}")
    except Exception as e:
        print(f"[plot] results.png failed: {e}")


def _plot_confusion_matrix(save_dir: Path, conf_mat: np.ndarray, names: dict) -> None:
    """Plot confusion matrix → confusion_matrix.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nc = conf_mat.shape[0] - 1
        labels = [names.get(i, str(i)) for i in range(nc)] + ["Background"]
        fig, ax = plt.subplots(figsize=(max(6, nc + 2), max(5, nc + 1)))
        im = ax.imshow(conf_mat, interpolation="nearest", cmap="Blues")
        ax.set_xticks(range(nc + 1))
        ax.set_yticks(range(nc + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (IoU@0.5)")
        plt.colorbar(im, ax=ax)
        thresh = conf_mat.max() / 2.0
        for i in range(nc + 1):
            for j in range(nc + 1):
                ax.text(j, i, str(conf_mat[i, j]), ha="center", va="center",
                        color="white" if conf_mat[i, j] > thresh else "black",
                        fontsize=9)
        fig.tight_layout()
        out = save_dir / "confusion_matrix.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {out}")
    except Exception as e:
        print(f"[plot] confusion_matrix.png failed: {e}")


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: DSYOLOv8m,
    val_loader: DataLoader,
    golden: torch.Tensor,
    device: str,
    nc: int,
    conf: float = 0.001,
    iou_nms: float = 0.7,
    max_det: int = 300,
    save_dir: Optional[Path] = None,
    names: Optional[dict] = None,
) -> dict:
    """Compute precision / recall / mAP@0.5 on the validation set."""
    model.eval()
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats: dict[str, list] = dict(tp_b=[], conf=[], pred_cls=[], target_cls=[])
    conf_mat = np.zeros((nc + 1, nc + 1), dtype=int)
    n_imgs_val = 0
    n_insts_val = 0
    n_imgs_per_cls = np.zeros(nc, dtype=int)
    t_pre = t_inf = t_post = 0.0

    pbar = tqdm(val_loader, leave=False, bar_format="{l_bar}{bar:10}{r_bar}")
    for batch in pbar:
        t0 = time.perf_counter()
        imgs = batch["img"].to(device).float() / 255.0
        targets = torch.cat(
            [batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]],
            dim=1,
        ).to(device)

        t1 = time.perf_counter()
        n_imgs_val += len(imgs)
        n_insts_val += int(targets.shape[0])
        for si2 in range(len(imgs)):
            for c in np.unique(targets[targets[:, 0] == si2, 1].cpu().numpy().astype(int)):
                if 0 <= c < nc:
                    n_imgs_per_cls[c] += 1
        preds = model(imgs, golden)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        # In eval mode the Detect head returns (decoded, raw_per_scale_list).
        decoded = preds[0] if isinstance(preds, (tuple, list)) else preds
        # Match yolo.val() exactly: per-class NMS with multi_label=True so the
        # same model produces the same candidate set as the Ultralytics
        # baselines (each box can emit one detection per class above conf_thres).
        try:
            outputs = non_max_suppression(decoded, conf, iou_nms,
                                          nc=nc, multi_label=(nc > 1),
                                          agnostic=False, max_det=max_det)
        except TypeError:
            outputs = non_max_suppression(decoded, conf, iou_nms, max_det=max_det)
        t3 = time.perf_counter()
        t_pre += t1 - t0
        t_inf += t2 - t1
        t_post += t3 - t2

        for si, output in enumerate(outputs):
            t = targets[targets[:, 0] == si][:, 1:]  # cls, xywh (normalized)
            if t.numel():
                t_xyxy = xywh2xyxy(t[:, 1:5]) * torch.tensor(
                    [imgs.shape[3], imgs.shape[2]] * 2, device=device
                )
                t_cls = t[:, 0]
            else:
                t_xyxy = torch.empty(0, 4, device=device)
                t_cls = torch.empty(0, device=device)

            if not len(output):
                if len(t_cls):
                    stats["target_cls"].append(t_cls.cpu().numpy())
                    for c in t_cls.cpu().numpy():
                        conf_mat[int(c), nc] += 1  # FN: GT -> background
                continue

            pred_xyxy = output[:, :4]
            pred_conf = output[:, 4]
            pred_cls  = output[:, 5]

            tp = torch.zeros(len(output), len(iouv), dtype=torch.bool, device=device)
            matched_gt_set: set = set()
            matched_pd_set: set = set()
            if len(t_xyxy):
                ious = box_iou(t_xyxy, pred_xyxy)
                correct_class = t_cls.unsqueeze(1) == pred_cls.unsqueeze(0)
                for i, iou_th in enumerate(iouv):
                    matches = (ious >= iou_th) & correct_class
                    if matches.any():
                        m_idx = matches.nonzero(as_tuple=False)
                        if len(m_idx) > 1:
                            m_idx = m_idx[ious[m_idx[:, 0], m_idx[:, 1]].argsort(descending=True)]
                            m_idx = m_idx[np.unique(m_idx[:, 1].cpu().numpy(), return_index=True)[1]]
                            m_idx = m_idx[np.unique(m_idx[:, 0].cpu().numpy(), return_index=True)[1]]
                        tp[m_idx[:, 1], i] = True
                        if i == 0:  # IoU=0.5: populate confusion matrix
                            for gi, pi in m_idx.cpu().numpy():
                                conf_mat[int(t_cls[gi]), int(pred_cls[pi])] += 1
                                matched_gt_set.add(int(gi))
                                matched_pd_set.add(int(pi))
                for gi in range(len(t_cls)):  # FN: unmatched GT
                    if gi not in matched_gt_set:
                        conf_mat[int(t_cls[gi]), nc] += 1
            for pi in range(len(pred_cls)):   # FP: unmatched predictions
                if pi not in matched_pd_set:
                    conf_mat[nc, int(pred_cls[pi])] += 1

            stats["tp_b"].append(tp.cpu().numpy())
            stats["conf"].append(pred_conf.cpu().numpy())
            stats["pred_cls"].append(pred_cls.cpu().numpy())
            stats["target_cls"].append(t_cls.cpu().numpy())

    pbar.close()
    stats = {k: np.concatenate(v, 0) if len(v) else np.array([]) for k, v in stats.items()}

    _empty_pc = dict(classes=[], p=[], r=[], ap50=[], ap=[],
                     n_imgs=[], n_inst=[])
    if stats["tp_b"].size == 0:
        return dict(precision=0.0, recall=0.0, map50=0.0, map=0.0,
                    n_images=n_imgs_val, n_instances=n_insts_val,
                    per_class=_empty_pc,
                    speed=dict(preprocess=0.0, inference=0.0, postprocess=0.0))

    names_map = names or {i: str(i) for i in range(nc)}
    if save_dir is not None:
        results = ap_per_class(
            stats["tp_b"], stats["conf"], stats["pred_cls"], stats["target_cls"],
            plot=True, save_dir=save_dir, names=names_map,
        )
        _plot_confusion_matrix(save_dir, conf_mat, names_map)
    else:
        results = ap_per_class(
            stats["tp_b"], stats["conf"], stats["pred_cls"], stats["target_cls"],
        )
    p, r, ap, unique_cls = results[2], results[3], results[5], results[6]
    n_inst_per_cls = np.bincount(stats["target_cls"].astype(int), minlength=nc)
    per_class = dict(
        classes=unique_cls.tolist(),
        p=p.tolist(),
        r=r.tolist(),
        ap50=ap[:, 0].tolist(),
        ap=ap.mean(1).tolist(),
        n_imgs=[int(n_imgs_per_cls[c]) for c in unique_cls],
        n_inst=[int(n_inst_per_cls[c]) for c in unique_cls],
    )
    n = max(n_imgs_val, 1)
    speed = dict(
        preprocess=t_pre / n * 1e3,
        inference=t_inf / n * 1e3,
        postprocess=t_post / n * 1e3,
    )
    return dict(
        precision=float(p.mean()),
        recall=float(r.mean()),
        map50=float(ap[:, 0].mean()),
        map=float(ap.mean()),
        n_images=n_imgs_val,
        n_instances=n_insts_val,
        per_class=per_class,
        speed=speed,
    )


# ---------------------------------------------------------------------------
# CRFM freeze / unfreeze helper
# ---------------------------------------------------------------------------
def _set_crfm_grad(model: "DSYOLOv8m", enabled: bool) -> None:
    """Enable or disable gradient computation for all three CRFM modules."""
    for m in [model.crfm_p3, model.crfm_p4, model.crfm_p5]:
        if hasattr(m, "parameters"):
            for p in m.parameters():
                p.requires_grad_(enabled)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    # Full determinism so reported mAP is reproducible and the multi-seed study
    # (P1) measures real architectural variance rather than RNG drift.
    import random as _random
    _random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_dir = Path(args.save_dir) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    # ---- Build model ---------------------------------------------------
    nc = 2  # binary OK / NG (overridden by data.yaml below)
    import yaml as _yaml
    with open(args.data) as f:
        data_yaml = _yaml.safe_load(f)
    nc = int(data_yaml.get("nc", len(data_yaml.get("names", [0, 1]))))
    names_map = data_yaml.get("names", {i: str(i) for i in range(nc)})
    if isinstance(names_map, list):
        names_map = {i: n for i, n in enumerate(names_map)}

    _fuse_scales = tuple(s.strip() for s in args.fuse_scales.split(",") if s.strip())
    model = DSYOLOv8m(num_classes=nc, variant=args.variant, fusion=args.fusion,
                      fuse_scales=_fuse_scales, use_gate=not args.no_gate,
                      learn_alpha=not args.fixed_alpha).to(args.device)
    if args.fusion == "crfm":
        print(f"[info] CRFM ablation: fuse_scales={_fuse_scales} "
              f"gate={'on' if not args.no_gate else 'OFF'} "
              f"alpha={'learnable' if not args.fixed_alpha else 'fixed=1'}")
    report = model.load_yolov8_pretrained(args.pretrained)
    print(f"[info] pretrained: copied {report['copied']} / {report['total_src']} keys\n")

    n_params = sum(p.numel() for p in model.parameters())
    n_grad   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        from thop import profile as thop_profile
        _dummy_cap  = torch.zeros(1, 3, args.imgsz, args.imgsz).to(args.device)
        _dummy_gold = torch.zeros(1, 3, args.imgsz, args.imgsz).to(args.device)
        _flops, _ = thop_profile(model, inputs=(_dummy_cap, _dummy_gold), verbose=False)
        _gflops = f"{_flops / 1e9:.1f} GFLOPs"
    except Exception:
        _gflops = "n/a"
    model.model_info(gflops=_gflops)
    print()

    # Strip thop's bookkeeping buffers (``total_ops``, ``total_params``) that
    # ``thop.profile`` injects into every submodule.  Leaving them on the
    # model would pollute every ``state_dict()`` call and break loading later
    # via ``load_state_dict(strict=True)``.
    for _module in model.modules():
        for _k in list(_module._buffers.keys()):
            if _k in ("total_ops", "total_params"):
                _module._buffers.pop(_k, None)

    # ---- Phase-1/2 training: freeze CRFM for first N epochs -----------
    # Phase 1 (epochs 0..freeze_crfm-1): CRFM frozen → model ≡ RGB baseline
    #   → DFL converges to the same quality as the RGB model (mAP50-95 ~0.826).
    # Phase 2 (epoch freeze_crfm..end): CRFM unfrozen → CRFM learns on top
    #   of a well-trained DFL head, adding detection quality without hurting
    #   localization precision.
    # Without freezing, CRFM gate gradients compete with DFL gradients from
    # epoch 1, preventing full DFL convergence → mAP50-95 plateau ~0.78.
    if args.freeze_crfm > 0:
        _set_crfm_grad(model, False)
        print(f"[info] phase-1/2: CRFM frozen for epochs 1-{args.freeze_crfm}, "
              f"unfreezes at epoch {args.freeze_crfm + 1}")

    # ---- Loss adapter --------------------------------------------------
    loss_model = _LossCompatModel(model).to(args.device)
    loss_fn = v8DetectionLoss(loss_model)

    # ---- Dataloaders ---------------------------------------------------
    # Ultralytics' build_yolo_dataset returns a YOLODataset; we then wrap
    # it in a standard PyTorch DataLoader.
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    cfg = get_cfg(DEFAULT_CFG, overrides=dict(
        imgsz=args.imgsz, batch=args.batch, mode="train",
        rect=False, cache=False,
        # CRITICAL: mosaic re-tiles 4 captures into 1 frame, destroying the
        # fixed geometric alignment between capture and golden that CRFM
        # depends on.  Set to 0.0 for single-product / fixed-fixture lines.
        mosaic=args.mosaic,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        # CRITICAL: fliplr must be 0 for DS-YOLO — Ultralytics flips the
        # capture image but NOT the golden reference.  A flipped capture
        # produces mirror-image feature maps that differ everywhere from the
        # unflipped golden, making the diff signal pure noise for half the
        # training batches and actively harming CRFM learning.
        flipud=0.0, fliplr=0.0,
        # No geometric aug -- any spatial distortion breaks capture↔golden
        # alignment and corrupts the diff signal that CRFM relies on.
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    ))
    data_root = Path(data_yaml.get("path", str(Path(args.data).parent)))
    if not data_root.is_absolute():
        data_root = Path(args.data).parent / data_root
    train_img = str(data_root / data_yaml["train"])
    val_img   = str(data_root / data_yaml["val"])
    train_ds = build_yolo_dataset(cfg, train_img, args.batch, data_yaml, mode="train")
    val_ds   = build_yolo_dataset(cfg, val_img,   args.batch, data_yaml, mode="val")

    if args.data_fraction < 1.0:
        n = max(1, int(len(train_ds) * args.data_fraction))
        indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.seed))[:n].tolist()
        train_ds = torch.utils.data.Subset(train_ds, indices)
        collate_fn = train_ds.dataset.collate_fn
        print(f"[info] data-fraction={args.data_fraction}: using {n}/{len(train_ds.dataset)} training images")
    else:
        collate_fn = train_ds.collate_fn

    def _seed_worker(worker_id: int) -> None:
        wseed = (args.seed + worker_id) % (2 ** 32)
        np.random.seed(wseed)
        import random as _r
        _r.seed(wseed)

    _loader_gen = torch.Generator()
    _loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn,
                              pin_memory=True, generator=_loader_gen,
                              worker_init_fn=_seed_worker)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, collate_fn=val_ds.collate_fn,
                              pin_memory=True)

    # Test loader (held-out, only used for final reporting after training).
    test_loader = None
    if "test" in data_yaml:
        test_img = str(data_root / data_yaml["test"])
        test_ds  = build_yolo_dataset(cfg, test_img, args.batch, data_yaml, mode="val")
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=args.workers, collate_fn=test_ds.collate_fn,
                                 pin_memory=True)

    # ---- Golden ---------------------------------------------------------
    golden = load_golden(args.golden, args.imgsz, args.device)

    # ---- Optimizer + scheduler -----------------------------------------
    # Split params into 3 groups (matches Ultralytics): BN+bias (no decay), weights (decay)
    # + a 4th catch-all group for any remaining params (e.g. CRFM alpha scalars).
    g_bn, g_bias, g_weights = [], [], []
    _grouped_ids: set[int] = set()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            g_bn.append(m.weight)
            _grouped_ids.add(id(m.weight))
        if hasattr(m, 'bias') and isinstance(m.bias, torch.nn.Parameter):
            g_bias.append(m.bias)
            _grouped_ids.add(id(m.bias))
        if hasattr(m, 'weight') and isinstance(m.weight, torch.nn.Parameter) \
                and not isinstance(m, torch.nn.BatchNorm2d):
            g_weights.append(m.weight)
            _grouped_ids.add(id(m.weight))
    # Catch-all: named scalars like CRFM's alpha that have neither .weight nor .bias
    g_other = [p for p in model.parameters() if id(p) not in _grouped_ids]
    optimizer = torch.optim.SGD(
        g_bias + g_bn, lr=args.lr0, momentum=args.momentum, nesterov=True,
    )
    # Match Ultralytics' weight-decay scaling: decay = wd * batch * accumulate /
    # nbs(64). At batch=4 (accumulate=16) this equals wd exactly; the scaling
    # keeps the two loops matched at ANY batch size (so the H3 fraction study or
    # an OOM-driven batch change can't silently re-introduce a wd confound).
    _accum = max(1, round(64 / args.batch))
    _wd_eff = args.weight_decay * args.batch * _accum / 64.0
    optimizer.add_param_group({'params': g_weights, 'weight_decay': _wd_eff})
    if g_other:
        optimizer.add_param_group({'params': g_other})
        print(f"[info] optimizer: {len(g_other)} extra param(s) in catch-all group "
              f"(e.g. CRFM alpha × {len(g_other)})")

    # LR schedule: Ultralytics one_cycle ((1-cos(e*pi/E))/2*(lrf-1)+1) so the
    # cosine SHAPE matches trainer.py (verified: e=50/E=100 -> 0.505, e=10 ->
    # 0.976, identical to Ultralytics). The previous schedule restarted the
    # cosine at epoch 3 over a shorter range, diverging by up to ~2.4pp of the
    # LR multiplier mid-training. A short linear warmup is applied as a
    # multiplier over the first warmup_epochs.
    lrf = 0.01
    warmup_epochs = 3
    warmup_momentum = 0.8
    def _one_cycle(e: int) -> float:
        return ((1 - math.cos(e * math.pi / max(args.epochs, 1))) / 2) * (lrf - 1) + 1
    def lr_lambda(e: int) -> float:
        if e < warmup_epochs:
            return _one_cycle(e) * (e + 1) / warmup_epochs
        return _one_cycle(e)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Gradient accumulation: simulate nominal batch=64 (matches Ultralytics nbs=64)
    accumulate = max(1, round(64 / args.batch))
    # Mixed precision
    use_amp = (args.device != "cpu") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # EMA: tau auto-shrinks on short runs so the EMA actually converges before
    # the final epoch.  Heuristic: tau ≈ 30 % of total optimiser updates,
    # floored at 50 (enough to smooth per-batch noise) and capped at 2000
    # (Ultralytics default for long training schedules).
    # NOTE: the old floor of 200 caused EMA contamination on short runs
    # (300 total updates, tau=200 → only 77% converged at end, leaving
    # 23% of initial pretrained weights in the EMA → DFL head pha loang
    # → mAP@50-95 thap).  Floor=50 gives >99% convergence at 300 updates.
    total_updates = math.ceil(len(train_loader) / accumulate) * args.epochs
    if args.ema_tau is not None:
        ema_tau = float(args.ema_tau)
    else:
        ema_tau = max(50.0, min(2000.0, 0.3 * total_updates))
    print(f"[info] EMA tau = {ema_tau:.0f} (total_updates ≈ {total_updates}, "
          f"override with --ema-tau)")
    ema = ModelEMA(model, tau=ema_tau)
    close_mosaic = 10  # disable mosaic in final N epochs (matches Ultralytics)

    # ---- CSV logger ----------------------------------------------------
    csv_path = save_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "box_loss", "cls_loss", "dfl_loss",
             "val_precision", "val_recall", "val_map50", "val_map", "lr"]
        )

    # ---- Train ---------------------------------------------------------
    _TRAIN_HDR = (
        f"{'Epoch':>10}{'GPU_mem':>10}{'box_loss':>10}{'cls_loss':>10}"
        f"{'dfl_loss':>10}{'Instances':>12}{'Size':>8}"
    )

    # Reset peak memory once before training; we'll read it back at the end and
    # include it in test_results.json so the paper's resource table is reproducible.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Best-checkpoint selection metric: use mAP@0.5:0.95 (val['map']) to MATCH
    # Ultralytics' fitness (w=[0,0,0,1] in this version), so all four variants
    # are checkpoint-selected on the SAME metric. Selecting on mAP@0.5 here while
    # the Ultralytics baselines select on mAP@0.5:0.95 biased DS-YOLO toward its
    # own reported mAP@0.5 column.
    best_fitness = -1.0
    for epoch in range(args.epochs):
        model.train()
        running = torch.zeros(4, device=args.device)
        n_batches = 0
        n_inst_ep = 0

        # Momentum warmup (epoch granularity) to match Ultralytics
        # warmup_momentum 0.8 -> momentum over the first warmup_epochs.
        if epoch < warmup_epochs:
            _mom = warmup_momentum + (args.momentum - warmup_momentum) * (epoch + 1) / warmup_epochs
        else:
            _mom = args.momentum
        for _pg in optimizer.param_groups:
            if "momentum" in _pg:
                _pg["momentum"] = _mom

        # Phase-2: unfreeze CRFM at the designated epoch.
        if args.freeze_crfm > 0 and epoch == args.freeze_crfm:
            _set_crfm_grad(model, True)
            if not args.no_ema:
                _set_crfm_grad(ema.ema, True)
            print(f"\n[info] Epoch {epoch+1}: CRFM unfrozen — phase-2 begins "
                  f"(DFL has converged; CRFM now learns on top)")

        # Close mosaic in the final `close_mosaic` epochs (matches Ultralytics)
        if epoch == args.epochs - close_mosaic:
            _ds = train_loader.dataset
            _ds = getattr(_ds, 'dataset', _ds)  # unwrap Subset if needed
            if hasattr(_ds, 'mosaic'):
                _ds.mosaic = 0.0
                print(f"\n[info] Epoch {epoch+1}: mosaic disabled for final {close_mosaic} epochs")

        print(_TRAIN_HDR)
        optimizer.zero_grad()
        pbar_t = tqdm(train_loader, total=len(train_loader),
                      bar_format="{l_bar}{bar:10}{r_bar}")
        for i, batch in enumerate(pbar_t):
            imgs = batch["img"].to(args.device).float() / 255.0

            # Synced horizontal flip, PER-IMAGE (matches Ultralytics RandomFlip).
            # Per-batch flipping gave only 2 outcomes/batch vs 2^B for the
            # baselines — a strictly less diverse augmentation. We flip each
            # sample independently and pair each flipped capture with a flipped
            # golden (built as a per-sample [B,3,H,W] tensor) so CRFM alignment
            # is preserved; the model accepts a batched golden (ds_yolo forward).
            golden_step = golden
            if args.fliplr > 0:
                B = imgs.shape[0]
                flip_mask = torch.rand(B) < args.fliplr          # CPU, seeded
                if flip_mask.any():
                    dmask = flip_mask.to(imgs.device)
                    imgs[dmask] = torch.flip(imgs[dmask], dims=[-1])
                    golden_step = golden.unsqueeze(0).expand(B, -1, -1, -1).clone()
                    golden_step[dmask] = torch.flip(golden_step[dmask], dims=[-1])
                    # mirror x_center of bboxes whose image was flipped
                    if batch["bboxes"].numel():
                        sel = flip_mask[batch["batch_idx"].view(-1).long()]
                        batch["bboxes"][sel, 0] = 1.0 - batch["bboxes"][sel, 0]

            # Random erasing on capture only (golden not modified so
            # capture↔golden alignment for CRFM is preserved).
            if args.erasing > 0:
                imgs = random_erase(imgs, prob=args.erasing)

            n_inst_ep += int(batch["cls"].shape[0])
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(imgs, golden_step,
                              golden_no_grad=args.no_grad_golden)
                loss, loss_items = loss_fn(preds, batch)  # loss_items = (box, cls, dfl)
            loss_scalar = loss.sum() if loss.ndim > 0 else loss
            scaler.scale(loss_scalar / accumulate).backward()

            if (i + 1) % accumulate == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if not args.no_ema:
                    ema.update(model)

            running[0] += loss_scalar.detach()
            running[1:] += loss_items.detach()
            n_batches += 1

            mem = (f"{torch.cuda.memory_reserved()/1e9:.3g}G"
                   if torch.cuda.is_available() else "")
            pbar_t.set_description(
                f"{f'{epoch+1}/{args.epochs}':>10}{mem:>10}"
                f"{float(running[1]/n_batches):>10.4g}"
                f"{float(running[2]/n_batches):>10.4g}"
                f"{float(running[3]/n_batches):>10.4g}"
                f"{n_inst_ep//n_batches:>12}{args.imgsz:>8}"
            )
        pbar_t.close()

        running /= max(n_batches, 1)
        train_loss, box_l, cls_l, dfl_l = [float(x) for x in running]
        val_model = model if args.no_ema else ema.ema
        val = evaluate(val_model, val_loader, golden, args.device, nc=nc)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        n_vi   = val.get("n_images", 0)
        n_inst = val.get("n_instances", 0)
        _hdr = '%22s' + '%11s' * 6
        _row = '%22s' + '%11i' * 2 + '%11.3g' * 4
        print(_hdr % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)'))
        print(_row % ('all', n_vi, n_inst,
                      val['precision'], val['recall'], val['map50'], val['map']))

        # Log alpha values (CRFM gate activation diagnostic).
        _crfm_modules = [("p3", model.crfm_p3), ("p4", model.crfm_p4), ("p5", model.crfm_p5)]
        _alpha_parts = [f"{tag}={float(m.alpha.item()):.4f}"
                        for tag, m in _crfm_modules if hasattr(m, "alpha")]
        if _alpha_parts:
            print(f"  [CRFM alpha] {', '.join(_alpha_parts)}")

        # Log CSV
        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [epoch + 1, f"{train_loss:.4f}", f"{box_l:.4f}", f"{cls_l:.4f}", f"{dfl_l:.4f}",
                 f"{val['precision']:.4f}", f"{val['recall']:.4f}",
                 f"{val['map50']:.4f}", f"{val['map']:.4f}", f"{lr:.6f}"]
            )

        # Save checkpoints
        ckpt = dict(
            model_class="DSYOLOv8m",
            fusion=args.fusion,
            num_classes=nc,
            state_dict=(model.state_dict() if args.no_ema else ema.ema.state_dict()),
            epoch=epoch + 1,
            metrics=val,
            args=vars(args),
        )
        torch.save(ckpt, weights_dir / "last.pt")
        if val["map"] > best_fitness:          # mAP@0.5:0.95 == Ultralytics fitness
            best_fitness = val["map"]
            torch.save(ckpt, weights_dir / "best.pt")

    # ---- Final validation on best.pt ------------------------------------
    best_pt = weights_dir / "best.pt"
    print(f"\nValidating {best_pt}...")
    best_ckpt = torch.load(best_pt, map_location=args.device, weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    n_layers = sum(1 for _ in model.modules())
    print(f"DS-YOLOv8{args.variant} summary (dual-stream, shared backbone): "
          f"{n_layers} layers, {n_params:,} parameters, "
          f"{n_grad:,} gradients, {_gflops}")

    val_final = evaluate(model, val_loader, golden, args.device, nc=nc,
                         save_dir=save_dir, names=names_map)
    _hdr = '%22s' + '%11s' * 6
    _row = '%22s' + '%11i' * 2 + '%11.3g' * 4
    print(_hdr % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)'))
    print(_row % ('all', val_final['n_images'], val_final['n_instances'],
                  val_final['precision'], val_final['recall'],
                  val_final['map50'], val_final['map']))
    for i, cls_idx in enumerate(val_final.get("per_class", {}).get("classes", [])):
        cname = names_map.get(int(cls_idx), str(int(cls_idx)))
        pc = val_final["per_class"]
        print(_row % (cname, pc['n_imgs'][i], pc['n_inst'][i],
                      pc['p'][i], pc['r'][i], pc['ap50'][i], pc['ap'][i]))
    sp = val_final.get("speed", {})
    print(f"Speed: {sp.get('preprocess',0):.1f}ms preprocess, "
          f"{sp.get('inference',0):.1f}ms inference, "
          f"0.0ms loss, {sp.get('postprocess',0):.1f}ms postprocess per image")
    print(f"Results saved to {save_dir}")

    # ---- Test set evaluation (final, unbiased reporting for paper) ------
    if test_loader is not None:
        print(f"\nEvaluating on test split...")
        test_val_dir = save_dir.parent / f"val_{save_dir.name}"
        test_val_dir.mkdir(exist_ok=True)
        val_test = evaluate(model, test_loader, golden, args.device, nc=nc,
                            save_dir=test_val_dir, names=names_map)
        print(_hdr % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)'))
        print(_row % ('all', val_test['n_images'], val_test['n_instances'],
                      val_test['precision'], val_test['recall'],
                      val_test['map50'], val_test['map']))
        for i, cls_idx in enumerate(val_test.get("per_class", {}).get("classes", [])):
            cname = names_map.get(int(cls_idx), str(int(cls_idx)))
            pc = val_test["per_class"]
            print(_row % (cname, pc['n_imgs'][i], pc['n_inst'][i],
                          pc['p'][i], pc['r'][i], pc['ap50'][i], pc['ap'][i]))
        sp_t = val_test.get("speed", {})
        print(f"Speed: {sp_t.get('preprocess',0):.1f}ms preprocess, "
              f"{sp_t.get('inference',0):.1f}ms inference, "
              f"0.0ms loss, {sp_t.get('postprocess',0):.1f}ms postprocess per image")
        peak_mem_gb = (float(torch.cuda.max_memory_reserved()) / 1e9
                       if torch.cuda.is_available() else 0.0)
        test_out = save_dir / "test_results.json"
        with test_out.open("w") as f:
            json.dump(dict(
                precision=val_test["precision"],
                recall=val_test["recall"],
                map50=val_test["map50"],
                map=val_test["map"],
                split="test",
                params=int(n_params),
                gflops=_gflops,
                peak_gpu_mem_gb=peak_mem_gb,
                no_grad_golden=bool(args.no_grad_golden),
                fliplr=float(args.fliplr),
                ema=bool(not args.no_ema),
            ), f, indent=2)
        print(f"Test results saved to {test_out} "
              f"(peak GPU mem reserved: {peak_mem_gb:.2f} GB)")

    # ---- Plots ----------------------------------------------------------
    _plot_results(save_dir, csv_path, variant=args.variant)


if __name__ == "__main__":
    main()
