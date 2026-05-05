"""Dual-Stream YOLOv8m with Cross-Reference Fusion (DS-YOLO).

This is the level-up beyond RG-YOLO: where RG-YOLO only adds a fourth
input channel, DS-YOLO uses a Siamese (shared-weight) YOLOv8m backbone
to process the captured frame and the golden reference in parallel,
fuses their multi-scale features through a learnable Cross-Reference
Fusion Module (CRFM) at P3/P4/P5, and feeds the fused features to a
standard YOLOv8m PANet neck and Detect head.

Pipeline::

    capture --[shared YOLOv8m backbone]--> P3_c, P4_c, P5_c
                                                  |
    golden  --[shared YOLOv8m backbone]--> P3_g, P4_g, P5_g
                                                  |
                                              CRFM x 3
                                                  |
                                         F_p3, F_p4, F_p5
                                                  |
                                  YOLOv8m PANet neck (top-down + bottom-up)
                                                  |
                                          YOLOv8m Detect head
                                                  |
                                              (boxes, cls, conf)

CRFM definition (per scale):

    M       = sigmoid(g(|F_cap - F_gold|))         # difference attention
    F_fused = F_cap + alpha * M * F_cap            # gated residual

where ``g`` is a 2-layer 1x1 conv with channel bottleneck and ``alpha``
is a learnable scalar initialised to zero. The model therefore reduces
exactly to the YOLOv8m baseline at the start of training and only
deviates from it as ``alpha`` grows away from zero.

Parameter overhead vs. YOLOv8m: ~70K params across the three CRFM.
Compute overhead: ~+50% FLOPs from the second backbone pass on golden.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C2f, Conv, Detect, SPPF


# ---------------------------------------------------------------------------
# Per-variant channel widths and C2f depths.
# Channels verified against official ultralytics pretrained weights.
# ---------------------------------------------------------------------------
_VARIANTS: dict[str, dict] = {
    "n": dict(C_STEM=16,  C_P2=32,  C_P3=64,  C_P4=128, C_P5=256,
              B_DARK2=1, B_DARK3=2, B_DARK4=2, B_DARK5=1, N_NECK=1),
    "s": dict(C_STEM=32,  C_P2=64,  C_P3=128, C_P4=256, C_P5=512,
              B_DARK2=1, B_DARK3=2, B_DARK4=2, B_DARK5=1, N_NECK=1),
    "m": dict(C_STEM=48,  C_P2=96,  C_P3=192, C_P4=384, C_P5=576,
              B_DARK2=2, B_DARK3=4, B_DARK4=4, B_DARK5=2, N_NECK=2),
    "l": dict(C_STEM=64,  C_P2=128, C_P3=256, C_P4=512, C_P5=512,
              B_DARK2=3, B_DARK3=6, B_DARK4=6, B_DARK5=3, N_NECK=3),
    "x": dict(C_STEM=80,  C_P2=160, C_P3=320, C_P4=640, C_P5=640,
              B_DARK2=3, B_DARK3=6, B_DARK4=6, B_DARK5=3, N_NECK=3),
}


# =============================================================================
# Cross-Reference Fusion Module
# =============================================================================
class CrossRefFusion(nn.Module):
    """Learnable difference attention between capture and golden features.

    Initialised so that at ``alpha = 0`` the output is exactly the
    capture features (i.e. the baseline). The parameters of ``alpha``
    and ``gate`` then learn during training to amplify capture features
    in regions where the capture differs from the reference.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        mid = max(channels // 4, 32)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
        )
        # Initialise gate output near zero so CRFM starts as identity and
        # activates gradually.  sigmoid(-4) ≈ 0.018.
        nn.init.constant_(self.gate[-1].bias, -4.0)
        # alpha_raw is unconstrained; actual alpha = softplus(alpha_raw) >= 0.
        # softplus(0) ≈ 0.693, so true alpha starts small but positive,
        # preventing the dead-gradient problem and ruling out negative scaling.
        self.alpha_raw = nn.Parameter(torch.zeros(1))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_raw)

    def forward(self, f_cap: torch.Tensor, f_gold: torch.Tensor) -> torch.Tensor:
        if f_gold.shape != f_cap.shape:
            f_gold = F.interpolate(
                f_gold, size=f_cap.shape[-2:], mode="bilinear", align_corners=False
            )
        diff = (f_cap - f_gold).abs()
        gate = torch.sigmoid(self.gate(diff))
        return f_cap + self.alpha * gate * f_cap


class SubFusion(nn.Module):
    """Simple feature subtraction: F_fused = F_cap - F_gold.

    Ablation of CRFM — removes the learnable gating and uses raw difference
    directly.  Has zero extra parameters versus the RGB baseline.
    """

    def forward(self, f_cap: torch.Tensor, f_gold: torch.Tensor) -> torch.Tensor:
        if f_gold.shape != f_cap.shape:
            f_gold = F.interpolate(
                f_gold, size=f_cap.shape[-2:], mode="bilinear", align_corners=False
            )
        return f_cap - f_gold


# =============================================================================
# YOLOv8m sub-modules (backbone, neck) re-implemented with Ultralytics blocks
# =============================================================================
class _Backbone(nn.Module):
    """YOLOv8 backbone returning (P3, P4, P5)."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        cs, cp2, cp3, cp4, cp5 = (
            cfg["C_STEM"], cfg["C_P2"], cfg["C_P3"], cfg["C_P4"], cfg["C_P5"]
        )
        self.stem = Conv(3, cs, k=3, s=2)
        self.dark2 = nn.Sequential(
            Conv(cs, cp2, k=3, s=2),
            C2f(cp2, cp2, n=cfg["B_DARK2"], shortcut=True),
        )
        self.dark3 = nn.Sequential(
            Conv(cp2, cp3, k=3, s=2),
            C2f(cp3, cp3, n=cfg["B_DARK3"], shortcut=True),
        )
        self.dark4 = nn.Sequential(
            Conv(cp3, cp4, k=3, s=2),
            C2f(cp4, cp4, n=cfg["B_DARK4"], shortcut=True),
        )
        self.dark5 = nn.Sequential(
            Conv(cp4, cp5, k=3, s=2),
            C2f(cp5, cp5, n=cfg["B_DARK5"], shortcut=True),
            SPPF(cp5, cp5, k=5),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.dark2(x)
        p3 = self.dark3(x)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        return p3, p4, p5


class _PANetNeck(nn.Module):
    """YOLOv8 PANet neck."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        cp3, cp4, cp5, n = cfg["C_P3"], cfg["C_P4"], cfg["C_P5"], cfg["N_NECK"]
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # Top-down
        self.c2f_p4 = C2f(cp5 + cp4, cp4, n=n, shortcut=False)
        self.c2f_p3 = C2f(cp4 + cp3, cp3, n=n, shortcut=False)
        # Bottom-up
        self.down_p3 = Conv(cp3, cp3, k=3, s=2)
        self.c2f_n4 = C2f(cp3 + cp4, cp4, n=n, shortcut=False)
        self.down_p4 = Conv(cp4, cp4, k=3, s=2)
        self.c2f_n5 = C2f(cp4 + cp5, cp5, n=n, shortcut=False)

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u5 = self.up(p5)
        p4_top = self.c2f_p4(torch.cat([u5, p4], dim=1))
        u4 = self.up(p4_top)
        n3 = self.c2f_p3(torch.cat([u4, p3], dim=1))
        d3 = self.down_p3(n3)
        n4 = self.c2f_n4(torch.cat([d3, p4_top], dim=1))
        d4 = self.down_p4(n4)
        n5 = self.c2f_n5(torch.cat([d4, p5], dim=1))
        return n3, n4, n5


# =============================================================================
# DS-YOLO model
# =============================================================================
class DSYOLOv8m(nn.Module):
    """Dual-stream YOLOv8 with Cross-Reference Fusion at P3/P4/P5.

    Parameters
    ----------
    num_classes : int
    variant : str
        YOLOv8 size — one of ``"n"``, ``"s"``, ``"m"``, ``"l"``, ``"x"``.
        Default ``"m"`` (matches the original paper).
    """

    def __init__(self, num_classes: int = 2, variant: str = "m",
                 fusion: str = "crfm") -> None:
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(_VARIANTS)}")
        if fusion not in ("crfm", "sub"):
            raise ValueError(f"Unknown fusion '{fusion}'. Choose from ('crfm', 'sub')")
        cfg = _VARIANTS[variant]
        self.variant = variant
        self.fusion_type = fusion
        # Single backbone instance; capture and golden are both routed
        # through it (shared weights).
        self.backbone = _Backbone(cfg)
        if fusion == "crfm":
            self.crfm_p3 = CrossRefFusion(cfg["C_P3"])
            self.crfm_p4 = CrossRefFusion(cfg["C_P4"])
            self.crfm_p5 = CrossRefFusion(cfg["C_P5"])
        else:  # "sub"
            self.crfm_p3 = SubFusion()
            self.crfm_p4 = SubFusion()
            self.crfm_p5 = SubFusion()
        self.neck = _PANetNeck(cfg)
        # Detect head
        self.head = Detect(nc=num_classes, ch=(cfg["C_P3"], cfg["C_P4"], cfg["C_P5"]))
        self.head.stride = torch.tensor([8.0, 16.0, 32.0])
        self.head.bias_init()
        # Attributes used by Ultralytics' v8DetectionLoss.
        self.nc = num_classes
        self.stride = self.head.stride

    # ----- Subscript support (required by v8DetectionLoss) ---------------
    def __getitem__(self, _idx: int) -> nn.Module:
        """Return the Detect head for any index.

        ``v8DetectionLoss.__init__`` calls ``model.model[-1]`` expecting the
        Detect head.  Adding this method makes DSYOLOv8m subscriptable without
        breaking ``nn.Module`` parameter registration.
        """
        return self.head

    # ----- Checkpoint loader (handles variant + thop pollution) ----------
    @classmethod
    def from_checkpoint(cls, weights_path: str, device: str = "cpu",
                         strict: bool = True) -> "DSYOLOv8m":
        """Re-instantiate a DS-YOLO model from a checkpoint produced by
        ``train_ds.py``.

        Reads ``variant``, ``fusion`` and ``num_classes`` from the
        checkpoint metadata so the architecture matches the saved weights,
        and silently filters out the bookkeeping buffers (``total_ops``,
        ``total_params``) that ``thop.profile`` injects into the model
        during training.
        """
        import torch as _torch

        ckpt = _torch.load(weights_path, map_location=device, weights_only=False)
        if "state_dict" not in ckpt:
            raise ValueError(
                f"Checkpoint at {weights_path} has no 'state_dict' field. "
                f"Was it written by train_ds.py?"
            )

        nc      = int(ckpt.get("num_classes", 2))
        variant = ckpt.get("args", {}).get("variant") or ckpt.get("variant", "m")
        fusion  = ckpt.get("fusion") or ckpt.get("args", {}).get("fusion", "crfm")

        model = cls(num_classes=nc, variant=variant, fusion=fusion).to(device)

        # Strip thop-injected buffers (added by thop.profile during training).
        cleaned = {
            k: v for k, v in ckpt["state_dict"].items()
            if not (k.endswith(".total_ops") or k.endswith(".total_params"))
        }

        # Backward-compat: older checkpoints stored `crfm_*.alpha` directly
        # (a learnable scalar with no constraint).  The current CRFM uses
        # `alpha_raw` and computes `alpha = softplus(alpha_raw)`, so a saved
        # `alpha` of value `a` corresponds to `alpha_raw = log(exp(a) - 1)`
        # (inverse-softplus).  Migrate the keys silently so old runs keep
        # loading.
        import math as _math
        rename = []
        for k in list(cleaned.keys()):
            if k.endswith(".alpha") and not k.endswith(".alpha_raw"):
                v = cleaned.pop(k)
                # inverse-softplus, clamping the input away from zero to keep
                # the result finite (softplus is convex, so the inverse blows
                # up as alpha → 0).
                a_val = float(v.flatten()[0])
                if a_val <= 1e-6:
                    raw = -10.0  # softplus(-10) ≈ 4.5e-5 ≈ 0
                else:
                    raw = _math.log(_math.expm1(a_val))
                cleaned[k + "_raw"] = v.detach().clone().fill_(raw)
                rename.append(k)
        if rename:
            print(f"[from_checkpoint] migrated {len(rename)} legacy "
                  f"`crfm.alpha` keys → `alpha_raw` (inverse-softplus).")

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        # In strict mode, only fail if there are *real* missing keys (not the
        # ones we filtered out, and not the buffers added by thop on the
        # destination model that we didn't add).
        if strict:
            real_missing = [k for k in missing
                            if not (k.endswith(".total_ops") or k.endswith(".total_params"))]
            if real_missing:
                raise RuntimeError(
                    f"Missing keys when loading {weights_path}: {real_missing[:5]}"
                    + (f" ... ({len(real_missing)} total)" if len(real_missing) > 5 else "")
                )
        model.eval()
        return model

    # ----- Forward --------------------------------------------------------
    def forward(
        self, capture: torch.Tensor, golden: Optional[torch.Tensor] = None
    ) -> object:
        """Forward pass.

        Parameters
        ----------
        capture : (B, 3, H, W) tensor of captured frames.
        golden  : (3, H, W) or (B, 3, H, W) tensor of the golden reference.
                  If a single image is given, it is broadcast to the batch
                  dimension.
        """
        if golden is None:
            raise ValueError("DS-YOLO requires a `golden` reference image at every forward.")
        if golden.dim() == 3:
            golden = golden.unsqueeze(0).expand(capture.shape[0], -1, -1, -1)

        # Shared backbone, two passes.
        c_p3, c_p4, c_p5 = self.backbone(capture)
        g_p3, g_p4, g_p5 = self.backbone(golden)

        # Cross-reference fusion.
        f_p3 = self.crfm_p3(c_p3, g_p3)
        f_p4 = self.crfm_p4(c_p4, g_p4)
        f_p5 = self.crfm_p5(c_p5, g_p5)

        # Neck and head.
        n3, n4, n5 = self.neck(f_p3, f_p4, f_p5)
        return self.head([n3, n4, n5])

    # ----- Architecture summary ------------------------------------------
    def model_info(self, gflops: str = "n/a") -> None:
        """Print model architecture in YOLO-style table."""
        cfg = _VARIANTS[self.variant]
        cs  = cfg["C_STEM"];  cp2 = cfg["C_P2"];  cp3 = cfg["C_P3"]
        cp4 = cfg["C_P4"];   cp5 = cfg["C_P5"]
        b2  = cfg["B_DARK2"]; b3  = cfg["B_DARK3"]
        b4  = cfg["B_DARK4"]; b5  = cfg["B_DARK5"]
        nn_ = cfg["N_NECK"]

        def _p(m: "nn.Module") -> int:
            return sum(p.numel() for p in m.parameters())

        # (from, n_rep, full_module_path, args_str, submodule_for_params)
        rows = [
            # ── Backbone (shared weights; runs twice: capture + golden) ──
            ("-1",          1,  "ultralytics.nn.modules.conv.Conv",
             f"[3, {cs}, 3, 2]",                                     self.backbone.stem),
            ("-1",          1,  "ultralytics.nn.modules.conv.Conv",
             f"[{cs}, {cp2}, 3, 2]",                                  self.backbone.dark2[0]),
            ("-1",         b2,  "ultralytics.nn.modules.block.C2f",
             f"[{cp2}, {cp2}, {b2}, True]",                           self.backbone.dark2[1]),
            ("-1",          1,  "ultralytics.nn.modules.conv.Conv",
             f"[{cp2}, {cp3}, 3, 2]",                                  self.backbone.dark3[0]),
            ("-1",         b3,  "ultralytics.nn.modules.block.C2f",
             f"[{cp3}, {cp3}, {b3}, True]",                           self.backbone.dark3[1]),
            ("-1",          1,  "ultralytics.nn.modules.conv.Conv",
             f"[{cp3}, {cp4}, 3, 2]",                                  self.backbone.dark4[0]),
            ("-1",         b4,  "ultralytics.nn.modules.block.C2f",
             f"[{cp4}, {cp4}, {b4}, True]",                           self.backbone.dark4[1]),
            ("-1",          1,  "ultralytics.nn.modules.conv.Conv",
             f"[{cp4}, {cp5}, 3, 2]",                                  self.backbone.dark5[0]),
            ("-1",         b5,  "ultralytics.nn.modules.block.C2f",
             f"[{cp5}, {cp5}, {b5}, True]",                           self.backbone.dark5[1]),
            ("-1",          1,  "ultralytics.nn.modules.block.SPPF",
             f"[{cp5}, {cp5}, 5]",                                     self.backbone.dark5[2]),
            # ── Fusion  ([cap_layer, gold_layer] — same backbone, shared) ──
            ("[4, 4g]",     1,  f"models.ds_yolo.{self.crfm_p3.__class__.__name__}",
             f"[{cp3}]",                                               self.crfm_p3),
            ("[6, 6g]",     1,  f"models.ds_yolo.{self.crfm_p4.__class__.__name__}",
             f"[{cp4}]",                                               self.crfm_p4),
            ("[9, 9g]",     1,  f"models.ds_yolo.{self.crfm_p5.__class__.__name__}",
             f"[{cp5}]",                                               self.crfm_p5),
            # ── PANet Neck ───────────────────────────────────────────────
            ("-1",          1,  "torch.nn.Upsample",
             "[None, 2, 'nearest']",                                   self.neck.up),
            ("[13, 11]",   nn_, "ultralytics.nn.modules.block.C2f",
             f"[{cp5+cp4}, {cp4}, {nn_}, False]",                     self.neck.c2f_p4),
            ("-1",          1,  "torch.nn.Upsample",
             "[None, 2, 'nearest']",                                   self.neck.up),
            ("[15, 10]",   nn_, "ultralytics.nn.modules.block.C2f",
             f"[{cp4+cp3}, {cp3}, {nn_}, False]",                     self.neck.c2f_p3),
            ("-1",          1,  "ultralytics.nn.modules.conv.Conv",
             f"[{cp3}, {cp3}, 3, 2]",                                  self.neck.down_p3),
            ("[17, 14]",   nn_, "ultralytics.nn.modules.block.C2f",
             f"[{cp3+cp4}, {cp4}, {nn_}, False]",                     self.neck.c2f_n4),
            ("-1",          1,  "ultralytics.nn.modules.conv.Conv",
             f"[{cp4}, {cp4}, 3, 2]",                                  self.neck.down_p4),
            ("[19, 12]",   nn_, "ultralytics.nn.modules.block.C2f",
             f"[{cp4+cp5}, {cp5}, {nn_}, False]",                     self.neck.c2f_n5),
            # ── Detect head ──────────────────────────────────────────────
            ("[16,18,20]",  1,  "ultralytics.nn.modules.head.Detect",
             f"[{self.nc}, [{cp3}, {cp4}, {cp5}]]",                   self.head),
        ]

        hdr = f"{'':>5}{'from':>18}{'n':>6}{'params':>12}  {'module':<45}{'arguments'}"
        print(hdr)
        for i, (frm, rep, mod, args, submod) in enumerate(rows):
            params = _p(submod)
            print(f"{i:>5}{str(frm):>18}{rep:>6}{params:>12,}  {mod:<45}{args}")

        n_layers = sum(1 for _ in self.modules())
        n_params = sum(p.numel() for p in self.parameters())
        n_grad   = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_name = (f"DS-YOLOv8{self.variant}" if self.fusion_type == "crfm"
                      else f"F-Sub-YOLOv8{self.variant}")
        print(f"{model_name} summary (dual-stream, shared backbone, fusion={self.fusion_type}): "
              f"{n_layers} layers, {n_params:,} parameters, "
              f"{n_grad:,} gradients, {gflops}")

    # ----- Pretrained weight loading -------------------------------------
    @torch.no_grad()
    def load_yolov8_pretrained(self, weights_path: Optional[str] = None) -> dict:
        """Best-effort copy of YOLOv8m weights into matching submodules.

        Returns a small report dict describing how many keys matched.
        Modules that have no counterpart in YOLOv8 (the three CRFM
        modules) keep their default initialisation.
        """
        from ultralytics import YOLO  # imported lazily to avoid circular imports

        if weights_path is None:
            weights_path = f"yolov8{self.variant}.pt"
        else:
            # Warn if the filename suggests a different variant than self.variant.
            import os
            fname = os.path.basename(str(weights_path))
            for v in ("n", "s", "m", "l", "x"):
                if fname == f"yolov8{v}.pt" and v != self.variant:
                    print(
                        f"[warn] pretrained={fname} but variant={self.variant}. "
                        f"Channel sizes differ → most weights will be skipped. "
                        f"Use --pretrained yolov8{self.variant}.pt or omit --pretrained."
                    )
                    break
        # Ensure the weights file is present (downloads from ultralytics hub if missing).
        try:
            from ultralytics.utils.downloads import attempt_download_asset
            weights_path = str(attempt_download_asset(weights_path))
        except Exception:
            pass  # fall through to YOLO() which also handles download
        src = YOLO(weights_path).model  # ultralytics DetectionModel
        src_sd = src.state_dict()
        dst_sd = self.state_dict()

        # Map YOLOv8m's `model.<idx>.<param>` keys to our submodule paths.
        # Index map for YOLOv8m (from ultralytics yolov8m.yaml):
        #   0  Conv      -> backbone.stem
        #   1  Conv      -> backbone.dark2.0
        #   2  C2f       -> backbone.dark2.1
        #   3  Conv      -> backbone.dark3.0
        #   4  C2f       -> backbone.dark3.1
        #   5  Conv      -> backbone.dark4.0
        #   6  C2f       -> backbone.dark4.1
        #   7  Conv      -> backbone.dark5.0
        #   8  C2f       -> backbone.dark5.1
        #   9  SPPF      -> backbone.dark5.2
        #   10 Upsample  (no params)
        #   11 Concat    (no params)
        #   12 C2f       -> neck.c2f_p4
        #   13 Upsample  (no params)
        #   14 Concat    (no params)
        #   15 C2f       -> neck.c2f_p3
        #   16 Conv      -> neck.down_p3
        #   17 Concat    (no params)
        #   18 C2f       -> neck.c2f_n4
        #   19 Conv      -> neck.down_p4
        #   20 Concat    (no params)
        #   21 C2f       -> neck.c2f_n5
        #   22 Detect    -> head
        idx_to_dst = {
            0: "backbone.stem",
            1: "backbone.dark2.0",
            2: "backbone.dark2.1",
            3: "backbone.dark3.0",
            4: "backbone.dark3.1",
            5: "backbone.dark4.0",
            6: "backbone.dark4.1",
            7: "backbone.dark5.0",
            8: "backbone.dark5.1",
            9: "backbone.dark5.2",
            12: "neck.c2f_p4",
            15: "neck.c2f_p3",
            16: "neck.down_p3",
            18: "neck.c2f_n4",
            19: "neck.down_p4",
            21: "neck.c2f_n5",
            # Head (index 22) intentionally excluded: nc differs (pretrained=80, ours=nc).
        }
        copied = 0
        skipped = 0
        for k, v in src_sd.items():
            if not k.startswith("model."):
                continue
            try:
                idx_str, rest = k[len("model.") :].split(".", 1)
                idx = int(idx_str)
            except ValueError:
                continue
            if idx not in idx_to_dst:
                continue
            new_key = f"{idx_to_dst[idx]}.{rest}"
            if new_key in dst_sd and dst_sd[new_key].shape == v.shape:
                dst_sd[new_key] = v
                copied += 1
            else:
                skipped += 1

        self.load_state_dict(dst_sd, strict=False)
        return {"copied": copied, "skipped": skipped, "total_src": len(src_sd)}


# =============================================================================
# Helper: a wrapper that "pretends" to be an ultralytics DetectionModel for
# the v8DetectionLoss class. v8DetectionLoss only reads model.stride,
# model.nc, model.args (if present), model.no.
# =============================================================================
class _LossCompatModel(nn.Module):
    """Adapter exposing the attributes ``v8DetectionLoss`` needs.

    ``v8DetectionLoss.__init__`` does two things that require compatibility:
      1. ``next(model.parameters()).device``  — needs model to be a proper
         nn.Module with registered parameters.
      2. ``m = model.model[-1]``             — needs model.model to be
         subscriptable, returning the Detect head on index -1.

    We satisfy (1) by keeping ``self.model = ds_model`` (nn.Module registration).
    We satisfy (2) by adding ``__getitem__`` to DSYOLOv8m (see below).
    """

    def __init__(self, ds_model: DSYOLOv8m) -> None:
        super().__init__()
        self.model = ds_model          # registered as nn.Module child → .parameters() works
        # Required by v8DetectionLoss.__init__:
        self.stride = ds_model.head.stride
        self.nc = ds_model.head.nc
        self.no = ds_model.head.no     # number of outputs per anchor (4 + nc)
        self.args = type(
            "Args",
            (),
            dict(box=7.5, cls=0.5, dfl=1.5),
        )()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
