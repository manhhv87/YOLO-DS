"""Patch a YOLOv8 model so it accepts ``in_channels`` input channels.

Strategy: replace the very first ``Conv2d`` of the backbone (default
``3 -> 64``) with a new ``in_channels -> 64`` convolution. The first three
input channels are initialised from the pretrained RGB weights; any extra
channel is initialised with the channel-mean of the pretrained weights, so
the model behaves close to the RGB baseline on the first iteration and is
fine-tuned afterwards.

Usage::

    from ultralytics import YOLO
    from model_4ch import patch_first_conv

    yolo = YOLO("yolov8m.pt")
    patch_first_conv(yolo.model, in_channels=4)
    yolo.train(data="rg-yolo.yaml", imgsz=1024, epochs=60, batch=4)
"""
from __future__ import annotations

import torch
from torch import nn


def patch_first_conv(model: nn.Module, in_channels: int) -> None:
    """In-place replace the first Conv2d of ``model`` to take ``in_channels``."""
    first_name, first_module = _find_first_conv(model)
    if first_module.in_channels == in_channels:
        return  # already correctly sized

    old_w = first_module.weight.data                # (out, 3, k, k)
    out_c, _, kh, kw = old_w.shape
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_c,
        kernel_size=(kh, kw),
        stride=first_module.stride,
        padding=first_module.padding,
        bias=first_module.bias is not None,
    )

    with torch.no_grad():
        new_w = torch.empty(out_c, in_channels, kh, kw, dtype=old_w.dtype, device=old_w.device)
        # Copy the first 3 RGB channels.
        copy_c = min(3, in_channels)
        new_w[:, :copy_c] = old_w[:, :copy_c]
        # Initialise extra channels to the per-output-channel mean of the RGB weights.
        if in_channels > 3:
            mean_w = old_w.mean(dim=1, keepdim=True)
            new_w[:, 3:] = mean_w.expand(-1, in_channels - 3, -1, -1)
        new_conv.weight.copy_(new_w)
        if first_module.bias is not None:
            new_conv.bias.copy_(first_module.bias.data)

    _set_module(model, first_name, new_conv)


def _find_first_conv(model: nn.Module) -> tuple[str, nn.Conv2d]:
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            return name, m
    raise RuntimeError("No Conv2d found in model.")


def _set_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parent = root
    parts = dotted_name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)
