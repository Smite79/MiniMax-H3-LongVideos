# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import os

import torch
import nodes
import comfy.utils
from h3_runtime import ensure_host_ram, _deep_cleanup

RESIZE_CHUNK = 32
UPSCALE_BATCH = 4
FRAME_MODES = ["off", "rtx", "model", "lanczos"]


def find_node(node_id):
    return (getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}).get(node_id)


def run_node(cls, **kwargs):
    out = cls.execute(**kwargs) if hasattr(cls, "define_schema") else getattr(cls(), cls.FUNCTION)(**kwargs)
    out = getattr(out, "result", out)
    return out[0] if isinstance(out, (tuple, list)) else out


def latent_models():
    try:
        import folder_paths
        d = os.path.join(folder_paths.models_dir, "latent_upscale_models")
        names = [f for f in sorted(os.listdir(d)) if f.lower().endswith((".pth", ".safetensors"))
                 and ("minimax" in f.lower() or "h3" in f.lower())]
    except Exception:
        names = []
    return ["off"] + names


def frame_models():
    try:
        import folder_paths
        return ["none"] + list(folder_paths.get_filename_list("upscale_models"))
    except Exception:
        return ["none"]


def upscale_latent(video, model_name, scale):
    if model_name in (None, "", "off") or float(scale) <= 1.0:
        return video, ""
    cls = find_node("MinimaxH3LatentUpscaler3D")
    if cls is None:
        return video, "latent upscale needs the Minimax H3 Latent Upscaler node pack"
    cuda = torch.cuda.is_available()
    try:
        up = run_node(cls, latent={"samples": video}, model_name=model_name,
                      mode={"mode": "scale by multiplier", "scale": float(scale)}, align=32, enable_chunking=True,
                      device="cuda" if cuda else "cpu", precision="fp16" if cuda else "fp32")
        up = up["samples"] if isinstance(up, dict) else up
    except Exception as e:
        return video, f"latent upscale failed ({type(e).__name__}: {e})"
    if not torch.is_tensor(up) or up.dim() != video.dim() or up.shape[2] != video.shape[2]:
        return video, "latent upscale returned an unexpected shape"
    return up.to(video.dtype), ""


def fit(frames, width, height, method="lanczos"):
    b, h, w, c = frames.shape
    if (w, h) == (width, height):
        return frames
    if frames.device.type == "cpu":
        ensure_host_ram(b * height * width * c * frames.element_size(), what="the resized video")
    out = torch.empty((b, height, width, c), dtype=frames.dtype, device=frames.device)
    for i in range(0, b, RESIZE_CHUNK):
        part = comfy.utils.common_upscale(frames[i:i + RESIZE_CHUNK].movedim(-1, 1), width, height, method, "disabled")
        out[i:i + RESIZE_CHUNK].copy_(part.movedim(1, -1))
    return out


def fit_short_edge(frames, target):
    h, w = int(frames.shape[1]), int(frames.shape[2])
    if h <= w:
        return fit(frames, max(32, int(round(target * w / h / 32) * 32)), int(target))
    return fit(frames, int(target), max(32, int(round(target * h / w / 32) * 32)))


def _chunked(frames, fn):
    out = None
    for i in range(0, int(frames.shape[0]), UPSCALE_BATCH):
        part = fn(frames[i:i + UPSCALE_BATCH]).detach().to("cpu", dtype=frames.dtype)
        if out is None:
            ensure_host_ram(int(frames.shape[0]) * (part.nbytes // max(1, int(part.shape[0]))), what="the upscaled video")
            out = torch.empty((int(frames.shape[0]),) + tuple(part.shape[1:]), dtype=part.dtype)
        out[i:i + int(part.shape[0])].copy_(part)
        del part
        _deep_cleanup()
    return out


def upscale_frames(frames, mode, model_name, target):
    if mode not in ("rtx", "model", "lanczos") or frames is None or not int(frames.shape[0]):
        return frames, ""
    note = ""
    try:
        if mode == "rtx":
            cls = find_node("RTXVideoSuperResolution")
            if cls is None:
                raise RuntimeError("the RTX Video Super Resolution node is not installed")
            short = min(int(frames.shape[1]), int(frames.shape[2]))
            scale = max(1, min(4, round(int(target) / short))) if target else 2
            frames = _chunked(frames, lambda p: run_node(cls, images=p, quality="ULTRA",
                                                         resize_type={"resize_type": "scale by multiplier",
                                                                      "scale": float(scale)}))
            note = f"RTX Video Super Resolution x{scale}"
        elif mode == "model":
            loader, apply = find_node("UpscaleModelLoader"), find_node("ImageUpscaleWithModel")
            if loader is None or apply is None or model_name in (None, "", "none"):
                raise RuntimeError("no upscale model chosen")
            model = run_node(loader, model_name=model_name)
            frames = _chunked(frames, lambda p: run_node(apply, upscale_model=model, image=p))
            note = f"upscaled with {model_name}"
    except Exception as e:
        note = f"{mode} upscale failed ({type(e).__name__}: {e})"
    if target:
        frames = fit_short_edge(frames, int(target))
        note = (note + "; " if note else "") + f"short edge {int(target)}px"
    elif mode == "lanczos":
        note = "lanczos upscale needs upscale_target_short_edge"
    return frames, note
