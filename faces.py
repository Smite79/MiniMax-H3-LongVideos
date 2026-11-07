# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import contextlib
import importlib
import importlib.util
import io
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
AUX_SRC = os.path.normpath(os.path.join(_HERE, "..", "comfyui_controlnet_aux", "src"))
AUX_CKPTS = os.path.normpath(os.path.join(_HERE, "..", "comfyui_controlnet_aux", "ckpts"))
DET = ("hr16/yolox-onnx", "yolox_l.torchscript.pt")
POSE = ("hr16/DWPose-TorchScript-BatchSize5", "dw-ll_ucoco_384_bs5.torchscript.pt")
CONF = 0.3
NOSE, NECK, RSHO, LSHO, REYE, LEYE, REAR, LEAR = 0, 1, 2, 5, 14, 15, 16, 17
CROP_HALF, CROP_UP = 0.9, 0.25


def _aux():
    try:
        if importlib.util.find_spec("custom_controlnet_aux") is None and os.path.isdir(AUX_SRC):
            sys.path.append(AUX_SRC)
            importlib.invalidate_caches()
        return importlib.util.find_spec("custom_controlnet_aux") is not None
    except (ImportError, ValueError):
        return False


def paths():
    root = os.environ.get("AUX_ANNOTATOR_CKPTS_PATH") or AUX_CKPTS
    try:
        root = str(importlib.import_module("custom_controlnet_aux.util").annotator_ckpts_path or root)
    except Exception:
        pass
    return tuple(os.path.join(root, repo, fn) for repo, fn in (DET, POSE))


def available():
    return _aux() and all(os.path.isfile(p) for p in paths())


@contextlib.contextmanager
def _offline():
    old = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as hub
        was, hub.HF_HUB_OFFLINE = hub.HF_HUB_OFFLINE, True
    except Exception:
        hub = None
    try:
        yield
    finally:
        os.environ.pop("HF_HUB_OFFLINE") if old is None else os.environ.update(HF_HUB_OFFLINE=old)
        if hub is not None:
            hub.HF_HUB_OFFLINE = was


class Detector:

    def __init__(self):
        self.est = None

    def people(self, frame):
        import comfy.model_management as mm
        if self.est is None:
            from custom_controlnet_aux.dwpose import DwposeDetector
            with _offline():
                det = DwposeDetector.from_pretrained(POSE[0], DET[0], det_filename=DET[1], pose_filename=POSE[1],
                                                     torchscript_device=mm.get_torch_device())
            self.est = det.dw_pose_estimation
        for m in (self.est.det, self.est.pose):
            m.to(mm.get_torch_device())
        img = np.ascontiguousarray(frame[..., :3].detach().float().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy())
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            found = self.est(img)
        return [] if found is None else [np.asarray(p[:18], dtype=np.float64) for p in np.asarray(found)]

    def close(self):
        if self.est is not None:
            for m in (self.est.det, self.est.pose):
                m.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _seen(kp, i):
    return kp.shape[0] > i and kp[i, 2] >= CONF


def head_box(kp, width, height):
    kp = np.asarray(kp, dtype=np.float64)
    if kp.ndim != 2 or kp.shape[1] < 3:
        return None
    head = [kp[i, :2] for i in (NOSE, REYE, LEYE, REAR, LEAR) if _seen(kp, i)]
    neck = kp[NECK, :2] if _seen(kp, NECK) else (kp[RSHO, :2] + kp[LSHO, :2]) / 2 if _seen(kp, RSHO) and _seen(kp, LSHO) else None
    if not head or neck is None:
        return None
    c = np.mean(head, axis=0)
    s = float(np.linalg.norm(c - neck))
    if s <= 1.0:
        return None
    m, half = c + CROP_UP * (c - neck), CROP_HALF * s
    x0, y0 = max(0, int(round(m[0] - half))), max(0, int(round(m[1] - half)))
    x1, y1 = min(int(width), int(round(m[0] + half))), min(int(height), int(round(m[1] + half)))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def crop(detector, picture):
    h, w = int(picture.shape[1]), int(picture.shape[2])
    boxes = [b for b in (head_box(p, w, h) for p in detector.people(picture[0])) if b]
    if not boxes:
        return None
    x0, y0, x1, y1 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return picture[:1, y0:y1, x0:x1, :].contiguous()


def count(detector, frame):
    return sum(1 for p in detector.people(frame[0]) if _seen(p, NOSE) and (_seen(p, REYE) or _seen(p, LEYE)))
