# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import bisect
import contextlib
import importlib
import importlib.util
import io
import logging
import math
import os
import sys
from collections import namedtuple
from itertools import permutations

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

AUX_SRC = os.path.normpath(os.path.join(_HERE, "..", "comfyui_controlnet_aux", "src"))
AUX_CKPTS_DEFAULT = os.path.normpath(os.path.join(_HERE, "..", "comfyui_controlnet_aux", "ckpts"))
DWPOSE_DET = ("hr16/yolox-onnx", "yolox_l.torchscript.pt")
DWPOSE_POSE = ("hr16/DWPose-TorchScript-BatchSize5", "dw-ll_ucoco_384_bs5.torchscript.pt")

NOSE, NECK = 0, 1
RSHO, RELB, RWRI = 2, 3, 4
LSHO, LELB, LWRI = 5, 6, 7
RHIP, RKNE, RANK = 8, 9, 10
LHIP, LKNE, LANK = 11, 12, 13
REYE, LEYE, REAR, LEAR = 14, 15, 16, 17
TORSO_POINTS = (NECK, RSHO, LSHO, RHIP, LHIP)
UPPER_POINTS = (NOSE, NECK, RSHO, LSHO, RELB, LELB, RWRI, LWRI, REYE, LEYE, REAR, LEAR)

ARMS_POSITIONS = ("behind the back", "in front of the body", "at the waist", "above the head")
LEGS_POSITIONS = ("ankles together", "held apart", "ankles to the wrists")
MODES = ("repair", "every", "falls")
DRAWS = ("everyone", "everyone, thick lines", "bound person only")

POSE_CONF = 0.3

POSE_ID_FRAMES = 6
POSE_CARRY_IOU_MIN = 0.5
POSE_CARRY_IOU_LEAD = 0.25
POSE_FIT_MAX = 0.35
POSE_FIT_LEAD = 0.15
POSE_CAST_TOLERANCE = 1
POSE_ID_MAX_PEOPLE = 8
POSE_CONTACT_GAP = 0.5

POSE_TRACK_HOLD = 6
POSE_TRACK_MISS_MAX = 0.25
POSE_TRACK_GATE = 1.0
POSE_TRACK_GATE_GROW = 0.15
POSE_TRACK_STALE_COST = 0.15
POSE_TRACK_NEW_COST = 2.0
POSE_TRACK_POSE_W = 0.3
POSE_TRACK_SCALE_W = 0.5
POSE_TRACK_VEL_POINTS = 3
POSE_TRACK_T_SMOOTH = 0.5
POSE_TRACK_EXACT_MAX = 12
POSE_SIG_W = 1.0
POSE_SIG_FRAMES = 6
POSE_SIG_SMOOTH = 0.3
POSE_SIG_TOL = 0.12
POSE_CROSS_MARGIN = 0.3
POSE_REACQ_RADIUS = 0.5
POSE_REACQ_GROW = 0.15
POSE_GAP_MAX = 3
POSE_CONTACT_FILL_MAX = 1
POSE_BOX_PAD = 0.15
POSE_CROSS_SIG_MARGIN = 0.06

POSE_APP_HUE_BINS = 12
POSE_APP_SAT_BINS = 2
POSE_APP_VAL_BINS = 3
POSE_APP_SAT_MIN = 0.25
POSE_APP_VAL_MIN = 0.15
POSE_APP_REGIONS = ("torso", "head", "legs")
POSE_APP_WEIGHTS = (0.5, 0.25, 0.25)
POSE_APP_MIN_SHARED = 0.5
POSE_APP_MIN_PIXELS = 24
POSE_APP_MAX_PIXELS = 4096
POSE_APP_TORSO_A = (0.15, 0.85)
POSE_APP_TORSO_SHRINK = 0.6
POSE_APP_TORSO_MIN = 0.10
POSE_APP_TORSO_MAX = 0.30
POSE_APP_HEAD_R = 0.15
POSE_APP_HEAD_UP = 0.10
POSE_APP_LEG_A = (0.15, 0.70)
POSE_APP_LEG_HALF = 0.07
POSE_APP_MATCH_MAX = 0.35
POSE_APP_DISTINCT_MIN = 0.35
POSE_APP_MARGIN = 0.15
POSE_APP_FRAMES = 6
POSE_APP_CLEAN_PAD = 0.25
POSE_APP_W = 1.0

POSE_T_MEDIAN = 3
POSE_SMOOTH = 2
POSE_VIEW_R0 = 0.20
POSE_VIEW_SPAN = 0.45
POSE_FACING_FRAMES = 12
POSE_FORWARD_DEADBAND = 0.02
POSE_HALF_SHOULDER = 0.39
POSE_PROFILE_WRISTS_W = 0.5
POSE_FACING_NEED_W = 0.7

ARM_FRONT = {
    "behind the back":      {"elbow": (0.55, 1.05), "wrist": (0.90, 0.06)},
    "in front of the body": {"elbow": (0.55, 0.95), "wrist": (0.95, 0.10)},
    "at the waist":         {"elbow": (0.50, 1.05), "wrist": (0.85, 0.55)},
    "above the head":       {"elbow": (-0.45, 1.10), "wrist": (-0.95, 0.08)},
}
ARM_PROFILE = {
    "behind the back":      {"elbow": (0.55, 0.28), "wrist": (0.92, 0.22)},
    "in front of the body": {"elbow": (0.55, 0.00), "wrist": (0.95, -0.30)},
    "at the waist":         {"elbow": (0.50, 0.15), "wrist": (0.85, -0.15)},
    "above the head":       {"elbow": (-0.45, -0.10), "wrist": (-0.95, -0.05)},
}
POSE_BLEND_FIT = 0.35
POSE_BLEND_FRAMES = 8

POSE_FIT_VIOLATION = 1.0
POSE_FIT_SQUARE_W = 0.9
POSE_FIT_PROFILE_W = 0.25
POSE_FIT_TOGETHER = 0.35
POSE_FIT_BAND = 1.0
POSE_FIT_WAIST_ELBOW = 1.35
POSE_FIT_WAIST_APART = 0.6
POSE_FIT_PROFILE_SIDE = 0.05

POSE_LATCH_RUN = 3
POSE_LATCH_LEG_FIT = 0.25
POSE_LATCH_PRESET_SHARE = 0.8
POSE_LATCH_NEAR_GAP = 1.0

POSE_ANKLE_GAP = 0.12
POSE_APART_MIN = 0.8
POSE_APART_MAX = 1.2
POSE_HOGTIE_BACK = 0.15
POSE_LIMB_DEFAULT = 0.85

POSE_BREAK_WRIST = 0.6
POSE_BREAK_RAISED_A = 0.45
POSE_BREAK_SIDE = 1.5
POSE_BREAK_SIDE_MIN_W = 0.5
POSE_BREAK_SIDE_NOISE = 0.1
POSE_BREAK_SIDE_RUN = 2
POSE_BREAK_SIDE_CLEAR = 0.25
POSE_BREAK_ELBOW = 0.45
POSE_BREAK_WRISTS_APART = 0.5
POSE_BREAK_ANKLES_EXTRA = 0.25
POSE_BREAK_RUN = 3
POSE_BREAK_SHARE = 0.08

POSE_FALL_SETTLE = True
POSE_FALL_WRIST_FLOOR = 0.35
POSE_FALL_WRIST_BELOW = 0.6
POSE_FALL_TILT_DEG = 45.0
POSE_FALL_NECK_ABOVE = 0.15
POSE_FALL_RAMP = 6
POSE_FALL_LEAVE_RUN = 3

POSE_CAPTOR_NEAR = 0.3
POSE_CAPTOR_MOVED = 0.4

POSE_HINT_DTYPE = torch.float32
POSE_SIGMA_START_PAD = 1e-3
POSE_END_DEFAULT = 0.6

_KP = namedtuple("_KP", "x y score id")
_DRAW_FN = None
_PRESET_CLS = None


def _person(a):
    if a is None:
        return None
    try:
        if torch.is_tensor(a):
            a = a.detach().to("cpu", torch.float64).numpy()
        a = np.asarray(a, dtype=np.float64)
    except Exception:
        return None
    if a.ndim != 2 or a.shape[0] < 1:
        return None
    if a.shape[1] == 2:
        a = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    out = np.zeros((18, 3), dtype=np.float64)
    n = min(18, a.shape[0])
    out[:n] = a[:n, :3]
    out[~np.isfinite(out).all(axis=1)] = 0.0
    return out


def _seq(x):
    if x is None:
        return []
    if torch.is_tensor(x):
        x = x.detach().cpu()
        return list(x.unbind(0)) if x.dim() else []
    if isinstance(x, np.ndarray):
        return list(x) if x.ndim else []
    try:
        return list(x)
    except TypeError:
        return []


def _vis(kp, i):
    return kp is not None and kp[i, 2] >= POSE_CONF


def _xy(kp, i):
    return np.array(kp[i, :2], dtype=np.float64)


def _rot(u):
    return np.array([-u[1], u[0]], dtype=np.float64)


def _neck(kp):
    if _vis(kp, NECK):
        return _xy(kp, NECK)
    if _vis(kp, RSHO) and _vis(kp, LSHO):
        return (_xy(kp, RSHO) + _xy(kp, LSHO)) / 2.0
    return None


def _midhip(kp):
    r, l = _vis(kp, RHIP), _vis(kp, LHIP)
    if r and l:
        return (_xy(kp, RHIP) + _xy(kp, LHIP)) / 2.0
    if r:
        return _xy(kp, RHIP)
    if l:
        return _xy(kp, LHIP)
    return None


def _torso_len(kp):
    n, h = _neck(kp), _midhip(kp)
    if n is not None and h is not None:
        d = float(np.linalg.norm(h - n))
        if d > 1.0:
            return d
    pts = kp[kp[:, 2] >= POSE_CONF, :2] if kp is not None else np.zeros((0, 2))
    if len(pts) >= 2:
        span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
        if span > 1.0:
            return span / 3.0
    return None


def _torso_box(kp):
    if kp is None:
        return None
    pts = [kp[i, :2] for i in TORSO_POINTS if kp[i, 2] >= POSE_CONF]
    if len(pts) < 2:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    pad = POSE_BOX_PAD * max(x1 - x0, y1 - y0, 1.0)
    return [float(x0 - pad), float(y0 - pad), float(x1 + pad), float(y1 + pad)]


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _window_avg(arr, k):
    n = len(arr)
    out = np.empty_like(arr)
    for i in range(n):
        out[i] = arr[max(0, i - k):i + k + 1].mean(axis=0)
    return out


def _window_median(arr, k):
    n = len(arr)
    return np.array([float(np.median(arr[max(0, i - k):i + k + 1])) for i in range(n)])


def _runs(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _spans(frames):
    frames = sorted(set(int(f) for f in frames))
    if not frames:
        return ""
    out, start, prev = [], frames[0], frames[0]
    for f in frames[1:] + [None]:
        if f is not None and f <= prev + 2:
            prev = f
            continue
        out.append(f"{start}-{prev}" if prev != start else f"{start}")
        if f is not None:
            start = prev = f
    return ", ".join(out)


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _free_cuda():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class _Geom:
    __slots__ = ("N", "u", "p", "T", "sR", "sL", "w", "front", "back", "idx", "forward_known",
                 "facing_unknown")


def _geometry(kps, idx):
    n = len(kps)
    if n == 0:
        return None
    Ns = [_neck(k) for k in kps]
    Hs = [_midhip(k) for k in kps]
    good = [i for i in range(n) if Ns[i] is not None and Hs[i] is not None
            and np.linalg.norm(Hs[i] - Ns[i]) > 1.0]
    if not good:
        return None
    goodset = set(good)
    N = np.zeros((n, 2))
    V = np.zeros((n, 2))
    last = None
    for i in range(n):
        if i in goodset:
            N[i], V[i], last = Ns[i], Hs[i] - Ns[i], i
            continue
        ref = last if last is not None else next(g for g in good if g > i)
        vref = Hs[ref] - Ns[ref]
        if Ns[i] is not None:
            N[i], V[i] = Ns[i], vref
        elif Hs[i] is not None:
            N[i], V[i] = Hs[i] - vref, vref
        else:
            N[i], V[i] = Ns[ref], vref
    T_raw = np.linalg.norm(V, axis=1)
    u_raw = V / T_raw[:, None]
    p_raw = np.array([_rot(u) for u in u_raw])

    sR = np.full(n, np.nan)
    sL = np.full(n, np.nan)
    for i, kp in enumerate(kps):
        r = float((_xy(kp, RSHO) - N[i]) @ p_raw[i]) if _vis(kp, RSHO) else None
        l = float((_xy(kp, LSHO) - N[i]) @ p_raw[i]) if _vis(kp, LSHO) else None
        if r is None and l is not None:
            r = -l
        if l is None and r is not None:
            l = -r
        if r is not None:
            sR[i], sL[i] = r / T_raw[i], l / T_raw[i]
    known = np.where(np.isfinite(sR))[0]
    for i in range(n):
        if not np.isfinite(sR[i]):
            if len(known):
                j = known[np.argmin(np.abs(known - i))]
                sR[i], sL[i] = sR[j], sL[j]
            else:
                sR[i], sL[i] = POSE_HALF_SHOULDER, -POSE_HALF_SHOULDER
    sR *= T_raw
    sL *= T_raw
    r = np.abs(sR - sL) / T_raw
    w = np.clip((r - POSE_VIEW_R0) / POSE_VIEW_SPAN, 0.0, 1.0)

    g = _Geom()
    g.idx = [int(x) for x in idx]
    g.T = _window_median(T_raw, POSE_T_MEDIAN)
    g.N = _window_avg(N, POSE_SMOOTH)
    u = _window_avg(u_raw, POSE_SMOOTH)
    un = np.linalg.norm(u, axis=1)
    bad = un < 1e-6
    u[bad] = u_raw[bad]
    un[bad] = 1.0
    g.u = u / un[:, None]
    g.p = np.array([_rot(x) for x in g.u])
    g.w = _window_avg(w, POSE_SMOOTH)
    g.sR = _window_avg(sR, POSE_SMOOTH)
    g.sL = _window_avg(sL, POSE_SMOOTH)

    face = np.array([any(_vis(k, j) for j in (NOSE, REYE, LEYE)) for k in kps])
    fwd = np.zeros(n)
    for i, kp in enumerate(kps):
        pts = [j for j in (NOSE,) if _vis(kp, j)] or [j for j in (REYE, LEYE) if _vis(kp, j)] \
            or [j for j in (REAR, LEAR) if _vis(kp, j)]
        if pts:
            off = float(np.mean([(_xy(kp, j) - g.N[i]) @ g.p[i] for j in pts]))
            if abs(off) > POSE_FORWARD_DEADBAND * g.T[i]:
                fwd[i] = math.copysign(1.0, off)
    g.front = np.zeros(n, dtype=bool)
    g.back = np.zeros(n)
    g.facing_unknown = np.zeros(n, dtype=bool)
    iarr = np.asarray(g.idx)
    knownf = np.where(fwd != 0)[0]
    g.forward_known = bool(len(knownf))
    for i in range(n):
        win = np.abs(iarr - iarr[i]) <= POSE_FACING_FRAMES
        fronts = int(face[win].sum())
        backs = int(win.sum()) - fronts
        g.front[i] = face[i] if fronts == backs else fronts > backs
        votes = fwd[win]
        s = float(votes.sum())
        if s != 0:
            f = math.copysign(1.0, s)
        elif fwd[i] != 0:
            f = fwd[i]
        elif len(knownf):
            j = knownf[np.argmin(np.abs(knownf - i))]
            near = knownf[np.abs(iarr[knownf] - iarr[j]) <= POSE_FACING_FRAMES]
            s2 = float(fwd[near].sum())
            f = math.copysign(1.0, s2) if s2 != 0 else fwd[j]
        else:
            f = 1.0
            g.facing_unknown[i] = True
        g.back[i] = -f
    return g


def _point(g, i, a, b_pix):
    return g.N[i] + a * g.T[i] * g.u[i] + b_pix * g.p[i]


def _arm_points(g, i, arms):
    w, T, back = float(g.w[i]), float(g.T[i]), float(g.back[i])
    out = {}
    for part, joints in (("elbow", (RELB, LELB)), ("wrist", (RWRI, LWRI))):
        af, bf = ARM_FRONT[arms][part]
        ap, bp = ARM_PROFILE[arms][part]
        a = w * af + (1.0 - w) * ap
        for j, s in zip(joints, (g.sR[i], g.sL[i])):
            b = w * bf * float(s) + (1.0 - w) * bp * T * back
            out[j] = _point(g, i, a, b)
    drawn = True
    if arms == "behind the back":
        drawn = (not bool(g.front[i])) or w < POSE_PROFILE_WRISTS_W
    return out, drawn


def _hogtie_ankle(g, i):
    ap, bp = ARM_PROFILE["behind the back"]["wrist"]
    T, back = float(g.T[i]), float(g.back[i])
    return _point(g, i, ap, (bp + POSE_HOGTIE_BACK) * T * back)


def _spec(raw):
    raw = raw or {}
    arms = str(raw.get("arms") or "").strip()
    legs = str(raw.get("legs") or "").strip()
    arms = arms if arms in ARMS_POSITIONS else ""
    legs = legs if legs in LEGS_POSITIONS else ""
    if legs == "ankles to the wrists" and not arms:
        arms = "behind the back"
    try:
        gap = float(raw.get("ankle_gap") or POSE_ANKLE_GAP)
    except (TypeError, ValueError):
        gap = POSE_ANKLE_GAP
    want = raw.get("latch_limbs") or ()
    if isinstance(want, str):
        want = (want,)
    try:
        want = {str(x).strip() for x in want}
    except TypeError:
        want = set()
    latch = tuple(x for x, held in (("arms", arms), ("legs", legs)) if held and x in want)
    return {"arms": arms, "legs": legs, "ankle_gap": max(0.0, gap),
            "anchored": bool(raw.get("anchored")), "fall": bool(raw.get("fall")),
            "latch_limbs": latch}


def _arm_rule_broken(kp, g, i, arms):
    w, T, N, p = float(g.w[i]), float(g.T[i]), g.N[i], g.p[i]
    wr = [j for j in (RWRI, LWRI) if _vis(kp, j)]
    if w >= POSE_FIT_SQUARE_W:
        s = max(abs(float(g.sR[i])), abs(float(g.sL[i])), 1e-6)

        def off(j):
            return abs(float((_xy(kp, j) - N) @ p))
        apart = (float(np.linalg.norm(_xy(kp, RWRI) - _xy(kp, LWRI))) / T) if len(wr) == 2 else 0.0
        if arms in ("behind the back", "in front of the body", "above the head"):
            if apart > POSE_FIT_TOGETHER:
                return True
            if any(off(j) > POSE_FIT_BAND * s for j in wr):
                return True
        elif arms == "at the waist":
            if any(off(j) > POSE_FIT_WAIST_ELBOW * s for j in (RELB, LELB) if _vis(kp, j)):
                return True
            if apart > POSE_FIT_WAIST_APART:
                return True
    elif w <= POSE_FIT_PROFILE_W and arms == "behind the back" and not g.facing_unknown[i]:
        back = float(g.back[i])
        if any(float((_xy(kp, j) - N) @ p) * back / T < POSE_FIT_PROFILE_SIDE for j in wr):
            return True
    return False


def _frame_fit(kp, g, i, spec):
    T = float(g.T[i])
    if spec["arms"]:
        pts, _ = _arm_points(g, i, spec["arms"])
        ds, seen = [], 0
        for j in (RELB, LELB):
            if _vis(kp, j):
                ds.append(float(np.linalg.norm(_xy(kp, j) - pts[j])) / T)
                seen += 1
        for j in (RWRI, LWRI):
            if _vis(kp, j):
                ds.append(float(np.linalg.norm(_xy(kp, j) - pts[j])) / T)
                seen += 1
            else:
                ds.append(0.0)
        if not seen:
            return None
        if _arm_rule_broken(kp, g, i, spec["arms"]):
            return POSE_FIT_VIOLATION
        return float(np.mean(ds))
    if not (_vis(kp, RANK) and _vis(kp, LANK)):
        return None
    sep = float(np.linalg.norm(_xy(kp, RANK) - _xy(kp, LANK))) / T
    if spec["legs"] == "held apart":
        return max(0.0, POSE_APART_MIN - sep)
    return max(0.0, sep - spec["ankle_gap"])


def _facing_needed(g, i, spec):
    return bool(spec["arms"]) and bool(g.facing_unknown[i]) and float(g.w[i]) < POSE_FACING_NEED_W


def _limb_spec(spec, limb):
    return {**spec, "legs": ""} if limb == "arms" else {**spec, "arms": ""}


def _latch_fits(kp, g, i, spec):
    for limb in spec["latch_limbs"]:
        sub = _limb_spec(spec, limb)
        if limb == "arms" and _facing_needed(g, i, sub):
            return False
        f = _frame_fit(kp, g, i, sub)
        if f is None or f > (POSE_FIT_MAX if limb == "arms" else POSE_LATCH_LEG_FIT):
            return False
    return True


def _latch_start(kps, idx, ps, g, spec, after):
    if not spec["latch_limbs"] or g is None:
        return None
    fits = [_latch_fits(k, g, i, spec) for i, k in enumerate(kps)]
    for i in range(len(kps)):
        if idx[i] < after:
            continue
        run = range(i, i + POSE_LATCH_RUN)
        if run[-1] < len(kps) and all(fits[j] for j in run) \
                and ps[run[-1]] - ps[i] == POSE_LATCH_RUN - 1:
            return i
    return None


def _posed_before(kps, idx, g, spec, after):
    if not spec["latch_limbs"] or g is None:
        return False
    early = [i for i in range(len(kps)) if idx[i] < after]
    if len(early) < POSE_LATCH_RUN:
        return False
    fitting = sum(1 for i in early if _latch_fits(kps[i], g, i, spec))
    return fitting >= POSE_LATCH_PRESET_SHARE * len(early)


def _touched_before(ti, upto, tracks, people):
    me = tracks[ti]["pos"]
    for tj, t in enumerate(tracks):
        if tj == ti:
            continue
        for p, pi in t["pos"].items():
            if p < upto and p in me and _in_contact(people[p][me[p]], people[p][pi],
                                                    gap=POSE_LATCH_NEAR_GAP):
                return True
    return False


def _held_from(latch, limb):
    if not latch or limb not in latch:
        return 0
    return latch[limb]


def _is_held(latch, limb, t):
    start = _held_from(latch, limb)
    return start is not None and t >= start


def _track_fit_why(kps, idx, spec):
    g = _geometry(kps, idx)
    if g is None:
        return None, "torso not fully visible"
    fits, blind = [], 0
    for i, k in enumerate(kps):
        if _facing_needed(g, i, spec):
            blind += 1
            continue
        f = _frame_fit(k, g, i, spec)
        if f is not None:
            fits.append(f)
    if fits:
        return float(np.median(fits)), ""
    return None, ("could not tell which way they face" if blind else "")


def _frame_breaks(kp, g, i, spec):
    why = set()
    side_clear = False
    T, N, u, p, w = float(g.T[i]), g.N[i], g.u[i], g.p[i], float(g.w[i])
    arms = spec["arms"]
    if arms:
        pts, _ = _arm_points(g, i, arms)
        for wj, ej, s in ((RWRI, RELB, g.sR[i]), (LWRI, LELB, g.sL[i])):
            if _vis(kp, wj):
                W = _xy(kp, wj)
                if np.linalg.norm(W - pts[wj]) > POSE_BREAK_WRIST * T:
                    why.add("hands away from the restraint")
                if arms != "above the head" and float((W - N) @ u) / T < POSE_BREAK_RAISED_A:
                    why.add("hands raised")
                if w >= POSE_BREAK_SIDE_MIN_W:
                    over = (abs(float((W - N) @ p)) - POSE_BREAK_SIDE * abs(float(s))) / T
                    if over > POSE_BREAK_SIDE_NOISE:
                        why.add("hands out to the sides")
                        side_clear = side_clear or over > POSE_BREAK_SIDE_CLEAR
            if _vis(kp, ej) and np.linalg.norm(_xy(kp, ej) - pts[ej]) > POSE_BREAK_ELBOW * T:
                why.add("elbows out of place")
        if arms in ("behind the back", "in front of the body", "above the head") \
                and _vis(kp, RWRI) and _vis(kp, LWRI) \
                and np.linalg.norm(_xy(kp, RWRI) - _xy(kp, LWRI)) > POSE_BREAK_WRISTS_APART * T:
            why.add("wrists apart")
    if spec["legs"] in ("ankles together", "ankles to the wrists") \
            and _vis(kp, RANK) and _vis(kp, LANK):
        sep = float(np.linalg.norm(_xy(kp, RANK) - _xy(kp, LANK)))
        if sep > (spec["ankle_gap"] + POSE_BREAK_ANKLES_EXTRA) * T:
            why.add("ankles apart")
    return why, side_clear


_LEG_BREAKS = frozenset({"ankles apart"})


def _check(kps, idx, g, spec, latch=None):
    per = []
    for i, kp in enumerate(kps):
        why, clear = _frame_breaks(kp, g, i, spec)
        arms_on, legs_on = _is_held(latch, "arms", idx[i]), _is_held(latch, "legs", idx[i])
        why = {r for r in why if (legs_on if r in _LEG_BREAKS else arms_on)}
        per.append((why, clear and arms_on))
    side = [("hands out to the sides" in w) for w, _c in per]
    run = [0] * len(side)
    i = 0
    while i < len(side):
        if not side[i]:
            i += 1
            continue
        j = i
        while j < len(side) and side[j]:
            j += 1
        run[i:j] = [j - i] * (j - i)
        i = j
    flags, frames, reasons = [], [], set()
    for i, (why, clear) in enumerate(per):
        why = set(why)
        if side[i] and not clear and run[i] < POSE_BREAK_SIDE_RUN:
            why.discard("hands out to the sides")
        flags.append(bool(why))
        if why:
            frames.append(int(idx[i]))
            reasons |= why
    n = len(flags)
    share = (sum(flags) / n) if n else 0.0
    broken = bool(n) and (_runs(flags) >= POSE_BREAK_RUN or share >= POSE_BREAK_SHARE)
    return broken, frames, reasons, 1.0 - share


def _fall_settle(kps, idx):
    out = [k.copy() for k in kps]
    n = len(kps)
    floor = -math.inf
    support = []
    for k in kps:
        legs = [k[j, 1] for j in (RKNE, LKNE, RANK, LANK) if _vis(k, j)]
        if legs:
            floor = max(floor, max(legs))
        N, H = _neck(k), _midhip(k)
        sup = False
        if N is not None and H is not None and math.isfinite(floor):
            T = float(np.linalg.norm(H - N))
            if T > 1.0:
                u = (H - N) / T
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(u[1])))))
                sh = [k[j, 1] for j in (RSHO, LSHO) if _vis(k, j)]
                sh_y = float(np.mean(sh)) if sh else float(N[1])
                for j in (RWRI, LWRI):
                    if _vis(k, j):
                        wy = float(k[j, 1])
                        if abs(floor - wy) <= POSE_FALL_WRIST_FLOOR * T \
                                and wy - sh_y > POSE_FALL_WRIST_BELOW * T \
                                and tilt > POSE_FALL_TILT_DEG:
                            sup = True
        support.append((sup, floor))
    applied = False
    weight, in_support, off_run, prev_idx = 0.0, False, 0, None
    for i in range(n):
        sup, fl = support[i]
        if sup:
            in_support, off_run = True, 0
        elif in_support:
            off_run += 1
            if off_run >= POSE_FALL_LEAVE_RUN:
                in_support = False
        step = 0.0 if prev_idx is None else (idx[i] - prev_idx) / float(POSE_FALL_RAMP)
        if in_support and prev_idx is None:
            step = 1.0 / POSE_FALL_RAMP
        prev_idx = idx[i]
        weight = min(1.0, weight + step) if in_support else max(0.0, weight - step)
        if weight <= 0.0:
            continue
        k = out[i]
        N, H = _neck(k), _midhip(k)
        if N is None or H is None or not math.isfinite(fl):
            continue
        v = N - H
        L = float(np.linalg.norm(v))
        if L <= 1.0:
            continue
        target = fl - POSE_FALL_NECK_ABOVE * L
        need = target - float(N[1])
        if need <= 0:
            continue
        pts = [j for j in UPPER_POINTS if k[j, 2] > 0]
        yt = target - float(H[1])
        if abs(yt) <= L:
            phi = math.atan2(v[1], v[0])
            c1 = math.asin(yt / L)
            cand = (c1, math.pi - c1)
            pick = cand[0] if (math.cos(cand[0]) >= 0) == (math.cos(phi) >= 0) else cand[1]
            th = (pick - phi + math.pi) % (2 * math.pi) - math.pi
            th *= weight
            c, s = math.cos(th), math.sin(th)
            R = np.array([[c, -s], [s, c]])
            for j in pts:
                k[j, :2] = H + R @ (k[j, :2] - H)
        else:
            for j in pts:
                k[j, 1] += need * weight
        applied = True
    return out, applied


def _two_bone(hip, ankle, l1, l2, u):
    d_vec = ankle - hip
    d = float(np.linalg.norm(d_vec))
    if d < 1e-6:
        return hip + l1 * u
    dirv = d_vec / d
    dc = min(max(d, abs(l1 - l2) + 1e-6), l1 + l2 - 1e-6)
    cosa = (l1 * l1 + dc * dc - l2 * l2) / (2 * l1 * dc)
    a = math.acos(max(-1.0, min(1.0, cosa)))
    best = None
    for sgn in (1.0, -1.0):
        c, s = math.cos(sgn * a), math.sin(sgn * a)
        kv = np.array([c * dirv[0] - s * dirv[1], s * dirv[0] + c * dirv[1]])
        knee = hip + l1 * kv
        score = float((knee - hip) @ u)
        if best is None or score > best[0]:
            best = (score, knee)
    return best[1]


def _limb_ratios(kps, g):
    out = {}
    for side, (h, k, a) in (("R", (RHIP, RKNE, RANK)), ("L", (LHIP, LKNE, LANK))):
        th, sh = [], []
        for i, kp in enumerate(kps):
            T = float(g.T[i])
            if _vis(kp, h) and _vis(kp, k):
                th.append(float(np.linalg.norm(_xy(kp, h) - _xy(kp, k))) / T)
            if _vis(kp, k) and _vis(kp, a):
                sh.append(float(np.linalg.norm(_xy(kp, k) - _xy(kp, a))) / T)
        out[side] = (float(np.median(th)) if th else POSE_LIMB_DEFAULT,
                     float(np.median(sh)) if sh else POSE_LIMB_DEFAULT)
    return out


def _apart_ratio(kps, g):
    for i, kp in enumerate(kps):
        if _vis(kp, RANK) and _vis(kp, LANK):
            sep = float(np.linalg.norm(_xy(kp, RANK) - _xy(kp, LANK))) / float(g.T[i])
            return min(POSE_APART_MAX, max(POSE_APART_MIN, sep))
    return (POSE_APART_MIN + POSE_APART_MAX) / 2.0


def _rewrite_legs(out, kp, g, i, spec, limbs, apart):
    legs = spec["legs"]
    T = float(g.T[i])
    if legs in ("ankles together", "held apart"):
        if not (_vis(kp, RANK) and _vis(kp, LANK)):
            return
        AR, AL = _xy(kp, RANK), _xy(kp, LANK)
        sep = float(np.linalg.norm(AR - AL))
        mid = (AR + AL) / 2.0
        d = (AR - AL) / sep if sep > 1e-6 else g.p[i]
        if legs == "ankles together":
            lim = spec["ankle_gap"] * T
            if sep <= lim:
                return
            target = lim
        else:
            target = apart * T
        newR, newL = mid + d * target / 2.0, mid - d * target / 2.0
        for a_j, k_j, new in ((RANK, RKNE, newR), (LANK, LKNE, newL)):
            corr = new - out[a_j, :2]
            out[a_j, :2] = new
            if out[k_j, 2] >= POSE_CONF:
                out[k_j, :2] += corr / 2.0
    elif legs == "ankles to the wrists":
        ankle = _hogtie_ankle(g, i)
        H = g.N[i] + T * g.u[i]
        for side, (h_j, k_j, a_j) in (("R", (RHIP, RKNE, RANK)), ("L", (LHIP, LKNE, LANK))):
            hip = _xy(kp, h_j) if _vis(kp, h_j) else H
            l1, l2 = limbs[side]
            out[k_j, :2] = _two_bone(hip, ankle, l1 * T, l2 * T, g.u[i])
            out[k_j, 2] = 1.0
            out[a_j, :2] = ankle
            out[a_j, 2] = 1.0


def _rewrite(kps1, kps, idx, g, spec, blend, latch=None):
    limbs = _limb_ratios(kps, g)
    apart = _apart_ratio(kps, g)
    out = []
    for i, kp in enumerate(kps):
        o = kp.copy()
        if not _is_held(latch, "arms", idx[i]):
            for j in (RELB, LELB, RWRI, LWRI):
                o[j] = kps1[i][j]
        elif spec["arms"]:
            pts, drawn = _arm_points(g, i, spec["arms"])
            alpha = min(1.0, idx[i] / float(POSE_BLEND_FRAMES)) if blend else 1.0
            for j, P in pts.items():
                is_wrist = j in (RWRI, LWRI)
                conf = 1.0 if (drawn or not is_wrist) else 0.0
                src = kps1[i]
                if alpha < 1.0 and _vis(src, j):
                    o[j, :2] = (1.0 - alpha) * _xy(src, j) + alpha * P
                    o[j, 2] = 1.0
                else:
                    o[j, :2] = P
                    o[j, 2] = conf
        if _is_held(latch, "legs", idx[i]):
            _rewrite_legs(o, kp, g, i, spec, limbs, apart)
        else:
            for j in (RKNE, LKNE, RANK, LANK):
                o[j] = kps1[i][j]
        out.append(o)
    return out


_APP_K = POSE_APP_HUE_BINS * POSE_APP_SAT_BINS + POSE_APP_VAL_BINS
_APP_R = len(POSE_APP_REGIONS)
POSE_APP_SIZE = _APP_R * _APP_K
_HEAD_POINTS = (NOSE, REYE, LEYE, REAR, LEAR)


def _rgb_u8(image):
    try:
        if torch.is_tensor(image):
            image = image.detach().cpu()
            if image.is_floating_point():
                image = (image.clamp(0, 1) * 255.0).round().to(torch.uint8)
            image = image.numpy()
        a = np.asarray(image)
    except Exception:
        return None
    if a.ndim != 3 or a.shape[2] < 3 or a.shape[0] < 1 or a.shape[1] < 1:
        return None
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.asarray(a, dtype=np.float64)
        if np.isfinite(a).all() and a.max() <= 1.0 + 1e-6:
            a = a * 255.0
        a = np.clip(np.nan_to_num(np.round(a)), 0, 255).astype(np.uint8)
    return a


def _gather(img, pts):
    H, W = img.shape[:2]
    xi = np.round(pts[:, 0]).astype(np.int64)
    yi = np.round(pts[:, 1]).astype(np.int64)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    return img[yi[ok], xi[ok]]


def _grid_counts(na, nb):
    if na * nb > POSE_APP_MAX_PIXELS:
        f = math.sqrt(POSE_APP_MAX_PIXELS / float(na * nb))
        na, nb = max(2, int(na * f)), max(2, int(nb * f))
    return na, nb


def _sample_band(img, p0, p1, half):
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1.0 or half < 0.5:
        return img[:0, 0]
    u = d / L
    q = _rot(u)
    na, nb = _grid_counts(max(2, int(math.ceil(L))), max(2, int(math.ceil(2 * half))))
    ta = np.linspace(0.0, L, na)
    tb = np.linspace(-half, half, nb)
    pts = p0[None, None, :] + ta[:, None, None] * u[None, None, :] + tb[None, :, None] * q[None, None, :]
    return _gather(img, pts.reshape(-1, 2))


def _sample_disc(img, c, r):
    if r < 1.0:
        return img[:0, 0]
    n, _ = _grid_counts(max(3, int(math.ceil(2 * r))), max(3, int(math.ceil(2 * r))))
    xs = np.linspace(-r, r, n)
    X, Y = np.meshgrid(xs, xs)
    m = X * X + Y * Y <= r * r
    return _gather(img, np.stack([X[m] + c[0], Y[m] + c[1]], axis=1))


def _colour_hist(px):
    rgb = px.astype(np.float64) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    v = rgb.max(axis=1)
    d = v - rgb.min(axis=1)
    s = np.where(v > 0, d / np.maximum(v, 1e-9), 0.0)
    h = np.zeros(len(rgb))
    dd = np.maximum(d, 1e-9)
    rm = (v == r) & (d > 0)
    gm = (v == g) & (d > 0) & ~rm
    bm = (d > 0) & ~rm & ~gm
    h[rm] = np.mod((g[rm] - b[rm]) / dd[rm], 6.0)
    h[gm] = (b[gm] - r[gm]) / dd[gm] + 2.0
    h[bm] = (r[bm] - g[bm]) / dd[bm] + 4.0
    h = h / 6.0
    out = np.zeros(_APP_K)
    col = (s >= POSE_APP_SAT_MIN) & (v >= POSE_APP_VAL_MIN)
    if col.any():
        fh = h[col] * POSE_APP_HUE_BINS - 0.5
        h0 = np.floor(fh)
        wh = fh - h0
        h0 = h0.astype(np.int64) % POSE_APP_HUE_BINS
        h1 = (h0 + 1) % POSE_APP_HUE_BINS
        fs = np.clip((s[col] - POSE_APP_SAT_MIN) / (1.0 - POSE_APP_SAT_MIN) * POSE_APP_SAT_BINS - 0.5,
                     0.0, POSE_APP_SAT_BINS - 1.0)
        s0 = np.floor(fs).astype(np.int64)
        ws = fs - s0
        s1 = np.minimum(s0 + 1, POSE_APP_SAT_BINS - 1)
        chrom = np.zeros((POSE_APP_HUE_BINS, POSE_APP_SAT_BINS))
        for hi, hw in ((h0, 1.0 - wh), (h1, wh)):
            for si, sw in ((s0, 1.0 - ws), (s1, ws)):
                np.add.at(chrom, (hi, si), hw * sw)
        out[:POSE_APP_HUE_BINS * POSE_APP_SAT_BINS] = chrom.ravel()
    grey = ~col
    if grey.any():
        fv = np.clip(v[grey] * POSE_APP_VAL_BINS - 0.5, 0.0, POSE_APP_VAL_BINS - 1.0)
        v0 = np.floor(fv).astype(np.int64)
        wv = fv - v0
        v1 = np.minimum(v0 + 1, POSE_APP_VAL_BINS - 1)
        tail = np.zeros(POSE_APP_VAL_BINS)
        np.add.at(tail, v0, 1.0 - wv)
        np.add.at(tail, v1, wv)
        out[POSE_APP_HUE_BINS * POSE_APP_SAT_BINS:] = tail
    tot = out.sum()
    return out / tot if tot > 0 else out


def person_appearance(image, keypoints):
    kp = _person(keypoints)
    img = _rgb_u8(image)
    if kp is None or img is None:
        return None
    N, Hm = _neck(kp), _midhip(kp)
    T = float(np.linalg.norm(Hm - N)) if N is not None and Hm is not None else 0.0
    if T <= 1.0:
        return None
    u = (Hm - N) / T
    regions = []
    if _vis(kp, RSHO) and _vis(kp, LSHO):
        half_sh = float(np.linalg.norm(_xy(kp, RSHO) - _xy(kp, LSHO))) / 2.0
    elif _vis(kp, RSHO) or _vis(kp, LSHO):
        half_sh = float(np.linalg.norm(_xy(kp, RSHO if _vis(kp, RSHO) else LSHO) - N))
    else:
        half_sh = POSE_HALF_SHOULDER * T
    half = min(max(POSE_APP_TORSO_SHRINK * half_sh, POSE_APP_TORSO_MIN * T), POSE_APP_TORSO_MAX * T)
    regions.append(_sample_band(img, N + POSE_APP_TORSO_A[0] * T * u, N + POSE_APP_TORSO_A[1] * T * u, half))
    hp = [_xy(kp, j) for j in _HEAD_POINTS if _vis(kp, j)]
    regions.append(_sample_disc(img, np.mean(hp, axis=0) - POSE_APP_HEAD_UP * T * u, POSE_APP_HEAD_R * T)
                   if hp else None)
    legs = []
    for h_j, k_j in ((RHIP, RKNE), (LHIP, LKNE)):
        if _vis(kp, h_j) and _vis(kp, k_j):
            a, b = _xy(kp, h_j), _xy(kp, k_j)
            legs.append(_sample_band(img, a + POSE_APP_LEG_A[0] * (b - a), a + POSE_APP_LEG_A[1] * (b - a),
                                     POSE_APP_LEG_HALF * T))
    regions.append(np.concatenate(legs, axis=0) if legs else None)
    out = np.full((_APP_R, _APP_K), np.nan)
    for r, px in enumerate(regions):
        if px is not None and len(px) >= POSE_APP_MIN_PIXELS:
            out[r] = _colour_hist(px)
    if not np.isfinite(out).any():
        return None
    return out.ravel().astype(np.float32)


def _app_vec(x):
    if x is None:
        return None
    try:
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        a = np.asarray(x, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if a.size != POSE_APP_SIZE or not np.isfinite(a).any():
        return None
    return a


def appearance_distance(a, b):
    a, b = _app_vec(a), _app_vec(b)
    if a is None or b is None:
        return None
    A, B = a.reshape(_APP_R, _APP_K), b.reshape(_APP_R, _APP_K)
    tot = wsum = 0.0
    for r, wgt in enumerate(POSE_APP_WEIGHTS):
        if not (np.isfinite(A[r]).all() and np.isfinite(B[r]).all()):
            continue
        pa, pb = np.clip(A[r], 0.0, None), np.clip(B[r], 0.0, None)
        norm = math.sqrt(float(pa.sum()) * float(pb.sum()))
        bc = float(np.sqrt(pa * pb).sum()) / norm if norm > 0 else 0.0
        tot += wgt * math.sqrt(max(0.0, 1.0 - bc))
        wsum += wgt
    if wsum < POSE_APP_MIN_SHARED - 1e-9:
        return None
    return tot / wsum


def _app_acc():
    return [np.zeros((_APP_R, _APP_K)), np.zeros(_APP_R)]


def _app_add(acc, v):
    if v is None:
        return
    blocks = np.asarray(v, dtype=np.float64).reshape(_APP_R, _APP_K)
    for r in range(_APP_R):
        if np.isfinite(blocks[r]).all():
            acc[0][r] += blocks[r]
            acc[1][r] += 1


def _app_mean(acc):
    if acc is None or not (acc[1] > 0).any():
        return None
    out = np.full((_APP_R, _APP_K), np.nan)
    for r in range(_APP_R):
        if acc[1][r] > 0:
            out[r] = acc[0][r] / acc[1][r]
    return out.ravel()


def _app_mean_of(vecs):
    acc = _app_acc()
    for v in vecs:
        _app_add(acc, v)
    return _app_mean(acc)


def _region_box(kp):
    T = _torso_len(kp)
    pts = [kp[j, :2] for j in (NECK, RSHO, LSHO, RHIP, LHIP, RKNE, LKNE) + _HEAD_POINTS
           if kp[j, 2] >= POSE_CONF]
    if not T or len(pts) < 2:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    pad = POSE_APP_CLEAN_PAD * T
    (x0, y0), (x1, y1) = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    return [float(x0), float(y0), float(x1), float(y1)]


def _clean_flags(plist):
    boxes = [_region_box(k) for k in plist]
    pboxes = [_point_box(k) for k in plist]
    out = []
    for i, rb in enumerate(boxes):
        ok = rb is not None
        for j, pb in enumerate(pboxes):
            if not ok:
                break
            if j != i and pb is not None and pb[0] <= rb[2] and pb[2] >= rb[0] \
                    and pb[1] <= rb[3] and pb[3] >= rb[1]:
                ok = False
        out.append(ok)
    return out


class _Scene:
    __slots__ = ("people", "index", "app", "clean", "has_app")

    def __init__(self, people, index, app=None):
        self.people, self.index = people, list(index)
        self.app = app
        self.has_app = app is not None and any(v is not None for fr in app for v in fr)
        self.clean = [_clean_flags(pl) for pl in people] if self.has_app else None

    def desc(self, p, pi):
        if not self.has_app or pi >= len(self.app[p]):
            return None
        return self.app[p][pi]

    def clean_desc(self, p, pi):
        return self.desc(p, pi) if self.has_app and self.clean[p][pi] else None

    def mean(self, pmap, lo=None, hi=None, count=None, from_end=False, clean=True):
        if not self.has_app:
            return None
        ps = [p for p in sorted(pmap) if (lo is None or p >= lo) and (hi is None or p <= hi)]
        vecs = [self.clean_desc(p, pmap[p]) for p in ps]
        vecs = [v for v in vecs if v is not None]
        if not vecs and not clean:
            vecs = [v for v in (self.desc(p, pmap[p]) for p in ps) if v is not None]
        if count:
            vecs = vecs[-count:] if from_end else vecs[:count]
        return _app_mean_of(vecs)


def _centre(kp):
    return _centre_q(kp)[0]


def _centre_q(kp):
    n, h = _neck(kp), _midhip(kp)
    if n is not None and h is not None and float(np.linalg.norm(h - n)) > 1.0:
        return (n + h) / 2.0, True
    box = _torso_box(kp)
    if box is not None:
        return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]), False
    m = kp[:, 2] >= POSE_CONF
    if m.any():
        return kp[m, :2].mean(axis=0), False
    return None, False


def _signature(kp):
    out = np.full(3, np.nan)
    n, h = _neck(kp), _midhip(kp)
    if n is None or h is None:
        return out
    T = float(np.linalg.norm(h - n))
    if T <= 1.0:
        return out
    if _vis(kp, RSHO) and _vis(kp, LSHO):
        out[0] = float(np.linalg.norm(_xy(kp, RSHO) - _xy(kp, LSHO))) / T
    for c, pairs in ((1, ((RSHO, RELB), (LSHO, LELB))), (2, ((RHIP, RKNE), (LHIP, LKNE)))):
        ls = [float(np.linalg.norm(_xy(kp, a) - _xy(kp, b))) / T for a, b in pairs
              if _vis(kp, a) and _vis(kp, b)]
        if ls:
            out[c] = float(np.mean(ls))
    return out


def _sig_mean(sigs):
    out = np.full(3, np.nan)
    if not len(sigs):
        return out
    a = np.asarray(sigs, dtype=np.float64)
    for c in range(3):
        col = a[:, c][np.isfinite(a[:, c])]
        if len(col):
            out[c] = float(col.mean())
    return out


def _sig_dist(a, b, worst=False):
    if a is None or b is None:
        return None
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return None
    d = np.abs(a[m] - b[m])
    return float(d.max() if worst else d.mean())


def _posture_dist(a, b, T):
    ca, cb = _centre(a), _centre(b)
    m = (a[:, 2] >= POSE_CONF) & (b[:, 2] >= POSE_CONF)
    if ca is None or cb is None or not m.any():
        return 0.0
    d = float(np.linalg.norm((a[m, :2] - ca) - (b[m, :2] - cb), axis=1).mean()) / T
    return min(d, 1.0)


def _predict(hist, t):
    hist = hist[-POSE_TRACK_VEL_POINTS:]
    if len(hist) == 1:
        return np.asarray(hist[0][1], dtype=np.float64)
    ts = np.array([h[0] for h in hist], dtype=np.float64)
    cs = np.array([h[1] for h in hist], dtype=np.float64)
    tm, cm = ts.mean(), cs.mean(axis=0)
    var = float(((ts - tm) ** 2).sum())
    if var <= 0.0:
        return cs[-1]
    v = ((ts - tm)[:, None] * (cs - cm)).sum(axis=0) / var
    return cm + v * (t - tm)


def _reach(gap):
    return POSE_TRACK_GATE + POSE_TRACK_GATE_GROW * max(0, gap - 1)


def _point_box(kp):
    pts = kp[kp[:, 2] >= POSE_CONF, :2]
    if len(pts) < 2:
        return None
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    return [float(x0), float(y0), float(x1), float(y1)]


def _in_contact(a, b, box=None, gap=None):
    box = box or _torso_box
    gap = POSE_CONTACT_GAP if gap is None else gap
    ba, bb = box(a), box(b)
    if ba is None or bb is None:
        return False
    Ts = [x for x in (_torso_len(a), _torso_len(b)) if x]
    if not Ts:
        return True
    dx = max(0.0, ba[0] - bb[2], bb[0] - ba[2])
    dy = max(0.0, ba[1] - bb[3], bb[1] - ba[3])
    return math.hypot(dx, dy) < gap * float(np.mean(Ts))


def _lsa():
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment
    except Exception:
        return None


def _match_exact(w):
    R, C = w.shape
    if C > POSE_TRACK_EXACT_MAX:
        pairs, ur, uc = [], set(), set()
        for v, r, c in sorted((w[r, c], r, c) for r in range(R) for c in range(C) if w[r, c] < 0):
            if r not in ur and c not in uc:
                pairs.append((r, c))
                ur.add(r)
                uc.add(c)
        return pairs
    memo = {}

    def best(r, used):
        if r == R:
            return 0.0, ()
        key = (r, used)
        if key not in memo:
            b = best(r + 1, used)
            for c in range(C):
                if w[r, c] < 0 and not (used >> c) & 1:
                    v, rest = best(r + 1, used | (1 << c))
                    if v + w[r, c] < b[0] - 1e-12:
                        b = (v + w[r, c], ((r, c),) + rest)
            memo[key] = b
        return memo[key]

    return list(best(0, 0)[1])


def _match(cost, new_cost):
    R, C = cost.shape
    if R == 0 or C == 0:
        return []
    w = np.where(np.isfinite(cost), np.minimum(cost - new_cost, 0.0), 0.0)
    if not (w < 0).any():
        return []
    lsa = _lsa()
    if lsa is not None:
        rows, cols = lsa(w)
        return [(int(r), int(c)) for r, c in zip(rows, cols) if w[r, c] < 0]
    return _match_exact(w)


def _new_track():
    return {"pos": {}, "last": None, "kp": None, "box": None, "T": None, "T_ok": False,
            "hist": [], "sig": np.full(3, np.nan), "app": _app_acc()}


def _track_hist(hist):
    good = [(f, c) for f, c, ok in hist if ok]
    return (good or [(f, c) for f, c, _ok in hist])[-POSE_TRACK_VEL_POINTS:]


def _extend(t, pos, pi, kp, frame=None, desc=None):
    t["pos"][pos] = pi
    t["last"] = pos
    t["kp"] = kp
    t["box"] = _torso_box(kp) or t["box"]
    c, ok = _centre_q(kp)
    T = _torso_len(kp)
    if T and (ok or not t["T_ok"]):
        if ok and not t["T_ok"]:
            t["T"], t["T_ok"] = T, True
        else:
            t["T"] = T if not t["T"] else (1 - POSE_TRACK_T_SMOOTH) * t["T"] + POSE_TRACK_T_SMOOTH * T
    if c is not None:
        keep = POSE_TRACK_VEL_POINTS + POSE_TRACK_HOLD + 1
        t["hist"] = (t["hist"] + [(pos if frame is None else frame, c, ok)])[-keep:]
    s, old = _signature(kp), t["sig"]
    fresh = np.isfinite(s) & ~np.isfinite(old)
    blend = np.isfinite(s) & np.isfinite(old)
    old[fresh] = s[fresh]
    old[blend] = (1 - POSE_SIG_SMOOTH) * old[blend] + POSE_SIG_SMOOTH * s[blend]
    if desc is not None:
        _app_add(t.setdefault("app", _app_acc()), desc)


def _track_cost(t, kp, pos, frame, desc=None):
    c = _centre(kp)
    if c is None or not t["hist"]:
        return math.inf
    gap = pos - t["last"]
    T = t["T"] or _torso_len(kp) or 100.0
    motion = float(np.linalg.norm(c - _predict(_track_hist(t["hist"]), frame))) / T
    if motion > _reach(gap):
        return math.inf
    cost = motion + POSE_TRACK_STALE_COST * (gap - 1)
    if t["kp"] is not None:
        cost += POSE_TRACK_POSE_W * _posture_dist(t["kp"], kp, T)
    Td = _torso_len(kp)
    if Td and t["T"]:
        cost += POSE_TRACK_SCALE_W * abs(math.log(Td / t["T"]))
    sd = _sig_dist(t["sig"], _signature(kp))
    if sd is not None:
        cost += POSE_SIG_W * sd
    if desc is not None:
        ad = appearance_distance(_app_mean(t.get("app")), desc)
        if ad is not None:
            if ad > POSE_APP_MATCH_MAX:
                return math.inf
            cost += POSE_APP_W * ad
    return cost


def _build_tracks(people, index=None, scene=None):
    index = list(range(len(people))) if index is None else list(index)
    tracks = []

    def desc(pos, pi):
        return scene.clean_desc(pos, pi) if scene is not None else None
    for pos, plist in enumerate(people):
        frame = index[pos]
        live = [ti for ti, t in enumerate(tracks) if pos - t["last"] <= POSE_TRACK_HOLD + 1]
        dets = [pi for pi, kp in enumerate(plist) if kp is not None and (kp[:, 2] >= POSE_CONF).any()]
        cost = np.full((len(live), len(dets)), math.inf)
        for r, ti in enumerate(live):
            for c, pi in enumerate(dets):
                cost[r, c] = _track_cost(tracks[ti], plist[pi], pos, frame, desc(pos, pi))
        matched = set()
        for r, c in _match(cost, POSE_TRACK_NEW_COST):
            _extend(tracks[live[r]], pos, dets[c], plist[dets[c]], frame, desc(pos, dets[c]))
            matched.add(c)
        for c, pi in enumerate(dets):
            if c not in matched:
                t = _new_track()
                _extend(t, pos, pi, plist[pi], frame, desc(pos, pi))
                tracks.append(t)
    return tracks


def _sightings(pos_map, people, index, measured=True):
    out, good = [], []
    for p in sorted(pos_map):
        kp = people[p][pos_map[p]]
        c, ok = _centre_q(kp)
        if c is not None:
            out.append((p, index[p], c, _torso_len(kp), _signature(kp)))
            if ok:
                good.append(out[-1])
    return (good if measured else []) or out


def _motion_cost(seq, start, gate, count=POSE_TRACK_VEL_POINTS):
    return _motion_terms(seq, start, gate, count)[0]


def _motion_terms(seq, start, gate, count=POSE_TRACK_VEL_POINTS):
    tot, terms = 0.0, 0
    for i in range(max(1, start), min(len(seq), start + count)):
        hist = seq[max(0, i - POSE_TRACK_VEL_POINTS):i]
        Ts = [h[3] for h in hist if h[3]]
        T = float(np.median(Ts)) if Ts else (seq[i][3] or 100.0)
        m = float(np.linalg.norm(seq[i][2] - _predict([(h[1], h[2]) for h in hist], seq[i][1]))) / T
        if gate and m > _reach(seq[i][0] - seq[i - 1][0]):
            return math.inf, terms + 1
        tot += m
        terms += 1
    return tot, terms


def _swap_terms(A, B, k):
    Ah, At = [s for s in A if s[0] < k], [s for s in A if s[0] >= k]
    Bh, Bt = [s for s in B if s[0] < k], [s for s in B if s[0] >= k]
    if not Ah or not At or not (Bh or Bt):
        return None
    swap = _motion_cost(Ah + Bt, len(Ah), True) + _motion_cost(Bh + At, len(Bh), True)
    if not math.isfinite(swap):
        return math.inf, math.inf
    keep = _motion_cost(Ah + At, len(Ah), False) + _motion_cost(Bh + Bt, len(Bh), False)

    def sig(seq):
        return _sig_mean([s[4] for s in seq]) if seq else None

    def d(a, b):
        v = _sig_dist(a, b)
        return 0.0 if v is None else v

    sAh, sAt = sig(Ah[-POSE_SIG_FRAMES:]), sig(At[:POSE_SIG_FRAMES])
    sBh, sBt = sig(Bh[-POSE_SIG_FRAMES:]), sig(Bt[:POSE_SIG_FRAMES])
    return swap - keep, (d(sAh, sBt) + d(sBh, sAt)) - (d(sAh, sAt) + d(sBh, sBt))


def _swap_margin(A, B, k):
    t = _swap_terms(A, B, k)
    if t is None:
        return None
    if not math.isfinite(t[0]):
        return math.inf
    return t[0] + POSE_SIG_W * t[1]


def _exchange_margin(A, B, k):
    ia = next((i for i, s in enumerate(A) if s[0] == k), None)
    ib = next((i for i, s in enumerate(B) if s[0] == k), None)
    if ia is None or ib is None:
        return None
    A2 = A[:ia] + [B[ib]] + A[ia + 1:]
    B2 = B[:ib] + [A[ia]] + B[ib + 1:]
    n = POSE_TRACK_VEL_POINTS + 1
    (sa, ta), (sb, tb) = _motion_terms(A2, ia, True, n), _motion_terms(B2, ib, True, n)
    if not ta and not tb:
        return None
    swap = sa + sb
    if not math.isfinite(swap):
        return math.inf
    keep = _motion_cost(A, ia, False, n) + _motion_cost(B, ib, False, n)

    def around(seq, i):
        near = seq[max(0, i - POSE_SIG_FRAMES):i] + seq[i + 1:i + 1 + POSE_SIG_FRAMES]
        return _sig_mean([s[4] for s in near])

    def d(a, b):
        v = _sig_dist(a, b)
        return 0.0 if v is None else v

    sA, sB = around(A, ia), around(B, ib)
    keep += POSE_SIG_W * (d(sA, A[ia][4]) + d(sB, B[ib][4]))
    swap += POSE_SIG_W * (d(sA, B[ib][4]) + d(sB, A[ia][4]))
    return swap - keep


def _exchanges(pa, pb, people, index):
    A = _sightings(pa, people, index, measured=False)
    B = _sightings(pb, people, index, measured=False)
    out = []
    for k in sorted(set(pa) & set(pb)):
        ka, kb = people[k][pa[k]], people[k][pb[k]]
        if not _in_contact(ka, kb, _point_box):
            continue
        if not (_centre_q(ka)[1] and _centre_q(kb)[1]):
            out.append(k)
            continue
        m = _exchange_margin(A, B, k)
        if m is not None and m < POSE_CROSS_MARGIN:
            out.append(k)
    return out


def _crossing(A, B):
    if not A or not B:
        return None
    bpos = [s[0] for s in B]
    lo, hi = A[0][0], A[-1][0]
    for k in sorted({s[0] for s in A} | set(bpos)):
        if k <= lo or k > hi:
            continue
        if not any(abs(p - k) <= POSE_TRACK_HOLD + 1 for p in bpos):
            continue
        m = _swap_margin(A, B, k)
        if m is not None and m < POSE_CROSS_MARGIN:
            return k
    return None


def _track_seq(track, people, positions=None):
    ps = sorted(track["pos"]) if positions is None else [p for p in sorted(track["pos"]) if p in positions]
    return ps, [people[p][track["pos"][p]] for p in ps]


def _fit_of(track, people, index, spec, positions):
    return _fit_why(track, people, index, spec, positions)[0]


def _fit_why(track, people, index, spec, positions):
    ps, kps = _track_seq(track, people, positions)
    if not ps:
        return None, ""
    return _track_fit_why(kps, [index[p] for p in ps], spec)


def _assign(names, cands, fits, specs):
    k = len(names)
    if len(cands) < k:
        return None, f"found {len(cands)} people for {k} restrained"
    if len(cands) > POSE_ID_MAX_PEOPLE:
        return None, f"too many people to tell apart ({len(cands)})"
    group = {nm: (specs[nm]["arms"], specs[nm]["legs"]) for nm in names}
    fits = {nm: {t: (math.inf if f is None else float(f)) for t, f in fits[nm].items()}
            for nm in names}
    scored = []
    for combo in permutations(cands, k):
        cost, worst = 0.0, 0.0
        for nm, t in zip(names, combo):
            f = fits[nm].get(t, math.inf)
            cost += f
            worst = max(worst, f)
        key = frozenset((g, frozenset(t for nm2, t in zip(names, combo) if group[nm2] == g))
                        for g in set(group.values()))
        scored.append((cost, worst, key, combo))
    scored.sort(key=lambda s: s[0])
    best = scored[0]
    second = next((s for s in scored[1:] if s[2] != best[2]), None)
    fb = ", ".join(f"{fits[nm].get(t, math.inf):.2f}" for nm, t in zip(names, best[3]))
    if not math.isfinite(best[1]) or best[1] > POSE_FIT_MAX:
        tail = f" / {second[0]:.2f}" if second and math.isfinite(second[0]) else ""
        return None, f"could not tell who is bound: fits {fb}{tail} (over {POSE_FIT_MAX:.2f})"
    if second is not None and second[0] - best[0] < POSE_FIT_LEAD:
        return None, f"could not tell who is bound: fits {best[0]:.2f} / {second[0]:.2f}"
    return dict(zip(names, best[3])), ""


def _has_torso(track, people, positions):
    return any(sum(people[p][pi][j, 2] >= POSE_CONF for j in TORSO_POINTS) >= 2
               for p, pi in track["pos"].items() if p in positions)


def _identify(tracks, people, index, specs, carry, notes, scene=None, carry_app=None,
              windows=None):
    window = set(range(min(POSE_ID_FRAMES, len(people))))
    windows = windows or {}

    def win(nm, ti):
        return windows[nm].get(ti, set()) if nm in windows else window

    names = sorted(specs)
    cands = [ti for ti, t in enumerate(tracks)
             if (window & set(t["pos"])) or any(win(nm, ti) & set(t["pos"]) for nm in names)]
    found = {}
    if carry:
        opening = [ti for ti in cands if window & set(tracks[ti]["pos"])]
        for nm in names:
            box = carry.get(nm)
            if box is None:
                continue
            ious = []
            for ti in opening:
                first = min(p for p in tracks[ti]["pos"] if p in window)
                ious.append((_iou(box, _torso_box(people[first][tracks[ti]["pos"][first]])), ti))
            ious.sort(reverse=True)
            if not ious:
                continue
            best = ious[0]
            nxt = ious[1][0] if len(ious) > 1 else 0.0
            if best[0] >= POSE_CARRY_IOU_MIN and best[0] - nxt >= POSE_CARRY_IOU_LEAD:
                want = _app_vec((carry_app or {}).get(nm))
                if want is not None and scene is not None and scene.has_app:
                    t = tracks[best[1]]
                    seen = scene.mean({p: pi for p, pi in t["pos"].items() if p in window}, clean=False)
                    d = appearance_distance(want, seen)
                    if d is not None and d > POSE_APP_MATCH_MAX:
                        notes.append(f"the carried box is on someone who does not look like {nm}")
                        return None, (f"the carried box is on someone who does not look like "
                                      f"{nm}; not guessing")
                    if d is None:
                        notes.append(f"too little of {nm} seen to compare looks; carried by place")
                found[nm] = best[1]
        taken = list(found.values())
        if len(set(taken)) != len(taken):
            found = {}
            notes.append("carried identity was contradictory; identified by pose instead")
        elif found:
            notes.append("identified across the cut: " + ", ".join(sorted(found)))
    rest = [nm for nm in names if nm not in found]
    if not rest:
        return found, ""
    free = [ti for ti in cands if ti not in found.values()]
    fits, whys = {nm: {} for nm in rest}, {}
    for nm in rest:
        for ti in free:
            if nm in windows and ti not in windows[nm]:
                fits[nm][ti] = math.inf
                continue
            f, why = _fit_why(tracks[ti], people, index, specs[nm], win(nm, ti))
            fits[nm][ti] = f
            if f is None and why:
                whys.setdefault(ti, why)
    got, why = _assign(rest, free, fits, specs)
    if got is None:
        measurable = [ti for ti in free if any(fits[nm][ti] is not None for nm in rest)]
        blind = [whys[ti] for ti in free if ti not in measurable and ti in whys]
        if len(measurable) < len(rest) and blind:
            return None, blind[0]
        return None, why
    if len(cands) > 1:
        group = {nm: (specs[nm]["arms"], specs[nm]["legs"]) for nm in rest}
        for nm in rest:
            mine = {got[o] for o in rest if group[o] == group[nm]}
            fitting = {ti for ti in free if fits[nm][ti] is not None and fits[nm][ti] <= POSE_FIT_MAX}
            if fitting != mine:
                vals = " / ".join(f"{v:.2f}" for v in sorted(fits[nm][ti] for ti in fitting))
                return None, (f"could not tell who is bound: {len(fitting)} people fit {nm}'s "
                              f"restraint (fits {vals})")
        chosen = set(got.values())
        for ti in free:
            if ti in chosen or any(fits[nm][ti] is not None for nm in rest):
                continue
            if _has_torso(tracks[ti], people, window.union(*(win(nm, ti) for nm in rest))):
                why = whys.get(ti) or "arms not visible"
                return None, f"could not measure everyone's arms ({why}); not guessing"
        bound_ts = chosen | set(found.values())
        for nm in rest:
            A = tracks[got[nm]]
            for ti in cands:
                if ti in bound_ts:
                    continue
                B = tracks[ti]
                for p in sorted(win(nm, got[nm])):
                    if p in A["pos"] and p in B["pos"] and \
                            _in_contact(people[p][A["pos"][p]], people[p][B["pos"][p]]):
                        return None, "two people in contact and no carried identity; not guessing"
    found.update(got)
    if len(rest) > 1 and len({(specs[nm]["arms"], specs[nm]["legs"]) for nm in rest}) == 1:
        notes.append("restrained people share one pose; not told apart by name")
    return found, ""


def _reappear_ok(pre, post, people, index, spec, scene):
    if scene is not None and scene.has_app:
        d = appearance_distance(scene.mean(pre), scene.mean(post, count=POSE_APP_FRAMES))
        if d is not None:
            return d <= POSE_APP_MATCH_MAX
    sa = _sightings(pre, people, index)[-POSE_SIG_FRAMES:]
    sb = _sightings(post, people, index)[:POSE_SIG_FRAMES]
    sd = _sig_dist(_sig_mean([s[4] for s in sa]), _sig_mean([s[4] for s in sb]), worst=True)
    if sd is not None and sd > POSE_SIG_TOL:
        return False
    if spec is None or not (spec["arms"] or spec["legs"]):
        return False
    start = min(post)
    f = _fit_of({"pos": post}, people, index, spec, set(range(start, start + POSE_ID_FRAMES)))
    return f is not None and f <= POSE_FIT_MAX


def _cut_at_gaps(ti, tracks, people, index, spec_at, scene):
    t = tracks[ti]
    ps = sorted(t["pos"])
    for k in range(1, len(ps)):
        if ps[k] - ps[k - 1] <= 1:
            continue
        end = next((j for j in range(k + 1, len(ps)) if ps[j] - ps[j - 1] > 1), len(ps))
        pre = {p: t["pos"][p] for p in ps[:k]}
        post = {p: t["pos"][p] for p in ps[k:end]}
        if _reappear_ok(pre, post, people, index, spec_at(ps[k]), scene):
            continue
        nt = _new_track()
        nt["pos"] = {p: t["pos"].pop(p) for p in ps[k:]}
        nt["last"] = ps[-1]
        t["last"] = ps[k - 1]
        tracks.append(nt)
        return True
    return False


def _follow(track_no, tracks, people, index, spec, taken, scene=None, spec_at=None):
    spec_at = spec_at or (lambda p: spec)
    _cut_at_gaps(track_no, tracks, people, index, spec_at, scene)
    pos = dict((p, (track_no, pi)) for p, pi in tracks[track_no]["pos"].items())
    used = set(taken) | {track_no}
    while True:
        end = max(pos)
        later = sorted((min(t["pos"]), ti) for ti, t in enumerate(tracks)
                       if ti not in used and t["pos"] and min(t["pos"]) > end)
        if not later:
            break
        chain = {p: pi for p, (_t, pi) in pos.items()}
        seen = _sightings(chain, people, index)
        if not seen:
            break
        hist = [(s[1], s[2]) for s in seen]
        Ts = [s[3] for s in seen[-POSE_SIG_FRAMES:] if s[3]]
        T = float(np.median(Ts)) if Ts else 100.0
        sig_b = _sig_mean([s[4] for s in seen[-POSE_SIG_FRAMES:]])
        app_b = scene.mean(chain) if scene is not None else None
        ok, near = [], []
        for start, ti in later:
            first = people[start][tracks[ti]["pos"][start]]
            c = _centre(first)
            if c is None:
                continue
            gap = start - seen[-1][0]
            if float(np.linalg.norm(c - _predict(hist, index[start]))) / T > \
                    POSE_REACQ_RADIUS + POSE_REACQ_GROW * gap:
                continue
            ad = None
            if app_b is not None:
                ad = appearance_distance(app_b, scene.mean(tracks[ti]["pos"], count=POSE_APP_FRAMES))
            if ad is not None:
                if ad > POSE_APP_MATCH_MAX:
                    continue
            else:
                gs = spec_at(start)
                f = _fit_of(tracks[ti], people, index, gs, set(range(start, start + POSE_ID_FRAMES))) \
                    if gs is not None and (gs["arms"] or gs["legs"]) else None
                if f is None or f > POSE_FIT_MAX:
                    continue
            near.append((start, ti))
            mine = _sightings(tracks[ti]["pos"], people, index)[:POSE_SIG_FRAMES]
            sd = _sig_dist(sig_b, _sig_mean([s[4] for s in mine]), worst=True)
            if sd is not None and sd <= POSE_SIG_TOL:
                ok.append((start, ti))
        if not ok:
            break
        start0, ti0 = ok[0]
        if any(t != ti0 and abs(s - start0) < POSE_ID_FRAMES for s, t in near):
            return pos, used, True
        used.add(ti0)
        _cut_at_gaps(ti0, tracks, people, index, spec_at, scene)
        for p, pi in tracks[ti0]["pos"].items():
            pos.setdefault(p, (ti0, pi))
    return pos, used, False


def _episodes(pa, pb, people):
    contact = {k for k in set(pa) & set(pb) if _in_contact(people[k][pa[k]], people[k][pb[k]])}
    for X, Y in ((pa, pb), (pb, pa)):
        xs = sorted(X)
        for k in Y:
            if k in X:
                continue
            i = bisect.bisect_left(xs, k)
            if i == 0 or i == len(xs):
                continue
            here = people[k][Y[k]]
            if any(_in_contact(people[q][X[q]], here, _point_box) for q in (xs[i - 1], xs[i])):
                contact.add(k)
    out, cur = [], None
    for k in sorted(set(pa) | set(pb)):
        if k in contact:
            cur = [k, k] if cur is None else [cur[0], k]
        elif k in pa and k in pb and cur is not None:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def _app_pick(mX, mY, a, b):
    dXa, dXb = appearance_distance(mX, a), appearance_distance(mX, b)
    if dXa is None and dXb is None:
        return None
    a_ok = dXa is not None and dXa <= POSE_APP_MATCH_MAX
    b_ok = dXb is not None and dXb <= POSE_APP_MATCH_MAX
    if a_ok and b_ok:
        keep, swap = dXa, dXb
        if mY is not None:
            dYa, dYb = appearance_distance(mY, a), appearance_distance(mY, b)
            if dYa is not None and dYb is not None:
                keep, swap = keep + dYb, swap + dYa
        if keep + POSE_APP_MARGIN <= swap:
            return "keep"
        if swap + POSE_APP_MARGIN <= keep:
            return "swap"
        return "both"
    if a_ok:
        return "keep"
    if b_ok:
        return "swap"
    return None


def _swap_tails(X, Y, after):
    tx = {p: X.pop(p) for p in [p for p in X if p > after]}
    ty = {p: Y.pop(p) for p in [p for p in Y if p > after]}
    X.update(ty)
    Y.update(tx)


def _cut_tail(X, after):
    return {p: X.pop(p) for p in [p for p in X if p > after]}


_TOO_ALIKE = "two people look too alike to keep apart"
_UNCLEAR_AFTER = "could not tell who is who after two people met"


def _settle_by_app(scene, X, Y, s, e, mX, mY, postX, postY, extra, swapped, dropped):
    for a, b in ((mX, mY), (postX, postY)):
        d = appearance_distance(a, b)
        if d is not None and d < POSE_APP_DISTINCT_MIN:
            return _TOO_ALIKE, False
    unclear = []
    for k in range(s, e + 1):
        if k in X and k in Y:
            pick = _app_pick(mX, mY, scene.desc(k, X[k]), scene.desc(k, Y[k]))
            if pick == "swap":
                X[k], Y[k] = Y[k], X[k]
                swapped.add(scene.index[k])
            elif pick is None:
                X.pop(k)
                Y.pop(k)
                dropped.add(scene.index[k])
            elif pick == "both":
                unclear.append(k)
    if unclear:
        for k in _exchanges(dict(X), dict(Y), scene.people, scene.index):
            if k in unclear:
                X.pop(k, None)
                Y.pop(k, None)
                dropped.add(scene.index[k])
    if postX is not None and postY is not None:
        pick = _app_pick(mX, mY, postX, postY)
        if pick not in ("keep", "swap"):
            return (_TOO_ALIKE if pick == "both" else _UNCLEAR_AFTER), False
        if pick == "swap":
            _swap_tails(X, Y, e)
            swapped.add(scene.index[e])
    elif postX is not None:
        dX, dY = appearance_distance(mX, postX), appearance_distance(mY, postX)
        if dX is None or dX > POSE_APP_MATCH_MAX or (dY is not None and dY + POSE_APP_MARGIN <= dX):
            extra.append(_cut_tail(X, e))
        elif dY is not None and dX + POSE_APP_MARGIN > dY:
            return _UNCLEAR_AFTER, False
    elif postY is not None:
        dX, dY = appearance_distance(mX, postY), appearance_distance(mY, postY)
        if dX is not None and dX <= POSE_APP_MATCH_MAX and (dY is None or dX + POSE_APP_MARGIN <= dY):
            _swap_tails(X, Y, e)
            swapped.add(scene.index[e])
    return "", not unclear


def _settle_by_motion(scene, X, Y, s, e, name, spec_at, extra):
    people, index = scene.people, scene.index
    tx = sorted(p for p in X if p > e)
    ty = [p for p in Y if p > e]
    if tx and ty:
        A, B = _sightings(X, people, index), _sightings(Y, people, index)
        posed = False
        for k in range(s + 1, tx[0] + 1):
            t = _swap_terms(A, B, k)
            if t is None:
                continue
            posed = True
            if t[0] < POSE_CROSS_MARGIN or t[1] < POSE_CROSS_SIG_MARGIN:
                return f"lost track of {name} where two people met (frame {index[k]})"
        if not posed:
            return f"lost track of {name} where two people met (frame {index[s]})"
    elif tx:
        pre = {p: X[p] for p in X if p < s}
        post = {p: X[p] for p in tx}
        if pre and not _reappear_ok(pre, post, people, index, spec_at(tx[0]), None):
            extra.append(_cut_tail(X, e))
    return ""


def _settle_pair(scene, X, Y, name, carried, spec_at, swapped, dropped):
    cursor, by_app, any_ep, extra = -1, True, False, []
    while X:
        eps = [ep for ep in _episodes(X, Y, scene.people) if ep[0] > cursor]
        if not eps:
            break
        s, e = eps[0]
        any_ep, cursor = True, e
        mX = scene.mean(X, hi=s - 1)
        if mX is None:
            mX = carried
        if scene.has_app and mX is not None:
            why, decisive = _settle_by_app(scene, X, Y, s, e, mX, scene.mean(Y, hi=s - 1),
                                           scene.mean(X, lo=e + 1, count=POSE_APP_FRAMES),
                                           scene.mean(Y, lo=e + 1, count=POSE_APP_FRAMES), extra,
                                           swapped, dropped)
            by_app = by_app and decisive
        else:
            by_app = False
            why = _settle_by_motion(scene, X, Y, s, e, name, spec_at, extra)
        if why:
            return why, False, extra
    return "", any_ep and by_app, extra


def _fill(frames, index, t, hold_ends):
    if not frames:
        return None
    ps = sorted(frames)
    before = [p for p in ps if index[p] <= t]
    after = [p for p in ps if index[p] >= t]
    if before and index[before[-1]] == t:
        return frames[before[-1]]
    if before and after:
        a, b = frames[before[-1]], frames[after[0]]
        ia, ib = index[before[-1]], index[after[0]]
        f = (t - ia) / float(ib - ia)
        out = np.zeros((18, 3))
        both = (a[:, 2] >= POSE_CONF) & (b[:, 2] >= POSE_CONF)
        out[both, :2] = (1 - f) * a[both, :2] + f * b[both, :2]
        out[both, 2] = np.minimum(a[both, 2], b[both, 2])
        return out
    edge = before[-1] if before else after[0]
    if hold_ends is None or abs(t - index[edge]) <= hold_ends:
        return frames[edge]
    return None


def _fill_bound(frames, index, t, dropped=None):
    if not frames:
        return None
    ps = sorted(frames)
    before = [p for p in ps if index[p] <= t]
    after = [p for p in ps if index[p] >= t]
    if before and index[before[-1]] == t:
        return frames[before[-1]]
    pa = before[-1] if before else None
    pb = after[0] if after else None
    if pa is not None and pb is not None:
        most = POSE_CONTACT_FILL_MAX if dropped and any(pa < q < pb for q in dropped) else POSE_GAP_MAX
        if pb - pa - 1 <= most:
            return _fill(frames, index, t, None)
    if pa is not None and (pa == len(index) - 1 or t < index[pa + 1]):
        return frames[pa]
    if pb is not None and (pb == 0 or t > index[pb - 1]):
        return frames[pb]
    return None


def _ensure_aux_path():
    try:
        if importlib.util.find_spec("custom_controlnet_aux") is not None:
            return True
    except (ImportError, ValueError):
        pass
    if os.path.isdir(AUX_SRC) and AUX_SRC not in sys.path:
        sys.path.append(AUX_SRC)
        importlib.invalidate_caches()
    try:
        return importlib.util.find_spec("custom_controlnet_aux") is not None
    except (ImportError, ValueError):
        return False


def _draw_bodypose_port(canvas, keypoints, xinsr_stick_scaling=False):
    import cv2
    CH, CW, _ = canvas.shape
    stickwidth = 4
    max_side = max(CW, CH)
    stick_scale = (1 if max_side < 500 else min(2 + (max_side // 1000), 7)) if xinsr_stick_scaling else 1
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11],
               [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
              [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
              [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
              [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for (k1, k2), color in zip(limbSeq, colors):
        p1, p2 = keypoints[k1 - 1], keypoints[k2 - 1]
        if p1 is None or p2 is None:
            continue
        Y = np.array([p1.x, p2.x])
        X = np.array([p1.y, p2.y])
        mX, mY = np.mean(X), np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth * stick_scale),
                                   int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])
    for kp, color in zip(keypoints, colors):
        if kp is None:
            continue
        cv2.circle(canvas, (int(kp.x), int(kp.y)), 4, color, thickness=-1)
    return canvas


def _drawer():
    global _DRAW_FN
    if _DRAW_FN is None:
        try:
            if not _ensure_aux_path():
                raise ImportError("custom_controlnet_aux")
            from custom_controlnet_aux.dwpose.util import draw_bodypose
            _DRAW_FN = draw_bodypose
        except Exception:
            _DRAW_FN = _draw_bodypose_port
    return _DRAW_FN


def _draw_person(canvas, kp, thick):
    H, W = canvas.shape[:2]
    pts = []
    for i in range(18):
        if kp[i, 2] >= POSE_CONF:
            x = float(min(max(kp[i, 0], -W), 2 * W))
            y = float(min(max(kp[i, 1], -H), 2 * H))
            pts.append(_KP(x, y, float(kp[i, 2]), i))
        else:
            pts.append(None)
    seen = [k for k in pts if k is not None]
    if not seen or all(abs(k.x) <= 1 and abs(k.y) <= 1 for k in seen):
        return canvas
    return _drawer()(canvas, pts, xinsr_stick_scaling=bool(thick))


def render_skeletons(skeletons_per_frame, height, width, thick=False):
    F = len(skeletons_per_frame)
    hint = torch.zeros((F, int(height), int(width), 3), dtype=POSE_HINT_DTYPE)
    for t, people in enumerate(skeletons_per_frame):
        if not people:
            continue
        canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        for kp in people:
            if kp is not None:
                canvas = _draw_person(canvas, kp, thick)
        hint[t] = torch.from_numpy(canvas).to(POSE_HINT_DTYPE) / 255.0
    return hint


def torso_boxes(people_one_frame, identified):
    out = {}
    people_one_frame = list(people_one_frame or [])
    for name, i in (identified or {}).items():
        if i is None:
            continue
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if not 0 <= i < len(people_one_frame):
            continue
        box = _torso_box(_person(people_one_frame[i]))
        if box is not None:
            out[name] = box
    return out


def identify_by_boxes(people_one_frame, boxes, *, appearance=None, carry_appearance=None):
    kps = [_person(p) for p in (people_one_frame or [])]
    apps = _seq(appearance) if appearance is not None else None
    carried = carry_appearance if hasattr(carry_appearance, "get") else None
    out = {}
    for name, box in (boxes or {}).items():
        ious = sorted(((_iou(box, _torso_box(k)), i) for i, k in enumerate(kps)), reverse=True)
        if ious and ious[0][0] >= POSE_CARRY_IOU_MIN \
                and ious[0][0] - (ious[1][0] if len(ious) > 1 else 0.0) >= POSE_CARRY_IOU_LEAD:
            i = ious[0][1]
            want = _app_vec(carried.get(name)) if (carried is not None and apps is not None) else None
            if want is not None:
                d = appearance_distance(want, apps[i] if i < len(apps) else None)
                if d is None or d > POSE_APP_MATCH_MAX:
                    out[name] = None
                    continue
            out[name] = i
        else:
            out[name] = None
    taken = [v for v in out.values() if v is not None]
    if len(set(taken)) != len(taken):
        out = {k: None for k in out}
    return out


def _parse_detections(det, frame_count):
    pairs, seen_t = [], set()
    apps_in = _seq(det.get("appearance"))
    for j, (t, plist) in enumerate(zip(_seq(det.get("index")), _seq(det.get("people")))):
        try:
            t = int(t)
        except (TypeError, ValueError):
            continue
        if not (0 <= t < frame_count) or t in seen_t:
            continue
        seen_t.add(t)
        single = (torch.is_tensor(plist) or isinstance(plist, np.ndarray)) and plist.ndim == 2
        if single:
            plist = [plist]
        kps = [_person(p) for p in _seq(plist)]
        kps = [k if k is not None else np.zeros((18, 3)) for k in kps]
        alist = apps_in[j] if j < len(apps_in) else None
        if (torch.is_tensor(alist) or isinstance(alist, np.ndarray)) and alist.ndim == 1:
            alist = [alist]
        avs = _seq(alist)
        vecs = [_app_vec(avs[i]) if i < len(avs) else None for i in range(len(kps))]
        pairs.append((t, kps, vecs))
    pairs.sort(key=lambda x: x[0])
    index = [t for t, _k, _a in pairs]
    people = [k for _t, k, _a in pairs]
    app = [a for _t, _k, a in pairs]
    return index, people, (app if any(v is not None for a in app for v in a) else None)


def _hint_skeletons(detections, frame_count, height, width, bound, carry=None, mode="repair",
                    draw="everyone", cast_count=None, carry_appearance=None, latch_after=None):
    report = {"identified": {nm: None for nm in (bound or {})}, "identified_frame": None,
              "identified_first": {nm: None for nm in (bound or {})}, "broken": False,
              "broken_frames": [], "skipped": "", "boxes_last": {}, "appearance_last": {},
              "latched": {nm: None for nm in (bound or {})}, "notes": []}
    notes = report["notes"]

    def skip(why):
        report["skipped"] = why
        return None, report, False

    mode = mode if mode in MODES else "repair"
    if draw not in DRAWS:
        notes.append(f"unknown draw option {draw!r}; drew everyone")
        draw = "everyone"
    specs = {nm: _spec(v) for nm, v in (bound or {}).items()}
    if any(s["anchored"] for s in specs.values()):
        return skip("fastened to an object")
    specs = {nm: s for nm, s in specs.items() if s["arms"] or s["legs"]}
    if not specs:
        return skip("no held arm or leg position to draw")
    try:
        frame_count, height, width = int(frame_count), int(height), int(width)
    except (TypeError, ValueError):
        return skip("bad frame size")
    if frame_count < 1 or height < 1 or width < 1:
        return skip("bad frame size")
    if latch_after is not None:
        try:
            latch_after = int(latch_after)
        except (TypeError, ValueError):
            return skip("bad latch frame")
    latch_on = latch_after is not None and any(s["latch_limbs"] for s in specs.values())

    det = detections if hasattr(detections, "get") else {}
    index, people, app = _parse_detections(det, frame_count)
    if not index:
        return skip("no frames analysed")
    n = len(index)
    scene = _Scene(people, index, app)

    window = range(min(POSE_ID_FRAMES, n))
    count = int(round(float(np.median([len(people[p]) for p in window]))))
    if count == 0:
        return skip("nobody detected")
    if count < len(specs):
        return skip(f"found {count} people for {len(specs)} restrained")
    if cast_count is not None:
        try:
            if abs(count - int(cast_count)) > POSE_CAST_TOLERANCE:
                return skip(f"found {count} people where the plan has {int(cast_count)}")
        except (TypeError, ValueError):
            pass

    tracks = _build_tracks(people, index, scene)

    windows, id_specs = {}, dict(specs)
    if latch_on:
        crowd = count > 1
        for nm, s in specs.items():
            if not s["latch_limbs"]:
                continue
            wins, preset, apart = {}, 0, 0
            for ti, t in enumerate(tracks):
                ps = sorted(t["pos"])
                kps = [people[p][t["pos"][p]] for p in ps]
                idx = [index[p] for p in ps]
                g = _geometry(kps, idx)
                if crowd and _posed_before(kps, idx, g, s, latch_after):
                    preset += 1
                    continue
                i0 = _latch_start(kps, idx, ps, g, s, latch_after)
                if i0 is None:
                    continue
                if crowd and not _touched_before(ti, ps[i0] + POSE_LATCH_RUN, tracks, people):
                    apart += 1
                    continue
                wins[ti] = set(range(ps[i0], ps[i0] + POSE_ID_FRAMES))
            if preset:
                notes.append(f"{preset} person(s) held {nm}'s restraint pose from the start; "
                             f"not taken as {nm}")
            if apart:
                notes.append(f"{apart} person(s) settled into {nm}'s restraint pose with nobody "
                             f"near them; not taken as {nm}")
            if wins:
                windows[nm] = wins
                continue
            rest = {**s, **{limb: "" for limb in s["latch_limbs"]}, "latch_limbs": ()}
            if not (rest["arms"] or rest["legs"]):
                return skip("the restraint never closed in the first pass")
            id_specs[nm] = rest
    found, why = _identify(tracks, people, index, id_specs, carry, notes, scene=scene,
                           carry_app=carry_appearance if hasattr(carry_appearance, "get") else None,
                           windows=windows)
    if found is None:
        return skip(why)

    def gate_spec(s):
        if not latch_on or not s["latch_limbs"]:
            return lambda p: s
        early = {**s, **{limb: "" for limb in s["latch_limbs"]}}
        early = early if (early["arms"] or early["legs"]) else None
        return lambda p: s if index[p] >= latch_after else early

    spec_at = {nm: gate_spec(specs[nm]) for nm in found}
    taken = set(found.values())
    bound_pos, bound_tracks = {}, set()
    for nm in sorted(found):
        pos, used, unclear = _follow(found[nm], tracks, people, index, specs[nm],
                                     taken - {found[nm]}, scene, spec_at[nm])
        if unclear:
            return skip(f"could not tell who came back as {nm}")
        bound_pos[nm] = pos
        bound_tracks |= used
        taken |= used

    chains = {nm: {p: pi for p, (_t, pi) in bound_pos[nm].items()} for nm in found}
    rivals = [dict(t["pos"]) for ti, t in enumerate(tracks) if ti not in bound_tracks and t["pos"]]
    group = {nm: (specs[nm]["arms"], specs[nm]["legs"]) for nm in found}
    carried = {nm: _app_vec(carry_appearance.get(nm)) if hasattr(carry_appearance, "get") else None
               for nm in found}
    pairs_by_app, swapped, dropped = set(), set(), set()
    for nm in sorted(found):
        i = 0
        while i < len(rivals):
            why, by_app, extra = _settle_pair(scene, chains[nm], rivals[i], nm, carried[nm],
                                              spec_at[nm], swapped, dropped)
            if why:
                return skip(why)
            if by_app:
                pairs_by_app.add((nm, i))
            rivals.extend(r for r in extra if r)
            i += 1
        for o in sorted(found):
            if o > nm and group[o] != group[nm]:
                why, by_app, extra = _settle_pair(scene, chains[nm], chains[o], nm, carried[nm],
                                                  spec_at[nm], swapped, dropped)
                if why:
                    return skip(why)
                if by_app:
                    pairs_by_app.add((nm, o))
                rivals.extend(r for r in extra if r)
        if not chains[nm]:
            return skip(f"lost track of {nm} where two people crossed")

    def by_app(nm, key):
        return (nm, key) in pairs_by_app or (key, nm) in pairs_by_app

    def others_of(nm):
        out = [(i, Y) for i, Y in enumerate(rivals) if Y and not by_app(nm, i)]
        return out + [(o, chains[o]) for o in sorted(found)
                      if o != nm and group[o] != group[nm] and not by_app(nm, o)]

    for nm in sorted(found):
        for _key, Y in others_of(nm):
            for k in _exchanges(chains[nm], Y, people, index):
                chains[nm].pop(k, None)
                Y.pop(k, None)
                dropped.add(index[k])
        if not chains[nm]:
            return skip(f"lost track of {nm} where two people crossed")
    if swapped:
        notes.append(f"who is who after contact read from appearance: frames {_spans(swapped)}")
    if dropped:
        notes.append(f"two people on one spot, left to the frames around them: frames {_spans(dropped)}")
    for nm in sorted(found):
        A = _sightings(chains[nm], people, index)
        for _key, Y in others_of(nm):
            k = _crossing(A, _sightings(Y, people, index))
            if k is not None:
                return skip(f"lost track of {nm} where two people crossed (frame {index[k]})")
    for nm in sorted(found):
        miss = 1.0 - len(chains[nm]) / float(n)
        if miss > POSE_TRACK_MISS_MAX:
            return skip(f"lost track of {nm} in {miss * 100:.0f}% of frames")

    last = n - 1
    report["identified_frame"] = index[last]
    for nm in sorted(found):
        report["identified"][nm] = chains[nm].get(last)
        report["identified_first"][nm] = chains[nm].get(0)
        if report["identified"][nm] is None:
            notes.append(f"{nm} not seen in the last analysed frame; nothing to carry")
        v = scene.mean(chains[nm], count=POSE_APP_FRAMES, from_end=True)
        if v is not None:
            report["appearance_last"][nm] = v.astype(np.float32)
    report["boxes_last"] = torso_boxes(people[last], report["identified"])

    final = {}
    pass1 = {}
    broken_frames = set()
    any_fall = False
    for nm in sorted(found):
        spec = specs[nm]
        ps = sorted(chains[nm])
        kps1 = [people[p][chains[nm][p]] for p in ps]
        idx = [index[p] for p in ps]
        g1 = _geometry(kps1, idx)
        if g1 is None:
            return skip(f"torso not fully visible ({nm})")
        latch = None
        if latch_on and spec["latch_limbs"]:
            i0 = _latch_start(kps1, idx, ps, g1, spec, latch_after)
            report["latched"][nm] = idx[i0] if i0 is not None else None
            latch = {limb: report["latched"][nm] for limb in spec["latch_limbs"]}
            notes.append(f"{nm}: " + (f"restraint closed at frame {idx[i0]}" if i0 is not None
                                      else "restraint never closed; drawn as detected"))
        if any(_facing_needed(g1, i, spec) and _is_held(latch, "arms", idx[i]) for i in range(len(ps))):
            return skip("could not tell which way they face")
        broken, bf, reasons, held = _check(kps1, idx, g1, spec, latch=latch)
        if broken:
            report["broken"] = True
            broken_frames |= set(bf)
            notes.append(f"{nm}: {', '.join(sorted(reasons))}, frames {_spans(bf)}")
        else:
            notes.append(f"{nm}: held in {held * 100:.0f}% of frames")
        kps, settled = kps1, False
        if spec["fall"]:
            any_fall = True
            if POSE_FALL_SETTLE:
                kps, settled = _fall_settle(kps1, idx)
                if settled:
                    notes.append(f"{nm}: fall settled onto the floor line")
        g = _geometry(kps, idx) if settled else g1
        if g is None:
            g = g1
        blend = False
        if spec["arms"] and _held_from(latch, "arms") == 0:
            f0 = _frame_fit(kps1[0], g1, 0, spec)
            if f0 is not None and f0 > POSE_BLEND_FIT:
                blend = True
                notes.append(f"{nm}: opening frame off the restraint (fit {f0:.2f}); "
                             f"blended into it over {POSE_BLEND_FRAMES} frames")
        out = _rewrite(kps1, kps, idx, g, spec, blend, latch=latch)
        final[nm] = dict(zip(ps, out))
        pass1[nm] = dict(zip(ps, kps1))
    report["broken_frames"] = sorted(broken_frames)

    if latch_on and not report["broken"] and not any_fall and \
            all(report["latched"][nm] is None for nm in found if specs[nm]["latch_limbs"]):
        return skip("the restraint never closed in the first pass")
    if mode == "repair" and not report["broken"] and not any_fall:
        return None, report, False
    if mode == "falls" and not any_fall:
        return skip("no bound fall in this shot")

    others = {}
    if draw != "bound person only":
        T_of = {}
        for nm in final:
            ps = sorted(final[nm])
            g = _geometry([pass1[nm][p] for p in ps], [index[p] for p in ps])
            T_of[nm] = dict(zip(ps, g.T)) if g is not None else {}
        for ri, t in enumerate(rivals):
            seq = {}
            for p, pi in t.items():
                kp = people[p][pi].copy()
                for nm in final:
                    if p not in final[nm]:
                        continue
                    T = float(T_of[nm].get(p, 0.0) or 0.0)
                    if T <= 0:
                        continue
                    for bw in (RWRI, LWRI):
                        if not _vis(pass1[nm][p], bw):
                            continue
                        old = _xy(pass1[nm][p], bw)
                        if np.linalg.norm(final[nm][p][bw, :2] - old) <= POSE_CAPTOR_MOVED * T:
                            continue
                        for ow, oe in ((RWRI, RELB), (LWRI, LELB)):
                            if _vis(kp, ow) and np.linalg.norm(_xy(kp, ow) - old) < POSE_CAPTOR_NEAR * T:
                                kp[ow, 2] = 0.0
                                kp[oe, 2] = 0.0
                seq[p] = kp
            if seq:
                others[ri] = seq

    gaps = [index[i + 1] - index[i] for i in range(n - 1)] or [1]
    hold_other = max(gaps)
    dropped_pos = {p for p in range(n) if index[p] in dropped}
    frames_out = []
    for t in range(frame_count):
        people_t = []
        for nm in sorted(final):
            k = _fill_bound(final[nm], index, t, dropped_pos)
            if k is not None:
                people_t.append(k)
        for ri in sorted(others):
            k = _fill(others[ri], index, t, hold_other)
            if k is not None:
                people_t.append(k)
        frames_out.append(people_t)
    return frames_out, report, draw == "everyone, thick lines"


def build_hint(detections, frame_count, height, width, bound, carry=None, mode="repair",
               draw="everyone", *, cast_count=None, carry_appearance=None, latch_after=None):
    frames, report, thick = _hint_skeletons(detections, frame_count, height, width, bound,
                                            carry=carry, mode=mode, draw=draw,
                                            cast_count=cast_count,
                                            carry_appearance=carry_appearance,
                                            latch_after=latch_after)
    if frames is None:
        return None, report
    return render_skeletons(frames, int(height), int(width), thick=thick), report


def _dig(obj, path):
    try:
        for step in path:
            obj = obj[step] if isinstance(step, int) else getattr(obj, step)
        return obj
    except Exception:
        return None


def _as_int(x):
    if isinstance(x, bool) or x is None:
        return None
    try:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if torch.is_tensor(x) and x.numel() == 1:
            return int(x.item())
    except Exception:
        return None
    return None


def _aux_ckpts_dir():
    try:
        mod = sys.modules.get("custom_controlnet_aux.util")
        if mod is None and _ensure_aux_path():
            mod = importlib.import_module("custom_controlnet_aux.util")
        path = getattr(mod, "annotator_ckpts_path", None)
        if path:
            return str(path)
    except Exception:
        pass
    return os.environ.get("AUX_ANNOTATOR_CKPTS_PATH") or AUX_CKPTS_DEFAULT


def dwpose_paths():
    ck = _aux_ckpts_dir()
    return tuple(os.path.join(ck, repo, fn) for repo, fn in (DWPOSE_DET, DWPOSE_POSE))


def dwpose_status():
    if not _ensure_aux_path():
        return False, ("pose control off: comfyui_controlnet_aux (custom_controlnet_aux) cannot "
                       "be imported; install it under custom_nodes for the DWPose estimator")
    det, pose = dwpose_paths()
    missing = [p for p in (det, pose) if not os.path.isfile(p)]
    if missing:
        cmd = "; ".join(f"hf download {repo} {fn} --local-dir {os.path.dirname(p)}"
                        for (repo, fn), p in zip((DWPOSE_DET, DWPOSE_POSE), (det, pose))
                        if p in missing)
        return False, (f"pose control off: the DWPose files must be at {det} and {pose} "
                       f"(missing: {', '.join(os.path.basename(m) for m in missing)}). "
                       f"Download: {cmd}")
    return True, ""


def _fun_patch_on(model):
    dit = _dig(model, ("model_options",))
    try:
        dit = dit["transformer_options"]["patches_replace"]["dit"]
    except Exception:
        return False
    if not isinstance(dit, dict):
        return False
    for patch in dit.values():
        seen = 0
        while patch is not None and seen < 64:
            if type(patch).__name__ == "MiniMaxH3FunControlBlockPatch":
                return True
            patch = getattr(patch, "previous", None)
            seen += 1
    return False


def pose_status(model, pose_cn, strength, hyperflow_two_time_on):
    try:
        try:
            st = float(strength or 0.0)
        except (TypeError, ValueError):
            st = 0.0
        if pose_cn is None or not st > 0.0:
            return False, ""
        inner = _dig(pose_cn, ("model",))
        if inner is None or not (hasattr(inner, "init_stream") and hasattr(inner, "injection_layers")):
            return False, ("pose control off: what is wired to pose_controlnet is not a MiniMax H3 "
                           "Fun controlnet (load the H3 Fun controlnet with Load Model Patch)")
        notes = []
        base_w = _as_int(_dig(model, ("model", "diffusion_model", "blocks", 0, "adaln_proj",
                                      "linear", "in_features")))
        cn_w = _as_int(_dig(inner, ("control_blocks", 0, "adaln_proj", "linear", "in_features")))
        if base_w is not None and cn_w is not None:
            if base_w != cn_w:
                return False, (f"pose control off: the pose controlnet takes a {cn_w}-wide timestep "
                               f"embedding and this base model gives {base_w}; it is built for the "
                               f"8-wide hybrid b25-49 base")
        else:
            notes.append("could not read the adaln widths of the base and the pose controlnet to "
                         "compare them; it is built for the 8-wide hybrid b25-49 base, and on any "
                         "other the first controlled step fails")
        if hyperflow_two_time_on:
            return False, ("pose control off: Hyperflow two-time is on; it needs a full-form base "
                           "and the pose controlnet the 8-wide curve base, so they never run together")
        ok, note = dwpose_status()
        if not ok:
            return False, note
        if _fun_patch_on(model):
            return False, ("pose control off: a MiniMax H3 Fun control patch is already on the "
                           "incoming model (Apply MiniMax H3 Fun ControlNet upstream); upstream Fun "
                           "control replays one clip from frame 0 every shot. Remove that node to "
                           "use pose_controlnet")
        return True, "; ".join(notes)
    except Exception as e:
        return False, f"pose control off: the setup check failed ({type(e).__name__}: {e})"


@contextlib.contextmanager
def _hf_offline():
    old = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    consts, old_c = None, None
    try:
        import huggingface_hub.constants as consts
        old_c = getattr(consts, "HF_HUB_OFFLINE", None)
        consts.HF_HUB_OFFLINE = True
    except Exception:
        consts = None
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old
        if consts is not None:
            try:
                consts.HF_HUB_OFFLINE = old_c
            except Exception:
                pass


class PoseDetector:

    def __init__(self, device=None):
        self.device = device
        self._est = None
        self._on_device = False

    def _device(self):
        if self.device is not None:
            return torch.device(self.device)
        try:
            import comfy.model_management as mm
            return mm.get_torch_device()
        except Exception:
            return torch.device("cpu")

    def available(self):
        return dwpose_status()

    def _move(self, dev):
        for name in ("det", "pose"):
            m = getattr(self._est, name, None)
            if m is not None and hasattr(m, "to"):
                m.to(dev)

    def _drop(self):
        self._est = None
        self._on_device = False
        try:
            mod = sys.modules.get("custom_controlnet_aux.dwpose")
            if mod is not None and hasattr(mod, "Wholebody"):
                mod.global_cached_dwpose = mod.Wholebody()
        except Exception:
            pass

    def _load(self):
        if self._est is None:
            ok, note = dwpose_status()
            if not ok:
                raise RuntimeError(note)
            from custom_controlnet_aux.dwpose import DwposeDetector
            with _hf_offline():
                det = DwposeDetector.from_pretrained(
                    DWPOSE_POSE[0], DWPOSE_DET[0], det_filename=DWPOSE_DET[1],
                    pose_filename=DWPOSE_POSE[1], torchscript_device=self._device())
            self._est = det.dw_pose_estimation
            self._on_device = False
        if not self._on_device:
            try:
                self._move(self._device())
            except Exception:
                self._drop()
                return self._load()
            self._on_device = True
        return self._est

    def detect(self, frames, stride=2):
        est = self._load()
        F = int(frames.shape[0])
        stride = max(1, int(stride))
        index = list(range(0, F, stride))
        if F and index[-1] != F - 1:
            index.append(F - 1)
        people, appearance = [], []
        with torch.no_grad():
            for i in index:
                img = frames[i, ..., :3].detach().to(torch.float32).clamp(0, 1).mul(255.0) \
                    .round().to(torch.uint8).cpu().numpy()
                img = np.ascontiguousarray(img)
                with contextlib.redirect_stdout(io.StringIO()):
                    info = est(img)
                if info is None:
                    people.append([])
                    appearance.append([])
                    continue
                plist = [np.asarray(p[:18], dtype=np.float32).copy() for p in np.asarray(info)]
                people.append(plist)
                appearance.append([person_appearance(img, p) for p in plist])
        return {"index": index, "people": people, "appearance": appearance}

    def close(self):
        if self._est is not None:
            try:
                self._move(torch.device("cpu"))
            except Exception:
                self._drop()
        self._on_device = False
        _free_cuda()

    def release(self):
        self.close()
        self._drop()
        _free_cuda()


def encode_hint(vae, hint, latent_shape):
    if vae is None or hint is None:
        return None
    try:
        target = tuple(int(x) for x in latent_shape)
    except Exception:
        return None
    pix = hint[..., :3]
    try:
        lat = vae.encode(pix)
    except Exception as e:
        if not _is_oom(e):
            logging.warning("[H3-LongVideos] pose hint encode failed: %s: %s", type(e).__name__, e)
            return None
        _free_cuda()
        try:
            lat = vae.encode_tiled(pix)
        except Exception as e2:
            logging.warning("[H3-LongVideos] pose hint tiled encode failed: %s: %s",
                            type(e2).__name__, e2)
            return None
    if not torch.is_tensor(lat) or tuple(lat.shape) != target:
        return None
    return lat.detach().to(device="cpu", dtype=torch.float32)


def _preset_patch_class():
    global _PRESET_CLS
    if _PRESET_CLS is None:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch

        class PresetFunControlPatch(MiniMaxH3FunControlPatch):
            preset_latent = None
            preset_shape = None

            def cleanup(self):
                super().cleanup()
                self.control_latent = self.preset_latent
                self.control_latent_shape = self.preset_shape

        _PRESET_CLS = PresetFunControlPatch
    return _PRESET_CLS


def install_pose_control(model, pose_cn, vae, hint_latent, latent_shape, strength, s_start, s_end):
    cls = _preset_patch_class()
    shape = tuple(int(x) for x in latent_shape)
    patch = cls(pose_cn, vae, None, None, None, float(strength), float(s_start), float(s_end))
    patch.preset_latent = hint_latent.detach().to(torch.float32)
    patch.preset_shape = shape
    patch.control_latent = patch.preset_latent
    patch.control_latent_shape = shape
    m = model.clone()
    patch.register(m)
    return m


def pose_sigma_window(sched, pose_end):
    vals = sched.tolist() if hasattr(sched, "tolist") else list(sched)
    vals = [float(v) for v in vals]
    if not vals:
        return 1.0 + POSE_SIGMA_START_PAD, -1.0
    steps = len(vals) - 1
    s_start = vals[0] + POSE_SIGMA_START_PAD
    if steps < 1:
        return s_start, -1.0
    try:
        end = float(pose_end)
    except (TypeError, ValueError, RuntimeError):
        end = POSE_END_DEFAULT
    if not math.isfinite(end):
        end = POSE_END_DEFAULT
    k = int(math.ceil(end * steps - 1e-9))
    k = max(1, min(steps, k))
    if k >= steps:
        return s_start, -1.0
    return s_start, (vals[k - 1] + vals[k]) / 2.0
