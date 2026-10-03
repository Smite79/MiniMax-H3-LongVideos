"""Tests for pose control (pose_control.py).

The reports behind it: a character cuffed behind the back catches a fall with both
hands, reaches for a door, spreads the arms; tied ankles walk apart. Pass 2 is held
to a skeleton in which the restrained person's limbs are rewritten into the held
shape, so the geometry that builds that skeleton is the specification here: the
torso frame, the arm templates in every view, the legs, the repair check, who is
identified as restrained (and that an unclear case is skipped, never guessed), the
tracking, the frame filling and the drawing.

Everything runs on synthetic keypoints on the CPU. No model is loaded, except one
optional check that the DWPose files load and run on a synthetic image when they are
installed.

Run: python test_pose.py
"""

import io
import math
import os
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU only, whatever the machine has
os.environ["HF_HUB_OFFLINE"] = "1"           # never a download from a test

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch

import pose_control as P

_fails = []

_COMFY_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def check(label, ok, extra=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else f"   {extra}"))
    if not ok:
        _fails.append(label)


# --------------------------------------------------------------------------------------
# Synthetic skeletons
# --------------------------------------------------------------------------------------

def body(neck=(300.0, 120.0), T=100.0, phi=90.0, width=0.78, view="front", fwd=1.0,
         arms=None, legs=None):
    """A COCO-18 (18, 3) skeleton. Local coordinates (a, b) are in T: a runs down the
    spine from the neck, b along p = u turned 90 degrees (image left for an upright
    body). phi is the angle of u (neck to hip) in the image: 90 is upright, 0 lies with
    the head on the left. view: front | back | profile; fwd: the face side along p in
    profile. arms / legs: {joint: (a, b) or None (hidden)} overriding the defaults
    (arms hanging at the sides, legs standing)."""
    u = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
    p = np.array([-u[1], u[0]])
    N = np.array(neck, dtype=float)
    kp = np.zeros((18, 3))

    def put(i, ab, c=1.0):
        if ab is None:
            kp[i] = 0.0
            return
        a, b = ab
        kp[i, :2] = N + a * T * u + b * T * p
        kp[i, 2] = c

    hs = width / 2.0
    sg = -1.0 if view == "back" else 1.0          # from behind, right and left swap sides
    hh = max(hs * 0.55, 0.03)
    put(P.NECK, (0.0, 0.0))
    put(P.RSHO, (0.0, sg * hs))
    put(P.LSHO, (0.0, -sg * hs))
    put(P.RHIP, (1.0, sg * hh))
    put(P.LHIP, (1.0, -sg * hh))
    if view == "front":
        put(P.NOSE, (-0.28, 0.0))
        put(P.REYE, (-0.33, 0.06))
        put(P.LEYE, (-0.33, -0.06))
        put(P.REAR, (-0.30, 0.12))
        put(P.LEAR, (-0.30, -0.12))
    elif view == "back":
        put(P.REAR, (-0.30, -0.12))
        put(P.LEAR, (-0.30, 0.12))
    else:
        put(P.NOSE, (-0.28, 0.15 * fwd))
        put(P.REYE, (-0.33, 0.10 * fwd))
        put(P.REAR, (-0.30, -0.04 * fwd))
    default_arms = {P.RELB: (0.55, sg * (hs + 0.03)), P.LELB: (0.55, -sg * (hs + 0.03)),
                    P.RWRI: (1.0, sg * (hs + 0.05)), P.LWRI: (1.0, -sg * (hs + 0.05))}
    default_legs = {P.RKNE: (1.85, sg * hh), P.LKNE: (1.85, -sg * hh),
                    P.RANK: (2.7, sg * hh), P.LANK: (2.7, -sg * hh)}
    for j, ab in {**default_arms, **(arms or {})}.items():
        put(j, ab)
    for j, ab in {**default_legs, **(legs or {})}.items():
        put(j, ab)
    return kp


def img(kp, j):
    return kp[j, :2].copy()


def geom(kp, n=6):
    seq = [kp.copy() for _ in range(n)]
    return P._geometry(seq, list(range(0, 2 * n, 2))), seq


def rewrite(seq, spec, blend=False):
    idx = list(range(0, 2 * len(seq), 2))
    g = P._geometry(seq, idx)
    return P._rewrite(seq, seq, idx, g, P._spec(spec), blend), g


BEHIND_FRONT = {P.RELB: (0.55, 1.05 * 0.39), P.LELB: (0.55, -1.05 * 0.39),
                P.RWRI: None, P.LWRI: None}
ARMS_OUT = {P.RELB: (0.0, 0.8), P.LELB: (0.0, -0.8), P.RWRI: (0.0, 1.3), P.LWRI: (0.0, -1.3)}


def dets(frames_people, stride=2):
    return {"index": [i * stride for i in range(len(frames_people))],
            "people": [list(fp) for fp in frames_people]}


def close(a, b, tol=1.0):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))) <= tol


# --------------------------------------------------------------------------------------
# Painted figures: the pixels the appearance vectors are read from
# --------------------------------------------------------------------------------------

SKIN = (225, 180, 150)
DRESS = {                         # top, trousers, hair
    "B": ((205, 60, 40), (40, 50, 110), (70, 40, 20)),
    "F": ((40, 150, 60), (125, 125, 125), (20, 20, 20)),
    "G": ((50, 90, 210), (25, 25, 25), (210, 180, 90)),
    "twin": ((205, 60, 40), (40, 50, 110), (70, 40, 20)),     # dressed as B
}


def _seg(img, a, b, half, colour):
    """Paint the segment a-b, `half` pixels either side of it (a disc when a == b)."""
    H, W = img.shape[:2]
    a, b = np.asarray(a, float), np.asarray(b, float)
    x0, y0 = np.floor(np.minimum(a, b) - half).astype(int)
    x1, y1 = np.ceil(np.maximum(a, b) + half).astype(int) + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    p = np.stack([xs, ys], axis=-1).astype(float)
    d = b - a
    L2 = max(float(d @ d), 1e-9)
    t = np.clip(((p - a) @ d) / L2, 0.0, 1.0)
    m = np.linalg.norm(p - (a + t[..., None] * d), axis=-1) <= half
    img[y0:y1, x0:x1][m] = colour


def paint(people, roles, H, W, light=1.0, dress=None):
    """An (H, W, 3) uint8 frame: each skeleton painted as a simple figure in its role's
    clothes, in list order (the last is in front). Numpy only."""
    dress = dress or DRESS
    yy = np.linspace(0, 1, H)[:, None, None]
    img = (85 + 50 * yy + np.zeros((H, W, 3))).astype(float)
    for kp, role in zip(people, roles):
        top, trousers, hair = dress[role]
        T = P._torso_len(kp) or 100.0
        xy = lambda j: kp[j, :2]
        for h, k, a in ((P.RHIP, P.RKNE, P.RANK), (P.LHIP, P.LKNE, P.LANK)):
            _seg(img, xy(h), xy(k), 0.16 * T, trousers)
            _seg(img, xy(k), xy(a), 0.12 * T, trousers)
        neck, mid = P._neck(kp), P._midhip(kp)
        _seg(img, neck, mid, 0.26 * T, top)
        _seg(img, xy(P.RSHO), xy(P.LSHO), 0.14 * T, top)
        for s, e, w in ((P.RSHO, P.RELB, P.RWRI), (P.LSHO, P.LELB, P.LWRI)):
            if kp[e, 2] >= P.POSE_CONF:
                _seg(img, xy(s), xy(e), 0.10 * T, top)
                if kp[w, 2] >= P.POSE_CONF:
                    _seg(img, xy(e), xy(w), 0.08 * T, SKIN)
        heads = [xy(j) for j in (P.NOSE, P.REYE, P.LEYE, P.REAR, P.LEAR) if kp[j, 2] >= P.POSE_CONF]
        hc = np.mean(heads, axis=0) if heads else neck - np.array([0.0, 0.3 * T])
        _seg(img, hc, hc, 0.22 * T, hair)
    rng = np.random.default_rng(int(light * 1000))
    return np.clip(img * light + rng.normal(0, 3.0, img.shape), 0, 255).astype(np.uint8)


def with_appearance(frames, roles, H, W, stride=2, dress=None, light=None, depth=None):
    """dets() of the frames plus the appearance vectors read from them painted: in the
    order of depth[role] (smallest first, so behind), else in list order (the last in
    front). light(i) scales frame i's brightness."""
    d = dets(frames, stride)
    apps = []
    for i, (fr, rl) in enumerate(zip(frames, roles)):
        order = sorted(range(len(fr)), key=lambda j: (depth or {}).get(rl[j], j))
        img = paint([fr[j] for j in order], [rl[j] for j in order], H, W,
                    light=light(i) if light else 1.0, dress=dress)
        apps.append([P.person_appearance(img, k) for k in fr])
    d["appearance"] = apps
    return d


# --------------------------------------------------------------------------------------
# Torso frame
# --------------------------------------------------------------------------------------

def test_torso_frame():
    """The frame every template is built in: N at the neck, u down the spine, T its
    length, the view weight from the shoulder width, and which way the person faces."""
    print("\n=== torso frame ===")
    g, _ = geom(body())
    check("upright front: T is the neck-to-hip length", abs(g.T[0] - 100.0) < 1e-6, g.T[0])
    check("upright front: u points down the image", close(g.u[0], (0, 1), 1e-6), g.u[0])
    check("upright front: square to camera (w = 1)", abs(g.w[0] - 1.0) < 1e-6, g.w[0])
    check("upright front: facing the camera", bool(g.front[0]))
    check("right shoulder offset is +0.39 T, left -0.39 T",
          abs(g.sR[0] - 39.0) < 1e-6 and abs(g.sL[0] + 39.0) < 1e-6, (g.sR[0], g.sL[0]))
    g, _ = geom(body(view="back"))
    check("back view: not facing the camera", not bool(g.front[0]))
    check("back view: the right shoulder offset changes sign", g.sR[0] < 0, g.sR[0])
    g, seq = geom(body(view="profile", width=0.1, fwd=1.0))
    check("profile: view weight 0", abs(g.w[0]) < 1e-6, g.w[0])
    nose_side = float((img(seq[0], P.NOSE) - g.N[0]) @ g.p[0])
    check("profile: the back is the side away from the nose", g.back[0] * nose_side < 0,
          (g.back[0], nose_side))
    g, _ = geom(body(width=0.425))
    check("three-quarter (r = 0.425): view weight 0.5", abs(g.w[0] - 0.5) < 1e-6, g.w[0])
    g, _ = geom(body(phi=0.0, view="profile", width=0.1, fwd=1.0))
    check("lying (head left): u runs along the image x axis", close(g.u[0], (1, 0), 1e-6), g.u[0])
    # A missing hip pair takes the last good frame's spine
    seq = [body() for _ in range(4)]
    seq[2][P.RHIP, 2] = 0.0
    seq[2][P.LHIP, 2] = 0.0
    g = P._geometry(seq, [0, 2, 4, 6])
    check("hips missing in one frame: the last good spine is held", abs(g.T[2] - 100.0) < 1e-6
          and close(g.u[2], (0, 1), 1e-6), (g.T[2], g.u[2]))
    seq = [body() for _ in range(3)]
    seq[0][P.RHIP, 2] = 0.0
    g = P._geometry(seq, [0, 2, 4])
    check("one hip missing: the other one is the mid-hip", g is not None and g.T[0] > 99.0)
    check("no torso at all: no frame", P._geometry([np.zeros((18, 3))], [0]) is None)


# --------------------------------------------------------------------------------------
# Arm templates
# --------------------------------------------------------------------------------------

def test_arm_templates_front_and_back():
    """Square to camera, each arm position lands where the design's table says: in
    torso lengths down the spine and in shoulder offsets out to the side."""
    print("\n=== arm templates: front and back ===")
    kp = body(neck=(500.0, 300.0), T=200.0)
    out, g = rewrite([kp] * 6, {"arms": "behind the back"})
    o = out[0]
    # N=(500,300), T=200, u=(0,1), p=(-1,0), sR=+78 px: elbow R at (500-1.05*78, 410)
    check("behind the back, front: right elbow at (0.55, 1.05 s)",
          close(img(o, P.RELB), (500 - 1.05 * 78, 410)), img(o, P.RELB))
    check("behind the back, front: left elbow mirrored",
          close(img(o, P.LELB), (500 + 1.05 * 78, 410)), img(o, P.LELB))
    check("behind the back, front: wrists left out",
          o[P.RWRI, 2] < P.POSE_CONF and o[P.LWRI, 2] < P.POSE_CONF, o[[P.RWRI, P.LWRI], 2])
    check("the torso itself is not touched",
          np.allclose(o[[P.NECK, P.RSHO, P.LSHO, P.RHIP, P.LHIP]], kp[[P.NECK, P.RSHO, P.LSHO, P.RHIP, P.LHIP]]))
    out, _ = rewrite([body(neck=(500.0, 300.0), T=200.0, view="back")] * 6, {"arms": "behind the back"})
    o = out[0]
    check("behind the back, from behind: wrists drawn",
          o[P.RWRI, 2] >= P.POSE_CONF and o[P.LWRI, 2] >= P.POSE_CONF)
    check("behind the back, from behind: wrists on the spine at 0.90 T",
          close(img(o, P.RWRI), (500 + 0.06 * 78, 480)) and close(img(o, P.LWRI), (500 - 0.06 * 78, 480)),
          (img(o, P.RWRI), img(o, P.LWRI)))
    check("from behind the right elbow stays on the right shoulder's side (image right)",
          o[P.RELB, 0] > 500 + 78, img(o, P.RELB))
    out, _ = rewrite([kp] * 6, {"arms": "in front of the body"})
    o = out[0]
    check("in front, front: elbows at (0.55, 0.95 s)",
          close(img(o, P.RELB), (500 - 0.95 * 78, 410)), img(o, P.RELB))
    check("in front, front: wrists drawn together at the belly (0.95, 0.10 s)",
          o[P.RWRI, 2] >= P.POSE_CONF and close(img(o, P.RWRI), (500 - 7.8, 490))
          and close(img(o, P.LWRI), (500 + 7.8, 490)), (img(o, P.RWRI), img(o, P.LWRI)))
    out, _ = rewrite([kp] * 6, {"arms": "at the waist"})
    o = out[0]
    check("at the waist: elbows at (0.50, 1.05 s), wrists at (0.85, 0.55 s)",
          close(img(o, P.RELB), (500 - 1.05 * 78, 400)) and close(img(o, P.RWRI), (500 - 0.55 * 78, 470)),
          (img(o, P.RELB), img(o, P.RWRI)))
    out, _ = rewrite([kp] * 6, {"arms": "above the head"})
    o = out[0]
    check("above the head: elbows at (-0.45, 1.10 s)",
          close(img(o, P.RELB), (500 - 1.10 * 78, 210)), img(o, P.RELB))
    check("above the head: wrists over the crown (-0.95, 0.08 s)",
          close(img(o, P.RWRI), (500 - 0.08 * 78, 110)) and o[P.RWRI, 2] >= P.POSE_CONF, img(o, P.RWRI))


def test_arm_templates_profile_and_blend():
    """In profile the arms go to the back -- the side away from the face -- and a
    three-quarter view is the blend of the two tables."""
    print("\n=== arm templates: profile and blend ===")
    for fwd, label in ((1.0, "facing image left"), (-1.0, "facing image right")):
        kp = body(neck=(500.0, 300.0), T=200.0, view="profile", width=0.1, fwd=fwd)
        out, g = rewrite([kp] * 6, {"arms": "behind the back"})
        o = out[0]
        nose_dx = kp[P.NOSE, 0] - 500
        check(f"profile {label}: elbows behind (opposite the nose)",
              (o[P.RELB, 0] - 500) * nose_dx < 0 and abs(abs(o[P.RELB, 0] - 500) - 56) < 1.0,
              (img(o, P.RELB), nose_dx))
        check(f"profile {label}: wrists drawn at (0.92, 0.22 back)",
              o[P.RWRI, 2] >= P.POSE_CONF and abs(o[P.RWRI, 1] - 484) < 1.0
              and (o[P.RWRI, 0] - 500) * nose_dx < 0, img(o, P.RWRI))
    kp = body(neck=(500.0, 300.0), T=200.0, view="profile", width=0.1, fwd=1.0)
    out, _ = rewrite([kp] * 6, {"arms": "in front of the body"})
    check("profile, in front: wrists forward (on the nose side)",
          (out[0][P.RWRI, 0] - 500) * (kp[P.NOSE, 0] - 500) > 0, img(out[0], P.RWRI))
    out, _ = rewrite([kp] * 6, {"arms": "at the waist"})
    check("profile, at the waist: wrists forward at 0.85 T",
          (out[0][P.RWRI, 0] - 500) * (kp[P.NOSE, 0] - 500) > 0 and abs(out[0][P.RWRI, 1] - 470) < 1.0,
          img(out[0], P.RWRI))
    out, _ = rewrite([kp] * 6, {"arms": "above the head"})
    check("profile, above the head: wrists above the neck",
          out[0][P.RWRI, 1] < 300 - 0.9 * 200, img(out[0], P.RWRI))
    # three-quarter: halfway between the two tables
    kp = body(neck=(500.0, 300.0), T=200.0, width=0.425, view="front")
    out, g = rewrite([kp] * 6, {"arms": "behind the back"})
    s = g.sR[0]
    bf = 1.05 * s
    bp = 0.28 * 200 * g.back[0]
    want = np.array([500.0, 300.0]) + 0.55 * 200 * np.array([0, 1]) + (0.5 * bf + 0.5 * bp) * g.p[0]
    check("three-quarter: the elbow is the w-blend of the front and profile points",
          close(img(out[0], P.RELB), want, 0.5), (img(out[0], P.RELB), want))


def test_arm_templates_postures():
    """Kneeling, lying face down from the side and from overhead need no special case:
    the frame is the torso's."""
    print("\n=== arm templates: kneeling and lying ===")
    stand = body(neck=(500.0, 300.0), T=200.0)
    kneel = body(neck=(500.0, 300.0), T=200.0,
                 legs={P.RKNE: (1.85, 0.21), P.LKNE: (1.85, -0.21), P.RANK: (1.95, 0.21), P.LANK: (1.95, -0.21)})
    a, _ = rewrite([stand] * 6, {"arms": "behind the back"})
    b, _ = rewrite([kneel] * 6, {"arms": "behind the back"})
    check("kneeling: the arms sit where they do standing",
          np.allclose(a[0][[P.RELB, P.LELB, P.RWRI, P.LWRI]], b[0][[P.RELB, P.LELB, P.RWRI, P.LWRI]]))
    # Face down seen from the side: head on the left, torso along x, nose to the floor.
    down = body(neck=(300.0, 500.0), T=200.0, phi=0.0, view="profile", width=0.1, fwd=1.0)
    check("(the synthetic face-down body has its nose toward the floor)", down[P.NOSE, 1] > 500)
    out, g = rewrite([down] * 6, {"arms": "behind the back"})
    o = out[0]
    check("face down, side: 'back' points up", float(g.back[0] * g.p[0][1]) < 0, (g.back[0], g.p[0]))
    check("face down, side: elbows ride on top of the back",
          o[P.RELB, 1] < 500 - 50 and abs(o[P.RELB, 0] - (300 + 0.55 * 200)) < 1.0, img(o, P.RELB))
    check("face down, side: wrists drawn on the back near the hips",
          o[P.RWRI, 2] >= P.POSE_CONF and o[P.RWRI, 1] < 500 and abs(o[P.RWRI, 0] - 484) < 1.0,
          img(o, P.RWRI))
    # Face down from overhead reads as a back view.
    over = body(neck=(300.0, 500.0), T=200.0, phi=0.0, view="back")
    out, g = rewrite([over] * 6, {"arms": "behind the back"})
    o = out[0]
    check("face down, overhead: read as a back view", not bool(g.front[0]))
    check("face down, overhead: wrists drawn together on the spine",
          o[P.RWRI, 2] >= P.POSE_CONF and abs(o[P.RWRI, 1] - 500) < 10 and abs(o[P.RWRI, 0] - 480) < 1.0,
          img(o, P.RWRI))
    # Face up, cuffed behind: a front view with the wrists left out.
    up = body(neck=(300.0, 500.0), T=200.0, phi=0.0, view="front")
    out, _ = rewrite([up] * 6, {"arms": "behind the back"})
    check("face up, behind the back: wrists left out", out[0][P.RWRI, 2] < P.POSE_CONF)


# --------------------------------------------------------------------------------------
# Legs
# --------------------------------------------------------------------------------------

def test_legs():
    print("\n=== legs ===")
    apart = {P.RKNE: (1.85, 0.3), P.LKNE: (1.85, -0.3), P.RANK: (2.7, 0.4), P.LANK: (2.7, -0.4)}
    kp = body(neck=(500.0, 300.0), T=200.0, legs=apart)
    out, _ = rewrite([kp] * 6, {"legs": "ankles together", "ankle_gap": 0.12})
    o = out[0]
    sep = np.linalg.norm(img(o, P.RANK) - img(o, P.LANK))
    check("ankles together: separation limited to ankle_gap x T", abs(sep - 0.12 * 200) < 1e-6, sep)
    check("ankles together: the midpoint stays",
          close((img(o, P.RANK) + img(o, P.LANK)) / 2, (img(kp, P.RANK) + img(kp, P.LANK)) / 2, 1e-6))
    corr = img(o, P.RANK) - img(kp, P.RANK)
    check("ankles together: each knee moves half its ankle's correction",
          close(img(o, P.RKNE) - img(kp, P.RKNE), corr / 2, 1e-6))
    near = body(neck=(500.0, 300.0), T=200.0)       # standing: ankles 0.43 T apart
    out, _ = rewrite([near] * 6, {"legs": "ankles together", "ankle_gap": 0.6})
    check("ankles already within a chain's gap are left alone",
          np.allclose(out[0][[P.RANK, P.LANK, P.RKNE, P.LKNE]], near[[P.RANK, P.LANK, P.RKNE, P.LKNE]]))
    hidden = body(neck=(500.0, 300.0), T=200.0, legs={**apart, P.LANK: None})
    out, _ = rewrite([hidden] * 6, {"legs": "ankles together"})
    check("one ankle hidden: nothing to hold together, left as it is",
          np.allclose(out[0][P.RANK], hidden[P.RANK]))
    for legs, want in (({P.RANK: (2.7, 0.1), P.LANK: (2.7, -0.1)}, 0.8),
                       ({P.RANK: (2.7, 1.0), P.LANK: (2.7, -1.0)}, 1.2),
                       ({P.RANK: (2.7, 0.5), P.LANK: (2.7, -0.5)}, 1.0)):
        kp = body(neck=(500.0, 300.0), T=200.0, legs=legs)
        out, _ = rewrite([kp] * 6, {"legs": "held apart"})
        sep = np.linalg.norm(img(out[0], P.RANK) - img(out[0], P.LANK)) / 200
        check(f"held apart: pass-1 {2 * legs[P.RANK][1]:.1f} T becomes {want:.1f} T", abs(sep - want) < 1e-6, sep)
    # Hogtie, face down from the side
    down = body(neck=(300.0, 500.0), T=200.0, phi=0.0, view="profile", width=0.1, fwd=1.0,
                arms=ARMS_OUT)
    out, g = rewrite([down] * 6, {"arms": "behind the back", "legs": "ankles to the wrists"})
    o = out[0]
    want = g.N[0] + 0.92 * 200 * g.u[0] + (0.22 + 0.15) * 200 * g.back[0] * g.p[0]
    check("hogtie: ankles at the profile wrist point plus 0.15 T toward the back",
          close(img(o, P.RANK), want, 1e-6) and close(img(o, P.LANK), want, 1e-6), (img(o, P.RANK), want))
    check("hogtie: the ankles are above the back (face down, side view)", o[P.RANK, 1] < 500)
    l1 = np.linalg.norm(img(o, P.RKNE) - img(down, P.RHIP))
    l2 = np.linalg.norm(img(o, P.RKNE) - img(o, P.RANK))
    check("hogtie: thigh and shin keep pass 1's lengths",
          abs(l1 - 0.85 * 200) < 1.0 and abs(l2 - 0.85 * 200) < 1.0, (l1, l2))
    check("hogtie: the knee is past the hips, not toward the head",
          float((img(o, P.RKNE) - img(down, P.RHIP)) @ g.u[0]) > 0, img(o, P.RKNE))
    out, _ = rewrite([down] * 6, {"legs": "ankles to the wrists"})
    check("hogtie with no arm position given: the arms go behind the back too",
          P._spec({"legs": "ankles to the wrists"})["arms"] == "behind the back"
          and out[0][P.RELB, 1] < 500)
    out, _ = rewrite([kp] * 6, {"legs": "ankles to the neck"})
    check("'ankles to the neck' leaves the legs as pass 1 drew them",
          np.allclose(out[0], kp))


# --------------------------------------------------------------------------------------
# Repair check
# --------------------------------------------------------------------------------------

def _run_check(base, bad, frames_bad, spec, n=24):
    seq = [bad.copy() if i in frames_bad else base.copy() for i in range(n)]
    idx = list(range(0, 2 * n, 2))
    g = P._geometry(seq, idx)
    return P._check(seq, idx, g, P._spec(spec))


def test_repair_check():
    """Pass 1 is kept when the limbs hold; a reach, a spread, a catch or a step breaks it.
    A hidden wrist is the normal look of hands cuffed behind the back -- never a break."""
    print("\n=== repair check ===")
    spec = {"arms": "behind the back"}
    ok = body(arms=BEHIND_FRONT)
    broken, frames, why, held = _run_check(ok, ok, set(), spec)
    check("a correct pose is not broken", not broken and held == 1.0, (broken, why))
    hidden = body(arms={P.RELB: None, P.LELB: None, P.RWRI: None, P.LWRI: None})
    broken, _f, why, _h = _run_check(hidden, hidden, set(), spec)
    check("hidden wrists (and elbows) are not a break", not broken, why)
    low = body(arms={**BEHIND_FRONT, P.RWRI: (0.2, 0.3)})
    low[P.RWRI, 2] = 0.2
    broken, _f, why, _h = _run_check(ok, low, set(range(5, 12)), spec)
    check("a low-confidence raised wrist is not a break", not broken, why)
    raised = body(arms={P.RELB: (-0.2, 0.6), P.LELB: (-0.2, -0.6), P.RWRI: (-0.6, 0.5), P.LWRI: (-0.6, -0.5)})
    broken, frames, why, _h = _run_check(ok, raised, {8, 9, 10}, spec)
    check("raised hands for 3 frames break it", broken and "hands raised" in why, why)
    check("...and the breaking frames are reported as output frames", frames == [16, 18, 20], frames)
    broken, _f, why, _h = _run_check(ok, body(arms=ARMS_OUT), {4, 5, 6}, spec)
    check("hands out to the sides break it", broken and "hands out to the sides" in why, why)
    broken, _f, why, _h = _run_check(ok, raised, {3, 15}, spec, n=40)
    check("two scattered frames out of 40 do not (no run of 3, under 8%)", not broken, why)
    broken, _f, why, _h = _run_check(ok, raised, {3, 9, 15, 21}, spec, n=40)
    check("four scattered frames out of 40 do (10%)", broken, why)
    front = {"arms": "in front of the body"}
    held_front = body(arms={P.RELB: (0.55, 0.37), P.LELB: (0.55, -0.37),
                            P.RWRI: (0.95, 0.04), P.LWRI: (0.95, -0.04)})
    apart = body(arms={P.RELB: (0.55, 0.37), P.LELB: (0.55, -0.37),
                       P.RWRI: (0.95, 0.33), P.LWRI: (0.95, -0.33)})
    broken, _f, why, _h = _run_check(held_front, held_front, set(), front)
    check("cuffed in front, held: not broken", not broken, why)
    broken, _f, why, _h = _run_check(held_front, apart, {2, 3, 4}, front)
    check("cuffed in front, wrists apart: broken", broken and "wrists apart" in why, why)
    legs = {"legs": "ankles together", "ankle_gap": 0.12}
    tied = body(legs={P.RANK: (2.7, 0.04), P.LANK: (2.7, -0.04)})
    stepping = body(legs={P.RANK: (2.7, 0.4), P.LANK: (2.7, -0.4)})
    broken, _f, why, _h = _run_check(tied, tied, set(), legs)
    check("ankles held together: not broken", not broken, why)
    broken, _f, why, _h = _run_check(tied, stepping, {10, 11, 12}, legs)
    check("ankles apart: broken", broken and "ankles apart" in why, why)
    prof = body(view="profile", width=0.1,
                arms={P.RELB: (0.55, -0.28), P.LELB: (0.55, -0.28), P.RWRI: (0.92, -0.22), P.LWRI: (0.92, -0.22)})
    broken, _f, why, _h = _run_check(prof, prof, set(), spec)
    check("profile, arms behind the back: not broken (no side rule at w = 0)", not broken, why)


# --------------------------------------------------------------------------------------
# Identification and tracking
# --------------------------------------------------------------------------------------

BOUND = {"Mara": {"arms": "behind the back", "legs": "", "ankle_gap": 0.12, "anchored": False, "fall": False}}


def two_people(n=12, bound_arms=BEHIND_FRONT, free_arms=ARMS_OUT, order=(0, 1), drift=0.0):
    frames = []
    for i in range(n):
        a = body(neck=(180.0 + drift * i, 100.0), arms=bound_arms)
        b = body(neck=(440.0, 100.0), arms=free_arms)
        pair = [a, b]
        frames.append([pair[order[0]], pair[order[1]]])
    return frames


def test_identification():
    """The bound person is the one whose arms fit their own template; an unclear case
    is skipped with the fits, because a wrong pick holds the captor's arms instead."""
    print("\n=== identification ===")
    for order in ((0, 1), (1, 0)):
        hint, rep = P.build_hint(dets(two_people(order=order)), 23, 300, 600, BOUND, mode="every")
        want = order.index(0)
        check(f"wrists behind vs arms out (bound listed {want + 1}{'st' if want == 0 else 'nd'}): "
              f"the bound person is found", rep["identified"].get("Mara") == want and hint is not None,
              (rep["identified"], rep["skipped"]))
    # Arms hanging at the sides, both wrists seen about shoulder-width apart, are free:
    # they do not fit "behind the back", so the bound person is found without a carry.
    hint, rep = P.build_hint(dets(two_people(free_arms=None)), 23, 300, 600, BOUND, mode="every")
    check("wrists behind vs arms hanging at the sides: hanging arms do not fit, bound person found",
          rep["identified"].get("Mara") == 0 and hint is not None, (rep["identified"], rep["skipped"]))
    carry = {"Mara": P._torso_box(body(neck=(180.0, 100.0)))}
    hint, rep = P.build_hint(dets(two_people(free_arms=None)), 23, 300, 600, BOUND, carry=carry, mode="every")
    check("...and found with the carried identity", rep["identified"].get("Mara") == 0 and hint is not None,
          (rep["identified"], rep["skipped"]))
    hint, rep = P.build_hint(dets(two_people(free_arms=BEHIND_FRONT)), 23, 300, 600, BOUND, mode="every")
    check("both with arms behind the back: skipped, not guessed",
          hint is None and rep["skipped"].startswith("could not tell who is bound")
          and rep["identified"]["Mara"] is None, rep["skipped"])
    carry = {"Mara": P._torso_box(body(neck=(440.0, 100.0)))}
    hint, rep = P.build_hint(dets(two_people(free_arms=BEHIND_FRONT)), 23, 300, 600, BOUND,
                             carry=carry, mode="every")
    check("...but the carried torso box from the cut settles it",
          rep["identified"].get("Mara") == 1 and hint is not None, (rep["identified"], rep["skipped"]))
    check("...and says so", any("across the cut" in n for n in rep["notes"]), rep["notes"])
    carry = {"Mara": [0.0, 0.0, 5.0, 5.0]}
    hint, rep = P.build_hint(dets(two_people()), 23, 300, 600, BOUND, carry=carry, mode="every")
    check("a carried box that matches nobody falls back to the template fit",
          rep["identified"].get("Mara") == 0, (rep["identified"], rep["skipped"]))
    hint, rep = P.build_hint(dets([[body(arms=BEHIND_FRONT)]] * 8), 15, 300, 600,
                             {"Mara": BOUND["Mara"], "Ines": BOUND["Mara"]}, mode="every")
    check("fewer people detected than are bound: skipped", hint is None and "found 1 people" in rep["skipped"],
          rep["skipped"])
    hint, rep = P.build_hint(dets(two_people()), 23, 300, 600, BOUND, mode="every", cast_count=4)
    check("detected count not within one of the planned cast: skipped",
          hint is None and "plan has 4" in rep["skipped"], rep["skipped"])
    hint, rep = P.build_hint(dets(two_people()), 23, 300, 600, BOUND, mode="every", cast_count=3)
    check("within one of the planned cast: fine", hint is not None, rep["skipped"])
    hint, rep = P.build_hint(dets([[]] * 8), 15, 300, 600, BOUND, mode="every")
    check("nobody detected: skipped", hint is None and rep["skipped"] == "nobody detected", rep["skipped"])
    anchored = {"Mara": {**BOUND["Mara"], "anchored": True}}
    hint, rep = P.build_hint(dets(two_people()), 23, 300, 600, anchored, mode="every")
    check("fastened to an object: skipped", hint is None and "fastened" in rep["skipped"], rep["skipped"])
    # Two bound people
    front = {"arms": "in front of the body", "legs": "", "ankle_gap": 0.12}
    cuffed_front = {P.RELB: (0.55, 0.37), P.LELB: (0.55, -0.37), P.RWRI: (0.95, 0.04), P.LWRI: (0.95, -0.04)}
    frames = []
    for _ in range(10):
        frames.append([body(neck=(100.0, 100.0), arms=ARMS_OUT),
                       body(neck=(300.0, 100.0), arms=cuffed_front),
                       body(neck=(500.0, 100.0), arms=BEHIND_FRONT)])
    hint, rep = P.build_hint(dets(frames), 19, 300, 640, {"Mara": BOUND["Mara"], "Ines": front}, mode="every")
    # From the front, cuffed in front and cuffed behind (wrists hidden, so "consistent")
    # differ only by a tenth of a shoulder at the elbows: the fit cannot tell them apart.
    check("two bound, cuffed in front vs behind seen from the front: skipped, not guessed",
          hint is None and rep["skipped"].startswith("could not tell"), (rep["identified"], rep["skipped"]))
    up = {"arms": "above the head", "legs": "", "ankle_gap": 0.12}
    overhead = {P.RELB: (-0.45, 0.43), P.LELB: (-0.45, -0.43), P.RWRI: (-0.95, 0.03), P.LWRI: (-0.95, -0.03)}
    frames = [[body(neck=(100.0, 150.0), arms=ARMS_OUT), body(neck=(300.0, 150.0), arms=overhead),
               body(neck=(500.0, 150.0), arms=BEHIND_FRONT)] for _ in range(10)]
    hint, rep = P.build_hint(dets(frames), 19, 450, 640, {"Mara": BOUND["Mara"], "Ines": up}, mode="every")
    check("two bound, behind the back vs above the head: each gets their own body",
          rep["identified"] == {"Mara": 2, "Ines": 1} and hint is not None, (rep["identified"], rep["skipped"]))
    frames = [[body(neck=(100.0, 100.0), arms=ARMS_OUT), body(neck=(300.0, 100.0), arms=BEHIND_FRONT),
               body(neck=(500.0, 100.0), arms=BEHIND_FRONT)] for _ in range(10)]
    hint, rep = P.build_hint(dets(frames), 19, 300, 640, {"Mara": BOUND["Mara"], "Ines": BOUND["Mara"]},
                             mode="every")
    check("two bound sharing one pose: both found, no assignment needed",
          sorted(rep["identified"].values()) == [1, 2] and hint is not None, (rep["identified"], rep["skipped"]))


def test_tracking():
    print("\n=== tracking ===")
    frames = two_people(n=20, drift=3.0)
    for i in (5, 6, 7):
        frames[i] = [frames[i][1]]
    frames_out, rep, _t = P._hint_skeletons(dets(frames), 39, 300, 700, BOUND, mode="every")
    check("missed for 3 analysed frames: still tracked", frames_out is not None and not rep["skipped"],
          rep["skipped"])
    bound_everywhere = frames_out is not None and all(
        any(abs(k[P.NECK, 0] - (180 + 1.5 * t)) < 2.0 for k in people) for t, people in enumerate(frames_out))
    check("...and the bound skeleton is in every output frame (gaps filled)", bound_everywhere)
    frames = two_people(n=20, drift=3.0)
    for i in (2, 5, 8, 11, 14, 17):
        frames[i] = [frames[i][1]]
    hint, rep = P.build_hint(dets(frames), 39, 300, 700, BOUND, mode="every")
    check("missing in more than 25% of frames: skipped", hint is None and "lost track of Mara" in rep["skipped"],
          rep["skipped"])
    frames = two_people(n=40, drift=1.0)
    for i in range(12, 20):
        frames[i] = [frames[i][1]]
    hint, rep = P.build_hint(dets(frames), 79, 300, 700, BOUND, mode="every")
    check("lost for 8 frames (longer than the hold), then re-identified by fit",
          hint is not None and not rep["skipped"], rep["skipped"])
    check("boxes_last holds the bound torso at the end of the shot",
          "Mara" in rep["boxes_last"] and rep["boxes_last"]["Mara"][0] < 260, rep["boxes_last"])


def test_carry_helpers():
    print("\n=== carry helpers ===")
    people = [body(neck=(180.0, 100.0)), body(neck=(440.0, 100.0))]
    boxes = P.torso_boxes(people, {"Mara": 1, "Ines": None, "Bad": 7})
    check("torso_boxes: one box per identified person", set(boxes) == {"Mara"}, boxes)
    b = boxes["Mara"]
    check("torso_boxes: the box surrounds the shoulders and hips",
          b[0] < 440 - 39 and b[2] > 440 + 39 and b[1] < 100 and b[3] > 200, b)
    ident = P.identify_by_boxes(list(reversed(people)), boxes)
    check("identify_by_boxes finds the stored person in a new frame", ident == {"Mara": 0}, ident)
    ident = P.identify_by_boxes(people, {"Mara": [0.0, 0.0, 3.0, 3.0]})
    check("identify_by_boxes: no overlap, no identity", ident == {"Mara": None}, ident)


# --------------------------------------------------------------------------------------
# Tracking through crossings, re-acquisition, identification by contact, carry contract,
# facing without a face (review fixes)
# --------------------------------------------------------------------------------------

PROF_BEHIND = {P.RELB: (0.55, 0.28), P.LELB: (0.55, 0.28), P.RWRI: (0.92, 0.22), P.LWRI: (0.92, 0.22)}
PROF_REACH = {P.RELB: (0.3, -0.4), P.LELB: (0.3, -0.4), P.RWRI: (0.2, -0.9), P.LWRI: (0.2, -0.9)}


def walkers(n, xb, vb, xf, vf, bound_arms=None, free_arms=None, hide=None, shuffle=True,
            bound_kw=None, free_kw=None):
    """Two people walking in profile (T = 100): the bound one faces image right (fwd -1,
    so behind the back is image left), the free one faces left. Returns (frames, roles);
    every third frame lists them in the other order, as DWPose does. hide(i, xb, xf) ->
    True drops the bound person from that frame."""
    frames, roles = [], []
    for i in range(n):
        b = body(neck=(xb + vb * i, 150.0), view="profile", width=0.1, fwd=-1.0,
                 arms=bound_arms(i) if callable(bound_arms) else (bound_arms or PROF_BEHIND),
                 **(bound_kw or {}))
        f = body(neck=(xf + vf * i, 150.0), view="profile", width=0.1, fwd=1.0,
                 arms=free_arms or ARMS_OUT, **(free_kw or {}))
        ks, r = [b, f], ["B", "F"]
        if hide is not None and hide(i, xb + vb * i, xf + vf * i):
            ks, r = [f], ["F"]
        if shuffle and i % 3 == 1:
            ks, r = ks[::-1], r[::-1]
        frames.append(ks)
        roles.append(r)
    return frames, roles


def track_roles(tracks, roles, n):
    return ["".join(roles[p][t["pos"][p]] if p in t["pos"] else "." for p in range(n)) for t in tracks]


def rewritten_inputs(fn):
    """Run fn() and return the pass-1 skeletons _rewrite was given, as [(frame, kp)]."""
    seen, orig = [], P._rewrite

    def cap(kps1, kps, idx, g, spec, blend, *a, **kw):
        seen.extend(zip(idx, kps1))
        return orig(kps1, kps, idx, g, spec, blend, *a, **kw)
    P._rewrite = cap
    try:
        res = fn()
    finally:
        P._rewrite = orig
    return res, seen


def rewrote_only(seen, frames, roles, role="B", stride=2):
    """Every skeleton handed to the rewrite is a detection of a person whose role is in
    `role`."""
    for t, kp in seen:
        i = t // stride
        mine = [k for k, r in zip(frames[i], roles[i]) if r in role]
        if not any(np.allclose(kp, k) for k in mine):
            return False
    return True


def test_track_assignment():
    """People crossing: one global assignment per frame with a constant-velocity
    prediction, never greedy, and a held track cannot take a person a live track explains."""
    print("\n=== tracking: global assignment through crossings ===")
    rng = np.random.default_rng(3)
    same = True
    for _ in range(60):
        r, c = rng.integers(1, 6), rng.integers(1, 6)
        cost = rng.uniform(0.0, 3.0, (r, c))
        cost[rng.uniform(size=(r, c)) < 0.3] = np.inf
        w = np.where(np.isfinite(cost), np.minimum(cost - 2.0, 0.0), 0.0)
        a = sum(w[i, j] for i, j in P._match(cost, 2.0))
        b = sum(w[i, j] for i, j in P._match_exact(w))
        same = same and abs(a - b) < 1e-9
    check("scipy's assignment and the exact search agree on the total (60 random frames)", same)
    w = np.array([[-1.9, -1.8], [-1.8, -0.1]])
    check("the exact search is global, not greedy (greedy takes -1.9 and leaves -0.1)",
          sorted(P._match_exact(w)) == [(0, 1), (1, 0)], P._match_exact(w))
    n = 24
    frames, roles = walkers(n, 100.0, 20.0, 560.0, -20.0)          # cross at frame 11.5
    people = [[P._person(k) for k in f] for f in frames]
    idx = list(range(0, 2 * n, 2))
    got = track_roles(P._build_tracks(people, idx), roles, n)
    check("two people walking past each other keep their own tracks",
          sorted(got) == ["B" * n, "F" * n], got)
    saved = P._lsa
    P._lsa = lambda: None
    try:
        got2 = track_roles(P._build_tracks(people, idx), roles, n)
    finally:
        P._lsa = saved
    check("...the same without scipy (exact search)", got2 == got, got2)
    # The bound person is hidden while the other passes over where they stood: the held
    # track must not take the other person, who stands where it was.
    frames, roles = walkers(30, 100.0, 12.0, 500.0, -12.0, hide=lambda i, xb, xf: abs(xb - xf) < 40)
    people = [[P._person(k) for k in f] for f in frames]
    got = track_roles(P._build_tracks(people, list(range(0, 60, 2))), roles, 30)
    check("a held (hidden) track never takes the person passing over its spot",
          all(set(s) <= {"B", "."} or set(s) <= {"F", "."} for s in got) and len(got) == 2, got)
    # Without appearance a person passing over the hidden bound person is a contact the
    # signature cannot settle (same build): skipped. With it, read from the pixels.
    carry = {"Mara": P._torso_box(frames[0][0])}
    (skel, rep, _t), seen = rewritten_inputs(lambda: P._hint_skeletons(
        dets(frames), 59, 460, 700, BOUND, carry=carry, mode="every"))
    check("...without appearance: the rewrite only ever reads the bound person's own skeleton, or "
          "the shot is skipped (motion and signature must both settle the contact)",
          (skel is None and "where two people met" in rep["skipped"])
          or (skel is not None and seen and rewrote_only(seen, frames, roles)), rep["skipped"])
    d = with_appearance(frames, roles, 460, 700, depth={"B": 0, "F": 1})
    (skel, rep, _t), seen = rewritten_inputs(lambda: P._hint_skeletons(
        d, 59, 460, 700, BOUND, carry=carry, mode="every"))
    check("...with appearance: built, and the rewrite only ever reads the bound person's own skeleton",
          skel is not None and seen and rewrote_only(seen, frames, roles), rep["skipped"])
    # Fast crossing, the bound person's arms break after the opening. Without appearance
    # the same build cannot be told apart after it: skipped. With appearance: rewritten,
    # never the free person; at frame 11 both torsos sit on one spot.
    frames, roles = walkers(n, 100.0, 20.0, 540.0, -20.0, free_arms=PROF_BEHIND,
                            bound_arms=lambda i: PROF_BEHIND if i < 6 else PROF_REACH)
    carry = {"Mara": P._torso_box(frames[0][0])}
    hint, rep = P.build_hint(dets(frames), 2 * n - 1, 460, 700, BOUND, carry=carry, mode="every")
    check("a crossing of the same build without appearance: skipped, not guessed",
          hint is None and rep["skipped"].startswith("lost track of Mara where two people met"),
          rep["skipped"])
    d = with_appearance(frames, roles, 460, 700, depth={"B": 0, "F": 1})
    (skel, rep, _t), seen = rewritten_inputs(lambda: P._hint_skeletons(
        d, 2 * n - 1, 460, 700, BOUND, carry=carry, mode="every"))
    check("...with appearance: hint built, the free person never rewritten",
          skel is not None and rewrote_only(seen, frames, roles), rep["skipped"])
    check("...the frames where the two torsos are mixed are left to the frames around them",
          any("on one spot" in x for x in rep["notes"]) and all(t != 22 for t, _k in seen),
          (rep["notes"], sorted(t for t, _k in seen)[9:14]))
    broken_drawn = skel is not None and any(
        any(np.allclose(d_, k) for d_ in skel[2 * i]) for i in range(6, n)
        for k, r in zip(frames[i], roles[i]) if r == "B")
    check("...and the bound person's broken pass-1 skeleton is never drawn", not broken_drawn)
    free_kept = skel is not None and all(
        any(np.allclose(d_[:, :2], k[:, :2]) for d_ in skel[2 * i])
        for i in range(n) if 2 * i not in (20, 22, 24) for k, r in zip(frames[i], roles[i]) if r == "F")
    check("...and the free person is drawn as detected", free_kept)
    # Two people who look the same crossing slowly: who is who afterwards cannot be told.
    frames, roles = walkers(24, 280.0, 2.0, 326.0, -2.0, free_arms=PROF_BEHIND,
                            free_kw={}, bound_kw={})
    for i in range(24):                    # same facing: nothing tells them apart
        for j, r in enumerate(roles[i]):
            if r == "F":
                frames[i][j] = body(neck=(326.0 - 2.0 * i, 150.0), view="profile", width=0.1,
                                    fwd=-1.0, arms=PROF_BEHIND)
    carry = {"Mara": P._torso_box(frames[0][0])}
    hint, rep = P.build_hint(dets(frames), 47, 400, 700, BOUND, carry=carry, mode="every")
    check("look-alikes crossing slowly: skipped, never guessed",
          hint is None and rep["skipped"].startswith("lost track of Mara"), rep["skipped"])
    check("...and nothing is carried out of the shot", rep["identified"]["Mara"] is None
          and rep["boxes_last"] == {}, (rep["identified"], rep["boxes_last"]))
    roles_twin = [["B" if r == "B" else "twin" for r in rl] for rl in roles]
    d = with_appearance(frames, roles_twin, 460, 700, depth={"B": 0, "twin": 1})
    hint, rep = P.build_hint(d, 47, 460, 700, BOUND, carry=carry, mode="every",
                             carry_appearance={"Mara": d["appearance"][0][roles[0].index("B")]})
    check("...dressed alike too, with appearance: still skipped",
          hint is None and (rep["skipped"] == "two people look too alike to keep apart"
                            or rep["skipped"].startswith("lost track of Mara")), rep["skipped"])


def test_reacquire():
    """After the bound track is lost, a later track is them only if it starts near the
    predicted spot, matches their body proportions and fits the template."""
    print("\n=== re-acquiring the bound person ===")

    def scene(n, leave, back, x_back=None, arms_back=BEHIND_FRONT, width_back=0.78,
              second=None):
        frames, roles = [], []
        for i in range(n):
            ks, r = [body(neck=(440.0, 100.0), arms=ARMS_OUT)], ["F"]
            if i < leave:
                ks.insert(0, body(neck=(180.0, 100.0), arms=BEHIND_FRONT))
                r.insert(0, "B")
            elif i >= back:
                ks.insert(0, body(neck=(x_back or 180.0, 100.0), arms=arms_back, width=width_back))
                r.insert(0, "G")
                if second is not None:
                    ks.append(body(neck=(second, 100.0), arms=BEHIND_FRONT))
                    r.append("H")
            frames.append(ks)
            roles.append(r)
        return frames, roles

    # Within the hold a body at the predicted spot is the same person (that is what the
    # repair is for); these cases are past it, or out of a held track's reach.
    frames, roles = scene(44, 36, 38, x_back=400.0)
    (skel, rep, _t), seen = rewritten_inputs(lambda: P._hint_skeletons(
        dets(frames), 87, 300, 800, BOUND, mode="every"))
    check("someone else walks in 2.2 T from where the bound person left: not taken as them",
          skel is not None and seen and rewrote_only(seen, frames, roles), rep["skipped"])
    check("...the bound person is drawn up to where they were last seen, nobody after",
          skel is not None and any(abs(k[P.NECK, 0] - 180) < 1 for k in skel[71])
          and not any(abs(k[P.NECK, 0] - 180) < 1 for k in skel[72])
          and not any(abs(k[P.NECK, 0] - 180) < 1 for k in skel[86]))
    newcomer = skel is not None and any(np.allclose(d, frames[40][0]) for d in skel[80])
    check("...and the newcomer keeps their own pass-1 skeleton", newcomer)
    frames, roles = scene(44, 24, 33, width_back=0.45)
    hint, rep = P.build_hint(dets(frames), 87, 300, 700, BOUND, mode="every")
    check("past the hold, same spot, other proportions (shoulders 0.45 T, not 0.78): not taken",
          hint is None and "lost track of Mara" in rep["skipped"], rep["skipped"])
    frames, roles = scene(44, 24, 33, arms_back=ARMS_OUT)
    hint, rep = P.build_hint(dets(frames), 87, 300, 700, BOUND, mode="every")
    check("past the hold, same spot and build, off the template (arms out): not taken",
          hint is None and "lost track of Mara" in rep["skipped"], rep["skipped"])
    frames, roles = scene(44, 24, 33)
    (skel, rep, _t), seen = rewritten_inputs(lambda: P._hint_skeletons(
        dets(frames), 87, 300, 700, BOUND, mode="every"))
    check("past the hold, same spot, same build, on the template: taken as the bound person",
          skel is not None and any(t >= 66 for t, _k in seen) and rewrote_only(seen, frames, roles, "BG"),
          rep["skipped"])
    frames, roles = scene(44, 24, 33, second=240.0)
    hint, rep = P.build_hint(dets(frames), 87, 300, 700, BOUND, mode="every")
    check("two people come back near the spot and both fit: skipped",
          hint is None and rep["skipped"] == "could not tell who came back as Mara", rep["skipped"])
    # The 25% rule still stands when the person never comes back.
    frames, roles = scene(40, 20, 26, x_back=400.0)
    hint, rep = P.build_hint(dets(frames), 79, 300, 800, BOUND, mode="every")
    check("left at frame 20 of 40, someone else 2.2 T away at 26: skipped (50% missing)",
          hint is None and "lost track of Mara in 50%" in rep["skipped"], rep["skipped"])


def test_identification_contact():
    """Without a carried identity, fit is trusted only for exactly one fitting person who
    is in contact with nobody else."""
    print("\n=== identification: contact and the carried identity ===")
    raised = {P.RELB: (-0.2, 0.6), P.LELB: (-0.2, -0.6), P.RWRI: (-0.6, 0.5), P.LWRI: (-0.6, -0.5)}

    def pair(dx, other_arms, n=10):
        return [[body(neck=(300.0, 100.0), arms=BEHIND_FRONT),
                 body(neck=(300.0 + dx, 105.0), arms=other_arms)] for _ in range(n)]
    hint, rep = P.build_hint(dets(pair(30.0, raised)), 19, 300, 700, BOUND, mode="every")
    check("someone right behind the bound person (torso boxes overlap): not guessed",
          hint is None and rep["skipped"] == "two people in contact and no carried identity; not guessing",
          rep["skipped"])
    hint, rep = P.build_hint(dets(pair(150.0, raised)), 19, 300, 700, BOUND, mode="every")
    check("someone beside them, 0.3 T between the torso boxes: not guessed",
          hint is None and "in contact" in rep["skipped"], rep["skipped"])
    hint, rep = P.build_hint(dets(pair(260.0, raised)), 19, 300, 700, BOUND, mode="every")
    check("someone 1.4 T away who does not fit: identified by fit",
          rep["identified"]["Mara"] == 0 and hint is not None, rep["skipped"])
    carry = {"Mara": P._torso_box(body(neck=(300.0, 100.0)))}
    hint, rep = P.build_hint(dets(pair(30.0, raised)), 19, 300, 700, BOUND, carry=carry, mode="every")
    check("the same contact with a carried identity: identified across the cut",
          rep["identified"]["Mara"] == 0 and hint is not None, rep["skipped"])
    hidden_arms = {P.RELB: None, P.LELB: None, P.RWRI: None, P.LWRI: None}
    hint, rep = P.build_hint(dets(pair(400.0, hidden_arms)), 19, 300, 900, BOUND, mode="every")
    check("another person whose arms cannot be measured: not guessed",
          hint is None and rep["skipped"].startswith("could not measure everyone's arms"), rep["skipped"])
    hint, rep = P.build_hint(dets([[body(neck=(300.0, 100.0), arms=BEHIND_FRONT)]] * 10), 19, 300, 700,
                             BOUND, mode="every")
    check("a lone person: identified by fit as before", rep["identified"]["Mara"] == 0, rep["skipped"])


def test_carry_contract():
    """report["identified"] indexes the last analysed frame, so the contract's
    torso_boxes(handoff people, identified) is boxes_last; detect() always reads it."""
    print("\n=== carry contract ===")
    frames = two_people(n=12)
    frames[-1] = frames[-1][::-1]                  # DWPose's order changes at the handoff
    hint, rep = P.build_hint(dets(frames), 23, 300, 600, BOUND, mode="every")
    check("identified is the index in the last analysed frame", rep["identified"] == {"Mara": 1}, rep["identified"])
    check("identified_first is the index in the first", rep["identified_first"] == {"Mara": 0},
          rep["identified_first"])
    check("identified_frame is that frame's output index", rep["identified_frame"] == 22, rep["identified_frame"])
    check("torso_boxes(handoff people, identified) == boxes_last",
          P.torso_boxes(frames[-1], rep["identified"]) == rep["boxes_last"], (rep["boxes_last"],))
    raised = {P.RELB: (-0.2, 0.6), P.LELB: (-0.2, -0.6), P.RWRI: (-0.6, 0.5), P.LWRI: (-0.6, -0.5)}
    shot2 = [[body(neck=(440.0, 100.0), arms=ARMS_OUT), body(neck=(180.0, 100.0), arms=raised)]] * 10
    _h, rep2 = P.build_hint(dets(shot2), 19, 300, 600, BOUND, mode="every",
                            carry=P.torso_boxes(frames[-1], rep["identified"]))
    check("the literal carry picks the bound person in the next shot (broken opening)",
          rep2["identified"] == {"Mara": 1} and any("across the cut" in x for x in rep2["notes"]), rep2)
    frames = two_people(n=12)
    frames[-1] = [frames[-1][1]]
    _h, rep = P.build_hint(dets(frames), 23, 300, 600, BOUND, mode="every")
    check("not in the last analysed frame: no index, no box, a note",
          rep["identified"] == {"Mara": None} and rep["boxes_last"] == {}
          and any("nothing to carry" in x for x in rep["notes"]), rep)

    class _Est:
        def __call__(self, img):
            return None
    det = P.PoseDetector(device="cpu")
    det._est, det._on_device = _Est(), True
    out = det.detect(torch.zeros((6, 8, 8, 3)), stride=4)
    check("detect() reads the last frame whatever the stride", out["index"] == [0, 4, 5], out["index"])


def test_facing_without_face():
    """No face cue: lying, the back is image up; upright, the nearest frame of the track
    that showed the face; never shown, the shot is skipped."""
    print("\n=== facing without a face cue ===")

    def faceless(kp):
        kp = kp.copy()
        kp[[P.NOSE, P.REYE, P.LEYE, P.REAR, P.LEAR], 2] = 0.0
        return kp
    # Lying with no face cue anywhere in the track: face up and face down look the same,
    # so the back is never assumed to be image up -- the shot is skipped.
    for phi, label in ((0.0, "head left"), (180.0, "head right")):
        kp = faceless(body(neck=(500.0, 400.0), T=150.0, phi=phi, view="profile", width=0.1,
                           arms={P.RELB: (0.55, 0.28), P.LELB: (0.55, 0.28),
                                 P.RWRI: (0.92, 0.22), P.LWRI: (0.92, 0.22)}))
        g, _s = geom(kp)
        check(f"lying, {label}, no face anywhere: the facing is unknown", bool(g.facing_unknown.all()),
              g.back)
        carry = {"Mara": P._torso_box(kp)}
        hint, rep = P.build_hint(dets([[kp]] * 10), 19, 800, 1000, BOUND, carry=carry, mode="every")
        check(f"...and the shot is skipped: 'could not tell which way they face' ({label})",
              hint is None and rep["skipped"] == "could not tell which way they face", rep["skipped"])
    # The face seen earlier in the same track, then lying with none (the face turned into
    # the floor, or a hood pulled up): the facing carries over -- either way up.
    for fwd, want_up, label in ((1.0, True, "face down"), (-1.0, False, "face up")):
        lying = body(neck=(500.0, 400.0), T=150.0, phi=0.0, view="profile", width=0.1, fwd=fwd)
        seq = [lying.copy() for _ in range(4)] + [faceless(lying) for _ in range(14)]
        out, g = rewrite(seq, {"arms": "behind the back"})
        up = out[-1][P.RELB, 1] < 400 - 30
        check(f"lying {label}, the face seen earlier in the track: the arms go to the back side "
              f"({'up' if want_up else 'down'}), facing known", up == want_up
              and not g.facing_unknown.any(), (img(out[-1], P.RELB), g.back[-1]))
    seq = [body(view="profile", width=0.1, fwd=1.0) for _ in range(3)] + \
          [faceless(body(view="profile", width=0.1, fwd=1.0)) for _ in range(12)]
    g = P._geometry(seq, list(range(0, 60, 4)))      # 4 apart: the last frames are past +-12
    check("upright, the face seen early in the track only: its facing is kept",
          bool(np.all(g.back < 0)) and not g.facing_unknown.any(), g.back)
    prof = [faceless(body(view="profile", width=0.1, fwd=1.0, arms=PROF_BEHIND)) for _ in range(10)]
    carry = {"Mara": P._torso_box(prof[0])}
    hint, rep = P.build_hint(dets([[k] for k in prof]), 19, 300, 600, BOUND, carry=carry, mode="every")
    check("upright in profile, the face never seen: skipped",
          hint is None and rep["skipped"] == "could not tell which way they face", rep["skipped"])
    hint, rep = P.build_hint(dets([[k] for k in prof]), 19, 300, 600, BOUND, mode="every")
    check("...and the same reason without a carried identity", rep["skipped"] == "could not tell which way they face",
          rep["skipped"])
    from_behind = {P.RELB: (0.55, -0.41), P.LELB: (0.55, 0.41), P.RWRI: (0.9, -0.02), P.LWRI: (0.9, 0.02)}
    back = [body(view="back", arms=from_behind) for _ in range(10)]
    hint, rep = P.build_hint(dets([[k] for k in back]), 19, 300, 600, BOUND, mode="every")
    check("seen from behind (square to the camera): the facing does not matter, not skipped",
          rep["skipped"] == "", rep["skipped"])


def test_review_low_items():
    print("\n=== long gaps, side noise, cropped torso, input types ===")
    a = body(neck=(100.0, 100.0))
    b = body(neck=(200.0, 100.0))
    index = list(range(0, 40, 2))
    frames = {0: a, 4: b}                              # 3 missed analysed frames
    check("a 3-frame gap is interpolated", P._fill_bound(frames, index, 4) is not None
          and np.allclose(P._fill_bound(frames, index, 4)[P.NECK, :2], (150.0, 100.0)))
    frames = {0: a, 5: b}                              # 4 missed
    check("a 4-frame gap: held up to the first missed analysed frame, then nobody",
          P._fill_bound(frames, index, 1) is a and P._fill_bound(frames, index, 2) is None
          and P._fill_bound(frames, index, 6) is None and P._fill_bound(frames, index, 9) is b)
    check("a track ending before the shot does: not held to the end",
          P._fill_bound({0: a, 3: b}, index, 7) is b and P._fill_bound({0: a, 3: b}, index, 8) is None)
    check("a track reaching the last analysed frame: held to the end",
          P._fill_bound({0: a, 19: b}, index, 45) is b)
    fr = two_people(n=20, drift=3.0)
    for i in range(8, 13):
        fr[i] = [fr[i][1]]
    skel, rep, _t = P._hint_skeletons(dets(fr), 39, 300, 700, BOUND, mode="every")
    near = lambda t: any(abs(k[P.NECK, 0] - (180 + 1.5 * t)) < 3 for k in skel[t])
    mid = [t for t in range(16, 25) if near(t)]
    check("through the pipeline: the middle of a 5-frame gap draws nobody for them",
          skel is not None and mid == [] and near(15) and near(25), mid)
    # Three-quarter view (w = 0.78, side limit 1.5 x 0.275 T): a wrist past it in one frame
    # is noise.
    spec = {"arms": "behind the back"}
    elb = {P.RELB: (0.55, 0.29), P.LELB: (0.55, -0.29)}
    base = body(width=0.55, arms={**elb, P.RWRI: None, P.LWRI: None})
    out1 = body(width=0.55, arms={**elb, P.RWRI: (0.9, 0.54), P.LWRI: None})    # 0.13 T past the limit
    far = body(width=0.55, arms={**elb, P.RWRI: (0.9, 0.70), P.LWRI: None})     # 0.29 T past it
    _b, f1, why1, _h = _run_check(base, out1, {5, 15}, spec, n=40)
    check("3/4 view: 'hands out to the sides' in single frames does not count", not f1, (f1, why1))
    _b, f2, why2, _h = _run_check(base, out1, {5, 6}, spec, n=40)
    check("...two frames in a row do", f2 == [10, 12] and "hands out to the sides" in why2, (f2, why2))
    _b, f3, why3, _h = _run_check(base, far, {9}, spec, n=40)
    check("...and one frame past the noise margin does", f3 == [18] and "hands out to the sides" in why3,
          (f3, why3))
    # Cropped torsos
    waist_up = body(arms=BEHIND_FRONT)
    waist_up[[P.RHIP, P.LHIP, P.RKNE, P.LKNE, P.RANK, P.LANK], 2] = 0.0
    off_edge = body(arms=BEHIND_FRONT)
    off_edge[[P.NECK, P.LSHO], 2] = 0.0                # DWPose drops the neck with a shoulder
    top_cut = body(arms=BEHIND_FRONT)
    top_cut[[P.NOSE, P.NECK, P.RSHO, P.LSHO, P.REYE, P.LEYE, P.REAR, P.LEAR], 2] = 0.0
    for kp, label in ((waist_up, "waist-up framing"), (off_edge, "a shoulder off the edge"),
                      (top_cut, "shoulders cut at the top")):
        _h, rep = P.build_hint(dets([[kp]] * 10), 19, 300, 600, BOUND, mode="every")
        check(f"{label}: skipped as 'torso not fully visible'", rep["skipped"] == "torso not fully visible",
              rep["skipped"])
    # Input types
    d = dets(two_people(n=9))
    for label, ix in (("a tensor", torch.tensor(d["index"])), ("an ndarray", np.asarray(d["index"]))):
        try:
            h, rep = P.build_hint({"index": ix, "people": d["people"]}, 17, 300, 600, BOUND, mode="every")
            ok = h is not None and rep["identified"] == {"Mara": 0}
        except Exception as e:
            ok, rep = False, f"{type(e).__name__}: {e}"
        check(f"detections['index'] as {label}", ok, rep if not ok else "")
    try:
        h, rep = P.build_hint({"index": d["index"], "people": [np.stack(f) for f in d["people"]]},
                              17, 300, 600, BOUND, mode="every")
        ok = h is not None
    except Exception as e:
        ok = False
    check("each frame's people as one (P, 18, 3) array", ok)
    sched8 = [1.0, 0.988, 0.973, 0.952, 0.923, 0.878, 0.8, 0.632, 0.0]
    want = P.pose_sigma_window(sched8, 0.6)
    for bad in (float("nan"), None, "x", float("inf")):
        try:
            got = P.pose_sigma_window(sched8, bad)
        except Exception as e:
            got = f"{type(e).__name__}: {e}"
        check(f"pose_sigma_window with pose_end {bad!r}: the default 0.6", got == want, got)


# --------------------------------------------------------------------------------------
# Appearance: the vectors, the hard gates, carrying them across a cut
# --------------------------------------------------------------------------------------

def test_appearance_vectors():
    """One vector per person from the pixels: torso, head and hair, upper legs. Light and
    shade move it little; other clothes move it far."""
    print("\n=== appearance vectors ===")
    H, W = 460, 700
    b, f = body(neck=(200.0, 150.0)), body(neck=(480.0, 150.0), arms=ARMS_OUT)
    img = paint([b, f], ["B", "F"], H, W)
    vb, vf = P.person_appearance(img, b), P.person_appearance(img, f)
    check("a float32 vector of POSE_APP_SIZE per person",
          vb is not None and vb.dtype == np.float32 and vb.shape == (P.POSE_APP_SIZE,), None if vb is None else vb.shape)
    blocks = vb.reshape(len(P.POSE_APP_REGIONS), -1)
    check("one normalised histogram per region (torso, head, legs), all seen here",
          all(abs(float(x.sum()) - 1.0) < 1e-4 for x in blocks), [float(x.sum()) for x in blocks])
    for light in (0.8, 1.15):
        d = P.appearance_distance(vb, P.person_appearance(paint([b, f], ["B", "F"], H, W, light=light), b))
        check(f"the same person at {light:.2f}x the light: well within POSE_APP_MATCH_MAX",
              d is not None and d < 0.5 * P.POSE_APP_MATCH_MAX, d)
    d = P.appearance_distance(vb, vf)
    check("two differently dressed people: past POSE_APP_DISTINCT_MIN", d is not None and d > P.POSE_APP_DISTINCT_MIN, d)
    nolegs = b.copy()
    nolegs[[P.RKNE, P.LKNE, P.RANK, P.LANK], 2] = 0.0
    vn = P.person_appearance(img, nolegs)
    check("knees hidden: the legs block is NaN, the rest still compares",
          vn is not None and np.isnan(vn.reshape(3, -1)[2]).all()
          and P.appearance_distance(vb, vn) is not None and P.appearance_distance(vb, vn) < 0.05,
          None if vn is None else P.appearance_distance(vb, vn))
    head_only = np.full(P.POSE_APP_SIZE, np.nan)
    head_only.reshape(3, -1)[1] = vb.reshape(3, -1)[1]
    check("a head alone is not enough to compare (None)", P.appearance_distance(vb, head_only) is None)
    check("no torso: no vector", P.person_appearance(img, np.zeros((18, 3))) is None)
    vt = P.person_appearance(torch.from_numpy(img).float() / 255.0, b)
    check("a 0..1 float tensor frame reads the same as uint8",
          vt is not None and P.appearance_distance(vb, vt) < 1e-3)
    over = body(neck=(230.0, 150.0), arms=ARMS_OUT)
    check("clean flags: apart, both clean; overlapping, neither",
          P._clean_flags([b, f]) == [True, True] and P._clean_flags([b, over]) == [False, False])

    class _Est:
        def __call__(self, image):
            return np.stack([b, f])
    det = P.PoseDetector(device="cpu")
    det._est, det._on_device = _Est(), True
    out = det.detect(torch.from_numpy(np.stack([img, img, img])).float() / 255.0, stride=2)
    ok = (len(out["appearance"]) == len(out["people"]) == 2
          and all(len(a) == len(p) == 2 for a, p in zip(out["appearance"], out["people"]))
          and P.appearance_distance(out["appearance"][0][0], vb) < 1e-3
          and P.appearance_distance(out["appearance"][1][1], vf) < 1e-3)
    check("detect() returns appearance per analysed frame, aligned with the people", ok)
    # The running mean of a stretch is built from its clean frames only.
    people = [[b, f], [b, over], [b, f]]
    scene = P._Scene(people, [0, 2, 4], [[vb, vf], [vf, vf], [vb, vf]])
    m = scene.mean({0: 0, 1: 0, 2: 0})
    check("a track's appearance mean leaves out the frames in contact", P.appearance_distance(m, vb) < 1e-3)


def _flip(arms):
    return {j: (ab[0], -ab[1]) if ab is not None else None for j, ab in arms.items()}


PROF_HANG = {P.RELB: (0.55, 0.0), P.LELB: (0.55, 0.0), P.RWRI: (1.05, -0.06), P.LWRI: (1.05, -0.06)}


def bouncers(n=24, k=10, v=20.0, free_arms=PROF_HANG, x0=150.0, x1=550.0):
    """Two people (same build, T = 100) walk toward each other in profile, meet where
    their images overlap at frame k and both turn back: the bound one (arms behind the
    back) and another whose arms hang (`free_arms`, given as for facing image right).
    Every third frame lists them in the other order."""
    frames, roles = [], []
    for i in range(n):
        s = v * min(i, k) - v * max(0, i - k)
        out = i <= k
        b = body(neck=(x0 + s, 150.0), view="profile", width=0.1, fwd=-1.0 if out else 1.0,
                 arms=PROF_BEHIND if out else _flip(PROF_BEHIND))
        f = body(neck=(x1 - s, 150.0), view="profile", width=0.1, fwd=1.0 if out else -1.0,
                 arms=_flip(free_arms) if out else free_arms)
        ks, r = [b, f], ["B", "F"]
        if i % 3 == 1:
            ks, r = ks[::-1], r[::-1]
        frames.append(ks)
        roles.append(r)
    return frames, roles


def test_appearance_gates():
    """Wherever identity could jump -- a track going on after a missed frame, a track
    taken up again after the person was lost, two people parting after contact -- the
    person must look like the bound person; without appearance, stricter geometry."""
    print("\n=== appearance as a hard gate ===")
    H, W = 460, 700

    def stay_and_swap(n, leave, back, arms_back=BEHIND_FRONT, width_back=0.78, newcomer="G"):
        frames, roles = [], []
        for i in range(n):
            ks, r = [body(neck=(500.0, 150.0), arms=ARMS_OUT)], ["F"]
            if i < leave:
                ks.insert(0, body(neck=(200.0, 150.0), arms=BEHIND_FRONT))
                r.insert(0, "B")
            elif i >= back:
                ks.insert(0, body(neck=(200.0, 150.0), arms=arms_back, width=width_back))
                r.insert(0, newcomer)
            frames.append(ks)
            roles.append(r)
        return frames, roles

    # Within the track hold: someone else, same build and pose, other clothes.
    frames, roles = stay_and_swap(30, 15, 18)
    d = with_appearance(frames, roles, H, W)
    (skel, rep, _t), seen = rewritten_inputs(lambda: P._hint_skeletons(d, 59, H, W, BOUND, mode="every"))
    check("a newcomer in other clothes on the bound person's spot within the hold: never taken as them",
          rewrote_only(seen, frames, roles) and (skel is None or not any(t >= 36 for t, _k in seen)),
          (rep["skipped"], sorted(t for t, _k in seen)[-3:]))
    check("...and with half the shot gone the shot is skipped", skel is None and "lost track of Mara" in rep["skipped"],
          rep["skipped"])
    # The same person back after the missed frames (hidden behind something): kept.
    frames, roles = stay_and_swap(30, 15, 18, newcomer="B")
    d = with_appearance(frames, roles, H, W)
    (skel, rep, _t), seen = rewritten_inputs(lambda: P._hint_skeletons(d, 59, H, W, BOUND, mode="every"))
    check("the bound person back in their own clothes after missed frames: followed on",
          skel is not None and any(t >= 36 for t, _k in seen) and rewrote_only(seen, frames, roles), rep["skipped"])
    # Re-acquisition after a loss longer than the hold.
    frames, roles = stay_and_swap(44, 24, 33)
    d = with_appearance(frames, roles, H, W)
    hint, rep = P.build_hint(d, 87, H, W, BOUND, mode="every")
    check("past the hold, same spot, build and pose, other clothes: not taken as them",
          hint is None and "lost track of Mara" in rep["skipped"], rep["skipped"])
    frames, roles = stay_and_swap(44, 24, 33, newcomer="B")
    d = with_appearance(frames, roles, H, W)
    (skel, rep, _t), seen = rewritten_inputs(lambda: P._hint_skeletons(d, 87, H, W, BOUND, mode="every"))
    check("...and the bound person themself coming back there is taken up again",
          skel is not None and any(t >= 66 for t, _k in seen), rep["skipped"])
    # Two people meet where their images overlap and both turn back.
    frames, roles = bouncers()
    carry = {"Mara": P._torso_box(frames[0][roles[0].index("B")])}
    for c, lab in ((None, "no carry"), (carry, "carried")):
        hint, rep = P.build_hint(dets(frames), 47, H, W, BOUND, carry=c, mode="every")
        check(f"a bounce without appearance ({lab}): skipped, never guessed",
              hint is None and "where two people met" in rep["skipped"], rep["skipped"])
        d = with_appearance(frames, roles, H, W, depth={"B": 0, "F": 1})
        (skel, rep, _t), seen = rewritten_inputs(lambda: P._hint_skeletons(d, 47, H, W, BOUND, carry=c,
                                                                          mode="every"))
        check(f"...with appearance ({lab}): built, and only the bound person is rewritten, on both sides of it",
              skel is not None and rewrote_only(seen, frames, roles)
              and any(t < 16 for t, _k in seen) and any(t > 26 for t, _k in seen), rep["skipped"])
        drawn_wrong = skel is not None and any(
            abs(sk[P.NECK, 0] - fr[rl.index("F")][P.NECK, 0]) < 1.0
            for i, (fr, rl) in enumerate(zip(frames, roles)) for sk in skel[2 * i][:1]
            if abs(fr[0][P.NECK, 0] - fr[1][P.NECK, 0]) > 60)
        check(f"...the bound drawing never sits on the other person ({lab})", not drawn_wrong)
    d = with_appearance(frames, [["B" if r == "B" else "twin" for r in rl] for rl in roles], H, W,
                        depth={"B": 0, "twin": 1})
    hint, rep = P.build_hint(d, 47, H, W, BOUND, mode="every")
    check("a bounce of two people dressed alike: 'two people look too alike to keep apart'",
          hint is None and rep["skipped"] == "two people look too alike to keep apart", rep["skipped"])


def test_strict_geometry_without_appearance():
    """Detections without appearance (fake detectors): a reappearance after any missed
    frame must pass the signature and template-fit gates, and a separation after contact
    must be clear on motion and signature both."""
    print("\n=== strict geometry without appearance ===")
    frames, roles = [], []
    for i in range(30):
        ks, r = [body(neck=(500.0, 150.0), arms=ARMS_OUT)], ["F"]
        if i < 15:
            ks.insert(0, body(neck=(200.0, 150.0), arms=BEHIND_FRONT))
            r.insert(0, "B")
        elif i >= 17:
            ks.insert(0, body(neck=(200.0, 150.0), arms=ARMS_OUT))
            r.insert(0, "G")
        frames.append(ks)
        roles.append(r)
    (skel, rep, _t), seen = rewritten_inputs(lambda: P._hint_skeletons(dets(frames), 59, 460, 700, BOUND,
                                                                      mode="every"))
    check("within the hold, someone with arms out at the spot: not merged into the bound track",
          rewrote_only(seen, frames, roles) and not any(t >= 34 for t, _k in seen), rep["skipped"])
    for i in range(17, 30):
        frames[i][0] = body(neck=(200.0, 150.0), arms=BEHIND_FRONT, width=0.45)
    (skel, rep, _t), seen = rewritten_inputs(lambda: P._hint_skeletons(dets(frames), 59, 460, 700, BOUND,
                                                                      mode="every"))
    check("...nor someone on the template of another build (shoulders 0.45 T, not 0.78)",
          rewrote_only(seen, frames, roles) and not any(t >= 34 for t, _k in seen), rep["skipped"])
    track = {"pos": {p: 0 for p in range(10)}}
    tracks = [track]
    people = [[body(arms=BEHIND_FRONT)] for _ in range(4)] + [[body(arms=ARMS_OUT)] for _ in range(6)]
    for p in (4, 5):
        del track["pos"][p]
    cut = P._cut_at_gaps(0, tracks, people, list(range(0, 20, 2)), lambda p: P._spec(BOUND["Mara"]), None)
    check("_cut_at_gaps: the stretch after the missed frames, off the template, becomes its own track",
          cut and sorted(tracks[0]["pos"]) == [0, 1, 2, 3] and sorted(tracks[1]["pos"]) == [6, 7, 8, 9])


def test_identification_apart():
    """Two people apart, the bound one on the template at the opening: identified again
    when the other's arms hang, sit on the hips or hold something apart -- from the front
    and from behind. Only a pose that truly fits (hands behind the back too) is a skip."""
    print("\n=== identification: two people apart ===")
    hang = None
    hips = {P.RELB: (0.5, 0.75), P.LELB: (0.5, -0.75), P.RWRI: (0.95, 0.35), P.LWRI: (0.95, -0.35)}
    holding = {P.RELB: (0.55, 0.45), P.LELB: (0.55, -0.45), P.RWRI: (0.85, 0.32), P.LWRI: (0.85, -0.32)}
    together = {P.RELB: (0.55, 0.37), P.LELB: (0.55, -0.37), P.RWRI: (0.95, 0.04), P.LWRI: (0.95, -0.04)}
    back_behind = {P.RELB: (0.55, -0.41), P.LELB: (0.55, 0.41), P.RWRI: (0.9, -0.02), P.LWRI: (0.9, 0.02)}
    for view, bound_arms in (("front", BEHIND_FRONT), ("back", back_behind)):
        for label, free in (("arms hanging", hang), ("hands on hips", hips), ("holding something apart", holding)):
            frames = [[body(neck=(180.0, 100.0), view=view, arms=bound_arms),
                       body(neck=(440.0, 100.0), view=view, arms=free)] for _ in range(10)]
            hint, rep = P.build_hint(dets(frames), 19, 300, 600, BOUND, mode="every")
            check(f"{view} view, the other person's {label}: the bound person is identified",
                  rep["identified"]["Mara"] == 0 and hint is not None, rep["skipped"])
    frames = [[body(neck=(180.0, 100.0), arms=BEHIND_FRONT), body(neck=(440.0, 100.0), arms=together)]
              for _ in range(10)]
    hint, rep = P.build_hint(dets(frames), 19, 300, 600, BOUND, mode="every")
    check("front view, the other's wrists together at the belly: that fits too (hidden wrists are not "
          "required, DWPose may score them), so skipped", hint is None
          and rep["skipped"].startswith("could not tell who is bound"), rep["skipped"])
    frames = [[body(neck=(180.0, 100.0), arms=BEHIND_FRONT), body(neck=(440.0, 100.0), arms=BEHIND_FRONT)]
              for _ in range(10)]
    hint, rep = P.build_hint(dets(frames), 19, 300, 600, BOUND, mode="every")
    check("both with the hands behind the back: still skipped", hint is None and
          rep["skipped"].startswith("could not tell who is bound"), rep["skipped"])
    # The rules one by one, on single frames.
    def broken(arms, kp):
        g, _s = geom(kp)
        return P._arm_rule_broken(kp, g, 0, arms)
    check("rule: hanging arms (both wrists seen, shoulder width apart) break 'behind the back'",
          broken("behind the back", body()) and P._frame_fit(body(), geom(body())[0], 0,
                                                             P._spec(BOUND["Mara"])) == P.POSE_FIT_VIOLATION)
    check("rule: wrists hidden (cuffed behind, seen from the front) do not",
          not broken("behind the back", body(arms=BEHIND_FRONT)))
    check("rule: from behind, wrists together on the spine fit 'behind the back'",
          not broken("behind the back", body(view="back", arms=back_behind)))
    check("rule: wrists together at the belly fit 'in front of the body'",
          not broken("in front of the body", body(arms=together)))
    check("rule: holding something apart breaks 'in front of the body'", broken("in front of the body", body(arms=holding)))
    waist = {P.RELB: (0.5, 1.05 * 0.39), P.LELB: (0.5, -1.05 * 0.39), P.RWRI: (0.85, 0.55 * 0.39),
             P.LWRI: (0.85, -0.55 * 0.39)}
    check("rule: hands on hips (elbows flared, wrists apart) break 'at the waist'; the template does not",
          broken("at the waist", body(arms=hips)) and not broken("at the waist", body(arms=waist)))
    prof_hang = body(view="profile", width=0.1, fwd=1.0,
                     arms={P.RELB: (0.55, 0.0), P.LELB: (0.55, 0.0), P.RWRI: (1.05, 0.06), P.LWRI: None})
    prof_behind = body(view="profile", width=0.1, fwd=1.0,
                       arms={P.RELB: (0.55, -0.28), P.LELB: None, P.RWRI: (0.92, -0.22), P.LWRI: None})
    check("rule: in profile a seen wrist in front of the spine breaks 'behind the back', one behind it does not",
          broken("behind the back", prof_hang) and not broken("behind the back", prof_behind))
    check("rule: three-quarter views have no hard rule", not broken("behind the back", body(width=0.55)))


def test_carry_appearance():
    """report["appearance_last"] carries how the bound person looks; with it, a carried
    box counts only on someone who looks like them, in build_hint and identify_by_boxes."""
    print("\n=== carrying appearance across a cut ===")
    H, W = 300, 600
    frames = two_people(n=12)
    roles = [["B", "F"]] * 12
    d = with_appearance(frames, roles, H, W)
    hint, rep = P.build_hint(d, 23, H, W, BOUND, mode="every")
    v = rep["appearance_last"].get("Mara")
    check("appearance_last holds the bound person's vector (float32), close to their own",
          v is not None and v.dtype == np.float32 and P.appearance_distance(v, d["appearance"][-1][0]) < 0.05
          and P.appearance_distance(v, d["appearance"][-1][1]) > P.POSE_APP_DISTINCT_MIN, rep["skipped"])
    check("...and no appearance in, none out", P.build_hint(dets(frames), 23, H, W, BOUND, mode="every")[1]
          ["appearance_last"] == {})
    # Next shot: both stand still; the bound person's arms are off the template at the
    # opening (fit alone would not find her).
    raised = {P.RELB: (-0.2, 0.6), P.LELB: (-0.2, -0.6), P.RWRI: (-0.6, 0.5), P.LWRI: (-0.6, -0.5)}
    shot2 = [[body(neck=(180.0, 100.0), arms=raised), body(neck=(440.0, 100.0), arms=ARMS_OUT)] for _ in range(10)]
    d2 = with_appearance(shot2, [["B", "F"]] * 10, H, W)
    box = rep["boxes_last"]
    _h, rep2 = P.build_hint(d2, 19, H, W, BOUND, carry=box, carry_appearance=rep["appearance_last"], mode="every")
    check("carried box and appearance agree: identified across the cut",
          rep2["identified"]["Mara"] == 0 and any("across the cut" in x for x in rep2["notes"]), rep2["skipped"])
    d3 = with_appearance(shot2, [["G", "F"]] * 10, H, W)       # someone else now stands there
    _h, rep3 = P.build_hint(d3, 19, H, W, BOUND, carry=box, carry_appearance=rep["appearance_last"], mode="every")
    check("the carried box now holds someone who looks different: not identified across the cut",
          not any("identified across the cut" in x for x in rep3["notes"]) and rep3["identified"]["Mara"] is None
          and any("does not look like Mara" in x for x in rep3["notes"]), (rep3["notes"], rep3["skipped"]))
    check("...and the shot skips rather than asking fit instead",
          _h is None and rep3["skipped"] == "the carried box is on someone who does not look like Mara; "
                                            "not guessing", rep3["skipped"])
    # The one in the carried box looks different AND sits on the template, while the
    # bound person stands apart with raised arms: fit alone would pick the wrong one.
    shot4 = [[body(neck=(180.0, 100.0), arms=BEHIND_FRONT), body(neck=(440.0, 100.0), arms=raised)]
             for _ in range(10)]
    d4 = with_appearance(shot4, [["G", "B"]] * 10, H, W)
    h4, rep4 = P.build_hint(d4, 19, H, W, BOUND, carry=box, carry_appearance=rep["appearance_last"],
                            mode="every")
    check("a carried box on a different-looking person who fits the template: skipped, never drawn on them",
          h4 is None and rep4["identified"]["Mara"] is None
          and rep4["skipped"].startswith("the carried box is on someone who does not look like Mara"),
          (rep4["skipped"], rep4["identified"]))
    people = shot2[0]
    apps = d2["appearance"][0]
    got = P.identify_by_boxes(people, box, appearance=apps, carry_appearance=rep["appearance_last"])
    check("identify_by_boxes with appearance: the carried name matches when the appearance agrees",
          got == {"Mara": 0}, got)
    got = P.identify_by_boxes(people, box, appearance=d3["appearance"][0], carry_appearance=rep["appearance_last"])
    check("...and not when the person in the box looks different", got == {"Mara": None}, got)
    got = P.identify_by_boxes(people, box, appearance=[None, apps[1]], carry_appearance=rep["appearance_last"])
    check("...nor when that person has no vector to compare", got == {"Mara": None}, got)
    check("...and boxes alone still work as before", P.identify_by_boxes(people, box) == {"Mara": 0})
    import inspect
    sig = inspect.signature(P.build_hint).parameters
    check("build_hint: carry_appearance and latch_after are keyword-only",
          all(sig[k].kind is inspect.Parameter.KEYWORD_ONLY for k in ("cast_count", "carry_appearance", "latch_after")))
    sig = inspect.signature(P.identify_by_boxes).parameters
    check("identify_by_boxes: appearance and carry_appearance are keyword-only",
          all(sig[k].kind is inspect.Parameter.KEYWORD_ONLY for k in ("appearance", "carry_appearance")))


def test_latch():
    """A restraint that goes on during the shot: the latch limbs are drawn as detected
    until the first run of POSE_LATCH_RUN fitting frames at or after latch_after, then
    held; identification reads from there; nothing latched and nothing else to repair
    skips the shot."""
    print("\n=== latch mode ===")
    latch = {"Mara": {**BOUND["Mara"], "latch_limbs": ("arms",)}}
    free = {P.RELB: (0.5, 0.9), P.LELB: (0.5, -0.9), P.RWRI: (0.3, 1.4), P.LWRI: (0.3, -1.4)}

    # The other person walks up to her (frames 4-7) and back before her arms settle:
    # with others in the shot, the one a restraint closes on was come near.
    near_x = [440, 440, 400, 350, 300, 300, 300, 300, 350, 400]

    def shot(n=24, on=10, off=None, near=True):
        frames = []
        for i in range(n):
            arms = BEHIND_FRONT if (i >= on and (off is None or not off[0] <= i < off[1])) else free
            x = float(near_x[i]) if (near and i < len(near_x)) else 440.0
            frames.append([body(neck=(180.0, 100.0), arms=arms), body(neck=(x, 100.0), arms=ARMS_OUT)])
        return frames
    frames = shot()
    (skel, rep, _t) = P._hint_skeletons(dets(frames), 47, 300, 600, latch, mode="every", latch_after=12)
    check("the arms latch at the first analysed frame of their run on the template (frame 20)",
          rep["latched"] == {"Mara": 20}, (rep["latched"], rep["skipped"]))
    check("...identified from the latch run on, without a carry (nobody fits at the opening)",
          rep["identified"]["Mara"] == 0 and skel is not None, rep["skipped"])
    bound_at = lambda t: [k for k in skel[t] if abs(k[P.NECK, 0] - 180) < 1][0]
    check("...before it the arms are drawn as detected",
          skel is not None and np.allclose(bound_at(8)[[P.RELB, P.RWRI], :2], frames[4][0][[P.RELB, P.RWRI], :2])
          and bound_at(8)[P.RWRI, 2] >= P.POSE_CONF)
    check("...from it they are held (wrists left out from the front, elbows on the template)",
          skel is not None and bound_at(30)[P.RWRI, 2] < P.POSE_CONF
          and close(img(bound_at(30), P.RELB), (180 - 1.05 * 39, 155), 1.0), img(bound_at(30), P.RELB))
    # Repair mode counts breaks only from the latch frame on.
    hint, rep = P.build_hint(dets(frames), 47, 300, 600, latch, mode="repair", latch_after=12)
    check("repair mode: free arms before the latch are not a break (nothing to repair)",
          hint is None and not rep["broken"] and rep["skipped"] == "" and rep["latched"] == {"Mara": 20}, rep)
    frames = shot(off=(15, 19))
    hint, rep = P.build_hint(dets(frames), 47, 300, 600, latch, mode="repair", latch_after=12)
    check("...a break after the latch is repaired, its frames from the latch on only",
          hint is not None and rep["broken"] and rep["broken_frames"] and min(rep["broken_frames"]) >= 20,
          (rep["broken_frames"], rep["skipped"]))
    # Never latches.
    frames = shot(on=99)
    carry = {"Mara": P._torso_box(frames[0][0])}
    for c, lab in ((carry, "carried"), (None, "no carry")):
        hint, rep = P.build_hint(dets(frames), 47, 300, 600, latch, carry=c, mode="every", latch_after=12)
        check(f"the restraint never closes ({lab}): skipped, 'the restraint never closed in the first pass'",
              hint is None and rep["skipped"] == "the restraint never closed in the first pass",
              (rep["skipped"], rep["latched"]))
    # A latch before latch_after does not count: the run must start at or after it.
    frames = shot(on=4, off=(14, 30))
    hint, rep = P.build_hint(dets(frames), 47, 300, 600, latch, carry=carry, mode="every", latch_after=28)
    check("a run on the template only before latch_after does not latch",
          hint is None and rep["skipped"] == "the restraint never closed in the first pass", rep["skipped"])
    # Limbs not in latch_limbs are held from frame 0.
    both = {"Mara": {**BOUND["Mara"], "legs": "ankles together", "latch_limbs": ("arms",)}}
    apart = {P.RANK: (2.7, 0.4), P.LANK: (2.7, -0.4)}
    frames = [[body(neck=(180.0, 100.0), arms=BEHIND_FRONT if i >= 10 else free, legs=apart),
               body(neck=(440.0, 100.0), arms=ARMS_OUT)] for i in range(24)]
    skel, rep, _t = P._hint_skeletons(dets(frames), 47, 400, 600, both, carry=carry, mode="every", latch_after=12)
    k0 = [k for k in skel[0] if abs(k[P.NECK, 0] - 180) < 1][0] if skel else None
    check("legs not in latch_limbs are held from frame 0, the latch arms drawn as detected there",
          k0 is not None and abs(np.linalg.norm(img(k0, P.RANK) - img(k0, P.LANK)) - 12.0) < 1e-6
          and np.allclose(k0[P.RWRI, :2], frames[0][0][P.RWRI, :2]), rep["skipped"])
    # Someone posed on the template from the start (a guard, hands clasped behind) is not
    # the one the restraint goes on, whether or not the bound person's restraint closes.
    guard = BEHIND_FRONT
    frames = [[body(neck=(180.0, 100.0), arms=free), body(neck=(440.0, 100.0), arms=guard)] for _ in range(24)]
    hint, rep = P.build_hint(dets(frames), 47, 300, 600, latch, mode="every", latch_after=12)
    check("a guard posed from the start, the bound person never latching: skipped, nothing drawn on the guard",
          hint is None and rep["skipped"] == "the restraint never closed in the first pass"
          and any("from the start" in x for x in rep["notes"]), (rep["skipped"], rep["notes"]))
    # A guard who puts his hands behind him mid-shot, standing apart, while the bound
    # person's restraint never closes: nobody touched him, so he is not taken either.
    frames = [[body(neck=(180.0, 100.0), arms=free),
               body(neck=(440.0, 100.0), arms=guard if i >= 4 else ARMS_OUT)] for i in range(24)]
    hint, rep = P.build_hint(dets(frames), 47, 300, 600, latch, mode="every", latch_after=12)
    check("a guard settling into the pose mid-shot, apart from everyone: skipped, nothing drawn on him",
          hint is None and rep["skipped"] == "the restraint never closed in the first pass"
          and any("nobody near them" in x for x in rep["notes"]), (rep["skipped"], rep["notes"]))
    # The captor walks up, puts the restraint on and steps back; a guard stands posed from
    # the start. The one touched before her pose settled is identified.
    cap_x = [380, 380, 330, 280, 230, 230, 230, 230, 300, 380] + [380] * 14
    frames = [[body(neck=(150.0, 100.0), arms=BEHIND_FRONT if i >= 10 else free),
               body(neck=(float(cap_x[i]), 100.0), arms=ARMS_OUT),
               body(neck=(520.0, 100.0), arms=guard)] for i in range(24)]
    skel, rep, _t = P._hint_skeletons(dets(frames), 47, 300, 600, latch, mode="every", latch_after=12)
    check("...the bound person, come near by her captor before her pose settled: she is the one identified",
          skel is not None and rep["identified"]["Mara"] == 0 and rep["latched"] == {"Mara": 20},
          (rep["skipped"], rep["identified"], rep["latched"], rep["notes"]))
    # Alone, someone posed from the start is the bound person (cuffs described as on).
    frames = [[body(neck=(180.0, 100.0), arms=BEHIND_FRONT)] for _ in range(24)]
    skel, rep, _t = P._hint_skeletons(dets(frames), 47, 300, 600, latch, mode="every", latch_after=12)
    check("alone and posed from the start: identified, latched at the first run from latch_after",
          skel is not None and rep["identified"]["Mara"] == 0 and rep["latched"] == {"Mara": 12},
          (rep["skipped"], rep["identified"], rep["latched"]))
    # Without latch_after the latch limbs are ordinary held limbs.
    frames = shot(on=0, near=False)
    hint, rep = P.build_hint(dets(frames), 47, 300, 600, latch, mode="every")
    check("latch_limbs without latch_after: held from frame 0, latched reported as None",
          hint is not None and rep["latched"] == {"Mara": None}, rep["skipped"])


def _real_frame_people():
    """Two real people in one frame: the aux package's test photo, and beside it a mirror
    copy whose shirt and hair are recoloured red (a differently dressed person); plus the
    same pair with the copy only darker (the same person in other light). None when the
    photo is not there."""
    path = os.path.join(_COMFY_ROOT, "custom_nodes", "comfyui_controlnet_aux", "tests", "pose.png")
    try:
        from PIL import Image
        photo = np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None
    f = photo.astype(np.float32) / 255.0
    v = f.max(-1)
    s = np.where(v > 0, (v - f.min(-1)) / np.maximum(v, 1e-6), 0.0)
    cloth = (s < 0.22) & (v > 0.38)
    red = f.copy()
    red[cloth] = v[cloth][:, None] * np.array([1.0, 0.25, 0.18])
    red = (np.clip(red, 0, 1) * 255).astype(np.uint8)
    darker = np.clip(photo * 0.8, 0, 255).astype(np.uint8)
    pad = np.full((photo.shape[0], 60, 3), 90, np.uint8)
    return [np.concatenate([pad, photo, pad, other[:, ::-1], pad], axis=1) for other in (red, darker)]


# --------------------------------------------------------------------------------------
# Hint frames and drawing
# --------------------------------------------------------------------------------------

def test_frames_and_interpolation():
    print("\n=== hint frames and interpolation ===")
    a = body(neck=(100.0, 100.0))
    b = body(neck=(120.0, 100.0))
    b[P.LWRI, 2] = 0.0
    mid = P._fill({0: a, 1: b}, [0, 2], 1, None)
    check("an in-between frame is the average of its neighbours",
          np.allclose(mid[P.NECK, :2], (110.0, 100.0)), mid[P.NECK])
    check("a point only one neighbour has is left out", mid[P.LWRI, 2] < P.POSE_CONF)
    check("past the last analysed frame the skeleton is held", P._fill({0: a, 1: b}, [0, 2], 5, None) is b)
    check("...unless the hold is limited", P._fill({0: a, 1: b}, [0, 2], 9, 2) is None)
    frames = two_people(n=9)
    hint, rep = P.build_hint(dets(frames), 17, 300, 600, BOUND, mode="every")
    check("hint has exactly frame_count frames at H x W x 3", hint is not None
          and tuple(hint.shape) == (17, 300, 600, 3), None if hint is None else tuple(hint.shape))
    check("hint is float in 0..1", hint is not None and hint.dtype.is_floating_point
          and float(hint.min()) >= 0.0 and float(hint.max()) <= 1.0)
    check("every frame is drawn (none left black)",
          hint is not None and all(float(hint[t].max()) > 0 for t in range(17)))
    # Detections on a stride of 3 that miss the last frame: the end is held.
    d = {"index": [0, 3, 6, 9], "people": [two_people(n=1)[0]] * 4}
    hint, rep = P.build_hint(d, 12, 300, 600, BOUND, mode="every")
    check("frames after the last analysed one are filled too",
          hint is not None and float(hint[11].max()) > 0)
    # repair mode on a held pose: nothing to do
    hint, rep = P.build_hint(dets(two_people(n=9)), 17, 300, 600, BOUND, mode="repair")
    check("repair mode, pose held: no hint, nothing skipped", hint is None and not rep["broken"]
          and rep["skipped"] == "", rep)
    frames = two_people(n=12)
    raised = {P.RELB: (-0.2, 0.6), P.LELB: (-0.2, -0.6), P.RWRI: (-0.6, 0.5), P.LWRI: (-0.6, -0.5)}
    for i in (7, 8, 9, 10):
        frames[i][0] = body(neck=(180.0, 100.0), arms=raised)
    hint, rep = P.build_hint(dets(frames), 23, 300, 600, BOUND, mode="repair")
    check("repair mode, hands raised mid-shot: a hint, frames reported",
          hint is not None and rep["broken"] and rep["broken_frames"] == [14, 16, 18, 20],
          (rep["broken_frames"], rep["skipped"]))
    fall = {"Mara": {**BOUND["Mara"], "fall": True}}
    hint, rep = P.build_hint(dets(two_people(n=9)), 17, 300, 600, fall, mode="repair")
    check("repair mode, a bound fall: always a hint", hint is not None, rep)
    hint, rep = P.build_hint(dets(two_people(n=9)), 17, 300, 600, BOUND, mode="falls")
    check("falls mode without a bound fall: skipped", hint is None and "no bound fall" in rep["skipped"],
          rep["skipped"])


def test_draw_options():
    print("\n=== draw options ===")
    frames = two_people(n=9)
    everyone, _ = P.build_hint(dets(frames), 17, 300, 600, BOUND, mode="every", draw="everyone")
    only, _ = P.build_hint(dets(frames), 17, 300, 600, BOUND, mode="every", draw="bound person only")
    thick, _ = P.build_hint(dets(frames), 17, 300, 600, BOUND, mode="every", draw="everyone, thick lines")
    right = slice(330, 600)
    check("everyone: the free person is drawn", float(everyone[:, :, right].max()) > 0)
    check("bound person only: the rest of the canvas is black", float(only[:, :, right].max()) == 0.0)
    check("bound person only: the bound person is still drawn", float(only[:, :, :300].max()) > 0)
    lit = lambda h: int((h.amax(dim=-1) > 0).sum())
    check("thick lines draw wider sticks (600 px wide: xinsr scale 2)", lit(thick) > 1.3 * lit(everyone),
          (lit(thick), lit(everyone)))
    # The OpenPose convention: joints in their own colours, limbs at 60%.
    kp = body(neck=(150.0, 60.0))
    canvas = P._draw_bodypose_port(np.zeros((300, 300, 3), np.uint8),
                                   [P._KP(float(x), float(y), 1.0, i) for i, (x, y, _c) in enumerate(kp)])
    check("neck joint drawn in OpenPose colour 1 (255, 85, 0)", tuple(canvas[60, 150]) == (255, 85, 0),
          tuple(canvas[60, 150]))
    rs = kp[P.RSHO]
    check("right shoulder joint in colour 2 (255, 170, 0)",
          tuple(canvas[int(rs[1]), int(rs[0])]) == (255, 170, 0), tuple(canvas[int(rs[1]), int(rs[0])]))
    try:
        P._ensure_aux_path()
        from custom_controlnet_aux.dwpose.util import draw_bodypose
    except Exception as e:
        print(f"  NOTE  custom_controlnet_aux not importable ({type(e).__name__}); "
              f"aux drawer comparison skipped")
        return
    pts = [P._KP(float(x), float(y), 1.0, i) if c > 0 else None for i, (x, y, c) in enumerate(kp)]
    a = draw_bodypose(np.zeros((300, 300, 3), np.uint8), list(pts))
    b = P._draw_bodypose_port(np.zeros((300, 300, 3), np.uint8), list(pts))
    check("the fallback drawer is pixel-identical to aux draw_bodypose", np.array_equal(a, b))
    a = draw_bodypose(np.zeros((600, 1344, 3), np.uint8), list(pts), xinsr_stick_scaling=True)
    b = P._draw_bodypose_port(np.zeros((600, 1344, 3), np.uint8), list(pts), xinsr_stick_scaling=True)
    check("...and with xinsr stick scaling", np.array_equal(a, b))
    check("build_hint draws with aux's own draw_bodypose", P._drawer() is draw_bodypose)


# --------------------------------------------------------------------------------------
# Fall settle, blend-in, captor arms
# --------------------------------------------------------------------------------------

def fall_frame(f):
    """A bound fall caught on the hands, the hips fixed. f = 0 is upright; f = 1 has the
    torso pitched to 20 degrees from horizontal with the wrists on the floor line under
    the shoulders. In between, the keypoints move linearly (a fall over a few frames)."""
    up = body(neck=(500.0, 460.0), T=100.0)
    up[P.RKNE, :2], up[P.LKNE, :2] = (520.0, 645.0), (480.0, 645.0)
    up[P.RANK, :2], up[P.LANK, :2] = (520.0, 660.0), (480.0, 660.0)
    down = body(neck=(406.0, 526.0), T=100.0, phi=20.0)
    down[P.RKNE, :2], down[P.LKNE, :2] = (505.0, 650.0), (495.0, 650.0)
    down[P.RANK, :2], down[P.LANK, :2] = (600.0, 660.0), (590.0, 660.0)
    down[P.RWRI, :2], down[P.LWRI, :2] = (400.0, 650.0), (412.0, 650.0)
    f = min(1.0, max(0.0, f))
    return (1 - f) * up + f * down


FALL = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75] + [1.0] * 10       # down by analysed frame 6


def test_fall_settle():
    print("\n=== fall settle ===")
    seq = [fall_frame(f) for f in FALL]
    idx = list(range(0, 2 * len(seq), 2))
    out, applied = P._fall_settle(seq, idx)
    check("support frames found and the settle applied", applied)
    first = next(i for i in range(len(seq)) if not np.allclose(out[i], seq[i]))
    check("before the support frames nothing moves", first == 6, first)
    neck = [P._neck(k)[1] for k in out]
    check("the neck ramps down (part way after one analysed frame)",
          seq[6][P.NECK, 1] < neck[6] < 645 - 1, neck[6])
    check("...and settles at the floor line minus 0.15 T after the ramp", abs(neck[10] - 645.0) < 1.0, neck[10])
    hip = P._midhip(out[12])
    check("the hips stay where they were", close(hip, P._midhip(seq[12]), 1e-6))
    check("the torso length is kept (a turn, not a stretch)",
          abs(np.linalg.norm(P._neck(out[12]) - hip) - np.linalg.norm(P._neck(seq[12]) - P._midhip(seq[12]))) < 1.0)
    seq2 = seq + [fall_frame(0.0) for _ in range(8)]
    out2, _ = P._fall_settle(seq2, list(range(0, 2 * len(seq2), 2)))
    check("when pass 1 leaves support it ramps back out", np.allclose(out2[-1], seq2[-1]))
    fall = {"Mara": {"arms": "behind the back", "legs": "", "fall": True}}
    frames = [[k] for k in seq]
    # Pass 1's arms hang free from the opening, which no longer fits the restraint: the
    # identity comes across the cut, as it does in a render.
    carry = {"Mara": P._torso_box(seq[0])}
    skel, rep, _t = P._hint_skeletons(dets(frames), 2 * len(frames) - 1, 720, 800, fall, carry=carry,
                                      mode="repair")
    check("the bound-fall hint is built", skel is not None, rep["skipped"])
    check("...noting the settle", any("settled" in n for n in rep["notes"]), rep["notes"])
    if skel is not None:
        k = skel[24][0]
        check("arms are re-derived from the turned torso (wrists off the floor support)",
              k[P.RWRI, 2] < P.POSE_CONF or abs(k[P.RWRI, 1] - 650.0) > 20, k[P.RWRI])


def test_blend_in():
    print("\n=== blend in from a broken first frame ===")
    seq = [body(neck=(300.0, 120.0), arms=ARMS_OUT if i == 0 else BEHIND_FRONT) for i in range(12)]
    out, g = rewrite(seq, {"arms": "behind the back"}, blend=True)
    check("frame 0 keeps pass 1's arms", np.allclose(out[0][P.RWRI, :2], seq[0][P.RWRI, :2]))
    mid = P._arm_points(g, 2, "behind the back")[0][P.RELB]
    check("output frame 4 is halfway between pass 1 and the template",
          np.allclose(out[2][P.RELB, :2], 0.5 * seq[2][P.RELB, :2] + 0.5 * mid))
    check("by output frame 8 the arms are on the template",
          np.allclose(out[4][P.RELB, :2], P._arm_points(g, 4, "behind the back")[0][P.RELB]))
    frames = [[k, body(neck=(440.0, 120.0), arms=ARMS_OUT)] for k in seq]
    frames_ok = [[body(neck=(180.0, 100.0), arms=BEHIND_FRONT)] + f[1:] for f in frames]
    frames_ok[0][0] = body(neck=(180.0, 100.0), arms={P.RELB: (0.0, 0.8), P.LELB: (0.55, -0.41),
                                                     P.RWRI: (0.0, 1.3), P.LWRI: None})
    hint, rep = P.build_hint(dets(frames_ok), 23, 300, 600, BOUND, mode="every")
    check("a hint built from a broken opening frame notes the blend",
          any("blended" in n for n in rep["notes"]), (rep["notes"], rep["skipped"]))


def test_captor_arm_removed():
    """Another person's hand on the bound person's pass-1 wrist grabs empty air once the
    rewrite moves that wrist; the hand and its elbow are taken out for those frames."""
    print("\n=== captor arm removed ===")
    n = 12
    reach = {P.RELB: (0.55, 1.05 * 0.39), P.LELB: (0.3, -0.9), P.RWRI: None, P.LWRI: (0.3, -1.6)}
    frames = []
    for i in range(n):
        bound_arms = reach if i >= 8 else BEHIND_FRONT
        a = body(neck=(180.0, 100.0), arms=bound_arms)
        grab = a[P.LWRI, :2].copy() if i >= 8 else None
        other = body(neck=(440.0, 100.0))
        if grab is not None:
            other[P.RWRI, :2] = grab + (5.0, 0.0)
            other[P.RELB, :2] = (other[P.RSHO, :2] + grab) / 2
        frames.append([a, other])
    # The other person's hanging arms also fit the template: identity comes across the cut.
    carry = {"Mara": P._torso_box(frames[0][0])}
    skel, rep, _t = P._hint_skeletons(dets(frames), 23, 300, 600, BOUND, carry=carry, mode="every")
    check("hint built", skel is not None, rep["skipped"])
    if skel is None:
        return
    other_at = lambda t: [k for k in skel[t] if abs(k[P.NECK, 0] - 440) < 1][0]
    check("the other person's arms are drawn while nothing is moved",
          other_at(4)[P.RWRI, 2] >= P.POSE_CONF and other_at(4)[P.RELB, 2] >= P.POSE_CONF)
    o = other_at(18)
    check("their hand and elbow on the moved wrist are removed",
          o[P.RWRI, 2] < P.POSE_CONF and o[P.RELB, 2] < P.POSE_CONF, o[[P.RELB, P.RWRI], 2])
    check("their other arm stays", o[P.LWRI, 2] >= P.POSE_CONF)


# --------------------------------------------------------------------------------------
# Sigma window, setup check, encode, install
# --------------------------------------------------------------------------------------

def test_sigma_window():
    print("\n=== sigma window ===")
    sched8 = [1.0, 0.988, 0.973, 0.952, 0.923, 0.878, 0.8, 0.632, 0.0]      # 8 steps, 9 points
    s0, s1 = P.pose_sigma_window(sched8, 0.6)
    check("8 steps, pose_end 0.6: starts just above the first sigma", abs(s0 - 1.001) < 1e-9, s0)
    check("8 steps, pose_end 0.6: ends between 0.923 and 0.878", abs(s1 - 0.9005) < 1e-9, s1)
    active = [s for s in sched8[:-1] if s1 <= s <= s0]
    check("8 steps, pose_end 0.6: exactly the first 5 model calls are controlled", len(active) == 5, active)
    check("8 steps, pose_end 0.5: the first 4", sum(1 for s in sched8[:-1]
                                                    if P.pose_sigma_window(sched8, 0.5)[1] <= s) == 4)
    check("8 steps, pose_end 1.0: every call (s_end -1)", P.pose_sigma_window(sched8, 1.0)[1] == -1.0)
    check("8 steps, pose_end 0.1: at least one call",
          sum(1 for s in sched8[:-1] if P.pose_sigma_window(sched8, 0.1)[1] <= s) == 1)
    landed = [1.0, 0.988, 0.973, 0.952, 0.923, 0.878, 0.8, 0.632, 0.3, 0.0]  # 9 steps, 10 points
    s0, s1 = P.pose_sigma_window(landed, 0.6)
    check("a 10-point (9-step, landing) schedule, pose_end 0.6: first 6 calls",
          abs(s1 - (0.878 + 0.8) / 2) < 1e-9 and sum(1 for s in landed[:-1] if s1 <= s) == 6, s1)
    check("float edge: 0.7 of 10 steps is 7 calls, not 8",
          sum(1 for s in [1.0 - i * 0.1 for i in range(11)][:-1]
              if P.pose_sigma_window([1.0 - i * 0.1 for i in range(11)], 0.7)[1] <= s) == 7)
    s0, s1 = P.pose_sigma_window(torch.tensor(sched8), 0.6)
    check("a tensor schedule works the same", abs(s1 - 0.9005) < 1e-6, s1)


class MiniMaxH3FunControlBlockPatch:          # matched by class name, as comfy's is
    def __init__(self, previous=None):
        self.previous = previous


class _OtherPatch:
    def __init__(self, previous=None):
        self.previous = previous


def _ns_model(width):
    lin = SimpleNamespace(in_features=width)
    blk = SimpleNamespace(adaln_proj=SimpleNamespace(linear=lin))
    return SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(blocks=[blk])),
                           model_options={})


def _ns_cn(width):
    lin = SimpleNamespace(in_features=width)
    blk = SimpleNamespace(adaln_proj=SimpleNamespace(linear=lin))
    return SimpleNamespace(model=SimpleNamespace(injection_layers=(0, 10, 20, 30, 40),
                                                 control_blocks=[blk], init_stream=lambda *a: None))


def test_pose_status():
    print("\n=== setup check (pose_status) ===")
    saved = (P.dwpose_status, P._ensure_aux_path, P.dwpose_paths)
    try:
        P.dwpose_status = lambda: (True, "")
        check("unwired: off, silently", P.pose_status(_ns_model(8), None, 1.0, False) == (False, ""))
        check("strength 0: off, silently", P.pose_status(_ns_model(8), _ns_cn(8), 0.0, False) == (False, ""))
        ok, note = P.pose_status(_ns_model(8), SimpleNamespace(model=SimpleNamespace()), 1.0, False)
        check("not an H3 Fun patch: off with a note", not ok and "not a MiniMax H3 Fun" in note, note)
        ok, note = P.pose_status(_ns_model(16), _ns_cn(8), 1.0, False)
        check("16-wide base, 8-wide controlnet: off, naming both widths and the base it is for",
              not ok and "16" in note and "8-wide" in note and "built for the 8-wide hybrid b25-49 base" in note,
              note)
        ok, note = P.pose_status(_ns_model(2688), _ns_cn(8), 1.0, False)
        check("full-form 2688 base: off", not ok and "2688" in note, note)
        ok, note = P.pose_status(_ns_model(8), _ns_cn(8), 1.0, False)
        check("8 = 8 and everything present: on, no note", ok and note == "", note)
        stub = SimpleNamespace(model=SimpleNamespace(injection_layers=(0,), init_stream=None))
        ok, note = P.pose_status(SimpleNamespace(), stub, 1.0, False)
        check("widths unreadable (stubs): cannot tell -- on, with a note", ok and "could not read" in note, note)
        ok, note = P.pose_status(_ns_model(8), _ns_cn(8), 1.0, True)
        check("Hyperflow two-time on: off with a note", not ok and "Hyperflow" in note, note)
        P.dwpose_status = saved[0]
        P._ensure_aux_path = lambda: False
        ok, note = P.pose_status(_ns_model(8), _ns_cn(8), 1.0, False)
        check("aux not importable: off with a note", not ok and "custom_controlnet_aux" in note, note)
        P._ensure_aux_path = saved[1]
        fake = ("/nonexistent/ckpts/hr16/yolox-onnx/yolox_l.torchscript.pt",
                "/nonexistent/ckpts/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt")
        P.dwpose_paths = lambda: fake
        ok, note = P.pose_status(_ns_model(8), _ns_cn(8), 1.0, False)
        if _aux_ok():
            check("DWPose files missing: off, with both exact paths and the download",
                  not ok and fake[0] in note and fake[1] in note and "hf download" in note, note)
        else:
            check("DWPose files missing (aux absent here): off with the aux note",
                  not ok and "custom_controlnet_aux" in note, note)
        P.dwpose_paths = saved[2]
        P.dwpose_status = lambda: (True, "")
        m = _ns_model(8)
        m.model_options = {"transformer_options": {"patches_replace": {"dit": {
            ("double_block", 0): _OtherPatch(previous=MiniMaxH3FunControlBlockPatch())}}}}
        ok, note = P.pose_status(m, _ns_cn(8), 1.0, False)
        check("a Fun block patch already on the model (even under another): off with a note",
              not ok and "already on the incoming model" in note, note)
        m.model_options = {"transformer_options": {"patches_replace": {"dit": {
            ("double_block", 0): _OtherPatch()}}}}
        check("a non-Fun patch (VSA) alone does not count", P.pose_status(m, _ns_cn(8), 1.0, False)[0])

        class Boom:
            @property
            def model(self):
                raise RuntimeError("boom")
        ok, note = P.pose_status(Boom(), _ns_cn(8), 1.0, False)
        check("a model that raises on access: never raises, cannot tell", ok and "could not read" in note, note)
        ok, note = P.pose_status(_ns_model(8), _ns_cn(8), "x", False)
        check("a nonsense strength: off, silently", (ok, note) == (False, ""))
    finally:
        P.dwpose_status, P._ensure_aux_path, P.dwpose_paths = saved


def _aux_ok():
    try:
        return bool(P._ensure_aux_path())
    except Exception:
        return False


class _FakeVAE:
    def __init__(self, shape, oom=False, fail=False):
        self.shape, self.oom, self.fail = shape, oom, fail
        self.calls = []

    def encode(self, pix):
        self.calls.append(("encode", tuple(pix.shape)))
        if self.fail:
            raise ValueError("broken")
        if self.oom:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return torch.zeros(self.shape, dtype=torch.bfloat16)

    def encode_tiled(self, pix):
        self.calls.append(("encode_tiled", tuple(pix.shape)))
        return torch.zeros(self.shape)


def test_encode_hint():
    print("\n=== encode_hint ===")
    hint = torch.zeros((17, 64, 96, 3))
    target = (1, 24, 5, 4, 6)
    v = _FakeVAE(target)
    lat = P.encode_hint(v, hint, target)
    check("right shape: the latent comes back, float32 on the CPU",
          lat is not None and tuple(lat.shape) == target and lat.dtype == torch.float32 and lat.device.type == "cpu")
    check("the VAE is handed [F, H, W, 3] frames", v.calls == [("encode", (17, 64, 96, 3))], v.calls)
    v = _FakeVAE((1, 24, 4, 4, 6))
    check("shape mismatch: None, no exception", P.encode_hint(v, hint, target) is None)
    v = _FakeVAE((24, 5, 4, 6))
    check("a 4-D result: None", P.encode_hint(v, hint, target) is None)
    v = _FakeVAE(target, oom=True)
    lat = P.encode_hint(v, hint, target)
    check("OOM on encode: falls back to encode_tiled", lat is not None and [c[0] for c in v.calls]
          == ["encode", "encode_tiled"], v.calls)
    v = _FakeVAE(target, fail=True)
    check("another encode error: None, no exception", P.encode_hint(v, hint, target) is None)
    check("no VAE or no hint: None", P.encode_hint(None, hint, target) is None
          and P.encode_hint(_FakeVAE(target), None, target) is None)


class _FakePatcher:
    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.wrappers = []
        self.cloned_from = None

    def clone(self):
        # comfy's clone copies the nested patch dicts (create_model_options_clone)
        c = _FakePatcher()
        to = dict(self.model_options["transformer_options"])
        to["patches_replace"] = {k: dict(v) for k, v in to.get("patches_replace", {}).items()}
        c.model_options = {"transformer_options": to}
        c.cloned_from = self
        return c

    def add_wrapper(self, kind, fn):
        self.wrappers.append((kind, fn))

    def set_model_patch_replace(self, patch, name, block, index):
        to = self.model_options["transformer_options"]
        pr = to.setdefault("patches_replace", {})
        pr.setdefault(name, {})[(block, index)] = patch


def test_install_pose_control():
    print("\n=== install_pose_control (comfy's patch class) ===")
    if _COMFY_ROOT not in sys.path:
        sys.path.append(_COMFY_ROOT)
    try:
        import comfy.cli_args as ca
        ca.args.cpu = True
        import comfy_extras.nodes_minimax_h3 as H
    except Exception as e:
        print(f"  NOTE  comfy not importable here ({type(e).__name__}: {e}); install test skipped")
        return
    base = _FakePatcher()
    base.set_model_patch_replace(_OtherPatch(), "dit", "double_block", 0)      # VSA-like, before
    cn = SimpleNamespace(model=SimpleNamespace(injection_layers=(0, 10, 20, 30, 40)))
    shape = (1, 24, 5, 4, 6)
    lat = torch.ones(shape)
    vae = _FakeVAE(shape)
    m = P.install_pose_control(base, cn, vae, lat, shape, 1.0, 1.001, 0.9005)
    check("a clone comes back, the incoming model untouched", m is not base and m.cloned_from is base
          and len(base.model_options["transformer_options"]["patches_replace"]["dit"]) == 1)
    dit = m.model_options["transformer_options"]["patches_replace"]["dit"]
    blocks = sorted(i for (_b, i) in dit)
    check("block patches on the injection layers", blocks == [0, 10, 20, 30, 40], blocks)
    bp = dit[("double_block", 0)]
    check("they are comfy's MiniMaxH3FunControlBlockPatch", isinstance(bp, H.MiniMaxH3FunControlBlockPatch))
    check("the earlier patch on block 0 is chained as previous", isinstance(bp.previous, _OtherPatch))
    patch = bp.control_patch
    check("the control patch is comfy's MiniMaxH3FunControlPatch", isinstance(patch, H.MiniMaxH3FunControlPatch))
    check("strength and window as given", (patch.strength, patch.sigma_start, patch.sigma_end) == (1.0, 1.001, 0.9005))
    check("a diffusion-model wrapper is registered", len(m.wrappers) == 1)
    patch.prepare_control_latent(torch.Size(shape))
    check("the preset latent is used: the patch never encodes", vae.calls == [] and patch.control_latent is not None
          and tuple(patch.control_latent.shape) == shape, vae.calls)
    patch.cleanup()
    check("after cleanup the preset is still there for a retry",
          patch.control_latent is not None and patch.control_latent_shape == shape)


# --------------------------------------------------------------------------------------
# Optional: the real DWPose files on the CPU
# --------------------------------------------------------------------------------------

def test_detector_optional():
    print("\n=== DWPose on the CPU (optional) ===")
    det = P.PoseDetector(device="cpu")
    ok, note = det.available()
    if not ok:
        print(f"  NOTE  DWPose not available, skipped: {note}")
        return
    frames = torch.zeros((3, 256, 192, 3))
    canvas = P.render_skeletons([[body(neck=(96.0, 60.0), T=60.0)]], 256, 192)[0]
    frames[:] = canvas
    frames[..., :] += 0.2
    frames = frames.clamp(0, 1)
    try:
        out = det.detect(frames, stride=2)
        good = (out["index"] == [0, 2] and len(out["people"]) == 2
                and all(isinstance(pl, list) for pl in out["people"])
                and all(np.asarray(k).shape == (18, 3) for pl in out["people"] for k in pl))
        check("PoseDetector loads on the CPU and detects without error", good, out["index"])
        check("...on the jit modules' own device", all(
            next(m.parameters()).device.type == "cpu" for m in (det._est.det, det._est.pose)))
        det.close()
        check("close() leaves the modules on the CPU", all(
            next(m.parameters()).device.type == "cpu" for m in (det._est.det, det._est.pose)))
        out2 = det.detect(frames[:1], stride=2)
        check("detect after close() works", out2["index"] == [0])
        check("detect() gives an appearance list per analysed frame, aligned with the people",
              len(out["appearance"]) == len(out["people"])
              and all(len(a) == len(p) for a, p in zip(out["appearance"], out["people"])))
        real = _real_frame_people()
        if real is None:
            print("  NOTE  the aux test photo is not there; the real two-person check is skipped")
        else:
            apps = [det.detect(torch.from_numpy(np.ascontiguousarray(fr)).float().div(255.0)[None],
                               stride=1)["appearance"][0] for fr in real]
            ok = all(len(a) == 2 and all(v is not None and v.dtype == np.float32
                                         and v.shape == (P.POSE_APP_SIZE,) for v in a) for a in apps)
            check("a real frame with two people: an appearance vector for each", ok,
                  [[None if v is None else v.shape for v in a] for a in apps])
            if ok:
                d_diff, d_same = P.appearance_distance(*apps[0]), P.appearance_distance(*apps[1])
                check(f"...two differently dressed people are told apart: {d_diff:.2f} past "
                      f"POSE_APP_MATCH_MAX {P.POSE_APP_MATCH_MAX}", d_diff > P.POSE_APP_MATCH_MAX, d_diff)
                check(f"...the same person in darker light stays within it: {d_same:.2f}",
                      d_same < P.POSE_APP_MATCH_MAX, d_same)
        det.release()
    except Exception as e:
        check("PoseDetector loads on the CPU and detects without error", False, f"{type(e).__name__}: {e}")


def main():
    test_torso_frame()
    test_arm_templates_front_and_back()
    test_arm_templates_profile_and_blend()
    test_arm_templates_postures()
    test_legs()
    test_repair_check()
    test_identification()
    test_tracking()
    test_carry_helpers()
    test_track_assignment()
    test_reacquire()
    test_identification_contact()
    test_carry_contract()
    test_facing_without_face()
    test_review_low_items()
    test_appearance_vectors()
    test_appearance_gates()
    test_strict_geometry_without_appearance()
    test_identification_apart()
    test_carry_appearance()
    test_latch()
    test_frames_and_interpolation()
    test_draw_options()
    test_fall_settle()
    test_blend_in()
    test_captor_arm_removed()
    test_sigma_window()
    test_pose_status()
    test_encode_hint()
    test_install_pose_control()
    test_detector_optional()
    print()
    if _fails:
        print(f"RESULT: {len(_fails)} FAILURE(S): " + "; ".join(_fails))
    else:
        print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
