# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79
# SPDX-License-Identifier: GPL-3.0-or-later. See LICENSE.

import contextlib

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from . import chain
except ImportError:
    import chain

CHUNK, MIN_FONT_PX, FIT_SHRINK, FIT_STEPS, LINE_SPACING = 64, 8, 0.92, 48, 1.15
FONTS = ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")
POSITIONS = {"bottom-right": (1.0, 1.0), "bottom-left": (0.0, 1.0), "bottom-center": (0.5, 1.0), "top-right": (1.0, 0.0),
             "top-left": (0.0, 0.0), "top-center": (0.5, 0.0), "center": (0.5, 0.5), "lower-third": (0.5, 0.72)}


def _font(name, px):
    for cand in ((name,) if name else ()) + FONTS:
        with contextlib.suppress(Exception):
            return ImageFont.truetype(cand, px)
    return ImageFont.load_default()


def _box(draw, text, font, stroke, spacing):
    return draw.multiline_textbbox((0, 0), text, font=font, align="center", stroke_width=stroke, spacing=spacing)


def _wrap(draw, text, font, max_w, stroke, spacing):
    out = []
    for hard in text.split("\n"):
        cur, *rest = hard.split() or [""]
        for word in rest:
            box = _box(draw, f"{cur} {word}", font, stroke, spacing)
            if box[2] - box[0] <= max_w:
                cur = f"{cur} {word}"
            else:
                out.append(cur)
                cur = word
        out.append(cur)
    return "\n".join(out)


def _layer(width, height, text, px, position, margin_pct, font_name, stroke):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw, stroke = ImageDraw.Draw(img), max(0, int(stroke))
    margin = int(min(width, height) * max(0.0, margin_pct) / 100.0)
    max_w, max_h = max(1, width - 2 * margin), max(1, height - 2 * margin)
    px = max(MIN_FONT_PX, int(px))
    for _ in range(FIT_STEPS):
        font, spacing = _font(font_name, px), int(max(0.0, px * (LINE_SPACING - 1.0)))
        fitted = _wrap(draw, text, font, max_w, stroke, spacing)
        box = _box(draw, fitted, font, stroke, spacing)
        if box[2] - box[0] <= max_w and box[3] - box[1] <= max_h or px <= MIN_FONT_PX:
            break
        px = max(MIN_FONT_PX, int(px * FIT_SHRINK))
    ax, ay = POSITIONS.get(position, POSITIONS["bottom-right"])
    x = margin + max(0, width - 2 * margin - (box[2] - box[0])) * ax - box[0]
    y = margin + max(0, height - 2 * margin - (box[3] - box[1])) * ay - box[1]
    draw.multiline_text((x, y), fitted, font=font, fill=(255, 255, 255, 255), align="center", spacing=spacing, stroke_width=stroke,
                        stroke_fill=(0, 0, 0, 255))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    ys, xs = np.nonzero(arr[..., 3] > 0.0)
    if not len(ys):
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return torch.from_numpy(arr[y0:y1, x0:x1, :3].copy()), torch.from_numpy(arr[y0:y1, x0:x1, 3:].copy()), (x0, y0, x1, y1)


def _blend(frames, layer, alpha):
    rgb, a, (x0, y0, x1, y1) = layer
    rgb, a = rgb.to(frames.dtype), a.to(frames.dtype)
    live = (alpha > 0).nonzero().flatten().tolist()
    for s in range(0, len(live), CHUNK):
        idx = live[s:s + CHUNK]
        k = a * alpha[idx].to(frames.dtype).view(-1, 1, 1, 1)
        frames[idx, y0:y1, x0:x1, :] = frames[idx, y0:y1, x0:x1, :] * (1.0 - k) + rgb * k


def _draw(frames, text, size_pct, position, margin_pct, font_name, stroke, alpha):
    h, w = int(frames.shape[1]), int(frames.shape[2])
    layer = _layer(w, h, text.strip(), min(w, h) * max(0.5, size_pct) / 100.0, position, margin_pct, font_name, stroke)
    if layer is not None:
        _blend(frames, layer, alpha)
    return layer is not None


def apply_overlays(frames, fps, watermark="", wm_position="bottom-right", wm_size_pct=4.0, wm_opacity=0.75, wm_margin_pct=3.0, intro="",
                   intro_seconds=3.0, intro_fade=0.6, intro_size_pct=9.0, intro_position="center", font_name="", stroke_px=0):
    if frames is None or frames.ndim != 4 or frames.shape[0] == 0:
        return frames, ""
    frames, notes, n = frames.contiguous(), [], int(frames.shape[0])
    if (watermark or "").strip():
        try:
            opacity = torch.full((n,), float(wm_opacity), dtype=torch.float32)
            if _draw(frames, watermark, wm_size_pct, wm_position, wm_margin_pct, font_name, stroke_px, opacity):
                notes.append(f"watermark composited ({wm_position}, {wm_opacity:.0%})")
        except Exception as e:
            notes.append(f"watermark skipped ({type(e).__name__}: {e})")
    if (intro or "").strip():
        try:
            hold, fade = (round(max(0.0, float(s)) * fps) for s in (intro_seconds, intro_fade))
            alpha, ramp = torch.zeros(n, dtype=torch.float32), min(fade, n - min(hold, n))
            alpha[:hold] = 1.0
            alpha[hold:hold + ramp] = torch.linspace(1.0, 0.0, ramp + 2)[1:-1]
            if _draw(frames, intro, intro_size_pct, intro_position, 6.0, font_name, stroke_px, alpha):
                notes.append(f"intro title composited ({hold}f hold + {fade}f fade)" if hold or fade else
                             "intro title skipped (no hold or fade frames)")
        except Exception as e:
            notes.append(f"intro title skipped ({type(e).__name__}: {e})")
    return frames, "; ".join(notes)


def copy_frames(images):
    out = chain.spool(images.shape[0], tuple(images.shape[1:]), images.dtype)
    for s in range(0, int(images.shape[0]), CHUNK):
        out[s:s + CHUNK].copy_(images[s:s + CHUNK])
    return out


class H3Overlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120,
                                "tooltip": "Only used to convert the intro's SECONDS into frames. Match your save node; H3 always "
                                           "renders 24."}),
            },
            "optional": {
                "watermark_text": ("STRING", {"default": "",
                                              "tooltip": "Composited onto every frame with PIL -- never asked of the model, which would "
                                                         "render drifting, re-spelling letterforms. Empty disables it."}),
                "watermark_position": ([p for p in POSITIONS if p != "lower-third"], {"default": "bottom-right"}),
                "watermark_size": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 40.0, "step": 0.5,
                                             "tooltip": "Cap height as a percentage of the frame's SHORT edge, so it looks the same at "
                                                        "every aspect ratio."}),
                "watermark_opacity": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "watermark_margin": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 25.0, "step": 0.5,
                                               "tooltip": "Inset from the frame edge, as a percentage of the short edge."}),
                "intro_text": ("STRING", {"multiline": True, "default": "",
                                          "tooltip": "Title card held over the opening frames, then faded out. Empty disables it."}),
                "intro_position": (["center", "lower-third", "top-center", "bottom-center"], {"default": "center"}),
                "intro_seconds": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.5,
                                            "tooltip": "How long the title holds at full opacity before the fade."}),
                "intro_fade": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 10.0, "step": 0.1,
                                         "tooltip": "Fade-out length in seconds, after the hold."}),
                "intro_size": ("FLOAT", {"default": 9.0, "min": 1.0, "max": 40.0, "step": 0.5,
                                         "tooltip": "Cap height as a percentage of the short edge."}),
                "overlay_font": ("STRING", {"default": "arial.ttf",
                                            "tooltip": "TrueType font for BOTH overlays: a bare name resolved against the system font "
                                                       "folder (arial.ttf, arialbd.ttf, segoeui.ttf) or a full path. Falls back to the "
                                                       "first font that loads."}),
                "overlay_stroke": ("INT", {"default": 0, "min": 0, "max": 20,
                                           "tooltip": "Black outline thickness around the white text. 0 keeps it pure white; 2-3 makes "
                                                      "it survive a bright sky or a white wall."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "run"
    CATEGORY = "MiniMax-H3/utils"
    DESCRIPTION = ("Composite a watermark and/or an intro title onto finished frames. Put it AFTER any upscale, so glyphs are rasterized "
                   "at the final pixel size instead of being interpolated up with the picture.")

    def run(self, images, fps=24, watermark_text="", watermark_position="bottom-right", watermark_size=4.0, watermark_opacity=0.75,
            watermark_margin=3.0, intro_text="", intro_position="center", intro_seconds=3.0, intro_fade=0.6, intro_size=9.0,
            overlay_font="arial.ttf", overlay_stroke=0):
        if not ((watermark_text or "").strip() or (intro_text or "").strip()):
            return images.detach().cpu(), "no overlay text given -- frames unchanged"
        frames, note = apply_overlays(copy_frames(images.detach()), max(1, int(fps)), watermark_text, watermark_position, watermark_size,
                                      watermark_opacity, watermark_margin, intro_text, intro_seconds, intro_fade, intro_size,
                                      intro_position, overlay_font, overlay_stroke)
        return frames, note or "overlays applied"


NODE_CLASS_MAPPINGS = {"H3Overlay": H3Overlay}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Overlay": "H3 Overlay (watermark + title)"}
