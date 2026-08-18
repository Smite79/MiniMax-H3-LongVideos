"""
PIL text overlays for H3 Long Videos -- watermark and intro title.

Text is COMPOSITED onto the decoded frames, never asked of the model. H3 (like
every video diffusion model) renders text as plausible-looking letterforms that
drift, warp and re-spell themselves frame to frame; a watermark that changes
shape every frame is worse than none. Compositing gives pixel-identical text on
every frame at zero sampling cost, and keeps the words out of the prompt where
they would otherwise steal conditioning from the actual shot.

Both overlays are WHITE text drawn on a fully transparent RGBA layer, then
alpha-blended over the video -- so only the glyphs themselves land on the frame
and the picture shows through everywhere else.

Everything here is best-effort: any failure returns the frames untouched with a
note, because a cosmetic overlay must never lose a finished render.
"""

import torch

BLEND_CHUNK = 64          # frames blended per slice -- bounds peak RAM on long chains

# Fonts to try when the requested one cannot be loaded. PIL resolves bare names
# against the system font directory, so "arial.ttf" works on Windows as-is.
FONT_FALLBACKS = ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")

# Anchor -> (x, y) as a fraction of the free space: 0 = hard against the left/top
# margin, 1 = hard against the right/bottom, 0.5 = centered.
POSITIONS = {
    "bottom-right":  (1.0, 1.0),
    "bottom-left":   (0.0, 1.0),
    "bottom-center": (0.5, 1.0),
    "top-right":     (1.0, 0.0),
    "top-left":      (0.0, 0.0),
    "top-center":    (0.5, 0.0),
    "center":        (0.5, 0.5),
    "lower-third":   (0.5, 0.72),
}


def _load_font(name, px):
    """A truetype font at px, falling back through the known-present faces and
    finally to PIL's bitmap default (which ignores size -- ugly, but never fatal)."""
    from PIL import ImageFont
    px = max(8, int(px))
    for cand in ([name] if name else []) + list(FONT_FALLBACKS):
        try:
            return ImageFont.truetype(cand, px)
        except Exception:
            continue
    return ImageFont.load_default()


def render_text_layer(width, height, text, font_px, position="bottom-right",
                      margin_pct=3.0, font_name="", stroke_px=0, line_spacing=1.15):
    """White text on a transparent RGBA canvas the size of one frame.

    Returns (rgb, alpha, bbox): rgb [H,W,3] float 0..1, alpha [H,W,1] float 0..1
    (zero everywhere except the glyphs and their optional stroke), and the tight
    (x0, y0, x1, y1) box of non-transparent pixels so the blend only has to touch
    the region the text actually occupies. None when there is nothing to draw."""
    from PIL import Image, ImageDraw
    import numpy as np

    text = (text or "").strip()
    if not text:
        return None

    img = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(font_name, font_px)
    stroke_px = max(0, int(stroke_px))
    spacing = int(max(0.0, font_px * (line_spacing - 1.0)))

    margin = int(min(width, height) * max(0.0, margin_pct) / 100.0)
    ax, ay = POSITIONS.get(position, POSITIONS["bottom-right"])

    # Measure first, so the block is placed by its real size rather than a guess.
    try:
        box = draw.multiline_textbbox((0, 0), text, font=font, align="center",
                                      stroke_width=stroke_px, spacing=spacing)
    except TypeError:                      # older Pillow: no stroke/spacing kwargs
        box = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    tw, th = box[2] - box[0], box[3] - box[1]

    free_w = max(0, int(width) - 2 * margin - tw)
    free_h = max(0, int(height) - 2 * margin - th)
    x = margin + free_w * ax - box[0]
    y = margin + free_h * ay - box[1]

    kwargs = dict(font=font, fill=(255, 255, 255, 255), align="center")
    if stroke_px:
        kwargs.update(stroke_width=stroke_px, stroke_fill=(0, 0, 0, 255))
    try:
        draw.multiline_text((x, y), text, spacing=spacing, **kwargs)
    except TypeError:
        draw.multiline_text((x, y), text, **kwargs)

    arr = np.asarray(img, dtype=np.float32) / 255.0        # [H, W, 4]
    alpha = arr[..., 3:4]
    if not alpha.any():
        return None
    # Tight bbox of drawn pixels: blending a whole 1344x768 frame for a corner
    # watermark would cost ~50x more work on a 3000-frame chain.
    ys, xs = np.nonzero(alpha[..., 0] > 0.0)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return (torch.from_numpy(arr[..., :3].copy()),
            torch.from_numpy(alpha.copy()),
            bbox)


def blend_layer(frames, layer, frame_alpha=None, opacity=1.0):
    """Alpha-composite a rendered layer over frames [N,H,W,3] in 0..1, in place.

    frame_alpha is an optional per-frame multiplier (length N) -- that is what
    makes an intro title hold and then fade instead of sitting on the whole
    video. Frames whose multiplier is 0 are skipped entirely."""
    if layer is None:
        return frames
    rgb, alpha, (x0, y0, x1, y1) = layer
    n = frames.shape[0]
    if frame_alpha is None:
        frame_alpha = torch.ones(n, dtype=torch.float32)
    frame_alpha = frame_alpha.to(torch.float32).clamp(0.0, 1.0) * float(opacity)

    a_crop = alpha[y0:y1, x0:x1, :].to(frames.dtype)
    c_crop = rgb[y0:y1, x0:x1, :].to(frames.dtype)
    live = (frame_alpha > 0).nonzero().flatten().tolist()
    for s in range(0, len(live), BLEND_CHUNK):
        idx = live[s:s + BLEND_CHUNK]
        fa = frame_alpha[idx].to(frames.dtype).view(-1, 1, 1, 1)
        sub = frames[idx, y0:y1, x0:x1, :]
        a = a_crop * fa
        frames[idx, y0:y1, x0:x1, :] = sub * (1.0 - a) + c_crop * a
    return frames


def hold_fade_alpha(total_frames, hold_frames, fade_frames):
    """Per-frame opacity for an intro: full through hold_frames, then a linear
    ramp to zero over fade_frames, then nothing. Returns a length-N tensor."""
    a = torch.zeros(int(total_frames), dtype=torch.float32)
    hold = max(0, min(int(hold_frames), int(total_frames)))
    a[:hold] = 1.0
    fade = max(0, min(int(fade_frames), int(total_frames) - hold))
    if fade:
        a[hold:hold + fade] = torch.linspace(1.0, 0.0, fade + 2)[1:-1]
    return a


def apply_overlays(frames, fps, watermark="", wm_position="bottom-right", wm_size_pct=4.0,
                   wm_opacity=0.75, wm_margin_pct=3.0, intro="", intro_seconds=3.0,
                   intro_fade=0.6, intro_size_pct=9.0, intro_position="center",
                   font_name="", stroke_px=0):
    """Composite the watermark (every frame) and the intro title (first seconds
    only, then faded out). Returns (frames, note). Never raises -- a cosmetic
    overlay must not be able to destroy a finished render."""
    notes = []
    if frames is None or frames.ndim != 4 or frames.shape[0] == 0:
        return frames, ""
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    frames = frames.contiguous()

    if (watermark or "").strip():
        try:
            layer = render_text_layer(w, h, watermark, h * max(0.5, wm_size_pct) / 100.0,
                                      wm_position, wm_margin_pct, font_name, stroke_px)
            if layer is not None:
                blend_layer(frames, layer, None, wm_opacity)
                notes.append(f"watermark composited ({wm_position}, {wm_opacity:.0%})")
        except Exception as e:
            notes.append(f"watermark skipped ({type(e).__name__}: {e})")

    if (intro or "").strip():
        try:
            layer = render_text_layer(w, h, intro, h * max(0.5, intro_size_pct) / 100.0,
                                      intro_position, 6.0, font_name, stroke_px)
            if layer is not None:
                hold = round(max(0.0, float(intro_seconds)) * fps)
                fade = round(max(0.0, float(intro_fade)) * fps)
                fa = hold_fade_alpha(n, hold, fade)
                if fa.max() > 0:
                    blend_layer(frames, layer, fa, 1.0)
                    notes.append(f"intro title composited ({hold}f hold + {fade}f fade)")
                else:
                    notes.append("intro title skipped (no hold or fade frames)")
        except Exception as e:
            notes.append(f"intro title skipped ({type(e).__name__}: {e})")

    return frames, "; ".join(notes)
