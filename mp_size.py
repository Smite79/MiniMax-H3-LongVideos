"""
H3 Megapixel Size  (pixel budget -> width/height)
=================================================
Sizes a render by a PIXEL BUDGET instead of by a fixed short edge, and emits
`width` / `height` for any node that takes them -- ComfyUI's own
`MiniMaxH3EmptyLatent` and friends take plain width/height ints at step 32.

Why a budget rather than a short edge: cost and training-distribution match are
functions of TOKEN COUNT -- (h/16)*(w/16)*frames -- which tracks total pixels.
A short-edge target does not, and the two disagree at the extremes of aspect
ratio:

    1:1  768x768    short edge 768, reads native      -> 0.56 MP, well under
    21:9 1536x672   short edge 672, reads sub-native  -> 0.98 MP, full budget

So a square preset that looks native is in fact 43% under budget, and an
ultra-wide that looks starved is at full budget. Holding megapixels constant
keeps VRAM and token count put when you change shape, which is what makes two
aspect ratios comparable at all.

Convention: 1 MP = 1024 x 1024 = 1,048,576 px, matching ComfyUI's own
`Scale Image to Total Pixels` node, so the same number means the same size
across the graph.

Method: constant-area square root. scale = sqrt(target_area / current_area),
applied to both axes, then snapped to `multiple` (32 for H3). Snapping moves the
real area slightly off the request, so `info` and the `megapixels` output report
what was ACTUALLY produced, never what was asked for.

This node does not touch the sampler. The H3 Long Videos node keeps its own
resolution dropdown; use this when you want a budget-sized latent, or to size
the core H3 nodes.
"""

import math

# 1 MP == 1024x1024, the same convention ComfyUI's own scaling node uses.
MP_UNIT = 1024 * 1024

# H3's spatial patch grid is 16; the core H3 nodes expose width/height at step 32,
# which is a safe multiple of it.
H3_MULTIPLE = 32

# H3's own preset shapes come FIRST, given as their literal dimensions so the ratio
# is exact. Their conventional names are approximations: 1344x768 is 1.750, which is
# 7:4, NOT 16:9 (1.778), and 1536x672 is 2.286, which is 16:7, not 21:9 (2.333).
# Computing from the nominal ratio therefore does NOT reproduce the native size --
# true 16:9 at 1.00MP lands on 1376x768. Both are offered: pick an "H3" entry to
# stay on a shape the model was actually trained on, or a true ratio when the
# geometry matters more than the preset.
ASPECTS = {
    "H3 wide (1344x768)": (1344, 768),
    "H3 tall (768x1344)": (768, 1344),
    "H3 4:3 (1024x768)": (1024, 768),
    "H3 3:4 (768x1024)": (768, 1024),
    "H3 square (768x768)": (1, 1),
    "H3 ultrawide (1536x672)": (1536, 672),
    "true 16:9": (16, 9),
    "true 9:16": (9, 16),
    "true 4:3": (4, 3),
    "true 3:4": (3, 4),
    "true 1:1": (1, 1),
    "true 21:9": (21, 9),
    "true 9:21": (9, 21),
    "true 3:2": (3, 2),
    "true 2:3": (2, 3),
    "custom": None,
}


def size_for_budget(aspect_w, aspect_h, megapixels, multiple=H3_MULTIPLE):
    """(width, height) hitting `megapixels` at the given ratio, snapped to `multiple`.

    Both axes are snapped independently, which is what keeps every result legal
    for the model; the cost is that the achieved ratio can differ from the request
    by up to half a step on each axis. At 32 on H3-sized images that is well under
    a percent, and a legal size that is a hair off ratio beats an exact ratio the
    model cannot take."""
    aspect_w = max(1e-6, float(aspect_w))
    aspect_h = max(1e-6, float(aspect_h))
    mp = max(0.001, float(megapixels))
    multiple = max(1, int(multiple))
    # Solve for the side lengths of a rectangle with this ratio and this area.
    unit = math.sqrt((mp * MP_UNIT) / (aspect_w * aspect_h))
    w = max(multiple, int(round(aspect_w * unit / multiple)) * multiple)
    h = max(multiple, int(round(aspect_h * unit / multiple)) * multiple)
    return w, h


class H3MegapixelSize:
    CATEGORY = "MiniMax-H3/utils"
    FUNCTION = "emit"
    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "megapixels", "info")
    DESCRIPTION = ("Size a render by a pixel budget instead of a short edge. "
                   "Outputs width/height snapped to a multiple of 32. "
                   "1 MP = 1024x1024.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.01,
                    "tooltip": "Pixel budget. 1.00 = 1024x1024 worth of pixels, the same "
                               "convention as ComfyUI's Scale Image to Total Pixels. START here: "
                               "at 1.00 every 'H3' aspect lands on that preset's native size "
                               "(1344x768, 1536x672, ...), then step down for speed, VRAM and "
                               "longer shots -- 0.83 gives a 704 short edge, 0.65 gives 640. "
                               "Holding this constant keeps VRAM and token count put when you "
                               "change aspect ratio."}),
                "aspect": (list(ASPECTS.keys()), {"default": "H3 wide (1344x768)",
                    "tooltip": "Shape only -- the budget decides the size. The 'H3' entries are the "
                               "model's own preset shapes, given exactly: 1344x768 is 1.750 (7:4), "
                               "NOT 16:9 (1.778), and 1536x672 is 16:7, not 21:9. Pick one of those "
                               "to stay on a shape H3 was trained on; pick a 'true' ratio when the "
                               "geometry matters more. 'custom' reads custom_aspect_w/h as a RATIO, "
                               "not as pixels."}),
            },
            "optional": {
                "custom_aspect_w": ("INT", {"default": 16, "min": 1, "max": 512,
                    "tooltip": "Only used when aspect is 'custom'. A ratio term, e.g. 21."}),
                "custom_aspect_h": ("INT", {"default": 9, "min": 1, "max": 512,
                    "tooltip": "Only used when aspect is 'custom'. A ratio term, e.g. 9."}),
                "multiple": ("INT", {"default": H3_MULTIPLE, "min": 8, "max": 128, "step": 8,
                    "tooltip": "Snap both axes to this. 32 for H3 (its patch grid is 16, and the "
                               "core H3 nodes step width/height by 32). Only lower it if you know "
                               "the model you are feeding accepts it."}),
            },
        }

    def emit(self, megapixels, aspect, custom_aspect_w=16, custom_aspect_h=9,
             multiple=H3_MULTIPLE):
        ratio = ASPECTS.get(aspect)
        if ratio is None:                       # 'custom', or an unknown label
            aw, ah = max(1, int(custom_aspect_w)), max(1, int(custom_aspect_h))
        else:
            aw, ah = ratio
        w, h = size_for_budget(aw, ah, megapixels, multiple)
        actual = (w * h) / MP_UNIT
        # Report the ACHIEVED budget and ratio, not the requested ones: snapping
        # moves both, and a readout of what you asked for hides what you got.
        info = (f"{aspect} @ {float(megapixels):.2f}MP requested -> {w}x{h} "
                f"({actual:.3f}MP actual, ratio {w / h:.3f} vs {aw / ah:.3f}, "
                f"multiple of {int(multiple)})")
        return (w, h, round(actual, 4), info)


NODE_CLASS_MAPPINGS = {"H3MegapixelSize": H3MegapixelSize}
NODE_DISPLAY_NAME_MAPPINGS = {"H3MegapixelSize": "H3 Megapixel Size"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
