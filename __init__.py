"""
H3-LongVideos
=============
Make long (up to ~120s) MiniMax-H3 videos from a single prompt + a single
length, in ComfyUI.

Nodes:
  * H3 Long Videos     (sampler.py)      - one prompt + shot length -> video+audio,
                                           covering BOTH H3 conditioning tasks:
                                           FL2VA (first/last frame anchors the shot)
                                           and REF2VA (reference images condition on
                                           what a character looks like). Connect no
                                           ref_image_* and it is pure FL2VA.
                                           (set plan_only=True to PREVIEW the shot
                                           split using the node's own settings, no
                                           render -- replaces the old Plan node)
  * H3 Shot Length     (shot_length.py)  - one shot length as seconds AND a valid
                                           H3 frame count (17k+5 grid, 362 cap);
                                           wire `seconds` -> the sampler and
                                           `frames` -> a preview override
  * H3 Model Inspector (inspector.py)    - report base precision (BF16/FP8/NVFP4/MXFP8)
                                           and whether this card runs it natively
  * H3 Megapixel Size  (mp_size.py)      - a pixel BUDGET (1 MP = 1024x1024) plus an
                                           aspect ratio -> width/height snapped to 32.
                                           Cost and training fit track token count,
                                           which follows total pixels rather than the
                                           short edge, so holding MP constant is what
                                           makes two aspect ratios comparable. Feeds
                                           any node taking width/height; the sampler
                                           keeps its own resolution dropdown.

The sampler registers under four keys -- H3LongVideos, H3LongVideosFL2VA,
H3LongVideosV1 and H3LongVideosREF2VA -- all aliases onto the same class, so every
workflow saved under any previous name keeps loading. REF2VA was briefly a second,
duplicated sampler; it was 94% the same file, every shared fix had to be made
twice, and one such mirror silently emitted a clause twice. It is now folded in.

Install: put this whole folder in ComfyUI/custom_nodes/ and restart ComfyUI.
"""

from .sampler import NODE_CLASS_MAPPINGS as _s_c, NODE_DISPLAY_NAME_MAPPINGS as _s_d
from .shot_length import NODE_CLASS_MAPPINGS as _sl_c, NODE_DISPLAY_NAME_MAPPINGS as _sl_d
from .inspector import NODE_CLASS_MAPPINGS as _i_c, NODE_DISPLAY_NAME_MAPPINGS as _i_d
from .mp_size import NODE_CLASS_MAPPINGS as _mp_c, NODE_DISPLAY_NAME_MAPPINGS as _mp_d

NODE_CLASS_MAPPINGS = {**_s_c, **_sl_c, **_i_c, **_mp_c}
NODE_DISPLAY_NAME_MAPPINGS = {**_s_d, **_sl_d, **_i_d, **_mp_d}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
