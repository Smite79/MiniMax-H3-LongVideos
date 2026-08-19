"""
H3-LongVideos-V1
================
Make long (up to ~120s) MiniMax-H3 videos from a single prompt + a single
length, in ComfyUI.

Nodes:
  * H3 Long Videos FL2VA  (sampler.py)      - one prompt + shot length -> video+audio
                                              (set plan_only=True to PREVIEW the shot
                                              split using the node's own settings, no
                                              render -- replaces the old Plan node)
  * H3 Shot Length        (shot_length.py)  - one shot length as seconds AND a valid
                                              H3 frame count (17k+5 grid, 362 cap);
                                              wire `seconds` -> the sampler and
                                              `frames` -> a preview override
  * H3 Model Inspector    (inspector.py)    - report base precision (BF16/FP8/NVFP4/MXFP8)
                                              and whether this card runs it natively
  * H3 Long Videos REF2VA (REF2VA/)         - variant sampler, developed alongside FL2VA
                                              and registered under its own key; shares
                                              the two utility nodes above

Install: put this whole folder in ComfyUI/custom_nodes/ and restart ComfyUI.
"""

from .sampler import NODE_CLASS_MAPPINGS as _s_c, NODE_DISPLAY_NAME_MAPPINGS as _s_d
from .shot_length import NODE_CLASS_MAPPINGS as _sl_c, NODE_DISPLAY_NAME_MAPPINGS as _sl_d
from .inspector import NODE_CLASS_MAPPINGS as _i_c, NODE_DISPLAY_NAME_MAPPINGS as _i_d

# REF2VA is a NESTED folder, so ComfyUI never imports it itself -- only top-level
# entries of custom_nodes/ are scanned. It is pulled in from here instead. A
# failure to import it must not take the whole pack down with it: while the
# variant is under development a syntax error in there would otherwise turn every
# node in this pack into a red "missing node" box.
try:
    from .REF2VA import (NODE_CLASS_MAPPINGS as _r_c,
                         NODE_DISPLAY_NAME_MAPPINGS as _r_d)
except Exception as _e:                      # pragma: no cover - dev-time guard
    print(f"[H3-LongVideos] REF2VA not loaded ({type(_e).__name__}: {_e})")
    _r_c, _r_d = {}, {}

NODE_CLASS_MAPPINGS = {**_s_c, **_sl_c, **_i_c, **_r_c}
NODE_DISPLAY_NAME_MAPPINGS = {**_s_d, **_sl_d, **_i_d, **_r_d}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
