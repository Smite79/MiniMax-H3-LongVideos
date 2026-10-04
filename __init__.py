# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

from .sampler import NODE_CLASS_MAPPINGS as _s_c, NODE_DISPLAY_NAME_MAPPINGS as _s_d
from .shot_length import NODE_CLASS_MAPPINGS as _sl_c, NODE_DISPLAY_NAME_MAPPINGS as _sl_d
from .overlay import NODE_CLASS_MAPPINGS as _o_c, NODE_DISPLAY_NAME_MAPPINGS as _o_d

NODE_CLASS_MAPPINGS = {**_s_c, **_sl_c, **_o_c}
NODE_DISPLAY_NAME_MAPPINGS = {**_s_d, **_sl_d, **_o_d}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
