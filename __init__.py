# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79
# SPDX-License-Identifier: GPL-3.0-or-later. See LICENSE.

from .node import NODE_CLASS_MAPPINGS as _n_c, NODE_DISPLAY_NAME_MAPPINGS as _n_d
from .shot_length import NODE_CLASS_MAPPINGS as _sl_c, NODE_DISPLAY_NAME_MAPPINGS as _sl_d
from .overlay import NODE_CLASS_MAPPINGS as _o_c, NODE_DISPLAY_NAME_MAPPINGS as _o_d

NODE_CLASS_MAPPINGS = {**_n_c, **_sl_c, **_o_c}
NODE_DISPLAY_NAME_MAPPINGS = {**_n_d, **_sl_d, **_o_d}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
