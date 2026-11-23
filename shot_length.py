# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

H3_MAX_FRAMES = 362


class H3ShotLength:
    CATEGORY = "MiniMax-H3/utils"
    FUNCTION = "emit"
    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("seconds", "frames", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_seconds": ("FLOAT", {"default": 5.0, "min": 0.2, "max": 15.1, "step": 0.5,
                                           "tooltip": "Length of each shot. Feeds the sampler's shot_seconds AND (as frames) "
                                                      "the preview override. Max ~15s (362 frames)."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            },
            "optional": {
                "cap_to_h3_max": ("BOOLEAN", {"default": True, "tooltip": "Clamp frames to 362 (~15s), H3's single-clip maximum."}),
            },
        }

    def emit(self, shot_seconds, fps, cap_to_h3_max=True):
        fps = max(1, int(fps))
        frames = max(5, int(round(float(shot_seconds) * fps)))
        frames += (5 - frames) % 17
        capped = cap_to_h3_max and frames > H3_MAX_FRAMES
        frames = H3_MAX_FRAMES if capped else frames
        info = f"{shot_seconds:g}s/shot @ {fps}fps -> {frames} frames{' (capped 362)' if capped else ''}"
        return round(float(shot_seconds), 3), frames, info


NODE_CLASS_MAPPINGS = {"H3ShotLength": H3ShotLength}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ShotLength": "H3 Shot Length"}
