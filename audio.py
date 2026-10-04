# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import torch
import comfy.nested_tensor
from h3_runtime import temporal_shape

SILENT_SECONDS = 2
SILENT_EDGE = 4
_SILENT_UNIT = {"lat": None, "key": None}


def silent_audio_latent(audio_vae, frame_count):
    try:
        sr = int(getattr(audio_vae, "audio_sample_rate", 0) or 0)
        if sr <= 0:
            return None
        want = temporal_shape(frame_count)[2]
        key = (id(audio_vae), sr)
        block = _SILENT_UNIT["lat"] if _SILENT_UNIT["key"] == key else None
        if block is None:
            enc = audio_vae.encode(torch.zeros((1, sr * SILENT_SECONDS, 2)))
            if enc is None or enc.dim() != 4 or enc.shape[1] != 32 or enc.shape[-1] <= 2 * SILENT_EDGE + 1:
                return None
            block = enc[..., SILENT_EDGE:-SILENT_EDGE].detach().to("cpu").clone()
            _SILENT_UNIT.update(lat=block, key=key)
        pieces, have = [], 0
        while have < want:
            pieces.append(block if len(pieces) % 2 == 0 else torch.flip(block, dims=[-1]))
            have += block.shape[-1]
        out = torch.cat(pieces, dim=-1)[..., :want].clone()
        return out if out.shape[-1] == want else None
    except Exception:
        return None


def pin_audio_silence(latent, silence, lead_frames=None):
    try:
        video, audio = latent["samples"].unbind()
        silence = silence.to(device=audio.device, dtype=audio.dtype)
        if silence.shape != audio.shape:
            return False
        mask = torch.ones_like(audio[:, :1])
        if lead_frames is None:
            mask.zero_()
        else:
            n = min(int(audio.shape[-1]), max(0, int(lead_frames)))
            if n <= 0:
                return False
            mask[..., :n] = 0
        latent["samples"] = comfy.nested_tensor.NestedTensor((video, silence))
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor((torch.ones_like(video[:, :1]), mask))
        return True
    except Exception:
        return False
