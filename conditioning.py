# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import node_helpers
from h3_runtime import H3_FPS, AUDIO_LATENT_FPS, _empty_av_latent, _resize, ref_image_canvas
from h3_audio import silent_audio_latent, pin_audio_silence

REF_NOISE_AUG = 0.999


def ref_blocks(vae, images, width, height):
    items, blocks = [], []
    for img in images:
        tw, th = ref_image_canvas(int(img.shape[2]), int(img.shape[1]), width, height)
        resized = _resize(img[:1], tw, th, "disabled")
        items.append({"type": "image", "data": resized})
        blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                       "latent": vae.encode(resized)})
    return items, blocks


def build_conditioning(clip, vae, audio_vae, prompt, width, height, length, handoff=None, refs=(),
                       silent=False, lead_seconds=0.0):
    latent, fc = _empty_av_latent(width, height, length, H3_FPS)
    items, blocks = ref_blocks(vae, [r for r in refs if r is not None], width, height)
    hand = _resize(handoff[:1], width, height, "disabled") if handoff is not None else None
    if hand is not None:
        items.append({"type": "image", "data": hand})
    tokens = clip.tokenize(prompt, minimax_ref_items=items) if items else clip.tokenize(prompt)
    cond = clip.encode_from_tokens_scheduled(tokens)
    vals = {}
    if blocks:
        vals["minimax_refs"] = blocks
        vals["minimax_visual_cond_noise_aug"] = REF_NOISE_AUG
    if hand is not None:
        vals["minimax_keyframes"] = [{"resolved_frame_index": 0, "latent": vae.encode(hand)}]
    if vals:
        cond = node_helpers.conditioning_set_values(cond, vals)
    pinned = False
    if audio_vae is not None and (silent or lead_seconds > 0):
        silence = silent_audio_latent(audio_vae, fc)
        if silence is not None:
            lead = None if silent else round(lead_seconds * AUDIO_LATENT_FPS)
            pinned = pin_audio_silence(latent, silence, lead)
    return cond, latent, fc, pinned
