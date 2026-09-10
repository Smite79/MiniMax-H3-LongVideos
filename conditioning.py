# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""Decisions about which pictures may condition a shot."""

import node_helpers
from h3_runtime import H3_FPS, AUDIO_LATENT_FPS, _empty_av_latent, _resize, ref_image_canvas
from h3_audio import _SILENCE_STATUS, _silent_audio_latent, _pin_audio_silence


def may_carry_room(previous_cast, current_cast, tagged_names):
    """A previous frame is safe as a reference only when it adds no subject."""
    previous = [name for name in (previous_cast or ()) if name]
    current = set(current_cast or ())
    tagged = set(tagged_names or ())
    return bool(previous) and all(name in current for name in previous) \
        and not any(name in tagged for name in previous)


def recoverable_subject(cast, tagged_names, returning_names, captured):
    """Return the sole safe recovered subject, or an empty string."""
    people = [name for name in (cast or ()) if name]
    if len(people) != 1:
        return ""
    name = people[0]
    return name if (name not in set(tagged_names or ())
                    and name in set(returning_names or ())
                    and captured.get(name) is not None) else ""


KEYFRAME_SAFE_AUG = 0.99       # below this, a ref aug would soften the keyframe too


def _keyframe_latent(vae, hand_img):
    """The keyframe latent for this shot: an ENCODE of the previous shot's last frame.

    This was briefly an optimisation -- pass the previous shot's own latent straight
    through and skip a VAE round trip per boundary. It was wrong, and it degraded
    every shot after the first.

    A keyframe is ONE pixel frame, and H3's grid puts that at 5f -> TWO latent
    frames. Slicing [:, :, -1:] off a finished shot hands over one. Worse, the video
    VAE is causal: the last latent of a 72-frame sequence encodes its temporal
    context, not a standalone opening frame, so even at the right count it does not
    mean what a keyframe means. The spatial-size guard could not see either problem.

    The round trip is real but it is one lossy step on a correctly formed anchor,
    which beats a cheap malformed one."""
    return vae.encode(hand_img)


def _build_ref_images(vae, images, gen_w, gen_h, mode="match"):
    """(tokenizer items, DiT blocks) for a list of reference IMAGE tensors.

    The tokenizer labels each one `<Picture N>:` itself, in the order given here --
    so the roster the prompt refers to is decided by input order, not by anything
    written in the prompt."""
    items, blocks = [], []
    for img in images:
        if img is None:
            continue
        h, w = int(img.shape[1]), int(img.shape[2])
        tw, th = ref_image_canvas(w, h, gen_w, gen_h, mode)
        resized = _resize(img[:1], tw, th, "disabled")
        items.append({"type": "image", "data": resized})
        blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                       "latent": vae.encode(resized)})
    return items, blocks


def build_conditioning(clip, vae, audio_vae, prompt, width, height, length,
                       handoff=None, refs=None,
                       ref_noise_aug=0.999, silent=False, ref_image_size="match",
                       handoff_as_ref=False, speech_lead_seconds=0.0):
    """Encode prompt, identity references, keyframe, and audio constraints for a shot."""
    latent, fc = _empty_av_latent(width, height, length, H3_FPS)
    refs = [r for r in (refs or []) if r is not None]

    hand_img = None
    if handoff is not None:
        hand_img = _resize(handoff[:1], width, height, "disabled")

    # REFERENCES AND THE KEYFRAME RIDE TOGETHER. This is the arrangement the node
    # had before I broke it, and the reason is in ComfyUI's own layout:
    #
    #   model_base.py:2183-2191  cond_video_latents = keyframe latents THEN ref latents
    #   model.py PackedLayout    emits keyframe "cond" segments THEN ref "ref_img" ones
    #
    # The two orders agree, so both channels coexist. A shot takes its references AND
    # a real keyframe: the keyframe ANCHORS the first frame, which is what continuity
    # needs, while a reference only supplies identity. They are not alternatives.
    #
    # I had read "<Picture 1>" as MEANING the first frame on fl2va, and rearranged the
    # roster around that. It does not. Which image is the first frame is decided by
    # resolved_frame_index in minimax_keyframes, not by a label's number -- the labels
    # are only how the images are shown to the VLM, and what they have to line up with
    # is the <Picture N> tags in the prompt.
    #
    # So references come FIRST and keep slots 1..N, which is what a sheet line's
    # `Name: <Picture 1>, ...` points at, and the handoff is appended AFTER them where
    # it disturbs no numbering. It has to be in the list at all because
    # tokenize_with_weights is either/or: passing minimax_ref_items makes it ignore
    # `images` outright, so leaving the handoff out means the VLM is never shown where
    # the shot left off and re-imagines the scenery -- same place, new room.
    keyframe_ok = ref_noise_aug is None or float(ref_noise_aug) >= KEYFRAME_SAFE_AUG
    # One aug covers every visual condition row, references AND the keyframe. Below
    # KEYFRAME_SAFE_AUG the keyframe latent would be noised and labelled at the wrong
    # timestep, so the handoff stops being an anchor and rides as an extra reference
    # instead: weaker continuity, but nothing pretending to anchor while carrying noise.
    # ...or because the caller asked for it. A shot that introduces somebody already
    # in position wants the room this picture carries and NOT the first frame it
    # would force, and that is a demotion the aug knows nothing about.
    carry_as_ref = bool(hand_img is not None
                        and (handoff_as_ref or (refs and not keyframe_ok)))

    enc_refs = refs + ([hand_img] if carry_as_ref else [])
    items, blocks = ([], [])
    if enc_refs:
        items, blocks = _build_ref_images(vae, enc_refs, width, height, ref_image_size)
    if hand_img is not None and not carry_as_ref:
        items = items + [{"type": "image", "data": hand_img}]

    if items:
        tokens = clip.tokenize(prompt, minimax_ref_items=items)
    else:
        tokens = clip.tokenize(prompt)
    cond = clip.encode_from_tokens_scheduled(tokens)

    vals = {}
    if blocks:
        vals["minimax_refs"] = blocks
        # How CLEAN the references are shown. One aug covers every conditioning
        # latent, keyframe included -- which is why softening references below
        # KEYFRAME_SAFE_AUG would soften the anchor too.
        if ref_noise_aug is not None:
            vals["minimax_visual_cond_noise_aug"] = float(ref_noise_aug)

    kfs = []
    if hand_img is not None and not carry_as_ref:
        kfs.append({"resolved_frame_index": 0,
                    "latent": _keyframe_latent(vae, hand_img)})
    # Audio keyframes are extra conditioning rows in H3's PackedLayout. Pin the
    # generated target stream instead, so the joint model also sees a quiet mouth.
    if silent or float(speech_lead_seconds or 0.0) > 0.0:
        _SILENCE_STATUS["asked"] += 1
        if audio_vae is None:
            _SILENCE_STATUS["why"] = "no audio VAE is wired to the node"
        else:
            sil = _silent_audio_latent(audio_vae, fc, H3_FPS)
            if sil is None:
                _SILENCE_STATUS["why"] = ("the audio VAE would not encode a silent "
                                          "second -- the wrong VAE is on the "
                                          "audio_vae input")
            else:
                lead = None if silent else round(float(speech_lead_seconds) *
                                                 AUDIO_LATENT_FPS)
                if _pin_audio_silence(latent, sil, lead):
                    _SILENCE_STATUS["applied"] += 1
                else:
                    _SILENCE_STATUS["why"] = "the silent latent did not match the shot"
    if kfs:
        vals["minimax_keyframes"] = kfs
    if vals:
        cond = node_helpers.conditioning_set_values(cond, vals)
    return cond, latent, fc, carry_as_ref
