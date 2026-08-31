"""
H3 Long Videos -- chain MiniMax-H3 shots into one continuous video with audio.

Rebuilt from scratch. The previous version grew a large prompt-engineering layer
that wrote continuity guards into every shot; measured, the user's own beat was
under 4% of the conditioning and the rest was boilerplate arguing with it. None of
that is here. What a shot is told is: your scene text, then your beat, verbatim.

What this node does is the part a prompt cannot do -- the mechanics of chaining:

  * splits the prompt into beats on blank lines, one beat per shot;
  * gives every shot the SAME length, so one seed is one noise field across the
    chain (noise is drawn to the latent's shape, so unequal lengths mean unrelated
    noise from the same seed, and detail resets at every cut);
  * hands each shot the previous shot's last frame as a keyframe LATENT, passed
    straight through rather than decoded and re-encoded -- a VAE round trip per
    boundary compounds over a long chain into visible softening;
  * keeps identity references on every shot, which is the only fixed anchor a long
    chain has against drift;
  * anchors the audio branch to real silence on shots with no quoted line, because
    H3 is a joint model and an unconditioned audio stream invents a voice that the
    picture then lip-syncs to.

Everything about what the video should CONTAIN is yours to write.
"""

import gc
import math
import re

import torch

import nodes
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.nested_tensor
import comfy.model_management as mm
import latent_preview
import node_helpers


H3_FPS = 24                    # H3 renders 24 fps, always
AUDIO_LATENT_FPS = 40          # audio latent frames per second
VAE_SPATIAL = 16               # video VAE spatial downsample
RES_MULTIPLE = 32
KEYFRAME_SAFE_AUG = 0.99       # below this, a ref aug would soften the keyframe too
AUTO_TILE_T = 8                # temporal chunk for a tiled decode
MAX_FRAMES = 362               # H3's own ceiling (~15s)
GB = 1024 ** 3

# H3-Base is trained at 768 on the short edge; below that the whole frame softens.
NATIVE_RES = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3":  (1024, 768),
    "3:4":  (768, 1024),
    "1:1":  (768, 768),
    "21:9": (1536, 672),
    "9:21": (672, 1536),
}


CANVAS_MULTIPLE = 32


REF_IMAGE_SHORT_EDGE = 2048


_LAST_MODEL_FP = {"fp": None}


_SILENT_UNIT = {"lat": None}


def _call_node(cls, model, shift_video, shift_audio):
    """Call the H3 sampling node whether it uses the V1 (INPUT_TYPES/FUNCTION) or
    V3 (define_schema/execute) API, mapping the shift args by name."""
    inst = cls()
    # V1 API
    if hasattr(cls, "INPUT_TYPES") and getattr(cls, "FUNCTION", None):
        req = cls.INPUT_TYPES().get("required", {})
        kwargs = {}
        for name in req:
            low = name.lower()
            if low == "model":
                kwargs[name] = model
            elif "video" in low:
                kwargs[name] = float(shift_video)
            elif "audio" in low:
                kwargs[name] = float(shift_audio)
        out = getattr(inst, cls.FUNCTION)(**kwargs)
        # A V3 node exposes INPUT_TYPES and a truthy FUNCTION ('EXECUTE_NORMALIZED')
        # for compatibility, so this branch runs on 0.31+ too -- and there it returns
        # a NodeOutput, not a tuple. Without the unwrap the caller got the wrapper
        # object where a MODEL belonged. Unreachable today because the direct patch
        # succeeds first, which is exactly why it went unnoticed.
        out = getattr(out, "result", out)
        return out[0] if isinstance(out, (tuple, list)) else out
    # V3 API: an execute()/patch() classmethod taking model + shift kwargs
    fn = None
    for cand in ("execute", "patch", "apply"):
        if hasattr(inst, cand):
            fn = getattr(inst, cand); break
    if fn is None:
        raise RuntimeError("unknown node API")
    out = fn(model=model, shift_video=float(shift_video), shift_audio=float(shift_audio))
    out = getattr(out, "result", out)                 # V3 NodeOutput
    return out[0] if isinstance(out, (tuple, list)) else out


def _is_audio_vae(v):
    """True when v looks like the H3 audio VAE (DAC/BigVGAN), False when it looks
    like a video/image VAE, None when it can't be told. The video VAEs carry a
    3-tuple upscale_ratio (t, y, x); the audio VAE carries a scalar and reports
    latent_dim 2 with an audio_sample_rate."""
    ur = getattr(v, "upscale_ratio", None)
    if isinstance(ur, (tuple, list)):
        return False
    if getattr(v, "audio_sample_rate", None) or getattr(v, "audio_sample_rate_output", None):
        return True
    if isinstance(ur, (int, float)) and getattr(v, "latent_dim", None) == 2:
        return True
    return None


# --- sizing -----------------------------------------------------------------

def align_frame_count(n):
    """Up to the next valid H3 frame count. The grid is 17k+5."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return min(n, MAX_FRAMES)


def video_latent_t(fc):
    return 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2


def temporal_shape(length, fps=H3_FPS):
    """(frame count, video latent frames, audio latent frames) for a shot.

    `fps` is accepted but deliberately IGNORED: the audio latent has to line up
    with 24 fps video or the shot's sound is stretched against its picture."""
    fc = align_frame_count(length)
    return fc, video_latent_t(fc), round(fc / H3_FPS * AUDIO_LATENT_FPS)


def parse_resolution(choice):
    text = (choice or "").strip()
    if text in NATIVE_RES:
        return NATIVE_RES[text]
    m = re.search(r"(\d+)\s*x\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return NATIVE_RES["16:9"]


def scale_to_megapixels(w, h, mp, multiple=RES_MULTIPLE):
    """Scale (w, h) to `mp` megapixels keeping the ratio, snapped to the grid.
    mp <= 0 keeps the preset's own size."""
    if not mp or mp <= 0:
        return w, h
    scale = math.sqrt((mp * 1024 * 1024) / float(w * h))
    sw = max(multiple, int(round(w * scale / multiple)) * multiple)
    sh = max(multiple, int(round(h * scale / multiple)) * multiple)
    return sw, sh


# --- prompt -> beats --------------------------------------------------------

def split_beats(prompt):
    """(scene, beats). Paragraphs are separated by a BLANK line.

    The first paragraph is the SCENE: it is prepended to every shot verbatim, and
    nothing is stripped from it. Every paragraph after it is one beat, one shot.
    A single-paragraph prompt is one shot with no separate scene text.

    Deliberately the whole of the text handling. The previous version rewrote beats
    -- binding descriptions, collapsing repeated names, scrubbing the scene, adding
    continuity clauses -- and the result was a shot whose own action was a few
    percent of what the model was told. What you type is what the shot gets."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", (prompt or "").strip()) if p.strip()]
    if not paras:
        return "", []
    if len(paras) == 1:
        return "", paras
    return paras[0], paras[1:]


_QUOTED = re.compile(r'["“][^"”]+["”]')


def has_speech(beat):
    """Does this beat contain a scripted line? Double quotes are the signal."""
    return bool(_QUOTED.search(beat or ""))


_PICTURE_TAG = re.compile(r"<\s*picture[\s_\-]*(\d+)\s*>", re.I)


def picture_tags(text):
    return sorted({int(m.group(1)) for m in _PICTURE_TAG.finditer(text or "")})

def check_audio_vae_loaded(audio_vae):
    """Catch an UNCONVERTED audio VAE checkpoint.

    comfy/ldm/minimax/audio_vae.py loads a checkpoint whose weight-norm has been
    folded into plain "*.weight" tensors. Feed it the raw upstream file (172
    weight_g/weight_v pairs, no latents_mean/latents_std) and load_state_dict
    reports the misses as a WARNING, not an error: every weight-normed conv keeps
    its random init and the two normalization buffers stay torch.empty(), i.e.
    uninitialized memory. Decoding then multiplies the latents by garbage and the
    audio comes out as noise -- with nothing in the log at render time to say why.

    latents_std is the cheapest tell: it is a real per-channel scale, so a
    non-finite or absurd value means the buffer was never filled."""
    m = getattr(audio_vae, "first_stage_model", None)
    mean, std = getattr(m, "latents_mean", None), getattr(m, "latents_std", None)
    if mean is None or std is None:
        return
    try:
        bad = (not torch.isfinite(mean).all() or not torch.isfinite(std).all()
               or float(std.min()) <= 0.0 or float(std.max()) > 1e3
               or float(mean.abs().max()) > 1e3)
    except Exception:
        return                       # never block a render on a failed introspection
    if bad:
        raise RuntimeError(
            "the audio VAE loaded but its weights are NOT initialized -- this is the raw "
            "upstream MiniMax-H3 audio checkpoint (weight_g/weight_v weight-norm pairs, no "
            "latents_mean/latents_std). ComfyUI's loader needs the CONVERTED file, with "
            "weight-norm folded into plain '*.weight' tensors. Look for the 'Missing VAE keys' "
            "warning in the log when the VAE loaded. Download the repackaged H3 audio VAE from "
            "the Comfy-Org release; rendering with this one produces noise, not speech.")


def ref_image_canvas(w, h, gen_w, gen_h, mode="match"):
    """Pure: the (width, height) a reference image is encoded at.

    'match' scales it (DOWN only, aspect kept) to the generation's pixel area, so a
    reference costs about as much as one frame of the shot. 'max' goes to the
    reference pipeline's 2048 short edge for the best identity fidelity, which on a
    long chain is several times slower because the rows are re-attended every step
    of every shot. Never upscales: a small reference stays small."""
    w, h = max(1, int(w)), max(1, int(h))
    if mode == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    else:
        scale = min(1.0, math.sqrt((int(gen_w) * int(gen_h)) / float(w * h)))
    snap = lambda v: max(CANVAS_MULTIPLE, round(v * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return snap(w), snap(h)


def shot_latent_cells(w, h, frames, fps):
    """Latent cells in one shot: what sampling VRAM actually scales with.

    Not a byte figure -- the constant depends on the quantisation path -- but it is
    exactly linear in both shot length and area, so ratios between settings are
    right even though the absolute number is not a prediction."""
    _, lt, _ = temporal_shape(frames, fps)
    return max(1, int(lt)) * max(1, w // 16) * max(1, h // 16)


def model_fingerprint(model):
    """A cheap, stable identity for the loaded DiT: (quant format, layer count,
    weight bytes, class name). Changes whenever the checkpoint changes -- a
    different quant, a pruned-vs-full build, or a different model entirely -- while
    staying identical across shots of the same run. Deliberately avoids hashing
    weights, which would cost more than the flush it guards."""
    try:
        dm = getattr(getattr(model, "model", None), "diffusion_model", None)
        fmts, n = {}, 0
        if dm is not None and hasattr(dm, "modules"):
            for mod in dm.modules():
                n += 1
                f = getattr(mod, "quant_format", None)
                if f:
                    fmts[f] = fmts.get(f, 0) + 1
        top = max(fmts.items(), key=lambda kv: kv[1])[0] if fmts else "none"
        size = 0
        try:
            size = int(model.model_size())
        except Exception:
            pass
        cls = type(dm).__name__ if dm is not None else "unknown"
        return (top, n, size, cls)
    except Exception:
        return None



# --- H3 plumbing, carried over unchanged: these were arrived at against the real
# model and the real VAEs, and none of it is prompt logic.

def _resize(image, width, height, crop):
    s = image[..., :3].movedim(-1, 1)
    s = comfy.utils.common_upscale(s, width, height, "lanczos", crop)
    return s.movedim(1, -1)


def _empty_av_latent(width, height, length, fps, batch_size=1):
    fc, lt, at = temporal_shape(length, fps)
    video = torch.zeros([batch_size, 24, lt, height // 16, width // 16], device=mm.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, at], device=mm.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, fc


def _auto_tile_t(n_latent_frames, requested=None):
    """Temporal tile for a tiled decode. An explicit value wins.

    The decode_tile_frames widget is gone, so this is where the value comes from
    now. It has to come from somewhere: ComfyUI's decode_tiled_3d defaults tile_t
    to 999, i.e. SPATIAL tiles only, and expanding the whole clip's time axis at
    once is the single largest allocation in a run. A "tiled" decode that keeps the
    full temporal extent barely lowers the peak, so the OOM retry that switches
    tiling on was, without this, retrying with almost the same footprint."""
    if requested:
        return int(requested)
    n = int(n_latent_frames or 0)
    return AUTO_TILE_T if n > AUTO_TILE_T else None


def _decode_video(vae, out_latent, tiled, free_first=None, tile_t=None, tile_xy=None):
    """Decode the video latent. If `free_first` is the diffusion model, unload it
    first: sampling is finished, and the ~5GB video VAE needs the room. Leaving the
    DiT (plus resident bypass-LoRA adapters) on the card while the VAE loads is a
    second ratchet -- ComfyUI would otherwise evict reactively, after spilling."""
    if free_first is not None:
        try:
            mm.free_memory(1e30, mm.get_torch_device(), keep_loaded=[])
        except Exception:
            pass
    latent = out_latent["samples"]
    if latent.is_nested:
        latent = latent.unbind()[0]
    if tiled:
        # Temporal + spatial tiling. Without tile_t the VAE expands the WHOLE latent
        # clip at once, which on a 243-frame 1344x768 shot is the single largest
        # allocation in the run -- and on an unpruned checkpoint that is already
        # streaming, it is what tips the card over. Decoding in temporal chunks
        # trades a little speed for a much lower peak; None keeps ComfyUI's defaults.
        args = {}
        tile_t = _auto_tile_t(latent.shape[2] if latent.ndim >= 5 else 0, tile_t)
        if tile_t:
            args["tile_t"] = int(tile_t)
            args["overlap_t"] = max(1, int(tile_t) // 8)
        if tile_xy:
            args["tile_x"] = int(tile_xy)
            args["tile_y"] = int(tile_xy)
        try:
            imgs = vae.decode_tiled(latent, **args) if args else vae.decode_tiled(latent)
        except TypeError:
            imgs = vae.decode_tiled(latent)      # older signature without tile_t
    else:
        imgs = vae.decode(latent)
    if len(imgs.shape) == 5:
        imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
    return imgs


def _decode_audio(audio_vae, out_latent):
    latent = out_latent["samples"]
    if latent.is_nested:
        latent = latent.unbind()[-1]
    audio = audio_vae.decode(latent).movedim(-1, 1)
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std
    sr = getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 44100))
    return {"waveform": audio, "sample_rate": sr}


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _deep_cleanup():
    """Release VRAM + RAM between shots so a long chain doesn't accumulate and OOM.
    Runs a Python GC pass (frees dereferenced tensors / CPU buffers), then hands
    ComfyUI its aggressive cache purge, then empties the CUDA allocator's cached
    blocks and IPC handles. Cheap relative to sampling; called once per beat."""
    gc.collect()
    try:
        mm.soft_empty_cache(True)      # aggressive (unload_all_models path)
    except TypeError:
        mm.soft_empty_cache()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _evict_all_but(keep_model):
    """Unload every model EXCEPT the diffusion model from the GPU.

    This is the fix for VRAM ratcheting across a long chain. soft_empty_cache()
    only drops the CUDA allocator's cached blocks -- it does NOT unload models, so
    ComfyUI keeps the Qwen3-VL text encoder (~14.6GB) and both VAEs resident in
    current_loaded_models alongside the DiT. Each shot re-encodes the prompt
    (text encoder), encodes the handoff keyframe (video VAE), then samples (DiT),
    so all three compete for the card; ComfyUI only evicts reactively, i.e. AFTER
    it has already spilled. With a bypass LoRA also holding 208 bf16 adapters
    resident there is no room left, and every shot leaves the card fuller.

    Freeing them explicitly, right after conditioning is built and before
    sampling, keeps only what the sampler actually needs on the GPU."""
    try:
        keep = []
        for lm in list(getattr(mm, "current_loaded_models", [])):
            try:
                if lm.model is keep_model or getattr(lm, "model", None) is getattr(keep_model, "model", None):
                    keep.append(lm)
            except Exception:
                pass
        mm.free_memory(1e30, mm.get_torch_device(), keep_loaded=keep)
    except Exception:
        try:
            mm.soft_empty_cache(True)
        except Exception:
            pass


def check_vae_wiring(vae, audio_vae):
    """Catch the commonest miswire -- the video VAE dropped into BOTH VAE inputs.
    Without this the run samples a whole shot, decodes the video fine, then dies
    deep inside comfy/sd.py with 'IndexError: tuple index out of range' when the
    video memory estimator indexes shape[4] of the 4-D audio latent."""
    if _is_audio_vae(audio_vae) is False:
        raise RuntimeError(
            "audio_vae is a video/image VAE, not the H3 audio VAE. Load the audio "
            "autoencoder (the DAC/BigVGAN one shipped with MiniMax-H3, e.g. "
            "minimax_h3_audio_vae.safetensors) in its own VAELoader and wire that "
            "into 'audio_vae'; the video VAE belongs on 'vae' only.")
    check_audio_vae_loaded(audio_vae)
    if _is_audio_vae(vae) is True:
        raise RuntimeError(
            "vae is the H3 audio VAE -- the video and audio VAE inputs are swapped. "
            "Wire the video VAE into 'vae' and the audio VAE into 'audio_vae'.")


def flush_for_model_change(model):
    """Detect a checkpoint swap since the last run and, if one happened, hard-flush
    GPU state before doing anything else.

    Why this matters: ComfyUI keeps previously-loaded models in current_loaded_models
    and only evicts reactively. Swapping checkpoints mid-session (e.g. NVFP4 -> FP8 ->
    MXFP8 while comparing quality) leaves the OLD DiT resident alongside the new one,
    plus any hooks/injections a previous LoRA installed and stale cached allocator
    blocks sized for the old model's layers. The result is a card that is already
    half full before the first shot samples -- which looks exactly like the node
    over-spilling, when in fact the budget was computed against memory the previous
    checkpoint never released.

    Returns a note for `info` when a change was detected (empty string otherwise)."""
    fp = model_fingerprint(model)
    prev = _LAST_MODEL_FP.get("fp")
    _LAST_MODEL_FP["fp"] = fp
    if prev is None or fp is None or prev == fp:
        return ""
    try:
        mm.unload_all_models()          # drop every resident model, not just the cache
    except Exception:
        pass
    # Never let a cleanup failure abort the run: the flush is best-effort hygiene,
    # and a partially-flushed card is still better than raising here.
    for _ in range(2):                  # 2nd pass frees blocks released by the 1st
        try:
            _deep_cleanup()
        except Exception:
            pass
    old_fmt, _n, old_sz, _c = prev
    new_fmt = fp[0]
    return (f"model changed since last run ({old_fmt} ~{old_sz / GB:.1f}GB -> {new_fmt} "
            f"~{fp[2] / GB:.1f}GB): flushed all resident models and VRAM caches")


def _silent_audio_latent(audio_vae, frame_count, fps):
    """A keyframe audio latent of actual SILENCE, or None if it cannot be made.

    H3 is a JOINT model: the mouth follows the audio branch. On a shot with no
    scripted line the branch is otherwise unconditioned, and an unconditioned audio
    branch invents a voice -- which the picture then lip-syncs to. The lips-closed
    sentence is arguing with a stream that has already decided someone is talking.

    Seeding the keyframe's audio channel with encoded silence anchors that stream
    instead. comfy/ldm/minimax/audio_vae.py encode() takes stereo [B, 2, L] at
    32 kHz and returns [B, 32, 2, T] on the same 40 Hz grid temporal_shape() uses.

    Everything here is defensive. The shape is CHECKED against what the layout
    expects rather than assumed, and any failure returns None so the shot falls
    back to today's behaviour instead of breaking the render."""
    try:
        sr = int(getattr(audio_vae, "audio_sample_rate", 0) or 0)
        if sr <= 0:
            return None
        _, _, want_t = temporal_shape(frame_count, fps)
        if want_t <= 0:
            return None
        unit = _SILENT_UNIT.get("lat")
        if unit is None:
            # CHANNELS LAST. comfy.sd.VAE.encode() does `pixel_samples.movedim(-1, 1)`
            # before handing off, so the audio VAE -- which wants [B, 2, L] -- must be
            # given [B, L, 2]. Passing [B, 2, L] raises inside the encoder, and the
            # first version did exactly that: swallowed by the guard below, so the
            # whole layer silently did nothing.
            #
            # ONE SECOND, encoded ONCE. Silence is homogeneous, so the result tiles
            # along time -- and encoding a full 15s shot instead cost a VAE pass big
            # enough to OOM mid-render on a 16GB card, where the failure again
            # degraded silently to no conditioning at all.
            enc = audio_vae.encode(torch.zeros((1, sr, 2)))
            if enc is None or enc.dim() != 4 or enc.shape[1] != 32 or enc.shape[-1] < 3:
                return None
            # ONE STEADY FRAME, from the MIDDLE. The encoder's zero-padding leaves
            # heavy edge artifacts -- measured deviation 0.351 at the first and last
            # frames against 0.002 in the interior, ~170x -- and tiling the whole
            # second therefore stamped a spike every 40 latent frames, which at 40Hz
            # is once per SECOND. That is a metronome in the audio conditioning of a
            # joint audio-video model, and the picture lip-syncs to it. Repeating a
            # single interior frame gives conditioning that is genuinely constant.
            mid = enc.shape[-1] // 2
            _SILENT_UNIT["lat"] = enc[..., mid:mid + 1].detach().to("cpu").clone()
        unit = _SILENT_UNIT["lat"]
        if unit.shape[-1] != 1:
            return None
        return unit.repeat(1, 1, 1, want_t).clone()
    except Exception:
        return None                         # never fail a render for a nicety


def _keyframe_latent(vae, hand_img, handoff_latent, width, height, notes=None):
    """The keyframe latent for this shot: the previous shot's OWN latent when it
    fits, otherwise a fresh encode of the decoded frame.

    Every boundary used to run latent -> decode -> image -> encode -> latent. One
    round trip is lossy; a chain of them compounds, because each shot is generated
    from the previous shot's already-degraded keyframe. Over ten beats that is ten
    round trips of softening and colour drift, which is quality falling away as the
    chain gets longer.

    The DiT only ever needed a latent, and the previous shot produced one. The
    decoded frame is still handed to the text encoder -- the VLM has to SEE it --
    but its degradation no longer feeds the picture.

    Falls back on any mismatch: a resolution backoff, a latent upscale, or a shape
    this build does not expect. Continuity is worth more than the round trip."""
    if handoff_latent is not None:
        try:
            lat = handoff_latent
            want = (int(height) // VAE_SPATIAL, int(width) // VAE_SPATIAL)
            if lat.ndim == 5 and tuple(lat.shape[-2:]) == want and lat.shape[2] >= 1:
                return lat[:, :, -1:].contiguous()
            if notes is not None:
                notes.append(f"handoff re-encoded (latent {tuple(lat.shape[-2:])} vs "
                             f"expected {want})")
        except Exception as e:
            if notes is not None:
                notes.append(f"handoff re-encoded ({type(e).__name__})")
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


def _sample_on_sigmas(model, seed, cfg, sampler_name, positive, negative, latent, sigmas):
    """common_ksampler, driven by an EXTERNAL sigma schedule.

    common_ksampler derives its sigmas from (sampler_name, scheduler, steps, denoise)
    and takes no schedule argument, so a schedule computed anywhere else cannot
    reach it. Under PDD that is fatal rather than merely inconvenient: the heads
    accept only their nine trained boundaries, and re-deriving the grid from
    widgets means hitting it by coincidence and losing it again the moment a step
    count changes.

    Mirrors nodes.common_ksampler's noise / mask / callback handling exactly -- the
    only substitution is comfy.sample.sample_custom for comfy.sample.sample."""
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None))
    noise = comfy.sample.prepare_noise(latent_image, seed, latent.get("batch_index"))
    # `steps` here only sizes the progress bar -- the schedule is `sigmas`, whose
    # step count is one less than its length (the trailing 0.0 is an endpoint).
    callback = latent_preview.prepare_callback(model, max(len(sigmas) - 1, 1))
    samples = comfy.sample.sample_custom(
        model, noise, cfg, comfy.samplers.sampler_object(sampler_name), sigmas,
        positive, negative, latent_image,
        noise_mask=latent.get("noise_mask"), callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


def _find_h3_sampling_node():
    """Locate the H3 sigma-shift node under ANY registered name. It was renamed
    to 'ModelSamplingMiniMaxH3' in a later patch (kijai PR #15243); older 0.30.x
    builds register it under a different id, so exact-key lookup misses it. Try
    the known names, then fuzzy-scan all node mappings for the H3 model-sampling
    node. Returns (class, key) or (None, None)."""
    maps = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    for key in ("ModelSamplingMiniMaxH3", "ModelSamplingMinimaxH3", "ModelSamplingMinimax", "ModelSamplingH3"):
        if key in maps:
            return maps[key], key
    for k, v in maps.items():
        kl = k.lower()
        if "sampl" in kl and (("minimax" in kl and "h3" in kl) or ("h3" in kl and "shift" in kl)):
            return v, k
    for k, v in maps.items():
        kl = k.lower()
        if ("minimax" in kl or "h3" in kl) and ("shift" in kl or "sampling" in kl):
            return v, k
    return None, None


def _direct_model_sampling(model, shift_video, shift_audio):
    """Fallback that sets the shift on the model's own model_sampling object
    without any node -- version-tolerant and V3-proof, since it uses model-level
    APIs (get_model_object / set_parameters / add_object_patch) rather than
    calling a node. Copies the sampling object so the base model isn't mutated,
    and applies audio_shift only if the installed set_parameters accepts it."""
    import inspect, copy
    m = model.clone()
    # deepcopy, not copy: model_sampling is an nn.Module, and a SHALLOW copy shares
    # its `_buffers` dict with the original. set_parameters() re-registers `sigmas`
    # into that shared dict, so a shallow copy silently rewrites the BASE model's
    # sigma table -- the very thing this copy exists to prevent. Our own run reads
    # the patched object either way, but ComfyUI caches the model across queue
    # runs, so the damage outlives this execution and reaches anything else holding
    # that model. The buffer is ~1000 floats; the deepcopy is free.
    ms = copy.deepcopy(m.get_model_object("model_sampling"))
    sig = inspect.signature(ms.set_parameters)
    kwargs = {}
    if "shift" in sig.parameters:
        kwargs["shift"] = float(shift_video)
    if "audio_shift" in sig.parameters:
        # NOTE: on ComfyUI 0.31 the audio latent is carried on the video schedule
        # scaled by audio_scale = shift_video / shift_audio (12/3 = 4.0), applied in
        # process_latent_in and undone in process_latent_out. Forcing that ratio to
        # 1.0 (audio_shift == shift_video) as a "legacy 0.30" emulation produces
        # SILENT output -- the model needs the scaling -- so it is not offered.
        kwargs["audio_shift"] = float(shift_audio)
    if not kwargs:
        raise RuntimeError("set_parameters takes no shift")
    ms.set_parameters(**kwargs)
    m.add_object_patch("model_sampling", ms)
    return m


def apply_h3_model_sampling(model, shift_video, shift_audio):
    """Apply H3's dual video/audio flow schedule from INSIDE the node so a missing
    upstream patch can't silently gibberish the audio.

    On ComfyUI 0.31+ the H3 nodes are V3-schema and don't live in the legacy
    NODE_CLASS_MAPPINGS the old way -- AND the model already defaults to the correct
    FLOW_AV schedule (12/3) at load. So the reliable path here is a DIRECT model-
    level patch (works regardless of node API); the node call is only a secondary.
    Order: direct model_sampling patch -> node under any name (V1/V3) -> give up with
    an informative, non-alarming note. Shifts aren't hardcoded (12/3 base, ~8 video
    for low-step MXFP8, ~4-6 audio for turbo)."""
    try:
        return _direct_model_sampling(model, shift_video, shift_audio), \
               f"model_sampling video {shift_video:g}/audio {shift_audio:g} (direct)"
    except Exception:
        pass
    cls, key = _find_h3_sampling_node()
    if cls is not None:
        try:
            return _call_node(cls, model, shift_video, shift_audio), \
                   f"model_sampling video {shift_video:g}/audio {shift_audio:g} (via {key})"
        except Exception:
            pass
    return model, (f"model_sampling not explicitly set (video {shift_video:g}/audio {shift_audio:g}); "
                   "on ComfyUI 0.30+ the model already defaults to the correct schedule, so this is "
                   "usually harmless -- only set shift_video/audio explicitly if you're on a low-step "
                   "MXFP8/turbo profile and the audio sounds wrong")


def sampling_oom_help(w, h, frames, fps, megapixels=0.0):
    """What to change, in this shot's own numbers, after a SAMPLING OOM.

    Tiling is a decode setting and cannot help here, so the generic "try tiling"
    advice is worse than useless -- it costs another full sampling pass before
    failing the same way. Give the two levers that do change sampling cost, each
    priced from the shot that just failed."""
    now = shot_latent_cells(w, h, frames, fps)
    secs = frames / float(fps or 24)
    out = [f"This is a SAMPLING out-of-memory, not a decode one, so tiled decode "
           f"cannot help it. The shot is {w}x{h} x {frames}f (~{secs:.1f}s) = "
           f"{now:,} latent cells, and sampling cost scales linearly with that."]
    opts = []
    for cut in (10.0, 7.0):
        if cut < secs - 0.4:
            f2 = align_frame_count(int(round(cut * (fps or 24))))
            opts.append(f"shot_seconds {cut:g} ({f2}f) is "
                        f"{100 - shot_latent_cells(w, h, f2, fps) * 100 // now}% smaller")
    if megapixels:
        for mp in (0.5, 0.35):
            if mp < megapixels - 0.02:
                w2, h2 = scale_to_megapixels(w, h, mp)
                opts.append(f"megapixels {mp:g} ({w2}x{h2}) is "
                            f"{100 - shot_latent_cells(w2, h2, frames, fps) * 100 // now}% smaller")
    if opts:
        out.append("Options: " + "; ".join(opts) + ".")
    out.append("Shot length is the stronger lever on a chain, because every shot pays it. "
               "H3's own cap is 362 frames and this shot is at or near it.")
    return " ".join(out)



# --- one shot's conditioning ------------------------------------------------

def build_conditioning(clip, vae, audio_vae, prompt, width, height, length,
                       handoff=None, handoff_latent=None, refs=None,
                       ref_noise_aug=0.999, silent=False, ref_image_size="match"):
    """Text + references + keyframe for a single shot.

    THE ONE RULE from H3's layout: a shot's conditioning rows are packed in the
    order the tokenizer is given them, and tokenize_with_weights is either/or --
    passing minimax_ref_items makes it ignore `images` outright. So the handoff
    frame has to go through the SAME channel as the references or the text encoder
    never sees where the shot left off.
    """
    latent, fc = _empty_av_latent(width, height, length, H3_FPS)
    refs = [r for r in (refs or []) if r is not None]
    items, blocks = [], []
    if refs:
        items, blocks = _build_ref_images(vae, refs, width, height, ref_image_size)

    hand_img = None
    if handoff is not None:
        hand_img = _resize(handoff[:1], width, height, "disabled")
        # Appended AFTER the references so <Picture N> numbering is untouched.
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
    if hand_img is not None:
        kfs.append({"resolved_frame_index": 0,
                    "latent": _keyframe_latent(vae, hand_img, handoff_latent,
                                               width, height, None)})
    # Silence on the audio branch for a shot with no scripted line. H3 is joint:
    # an unconditioned audio stream invents a voice and the picture lip-syncs to it,
    # and no sentence in the prompt outvotes a stream that has already decided
    # someone is talking. PackedLayout emits a video segment only when a keyframe
    # carries a `latent`, so an audio-only keyframe is legal and costs no frame.
    if silent and audio_vae is not None:
        sil = _silent_audio_latent(audio_vae, fc, H3_FPS)
        if sil is not None:
            kfs.append({"resolved_frame_index": 0, "audio_latent": sil})
    if kfs:
        vals["minimax_keyframes"] = kfs
    if vals:
        cond = node_helpers.conditioning_set_values(cond, vals)
    return cond, latent, fc


def sample_shot(model, cond, negative, latent, seed, steps, cfg, sampler_name,
                scheduler, sigmas=None):
    """One sampling pass. denoise is fixed at 1.0: partial denoise desyncs the
    joint audio/video schedule."""
    if sigmas is not None and len(sigmas):
        return _sample_on_sigmas(model, seed, cfg, sampler_name, cond, negative,
                                 latent, sigmas)
    (out,) = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                   cond, negative, latent, denoise=1.0)
    return out


class H3LongVideos:
    """One prompt -> a chain of MiniMax-H3 shots, joined into one video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "forceInput": True,
                    "tooltip": "Paragraph 1 is the SCENE, prepended to every shot verbatim. "
                               "Every paragraph after it is one beat = one shot.\n\n"
                               "Nothing is rewritten. What you type is what the shot is told, "
                               "plus the scene line. Put a quoted \"line of dialogue\" in a beat "
                               "and that shot keeps its audio; beats without one are silenced."}),
                "resolution": (list(NATIVE_RES), {"default": "16:9",
                    "tooltip": "Aspect ratio. megapixels sets the size."}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "1.0 = 1024x1024 worth of pixels, H3's native budget. Lower is "
                               "faster and leaner; 0 keeps the preset's own dimensions. Cost "
                               "scales with latent cells and attention is quadratic in them."}),
                "shot_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5,
                    "tooltip": "Length of EVERY shot. Uniform on purpose: noise is drawn to the "
                               "latent's shape, so shots of different lengths get unrelated noise "
                               "from the same seed and the grain resets at every cut. Snapped to "
                               "H3's 17k+5 frame grid."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100,
                    "tooltip": "6-8 with a turbo/distill LoRA; 20+ without one."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "H3 is CFG-free. At 1.0 the negative prompt is never evaluated -- "
                               "which is why nothing here is phrased as a negation."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "One seed for the whole chain. Every shot is the same length, so "
                               "they share a noise field."}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip":
                    "Pins the opening frame of shot 1 -- the only shot with no previous frame to "
                    "continue from. If shot 1 has to start in a particular pose or position, this "
                    "is the mechanism; text cannot outrank a picture."}),
                "ref_image_1": ("IMAGE", {"tooltip":
                    "Identity reference, applied to every shot unless the prompt places it with a "
                    "<Picture 1> tag. Kept on every shot on purpose: it is the only fixed anchor a "
                    "long chain has, and without it shot 11 is drift piled on drift."}),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
                "negative": ("CONDITIONING", {"tooltip":
                    "Ignored at cfg 1.0, which is where H3 runs. Wired for completeness."}),
                "sigmas": ("SIGMAS", {"tooltip":
                    "An external schedule (PDD Acc's Apply node). Drives the sampler directly; "
                    "steps and scheduler are then only for the progress bar."}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Keep video:audio near 4:1. H3 carries the audio latent on the "
                               "video schedule scaled by that ratio; flattening it breaks audio."}),
                "apply_model_sampling": ("BOOLEAN", {"default": True,
                    "tooltip": "Patch the dual video/audio schedule inside the node. Turn off only "
                               "if you patch it upstream yourself."}),
                "silence_nonspeech": ("BOOLEAN", {"default": True,
                    "tooltip": "Anchor the audio branch to real silence on any shot with no quoted "
                               "line. H3 is joint -- an unconditioned audio stream invents a voice "
                               "and the picture lip-syncs to it. This conditions the stream itself "
                               "rather than asking the prompt to stop it."}),
                "trim_seam": ("BOOLEAN", {"default": True,
                    "tooltip": "Drop the first frame of every shot after the first: it is the "
                               "model's own reproduction of the keyframe, so it is a duplicate."}),
                "ref_noise_aug": ("FLOAT", {"default": 0.999, "min": 0.5, "max": 1.0, "step": 0.005,
                    "tooltip": "How CLEAN a reference is shown. 0.999 (H3's default) hands over a "
                               "noise-free image, which invites the model to REPRODUCE it -- "
                               "including its background and pose -- in the opening frames. Lower "
                               "says approximate: try 0.95, then 0.90. One aug covers every "
                               "conditioning latent, so below 0.99 the keyframe rides as a "
                               "reference instead of an anchor."}),
                "tiled_decode": ("BOOLEAN", {"default": True,
                    "tooltip": "Decode in tiles. The whole-clip decode is the single largest "
                               "allocation in a run and the usual point a big checkpoint spills."}),
                "cleanup_between_shots": ("BOOLEAN", {"default": True,
                    "tooltip": "Move each finished shot to system RAM and purge VRAM between "
                               "shots, so a long chain does not accumulate on the card."}),
                "plan_only": ("BOOLEAN", {"default": False,
                    "tooltip": "Report the shot split, lengths and warnings without rendering."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("images", "audio", "info", "script", "frames_per_shot", "total_frames",
                    "shots", "video_seconds")
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"
    DESCRIPTION = ("Chain MiniMax-H3 shots into one continuous video with synchronised audio. "
                   "One paragraph per shot; the first paragraph is the scene. Your text is "
                   "passed through verbatim.")

    def run(self, model, clip, vae, audio_vae, prompt, resolution, megapixels, shot_seconds,
            steps, cfg, sampler_name, scheduler, seed,
            first_frame=None, ref_image_1=None, ref_image_2=None, ref_image_3=None,
            ref_image_4=None, negative=None, sigmas=None,
            shift_video=12.0, shift_audio=3.0, apply_model_sampling=True,
            silence_nonspeech=True, trim_seam=True, ref_noise_aug=0.999,
            tiled_decode=True, cleanup_between_shots=True, plan_only=False):

        notes = []
        swap = flush_for_model_change(model)
        if swap:
            notes.append(swap)
        check_vae_wiring(vae, audio_vae)

        scene, beats = split_beats(prompt)
        if not beats:
            raise RuntimeError("H3 Long Videos: the prompt is empty. Write at least one "
                               "paragraph; each paragraph is one shot.")

        w, h = scale_to_megapixels(*parse_resolution(resolution), megapixels)
        frames = align_frame_count(int(round(float(shot_seconds) * H3_FPS)))
        shots = [f"{scene} {b}".strip() if scene else b for b in beats]
        speech = [has_speech(b) for b in beats]

        refs_all = [r for r in (ref_image_1, ref_image_2, ref_image_3, ref_image_4)
                    if r is not None]
        tagged = any(picture_tags(s) for s in shots)

        notes.append(f"{len(shots)} shot(s) x {frames}f (~{frames / H3_FPS:.1f}s) at {w}x{h} "
                     f"= ~{len(shots) * frames / H3_FPS:.1f}s total")
        if refs_all:
            notes.append(f"{len(refs_all)} reference image(s), "
                         + ("placed by <Picture N> tags" if tagged else "on every shot")
                         + (f", ref_noise_aug {float(ref_noise_aug):g}"
                            f"{' -- near-clean, so the reference tends to be reproduced in the '
                              'opening frames, pose and background included'
                              if float(ref_noise_aug) >= 0.99 else ''}"))
        n_silent = sum(1 for s in speech if not s)
        if silence_nonspeech and n_silent:
            notes.append(f"{n_silent} shot(s) have no quoted line and are silenced")
        if first_frame is None:
            notes.append("no first_frame: shot 1 has nothing pinning its opening frame, so its "
                         "starting pose and framing come from the text and any reference")
        if float(cfg) != 1.0:
            notes.append(f"cfg is {float(cfg):g}; H3 is CFG-free and expects 1.0")

        script = "\n---\n".join(f"[Shot {i}] {s}" for i, s in enumerate(shots, 1))
        info = " | ".join(notes)
        if plan_only:
            empty = torch.zeros((1, h, w, 3))
            return (empty, {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 44100},
                    "PLAN ONLY -- nothing rendered. " + info, script,
                    frames, 0, len(shots), 0.0)

        if apply_model_sampling:
            model, ms_note = apply_h3_model_sampling(model, shift_video, shift_audio)
            notes.append(ms_note)
        if negative is None:
            negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

        handoff, handoff_lat = first_frame, None
        vid_out, aud_out, sr = [], [], 44100
        _deep_cleanup()

        for i, shot_prompt in enumerate(shots):
            shot_refs = refs_all
            if tagged:
                want = picture_tags(shot_prompt)
                shot_refs = [refs_all[n - 1] for n in want if 1 <= n <= len(refs_all)]
            silent = bool(silence_nonspeech and not speech[i])

            cond, latent, fc = build_conditioning(
                clip, vae, audio_vae, shot_prompt, w, h, frames,
                handoff=handoff, handoff_latent=handoff_lat, refs=shot_refs,
                ref_noise_aug=ref_noise_aug, silent=silent)
            _evict_all_but(model)
            try:
                out = sample_shot(model, cond, negative, latent, seed, steps, cfg,
                                  sampler_name, scheduler, sigmas)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom(e):
                    raise
                raise RuntimeError(
                    f"H3 Long Videos: shot {i + 1} of {len(shots)} ran out of VRAM while "
                    f"sampling. " + sampling_oom_help(w, h, fc, H3_FPS, megapixels)) from e

            # The next shot's keyframe, taken from THIS shot's own latent: passing it
            # straight through skips a decode/encode round trip per boundary, and those
            # compound over a chain into visible softening.
            try:
                parts = out["samples"].unbind() if hasattr(out["samples"], "unbind") else None
                handoff_lat = (parts[0][:, :, -1:].detach().to("cpu", copy=True)
                               if parts else None)
            except Exception:
                handoff_lat = None

            imgs = _decode_video(vae, out, tiled_decode, free_first=model)
            wav = _decode_audio(audio_vae, out)
            sr = wav["sample_rate"]
            del out

            handoff = imgs[-1:].detach().to("cpu", copy=True)
            if trim_seam and i > 0:
                imgs = imgs[1:]
                wav["waveform"] = wav["waveform"][..., max(0, round(sr / H3_FPS)):]
            vid_out.append(imgs.to("cpu", copy=True) if cleanup_between_shots else imgs)
            aud_out.append(wav["waveform"].to("cpu", copy=True) if cleanup_between_shots
                           else wav["waveform"])
            del imgs, wav
            if cleanup_between_shots:
                _deep_cleanup()

        video = torch.cat(vid_out, dim=0)
        audio = torch.cat(aud_out, dim=-1)
        total = video.shape[0]
        notes.append(f"rendered {total} frames (~{total / H3_FPS:.1f}s)")
        return (video, {"waveform": audio, "sample_rate": sr}, " | ".join(notes), script,
                frames, total, len(shots), round(total / H3_FPS, 2))


NODE_CLASS_MAPPINGS = {"H3LongVideos": H3LongVideos}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LongVideos": "H3 Long Videos"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
