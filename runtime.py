# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import logging
import math
import time

import torch
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.nested_tensor
import comfy.model_management as mm
import latent_preview


class FrameAccumulator:

    def __init__(self, capacity, dtype, store_on_cpu):
        self.capacity = int(capacity)
        self.dtype = dtype
        self.store_on_cpu = bool(store_on_cpu)
        self.tensor = None
        self.used = 0
        self.overflow = []

    def add(self, frames):
        count = int(frames.shape[0])
        if self.tensor is None and count:
            device = torch.device("cpu") if self.store_on_cpu else frames.device
            self.tensor = torch.empty(
                (max(count, self.capacity),) + tuple(frames.shape[1:]),
                dtype=self.dtype, device=device)
        if (not self.overflow and self.tensor is not None
                and self.used + count <= self.tensor.shape[0]):
            self.tensor[self.used:self.used + count].copy_(frames)
            self.used += count
            return
        self.overflow.append(frames.to("cpu", self.dtype, copy=True)
                             if self.store_on_cpu else frames)

    def release(self):
        self.tensor = None
        self.overflow = []
        self.used = 0

    def finish(self):
        if not self.overflow:
            if self.tensor is None:
                return torch.cat(self.overflow, dim=0)
            out = self.tensor if self.used == self.tensor.shape[0] else self.tensor[:self.used]
            self.tensor = None
            return out

        extra = sum(int(piece.shape[0]) for piece in self.overflow)
        reference = self.tensor if self.tensor is not None else self.overflow[0]
        out = torch.empty((self.used + extra,) + tuple(reference.shape[1:]),
                          dtype=self.dtype, device=reference.device)
        if self.tensor is not None and self.used:
            out[:self.used].copy_(self.tensor[:self.used])
        at = self.used
        while self.overflow:
            piece = self.overflow.pop(0)
            count = int(piece.shape[0])
            out[at:at + count].copy_(piece)
            at += count
        self.tensor = None
        return out

H3_FPS = 24

AUDIO_LATENT_FPS = 40

AUTO_TILE_T = 8

MAX_FRAMES = 362

CANVAS_MULTIPLE = 32

REF_IMAGE_SHORT_EDGE = 2048


def align_frame_count(n):
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return min(n, MAX_FRAMES)


def video_latent_t(fc):
    return 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2


def temporal_shape(length, fps=H3_FPS):
    fc = align_frame_count(length)
    return fc, video_latent_t(fc), round(fc / H3_FPS * AUDIO_LATENT_FPS)


def ref_image_canvas(w, h, gen_w, gen_h, mode="match"):
    w, h = max(1, int(w)), max(1, int(h))
    if mode == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    else:
        scale = min(1.0, math.sqrt((int(gen_w) * int(gen_h)) / float(w * h)))
    snap = lambda v: max(CANVAS_MULTIPLE, round(v * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return snap(w), snap(h)


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
    if requested:
        return int(requested)
    n = int(n_latent_frames or 0)
    return AUTO_TILE_T if n > AUTO_TILE_T else None


def _decode_video(vae, out_latent, tiled, free_first=None, tile_t=None, tile_xy=None,
                  keep=()):
    latent = out_latent["samples"]
    if latent.is_nested:
        latent = latent.unbind()[0]
    if free_first is not None:
        try:
            mm.free_memory(_decode_headroom(vae, latent), mm.get_torch_device(),
                           keep_loaded=_resident(keep or (vae,)))
        except Exception:
            pass
    if tiled and _vae_owns_tiling(vae):
        imgs = vae.decode(latent)
    elif tiled:
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
            imgs = vae.decode_tiled(latent)
    else:
        imgs = vae.decode(latent)
    if len(imgs.shape) == 5:
        imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
    return imgs


def _vae_owns_tiling(vae):
    return bool(getattr(vae, "handles_tiling", False) and getattr(
        getattr(vae, "first_stage_model", None), "comfy_has_chunked_io", False))

DECODE_RAM_COPIES = 2
TILED_DECODE_RAM_COPIES = 4


def _decode_ram(vae, out_latent, tiled):
    try:
        latent = out_latent["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        b, _, t, lh, lw = (int(x) for x in latent.shape)
        rt, rh, rw = vae.upscale_ratio
        frames = int(rt(t)) if callable(rt) else t * int(rt)
        shot = b * frames * lh * int(rh) * lw * int(rw) * 3 * \
            torch.empty((), dtype=_image_out_dtype()).element_size()
    except Exception:
        return 0
    copies = (TILED_DECODE_RAM_COPIES if tiled and not _vae_owns_tiling(vae)
              else DECODE_RAM_COPIES)
    return int(shot) * copies


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


OOM_TEXT = ("out of memory", "vram grow failed", "vram reservation failed")


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or any(t in str(e).lower() for t in OOM_TEXT)


def _deep_cleanup():
    try:
        mm.soft_empty_cache(True)
    except TypeError:
        mm.soft_empty_cache()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

DECODE_HEADROOM = 1.25

SAMPLE_HEADROOM = 1.35


def _decode_headroom(vae, latent):
    try:
        dtype = getattr(vae, "vae_dtype", None) or latent.dtype
        need = float(vae.memory_used_decode(tuple(latent.shape), dtype))
        if need > 0:
            return need * DECODE_HEADROOM
    except Exception:
        pass
    return 1e30


def _resident(models):
    out = []
    for lm in list(getattr(mm, "current_loaded_models", [])):
        for m in models or ():
            if m is None:
                continue
            try:
                p = getattr(m, "patcher", None)
                if (lm.model is m or (p is not None and lm.model is p)
                        or getattr(lm, "model", None) is getattr(m, "model", None)):
                    if lm not in out:
                        out.append(lm)
            except Exception:
                pass
    return out


def _image_out_dtype():
    try:
        return mm.intermediate_dtype()
    except Exception:
        return torch.float32


def _evict_all_but(keep_model, latent=None):
    need = 1e30
    try:
        if latent is not None:
            shape = latent["samples"].shape if isinstance(latent, dict) else latent.shape
            need = float(keep_model.model.memory_required(tuple(shape))) * SAMPLE_HEADROOM
            if not (need > 0):
                need = 1e30
    except Exception:
        need = 1e30
    keep = _resident([keep_model])
    try:
        mm.free_memory(need, mm.get_torch_device(), keep_loaded=keep)
    except Exception:
        try:
            mm.soft_empty_cache(True)
        except Exception:
            pass
    try:
        _release_pins_for(keep_model, keep)
    except Exception:
        pass

_PIN_SUBSETS = ("weights", "patches", "weights-loaded", "patches-loaded")


def _pinned_bytes(patcher):
    state = patcher.model.dynamic_pins[patcher.load_device]
    return sum(int(state[s][3][0]) for s in _PIN_SUBSETS if s in state)


def _ram_available():
    import comfy.system_memory
    return int(comfy.system_memory.virtual_memory_available())


def _ram_headroom():
    try:
        import comfy.memory_management as _cmm
        return max(_cmm.RAM_CACHE_HEADROOM / 2, 2048 * 1024 ** 2)
    except Exception:
        return 2048 * 1024 ** 2


def _pin_shortfall(keep_model):
    want = int(keep_model.model_size()) - _pinned_bytes(keep_model)
    if want <= 0:
        return 0
    short = want + _ram_headroom() - _ram_available()
    cap = getattr(mm, "MAX_PINNED_MEMORY", -1) or -1
    if cap > 0:
        short = max(short, getattr(mm, "TOTAL_PINNED_MEMORY", 0) + want - cap)
    return max(0, int(short))


def _release_pins_for(keep_model, keep):
    short = _pin_shortfall(keep_model)
    if short <= 0:
        return 0
    short += int(getattr(mm, "PIN_PRESSURE_HYSTERESIS", 256 * 1024 ** 2))
    others = [lm.model for lm in list(mm.current_loaded_models)
              if lm not in keep and lm.model is not None and lm.model is not keep_model
              and lm.model.is_dynamic()]
    others.sort(key=_pinned_bytes, reverse=True)
    if not others:
        return 0
    _sync = getattr(mm, "synchronize", None)
    if _sync is not None:
        _sync()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
    freed = 0
    for patcher in others:
        if freed >= short:
            break
        freed += int(patcher.partially_unload_ram(short - freed) or 0)
    return freed


def _host_bytes(patcher):
    state = patcher.model.dynamic_pins[patcher.load_device]
    total = 0
    for s in _PIN_SUBSETS:
        if s in state:
            size = getattr(state[s][0], "size", None)
            total += int(size) if size is not None else int(state[s][3][0])
    return total


def ensure_host_ram(need, keep=(), what="an allocation"):
    try:
        headroom = int(_ram_headroom())
        short = int(need) + headroom - _ram_available()
    except Exception:
        return 0
    if short <= 0:
        return 0
    try:
        import comfy.memory_management as _cmm
        _cmm.extra_ram_release(int(need) + headroom)
        short = int(need) + headroom - _ram_available()
    except Exception:
        pass
    if short <= 0:
        return 0
    short += int(getattr(mm, "PIN_PRESSURE_HYSTERESIS", 256 * 1024 ** 2))
    kept = _resident(keep)
    holders = []
    for lm in list(getattr(mm, "current_loaded_models", [])):
        patcher = getattr(lm, "model", None)
        try:
            if patcher is not None and patcher.is_dynamic():
                held = _host_bytes(patcher)
                if held > 0:
                    holders.append((lm in kept, -held, patcher))
        except Exception:
            continue
    holders.sort(key=lambda h: h[:2])
    freed = 0
    if holders:
        _sync = getattr(mm, "synchronize", None)
        if _sync is not None:
            _sync()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        for _, _, patcher in holders:
            if freed >= short:
                break
            try:
                freed += int(patcher.partially_unload_ram(short - freed) or 0)
            except Exception:
                continue
        if freed > 64 * 1024 ** 2:
            time.sleep(0.05)
    try:
        left = int(need) + headroom - _ram_available()
    except Exception:
        left = 0
    if left > 0:
        logging.warning(
            "H3-LongVideos: %s needs %.1f GB of RAM with %.1f GB kept free, and is "
            "still %.1f GB short after releasing %.1f GB of model weights. The "
            "finished frames are what is left holding it -- fewer shots per run or a "
            "lower megapixels is the lever.",
            what, need / 1024 ** 3, headroom / 1024 ** 3, left / 1024 ** 3,
            freed / 1024 ** 3)
    return freed


def chain_noise(latent_image, seed, noise_inds=None, fallback=None):
    parts = latent_image.unbind() if latent_image.is_nested else [latent_image]
    if noise_inds is not None or not parts or any(p.ndim not in (4, 5) for p in parts):
        return (fallback or comfy.sample.prepare_noise)(latent_image, seed, noise_inds)
    out = []
    for k, part in enumerate(parts):
        axis = 2 if part.ndim == 5 else part.ndim - 1
        gen = torch.Generator(device="cpu").manual_seed(
            (int(seed) + k * 0x9E3779B97F4A7C15) % (1 << 64))
        shape = list(part.shape)
        frames = shape.pop(axis)
        noise = torch.randn([frames] + shape, generator=gen, dtype=torch.float32,
                            device="cpu")
        out.append(noise.movedim(0, axis).contiguous().to(dtype=part.dtype))
    if latent_image.is_nested:
        return comfy.nested_tensor.NestedTensor(out)
    return out[0]


class _ChainNoise:

    def __enter__(self):
        self._was = getattr(comfy.sample, "prepare_noise", None)
        if self._was is not None:
            was = self._was
            comfy.sample.prepare_noise = (
                lambda img, seed, inds=None: chain_noise(img, seed, inds, fallback=was))
        return self

    def __exit__(self, *exc):
        if self._was is not None:
            comfy.sample.prepare_noise = self._was
        return False


def _sample_on_sigmas(model, seed, cfg, sampler_name, positive, negative, latent, sigmas):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None))
    noise = chain_noise(latent_image, seed, latent.get("batch_index"))
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

GRADE_POOL = 256
GRADE_POINTS = 33
GRADE_LUT = 1024
GRADE_JUMP = (0.15, 0.08, 0.08)
GRADE_FLAT = 0.02
GRADE_MEDIAN = 0.12
GRADE_WASHED = 0.85
GRADE_FLOOR = 1.0 / 255.0
_YCC = torch.tensor([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
_RGB = torch.linalg.inv(_YCC)
_MID = torch.tensor([0.0, 0.5, 0.5])


def _ycc(x):
    return x @ _YCC.T.to(x) + _MID.to(x)


def _rgb(y):
    return (y - _MID.to(y)) @ _RGB.T.to(y)


def tone(img):
    x = img[0] if img.dim() == 4 else img
    if x.dim() != 3 or int(x.shape[-1]) < 3 or min(int(x.shape[0]), int(x.shape[1])) < 2:
        return None
    x = x[..., :3].float().permute(2, 0, 1).unsqueeze(0)
    x = torch.nn.functional.adaptive_avg_pool2d(x, GRADE_POOL)[0].permute(1, 2, 0).reshape(-1, 3)
    return torch.quantile(_ycc(x).cpu(), torch.linspace(0, 1, GRADE_POINTS), dim=0).T


def _remap(x, src, dst):
    out = torch.empty_like(x)
    for c in range(3):
        s = src[c] + torch.arange(GRADE_POINTS, dtype=src.dtype) * 1e-6
        v = x[..., c].contiguous()
        i = torch.searchsorted(s, v).clamp(1, GRADE_POINTS - 1)
        w = ((v - s[i - 1]) / (s[i] - s[i - 1])).clamp(0, 1)
        out[..., c] = dst[c][i - 1] + w * (dst[c][i] - dst[c][i - 1]) + v - v.clamp(float(s[0]), float(s[-1]))
    return out


def _lut(src, dst):
    return _remap(torch.linspace(0, 1, GRADE_LUT).view(-1, 1).expand(-1, 3).contiguous(), src, dst)


def _flat(q):
    return q is None or float(q[0][-2] - q[0][1]) < GRADE_FLAT


def _washed(ref, q):
    spread = lambda x: float(x[0][-2] - x[0][1])
    colour = lambda x: float((x[1:] - 0.5).abs().mean())
    return (abs(float(q[0][GRADE_POINTS // 2] - ref[0][GRADE_POINTS // 2])) <= GRADE_MEDIAN
            and (spread(q) < GRADE_WASHED * spread(ref) or colour(q) < GRADE_WASHED * colour(ref)))


def shot_grade(given, first, last):
    if given is None:
        return None
    qg, qf, ql = tone(given), tone(first), tone(last)
    if _flat(qg) or _flat(qf) or _flat(ql):
        return None
    if not bool(((qg - qf).abs().mean(dim=1) <= torch.tensor(GRADE_JUMP)).all()) and not _washed(qg, qf):
        return None
    end = _remap(ql.T, qf, qg).T
    if float((qg - qf).abs().max()) < GRADE_FLOOR and float((end - ql).abs().max()) < GRADE_FLOOR:
        return None
    return _lut(qf, qg), _lut(ql, end), end


def grade_frames(frames, grade, chunk=16):
    n = int(frames.shape[0])
    if n == 0 or int(frames.shape[-1]) < 3:
        return frames
    dev = mm.get_torch_device() if torch.cuda.is_available() else frames.device
    first, last = (t.to(dev) for t in grade[:2])
    for a in range(0, n, max(1, int(chunk))):
        x = _ycc(frames[a:a + chunk, ..., :3].to(dev, torch.float32))
        k = int(x.shape[0])
        t = (torch.arange(a, a + k, dtype=torch.float32, device=dev) / max(1, n - 1)).view(k, 1, 1)
        lut = first.unsqueeze(0) * (1 - t) + last.unsqueeze(0) * t
        pos = x.clamp(0, 1).reshape(k, -1, 3) * (GRADE_LUT - 1)
        i0 = pos.floor().long().clamp(0, GRADE_LUT - 2)
        lo, hi = torch.gather(lut, 1, i0), torch.gather(lut, 1, i0 + 1)
        y = (lo + (pos - i0) * (hi - lo)).view_as(x)
        frames[a:a + k, ..., :3] = _rgb(y).clamp(0.0, 1.0).to(frames.device, frames.dtype)
    return frames


def look(img):
    q = tone(img)
    return None if _flat(q) else (float(q[0][-2] - q[0][1]), float((q[1:] - 0.5).abs().mean()))


def match_frame(frame, target):
    q = tone(frame)
    if _flat(q) or _flat(target):
        return frame
    lut = _lut(q, target)
    return grade_frames(frame, (lut, lut))
    g0, o0 = (t[:c].float().to(frames.device) for t in start)
    g1, o1 = (t[:c].float().to(frames.device) for t in end)
    for a in range(0, n, max(1, int(chunk))):
        x = frames[a:a + chunk, ..., :c].float()
        k = int(x.shape[0])
        t = (torch.arange(a, a + k, dtype=torch.float32, device=frames.device)
             / max(1, n - 1)).view(k, 1)
        g = torch.exp(g0 + (g1 - g0) * t).view(k, 1, 1, c)
        o = (o0 + (o1 - o0) * t).view(k, 1, 1, c)
        m = x.mean(dim=(1, 2), keepdim=True)
        frames[a:a + k, ..., :c] = ((x - m) * g + m + o).clamp(0.0, 1.0).to(frames.dtype)
    return frames
