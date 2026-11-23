# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import contextlib
import math
import os
import tempfile

import torch
import torch.nn.functional as F
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management as mm
import latent_preview
import node_helpers
import nodes

FPS, AUDIO_FPS, REF_AUG, SAMPLE_HEADROOM = 24, 40, 0.999, 1.35
FOLEY_DROP = ("minimax_keyframes", "minimax_refs", "minimax_visual_cond_noise_aug")
PINS = ("weights", "patches", "weights-loaded", "patches-loaded", "weights-fast", "patches-fast")
POOL, POINTS, LUT = 256, 33, 1024
JUMP, FLAT, MEDIAN, WASHED, FLOOR = (0.15, 0.08, 0.08), 0.02, 0.12, 0.85, 1.0 / 255.0
TOL, LIFT, BOOST = 0.03, 1.25, 1.25
VOICE_LO, VOICE_HI, VOICED, LOUDER, VOICE_BAND, VOICE_SHARE, VOICE_GAP, VOICE_RUN = 75, 400, 0.5, 3.0, 1000, 0.5, 5, 10
FINE, CRISP_TOL, CRISP_MIN, CRISP_STEP = 400, 0.05, 0.6, 4
FRESH_SCALE, FRESH_FINE, FRESH_THIN, FRESH_STREAK, FRESH_TURNS, FRESH_MARGIN, FRESH_STEP = 96, 256, 768, 48, 16, 24, 6
FRESH_LO, FRESH_STILL, FRESH_HI, FRESH_LEVEL, FRESH_DARK, FRESH_LINE, FRESH_SURE = 0.04, 0.06, 0.10, 0.3, 0.02, 0.03, 0.04
_YCC = torch.tensor([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
_RGB = torch.linalg.inv(_YCC)
_MID = torch.tensor([0.0, 0.5, 0.5])
_SILENCE = {}


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or any(
        t in str(e).lower() for t in ("out of memory", "vram grow failed", "vram reservation failed"))


def cleanup():
    mm.soft_empty_cache(True)


def _device(x):
    return mm.get_torch_device() if torch.cuda.is_available() else x.device


def resize(img, w, h):
    x = img[..., :3]
    if (int(x.shape[2]), int(x.shape[1])) == (w, h):
        return x
    way = "bicubic" if w * h > int(x.shape[2]) * int(x.shape[1]) else "area"
    return comfy.utils.common_upscale(x.movedim(-1, 1), w, h, way, "disabled").movedim(1, -1).clamp(0, 1)


def silence(audio_vae, length):
    sr = int(getattr(audio_vae, "audio_sample_rate", 0) or 0)
    if sr <= 0:
        return None
    if _SILENCE.get("key") != (id(audio_vae), sr):
        enc = audio_vae.encode(torch.zeros((1, sr * 2, 2)))
        if enc is None or enc.dim() != 4 or enc.shape[-1] <= 9:
            return None
        _SILENCE.update(key=(id(audio_vae), sr), block=enc[..., 4:-4].detach().to("cpu").clone())
    block = _SILENCE["block"]
    pieces = [block if k % 2 == 0 else torch.flip(block, dims=[-1]) for k in range(-(-length // block.shape[-1]))]
    return torch.cat(pieces, dim=-1)[..., :length]


def spans(audio, cues):
    mask = torch.zeros_like(audio[:, :1])
    for a, b in cues:
        mask[..., round(a * AUDIO_FPS):round(b * AUDIO_FPS)] = 1.0
    return mask


def gaps(cues, length):
    out, at = [], 0.0
    for a, b in sorted(cues):
        out += [(at, a)] if a > at else []
        at = max(at, b)
    return out + ([(at, length)] if length > at else [])


def unvoice(wave, sr, keep):
    x = wave.mean(dim=1)[0].float()
    hop, win = sr // 100, sr // 25
    if x.numel() < win:
        return wave
    frames = x.unfold(0, win, hop)
    frames = frames - frames.mean(dim=1, keepdim=True)
    power = torch.fft.rfft(frames, n=2 * win).abs() ** 2
    hz = torch.fft.rfftfreq(2 * win, 1 / sr, device=x.device)
    low = power * ((hz >= VOICE_LO) & (hz <= VOICE_BAND))
    ac = torch.fft.irfft(low, n=2 * win)[:, :win]
    peak = ac[:, sr // VOICE_HI:sr // VOICE_LO + 1].max(dim=1).values / ac[:, 0].clamp(min=1e-12)
    share = low.sum(dim=1) / power.sum(dim=1).clamp(min=1e-12)
    rms = frames.pow(2).mean(dim=1).sqrt()
    t = torch.arange(len(rms), device=rms.device) * hop / sr
    gap = torch.ones_like(rms, dtype=torch.bool)
    for a, b in keep:
        gap &= (t + win / sr <= a) | (t >= b)
    if not bool(gap.any()):
        return wave
    floor = torch.quantile(rms[gap], 0.2)
    voiced = (peak > VOICED) & (share > VOICE_SHARE) & (rms > LOUDER * floor) & gap
    grow = lambda m, k: F.max_pool1d(m.float()[None, None], 2 * k + 1, 1, k)[0, 0] > 0
    joined = ~grow(~grow(voiced, VOICE_GAP), VOICE_GAP)
    voiced = grow(voiced & grow(~grow(~joined, VOICE_RUN), VOICE_RUN), 8) & gap
    gain = torch.where(voiced, (floor / rms.clamp(min=1e-9)).clamp(max=1.0), torch.ones_like(rms))
    gain = F.avg_pool1d(gain[None, None], 5, 1, 2, count_include_pad=False)
    return wave * F.interpolate(gain, size=int(wave.shape[-1]), mode="linear")[0, 0].to(wave)


def build(clip, vae, audio_vae, text, w, h, frames, handoff=None, refs=(), memory=None, silent=False, cues=(), memo=None):
    video = torch.zeros([1, 24, 2 if frames <= 5 else (frames - 5) // 17 * 5 + 2, h // 16, w // 16], device=mm.intermediate_device())
    audio = torch.zeros([1, 32, 2, round(frames / FPS * AUDIO_FPS)], device=mm.intermediate_device())
    memo, blocks = {} if memo is None else memo, []

    def picture(img, tw, th):
        if (id(img), tw, th) not in memo:
            pic = resize(img[:1], tw, th)
            memo[(id(img), tw, th)] = (img, pic, vae.encode(pic))
        blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": memo[(id(img), tw, th)][2]})
        return {"type": "image", "data": memo[(id(img), tw, th)][1]}

    def fit(img):
        s = min(1.0, math.sqrt(w * h / float(max(1, int(img.shape[2]) * int(img.shape[1])))))
        return tuple(max(32, round(v * s / 32) * 32) for v in (int(img.shape[2]), int(img.shape[1])))

    items = [picture(img, *fit(img)) for img in refs if img is not None]
    hand = resize(handoff[:1], w, h) if handoff is not None else None
    items += ([{"type": "image", "data": hand}] if hand is not None else []) + ([picture(memory, w, h)] if memory is not None else [])
    vals = {"minimax_refs": blocks, "minimax_visual_cond_noise_aug": REF_AUG} if blocks else {}
    if hand is not None:
        vals["minimax_keyframes"] = [{"resolved_frame_index": 0, "latent": vae.encode(hand)}]
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text, minimax_ref_items=items) if items else clip.tokenize(text))
    cond = node_helpers.conditioning_set_values(cond, vals) if vals else cond
    quiet = silence(audio_vae, int(audio.shape[-1])) if silent or cues else None
    if quiet is None:
        return cond, {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    return cond, {"samples": comfy.nested_tensor.NestedTensor((video, quiet.to(audio))),
                  "noise_mask": comfy.nested_tensor.NestedTensor((torch.ones_like(video[:, :1]), spans(audio, cues)))}


def spool(n, shape, dtype):
    size = n * math.prod(shape)
    need = size * torch.tensor([], dtype=dtype).element_size()
    try:
        with tempfile.NamedTemporaryFile(prefix="h3longvideos-", suffix=".frames") as f:
            os.posix_fallocate(f.fileno(), 0, need)
            return torch.from_file(f.name, shared=True, size=size, dtype=dtype).view(n, *shape)
    except Exception as e:
        free = mm.get_free_memory(torch.device("cpu"))
        if need > free:
            raise MemoryError(f"the frames ({need / 2 ** 30:.1f} GB) cannot go to a temp file ({e}) and only "
                              f"{free / 2 ** 30:.1f} GB of RAM is free") from e
        return torch.empty((n, *shape), dtype=dtype, device="cpu")


def sample(model, cond, negative, latent, seed, steps, sampler, scheduler, sigmas):
    try:
        if sigmas is None:
            return nodes.common_ksampler(model, seed, steps, 1.0, sampler, scheduler, cond, negative, latent, denoise=1.0)[0]
        img = latent["samples"]
        out = comfy.sample.sample_custom(model, comfy.sample.prepare_noise(img, seed), 1.0, comfy.samplers.sampler_object(sampler),
                                         sigmas, cond, negative, img, noise_mask=latent.get("noise_mask"),
                                         callback=latent_preview.prepare_callback(model, len(sigmas) - 1),
                                         disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
        return dict(latent, samples=out)
    except Exception as e:
        if is_oom(e):
            raise RuntimeError(f"not enough VRAM for this shot ({e}): lower megapixels or shot_seconds") from e
        raise


def foley(model, cond, negative, latent, out, seed, steps, sampler, scheduler, sigmas, scale, cues=None):
    video, audio = out["samples"].unbind()
    small = video
    if scale < 1.0 and min(video.shape[-2:]) >= 32:
        small = F.interpolate(video.float(), size=(video.shape[2],) + tuple(max(2, 2 * round(n * scale / 2)) for n in video.shape[-2:]),
                              mode="area").to(video.dtype)
        cond = [[c, {k: v for k, v in d.items() if k not in FOLEY_DROP}] for c, d in cond]
    heard = torch.ones_like(audio[:, :1]) if cues is None else spans(audio, cues)
    lat = dict(latent, samples=comfy.nested_tensor.NestedTensor((small, torch.zeros_like(audio) if cues is None else audio)),
               noise_mask=comfy.nested_tensor.NestedTensor((torch.zeros_like(small[:, :1]), heard)))
    res = sample(model, cond, negative, lat, seed, steps, sampler, scheduler, sigmas)
    return dict(res, samples=comfy.nested_tensor.NestedTensor((video, res["samples"].unbind()[1])))


def _pinned(patcher):
    try:
        state = patcher.model.dynamic_pins[patcher.load_device]
        return sum(int(state[s][3][0]) for s in PINS if s in state)
    except Exception:
        return 0


def _loaded(models):
    return [lm for lm in list(mm.current_loaded_models)
            if any(m is not None and (lm.model is m or lm.model is getattr(m, "patcher", None)) for m in models)]


def free_ram(need, keep=()):
    with contextlib.suppress(Exception):
        import comfy.memory_management as cmm
        import comfy.system_memory as sm
        room = max(cmm.RAM_CACHE_HEADROOM // 2, 2 << 30)
        short = lambda: int(need) + room - int(sm.virtual_memory_available())
        if short() <= 0:
            return
        cmm.extra_ram_release(int(need) + room)
        if short() <= 0:
            return
        kept = _loaded(keep)
        held = sorted((lm for lm in list(mm.current_loaded_models) if lm.model is not None and lm.model.is_dynamic()),
                      key=lambda lm: (lm in kept, -_pinned(lm.model)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for lm in held:
            if (left := short()) <= 0:
                break
            lm.model.partially_unload_ram(left + (256 << 20))


def make_room(model, latent):
    try:
        need = float(model.model.memory_required(tuple(latent["samples"].shape))) * SAMPLE_HEADROOM
    except Exception:
        need = 1e30
    try:
        mm.free_memory(need, mm.get_torch_device(), keep_loaded=_loaded([model]))
    except Exception:
        cleanup()
    with contextlib.suppress(Exception):
        free_ram(max(0, int(model.model_size()) - _pinned(model)), keep=[model])


def decode(vae, audio_vae, out, keep=()):
    video, audio = out["samples"].unbind()
    frames = (int(video.shape[2]) - 2) // 5 * 17 + 5 if int(video.shape[2]) > 2 else 5
    free_ram(2 * frames * int(video.shape[-2]) * 16 * int(video.shape[-1]) * 16 * 3 * 2, keep=keep)
    try:
        imgs = vae.decode(video)
    except Exception as e:
        if not is_oom(e):
            raise
        cleanup()
        imgs = vae.decode_tiled(video)
    wave = audio_vae.decode(audio).movedim(-1, 1)
    wave = wave / (torch.std(wave, dim=[1, 2], keepdim=True) * 5.0).clamp(min=1.0)
    sr = getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 44100))
    return imgs.reshape(-1, *imgs.shape[-3:]), wave, int(sr)


def _ycc(x):
    return x @ _YCC.T.to(x) + _MID.to(x)


def _rgb(y):
    return (y - _MID.to(y)) @ _RGB.T.to(y)


def tone(img):
    x = img[0] if img.dim() == 4 else img
    x = F.adaptive_avg_pool2d(x[..., :3].float().permute(2, 0, 1).unsqueeze(0), POOL)
    return torch.quantile(_ycc(x[0].permute(1, 2, 0).reshape(-1, 3)).cpu(), torch.linspace(0, 1, POINTS), dim=0).T


def _stats(q):
    return float(q[0][-2] - q[0][1]), float((q[1:] - 0.5).abs().mean())


def look(img):
    stats = _stats(tone(img))
    return None if stats[0] < FLAT else stats


def _remap(x, src, dst):
    out = torch.empty_like(x)
    for c in range(3):
        s = src[c] + torch.arange(POINTS, dtype=src.dtype) * 1e-6
        v = x[..., c].contiguous()
        i = torch.searchsorted(s, v).clamp(1, POINTS - 1)
        w = ((v - s[i - 1]) / (s[i] - s[i - 1])).clamp(0, 1)
        out[..., c] = dst[c][i - 1] + w * (dst[c][i] - dst[c][i - 1]) + v - v.clamp(float(s[0]), float(s[-1]))
    return out


def grade(given, first, last):
    if given is None:
        return None
    qg, qf, ql = tone(given), tone(first), tone(last)
    if any(_stats(q)[0] < FLAT for q in (qg, qf, ql)):
        return None
    (spread, colour), (was_spread, was_colour) = _stats(qf), _stats(qg)
    washed = abs(float(qf[0][POINTS // 2] - qg[0][POINTS // 2])) <= MEDIAN and (
        spread < WASHED * was_spread or colour < WASHED * was_colour)
    if not bool(((qg - qf).abs().mean(dim=1) <= torch.tensor(JUMP)).all()) and not washed:
        return None
    end = _remap(ql.T, qf, qg).T
    if float((qg - qf).abs().max()) < FLOOR and float((end - ql).abs().max()) < FLOOR:
        return None
    ramp = torch.linspace(0, 1, LUT).view(-1, 1).expand(-1, 3).contiguous()
    return _remap(ramp, qf, qg), _remap(ramp, ql, end)


def _each(frames, dev, fn):
    n = int(frames.shape[0])
    for s in range(0, n, 16):
        x = frames[s:s + 16, ..., :3].to(dev, torch.float32)
        t = torch.arange(s, s + int(x.shape[0]), dtype=torch.float32, device=dev) / max(1, n - 1)
        frames[s:s + int(x.shape[0]), ..., :3] = fn(x, t, s).clamp(0, 1).to(frames.device, frames.dtype)
    return frames


def apply(frames, luts):
    dev = _device(frames)
    first, last = (t.to(dev) for t in luts)

    def graded(x, t, s):
        x, k = _ycc(x), int(x.shape[0])
        lut = first.unsqueeze(0) * (1 - t.view(k, 1, 1)) + last.unsqueeze(0) * t.view(k, 1, 1)
        pos = x.clamp(0, 1).reshape(k, -1, 3) * (LUT - 1)
        i0 = pos.floor().long().clamp(0, LUT - 2)
        lo, hi = torch.gather(lut, 1, i0), torch.gather(lut, 1, i0 + 1)
        return _rgb((lo + (pos - i0) * (hi - lo)).view_as(x))
    return _each(frames, dev, graded)


def _held(want, have):
    have = max(have, 1e-6)
    return min(LIFT, max(1.0, want * (1 - TOL) / have), want * (1 + TOL) / have)


def _bands(y):
    s = max(1.0, int(y.shape[-2]) / FINE)
    base = _blur(y, s)
    return y - base, base - _blur(base, 4 * s), s


def _rate(x):
    y = _ycc(x[..., :3].float())[..., :1].permute(0, 3, 1, 2)
    fine, mid, s = _bands(y)
    small = _blur(F.avg_pool2d(y, 4), s)
    coarse = (small - _blur(small, 4 * s)).abs().mean(dim=(1, 2, 3)).clamp(min=1e-6)
    return torch.stack([fine.abs().mean(dim=(1, 2, 3)), mid.abs().mean(dim=(1, 2, 3))], 1) / coarse[:, None]


def crisp(img):
    return tuple(float(v) for v in _rate(img.reshape(-1, *img.shape[-3:])[:1])[0])


def _ease(x, keep):
    fine, mid, _ = _bands(_ycc(x)[..., :1].permute(0, 3, 1, 2))
    cut = (1 - keep[:, :1]).view(-1, 1, 1, 1) * fine + (1 - keep[:, 1:]).view(-1, 1, 1, 1) * mid
    return x - cut.permute(0, 2, 3, 1)


def _keep(frames, aim, dev):
    n = int(frames.shape[0])
    if not aim or n < 2:
        return None
    at, ks = torch.linspace(0, n - 1, max(2, -(-n // CRISP_STEP))).round().long(), []
    want = torch.tensor(aim, device=dev) * (1 + CRISP_TOL)
    for s in range(0, len(at), 16):
        x = frames[at[s:s + 16]].to(dev, torch.float32)
        r0 = _rate(x)
        k0 = (want / r0.clamp(min=1e-6)).clamp(CRISP_MIN, 1.0)
        cut = ((r0 - want) / (r0 - _rate(_ease(x, k0))).clamp(min=1e-6) * (1 - k0)).clamp(0.0, 1.0 - CRISP_MIN)
        ks.append(torch.where(r0 > want, 1 - cut, torch.ones_like(r0)).cpu())
    k = torch.cat(ks).T.unsqueeze(0)
    if float(k.min()) >= 1.0:
        return None
    k = F.avg_pool1d(F.pad(k, (1, 1), mode="replicate"), 3, 1)
    return F.interpolate(k, size=n, mode="linear", align_corners=True)[0].T


def hold(frames, scene, aim=None):
    end = look(frames[-1]) if int(frames.shape[0]) > 1 and scene else None
    gs, gc = (_held(a, e) for a, e in zip(scene, end)) if end else (1.0, 1.0)
    dev = _device(frames)
    keep = _keep(frames, aim, dev)
    if gs == gc == 1.0 and keep is None:
        return False, 1.0

    def held(x, t, s):
        x, t = _ycc(_ease(x, keep[s:s + int(x.shape[0])].to(dev)) if keep is not None else x), t.view(-1, 1, 1, 1)
        m = x[..., :1].mean(dim=(1, 2), keepdim=True)
        x[..., :1] = (x[..., :1] - m) * (1 + (gs - 1) * t) + m
        x[..., 1:] = (x[..., 1:] - 0.5) * (1 + (gc - 1) * t) + 0.5
        return _rgb(x)
    _each(frames, dev, held)
    return gs != 1.0 or gc != 1.0, float(keep.min()) if keep is not None else 1.0


def soften(frame, boost):
    sc, kc = (min(BOOST, max(1.0, float(b))) for b in boost) if boost else (1.0, 1.0)
    if sc == kc == 1.0:
        return frame
    x = frame.clone()
    y = _ycc(x[..., :3].float())
    m = y[..., :1].mean(dim=(-3, -2), keepdim=True)
    y[..., :1] = (y[..., :1] - m) / sc + m
    y[..., 1:] = (y[..., 1:] - 0.5) / kc + 0.5
    x[..., :3] = _rgb(y).clamp(0.0, 1.0).to(x.dtype)
    return x


def _blur(x, s):
    r = min(int(3 * s) + 1, int(min(x.shape[-2:])) - 1)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) / s) ** 2)
    k = (k / k.sum()).view(1, 1, 1, -1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.transpose(2, 3))


def _streaks(n, dev):
    t, ks = torch.linspace(-(n // 2), n // 2, 4 * n), torch.zeros(FRESH_TURNS, 1, n, n)
    for i in range(FRESH_TURNS):
        a = math.pi * i / FRESH_TURNS
        ks[i, 0, (n // 2 + t * math.sin(a)).round().long(), (n // 2 + t * math.cos(a)).round().long()] = 1
    return (ks / ks.sum((2, 3), keepdim=True)).to(dev)


def refresh(frame, backdrop, frames):
    if backdrop is None or frame.shape != backdrop.shape:
        return frame
    h, w, dev = int(frame.shape[1]), int(frame.shape[2]), _device(frame)
    rgb = lambda t: t[..., :3].to(dev, torch.float32).permute(0, 3, 1, 2)
    soft = lambda t, s: _blur(t.reshape(-1, 1, h, w), max(1.0, h / s)).view_as(t)
    lines = _streaks(2 * max(1, round(h / FRESH_STREAK)) + 1, dev)

    def ramp(p, q, lo, hi=FRESH_HI, d=None):
        gain = ((torch.maximum(p, q).amax(1, keepdim=True) + FRESH_DARK) / (FRESH_LEVEL + FRESH_DARK)).clamp(max=1.0)
        return ((((p - q).abs() if d is None else d).amax(1, keepdim=True) / gain - lo) / (hi - lo)).clamp(0, 1)

    def thin(t):
        near, broad = soft(t, FRESH_THIN), soft(t, FRESH_SCALE)
        return near, broad, near - broad

    def line(p, q):
        d = F.conv2d((p[2] - q[2]).reshape(-1, 1, h, w), lines, padding=lines.shape[-1] // 2).abs().amax(1).view_as(p[2])
        return ramp(p[0], q[0], FRESH_LINE, FRESH_SURE, d)

    x, a = rgb(frame), rgb(backdrop)
    now, was, end = thin(x), thin(a), soft(x, FRESH_FINE)
    moved = torch.maximum(ramp(now[1], was[1], FRESH_LO), line(now, was))
    for i in range(0, int(frames.shape[0]), FRESH_STEP):
        f = F.interpolate(rgb(frames[i:i + 1]), size=(h, w), mode="area")
        moved = torch.maximum(moved, torch.maximum(ramp(soft(f, FRESH_FINE), end, FRESH_STILL), line(thin(f), now)))
    r = max(1, round(h / FRESH_MARGIN))
    same = 1 - soft(F.max_pool2d(moved, 2 * r + 1, stride=1, padding=r), FRESH_SCALE)
    out = frame.clone()
    out[..., :3] = (x + same * (a - x)).permute(0, 2, 3, 1).clamp(0, 1).to(frame.device, frame.dtype)
    return out
