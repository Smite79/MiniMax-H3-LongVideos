# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import math
import sys
import time

import torch
import comfy.samplers
import comfy.model_management as mm

try:
    from . import chain, faces, modelprep, planner
except ImportError:
    import chain
    import faces
    import modelprep
    import planner

NATIVE = {"16:9": (1344, 768), "9:16": (768, 1344), "4:3": (1024, 768), "3:4": (768, 1024), "1:1": (768, 768),
          "21:9": (1536, 672), "9:21": (672, 1536)}
FOLEY = {"off": 0.0, "quarter": 0.25, "half": 0.5, "full": 1.0}
RECALL = " <Picture {}> is this same place a little earlier, with {} in view: {} {} exactly as there on coming back into view."


def show(node_id, lines):
    server = getattr(getattr(sys.modules.get("server"), "PromptServer", None), "instance", None)
    if server is not None and node_id is not None:
        server.send_progress_text("\n".join(lines), str(node_id))


def frame_size(choice, megapixels):
    w, h = NATIVE.get(choice, NATIVE["16:9"])
    s = math.sqrt(megapixels) if megapixels and megapixels > 0 else 1.0
    out = tuple(max(32, round(v * s / 32) * 32) for v in (w, h))
    return (out if s <= 1 else (w, h)), out


class H3LongVideos:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "resolution": (list(NATIVE), {"default": "16:9"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "shot_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
                "sigmas": ("SIGMAS",),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "silence_wordless": ("BOOLEAN", {"default": True}),
                "plan_only": ("BOOLEAN", {"default": False}),
                "anchor": ("STRING", {"multiline": True, "default": ""}),
                "character_memory": ("STRING", {"multiline": True, "default": ""}),
                "shot_length": (["from the beat", "fixed"], {"default": "from the beat"}),
                "foley_resolution": (list(FOLEY), {"default": "half"}),
            },
            "hidden": {"graph": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("images", "audio", "info", "script", "frames_per_shot", "total_frames", "shots", "video_seconds")
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"

    def run(self, model, clip, vae, audio_vae, prompt, resolution, megapixels, shot_seconds, steps, sampler_name, scheduler,
            seed, first_frame=None, ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None, sigmas=None,
            shift_video=12.0, shift_audio=3.0, silence_wordless=True, plan_only=False, anchor="", character_memory="",
            shot_length="from the beat", foley_resolution="half", graph=None, unique_id=None, **legacy):
        t0 = time.perf_counter()
        modelprep.check_vaes(vae, audio_vae)
        modelprep.mend(model, graph, unique_id)
        (w, h), (ow, oh) = frame_size(resolution, megapixels)
        pictures = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        refs_ok, why = modelprep.takes_pictures(model, graph, unique_id)
        shots, warn = planner.plan(prompt, character_memory, anchor, shot_seconds, shot_length != "fixed",
                                   first_frame is not None, pictures, silence_wordless)
        if not shots:
            raise ValueError("the prompt has no beats")
        script = "\n\n".join(f"[shot {s['n']}]\n{s['prompt']}" for s in shots)
        info = [f"{w}x{h}" + (f", scaled up to {ow}x{oh} after rendering" if (ow, oh) != (w, h) else "") + f", {len(shots)} shots"] + warn
        info += [f"LORA ONLY PARTLY LOADED: {', '.join(part)} carries timestep layers this checkpoint has no place for"
                 for part in [modelprep.partial(model, graph, unique_id)] if part]
        info += [f"{why}, so no earlier frame of the scene is fed back in when a face comes back into view"] if not refs_ok else []
        info += [f"ignored inputs from an older version of this node: {', '.join(sorted(legacy))}"] if legacy else []
        if plan_only:
            return (torch.zeros((1, oh, ow, 3)), {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 44100},
                    " | ".join(info + [s["line"] for s in shots]), script, shots[0]["frames"], 0, len(shots), 0.0)
        model, steps, sampler_name, sigmas, fine, notes = modelprep.prepare(model, steps, sampler_name, scheduler, sigmas,
                                                                            shift_video, shift_audio, graph, unique_id)
        info += notes
        foley_sigmas, voice_sigmas = fine or (sigmas, None)
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        total = sum(s["frames"] - (1 if s["keyed"] and i else 0) for i, s in enumerate(shots))
        detector = faces.Detector() if faces.available() else None
        crops, store, used, waves, sr, memo = {}, None, 0, [], 44100, {}
        try:
            wanted = sorted({k for s in shots for k, face in zip(s["refs"], s["faces"]) if face})
            info += ["the pictures went in whole: cropping them to face and hair needs comfyui_controlnet_aux and its DWPose "
                     "files"] if wanted and not detector else []
            for k in wanted if detector else ():
                crops[k] = faces.crop(detector, pictures[k])
                info += [f"ref_image_{k + 1}: no face found, the whole picture went in"] if crops[k] is None else []
            handoff = first_frame[:1] if first_frame is not None else None
            backdrop, scene, aim, recalled, shown, boost, base = handoff, None, None, None, 0, None, None
            scale = FOLEY.get(foley_resolution, 0.5)
            for i, shot in enumerate(shots):
                started = time.perf_counter()
                show(unique_id, info + [f"RENDERING SHOT {shot['n']} OF {len(shots)}, {started - t0:.0f}s in:"]
                     + [s["line"] for s in shots[i:]])
                given = handoff if shot["keyed"] else None
                fed = chain.soften(given, boost) if given is not None else None
                refs = [crops.get(k) if face and crops.get(k) is not None else pictures[k]
                        for k, face in zip(shot["refs"], shot["faces"])]
                recall = recalled[0] if recalled and given is not None and shot["coming"] and shown < len(shot["cast"]) \
                    and set(shot["coming"]) <= recalled[2] else None
                back = planner.join(shot["coming"]) if recall is not None else ""
                text = shot["prompt"] + (RECALL.format(len(refs) + 2, back, back, "looks" if len(shot["coming"]) == 1 else "look")
                                         if back else "")
                lines, muffles = ([(a, b) for a, b, kind in shot["cues"] if kind == k] for k in ("line", "muffle"))
                cond, latent = chain.build(clip, vae, audio_vae, text, w, h, shot["frames"], fed, refs, recall,
                                           shot["silent"], lines, memo)
                took = [time.perf_counter()]
                chain.make_room(model, latent)
                out = chain.sample(model, cond, negative, latent, seed, steps, sampler_name, scheduler, sigmas)
                took.append(time.perf_counter())
                if scale and voice_sigmas is not None and lines:
                    chain.make_room(model, latent)
                    out = chain.foley(model, cond, negative, latent, out, seed, steps, sampler_name, scheduler, voice_sigmas, scale,
                                      lines)
                if scale and (shot["silent"] or lines):
                    sound, cued = clip.encode_from_tokens_scheduled(clip.tokenize(shot["sound"])), lines + muffles
                    for windows in [muffles] * bool(muffles) + [chain.gaps(cued, shot["frames"] / planner.FPS) if cued else None]:
                        chain.make_room(model, latent)
                        out = chain.foley(model, sound, negative, latent, out, seed, steps, sampler_name, scheduler, foley_sigmas,
                                          scale, windows)
                took.append(time.perf_counter())
                del cond, latent
                imgs, wave, sr = chain.decode(vae, audio_vae, out, keep=(vae, audio_vae))
                wave = chain.unvoice(wave, sr, lines + muffles) if shot["mute"] else wave
                took.append(time.perf_counter())
                del out
                opening = imgs[1] if given is not None and len(imgs) > 2 else imgs[0]
                raw, ref, fed_look = chain.look(opening), *(chain.look(x) if x is not None else None for x in (given, fed))
                if raw and fed_look:
                    gain = (raw[0] / fed_look[0], raw[1] / max(fed_look[1], 1e-6))
                    boost = gain if boost is None else tuple((a + b) / 2 for a, b in zip(boost, gain))
                luts = chain.grade(given, opening, imgs[-1]) if len(imgs) > 1 else None
                if luts:
                    chain.apply(imgs, luts)
                scene = (scene or ref) if given is not None else chain.look(imgs[len(imgs) // 2])
                aim = aim if given is not None and aim else chain.crisp(imgs[len(imgs) // 2])
                held, eased = chain.hold(imgs, scene, aim)
                seen, sharp = chain.look(imgs[-1]), chain.crisp(imgs[-1])
                base = base or (seen, sharp)
                handoff = imgs[-1:].detach().clamp(0, 1).to("cpu", copy=True)
                handoff = backdrop = chain.refresh(handoff, backdrop, imgs) if given is not None else handoff
                if detector and refs_ok:
                    shown = faces.count(detector, handoff)
                    detector.close()
                    recalled = None if given is None or shot["changed"] else recalled
                    if shown and shown >= max(recalled[1] if recalled else 0, len(shot["cast"])):
                        recalled = (handoff, shown, set(shot["cast"]))
                if given is not None and i:
                    imgs, wave = imgs[1:], wave[..., round(sr / planner.FPS):]
                want = round(int(imgs.shape[0]) * sr / planner.FPS)
                wave = torch.nn.functional.pad(wave[..., :want], (0, max(0, want - int(wave.shape[-1]))))
                if store is None:
                    store = chain.spool(total, (oh, ow) + tuple(imgs.shape[3:]), mm.intermediate_dtype())
                n = min(int(imgs.shape[0]), total - used)
                for a in range(0, n, 16):
                    store[used + a:used + min(n, a + 16)].copy_(chain.resize(imgs[a:min(n, a + 16)], ow, oh))
                used += n
                waves.append(wave.to("cpu", copy=True))
                del imgs, wave
                chain.cleanup()
                figures = [f"opened at contrast {raw[0] / ref[0]:.2f}, colour {raw[1] / max(ref[1], 1e-6):.2f} of the frame "
                           f"before"] if raw and ref else []
                figures += ["graded onto the frame before"] if luts else []
                figures += ["look held to the scene"] if held else []
                figures += [f"detail eased to {eased:.2f} against oversharpening"] if eased < 1.0 else []
                figures += [f"end against shot 1: contrast {seen[0] / base[0][0]:.2f}, colour {seen[1] / max(base[0][1], 1e-6):.2f}, "
                            f"sharpness {sum(a / max(b, 1e-6) for a, b in zip(sharp, base[1])) / 2:.2f}"] if seen and base[0] else []
                took = [b - a for a, b in zip([started] + took, took + [time.perf_counter()])]
                info.append(", ".join([shot["line"]] + ([f"an earlier frame with {back} in view went in"] if recall is not None
                                                         else []) + figures + [f"{sum(took):.0f}s (" + ", ".join(
                    f"{k} {v:.0f}s" for k, v in zip(("encode", "sampling", "sound", "decode", "finish"), took)
                    if k != "sound" or v >= 0.5) + ")"]))
        finally:
            if detector:
                detector.close()
        video = store[:used]
        info.append(f"{used} frames, {used / planner.FPS:.1f}s, rendered in {time.perf_counter() - t0:.0f}s")
        return (video, {"waveform": torch.cat(waves, dim=-1), "sample_rate": sr}, " | ".join(info), script,
                shots[0]["frames"], used, len(shots), used / planner.FPS)


NODE_CLASS_MAPPINGS = {name: H3LongVideos for name in ("H3LongVideos", "H3LongVideosFL2VA", "H3LongVideosV1", "H3LongVideosREF2VA")}
NODE_DISPLAY_NAME_MAPPINGS = {name: "H3-LongVideos" for name in NODE_CLASS_MAPPINGS}
