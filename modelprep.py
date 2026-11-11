# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import copy
import functools
import inspect
import json
import os
import re

import torch
import comfy.samplers
import comfy.utils
import comfy.patcher_extension as pe
from comfy.ldm.minimax.model import TimeEmbedder, time_shift_sigma

PDD_KEY = "minimax_h3_pdd_acc"
PDD_NODE = "MiniMaxH3PDDAccApply"
FAST_SHIFT = 10.0
HYPERFLOW = {"sigmas": (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0), "video": 12.0, "audio": 3.0}
FLASH = {4: (1.0, 0.681872, 0.205019, 0.036496, 0.0), 6: (1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0),
         8: HYPERFLOW["sigmas"]}
FLASH_KEYS = {"endpoint_time_embedder.proj_in.weight", "hyperflow_sigmas"}
FINE_BELOW, REVOICE_FROM = 6, 0.88
RECIPES = {"flashgen": {"grid": (1.0, 0.7, 0.4, 0.15, 0.0), "sampler": "euler", "video": 12.0, "audio": 3.0},
           "dmad": {"steps": 4, "sampler": "lcm", "video": 12.0, "audio": 2.0}}
SPEED = re.compile(r"\d+[\s_-]?(?:to[\s_-]?\d+[\s_-]?)?steps?|turbo|lightning|distill|lcm", re.I)
BLIND = re.compile(r"fl2v|t2va|t2v|\d+[\s_-]?steps?|turbo|lightning|distill|dmad|flashgen|lcm", re.I)
CHECKPOINT_KEYS = ("unet_name", "ckpt_name", "model_name")
TOKEN_BYTES, FIXED_BYTES, TOKEN_VALUES = 120 * 1024, 1024 ** 3, 96


def meta(path):
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            head = json.loads(f.read(n)) if 0 < n < 64 << 20 else {}
        return set(head) - {"__metadata__"}, head.get("__metadata__") or {}
    except Exception:
        return set(), {}


def full_path(kind, name):
    try:
        import folder_paths
        return folder_paths.get_full_path(kind, name) or ""
    except Exception:
        return ""


def upstream(graph, node_id):
    seen, todo, out = set(), [str(node_id)], []
    while todo and isinstance(graph, dict):
        nid = todo.pop(0)
        if nid in seen or not isinstance(graph.get(nid), dict):
            continue
        seen.add(nid)
        inputs = graph[nid].get("inputs") or {}
        if nid != str(node_id):
            out.append(inputs)
            if any(k in CHECKPOINT_KEYS and isinstance(v, str) for k, v in inputs.items()):
                break
        todo += [str(v[0]) for k, v in inputs.items() if isinstance(v, list) and len(v) == 2
                 and (k == "model" or nid != str(node_id))]
    return out


def wired(graph, node_id, name):
    src = ((graph.get(str(node_id)) or {}).get("inputs") or {}).get(name) if isinstance(graph, dict) else None
    return str((graph.get(str(src[0])) or {}).get("class_type", "")) if isinstance(src, list) and len(src) == 2 else ""


def model_file(graph, node_id):
    for inputs in upstream(graph, node_id):
        for k, v in inputs.items():
            path = full_path("diffusion_models", v) or full_path("checkpoints", v) if isinstance(v, str) else ""
            if path or k in CHECKPOINT_KEYS and isinstance(v, str):
                return v, path
    return "", ""


def checkpoint(graph, node_id):
    name, path = model_file(graph, node_id)
    info = meta(path)[1] if path else {}
    return name, " ".join(str(info.get(k, "")) for k in ("source_a", "source_b", "source", "base_model", "description"))


def loras(graph, node_id=None):
    try:
        import folder_paths
        known = set(folder_paths.get_filename_list("loras"))
    except Exception:
        return []
    found = []

    def walk(v, on=True):
        if isinstance(v, str) and v in known:
            found.extend([v] if on else [])
        elif isinstance(v, str) and v[:1] in "[{":
            try:
                walk(json.loads(v), on)
            except ValueError:
                pass
        elif isinstance(v, dict):
            on = on and v.get("on") is not False and v.get("enabled") is not False and not any(
                v.get(k) in (0, 0.0) for k in ("str", "strength", "strength_model"))
            for k, x in v.items():
                walk(x, on and k not in CHECKPOINT_KEYS)
        elif isinstance(v, list) and not (len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)):
            for x in v:
                walk(x, on)
    for inputs in upstream(graph, node_id) if node_id is not None else [
            n.get("inputs") or {} for n in (graph or {}).values() if isinstance(n, dict)]:
        walk(inputs)
    return list(dict.fromkeys(found))


def _audio(vae):
    if isinstance(getattr(vae, "upscale_ratio", None), (tuple, list)):
        return False
    return True if getattr(vae, "audio_sample_rate", None) or getattr(vae, "audio_sample_rate_output", None) else None


def check_vaes(vae, audio_vae):
    if _audio(audio_vae) is False:
        raise RuntimeError("audio_vae is a video VAE: wire the MiniMax-H3 audio VAE into audio_vae.")
    if _audio(vae):
        raise RuntimeError("vae is the audio VAE: the video and audio VAE inputs are swapped.")
    stage = getattr(vae, "first_stage_model", None)
    if stage is not None and hasattr(stage, "decoder") and not hasattr(stage, "encoder"):
        raise RuntimeError("vae has no encoder: every shot after the first starts on the last frame, which needs the full "
                           "MiniMax-H3 video VAE.")


class LowRank(torch.nn.Module):
    weight, comfy_force_forward = None, True

    def __init__(self, pairs):
        super().__init__()
        self.count = len(pairs)
        for i, (down, up) in enumerate(pairs):
            self.register_parameter(f"down{i}", torch.nn.Parameter(down, requires_grad=False))
            self.register_parameter(f"up{i}", torch.nn.Parameter(up, requires_grad=False))

    def forward(self, x):
        cast = lambda w: w if w.device == x.device and w.dtype == x.dtype else w.to(device=x.device, dtype=x.dtype)
        out = [torch.nn.functional.linear(torch.nn.functional.linear(x, cast(getattr(self, f"down{i}"))), cast(getattr(self, f"up{i}")))
               for i in range(self.count)]
        return out[0] if self.count == 1 else torch.cat(out, dim=-1)


def endpoint(end, module, args, out):
    if end.step is None:
        return out
    t = args[0]
    r = t.to(torch.float32)
    for a, b in end.step:
        r = torch.where((t - a).abs() < 1e-5, torch.full_like(r, b), r)
    end.to(t.device)
    return out + end.gate * (end(r).to(out.dtype) - out)


def stepped(end, executor, x, timestep, context, transformer_options, *args, **kwargs):
    sigmas = transformer_options.get("sample_sigmas")
    if sigmas is not None:
        s = sigmas.flatten().float().cpu()
        now = float(timestep.flatten()[0]) / 1000.0
        then = float(s[min(int((s - now).abs().argmin()) + 1, len(s) - 1)])
        video, audio = (float(transformer_options.get(f"minimax_h3_sigma_shift_{k}", d)) for k, d in (("video", 12.0), ("audio", 3.0)))
        heard = lambda v: float(time_shift_sigma(torch.tensor(max(v, 1e-6)), video, audio)) if v > 0 else 0.0
        end.step = ((1.0 - now, 1.0 - then), (1.0 - heard(now), 1.0 - heard(then)))
    try:
        return executor(x, timestep, context, transformer_options, *args, **kwargs)
    finally:
        end.step = None


def mend(model, graph, node_id):
    try:
        net = model.get_model_object("diffusion_model")
        empty = [n for n, m in net.named_modules() if hasattr(m, "in_features") and getattr(m, "weight", 0) is None]
    except Exception:
        return
    if not empty:
        return
    name, path = model_file(graph, node_id)
    keys = meta(path)[0] if path else set()
    stems = {n: [f"{n.rsplit('.', 1)[0]}.{c}_proj" for c in "qkv"] if n.endswith(".qkv_proj") else [n] for n in empty}
    missing = [n for n, ss in stems.items() if not all({f"{x}.down.weight", f"{x}.up.weight"} <= keys for x in ss)]
    if missing or not FLASH_KEYS <= keys:
        raise RuntimeError(f"{os.path.basename(name) or 'The model'}: {len(empty)} layers came out of the loader with no weights "
                           f"({(missing or empty)[0]} and more), and the checkpoint does not hold them in a form the node can fill in; "
                           f"load it with the loader it was made for.")
    from safetensors import safe_open
    with safe_open(path, "pt", device="cpu") as f:
        for n, ss in stems.items():
            comfy.utils.set_attr(net, n, LowRank([(f.get_tensor(f"{x}.down.weight"), f.get_tensor(f"{x}.up.weight")) for x in ss]))
        w = {k: f.get_tensor(f"endpoint_time_embedder.{k}") for k in ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias")}
        end = TimeEmbedder(w["proj_in.weight"].shape[1], w["proj_in.weight"].shape[0], w["proj_out.weight"].shape[0],
                           dtype=torch.float32, device="cpu", operations=torch.nn)
        end.load_state_dict({k: v.float() for k, v in w.items()})
        end.gate = float(f.get_tensor("hyperflow_gate").flatten()[0]) if "hyperflow_gate" in keys else 0.25
    end.step = None
    end.requires_grad_(False)
    net.endpoint_time_embedder = end
    net.time_embedder.register_forward_hook(functools.partial(endpoint, end))
    model.size = 0


def is_fast(model):
    try:
        return getattr(model.model.diffusion_model.blocks[0].attn, "to_gate_compress", None) is not None
    except Exception:
        return False


def takes_pictures(model, graph, node_id):
    if is_fast(model):
        return False, "FastH3 was distilled without reference pictures"
    name, sources = checkpoint(graph, node_id)
    if name and "ref2va" not in f"{name} {sources}".lower():
        return False, f"{os.path.basename(name)} was not trained on reference pictures"
    for n in loras(graph, node_id):
        info, base = meta(full_path("loras", n))[1], os.path.basename(n)
        if info.get("tasks"):
            ok = "ref2va" in str(info["tasks"]).lower()
        else:
            ok = "ref2va" in " ".join([base] + [str(v) for v in info.values()]).lower() or not BLIND.search(" ".join(
                [base] + [str(info.get(k, "")) for k in ("model", "base_model", "source", "resized_from", "source_filename")]))
        if not ok:
            return False, f"{base} was trained without reference pictures"
    return True, ""


def partial(model, graph, node_id):
    try:
        if not getattr(model.get_model_object("diffusion_model"), "use_adaln_curves", False):
            return []
    except Exception:
        return []
    return [os.path.basename(n) for n in loras(graph, node_id)
            if any("time_embedder" in k for k in meta(full_path("loras", n))[0])]


def hyperflow(model, graph):
    try:
        info = model.get_attachment("lora_metadata")
    except Exception:
        info = None
    flagged = lambda m: isinstance(m, dict) and str(m.get("hyperflow", "")).strip().lower() == "true"
    if not flagged(info):
        name = next((n for n in loras(graph) if "hyperflow" in n.lower() or flagged(meta(full_path("loras", n))[1])), "")
        if not name:
            return None
        info = meta(full_path("loras", name))[1]
    def num(key, default):
        try:
            return float(info.get(key) or default)
        except (TypeError, ValueError):
            return default
    try:
        grid = tuple(float(x) for x in json.loads(info.get("hyperflow_sigmas") or "null"))
    except (TypeError, ValueError):
        grid = ()
    ok = len(grid) > 1 and grid[0] == 1.0 and grid[-1] == 0.0 and all(b < a for a, b in zip(grid, grid[1:]))
    return {"grid": grid if ok else HYPERFLOW["sigmas"], "video": num("hyperflow_video_shift", HYPERFLOW["video"]),
            "audio": num("hyperflow_audio_shift", HYPERFLOW["audio"]), "sampler": "euler", "source": "Hyperflow"}


def flash(model, graph, node_id, steps):
    try:
        on = getattr(model.get_model_object("diffusion_model"), "endpoint_time_embedder", None) is not None
    except Exception:
        on = False
    path = "" if on else model_file(graph, node_id)[1]
    if not on and not (path and FLASH_KEYS & meta(path)[0]):
        return None
    n = min(FLASH, key=lambda k: abs(k - int(steps)))
    return {"grid": FLASH[n], "fine": FLASH[8], "video": HYPERFLOW["video"], "audio": HYPERFLOW["audio"], "sampler": "euler",
            "source": "HyperFlow H3 Flash"}


def speed_recipe(graph, node_id):
    for n in loras(graph):
        said = str(meta(full_path("loras", n))[1].get("sampling", ""))
        steps = re.search(r"(\d+)\s*steps?\b", said, re.I)
        shift = lambda kind: re.search(rf"{kind}\s+shift\s*([\d.]+)", said, re.I)
        if steps:
            return {"steps": int(steps.group(1)), "source": os.path.basename(n),
                    "video": float(shift("video").group(1)) if shift("video") else None,
                    "audio": float(shift("audio").group(1)) if shift("audio") else None,
                    "sampler": "lcm" if re.search(r"re-?noise", said, re.I) else "euler"}
        known = next((r for k, r in RECIPES.items() if k in n.lower()), None)
        if known:
            return dict(known, source=os.path.basename(n))
    name, sources = checkpoint(graph, node_id)
    merged = next((r for k, r in RECIPES.items() if k in f"{name} {sources}".lower()), None)
    if merged:
        return dict(merged, source=f"{os.path.basename(name)} (speed model merged in)")
    speed = next((n for n in loras(graph) if SPEED.search(os.path.basename(n))), "") or next(
        (x for x in sources.split() if SPEED.search(os.path.basename(x))), "")
    return {"sampler": "euler", "source": os.path.basename(speed)} if speed else None


def shifted(grid, video):
    return torch.tensor([video * x / (1.0 + (video - 1.0) * x) for x in grid], dtype=torch.float32)


def set_shift(model, video, audio):
    m = model.clone()
    try:
        ms = copy.deepcopy(m.get_model_object("model_sampling"))
        names = inspect.signature(ms.set_parameters).parameters
        kw = {k: float(v) for k, v in (("shift", video), ("audio_shift", audio)) if k in names}
        if kw:
            ms.set_parameters(**kw)
            m.add_object_patch("model_sampling", ms)
    except Exception:
        pass
    to = m.model_options["transformer_options"] = dict(m.model_options.get("transformer_options") or {})
    to["minimax_h3_sigma_shift_video"], to["minimax_h3_sigma_shift_audio"] = float(video), float(audio)
    return m


def finer(model, steps, scheduler, sigmas, video, audio, grid=None):
    base = [float(s) for s in (sigmas if sigmas is not None else
                               comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps))]
    if len(base) - 1 >= FINE_BELOW:
        return None
    x = [s / (video - (video - 1.0) * s) for s in base]
    grid = grid or sorted(set(x) | {(a + b) / 2 for a, b in zip(x, x[1:])}, reverse=True)
    at = next(i for i, g in enumerate(grid) if audio * g / (1.0 + (audio - 1.0) * g) <= REVOICE_FROM)
    return shifted(grid, video), shifted(grid[at:], video)


def landing(sigmas, video, audio, target=0.03, coarse=0.10):
    to_audio = lambda s: audio * (s / (video + s * (1 - video))) / (1 + (audio - 1) * (s / (video + s * (1 - video))))
    if len(sigmas) < 3 or sigmas[-1] != 0.0 or to_audio(sigmas[-2]) <= coarse:
        return sigmas
    base = target / (audio - target * (audio - 1))
    land = base * video / (1 - base + base * video)
    return sigmas[:-1] + [land, 0.0] if 0.0 < land < sigmas[-2] else sigmas


def activations(shape, cond_shapes=None):
    if len(shape) == 3:
        return int(shape[0]) * (int(shape[-1]) // TOKEN_VALUES * TOKEN_BYTES + FIXED_BYTES)
    b, t, h, w = (int(shape[i]) for i in (0, 2, -2, -1))
    return b * (t * (h // 2) * (w // 2) * TOKEN_BYTES + FIXED_BYTES)


def reserve(executor, model, *args, **kwargs):
    base = model.model
    had = base.__dict__.get("memory_required")
    base.memory_required = activations
    try:
        return executor(model, *args, **kwargs)
    finally:
        if had is None:
            del base.memory_required
        else:
            base.memory_required = had


def fast_vsa(model):
    if ((model.model_options.get("transformer_options") or {}).get("patches_replace") or {}).get("dit"):
        return model, "FastH3 VSA left to the attention patch already on the model"
    if "cudamallocasync" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").lower():
        return model, "FastH3 VSA off: restart ComfyUI with --disable-cuda-malloc"
    try:
        from comfy_extras.nodes_sparse_attention import apply_block_sparse_attention, parse_block_list
        return apply_block_sparse_attention(model, tau=1.3, topk_ratio=0.10, vsa=True, start_percent=0.20, end_percent=1.0,
                                            min_tokens=12288, dense_blocks=parse_block_list(""),
                                            sink_conditioning="exact_kv_and_rows", extra_tokens=0, verbose=False), \
            "FastH3 VSA on"
    except Exception as e:
        return model, f"FastH3 VSA off ({type(e).__name__}: {e})"


def prepare(model, steps, sampler, scheduler, sigmas, video, audio, graph, node_id):
    notes = []
    pdd = PDD_KEY in ((getattr(model, "wrappers", None) or {}).get(pe.WrappersMP.DIFFUSION_MODEL) or {})
    if sigmas is not None and not pdd and wired(graph, node_id, "sigmas") == PDD_NODE:
        sigmas = None
        notes.append(f"{PDD_NODE}'s sigmas are wired in but its model is not, so they were left out: wire its model output "
                     f"into this node's model")
    if pdd and sigmas is None:
        raise ValueError(f"{PDD_NODE} is on the model: wire its sigmas output into this node's sigmas input.")
    fast = not pdd and is_fast(model)
    recipe = None
    if pdd:
        video, audio, sampler = 12.0, 3.0, "euler"
        notes.append(f"PDD Acc: its {len(sigmas) - 1}-step sigmas, euler, shift 12/3")
    elif fast:
        video = FAST_SHIFT if float(video) == 12.0 else video
        notes.append(f"FastH3: shift {float(video):g}/{float(audio):g}, {steps} steps")
    elif sigmas is None:
        recipe = hyperflow(model, graph) or flash(model, graph, node_id, steps) or speed_recipe(graph, node_id)
    if recipe:
        video, audio = recipe.get("video") or video, recipe.get("audio") or audio
        sampler = recipe.get("sampler") or sampler
        sigmas = shifted(recipe["grid"], video) if recipe.get("grid") else None
        steps = len(sigmas) - 1 if sigmas is not None else recipe.get("steps") or steps
        notes.append(f"{recipe['source']}: {steps} steps, shift {float(video):g}/{float(audio):g}, {sampler}")
    model = set_shift(model, video, audio)
    model.add_wrapper_with_key(pe.WrappersMP.PREPARE_SAMPLING, "h3_longvideos_activations", reserve)
    end = getattr(model.get_model_object("diffusion_model"), "endpoint_time_embedder", None)
    if hasattr(end, "step"):
        model.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "h3_longvideos_endpoint", functools.partial(stepped, end))
        notes.append("HyperFlow H3 Flash: compressed blocks and two-time conditioning filled in from the checkpoint")
    if fast:
        model, note = fast_vsa(model)
        notes.append(note)
    if sigmas is None and not fast and not recipe:
        base = [float(x) for x in comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)]
        landed = landing(base, float(video), float(audio))
        if len(landed) != len(base):
            sigmas = torch.tensor(landed, dtype=torch.float32)
            notes.append("one audio landing step added")
    fine = finer(model, steps, scheduler, sigmas, float(video), float(audio), (recipe or {}).get("fine"))
    notes += [f"sound passes on {len(fine[0]) - 1} steps, and each spoken line said again over the last {len(fine[1]) - 1} of them, "
              f"so the words come out clear"] if fine else []
    return model, steps, sampler, sigmas, fine, notes
