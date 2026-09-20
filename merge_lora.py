#!/usr/bin/env python3
# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""Bake LoRAs into a checkpoint at full precision, once, offline.

    python merge_lora.py BASE.safetensors OUT.safetensors LORA[:strength] [LORA...]

    --dtype bf16|fp16|fp32   what to write   (default bf16)
    --plan                   report what would happen and write nothing
    --device cuda            do the matmuls on the GPU (default cpu)

WHY BAKE ANYTHING. A LoRA applied at load time is applied to every render, through
whatever path ComfyUI picks for that checkpoint on that card -- merged into the stored
weight on one, handed to the layer as a function on another, and the two do not have
the same arithmetic. Baking settles it: the weights on disk are the weights that run,
the LoRA is in them exactly once, at fp32, and a render that still goes wrong has one
fewer thing it could be.

WHAT THIS DOES NOT DO is make a small LoRA survive a small dtype. A delta measured at
8e-5 of the weight norm is below bf16's own resolution -- storing W+d in bf16 loses
most of d, and no merge tool can change that. So the outcome is MEASURED and reported:
how far the change actually stored is from the delta that was asked for, in units of
that delta. Under 0.1 the LoRA is in the file; at 1.0 or over it is not, and the merge
has done nothing but move the weights. Reading that number is the point of running
this.

STREAMS. One tensor is in memory at a time, so a 34 GB base needs about as much RAM as
its largest layer. The safetensors header is written first with the offsets computed up
front, then each tensor's bytes are appended in header order.

DEQUANTIZES, AND SAYS SO. A quantized base is written out dequantized -- int8 data and
its weight_scale become plain weights in `--dtype`, and the comfy_quant markers are
DROPPED, because leaving them would have ComfyUI read bf16 as int8. The output is a
bigger file than the input and an ordinary unquantized checkpoint.

Formats it will not touch are refused by name rather than guessed at: NVFP4 (its block
scales and second-level scale are not reconstructed here) and anything else carrying a
comfy_quant it does not recognise. A wrong dequantisation is silent and ruins a render;
a refusal costs a message.
"""

import argparse
import json
import os
import re
import struct
import sys

import torch

# safetensors dtype tag -> (torch dtype, bytes per element)
_DT = {"F64": (torch.float64, 8), "F32": (torch.float32, 4), "F16": (torch.float16, 2),
       "BF16": (torch.bfloat16, 2), "I64": (torch.int64, 8), "I32": (torch.int32, 4),
       "I16": (torch.int16, 2), "I8": (torch.int8, 1), "U8": (torch.uint8, 1),
       "BOOL": (torch.bool, 1), "F8_E4M3": (torch.float8_e4m3fn, 1),
       "F8_E5M2": (torch.float8_e5m2, 1)}
_TAG = {v[0]: k for k, v in _DT.items()}
_OUT_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16,
              "fp32": torch.float32, "float32": torch.float32}

# Quantisation this script can undo. int8 with a per-output-channel (or per-tensor)
# scale is a multiply; convrot does not change that, because the rotation lives in the
# int8 KERNEL and not in the stored weight -- comfy passes it as an argument to
# int8_linear and dequantize() hands back the plain matrix either way.
_CAN_DEQUANT = ("int8_tensorwise", "int8_perchannel", "int8")


def _header(path):
    """(header dict, byte offset where the data starts)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class Reader:
    """One tensor at a time, by offset, without mapping the whole file."""

    def __init__(self, path):
        self.path = path
        self.header, self.start = _header(path)
        self.fh = open(path, "rb")

    def keys(self):
        return [k for k in self.header if k != "__metadata__"]

    def meta(self):
        return self.header.get("__metadata__") or {}

    def info(self, key):
        e = self.header[key]
        return e["dtype"], tuple(e["shape"])

    def get(self, key):
        e = self.header[key]
        dtype, size = _DT[e["dtype"]]
        a, b = e["data_offsets"]
        self.fh.seek(self.start + a)
        raw = self.fh.read(b - a)
        t = torch.frombuffer(bytearray(raw), dtype=dtype)
        return t.reshape(tuple(e["shape"])) if e["shape"] else t

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


def lora_pairs(reader):
    """{target key without the diffusion_model prefix: (A, B, alpha key or None)}.

    Covers the spellings these files actually come in: lora_A/lora_B from a PEFT
    export and lora_down/lora_up from kohya, with or without a leading
    `diffusion_model.`, and an optional `.alpha` beside them. The target is stored
    unprefixed so a LoRA that names the prefix and one that does not both land on the
    same weight -- taomate ships without it and every other H3 LoRA here ships with it.
    """
    out = {}
    for k in reader.keys():
        m = re.match(r"^(.*?)\.(lora_A|lora_down)\.weight$", k)
        if not m:
            continue
        base, kind = m.group(1), m.group(2)
        up = f"{base}.{'lora_B' if kind == 'lora_A' else 'lora_up'}.weight"
        if up not in reader.header:
            continue
        target = base[len("diffusion_model."):] if base.startswith("diffusion_model.") else base
        alpha = f"{base}.alpha" if f"{base}.alpha" in reader.header else None
        out[target] = (k, up, alpha)
    return out


def delta_for(reader, entry, strength):
    """The weight delta this LoRA contributes, in fp32, or None if it cannot be read."""
    a_key, b_key, alpha_key = entry
    A = reader.get(a_key).to(torch.float32)
    B = reader.get(b_key).to(torch.float32)
    scale = float(strength)
    if alpha_key is not None:
        try:
            scale *= float(reader.get(alpha_key).reshape(-1)[0]) / A.shape[0]
        except Exception:
            pass
    if A.ndim > 2:
        A = A.flatten(1)
    if B.ndim > 2:
        B = B.flatten(1)
    if A.ndim != 2 or B.ndim != 2 or B.shape[1] != A.shape[0]:
        return None
    return (B @ A) * scale


def quant_format(header, key):
    """The comfy_quant format for a weight, or "" when it is stored plain."""
    conf = header.get(key.rsplit(".weight", 1)[0] + ".comfy_quant")
    if conf is None:
        return ""
    return "int8"          # the marker's own JSON needs the file; the scale tells us more


def plan(base, loras, out_dtype):
    """(entries to write, per-LoRA report, refusals). Nothing is read twice later."""
    hdr = base.header
    plain = []              # (out_key, source_key, kind, extra)
    skipped_keys = set()    # sidecars that must NOT be written once dequantized
    refused = []
    applied = {name: [0, 0] for name, _, _ in loras}     # name -> [applied, shape-skipped]
    targets = {}            # out_key -> [(lora index, entry)]

    for k in base.keys():
        if k.endswith(".comfy_quant") or k.endswith(".weight_scale") or k.endswith(".weight_scale_2"):
            skipped_keys.add(k)

    for k in base.keys():
        if k in skipped_keys:
            continue
        dt, shape = base.info(k)
        quantized = (k.rsplit(".weight", 1)[0] + ".weight_scale") in hdr if k.endswith(".weight") else False
        if quantized:
            if (k.rsplit(".weight", 1)[0] + ".weight_scale_2") in hdr or dt == "F8_E4M3":
                refused.append(k)
                continue
            if dt not in ("I8",):
                refused.append(k)
                continue
        want = k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k
        base_name = want[:-len(".weight")] if want.endswith(".weight") else None
        hits = []
        if base_name is not None:
            for i, (name, reader, strength) in enumerate(loras):
                entry = reader.pairs.get(base_name)
                if entry is not None:
                    hits.append((i, entry))
        if hits:
            targets[k] = hits
        plain.append((k, dt, shape, quantized))
    return plain, targets, applied, refused


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Bake LoRAs into a checkpoint at full precision.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("lora", nargs="+", help="path, or path:strength (default 1.0)")
    ap.add_argument("--dtype", default="bf16", choices=sorted(_OUT_DTYPE))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plan", action="store_true", help="report and write nothing")
    args = ap.parse_args(argv)

    out_dtype = _OUT_DTYPE[args.dtype]
    base = Reader(args.base)
    loras = []
    for spec in args.lora:
        path, _, s = spec.rpartition(":")
        if not path or not os.path.exists(path):
            path, s = spec, "1.0"
        try:
            strength = float(s)
        except ValueError:
            path, strength = spec, 1.0
        if not os.path.exists(path):
            print(f"no such LoRA: {path}", file=sys.stderr)
            return 1
        r = Reader(path)
        r.pairs = lora_pairs(r)
        loras.append((os.path.basename(path), r, strength))

    entries, targets, applied, refused = plan(base, loras, out_dtype)

    print(f"base   : {os.path.basename(args.base)}  ({len(base.keys())} tensors)")
    for name, r, strength in loras:
        print(f"lora   : {name}  strength {strength:g}  ({len(r.pairs)} adapter pairs)")
    print(f"output : {args.out}  as {args.dtype}")
    if refused:
        print(f"\nREFUSED -- {len(refused)} weights are in a quantisation this script does "
              f"not reconstruct (NVFP4 or unknown). Nothing was written. Merge against a "
              f"checkpoint stored plain or int8.\n  e.g. {refused[0]}", file=sys.stderr)
        return 2

    # The header is written first, so every shape and offset is decided before any
    # tensor is computed. Sizes are known without touching the data.
    header, cursor = {}, 0
    for k, dt, shape, quantized in entries:
        write_dt = out_dtype if (quantized or k in targets) else _DT[dt][0]
        if k in targets and _DT[dt][0].is_floating_point:
            write_dt = out_dtype
        n = 1
        for d in shape:
            n *= d
        size = n * torch.finfo(write_dt).bits // 8 if write_dt.is_floating_point else n * _DT[dt][1]
        header[k] = {"dtype": _TAG[write_dt] if write_dt.is_floating_point else dt,
                     "shape": list(shape), "data_offsets": [cursor, cursor + size]}
        cursor += size
    meta = dict(base.meta())
    meta["merged_loras"] = "; ".join(f"{n}:{s:g}" for n, _, s in loras)
    meta["merged_dtype"] = args.dtype
    header["__metadata__"] = meta

    if args.plan:
        touched = len(targets)
        print(f"\nwould write {len(entries)} tensors, {touched} of them carrying a LoRA")
        print(f"estimated size: {cursor / 2**30:.1f} GiB")
        return 0

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(blob)) % 8
    blob += b" " * pad
    fid = []                     # per-layer fidelity, for the report at the end
    with open(args.out, "wb") as fo:
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        for k, dt, shape, quantized in entries:
            t = base.get(k)
            if quantized:
                scale = base.get(k.rsplit(".weight", 1)[0] + ".weight_scale").to(torch.float32)
                t = t.to(torch.float32) * scale.reshape(scale.shape[0], *([1] * (t.ndim - 1)))
            hits = targets.get(k)
            if hits:
                W = t.to(torch.float32).to(args.device)
                total = torch.zeros_like(W)
                for i, entry in hits:
                    name, reader, strength = loras[i]
                    d = delta_for(reader, entry, strength)
                    if d is None or d.numel() != W.numel():
                        applied[name][1] += 1
                        continue
                    total += d.reshape(W.shape).to(W.device)
                    applied[name][0] += 1
                if total.abs().sum() > 0:
                    merged = W + total
                    # MEASURED AGAINST THE SAME DTYPE, not against W in fp32. Writing
                    # W alone as bf16 already moves it, and charging that to the LoRA
                    # made a tiny delta report 8000000% -- the number was the dtype's
                    # error on W, which the LoRA had nothing to do with. What is asked
                    # here is only: of the change the LoRA asked for, how much of it is
                    # in the file. Anything at or under 1.0 means the stored change is
                    # further from the delta than zero would have been.
                    applied_d = (merged.to(out_dtype).to(torch.float32)
                                 - W.to(out_dtype).to(torch.float32))
                    fid.append(((applied_d - total).norm()
                                / total.norm().clamp(min=1e-20)).item())
                    t = merged.to("cpu")
                else:
                    t = W.to("cpu")
            want = header[k]["dtype"]
            t = t.to(_DT[want][0])
            fo.write(t.contiguous().view(torch.uint8).numpy().tobytes()
                     if t.dtype not in (torch.uint8,) else t.contiguous().numpy().tobytes())
    base.close()
    for _, r, _ in loras:
        r.close()

    print()
    for name, (ok, bad) in applied.items():
        line = f"{name}: {ok} layers merged"
        if bad:
            line += f", {bad} SKIPPED for shape -- built for a different variant of this model"
        print(line)
    if fid:
        err = sum(fid) / len(fid)
        print(f"\nLoRA fidelity: stored change differs from the intended delta by "
              f"{err:.3f}x the delta, as {args.dtype}")
        wider = {"bf16": "fp16, then fp32", "fp16": "fp32", "fp32": None}[args.dtype]
        if err < 0.1:
            print("  The LoRA is in the file.")
        elif err < 0.9:
            print("  Most of it is there, and some was rounded away: this delta is near "
                  f"what {args.dtype} can resolve."
                  + (f" --dtype {wider} would hold more of it." if wider else
                     " fp32 is already the widest this writes -- the delta is simply that "
                     "small next to the weights."))
        else:
            print(f"  IT DID NOT FIT. The delta is below {args.dtype}'s resolution, so what "
                  f"is stored is further from the LoRA than leaving it out would have been."
                  + (f" Re-run with --dtype {wider}." if wider else
                     " Even fp32 cannot hold it: this LoRA is too small relative to these "
                     "weights to survive being merged at all, and belongs applied at "
                     "runtime rather than baked."))
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out) / 2**30:.1f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
