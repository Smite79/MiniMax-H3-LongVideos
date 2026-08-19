"""
H3 Model Inspector  (detect the base precision / quant format)
==============================================================
Reads the loaded MODEL and reports which precision/quant format the H3 DiT is
stored in: BF16, FP8 (e4m3 / e5m2), INT8 (+ convrot), NVFP4, MXFP8,
ConvRot-W4A4, W4A8, or a mix. Report-only — a manual hint you read and act on.

WHY IT'S FUTURE-PROOF (incl. MXFP8 "once one comes out")
--------------------------------------------------------
It doesn't sniff dtypes and guess. ComfyUI tags every quantized layer at load
with module.quant_format, using fixed strings it already recognizes:
  nvfp4, mxfp8, float8_e4m3fn, float8_e5m2, int8_tensorwise, convrot_w4a4,
  asym_w4a8_int8  (see comfy/ops.py).
This node reads that tag. MXFP8 is already a recognized format in ComfyUI
(comfy/ops.py + comfy/float.py + model_management.supports_mxfp8_compute), so
the day someone ships an MXFP8 H3 checkpoint, this node labels it correctly
with no change. Any brand-new tag lands under "other: <tag>" instead of
crashing, so it degrades gracefully.

It also reports whether YOUR card can run NVFP4 / MXFP8 natively
(model_management.supports_nvfp4_compute / supports_mxfp8_compute).

NOT detected: pruned vs full (that's an architecture axis — factorized AdaLN —
not a quant format). Reported as a caveat, not guessed.

INSTALL: drop into ComfyUI/custom_nodes/, restart.
Node: MiniMax-H3 -> Model Inspector.
"""

# ---- pure helpers (no torch; unit-testable) -------------------------------
_FRIENDLY = {
    "float8_e4m3fn": "FP8 (e4m3)",
    "float8_e5m2": "FP8 (e5m2)",
    "mxfp8": "MXFP8",
    "nvfp4": "NVFP4",
    "int8_tensorwise": "INT8",
    "int8_tensorwise+convrot": "INT8 convrot",
    "convrot_w4a4": "ConvRot W4A4 (int4)",
    "asym_w4a8_int8": "W4A8 (int4/int8)",
    "bf16": "BF16",
    "fp16": "FP16",
}

# implication note per format, tied to a Blackwell 16GB context
_IMPLICATION = {
    "NVFP4": "native on Blackwell (sm_120); half the size of INT8.",
    "MXFP8": "needs Blackwell + torch >= 2.10 for native compute.",
    "FP8 (e4m3)": "fp8 storage; runs on Ada/Blackwell.",
    "FP8 (e5m2)": "fp8 storage; runs on Ada/Blackwell.",
    "INT8": "int8 storage.",
    "INT8 convrot": "int8+ConvRot — needs working sm_120 kernels (absent on some 50-series setups).",
    "ConvRot W4A4 (int4)": "4-bit ConvRot; requires the matching custom nodes/branch.",
    "W4A8 (int4/int8)": "4-bit weight / 8-bit activation.",
    "BF16": "full precision; largest footprint, cleanest LoRA apply.",
    "FP16": "half precision.",
}


def friendly(fmt):
    return _FRIENDLY.get(fmt, f"other: {fmt}")


def summarize(counts):
    """counts: {raw_format: n}. Returns (label, per_format_summary_lines).
    Label = the dominant NON-bf16/fp16 quant format if any (the main blocks),
    else the dominant plain dtype."""
    quant = {k: v for k, v in counts.items() if k not in ("bf16", "fp16")}
    lines = []
    for raw, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {friendly(raw)}: {n} layer(s)")
    if quant:
        top = max(quant.items(), key=lambda kv: kv[1])[0]
        label = friendly(top)
    elif counts:
        top = max(counts.items(), key=lambda kv: kv[1])[0]
        label = friendly(top)
    else:
        label = "unknown"
    return label, lines


# ---- ComfyUI node ---------------------------------------------------------
def _detect(model):
    """Walk the DiT modules, tally quant_format tags (and dtype for the rest).
    Returns (label, counts, report_lines). Imports torch/mm lazily so the pure
    helpers above stay importable without a ComfyUI runtime."""
    import torch
    import comfy.model_management as mm

    # locate the diffusion model inside the ModelPatcher
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    if dm is None:
        dm = getattr(model, "model", None) or model

    def dtype_label(dt):
        return {
            torch.bfloat16: "bf16", torch.float16: "fp16",
            torch.float8_e4m3fn: "float8_e4m3fn", torch.float8_e5m2: "float8_e5m2",
            torch.int8: "int8_tensorwise",
        }.get(dt, str(dt).replace("torch.", ""))

    counts = {}
    if hasattr(dm, "modules"):
        for m in dm.modules():
            fmt = getattr(m, "quant_format", None)
            if fmt is not None:
                # distinguish int8 convrot via the packed weight's params
                if fmt == "int8_tensorwise":
                    params = getattr(getattr(m, "weight", None), "_params", None)
                    if getattr(params, "convrot", False):
                        fmt = "int8_tensorwise+convrot"
                counts[fmt] = counts.get(fmt, 0) + 1
                continue
            w = getattr(m, "weight", None)
            if w is not None and hasattr(w, "dtype"):
                counts[dtype_label(w.dtype)] = counts.get(dtype_label(w.dtype), 0) + 1

    label, lines = summarize(counts)

    # hardware capability for the relevant 4-bit/8-bit formats
    try:
        nv = mm.supports_nvfp4_compute()
    except Exception:
        nv = None
    try:
        mx = mm.supports_mxfp8_compute()
    except Exception:
        mx = None

    report = [f"Detected base precision: {label}"]
    report += lines
    impl = _IMPLICATION.get(label)
    if impl:
        report.append(f"  -> {impl}")
    report.append(f"  card supports NVFP4 compute: {nv};  MXFP8 compute: {mx}")
    if label == "MXFP8" and mx is False:
        report.append("  WARNING: MXFP8 file but this card/torch can't run it natively.")
    if label == "NVFP4" and nv is False:
        report.append("  WARNING: NVFP4 file but this card can't run it natively.")
    report.append("  (pruned-vs-full is a separate architecture axis; not detected here.)")
    return label, counts, "\n".join(report)


class H3ModelInspector:
    CATEGORY = "MiniMax-H3/multishot"
    FUNCTION = "inspect"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("format", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    def inspect(self, model):
        label, _counts, report = _detect(model)
        return (label, report)


NODE_CLASS_MAPPINGS = {"H3ModelInspector": H3ModelInspector}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ModelInspector": "H3 Model Inspector"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


if __name__ == "__main__":
    # exercise the pure aggregation logic with mocked layer tallies
    cases = {
        "NVFP4 file (200 main + bf16 rest)": {"nvfp4": 200, "bf16": 132},
        "INT8 convrot": {"int8_tensorwise+convrot": 170, "bf16": 30},
        "plain bf16": {"bf16": 340},
        "FP8": {"float8_e4m3fn": 200, "bf16": 140},
        "MXFP8 (future file)": {"mxfp8": 200, "bf16": 132},
        "some unknown new tag": {"fp6_e3m2": 200, "bf16": 132},
    }
    for name, counts in cases.items():
        label, lines = summarize(counts)
        print(f"{name:38s} -> {label}")
