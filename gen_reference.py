"""Regenerate the field tables in REFERENCE.md from the node's own schema.

REFERENCE.md went stale because it was hand-maintained alongside a 69-field node:
it documented a `total_seconds` input that had been gone for months and was missing
17 fields, including every guard. Hand-copying a schema is a losing game, so the
field-by-field part is generated from INPUT_TYPES and RETURN_NAMES instead. The
prose around it stays hand-written -- reasoning is the part worth writing by hand.

Everything between the AUTOGEN markers is replaced; everything else is untouched.

    python gen_reference.py            # rewrite REFERENCE.md in place
    python gen_reference.py --check    # exit 1 if it is out of date (for CI)

Run it from the node directory with ComfyUI importable (the portable python works:
  ..\..\..\python_embeded\python.exe gen_reference.py
).
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BEGIN = "<!-- AUTOGEN:FIELDS BEGIN -- edit gen_reference.py, not this block -->"
END = "<!-- AUTOGEN:FIELDS END -->"

# Which group each input belongs to, so the tables read as sections rather than as
# one 69-row dump. First match wins; anything unmatched lands in "Other".
GROUPS = [
    ("Wiring", ["model", "clip", "vae", "audio_vae", "first_frame",
                "ref_image_1", "ref_image_2", "ref_image_3", "ref_image_4"]),
    ("The scene", ["prompt", "character_memory", "anchor_override", "beat_split",
                   "per_beat_length"]),
    ("Size and length", ["resolution", "megapixels", "shot_seconds",
                         "allow_oversize_shots", "allow_res_backoff", "fps"]),
    ("Sampling", ["steps", "cfg", "sampler_name", "scheduler", "seed",
                  "vary_seed_per_shot", "apply_model_sampling", "shift_video",
                  "shift_audio"]),
    ("Chaining shots", ["trim_seam", "handoff_offset", "cleanup_between_shots",
                        "vram_headroom_gb"]),
    ("Audio", ["global_soundscape", "non_diegetic_music", "auto_soundscape",
               "auto_silence_nonspeech", "mute_nonspeech_audio", "mute_fade_ms"]),
    ("Consistency guards", ["subject_count_guard", "anatomy_guard", "solidity_guard",
                            "motion_guard", "contact_guard", "lock_restraints",
                            "auto_wardrobe", "auto_props", "prevent_nudity",
                            "exposed_terms"]),
    ("Reference images", ["ref_mode", "ref_image_size", "ref_noise_aug"]),
    ("Decode and upscale", ["decode_tile_frames", "decode_tile_size", "upscale",
                            "upscale_model", "upscale_target_short_edge",
                            "upscale_batch"]),
    ("Overlays", ["watermark_text", "watermark_position", "watermark_size",
                  "watermark_opacity", "watermark_margin", "intro_text",
                  "intro_position", "intro_seconds", "intro_fade", "intro_size",
                  "overlay_font", "overlay_stroke"]),
    ("Preview", ["plan_only"]),
]


def load_node():
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # custom_nodes/
    spec = importlib.util.spec_from_file_location(
        "_h3ref", os.path.join(HERE, "__init__.py"),
        submodule_search_locations=[HERE])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_h3ref"] = mod
    spec.loader.exec_module(mod)
    return mod.NODE_CLASS_MAPPINGS


def describe(spec):
    """(type, constraints) for one input's schema tuple."""
    t, opts = spec[0], (spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {})
    if isinstance(t, list):
        return "choice", ", ".join(f"`{o}`" for o in t)
    bits = []
    if opts.get("forceInput"):
        bits.append("**input socket**")
    if "default" in opts and not opts.get("forceInput"):
        d = opts["default"]
        bits.append(f"default `{d!r}`" if isinstance(d, str) else f"default `{d}`")
    if "min" in opts or "max" in opts:
        bits.append(f"range `{opts.get('min', '-')}`–`{opts.get('max', '-')}`")
    if opts.get("multiline"):
        bits.append("multiline")
    return t if isinstance(t, str) else "?", ", ".join(bits) or "—"


def tooltip_of(spec):
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    tip = " ".join((opts.get("tooltip") or "").split())
    return tip or "_(no tooltip)_"


def build(cls):
    it = cls.INPUT_TYPES()
    req, opt = it.get("required", {}), it.get("optional", {})
    everything = {**req, **opt}
    seen, out = set(), []

    out.append(f"There are **{len(everything)}** inputs and "
               f"**{len(cls.RETURN_NAMES)}** outputs. Required inputs are marked "
               f"**R**; everything else is optional and has a working default.\n")

    for title, names in GROUPS:
        rows = [n for n in names if n in everything]
        if not rows:
            continue
        seen.update(rows)
        out.append(f"### {title}\n")
        out.append("| field | | type | constraints |")
        out.append("|---|---|---|---|")
        for n in rows:
            t, c = describe(everything[n])
            out.append(f"| `{n}` | {'**R**' if n in req else ''} | {t} | {c} |")
        out.append("")
        for n in rows:
            tip = tooltip_of(everything[n])
            if tip != "_(no tooltip)_":
                out.append(f"**`{n}`** — {tip}\n")

    leftover = [n for n in everything if n not in seen]
    if leftover:
        out.append("### Other\n")
        out.append("| field | | type | constraints |")
        out.append("|---|---|---|---|")
        for n in leftover:
            t, c = describe(everything[n])
            out.append(f"| `{n}` | {'**R**' if n in req else ''} | {t} | {c} |")
        out.append("")
        for n in leftover:
            tip = tooltip_of(everything[n])
            if tip != "_(no tooltip)_":
                out.append(f"**`{n}`** — {tip}\n")

    out.append("### Outputs\n")
    out.append("| slot | name | type |")
    out.append("|---|---|---|")
    for i, (n, t) in enumerate(zip(cls.RETURN_NAMES, cls.RETURN_TYPES)):
        out.append(f"| {i} | `{n}` | {t} |")
    out.append("")
    out.append("Outputs are only ever **appended**. A workflow stores an output link "
               "by slot index, so inserting one would silently re-target every wire "
               "after it.\n")
    return "\n".join(out)


def main(check_only=False):
    mapping = load_node()
    body = build(mapping["H3LongVideos"])

    path = os.path.join(HERE, "REFERENCE.md")
    with open(path, encoding="utf-8") as f:
        doc = f.read()
    if BEGIN not in doc or END not in doc:
        sys.exit(f"Markers not found in REFERENCE.md. Add:\n{BEGIN}\n{END}")
    head, rest = doc.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{BEGIN}\n\n{body}\n{END}{tail}"

    if new == doc:
        print("REFERENCE.md is up to date.")
        return
    if check_only:
        sys.exit("REFERENCE.md is OUT OF DATE. Run: python gen_reference.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    print(f"REFERENCE.md field tables regenerated "
          f"({len(mapping['H3LongVideos'].INPUT_TYPES().get('required', {}))} required + "
          f"{len(mapping['H3LongVideos'].INPUT_TYPES().get('optional', {}))} optional).")


if __name__ == "__main__":
    main("--check" in sys.argv)
