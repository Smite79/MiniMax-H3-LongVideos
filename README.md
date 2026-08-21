# H3-LongVideos

Make long (up to ~120s) **MiniMax-H3 video + synchronised audio** from a single
prompt, in ComfyUI. Self-contained — it uses only ComfyUI core's H3 support.

H3 renders one shot at a time. This node turns a written scene into a **chain of
shots**: it splits your prompt into beats, sizes each shot to what that beat
actually stages, chains every shot from the previous one's last frame, and keeps
your characters, their clothing and your props consistent from shot to shot —
things that otherwise drift, duplicate or quietly reset at every shot boundary.

One node covers both H3 conditioning tasks: **FL2VA** (a frame anchors the shot)
and **REF2VA** (reference images say what a character looks like).

---

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart the ComfyUI **server**
(not just a browser refresh).

## Quick start

```
UNETLoader ─┐
CLIPLoader ─┼─> H3 Long Videos ─> images ─> Video Combine
VAELoader ──┘                     audio  ─┘
```

You write four things:

**1. The prompt** — the first paragraph is the *anchor* (scene and style, kept on
every shot); each later paragraph is one **beat**, and one beat is one shot.

```
Natural daylight, hard sun and deep shadow. Shallow depth of field, background
falling soft. Fine grain, slight motion blur, neutral colour. A farm with a barn.

Dom drives a van down the driveway and stops in front of the barn.

Dom gets out and walks to the back of it.

Mara steps out of the barn and asks him: "Is that the last one?"
```

**2. `character_memory`** — who is in it and what they wear. This is the only
channel that can change mid-chain:

```
Dom = he, tall, 35, brunette, white t-shirt, blue jeans, work boots
Mara = she, 30, red hair, grey coat, black jeans
```

**3. `resolution`** — presets only, all valid H3 sizes, three tiers per ratio.

**4. `shot_seconds`** — a **ceiling**, not the length of every shot. Leave it at 0
to let the VRAM budget decide.

Set `plan_only` to preview the shot split, lengths and every warning **without
rendering**. Do that first; it is near-instant.

## What it handles for you

- **Beats → shots.** One paragraph, one shot. Nothing can silently collapse them.
- **Pacing.** Each shot is sized from what its beat stages (~2s + ~2.5s per action
  clause, or its spoken line). A 3-second action in a 12-second shot is how a model
  ends up repeating or *reversing* the action.
- **Characters.** Descriptions bind once per shot, at the first mention; repeat
  names collapse to pronouns, because naming someone twice renders them twice.
- **Wardrobe.** Clothing lives in one mutable channel, tracked per person. Removals
  are read from your prose ("takes off her jacket", "steps out of her jeans", "the
  coat falls to the ground") and stated with direction so they don't play in
  reverse. A garment named in a quoted line is an instruction, not an action, so
  asking for something to come off doesn't remove it a shot early. Whatever is
  still on underneath is named, so a removal doesn't read as more than it was.
- **Props.** "the van" in a later shot means the van from the earlier one.
- **Uncovered zones.** The node tracks two body zones, `lower` and `upper`. When a
  removal leaves one with nothing on it, it keeps that state **stated** in every
  later shot until something covers the zone again — because deleting a garment is
  only a silence, and a video model's default is a clothed person, so silence puts
  the clothes back on a shot or two later.

  `exposed_terms` is where you choose the wording, per character. Same syntax as
  the sheet: a pronoun sets it for everyone who declares that pronoun, a name
  overrides one person, and a trailing `upper` targets that zone instead of the
  default `lower`. Anything after the `=` is passed through verbatim, so LoRA
  trigger words ride along:

  ```
  she = <wording for the lower zone>
  he  = <wording for the lower zone>, <lora trigger>
  Mara upper = <wording for Mara's upper zone>
  ```

  Left unset, the node uses its own neutral wording, matched to the character's
  declared pronoun. A key that matches no character and no pronoun is reported in
  `info` rather than silently doing nothing — which is what a mistyped name, or an
  object form like `her` instead of `she`, would otherwise do.

  A character can also **start** with a zone uncovered rather than arriving there
  through a removal — add `nude` (or `naked`, `undressed`, `unclothed`) for both
  zones, `topless` or `bottomless` for one, to their `character_memory`, and the
  wording applies from shot 1. This has to be written explicitly: a sheet that
  simply doesn't list clothes (`Jon = he, 35, bald`) is read as under-specified,
  never as a declaration.

  Configuring any of this **is** the intent, so it overrides `prevent_nudity` — no
  second switch to remember. The shot after a removal also starts **fresh**,
  without the handoff frame, because continuing from a frame that still shows the
  garment is how it comes back: a picture outvotes the sentence.
- **`prevent_nudity`.** **On by default**: the prompt never asserts that a body is
  uncovered. Removals still happen — what is gated is the sentence, and since the
  model's default is a clothed person, it covers what nobody described. `info`
  still reports any zone a removal left uncovered, so you find out either way.
- **Silence.** Beats with no quoted dialogue get a lips-closed clause, a no-voice
  soundscape, and optionally muted audio — H3 babbles otherwise.
- **Overlays.** Optional PIL watermark and intro title, composited after any
  upscale, never asked of the model.

`info` reports what it did and warns before you waste a render — thin beats,
dialogue that will be cut off or padded with invented speech, a removal that leaves
a body zone bare, anchor content that misfires on every shot.

## Reference images (REF2VA)

Connect up to four images to `ref_image_1…4`. By default (`ref_mode: where
tagged`) they land on the shot whose text names them:

```
Dom, <Picture 1>, drives a van down the driveway.
```

Only that shot is reference-conditioned; every other shot keeps its handoff, and a
tagged shot carries the previous frame as an extra reference so a tag is never a
cut. If a reference gets reproduced in the opening frames, lower `ref_noise_aug`
(0.95, then 0.90).

## Speed: Sol-Attn (optional, third-party)

[**ComfyUI-sol-attn**](https://github.com/Saganaki22/ComfyUI-sol-attn) is a
**separate pack** (Apache-2.0, wrapping NVIDIA's Sol-Attn kernel) — not part of
this one. It ships MiniMax-H3-specific sparse attention and is worth having on a
long chain: its own benchmarks put it at 1.38–1.65× over SageAttention on H3
shapes. It chains straight in:

```
UNETLoader ─> MiniMax H3 Memory Efficient Sol Attention Patch ─> H3 Long Videos
```

Nothing here depends on it, and it patches attention while this node only patches
the sampling schedule, so they don't collide.

### Pair it with an SLA LoRA

Sparse attention drops long-range coherence first, and in a video DiT that renders
as **the same person twice**. The fix is an **SLA LoRA** — a turbo LoRA fine-tuned
*with* sparse attention in the loop, so the weights have already adapted to the
approximation. The two are a matched pair:

| | sparse attention ON | OFF |
|---|---|---|
| **SLA LoRA** | the pairing you want | pays the LoRA's quality cost, collects no speedup |
| **ordinary LoRA** | duplicated subjects | normal dense render |

The node detects both halves and warns in `info` when they don't match — including
under `plan_only`, so you find out before spending a render, not after.

Detection reads the **filename** off the workflow graph, because that is the only
place the information exists: an SLA LoRA carries no marker in its tensor names or
its metadata and is byte-shape-identical to any un-resized rank-128 turbo LoRA. Any
LoRA with `sla` as a delimited token in its name counts (`..._768p_sla_...`);
`slack`, `translate` and `SLAYER` do not.

## What the node reads off a LoRA

It reports where a LoRA's declared training disagrees with your settings. It never
overrides a widget — a render has to stay reproducible from what the graph shows.

| Checked | Source |
|---|---|
| Base model is MiniMax-H3 | metadata (`base_model` / `ss_base_model_version`) |
| Step count vs your `steps` | **filename** (`..._4step_...`) |
| Training resolution vs your preset | **filename** (`..._768p_...`) |

Notes say which source they came from, because the two aren't equally trustworthy:
metadata is what the trainer wrote, a filename is a convention anyone can break by
renaming.

**Not available, so not offered.** LoRA files carry no field for a recommended
sampler, scheduler, cfg or shift — no metadata standard defines one — so the node
does not pretend to know them. Trigger words are also unreadable in practice: they
live in `ss_tag_frequency`, which kohya writes and ai-toolkit does not, so a LoRA's
trigger still has to be typed in yourself — into the prompt, the sheet, or
`exposed_terms`, depending on where it needs to land.

**On ComfyUI portable its Triton kernels will not build**, and they fail *silently*
— the patch reports itself inactive and you simply get the slower path. The
embedded Python ships without development files:

```
python_embeded\Include\   contains only greenlet\
python_embeded\libs\      does not exist
```

Fix (verified on **Python 3.13.12**, Triton 3.7.0, CUDA 13.3, SageAttention 2.2.0,
SM120 — sol-attn's own test suite goes 3/7 → **7/7**):

1. Check your version: `python_embeded\python.exe --version`
2. Download the matching CPython NuGet package (it is a zip):
   `https://api.nuget.org/v3-flatcontainer/python/3.13.12/python.3.13.12.nupkg`
3. Copy `tools\include\*` into `python_embeded\Include\`
4. Copy `tools\libs\python313.lib` into `python_embeded\libs\` (create it)

Purely additive. This unblocks Triton generally, not just Sol-Attn. Redo it if a
ComfyUI update replaces `python_embeded`.

Note the sparse paths are **approximate** — A/B a shot before adopting them.

## Requirements

- ComfyUI 0.30+ with native MiniMax-H3 support
- **Pillow** only for the text overlays (ComfyUI already ships it)
- No negative prompt — H3 is CFG-free at `cfg 1`; the node makes an empty one
- No denoise input — fixed at 1.0; partial denoise desyncs the audio schedule

## Full reference

Every field, every warning and the reasoning behind each behaviour:
**[REFERENCE.md](REFERENCE.md)**

## Disclaimer

The owner of this repo will not be responsible for any copyright strikes
incurred because of use. You are responsible for your works. Use this node
responsibly and ethically.
