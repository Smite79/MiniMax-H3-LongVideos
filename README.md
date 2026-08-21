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
- **Wardrobe.** Clothing lives in one mutable channel. Removals are read from your
  prose ("takes off her jacket", "steps out of her jeans", "the coat falls to the
  ground"), stated with direction so they don't play in reverse, and an under-layer
  is named so a removal doesn't come out as nudity.
- **Props.** "the van" in a later shot means the van from the earlier one.
- **Exposed state.** When a removal empties a body zone, `exposed_terms` says what
  that is called, per character, and the node keeps stating it in every later shot
  until something covers the zone again — no retyping it into each beat. Same
  syntax as the sheet, pronoun for a whole cast, name to override one person, and
  LoRA trigger words ride along:

  ```
  she = visible vagina
  he  = visible penis, mpenis
  Mara upper = bare breasts
  ```

  Requires `prevent_nudity` off.
- **Nudity.** `prevent_nudity` is **on by default**: the prompt never states that a
  body is bare. Removals still happen — what is gated is the sentence, and a video
  model's default is a clothed person, so it covers what nobody described. `info`
  still reports any zone a removal left uncovered. Turn it off only when nudity is
  intended.
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
