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
UNETLoader ─┐                     images ─> Video Combine
CLIPLoader ─┼─> H3 Long Videos ─> audio  ─┘
VAELoader ──┘                     latent ─> (optional) latent post-processing
```

The **`latent`** output carries the sampled latents, joined on the time axis, for
things like a latent upscaler. It is emitted *as well as* `images`, never instead:
the shot chain hands each shot the previous one's decoded last frame, so decoding
cannot be deferred.

It is **not** the latent form of `images` on a multi-shot run. `trim_seam` and
`handoff_offset` cut decoded frames, and H3 compresses time — one pixel frame is
not one latent step — so those cuts have no exact latent equivalent and the seam
frames are still present. On a **single-shot** run nothing trims and it matches
exactly. `info` says which you got.

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

**3. `resolution` + `megapixels`** — the dropdown picks the **shape**, the number
picks the **size**. They are independent: changing aspect ratio does not change
cost. `1.0` = 1024×1024 worth of pixels (ComfyUI's own convention), and at 1.00
every ratio lands on H3's native size. Step down for speed, VRAM and longer shots.

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
- **Restraints stay on.** `lock_restraints` (on by default) keeps handcuffs,
  shackles, manacles, fetters, irons, gags, blindfolds, harnesses and leashes —
  plus qualified forms like `ankle chain` or `leather wrist straps` — from being
  removed by prose. A restraint is a plot state, not a garment. Without this they
  came off by *accident*: "steps out of her jacket and the chain falls away" would
  drop the ankle chain as a side effect of a beat about a jacket, because the
  removal window reaches any tracked item near the cue. To take one off, say so
  directly: `wardrobe: Mara -= handcuffs`. Bare `chain`, `collar`, `strap` and
  `belt` are **not** treated as restraints — they are jewellery, a shirt part, a
  dress part and a garment at least as often. It also states what the restraint
  **does**: a cuffed character otherwise walks with their arms swinging, because
  nothing said the body could not move freely — the restraint present and inert,
  which reads as it having broken. The clause names the bound region positively
  (`the wrists stay bound close together, the arms moving as one`), only for people
  actually in the shot, and it disappears the moment the restraint is removed.
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
- **Shift, automatically.** Load a distill/turbo LoRA and `auto_shift` sets
  `shift_video` / `shift_audio` to match your step count. H3's 12/3 defaults are for
  the ~20 steps it ships for; at 4 steps they put **80% of the denoising into the
  final step**, which renders soft and painterly. The value is derived, not looked
  up — the shift whose worst step carries the same share as 12 does at 20:

  | steps | shift_video | shift_audio | worst step |
  |---|---|---|---|
  | 4 | 1.89 | 0.47 | 38.7% |
  | 6 | 3.15 | 0.79 | 38.7% |
  | 8 | 4.42 | 1.10 | 38.7% |
  | 20 | 12.00 | 3.00 | 38.7% |

  **A model that declares its own shift wins.** A repacked checkpoint whose config
  carries different `sampling_settings`, or an upstream `ModelSamplingMiniMaxH3`
  node, both land on the live model — and either is a deliberate choice. `auto_shift`
  keeps it, passes it through so nothing overwrites it, and reports the conflict:
  *"the model already declares shift_video 8 … a 4-step run would want ~1.89."*
  You settle it by typing a shift, which outranks both.

  `shift_audio` moves *with* it, holding `audio_scale` at 4.0 — flattening that
  ratio breaks the audio branch. **Type either shift by hand and this stops**: a
  value you set is a decision and is never overridden.

- **Anatomy.** `anatomy_guard` states each person's limb *count* — one head, two
  arms, two hands with five fingers, two legs — on shots that have people in them.
  A **negative prompt cannot do this**: H3 is CFG-free at `cfg 1`, so the negative
  is never evaluated and "extra limbs" there does nothing. Naming the number gives
  the model a target; negating one only puts the word in the prompt. Never added to
  the anchor, and never to a shot with nobody in it — describing a body in an empty
  frame is what burned faces into opening frames before. `auto` = on below a 768
  short edge or when a LoRA is applied.
- **Silence, in three layers.** A prompt clause alone was never enough, because
  two of the three causes aren't text.
  1. **Text** — beats with no quoted dialogue get a lips-closed clause and a
     no-voice soundscape.
  2. **Picture** — a dialogue shot handing its *last* frame to a silent shot seeds
     an open mouth mid-word, and a picture outvotes a sentence. The handoff frame is
     taken 3 frames (~125 ms) earlier at exactly that boundary, automatically.
  3. **Audio** — H3 is a *joint* model: the mouth follows the audio branch. On a
     shot with no line that branch is otherwise unconditioned, invents a voice, and
     the picture lip-syncs to it. The keyframe's audio channel is anchored to
     encoded silence instead.

  `mute_nonspeech_audio` is a fourth, weaker thing: it zeroes the waveform *after*
  generation, so it silences the track but cannot close a mouth.
- **Overlays.** Optional PIL watermark and intro title, composited after any
  upscale, never asked of the model.

`info` reports what it did and warns before you waste a render — thin beats,
dialogue that will be cut off or padded with invented speech, a removal that leaves
a body zone bare, anchor content that misfires on every shot.

## Resolution and megapixels

Two widgets, and they do different jobs. **`resolution` picks the shape,
`megapixels` picks the size.**

`megapixels` is a **pixel budget**: `1.0` means 1024×1024 worth of pixels —
1,048,576 — the same convention as ComfyUI's own `Scale Image to Total Pixels`, so
the number means the same thing across your graph. The preset's aspect ratio is
kept and both axes are snapped to a multiple of 32, which is what H3's latent grid
requires. Set `megapixels` to **0** to switch it off and use the preset's own
dimensions verbatim.

### Why a budget instead of a short edge

Cost and training-distribution match are functions of **token count** —
`(h/16) · (w/16) · frames` — which tracks *total pixels*. The short edge does not,
and the two disagree badly at the extremes of aspect ratio:

| preset | short edge | reads as | actual |
|---|---|---|---|
| `1:1 768x768` | 768 | native | **0.56 MP** — 43% under budget |
| `21:9 1536x672` | 672 | sub-native | **0.98 MP** — full budget |

So the square preset that looks native is starved, and the ultra-wide that looks
starved is fine. Judging by short edge gets both backwards. Holding megapixels
constant is what makes two aspect ratios genuinely comparable — VRAM and token
count stay put when you change shape.

### Start at 1.00, then step down

At **1.00MP** every ratio reproduces H3's native dimensions, so it is the natural
starting point. Lower budgets buy speed, VRAM headroom and longer shots — the
shot-length budget is resolution-aware and rescales automatically.

| ratio | 0.44MP | 0.65MP | 1.00MP | 1.20MP |
|---|---|---|---|---|
| `16:9` | 896×512 | 1088×640 | 1344×768 | 1472×832 |
| `9:16` | 512×896 | 640×1088 | 768×1344 | 832×1472 |
| `4:3` | 800×576 | 960×704 | 1184×896 | 1280×960 |
| `3:4` | 576×800 | 704×960 | 896×1184 | 960×1280 |
| `1:1` | 672×672 | 832×832 | 1024×1024 | 1120×1120 |
| `21:9` | 1024×448 | 1248×544 | 1536×672 | 1696×736 |
| `9:21` | 448×1024 | 544×1248 | 672×1536 | 736×1696 |

Those columns are roughly the old `fast` / `balanced` / `native` tiers, which were
only ever three points on this axis. `megapixels` has no off-switch — a bare ratio
has no size to fall back to — and its floor is 0.10.

### The ratio names are approximations

Worth knowing, because it explains why scaling works the way it does:

```
1344 / 768  = 1.750  ->  7:4    NOT 16:9, which is 1.778
1536 / 672  = 2.286  ->  16:7   NOT 21:9, which is 2.333
```

Scaling runs from each ratio's **reference dimensions**, not from the nominal
ratio in its name. That is precisely what makes 1.00MP land exactly on 1344×768 rather than
on 1376×768, which is where a true 16:9 at the same budget would put you.

### What gets reported

`info` prints the size and MP **actually produced**, never what was requested.
Snapping to the 32-grid moves the real area — typically by 1–2%, up to about 4% at
the smallest budgets where a 32px step is a larger fraction of the image — and
echoing your input back would hide what the render used:

```
megapixels 1.00 -> 1024x1024 (1.000MP actual; preset was 768x768 @ 0.562MP)
```

Both `plan_only` and a full render report it, so you can check the size before
spending anything.

**One thing this does not touch: sampling.** H3's shift is a fixed `12.0` in its
model config with no resolution-dependent term — unlike Flux and SD3, there is no
dynamic shift derived from sequence length. Changing `megapixels` changes cost and
detail, not your sigma schedule.

## Reference images (REF2VA)

Connect up to four images to `ref_image_1…4`. By default (`ref_mode: where
tagged`) they land on the shot whose text names them:

```
Dom, <Picture 1>, drives a van down the driveway.
```

Only that shot is reference-conditioned; every other shot keeps its handoff. A
tagged shot **also carries the previous frame as a real keyframe**, so a tag
anchors rather than cuts — the keyframe fixes the opening frame, the references
supply identity.

If a reference gets reproduced in the opening frames, lower `ref_noise_aug` (0.95,
then 0.90). Note the trade: `visual_cond_noise_aug` is a single value covering
*every* conditioning latent, so a softened reference would soften the anchor with
it. Below **0.99** the node therefore drops back to carrying the previous frame as
an extra *reference* instead — weaker for continuity, but it leaves no anchor to
compromise. `info` says which of the two you got.

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

- ComfyUI 0.31+ with native MiniMax-H3 support (tested on 0.33; on 0.30 the audio
  shifts behave differently -- see Requirements in REFERENCE.md)
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
