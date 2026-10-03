---
tags:
  - comfyui
  - comfyui-nodes
  - minimax
  - minimax-h3
  - video
  - text-to-video
  - audio
  - not-for-all-audiences
license: other
license_name: h3-longvideos-no-redistribution
license_link: LICENSE
---

[![Support me on Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/smite79)

**This node is a constant work in progress! If you are noticing bugs or features that do
not work, please ensure that you are pulling the most recent version and updating your
workflows.**

**Please note that RealRebelAI has been blacklisted from this project. If you want further
updates for the node, please continue to use my updates as the node is being continously
worked on. Don't support those who steal other people's work for their own credit.**

# H3-LongVideos

Long **MiniMax-H3 video with synchronised audio** from a single prompt, in ComfyUI.

H3 renders about 15 seconds at a time. This node turns a written scene into a chain of
shots and joins them into one continuous video. Your text reaches the model word for
word.

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart the ComfyUI **server**.

Requires ComfyUI 0.31+ with native MiniMax-H3 support (tested on 0.33).

## What to load

| | |
|---|---|
| **UNET** | a MiniMax-H3 model — a [hybrid fl2va/ref2va merge](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models) works best |
| **CLIP** | H3's text encoder, loader type `minimax` |
| **VAE** | the H3 **video** VAE |
| **audio VAE** | the H3 **audio** VAE — a separate file, and it must be the *converted* one |

```
UNETLoader ─┐                     images ─> Video Combine / Save Video
CLIPLoader ─┼─> H3-LongVideos  ─> audio  ─┘
VAELoader ──┘                     info   ─> Show Text
```

`prompt` is an input socket — wire a multiline text node into it.

**Set `plan_only` first.** It reports the shot split, the lengths and every warning,
without rendering.

## Writing the prompt

**One paragraph = one shot**, separated by a blank line. The first paragraph is the
**scene**, prepended to every shot; the rest are beats.

```
Natural daylight, hard sun. A farm with a barn.

Dom drives a van down the driveway and stops in front of the barn.

Dom gets out and walks to the back of it.

Mara steps out of the barn and asks him: "Is that the last one?"
```

Dialogue goes in **double quotes**. Use the `anchor` widget if you want the framing
carried separately — then every paragraph is a beat.

### A line that must reach the model word for word

Put it on its own line inside the beat:

```
Mara walks Ana to the entrance.
exact: her wrists stay behind her back the whole way
```

It is placed straight after the beat, in your words, and nothing in the node reads,
scopes, scrubs, reorders or drops it. That matters because the node's own continuity
clauses compete for room: on a short beat they can be 70% of what the shot is told
against the beat's 8%, and `info` reports that balance every run. An `exact:` line is
not a guard and has no budget to lose.

Nothing reads it either, on purpose: a name in it puts nobody in the shot, a garment in
it removes nothing, and a door in it stages no change. Write what must be **said**, and
let the beat stage what happens. `exactly:` and `verbatim:` do the same thing.

### Keeping a restraint on: `hold:`

The node reads restraints and gags out of the beat's wording, and a wording it does not
know registers nothing, so no later shot holds it. A `hold:` line under the beat that
puts something on registers it directly:

```
Dan pushes Mara against the wall.
hold: Mara, handcuffs behind her back; duct tape over her mouth

Mara struggles.

remove: duct tape
Dan looks at her.
```

The person comes first, then each piece with where it is, split by `;`. From that beat
on each piece is on that person: the shot is told it goes on, and every later shot that
it stays, whatever the beats say. Its body part and position come from your words
(`cuffs on her ankles` is on the ankles; `behind her back` places the arms). `remove:`
takes a piece off by name and leaves the rest on. With one person on the sheet the name
can be left out. The line itself is read, never sent to the model.

`info` lists what is registered on each shot (`shot 3: Mara -- handcuffs (wrists, behind
the back), duct tape (mouth)`), and warns about any shot whose beat names a restraint or
gag that registered nothing (`shot 2 names 'gag' but nothing was registered -- add a
hold: line if it should stay on`).

### The character sheet

A paragraph of `Name: attributes` lines, or the `character_memory` widget. Each shot is
given the entries for the people its beat names, and nobody else.

```
Nora: <Picture 1>, 34, she, tall, red hair, green canvas jacket, brown boots.
Mike: he, 41, dark hair, navy overalls.
```

- **Declare a pronoun.** It is what lets *"he takes her coat off"* find the right two
  people.
- **Declare an age.** Where a shot describes a bare region, the body named is the age
  the sheet states — without one, an unstated attribute is filled from the model's prior,
  which is a twenty-something whatever you wrote. A declared age under 18 gets **no body
  described for it at all**, and a sheet declaring a minor alongside a script that stages
  nudity or sex refuses to render.
- **One name per person.** `Dan` in some beats and `Mike` in others reads as two.
- **Every name needs an entry**, or the model invents that person differently each shot.

`<Picture N>` means `ref_image_N`, the socket. Tag it onto the person or thing it
depicts and it follows them; untagged, a reference goes on every shot.

### LoRAs

A LoRA is the one input to a shot this node neither writes nor can read out of your
text, so `info` reports what is attached: how many are stacked, over how many weights,
at what strength, whether the **text encoder** carries them too, and the last one's
name if its metadata has one. Two runs whose prompts are identical can render
differently, and nothing else in the run says why.

### Baking a LoRA in, instead of applying it every render

`merge_lora.py` writes a checkpoint with the LoRAs already in the weights:

```
python merge_lora.py BASE.safetensors OUT.safetensors LORA[:strength] [LORA...]
    --dtype bf16|fp16|fp32     what to write (default bf16)
    --plan                     report what would happen and write nothing
```

It takes any number of LoRAs with per-LoRA strengths, reads both key spellings
(`lora_A`/`lora_B` and kohya's `lora_down`/`lora_up`, with or without the
`diffusion_model.` prefix, `alpha` honoured where present), does the arithmetic in
fp32, and **streams** — one tensor in memory at a time, so a 34 GB base does not need
34 GB of RAM.

A quantized base comes out **dequantized**: int8 data and its `weight_scale` become
plain weights, and the `comfy_quant` markers are dropped, because leaving them would
have ComfyUI read bf16 as int8. The output is therefore much larger than the input —
the int8 H3 is 34 GB in and about 62 GB out at bf16. NVFP4 is refused by name rather
than guessed at.

It **measures what it achieved** and says so:

> LoRA fidelity: stored change differs from the intended delta by 0.14x the delta, as bf16

Under 0.1 the LoRA is in the file. At 1.0 or over it is not — the delta was below what
that dtype can resolve, and the merge moved the weights without carrying the LoRA. That
number is the point of running the tool, because a small LoRA cannot always be baked in
at all: measured on the shipped H3 distill LoRAs, deltas run 1e-4 of the weight norm and
some of them need fp32 to survive being merged.

Shape-mismatched layers are skipped and counted, which is the case below.

### A LoRA built for the wrong variant half-loads

H3 ships in variants whose **AdaLN input differs** — 2688 on the full `fl2va`, **8** on
the pruned build and on the hybrid this node recommends — while attention and MLP are
identical between them. A LoRA trained on one and used on the other therefore looks
like it loads.

It does not. ComfyUI applies each LoRA pair as `(B @ A).reshape(weight.shape)` inside a
bare `try/except`: a pair that will not reshape logs one `ERROR` line and the weight is
returned untouched, while every pair that *does* fit is applied. The LoRA ends up **half
on** — and the half that goes missing is `adaln_proj`, the per-block timestep modulation.

A distilled few-step trajectory applied to the attention stack with its modulation
missing is anatomy that does not resolve: a third leg, a limb that starts and stops. It
happens on some LoRAs and not others, which is exactly what makes it hard to attribute.

`info` now reports it — the layer family, how many blocks, and both shapes. If you see

> LoRA PARTLY APPLIED -- 51 x diffusion_model.blocks.N.adaln_proj.linear wanted
> (96768, 2688) for a (96768, 8) weight

then that LoRA is for the full `fl2va` checkpoint, not this one. Use it on the model it
was built for, or on a build of it converted for this one. Nothing about the prompt,
the step count or the schedule will fix it.

It also reads the **step count out of the LoRA's file name** — `4step`, `8step`,
`3step` — because that is the only place a distilled LoRA states one. Its safetensors
metadata carries rank, alpha and conversion provenance and no schedule at all, and
ComfyUI keeps only that metadata, dropping the path, so the name survives nowhere but
the workflow graph. `info` says when your `steps` disagrees with it, and when two
stacked LoRAs disagree with each other. A training checkpoint in a name — `step600` —
is not a step target and is not read as one.

### The step count is the schedule, and the LoRA owns it

`shift_video` 12 / `shift_audio` 3 are H3's own defaults and they are what gets used.
The node does **not** adjust them for your step count.

It briefly did, and that was a mistake worth recording. The reasoning was that at 8
steps shift 12 leaves 0.63 of video noise for the final step to clear in one
evaluation, and that structure cannot resolve inside a jump that big. That is true of
an undistilled model and exactly wrong in front of a turbo LoRA: **a LoRA distilled at
8 steps is trained to cross that jump in one evaluation** — the big final step is not
a defect in the schedule, it is what distillation buys. Solving `shift_video` down to
4.66 took the schedule away from the one the LoRA learned and left it running a
trajectory it had never seen. Reported as third legs and half-rendered body parts, on
some LoRAs and not others, because how far each one sits from its trained schedule
differs.

If a LoRA wants a different shift, set it. The LoRA knows what schedule it was trained
on better than an analysis of the sigma curve does.

`info` still reports when your `steps` disagrees with the step count in a LoRA's file
name, and when two stacked LoRAs disagree with each other — both read, neither acted
on.

Whatever shifts run, the node writes them both into the sampler's schedule **and** into
the stamp the H3 DiT reads its shifts from (as ComfyUI's own `ModelSamplingMiniMaxH3`
does). The DiT derives the audio timesteps and rescales the carried audio from that
stamp, and falls back to 12/3 without one — so a schedule set any other way used to
run the audio for a different schedule than the one sampling.

### FastH3 and Hyperflow

**FastH3** (FastVideo's 8-step distill, detected by its VSA gate layers) runs at
10/3 and with the VSA attention it was trained on: the node applies ComfyUI's Model
Sparse Attention as `vsa`, keep 10%, from 20% of the schedule — unless you already
wired one, or torch is on the `cudaMallocAsync` allocator, where sparse attention
aborts the process (restart with `--disable-cuda-malloc`).

**Hyperflow** (Video Rebirth's 8-step LoRA) is the one LoRA that *does* state its
schedule: its metadata carries the sigma grid and both shifts, and ComfyUI keeps that
metadata on the model. The node samples every shot on that grid, shifted at 12 for
video, audio at 3, on `euler`, with no extra audio step. Leave `sigmas` unwired. If a
LoRA stacked after it hides the metadata, the LoRA's file name is enough.

Hyperflow is trained **two-time**: each step is conditioned on where it starts and
where it lands. The ComfyUI conversion of the LoRA drops the endpoint embedder that
does the second half. The node adds it back from the original release's time-embedder
tensors — a `hyperflow_endpoint_*.safetensors` beside the node, or the original
`minimax_h3_hyperflow_8step_v1.0.safetensors` in `models/loras` — and `info` says
whether it is on. It needs a checkpoint **with a time embedder**: pruned curve-form
builds (an `adaln_t_table` in its place) cannot load Hyperflow's time weights at all,
and `info` says so.

### Pose control for restrained characters

The prompt and the keyframe can both say the wrists are cuffed behind the back, and the
model can still bring the hands out to catch a fall or reach for a door. Pose control
holds the limbs with a skeleton instead of words. On a shot where somebody is
restrained, the shot renders as usual, DWPose reads the people in its frames, the
restrained person's elbows, wrists, knees and ankles are redrawn in the held position
(in that person's own torso frame, so kneeling, falling and lying down still happen),
and the shot renders a second time with the same seed, following that skeleton for its
first steps. Everybody else keeps the motion of the first render.

To wire it:

1. Put the MiniMax H3 Fun ControlNet where **Load Model Patch** can see it, e.g. a
   symlink from `models/controlnet/` into `models/model_patches/`.
2. Add a **Load Model Patch** node, pick the file, and wire it to `pose_controlnet`.
3. Install `comfyui_controlnet_aux` with the two DWPose TorchScript files
   (`yolox_l.torchscript.pt` and `dw-ll_ucoco_384_bs5.torchscript.pt` under its
   `ckpts/hr16/`). The node never downloads them; `info` gives the paths if they are
   missing.

It runs only on the **hybrid b25-49** checkpoint
(`minimax_h3_hybrid_fl2va_ref2va_b25-49`): the controlnet is built for its 8-wide
timestep table. On any other base, with Hyperflow two-time, or with `hold_restraints`
off, the node turns pose control off and `info` says why.

| setting | what it does |
|---|---|
| `pose_strength` | how strongly the second render follows the skeleton; 1.0 default, 0 is off |
| `pose_end` | the share of the steps that follow it, from the first: 0.6 of 8 steps is 5. The rest run free, so hands and cuffs keep their detail |
| `pose_shots` | **repair broken shots** (default): every restrained shot is checked after its first render, and only one whose held limbs came apart, or where a bound body falls, renders again. **every restrained shot**: all of them render twice. **bound falls only**: only the falls |
| `pose_draw` | **everyone**, **everyone, thick lines**, or **bound person only** |

A shot where the restraints go on is held in latch mode, since that is where they
most often break. The limbs that shot puts on are drawn as DWPose saw them until, from
40% of the shot on, they settle into the held position for a few frames running; from
there they are held. Limbs that were already held stay held from the first frame. If
nothing settles and nothing else breaks, the shot keeps its first render. `info` gives
the frame it latched at (`shot 2: pose latched (Mara: arms behind the back) at frame
21`).

Shots where the restraints come off, and bodies fastened to something, keep their first
render. Whenever a shot is unclear -- who is restrained, how many people there are --
it keeps its first render too, because a wrong guess would hold the captor's arms.
`info` has one line per shot (`shot 4: pose repaired (Mara: arms behind the back)`,
`shot 5: pose checked, nothing broken`, `shot 6: pose skipped -- ...`), and `plan_only`
lists which shots will be looked at. Render time doubles only on the shots
that render twice; a shot that passes the check costs one DWPose read.

## Settings

| setting | value |
|---|---|
| `sampler_name` | `res_multistep`, or `euler` with PDD Acc |
| `scheduler` | `simple` |
| `shift_video` / `shift_audio` | **12 / 3**. `shift_video` is lowered automatically when `steps` is too low for it — see below |
| `steps` | 6–8 with a turbo/distill LoRA, 20+ without |
| `megapixels` | 1.0 is H3's native budget; lower is faster and leaner |
| `shot_seconds` | the cap on each shot |
| `shot_length` | `from the beat` sizes each shot from its own line; `fixed` gives every shot `shot_seconds` |
| `ambient_audio` | optional — wire a recording to play under the whole soundtrack. The node no longer builds one: the audio is the model's |
| `ambient_level` | how loud that recording plays. 0 is off; 0.15–0.3 is a bed you notice only when it stops. Does nothing with nothing wired |

Ambience is mixed, never prompted. Scoring a silent shot from text needs the audio
branch left open, and an open branch on a joint model invents a voice for the face to
lip-sync to. The bed is built from the room your scene names, so it needs no file,
and shaped noise cannot speak. It makes tone — air, rumble, hum, water, a clock — so
a scene whose ambience is birdsong gets the room, not the birds; `info` says when.

Everything else has a tooltip explaining what it does and what it costs. Hover before
you change one.

## What the node decides for you

Seven settings used to be widgets. Each had one right answer the node could reach and
you could not, so each was a question whose wrong answer only made the render worse.
They are gone, and `info` says what was chosen whenever it matters.

| was a widget | now |
|---|---|
| `cfg` | pinned to **1.0**. H3 is CFG-free, and every clause this node writes is phrased positively *because* of that — above 1.0 the negative starts being read, the prompting strategy stops being the right one, and the run costs double |
| `apply_model_sampling` | read from the model. ComfyUI's `MiniMaxH3SigmaShift` stamps the shifts it applied into `transformer_options`, so a deliberate upstream patch announces itself — and the node stands down and says so, instead of patching twice |
| `tiled_decode` | measured. The VAE's own `memory_used_decode` against free VRAM, with 25% over it for working allocations. You get a whole-clip decode — and no tile seams — whenever the card has room |
| `upscale_batch` | measured, from free VRAM and the frame size |
| `trim_seam` | pinned on. The first frame of a continued shot is the model's redraw of the keyframe it was handed: a duplicate either way |
| `silence_nonspeech` | pinned on. It was already decided per shot — it fires where there is no quoted line — and the switch only ever turned a correct decision off |
| `cleanup_between_shots` | pinned on. The RAM copy costs a fraction of one shot; VRAM ratcheting across a long chain ends the render |
| `character_guard`, `hold_gaze`, `hold_scene_state`, `mouths_shut_when_no_line`, `hold_camera`, `auto_sound`, `beat_leads` | pinned on. Seven switches, each turning one continuity clause off for the whole run, all seven shipped on. Each answers a reported failure — a face lip-syncing to a line nobody wrote, a drifting camera, a door that shuts itself — and turning one off returns that failure rather than trading it for anything |
| `verbatim` | removed. It sent your text with none of those clauses, to tell the node's doing from the model's |

### What you give up

`verbatim` was the only way to render your text without this node's sentences over it,
and nothing replaces it. `info` still reports what every clause *would* have said, which
is most of what it was read for, but the A/B render is gone.

The guards are also what pays the namings that can draw a duplicate character, and they
are no longer switchable off. A per-shot naming *budget* was tried first — shed the
lowest-ranked clauses on a crowded shot — and it does not work. Nearly every clause here
names its *subject*: "McKenna is lying down", "Only Kate speaks". The ones that mention
somebody only in passing are about the room, the take and the framing, and measured on
real shots those name nobody at all. So a pass restricted to them changes nothing, and a
pass allowed past them costs a fact every time it fires. There is no clause that both
names a person and is safe to drop. `fit_guards` records this.

### A repeated naming is spent as a pronoun

What works instead is the advice this node has always printed: *a pronoun costs nothing.*
The **first** naming in the node's own clause text stands; the repeats become "she",
"his", "him". The fact stays exactly where it was and the naming is not spent twice:

> Kate takes off her scarf and says: "It is warm in here." … **Kate is sitting.** Only
> **she** speaks; every other mouth in the shot stays closed.

Three restrictions, and they are what make it safe:

- **Your words are never touched.** Only clauses this node wrote are rewritten. Your
  scene, your beat and your `exactly:` lines reach the model as you typed them.
- **Nothing is rewritten where the pronoun would not resolve** — two people in the shot
  answering to "she", or a beat staging extras who are not on the sheet to be counted.
  An unresolvable pronoun is the ambiguity the naming existed to prevent.
- **The attribution keeps its name.** "The sobbing is Mara's" stays, because that clause
  exists to say whose vocal it is so nobody else's mouth is opened for it.

Measured on guard-heavy shots this takes 44 namings to 41 and empties the five-naming
bucket. Modest, and it is the whole of what is safely available — the rest of the
namings are in your own text. `info` reports each swap, per shot and per person.

The lever that remains is your beat: "she turns" for a second "Mara turns" takes a
naming off the shot without losing a word of what you asked for.

Every measurement **fails toward what shipped**. A VAE that cannot estimate itself, a
card that will not report free memory, a probe that raises — all of them tile the
decode and take the old chunk size, because tiling something that would have fit costs
seams on one clip and not tiling something that will not fit ends the render.

## Outputs

| slot | what it is |
|---|---|
| `images` | the finished frames |
| `audio` | the synchronised soundtrack |
| `info` | what the node did, and every warning — **read this** |
| `script` | the exact per-shot text it sent |
| `frames_per_shot`, `total_frames`, `shots`, `video_seconds` | for downstream nodes |

When a shot renders something you did not expect, read `script`. It shows precisely what
that shot was told, and `info` says why.

## Other nodes here

- **H3 Shot Length** — seconds to a valid H3 frame count.
- **H3 Overlay** — watermark and intro title composited onto finished frames.
- **H3 Model Inspector** — checkpoint precision, and whether your card runs it natively.

## Notes

- No negative prompt: H3 is CFG-free at `cfg 1`.
- No denoise control: fixed at 1.0, because partial denoise desyncs the joint
  audio/video schedule.
- **Widgets are restored by position.** If they read NaN, right-click the node →
  **Fix node (recreate)**, set your values, and save the workflow again.

## Disclaimer

The owner of this repo will not be responsible for any copyright strikes incurred
because of use. You are responsible for your works. Use this node responsibly and
ethically.
