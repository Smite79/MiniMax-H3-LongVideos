# H3-LongVideos

Long **MiniMax-H3 video with synchronised audio** from a single prompt, in ComfyUI.

H3 renders about 15 seconds at a time. This node turns a written scene into a
**chain of shots**: it splits your prompt into beats, sizes each shot, chains every
shot from the previous one's last frame, and keeps your characters, their clothing
and your props consistent from shot to shot — the things that otherwise drift or
reset at every shot boundary.

One node covers both H3 conditioning tasks: **FL2VA** (a frame anchors the shot) and
**REF2VA** (reference images say what a character looks like).

---

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart the ComfyUI **server**
(not just a browser refresh). No extra Python packages are needed — it uses only
ComfyUI core's H3 support, plus Pillow (already shipped) for the text overlays.

**Requires ComfyUI 0.31 or newer** with native MiniMax-H3 support. Tested on 0.33.

**Upgrading from an earlier version:** the node's type is now `H3LongVideosV2`, so a
workflow saved against the old node shows it as missing. Delete it and add a fresh
**H3 Long Videos**. This is deliberate — ComfyUI restores widgets by position, and
the widget list changed, so an old node would silently load every setting into the
wrong widget.

## What you need loaded

| | |
|---|---|
| **UNET** | a MiniMax-H3 diffusion model |
| **CLIP** | H3's text encoder, loader type `minimax` |
| **VAE** | the H3 **video** VAE |
| **audio VAE** | the H3 **audio** VAE (a separate file) |

## Quick start

```
UNETLoader ─┐                     images ─> (H3 Overlay) ─> Video Combine / Save Video
CLIPLoader ─┼─> H3 Long Videos ─> audio  ─────────────────┘
VAELoader ──┘                     info   ─> Show Text
```

**H3 Overlay** is optional: wire it in only when you want a watermark or an intro
title composited onto the frames. Put it after any upscale.

The text fields are **input sockets, not boxes on the node** — wire a multiline text
node into them. `prompt` is required; leave it unconnected and the graph errors
rather than rendering blank.

Set **`plan_only`** first. It previews the shot split, the lengths and every warning
in seconds, without rendering anything.

## The prompt

**One paragraph = one shot.** The first paragraph is the *anchor* — scene and style,
repeated on every shot. Each paragraph after it is one **beat**.

```
Natural daylight, hard sun and deep shadow. Shallow depth of field. A farm
with a barn.

Dom drives a van down the driveway and stops in front of the barn.

Dom gets out and walks to the back of it.

Mara steps out of the barn and asks him: "Is that the last one?"
```

That is 3 beats, so 3 shots. Dialogue goes in **double quotes** — that is how the
node knows which shots have speech.

## `character_memory`

Who is in it and what they wear. This is the channel that stays consistent across
the whole chain:

```
Dom = he, tall, 35, brunette, white t-shirt, blue jeans, work boots
Mara = she, 30, red hair, grey coat, black jeans
```

Declaring a pronoun matters: it is how *"she takes off her coat"* is attributed to
the right person when two people are on screen.

## Size and length

**`resolution` picks the shape, `megapixels` picks the size.** They are independent
— changing aspect ratio does not change cost.

- `megapixels 1.0` = 1024×1024 worth of pixels, which is H3's native budget.
- Step down for speed, VRAM and longer shots. `0.7` is a good working value on a
  16 GB card.

**`shot_seconds`** is a ceiling, not the length of every shot. Wire the **H3 Shot
Length** node into it (it also reports the matching frame count), or leave it
unconnected and the VRAM budget decides.

**`min_shot_seconds`** (default 10) is the floor. It exists for seed consistency:
noise is drawn to the latent's *shape*, so shots of different frame counts get
unrelated noise from the same seed and the grain resets at every cut. A floor near
the ceiling makes the lengths uniform, which is what makes one seed hold across a
chain. The cost is pacing — a one-action beat in a 10s shot leaves the model time it
was told nothing about, and it fills that by repeating or reversing the action, so
`info` flags those beats as THIN. Give a thin beat a second action, or its own
`seconds:` line, which always wins. Set it to 0 for pure content sizing.

Cost scales with **latent cells** — resolution *and* duration — and attention is
quadratic in them. Lowering megapixels does far more for speed than any other
setting.

## Sampling

| setting | value |
|---|---|
| `cfg` | **1.0** — H3 is CFG-free; there is no negative prompt |
| `sampler_name` | `res_multistep`, or `euler` with PDD Acc |
| `scheduler` | `simple` |
| `shift_video` / `shift_audio` | **12 / 3** |

Keep the two shifts about 4:1 apart. H3 carries the audio latent on the video
schedule scaled by that ratio, so flattening it toward 1:1 breaks the audio.

`steps` depends on your LoRA: 4-step turbo LoRAs work at 4, but 6–8 looks
noticeably better, and 8 is the top of the useful range.

## Outputs

| slot | what it is |
|---|---|
| `images` | the finished frames |
| `audio` | the synchronised soundtrack |
| `info` | what the node did, and every warning — **read this** |
| `script` | the exact per-shot text it built |
| `soundscape` | the ambient bed actually used |
| `latent` | the sampled latents, for latent post-processing |
| `frames_per_shot`, `total_frames`, `shots`, `video_seconds`, `fps`, `fps_int` | numbers for downstream nodes |

`info` is the one to wire to a Show Text node. Nearly every problem the node can
detect is reported there.

## What it handles for you

- **Beats → shots.** One paragraph, one shot. Nothing silently merges them.
- **Pacing.** Each shot is sized from what its beat actually stages. A 3-second
  action in a 12-second shot is how a model ends up repeating or reversing it.
- **Characters.** Descriptions bind once per shot, at the first mention. Repeat
  names collapse to pronouns, because naming someone twice can render them twice.
- **Clothing.** Tracked per person. Removals are read from your prose (*"takes off
  her jacket"*) and stated with direction so they don't play in reverse.
- **Props.** *"the van"* in a later shot means the van from the earlier one.
- **Continuity guards.** Limb counts, solid objects, continuous motion and
  two-body arrangements are each stated positively when the shot needs them. They
  have to be positive: at `cfg 1` the negative prompt is never evaluated, and a
  negation in the positive names the thing it forbids.
- **Audio.** Shots without dialogue are silenced so the model doesn't invent
  speech; the ambient bed is matched in level across shots and carried between them
  so the room doesn't change at every cut.
- **Seams.** Each shot continues from the previous one's last frame, and the
  duplicate frame at the join is trimmed.
- **Guards.** One `guards` setting (off / auto / on) covers limb count, subject
  count, solid objects, continuous motion and two-body contact. Leave it on `auto`.
- **Overlays.** Watermark and intro title live on the separate **H3 Overlay** node.

## Reference images

Connect up to four images to `ref_image_1…4`. By default they land on the shot whose
text names them:

```
Dom, <Picture 1>, drives a van down the driveway.
```

Every reference-conditioned shot **also** carries the previous frame as a keyframe,
so using references never costs you continuity.

## Optional third-party packs

None of these are required, and the node works without them.

- **Latent upscale** — `latent_upscale` drives the **MiniMax-H3 Latent Upscaler by
  [LBH-123-AI](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)**.
  Weights go in `models/latent_upscale_models`; the nodes that run them are the
  `Comfyui_Minimax_h3_latent_Upscaler` pack. All credit for the model and those
  nodes goes to LBH-123-AI. It upscales between sampling and decode, so the shot is
  *sampled* small and only *decoded* large — much cheaper than sampling large.
  Without the pack installed the setting does nothing and `info` says so.
- **PDD Acc LoRAs** — [alibaba-pai/MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)
  via the `ComfyUI-MiniMax-H3-PDD-Acc` pack. Files go in `models/pdd_acc/`, not
  `models/loras/` — a plain LoRA loader cannot run them. Connect the Apply node's
  `sigmas` output to this node's `sigmas` input and keep `sampler_name` on `euler`.
- **Sparse attention** — the `H3 SLA Attention` node, paired with an SLA turbo LoRA.
  Place it on the MODEL wire **after** your LoRA loaders and last before this node,
  or ComfyUI prunes it and it never runs.

## If something looks wrong

| symptom | first thing to check |
|---|---|
| out of VRAM while sampling | lower `megapixels`, then `shot_seconds`. Tiled decode cannot help a sampling OOM |
| very slow, or wildly varying `s/it` | total staged model size against your system RAM — paging weights dominates everything else |
| characters drift between shots | `vary_seed_per_shot` off, and check `info` for shots that lost their handoff |
| a shot cuts instead of continuing | `info` names shots that dropped the keyframe, and why |
| speech where there should be none | shots without a quoted line are silenced automatically; `info` reports which |
| a LoRA errors on every block | the LoRA does not match your checkpoint's shapes — see `info` |

`plan_only` costs seconds and answers most of these before a render.

## Requirements

- ComfyUI 0.31+ with native MiniMax-H3 support (tested on 0.33)
- Pillow, for the text overlays only — ComfyUI already ships it
- No negative prompt: H3 is CFG-free at `cfg 1`
- No denoise input: fixed at 1.0, because partial denoise desyncs the audio schedule

## Credits

The **MiniMax-H3 Latent Upscaler** behind the `latent_upscale` setting is the work of
**[LBH-123-AI](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)**, and
is distributed with its own ComfyUI nodes. All credit for the model and those nodes
goes there; this node only calls them, and works without them.

## Disclaimer

The owner of this repo will not be responsible for any copyright strikes
incurred because of use. You are responsible for your works. Use this node
responsibly and ethically.
