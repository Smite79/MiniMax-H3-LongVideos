# H3-LongVideos

Long **MiniMax-H3 video with synchronised audio** from a single prompt, in ComfyUI.

H3 renders about 15 seconds at a time. This node turns a written scene into a
**chain of shots** and joins them into one continuous video.

**Your text is passed through verbatim.** The node does the chaining, not the
writing — it adds nothing to your prompt and rewrites none of it.

---

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart the ComfyUI **server**
(not just a browser refresh).

**Requires ComfyUI 0.31 or newer** with native MiniMax-H3 support. Tested on 0.33.

## What you need loaded

| | |
|---|---|
| **UNET** | a MiniMax-H3 diffusion model |
| **CLIP** | H3's text encoder, loader type `minimax` |
| **VAE** | the H3 **video** VAE |
| **audio VAE** | the H3 **audio** VAE (a separate file, and it must be the *converted* one) |

## Quick start

```
UNETLoader ─┐                     images ─> Video Combine / Save Video
CLIPLoader ─┼─> H3 Long Videos ─> audio  ─┘
VAELoader ──┘                     info   ─> Show Text
```

`prompt` is an **input socket** — wire a multiline text node into it.

Set **`plan_only`** first: it reports the shot split, the lengths and every
warning in seconds, without rendering.

## The prompt

**One paragraph = one shot.** The first paragraph is the **scene**, prepended to
every shot. Everything after it is a beat.

```
Natural daylight, hard sun, shallow depth of field. A farm with a barn.

Dom drives a van down the driveway and stops in front of the barn.

Dom gets out and walks to the back of it.

Mara steps out of the barn and asks him: "Is that the last one?"
```

Three beats, three shots. Each shot is told exactly:

> *Natural daylight, hard sun, shallow depth of field. A farm with a barn. Dom gets
> out and walks to the back of it.*

Nothing more. If a shot should say something about posture, position, clothing or
continuity, write it in the beat — the node will not write it for you, and it will
not argue with what you wrote.

Dialogue goes in **double quotes**, or in H3's own marker `<d>like this</d>`. Either
tells the node which shots have speech; the rest are silenced.

If spoken lines are coming out as **on-screen subtitles**, try `<d>…</d>`. H3's
tokenizer registers dedicated tokens for dialogue (`<d>`, `</d>`), captions
(`<|caption_start|>`) and lyrics — so the model distinguishes speech from text it is
meant to *draw*. Text in plain quotes is not marked as any of them, and a model with
a caption channel is entitled to read it as a caption.

## Shot length

**`shot_length` = "from the beat"** (default) sizes each shot from what its own line
stages — capped by `shot_seconds`, floored at one action's worth. A beat with one
action stops getting a shot with room for two.

That matters because a shot which runs longer than its beat leaves the model seconds
it was told nothing about, and the cheapest way to fill them is to **carry on with
the action**: shears that cut a garment off keep cutting into what is underneath.

```
Dan cuts off her bra and throws it away.               ~7s of content
Dan cuts off her bra and throws it away, then sets     ~12s
the shears down and steps back.
```

So the first gets a ~7s shot and the second a ~10s one. The estimate leans **short**
on purpose: a shot that ends before its action does hands a mid-motion frame to the
next shot, which the chain continues from, while a shot that outlasts its action has
to invent the remainder.

**`shot_length` = "fixed"** gives every shot `shot_seconds` instead. The reason to
want that: uniform lengths mean uniform latent *shapes*, and noise is drawn to the
shape — so one seed gives the whole chain one noise field and surface detail does not
reset at each cut. That consistency is what you trade for pacing, and `info` says so
whenever the lengths differ.

Either way `info` flags any beat its shot still outlasts, by shot number.

## Clothing, and layers

**Describe what is visible.** A scene that lists every layer at once — jacket, shirt,
underwear — tells the model the character is wearing all of them *simultaneously*,
with nothing saying which is hidden. The keyframe holds the first frame, so the shot
starts right; by the last frame only the text is governing, and the under layer starts
showing through the top one.

So list the outer layer, and bring each one in as it becomes visible:

```
A basement. Kate is 20, blonde, wearing a grey jacket.

Dan cuts off her jacket and throws it away.
remove: jacket
add: her white shirt underneath is now visible

Dan cuts off her shirt and throws it away.
remove: shirt
add: her black bra is now visible

Kate looks up at him.
```

Which produces:

```
shot 1  ... Kate is 20, blonde. Her white shirt underneath is now visible.
shot 2  ... Kate is 20, blonde. Her black bra is now visible.
shot 3  ... Kate is 20, blonde. Her black bra is now visible.
```

`remove:` drops any part of the scene naming that item, from that shot onward.
`add:` (or `wear:`) appends your phrase to the scene, in your words, from that shot
onward — and retires automatically when a later `remove:` names it.

Both take effect on their own shot: the keyframe already shows the previous state at
the start, and it is the *description* claiming a garment is still worn that puts it
back. The directive lines never reach the model.

This is the only place the node edits your text, and it does exactly what you tell
it — no removals are inferred from your prose.

## Settings

| setting | value |
|---|---|
| `cfg` | **1.0** — H3 is CFG-free; the negative prompt is never evaluated |
| `sampler_name` | `res_multistep`, or `euler` with PDD Acc |
| `scheduler` | `simple` |
| `shift_video` / `shift_audio` | **12 / 3** — keep them near 4:1 or the audio breaks |
| `steps` | 6–8 with a turbo/distill LoRA, 20+ without |
| `megapixels` | 1.0 is H3's native budget; lower is faster and leaner |
| `shot_seconds` | length of **every** shot (see below) |

**Why every shot is the same length:** noise is drawn to the latent's *shape*, so
shots of different frame counts get unrelated noise from the same seed, and grain
and surface detail reset at every cut. Uniform lengths are what make one seed hold
across a chain.

## Continuity

Each shot after the first starts from the previous shot's last frame, encoded as a
keyframe the way H3 expects one: a keyframe is a *single* pixel frame, which lands
on the 5-frame grid point and encodes to two latent frames. Handing over a slice of
the previous shot's own latent instead looks like a free optimisation and is not —
a causal encoder's last latent is not a standalone opening frame, and using it
degrades every shot after the first.

- **`first_frame`** pins the opening frame of shot 1, the only shot with no
  previous frame. If shot 1 must start in a particular pose or position, this is
  the mechanism — text does not outrank a picture.
- **`ref_image_1…4`** are identity references, applied to every shot unless a
  `<Picture 1>` tag in a beat places them. Keeping them on every shot is
  deliberate: they are the only fixed anchor a long chain has against drift.
- **`ref_noise_aug`** is how *clean* a reference is shown. At the default 0.999 the
  model tends to reproduce the reference — its pose and background included — in
  the opening frames. Lowering it says "approximate".

  **But one aug covers every visual condition row — references *and* the keyframe.**
  Below **0.99** the keyframe latent would be noised and labelled at the wrong
  timestep, which corrupts the anchor of every shot after the first and shows up as
  those shots degrading *during sampling*. So below 0.99 the node stops sending the
  handoff as a keyframe and rides it as an extra reference instead: continuity is
  weaker, but nothing is corrupted. `info` says when this happens. If you want a
  real keyframe, keep `ref_noise_aug` at 0.99 or above.

## Speed

`info` reports where each render actually spent its time:

```
rendered 2673 frames (~111.4s) in 940s -- sampling 380s (40%), decode 505s (54%),
other 55s (6%); per shot 34.5s + 45.9s
```

Use that before changing anything, because the two halves trade against each other.

- **`megapixels`** is the strongest lever and the only one that lowers *both*.
  Attention is quadratic in latent cells, so 1.0 → 0.7 is roughly half the
  attention, and 1.0 → 0.5 about a quarter.
- **`steps`** is linear on the sampling half only.
- **Shot length**: per-shot cost is quadratic, so for a fixed total runtime more
  shorter shots is cheaper — 110s as 15×7s costs about 71% of 11×10s. The price is
  more boundaries to hold together, and shots of different lengths break seed
  consistency, so keep them uniform.
- **`latent_upscale` is not a free win.** It samples small and decodes large, so it
  moves cost from sampling to decode — a 2× latent is 4× the decode, and tiling is
  forced on top. It pays off when sampling dominates (20+ steps). At 6–8 steps with
  a distill LoRA, decode is already the larger half and this makes the render
  *slower*. `info` says so when decode outweighs sampling.
- **`cleanup_between_shots`** and **`tiled_decode`** are insurance against OOM and
  both cost time. Turn them off if you are not near the limit.
- **If `s/it` varies wildly between identical runs**, the working set is not fitting
  in system RAM and everything above is noise until it does.

## Upscaling

Two independent passes, both off by default:

- **`latent_upscale`** (+ `latent_upscale_scale`) runs *between sampling and decode*,
  so the shot is **sampled small and only decoded large**. That is the one that saves
  time — cost scales with latent cells and attention is quadratic in them. Needs the
  *Minimax H3 Latent Upscaler* pack (model and nodes by
  [LBH-123-AI](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)) with its
  weights in `models/latent_upscale_models`. Without the pack it does nothing and
  `info` says so. Tiled decode is forced while it is on, since a 2× latent is roughly
  4× the decode memory.
- **`upscale`** (`rtx` / `model` / `lanczos`, with `upscale_model`,
  `upscale_target_short_edge`, `upscale_batch`) is a post-pass on the finished frames,
  applied once after the shots are joined.

The chain always hands on the **sampled** latent, never the upscaled one — otherwise
the upscaler's reinterpretation feeds the next shot and compounds down the chain.

## Text in the frame

The node composites nothing onto your video — watermarks and title cards live on the
separate **H3 Overlay** node, and only if you wire it in. If text is appearing in the
output, it came from the model, and there are three things worth checking.

**Is H3 Overlay wired in** with `watermark_text` or `intro_text` set? That is the one
thing that puts text there deliberately.

**Does your prompt name text?** H3 draws letterforms when asked. `info` flags words
like *subtitle*, *caption*, *watermark*, *logo*, *credits*, *timestamp* and *title
card* if they appear in a beat.

**Is it a leftover field label?** The previous version of this node printed
`overall_soundscape:` and `non_diegetic_music:` at the bottom of every shot. Paste
one of those old scripts back in as a prompt and — now that your text goes to the
model verbatim — those labels are read as *text to put on the picture*. They are
stripped automatically and `info` says how many.

**What will not work:** putting *"no watermark, no subtitles"* in a negative prompt.
H3 runs at `cfg 1.0`, where the negative is never evaluated. And putting it in the
*positive* names the thing you are trying to avoid, which invites it. If the model
draws a watermark unprompted, the levers are a different checkpoint or LoRA, or a
crop after the fact — not the prompt.

## Audio

H3 is a joint model: the mouth follows the audio branch. A shot with no quoted
line has an unconditioned audio stream, which invents a voice that the picture then
lip-syncs to. `silence_nonspeech` anchors that stream to real encoded silence
instead — conditioning the stream rather than asking the prompt to stop it.

## Outputs

| slot | what it is |
|---|---|
| `images` | the finished frames |
| `audio` | the synchronised soundtrack |
| `info` | what the node did, and every warning — **read this** |
| `script` | the exact per-shot text it sent |
| `frames_per_shot`, `total_frames`, `shots`, `video_seconds` | for downstream nodes |

`script` is the one to check when a shot renders something you did not expect: it
shows precisely what that shot was told.

## Other nodes here

- **H3 Shot Length** — seconds and a valid H3 frame count (17k+5 grid, 362 cap).
- **H3 Overlay** — watermark and intro title composited onto finished frames.
- **H3 Model Inspector** — checkpoint precision, and whether your card runs it
  natively.

## Requirements

- ComfyUI 0.31+ with native MiniMax-H3 support (tested on 0.33)
- No negative prompt: H3 is CFG-free at `cfg 1`
- No denoise control: fixed at 1.0, because partial denoise desyncs the joint
  audio/video schedule

## Disclaimer

The owner of this repo will not be responsible for any copyright strikes incurred
because of use. You are responsible for your works. Use this node responsibly and
ethically.
