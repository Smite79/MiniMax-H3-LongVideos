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

## Clothing that comes off

The scene paragraph is stamped on every shot, so a garment described there is still
being described after a beat takes it off — and a description of a worn garment
beats a sentence saying it came off. Say what came off:

```
A basement. Kate is 20, blonde, grey jacket, white shirt, black boots.

Dan cuts off her jacket and throws it away.
remove: jacket

Dan cuts off her shirt and throws it away.
remove: shirt

Kate looks up at him.
```

From that shot onward the item is dropped from the scene:

```
shot 1  ... blonde, white shirt, black boots ...  Dan cuts off her jacket...
shot 2  ... blonde, black boots ...               Dan cuts off her shirt...
shot 3  ... blonde, black boots ...               Kate looks up at him.
```

`remove:` (or `off:`) takes a comma-separated list, and the line itself never
reaches the model. It applies to the removing shot too — the keyframe already shows
the garment on at the start, and it is the *description* saying it is still worn
that puts it back.

This is the only place the node edits your text, and it only ever deletes. It does
not infer removals from your prose; you say what came off.

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
  the opening frames. Lower it (0.95, then 0.90) if a reference is fighting your
  staging.

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
