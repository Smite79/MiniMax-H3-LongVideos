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

Long MiniMax-H3 video with synchronised audio from a single prompt, in ComfyUI.

## What it does

H3 renders at most about 15 seconds at a time. H3-LongVideos takes a whole scene written as beats, renders it shot by shot
and joins the shots into one video with one soundtrack. People, clothes, poses and objects stay the same from shot to shot,
and every spoken line is lip-synced to whoever says it.

## How it works

1. **Plan.** Each paragraph of the prompt is a beat, and each beat becomes a shot as long as it takes in real life (at least
   5.2 seconds). A beat that needs more than `shot_seconds` runs on into the next shot.
2. **Write.** Each shot's text is the scene, the characters and the beat, plus notes that carry over what changed in the
   shots before.
3. **Render.** Each shot starts on the last frame of the shot before. Reference pictures go into every shot their person is in.
4. **Sound.** Each spoken line is lip-synced to its speaker. Shots or stretches with no spoken line get their sound from a
   separate audio-only pass.
5. **Join.** Each shot is matched to the one before in colour, contrast and sharpness, and the still background is carried
   over, so nothing drifts. The shots come out as one video with one soundtrack.

## The prompt

Characters go in `character_memory`, one per line. `<Picture 1>` … `<Picture 4>` beside a name gives that person the picture
wired into `ref_image_1` … `ref_image_4`:

```
Sarah <Picture 1>: a woman in her twenties, long brown hair, white blouse, black skirt
Dana: a woman in her thirties, short red hair, black blazer, blue shirt, grey trousers
```

The prompt is the scene, then one paragraph per beat. Anything in quotes is spoken:

```
A dim office at night.

Dana walks in and stops at Sarah's desk. "You're still here?" she asks.

Sarah looks up and smiles. "I was waiting for you."

Dana takes off her blazer and hangs it on the back of a chair.

Dana walks out of the room.
exit: Dana
```

- A first paragraph that names none of your characters and has no quotes is the scene, kept at the top of every shot. Text
  in `anchor` is kept there too.
- `cut` on its own line starts the next shot fresh instead of on the last frame.
- A `seconds: 12` line in a beat sets that shot's length.
- A tag line under a beat says outright what happens, instead of leaving it to the node to read:

| tag | |
|---|---|
| `hold: Name, item` | the item goes on in this shot and stays on |
| `release: Name, item` | the item comes off; `release: Name` takes everything off |
| `remove: Name, garment` | the garment comes off |
| `wear: Name, garment` | the garment goes on, or back into place |
| `lift: Name, garment` | the garment is pulled out of the way and stays on |
| `tear: Name, garment` | the garment is cut or torn off |
| `mark: Name, mark` | writing or a mark goes on their skin and stays on |
| `unmark: Name, mark` | the mark comes off; `unmark: Name` takes every mark off |
| `exit: Name` | the person leaves and is out of the shots after |

A `hold:` or `mark:` line in `character_memory` puts the item or mark on from the first shot.

## How to wire it up

Copy this folder into `ComfyUI/custom_nodes/` and restart the ComfyUI server. It needs ComfyUI with native MiniMax-H3
support. Coming from an earlier version, right-click the node → **Fix node (recreate)** and set your values again.

```
UNETLoader ──────── model ─────┐
CLIPLoader ──────── clip ──────┤
VAELoader (video) ─ vae ───────┼─> H3-LongVideos ─┬─ images ─┬─> Create Video ─> Save Video
VAELoader (audio) ─ audio_vae ─┤                  ├─ audio ──┘
text node ───────── prompt ────┘                  └─ info, script ─> Preview as Text
```

| input | |
|---|---|
| `model` | a MiniMax-H3 model; a [hybrid fl2va/ref2va merge](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models) works best |
| `clip` | H3's text encoder, loaded with type `minimax` |
| `vae` | the full H3 video VAE; a decoder-only VAE cannot start a shot on the last frame |
| `audio_vae` | the H3 audio VAE, a separate file (the converted one) |
| `prompt` | a multiline text node |
| `first_frame` | optional: a picture for the first shot to start on |
| `ref_image_1` … `ref_image_4` | optional: pictures of your characters (see *The prompt*) |
| `sigmas` | optional: your own schedule |

- **PDD Acc:** run the model through `MiniMaxH3PDDAccApply`, then wire its `MODEL` output into `model` and its `sigmas`
  output into `sigmas`.
- **Speed LoRAs, Hyperflow and FastH3** need no wiring: the node finds them and samples them the way they were trained.
  Wired `sigmas` override that.
- **HyperFlow H3 Flash** (the INT8 convrot checkpoint) loads with ComfyUI's own Load Diffusion Model, with no extra node
  packs, and runs on its own 4-, 6- or 8-step grid, whichever is nearest `steps`.
- **Fewer than 6 steps:** the sound passes run on a finer schedule so spoken words stay clear. With `foley_resolution` off,
  they don't.
- **Reference pictures** are cropped to face and hair when `comfyui_controlnet_aux` is installed with its DWPose files
  (`yolox_l.torchscript.pt` and `dw-ll_ucoco_384_bs5.torchscript.pt`). Without them, the pictures go in whole.

| widget | |
|---|---|
| `resolution`, `megapixels` | aspect ratio and size; 1.0 is the ratio's own H3 size (16:9 is 1344×768); above 1.0 the frames are rendered at 1.0 and scaled up |
| `shot_seconds` | the longest a shot may be, up to 15 seconds |
| `shot_length` | *from the beat*: each shot as long as its beat takes; *fixed*: every shot is `shot_seconds` |
| `steps`, `sampler_name`, `scheduler`, `seed` | as in KSampler; one seed for the whole video |
| `shift_video`, `shift_audio` | H3's sigma shift, 12 and 3 by default |
| `silence_wordless` | render shots with no spoken line silent, then make their sound |
| `foley_resolution` | the picture size the sound pass works from; *off* leaves those shots silent |
| `plan_only` | write every shot's text to `script` and `info` without rendering |
| `anchor` | text kept at the top of every shot; with it, every paragraph is a shot |
| `character_memory` | the characters, their pictures and what they have on from the start |

| output | |
|---|---|
| `images`, `audio` | the whole video and its soundtrack |
| `info` | every shot's cast, what goes on or comes off, warnings and timings |
| `script` | the exact text each shot was given |
| `frames_per_shot`, `total_frames`, `shots`, `video_seconds` | counts |

While a video renders, the node shows the plan and the shot it is on, live.

Two helper nodes come with it: **H3 Shot Length** gives a shot length as seconds and as a valid H3 frame count, and
**H3 Overlay** puts a watermark and an intro title on the finished frames. The overlay works through the video from a temp
file on disk, so the temp folder needs room for one more copy of it.

## Disclaimer

The owner of this repo will not be responsible for any copyright strikes incurred
because of use. You are responsible for your works. Use this node responsibly and
ethically.
