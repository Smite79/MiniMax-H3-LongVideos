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

H3 renders up to about 15 seconds at a time. This node renders your scene shot by shot,
starts each shot on the last frame of the one before, and joins them into one video
with one soundtrack. Your text reaches the model word for word.

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart the ComfyUI **server**.
Requires ComfyUI with native MiniMax-H3 support.

**Updating from an earlier version:** the node's widgets have changed. Right-click the
node → **Fix node (recreate)**, set your values again and save the workflow. Inputs an
old workflow still has linked that the node no longer uses are ignored and named in
`info`.

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

Turn on **`plan_only`** to see the exact text every shot will get, without rendering.

## Writing the prompt

Paragraphs are separated by a blank line.

- The **first paragraph is the scene**: who is there, what they look like, where they
  are. It opens every shot, so keep actions out of it.
- **Every paragraph after it is one shot**, sent word for word.
- A prompt with a single paragraph is one shot.
- **`anchor`** (optional) is framing for the whole film: look, camera, lighting,
  location. It goes at the front of every shot. When it is filled in, every paragraph
  of the prompt is a shot and there is no scene paragraph.
- **`character_memory`** (optional) is who is in the film and what they wear. It
  follows the scene in every shot.

```
A dim cell with a cot. Mara <Picture 1> is a tall woman in a grey dress. Dan <Picture 2> is a guard in a dark uniform.

Mara sits on the cot and stares at the door.

Dan handcuffs her wrists behind her back.
hold: Mara, handcuffs behind her back

Dan presses duct tape over her mouth.
hold: Mara, duct tape over her mouth

Dan says "Not a sound." He walks out.
```

### Lines the node reads

These lines are taken out of the text before the model sees it. Nothing else is
guessed from your writing.

| line | what it does |
|---|---|
| `hold: Name, item; item` | Name is wearing or bound with these items. From the next shot on, every prompt ends with `Name: item; item.` until the item is released. |
| `release: Name, item` | the item comes off in this shot. `release: Name` takes everything off Name. |
| `seconds: 6` | this shot's length, instead of `shot_seconds`. |
| `cut` | this shot starts fresh instead of on the previous shot's last frame. Use it for a new place or time. |

A `hold:` line in the scene paragraph, the anchor or the character memory means the
person starts the video already held.

Write each item the way it should look: `handcuffs behind her back`,
`wrists zip tied in front`, `rope around her ankles`, `duct tape over her mouth`.
The shot where an item goes on is described by your own sentence. The item joins the
held line from the shot after, so it is not drawn before the action happens.

### Reference pictures

Wire up to four pictures into `ref_image_1` … `ref_image_4` and refer to them in the text
as `<Picture 1>` … `<Picture 4>`. A tag with no picture wired is removed.

Once a person is held, their own picture (written right after their name, as in
`Mara <Picture 1>`) is left out of shots that start on the previous frame. A portrait
without the cuffs or the tape pulls them back off. The picture comes back after a `cut`.

### Sound

A shot with a quoted line (`"…"` or `<d>…</d>`) speaks, and its first half second is
held quiet so the line does not start on the cut. A shot without one is held silent so
no voice is invented. Turn `silence_wordless` off to let the model add sound there.

## Pose control

Text alone cannot stop H3 from freeing restrained arms or catching a fall with cuffed
hands. Pose control checks each shot with DWPose and renders it again where the
restraint broke.

**Wiring:** add a **Load Model Patch** node with
`minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` (in
`models/model_patches`) and connect it to `pose_controlnet`.

**Needs:**

- the **hybrid b25-49** checkpoint. The controlnet is built for its 8-wide timestep
  embedding, and `info` says so when the model cannot carry it.
- the DWPose files from `comfyui_controlnet_aux`:
  - `ckpts/hr16/yolox-onnx/yolox_l.torchscript.pt`
  - `ckpts/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt`

**What it does:**

- Every shot with a held arm or ankle position is checked. The positions come from your
  `hold:` lines: *behind her back*, *in front*, *above her head*, *at her waist*,
  *ankles*, *ankles to her wrists*.
- If the restrained person's limbs left that position, the shot is rendered again on
  the same seed, with their skeleton held in place for the first `pose_end` of the steps.
- In the shot where the restraint goes on, the limbs are held only from the moment they
  close.
- Some shots keep their first render instead:
  - a restraint fastened to an object (`to the bed`)
  - a restraint coming off
  - any shot where the node cannot tell who is restrained
- `info` has a line for every checked shot.
- Each repaired shot costs one extra render.

## Upscaling

- **`latent_upscale`** samples each shot at `resolution` and enlarges it in latent space
  before decoding, which is much cheaper than sampling large.
  - It needs the Minimax H3 Latent Upscaler node pack, with its H3 model in
    `models/latent_upscale_models`.
  - `latent_upscale_scale` sets the factor.
  - The next shot is still handed a frame at the sampled size.
- **`upscale`** enlarges the finished video:
  - `rtx`: NVIDIA RTX Video Super Resolution (needs the `comfyui_nvidia_rtx_nodes` pack)
  - `model`: the model picked in `upscale_model`, from `models/upscale_models`
  - `lanczos`: a plain resize
- **`upscale_target_short_edge`** fits the result's short edge to that many pixels.
  `lanczos` needs it; for `rtx` it also sets the factor.

## FastH3 and Hyperflow

- **FastH3**, detected from the model, runs at shift 10/3 with the VSA attention it was
  trained on (keep 10%, from 20% of the schedule). If torch is on the `cudaMallocAsync`
  allocator, restart ComfyUI with `--disable-cuda-malloc`.
- **Hyperflow**, detected from the LoRA's metadata or its file name, samples every shot
  on its own 8-step grid (shift 12/3, `euler`). Leave `sigmas` unwired.
  - Its endpoint conditioning is added back from `hyperflow_endpoint_v1.0.safetensors`
    beside the node, or from the original `minimax_h3_hyperflow_8step_v1.0.safetensors`
    in `models/loras`.
  - That needs a checkpoint with a time embedder, so Hyperflow two-time and pose control
    never run together.

## Settings

| widget | |
|---|---|
| `resolution`, `megapixels` | aspect preset and size; at 1.0 each preset is its native size, 0 keeps it exactly |
| `shot_seconds` | default shot length |
| `steps`, `sampler_name`, `scheduler`, `seed` | as in KSampler; one seed for the whole video |
| `first_frame` | the first shot starts on this picture |
| `sigmas` | your own schedule (overrides Hyperflow's grid) |
| `shift_video`, `shift_audio` | H3's sigma shift, 12/3 by default |
| `silence_wordless` | keep shots without a quoted line silent |
| `plan_only` | report the shots and their text without rendering |
| `pose_strength`, `pose_end` | how strongly, and for how much of the schedule, the skeleton holds |
| `anchor` | framing at the front of every shot; filled in, every paragraph is a shot |
| `character_memory` | who is in the film, after the scene in every shot |
| `latent_upscale`, `latent_upscale_scale` | upscale each shot in latent space, and by how much |
| `upscale`, `upscale_model`, `upscale_target_short_edge` | upscale the finished video |

## Outputs

| slot | what it is |
|---|---|
| `images` | the finished frames |
| `audio` | the synchronised soundtrack |
| `info` | what the node did for each shot |
| `script` | the exact text each shot was given |
| `frames_per_shot`, `total_frames`, `shots`, `video_seconds` | for downstream nodes |

## Other nodes here

- **H3 Shot Length**: one shot length as seconds and as a valid H3 frame count.
- **H3 Overlay**: watermark and intro title composited onto the finished frames.

## Notes

- No negative prompt and no cfg setting: H3 runs at cfg 1.
- Denoise is fixed at 1.0: partial denoise desyncs the joint audio/video schedule.
- Widgets are restored by position. If they read NaN, recreate the node as described
  under Install.

## Disclaimer

The owner of this repo will not be responsible for any copyright strikes incurred
because of use. You are responsible for your works. Use this node responsibly and
ethically.
