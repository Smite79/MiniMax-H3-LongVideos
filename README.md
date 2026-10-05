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

- The **first paragraph is the scene**: where it takes place, who the characters are
  and what they look like. It opens every shot, so keep actions out of it. It does not
  put anyone in a shot: the beats do.
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

### Restraints

The node reads restraints and gags from your beats and carries them from shot to shot:
handcuffs, zip ties, rope, chains, shackles, duct tape, gags and ball gags, blindfolds and
collars. It reads them being put on (*Dan handcuffs her wrists behind her back*, *Dan
grabs Mara's wrists and cuffs them*, *Dan takes out the handcuffs and snaps them onto her
wrists*, *Dan wraps duct tape around her mouth*), already on (*her wrists cuffed behind
her back*, *Mara sits in handcuffs*, *Mara sits with duct tape over her mouth*), and
coming off (*Dan pulls the tape off*, *Dan unlocks the cuffs*).

- Whoever does it is never the one restrained. In *Dan grabs Mara and cuffs her* the
  cuffs go on Mara, and in *Dan gags her* the pronoun is read as the one other person
  present.
- Something put on during a beat is in plain view at the end of that shot and held from
  the next. Something a beat describes as already on is held from that shot's first frame.
- When it cannot tell who is meant, it lists the sentence in `info` instead of guessing.
  Add a `hold:` line for those.
- Handcuffs, zip ties, shackles, ball gags or blindfolds that a beat mentions but that
  could not be placed on anyone (*Dan carries handcuffs on his belt*) are listed in `info`
  as not read. If they belong on someone, add a `hold:` line.
- Something comes off only when a beat takes it off directly (*removes the tape*,
  *peels the tape off her hips*). Mentioning it, or cutting more tape, does not count.
  When more than one piece could be meant, the sentence is listed in `info` instead.
- Each shot's line in `info` says what goes on, what is held, and what comes off.
- Tape or rope around the hips or waist, or between the legs, is carried as you wrote
  it and gets no pose; legs tied together count as the ankles.

### Clothing

The node reads clothes coming off and going back on (*Max takes off her jacket*,
*Crystal pulls her thong down her legs*, *Crystal puts her jacket back on*). From the
next shot on, a garment that came off is taken out of that person's description in
the scene paragraph, the anchor and `character_memory`, so no shot asks for it again.
It comes back once they put it on.

- A garment belongs to the person its pronoun or name points to, or else to the one
  person whose description mentions it.
- When it cannot tell whose it is, the sentence is listed in `info`.

### Lines the node reads

These lines are taken out of the text before the model sees it. `hold:` and `release:`
correct the restraint reading, and `remove:` and `wear:` the clothing reading. For the
person they name, they replace whatever was read from that beat.

| line | what it does |
|---|---|
| `hold: Name, item; item` | Name is wearing or bound with these items. From the next shot on, every prompt states `Name: item; item.`, that it all stays on for the whole shot, and where held wrists or ankles stay, until the item comes off. |
| `release: Name, item` | the item comes off in this shot. `release: Name` takes everything off Name. |
| `seconds: 6` | this shot's length, whatever `shot_length` says. |
| `remove: Name, garment` | the garment comes off in this shot and leaves Name's description from the next shot on. |
| `wear: Name, garment` | the garment goes back on, and back into the description from the next shot on. |
| `exit: Name` | Name leaves during this shot and is gone from the next. A named exit such as `Dan walks out` is read without it; use it for anything else, such as `He leaves.` |
| `cut` | this shot starts fresh instead of on the previous shot's last frame. Use it for a new place or time. |

A `hold:` line in the scene paragraph, the anchor or the character memory means the
person starts the video already held.

The node also reads these lines when they are written on the same line as the
sentence, as their own paragraph, with a dash or colon after the name, or without the
comma. A line it still cannot read, such as a hold with no name, is listed in `info`.

In a `hold:` line, write each item the way it should look: `handcuffs behind her back`,
`wrists zip tied in front`, `rope around her ankles`, `duct tape over her mouth`.
The shot where an item goes on is described by your own sentence, and it closes with
the item in plain view at its last frame. The item joins the held line from the shot
after, so it is not drawn before the action happens.

### Who is in each shot

The node keeps track of who is present, so people appear only when they should.

- A character is anyone with a line in `character_memory` (`Mara: a tall woman`), a
  name written right before a picture tag (`Mara <Picture 1>`), or a name on a `hold:`
  or `exit:` line.
- Anyone named in a beat is present from that shot on, until they leave. So is someone
  a beat calls *her* or *him*, when they are the only woman or man among the characters.
- The scene paragraph and the anchor describe people but do not put them in a shot.
- A beat with nobody in it (*The camera pans across the empty bathroom*) says that nobody
  is in the shot.
- After a `cut`, only the people that beat names are present.

For someone who is not present:

- their picture is left out
- their `character_memory` line is left out
- any scene sentence that is only about them is left out

When everyone in a shot is a declared character, a shot with one or two people also
says how many are in it. The count is left out whenever someone else might be there:
a name that is not declared, or *a guard*, *the man*, *a crowd*. Declare every
character, with a `character_memory` line or a picture tag, so the guarding covers
them all.

### Continuity

Every shot that starts on the previous shot's last frame says so in its prompt: the
same place and the same people, one moment earlier, with nobody new joining. That
frame reaches H3 as a picture, and a picture the prompt does not mention is read as
another person.

After a `cut`, someone who was last seen alone is given that frame as their current
look, in place of their portrait. Their clothes and anything held on them carry over.

### Reference pictures

Wire up to four pictures into `ref_image_1` … `ref_image_4` and refer to them in the text
as `<Picture 1>` … `<Picture 4>`. A tag with no picture wired is removed.

Once a person is held, their own picture (written right after their name, as in
`Mara <Picture 1>`) is left out of shots that start on the previous frame. A portrait
without the cuffs or the tape pulls them back off. After a `cut` the picture comes back,
unless a later frame of them stands in for it.

FastH3 never gets the pictures (see below).

### Sound

- **Speech:** a shot with a quoted line (`"…"` or `<d>…</d>`) speaks. Its first half
  second is held quiet so the line does not start on the cut.
- **No talking without a line:** every shot without a quoted line renders its picture
  with the audio held silent and says that nobody speaks and every mouth stays closed,
  so nobody's mouth moves to mumbling. The closed-mouth wording is left out when the beat
  has its own vocal sound (*screams*, *gasps*) or a gag holds a mouth open.
- **Foley and ambience:** such a shot keeps the sounds its beat brings (footsteps, a
  door, cuffs closing, tape tearing) and the ambience of the scene (rain, traffic, a
  quiet house). Its sound is made in a second, audio-only pass over the finished
  picture, so the sound follows what is on screen and has no moving mouths to put a
  voice to. That pass costs about one extra render of the shot. A beat that describes
  its own sounds keeps those.
- **Silence:** a shot with nothing to hear stays silent. Turn `silence_wordless` off to
  render every shot in one pass with free audio instead.
- **Gagged speech:** a gagged person in a shot with speech or vocal sounds is muffled.
  Their lips stay shut under tape, or the mouth stays held open around a ball or ring
  gag.
- **Ambient bed:** wire an audio file into `ambient_audio` to have it looped under the
  whole soundtrack at `ambient_level`. It plays under the model's sound and does not
  change it.

## Pose control

Text alone cannot stop H3, or a LoRA, from freeing restrained arms or catching a fall
with cuffed hands. Every shot where someone's arms or ankles are held is checked with
DWPose after it renders, in one of two ways.

**On the hybrid b25-49 checkpoint: the skeleton is held.** The node loads
`minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` from
`models/model_patches` by itself, or uses the one wired into `pose_controlnet`.

- If the restrained person's limbs left their position, the shot is rendered again on
  the same seed, with their skeleton held in place for the first `pose_end` of the steps.
- Each repaired shot costs one extra render.
- The controlnet only fits that checkpoint's 8-wide timestep embedding.

**On any other checkpoint: the shot is rendered again.** If the restraint broke, the
shot is rendered again on a new seed, up to `pose_retries` more times. The node keeps
the take where it held, or else the take where it broke in the fewest frames. Set
`pose_retries` to 0 to turn this off.

**Needs:** the DWPose files from `comfyui_controlnet_aux`:

- `ckpts/hr16/yolox-onnx/yolox_l.torchscript.pt`
- `ckpts/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt`

**What is checked:**

- The held positions come from the restraints the node reads, or from your `hold:` lines:
  *behind her back*, *in front*, *above her head*, *at her waist*, *ankles*, *ankles to
  her wrists*.
- In the shot where the restraint goes on, the limbs are checked only from the moment
  they close.
- Some shots are not checked:
  - a restraint fastened to an object (`to the bed`)
  - a restraint coming off
  - any shot where the node cannot tell who is restrained
- `info` says which way the restraints are being kept and has a line for every checked
  shot.

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
  - FastH3 was distilled for text and first-frame shots only, not for reference pictures,
    and it draws them in as extra people. So with FastH3 the node leaves `ref_image_1` …
    `ref_image_4` and their tags out. People carry from shot to shot through each shot's
    opening frame. After a `cut` they are drawn from their descriptions in the scene
    paragraph and `character_memory`. For faces that match your pictures after a cut, use
    the base or hybrid checkpoint.
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
| `shot_seconds` | the longest a shot may be, and every shot's length when `shot_length` is *fixed* |
| `shot_length` | *from the beat* sizes each shot from its own beat: about 2.2s per action, or the spoken line, plus one action's time where something is put on, capped by `shot_seconds`. *fixed* gives every shot `shot_seconds` |
| `steps`, `sampler_name`, `scheduler`, `seed` | as in KSampler; one seed for the whole video |
| `first_frame` | the first shot starts on this picture |
| `sigmas` | your own schedule (overrides Hyperflow's grid) |
| `shift_video`, `shift_audio` | H3's sigma shift, 12/3 by default |
| `silence_wordless` | keep shots without a quoted line silent |
| `plan_only` | report the shots and their text without rendering |
| `pose_strength`, `pose_end` | how strongly, and for how much of the schedule, the skeleton holds |
| `pose_retries` | without pose control, how many times a shot whose restraint broke is rendered again |
| `anchor` | framing at the front of every shot; filled in, every paragraph is a shot |
| `character_memory` | who is in the film, after the scene in every shot |
| `latent_upscale`, `latent_upscale_scale` | upscale each shot in latent space, and by how much |
| `upscale`, `upscale_model`, `upscale_target_short_edge` | upscale the finished video |
| `ambient_audio`, `ambient_level` | an audio bed looped under the whole soundtrack, and how loud |

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
