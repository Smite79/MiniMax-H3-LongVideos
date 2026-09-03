# H3-LongVideos

Long **MiniMax-H3 video with synchronised audio** from a single prompt, in ComfyUI.

H3 renders about 15 seconds at a time. This node turns a written scene into a chain of
shots and joins them into one continuous video. Your text reaches the model word for
word; the node adds continuity sentences on top, each with a switch, and `info` reports
everything it added.

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
CLIPLoader ─┼─> H3 Long Videos ─> audio  ─┘
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

Use the `anchor` widget instead if you want the framing carried separately — then
**every** paragraph is a beat.

Dialogue goes in **double quotes**, or in H3's own marker `<d>like this</d>`.

### The character sheet

A paragraph of `Name: attributes` lines — or the `character_memory` widget — describes
the cast. Each shot is given the entries for the people its beat names, and nobody
else.

```
Nora: <Picture 1>, 34, she, tall, red hair, green canvas jacket, brown boots.
Mike: he, 41, dark hair, navy overalls.
```

- **Declare a pronoun** (`she`, `he`, `they`). It is what lets *"he takes her coat
  off"* resolve to the right two people. Where two characters share a pronoun, the
  one in the previous beat wins; if that settles nothing the guard describes
  **neither** and says so, so write the name in that beat instead.
- **One name per person.** `Dan` in some beats and `Mike` in others reads as two
  people, one of whom is described nowhere.
- **Every name needs an entry.** A name without one is a person the shot describes in
  no way at all, so the model invents them differently each time.

`info` names anyone it could not resolve, and which shots.

**Introducing a character: say whether they arrive or are already there.** Each shot
continues from the previous shot's last frame, and somebody appearing for the first
time is not in it. Write the entrance — *"Dan walks in through the side door"* — and
he arrives on screen with the join intact. Write him in position — *"Dan is already
sitting on the crate"* — and the shot starts fresh instead, because there is no frame
to inherit that has him in it; that costs a cut where the character appears, and `info`
says when it happens.

### Describing a state

**A state you write down is a state the model can render by arriving at it.** *"A van
with its doors closed"* names a state and never says when it is true, so the shot opens
on the doors open and somebody closes them — the state becomes the most interesting
event in the sentence.

`hold_scene_state` (on) says the state is already true at the first frame, for doors,
gates, windows, curtains, blinds, shutters, hatches, tailgates, lids and drawers. A beat
that **works** the thing — *"Mara opens the van doors"* — is left alone, and once a beat
has changed a state, no later shot is told the old one.

This is worse at low step counts. On a 4-step distill LoRA the layout is committed almost
immediately and there are no later steps to argue a wrong opening frame back.

Two things it does not cover: scenery outside that list, and a state your **scene**
paragraph keeps asserting after a beat changed it — your text reaches the model word for
word, so put changing scenery in the beat rather than the scene. `first_frame` pins
shot 1's opening outright if a state has to be exact.

## Clothing

**List what is visible**, not every layer at once. `auto_remove` (on) takes a garment
off when a beat says so:

```
Mara pulls off her coat, showing the navy jumper underneath.
```

It needs a removal verb *and* a garment the scene already says is worn. Removal verbs
work in both orders — *"kicks off her boots"*, *"kicks her boots off"* — and include
*"steps out of"*, *"lifts it over her head"*, and the undoing verbs *"unlocks"*,
*"unbuckles"*, *"unlaces"*, *"undoes"*.

*"undresses"*, *"strips out of their clothes"* and *"is naked"* name no garment, so
there the wardrobe is read off the sheet and all of it comes off — for the people that
beat names, and never the restraints.

Directives go on their own line inside a beat:

```
Mara pulls off her coat and hangs it up.
remove: coat
add: her navy jumper underneath is now visible
```

`remove:` drops anything naming that item from the scene, from that shot on. `add:`
appends your phrase from that shot on, and putting something back on later works the
same way.

## Reference images

`ref_image_1…4` carry identity. **Tag one onto the person or thing it depicts:**

```
Nora: <Picture 1>, 34, she, red hair, a silver locket <Picture 2>, green jacket.
```

The tag decides which shots get that image — every beat naming it. A person's tag in
their sheet entry travels with them; an object's tag comes off when the object does and
returns when it does. A reference with no tag anywhere goes on every shot.

**A picture the prompt never refers to is read as another subject** — a second person
with the same face and clothes. That is what the tag prevents, so tag every reference
you connect.

`ref_noise_aug` is how *clean* a reference is shown. At **0.999** the model tends to
reproduce it, framing and background included; lower it (0.97, 0.95) to say
*approximate*. Below **0.99** the shot handoff stops being a keyframe, so continuity
weakens as identity strengthens.

`first_frame` pins shot 1's opening frame. It pins the **whole** frame, so give it a
composed shot rather than a portrait crop.

## Audio

H3 is a joint model: the audio branch drives the face. A shot whose audio is left free
fills itself with a voice and the mouth follows — so with `silence_nonspeech` on, a
shot's audio is opened only by

- a quoted line or `<d>…</d>`,
- a sound **you** describe in the beat, or
- a beat staging **effort** — *thrashes, writhes, strains, shudders, grips*.

Anything the node infers — footsteps from *"walks in"*, room tone from the location —
is text only and never opens a shot. A shot with none of the three is silent unless you
write the sound into it:

```
Nora walks to the window, her boots loud on the concrete,
a low hum off the strip light.
```

`auto_sound` (on) adds the sound an action implies to shots that are already open, and
a beat naming its own sound is left alone.

**Give a line a shot it can fill.** Once a shot has dialogue its audio branch is open
for the whole shot, so the seconds the line does not cover are unconditioned in a shot
the model knows has a voice in it — which is where speech carries on past the line and
turns into babble. `info` reports the gap per shot. `shot_length: from the beat` sizes
to the line and removes it; `fixed` does not, so with a 15 s cap a four-word line
leaves about 13 s of open branch.

## Settings

| setting | value |
|---|---|
| `cfg` | **1.0** — H3 is CFG-free; the negative prompt is never evaluated |
| `sampler_name` | `res_multistep`, or `euler` with PDD Acc |
| `scheduler` | `simple` |
| `shift_video` / `shift_audio` | **12 / 3** — keep them near 4:1 or the audio breaks |
| `steps` | 6–8 with a turbo/distill LoRA, 20+ without |
| `megapixels` | 1.0 is H3's native budget; lower is faster and leaner |
| `shot_seconds` | the cap on each shot |
| `shot_length` | `from the beat` sizes each shot from its own line; `fixed` gives every shot `shot_seconds` |
| `pace` | scales that sizing — lower is brisker |

`fixed` is worth it when grain matters: noise is drawn to the latent's *shape*, so
shots of different lengths get unrelated noise from the same seed and surface detail
resets at every cut.

### Switches

| | |
|---|---|
| `plan_only` | report the plan, render nothing |
| `character_guard` | describe only the people a beat names |
| `auto_remove` | read removals from the prose |
| `restart_after_removal` | break the chain after a removal, so a garment cannot be inherited back |
| `hold_restraints` | keep hardware fastened, and the same object, once it is on |
| `hold_scene_state` | put a described state at the first frame instead of leaving it to be performed |
| `auto_sound` | add the sound an action implies |
| `silence_nonspeech` | silence shots with no line and no sound |
| `trim_seam` | drop the duplicated frame at each cut |
| `tiled_decode`, `cleanup_between_shots` | lower VRAM; leave on |
| `upscale`, `latent_upscale` | optional, off by default |

## Outputs

| slot | what it is |
|---|---|
| `images` | the finished frames |
| `audio` | the synchronised soundtrack |
| `info` | what the node did, and every warning — **read this** |
| `script` | the exact per-shot text it sent |
| `frames_per_shot`, `total_frames`, `shots`, `video_seconds` | for downstream nodes |

When a shot renders something you did not expect, check `script`: it shows precisely
what that shot was told.

## Performance

A long chain holds every finished shot in system RAM, and shares that RAM with the
models — ComfyUI offloads weights there rather than discarding them. Once the frames
crowd the weights out, each shot boundary re-reads them from disk. `info` reports what
the chain is holding. If the machine is thrashing: fewer frames per run, a lower
`megapixels`, or a smaller diffusion quant.

## Other nodes here

- **H3 Shot Length** — seconds to a valid H3 frame count (17k+5 grid, 362 cap).
- **H3 Overlay** — watermark and intro title composited onto finished frames.
- **H3 Model Inspector** — checkpoint precision, and whether your card runs it natively.

## Notes

- No negative prompt: H3 is CFG-free at `cfg 1`.
- No denoise control: fixed at 1.0, because partial denoise desyncs the joint
  audio/video schedule.

## Disclaimer

The owner of this repo will not be responsible for any copyright strikes incurred
because of use. You are responsible for your works. Use this node responsibly and
ethically.
