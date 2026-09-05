[![Support me on Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/smite79)

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

### Where somebody is looking

*"She is looking at the TV"* says it once, and two things pull the other way: a person
in frame faces the camera unless something says otherwise, and a near-clean reference
asks for the **portrait's** pose — which looks at the lens, because photographs of people
do. The result is somebody posing for the camera instead of watching what you named.

`hold_gaze` (on) says it a second time, as a fact about the eyes and the head rather
than an activity. It reads *looks at*, *stares at*, *glances at*, *peers into*,
*watching*, *studies*. It says nothing about where the camera is, so a shot down the
line of sight is unaffected, and looking at a **person** is left alone.

If it still pulls to camera, `ref_noise_aug` is the dial — a near-clean reference is
reproducing a portrait, gaze included.

### Restraints and where they hold

**Put the hardware on in a beat, not on the sheet.** A sheet entry listing handcuffs
goes into *every* shot, including the ones before they are applied — so she wears them
before she is caught, and the applying shot is told they are already fastened. `info`
says when your sheet and your beat disagree like this.

Once a beat puts hardware on, the node carries the item forward by name. The holds say
a restraint stays fastened but never say *what* it is, and a shot told a restraint
exists with no object to draw renders the consequence without the hardware: held hands
and a restrained posture, bare wrists. A `remove:` naming the item releases it.

**The shot that puts hardware on is told both ends** — open and off the body at the
first frame, closed on it by the last — instead of the standing hold. The standing hold
says the restraint is fastened as it was put on and still fastened at the last frame,
which read at frame 1 means it is *already closed*: the cuffs snap on immediately and
the catching and struggling happen around them, in whatever order is left. From the next
shot the standing hold is right again, because by then it is on.

Say where fastened limbs are held — *"cuffed above her head to the bed frame"*, *"cuffed
behind her back"* — and every later shot is told the same, until a `remove:` names the
hardware. Without it the hold only kept the cuffs **shut**; the position was carried by
the picture alone, and the picture is the previous shot's last frame.

**Which is why a close shot loses it.** Tight framing crops the anchor point out, so the
next shot inherits a picture that never showed it. The text now carries it instead, and
`info` names the shots where the framing is that tight. If it still drifts, give that
beat a wider frame.

### Describing a state

**A state you write down is a state the model can render by arriving at it.** *"A van
with its doors closed"* names a state and never says when it is true, so the shot opens
on the doors open and somebody closes them — the state becomes the most interesting
event in the sentence.

`hold_scene_state` (on) says the state is already true at the first frame, for doors,
gates, windows, curtains, blinds, shutters, hatches, tailgates, lids and drawers. Once a
beat has changed a state, no later shot is told the old one.

**And a staged change gets both of its ends.** Some LoRAs — silveroxide's 4-step among
them — render an action **backwards**: the beat opens the doors and the shot closes them.
A beat naming one state names neither end, so the reverse answers it just as well. The
shot working the thing is told *"The doors are shut at the first frame and open by the
last"* instead of being told the state holds.

Verbs that genuinely go either way — *pulls*, *draws*, *slides*, *swings* — get no
anchor. Drawing the curtains closes them; a wrong anchor asks for the reversal rather
than allowing it. Two sentences per shot at most, both kinds sharing that budget.

A held thing is also dropped from `auto_sound`. H3 is joint, so *"a door on its hinges"*
is not a decoration on a shot with a door in it — it is a request for a door to swing,
and it beats a sentence saying the door stays shut. A beat that genuinely works the door
keeps the sound.

Both are worse at low step counts. On a 4-step distill LoRA the layout is committed
almost immediately and there are no later steps to argue a wrong opening frame back.

**Reversal is likeliest in shot 1**, which has no previous last frame pinning where it
starts — later shots inherit one. `first_frame` pins shot 1's outright.

What this does not cover: scenery outside that list, actions other than opening and
shutting, and a state your **scene** paragraph keeps asserting after a beat changed it.
Your text reaches the model word for word, so put changing scenery in the beat rather
than the scene.

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

**Moved is not removed.** *"Pulls down her shorts"*, *"pushes up her top"*, *"shoves the
coat aside"* leave the garment **on**, so it stays in the scene and later shots are told
where it now sits. Taken as a removal it would be scrubbed instead, and a garment that
stops being described is one the model re-invents — the same pair coming back a
different pair. Putting it back (*"pulls them back up"*) releases it; a real removal
(*"pulls off"*, *"steps out of"*) or a `remove:` empties it for good.

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

**`<Picture N>` means `ref_image_N`, the socket** — `<Picture 3>` is whatever is wired to
`ref_image_3`, whether or not `ref_image_2` has anything on it. A tag naming an empty
socket is dropped from the text and `info` says so.

**The number in `script` will not always be the number you wrote, and that is correct.**
H3 reads the number as the picture's place in *that shot's* reference list, so a shot
carrying one reference always says `<Picture 1>`, whichever socket it came from. The
image is still that person's. What would be wrong is a shot carrying two references and
naming only one — a picture the text never names is read as another subject.

**A picture the prompt never refers to is read as another subject** — a second person
with the same face and clothes. That is what the tag prevents, so tag every reference
you connect.

`ref_noise_aug` is how *clean* a reference is shown. At **0.999** the model tends to
reproduce it, framing and background included; lower it (0.97, 0.95) to say
*approximate*. Below **0.99** the shot handoff stops being a keyframe and is encoded as
an extra reference, so continuity weakens as identity strengthens. That extra picture is
named in the text as the frame the shot opens on — unnamed it was read as another
subject, which put a duplicate of whoever was on screen into the later shots.

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

### Mouths on shots with no line

A shot with no line but a sound **you** wrote used to keep its audio branch open — and
an open branch invents a voice the face lip-syncs to. Nobody is speaking and the mouth
moves anyway.

`mouths_shut_when_no_line` (on) conditions those shots to silence like any other
wordless shot, and adds one sentence saying the mouths are closed. The conditioning is
what settles it; the sentence alone loses to a stream that has decided somebody is
talking.

**The cost: that shot gives up the sound you wrote for it.** `info` names those shots.
Turn the switch off to keep the ambience and accept the mouth.

**A line that belongs to a machine.** `The TV says: "Storms tonight."` is a quote, so
the shot reads as a speaking one — which opens the branch *and* turns the mouth guard
off, and the only face in frame gets handed the line. Write the set as the speaker and
the voice is given back to it, the mouths are held closed, and the audio still plays:
the television is meant to be heard. Reads *television, TV, radio, screen, speaker,
stereo, intercom, phone, laptop, PA*.

A line anybody in the room might have stays theirs — including an unattributed quote,
which is read as a person talking. Muting a real line is worse than a mouth moving.

Two exemptions, both deliberate. A beat staging **effort** — *strains, thrashes* — is
vocal, so its mouth should be open and it keeps its audio. And a **scenery** beat with
nobody in it is told nothing about mouths: describing a mouth for a person who is not
there can only be satisfied by drawing one in. Those shots are still silenced — an empty
room still babbles — the two are separate conditions.

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
| `hold_restraints` | keep hardware fastened, the same object, and where it was fastened |
| `hold_scene_state` | put a described state at the first frame, and give a staged change both its ends |
| `auto_sound` | add the sound an action implies |
| `silence_nonspeech` | silence shots with no line and no sound |
| `mouths_shut_when_no_line` | silence a wordless shot even if you wrote it a sound, so the mouth cannot move |
| `hold_gaze` | put the eyes on the thing the beat says they are looking at |
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
