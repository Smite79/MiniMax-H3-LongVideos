# H3-LongVideos

Long **MiniMax-H3 video with synchronised audio** from a single prompt, in ComfyUI.

H3 renders about 15 seconds at a time. This node turns a written scene into a
**chain of shots** and joins them into one continuous video.

**Your text is never rewritten.** The node does the chaining, not the writing: what
you type reaches the model word for word.

It does *append* to it — a garment told to finish coming off, hardware told to stay
fastened, the sound an action makes. Each of those answers a failure that text alone
could not fix, each is one sentence, each has a switch, and `info` reports what was
added and what share of the shot it came to.

---

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart the ComfyUI **server**
(not just a browser refresh).

**Requires ComfyUI 0.31 or newer** with native MiniMax-H3 support. Tested on 0.33.

## What you need loaded

| | |
|---|---|
| **UNET** | a MiniMax-H3 diffusion model — a **hybrid fl2va/ref2va merge** gives the best results here, see [below](#which-checkpoint) |
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

### Everyone in a beat needs an entry

A `Name: attributes` paragraph — or `character_memory` — describes the cast, and the
node stamps the entries for the people a beat names into that shot.

**One name per person, and an entry for every name.** A name the beats use and the
sheet never describes is a person nothing in the shot describes: no age, no clothes,
no face, so the model invents them, differently in each shot. And when that is the
*only* person a beat names, the shot falls back to the previous beat's people — so it
describes someone who is not in it and stays silent about the one who is.

The commonest way in is calling one person two things: `Dan` in some beats and `Mike`
in others reads as two people, one of them a stranger.

`info` names every such person and the shots they are in. It is only ever reported —
whether that name is somebody already on the sheet or a third person in the room is
not answerable from the text, and guessing would be the node rewriting your script.

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
the action**: a hand that pulls a coat off a shoulder keeps pulling.

```
Mara pulls off her wet coat and drops it on the bench.      ~7s of content
Mara pulls off her wet coat and drops it on the bench,      ~12s
then wipes the rain off her face and sits down.
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

**Describe what is visible.** A scene that lists every layer at once — coat, jumper,
shirt — tells the model the character is wearing all of them *simultaneously*,
with nothing saying which is hidden. The keyframe holds the first frame, so the shot
starts right; by the last frame only the text is governing, and the under layer starts
showing through the top one.

So list the outer layer, and bring each one in as it becomes visible:

```
A hallway, cold light. Mara is 30, red hair, wearing a wet grey coat.

Mara pulls off her coat and hangs it on the hook.
remove: coat
add: her navy jumper underneath is now visible

She pulls the jumper over her head and drops it on the chair.
remove: jumper
add: her white shirt is now visible

Mara looks back at the door.
```

Which produces:

```
shot 1  ... Mara is 30, red hair. Her navy jumper underneath is now visible.
shot 2  ... Mara is 30, red hair. Her white shirt is now visible.
shot 3  ... Mara is 30, red hair. Her white shirt is now visible.
```

**Removals are read from your beats.** `auto_remove` (on by default) takes the
garment off when the beat says so — no directive needed:

```
A hallway, cold light. Mara is 30, red hair, a wet grey coat, a navy jumper.

Mara pulls off her coat and hangs it up, showing the navy jumper underneath.

She folds her arms.
```

```
shot 1  ... red hair, a navy jumper.   Mara pulls off her coat...
shot 2  ... red hair, a navy jumper.   She folds her arms.
```

Two conditions are both required, because a wrong removal is worse than a missed
one: the beat must contain a **removal verb**, and the thing named must be something
the **scene already says is worn**. Only the verb's own object counts — the span up
to the next clause boundary — so *"pulls off her coat, showing the jumper"* takes the
coat and leaves the jumper. `info` reports every removal it reads, by shot.

The verb can sit either side of its particle — *"kicks off her boots"* and *"kicks
her boots off"* both read — and *"steps out of"*, *"wriggles out of"*, *"slides off"*
and *"lifts it over her head"* are removals too. A particle belongs to the nearest
verb in front of it, so *"kicks the chair and walks off"* is not one, and in the
trailing form the particle ends the object: *"takes her coat off and drops it on the
chair"* takes the coat and leaves the chair alone.

**Undressing completely is the one case read from the *wardrobe* rather than the
beat.** *"strips out of their clothes"*, *"undresses"*, *"is naked"* name no garment,
so nothing else has anything to take off — and the scene goes on listing the whole
wardrobe, restamped into every later shot, which is how the clothes come back on. So
the garments are read off the character sheet instead and all of them come off:

```
Nora: 34, she, red hair, green canvas jacket, grey wool jumper, white t-shirt,
black jeans, brown leather boots.

Nora undresses completely and steps into the shower.
```

```
shot 1  ... red hair, green canvas jacket, grey wool jumper, ... brown leather boots.
shot 2  ... red hair.   Nora undresses completely...
shot 3  ... red hair.   Nora reaches for a towel.
```

It is scoped to the people the beat names, so undressing one character does not take
the other one's clothes off, and it says so in **one** sentence rather than reciting
five garments back at a shot whose point is that there are none. **Restraints are not
clothing** — taking clothes off unlocks nothing, and hardware is still cleared only by
an explicit `remove:`. `info` lists exactly which entries were cleared, so anything
the wardrobe vocabulary missed is visible rather than silent; name those in a
`remove:` line.

`remove:` still works and is added to whatever is inferred — use it when the wording
is unusual enough that the beat is not read correctly, or when something comes off
that the prose does not name as a removal.

`remove:` drops any part of the scene naming that item, from that shot onward.
`add:` (or `wear:`) appends your phrase to the scene, in your words, from that shot
onward — and retires automatically when a later `remove:` names it.

Both take effect on their own shot: the keyframe already shows the previous state at
the start, and it is the *description* claiming a garment is still worn that puts it
back. The directive lines never reach the model.

`remove:` also tells that shot to **finish the job**: it appends one sentence saying
the item comes off during this shot and is away by the last frame, fully removed and
out of frame. That matters because the last frame becomes the next shot's keyframe —
a cut still in progress hands on a garment still half worn, and the next beat has
moved on and never contradicts the picture, so it stays on.

That sentence is said **once**, in the removing shot. Later shots simply never
mention the garment: naming it again — even to say it is gone — is a presence cue,
and "no longer wearing the red jacket" is what put garments back on in the previous
version of this node.

**If a garment still comes back with the text clean, the picture is doing it.** Every
shot is anchored to the previous shot's last frame. If the model does not finish
taking the garment off inside its own shot, that frame still shows it — and a
keyframe is a *picture*, which outvotes any sentence. Inherit it once and every later
shot inherits it too, with no wording able to undo it.

`restart_after_removal` (on by default) breaks that inheritance: the shot after a
`remove:` starts fresh instead of continuing from that frame. The cost is a visible
cut there, and that shot re-deriving its pose and framing from the text — which is
why it is one boundary and not every boundary. `info` names the shots it applies to.

Check the `script` output first. If the garment is absent from the text and still on
screen, it is coming through the keyframe and this is the setting that stops it.

**The same applies to your own beats.** They go to the model word for word, so a
later beat that says *"her coat"* puts the coat back — the scene is clean and
the removal was honoured, and then the beat asks for it. In a ten-beat script that
reads as the removal failing at random. `info` names the beat and the garment when it
happens; the removing beat itself is not flagged, since it has to name it.

This is the only place the node edits your text, and it does exactly what you tell
it — no removals are inferred from your prose.

## Things that must stay the same

**Fastened hardware is held for you.** `hold_restraints` (on by default) notices when
something that fastens is put on a person — handcuffs, a chain, rope, tape — and from
that shot onward every shot carries one sentence: *every restraint stays whole and
closed, fastened exactly as it was put on*. It latches, so a beat that does not
mention it does not mean it came off, and a `remove:` naming it clears it.

This is the **only** continuity fact the node asserts on its own, because it is the
one that cannot be recovered: hardware that renders open or snapped is not a detail
that drifted, it is the scene ceasing to make sense. Ambiguous items need a
fastening verb or a body part alongside them, so a chain-link fence and a leather
belt do not arm it.

Four refinements ride along with it, each only where it applies:

- **Hardware gets somewhere to sit.** A collar with no neck named beside it is a
  band with no place to be, and a model handed a band-shaped object and no anatomy
  puts it where bands usually go — on the head. Where an item is named with no body
  part within a sentence of it, the shot says where it belongs: a collar at the
  neck, a gag in the mouth, cuffs at the wrists, a leash clipped to the collar. That
  is not a creative choice, it is what the object is — and it applies to an item
  merely held up and shown, not only one being fastened. **Name the part yourself
  and nothing is added**: what you wrote wins.

- **Rigid hardware stays rigid.** Steel is not rope, but a model with no reason to
  think otherwise draws a chain as a soft cord — sagging, stretching to wherever a
  limb is going, allowing movement the hardware does not allow. Where a chain,
  padlock, cuffs or a bar are named, the shot adds that the links keep their size,
  the run between the fastenings stays straight and taut, and the body reaches only
  as far as the metal allows. Rope, tape and straps do flex, so nothing claims
  otherwise for them.
- **A bound body falls as one piece.** A falling body puts its hands out; with the
  hands fastened, the cheapest way for the model to resolve that is to free them,
  which renders as the hardware giving way. Where a beat has someone go down, the
  shot says what takes the landing instead.
- **A turn shows a side the keyframe never pinned**, and the model fills it from a
  clothed prior. Where a beat turns or moves a body, the shot says what is on it now
  is all that is on it, from every side.

**Turning is handled too.** The keyframe pins the *front* of the body. When a beat
turns someone — or brings the camera round behind them — the model is filling in a
surface it has never been shown, and its prior for an undescribed body is a
**clothed** one. That is a removed garment coming back, often stacked in the wrong
order, and anything on the far side re-inventing itself as it rotates into view.

**Being moved counts too** — lifted, carried, rolled, laid down. Same reason: the
keyframe pinned one pose seen from one side, and moving the body puts it where that
frame never showed it. The verb needs a *person* as its object and somewhere to go,
so *"lifts her onto the table"* counts while *"lifts the crate"* and *"positions her
legs"* do not.

On those shots, and only once there is state worth holding, one sentence is added:
*the body reads the same from every angle and in every position — what is on it now
is all that is on it, front, side and behind, and whatever is fastened stays
fastened.* It names no garment and no person.

**Anything else that has to hold across the chain goes in the scene paragraph**,
which reaches every shot — a property of the light, a fact about the room, a
condition of a costume. Use `add:` if it only becomes true partway through, and
`remove:` when it stops being true.

Two rules worth knowing, both learned the hard way:

- **State what IS, not what is not.** At `cfg 1` the negative prompt is never
  evaluated, so a negation in the positive only names the thing it forbids. *"the
  glass stays intact"* works; *"the glass does not break"* names breaking.
- **Do not re-state a removed garment.** Once something is off, never mention it
  again — not even to say it is gone. A mention is a presence cue.

`remove:` already covers the removing shot itself: it tells that shot to finish the
removal by the last frame, and adds that *everything else on the body stays exactly
as it is, untouched and still fastened* — which is what stops an action running on
into whatever is next.

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
- **`ref_image_1…4`** are identity references. Tag one onto the person it depicts —
  `Nora: <Picture 1>, 34, she, …` — and it rides alongside the keyframe rather than
  instead of it. See [below](#a-reference-and-the-keyframe-ride-together).
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

### Which checkpoint

**Use a hybrid fl2va/ref2va merge.** This node leans on *both* halves of H3 at once —
the keyframe chain that joins one shot to the next, and reference images for identity
— and MiniMax split those across two checkpoints that are each weak at the other's
job:

| | keyframe chain | reference conditioning | output quality |
|---|---|---|---|
| **fl2va** | trained for it | not trained for it | better |
| **ref2va** | — | trained for it | noticeably worse |

So on fl2va a reference fights the staging your beat describes, and on ref2va you pay
for it in picture and audio quality across every shot. Neither is a good fit for a
chain that wants both.

The hybrid merges resolve that. They combine the two official checkpoints at the
tensor level — each weight taken from one side or the other, no fine-tuning — to keep
fl2va's quality while retaining ref2va's reference conditioning:

**[smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models)**
— several variants, around 21 GB each. Put one in `models/diffusion_models/` and load
it with the UNETLoader as usual; nothing in the node needs changing.

If you are on a plain **fl2va** checkpoint, references still work — they go where
tagged, on every beat naming them, which is what holds a face across a chain. But
fl2va was never trained on reference conditioning, so a near-clean reference there
also pulls pose and framing towards the picture, and on a shot that is not
introducing the character that competes with the staging your beat describes. That is
the arrangement a hybrid removes; on fl2va, `ref_noise_aug` is the dial for it.


### A reference and the keyframe ride together

They are not alternatives, and a shot uses both:

- the **keyframe** — the previous shot's last frame — anchors where this shot *starts*
- a **reference** only says *who somebody is*

ComfyUI packs both, in orders that agree: `model_base.py` builds the conditioning
latents as keyframe-latents-then-reference-latents, and `PackedLayout` emits keyframe
segments then reference segments. So both channels coexist and neither displaces the
other.

**Tag the reference onto the person it depicts**, in their sheet entry:

```
Nora: <Picture 1>, 34, she, tall, red hair tied back, green canvas jacket.
```

References keep slots `1…N`, which is what that tag points at; the handoff is appended
*after* them, where it disturbs no numbering. Tags are renumbered per shot, because
the encoder numbers by the order it receives images — a shot carrying only slot 2
receives that image as `<Picture 1>`.

**References go where tagged — every beat that names them.** That is what holds a face
across a chain instead of letting it drift down the keyframe handoffs, and it is the
reason to tag the *person* rather than a single shot: written in the character sheet,
the tag travels with them into every shot they appear in, and a shot the character
guard trims them out of carries no reference at all.

A picture the prompt never refers to is read as *another* subject, which is why `info`
reports any shot carrying one it does not name.

**With no tag anywhere, references go on every shot.** Placing by tag would otherwise
place them nowhere, which is a connected input silently doing nothing.

**The trade on a plain fl2va checkpoint.** A near-clean reference asks the model to
reproduce the *picture* — pose and framing, not only the face. On the shot introducing
a character that is the point; on later shots it competes with the staging your beat
describes, and the referenced person can hold the portrait's gaze while anyone without
a reference gets placed relative to that composition and then travels to where the
text put them. That is the price of the face holding, and `info` says when you are
paying it. A [hybrid checkpoint](#which-checkpoint) is trained for reference
conditioning and does not make this trade; on fl2va, `ref_noise_aug` is the dial.

### If the reference turns up as the opening frame

**Lower `ref_noise_aug`.** That is the dial for it, and the symptom is a matter of
degree rather than a format error. At the default **0.999** a reference is handed over
essentially noise-free, and a noise-free image is an invitation to *reproduce* it —
framing and background along with the face. Try **0.95**, then **0.90**: the reference
then informs the face without being copied.

It shows most on **shot 1**, which has no previous frame and therefore no keyframe
competing with that invitation. Below 0.99 the handoff stops being a keyframe and
rides as an extra reference instead, so continuity weakens as identity strengthens —
`info` says when that happens.

**Which image is the first frame is not decided by a label's number.** It is
`resolved_frame_index` in the keyframe payload. The `<Picture N>` labels are only how
the images are shown to the text encoder, and the one thing they have to line up with
is the tags in your prompt. Reading slot 1 as *meaning* "the first frame" is wrong,
and building around that belief is what briefly turned a connected reference into the
frame each shot opened on.

The handoff has to be in that list at all because the tokenizer is either/or: passing
reference items makes it ignore the plain image channel outright. Leave the handoff
out and the encoder is never shown where the shot left off — it is told the location
in words, given a latent anchor, and re-imagines the scenery. Same place, new room.

`ref_noise_aug` is how *clean* a reference is shown. At the default 0.999 the model
tends to reproduce it, framing included, so a head-and-shoulders reference pulls
towards head-and-shoulders framing. One aug covers every visual condition row,
keyframe included — so below **0.99** the handoff stops being a keyframe and rides as
an extra reference instead: weaker continuity, but nothing pretending to anchor while
carrying noise. `info` says when that happens.


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

### Scoring a scene

Because the model is joint, **the same prose conditions the audio branch**. A
soundscape is written, not configured — there is no sound field, and there should
not be one:

```
A cold concrete basement, a low hum off the strip light and a drip somewhere behind
the wall.

Jon's boots echo on the stone floor as he walks in.

The chain drags and rattles across the concrete as she shifts her weight.
```

**`auto_sound`** (on by default) does most of this for you: it reads each beat and
gives the shot the sound its own action implies — walking gets footsteps, scissors
get blades through fabric, a chain gets links dragging, a lock gets a lock closing.
Three at most, so the shot gets a cue rather than an inventory. It reads the **beat
only**, never the scene, so a chain standing in the scene does not rattle in a shot
where nobody moves. A beat that already describes its own sound is left alone.

It also reads the **space** from the scene and carries that under every shot — a
concrete basement gets hard walls giving the sound back, a carpeted room gets little
echo, outdoors gets no walls close by. That is the one thing read from the scene
rather than the beat, because a room is hard in every shot whatever happens in it,
while a chain standing in the scene must not rattle where nobody moves.

Room tone is what separates a recording from a sound effect: real footage has a bed
under the events, and digital silence between them is what makes a scene sound
staged.

### Nothing the node infers can open the audio branch

This is the rule that stops the mouth moving, and it is worth stating on its own.

H3 is **joint**: the picture follows the audio. Leave the audio branch free on a shot
with no line and it fills itself with a **voice** — and the face lip-syncs to the
babble. No wording suppresses that. *"The only sounds are footsteps"* was tried, and
the mouth still moved. The only thing that settles the branch is **conditioning** it,
and the silent keyframe pins the shot's whole length, not just its opening.

So the branch is opened by **what you wrote**, and by nothing this file worked out:

| | opens the branch |
|---|---|
| a quoted line, or `<d>…</d>` | **yes** |
| a sound *you* described in the beat | **yes** |
| footsteps `auto_sound` inferred from "walks in" | no |
| room tone read from the location | no |

A shot with neither is pinned to silence and is told **nothing** about sound — a
sound sentence there would describe an acoustic the conditioning removes.

`auto_sound` is therefore **text only**. It adds the sound an action implies to shots
that are already open, and it can never unsilence one. On a shot that is open but has
no line, it closes the list —

> *The only sounds are footsteps and a large room with a long tail.*

— which shapes a branch that is legitimately free. Phrased positively on purpose: at
cfg 1 H3 is CFG-free and no negative prompt is evaluated, so *"nobody speaks"* is not
a prohibition, it is the word *speaks* in the prompt. A shot with a line keeps the
open form (*"It sounds like…"*), since closing the list there would say the line is
not among the sounds.

**The trade this makes.** A shot with no dialogue gets no ambience unless you write
it. That is the cost of a hard guarantee, and on a joint model there is no third
option — either the audio is pinned, or it can talk. To score a silent shot, describe
it yourself and the branch opens for it:

```
Nora walks to the window, her boots loud on the concrete,
a low hum off the strip light.
```

`info` counts both groups every run, so you can see which shots are silent and why.

Write it yourself when you want something neither the action nor the space implies —
weather, a noise off-screen, a machine, a specific quality of hum.

Sound in the **anchor** or first paragraph carries through every shot — room tone,
the space, the ambience. Sound in a **beat** belongs to that shot — footsteps,
a door, a chain, breath.

Two things to know:

- **Silence is not "no speech", it is "no sound at all".** `silence_nonspeech`
  conditions the audio branch on encoded silence, so a shot it silences has no
  footsteps, no room tone, nothing. A beat that *describes* a sound is exempt — it
  is asking for audio on purpose — but a beat that describes none is scored as
  silent. `info` says which shots were which, every run.
- **Never label it.** `sound:` or `soundscape:` at the start of a line is read as
  text to *draw*, and turns up on screen. Write it as prose, in the sentence.

Turning `silence_nonspeech` off gives every shot a free audio branch: full ambience
everywhere, at the risk that a shot with no line invents speech and the mouth
follows it.

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
