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

H3 renders at most about 15 seconds at a time. H3-LongVideos takes a whole scene written
as beats, renders it shot by shot and joins the shots into one video with one soundtrack.
People and their clothes stay the same from shot to shot, and every spoken line is
lip-synced to whoever says it.

## How it works

1. **Plan.** Each paragraph of the prompt is a beat, and each beat becomes a shot as long
   as it takes in real life (at least 5.2 seconds). A beat that needs more than
   `shot_seconds` runs on into the next shot. The node reads who is in each beat, who says
   each line and what goes on or comes off whom, and carries that from shot to shot.
2. **Write.** Each shot's text is the scene, the characters and your beat, followed by
   notes that spell out each step of what goes on or comes off, keep everyone's clothes
   as they are and keep everyone whole and in frame. A sink, stove, bed, desk or counter the
   scene names once is said to be the only one, always in the same spot, so it is never doubled.
3. **Render.** Each shot starts on the last frame of the shot before, so the scene runs on
   without a break. Reference pictures go into every shot their person is in.
4. **Sound.** Each spoken line gets its own stretch of sound, lip-synced to its speaker. A shot with no line is rendered silent
   and gets its sound from a second, audio-only pass, and so does the rest of a shot around its lines. That pass is given the
   shot's sound alone, written the way H3 was trained to read it and never with its character sheets or notes: the room tone or
   open-air ambience of the place, what happens, restraints clinking, creaking, locking and coming off, a gag's muffled sounds,
   and who stays silent, speakers included, since their own lines are never in it, with not one word about talking in it. Talk
   with no words in quotes is left out, so nobody says words the prompt does not give. A stray voice in a wordless stretch is
   brought down to the room's level, while chains, cuffs, footsteps and knocks keep theirs.
5. **Join.** Each shot is matched to the one before in colour, contrast and sharpness, and
   the still background is carried over, so nothing drifts from shot to shot. Anything that
   moved or changed is kept as it is, down to a thin chain fastened or pushed aside, so the
   old picture is never pasted over it. The shots come out as one video with one soundtrack.

### The prompt

Characters go in `character_memory`, one per line. `<Picture 1>` … `<Picture 4>` beside a
name gives that person the picture wired into `ref_image_1` … `ref_image_4`:

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

- A first paragraph that names none of your characters and has no quotes is the scene,
  kept at the top of every shot. Text in `anchor` is kept there too.
- `cut` on its own line starts the next shot fresh instead of on the last frame.
- A `seconds: 12` line in a beat sets that shot's length.
- A tag line under a beat says outright what happens, instead of leaving it to the node to
  read:

| tag | |
|---|---|
| `hold: Name, item` | the item goes on in this shot and stays on; a bare item ("crotch chain") is placed where it goes |
| `release: Name, item` | the item comes off; `release: Name` takes everything off |
| `remove: Name, garment` | the garment comes off |
| `wear: Name, garment` | the garment goes on, or back into place |
| `lift: Name, garment` | the garment is pulled out of the way and stays on |
| `tear: Name, garment` | the garment is cut or torn off |
| `mark: Name, writing or brand` | the mark goes on her skin in this shot and stays on |
| `unmark: Name, mark` | the mark comes off; `unmark: Name` takes every mark off |
| `exit: Name` | the person leaves and is out of the shots after |

A `hold:` or `mark:` line in `character_memory` puts the item or mark on from the first shot.

- **Words written on a body and brands** ("Dana writes "SLAVE" across Sarah's stomach", "Dana brands Sarah's left hip") are
  read from the beat and stay on her skin in every shot after, until the beat washes the writing off; a brand stays. A word in
  quotes after a writing verb is read as writing, not as a spoken line. Say whose body it is by name, or with her or his when
  that can only mean one person: the node never guesses between two, and says so in `info` instead.
- **Wrists fastened to something** (a wall, pipe, chair, hook) keep the pose they were cuffed in, behind the back or in front,
  said only in terms of where they are: in front means hands together at belly height, and the fastening is at waist height (a
  short chain down to a ring in the floor, a long slack one from a hook in the ceiling or to a headboard). With no pose given they
  rest in front. A headboard says nothing about the arms; wrists chained to the ceiling or a beam with no pose given still hang
  raised. Whatever wrists or ankles are fastened to, they stay at one fixed point on it, the point it holds in the frame each shot
  opens on. A ladder, its rungs, a sawhorse or a trestle can be fastened to like any fixture. Wrists or ankles spread to two
  points (the corners of a ladder's base, the legs of a table, the bed posts) are fastened one at a time, left then right, by one
  person who reaches across from her left side, and every shot says which wrist or ankle is at which point. Whoever fastens
  someone to a fixture does it from her left side and is said to appear only once in the frame, so the model never mirrors them
  onto both sides of her, and stays there in later shots while she is held and they only look on (until a beat moves them or has
  them do something to her); a ladder, sawhorse or pillory, like a bed, is said to be the only one; the legs or arms of a ladder,
  table, chair or bed are never read as hers, and "spreads her legs wide" never makes "wide cuffs".
- **A pose** someone takes (bent over the bed, kneeling, sitting, lying down, on all fours, crouching, legs spread) holds in every
  shot after it until a beat changes it, whether they take it themselves, are put into it ("Dana throws Sarah onto the bed", "pins
  her face down on the bed", "knocks her to the floor", "pushes her into a chair", "forces her to her knees", "Sarah is thrown
  onto the bed"), are bent or pushed over something ("Dana bends Sarah over the bed"), are ordered into it, or obey a spoken order
  ("Sarah does as she is told"). Every later shot says where and which way she is (lying face down on the bed, lying on the floor,
  sitting in the chair), restraints put on her there go on where she is without standing her up or laying her down again, and
  pulled off the bed or out of the chair she is up again. Ankles cuffed to the legs of the table she is bent over keep her bent
  over it; the legs or arms of a table, chair or bed are never read as hers. Bent over a bed, desk, table or bench, her chest
  rests on it (over a sink or anything else she only leans over it), and cuffed hands stay where they are cuffed; with her hands
  cuffed behind her, "on all fours" becomes kneeling bent forward, and bent forward her locked hands lie on top of the small of
  her back. Cuffed behind her back, every shot she does not move in says she stays standing upright, or, when the scene has not
  said how she is posed, in the same position as in the frame the shot opens on, with her shoulders back and her hands locked
  behind her back, a shot where someone else only talks or picks something up included. Getting up, and going down onto her knees,
  a seat or a bed, her arms stay behind her the whole way; a scene that opens on a picture with her already cuffed starts with her
  drawing both arms back behind her. With anyone close behind her ("walks up behind her", "holds her from behind", "is grabbed
  from behind by", "against her back", "leans back against him", "spoons her", sex from behind), her locked hands stay at the
  small of her back, between her back and that body; face to face ("hugs her", "wraps his arms around her", "pulls her close",
  "kisses her", "stops in front of her", "walks up to her", standing sex, riding), they stay on the far side of her body from that
  person, under their forearms when they hold her; with her back to a wall or furniture ("pins her against the wall", "leans back
  against the counter"), they stay between her back and it. Each holds until a cut, until one of them moves away or turns round,
  or, for a wall, until she moves; spooned, she lies on her side, not her back. A beat that needs her hands while they are bound
  behind her, a chore such as washing included, is flagged in `info`. Turning round ends a bend, and the camera keeps a bent-over
  person whole in view rather than her face. "Jake spreads her legs" spreads hers, not his, and riding someone is sitting astride
  them. Someone with free hands doing a two-handed chore (washing or doing the dishes, scrubbing, rinsing, drying, wiping,
  polishing, folding, kneading, chopping, peeling, sweeping, mopping, wringing, ironing, typing), however the beat words it ("is
  washing", "starts to wash", "picks up a plate and washes it", "she" when only one person fits), is shown doing it with both
  hands, such as holding each piece in one hand and scrubbing it with the other.
- **A leash**, or any chain fastened to a collar ("chains it to the wall", "clips a chain to her collar and padlocks it to the
  radiator", "a collar with a chain leash" in a `hold:` line or in her description), is clipped to the ring at the front of the
  collar, and every shot says where its other end is. Tied off, a chain is padlocked at one fixed point on what it is tied to, the
  point it holds in the frame each shot opens on, where the padlock hangs on its own; from her throat the slack chain runs across
  her chest to the side of her nearest that point, drops beside her and lies along the floor to it, never straight down the middle
  of her front, so anyone who comes near steps over it or nudges it aside; it runs taut only while the beat pulls it tight
  ("strains against the chain"), and from a ceiling hook it hangs down. Otherwise it is in a hand: whoever clips it on keeps hold
  of it, and it changes hands as the beat says ("leads Sarah by the leash", "wraps it around his hand", "hands the end to Mia",
  "unhooks the chain from the radiator"). Dropped, or let go by someone who leaves, it drops beside her from her throat to the
  floor. Each shot names its only fixed points: the ring at her collar and what or who holds its other end. Handcuffs or shackles
  brought out near it are said to be a separate pair, linked by a few small steel links, from the shot they show up in until they
  lock on or are put away; they lock on as a pair taken up for the purpose, wherever the beat puts her hands. The leash and her
  cuffs or bound wrists are never described in the same sentence, so the leash is never drawn as part of them, and the leash comes
  first, so the last word on her hands is where they are cuffed. A drawer, door or lid opened near it moves only itself: while it
  opens, stays open and closes, the leash stays one unbroken chain between its two ends, and the drawer or door only pushes the
  slack aside.
- **Penetration** is lined up with the body, however the beat words it ("fucks her from behind", "slides his cock into her
  pussy", "rides him", "lowers herself onto his cock", "is fucked by", "pushes a dildo into her ass", "fingers her", a strap-on,
  a plug, two men, two women): the penis, toy or fingers go in at the center of her crotch, between her thighs, in line with her
  body. Standing, kneeling, squatting or riding, that line is vertical, parallel to her thighs (her spine when riding): it goes
  into her from below and points straight up, and a toy is held by its base with the hand low between her thighs, below her
  crotch, its top end inside her and its base pointing straight down at the floor. Lying down, bent over, on all fours or
  spooned, it is level and points into her toward her head. When nothing says how she is posed she is taken to be standing. With
  her hands cuffed behind her, a toy she uses on herself is never put in her hands: only its base shows between her thighs. The
  position is spelled out when the beat or her pose gives one: right behind her with his hips against her buttocks (from
  behind, bent over, on all fours, face down), on top of her between her thighs (lying on her back, missionary), her sitting
  astride his hips (riding), side by side behind her (spooning) or standing face to face (against a wall). It carries into the
  next shot while the beat keeps going ("thrusts harder") and ends when he pulls out; a toy left inside is gone once something
  else goes into the same opening. A mouth, a clit or a hand is not penetration.
- **An object** someone brings in, picks up, uses, hands over, sets down, clips on or puts inside keeps its look and its place
  (toys, paddles and crops, nipple clamps, a hood, plates and pans): named again, it is the same object and starts the shot where it
  was left, free to be moved by the beat; whatever is done with it next takes it there ("puts the plates on the counter and washes
  them in the sink" leaves them in the sink). While the beat is elsewhere it stays put, or in the hand that last took it up. A
  passing state such as dirty, wet or empty is not kept as part of its look. A toy left inside someone shows only its base.
- **Restraints keep their look**: "a red ball gag", "black leather cuffs" and "red rope" stay red, leather and rope in every shot,
  and soft cuffs are buckled and clipped together, never rigid steel handcuffs. Handcuffs stay one pair whatever happens around
  them: their two cuffs are joined by their own short, rigid links that hold them together side by side. Each restraint someone
  has on gets a sentence of its own, and no sentence names her bound hands together with clothes coming off, going up or coming
  down, or with another restraint going on, so the model never moves her hands with them; her pose alone says where her hands are.
  Whoever cuffs or ties someone's hands behind her back steps behind her, takes her wrists and draws her arms back from wherever
  they are, and closes the cuffs, rope, zip tie or tape at the small of her back, where her body hides her hands from the front.
  Taking them off happens there too: whoever unlocks, unties or cuts them steps behind her (or leans over her when she lies face
  down) and does it at the small of her back, and her hands stay behind her back until her wrists are free; freeing her own hands,
  she works behind her back by feel. Whenever the one doing it would also be "her" (or "his"), every step names whose wrists or
  ankles the restraint goes on or comes off, so the model never moves it onto the one doing it, and a restraint taken off is
  dropped to the floor unless the beat says where it goes ("puts them on the bed", "hands them to her"), so it never stays in
  anyone's hands or sits beside a leash. A beat that takes off several things at once ("takes the gag out of her mouth and unlocks
  her handcuffs", "removes her gag and handcuffs") takes off each one. A hogtie, her ankles drawn up behind her to her cuffed or
  tied wrists, has her lying face down: whoever ties it lays her face down where the beat says (on the floor when it names
  nowhere), or rolls her onto her front from her back, before lifting her ankles, and every shot it is on says she lies face down
  with her hands on top of the small of her back; "on her side" or wrists cuffed "in front of her" in the text stay that way. It
  is made of what the beat names (handcuffs, leather cuffs, chains, shackles, rope, straps, zip ties, tape), else of what is
  already on her wrists or ankles, else of what was last brought out, and of rope only when nothing is; the restraint on her
  ankles keeps the same look ("steel chains", "leather cuffs"), and the link to her wrists is of the same kind: a chain for cuffs,
  chains and shackles, else a rope, strap, zip tie or strip of tape. The shot where they go on decides where they stay: every shot
  after it opens on its last frame. A restraint someone already wears is on her from the first shot with its full description,
  wherever it is written: in `anchor` ("Sarah wears a crotch chain", "Sarah has a crotch chain locked on", "Sarah: crotch chain"),
  in her character_memory description ("naked except for a crotch chain", "with leather cuffs on her wrists and ankles"), as a
  bare `hold:` item ("crotch chain", "handcuffs", "a ball gag"), or in a beat ("stands by the bed, handcuffs on her wrists"),
  where it belongs to the one wearing it; a necklace, shirt cuffs or handcuffs on a belt stay what they are. Nothing else is
  clipped, hooked or locked onto a crotch chain or rope unless the beat fastens something to it, and hands cuffed behind her rest
  just above its back, free of it.
- **Reactions come after their cause**: a scream into a gag or a spoken line after a blow, or written "as" it lands, starts once
  the blow has landed, and every shot tells the picture that each reaction follows what causes it.

## How to wire it up

Copy this folder into `ComfyUI/custom_nodes/` and restart the ComfyUI server. It needs
ComfyUI with native MiniMax-H3 support. Coming from an earlier version, right-click the
node → **Fix node (recreate)** and set your values again.

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

- **PDD Acc:** run the model through `MiniMaxH3PDDAccApply`, then wire its `MODEL` output
  into `model` and its `sigmas` output into `sigmas`.
- **Speed LoRAs, Hyperflow and FastH3** need no wiring: the node finds them and samples
  them the way they were trained. Wired `sigmas` override that.
- **Fewer than 6 steps** (a 4-step LoRA, or Flash on 4 steps) form the sound in a few large jumps, which blurs words. The sound
  passes then run on a schedule twice as fine (Flash's own 8-step grid), and each spoken line is said again over its own
  stretch from part-way down that schedule, so it keeps its timing to the lips while the words come out clear. With
  `foley_resolution` off, neither happens.
- **HyperFlow H3 Flash** (the INT8 convrot checkpoint) works from ComfyUI's own Load Diffusion Model, with no extra node packs:
  the node fills in its low-rank blocks and its two-time conditioning from the checkpoint, and samples it on the model's own
  4-, 6- or 8-step grid, the one nearest `steps`, with euler at shift 12/3. Any other model that comes out of its loader with
  empty layers stops the node at once.
- **Reference pictures** are cropped to face and hair when `comfyui_controlnet_aux` is
  installed with its DWPose files (`yolox_l.torchscript.pt` and
  `dw-ll_ucoco_384_bs5.torchscript.pt`). Without them, the pictures go in whole.

| widget | |
|---|---|
| `resolution`, `megapixels` | aspect ratio and size; 1.0 is the ratio's own H3 size (16:9 is 1344×768) and lower renders smaller; above 1.0 the model still renders at its own size and the finished frames are scaled up, since past it H3 repeats the scene across the frame, mirrored |
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

Two helper nodes come with it: **H3 Shot Length** gives a shot length as seconds and as a
valid H3 frame count, and **H3 Overlay** puts a watermark and an intro title on the
finished frames. Like the main node, the overlay keeps the video in a temp file on disk
and works through it in pieces, so a long video is never copied into RAM; the temp folder
needs room for one more copy of it.

## Disclaimer

The owner of this repo will not be responsible for any copyright strikes incurred
because of use. You are responsible for your works. Use this node responsibly and
ethically.
