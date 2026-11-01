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
| **VAE** | the full H3 **video** VAE (a decoder-only one such as LightVAE cannot start a shot on the last frame) |
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

- The **first paragraph is the scene** when it only describes: where it takes place, who
  the characters are and what they look like (*is a tall woman*, *has red hair*, *wears a
  grey hoodie*, or someone introduced with their picture tag). It opens every shot, and
  does not put anyone in a shot: the beats do. A sentence in it about someone who is not
  in a shot is left out of that shot.
- When someone in the first paragraph does something (*Sarah and Dana sit on the couch
  talking*, *Sarah is lying on the bed*, *Dana cuffs Sarah's wrists*), when anyone
  speaks, or when the camera moves, the first paragraph is the **first shot** instead,
  and nothing of it is repeated in the others.
- **Every other paragraph is one shot**, sent word for word.
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
- *Behind her* right after her hands, wrists, arms or the cuffing (*Dan cuffs her hands
  behind her*) counts as behind her back. When cuffs were first read with no position
  (*Dan cuffs her*), a later beat that gives one (*her hands cuffed behind her back*)
  moves them there from that shot on. Cuffs with no position at all are drawn wherever
  the model likes, usually in front, so say where they go.
- Wrists held behind the back or in front stay locked together for the whole shot. When
  someone else in the shot does something with their hands (*Mike grabs the chain and
  pulls it*), the prompt says it is Mike who does it, with his own hands, while her hands
  stay still, so she does not slip free to do it herself. That line is left out when she
  does something with her own hands, or when the beat is about her hands.
- A leash goes with the collar it is clipped to (*Dan clips a chain leash to her collar*,
  *Mara kneels in a collar and chain leash*). While it is on, each shot names the hand
  holding it (*The leash runs from Mara's collar to Dan's hand.*), taken from the beat
  that has someone hold, lead, pull or clip it, so the other end is never drawn on a
  stranger. Unclipping the leash leaves the collar on.
- Whenever someone in a shot is held, the prompt names everyone else in it as wearing
  none of it, so a collar, cuffs or a leash are not copied onto them.
- Tape, rope or chain wrapped around the body, torso, chest or arms to the sides, or
  around the hips, waist or between the legs, is carried as wound all the way around,
  front and back, so later shots do not shrink it to strips on the front. These wraps
  get no pose. A wrap around her with no body part named counts as a torso wrap. Legs
  tied together count as the ankles.

### Clothing

The node reads clothes coming off and going back on (*Max takes off her jacket*,
*Crystal pulls her thong down her legs*, *Sarah pulls down her skirt*, *Sarah steps out
of her skirt*, *Sarah lets her skirt fall*, *Sarah drops her skirt*, *Sarah strips
naked*, *Dan undresses her*, *Sarah pulls her top over her head*, *Sarah gets out of her
jeans*, *Her diaper falls to the floor*, *Sarah's skirt comes off*, *She takes it off*,
*Sarah pushes her pants down to her ankles*, *Crystal puts her jacket back on*, *Sarah
slips into her dress*, *Dana fastens the diaper around Sarah's hips*, *Dana tapes it
shut*, *The diaper is now secured on Sarah's body*, *Sarah gets dressed*, which puts back
everything she took off),
typos such as *slides of her skirt* included. *It* or *them* is read as the garment
named just before it. A miniskirt is a skirt, a crop top is a top and a nappy is a
diaper, whichever word the description and the beat use. Besides the usual clothes,
underwear, diapers, pull-ups, bottoms, undies, joggers, trunks, overalls, towels and
the like, anything a description says someone
is *wearing*, *dressed in* or *clad in* (*wearing a blue yukata*) can come off the same
way. Only what happens counts, never what someone says (*"Put your diaper back on," Dan
says.* changes nothing), and putting a garment on a chair, the bed or someone's lap is
not putting it on. From the next shot on, a garment that came
off is taken out of that person's description in the scene paragraph, the anchor and
`character_memory`, so no shot asks for it again. It comes back once they put it on.

- Once nothing they still wear covers a part of the body, every shot they are in says
  so: *Sarah is naked from the waist down the whole shot.* (or from the waist up, or
  naked). It names no body part, because a named part is one the model turns her to
  show. The shot it comes off in says they stay that way from the moment it is off. A
  beat that says how they are (*completely naked*, *topless*, *naked from the waist
  down*) counts from that shot on.
- A reference picture shows its person in the clothes they had on, and H3 copies what it
  sees, print and all, back onto them. So the picture goes in cropped to their face and
  hair (see Reference pictures), and once a garment is off the shot also says the picture
  is for their face and hair only. `info` shows *naked:* for each undressed person.
- The prompt names the garment no more than the beat does, since every mention is one
  more pull toward drawing it.
- A spoken line that names a garment that is off (*"I'm not putting that diaper back
  on."*), or any clothing for a part of someone in the shot that is bare (*"No
  underwear, no pants."* once she is naked from the waist down; *clothes* once she is
  naked), would draw it back on. H3 acts out what is said, so even another word in its
  place (*so Mom can put a new thing on me*) is acted out, down to an extra pair of
  arms for someone who is not there. So the shot's picture is made with that whole
  sentence replaced by the speaker's own words before it, repeated to the same length,
  and its sound is then made from the real line over the finished picture; the lips
  match every sentence but that one. A line that is only about it is left out of the
  picture. `info` says *spoken diaper and underwear kept out of the picture*. Narration
  that names it (*the diaper on the floor*) is left as it is.

- A garment belongs to the person its pronoun or name points to, or else to the one
  person whose description mentions it.
- A garment that comes off takes its whole description with it: *a thong made of black
  latex*, *with lace trim*, *cut high on her hips*, *thong underwear*, and a following
  sentence such as *It is black latex.* Nothing of it is left for H3 to paint onto her.
  Its brand goes too (*white Pampers diaper with Pampers logo*; Pampers, Huggies and
  GoodNites are diapers). Only the garment's own words go: they stop at a comma, so in
  *crop top, white Pampers diaper* the crop top stays, and in *in a diaper with the toilet
  behind her, standing by the door*, the room, where she is and what she does stay. A
  detail after the garment goes with it (*leggings with a zipper that runs between her
  legs*) unless it starts with *the*, *her* or someone's name (*with the toilet behind
  her*).
- Whoever has something come off or go on in a shot is in that shot (*Dana removes Sarah's
  diaper*).
- Anything put on someone that their description does not have stays on them: a garment
  (*Sarah puts on a red hat*) or something worn (*Dana puts a pink bow in Sarah's hair*,
  *Dana clips a necklace around her neck*, *Dana puts a bib on her*: a bow, ribbon,
  necklace, bracelet, ring, earrings, glasses, pacifier, bib, hair clip and the like). From
  the next shot on, every shot they are in says so (*Sarah wears a pink bow in her hair.*),
  whether or not they were in view in between, until a beat takes it off (*Dana takes the
  bow out of Sarah's hair*). Stripping naked takes off the clothes, not the jewellery.
  Hands, cups and wipes are not additions; restraints have their own handling.
- What is worn under something else (a diaper, underwear or a bra under leggings, jeans,
  a skirt, a dress, a top) is left out of the description until what covers it comes off,
  since H3 draws everything a description names, on the outside. The shot it comes off in
  names both and says *Her leggings cover her diaper until they come off.*
- When it cannot tell whose it is, the sentence is listed in `info`.
- Something worn with a detail on one side (*chastity belt with tight rear heavy duty
  security chain going through her ass*, *a dress with a zip down the back*, *a collar
  with a ring at the front*) is stated the right way round in every shot that person is
  in. The detail can be written in the same sentence or the next (*A chain is attached
  to the back of the belt.*), with or without *a* or *her*. A chastity belt gets this
  even with no detail written. Each side is pinned to a place on the body: for a belt or
  anything on the hips, the front low on the belly and the back on the backside, with a
  fastened chain or strap between the buttocks up to the small of the back; for a
  collar, the throat and the nape of the neck; for a dress or top, the chest and between
  the shoulder blades. It turns with them, and their body never twists at the waist.
  That keeps the front of a belt from being drawn on their backside when they are seen
  from behind, or after it leaves the frame. `info` names each thing kept this way.
- A chain, strap or cord on something worn that is described as fastened or on one side
  (*a chain on the back that keeps it secured to her body*, *a rear chain*) is said to
  stay fastened at both ends, snug against them, in every shot, so it never comes back
  hanging or dragging on the floor. One you describe as hanging is left as you wrote it.
  A beat that pulls it (*Dan pulls her up by the chain*), or pulls a leash, says it keeps
  its length: the links are solid and never stretch, and the pull moves her by the belt
  or collar instead, unless the beat itself breaks, cuts or unclips it. In that shot the
  pull line stands in for the fastened line. When she wears a leash and a chained belt,
  the prompt says they are two separate chains, so the belt chain is not drawn running
  out to a hand like a leash.

### Penetration

Once a beat puts him inside her (*allowing his penis to enter her vagina*, *Sarah sinks
onto his cock*, *Sarah rides him*, *Dan fucks her*, *Dan takes her from behind*, *Sarah
has sex with Dan*; *into her ass* or *anal* makes it her anus), every shot of the two of
them says so, in terms of what shows: *Dan's penis is buried deep inside Sarah's vagina
the whole shot: her crotch and his groin stay pressed tight together, so only the base of
his penis shows, and their hips rock and grind together as they move.* The shot it starts
in says it stays buried once it is in. The motion is written as their hips moving, never
as the penis sliding, because a sliding penis is one the model draws outside her. It lasts until
a beat ends it: he pulls out, she climbs or gets off him, either of them stands or gets
up, or one of them leaves. Pulling out and going back in, or standing up while carrying
her, keeps it. While it lasts, a shot with one of them has the other in it too. His
cock in her mouth or hand is not this. `info` shows *inside: Dan in Sarah* for those
shots.

It works for any number of people. Each act is its own: who is inside whom, and where.
The sentence says which is which: whose cock it is, or who *fucks*, *enters* or *takes
her from behind*, is the one inside, and who *rides*, *mounts* or *sinks onto* him is
the one entered, a man as well as a woman. *Him* or *her* is the last man or woman named
before it. Two men in one woman (*Sarah rides Dan while Jake takes her from behind*) are
two acts, the second in her other hole unless the beat says which; two couples are two
pairs; and each act ends on its own (*Jake pulls out* ends only Jake's). Everyone joined
to someone in a shot is in it.

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

- A character is anyone `character_memory` describes, however the memory is laid out:
  `Mara: a tall woman`, `Mara is a tall woman`, `Mara, a tall woman`, `- Mara: …`,
  `**Mara**: …`, `Mara (25): …`, `Mara - …`, `MARA: …`, `mara: …`, `Name: Mara`,
  `Female: Mara, …`, `The woman: Mara, …`, or the name alone
  on its own line (with or without a picture tag or a colon) and the description on the
  lines under it, as sentences, bullets or fields such as `Hair: …` and `Outfit: …`.
  Everything up to the next name or a blank line belongs to that character, so a shot
  they are not in leaves all of it out. A `Setting:`, `Location:`, `Mood:` or similar
  line belongs to the scene, not to the character above it. Anyone written right before a
  picture tag (`Mara <Picture 1>`) or named on a `hold:` or `exit:` line is a character
  too.
- So is anyone the scene or the beats treat as a person, even with no memory entry:
  someone with a body part (*Dan's waist*, *Dan's cock*), someone a beat does something
  to or with (*rides Dan*, *straddles Dan*, *sits on Dan*, *stands in front of Dan*, *has
  sex with Dan*), or someone doing what only a person does (*Dan sits*, *Dan smiles*,
  *Dan strokes himself*). Rain, light, music, places, days and things such as *the
  Ferrari* are not. Such a character has no description or picture, so a shot with them
  says they look nothing like the declared characters, and *her* or *him* in a beat
  where they act on someone (*Dan watches her*) is the other person, which also tells
  the node who is the woman.
- A shot holds only the characters its beat names or refers to. Naming someone once does
  not put them in the shots after it. A name that is only mentioned does not count: *a
  letter from Mara*, *a text from Mara*, *Mara's photo*, *thinking about Mara*, *calls
  Mara on the phone*, *waiting for Mara*, *hears Mara upstairs*, *before Mara gets home*,
  *Mara is at work*, *Mara has left*, *screams for Mara*, `Mara (V.O.):`, and any name,
  *she* or *he* inside spoken words (*"Mara, where are you?"*). *Walks over to Mara*,
  *takes the keys from Mara*, *calls Mara over*, *kneels before Mara*, *Mara's hand* or
  Mara speaking outside the quotes (*"Stop!" Mara begs*) do. So does anyone the beat cuffs, tapes, ties or frees, and someone
  a beat calls *her* or *him*, when they are the only woman or man among the characters,
  unless the beat already names someone that *her* or *him* can mean (*The Aurora
  raises her sails as she leaves port* does not bring in another ship), or has an
  unnamed person they could mean (*A waitress brings him a drink. She smiles.*, *A man
  knocks on her door. He waits.*, *the receptionist*, *someone*): then the pronoun is
  that person and no declared character is brought in. Cuffs or clothes on such a
  pronoun (*A woman walks in. Dan handcuffs her.*) are listed in `info` instead of being
  put on a declared character. *Mara, a slim woman* or *Dan is a man* describe a named
  character and do not count as an unnamed person.
- A character whose memory never says *woman*, *man*, *she* or *he* can still be the one
  *her* or *his* means, so *Dana drops her bag* stays Dana's shot and never brings in the
  only character known to be a woman. A *She …* or *Her …* sentence in a character's own
  memory lines gives their gender. *Her own wrists* are the wrists of whoever is doing the
  cuffing, and *her hoodie* is the hoodie of whoever memory says wears one.
- Everyone still in the scene is in the shot when the beat says *both of them*, *each
  other*, *the two of them*, *everyone*, *the others* or *they … together*, or names
  nobody and says *they*. *Her wrists crossed over each other* does not count.
- A beat that names nobody (*The lights flicker.*, or *She sighs.* with two women in the
  scene) keeps the people the shot before it showed.
- People stay in the scene until they leave: walking or heading out, off, away or home,
  or going upstairs, downstairs, to bed or into another room (*into the bathroom*),
  unless that room is where the scene paragraph sets the scene.
- When a continued shot opens on a frame with someone the beat leaves out, the prompt
  says this shot frames only the beat's people.
- Except: someone still in the room stays in view through beats about others when a
  later beat needs them where they are (*Sarah lies down on the table.* / *Dana opens the
  bag.* / *Dana takes off Sarah's diaper.*), so they never drop out and walk back in, which
  H3 draws as them getting up and coming over. When that later beat has them walk, come
  back, join or head over (*Lena joins her at the window*), or they leave first, the shots
  between frame only the beat's people as above. A beat that sets the camera's framing
  (*close-up*, *the camera frames her face*) always holds only its own people.
- If the node's files change after ComfyUI loads them, `info` starts with `RESTART COMFYUI`,
  because that run used the old code.
- In `info`, anyone in a shot whom the beat does not name says why, for example
  `cast: Sarah, Dana (from the shot before)`: *she/he*, *held*, *freed*, *everyone*,
  *from the shot before* or *in the scene*.
- The scene paragraph and the anchor describe people but do not put them in a shot.
- A beat with nobody in it (*The camera pans across the empty bathroom*) says that nobody
  is in the shot.
- After a `cut`, only the people that beat names are present.

For someone who is not present:

- their picture is left out
- their `character_memory` line is left out
- any scene sentence that is only about them is left out, along with the sentences
  after it that start with *she* or *he* and so are about them too. When only a side
  clause is about them (*A harbour at dawn, where the Aurora sits at anchor.*), just
  that clause goes and the place stays (*A harbour at dawn.*)

When everyone in a shot is a declared character, a shot with one or two people also
says how many are in it. The count is left out whenever someone else might be there:
a name that is not declared, or *a guard*, *the man*, *a crowd*. Words that only
describe someone (*a slim Asian woman*, *Blue eyes.*) do not count as another person.
Declare every character, with a `character_memory` line or a picture tag, so the
guarding covers them all.

A character does not have to be a person. One whose description first names a ship,
vehicle, planet, robot or animal (`Valiant <Picture 1>: a long white starship`, `Rex: a
big dog`) is treated as a thing: it is never counted as a person, and a shot with only
things in it says there are no people anywhere in it, with no talk of bodies, faces or
mouths, so a scene of ships stays a scene of ships. A person aboard still counts, and a
beat that mentions a crew, passengers or pilots brings them in as extras.

Extras are welcome: a beat can bring in *a woman*, *a crowd* or a name you did not
declare (a brand such as *Pampers* or *Nike shoes* is not a name). That shot then says
everyone other than your characters looks nothing like
them, with a different face, hair, body and clothes and none of what they wear, so an
extra is never drawn as a copy of someone in `character_memory`.

### Continuity

Every shot that starts on the previous shot's last frame says so in its prompt: the
same place and the same people, one moment earlier, with nobody new joining, or, when
the beat brings someone in, with that person coming into view and nobody else. When the
beat brings in extras, they come into view too. That
frame reaches H3 as a picture, and a picture the prompt does not mention is read as
another person.

After a `cut`, someone who was last seen alone is given that frame as their current
look, in place of their portrait. Their clothes and anything held on them carry over.

When a beat shows someone from behind (*the camera moves behind Crystal*, *her back to
the camera*, *she faces away from the camera*, *a rear view of Crystal*), the prompt
says they face away from the camera from head to feet, so their back and the back of
everything they wear show, and that their body never twists at the waist. That keeps a
belt or harness from being drawn back to front when the shot continues from a frame
that showed its front, without turning only half of them around. Say what the back of
an item looks like (*a chain hanging from the back*) so it can be drawn.

H3 comes out a little more saturated and contrasty each time it continues from a frame,
and left alone that burns the picture more with every shot. The first shot, and the
first shot after a `cut`, are left as rendered and set the look. Every shot that
continues from a frame is graded where it picks up: the first frame you see is matched
to the frame it continues from (brightness, contrast, saturation and the shape of the
shadows and highlights), since both show the same moment, and that same correction is
carried through the rest of the shot. Nothing later in a shot is forced onto another
frame's tones, so someone stepping in, the lights going down or the camera turning keep
their own look. A continued shot that opens washed out gets its colour back the same way.
A black frame, or a shot that does not pick up where the last one ended, is left as
rendered. H3 also creeps more saturated and contrasty within a shot, and each shot hands
its last frame on, so that creep would add up shot after shot. So when a shot ends more
than 3% more saturated or contrasty than the scene's look (its first shot, or the first
after a `cut`), the excess is eased out across the shot, from nothing at its first frame,
and it hands on a frame at the scene's look. A shot that gets darker or paler is never
touched. Each shot also starts from the last frame of the one before, decoded and encoded again,
and H3 takes everything in that frame as real: its small flaws are copied and built on,
shot after shot, until surfaces turn blotchy and crusty, like a picture saved as a JPEG
over and over. H3 also opens each continued shot a little rougher than the frame it was
given. So when a shot opens or ends with more than 3% more texture on its smooth surfaces
(skin, walls, cloth) than the scene's opening, the excess is eased out at that end and
blended across the shot in between, leaving edges and outlines as they are. Both ends of
every shot sit at the scene's texture, so shots join without a jump and the frame each one
hands on is back at the scene's texture.
Those corrections act on finished frames, so they do not stop H3 itself from drifting:
each continued shot comes out of H3 more saturated and contrasty than the frame it was
given. So the node measures that boost on every continued shot (averaged over the
run) and hands the next shot a frame toned down by it, which H3 then brings back up to the
true look. That frame is trimmed from the video. The log gives each shot's rendered
opening and rendered end, before any correction, next to its finished look, and its
texture against the scene before and after it is held. Each continued shot's numbers (how its opening came out against the frame it
continues from, whether it was graded, and its finished look against shot 1) go to the
ComfyUI log.

### Reference pictures

Wire up to four pictures into `ref_image_1` … `ref_image_4` and refer to them in the text
as `<Picture 1>` … `<Picture 4>`. A tag with no picture wired is removed.

A tag belongs to the name it sits beside, or else to the one name on its line or
sentence, so a `character_memory` line can carry it anywhere: `Aurora <Picture 1>:`,
`HMS Aurora - <Picture 1>:`, `<Picture 1>: HMS Aurora, …`, `HMS Aurora, a three-masted
frigate (<Picture 1>).` A memory line with a tag declares the one it is about, so its
picture goes only into the shots that one is in. Each shot's line in `info` lists the
pictures it sends and whose they are (`pictures: Aurora (ref_image_1)`).

Once a person is held, their own picture (written right after their name, as in
`Mara <Picture 1>`) is left out of shots that start on a frame they are already in. A
portrait without the cuffs or the tape pulls them back off. In the shot where they come
in, the picture still goes in. After a `cut` the picture comes back, unless a later frame
of them stands in for it.

The pictures work on every checkpoint and LoRA. The hybrid b25-49 (`fl2va_ref2va`) and any
checkpoint whose file name says `ref2va` were trained on them and get them in every shot.
FL2VA checkpoints, such as the base `minimax_h3_fl2va` and 10Eros, and FastH3 were not:
when a person is already in the frame a shot opens on and their picture comes in as well,
they draw that person twice. So on those, a person's picture is left out of a shot that
continues from a frame they are in, and that frame carries their look. The first shot,
the shots after a `cut`, and anyone walking into a continued shot still get their
picture. The node reads which checkpoint is loaded from its own `model` link.

The same goes for a LoRA trained without them, found in any loader on that link
(ComfyUI's own LoRA loader and LoRA Stack, a LoRA stack such as DaSiWa's, or any loader that
names its LoRA files;
a slot switched off or at strength 0 does not count). A LoRA whose metadata lists
`ref2va` was trained on pictures. One whose name or metadata says T2V, FL2V or a
few-step speed LoRA (*4step*, *turbo*, *lightning*, *distill*, DMAD, FlashGen) was not, and
`info` names it. Style LoRAs change nothing here.

A person's picture always goes in cropped to their face and hair (found with DWPose), so
their clothes come only from the words: the clothes in the picture never end up on top of
the ones the scene describes. `info` marks it *face and hair* under *pictures:*, and says
when no face could be found and the whole picture went in.

### Sound

- **Speech:** a shot with a quoted line (`"…"`, `'…'`, `“…”`, `‘…’` or `<d>…</d>`)
  speaks. Every line reaches H3 inside `<d>…</d>`, the speech tags it was trained on,
  with the words exactly as written, except that a garment that is off is left out of
  the line the picture is made from (see Clothing). Its first half second is held quiet
  so the line does not start on the cut.
- **Words are not acted out:** H3 draws what a line talks about unless told otherwise,
  so every shot with a line says *Spoken words are only heard: the shot shows what it
  describes, not what anyone talks about.* What the beat itself describes still happens.
  A beat that is only someone talking (*Dana says to Sarah: "Lie down."*) says *This shot
  is only the line being spoken: everyone stays where and as they are.* instead.
- **Room to speak:** a shot gets about 3.5 words a second, H3's own pace, plus 2.5 seconds
  (H3 starts talking about 2 seconds in) for its line, even past `shot_seconds`, up to
  H3's 15 seconds. No more: time left over after a line is filled with made-up talk or with
  acting the words out.
  A `seconds:` line still wins. A line too long for one shot is flagged in `info`: split
  it across beats.
- **No talking without a line:** every shot without a quoted line renders its picture
  with the audio held silent and says that nobody speaks and every mouth stays closed,
  so nobody's mouth moves to mumbling. The closed-mouth wording is left out when the beat
  has its own vocal sound (*screams*, *gasps*) or a gag holds a mouth open.
- **Foley and ambience:** such a shot keeps the sounds its beat brings (footsteps, a
  door, cuffs closing, tape tearing) and the ambience of the scene (rain, traffic, a
  quiet house). The node never adds a steady hum or whir (a fridge, a strip light, a
  computer fan) the scene does not name, since it reads as a drone. Its sound is made in a second, audio-only pass over the finished
  picture, so the sound follows what is on screen and has no moving mouths to put a
  voice to. That pass listens to a half-size copy of the picture, which makes it about
  six times cheaper than rendering the shot again, and the picture itself is never
  touched. Set `foley_resolution` to *full* for a full-size pass. A beat that describes
  its own sounds keeps those, and a scene paragraph that describes sounds (*Full battle
  sounds.*) keeps them in every shot, with the beat's own sounds added.
- **No music:** every shot says there is no background music, unless its beat or the
  scene mentions music, a radio, singing or an instrument.
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

**With the H3 Fun controlnet: the skeleton is held, on any H3 checkpoint and LoRA.** The
node loads `minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` from
`models/model_patches` by itself, or uses the one wired into `pose_controlnet`.

- Each such shot is first drafted at half size from the same text and seed. DWPose reads
  the draft, and the shot itself is then rendered at full size with the restrained
  person's skeleton held in place for the first `pose_end` of the steps. If the draft
  gives nothing to hold, the shot is rendered at full size without it.
- The draft costs about a tenth of a full render, so a checked shot costs about 1.1
  renders instead of 2. The movement follows the draft.
- The controlnet was trained on the hybrid b25-49 checkpoint and reads its 8-wide
  timestep curve. On any other checkpoint the node reads each step's timestep off that
  checkpoint and hands the controlnet the matching point on the hybrid's curve, so keep
  the hybrid checkpoint in `models/diffusion_models`; only its small curve is read.

**Without the controlnet: the shot is rendered again.** If the restraint broke, the
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
  - FastH3 was distilled for text and first-frame shots only, so it gets reference
    pictures only where nobody can be drawn twice (see Reference pictures).
- **Hyperflow**, detected from the LoRA's metadata or its file name in any loader (its
  settings are read from its own file when the loader does not pass them on), samples every shot
  on its own 8-step grid (shift 12/3, `euler`). Leave `sigmas` unwired.
  - Its endpoint conditioning is added back from `hyperflow_endpoint_v1.0.safetensors`
    beside the node, or from the original `minimax_h3_hyperflow_8step_v1.0.safetensors`
    in `models/loras`.
  - That needs a checkpoint with a full time embedder. On curve checkpoints (hybrid
    b25-49, 10Eros, FastH3) Hyperflow runs without it, and `info` says so.
- **Speed LoRAs** are sampled the way they were trained, whatever `sampler_name` says:
  - a LoRA whose metadata gives its sampling (DMAD: *4 steps, video shift 12, audio shift
    2, re-noise step rule*) gets exactly that, a re-noise rule as `lcm`;
  - FlashGen gets its published recipe: 4 `euler` steps on its own schedule
    `[1.0, 0.7, 0.4, 0.15, 0.0]` shifted at 12/3;
  - any other few-step LoRA (*4step*, *8step*, *4to8step*, *turbo*, *lightning*,
    *distill*, *lcm* in its name) gets `euler` on the steps you set, with no extra audio
    step: a multistep sampler such as `res_multistep` overshoots on them into halos,
    blown colour and clipped audio.
  - Wired `sigmas` override all of this. `info` says what each LoRA got.
- **PDD Acc** (`MiniMaxH3PDDAccApply` on the model) is sampled the way it was trained:
  on its own sigmas, with `euler` and shift 12/3, whatever `sampler_name` and the shift
  widgets say. Wire its `sigmas` output into `sigmas` and its `MODEL` output into `model`;
  without its sigmas the run stops and says so. `info` says *PDD Acc*.
- **A LoRA with timestep layers** (Hyperflow has them) on a curve checkpoint loads only
  in part: ComfyUI leaves those layers out. Near the top of `info`, *LORA ONLY PARTLY
  LOADED* names it; use it on a checkpoint with a full time embedder, such as the original
  FL2VA, or use a LoRA without timestep layers.

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
| `foley_resolution` | the size of the picture copy the sound pass listens to: *half* (about six times faster) or *full* |

## Outputs

| slot | what it is |
|---|---|
| `images` | the finished frames |
| `audio` | the synchronised soundtrack |
| `info` | what the node did for each shot, how long it took, and its contrast and colour against shot 1 |
| `script` | the exact text each shot was given |
| `frames_per_shot`, `total_frames`, `shots`, `video_seconds` | for downstream nodes |

## Other nodes here

- **H3 Shot Length**: one shot length as seconds and as a valid H3 frame count.
- **H3 Overlay**: watermark and intro title composited onto the finished frames.

## Notes

- No negative prompt and no cfg setting: H3 runs at cfg 1.
- Denoise is fixed at 1.0: partial denoise desyncs the joint audio/video schedule.
- A LoRA whose timestep layers were built for another checkpoint's time curve, such as
  one made for an 8-wide hybrid and used on 10Eros or the full FL2VA, cannot attach
  those layers as it is, and the sampler loses part of its timing, which shows as faded
  beats. The node re-fits them onto the loaded checkpoint's curve, which needs the
  checkpoint the LoRA was built for in `models/diffusion_models`; `info` says how many.
- ComfyUI sizes H3's working memory at a fraction of what a shot uses (a block needs
  about 105 KB per token), so on long shots it keeps too much of the model in VRAM and
  runs out. The node tells it 120 KB per token instead, so the rest of the model streams
  in as needed. If a shot still does not fit, the error says to lower `megapixels` or
  `shot_seconds`.
- Widgets are restored by position. If they read NaN, recreate the node as described
  under Install.

## Disclaimer

The owner of this repo will not be responsible for any copyright strikes incurred
because of use. You are responsible for your works. Use this node responsibly and
ethically.
