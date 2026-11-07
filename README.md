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
VAELoader ──┘                     info   ─> Preview as Text
```

`prompt` is an input socket — wire a multiline text node into it. Show `info` and
`script` with ComfyUI's own **Preview as Text** node (*utilities*), which refreshes on
every run. While a video renders, the node itself shows the plan and the shot it is on,
live.

Turn on **`plan_only`** to see the exact text every shot will get, without rendering.

## Writing the prompt

Every paragraph is a beat, and a beat is one shot unless it needs more time than one shot
holds (see *Real-life speed* below). Each shot starts on the last frame of the shot before,
so the scene runs on without a break.

- **The scene:** a first paragraph that names none of your characters and has no spoken
  line (*An office at night.*) is the scene, kept at the top of every shot. Text in
  `anchor` is kept at the top of every shot too, and with an anchor every paragraph is a
  shot.
- **Characters** go in `character_memory`: `Name: description`, `Name - description`,
  `**Name**: …`, `- Name: …`, or the name alone on a line with the description on the
  lines under it (`Hair: …`, `Outfit: …` lines included), up to a blank line. Any name
  works: *Zoë*, *Anne-Marie*, *O'Brien*, *McKenzie*, *Mary Jane*, *SARAH*.
- **Speech:** anything in quotes is spoken. Each line gets a stretch of sound of its own,
  about as long as the line takes to say, placed after the actions the beat puts before
  it; the sound before, between and after the lines is held silent, so nothing is babbled
  around them. A shot is lengthened to fit its lines. A shot without a line is made silent
  and gets its sound afterwards (see Sound).
- **Who says it:** every line goes to its speaker: the said-tag (*"…," she whispers*,
  *Dana says, "…"*), a name and a colon (*Dana: "…"*), or, with no tag, whoever acts right
  before the line; a line that carries on stays with the same speaker. *She* or *he* is the
  only woman or man in view, otherwise whoever just acted (also an unnamed person, see Who
  is in each shot), and a plain *says* never goes to
  someone whose mouth is taped or gagged. The prompt then writes the line as *black-haired
  Dana whispers: \<d>…\</d>* and tells H3 that only she speaks and nobody else's lips
  move. A line from someone taped or gagged becomes a muffled, closed-mouth *mmmph* with no
  words, in a short stretch of sound of its own (see Sound): a gag sound
  (*"Mmph!"*) or her own plea (*"Let me go!"*) is hers, a line spoken to her (*"Nobody can
  hear you"*) is the other one's. Whoever a line calls by name (*"Look at me, May"*) or a
  tag names as the listener (*she tells Mia*) is not the speaker, and an untagged answer to
  a question comes from the one asked. A quoted thought (*Sarah thinks*) is not spoken. A
  voice the description gives (*Voice: low and husky*, *a soft, breathy voice*) goes with
  every line that person says. `info` lists who says each line, and flags a line it cannot
  place: name who says it.
- **Real-life speed:** every shot is as long as what happens in it takes in real life,
  and never shorter than 5.2 seconds, the shortest clip H3 was trained on. Each action
  gets about 2.2 seconds (*Dana stands, walks to the door and opens it* is three), a
  speech tag (*she says*) gets none beyond its line, and everything that goes on or comes
  off gets its own real time: handcuffs about 5 seconds, a ball gag 5, a blindfold 4, rope
  8, tape over her mouth 3 and around her waist 8, a crotch chain 7, a little more when it
  is fastened to something; shoes 3, a top 4, a blouse or jeans 6, a bra 4, panties 3.
  Walking, climbing stairs or hopping with her ankles bound takes about 6 seconds, since
  she can only take tiny or hobbled steps.
  *Dana strips Sarah naked* takes as long as everything she has on. A clause that only
  finishes the restraint before it (*...and locks it*, *knotting it twice*, *and pulls
  them off*) is part of that restraint's time, not an action of its own. A beat that needs
  more than `shot_seconds` runs on into the next shot instead of being rushed: it is split
  between sentences, and a sentence that holds too much on its own (*strips her naked*,
  *cuffs her, gags her and blindfolds her*) is split garment by garment or restraint by
  restraint. A crotch chain is never split: it always goes on whole in one shot. Each
  earlier part gets a short sentence saying only what happens in it (*Dana takes off
  Sarah's black heels, white blouse and black skirt.*) and keeps everyone the beat brings
  in, unnamed people too (*Two masked men…*), the last part keeps your sentence as written
  minus the restraints an earlier part already put on or took off (*Dana cuffs Sarah's
  hands behind her back and puts a crotch chain on her, locking it to her handcuffs*
  becomes *Dana puts a crotch chain on Sarah, locking it to her handcuffs*), and every line
  keeps the speaker the whole beat gives it. `info` marks these shots *beat 4, part 1 of
  2*. A beat up to about 15% over `shot_seconds` stays one shot.
  A shot that still has more to do than time for (one long sentence of plain actions, or
  a `seconds:` line that is too short) is flagged in `info`.
- **`cut`** on its own line starts the next shot fresh instead of on the last frame.
- **`seconds: 12`** sets one shot's length and keeps the beat in one shot.

### Restraints and clothes

The node reads what goes on, comes off or is pulled out of the way from your beats, and
carries it from shot to shot. It reads a change only when the sentence makes clear who it
happens to:

- a name on the body part or garment: *Dana cuffs Sarah's hands behind her back*,
  *Dana straps Sarah's wrists to the bed frame*, *Dana pulls Sarah's pajama bottoms off*;
- a name after the verb: *Dana cuffs Sarah*, *snaps the handcuffs on Sarah*, *cuffs
  herself to the bed frame*;
- *his* or *her* when only one man or one woman is in view;
- *their* after two names, or two names on one body part, on both of them (*Sarah and Dana
  sit on the floor, their ankles handcuffed together*, *tapes Mia's and Zoë's mouths
  shut*);
- an *as* or *while* part with its own name has its own doer, whatever the first part
  does (*Sarah kicks and struggles as Dana handcuffs her ankles together* has Dana cuff
  Sarah's ankles, and *Luis kicks his legs as Dana pulls his jeans off* has Dana take them
  off);
- otherwise a bare *her* goes to the one not doing it (*Dana tapes her mouth shut* tapes
  Sarah), and a name something is done to never becomes the one doing the next thing
  (*Dana strips Sarah naked and chains her waist* chains Sarah, with Dana doing it, and so
  does *Dana leaves Sarah tied to the chair and gags her*); *she* means the only woman in
  view, or, with more than one, the one acting rather than the one watching or
  struggling, whichever name came last (*Dana watches as Sarah struggles. She gags her.*
  and *Dana pins Sarah down while Sarah struggles, and gags her* both have Dana gag
  Sarah); a sentence that starts with *she* and names one of two people is about the
  other (*She shoves the panties into Sarah's mouth* is Dana), and so is a *she* after an
  opening *-ing* part with *her* in it (*Sarah sobs. Ignoring her, she ties the rope.* is
  Dana); a *she* who falls, screams or struggles right after someone acts on her is the one
  acted on, unless the one struggling was already her (*Dana pushes Sarah down. She falls,
  handcuffed.* cuffs Sarah); each *, and Lena…* or *,
  and she…* has its own doer (*Mia opens her mouth, and Lena pushes the gag in* is Lena;
  *a stranger hands his coat to Chloe, and she puts it on* puts it on Chloe); a
  restraint told without a doer (*her mouth is sealed with tape, and O'Brien rubs it
  down*) is put on by whoever that sentence has acting, goes on in the shot when it
  happens *a moment later*, and is already on her when nobody else is named and her own
  wrists are bound; more tape on *her* goes to whoever is already restrained, and someone
  whose wrists are bound cannot put anything on anyone;
- a garment goes to the one whose description has it (*Dana pulls her tank top up*
  lifts Sarah's when only Sarah wears a tank top), and one worn by someone unnamed (*next
  to a stranger in a torn shirt*) is nobody's in the cast;
- *it*, *them*, *the other* and *does the same* carry on with the last restraint or
  garment, and a garment or cloth from the beat before still counts when it goes into or
  over a mouth (*Ethan peels off his socks.* then *He jams them into Zoe's mouth.*); a
  torn-off *strip* or *piece* is tape;
- *bound and gagged*, *is cuffed*, *her taped wrists*, *Sarah sits on the bed, handcuffed*,
  *walks Luis down the corridor in handcuffs*, *shuffles in wearing leg irons*, *Sarah,
  shackled at the ankles, ...*, *bound hand and foot*, *Sarah wears a crotch chain* and
  *with her panties stuffed in her mouth* say it is already on, on the one the sentence is
  about, and so does a sentence about the restraint itself (*The chain around Sarah's waist
  runs between her legs and up to her handcuffs*) or one that says it is on with nobody
  doing it (*Sarah's waist is wrapped in chains*);
- a list of restraints puts each one where it goes (*puts Sarah in a belly chain, handcuffs
  and leg irons*: a chain around her waist, handcuffs behind her back, leg irons on her
  ankles), and a spot named after a comma takes the restraint before it (*zip-ties her
  wrists, then her ankles*).

Once the node knows who a *her* or *his* means, the beat H3 gets says the name (*Dana
pulls off Sarah's panties and stuffs them into her mouth* reaches H3 as *... stuffs them
into Sarah's mouth*), so H3 does not have to guess whose mouth or wrists *her* means.

Body spots include the crotch: *between her legs* or her crotch by any name. Tape around
her waist and between her legs is two things on her (*duct tape around her waist*,
*duct tape sealed over her crotch, between her legs*), and both stay on.

**Crotch chain.** A crotch chain has two parts: a waist chain locked tight around her
waist like a belt, and a center chain locked to the front of the waist chain below her
navel and to the back of it at the small of her back, running under her crotch, pulled
tight over her vagina, and up between her buttocks, so all of it stays at her hips. Name
it (*Dana puts a crotch chain on Sarah*, *crotch-chains her*, *Sarah wears a crotch
chain*) or tell it as a chain around her waist and between her legs (*wraps chains around
Sarah's waist and between her legs*, *runs a chain between her legs*, *chains her waist
and crotch*): either way it is one thing on her (*a crotch chain locked tight around her
waist, its center chain pulled tight under her crotch and over her vagina from the front
of her waist to the small of her back*), and every shot she wears it in says what it is,
that all of it stays at her hips, and that the center chain stays taut and centered and
never hangs loose, slips to one side or sags. It goes on in one step, in one shot: the one
putting it on takes a separate crotch chain from their pocket, wraps its waist chain tight
around her waist, runs its center chain from the front of the waist chain under her crotch,
centers it tight over her vagina, brings it up between her buttocks to the small of her
back and padlocks it to the back of the waist chain. Someone kneeling at her feet stands up
behind her first, and with her ankles bound the step ends with them staying just as they
are (*Marcus stands up behind Sarah, takes a separate crotch chain from his pocket, ...,
while her handcuffed ankles stay just as they are*). In test renders, a shot that opened
with his hands still on her ankle cuffs had him take the chain off her ankles and pull it
up her legs as the crotch chain, so whoever kneels or crouches to put something on lets go
and stands back up before the shot ends (see *Bound ankles*). Its center chain is never said
to run *down between her legs*: with her ankles cuffed, H3 took the short chain between her
ankle cuffs for it. A man's goes over his
crotch. Rope around her waist and between her legs is a crotch rope, put on the same way
and tied off to the back of the waist rope in a knot. A chain already around her waist
becomes a crotch chain when a later beat runs it between her legs: a center chain is
hooked to the front of it, run under her crotch and padlocked to the back. Taking it off
unlocks the padlock, pulls the center chain out from under her crotch and takes it off her
waist.

Locked to her handcuffs (*puts a crotch chain on Sarah and locks it to her handcuffs
behind her back*, *wraps chains around her waist and between her legs and chains them to
her handcuffs*, *padlocks her handcuffs to the back of the crotch chain*, *chaining her
cuffed hands to it*, *ties it to her wrists* for rope), it is linked to them by a separate
short chain of its own (a rope for rope), padlocked from the chain between her handcuffs
to the back of the crotch chain once the crotch chain is on, and the beat itself says so
(*... locks it to her handcuffs behind her back with a separate short chain.*). The crotch
chain is never passed through her handcuffs, and the chain between her cuffs is never part
of it. The shot says her cuffed hands are held to the back of the crotch chain by that
separate short chain; with her hands cuffed in front, it goes to the front of the crotch
chain and her elbows bend. The cuffs it is linked to are never a second pair, and if
nothing has cuffed her yet, they are on her from the start. Cuffs put on to a crotch chain
already on her (*cuffs her hands behind her back to the crotch chain*) close first, then
the short chain is padlocked on; linking them later (*Dana locks the crotch chain to her
handcuffs*, *padlocks her handcuffs to the back of her crotch chain*) does just that.
Taking the crotch chain off or uncuffing her takes the short chain away first, and
*unlocks the crotch chain from her handcuffs* takes away only the short chain. A chain
around her waist alone (*wraps a chain around her waist and locks it to her handcuffs*,
*padlocks her handcuffs to the back of the waist chain*) is linked to her cuffs the same
way.

**Chastity belt.** A chastity belt (*Dana locks a chastity belt on Sarah*, *fastens a
chastity belt around her waist*, *Sarah, locked in a chastity belt, ...*, *wears a chastity
belt*) is locked on until a beat unlocks it (*Dana unlocks Sarah's chastity belt and takes
it off*). Every shot that starts with it on says so in her description line, before the
beat (*Her chastity belt is locked on her, its band around her waist and its shield between
her legs over her crotch.*), and no line about her being naked leaves it out. Its picture,
when you give one, goes in every shot from the one it goes on in: put the picture's tag
next to the words anywhere, in a beat (*Dana locks the chastity belt <Picture 3> on
Sarah*), in `anchor` or in `character_memory` (*<Picture 3> is a steel chastity belt.*);
her own picture is never taken for it. Panties or tights under it stay on until it comes
off (*Dana pulls off Sarah's panties* becomes *Dana leaves Sarah's panties on under her
chastity belt*, and *strips her naked* leaves them on under it), and clothes over it come
off as usual while it stays locked.

**Bound ankles.** Handcuffs on her ankles (*cuffs her ankles*, *cuffs her feet together*,
*ankle cuffs*, *leg cuffs*), leg irons (*leg irons*, *shackles*, *ankle chains*, *ankle
shackles*, *fetters*, *a hobble chain*, *in irons*), a chain, rope, tape, a zip tie or
straps around her ankles, knees or legs, and a spreader bar all hold her ankles. Every
shot that starts with them on says so in her description line, before the beat (*Her
ankles are handcuffed together.*), so H3 reads whatever the beat has her do with her ankles
already bound. The shot then says what holds them, where, and how far apart her feet can
be: *a pair of rigid steel cuffs is locked around them just above her feet, joined by
their own short chain of a few links that never stretches, bends or gets longer, so her
ankles stay a few centimetres apart at most and her legs stay pressed together and always
move together, both at once*; leg irons are a pair of steel shackles just above her feet
with their own heavy chain, about forty centimetres long, that never stretches; rope, tape,
zip ties or a chain are tied, wound or wrapped tight around them just above her feet (or
around her knees). Saying the cuffs as solid steel down at her feet keeps H3 from turning
them into a crotch chain she also wears: described only as a chain that runs between her
ankles, they melted into it. A spreader bar holds her feet about
sixty centimetres apart. Putting them on holds her
ankles together first (*holds her ankles pressed together, both feet resting where they
are, and closes one cuff around each ankle with a click, the short chain between them, then
lets go and stands back up*), so the next shot does not open with hands on the cuffs.
*Unties her legs* or *frees her feet* takes off whatever holds her ankles or knees, and
uncuffing her wrists takes off only the cuffs on her wrists. Clothes on her legs stay on,
pulled down to just above whatever binds them (see *Clothes against restraints*).

**Separate restraints.** Every restraint keeps a chain of its own: the handcuffs on her
wrists and the cuffs on her ankles are each joined by their own short chain, and a crotch
chain has its own. When she is restrained in more than one place (her wrists, her ankles,
a crotch chain or a chain around her waist, a chastity belt), every shot says they are
separate pieces with nothing joining one to another (*The handcuffs on Sarah's wrists, the
handcuffs on her ankles and her crotch chain are separate restraints, each on its own with
nothing joining it to another.*), to keep H3 from adding a chain from her cuffs to her
ankles or to her crotch chain. A link a beat makes is named as the one separate short
chain between them (*... and the only link between any of them is a separate short chain
from the chain between her handcuffs to the back of her crotch chain.*). A join the node
does not draw (*locks her collar to her handcuffs*) leaves the line out from that beat on,
and so does a join in `anchor` or `character_memory`.

**Hogtie.** Her ankles linked to her wrists (*padlocks Sarah's ankle cuffs to her
handcuffs*, *chains her ankles to her wrists*, *ties her wrists to her ankles*, *padlocks
the chain between her ankle cuffs to her handcuffs*, *hogties her*, *hogties her with
rope*, *Sarah lies on the floor, hogtied*) is a separate short chain (a rope for rope)
from the chain between her handcuffs to the chain between her ankle cuffs, and both cuffs
keep their own chains. Whatever her wrists or ankles still lack goes on first, behind her
back (rope for a plain *hogtied*). The beat says how (*Dana padlocks Sarah's ankle cuffs
to her handcuffs with a separate short chain.*, *Dana hogties Sarah, her ankles held to
her handcuffs behind her back by a separate short chain.*), the one doing it bends her
knees and draws her feet up behind her toward her hands before padlocking the short chain
on, and every shot after says so in her description line and in the shot (*A separate
short chain runs from the chain between her ankle cuffs to the chain between her
handcuffs, so her knees stay bent and her feet stay drawn up behind her, close to her
hands, the whole time.*). *Unlocks the chain between her wrists and ankles*, *unhooks her
ankle cuffs from her handcuffs* or *unties the hogtie* takes away only the short chain;
uncuffing her wrists or her ankles takes it away first. A later beat that only mentions
her ankle cuffs leaves the link on.

**Legs, as written.** Legs that are free stay exactly as the beat says them, and a bound
pair is said once, as one pair. Pair words the node used to add (*both her legs*, *one in
each hand*, *one cuff on each ankle*, *her feet side by side*, *her left leg and her right
leg*) had H3 draw a second, mirrored pair of legs, so none of them is ever added, bound or
not: *Sarah kicks her legs* and *Luis grabs her ankles and lifts her legs up* go to H3 as
written. *Her legs* on a bound woman gets her name when it could be either of two women.

Whenever the beat moves her bound legs, the beat itself says how, right where it happens,
so H3 reads the movement and the restraint together. Her legs that move get *together*
right after them, and the end of that part of the sentence says what holds them: *Sarah
lifts her legs together into the air, her ankles handcuffed together.*; *Luis grabs her by
her ankles, her ankles handcuffed together, and lifts her legs up together.* A beat that
moves her without naming her legs (*Sarah struggles and kicks*) gets the same ending
(*Sarah struggles, her ankles handcuffed together, and kicks.*). One leg, foot, ankle or
knee of a bound pair becomes the pair (*lifts one leg* becomes *lifts her legs*, *grabs her
by the ankle* becomes *grabs her by her ankles*), since neither can move without the other,
and info lists each change. Toes and heels are left alone, since heels are also shoes.
That covers anything she does with her legs, knees, feet or ankles (*raises her legs*,
*moves both ankles up and down*, *pulls her knees up to her chest*, *kicks*, *struggles*,
*squirms*, *crawls*), anything someone else does with them (*Dana grabs her ankles*,
*drags her by the feet*, *lifts her legs onto the bed*), and getting up, sitting up,
rolling over, curling up or going down on her knees. So does any change from how she was
at the end of the beat before: standing there and then *lies on the floor* (also *lays*,
*laid*, *is laying*, *lay face down*, *fell onto the floor*, *Dana lays her on the
floor*, *sits her on the couch*) is her going down, and the shot says how she lies once
she is down. Two people bound alike in one
sentence each get their own, by name (*Sarah and Dana kick their legs together, Sarah's
ankles handcuffed together, Dana's ankles handcuffed together.*). Walking, or being led or marched, she goes *in tiny
shuffling steps, her ankles handcuffed together* (*in short, hobbled steps* in leg irons,
*waddling with both legs held wide apart* with a spreader bar); on stairs, *one stair at a
time, both feet together on each stair*. Leg irons let her feet move apart as far as the
chain reaches, so one leg alone stays as written, and they get *her ankles joined by the
heavy chain of the shackles*. Words that only describe her
bound ankles (*lies on the floor with her ankles handcuffed together*, *sits with her
ankles zip-tied*, *in leg irons*) are left as written. All of it is said as what H3 should
show: a line like *never one leg alone* would put *one leg alone* in front of H3. For the
same reason nobody else in the shot is said to have free legs: H3 ties words to names only
loosely, and *legs are free* sat right beside her bound legs. Only while her legs stay
still does the shot say how she kneels, sits, lies or stands with them (*She lies with
her legs together.*): that pose next to a beat that moves her legs
had H3 move one leg and leave the other lying still.

Putting them on, the one doing it crouches at her feet (kneels, if she is lying or the beat
says so; behind her if she is kneeling; bends down if she does it herself), holds her
ankles pressed together, both feet resting where they are, and only then puts the
restraint on, so neither leg is lifted to the one doing it; taking them off starts the
same way. Cuffs put on while she
fights it (*Sarah kicks and struggles as Dana handcuffs her ankles together*) go on all the
same. Fastened to something, she stays there: ankles cuffed to the legs of a chair seat her
on it with her knees bent, and chained to a post or a ring in the floor her feet stay at
it. One ankle chained to something (*chains her left ankle to the bed frame*) leaves her
other leg free, and she can move as far as the chain reaches. A beat that needs her legs
apart while they are bound (*crosses her legs*, *spreads her legs*, *straddles the chair*,
*steps over it*, *runs for the stairs*), or has her walk while they are fastened to
something, is flagged in `info`: H3 would break the restraint or draw her an extra leg.
*Tries to run* is not flagged.

When tape goes on, the shot says who holds the roll and how the tape goes on: around her
waist or hips the roll is carried all the way round her body, passed from one hand to the
other behind her back; between her legs the tape runs from the front of her waist down and
up to the back; around wrists or ankles it is wound round them, then taped to whatever they
are fastened to (*Rose tapes Will to his office chair*); over her mouth a strip is
torn off and pressed flat. The roll stays whole in that person's hand and outside her body,
and is torn off and put down once the tape is on, so H3 never pushes it into her.

Everything that goes on or comes off is spelled out step by step, in the order the beat
gives, by the person doing it and the way it really happens: *brings Sarah's arms behind
her back and closes one cuff around her left wrist with a click, then the other around her
right wrist*; *crouches at her feet, holds her ankles pressed together, both feet resting
where they are, and wraps the rope several times around both of them together, cinches it
between them and ties it off in a firm knot*; *pushes the
ball into her mouth behind her teeth, pulls the strap around behind her head and buckles
it*; *lays Sarah down on her back on the table, lays her arms flat on it at her sides and
chains her wrists to the table*; *wraps the rope around her waist and ties it off in a firm
knot*; *unbuttons her white blouse from the top down and slides it off her shoulders and
down her arms*; *opens her jeans at the waist and pulls them down her legs and off over her
feet*; *unlocks the handcuffs with a small key and takes them off her wrists*. Clothes
taken off with no order given (*strips her naked*, *strips her down to her panties*,
*undresses her*, *removes her clothes*, *takes her clothes off*) come off in a real order:
jacket, shoes, top, skirt or trousers, stockings, bra, panties; *rips her clothes off*
tears them off and takes her shoes off whole. The
shot then says it all happens at natural, real-life speed, one step after another, over
about so many seconds, with no step skipped, sped up or cut short, and no restraint or
piece of clothing appearing, vanishing or changing by itself, so H3 is never left to guess
how a garment comes off or a restraint goes on. A shot where nothing goes on or comes off
says that everything happens at natural, real-life speed. A change from a tag line names
nobody doing it, so it gets that speed line but no steps.

Something comes off only when the sentence says so about that thing: *peels the tape off
Sarah's mouth* takes off the mouth tape and nothing else, *uncuffs her* or *frees her
ankles* take off what is there, and a gag or blindfold pulled down around her neck, pushed
up onto her forehead or spat out is off. *Dana peels the tape off* when Sarah has tape in
more than one place is flagged in `info` and nothing comes off: name the spot. Unlocking
*one cuff* leaves her cuffed; *re-cuffs her hands in front* or *cuffs her to the headboard
instead* moves the cuffs.

Underwear, socks, a rag or any cloth stuffed into her mouth is a gag made of that thing
(*black panties stuffed in her mouth as a gag*), and if someone was wearing it, it comes
off them. The shot says the one doing it pushes it into her mouth with their fingers and
that their own mouth stays empty. A scarf or bandana tied over her mouth or eyes is a gag
or blindfold made of it. Pulling it *out of her mouth* takes the gag off.

Bound wrists go where the words on the restraint put them (*cuffs her hands in front of
her*, *between her legs*, *at her waist*), where restraints she already has hold them, and
behind her back when nothing says; where someone stands does not count (*Dana stands in
front of Sarah and cuffs her wrists* puts them behind her back). A place for her hands is
never a second restraint: *cuffs Sarah's hands between her legs* brings her hands down
between her legs and cuffs her wrists, with nothing on her crotch, and *at her waist* puts
nothing around her waist. Handcuffs on *her arms* go on her wrists, and hands someone uses
(*Dana uses both hands to pull the chain between her legs*) are that person's own. A beat
where someone whose wrists are bound behind her back uses her hands (*Sarah wipes her
eyes*) is flagged in `info`: H3 would bring her hands forward to do it.

A restraint fastened to something stays fastened to it: *chained to the table*, *chained
down on a table*, *strapped to the gurney*, *her wrists chained to it*, *tied to the
bedposts*, *chained to its corners*, *tied spread-eagle on the bed*. On a table, bed,
bench, cot or gurney she lies on her back on it (or on her front, *face down*), her arms
down on it at her sides unless the beat says *above her head* or *spread* (*corners* and
*bedposts* spread them), and the shot says the chains run straight down to it and her arms
and legs stay down on it and never rise into the air. She sits instead when the story so
far has her sitting at it (*Sarah sits at the table, her wrists chained to it* keeps her
forearms on the tabletop), or standing by it. Tied to a chair or armchair, she sits on it
with her wrists behind its back; tied to its arms (*ties her wrists to the arms of the
chair*, *cuffs him to its wooden arm*), she sits on it with her forearms resting along
them and her hands down on them. The arm of a chair is never her arm. On the floor her
hands stay down at the floor. Her arms go up only when the beat hangs her from above
(*hangs from the ceiling*, *chained to the beam*). Lying face down on a bed in handcuffs
(*Sarah lies face down on the bed, her wrists handcuffed behind her back*) is not being
fastened to it; *chained face down to the bed* is. Somewhere she walks, hops or is led to
is never something she is fastened to (*walks to the post in handcuffs*, *hops to the door
with her ankles cuffed*), and parts chained *together* are chained to each other, not to
what she lies on (*Sarah lies on the floor, her ankles chained together*). A closing
*chained down* or *tied down* (*Sarah lies on the bed, chained down.*) fastens her to what
she lies on.

A garment comes off when it is taken *off*, *over her head*, *out of*, or ends *on the
floor*; it is *torn off* when it is cut, ripped or torn off, which works past cuffs. It is
pulled out of the way and stays on when it goes *up*, *down*, *aside*, *open* or *off her
shoulders*, and goes back on with *back on*, *back down*, *closed* or *into place*; one put
*back in the drawer*, *on the chair* or thrown at her stays off. *Strips her down to her
thong* takes off everything but the thong. The garment is named the way the description
names it (*tank top* in the beat becomes *white tank top*), and *pants*, *top*, *shirt* or
*underwear* mean the one garment of that kind she has on (*pants* takes off her jeans, *top*
her t-shirt and not the jacket over it). A garment her description does not have is
flagged in `info` and nothing is said about it in the prompt.

Nothing changes for a struggle, a touch, a look or a try (*Sarah tries to pull free of the
cuffs*, *Dana runs a finger over the tape*), for gear that is picked up, put down or put
back (*takes the handcuffs off the nightstand*, *tears a strip off the roll*), or for a
person who is moved (*pushes her off the bed*, *lays her back down*, *sits up in her
pajamas*). Every change read is listed in that shot's `info` line with the sentence it came
from. A sentence it cannot place is named in `info` with the reason, and nothing is
carried from it: name the person or the spot, or add a tag line.

### Tag lines

A tag line in a beat replaces what the beat would be read as. It goes under the beat (or
at the end of its last sentence), names the person first, and puts several things in one
line with `;`:

```
Dana grabs a roll of duct tape. She wraps it tightly around Sarah's mouth.
hold: Sarah, duct tape over her mouth

Dana lifts Sarah's shirt.
lift: Sarah, shirt

Dana pulls down Sarah's leggings.
remove: Sarah, leggings; thong

Dana peels the tape off.
release: Sarah, duct tape

Dana walks out of the room.
exit: Dana
```

| tag | what it does |
|---|---|
| `hold: Name, item` | the item goes on during this shot and stays on until a `release:` |
| `release: Name, item` | the item comes off during this shot; `release: Name` takes everything off |
| `remove: Name, garment` | the garment comes off during this shot and is gone from the description after it. List several with commas or *and*; `bottoms`, `top` and `underwear` take off what the description has there; `remove: Name`, `clothes` or `all her clothes` take everything off |
| `wear: Name, garment` | the garment goes on during this shot and stays on; for a garment that was lifted, it is pulled back into place |
| `lift: Name, garment` | the garment is pulled out of the way during this shot and stays on, pulled, until a `wear:` or `remove:`. By default a top goes up above her breasts, bottoms go down to her knees, a skirt or dress goes up to her waist and a jacket or robe falls open; say where to change it (`lift: Sarah, pajama bottoms to her ankles`). `pull:` is the same tag |
| `tear: Name, garment` | like `remove:`, but cut or torn off, so it comes off even past cuffed wrists or bound ankles |
| `exit: Name` | the person leaves during this shot and is out of the shots after it |

A `hold:`, `wear:`, `remove:`, `tear:` or `lift:` line in `character_memory` is on from the first shot.

**Layers:** underwear that another garment covers (a thong under pajama bottoms, a bra
under a shirt) is left out of the description, so H3 does not draw it showing through or
peeking out. It shows once what covers it comes off or is lifted, and that shot says so
(*Under them, Sarah has a black thong on, now in plain view.*). A sheer, see-through or
open garment hides nothing.

**Clothes against restraints:** a top, bra or dress cannot come off past cuffed or tied
wrists, and jeans, a skirt, panties, tights or any other bottoms cannot come off past
whatever binds her ankles, knees or thighs. Removing one of these anyway pulls it out of
the way instead, still on her and with the restraint locked, and `info` says so: up above
her breasts, or down to just above what binds her legs (*down to just above the handcuffs
on her ankles*, *down to just above the rope around her knees*), so the restraint stays
in plain view. The beat itself says so too, so H3 never reads that the garment comes off:
*Luis pulls off her jeans* becomes *Luis pulls her jeans down to just above the handcuffs
on her ankles*, *takes off her sneakers and her jeans* becomes *takes off her sneakers and
pulls her jeans down to just above the handcuffs on her ankles*, and *strips her naked*,
*undresses her* or *removes her clothes* gets *, her blue jeans and black panties pulled
down to just above the handcuffs on her ankles, where they stay*. This counts restraints
put on earlier in the same beat (*cuffs her ankles together and pulls her jeans off*). A
garment pulled down past what binds her legs (*pulls her jeans down to her ankles* with
her knees tied) stops just above it the same way. With her ankles bound, her shoes slip
off with her feet staying together, never one foot at a time,
and a dress comes off over her head. Release the restraint in an earlier beat to take
the garment off, or have it cut or torn off (`tear:`), which needs no release. Whatever
can come off still does, and the shot says the restraints stay locked while it comes off.

With her wrists cuffed or tied, every change to her clothes says in the beat itself that
her hands stay where they are (*Dana pulls Sarah's sweatpants down and off, with Sarah's
hands still locked together behind her back.*), and so do the steps. Standing, she steps
out of her bottoms one foot after the other with her hands still locked, so H3 never frees
a hand to keep her balance (*pulls them down her legs to her ankles, and Sarah steps out
of them one foot at a time, with Sarah's hands still locked
together behind her back*); lying or sitting, they come off over her feet. Taking off her
own bottoms she pushes them down over her hips and wriggles out of them, and her shoes she
pushes off with her feet, never with her hands. A beat that starts with *it* or *them*
for the garment the beat before ended on (*Dana pulls them back up.*) is read as that
garment.

Every shot then tells H3, in plain words:

- what each person has on and that it stays on (*Sarah: handcuffs behind her back; duct
  tape over her mouth. All of it stays on, exactly as described, for the whole shot.*),
  what goes on or comes off by the last frame and who has let go (*Dana and Rose
  have let go*; nobody lets go of what she put on herself), and that the others are free
  where she is not, except for their legs (*Dana's mouth is empty and uncovered and her
  hands are free.*);
  nothing behind her back is asked to be *in plain view*;
- each step of what goes on or comes off, by the person doing it, at natural, real-life
  speed (see above); who holds the roll when tape goes on, how it goes around her, and
  that the roll stays whole and outside her body; that someone who pushes a gag into her
  mouth keeps their own mouth empty;
- that someone with a taped or gagged mouth cannot speak;
- where every bound limb is, at its full length, from every side the camera can see it:
  behind her back, her arms run from her shoulders down behind her to her hands at the
  small of her back, hidden by her body from the front and in plain view, whole and locked
  together, from the side or behind, whatever she does in the shot; lying on her back, she
  lies on her arms, her locked hands hidden under the small of her back (told she was in
  plain view from the side, a woman lying on her back got her arms laid free at her sides
  in a test render); bound ankles say what
  holds them, how far apart her feet can be and how she walks, kneels, sits, lies, stands
  or struggles with them (see *Bound ankles*); fastened to a table or bed, she lies on it
  with her arms and legs down on it, as above; all of it holds while clothes come off or
  are lifted;
- for everyone who is bound, that wherever the camera looks she is whole and in natural
  proportion: head, shoulders, both arms and hands, torso, hips, both legs and feet, each
  at its natural size, joined to her body the natural way, so a camera that moves to her
  legs or comes round behind her finds them where they belong;
- that handcuffs and chains are rigid steel: the short chain between the cuffs and every
  chain link keep their length and never stretch;
- that a lifted garment stays on, pulled where it is, and what that leaves bare;
- what is left once a garment is off (*Below the waist, she wears only a black thong.*);
- *naked from the waist down*, *from the waist up* or *completely naked* once the
  garments her description gave for that part of the body are all off, with any tape or
  rope still on that part named (*apart from the duct tape sealed over her crotch*).

When two or more people without a picture are in view, these lines tell them apart by
the hair their description gives (*Mara: long red hair* makes *red-haired Mara*).

`info` lists, for every shot, who is in it and what is on, goes on and comes off, so a
missing or misspelled tag shows up before you render (turn on `plan_only`). A garment
that is not in the person's description (*pants* when it says *leggings*) is flagged,
since only the description's own word can be taken out of it. A tag that
names someone with no character line is applied as written and flagged in `info`.

### Who is in each shot

A character is in the scene from the first beat that names them (outside of quotes)
until their `exit:` line. Everyone in the scene stays in view: every shot tells the
camera to keep their faces in frame, since a face that leaves the frame is drawn anew
when it comes back. Bound ankles leave the frame the same way, most of all when she lies
on the floor, and come back without their restraint, so anyone whose ankles, knees or
legs are bound in the shot is kept in frame down to them (*The camera keeps the faces of
Sarah and Dana in frame for the whole shot, and stays far enough back to keep all of
Sarah in frame too, down to her handcuffed ankles.*). A beat that directs the camera
itself (*close-up*, *the camera frames her face*, *we see her legs*, *a shot of her
feet*) gets no camera line. After a `cut`, a shot holds the people its beat names. Every
shot also says how many people it holds, one body each.

**Unnamed people** (*a man*, *the guard*, *two women*, *a crowd*, *someone*) are kept
apart from your named characters. In a shot whose beat names them, they are counted in
the head count (*There are three people in the shot: Sarah, Dana and the man*), and the
shot says each of them is a different person who moves only as the beat describes, in
their own way and at their own pace, never copies the named characters' movements,
actions or words, and that nothing done to one side is done to the other. A later beat
that does not name them says nothing about them (nobody is added to or taken out of the
picture), and *the man* there still means the same man; he is gone once he leaves (*the
man walks out*, *a stranger walks past*) or after a `cut`. *The woman* or *the guard*
means a named character when a named character's description fits (*Dana: a prison
guard*), unless the beat describes someone new (*the man in the cowboy hat*).

**He and she are your named characters.** Wherever *she*, *he*, *her*, *him* or *his* in
a beat can only mean one of your named characters, the beat H3 gets says the name, so H3
never draws a second copy of a character for the pronoun: *Sarah walks in. She sits
down.* reaches H3 as *Sarah walks in. Sarah sits down.*, and *He takes her hand* as *Jake
takes Sarah's hand*. A *she* is the woman the sentence before was about (*Sarah grabs
Dana. She pushes her away.* is Sarah pushing Dana), or the one acted on when she only
reacts (*Dana pushes Sarah onto the bed. She screams.* is Sarah); *her* right after her
own name stays *her* (*Sarah brushes her hair*). A pronoun that could mean either of two
people (*Sarah and Dana sit. She laughs.*) or an unnamed person (*A man walks in. He
sits.*) is left as you wrote it: name the person to be sure.

**Nobody is brought in who is not there.** An unnamed person seen on a screen (*a man on
her computer screen*, *a woman on the news*, *Sarah is on a video call with a man*), heard
on a phone or off screen (*a man on the phone*, *a man's voice calls from the hallway*), or
shown in a photo or poster is not counted and is said to be only on the screen, only a
voice or only a picture, never in the room, a different person from your named characters
with a face of their own; the shot also says your named characters are in the room, never
on that screen. Your named characters are always in the room. The man on the screen stays
on it in the next beats (*the man laughs*) until the call ends (*she hangs up*, *closes
the laptop*, *turns off the TV*). Someone only talked about or thought of (*calls a
doctor*, *tells Dana about a man*, *a text from a man*, *dreams of a man*), a thing named
after a person (*the driver's seat*, *a man's jacket on the chair*, *a cop car*,
*man-made*) and a reflection (*Dana in the mirror*) bring in nobody. `info` lists unnamed
people as *unnamed:*, with where they are (*the man on the computer screen*).

Unnamed people can be men, boys, women or girls, so *she* or *he* right after an unnamed
woman, girl, man or boy acts means them, even beside named women or men (*A woman grabs
Sarah. She cuffs her hands…* reaches H3 as *The woman cuffs Sarah's hands…*). A line
tagged with an unnamed person (*"Come with me," the man says*), or a *he* or *she* right
after one acts, is that person's: the prompt writes *The man says: <d>…</d>* and no named
character's lips move; a voice on a phone or off screen moves nobody's lips. Something an
unnamed person does to a named character is read as usual and says who did it (*the man
has let go*); something done to an unnamed person lands on nobody named.

### Reference pictures

Wire up to four pictures into `ref_image_1` … `ref_image_4` and tag each one
`<Picture 1>` … `<Picture 4>` beside a name or anywhere in that person's lines in
`character_memory` (`Sarah <Picture 1>: …`, `Sarah: <Picture 1> …`, `Sarah (<Picture 1>):
…`), or beside the name in a beat (`Sarah <Picture 1> walks in.`). A tag in the scene or
anchor that belongs to nobody goes in every shot. A person's picture goes in, cropped to
face and hair (found with
DWPose from `comfyui_controlnet_aux`, with `ckpts/hr16/yolox-onnx/yolox_l.torchscript.pt`
and `ckpts/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt`), in
every shot they are in, so their clothes come only from the words and their face stays
theirs from shot to shot. This holds on every checkpoint and LoRA, also ones not trained on
pictures (anything but ref2va). Tags are renumbered for each shot, and the frame a shot
opens on is the picture after them.

### Continuity

- **Look:** every continued shot is graded onto the frame it continues from (the whole
  tone curve, so a shot that opens washed out gets its colour back), and its end is
  brought back to the scene's contrast and colour when it is more than 3% off (a lift by
  at most a quarter, brightness never changed). H3 also adds contrast and colour to every
  continued shot, so the frame handed on is toned down by the boost it measured, and H3
  brings it back up to the true look. `info` gives each shot's opening against the frame
  before and its end against shot 1 (contrast, colour and sharpness).
- **Sharpness:** every hand-off and every contrast lift adds a little edge and texture, so
  shot after shot would look more and more oversharpened. In every frame, the finest detail
  (a pixel or two across) and the detail just above it (up to about eight pixels) are each
  measured against the picture's larger shapes, a ratio that holds steady when what is in
  view changes. Where either is more than 5% over the scene's first shot, that detail alone
  is eased back down to it (never below 60% of it); tones, colours and larger shapes are
  left as they are, and nothing is ever sharpened. A cut starts a new scene. `info` says
  when a shot was eased.
- **Background:** the frame handed on is decoded and encoded again, and that round trip
  is not exact, so walls, furniture and gardens would turn blotchy shot after shot. Every
  part of the frame that stayed still through the whole shot and still matches the
  frame handed on before it is put back to that frame's pixels; whatever moved, and a
  margin around it, is kept as rendered. A change counts against how bright that part of
  the frame is, so in a dim room a leg that moved is still seen as moved and is never
  pasted back where it was, next to where it went.
- **Someone coming back:** the node keeps the latest frame it handed on with everyone's
  face in view. When someone comes back into the scene (they left it, or a later beat
  brings them in again), that frame goes in as one more picture, the same place a little
  earlier, and the shot says that this person looks exactly as there on coming back into
  view. A face that is only turned away or hidden (lying face down, a shot of her legs)
  gets nothing: fed in for those, the earlier frame had H3 put the whole scene back the
  way it was there, a woman lying on the floor standing up again with her ankles being
  cuffed. A shot where anything goes on or comes off, and a cut, start it afresh. This is
  off on a checkpoint or LoRA not trained on pictures, and `info` says so.

### Sound

Sound is made only where it belongs. Each spoken line is rendered with the picture in its
own stretch of sound, so the lips move with it, and everything before, between and after
the lines is held silent. A muffled sound from a taped or gagged mouth, quoted (*"Mmph!"*)
or told in the beat (*whimpers into the tape*, *her muffled screams*), is made afterwards
in its own short stretch, by a second pass that changes only the audio, so the taped mouth
never moves to it and nothing is babbled around it. A shot with no line and no muffled
sound is rendered silent, so no lips move, and then gets all of its sound in that second
pass. The second pass works from a copy of the picture at `foley_resolution` (`off` leaves
those shots silent). `info` gives each line's time in the shot.

## Speed models

- **PDD Acc:** run the model through `MiniMaxH3PDDAccApply`, wire its `MODEL` output into
  `model` and its `sigmas` output into `sigmas`. It samples on its own sigmas with `euler`
  at shift 12/3. Its sigmas wired in without its model are left out.
- **Speed LoRAs** are sampled the way they were trained: a LoRA whose metadata gives its
  sampling (DMAD) gets exactly that, FlashGen gets its 4-step schedule, and any other
  few-step LoRA (*4step*, *8step*, *turbo*, *lightning*, *distill*, *lcm*) gets `euler` on
  your steps. The same goes for a checkpoint with one merged in.
- **Hyperflow** samples on its own 8-step grid at shift 12/3 with `euler`.
- **FastH3** runs at shift 10/3 with its sparse attention.
- Wired `sigmas` override all of these. `info` says what the model got.

## Settings

| widget | |
|---|---|
| `resolution`, `megapixels` | aspect preset and size; at 1.0 each preset is its native size |
| `shot_seconds` | the longest a shot may be (a beat that needs longer runs on into more shots), and every shot's length when `shot_length` is *fixed* |
| `shot_length` | *from the beat*: as long as the beat takes at real-life speed (see *Real-life speed*), at least 5.2s; *fixed*: every shot is `shot_seconds` |
| `steps`, `sampler_name`, `scheduler`, `seed` | as in KSampler; one seed for the whole video |
| `first_frame` | the first shot starts on this picture |
| `sigmas` | your own schedule |
| `shift_video`, `shift_audio` | H3's sigma shift, 12/3 by default |
| `silence_wordless` | render shots without a spoken line silent, then make their sound |
| `plan_only` | report the shots and their text without rendering |
| `anchor` | text at the top of every shot; with it, every paragraph is a shot |
| `character_memory` | the characters, their pictures and what they have on from the start |
| `foley_resolution` | the size of the picture copy the sound pass works from: *quarter*, *half*, *full* or *off* |

The finished `images` are ordinary frames: upscale them with ComfyUI's own upscale nodes.

## Outputs

| output | |
|---|---|
| `images`, `audio` | the whole video and its soundtrack |
| `info` | the setup, every shot's cast and state, and timings |
| `script` | the exact text each shot was given |

ComfyUI hands `info` and `script` on only when the whole video is done, so a text box
on them shows the last run's text until then. The node's live text shows the plan and
progress right away.
| `frames_per_shot`, `total_frames`, `shots`, `video_seconds` | counts |

## Other nodes here

- **H3 Shot Length**: one shot length as seconds and as a valid H3 frame count.
- **H3 Overlay**: watermark and intro title composited onto the finished frames.

## Notes

- No negative prompt and no cfg setting: H3 runs at cfg 1.
- Each shot's `info` line ends with how long it took and where the time went (*172s (encode
  24s, sampling 95s, sound 21s, decode 29s, finish 3s)*): *encode* is the text encoder
  reading the shot's text and pictures, *sound* the sound pass, *finish* the look and
  hand-off work. Reference pictures go through the VAE once per run, not once per shot.
- Finished frames wait in a file in the system temp folder instead of in RAM (about 1.5 GB
  per 10-second shot at 1 megapixel), so a long video no longer pushes the text encoder and
  the model out of RAM, where they would be read back from disk before every shot. The file
  is deleted as soon as it is made, and its space comes back when ComfyUI lets go of the
  video. With too little disk space the frames stay in RAM as before.
- Denoise is fixed at 1.0: partial denoise desyncs the joint audio/video schedule.
- The node tells ComfyUI H3's real working memory (about 120 KB per token), so on long
  shots the rest of the model streams in as needed. If a shot still does not fit, the
  error says to lower `megapixels` or `shot_seconds`.
- If the node's files change while ComfyUI runs, `info` starts with `RESTART COMFYUI`.
- Widgets are restored by position. If they read wrong, recreate the node as described
  under Install.

## Disclaimer

The owner of this repo will not be responsible for any copyright strikes incurred
because of use. You are responsible for your works. Use this node responsibly and
ethically.
