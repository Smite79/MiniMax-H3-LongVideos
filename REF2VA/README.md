# H3 Long Videos REF2VA

Make long (up to ~120s) MiniMax-H3 videos from a single prompt **plus reference
images**, in ComfyUI. Self-contained: uses only ComfyUI core's H3 support (no
other packs).

This is the FL2VA sampler with H3's **ref2va** task wired in. Everything below
describes the shared behaviour; the reference-specific part is here.

## Install
This folder ships **inside** `H3-LongVideos-V1/` and is loaded by that pack's
`__init__.py` — there is nothing separate to install. Restart the ComfyUI
**server** (not just a browser refresh) and the node appears alongside FL2VA.

## Reference images (the REF2VA part)

Connect up to four images to `ref_image_1` … `ref_image_4`. The tokenizer labels
them `<Picture 1>` … `<Picture 4>` **in input order** and appends your prompt
after them, so you can bind one to a character by name in the prompt —
`Kristy, <Picture 1>, walks around the garage` — or say nothing and let them work
as a general appearance anchor.

**A shot carries either references or the last-frame handoff — never both.** They
are two different task conditionings competing for the same `cond_video_latents`
slot inside ComfyUI's H3 model wrapper: the reference branch overwrites what the
keyframe branch wrote, while the packed layout still reserves rows for both, so a
shot given both would hand the DiT fewer latents than it has condition rows.
`ref_mode` is how you choose, per run:

| `ref_mode` | Shot 1 | Shots 2+ | Trade |
|---|---|---|---|
| `first shot` *(default)* | references | handoff | Continuity unbroken; identity is established once and then carried by the frames |
| `every shot` | references | references | Strongest identity; **no handoff**, so beats meet as cuts, not one continuous take |
| `every shot + handoff ref` | references | references **+ previous last frame as an extra reference** | Continuity returns as a soft signal — the model is *shown* where the last shot ended rather than told to start exactly there |

`ref_image_size` sets how large each reference is encoded. `match` scales it down
to the generation's pixel area, so a reference costs roughly one frame per step.
`max` uses the reference pipeline's 2048 short edge for the best identity
fidelity — but reference rows are re-attended **every step of every
ref-conditioned shot**, so on a long chain `max` is several times slower. Neither
mode ever upscales a small reference.

With no reference connected, the node behaves exactly like FL2VA.

`info` reports which shots took the reference channel and which kept the handoff,
and `plan_only` previews the same split before you spend a render on it.

## Node: **H3 Long Videos REF2VA**  (category: sampling/minimax)
*Registered as `H3LongVideosREF2VA`, separate from the FL2VA node in the parent
folder. The two are independent nodes and can sit in the same graph.*

One node. You set just two things:

1. **prompt** — this is the `integrated_multimodal_description` (the visual +
   action timeline). The first paragraph is the look/character kept across the
   whole video; each later paragraph is a scene beat. Put dialogue and any
   "lips closed / not speaking" beats here. (Blank line between paragraphs.)
2. **total_seconds** — how long the finished video should be.

The node reads the paragraphs, spreads `total_seconds` evenly across them,
splits that into shots that fit both H3's 15s ceiling and your VRAM, chains
each shot from the previous one's last frame, trims the seams, and outputs the
finished **images** + **audio** (plus **info** and the **script** it built).
Optionally it also upscales the result and composites a **watermark** and/or an
**intro title** onto the finished frames — see [Text overlays](#text-overlays-watermark-and-intro-title-overlaypy).

Also on the node: a **resolution** dropdown — a **native** 768-short-edge tier
per ratio (best detail) and a **fast** 512-short-edge tier per ratio (for the
generate-low-then-upscale workflow), plus a **balanced** 640 tier. Every option is a
valid multiple of 32 — there's no custom width/height to snap or mis-type. The
fast tier is ~4× fewer pixels, so it renders faster, frees VRAM, and
(because the length budget is resolution-aware) unlocks **longer shots** — on a
16GB card a 512 shot reaches the full 362f/15s where native only manages ~243f/
~10s. Best for close/medium shots; H3 distorts faces on *wide* shots at any
resolution, so keep faces reasonably large in frame. Pair the fast tier with an
external LTX 2.3 upscale pass (with correct sigmas) to bring finals back to high
resolution at near-native quality. Also: seed (with control-after-generate),
steps, cfg, sampler, scheduler. Everything else (fps, seam trim, **handoff offset**,
VRAM headroom, explicit anchor, ambient `global_soundscape`,
`non_diegetic_music`, `character_memory`) is optional with sensible defaults.

**Shot length is sized from what each beat stages** (`per_beat_length`, **on by
default**). A beat's time is ~2s of setup plus ~2.5s per action clause, or its
spoken line, whichever is longer — so "she takes off her jacket and drops it on
the bench" gets ~7s while a three-part beat gets more. `shot_seconds` and the VRAM
budget are the **ceiling**, not the length of every shot.

This exists because the opposite is a real failure, not an aesthetic one. A 3-second
action in a 12-second shot leaves nine seconds the model was told nothing about,
and it fills them by repeating or **reversing** the action — which is why clothing
came off and went back on. The estimate deliberately leans **short**: an unfinished
action is continued by the next shot from the handoff frame, while an overlong shot
is unrecoverable.

Override any single beat with `seconds: 8` on its own line inside that paragraph —
that wins over the estimate, over `shot_seconds`, and over the toggle, and it is
honored down to H3's 5-frame minimum. Turn `per_beat_length` **off** to give every
shot the full ceiling; `info` then warns about any beat too thin for the length it
got, since that is the case the node cannot size for you.

*Note for workflows saved before this:* ComfyUI stores widget values, so an existing
node still holds the old `per_beat_length` value and must be ticked by hand.

**A short `shot_seconds` used to be ignored.** Values below ~5.2s were raised to the
124-frame budget floor, so 1s, 2s, 3s and 4s all rendered identically. That floor is
the fallback for what the node *guesses*, and must never override what you asked for;
an explicit request is now honored down to H3's 5-frame minimum.

**`info` not updating between runs.** The node had no `IS_CHANGED`, so ComfyUI keyed
its cache on the inputs alone and re-queueing with unchanged widgets returned the
previous outputs — `info` among them. That is actively misleading here, because both
`info` and the chosen shot length depend on live free VRAM, which is not an input.
`plan_only` now always recomputes (it is near-instant, and a stale plan is worse than
none); a real render stays cacheable, so change the seed or any widget to force one.

**Every shot coming out ~5s (the 124f floor).** The length budget is card capacity
minus measured weight size minus `vram_headroom_gb`. If that came out at or below
zero it returned the internal 124-frame floor immediately — so a checkpoint that
*fits* but leaves less than the headroom (e.g. ~14.6GB of weights on a 16GB card)
pinned **every** shot to ~5.2s no matter what the beats asked for, and dropping to the
fast 512 tier changed nothing, because the early return skipped the resolution scaling
entirely. That case now runs the normal arithmetic: ~14.6GB weights on a 16GB card
gives 226f (~9.4s) instead of 124f (~5.2s). Weights that genuinely exceed the card
still floor at every resolution — that one is real, and no shot length fixes it.

A deficit is weights, not latent, so it is deliberately *not* scaled by resolution;
in that regime the tiers report the same length. Resolution only buys frames when
there is a surplus to scale. And when the floor is hit for real, `info` now says so
outright — which knob moved it, and by how much — instead of quietly handing back 5s
shots.

**Beat counts can no longer be collapsed by a setting.** `beat_split` used to offer a
strict `blank line` mode, and it was the only control on the node that could silently
lose beats: six beats typed as two blocks of three rendered as **two** shots, with
nothing in `info` to say why (the split note is only written when a paragraph is
actually split). That option is **removed**. `beat_split` now offers `auto` and
`each line`, which produce identical results — neither can drop a beat — and a
workflow that still stores `blank line` reads as `auto`. The widget itself is kept in
place rather than deleted, because removing a widget shifts every stored value after
it in already-saved graphs.

Nothing else on the node changes the beat count. The only other control that alters
it is `anchor_override`: leave it empty and paragraph 1 is consumed as the anchor
(see below) — intended behaviour, and now guarded against the case where that
silently deletes a shot.

**Repeat name mentions are collapsed automatically.** Naming one person twice in a
single beat — "Kristy finds Dan… she walks over to **Dan**" — is the most reliable
way to make H3 render that person twice, and binding the description once doesn't
fix it, because the bare repeated name is what duplicates. The node now rewrites the
second and later mentions to the right pronoun by grammatical case: subject
(`and Dan takes it` → `and he takes it`), object (`over to Dan` → `over to him`),
possessive (`Kristy's toolbox` → `her toolbox`). Write the beats however reads
naturally; you don't have to police your own repeats.

It only fires where the result is unambiguous. The person's pronoun must be known —
declared in their sheet (`Dan = he, …`) or inferable from a gender word — and no one
else in the shot may share it, or "he" couldn't be traced back. Words inside double
quotes are never touched, so a name in a spoken line (`"Kristy, over here!"`) stays
exactly as written. The first mention always survives, so the description still has
a name to bind to.

**A first paragraph that is really a beat is now rescued, not eaten** — including with
no `character_memory` set, which is the common case and the one the first version of
this guard could not see. A first paragraph is kept as a beat when every sentence in
it stages an action ("Kristy walks around in a garage looking for engine parts.") and
its subject recurs later in the prompt, or when it strips to nothing. A scene/style
anchor never matches; neither does a mixed paragraph that still carries scene text.
Without
`anchor_override`, paragraph 1 becomes the identity anchor — so three paragraphs
render as two shots. That is by design, but it turns into silent data loss when
paragraph 1 is itself an action beat naming a tracked character ("Kristy walks
around in a garage looking for engine parts."): the anchor is stamped on every
shot, so any sentence naming a tracked character is stripped out of it to stop
that character being introduced twice — which leaves nothing at all. You lose the
shot *and* the only scene text you wrote. The node now detects that exact case
(anchor strips to empty), keeps the paragraph as a **beat**, and puts a WARNING at
the front of `info`. It can't misfire on a normal anchor: a `wardrobe:` line, or
any prose that survives the strip, is left alone, and with no `character_memory`
nothing is tracked so nothing is stripped. The real fix is still to put the
setting and style — **with no character names** — in `anchor_override`.

**Character / wardrobe memory.** The keyframe handoff only carries what the
*last frame* showed — so if a shot ends zoomed in on the face, the pants aren't
in that frame and the next shot reinvents them. The fix is to keep wardrobe in
one **mutable text channel** that's re-stamped into every shot, independent of
framing — and, crucially, to keep clothing **out of the permanent anchor prose**,
because the anchor is immutable and would re-assert a garment you're trying to
remove.

The rule: the first paragraph's prose is **permanent identity only** (hair,
face, build). Clothing goes on a `wardrobe:` line — either in the first
paragraph or in the dedicated `character_memory` field (that field wins if both
are set). Example first paragraph:

    Maya: short silver hair, scar over left eyebrow, athletic build. Cinematic.
    wardrobe: grey cargo shorts, red flight jacket, black boots

**Automatic removal from your prose (no `wardrobe:` line needed).** With
`auto_wardrobe` on (the default), the node reads clothing *removals* straight
from a beat's own action text — "she takes off her jacket," "he sheds his coat,"
"she peels off her gloves" — and drops that item, with no directive to type.
It's safe by design: a removal only fires on an item the character is already
wearing, so non-garment phrases like "the plane takes off down the runway"
match nothing and change nothing. If a name precedes the action ("Maya takes
off…"), only that person is affected.

**The keyframe carries the start state; the prompt carries the end state.** From
the removal shot onward the garment is gone from the person's description, and
that shot alone states the change outright: *"She starts this shot wearing the red
jacket and takes it off during the shot; by the last frame the red jacket is off
and she is not wearing it. The motion runs one way only: the clothing comes off
and is never put back on, never re-worn, and the action never plays in reverse."*
It uses the person's **pronoun**, never a bare name (a bare name re-introduces
them, and re-introducing someone is what makes the model render them twice).

Both halves of that exist for a reason, and both were bugs first:

- **Listing the garment as worn in the shot that removes it made the video play
  backwards.** A removal is the one wardrobe change whose motion is symmetric —
  the same frames reversed are a person putting the garment *on*, and both
  readings satisfy "takes off her red jacket" equally. When the shot's own
  description still said "wearing a red jacket", backwards was the reading that
  matched the description, so the removal rendered in reverse and the jacket came
  back. The start state does not need the description: for every shot after the
  first, the handoff keyframe already shows the garment still on.
- **Naming the garment in a LATER shot put it back on.** To a video model a
  mention is a presence cue and a negation is a weak one, so "she is no longer
  wearing the red jacket" in the following shot was itself enough to re-dress her.
  No shot after the removal names the item at all now — they simply describe what
  she *is* wearing.

Plural garments get plural agreement ("the navy overalls **are** off, he is not
wearing **them**"), because the clause is read literally by the text encoder.

This works whether the garment lives in the wardrobe channel or **only in your
anchor prose** ("A woman in a red jacket and a man in a black t-shirt"): the
phrase is scrubbed from the anchor so it can't re-apply itself forever. The
garment's **head noun** is what gets matched — "takes off her red jacket" reads as
*jacket*, not *red* — because matching the adjective used to pull out "A woman in
a red" and scrub the **person** out of the scene while leaving the jacket behind.

Additions and swaps still use an explicit one-token line (`wardrobe: += mirrored
sunglasses`), which also overrides the auto-detection. Turn `auto_wardrobe` off
to drive wardrobe purely through explicit `wardrobe:` lines. Either way, the
`script` output shows the resolved wardrobe for every shot, so you can always
see exactly what each shot inherited.

**No gibberish / no mouths moving before dialogue.** H3 will animate — and
vocalize (as babble) — a mouth on any shot it thinks involves speech, which is
why action shots leading up to a line often show moving mouths or gibberish
audio. With `auto_silence_nonspeech` on (the default), any beat with **no
scripted dialogue** automatically gets a "lips closed, no dialogue" clause, so
mouths stay shut and silent until real speech. A beat counts as dialogue only if
it contains **double-quoted** words (or an `<d>…</d>` tag) — so to make someone
speak, put the words in double quotes: `She says, "Tower, ready for departure."`
Those shots are left alone; every other shot is silenced. Turn the toggle off to
manage lip state yourself. Pair it with `handoff_offset` if a dialogue shot still
hands a mid-word open mouth to the next shot.

Silencing now covers **both channels**. The lips-closed clause constrains the
picture only; H3 builds audio from its own fields, and an **absent**
`overall_soundscape:` leaves that branch unconditioned — which is exactly when it
fills a silent shot with speech-like babble. Every silenced shot therefore also
carries a soundscape line that says *no voices, no speech, no talking* outright.
If you supply a `global_soundscape`, it is kept and the no-voice constraint is
appended to it on those shots only.

**`mute_nonspeech_audio` is ON by default** — the deterministic backstop. Prompt
and soundscape clauses *ask* H3 not to vocalize; muting guarantees it by zeroing
the audio of every shot with no quoted line (neighbouring audible shots get a
short `mute_fade_ms` ramp so nothing clicks). The trade-off is real and `info`
reports it on every run: that shot's generated **ambience goes with the babble**,
so lay a continuous ambient bed under the video in post — or untick the widget to
keep H3's own sound and rely on the prompt-side silencing alone.

To change or remove an item mid-chain *explicitly*, put a `wardrobe:` line **inside** the
beat where it changes (not as its own paragraph). You have two ways, and you do
**not** have to restate the whole outfit:

- **One-token edit (easiest):** `wardrobe: -= jacket` removes any item matching
  "jacket"; `wardrobe: += mirrored sunglasses` adds one. Everything else the
  person is wearing carries forward untouched.
- **Full replace:** `wardrobe: grey cargo shorts, black tank top` sets the whole
  outfit (use when several things change at once).

Either way the change is sticky from that shot onward, and because clothing
lives only in this channel (never the anchor prose), a removal actually stays
gone.

**Two or more people.** Name each person, separated by `;`, and put their
**identity and clothing together** in the named channel:

    wardrobe: Maya = silver hair, scar, grey cargo shorts, red flight jacket; Jon = bald, bearded, navy overalls

Then use their names in the beats. The node binds each person's description
**inline at the single place their name appears** — `Maya (silver hair, grey
shorts, red jacket) greets Jon (bald, navy overalls)` — so each name occurs once
per shot. This matters: the earlier approach emitted a separate `Maya: clothes`
sentence *and* left `Maya walks…` in the beat, and two mentions of a name make
text-to-video render the person **twice**. Binding inline fixes that. A tracked
person you don't name in a beat is left out of that shot (not forced in).

Critical rule to avoid duplicates: **keep names out of the anchor.** The anchor
is stamped on every shot, so a name there is an extra mention that re-triggers
duplication. Anchor = scene and style only (no people); all per-person identity
and clothing go in the named channel; who's in a given shot is decided by which
names you use in that beat.

**Writing with pronouns (recommended for two people).** To keep prose natural
and cut duplication further, you can refer to people as "she"/"he"/"they" in the
beats instead of repeating names — but then declare each person's pronoun in
their sheet so the node knows who's who:

    wardrobe: Maya = she, silver hair, scar, grey shorts, red jacket; Jon = he, bald, navy overalls

Now "She takes off her jacket" attributes the removal to Maya, and each person's
description binds at the pronoun (`She (silver hair, grey shorts…) greets him
(bald, navy overalls)`) — no names, no doubling, and removals still stick. In a
one-person scene any pronoun maps to that person automatically (no declaration
needed). Two people of the **same** pronoun can't be told apart by "she" alone —
name them in the beat where it matters, or use an explicit `wardrobe: Maya -=`
line. The pronoun token itself is used only for resolution; it never shows in
the description.

Edit one at a time — `wardrobe: Maya -= jacket` drops Maya's jacket and
leaves Jon exactly as he was. The name can be written with or without a colon
(`Maya -= jacket` and `Maya: -= jacket` are the same, and you can seed with
`Maya: grey shorts, red jacket` too — whichever reads naturally). Names you
don't mention are untouched; a person with all items removed simply drops out of
the sheet. This is the node-side of multi-person; H3 itself is weakest at
multi-subject identity binding, attribute cross-wiring, and multi-speaker audio
— keep people visually distinct, add spatial cues ("Maya on the left"), and
prefer one speaker per shot for clean dialogue.

**Steps:** default is **20**, the right value for the base H3 model
(res_multistep + simple). Only drop to 6–8 if you have a working 4-step
distill/turbo LoRA or a low-step MXFP8 checkpoint — on the bare base model,
low steps are the main cause of soft/under-formed frames (faces worst).

**Handoff offset:** if chained shots open with moving or "talking" mouths, set
`handoff_offset` to 2–4. The node then ends each shot that many frames early
and hands *that* frame to the next shot instead of the literal last frame
(which can catch a mid-word open mouth), trimming the matching audio tail so
A/V stays aligned. 0 = use the last frame (original behavior).

**Audio, the three H3 sections.** You don't type any field labels — the node
assembles them:
- **Visual + dialogue** → the **prompt** box. It *is* the
  `integrated_multimodal_description`. Speech, lip state, and diegetic sound a
  character makes/hears all go here, in the beats.
- **Ambient bed** → the **global_soundscape** widget. Appended to every shot as
  `overall_soundscape:` — environmental sound only (rain, room tone, engines).
- **Score** → the **non_diegetic_music** widget. Music is **opt-in**: leave it
  blank and the node writes `non_diegetic_music: N/A` on every shot, so H3 adds
  no score (a blank field otherwise makes H3 improvise its own music). Fill it in
  to request a specific score (instrumentation, tempo). Music a character plays
  or hears is diegetic and belongs in the prompt beat.

Both audio widgets are global (stamped on every shot) so a bed/score stays
consistent across the whole video. Leave either blank to omit that section.

**VRAM:** it measures the model's size (via ComfyUI's own accounting, so
quantized checkpoints report correctly), picks the largest shot length your
card can attempt, and on a caught out-of-memory quietly backs off (tiled
decode → lower resolution) instead of crashing.

Forcing `shot_seconds` now still respects that budget: a forced length that
won't fit is **clamped down** to what fits and the clamp is reported in `info`
(e.g. 15s on a 16GB card clamps to ~10s/243f). Set `allow_oversize_shots` to
override and honor the requested length anyway — but then the render may spill
into system RAM (slow) or OOM.

Note on the reactive backoff: it can only fire on a *caught* OOM. If NVIDIA's
"CUDA – Sysmem Fallback Policy" is on (the default on Windows), an over-budget
run silently spills VRAM into system RAM instead of raising, so the backoff
never triggers and you get a slow, over-cap run. Set that policy to *Prefer No
Sysmem Fallback* (per-app is fine) so over-budget shots raise and the backoff
can do its job. The predictive clamp above is the guard when the fallback is on.

Between-shot cleanup (`cleanup_between_shots`, on by default): after each beat
the node moves that shot's decoded video+audio to system RAM and runs a full
VRAM+RAM purge (Python GC, ComfyUI's aggressive cache clear, and the CUDA
allocator's `empty_cache` + `ipc_collect`). Without this, every shot's frames
stay resident on the GPU and accumulate across the chain — the main reason a
long (12-shot) run OOMs partway through even when a single shot fits. The
handoff keyframe is kept off-GPU too and re-encoded next shot. Leave it on for
16GB; turn it off only on a large card if you want to skip the small per-shot
cleanup cost. (Trade-off: the finished frames now accumulate in system RAM
instead — expected, since the full video has to live somewhere — so for very
long high-res renders, watch RAM rather than VRAM.)

## Built-in upscale (optional post-pass)
Set `upscale` to enable a post-generation pass on the finished frames:
- **rtx** — NVIDIA **RTX Video Super Resolution**, running on RTX Tensor Cores.
  Fastest option by a wide margin (NVIDIA claims ~30× vs other local upscalers)
  and generally cleaner on video than UltraSharp-class models. Requires the
  `Nvidia_RTX_Nodes_ComfyUI` pack (`git clone https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI`
  plus `nvidia-vfx` from its requirements; it may not appear in ComfyUI Manager).
  If the pack isn't installed the node falls back automatically — it never breaks
  a render. On 16GB, keep `upscale_batch` low: long clips can exhaust system RAM.
- **model** — runs a Real-ESRGAN / UltraSharp-class upscale model from
  `models/upscale_models` (pick it in `upscale_model`), chunked with cleanup so
  a long clip doesn't OOM. Real per-frame sharpening/detail.
- **lanczos** — plain high-quality resize to `upscale_target_short_edge`
  (enlarges; adds no new detail).

`upscale_target_short_edge` fits the result's short edge to a target (0 = keep
the model's native factor). Typical use: generate on the **fast 512 tier** for
speed/length, then set the target to 768 to land back near native size.

**Honest ceiling:** this pass sharpens and enlarges; it does **not** reconstruct
video detail the way a second-model re-generation does. For true near-native
recovery from a low-res render, a separate **LTX 2.3** upscale pass is the gold
standard (it re-generates detail and holds lip-sync) — but it needs its own
model loaded, which is why it lives outside this node rather than in it. Use the
built-in model upscale for a quick, self-contained quality lift; use an external
LTX 2.3 pass (with correct sigmas) when you need the best possible result.

## Text overlays: watermark and intro title (`overlay.py`)
Two optional PIL overlays, **composited onto the finished frames — never asked of
the model and never added to the prompt.** H3, like every video diffusion model,
renders text as plausible-looking letterforms that drift, warp and re-spell
themselves frame to frame; a watermark that changes shape every frame is worse
than none. Compositing gives pixel-identical text on every frame at zero sampling
cost, and keeps the words out of the prompt where they'd otherwise steal
conditioning from the actual shot.

Both draw **white glyphs on a fully transparent layer** that is alpha-blended over
the video, so only the letters land on the picture and the image shows through
everywhere else.

- **`watermark_text`** — stamped on **every frame**. `watermark_position` (7
  anchors: the four corners, `top-center`, `bottom-center`, `center`),
  `watermark_size` (cap height as a **% of the short edge**, default 4.0, so the
  mark keeps its relative size at any resolution, ratio or upscale factor),
  `watermark_opacity` (default 0.75 — reads as a watermark without burying the
  picture), `watermark_margin` (inset as a % of the **short** edge).
- **`intro_text`** — a title over the **opening of the finished video**, not a
  replacement card: the first shot plays underneath it. Multi-line is centered as
  a block. It holds at full opacity for `intro_seconds` (default 3.0), then
  linearly fades over `intro_fade` (default 0.6; 0 = hard cut). `intro_position`
  offers `center`, `lower-third`, `top-center`, `bottom-center`; `intro_size`
  defaults to 9.0% of the short edge.
- **Fits every preset automatically.** Both sizes are measured from the **short
  edge**, and the block is word-wrapped and then shrunk until it sits inside the
  margins. Sizing from the *height* meant a portrait canvas (9:16, 3:4) drew the
  text ~1.75× larger on the canvas with the least room for it, and anything that
  ran past the frame was silently clipped by Pillow — no error, no note, just
  missing characters. A long title now wraps onto as many lines as it needs and
  reads the same at 512×896 as at 1536×672.
- **`overlay_font`** — TrueType face for **both** overlays: a bare name resolved
  against the system font folder (`arial.ttf`, `arialbd.ttf`, `segoeui.ttf`) or a
  full path to a `.ttf`/`.otf`. If it won't load, the node falls back through
  Arial → Segoe UI → DejaVu Sans → Liberation Sans, and finally to PIL's bitmap
  default (which ignores size — ugly, but never fatal).
- **`overlay_stroke`** — black outline in pixels around the white text. 0 keeps it
  pure white; **2–3 makes it survive a bright sky or a white wall.**

Leave a text field empty to skip that overlay; both are off by default.

**Applied last, after any upscale**, so glyphs are rasterized at the final pixel
size instead of being interpolated up along with the picture. Two consequences
worth knowing: the overlays run once on the **whole concatenated video**, so the
intro sits over the opening of the finished piece rather than the top of every
shot; and because it happens post-upscale, the text is crisp even on a
fast-512-then-upscale workflow.

Everything here is **best-effort by design** — any failure (missing Pillow, an
unloadable font, a bad position) returns the frames untouched and writes a note
into `info`, because a cosmetic overlay must never destroy a finished render.
Blending is chunked 64 frames at a time and cropped to the text's tight bounding
box, so a corner watermark on a 3000-frame chain doesn't blend 3000 full frames.

**Checkpoint swaps are detected and flushed.** ComfyUI keeps previously-loaded
models resident and only evicts reactively, so switching checkpoints mid-session
(e.g. NVFP4 → FP8 → MXFP8 while comparing quality) leaves the *old* DiT on the
card alongside the new one, plus any hooks a previous LoRA installed and stale
allocator blocks sized for the old layers. The card is then already half full
before the first shot samples — which looks like the node over-spilling, when in
fact the budget was measured against memory the previous checkpoint never
released. The node fingerprints the model (quant format, layer count, weight
size) and, when it changes between runs, calls `unload_all_models()` plus a
double VRAM/RAM purge **before** any measurement or patching. `info` reports it:
`model changed since last run (nvfp4 ~11.7GB -> mxfp8 ~19.5GB): flushed all
resident models and VRAM caches`. Identical models across runs are untouched, so
there's no cost to a normal chain.


**Character duplication at low resolution (`subject_count_guard`).** Duplication
gets markedly more likely *below* H3's native 768 short edge: fewer pixels per
subject pushes the sample away from the training distribution and the model tiles
the figure. The strongest prompt-side defence is an explicit count, so the node
can prepend one to each shot — `Exactly two people in this shot, no duplicates,
no other people in frame.` — counting only the characters actually referenced in
that beat. `auto` (default) enables it when the short edge is under 768 **or when a LoRA is
applied** — a distilled LoRA compresses ~20 steps into 4–8, so it fixes global
composition (including how many people are in frame) within the first step or two
and then reinforces that choice rather than revising it. That is why turbo LoRAs
duplicate subjects even at native resolution with a clean prompt. On LoRA runs the
count clause is also moved to the **front** of the prompt, ahead of scene and
style, so it binds before composition settles. Both stock-loader LoRAs (weight
patches) and bypass LoRAs (injections) are detected. `on` always; `off` never.
Scenery beats with no characters never get the clause.

If duplication persists at native resolution, that's the model rather than the
prompt — keep subjects visually distinct, add spatial cues ("Kristy at the left
wing"), and avoid having two characters overlap in frame.


**Write attributes, not noun phrases.** In `character_memory`, describe people as
bare attributes — `Kristy = she, 27, silver hair, red jacket` — not as noun
phrases like `a woman with silver hair`. A noun phrase renders inline as
`She (a woman with silver hair)`, which puts **two subject nouns in one clause**
("She" and "a woman") and reads to the model as two people: character duplication
from the very first shot, at any resolution. The node now strips these
automatically (`a young woman, silver hair` → `young, silver hair`, and a bare
`a woman` is dropped entirely), but writing attributes directly is clearer and
avoids relying on the cleanup.


**Spilling at native resolution (`decode_tile_frames` / `decode_tile_size`).** The
VAE decode — not sampling — is usually the peak allocation in a run: without
temporal tiling the video VAE expands the *entire* clip at once (a 243-frame
1344x768 shot is the largest single tensor the node ever creates). On a checkpoint
that already exceeds VRAM and streams, that decode is what tips the card into
shared memory.

Set `decode_tile_frames` to 8–16 to decode in temporal chunks, and
`decode_tile_size` to 256 for spatial tiles. Both default to 0 (ComfyUI's
defaults). Lower values mean lower peak VRAM and slightly slower decode. Try these
before dropping resolution — they cost speed, not picture quality.

**Structural limit:** if the weights alone exceed your VRAM (e.g. an unpruned
~19.5GB MXFP8 on a 16GB card), *some* spill is unavoidable no matter what the node
does — the model is streaming before a single frame is allocated. Tiling reduces
the peak on top of that baseline; only a checkpoint that fits removes it.


**Camera direction can summon a phantom person.** Writing motion guidance in the
anchor — "slow camera movement, the camera follows *the subject*", "moves toward
*the person*", "tracks *the figure*" — leaves an unnamed person reference in text
that is stamped into **every** shot, alongside your named cast. The model renders
it as an extra body matching no character sheet. The node now rewrites those
generic references ("the subject/person/figure/character") to refer to the scene
instead, keeping the camera direction intact. Safest is to phrase motion without
a person at all: `Slow, smooth camera movement. Minimal motion blur.`


## Per-beat directives (`key: value` lines)
A beat can carry directive lines that configure it rather than describe it. They
are **stripped out of the prose** before the prompt is built, and they are never
beats of their own — a directive attaches to the next content line, or to the
previous beat if it trails the paragraph. So a `wardrobe:` line inside a beat
won't accidentally become its own shot.

- **`wardrobe:`** — clothing and identity. Covered in full above.
- **`seconds: 8`** (alias **`duration: 8`**) — an explicit length for *this* beat,
  in seconds. This is the highest-priority length signal: it is **honored even
  when `per_beat_length` is off**, because you stated a duration outright. It is
  still clamped to the VRAM budget, which is a hard ceiling — a per-beat length
  can only make a shot **shorter** than the card allows, never longer. Use it to
  give one beat room ("`seconds: 10`") while the rest take the default.
- **`exit: Jon`** — Jon leaves the scene. Like auto-removals, exits are
  **deferred**: he is still present in the shot that *shows* him leaving, and
  absent from every shot after it. This stops a character the story has written
  out from wandering back in. Exits are also detected automatically from the
  action text; the directive is the explicit override.
- **`enter: Jon`** — undoes a previous exit, bringing him back into play.
- **`soundscape:` / `overall_soundscape:`** and **`music:` /
  `non_diegetic_music:`** — per-beat audio that overrides the global widgets for
  that shot only. A beat that sets its own skips the global stamp.

Without a `seconds:` line the length falls back to quoted dialogue (~2.5 words/sec
plus 1s of air, when `per_beat_length` is on) and otherwise to the full budget.
Action prose carries no reliable duration signal — "walks across the tarmac" is 2s
or 12s depending on the tarmac — so a silent beat is **never guessed short**.

## Preview the split without rendering (`plan_only`)
Set `plan_only` on the main node to **True** to see how a job will split —
shots, frames per shot, seconds, total length — near-instantly, with **no
render**. It uses the node's *own* settings (resolution, `shot_seconds`, fps,
prompt), so there's nothing to re-enter and nothing can drift out of sync. The
plan appears in the `info` output and the `shots` / `frames_per_shot` /
`video_seconds` outputs are populated. Turn it off to render for real. (This
replaces the old separate Plan node, which required duplicating settings by
hand.)



## Node: **H3 Shot Length**  (category: MiniMax-H3/utils)
Holds ONE shot length and emits it as both `seconds` and a valid H3 frame count
(17k+5 grid, capped at 362 unless you turn the cap off). Wire:

    H3 Shot Length (seconds) → H3 Long Videos FL2VA (shot_seconds)
    H3 Shot Length (frames)  → Model Preview Override (preview_frames)

One value entered once drives both, so they can't drift apart. It never reads
the model, so it can sit upstream of a preview override without creating a
wiring cycle. The main node also outputs `frames_per_shot` and `total_frames`
if you'd rather read back the value it chose.

## Optional node: **H3 Model Inspector**
Reports the base precision of a loaded model (BF16 / FP8 / INT8 / NVFP4 /
MXFP8) and whether your card runs it natively. Use it to confirm you're on a
base checkpoint (so 20 steps is right) vs a distill/low-step path.

## Requirements
- ComfyUI 0.30+ with native MiniMax-H3 support.
- The node applies **ModelSamplingMiniMaxH3** (the video/audio flow schedule)
  internally by default via `apply_model_sampling`, so you no longer have to
  wire it upstream — a missing patch is the usual cause of gibberish audio.
  Turn it off if you patch upstream yourself. Shifts are exposed
  (`shift_video`/`shift_audio`): base H3 = 12/3 (the defaults), a low-step MXFP8
  checkpoint ≈ 8 video, a 4-step distill/turbo LoRA ≈ 4–6 audio.
- **Pillow (PIL)** only if you use the text overlays. ComfyUI already ships it, so
  this is effectively always satisfied; if it were missing, the overlays are
  skipped with a note in `info` and the render still completes.
- No negative prompt needed — H3 is CFG-free (cfg 1) and the node makes an empty one internally.
- No denoise input — it's fixed at 1.0 internally (partial denoise desyncs the joint audio/video schedule).

## Disclaimer

The owner of this repo will not be responsible for any copyright strikes
incurred because of use. You are responsible for your works. Use this node
responsibly and ethically.
