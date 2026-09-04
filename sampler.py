"""
H3 Long Videos -- chain MiniMax-H3 shots into one continuous video with audio.

Rebuilt from scratch. The previous version grew a large prompt-engineering layer
that wrote continuity guards into every shot; measured, the user's own beat was
under 4% of the conditioning and the rest was boilerplate arguing with it. None of
that is here. What a shot is told is: your scene text, then your beat, verbatim.

What this node does is the part a prompt cannot do -- the mechanics of chaining:

  * splits the prompt into beats on blank lines, one beat per shot;
  * gives every shot the SAME length, so one seed is one noise field across the
    chain (noise is drawn to the latent's shape, so unequal lengths mean unrelated
    noise from the same seed, and detail resets at every cut);
  * hands each shot the previous shot's last frame as its keyframe, encoded the
    way H3 expects a keyframe to be encoded (one frame -> the 5f grid point);
  * keeps identity references on every shot, which is the only fixed anchor a long
    chain has against drift;
  * anchors the audio branch to real silence on shots with no quoted line, because
    H3 is a joint model and an unconditioned audio stream invents a voice that the
    picture then lip-syncs to.

Everything about what the video should CONTAIN is yours to write.
"""

import gc
import json
import math
import os
import re
import sys
import time

import torch

import nodes
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.nested_tensor
import comfy.model_management as mm
import latent_preview
import node_helpers


H3_FPS = 24                    # H3 renders 24 fps, always
AUDIO_LATENT_FPS = 40          # audio latent frames per second
VAE_SPATIAL = 16               # video VAE spatial downsample
RES_MULTIPLE = 32
KEYFRAME_SAFE_AUG = 0.99       # below this, a ref aug would soften the keyframe too
AUTO_TILE_T = 8                # temporal chunk for a tiled decode
MAX_FRAMES = 362               # H3's own ceiling (~15s)
# Latent frames decoded from the PRE-upscale latent to source the handoff. Enough
# for the VAE's temporal context to produce a clean last frame, and cheap.
HANDOFF_LATENT_TAIL = 8
GB = 1024 ** 3

# H3-Base is trained at 768 on the short edge; below that the whole frame softens.
NATIVE_RES = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3":  (1024, 768),
    "3:4":  (768, 1024),
    "1:1":  (768, 768),
    "21:9": (1536, 672),
    "9:21": (672, 1536),
}


CANVAS_MULTIPLE = 32


REF_IMAGE_SHORT_EDGE = 2048


_LAST_MODEL_FP = {"fp": None}


_SILENT_UNIT = {"lat": None}


def _call_node(cls, model, shift_video, shift_audio):
    """Call the H3 sampling node whether it uses the V1 (INPUT_TYPES/FUNCTION) or
    V3 (define_schema/execute) API, mapping the shift args by name."""
    inst = cls()
    # V1 API
    if hasattr(cls, "INPUT_TYPES") and getattr(cls, "FUNCTION", None):
        req = cls.INPUT_TYPES().get("required", {})
        kwargs = {}
        for name in req:
            low = name.lower()
            if low == "model":
                kwargs[name] = model
            elif "video" in low:
                kwargs[name] = float(shift_video)
            elif "audio" in low:
                kwargs[name] = float(shift_audio)
        out = getattr(inst, cls.FUNCTION)(**kwargs)
        # A V3 node exposes INPUT_TYPES and a truthy FUNCTION ('EXECUTE_NORMALIZED')
        # for compatibility, so this branch runs on 0.31+ too -- and there it returns
        # a NodeOutput, not a tuple. Without the unwrap the caller got the wrapper
        # object where a MODEL belonged. Unreachable today because the direct patch
        # succeeds first, which is exactly why it went unnoticed.
        out = getattr(out, "result", out)
        return out[0] if isinstance(out, (tuple, list)) else out
    # V3 API: an execute()/patch() classmethod taking model + shift kwargs
    fn = None
    for cand in ("execute", "patch", "apply"):
        if hasattr(inst, cand):
            fn = getattr(inst, cand); break
    if fn is None:
        raise RuntimeError("unknown node API")
    out = fn(model=model, shift_video=float(shift_video), shift_audio=float(shift_audio))
    out = getattr(out, "result", out)                 # V3 NodeOutput
    return out[0] if isinstance(out, (tuple, list)) else out


def _is_audio_vae(v):
    """True when v looks like the H3 audio VAE (DAC/BigVGAN), False when it looks
    like a video/image VAE, None when it can't be told. The video VAEs carry a
    3-tuple upscale_ratio (t, y, x); the audio VAE carries a scalar and reports
    latent_dim 2 with an audio_sample_rate."""
    ur = getattr(v, "upscale_ratio", None)
    if isinstance(ur, (tuple, list)):
        return False
    if getattr(v, "audio_sample_rate", None) or getattr(v, "audio_sample_rate_output", None):
        return True
    if isinstance(ur, (int, float)) and getattr(v, "latent_dim", None) == 2:
        return True
    return None


# --- sizing -----------------------------------------------------------------

def align_frame_count(n):
    """Up to the next valid H3 frame count. The grid is 17k+5."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return min(n, MAX_FRAMES)


def align_frame_count_nearest(n):
    """The NEAREST 17k+5 grid point, not the next one up.

    align_frame_count always rounds up, which is right for a length you asked for
    -- never give back less than requested. It is wrong for an ESTIMATE: the grid
    steps 17 frames (~0.7s), and rounding an estimate up lengthens the shot in the
    one direction that causes trouble."""
    n = max(5, int(n))
    lo = n - ((n - 5) % 17)
    hi = lo + 17
    return min(MAX_FRAMES, lo if (n - lo) <= (hi - n) else hi)


def video_latent_t(fc):
    return 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2


def temporal_shape(length, fps=H3_FPS):
    """(frame count, video latent frames, audio latent frames) for a shot.

    `fps` is accepted but deliberately IGNORED: the audio latent has to line up
    with 24 fps video or the shot's sound is stretched against its picture."""
    fc = align_frame_count(length)
    return fc, video_latent_t(fc), round(fc / H3_FPS * AUDIO_LATENT_FPS)


def parse_resolution(choice):
    text = (choice or "").strip()
    if text in NATIVE_RES:
        return NATIVE_RES[text]
    m = re.search(r"(\d+)\s*x\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return NATIVE_RES["16:9"]


def scale_to_megapixels(w, h, mp, multiple=RES_MULTIPLE):
    """Scale (w, h) to `mp` megapixels keeping the ratio, snapped to the grid.
    mp <= 0 keeps the preset's own size."""
    if not mp or mp <= 0:
        return w, h
    scale = math.sqrt((mp * 1024 * 1024) / float(w * h))
    sw = max(multiple, int(round(w * scale / multiple)) * multiple)
    sh = max(multiple, int(round(h * scale / multiple)) * multiple)
    return sw, sh


# --- prompt -> beats --------------------------------------------------------

def split_beats(prompt):
    """(scene, beats). Paragraphs are separated by a BLANK line.

    The first paragraph is the SCENE: it is prepended to every shot verbatim, and
    nothing is stripped from it. Every paragraph after it is one beat, one shot.
    A single-paragraph prompt is one shot with no separate scene text.

    Deliberately the whole of the text handling. The previous version rewrote beats
    -- binding descriptions, collapsing repeated names, scrubbing the scene, adding
    continuity clauses -- and the result was a shot whose own action was a few
    percent of what the model was told. What you type is what the shot gets."""
    paras = paragraphs(prompt)
    if not paras:
        return "", []
    if len(paras) == 1:
        return "", paras
    return paras[0], paras[1:]


def paragraphs(text):
    """Non-empty paragraphs, separated by a BLANK line."""
    return [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]


# A line of a character sheet: `Name: attributes`. The directive lines are excluded
# by name -- they are instructions to this node, not people.
_SHEET_LINE = re.compile(r"^\s*(?!(?:remove|off|add|wear|wardrobe)\s*:)"
                         r"[A-Z][\w'’-]{0,24}\s*:\s*\S", re.I)


def is_character_sheet(par):
    """A paragraph that DESCRIBES people rather than staging an action.

    Every line reads `Name: attributes` -- "McKenna: 22, blonde, grey coat." Handed
    to the model as a beat, a sheet spends a whole shot rendering a static
    description. Worse, the wardrobe then lives in ONE shot instead of being
    re-stamped into all of them: later shots describe no clothing at all, so the
    model invents it, and a removal has nothing to scrub because what it would
    scrub was never in the scene.

    A sheet lists ATTRIBUTES. A line that stages an action is a beat, however it is
    labelled -- "McKenna: thrashes in her restraints" and "Camera: pushes in slowly"
    are shots, not descriptions. Getting that wrong is expensive in one direction
    only: a sheet mistaken for a beat costs one visible shot, while a beat mistaken
    for a sheet never renders AND has its words stamped onto every other shot. So
    anything that opens with a verb is treated as a beat.

    A line with speech in it is a beat too -- 'Dan: "Hello."' stages something."""
    lines = [ln for ln in (par or "").splitlines() if ln.strip()]
    if not lines or _QUOTED.search(par) or _DIALOGUE_TAG.search(par):
        return False
    return all(_SHEET_LINE.match(ln) and not _ACTION_AFTER_LABEL.search(ln)
               for ln in lines)


# What follows `Name:` in a sheet is an attribute -- a pronoun, an age, a colour, a
# <Picture N> tag. An inflected verb there means the line stages something instead.
# The participles excepted below introduce attributes rather than actions.
_ACTION_AFTER_LABEL = re.compile(
    r":\s*(?!(?:wearing|dressed|carrying|holding|sporting|wrapped|covered)\b)"
    r"(?:is|are|was|were|has|have|had|does|do|[\w-]+(?:s|es|ed|ing))\b", re.I)


def pull_character_sheets(beats):
    """(the beats that stage something, the sheet paragraphs joined)."""
    beats = beats or []
    sheets = [b for b in beats if is_character_sheet(b)]
    return [b for b in beats if not is_character_sheet(b)], "\n".join(sheets)


def sheet_lines(sheet):
    """[(name or None, line)] for a character sheet, in order. A line with no
    `Name:` label belongs to everyone and is never dropped."""
    out = []
    for ln in (sheet or "").splitlines():
        if not ln.strip():
            continue
        m = re.match(r"\s*([A-Z][\w'’-]{0,24})\s*:\s*\S", ln)
        out.append((m.group(1) if m else None, ln.strip()))
    return out


def sheet_for_beat(sheet, beat, previous=None):
    """(the sheet lines for the people this beat involves, the names kept).

    The sheet is re-stamped into every shot so clothing holds -- but describing
    EVERYONE in every shot puts everyone in every shot. A beat about one person
    renders two, because the text standing beside it says the other one is there,
    and a described person is a person the model draws.

    A PRONOUN counts as naming someone: "Jon takes her jacket off" is about both of
    them, and dropping Maya there would leave the garment being removed undescribed
    in the very shot that removes it. Who "her" refers to is not resolvable from the
    sentence, so it keeps whoever the last beat kept.

    A beat that names nobody at all keeps the last beat's people too, so "She lies
    still." does not empty the frame."""
    rows = sheet_lines(sheet)
    # CASE-SENSITIVE. Prose capitalises a name, and matching without case made the
    # word "will" find a character called Will, and "grace" find Grace.
    named = [n for n, _ in rows
             if n and re.search(r"\b" + re.escape(n) + r"\b", beat or "")]
    used = {m.group(0).lower() for m in _PRONOUN.finditer(beat or "")}
    if used:
        # Resolve a pronoun to the person whose sheet DECLARES it. Adding the whole
        # previous cast on any pronoun put someone in a shot they were not in --
        # "Jon walks out and shuts the door behind him" kept the other character,
        # because "him" was read as evidence that somebody else was present.
        # ONE PRONOUN IS ONE PERSON. Resolved per pronoun GROUP, not per sheet entry:
        # walking the entries and taking everyone who declares "she" is fine with one
        # woman on the sheet and a guess with two, and it used to take BOTH -- a third
        # character pulled into a shot that named two.
        matched = False
        for group, words in _PRONOUN_SET.items():
            if not used & words:
                continue
            # Already accounted for by somebody the beat names outright: "Nora and Dan
            # look at her hands" needs nobody else for "her".
            if any(sheet_pronoun(ln) == group for n, ln in rows if n and n in named):
                matched = True
                continue
            cands = [n for n, ln in rows
                     if n and n not in named and sheet_pronoun(ln) == group]
            if len(cands) == 1:
                named.append(cands[0])
                matched = True
            elif len(cands) > 1:
                # Two people declare it. The scene continuing is the only evidence
                # available, so take the one who was in the last beat -- and if that
                # does not single anybody out, add NOBODY. Naming a person the beat
                # did not is the failure being fixed; leaving them to the keyframe is
                # recoverable.
                narrowed = [n for n in cands if n in (previous or [])]
                if len(narrowed) == 1:
                    named.append(narrowed[0])
                    matched = True
        # A sheet that declares no pronouns tells us nothing, so fall back to the
        # last beat's people rather than guessing.
        if not matched:
            named += [n for n in (previous or []) if n not in named]
    # Somebody is in it, but the beat does not say who -- "Someone knocks at the
    # door." Keep the last beat's people, since a scene usually continues with them.
    # With nobody before it, describing the WHOLE sheet is the same failure in
    # miniature: it puts everyone in a shot on the strength of not knowing. One
    # person on the sheet is unambiguous and still resolves; two or more is a guess,
    # and the guard exists precisely not to make it.
    if not named:
        named = list(previous or [])
    if not named:
        _all = [n for n, _ in rows if n]
        named = _all if len(_all) == 1 else []
    keep = [ln for n, ln in rows if n is None or n in named]
    return "\n".join(keep), named


# A beat that stages somebody ARRIVING. The chain is right for these: the previous
# shot's last frame is where they walk in from. A beat that stages no entrance is
# describing where somebody already IS, and there is no frame to inherit that has them
# in it.
_ENTRANCE = re.compile(
    r"\b(?:walk|step|come|run|stride|hurry|move|wander|burst|barge|slip|climb)"
    r"(?:s|ed|ing)?\s+(?:in|into|through|up|over|back|out\s+of)\b"
    r"|\benter(?:s|ed|ing)?\b|\barriv(?:es?|ed|ing)\b|\bappear(?:s|ed|ing)?\b"
    r"|\bjoin(?:s|ed|ing)?\b|\breturn(?:s|ed|ing)?\b|\bfollow(?:s|ed|ing)?\b"
    r"|\bshows?\s+up\b|\bturns?\s+up\b|\blets?\s+\w+\s+in\b", re.I)


def arrives_in(text):
    """Does this beat stage somebody arriving?"""
    return bool(_ENTRANCE.search(text or ""))


def unresolved_pronouns(sheet, beat, previous=None):
    """[(pronoun group, the people who could answer to it)] this beat cannot settle.

    Two people declaring "she" and a beat saying "her" is a guess, and the guard makes
    none: it adds nobody rather than both. Nobody being described is recoverable --
    the keyframe still carries them -- but it is worth saying, because the fix is to
    write the name instead of the pronoun."""
    rows = sheet_lines(sheet)
    named = [n for n, _ in rows
             if n and re.search(r"\b" + re.escape(n) + r"\b", beat or "")]
    used = {m.group(0).lower() for m in _PRONOUN.finditer(beat or "")}
    out = []
    for group, words in _PRONOUN_SET.items():
        if not used & words:
            continue
        if any(sheet_pronoun(ln) == group for n, ln in rows if n and n in named):
            continue
        cands = [n for n, ln in rows
                 if n and n not in named and sheet_pronoun(ln) == group]
        if len(cands) > 1 and len([n for n in cands if n in (previous or [])]) != 1:
            out.append((group, cands))
    return out


_PRONOUN_SET = {"she": {"she", "her", "hers"},
                "he": {"he", "him", "his"},
                "they": {"they", "them", "their", "theirs"}}


def sheet_pronoun(line):
    """Which pronoun this sheet entry declares for its person, or None.

    Writing the pronoun into the sheet -- "Maya: 27, she, grey coat" -- is what lets
    "her coat" in a beat be resolved to Maya rather than to whoever was in the last
    shot."""
    body = (line or "").split(":", 1)[-1]
    for group, words in _PRONOUN_SET.items():
        if any(re.search(r"\b" + w + r"\b", body, re.I) for w in words):
            return group
    return None


_PRONOUN = re.compile(r"\b(?:she|he|her|hers|his|him|they|them|their|theirs)\b", re.I)


# A determiner in front means the capitalised word DESCRIBES something rather than
# doing something: "her Nike leggings" names a garment, not somebody in the room.
_DETERMINER = frozenset("a an the her his its their our my your this that".split())
_CAPITALISED = re.compile(r"\b([A-Z][a-z’'-]{1,24})\b")


def unknown_people(beats, sheet):
    """{name: [1-based shot numbers]} -- names the beats use as PEOPLE that the
    character sheet never describes.

    A person the sheet does not describe is a person no shot describes. The guard
    keeps the entries for the people a beat names, and there is no entry to keep, so
    the beat stages somebody the model has been told nothing about -- no age, no
    clothes, no face -- and it invents them, differently in each shot. Worse, a beat
    whose ONLY person is undescribed falls back to the previous beat's cast, so the
    shot describes someone who is not in it and stays silent about the one who is.

    It is also how one person written under two names becomes two people, one of
    them a stranger.

    A capitalised word only counts once it has appeared MID-sentence somewhere in
    the script. That is what separates a name from an ordinary word that happens to
    open a sentence, and it needs no list of ordinary words to do it.

    Reported, never acted on: whether a name is somebody already on the sheet under
    another name or a third person in the room is not answerable from the text, and
    guessing would be the node rewriting the script."""
    known = {n.lower() for n, _ in sheet_lines(sheet) if n}
    seen, mid_sentence = {}, set()
    for i, beat in enumerate(beats or [], 1):
        for m in _CAPITALISED.finditer(beat or ""):
            # "Jon's kitchen" is Jon. The apostrophe is in the class for O'Neill.
            word = re.sub(r"['’]s$", "", m.group(1))
            before = (beat[:m.start()]).rstrip()
            prev = re.search(r"([\w’'-]+)\W*$", before)
            if prev and prev.group(1).lower() in _DETERMINER:
                continue
            # Opening a sentence -- or a quoted line -- capitalises anything, so
            # only a mid-sentence appearance is evidence of a name.
            if before and before[-1] not in ".!?:\"”":
                mid_sentence.add(word)
            if i not in seen.setdefault(word, []):
                seen[word].append(i)
    return {w: s for w, s in seen.items()
            if w in mid_sentence and w.lower() not in known}


# Where a beat says something becomes VISIBLE. The other half of a removal: "cuts
# off her coat to expose the jumper" names the coat as coming off AND the jumper as
# what was under it.
_EXPOSE_CUE = re.compile(r"\b(?:to\s+expose|to\s+reveal|to\s+show|exposing|revealing|"
                         r"showing|uncovering|baring)\b", re.I)


def exposed_by(beat, scene):
    """Garments this beat says become visible. [] when none."""
    out = []
    for m in _EXPOSE_CUE.finditer(beat or ""):
        tail = beat[m.end():]
        cut = re.search(r"[,;.]|\band\s+(?:then|he|she|they)\b", tail, re.I)
        span = tail[:cut.start()] if cut else tail
        for word in re.findall(r"\b[\w-]{3,}\b", span):
            low = word.lower().strip("-")
            if not low or low in out or low in _NOT_A_GARMENT:
                continue
            if _RESTRAINT_WORD.match(low) or not _is_entry_head(word, scene):
                continue
            out.append(low)
    return out


def infer_layers(bodies, scene):
    """{under: over} -- which garment covers which, read from the script's own words.

    A sheet lists every layer at once, which tells the model all of them are on show
    simultaneously. Nothing says which is hidden, so the under layer bleeds through
    the top one -- and by the last frame, where only the text governs, it is simply
    drawn on top.

    The script already says what covers what: a beat that takes A off "to expose B"
    has stated that B was under A. Read it from there rather than asking for it."""
    covers = {}
    for body in bodies or []:
        off = infer_removals(body, scene)
        for under in exposed_by(body, scene):
            for over in off:
                if under != over:
                    covers.setdefault(under, over)
    return covers


def hidden_layers(covers, gone):
    """Garments still underneath something that has not come off yet."""
    return [u for u, o in (covers or {}).items() if o not in gone and u not in gone]


def merge_sheets(*sources):
    """(one sheet, the names that were described more than once).

    character_memory and a `Name:` paragraph in the prompt are the same channel by
    two routes, and using both -- the natural thing to do once the widget exists --
    put the person in every shot TWICE:

        A basement. Maya: 27, silver hair, grey coat. Maya: 27, silver hair, grey
        coat. Maya lies still on the floor.

    A model told about one person twice renders two of them. One entry per name, and
    no line repeated. The earlier source wins, so character_memory overrides a sheet
    left in the prompt."""
    seen_names, seen_lines, out, dupes = set(), set(), [], []
    for src in sources:
        for name, line in sheet_lines(src):
            key = name.lower() if name else None
            if key and key in seen_names:
                if name not in dupes:
                    dupes.append(name)
                continue
            if line in seen_lines:
                continue
            if key:
                seen_names.add(key)
            seen_lines.add(line)
            out.append(line)
    return "\n".join(out), dupes


def terminate_lines(text):
    """Give every line a full stop, so what follows does not run into it.

    The sheet is assembled ahead of the beat, and a line ending "grey coat" welds
    onto the beat as "grey coat Maya lies still". A name fused to the end of an
    attribute list reads as one more item in the list -- another person in shot."""
    out = []
    for ln in (text or "").splitlines():
        s = ln.rstrip().rstrip(",;:")
        if s and s[-1] not in ".!?":
            s += "."
        if s:
            out.append(s)
    return "\n".join(out)


def build_scene(anchor, first_para, character_memory, sheet):
    """The text every shot carries, in reading order: the anchor frames the film,
    the opening paragraph sets the scene, and the character sheet says who is in it
    and what they are wearing.

    One string on purpose -- a removal scrubs all of it. The previous node kept the
    anchor immutable, and clothing written there could never be taken off: the
    anchor put it back on every shot, under a beat that had just removed it."""
    parts = [(anchor or "").strip(), (first_para or "").strip(),
             (character_memory or "").strip(), (sheet or "").strip()]
    return "\n".join(terminate_lines(p) for p in parts if p)


_QUOTED = re.compile(r'["“][^"”]+["”]')
# H3's OWN dialogue delimiter. comfy/text_encoders/minimax.py registers <d> and </d>
# as special tokens, alongside a caption channel (<|caption_start|>...) and a lyrics
# one -- so the model distinguishes speech, captions and lyrics explicitly. Text in
# plain quotes is not marked as any of them, and a model with a caption channel is
# entitled to read it as a caption, which renders as text ON the picture.
_DIALOGUE_TAG = re.compile(r"<\s*d\s*>(.+?)<\s*/\s*d\s*>", re.I | re.S)
# Tokens that ASK for text on the frame. If one of these is in the prompt, the
# subtitles are being requested, not invented.
_CAPTION_TOKEN = re.compile(r"<\|(?:caption|lyrics)_(?:start|end)\|>", re.I)


# A shot LONGER than its action does not get filled with more action -- it gets
# filled by performing the same action more slowly, which reads as the whole film
# being in slow motion. Measured: "Maya walks to the window" is a few steps, under
# two seconds of real movement, and the old constants gave it a 4.5s shot.
#
# The base was the larger error. It was meant as setup and settle, but a chained shot
# continues from the previous frame -- it opens mid-scene, with nothing to set up.
BEAT_BASE_SEC = 0.8            # a little room to settle, not a whole beat of it
SECONDS_PER_ACTION = 2.2       # screen time one staged action clause needs
WORDS_PER_SEC = 2.5            # spoken delivery
# A new coordinated verb phrase starts a new action.
_CLAUSE_SPLIT = re.compile(
    r"(?:[.!?;]+|,?\s+(?:and then|then|and|before|after|while|as|until)\s+|,\s+(?=\w+ing\b))")


def beat_seconds(beat):
    """Roughly how much screen time this beat's content asks for.

    Action and dialogue OVERLAP -- people talk while they move -- so it is the
    larger of the two, not the sum. Deliberately rough: the point is not to size
    the shot (the node does not), it is to notice when a shot is much longer than
    anything the beat gives it to do."""
    text = _DIALOGUE_TAG.sub(" ", _QUOTED.sub(" ", beat or ""))
    text = _REMOVE_LINE.sub("", _ADD_LINE.sub("", text))
    clauses = [p for p in _CLAUSE_SPLIT.split(text) if p and len(p.split()) >= 2]
    action = (BEAT_BASE_SEC + SECONDS_PER_ACTION * len(clauses)) if clauses else 0.0
    spoken = sum(len(q.split()) for q in _QUOTED.findall(beat or "")) \
        + sum(len(q.split()) for q in _DIALOGUE_TAG.findall(beat or ""))
    return max(action, (spoken / WORDS_PER_SEC + 1.0) if spoken else 0.0)


MIN_AUTO_FRAMES = 73           # ~3.0s: the shortest shot that can hold one action


def plan_lengths(beats, ceiling_frames, from_beat, pace=1.0):
    """Frames for each shot. Returns (lengths, note).

    'fixed' gives every shot the ceiling. 'from the beat' sizes each shot from what
    its own line stages, capped by that same ceiling and floored at one action's
    worth -- so a beat with one action stops getting a shot with room for two, which
    is what makes an action carry on past its end.

    The estimate leans SHORT deliberately. A shot that ends before its action does
    hands a mid-motion frame to the next shot, and the chain is built to continue
    from exactly that. A shot that outlasts its action does not invent more action --
    it performs the same action more slowly, which is what slow-looking footage is.

    `pace` scales the whole estimate: below 1.0 the shots get shorter and the motion
    in them brisker, above 1.0 they get longer and slower."""
    if not from_beat:
        return [ceiling_frames] * len(beats), ""
    pace = max(0.05, float(pace if pace else 1.0))
    lens = []
    for b in beats:
        need = beat_seconds(b) * pace
        want = align_frame_count_nearest(int(round(need * H3_FPS))) if need else MIN_AUTO_FRAMES
        lens.append(max(MIN_AUTO_FRAMES, min(want, ceiling_frames)))
    note = ""
    if len(set(lens)) > 1:
        note = ("shot lengths are sized from each beat ("
                + ", ".join(f"{n}f/{n / H3_FPS:.1f}s" for n in lens)
                + "). They differ, so one seed does not give them one noise field -- "
                  "noise is drawn to the latent's shape -- and surface detail resets at "
                  "each cut. Set shot_length to 'fixed' if that matters more than pacing")
    return lens, note


def thin_beats(beats, seconds):
    """Beats with far less content than the shot they are given.

    A shot that outlasts its action leaves the model seconds it was told nothing
    about, and the cheapest way to fill them is to CARRY ON: an action that has
    finished its object repeats it on whatever is nearest. Pure arithmetic -- it
    cannot know whether "walks
    across the room" is two seconds or ten, but it can see one action sitting in a
    ten second shot and say so before the render."""
    out = []
    for i, b in enumerate(beats or [], 1):
        need = beat_seconds(b)
        # The GAP matters more than the ratio: "takes off her coat and hangs it up"
        # asks for about 7s, and in a 10s shot the three spare seconds are enough for
        # the action to run on past the thing it was given. A small ratio guard
        # keeps it quiet when the shot only slightly outlasts a long beat.
        if need and (seconds - need) >= 2.5 and seconds > need * 1.25:
            out.append(f"shot {i}: ~{need:.0f}s of content in a {seconds:.0f}s shot")
    return out


# Effort and reaction: the beats where a face has something to do.
_EXERTION = re.compile(
    r"\b(?:thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|writh(?:e|es|ing|ed)|"
    r"strain(?:s|ing|ed)?|fight(?:s|ing)?|kick(?:s|ing|ed)?|jerk(?:s|ing|ed)?|"
    r"gasp(?:s|ing|ed)?|pant(?:s|ing|ed)?|cr(?:y|ies|ying)|sob(?:s|bing|bed)?|"
    r"scream(?:s|ing|ed)?|shout(?:s|ing|ed)?|yell(?:s|ing|ed)?|moan(?:s|ing|ed)?|"
    r"whimper(?:s|ing|ed)?|laugh(?:s|ing|ed)?|flinch(?:es|ing|ed)?|"
    r"trembl(?:e|es|ing|ed)|shak(?:e|es|ing)|shiver(?:s|ing|ed)?|"
    r"freak(?:s|ing)?\s+out|wakes?\s+up|woke\s+up|panic(?:s|king|ked)?)\b", re.I)


def exertion_in(beat):
    """Does this beat stage effort or reaction -- something a face performs?"""
    return bool(_EXERTION.search(beat or ""))


# Sound the text asks for. H3 is joint, so the same prose conditions the audio
# branch -- a scene is scored by describing it, not by a setting.
_SOUND_CUE = re.compile(
    r"\b(?:sounds?|noises?|echo(?:e?s|ing)?|silence|rattl(?:e|es|ing)|clank(?:s|ing)?|"
    r"clink(?:s|ing)?|creak(?:s|ing)?|scrap(?:e|es|ing)|thud(?:s|ding)?|bang(?:s|ing)?|"
    r"slam(?:s|ming)?|clatter(?:s|ing)?|jingl(?:e|es|ing)|squeak(?:s|ing)?|"
    r"footsteps?|breath(?:s|es|ing)?|pant(?:s|ing)?|gasp(?:s|ing)?|sigh(?:s|ing)?|"
    r"whimper(?:s|ing)?|moan(?:s|ing)?|groan(?:s|ing)?|sob(?:s|bing)?|"
    r"scream(?:s|ing)?|shout(?:s|ing)?|whisper(?:s|ing)?|laugh(?:s|ing|ter)?|"
    r"hum(?:s|ming)?|buzz(?:es|ing)?|hiss(?:es|ing)?|drip(?:s|ping)?|"
    r"rustl(?:e|es|ing)|click(?:s|ing)?|snap(?:s|ping)?|zip(?:s|ping)?|"
    r"rings?|ringing|wind|rain|thunder|traffic|music|hollow|muffled|reverb|"
    # How a sound is usually written when the noun is not itself a sound word.
    # "her boots loud on the concrete" describes a sound and named none of the above,
    # so it was read as staging nothing audible and the shot was silenced -- which is
    # the one thing the docs tell you to do to score a silent shot.
    # Adverbs only where the bare adjective describes something other than a sound --
    # "quietly closes the door" is a sound being made, while "the workshop is quiet",
    # "she is quiet" and "a faint smile" are the absence of one or nothing to do with
    # one. Opening the branch on those is a free branch with no line in the shot,
    # which is where an invented voice comes from.
    r"loud(?:ly)?|quietly|faintly|audible|noisy|deafening|"
    r"scuff(?:s|ing|ed)?|crunch(?:es|ing|ed)?|thump(?:s|ing|ed)?|"
    r"patter(?:s|ing)?|whirr?(?:s|ing)?|whine(?:s|d)?|whining|rumbl(?:e|es|ing)|"
    r"growl(?:s|ing)?|roar(?:s|ing)?|chime(?:s|d)?|ticking|"
    r"knock(?:s|ing)?|tap(?:s|ping)?|whoosh(?:es|ing)?|sizzl(?:e|es|ing))\b", re.I)


def sound_described(text):
    """Does this beat ask for a sound the audio branch should make?"""
    return bool(_SOUND_CUE.search(text or ""))


# What a staged action sounds like. The beat already says what happens; the sound it
# makes follows from that, so it does not have to be written twice.
#
# Matched against the BEAT only, never the scene. Sourcing it from the scene as well
# would put a chain rattling into a shot where nobody moves, because the scene says
# there is a chain -- the beat is what decides whether anything makes a noise.
_SOUND_FROM = (
    (r"\b(?:walk(?:s|ed|ing)?|step(?:s|ped|ping)?|pace[sd]?|enters?|runs?|"
     r"approach(?:es|ed)?|creep(?:s|ing)?|crept|sneak(?:s|ing)?|shuffl(?:e|es|ing)|"
     r"stumbl(?:e|es|ing)|stagger(?:s|ing)?|feet)\b",  "footsteps"),
    (r"\bchains?\b",                                "chain links dragging"),
    (r"\b(?:handcuff(?:s|ed)?|cuffs?|cuffed|shackle[sd]?|manacle[sd]?)\b",
                                                    "cuffs knocking"),
    # NOT "locks eyes with her" -- that is a look, and it was giving the shot the
    # sound of a padlock closing.
    (r"\b(?:padlock(?:s|ed)?|locks?|locked|locking)\b(?!\s+(?:eyes|gaze|horns|onto))",
                                                    "a lock snapping shut"),
    (r"\b(?:drag(?:s|ged|ging)?|haul(?:s|ed|ing)?|shov(?:e|es|ing)|slid(?:e|es|ing))\b",
                                                    "something dragging on the floor"),
    (r"\b(?:buckle(?:s|d)?|unbuckle(?:s|d)?|clasp(?:s|ed)?|strap(?:s|ped)?|harness)\b",
                                                    "a buckle and leather creaking"),
    (r"\b(?:pour(?:s|ed|ing)?|water|splash(?:es|ed)?|wet|puddle)\b",
                                                    "water"),
    (r"\b(?:van|car|engine|truck|motor)\b",         "an engine outside"),
    (r"\b(?:fabric|cloth|coat|jacket|shirt|dress|skirt)\b", "fabric rustling"),
    (r"\b(?:scissors|shears|cut(?:s|ting)?)\b",     "blades through fabric"),
    (r"\bdoors?\b",                                 "a door on its hinges"),
    (r"\b(?:drops?|dropped|throw(?:s|n)?|threw|toss(?:es|ed)?)\b",
                                                    "something landing"),
    (r"\b(?:smack(?:s|ed)?|slap(?:s|ped)?|hits?|strikes?|struck)\b", "a sharp impact"),
    # Only where there is something to pull against. "McKenna thrashes on the bed"
    # was getting restraints she is not wearing, because the verb alone armed it.
    (r"(?=[\s\S]*\b(?:cuffs?|handcuffs?|shackles?|manacles?|chains?|ropes?|cords?|"
     r"straps?|restraints?|bindings?|ties?|tape|harness|collar)\b)"
     r"\b(?:thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|writh(?:e|es|ing|ed)|"
     r"strain(?:s|ing|ed)?|pull(?:s|ing|ed)?\s+against)\b",
                                                    "restraints pulling taut"),
    # A body under effort makes a VOICE, not only movement. H3 is joint, so this is
    # also what stops the face going flat: conditioning the audio on silence tells the
    # model the person makes no sound, and a person making no sound is rendered still.
    # A beat that already names the sound is left alone -- "she moans" is in
    # _SOUND_CUE, so what you wrote wins and none of this is added.
    (r"\b(?:thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|writh(?:e|es|ing|ed)|"
     r"strain(?:s|ing|ed)?|arch(?:es|ed|ing)?|shudder(?:s|ed|ing)?|"
     r"trembl(?:e|es|ing|ed)|shiver(?:s|ed|ing)?|buck(?:s|ed|ing)?|"
     r"grind(?:s|ing)?|rock(?:s|ed|ing)?|thrust(?:s|ing)?|"
     r"clutch(?:es|ed|ing)?|grip(?:s|ped|ping)?|clench(?:es|ed|ing)?)\b",
                                                    "unsteady breathing, with gasps and "
                                                    "moans of effort"),
    (r"\b(?:zip(?:s|ped|ping)?|unzip(?:s|ped|ping)?)\b", "a zip running"),
    (r"\btap(?:e|es|ed|ing)\b",                     "tape pulling off"),
    (r"\b(?:wakes?\s+up|woke|gasp(?:s|ing)?|pant(?:s|ing)?|breath(?:es|ing)?)\b",
                                                    "breathing"),
)
MAX_SOUNDS = 3      # a shot's audio needs a cue, not an inventory

# The SPACE, as opposed to the things in it. Read from the scene, and this is the one
# thing that safely can be: a chain standing in the scene must not rattle in a shot
# where nobody moves, but a concrete room is hard in every shot whatever happens in
# it. That is the difference between a recording and a sound effect -- real footage
# has a bed under the events, and digital silence between them is what makes a scene
# sound staged.
_ROOM_TONE = (
    (r"\b(?:bathroom|shower|tiled?|tiles)\b",       "tiled walls ringing"),
    (r"\b(?:basement|cellar|warehouse|garage|hangar|tunnel|stairwell|"
     r"corridor|concrete|stone|brick|bare walls?)\b", "hard walls giving the sound back"),
    # "shallow depth of field" and "field of view" are the LENS, not a location.
    # Every anchor written for this node says one of them, so every interior scene
    # was being told it sounds like open air.
    (r"\b(?:outside|outdoors|street|road|yard|garden|forest|beach|park)\b"
     r"|(?<!depth of )\bfield\b(?! of view)",       "open air with no walls close by"),
    (r"\b(?:carpet(?:ed)?|curtains?|bedroom|sofa|cushions?)\b",
                                                    "a soft room with little echo"),
    (r"\b(?:barn|attic|loft|shed|workshop|hall|church)\b", "a large room with a long tail"),
)


def room_tone(scene, opening=""):
    """How the space itself sounds. One room, one acoustic -- the first match wins.

    `opening` is the first beat, and it is read only when the scene names no space at
    all. With `anchor` set there is no scene PARAGRAPH -- the anchor is the whole of
    it -- and an anchor describes the CAMERA, not the room. The location is then
    written in the first beat, so reading nothing but the lens line left the acoustic
    to be guessed from words like "depth of field"."""
    for text in (scene, opening):
        for pat, phrase in _ROOM_TONE:
            if re.search(pat, text or "", re.I):
                return phrase
    return ""


# Sounds that are a THING IN MOTION, and which thing. H3 is joint: the prose
# conditions the audio branch and the picture follows the audio, so "a door on its
# hinges" is not a decoration on a shot with a door in it -- it is a request for a
# door to swing. Asked for beside a sentence holding that same door shut, the sound
# wins, because it describes something happening and the hold describes something
# not happening.
#
# Reported exactly that way: the doors started closed, as the hold asked, and were
# then opened. Two guards, one contradicting the other.
_SOUND_OF_MOVING = {"a door on its hinges": ("door",)}


def sounds_for(beat, held=()):
    """The sounds this beat's own action implies. [] when it stages nothing audible.

    `held` is the scenery this shot is holding still. A sound of one of those moving
    is dropped: the shot cannot be asked to keep the doors shut and to sound like a
    door swinging."""
    held = set(held or ())
    out = []
    for pat, phrase in _SOUND_FROM:
        if len(out) >= MAX_SOUNDS:
            break
        if held.intersection(_SOUND_OF_MOVING.get(phrase, ())):
            continue
        if phrase not in out and re.search(pat, beat or "", re.I):
            out.append(phrase)
    return out


def sound_clause(phrases, only=False):
    """One sentence naming what the shot is heard as.

    `only` closes the list. H3 is joint, so the audio branch drives the face: a shot
    whose audio is left free but only loosely described will fill the rest with a
    VOICE, and the mouth moves to it in a shot that has no line. Saying these are the
    only sounds leaves nothing for a voice to fill.

    Positively phrased, because that is the only phrasing this model gets: at cfg 1
    H3 is CFG-free and no negative prompt is evaluated, so "nobody speaks" is not a
    prohibition, it is the word "speaks" in the prompt. "The only sound is X" excludes
    speech by saying what IS there.

    Plain prose, and deliberately not a labelled line: `sound:` at the start of a
    line is read as text to DRAW and turns up on screen, which is the whole reason
    the old node's field labels had to be stripped out."""
    if not phrases:
        return ""
    if len(phrases) == 1:
        heard = phrases[0]
    else:
        heard = ", ".join(phrases[:-1]) + " and " + phrases[-1]
    if only:
        verb = "is" if len(phrases) == 1 else "are"
        return f" The only sound{'' if len(phrases) == 1 else 's'} {verb} {heard}."
    return f" It sounds like {heard}."


def has_speech(beat):
    """Does this beat contain a scripted line?

    Either H3's own <d>...</d> marker or plain double quotes. Only checking quotes
    meant a beat written the way the model expects was treated as silent, and its
    audio muted."""
    text = beat or ""
    return bool(_DIALOGUE_TAG.search(text) or _QUOTED.search(text))


_PICTURE_TAG = re.compile(r"<\s*picture[\s_\-]*(\d+)\s*>", re.I)


def picture_tags(text):
    return sorted({int(m.group(1)) for m in _PICTURE_TAG.finditer(text or "")})


def resolve_tags(text, ref_list):
    """(text with its tags renumbered, the images that shot carries, dropped slots).

    A <Picture N> tag is the BINDING between an image and the subject the prompt
    describes, and it belongs IN the prompt. comfy_extras/nodes_minimax_h3.py says so
    outright: "Ordinals are 1-based per type, so the prompt refers to them as
    <Picture i>", and the node's own description is "Use the same tags when
    prompting."

    The rule that follows governs every reference decision in this file:

        a picture the prompt REFERS TO is that subject;
        a picture the prompt does NOT refer to is ANOTHER subject.

    So taking a tag out of the text does not remove a spare person, it CREATES one --
    the image arrives labelled and unclaimed, and the model renders it as somebody
    else. It is also why the handoff frame must not enter this channel at all: no
    wording refers to it, so it would arrive as a stranger.

    comfy/text_encoders/minimax.py writes the "<Picture N>: " label itself, numbering
    by the order it receives the images -- so a shot that uses only <Picture 2>
    receives that image labelled <Picture 1>, and text still saying <Picture 2> points
    at nothing. The tags are renumbered per shot to match what the shot actually
    carries: slot 2 alone becomes <Picture 1>; slots 2 and 4 become <Picture 1> and
    <Picture 2>.

    A tag naming a slot with no image connected refers to nothing at all, so it is
    removed from the text rather than left for the encoder to puzzle over."""
    wanted = picture_tags(text)
    live = [n for n in wanted if 1 <= n <= len(ref_list or [])]
    dropped = [n for n in wanted if n not in live]
    renum = {old: new for new, old in enumerate(live, 1)}

    def sub(m):
        n = int(m.group(1))
        return f"<Picture {renum[n]}>" if n in renum else ""

    out = _PICTURE_TAG.sub(sub, text or "")
    out = re.sub(r"\s+([,.;:])", r"\1", out)      # " ," left by a removed tag
    out = re.sub(r"([:,;])\s*,", r"\1", out)      # ",," where the tag was the only item
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip(), [ref_list[n - 1] for n in live], dropped

def check_audio_vae_loaded(audio_vae):
    """Catch an UNCONVERTED audio VAE checkpoint.

    comfy/ldm/minimax/audio_vae.py loads a checkpoint whose weight-norm has been
    folded into plain "*.weight" tensors. Feed it the raw upstream file (172
    weight_g/weight_v pairs, no latents_mean/latents_std) and load_state_dict
    reports the misses as a WARNING, not an error: every weight-normed conv keeps
    its random init and the two normalization buffers stay torch.empty(), i.e.
    uninitialized memory. Decoding then multiplies the latents by garbage and the
    audio comes out as noise -- with nothing in the log at render time to say why.

    latents_std is the cheapest tell: it is a real per-channel scale, so a
    non-finite or absurd value means the buffer was never filled."""
    m = getattr(audio_vae, "first_stage_model", None)
    mean, std = getattr(m, "latents_mean", None), getattr(m, "latents_std", None)
    if mean is None or std is None:
        return
    try:
        bad = (not torch.isfinite(mean).all() or not torch.isfinite(std).all()
               or float(std.min()) <= 0.0 or float(std.max()) > 1e3
               or float(mean.abs().max()) > 1e3)
    except Exception:
        return                       # never block a render on a failed introspection
    if bad:
        raise RuntimeError(
            "the audio VAE loaded but its weights are NOT initialized -- this is the raw "
            "upstream MiniMax-H3 audio checkpoint (weight_g/weight_v weight-norm pairs, no "
            "latents_mean/latents_std). ComfyUI's loader needs the CONVERTED file, with "
            "weight-norm folded into plain '*.weight' tensors. Look for the 'Missing VAE keys' "
            "warning in the log when the VAE loaded. Download the repackaged H3 audio VAE from "
            "the Comfy-Org release; rendering with this one produces noise, not speech.")


def ref_image_canvas(w, h, gen_w, gen_h, mode="match"):
    """Pure: the (width, height) a reference image is encoded at.

    'match' scales it (DOWN only, aspect kept) to the generation's pixel area, so a
    reference costs about as much as one frame of the shot. 'max' goes to the
    reference pipeline's 2048 short edge for the best identity fidelity, which on a
    long chain is several times slower because the rows are re-attended every step
    of every shot. Never upscales: a small reference stays small."""
    w, h = max(1, int(w)), max(1, int(h))
    if mode == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    else:
        scale = min(1.0, math.sqrt((int(gen_w) * int(gen_h)) / float(w * h)))
    snap = lambda v: max(CANVAS_MULTIPLE, round(v * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return snap(w), snap(h)


def shot_latent_cells(w, h, frames, fps):
    """Latent cells in one shot: what sampling VRAM actually scales with.

    Not a byte figure -- the constant depends on the quantisation path -- but it is
    exactly linear in both shot length and area, so ratios between settings are
    right even though the absolute number is not a prediction."""
    _, lt, _ = temporal_shape(frames, fps)
    return max(1, int(lt)) * max(1, w // 16) * max(1, h // 16)


def model_fingerprint(model):
    """A cheap, stable identity for the loaded DiT: (quant format, layer count,
    weight bytes, class name). Changes whenever the checkpoint changes -- a
    different quant, a pruned-vs-full build, or a different model entirely -- while
    staying identical across shots of the same run. Deliberately avoids hashing
    weights, which would cost more than the flush it guards."""
    try:
        dm = getattr(getattr(model, "model", None), "diffusion_model", None)
        fmts, n = {}, 0
        if dm is not None and hasattr(dm, "modules"):
            for mod in dm.modules():
                n += 1
                f = getattr(mod, "quant_format", None)
                if f:
                    fmts[f] = fmts.get(f, 0) + 1
        top = max(fmts.items(), key=lambda kv: kv[1])[0] if fmts else "none"
        size = 0
        try:
            size = int(model.model_size())
        except Exception:
            pass
        cls = type(dm).__name__ if dm is not None else "unknown"
        return (top, n, size, cls)
    except Exception:
        return None



# --- H3 plumbing, carried over unchanged: these were arrived at against the real
# model and the real VAEs, and none of it is prompt logic.

def _resize(image, width, height, crop):
    s = image[..., :3].movedim(-1, 1)
    s = comfy.utils.common_upscale(s, width, height, "lanczos", crop)
    return s.movedim(1, -1)


def _empty_av_latent(width, height, length, fps, batch_size=1):
    fc, lt, at = temporal_shape(length, fps)
    video = torch.zeros([batch_size, 24, lt, height // 16, width // 16], device=mm.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, at], device=mm.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, fc


def _auto_tile_t(n_latent_frames, requested=None):
    """Temporal tile for a tiled decode. An explicit value wins.

    The decode_tile_frames widget is gone, so this is where the value comes from
    now. It has to come from somewhere: ComfyUI's decode_tiled_3d defaults tile_t
    to 999, i.e. SPATIAL tiles only, and expanding the whole clip's time axis at
    once is the single largest allocation in a run. A "tiled" decode that keeps the
    full temporal extent barely lowers the peak, so the OOM retry that switches
    tiling on was, without this, retrying with almost the same footprint."""
    if requested:
        return int(requested)
    n = int(n_latent_frames or 0)
    return AUTO_TILE_T if n > AUTO_TILE_T else None


def _decode_video(vae, out_latent, tiled, free_first=None, tile_t=None, tile_xy=None,
                  keep=()):
    """Decode the video latent.

    `free_first` is the diffusion model: sampling is finished, and the video VAE
    needs the room for THIS decode -- the free runs immediately before it, not to
    make room for the next shot. On a card where the DiT is most of the VRAM, the
    decode does not fit until it goes.

    `keep` is what must NOT be evicted on the way. It was `keep_loaded=[]`, which
    unloaded every resident model -- including the video VAE, which ComfyUI then
    reloaded three lines later to run the decode. An evict-and-reload of the thing
    about to be used, once per shot, on every card. Peak VRAM is identical either
    way, since the VAE has to be resident to decode; the round trip was pure cost.

    Note what is NOT changed here: memory_required is still 1e30, which skips
    ComfyUI's partially_unload path (model_management.py:811) and forces a full
    detach + unpatch of everything else. That is the expensive half on a card with
    headroom, and it cannot be sized honestly without measuring on real hardware."""
    if free_first is not None:
        try:
            mm.free_memory(1e30, mm.get_torch_device(),
                           keep_loaded=_resident(keep or (vae,)))
        except Exception:
            pass
    latent = out_latent["samples"]
    if latent.is_nested:
        latent = latent.unbind()[0]
    if tiled:
        # Temporal + spatial tiling. Without tile_t the VAE expands the WHOLE latent
        # clip at once, which on a 243-frame 1344x768 shot is the single largest
        # allocation in the run -- and on an unpruned checkpoint that is already
        # streaming, it is what tips the card over. Decoding in temporal chunks
        # trades a little speed for a much lower peak; None keeps ComfyUI's defaults.
        args = {}
        tile_t = _auto_tile_t(latent.shape[2] if latent.ndim >= 5 else 0, tile_t)
        if tile_t:
            args["tile_t"] = int(tile_t)
            args["overlap_t"] = max(1, int(tile_t) // 8)
        if tile_xy:
            args["tile_x"] = int(tile_xy)
            args["tile_y"] = int(tile_xy)
        try:
            imgs = vae.decode_tiled(latent, **args) if args else vae.decode_tiled(latent)
        except TypeError:
            imgs = vae.decode_tiled(latent)      # older signature without tile_t
    else:
        imgs = vae.decode(latent)
    if len(imgs.shape) == 5:
        imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
    return imgs


def _decode_audio(audio_vae, out_latent):
    latent = out_latent["samples"]
    if latent.is_nested:
        latent = latent.unbind()[-1]
    audio = audio_vae.decode(latent).movedim(-1, 1)
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std
    sr = getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 44100))
    return {"waveform": audio, "sample_rate": sr}


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _deep_cleanup():
    """Release VRAM + RAM between shots so a long chain doesn't accumulate and OOM.
    Runs a Python GC pass (frees dereferenced tensors / CPU buffers), then empties
    the CUDA allocator's cached blocks and IPC handles. Cheap relative to sampling;
    called once per beat.

    It unloads NOTHING. soft_empty_cache(force) ignores `force` in current ComfyUI
    (model_management.py:2050) -- the body only reaches empty_cache() and
    ipc_collect() -- so this drops cached blocks, not models. The `True` is kept
    only for older builds that read it; the older comment here claimed this took an
    unload_all_models path, and it does not."""
    gc.collect()
    try:
        mm.soft_empty_cache(True)
    except TypeError:
        mm.soft_empty_cache()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _resident(models):
    """The LoadedModel entries ComfyUI currently holds for `models`.

    That is the form free_memory's keep_loaded wants: it compares against the
    entries in current_loaded_models, not against the ModelPatcher objects a node
    is holding. Anything not matched is simply not kept, so a model that is not
    resident costs nothing here."""
    out = []
    for lm in list(getattr(mm, "current_loaded_models", [])):
        for m in models or ():
            if m is None:
                continue
            try:
                if lm.model is m or getattr(lm, "model", None) is getattr(m, "model", None):
                    if lm not in out:
                        out.append(lm)
            except Exception:
                pass
    return out


def _evict_all_but(keep_model):
    """Unload every model EXCEPT the diffusion model from the GPU.

    This is the fix for VRAM ratcheting across a long chain. soft_empty_cache()
    only drops the CUDA allocator's cached blocks -- it does NOT unload models, so
    ComfyUI keeps the Qwen3-VL text encoder (~14.6GB) and both VAEs resident in
    current_loaded_models alongside the DiT. Each shot re-encodes the prompt
    (text encoder), encodes the handoff keyframe (video VAE), then samples (DiT),
    so all three compete for the card.

    ComfyUI does free ahead of each load -- load_models_gpu() calls free_memory()
    for what it is about to need (model_management.py:975), so the weight path is
    not purely reactive. What it cannot size for is a long chain's ACTIVATIONS on
    a card where the DiT is most of the VRAM. Freeing explicitly, right after
    conditioning is built and before sampling, keeps only what the sampler needs.

    Still 1e30, deliberately: see the note in _decode_video."""
    try:
        mm.free_memory(1e30, mm.get_torch_device(),
                       keep_loaded=_resident([keep_model]))
    except Exception:
        try:
            mm.soft_empty_cache(True)
        except Exception:
            pass


def check_vae_wiring(vae, audio_vae):
    """Catch the commonest miswire -- the video VAE dropped into BOTH VAE inputs.
    Without this the run samples a whole shot, decodes the video fine, then dies
    deep inside comfy/sd.py with 'IndexError: tuple index out of range' when the
    video memory estimator indexes shape[4] of the 4-D audio latent."""
    if _is_audio_vae(audio_vae) is False:
        raise RuntimeError(
            "audio_vae is a video/image VAE, not the H3 audio VAE. Load the audio "
            "autoencoder (the DAC/BigVGAN one shipped with MiniMax-H3, e.g. "
            "minimax_h3_audio_vae.safetensors) in its own VAELoader and wire that "
            "into 'audio_vae'; the video VAE belongs on 'vae' only.")
    check_audio_vae_loaded(audio_vae)
    if _is_audio_vae(vae) is True:
        raise RuntimeError(
            "vae is the H3 audio VAE -- the video and audio VAE inputs are swapped. "
            "Wire the video VAE into 'vae' and the audio VAE into 'audio_vae'.")


def flush_for_model_change(model):
    """Detect a checkpoint swap since the last run and, if one happened, hard-flush
    GPU state before doing anything else.

    Why this matters: ComfyUI keeps previously-loaded models in current_loaded_models
    and only evicts reactively. Swapping checkpoints mid-session (e.g. NVFP4 -> FP8 ->
    MXFP8 while comparing quality) leaves the OLD DiT resident alongside the new one,
    plus any hooks/injections a previous LoRA installed and stale cached allocator
    blocks sized for the old model's layers. The result is a card that is already
    half full before the first shot samples -- which looks exactly like the node
    over-spilling, when in fact the budget was computed against memory the previous
    checkpoint never released.

    Returns a note for `info` when a change was detected (empty string otherwise)."""
    fp = model_fingerprint(model)
    prev = _LAST_MODEL_FP.get("fp")
    _LAST_MODEL_FP["fp"] = fp
    if prev is None or fp is None or prev == fp:
        return ""
    try:
        mm.unload_all_models()          # drop every resident model, not just the cache
    except Exception:
        pass
    # Never let a cleanup failure abort the run: the flush is best-effort hygiene,
    # and a partially-flushed card is still better than raising here.
    for _ in range(2):                  # 2nd pass frees blocks released by the 1st
        try:
            _deep_cleanup()
        except Exception:
            pass
    old_fmt, _n, old_sz, _c = prev
    new_fmt = fp[0]
    return (f"model changed since last run ({old_fmt} ~{old_sz / GB:.1f}GB -> {new_fmt} "
            f"~{fp[2] / GB:.1f}GB): flushed all resident models and VRAM caches")


def _silent_audio_latent(audio_vae, frame_count, fps):
    """A keyframe audio latent of actual SILENCE, or None if it cannot be made.

    H3 is a JOINT model: the mouth follows the audio branch. On a shot with no
    scripted line the branch is otherwise unconditioned, and an unconditioned audio
    branch invents a voice -- which the picture then lip-syncs to. The lips-closed
    sentence is arguing with a stream that has already decided someone is talking.

    Seeding the keyframe's audio channel with encoded silence anchors that stream
    instead. comfy/ldm/minimax/audio_vae.py encode() takes stereo [B, 2, L] at
    32 kHz and returns [B, 32, 2, T] on the same 40 Hz grid temporal_shape() uses.

    Everything here is defensive. The shape is CHECKED against what the layout
    expects rather than assumed, and any failure returns None so the shot falls
    back to today's behaviour instead of breaking the render."""
    try:
        sr = int(getattr(audio_vae, "audio_sample_rate", 0) or 0)
        if sr <= 0:
            return None
        _, _, want_t = temporal_shape(frame_count, fps)
        if want_t <= 0:
            return None
        unit = _SILENT_UNIT.get("lat")
        if unit is None:
            # CHANNELS LAST. comfy.sd.VAE.encode() does `pixel_samples.movedim(-1, 1)`
            # before handing off, so the audio VAE -- which wants [B, 2, L] -- must be
            # given [B, L, 2]. Passing [B, 2, L] raises inside the encoder, and the
            # first version did exactly that: swallowed by the guard below, so the
            # whole layer silently did nothing.
            #
            # ONE SECOND, encoded ONCE. Silence is homogeneous, so the result tiles
            # along time -- and encoding a full 15s shot instead cost a VAE pass big
            # enough to OOM mid-render on a 16GB card, where the failure again
            # degraded silently to no conditioning at all.
            enc = audio_vae.encode(torch.zeros((1, sr, 2)))
            if enc is None or enc.dim() != 4 or enc.shape[1] != 32 or enc.shape[-1] < 3:
                return None
            # ONE STEADY FRAME, from the MIDDLE. The encoder's zero-padding leaves
            # heavy edge artifacts -- measured deviation 0.351 at the first and last
            # frames against 0.002 in the interior, ~170x -- and tiling the whole
            # second therefore stamped a spike every 40 latent frames, which at 40Hz
            # is once per SECOND. That is a metronome in the audio conditioning of a
            # joint audio-video model, and the picture lip-syncs to it. Repeating a
            # single interior frame gives conditioning that is genuinely constant.
            mid = enc.shape[-1] // 2
            _SILENT_UNIT["lat"] = enc[..., mid:mid + 1].detach().to("cpu").clone()
        unit = _SILENT_UNIT["lat"]
        if unit.shape[-1] != 1:
            return None
        return unit.repeat(1, 1, 1, want_t).clone()
    except Exception:
        return None                         # never fail a render for a nicety


_POSTURE = re.compile(
    r"\b(?:lying|laying|lies|lays|kneel(?:s|ing)?|knelt|sit(?:s|ting)?|sat|"
    r"crouch(?:es|ing|ed)?|curled|sprawled|slumped|face[- ]?down|face[- ]?up|"
    r"on (?:her|his|their) (?:side|back|front|knees|stomach|belly))\b", re.I)


def posture_note(scene, has_first_frame):
    """Warn when shot 1's opening pose is left to the text alone.

    Shot 1 is the only shot with no keyframe -- there is no previous frame to
    continue from -- so its opening pose comes from the text and from nothing else.
    A posture sentence sitting at the end of a long sheet is the least-weighted
    thing the model reads, and text cannot outrank a picture anyway. This does not
    reorder anything: the node sends what you wrote, in the order you wrote it."""
    if has_first_frame or not (scene or "").strip():
        return ""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", scene.strip()) if s.strip()]
    where = [i for i, s in enumerate(sents) if _POSTURE.search(s)]
    if not where:
        return ""
    return (f"shot 1 has no keyframe, so its opening pose comes from the text alone -- "
            f"and the sentence describing the pose is {where[0] + 1} of {len(sents)}. "
            f"first_frame pins it, but it pins the WHOLE opening frame, so it has to be "
            f"a composed frame of the shot you want: a head-and-shoulders picture wired "
            f"there makes the first frame a head-and-shoulders picture. An identity "
            f"portrait belongs on ref_image_1 instead")


def reference_note(n_refs, aug, has_first_frame):
    """What a near-clean reference actually asks the model to do.

    ONE aug covers every visual conditioning row. At H3's default of 0.999 a
    reference is handed over essentially noise-free, and a noise-free image is an
    invitation to REPRODUCE it -- its framing and background along with its subject.
    That is a matter of DEGREE, not a format error, and this is the dial for it: the
    symptom is a shot that opens on the reference and moves off it, and the answer is
    to lower the aug until it informs the face without being copied.

    Shot 1 is where it shows most, because it has no keyframe pinning its opening
    frame -- the reference is the only picture it has, so there is nothing competing
    with the invitation to reproduce."""
    if not n_refs or aug is None:
        return ""
    if float(aug) >= KEYFRAME_SAFE_AUG:
        note = (f"{n_refs} reference image(s) at ref_noise_aug {float(aug):.3f}, which is "
                f"near-clean -- that asks the model to REPRODUCE them, framing and "
                f"background included, in the opening frames. Lower it to say "
                f"approximate: try 0.95, then 0.90. Below 0.99 the handoff stops being "
                f"a keyframe and rides as an extra reference, so continuity weakens as "
                f"identity strengthens")
    else:
        note = (f"{n_refs} reference image(s) at ref_noise_aug {float(aug):.3f} -- "
                f"softened, so they inform the face rather than being copied. Below "
                f"0.99 one aug would also soften the keyframe, so the handoff rides as "
                f"an extra reference instead of anchoring: weaker continuity, nothing "
                f"pretending to anchor while carrying noise")
    if not has_first_frame:
        note += (". Shot 1 has no keyframe, so the reference is its only picture and "
                 "nothing competes with reproducing it -- that shot is where a "
                 "near-clean reference shows up as the opening frame")
    return note


def frame_detail(img):
    """(detail, contrast) for one frame in 0..1, HWC.

    Detail is mean absolute neighbour difference -- a cheap stand-in for how much
    fine structure survives. Contrast is the luminance spread. Neither is an
    absolute measure of anything; what matters is the TREND across shots.

    Every shot boundary decodes a latent to pixels, takes the last frame and
    re-encodes it as the next shot's keyframe. That round trip is lossy, and the
    frame it runs on is the model's own output, so shot 11 is sampled from a
    picture that has been through ten decode/encode cycles. Softening that
    compounds is invisible shot to shot and obvious end to end -- so measure it."""
    x = img.float()
    if x.dim() == 3 and x.shape[-1] >= 3:
        x = x[..., :3].mean(dim=-1)
    elif x.dim() == 3:
        x = x[..., 0]
    if x.dim() != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return 0.0, 0.0
    gx = (x[:, 1:] - x[:, :-1]).abs().mean()
    gy = (x[1:, :] - x[:-1, :]).abs().mean()
    return float((gx + gy) * 0.5), float(x.std())


def detail_report(per_shot):
    """One line saying whether the chain is softening, and by how much.

    per_shot is [(detail, contrast), ...] measured on each shot's last frame."""
    vals = [d for d, _ in per_shot if d > 0]
    if len(vals) < 2:
        return ""
    first, last = vals[0], vals[-1]
    drop = (first - last) / first * 100.0 if first else 0.0
    trend = " ".join(f"{d:.4f}" for d, _ in per_shot)
    line = f"detail per shot (last frame): {trend}"
    if drop >= 10.0:
        line += (f" -- DOWN {drop:.0f}% from shot 1 to shot {len(vals)}. Each boundary "
                 f"decodes a shot, takes its LAST frame and re-encodes it as the next "
                 f"shot's keyframe, so the loss of one round trip is carried into the "
                 f"next and compounds. Break the chain to stop it accumulating: "
                 f"restart_after_removal starts a shot from the text instead of the "
                 f"previous frame, at the cost of a visible cut there")
    elif drop <= -10.0:
        line += f" -- UP {-drop:.0f}%, so the chain is not softening"
    else:
        line += f" -- flat within {abs(drop):.0f}%"
    return line


def _keyframe_latent(vae, hand_img):
    """The keyframe latent for this shot: an ENCODE of the previous shot's last frame.

    This was briefly an optimisation -- pass the previous shot's own latent straight
    through and skip a VAE round trip per boundary. It was wrong, and it degraded
    every shot after the first.

    A keyframe is ONE pixel frame, and H3's grid puts that at 5f -> TWO latent
    frames. Slicing [:, :, -1:] off a finished shot hands over one. Worse, the video
    VAE is causal: the last latent of a 72-frame sequence encodes its temporal
    context, not a standalone opening frame, so even at the right count it does not
    mean what a keyframe means. The spatial-size guard could not see either problem.

    The round trip is real but it is one lossy step on a correctly formed anchor,
    which beats a cheap malformed one."""
    return vae.encode(hand_img)


def _build_ref_images(vae, images, gen_w, gen_h, mode="match"):
    """(tokenizer items, DiT blocks) for a list of reference IMAGE tensors.

    The tokenizer labels each one `<Picture N>:` itself, in the order given here --
    so the roster the prompt refers to is decided by input order, not by anything
    written in the prompt."""
    items, blocks = [], []
    for img in images:
        if img is None:
            continue
        h, w = int(img.shape[1]), int(img.shape[2])
        tw, th = ref_image_canvas(w, h, gen_w, gen_h, mode)
        resized = _resize(img[:1], tw, th, "disabled")
        items.append({"type": "image", "data": resized})
        blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                       "latent": vae.encode(resized)})
    return items, blocks


def _sample_on_sigmas(model, seed, cfg, sampler_name, positive, negative, latent, sigmas):
    """common_ksampler, driven by an EXTERNAL sigma schedule.

    common_ksampler derives its sigmas from (sampler_name, scheduler, steps, denoise)
    and takes no schedule argument, so a schedule computed anywhere else cannot
    reach it. Under PDD that is fatal rather than merely inconvenient: the heads
    accept only their nine trained boundaries, and re-deriving the grid from
    widgets means hitting it by coincidence and losing it again the moment a step
    count changes.

    Mirrors nodes.common_ksampler's noise / mask / callback handling exactly -- the
    only substitution is comfy.sample.sample_custom for comfy.sample.sample."""
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None))
    noise = comfy.sample.prepare_noise(latent_image, seed, latent.get("batch_index"))
    # `steps` here only sizes the progress bar -- the schedule is `sigmas`, whose
    # step count is one less than its length (the trailing 0.0 is an endpoint).
    callback = latent_preview.prepare_callback(model, max(len(sigmas) - 1, 1))
    samples = comfy.sample.sample_custom(
        model, noise, cfg, comfy.samplers.sampler_object(sampler_name), sigmas,
        positive, negative, latent_image,
        noise_mask=latent.get("noise_mask"), callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


def _find_h3_sampling_node():
    """Locate the H3 sigma-shift node under ANY registered name. It was renamed
    to 'ModelSamplingMiniMaxH3' in a later patch (kijai PR #15243); older 0.30.x
    builds register it under a different id, so exact-key lookup misses it. Try
    the known names, then fuzzy-scan all node mappings for the H3 model-sampling
    node. Returns (class, key) or (None, None)."""
    maps = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    for key in ("ModelSamplingMiniMaxH3", "ModelSamplingMinimaxH3", "ModelSamplingMinimax", "ModelSamplingH3"):
        if key in maps:
            return maps[key], key
    for k, v in maps.items():
        kl = k.lower()
        if "sampl" in kl and (("minimax" in kl and "h3" in kl) or ("h3" in kl and "shift" in kl)):
            return v, k
    for k, v in maps.items():
        kl = k.lower()
        if ("minimax" in kl or "h3" in kl) and ("shift" in kl or "sampling" in kl):
            return v, k
    return None, None


def _direct_model_sampling(model, shift_video, shift_audio):
    """Fallback that sets the shift on the model's own model_sampling object
    without any node -- version-tolerant and V3-proof, since it uses model-level
    APIs (get_model_object / set_parameters / add_object_patch) rather than
    calling a node. Copies the sampling object so the base model isn't mutated,
    and applies audio_shift only if the installed set_parameters accepts it."""
    import inspect, copy
    m = model.clone()
    # deepcopy, not copy: model_sampling is an nn.Module, and a SHALLOW copy shares
    # its `_buffers` dict with the original. set_parameters() re-registers `sigmas`
    # into that shared dict, so a shallow copy silently rewrites the BASE model's
    # sigma table -- the very thing this copy exists to prevent. Our own run reads
    # the patched object either way, but ComfyUI caches the model across queue
    # runs, so the damage outlives this execution and reaches anything else holding
    # that model. The buffer is ~1000 floats; the deepcopy is free.
    ms = copy.deepcopy(m.get_model_object("model_sampling"))
    sig = inspect.signature(ms.set_parameters)
    kwargs = {}
    if "shift" in sig.parameters:
        kwargs["shift"] = float(shift_video)
    if "audio_shift" in sig.parameters:
        # NOTE: on ComfyUI 0.31 the audio latent is carried on the video schedule
        # scaled by audio_scale = shift_video / shift_audio (12/3 = 4.0), applied in
        # process_latent_in and undone in process_latent_out. Forcing that ratio to
        # 1.0 (audio_shift == shift_video) as a "legacy 0.30" emulation produces
        # SILENT output -- the model needs the scaling -- so it is not offered.
        kwargs["audio_shift"] = float(shift_audio)
    if not kwargs:
        raise RuntimeError("set_parameters takes no shift")
    ms.set_parameters(**kwargs)
    m.add_object_patch("model_sampling", ms)
    return m


def apply_h3_model_sampling(model, shift_video, shift_audio):
    """Apply H3's dual video/audio flow schedule from INSIDE the node so a missing
    upstream patch can't silently gibberish the audio.

    On ComfyUI 0.31+ the H3 nodes are V3-schema and don't live in the legacy
    NODE_CLASS_MAPPINGS the old way -- AND the model already defaults to the correct
    FLOW_AV schedule (12/3) at load. So the reliable path here is a DIRECT model-
    level patch (works regardless of node API); the node call is only a secondary.
    Order: direct model_sampling patch -> node under any name (V1/V3) -> give up with
    an informative, non-alarming note. Shifts aren't hardcoded (12/3 base, ~8 video
    for low-step MXFP8, ~4-6 audio for turbo)."""
    try:
        return _direct_model_sampling(model, shift_video, shift_audio), \
               f"model_sampling video {shift_video:g}/audio {shift_audio:g} (direct)"
    except Exception:
        pass
    cls, key = _find_h3_sampling_node()
    if cls is not None:
        try:
            return _call_node(cls, model, shift_video, shift_audio), \
                   f"model_sampling video {shift_video:g}/audio {shift_audio:g} (via {key})"
        except Exception:
            pass
    return model, (f"model_sampling not explicitly set (video {shift_video:g}/audio {shift_audio:g}); "
                   "on ComfyUI 0.30+ the model already defaults to the correct schedule, so this is "
                   "usually harmless -- only set shift_video/audio explicitly if you're on a low-step "
                   "MXFP8/turbo profile and the audio sounds wrong")


def sampling_oom_help(w, h, frames, fps, megapixels=0.0):
    """What to change, in this shot's own numbers, after a SAMPLING OOM.

    Tiling is a decode setting and cannot help here, so the generic "try tiling"
    advice is worse than useless -- it costs another full sampling pass before
    failing the same way. Give the two levers that do change sampling cost, each
    priced from the shot that just failed."""
    now = shot_latent_cells(w, h, frames, fps)
    secs = frames / float(fps or 24)
    out = [f"This is a SAMPLING out-of-memory, not a decode one, so tiled decode "
           f"cannot help it. The shot is {w}x{h} x {frames}f (~{secs:.1f}s) = "
           f"{now:,} latent cells, and sampling cost scales linearly with that."]
    opts = []
    for cut in (10.0, 7.0):
        if cut < secs - 0.4:
            f2 = align_frame_count(int(round(cut * (fps or 24))))
            opts.append(f"shot_seconds {cut:g} ({f2}f) is "
                        f"{100 - shot_latent_cells(w, h, f2, fps) * 100 // now}% smaller")
    if megapixels:
        for mp in (0.5, 0.35):
            if mp < megapixels - 0.02:
                w2, h2 = scale_to_megapixels(w, h, mp)
                opts.append(f"megapixels {mp:g} ({w2}x{h2}) is "
                            f"{100 - shot_latent_cells(w2, h2, frames, fps) * 100 // now}% smaller")
    if opts:
        out.append("Options: " + "; ".join(opts) + ".")
    out.append("Shot length is the stronger lever on a chain, because every shot pays it. "
               "H3's own cap is 362 frames and this shot is at or near it.")
    return " ".join(out)




# --- removals ----------------------------------------------------------------
# The one place the node edits your text, and it only ever DELETES.
#
# The scene paragraph is stamped on every shot, so a garment described there is
# still being described after a beat takes it off -- and a description of a worn
# garment beats a sentence saying it came off. The old node inferred removals from
# prose, which meant guessing, and the guessing is most of what made it
# unpredictable. This does not guess. You say what came off:
#
#     Dan cuts off her jacket and throws it away.
#     remove: jacket
#
# From that shot onward, any part of the scene naming "jacket" is dropped. The
# directive line itself never reaches the model.

_REMOVE_LINE = re.compile(r"^[ \t]*(?:remove|removed|off)[ \t]*:[ \t]*(.+?)[ \t]*$",
                          re.I | re.M)

# Field labels the OLD version of this node printed at the bottom of every shot it
# built. Paste one of those old scripts back in as a prompt and the labels now go
# to the model verbatim -- and a line reading "overall_soundscape: room tone" is
# read as text to put ON THE PICTURE. They are never scene description, so they are
# dropped, and info says so.
# A whole line that is nothing but one of those labels. Only the exact field names
# the old node emitted -- a bare "music:" could be someone's own scene note.
_LEGACY_FIELD = re.compile(
    r"^[ \t]*(?:overall_soundscape|non_diegetic_music)[ \t]*:.*$", re.I | re.M)
# ...and the shot tag it put at the FRONT of a line that also carries real text, so
# only the tag comes off.
_LEGACY_PREFIX = re.compile(r"^[ \t]*\[(?:Generation|Shot)[ \t]*\d+\][ \t]*", re.I | re.M)

# Words that ask for letterforms in the frame. H3 renders text when the prompt
# names text, and at cfg 1 there is no negative prompt to take it back -- so this
# warns rather than edits: only you know whether "a neon sign" is set dressing you
# want or a watermark you do not.
_TEXT_CUE = re.compile(
    r"\b(?:subtitle[sd]?|caption(?:s|ed)?|closed[- ]caption\w*|watermark(?:ed|s)?|"
    r"logo|logos|credits|title card|end card|lower third|chyron|"
    r"timestamp|time stamp|date stamp|timecode|"
    r"text overlay|on-?screen text|banner|karaoke)\b", re.I)


def strip_legacy_fields(text):
    """(text, how many field-label lines were dropped)."""
    text = text or ""
    n = len(_LEGACY_FIELD.findall(text)) + len(_LEGACY_PREFIX.findall(text))
    if not n:
        return text, 0
    out = _LEGACY_PREFIX.sub("", _LEGACY_FIELD.sub("", text))
    # The field lines leave blank lines behind, and a blank line is a beat boundary
    # here -- collapsing them keeps the shot count the author intended.
    out = re.sub(r"[ \t]*\n[ \t]*\n[ \t]*\n+", "\n\n", out)
    return out.strip(), n


_ADD_LINE = re.compile(r"^[ \t]*(?:add|wear|wearing)[ \t]*:[ \t]*(.+?)[ \t]*$", re.I | re.M)

# Prose that reads as taking something off. NOT used to remove anything -- inferring
# removals from prose is what made the old node unpredictable. It is used only to
# notice that a beat looks like a removal while the scene still describes the
# garment, and to say so, because that combination is a garment that comes back.
# Verbs that mean REMOVAL only with a particle. On their own, "cuts the rope",
# "takes her hand", "pulls her closer" and "throws the bag on the floor" are
# ordinary actions -- and reading one as a removal deletes that garment's entry
# from the scene, after which it is still worn but UNDESCRIBED. An undescribed
# garment is one the model invents, and what it invents is plain and pale. That
# is how a black shiny latex crop top comes back white.
#
# The particle's POSITION settles the ambiguous case. Straight after the verb it
# is a removal ("pulls down her shorts"); trailing after the object, only "off"
# and "away" are -- "takes her coat off" removes it, "pulls her crop top down"
# only adjusts it, and adjusting a garment must not cost it its description.
_STRIP_VERB = (r"take[sn]?|took|taking|pull(?:s|ed|ing)?|peel(?:s|ed|ing)?|"
               r"strip(?:s|ped|ping)?|cut(?:s|ting)?|rip(?:s|ped|ping)?|tear[s]?|tore|"
               r"slip(?:s|ped)?|shrug(?:s|ged)?|yank(?:s|ed)?|tug(?:s|ged)?|"
               r"toss(?:es|ed)?|throw[s]?|threw|"
               # How clothes actually come off, in the words people write it in.
               # Without these a beat took the garment off on screen while the scene
               # kept saying it was worn -- and the scene is re-stamped into every
               # later shot, so it came back on and stayed on.
               r"kick(?:s|ed|ing)?|step(?:s|ped|ping)?|lift(?:s|ed|ing)?|"
               r"slide[s]?|slid|wriggle[sd]?|wiggle[sd]?|work(?:s|ed)?")
# The verbs above that stay a removal when the particle TRAILS the object -- "kicks
# her boots off". The rest are removals only with the particle straight after them:
# "steps out of her leggings" is one, "steps back" while a light goes off later in
# the sentence is not, and the trailing form would read that as a removal.
_TRAILING_VERB = (r"take[sn]?|took|taking|pull(?:s|ed|ing)?|peel(?:s|ed|ing)?|"
                  r"strip(?:s|ped|ping)?|cut(?:s|ting)?|rip(?:s|ped|ping)?|tear[s]?|"
                  r"tore|slip(?:s|ped)?|shrug(?:s|ged)?|yank(?:s|ed)?|tug(?:s|ged)?|"
                  r"toss(?:es|ed)?|throw[s]?|threw|kick(?:s|ed|ing)?|"
                  r"slide[s]?|slid|wriggle[sd]?|wiggle[sd]?")
# ...and verbs that are a removal on their own, needing no particle.
_UNDO_VERB = (r"remove[sd]?|removing|undress(?:es|ed)?|unzip(?:s|ped)?|"
              r"unbutton(?:s|ed)?|unhook(?:s|ed)?|unclasp(?:s|ed)?|unfasten(?:s|ed)?|"
              # Hardware comes off by being UNDONE, and these were missing: a beat
              # saying "unlocks the belt" left it described as worn for the rest of
              # the film, because nothing here read as a removal at all.
              r"unlock(?:s|ed)?|unbuckle[sd]?|unclip(?:s|ped)?|unstrap(?:s|ped)?|"
              r"unlace[sd]?|untie[sd]?|unties|unwrap(?:s|ped)?|"
              r"undo(?:es)?|undid")

_REMOVAL_PROSE = re.compile(
    r"\b(?:" + _UNDO_VERB + r")\b"
    r"|\b(?:" + _STRIP_VERB + r")\s+(?:off|away|out\s+of|down)\b"
    r"|\b(?:" + _TRAILING_VERB + r")\b(?=[^.;!?]{0,40}?\b(?:off|away)\b)"
    # Over the head is off. The only way a garment goes over a head is coming off
    # or going on, and the strip verbs are one-directional. A LOOKAHEAD, because
    # the garment sits between the verb and the particle -- "lifts her top over her
    # head" -- and the object span is read forward from the end of the match.
    r"|\b(?:" + _STRIP_VERB + r")\b"
    r"(?=[^.;!?]{0,40}?\bover\s+(?:her|his|their|the)\s+head\b)",
    re.I)


_HAS_VERB = re.compile(
    r"\b(?:is|are|was|were|be|being|been|has|have|had|wears?|wearing|dressed|"
    r"walks?|walked|stands?|stood|sits?|sat|lies?|lying|holds?|holding|"
    r"cuts?|pulls?|takes?|steps?|turns?|looks?|comes?|goes)\b", re.I)


def off_by_last_frame(items):
    """State that a removal FINISHES inside this shot. Empty when nothing came off.

    Scrubbing the scene stops a garment being described. It does not tell the model
    to complete the removal, and the last frame is what the next shot inherits as
    its keyframe -- so a cut still in progress hands on a garment still half worn,
    and the next beat has moved on and never contradicts the picture. The garment
    stays. That is a garment "coming back" even though the text was right.

    Said ONCE, in the removing shot, and never again. A later shot that says "no
    longer wearing the coat" names the coat, and to a video model a mention is a
    presence cue -- that phrasing put garments back on in the previous version of
    this node. Afterwards the item is simply absent from the text."""
    items = [i.strip() for i in (items or []) if i and i.strip()]
    if not items:
        return ""
    what = " and ".join(f"the {i}" for i in items)
    plural = len(items) > 1 or bool(_PLURAL_ITEM.search(items[-1]))
    verb, are = ("come", "are") if plural else ("comes", "is")
    sentence = (f"{what} {verb} off during this shot and {are} away by the last frame, "
                f"fully removed and no longer on the body, dropped out of frame.")
    # BOUND the action. Saying what comes off does not say where to STOP, and an
    # action with time left over runs on to whatever is next: a hand that finishes
    # one garment starts on the next one, or on the body under it. Said as what
    # STAYS -- at
    # cfg 1 there is no negative prompt, and a negation in the positive names the
    # thing it forbids. It also names no garment, so it summons none.
    # About what is WORN, not about the body. "Everything else on the body stays
    # exactly as it is for the whole shot" reads as an instruction to hold still.
    bound = "Everything else worn stays exactly as it is, untouched and still fastened."
    return " " + sentence[0].upper() + sentence[1:] + " " + bound


# Garments that are grammatically plural, so the sentence above agrees with them.
_PLURAL_ITEM = re.compile(r"\b(?:s|shorts|trousers|pants|jeans|boots|shoes|gloves|"
                          r"tights|leggings|briefs|knickers|cuffs)$", re.I)


# --- restraints ---------------------------------------------------------------
# The one continuity fact the node asserts on its own, because it is the one that
# cannot be recovered: a cuff that renders open is not a detail that drifts, it is
# the scene stopping making sense. Once hardware is on, it stays on.
#
# ONE sentence, impersonal, positive. The previous version had a per-limb effect
# table, pose tracking and a hardware clause, and between them the beat became 4% of
# the prompt. This is the fact and nothing else.
# What a `remove:` has to name to switch the hold off again.
RESTRAINT_HOLD_KEY = ("handcuffs cuffs chains rope ropes tape gag collar restraints "
                      "shackles clamp clamps clip clips")
# Every one of these constrains the HARDWARE, never the body. An earlier wording said
# the restraint held "the same way from the first frame to the last" and the chain let
# the body reach "only as far as the metal allows before it stops" -- read plainly,
# that is an instruction to hold still, and stacked together the holds came to 64% of
# a shot whose beat was 11%. The performance died under its own continuity guards.
# Say what the metal does; leave the body to the beat.
# Staying closed is not the same as staying itself. Every hold above constrains
# the fastening; none of them says the thing is still made of what it was made
# of. A strip of tape, decoded and re-encoded once a shot, has nothing in the
# text holding it to being tape, and it drifts to the nearest commoner object.
# One short sentence, because these holds are already the longest thing a
# restrained shot carries.
FORM_HOLD = ", and each piece keeps the material and shape it was put on with."

RESTRAINT_HOLD = (" Every restraint stays whole and closed, fastened exactly as it was put "
                  "on, still fastened at the last frame") + FORM_HOLD


def restraint_wearers(sheet):
    """The people whose own sheet entry describes hardware.

    Read from the entries rather than the beat, because the entry is what says who is
    WEARING it -- a beat can mention a chain without anyone being in it."""
    return [n for n, ln in sheet_lines(sheet) if n and restraint_present(ln)]


def own_hold(hold, wearers, described):
    """Attribute a hold to whoever actually wears the hardware.

    The holds say "every restraint stays fastened" and name nobody, which was fine
    while a shot meant one person. Put a second person in the frame and it becomes an
    instruction about whoever is on screen: the belt locked onto one character turned
    up on the other, over their clothes, because the sentence never said whose it was.

    Only when the shot describes more than one person -- with one there is no
    ambiguity, and the extra words are shot budget spent on nothing. Positively
    phrased: saying who wears it is what excludes everyone else, where "nobody else
    is wearing one" asks the model to render an absence."""
    if not hold or not wearers or len(described) < 2:
        return hold

    def _and(names):
        return names[0] if len(names) == 1 else \
            ", ".join(names[:-1]) + " and " + names[-1]

    who = _and(wearers)
    tail = (f" The hardware is {who}'s, worn on the body it was locked to. Everyone "
            f"else in the shot has on exactly what their own entry lists.")
    return hold.replace("Every restraint", f"Every restraint on {who}", 1).rstrip() + tail

# Hardware that means restraint on its own.
_RESTRAINT_PLAIN = re.compile(
    r"\b(?:handcuff(?:s|ed)?|cuffed|shackle[sd]?|manacle[sd]?|hogtied|hog-?tied|"
    r"hogcuffed|hog-?cuffed|gag(?:ged|s)?|blindfold(?:ed|s)?|zip[- ]ties?|"
    r"cable[- ]ties?|restrain(?:t|ts|ed)|bound|bindings?|straitjacket|"
    r"spreader bar)\b", re.I)
# Hardware that is only a restraint in context -- a chain-link fence, a rope on a
# boat and a leather belt are none of the node's business.
# A clamp belongs here rather than in the list above: clamped to a bench it is a
# tool, clamped to a body it is hardware, and only the context tells them apart.
_RESTRAINT_MAYBE = re.compile(
    r"\b(?:chains?|ropes?|cords?|cuffs?|straps?|collars?|tapes?|taped|taping|"
    r"belts?|harness|hobble|clamps?|clips?)\b", re.I)
# VERB forms only. An earlier version listed "chain" and "cuff" here as well as in
# the noun list, so a chain-link fence matched both halves and armed the rule.
_BINDING_VERB = re.compile(
    r"\b(?:cuffed|chained|tied|tying|bound|binds?|binding|locked|locks|"
    r"strapped|taped|taping|gagged|shackled|fastened|fastens|secured|secures|"
    r"padlocked|trussed|lashed|wrapped|clamped|clamping|clipped|clipping|"
    r"pinned|attached|affixed)\b", re.I)
# NOTE the bare "clamps" and "clips" are deliberately absent above while "clamp" and
# "clip" are in the noun list. A word in BOTH lists satisfies both halves of the rule
# by itself, which is how "clamps the board to the workbench" armed the restraint
# hold -- the same way a chain-link fence did before "chain" was taken out of the
# verbs. Same reason "tapes" is a noun here and only "taped"/"taping" are verbs.
_BODY_PART = re.compile(
    r"\b(?:wrists?|ankles?|arms?|legs?|hands?|feet|neck|throat|mouth|waist|hips?|"
    r"thighs?|knees?|elbows?|thumbs?|eyes)\b", re.I)


# A turn shows a surface the shot has never shown. The keyframe pins the FRONT, so
# once the body rotates the model is filling in from its prior -- and its prior for
# an undescribed body is a CLOTHED one. That is a removed garment coming back, often
# stacked in the wrong order because nothing said which layer was where, and hardware
# on the far side being re-invented as it rotates into view.
#
# One sentence, only on shots that turn, and only once there is state worth holding.
# It names no garment and no person, so it summons neither.
TURN_HOLD = (" What is on the body now is all that is on it, front, side and behind, and "
             "whatever is fastened stays fastened and closed as the view comes round.")

_TURN_CUE = re.compile(
    r"\b(?:turn(?:s|ed|ing)?|rotat(?:es?|ed|ing)|spin(?:s|ning)?|swivel(?:s|led)?|"
    r"roll(?:s|ed|ing)?\s+(?:over|onto)|faces?\s+away|face[sd]?\s+the\s+other|"
    r"over\s+(?:her|his|their)\s+shoulder|from\s+behind|back\s+to\s+the\s+camera|"
    r"shows?\s+(?:her|his|their)\s+back|other\s+side)\b", re.I)


# Being MOVED does the same damage as turning, for the same reason: the keyframe
# pinned one pose seen from one side, and lifting, dragging or rolling someone puts
# the body somewhere that frame never showed. The verb needs a PERSON as its object
# -- "lifts her onto the table" moves her, "lifts the crate" does not, and
# "positions her legs" moves a limb, not the body.
_MOVE_VERB = re.compile(
    r"\b(?:lifts?|lifted|carr(?:ies|ied)|drags?|dragged|hauls?|hauled|hoists?|hoisted|"
    r"picks?\s+up|picked\s+up|sets?\s+down|set\s+down|lays?|laid|"
    r"lowers?|lowered|rolls?|rolled|flips?|flipped|props?|propped|"
    r"moves?|moved|repositions?|repositioned|pulls?|pulled|pushes|pushed|"
    r"shoves?|shoved|throws?|threw|drops?|dropped|turns?|turned)\s+", re.I)
_PERSON_OBJ = r"(?:the\s+|a\s+)?(?:her|him|them"


def body_moved(text, names=()):
    """Is a PERSON being moved in this beat, rather than an object or a limb?"""
    toks = [re.escape(n) for n in (names or []) if n]
    obj = re.compile(_PERSON_OBJ + (("|" + "|".join(toks)) if toks else "") + r")\b"
                     # ...not a possessive, and not a LIMB: "positions her legs" moves
                     # the legs, not the body. An earlier guard rejected any following
                     # word ending in "s", which threw out "drags her across the floor".
                     r"(?!\s*['’]s)"
                     r"(?!\s+(?:legs?|arms?|wrists?|ankles?|hands?|feet|foot|head|hair|"
                     r"hips?|shoulders?|knees?|elbows?|thighs?|face|chin)\b)"
                     # A moved BODY goes somewhere: the object is followed by a word
                     # of motion, or the clause simply ends. Without this, "pulls her
                     # shorts off" reads as moving her rather than the shorts.
                     r"(?=\s*(?:[.,;!?]|$)"
                     r"|\s+(?:onto|into|on|in|to|across|down|up|over|under|back|out|"
                     r"away|upright|off|against|toward|towards|through|round|around|"
                     r"beside|behind|clear)\b)", re.I)
    return any(obj.match(text[m.end():]) for m in _MOVE_VERB.finditer(text or ""))


def turns_in(text, names=()):
    """Does this beat rotate a body, move one, or bring the view around it?"""
    return bool(_TURN_CUE.search(text or "")) or body_moved(text, names)


# A falling body's reflex is to put its hands out. When the hands are fastened, the
# model has to resolve that conflict, and the cheapest resolution is to free them --
# which renders as the cuffs opening or the chain snapping mid-fall. Nothing in the
# restraint hold covers it, because the hold says the hardware is whole and says
# nothing about what the body does on the way down.
#
# So say what DOES take the landing. Positive, and it names no person: at cfg 1
# there is no negative prompt, and "does not catch itself" names catching.
FALL_HOLD = (" A bound body falls as one piece: the fastened limbs stay fastened and travel "
             "with it, the arms staying in the hold, and the shoulder, hip or side takes "
             "the landing.")

_FALL_CUE = re.compile(
    r"\b(?:falls?|fell|falling|drops?\s+to|dropped\s+to|collapse[sd]?|collapsing|"
    r"topple[sd]?|topples|tips?\s+over|tipped\s+over|keels?\s+over|goes\s+down|"
    r"went\s+down|slumps?|slumped|stumbles?|stumbled|overbalance[sd]?|"
    r"loses?\s+(?:her|his|their)\s+balance|lost\s+(?:her|his|their)\s+balance|"
    # ...and being put down by someone else: "pushes her over", "knocked him down".
    r"(?:push|knock|shove|pull|drag|throw|thr[eo]w)(?:es|s|ed|n)?\s+"
    r"(?:her|him|them|\w+\s+)?(?:over|down|to\s+the\s+(?:floor|ground))|"
    r"hits?\s+the\s+(?:floor|ground|deck))\b", re.I)


def falls_in(text):
    """Does a body go down in this beat?"""
    return bool(_FALL_CUE.search(text or ""))


# Steel does not behave like rope. A model with no reason to think otherwise draws a
# chain as a soft cord: it sags, stretches to wherever a limb is going, and lets the
# body move as if nothing were fastened. The restraint hold says the hardware stays
# WHOLE; it says nothing about how it behaves while whole.
#
# Positive and impersonal, like the other holds -- at cfg 1 there is no negative
# prompt, so "does not stretch" only names stretching.
# REPLACES the restraint hold rather than joining it -- the two said "stays whole and
# closed" twice, and two clauses saying the same thing is twice the stasis for one
# guarantee.
CHAIN_HOLD = (" Every restraint stays whole and closed, fastened exactly as it was put on, "
              "and still fastened at the last frame; its links keep their size and the run "
              "between them stays straight and taut") + FORM_HOLD

# When hardware is what PUTS a body in a position, the length of that hardware is the
# whole reason the position holds. Saying the metal keeps its shape is not enough: a
# chain that keeps its shape can still be drawn as having slack, and slack is room to
# stand up out of a squat the chain was locked to enforce.
#
# It replaces the clause above rather than joining it, and it is careful to leave the
# body free to act: straining and pulling is exactly what should happen, and the last
# thing this should say is that anything holds still.
CHAIN_POSE_HOLD = (" Every restraint stays whole and closed, fastened exactly as it was put "
                   "on; the metal is already drawn out to its full length, so the position "
                   "it fixes is the position that keeps, and the body strains and pulls "
                   "against it while the fastenings hold at exactly the length they were "
                   "locked to") + FORM_HOLD

# A position that hardware can be locked to enforce.
_FORCED_POSE = re.compile(
    r"\b(?:squat(?:s|ting|ted)?|kneel(?:s|ing)?|knelt|crouch(?:es|ing|ed)?|"
    r"hogtied|hog-?tied|hogcuffed|hog-?cuffed|trussed|"
    r"bent\s+(?:over|double)|doubled\s+over|folded\s+(?:up|forward)|"
    r"spread[- ]eagled?|curled\s+up|"
    r"on\s+(?:her|his|their)\s+(?:knees|haunches))\b", re.I)


def forced_pose(text):
    """Does this text put a body into a position that hardware can enforce?"""
    return bool(_FORCED_POSE.search(text or ""))

# Hardware that is rigid by nature. Only consulted once a restraint is established,
# so a chain-link fence in the scenery cannot arm it on its own.
# Named hardware only. "steel" was in this list, which meant any steel object earned
# the chain clause -- and that clause talks about LINKS and the RUN between fastenings,
# which is nonsense said of a steel clamp. A clamp is rigid, but it is not a chain: it
# gets the plain restraint hold, which is what "it stays on" needs anyway.
_RIGID_HARDWARE = re.compile(
    r"\b(?:chain(?:s|ed|ing)?|padlock(?:s|ed|ing)?|shackle[sd]?|manacle[sd]?|"
    r"handcuff(?:s|ed)?|cuffs?|cuffed|irons|spreader\s+bar|"
    r"hogcuffed|hog-?cuffed)\b", re.I)


# Where each piece of hardware goes. Not a creative choice -- it is what the object
# IS. A collar without a neck is a band with no place to be, and a model handed a
# band-shaped object and no anatomy puts it where bands most often sit in its
# training data: on the head. That is the reported failure, and it happens whether
# the item is being fastened or merely held up and shown.
#
# (item pattern, the phrase that places it)
_TAPE_GAG = (r"(?:duct[\s-]*)?tape\s+gag|"
             r"gag(?:s|ged|ging)?\s+\w{0,12}\s*with\s+"
             r"(?:duct\s+|packing\s+|masking\s+)?tape|"
             r"tape\s+(?:over|across)\s+(?:her|his|their|the)\s+mouth")
_TAPE_GAG_CLAUSE = "a strip of tape lies flat across the mouth"
_GAG_CLAUSE = "a gag sits in the mouth"

_HARDWARE_ANCHOR = (
    (r"collar(?:s|ed)?",                 "a collar closes around the neck"),
    (r"leash(?:es)?|lead\b",             "a leash clips to the collar at the neck and hangs down from it"),
    # Tape is a gag that lies flat against the face. Told "a gag sits in the
    # mouth" it is given bulk it does not have, and bulk over the mouth,
    # re-encoded shot after shot, settles into a mask.
    (_TAPE_GAG,                          _TAPE_GAG_CLAUSE),
    (r"gag(?:s|ged)?|ball\s*gag",        _GAG_CLAUSE),
    (r"blindfold(?:s|ed)?",              "a blindfold covers the eyes"),
    (r"handcuff(?:s|ed)?",               "handcuffs close around the wrists"),
    (r"shackle[sd]?|leg\s+irons",        "shackles close around the ankles"),
    (r"harness(?:es)?",                  "a harness sits on the torso"),
    (r"spreader\s+bar",                  "a spreader bar holds the ankles apart"),
    # No entry for a chastity belt, and the lookbehind below keeps the plain belt off
    # it too, so it gets no placement clause at all. It is the item most likely to
    # arrive with its own <Picture N>, and a written description of where the shield
    # and the lock sit argues with the picture rather than adding to it. Where the
    # reference shows the object, the object is already placed; describe it in your
    # own words if you want it stated.
    (r"(?<!chastity\s)belt(?:s|ed)?",    "a belt closes around the waist and hips"),
)

# Anatomy that already places something, so the writer's own wording wins.
_BODY_PART = re.compile(
    r"\b(?:neck|throat|wrist|wrists|ankle|ankles|mouth|lips|jaw|eyes|face|head|"
    r"waist|hips?|chest|torso|stomach|belly|shoulders?|arms?|legs?|thighs?|"
    r"knees?|feet|foot|hands?)\b", re.I)


def unanchored_hardware(text):
    """Phrases placing any hardware that is named with no body part beside it.

    A window of 60 characters either side counts as 'beside'. If the text already
    says where the thing goes, nothing is added -- what you wrote wins."""
    out = []
    t = text or ""
    for pat, phrase in _HARDWARE_ANCHOR:
        placed = False
        found = False
        for m in re.finditer(r"\b(?:" + pat + r")", t, re.I):
            found = True
            window = t[max(0, m.start() - 60):m.end() + 60]
            if _BODY_PART.search(window):
                placed = True
                break
        if found and not placed and phrase not in out:
            out.append(phrase)
    # A tape gag answers the gag entry as well, and the two clauses disagree
    # about whether the thing has bulk. The flat one is the true one.
    if _TAPE_GAG_CLAUSE in out and _GAG_CLAUSE in out:
        out.remove(_GAG_CLAUSE)
    return out


def anchor_clause(phrases):
    """One sentence saying where the named hardware sits."""
    if not phrases:
        return ""
    return " Each piece of hardware sits where it belongs: " + "; ".join(phrases) + "."


# A state written down is a state the model can render by ARRIVING at it.
#
# Reported: "stand behind a van with its doors closed" put the doors open and the
# characters closing them. The text named a state and never said WHEN it was true,
# and a video model asked for a door renders the thing a door does. The state is
# the most interesting event in the sentence, so it gets performed.
#
# Fewer sampling steps make it worse rather than better. On a 4-step distill
# schedule the layout is committed almost immediately, so an opening frame that
# guessed wrong is never argued out of it by the later steps -- there are none.
# Saying the state is already true at the first frame costs one sentence and takes
# the event away.
#
# Scenery only, and only words that are not also something worn: no "boots", no
# "hood", no "bonnet". A character sheet lives in this same text.
_STATE_THING = (r"doors?|gates?|windows?|curtains?|blinds?|shutters?|"
                r"hatch(?:es)?|tailgates?|lids?|drawers?")
_STATE_WORD = r"closed|shut|open|locked|unlocked|latched|bolted|drawn|ajar|sealed"
# Verbs that CHANGE one of those states. Several are also the state word itself --
# "closed" is both -- which the reader below has to tell apart.
_STATE_ACTS = (r"opens?|opened|opening|closes?|closed|closing|shuts?|shutting|"
               r"slams?|slammed|slamming|slides?|slid|sliding|pulls?|pulled|pulling|"
               r"pushes?|pushed|pushing|draws?|drew|drawing|locks?|locked|locking|"
               r"unlocks?|unlocked|unlocking|lifts?|lifted|lifting|raises?|raised|"
               r"lowers?|lowered|swings?|swung|yanks?|yanked|wrenches|wrenched")
# The gap takes apostrophes: "closed the van's doors" is a determiner phrase, and a
# gap of bare \w+ does not match one, so the whole act went unseen.
_STATE_ACT = re.compile(r"\b(" + _STATE_ACTS + r")\s+((?:[\w']+\s+){0,3}?)(" +
                        _STATE_THING + r")\b", re.I)
# What tells "closed the rear doors" from "closed rear doors": a determiner. The verb
# reading needs one -- you close THE doors, ITS doors, THE VAN'S doors -- and the
# adjective reading cannot have one, because the determiner belongs in front of the
# whole phrase ("a van with closed rear doors").
#
# Getting this wrong is not a missed guard, it is an inverted one. Read as a verb,
# "a van with closed rear doors" earned the anchor "the doors are open at the first
# frame and shut by the last" -- the node itself asking for the doors to start open
# and be closed on camera, which is the bug it was written to fix.
_STATE_DET = re.compile(r"\b(?:the|a|an|its|his|her|their|our|my|your|this|that|these|"
                        r"those|both|all|each|every|another|one|two|three|\w+'s)\b", re.I)


def _adjectival(verb, gap):
    """Is this state word describing the noun rather than acting on it?

    Only words that are ALSO states can be adjectives: "opens" is a verb however it
    is placed. Modifiers may sit between -- "closed rear doors", "shut cargo doors" --
    so the test is the determiner, not the distance."""
    return (bool(re.fullmatch(_STATE_WORD, verb, re.I))
            and not _STATE_DET.search(gap or ""))


# "closed doors", and "closed rear doors" -- the state in front of its noun, with the
# modifiers a real sentence puts between them.
_STATE_ADJ = re.compile(r"\b(" + _STATE_WORD + r")\s+((?:[\w']+\s+){0,2}?)(" +
                        _STATE_THING + r")\b", re.I)
# "the doors are closed", "the doors closed", "the doors are still shut". The gap is
# copulas and nothing else, so a state word further off in the sentence -- belonging
# to some other object -- is not dragged onto this one.
_STATE_PRED = re.compile(r"\b(" + _STATE_THING + r")\s+" +
                         r"((?:(?:are|is|was|were|remains?|stay|stays|still|both|all)\s+){0,2})(" +
                         _STATE_WORD + r")\b", re.I)


def _state_key(thing):
    """One key for 'door' and 'doors', so a beat acting on either clears both."""
    t = (thing or "").lower()
    return t[:-2] if t.endswith("es") and t.startswith("hatch") else t.rstrip("s")


# Which way a verb runs. Reported: some distill LoRAs render an action BACKWARDS --
# the beat opens the doors and the shot closes them. A single staged action is
# direction-ambiguous to a model that has learned to treat a clip and its reverse as
# the same clip, which is what time-flip augmentation teaches. Naming the two ends
# settles it, and it is the same thing a removal already does: "off during this shot
# and away by the last frame".
#
# Only verbs that HAVE a direction. "pulls", "draws", "slides" and "swings" do not:
# drawing the curtains closes them and pulling a door can do either, and a wrong
# anchor is worse than none -- it asks for the reversal instead of merely allowing it.
_OPENS = re.compile(r"(?:opens?|opened|opening|unlocks?|unlocked|unlocking|"
                    r"lifts?|lifted|lifting|raises?|raised)\Z", re.I)
_SHUTS = re.compile(r"(?:closes?|closed|closing|shuts?|shutting|slams?|slammed|"
                    r"slamming|locks?|locked|locking|lowers?|lowered)\Z", re.I)


def state_changes(text):
    """[(thing, 'open'|'shut'|None)] for the scenery this text actually works.

    A state word sitting straight in front of its noun is an adjective describing
    the thing, not a verb acting on it: "the closed doors" says nothing happens.
    The direction is None where the verb does not carry one."""
    out, seen = [], set()
    for m in _STATE_ACT.finditer(text or ""):
        verb, gap, thing = m.group(1), m.group(2), m.group(3)
        if _adjectival(verb, gap):
            continue
        key = _state_key(thing)
        if key in seen:
            continue
        seen.add(key)
        way = ("open" if _OPENS.match(verb) else
               "shut" if _SHUTS.match(verb) else None)
        out.append((thing.lower(), way))
    return out


def state_acts(text):
    """Which of those things this text works, in either direction."""
    return [_state_key(t) for t, _ in state_changes(text)]


def direction_anchor(changes):
    """Say which end of a staged change is which, for the ones that have a direction.

    Two at most, and the caller trades these against the held states: a shot carrying
    four continuity sentences is a shot that has stopped being about its beat."""
    said = []
    for thing, way in changes:
        if not way:
            continue
        start, end = ("shut", "open") if way == "open" else ("open", "shut")
        said.append(f"The {thing} {'are' if thing.endswith('s') else 'is'} {start} at "
                    f"the first frame and {end} by the last.")
        if len(said) == 2:
            break
    return (" " + " ".join(said)) if said else ""


def stated_states(text):
    """(thing, state) for every scenery state this text asserts but does not stage."""
    # A thing this same text WORKS is not a thing standing in a state: "slams the
    # tailgate shut" reads as both, and the action is the true reading. Left to the
    # caller this came back twice, once held and once anchored, disagreeing.
    out, seen = [], set(state_acts(text))
    for pat, order in ((_STATE_ADJ, "sn"), (_STATE_PRED, "ns")):
        for m in pat.finditer(text or ""):
            if order == "sn":
                state, gap, thing = m.group(1), m.group(2), m.group(3)
                # The same determiner test, from the other side: with one, this is
                # somebody closing the doors, and the state is not standing at all.
                if not _adjectival(state, gap):
                    continue
            else:
                state, thing = m.group(3), m.group(1)
            key = _state_key(thing)
            if key in seen:
                continue
            seen.add(key)
            out.append((thing.lower(), state.lower()))
    return out


# Getting OUT of a vehicle. A person leaving a van opens a door to do it, so a beat
# staging an exit and a state saying the doors are shut are two instructions that
# cannot both be followed. The beat wins -- it stages an action, and an action beats
# a state -- and the hold is left arguing with the script it is supposed to serve.
#
# The node must not touch the wording either way: those are the author's words, and
# "out of the van" may be exactly what they mean. So it says so instead. Three rounds
# of this went by as a silent bad render when one line of info would have placed it.
_EXIT_VEHICLE = re.compile(
    r"\b(?:get|gets|got|climb(?:s|ed)?|step(?:s|ped)?|jump(?:s|ed)?|slid(?:e|es)|"
    r"come|comes|came|walk(?:s|ed)?|hop(?:s|ped)?|pile)\s+(?:down\s+|back\s+)?out\s+"
    r"of\s+(?:the\s+|a\s+|an\s+|his\s+|her\s+|their\s+|its\s+)?"
    r"(?:back\s+of\s+(?:the\s+|a\s+)?)?(?:van|car|truck|cab|vehicle|lorry|bus)\b"
    r"|\bexits?\s+(?:the\s+|a\s+)?(?:van|car|truck|cab|vehicle)\b"
    r"|\bout\s+of\s+(?:the\s+|a\s+)?(?:van|car|truck|cab)\b", re.I)


def exits_vehicle(text):
    """Does this beat stage somebody getting out of a vehicle?"""
    return bool(_EXIT_VEHICLE.search(text or ""))


def state_hold(pairs):
    """One sentence putting those states at the first frame instead of in the action.

    Two at most. These sentences are continuity, and continuity that outgrows the
    beat is what the beat stops being about."""
    said = []
    for thing, state in pairs[:2]:
        plural = thing.endswith("s")
        # BOUNDED, for the same reason a removal says "by the last frame": "stays
        # closed" has no end on it, and a state with time left over is a state
        # something can happen to before the shot is out.
        said.append(f"The {thing} {'are' if plural else 'is'} already {state} at the "
                    f"first frame and {'stay' if plural else 'stays'} {state} for the "
                    f"whole shot.")
    return (" " + " ".join(said)) if said else ""


def rigid_hardware(text):
    """Is the hardware here the kind that cannot flex?"""
    return bool(_RIGID_HARDWARE.search(text or ""))


def restraint_present(text):
    """Is a restraint being applied or worn, in this text?

    Plain hardware counts on its own. Ambiguous hardware needs a binding verb or a
    body part alongside it, so a chain-link fence and a leather belt do not arm a
    continuity rule about restraints."""
    t = text or ""
    if _RESTRAINT_PLAIN.search(t):
        return True
    return bool(_RESTRAINT_MAYBE.search(t)
                and (_BINDING_VERB.search(t) or _BODY_PART.search(t)))


def names_any(text, tokens):
    """Does `text` name any of these items?"""
    return any(re.search(r"\b" + re.escape(t) + r"\b", text or "", re.I)
               for t in (tokens or []) if t)


# Where a removal verb's object ENDS. "pulls off her coat and drops it, showing the
# jumper" takes off the coat; the jumper is what becomes visible. The old version of
# this node matched garment words anywhere in the beat and took both off, which is
# the failure that made prose inference untrustworthy.
def person_tags(text):
    """The <Picture N> tags that belong to a PERSON rather than to an object.

    Decided by what stands immediately BEFORE the tag. A name -- capitalised, with or
    without its colon -- means the picture is of that person: "Nora: <Picture 1>",
    "Nora <Picture 1> in a grey coat". A lowercase noun means it is a picture OF the
    thing it is standing next to: "a silver locket <Picture 2>".

    That distinction is what lets an object's reference come off with the object. A
    person's tag has to survive a removal that shares its fragment, or the shot loses
    its identity reference; an object's tag has to go, or it keeps asserting the thing
    that was just taken off."""
    out = []
    for m in _PICTURE_TAG.finditer(text or ""):
        before = (text[:m.start()]).rstrip().rstrip(",").rstrip()
        w = re.search(r"([\w'’-]+)$", before)
        # An object owns the tag only when a lowercase NOUN stands immediately before
        # it -- "a silver locket <Picture 2>". Everything else is the person's: a
        # name, a colon, an age ("Kate is 20, <Picture 1> blonde crop top"), or
        # nothing at all. Erring this way on purpose, because losing a person's
        # identity reference costs the shot its face, while an object tag left behind
        # only keeps describing something already taken off.
        if not before.endswith(":") and w:
            head = w.group(1)[:1]
            if head.isalpha() and head.islower():
                continue                      # the picture belongs to the object
        out.append(m.group(1))
    return out


_OBJECT_END = re.compile(r"(?:,|;|\.|\bexposing\b|\brevealing\b|\bshowing\b|\bleaving\b|"
                         r"\bto\s+expose\b|\bto\s+reveal\b|\bthen\b|\buntil\b)", re.I)

# Words that sit in a removal's object span but are never the thing that comes off:
# grammar, the prepositions that place a garment, and the body it is placed on.
# "cuts the tight top away from her back" names ONE garment; the rest is syntax and
# anatomy. Without this, every word the beat happened to share with the scene was
# taken off -- "the tight and the her and the back come off during this shot".
_NOT_A_GARMENT = frozenset("""
the a an and or her his its their our your this that these those
off from over under onto into out down up away through across behind
front side left right rest way bit end edge
back neck chest waist hips hip wrist wrists ankle ankles arm arms hand hands
leg legs thigh thighs knee knees foot feet shoulder shoulders head face mouth
lips hair skin body torso stomach belly chin jaw eyes ear ears
floor ground wall room air
""".split())

# Where a scene's wardrobe entry ENDS. A garment word is the HEAD of its phrase --
# "black boots," "grey coat and", "wool scarf." -- while a modifier is followed by
# more of the phrase ("tight white crop top": tight, white and crop all fail this,
# top passes). Adjectives cannot be listed, so test position instead of vocabulary.
_ENTRY_END = re.compile(r"^\s*(?:[,;.!?]|$|(?:and|over|under|beneath|above|with|plus)\b)",
                        re.I)

# Hardware, not clothing. Inference never takes a restraint off: the standing rule is
# that once one goes on it stays on, and an explicit `remove:` is the only thing that
# clears it. A beat that cuts a rope must not silently unlock the cuffs as well.
_RESTRAINT_WORD = re.compile(
    r"^(?:handcuffs?|cuffs?|shackles?|manacles?|chains?|ropes?|cords?|straps?|"
    r"collars?|gags?|blindfolds?|restraints?|bindings?|tape|ties?|harness|"
    r"straitjacket|spreader|hogtie|clamps?|clips?)$", re.I)


# A <Picture N> immediately after a word, so the entry-end test can look past an
# object's own reference to the comma that actually ends its entry.
_LEADING_TAG = re.compile(r"^\s*<\s*picture[\s_\-]*\d+\s*>", re.I)


def _is_entry_head(word, scene):
    """Is `word` the head of a wardrobe entry in the scene, rather than a modifier
    inside one or a fragment of a hyphenated compound?"""
    for m in re.finditer(r"\b" + re.escape(word) + r"\b", scene, re.I):
        # "tight" inside "skin-tight" is half a word, not a garment.
        if m.start() and scene[m.start() - 1] == "-":
            continue
        if m.end() < len(scene) and scene[m.end()] == "-":
            continue
        # An object's own reference sits between the noun and the comma that ends its
        # entry -- "a silver locket <Picture 2>, green jacket" -- so the entry-end
        # test has to look past it. Without this, a tagged object is never the head of
        # anything, which means auto_remove can never take it off: it needed an
        # explicit `remove:` line while an untagged one came off from the prose.
        tail = _LEADING_TAG.sub("", scene[m.end():], count=1)
        if _ENTRY_END.match(tail):
            return True
    return False


def infer_removals(beat, scene):
    """Garments this beat takes off, read from its own prose. [] when none.

    Two conditions, both required, because a wrong removal is worse than a missed
    one: the beat has to contain a REMOVAL verb, and the thing named has to be
    something the SCENE already says is worn. A beat cannot take off what the
    character was never described wearing.

    Only the verb's own object counts -- the span from the verb to the next clause
    boundary. That is what keeps "pulls off her coat, showing the jumper" to the
    coat."""
    if not beat or not scene:
        return []
    found = []
    for m in _REMOVAL_PROSE.finditer(beat):
        tail = beat[m.end():]
        cut = _OBJECT_END.search(tail)
        span = tail[:cut.start()] if cut else tail
        # In the TRAILING form the object sits between the verb and the particle --
        # "takes her jacket off" -- so the particle ends the object, and what comes
        # after it is a new clause: in "takes her jacket off and drops it on the
        # chair" the chair is furniture the beat mentions, not something worn.
        #
        # A verb before the particle means the particle is not ours. "kicks the
        # chair and Mike walks off" ends in "off", but it is the walking that is off,
        # and reading that as a removal deleted the chair from the scene.
        #
        # Neither test applies to a verb that already swallowed its particle
        # ("pulls off her coat") or needs none ("unzips her jacket and pulls it
        # off"), where the object follows the verb and the sentence runs on.
        if not (re.fullmatch(_UNDO_VERB, m.group(0), re.I)
                or re.search(r"\b(?:off|away|out\s+of|down)$", m.group(0), re.I)):
            part = re.search(r"\b(?:off|away)\b", span, re.I)
            if part:
                if _HAS_VERB.search(span[:part.start()]):
                    continue
                span = span[:part.start()]
        for word in re.findall(r"\b[\w-]{3,}\b", span):
            low = word.lower().strip("-")
            if not low or low in found:
                continue
            # Grammar, prepositions and anatomy are not garments.
            if low in _NOT_A_GARMENT:
                continue
            # Hardware is cleared by an explicit `remove:` and by nothing else.
            if _RESTRAINT_WORD.match(low):
                continue
            # It has to be worn: the HEAD of something the scene lists, not a
            # modifier inside it and not half of a hyphenated compound.
            if not _is_entry_head(word, scene):
                continue
            # ...and not a person or a place.
            if re.search(r"\b" + re.escape(word) + r"\b\s*(?:is|was|walks|stands|sits|=)",
                         scene, re.I):
                continue
            found.append(low)
    return found


# Clothing, for the one case that names no garment at all: "strips out of their
# clothes". A vocabulary is the wrong tool for reading a removal out of prose -- which
# is why infer_removals tests POSITION instead -- but here the beat says nothing about
# WHAT comes off, so the only place left to read it from is the wardrobe itself.
#
# Anything this misses stays described, and the note says which entries were cleared,
# so a gap is visible rather than silent.
_GARMENT_WORD = re.compile(
    r"^(?:shirt|t-shirt|tshirt|top|blouse|jumper|sweater|sweatshirt|hoodie|cardigan|"
    r"jacket|coat|blazer|vest|waistcoat|gilet|dress|gown|skirt|trousers|pants|jeans|"
    r"leggings|shorts|tights|stockings|socks|boots|shoes|trainers|sneakers|sandals|"
    r"heels|slippers|hat|cap|beanie|scarf|gloves|mittens|tie|apron|overalls|dungarees|"
    r"uniform|robe|dressing-gown|pyjamas|pajamas|nightdress|nightie|swimsuit|bikini|"
    r"trunks|briefs|boxers|underwear|undershirt|underclothes|bra|knickers|panties|"
    r"thong|lingerie|camisole|slip|corset|romper|jumpsuit|tracksuit|anorak|parka|"
    r"poncho|kilt|sari|kimono|cloak|clothes|clothing|outfit)s?$", re.I)


def garments_in(text):
    """Every garment word the given wardrobe text names, in order.

    Restraints are excluded on purpose: taking clothes off does not unlock anything,
    and the standing rule is that hardware is cleared by an explicit `remove:` and by
    nothing else."""
    out = []
    for word in re.findall(r"\b[\w-]{3,}\b", text or ""):
        low = word.lower().strip("-")
        if low in out or _RESTRAINT_WORD.match(low):
            continue
        if _GARMENT_WORD.match(low):
            out.append(low)
    return out


# A beat that undresses somebody completely without naming one garment. Every other
# removal path needs the thing to be named; this is the case where the SCRIPT does not
# name it, so nothing came off and the scene went on listing the whole wardrobe --
# which is re-stamped into every later shot, so the clothes came back on.
#
# "naked eye" and "naked flame" are not people.
_NAKED_CUE = re.compile(
    r"\bnaked\b(?!\s+(?:eye|flame))"
    r"|\bnude\b|\bin\s+the\s+nude\b"
    r"|\bundress(?:es|ed|ing)?\b"
    r"|\bstrips?\s+(?:out\s+of|off|down|naked|bare)\b|\bstripp(?:ed|ing)\s+"
    r"(?:out\s+of|off|down|naked|bare)\b"
    r"|\btakes?\s+(?:everything|it\s+all|all\s+of\s+it|the\s+lot)\s+off\b"
    r"|\bwearing\s+nothing\b|\bwith\s+no\s+clothes\b|\bbare\s+skin\b", re.I)


def strips_bare(text):
    """Does this beat say somebody ends up with no clothes on?"""
    return bool(_NAKED_CUE.search(text or ""))


# Said once, in place of listing every garment separately. Restraints are named
# because they do NOT come off here, and a sentence about everything coming off would
# otherwise be read as including them.
BARE_HOLD = (" Everything worn comes off during this shot and is away by the last "
             "frame, leaving bare skin from the shoulders down; whatever is fastened "
             "to the body stays fastened exactly as it was.")


def missing_removals(beat, scene, already):
    """Garment words the SCENE still describes, in a beat whose prose takes
    something off and which carries no `remove:` line for them.

    Reports; never acts."""
    if not scene or not _REMOVAL_PROSE.search(beat or ""):
        return []
    hits = []
    for word in re.findall(r"\b[\w-]{4,}\b", beat or ""):
        low = word.lower().strip("-")
        if not low or low in already or low in hits or low in _NOT_A_GARMENT:
            continue
        # Same discipline as the inference: the head of an entry, not a modifier
        # inside one. Reporting "back" and "her" as unremoved garments is noise
        # that buries the one line that matters.
        if _is_entry_head(word, scene):
            hits.append(low)
    # Words that are in the scene because they are the PERSON or the place, not
    # something worn. A name or a room is not a garment.
    return [h for h in hits if not re.search(
        r"\b" + re.escape(h) + r"\b\s*(?:is|was|walks|stands|sits)", scene, re.I)]


def extract_directives(beat):
    """(beat text with directive lines taken out, [removed tokens], [added phrases]).

    `add:` is the other half of `remove:`, and it exists because of a specific
    failure: a scene that lists every layer at once -- coat, jumper, shirt --
    tells the model the character is wearing all of them simultaneously, with
    nothing saying which is hidden. The keyframe pins the first frame, so early
    frames look right; by the last frame only the text is governing, and the under
    layer starts showing through the top one.

    So describe what is VISIBLE, and add a layer when it becomes visible:

        Dan cuts off her jacket and throws it away.
        remove: jacket
        add: her white shirt underneath

    The added phrase is appended to the scene from that shot onward, in your words,
    unchanged."""
    removed, added = [], []

    def take_removed(m):
        removed.extend(t.strip() for t in m.group(1).split(",") if t.strip())
        return ""

    def take_added(m):
        phrase = m.group(1).strip()
        if phrase:
            added.append(phrase)
        return ""

    body = _ADD_LINE.sub(take_added, _REMOVE_LINE.sub(take_removed, beat or ""))
    return re.sub(r"\n{2,}", "\n", body).strip(), removed, added


def extract_removals(beat):
    """Back-compatible shim: (body, removed tokens)."""
    body, removed, _ = extract_directives(beat)
    return body, removed


def scrub_removed(text, tokens):
    """Drop the parts of `text` that name a removed item.

    Comma-separated fragments first, because that is how a scene lists what someone
    is wearing ("blonde, 20, grey jacket, black boots"). A sentence that is left
    with no words at all is dropped whole, so "She wears a red coat." disappears
    rather than becoming a stub."""
    if not text or not tokens:
        return text
    live = [t for t in tokens if t]
    pats = [re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in live]
    # A scene lists what someone wears as comma-separated NOUN PHRASES ("blonde,
    # pale blue cotton shirt, heavy black waxed canvas jacket"). For those, the
    # whole entry goes: trimming a fixed number of modifiers off the front left
    # orphans like "heavy black waxed" sitting in the list, and an orphan
    # description is read as some garment -- which is a garment coming back.
    #
    # A fragment with a VERB in it is prose, not a list entry, and there the entry
    # is only part of the sentence, so it gets the surgical treatment below.
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        frags = sent.split(",")
        out_frags = []
        for frag in frags:
            if any(p.search(frag) for p in pats) and not _HAS_VERB.search(frag):
                # Restraint hardware is not clothing. An entry describing it goes
                # only when a token NAMES it: dropping "wrists handcuffed behind
                # her back" whole because a removal named "back" takes the cuffs
                # out of the prompt entirely, and hardware absent from the text
                # renders absent. Keep the fragment; the surgical pass below still
                # trims the token's own words out of it.
                if restraint_present(frag) and not any(_RESTRAINT_WORD.match(t)
                                                       for t in live):
                    out_frags.append(frag)
                    continue
                # One entry can carry two garments joined by "and" -- "a grey coat
                # and black boots". Dropping it whole takes the innocent one with
                # it, and an undescribed garment is one the model re-invents. So
                # drop only the side that names the removed item.
                sides = re.split(r"\s+\band\b\s+", frag, flags=re.I)
                gone = [s for s in sides if any(p.search(s) for p in pats)]
                keep = [s for s in sides if s not in gone] if len(sides) > 1 else []
                # A PERSON's tag must not leave with a garment that happened to share
                # its fragment -- losing it costs that shot its identity reference.
                # An OBJECT's tag is the opposite case: "a silver locket <Picture 2>"
                # is a picture OF the locket, so when the locket comes off the tag has
                # to come off with it. Left behind it kept asserting the thing that
                # was just removed, and a tag pointing at a picture nothing in the
                # text accounts for is also how a spare subject gets drawn.
                #
                # The person's tag is the one in the fragment carrying their LABEL --
                # "Nora: <Picture 1>" -- because that is where a sheet entry puts it.
                # Any other tag belongs to whatever it is standing next to.
                tags = [n for s in (gone or [frag]) for n in person_tags(s)]
                piece = " and ".join(k for k in keep if k.strip())
                if tags:
                    piece = ((piece + " ") if piece.strip() else "") + \
                            " ".join(f"<Picture {n}>" for n in tags)
                if piece.strip():
                    out_frags.append(piece)
                continue                      # the rest of the entry goes
            out_frags.append(frag)
        rebuilt = ",".join(out_frags)
        # A sentence's full stop lives on its LAST fragment. Dropping that fragment
        # -- which is exactly what removing the last-listed garment does -- takes the
        # full stop with it and runs the sentence into the next one: "blue eyes
        # Wrists cuffed behind back." Put the terminator back.
        end = re.search(r"([.!?])\s*$", sent)
        if end and rebuilt.strip() and not re.search(r"[.!?]\s*$", rebuilt):
            rebuilt = rebuilt.rstrip().rstrip(",;") + end.group(1)
        kept.append(rebuilt)
    out = " ".join(k for k in kept if k.strip())
    for t in live:
        # The item and the words that belong to it -- an article and up to two
        # modifiers -- and nothing else. Deleting the whole comma fragment took
        # neighbours with it: removing "jacket" from "a grey jacket over a white
        # shirt" deleted the shirt too, and an undescribed garment is one the model
        # re-invents, which looks like the clothing changing by itself.
        out = re.sub(r"\b(?:(?:a|an|the|her|his|their)\s+)?(?:[\w-]+\s+){0,2}"
                     + re.escape(t) + r"\b", "", out, flags=re.I)
    # Tidy what the deletion left behind, without touching anything it did not.
    # Twice: removing a stranded verb can strand the conjunction in front of it
    # ("Kate is 20 and wears a grey jacket" -> "... and wears" -> "... and").
    for _ in range(2):
        out = re.sub(r"\s{2,}", " ", out)
        # "wearing and black boots" / "wears over a white shirt"
        out = re.sub(r"\b(wearing|wears|in|dressed)\s+(?:and|over|under|with)\s+",
                     r"\1 ", out, flags=re.I)
        # a clothing verb with nothing left to govern
        out = re.sub(r"\s*\b(?:wearing|wears|dressed in)\s*(?=[.,;]|$)", "", out, flags=re.I)
        # a connector left hanging before punctuation or the end
        out = re.sub(r"\s+(?:and|over|under|with)\s*(?=[.,;]|$)", "", out, flags=re.I)
        out = re.sub(r",\s*(?=,)", "", out)
        out = re.sub(r"\s*,\s*(?=[.!?])", "", out)
        out = re.sub(r"\s+([.,;!?])", r"\1", out)
        # A dropped entry can leave its comma flush against the next one. Not
        # before a digit, so a thousands separator survives ("1,500").
        out = re.sub(r",(?=[^\s,\d])", ", ", out)
    out = re.sub(r"\s{2,}", " ", out)
    # Drop a sentence the deletion emptied, and one it reduced to a bare subject
    # ("She wears a red coat." -> "She.") -- which describes nobody and is one more
    # mention of a person, which is its own problem.
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+", out):
        s = sent.strip()
        if not re.search(r"[A-Za-z0-9]", s):
            continue
        if re.fullmatch(r"(?:he|she|they|it|[A-Z][\w-]*)\s*[.!?]?", s, re.I):
            continue
        kept.append(s if s[-1] in ".!?" else s + ".")
    return " ".join(kept).strip()


# --- upscaling ---------------------------------------------------------------

def _resize_short_edge(frames, target, method="lanczos"):
    """Resize a [B,H,W,C] frame batch so its short edge == target (keeping aspect,
    snapped to /32). Plain high-quality resize -- enlarges, doesn't add detail."""
    b, h, w, c = frames.shape
    if min(h, w) == target:
        return frames
    if h <= w:
        nh = target; nw = max(32, int(round(target * w / h / 32) * 32))
    else:
        nw = target; nh = max(32, int(round(target * h / w / 32) * 32))
    s = frames.movedim(-1, 1)
    s = comfy.utils.common_upscale(s, nw, nh, method, "disabled")
    return s.movedim(1, -1)


def _upscale_frames(frames, mode, model_name, target_short_edge, batch=4):
    """Optional post-pass upscale of the finished frames (on CPU).
      mode 'model'   : run a ComfyUI upscale model (Real-ESRGAN/UltraSharp class)
                       via the registered loader+apply nodes, chunked with cleanup
                       so 2000+ frames don't OOM; then fit to target short edge.
      mode 'rtx'     : NVIDIA RTX Video Super Resolution (Tensor Cores; fastest,
                       best quality for video -- needs Nvidia_RTX_Nodes_ComfyUI).
      mode 'lanczos' : plain high-quality resize to the target short edge.
    Any failure falls back to lanczos (or the raw frames), so it never breaks a
    render. Returns (frames, note). NOTE: this SHARPENS/ENLARGES; it does not
    reconstruct video detail the way a second-model (LTX 2.3) pass does."""
    if mode == "off" or frames is None or getattr(frames, "shape", [0])[0] == 0:
        return frames, ""
    note = ""
    if mode == "rtx":
        # NVIDIA RTX Video Super Resolution (Comfy-Org/Nvidia_RTX_Nodes_ComfyUI).
        # Runs on RTX Tensor Cores -- far faster than ESRGAN-class models and
        # generally cleaner on video, though like them it enhances/enlarges rather
        # than reconstructing detail (an LTX 2.3 re-generation does that).
        try:
            rtx = (_find_node(["rtx", "video", "super"]) or _find_node(["rtxvideosuperresolution"])
                   or _find_node(["rtx", "upscale"]))
            if rtx is None:
                raise RuntimeError("RTX node not installed (Nvidia_RTX_Nodes_ComfyUI)")
            scale = 2
            if target_short_edge and int(target_short_edge) > 0:
                cur = min(frames.shape[1], frames.shape[2])
                if cur > 0:
                    scale = max(1, min(4, int(round(int(target_short_edge) / cur))))
            out = []
            n = frames.shape[0]
            step = max(1, int(batch))
            for st in range(0, n, step):
                part = frames[st:st + step]
                res = None
                for kw in ({"image": part, "scale": scale}, {"images": part, "scale": scale},
                           {"image": part, "scale_factor": scale}, {"image": part}):
                    try:
                        res = _invoke_node(rtx, **kw); break
                    except TypeError:
                        continue
                if res is None:
                    raise RuntimeError("RTX node signature not recognized")
                out.append(res.detach().to("cpu"))
                del res, part
                _deep_cleanup()
            frames = torch.cat(out, dim=0)
            note = f"RTX Video Super Resolution x{scale}"
            if target_short_edge and int(target_short_edge) > 0:
                frames = _resize_short_edge(frames, int(target_short_edge))
                note += f"; fit to {int(target_short_edge)}px short edge"
            return frames, note
        except Exception as e:
            mode = "model"
            note = f"RTX upscale unavailable ({e}); fell back to model/lanczos"
    if mode == "model" and model_name and model_name != "none":
        try:
            loader = _find_node(["upscale", "model", "load"]) or _find_node(["loadupscalemodel"])
            applier = _find_node(["imageupscale", "model"]) or _find_node(["upscaleimageusingmodel"])
            if loader is None or applier is None:
                raise RuntimeError("upscale-model nodes not found")
            up_model = _invoke_node(loader, model_name=model_name)
            out = []
            n = frames.shape[0]
            for s in range(0, n, max(1, int(batch))):
                part = frames[s:s + max(1, int(batch))]
                res = _invoke_node(applier, upscale_model=up_model, image=part)
                out.append(res.detach().to("cpu"))
                del res, part
                _deep_cleanup()
            frames = torch.cat(out, dim=0)
            note = f"upscaled with {model_name}"
        except Exception as e:
            mode = "lanczos"
            note = f"model upscale unavailable ({e}); used lanczos"
    if target_short_edge and int(target_short_edge) > 0:
        try:
            frames = _resize_short_edge(frames, int(target_short_edge))
            note = (note + "; " if note else "") + f"fit to {int(target_short_edge)}px short edge"
        except Exception as e:
            note = (note + "; " if note else "") + f"resize failed ({e})"
    elif mode == "lanczos" and not note:
        note = "lanczos selected but no target set -> unchanged"
    return frames, note


def _upscale_model_list():
    """Filenames in models/upscale_models, plus 'none'. Read fresh at INPUT_TYPES
    time so newly-added models show up on a graph reload."""
    try:
        import folder_paths
        return ["none"] + list(folder_paths.get_filename_list("upscale_models"))
    except Exception:
        return ["none"]


def upscale_video_latent(video, model_name, scale):
    """(upscaled_video_latent, note). Never raises -- a failure returns the input.

    Spatial only: the temporal length comes back unchanged, which is what lets this
    sit between sampling and decode without touching the audio half or the frame
    count the rest of the chain has already committed to."""
    if not model_name or model_name == "off" or float(scale) <= 1.0:
        return video, ""
    cls = latent_upscaler_node()
    if cls is None:
        return video, ("latent_upscale is set but the 'Minimax H3 Latent Upscaler' node pack is "
                       "not installed, so the shots were rendered at their sampled size. Install "
                       "Comfyui_Minimax_h3_latent_Upscaler, or set latent_upscale to 'off'")
    try:
        before = tuple(video.shape)
        # Its UpscaleMode is a str-Enum, so the literal VALUE compares equal without
        # importing the pack. Read the enum off the class when it is reachable, and
        # fall back to the literal -- hardcoding a foreign string is the fragile part
        # of this integration, so it is not the only path.
        mode_val = "scale by multiplier"
        try:
            mode_val = sys.modules[cls.__module__].UpscaleMode.SCALE_BY
        except Exception:
            pass
        out = _invoke_node(cls, latent={"samples": video},
                           model_name=model_name,
                           mode={"mode": mode_val, "scale": float(scale)},
                           align=32, device="cuda", precision="fp16")
        up = out["samples"] if isinstance(out, dict) else out
        if up is None or up.dim() != video.dim() or up.shape[2] != video.shape[2]:
            # A temporal change would desync the audio half and the frame count.
            return video, ("the latent upscaler returned an unexpected shape, so the shot was "
                           "left at its sampled size")
        return up.to(video.dtype), (f"latent upscale {model_name} x{float(scale):g}: "
                                    f"{before[-2]}x{before[-1]} -> {up.shape[-2]}x{up.shape[-1]} "
                                    f"latent cells per frame, sampled small and decoded large")
    except Exception as e:
        return video, (f"latent upscale failed ({type(e).__name__}), so the shots were rendered "
                       f"at their sampled size")


def _latent_upscale_model_list():
    """H3 latent-upscaler weights in models/latent_upscale_models, plus 'off'.

    Filtered to H3 builds: that folder also holds LTX spatial/temporal upscalers,
    and offering one here would let it be picked for a model it cannot take -- the
    first conv is [512, 24, 3, 3, 3] and 24 is H3's latents_dim specifically.

    Listed whether or not the node pack that RUNS them is installed. The widget has
    to exist unconditionally or a saved workflow would lose its widget positions the
    moment the pack was uninstalled; being unable to run is handled at render time."""
    try:
        import folder_paths
        d = os.path.join(folder_paths.models_dir, "latent_upscale_models")
        names = [f for f in sorted(os.listdir(d))
                 if f.lower().endswith((".pth", ".safetensors"))
                 and ("minimax" in f.lower() or "h3" in f.lower())]
    except Exception:
        names = []
    return ["off"] + names


def _find_node(substrings):
    """Find a registered node whose key contains all of `substrings` (lowercased)."""
    maps = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    for k, v in maps.items():
        kl = k.lower()
        if all(s in kl for s in substrings):
            return v
    return None


def _invoke_node(cls, **kwargs):
    """Call a registered ComfyUI node (V1 FUNCTION or V3 execute) with kwargs and
    return its first output. Used to reuse ComfyUI's own upscale-model loader/apply
    so we don't reimplement spandrel loading or tiled scaling."""
    inst = cls()
    fn = None
    if getattr(cls, "FUNCTION", None) and hasattr(inst, cls.FUNCTION):
        fn = getattr(inst, cls.FUNCTION)
    else:
        for cand in ("execute", "upscale", "load_model", "load"):
            if hasattr(inst, cand):
                fn = getattr(inst, cand); break
    if fn is None:
        raise RuntimeError("no callable entrypoint")
    out = fn(**kwargs)
    out = getattr(out, "result", out)
    return out[0] if isinstance(out, (tuple, list)) else out


def latent_upscaler_node():
    return _find_node(["minimaxh3latentupscaler", "3d"]) or _find_node(["minimaxh3latentupscaler"])


# --- one shot's conditioning ------------------------------------------------

def build_conditioning(clip, vae, audio_vae, prompt, width, height, length,
                       handoff=None, refs=None,
                       ref_noise_aug=0.999, silent=False, ref_image_size="match"):
    """Text + references + keyframe for a single shot.

    THE ONE RULE from H3's layout: a shot's conditioning rows are packed in the
    order the tokenizer is given them, and tokenize_with_weights is either/or --
    passing minimax_ref_items makes it ignore `images` outright. So a reference and
    a keyframe cannot be handed over separately; whatever the encoder is to see goes
    in one list, numbered by position.

    So one roster, and it has to be readable under ONE format. A shot with a keyframe
    is fl2va -- the keyframe is <Picture 1> -- and reference images are dropped for
    that shot, because on fl2va slot 2 means the LAST frame rather than a second
    subject. See the comments below.
    """
    latent, fc = _empty_av_latent(width, height, length, H3_FPS)
    refs = [r for r in (refs or []) if r is not None]

    hand_img = None
    if handoff is not None:
        hand_img = _resize(handoff[:1], width, height, "disabled")

    # REFERENCES AND THE KEYFRAME RIDE TOGETHER. This is the arrangement the node
    # had before I broke it, and the reason is in ComfyUI's own layout:
    #
    #   model_base.py:2183-2191  cond_video_latents = keyframe latents THEN ref latents
    #   model.py PackedLayout    emits keyframe "cond" segments THEN ref "ref_img" ones
    #
    # The two orders agree, so both channels coexist. A shot takes its references AND
    # a real keyframe: the keyframe ANCHORS the first frame, which is what continuity
    # needs, while a reference only supplies identity. They are not alternatives.
    #
    # I had read "<Picture 1>" as MEANING the first frame on fl2va, and rearranged the
    # roster around that. It does not. Which image is the first frame is decided by
    # resolved_frame_index in minimax_keyframes, not by a label's number -- the labels
    # are only how the images are shown to the VLM, and what they have to line up with
    # is the <Picture N> tags in the prompt.
    #
    # So references come FIRST and keep slots 1..N, which is what a sheet line's
    # `Name: <Picture 1>, ...` points at, and the handoff is appended AFTER them where
    # it disturbs no numbering. It has to be in the list at all because
    # tokenize_with_weights is either/or: passing minimax_ref_items makes it ignore
    # `images` outright, so leaving the handoff out means the VLM is never shown where
    # the shot left off and re-imagines the scenery -- same place, new room.
    keyframe_ok = ref_noise_aug is None or float(ref_noise_aug) >= KEYFRAME_SAFE_AUG
    # One aug covers every visual condition row, references AND the keyframe. Below
    # KEYFRAME_SAFE_AUG the keyframe latent would be noised and labelled at the wrong
    # timestep, so the handoff stops being an anchor and rides as an extra reference
    # instead: weaker continuity, but nothing pretending to anchor while carrying noise.
    carry_as_ref = bool(hand_img is not None and refs and not keyframe_ok)

    enc_refs = refs + ([hand_img] if carry_as_ref else [])
    items, blocks = ([], [])
    if enc_refs:
        items, blocks = _build_ref_images(vae, enc_refs, width, height, ref_image_size)
    if hand_img is not None and not carry_as_ref:
        items = items + [{"type": "image", "data": hand_img}]

    if items:
        tokens = clip.tokenize(prompt, minimax_ref_items=items)
    else:
        tokens = clip.tokenize(prompt)
    cond = clip.encode_from_tokens_scheduled(tokens)

    vals = {}
    if blocks:
        vals["minimax_refs"] = blocks
        # How CLEAN the references are shown. One aug covers every conditioning
        # latent, keyframe included -- which is why softening references below
        # KEYFRAME_SAFE_AUG would soften the anchor too.
        if ref_noise_aug is not None:
            vals["minimax_visual_cond_noise_aug"] = float(ref_noise_aug)

    kfs = []
    if hand_img is not None and not carry_as_ref:
        kfs.append({"resolved_frame_index": 0,
                    "latent": _keyframe_latent(vae, hand_img)})
    # Silence on the audio branch for a shot with no scripted line. H3 is joint:
    # an unconditioned audio stream invents a voice and the picture lip-syncs to it,
    # and no sentence in the prompt outvotes a stream that has already decided
    # someone is talking. PackedLayout emits a video segment only when a keyframe
    # carries a `latent`, so an audio-only keyframe is legal and costs no frame.
    if silent and audio_vae is not None:
        sil = _silent_audio_latent(audio_vae, fc, H3_FPS)
        if sil is not None:
            kfs.append({"resolved_frame_index": 0, "audio_latent": sil})
    if kfs:
        vals["minimax_keyframes"] = kfs
    if vals:
        cond = node_helpers.conditioning_set_values(cond, vals)
    return cond, latent, fc, carry_as_ref


def sample_shot(model, cond, negative, latent, seed, steps, cfg, sampler_name,
                scheduler, sigmas=None):
    """One sampling pass. denoise is fixed at 1.0: partial denoise desyncs the
    joint audio/video schedule."""
    if sigmas is not None and len(sigmas):
        return _sample_on_sigmas(model, seed, cfg, sampler_name, cond, negative,
                                 latent, sigmas)
    (out,) = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                   cond, negative, latent, denoise=1.0)
    return out


_HERE = os.path.dirname(os.path.abspath(__file__))


# (default, min, max, cast) for every numeric widget, so a value that cannot be used
# as a number can be replaced by the one the widget was built with.
_WIDGET_RANGE = {
    "megapixels": (1.0, 0.0, 2.0, float),
    "shot_seconds": (10.0, 1.0, 15.0, float),
    "steps": (8, 1, 100, int),
    "cfg": (1.0, 1.0, 20.0, float),
    "shift_video": (12.0, 1.0, 20.0, float),
    "shift_audio": (3.0, 1.0, 20.0, float),
    "ref_noise_aug": (0.999, 0.5, 1.0, float),
    "latent_upscale_scale": (2.0, 1.0, 4.0, float),
    "upscale_target_short_edge": (0, 0, 4096, int),
    "upscale_batch": (4, 1, 64, int),
    "pace": (1.0, 0.25, 2.0, float),
}


def sane_widgets(values):
    """(repaired values, notes) for the numeric widgets.

    Saved workflows restore widget values BY POSITION, with no names stored. Remove or
    reorder a widget and every later value shifts up one, so a boolean can land in a
    FLOAT slot -- which is where a widget reading NaN comes from, and a NaN pace makes
    NaN shot lengths and a render that never starts.

    A value that will not become a finite number falls back to the widget's built-in
    default; one that is merely out of range is clamped. Reported either way, because
    silently substituting a number the user did not choose is how a wrong render looks
    like a broken node."""
    out, notes = dict(values), []
    for name, (default, lo, hi, cast) in _WIDGET_RANGE.items():
        if name not in out:
            continue
        raw = out[name]
        try:
            if isinstance(raw, bool):
                raise TypeError("a boolean is not a setting for this widget")
            num = float(raw)
            if num != num or num in (float("inf"), float("-inf")):
                raise ValueError("not a finite number")
        except (TypeError, ValueError):
            out[name] = default
            notes.append(f"{name} came in as {raw!r}, which is not a usable number, so "
                         f"the built-in default {default} was used. A saved workflow "
                         f"restores widget values by POSITION, so this usually means the "
                         f"node gained or lost a widget above this one -- open the node, "
                         f"set the values you want, and save the workflow again")
            continue
        clamped = min(max(num, lo), hi)
        if clamped != num:
            notes.append(f"{name} was {num:g}, outside {lo:g}..{hi:g}, so it was clamped "
                         f"to {clamped:g}")
        out[name] = cast(clamped)
    return out, notes


class H3LongVideos:
    """One prompt -> a chain of MiniMax-H3 shots, joined into one video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "forceInput": True,
                    "tooltip": "Paragraph 1 is the SCENE, prepended to every shot verbatim. "
                               "Every paragraph after it is one beat = one shot.\n\n"
                               "Nothing is rewritten. What you type is what the shot is told, "
                               "plus the scene line. Put a quoted \"line of dialogue\" in a beat "
                               "and that shot keeps its audio; beats without one are silenced."}),
                "resolution": (list(NATIVE_RES), {"default": "16:9",
                    "tooltip": "Aspect ratio. megapixels sets the size."}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "1.0 = 1024x1024 worth of pixels, H3's native budget. Lower is "
                               "faster and leaner; 0 keeps the preset's own dimensions. Cost "
                               "scales with latent cells and attention is quadratic in them."}),
                "shot_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5,
                    "tooltip": "Length of EVERY shot. Uniform on purpose: noise is drawn to the "
                               "latent's shape, so shots of different lengths get unrelated noise "
                               "from the same seed and the grain resets at every cut. Snapped to "
                               "H3's 17k+5 frame grid."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100,
                    "tooltip": "6-8 with a turbo/distill LoRA; 20+ without one."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "H3 is CFG-free. At 1.0 the negative prompt is never evaluated -- "
                               "which is why nothing here is phrased as a negation."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "One seed for the whole chain. Every shot is the same length, so "
                               "they share a noise field."}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip":
                    "Pins the opening frame of shot 1 -- the only shot with no previous frame to "
                    "continue from.\n\n"
                    "It pins the WHOLE frame, so give it a composed frame of the shot you want: "
                    "subject, pose, framing, background. A head-and-shoulders portrait wired here "
                    "makes shot 1 a head-and-shoulders portrait. An identity portrait belongs on "
                    "ref_image_1, which says who the person is without dictating the frame."}),
                "ref_image_1": ("IMAGE", {"tooltip":
                    "Identity reference, applied to every shot unless the prompt places it with a "
                    "<Picture 1> tag. Kept on every shot on purpose: it is the only fixed anchor a "
                    "long chain has, and without it shot 11 is drift piled on drift."}),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
                "negative": ("CONDITIONING", {"tooltip":
                    "Ignored at cfg 1.0, which is where H3 runs. Wired for completeness."}),
                "sigmas": ("SIGMAS", {"tooltip":
                    "An external schedule (PDD Acc's Apply node). Drives the sampler directly; "
                    "steps and scheduler are then only for the progress bar."}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Keep video:audio near 4:1. H3 carries the audio latent on the "
                               "video schedule scaled by that ratio; flattening it breaks audio."}),
                "apply_model_sampling": ("BOOLEAN", {"default": True,
                    "tooltip": "Patch the dual video/audio schedule inside the node. Turn off only "
                               "if you patch it upstream yourself."}),
                "silence_nonspeech": ("BOOLEAN", {"default": True,
                    "tooltip": "Anchor the audio branch to real silence on any shot with no quoted "
                               "line. H3 is joint -- an unconditioned audio stream invents a voice "
                               "and the picture lip-syncs to it. This conditions the stream itself "
                               "rather than asking the prompt to stop it."}),
                "trim_seam": ("BOOLEAN", {"default": True,
                    "tooltip": "Drop the first frame of every shot after the first: it is the "
                               "model's own reproduction of the keyframe, so it is a duplicate."}),
                "ref_noise_aug": ("FLOAT", {"default": 0.999, "min": 0.5, "max": 1.0, "step": 0.005,
                    "tooltip": "How CLEAN a reference is shown. 0.999 (H3's default) hands over a "
                               "noise-free image, which invites the model to REPRODUCE it -- "
                               "including its background and pose -- in the opening frames. Lower "
                               "says approximate: try 0.95, then 0.90. One aug covers every "
                               "conditioning latent, so below 0.99 the keyframe rides as a "
                               "reference instead of an anchor."}),
                "tiled_decode": ("BOOLEAN", {"default": True,
                    "tooltip": "Decode in tiles. The whole-clip decode is the single largest "
                               "allocation in a run and the usual point a big checkpoint spills."}),
                "cleanup_between_shots": ("BOOLEAN", {"default": True,
                    "tooltip": "Move each finished shot to system RAM and purge VRAM between "
                               "shots, so a long chain does not accumulate on the card."}),
                "latent_upscale": (_latent_upscale_model_list(), {"default": "off",
                    "tooltip": "Upscale each shot in LATENT space, between sampling and decode, "
                               "so the shot is SAMPLED small and only DECODED large. That is the "
                               "cheap one: cost scales with latent cells and attention is "
                               "quadratic in them, so sampling 512x512 and upscaling 2x is far "
                               "less work than sampling 1024x1024.\n\n"
                               "Model and nodes by LBH-123-AI; needs the separate Minimax H3 "
                               "Latent Upscaler pack and its weights in "
                               "models/latent_upscale_models. Without the pack this does nothing "
                               "and info says so. Spatial only, so frame count and audio are "
                               "untouched, and tiled decode is forced while it is on."}),
                "latent_upscale_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0,
                    "step": 0.05,
                    "tooltip": "Latent upscale factor on both axes. 2.0 doubles each side. "
                               "1.0 disables it as surely as 'off'."}),
                "upscale": (["off", "rtx", "model", "lanczos"], {"default": "off",
                    "tooltip": "Post-pass on the FINISHED frames, after the latent pass and after "
                               "the shots are joined. 'rtx' = NVIDIA RTX Video Super Resolution "
                               "(needs the Nvidia_RTX_Nodes_ComfyUI pack, falls back if absent); "
                               "'model' = an upscale model from upscale_models; 'lanczos' = a "
                               "plain resize. These ENLARGE; for real detail reconstruction from a "
                               "low-res render use a separate pass."}),
                # Explicit, not left to fall back to the list's first entry: the list
                # is built from what is installed, so leaving it implicit makes the
                # default depend on the machine.
                "upscale_model": (_upscale_model_list(), {"default": "none",
                    "tooltip": "Which model, when upscale = model. From models/upscale_models."}),
                "upscale_target_short_edge": ("INT", {"default": 0, "min": 0, "max": 4096,
                    "step": 32,
                    "tooltip": "Fit the result's short edge to this many pixels. 0 keeps the "
                               "model's own factor."}),
                "upscale_batch": ("INT", {"default": 4, "min": 1, "max": 64,
                    "tooltip": "Frames per chunk for the model upscale. Lower = less VRAM, "
                               "slower."}),
                "shot_length": (["from the beat", "fixed"], {"default": "from the beat",
                    "tooltip": "How long each shot is.\n\n"
                               "'from the beat' sizes every shot from what its own line "
                               "stages, capped by shot_seconds and floored at one action's "
                               "worth. A beat with one action stops getting a shot with room "
                               "for two -- which is what makes an action carry on past its "
                               "end, repeating itself on whatever is nearest once it has "
                               "run out of what it was given.\n\n"
                               "'fixed' gives every shot shot_seconds. Uniform lengths mean "
                               "uniform latent SHAPES, and noise is drawn to the shape -- so "
                               "one seed gives the whole chain one noise field and surface "
                               "detail does not reset at each cut. That consistency is what "
                               "you trade away for pacing.\n\n"
                               "The estimate leans short on purpose: a shot that ends before "
                               "its action does hands a mid-motion frame to the next shot, "
                               "which the chain continues from. A shot that outlasts its "
                               "action has to invent the rest."}),
                "auto_remove": ("BOOLEAN", {"default": True,
                    "tooltip": "Read removals out of the beat itself, so a garment comes "
                               "off without a 'remove:' line.\n\n"
                               "Two conditions, both required, because a wrong removal is "
                               "worse than a missed one: the beat has to contain a removal "
                               "verb, and the thing named has to be the HEAD of something "
                               "the SCENE already lists as worn -- not a modifier inside an "
                               "entry, not a body part, and never restraint hardware. Only "
                               "the verb's own object counts, the span up to the next clause "
                               "boundary, so 'pulls off her coat, showing the jumper' takes "
                               "off the coat and leaves the jumper.\n\n"
                               "info reports every removal it reads, by shot. An explicit "
                               "'remove:' line still works and is added to whatever is "
                               "inferred."}),
                "restart_after_removal": ("BOOLEAN", {"default": True,
                    "tooltip": "After a shot with a 'remove:', start the NEXT shot fresh "
                               "instead of continuing from that shot's last frame.\n\n"
                               "Every shot is anchored to the previous shot's last frame. If "
                               "the model does not finish taking the garment off inside its "
                               "own shot, that frame still shows it -- and a keyframe is a "
                               "PICTURE, which outvotes any sentence. Inherit it once and every "
                               "later shot inherits it too, with no wording able to undo it. "
                               "This breaks that inheritance at the one boundary where the "
                               "state changes.\n\n"
                               "The cost is a visible cut there, and that shot re-deriving its "
                               "pose and framing from the text. Turn it off if your removals do "
                               "complete on screen and you would rather keep the continuity."}),
                "hold_restraints": ("BOOLEAN", {"default": True,
                    "tooltip": "Once a restraint is put on, keep it whole. From the shot "
                               "that applies it onward, every shot carries one sentence: "
                               "every restraint stays whole and closed, fastened exactly as "
                               "it was put on. Cleared by a 'remove:' naming the hardware.\n\n"
                               "This is the ONE continuity fact the node asserts by itself, "
                               "because it is the one that cannot be recovered -- a cuff "
                               "that renders open is not a detail that drifted, it is the "
                               "scene ceasing to make sense. Everything else is yours to "
                               "write."}),
                "plan_only": ("BOOLEAN", {"default": False,
                    "tooltip": "Report the shot split, lengths and warnings without rendering."}),
                # Appended LAST on purpose. Saved workflows restore widget values by
                # POSITION, with no names stored, so inserting a widget anywhere above
                # this shifts every later value in every workflow already saved.
                "anchor": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Framing that belongs to the whole film -- look, camera, "
                               "lighting, location. Carried at the FRONT of every shot.\n\n"
                               "FILLING THIS IN MAKES EVERY PARAGRAPH OF THE PROMPT A "
                               "BEAT. The anchor is then the scene, so the prompt is "
                               "pure action and nothing is taken out of it to serve as "
                               "scene text.\n\n"
                               "Leave it empty and the first paragraph of the prompt is "
                               "the scene instead, as before. Use one or the other: with "
                               "both, put ALL the framing here, because the prompt's "
                               "first paragraph will be rendered as a shot."}),
                "character_memory": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Who is in the film and what they are wearing, re-stamped "
                               "into EVERY shot.\n\n"
                               "Write it as a sheet, one person per line:\n"
                               "  Maya: 27, silver hair, grey shorts, red jacket\n"
                               "  Jon: 34, navy overalls\n\n"
                               "This is what makes clothing hold across a chain. A "
                               "garment described in one beat is described in ONE shot; "
                               "every later shot then says nothing about it, and what "
                               "the model is not told, it invents -- which is a garment "
                               "changing colour, or coming back after it came off.\n\n"
                               "It is also what a removal scrubs. `remove:` and the "
                               "automatic inference take the item out of this sheet from "
                               "that shot onward, so the text stops describing what the "
                               "beat took off.\n\n"
                               "A `Name: ...` paragraph in the prompt itself is folded in "
                               "here automatically -- a sheet is not a beat, and spending "
                               "a shot rendering a description is the visible symptom."}),
                "character_guard": ("BOOLEAN", {"default": True,
                    "tooltip": "Describe only the people a beat actually involves.\n\n"
                               "The sheet has to be in every shot for clothing to hold. "
                               "But describing EVERYONE in every shot puts everyone in "
                               "every shot: a beat about one person renders two, because "
                               "the text standing beside it says the other one is there, "
                               "and a described person is a person the model draws.\n\n"
                               "A beat naming nobody keeps whoever the last one kept, so "
                               "'She lies still.' does not empty the frame. Off, every "
                               "sheet line goes into every shot. info names who each shot "
                               "kept."}),
                "pace": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.05,
                    "tooltip": "Scales how much screen time each beat is given, when "
                               "shot_length is 'from the beat'.\n\n"
                               "A shot longer than its action does not get filled with "
                               "MORE action -- the model performs the same action more "
                               "slowly to reach the end of the shot. That is what "
                               "slow-looking footage is. Below 1.0 shortens every shot "
                               "and the motion in it quickens; above 1.0 lengthens and "
                               "slows.\n\n"
                               "Try 0.75 if the movement drags. Shots are still floored "
                               "at one action's worth and capped by shot_seconds, and "
                               "'fixed' ignores this entirely. info reports the seconds "
                               "each staged action ends up with."}),
                "auto_sound": ("BOOLEAN", {"default": True,
                    "tooltip": "Give each shot the sound its own action implies.\n\n"
                               "H3 is joint, so the same prose conditions the audio "
                               "branch -- and a beat that says what happens has already "
                               "said what it sounds like. Walking gets footsteps, a "
                               "chain gets links dragging, scissors get blades through "
                               "fabric, a lock gets a lock closing.\n\n"
                               "Read from the BEAT only, never the scene: a chain "
                               "standing in the scene does not rattle in a shot where "
                               "nobody moves. Three sounds at most, so the shot gets a "
                               "cue rather than an inventory.\n\n"
                               "A beat that already describes its own sound is left "
                               "alone -- what you wrote wins. A shot given sound is also "
                               "not silenced, since it is now asking for audio. info "
                               "lists which shots got one."}),
                # APPENDED, like every widget before it. Saved workflows restore these
                # positionally with no names stored, so inserting one shifts every
                # value after it into the wrong control.
                "hold_scene_state": ("BOOLEAN", {"default": True,
                    "tooltip": "Put a described state at the first frame instead of "
                               "leaving it to be performed.\n\n"
                               "'A van with its doors closed' names a state and never "
                               "says when it is true. A video model asked for a door "
                               "renders what a door does, so the shot opens on the doors "
                               "open and the characters close them -- the state arrives "
                               "as the action, because that is the most interesting "
                               "event in the sentence.\n\n"
                               "Doors, gates, windows, curtains, blinds, shutters, "
                               "hatches, tailgates, lids and drawers. Two at most per "
                               "shot.\n\n"
                               "A beat that WORKS the thing is not held -- 'Mara opens the "
                               "doors' is asking for exactly that motion. It is given the "
                               "two ENDS of the change instead: shut at the first frame, "
                               "open by the last. Some distill LoRAs render an action "
                               "backwards, and a beat naming one state names neither end, "
                               "so the reverse answers it just as well. Verbs that go "
                               "either way -- pulls, draws, slides, swings -- get no "
                               "anchor, since a wrong one asks for the reversal rather "
                               "than allowing it.\n\n"
                               "Once a beat has changed a state, no later shot is told "
                               "the old one, even though the scene paragraph still says "
                               "it. Two sentences per shot at most, the two kinds sharing "
                               "that budget. info lists which shots got which."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("images", "audio", "info", "script", "frames_per_shot", "total_frames",
                    "shots", "video_seconds")
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"
    DESCRIPTION = ("Chain MiniMax-H3 shots into one continuous video with synchronised audio. "
                   "One paragraph per shot; the first paragraph is the scene. Your text is "
                   "passed through verbatim.")

    def run(self, model, clip, vae, audio_vae, prompt, resolution, megapixels, shot_seconds,
            steps, cfg, sampler_name, scheduler, seed,
            first_frame=None, ref_image_1=None, ref_image_2=None, ref_image_3=None,
            ref_image_4=None, negative=None, sigmas=None,
            shift_video=12.0, shift_audio=3.0, apply_model_sampling=True,
            silence_nonspeech=True, trim_seam=True, ref_noise_aug=0.999,
            tiled_decode=True, cleanup_between_shots=True, plan_only=False,
            latent_upscale="off", latent_upscale_scale=2.0,
            upscale="off", upscale_model="none", upscale_target_short_edge=0,
            upscale_batch=4, shot_length="from the beat", hold_restraints=True,
            restart_after_removal=True, auto_remove=True, anchor="", character_memory="",
            character_guard=True, pace=1.0, auto_sound=True, hold_scene_state=True,
            **_removed):
        # **_removed: a workflow saved with the old `save_defaults` widget still sends
        # it. Swallowed rather than raising, so an existing workflow keeps loading.

        notes = []
        # Before anything reads them. A widget value that arrives as NaN -- which is
        # what a positional shift in a saved workflow produces -- would otherwise flow
        # into the frame arithmetic and come out as a shot length of nan.
        _fixed, _fixnotes = sane_widgets(dict(
            megapixels=megapixels, shot_seconds=shot_seconds, steps=steps, cfg=cfg,
            shift_video=shift_video, shift_audio=shift_audio,
            ref_noise_aug=ref_noise_aug, latent_upscale_scale=latent_upscale_scale,
            upscale_target_short_edge=upscale_target_short_edge,
            upscale_batch=upscale_batch, pace=pace))
        megapixels, shot_seconds = _fixed["megapixels"], _fixed["shot_seconds"]
        steps, cfg = _fixed["steps"], _fixed["cfg"]
        shift_video, shift_audio = _fixed["shift_video"], _fixed["shift_audio"]
        ref_noise_aug = _fixed["ref_noise_aug"]
        latent_upscale_scale = _fixed["latent_upscale_scale"]
        upscale_target_short_edge = _fixed["upscale_target_short_edge"]
        upscale_batch, pace = _fixed["upscale_batch"], _fixed["pace"]
        notes.extend(_fixnotes)
        swap = flush_for_model_change(model)
        if swap:
            notes.append(swap)
        check_vae_wiring(vae, audio_vae)

        prompt, n_legacy = strip_legacy_fields(prompt)
        if n_legacy:
            notes.append(f"dropped {n_legacy} field-label line(s) left over from an older "
                         f"version of this node (overall_soundscape:, [Generation N] and the "
                         f"like) -- your text now goes to the model verbatim, and a label like "
                         f"that is read as text to put ON the picture")
        if (anchor or "").strip():
            # The anchor IS the scene, so nothing has to be taken out of the prompt to
            # be one, and every paragraph is a beat. Otherwise the first ACTION becomes
            # the scene: prepended to every shot, repeated to the end of the film, and
            # never given a shot of its own. A removal written in it can never stick
            # either, because the scene restates the garment on every later shot.
            scene, beats = "", paragraphs(prompt)
        else:
            scene, beats = split_beats(prompt)
        # A character sheet is not a beat. Pulled out of the beat list and folded into
        # the scene, so it is re-stamped into EVERY shot -- which is what makes a
        # removal stick and what stops a later shot describing no clothing at all.
        beats, sheet = pull_character_sheets(beats)
        # The sheet is kept APART from the rest of the scene: it is the part that
        # varies per shot, because only the people a beat involves should be
        # described in it. Everything else is stamped on every shot unchanged.
        sheet, _dupes = merge_sheets((character_memory or "").strip(), sheet)
        if _dupes:
            notes.append(
                f"{', '.join(_dupes)} described more than once -- character_memory and a "
                f"'Name:' paragraph in the prompt are the same channel by two routes, and "
                f"using both put the person in every shot twice. A model told about one "
                f"person twice renders two of them. Kept the character_memory entry and "
                f"dropped the duplicate")
        static = build_scene(anchor, scene, "", "")
        scene = build_scene(anchor, scene, "", sheet)      # the whole of it, for inference
        if sheet:
            notes.append(f"folded {sheet.count(chr(10)) + 1} character-sheet line(s) into "
                         f"the scene instead of spending a shot on them -- a sheet "
                         f"describes people, it does not stage anything, and it has to "
                         f"be in EVERY shot for a removal to have something to scrub")
        # Somebody the beats stage and the sheet never describes. Nothing in the shot
        # says who they are, so the model invents them -- and a beat whose only person
        # is undescribed falls back to the previous beat's cast, which describes
        # someone who is not in the shot and says nothing about the one who is.
        for _who, _in in unknown_people([extract_directives(b)[0] for b in beats],
                                        sheet).items():
            notes.append(
                f"shot(s) {', '.join(str(n) for n in _in)} name {_who}, who has no entry "
                f"in the character sheet. {_who} is IN those shots and nothing describes "
                f"them -- no age, no clothes, no face -- so the model invents them, "
                f"differently each time. Where that is the ONLY person a beat names, the "
                f"shot falls back to the previous beat's people, and then it describes "
                f"someone who is not in it and nobody who is. If {_who} is already on the "
                f"sheet under another name, use one name throughout; otherwise add "
                f"'{_who}: ...' to character_memory")
        # Account for every paragraph, so a beat that quietly went somewhere else is
        # visible. Two ways one disappears: it reads as a character sheet and is folded
        # into the scene, or it was never a separate paragraph to begin with.
        _given = len(paragraphs(prompt))
        _sheets = len(sheet_lines(sheet)) if sheet else 0
        notes.append(f"{_given} paragraph(s) in the prompt: {len(beats)} rendered as "
                     f"shots" + (f", {_sheets} folded in as character sheet(s)"
                                 if _sheets else "")
                     + ("" if (anchor or "").strip() else ", 1 kept as the scene"))
        # Paragraphs are separated by a BLANK line. Lines joined by a single newline
        # are ONE beat, so three actions written on three lines become one shot with
        # three actions in it, and two of them look like they were absorbed.
        _multi = [i for i, b in enumerate(beats, 1) if "\n" in b]
        if _multi:
            notes.append(
                f"shot(s) {', '.join(str(i) for i in _multi)} carry more than one line. "
                f"Paragraphs are separated by a BLANK line, so lines with only a single "
                f"newline between them are one beat and share one shot. If those were "
                f"meant to be separate shots, put an empty line between them")
        if not beats:
            raise RuntimeError("H3 Long Videos: no beat to render. Every paragraph after "
                               "the first is one shot; a character sheet ('Name: ...') "
                               "is folded into the scene and does not count as one.")

        w, h = scale_to_megapixels(*parse_resolution(resolution), megapixels)
        ceiling = align_frame_count(int(round(float(shot_seconds) * H3_FPS)))
        # 'remove:' lines take their item out of the SCENE from that shot onward, so
        # the scene stops describing a garment a beat has taken off. It applies to
        # the removing shot too: the keyframe already shows the garment on at the
        # start, and a description saying it is still worn is what puts it back.
        shots, speech, gone, shown = [], [], [], []
        sounded = []                # beats that ask for a sound of their own
        inferred_sound = []         # shots given one derived from their action
        restrained = posed = rigid_latched = False
        # Scenery whose state a beat has CHANGED. After that the node stops asserting
        # the state it was written with, because it is no longer the state: a van
        # opened in shot 2 must not be told it is shut in shot 3, and the scene
        # paragraph goes into every shot still saying "doors closed".
        state_acted = set()
        stated_shots = []           # shots given a state put at the first frame
        turned_shots = []           # shots given both ends of a staged change
        stripped_shots = set()      # 0-based shots that took something off
        # Names, so "lifts Kate onto the table" reads as moving a person rather than
        # an object. A sheet LABELS them, which beats scanning prose for capitals --
        # that way "Medium shadows" is not a member of the cast, and a name with an
        # inner capital (McKenna) is not missed.
        cast = re.findall(r"^\s*([A-Z][\w'’-]{1,24})\s*:", sheet or "", re.M)
        if not cast:
            cast = re.findall(r"\b[A-Z][a-z]{2,}\b", scene or "")
        # Which garment is under which, read from the script's own "takes A off to
        # expose B". A sheet lists every layer at once, and a layer the model is told
        # about is a layer it draws -- through the one on top of it.
        covers = infer_layers([extract_directives(b)[0] for b in beats], scene)
        if covers:
            notes.append("read as layers, from the script's own wording: "
                         + "; ".join(f"{u} under {o}" for u, o in covers.items())
                         + " -- each is left out of the scene text until the thing "
                           "over it comes off, so it is not described as visible "
                           "while it is covered")
        _pose = posture_note(scene, first_frame is not None)
        if _pose:
            notes.append(_pose)
        _ref = reference_note(len([r for r in (ref_image_1, ref_image_2, ref_image_3,
                                               ref_image_4) if r is not None]),
                              ref_noise_aug, first_frame is not None)
        if _ref:
            notes.append(_ref)
        # The acoustic of the space, read once: it is the same room in every shot.
        # The opening beat is the fallback: with `anchor` set there is no scene
        # paragraph, and an anchor describes the camera rather than the room.
        _opening = extract_directives(beats[0])[0] if beats else ""
        _room = room_tone(scene, _opening) if auto_sound else ""
        _room_src = "the scene" if room_tone(scene) else "the opening beat"
        if _room:
            notes.append(f"room tone read from {_room_src}: {_room}. It goes under the "
                         f"shots whose audio branch is already open -- ones with a line, "
                         f"or with a sound you described yourself -- so those are not "
                         f"conditioned on digital silence, and nothing real is that "
                         f"quiet. It can never OPEN a branch: a shot with no line and no "
                         f"sound of your own stays pinned to silence and carries no room "
                         f"tone either, because the clause would describe an acoustic the "
                         f"conditioning says is not there. That is what stops the mouth "
                         f"moving. H3 is joint, so a free branch fills itself with a "
                         f"voice and the face lip-syncs to the babble, and no wording "
                         f"suppresses that -- only the silent keyframe does, and it pins "
                         f"the whole shot rather than just its opening")
        active = []                 # the people the previous beat involved
        _seen_before = set()        # everyone a shot has described so far
        _returns = []               # (shot, names back after a shot away)
        _placed_shots = set()       # 0-based shots introducing somebody in position
        shot_cast = []              # the names each shot describes
        guard_words = beat_words = total_words = sound_words = 0
        for b in beats:
            body, toks, adds = extract_directives(b)
            # Who this beat involves, decided BEFORE the removals: a beat that
            # undresses somebody names no garment, so the wardrobe to clear is read
            # off their sheet entries -- and only theirs. Undressing one person must
            # not take the other one's clothes off.
            if character_guard:
                _was = list(active)
                shot_sheet, active = sheet_for_beat(sheet, body, active)
                if len(sheet_lines(sheet)) > len(sheet_lines(shot_sheet)):
                    notes.append(f"shot {len(shots) + 1} describes only "
                                 f"{', '.join(active) or 'the scene'} -- the rest of the "
                                 f"sheet is held back, because a person the text "
                                 f"describes is a person the model draws")
                # Somebody back after a shot away. The keyframe is the PREVIOUS shot's
                # last frame, so a person who was not in that shot is not in the
                # picture this one starts from -- their appearance is carried by the
                # sheet text and nothing else, and text drifts where a picture does
                # not. This is what "walks out of frame and comes back looking
                # different" is.
                for _grp, _who_all in unresolved_pronouns(sheet, body, _was):
                    notes.append(
                        f"shot {len(shots) + 1} says '{_grp}' and "
                        f"{' and '.join(_who_all)} all answer to it, so the guard could "
                        f"not tell which -- and it describes NEITHER rather than both, "
                        f"because naming somebody the beat did not is how an extra "
                        f"character walks into a shot. Write the name instead of the "
                        f"pronoun in that beat and it resolves")
                # First appearance, with the beat saying where they ARE rather than
                # staging them arriving. See the handoff decision in the render loop.
                _new = [n for n in active if n not in _seen_before]
                if _new and not arrives_in(body) and shots:
                    _placed_shots.add(len(shots))
                    notes.append(
                        f"shot {len(shots) + 1} introduces {', '.join(_new)} in "
                        f"position rather than arriving, so it starts FRESH instead of "
                        f"continuing from the previous shot's last frame -- that frame "
                        f"does not have them in it, and a keyframe is a picture, so "
                        f"they would have to appear out of nothing and travel to the "
                        f"spot the beat describes. Costs a cut where a new character "
                        f"appears. Write the entrance -- 'walks in', 'steps through' -- "
                        f"if you would rather they arrive on screen and keep the join")
                _back = [n for n in active if n not in _was and n in _seen_before]
                if _back:
                    _returns.append((len(shots) + 1, list(_back)))
                _seen_before.update(active)
            else:
                shot_sheet = sheet
            # Read the removal out of the beat itself. Explicit 'remove:' lines still
            # win and are added to whatever is inferred.
            if auto_remove:
                inferred = [t for t in infer_removals(body, scene)
                            if t not in toks and t not in gone]
                if inferred:
                    toks = list(toks) + inferred
                    notes.append(f"shot {len(shots) + 1}: read '{', '.join(inferred)}' as "
                                 f"coming off, from the beat's own wording")
            # "...strip out of their clothes, becoming naked" names nothing, so every
            # other path had nothing to take off and the scene went on listing the
            # whole wardrobe -- in every later shot, which is how the clothes came
            # back on. Here the garments are read off the sheet instead of the beat.
            bare = auto_remove and strips_bare(body)
            if bare:
                stripped = [g for g in garments_in(shot_sheet)
                            if g not in toks and g not in gone]
                if stripped:
                    toks = list(toks) + stripped
                    notes.append(
                        f"shot {len(shots) + 1} reads as undressing "
                        f"{', '.join(active) if character_guard and active else 'the cast'}"
                        f" completely, and the beat names no garment -- so the wardrobe was "
                        f"read off the character sheet and all of it taken off: "
                        f"{', '.join(stripped)}. Anything worn that is not in that list is "
                        f"still described as on; name it in a 'remove:' line if so")
                elif not gone:
                    notes.append(
                        f"shot {len(shots) + 1} reads as undressing completely, but no "
                        f"garment was recognised in the character sheet, so nothing was "
                        f"taken off and every later shot still describes the clothes. Add "
                        f"a 'remove:' line naming them")
            # A beat's own words go to the model verbatim. Naming a garment that came
            # off in an EARLIER beat puts it back -- the scene is clean, the removal
            # was honoured, and then the beat itself asks for it. The removing beat
            # names it legitimately, so only later ones are reported.
            revived = [t for t in gone if names_any(body, [t])]
            if revived:
                notes.append(
                    f"shot {len(shots) + 1} names {', '.join(revived)} in its own text, and "
                    f"that came off earlier. Beats are sent to the model word for word, so "
                    f"naming it puts it back on -- the scene no longer mentions it, but this "
                    f"beat does. Reword the beat if it should stay off")
            if toks:
                stripped_shots.add(len(shots))
                gone.extend(t for t in toks if t not in gone)
                # An added layer is subject to removal too: once the shirt comes off,
                # the phrase that introduced it goes with it, or the scene keeps
                # describing a garment that is no longer there. Retired HERE, at the
                # moment of removal, so it retires the phrases that exist NOW -- an
                # add written later is putting the thing back on and must survive.
                _retired = [a for a in shown if names_any(a, toks)]
                if _retired:
                    shown = [a for a in shown if a not in _retired]
                    notes.append(f"shot {len(shots) + 1} takes off something an earlier "
                                 f"'add:' had put on, so that line retires with it: "
                                 + "; ".join(_retired))
                notes.append(f"removed from the scene from shot {len(shots) + 1} on: "
                             + ", ".join(toks))
            maybe = missing_removals(body, scene, gone) if not auto_remove else []
            if maybe:
                notes.append(f"shot {len(shots) + 1} reads as taking something off, but the "
                             f"scene still describes {', '.join(maybe)} and there is no "
                             f"'remove:' line for it -- so every shot keeps saying it is worn. "
                             f"Add 'remove: {maybe[0]}' to that beat")
            if adds:
                shown.extend(a for a in adds if a not in shown)
                notes.append(f"added to the scene from shot {len(shots) + 1} on: "
                             + "; ".join(adds))
            # The scrub applies to the removing shot too -- but only because that
            # shot's KEYFRAME already shows the garment on at the start, so the text
            # saying it is worn would put it back at the end.
            #
            # A shot with no keyframe has no such picture. Scrubbing there deletes the
            # only statement that the garment was ever on, and the shot then says: it
            # is not worn, take it off, and the thing under it is already showing.
            # The model renders that contradiction as a garment half present -- open,
            # or partly cut -- with the layer beneath it on display.
            i_shot = len(shots)
            has_keyframe = ((i_shot > 0 or first_frame is not None)
                            and not (restart_after_removal
                                     and (i_shot - 1) in stripped_shots))
            visible = gone if has_keyframe else [g for g in gone if g not in toks]
            if toks and not has_keyframe:
                notes.append(f"shot {i_shot + 1} takes something off and has no keyframe, "
                             f"so {', '.join(toks)} stays described as worn HERE -- the "
                             f"text is the only thing saying it was on to start with. It "
                             f"is scrubbed from the next shot on")
            # A garment still underneath something stays out of the text: described,
            # it gets drawn, and it is drawn through whatever is over it.
            covered = hidden_layers(covers, visible)
            # Terminated, or the last sheet line welds onto the beat -- "grey coat
            # Maya lies still" -- and a name fused to the end of an attribute list is
            # read as one more item in it.
            shot_scene = scrub_removed(
                "\n".join(terminate_lines(p) for p in (static, shot_sheet) if p.strip()),
                visible + covered)
            # Retirement is handled at the moment of removal, above, so this is just
            # what is currently on. Filtering here against the whole history of `gone`
            # meant an add could never put anything BACK: the token stays in `gone`
            # for the rest of the film, so "add: her locket is back on" was suppressed
            # by the removal that took it off in the first place.
            live = list(shown)
            if live:
                tail = ". ".join(a.rstrip(".") for a in live) + "."
                tail = tail[0].upper() + tail[1:]
                shot_scene = f"{shot_scene} {tail}".strip() if shot_scene else tail
            # The removal has to FINISH inside this shot, because its last frame is
            # the next shot's keyframe. Stated only here; naming the garment again
            # later would put it back.
            #
            # A full strip says it once rather than reciting the wardrobe: listing
            # eight garments coming off is eight more mentions of clothing in a shot
            # whose point is that there is none.
            tail = BARE_HOLD if (bare and toks) else off_by_last_frame(toks)
            # Once hardware is on, it stays on. Latched, not re-detected: a beat that
            # does not mention the cuffs does not mean they came off, and a cuff that
            # renders open is not a detail that drifts -- it is the scene ceasing to
            # make sense. Cleared only by a `remove:` that names the hardware.
            if hold_restraints:
                if names_any(RESTRAINT_HOLD_KEY, toks) or any(
                        restraint_present(t) for t in toks):
                    restrained = posed = rigid_latched = False
                elif restraint_present(body) or restraint_present(shot_scene):
                    restrained = True
            # Rigidity latches like the hardware itself. Steel locked on in shot 1 is
            # still steel in shot 5, and a beat that does not happen to say "chain"
            # does not mean the chain became rope -- but tested per shot, that is
            # exactly what happened: the shot naming it got the rigid clause and every
            # shot after it fell back to the soft one. Which is where the slack came
            # back from.
            if restrained and rigid_hardware(f"{body} {shot_scene}"):
                rigid_latched = True
            # And a position that hardware was locked to enforce latches too: the chain
            # that put a body in a squat is still that length three shots later, so the
            # squat is still the position.
            if rigid_latched and forced_pose(f"{body} {shot_scene}"):
                posed = True
            # A turn shows a surface the keyframe never pinned, and the model fills
            # it from a clothed prior. Only on shots that turn, and only once there
            # is something to hold -- a removal already made, or hardware on.
            turn = TURN_HOLD if (turns_in(body, cast)
                                 and (gone or shown or restrained)) else ""
            # Going down with the hands fastened: say what takes the landing, or the
            # model frees the hands to break the fall and the hardware gives way.
            fall = FALL_HOLD if (restrained and falls_in(body)) else ""
            # Steel is not rope. Without being told, the model draws a chain slack --
            # sagging, stretching to wherever a limb is going, allowing movement the
            # hardware does not allow. Only where such hardware is actually named.
            rigid = restrained and rigid_latched
            # Where the hardware is holding a POSITION, its length is the reason the
            # position holds -- and a chain drawn with slack is room to stand out of it.
            chain = (CHAIN_POSE_HOLD if (rigid and posed)
                     else CHAIN_HOLD if rigid else "")
            # Hardware named with nowhere to sit. A collar with no neck beside it is a
            # band with no place to be, and it ends up on the head. Only where this
            # beat itself raises the item, and only when the text has not already put
            # it somewhere -- what you wrote wins.
            anchors = anchor_clause(unanchored_hardware(body))
            if anchors:
                notes.append(f"shot {i_shot + 1} names hardware with no body part beside "
                             f"it, so the shot says where it sits: "
                             f"{anchors.split(': ', 1)[1].rstrip('.')}")
            line = f"{shot_scene} {body}".strip() if shot_scene else body
            # A state the text asserts but does not stage. Read from the whole line,
            # because the van usually stands in the scene paragraph rather than in
            # the beat -- and suppressed for anything this beat is actually working,
            # since a shot that opens the doors is a shot about the doors opening.
            _pairs, _moves = [], []
            if hold_scene_state:
                _moves = state_changes(body)
                _acting = [_state_key(t) for t, _ in _moves]
                _pairs = [(t, s) for t, s in stated_states(line)
                          if _state_key(t) not in state_acted and _state_key(t) not in _acting]
            # Which end of the action is which. Some distill LoRAs render a staged
            # change backwards, and a beat that names one state names neither end.
            _turn = direction_anchor(_moves)
            # The two share a budget. Holding a state and anchoring a change are both
            # continuity, and four such sentences is a shot about its own continuity.
            _state = state_hold(_pairs[:max(0, 2 - _turn.count("first frame"))]) + _turn
            if _pairs:
                stated_shots.append(len(shots) + 1)
            if _turn:
                turned_shots.append(len(shots) + 1)
            # The beat and the hold asking for opposite things. Reported three times
            # running as "the doors keep opening", and every time the node text was
            # by then correct -- it was the beat staging an exit the doors have to
            # open for. Say it; do not touch the wording.
            if _pairs and exits_vehicle(body) and any(
                    _state_key(t) in ("door",) for t, _ in _pairs):
                notes.append(
                    f"shot {len(shots) + 1} says somebody gets OUT of a vehicle and also "
                    f"says the doors are closed. Those are opposite instructions and the "
                    f"beat wins: a person leaving a van opens a door to do it, so the "
                    f"doors open however firmly the text says they are shut. If they are "
                    f"meant to be shut the whole shot, the people cannot be leaving the "
                    f"vehicle in it -- write them already out and standing ('Mara and Dom "
                    f"stand behind the van, its rear doors closed'), or put the exit in "
                    f"its own earlier shot. Your wording is never edited, so this is "
                    f"yours to resolve.")
            # Latch what this beat changed, so no later shot re-asserts the old state.
            state_acted.update(_state_key(t) for t, _ in _moves)
            # The chain clause SUBSUMES the restraint hold -- it says "whole and closed"
            # itself. Emitting both said it twice, which is twice the stasis for one
            # guarantee.
            hold = chain if chain else (RESTRAINT_HOLD if restrained else "")
            # ...and say WHOSE. Unattributed, "every restraint stays fastened" is an
            # instruction about whoever is on screen, so hardware locked onto one
            # character turned up on the other, over their clothes. Read from the sheet
            # entries, which are what say who is wearing it.
            _wearers = [n for n in restraint_wearers(shot_sheet)
                        if not character_guard or n in active]
            hold = own_hold(hold, _wearers,
                            active if character_guard else
                            [n for n, _ in sheet_lines(shot_sheet) if n])
            # What you wrote wins: a beat that already describes its own sound is left
            # alone, and only one that describes none gets the sound its action implies.
            # ONLY WHAT THE AUTHOR WROTE OPENS THE AUDIO BRANCH.
            #
            # H3 is joint: the mouth follows the audio. Leave that branch free on a
            # shot with no line and it fills itself with a voice, and the face
            # lip-syncs to the babble. Text cannot stop it -- "the only sounds are
            # footsteps" was tried and the mouth still moved -- because the only thing
            # that actually settles the branch is CONDITIONING it, and the silent
            # keyframe pins the whole shot, not just its opening.
            #
            # So nothing this node infers may unsilence a shot. A quoted line is a
            # request for audio; a sound the AUTHOR described is a request for audio;
            # footsteps this file worked out from "walks in" is not, and neither is
            # room tone. That is the whole rule, and it is the only one that holds --
            # every version that let an inference open the branch babbled.
            _speaks = has_speech(body)
            _own = sound_described(body)
            # A beat staging EFFORT or vocal reaction is asking for a voice, and that
            # is read from the author's own verbs -- "thrashes", "writhes", "moans" --
            # so it belongs with a quoted line and a written sound, not with the things
            # this file infers. Silencing it says the person makes no sound, and a
            # person making no sound is rendered still: it is the flat, unreacting
            # face, and it is why a body under effort came out mute.
            _voiced = exertion_in(body)
            _will_silence = bool(silence_nonspeech and not _speaks and not _own
                                 and not _voiced)
            # The held scenery goes in, so the shot is not asked to keep the doors
            # shut and to sound like a door swinging in the same breath.
            heard = ([] if (not auto_sound or _own)
                     else sounds_for(body, held=[_state_key(t) for t, _ in _pairs]))
            if _will_silence:
                # The audio is pinned to silence for this shot's whole length, so a
                # sentence saying what it sounds like would describe an acoustic the
                # conditioning says is not there.
                heard = []
            elif auto_sound and _room:
                heard = heard + [_room]
            if heard:
                inferred_sound.append(len(shots) + 1)
            # The branch is free on this shot, so SOMETHING fills it. Naming the sound
            # as the only thing heard leaves nothing for a voice to be -- it is not
            # the guard, the silence is, but it is what shapes a branch that is
            # legitimately open. Positively phrased: "the only sound is X" says what
            # IS there, where "nobody speaks" asks the model to render an absence.
            _sound = sound_clause(heard, only=not _speaks)
            shot_text = (line + tail + anchors + _state + _sound + hold + fall + turn).strip()
            # Sound direction is not a continuity guard -- it asks for something to
            # HAPPEN rather than for something to stay as it is -- so it is counted
            # apart, or the balance report blames the wrong text for crowding the beat.
            sound_words += len(_sound.split())
            guard_words += (len(shot_text.split()) - len(_sound.split())
                            - len(f"{shot_scene} {body}".split()))
            beat_words += len(body.split())
            total_words += len(shot_text.split())
            shots.append(shot_text)
            shot_cast.append(list(active) if character_guard else [])
            speech.append(_speaks)
            # What the AUTHOR wrote, and nothing this file worked out. See above --
            # effort counts, because the verb staging it is theirs.
            sounded.append(_own or _voiced)

        # What share of a shot is the node talking rather than the script. Continuity
        # clauses all say some version of "this stays as it is", and enough of them
        # drown the one sentence describing what HAPPENS -- which renders as a shot
        # where nothing does. The previous node reached 96%; this is here so the creep
        # is visible before it gets there again.
        # Somebody back after a shot away, with nothing pictorial carrying them.
        if _returns:
            _lines = "; ".join(f"shot {n}: {', '.join(w)}" for n, w in _returns)
            _tagged_back = {w for _, ws in _returns for w in ws
                            if re.search(r"^\s*" + re.escape(w) + r"\s*:.*<\s*picture",
                                         sheet or "", re.I | re.M)}
            _bare = sorted({w for _, ws in _returns for w in ws} - _tagged_back)
            notes.append(
                f"back after a shot away -- {_lines}. Each shot starts from the PREVIOUS "
                f"shot's last frame, so somebody who was not in that shot is not in the "
                f"picture this one begins from: their appearance comes from the sheet "
                f"text and nothing else, and text drifts where a picture does not. That "
                f"is a character walking out of frame and coming back looking different"
                + (f". {', '.join(_bare)} " + ("has" if len(_bare) == 1 else "have")
                   + " no <Picture N> tag, so there is no picture of them anywhere in the "
                     "run -- tag a reference to them and it is carried into every shot "
                     "they are named in, this one included"
                   if _bare else
                   ". All of them carry a reference tag, which is what pins them here"))
        if total_words:
            notes.append(
                f"prompt balance: the beat is {100 * beat_words / total_words:.0f}% of "
                f"what each shot is told, continuity clauses "
                f"{100 * guard_words / total_words:.0f}%, sound "
                f"{100 * sound_words / total_words:.0f}%, scene and sheet the rest"
                + (" -- the guards are outweighing the action, which reads as a shot "
                   "where nothing happens. Fewer restraints named, or a beat with more "
                   "in it, shifts the balance back"
                   if guard_words > beat_words * 3 else ""))
        refs_all = [r for r in (ref_image_1, ref_image_2, ref_image_3, ref_image_4)
                    if r is not None]

        lens, len_note = plan_lengths(beats, ceiling, shot_length == "from the beat", pace)
        # How much of a SPEAKING shot the line does not cover. The branch is free for
        # the whole shot, so whatever the line does not fill is unconditioned audio in
        # a shot the model knows somebody is talking in -- which is where invented
        # speech after the line comes from. Reported per shot, because the fix is the
        # author's: a longer line, or a shorter shot.
        _tail = []
        for _i, _b in enumerate(beats):
            if _i >= len(lens) or not has_speech(_b):
                continue
            _words = (sum(len(q.split()) for q in _QUOTED.findall(_b))
                      + sum(len(q.split()) for q in _DIALOGUE_TAG.findall(_b)))
            _say = _words / WORDS_PER_SEC
            _shot = lens[_i] / H3_FPS
            if _shot - _say >= 3.0:
                _tail.append((_i + 1, _words, _say, _shot))
        if _tail:
            notes.append(
                "dialogue headroom -- "
                + "; ".join(f"shot {n}: {w} word(s), about {s:.1f}s of a {t:.1f}s shot"
                            for n, w, s, t in _tail)
                + ". The audio branch is open for the whole shot, so the seconds the "
                  "line does not fill are unconditioned in a shot the model already "
                  "knows has a voice in it -- that is where speech carries on after the "
                  "line, or turns into babble. Give the beat a longer line, or a "
                  "shorter shot: shot_length 'from the beat' sizes to the line, while "
                  "'fixed' gives every shot shot_seconds whatever the line needs")
        # Seconds of shot per staged action -- the number that decides whether the
        # motion looks brisk or stretched. A shot longer than its action is filled by
        # performing the action more slowly, not by inventing more of it.
        _clauses = sum(max(1, len([p for p in _CLAUSE_SPLIT.split(b)
                                   if p and len(p.split()) >= 2])) for b in beats)
        if _clauses and lens:
            _per = sum(lens) / H3_FPS / _clauses
            notes.append(
                f"pacing: {_per:.1f}s of shot per staged action across {len(beats)} "
                f"beat(s), at pace {float(pace):.2f}"
                + (" -- a staged action is usually 2 to 3 seconds on screen, and a shot "
                   "longer than its action is filled by performing it more slowly. Lower "
                   "pace for brisker movement" if _per > 3.5 else ""))
        if len(set(lens)) == 1:
            notes.append(f"{len(shots)} shot(s) x {lens[0]}f (~{lens[0] / H3_FPS:.1f}s) "
                         f"at {w}x{h} = ~{sum(lens) / H3_FPS:.1f}s total")
        else:
            notes.append(f"{len(shots)} shot(s) at {w}x{h}, sized per beat: "
                         + ", ".join(f"{n}f/{n / H3_FPS:.1f}s" for n in lens)
                         + f" = ~{sum(lens) / H3_FPS:.1f}s total")
        if len_note:
            notes.append(len_note)
        if stated_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in stated_shots)} describe scenery in a "
                f"state -- doors closed, curtains drawn -- so the shot is told that state "
                f"is already true at the first frame. A state written down and not placed "
                f"in time is a state the model can render by arriving at it, which is a "
                f"van whose doors open so somebody can close them. A beat that works the "
                f"thing itself is left alone, and once a beat has changed a state no "
                f"later shot is told the old one. Off with hold_scene_state.")
        if turned_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in turned_shots)} stage a change with a "
                f"direction -- something opened or shut -- so the shot is told both ends: "
                f"what is true at the first frame and what is true by the last. Some "
                f"distill LoRAs render an action backwards, and a beat naming one state "
                f"names neither end, so the reverse reads as an equally good answer. Verbs "
                f"that genuinely go either way -- pulls, draws, slides, swings -- get no "
                f"anchor, because a wrong one asks for the reversal instead of allowing "
                f"it. Reversal is likeliest in shot 1, which has no previous last frame "
                f"pinning where it starts; first_frame pins it. Off with hold_scene_state.")
        if inferred_sound:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in inferred_sound)} were given the "
                f"sound their own action implies -- H3 is joint, so the same prose "
                f"conditions the audio branch, and a beat that says what happens has "
                f"said what it sounds like. Read from the beat, never the scene, so a "
                f"chain standing in the scene does not rattle where nobody moves. A beat "
                f"that describes its own sound is left alone. This is TEXT ONLY and can "
                f"never unsilence a shot: it is added to shots whose audio branch is "
                f"already open, meaning ones with a line or with a sound you wrote "
                f"yourself. A shot with neither stays pinned to silence and gets no "
                f"sound sentence, because the mouth follows the audio and an inference "
                f"is not a good enough reason to let it move")
        n_silent = sum(1 for s, snd in zip(speech, sounded) if not s and not snd)
        n_kept = sum(1 for s, snd in zip(speech, sounded) if not s and snd)
        if silence_nonspeech and n_kept:
            notes.append(
                f"{n_kept} shot(s) have no line but either describe a sound IN THE BEAT or "
                f"stage EFFORT, so their audio is left free to make it -- writing the "
                f"sound, or the verb that produces one, is asking for audio on purpose. "
                f"Those are the only shots without a line "
                f"where the branch is open, and an open branch on a joint model can "
                f"still put a voice in the gap. If one of them babbles, that beat's own "
                f"sound wording is what opened it")
        if silence_nonspeech and n_silent:
            notes.append(
                f"{n_silent} shot(s) have no quoted line and no sound described, so they "
                f"are conditioned on real silence -- which is not 'no speech', it is 'no "
                f"sound at all': no footsteps, no room tone, nothing. H3 is joint, so the "
                f"way to score a scene is to DESCRIBE it in the prose: 'boots on concrete, "
                f"a chain dragging, a low hum off the strip light'. Write it into a beat "
                f"for that shot, or into the anchor to carry it through the film. Do not "
                f"use a label like 'sound:' -- a labelled line is read as text to draw")
            # The note that used to sit here warned that a beat staging effort was
            # being silenced, which read as a flat, unreacting face. It cannot happen
            # any more: effort opens the audio branch, because the verb staging it is
            # the author's. See _voiced in the shot loop.

        if first_frame is None:
            notes.append("no first_frame: shot 1 has nothing pinning its opening frame, so its "
                         "starting pose and framing come from the text and any reference")
        # Text in the frame. H3 draws letterforms when the prompt names them, and at
        # cfg 1 there is no negative prompt to take them back -- adding "no watermark"
        # to the positive only names it again, which is how a mention becomes a
        # presence cue. So: point at the words, and leave the decision to the author.
        # H3 has a caption channel of its own. A prompt carrying those tokens is
        # ASKING for text on the picture.
        if any(_CAPTION_TOKEN.search(s) for s in shots):
            notes.append("the prompt contains H3's caption/lyrics tokens "
                         "(<|caption_start|> and friends) -- those request text ON the "
                         "picture. Remove them unless you want subtitles burned in")
        # Quoted dialogue with no <d> marker. H3 distinguishes speech, captions and
        # lyrics with explicit tokens; unmarked quoted text is not identified as any
        # of them, and a model with a caption channel may render it rather than say
        # it. Worth trying if subtitles are appearing under spoken lines.
        n_bare = sum(1 for b in beats
                     if _QUOTED.search(b) and not _DIALOGUE_TAG.search(b))
        if n_bare:
            notes.append(f"{n_bare} beat(s) carry dialogue in plain quotes. H3 has its own "
                         f"dialogue marker -- <d>like this</d> -- and a caption channel "
                         f"besides. If spoken lines are coming out as on-screen subtitles, "
                         f"wrap them in <d>...</d> and compare")
        cued = sorted({m.group(0).lower() for s in shots for m in _TEXT_CUE.finditer(s)})
        if cued:
            notes.append(f"the prompt names on-screen text ({', '.join(cued)}) -- H3 draws "
                         f"letterforms when asked, and at cfg 1 no negative prompt can take "
                         f"them back. Remove the words if you do not want the text")
        # Each beat against ITS OWN shot length; thin_beats numbers from 1, so the
        # shot number is restored here.
        thin = [t.replace("shot 1:", f"shot {i + 1}:")
                for i, b in enumerate(beats)
                for t in thin_beats([b], lens[i] / H3_FPS)]
        if thin:
            notes.append(
                "THIN BEATS -- the shot outlasts what the beat gives it to do, and the "
                "cheapest way for the model to fill the rest is to CARRY ON with the "
                "action, repeating it on whatever is nearest: "
                + "; ".join(thin)
                + ". Give the beat a second action -- what happens after it -- or lower "
                "shot_seconds")
        if float(cfg) != 1.0:
            notes.append(f"cfg is {float(cfg):g}; H3 is CFG-free and expects 1.0")

        # Resolve the <Picture N> tags before `script` is written, so what you read is
        # what the model is given. Which roster they resolve against depends entirely
        # on the format -- see build_conditioning.
        # Tags PLACE the references. With none written anywhere, placing by tag would
        # place them nowhere -- a connected reference that silently does nothing at
        # all. The old node fell back rather than no-op, and so does this.
        _tagged = any(picture_tags(s) for s in shots)
        _tagged_names = {n for n, ln in sheet_lines(sheet) if n and picture_tags(ln)}
        if refs_all and not _tagged:
            notes.append(
                f"{len(refs_all)} reference image(s) connected and no <Picture N> tag "
                f"anywhere, so they go on EVERY shot -- placing by tag would place them "
                f"nowhere. To aim them, write the tag on the person they depict: 'Nora: "
                f"<Picture 1>, 34, she, ...'. Each then travels with that person into "
                f"the shots she is in, and only those")
        shot_refs_all = []
        for _i, _s in enumerate(shots):
            # The tag is the BINDING between a picture and the subject the prompt
            # describes, and it stays IN the text -- comfy_extras/nodes_minimax_h3.py:
            # "the prompt refers to them as <Picture i>", "Use the same tags when
            # prompting". Renumbered per shot, because the encoder numbers by the
            # order it receives images and a shot carrying only slot 2 receives that
            # image as <Picture 1>.
            if not _tagged:
                shot_refs_all.append(list(refs_all))
                continue
            _s, _r, _missing = resolve_tags(_s, refs_all)
            shots[_i] = _s
            shot_refs_all.append(_r)
            for _n in _missing:
                _msg = f"<Picture {_n}> names a slot with no image connected"
                if _msg not in notes:
                    notes.append(_msg)
        if refs_all:
            _named = sum(1 for s in shots if picture_tags(s))
            notes.append(
                f"{len(refs_all)} reference image(s) supply IDENTITY, and they go WHERE "
                f"TAGGED: every shot whose text names <Picture N> carries image N, which "
                f"is what holds a face across beats instead of letting it drift down the "
                f"keyframe chain. Put the tag on the person -- 'Nora: <Picture 1>, 34, "
                f"she, ...' -- and it travels with her. {_named} shot(s) claim one here. "
                f"References ride alongside the keyframe rather than instead of it: the "
                f"keyframe anchors the first frame, a reference only says who somebody "
                f"is, and ComfyUI packs both (keyframe rows then ref rows, in the same "
                f"order model_base builds the latents). References keep slots 1..N so the "
                f"tag points at the right image; the handoff is appended after them and "
                f"disturbs no numbering")
            if _named > 1 and float(ref_noise_aug) >= KEYFRAME_SAFE_AUG:
                notes.append(
                    f"a reference on {_named} shots at ref_noise_aug "
                    f"{float(ref_noise_aug):g} is the trade this makes. Near-clean, a "
                    f"reference asks the model to reproduce the PICTURE -- pose and "
                    f"framing, not only the face -- and on a shot that is not introducing "
                    f"the character that competes with the staging the beat describes: "
                    f"the referenced person can hold the portrait's gaze while anyone "
                    f"without a reference is placed relative to that composition and then "
                    f"travels to where the text put them. It is the price of the face "
                    f"holding. A hybrid fl2va/ref2va checkpoint is trained for reference "
                    f"conditioning and does not make this trade; on a plain fl2va one, "
                    f"lowering ref_noise_aug is the dial")
            if _named < len(shots):
                notes.append(
                    f"{len(shots) - _named} shot(s) name no <Picture N> at all, so they "
                    f"carry no reference. Claim it on the person it depicts -- 'Nora: "
                    f"<Picture 1>, 34, she, ...' -- and it travels with her into the shots "
                    f"she is in, and only those. A picture the prompt never refers to is "
                    f"read as ANOTHER subject")


        script = "\n---\n".join(f"[Shot {i}] {s}" for i, s in enumerate(shots, 1))
        info = " | ".join(notes)
        if plan_only:
            empty = torch.zeros((1, h, w, 3))
            return (empty, {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 44100},
                    "PLAN ONLY -- nothing rendered. " + info, script,
                    lens[0], 0, len(shots), 0.0)

        if apply_model_sampling:
            model, ms_note = apply_h3_model_sampling(model, shift_video, shift_audio)
            notes.append(ms_note)
        if negative is None:
            negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

        handoff = first_frame
        # Where the time actually goes. Sampling and decode trade off against each
        # other -- latent_upscale buys cheaper sampling and pays for it at decode,
        # and which side wins depends on `steps`. Reported so the trade is a
        # measurement rather than an argument.
        t_sample = t_decode = 0.0
        _aug_warned = False
        fresh = []
        t_start = time.perf_counter()
        vid_out, aud_out, sr = [], [], 44100
        av_fix = 0                  # samples of A/V drift corrected across the chain
        _captured = {}              # name -> a frame from the last shot they were in
        _captured_from = {}         # name -> which shot that frame came from
        _recovered = []             # (shot, name, source shot) actually pinned
        shot_detail = []            # (detail, contrast) per shot, on its last frame
        _deep_cleanup()

        for i, shot_prompt in enumerate(shots):
            silent = bool(silence_nonspeech and not speech[i] and not sounded[i])

            # A shot that follows a removal starts FRESH. Every shot is anchored to
            # the previous one's last frame, so if the model did not finish taking
            # the garment off inside its own shot, that frame still shows it -- and a
            # keyframe is a PICTURE, which outvotes any sentence. Inherit it once and
            # every later shot inherits it too, with no wording able to undo it.
            # Breaking the chain at the one boundary where the state changes costs a
            # cut exactly where a cut belongs.
            shot_handoff = handoff
            if restart_after_removal and (i - 1) in stripped_shots:
                shot_handoff = None
                fresh.append(i + 1)
            # ...and so does a shot that INTRODUCES somebody already in position.
            #
            # Same reasoning, same evidence. The keyframe is the previous shot's last
            # frame, and a character appearing for the first time is not in it. The
            # beat says where they are; the picture says they are nowhere. The picture
            # wins, so the model starts from a frame without them and has to put them
            # there during the shot -- which renders as the person arriving out of
            # nothing and then travelling to the spot the beat described.
            #
            # Only when the beat does NOT stage an entrance. "Dan walks in through the
            # side door" is a person who SHOULD arrive, and continuing from the frame
            # before is exactly right there. "Dan is already sitting on the crate" is
            # a person who should be there at the first frame, and there is no frame to
            # inherit that has him in it.
            elif i in _placed_shots:
                shot_handoff = None
                fresh.append(i + 1)

            # SOMEBODY BACK AFTER A SHOT AWAY, with no picture of them anywhere.
            #
            # This shot starts from the previous shot's last frame, and they were not
            # in that shot -- so nothing pictorial carries their appearance and the
            # sheet text is on its own. A frame from the last shot they WERE in fixes
            # that, and the node has one: it rendered it.
            #
            # Narrow on purpose. Only when this shot describes that person ALONE,
            # because the recovered frame contains whoever else was on screen when it
            # was taken, and an unexplained person in a reference is how a second one
            # gets drawn. A multi-character return is reported and left alone.
            #
            # Skipped for anyone with a <Picture N> tag: their own reference already
            # travels into every shot they are named in, and a second picture of the
            # same person is just a second picture.
            _extra = []
            _cast = shot_cast[i] if i < len(shot_cast) else []
            if len(_cast) == 1 and _cast[0] not in _tagged_names:
                _who = _cast[0]
                if any(n == i + 1 and _who in ws for n, ws in _returns) \
                        and _captured.get(_who) is not None:
                    _extra = [_captured[_who]]
                    _recovered.append((i + 1, _who, _captured_from.get(_who, 0)))
                    # CLAIM IT IN THE PROSE. A picture the prompt refers to is that
                    # subject; one it never mentions is ANOTHER subject. Sent
                    # unclaimed, a recovered frame of somebody is read as a second
                    # person who looks exactly like them -- same face, same clothes --
                    # standing beside the one the beat asked for.
                    #
                    # Its number is its place in the roster: the shot's own references
                    # first, this after them. The handoff follows and stays unclaimed,
                    # which is H3's own first-frame shape.
                    _n = len(shot_refs_all[i]) + 1
                    _tag = f"<Picture {_n}>"
                    if f"{_who}:" in shot_prompt:
                        shot_prompt = shot_prompt.replace(
                            f"{_who}:", f"{_who}: {_tag},", 1)
                    else:
                        shot_prompt = f"{shot_prompt} {_who} is the person in {_tag}."
            cond, latent, fc, demoted = build_conditioning(
                clip, vae, audio_vae, shot_prompt, w, h, lens[i],
                handoff=shot_handoff, refs=list(shot_refs_all[i]) + _extra,
                ref_noise_aug=ref_noise_aug, silent=silent)
            if demoted and not _aug_warned:
                _aug_warned = True
                notes.append(
                    f"ref_noise_aug is {float(ref_noise_aug):g}, below {KEYFRAME_SAFE_AUG:g} -- "
                    f"one aug covers references AND the keyframe, so at this value the "
                    f"anchor would be noised and mis-timestepped, and every shot after the "
                    f"first degrades while sampling. The handoff is riding as an extra "
                    f"reference instead: continuity is weaker but nothing is corrupted. "
                    f"Raise it to {KEYFRAME_SAFE_AUG:g}+ for a real keyframe")
            _evict_all_but(model)
            try:
                _t0 = time.perf_counter()
                out = sample_shot(model, cond, negative, latent, seed, steps, cfg,
                                  sampler_name, scheduler, sigmas)
                t_sample += time.perf_counter() - _t0
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom(e):
                    raise
                raise RuntimeError(
                    f"H3 Long Videos: shot {i + 1} of {len(shots)} ran out of VRAM while "
                    f"sampling. " + sampling_oom_help(w, h, fc, H3_FPS, megapixels)) from e

            # The video latent, for the latent upscale below. NOT used as the next
            # shot's keyframe -- see _keyframe_latent for why that failed.
            try:
                parts = out["samples"].unbind() if hasattr(out["samples"], "unbind") else None
            except Exception:
                parts = None

            # LATENT upscale, between sampling and decode: the shot is SAMPLED small
            # and only DECODED large, which is where the saving is -- cost scales with
            # latent cells and attention is quadratic in them. Note the handoff latent
            # was taken ABOVE, before this: the chain must inherit the sampled latent,
            # not the upscaler's reinterpretation of it, or that guess compounds.
            shot_tiled = tiled_decode
            pre_up = None            # the SAMPLED video latent, when upscaling ran
            if latent_upscale and latent_upscale != "off" and parts and len(parts) == 2:
                vid_up, up_note = upscale_video_latent(parts[0], latent_upscale,
                                                       latent_upscale_scale)
                if vid_up is not parts[0]:
                    pre_up = parts[0]
                    out["samples"] = comfy.nested_tensor.NestedTensor((vid_up, parts[1]))
                    shot_tiled = True      # a 2x latent is ~4x the decode memory
                if up_note and up_note not in notes:
                    notes.append(up_note)

            _t0 = time.perf_counter()
            # The DiT goes so the decode fits; the two VAEs stay, because both are
            # used in the next two lines and evicting them only buys a reload.
            imgs = _decode_video(vae, out, shot_tiled, free_first=model,
                                 keep=(vae, audio_vae))
            wav = _decode_audio(audio_vae, out)
            t_decode += time.perf_counter() - _t0
            sr = wav["sample_rate"]
            del out

            # The chain must not inherit the UPSCALER's reinterpretation. The shot's
            # own frames stay upscaled, but the handoff comes from the sampled latent
            # -- otherwise every boundary hands on an upscaled-then-downscaled frame,
            # and eleven shots of that compounds into colour cast and mush.
            hand_src = imgs
            if pre_up is not None:
                try:
                    n = min(int(pre_up.shape[2]), HANDOFF_LATENT_TAIL)
                    tail = _decode_video(vae, {"samples": pre_up[:, :, -n:].contiguous()},
                                         True)
                    if tail is not None and tail.shape[0] > 0:
                        hand_src = tail
                except Exception:
                    pass                  # fall back to the upscaled frames
            # Clamp before it becomes a keyframe. A decode can land slightly outside
            # 0..1, and feeding that back in to be re-encoded every boundary is a
            # drift that accumulates rather than cancels.
            handoff = hand_src[-1:].detach().clamp(0.0, 1.0).to("cpu", copy=True)
            # Keep a frame for the shot they come back on -- but ONLY from a shot that
            # was theirs alone.
            #
            # A frame is a picture of everyone who was in it. Captured from a shot with
            # two people and sent later as a reference, it brings the other one back
            # into a shot that does not call for them. That is the second character
            # turning up uninvited, and it was this code: the destination was guarded
            # (the return shot has to describe one person) and the SOURCE was not.
            #
            # The MIDDLE frame, not the last: somebody walking out during the shot is
            # gone by the last frame -- which is the whole failure -- and somebody
            # walking in is missing from the first.
            try:
                if (hand_src.shape[0] and shot_cast and i < len(shot_cast)
                        and len(shot_cast[i]) == 1):
                    _mid = hand_src.shape[0] // 2
                    _keep = hand_src[_mid:_mid + 1].detach().clamp(0.0, 1.0).to(
                        "cpu", copy=True)
                    for _who in shot_cast[i]:
                        _captured[_who] = _keep
                        _captured_from[_who] = i + 1
            except Exception:
                pass                       # a recovered frame is a nicety, not the render
            del hand_src
            if trim_seam and i > 0:
                imgs = imgs[1:]
                wav["waveform"] = wav["waveform"][..., max(0, round(sr / H3_FPS)):]
            # Make the sound exactly as long as the picture it belongs to.
            #
            # The audio latent count is round(frames / 24 * 40), which lands exactly
            # only when the frame count divides by 3 -- so most of H3's 17k+5 grid
            # leaves a shot's audio 8.3 ms longer or shorter than its video. On its own
            # that is inaudible. Concatenated it is not: with shots of equal length the
            # error carries the same sign every time and adds up, and eleven 73-frame
            # shots finish 92 ms out, which is plainly visible on a mouth.
            #
            # Correcting per shot rather than once at the end keeps every cut aligned
            # too, instead of only the final duration.
            want = int(round(imgs.shape[0] * sr / H3_FPS))
            have = int(wav["waveform"].shape[-1])
            if have > want:
                wav["waveform"] = wav["waveform"][..., :want]
            elif have < want:
                shape = list(wav["waveform"].shape)
                shape[-1] = want - have
                wav["waveform"] = torch.cat(
                    [wav["waveform"], torch.zeros(shape, dtype=wav["waveform"].dtype,
                                                  device=wav["waveform"].device)], dim=-1)
            av_fix += have - want
            # Measured on the frame that becomes the next shot's keyframe, because
            # that is the one whose losses are inherited.
            try:
                if imgs is not None and imgs.shape[0]:
                    shot_detail.append(frame_detail(imgs[-1]))
            except Exception:
                pass
            # HALF PRECISION IN RAM. The finished shots are the largest thing this node
            # holds, and they compete with the weights for system memory -- ComfyUI
            # offloads models to RAM rather than discarding them, so a shot boundary is
            # a PCIe copy only while that RAM is there. Once the frames crowd the
            # weights out, the "reload" becomes a disk read, and on a chain that is
            # once per shot per model.
            #
            # A 107s chain at 1056x608 is ~2580 frames, 18.5GB as float32 and 9.3GB as
            # float16, against ~39GB of weights on a 64GB machine. That 9GB is the
            # difference between the weights staying resident and not.
            #
            # Free, not a trade: fp16 carries ~3 decimal digits over 0..1, and the
            # output is 8-bit. Converted back at the join, so nothing downstream sees
            # a different dtype.
            vid_out.append(imgs.to("cpu", torch.float16, copy=True)
                           if cleanup_between_shots else imgs)
            aud_out.append(wav["waveform"].to("cpu", copy=True) if cleanup_between_shots
                           else wav["waveform"])
            del imgs, wav
            if cleanup_between_shots:
                _deep_cleanup()

        if _recovered:
            notes.append(
                "recovered a face for "
                + "; ".join(f"{who} on shot {n}, from shot {src}"
                            for n, who, src in _recovered)
                + ". They were back after a shot away with no picture of them anywhere "
                  "-- the keyframe is the previous shot's last frame and they were not "
                  "in it -- so a frame from the middle of the last shot that was THEIRS "
                  "ALONE was sent as a reference. The middle, because somebody walking "
                  "out is gone by the last frame and somebody walking in is missing from "
                  "the first. Both ends have to be solo: a frame is a picture of "
                  "everyone in it, so one taken from a shared shot would carry the other "
                  "person into a shot that does not call for them. A character never on "
                  "screen alone gets nothing, which beats importing somebody. Skipped "
                  "for anyone with a <Picture N> tag of their own. The frame is "
                  "CLAIMED on their sheet entry for that shot -- a picture the "
                  "prompt never refers to is read as another subject, so an "
                  "unclaimed one would arrive as a second person with the same "
                  "face and the same clothes. `script` is written before the render, so it does not show that tag")
        video = torch.cat(vid_out, dim=0)
        if video.dtype != torch.float32:
            video = video.float()          # back to what every downstream node expects
        # PIXEL upscale, once, on the finished chain. After the latent pass and after
        # the join, so a model-based upscaler sees whole frames and the seam is not
        # upscaled twice.
        if upscale and upscale != "off":
            video, up_note = _upscale_frames(video, upscale, upscale_model,
                                             upscale_target_short_edge, upscale_batch)
            if up_note:
                notes.append(up_note)
        audio = torch.cat(aud_out, dim=-1)
        total = video.shape[0]
        # The finished chain is the largest thing this node holds, and it competes with
        # the MODELS for system RAM: ComfyUI offloads weights to RAM rather than
        # discarding them, so a shot boundary is a PCIe copy while that RAM is there
        # and a disk read once the frames have crowded the weights out.
        if cleanup_between_shots and total:
            _held = total * int(w) * int(h) * 3 * 2 / GB          # fp16, as stored
            if _held >= 2.0:
                notes.append(
                    f"the finished chain is {_held:.1f}GB in system RAM ({total} frames at "
                    f"{w}x{h}, held as float16 -- float32 would be {_held * 2:.1f}GB). It "
                    f"shares that RAM with the models, which ComfyUI offloads to it "
                    f"rather than discarding: while they fit, a shot boundary is a PCIe "
                    f"copy; once the frames crowd them out it becomes a disk read, once "
                    f"per model per shot. If the machine is thrashing, the levers are "
                    f"fewer frames per run (lower shot_seconds, or split a long script "
                    f"and join the parts outside the node), a lower megapixels, or a "
                    f"smaller diffusion quant -- every GB of weights is a GB not "
                    f"available to hold the render")
        if fresh:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in fresh)} start fresh, because the shot "
                f"before each took something off -- continuing from a frame that may still "
                f"show the garment is how it comes back, and a picture outvotes the text. "
                f"That costs a cut there. Turn restart_after_removal off to keep the "
                f"continuity instead")
        wall = time.perf_counter() - t_start
        n = max(1, len(shots))
        other = max(0.0, wall - t_sample - t_decode)
        notes.append(
            f"rendered {total} frames (~{total / H3_FPS:.1f}s) in {wall:.0f}s -- "
            f"sampling {t_sample:.0f}s ({100 * t_sample / wall:.0f}%), "
            f"decode {t_decode:.0f}s ({100 * t_decode / wall:.0f}%), "
            f"other {other:.0f}s ({100 * other / wall:.0f}%); "
            f"per shot {t_sample / n:.1f}s + {t_decode / n:.1f}s")
        if av_fix:
            per_shot = abs(av_fix) / sr * 1000 / max(1, len(shots))
            notes.append(
                f"audio realigned to the picture by ~{abs(av_fix) / sr * 1000:.0f} ms "
                f"across {len(shots)} shot(s), {per_shot:.1f} ms each. H3's audio latent "
                f"runs at {AUDIO_LATENT_FPS}/s against {H3_FPS} fps video, so a shot's "
                f"sound lands exactly only when its frame count divides by 3 -- otherwise "
                f"it is up to 8.3 ms out, with the same sign every time when the shots "
                f"are the same length, which is how a chain drifts out of sync"
                + (". That is far more than the 8.3 ms the grid accounts for, so the "
                   "audio VAE is not returning the length its latent implies -- check "
                   "that the audio VAE is H3's own converted one"
                   if per_shot > 50 else ""))
        _detail = detail_report(shot_detail)
        if _detail:
            notes.append(_detail)
        if t_decode > t_sample:
            notes.append("decode is costing more than sampling here -- latent_upscale "
                         "trades cheaper sampling for a 4x more expensive decode, so it "
                         "is the wrong way round at this step count. megapixels is the "
                         "lever that lowers both")
        return (video, {"waveform": audio, "sample_rate": sr}, " | ".join(notes), script,
                lens[0], total, len(shots), round(total / H3_FPS, 2))


NODE_CLASS_MAPPINGS = {"H3LongVideos": H3LongVideos}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LongVideos": "H3 Long Videos"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
