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
    named = [n for n, _ in rows
             if n and re.search(r"\b" + re.escape(n) + r"\b", beat or "", re.I)]
    if _PRONOUN.search(beat or ""):
        named += [n for n in (previous or []) if n not in named]
    if not named:
        named = list(previous) if previous else [n for n, _ in rows if n]
    keep = [ln for n, ln in rows if n is None or n in named]
    return "\n".join(keep), named


_PRONOUN = re.compile(r"\b(?:she|he|her|hers|his|him|they|them|their|theirs)\b", re.I)


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


def build_scene(anchor, first_para, character_memory, sheet):
    """The text every shot carries, in reading order: the anchor frames the film,
    the opening paragraph sets the scene, and the character sheet says who is in it
    and what they are wearing.

    One string on purpose -- a removal scrubs all of it. The previous node kept the
    anchor immutable, and clothing written there could never be taken off: the
    anchor put it back on every shot, under a beat that had just removed it."""
    parts = [(anchor or "").strip(), (first_para or "").strip(),
             (character_memory or "").strip(), (sheet or "").strip()]
    return "\n".join(p for p in parts if p)


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


BEAT_BASE_SEC = 2.0            # setup and settle, whatever the beat says
SECONDS_PER_ACTION = 2.5       # screen time one staged action clause needs
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


def plan_lengths(beats, ceiling_frames, from_beat):
    """Frames for each shot. Returns (lengths, note).

    'fixed' gives every shot the ceiling. 'from the beat' sizes each shot from what
    its own line stages, capped by that same ceiling and floored at one action's
    worth -- so a beat with one action stops getting a shot with room for two, which
    is what makes an action carry on past its end.

    The estimate leans SHORT deliberately. A shot that ends before its action does
    hands a mid-motion frame to the next shot, and the chain is built to continue
    from exactly that. A shot that outlasts its action has to invent the remainder."""
    if not from_beat:
        return [ceiling_frames] * len(beats), ""
    lens = []
    for b in beats:
        need = beat_seconds(b)
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

    comfy/text_encoders/minimax.py writes the "<Picture N>: " label ITSELF, numbering
    by the order it receives the images -- so a shot that uses only <Picture 2>
    receives that image labelled <Picture 1>, and text still saying <Picture 2>
    points at nothing. The tags are renumbered per shot to match what the shot
    actually carries: slot 2 alone becomes <Picture 1>; slots 2 and 4 become
    <Picture 1> and <Picture 2>.

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
    out = re.sub(r",\s*,+", ",", out)             # ",," where the tag was the only item
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


def _decode_video(vae, out_latent, tiled, free_first=None, tile_t=None, tile_xy=None):
    """Decode the video latent. If `free_first` is the diffusion model, unload it
    first: sampling is finished, and the ~5GB video VAE needs the room. Leaving the
    DiT (plus resident bypass-LoRA adapters) on the card while the VAE loads is a
    second ratchet -- ComfyUI would otherwise evict reactively, after spilling."""
    if free_first is not None:
        try:
            mm.free_memory(1e30, mm.get_torch_device(), keep_loaded=[])
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
    Runs a Python GC pass (frees dereferenced tensors / CPU buffers), then hands
    ComfyUI its aggressive cache purge, then empties the CUDA allocator's cached
    blocks and IPC handles. Cheap relative to sampling; called once per beat."""
    gc.collect()
    try:
        mm.soft_empty_cache(True)      # aggressive (unload_all_models path)
    except TypeError:
        mm.soft_empty_cache()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _evict_all_but(keep_model):
    """Unload every model EXCEPT the diffusion model from the GPU.

    This is the fix for VRAM ratcheting across a long chain. soft_empty_cache()
    only drops the CUDA allocator's cached blocks -- it does NOT unload models, so
    ComfyUI keeps the Qwen3-VL text encoder (~14.6GB) and both VAEs resident in
    current_loaded_models alongside the DiT. Each shot re-encodes the prompt
    (text encoder), encodes the handoff keyframe (video VAE), then samples (DiT),
    so all three compete for the card; ComfyUI only evicts reactively, i.e. AFTER
    it has already spilled. With a bypass LoRA also holding 208 bf16 adapters
    resident there is no room left, and every shot leaves the card fuller.

    Freeing them explicitly, right after conditioning is built and before
    sampling, keeps only what the sampler actually needs on the GPU."""
    try:
        keep = []
        for lm in list(getattr(mm, "current_loaded_models", [])):
            try:
                if lm.model is keep_model or getattr(lm, "model", None) is getattr(keep_model, "model", None):
                    keep.append(lm)
            except Exception:
                pass
        mm.free_memory(1e30, mm.get_torch_device(), keep_loaded=keep)
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
    invitation to REPRODUCE it -- its framing along with its subject. A portrait
    reference therefore pulls the shot towards portrait framing, which is at odds
    with a full-body pose no matter what the text says."""
    if not n_refs or aug is None or float(aug) < KEYFRAME_SAFE_AUG:
        return ""
    note = (f"{n_refs} reference image(s) at ref_noise_aug {float(aug):.3f}, which is "
            f"near-clean -- that asks the model to reproduce them, FRAMING included. A "
            f"head-and-shoulders reference pulls every shot towards head-and-shoulders "
            f"framing, whatever the text says about the body")
    if not has_first_frame:
        note += (". Shot 1 has no keyframe, so on that shot the aug only touches the "
                 "references -- there is no anchor there for a lower value to damage")
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
               r"toss(?:es|ed)?|throw[s]?|threw")
# ...and verbs that are a removal on their own, needing no particle.
_UNDO_VERB = (r"remove[sd]?|removing|undress(?:es|ed)?|unzip(?:s|ped)?|"
              r"unbutton(?:s|ed)?|unhook(?:s|ed)?|unclasp(?:s|ed)?|unfasten(?:s|ed)?")

_REMOVAL_PROSE = re.compile(
    r"\b(?:" + _UNDO_VERB + r")\b"
    r"|\b(?:" + _STRIP_VERB + r")\s+(?:off|away|out\s+of|down)\b"
    r"|\b(?:" + _STRIP_VERB + r")\b(?=[^.;!?]{0,40}?\b(?:off|away)\b)",
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
RESTRAINT_HOLD = (" Every restraint stays whole and closed, fastened exactly as it was put "
                  "on, and still fastened at the last frame.")

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
              "between them stays straight and taut.")

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
                   "locked to.")

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
_HARDWARE_ANCHOR = (
    (r"collar(?:s|ed)?",                 "a collar closes around the neck"),
    (r"leash(?:es)?|lead\b",             "a leash clips to the collar at the neck and hangs down from it"),
    (r"gag(?:s|ged)?|ball\s*gag",        "a gag sits in the mouth"),
    (r"blindfold(?:s|ed)?",              "a blindfold covers the eyes"),
    (r"handcuff(?:s|ed)?",               "handcuffs close around the wrists"),
    (r"shackle[sd]?|leg\s+irons",        "shackles close around the ankles"),
    (r"harness(?:es)?",                  "a harness sits on the torso"),
    (r"spreader\s+bar",                  "a spreader bar holds the ankles apart"),
    # Its own entry, ahead of the plain belt: a chastity belt is not a waistband, and
    # the generic phrase said nothing about where it fastens, so the model put the
    # lock wherever locks usually go on a strap -- behind. The lookbehind on the plain
    # belt below keeps the two from both matching "chastity belt" and stating it twice.
    (r"chastity\s+belts?",               "a chastity belt sits on the waist and hips with "
                                         "its front shield running between the legs, and it "
                                         "locks at the front"),
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
    return out


def anchor_clause(phrases):
    """One sentence saying where the named hardware sits."""
    if not phrases:
        return ""
    return " Each piece of hardware sits where it belongs: " + "; ".join(phrases) + "."


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


def _is_entry_head(word, scene):
    """Is `word` the head of a wardrobe entry in the scene, rather than a modifier
    inside one or a fragment of a hyphenated compound?"""
    for m in re.finditer(r"\b" + re.escape(word) + r"\b", scene, re.I):
        # "tight" inside "skin-tight" is half a word, not a garment.
        if m.start() and scene[m.start() - 1] == "-":
            continue
        if m.end() < len(scene) and scene[m.end()] == "-":
            continue
        if _ENTRY_END.match(scene[m.end():]):
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
                # A <Picture N> tag is not clothing and must never leave with a
                # garment that happened to share its fragment -- losing it costs
                # that shot its identity reference.
                tags = [n for s in (gone or [frag]) for n in _PICTURE_TAG.findall(s)]
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
    passing minimax_ref_items makes it ignore `images` outright. So the handoff
    frame has to go through the SAME channel as the references or the text encoder
    never sees where the shot left off.
    """
    latent, fc = _empty_av_latent(width, height, length, H3_FPS)
    refs = [r for r in (refs or []) if r is not None]

    hand_img = None
    if handoff is not None:
        hand_img = _resize(handoff[:1], width, height, "disabled")

    # ONE aug covers every visual condition row -- references AND the keyframe.
    # comfy/ldm/minimax/model.py: _cond_video_rows() noises the keyframe latent by
    # (1 - aug), and seg_t["cond"] labels it max(t_v, aug) instead of 0.999. So
    # softening references below KEYFRAME_SAFE_AUG does not just soften them: it
    # noises and mis-timesteps the ANCHOR of every shot after the first, which shows
    # up as those shots degrading during sampling.
    #
    # When that would happen, the handoff stops being a keyframe and rides as an
    # extra reference instead. Weaker continuity, but nothing is pretending to be an
    # anchor while carrying noise.
    keyframe_ok = ref_noise_aug is None or float(ref_noise_aug) >= KEYFRAME_SAFE_AUG
    carry_as_ref = bool(hand_img is not None and refs and not keyframe_ok)

    enc_refs = refs + ([hand_img] if carry_as_ref else [])
    items, blocks = ([], [])
    if enc_refs:
        items, blocks = _build_ref_images(vae, enc_refs, width, height, ref_image_size)
    if hand_img is not None and not carry_as_ref:
        # Appended AFTER the references so <Picture N> numbering is untouched.
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
DEFAULTS_FILE = os.path.join(_HERE, "defaults.json")
DEFAULTS_EXAMPLE = os.path.join(_HERE, "defaults.example.json")
# Sockets, not widgets: nothing to carry a default.
_NOT_A_WIDGET = frozenset(("MODEL", "CLIP", "VAE", "IMAGE", "CONDITIONING", "SIGMAS",
                           "LATENT", "AUDIO", "MASK"))


def saved_defaults():
    """Widget defaults from defaults.json, or {} when there is none.

    Every time this node gains a widget its INPUT_TYPES changes, and a node added
    afresh comes up with the BUILT-IN defaults -- so a setting has to be put back by
    hand after every edit. A defaults.json beside this file outlives those edits: it
    is read when the node loads and any key in it replaces the built-in default for
    the widget of that name.

    Unknown keys are ignored, so the file survives a widget being renamed or
    dropped, and a missing or malformed file falls back to the built-ins rather
    than stopping the node from loading."""
    try:
        with open(DEFAULTS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def apply_saved_defaults(schema, saved=None):
    """Replace built-in defaults with saved ones, in place. Returns the schema."""
    saved = saved_defaults() if saved is None else saved
    if not saved:
        return schema
    for group in schema.values():
        if not isinstance(group, dict):
            continue
        for name, spec in group.items():
            if name not in saved or len(spec) < 2 or not isinstance(spec[1], dict):
                continue
            value = saved[name]
            # A combo can only default to one of its own choices -- an upscale model
            # that is no longer installed must not become the default.
            if isinstance(spec[0], (list, tuple)) and value not in spec[0]:
                continue
            spec[1]["default"] = value
    return schema


# What gets written when save_defaults is on. Sockets are excluded -- there is no
# value to keep -- and save_defaults itself is excluded, or a saved True would arm
# every fresh node to re-save on its next run.
_SAVEABLE = ("resolution", "megapixels", "shot_seconds", "steps", "cfg", "sampler_name",
             "scheduler", "seed", "shift_video", "shift_audio", "apply_model_sampling",
             "silence_nonspeech", "trim_seam", "ref_noise_aug", "tiled_decode",
             "cleanup_between_shots", "latent_upscale", "latent_upscale_scale", "upscale",
             "upscale_model", "upscale_target_short_edge", "upscale_batch", "shot_length",
             "auto_remove", "restart_after_removal", "hold_restraints", "plan_only",
             "anchor", "character_memory", "character_guard")


def write_defaults(values):
    """Persist the current widget values as this node's defaults. Returns a note."""
    keep = {k: v for k, v in values.items() if k in _SAVEABLE and v is not None}
    try:
        with open(DEFAULTS_FILE, "w", encoding="utf-8") as fh:
            json.dump(keep, fh, indent=2, sort_keys=True)
    except Exception as exc:
        return (f"could not write {DEFAULTS_FILE}: {exc}. The values in this node are "
                f"unaffected; only saving them as defaults failed")
    return (f"saved {len(keep)} widget value(s) to defaults.json. Every node added from "
            f"now on starts with these, through a restart and through an update -- the "
            f"file is gitignored. Turn save_defaults back OFF, or every run rewrites it")


def write_defaults_example(schema):
    """Write defaults.example.json once, so defaults.json is discoverable."""
    if os.path.exists(DEFAULTS_EXAMPLE) or os.path.exists(DEFAULTS_FILE):
        return
    out = {}
    for group in schema.values():
        if not isinstance(group, dict):
            continue
        for name, spec in group.items():
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if opts.get("forceInput"):
                continue
            if "default" in opts:
                out[name] = opts["default"]
            elif isinstance(spec[0], (list, tuple)) and spec[0]:
                out[name] = spec[0][0]          # a combo defaults to its first choice
            elif spec[0] in _NOT_A_WIDGET:
                continue
    try:
        with open(DEFAULTS_EXAMPLE, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
    except Exception:
        pass                                    # read-only install: not worth failing over


class H3LongVideos:
    """One prompt -> a chain of MiniMax-H3 shots, joined into one video."""

    @classmethod
    def INPUT_TYPES(cls):
        schema = cls._schema()
        write_defaults_example(schema)
        # Saved defaults last, so a preference outlives every edit to this file.
        return apply_saved_defaults(schema)

    @classmethod
    def _schema(cls):
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
                "upscale_model": (_upscale_model_list(), {
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
                "save_defaults": ("BOOLEAN", {"default": False,
                    "tooltip": "Write this node's CURRENT widget values to defaults.json, "
                               "so every node added from now on starts with them.\n\n"
                               "Set the node up the way you want it, turn this on, run "
                               "once, then turn it off. The values survive a restart and "
                               "an update to this node: the file is gitignored, and a "
                               "widget added later cannot shift them, because they are "
                               "matched by NAME rather than by position.\n\n"
                               "It does not touch nodes already in a workflow -- those "
                               "keep whatever was saved with them. It covers the widgets "
                               "only, not the images or the model, and not ComfyUI's own "
                               "'control after generate', which belongs to the editor."}),
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
            character_guard=True, save_defaults=False):

        notes = []
        if save_defaults:
            notes.append(write_defaults(locals()))
        _saved = saved_defaults()
        if _saved:
            notes.append(f"defaults.json supplies {len(_saved)} widget default(s), so a "
                         f"node added fresh comes up with those instead of the built-in "
                         f"ones. It does not touch a node already in a workflow -- that "
                         f"one keeps whatever was saved with it")
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
        sheet = "\n".join(p for p in ((character_memory or "").strip(), sheet) if p)
        static = build_scene(anchor, scene, "", "")
        scene = build_scene(anchor, scene, "", sheet)      # the whole of it, for inference
        if sheet:
            notes.append(f"folded {sheet.count(chr(10)) + 1} character-sheet line(s) into "
                         f"the scene instead of spending a shot on them -- a sheet "
                         f"describes people, it does not stage anything, and it has to "
                         f"be in EVERY shot for a removal to have something to scrub")
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
        restrained = posed = rigid_latched = False
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
        active = []                 # the people the previous beat involved
        guard_words = beat_words = total_words = 0
        for b in beats:
            body, toks, adds = extract_directives(b)
            # Read the removal out of the beat itself. Explicit 'remove:' lines still
            # win and are added to whatever is inferred.
            if auto_remove:
                inferred = [t for t in infer_removals(body, scene)
                            if t not in toks and t not in gone]
                if inferred:
                    toks = list(toks) + inferred
                    notes.append(f"shot {len(shots) + 1}: read '{', '.join(inferred)}' as "
                                 f"coming off, from the beat's own wording")
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
            # Only the people this beat involves. Describing everyone in every shot
            # is what puts the other character in a scene they are not in.
            if character_guard:
                shot_sheet, active = sheet_for_beat(sheet, body, active)
                if len(sheet_lines(sheet)) > len(sheet_lines(shot_sheet)):
                    notes.append(f"shot {len(shots) + 1} describes only "
                                 f"{', '.join(active) or 'the scene'} -- the rest of the "
                                 f"sheet is held back, because a person the text "
                                 f"describes is a person the model draws")
            else:
                shot_sheet = sheet
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
            shot_scene = scrub_removed(
                "\n".join(p for p in (static, shot_sheet) if p.strip()),
                visible + covered)
            # An added layer is subject to removal too: once the shirt comes off, the
            # phrase that introduced it has to go with it, or the scene keeps
            # describing a garment that is no longer there.
            live = [a for a in shown if not names_any(a, gone)]
            if live:
                tail = ". ".join(a.rstrip(".") for a in live) + "."
                tail = tail[0].upper() + tail[1:]
                shot_scene = f"{shot_scene} {tail}".strip() if shot_scene else tail
            # The removal has to FINISH inside this shot, because its last frame is
            # the next shot's keyframe. Stated only here; naming the garment again
            # later would put it back.
            tail = off_by_last_frame(toks)
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
            # The chain clause SUBSUMES the restraint hold -- it says "whole and closed"
            # itself. Emitting both said it twice, which is twice the stasis for one
            # guarantee.
            hold = chain if chain else (RESTRAINT_HOLD if restrained else "")
            shot_text = (line + tail + anchors + hold + fall + turn).strip()
            guard_words += len(shot_text.split()) - len(f"{shot_scene} {body}".split())
            beat_words += len(body.split())
            total_words += len(shot_text.split())
            shots.append(shot_text)
            speech.append(has_speech(body))

        # What share of a shot is the node talking rather than the script. Continuity
        # clauses all say some version of "this stays as it is", and enough of them
        # drown the one sentence describing what HAPPENS -- which renders as a shot
        # where nothing does. The previous node reached 96%; this is here so the creep
        # is visible before it gets there again.
        if total_words:
            notes.append(
                f"prompt balance: the beat is {100 * beat_words / total_words:.0f}% of "
                f"what each shot is told, continuity clauses {100 * guard_words / total_words:.0f}%, "
                f"scene and sheet the rest"
                + (" -- the guards are outweighing the action, which reads as a shot "
                   "where nothing happens. Fewer restraints named, or a beat with more "
                   "in it, shifts the balance back"
                   if guard_words > beat_words * 3 else ""))
        refs_all = [r for r in (ref_image_1, ref_image_2, ref_image_3, ref_image_4)
                    if r is not None]
        tagged = any(picture_tags(s) for s in shots)

        lens, len_note = plan_lengths(beats, ceiling, shot_length == "from the beat")
        if len(set(lens)) == 1:
            notes.append(f"{len(shots)} shot(s) x {lens[0]}f (~{lens[0] / H3_FPS:.1f}s) "
                         f"at {w}x{h} = ~{sum(lens) / H3_FPS:.1f}s total")
        else:
            notes.append(f"{len(shots)} shot(s) at {w}x{h}, sized per beat: "
                         + ", ".join(f"{n}f/{n / H3_FPS:.1f}s" for n in lens)
                         + f" = ~{sum(lens) / H3_FPS:.1f}s total")
        if len_note:
            notes.append(len_note)
        if refs_all:
            notes.append(f"{len(refs_all)} reference image(s), "
                         + ("placed by <Picture N> tags" if tagged else "on every shot")
                         + (f", ref_noise_aug {float(ref_noise_aug):g}"
                            f"{' -- near-clean, so the reference tends to be reproduced in the '
                              'opening frames, pose and background included'
                              if float(ref_noise_aug) >= 0.99 else ''}"))
        n_silent = sum(1 for s in speech if not s)
        if silence_nonspeech and n_silent:
            notes.append(f"{n_silent} shot(s) have no quoted line and are silenced")
            # H3 is JOINT: the face follows the audio branch. Conditioning that branch
            # on real silence says this person is making no sound -- and a person
            # making no sound is rendered still. On a beat that stages effort or
            # reaction that is the wrong instruction, and it reads as a flat face.
            n_effort = sum(1 for b, sp in zip(beats, speech)
                           if not sp and exertion_in(b))
            if n_effort:
                notes.append(
                    f"{n_effort} of those stage effort or reaction with no line to speak. "
                    f"H3 is joint -- the face follows the audio branch -- so conditioning "
                    f"it on silence says the person makes no sound, and a person making no "
                    f"sound is rendered still. That is the flat, unreacting face. Turning "
                    f"silence_nonspeech OFF lets the branch generate breath and effort, "
                    f"which the face performs to; the cost is that an unconditioned branch "
                    f"can also invent speech, and the mouth will follow that too")
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
        shot_detail = []            # (detail, contrast) per shot, on its last frame
        _deep_cleanup()

        for i, shot_prompt in enumerate(shots):
            shot_refs = refs_all
            if tagged:
                shot_prompt, shot_refs, missing = resolve_tags(shot_prompt, refs_all)
                for n in missing:
                    msg = f"<Picture {n}> names a slot with no image connected"
                    if msg not in notes:
                        notes.append(msg)
            silent = bool(silence_nonspeech and not speech[i])

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

            cond, latent, fc, demoted = build_conditioning(
                clip, vae, audio_vae, shot_prompt, w, h, lens[i],
                handoff=shot_handoff, refs=shot_refs,
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
            imgs = _decode_video(vae, out, shot_tiled, free_first=model)
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
            vid_out.append(imgs.to("cpu", copy=True) if cleanup_between_shots else imgs)
            aud_out.append(wav["waveform"].to("cpu", copy=True) if cleanup_between_shots
                           else wav["waveform"])
            del imgs, wav
            if cleanup_between_shots:
                _deep_cleanup()

        video = torch.cat(vid_out, dim=0)
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
