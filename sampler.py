# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""Plan MiniMax-H3 shots, render their audio/video, and preserve continuity.

The node interface and prompt planning live here. Audio policy and synthesis,
conditioning assembly, and tensor/runtime operations have separate owner modules.
"""

import math
import os
import re
import sys
import time
import uuid

import torch

import nodes
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.nested_tensor
import comfy.model_management as mm

import importlib.util as _ilu


def _load_local(name, filename):
    spec = _ilu.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))
    module = _ilu.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


engine = _load_local("h3_engine", "engine.py")
_plan_module = _load_local("h3_shot_plan", "shot_plan.py")
_runtime_module = _load_local("h3_runtime", "runtime.py")
_audio_module = _load_local("h3_audio", "audio.py")
_cond_module = _load_local("h3_conditioning", "conditioning.py")
ShotPlan = _plan_module.ShotPlan
PreparedVideo = _plan_module.PreparedVideo

# Internal helper exports retained for existing callers.
ShotAudio = _audio_module.ShotAudio
FrameAccumulator = _runtime_module.FrameAccumulator
apply_levels = _runtime_module.apply_levels
H3_FPS = _runtime_module.H3_FPS
AUDIO_LATENT_FPS = _runtime_module.AUDIO_LATENT_FPS
KEYFRAME_SAFE_AUG = _cond_module.KEYFRAME_SAFE_AUG
MAX_FRAMES = _runtime_module.MAX_FRAMES
_SILENT_UNIT = _audio_module._SILENT_UNIT
align_frame_count = _runtime_module.align_frame_count
video_latent_t = _runtime_module.video_latent_t
temporal_shape = _runtime_module.temporal_shape
_decode_video = _runtime_module._decode_video
_decode_audio = _runtime_module._decode_audio
_seamless_loop = _audio_module._seamless_loop
mix_ambient = _audio_module.mix_ambient
_is_oom = _runtime_module._is_oom
_deep_cleanup = _runtime_module._deep_cleanup
_decode_headroom = _runtime_module._decode_headroom
_resident = _runtime_module._resident
_image_out_dtype = _runtime_module._image_out_dtype
_evict_all_but = _runtime_module._evict_all_but
_SILENCE_STATUS = _audio_module._SILENCE_STATUS
_silent_audio_latent = _audio_module._silent_audio_latent
_pin_audio_silence = _audio_module._pin_audio_silence
HandoffLevels = _cond_module.HandoffLevels
_keyframe_latent = _cond_module._keyframe_latent
_sample_on_sigmas = _runtime_module._sample_on_sigmas
RESIZE_CHUNK = _runtime_module.RESIZE_CHUNK
_stream_chunks = _runtime_module._stream_chunks
_resize_short_edge = _runtime_module._resize_short_edge
_upscale_frames = _runtime_module._upscale_frames
decode_fits_untiled = _runtime_module.decode_fits_untiled
upscale_batch_for = _runtime_module.upscale_batch_for
_find_node = _runtime_module._find_node
_invoke_node = _runtime_module._invoke_node
build_conditioning = _cond_module.build_conditioning

RES_MULTIPLE = 32
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


_LAST_MODEL_FP = {"fp": None}


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
    lead = []
    while paras and is_character_sheet(paras[0]) and all(
            sheet_pronoun(ln) or age_in(ln)
            for ln in paras[0].splitlines() if ln.strip()):
        lead.append(paras.pop(0))
    if not paras:
        return "", lead
    if len(paras) == 1:
        return "", lead + paras
    return paras[0], lead + paras[1:]


def paragraphs(text):
    """Non-empty paragraphs, separated by a BLANK line."""
    return [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]


_SHEET_LINE = re.compile(r"^\s*(?!(?i:remove|off|add|wear|wardrobe)\s*:)"
                         r"[A-Za-z][\w'’-]{0,24}(?:\s+[A-Z][\w'’-]{0,24}){0,2}\s*:\s*\S")


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
        m = re.match(r"\s*([A-Z][\w'’-]{0,24}(?:\s+[A-Z][\w'’-]{0,24}){0,2})"
                     r"\s*:\s*\S", ln)
        out.append((m.group(1) if m else None, ln.strip()))
    return out


_PLURAL_CUE = re.compile(
    r"\b(?:both|each\s+other|one\s+another|the\s+two\s+of\s+(?:them|us|you)|"
    r"the\s+pair\s+of\s+(?:them|us|you)|all\s+of\s+(?:them|us|you))\b", re.I)
_THEY = re.compile(r"\bthey\b", re.I)


def group_beat(beat, rows):
    """Does this beat talk about the people as a GROUP rather than an individual?

    `rows` is sheet_lines(sheet). A they/them that some entry DECLARES as its own
    pronoun is that person, not the group -- so it is only a group cue when nobody
    on the sheet uses it."""
    b = beat or ""
    if _PLURAL_CUE.search(b):
        return True
    if not _THEY.search(b):
        return False
    return not any(sheet_pronoun(ln) == "they" for n, ln in (rows or []) if n)


def entry_heads(line):
    """Every head noun in one sheet entry's wardrobe, whatever kind of thing it is.

    garments_in knows garments and restraint_words knows hardware, and a chastity
    belt is neither: it is in no garment list and "belt" is not a restraint word,
    so both readers return nothing for it. This is the list used to decide WHOSE
    thing a beat is handling, and for that the category does not matter -- only
    that the sheet gave this person that item.

    Age, pronoun and bare adjectives are not things: an entry has to end in a word
    that could be a noun, and the numeric and pronoun entries are dropped."""
    out = []
    for item in re.split(r"[,;.]", str(line or "").split(":", 1)[-1]):
        item = re.sub(r"<\s*picture\s+\d+\s*>", " ", item, flags=re.I)
        item = _LEADING_TAG.sub("", re.sub(r"\s+", " ", item)).strip()
        if not item:
            continue
        head = item.split()[-1].lower().strip("-")
        if (len(head) < 3 or head.isdigit() or head in _NOT_A_GARMENT
                or head in {"she", "he", "they", "her", "his", "them", "old"}):
            continue
        if head not in out:
            out.append(head)
    return out


_SPOKEN_SPAN = engine._SPOKEN_SPAN
_outside_speech = engine._outside_speech


_PRONOUN_AT = re.compile(
    r"\b(?:beside|alongside|next\s+to|opposite|toward|towards|at|to|over|onto|into|"
    r"against|with|near|by)\s+(her|him|them)\b"
    r"(?=\s*[.,;:!?]|\s+(?:and|but|then|while|as|so|who|before|after)\b|\s*$)", re.I)


_PRONOUN_DOES_TO = re.compile(
    r"\b(?:hugs?|hugged|hugging|embrace[sd]?|embracing|kiss(?:es|ed|ing)?|"
    r"joins?|joined|joining|follows?|followed|following|"
    r"watch(?:es|ed|ing)?|greets?|greeted|greeting|thanks?|thanked|thanking|"
    r"comforts?|comforted|comforting|grabs?|grabbed|grabbing|"
    r"catch(?:es|ing)?|caught|shov(?:es|ed|ing)|push(?:es|ed|ing)?|"
    r"drags?|dragged|dragging|escorts?|escorted|escorting|"
    r"helps?|helped|helping|leads?|leading|led|guid(?:es|ed|ing)|"
    r"hands?|handed|handing|pass(?:es|ed|ing)?|giv(?:es|ing)|gave|"
    r"shows?|showed|showing|tells?|telling|told|asks?|asked|asking)"
    r"\s+(her|him|them)\b"
    r"(?=\s*[.,;:!?]|\s+(?:and|but|then|while|as|so|who|before|after)\b|\s*$"
    r"|\s+(?:a|an|the|another|his|her|their|its|some|one|two)\b)", re.I)


def pronoun_points_away(beat):
    """Does this beat aim a pronoun at somebody OTHER than the person it names?"""
    b = str(beat or "")
    return bool(_PRONOUN_AT.search(b) or _PRONOUN_DOES_TO.search(b))


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
    _here = set(engine.names_in(beat, [n for n, _ in rows if n]))
    named = [n for n, _ in rows if n in _here]
    for n, ln in rows:
        if not n or n in named:
            continue
        if any(re.search(r"\b" + re.escape(g) + r"\b", beat or "", re.I)
               for g in entry_heads(ln)):
            named.append(n)
    if group_beat(beat, rows):
        everyone = [n for n, _ in rows if n]
        for n in (previous or []):
            if n in everyone and n not in named:
                named.append(n)
        named = ([n for n in everyone if n in named] if len(named) >= 2
                 else everyone)
        return "\n".join(ln for n, ln in rows if n in named), named
    used = {m.group(0).lower() for m in _PRONOUN.finditer(engine.staged_text(beat or ""))}
    if used:
        matched = False
        for group, words in _PRONOUN_SET.items():
            if not used & words:
                continue
            _present = {n for n in (previous or []) if n}
            _away = (len(named) == 1 and pronoun_points_away(beat))
            if _away and _present:
                _others = [n for n, ln in rows
                           if n and n not in named and n in _present
                           and sheet_pronoun(ln) == group]
                if len(_others) == 1:
                    named.append(_others[0])
                    matched = True
                    continue
            if any(sheet_pronoun(ln) == group for n, ln in rows if n and n in named):
                matched = True
                continue
            cands = [n for n, ln in rows
                     if n and n not in named and sheet_pronoun(ln) == group]
            if len(cands) == 1:
                named.append(cands[0])
                matched = True
            elif len(cands) > 1:
                narrowed = [n for n in cands if n in (previous or [])]
                if len(narrowed) == 1:
                    named.append(narrowed[0])
                    matched = True
        if not matched:
            named += [n for n in (previous or []) if n not in named]
    if not named:
        named = list(previous or [])
    if not named:
        _all = [n for n, _ in rows if n]
        named = _all if len(_all) == 1 else []
    keep = [ln for n, ln in rows if n is None or n in named]
    return "\n".join(keep), named


_ENTRANCE = re.compile(
    r"\b(?:walk|step|come|run|stride|hurry|move|wander|burst|barge|slip|climb)"
    r"(?:s|ed|ing)?\s+(?:in|into|through|up|over|back|out\s+of)\b"
    r"|\benter(?:s|ed|ing)?\b|\barriv(?:es?|ed|ing)\b"
    r"|\bjoin(?:s|ed|ing)?\b|\breturn(?:s|ed|ing)?\b|\bfollow(?:s|ed|ing)?\b"
    r"|\blets?\s+\w+\s+in\b", re.I)


def arrives_in(text):
    """Does this beat stage somebody arriving -- moving into the frame?

    A word that only says they are suddenly THERE does not count, however much it
    reads like an entrance -- see the note on _ENTRANCE."""
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
    used = {m.group(0).lower() for m in _PRONOUN.finditer(engine.staged_text(beat or ""))}
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


_EXIT_ROOMS = "|".join(p for p in engine.PLACES.split("|")
                       if p not in {"shower", "showers", "pool", "sauna", "van", "truck",
                                    "elevator", "cell", "steps", "stairs", "court"})
_EXIT_OUT_OF = (r"(?:(?:the|this|that|his|her|their|our)\s+)?(?:frame|shot|view|sight)"
                r"|(?:the|this|that|his|her|their|our)\s+(?:[\w-]+\s+)?(?:" + _EXIT_ROOMS
                + r"|house|home|building|apartment|flat|door|front\s+door|gate)")
_EXIT = re.compile(
    r"\b(?:leaves?|left|leaving)(?=\s*(?:[.,;:!?]|$)"
    r"|\s+(?:again|together|without|through|by|via|for|with|and|then|now|quietly|alone)\b"
    r"|\s+(?:the|this|that|his|her|their|our)\s+(?:[\w-]+\s+)?(?:" + _EXIT_ROOMS
    + r"|house|home|building|apartment|flat)\b)"
    r"|\bexit(?:s|ed|ing)?\b"
    r"|\b(?:walk(?:s|ed|ing)?|go(?:es|ing)?|went|head(?:s|ed|ing)?|step(?:s|ped|ping)?|"
    r"run(?:s|ning)?|ran|storm(?:s|ed|ing)?|hurr(?:y|ies|ied|ying)|slip(?:s|ped|ping)?|"
    r"wander(?:s|ed|ing)?|strid(?:e|es|ing)|strode|march(?:es|ed|ing)?|"
    r"rush(?:es|ed|ing)?|back(?:s|ed|ing)?|driv(?:e|es|ing)|drove|sneak(?:s|ed|ing)?|"
    r"snuck|dash(?:es|ed|ing)?|bolt(?:s|ed|ing)?|stomp(?:s|ed|ing)?|limp(?:s|ed|ing)?)"
    r"\s+(?:\w+ly\s+)?"
    r"(?:out\b(?!\s+of\s+(?!" + _EXIT_OUT_OF + r"))|off\b(?!\s+(?:the|a|an|his|her|their)\b)"
    r"|away\b(?!\s+from\b)|outside\b|home\b)"
    r"|\b(?:disappear|vanish)(?:s|es|ed|ing)?\b"
    r"|\bout\s+of\s+(?:(?:the|this|that|his|her|their)\s+)?(?:frame|shot|view|sight)\b",
    re.I)
_CLAUSE_OPEN = re.compile(r"(?:^|[,;:]|\b(?:and|then|but|while|as|when|before|after|so)\b)\s*$",
                          re.I)


def _movers(rx, beat, sheet, pool, alone_is_it=False):
    """The people a movement in this beat belongs to -- the subject of each match of rx.

    A name or a subject pronoun opening the clause, reached back across "and" to the
    predicate it continues: "Crystal hands Dan the keys and leaves" is Crystal, and
    "...and he leaves" is Dan. A pronoun resolves only to one person in `pool` who
    declares it. A movement pinned on nobody is the sole person in the pool's when
    `alone_is_it`, and otherwise nobody's."""
    text = engine.staged_text(beat or "")
    rows = [(n, ln) for n, ln in sheet_lines(sheet) if n]
    names = [n for n, _ in rows]
    pool = [n for n in (pool or []) if n]
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for m in rx.finditer(sentence):
            before = sentence[:m.start()]
            subj = []
            spots = []
            for n in names:
                spots += [(k.start(), k.end(), [n])
                          for k in re.finditer(r"\b" + re.escape(n) + r"\b", before)]
            for k in re.finditer(r"\b(she|he|they)\b", before, re.I):
                word = k.group(1).lower()
                if word == "they" and not any(sheet_pronoun(ln) == "they" for _, ln in rows):
                    who = list(pool)
                else:
                    who = [n for n, ln in rows if n in pool and sheet_pronoun(ln) == word]
                spots.append((k.start(), k.end(), who if len(who) == 1 or word == "they"
                              else []))
            spots.sort()
            for idx in range(len(spots) - 1, -1, -1):
                s, e, who = spots[idx]
                if (not _CLAUSE_OPEN.search(before[:s])
                        and not (idx == len(spots) - 1
                                 and re.fullmatch(r"\s+(?:\w+ly\s+)?", before[e:]))):
                    continue
                subj = list(who)
                j = idx
                while (j > 0 and re.fullmatch(r"\s*(?:,|and|,\s*and)\s*",
                                              before[spots[j - 1][1]:spots[j][0]], re.I)):
                    j -= 1
                    subj = list(spots[j][2]) + subj
                if j != idx and not _CLAUSE_OPEN.search(before[:spots[j][0]]):
                    subj = list(who)
                break
            else:
                subj = list(pool) if (alone_is_it and len(pool) == 1) else []
            out += [n for n in subj if n not in out]
    return out


def leaves_in(beat, sheet, present=()):
    """The people this beat takes OUT of the frame -- see _EXIT.

    A leaving the beat does not pin on anybody is the one person in the frame's, or
    nobody's: keeping somebody in the picture costs a reference, and taking out
    somebody who is still there costs a second copy of them."""
    return _movers(_EXIT, beat, sheet, present, alone_is_it=True)


_COMES_IN = re.compile(
    r"\b(?:walk(?:s|ed|ing)?|com(?:e|es|ing)|came|step(?:s|ped|ping)?|run(?:s|ning)?|ran|"
    r"hurr(?:y|ies|ied|ying)|burst(?:s|ing)?|barg(?:e|es|ed|ing)|slip(?:s|ped|ping)?|"
    r"strid(?:e|es|ing)|strode|stroll(?:s|ed|ing)?|wander(?:s|ed|ing)?|rush(?:es|ed|ing)?|"
    r"storm(?:s|ed|ing)?|march(?:es|ed|ing)?|sneak(?:s|ed|ing)?|snuck|limp(?:s|ed|ing)?|"
    r"stagger(?:s|ed|ing)?)\s+(?:\w+ly\s+)?(?:back\s+)?"
    r"(?:in\b(?!\s+(?:the|a|an|his|her|their)\b)|inside\b"
    r"|into\s+(?:the|this|that|a)\s+(?:[\w-]+\s+)?(?:" + _EXIT_ROOMS
    + r"|house|building|apartment|flat)\b)"
    r"|\benter(?:s|ed|ing)?\b(?!\s+(?:the|a|his|her)\s+(?:code|number|password|data|pin)\b)"
    r"|\barriv(?:e|es|ed|ing)\b"
    r"|\b(?:com(?:e|es|ing)|came)\s+back\b(?=\s*(?:[.,;:!?]|$)"
    r"|\s+(?:in|into|inside|home|with|and|carrying|holding)\b)"
    r"|\breturn(?:s|ed|ing)?\b(?!\s+(?:the|a|an|his|her|their|it|them|to\s+(?:the|his|her|their)\s+"
    r"(?:table|desk|couch|sofa|chair|bed|seat|work|book|screen|sink|stove|counter)))",
    re.I)


def comes_in(beat, sheet):
    """The people this beat stages ARRIVING in the frame -- see _COMES_IN."""
    return _movers(_COMES_IN, beat, sheet, [n for n, _ in sheet_lines(sheet) if n])


_SHE_NOUNS = {"woman", "girl", "lady", "female", "mother", "wife", "sister", "daughter",
              "aunt", "grandmother", "niece"}
_HE_NOUNS = {"man", "boy", "guy", "gentleman", "male", "father", "husband", "brother",
             "son", "uncle", "grandfather", "nephew"}
_PERSON_NOUN = re.compile(
    r"^(?:(?:a|an|the)\s+)?(?:[\w-]+\s+){0,2}(" + "|".join(sorted(_SHE_NOUNS | _HE_NOUNS))
    + r")\b(?!['\u2019])", re.I)
_PRONOUN_SET = {"she": {"she", "her", "hers"},
                "he": {"he", "him", "his"},
                "they": {"they", "them", "their", "theirs"}}


def sheet_pronoun(line):
    """Which pronoun this sheet entry declares for its person, or None.

    Writing the pronoun into the sheet -- "Maya: 27, she, grey coat" -- is what lets
    "her coat" in a beat be resolved to Maya rather than to whoever was in the last
    shot."""
    body = (line or "").split(":", 1)[-1]
    group_of = {w: g for g, words in _PRONOUN_SET.items() for w in words}
    for item in body.split(","):
        word = item.strip().strip(".;").lower()
        if word in _PRONOUN_SET:
            return word
    hits = [(m.start(), group_of[m.group(0).lower()])
            for m in re.finditer(r"\b(?:" + "|".join(group_of) + r")\b", body, re.I)]
    if hits:
        return min(hits)[1]
    for item in body.split(","):
        m = _PERSON_NOUN.match(item.strip())
        if m:
            return "she" if m.group(1).lower() in _SHE_NOUNS else "he"
    return None


ADULT_AGE = 18                 # below this the node describes no body at all

_AGE_WORD = r"(?:y\.?o\.?|yrs?|years?(?:\s+old)?|year-old)"
_AGE_AT = re.compile(
    # "aged 24", "age 24", "24yo", "24 years old", "24-year-old"
    r"\bage[d]?\s+(\d{1,3})\b"
    r"|\b(\d{1,3})\s*-?\s*" + _AGE_WORD + r"\b"
    r"|(?:^|,)\s*(\d{1,3})\s*(?=[,;.]|$)", re.I)
_DECADE = {"twenties": 20, "thirties": 30, "forties": 40, "fifties": 50,
           "sixties": 60, "seventies": 70, "eighties": 80}
_DECADE_AT = re.compile(
    r"\b(early|mid|middle|late)?\s*-?\s*"
    r"(?:(twenties|thirties|forties|fifties|sixties|seventies|eighties)"
    r"|(\d0)\s*s)\b", re.I)


def age_in(line):
    """The age this sheet entry declares, or 0 when it declares none.

    Read off the attribute list, never off a beat: a beat saying "twenty years later"
    is not somebody's age, and the sheet is where the author states what is true of a
    person for the whole film."""
    body = str(line or "").split(":", 1)[-1]
    # The tag carries digits of its own, and they are a slot number.
    body = re.sub(r"<\s*picture[\s_]*\d+\s*>", " ", body, flags=re.I)
    m = _AGE_AT.search(body)
    if m:
        got = int(next(g for g in m.groups() if g))
        return got if 1 <= got <= 120 else 0
    m = _DECADE_AT.search(body)
    if m:
        base = _DECADE.get((m.group(2) or "").lower())
        if base is None and m.group(3):
            base = int(m.group(3))
        if base in _DECADE.values():
            q = (m.group(1) or "").lower()
            return base + (2 if q == "early" else 8 if q == "late" else 5)
    return 0


_PRONOUN = re.compile(r"\b(?:she|he|her|hers|his|him|they|them|their|theirs)\b", re.I)


_DETERMINER = frozenset("a an the her his its their our my your this that".split())
_CAPITALISED = re.compile(r"\b([A-Z][a-z\u2019'-]{1,24})\b")
_NEVER_A_NAME = _DETERMINER | frozenset("""
i we you he she it they me him us them myself yourself himself herself itself
themselves mine yours hers ours theirs
and but or nor so yet then than as at in on of off to into onto from with without
if when while because though although after before until once since
there here what which who whom whose why how where whether
no not now never always again also just only even still both each either neither
one two three four five six seven eight nine ten first second next last another
yes ok okay oh ah well right left up down out over under across back forward
""".split())


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
            if word.lower() in _NEVER_A_NAME:
                continue
            before = (beat[:m.start()]).rstrip()
            prev = re.search(r"([\w’'-]+)\W*$", before)
            if prev and prev.group(1).lower() in _DETERMINER:
                continue
            if before and before[-1] not in ".!?:\"”":
                mid_sentence.add(word)
            if i not in seen.setdefault(word, []):
                seen[word].append(i)
    return {w: s for w, s in seen.items()
            if w in mid_sentence and w.lower() not in known}


_EXPOSE_CUE = re.compile(r"\b(?:to\s+expose|to\s+reveal|to\s+show|exposing|revealing|"
                         r"showing|uncovering|baring)\b", re.I)


_UNDER_BY_REGION = engine._UNDER_BY_REGION
_OUTER_BY_REGION = engine._OUTER_BY_REGION
implied_layers = engine.implied_layers
hidden_layers = engine.hidden_layers
is_undergarment = engine.is_undergarment


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
            if _modifier_of_a_named_entry(word, span, scene):
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


def revealed_by(covers, gone):
    """Under-layers brought into view because the thing over them has just come off."""
    return [u for u, o in (covers or {}).items() if o in (gone or [])]


_REGION_OF = engine._REGION_RX


def body_of(pronoun, age=0):
    """The body the sheet's declared pronoun and age mean. "" where nothing is declared.

    AN UNSPECIFIED BODY IS FILLED FROM THE PRIOR, which is the lesson this file
    already recorded for the chest -- "this said 'The arms and shoulders are bare' and
    stopped there, so the one region a bra occupies was unspecified, and an unspecified
    region is filled by the model's own prior". The hip-down clause had the same gap
    and it was never closed: "The legs are bare from the hip down" names the region and
    says nothing about whose body it is, so the anatomy at the hip came from the prior
    too. Reported as the wrong anatomy on a female character.

    Read from the pronoun the author DECLARED, which the README already requires for
    every entry, so this asserts nothing the sheet does not already say. `they` returns
    nothing: an undeclared body is not a licence to guess one.

    ...AND THE AGE THEY DECLARED, for the same reason one step further on. "A woman's
    body" is true of a woman of 22 and a woman of 62, so it settles nothing between
    them, and what fills the gap is the prior -- which is a woman in her twenties
    whatever the sheet says. Reported as a character written at one age rendering at
    another. The age is the author's own word, already in the sheet and already going
    to the model inside it; this only stops it being the one attribute nothing here
    reads.

    NO BODY IS DESCRIBED FOR A DECLARED AGE UNDER 18. Not a softer description -- none,
    and this returns "" so every clause built on it stays silent. An age the author
    states is the one fact here that is not a guess, and a generator has no business
    composing anatomy for a child. See also the refusal in _prepare: a script that
    declares a minor and stages nudity or sex does not render at all."""
    who = {"she": "woman", "he": "man"}.get(str(pronoun or "").strip().lower(), "")
    if not who:
        return ""
    age = int(age or 0)
    if age and age < ADULT_AGE:
        return ""
    return f"a {who}'s body" if not age else f"the body of a {who} of {age}"


_FIGURE = (
    (18, 24, "grown and firm, sitting high on the chest"),
    (25, 34, "fully grown and full, sitting a little lower than in her early twenties"),
    (35, 44, "full and softer, settled lower with the weight of middle age"),
    (45, 54, "mature and heavier, softened and lower again, with less tension in them"),
    (55, 120, "older and slacker, hanging low and soft, loose and lined"),
)


_SEXUAL_STAGING = re.compile(
    r"\b(?:sex|sexual|fucks?|fucking|fucked|intercourse|penetrat\w*|blow\s?job|"
    r"handjob|masturbat\w*|orgasms?|orgasmic|climax(?:es|ed|ing)?|cums?|cumming|"
    r"aroused|arousal|horny|erotic\w*|nipples?|genitals?|vagina\w*|penis\w*|"
    r"cocks?|dicks?|pussy|clit\w*|erections?|foreplay|straddl\w*|"
    r"topless|bottomless|naked|nude|nudity|undress\w*|strips?\s+(?:off|naked|bare)|"
    r"moans?|moaning|moaned)\b", re.I)


def minor_with_sexual_staging(sheet, script):
    """A refusal message when a sheet declares a minor and the script stages sex. "" otherwise.

    Both halves required. An age under 18 on its own renders -- children exist in
    films -- and gets no body described for them by anything here. Sexual staging on
    its own renders, which is what this node is for."""
    named = [(n, age_in(ln)) for n, ln in sheet_lines(sheet or "") if n]
    minors = sorted({n for n, a in named if 0 < a < ADULT_AGE})
    if not minors:
        return ""
    m = _SEXUAL_STAGING.search(str(script or ""))
    if not m:
        return ""
    return (f"REFUSED, and nothing was rendered. The character sheet declares "
            f"{_join_names(minors)} as under {ADULT_AGE}, and the script stages sexual "
            f"or nude content -- it contains {m.group(0)!r}. This node will not "
            f"generate that combination, whichever character the wording is about and "
            f"whatever was intended by it. Nothing here tried to work out who: a film "
            f"holding both is refused whole.\n\n"
            f"If an age is a typo, fix the sheet and run again -- an adult age renders "
            f"normally. If the character is an adult, state an adult age. A scene with "
            f"a child in it and no sexual or nude content renders as it always did, "
            f"and no body is described for them by this node.")


def _pron_age(sheet, name):
    """(pronoun, age) off one person's own sheet entry. ("" , 0) when it has neither."""
    line = dict(sheet_lines(sheet)).get(name, "")
    return sheet_pronoun(line), age_in(line)


def figure_of(pronoun, age=0):
    """Age-consistent adult chest description, or "". See _FIGURE and body_of.

    Returns nothing at all without BOTH a declared "she" and a declared adult age:
    with no age there is nothing to be consistent with, and the old silence is better
    than a guess."""
    if str(pronoun or "").strip().lower() != "she":
        return ""
    age = int(age or 0)
    if age < ADULT_AGE:
        return ""
    for lo, hi, said in _FIGURE:
        if lo <= age <= hi:
            return f"the breasts {said}"
    return ""


def groin_of(pronoun, age=0):
    """What the hip region is, once the last thing on it has come off. "" otherwise.

    The sister of figure_of, for the same reason and with the same gating. "The legs
    are bare from the hip down" names the LEGS and stops, so the one part of that
    region underwear occupies is left unspecified -- and this file's own note beside
    REGION_OF says what fills an unspecified region: "the prior for a hip is
    underwear". So the shot that takes the last layer off is answered by the model
    putting another one back, or by a smoothed-over blank where the anatomy should
    be. Reported as underwear that cannot be removed.

    It is reached ONLY through a region this sentence has already called bare, which
    means nothing on the sheet covers it and nothing under it was named. Saying what
    is there is not an escalation; it is the same sentence finishing.

    NO BODY IS DESCRIBED FOR A DECLARED AGE UNDER 18, and none for an entry that
    declares no pronoun -- the identical rule body_of and figure_of keep, returning ""
    so every clause built on this stays silent. See also the refusal in _prepare: a
    script that declares a minor and stages nudity does not render at all."""
    who = {"she": "woman", "he": "man"}.get(str(pronoun or "").strip().lower(), "")
    if not who:
        return ""
    age = int(age or 0)
    if not age or age < ADULT_AGE:
        return ""
    return "the hips and groin bare as well, the genitals uncovered and in plain view"


def bare_clause(gone, covers=None, worn="", body="", figure="", groin=""):
    """Say the uncovered region is BARE, when the sheet names nothing under it.

    A removal clause is emphatic -- off the body, dropped out of frame -- and then
    says nothing about what occupies the space it left. An unspecified region is
    where the model's own prior fills in, and for legs that prior is legwear: the
    shot invents leggings, tights or stockings that appear nowhere in the prompt,
    and the keyframe then carries the invention into every later shot.

    Positively phrased, and it names a BODY PART, never a garment. At cfg 1 there
    is no negative prompt, so "no leggings" would be read as leggings; "the legs
    are bare" fills the same region with something that is actually wanted.

    Silent when the sheet already answers the question -- reveal_clause covers the
    case where something IS underneath, and the two must never both speak -- and
    silent when another garment the character still wears covers the same region."""
    if not gone:
        return ""
    regions = []
    for item in gone:
        r = engine.region_of(item)
        if r and r not in regions:
            regions.append(r)
    return bare_hold(regions, covers, worn, gone, body=body, figure=figure,
                     groin=groin)


def bare_hold(regions, covers=None, worn="", gone=(), whose="", body="", figure="",
              groin=""):
    """Say those regions are bare -- from STATE, so it outlives its beat.

    The same suppression as the removal beat, because it is the same sentence:
    silent when the sheet names a layer underneath (reveal_clause has that one),
    and silent when a garment still worn covers the region.

    The reason it exists apart from bare_clause is the report: a bra coming back
    on somebody topless, on a character with no bra anywhere on the sheet. The
    clause only ever fired on the beat that uncovered the region, so every shot
    after it said nothing about that region -- and an unspecified region is
    filled by the model's own prior. Nothing was restoring the bra. The prior was
    inventing one, and the keyframe then carried the invention forward."""
    if not regions:
        return ""
    spoke = []                     # the regions this clause actually speaks about
    under = {str(u).lower() for u in (covers or {}) if not names_any(u, gone)}
    said, out = set(), []
    for _region in regions:
        for rx, region, sentence in _REGION_OF:
            if region != _region or region in said:
                continue
            if any(rx.search(t) for w in (worn or "").split(",")
                   for _sep, t in entry_parts(w) if not names_any(t, gone)):
                said.add(region)
                break
            if any(rx.search(u) or re.search(_UNDER_BY_REGION.get(
                       "lower" if region == "legs" else
                       "upper" if region == "torso" else "", "(?!)"), u, re.I)
                   for u in under):
                said.add(region)
                break
            said.add(region)
            out.append(sentence)
            spoke.append(region)
            break
    if not out:
        return ""
    out = out[:2]
    joined = out[0] + "".join(", and " + s[0].lower() + s[1:] for s in out[1:])
    if whose:
        joined = f"{whose}'s " + joined[4:] if joined.startswith("The ") else \
            f"{whose}: " + joined
    said_fig = f", {figure}" if (figure and "torso" in spoke[:2]) else ""
    said_low = f", {groin}" if (groin and "legs" in spoke[:2]) else ""
    return " " + joined + (f", on {body}" if body else "") + said_fig + said_low + \
        ", the skin itself the outermost surface there."


def defer_tag_for(text, items):
    """Take the <Picture N> off an item that is covered THIS SHOT, keeping its
    words. The tag comes back the moment the cover comes off.

    THIS IS A DEFERRAL, NOT A REMOVAL, and the distinction is the whole point.
    The item stays in the character memory in every shot, exactly as written. What
    waits is its reference, and only on the shots where the thing is under
    something else.

    It waits because a reference is an instruction to REPRODUCE AN IMAGE. At the
    near-clean ref_noise_aug this node runs at, the node's own report says so:
    "that asks the model to REPRODUCE them, framing and background included". A
    picture of a chastity belt, handed to the model for a shot in which the belt
    is under a skirt, is an instruction to draw the belt, and it outweighs any
    sentence about what is on top of what. Measured twice, from two different
    directions: every configuration that sent the picture while the garment was
    covered rendered it through the cover, including one where the cover was
    described as whole, opaque and unbroken.

    There is no third option available. Reference strength is ref_noise_aug and it
    is one number for every image, so the belt's picture cannot be weakened
    without weakening the face. The tag is what routes the image, so the tag is
    what waits -- leaving it in while withholding the image would name a picture
    the shot does not carry, which is its own bug."""
    out = str(text or "")
    for item in items or []:
        if not str(item).strip():
            continue
        w = re.escape(str(item).strip())
        out = re.sub(r"<\s*Picture\s*\d+\s*>\s*((?:a|an|the)\s+)?" + w,
                     lambda m: (m.group(1) or "") + str(item).strip(), out,
                     flags=re.I)
        out = re.sub(w + r"\s*<\s*Picture\s*\d+\s*>", str(item).strip(), out,
                     flags=re.I)
    return out


_COVER_PART = {
    "legs": "the hips and waist",
    "torso": "the chest and stomach",
    "feet": "the feet and ankles",
    "hands": "the hands",
    "head": "the head",
}


def cover_part(garment):
    """Where this garment covers, for the clause that says it is unbroken there.

    A garment that is the whole outfit covers two of them, and saying only one left
    the other half of the body unaccounted for in the same sentence that claims to
    name the only thing in view."""
    regions = engine.regions_of(garment)
    if "torso" in regions and "legs" in regions:
        return "the chest, stomach, hips and waist"
    for r in regions:
        if r in _COVER_PART:
            return _COVER_PART[r]
    return "the hips and waist"


def under_clause(pairs):
    """Say that an under-layer is UNDER, rather than deleting it from the sheet.

    Layering used to work by scrubbing: a garment read as covered came out of the
    shot text entirely, and its <Picture N> with it. The reasoning was sound as
    far as it went -- a described thing is a drawn thing, and an under-layer
    described flatly beside its cover gets drawn on top of it -- but the cost was
    the author's own words disappearing, which was reported three times, the last
    of them a chastity belt with a reference image attached to it.

    Deleting a thing is not the only way to stop it being drawn on top. Saying
    where it is works better and keeps the text: the model is told the belt is
    under the jeans, which is a spatial fact it can render, rather than being
    told nothing and left to guess. Positively phrased, because at cfg 1 there is
    no negative prompt -- this says where the thing IS, never where it is not.

    Panties, knickers, thongs, briefs, boxers, underwear, bras, corsets and
    chastity belts, devices and cages are all in _UNDER_BY_REGION, so they are
    always the under-layer whatever order the sheet lists them in."""
    pairs = [(p[0], p[1], p[2] if len(p) > 2 else "")
             for p in (pairs or []) if p[0] and p[1]]
    if not pairs:
        return ""

    def _plural(w):
        return w.endswith("s") and not w.endswith("ss")

    def _one(u, o, who=""):
        cover = "cover" if _plural(o) else "covers"
        whose = f"{who}'s " if who else "The "
        return (f"{whose}{o} {cover} {cover_part(o)} completely: whole, "
                f"opaque and unbroken, the outermost layer there and the only "
                f"one in view.")

    return " " + " ".join(_one(*p) for p in pairs[:2])


def reveal_clause(items, scene=""):
    """Say what is underneath is what shows now, on the shot that uncovers it.

    The removal clause is emphatic and specific -- off the body, dropped out of frame
    -- while the layer beneath is one item in an attribute list. Against a model whose
    prior for trousers coming off is bare skin, a list entry does not compete. It has
    to be told what fills the space the garment left.

    IN THE SHEET'S OWN WORDS, which is the same call off_by_last_frame makes and for
    the same reason. `items` are identity KEYS -- head nouns -- and this sentence is
    PROSE the model reads, so it said "The panties underneath are what shows there
    now" on a shot whose sheet says "red lace panties". One garment named twice, once
    with its description and once without, and the bare mention is the one the prior
    answers: reported as underwear always coming out black whatever it was written as.
    """
    if not items:
        return ""
    named = [scene_name_for(i, scene) or i for i in items] if scene else list(items)
    said = " and ".join(f"the {i}" for i in named[:2])
    plural = len(items) > 1 or plural_item(named[0])
    return (f" {said[0].upper()}{said[1:]} underneath {'are' if plural else 'is'} what "
            f"shows there now, on and unchanged.")


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
    seen_rows = []
    def _same_person(name, line):
        words = set(name.lower().split())
        for other, other_line in seen_rows:
            theirs = set(other.lower().split())
            if not (words <= theirs or theirs <= words):
                continue
            p1, p2 = sheet_pronoun(line), sheet_pronoun(other_line)
            a1, a2 = age_in(line), age_in(other_line)
            if (p1 and p2 and p1 != p2) or (a1 and a2 and a1 != a2):
                continue
            return True
        return False
    for src in sources:
        for name, line in sheet_lines(src):
            key = name.lower() if name else None
            if key and (key in seen_names or _same_person(name, line)):
                if name not in dupes:
                    dupes.append(name)
                continue
            if key:
                seen_rows.append((name, line))
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


_SETTING_PHRASE = re.compile(
    r"\b(?:in|inside|outside|at)\s+(?:a|an|the|this|that)\b", re.I)
_SPOT_PHRASE = re.compile(
    r"\b(?:on|by|near|beside|under|behind)\s+(?:a|an|the|this|that)\b", re.I)
_PERSON_WORD = re.compile(r"\b(?:her|his|their|him|them|herself|himself)\b", re.I)
_LEADING_PRONOUN = re.compile(r"^\s*(?:she|he|they|her|his|their)\b", re.I)


def _name_forms(name):
    """A name as a script writes it: whole, and a two-part name by either part."""
    forms = {name}
    parts = [p for p in name.split() if len(p) >= 3 and p[:1].isupper()]
    if len(parts) > 1:
        forms.update(parts)
    return forms


def _setting_of(sentence):
    """Where a sentence happens, without who is there. "" when it names no place.

    From the first "in a / at the / on the ..." to the end of the sentence, so
    "Maya sits at her desk in an office" gives "In an office" -- "at her desk" is
    hers, not the room's, and is passed over because it is not "at a" or "at the"."""
    m = _SETTING_PHRASE.search(sentence or "") or _SPOT_PHRASE.search(sentence or "")
    if not m:
        return ""
    phrase = sentence[m.start():].strip().rstrip(".!?;, ")
    if not phrase or _PERSON_WORD.search(phrase):
        return ""
    return phrase[0].upper() + phrase[1:] + "."


def static_for_shot(static, sheet, shot_sheet):
    """The anchor and opening paragraph for one shot: nobody named who is not in it.

    Names are the sheet's, matched case-sensitively as sheet_for_beat matches them,
    so "will" is never Will. A sentence that opens on a pronoun straight after one
    that was cut goes with it -- "Maya sits at her desk. She types." in a shot
    without Maya leaves no "She" behind to be drawn."""
    if not (static or "").strip() or not (sheet or "").strip():
        return static
    here = {n for n, _ in sheet_lines(shot_sheet or "") if n}
    absent = set()
    for name, _ in sheet_lines(sheet):
        if name and name not in here:
            absent |= _name_forms(name)
    for name in here:
        absent -= _name_forms(name)
    if not absent:
        return static
    named = re.compile(r"(?<![\w'\u2019-])(?:" + "|".join(
        re.escape(f) for f in sorted(absent, key=len, reverse=True)) + r")(?![\w-])")

    out = []
    for line in static.split("\n"):
        kept, cut_last = [], False
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            if not sentence:
                continue
            if named.search(sentence) or (cut_last and _LEADING_PRONOUN.match(sentence)):
                setting = _setting_of(sentence)
                if setting and not named.search(setting):
                    kept.append(setting)
                cut_last = True
                continue
            cut_last = False
            kept.append(sentence)
        if kept:
            out.append(" ".join(kept))
    return "\n".join(out)


_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|<d>.*?</d>", re.S)
_DIALOGUE_TAG = re.compile(r"<\s*d\s*>(.+?)<\s*/\s*d\s*>", re.I | re.S)
_CAPTION_TOKEN = re.compile(r"<\|(?:caption|lyrics)_(?:start|end)\|>", re.I)


BEAT_BASE_SEC = 0.8            # a little room to settle, not a whole beat of it
SECONDS_PER_ACTION = 2.2       # screen time one staged action clause needs
WORDS_PER_SEC = 2.5            # spoken delivery
_CLAUSE_SPLIT = re.compile(
    r"(?:[.!?;]+|,?\s+(?:and then|then|and|before|after|while|as|until)\s+"
    r"|,\s+(?=\w+(?:ing|es|s|ed)\b))")


def spoken_words(beat):
    """How many words this beat SPEAKS, counted once.

    _QUOTED matches H3's own <d>...</d> as well as plain quotes, and both places that
    counted speech added _DIALOGUE_TAG on top of it -- so a line written the way this
    node's own note tells an author to write it counted DOUBLE. A sixteen-word line
    was believed to take 12.8 seconds instead of 6.4: its shot was planned at 328
    frames instead of 175, the dialogue-headroom warning never fired because the line
    already "fitted", and the tail silence pin started late and held 1.6 seconds less
    than it should. Written as one reader so the two cannot drift apart again.

    The delimiters are not words: "<d>Hold this.</d>" is two."""
    b = str(beat or "")
    return sum(len(re.sub(r"</?\s*d\s*>|[\"“”]", " ", q).split())
               for q in _QUOTED.findall(b))


def travel_spaces(beat):
    """How many distinct spaces this beat shows on screen. 0 when it goes nowhere.

    THE WALK IS THE EXPENSIVE PART OF A TRANSIT, AND IT WAS INVISIBLE TO THE SIZING.
    beat_seconds counts ACTION CLAUSES, so the grammar of the sentence set the time
    and the ground covered did not: "McKenna walks down the hallway to the living
    room" is one verb phrase, so it was sized for one action -- 3.0s, the floor, the
    SHORTEST shot in its script -- and then told to show three rooms inside it, while
    "gets up and comes out of her bedroom" got 5.2s to stand up in one room. The beats
    doing the most spatial work were getting the least time to do it.

    A model handed 73 frames, a bedroom keyframe and instructions to reach a living
    room cannot TRAVEL, so it blends the two into one hybrid space -- which is a
    living room with a bed in it, the third route to a bug already fixed twice in the
    text. _CLAUSE_SPLIT's own comment names this failure exactly, "a walk down a
    hallway arriving as a cut to the far end", and fixed it only for comma lists.

    AN INTRA-ROOM WALK IS NOT THIS and must stay short: test_pace measured "Maya walks
    to the window" as under two seconds of real movement and the constants were tuned
    down for it. A window is not a place, so it crosses nothing here. Only a beat that
    actually ARRIVES somewhere counts, which is the same test travel_anchor applies
    before it will say a journey happened at all.

    The origin counts even when the beat does not name it: the shot opens in the room
    it was already in, that room is on screen at frame one, and it has to be left."""
    text = _DIALOGUE_TAG.sub(" ", _QUOTED.sub(" ", str(beat or "")))
    frm, via, to = travel_legs(text)
    if not to:
        return 2 if moved_to(text) else 0
    named = [p for p in (frm, via, to) if p]
    return len(named) + (0 if frm else 1)


def beat_seconds(beat):
    """Roughly how much screen time this beat's content asks for.

    Action and dialogue OVERLAP -- people talk while they move -- so it is the
    larger of the two, not the sum. Deliberately rough: the point is not to size
    the shot (the node does not), it is to notice when a shot is much longer than
    anything the beat gives it to do."""
    text = _DIALOGUE_TAG.sub(" ", _QUOTED.sub(" ", beat or ""))
    text = _REMOVE_LINE.sub("", _ADD_LINE.sub("", text))
    clauses = [p for p in _CLAUSE_SPLIT.split(text) if p and len(p.split()) >= 2]
    crossings = max(0, travel_spaces(text) - 1)
    action = (BEAT_BASE_SEC + SECONDS_PER_ACTION * (len(clauses) + crossings)) \
        if (clauses or crossings) else 0.0
    spoken = spoken_words(beat)
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
    lens, capped = [], []
    for b in beats:
        need = beat_seconds(b) * pace
        want = align_frame_count_nearest(int(round(need * H3_FPS))) if need else MIN_AUTO_FRAMES
        if want > ceiling_frames:
            capped.append((len(lens) + 1, want))
        lens.append(min(max(MIN_AUTO_FRAMES, want), ceiling_frames))
    note = ""
    if ceiling_frames < MIN_AUTO_FRAMES:
        note += (f"shot_seconds is {ceiling_frames / H3_FPS:.1f}s, below the "
                 f"{MIN_AUTO_FRAMES / H3_FPS:.1f}s one staged action needs. Every shot "
                 f"is held to it, so each beat performs faster than it reads -- if the "
                 f"motion looks clipped, that is this number. ")
    if capped:
        note += ("shot(s) "
                + ", ".join(f"{n} (wants {w / H3_FPS:.1f}s)" for n, w in capped[:6])
                + f" stage more than shot_seconds allows, so they are cut to "
                  f"{ceiling_frames / H3_FPS:.1f}s and perform the whole beat faster "
                  f"-- which is a walk down a hallway arriving as a cut to the far "
                  f"end. Raise shot_seconds (H3's own ceiling is "
                  f"{MAX_FRAMES / H3_FPS:.1f}s), or give the beat fewer actions and "
                  f"let the next one carry the rest. ")
    if len(set(lens)) > 1:
        note += (
                "shot lengths are sized from each beat ("
                + ", ".join(f"{n}f/{n / H3_FPS:.1f}s" for n in lens)
                + "). They differ, so one seed does not give them one noise field -- "
                  "noise is drawn to the latent's shape -- and surface detail resets at "
                  "each cut. Set shot_length to 'fixed' if that matters more than "
                  "pacing. The frames_per_shot output is ONE number and cannot "
                  "describe shots of different lengths: it reports the first one, so "
                  "do not split or index the image batch with it here -- the list "
                  "above is the split")
    return lens, note


def pace_clause(need, have):
    """Spread a short action across a long shot. "" when the shot is not long.

    thin_beats has always been able to SEE this -- one action sitting in a ten
    second shot -- and only ever reported it. The shot was still told what happens
    and nothing about when, so the action was performed at once and the spare
    seconds filled by carrying on: the same movement repeated on whatever was
    nearest. Reported as actions running way ahead of schedule.

    A timing anchor, the same shape the node already uses for a door ("open at the
    first frame and shut by the last") and for a removal ("away by the last
    frame"). It names WHEN, not how fast: "slowly" is a style instruction and this
    is not one -- it says the action occupies the shot it was given.

    Only where the gap is real. thin_beats' own thresholds: at least 2.5 spare
    seconds and a quarter again longer than the content, so a shot that only
    slightly outlasts a long beat stays quiet."""
    try:
        need, have = float(need), float(have)
    except (TypeError, ValueError):
        return ""
    if need <= 0 or (have - need) < 2.5 or have <= need * 1.25:
        return ""
    return (" What the beat stages runs at an even pace across the whole shot, "
            "beginning at the first frame and finishing on the last.")


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

_EFFORT_OBJ = (r"(?:her|his|their|the)\s+(?:backs?|hips?|thighs?|shoulders?|arms?|"
               r"wrists?|waist|hair|neck|sheets?|bedding|blankets?|pillows?|"
               r"mattress|headboard|bars?|restraints?)")

_EXERTION_NARROW_SRC = (
    # arching a back, not an eyebrow
    r"arch(?:es|ed|ing)?\s+(?:her|his|their)\s+backs?\b|"
    r"arch(?:es|ed|ing)?\s+(?:up|upwards?|off)\b|"
    # sustained movement, always against or with something
    r"(?:rock|grind|thrust|buck|push|move)(?:s|ed|ing)?\s+"
    r"(?:against|into|together|beneath|underneath|under|onto)\b|"
    # ...or the same verbs with no object at all, which is the intransitive sense
    r"(?:rocks?|rocked|rocking|grinds?|ground|grinding|thrusts?|thrusting|"
    r"bucks?|bucked|bucking|clench(?:es|ed|ing)?)\s*(?=[.,;!?]|$)|"
    # a hand closing on a body or the bedding, not on a railing
    r"(?:clutch(?:es|ed|ing)?|grip(?:s|ped|ping)?|claw(?:s|ed|ing)?)\s+"
    r"(?:at\s+)?" + _EFFORT_OBJ + r"|"
    r"(?:clutch(?:es|ed|ing)?|grip(?:s|ped|ping)?|claw(?:s|ed|ing)?)\s+at\b|"
    # involuntary, and rarely said of a prop
    r"shudder(?:s|ed|ing)?\b")
_EXERTION_NARROW = re.compile(r"\b(?:" + _EXERTION_NARROW_SRC + r")", re.I)


def exertion_in(beat):
    """Does this beat stage effort or reaction -- something a face performs?

    Two lists: verbs that are inherently about distress or exertion and mean it
    wherever they appear, and generic motion verbs that mean it only with the right
    complement. See _EXERTION_NARROW for why the second group may not fire alone."""
    b = beat or ""
    return bool(_EXERTION.search(b) or _EXERTION_NARROW.search(b))


_SOUND_CUE = re.compile(
    r"\b(?:sounds?|noises?|echo(?:e?s|ing)?|rattl(?:e|es|ing)|clank(?:s|ing)?|"
    r"clink(?:s|ing)?|creak(?:s|ing)?|scrap(?:e|es|ing)|thud(?:s|ding)?|bang(?:s|ing)?|"
    r"slam(?:s|ming)?|clatter(?:s|ing)?|jingl(?:e|es|ing)|squeak(?:s|ing)?|"
    r"footsteps?|breath(?:s|es|ing)?|pant(?:s|ing)?|gasp(?:s|ing)?|sigh(?:s|ing)?|"
    r"whimper(?:s|ing)?|moan(?:s|ing)?|groan(?:s|ing)?|sob(?:s|bing)?|"
    r"scream(?:s|ing)?|shout(?:s|ing)?|whisper(?:s|ing)?|laugh(?:s|ing|ter)?|"
    r"hum(?:s|ming)?|buzz(?:es|ing)?|hiss(?:es|ing)?|drip(?:s|ping)?|"
    r"rustl(?:e|es|ing)|click(?:s|ing)?|snap(?:s|ping)?|zip(?:s|ping)?|"
    r"rings?|ringing|wind|rain|thunder|traffic|music|hollow|muffled|reverb|"
    r"loud(?:ly)?|quietly|faintly|audible|noisy|deafening|"
    r"scuff(?:s|ing|ed)?|crunch(?:es|ing|ed)?|thump(?:s|ing|ed)?|"
    r"patter(?:s|ing)?|whirr?(?:s|ing)?|whine(?:s|d)?|whining|rumbl(?:e|es|ing)|"
    r"growl(?:s|ing)?|roar(?:s|ing)?|chime(?:s|d)?|ticking|"
    r"knock(?:s|ing)?|tap(?:s|ping)?|whoosh(?:es|ing)?|sizzl(?:e|es|ing))\b", re.I)


_BREATH_WORD = re.compile(r"\bbreath(?:s|es|ing)?\b|\bbreathe[sd]?\b", re.I)
_BREATH_PREP = re.compile(
    r"\b(?:takes?|took|taking|draws?|drew|drawing|catch(?:es)?|caught|"
    r"suck(?:s|ed)?|pull(?:s|ed)?|lets?\s+out|releases?)\s+"
    r"(?:in\s+)?(?:a|an|her|his|their|one|another|deep|long|slow|sharp|\s)*"
    r"breath\b|\bwith\s+a\s+breath\b|\ba\s+(?:deep\s+|long\s+|slow\s+|sharp\s+)?"
    r"breath\b", re.I)


def sound_described(text):
    """Does this beat ask for a sound the audio branch should make?

    A breath on its own does not: see _BREATH_ONLY."""
    t = text or ""
    hits = [h for h in (m.group(0).strip() for m in _SOUND_CUE.finditer(t)) if h]
    if not hits:
        return False
    if all(_BREATH_WORD.fullmatch(h) for h in hits) and _BREATH_PREP.search(t):
        return False
    return True


_VOCAL_FROM = (
    (r"\bwhimper(?:s|ing|ed)?\b",                   "whimpering"),
    (r"\bsob(?:s|bing|bed)?\b",                     "sobbing"),
    (r"\bmoan(?:s|ing|ed)?\b",                      "moaning"),
    (r"\bgroan(?:s|ing|ed)?\b",                     "groaning"),
    (r"\bscream(?:s|ing|ed)?\b",                    "screaming"),
    (r"\bwhin(?:e|es|ing|ed)\b",                    "whining"),
)

EFFORT_BREATH = "unsteady breathing, with gasps and moans of effort"

_SOUND_FROM = (
    *_VOCAL_FROM,
    (r"\b(?:walk(?:s|ed|ing)?|step(?:s|ped|ping)?|pace[sd]?|enters?|runs?|"
     r"approach(?:es|ed)?|creep(?:s|ing)?|crept|sneak(?:s|ing)?|shuffl(?:e|es|ing)|"
     r"stumbl(?:e|es|ing)|stagger(?:s|ing)?|feet)\b",  "footsteps"),
    (r"\bchains?\b",                                "chain links dragging"),
    (r"\A(?=[\s\S]*\b(?:handcuff|cuff|shackle|manacle)\w*\b)"
     r"(?=[\s\S]*\b(?:ratchet(?:s|ed|ing)?|clos(?:e|es|ing|ed)|snap(?:s|ped|ping)?|"
     r"lock(?:s|ed|ing)?|tighten(?:s|ed|ing)?|click(?:s|ed|ing)?)\b)",
                                                    "cuffs ratcheting closed"),
    (r"\b(?:handcuff(?:s|ed)?|cuffs?|cuffed|shackle[sd]?|manacle[sd]?)\b",
                                                    "cuffs knocking"),
    (r"\b(?:bolt|latch|catch)(?:es|ed|ing)?\b",     "a metal bolt sliding"),
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
    (r"\A(?=[\s\S]*\b(?:cuffs?|handcuffs?|shackles?|manacles?|chains?|ropes?|cords?|"
     r"straps?|restraints?|bindings?|ties?|tape|harness|collar)\b)"
     r"(?=[\s\S]*\b(?:thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|writh(?:e|es|ing|ed)|"
     r"strain(?:s|ing|ed)?|pull(?:s|ing|ed)?\s+against)\b)",
                                                    "restraints pulling taut"),
    (r"\b(?:thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|writh(?:e|es|ing|ed)|"
     r"strain(?:s|ing|ed)?|trembl(?:e|es|ing|ed)|shiver(?:s|ed|ing)?)\b"
     r"|\b(?:" + _EXERTION_NARROW_SRC + r")",
                                                    EFFORT_BREATH),
    (r"\b(?:zip(?:s|ped|ping)?|unzip(?:s|ped|ping)?|zipper)\b", "a zip running"),
    (r"\btap(?:e|es|ed|ing)\b",                     "tape pulling off"),
    (r"\A(?=[\s\S]*\b(?:bed|mattress|springs?|bunk|couch|sofa|headboard|"
     r"frame|table|desk|floorboards?)\b)"
     r"(?=[\s\S]*\b(?:rock(?:s|ed|ing)?|thrust(?:s|ing)?|grind(?:s|ing)?|"
     r"buck(?:s|ed|ing)?|writh(?:e|es|ing|ed)|arch(?:es|ed|ing)?|"
     r"thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|move(?:s|d)?\s+together|"
     r"shift(?:s|ed|ing)?\s+under)\b)",              "a bed frame working"),
    (r"\bvelcro\b",                                 "velcro tearing open"),
    (r"\b(?:rope|cord|twine|zip\s?tie)s?\b",        "rope creaking as it goes tight"),
    (r"\b(?:shorts|trousers|pants|jeans|leggings|tights|socks|boots|shoes|"
     r"gloves|top|vest|jumper|sweater|hoodie|trousers)\b", "fabric rustling"),
    (r"\bkeys?\b",                                  "keys on a ring"),
    (r"\b(?:wakes?\s+up|woke|gasp(?:s|ing)?|pant(?:s|ing)?|breath(?:es|ing)?)\b",
                                                    "breathing"),
)
MAX_SOUNDS = 3      # a shot's audio needs a cue, not an inventory
_VOCAL_RETIRES = (EFFORT_BREATH, "breathing")
_VOCAL_BETWEEN = "breathing"
# The six above, as a set: see the tail of sounds_for for why they are special-cased.
_NAMED_VOCALS = frozenset(("whimpering", "sobbing", "moaning", "groaning",
                           "screaming", "whining"))
_SOUND_SUPERSEDES = {
    "cuffs ratcheting closed": ("cuffs knocking",),
    EFFORT_BREATH: ("breathing",),
    "whimpering": _VOCAL_RETIRES,
    "sobbing": _VOCAL_RETIRES,
    "moaning": _VOCAL_RETIRES,
    "groaning": _VOCAL_RETIRES,
    "screaming": _VOCAL_RETIRES,
    "whining": _VOCAL_RETIRES,
}

_ROOM_TONE = (
    (r"\b(?:bathroom|shower|tiled?|tiles)\b",       "tiled walls ringing"),
    (r"\b(?:basement|cellar|warehouse|garage|hangar|tunnel|stairwell|"
     r"corridor|concrete|stone|brick|bare walls?)\b", "hard walls giving the sound back"),
    (r"\b(?:outside|outdoors|street|road|yard|garden|forest|beach|park)\b"
     r"|(?<!depth of )\bfield\b(?! of view)",       "open air with no walls close by"),
    (r"\b(?:carpet(?:ed)?|curtains?|bedroom|sofa|cushions?)\b",
                                                    "a soft room with little echo"),
    (r"\b(?:barn|attic|loft|shed|workshop|hall|church)\b", "a large room with a long tail"),
)


_AMBIENT = (
    (r"\brain(?:ing|y)?\b|\bdownpour\b|\bdrizzl", "rain against the glass"),
    (r"\bstorm|\bthunder", "a storm somewhere outside"),
    (r"\bwind(?:y)?\b|\bgale\b", "wind against the building"),
    (r"\bbeach\b|\bsea\b|\bocean\b|\bshore\b", "the sea a long way off"),
    (r"\bforest\b|\bwoods?\b", "wind in the trees"),
    (r"\bgarden\b|\byard\b|\bpark\b", "birdsong"),
    (r"\bstreet\b|\broad\b|\btraffic\b|\bcity\b|\bpavement\b",
     "traffic somewhere off the street"),
    (r"\bcar\b|\bvan\b|\btruck\b|\bdriving\b", "an engine idling"),
    (r"\bkitchen\b", "a fridge humming"),
    (r"\bbathroom\b|\bshower\b", "water moving in the pipes"),
    (r"\bnursery\b|\bbaby\b|\bcot\b|\bcrib\b", "a clock ticking"),
    (r"\bbedroom\b", "the quiet of a bedroom"),
    (r"\boffice\b|\bstudy\b", "a computer fan"),
    (r"\bworkshop\b|\bgarage\b|\bfactory\b", "a strip light humming"),
    (r"\bbasement\b|\bcellar\b|\bboiler\b", "a low hum off the strip light"),
    (r"\bhospital\b|\bward\b|\bclinic\b", "a monitor somewhere down the corridor"),
    (r"\bcafe\b|\bbar\b|\brestaurant\b|\bpub\b", "cutlery and moving chairs"),
    (r"\bschool\b|\bclassroom\b", "a corridor beyond the door"),
    (r"\bchurch\b|\bhall\b", "the air of a large empty room"),
    (r"\bstairs?\b|\bstairwell\b|\bhallway\b|\bcorridor\b",
     "the hollow quiet of a hallway"),
    (r"\bnight\b|\blate evening\b", "the quiet of a night"),
    # The generic interior LAST, so a named room wins.
    (r"\bhome\b|\bhouse\b|\bflat\b|\bapartment\b|\bliving room\b|\blounge\b"
     r"|\bindoors?\b|\broom\b", "the quiet of a house"),
)

def scene_ambient(*texts):
    """One ambient bed for the film, read from the anchor and the scene. "" if none.

    First match wins, and the table is ordered most specific first: weather and
    named places before the generic interior. ONE bed, not a list -- a shot told
    four things to sound like is a shot inventing which."""
    joined = " ".join(str(t or "") for t in texts)
    if not joined.strip():
        return ""
    for pat, phrase in _AMBIENT:
        if re.search(pat, joined, re.I):
            return phrase
    return ""


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
    for specific, general in _SOUND_SUPERSEDES.items():
        if specific in out:
            out = [p for p in out if p == specific or p not in general]
    if out and all(p in _NAMED_VOCALS for p in out):
        return []
    return out


def named_vocals_in(beat):
    """The non-speech vocals THIS BEAT NAMES, in the order the table lists them.

    sounds_for deliberately returns [] when a vocal is all the beat says: the beat
    goes to the model verbatim and the node has nothing to add over the top of it.
    That is right where the node then says nothing -- and wrong the moment it says
    something EXCLUSIVE.

    "She screams." is sound_described, so _own is true and the inferred list is
    zeroed; exertion_in is also true, so _voiced keeps the branch open rather than
    letting the shot be muted; then the ambient bed is appended and only=not _speaks
    closes the list. The shot was conditioned on "The only sound is an engine
    idling" -- an exclusive claim, against a beat that says she screams, on the one
    kind of shot whose branch is open and therefore has to fill itself with
    something. Reproduced on "She screams.", "She sobs quietly." and "She starts
    whimpering and thrashes in her restraints."

    So the closed list gets the author's own vocal put back into it. This adds
    nothing the node inferred -- these are the author's words, matched literally --
    and it is what keeps the exclusive sentence true."""
    b = str(beat or "")
    return [phrase for pat, phrase in _VOCAL_FROM if re.search(pat, b, re.I)]


def sound_clause(phrases, only=False, written=False):
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
        if written:
            return (f" The only sounds are the ones this beat describes, with "
                    f"{heard} under them.")
        verb = "is" if len(phrases) == 1 else "are"
        return f" The only sound{'' if len(phrases) == 1 else 's'} {verb} {heard}."
    return f" It sounds like {heard}."


_TALKER_DEVICE = (r"(?:televisions?|tvs?|telly|screens?|radios?|speakers?|stereos?|"
                  r"tannoys?|intercoms?|phones?|telephones?|laptops?|monitors?|"
                  r"record\s+players?|pa\s+systems?|answerphones?|announcements?)")
_DEVICE_SAYS = re.compile(
    r"\b" + _TALKER_DEVICE + r"\b(?:\s+[\w,']+){0,3}?\s+"
    r"(?:says?|said|announces?|announced|blares?|blared|plays?|played|calls?|called|"
    r"reads?|talks?|talking|goes|went|crackles?|drones?|repeats?|asks?)\b", re.I)
_NOT_A_NAME = (r"(?!(?:The|A|An|It|This|That|These|Those|There|Then|Here|His|Her|Their|"
               r"Its|Our|My|Your|When|While|As|But|And|One|Now|So|No|Yes|Somebody|"
               r"Someone|Nobody|Everyone|"
               r"TV|TVs|PA|Television|Televisions|Telly|Radio|Radios|Screen|Screens|"
               r"Speaker|Speakers|Stereo|Intercom|Phone|Telephone|Laptop|Monitor)\b)")
_SAYS = (r"says?|said|asks?|asked|whispers?|whispered|shouts?|shouted|calls?|"
         r"called|repl(?:y|ies|ied)|answers?|answered|adds?|added|murmurs?|"
         r"murmured|mutters?|muttered|tells?|told|begs?|begged|snaps?|snapped|"
         r"breathes?|breathed|hisses|hissed")


_TO_VERB = r"[^.!?\n]{0,80}?"

# A possessive is not a speaker. "Dana's phone says" is the phone talking.
_NOT_POSSESSIVE = r"(?!['\u2019]s\b)"

_PERSON_SAYS = re.compile(
    r"\b(?:he|she|they|i|we|you|" + _NOT_A_NAME + r"[A-Z][\w-]+)\b" + _NOT_POSSESSIVE
    + _TO_VERB + r"\s(?:" + _SAYS + r")\b")


def speech_is_a_devices(beat, sheet=""):
    """Is the only spoken line in this beat coming out of a machine?

    False whenever a person might have it, including when nothing attributes the
    line at all -- an unattributed quote in a beat about people is a person talking."""
    b = beat or ""
    if not has_speech(b) or not _DEVICE_SAYS.search(b):
        return False
    outside = _DIALOGUE_TAG.sub(" ", _QUOTED.sub(" ", b))
    if _PERSON_SAYS.search(outside):
        return False
    for n, _ in sheet_lines(sheet):
        if n and re.search(r"\b" + re.escape(n) + r"\b" + _NOT_POSSESSIVE + _TO_VERB
                           + r"\s(?:" + _SAYS + r")\b", outside, re.I):
            return False
    return True


def device_voice_clause(beat):
    """Say which machine the voice is coming out of, so no face is given it."""
    b = beat or ""
    _said = _DEVICE_SAYS.search(b)
    _span = _said.group(0) if _said else b
    _hits = list(re.finditer(r"\b" + _TALKER_DEVICE + r"\b", _span, re.I))
    if not _hits:
        return ""
    m = _hits[-1]
    thing = re.sub(r"\s+", " ", m.group(0))
    return (f" The voice in this shot is the {thing}'s, coming out of it across the "
            f"room, and the people listening let it play, their own mouths closed.")


_SAYS_NOTHING = re.compile(
    r"\b(?:says?|said|speaks?|spoke)\s+(?:absolutely\s+|almost\s+)?"
    r"(?:nothing|not\s+a\s+word|no\s+more|none)\b"
    r"|\b(?:does|do|did|would|will|could)\s*n[o']?t\s+(?:say|speak|answer|reply)\b"
    r"|\bnever\s+(?:says?|said|speaks?|spoke)\b"
    r"|\b(?:stays?|stayed|remains?|remained|keeps?|kept)\s+(?:quiet|silent)\b"
    r"|\bin\s+silence\b|\bwithout\s+(?:a\s+word|speaking|answering)\b", re.I)


def _in_beat_order(names, beat):
    """The names sorted by where the BEAT first mentions them.

    speakers_in walks the sheet, so it returned sheet order -- and the lock clause
    now says "Dan speaks first, then Mara", which is a claim about the beat."""
    b = beat or ""
    def at(n):
        m = re.search(r"\b" + re.escape(n) + r"\b", b, re.I)
        return m.start() if m else len(b)
    return sorted([n for n in names if n], key=at)


def speakers_in(beat, sheet=""):
    """Who this beat gives a line to. [] when it names nobody.

    A shot where one of two people speaks is a SPEAKING shot, so the mouth guard
    stood down for both -- and the listener's mouth was left as free as the
    speaker's. That is the commonest scene there is, and the lip-sync problem the
    guard exists for lands squarely on the person saying nothing."""
    b, out = beat or "", []
    b = " ".join(part for part in re.split(r"(?<=[.!?\"\u201d>])\s+", b)
                 if not _SAYS_NOTHING.search(part))
    for n, _ in sheet_lines(sheet):
        if not n:
            continue
        if re.search(r"\b" + re.escape(n) + r"\b" + _UP_TO_TWO_WORDS
                     + r"\s+(?:" + _SAYS + r")\b", b, re.I):
            out.append(n)
    if not out:
        for n, _ in sheet_lines(sheet):
            if not n:
                continue
            if re.search(r"[\"'”’]|</d>", b) and re.search(
                    r"(?:[\"'”’]|</d>)\s*[,.;]?\s*(?:" + _SAYS + r")\s+"
                    + re.escape(n) + r"\b", b, re.I):
                out.append(n)
    if not out and has_speech(b):
        at = {}
        for n, ln in sheet_lines(sheet):
            if not n:
                continue
            m = re.search(r"\b" + re.escape(n) + r"\b", b)
            if m:
                at[n] = m.start()
            group = sheet_pronoun(ln)
            if not group:
                continue
            if sum(1 for _n, _l in sheet_lines(sheet)
                   if _n and sheet_pronoun(_l) == group) != 1:
                continue
            pm = re.search(r"\b(?:" + "|".join(sorted(_PRONOUN_SET[group]))
                           + r")\b", b, re.I)
            if pm and (n not in at or pm.start() < at[n]):
                at[n] = pm.start()
        if at:
            out.append(min(at, key=at.get))
    return _in_beat_order(out, beat)


SPOKEN_LANGUAGE = "English"
LANGUAGE_HOLD = " The language is {lang}."

_NON_LATIN = re.compile(
    r"[^\x00-\x7F\u00C0-\u024F\u0300-\u036F"
    r"\u2018\u2019\u201C\u201D\u2013\u2014\u2026]")


_HARD_TO_SAY = re.compile(
    r"\b\d[\d:.,/\-]*\d\b|\b\d\b"
    r"|\b(?:Mr|Mrs|Ms|Dr|Prof|Sgt|Lt|Capt|Rev|Hon|St|Ave|Rd|Blvd|Jr|Sr|"
    r"vs|etc|approx|dept|Inc|Ltd|Co)\."
    r"|[&%$#@+=]", re.I)


def non_latin_in(text):
    """The distinct non-Latin characters in this text, in order. [] when clean."""
    out = []
    for ch in str(text or ""):
        if _NON_LATIN.match(ch) and ch not in out:
            out.append(ch)
    return out


_ORDERED = re.compile(
    r"\b(?:take|takes|taking|pull|pulls|remove|removes|undo|undoes|unfasten|"
    r"unbuckle|unzip|slip|slips|step|steps|get|gets|lie|lies|lay|lays|sit|sits|"
    r"kneel|kneels|stand|stands|turn|turns|come|comes|go|goes|put|puts|hold|"
    r"holds|open|opens|close|closes)\b", re.I)


def told_to_act(beat, speakers, described):
    """Who is being TOLD to do something in this beat's dialogue. [] when nobody.

    Only where the quoted line contains an action verb, and only for people the
    shot describes who are not the one speaking -- the listener is the one whose
    body the instruction is about, and the one the model will move early."""
    b = str(beat or "")
    if not b:
        return []
    said = " ".join(m.group(0) for m in _QUOTED.finditer(b))
    if not said or not _ORDERED.search(said):
        return []
    talking = {n for n in (speakers or []) if n}
    return [n for n in (described or []) if n and n not in talking]


def told_hold(listeners):
    """Give the listener something to be doing while the line is said."""
    who = [n for n in (listeners or []) if n]
    if not who:
        return ""
    if len(who) == 1:
        return f" {who[0]} listens, wearing what the sheet already lists."
    said = ", ".join(who[:-1]) + " and " + who[-1]
    return f" {said} listen, wearing what the sheet already lists."


# The tail both voice guards end on, defined once so they cannot drift apart.
MOUTH_HOLD_REST = "every other mouth in the shot stays closed, those expressions moving"
_UP_TO_TWO_WORDS = r"(?:\s+(?!and\b|but\b|then\b|who\b|,\s*who\b)[\w,']+){0,2}?"


_VOCAL_SOURCE = (
    (r"whimper(?:s|ing|ed)?", "whimpering"), (r"sob(?:s|bing|bed)?", "sobbing"),
    (r"moan(?:s|ing|ed)?", "moaning"),       (r"groan(?:s|ing|ed)?", "groaning"),
    (r"scream(?:s|ing|ed)?", "screaming"),   (r"whin(?:e|es|ing|ed)", "whining"),
    (r"gasp(?:s|ing|ed)?", "gasping"),       (r"pant(?:s|ing|ed)?", "panting"),
    (r"cr(?:y|ies|ying|ied)", "crying"),     (r"sigh(?:s|ing|ed)?", "sighing"),
    (r"shriek(?:s|ing|ed)?", "shrieking"),   (r"yelp(?:s|ing|ed)?", "yelping"),
    (r"grunt(?:s|ing|ed)?", "grunting"),     (r"weep(?:s|ing)?", "weeping"),
    (r"wail(?:s|ing|ed)?", "wailing"),       (r"laugh(?:s|ing|ed)?", "laughing"),
)


def vocal_sources_in(beat, sheet=""):
    """Who this beat says is making a non-speech vocal, and what it is.

    [(name, phrase)], in sheet order. Same shape as speakers_in, including its
    conjunction guard: "Dan holds the door and McKenna sobs" must not credit Dan,
    because `and` starts a new predicate with its own subject -- and crediting the
    wrong person here is worse than crediting nobody, since the shot would then hold
    the mouth of whoever is actually making the noise."""
    b, out = beat or "", []
    for n, _ in sheet_lines(sheet):
        if not n:
            continue
        for pat, phrase in _VOCAL_SOURCE:
            if re.search(r"\b" + re.escape(n) + r"\b" + _UP_TO_TWO_WORDS
                         + r"\s+(?:" + pat + r")\b", b, re.I):
                out.append((n, phrase))
                break
            if re.search(r"\b" + re.escape(n) + r"\b(?:\s*,\s*[\w'\u2019-]+)*"
                         r"\s+and\s+[\w'\u2019-]+\s+"
                         r"(?:" + pat + r")\b", b, re.I):
                out.append((n, phrase))
                break
    return out


def _joined(names):
    """'Dan', 'Dan and Sam', 'Dan, Sam and Mara'."""
    ns = [n for n in (names or []) if n]
    if len(ns) < 2:
        return ns[0] if ns else ""
    return ", ".join(ns[:-1]) + " and " + ns[-1]


def voice_sources(talkers, vocal, vocalisers, silent):
    """Say whose voice is whose, and close the mouths that are neither.

    Two jobs, and the second is the reported one. Closing the rest stops the
    listener babbling on a branch somebody else's sob opened. NAMING THE SOURCES
    stops the model swapping them -- two voices in one shot with nothing saying
    which is which is a shot where he can be given her whimper and she his line.

    So the sentence is emitted for two DIFFERENT sources even when nobody is left to
    hold: with one source and nobody silent there is nothing to disambiguate and
    nothing to close, and the shot is left alone."""
    parts = []
    if len(talkers or []) == 1:
        parts.append(f"only {talkers[0]} speaks")
    elif talkers:
        parts.append(f"{talkers[0]} speaks first, then "
                     + ", then ".join(talkers[1:]))
    if vocalisers and vocal:
        parts.append(f"the {vocal} is {_joined(vocalisers)}'s")
    if not parts or (len(parts) == 1 and not silent and len(talkers or []) < 2):
        return ""
    if silent:
        parts.append(MOUTH_HOLD_REST)
    said = "; ".join(parts)
    return f" {said[0].upper()}{said[1:]}."

ONE_VOICE = (" Only the person speaking has their mouth moving; every other mouth "
             "in the shot stays closed, those expressions moving.")


_PLAIN_QUOTED = re.compile(r"[\"“]([^\"“”]{1,400}?)[\"”]")


def mark_dialogue(beat):
    """Wrap plainly-quoted speech in H3's <d>...</d>. Unchanged when there is none.

    Left alone where the author has already marked it, and where a quote is not
    speech at all. A LINE ends in terminal punctuation and a scare quote does not:
    "Wait." is one word and is speech, a "vintage" coat is emphasis. Counting words
    got both of those backwards."""
    b = str(beat or "")
    if not b or "<d>" in b:
        return b

    def _wrap(m):
        said = m.group(1).strip()
        if not said:
            return m.group(0)
        if said[-1] in ".!?":
            return "<d>" + said + "</d>"
        if len(said.split()) < 2:
            return m.group(0)
        before = b[max(0, m.start() - 40):m.start()]
        if re.search(r"(?:" + _SAYS + r")\b[^.]{0,12}$|[:,]\s*$", before, re.I):
            return "<d>" + said + "</d>"
        after = b[m.end():m.end() + 40]
        if re.match(r"[\s,]*(?:[A-Za-z][\w'’-]*\s+){0,2}?(?:" + _SAYS + r")\b",
                    after, re.I):
            return "<d>" + said + "</d>"
        return m.group(0)

    return _PLAIN_QUOTED.sub(_wrap, b)


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
    try:
        _deep_cleanup()
    except Exception:
        pass
    old_fmt, _n, old_sz, _c = prev
    new_fmt = fp[0]
    return (f"model changed since last run ({old_fmt} ~{old_sz / GB:.1f}GB -> {new_fmt} "
            f"~{fp[2] / GB:.1f}GB): flushed all resident models and VRAM caches")


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
                 "near-clean reference shows up as the opening frame, AND IT DOES NOT "
                 "STAY THERE: every later shot opens on the previous shot's last "
                 "frame, so whatever composition shot 1 settles on is handed down the "
                 "whole chain. A portrait reproduced at shot 1 is therefore a portrait "
                 "framing for the film, which is what 'the camera is fixated on her' "
                 "is. Wire a wide establishing frame into first_frame and shot 1 is "
                 "pinned to that composition instead -- it is the one input that "
                 "outranks a reference, because it IS frame one")
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
    step = max(1, max(int(img.shape[0]), int(img.shape[1])) // 256)
    x = img[::step, ::step].float()
    if x.dim() == 3 and x.shape[-1] >= 3:
        x = x[..., :3].mean(dim=-1)
    elif x.dim() == 3:
        x = x[..., 0]
    if x.dim() != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return 0.0, 0.0
    gx = (x[:, 1:] - x[:, :-1]).abs().mean()
    gy = (x[1:, :] - x[:-1, :]).abs().mean()
    return float((gx + gy) * 0.5), float(x.std())


def levels_report(levels, shots, strength=None):
    """What hold_levels measured, and what it did about it.

    Worth printing even when it corrected nothing: the measurement is the evidence that
    the chain is or is not cooking, and a run that measured a drift too small to act on
    is a different thing from a run that never looked."""
    if levels is None:
        return ""
    g, o = levels.estimate()
    if g is None:
        return ""
    pct = "/".join(f"{(float(torch.exp(v)) - 1.0) * 100.0:+.1f}%" for v in g)
    lvl = "/".join(f"{float(v):+.4f}" for v in o)
    line = (f"hold_levels: measured the chain drifting {pct} of contrast and {lvl} of level "
            f"per boundary, per R/G/B channel, from {len(levels._bg)} boundary(ies)")
    n = len(levels.applied)
    if not n:
        if strength is not None and strength <= 0:
            line += (" -- and did nothing about it, because hold_levels is 0. The "
                     "measurement above is what the chain is doing unattended; raise "
                     "hold_levels to take it back out")
        else:
            line += (" -- below the 8-bit floor a handoff is quantised to, so nothing was "
                     "applied rather than claiming a correction that would be erased")
    else:
        last = levels.applied[-1][0]
        line += (f", and took it back out of {n} handoff(s); the last gain applied was "
                 f"{'/'.join(f'{float(v):.3f}' for v in last)}. The contrast line above is "
                 f"measured on the corrected frames, so it is the residual, not the defect")
    return line


def detail_report(per_shot):
    """Two lines: whether the chain is softening, and whether it is COOKING.

    per_shot is [(detail, contrast), ...] measured on each shot's last frame.

    Contrast used to be measured here and thrown away, which was the worst possible
    arrangement: the surviving metric RISES with burn-in -- expanding contrast creates
    neighbour differences -- so a chain visibly cooking printed "UP n%, so the chain is
    not softening" and read as reassurance. The reported symptom was being measured on
    exactly the right frame and never shown. Both trends are reported now, and the
    detail line no longer pronounces on a rise it cannot explain by itself."""
    ds = [d for d, _ in per_shot if d > 0]
    cs = [c for _, c in per_shot if c > 0]
    if len(ds) < 2:
        return ""
    out = []
    drop = (ds[0] - ds[-1]) / ds[0] * 100.0 if ds[0] else 0.0
    line = "detail per shot (last frame): " + " ".join(f"{d:.4f}" for d, _ in per_shot)
    if drop >= 10.0:
        line += (f" -- DOWN {drop:.0f}% from shot 1 to shot {len(ds)}. Each boundary "
                 f"decodes a shot, takes its LAST frame and re-encodes it as the next "
                 f"shot's keyframe, so the loss of one round trip is carried into the "
                 f"next and compounds. Break the chain to stop it accumulating: "
                 f"restart_after_removal stops a shot opening on the previous frame, "
                 f"at the cost of a cut there")
    elif drop <= -10.0:
        line += (f" -- UP {-drop:.0f}%. Read the contrast line before taking that as good "
                 f"news: expanding contrast raises this number too")
    else:
        line += f" -- flat within {abs(drop):.0f}%"
    out.append(line)
    if len(cs) >= 2:
        rise = (cs[-1] - cs[0]) / cs[0] * 100.0 if cs[0] else 0.0
        cl = "contrast per shot (last frame): " + " ".join(f"{c:.4f}" for _, c in per_shot)
        if rise >= 10.0:
            cl += (f" -- UP {rise:.0f}% from shot 1 to shot {len(cs)}, which is the chain "
                   f"COOKING: every shot is sampled from the previous shot's last frame, "
                   f"the model reproduces it with a little more contrast, and the VAE "
                   f"clamps the result to 0..1 -- so the headroom each pass spends is "
                   f"never given back, and it shows as crushed blacks and blown "
                   f"highlights rather than merely as more contrast. hold_levels takes "
                   f"the per-boundary part of it back out")
        elif rise <= -10.0:
            cl += f" -- DOWN {-rise:.0f}%, so the chain is flattening rather than cooking"
        else:
            cl += f" -- flat within {abs(rise):.0f}%"
        out.append(cl)
    return " | ".join(out)


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
    ms = copy.deepcopy(m.get_model_object("model_sampling"))
    sig = inspect.signature(ms.set_parameters)
    kwargs = {}
    if "shift" in sig.parameters:
        kwargs["shift"] = float(shift_video)
    if "audio_shift" in sig.parameters:
        kwargs["audio_shift"] = float(shift_audio)
    if not kwargs:
        raise RuntimeError("set_parameters takes no shift")
    ms.set_parameters(**kwargs)
    m.add_object_patch("model_sampling", ms)
    return m


def audio_sigma_of(video_sigma, shift_video, shift_audio):
    """The audio branch's sigma at a given video sigma. comfy's time_shift_sigma."""
    v, a, s = float(shift_video), float(shift_audio), float(video_sigma)
    base = s / (v + s * (1.0 - v))
    return a * base / (1.0 + (a - 1.0) * base)


def video_sigma_for_audio(target_audio, shift_video, shift_audio):
    """The video sigma that puts the audio branch on `target_audio`. The inverse."""
    v, a, t = float(shift_video), float(shift_audio), float(target_audio)
    base = t / (a - t * (a - 1.0))
    return base * v / (1.0 - base + base * v)


def insert_audio_landing(sigmas, shift_video, shift_audio,
                         target=0.03, coarse=0.10):
    """One extra step so the audio branch does not land from a great height.

    Returns a new list, or the input unchanged when there is nothing to do. This
    runs inside the render path, so anything unexpected -- an empty schedule, no
    trailing zero, a tail that is already soft -- returns the input rather than
    raising. It never inserts twice: after one pass the tail is below `coarse`.

    `target` is 0.03 because that is roughly what `normal` achieves on its own, and
    it is comfortably above the 0.003 kl_optimal leaves -- close enough to free, far
    enough from zero that the extra step is doing work rather than nothing."""
    try:
        out = [float(x) for x in (sigmas or [])]
    except (TypeError, ValueError):
        return sigmas
    if len(out) < 3 or out[-1] != 0.0 or out[-2] <= 0.0:
        return sigmas
    if audio_sigma_of(out[-2], shift_video, shift_audio) <= coarse:
        return sigmas
    land = video_sigma_for_audio(target, shift_video, shift_audio)
    # Strictly inside the final jump, or the schedule stops being monotonic.
    if not (0.0 < land < out[-2]):
        return sigmas
    return out[:-1] + [land, 0.0]


def last_audio_sigma(steps, shift_audio, scheduler="simple", shift_video=None):
    """How much audio noise is still left going into the FINAL sampling step.

    The audio branch runs on its own shifted timeline: time_shift_sigma inverts the
    video shift and re-applies the audio one, so what reaches the last step depends
    on the STEP COUNT and shift_audio -- and not at all on shift_video, which is the
    dial everybody reaches for.

    The base grid's last position before zero is 1/steps, so

        sigma_audio(last) = shift_audio / (steps + shift_audio - 1)

    At the 8 steps this node defaults to, shift_audio 3.0 leaves 0.30. At the 4 a
    distilled LoRA wants, the same 3.0 leaves 0.50 -- half of the audio denoising
    crammed into one step, and an audio branch resolving half its noise in a single
    jump is one that invents whatever is easiest. Reported as babble starting at
    step 3 of 4, which is that step.
    """
    try:
        n = max(1, int(steps))
        a = float(shift_audio)
    except (TypeError, ValueError):
        return 0.0
    v = float(shift_video) if shift_video else _WIDGET_RANGE["shift_video"][0]
    try:
        import comfy.samplers as _cs
        import comfy.model_sampling as _cms
        _calc = getattr(_cs, "calculate_sigmas", None)
        if _calc is not None:
            _ms = _cms.ModelSamplingDiscreteFlow()
            _ms.set_parameters(shift=v)
            _sig = [float(x) for x in _calc(_ms, str(scheduler), n)]
            _last = next((x for x in reversed(_sig) if x > 0.0), 0.0)
            # invert to the base grid at shift_video, re-apply shift_audio
            _base = _last / (v + _last * (1.0 - v))
            return a * _base / (1.0 + (a - 1.0) * _base)
    except Exception:
        pass
    return a / (n + a - 1.0) if (n + a - 1.0) > 0 else 0.0


def scheduler_that_finishes_audio(steps, shift_audio, shift_video=None,
                                  current="simple", target=0.10):
    """The shipped scheduler that leaves the LEAST audio noise on the last step.

    ONLY ONE THAT HONOURS shift_video, and that restriction is the whole of what
    this function got wrong. Reported: switching to kl_optimal put watery waves on
    the picture.

    comfy/samplers.py grades its schedulers by `use_ms`. A handler with use_ms True
    is called as handler(model_sampling, steps) and sees the shift; one with use_ms
    False is called as handler(n, sigma_min, sigma_max) and NEVER SEES IT. kl_optimal
    exponential and karras are all in the second group, so recommending them threw
    shift_video away silently. At 5 steps and shift 12 the difference is the whole
    schedule:

        simple      1.0  0.9796  0.9474  0.8889  0.7500   <- stays high, as shift 12 asks
        kl_optimal  1.0  0.6725  0.4212  0.2082  0.0119   <- shift discarded

    The video branch gets almost no time at high sigma, so structure never resolves
    and the remaining steps polish detail with nothing underneath it. That is what
    watery looks like. The audio tail WAS better; it was better because the schedule
    had stopped being the one that was asked for.

    Returns (name, sigma) when a shift-honouring scheduler would get under `target`
    and beat what is selected, else None. Named rather than silently switched: the
    schedule shape changes the picture too, and that is the reader's call."""
    try:
        import comfy.samplers as _cs
        names = [n for n in (getattr(_cs.KSampler, "SCHEDULERS", []) or [])
                 if getattr(_cs.SCHEDULER_HANDLERS.get(n, None), "use_ms", False)]
    except Exception:
        return None
    def _video_ok(nm):
        try:
            import comfy.model_sampling as _cms
            _ms = _cms.ModelSamplingDiscreteFlow()
            _ms.set_parameters(shift=float(shift_video or 12.0))
            sig = [float(x) for x in _cs.calculate_sigmas(_ms, nm, int(steps))]
        except Exception:
            return False
        return (len(sig) >= 3 and sig[0] >= 0.999
                and sum(1 for x in sig if x > 0.5) >= max(1, int(steps) - 1))

    now = last_audio_sigma(steps, shift_audio, current, shift_video)
    best, best_s = None, now
    for nm in names:
        if nm == current:
            continue
        try:
            sg = last_audio_sigma(steps, shift_audio, nm, shift_video)
        except Exception:
            continue
        if sg > 0.0 and sg < best_s and _video_ok(nm):
            best, best_s = nm, sg
    return (best, best_s) if (best is not None and best_s <= target) else None


# What shift_audio 3.0 leaves on the last step at the 8 this node defaults to.
DEFAULT_LAST_AUDIO_SIGMA = 0.30


def shift_audio_for(steps, target=None):
    """The shift_audio that reproduces a chosen last-step sigma at THIS step count.

    Inverting sigma = a / (steps + a - 1):

        a = sigma * (steps - 1) / (1 - sigma)

    The DIRECTION matters more than the arithmetic. sigma rises monotonically with
    shift_audio -- d/da = (steps - 1) / (steps + a - 1)**2, positive for every step
    count above one -- so fewer steps need a SMALLER shift_audio, not a larger one.

    The note this feeds scaled the other way: 3.0 * 8 / steps, which at the 4 steps
    a distilled LoRA wants advised 6.0 and took the last step from 0.50 to 0.67.
    That is the babble dial turned the wrong way, printed on the one report that
    only fires when somebody is already hearing babble. Nothing caught it because
    the tests covered last_audio_sigma, which was right, and not the advice.

    Clamped to the widget's own range so the number printed is one that can be
    typed in; where the floor binds, the caller reports the sigma it really gives
    rather than the one that was asked for.
    """
    s = DEFAULT_LAST_AUDIO_SIGMA if target is None else float(target)
    try:
        n = max(1, int(steps))
    except (TypeError, ValueError):
        return 0.0
    if not 0.0 < s < 1.0:
        return 0.0
    lo, hi = _WIDGET_RANGE["shift_audio"][1], _WIDGET_RANGE["shift_audio"][2]
    return min(max(s * (n - 1) / (1.0 - s), lo), hi)


# What shift_video 12 -- ComfyUI's own H3 default, nodes_minimax_h3.py -- leaves on
# the final step at the ~20 steps the UNDISTILLED model is sampled at. 12 is not a
# wrong number; it is a number that was chosen against a step count, and it stops
# being right the moment a turbo LoRA drops that count. This is the target every
# correction below aims at, so "corrected" means "the schedule H3's own default
# already produces when it has the steps it was picked for".
REFERENCE_FINAL_JUMP = 0.40

# A distilled step target lives in the FILENAME and nowhere else. Digits BEFORE the
# word, so `4step`, `8step` and `3step` match and `step600` does NOT: that is a
# training checkpoint -- minimax_h3_turbo_v4_step600 and the lightx2v dareties build
# both carry it -- and reading it as a sampling target would advise 600 steps off a
# 4-step LoRA. Two digits max for the same reason.
_LORA_STEPS_RX = re.compile(r"(?<![a-z0-9])(\d{1,2})\s*[-_ ]?step(?![a-z0-9])", re.I)


def lora_step_targets(graph):
    """[(steps, filename)] for every LoRA in the workflow whose NAME states a step count.

    NOTHING INSIDE A LORA SAYS WHAT SCHEDULE IT WANTS. The safetensors metadata of a
    distilled H3 LoRA carries rank, alpha, baked_scale and conversion provenance --
    and no sigma, no shift and no step count. comfy stores only that metadata dict
    (comfy/sd.py, set_attachments("lora_metadata")) and keeps the path it loaded from
    in the loader's own cache, so by the time a MODEL reaches this node the file name
    is gone. lora_facts() can still count the stack and read its strengths; it cannot
    say what any of them were trained for.

    Which leaves the file name as the only machine-readable statement of intent a
    turbo LoRA makes, and the hidden PROMPT as the only place it survives. `graph` is
    that dict: {node_id: {"class_type": str, "inputs": {...}}}. Every input whose key
    contains "lora_name" is read, which covers LoraLoader, LoraLoaderModelOnly and the
    stacker nodes that number their slots lora_name_1, lora_name_2 and so on.

    Returns newest-first by nothing in particular -- order is the graph's -- and drops
    duplicates, so the same LoRA wired to model and CLIP is one entry, not two.
    """
    out = []
    if not isinstance(graph, dict):
        return out
    for node in graph.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        for key, value in inputs.items():
            if "lora_name" not in str(key).lower() or not isinstance(value, str):
                continue
            m = _LORA_STEPS_RX.search(value)
            if not m:
                continue
            try:
                n = int(m.group(1))
            except (TypeError, ValueError):
                continue
            if 1 <= n <= _WIDGET_RANGE["steps"][2] and (n, value) not in out:
                out.append((n, value))
    return out


def final_video_jump(steps, shift_video, scheduler="simple"):
    """How much VIDEO noise the last sampling step has to clear on its own.

    comfy's schedules end at zero, so whatever sigma stands before that zero is
    removed in ONE evaluation. That number, not the step count, is what decides
    whether structure resolves: at shift 12 the video branch spends every step but
    the last nibbling the top of the schedule --

        steps=8   1.0  0.988  0.973  0.952  0.923  0.878  0.8  0.632  0.0

    -- and then crosses 0.632 in a single jump. Seven steps of polish on top of noise,
    one step to invent the anatomy underneath. Reported as limbs and faces that render
    partially: half an arm, a hand that stops. At the 3-4 steps a distilled LoRA wants
    the same shift leaves 0.86 and 0.80, which is nearly the whole denoise in one move.

    For `simple` this is exactly shift / (steps + shift - 1) -- the same inversion
    last_audio_sigma() runs on the audio branch -- but the real schedule is asked for
    when comfy is importable so any scheduler answers honestly.
    """
    try:
        n = max(1, int(steps))
        v = float(shift_video)
    except (TypeError, ValueError):
        return 0.0
    if v <= 0.0:
        return 0.0
    try:
        import comfy.samplers as _cs
        import comfy.model_sampling as _cms
        _ms = _cms.ModelSamplingDiscreteFlow()
        _ms.set_parameters(shift=v)
        sig = [float(x) for x in _cs.calculate_sigmas(_ms, str(scheduler), n)]
        last = next((x for x in reversed(sig) if x > 0.0), 0.0)
        if last > 0.0:
            return last
    except Exception:
        pass
    return v / (n + v - 1.0) if (n + v - 1.0) > 0 else 0.0


def shift_video_for_jump(steps, scheduler="simple", target=None):
    """The shift_video that leaves `target` on the final step at THIS step count.

    Inverting sigma = v / (steps + v - 1), the same shape shift_audio_for() inverts:

        v = sigma * (steps - 1) / (1 - sigma)

    and the DIRECTION is the half worth stating: sigma rises with shift, so FEWER
    steps need a SMALLER shift_video, not a larger one. The instinct runs the other
    way -- a short schedule feels like it needs more shift to hold structure -- and
    following it is what turns 4 steps into a 0.8 final jump.

    Bisected against the real schedule where comfy is importable, because only the
    shift-honouring schedulers (use_ms True, the ones scheduler_that_finishes_audio
    already restricts itself to) have a closed form at all; the analytic inverse is
    the fallback. Clamped to the widget's own range so the number reported is one
    that can be typed in.
    """
    lo, hi = _WIDGET_RANGE["shift_video"][1], _WIDGET_RANGE["shift_video"][2]
    s = REFERENCE_FINAL_JUMP if target is None else float(target)
    try:
        n = max(1, int(steps))
    except (TypeError, ValueError):
        return None
    if not 0.0 < s < 1.0 or n < 2:
        return None                       # one step clears everything; nothing to aim at
    if final_video_jump(n, lo, scheduler) > s:
        return lo                         # even the floor overshoots: take the floor
    if final_video_jump(n, hi, scheduler) <= s:
        return hi
    a, b = lo, hi
    for _ in range(40):
        mid = (a + b) / 2.0
        if final_video_jump(n, mid, scheduler) <= s:
            a = mid
        else:
            b = mid
    # FLOORED, not rounded. The jump rises with shift, so rounding 4.6666 up to 4.67
    # puts the result back OVER the target it was just solved for -- by 0.0002, which
    # is nothing to look at and enough to make the caller report a shortfall that is
    # not there. Two decimals because that is what the widget steps in.
    return min(max(math.floor(a * 100.0) / 100.0, lo), hi)


def upstream_h3_shift(model):
    """(shift_video, shift_audio) if something upstream already set H3's schedule, else None.

    REPLACES A WIDGET THAT ASKED THE READER TO REPORT THIS. apply_model_sampling said
    "turn off only if you patch it upstream yourself" -- a question about the graph,
    put to the person least able to be sure of the answer, whose wrong answer patches
    the schedule twice or leaves it unset.

    comfy's own MiniMaxH3SigmaShift stamps what it applied into transformer_options
    (nodes_minimax_h3.py: to["minimax_h3_sigma_shift_video"] = shift_video), so a
    deliberate upstream patch announces itself and carries its own numbers.

    THE MODEL_SAMPLING OBJECT IS NOT THE TEST, and reading it instead is the trap
    this function exists to avoid: an H3 checkpoint already loads with the correct
    FLOW_AV 12/3 schedule on it, so "is a shift set" is true before anybody has
    touched anything and would stand the node's own patch down on every clean run.
    The stamp is the only thing that distinguishes a patch from a default."""
    try:
        to = (getattr(model, "model_options", None) or {}).get("transformer_options")
        if not isinstance(to, dict) or "minimax_h3_sigma_shift_video" not in to:
            return None
        return (float(to["minimax_h3_sigma_shift_video"]),
                float(to.get("minimax_h3_sigma_shift_audio", 0.0)))
    except (TypeError, ValueError, AttributeError):
        return None


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


def sparse_dit_patched(model):
    """True when something upstream installed a per-block DiT replace patch, False when it
    provably did not, None when this model cannot say.

    That patch is how ComfyUI's Model Sparse Attention node registers itself
    (set_model_patch_replace -> model_options["transformer_options"]["patches_replace"]
    ["dit"]), whatever method it was set to. None is distinct from False on purpose: a stub
    or hand-built model carries no model_options at all, and the one caller of this turns a
    True into a refusal, so "cannot tell" must never be read as "not there"."""
    opts = getattr(model, "model_options", None)
    if not isinstance(opts, dict):
        return None
    tops = opts.get("transformer_options") or {}
    if not isinstance(tops, dict):
        return None
    return bool((tops.get("patches_replace") or {}).get("dit"))


def sparse_attention_allocator_abort(model):
    """The one configuration that does not raise, it ABORTS. Returns why, or "".

    cudaMallocAsync is stream-ordered: a block allocated on one CUDA stream must be freed
    consistently with that stream. comfy_kitchen's chunked sparse-attention producer is a
    GENERATOR consumed from inside the kernel call, across the stream boundary that
    --enable-dynamic-vram's prefetch machinery sets up, and freeing its per-chunk tensor
    there returns CUDA_ERROR_INVALID_VALUE from cuMemFreeAsync. That throws out of a tensor
    DESTRUCTOR, where there is no Python frame to catch it, so the process calls
    std::terminate: a core dump, not an exception, taking the server and the rest of the
    queue with it.

    Worth refusing rather than warning for exactly that reason -- there is nothing to
    recover from an abort, and nothing downstream gets the chance to try. Losing one render
    to a readable error is the better trade.

    Nobody chooses this, either. ComfyUI force-enables the allocator on any CUDA 13 torch
    build and does not consult its own card blacklist on that path (cuda_malloc.py), so a
    current install arrives here by default."""
    conf = str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") or "")
    if "cudamallocasync" not in conf.lower():
        return ""
    if sparse_dit_patched(model) is not True:
        return ""
    return ("this render would ABORT the ComfyUI process rather than fail: a sparse-attention "
            "patch is on the model AND torch is using the cudaMallocAsync allocator "
            f"(PYTORCH_CUDA_ALLOC_CONF={conf}). Freeing the attention producer's per-chunk "
            "tensor under that allocator returns CUDA_ERROR_INVALID_VALUE from cuMemFreeAsync, "
            "inside a tensor destructor where nothing can catch it -- so the process "
            "core-dumps and the queue goes with it, which is why this stops here instead.\n\n"
            "Either restart ComfyUI with --disable-cuda-malloc, which is the flag ComfyUI's "
            "own cuda_malloc.py names for this failure, or take the Model Sparse Attention "
            "node out of the graph. ComfyUI turns that allocator on by itself on every CUDA 13 "
            "torch build without checking whether the card supports it, so this is the default "
            "rather than anything you picked.")


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


_REMOVE_LINE = re.compile(r"^[ \t]*(?:remove|removed|off)[ \t]*:[ \t]*(.+?)[ \t]*$",
                          re.I | re.M)

_LEGACY_FIELD = re.compile(
    r"^[ \t]*(?:overall_soundscape|non_diegetic_music)[ \t]*:.*$", re.I | re.M)
_LEGACY_PREFIX = re.compile(r"^[ \t]*\[(?:Generation|Shot)[ \t]*\d+\][ \t]*", re.I | re.M)

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
    out = re.sub(r"[ \t]*\n[ \t]*\n[ \t]*\n+", "\n\n", out)
    return out.strip(), n


_ADD_LINE = re.compile(r"^[ \t]*(?:add|wear|wearing)[ \t]*:[ \t]*(.+?)[ \t]*$", re.I | re.M)

_EXACT_LINE = re.compile(r"^[ \t]*(?:exact|exactly|verbatim)[ \t]*:[ \t]*(.+?)[ \t]*$",
                         re.I | re.M)


def exact_lines(beat):
    """[the author's verbatim sentences] for this beat, in the order written."""
    return [m.group(1).strip() for m in _EXACT_LINE.finditer(beat or "") if m.group(1).strip()]

_STRIP_VERB = engine._STRIP_VERB
_TRAILING_VERB = engine._TRAILING_VERB
_UNDO_VERB = engine._UNDO_VERB
_OUT_OF_VERB = (r"get(?:s|ting)?|got|shimm(?:y|ies|ied|ying)|squirm(?:s|ed|ing)?|"
                r"climb(?:s|ed|ing)?|ease[sd]?|easing|back(?:s|ed|ing)?|"
                r"step(?:s|ped|ping)?|wriggle[sd]?|wiggle[sd]?|struggle[sd]?")
_PUSH_VERB = (r"push(?:es|ed|ing)?|shove[sd]?|skim(?:s|med|ming)?|ease[sd]?|"
              r"easing|roll(?:s|ed|ing)?|work(?:s|ed|ing)?")

_OPENER_VERB = (r"unzip(?:s|ped)?|unbutton(?:s|ed)?|unfasten(?:s|ed)?|undo(?:es)?|undid|"
                r"unhook(?:s|ed)?|unclasp(?:s|ed)?")
_FINISHES_REMOVAL = re.compile(r"\b(?:off|away|out\s+of|remove[sd]?|removing|drops?|"
                               r"dropped|discard(?:s|ed)?|sheds?|"
                               r"lets?\s+(?:it|them)\s+(?:fall|drop|slide)|"
                               r"falls?\s+(?:to|down|away|off))\b", re.I)

_REMOVAL_PROSE = re.compile(
    r"\b(?:" + _UNDO_VERB + r")\b"
    r"|\b(?:" + _STRIP_VERB + r")\s+(?:off|away|out\s+of)\b"
    r"|\b(?:" + _TRAILING_VERB + r")\b(?=[^.;!?]{0,40}?\b(?:off|away)\b)"
    r"|\b(?:" + _STRIP_VERB + r")\b"
    r"(?=[^.;!?]{0,40}?\bover\s+(?:her|his|their|the)\s+head\b)"
    r"|\b(?:" + _OUT_OF_VERB + r")\s+(?:out|clear|free)\s+of\b"
    r"|\b(?:" + _PUSH_VERB + r")\s+(?:off|away)\b"
    r"|\b(?:" + _PUSH_VERB + r")\b(?=[^.;!?]{0,40}?\b(?:off|away)\b)"
    r"|\b(?:" + _STRIP_VERB + r"|" + _PUSH_VERB + r")\b"
    r"(?=[^.;!?]{0,40}?\bdown\s+(?:(?:her|his|their|the)\s+"
    r"(?:legs?|knees?|calves|shins?|ankles?|feet)\b|(?:and\s+)?(?:off|away)\b|"
    + engine.TO_THE_FLOOR + r"))"
    r"|\b(?:" + _STRIP_VERB + r"|" + _PUSH_VERB + r"|drop(?:s|ped|ping)?|"
    r"let(?:s|ting)?|lob(?:s|bed)?|fling(?:s|ing)?|flung|discard(?:s|ed|ing)?)\b"
    r"(?=[^.;!?]{0,40}?" + engine.TO_THE_FLOOR + r")",
    re.I)


_HAS_VERB = re.compile(
    r"\b(?:is|are|was|were|be|being|been|has|have|had|wears?|wearing|dressed|"
    r"walks?|walked|stands?|stood|sits?|sat|lies?|lying|holds?|holding|"
    r"cuts?|pulls?|takes?|steps?|turns?|looks?|comes?|goes)\b", re.I)


_ASKS = re.compile(r"\b(?:asks?|asked|begs?|begged|tells?|told|wants?|wanted|"
                   r"pleads?|pleaded|has|have|had|gets?|got)\b", re.I)


def _clause_about(beat, item=""):
    """The sentence/clause of `beat` that names `item`; the whole beat if it does not.

    An ask governs the garment it is ASKING about, not every garment in the beat.
    "Kate takes off her coat ... and asks him to get the scarf off" has one removal
    by her hands and one by his, and reading the ask against the whole beat gave
    both to him."""
    if not beat or not item:
        return beat or ""
    head = str(item).split()[-1]
    for part in re.split(r"(?<=[.;!?])\s+", str(beat)):
        if re.search(r"\b" + re.escape(head) + r"\b", part, re.I):
            return part
    return beat


def removal_agent(beat, cast, wearer=None, item=""):
    """Who takes the garment off in this beat. '' when the beat does not say.

    A beat with one person in it is that person undressing. With two, the one who is
    NOT the wearer is doing it when the wearer asks -- and when nobody asks, whoever
    the beat names first is acting, the same reading restrained_by_beat uses."""
    people = [n for n in (cast or []) if n]
    if not people:
        return ""
    if len(people) == 1:
        return people[0]
    b = beat or ""
    others = [n for n in people if n != wearer]
    if wearer and others and _ASKS.search(_clause_about(b, item)):
        return others[0]
    scope = _clause_about(b, item)
    first, at = "", len(scope) + 1
    for n in people:
        m = re.search(r"\b" + re.escape(n) + r"\b", scope, re.I)
        if m and m.start() < at:
            first, at = n, m.start()
    if first:
        return first
    # Nobody is named in that clause: the wearer is undressing themselves.
    return wearer or people[0]


def beat_stages_removal(beat, item, agent=""):
    """Does the BEAT already say this garment comes off, by this agent's hands?

    The clause exists to guarantee the removal FINISHES inside the shot -- the last
    frame is the next shot's keyframe, and a cut mid-removal hands on a garment
    still half worn. That guarantee is needed whether or not the beat stages it.

    But when the beat already says "McKenna takes off her shorts and steps out of
    them", the full clause repeats the whole action -- who, what, and that it comes
    off -- and the shot carries the same removal twice. Two statements of one action
    is an invitation to render it twice.

    True when the beat names the garment's head noun near a removal verb, and either
    names the agent or the beat has no other actor. The caller then says only the
    part the beat does NOT cover: that it is finished by the last frame.
    """
    b = str(beat or "")
    head = str(item or "").split()[-1] if item else ""
    if not b or not head:
        return False
    if not re.search(r"\b" + re.escape(head) + r"\b", b, re.I):
        return False
    # A removal verb in the same sentence as the garment.
    for part in re.split(r"(?<=[.;!?])\s+", b):
        if not re.search(r"\b" + re.escape(head) + r"\b", part, re.I):
            continue
        if not _REMOVAL_PROSE.search(part):
            continue
        # ...and not merely ASKED for: a request is not the act. See _in_a_request.
        m = _REMOVAL_PROSE.search(part)
        if m and _in_a_request(part, m.start()):
            continue
        if not agent:
            return True
        return bool(re.search(r"\b" + re.escape(agent) + r"\b", part, re.I)
                    # "she takes off her shorts" -- a pronoun for the only actor.
                    or re.search(r"\b(?:she|he|they)\b", part, re.I))
    return False


def scene_tag_for(head, scene):
    """The <Picture N> tag on the sheet entry whose head noun is `head`. "" if none.

    The tag lives INSIDE the wardrobe entry -- "chastity belt <Picture 2>" -- so
    scrubbing the entry when the garment comes off takes the picture with it. That
    is right for the description and wrong for the reference: the shot that takes a
    thing off is the shot it is handled in and most needs to look like itself, and
    without the tag it carries no image at all. Reported as the belt not matching
    its reference on the shot that removes it."""
    head = (head or "").strip().lower()
    if not head or not scene:
        return ""
    for line in str(scene).split("\n"):
        for item in re.split(r"[,;.]", line.split(":", 1)[-1]):
            m = re.search(r"<\s*picture\s+\d+\s*>", item, re.I)
            if not m:
                continue
            bare = re.sub(r"<\s*picture\s+\d+\s*>", " ", item, flags=re.I)
            bare = re.sub(r"\s+", " ", bare).strip()
            if bare and bare.split()[-1].lower() == head:
                return m.group(0)
    return ""


def off_by_last_frame(items, agent="", scene="", beat=""):
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
    named = []
    for i in items:
        nm = scene_name_for(i, scene) or i
        tag = scene_tag_for(i, scene)
        nm = f"{nm} {tag}" if tag else nm
        if nm not in named:
            named.append(nm)
    what = " and ".join(f"the {i}" for i in named)
    plural = len(named) > 1 or plural_item(named[-1])
    verb, are = ("come", "are") if plural else ("comes", "is")
    if beat and all(beat_stages_removal(beat, i, agent) for i in items):
        return (f" {what[0].upper()}{what[1:]} {verb} off during this shot and "
                f"{are} away by the last frame -- fully removed and clear of "
                f"the body.")
    if agent:
        sentence = (f"{agent} takes {what} off during this shot, with {agent}'s own "
                    f"hands, and {what} {are} away by the last frame -- fully removed "
                    f"and clear of the body.")
    else:
        sentence = (f"{what} {verb} off during this shot and {are} away by the last "
                    f"frame, fully removed and clear of the body, dropped out of "
                    f"frame.")
    bound = "Everything else worn stays exactly as it is, untouched and fastened."
    return " " + sentence[0].upper() + sentence[1:] + " " + bound


# Garments that are grammatically plural, so the sentence above agrees with them.
def plural_item(name):
    """Does this thing take a plural verb? "the boots ARE", "the belt IS".

    Was a list of the plural garments this file had thought of, anchored with \\b --
    which meant it only ever matched a WHOLE word, so "handcuffs" was not "cuffs"
    and the removal said "The handcuffs comes off during this shot and is away by
    the last frame". Written as the rule instead of the list, and the same rule
    restraint_sentence already uses: ends in s, and not in ss, so a dress and a
    harness stay singular. A reference tag is not part of the name."""
    w = re.sub(r"<[^>]*>", " ", str(name or "")).strip().rstrip(".").lower().split()
    last = w[-1] if w else ""
    return bool(last) and last.endswith("s") and not last.endswith("ss")


_A_DETERMINER = (r"(?<!\bthe\s)(?<!\bher\s)(?<!\bhis\s)(?<!\ba\s)(?<!\bmy\s)"
                 r"(?<!\bits\s)(?<!\btheir\s)(?<!\byour\s)(?<!\bthose\s)"
                 r"(?<!\bthese\s)(?<!\bsome\s)(?<!\bboth\s)"
                 r"(?<!\bof\s)(?<!\bpair\s)(?<!\bset\s)"
                 r"(?<!\btwo\s)(?<!\bthree\s)(?<!\bmore\s)(?<!\bseveral\s)")


_A_SURFACE = (r"(?:bench(?:es)?|tables?|desks?|counters?|worktops?|shel(?:f|ves)|"
              r"floors?|grounds?|chairs?|stools?|seats?|beds?|sofas?|couch(?:es)?|"
              r"hooks?|rails?|racks?|pegs?|hangers?|lines?|"
              r"box(?:es)?|crates?|baskets?|hampers?|bins?|trays?|drawers?|"
              r"cupboards?|cabinets?|ledges?|sills?|windowsills?|steps?|stairs?|"
              r"mats?|rugs?|carpets?|piles?|heaps?|stacks?|roofs?|bonnets?)")
_PUTS_ON = re.compile(
    r"\b(?:put(?:s|ting)?|pull(?:s|ing|ed)?|slip(?:s|ping|ped)?|tug(?:s|ging|ged)?|"
    r"draw(?:s|ing)?|drew|get(?:s|ting)?|got|climb(?:s|ing|ed)?|"
    r"step(?:s|ping|ped)?|wriggle(?:s|d)?)\b"
    r"(?:(?!\boff\b)[^.;!?]){0,40}?\b(?:back\s+on|back\s+into|on|into)\b"
    r"(?!\s+(?:the|a|an|her|his|their|its|that|this)?\s*(?:\w+\s+){0,1}?"
    + _A_SURFACE + r"\b)", re.I)
_DRESSES = re.compile(
    _A_DETERMINER
    + r"\b(?:dress(?:es|ing)|redress(?:es|ing)?|"
    r"button(?:s|ing)(?:\s+up)?|zip(?:s|ping)?\s+up|"
    r"fasten(?:s|ing)|laces?\s+up|puts?\s+back\s+on)\b", re.I)


def beat_stages_wearing(beat, item):
    """Does the BEAT say this garment goes ON during this shot?

    Only then is the both-ends clause right. `add:` has a second, older job -- it
    reveals a layer that was under something all along ("add: her white shirt
    underneath", after the jacket is cut off) -- and that garment was already worn.
    Telling the shot it goes on during these frames would stage a dressing that
    never happens, which is the same defect pointing the other way."""
    b = str(beat or "")
    if not b.strip():
        return False
    head = str(item or "").strip().lower()
    if not head:
        return False
    for pat in (_PUTS_ON, _DRESSES):
        for m in pat.finditer(b):
            window = b[max(0, m.start() - 60):min(len(b), m.end() + 60)]
            if re.search(r"\b" + re.escape(head.split()[-1]) + r"\b", window, re.I):
                return True
    return False


def wearing_clause(phrases):
    """Give putting something on BOTH ENDS: off as the shot opens, on by the last.

    The same shape direction_anchor uses for a door and removal_clause uses for a
    garment coming off. Phrased as where the garment IS at each end rather than as
    what it is not, because at cfg 1 there is no negative prompt and naming an
    unwanted state in the positive asks for it."""
    items = [str(p or "").strip().rstrip(".") for p in (phrases or []) if str(p or "").strip()]
    if not items:
        return ""
    what = " and ".join(items)
    plural = len(items) > 1 or plural_item(items[-1])
    are = "are" if plural else "is"
    return (f" {what[0].upper()}{what[1:]} {are} off the body as the shot opens and "
            f"fully on by the last frame, put on during this shot.")


RESTRAINT_HOLD_KEY = " ".join(dict.fromkeys(
    w for _p, _n, _pt in engine.HARDWARE
    for w in (_n, _n if _n.endswith("s") else _n + "s",
              _n[:-1] if _n.endswith("s") and not _n.endswith("ss") else _n,
              _n.split()[-1])
    if w))
MOUTH_HOLD = " Mouths in the shot stay closed, the expressions moving."


_MOUTH_WORKS = re.compile(
    r"\b(?:smil(?:e|es|ed|ing)|grin(?:s|ned|ning)?|smirk(?:s|ed|ing)?|"
    r"sneer(?:s|ed|ing)?|grimac(?:e|es|ed|ing)|pout(?:s|ed|ing)?|"
    r"yawn(?:s|ed|ing)?|gape(?:s|d|ing)?|chew(?:s|ed|ing)?|"
    r"kiss(?:es|ed|ing)?)\b"
    # Spitting needs somewhere to spit. Bare `spits` is what an engine does.
    r"|\bspits?\s+(?:it\s+)?(?:on|at|out|into|onto)\b"
    r"|\b(?:bite|bites|biting|bit)\s+(?:down\s+on\s+)?(?:her|his|their|the)\s+lips?\b"
    r"|\blick(?:s|ed|ing)?\s+(?:her|his|their|the)\s+lips\b"
    r"|\bpurs(?:e|es|ed|ing)\s+(?:her|his|their|the)\s+lips\b"
    r"|\bbar(?:e|es|ed|ing)\s+(?:her|his|their|the)\s+teeth\b"
    r"|\bmouth(?:s|ed|ing)?\s+(?:the\s+)?words?\b"
    r"|\b(?:her|his|their|the|[\w-]+['\u2019]s)\s+(?:mouth|jaw)\s+"
    r"(?:falls?|fell|drops?|dropped|hangs?|hung|opens?|opened)\b"
    r"|\b(?:her|his|their|the|[\w-]+['\u2019]s)\s+lips?\s+(?:parts?|parted)\b", re.I)


_DISTRESS = re.compile(
    r"\b(?:thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|writh(?:e|es|ing|ed)|"
    r"strain(?:s|ing|ed)?|squirm(?:s|ing|ed)?|kick(?:s|ing|ed)?|jerk(?:s|ing|ed)?|"
    r"sob(?:s|bing|bed)?|cr(?:y|ies|ying|ied)|weep(?:s|ing)?|"
    r"scream(?:s|ing|ed)?|shriek(?:s|ing|ed)?|whimper(?:s|ing|ed)?|"
    r"beg(?:s|ging|ged)?|plead(?:s|ing|ed)?|"
    r"flinch(?:es|ing|ed)?|winc(?:e|es|ing|ed)|recoil(?:s|ing|ed)?|"
    r"trembl(?:e|es|ing|ed)|shiver(?:s|ing|ed)?|panic(?:s|king|ked)?|"
    r"freak(?:s|ing)?\s+out)\b"
    r"|\b(?:goes|went|going)\s+limp\b", re.I)

DURESS_MOOD = " The mood is grim."
DURESS_FACE = " The mood is grim; the face shows the strain of it, the mouth set."


_OBJ_IS_THE_PERSON = (
    r"(?=\s*(?:[.,;:!?\"\u201d]|$)|\s+(?:into|out|off|from|down|up|towards?|to|"
    r"against|across|onto|through|back|away|and|by|in|on|over|behind|while|as|"
    r"before|after|until|so|but|with|without|aside|apart|hard|roughly|violently|"
    r"bodily|clear|free|upright|sideways|forward|backwards?)\b)")

_COERCION = re.compile(
    r"\b(?:grab(?:s|bed|bing)?|drag(?:s|ged|ging)?|forc(?:e|es|ed|ing)|"
    r"shov(?:e|es|ed|ing)|haul(?:s|ed|ing)?|bundl(?:e|es|ed|ing)|"
    r"seiz(?:e|es|ed|ing)|snatch(?:es|ed|ing)?|pin(?:s|ned|ning)?|"
    r"restrain(?:s|ed|ing)?|manhandl(?:e|es|ed|ing)|overpower(?:s|ed|ing)?|"
    r"subdu(?:e|es|ed|ing)|wrestl(?:e|es|ed|ing))\s+"
    r"(?:her|him|them|(?-i:[A-Z][\w-]+))" + _OBJ_IS_THE_PERSON
    # ...and the phrases that carry it without a bare transitive verb.
    + r"|\b(?:holds?|held|holding|pins?|pinned|forces?|forced)\s+"
    r"(?:her|him|them|(?-i:[A-Z][\w-]+))\s+(?:down|still|against|into|in)\b"
    r"|\bcover(?:s|ed|ing)?\s+(?:her|his|their|[\w-]+['\u2019]s)\s+mouth\b"

    r"|\bagainst\s+(?:her|his|their)\s+will\b"
    r"|\b(?:was|were|is|are|been|being|got)\s+(?:\w+\s+){0,2}?"
    r"(?:grabbed|dragged|forced|shoved|hauled|bundled|seized|snatched|pinned|"
    r"restrained|manhandled|overpowered|subdued|taken|carried|marched|walked|"
    r"loaded|bundled|driven|led)\s+"
    r"(?:from|into|out|off|away|down|to|in|through|aboard|across|onto|with)\b"
    # CAPTIVITY, which often has no verb of violence in it at all.
    r"|\bheld\s+(?:captive|prisoner|hostage)\b"
    r"|\b(?:captors?|hostages?|abduction|kidnapping)\b"
    r"|\b(?:held|taken|kept)\s+captive\b"
    r"|\blocked\s+(?:in|inside|up)\b"
    r"|\bkidnap(?:s|ped|ping)?\b|\babduct(?:s|ed|ing|ion)?\b|\bhostage\b"
    # Trying to get out is duress by definition.
    r"|\btr(?:y|ies|ied|ying)\s+to\s+(?:get\s+away|get\s+out|escape|run|pull\s+free)\b"
    r"|\b(?:break(?:s|ing)?|broke|pull(?:s|ed|ing)?)\s+free\b"
    r"|\bescap(?:e|es|ed|ing)\b", re.I)

_BINDABLE = (r"wrists?|ankles?|hands|feet|legs?|arms?|mouth|thumbs?|knees|elbows")
_TIE_TO = (r"chair|bed|bedframe|headboard|radiator|pipe|post|stake|banister|"
           r"bannister|frame|hook|ring|beam|column|tree")
_BINDING_ACT = re.compile(
    # ties her wrists, cuffs his ankles, gags her, tapes McKenna's mouth
    r"\b(?:ties?|tying|tied|bind(?:s|ing)?|bound|cuff(?:s|ed|ing)?|"
    r"shackl(?:e|es|ed|ing)|chain(?:s|ed|ing)?|zip-?ti(?:e|es|ed))\s+(?:up\s+)?"
    r"(?:her|his|their|(?-i:[A-Z][\w-]+)'s)\s+(?:" + _BINDABLE + r")\b"
    # taping is strapping unless it reaches a mouth, or wrists held together
    r"|\btap(?:e|es|ed|ing)\s+(?:up\s+)?(?:her|his|their|(?-i:[A-Z][\w-]+)'s)\s+"
    r"(?:mouth\b|(?:wrists?|ankles?|hands)\s+(?:together|behind|to)\b)"
    # wrists cable-tied, ankles taped -- the participle fragment
    r"|\b(?:" + _BINDABLE + r")\s+(?:\w+\s+){0,2}?"
    r"(?:tied|taped|cuffed|bound|chained|shackled|zip-?tied|strapped)\b"
    # tied TO the things people get tied to
    r"|\b(?:tied|cuffed|bound|shackled|chained|strapped|handcuffed)\s+"
    r"(?:her|him|them|(?-i:[A-Z][\w-]+)\s+)?(?:to|against)\s+"
    r"(?:the|a|an|that|this|his|her|their)\s+(?:" + _TIE_TO + r")\b"
    # hardware named as being ON a body
    r"|\b(?:handcuffs?|cuffs|rope|ropes|cord|cords|chains?|shackles|zip\s*ties?|"
    r"cable\s*ties?|duct\s*tape|tape|gag|blindfold)\s+(?:\w+\s+){0,2}?"
    r"(?:on|around|round|over|across|behind)\s+"
    r"(?:her|his|their|(?-i:[A-Z][\w-]+)'s|the)\s+(?:" + _BINDABLE + r"|head|eyes|face)\b"
    # A person gagged -- the person, not a smell he gagged at.
    r"|\bgag(?:s|ged|ging)\s+(?:her|him|them|(?-i:[A-Z][\w-]+))\b"
    r"|\b(?:a|the|another)\s+(?:bag|hood|sack|pillowcase)\s+over\s+"
    r"(?:her|his|their|(?-i:[A-Z][\w-]+)'s|the)\s+head\b"
    r"|\bblindfold(?:s|ed|ing)?\s+(?:her|him|them|(?-i:[A-Z][\w-]+))\b"
    r"|\b(?:she|he|they|(?-i:[A-Z][\w-]+))\s+(?:\w+\s+){0,2}?"
    r"(?:is|are|was|were|had\s+been|has\s+been|got)\s+(?:\w+\s+){0,2}?"
    r"(?:bound(?!\s+for\b)|gagged|cuffed|handcuffed|shackled)\b",
    re.I)


_DURESS_STRONG = re.compile(
    r"\bheld\s+(?:captive|prisoner|hostage)\b"
    r"|\b(?:captors?|hostages?|abduction|kidnapping)\b"
    r"|\b(?:held|taken|kept)\s+captive\b"
    r"|\bkidnap(?:s|ped|ping)?\b|\babduct(?:s|ed|ing|ion)?\b"
    r"|\blocked\s+(?:in|inside|up)\b"
    r"|\bagainst\s+(?:her|his|their)\s+will\b", re.I)


def beat_duress_strength(beat):
    """'' , 'weak' or 'strong'. See _DURESS_STRONG for why the grading exists."""
    b = beat or ""
    if _DURESS_STRONG.search(b) or _BINDING_ACT.search(b):
        return "strong"
    if _DISTRESS.search(b) or _COERCION.search(b):
        return "weak"
    return ""


def beat_stages_duress(beat, film_duress=True):
    """Does this BEAT stage duress?

    Strong evidence always counts. Weak evidence counts only where the FILM is
    already grim, because that is the context that says which meaning an ambiguous
    verb has. Defaults to True so a caller asking about a beat in isolation gets
    the old, generous reading."""
    strength = beat_duress_strength(beat)
    return strength == "strong" or (strength == "weak" and bool(film_duress))


_MOOD_GRIM = re.compile(
    r"\b(?:grim|bleak|tense|menacing|sinister|harrowing|distressing|brutal|"
    r"frightening|terrifying|desperate|oppressive|claustrophobic|ominous|"
    r"threatening|violent|grave|sombre|somber|dread|hostile|cruel|"
    r"kidnap(?:ping)?|abduction|captivity|hostage|abusive|coercive)\b", re.I)
_MOOD_LIGHT = re.compile(
    r"\b(?:warm|comic|comedy|cheerful|joyful|joyous|happy|light[-\s]?hearted|"
    r"playful|romantic|tender|sunny|upbeat|gentle|affectionate|celebratory|"
    r"whimsical|carefree|domestic\s+bliss|feel[-\s]?good)\b", re.I)


def mood_declared(anchor):
    """'grim', 'light' or '' -- what the ANCHOR says the film's tone is.

    Both directions matter. A film declared warm must never be handed a grim mood
    however its beats read, because that is the one reliable way out of a wrong
    inference; and a film declared grim needs no inference at all."""
    a = anchor or ""
    if _MOOD_LIGHT.search(a):
        return "light"
    if _MOOD_GRIM.search(a):
        return "grim"
    return ""


def film_stages_duress(beats, sheet="", anchor=""):
    """Does this FILM stage duress anywhere -- binding hardware, or a distress verb?

    Read once, over the whole script, because a shot of the captor alone is grim on
    account of what is on her wrists three beats ago. The same two signals the face
    clause uses, and the same refusals: a collar alone is not duress, and a film
    that stages neither is left alone in every shot. The node does not get to decide
    that somebody's film is bleak."""
    said = mood_declared(anchor)
    if said:
        return said == "grim"
    for _, ln in sheet_lines(sheet or ""):
        if _BOUND_HARDWARE.search(ln or ""):
            return True
    return any(beat_duress_strength(b) == "strong" for b in (beats or []))


_BOUND_HARDWARE = re.compile(
    r"\b(?:handcuffs?|cuffs?|shackles?|manacles?|irons|"
    r"ropes?|cords?|twine|zip\s*ties?|cable\s*ties?|"
    r"chains?|chained|tape|taped|gag|gagged|bound|tied|bindings?)\b", re.I)


def duress_face(beat, wearers, described, film_duress=False):
    """One short sentence about the face, on a shot whose scene already stages duress.

    IMPERSONAL, the choice gaze_hold already made and for the same reason: a named
    person is a person the model draws, and naming somebody twice in one shot is what
    put a second girl in frame at the moment of cuffing. On the shot where that could
    be ambiguous -- two people, one of them restrained -- the hardware hold has
    already said "Every restraint on Nora", so the shot is not short of an
    attribution. It is short of a sentence about her face."""
    _emotion = emotion_in(beat)
    if _emotion and described:
        if len(described) < 2:
            return mood_face(_emotion)
        _pairs = emotion_pairs(beat, described)
        return mood_faces(_pairs)
    if mouth_performs(beat):
        return ""
    if not described:
        return ""
    who = [n for n, ln in (wearers or []) if n and _BOUND_HARDWARE.search(ln or "")]
    if not who and beat_stages_duress(beat, film_duress):
        who = [n for n in (described or []) if n]
    if who:
        return DURESS_FACE
    return DURESS_MOOD if film_duress else ""


_EMOTION = re.compile(
    r"\b(?:happy|happily|happiness|delighted|delight(?:ed)?|thrilled|overjoyed|"
    r"joyful|joyous|elated|ecstatic|beaming|gleeful|glee|cheerful|cheery|"
    r"pleased|excited|excitement|grateful|relieved|relief|proud|smug|amused|"
    r"terrified|terror|frightened|afraid|scared|fearful|panicked|panicking|panic|"
    r"furious|fury|angry|angrily|anger|enraged|livid|seething|indignant|"
    r"desperate|desperation|distraught|devastated|grief|grieving|heartbroken|"
    r"miserable|wretched|ashamed|shame|humiliated|mortified|disgusted|horrified|"
    r"anguished|anguish|agony|bereft|despair(?:ing)?)\b", re.I)


_BEAMS_AT = re.compile(r"\b(?:she|he|they|[A-Z][a-z]+)\s+beams\b")


def emotion_in(beat):
    """The emotion this beat states, in the author's own word. "" when it states none."""
    text = str(beat or "")
    m = _EMOTION.search(text)
    if m:
        return m.group(0).lower()
    return "beaming" if _BEAMS_AT.search(text) else ""


def emotion_owner(beat, names, word):
    """Whose feeling it is: the person the beat puts in front of it. "" if nobody.

    The shape subjects_for uses, conjunction guard included, so "Dan holds the door
    and McKenna is terrified" does not hand the terror to Dan. Takes a name list
    rather than a sheet because the caller already has the shot's cast."""
    b = str(beat or "")
    for n in (names or []):
        if n and re.search(r"\b" + re.escape(n) + r"\b" + _UP_TO_TWO_WORDS
                           + r"\s+(?:is|was|looks?|looked|seems?|feels?|felt|sounds?|"
                             r"becomes?|became|goes|went|turns?|gets?|got)?\s*"
                           + re.escape(word) + r"\b", b, re.I):
            return n
    return ""


def emotion_pairs(beat, names):
    """[(who, feeling)] for the feelings this beat pins on people. Two at most.

    TWO PEOPLE CAN FEEL DIFFERENT THINGS IN ONE SHOT. "Dan is furious and McKenna is
    terrified" gave only the first of them, so one face was performing and the other
    was left to the prior -- the same half-fix as naming one of two speakers. Two at
    most, like the layering clause: a shot carrying four feelings has stopped being
    about its beat."""
    out, seen = [], set()
    for m in _EMOTION.finditer(str(beat or "")):
        word = m.group(0).lower()
        who = emotion_owner(beat, names, word)
        if who and who not in seen:
            seen.add(who)
            out.append((who, word))
        if len(out) >= 2:
            break
    return out


def mood_faces(pairs):
    """Say whose feeling is whose, for one or two people. "" for none."""
    ps = [(w, e) for w, e in (pairs or []) if w and e]
    if not ps:
        return ""
    if len(ps) == 1:
        return mood_face(ps[0][1], ps[0][0])
    return (f" {ps[0][0]}'s face carries {ps[0][1]} and {ps[1][0]}'s carries "
            f"{ps[1][1]}, each played in the eyes and the mouth.")


def mood_face(word, who=""):
    """Say the face plays the feeling the author named. "" when they named none.

    Their word, not a synonym: "terrified" and "grim" are not the same performance,
    and the generic one was replacing the specific one on every shot.

    NAMED ONCE A SECOND PERSON IS IN THE SHOT, which is the call gaze_hold already
    makes for the same reason. Said impersonally, "the face carries it: the expression
    is terrified" is a sentence about whoever is on screen -- so in a two-hander the
    captor wore his victim's terror. Reported as actions being performed by all the
    characters at once. With one person there is nobody else it could be, and naming
    them again is a second mention of a person, which has its own cost."""
    if not word:
        return ""
    if who:
        return (f" {who}'s face carries it: the expression is {word}, played in the "
                f"eyes and the mouth together.")
    return (f" The face carries it: the expression is {word}, played in the eyes and "
            f"the mouth together.")


def mouth_performs(beat):
    """Does the beat itself put the MOUTH to work?

    The beat has already said what the mouth does, so the guard has nothing to add
    over the top -- and what it was adding contradicted it. Same shape as the LONE
    vocal that sounds_for leaves alone: where the author wrote it, the node is
    quiet."""
    return bool(_MOUTH_WORKS.search(beat or ""))

_PERSON_WORD = re.compile(
    r"\b(?:he|she|they|him|her|hers|them|his|their|theirs|himself|herself|themselves|"
    r"nobody|somebody|anyone|everyone|man|woman|men|women|boy|girl|person|people|"
    r"figure|guard|driver|doctor|nurse|officer)\b", re.I)


def beat_puts_somebody_on_screen(beat, sheet=""):
    """Does the BEAT itself put a person in the shot?

    Deliberately not "is a person described in this shot's text": the character
    guard carries the previous shot's cast forward so a wordless beat does not empty
    the frame, and falls back to the sole sheet entry when there is no previous. So
    a scenery beat has a person described beside it before anybody has walked in,
    and reading that as "somebody is here" is what put a face in an empty yard."""
    b = beat or ""
    if _PERSON_WORD.search(b):
        return True
    return any(n and re.search(r"\b" + re.escape(n) + r"\b", b, re.I)
               for n, _ in sheet_lines(sheet))

FORM_HOLD = ", the same object in the same material."
OTHERS_UNCHANGED = " Everyone else in the shot has on exactly what their own entry lists."

RESTRAINT_GOING_ON = (" The hardware goes on during this shot: it is open and off the "
                      "body at the first frame, and closed on it by the last.")
RESTRAINT_ENDS_AT = " By the last frame the {part} are {where}, and stay there."
CHAIN_RIGID_TAIL = " Its links keep their size and the run between them stays taut."
_APPLY_NOW = re.compile(
    _A_DETERMINER +
    r"\b(?:cuffs|handcuffs|chains|ties|binds|locks|straps|tapes|gags|shackles|"
    r"fastens|secures|padlocks|buckles|clamps|clips|snaps|trusses|lashes|wraps|"
    r"restrains|immobili[sz]es|pinions|fetters|collars|hobbles|"
    r"hog-?ties|straitjackets|manacles|blindfolds|leashes|"
    r"cinches|tightens)\b", re.I)
_APPLY_PHRASE = re.compile(
    r"\b(?:put|puts|putting|pull|pulls|pulling|force|forces|forcing|get|gets|"
    r"getting|work|works|snap|snaps)\s+(?:[\w,']+\s+){0,4}?"
    r"(?:on|onto|around|behind|together|shut|closed)\b"
    r"|\b(?:loops?|wraps?|winds?|coils?|threads?|passes|runs|cinch(?:es)?|knots?|"
    r"laces?|hitch(?:es)?|slings?)\s+(?:[\w,']+\s+){0,5}?"
    r"(?:around|round|through|under|over|behind|between)\b", re.I)


_TAPE = r"(?:duct|gaffer|packing|masking|electrical|parcel)"
_HARDWARE_NOUN = re.compile(
    r"\b(?:(steel|metal|leather|nylon|plastic|padded|heavy|thin|black|chrome|"
    r"ball|ring|bit|rubber|canvas|webbing)\s+)?"
    r"(" + _TAPE + r"\s+tape\s+gags?|tape\s+gags?|" + _TAPE + r"\s+tape|tapes|tape|"
    r"handcuffs|cuffs|manacles|shackles|leg\s+irons|chains|chain|ropes|rope|cords|"
    r"cord|straps|strap|collars|collar|gags|gag|blindfolds|blindfold|"
    r"spreader\s+bars|spreader\s+bar|zip\s+ties|zip\s+tie|cable\s+ties|cable\s+tie)\b",
    re.I)


def hardware_named(text):
    """The hardware this text names, as written. '' when it names none."""
    best = ""
    for m in _HARDWARE_NOUN.finditer(text or ""):
        phrase = re.sub(r"\s+", " ", " ".join(g for g in m.groups() if g)).strip()
        if len(phrase) > len(best):
            best = phrase
    if not best:
        return ""
    item = best.lower()
    return "tape" if item == "tapes" else item


def hardware_all_named(text):
    """EVERY piece of hardware this text names, longest phrase per match, in order.

    hardware_named returns one item -- the most specific -- and the caller appended
    that single string to the worn list. So a beat that puts on two things at once,
    which is the ordinary way to write it:

        The guard handcuffs Ana's wrists behind her back and locks a steel collar
        around her neck, chained to the wall.

    recorded the collar and lost the handcuffs. From the next shot on, the cuffs
    were not named in the prompt at all -- not "stays fastened", not mentioned --
    and hardware nobody mentions is hardware the model stops drawing. Reported as
    her breaking out of the handcuffs, which is the model rendering exactly what it
    was told: a woman with a collar and free hands.

    The across-shots case was already fixed -- worn_item used to be overwritten by
    the next shot's item -- and the same bug within a single beat was left.
    """
    out = []
    for m in _HARDWARE_NOUN.finditer(text or ""):
        phrase = re.sub(r"\s+", " ", " ".join(g for g in m.groups() if g)).strip().lower()
        if phrase == "tapes":
            phrase = "tape"
        if not phrase:
            continue
        dupe = next((i for i, p in enumerate(out)
                     if p in phrase or phrase in p), None)
        if dupe is None:
            out.append(phrase)
        elif len(phrase) > len(out[dupe]):
            out[dupe] = phrase
    return out


_UNDO_NOW = re.compile(
    r"\b(?:unlocks?|unlocked|unlocking|uncuffs?|uncuffed|unbinds?|unbound|"
    r"unties?|untied|untying|unbuckles?|unbuckled|unstraps?|unstrapped|"
    r"unclips?|unclipped|unfastens?|unfastened|unshackles?|unshackled|"
    r"ungags?|ungagged|releases?|released|frees?|freed|cuts?\s+(?:off|away|free)|"
    r"slips?\s+off|takes?\s+off|pulls?\s+off|lifts?\s+(?:off|away))\b"
    r"[^.;!?]{0,40}?"
    r"\b(?:cuffs?|handcuffs?|chains?|ropes?|cords?|ties|straps?|tape|gags?|"
    r"collars?|shackles?|clamps?|clips?|restraints?|belt|them|it)\b", re.I)
# ...and the object-first form: "the cuffs come off", "the rope is untied".
_UNDO_PHRASE = re.compile(
    r"\b(?:cuffs?|handcuffs?|chains?|ropes?|cords?|ties|straps?|tape|gags?|"
    r"collars?|shackles?|clamps?|clips?|restraints?)\b\s+"
    r"(?:[\w,']+\s+){0,3}?"
    r"\b(?:come|comes|came|drop|drops|dropped|fall|falls|fell)\s+"
    r"(?:off|away|to\s+the\s+floor|to\s+the\s+ground)\b"
    r"|\b(?:is|are|was|were|gets?|got)\s+"
    r"(?:unlocked|untied|unbound|removed|taken\s+off|cut\s+(?:off|away|free))\b",
    re.I)


def restraint_words(line):
    """The restraint HARDWARE named in one sheet entry, as its own head nouns.

    Used to take hardware out of the sheet when a beat unlocks it: the hold can be
    cleared, but while the entry still lists the cuffs the next shot reads them back
    out of the scene text and latches the hold again."""
    out = []
    for item in re.split(r"[,;.]", str(line or "")):
        item = _LEADING_TAG.sub("", re.sub(r"\s+", " ", item)).strip()
        if not item:
            continue
        for _canon, _pt, _written, _at in engine.hardware_spans(item):
            for _w in (str(_written).split()[-1].lower(), str(_canon).split()[-1].lower()):
                if _w and _w not in out:
                    out.append(_w)
        head = item.split()[-1].lower().strip("-")
        if head and _RESTRAINT_WORD.match(head) and head not in out:
            out.append(head)
    return out


def restraint_coming_off(beat):
    """Does this beat stage hardware being TAKEN OFF, rather than merely mentioned?

    The hold latches, and it was cleared only by an explicit `remove:` naming the
    hardware -- deliberately, because a beat that does not mention cuffs is not a
    beat that removes them. But auto_remove never puts hardware in `toks` (restraint
    words are filtered out of infer_removals on purpose), so a script that unlocks
    the cuffs IN ITS PROSE and writes no remove: line never cleared the latch: the
    beat said they were unlocked and dropped to the floor, and every shot after went
    on insisting they stay closed and fastened. Reported as the hold still firing
    several shots after the hardware came off.

    Narrow, like the apply patterns it mirrors: an UNDOING verb with the hardware or
    a pronoun as its object. "She looks at the cuffs" or "the key is on the table"
    must not clear a restraint that is still on.
    """
    b = beat or ""
    return bool(_UNDO_NOW.search(b) or _UNDO_PHRASE.search(b))


def restraint_going_on(beat):
    """Does this beat stage hardware being APPLIED, rather than already worn?

    THE TENSE IS WHAT THIS ANSWERS, which is why it cannot simply defer to
    engine.applies_hardware. The engine reads "her wrists cuffed" and "Mara is
    handcuffed to the rail" as hardware going on, because for its purposes it is --
    it is recording what is on whom. Here the question is narrower: is this the shot
    where it CLOSES? Answering yes for a state that already holds puts "open and off
    the body at the first frame" on a woman who has been in cuffs for five shots.
    So the engine's vocabulary is mirrored in the -s forms above, deliberately, and
    test_smoke walks the two lists against each other."""
    b = beat or ""
    return bool(_APPLY_NOW.search(b) or _APPLY_PHRASE.search(b))


# Keep this compact: it is repeated in every shot while restraints remain present.
RESTRAINT_HOLD = (" Every restraint stays closed and fastened as it was put on") + FORM_HOLD


def restraint_wearers(sheet):
    """The people whose own sheet entry describes hardware.

    Read from the entries rather than the beat, because the entry is what says who is
    WEARING it -- a beat can mention a chain without anyone being in it."""
    return [n for n, ln in sheet_lines(sheet) if n and restraint_present(ln)]


GUARD_FLOOR_WORDS = 90
RESTRAINT_FLOOR_WORDS = 200
FALL_FLOOR_WORDS = 45
GUARD_WORDS_PER_BEAT_WORD = 5


def fit_guards(clauses, beat_words, floor=None):
    """(kept text, dropped names) for continuity clauses, ranked, within a budget.

    `clauses` is [(priority, name, text)] with 1 the most important. Order in the
    OUTPUT follows the list as given, not the priority -- the ranking decides what
    survives, not where it sits in the sentence.

    `floor` raises the minimum for a shot that cannot afford to lose what it is
    carrying. See RESTRAINT_FLOOR_WORDS.

    NO NAMING BUDGET, AND THE ATTEMPT IS WORTH RECORDING. Over-naming is what draws a
    duplicate -- a person named three times in one shot -- and every clause here pays
    a naming to say whose fact it is, so shedding the lowest-ranked clauses until
    nobody is over a cap looks like the obvious automation. It does not work, and the
    reason is structural rather than a matter of tuning.

    A clause names somebody for one of two reasons. Either the person is its SUBJECT
    -- "McKenna is lying down", "McKenna listens", "Only Kate speaks" -- and dropping
    it throws the fact away with the naming, which on a speaking shot is a face
    lip-syncing to a line it was never given. Or the clause is about the room, the
    take, the framing, the scene state, and mentions nobody at all. Measured on real
    shots: the second kind names NO ONE, so a pass restricted to them is a no-op, and
    a pass allowed past them costs a fact every time it fires. There is no clause
    that both names a person and is safe to drop.

    What the node's own over-naming note says is the way out, and it is not this
    function's to take: "a pronoun costs nothing". Replacing a name with a pronoun
    keeps the fact and spends no naming -- safe wherever the person is the only one
    of their gender in the shot, and ambiguous exactly where it is not. That is a
    rewrite of the clause, not a choice of which to keep."""
    budget = max(int(floor or GUARD_FLOOR_WORDS), GUARD_FLOOR_WORDS,
                 int(beat_words) * GUARD_WORDS_PER_BEAT_WORD)
    spent, keep = 0, set()
    for _, name, text in sorted(clauses, key=lambda c: c[0]):
        if not text:
            continue
        cost = len(text.split())
        if spent + cost > budget and spent > 0:
            continue
        spent += cost
        keep.add(name)
    kept = "".join(t for _, n, t in clauses if n in keep and t)
    dropped = [n for _, n, t in clauses if t and n not in keep]
    return kept, dropped


_EXTRA_WORDS = (r"crowds?|groups?|others|onlookers|bystanders|passers-?by|spectators|"
                r"people|dancers|guests|customers|patrons|strangers|students|staff|"
                r"tourists|girls|women|men|boys|guys|ladies|blondes|brunettes|"
                r"figures|silhouettes")
_EXTRA_PEOPLE = re.compile(r"\b(?:" + _EXTRA_WORDS + r")\b", re.I)


_NOT_STAGED = re.compile(
    r"\b(?:gone|left|leaving|went|departed|vanished|absent|empty|alone|"
    r"outside|elsewhere|away|upstairs|downstairs|next\s+door|beyond|"
    r"no\s+one|no[- ]?body|none|without|hears?|heard|hearing|listens?|"
    r"remembers?|imagines?|thinks?\s+of|expects?|waits?\s+for|"
    r"ledgers?|invoices?|accounts|spreadsheets?|columns?|receipts?|payroll|"
    r"balance\s+sheets?|paperwork)\b", re.I)


_A_PLACE = (r"(?:room|rooms|yard|street|road|hall|hallway|corridor|house|flat|"
            r"place|space|building|shop|store|bar|cafe|kitchen|office|garage|"
            r"platform|station|carriage|car\s*park|lot|field|beach|park|"
            r"church|theatre|theater|hangar|warehouse|workshop|studio|"
            r"landing|stairwell|lobby|foyer|courtyard|square|market|"
            r"pool|deck|garden|barn|shed|cell|ward|dorm|gym|pitch|court)")
_ALONE = re.compile(
    r"\b(?:alone|by\s+(?:her|him|them)self|on\s+(?:her|his|their)\s+own|"
    r"deserted|to\s+(?:her|him|them)self)\b"
    r"|\bempty\s+" + _A_PLACE + r"\b"
    r"|\b(?:the\s+)?" + _A_PLACE + r"\s+(?:is|was|looks|stands|sits|feels|lies)\s+"
    r"(?:quite\s+|completely\s+|totally\s+)?empty\b"
    r"|\b(?:it|everything|everywhere|the\s+place)\s+(?:is|was)\s+empty\b", re.I)


_PEOPLE_MODIFIER = re.compile(
    r"\b(?:" + _EXTRA_WORDS + r")"
    r"(?:['’]s?(?!\w)"
    r"|\s+(?:rooms?|areas?|sections?|quarters|entrances?|exits?|canteens?|"
    r"kitchens?|lounges?|toilets?|washrooms?|lockers?|cloakrooms?|"
    r"car\s*parks?|carriers?|lists?|registers?|ledgers?|books?|"
    r"invoices?|records?|files?|accounts?|columns?|reports?|receipts?|"
    r"departments?|desks?|meetings?|notices?|boards?|rotas?|shifts?|"
    r"uniforms?|overalls|clothing|clothes|wear|shoes|aisles?|"
    r"unions?|clubs?|nights?|members?|badges?|handbooks?|policy|policies)\b)", re.I)


def extras_in(beat):
    """Does this beat stage people beyond the ones the sheet names, IN the frame?"""
    b = str(beat or "")
    hits = list(_EXTRA_PEOPLE.finditer(b))
    if not hits:
        return False
    if all(_PEOPLE_MODIFIER.match(b, m.start()) for m in hits):
        return False
    return not _NOT_STAGED.search(b)


def extras_dismissed(beat):
    """Does this beat say the people the sheet does not name are no longer there?"""
    b = str(beat or "")
    if _ALONE.search(b):
        return True
    return bool(_EXTRA_PEOPLE.search(b) and _NOT_STAGED.search(b))


_CONTACT_SRC = (
    r"kiss(?:es|ed|ing)?|hug(?:s|ged|ging)?|embrac(?:e|es|ed|ing)|"
    r"straddl(?:e|es|ed|ing)|mount(?:s|ed|ing)?|caress(?:es|ed|ing)?|"
    r"strok(?:es|ed|ing)?|cuddl(?:e|es|ed|ing)|hold(?:s|ing)?|held|"
    r"grab(?:s|bed|bing)?|touch(?:es|ed|ing)?|caught|catch(?:es|ing)?|"
    r"pull(?:s|ed|ing)?|take[sn]?|took|taking|push(?:es|ed|ing)?|"
    r"danc(?:e|es|ed|ing)\s+with|lean(?:s|ed|ing)?\s+(?:on|against|into)|"
    r"press(?:es|ed|ing)?\s+(?:against|into)|sit(?:s|ting)?\s+on|"
    r"wraps?\s+(?:her|his|their)\s+arms?\s+around|"
    r"reach(?:es|ed|ing)?\s+for|undress(?:es|ed|ing)?")
_CONTACT_SPLIT = re.compile(r"(?<=[.;!?])\s+|\s+\b(?:while|as|and|then)\b\s+|,\s+", re.I)


def contact_pairs(beat, names):
    """[(who, whom)] the beat puts in physical contact. Two at most.

    Two, like the layering clause: a shot restating four pairings has stopped being
    about its beat."""
    b = _DIALOGUE_TAG.sub(" ", _QUOTED.sub(" ", str(beat or "")))
    people = [n for n in (names or []) if n]
    out = []
    for part in _CONTACT_SPLIT.split(b):
        for a in people:
            m = re.search(r"\b" + re.escape(a) + r"\b\s+(?:\w+\s+){0,2}?(?:"
                          + _CONTACT_SRC + r")\b", part, re.I)
            if not m:
                continue
            tail = part[m.end():]
            for c in people:
                if c == a:
                    continue
                if re.match(r"\W{0,14}(?:the\s+)?" + re.escape(c) + r"\b", tail, re.I):
                    if not any({a, c} == set(p) for p in out):
                        out.append((a, c))
                    break
        if len(out) >= 2:
            break
    return out[:2]


def contact_hold(pairs):
    """Say which body is with which. "" when the beat pairs nobody.

    Positive, like every other clause here: it says what the pairing IS, never that
    anybody is not paired. Naming both sides is the point -- an unnamed "they kiss" in
    a shot with four people is the sentence that let the model choose."""
    ps = [(a, b) for a, b in (pairs or []) if a and b]
    if not ps:
        return ""
    if len(ps) == 1:
        return f" The contact is {ps[0][0]} with {ps[0][1]}: those two bodies together."
    return (f" The contact is {ps[0][0]} with {ps[0][1]}, and {ps[1][0]} with "
            f"{ps[1][1]}: two pairs, each body with its own partner.")


def lora_facts(patcher):
    """(stacked LoRAs, weights touched, [strengths]) for a model or a CLIP patcher."""
    patches = getattr(patcher, "patches", None)
    if not isinstance(patches, dict) or not patches:
        return (0, 0, [])
    stacked = max((len(v) for v in patches.values() if isinstance(v, (list, tuple))), default=0)
    strengths = []
    for entries in patches.values():
        for entry in entries if isinstance(entries, (list, tuple)) else []:
            try:
                value = round(float(entry[0]), 3)
            except (TypeError, ValueError, IndexError):
                continue
            if value not in strengths:
                strengths.append(value)
    return (stacked, len(patches), sorted(strengths, reverse=True))


def lora_name_of(patcher):
    """The last-applied LoRA's own name, if its metadata carries one."""
    meta = getattr(patcher, "attachments", {}) or {}
    meta = meta.get("lora_metadata") if isinstance(meta, dict) else None
    if not isinstance(meta, dict):
        return ""
    for key in ("modelspec.title", "ss_output_name", "ss_session_id"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    return ""


_PRONOUN_FORMS = {"she":  {"subject": "she",  "object": "her",  "possessive": "her"},
                  "he":   {"subject": "he",   "object": "him",  "possessive": "his"},
                  "they": {"subject": "they", "object": "them", "possessive": "their"}}

# A name straight after one of these is the OBJECT of it -- "looks at Mara", "beside
# Mara" -- and takes the object form. Only prepositions are listed. Verbs would catch
# more ("watches Mara") and cost more when wrong, and the subject form is the safe
# default: "she is lying down" reads as intended where "her is lying down" does not.
_TAKES_OBJECT = frozenset((
    "at", "to", "with", "behind", "beside", "on", "of", "for", "from", "near", "over",
    "under", "against", "between", "into", "onto", "around", "toward", "towards",
    "past", "beneath", "above", "below", "across", "alongside", "opposite", "facing",
    "beyond", "through", "along", "before", "after", "inside", "outside", "upon"))


def pronoun_rewrite(text, people, extras=False):
    """(text, [(name, swaps)]) with REPEATED namings in the node's own clauses
    turned into pronouns.

    Naming somebody three times in one shot is what draws a second copy of them, and
    every clause that owns a fact pays a naming to say whose fact it is. Dropping the
    clause to save the naming does not work -- the fact goes with it, and fit_guards
    records why -- so the naming is spent differently instead. This is the node taking
    its own advice: the over-naming report has always ended "a pronoun costs nothing".

    THE AUTHOR'S WORDS ARE NEVER TOUCHED. Only the clause text this node wrote is
    rewritten, and the FIRST naming in it survives: something has to say whose fact it
    is before a pronoun can point back at it. What goes is the second and the third.

    SAFE ONLY WHERE THE PRONOUN RESOLVES, which is the whole of the restraint here.
    "She is lying down" in a shot with two women is not a saving, it is the ambiguity
    the naming existed to prevent -- and unresolved_pronouns() already refuses to guess
    on the author's behalf for exactly this reason. So a name is rewritten only when
    nobody else in the shot declares the same pronoun, and nothing is rewritten at all
    on a shot staging EXTRAS, where the people who could answer to "she" are not on the
    sheet to be counted.

    Form follows position: "Mara's hands" takes the possessive, a name straight after
    a preposition takes the object form, everything else takes the subject form, and a
    name that opened a sentence hands its capital to the pronoun.

    A NAME IN PREDICATE-POSSESSIVE POSITION IS LEFT ALONE -- "the sobbing is Mara's",
    with nothing after the 's. Two reasons, and either would do. The form there is the
    absolute possessive, "hers", not the determiner "her", and "the sobbing is her" is
    not English. And that clause is the attribution itself: it exists to say whose
    vocal this is so that nobody else's mouth is opened for it, which is the one place
    a name is doing work no pronoun can take over."""
    rows = [(n, g) for n, g in (people or ()) if n and g in _PRONOUN_FORMS]
    if extras or not text or not rows:
        return text, []
    sole = {}
    for _, g in rows:
        sole[g] = sole.get(g, 0) + 1
    swapped = []
    for name, group in rows:
        if sole[group] != 1:
            continue
        forms = _PRONOUN_FORMS[group]
        hits = list(re.finditer(r"\b" + re.escape(name) + r"\b('s\b)?", text))
        if len(hits) < 2:
            continue
        out, last, done = [], 0, 0
        for hit in hits[1:]:                  # the first naming stays a name
            before = text[:hit.start()]
            lead = re.search(r"(\w+)\W*$", before)
            if hit.group(1):
                if not re.match(r"\s+\w", text[hit.end():]):
                    continue              # predicate possessive: the attribution itself
                word = forms["possessive"]
            elif lead is not None and lead.group(1).lower() in _TAKES_OBJECT:
                word = forms["object"]
            else:
                word = forms["subject"]
            if re.search(r"(?:^|[.!?])[\s\"']*$", before):
                word = word[:1].upper() + word[1:]
            out.append(text[last:hit.start()])
            out.append(word)
            last = hit.end()
            done += 1
        out.append(text[last:])
        text = "".join(out)
        if done:
            swapped.append((name, done))
    return text, swapped


def cast_hold(names, beat="", extras=False):
    """A positive body-count constraint for a one- or two-person composition.

    THE SOLO SHOT HAD NO COUNT AT ALL, and a solo shot is where a duplicate of the
    one person in it has nothing standing against it. The pair case has been asserted
    since this function was written; the one-person case returned "" with no recorded
    reason for it, so the shot most at risk of being rendered as twins was the shot
    that said nothing about how many bodies were in it. Reported, repeatedly, as
    duplicate characters.

    STANDS DOWN WHERE THE BEAT STAGES EXTRAS. "There are two people in the shot,
    with one body for each person" is exactly right against a duplicated character
    and exactly wrong against "two women dance behind them": it forbids, as a
    positive fact, the people the author just asked for. Reported as extras refusing
    to appear. The author's words outrank anything inferred from them, which is this
    file's standing rule, so a beat that puts more bodies in the frame keeps them and
    the count goes unsaid."""
    people = list(dict.fromkeys(n for n in (names or []) if n))
    if extras or extras_in(beat):
        return ""
    if len(people) == 1:
        return " There is one person in the shot: one body, one face."
    if len(people) == 2:
        return " There are two people in the shot, with one body for each person."
    return ""


def restrained_by_beat(beat, cast):
    """Who this beat puts in the hardware. The agent is not the one wearing it.

    `restrained` was a film-level latch: once anything was on anybody, every later
    shot got the hold. So a shot describing only the man who applied it was told
    there were cuffs holding wrists behind a back -- with nobody in the text those
    wrists could belong to. The model has to draw the person the sentence describes,
    so it invents one. That is the duplicate.

    One person in the shot is the one wearing it. Two or more and the first named is
    the one doing it, which is how these beats are written: "Dan walks in and cuffs
    her wrists"."""
    people = [n for n in (cast or []) if n]
    if len(people) <= 1:
        return set(people)
    b = _outside_speech(beat or "")
    verb = None
    for pat in (_APPLY_NOW, _APPLY_PHRASE):
        for m in pat.finditer(b):
            verb = m.start() if verb is None else min(verb, m.start())
    if verb is None:
        return set(people)
    agent, at = None, -1
    for n in people:
        for m in re.finditer(r"\b" + re.escape(n) + r"\b", b, re.I):
            if at < m.start() < verb:
                agent, at = n, m.start()
    return {n for n in people if n != agent} if agent else set(people)


_HELD_PART = (
    (r"\b(?:collars?|leash(?:es)?|leads?|chokers?|neck\s*(?:chain|iron)s?)\b", "neck"),
    (r"\b(?:leg\s*irons?|ankle\s*(?:cuffs?|chains?|straps?)|hobbles?|"
     r"shackles?)\b", "ankles"),
    (r"\b(?:harness(?:es)?|body\s*belts?)\b", "body"),
    (r"\b(?:waist\s*(?:chain|belt)s?)\b", "waist"),
)


def held_part(items):
    """The body part an anchored restraint holds, read from the hardware itself."""
    text = " ".join(items or [])
    for pat, part in _HELD_PART:
        if re.search(pat, text, re.I):
            return part
    return "wrists"          # cuffs, rope and tape, which is the common case


_POSE_OF_POSITION = {
    "behind the back": ("Both arms are behind the body, wrists together at the "
                        "small of the back"),
    "above the head": ("Both arms are raised, wrists together above the head, "
                       "the body stretched long"),
    "in front of the body": ("Both arms are in front of the body, wrists "
                             "together at the waist"),
    "out to the sides": ("Both arms are held out level with the shoulders, one "
                         "hand to each side"),
    "at the waist": "Both arms are at the sides, wrists together at the waist",
}


POSE_LYING_WEIGHT = "The shoulder and the hip take the weight of the body"


_LEG_WORD = r"(?:ankles?|legs?|feet|knees?|thighs?|calves)"
_LEG_FASTEN = (r"(?:cuffed|shackled|chained|tied|bound|strapped|secured|fastened|"
               r"locked|linked|clipped|hooked|lashed|drawn|pulled|folded|bent|"
               r"looped|wrapped|wound|coiled|threaded|passed|slung|knotted|cinched)")
_LEG_TIE = r"(?:" + _LEG_FASTEN + r"|around|round)"
_LEG_JOIN = r"(?:" + _LEG_FASTEN + r"|around|round|from|to|down\s+to|up\s+to)"
_LEG_ANCHOR = (
    # A HOGTIE, by its name or by what it does: the ankles held to the wrists.
    (r"\bhog-?(?:tie|ties|tied|tying|cuff|cuffs|cuffed|chains?|chained|bound)\b"
     r"|\btruss(?:es|ed|ing)?\s+(?:\w+\s+){0,2}?up\b|\btrussed\b"
     r"|" + _LEG_WORD + r"\s+(?:\w+\s+){0,4}?" + _LEG_TIE + r"\s+(?:\w+\s+){0,3}?"
     r"to\s+(?:her|his|their|the)\s+(?:wrists?|hands?|arms?|cuffs?)"
     r"|(?:wrists?|hands?|cuffs?)\s+(?:\w+\s+){0,4}?" + _LEG_TIE +
     r"\s+(?:\w+\s+){0,3}?to\s+(?:her|his|their|the)\s+" + _LEG_WORD,
     "ankles to the wrists"),
    # Held apart, which is what a bar between them is for.
    (r"\bspreader\s+bars?\b"
     r"|" + _LEG_WORD + r"\s+(?:\w+\s+){0,3}?(?:held\s+)?(?:apart|spread\s+(?:wide|apart))"
     r"|" + _LEG_TIE + r"\s+(?:\w+\s+){0,2}?" + _LEG_WORD + r"\s+(?:\w+\s+){0,2}?apart",
     "held apart"),
    (_LEG_WORD + r"\s+(?:\w+\s+){0,3}?" + _LEG_TIE + r"\s+(?:\w+\s+){0,2}?together"
     r"|" + _LEG_TIE + r"\s+" + _LEG_WORD + r"\s+together"
     r"|" + _LEG_WORD + r"\s+crossed\s+and\s+" + _LEG_TIE,
     "ankles together"),
    (_LEG_WORD + r"\s+(?:\w+\s+){0,2}?(?:together|crossed)\b", "ankles together", True),
    (r"(?:neck|throat)\b[^.]{0,40}?" + _LEG_JOIN + r"[^.]{0,30}?" + _LEG_WORD
     + r"|" + _LEG_WORD + r"\b[^.]{0,40}?" + _LEG_JOIN + r"[^.]{0,30}?(?:neck|throat)",
     "ankles to the neck"),
    (r"(?:" + _LEG_TIE + r")\s+(?:\w+\s+){0,2}?(?:her|his|their|the)\s+"
     r"(?:\w+\s+){0,2}?" + _LEG_WORD, "ankles together"),
    # ...or back under the body, which is the kneeling half of a hogtie.
    (_LEG_WORD + r"\s+(?:\w+\s+){0,3}?" + _LEG_TIE + r"\s+(?:\w+\s+){0,2}?"
     r"(?:(?:back|up)\s+)?behind\s+(?:her|his|their)\b",
     "drawn back"),
)
_POSE_OF_LEGS = {
    "ankles to the wrists": ("Both legs are bent back at the knee, the ankles drawn up "
                             "behind the body and held there with the wrists, the feet "
                             "off the floor and the knees taking the weight"),
    "held apart": ("Both legs are held apart at the ankle, each foot fixed where it is, "
                   "the gap between them the same from the first frame to the last"),
    "ankles together": "Both ankles are together, fastened one against the other",
    "ankles to the neck": ("Both legs are bent back at the knee, the ankles drawn up "
                           "behind the body and held there by the line running to the "
                           "neck, the feet off the floor and the knees bent"),
    "drawn back": ("Both legs are folded back under the body, the ankles behind and the "
                   "knees bent double"),
}


_FASTENING_NEAR = re.compile(
    _LEG_FASTEN + r"|\b(?:" + "|".join(p for p, _n, _pt in engine.HARDWARE) + r")\b",
    re.I)


def legs_anchor(text):
    """Where fastened LEGS are being held, as a phrase. '' when the text says none."""
    body = text or ""
    for entry in _LEG_ANCHOR:
        pat, phrase = entry[0], entry[1]
        weak = len(entry) > 2 and entry[2]
        if not re.search(pat, body, re.I):
            continue
        if weak and not _FASTENING_NEAR.search(body):
            continue
        return phrase
    return ""


def pose_clause(position, lying=False, legs=""):
    """One sentence describing the BODY a limb position makes. "" when unknown.

    `lying` adds what is under it -- see POSE_LYING_WEIGHT. `legs` adds where the
    legs are held, which is a second fact and not an alternative: a hogtie has its
    arms behind the back AND its ankles drawn to them, and the one this file knew
    how to say was the arms."""
    key = str(position or "").strip().lower()
    said = _POSE_OF_POSITION.get(key, "")
    legs_said = _POSE_OF_LEGS.get(str(legs or "").strip().lower(), "")
    if not said and not legs_said:
        return ""
    if said and lying and key == "behind the back":
        said = f"{said}. {POSE_LYING_WEIGHT}"
    return "".join(f" {part}." for part in (said, legs_said) if part)


def merge_hardware_names(items):
    """One name per piece of hardware, keeping the fullest wording of each.

    Substring-aware, because the beats name the same thing differently from shot to
    shot: "handcuffs" in shot 1 and "the cuffs" in shot 4 is ONE pair of handcuffs,
    and an exact-match check listed both -- "The handcuffs, steel collar, chain and
    cuffs stay closed", which reads as four things and invites the model to draw a
    spare set."""
    out = []
    for it in (items or []):
        it = str(it or "").strip()
        if not it:
            continue
        same = next((k for k, p in enumerate(out) if p in it or it in p), None)
        if same is None:
            out.append(it)
        elif len(it) > len(out[same]):
            out[same] = it
    return out


def restraint_sentence(item, wearers, described, anchor="", rigid=False, posed=False,
                       part=""):
    """ONE sentence for the hardware: what it is, that it is closed, and where it holds.

    These used to be three, written at three different times for three different bug
    reports, and each of them names the same object again:

        Every restraint stays closed and fastened as it was put on, ... (29 w)
        The cuffs are still on her, in plain sight where they were put. (13 w)
        The fastened wrists stay behind the back, where they were locked. (11 w)

    53 words about one pair of handcuffs, beside a nine-word beat. Measured on a real
    scene the guards had reached 65% of the shot against a 12% beat -- the number this
    node was rebuilt to escape, arrived at again by adding a clause per report with no
    budget on the total. Merged, the same facts cost 25.

    Every guarantee survives: the thing is named so it gets drawn, it is closed, it is
    the same object in the same material, and it is where it was fastened."""
    items = [i.strip() for i in (item or "").split(",") if i.strip()]
    if len(items) > 1:
        item = ", ".join(items[:-1]) + " and " + items[-1]
        plural = True
    else:
        plural = bool(item) and item.endswith("s") and not item.endswith("ss")
    who = ""
    if wearers and len(described) >= 2:
        who = (wearers[0] if len(wearers) == 1
               else ", ".join(wearers[:-1]) + " and " + wearers[-1])
    if item:
        subject = f"The {item} on {who}" if who else f"The {item}"
        verb = "stay" if plural else "stays"
    else:
        subject = f"Every restraint on {who}" if who else "Every restraint"
        verb = "stays"
    it, was = ("they", "were") if plural else ("it", "was")
    _soft_word = re.compile(r"\b(?:rope|ropes|cord|cords|twine|string|strap|straps|"
                            r"tape|scarf|belt|stocking|stockings|zip\s*ties?|"
                            r"cable\s*ties?|laces?)\b", re.I)
    soft = bool(items) and all(_soft_word.search(i) for i in items)
    shut = "tied and holding as" if soft else "closed and fastened as"
    out = f" {subject} {verb} {shut} {it} {was} put on"
    if anchor:
        _m = re.match(r"^(.*?),?\s*(at the .+)$", anchor)
        _pos, _point = (_m.group(1).strip(), _m.group(2)) if _m else (anchor, "")
        _part = part or held_part(items)
        if _point:
            out += f", holding the {_part} fast {_point}"
        elif not _pos:
            out += f", holding the {_part}"
    if posed:
        _stuff = ("the metal" if (rigid or (item and rigid_hardware(item)))
                  else "it" if not plural else "they")
        _drawn = "is" if _stuff != "they" else "are"
        out += (f"; {_stuff} {_drawn} already drawn to {'their' if _stuff == 'they' else 'its'} "
                "full length, so the position it fixes is the position that keeps, "
                "and the body strains against it while the fastenings hold")
    elif rigid:
        out += (f", {'their' if plural else 'its'} links keeping their size and the run "
                f"between them taut")
    out += FORM_HOLD
    if who:
        out += OTHERS_UNCHANGED
    return out


def own_body(clause, who, described):
    """Say WHOSE body a bare-skin clause is about, when more than one is described.

    "Everything worn comes off during this shot" and "The legs are bare from the
    hip down" name nobody. With one person in the shot that is unambiguous; with
    two it is an instruction about whoever is on screen, and the second character
    undresses alongside the first. Reported as one character mimicking the other's
    actions -- and it is the same defect own_hold was written for, in the clause
    next door.

    Positively phrased, like own_hold: naming whose body it is excludes everyone
    else, where "nobody else undresses" asks the model to render an absence. The
    other people are pinned to their own entries in one short sentence rather than
    named individually, which costs a second mention of each."""
    if not clause or not who or len(described or []) < 2:
        return clause
    names = [n for n in (who if isinstance(who, (list, tuple)) else [who]) if n]
    if not names:
        return clause
    subject = names[0] if len(names) == 1 else \
        ", ".join(names[:-1]) + " and " + names[-1]
    body = clause.strip()
    body = re.sub(r"^The\s+", f"{subject}'s ", body)
    body = re.sub(r"^Everything worn\b", f"Everything {subject} is wearing", body)
    return (" " + body
            + " Everyone else in the shot keeps on exactly what their own entry "
              "lists.")


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
    tail = OTHERS_UNCHANGED
    return hold.replace("Every restraint", f"Every restraint on {who}", 1).rstrip() + tail

# Hardware that means restraint on its own.
_RESTRAINT_PLAIN = re.compile(
    r"\b(?:(?:leg|ankle|wrist)\s?irons?|tethers?|spreader\s+bars?|hobbles?|"
    r"(?:braided\s+)?(?:steel|wire)\s+cables?|(?:bike|bicycle)\s+locks?|[ud]-?locks?|"
    r"cling\s?film|plastic\s+wrap|straitjackets?|leash(?:es)?|"
    r"hog-?(?:tie|ties|tying|cuffs|cuffing)|truss(?:es|ing)|hobbl(?:es|ing)|"
    r"zip[-\s]?(?:ties?|tied|tying)|cable[-\s]?(?:ties?|tied|tying)|"
    r"handcuff(?:s|ed|ing)?|cuffed|shackle[sd]?|manacle[sd]?|hogtied|hog-?tied|"
    r"hogcuffed|hog-?cuffed|gag(?:ged|s)?|blindfold(?:ed|s)?|zip[- ]ties?|"
    r"cable[- ]ties?|restrain(?:t|ts|ed)|bound|bindings?|straitjacket|"
    r"collared|leashed|tethered|manacled|fettered|chained\s+up|hobbled|"
    r"restrain(?:s|ing)|immobili[sz](?:e|es|ed|ing)|pinion(?:s|ed|ing)|fetters|"
    + _A_DETERMINER + r"collars|"
    r"(?:steel|iron|metal|chrome|brass|leather|padded|locked|lockable|heavy|"
    r"thick|studded|spiked|posture|shock|bondage|slave)\s+collars?|"
    r"collars?\s+(?:and|with)\s+(?:a\s+)?(?:lock|padlock|leash|lead|chain|ring)|"
    r"spreader bar)\b", re.I)
_RESTRAINT_MAYBE = re.compile(
    r"\b(?:chains?|ropes?|cords?|cuffs?|straps?|collars?|tapes?|taped|taping|"
    r"twine|chokers?|harness(?:es)?|(?:steel|baling)\s+wires?|"
    r"belts?|hobble|clamps?|clips?)\b", re.I)
_HANDLING_VERB = re.compile(
    r"\b(?:drops?|dropped|dropping|throws?|threw|thrown|throwing|tosses|tossed|"
    r"tossing|kicks?|kicked|kicking|carries|carried|carrying|"
    r"picks?\s+up|picked\s+up|picking\s+up|puts?\s+(?:it|them|the\s+\w+\s+)?"
    r"(?:down|away|back)|sets?\s+(?:it|them)?\s*down|lays?\s+(?:it|them)?\s*down|"
    r"pockets?|pocketed|stows?|stowed|packs?\s+(?:up|away)|hangs?\s+up)\b", re.I)
_SHOWN_VERB = re.compile(
    r"\b(?:holds?\s+up|held\s+up|holding\s+up|shows?|showed|showing|"
    r"lifts?|lifted|lifting|dangles?|dangled|dangling|"
    r"weighs?\s+(?:it|them)|turns?\s+(?:it|them)\s+over|"
    r"inspects?|inspecting|examines?|examining)\b", re.I)
_BINDING_VERB = re.compile(
    r"\b(?:cuffed|chained|tied|tying|bound|binds?|binding|locked|locks|"
    r"strapped|taped|taping|gagged|shackled|fastened|fastens|secured|secures|"
    r"padlocked|trussed|lashed|wrapped|clamped|clamping|clipped|clipping|"
    r"pinned|attached|affixed)\b", re.I)


TURN_HOLD = (" What is on the body now is all that is on it, front, side and behind, and "
             "whatever is fastened stays fastened and closed as the view comes round.")

_TURN_CUE = re.compile(
    r"\b(?:turn(?:s|ed|ing)?|rotat(?:es?|ed|ing)|spin(?:s|ning)?|swivel(?:s|led)?|"
    r"roll(?:s|ed|ing)?\s+(?:over|onto)|faces?\s+away|face[sd]?\s+the\s+other|"
    r"over\s+(?:her|his|their)\s+shoulder|from\s+behind|back\s+to\s+the\s+camera|"
    r"shows?\s+(?:her|his|their)\s+back|other\s+side)\b", re.I)


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
                     r"(?!\s*['’]s)"
                     r"(?!\s+(?:legs?|arms?|wrists?|ankles?|hands?|feet|foot|head|hair|"
                     r"hips?|shoulders?|knees?|elbows?|thighs?|face|chin)\b)"
                     r"(?=\s*(?:[.,;!?]|$)"
                     r"|\s+(?:onto|into|on|in|to|across|down|up|over|under|back|out|"
                     r"away|upright|off|against|toward|towards|through|round|around|"
                     r"beside|behind|clear)\b)", re.I)
    return any(obj.match(text[m.end():]) for m in _MOVE_VERB.finditer(text or ""))


def turns_in(text, names=()):
    """Does this beat rotate a body, move one, or bring the view around it?"""
    return bool(_TURN_CUE.search(text or "")) or body_moved(text, names)


FALL_HOLD = (" A bound body falls as one piece: the fastened limbs stay fastened and travel "
             "with it, the arms staying in the hold, the shoulder, hip or side takes "
             "the landing, and the legs fold together under the body.")

FALL_HOLD_FREE = (" The body falls as one piece: the arms stay with it and the shoulder, "
                  "hip or side takes the landing, the legs folding together under it.")

_FALL_CUE = re.compile(
    r"\b(?:falls?|fell|falling|drops?\s+to|dropped\s+to|collapse[sd]?|collapsing|"
    r"topple[sd]?|topples|tips?\s+over|tipped\s+over|keels?\s+over|goes\s+down|"
    r"went\s+down|slumps?|slumped|stumbles?|stumbled|overbalance[sd]?|"
    r"loses?\s+(?:her|his|their)\s+balance|lost\s+(?:her|his|their)\s+balance|"
    r"(?:push|knock|shove|pull|drag|throw|thr[eo]w)(?:es|s|ed|n)?\s+"
    r"(?:(?:her|him|them|herself|himself|themselves|[A-Z][\w-]+)\s+"
    r"(?:over|down|to\s+the\s+(?:floor|ground))|to\s+the\s+(?:floor|ground))|"
    r"hits?\s+the\s+(?:floor|ground|deck))\b", re.I)


_OBJECT_FALLER = re.compile(
    r"\b(?:it|its|belt|belts|top|tops|shirt|shorts|jeans|trousers|skirt|dress|"
    r"coat|jacket|jumper|sweater|scarf|tie|boot|boots|shoe|shoes|sock|socks|"
    r"glove|gloves|hat|bag|towel|sheet|blanket|cuffs?|handcuffs?|chain|chains|"
    r"rope|ropes|tape|gag|collar|key|keys|phone|glass|bottle|cup|plate|book|"
    r"clothes|clothing|garment|garments|thing|things)\b", re.I)
# A person going down. A NAME, or a personal pronoun that is not "it".
_PERSON_FALLER = re.compile(
    r"\b(?:she|he|they|her|him|them|herself|himself|themselves|"
    r"[A-Z][\w-]{1,24})\b")


def falls_in(text):
    """Does a BODY go down in this beat? A dropped garment is not a fall.

    The fall guard tells the shot what takes the landing and what the legs do, so a
    match on something that is not a person aims all of that at the wrong subject
    and the shot puts a body on the floor to satisfy it.

    The subject is whatever sits between the start of the clause and the verb. An
    object there -- "it drops to the ground", "the belt falls to the floor" -- is
    the thing being let go of, not somebody going down."""
    t = text or ""
    for m in _FALL_CUE.finditer(t):
        head = t[:m.start()]
        cut = max((c.end() for c in
                   re.finditer(r"[.;!?]\s+|,\s*|\s+(?:and|but|then|so)\s+", head)),
                  default=0)
        subject = head[cut:]
        if _OBJECT_FALLER.search(subject):
            continue                      # a thing came down, not a person
        if not subject.strip() or _PERSON_FALLER.search(subject):
            return True
        return True
    return False


CHAIN_HOLD = (" Every restraint stays closed and fastened as it was put on, its links "
              "keeping their size and the run between them taut") + FORM_HOLD

CHAIN_POSE_HOLD = (" Every restraint stays closed and fastened as it was put on; the metal "
                   "is already drawn to its full length, so the position it fixes is the "
                   "position that keeps, and the body strains against it while the "
                   "fastenings hold") + FORM_HOLD

# A position that hardware can be locked to enforce.
_FORCED_POSE = re.compile(
    r"\b(?:squat(?:s|ting|ted)?|kneel(?:s|ing)?|knelt|crouch(?:es|ing|ed)?|"
    r"hogtied|hog-?tied|hogcuffed|hog-?cuffed|trussed|"
    r"bent\s+(?:over|double)|doubled\s+over|folded\s+(?:up|forward)|"
    r"spread[- ]eagled?|curled\s+up|"
    r"on\s+(?:her|his|their)\s+(?:knees|haunches))\b", re.I)


_LIMB_EV = (r"\b(?:hands?|wrists?|arms?|cuffed|handcuffed|bound|tied|shackled|"
            r"manacled|strapped|secured|fastened|locked|pinned|chained|clasped|"
            r"held|clipped|hooked)")

_LIMB_ANCHOR = (
    (r"(?:cuffed|handcuffed|bound|tied|shackled|manacled|strapped|secured|"
     r"fastened|locked|pinned|chained|clipped|hooked|suspended|hoisted)\s+"
     r"(?:\w+\s+){0,3}?(?:above|over)\s+(?:her|his|their|the)\s+head|"
     r"(?:hands?|wrists?|arms?)\s+(?:\w+\s+){0,4}?"
     r"(?:above|over)\s+(?:her|his|their|the)\s+head|"
     r"(?:hands?|wrists?|arms?)\s+(?:\w+\s+){0,3}?overhead|"
     r"(?:hands?|wrists?|arms?)\s+(?:\w+\s+){0,2}?stretched\s+(?:up|upward)",
     "above the head"),
    (_LIMB_EV + r"\s+(?:\w+\s+){0,3}?behind\s+(?:her|his|their|the)\s+back",
     "behind the back"),
    (r"(?:hands?|wrists?|arms?)\s+(?:\w+\s+){0,3}?behind\s+(?:her|his|their)\b",
     "behind the back"),
    (r"(?:cuffed|handcuffed|bound|tied|shackled|manacled|strapped|secured|"
     r"fastened|locked|pinned|clasped|held)\s+(?:\w+\s+){0,2}?"
     r"behind\s+(?:her|his|their)\b", "behind the back"),
    (r"at\s+the\s+small\s+of\s+(?:her|his|their|the)\s+back", "behind the back"),
    (r"\b(?:hands?|wrists?|arms?)\s+behind\s+back\b", "behind the back"),
    (_LIMB_EV + r"\s+(?:\w+\s+){0,3}?in\s+front\s+of\s+(?:her|his|their)\s+"
     r"(?:body|chest|waist)", "in front of the body"),
    (_LIMB_EV + r"\s+(?:\w+\s+){0,3}?(?:(?:out\s+)?to\s+the\s+sides?|spread\s+wide)",
     "out to the sides"),
    (_LIMB_EV + r"\s+(?:\w+\s+){0,3}?at\s+(?:her|his|their|the)\s+waist",
     "at the waist"),
)
_FASTEN_PART = (r"(?:chained|cuffed|handcuffed|shackled|manacled|locked|padlocked|"
                r"fastened|secured|tethered|bound|tied|strapped|clipped|hooked|"
                r"bolted|attached|anchored|leashed|roped|affixed|fixed|pinned|"
                r"hitched|moored|lashed|chaining|cuffing|locking|fastening|"
                r"securing|tethering|tying|strapping|clipping|hooking|bolting|"
                r"attaching|anchoring|padlocking)")
_FASTEN_S = (r"(?<!\bthe\s)(?<!\ba\s)(?<!\ban\s)(?<!\bthese\s)(?<!\bthose\s)"
             r"(?<!\btwo\s)(?<!\bsome\s)(?<!\bmore\s)"
             r"(?:chains|cuffs|handcuffs|shackles|manacles|locks|padlocks|fastens|"
             r"secures|tethers|ties|straps|clips|hooks|bolts|attaches|anchors|"
             r"leashes|ropes|pins)")
_FASTEN_WEAK = (r"(?:chains?|ropes?|cords?|cables?|leash(?:es)?|leads?|straps?|"
                r"tethers?|links?|lines?)\s+(?:\S+\s+){0,4}?"
                r"(?:run|hold|lead|stretch|extend|go|reach|drop|hang)(?:s|es|ing)?")
_ANCHOR_POINT = re.compile(
    r"\b(?:" + _FASTEN_PART + r"|" + _FASTEN_S + r"|" + _FASTEN_WEAK + r")"
    r"\b(?:\s+\S+){0,5}?\s+to\s+" + engine.ANCHOR_DET +
    r"((?:bed\s*frames?|bed\s*heads?|headboards?|bed\s*posts?|beds?|rails?|railings?|"
    r"bars?|posts?|rings?|hooks?|pipes?|radiators?|chairs?|tables?|beams?|frames?|"
    r"grates?|grilles?|fences?|walls?|floors?|grounds?|ceilings?|pillars?|columns?|"
    r"stakes?|eye\s*bolts?|bolts?|brackets?|cages?|bunks?|benches?|ladders?|"
    r"girders?|struts?|anchors?|loops?))\b", re.I)


def limb_anchor(text):
    """Where fastened limbs are being held, as a phrase. '' when the text says none."""
    t = text or ""
    where = next((phrase for pat, phrase in _LIMB_ANCHOR
                  if re.search(pat, t, re.I)), "")
    m = _ANCHOR_POINT.search(t)
    point = ("at the " + re.sub(r"\s+", " ", m.group(1).lower())) if m else ""
    if where and point:
        return f"{where}, {point}"
    return where or point


_TIGHT_FRAME = re.compile(
    r"\bclose[-\s]?up|\bclose\s+(?:shot|on)\b|\btight\s+(?:on|shot)\b|"
    r"\bfills?\s+the\s+frame\b|\bmacro\b", re.I)


_WHOLE_BODY = re.compile(
    r"\b(?:serves?|serving|throws?|throwing|kicks?|kicking|hits?|hitting|"
    r"swings?|swinging|spikes?|blocks?|blocking|jumps?|jumping|runs?|running|"
    r"sprints?|sprinting|dances?|dancing|plays?|playing|climbs?|climbing|"
    r"lifts?|lifting|carries|carrying|pushes|pushing|pulls?|pulling|"
    r"swims?|swimming|stretches|stretching|wrestles?|fights?|fighting)\b", re.I)
_FRAME_SIZE = re.compile(
    r"\bclose[-\s]?ups?|\bclose\s+(?:shots?|on)\b|\btight\s+(?:shots?|on)\b|\bmacro\b|"
    r"\bwide\s+(?:shots?|angle)\b|\bwide\b|\bestablishing\b|\blong\s+shots?\b|"
    r"\bfull\s+(?:shots?|body|figure|length)\b|\bmedium\s+shots?\b|\bmid\s+shots?\b|"
    r"\btwo[-\s]?shots?\b|\bover[-\s]the[-\s]shoulder\b|\bpov\b|"
    r"\bhead\s+and\s+shoulders\b|\bportrait\b|\bwaist[-\s]up\b|"
    r"\bknees?[-\s]up\b|\bhead\s+to\s+(?:toe|foot|feet)\b", re.I)


_CAMERA_ASKED = re.compile(
    r"\bcameras?\b|\blens\b|\bshot\s+on\b|\bpans?\b|\bpanning\b|\btilts?\b|\btilting\b|"
    r"\bdolly(?:ing)?\b|\btracking\s+shot\b|\btrucks?\s+(?:in|out|left|right)\b|"
    r"\bzoom(?:s|ing|ed)?\b|\bpush(?:es|ing)?\s+in\b|\bpull(?:s|ing)?\s+(?:back|out)\b|"
    r"\bcrane\b|\bjib\b|\bsteadicam\b|\bhand-?held\b|\bgimbal\b|\bdrone\b|"
    r"\borbit(?:s|ing)?\b|\barc(?:s|ing)?\s+around\b|\bcircles?\s+around\b|"
    r"\bwhip\s+pan\b|\brack\s+focus\b|\bfollow(?:s|ing)?\s+shot\b|\bpov\b|"
    r"\blocked[-\s]off\b|\bstatic\s+(?:shot|frame|camera)\b|\bcrash\s+zoom\b", re.I)


def camera_hold(beat, anchor="", moving=False):
    """One sentence holding the camera still, where nothing has placed it.

    `moving` stands it down for a shot that travels between places: a journey the
    node has already asked to keep every step in frame is a shot whose camera has to
    go with them, and telling it to stay put contradicts the beat."""
    if moving:
        return ""
    if _CAMERA_ASKED.search(str(beat or "")) or _CAMERA_ASKED.search(str(anchor or "")):
        return ""
    return " The shot is one unbroken take from one position, angle and distance."


def frame_hold(beat, anchor="", people=1):
    """Say the frame holds a whole body, where nothing else says what the frame is.

    THE PORTRAIT IS WHAT AN UNSTATED FRAME BECOMES. This file already records the
    reason: "an attribute a prompt does not state is not LEFT to the model, it is
    left to the model's prior -- which for a named, described person is a PORTRAIT:
    facing the lens, pleasantly, because that is what photographs of people are."
    The sheet describes a face in every shot, because clothing continuity needs it,
    and the mouth guard describes a mouth in every silent shot, because babble needs
    it -- so the text is weighted towards a face and nothing in it says how much of
    the person to show. Measured on a volleyball beat: 15 words of appearance and 9
    of mouth against 8 of action. Reported as the camera staying fixated on one
    character, staring into the lens, with no reference image anywhere in the run.

    Only where the beat stages something a portrait cannot contain, and only where
    the author has said nothing about the camera -- in the beat or in the anchor.
    Their framing always wins, a close-up included, because a close-up is a frame
    somebody asked for. Impersonal, like the other picture guards, and positively
    phrased: it says what the frame holds, never what it is not."""
    b = str(beat or "")
    if _FRAME_SIZE.search(b) or _FRAME_SIZE.search(str(anchor or "")):
        return ""
    if tight_framing(b) or tight_framing(str(anchor or "")):
        return ""
    if not (_WHOLE_BODY.search(b) or _TRAVEL_VERB.search(b)):
        return ""
    if int(people or 1) > 1:
        return (" The frame holds every body in it whole, head to feet, with the room "
                "around them.")
    return (" The frame holds the whole body, head to feet, with the room around it.")


def tight_framing(text):
    """Does this beat call for a frame close enough to lose the anchor point?"""
    return bool(_TIGHT_FRAME.search(text or ""))


_FRAME_ON = re.compile(
    r"\b(?:close[-\s]?up|close\s+shot|tight\s+shot|macro(?:\s+lens)?)\b[^.;]{0,24}?"
    r"\bon\s+(?:her|his|their|its|the)\s+([\w][\w\- ]{1,20})"
    r"|\b(?:close|tight)\s+on\s+(?:her|his|their|its|the)\s+([\w][\w\- ]{1,20})", re.I)
_FRAME_HOLDS = (
    (r"face|eyes?|mouth|lips|head|hair|jaw|cheeks?|ears?|nose|expression",
     frozenset(("torso",))),
    (r"hands?|fingers?|wrists?|palms?|knuckles?", frozenset(("hands",))),
    (r"feet|foot|ankles?|toes?", frozenset(("feet",))),
    (r"chest|breasts?|torso|shoulders?|stomach|belly|waist|back",
     frozenset(("torso",))),
    (r"legs?|thighs?|hips?|knees?|calves|calf", frozenset(("legs", "feet"))),
)


def frame_holds(text):
    """The garment regions a named close frame can still contain.

    None when the text names no close frame, or names one without saying what it
    is close ON -- a bare "close-up" gives no way to know what is in it, and
    guessing would be the node cropping the author's wardrobe on a coin toss."""
    m = _FRAME_ON.search(text or "")
    if not m:
        return None
    subject = (m.group(1) or m.group(2) or "").strip().lower()
    for pat, regions in _FRAME_HOLDS:
        if re.search(r"\b(?:" + pat + r")\b", subject, re.I):
            return regions
    return None


def out_of_frame_garments(scene, holds):
    """Garments in `scene` whose region the frame cannot show.

    A garment that cannot be placed at all is KEPT: an unplaceable item is one this
    file does not recognise, and cropping what it does not understand is how a
    wardrobe quietly loses things the author wrote."""
    if not holds:
        return []
    out = []
    for g in garments_in(scene or ""):
        r = region_of(g)
        if r and r not in holds:
            out.append(g)
    return out


_GAZE_PREP = r"(?:at|to|towards?|into|onto|over\s+at)"
_GAZE_TAIL = (r"(?=[.,;:!?]|\s+(?:and|as|while|when|who|which|that|with|for|from|in|on|"
              r"before|after|until)\b|$)")
_GAZE_DET = r"(?:the|a|an|her|his|their|its|that|this)\s+"
_LOOK_AT = re.compile(
    r"\b(?:look(?:s|ed|ing)?|star(?:e|es|ed|ing)|gaz(?:e|es|ed|ing)|"
    r"glanc(?:e|es|ed|ing)|peer(?:s|ed|ing)?|squint(?:s|ed|ing)?)\s+"
    r"(?:back\s+|down\s+|up\s+|over\s+|round\s+|around\s+|straight\s+|right\s+)?"
    + _GAZE_PREP + r"\s+" + _GAZE_DET + r"([\w][\w\- ]{0,24}?)" + _GAZE_TAIL, re.I)
_WATCH = re.compile(
    r"\b(?:watch(?:es|ed|ing)?|stud(?:y|ies|ied|ying)|examin(?:e|es|ed|ing))\s+"
    + _GAZE_DET + r"([\w][\w\- ]{0,24}?)" + _GAZE_TAIL, re.I)
_NOT_A_TARGET = frozenset(
    "him her them it me us you himself herself themselves one other others "
    "time moment thing things way".split())


_LOOK_AT_WHO = re.compile(
    r"\b(?:look(?:s|ed|ing)?|star(?:e|es|ed|ing)|gaz(?:e|es|ed|ing)|"
    r"glanc(?:e|es|ed|ing)|peer(?:s|ed|ing)?)\s+"
    r"(?:back\s+|down\s+|up\s+|over\s+|round\s+|around\s+|straight\s+|right\s+)?"
    + _GAZE_PREP + r"\s+([A-Z][\w-]+|him|her|them|he|she|they)\b"
    r"|\b(?:watch(?:es|ed|ing)?|stud(?:y|ies|ied|ying)|examin(?:e|es|ed|ing))\s+"
    r"([A-Z][\w-]+|him|her|them)\b", re.I)
_PRONOUN_SEX = {"him": "he", "he": "he", "her": "she", "she": "she"}


def _person_looked_at(beat, sheet="", described=()):
    """The PERSON this beat says somebody is watching. '' when it is not resolvable.

    Only where it is unambiguous. Three people and a bare "him" resolves to nobody,
    and guessing which one is worse than saying nothing: a shot told the wrong
    sightline is a shot that has to be reshot, while a shot told none is only back
    where it was."""
    m = _LOOK_AT_WHO.search(beat or "")
    if not m:
        return ""
    raw = (m.group(1) or m.group(2) or "").strip()
    if not raw:
        return ""
    rows = {n: ln for n, ln in sheet_lines(sheet or "") if n}
    # A NAME, spelled as the sheet spells it.
    for n in rows:
        if raw.lower() == n.lower():
            return n
    want = _PRONOUN_SEX.get(raw.lower())
    here = [n for n in (described or []) if n in rows]
    # The looker is not the one being looked at.
    looker = subjects_for(beat, sheet, _LOOK_VERB_SRC)
    here = [n for n in here if n not in set(looker)]
    if want:
        here = [n for n in here
                if re.search(r"\b" + want + r"\b", rows.get(n, ""), re.I)]
    return here[0] if len(here) == 1 else ""


def look_target(beat, sheet="", described=()):
    """What this beat says somebody is looking at. '' when it names nothing."""
    for pat in (_LOOK_AT, _WATCH):
        m = pat.search(beat or "")
        if not m:
            continue
        target = re.sub(r"\s+", " ", m.group(1)).strip(" -")
        if not target or target.lower() in _NOT_A_TARGET:
            continue
        return target
    return _person_looked_at(beat, sheet, described)


_MOVES_OFF_SRC = (r"walks?|walked|runs?|ran|steps?|stepped|moves?|moved|crosses|"
                  r"crossed|leaves?|left|exits?|exited|goes|went|heads?|headed|"
                  r"climbs?|climbed|follows?|followed")
_MOVES_OFF = re.compile(r"\b(?:" + _MOVES_OFF_SRC + r")\b", re.I)


_LOOK_VERB_SRC = (r"look(?:s|ed|ing)?|star(?:e|es|ed|ing)|gaz(?:e|es|ed|ing)|"
                  r"glanc(?:e|es|ed|ing)|peer(?:s|ed|ing)?|watch(?:es|ed|ing)?|"
                  r"stud(?:y|ies|ied|ying)")
_LOOK_VERB = re.compile(r"\b(?:" + _LOOK_VERB_SRC + r")\b", re.I)


def subjects_for(beat, sheet, verbs):
    """Which people on the sheet this beat puts in front of one of these verbs.

    The shared shape behind speakers_in and vocal_sources_in, including the
    conjunction guard: "Dan holds the door and McKenna looks away" must not credit
    Dan, because `and` opens a new predicate with its own subject."""
    b, out = beat or "", []
    for n, _ in sheet_lines(sheet):
        if n and re.search(r"\b" + re.escape(n) + r"\b" + _UP_TO_TWO_WORDS
                           + r"\s+(?:" + verbs + r")\b", b, re.I):
            out.append(n)
    return out


def looks_somewhere(beat):
    """Does this beat stage a look at all, nameable target or not?

    "Mara looks at her" names no target this node can restate -- but it does
    move the look, and holding the previous target across it says her eyes are
    on a television she has just turned away from."""
    return bool(_LOOK_VERB.search(beat or ""))


def gaze_hold(target, who="", is_person=False):
    """One sentence putting the eyes and the head on the thing the beat named.

    NAMED when the caller says to, which it does once a second person is in the
    shot. Reported: "the girl is stuck gazing at a camera while the other character
    does his part" -- a look staged by one person went on being said impersonally in
    shots she was not in, so it landed on whoever was. Impersonal is still right with
    one person in frame: naming somebody is a second mention of them, and a described
    person is a person the model draws.

    Impersonal, like the hardware placement clause: naming the person again is one
    more mention of a person, and that has its own cost. Says nothing about where the
    camera is -- the shot may be looking straight down the line of sight -- only that
    the head is turned to face what the eyes are on."""
    if not target:
        return ""
    what = target if is_person else f"the {target}"
    if who:
        return f" {who[0].upper()}{who[1:]} eyes and head are turned to {what}."
    return f" The eyes and the head are turned to {what}."


def dialogue_gaze(n_people):
    """One impersonal sentence turning speakers and listeners to each other.

    gaze_hold restates a look the beat named. A dialogue beat that names none
    leaves both faces to the portrait prior, and the prior is the lens: reported
    as "it looks like they are talking to a camera and not to each other". A
    spoken line has an addressee whether or not the beat wrote one, and the
    addressee is in the shot, so turning the faces to each other is the one thing
    that can be said without inventing anything.

    Impersonal, like gaze_hold, and for the same reason: on a dialogue shot the
    speaker's name is spent by the mouth guard and the listener's by told_hold,
    and a third mention is a third person. Positively phrased -- at cfg 1 naming
    the lens would ask for it. Says nothing about where the camera is."""
    if n_people < 2:
        return ""
    if n_people == 2:
        return " They face each other, eyes on each other."
    return " Eyes on whoever is speaking, faces turned to them."


def forced_pose(text):
    """Does this text put a body into a position that hardware can enforce?"""
    return bool(_FORCED_POSE.search(text or ""))

_RIGID_HARDWARE = re.compile(
    r"\b(?:chain(?:s|ed|ing)?|padlock(?:s|ed|ing)?|shackle[sd]?|manacle[sd]?|"
    r"handcuff(?:s|ed)?|cuffs?|cuffed|irons|spreader\s+bar|"
    r"hogcuffed|hog-?cuffed)\b", re.I)


_TAPE_GAG = (r"(?:duct[\s-]*)?tape\s+gag|"
             r"gag(?:s|ged|ging)?\s+\w{0,12}\s*with\s+"
             r"(?:duct\s+|packing\s+|masking\s+)?tape|"
             r"tape\s+(?:over|across)\s+(?:her|his|their|the)\s+mouth")
_TAPE_GAG_CLAUSE = "a strip of tape lies flat across the mouth"
_GAG_CLAUSE = "a gag sits in the mouth"

_HARDWARE_ANCHOR = (
    (r"collar(?:s|ed)?",                 "a collar closes around the neck"),
    (r"leash(?:es)?|lead\b",             "a leash clips to the collar at the neck and hangs down from it"),
    (_TAPE_GAG,                          _TAPE_GAG_CLAUSE),
    (r"gag(?:s|ged)?|ball\s*gag",        _GAG_CLAUSE),
    (r"blindfold(?:s|ed)?",              "a blindfold covers the eyes"),
    (r"handcuff(?:s|ed)?",               "handcuffs close around the wrists"),
    (r"shackle[sd]?|leg\s+irons",        "shackles close around the ankles"),
    (r"harness(?:es)?",                  "a harness sits on the torso"),
    (r"spreader\s+bar",                  "a spreader bar holds the ankles apart"),
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
    if _TAPE_GAG_CLAUSE in out and _GAG_CLAUSE in out:
        out.remove(_GAG_CLAUSE)
    return out


def anchor_clause(phrases):
    """One sentence saying where the named hardware sits."""
    if not phrases:
        return ""
    return " Each piece of hardware sits where it belongs: " + "; ".join(phrases) + "."


_STATE_THING = (r"doors?|gates?|windows?|curtains?|blinds?|shutters?|"
                r"hatch(?:es)?|tailgates?|lids?|drawers?")
_STATE_WORD = r"closed|shut|open|locked|unlocked|latched|bolted|drawn|ajar|sealed"
_STATE_ACTS = (r"opens?|opened|opening|closes?|closed|closing|shuts?|shutting|"
               r"slams?|slammed|slamming|slides?|slid|sliding|pulls?|pulled|pulling|"
               r"pushes?|pushed|pushing|draws?|drew|drawing|locks?|locked|locking|"
               r"unlocks?|unlocked|unlocking|lifts?|lifted|lifting|raises?|raised|"
               r"lowers?|lowered|swings?|swung|yanks?|yanked|wrenches|wrenched")
_STATE_ACT = re.compile(r"\b(" + _STATE_ACTS + r")\s+((?:[\w']+\s+){0,3}?)(" +
                        _STATE_THING + r")\b", re.I)
_STATE_DET = re.compile(r"\b(?:the|a|an|its|his|her|their|our|my|your|this|that|these|"
                        r"those|both|all|each|every|another|one|two|three|\w+'s)\b", re.I)


def _adjectival(verb, gap):
    """Is this state word describing the noun rather than acting on it?

    Only words that are ALSO states can be adjectives: "opens" is a verb however it
    is placed. Modifiers may sit between -- "closed rear doors", "shut cargo doors" --
    so the test is the determiner, not the distance."""
    return (bool(re.fullmatch(_STATE_WORD, verb, re.I))
            and not _STATE_DET.search(gap or ""))


_STATE_ADJ = re.compile(r"\b(" + _STATE_WORD + r")\s+((?:[\w']+\s+){0,2}?)(" +
                        _STATE_THING + r")\b", re.I)
_STATE_PRED = re.compile(r"\b(" + _STATE_THING + r")\s+" +
                         r"((?:(?:are|is|was|were|remains?|stay|stays|still|both|all)\s+){0,2})(" +
                         _STATE_WORD + r")\b", re.I)


_POSTURE_OF = engine._POSTURE_OF
_NOT_A_BODY = engine._NOT_A_BODY


_POSE_DIRECTIONS = frozenset(
    "down up back onto into on in over out away flat still there here".split())
# ...and the pronouns that ARE an object.
_OBJECT_PRONOUN = frozenset(
    "her him them herself himself themselves".split())


def posture_in(beat, cast):
    """{name: posture} this beat puts somebody into. {} when it stages none.

    Attributed by CLAUSE, so "Kate sits down and Sam stays by the door" does not
    seat them both. A clause with a posture verb and no name belongs to whoever the
    beat names first, which is the same reading the removal agent uses."""
    b = str(beat or "")
    people = [n for n in (cast or []) if n]
    if not b or not people:
        return {}
    out = {}
    at0 = 0
    for part in re.split(r"(?<=[.;!?])[\"'”’]?\s+", b):
        base = b.find(part, at0)
        base = at0 if base < 0 else base
        at0 = base + len(part)
        if _NOT_A_BODY.search(part):
            continue
        hits = sorted(((m.start(), pose) for pose, rx in _POSTURE_OF
                       for m in rx.finditer(part)
                       if not _in_a_request(b, base + m.start())
                       and not engine.denied_posture(part, m.start())),
                      key=lambda h: h[0])
        prev = 0
        for at, pose in hits:
            span = part[prev:at]
            tail = part[at:]
            obj = re.match(r"\w+\s+(?:the\s+)?([\w'-]+)\s+"
                           r"(?:down|up|back|onto|into|on|in)\b", tail, re.I)
            if obj and obj.group(1).lower() not in _POSE_DIRECTIONS:
                word = obj.group(1)
                named = [n for n in people
                         if n.lower() == word.lower()]
                if named:
                    for n in named:
                        out[n] = pose
                    prev = at
                    continue
                if word.lower() not in _OBJECT_PRONOUN:
                    obj = None          # not a name on the sheet, not a pronoun
                actor = [n for n in people
                         if re.search(r"\b" + re.escape(n) + r"\b", span, re.I)]
                others = [n for n in people if n not in actor]
                if obj and len(others) == 1:
                    out[others[0]] = pose
                    prev = at
                    continue
            here = [n for n in people
                    if re.search(r"\b" + re.escape(n) + r"\b", span, re.I)]
            if not here:
                first = next((n for n in people
                              if re.search(r"\b" + re.escape(n) + r"\b", b, re.I)), None)
                here = [first] if first else (people[:1] if len(people) == 1 else [])
            for n in here:
                out[n] = pose
            prev = at
    return out


_HANDLES = re.compile(
    r"\b(?:takes?|took|taking|picks?|picked|picking|places?|placed|placing|"
    r"puts?|putting|sets?|setting|lifts?|lifted|lifting|carries|carried|"
    r"carrying|fetch(?:es|ed|ing)?|hands?|handed|handing|passes|passed|"
    r"opens?|opened|opening|closes?|closed|closing|pours?|poured|pouring)\b",
    re.I)


def _real_travel(span):
    """Does this span actually move somebody, or does it just LOOK like it?

    "takes off her shirt" matched the travel list on "takes" -- the list has it
    for "takes her to the car" -- so undressing cleared the posture latch and the
    shot after was told nothing about how the body was left. Reported as a squat
    not being held: she stood up on her own.

    Only this one phrasal is excluded. "walks off" and "runs off" are locomotion
    and the particle does not change that; it is the HANDLING verb that makes
    "takes off" mean something else entirely."""
    for m in _TRAVEL_VERB.finditer(span or ""):
        if re.match(r"\s+off\b", (span or "")[m.end():m.end() + 6], re.I) \
                and re.match(r"(?:takes?|took|taking)$", m.group(0), re.I):
            continue
        return True
    return False


def posture_cleared(beat, poses):
    """{name} whose latched posture this beat contradicts without restating one.

    The beat is the author's own words and outranks a hold: where it puts somebody
    on their feet, the hold has to let go or it argues with the shot it is standing
    next to."""
    b = _outside_speech(str(beat or ""))
    out = set()
    if not b:
        return out
    for name, pose in (poses or {}).items():
        m = re.search(r"\b" + re.escape(name) + r"\b", b, re.I)
        if not m:
            continue
        # What this beat has them doing, up to the end of the clause.
        stop = re.search(r"[.;!?]", b[m.end():])
        span = b[m.end():m.end() + (stop.start() if stop else len(b))]
        if _real_travel(span):
            out.add(name)
        elif pose == "lying down" and _HANDLES.search(span):
            out.add(name)
    return out


def posture_hold(poses, described):
    """One short sentence keeping people in the pose an earlier beat put them in.

    Only for people this shot DESCRIBES -- a pose belonging to somebody the text
    does not mention is a pose for nobody, and the model draws the person that
    sentence implies. Short on purpose: this is latched, so it lands in every shot
    after the one that stages it, and a long clause repeated is the guard bloat
    this node was rebuilt to escape."""
    who = [(n, p) for n, p in (poses or {}).items()
           if n in set(described or []) and p != "standing"]
    if not who:
        return ""
    if len(who) == 1:
        return f" {who[0][0]} is {who[0][1]}."
    _poses = {p for _n, p in who}
    if len(_poses) == 1 and len(who) == len(set(described or [])):
        return (f" Both are {who[0][1]}." if len(who) == 2
                else f" Everyone in the shot is {who[0][1]}.")
    said = "; ".join(f"{n} is {p}" for n, p in who[:2])
    return f" {said}."


def _state_key(thing):
    """One key for 'door' and 'doors', so a beat acting on either clears both."""
    t = (thing or "").lower()
    return t[:-2] if t.endswith("es") and t.startswith("hatch") else t.rstrip("s")


_OPENS = re.compile(r"(?:opens?|opened|opening|unlocks?|unlocked|unlocking|"
                    r"lifts?|lifted|lifting|raises?|raised)\Z", re.I)
_SHUTS = re.compile(r"(?:closes?|closed|closing|shuts?|shutting|slams?|slammed|"
                    r"slamming|locks?|locked|locking|lowers?|lowered)\Z", re.I)


_WAY_WORD = re.compile(r"\A(?:(open|wide|back|apart|aside)|(shut|closed))\b", re.I)
_STATE_ACT_REV = re.compile(
    r"\b(?:the|a|an|its|his|her|their|our|my|your|this|that|these|those|both|all|"
    r"each|every|another|one|two|three|\w+'s)\s+((?:[\w']+\s+){0,2}?)(" + _STATE_THING
    + r")\s+((?:[\w']+\s+){0,1}?)(" + _STATE_ACTS + r")\b", re.I)


def state_changes(text):
    """[(thing, 'open'|'shut'|None)] for the scenery this text actually works.

    A state word sitting straight in front of its noun is an adjective describing
    the thing, not a verb acting on it: "the closed doors" says nothing happens.
    The direction is None where neither the verb nor a word beside it carries one."""
    out, seen = [], set()
    found = [(m.group(1), m.group(2), m.group(3), m.end(), False)
             for m in _STATE_ACT.finditer(text or "")]
    found += [(m.group(4), m.group(3), m.group(2), m.end(), True)
              for m in _STATE_ACT_REV.finditer(text or "")]
    for verb, gap, thing, end, reverse in sorted(found, key=lambda f: f[3]):
        if reverse:
            if re.fullmatch(_STATE_WORD, verb, re.I):
                continue
        elif _adjectival(verb, gap):
            continue
        key = _state_key(thing)
        if key in seen:
            continue
        seen.add(key)
        way = ("open" if _OPENS.match(verb) else
               "shut" if _SHUTS.match(verb) else None)
        if way is None:
            # The beat named the end even though the verb does not.
            said = _WAY_WORD.match((text or "")[end:].lstrip())
            if said:
                way = "open" if said.group(1) else "shut"
        out.append((thing.lower(), way))
    return out


def state_acts(text):
    """Which of those things this text works, in either direction."""
    return [_state_key(t) for t, _ in state_changes(text)]


_PLACE = engine.PLACES
_PLACE_WORD = engine._PLACE_WORD
_PLACE_ALSO_A_VERB = engine.PLACE_ALSO_A_VERB
_NOT_A_ROOM_MODIFIER = {"the", "a", "an", "this", "that", "her", "his", "their",
                        "its", "my", "our", "your", "in", "into", "inside", "of",
                        "from", "to", "at", "on", "and", "or", "same", "other"}
_MOD = engine._ROOM_MOD
_DET_POSS = engine.DET_POSS
_GOES_TO = re.compile(r"\b(?:to|into|toward|towards|through\s+to|"
                      r"enters?|entered|entering|reaches|reached|arrives?\s+(?:at|in)|"
                      r"steps?\s+into|stepped\s+into)\s+"
                      + _DET_POSS + r"\s+" + _MOD
                      + r"(" + _PLACE + r")\b", re.I)
# "down the hallway", "along the corridor" -- what it passes THROUGH.
_GOES_VIA = re.compile(r"\b(?:down|along|across|through|up|via|past)\s+"
                       + _DET_POSS + r"\s+" + _MOD
                       + r"(" + _PLACE + r")\b", re.I)
# "from the living room", "out of the kitchen" -- where it STARTS.
_GOES_FROM = re.compile(r"\b(?:from|out\s+of|leaves?|leaving)\s+"
                        + _DET_POSS + r"?\s*" + _MOD
                        + r"(" + _PLACE + r")\b", re.I)
# A verb that actually MOVES somebody. "looks to the bedroom" is not travel.
_TRAVEL_VERB = re.compile(
    r"\b(?:walk|walks|walked|walking|lead|leads|led|leading|take|takes|took|taking|"
    r"go|goes|went|going|head|heads|headed|heading|move|moves|moved|moving|"
    r"carry|carries|carried|carrying|follow|follows|followed|following|"
    r"step|steps|stepped|stepping|climb|climbs|climbed|climbing|"
    r"run|runs|ran|running|come|comes|came|coming|"
    r"leave|leaves|left|leaving|enter|enters|entered|entering|"
    r"cross|crosses|crossed|crossing|exit|exits|exited|exiting|"
    r"escort|escorts|escorted|escorting|usher|ushers|ushered|ushering|"
    r"march|marches|marched|marching|guide|guides|guided|guiding|"
    r"bring|brings|brought|bringing|drag|drags|dragged|dragging|"
    r"haul|hauls|hauled|hauling|"
    r"return|returns|returned|returning)\b", re.I)


def travel_in(beat):
    """(from, via, to) for a beat that moves somebody between places.

    All three may be "". Only when a MOVEMENT verb is present: "she looks to the
    bedroom" names a place and goes nowhere."""
    b = str(beat or "")
    if not b or not _TRAVEL_VERB.search(b):
        return ("", "", "")

    def _one(rx):
        m = rx.search(b)
        return re.sub(r"\s+", " ", m.group(1)).strip().lower() if m else ""

    to, via, frm = _one(_GOES_TO), _one(_GOES_VIA), _one(_GOES_FROM)
    # A place cannot be two ends of the same journey.
    if via and via == to:
        via = ""
    if frm and frm in (to, via):
        frm = ""
    return (frm, via, to)


def travel_legs(beat):
    """(from, via, to) for a beat that moves somebody, with the one promotion that
    the render and the SIZING have to agree on.

    A BEAT THAT TRAVELS ALONG A PLACE ENDS IN IT. "walks down the hallway" reads as a
    via with no destination, and travel_anchor says nothing without one -- so that
    shot was told nothing about where it was, the tracked room kept the bedroom, and
    the NEXT beat opened "in the bedroom" on its way to the kitchen. Reported as a bed
    in the hallway. The place travelled along is where the beat arrives, so the shot
    opens where the last one left off and walks into it, in frame.

    Here rather than inline in the shot loop because travel_spaces reads the same
    fact to size the shot. Kept in two places it would drift, and a transit rendered
    as a walk while being sized as if it went nowhere is exactly the split that put a
    three-room walk in a three-second shot."""
    frm, via, to = travel_in(beat)
    if via and not to:
        via, to = "", via
    return frm, via, to


def where_hold(here, scene):
    """Say which room the shot is in, once the film has left the one in the scene.

    The scene paragraph is stamped into EVERY shot -- it has to be, or a removal
    has nothing to scrub -- so a script that walks from the living room to the
    bedroom goes on opening every later shot with "A living room." while the beat
    has them on the bed. The shot then holds two places at once, and the picture
    settles on whichever the model weighs more heavily, differently each time.
    That is a scene that keeps changing and resetting.

    The author's scene text is NOT edited. This states where the shot is now, and
    only where that disagrees with what the scene says, so a script that never
    moves is untouched and costs nothing."""
    here = (here or "").strip().lower()
    if not here:
        return ""
    txt = str(scene or "")
    if not txt.strip():
        return ""
    # Nothing to correct if the scene already names this room.
    if re.search(r"\b" + re.escape(here) + r"\b", txt, re.I):
        return ""
    if not _PLACE_WORD.search(txt):
        return ""
    return (f" This shot takes place in the {here}: the walls, floor, light and "
            f"furniture are the {here}'s throughout.")


_NOT_A_DESTINATION = frozenset("""
door doors doorknob handle window windows curtain curtains blind blinds mirror
sink basin bath tap taps table desk counter worktop bench chair seat stool sofa
couch armchair bed mattress headboard pillow cushion duvet quilt sheets blanket
cupboard cabinet drawer drawers shelf shelves wardrobe closet locker lockers
fridge freezer oven stove hob kettle microwave dishwasher washer machine
floor ground ceiling wall walls rail railing bannister bars bar post pole fence
light lights lamp switch socket screen tv television phone radio speaker camera
bag bags box crate case suitcase trunk basket bin sack tray bottle glass cup
car van truck bike motorbike trailer boat seat
edge middle centre center side sides end front back rear top bottom corner corners
spot position point row line queue
girl boy man woman lady guy person stranger guard nurse doctor teacher
hand hands arm arms elbow shoulder shoulders knee knees foot feet leg legs lap
face mouth lips chin neck throat hair head chest breast breasts stomach belly
waist hip hips thigh thighs wrist wrists ankle ankles bum butt crotch
""".split())

_MOVES_TO_ANY = re.compile(
    r"\b(?:to|into|toward|towards|inside|through\s+to|"
    r"enters?|entered|entering|steps?\s+into|stepped\s+into)\s+"
    + _DET_POSS + r"\s+((?:(?!(?:and|or|then|but|while|as|before|after|with|for|"
    r"to|into|onto|from|at|on|in|of)\b)[A-Za-z][\w-]*\s+){0,2}"
    r"(?!(?:and|or|then|but|while|as)\b)[A-Za-z][\w-]*)\b", re.I)


def moved_to(beat, people=()):
    """Where this beat MOVES somebody, whatever the place is called. "" if nowhere.

    The place-list readers answer first and more richly, because a known room can be
    named at both ends of the journey. This is what answers when they cannot: a
    travel verb, a destination, and a head noun that is not furniture, a body part,
    a vehicle or a person. It establishes NO room state -- it does not decide cuts,
    it does not feed where_hold or the acoustics, and it never claims to know what
    kind of space it is. All it does is make the arrival be PERFORMED, which is the
    one thing the reported failure was missing."""
    b = _DIALOGUE_TAG.sub(" ", _QUOTED.sub(" ", str(beat or "")))
    if not _TRAVEL_VERB.search(b):
        return ""
    names = {str(n).strip().lower() for n in (people or ()) if str(n).strip()}
    for m in _MOVES_TO_ANY.finditer(b):
        dest = re.sub(r"\s+", " ", m.group(1)).strip()
        head = dest.split()[-1].lower().strip("-")
        if (len(head) < 3 or head in _NOT_A_DESTINATION or head in names
                or dest.lower() in names or _EXTRA_PEOPLE.search(dest)):
            continue
        return dest.lower()
    return ""


_GOES_BACKWARD = re.compile(
    r"\b(?:backwards?|in\s+reverse|backs?\s+(?:away|out|up|off)|backing\s+(?:away|out|up)|"
    r"retreats?|retreating|reverses?|reversing)\b", re.I)


def facing_phrase(beat=""):
    """"each body facing the way it goes", unless the beat walks backwards."""
    return "" if _GOES_BACKWARD.search(str(beat or "")) else " each body facing the way it goes"


def move_clause(dest, beat=""):
    """Perform an arrival the place list cannot name. "" when there is nowhere."""
    if not dest:
        return ""
    facing = facing_phrase(beat)
    return (f" The shot travels to the {dest} on screen, the whole move played out "
            f"from its first step to its last,{facing + ' and' if facing else ''} the "
            f"{dest} nearer at the last frame than at the first.")


def travel_anchor(frm, via, to, here="", beat=""):
    """Say where the shot starts, what it passes, and where it ends. "" if nowhere.

    `here` is the room an earlier beat established, used when the beat names no
    origin -- a journey with only a destination is what renders as a cut.

    Short on purpose: this lands on travel beats, which already carry an action,
    and the node's whole balance problem is continuity crowding the beat out."""
    start = frm or here
    if not to or start == to:
        return ""
    facing = facing_phrase(beat)
    walk = ("the walk between them played out on screen, every step in frame"
            + (f",{facing}." if facing else "."))
    if via:
        return (f" The shot opens in the {start}, carries along the {via}, and "
                f"arrives in the {to}, {walk}") if start else (
                f" The shot carries along the {via} and arrives in the {to}, "
                f"{walk}")
    if not start:
        return (f" The shot enters the {to} on screen: the way in first, then the "
                f"{to} itself, the arrival played out, every step in frame"
                + (f",{facing}." if facing else "."))
    return f" The shot opens in the {start} and arrives in the {to}, {walk}"


_IS_IN = re.compile(r"\b(?:in|inside|within|at)\s+(?:the|her|his|their|a)\s+"
                    + _MOD + r"(" + _PLACE + r")\b", re.I)


def first_place(text):
    """The first place this text names at all. "" when it names none.

    place_named wants "in the kitchen"; a scene paragraph is more often just "A
    living room." with no preposition to hang on."""
    for m in _PLACE_WORD.finditer(str(text or "")):
        got = re.sub(r"\s+", " ", m.group(0)).strip().lower()
        if got in _PLACE_ALSO_A_VERB:
            continue
        if got == "room":
            before = re.search(r"(\w+)\s+$", str(text or "")[:m.start()])
            word = before.group(1).lower() if before else ""
            if word in _NOT_A_ROOM_MODIFIER or not word:
                continue
            return (word + " room")
        return got
    return ""


_AIMED_AT = re.compile(
    r"\b(?:look|stare|glance|gaze|peer|point|gestur|wave|nod|shout|call|yell|squint|"
    r"glare|beckon|aim)\w*"
    r"(?:\s+(?:out|over|up|down|back|across|around|round|through|away|off|in))*"
    r"(?:\s+(?:of|through|from|across|over)\s+(?:the|a|an|her|his|their)\s+[\w-]+"
    r"(?:\s+[\w-]+)?)?\s*$", re.I)


def place_named(text):
    """The place this text says somebody is IN, without travelling. "" if none."""
    text = str(text or "")
    for m in _IS_IN.finditer(text):
        if _AIMED_AT.search(text[:m.start()]):
            continue
        return re.sub(r"\s+", " ", m.group(1)).strip().lower()
    return ""


def rooms_named(text):
    """Every room this text names, lowercased. "" -> [].

    first_place returns only the FIRST one, which is what a tracked position needs.
    A scene paragraph often names two -- "Her bedroom has an unmade bed. The kitchen
    is small." -- and deciding whether a SENTENCE is about the room we are in means
    accounting for all of them.

    The same two exclusions as first_place, for the same reasons: a word that is also
    an ordinary verb cannot win in free text with no preposition in front of it, and a
    bare "room" names nowhere unless the word in front qualifies it."""
    out = []
    s = str(text or "")
    for m in _PLACE_WORD.finditer(s):
        got = re.sub(r"\s+", " ", m.group(0)).strip().lower()
        if got in _PLACE_ALSO_A_VERB:
            continue
        if got == "room":
            before = re.search(r"(\w+)\s+$", s[:m.start()])
            word = before.group(1).lower() if before else ""
            if word in _NOT_A_ROOM_MODIFIER or not word:
                continue
            got = word + " room"
        if got not in out:
            out.append(got)
    return out


def split_sheet(scene, names=()):
    """(everything that is not a character sheet entry, the sheet entries).

    WHAT LEADS A PROMPT DECIDES ITS COMPOSITION. This file already recorded that --
    "anatomy in the opening tokens is what a distilled LoRA settles composition on",
    which is why the gaze clause was moved to follow the beat rather than lead it --
    and then left the biggest anatomy block in the prompt leading every shot: the
    character sheet. "McKenna: she, 22, tall, long blonde hair, blue eyes, freckles"
    is sixteen words of face, it has to be in every shot because clothing continuity
    needs it there, and it sat in front of the action.

    Measured on a volleyball beat: 69% of the shot's words were in sentences about a
    face, and turning off every face guard only took that to 63%, because the sheet is
    most of it. Reported across many attempts as the camera fixated on one character
    staring into the lens -- with no reference image, no LoRA and a pinned first frame,
    none of which touched it, because none of them were what was leading the prompt.

    Splitting lets the scene keep the front, the beat follow it, and the appearance
    come after the thing it is describing. The words are identical; only the order
    changes, which is the one thing about this that was never tried."""
    cast = [str(n).strip() for n in (names or ()) if str(n).strip()]
    rest, sheet = [], []
    for raw in str(scene or "").split("\n"):
        for unit in re.split(r"(?<=[.!?])\s+", raw):
            u = unit.strip()
            if not u:
                continue
            if cast:
                entry = any(re.match(r"^" + re.escape(n) + r"\s*:", u, re.I) for n in cast)
            else:
                entry = ":" in u
            (sheet if entry else rest).append(u)
    return " ".join(rest), " ".join(sheet)


_CLAUSE_END = r"(?<=[.!?;])\s+"


def scene_for_here(scene, here, always="", names=(), beat=""):
    """(text to send, rooms held back, True if it declined to hold anything).

    THE SCENE PARAGRAPH IS STAMPED INTO EVERY SHOT, and it has to be -- a removal
    needs the text to have something to scrub, and where_hold's own comment says the
    paragraph "still names the room they started in and is stamped into every shot".
    But a paragraph that describes the opening ROOM describes its FURNITURE too, and
    furniture does not travel. Reported: a flat whose scene paragraph read "Her
    bedroom has an unmade bed and a lamp", a walk from the bedroom down the hallway
    to the living room, and then A BED IN THE LIVING ROOM. where_hold had the room's
    NAME right in every shot; the bed was in the text standing beside it, and at cfg 1
    there is no negative prompt that can take a named thing back.

    THE ROOM THE SHOT ENDS IN decides this, not every room it passes through, and the
    difference is the whole fix. A walk out of the bedroom genuinely shows the bedroom
    in its opening frames -- but that shot's LAST frame is the next shot's keyframe, so
    a bed drawn at the end of the walk is inherited by the shot after it, which is the
    second route the same bed took into the living room. Nothing is lost by holding it
    there: the opening room arrives as a PICTURE regardless, because the keyframe is
    the previous shot's last frame and that frame IS the room being left. So the words
    describe where the shot ends and the frame carries where it began.

    A WITHHOLDING, NOT AN EDIT, exactly like the covered-garment deferral: the
    author's paragraph is untouched, every reader inside this file still sees all of
    it, this is only what the model is told for THIS shot, and a beat that walks back
    into the bedroom gets the bed back in full.

    TWO GUARDS. A sentence carrying a LABEL -- "McKenna: she, 22, ..." -- is a
    character sheet entry and is never touched whatever it names, because losing a
    person's line is the failure hide_item exists to prevent. And if holding would
    leave the shot no scene sentence at all, nothing is held: a sentence that welds
    the film's own framing to one room's furniture ("A small flat at night, her
    bedroom with an unmade bed") would otherwise take the night away with the bedroom,
    and a shot with no scene is a bigger change than the bug. The caller reports that
    case so the author can split the sentence.

    `always` IS THE ANCHOR AND IS NEVER HELD. build_scene fuses the anchor and the
    scene paragraph into one string before either reaches a shot, and an anchor is
    documented as what belongs to the WHOLE film -- "look, camera, lighting,
    location". So an anchor reading "Shot on 35mm in a cramped kitchen" names a room,
    and without this it was held on every shot outside that kitchen: the film lost its
    stock and its lens to a rule about furniture. The anchor's sentences are spared by
    text, which survives terminate_lines adding a full stop to them.

    `names` IS THE DECLARED CAST, and it is what identifies a sheet entry. A bare
    colon test was the first guard and it had a hole both ways: "Her bedroom: an
    unmade bed and a lamp." is the author describing a room, not a person, and it was
    protected as though it were somebody's line -- so the bed survived in that
    phrasing. Matching the LABEL against a name the sheet actually declares closes it
    without ever risking a person: with no cast passed it falls back to protecting any
    colon, because losing somebody's line is worse than a bed in one shot.

    A ROOM THE BEAT ITSELF NAMES IS NEVER HELD. `here` goes stale whenever the beat's
    verb is not one the movement readers know -- "McKenna pads into the kitchen" moves
    nobody as far as place_in is concerned -- and a stale room would hold the
    description of the room the shot is actually IN. The beat's own words outrank
    anything inferred from them, which is this file's standing rule, so a sentence
    about a room the beat mentions stays whatever the tracked room says.

    Holding NOTHING returns the text unchanged, byte for byte, so a script that never
    leaves one room is untouched and costs nothing."""
    text = str(scene or "")
    room = (here or "").strip().lower()
    if not text.strip() or not room:
        return text, [], False, []
    cast = [str(n).strip() for n in (names or ()) if str(n).strip()]

    def _is_sheet_entry(unit):
        u = unit.strip()
        for n in cast:
            if re.match(r"^" + re.escape(n) + r"\s*:", u, re.I):
                return True
        return bool(not cast and ":" in u)

    beat_rooms = set(rooms_named(
        _DIALOGUE_TAG.sub(" ", _QUOTED.sub(" ", str(beat or "")))))
    spared = set()
    for unit in re.split(_CLAUSE_END, str(always or "")):
        u = unit.strip().rstrip(".!?; ").lower()
        if u:
            spared.add(u)
    lines, held, held_text, survived = [], [], [], False
    for raw in text.split("\n"):
        kept, cut_here = [], False
        for unit in re.split(_CLAUSE_END, raw):
            # A sheet entry. Never touched.
            if _is_sheet_entry(unit):
                kept.append(unit)
                continue
            # ...nor anything the anchor said. It frames the whole film.
            if unit.strip().rstrip(".!?; ").lower() in spared:
                kept.append(unit)
                survived = True
                continue
            named = rooms_named(unit)
            if named and room not in named and not (beat_rooms & set(named)):
                for r in named:
                    if r not in held:
                        held.append(r)
                if unit.strip() not in held_text:
                    held_text.append(unit.strip())
                cut_here = True
                continue
            kept.append(unit)
            survived = True
        if cut_here:
            mended = []
            for unit in (k for k in kept if k.strip()):
                unit = re.sub(r";$", ".", unit.strip())
                if (unit[:1].islower()
                        and ((mended and mended[-1].endswith(".")) or not mended)):
                    unit = unit[0].upper() + unit[1:]
                mended.append(unit)
            kept = mended
        lines.append(" ".join(k for k in kept if k.strip()))
    if not held:
        return text, [], False, []
    if not survived:
        return text, held, True, held_text
    return "\n".join(l for l in (s.strip() for s in lines) if l), held, False, held_text


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
    out, seen = [], set(state_acts(text))
    for pat, order in ((_STATE_ADJ, "sn"), (_STATE_PRED, "ns")):
        for m in pat.finditer(text or ""):
            if order == "sn":
                state, gap, thing = m.group(1), m.group(2), m.group(3)
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


def renumber_reference_tags(text, wired):
    """Rewrite <Picture N> from INPUT number to position in the reference roster.

    The roster is packed dense -- the wired images become picture 1, 2, 3 in the
    order of their sockets -- but nobody writing a sheet knows that. They write the
    number on the socket, which is what the README documents. Wire ref_image_1 and
    ref_image_3 and the two conventions disagree: <Picture 3> names nothing in a
    roster of two, so the tag was stripped and the image was dropped in silence.

    `wired` is the socket numbers that actually have an image, in socket order. With
    no gaps this is the identity mapping and nothing changes, which is why the fault
    stayed hidden -- everybody fills the sockets from the top until they don't."""
    seat = {slot: i + 1 for i, slot in enumerate(wired)}
    if not text or all(k == v for k, v in seat.items()):
        return text
    return _PICTURE_TAG.sub(
        lambda m: (f"<Picture {seat[int(m.group(1))]}>"
                   if int(m.group(1)) in seat else m.group(0)), text)


def unwired_reference_tags(text, wired):
    """Tag numbers naming a socket with no image on it. Sorted, no repeats."""
    return sorted({int(m.group(1)) for m in _PICTURE_TAG.finditer(text or "")}
                  - set(wired or ()))


def handoff_rides_as_ref(handoff, refs, ref_noise_aug):
    """Is the shot handoff about to be encoded as a subject reference?

    Mirrors the demotion in build_conditioning. Read in the render loop as well,
    because the text has to claim the picture and the text is written up there."""
    return bool(handoff is not None and refs
                and not (ref_noise_aug is None
                         or float(ref_noise_aug) >= KEYFRAME_SAFE_AUG))


def handoff_claim(n):
    """Name the demoted handoff as this shot's opening frame.

    Below KEYFRAME_SAFE_AUG the handoff stops being a keyframe and is encoded as an
    extra reference -- and it was going in unclaimed, on the reasoning that a first
    frame is not a subject and needs no tag. It needs one HERE. In the reference
    rows it is not a first frame any more, it is picture N of N, and the rule that
    governs those is the node's oldest: a picture the prompt names is that subject,
    and a picture it never names is ANOTHER subject.

    So the last shot of a run carried a second person wearing the previous shot's
    clothes and face -- reported as a duplicate at the end of the video, and only
    ever below 0.99, which is why lowering the aug to strengthen identity was what
    produced the twin."""
    return (f" <Picture {n}> is the frame this shot opens on: the same place and the "
            f"same people, one moment earlier, carried forward rather than joined by "
            f"anybody new.")


_COUNT_ONE = " There is one person in the shot: one body, one face."
_COUNT_TWO = " There are two people in the shot, with one body for each person."


def recount_with_claim(prompt, described, added):
    """Put the people a carried frame's claim NAMES into that shot's body count.

    The count is written a whole phase before the carry is decided -- the carry needs
    a frame to carry, and the plan has none -- so it only ever counted the cast the
    shot describes. A shot of McKenna alone then went out saying "There is one person
    in the shot: one body, one face" AND carrying a picture of the shot before it,
    claimed as "Dan is the person there. McKenna is in this room too". One body
    counted, two people named, and a photograph of the one who is not counted. A
    picture is a subject and no sentence outranks it: reported as duplicate
    characters, and as a face arriving in a shot that never asked for it.

    The objection recorded against counting the people a FRAME carries was that it
    "asserted bodies the text could not identify at all, and that is how a stranger
    arrives". That is the whole difference here: this claim names them.

    Three or more is left uncounted, which is what cast_hold does with the same
    number and for the same reason -- the more people the count holds, the likelier
    one of them is described without being in frame, and asserting four bodies is a
    request for a fourth."""
    total = list(dict.fromkeys([n for n in (described or []) if n]
                               + [n for n in (added or []) if n]))
    if not added or len(total) < 2 or _COUNT_ONE not in prompt:
        return prompt
    return prompt.replace(_COUNT_ONE, _COUNT_TWO if len(total) == 2 else "", 1)


def room_claim(n, present, joining):
    """Claim a handoff carried as a reference because somebody NEW is in the shot.

    The keyframe used to be thrown away here, and throwing it away is what the
    node's own note in build_conditioning warns about: with no handoff the VLM is
    never shown where the shot left off and re-imagines the scenery -- same place,
    new room. Reported as the scene not staying the same between shots.

    A keyframe and a reference are different instruments. A keyframe IS frame one,
    so a newcomer absent from it has to walk in from nowhere, which is the bug the
    fresh start was for. A reference only supplies appearance, so the same picture
    carries the room and the people already in it while the newcomer is simply
    there at the first frame.

    Claimed, and specifically. An unclaimed picture of somebody is another person
    who looks like them, and the standing claim is worse than nothing here: it says
    the shot is joined by nobody new, in the one case where it is."""
    said = (f" <Picture {n}> is this room a moment earlier: the same walls, floor, "
            f"furniture and light, from the same camera.")
    if present:
        said += (f" {' and '.join(present)} "
                 f"{'are the people' if len(present) > 1 else 'is the person'} there.")
    if joining:
        said += (f" {' and '.join(joining)} {'are' if len(joining) > 1 else 'is'} in "
                 f"this room too, already in place at the first frame.")
    return said


def carried_people_claim(n, present, was_room="", now_room=""):
    """Claim the previous frame carried as a reference across a cut to another room.

    The people come with it and the room does not. Naming both rooms is what keeps
    the picture from pulling the old walls in: it says where the picture was taken
    and where this shot is."""
    who = _join_names(present)
    said = (f" <Picture {n}> is {who} a moment earlier"
            f"{f', in the {was_room}' if was_room else ''}: the same "
            f"{'faces, hair and clothes' if len(present) > 1 else 'face, hair and clothes'}.")
    if now_room:
        said += f" This shot is in the {now_room}."
    return said


def returning_room_claim(n, room, present, arriving):
    """Claim a frame of a room the film showed before and has come back to.

    Its own claim, not room_claim's: that one says "a moment earlier", and this
    picture is from shots ago. It names who is in it, because an unclaimed person
    in a picture is another person."""
    said = (f" <Picture {n}> is the {room} as the film last showed it"
            f"{', where this shot arrives' if arriving else ''}: the same walls, floor, "
            f"furniture and light.")
    if present:
        said += (f" {_join_names(present)} "
                 f"{'are the people' if len(present) > 1 else 'is the person'} in it.")
    return said


def _join_names(names):
    """"Nora", "Nora and Dan", "Nora, Dan and Mara" -- a list a reader can read."""
    names = [str(n) for n in (names or []) if str(n).strip()]
    if len(names) < 2:
        return names[0] if names else ""
    return ", ".join(names[:-1]) + " and " + names[-1]


def plate_claim(n):
    """Claim shot 1's first_frame when it is carried as the SET rather than frame one.

    room_claim cannot serve here and saying so is the point: it calls the picture
    "this room a moment earlier" and names who was standing in it, and on shot 1
    there is no earlier and nobody was. A plate is a picture of a place with no
    people in it, and the claim has to say exactly that -- an unclaimed picture is
    read as another subject, and a picture of an empty room claimed as a person is
    how a figure gets invented to stand in it."""
    return (f" <Picture {n}> is the set this shot takes place in: the same walls, "
            f"floor, furniture and light, from the same camera. It is a picture of "
            f"the place only, with nobody in it -- the people in this shot are the "
            f"ones named above, standing where the text puts them.")


def state_hold(pairs):
    """One sentence putting those states at the first frame instead of in the action.

    Two at most. These sentences are continuity, and continuity that outgrows the
    beat is what the beat stops being about."""
    said = []
    for thing, state in pairs[:2]:
        plural = thing.endswith("s")
        said.append(f"The {thing} {'are' if plural else 'is'} already {state} at the "
                    f"first frame and {'stay' if plural else 'stays'} {state} for the "
                    f"whole shot.")
    return (" " + " ".join(said)) if said else ""


def rigid_hardware(text):
    """Is the hardware here the kind that cannot flex?"""
    return bool(_RIGID_HARDWARE.search(text or ""))


def _merely_handled(part, shown=True):
    """Does this clause MOVE hardware without fastening it to anybody?

    A restraint is an object before it is a restraint, and an object can be picked
    up, dropped in a toolbox or thrown on a bench. Nothing in the clause fastens it
    to a body, holds a body part, or ties it to anything that does not move.

    `shown` counts holding one UP as handling it, which is right for the hold and
    wrong for the clause that says where a piece of hardware sits."""
    moved = _HANDLING_VERB.search(part) or (shown and _SHOWN_VERB.search(part))
    return bool(moved) and not (
        _BINDING_VERB.search(part) or _BODY_PART.search(part)
        or _ANCHOR_POINT.search(part))


def hardware_handled(text):
    """Is every mention of hardware here a mention of it being CARRIED?

    False when the text names no hardware at all: this answers "is it only being
    handled", not "is there none"."""
    seen = False
    for part in re.split(r"(?<=[.;!?])\s+", str(text or "")):
        if not (_RESTRAINT_PLAIN.search(part) or _RESTRAINT_MAYBE.search(part)):
            continue
        if not _merely_handled(part, shown=False):
            return False
        seen = True
    return seen


def restraint_present(text):
    """Is a restraint being applied or worn, in this text?

    Plain hardware counts on its own. Ambiguous hardware needs a binding verb or a
    body part alongside it, so a chain-link fence and a leather belt do not arm a
    continuity rule about restraints."""
    t = text or ""
    if engine.applies_hardware(t) and (_BODY_PART.search(t)
                                       or engine._APPLIED_TO_PRONOUN.search(t)):
        return True
    for part in re.split(r"(?<=[.;!?])\s+", t):
        if not _RESTRAINT_PLAIN.search(part):
            continue
        if _merely_handled(part):
            continue
        return True
    for part in re.split(r"(?<=[.;!?])\s+", t):
        if _RESTRAINT_MAYBE.search(part) and (_BINDING_VERB.search(part)
                                              or _BODY_PART.search(part)):
            return True
        if _RESTRAINT_MAYBE.search(part) and _ANCHOR_POINT.search(part):
            return True
    return False


def names_any(text, tokens):
    """Does `text` name any of these items?"""
    return any(re.search(r"\b" + re.escape(t) + r"\b", text or "", re.I)
               for t in (tokens or []) if t)


def person_tags(text, objects=None):
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
        if not before.endswith(":") and w:
            head = w.group(1)[:1]
            if head.isalpha() and head.islower():
                continue                      # the picture belongs to the object
        if objects and w is None and not before.endswith(":"):
            ahead = (text[m.end():]).lstrip()
            if any(re.match(r"(?:(?:a|an|the|her|his|their)\s+)?(?:[\w-]+\s+){0,2}"
                            + re.escape(o) + r"\b", ahead, re.I) for o in objects if o):
                continue
        out.append(m.group(1))
    return out


_OBJECT_END = re.compile(r"(?:,|;|\.|\bexposing\b|\brevealing\b|\bshowing\b|\bleaving\b|"
                         r"\bto\s+expose\b|\bto\s+reveal\b|\bthen\b|\buntil\b)", re.I)

_NOT_A_GARMENT = frozenset("""
the a an and or her his its their our your this that these those
"""
             """
she he him them they us we you one both each either neither
herself himself themselves myself yourself itself
somebody someone anybody anyone nobody everybody everyone
off from over under onto into out down up away through across behind
front side left right rest way bit end edge
back neck chest waist hips hip wrist wrists ankle ankles arm arms hand hands
leg legs thigh thighs knee knees foot feet shoulder shoulders head face mouth
lips hair skin body torso stomach belly chin jaw eyes ear ears
floor ground wall room air
"""
             """
shower showers bath baths bathtub tub tubs basin sink sinks toilet loo cubicle
stall stalls bed beds sofa sofas couch couches chair chairs stool stools bench
seat seats armchair table tables desk desks counter shelf shelves cupboard
cabinet drawer drawers door doors doorway window windows mirror curtain
car cars cab taxi van truck lift elevator stairs step steps
kitchen bathroom bedroom hallway corridor landing garden street pavement
water pool puddle steam tiles tile mat mats rug rugs carpet basket hamper
""".split())

_ENTRY_END = re.compile(r"^\s*(?:[,;.!?]|$|(?:and|over|under|beneath|above|with|plus)\b)",
                        re.I)

_RESTRAINT_WORD = engine._NOT_CLOTHING


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
        tail = _LEADING_TAG.sub("", scene[m.end():], count=1)
        if _ENTRY_END.match(tail):
            return True
    return False


def _modifier_of_a_named_entry(word, span, scene):
    """Is `word` a MODIFIER of a longer garment the same span already names?

    "Dan pulls off her jeans shorts" names one garment. But "jeans" is also the
    head of Dan's own entry, so the reader matched it against his line and took
    HIS jeans off as well -- in a beat that never mentions him. His trousers came
    off automatically, and stayed off.

    The span is what the beat says is coming off. If the word is immediately
    followed there by another garment word, it is describing that one, not naming
    a second: the phrase is "jeans shorts", and only "shorts" is the head.
    """
    m = re.search(r"\b" + re.escape(word) + r"\b\s+([\w-]{3,})", span or "", re.I)
    if not m:
        return False
    nxt = m.group(1).lower().strip("-")
    return bool(nxt and nxt not in _NOT_A_GARMENT and _is_entry_head(nxt, scene))


_DISPLACE = engine._DISPLACE
scene_name_for = engine.scene_name_for
displaced_garments = engine.displaced_garments
puts_it_back = engine.puts_it_back
restored_garments = engine.restored_garments


def displaced_hold(items):
    """Say where a moved garment now sits, so the next shot does not put it back.

    Without this the garment is described by the sheet in the state it was WORN, and
    the sheet is re-stamped into every shot -- so shorts pulled down are pulled back
    up by the next beat, or come back looking like a different pair."""
    if not items:
        return ""
    said = ", ".join(f"the {thing} {how}" for thing, how in items[:2])
    return f" On the body and {said}, left exactly where the beat put them."


_ASK_VERB = (r"asks?|asked|asking|begs?|begged|begging|pleads?|pleaded|pleading|"
             r"wants?|wanted|wishes|wished|tells?|told|orders?|ordered|demands?|"
             r"demanded|whispers?|whispered|says?|said|shouts?|shouted|screams?|"
             r"screamed")
_REQUEST = re.compile(
    # "asks him TO take it off"
    r"\b(?:" + _ASK_VERB + r")\b[^.;!?]{0,60}?\bto\s+(?=[a-z])"
    # "asks FOR the belt to come off"
    r"|\b(?:asks?|asked|begs?|begged|pleads?|pleaded)\b[^.;!?]{0,40}?\bfor\b"
    r"|\b(?:asks?|asked|asking|wonders?|wondered)\b[^.;!?]{0,40}?\b(?:if|whether)\b",
    re.I)


def _in_quotes(text, at):
    """Is position `at` inside a span of dialogue?"""
    return any(m.start() <= at < m.end() for m in _QUOTED.finditer(text or ""))


_QUESTION = re.compile(r"[^.;!?]*\?")


def _in_a_question(text, at):
    """Is the removal verb at `at` inside a sentence that ends in a question mark?"""
    return any(m.start() <= at < m.end() for m in _QUESTION.finditer(text or ""))


def _in_a_request(text, at):
    """Is the removal verb at `at` inside a request rather than an action?"""
    # Asked in someone's own words, or asked as a question: either way, not done.
    if _in_quotes(text, at) or _in_a_question(text, at):
        return True
    start = max((m.end() for m in _REQUEST.finditer(text or "") if m.end() <= at),
                default=None)
    if start is None:
        return False
    stop = re.search(r"[.;!?]|,\s*(?:and|then|so|but)\b", (text or "")[start:])
    return at <= (start + stop.start() if stop else len(text or ""))


_PRONOUN_OBJECT = re.compile(r"\s*(?:it|them|these|those)\b", re.I)
_SENTENCE_BREAK = re.compile(r"[.;!?]\s+")


def _sentence_before(beat, at):
    """The sentence `at` is in, up to `at`. The pronoun's antecedent lives here.

    A beat is a paragraph and can hold several sentences. "It" reaches back across
    a comma or an "and", not across a full stop."""
    cut = max((m.end() for m in _SENTENCE_BREAK.finditer(beat[:at])), default=0)
    return beat[cut:at]


_GARMENT_FAMILIES = (
    ("shoes", "boots", "sneakers", "trainers", "heels", "sandals", "loafers", "slippers",
     "flats", "pumps", "clogs", "brogues", "moccasins", "espadrilles", "wedges"),
    ("sweater", "jumper", "sweatshirt", "hoodie", "pullover", "cardigan"),
    ("coat", "jacket", "parka", "blazer", "overcoat", "raincoat", "anorak", "windbreaker",
     "peacoat"),
    ("top", "shirt", "blouse", "tee", "t-shirt", "tshirt", "camisole"),
    ("trousers", "pants", "jeans", "slacks", "chinos", "joggers", "sweatpants"),
    ("hat", "cap", "beanie", "beret"),
    ("gloves", "mittens"),
)
_GARMENT_KIN = {word: tuple(w for w in family if w != word)
                for family in _GARMENT_FAMILIES for word in family}


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
        # Asked for is not done. See _in_a_request.
        if _in_a_request(beat, m.start()):
            continue
        if re.fullmatch(_OPENER_VERB, m.group(0), re.I):
            _rest = re.split(r"[.;!?]", beat[m.end():])[0]
            if not (_FINISHES_REMOVAL.search(_rest) or restraint_present(_rest)):
                continue
        _before = len(found)
        tail = beat[m.end():]
        cut = _OBJECT_END.search(tail)
        span = tail[:cut.start()] if cut else tail
        if not (re.fullmatch(_UNDO_VERB, m.group(0), re.I)
                or re.search(r"\b(?:off|away|out\s+of|down)$", m.group(0), re.I)):
            part = re.search(r"\b(?:off|away)\b", span, re.I)
            if part:
                if _HAS_VERB.search(span[:part.start()]):
                    continue
                span = span[:part.start()]
        for word in re.findall(r"\b[\w-]{3,}\b", span):
            low = engine.singular_garment(word)
            if not low or low in found:
                continue
            # Grammar, prepositions and anatomy are not garments.
            if low in _NOT_A_GARMENT:
                continue
            # Hardware is cleared by an explicit `remove:` and by nothing else.
            if _RESTRAINT_WORD.match(low):
                continue
            if not _is_entry_head(low, scene):
                _kin = [k for k in _GARMENT_KIN.get(low, ()) if _is_entry_head(k, scene)]
                if len(_kin) != 1 or _kin[0] in found:
                    continue
                low = _kin[0]
            if _modifier_of_a_named_entry(word, span, scene):
                continue
            # ...and not a person or a place.
            if re.search(r"\b" + re.escape(word) + r"\b\s*(?:is|was|walks|stands|sits|=)",
                         scene, re.I):
                continue
            found.append(low)
        if len(found) == _before and _PRONOUN_OBJECT.match(span):
            _near = []
            for _g in garments_in(_sentence_before(beat, m.start())):
                _low = engine.singular_garment(_g) or _g
                if (_low in _NOT_A_GARMENT or _RESTRAINT_WORD.match(_low)
                        or not _is_entry_head(_low, scene) or _low in _near):
                    continue
                _near.append(_low)
            if len(_near) == 1 and not any(engine._garment_key(x)
                                           == engine._garment_key(_near[0])
                                           for x in found):
                found.append(_near[0])
    shown_off = exposed_by(beat, scene)
    return [f for f in found if f not in shown_off]


garments_in = engine.garment_words
region_of = engine.region_of


_NAKED_CUE = engine.STRIPS_BARE


def strips_who(beat, cast):
    """Who this beat undresses. [] when it cannot tell.

    strips_bare only answers WHETHER somebody ends up with no clothes on. The
    wardrobe was then read off the whole shot sheet, so in a shot describing two
    people BOTH were stripped -- one character undressing made the other undress
    too. Reported as the second character mimicking the first.

    The subject is the name before the cue, the same reading posture_in uses. With
    one person in the shot there is nobody else it can be."""
    people = [n for n in (cast or []) if n]
    b = str(beat or "")
    if not people or not b:
        return []
    if len(people) == 1:
        return people[:1]
    m = _NAKED_CUE.search(b)
    if not m:
        return []
    before = b[:m.start()]
    cut = max((c.end() for c in
               re.finditer(r"[.;!?]\s+|,\s*|\s+(?:as|while|and then|then|but)\s+",
                           before)), default=0)
    span = before[cut:]
    here = [n for n in people
            if re.search(r"\b" + re.escape(n) + r"\b", span, re.I)]
    if here:
        return here
    # No name before it: the beat's own first-named person is acting.
    first = next((n for n in people
                  if re.search(r"\b" + re.escape(n) + r"\b", b, re.I)), None)
    return [first] if first else []


def strips_bare(text):
    """Does this beat say somebody ends up with no clothes on?"""
    return bool(_NAKED_CUE.search(text or ""))


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
        if _is_entry_head(word, scene):
            hits.append(low)
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

    body = _EXACT_LINE.sub("", _ADD_LINE.sub(take_added, _REMOVE_LINE.sub(take_removed, beat or "")))
    return re.sub(r"\n{2,}", "\n", body).strip(), removed, added


_PRINT_WORDS = re.compile(
    r"\b(?:print(?:ed|s)?|lettering|letter(?:s|ed)?|text|reads?|reading|says|"
    r"written|writing|embroider(?:ed|y)|emblazoned|stitched|stamped|logo|slogan|"
    r"motto|monogram(?:med)?|words?|font|spell(?:s|ed|ing)?|its|it)\b", re.I)
_QUOTED_SPAN = re.compile(r'["“][^"”]+["”]')
_CAPS_WORD = re.compile(r"\b[A-Z]{2,}\b")                           # case matters
_ON_GARMENT = re.compile(
    r"\b(?:across|on|along|down|over)\s+(?:the|its|her|his|their)\s+"
    r"(?:front|back|chest|waistband|hem|seat|rear|crotch|straps?|cups?|hips?|"
    r"bum|butt)\b", re.I)
_CONTINUES = re.compile(r"^\s*(?:with|across|along|down|over|on|at|bearing|"
                        r"reading|printed|lettered|emblazoned)\b", re.I)


def _continues_item(unit):
    return bool(_PRINT_WORDS.search(unit) or _QUOTED_SPAN.search(unit)
                or _CONTINUES.search(unit)
                or (_CAPS_WORD.search(unit) and _ON_GARMENT.search(unit)))


def hide_item(text, items):
    """Take the named items out of a sheet line, keeping everything else.

    SURGICAL, unlike scrub_removed, which drops the whole comma-separated
    fragment -- that is right for a garment that has come off and wrong here: it
    took "green dress, steel collar" down to nothing and the person's line with
    it, leaving shots with nobody described in them.

    This removes the item and the adjectives attached to it, and stops. A
    fragment that held only that item disappears; a fragment holding anything
    else keeps the rest. A fragment carrying the person's LABEL ("McKenna: she,
    22") never disappears, whatever else is in it."""
    if not text or not items:
        return text
    _NOT_PAST = (r"(?:\b(?!(?:over|under|underneath|beneath|above|below|with|and|"
                 r"plus|inside)\b)\w+[\w-]*\s+){0,3}?")
    pats = [re.compile(_NOT_PAST + r"\b" + re.escape(str(i).strip()) + r"\b",
                       re.I) for i in items if str(i).strip()]
    out_lines = []
    for line in str(text).split("\n"):
        frags, kept = line.split(","), []
        trailing = False        # the unit just before this one went with its garment
        entry = ":" in line     # a labelled sheet entry: where attribute lists live
        for frag in frags:
            units, kept_units = re.split(r"(?<=[.!?])\s+", frag), []
            for unit in units:
                new = unit
                for p in pats:
                    # The item, plus any adjectives sitting directly in front of it.
                    new = p.sub("", new)
                removed = new != unit
                if removed:
                    new = re.sub(r"\s*\b(?:over|under|underneath|beneath|above|below|"
                                 r"with|and|plus|inside)\b\s*(?=[,.;]|$)", "", new)
                if (removed and ":" not in unit
                        and not garments_in(new) and re.search(r"\w", new)):
                    trailing = True
                    continue
                if not re.sub(r"\b(?:a|an|the|and|with|in)\b|[\s,.;]", "", new):
                    if removed:
                        trailing = True
                    continue
                if (entry and not removed and trailing and ":" not in unit
                        and not garments_in(unit) and _continues_item(unit)):
                    continue
                kept_units.append(new)
                trailing = False
            if kept_units:
                kept.append(" ".join(kept_units))
        joined = ",".join(kept)
        # Tidy the seams the removal leaves: doubled commas and spaces.
        joined = re.sub(r"\s*,\s*,+", ",", joined)
        joined = re.sub(r"\s{2,}", " ", joined).strip()
        joined = re.sub(r",\s*([.;]|$)", r"\1", joined)
        joined = re.sub(r"([.!?])\s*,\s*", r"\1 ", joined)
        joined = re.sub(r":\s*,\s*", ": ", joined)
        joined = re.sub(r":\s*(?=[.;]|$)", "", joined)
        if (line.rstrip().endswith((".", "!", "?")) and joined
                and not joined.endswith((".", "!", "?"))):
            joined += "."
        out_lines.append(joined)
    return "\n".join(out_lines)


def strippers_in(beat, sheet):
    """Who this beat says takes something off. [] when it does not say.

    subjects_for with the compound subject vocal_sources_in already needed: "McKenna
    and Tess take off their shirts" shares ONE verb between two names, and the
    conjunction guard -- right about "Dan holds the door and McKenna undresses", where
    `and` opens a new predicate -- cannot tell that apart on its own. Getting this
    wrong in the narrowing direction would leave a garment on somebody who took it
    off, so both names are kept."""
    verbs = engine._STRIP_VERB + "|" + engine._UNDO_VERB
    b = str(beat or "")
    out = list(subjects_for(b, sheet, verbs))
    for n, _ln in sheet_lines(sheet):
        if not n or n in out:
            continue
        if re.search(r"\b" + re.escape(n) + r"\b(?:\s*,\s*[\w'\u2019-]+)*"
                     r"\s+and\s+[\w'\u2019-]+\s+(?:" + verbs + r")\b", b, re.I):
            out.append(n)
    return out


_ENTRY_SEP = re.compile(r"(\s+(?:over|under|beneath|underneath|on\s+top\s+of|and)\s+)", re.I)
_LAYER_SEP = re.compile(r"^\s+(?:over|under|beneath|underneath|on\s+top\s+of)\s+$", re.I)
_GARMENT_PART = {"sleeve", "sleeves", "hood", "collar", "lapel", "lapels", "pocket",
                 "pockets", "button", "buttons", "zip", "zipper", "lining", "trim",
                 "fringe", "laces", "hem", "neckline", "print", "logo", "stripes",
                 "pattern", "cuffs", "cuff", "straps", "strap", "buckle", "badge"}


def entry_parts(frag):
    """[(separator before it, text)] -- the garments one comma entry names, in order."""
    bits = _ENTRY_SEP.split(frag or "")
    parts = [("", bits[0])]
    for i in range(1, len(bits) - 1, 2):
        sep, text = bits[i], bits[i + 1]
        head = (re.findall(r"[a-z]+", text.lower()) or [""])[-1]
        joins_with = (not _LAYER_SEP.match(sep) and re.search(r"\bwith\b", parts[-1][1], re.I)
                      and head in _GARMENT_PART)
        if joins_with:
            parts[-1] = (parts[-1][0], parts[-1][1] + sep + text)
        else:
            parts.append((sep, text))
    return parts


def join_entry_parts(parts):
    """The entry again from the parts kept, the first one losing its separator."""
    out = ""
    for i, (sep, text) in enumerate(parts):
        out += (text if i == 0 else sep + text)
    return out.strip()


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
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        _person_tags = set(person_tags(sent))
        frags = sent.split(",")
        out_frags = []
        for frag in frags:
            if any(p.search(frag) for p in pats) and not _HAS_VERB.search(frag):
                if restraint_present(frag) and not any(_RESTRAINT_WORD.match(t)
                                                       for t in live):
                    out_frags.append(frag)
                    continue
                _parts = entry_parts(frag)
                gone = [t for _sep, t in _parts if any(p.search(t) for p in pats)]
                _kept_parts = ([(sep, t) for sep, t in _parts if t not in gone]
                               if len(_parts) > 1 else [])
                keep = [join_entry_parts(_kept_parts)] if _kept_parts else []
                tags = [n for s in (gone or [frag]) for n in picture_tags(s)
                        if str(n) in _person_tags]
                piece = " and ".join(k for k in keep if k.strip())
                if tags:
                    piece = ((piece + " ") if piece.strip() else "") + \
                            " ".join(f"<Picture {n}>" for n in tags)
                if piece.strip():
                    out_frags.append(piece)
                continue                      # the rest of the entry goes
            out_frags.append(frag)
        rebuilt = ",".join(out_frags)
        end = re.search(r"([.!?])\s*$", sent)
        if end and rebuilt.strip() and not re.search(r"[.!?]\s*$", rebuilt):
            rebuilt = rebuilt.rstrip().rstrip(",;") + end.group(1)
        kept.append(rebuilt)
    out = " ".join(k for k in kept if k.strip())
    for t in live:
        out = re.sub(r"(?:<\s*picture[\s_\-]*\d+\s*>\s*)?"
                     r"\b(?:(?:a|an|the|her|his|their)\s+)?(?:[\w-]+\s+){0,2}"
                     + re.escape(t) + r"\b(?:\s*<\s*picture[\s_\-]*\d+\s*>)?",
                     "", out, flags=re.I)
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
        out = re.sub(r",(?=[^\s,\d])", ", ", out)
        out = re.sub(r"(,\s*)(?:and|or)\s+", lambda m: m.group(1), out, flags=re.I)
    out = re.sub(r"\s{2,}", " ", out)
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+", out):
        s = sent.strip()
        if not re.search(r"[A-Za-z0-9]", s):
            continue
        if re.fullmatch(r"(?:he|she|they|it|[A-Z][\w-]*)"
                        r"(?:\s+(?:is|are|was|were|has|have|had))?\s*[.!?]?",
                        s, re.I):
            continue
        kept.append(s if s[-1] in ".!?" else s + ".")
    return " ".join(kept).strip()


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


def latent_upscaler_node():
    return _find_node(["minimaxh3latentupscaler", "3d"]) or _find_node(["minimaxh3latentupscaler"])


def landing_schedule(model, scheduler, steps, shift_video, shift_audio):
    """The sigmas this shot will run on, with the audio landing added. None if not.

    None means "take the ordinary path and change nothing", and it is returned for
    every reason there is: comfy not reachable, a schedule that already lands
    softly, anything unexpected. The schedule is built the way KSampler builds it --
    calculate_sigmas(model_sampling, scheduler, steps) at denoise 1.0 -- so what is
    handed back is the shot's own schedule with one step spliced into the end,
    never a different one."""
    try:
        import comfy.samplers as _cs
        _ms = model.get_model_object("model_sampling")
        base = [float(x) for x in _cs.calculate_sigmas(_ms, str(scheduler), int(steps))]
        landed = insert_audio_landing(base, shift_video, shift_audio)
        if len(landed) == len(base):
            return None
        return torch.tensor(landed, dtype=torch.float32)
    except Exception:
        return None


def sample_shot(model, cond, negative, latent, seed, steps, cfg, sampler_name,
                scheduler, sigmas=None, shift_video=None, shift_audio=None,
                soft_landing=False):
    """One sampling pass. denoise is fixed at 1.0: partial denoise desyncs the
    joint audio/video schedule."""
    if sigmas is not None and len(sigmas):
        return _sample_on_sigmas(model, seed, cfg, sampler_name, cond, negative,
                                 latent, sigmas)
    if soft_landing:
        _own = landing_schedule(model, scheduler, steps, shift_video, shift_audio)
        if _own is not None:
            return _sample_on_sigmas(model, seed, cfg, sampler_name, cond, negative,
                                     latent, _own)
    (out,) = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                   cond, negative, latent, denoise=1.0)
    return out


_HERE = os.path.dirname(os.path.abspath(__file__))


_WIDGET_RANGE = {
    "megapixels": (1.0, 0.0, 2.0, float),
    "shot_seconds": (10.0, 1.0, 15.0, float),
    "steps": (8, 1, 100, int),
    "shift_video": (12.0, 1.0, 20.0, float),
    "shift_audio": (3.0, 1.0, 20.0, float),
    "ref_noise_aug": (0.999, 0.5, 1.0, float),
    "latent_upscale_scale": (2.0, 1.0, 4.0, float),
    "upscale_target_short_edge": (0, 0, 4096, int),
    "pace": (1.0, 0.25, 2.0, float),
    "ambient_level": (0.25, 0.0, 1.0, float),
    "foley_level": (0.35, 0.0, 1.0, float),
    "speech_lead_seconds": (0.5, 0.0, 2.0, float),
    "speech_tail_seconds": (2.0, 0.0, 10.0, float),
    "hold_levels": (0.8, 0.0, 1.0, float),
}


def misaligned_widgets(values, options):
    """[(widget, value, what it should have been)] for choices that are not choices.

    sane_widgets repairs a NUMBER that arrives as NaN, and that is the visible symptom
    of a positional shift. It cannot see the cause, and it cannot help the widgets
    whose values are WORDS: a shift puts a scheduler's name into sampler_name and a
    seed into scheduler, and those pass straight through into the render.

    A combo holding a value that is not one of its own options is not a preference
    this node can honour. It is proof the list is out of step -- values are restored
    by POSITION, so converting one widget to an input, or adding or removing one,
    slides every value after it into the wrong slot."""
    bad = []
    for name, choices in (options or {}).items():
        if name not in values:
            continue
        got = values[name]
        if got not in choices:
            bad.append((name, got, choices))
    return bad


def combo_options(spec):
    """{widget: [options]} for every choice widget the node declares."""
    out = {}
    for section in ("required", "optional"):
        for name, decl in (spec or {}).get(section, {}).items():
            if decl and isinstance(decl[0], list):
                out[name] = list(decl[0])
    return out


def alignment_error(bad):
    """The message for a workflow whose widget values have slid out of position."""
    if not bad:
        return ""
    shown = "; ".join(f"{n} = {v!r}, which is not one of {c[:3]}"
                      + ("..." if len(c) > 3 else "") for n, v, c in bad[:3])
    return (
        "H3-LongVideos: this node's saved widget values are out of position. "
        + shown + ".\n\n"
        "Widget values are restored by POSITION, with no names stored, so converting "
        "a widget to an input -- or adding or removing one -- slides every value after "
        "it into the wrong slot. A scheduler's name lands in sampler_name, a seed in "
        "scheduler, and a number with nowhere to go reads as NaN.\n\n"
        "To fix it: right-click the node and choose 'Fix node (recreate)', or convert "
        "any widget you turned into an input back to a widget. Then set the values you "
        "want and save the workflow again. Nothing is wrong with the model or the "
        "prompt, and rendering with these values would use settings you did not pick.")


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
    out, notes, unusable = dict(values), [], []
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
            unusable.append(f"{name} was {raw!r}, now {default}")
            continue
        clamped = min(max(num, lo), hi)
        if clamped != num:
            notes.append(f"{name} was {num:g}, outside {lo:g}..{hi:g}, so it was clamped "
                         f"to {clamped:g}")
        out[name] = cast(clamped)
    if unusable:
        notes.insert(0, "widget values that were not usable numbers, replaced with "
                        "their defaults: " + "; ".join(unusable)
                     + ". Values are restored BY POSITION with no names stored, so this "
                       "means the node's widget list and the saved one disagree -- "
                       "usually because a widget was converted to an input, or the node "
                       "gained one. It repairs itself for THIS run only: the graph still "
                       "holds the bad values, so it comes back every restart until the "
                       "node is fixed. Right-click the node and choose 'Fix node "
                       "(recreate)', set your values, and save the workflow")
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
                               "and that shot keeps its audio; beats without one are silenced.\n\n"
                               "A LINE THAT MUST REACH THE MODEL WORD FOR WORD goes on its own "
                               "line in the beat:\n"
                               "  exact: her wrists stay behind her back the whole way\n\n"
                               "It is placed straight after the beat in your words, and nothing "
                               "in this node reads, scopes, scrubs, reorders or drops it. On a "
                               "short beat the node's own continuity clauses can be 70% of a "
                               "shot and the beat 8%, and this is the one instruction that does "
                               "not compete with them for room.\n\n"
                               "Nothing reads it either, on purpose: a name in it puts nobody in "
                               "the shot, a garment in it removes nothing, and a door in it "
                               "stages no change. Write what must be SAID; let the beat stage "
                               "what happens. `exactly:` and `verbatim:` do the same thing."}),
                "resolution": (list(NATIVE_RES), {"default": "16:9",
                    "tooltip": "Aspect ratio. megapixels sets the size."}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "1.0 = 1024x1024 worth of pixels, H3's native budget. Lower is "
                               "faster and leaner; 0 keeps the preset's own dimensions. Cost "
                               "scales with latent cells and attention is quadratic in them."}),
                "shot_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5,
                    "tooltip": "Maximum shot length. With 'from the beat', each shot is sized "
                               "independently up to this cap; with 'fixed', every shot uses this "
                               "length. Snapped to H3's 17k+5 frame grid."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100,
                    "tooltip": "6-8 with a turbo/distill LoRA; 20+ without one."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
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
                    "ref_image_1, which says who the person is without dictating the frame.\n\n"
                    "OR GIVE IT THE SET, with nobody in it, and the node will read it that way: "
                    "when beat 1 PLACES the cast rather than staging an entrance, and every one of "
                    "them already has a <Picture N> reference of their own, this picture carries "
                    "the room, the light and the furniture as a reference and never becomes frame "
                    "one. That is the difference between a set and an opening frame -- pinned as "
                    "frame one, a picture with nobody in it makes the cast appear out of nothing "
                    "during shot 1, which is the same reason a later shot refuses the previous "
                    "frame when it introduces somebody in position.\n\n"
                    "Both readings are reported in info, so you can see which one you got. To "
                    "force the pinned reading, put the cast in the frame and drop their "
                    "<Picture N> tags, or write the entrance into beat 1."}),
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
                "ref_noise_aug": ("FLOAT", {"default": 0.999, "min": 0.5, "max": 1.0, "step": 0.005,
                    "tooltip": "How CLEAN a reference is shown. 0.999 (H3's default) hands over a "
                               "noise-free image, which invites the model to REPRODUCE it -- "
                               "including its background and pose -- in the opening frames. Lower "
                               "says approximate: try 0.95, then 0.90. One aug covers every "
                               "conditioning latent, so below 0.99 the keyframe rides as a "
                               "reference instead of an anchor."}),
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
                "upscale_model": (_upscale_model_list(), {"default": "none",
                    "tooltip": "Which model, when upscale = model. From models/upscale_models."}),
                "upscale_target_short_edge": ("INT", {"default": 0, "min": 0, "max": 4096,
                    "step": 32,
                    "tooltip": "Fit the result's short edge to this many pixels. 0 keeps the "
                               "model's own factor."}),
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
                    "tooltip": "After a shot that takes something off, the NEXT shot does "
                               "not open on that shot's last frame.\n\n"
                               "Every shot is anchored to the previous shot's last frame. If "
                               "the model does not finish taking the garment off inside its "
                               "own shot, that frame still shows it -- and a keyframe is a "
                               "PICTURE, which outvotes any sentence. Inherit it once and every "
                               "later shot inherits it too, with no wording able to undo it. "
                               "This breaks that inheritance at the one boundary where the "
                               "state changes.\n\n"
                               "The frame still rides as a REFERENCE, so the room, the faces "
                               "and the clothes carry across; only when nobody is left in it, or "
                               "somebody in it also has a portrait riding the next shot, is "
                               "nothing carried. The cost is a cut there, with that shot re-deriving its "
                               "pose and framing. Turn it off if your removals do complete on "
                               "screen and you would rather keep the continuity."}),
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
                # APPENDED. Saved workflows restore widgets by position.
                # APPENDED. Saved workflows restore widgets by position.
                "ambient_audio": ("AUDIO", {"tooltip":
                    "Wire a recording to play UNDER the finished soundtrack. Empty "
                    "means no bed at all.\n\n"
                    "This used to be an override on a bed the node BUILT out of the "
                    "scene's own wording. That builder is gone -- reported as sounding "
                    "horrid -- so the soundtrack is the model's, and this is the one "
                    "way to put a room under it.\n\n"
                    "It is PLAYED, not conditioned on, and that is the point: ambience "
                    "needs no cooperation from a joint model, has nothing to lip-sync "
                    "to, and so cannot put a voice in a wordless shot. It is resampled "
                    "and looped with a crossfade to the length of the film. "
                    "ambient_level sets how loud."}),
                "ambient_level": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How loud the recording wired to ambient_audio plays "
                               "under the finished soundtrack. With nothing wired this "
                               "does nothing -- the bed the node used to BUILD from "
                               "the scene is gone, reported as sounding horrid, and "
                               "the audio is the model's.\n\n"
                               "0.15-0.3 is a bed you notice only when it stops.\n\n"
                               "If the sum would clip, the whole mix is scaled down "
                               "rather than clipped, because clipping distorts the "
                               "line, which is the part worth keeping."}),
                # APPENDED. Saved workflows restore widget values by position.
                "foley_level": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "DOES NOTHING. Kept only so saved workflows keep "
                               "loading: widget values are restored by POSITION with "
                               "no names stored, so deleting this one would load the "
                               "wrong number into the four widgets after it.\n\n"
                               "It used to set how loud the sounds this node BUILT "
                               "were -- a click, a rattle, a rustle, mixed into the "
                               "shots whose audio branch is pinned to silence, which "
                               "cannot get audio from the model at all because prompt "
                               "text never opens a branch. Removed on the report that "
                               "it sounded horrid; the soundtrack is the model's now, "
                               "whole.\n\n"
                               "WHAT THAT COSTS, said plainly: a shot with no line "
                               "and no sound you described is pinned to silence and "
                               "is SILENT. The pin stays -- it is what stops a free "
                               "branch filling itself with a voice and the face "
                               "lip-syncing to the babble. To put sound in such a "
                               "shot, write the sound into that beat, which opens its "
                               "branch on purpose and lets the model make it; or wire "
                               "a track to ambient_audio; or lay one under the "
                               "finished video outside the node."}),
                # APPENDED. Saved workflows restore widget values by position.
                "speech_lead_seconds": ("FLOAT", {"default": 0.5, "min": 0.0,
                    "max": 2.0, "step": 0.1,
                    "tooltip": "Pin generated audio to encoded silence at the start of each "
                               "dialogue shot. This stops pre-babble and keeps the joint "
                               "model's mouth still during that span. 0 disables it; a long "
                               "lead can trim the first word."}),
                "speech_tail_seconds": ("FLOAT", {"default": 2.0, "min": 0.0,
                    "max": 10.0, "step": 0.5,
                    "tooltip": "Free audio kept AFTER a dialogue shot's line, in seconds. The "
                               "line's length is estimated from its words; past lead + line + "
                               "this margin the audio is pinned to encoded silence, the way "
                               "the lead-in pins the opening. A short line in a long shot "
                               "otherwise leaves seconds of open branch the model fills with "
                               "more speech -- babble, or the line again. The model chooses "
                               "WHEN to speak, so a small margin can clip the last word: raise "
                               "it if it does. 0 disables it."}),
                # APPENDED. Saved workflows restore widget values by position.
                # APPENDED. Saved workflows restore widget values by position.
                "hold_levels": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Take the grade the chain adds to itself back out of each "
                               "handoff.\n\n"
                               "Every shot after the first is sampled from the previous "
                               "shot's last frame. The model reproduces that frame "
                               "faithfully -- which is what continuity needs -- so it "
                               "inherits whatever is already in it, and it SYNTHESISES the "
                               "opening frame rather than copying it, so its own bias lands "
                               "on top. The VAE then clamps every decode to 0..1, which "
                               "makes the expansion a ratchet: headroom spent is not given "
                               "back. Eleven shots of that is crushed blacks, blown "
                               "highlights and lurid colour, invisible shot to shot and "
                               "obvious end to end.\n\n"
                               "What makes this correctable without knowing anything about "
                               "your scene: at every boundary the render holds two pictures "
                               "that are supposed to be the SAME frame -- the handoff it "
                               "gave the shot, and the opening frame that came back. "
                               "Nothing was asked to change between them, so everything "
                               "separating them is the chain's doing and none of it is "
                               "yours. That difference is what is measured, per colour "
                               "channel, per boundary, and the median across boundaries is "
                               "what is taken back out.\n\n"
                               "It does NOT aim at a target and never compares a shot to "
                               "shot 1, so a beat that walks into a darker room stays "
                               "darker: measured, a deliberate lighting step keeps about "
                               "98% of its size. The correction is a capped fraction per "
                               "boundary rather than a reset, because shot N's frames reach "
                               "the video ungraded while N+1 is sampled from a corrected "
                               "keyframe -- an uncapped correction would trade burn-in for "
                               "a pop at every cut.\n\n"
                               "1.0 flattens the trend hardest; lower leaves more of the "
                               "look alone. 0 is off. Watch the contrast line in info: if "
                               "it still says UP, raise this. It cannot undo clipping that "
                               "earlier shots already baked in, and it corrects levels "
                               "only -- not softening, and nothing spatial."}),
                # APPENDED. Saved workflows restore widget values by position.
                # APPENDED. Saved workflows restore widget values by position.
            },
            "hidden": {"graph": "PROMPT"},
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
            steps, sampler_name, scheduler, seed,
            first_frame=None, ref_image_1=None, ref_image_2=None, ref_image_3=None,
            ref_image_4=None, negative=None, sigmas=None,
            shift_video=12.0, shift_audio=3.0, ref_noise_aug=0.999, plan_only=False,
            latent_upscale="off", latent_upscale_scale=2.0,
            upscale="off", upscale_model="none", upscale_target_short_edge=0,
            shot_length="from the beat", hold_restraints=True,
            restart_after_removal=True, auto_remove=True, anchor="", character_memory="",
            pace=1.0,
            ambient_audio=None, ambient_level=0.25, foley_level=0.35,
            speech_lead_seconds=0.5, speech_tail_seconds=2.0,
            hold_levels=0.8, graph=None,
            **_removed):

        self._frames = None
        prepared = self._prepare(
            model=model, clip=clip, vae=vae,
            audio_vae=audio_vae, prompt=prompt, resolution=resolution,
            megapixels=megapixels, shot_seconds=shot_seconds, steps=steps,
            sampler_name=sampler_name, scheduler=scheduler,
            seed=seed, first_frame=first_frame, ref_image_1=ref_image_1,
            ref_image_2=ref_image_2, ref_image_3=ref_image_3, ref_image_4=ref_image_4,
            negative=negative, sigmas=sigmas, shift_video=shift_video,
            shift_audio=shift_audio, ref_noise_aug=ref_noise_aug,
            plan_only=plan_only, latent_upscale=latent_upscale,
            latent_upscale_scale=latent_upscale_scale, upscale=upscale, upscale_model=upscale_model,
            upscale_target_short_edge=upscale_target_short_edge, shot_length=shot_length,
            hold_restraints=hold_restraints, restart_after_removal=restart_after_removal, auto_remove=auto_remove,
            anchor=anchor, character_memory=character_memory,
            pace=pace, ambient_audio=ambient_audio,
            ambient_level=ambient_level, foley_level=foley_level, speech_lead_seconds=speech_lead_seconds,
            speech_tail_seconds=speech_tail_seconds,
            hold_levels=hold_levels, graph=graph,
            **_removed)
        if isinstance(prepared, PreparedVideo):
            try:
                return self._render(prepared)
            except BaseException:
                _f, self._frames = self._frames, None
                if _f is not None:
                    try:
                        _f.release()
                    except Exception:
                        pass            # teardown must not mask the interrupt
                raise
            finally:
                self._frames = None
        return prepared

    def _prepare(self, model, clip, vae, audio_vae, prompt, resolution, megapixels, shot_seconds,
            steps, sampler_name, scheduler, seed,
            first_frame=None, ref_image_1=None, ref_image_2=None, ref_image_3=None,
            ref_image_4=None, negative=None, sigmas=None,
            shift_video=12.0, shift_audio=3.0, ref_noise_aug=0.999, plan_only=False,
            latent_upscale="off", latent_upscale_scale=2.0,
            upscale="off", upscale_model="none", upscale_target_short_edge=0,
            shot_length="from the beat", hold_restraints=True,
            restart_after_removal=True, auto_remove=True, anchor="", character_memory="",
            pace=1.0,
            ambient_audio=None, ambient_level=0.25, foley_level=0.35,
            speech_lead_seconds=0.5, speech_tail_seconds=2.0,
            hold_levels=0.8, graph=None,
            **_removed):

        notes = []
        _bad = misaligned_widgets(
            dict(resolution=resolution, sampler_name=sampler_name, scheduler=scheduler,
                 shot_length=shot_length, upscale=upscale, latent_upscale=latent_upscale,
                 upscale_model=upscale_model),
            combo_options(self.INPUT_TYPES()))
        if _bad:
            raise RuntimeError(alignment_error(_bad))
        _fixed, _fixnotes = sane_widgets(dict(
            megapixels=megapixels, shot_seconds=shot_seconds, steps=steps,
            shift_video=shift_video, shift_audio=shift_audio,
            ref_noise_aug=ref_noise_aug, latent_upscale_scale=latent_upscale_scale,
            upscale_target_short_edge=upscale_target_short_edge,
            pace=pace,
            ambient_level=ambient_level, foley_level=foley_level,
            speech_lead_seconds=speech_lead_seconds,
            speech_tail_seconds=speech_tail_seconds, hold_levels=hold_levels))
        megapixels, shot_seconds = _fixed["megapixels"], _fixed["shot_seconds"]
        steps = _fixed["steps"]
        shift_video, shift_audio = _fixed["shift_video"], _fixed["shift_audio"]
        ref_noise_aug = _fixed["ref_noise_aug"]
        latent_upscale_scale = _fixed["latent_upscale_scale"]
        upscale_target_short_edge = _fixed["upscale_target_short_edge"]
        pace = _fixed["pace"]
        ambient_level, foley_level = _fixed["ambient_level"], _fixed["foley_level"]
        speech_lead_seconds = _fixed["speech_lead_seconds"]
        speech_tail_seconds = _fixed["speech_tail_seconds"]
        hold_levels = _fixed["hold_levels"]
        notes.extend(_fixnotes)
        # WIDGETS THAT WERE NOT CHOICES. Each of these had one right answer that the
        # node could reach and the reader could not, so each was a question whose
        # wrong answer only ever made the render worse.
        #
        # cfg: H3 is CFG-free. Every clause this file writes is phrased positively
        #   BECAUSE of that -- at cfg 1 the negative is never evaluated and naming
        #   what you do not want names it. Above 1 the negative starts being read,
        #   the prompting strategy stops being the right one, and the run costs
        #   double. There was no setting here, only a way to break it.
        # trim_seam: the first frame of a continued shot is the model's own redraw
        #   of the keyframe it was handed. It is a duplicate whether or not anyone
        #   ticks a box.
        # silence_nonspeech: already decided per shot -- it fires where there is no
        #   quoted line. The switch only ever turned a correct decision off.
        # cleanup_between_shots: the RAM copy costs a fraction of one shot; VRAM
        #   ratcheting across a long chain ends the render.
        cfg = 1.0
        trim_seam = True
        silence_nonspeech = True
        cleanup_between_shots = True
        # THE CONTINUITY GUARDS. Seven switches, each turning one clause off for the
        # WHOLE RUN, every one of them shipped on. Each exists for a reported failure
        # -- a face lip-syncing to a line nobody wrote, the camera drifting until the
        # room is a different room, a door that opens and shuts itself, two people
        # sharing one gaze, a shot whose sound came from nowhere -- and turning one off
        # does not trade the failure for anything. It just returns it.
        #
        # They were A/B switches: isolate one guard, render twice, see what it did.
        # That is a developer's tool, and it was sitting in the reader's node costing
        # them seven decisions on every workflow.
        #
        # THE HONEST CAVEAT, since it was the argument for keeping them: these guards
        # are also what pays the namings that can draw a duplicate, and they are now
        # not switchable off. The per-shot naming budget meant to replace them does not
        # exist -- see fit_guards, which records why it cannot. What is left is the
        # report: info still names every over-named person and which namings came from
        # this node, and a shot that keeps duplicating is answered by rewriting the
        # beat, which was always the better lever.
        character_guard = True
        hold_gaze = True
        hold_scene_state = True
        mouths_shut_when_no_line = True
        hold_camera = True
        auto_sound = True
        beat_leads = True
        # verbatim sent the prompt with none of the above, to tell the node's doing
        # from the model's. Gone at the reader's word. The diagnosis it served is the
        # one thing here with no replacement, so it is worth saying plainly: with this
        # switch removed, nothing renders your text without the node's sentences over
        # it, and info's account of what each clause WOULD have said is what is left.
        verbatim = False
        # apply_model_sampling asked the reader whether the graph had already set the
        # schedule. comfy's own MiniMaxH3SigmaShift stamps that into the model, so the
        # model answers it.
        _upstream = upstream_h3_shift(model)
        apply_model_sampling = _upstream is None
        if _upstream is not None:
            notes.append(
                f"the H3 schedule is ALREADY SET upstream (video {_upstream[0]:g}"
                + (f"/audio {_upstream[1]:g}" if _upstream[1] else "")
                + f"), so this node is not patching it again and its own shift_video "
                  f"{shift_video:g} and shift_audio {shift_audio:g} are NOT what is "
                  f"running -- the upstream node's are. Read from the stamp comfy's "
                  f"MiniMaxH3SigmaShift leaves in transformer_options. Remove that node "
                  f"to sample on the shifts set here")
        # THE STEP COUNT IS THE SCHEDULE. shift_video 12 is H3's own default and it
        # was chosen against ~20 steps; a turbo LoRA drops that to 3-8 and nobody
        # moves the shift, so the last step is left clearing 0.63-0.86 in one jump
        # and the anatomy under the polish never resolves.
        _lora_steps = lora_step_targets(graph)
        _shift_live = bool(apply_model_sampling
                           and not (sigmas is not None and len(sigmas)))
        if _lora_steps:
            _targets = sorted({n for n, _ in _lora_steps})
            if len(_targets) > 1:
                notes.append(
                    "two or more LoRAs in this workflow state DIFFERENT step counts ("
                    + "; ".join(f"{n} from {nm}" for n, nm in sorted(_lora_steps))
                    + "). A distilled LoRA collapses the denoising trajectory onto the "
                      "step count it was trained for, so stacking two that disagree asks "
                      "the model for both at once and it renders neither. This is the one "
                      "thing here that is really schedules fighting; steps is currently "
                      f"{steps}")
            elif int(steps) != _targets[0]:
                notes.append(
                    f"{_lora_steps[0][1]} is built for {_targets[0]} steps and steps is "
                    f"{steps}. Read out of the FILE NAME, which is the only place a LoRA "
                    f"states it -- its metadata carries rank and alpha and no schedule at "
                    f"all. Running a distilled LoRA off its own step count denoises past "
                    f"or short of where its trajectory lands")
        if _shift_live:
            _jump = final_video_jump(steps, shift_video, scheduler)
            if _jump > REFERENCE_FINAL_JUMP:
                _new = shift_video_for_jump(steps, scheduler)
                if _new is not None and _new < shift_video:
                    _after = final_video_jump(steps, _new, scheduler)
                    notes.append(
                        f"shift_video LOWERED {shift_video:g} -> {_new:g}. At {steps} steps "
                        f"the {shift_video:g} typed in left {_jump:.2f} of video noise for "
                        f"the final step to clear alone, against the "
                        f"{REFERENCE_FINAL_JUMP:.2f} this aims at; {_new:g} leaves "
                        f"{_after:.2f}. comfy's "
                        f"schedules end at zero, so that sigma is crossed in ONE evaluation "
                        f"-- every step before it polishes the top of the schedule and the "
                        f"last one has to invent the structure underneath, which is what a "
                        f"partly rendered limb or face is. 0.40 is what H3's own default "
                        f"shift of 12 already leaves at the ~20 steps the undistilled model "
                        f"is sampled at, so this is the schedule H3's own default already "
                        f"gives when it has the steps it was picked for. Wire `sigmas`, or turn "
                        f"apply_model_sampling off, to keep shift_video exactly as typed"
                        + (f". It does NOT reach {REFERENCE_FINAL_JUMP:.2f} here -- {_after:.2f} is "
                           f"the best {steps} steps can do, with shift_video already at its "
                           f"{_new:g} floor. RAISE steps: the jump falls as the schedule gets "
                           f"more places to stand"
                           if _after > REFERENCE_FINAL_JUMP + 1e-6 else ""))
                    # THE RATIO IS LOAD-BEARING, and lowering shift_video alone breaks
                    # it. ModelSamplingAV.audio_scale IS shift_video/shift_audio, and
                    # MiniMaxH3.process_latent_in carries the audio slice multiplied by
                    # it (model_base.py) -- the packed latent holds audio_scale * x_audio
                    # and process_latent_out divides it back out. It is the limit of
                    # sigma_v/sigma_a as sigma falls, which is what time_shift_sigma
                    # converges to. So shift_audio moves with shift_video, by the same
                    # factor, and the stream keeps riding at the scale it was riding at.
                    # (What does NOT depend on shift_video is where the audio branch
                    # LANDS -- last_audio_sigma inverts the video shift back out. Two
                    # different quantities; only one of them is free.)
                    _lo_a, _hi_a = _WIDGET_RANGE["shift_audio"][1], _WIDGET_RANGE["shift_audio"][2]
                    _ratio = float(shift_video) / float(shift_audio or 1.0)
                    _new_a = min(max(_new / _ratio, _lo_a), _hi_a) if _ratio else shift_audio
                    if abs(_new_a - shift_audio) > 1e-9:
                        _got = _new / _new_a if _new_a else _ratio
                        notes.append(
                            f"shift_audio follows it {shift_audio:g} -> {_new_a:g}, holding "
                            f"video:audio at {_got:.2g}:1"
                            + ("" if abs(_got - _ratio) < 0.05 else
                               f" -- NOT the {_ratio:.2g}:1 you set, because shift_audio hit "
                               f"its {_lo_a:g} floor and could not go lower")
                            + f". That ratio is audio_scale -- the "
                            f"factor the packed latent carries the audio stream at -- so "
                            f"moving shift_video without it would rescale the audio against "
                            f"a picture that did not move. It also lands the audio branch "
                            f"softer: {last_audio_sigma(steps, shift_audio, scheduler, shift_video):.2f} "
                            f"-> {last_audio_sigma(steps, _new_a, scheduler, _new):.2f} on the "
                            f"final step")
                        shift_audio = _new_a
                    shift_video = _new
                elif _new is not None:
                    notes.append(
                        f"the final step still clears {_jump:.2f} of video noise at {steps} "
                        f"steps and shift_video {shift_video:g}, above the "
                        f"{REFERENCE_FINAL_JUMP:.2f} it aims at, and shift_video cannot go lower "
                        f"than {_new:g} -- it is already at the widget floor. RAISE steps: "
                        f"the jump falls as the schedule gets more places to stand")
        _wired = [n for n, r in enumerate((ref_image_1, ref_image_2, ref_image_3,
                                           ref_image_4), 1) if r is not None]
        _missing = unwired_reference_tags(f"{prompt}\n{character_memory}", _wired)
        if _wired and list(_wired) != list(range(1, len(_wired) + 1)):
            notes.append(
                f"reference sockets {', '.join('ref_image_' + str(n) for n in _wired)} "
                f"are wired with a gap, so <Picture N> has been read as the SOCKET "
                f"number and renumbered onto the packed roster "
                f"({', '.join(f'{n}->{i}' for i, n in enumerate(_wired, 1))}). Without "
                f"this a tag naming a socket past the end of the roster matched nothing, "
                f"and its image was dropped in silence")
        prompt = renumber_reference_tags(prompt, _wired)
        character_memory = renumber_reference_tags(character_memory, _wired)
        if _missing:
            notes.append(
                f"<Picture {'>, <Picture '.join(str(n) for n in _missing)}> "
                f"{'names a socket' if len(_missing) == 1 else 'name sockets'} with no "
                f"image on it: nothing is wired to "
                f"{', '.join('ref_image_' + str(n) for n in _missing)}. The tag is "
                f"dropped from the text, because a tag pointing at no picture is a "
                f"person the model is told to look up and cannot find. Wire the image, "
                f"or take the tag out")
        swap = flush_for_model_change(model)
        if swap:
            notes.append(swap)
        _abort = sparse_attention_allocator_abort(model)
        if _abort:
            raise RuntimeError(_abort)
        check_vae_wiring(vae, audio_vae)
        _refuse = minor_with_sexual_staging(
            "\n".join([(character_memory or ""), (prompt or "")]), "\n".join(
                [(prompt or ""), (anchor or ""), (character_memory or "")]))
        if _refuse:
            raise RuntimeError(_refuse)

        prompt, n_legacy = strip_legacy_fields(prompt)
        if n_legacy:
            notes.append(f"dropped {n_legacy} field-label line(s) left over from an older "
                         f"version of this node (overall_soundscape:, [Generation N] and the "
                         f"like) -- your text now goes to the model verbatim, and a label like "
                         f"that is read as text to put ON the picture")
        if (anchor or "").strip():
            scene, beats = "", paragraphs(prompt)
        else:
            scene, beats = split_beats(prompt)
        beats, sheet = pull_character_sheets(beats)
        _exact_all = [exact_lines(b) for b in beats]
        beats = [_EXACT_LINE.sub("", b).strip() for b in beats]
        sheet, _dupes = merge_sheets((character_memory or "").strip(), sheet)
        _minors = sorted({_n for _n, _ln in sheet_lines(sheet)
                          if _n and 0 < age_in(_ln) < ADULT_AGE})
        if _minors:
            notes.append(
                f"{_join_names(_minors)} "
                f"{'are' if len(_minors) > 1 else 'is'} declared under {ADULT_AGE} on "
                f"the sheet, so NO body is described for "
                f"{'them' if len(_minors) > 1 else _minors[0]} by this node -- not a "
                f"softer description, none. Every clause that would name a body, a bare "
                f"region's anatomy or a figure stays silent for that entry, and the rest "
                f"of the film is unaffected. The scene renders. Had the script also "
                f"staged nudity or sex anywhere in it, nothing would have rendered at "
                f"all. If the age is a typo, fix it and the entry behaves like any other"
            )
        _film_mood = mood_declared(anchor)
        _film_duress = film_stages_duress(beats, sheet, anchor)
        if _dupes:
            notes.append(
                f"{', '.join(_dupes)} described more than once -- character_memory and a "
                f"'Name:' paragraph in the prompt are the same channel by two routes, and "
                f"using both put the person in every shot twice. A model told about one "
                f"person twice renders two of them. Kept the character_memory entry and "
                f"dropped the duplicate")
        _undeclared = [n for n, ln in sheet_lines(sheet) if n and not sheet_pronoun(ln)]
        if _undeclared and any(re.search(r"\b(?:he|she|him|her|his|hers)\b", b or "", re.I)
                               for b in beats):
            notes.append(
                f"{_join_names(_undeclared)} {'have' if len(_undeclared) > 1 else 'has'} no "
                f"pronoun on the sheet, and the script uses he/she -- so a beat that says "
                f"\"he\" or \"she\" instead of a name cannot be resolved to "
                f"{'them' if len(_undeclared) > 1 else 'that entry'}, and the shot keeps "
                f"whoever the previous one described instead. Write the pronoun into each "
                f"entry (\"Owen: he, 42, ...\")")
        static = build_scene(anchor, scene, "", "")
        scene = build_scene(anchor, scene, "", sheet)      # the whole of it, for inference
        _described_rooms = set(rooms_named(static))
        if sheet:
            notes.append(f"folded {sheet.count(chr(10)) + 1} character-sheet line(s) into "
                         f"the scene instead of spending a shot on them -- a sheet "
                         f"describes people, it does not stage anything, and it has to "
                         f"be in EVERY shot for a removal to have something to scrub")
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
        _given = len(paragraphs(prompt))
        _sheets = len(sheet_lines(sheet)) if sheet else 0
        notes.append(f"{_given} paragraph(s) in the prompt: {len(beats)} rendered as "
                     f"shots" + (f", {_sheets} folded in as character sheet(s)"
                                 if _sheets else "")
                     + ("" if (anchor or "").strip() else ", 1 kept as the scene"))
        _first_para = beats[0] if beats else ""
        if ((anchor or "").strip() and len(beats) > 1 and _first_para
                and not beat_puts_somebody_on_screen(_first_para, sheet)):
            _quoted = " ".join(_first_para.split())
            notes.append(
                f"the prompt's first paragraph describes a PLACE rather than staging "
                f"anything, and with `anchor` filled in every paragraph is a beat -- so "
                f"it is being spent as shot 1 and is not carried into any other shot: "
                f"\"{_quoted[:100]}{'...' if len(_quoted) > 100 else ''}\". Whatever it "
                f"establishes -- which way a vehicle faces, the room, the time of night "
                f"-- is said once and then gone, and the shots after it are free to put "
                f"it back differently. That is what a van changing direction between "
                f"shots looks like from the outside. Move it into `anchor`, which is "
                f"carried at the front of EVERY shot, or clear `anchor` and let the "
                f"first paragraph be the scene as it is without one. Use one or the "
                f"other: with both, all of the standing description belongs in `anchor`")
        _multi = [i for i, b in enumerate(beats, 1) if "\n" in b]
        if _multi:
            notes.append(
                f"shot(s) {', '.join(str(i) for i in _multi)} carry more than one line. "
                f"Paragraphs are separated by a BLANK line, so lines with only a single "
                f"newline between them are one beat and share one shot. If those were "
                f"meant to be separate shots, put an empty line between them")
        if not beats:
            raise RuntimeError("H3-LongVideos: no beat to render. Every paragraph after "
                               "the first is one shot; a character sheet ('Name: ...') "
                               "is folded into the scene and does not count as one.")

        w, h = scale_to_megapixels(*parse_resolution(resolution), megapixels)
        ceiling = align_frame_count(int(round(float(shot_seconds) * H3_FPS)))
        plan = ShotPlan()
        gone, shown = [], []
        gone_by = {}
        _extras_seen = False        # the film has staged people the sheet does not name
        untracked_strip = []        # (shot, items) a group removal the sheet cannot hold
        inferred_sound = []         # shots given one derived from their action
        restrained = posed = rigid_latched = False
        beat_said_posture = False
        restrained_who = set()    # who is actually in the hardware
        anchored = ""             # where fastened limbs are held
        worn_item = ""            # the hardware, in the author's words
        worn_items = []           # ...each piece of it, in order
        displaced = {}            # garment -> how it was moved
        moved_shots = []          # shots reminded of it
        revealed_shots = []       # shots that uncover a layer
        unattributed = []         # shots whose line names no speaker
        mouth_named = []          # shots with a line, holding the OTHER mouths
        language_shots = []       # shots told which language the line is in
        _spoken_words = {}        # shot -> words actually inside the quotes
        _breath_shots = []        # shots whose only sound was a breath
        _langs_used = []          # ...and which languages those turned out to be
        _script_voted = engine.language_of(engine.spoken_text(prompt or ""),
                                           fallback="")
        _script_lang = (_script_voted or engine.language_named(prompt or "")
                        or "English")
        told_shots = []           # shots whose line orders somebody about
        dialogue_marked = []      # shots whose quotes became <d>...</d>
        poses = {}                # name -> the posture a beat put them in
        here = place_named(scene) or first_place(scene)
        _opening = extract_directives(beats[0])[0] if beats else ""
        ambient_bed = (scene_ambient(anchor, scene)
                       or scene_ambient(anchor, _opening)) if auto_sound else ""
        _bed_src = ("the anchor and the scene" if scene_ambient(anchor, scene)
                    else "the opening beat")
        ambient_shots = []        # shots given the bed
        posture_shots = []        # shots told to keep a standing posture
        travel_shots = []         # shots that move between places
        where_shots = []          # shots in a room the scene does not name
        acoustic_shots = []       # ...and the ones whose sound followed them there
        paced_shots = []          # shots told to spread their action
        staging_shots = set()     # shots that MOVE a garment on screen
        bared_shots = []          # ...and shots that uncover skin
        crowded = []              # (shot, clauses dropped for room)
        pronouned = []            # (shot, [(name, namings turned into pronouns)])
        absent_hold = []          # shots where the wearer is not on screen
        exposed_by_beat = []      # (shot, garments the beat names while covered)
        named_shots = []          # shots reminded the thing is still there
        anchored_shots = []       # shots reminded of it
        gaze_shots = []           # shots told where the look goes
        dialogue_gaze_shots = []  # dialogue shots turned to face each other
        scene_held = []           # (shot, rooms) whose scene description waited
        scene_welded = []         # ...and shots where it could not be held
        looking_at = {}           # {name: target}, each held until it changes
        fall_shots = []           # shots told what takes the landing
        device_shots = []         # shots whose line belongs to a machine
        applied_shots = []        # shots that put the hardware on
        early_hardware = []       # ...where the sheet already claimed it
        tight_shots = []          # ...where the framing also crops it
        cropped_wardrobe = []     # garments a named close frame stopped describing
        _anchor_tight = tight_framing(anchor)
        state_acted = set()
        stated_shots = []           # shots given a state put at the first frame
        turned_shots = []           # shots given both ends of a staged change
        mouth_shut = []             # shots told every mouth is closed
        mouth_acting = []           # ...and the ones whose beat works the mouth
        duress_shots = []           # shots told what the face is doing
        vocal_shots = []            # shots where a vocal was given an owner
        muted_sound = []            # shots whose written sound was given up for it
        stripped_shots = set()      # 0-based shots that took something off
        cut_shots = set()           # 0-based shots opening in a room the keyframe is not in
        shot_rooms = {}             # 0-based shot -> (room it opens in, room it ends in)
        hardware_changed = set()    # 1-based shots that put hardware on or take it off
        _undescribed = []           # rooms the film enters that the prompt never describes
        open_moves = []             # (shot, where) moves to a place the list cannot name
        frame_shots = []            # shots told what the frame holds
        legs_held = ""              # where a beat or the sheet fastened the legs
        exact_shots = []            # shots carrying an exact: line of the author's
        camera_shots = []           # shots told the camera holds still
        named_often = []            # (shot, name, times named, times this node named them)
        contact_shots = []          # shots told which body is with which
        led_shots = []              # shots whose beat was put ahead of the sheet
        restarted = []              # shots started fresh after a removal
        restored = []               # garments an add: put back on
        wearing_shots = []          # shots that put one back on, given both ends
        cast = re.findall(r"^\s*([A-Z][\w'’-]{1,24})\s*:", sheet or "", re.M)
        if not cast:
            cast = re.findall(r"\b[A-Z][a-z]{2,}\b", scene or "")
        deferred_shots = []       # (shot, items whose picture waits this shot)
        _shown_under = []
        covers, cover_owner = {}, {}
        for _who, _line in sheet_lines(sheet):
            for _u, _o in implied_layers(_line).items():
                covers[_u] = _o
                if _who:
                    cover_owner[_u] = _who
        for _u, _o in implied_layers(static or "").items():
            covers.setdefault(_u, _o)
        covers.update(infer_layers([extract_directives(b)[0] for b in beats], scene))
        if covers:
            notes.append("read as layers -- underwear goes under whatever the sheet "
                         "also puts over it, and anything the script itself pairs by "
                         "taking one off to expose the other: "
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
                         f"suppresses that -- only the audio denoise mask does, and it pins "
                         f"the whole shot rather than just its opening")
        active = []                 # the people the previous beat involved
        _seen_before = set()        # everyone a shot has described so far
        _returns = []               # (shot, names back after a shot away)
        _in_frame = []              # who the previous shot's last frame shows, described or not
        shot_frames = {}            # 0-based shot -> (who its frames show, who is still there at its end)
        reentry_shots = {}          # 0-based shot -> who walks in while the keyframe still has them
        _placed_shots = {}          # 0-based shot -> who it introduces in position
        _have_slot = {_i + 1 for _i, _r in enumerate(
            (ref_image_1, ref_image_2, ref_image_3, ref_image_4)) if _r is not None}
        _portrait_of = {_n for _n, _ln in sheet_lines(sheet)
                        if _n and (set(picture_tags(_ln)) & _have_slot)}
        _first_is_plate = False
        guard_words = beat_words = total_words = sound_words = 0
        _state = engine.SceneState(place=engine.place_in(scene or ""))
        _staged_at = engine.staged_applications(
            [extract_directives(b)[0] for b in beats])
        _sheet_hw = {c for c, _p, _w, _a in engine.hardware_spans(sheet or "")}
        for b in beats:
            body, toks, adds = extract_directives(b)
            _said = _exact_all[len(plan)] if len(plan) < len(_exact_all) else []
            _exact = (" " + " ".join(terminate_lines(x) for x in _said)) if _said else ""
            if _said:
                exact_shots.append(len(plan) + 1)
            _marked = mark_dialogue(body)
            if _marked != body:
                dialogue_marked.append(len(plan) + 1)
                body = _marked
            _later_for_state = {c for c, at in _staged_at.items()
                                if at > len(plan) + 1}
            for _n, _line in sheet_lines(sheet):
                if _n:
                    _state.declare(_n, _line, staged_later=_later_for_state)
            _ch = _state.read(body, cast=[n for n, _ in sheet_lines(sheet) if n],
                              shot=len(plan) + 1)
            if _ch.get("applied") or _ch.get("released"):
                hardware_changed.add(len(plan) + 1)
            _was = list(active)
            _back_cands = []
            _carried_on, _carried = [], []
            if character_guard:
                shot_sheet, active = sheet_for_beat(sheet, body, active)
                if len(sheet_lines(sheet)) > len(sheet_lines(shot_sheet)):
                    notes.append(f"shot {len(plan) + 1} describes only "
                                 f"{', '.join(active) or 'the scene'} -- the rest of the "
                                 f"sheet is held back, because a person the text "
                                 f"describes is a person the model draws")
                for _grp, _who_all in unresolved_pronouns(sheet, body, _was):
                    if any(n in (active or []) for n in _who_all):
                        continue
                    notes.append(
                        f"shot {len(plan) + 1} says '{_grp}' and "
                        f"{' and '.join(_who_all)} all answer to it, so the guard could "
                        f"not tell which -- and it describes NEITHER rather than both, "
                        f"because naming somebody the beat did not is how an extra "
                        f"character walks into a shot. Write the name instead of the "
                        f"pronoun in that beat and it resolves")
                _new = [n for n in active if n not in _seen_before]
                if (_new and not arrives_in(body) and not plan
                        and first_frame is not None
                        and all(_n in _portrait_of for _n in _new)):
                    _first_is_plate = True
                    notes.append(
                        f"first_frame is being read as the SET, not as shot 1's opening "
                        f"frame, so it carries the room while "
                        f"{_join_names(_new)} {'are' if len(_new) > 1 else 'is'} placed by "
                        f"the text and held by "
                        f"{'their own reference images' if len(_new) > 1 else 'a reference image of their own'}"
                        f". Beat 1 puts "
                        f"{'them' if len(_new) > 1 else _new[0]} "
                        f"in position rather than staging an entrance, and every one of "
                        f"them already has a <Picture N> of their own -- so this picture "
                        f"has nothing left to say about who they are, only about where "
                        f"they are. Pinned as frame one it would be a picture they are "
                        f"not in, and they would have to appear out of nothing during "
                        f"shot 1: that is the same reason a later shot refuses the "
                        f"previous frame when it introduces somebody in position. It is "
                        f"NOT discarded -- the room, the light and the furniture come "
                        f"with it as a reference. To pin frame one exactly instead, put "
                        f"the cast IN that frame and take their <Picture N> tags off the "
                        f"sheet, or write the entrance into beat 1")
                if _new and not arrives_in(body) and plan:
                    _placed_shots[len(plan)] = list(_new)
                    notes.append(
                        f"shot {len(plan) + 1} introduces {', '.join(_new)} in "
                        f"position rather than arriving, so the previous shot's last "
                        f"frame stops being this shot's FIRST frame -- that frame does "
                        f"not have them in it, and a keyframe is a picture, so they "
                        f"would have to appear out of nothing and travel to the spot "
                        f"the beat describes. The frame is still carried, as a "
                        f"reference, so the room comes with it. Write the entrance -- "
                        f"'walks in', 'steps through' -- if you would rather they "
                        f"arrive on screen and keep the frame as the anchor")
                _back_cands = [n for n in active if n not in _was and n in _seen_before]
                _seen_before.update(active)
            else:
                shot_sheet = sheet
            _frm, _via, _to = travel_legs(body)
            _is_travel = bool(travel_anchor(_frm, _via, _to, here, body))
            _room_before = here
            _place_now = _to or _frm or place_named(body) or here
            _opens_in = _frm or (_room_before if _is_travel else _place_now)
            _is_cut = bool(len(plan) and _opens_in and _room_before
                           and _opens_in != _room_before)
            _prev_stays = shot_frames.get(len(plan) - 1, ([], []))[1]
            _no_carry = not _cond_module.may_carry_frame(
                _prev_stays, active,
                {n for n, ln in sheet_lines(sheet) if n and picture_tags(ln)})
            _fresh = (_is_cut
                      or (restart_after_removal and (len(plan) - 1) in stripped_shots
                          and _no_carry)
                      or (len(plan) in _placed_shots and _no_carry)
                      or bool(_ALONE.search(engine.staged_text(body))))
            _kept = [] if _fresh else list(_in_frame)
            _again = [n for n in comes_in(body, sheet)
                      if n in _kept and n not in _was] if (plan and not _is_travel) else []
            if _again:
                reentry_shots[len(plan)] = _again
                _kept = []
            _carry = [n for n in _kept if n not in active]
            if auto_remove:
                inferred = [t for t in infer_removals(body, scene)
                            if t not in toks and t not in gone]
                if hold_restraints and restraint_coming_off(body):
                    for _n, _ln in sheet_lines(sheet if sheet_lines(sheet) else scene):
                        for _hw in restraint_words(_ln):
                            _named = re.search(r"\b" + re.escape(_hw) + r"\b",
                                               body or "", re.I)
                            _pron = (len(restraint_words(_ln)) == 1
                                     and re.search(r"\b(?:them|it)\b", body or "", re.I))
                            if (_named or _pron) and _hw not in toks \
                                    and _hw not in gone and _hw not in inferred:
                                inferred.append(_hw)
                if inferred:
                    toks = list(toks) + inferred
                    notes.append(f"shot {len(plan) + 1}: read '{', '.join(inferred)}' as "
                                 f"coming off, from the beat's own wording")
            bare = auto_remove and strips_bare(body)
            if bare:
                _strippers = strips_who(body, active if character_guard and active
                                        else [n for n, _ in sheet_lines(shot_sheet) if n])
                _their_sheet = "\n".join(
                    ln for n, ln in sheet_lines(shot_sheet) if n in set(_strippers)
                ) or shot_sheet
                stripped = [g for g in garments_in(_their_sheet)
                            if g not in toks and g not in gone]
                if stripped:
                    toks = list(toks) + stripped
                    notes.append(
                        f"shot {len(plan) + 1} reads as undressing "
                        f"{', '.join(active) if character_guard and active else 'the cast'}"
                        f" completely, and the beat names no garment -- so the wardrobe was "
                        f"read off the character sheet and all of it taken off: "
                        f"{', '.join(stripped)}. Anything worn that is not in that list is "
                        f"still described as on; name it in a 'remove:' line if so")
                elif not gone:
                    notes.append(
                        f"shot {len(plan) + 1} reads as undressing completely, but no "
                        f"garment was recognised in the character sheet, so nothing was "
                        f"taken off and every later shot still describes the clothes. Add "
                        f"a 'remove:' line naming them")
            revived = [t for t in gone if names_any(body, [t])]
            if revived:
                notes.append(
                    f"shot {len(plan) + 1} names {', '.join(revived)} in its own text, and "
                    f"that came off earlier. Beats are sent to the model word for word, so "
                    f"naming it puts it back on -- the scene no longer mentions it, but this "
                    f"beat does. Reword the beat if it should stay off")
            if toks:
                stripped_shots.add(len(plan))
                gone.extend(t for t in toks if t not in gone)
                if extras_in(body):
                    untracked_strip.append((len(plan) + 1, list(toks)))
                _took = strippers_in(body, shot_sheet if shot_sheet else sheet)
                for _t in toks:
                    _wears = [n for n, _wl in sheet_lines(sheet)
                              if n and re.search(r"\b" + re.escape(_t) + r"\b",
                                                 _wl or "", re.I)]
                    gone_by.setdefault(_t, set()).update(
                        [n for n in _took if n in _wears] or _wears)
                _retired = [a for a in shown if names_any(a, toks)]
                if _retired:
                    shown = [a for a in shown if a not in _retired]
                    notes.append(f"shot {len(plan) + 1} takes off something an earlier "
                                 f"'add:' had put on, so that line retires with it: "
                                 + "; ".join(_retired))
                notes.append(f"removed from the scene from shot {len(plan) + 1} on: "
                             + ", ".join(scene_name_for(t, scene) or t for t in toks))
            maybe = missing_removals(body, scene, gone) if not auto_remove else []
            if maybe:
                notes.append(f"shot {len(plan) + 1} reads as taking something off, but the "
                             f"scene still describes {', '.join(maybe)} and there is no "
                             f"'remove:' line for it -- so every shot keeps saying it is worn. "
                             f"Add 'remove: {maybe[0]}' to that beat")
            if auto_remove and gone:
                for _g in list(gone):
                    if _g in restored or any(names_any(a, [_g]) for a in (adds or [])):
                        continue
                    _head = str(_g).lower().split()[-1]
                    if not (names_any(body, [_head]) and beat_stages_wearing(body, _head)):
                        continue
                    _name = scene_name_for(_head, sheet or scene) or _g
                    adds = list(adds or []) + [_name]
                    notes.append(f"shot {len(plan) + 1}: read '{_name}' as put back on, "
                                 f"the way an 'add:' line would say it")
            _wearing = ""             # the both-ends clause for a garment going on
            _staged_add = []          # ...and the phrases it covers, held out of
                                      #    this shot's static wardrobe
            if adds:
                shown.extend(a for a in adds if a not in shown)
                _back = [g for g in gone
                         if any(names_any(a, [g]) for a in adds)
                         and g not in restored]
                if _back:
                    restored.extend(_back)
                    notes.append(
                        f"shot {len(plan) + 1} puts " + ", ".join(_back)
                        + " back on, so anything it covers is hidden again from "
                          "here. A garment coming back has to un-cover as well as "
                          "re-cover, or the layer under it stays described for the "
                          "rest of the run")
                    _worn_now = [a for a in adds
                                 if any(beat_stages_wearing(body, g) for g in _back)
                                 and any(names_any(a, [g]) for g in _back)]
                    if _worn_now:
                        _wearing = wearing_clause(_worn_now)
                        _staged_add = list(_worn_now)
                        wearing_shots.append(len(plan) + 1)
                notes.append(f"added to the scene from shot {len(plan) + 1} on: "
                             + "; ".join(adds))
            i_shot = len(plan)
            _anchoring = (ref_noise_aug is None
                          or float(ref_noise_aug) >= KEYFRAME_SAFE_AUG)
            has_keyframe = ((i_shot > 0 or first_frame is not None)
                            and _anchoring
                            and not (restart_after_removal
                                     and (i_shot - 1) in stripped_shots))
            visible = gone if has_keyframe else [g for g in gone if g not in toks]
            if (i_shot > 0 and restart_after_removal
                    and (i_shot - 1) in stripped_shots):
                restarted.append(i_shot + 1)
            if toks and not has_keyframe:
                _why = ("its opening frame is not anchored -- ref_noise_aug "
                        f"{float(ref_noise_aug):g} is below {KEYFRAME_SAFE_AUG:g}, so "
                        "the handoff rides as a reference rather than holding the "
                        "first frame" if not _anchoring else
                        "it has no keyframe")
                notes.append(f"shot {i_shot + 1} takes something off and {_why}, "
                             f"so {', '.join(toks)} stays described as worn HERE -- the "
                             f"text is the only thing saying it was on to start with. It "
                             f"is scrubbed from the next shot on")
            _moved_now = {g for g, _h in displaced_garments(body, shot_sheet or sheet)}
            _back_now = set(restored_garments(body, shot_sheet or sheet))
            if puts_it_back(body) and len(displaced) == 1:
                _back_now |= set(displaced)
            _heads_back = {str(g).lower().split()[-1] for g in _back_now}
            _moved_now = {g for g in _moved_now
                          if str(g).lower().split()[-1] not in _heads_back}
            covered = hidden_layers(covers,
                                    [g for g in visible if g not in restored],
                                    (set(displaced) | _moved_now)
                                    - {g for g in (set(displaced) | _moved_now)
                                       if str(g).lower().split()[-1]
                                       in _heads_back})
            _said = [g for g in covered
                     if re.search(r"\b" + re.escape(g) + r"\b", body or "", re.I)]
            if _said:
                exposed_by_beat.append((len(plan) + 1, _said))
            _revealed = reveal_clause([u for u in revealed_by(covers, toks)
                                       if u not in visible and not names_any(u, toks)],
                                      scene)
            if _revealed:
                revealed_shots.append(len(plan) + 1)
            _one_line = dict(sheet_lines(shot_sheet)).get((active or [""])[0], "")
            _one_pron = sheet_pronoun(_one_line)
            _one_age = age_in(_one_line)
            _one_body = (body_of(_one_pron, _one_age) if len(active or []) == 1 else "")
            _one_fig = (figure_of(_one_pron, _one_age) if len(active or []) == 1 else "")
            _one_groin = (groin_of(_one_pron, _one_age) if len(active or []) == 1 else "")
            _bare_sheet = "\n".join(ln for n, ln in sheet_lines(shot_sheet)
                                    if n and names_any(ln, toks)) or shot_sheet
            _bare = ("" if (_revealed or bare)
                     else bare_clause(toks, covers, _bare_sheet, body=_one_body,
                                      figure=_one_fig, groin=_one_groin))
            if not _bare and not bare and not _revealed:
                _who_here = (active if character_guard else
                             [n for n, _ in sheet_lines(shot_sheet) if n])
                _still_here = shot_frames.get(len(plan) - 1, ([], []))[1] if plan else []
                _carried_on = [n for n in (_was or [])
                               if n not in (_who_here or []) and n in _still_here]
                _rows = []
                for _n in list(_who_here or []) + _carried_on:
                    _q = _state.people.get(_n)
                    if _q and _q.bare:
                        _rows.append((_n, list(_q.bare), ", ".join(_q.worn)))
                _name_it = (len(_rows) > 1 or len(_who_here or []) > 1
                            or any(_n in _carried_on for _n, _r, _o in _rows))
                _bare = "".join(
                    bare_hold(_rg, covers, _on,
                              [g for g in gone if g not in restored],
                              whose=(_n if _name_it else ""),
                              body=body_of(*_pron_age(shot_sheet, _n)),
                              figure=figure_of(*_pron_age(shot_sheet, _n)),
                              groin=groin_of(*_pron_age(shot_sheet, _n)))
                    for _n, _rg, _on in _rows)
            if _bare:
                bared_shots.append(len(plan) + 1)
            _here = set(active if character_guard
                        else [n for n, _ in sheet_lines(shot_sheet) if n])
            for _u in list(covered):
                if (is_undergarment(_u)
                        and (names_any(body, [_u])
                             or _u in restored
                             or any(names_any(a, [_u]) for a in (adds or [])))
                        and _u not in _shown_under):
                    _shown_under.append(_u)
            _worn_under = [u for u in covered
                           if is_undergarment(u)
                           and u not in _shown_under
                           and (cover_owner.get(u) in _here
                                or u not in cover_owner)]
            _hidden = [u for u in covered
                       if u not in _worn_under and u not in _shown_under]
            _toks_all = visible + _hidden
            # Scoped to this shot's people first: see static_for_shot.
            _static_here = static_for_shot(static, sheet, shot_sheet)
            _scrubbed = ([scrub_removed(terminate_lines(_static_here), _toks_all)]
                         if _static_here.strip() else [])
            for _ln in (terminate_lines(shot_sheet).split("\n")
                        if shot_sheet.strip() else []):
                _m = re.match(r"\s*([A-Za-z][\w'\u2019-]*)\s*:", _ln)
                _who = _m.group(1).lower() if _m else ""
                _allow = [t for t in _toks_all
                          if not (_who and gone_by.get(t) and _who not in
                                  {str(x).lower() for x in gone_by[t]})]
                _scrubbed.append(scrub_removed(_ln, _allow))
            shot_scene = "\n".join(p for p in _scrubbed if p.strip())
            _holds = frame_holds(anchor) or frame_holds(body)
            _cropped = out_of_frame_garments(shot_scene, _holds)
            if _cropped:
                shot_scene = hide_item(shot_scene, _cropped)
                for _c in _cropped:
                    if _c not in cropped_wardrobe:
                        cropped_wardrobe.append(_c)
            _deferred = list(_worn_under)
            shot_scene = defer_tag_for(shot_scene, _worn_under)
            shot_scene = hide_item(shot_scene, _worn_under)
            if _deferred and len(shot_scene) >= 0:
                deferred_shots.append((len(plan) + 1, list(_worn_under)))
            _under = under_clause(
                [(u, covers.get(u, ""),
                  cover_owner.get(u, "") if len(_here) > 1 else "")
                 for u in _worn_under])
            _sheet_says_early = [c for c, at in _staged_at.items()
                                 if c in _sheet_hw and at > len(plan) + 1]
            _scene_for_state = (scrub_removed(shot_scene, _sheet_says_early)
                                if _sheet_says_early else shot_scene)
            _sheet_says_now_or_later = [c for c, at in _staged_at.items()
                                        if c in _sheet_hw and at >= len(plan) + 1]
            _scene_before_now = (
                scrub_removed(shot_scene, _sheet_says_now_or_later)
                if _sheet_says_now_or_later else shot_scene)
            live = [a for a in shown if a not in _staged_add]
            _here_names = {n for n, _ in sheet_lines(shot_sheet or "") if n}
            _said = []
            for a in live:
                _head = (re.findall(r"[a-z]+", a.lower()) or [""])[-1]
                _owners = [n for n, ln in sheet_lines(sheet or "")
                           if n and _head and names_any(ln.split(":", 1)[-1], [_head])]
                if len(_owners) == 1 and _here_names:
                    if _owners[0] not in _here_names:
                        continue
                    _bare_name = engine.bare_name(a.strip().rstrip("."))
                    _said.append(f"{_owners[0]} is wearing the {_bare_name}")
                else:
                    _said.append(a.rstrip("."))
            if _said:
                tail = ". ".join(_said) + "."
                tail = tail[0].upper() + tail[1:]
                shot_scene = f"{shot_scene} {tail}".strip() if shot_scene else tail
            _who_sheet = shot_sheet if sheet_lines(shot_sheet) else scene
            _listed = [n for n, ln in sheet_lines(_who_sheet)
                       if n and names_any(ln, toks)]
            if len(_listed) > 1:
                _by_state = [w for w, _g in (_ch.get("removed") or []) if w in _listed]
                _by_beat = engine.names_in(body, _listed)
                _wearer = (_by_state or _by_beat or _listed)[0]
            else:
                _wearer = _listed[0] if _listed else None
            _bare = own_body(_bare, _wearer or (active[:1] if active else []),
                             active if character_guard else
                             [n for n, _ in sheet_lines(_who_sheet) if n])
            _cast_here = (active if (character_guard and active) else
                          [n for n, _ in sheet_lines(_who_sheet) if n])
            _by_agent = {}
            for _t in (toks if not bare else []):
                _w = next((n for n, ln in sheet_lines(_who_sheet)
                           if n and names_any(ln, [_t])), _wearer)
                _a = removal_agent(body, _cast_here, _w, _t)
                _by_agent.setdefault(_a, []).append(_t)
            tail = (own_body(BARE_HOLD, _wearer or (active[:1] if active else []),
                             active if character_guard else
                             [n for n, _ in sheet_lines(_who_sheet) if n])
                    if (bare and toks)
                    else "".join(off_by_last_frame(_items, _a, scene, body)
                                 for _a, _items in _by_agent.items()))
            _was_restrained = restrained
            if hold_restraints:
                if (names_any(RESTRAINT_HOLD_KEY, toks)
                        or any(restraint_present(t) for t in toks)
                        or (restraint_coming_off(body)
                            and any(_RESTRAINT_WORD.match(str(t)) for t in toks))):
                    restrained = posed = rigid_latched = False
                    anchored = ""
                    worn_item = ""
                    worn_items = []
                    restrained_who = set()
                elif restraint_present(body) or restraint_present(_scene_for_state):
                    restrained = True
                    if not _was_restrained or restraint_going_on(body):
                        _staged_here = (engine.applies_hardware(body)
                                        or restraint_going_on(body)
                                        or (restraint_present(body)
                                            and not restraint_present(_scene_for_state)))
                        _w = (engine.wearer_of(body, [n for n, _ in sheet_lines(sheet) if n])
                              if _staged_here else "")
                        _new = ({_w} if _w else
                                set(restraint_wearers(sheet)) or restrained_by_beat(body, active))
                        restrained_who |= (_new if _new else set(active))
            _named_item = hardware_named(body) if restrained else ""
            _eng_hw = [r for p in _state.people.values()
                       for r in p.hardware.values()]
            _hw_by_wearer = {}
            for _nm, _pp in _state.people.items():
                for _r in _pp.hardware.values():
                    _hw_by_wearer.setdefault(_nm, []).append(_r.item)
            worn_items = merge_hardware_names([_r.item for _r in _eng_hw])
            worn_item = ", ".join(worn_items)
            _applying = bool(restrained and not _was_restrained
                             and not restraint_present(_scene_before_now)
                             and restraint_going_on(body))
            if (not early_hardware and restraint_going_on(body)
                    and restraint_present(_scene_for_state)):
                early_hardware.append(len(plan) + 1)
            if restrained and rigid_hardware(f"{body} {shot_scene}"):
                rigid_latched = True
            if rigid_latched and forced_pose(f"{body} {shot_scene}"):
                posed = True
            _have = plan_lengths([body], ceiling,
                                 shot_length == "from the beat", pace)[0][0] / H3_FPS
            _pace = pace_clause(beat_seconds(body), _have)
            if _pace:
                paced_shots.append(len(plan) + 1)
            _travel = travel_anchor(_frm, _via, _to, here, body)
            if _travel:
                travel_shots.append(len(plan) + 1)
            else:
                _open_to = moved_to(body, active)
                _travel = move_clause(_open_to, body)
                if _travel:
                    open_moves.append((len(plan) + 1, _open_to))
            here = _place_now
            if _is_cut:
                cut_shots.add(len(plan))
            shot_rooms[len(plan)] = (_opens_in or "", here or "")
            _carry = [n for n in _kept if n not in active]
            _shows = list(active) + _carry
            # A walk to another room leaves behind whoever it does not describe.
            _ends_with = list(active) + ([] if (_to and _to != _room_before) else _carry)

            _back = [n for n in _back_cands if n not in _kept]
            if _back:
                _returns.append((len(plan) + 1, list(_back)))
            _gone = leaves_in(body, sheet, _shows)
            _in_frame = [n for n in _ends_with if n not in _gone]
            shot_frames[len(plan)] = (_shows, list(_in_frame))
            if here and here not in _described_rooms and here not in _undescribed:
                _undescribed.append(here)
            _where = where_hold(here, scene) if not _travel else ""
            if _where:
                where_shots.append(len(plan) + 1)
            _room_now = (room_tone(here) or _room) if (auto_sound and _where) else _room
            _bed_now = ((scene_ambient(here) or ambient_bed)
                        if (auto_sound and _where) else ambient_bed)
            if _where and auto_sound and (_room_now != _room or _bed_now != ambient_bed):
                acoustic_shots.append((len(plan) + 1, here))
            _pose_now = posture_in(body, active if character_guard and active
                                   else [n for n, _ in sheet_lines(_who_sheet) if n])
            for _gone_pose in posture_cleared(body, poses):
                poses.pop(_gone_pose, None)
            _posture = ("" if not hold_scene_state
                        else posture_hold({n: p for n, p in poses.items()
                                           if n not in _pose_now},
                                          active if character_guard else
                                          [n for n, _ in sheet_lines(_who_sheet) if n]))
            if _posture:
                posture_shots.append(len(plan) + 1)
            poses.update(_pose_now)
            _anchor_now = limb_anchor(body) if restrained else ""
            if _anchor_now:
                anchored = _anchor_now
            _legs_now = legs_anchor(body) if restrained else ""
            if _legs_now:
                legs_held = _legs_now
            _holding = bool(restrained and anchored and not _anchor_now)
            if _holding:
                anchored_shots.append(len(plan) + 1)
            if _holding and (_anchor_tight or tight_framing(body)):
                tight_shots.append(len(plan) + 1)
            turn = TURN_HOLD if (turns_in(body, cast)
                                 and (gone or shown or restrained)) else ""
            _falls = falls_in(body)
            fall = (FALL_HOLD if (restrained and _falls)
                    else FALL_HOLD_FREE if _falls else "")
            if fall:
                fall_shots.append(len(plan) + 1)
            rigid = restrained and rigid_latched
            chain = (CHAIN_POSE_HOLD if (rigid and posed)
                     else CHAIN_HOLD if rigid else "")
            _off_here = (restraint_coming_off(body)
                         or names_any(RESTRAINT_HOLD_KEY, toks)
                         or any(restraint_present(t) for t in toks))
            anchors = ("" if (_off_here or hardware_handled(body))
                       else anchor_clause(unanchored_hardware(body)))
            if anchors:
                notes.append(f"shot {i_shot + 1} names hardware with no body part beside "
                             f"it, so the shot says where it sits: "
                             f"{anchors.split(': ', 1)[1].rstrip('.')}")
            _scene_sent, _held_rooms, _held_blocked, _held_text = scene_for_here(
                shot_scene, here, anchor,
                [n for n, _ in sheet_lines(shot_sheet) if n], body)
            if _held_rooms and _held_blocked:
                scene_welded.append((len(plan) + 1, list(_held_rooms)))
            elif _held_rooms:
                scene_held.append((len(plan) + 1, list(_held_rooms), list(_held_text)))
            if beat_leads and _scene_sent and not picture_tags(f"{_scene_sent} {body}"):
                _scene_part, _sheet_part = split_sheet(
                    _scene_sent, [n for n, _ in sheet_lines(shot_sheet) if n])
                line = " ".join(p for p in (_scene_part, body, _sheet_part) if p).strip()
                if _sheet_part:
                    led_shots.append(len(plan) + 1)
            else:
                line = f"{_scene_sent} {body}".strip() if _scene_sent else body
            _pairs, _moves = [], []
            if hold_scene_state:
                _moves = state_changes(body)
                _acting = [_state_key(t) for t, _ in _moves]
                _pairs = [(t, s) for t, s in stated_states(line)
                          if _state_key(t) not in state_acted and _state_key(t) not in _acting]
            _turn = direction_anchor(_moves)
            _state_clause = state_hold(_pairs[:max(0, 2 - _turn.count("first frame"))]) + _turn
            if _pairs:
                stated_shots.append(len(plan) + 1)
            if _turn:
                turned_shots.append(len(plan) + 1)
            if _pairs and exits_vehicle(body) and any(
                    _state_key(t) in ("door",) for t, _ in _pairs):
                notes.append(
                    f"shot {len(plan) + 1} says somebody gets OUT of a vehicle and also "
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
            _ends_at = ""
            if _applying and _anchor_now:
                _pos = _anchor_now.split(", at the")[0].strip()
                if _pos and not _pos.startswith("at the "):
                    _ends_at = RESTRAINT_ENDS_AT.format(
                        part=engine.held_part_of(worn_items) or "wrists",
                        where=_pos)
            _lying_now = any(_p == "lying down" for _p in poses.values())
            if engine.posture_in(body):
                beat_said_posture = True
            if not _lying_now and restrained and not beat_said_posture:
                _watch = restrained_who or set()
                _free = (not any(n in poses for n in _watch)) if _watch else (not poses)
                if _free and engine.posture_in(_scene_for_state) == "lying down":
                    _lying_now = True
            _pose_pos = (_anchor_now or anchored
                         or (limb_anchor(_scene_for_state) if restrained else ""))
            _legs_pos = (_legs_now or legs_held
                         or (legs_anchor(_scene_for_state) if restrained else ""))
            _arms_pos = _pose_pos.split(", at the")[0].strip()
            if not _arms_pos and _legs_pos == "ankles to the wrists":
                _arms_pos = "behind the back"
            _pose = pose_clause(_arms_pos, lying=_lying_now, legs=_legs_pos)
            hold = (RESTRAINT_GOING_ON + (CHAIN_RIGID_TAIL if rigid else "") + _ends_at
                    if _applying
                    else chain if chain else (RESTRAINT_HOLD if restrained else ""))
            if _applying:
                applied_shots.append(len(plan) + 1)

            _staged_here = displaced_garments(body, shot_scene)
            if _staged_here:
                staging_shots.add(len(plan) + 1)
            for _g, _how in _staged_here:
                _prev_state = displaced.get(_g, "")
                # Put back up again is a restore, not a new displacement.
                if _prev_state == "pulled down" and _how in ("pulled up", "pulled back"):
                    displaced.pop(_g, None)
                else:
                    displaced[_g] = _how
            for _g in [g for g in displaced if names_any(g, toks)]:
                displaced.pop(_g, None)
            for _g in restored_garments(body, shot_scene):
                _head = str(_g).lower().split()[-1]
                for _k in [k for k in displaced
                           if str(k).lower().split()[-1] == _head]:
                    displaced.pop(_k, None)
            if len(displaced) == 1 and puts_it_back(body):
                displaced.clear()
            _body_low = (body or "").lower()
            _moved = displaced_hold([(g, h) for g, h in displaced.items()
                                     if not re.search(r"\b" + re.escape(g.split()[-1])
                                                      + r"\b", _body_low)])
            if _moved:
                moved_shots.append(len(plan) + 1)

            _wearers = [n for n in restraint_wearers(shot_sheet)
                        if not character_guard or n in active]
            _described = (active if character_guard else
                         [n for n, _ in sheet_lines(shot_sheet) if n])
            if extras_in(body):
                _extras_seen = True
            elif extras_dismissed(body):
                _extras_seen = False

            _look_now = (look_target(body, shot_sheet, _described)
                         if hold_gaze else "")
            _lookers = (subjects_for(body, shot_sheet, _LOOK_VERB_SRC)
                        if hold_gaze else [])
            _look_is_person = bool(_look_now) and any(
                _look_now == _n for _n, _ in sheet_lines(shot_sheet))
            if _look_now:
                for _n in (_lookers or (_described or [])):
                    looking_at[_n] = (_look_now, _look_is_person)
            elif (looks_somewhere(body) or arrives_in(body) or falls_in(body)
                  or turns_in(body, cast) or _MOVES_OFF.search(body or "")):
                _ends = (_lookers
                         or subjects_for(body, shot_sheet, _MOVES_OFF_SRC))
                for _n in (_ends or list(looking_at)):
                    looking_at.pop(_n, None)
            _gone_now = (set(leaves_in(body, sheet, _shows))
                         | set(subjects_for(body, shot_sheet, _MOVES_OFF_SRC)))
            if _gone_now:
                for _n, (_t, _is_person) in list(looking_at.items()):
                    if _is_person and _t in _gone_now:
                        looking_at.pop(_n, None)
            if _ALONE.search(engine.staged_text(body)):
                looking_at.clear()
            _carried = [n for n in _was
                        if n not in set(_described or []) and looking_at.get(n)
                        and n not in set(subjects_for(body, sheet, _MOVES_OFF_SRC))]
            _gazers = [n for n in (_described or []) if looking_at.get(n)] + _carried
            _gaze = ""
            _faces = ""     # the eye-line inferred for a dialogue shot
            _contact = (contact_hold(contact_pairs(body, _described))
                        if len(_described or []) > 2 else "")
            if _contact:
                contact_shots.append(len(plan) + 1)
            _frame = frame_hold(body, anchor, len(_described or []) or 1)
            if _frame:
                frame_shots.append(len(plan) + 1)
            _camera = camera_hold(body, anchor, moving=bool(_travel)) if hold_camera else ""
            if _camera:
                camera_shots.append(len(plan) + 1)
            _wearer_here = (not restrained_who
                            or not character_guard
                            or not (_described or [])
                            or bool(restrained_who & set(_described or [])))
            if not _wearer_here:
                hold = ""
                _pose = ""
                if (len(plan) + 1) in anchored_shots:
                    anchored_shots.remove(len(plan) + 1)
                absent_hold.append(len(plan) + 1)
            elif not _applying and restrained:
                _here_items = merge_hardware_names(
                    [i for n in (_described or []) for i in _hw_by_wearer.get(n, [])]
                ) or worn_items
                _here_item = ", ".join(_here_items)
                _here_rigid = bool(rigid) and (rigid_hardware(_here_item)
                                               if _here_item else True)
                hold = restraint_sentence(
                    _here_item if not _named_item else "",
                    _wearers, _described, anchor=("" if _anchor_now else anchored),
                    rigid=_here_rigid, posed=bool(posed),
                    part=held_part(_here_items or ([_here_item] if _here_item else [])))
                if _here_item and not _named_item:
                    named_shots.append(len(plan) + 1)
            else:
                hold = own_hold(hold, _wearers, _described)
            _speaks = has_speech(body)
            _own = sound_described(body)
            if not _own and not _speaks and _BREATH_PREP.search(body):
                _breath_shots.append(len(plan) + 1)
            _voiced = bool(exertion_in(body) or named_vocals_in(body))
            _bed = _bed_now if auto_sound and _bed_now else ""
            if _bed:
                ambient_shots.append(len(plan) + 1)
            _mute_written = bool(mouths_shut_when_no_line and _own and not _speaks
                                 and not _voiced)
            _will_silence = bool(silence_nonspeech and not _speaks and not _voiced
                                 and (not _own or _mute_written))
            if _mute_written and _will_silence:
                muted_sound.append(len(plan) + 1)
            _has_people = beat_puts_somebody_on_screen(body, sheet)
            _device_line = (mouths_shut_when_no_line
                            and speech_is_a_devices(body, sheet))
            _duress = (duress_face(
                body,
                [(n, ln) for n, ln in sheet_lines(shot_sheet) if n in set(_wearers)],
                _described, _film_duress) if hold_gaze else "")
            if _duress:
                duress_shots.append(len(plan) + 1)
            _mouth_busy = bool(mouth_performs(body) or emotion_in(body))
            _mouth = MOUTH_HOLD if (mouths_shut_when_no_line and _has_people
                                    and (not _speaks or _device_line)
                                    and not _voiced and not _mouth_busy) else ""
            _mouth_from_silence = bool(_mouth)
            _vocal_src = vocal_sources_in(body, shot_sheet) if _voiced else []
            _voicers = [n for n, _ in _vocal_src]
            _vocal_word = _vocal_src[0][1] if _vocal_src else ""
            if (not _mouth and mouths_shut_when_no_line and (_speaks or _voicers)
                    and not _mouth_busy and not _device_line
                    and not (_voiced and not _voicers)):
                _talkers = speakers_in(body, shot_sheet) if _speaks else []
                _open = set(_talkers) | set(_voicers)
                _here_too = [n for n in _was
                             if n not in set(_described or [])
                             and n not in set(subjects_for(body, sheet,
                                                           _MOVES_OFF_SRC))]
                _silent = [n for n in list(_described or []) + _here_too
                           if n not in _open]
                _mouth = voice_sources(_talkers, _vocal_word, _voicers, _silent)
                if _mouth and _voicers:
                    vocal_shots.append(len(plan) + 1)
                if (not _mouth and _speaks and not _talkers and not _voicers
                        and len(_described or []) > 1):
                    _mouth = ONE_VOICE
                    unattributed.append(len(plan) + 1)
            if _mouth:
                (mouth_shut if _mouth_from_silence
                 else mouth_named).append(len(plan) + 1)
            elif _mouth_busy and mouths_shut_when_no_line and _has_people:
                mouth_acting.append(len(plan) + 1)
            _shot_lang = engine.language_of(engine.spoken_text(body),
                                            fallback=_script_lang,
                                            named=engine.language_named(body))
            _lang = (LANGUAGE_HOLD.format(lang=_shot_lang)
                     if (_speaks and not _voiced) else "")
            if _lang and _shot_lang not in _langs_used:
                _langs_used.append(_shot_lang)
            _told = told_hold(told_to_act(
                body, speakers_in(body, _who_sheet),
                _described if character_guard else
                [n for n, _ in sheet_lines(_who_sheet) if n])) if _speaks else ""
            if _told:
                told_shots.append(len(plan) + 1)
            if _lang:
                language_shots.append(len(plan) + 1)
            _said_words = len(engine.spoken_text(body).split())
            if _said_words:
                _spoken_words[len(plan) + 1] = _said_words
            _device = device_voice_clause(body) if (_device_line and _has_people) else ""
            if _device:
                device_shots.append(len(plan) + 1)
            heard = ([] if (not auto_sound or _own)
                     else sounds_for(body, held=[_state_key(t) for t, _ in _pairs]))
            if _own:
                heard = [v for v in named_vocals_in(body) if v not in heard] + heard
                if heard and all(v in _NAMED_VOCALS for v in heard):
                    heard = heard + [_VOCAL_BETWEEN]
            if _will_silence:
                heard = []
            elif _bed:
                heard = heard + [_bed] + ([_room_now] if _room_now else [])
            elif auto_sound and _room_now:
                heard = heard + [_room_now]
            if heard:
                inferred_sound.append(len(plan) + 1)
            _own_unsaid = bool(_own) and not named_vocals_in(body)
            _sound = sound_clause(heard, only=not _speaks, written=_own_unsaid)
            if hold_gaze and _gazers:
                _g = _gazers[0]
                _target, _is_person = looking_at[_g]
                _elsewhere = " ".join([
                    hold, _posture, _pose, _travel, _where, _told, turn, _duress,
                    _mouth, _revealed, _under, _bare, _wearing, tail, _moved,
                    anchors, _state_clause, _device, _sound, _pace, fall])

                def _named_already(_n):
                    return bool(re.search(r"\b" + re.escape(_n) + r"\b", _elsewhere))

                def _their_pronoun(_n):
                    """'her'/'his'/'their', if nobody else in the shot shares it."""
                    _rows = {a: b for a, b in sheet_lines(shot_sheet) if a}
                    _m = re.search(r"\b(she|he|they)\b", _rows.get(_n, ""), re.I)
                    if not _m:
                        return ""
                    _sex = _m.group(1).lower()
                    for _o in (_described or []):
                        if _o != _n and re.search(r"\b" + _sex + r"\b",
                                                  _rows.get(_o, ""), re.I):
                            return ""
                    return {"she": "her", "he": "his", "they": "their"}[_sex]

                if _is_person:
                    if not _named_already(_target):
                        _gaze = gaze_hold(_target, "", True)
                elif len(_described or []) >= 2 or _g not in set(_described or []):
                    _who = "" if _named_already(_g) else f"{_g}'s"
                    if not _who:
                        _who = _their_pronoun(_g)
                    if _who:
                        _gaze = gaze_hold(_target, _who)
                else:
                    _gaze = gaze_hold(_target)
            if _gaze:
                gaze_shots.append(len(plan) + 1)
            if (hold_gaze and not _gaze and _speaks and not _look_now
                    and not _device_line and len(_described or []) >= 2):
                _faces = dialogue_gaze(len(_described))
                if _faces:
                    dialogue_gaze_shots.append(len(plan) + 1)
            _guards = [
                (1, "removal", tail),        # the beat's own action, completing
                (1, "wearing", _wearing),    # ...and its mirror, a garment going on
                (2, "revealed", _revealed),  # what shows where it was
                (2, "under", _under),         # ...and what is underneath, still on
                (2, "bare", _bare),          # ...or that nothing does
                (3, "hold", hold),           # hardware coming open is not a drift
                (4, "fall", fall),           # a body going down needs a landing
                (4, "travel", _travel),      # a journey needs both its ends
                (4, "where", _where),        # ...and later shots need the new room
                (5, "pace", _pace),          # ...and a short action needs the whole shot
                (5, "device", _device),      # a voice that is not hers
                (6, "moved", _moved),        # a garment left where it was put
                (7, "anchors", anchors),     # hardware with nowhere to sit
                (10, "state", _state_clause),
                (9, "posture", _posture),   # where the last beat left the body
                (3, "pose", _pose),
                (11, "gaze", _gaze),
                (12, "duress", _duress),
                (12, "mouth", _mouth),
                (12, "language", _lang),   # ...and in which language
                (3, "contact", _contact),
                (15, "faces", _faces),
                (15, "frame", _frame),
                (13, "camera", _camera),
                (6, "told", _told),          # a listener given an order to ignore
                (13, "turn", turn),
                (14, "sound", _sound),
            ]
            _floor = RESTRAINT_FLOOR_WORDS if (hold or _pose or anchors) else None
            if fall:
                _floor = (_floor or GUARD_FLOOR_WORDS) + FALL_FLOOR_WORDS
            _kept, _dropped = fit_guards(_guards, len(body.split()), floor=_floor)
            if _dropped:
                crowded.append((len(plan) + 1, _dropped))
            for _gone in _dropped:
                _tracker = {
                    "wearing": wearing_shots, "fall": fall_shots,
                    "travel": travel_shots, "where": where_shots,
                    "pace": paced_shots, "device": device_shots,
                    "state": stated_shots, "posture": posture_shots,
                    "gaze": gaze_shots, "duress": duress_shots,
                    "mouth": mouth_named, "language": language_shots,
                    "contact": contact_shots, "frame": frame_shots,
                    "camera": camera_shots, "told": told_shots,
                    "turn": turned_shots, "sound": inferred_sound,
                }.get(_gone)
                while _tracker is not None and (len(plan) + 1) in _tracker:
                    _tracker.remove(len(plan) + 1)
            _also_named = [n for n in dict.fromkeys(
                               list(_in_frame or []) + list(_carried_on) + list(_carried))
                           if n not in (_described or [])
                           and re.search(r"\b" + re.escape(n) + r"\b", _kept)]
            _cast_hold = cast_hold(list(_described or []) + _also_named, body, _extras_seen)
            # A REPEATED NAMING IS SPENT AS A PRONOUN. Only the clauses this node
            # wrote, only where the pronoun resolves to one person, and only after
            # cast_hold has counted the bodies -- the first naming survives, so the
            # body count reads the same text it always did.
            _who_here = [(_n, sheet_pronoun(_ln)) for _n, _ln in sheet_lines(sheet)
                         if _n and re.search(r"\b" + re.escape(_n) + r"\b",
                                             line + _exact + _kept)]
            _kept, _swapped = pronoun_rewrite(_kept, _who_here, extras=_extras_seen)
            if _swapped:
                pronouned.append((len(plan) + 1, _swapped))
            shot_text = (line + _exact + _cast_hold + _kept).strip()
            for _n in (_described or []):
                _total = len(re.findall(r"\b" + re.escape(_n) + r"\b", shot_text))
                if _total >= 3:
                    _mine = _total - len(re.findall(r"\b" + re.escape(_n) + r"\b",
                                                    f"{_scene_sent} {body} {_exact}"))
                    named_often.append((len(plan) + 1, _n, _total, _mine))
            _sound_kept = "" if "sound" in _dropped else _sound
            sound_words += len(_sound_kept.split())
            guard_words += (len(shot_text.split()) - len(_sound_kept.split())
                            - len(f"{_scene_sent} {body}".split()) - len(_exact.split()))
            beat_words += len(body.split()) + len(_exact.split())
            total_words += len(shot_text.split())
            _events = (list(sounds_for(body, held=[_state_key(t)
                                                   for t, _ in _pairs]))
                       if auto_sound else [])
            plan.add(shot_text,
                     list(active) if character_guard else [],
                     _speaks, (_own and not _mute_written) or _voiced,
                     _voiced and not _own, _events)

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
        _lora_model = lora_facts(model)
        _lora_clip = lora_facts(getattr(clip, "patcher", None))
        if _lora_model[0] or _lora_clip[0]:
            _name = lora_name_of(model) or lora_name_of(getattr(clip, "patcher", None))
            _said = []
            if _lora_model[0]:
                _said.append(f"{_lora_model[0]} on the model over {_lora_model[1]} weights at "
                             f"strength {', '.join(f'{v:g}' for v in _lora_model[2][:4])}")
            if _lora_clip[0]:
                _said.append(f"{_lora_clip[0]} on the TEXT ENCODER over {_lora_clip[1]} weights at "
                             f"strength {', '.join(f'{v:g}' for v in _lora_clip[2][:4])}")
            notes.append(
                f"LoRA: {'; '.join(_said)}"
                + (f" -- last one applied: {_name}" if _name else "")
                + ". Reported because it is the one input to a shot this node neither "
                  "writes nor can read out of your text: two runs whose prompts are "
                  "identical render differently and nothing else here says why")
        if verbatim:
            notes.insert(0,
                "VERBATIM is on: each shot was sent your scene, your beat and the sheet "
                "entries for the people it names, and nothing this node writes -- no body "
                "count, no mouth guard, no camera take, no two-ended anchor for a door or "
                "a walk, no posture, gaze, bare region, held state or sound direction. "
                "Every note below still reports what a clause WOULD have said, which is "
                "what makes this worth running: it tells you whether something you are "
                "looking at is the node's doing or the model's. The mechanisms are "
                "untouched -- the keyframe chain, the reference claims, silence pinning, "
                "shot sizing, and the scoping that decides which of your own sentences a "
                "shot gets")
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
        if refs_all and covers and not _PICTURE_TAG.search(f"{scene}\n" + "\n".join(beats)):
            notes.append(
                f"{len(refs_all)} reference image(s) and not one <Picture N> tag anywhere, "
                f"while the wardrobe has layers in it ("
                + "; ".join(f"{u} under {o}" for u, o in list(covers.items())[:3])
                + "). An untagged reference goes into EVERY shot, so a picture of "
                  "something that is currently underneath something else is still sent "
                  "on the shots where it is covered -- and the text having stopped "
                  "describing it does not stop the model drawing it. That is an under "
                  "layer rendered on top. Tag the image onto the thing it shows -- "
                  "'a chastity belt <Picture 2>' -- and it is sent only where that "
                  "thing is actually visible")

        lens, len_note = plan_lengths(beats, ceiling, shot_length == "from the beat", pace)
        plan.set_frame_counts(lens)
        plan.validate()
        _tail = []
        _tailpin = []               # (shot, seconds) pinned past the line's end
        for _i, _b in enumerate(beats):
            if _i >= len(lens) or not has_speech(_b):
                continue
            _words = spoken_words(_b)
            _say = _words / WORDS_PER_SEC
            plan.shots[_i].line_seconds = _say
            _tf = ShotAudio(True, True, False, bool(silence_nonspeech), speech_lead_seconds,
                            AUDIO_LATENT_FPS, _say, speech_tail_seconds, lens[_i]).tail_frames
            if _tf:
                _tailpin.append((_i + 1, _tf / AUDIO_LATENT_FPS))
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
        _clauses = sum(max(1, len([p for p in _CLAUSE_SPLIT.split(
                                       _REMOVE_LINE.sub("", _ADD_LINE.sub(
                                           "", _DIALOGUE_TAG.sub(
                                               " ", _QUOTED.sub(" ", b or "")))))
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
            notes.append(f"{len(plan)} shot(s) x {lens[0]}f (~{lens[0] / H3_FPS:.1f}s) "
                         f"at {w}x{h} = ~{sum(lens) / H3_FPS:.1f}s total")
        else:
            notes.append(f"{len(plan)} shot(s) at {w}x{h}, sized per beat: "
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
                f"later shot is told the old one.")
        if paced_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in paced_shots)} stage less than "
                f"their length, so each is told its action runs across the whole "
                f"shot. A shot told WHAT happens and nothing about WHEN performs it "
                f"at once, and the cheapest way to fill the seconds left is to carry "
                f"on -- the same movement repeated on whatever is nearest. It names "
                f"when, never how fast: 'slowly' is a style instruction and this is "
                f"not one. Give the beat more to do, or shorten the shot, and it "
                f"stops being needed")
        if scene_held:
            _rooms = sorted({r for _, rs, _ in scene_held for r in rs})
            _waited = sorted({t for _, _, ts in scene_held for t in ts})
            notes.append(
                f"the scene paragraph describes {', '.join(_rooms)}, and a paragraph that "
                f"describes a room describes its FURNITURE too -- so that description WAITS "
                f"OUTSIDE it, on shot(s) {', '.join(str(n) for n, _, _ in scene_held)}. The "
                f"paragraph is stamped into every shot, which is what gives a removal "
                f"something to scrub, and the room's NAME was already right in every shot -- "
                f"but the bed was in the text standing beside it, and at cfg 1 there is no "
                f"negative prompt that can take a named thing back. Reported as a bed in the "
                f"living room two beats after she left the bedroom. The room a shot ENDS in "
                f"decides this, not every room it passes through: a walk out of the bedroom "
                f"does show it in the opening frames, but that shot's LAST frame is the next "
                f"shot's keyframe, so a bed drawn at the end of the walk is inherited by the "
                f"shot after it -- and the room being left arrives as a PICTURE anyway, "
                f"because the keyframe IS the previous shot's last frame. So the words say "
                f"where the shot ends and the frame carries where it began. Your paragraph is "
                f"not edited: this is per shot, and a beat that walks back in gets it back in "
                f"full. A character sheet line is never touched, whatever it names, and "
                f"neither is the anchor or a room your beat mentions. What waited: "
                + "; ".join(f'"{t}"' for t in _waited[:3])
                + ". If one of those also carried the film's own hour or light, split it: "
                  "one sentence for the film, one for the room")
        if scene_welded:
            notes.append(
                f"shot(s) {', '.join(str(n) for n, _ in scene_welded)} are not in the room the "
                f"scene paragraph describes, and that description was KEPT anyway, because "
                f"holding it would have left those shots no scene sentence at all. The "
                f"paragraph welds the film's own framing to one room's furniture in a single "
                f"sentence, so taking the room would take the lighting and the hour with it, "
                f"and a shot with no scene is a bigger change than a bed in the wrong room. "
                f"Split it in two -- one sentence for the film ('A small flat at night.') and "
                f"one for the room ('Her bedroom has an unmade bed and a lamp.') -- and the "
                f"room's half will wait outside that room on its own")
        if reentry_shots:
            notes.append(
                f"START FRESH where somebody still in the frame is staged walking in -- "
                + "; ".join(f"shot {k + 1}: {_join_names(v)}"
                            for k, v in sorted(reentry_shots.items()))
                + ". Nothing walked them out, so the frame the shot opens on still has "
                f"them in it, and the beat brings them in again: kept, that is two of "
                f"them. Write them leaving first ('Dan goes out to the car') and the shot "
                f"keeps its keyframe")
        if cut_shots:
            notes.append(
                f"shot(s) {', '.join(str(n + 1) for n in sorted(cut_shots))} CUT, because "
                f"they OPEN IN A DIFFERENT ROOM from the one the shot before ended in. Every "
                f"shot is anchored to the previous shot's last frame, and a keyframe is a "
                f"PICTURE, which outvotes any sentence -- so a living-room shot opening on a "
                f"frame of the kitchen renders neither of them, it renders a blend, and a "
                f"kitchen blended with the words 'living room' is a bathroom: tiles, a sink, "
                f"cabinets. So that frame is not frame one there. It still rides as a "
                f"reference for the PEOPLE in it wherever all of them are in the new shot, so "
                f"they keep their faces and clothes across the cut. A WALK IS NOT "
                f"THIS: a travel beat opens in the room it is leaving, so that frame is the "
                f"right one and the shot keeps its keyframe -- write the move as a journey "
                f"('she walks through to the kitchen') and you get the walk instead of a cut")
        if led_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in led_shots)} put the BEAT in front of "
                f"the character sheet. The sheet has to be in every shot -- clothing "
                f"continuity is read out of it -- but it is a description of a FACE, and it "
                f"was leading every prompt ahead of the action. Measured: 69% of a shot's "
                f"words sat in sentences about a face, and turning every face guard off only "
                f"reached 63%, because the sheet is most of it. What LEADS a prompt decides "
                f"its composition -- anatomy in the opening tokens is what a distilled model "
                f"settles the frame on, and at cfg 1 no later sentence outvotes it. Your "
                f"words are identical and none are rewritten; only the order changed, which "
                f"is the one thing about this that had never been tried")
        if contact_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in contact_shots)} have three or more "
                f"people and a beat that puts two of them in contact, so the shot is told "
                f"WHICH body is with which. Your beat already says it, and it was the only "
                f"thing that did: one sentence among everyone's appearance, and at cfg 1 "
                f"the model reads the prompt as a bag of words and pairs by its own prior. "
                f"Reported as girls kissing each other when they should have been kissing "
                f"the boys. Both sides are named, because an unnamed pairing in a shot with "
                f"four people is the sentence that let it choose. Read from your own words "
                f"only -- a beat that pairs nobody by name ('they kiss') gets nothing, "
                f"because guessing which two is the bug. With two people in the shot "
                f"nothing is said: there is nobody else to pair with")
        if named_often:
            _worst = sorted(named_often, key=lambda r: -r[2])[:6]
            notes.append(
                "named more than twice in one shot -- "
                + "; ".join(f"shot {n}: {who} {times}x ({mine} from this node)"
                            for n, who, times, mine in _worst)
                + ". Naming a person twice in one shot is what draws a second copy of "
                  "them, and every clause that owns a fact -- a pose, a look, whose "
                  "voice it is, who is wearing what -- pays a naming to say whose fact "
                  "it is. Worth reading when duplicates persist. The guards that write "
                  "these are no longer switchable -- each answers a failure of its own, "
                  "and turning one off returns that failure rather than trading it for "
                  "anything -- so the lever is the BEAT: a pronoun costs nothing, and "
                  "'she turns' in place of a second 'Mara turns' takes a naming off this "
                  "shot without losing a word of what you asked for")
        if camera_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in camera_shots)} say nothing about the "
                f"camera, so each is told it is one unbroken TAKE from one position, angle "
                f"and distance. "
                f"Reported as the camera moving on its own and breaking continuity -- and "
                f"the chain is what makes that expensive, because every shot opens on the "
                f"PREVIOUS shot's last frame. A shot that drifts hands the drifted "
                f"viewpoint on, the next adds its own, and the room stops being the room. "
                f"An unstated attribute is left to the model's prior, and for a video model "
                f"that prior is movement. Your words always win: any camera note in the "
                f"beat or the anchor stands it down there, and a journey between places "
                f"keeps its moving camera")
        if exact_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in exact_shots)} carry an exact: line. "
                f"It is placed straight after the beat in your words, and nothing in this "
                f"node reads, scopes, scrubs, reorders or drops it -- it is not a guard "
                f"and has no budget to lose, which is what makes it the one instruction "
                f"that reaches the model exactly as written. Nothing reads it either: a "
                f"name in it puts nobody in the shot, a garment in it removes nothing and "
                f"a door in it stages no change, so write what must be SAID there and let "
                f"the beat stage what happens. Counted against the beat in the balance "
                f"below, because it is your text")
        if frame_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in frame_shots)} stage something a "
                f"portrait cannot contain and say nothing about the camera, so they are "
                f"told what the frame HOLDS: the whole body, head to feet, with the room "
                f"around it. An attribute a prompt does not state is not left to the model, "
                f"it is left to the model's PRIOR -- and the prior for a named, described "
                f"person is a portrait facing the lens. The sheet describes a face in every "
                f"shot because clothing continuity needs it there, and the mouth guard "
                f"describes a mouth in every silent shot because babble needs it, so the "
                f"text leans towards a face and nothing in it said how much of the person to "
                f"show. Reported as the camera fixated on one character staring into the "
                f"lens, with no reference image in the run at all. Your camera always wins: "
                f"write any framing in the beat or the anchor -- a close-up included, since "
                f"a close-up is a frame somebody asked for -- and this stands down. It is "
                f"ranked below everything your own words imply, so a crowded shot drops it "
                f"first")
        if open_moves:
            notes.append(
                "shot(s) " + ", ".join(f"{n} (to the {w})" for n, w in open_moves[:6])
                + " move somewhere the place list cannot name, so the arrival is told to be "
                  "PERFORMED -- the whole move on screen, first step to last. A closed list "
                  "of room words can never cover a script nobody has written yet, and a move "
                  "nobody is told to make is a move the model CUTS to: reported as the set "
                  "changing under the characters instead of them walking into it. This reads "
                  "the destination from your own words and claims nothing else about it -- no "
                  "room state, no acoustic, no cut decision -- so a dungeon, a cargo bay or a "
                  "stable all work without being listed anywhere. Anything that is furniture, "
                  "a body part, a vehicle or a person is left alone")
        if untracked_strip:
            _items = sorted({t for _n, ts in untracked_strip for t in ts})
            notes.append(
                f"shot(s) {', '.join(str(n) for n, _ in untracked_strip)} take "
                f"{', '.join(_items)} off PEOPLE THE SHEET DOES NOT NAME, and the "
                f"removal reaches only the ones it does name. A character sheet entry is "
                f"what a removal scrubs and what carries the bare region into every later "
                f"shot; an unnamed woman has neither, so her own words come off in the "
                f"beat that says so and nothing holds them off afterwards -- the next shot "
                f"says nothing about her, and what it opens on is a keyframe taken while "
                f"she was still half in them. Reported as some of the skirts still being "
                f"on when all of them should have come off. Give each of them an entry, "
                f"however short -- 'Girl 1: she, 20, a denim skirt.' -- and their removals "
                f"hold exactly like the named character's. Your words are never rewritten "
                f"either way; this is about what the node can keep saying after the beat "
                f"that said it")
        if _undescribed:
            _them = "them" if len(_undescribed) > 1 else "it"
            notes.append(
                f"the film enters {', '.join(_undescribed)}, and your prompt never "
                f"describes {_them} -- a room the text only NAMES is a room the model "
                f"invents, and what it invents from is the frame the shot opened on plus "
                f"whatever the other rooms suggest. Reported as a living room turning into "
                f"a bathroom: a kitchen frame and the words 'living room' share tiles, a "
                f"sink and cabinets, and nothing in the text said otherwise. Give each "
                f"room a sentence of its own -- 'The living room has a green sofa and a low "
                f"table.' -- either in the beat that enters it or in the scene paragraph. A "
                f"scene sentence that names a room is carried ONLY in the shots that are in "
                f"that room, so it costs every other shot nothing")
        if where_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in where_shots)} are in a room the "
                f"scene text does not name, so each is told which one. The scene "
                f"paragraph is stamped into EVERY shot -- it has to be, or a removal "
                f"has nothing to scrub -- so a script that walks from one room to "
                f"another goes on opening every later shot with the room it started "
                f"in, while the beat has them somewhere else. The shot then holds two "
                f"places at once and settles on whichever the model weighs more, "
                f"differently each time. Your scene text is not edited: move the "
                f"location into the beats, or keep the scene general, and this stops "
                f"being needed")
        if wearing_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in wearing_shots)} put a garment back "
                f"ON, so each is told both ends: off the body as the shot opens, fully "
                f"on by the last frame. An 'add:' used to go straight into the scene "
                f"block as a worn item, which told a shot inheriting a last frame "
                f"WITHOUT the garment that it flatly has it -- a disagreement rather "
                f"than a change, and the model settles those in the opening frames by "
                f"turning whatever is on the body into the garment. That reads as one "
                f"thing instantly becoming another, a beat before the beat that puts it "
                f"on, which is what those opening frames are. The garment joins the "
                f"static wardrobe from the NEXT shot, the way a removal scrubs from its "
                f"own. An 'add:' that merely reveals a layer already underneath is left "
                f"alone: nothing is being put on there")
        if pronouned:
            notes.append(
                "a repeated naming was spent as a PRONOUN in this node's own clauses -- "
                + "; ".join(f"shot {n}: " + ", ".join(f"{who} x{k}" for who, k in sw)
                            for n, sw in pronouned)
                + ". Naming somebody three times in one shot is what draws a second "
                  "copy of them, and a clause cannot simply be dropped to save the "
                  "naming -- the fact it carries goes with it. So the first naming in "
                  "the clause text stands and the repeats become 'she', 'his', 'him'. "
                  "Your own words are never touched, and nothing is rewritten where a "
                  "second person in the shot answers to the same pronoun, or where the "
                  "beat stages extras: an unresolvable pronoun is the ambiguity the "
                  "naming existed to prevent. The namings left are in your text, and a "
                  "pronoun there costs nothing either")
        if crowded:
            notes.append(
                "guard clauses dropped for room -- "
                + "; ".join(f"shot {n}: {', '.join(d)}" for n, d in crowded)
                + f". Each shot's continuity text is capped at "
                  f"{GUARD_WORDS_PER_BEAT_WORD} words per word of beat, floored at "
                  f"{GUARD_FLOOR_WORDS}, and the lowest-ranked clauses give way "
                  f"first. The cap is set to catch a runaway rather than to trim "
                  f"routinely, so this firing at all means one shot is carrying far "
                  f"more continuity than its beat -- usually a one-line beat in a "
                  f"scene holding a lot of state. Giving that beat more to do buys "
                  f"back the room, and is better than raising the cap: every clause "
                  f"below the line is answering something")
        if acoustic_shots:
            notes.append(
                "the sound followed them into the new room on "
                + "; ".join(f"shot {n}: {r}" for n, r in acoustic_shots)
                + ". The ambient bed and the room tone were read once, before the "
                  "first shot, out of the scene -- so a film that walked into a "
                  "tiled bathroom went on being told it sounds like the carpeted "
                  "room it left. H3 is joint, so that is the picture told one room "
                  "and the audio told another inside the same conditioning, which is "
                  "the contradiction the room hold exists to end, arriving through "
                  "the other branch. Only where the room actually changed and only "
                  "where the new room has a sound of its own: otherwise the film's "
                  "own bed stands, because one bed across a chain is part of what "
                  "makes it one film")
        if travel_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in travel_shots)} move between "
                f"places, so the shot is told where it BEGINS as well as where it "
                f"ends. A journey given only its destination is a journey the model "
                f"can satisfy by starting there -- the living room becomes the "
                f"bedroom at the first frame and the hallway between them is never "
                f"seen. Named both ends, it has to travel. The starting place is "
                f"read from the beat, or from wherever the last one left everybody")
        if posture_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in posture_shots)} are told to keep "
                f"the posture an earlier beat put somebody in -- seated, kneeling, "
                f"lying down. The scene-state reader tracks scenery and nothing about "
                f"the body, so a shot that ended with somebody seated was followed by "
                f"one free to stand them up: the keyframe carries the pose as a "
                f"picture, but the text is what the model reconciles it against, and "
                f"text that says nothing loses to a reference that says something. "
                f"Standing is never held -- it is the default pose, so the clause "
                f"would cost a naming of the person and buy nothing")
        if unattributed:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in unattributed)} carry a line that "
                f"names no speaker, and more than one person is in them -- so which "
                f"mouth to hold is unknowable and the shot is told only that there is "
                f"ONE voice. H3 is joint, so an unheld mouth beside an open audio "
                f"branch is where a second voice comes from, and that voice is the "
                f"babble. Attribute the line -- 'Nora says: \"...\"' -- and the "
                f"listener's mouth is held shut by name instead")
        if revealed_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in revealed_shots)} take off a "
                f"garment that was covering another, so the shot is told what shows "
                f"there now. The removal clause is emphatic and specific -- off the "
                f"body, dropped out of frame -- while the layer underneath is one "
                f"entry in an attribute list, and against a prior that says trousers "
                f"coming off means bare skin, a list entry does not compete. Said only "
                f"on the shot that uncovers it; after that it is simply worn")
        if bared_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in bared_shots)} take off a garment "
                f"with nothing named underneath it, so the shot is told that region is "
                f"BARE. Left unsaid, the space a garment leaves is unspecified, and an "
                f"unspecified region is filled by the model's own prior -- for legs "
                f"that prior is legwear, so leggings or tights appear that the prompt "
                f"never asked for, and the keyframe carries them into every later shot. "
                f"It names a body part and never a garment: at cfg 1 there is no "
                f"negative prompt, so naming the unwanted thing would summon it. Name "
                f"an under-layer in the sheet and this gives way to that instead")
        if restarted:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in restarted)} start FRESH rather "
                f"than from the previous shot's last frame, because the shot before "
                f"took something off -- that is restart_after_removal, and it is what "
                f"stops a garment being inherited back through the keyframe. It costs "
                f"a visible cut at each of those points. Turn it off to keep the "
                f"chain unbroken and accept the risk")
        if exposed_by_beat:
            notes.append(
                "a beat NAMES something the wardrobe says is covered: "
                + "; ".join(f"shot {n}: {', '.join(g)}" for n, g in exposed_by_beat)
                + ". Your beats are passed through word for word and are never "
                  "scrubbed, so the layering can take it out of the sheet and the "
                  "beat puts it straight back -- and a described thing is a drawn "
                  "thing, drawn over whatever is on top of it. Worse, the next shot "
                  "starts from this one's last frame, so once it is rendered on top "
                  "it is carried forward and looks permanent. Take the name out of "
                  "the beat while it is underneath, or take the outer garment off "
                  "first. Nothing here edits your wording")
        if absent_hold:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in absent_hold)} describe nobody who "
                f"is wearing the hardware, so the restraint hold is left out of them. It "
                f"says cuffs are closed on wrists, and in a shot where the person wearing "
                f"them is not described those wrists belong to nobody the text mentions -- "
                f"so the model draws the person the sentence implies, which is a duplicate "
                f"nobody asked for. The hold latches, so the shot they come back in has it "
                f"again")
        if moved_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in moved_shots)} carry a garment "
                f"MOVED rather than taken off -- pulled down, pushed up, shoved aside. "
                f"It is still on the body, so it stays in the scene and is described "
                f"where the beat left it. Counted as a removal it would be scrubbed "
                f"instead, and every later shot would describe nothing where something "
                f"still is -- which is the garment coming back looking like a "
                f"different one. Putting it back ('pulls them back up') releases it, "
                f"and a real removal or a `remove:` empties it for good")
        if named_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in named_shots)} name the hardware "
                f"itself, because their own text does not. The holds say a restraint "
                f"stays whole and closed and never say WHAT it is, so a shot after the "
                f"one that applied it is told a restraint exists with no object to "
                f"draw -- and what renders is the consequence without the hardware: "
                f"held hands and a restrained posture, bare wrists. Taken from your own "
                f"wording at the shot that put it on, and released by a `remove:` "
                f"naming it")
        if early_hardware:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in early_hardware)} stage hardware "
                f"going ON, but the character sheet already lists it as worn. The sheet "
                f"goes into every shot, so it DESCRIBES the hardware in the shots "
                f"BEFORE this happens, and a described item is a drawn item. What the "
                f"node will not do is assert it: no shot before this one is told the "
                f"restraint is fastened, and this one is told both ends rather than "
                f"the standing hold. Take the hardware off the sheet entry and let the "
                f"beat put it on, or drop the beat if she wears it throughout. Your "
                f"wording is never edited, so this one is yours")
        if deferred_shots:
            _items = sorted({i for _n, its in deferred_shots for i in its})
            notes.append(
                f"{', '.join(_items)} WAITS on shot(s) "
                f"{', '.join(str(n) for n, _ in deferred_shots)}, where it is under "
                f"something else -- BOTH the words and the picture. It is deferred, "
                f"never removed: your character memory is not edited, and the item "
                f"comes back in full on the shot that lifts, moves or removes what "
                f"covers it. A "
                f"reference is an instruction to REPRODUCE an image, so handing the "
                f"model a picture of a thing that is under a skirt draws it through "
                f"the skirt -- measured twice, including with the cover described as "
                f"whole and opaque. Reference strength is ref_noise_aug and it is one "
                f"number for every image, so this one cannot be weakened without "
                f"weakening the face")
        if applied_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in applied_shots)} put the hardware "
                f"ON, so they are told both ends -- open and off at the first frame, "
                f"closed on the body by the last -- instead of the standing hold. The "
                f"standing hold says the restraint is fastened as it was put on and "
                f"still fastened at the last frame, which read at frame 1 means it is "
                f"already closed. A first frame that already has the cuffs on leaves "
                f"the catching and the struggling to happen in whatever order is left, "
                f"which is being restrained and THEN caught. From the next shot the "
                f"standing hold is correct again, because by then it is on")
        if device_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in device_shots)} have a spoken "
                f"line that belongs to a machine, not to anybody in the room. H3 is "
                f"joint and the audio branch has no idea a voice came out of a set, so "
                f"a quote made the shot a speaking one and the only face in frame was "
                f"handed the line. The branch still opens -- the set is meant to be "
                f"heard -- but the mouths are held closed and the voice is given back "
                f"to the thing it came out of. A line anybody in the room might have "
                f"stays theirs: an unattributed quote is a person talking")
        if fall_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in fall_shots)} put a body down, so "
                f"the shot is told what takes the landing and what the legs do. A fall "
                f"is the frame where limbs are least determined -- fast motion, heavy "
                f"occlusion, and a middle the model has to invent -- and leaving it to "
                f"work out what catches the body leaves it free to add something that "
                f"can, which is where a spare limb comes from. Said as what the limbs "
                f"DO, never as how many there are: a count is also a mention, and "
                f"naming legs to ask for two is a way of asking for legs")
        if gaze_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in gaze_shots)} name something to "
                f"look at, so the eyes and the head are put on it in so many words. "
                f"The beat says it once and two things pull the other way: a person in "
                f"frame faces the camera unless something says otherwise, and a "
                f"near-clean reference asks for the portrait's pose -- which looks at "
                f"the lens, because photographs of people do. Nothing is said about "
                f"where the camera is. It is HELD until something moves it, and a "
                f"look belongs to whoever is doing the looking -- so it is said only "
                f"in shots that describe that person, and named once a second person "
                f"is in frame with them. Reported as one character stuck gazing at the "
                f"camera while the other does his part: the target was one string with "
                f"no owner, said impersonally, so a look she staged went on being said "
                f"in shots she was not in and landed on whoever was")
        if dialogue_gaze_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in dialogue_gaze_shots)} carry a line "
                f"and two or more people, and the beat names nothing to look at, so the "
                f"faces are turned to each other. Reported as two people talking to the "
                f"camera instead of each other: with no look staged, both faces fall to "
                f"the model's prior -- a portrait, facing the lens -- and a near-clean "
                f"reference asks for exactly that pose. A line has an addressee whether "
                f"or not the beat wrote one, and the addressee is in the shot, so this is "
                f"the one thing that can be said without inventing. One impersonal "
                f"sentence: both names in the shot are already spent, and a third "
                f"mention is a third person. Write 'looks at' or 'turns to' in the beat "
                f"and that is said instead")
        if anchored_shots:
            notes.append(
                f"fastened limbs held in place on shot(s) {', '.join(str(n) for n in anchored_shots)}"
                f" -- the shot that staged it said where, and every shot after it is "
                f"told the same, because the restraint hold keeps the hardware SHUT and "
                f"says nothing about where it is. Position was being carried by the "
                f"picture alone, and the picture is the previous shot's last frame. "
                f"Cleared by a `remove:` naming the hardware, like the hold itself")
        if cropped_wardrobe:
            notes.append(
                "the anchor names a close frame and says what it is close ON, so the "
                "wardrobe that frame cannot hold stopped being described: "
                + ", ".join(cropped_wardrobe)
                + ". The camera was always reaching the model -- it is a tenth of a "
                  "shot's text -- and the rest of the shot was asserting clothes the "
                  "frame has no room for, which is a wider frame said at length. "
                  "ONLY WARDROBE GOES. Where the limbs are held and what is fastened "
                  "to them are still said, because a close frame crops the anchor "
                  "point out of the picture the NEXT shot inherits and the text is "
                  "then the only thing that knows. Write the frame without naming a "
                  "subject -- \"close-up\" and no more -- and nothing is cropped, "
                  "because there is no way to know what it is close on")
        if tight_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in tight_shots)} frame tight enough to "
                f"crop the anchor point out. That matters past this shot: the next one "
                f"starts from THIS one's last frame, so whatever the close framing cut "
                f"off is missing from the picture the next shot inherits, and the text is "
                f"the only thing that still knows where the limbs are fastened. It is "
                f"being said. If the position still drifts, give the beat a wider frame "
                f"so the anchor is in the picture the chain hands on")
        if ambient_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in ambient_shots)} were given an "
                f"ambient bed read from {_bed_src} -- \"{ambient_bed}\". "
                f"It goes under shots whose audio branch is ALREADY open: ones with a "
                f"line, or with a sound you wrote yourself. It can never open one. "
                f"AMBIENCE ON EVERY SHOT WAS TRIED AND DOES NOT WORK: the bed was "
                f"allowed to open a branch, which is the one thing nothing inferred "
                f"here may do, and an open branch on a joint model fills itself. At "
                f"4-8 steps the final audio step clears 50%-30% of its denoising in "
                f"one jump, and what a branch resolving that much at once invents is "
                f"a voice -- so every wordless shot got ambience and a babbling mouth "
                f"with it. Ambience everywhere and silence are mutually exclusive by "
                f"construction: the silence latent IS the audio, and there is no room "
                f"in it for a room tone. To score a silent shot, write the sound into "
                f"that beat -- that is you asking for audio on purpose -- or lay an "
                f"ambient track under the finished video outside the model, where it "
                f"costs nothing and cannot speak")
        if dialogue_marked:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in dialogue_marked)} had their "
                f"quoted speech wrapped in H3's own dialogue marker, <d>...</d>. "
                f"Those are special tokens the model was trained with, and they say "
                f"a span is SPOKEN; quotation marks say nothing at all, so a quoted "
                f"instruction reached the model as an imperative sentence and was "
                f"performed -- often a beat before anybody said it. Every word you "
                f"wrote is kept in order; only the quotation marks are exchanged. "
                f"Mark them yourself and this leaves them alone")
        if told_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in told_shots)} carry a line that "
                f"ORDERS somebody to do something, so the listener is given "
                f"something to be doing while it is said. The node does not stage "
                f"what a quoted line asks for -- the readers refuse speech -- but "
                f"the words are still in the shot, because beats go to the model "
                f"verbatim, and a video model does not tell a quoted instruction "
                f"from a stage direction: it renders what the words describe, and "
                f"the action lands a beat early. The words cannot be removed "
                f"without breaking the one promise this node makes about your text. "
                f"If it still happens, put the order in narration instead -- 'Dana "
                f"tells her to lie down' -- and keep the quoted line for something "
                f"that is not an instruction")
        if language_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in language_shots)} carry a line, "
                f"so each is told which language it is spoken in -- "
                f"{', '.join(_langs_used) or SPOKEN_LANGUAGE}, read from the line "
                f"itself rather than fixed. H3 is joint and multilingual: the prose "
                f"conditions the audio branch, and a branch told a line is spoken "
                f"but never told in WHAT will pick a language -- fluent delivery in "
                f"one nobody asked for sounds like babble to anybody expecting the "
                f"one they wrote. Said positively, because at cfg 1 there is no "
                f"negative prompt and naming the unwanted language would ask for it. "
                f"Write the dialogue in the language you want spoken; a line too "
                f"short to tell falls back to the rest of the script, then to "
                f"{SPOKEN_LANGUAGE}")
        _roomy = []
        for _n, _w in sorted(_spoken_words.items()):
            _sec = (lens[_n - 1] / H3_FPS) if _n - 1 < len(lens) else 0.0
            _need = _w / 2.5 + 0.5      # ~150 words a minute, plus a breath
            if _sec > 0 and _sec > _need * 2:
                _roomy.append((_n, _w, _sec, _need))
        if _roomy:
            notes.append(
                "shot(s) " + ", ".join(
                    f"{n} ({w} word{'s' if w != 1 else ''} of line, about "
                    f"{need:.1f}s, in a {sec:.1f}s shot)"
                    for n, w, sec, need in _roomy)
                + " leave more than half their length with no line in it. The audio "
                  "branch runs for the whole shot and fills what is left, and what "
                  "it fills it with is the line again -- that is where doubled "
                  "dialogue comes from. Shorten those shots (shot_length 'from the "
                  "beat', or a lower shot_seconds), or give the beat more to say. "
                  "Room tone is already laid under them, which is what makes the "
                  "silence survivable at all")
        if _breath_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in _breath_shots)} stage a "
                f"breath and nothing else audible, so they are conditioned on "
                f"silence and the breath is NOT heard. A single indrawn breath "
                f"is half a second; holding the audio branch open for a whole "
                f"shot to render it leaves the rest of that shot open, and an "
                f"open branch on a joint model fills itself with a voice -- "
                f"which is the babble that arrives just before somebody speaks. "
                f"To hear it, put the breath in the same beat as the line, or "
                f"give the shot a sound that lasts: breathing hard, a chain, "
                f"footsteps")
        _hard = []
        _said_all = engine.spoken_text(prompt or "")
        for _m in _HARD_TO_SAY.finditer(_said_all):
            _t = _m.group(0).strip()
            if _t and _t not in _hard:
                _hard.append(_t)
        if _hard:
            notes.append(
                f"the dialogue contains {len(_hard)} thing(s) with no single way "
                f"to say them out loud: {', '.join(_hard[:10])}. A joint model "
                f"reads the line as text and chooses a pronunciation -- \"7:30\" is "
                f"\"seven thirty\" and equally \"seven three zero\", \"Dr.\" is "
                f"\"doctor\" and equally \"dee arr\" -- and the choice is where "
                f"mispronounced dialogue comes from. Spell them the way they should "
                f"be SPOKEN and there is nothing left to choose. They are NOT "
                f"rewritten: your words go to the model as you wrote them")
        _odd = non_latin_in(prompt) + non_latin_in(character_memory or "") \
            + non_latin_in(anchor or "")
        _odd = list(dict.fromkeys(_odd))
        if _odd:
            notes.append(
                f"the prompt contains {len(_odd)} character(s) that are not Latin "
                f"text: {' '.join(_odd[:12])}. A multilingual model reads those as a "
                f"strong signal about which language to speak, and one pasted glyph "
                f"is easy to miss by eye. They are NOT removed -- the node passes "
                f"your words through -- so retype them if the delivery is coming out "
                f"in a language you did not ask for"
                + (f". Your dialogue reads as {_script_lang}, though, so these are "
                   f"most likely meant to be here -- the lines are told they are "
                   f"spoken in {_script_lang}"
                   if _script_lang != SPOKEN_LANGUAGE else ""))
        if mouth_named:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in mouth_named)} have a line, so "
                f"the shot is told who is speaking and every other mouth in it is "
                f"held closed. One of two people speaking still leaves the OTHER "
                f"one's mouth free, and the listener is exactly who invented "
                f"lip-sync lands on")
        if mouth_shut:
            notes.append(
                f"mouths held closed on shot(s) {', '.join(str(n) for n in mouth_shut)} -- "
                f"no scripted line and no effort staged in them. H3 is joint, so the face "
                f"follows the audio branch: the sentence is the picture half and the "
                f"silent conditioning is the half that actually settles it, since a "
                f"lips-closed line loses to a stream that has decided somebody is "
                f"talking. Shots staging effort are left out on purpose -- straining is "
                f"vocal and that mouth should be open")
        if _film_mood == "grim":
            notes.append(
                "the anchor declares the film's tone, so every shot carries \"The mood "
                "is grim.\" and no guessing is done. That is the reliable way to set "
                "it: swept over 512 beats, inferring a mood from the beats alone "
                "called an ORDINARY film grim as often as a duress one, because the "
                "words overlap -- screams is a waterslide, tied is a boat, bound is a "
                "flight to Lisbon, chained is a desk job")
        elif _film_mood == "light":
            notes.append(
                "the anchor declares a light tone, so no grim mood is applied to any "
                "shot whatever the beats say. That is the override for a wrong "
                "reading, and it wins outright")
        elif _film_duress:
            notes.append(
                "no tone is declared in the anchor, and the beats or the character "
                "sheet carry UNAMBIGUOUS duress -- hardware on a body, a captor, an "
                "abduction, being locked in -- so every shot carries \"The mood is "
                "grim.\" Only unambiguous evidence counts here: ordinary coercion "
                "verbs and distress words are not enough on their own, because "
                "grabbing, dragging and screaming are as much a garden centre and a "
                "waterslide as an abduction. Write the tone into the anchor to settle "
                "it either way")
        elif any(beat_duress_strength(b) for b in beats):
            notes.append(
                "some beats read as though they MIGHT stage duress -- coercion or "
                "distress verbs -- but nothing unambiguous, so no mood was applied and "
                "every face is left to the model, whose prior for a described person "
                "is a pleasant posed portrait. If this film has a tone, write it into "
                "the anchor: 'grim', 'tense', 'a kidnapping' and the like turn it on "
                "for every shot, and 'warm' or 'comic' turn it off for good. Measured "
                "over 512 beats, guessing from the beats alone is no better than a "
                "coin toss, so it does not guess")
        if vocal_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in vocal_shots)} have a vocal that "
                f"belongs to somebody -- a whimper, a sob, a moan -- so the shot is "
                f"told whose it is, and the mouths owning neither a line nor a sound "
                f"are closed. Reported as one character's whimpering opening up "
                f"another's ability to babble. A vocal opens the audio branch, which "
                f"is right -- it is meant to be heard -- but the flag saying so was "
                f"shot-level with no owner, and BOTH mouth guards stood down on it "
                f"for everybody in the shot. The person straining should have an open "
                f"mouth; the person watching them should not, and theirs was the face "
                f"an invented voice landed on. Two sources in one shot are named "
                f"separately for the same reason, so the line and the vocal cannot be "
                f"swapped between them. A vocal the beat does not pin on anybody "
                f"holds nobody: closing mouths on a guess could close the mouth "
                f"making the noise. This changes which faces move and never what the "
                f"audio is conditioned on")
        if duress_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in duress_shots)} are told what the "
                f"face is doing, because the scene already stages duress -- restraint "
                f"hardware the sheet lists on somebody in the shot, or your own "
                f"distress verbs in the beat. Reported as somebody smiling at the "
                f"camera in a scene of duress: a four-shot scene of a woman handcuffed "
                f"in a van had not one word in it about anybody's face, and an "
                f"attribute a prompt does not state is not LEFT to the model, it is "
                f"left to the model's prior -- which for a named, described person is "
                f"a portrait, facing the lens, pleasantly, because that is what "
                f"photographs of people are. The eyes have had a clause since "
                f"hold_gaze; the expression never had one. It is one sentence, it "
                f"names no camera, and it reads the staging rather than inventing a "
                f"feeling -- a shot staging neither gets nothing, and a beat that "
                f"already says what the face does is never argued with. Picture only: "
                f"it can never open the audio branch")
        if mouth_acting:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in mouth_acting)} kept their mouths "
                f"because the beat itself puts the mouth to work -- a grin, a yawn, a "
                f"bitten lip, a jaw dropping. The guard holds mouths closed on every "
                f"shot with nobody speaking, and against a face doing nothing that is "
                f"right; against these it was countermanding the only performance "
                f"direction the shot has, in one case word for word. The beat has said "
                f"what the mouth does, so nothing is added over the top. This frees the "
                f"PICTURE only: a smile is silent, and the audio branch is left exactly "
                f"where it was, because a silent expression is the commonest beat there "
                f"is and letting one open a branch would be the invented voice back at "
                f"its widest point. A stare or a wince is a face acting with its mouth "
                f"shut and is still held")
        if muted_sound:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in muted_sound)} gave up the sound you "
                f"wrote for them so the mouths could be held shut. Those shots have no "
                f"line, and a sound alone was enough to leave the audio branch open -- "
                f"which is where the invented voice and the lip-sync came from. This is "
                f"the trade and it is the only one available: the ambience cannot be kept "
                f"while the branch is conditioned to silence. Turn off "
                f"mouths_shut_when_no_line to keep the sound and accept the mouth")
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
                f"pinning where it starts; first_frame pins it.")
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
        _open_br = [i + 1 for i, (s_, snd) in enumerate((shot.speech, shot.sounded) for shot in plan.shots)
                    if not s_ and snd]
        _pinned = [i + 1 for i, (s_, snd) in enumerate((shot.speech, shot.sounded) for shot in plan.shots)
                   if not s_ and not snd]
        n_silent, n_kept = len(_pinned), len(_open_br)
        if silence_nonspeech and n_kept:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in _open_br)} have no line but either "
                f"describe a sound IN THE BEAT or "
                f"stage EFFORT, so their audio is left free to make it -- writing the "
                f"sound, or the verb that produces one, is asking for audio on purpose. "
                f"Those are the only shots without a line "
                f"where the branch is open, and an open branch on a joint model can "
                f"still put a voice in the gap. If one of them babbles, that beat's own "
                f"sound wording is what opened it")
        if silence_nonspeech and n_silent:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in _pinned)} have no quoted line and no "
                f"sound described, so they "
                f"are conditioned on real silence -- which is not 'no speech', it is 'no "
                f"sound at all': no footsteps, no room tone, nothing. H3 is joint, so the "
                f"way to score a scene is to DESCRIBE it in the prose: 'boots on concrete, "
                f"a chain dragging, a low hum off the strip light'. Write it into a beat "
                f"for that shot, or into the anchor to carry it through the film. Do not "
                f"use a label like 'sound:' -- a labelled line is read as text to draw")

        if first_frame is None:
            notes.append(
                "NO first_frame IS WIRED, so shot 1 is the only shot in this film whose "
                "opening frame is pinned by NOTHING. Every other shot opens on the "
                "previous shot's last frame, which fixes its pose, its framing and the "
                "arrangement of everything on the body; shot 1 has only the text and "
                "any reference. So the chain agrees with itself and shot 1 is the one "
                "that can disagree -- which from outside looks like the person changing "
                "between the first beat and the rest, and is also where a one-shot-only "
                "oddity comes from: hair sitting differently against a collar, a "
                "garment hanging differently, a pose the beat did not ask for. "
                "ref_noise_aug IS NOT THE DIAL FOR THIS and raising it cannot help: a "
                "reference says WHO somebody is and a keyframe says what the opening "
                "frame HOLDS, and they are not alternatives -- there is no frame on "
                "shot 1 for a cleaner reference to sharpen. Wire first_frame to fix it"
                + (", and see the ref_noise_aug note above for what to put in it -- it "
                   "pins the WHOLE frame, so a composed frame of the shot you want and "
                   "not an identity portrait"
                   if refs_all else
                   ". It pins the WHOLE frame, so give it a composed frame of the shot "
                   "you want: subject, pose, framing, background. The last frame of a "
                   "previous run, or any still matching how beat 1 should open")
                + ". Leaving it empty is fine when beat 1 is meant to establish the "
                  "look and the rest follow it -- which is what is happening now")
        if any(_CAPTION_TOKEN.search(s) for s in plan.prompts):
            notes.append("the prompt contains H3's caption/lyrics tokens "
                         "(<|caption_start|> and friends) -- those request text ON the "
                         "picture. Remove them unless you want subtitles burned in")
        n_bare = sum(1 for b in beats
                     if _QUOTED.search(b) and not _DIALOGUE_TAG.search(b)
                     and mark_dialogue(b) == b)
        if n_bare:
            notes.append(f"{n_bare} beat(s) carry quotes that were NOT read as speech: "
                         f"no full stop, question mark or exclamation inside them, and "
                         f"no speech verb in front. A quote like that is usually "
                         f"emphasis or a title, so it was left exactly as written. If "
                         f"one of them IS a line, end it with punctuation or mark it "
                         f"yourself with <d>...</d> and it will be spoken rather than "
                         f"drawn")
        cued = sorted({m.group(0).lower() for s in plan.prompts for m in _TEXT_CUE.finditer(s)})
        if cued:
            notes.append(f"the prompt names on-screen text ({', '.join(cued)}) -- H3 draws "
                         f"letterforms when asked, and at cfg 1 no negative prompt can take "
                         f"them back. Remove the words if you do not want the text")
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

        _written = "\n".join([scene or ""] + list(beats))
        _tagged = bool(picture_tags(_written)
                       or any(picture_tags(s) for s in plan.prompts))
        _tagged_names = {n for n, ln in sheet_lines(sheet) if n and picture_tags(ln)}
        _claimed_untagged, _held_untagged = [], []
        if refs_all and not _tagged:
            notes.append(
                f"{len(refs_all)} reference image(s) connected and no <Picture N> tag "
                f"anywhere. A picture the text NAMES is that subject; one it never "
                f"mentions is another subject standing beside them -- which is a second "
                f"person no wording in this node can argue with, because it arrives as a "
                f"picture. So an untagged reference is claimed where the claim is "
                f"unambiguous -- one picture, one person described in the shot, the tag "
                f"written onto their sheet entry -- and held back where it is not. TAG "
                f"IT and neither happens: 'Nora: <Picture 1>, 34, she, ...' sends it into "
                f"the shots Nora is in, and only those")
        for _i, _s in enumerate(plan.prompts):
            if not _tagged:
                _here = [n for n in plan.shots[_i].cast if n]
                if not _here:
                    _here = [n for n, _ln in sheet_lines(sheet)
                             if n and re.search(r"\b" + re.escape(n) + r"\b", _s)]
                if not _here:
                    plan.shots[_i].refs = list(refs_all)
                elif len(refs_all) == 1 and f"{_here[0]}:" in _s and len(_here) == 1:
                    plan.shots[_i].prompt = _s.replace(f"{_here[0]}:", f"{_here[0]}: <Picture 1>,", 1)
                    plan.shots[_i].refs = list(refs_all)
                    _claimed_untagged.append(_i + 1)
                else:
                    plan.shots[_i].refs = []
                    _held_untagged.append(_i + 1)
                continue
            _s, _r, _missing = resolve_tags(_s, refs_all)
            plan.shots[_i].prompt = _s
            plan.shots[_i].refs = _r
            for _n in _missing:
                _msg = f"<Picture {_n}> names a slot with no image connected"
                if _msg not in notes:
                    notes.append(_msg)
        if _claimed_untagged:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in _claimed_untagged)} had the untagged "
                f"reference claimed on the one person they describe, so the picture has a "
                f"subject in the text instead of arriving as a stranger")
        if not sheet_lines(sheet) and refs_all:
            notes.append(
                "a picture is riding with nothing in the text naming it, and there is "
                "no character sheet to write a tag onto. If that picture is a PERSON, "
                "the prompt never says who it is -- and a picture the prompt does not "
                "mention is read as ANOTHER person standing beside the ones described, "
                "which no sentence here can argue with. Give them an entry and tag it: "
                "'Ana: <Picture 1>, she, 30, a grey apron'. If it is a location or a "
                "look, nothing needs doing")
        if _held_untagged:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in _held_untagged)} were sent NO "
                f"reference: more than one picture or more than one person is in them, and "
                f"which picture is whom is not something this node can guess. Sent "
                f"unclaimed it would be a second person in the shot; held back it costs "
                f"likeness there. Tag the pictures -- 'Dan: <Picture 1>, ...' -- and they "
                f"ride every shot that names their subject, claimed")
        _twinned = []
        if refs_all and _tagged_names:
            for _i, _s in enumerate(plan.prompts):
                if not picture_tags(_s):
                    continue
                _cast_here = plan.shots[_i].cast
                _cast_here = [n for n in _cast_here if n] or [
                    n for n, _ in sheet_lines(sheet) if n]
                _bare = [n for n in _cast_here if n not in _tagged_names]
                if _bare and any(n in _tagged_names for n in _cast_here):
                    _twinned.append((_i + 1, _bare))
        if _twinned:
            _who = sorted({n for _, ns in _twinned for n in ns})
            notes.append(
                f"shot(s) {', '.join(str(n) for n, _ in _twinned)} carry a reference "
                f"picture for one person and also describe "
                f"{', '.join(_who)}, who {'has' if len(_who) == 1 else 'have'} no "
                f"<Picture N> of their own. That "
                f"is one photographed face and two people to draw, and a reference is "
                f"the strongest identity signal in the prompt -- much stronger than a "
                f"line of description -- so the face that exists tends to be used "
                f"twice and the second character arrives as a copy of the first. Wire "
                f"a picture of {', '.join(_who)} to a free ref_image slot and tag it "
                f"on their sheet "
                f"{'entries' if len(_who) > 1 else 'entry'} -- "
                f"'{_who[0]}: <Picture 2>, ...' -- so every shot with both of them "
                f"carries both faces. No wording fixes this: nothing in the text "
                f"outranks a photograph")
        if not character_guard and len([n for n, _ in sheet_lines(sheet) if n]) > 1:
            _wardrobes = [f"{n} ({', '.join(garments_in(ln)[:3])})"
                          for n, ln in sheet_lines(sheet) if n and garments_in(ln)]
            notes.append(
                "character_guard is OFF, so EVERY sheet line is in EVERY shot -- "
                "including the wardrobe of everyone the beat does not involve. "
                + ("With " + "; ".join(_wardrobes[:4]) + ", " if _wardrobes else "")
                + "a shot about one person is also describing what the others have on, "
                "and at cfg 1 the model reads the prompt as a bag of words before it "
                "reads a label: a garment listed for one character lands on whichever "
                "body is in frame. Reported as boys wearing stockings. Measured: with "
                "the guard ON, a shot that names only the men carries no word of the "
                "women's clothing at all, because only the people a beat involves are "
                "described. Turn it on. For extras nobody has an entry for, write them "
                "into the beat instead -- your words reach the model verbatim and an "
                "unnamed person needs no entry, though a removal cannot be held for one")
        if refs_all and _tagged and not character_guard:
            notes.append(
                f"character_guard is OFF and {len(refs_all)} reference image(s) are tagged -- "
                f"the combination that fixes the camera on one person. Off, EVERY sheet line "
                f"goes into every shot, including the line carrying <Picture N>, so the "
                f"reference is named in every shot and rides all of them. At ref_noise_aug "
                f"{float(ref_noise_aug):g} that asks the model to reproduce the PICTURE -- "
                f"pose and framing, not only the face -- so the portrait's composition "
                f"becomes every shot's composition, and anyone without a reference is placed "
                f"relative to it. AND TURNING THE GUARD OFF ADDS NOBODY: it describes the "
                f"people your SHEET already names, in every shot, whether the beat involves "
                f"them or not. For extras nobody has a sheet entry for, leave the guard ON "
                f"and write them into the beat -- your words reach the model verbatim and an "
                f"unnamed person needs no entry. To loosen the framing instead, lower "
                f"ref_noise_aug (try 0.95, then 0.90) or crop the reference to head and "
                f"shoulders, so there is less composition in it to reproduce")
        if refs_all:
            _named = sum(1 for s in plan.prompts if picture_tags(s))
            notes.append(
                f"{len(refs_all)} reference image(s) supply IDENTITY, and they go WHERE "
                f"TAGGED: every shot whose text names <Picture N> carries the image on "
                f"ref_image_N, which is what holds a face across beats instead of "
                f"letting it drift down the "
                f"keyframe chain. Put the tag on the person -- 'Nora: <Picture 1>, 34, "
                f"she, ...' -- and it travels with her. {_named} shot(s) claim one here. "
                f"References ride alongside the keyframe rather than instead of it: the "
                f"keyframe anchors the first frame, a reference only says who somebody "
                f"is, and ComfyUI packs both (keyframe rows then ref rows, in the same "
                f"order model_base builds the latents). References keep slots 1..N so the "
                f"tag points at the right image; the handoff is appended after them and "
                f"disturbs no numbering. Expect the NUMBER in script to differ from the "
                f"one you wrote: it is the picture's place in THAT shot's reference "
                f"list, not a name for the image, so a shot carrying one reference "
                f"always says <Picture 1> whichever socket it came from. The image is "
                f"still that person's -- what would be wrong is a shot carrying two "
                f"references and naming only one, since a picture the text never names "
                f"is read as another subject")
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
            if _named < len(plan):
                notes.append(
                    f"{len(plan) - _named} shot(s) name no <Picture N> at all, so they "
                    f"carry no reference. Claim it on the person it depicts -- 'Nora: "
                    f"<Picture 1>, 34, she, ...' -- and it travels with her into the shots "
                    f"she is in, and only those. A picture the prompt never refers to is "
                    f"read as ANOTHER subject")


        _last_a = last_audio_sigma(steps, shift_audio, scheduler, shift_video)
        # A SCHEDULER CAN END THIS OUTRIGHT, and this note used to deny it.
        _alt_sched = scheduler_that_finishes_audio(steps, shift_audio, shift_video,
                                                   scheduler)
        _fix_a = min(shift_audio_for(steps), float(shift_audio or 0.0) or 1.0)
        _soft_landing = bool(apply_model_sampling
                             and not (sigmas is not None and len(sigmas)))
        _landing_on = bool(_soft_landing and _last_a > 0.10)
        if _landing_on:
            notes.append(
                f"the audio branch was landing from sigma {_last_a:.3f} on its final "
                f"step, so ONE extra step is spliced into the end of the schedule to "
                f"put it down at about 0.030 instead. That step costs one model "
                f"evaluation per shot and nothing else: every earlier sigma is exactly "
                f"where '{scheduler}' put it, so the picture keeps the schedule you "
                f"chose. This is the one lever prose cannot reach -- every clause in "
                f"this node changes what the branch is TOLD, and none of them changes "
                f"how much noise it still has to clear when it stops. A branch "
                f"resolving 43% of its denoising in one jump invents whatever is "
                f"easiest, which on a branch told somebody speaks is a voice, and it "
                f"lands at the OPENING of the shot because that is where there is "
                f"least conditioning to anchor it. Choosing a scheduler that already "
                f"finishes the audio"
                + (f" -- '{_alt_sched[0]}' leaves {_alt_sched[1]:.3f}" if _alt_sched
                   else "")
                + " is still the better fix and costs no step; this one fires only "
                "while the tail is steep. It lands at 0.030 whatever shift_audio is "
                "set to -- 1, 3 and 5 all end up there -- so shift_audio does NOT "
                "need tuning by hand for this any more, and the older advice to "
                "lower it does not apply while this is on. Off by wiring your own "
                "`sigmas`, or with apply_model_sampling")
        if _last_a > 0.4 and not _landing_on:
            notes.append(
                f"the audio branch still has sigma {_last_a:.2f} to clear on its FINAL "
                f"step at {int(steps)} steps with shift_audio {float(shift_audio):g} -- "
                f"about {_last_a * 100:.0f}% of its denoising in one jump, and a branch "
                f"resolving that much at once invents whatever is easiest, which is a "
                f"voice. It is the step where babble appears. shift_VIDEO does not "
                f"change this: time_shift_sigma inverts the video shift and re-applies "
                f"the audio one. "
                + (f"'{_alt_sched[0]}' at these same {int(steps)} steps and the same "
                   f"shift_audio leaves {_alt_sched[1]:.3f} instead of {_last_a:.2f}, "
                   f"and it honours shift_video, so the picture keeps the schedule "
                   f"shape you asked for. DO NOT reach for kl_optimal, exponential or "
                   f"karras for this: comfy grades schedulers by use_ms, those three "
                   f"are called with sigma_min and sigma_max ONLY and never see the "
                   f"shift at all, so the video schedule collapses off its high-sigma "
                   f"steps and the picture comes out watery. Reported exactly that "
                   f"way. "
                   if _alt_sched else "")
                + f"Otherwise LOWER shift_audio or raise steps -- sigma rises with "
                f"shift_audio, so raising it makes this worse. shift_audio "
                f"{_fix_a:.2f} at {int(steps)} steps leaves "
                f"{last_audio_sigma(steps, _fix_a, scheduler, shift_video):.2f}, "
                f"against the {DEFAULT_LAST_AUDIO_SIGMA:.2f} the default 3.0 leaves "
                f"at 8 steps on 'simple'.")
        if silence_nonspeech or speech_lead_seconds > 0 or speech_tail_seconds > 0:
            if audio_vae is None:
                notes.append(
                    "SILENCE CANNOT BE APPLIED: no audio VAE is wired to the node's "
                    "audio_vae input, so every shot listed above as conditioned on "
                    "real silence has an audio branch that is NOT pinned. H3 is "
                    "joint, so an unconditioned branch invents a voice and the "
                    "picture lip-syncs to it -- a shot babbling with nothing "
                    "scripted to say")
            elif _silent_audio_latent(audio_vae, lens[0], H3_FPS) is None:
                notes.append(
                    "SILENCE CANNOT BE APPLIED: the VAE on the audio_vae input would "
                    "not encode a silent second, so every shot listed above as "
                    "conditioned on real silence has an audio branch that is NOT "
                    "pinned -- and an unconditioned branch on a joint model invents "
                    "a voice the picture then lip-syncs to. That input wants the "
                    "MiniMax H3 AUDIO vae (minimax_h3_audio_vae.safetensors) in its "
                    "own VAELoader. Every VAE carries an audio_sample_rate "
                    "attribute, so a video VAE wired here passes every check "
                    "until the encode itself fails -- which is caught and "
                    "turned into no conditioning at all")
            else:
                notes.append(
                    f"silence can be applied: the audio VAE encodes silence, so the "
                    f"{n_silent} line-free shot(s) above can be pinned to it rather "
                    f"than merely told to be quiet"
                    + (f", and dialogue gets a {speech_lead_seconds:g}s silent lead-in"
                       if speech_lead_seconds > 0 else "")
                    + ((", and a silent tail past the line on shot(s) "
                        + ", ".join(f"{n} (last {s:.1f}s)" for n, s in _tailpin)
                        + f" -- everything after lead + the line's estimate + "
                        f"{speech_tail_seconds:g}s is pinned, so the branch cannot carry on "
                        f"talking into the seconds the line does not fill. The model chooses "
                        f"when to speak: if a last word is clipped, raise speech_tail_seconds")
                       if _tailpin else ""))
        script = "\n---\n".join(f"[Shot {i}] {s}" for i, s in enumerate(plan.prompts, 1))
        info = " | ".join(notes)
        if plan_only:
            empty = torch.zeros((1, h, w, 3))
            return (empty, {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 44100},
                    "PLAN ONLY -- nothing rendered. " + info, script,
                    lens[0], 0, len(plan), 0.0)

        return PreparedVideo(
            _placed_shots=_placed_shots, _first_is_plate=_first_is_plate,
            _returns=_returns, _soft_landing=_soft_landing, _tagged_names=_tagged_names,
            ambient_audio=ambient_audio, ambient_level=ambient_level, apply_model_sampling=apply_model_sampling,
            audio_vae=audio_vae, auto_sound=auto_sound, bared_shots=bared_shots,
            cfg=cfg, cleanup_between_shots=cleanup_between_shots, clip=clip,
            first_frame=first_frame, foley_level=foley_level, h=h,
            latent_upscale=latent_upscale, latent_upscale_scale=latent_upscale_scale,
            megapixels=megapixels, model=model, moved_shots=moved_shots,
            negative=negative, notes=notes, plan=plan,
            ref_noise_aug=ref_noise_aug, restart_after_removal=restart_after_removal, revealed_shots=revealed_shots,
            sampler_name=sampler_name, scheduler=scheduler, seed=seed,
            shift_audio=shift_audio, shift_video=shift_video,
            sigmas=sigmas, silence_nonspeech=silence_nonspeech, trim_seam=trim_seam,
            speech_lead_seconds=speech_lead_seconds, speech_tail_seconds=speech_tail_seconds, hold_levels=hold_levels, staging_shots=staging_shots, steps=steps,
            stripped_shots=stripped_shots, cut_shots=cut_shots,
            shot_rooms=shot_rooms, hardware_changed=hardware_changed, shot_frames=shot_frames,
            reentry_shots=reentry_shots,
            upscale=upscale, upscale_model=upscale_model,
            upscale_target_short_edge=upscale_target_short_edge, vae=vae, w=w,
        )

    def _render(self, prepared):
        """Execute the prepared shots and assemble the video and soundtrack."""
        _placed_shots = prepared._placed_shots
        _first_is_plate = prepared._first_is_plate
        _returns = prepared._returns
        _soft_landing = prepared._soft_landing
        _tagged_names = prepared._tagged_names
        shot_rooms = prepared.shot_rooms or {}
        _shot_frames = prepared.shot_frames or {}
        reentry_shots = prepared.reentry_shots or {}
        hardware_changed = prepared.hardware_changed or set()
        ambient_audio = prepared.ambient_audio
        ambient_level = prepared.ambient_level
        apply_model_sampling = prepared.apply_model_sampling
        audio_vae = prepared.audio_vae
        auto_sound = prepared.auto_sound
        bared_shots = prepared.bared_shots
        cfg = prepared.cfg
        cleanup_between_shots = prepared.cleanup_between_shots
        clip = prepared.clip
        first_frame = prepared.first_frame
        foley_level = prepared.foley_level
        h = prepared.h
        latent_upscale = prepared.latent_upscale
        latent_upscale_scale = prepared.latent_upscale_scale
        megapixels = prepared.megapixels
        model = prepared.model
        moved_shots = prepared.moved_shots
        negative = prepared.negative
        notes = prepared.notes
        plan = prepared.plan
        ref_noise_aug = prepared.ref_noise_aug
        restart_after_removal = prepared.restart_after_removal
        revealed_shots = prepared.revealed_shots
        sampler_name = prepared.sampler_name
        scheduler = prepared.scheduler
        seed = prepared.seed
        shift_audio = prepared.shift_audio
        shift_video = prepared.shift_video
        sigmas = prepared.sigmas
        silence_nonspeech = prepared.silence_nonspeech
        speech_lead_seconds = prepared.speech_lead_seconds
        speech_tail_seconds = prepared.speech_tail_seconds
        hold_levels = prepared.hold_levels
        staging_shots = prepared.staging_shots
        steps = prepared.steps
        stripped_shots = prepared.stripped_shots
        cut_shots = prepared.cut_shots
        trim_seam = prepared.trim_seam
        upscale = prepared.upscale
        upscale_model = prepared.upscale_model
        upscale_target_short_edge = prepared.upscale_target_short_edge
        vae = prepared.vae
        w = prepared.w

        def _frame_cast(k, last=False):
            _c = [n for n in plan.shots[k].cast if n]
            return list(_shot_frames.get(k, (_c, _c))[1 if last else 0])

        if apply_model_sampling:
            model, ms_note = apply_h3_model_sampling(model, shift_video, shift_audio)
            notes.append(ms_note)
        if negative is None:
            negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

        handoff = first_frame
        t_sample = t_decode = 0.0
        _aug_warned = False
        fresh = []
        t_start = time.perf_counter()
        aud_out, sr = [], 44100
        frame_capacity = sum(shot.frame_count for shot in plan.shots)
        frames = FrameAccumulator(frame_capacity, _image_out_dtype(), cleanup_between_shots)
        # Reachable from run(), so an interrupt can drop it before unwinding. See there.
        self._frames = frames
        av_fix = 0                  # samples of A/V drift corrected across the chain
        _captured = {}              # name -> a frame from the last shot they were in
        _captured_from = {}         # name -> which shot that frame came from
        _captured_gen = {}          # name -> the wardrobe generation that frame shows
        _soft_cuts = []             # (shot, why) cuts that carried the frame as a reference
        _recovered = []             # (shot, name, source shot) actually pinned
        _evened = []                # (shot, name, source shot) given a face beside a tagged one
        _room_frames = {}           # room -> [(last frame there, who was in it, wardrobe generation, shot)], newest first
        _room_returns = []          # (shot, room, source shot) actually carried
        _wardrobe_gen = 0           # bumped by every shot that changes what anybody wears or is held by
        _handoff_claimed = []       # shots whose opening frame is named in the text
        _aug_claimed = []
        _untrimmed = []             # shots that opened on no keyframe, so kept frame one
        _plate_on = 0               # the shot whose first_frame rides as the SET
        _carried = []               # (shot, who was there, who joins) room carried on
        shot_detail = []            # (detail, contrast) per shot, on its last frame
        _levels = HandoffLevels()
        _SILENCE_STATUS.update(asked=0, applied=0, why="")
        _deep_cleanup()

        for i, shot in enumerate(plan.shots):
            shot_prompt = shot.prompt
            _audio = ShotAudio(plan.shots[i].speech, plan.shots[i].sounded, plan.shots[i].voiced_only,
                               bool(silence_nonspeech), speech_lead_seconds,
                               AUDIO_LATENT_FPS, shot.line_seconds, speech_tail_seconds,
                               shot.frame_count)
            silent = _audio.pinned

            shot_handoff = handoff
            _handoff_ref = False
            _prev_people = _frame_cast(i - 1, last=True) if i else []
            _carry_ok = bool(i and handoff is not None and i not in reentry_shots
                             and (_cond_module.may_carry_room if i in cut_shots
                                  else _cond_module.may_carry_frame)(
                                 _prev_people, plan.shots[i].cast, _tagged_names))
            _carry_rooms = None
            if restart_after_removal and (i - 1) in stripped_shots:
                if _carry_ok:
                    _handoff_ref = True
                    _carry_rooms = (shot_rooms.get(i - 1, ("", ""))[1],
                                    shot_rooms.get(i, ("", ""))[0])
                    _soft_cuts.append((i + 1, "removal"))
                else:
                    shot_handoff = None
                    fresh.append(i + 1)
            elif i in cut_shots and _carry_ok:
                _handoff_ref = True
                _carry_rooms = (shot_rooms.get(i - 1, ("", ""))[1],
                                shot_rooms.get(i, ("", ""))[0])
                _soft_cuts.append((i + 1, "room"))
            elif i in cut_shots or i in reentry_shots:
                shot_handoff = None
            elif i == 0 and _first_is_plate and shot_handoff is not None:
                _handoff_ref = True
                _plate_on = i + 1
            elif i in _placed_shots:
                _was_here = _frame_cast(i - 1, last=True)
                _here_now = plan.shots[i].cast
                if _cond_module.may_carry_frame(_was_here, _here_now, _tagged_names):
                    _handoff_ref = True
                    _carried.append((i + 1, list(_was_here),
                                     list(_placed_shots[i])))
                else:
                    shot_handoff = None
                    fresh.append(i + 1)

            _extra = []
            _evened_who = ""            # who the evening-up frame below pictures
            _cast = plan.shots[i].cast
            _returning = {w for n, ws in _returns if n == i + 1 for w in ws}
            _who = _cond_module.recoverable_subject(
                _cast, _tagged_names, _returning,
                {k: v for k, v in _captured.items()
                 if _captured_gen.get(k) == _wardrobe_gen})
            if _who and _carry_rooms is not None and _who in _prev_people:
                _who = ""               # the carried frame is already a picture of them
            if _who:
                _extra = [_captured[_who]]
                _recovered.append((i + 1, _who, _captured_from.get(_who, 0)))
                _n = len(shot.refs) + 1
                _tag = f"<Picture {_n}>"
                if f"{_who}:" in shot_prompt:
                    shot_prompt = shot_prompt.replace(
                        f"{_who}:", f"{_who}: {_tag},", 1)
                else:
                    shot_prompt = f"{shot_prompt} {_who} is the person in {_tag}."
            elif _tagged_names and len(_cast) > 1 and any(n in _tagged_names for n in _cast):
                _short = [n for n in _cast
                          if n and n not in _tagged_names
                          and _captured.get(n) is not None
                          and _captured_gen.get(n) == _wardrobe_gen]
                if (len(_short) == 1 and f"{_short[0]}:" in shot_prompt
                        and not ((_carry_rooms is not None or _handoff_ref)
                                 and _short[0] in _prev_people)):
                    _extra = [_captured[_short[0]]]
                    _evened_who = _short[0]
                    _evened.append((i + 1, _short[0], _captured_from.get(_short[0], 0)))
                    _tag = f"<Picture {len(shot.refs) + 1}>"
                    shot_prompt = shot_prompt.replace(
                        f"{_short[0]}:", f"{_short[0]}: {_tag},", 1)
            _pictured_here = {n for n in (_who, _evened_who) if n}
            _opens, _ends = shot_rooms.get(i, ("", ""))
            _prev_end = shot_rooms.get(i - 1, ("", ""))[1] if i else ""
            _back, _arriving = "", False
            if (i in cut_shots or i in reentry_shots) and _opens in _room_frames:
                _back = _opens
            elif _ends and _ends != _prev_end and _ends != _opens and _ends in _room_frames:
                _back, _arriving = _ends, True
            if _back:
                _cast_now = set(plan.shots[i].cast)
                _in_keyframe = (set(_frame_cast(i - 1, last=True))
                                if (_arriving and i and shot_handoff is not None) else set())
                for _frame, _in_it, _gen, _from in _room_frames[_back]:
                    if (_gen == _wardrobe_gen
                            and all(n in _cast_now for n in _in_it)
                            and not any(n in _tagged_names for n in _in_it)
                            and not any(n in _in_keyframe for n in _in_it)
                            and not any(n in _pictured_here for n in _in_it)
                            and not (_carry_rooms is not None
                                     and any(n in _prev_people for n in _in_it)
                                     and not all(n in _in_it for n in _prev_people))):
                        if _carry_rooms is not None and any(n in _prev_people for n in _in_it):
                            _handoff_ref, shot_handoff, _carry_rooms = False, None, None
                            _soft_cuts.pop()
                        _extra.append(_frame)
                        shot_prompt = shot_prompt + returning_room_claim(
                            len(shot.refs) + len(_extra), _back, _in_it, _arriving)
                        _room_returns.append((i + 1, _back, _from))
                        break
            _shot_refs = list(shot.refs) + _extra
            if _handoff_ref and _plate_on == i + 1:
                # A SET, not a room a moment earlier. See plate_claim.
                shot_prompt = shot_prompt + plate_claim(len(_shot_refs) + 1)
                _handoff_claimed.append(i + 1)
            elif _carry_rooms is not None:
                _was_room, _now_room = _carry_rooms
                if _was_room and _now_room and _was_room != _now_room:
                    shot_prompt = shot_prompt + carried_people_claim(
                        len(_shot_refs) + 1, _prev_people, _was_room, _now_room)
                else:
                    shot_prompt = shot_prompt + room_claim(len(_shot_refs) + 1,
                                                           _prev_people, [])
                shot_prompt = recount_with_claim(shot_prompt, plan.shots[i].cast,
                                                 _prev_people)
                _handoff_claimed.append(i + 1)
            elif _handoff_ref:
                _was, _join = next(((w, j) for s, w, j in _carried if s == i + 1),
                                   ([], []))
                shot_prompt = shot_prompt + room_claim(len(_shot_refs) + 1, _was, _join)
                shot_prompt = recount_with_claim(shot_prompt, plan.shots[i].cast, _was)
                _handoff_claimed.append(i + 1)
            elif handoff_rides_as_ref(shot_handoff, _shot_refs, ref_noise_aug):
                shot_prompt = shot_prompt + handoff_claim(len(_shot_refs) + 1)
                _handoff_claimed.append(i + 1)
                _aug_claimed.append(i + 1)
            # Whatever this shot ends up being, that is what `script` reports.
            shot.prompt = shot_prompt
            cond, latent, fc, demoted = build_conditioning(
                clip, vae, audio_vae, shot_prompt, w, h, shot.frame_count,
                handoff=shot_handoff, refs=list(shot.refs) + _extra,
                ref_noise_aug=ref_noise_aug, silent=silent,
                handoff_as_ref=_handoff_ref,
                speech_lead_seconds=(_audio.lead_frames / AUDIO_LATENT_FPS),
                speech_tail_frames=_audio.tail_frames)
            if (demoted and not _aug_warned and ref_noise_aug is not None
                    and float(ref_noise_aug) < KEYFRAME_SAFE_AUG):
                _aug_warned = True
                notes.append(
                    f"ref_noise_aug is {float(ref_noise_aug):g}, below {KEYFRAME_SAFE_AUG:g} -- "
                    f"one aug covers references AND the keyframe, so at this value the "
                    f"anchor would be noised and mis-timestepped, and every shot after the "
                    f"first degrades while sampling. The handoff is riding as an extra "
                    f"reference instead: continuity is weaker but nothing is corrupted. "
                    f"Raise it to {KEYFRAME_SAFE_AUG:g}+ for a real keyframe")
            _evict_all_but(model, latent)
            try:
                _t0 = time.perf_counter()
                out = sample_shot(model, cond, negative, latent, seed, steps, cfg,
                                  sampler_name, scheduler, sigmas,
                                  shift_video, shift_audio, _soft_landing)
                t_sample += time.perf_counter() - _t0
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom(e):
                    raise
                raise RuntimeError(
                    f"H3-LongVideos: shot {i + 1} of {len(plan)} ran out of VRAM while "
                    f"sampling. " + sampling_oom_help(w, h, fc, H3_FPS, megapixels)) from e

            try:
                parts = out["samples"].unbind() if hasattr(out["samples"], "unbind") else None
            except Exception:
                parts = None

            shot_tiled = not decode_fits_untiled(vae, out)
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
            imgs = _decode_video(vae, out, shot_tiled, free_first=model,
                                 keep=(vae, audio_vae))
            wav = _decode_audio(audio_vae, out)
            t_decode += time.perf_counter() - _t0
            sr = wav["sample_rate"]
            del out

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
            try:
                if (shot_handoff is not None and not demoted and imgs is not None
                        and imgs.shape[0] > 1 and hand_src is not None and hand_src.shape[0]):
                    _levels.observe(shot_handoff, imgs[0], imgs[-1], hand_src[-1])
            except Exception:
                pass
            _grade = None               # the correction, kept for the captured face
            try:
                if hold_levels > 0 and hand_src is not None and hand_src.shape[0]:
                    _lg, _lo = _levels.gains(hold_levels)
                    if _lg is not None:
                        hand_src = apply_levels(hand_src, _lg, _lo)
                        _grade = (_lg, _lo)
                        _levels.note(_lg, _lo)   # recorded for the end-of-run report
            except Exception:
                pass
            handoff = hand_src[-1:].detach().clamp(0.0, 1.0).to("cpu", copy=True)
            _n = i + 1
            _wardrobe_normal = not (i in stripped_shots
                                    or _n in moved_shots
                                    or _n in revealed_shots
                                    or _n in bared_shots
                                    or _n in staging_shots)
            try:
                if (imgs.shape[0] and len(plan.shots[i].cast) == 1
                        and len(_frame_cast(i)) == 1 and _wardrobe_normal):
                    _mid = imgs.shape[0] // 2
                    _keep = imgs[_mid:_mid + 1]
                    if _grade is not None:
                        _keep = apply_levels(_keep, _grade[0], _grade[1])
                    _keep = _keep.detach().clamp(0.0, 1.0).to("cpu", copy=True)
                    for _who in plan.shots[i].cast:
                        _captured[_who] = _keep
                        _captured_from[_who] = i + 1
                        _captured_gen[_who] = _wardrobe_gen
            except Exception:
                pass                       # a recovered frame is a nicety, not the render
            if not _wardrobe_normal or _n in hardware_changed:
                _wardrobe_gen += 1
            else:
                _room_end = shot_rooms.get(i, ("", ""))[1]
                try:
                    if _room_end and hand_src.shape[0]:
                        _room_frames[_room_end] = ([(
                            hand_src[-1:].detach().clamp(0.0, 1.0).to("cpu", copy=True),
                            _frame_cast(i, last=True), _wardrobe_gen, i + 1)]
                            + _room_frames.get(_room_end, []))[:3]
                except Exception:
                    pass                   # a carried room is a nicety, not the render
            del hand_src
            if trim_seam and i > 0 and shot_handoff is not None and not demoted:
                imgs = imgs[1:]
                wav["waveform"] = wav["waveform"][..., max(0, round(sr / H3_FPS)):]
            elif trim_seam and i > 0:
                _untrimmed.append(i + 1)
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
            try:
                if handoff is not None and handoff.shape[0]:
                    shot_detail.append(frame_detail(handoff[0]))
                elif imgs is not None and imgs.shape[0]:
                    shot_detail.append(frame_detail(imgs[-1]))
            except Exception:
                pass
            frames.add(imgs)
            aud_out.append(wav["waveform"].to("cpu", copy=True) if cleanup_between_shots
                           else wav["waveform"])
            del imgs, wav
            if cleanup_between_shots:
                _deep_cleanup()

        if _untrimmed:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in _untrimmed)} kept their FIRST frame "
                f"even though trim_seam is on, because they did not open on a keyframe. "
                f"The trim exists to drop a duplicate -- the model's own reproduction of "
                f"the frame it was handed -- and these shots were handed none: the chain "
                f"is broken deliberately after a removal, and a character introduced "
                f"already in position gets the previous frame as a REFERENCE rather than "
                f"as frame one. Their first frame is the real opening frame of a cut, so "
                f"trimming it threw away footage and left frame TWO meeting the shot "
                f"before -- reported as the last frame and the first frame of the next "
                f"beat not matching up. The audio is trimmed with the picture or not at "
                f"all, so the two cannot come apart")
        if _handoff_claimed and not _aug_claimed:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in _handoff_claimed)} name the frame "
                f"they open on in their own text -- the set, or the room and who was in "
                f"it. A picture the prompt never refers to is read as another subject, "
                f"so an unnamed one would arrive as a second person with the same face "
                f"and the same clothes")
        if _aug_claimed:
            notes.append(
                f"ref_noise_aug is below {KEYFRAME_SAFE_AUG:g}, so on shot(s) "
                f"{', '.join(str(n) for n in _aug_claimed)} the handoff is encoded as "
                f"a reference rather than a keyframe, and the text now NAMES it as the "
                f"frame the shot opens on. Unnamed it was a picture of the previous shot "
                f"-- the same people, a moment earlier -- sitting in the reference rows "
                f"with nothing claiming it, and a picture the prompt never names is read "
                f"as another subject. That is a duplicate of whoever was on screen, "
                f"appearing on the later shots because those are the ones with both a "
                f"handoff and a reference. Raising ref_noise_aug to {KEYFRAME_SAFE_AUG:g} "
                f"or above keeps the handoff a keyframe and the question does not arise")
        if _room_returns:
            notes.append(
                "carried a room back: "
                + "; ".join(f"the {room} on shot {n}, from shot {src}"
                            for n, room, src in _room_returns)
                + ". The film returns to a room it already showed, and without a picture "
                  "the words rebuild it as a different room. The last frame from the shot "
                  "that was last there went in as a reference, claimed as that room. Only "
                  "where everybody in that frame is in the shot and nobody's clothes or "
                  "hardware have changed since, or the picture would carry the old ones back")
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
                  "face and the same clothes. The tag is in the `script` output: the "
                  "finished prompt is written back to the shot, and `script` is built "
                  "from those at the end of the run")
        video = frames.finish()
        if upscale and upscale != "off":
            video, up_note = _upscale_frames(video, upscale, upscale_model,
                                             upscale_target_short_edge,
                                             upscale_batch_for(video))
            if up_note:
                notes.append(up_note)
        audio = torch.cat(aud_out, dim=-1)
        if audio.dtype != torch.float32:
            audio = audio.float()
        _bed_in = ambient_audio
        if _bed_in is None and float(ambient_level or 0.0) > 0.0:
            notes.append(
                f"ambient_level is {float(ambient_level):.2f} and nothing is wired to "
                f"ambient_audio, so no bed went under the soundtrack -- and that is "
                f"now the only way to get one. The node used to BUILD a room tone out "
                f"of the scene's wording, and a layer of foley into every shot pinned "
                f"to silence; both are gone, because they were reported as sounding "
                f"horrid and synthesis that measures right and sounds wrong is the end "
                f"of that road. The audio is the model's, whole. This widget still "
                f"sets the level for a recording you wire yourself, which is played "
                f"under the finished track and conditions nothing")
        if auto_sound and float(foley_level or 0.0) > 0.0:
            notes.append(
                f"foley_level is {float(foley_level):.2f} and does nothing any more. "
                f"It set how loud the sounds this node BUILT were -- a click, a "
                f"rattle, a rustle, mixed into the shots whose audio branch is pinned "
                f"to silence, because prompt text can never open a branch and those "
                f"shots could not make their own. That is removed: the soundtrack is "
                f"the model's. The widget stays at this position because saved "
                f"workflows restore values by position and shifting it would load the "
                f"wrong number into every widget after it. "
                f"THE CONSEQUENCE, said rather than left to be found: a shot with no "
                f"line and no sound you described is pinned to silence and is SILENT. "
                f"The pin is deliberate and untouched -- it is what stops a free "
                f"branch filling itself with a voice and the face lip-syncing to the "
                f"babble. To put sound in such a shot, write the sound into that beat, "
                f"which opens its branch on purpose and lets the model make it; or "
                f"wire a track to ambient_audio; or lay one under the finished video "
                f"outside the node")
        audio, _bed_note = mix_ambient(audio, sr, _bed_in, ambient_level)
        if _bed_note:
            notes.append(_bed_note)
        total = video.shape[0]
        if cleanup_between_shots and total:
            _bytes = video.element_size()
            _held = total * int(w) * int(h) * 3 * _bytes / GB
            _dt = "float16" if _bytes == 2 else "float32"
            if _held >= 2.0:
                notes.append(
                    f"the finished chain is {_held:.1f}GB in system RAM ({total} frames at "
                    f"{w}x{h}, {_dt}). It "
                    f"shares that RAM with the models, which ComfyUI offloads to it "
                    f"rather than discarding: while they fit, a shot boundary is a PCIe "
                    f"copy; once the frames crowd them out it becomes a disk read, once "
                    f"per model per shot. If the machine is thrashing, the levers are "
                    f"fewer frames per run (lower shot_seconds, or split a long script "
                    f"and join the parts outside the node), a lower megapixels, or a "
                    f"smaller diffusion quant -- every GB of weights is a GB not "
                    f"available to hold the render")
        if _evened:
            notes.append(
                "; ".join(f"shot {n} gave {who} a face of their own, from shot {src}"
                          for n, who, src in _evened)
                + " -- each of those shots carried a reference for somebody else and "
                  "described them with none, which is one photographed face and two "
                  "people to draw. A reference is the strongest identity signal in a "
                  "prompt, so the one that exists gets used for both bodies and the "
                  "second character arrives as a copy of the first. The frame sent is "
                  "one this run rendered, from a shot that held them alone in the "
                  "clothes they are wearing now, and it is claimed on their own sheet "
                  "entry. Tagging them with a <Picture N> of their own does the same "
                  "thing from the first shot instead of the second"
            )
        if _soft_cuts:
            notes.append(
                "carried the previous frame as a REFERENCE across a cut -- "
                + "; ".join(f"shot {n} ({'something came off in the shot before' if why == 'removal' else 'it opens in another room'})"
                            for n, why in _soft_cuts)
                + ". Not as frame one, so a garment left half off is not pinned into the "
                  "opening and the old room is not blended into a new one, but the faces, "
                  "hair and clothes come with it instead of being re-imagined from the text")
        if fresh:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in fresh)} start fresh, because the shot "
                f"before each took something off and its last frame could not ride as a "
                f"reference -- nobody is left in it to claim, or somebody in it has a "
                f"portrait of their own riding the next shot. Continuing from a frame that "
                f"may still show the garment is how "
                f"it comes back, and a picture outvotes the text. That costs a cut there, "
                f"with nothing carried. Turn restart_after_removal off to keep the "
                f"continuity instead")
        if _carried:
            notes.append(
                "; ".join(
                    f"shot {s} carries the previous frame as a REFERENCE rather than "
                    f"as its first frame, so the room, the light and "
                    f"{' and '.join(w)} come with it while "
                    f"{' and '.join(j)} {'are' if len(j) > 1 else 'is'} already in "
                    f"place instead of walking in"
                    for s, w, j in _carried)
                + " -- a keyframe is frame one and a reference is not, which is what "
                  "lets a shot introduce somebody without re-imagining the room")
        wall = time.perf_counter() - t_start
        n = max(1, len(plan))
        other = max(0.0, wall - t_sample - t_decode)
        notes.append(
            f"rendered {total} frames (~{total / H3_FPS:.1f}s) in {wall:.0f}s -- "
            f"sampling {t_sample:.0f}s ({100 * t_sample / wall:.0f}%), "
            f"decode {t_decode:.0f}s ({100 * t_decode / wall:.0f}%), "
            f"other {other:.0f}s ({100 * other / wall:.0f}%); "
            f"per shot {t_sample / n:.1f}s + {t_decode / n:.1f}s")
        if av_fix:
            per_shot = abs(av_fix) / sr * 1000 / max(1, len(plan))
            notes.append(
                f"audio realigned to the picture by ~{abs(av_fix) / sr * 1000:.0f} ms "
                f"across {len(plan)} shot(s), {per_shot:.1f} ms each. H3's audio latent "
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
        _lvl = levels_report(_levels, len(plan.shots), hold_levels)
        if _lvl:
            notes.append(_lvl)
        if t_decode > t_sample:
            notes.append("decode is costing more than sampling here -- latent_upscale "
                         "trades cheaper sampling for a 4x more expensive decode, so it "
                         "is the wrong way round at this step count. megapixels is the "
                         "lever that lowers both")
        script = "\n---\n".join(f"[Shot {i}] {s}" for i, s in enumerate(plan.prompts, 1))
        if _SILENCE_STATUS["asked"]:
            _missed = _SILENCE_STATUS["asked"] - _SILENCE_STATUS["applied"]
            if _missed > 0:
                notes.append(
                    f"SILENCE WAS ASKED FOR ON {_SILENCE_STATUS['asked']} shot(s) AND "
                    f"WENT ON {_SILENCE_STATUS['applied']}: {_missed} shot(s) have no "
                    f"working audio lock, because "
                    f"{_SILENCE_STATUS['why'] or 'the silent latent could not be built'}"
                    f". H3 is joint, so an unconditioned branch invents a voice and the "
                    f"picture lip-syncs to it -- a shot babbling with nothing scripted "
                    f"to say. The lips-closed sentence is still in the prompt and still "
                    f"loses to the stream")
            else:
                notes.append(
                    f"silence went on all {_SILENCE_STATUS['applied']} shot(s) that "
                    f"asked for it -- the requested full shot or dialogue lead-in is "
                    f"pinned to encoded silence, not merely told to be quiet")
        return (video, {"waveform": audio, "sample_rate": sr}, " | ".join(notes), script,
                plan.shots[0].frame_count, total, len(plan), round(total / H3_FPS, 2))


_NODE_IDS = ("H3LongVideos", "H3LongVideosFL2VA", "H3LongVideosV1",
             "H3LongVideosREF2VA")
NODE_CLASS_MAPPINGS = {name: H3LongVideos for name in _NODE_IDS}
NODE_DISPLAY_NAME_MAPPINGS = {name: "H3-LongVideos" for name in _NODE_IDS}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
