# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""Plan MiniMax-H3 shots, render their audio/video, and preserve continuity.

The node interface and prompt planning live here. Audio policy and synthesis,
conditioning assembly, and tensor/runtime operations have separate owner modules.
"""

import glob
import inspect
import json
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
# Pose control for restrained characters, loaded the same way. A module that fails to
# load costs pose control only, and says so in info; the node itself still loads.
try:
    pose_control = _load_local("h3_pose_control", "pose_control.py")
    _POSE_LOAD_ERROR = ""
except Exception as _pose_e:          # pragma: no cover - only on a broken install
    pose_control = None
    _POSE_LOAD_ERROR = f"{type(_pose_e).__name__}: {_pose_e}"
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
ensure_host_ram = _runtime_module.ensure_host_ram
_decode_ram = _runtime_module._decode_ram
shot_grade = _runtime_module.shot_grade
grade_frames = _runtime_module.grade_frames
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


_SHEET_LINE = re.compile(r"^\s*(?!(?i:remove|off|add|wear|wardrobe|hold)\s*:)"
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


# Into a PERSON is not into a room: "Dan enters her", "comes inside her", "slips into
# Kate". Read as an arrival, a partner already in the frame was staged walking in
# again, and the shot threw its keyframe away -- the room and both bodies redrawn in
# the middle of a scene nobody left. "Enters her bedroom" is still an entrance.
_NOT_INTO_A_BODY = (
    r"(?!\s+(?:her|him|them|me|you|us|herself|himself|themselves)\b"
    r"(?!\s+(?:[\w-]+\s+)?(?:" + engine.PLACES + r"|house|home|building|flat|"
    r"apartment|car)\b)"
    r"|\s+(?:(?:her|his|their)\s+)?(?:mouth|body|throat|ass|arse|anus|vagina|pussy|"
    r"cunt|cock|penis|dick|hand|hands|fist|thighs?)\b"
    r"|(?-i:\s+(?!(?i:" + engine.PLACES + r"|house|home|building|flat|apartment)\b)"
    r"[A-Z][a-z-]+\b(?!['\u2019]s)))")
_ENTRANCE = re.compile(
    r"\b(?:walk|step|come|run|stride|hurry|move|wander|burst|barge|slip|climb)"
    r"(?:s|ed|ing)?\s+(?:in|into|through|up|over|back|out\s+of)\b" + _NOT_INTO_A_BODY
    + r"|\benter(?:s|ed|ing)?\b" + _NOT_INTO_A_BODY + r"|\barriv(?:es?|ed|ing)\b"
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
    return list(dict.fromkeys(_movers(_EXIT, beat, sheet, present, alone_is_it=True)
                              + taken_out(beat, sheet, present, "out")))


_COMES_IN = re.compile(
    r"\b(?:walk(?:s|ed|ing)?|com(?:e|es|ing)|came|step(?:s|ped|ping)?|run(?:s|ning)?|ran|"
    r"hurr(?:y|ies|ied|ying)|burst(?:s|ing)?|barg(?:e|es|ed|ing)|slip(?:s|ped|ping)?|"
    r"strid(?:e|es|ing)|strode|stroll(?:s|ed|ing)?|wander(?:s|ed|ing)?|rush(?:es|ed|ing)?|"
    r"storm(?:s|ed|ing)?|march(?:es|ed|ing)?|sneak(?:s|ed|ing)?|snuck|limp(?:s|ed|ing)?|"
    r"stagger(?:s|ed|ing)?)\s+(?:\w+ly\s+)?(?:back\s+)?"
    r"(?:in\b(?!\s+(?:the|a|an|his|her|their)\b)|inside\b" + _NOT_INTO_A_BODY
    + r"|into\s+(?:the|this|that|a)\s+(?:[\w-]+\s+)?(?:" + _EXIT_ROOMS
    + r"|house|building|apartment|flat)\b)"
    r"|\benter(?:s|ed|ing)?\b(?!\s+(?:the|a|his|her)\s+(?:code|number|password|data|pin)\b)"
    + _NOT_INTO_A_BODY +
    r"|\barriv(?:e|es|ed|ing)\b"
    r"|\b(?:com(?:e|es|ing)|came)\s+back\b(?=\s*(?:[.,;:!?]|$)"
    r"|\s+(?:in|into|inside|home|with|and|carrying|holding)\b)"
    r"|\breturn(?:s|ed|ing)?\b(?!\s+(?:the|a|an|his|her|their|it|them|to\s+(?:the|his|her|their)\s+"
    r"(?:table|desk|couch|sofa|chair|bed|seat|work|book|screen|sink|stove|counter)))",
    re.I)


def comes_in(beat, sheet):
    """The people this beat stages ARRIVING in the frame -- see _COMES_IN."""
    _all = [n for n, _ in sheet_lines(sheet) if n]
    return list(dict.fromkeys(_movers(_COMES_IN, beat, sheet, _all)
                              + taken_out(beat, sheet, _all, "in")))


_TAKES = (r"(?:leads?|led|leading|drags?|dragged|dragging|carries|carried|carrying|takes?|"
          r"took|taking|pulls?|pulled|pulling|pushes|pushed|pushing|escorts?|escorted|"
          r"escorting|march(?:es|ed|ing)?|frog-?march(?:es|ed|ing)?|guides?|guided|"
          r"hauls?|hauled|walks?|walked|shoves?|shoved|brings?|brought|bringing|"
          r"wheels?|wheeled|steers?|steered|herds?|herded)")
_TAKEN_TO = {
    "out": (r"(?:out\b(?!\s+of\s+(?:her|his|their)\s+\w+)|outside\b|out\s+of\s+the\s+\w+|"
            r"through\s+the\s+(?:side\s+|back\s+|front\s+)?door(?:way)?\b)"),
    "in": (r"(?:back\s+)?(?:in\b(?!\s+(?:the|a|an|his|her|their)\b)|inside\b|"
           r"into\s+the\s+(?:room|cell|basement|warehouse|house|kitchen|bedroom|hall)\b)"),
}


def taken_out(beat, sheet, present=(), way="out"):
    """Who a beat TAKES out (way="out") or brings in ("in") -- the one doing it and the
    one it is done to. "Dan leads Ana out through the side door" moved nobody: the exit
    reader knew only people leaving on their own feet, and being led, dragged or carried
    out is how a restrained person leaves a scene. So she stayed described -- cuffs and
    collar -- in the shots after she had gone, and, never having left, got no picture of
    herself when she was brought back. REPORTED as restraints and clothing not looking
    the same when a character leaves the shot and comes back."""
    b = str(beat or "")
    rows = [(n, ln) for n, ln in sheet_lines(sheet) if n]
    out = []
    for m in re.finditer(r"\b" + _TAKES + r"\s+(\S+)\s+(?:\w+\s+){0,2}?" + _TAKEN_TO[way],
                         b, re.I):
        # The doer is the nearest name before the verb in the same sentence: "Dan comes
        # back AND leads Ana out" puts two verbs between him and the one that moves her.
        _sentence = re.split(r"[.;!?]", b[:m.start()])[-1]
        _before = [(mm.start(), n) for n, _l in rows
                   for mm in re.finditer(r"\b" + re.escape(n) + r"\b", _sentence)]
        n = max(_before)[1] if _before else ""
        obj = m.group(1).strip(".,;").lower()
        whom = next((o for o, _l in rows if o.lower() == re.sub(r"'s$", "", obj)), "")
        if not whom and obj in ("her", "him", "them"):
            group = {"her": "she", "him": "he", "them": "they"}[obj]
            fits = [o for o, ol in rows if o != n and sheet_pronoun(ol) == group
                    and (not present or o in present)]
            whom = fits[0] if len(fits) == 1 else ""
        if whom and whom != n:
            out += ([n] if n else []) + [whom]
    return list(dict.fromkeys(out))


_LEFT_ALONE = re.compile(
    r"(?<!\bnot\s)(?<!\bnever\s)(?<!\blonger\s)"
    r"\b(?:alone|by\s+(?:her|him)self|on\s+(?:her|his)\s+own)\b", re.I)


def left_alone(beat, sheet, in_frame):
    """The one person a beat leaves ALONE in the frame, or "".

    "Dan walks away from Ana and she stands alone by the crates" left both of them in
    the last frame -- walking away is not walking out -- so there was never a frame of
    her by herself, in her cuffs and her clothes, to bring back when she returned. The
    person is the nearest one named before "alone", or the pronoun only one of them
    answers to; "Dan and Ana are alone" is both of them, and is nobody's."""
    b = str(beat or "")
    if len(in_frame or ()) < 2:
        return ""
    m = _LEFT_ALONE.search(b)
    if not m:
        return ""
    clause = re.split(r"[.;!?,]|\b(?:but|while|as|until)\b", b[:m.start()])[-1]
    if re.search(r"\b(?:are|were|they|them|both|together|two|each)\b", clause, re.I):
        return ""
    rows = dict((n, ln) for n, ln in sheet_lines(sheet) if n)
    marks = [(mm.start(), n) for n in in_frame
             for mm in re.finditer(r"\b" + re.escape(n) + r"\b", clause)]
    for mm in re.finditer(r"\b(she|he|her|him)\b", clause, re.I):
        group = {"her": "she", "him": "he"}.get(mm.group(1).lower(), mm.group(1).lower())
        fits = [n for n in in_frame if sheet_pronoun(rows.get(n, "")) == group]
        if len(fits) == 1:
            marks.append((mm.start(), fits[0]))
    if not marks:
        return ""
    at, who = max(marks)
    # "Dan and Ana stand alone" -- two people joined as the subject are both alone.
    # ("...away from Ana and she stands alone" is a new clause, and hers.)
    if re.fullmatch(r"\s*(?:the\s+)?\w+\s+and\s+", clause[:at], re.I):
        return ""
    return who


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
    # A SUBJECT pronoun is the person; "his" and "her" are as often somebody else's.
    # "Kate: 25, woman, his hoodie" was read as a man off the hoodie's owner -- and
    # the body, the figure and the genitals every bare shot describes followed it.
    m = re.search(r"\b(she|he)\b", body, re.I)
    if m:
        return m.group(1).lower()
    for item in body.split(","):
        m = _PERSON_NOUN.match(item.strip())
        if m:
            return "she" if m.group(1).lower() in _SHE_NOUNS else "he"
    hits = [(m.start(), group_of[m.group(0).lower()])
            for m in re.finditer(r"\b(?:" + "|".join(group_of) + r")\b", body, re.I)]
    if hits:
        return min(hits)[1]
    return None


def pronoun_is_a_guess(line):
    """Did sheet_pronoun only have a "his" or a "her" to go on?

    "Kate: 25, blonde, wearing his hoodie" reads as a man, and nothing in the line
    can say otherwise. Worth telling the author: the pronoun decides the body, and
    one written into the entry ("Kate: she, 25, ...") settles it."""
    body = (line or "").split(":", 1)[-1]
    if not sheet_pronoun(line):
        return False
    if any(i.strip().strip(".;").lower() in _PRONOUN_SET for i in body.split(",")):
        return False
    if re.search(r"\b(?:she|he)\b", body, re.I):
        return False
    return not any(_PERSON_NOUN.match(i.strip()) for i in body.split(","))


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


# Bodily contact between two people, as a sex scene is actually written: most of its
# beats name none of _SEXUAL_STAGING's words -- "Dan enters her", "Kate rides him",
# "they kiss" -- and the partner rule below needs to know the scene is one.
_INTIMATE = re.compile(
    r"\b(?:kiss(?:es|ed|ing)?|straddl\w*|thrust\w*|grind(?:s|ing)?|caress\w*|fondl\w*|"
    r"lick(?:s|ed|ing)?|suck(?:s|ed|ing)?|mak(?:e|es|ing)\s+love|made\s+love|"
    r"rid(?:e|es|ing)\s+(?:him|her|them)|on\s+top\s+of\s+(?:him|her|them)|"
    r"between\s+(?:her|his|their)\s+(?:legs|thighs)|"
    r"(?:enter(?:s|ed|ing)?|inside|into)\s+(?:her|him|them)\b(?!\s+(?:[\w-]+\s+)?"
    r"(?:room|bedroom|house|flat|car)\b)|"
    r"spreads?\s+(?:her|his|their)\s+legs)\b", re.I)


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
    # WHOSE, in the same breath. "the genitals" named nobody's, so in a shot with a
    # man and a woman in it the model drew whichever its prior reached for first --
    # reported as the wrong genitalia. The sex was worked out above and never said.
    return (f"the hips and groin bare as well, a {who}'s genitals uncovered and in "
            f"plain view")


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


def _subject_clauses(sentence, forms):
    """A sentence cut where a new clause opens on a NAME. [sentence] when none does.

    "Kate wears a black thong and Dan wears boxers." is two statements about two
    people. A shot without Dan dropped the whole sentence, and Kate's thong with it.
    "Kate and Dan sit on the bed" is ONE statement -- a piece that is nothing but a
    name is half a compound subject, not a clause -- and is not cut."""
    s = str(sentence or "")
    if not forms or not s.strip():
        return [s]
    alt = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    cut = re.compile(r"(?:\s*[,;]\s*(?:and\s+|but\s+|while\s+)?|\s+(?:and|but|while)\s+)"
                     r"(?=(?:" + alt + r")(?![\w-])\s+(?!(?:and|or)\b)[a-z])")
    pieces = [p.strip() for p in cut.split(s) if p.strip()]
    if len(pieces) < 2 or any(re.fullmatch(r"(?:" + alt + r")", p.strip(" .!?"))
                              for p in pieces):
        return [s]
    return pieces


def static_wardrobe(static, names):
    """{name: [garment as written]} that the anchor and opening paragraph dress each
    person in.

    Garments written there are worn exactly as much as the sheet's -- but only the
    sheet's were ever counted as worn, so taking the jeans off somebody the anchor had
    put in a black thong told the shot the hips and groin were bare, the genitals
    uncovered, over a thong that was still on and still described. A sentence about
    one person is theirs; one that opens on "she" or "he" is the last-named person's."""
    names = [n for n in (names or []) if n]
    forms_of = {n: _name_forms(n) for n in names}
    every = set().union(*forms_of.values()) if forms_of else set()
    out, last = {}, ""
    for line in str(static or "").split("\n"):
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            for piece in _subject_clauses(sentence, every):
                who = [n for n in names
                       if any(re.search(r"(?<![\w'’-])" + re.escape(f) + r"(?![\w-])",
                                        piece) for f in forms_of[n])]
                if not who and last and _LEADING_PRONOUN.match(piece):
                    who = [last]
                if len(who) != 1:
                    last = ""
                    continue
                last = who[0]
                for g in engine.garments_in(piece):
                    if g not in out.setdefault(last, []):
                        out[last].append(g)
    return out


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
    every = set().union(*(_name_forms(n) for n, _ in sheet_lines(sheet) if n))

    out = []
    for line in static.split("\n"):
        kept, cut_last = [], False
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            if not sentence:
                continue
            # Only the clauses about the absent go. See _subject_clauses.
            if named.search(sentence) and not cut_last:
                pieces = _subject_clauses(sentence, every)
                mine = [p for p in pieces if not named.search(p)]
                if len(pieces) > 1 and mine:
                    said = "; ".join(p.rstrip(" .!?,;") for p in mine)
                    kept.append(said + (sentence.rstrip()[-1]
                                        if sentence.rstrip()[-1:] in ".!?" else "."))
                    cut_last = False
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


# WHAT THE OPENING PARAGRAPH DOES, AS OPPOSED TO WHAT IT IS. The paragraph is stamped
# into every shot, so "Mara sits on the sofa reading a paperback. A dog barks
# somewhere outside." sat her back down with the book and set the dog off again in
# every later shot, whatever that shot's beat had her doing. REPORTED as characters
# doing things the beat never wrote, and the beat losing its share of the prompt. The
# place, the time, the light, the weather and the furniture are what the scene IS and
# stay; a sentence that stages an action or a posture for somebody on the sheet, or an
# event, is the opening shot's and is withheld after it. A sentence that names a
# garment or a restraint always stays -- the removal scrub and static_wardrobe read
# them, and what somebody wears is what IS.
_SCENE_EVENT = re.compile(
    r"\b(?:barks?|barking|rings?|ringing|buzz(?:es|ing)?|knocks?|knocking|slams?|"
    r"slamming|bangs?|banging|crash(?:es|ing)?|shatters?|explodes?|honks?|beeps?|"
    r"chimes?|howls?|screech(?:es|ing)?|pulls?\s+up|bursts?\s+open|swings?\s+open|"
    r"creaks?\s+open|flies\s+open)\b", re.I)
# ...but weather, light and the fixtures of a room are what the scene IS, sounding or
# not: "The wind howls outside", "A fluorescent tube buzzes overhead" were withheld
# after shot 1 as if they were events. REPORTED.
_SCENE_FIXTURE = re.compile(
    r"^\s*(?:the|a|an|some|distant|far-off)?\s*(?:\w+\s+){0,2}?"
    r"(?:wind|winds|rain|thunder|lightning|storm|snow|hail|sleet|sea|waves|surf|river|"
    r"stream|light|lights|tube|lamp|lamps|bulb|bulbs|neon|sign|signs|fridge|"
    r"refrigerator|radiator|pipes?|clock|fan|heater|generator|air\s+conditioner|"
    r"traffic|crickets|cicadas|insects|frogs|birds|surf|fire|fireplace|candles?|"
    r"jukebox|radio|television|tv|bells?)\b", re.I)
_SCENE_PRONOUN_ACTS = re.compile(
    r"^\s*(?:she|he|they)\s+(?:\w+ly\s+)?(?!(?:is|was|are|were|has|had|have)\b)"
    r"(?-i:[a-z]+(?:s|es|ed))\b", re.I)


def _scene_key(sentence):
    """One spelling of a sentence for matching it across terminate_lines and spacing."""
    return " ".join(str(sentence or "").split()).rstrip(".!?;, ").lower()


def scene_staging(first_para, sheet):
    """{sentence key: (what stays of it, names in it)} for the opening paragraph's
    sentences that stage something -- see _SCENE_EVENT above. What stays is the
    sentence's setting ("On the sofa.") or "" when it names none."""
    cast = [n for n, _ in sheet_lines(sheet or "") if n]
    out, staged_last = {}, False
    for line in str(first_para or "").split("\n"):
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            s = sentence.strip()
            if not s:
                continue
            if engine.garments_in(s) or engine.hardware_spans(s) or restraint_words(s):
                staged_last = False
                continue
            who = [n for n in cast
                   if re.search(r"(?<![\w'’-])" + re.escape(n) + r"(?![\w'’-])", s)]
            acts = (any(acts_in(s, n, sheet) for n in who)
                    or bool(posture_in(s, who)) if who else False)
            if not who and _SCENE_PRONOUN_ACTS.match(s):
                acts = True
                who = []
            # A subject pronoun carries the staging on; "Her phone lies on the
            # nightstand" is a thing, not her. REPORTED.
            if not (acts or (not who and _SCENE_EVENT.search(s)
                             and not _SCENE_FIXTURE.match(s))
                    or (staged_last and re.match(r"\s*(?:she|he|they)\b", s, re.I))):
                staged_last = False
                continue
            staged_last = True
            setting = _setting_of(s)
            if setting:
                # The place, cut where the action resumes: "On the sofa reading a
                # paperback" is still her reading.
                setting = re.split(r",|\s+(?:and|while|as|then)\s+|\s+\w+ing\b",
                                   setting.rstrip("."), maxsplit=1)[0].strip()
                setting = (setting + ".") if len(setting.split()) >= 2 else ""
            out[_scene_key(s)] = (setting, who, engine.posture_in(s))
    return out


def withhold_staging(text, staged, keep=(), lying=()):
    """`text` without the opening paragraph's staging sentences -- see scene_staging --
    each one replaced by its setting where it has one. A sentence about somebody in
    `keep` stays: the shot that first shows them is where the paragraph put them, and a
    body in hardware is held as it was placed. So does a body the paragraph laid down,
    for as long as no beat has got them up (`lying`): lying there is what IS."""
    if not staged or not str(text or "").strip():
        return text
    keep = set(keep or ())
    lying = set(lying or ())
    out = []
    for line in str(text).split("\n"):
        kept, said = [], set()
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            if not sentence:
                continue
            hit = staged.get(_scene_key(sentence))
            if hit is None or (keep and set(hit[1]) & keep):
                kept.append(sentence)
                said.add(_scene_key(sentence))
                continue
            # ...the posture and the place only: "Mara lies on the bed reading a
            # paperback" is "Mara is lying on the bed." The reading was shot 1's.
            if hit[2] == "lying down" and hit[1] and set(hit[1]) <= lying:
                _who = hit[1]
                _subj = (_who[0] if len(_who) == 1
                         else ", ".join(_who[:-1]) + " and " + _who[-1])
                _place = str(hit[0] or "").rstrip(". ")
                _place = (_place[:1].lower() + _place[1:]) if _place else "down"
                kept.append(f"{_subj} {'is' if len(_who) == 1 else 'are'} lying {_place}.")
                said.add(_scene_key(sentence))
                if hit[0]:
                    said.add(_scene_key(hit[0]))
                continue
            setting = hit[0]
            if setting and _scene_key(setting) not in said:
                kept.append(setting)
                said.add(_scene_key(setting))
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


def plan_lengths(beats, ceiling_frames, from_beat, pace=1.0, applying=()):
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
    in them brisker, above 1.0 they get longer and slower.

    `applying` is the 1-based shots whose beat puts a restraint, gag or seal on. Each
    gets one more action's time on top of its floored estimate and is not leaned
    short, `pace` below 1.0 included: its last frame is the one the next shot is
    pinned to, and that frame has to show the piece on and the hands clear of it, not
    the putting-on half done. Reported as restraints and tape gone in the next beat."""
    if not from_beat:
        return [ceiling_frames] * len(beats), ""
    pace = max(0.05, float(pace if pace else 1.0))
    _applying = {int(n) for n in (applying or ())}
    lens, capped, held = [], [], []
    for _n, b in enumerate(beats, 1):
        need = beat_seconds(b) * pace
        if _n in _applying:
            need = (max(beat_seconds(b), MIN_AUTO_FRAMES / H3_FPS)
                    + SECONDS_PER_ACTION) * max(1.0, pace)
        want = align_frame_count_nearest(int(round(need * H3_FPS))) if need else MIN_AUTO_FRAMES
        if want > ceiling_frames:
            capped.append((len(lens) + 1, want))
        lens.append(min(max(MIN_AUTO_FRAMES, want), ceiling_frames))
        if _n in _applying and want <= ceiling_frames:
            held.append(_n)         # a capped one is reported as capped, below
    note = ""
    if held:
        note += (f"shot(s) {', '.join(str(n) for n in held)} put a restraint, gag or seal "
                 f"on, so each is given one more action's time: the piece goes on in the "
                 f"first half and is held for the rest, and the last frame -- the one the "
                 f"next shot opens on -- shows it in plain view with the hands clear of "
                 f"it. ")
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
                + "). They differ, and they still share one noise field: the seed's "
                  "noise is drawn frame by frame, so a frame's noise does not depend on "
                  "how long its shot is. The frames_per_shot output is ONE number and cannot "
                  "describe shots of different lengths: it reports the first one, so "
                  "do not split or index the image batch with it here -- the list "
                  "above is the split")
    return lens, note


# ACTIONS WITH AN END POINT. Spreading a beat across the shot only means something for
# an action that finishes -- a walk to the door, picking something up, a door opening,
# a garment coming off, sitting down. Said over "Mara sits on the bed", "Dan reads" or a
# line of dialogue it asked a held posture or an ongoing activity to "finish on the last
# frame", and the model invented an action to finish. REPORTED as characters doing
# things the beat never wrote.
_COMPLETIVE = re.compile(
    r"\b(?:walk(?:s|ed)?|go(?:es)?|went|runs?|ran|cross(?:es|ed)?|head(?:s|ed)?|"
    r"driv(?:e|es)|drove|climb(?:s|ed)?|mov(?:e|es|ed)|step(?:s|ped)?|crawl(?:s|ed)?|"
    r"hurr(?:y|ies|ied)|rush(?:es|ed)?|return(?:s|ed)?|comes?|came|wanders?|wandered|"
    r"stroll(?:s|ed)?|march(?:es|ed)?|backs?|backed)\s+(?:back\s+|over\s+|across\s+|up\s+|"
    r"down\s+|out\s+|away\s+)?(?:to|into|toward|towards|across|through|onto|inside|"
    r"outside|downstairs|upstairs|home|out\s+of|up\s+to|over\s+to)\b"
    r"|\b(?:pick(?:s|ed)?|puts?|sets?|lays?|laid|takes?|took|pulls?|pulled|peels?|"
    r"peeled|slips?|slipped)\s+(?:[\w'’-]+\s+){0,3}?(?:up|down|off|away|back|on)\b"
    # Not "opens her eyes", "raises her left hand" or "looks left": a body part and
    # a direction are not an errand finished. REPORTED.
    r"|\b(?:opens?|opened|closes?|closed|shuts?|unlocks?|unlocked|locks?|locked)\s+"
    r"(?:the|a|an|her|his|their|its)\b(?!\s+(?:eyes|mouth|lips|hands?|fists?|arms?|legs?))"
    r"|\b(?:sits?|sat|lies|lay|kneels?|knelt)\s+down\b|\b(?:stands?|stood|gets?|got)\s+up\b"
    r"|\b(?:arrives?|arrived|leaves|exits?|exited|enters?|entered)\b"
    r"|\b(?:has|had|have)\s+left\b|\bleft(?=\s+(?:the|a|an|through|by|for|without|home|"
    r"work)\b)"
    # ...and the crossings and carries it missed: "crosses the room", "carries her to
    # the bed". REPORTED.
    r"|\bcross(?:es|ed)?\s+(?:the|a)\s+(?:\w+\s+)?(?:room|street|road|floor|hall|"
    r"hallway|yard|lobby|kitchen|bar|courtyard|bridge|field|lawn|square|corridor|car\s+"
    r"park|parking\s+lot)\b"
    r"|\b(?:carr(?:ies|ied)|drags?|dragged|leads|led|walks|walked|takes|took|pulls?|"
    r"pulled|pushes|pushed)\s+(?:her|him|them|(?-i:[A-Z][\w'’-]+))\s+(?:back\s+)?"
    r"(?:to|into|toward|towards|across|through|onto|out\s+of|up\s+to|over\s+to)\b"
    r"|\b(?:undress(?:es|ed)?|strips?|stripped|unbuttons?|unbuttoned|unzips?|unzipped)\b",
    re.I)


def pace_clause(need, have, beat=None):
    """Spread a short action across a long shot. "" when the shot is not long.

    With `beat`, only an action that finishes, and never a spoken one -- see
    _COMPLETIVE.

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
    if beat is not None and (has_speech(beat)
                             or not _COMPLETIVE.search(engine.acted_text(str(beat)))):
        return ""
    return " The action runs at an even pace across the whole shot, finishing on the last frame."


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


_VOICE_VERB = re.compile(
    r"\b(?:gasp(?:s|ing|ed)?|pant(?:s|ing|ed)?|cr(?:y|ies|ying|ied)|sob(?:s|bing|bed)?|"
    r"scream(?:s|ing|ed)?|shout(?:s|ing|ed)?|yell(?:s|ing|ed)?|moan(?:s|ing|ed)?|"
    r"whimper(?:s|ing|ed)?|laugh(?:s|ing|ed)?)\b", re.I)


def voice_in(beat):
    """Does this beat give somebody a VOICE -- a sound a mouth makes?

    This, not exertion_in, is what opens a shot's audio branch when it has no line.
    Exertion used to: every verb of effort or reaction -- shakes her head, kicks the
    door shut, trembles, flinches, wakes up, grips the sheets, arches her back -- left
    the branch open and every mouth free, and told the model the shot sounds like
    gasps and moans of effort. An open branch with a face free and nothing to say is
    a voice inventing words. REPORTED as characters babbling on beats that were only
    actions: measured, sixteen of seventeen plain action beats opened the branch.

    A body working is silent now unless the beat says a mouth makes a sound -- she
    moans, he grunts, she pants, he laughs -- and then that sound is heard. named_vocals_in
    is the named half of this; the verbs below are the voices that are not named
    vocals of their own."""
    b = str(beat or "")
    return bool(named_vocals_in(b) or _VOICE_VERB.search(b))


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


# Pleasure said as a noun names no sound a branch can make. Heard, it is moaning.
_PLEASURE_SOUND = (r"(?:sounds?|noises?|cries|moans?|gasps?|sighs?|groans?)\s+of\s+"
                   r"(?:pleasure|ecstasy|passion|delight|arousal)|"
                   r"pleasure\s+(?:sounds?|noises?)")

_VOCAL_FROM = (
    (r"\b(?:" + _PLEASURE_SOUND + r")\b",           "moaning"),
    (r"\bwhimper(?:s|ing|ed)?\b",                   "whimpering"),
    (r"\bsob(?:s|bing|bed)?\b",                     "sobbing"),
    (r"\bmoan(?:s|ing|ed)?\b",                      "moaning"),
    (r"\bgroan(?:s|ing|ed)?\b",                     "groaning"),
    (r"\bscream(?:s|ing|ed)?\b",                    "screaming"),
    (r"\bwhin(?:e|es|ing|ed)\b",                    "whining"),
    # The rest were already read as somebody's voice (_VOCAL_SOURCE) and never as a
    # sound, so a beat that said one had its mouth opened and was then told "the only
    # sound is" the room -- an open branch described as silent fills itself with a
    # voice. Reported as sounds of pleasure coming out as gibberish.
    (r"\bcr(?:y|ies|ied|ying)\s+out\b",             "crying out"),
    (r"\bgrunt(?:s|ing|ed)?\b",                     "grunting"),
    (r"\bsigh(?:s|ing|ed)?\b",                      "sighing"),
    (r"\bgasp(?:s|ing|ed)?\b",                      "gasping"),
)

# BREATH, NOT A VOICE. Effort was heard as "unsteady breathing, with wordless gasps and
# moans of effort": two vocals the beat never named, and a vocal is a mouth the model
# opens. REPORTED as characters doing things the beat never wrote. A body working hard
# breathes; anything louder is the author's to write.
EFFORT_BREATH = "breathing"

# OBJECT SOUNDS FROM A VERB ON THE OBJECT, NEVER FROM A MENTION. "A van with closed rear
# doors" was heard as a door swinging and an engine running; "sits in cuffs" as cuffs
# knocking; "in a grey sweater" as fabric rustling. Each is a sound of something
# happening, so each asked the shot for an action nobody wrote. REPORTED as characters
# doing things the beat never wrote. A sound now needs a verb acting on its object (or
# the object itself doing the sounding: "the cuffs ratchet closed").
_ON = r"\s+(?:at\s+|on\s+|against\s+|in\s+|with\s+)?(?:[\w'’-]+\s+){0,3}?"
_GARMENT_WORDS = (r"(?:fabric|cloth|coat|jacket|shirt|t-shirt|dress|skirt|shorts|trousers|"
                  r"pants|jeans|leggings|tights|socks|boots|shoes|gloves|top|vest|jumper|"
                  r"sweater|hoodie|blouse|cardigan|robe|gown|towel|sheet|blanket)s?")
_SOUND_FROM = (
    *_VOCAL_FROM,
    (r"\b(?:walk(?:s|ed|ing)?|step(?:s|ped|ping)?|pace[sd]?|enters?|runs?|"
     r"approach(?:es|ed)?|creep(?:s|ing)?|crept|sneak(?:s|ing)?|shuffl(?:e|es|ing)|"
     r"stumbl(?:e|es|ing)|stagger(?:s|ing)?)\b(?!\s+(?:her|his|their)\s+(?:fingers?|"
     r"hands?|tongue|eyes))",                       "footsteps"),
    (r"\b(?:drag(?:s|ged|ging)?|rattl(?:e|es|ed|ing)|pull(?:s|ed|ing)?|yank(?:s|ed|ing)?|"
     r"tug(?:s|ged|ging)?|jerk(?:s|ed|ing)?|lift(?:s|ed|ing)?|drop(?:s|ped|ping)?|"
     r"wrap(?:s|ped|ping)?|loop(?:s|ed|ing)?|wind(?:s|ing)?|wound|thread(?:s|ed|ing)?|"
     r"run(?:s|ning)?|pass(?:es|ed|ing)?|unlock(?:s|ed|ing)?|unwind(?:s|ing)?|"
     r"shak(?:e|es|ing)|haul(?:s|ed|ing)?|thrash(?:es|ed|ing)?|struggl(?:e|es|ed|ing)|"
     r"strain(?:s|ed|ing)?|fasten(?:s|ed|ing)?|padlock(?:s|ed|ing)?|clip(?:s|ped|ping)?)"
     + _ON + r"chains?\b"
     r"|\bchains?\s+(?:rattl|clank|clink|drag|jangl|swing|go(?:es)?\s+taut)\w*",
                                                    "chain links dragging"),
    (r"\b(?:snap(?:s|ped|ping)?|lock(?:s|ed|ing)?|click(?:s|ed|ing)?|clos(?:e|es|ed|ing)|"
     r"ratchet(?:s|ed|ing)?|tighten(?:s|ed|ing)?|clamp(?:s|ed|ing)?|squeez(?:e|es|ed|ing))"
     + _ON + r"(?:hand)?cuffs?\b"
     r"|\b(?:hand)?cuffs?\s+(?:[\w'’-]+\s+){0,2}?(?:snap|lock|click|close|ratchet|tighten)\w*",
                                                    "cuffs ratcheting closed"),
    (r"\b(?:hand)?cuff(?:s|ed)\s+(?:her|him|them|(?-i:[A-Z][\w'’-]+))\b"
     r"|\b(?:rattl(?:e|es|ed|ing)|shak(?:e|es|ing)|pull(?:s|ed|ing)?|tug(?:s|ged|ging)?|"
     r"yank(?:s|ed|ing)?|jerk(?:s|ed|ing)?|twist(?:s|ed|ing)?)" + _ON
     + r"(?:hand)?cuffs?\b|\b(?:shackl|manacl)(?:es|ed)\s+(?:her|him|them|(?-i:[A-Z]\w+))\b",
                                                    "cuffs knocking"),
    (r"\bbolt(?:s|ed|ing)\s+(?:the\s+)?(?:door|gate|window|hatch)\b"
     r"|\blatch(?:es|ed|ing)\s+(?:the\s+)?\w+"
     r"|\b(?:slid(?:e|es|ing)|draw(?:s|ing)?|drew|shoot(?:s|ing)?|shot|throw(?:s|ing)?|"
     r"threw|lift(?:s|ed|ing)?|flick(?:s|ed|ing)?|undo(?:es)?|undid)\s+(?:the\s+)?"
     r"(?:\w+\s+)?(?:bolt|latch)\b",                "a metal bolt sliding"),
    (r"\b(?:locks|locking|padlocks|padlocked|padlocking|locked)\s+"
     r"(?!(?:eyes|gaze|horns|onto|up\s+inside)\b)(?:it|them|the|a|her|his|their|up)\b"
     r"|\b(?:snap(?:s|ped)?|click(?:s|ed)?)\s+(?:the\s+)?(?:\w+\s+)?(?:pad)?lock\b",
                                                    "a lock snapping shut"),
    (r"\b(?:drag(?:s|ged|ging)?|haul(?:s|ed|ing)?|shov(?:e|es|ing)|slid(?:e|es|ing))\b",
                                                    "something dragging on the floor"),
    (r"\b(?:un)?buckl(?:es|ed|ing)\s+(?:up\b|(?:the|her|his|their|it|a|him|them)\b)"
     r"|\bstrap(?:s|ped|ping)\s+(?:her|him|them|the|it|(?-i:[A-Z][\w'’-]+))\b"
     r"|\b(?:fasten(?:s|ed|ing)?|tighten(?:s|ed|ing)?|clasp(?:s|ed|ing)?|"
     r"unfasten(?:s|ed|ing)?)\s+(?:the|her|his|their|a)\s+(?:\w+\s+)?"
     r"(?:buckle|strap|harness|belt|collar)s?\b",   "a buckle and leather creaking"),
    (r"\b(?:pour(?:s|ed|ing)?|splash(?:es|ed|ing)?)\b"
     r"|\b(?:runs?|ran|turns?\s+on)\s+the\s+(?:tap|taps|water|bath|shower)\b"
     r"|\bfills?\s+(?:the|a)\s+(?:\w+\s+)?(?:glass|bath|sink|kettle|bucket|tub)\b",
                                                    "water"),
    (r"\b(?:start(?:s|ed)?|rev(?:s|ved)?|driv(?:e|es|ing)|drove|park(?:s|ed)?)\s+"
     r"(?:[\w'’-]+\s+){0,2}?(?:van|car|engine|truck|motor)\b"
     r"|\b(?:van|car|engine|truck|motor)\s+(?:[\w'’-]+\s+){0,1}?(?:starts?|revs?|roars?|"
     r"idles?|pulls?\s+(?:up|away|in|out|off)|drives?\s+(?:off|away|up|in)|"
     r"speeds?\s+(?:off|away))\b|\bdrives?\s+(?:off|away)\b",
                                                    "an engine outside"),
    (r"\b(?:cuts?|cutting|snips?|snipping|slices?|slicing)\s+(?:[\w'’-]+\s+){0,3}?"
     r"(?:through\b|" + _GARMENT_WORDS + r"|clothes|sleeves?|bra|panties|underwear)"
     r"|\b(?:scissors|shears)\s+(?:cut|snip|slice)\w*",
                                                    "blades through fabric"),
    (r"\b(?:open(?:s|ed|ing)?|clos(?:e|es|ed|ing)|shut(?:s|ting)?|slam(?:s|med|ming)?|"
     r"push(?:es|ed|ing)?|pull(?:s|ed|ing)?|kick(?:s|ed|ing)?|swing(?:s|ing)?|swung|"
     r"bang(?:s|ed|ing)?|knock(?:s|ed|ing)?\s+on|yank(?:s|ed|ing)?)\s+"
     r"(?:open\s+|shut\s+)?(?:the|a|an|her|his|their|its|both|one|that|this)\s+"
     r"(?:[\w'’-]+\s+){0,2}?doors?\b"
     r"|\bdoors?\s+(?:[\w'’-]+\s+){0,1}?(?:opens?|closes?|shuts?|slams?|swings?|creaks?|"
     r"bangs?)\b"
     r"|\b(?:comes?|came|walks?|steps?|bursts?|goes|went|leaves|left)\s+(?:in\s+|out\s+)?"
     r"through\s+the\s+(?:\w+\s+)?door\b",          "a door on its hinges"),
    (r"\b(?:drops?|dropped|throw(?:s|n)?|threw|toss(?:es|ed)?)\b",
                                                    "something landing"),
    (r"\b(?:smack(?:s|ed)?|slap(?:s|ped)?|hits?|strikes?|struck)\b", "a sharp impact"),
    (r"\b(?:thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|writh(?:e|es|ing|ed)|"
     r"strain(?:s|ing|ed)?|pull(?:s|ing|ed)?|tug(?:s|ging|ged)?|yank(?:s|ing|ed)?|"
     r"twist(?:s|ing|ed)?|jerk(?:s|ing|ed)?|fight(?:s|ing)?)\s+"
     r"(?:against|at|in|on)\s+(?:the|her|his|their)\s+(?:\w+\s+)?"
     r"(?:cuffs?|handcuffs?|shackles?|manacles?|chains?|ropes?|cords?|straps?|"
     r"restraints?|bindings?|ties|tape|harness|collar)\b"
     # ...or the hardware doing the holding while she fights it, in one sentence.
     r"|\b(?:cuffs?|handcuffs?|shackles?|manacles?|chains?|ropes?|cords?|straps?|"
     r"restraints?|bindings?)\s+(?:[\w'’-]+\s+){0,3}?(?:holds?|bites?|digs?|is\s+taut|"
     r"are\s+taut|goes\s+taut|go\s+taut|pulls?\s+tight)\b[^.!?]*?\b(?:thrash(?:es|ing)?|"
     r"struggl(?:es|ing)|writh(?:es|ing)|strain(?:s|ing)?|fights?|twists?)\b",
                                                    "restraints pulling taut"),
    (r"\b(?:thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|writh(?:e|es|ing|ed)|"
     r"strain(?:s|ing|ed)?|trembl(?:e|es|ing|ed)|shiver(?:s|ed|ing)?)\b"
     r"|\b(?:" + _EXERTION_NARROW_SRC + r")",
                                                    EFFORT_BREATH),
    (r"\b(?:un)?zip(?:s|ped|ping)\b"
     r"|\b(?:pull(?:s|ed|ing)?|tug(?:s|ged|ging)?|yank(?:s|ed|ing)?|draw(?:s|ing)?|"
     r"run(?:s|ning)?|slid(?:e|es|ing))\s+(?:[\w'’-]+\s+){0,2}?zipper\b", "a zip running"),
    (r"\b(?:tear(?:s|ing)?|tore|rip(?:s|ped|ping)?|peel(?:s|ed|ing)?|pull(?:s|ed|ing)?|"
     r"yank(?:s|ed|ing)?|unroll(?:s|ed|ing)?|stretch(?:es|ed|ing)?|wrap(?:s|ped|ping)?|"
     r"wind(?:s|ing)?|wound|press(?:es|ed|ing)?|smooth(?:s|ed|ing)?)\s+"
     r"(?:[\w'’-]+\s+){0,3}?(?:duct\s+|gaffer\s+|packing\s+|masking\s+)?tape\b"
     r"|\btap(?:es|ed|ing)\s+(?:up\s+)?(?:her|his|their|him|them|(?-i:[A-Z][\w'’-]+))\b",
                                                    "tape pulling off"),
    (r"\A(?=[\s\S]*\b(?:bed|mattress|springs?|bunk|couch|sofa|headboard|"
     r"frame|table|desk|floorboards?)\b)"
     r"(?=[\s\S]*\b(?:rock(?:s|ed|ing)?|thrust(?:s|ing)?|grind(?:s|ing)?|"
     r"buck(?:s|ed|ing)?|writh(?:e|es|ing|ed)|arch(?:es|ed|ing)?|"
     r"thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|move(?:s|d)?\s+together|"
     r"shift(?:s|ed|ing)?\s+under)\b)",              "a bed frame working"),
    (r"\b(?:rip(?:s|ped|ping)?|tear(?:s|ing)?|tore|pull(?:s|ed|ing)?|open(?:s|ed|ing)?|"
     r"undo(?:es)?|undid)\s+(?:[\w'’-]+\s+){0,2}?velcro\b"
     r"|\bvelcro\s+(?:straps?\s+)?(?:rips?|tears?)\b", "velcro tearing open"),
    (r"\b(?:pull(?:s|ed|ing)?|tug(?:s|ged|ging)?|yank(?:s|ed|ing)?|tighten(?:s|ed|ing)?|"
     r"cinch(?:es|ed|ing)?|knot(?:s|ted|ting)?|ties|tied|tying|wrap(?:s|ped|ping)?|"
     r"loop(?:s|ed|ing)?|wind(?:s|ing)?|wound|haul(?:s|ed|ing)?|strain(?:s|ed|ing)?|"
     r"thrash(?:es|ed|ing)?|struggl(?:e|es|ed|ing)|test(?:s|ed|ing)?|jerk(?:s|ed|ing)?)"
     + _ON + r"(?:rope|cord|twine|zip\s?tie)s?\b"
     r"|\b(?:rope|cord)s?\s+(?:creak|tighten|go(?:es)?\s+tight|bite|dig)\w*",
                                                    "rope creaking as it goes tight"),
    (r"\b(?:pull(?:s|ed|ing)?|tug(?:s|ged|ging)?|take(?:s|n)?|took|slip(?:s|ped|ping)?|"
     r"peel(?:s|ed|ing)?|strip(?:s|ped|ping)?|unbutton(?:s|ed|ing)?|button(?:s|ed|ing)?|"
     r"tear(?:s|ing)?|tore|rip(?:s|ped|ping)?|yank(?:s|ed|ing)?|drop(?:s|ped|ping)?|"
     r"throw(?:s|n)?|threw|fold(?:s|ed|ing)?|shrug(?:s|ged|ging)?|puts?|putting|"
     r"remov(?:e|es|ed|ing)|straighten(?:s|ed|ing)?|smooth(?:s|ed|ing)?|"
     r"adjust(?:s|ed|ing)?|lift(?:s|ed|ing)?|hik(?:e|es|ed|ing)|wriggl(?:e|es|ed|ing)|"
     r"kick(?:s|ed|ing)?)\s+(?:[\w'’-]+\s+){0,3}?" + _GARMENT_WORDS + r"\b"
     r"|\b(?:undress(?:es|ed|ing)?|strips?\s+(?:off|naked|down))\b",
                                                    "fabric rustling"),
    (r"\b(?:jingl|rattl|fumbl|drop|pull|take|took|turn|hand|toss|throw|threw|pocket|grab|"
     r"fish|dangl|pick)\w*\s+(?:[\w'’-]+\s+){0,3}?keys?\b"
     r"|\bkeys?\s+(?:jingl|rattl|turn|clink)\w*",  "keys on a ring"),
    (r"\b(?:wakes?\s+up|woke|gasp(?:s|ing)?|pant(?:s|ing)?|breath(?:es|ing)?)\b",
                                                    "breathing"),
)
MAX_SOUNDS = 3      # a shot's audio needs a cue, not an inventory
_VOCAL_RETIRES = (EFFORT_BREATH,)
_VOCAL_BETWEEN = "breathing"
# The vocals above, as a set: see the tail of sounds_for for why they are special-cased.
_NAMED_VOCALS = frozenset(phrase for _, phrase in _VOCAL_FROM)
_SOUND_SUPERSEDES = {
    "cuffs ratcheting closed": ("cuffs knocking",),
    **{v: _VOCAL_RETIRES for v in _NAMED_VOCALS},
}


def wordless(phrases):
    """The heard list with its vocals said to be WORDLESS, folded into one phrase.

    A vocal is a voice, and a voice is what a joint model's audio branch reaches for
    speech with: at the last few steps it resolves whatever is easiest, and a human
    voice with nothing saying what shape it takes comes out as syllables. Reported
    as moaning and sounds of pleasure turned into gibberish. This is the positive
    way to say it -- at cfg 1 no negative prompt is evaluated, so "no words" is the
    word "words" -- and it is the word audio captions use for exactly this.

    Folded where the first vocal stood ("wordless moaning and grunting"), so the list
    keeps its order and gains one word, not one per vocal."""
    vocals = [p for p in phrases if p in _NAMED_VOCALS]
    if not vocals:
        return list(phrases)
    one = "wordless " + (vocals[0] if len(vocals) == 1
                         else ", ".join(vocals[:-1]) + " and " + vocals[-1])
    out, done = [], False
    for p in phrases:
        if p in _NAMED_VOCALS:
            if not done:
                out.append(one)
                done = True
            continue
        out.append(p)
    return out

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
    return list(dict.fromkeys(phrase for pat, phrase in _VOCAL_FROM
                              if re.search(pat, b, re.I)))


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
_SAYS = (r"says?|said|asks?|asked|whispers?|whispered|shouts?|shouted|yells?|yelled|calls?|"
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
    # A PRONOUN SAYING IT, before anything positional. 'Dan holds her. "Stay," she
    # whispers.' fell through to "whoever the beat names first" -- Dan -- so the shot
    # went out as "Only Dan speaks" with HER mouth held shut: the listener lip-syncing
    # the speaker's line. The pronoun is the subject of the verb; it resolves where
    # exactly one person on the sheet declares it.
    if not out and has_speech(b):
        said_by = [g for g in ("she", "he", "they")
                   if re.search(r"\b" + g + r"\b" + _UP_TO_TWO_WORDS
                                + r"\s+(?:" + _SAYS + r")\b", b, re.I)
                   or re.search(r"(?:[\"'”’]|</d>)\s*[,.;]?\s*(?:" + _SAYS + r")\s+"
                                + g + r"\b", b, re.I)]
        for n, ln in sheet_lines(sheet):
            group = sheet_pronoun(ln) if n else None
            if (group in said_by
                    and sum(1 for _n, _l in sheet_lines(sheet)
                            if _n and sheet_pronoun(_l) == group) == 1):
                out.append(n)
        # A pronoun said it and fits more than one person: naming whoever comes first
        # is a guess that can hand her line to him. Unattributed, the shot still gets
        # "only the person speaking" -- which holds the listener without choosing.
        if not out and said_by:
            return []
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


_REMOTE = re.compile(r"\b(?:phone|phones|mobile|cell|radio|walkie|intercom|speaker|"
                     r"voicemail|call|calls|calling|texts?|message|letter|note|screen|"
                     r"video\s+call|through\s+the\s+(?:door|wall|window)|from\s+"
                     r"(?:outside|another\s+room|upstairs|downstairs))\b", re.I)


# A line that OPENS on a command is said to somebody: "Go to the kitchen.", "Get out.",
# "Please don't cuff me."
_COMMAND = re.compile(
    r"^\s*(?:(?:please|now|just|okay|ok|come\s+on)[,!]?\s+)?(?:(?:don't|do\s+not|never)\s+)?"
    r"(?:go|get|come|sit|stand|kneel|lie|lay|turn|take|put|give|open|close|look|stop|"
    r"wait|stay|hold|move|strip|undress|spread|bend|crawl|walk|run|leave|keep|show|"
    r"face|pull|push|drop|raise|lift|lower|shut|be|eat|drink|relax|breathe|listen|"
    r"watch|follow|let|help|untie|release|touch|kiss|hurt|cuff|tie|gag|beg|say|tell|"
    r"answer|hurry|calm|quiet|shush|hush|sleep|wake|climb|lean|step|roll|arch|smile|"
    r"swallow|suck|open|obey|behave|remove|unbutton|unzip)\b", re.I)


def addressed_in(beat, sheet):
    """The one person a beat's dialogue is SPOKEN TO, or "".

    Only a line that says "you" to somebody -- never a question, which is how absence
    is written -- and only where exactly one person on the sheet is not speaking.
    Nobody down a phone, a radio or through a door."""
    b = str(beat or "")
    said = [m.group(0) for m in _QUOTED.finditer(b)]
    if not said or _REMOTE.search(b):
        return ""
    lines = [x for x in re.split(r"(?<=[.!?])\s+", " ".join(
        re.sub(r"</?d>|[\"“”]", " ", x) for x in said)) if x.strip()]
    if not any((re.search(r"\byou(?:r|rs|rself)?\b", x, re.I) or _COMMAND.match(x))
               and not x.rstrip(" .\"”").endswith("?") for x in lines):
        return ""
    talking = set(speakers_in(b, sheet) or [])
    if not talking:
        return ""
    others = [n for n, _l in sheet_lines(sheet) if n and n not in talking]
    return others[0] if len(others) == 1 else ""


_LOWER_OUTER = re.compile(r"\b(?:trousers|pants|jeans|slacks|chinos|shorts|skirt|kilt|"
                          r"joggers|sweatpants|leggings|overalls|dungarees|cargos?)\b", re.I)


def worn_belt(line):
    """The belt this sheet line WEARS with its clothes -- "a leather belt" -- or "".
    Hardware is not a belt worn: steel, locked, chastity, garter and suspender belts
    are left to what they are."""
    for m in re.finditer(r"(?:[\w-]+\s+){0,2}belts?\b", str(line or ""), re.I):
        if re.search(r"chastity|steel|metal|iron|chrome|brass|lock|restraint|body|garter|"
                     r"suspender|utility|tool|seat|conveyor|black\s+belt|karate", m.group(0),
                     re.I):
            continue
        return m.group(0).strip()
    return ""


# WHAT AN ORDER OR AN INTENTION DEFERS, by kind. Only these hold anybody: an action
# the shot could stage early -- moving, a posture, clothes coming off or going on, a
# restraint. "Tells her to explain" or "is going to say sorry" defers nothing a body
# could do too soon.
_DEFER_WEAR = re.compile(
    r"^(?:take|pull|get|slip|peel)\s+(?:[\w'’-]+\s+){0,3}?off\b"
    r"|^(?:strip|undress|unbutton|unzip|unbuckle|unfasten|remove|disrobe|change)\b"
    r"|^get\s+(?:un)?dressed\b|^dress\b|^put\s+(?:[\w'’-]+\s+){0,3}?on\b|^wear\b"
    r"|^pull\s+(?:[\w'’-]+\s+){0,3}?down\b", re.I)
_DEFER_MOVE = re.compile(
    r"^(?:go|come|walk|run|leave|follow|move|step|crawl|climb|get|stand|sit|kneel|"
    r"lie|lay|bend|lean|spread|arch|face|turn|roll|approach|back|hurry|crouch|squat|"
    r"raise|lift|lower|drop|bow|hand|give|bring|fetch|open|close|"
    r"tie|cuff|handcuff|gag|bind|blindfold|tape|chain|shackle|restrain|strap|lock)\b",
    re.I)
_DEFER_BIND = re.compile(
    r"(?:tie|cuff|handcuff|gag|bind|blindfold|tape|chain|shackle|restrain|strap|lock)\b",
    re.I)
_ORDER_TO = re.compile(
    r"\b(?:tells?|told|telling|asks?|asked|asking|orders?|ordered|ordering|commands?|"
    r"commanded|instructs?|instructed|begs?|begged|begging|urges?|urged|warns?|warned|"
    r"invites?|invited)\s+(?P<who>[\w'’-]+)\s+to\s+(?P<act>[^,;.!?]*)", re.I)
_INTEND_TO = re.compile(
    r"\b(?:(?:is|are|was|were|am|'s|'re)\s+(?:going|about)|plans?|planned|planning|"
    r"threatens?|threatened|threatening)\s+to\s+(?P<act>[^,;.!?]*)", re.I)


def _deferred_kind(act):
    """'wear', 'move' or '' for the action an order or an intention puts off."""
    a = str(act or "").strip()
    if _DEFER_WEAR.match(a):
        return "wear"
    if _DEFER_MOVE.match(a):
        return "move"
    return ""


def deferred_holds(beat, described, acted, sheet="", speakers=()):
    """[(name, kind)] for the people a beat only ORDERS or INTENDS a staged action for,
    and who do not act in it. kind is 'wear' for clothes, 'move' for the rest.

    "Dan tells Ana to take off her shorts" asks; "Dan is going to tie her up" means to.
    Neither does anything in this shot, and read as a description of it both were
    performed in it -- a beat early. REPORTED as actions happening before they are
    supposed to take place. Only the person the order is about -- the one told, or the
    one who means to -- and only for a staged action: a bare "could", "wants to" or "is
    told to answer" holds nobody. A line said to one person present ("Take off your
    shirt.") is an order to them."""
    rows = dict((n, ln) for n, ln in sheet_lines(sheet) if n)
    people = [n for n in (described or []) if n]
    staged = engine.staged_text(beat)
    found = []

    def _named_before(text):
        hits = [(mm.start(), n) for n in people
                for mm in re.finditer(r"\b" + re.escape(n) + r"\b(?!['’]s)", text)]
        return max(hits)[1] if hits else ""

    def _by_pronoun(word, besides=""):
        group = {"her": "she", "she": "she", "him": "he", "he": "he",
                 "them": "they", "they": "they"}.get(word.lower(), "")
        if not group:
            return ""
        pool = [n for n in people if n != besides]
        fits = [n for n in pool if sheet_pronoun(rows.get(n, "")) == group]
        if len(fits) == 1:
            return fits[0]
        if not any(sheet_pronoun(rows.get(n, "")) for n in pool) and len(pool) == 1:
            return pool[0]
        return ""

    for m in _ORDER_TO.finditer(staged):
        kind = _deferred_kind(m.group("act"))
        if not kind or re.match(r"not\b", m.group("act").strip(), re.I):
            continue
        sentence = re.split(r"[.;!?]", staged[:m.start()])[-1]
        giver = _named_before(sentence)
        who = m.group("who")
        name = next((n for n in people if n.lower() == who.lower()), "") \
            or _by_pronoun(who, giver)
        if name and name != giver:
            found.append((name, kind))
    for m in _INTEND_TO.finditer(staged):
        kind = _deferred_kind(m.group("act"))
        if not kind:
            continue
        sentence = re.split(r"[.;!?]", staged[:m.start()])[-1]
        subj = re.search(r"\b(she|he|they)\s+$", sentence, re.I)
        name = (_by_pronoun(subj.group(1)) if subj else "") or _named_before(sentence)
        # THE PERSON IT IS DONE TO, for clothes and restraints: "Dan is going to take
        # off her sweater", "threatens to strip her", "plans to tie her up" are about
        # her sweater and her wrists. Holding Dan kept the wrong person in place and
        # left hers free to go a beat early. REPORTED.
        if kind == "wear" or _DEFER_BIND.match(m.group("act").strip()):
            _obj = re.search(r"\b(?:(her|him|them|his|their)|(?-i:([A-Z][\w'’-]+)))\b",
                             m.group("act"))
            _target = ""
            if _obj and _obj.group(1):
                _target = _by_pronoun({"his": "him", "their": "them"}.get(
                    _obj.group(1).lower(), _obj.group(1)), name)
            elif _obj:
                _target = next((n for n in people if n == _obj.group(2)), "")
            if _target and _target != name:
                name = _target
        if name:
            found.append((name, kind))
    # A line said to the one person present who is not saying it.
    said = [x.group(0) for x in _QUOTED.finditer(str(beat or ""))]
    if said and not _REMOTE.search(str(beat or "")):
        talking = set(speakers or [])
        heard = [n for n in people if n not in talking]
        if talking and len(heard) == 1:
            for line in re.split(r"(?<=[.!?])\s+", " ".join(
                    re.sub(r"</?d>|[\"“”]", " ", x) for x in said)):
                line = line.strip()
                if not line or line.rstrip(" .\"”").endswith("?"):
                    continue
                m = re.match(r"(?:(?:please|now|just|okay|ok|come\s+on)[,!]?\s+)?"
                             r"(?P<neg>(?:don't|do\s+not|never)\s+)?(?P<act>.*)", line, re.I)
                kind = _deferred_kind(m.group("act")) if m and not m.group("neg") else ""
                if kind:
                    found.append((heard[0], kind))
    out = {}
    for name, kind in found:
        if acts_in(acted, name, sheet, described):
            continue
        out[name] = "wear" if "wear" in (kind, out.get(name)) else kind
    return list(out.items())[:2]


def acts_in(acted, name, sheet="", described=()):
    """Does `name` -- or the pronoun only they answer to here -- DO something in the
    acted text? A name followed by a verb of its own: "she kneels", "Dan, shaking, sits"."""
    t = str(acted or "")
    who = [re.escape(name)]
    rows = dict((n, ln) for n, ln in sheet_lines(sheet) if n)
    pron = sheet_pronoun(rows.get(name, ""))
    if pron in ("she", "he") and sum(1 for n in (described or [])
                                     if sheet_pronoun(rows.get(n, "")) == pron) == 1:
        who.append(pron)
    return bool(re.search(
        r"\b(?:" + "|".join(who) + r")\b(?:\s*,[^,.;]*,)?\s*,?\s*(?:(?:and|then)\s+)?(?:\w+ly\s+)?"
        r"(?!(?:is|was|and|or|but|then|to|as|in|on|at|with|by|of|for)\b)[a-z]+(?:s|es|ed)\b",
        t, re.I))


def told_hold(listeners, pronouns=None, wearing=()):
    """Keep what an order or an intention asks for OUT of this shot: where the person
    it is about is, and -- for clothes -- what they wear, to the last frame.

    It used to give the listener something to do as well: "What is yet to come happens
    in a later shot: Ana listens and reacts, staying in place". REPORTED as characters
    doing things the beat never wrote, and the beat losing its share of the prompt. The
    reaction was the node's, not the author's; where she is and what she wears is the
    guarantee."""
    who = [n for n in (listeners or []) if n]
    if not who:
        return ""
    pron = dict(pronouns or {})
    wear = set(wearing or ())
    out = ""
    for n in who[:2]:
        p = pron.get(n, "")
        place = (f"where {p} {'are' if p == 'they' else 'is'}" if p in ("she", "he", "they")
                 else "in place")
        poss = {"she": "her", "he": "his", "they": "their"}.get(p, f"{n}'s")
        out += (f" {n} stays {place}"
                + (f", wearing what {poss} entry lists" if n in wear else "") + ".")
    return out


# The tail both voice guards end on, defined once so they cannot drift apart.
# THE MOUTHS ONLY. It ended "those expressions moving", which is the node directing a
# face the beat never wrote. REPORTED as characters doing things the beat never said
# and the beat losing its share of the prompt. Closed mouths and whose voice it is are
# the guarantee; the silence pin on the audio branch does the rest.
MOUTH_HOLD_REST = "every other mouth in the shot stays closed"
_UP_TO_TWO_WORDS = r"(?:\s+(?!and\b|but\b|then\b|who\b|,\s*who\b)[\w,']+){0,2}?"


_VOCAL_SOURCE = (
    # Before the bare verbs: "makes sounds of pleasure" is hers, and "cries out" is not
    # crying. Unread, the first left her out of the voices and CLOSED HER MOUTH on the
    # shot's own sound whenever somebody else in the beat was named making one.
    (r"(?:makes?|making|made|lets?\s+out|letting\s+out)\s+(?:[\w,']+\s+){0,2}?"
     r"(?:" + _PLEASURE_SOUND + r")", "moaning"),
    (r"cr(?:y|ies|ying|ied)\s+out", "crying out"),
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


def second_vocal(beat, name, others=()):
    """The vocal `name` makes as the SECOND verb of their own clause -- "Mara thrashes
    and screams", "struggles against the cuffs and screams" -- with nobody else named
    in between. "" when there is none. vocal_sources_in reads only a vocal right after
    the name, so a gagged woman's scream written this way went unmuffled and unowned.
    REPORTED. Used for the one gagged person in a shot."""
    b = str(beat or "")
    for pat, phrase in _VOCAL_SOURCE:
        for m in re.finditer(r"\b" + re.escape(name) + r"\b([^.;!?]*?)\band\s+"
                             r"(?:\w+ly\s+)?(?:" + pat + r")\b", b, re.I):
            if not any(re.search(r"\b" + re.escape(o) + r"\b", m.group(1))
                       for o in others if o and o != name):
                return phrase
    return ""


def unpinned_vocal(beat):
    """Does a pronoun make a vocal here -- 'she moans' -- that no name can be given?

    vocal_sources_in reads names only, so in "Dan grunts, and she makes sounds of
    pleasure" it finds Dan alone, and the guard closing every OTHER mouth closes hers:
    the one making the sound. A vocal the beat does not pin on anybody holds nobody,
    and that has to hold when it sits beside one that is pinned."""
    b = str(beat or "")
    return any(re.search(r"\b(?:she|he|they)\b" + _UP_TO_TWO_WORDS + r"\s+(?:" + pat + r")\b",
                         b, re.I) for pat, _ in _VOCAL_SOURCE)


def _joined(names):
    """'Dan', 'Dan and Sam', 'Dan, Sam and Mara'."""
    ns = [n for n in (names or []) if n]
    if len(ns) < 2:
        return ns[0] if ns else ""
    return ", ".join(ns[:-1]) + " and " + ns[-1]


def voice_sources(talkers, vocal, vocalisers, silent, pairs=None, rest=None):
    """Say whose voice is whose, and close the mouths that are neither.

    Two jobs, and the second is the reported one. Closing the rest stops the
    listener babbling on a branch somebody else's sob opened. NAMING THE SOURCES
    stops the model swapping them -- two voices in one shot with nothing saying
    which is which is a shot where he can be given her whimper and she his line.

    So the sentence is emitted for two DIFFERENT sources even when nobody is left to
    hold: with one source and nobody silent there is nothing to disambiguate and
    nothing to close, and the shot is left alone.

    `pairs` is vocal_sources_in's (name, vocal) list. Each vocal is credited to whoever
    makes it: one word for everybody gave "the grunting is Dan and Mara's" to a beat
    where she was moaning -- her sound, described as his."""
    parts = []
    if len(talkers or []) == 1:
        parts.append(f"only {talkers[0]} speaks")
    elif talkers:
        parts.append(f"{talkers[0]} speaks first, then "
                     + ", then ".join(talkers[1:]))
    by = {}
    if vocalisers and vocal:
        for n, ph in (pairs or [(n, vocal) for n in vocalisers]):
            by.setdefault(ph, []).append(n)
        parts.append(" and ".join(f"the {ph} is {_joined(ns)}'s" for ph, ns in by.items()))
    # Two different sources is a line and a vocal, two lines, or two vocals.
    if not parts or (not silent and len(talkers or []) + len(by) < 2):
        return ""
    if silent:
        parts.append(rest or MOUTH_HOLD_REST)
    said = "; ".join(parts)
    return f" {said[0].upper()}{said[1:]}."

ONE_VOICE = (" Only the person speaking has their mouth moving; every other mouth "
             "in the shot stays closed.")

# The same guards where the beat puts somebody's mouth to work -- a smile, a kiss, a
# bitten lip, an angry face. "Stays closed" would argue with that, and the guard used
# to stand down ENTIRELY there instead, leaving the listener's mouth as free as the
# speaker's. REPORTED as other characters babbling in a beat where only one has a
# line. Silent says what matters -- no voice -- and leaves the mouth to the beat.
MOUTH_SILENT_REST = "every other mouth in the shot is silent"
ONE_VOICE_BUSY = (" Only the person speaking has a voice; every other mouth in the "
                  "shot is silent.")


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


def untagged(line):
    """A sheet line with its <Picture N> tags taken out and the punctuation mended."""
    out = _PICTURE_TAG.sub("", str(line or ""))
    out = re.sub(r":\s*,", ":", out)
    out = re.sub(r",\s*(?=,)", "", out)
    out = re.sub(r"\s+([,.])", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip()


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


def drop_portraits(text, refs, slots):
    """(text, refs) with the pictures in `slots` (1-based) taken out of the shot.

    Their tags go with them and every tag left is renumbered to match what the shot
    still carries -- resolve_tags' own renumbering, run on the remainder. Used where a
    frame of the person as they are now stands in for their portrait."""
    gone = {int(k) for k in (slots or ())}
    refs = list(refs or [])
    if not gone:
        return text, refs
    out = _PICTURE_TAG.sub(
        lambda m: "\x00" if int(m.group(1)) in gone else m.group(0), text or "")
    out = re.sub(r"([.!?])[ \t]*\x00[ \t]*\.", r"\1", out)     # "jeans. <Picture 1>."
    out = re.sub(r":[ \t]*\x00[ \t]*,[ \t]*", ": ", out)        # "Mara: <Picture 1>, she"
    out = re.sub(r",[ \t]*\x00[ \t]*(?=[,.;])", "", out)        # ", <Picture 1>."
    out = re.sub(r"[ \t]*\x00", "", out)
    out, kept, _ = resolve_tags(out, refs)
    return out, kept


def held_picture_note(n, who, items=None, where=None):
    """One sentence for a portrait that predates what holds its subject now.

    The portrait shows them before the restraint or gag went on, and a picture is the
    strongest thing in the prompt. Where no frame of them as they are now exists, the
    portrait stays for the face and this names what is on them in this shot. Reported
    as tape and cuffs gone in the shot after they went on."""
    names = merge_hardware_names(items or []) or ["restraints"]
    said, placed = [], False
    for it in names:
        at = _where_of(it, where, who=who) if where else ""
        placed = placed or bool(at)
        said.append(f"the {it} {at}" if at else f"the {it}")
    plural = len(names) > 1 or names[0].lower().endswith("s")
    return (f" <Picture {n}> shows who {who} is; {_join_names(said)} "
            f"{'are' if plural else 'is'} in place{'' if placed else f' on {who}'} now.")


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
                 f"turning keep_frame_after_removal off stops a shot opening on the "
                 f"previous frame after a removal, at the cost of a cut there")
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
    # ...AND THE STAMP, which is where the DiT reads its shifts. model_sampling sets
    # the sampler's schedule and the audio carry's scale; the DiT derives the audio
    # timestep, and converts the carried audio velocity, from transformer_options --
    # falling back to its own 12/3 when nothing is stamped. Patching one and not the
    # other ran FastH3's 10/3 schedule through a DiT reading 12/3: every audio step
    # timestepped and rescaled for a schedule that was not the one sampling. comfy's
    # own MiniMaxH3SigmaShift writes both, and so does this now.
    _write_h3_stamp(m, shift_video, shift_audio)
    return m


def _write_h3_stamp(m, shift_video, shift_audio):
    """Write the H3 shifts into m's transformer_options, as MiniMaxH3SigmaShift does.
    m must already be a clone: the dict is replaced, never edited in place."""
    if not isinstance(getattr(m, "model_options", None), dict):
        return m                    # a model that carries no options has nowhere to stamp
    to = m.model_options["transformer_options"] = dict(
        m.model_options.get("transformer_options", {}) or {})
    to["minimax_h3_sigma_shift_video"] = float(shift_video)
    to["minimax_h3_sigma_shift_audio"] = float(shift_audio)
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


# FastVideo's FastH3 V2, as ComfyUI's own guide runs it: 8 steps, res_multistep on
# simple, sigma shift 10 video / 3 audio, VSA at keep 10 from 20% of the schedule.
FAST_H3_STEPS = 8
FAST_H3_SHIFT_VIDEO = 10.0
FAST_H3_SHIFT_AUDIO = 3.0


def fast_h3(model):
    """Is this a FastVideo FastH3 checkpoint?

    The one H3 build with VSA gate layers (attn.to_gate_compress): ComfyUI's own
    detection marks exactly those checkpoints "VSA-trained", and FastH3 is what is
    trained that way. It MATTERS here because FastH3 is a DMD2 distill of first/last
    frame generation only -- "Ref2VA (multi-reference conditioning) was not distilled"
    -- and this node leans on reference rows everywhere: tagged pictures, recovered
    and evened faces, returning rooms, the frame carried across a cut. A model that
    never learned what a reference row is draws the picture as somebody else.
    REPORTED as character duplication coming back on a FastH3 model."""
    try:
        blk = model.model.diffusion_model.blocks[0]
        return getattr(blk.attn, "to_gate_compress", None) is not None
    except Exception:
        return False


# VIDEO REBIRTH'S HYPERFLOW, an 8-step LoRA for H3 distilled onto a FIXED sigma grid.
# The one LoRA that does say what schedule it wants: its safetensors metadata carries
# the grid and both shifts (hyperflow_sigmas, hyperflow_video_shift/audio_shift), and
# comfy keeps that metadata on the MODEL it hands this node. The grid is the base grid
# -- symmetric about 0.5, as upstream's "manual sigmas" are -- and the video shift is
# applied to it the way diffusers' flow scheduler applies its shift to manual sigmas.
# Fed unshifted, half its steps would land in the last 8% of the noise.
# These are v1.0's numbers, for a workflow whose metadata was lost behind a later LoRA.
HYPERFLOW_SIGMAS = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764,
                    0.068494, 0.0)
HYPERFLOW_SHIFT_VIDEO = 12.0
HYPERFLOW_SHIFT_AUDIO = 3.0
HYPERFLOW_SAMPLER = "euler"


def _hyperflow_grid(value):
    """A metadata sigma list, as floats, if it is a usable grid -- 1 down to 0."""
    try:
        grid = [float(x) for x in (json.loads(value) if isinstance(value, str) else value)]
    except (TypeError, ValueError):
        return None
    if (len(grid) < 2 or abs(grid[0] - 1.0) > 1e-6 or abs(grid[-1]) > 1e-6
            or any(b >= a for a, b in zip(grid, grid[1:]))):
        return None
    return tuple(grid)


def hyperflow_lora(model, graph=None):
    """{sigmas, shift_video, shift_audio, gate, source} if Hyperflow is on this model.

    Read from the LoRA metadata comfy attaches to the model first -- that is the
    file's own statement, numbers and all. comfy keeps only the LAST LoRA's metadata,
    so a LoRA stacked after Hyperflow hides it; the workflow's LoRA file names are
    the fallback there, with v1.0's numbers. None when neither says Hyperflow."""
    meta = None
    try:
        get = getattr(model, "get_attachment", None)
        meta = get("lora_metadata") if callable(get) else \
            (getattr(model, "attachments", None) or {}).get("lora_metadata")
    except Exception:
        meta = None
    if isinstance(meta, dict) and str(meta.get("hyperflow", "")).strip().lower() == "true":
        grid = _hyperflow_grid(meta.get("hyperflow_sigmas")) or HYPERFLOW_SIGMAS

        def _num(key, default):
            try:
                return float(meta.get(key, default))
            except (TypeError, ValueError):
                return default
        return {"sigmas": grid,
                "shift_video": _num("hyperflow_video_shift", HYPERFLOW_SHIFT_VIDEO),
                "shift_audio": _num("hyperflow_audio_shift", HYPERFLOW_SHIFT_AUDIO),
                "gate": _num("hyperflow_gate", 0.0),
                "version": str(meta.get("hyperflow_version", "")),
                "source": "the LoRA's own metadata"}
    for node in (graph.values() if isinstance(graph, dict) else ()):
        inputs = node.get("inputs") if isinstance(node, dict) else None
        for key, value in (inputs.items() if isinstance(inputs, dict) else ()):
            if ("lora_name" in str(key).lower() and isinstance(value, str)
                    and "hyperflow" in value.lower()):
                return {"sigmas": HYPERFLOW_SIGMAS, "shift_video": HYPERFLOW_SHIFT_VIDEO,
                        "shift_audio": HYPERFLOW_SHIFT_AUDIO, "gate": 0.0, "version": "",
                        "source": f"the file name {value}"}
    return None


def hyperflow_sigmas(grid, shift_video):
    """The VIDEO sigmas a sampler runs on: the base grid through H3's flow shift.

    The DiT takes a sampler sigma as the video sigma as it stands and derives the
    audio one from it (comfy's time_shift_sigma), so this is the only shift applied
    by hand -- the audio branch follows from the stamp. See stamp_h3_shift."""
    s = float(shift_video)
    return torch.tensor([s * x / (1.0 + (s - 1.0) * x) for x in grid], dtype=torch.float32)


def stamp_h3_shift(model, shift_video, shift_audio):
    """A clone carrying the shifts in transformer_options, as MiniMaxH3SigmaShift does.

    The DiT reads its shifts from that stamp, not from model_sampling, and falls back
    to its own 12/3 without one. A hand-built schedule has to be read back with the
    shifts it was built with, so it says so rather than leaning on the default --
    or on whatever an upstream patch stamped for a different schedule."""
    try:
        return _write_h3_stamp(model.clone(), shift_video, shift_audio)
    except Exception:
        return model


# HYPERFLOW'S SECOND TIME. Upstream conditions every step on where it STARTS (t) and
# where it LANDS (r = 1 - sigma_next), blended at a gate stored in the file:
#     t_emb = emb_t(t) + gate * (emb_r(r) - emb_t(t))
# emb_r is a second copy of the base time embedder with a LoRA of its own. comfy's H3
# has one embedder and no idea of an endpoint, and the ComfyUI conversion of the LoRA
# dropped emb_r's LoRA for that reason -- so it ran one-time, on a distill trained
# two-time. The endpoint LoRA is read from the original release instead (only its
# time-embedder tensors are needed, 28 MB of 2.8 GB) and blended in here.
HYPERFLOW_ENDPOINT_KEY = "transformer.endpoint_time_embedder.linear_1.lora_A.weight"
HYPERFLOW_WRAPPER_KEY = "h3_longvideos_hyperflow_two_time"
_HYPERFLOW_TE_KEY = "diffusion_model.time_embedder.proj_in.weight"
_HYPERFLOW_DELTAS = {}


def _safetensors_keys(path):
    """(keys, metadata) from a safetensors header, without reading any tensor."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if not 0 < n < 64 << 20:
                return set(), {}
            h = json.loads(f.read(n))
        return set(h) - {"__metadata__"}, (h.get("__metadata__") or {})
    except Exception:
        return set(), {}


def hyperflow_endpoint_file():
    """A file holding Hyperflow's endpoint time-embedder LoRA, or "".

    The sidecar this node keeps beside itself first (hyperflow_endpoint_*.safetensors:
    the time-embedder tensors of the original release), then any LoRA with
    "hyperflow" in its name that has them -- the full original file in models/loras."""
    cands = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "hyperflow_endpoint_*.safetensors")))
    try:
        import folder_paths
        for name in folder_paths.get_filename_list("loras"):
            if "hyperflow" in str(name).lower():
                cands.append(folder_paths.get_full_path("loras", name))
    except Exception:
        pass
    for p in cands:
        if p and HYPERFLOW_ENDPOINT_KEY in _safetensors_keys(p)[0]:
            return p
    return ""


def hyperflow_te_strength(model):
    """The strength Hyperflow's time-embedder LoRA is on the model at, or None.

    None is the case that matters: a curve-form checkpoint (adaln_t_table, no time
    embedder) has nowhere for those weights to go, and comfy drops them with a log
    line nobody reads."""
    try:
        entries = (getattr(model, "patches", None) or {}).get(_HYPERFLOW_TE_KEY) or []
        return float(entries[0][0]) if entries else None
    except Exception:
        return None


def hyperflow_endpoint_deltas(path, strength):
    """(d_in, d_out), fp32: what turns the time embedder AS PATCHED -- base plus the
    LoRA's time delta at `strength` -- into the endpoint embedder, base plus its own
    delta at the same strength. Linear in the weights, so it is one difference each."""
    key = (path, os.path.getmtime(path), float(strength))
    if key in _HYPERFLOW_DELTAS:
        return _HYPERFLOW_DELTAS[key]
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        try:
            scale = float(meta.get("lora_alpha", 1)) / float(meta.get("lora_rank", 1))
        except (TypeError, ValueError, ZeroDivisionError):
            scale = 1.0

        def delta(module, linear):
            b = f.get_tensor(f"transformer.{module}.{linear}.lora_B.weight").float()
            a = f.get_tensor(f"transformer.{module}.{linear}.lora_A.weight").float()
            return b @ a
        s = float(strength) * scale
        out = (s * (delta("endpoint_time_embedder", "linear_1") - delta("time_embedder", "linear_1")),
               s * (delta("endpoint_time_embedder", "linear_2") - delta("time_embedder", "linear_2")))
    _HYPERFLOW_DELTAS.clear()
    _HYPERFLOW_DELTAS[key] = out
    return out


def _tss(sigma, from_shift, to_shift):
    """comfy's time_shift_sigma: a video sigma moved onto the audio schedule."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def hyperflow_two_time_wrapper(gate, d_in, d_out):
    """A DIFFUSION_MODEL wrapper blending Hyperflow's endpoint into the time embedding.

    Per call: the step's endpoint is the next sigma down the schedule the sampler is
    running (transformer_options["sample_sigmas"]); video and text rows land at
    1 - sigma_next, audio rows at the same point on the audio schedule, and rows
    pinned at their own time -- keyframes, references -- at that time (r = t). Where
    video and audio share a time (the first step, sigma 1) they share a table row in
    the DiT, and the video endpoint wins it. The embedder's forward is swapped only
    for the length of the call and put back however it ends."""
    on_device = {}

    def _on(delta, like):
        k = (id(delta), like.device, like.dtype)
        if k not in on_device:
            on_device.clear() if len(on_device) > 4 else None
            on_device[k] = delta.to(device=like.device, dtype=like.dtype)
        return on_device[k]

    def wrapper(executor, x, timestep, context, transformer_options={}, *args, **kwargs):
        dm = getattr(executor, "class_obj", None)
        te = getattr(dm, "time_embedder", None)
        ss = (transformer_options or {}).get("sample_sigmas")
        if te is None or getattr(dm, "use_adaln_curves", False) or ss is None or not gate:
            return executor(x, timestep, context, transformer_options, *args, **kwargs)
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video",
                                                getattr(dm, "sigma_shift_video", 12.0)))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio",
                                                getattr(dm, "sigma_shift_audio", 3.0)))
        sigma = max(float(timestep.flatten()[0]) / 1000.0, 1e-6)    # as the DiT reads it
        grid = [float(v) for v in torch.as_tensor(ss).flatten().tolist()]
        nxt = max((v for v in grid if v < sigma - 1e-6), default=0.0)
        t_v, t_a = 1.0 - sigma, 1.0 - _tss(sigma, shift_v, shift_a)
        r_v, r_a = 1.0 - nxt, 1.0 - _tss(nxt, shift_v, shift_a)
        had = te.__dict__.get("forward")
        inner = te.forward

        def endpoint(r):
            def add(delta):
                def hook(_mod, inp, out):
                    return out + torch.nn.functional.linear(inp[0].to(out.dtype), _on(delta, out))
                return hook
            hooks = (te.proj_in.register_forward_hook(add(d_in)),
                     te.proj_out.register_forward_hook(add(d_out)))
            try:
                return inner(r)
            finally:
                for h in hooks:
                    h.remove()

        def two_time(t):
            e_t = inner(t)
            tt = t.to(torch.float32)
            r = torch.where((tt - t_a).abs() < 1e-5, torch.full_like(tt, r_a), tt)
            r = torch.where((tt - t_v).abs() < 1e-5, torch.full_like(tt, r_v), r)
            e_r = endpoint(r.to(t.dtype)).to(e_t.dtype)
            return e_t + float(gate) * (e_r - e_t)

        te.forward = two_time
        try:
            return executor(x, timestep, context, transformer_options, *args, **kwargs)
        finally:
            if had is not None:
                te.forward = had
            else:
                te.__dict__.pop("forward", None)
    return wrapper


def _hyperflow_patched_elsewhere(model):
    """True when something else -- the ComfyUI-HyperFlow pack's loader -- already put
    a two-time patch on this model. Applying a second would blend the endpoint twice."""
    try:
        for table in (getattr(model, "wrappers", None) or {},
                      getattr(model, "callbacks", None) or {}):
            for keyed in table.values():
                if any("hyperflow" in str(k).lower() and k != HYPERFLOW_WRAPPER_KEY
                       for k in (keyed or {})):
                    return True
        if any("time_embedder" in str(k) or "hyperflow" in str(k).lower()
               for k in (getattr(model, "object_patches", None) or {})):
            return True
        te = getattr(model.get_model_object("diffusion_model"), "time_embedder", None)
        return bool(te is not None and "forward" in te.__dict__)
    except Exception:
        return False


def hyperflow_two_time_plan(model, hyper):
    """(ready, what to tell the reader). Ready means install_hyperflow_two_time will
    run; otherwise the note says what is missing and what that costs."""
    try:
        dm = model.get_model_object("diffusion_model")
    except Exception:
        dm = None
    if dm is not None and (getattr(dm, "use_adaln_curves", False)
                           or getattr(dm, "time_embedder", "absent") is None):
        return False, (
            "THIS CHECKPOINT CANNOT CARRY HYPERFLOW'S TIME CONDITIONING: it is a pruned "
            "curve-form build, with a precomputed adaln_t_table where the time embedder "
            "was, so the LoRA's time-embedder weights have nowhere to load -- comfy drops "
            "them -- and its endpoint conditioning has nothing to attach to. Every step "
            "is timed by the BASE model's embedding under a LoRA distilled for its own. "
            "Use a checkpoint with a time embedder (minimax_h3_fl2va_int8_convrot is one) "
            "for Hyperflow")
    if _hyperflow_patched_elsewhere(model):
        return False, ("a two-time Hyperflow patch is already on the model upstream (the "
                       "ComfyUI-HyperFlow pack's loader), so this node leaves it to that "
                       "one rather than blending the endpoint in twice")
    s = hyperflow_te_strength(model)
    if s is None:
        return False, ("Hyperflow's time-embedder LoRA is not on this model, so its "
                       "endpoint conditioning cannot be added -- check that the LoRA "
                       "loader's strength_model is not 0")
    path = hyperflow_endpoint_file()
    if not path:
        return False, (
            "Hyperflow's endpoint conditioning is OFF: this ComfyUI conversion of the "
            "LoRA leaves out the endpoint time embedder, and no file holding it was "
            "found. Put the original minimax_h3_hyperflow_8step_v1.0.safetensors "
            "(videorebirth/hyperflow on Hugging Face) in models/loras -- it is only read "
            "for its time-embedder tensors -- and every step will be conditioned on where "
            "it lands as well as where it starts, as the LoRA was trained")
    gate = float(hyper.get("gate") or 0.0) or float(
        _safetensors_keys(path)[1].get("hyperflow_gate", 0.0) or 0.0)
    if not gate:
        return False, "Hyperflow's endpoint gate is 0 in its metadata, so there is nothing to blend"
    hyper["gate"], hyper["endpoint_file"], hyper["te_strength"] = gate, path, s
    return True, (f"Hyperflow's endpoint conditioning is ON: each step's time embedding "
                  f"blends where it starts with where it lands at gate {gate:g}, as the "
                  f"LoRA was trained -- endpoint weights from {os.path.basename(path)}")


def install_hyperflow_two_time(model, hyper):
    """A clone carrying the two-time wrapper. `hyper` must have passed the plan."""
    d_in, d_out = hyperflow_endpoint_deltas(hyper["endpoint_file"], hyper["te_strength"])
    import comfy.patcher_extension as _pe
    m = model.clone()
    m.add_wrapper_with_key(_pe.WrappersMP.DIFFUSION_MODEL,
                           HYPERFLOW_WRAPPER_KEY,
                           hyperflow_two_time_wrapper(hyper["gate"], d_in, d_out))
    return m


# FASTH3 IS VSA-TRAINED. ComfyUI's guide runs it through Model Sparse Attention set to
# vsa, keep 10%, from 20% of the schedule; without it the coarse-branch gate layers the
# distill learned sit unused and the attention is not the one it was trained with.
FAST_H3_VSA_KEEP = 0.10
FAST_H3_VSA_START = 0.20


def apply_fast_h3_vsa(model):
    """(model, note): VSA put on a FastH3 model, unless it cannot or need not be.

    Never over a DiT block patch already on the model -- the reader's own
    sparse-attention node, or an upstream Fun ControlNet, whose control VSA would
    replace (see sparse_dit_patched) -- and never under the cudaMallocAsync allocator --
    that combination aborts the process (see sparse_attention_allocator_abort), and here
    it would be the node's doing."""
    if sparse_dit_patched(model) is not False:
        return model, ""            # already patched upstream, or a model that cannot say
    conf = str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") or "")
    if "cudamallocasync" in conf.lower():
        return model, ("FastH3's VSA attention was NOT applied: torch is on the "
                       "cudaMallocAsync allocator, and sparse attention under it aborts "
                       "the ComfyUI process. Restart ComfyUI with --disable-cuda-malloc and "
                       "the node applies it")
    try:
        from comfy_extras.nodes_sparse_attention import (apply_block_sparse_attention,
                                                         parse_block_list)
        m = apply_block_sparse_attention(
            model, tau=1.3, topk_ratio=FAST_H3_VSA_KEEP, vsa=True,
            start_percent=FAST_H3_VSA_START, end_percent=1.0, min_tokens=12288,
            dense_blocks=parse_block_list(""), sink_conditioning="exact_kv_and_rows",
            extra_tokens=0, verbose=False)
    except Exception as e:
        return model, (f"FastH3's VSA attention could not be applied ({type(e).__name__}: "
                       f"{e}); it is running dense, which it was not trained on")
    return m, (f"FastH3's VSA attention applied: keep {FAST_H3_VSA_KEEP * 100:g}%, from "
               f"{FAST_H3_VSA_START * 100:g}% of the schedule, as it was trained")


def sparse_dit_patched(model):
    """True when something upstream installed a per-block DiT replace patch, False when it
    provably did not, None when this model cannot say.

    That patch is how ComfyUI's Model Sparse Attention node registers itself
    (set_model_patch_replace -> model_options["transformer_options"]["patches_replace"]
    ["dit"]), whatever method it was set to. None is distinct from False on purpose: a stub
    or hand-built model carries no model_options at all, and a caller of this turns a
    True into a refusal, so "cannot tell" must never be read as "not there".

    ANY patch in those slots counts, a Fun control block included. apply_fast_h3_vsa
    reads this: comfy's apply_block_sparse_attention REPLACES every slot rather than
    wrapping it, so VSA put over an upstream "Apply MiniMax H3 Fun ControlNet" would
    drop that control silently. Pose control installs its own block per shot, on a clone
    made after VSA, so it never reaches this check."""
    opts = getattr(model, "model_options", None)
    if not isinstance(opts, dict):
        return None
    tops = opts.get("transformer_options") or {}
    if not isinstance(tops, dict):
        return None
    return bool((tops.get("patches_replace") or {}).get("dit"))


def sparse_attention_patched(model):
    """sparse_dit_patched, with a slot holding only a MiniMax H3 Fun control block (or a
    chain of them) not counted: that block is not sparse attention. For the allocator
    check alone, whose question is whether sparse attention runs at all."""
    found = sparse_dit_patched(model)
    if not found:
        return found
    dit = model.model_options["transformer_options"]["patches_replace"]["dit"]
    if not isinstance(dit, dict):
        return True
    return any(_holds_sparse_patch(p) for p in dit.values())


# comfy's MiniMax H3 Fun ControlNet registers its blocks in the same "dit" replace slots
# as sparse attention (comfy_extras/nodes_minimax_h3.py). It is not sparse attention: it
# keeps whatever patch was on the slot before it as `.previous` and calls through it. So
# for the allocator check a Fun block counts only through what it wraps. Matched by class
# name, so comfy_extras need not be importable here.
_FUN_CONTROL_BLOCK = "MiniMaxH3FunControlBlockPatch"


def _holds_sparse_patch(patch):
    """False for a dit replace entry that is the Fun control block patch alone (or a chain
    of them); True for anything else, including a Fun block wrapping another patch."""
    seen = 0
    while type(patch).__name__ == _FUN_CONTROL_BLOCK and seen < 64:
        patch = getattr(patch, "previous", None)
        if patch is None:
            return False
        seen += 1
    return True


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
    if sparse_attention_patched(model) is not True:
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
    r"(?=[^.;!?]{0,40}?" + engine.TO_THE_FLOOR + r")"
    # "gets her jacket off" -- the object between, and no second verb.
    r"|\b(?:get(?:s|ting)?|got)\b(?=(?:\s+(?!(?:and|then|up|out|back|down|in|on|into|"
    r"onto|to|off)\b)[\w’'-]+){1,4}\s+off\b)",
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


def off_by_last_frame(items, agent="", scene="", beat="", wearer_sheet=""):
    """State that a removal FINISHES inside this shot. Empty when nothing came off.

    `wearer_sheet` is the sheet line of the person it comes off, read first for the
    garment's name: two people in shirts are two shirts, and the whole scene's
    longest "shirt" was the other person's.

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
        nm = scene_name_for(i, wearer_sheet) or scene_name_for(i, scene) or i
        tag = scene_tag_for(i, wearer_sheet) or scene_tag_for(i, scene)
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
    _named = engine.garment_masked(b)       # "on top of her" is not a top -- see there
    for pat in (_PUTS_ON, _DRESSES):
        for m in pat.finditer(b):
            window = _named[max(0, m.start() - 60):min(len(b), m.end() + 60)]
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
# Closed, and nothing more: "the expressions moving" asked every face for a
# performance the beat did not write -- see MOUTH_HOLD_REST.
MOUTH_HOLD = " Mouths in the shot stay closed."


# A GAGGED MOUTH IS NOT A MOUTH THE SHOT CAN USE.
#
# The tape was one item in the hardware sentence -- "The cuffs on the wrists and duct
# tape over the mouth stay closed and fastened as they were put on, a closed ring locked
# round each wrist" -- words written for cuffs, eight sentences down, while three other
# clauses in the same shot handed her mouth a job: "the screaming is Mara's", "the
# expression is terrified, played in the eyes and the mouth together", "the mouth set".
# A mouth told to scream and to act cannot be under tape, so the model took the tape
# off. REPORTED as gags like duct tape disappearing once another action happens.
#
# So the gag gets its own sentence, in the opening tokens beside the pose; a sound
# from behind it is muffled; and a face that acts does it above it. Positively, like
# everything here: where the tape IS, never where the mouth is not.
def gag_hold(item, who="", new=False, muffled=False):
    """One person's gag, held for the whole shot. "" when there is no item.

    `new` is the shot that puts it on -- off at the first frame, on in the first half,
    held for the rest, and in plain view with the hand off it at the last frame.
    `muffled` adds what a voice behind it sounds like, for a shot that gives that
    person a sound or a line."""
    item = str(item or "").strip()
    if not item:
        return ""
    mouth = (f"{who} mouth" if who in ("her", "his", "their")
             else f"{who}'s mouth" if who else "the mouth")
    taped = bool(re.search(r"\btape\b", item, re.I))
    if new:
        # ON BY MID-SHOT, HAND AWAY AT THE END. "In place at the last frame" put the
        # deadline on the one frame the next shot opens on, so that frame often showed
        # the hand pressing it on and no tape. Reported as tape gone in the next
        # beat. The second half of the shot is the hold, and the last frame shows it.
        _on = "across" if taped else "over"
        out = (f" The {item} goes {_on} {mouth} during this shot: it is on in the first "
               f"half of the shot and "
               f"{'stuck flat over the lips' if taped else 'fastened in place'} for the "
               f"rest of it, and at the last frame it is {_on} {mouth} in plain view and "
               f"the hand that put it there has let go and is clear of the face.")
    else:
        # SHORT ONCE IT IS ON. The held form ran to 25-30 words in every shot after the
        # taping ("sealing the lips from cheek to cheek from the first frame to the
        # last, in place through every movement in the shot"), and "every movement"
        # read as a call for some. REPORTED as gag shots getting heavier and the beat
        # losing its share of the prompt. Where it is and that it stays is the hold.
        out = (f" The {item} stays stuck flat across {mouth}." if taped
               else f" The {item} stays fastened in place over {mouth}.")
    if muffled:
        # ...around a ball, a bit, a ring or a stuffed cloth the mouth is held open, not
        # shut: see engine.mouth_held_open.
        _lips = ("the mouth held open around the" if engine.mouth_held_open(item)
                 else "the lips held shut under the")
        out += f" Every sound from behind it comes out muffled, {_lips} {item.split()[-1]}."
    return out


def mouths_closed_but(held_open, others):
    """MOUTH_HOLD for everybody but the people whose gag holds the mouth open -- the gag
    sentence has already said where theirs is. "" when nobody else is in the shot."""
    if not held_open:
        return MOUTH_HOLD
    if not others:
        return ""
    return " " + MOUTH_HOLD_REST[0].upper() + MOUTH_HOLD_REST[1:] + "."


def gagged_in(state, names):
    """{name: the item over their mouth} for the people in `names` who are gagged."""
    out = {}
    for n in (names or []):
        q = getattr(state, "people", {}).get(n)
        if q is None:
            continue
        on_mouth = [r.item for r in q.hardware.values() if r.part == "mouth" and r.item]
        if on_mouth:
            out[n] = merge_hardware_names(on_mouth)[0]
    return out


def eyes_above(face, item, who=""):
    """A face clause with its acting moved up to the eyes and brow, above a gag --
    `who`'s sentence only, when given."""
    if not face:
        return face
    head = (str(item or "").split() or ["gag"])[-1]
    lead = (re.escape(who) + r"['’]s") if who else r"(?:[\w'’-]+|The)"
    out = re.sub(r"(" + lead + r" expression is [\w-]+)\.",
                 lambda m: f"{m.group(1)}, in the eyes and the brow above the {head}.", face)
    out = re.sub(r"(" + lead + r" faces? shows? the strain), the mouths? set\.",
                 lambda m: f"{m.group(1)} in the eyes and the brow above the {head}.", out)
    return out


# A MOUTH IN USE, for the closed-mouth line only: "Sasha drinks the lemonade in one
# go" and "blows out the candles" were told mouths stay closed. REPORTED. Not a voice,
# so the silence pin is untouched.
_MOUTH_EATS = re.compile(
    r"\b(?:eat(?:s|ing)?|ate|drink(?:s|ing)?|drank|sip(?:s|ped|ping)?|gulp(?:s|ed|ing)?|"
    r"swallow(?:s|ed|ing)?|blow(?:s|ing)?|blew|whistl(?:e|es|ed|ing)|pant(?:s|ed|ing)?|"
    r"catch(?:es|ing)?\s+(?:her|his|their)\s+breath|caught\s+(?:her|his|their)\s+breath|"
    r"bit(?:e|es|ing)?\s+(?:into|off|down)|chomp(?:s|ed|ing)?|munch(?:es|ed|ing)?|"
    r"slurp(?:s|ed|ing)?|lick(?:s|ed|ing)?|suck(?:s|ed|ing)?\s+(?:on|at)|"
    r"tast(?:e|es|ed|ing)|spit(?:s|ting)?|spat)\b", re.I)
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


def _bound_by_beat(acted, people, sheet=""):
    """Who among `people` this beat's own words put a strong binding act on."""
    rows = dict((n, ln) for n, ln in sheet_lines(sheet) if n)
    out = []
    for m in _BINDING_ACT.finditer(str(acted or "")):
        txt = m.group(0)
        named = [n for n in people if re.search(r"\b" + re.escape(n) + r"\b", txt)]
        if named:
            out += named
            continue
        pr = re.search(r"\b(her|him|his|them|their|she|he|they)\b", txt, re.I)
        group = {"her": "she", "she": "she", "him": "he", "his": "he", "he": "he",
                 "them": "they", "their": "they", "they": "they"}.get(
                     pr.group(1).lower() if pr else "", "")
        fits = [n for n in people if group and sheet_pronoun(rows.get(n, "")) == group]
        if len(fits) == 1:
            out.append(fits[0])
    return list(dict.fromkeys(out))


# THE FACE OF SOMEBODY HELD, AND NOTHING ABOUT THE MOOD. "The mood is grim." went on
# every shot of a film the node had judged bleak, captor-only shots included, and the
# face clause rode on the same reading, impersonally. REPORTED as characters doing
# things the beat never wrote, and the beat losing its share of the prompt. A film's
# mood is the author's to set in the anchor; the face is said only for a described
# person who is held, and named -- see duress_face.
def strain_face(who):
    """The held face, named: one sentence per person, so a gag's eyes-and-brow wording
    lands on the gagged face alone -- see eyes_above."""
    return "".join(f" {n}'s face shows the strain, the mouth set."
                   for n in [n for n in (who or []) if n][:2])


# A STRUGGLE OR AN EFFORT the beat gives somebody: the only thing, besides a feeling
# or the binding itself, that puts a strained face on them. "Mara's face shows the
# strain" went on every shot she was held in, read off the state -- three times the
# old count, on shots where she sat reading. REPORTED as the node directing the actors.
_EFFORT_SRC = (r"struggl(?:e|es|ed|ing)|thrash(?:es|ed|ing)?|writh(?:e|es|ed|ing)|"
               r"strain(?:s|ed|ing)?|squirm(?:s|ed|ing)?|kick(?:s|ed|ing)?|jerk(?:s|ed|ing)?|"
               r"buck(?:s|ed|ing)?|wriggl(?:e|es|ed|ing)|twist(?:s|ed|ing)?|fight(?:s|ing)?|"
               r"fought|flail(?:s|ed|ing)?|heav(?:e|es|ed|ing)|wrench(?:es|ed|ing)?|"
               r"(?:pull(?:s|ed|ing)?|tug(?:s|ged|ging)?|yank(?:s|ed|ing)?|"
               r"work(?:s|ed|ing)?)\s+(?:hard\s+)?(?:at|against)|"
               r"tr(?:ies|ied|ying)\s+to\s+(?:free|break|pull|wriggle|twist|get\s+free)")


def _efforts_of(acted, people, sheet=""):
    """Who among `people` this beat gives a struggle or an effort to -- by name, by a
    name's body part ("Mara's wrists strain"), or by a pronoun only one of them takes."""
    t = acted or ""
    # ...one of several subjects too: "Mara and Ana struggle".
    _also = r"(?:(?:\s*,\s*|\s+and\s+)(?-i:[A-Z][\w'’-]+))*"
    out = [n for n in people if re.search(
        r"\b" + re.escape(n)
        + r"(?:['’]s\s+\w+)?\b" + _also + _UP_TO_TWO_WORDS
        + r"\s+(?:" + _EFFORT_SRC + r")\b", t, re.I)]
    rows = dict((n, ln) for n, ln in sheet_lines(sheet) if n)
    for g in ("she", "he", "they"):
        if re.search(r"\b" + g + r"\b" + _UP_TO_TWO_WORDS + r"\s+(?:" + _EFFORT_SRC
                     + r")\b", t, re.I):
            fits = [n for n in people if sheet_pronoun(rows.get(n, "")) == g]
            if len(fits) == 1 and fits[0] not in out:
                out.append(fits[0])
    return out


def _binders_of(acted, people, sheet=""):
    """Who among `people` this beat has DOING a binding: the name or pronoun leading
    the sentence or clause that holds a binding verb used as an act. "Dan forces her
    wrists into handcuffs" read the cuffs onto Dan, and the strained face followed
    them onto the man doing the cuffing. REPORTED."""
    t = acted or ""
    rows = dict((n, ln) for n, ln in sheet_lines(sheet) if n)
    out = []
    for sent in re.split(r"(?<=[.!?;])\s+", t):
        acts = [m for m in _BIND_ANY.finditer(sent)
                if not _state_form(sent, m)
                and (_PARTICIPLE_FORM.search(m.group(0))
                     or not _NOUN_BEFORE.search(sent[:m.start()]))]
        if not acts:
            continue
        lead = sent[:acts[0].start()]
        first = re.match(r"\s*(?:then\s+|now\s+)?([A-Z][\w'’-]*)", lead)
        word = first.group(1) if first else ""
        if word in people:
            out.append(word)
        elif word.lower() in ("she", "he", "they"):
            fits = [n for n in people if sheet_pronoun(rows.get(n, "")) == word.lower()]
            if len(fits) == 1:
                out.append(fits[0])
    return list(dict.fromkeys(out))


def duress_face(beat, described, sheet="", held=None, applied_to=()):
    """One short sentence per face: the feeling the beat names, or the strain of
    somebody held. "" otherwise.

    ONLY WHAT THE BEAT WROTE. A face is said for a feeling the beat names, for a held
    person (`held`, read off the engine state by the caller) the beat gives a struggle
    or an effort to, and for the person the binding goes on in this shot (`applied_to`,
    or a binding the beat's own words put on them). It used to go on every shot a
    held person was in. REPORTED as characters doing things the beat never wrote.
    Never on whoever does the binding, unless the beat binds them too, and never on
    somebody with a line: "the mouth set" on a speaker argues with the line."""
    people = [n for n in (described or []) if n]
    if not people:
        return ""
    pairs = emotion_pairs(beat, people, sheet)
    if pairs:
        return mood_faces(pairs)
    acted = engine.acted_text(str(beat or ""))
    if mouth_performs(acted):
        return ""
    bound = set(_bound_by_beat(acted, people, sheet))
    doers = set(_binders_of(acted, people, sheet))
    talking = set(speakers_in(beat, sheet))
    held_set = set(held or ())
    struggling = set(_efforts_of(acted, people, sheet))
    who = [n for n in people
           if n not in talking
           and ((n in held_set and n in struggling)
                or ((n in set(applied_to or ()) or n in bound) and n not in doers))]
    return strain_face(who)


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


# FEELINGS THE BEAT GIVES SOMEBODY, NOT ONES IT MENTIONS. Read off the whole beat, a
# line of dialogue ("I'm not angry") or a negation ("without panic") was handed to a face
# as its expression. REPORTED as characters doing things the beat never wrote. Only the
# acted text, outside quotes, and never a word with not/no/never/n't/without just before.
def _negated(text, at):
    """Is the word at `at` negated within the three words before it?"""
    words = re.findall(r"[\w'’]+", str(text or "")[:at])[-3:]
    return any(w.lower() in ("not", "no", "never", "without", "nor")
               or w.lower().endswith(("n't", "n’t")) for w in words)


def emotion_in(beat):
    """The emotion this beat states, in the author's own word. "" when it states none.
    Read from what the beat acts out -- see _negated."""
    text = engine.acted_text(str(beat or ""))
    for m in _EMOTION.finditer(text):
        if not _negated(text, m.start()):
            return m.group(0).lower()
    m = _BEAMS_AT.search(text)
    return "beaming" if m and not _negated(text, m.start()) else ""


def emotion_owner(beat, names, word, sheet=""):
    """Whose feeling it is: the person the beat puts in front of it. "" if nobody.

    The shape subjects_for uses, conjunction guard included, so "Dan holds the door
    and McKenna is terrified" does not hand the terror to Dan. Takes a name list
    rather than a sheet because the caller already has the shot's cast; with `sheet`
    a pronoun only one of them answers to counts as well ("she is terrified")."""
    b = str(beat or "")
    rows = dict((n, ln) for n, ln in sheet_lines(sheet) if n)
    tail = (_UP_TO_TWO_WORDS + r"\s+(?:is|was|looks?|looked|seems?|feels?|felt|sounds?|"
            r"becomes?|became|goes|went|turns?|gets?|got)?\s*" + re.escape(word) + r"\b")
    for n in (names or []):
        if n and re.search(r"\b" + re.escape(n) + r"\b" + tail, b, re.I):
            return n
    for n in (names or []):
        p = sheet_pronoun(rows.get(n, "")) if n else ""
        if (p in ("she", "he") and sum(1 for o in (names or [])
                                       if sheet_pronoun(rows.get(o, "")) == p) == 1
                and re.search(r"\b" + p + r"\b" + tail, b, re.I)):
            return n
    # ...and trailing its clause: "McKenna stares at the door, desperate" is hers, the
    # clause's subject, where nobody else is named between.
    for m in re.finditer(r",\s*(?:\w+ly\s+)?" + re.escape(word) + r"\b", b, re.I):
        clause = re.split(r"[.;!?]|\b(?:and|but|while|as|then)\b", b[:m.start()])[-1]
        named = [n for n in (names or []) if n and re.search(
            r"\b" + re.escape(n) + r"\b(?!['’]s)", clause)]
        if len(named) == 1 and re.match(r"\s*" + re.escape(named[0]) + r"\b", clause):
            return named[0]
    return ""


def emotion_pairs(beat, names, sheet=""):
    """[(who, feeling)] for the feelings this beat pins on people. Two at most.

    TWO PEOPLE CAN FEEL DIFFERENT THINGS IN ONE SHOT. "Dan is furious and McKenna is
    terrified" gave only the first of them, so one face was performing and the other
    was left to the prior -- the same half-fix as naming one of two speakers. Two at
    most, like the layering clause: a shot carrying four feelings has stopped being
    about its beat. Only what the beat acts out, never negated, never ownerless -- see
    _negated."""
    text = engine.acted_text(str(beat or ""))
    out, seen = [], set()
    for m in _EMOTION.finditer(text):
        if _negated(text, m.start()):
            continue
        word = m.group(0).lower()
        who = emotion_owner(text, names, word, sheet)
        if who and who not in seen:
            seen.add(who)
            out.append((who, word))
        if len(out) >= 2:
            break
    return out


def mood_faces(pairs):
    """Say whose feeling is whose, for one or two people. "" for none."""
    ps = [(w, e) for w, e in (pairs or []) if w and e]
    return "".join(mood_face(e, w) for w, e in ps[:2])


def mood_face(word, who=""):
    """Say the face plays the feeling the author named. "" when they named none.

    Their word, not a synonym: "terrified" and "grim" are not the same performance.
    NAMED: said impersonally in a two-hander, the captor wore his victim's terror.
    SHORT: "the face carries it ... played in the eyes and the mouth together" asked for
    a performance on top of the author's word. REPORTED as characters doing things the
    beat never wrote. The word, on its owner, is the whole of it."""
    if not word:
        return ""
    if who:
        return f" {who}'s expression is {word}."
    return f" The expression is {word}."


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
    # Either part of a two-part name: "Maya" is "Maya Brooks" (see _name_forms). It now
    # decides whether a shot describes anybody at all -- see _plain_people.
    return any(n and re.search(r"\b" + re.escape(f) + r"\b", b, re.I)
               for n, _ in sheet_lines(sheet) for f in _name_forms(n))

FORM_HOLD = ", the same object in the same material."
OTHERS_UNCHANGED = " Everyone else in the shot has on exactly what their own entry lists."

# CLOSED BY MID-SHOT, HANDS CLEAR AT THE END. "Closed on it by the last" set the
# deadline on the frame the next shot is pinned to, and a shot that ends mid-motion
# hands on open cuffs still in somebody's hands. Reported as restraints breaking in the
# beat after they go on. The closing comes first; the rest of the shot holds it.
RESTRAINT_GOING_ON = (" The hardware goes on during this shot: it is open and off the "
                      "body at the first frame, closed on the body by the middle of the "
                      "shot and closed for the rest of it, and by the last frame it is "
                      "on the body in plain view and the hands that closed it have let "
                      "go and are clear of it.")


def newly_on_clause(items, where=None, who="", posed=False):
    """RESTRAINT_GOING_ON for a piece added to somebody ALREADY restrained -- the rope
    on the ankles of a woman who has been cuffed for three shots. Said per piece and
    where it goes, because the rest of what she wears is NOT going on now, and "the
    hardware goes on during this shot" would take the cuffs off at the first frame.

    Once on, on: the beat usually does something after it -- "ties her ankles, then
    drags her to the chair" -- and the end of the shot is where it went missing. So it
    is fastened by the middle of the shot and held for the rest, and the last frame,
    which the next shot opens on, shows it in plain view with the hands clear of it."""
    out = ""
    for item in [str(i).strip() for i in (items or []) if str(i).strip()]:
        plural = item.endswith("s") and not item.endswith("ss")
        at, _sep, fast = _where_of(item, where, who).rstrip(",").partition(", fast at the ")
        _it = "them" if plural else "it"
        out += (f" The {item} {'go' if plural else 'goes'}"
                + (f" {at}" if at else " on")
                + " during this shot"
                + (f", fastened to the {fast}" if fast else "")
                + ": off the body at the first frame, on and fastened by the middle of "
                  "the shot"
                # ...and holding its shape from the moment it closes, in the words for
                # what it IS -- see rigid_tail. A chain going on is still a chain, and
                # one that forces a position is drawn to its length once it closes.
                + (f" and drawn to {'their' if plural else 'its'} full length, so the "
                   f"position {'they fix' if plural else 'it fixes'} is the position "
                   f"that keeps"
                   if (posed and rigid_hardware(item) and not (
                       _CUFF_FORM.search(item) and not re.search(r"\bchain", item, re.I)))
                   else rigid_tail(item, "wrists", plural, where=where).rstrip(",")
                   if rigid_hardware(item) else "")
                + f"; {'they stay' if plural else 'it stays'} fastened for the rest of "
                  f"the shot, and at the last frame {'they are' if plural else 'it is'} "
                  f"on the body in plain view and the hands that fastened {_it} have let "
                  f"go and are clear of {_it}.")
    return out
# WHERE THE HARDWARE CLOSES, not only where the limbs end up. RESTRAINT_GOING_ON
# gives the hardware both of its ends -- open and off at the first frame, closed by
# mid-shot -- and this used to give the limbs only their last one. Between those two
# facts nothing said where the closing HAPPENS, and a video model asked to go from no
# cuffs to cuffs does the likeliest thing in front of the body and leaves them there.
# Reported as wrists cuffed in front on the shot that applies them, while every shot
# after it holds them correctly behind: the later shots read the standing pose, and
# the applying shot was the one with a gap in it.
#
# The second sentence is the original and is left word for word, because the shot
# after this one inherits the last frame and that is the sentence that pins it.
# Positively phrased, like everything else here -- at cfg 1 naming where they are NOT
# is naming it. The closing is placed in the first half, as RESTRAINT_GOING_ON places
# it, so the last frame is the hold and not the act.
RESTRAINT_ENDS_AT = (" The {part} are already {where} when the hardware closes in the "
                     "first half of the shot, and it closes on them there. By the last "
                     "frame the {part} are {where}, and stay there.")
# WHAT A PAIR OF CUFFS IS, as opposed to what a chain is.
#
# _RIGID_HARDWARE puts handcuffs, manacles, shackles and irons in the same bucket as
# chains and padlocks, which is right about the one thing it was asked -- none of them
# flex. The SENTENCE built from it was written for a chain and says so: "its links keep
# their size and the run between them stays taut". Links, a run between them, taut.
# Handed that about a pair of handcuffs, with nothing anywhere saying what handcuffs
# look like, the model draws the thing the words describe. Reported as cuffs turning
# into chains in the shot that uses them.
#
# A chain's rigidity is about its LENGTH holding. A cuff's is about two closed rings a
# fixed distance apart. Same guarantee, and it cannot be said in the same words.
_CUFF_FORM = re.compile(
    r"\b(?:handcuffs?|cuffs?|cuffed|manacles?|shackles?|"
    r"(?:leg|ankle|wrist)\s*irons?|irons)\b", re.I)
_ONE_OF = {"wrists": "wrist", "ankles": "ankle", "hands": "hand", "legs": "leg",
           "arms": "arm", "thumbs": "thumb", "toes": "toe"}


def cuff_part(items, where=None):
    """The part the CUFFS among these items close round, or "" if none are cuffs.

    Not the part of the hardware as a whole. held_part answers for the first thing it
    recognises, and beside a collar that is the neck -- so steel handcuffs and a leather
    collar were written as "a closed ring locked round each neck", and the arms as "the
    neck are behind the back". REPORTED as bondage equipment not looking the same, or
    not being where it was, from one shot to the next. The state's own record of where
    each piece is fastened wins; the item's name is the fallback."""
    pieces = [p.strip() for it in (items or [])
              for p in re.split(r",|\s+and\s+", str(it or "")) if p.strip()]
    cuffs = [p for p in pieces
             if _CUFF_FORM.search(p) and not re.search(r"\bchain", p, re.I)]
    for c in cuffs:
        at = (where or {}).get(c)
        if at is None and where:
            head = c.split()[-1].lower()
            at = next((v for k, v in where.items()
                       if k.split() and k.split()[-1].lower() == head), None)
        if at and at[0][0]:
            return at[0][0]
    return held_part(cuffs) if cuffs else ""


def rigid_tail(item, part="wrists", plural=False, where=None):
    """The clause that says HOW a rigid restraint holds its shape.

    Cuffs get what cuffs are -- a closed ring round each limb, the pair held a fixed
    distance apart. Everything else keeps the chain wording, which is correct for a
    chain and was only ever wrong when it was handed to something that is not one.

    The distance is given as a hand's width because a length that is not stated is a
    length the model picks, and the one it picks for metal between two wrists is a
    chain's."""
    part = cuff_part([item], where) or part
    one = _ONE_OF.get(str(part or "wrists").lower(), str(part or "wrist").rstrip("s"))
    if _CUFF_FORM.search(str(item or "")) and not re.search(r"\bchain", str(item or ""), re.I):
        return (f", a closed ring locked round each {one} and the two held a hand's "
                f"width apart, that spacing keeping")
    return (f", {'their' if plural else 'its'} links keeping their size and the run "
            f"between them taut")


def cuff_rigid_sentence(part="wrists"):
    """The standalone form of the cuff shape, for the shot that puts it ON.

    Takes the part from the hardware rather than assuming wrists -- leg irons close
    round ankles, and a sentence saying wrists about a pair of leg irons is the same
    class of error as calling cuffs a chain."""
    one = _ONE_OF.get(str(part or "wrists").lower(), str(part or "wrist").rstrip("s"))
    return (f" It is a closed ring locked round each {one}, the two held a hand's "
            f"width apart, and that spacing keeps.")


CUFF_RIGID_TAIL = cuff_rigid_sentence("wrists")

CHAIN_RIGID_TAIL = " Its links keep their size and the run between them stays taut."
_APPLY_NOW = re.compile(
    _A_DETERMINER +
    # "Sits on the bench IN handcuffs", "WITH chains on her ankles": a noun after
    # in/with is what she is wearing, and it read as the verb -- so a woman already
    # cuffed was told the cuffs go on during the shot. REPORTED as restraints staged
    # going on when the beat says they are on. "Puts her in handcuffs" is read below.
    r"(?<!\bin\s)(?<!\bwith\s)"
    r"\b(?:cuffs|handcuffs|chains|ties|binds|locks|straps|tapes|gags|shackles|"
    r"fastens|secures|padlocks|buckles|clamps|clips|snaps|trusses|lashes|wraps|"
    r"restrains|immobili[sz]es|pinions|fetters|collars|hobbles|"
    r"hog-?ties|straitjackets|manacles|blindfolds|leashes|"
    r"cinches|tightens)\b", re.I)
_APPLY_PHRASE = re.compile(
    r"\b(?:put|puts|putting|pull|pulls|pulling|force|forces|forcing|get|gets|"
    r"getting|work|works|snap|snaps)\s+(?:[\w,']+\s+){0,4}?"
    r"(?:on|onto|around|behind|together|shut|closed)\b"
    # Tape goes on by being pressed, stuck or slapped OVER something -- the engine's
    # APPLY_VERB, mirrored here in the -s forms.
    r"|\b(?:puts|places|presses|sticks|slaps|smooths|plasters|applies)\s+"
    r"(?:[\w,'’]+\s+){0,5}?"
    r"(?:over|across|to)\s+(?:[\w'’]+\s+){0,2}?(?:mouth|lips|eyes|face)\b"
    r"|\bcovers\s+(?:her|his|their|the|[\w'’]+['’]s)\s+(?:mouth|lips|eyes|face)\s+with\b"
    r"|\b(?:loops?|wraps?|winds?|coils?|threads?|passes|runs|cinch(?:es)?|knots?|"
    r"laces?|hitch(?:es)?|slings?)\s+(?:[\w,']+\s+){0,5}?"
    r"(?:around|round|through|under|over|behind|between)\b"
    # ...and INTO them: "puts her in handcuffs", "locks Ana in chains".
    r"|\b(?:puts|places|locks|clamps|gets|forces|clicks|snaps|shoves)\s+"
    r"(?:[\w'’]+\s+){0,2}?in(?:to)?\s+(?:(?:a|the|some|her|his|their|steel|metal|"
    r"leather|heavy|plastic|iron)\s+){0,2}(?:(?:hand)?cuffs|chains|shackles|manacles|"
    r"irons|restraints|straitjacket)\b"
    # ...and the piece itself slid, clicked or hooked ON, which the engine already read
    # as going on: "slips the cuffs onto her wrists", "hooks a leash to her collar".
    r"|\b(?:slips|slides|clicks|fits|presses|hooks|attaches|clips|fastens)\s+"
    r"(?:[\w'’-]+\s+){0,3}?(?:" + "|".join(p for p, _n, _pt in engine.HARDWARE) + r")"
    r"\s+(?:[\w'’]+\s+){0,1}?(?:on|onto|shut|to|around|round)\b", re.I)


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
    # ...but not "pulls off A STRIP OF duct tape": that is tape being got ready, and it
    # took her handcuffs off -- the sheet's only restraint, matched to the "it" the
    # tape went on as. REPORTED as cuffs breaking the beat the tape went on.
    r"(?!\s+(?:a|an|another|some|one|two|the)\s+(?:\w+\s+)?"
    r"(?:strip|piece|length|bit|section|square|tear)s?\s+of\b)"
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
    b = _worn_masked(beat or "")
    return bool(_APPLY_NOW.search(b) or _APPLY_PHRASE.search(b))


# WHAT SHE IS ALREADY WEARING, named in passing: "in handcuffs", "with rope around her
# wrists", "with her wrists cuffed". Read as hardware going on, a woman the beat
# describes as tied to a chair was told the rope is "off the body at the first frame"
# and goes on during the shot, with nobody there to tie it. REPORTED as restraints
# staged going on when the beat says they are already on. Not where a verb in the same
# clause puts them on: "puts her IN handcuffs" is a cuffing, and "ties her wrists to
# the bedpost WITH rope" names what she is tied with.
_WORN_PHRASE = re.compile(
    r"\b(?:in|with|wearing|wears|wore)\s+"
    r"(?:(?!(?:and|then|but|while|as|to|she|he|they|it|one)\b)[\w'’-]+\s+){0,3}?"
    r"(?:" + "|".join(p for p, _n, _pt in engine.HARDWARE) + r")\b"
    r"(?:\s+(?:\w+\s+)?(?:around|round|on|over|across|behind|between|at|through|"
    r"about)\s+(?:her|his|their|the|[A-Z][\w'’-]*['’]s)(?:\s+\w+){1,2})?", re.I)
_PUTS_INTO = re.compile(
    r"\b(?:put|puts|putting|places|placing|locks|locking|clamps|clamping|gets|getting|"
    r"forces|forcing|shoves|clicks|snaps)\s+(?:[\w'’]+\s+){0,2}$", re.I)
_CLAUSE_CUT = re.compile(r"[,;:.!?]|\b(?:while|as|but|then|before|after|until)\b", re.I)
# A BINDING VERB IN ANY TENSE. The -s list above answers "is this the shot it closes"
# for the first restraint of a run; this one is wider, because "binding them with
# rope", "attaches her wrists to the headboard with handcuffs" and "cuffed her" are
# all somebody putting it on. REPORTED as the person being restrained dropped from the
# shot, and the shot that cuffs her told the cuffs were on from its first frame.
_BIND_ANY = re.compile(
    r"\b(?:(?:hand)?cuff(?:s|ed|ing)?|zip-?ti(?:es|ed|eing)|ti(?:es|ed|e)|tying|"
    r"bind(?:s|ing)?|bound|tap(?:es|ed|ing)|gag(?:s|ged|ging)?|blindfold(?:s|ed|ing)?|"
    r"chain(?:s|ed|ing)?|shackl(?:es|ed|ing|e)|strap(?:s|ped|ping)?|lash(?:es|ed|ing)?|"
    r"truss(?:es|ed|ing)?|secur(?:es|ed|ing|e)|fasten(?:s|ed|ing)?|"
    r"restrain(?:s|ed|ing)?|tether(?:s|ed|ing)?|anchor(?:s|ed|ing)?|"
    r"attach(?:es|ed|ing)?|bolt(?:s|ed|ing)?|pin(?:s|ned|ning)?|hitch(?:es|ed|ing)?|"
    r"fix(?:es|ed|ing)?|link(?:s|ed|ing)?|connect(?:s|ed|ing)?|latch(?:es|ed|ing)?|"
    r"moor(?:s|ed|ing)?|confin(?:es|ed|ing|e)|lock(?:s|ed|ing)?|clip(?:s|ped|ping)?|"
    r"clamp(?:s|ed|ing)?|padlock(?:s|ed|ing)?|manacl(?:es|ed|ing|e)|"
    r"collar(?:s|ed|ing)?|leash(?:es|ed|ing)?|pinion(?:s|ed|ing)?|fetter(?:s|ed|ing)?|"
    r"hobbl(?:es|ed|ing|e)|buckl(?:es|ed|ing|e)|cinch(?:es|ed|ing)?|"
    r"immobili[sz](?:es|ed|ing|e)|muzzl(?:es|ed|ing|e)|ratchet(?:s|ed|ing)?)\b", re.I)
# ...and the forms of it that describe her rather than somebody's deed: after a be-verb
# or a posture ("is handcuffed", "sits tied"), hung off a comma ("Mara, cuffed, falls",
# "kneels, wrists cuffed"), after a body part ("her wrists cuffed to the headboard")
# or in front of a noun ("her cuffed wrists"). Not "gets cuffed", "has her wrists
# tied", "is cuffed by Dan" or "cuffed her": those are the act.
_STATE_BEFORE = re.compile(
    r"(?:\b(?:is|are|was|were|been|be|sits?|sat|sitting|lies|lay|lying|kneels?|knelt|"
    r"kneeling|stands?|stood|standing|hangs?|hung|hanging|stays?|stayed|remains?|"
    r"remained|waits?|waited|rests?|rested|slumps?|slumped|sprawls?|sprawled|"
    r"crouches|crouched|leans?|leaned|perches?|perched|wakes?|woke|left|found|seen)"
    r"(?:\s+[\w'’-]+){0,3}?"
    r"|,\s*(?:(?:already|still|now|\w+ly)\s+)?(?:(?:her|his|their|the)\s+)?"
    r"(?:(?:wrists?|ankles?|hands?|arms?|legs?|feet|mouth|lips|eyes|neck|knees?)\s+)?"
    r"|\b(?:her|his|their|the|[A-Z][\w'’-]*['’]s)\s+"
    r"(?:wrists?|ankles?|hands?|arms?|legs?|feet|mouth|lips|eyes|neck|knees?)"
    r"|\b(?:her|his|their|the|a|an|[A-Z][\w'’-]*['’]s)"
    r"|\b(?:already|still))\s*$", re.I)
_STATE_NOT = re.compile(
    r"\b(?:gets?|got|getting|has|have|had|having|being)\b(?:\s+[\w'’-]+){0,3}\s*$", re.I)
_DEED_AFTER = re.compile(
    r"\s+(?:her|him|them|it|the|a|an|some|by)\b|\s+(?-i:[A-Z][\w'’-]+)(?!['’]s)\b", re.I)
_PARTICIPLE_FORM = re.compile(r"(?:ed|bound|tied)$", re.I)
# The piece named, not a verb: "the cuffs", "in handcuffs", "a chain".
_NOUN_BEFORE = re.compile(
    r"\b(?:the|a|an|her|his|their|its|some|steel|metal|leather|heavy|iron|plastic|of|in|"
    r"with|those|these|[A-Z][\w-]*['’]s)\s+$", re.I)
# Which binding verbs speak for which piece: "cuffed" for the cuffs, "taped" for the
# tape. A rope, a cord or a zip tie is tied, bound or secured, so those -- and the
# piece no stem names -- answer to the generic verbs.
_ITEM_STEM = (("cuff", r"(?:hand)?cuff|ratchet"), ("tape", r"tap"), ("gag", r"gag|muzzl"),
              ("blindfold", r"blindfold"), ("chain", r"chain|padlock"),
              ("shackle", r"shackl"), ("collar", r"collar"), ("leash", r"leash"),
              ("manacle", r"manacl"), ("strap", r"strap|buckl"), ("hobble", r"hobbl"),
              ("tether", r"tether"))
_ACTS_ON = re.compile(
    r"\b(?!(?:watches|sees|finds|leaves|keeps|eyes|studies|notices|spots|regards|"
    r"faces|observes|joins|has|is|was|does|wears|sits|lies|stands)\b)"
    r"[a-z]+(?:s|ed)\s+(?:her|him|them|(?-i:[A-Z][\w'’-]+))\b", re.I)


def _state_form(text, m):
    """Does the binding verb matched at `m` describe a state rather than an act? See
    _STATE_BEFORE."""
    word = m.group(0)
    if not _PARTICIPLE_FORM.search(word):
        return False
    lead = text[:m.start()]
    cuts = list(_CLAUSE_CUT.finditer(lead))
    # Keep a comma itself: "Mara, cuffed," is hung off it.
    clause = lead[cuts[-1].start():] if cuts else lead
    if _STATE_NOT.search(clause):
        return False
    if _DEED_AFTER.match(text, m.end()):
        return False
    return bool(_STATE_BEFORE.search(clause))


def _deed_in(clause):
    """Does this clause hold a binding verb used as an act, in any tense?"""
    return any(not _state_form(clause, m) for m in _BIND_ANY.finditer(clause))


def _worn_spans(beat):
    """[(start, end)] of every phrase in `beat` naming hardware as already worn."""
    b = beat or ""
    out = []
    for m in _WORN_PHRASE.finditer(b):
        lead = m.group(0).split()[0].lower()
        clause = _CLAUSE_CUT.split(b[:m.start()])[-1]
        if lead == "in" and _PUTS_INTO.search(clause):
            continue
        # WITH WHAT SHE IS TIED, not what she wears: a binding verb in the clause, in
        # any tense ("binding them with rope", "cuffing her wrists with steel
        # handcuffs"), or any verb acting on her ("attaches her wrists to the
        # headboard with handcuffs"). Only a clause that does neither -- "sits with
        # tape over her mouth", "gets up with rope around her wrists" -- is worn.
        if lead == "with" and (_APPLY_NOW.search(clause + "with")
                               or _APPLY_PHRASE.search(clause + "with")
                               or _deed_in(clause) or _ACTS_ON.search(clause)):
            continue
        out.append((m.start(), m.end()))
    return out


def _worn_masked(beat):
    """`beat` with its already-worn hardware phrases blanked out. See _WORN_PHRASE."""
    b = beat or ""
    for lo, hi in reversed(_worn_spans(b)):
        b = b[:lo] + " " * (hi - lo) + b[hi:]
    return b


def staged_on_now(beat, item=""):
    """For a piece the state recorded as applied in this beat: is it put on now, rather
    than described as already on?

    A first mention is not a fastening. "Jade sits tied to a chair with rope around
    her wrists", "her wrists cuffed to the headboard", "Mara is handcuffed to the
    radiator": each puts a restraint in the state for the first time, and each was
    told it goes on during the shot. REPORTED. A piece the beat names as worn ("with
    tape over her mouth while he ties her ankles") is not the one going on.

    Everything else the state recorded IS going on. This used to require the -s verbs
    of restraint_going_on as well, so "binding them with rope", "ratchets the cuffs onto
    her wrists" and "cuffed her" were read as already on: the person being restrained
    dropped out of her own shot and the cuffing shot said the cuffs were on from the
    first frame. REPORTED. So the test is only for the worn wordings: a worn phrase
    (see _worn_spans), or the piece's verb in a state form (see _STATE_BEFORE)."""
    b = beat or ""
    if item:
        canon = {c for c, _p, _w, _a in engine.hardware_spans(str(item))}
        head = str(item).split()[-1].lower().rstrip("s") if str(item).split() else ""
        for lo, hi in _worn_spans(b):
            _in = engine.hardware_spans(b[lo:hi])
            if any(c in canon for c, _p, _w, _a in _in) or (
                    head and any(str(w or c).split()[-1].lower().rstrip("s") == head
                                 for c, _p, w, _a in _in)):
                return False
    verbs = [m for m in _BIND_ANY.finditer(b)
             if _PARTICIPLE_FORM.search(m.group(0)) or not _NOUN_BEFORE.search(b[:m.start()])]
    stems = [st for key, st in _ITEM_STEM if key in str(item).lower()]
    mine = [m for m in verbs if stems and re.match(stems[0], m.group(0), re.I)]
    if not mine:
        # No verb of its own: the generic ones (tied, bound, secured), not another
        # piece's -- "Mara, already cuffed, watches Dan tie her ankles" is new rope.
        mine = [m for m in verbs
                if not any(re.match(st, m.group(0), re.I) for _k, st in _ITEM_STEM)]
    return not (mine and all(_state_form(b, m) for m in mine))


# Keep this compact: it is repeated in every shot while restraints remain present.
RESTRAINT_HOLD = (" Every restraint stays closed and fastened as it was put on") + FORM_HOLD


def restraint_wearers(sheet):
    """The people whose own sheet entry describes hardware.

    Read from the entries rather than the beat, because the entry is what says who is
    WEARING it -- a beat can mention a chain without anyone being in it."""
    return [n for n, ln in sheet_lines(sheet) if n and restraint_present(ln)]


_WORN_IN = re.compile(
    r"\b(?:in|wearing|wears|wore)\s+(?:(?:a|an|the|some|her|his|their|pair\s+of|set\s+of)\s+)*"
    r"(?:[\w-]+\s+){0,2}?(?:" + "|".join(p for p, _n, _pt in engine.HARDWARE) + r")\b", re.I)


def scene_restraints(text, cast, pronouns=None):
    """[(wearer, sentence)] for the scene paragraph's sentences that put a piece ON
    somebody -- "Mara is handcuffed to the rail", "McKenna lies in the back, wrists
    cuffed behind her back". The state takes these as it takes a sheet entry, so the
    hold is armed by a piece with a wearer; a piece lying on a table, carried, or in
    somebody's hand is not one. Sheet lines are skipped: the sheet is declared itself.
    A wearer the sentence does not settle is "" -- see the latch."""
    people = [n for n in (cast or []) if n]
    head = (re.compile(r"\s*(?:" + "|".join(re.escape(n) for n in people) + r")\s*:")
            if people else None)
    out = []
    for line in str(text or "").split("\n"):
        if head is not None and head.match(line):
            continue
        for s in re.split(r"(?<=[.!?])\s+", line.strip()):
            if not s or not engine.hardware_spans(s) or not restraint_present(s):
                continue
            if not (engine.applies_hardware(s) or _WORN_IN.search(s)):
                continue
            named = engine.names_in(s, people)
            _pr = re.search(r"\b(she|her|he|him|his)\b", s, re.I)
            _group = ({"her": "she", "him": "he", "his": "he"}.get(_pr.group(1).lower(),
                                                                  _pr.group(1).lower())
                      if _pr else "")
            _fits = [n for n in people if _group and (pronouns or {}).get(n) == _group]
            if len(named) > 1:
                who = engine.wearer_of(s, people)
            elif named:
                # "Dan stands over her, her wrists cuffed to the rail" is hers.
                who = (_fits[0] if (len(_fits) == 1 and _fits[0] != named[0]
                                    and (pronouns or {}).get(named[0]))
                       else named[0])
            else:
                who = (_fits[0] if len(_fits) == 1
                       else people[0] if len(people) == 1 else "")
            out.append((who, s))
    return out


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


# ONE PERSON THE SHEET DOES NOT NAME: a waiter, an old man, a stranger, her mother, an
# officer, children. Only crowds counted, so "A waiter brings the bill" was told there
# is one person in the shot -- the count forbidding, as a positive fact, the waiter the
# author just asked for. A plain man or woman needs "a"/"an": "the man" is as often
# somebody the sheet does name. Not somebody only talked about, called or texted.
_EXTRA_ROLE = (r"waiters?|waitress(?:es)?|bartenders?|barmen|barman|barmaid|clerks?|"
               r"cashiers?|receptionists?|strangers?|officers?|policem[ae]n|police\s+officers?|"
               r"cops?|guards?|soldiers?|doctors?|nurses?|drivers?|cabbies|cabbie|"
               r"neighbou?rs?|shopkeepers?|landlord|landlady|priests?|vendors?|porters?|"
               r"butlers?|maids?|mother|father|mum|mom|dad|brother|sister|grandmother|"
               r"grandfather|husband|wife|boyfriend|girlfriend|children|kids|child|"
               r"toddler|baby|teenagers?|passengers?|security\s+guards?|bouncers?")
_EXTRA_ONE = re.compile(
    r"(?<!\babout\s)(?<!\bof\s)(?<!\bcalls\s)(?<!\btexts\s)(?<!\bphones\s)"
    r"\b(?:(?:a|an|another|the|her|his|their|some|two|three|several)\s+"
    r"(?:(?:old|young|elderly|tall|short|older|younger|little|small|uniformed|masked|"
    r"bearded|grey-haired|middle-aged|fat|thin|big|large)\s+){0,2}"
    r"(?:" + _EXTRA_ROLE + r")"
    r"|(?:a|an|another)\s+(?:(?:old|young|elderly|tall|short|older|younger|little|"
    r"uniformed|masked|bearded|grey-haired|middle-aged)\s+){0,2}"
    r"(?:man|woman|boy|girl|guy|lady|gentleman|person|figure|stranger)"
    r"|children|kids)\b(?!['’]s\b)", re.I)


# WHERE ONE ROLE NOUN IS A PERSON IN THE FRAME. As a modifier ("the passenger seat",
# "the guard rail", "the baby monitor", "the driver door") it is a thing, and beside a
# name ("her husband Dan", "Mara, a nurse,", "the guard Dan") it is somebody the sheet
# already counts; "like a stranger" is nobody at all. Each took the two-person count off
# a two-person shot. REPORTED.
_ROLE_NEXT_OK = re.compile(
    r"\s*(?:[.,;:!?)\"”]|$)|\s+(?:[a-z]+(?:s|ed)|is|was|are|were|has|had|will|can|"
    r"who|that|and|or|with|in|on|at|by|from|to|into|onto|behind|beside|near|across|"
    r"through|outside|inside|over|under|of|for|as|then|while|stands?|sits?|waits?|"
    r"comes?|came|runs?|ran|walks?|looks?|takes?|took|gives?|gave|holds?|held|"
    r"steps?|leans?|nods?|says?|said|asks?|tells?|calls?|shouts?)\b")


def _heads_its_phrase(text, m):
    """Does the role noun matched at `m` head its own noun phrase, as a person? See
    _ROLE_NEXT_OK."""
    after = text[m.end():]
    if not _ROLE_NEXT_OK.match(after) or re.match(r"\s+(?-i:[A-Z][\w'’-]+)", after):
        return False
    before = text[:m.start()]
    if re.search(r"\b(?:like|as)\s+$", before, re.I):
        return False
    # "Mara, a nurse," -- an appositive after a name.
    if re.search(r"(?-i:[A-Z][\w'’-]+)\s*,\s*$", before):
        return False
    return True


def extras_in(beat, singular=True):
    """Does this beat stage people beyond the ones the sheet names, IN the frame?

    `singular` counts one unnamed person too -- see _EXTRA_ONE. The film-long latch
    reads crowds only, so one waiter does not take the body count off every shot
    after the one he serves in."""
    b = str(beat or "")
    hits = list(_EXTRA_PEOPLE.finditer(b))
    if singular and not hits:
        _st = engine.staged_text(b)
        if any(_heads_its_phrase(_st, m) for m in _EXTRA_ONE.finditer(_st)):
            return not (_NOT_STAGED.search(b) or _REMOTE.search(b))
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
    r"strok(?:es|ed|ing)?|cuddl(?:e|es|ed|ing)|"
    r"grab(?:s|bed|bing)?|touch(?:es|ed|ing)?|"
    r"danc(?:e|es|ed|ing)\s+with|lean(?:s|ed|ing)?\s+(?:on|against|into)|"
    r"press(?:es|ed|ing)?\s+(?:against|into)|"
    r"wraps?\s+(?:her|his|their)\s+arms?\s+around")
# HANDLING VERBS ONLY WITH THE PERSON RIGHT AFTER THEM. "Dan takes Mara's coat", "pulls
# the chair out for Mara", "reaches for the phone Mara left" each paired two bodies in
# contact, from a verb whose object was a thing. REPORTED as characters doing things the
# beat never wrote. "Takes Mara to the car", "pulls Mara close" still are.
_HANDLING_CONTACT_SRC = (
    r"hold(?:s|ing)?|held|caught|catch(?:es|ing)?|pull(?:s|ed|ing)?|take[sn]?|took|"
    r"taking|push(?:es|ed|ing)?|sit(?:s|ting)?\s+on|sat\s+on|reach(?:es|ed|ing)?\s+for|"
    r"undress(?:es|ed|ing)?")
# ...and the person is the object itself, ending the clause or followed by a
# preposition or a particle -- never a possessive.
_CONTACT_OBJ_END = (r"(?!['’]s\b)(?=\s*(?:[.,;:!?]|$)|\s+(?:to|with|on|onto|in|into|"
                    r"against|close|closer|tight|tighter|tightly|back|up|down|over|"
                    r"around|round|from|by|across|toward|towards|for|at|under|through|"
                    r"again|away|off|near|beside|gently|softly|hard|deeply|slowly|"
                    r"tenderly|passionately|and|\w+ly)\b)")
# ...a contact verb's person, whole or by a part of the body.
_CONTACT_BODY_PART = (r"['’]s\s+(?:\w+\s+)?(?:neck|nape|hair|wrists?|hands?|face|"
                      r"cheeks?|lips|mouth|shoulders?|arms?|back|waist|hips?|thighs?|legs?|"
                      r"knees?|feet|foot|ankles?|chest|breasts?|stomach|belly|forehead|jaw|"
                      r"chin|ears?|throat|fingers?|palms?|skin|body|head|side|temple|brow)\b")
_CONTACT_PART_END = r"(?:" + _CONTACT_BODY_PART + r"|(?!['’]s\b))"
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
            # A contact verb takes the person or a part of them ("kisses Mara's neck",
            # "kisses Mara hungrily"); only a HANDLING verb needs the person as its
            # whole object. Applied to both, the handling test unpaired every kiss on a
            # neck and every adverb off its list. REPORTED.
            for src, obj_end in ((_CONTACT_SRC, _CONTACT_PART_END),
                                 (_HANDLING_CONTACT_SRC,
                                  r"(?:" + _CONTACT_OBJ_END + r"|" + _CONTACT_BODY_PART
                                  + r")")):
                m = re.search(r"\b" + re.escape(a) + r"\b\s+(?:\w+\s+){0,2}?(?:"
                              + src + r")\b", part, re.I)
                if not m:
                    continue
                tail = part[m.end():]
                lead = (r"\s+(?:the\s+\w+\s+of\s+)?" if src is _CONTACT_SRC else r"\s+")
                hit = next((c for c in people if c != a and re.match(
                    lead + re.escape(c) + r"\b" + obj_end, tail, re.I)), None)
                if hit:
                    if not any({a, hit} == set(p) for p in out):
                        out.append((a, hit))
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
    # Who is with whom, and no more: "those two bodies together" asked for more contact
    # than the beat's own verb.
    if len(ps) == 1:
        return f" The contact is {ps[0][0]} with {ps[0][1]}."
    return f" The contact is {ps[0][0]} with {ps[0][1]}, and {ps[1][0]} with {ps[1][1]}."


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


def lora_patch_mismatches(patcher):
    """[(family, count, produced, target)] for LoRA patches this model cannot take.

    A LORA THAT DOES NOT FIT IS NOT REFUSED. comfy applies each pair as
    (B @ A).reshape(weight.shape) inside a bare try/except that logs one ERROR line
    and hands the weight back untouched (comfy/weight_adapter/lora.py). The keys that
    DO fit are applied anyway, so a LoRA built for a different variant of the same
    model half-loads: attention and MLP adapted, and whatever did not fit missing.

    WHICH IS THE WORST SHAPE THIS FAILURE COULD TAKE, because what does not fit is
    usually adaln_proj -- the per-block timestep modulation. H3 ships in variants
    whose AdaLN input differs (2688 on the full fl2va, 8 on the pruned and on the
    hybrid this node recommends), while attention and MLP are identical between them.
    So a LoRA trained on the full model drops exactly its 51 AdaLN pairs onto a hybrid
    and keeps all 208 of the rest: a distilled few-step trajectory applied to the
    attention stack with the modulation that was meant to go with it missing. The
    picture still renders. What it renders is anatomy that does not resolve -- a third
    leg, a limb that starts and stops -- on some LoRAs and not others, which is what
    makes it so hard to attribute.

    Reported per FAMILY rather than per key: 51 lines saying the same thing about 51
    blocks is not a report anybody reads."""
    patches = getattr(patcher, "patches", None)
    if not isinstance(patches, dict) or not patches:
        return []
    try:
        sd = patcher.model_state_dict()
    except Exception:
        return []
    seen = {}
    for key, entries in patches.items():
        target = getattr(sd.get(key), "shape", None)
        if target is None:
            continue
        want = 1
        for d in target:
            want *= int(d)
        for entry in entries if isinstance(entries, (list, tuple)) else ():
            weights = getattr(entry[1] if len(entry) > 1 else None, "weights", None)
            if not weights or len(weights) < 2:
                continue
            up, down = getattr(weights[0], "shape", None), getattr(weights[1], "shape", None)
            if not up or not down or len(up) < 1 or len(down) < 2:
                continue
            got = int(up[0]) * int(down[1])
            if got == want:
                continue
            fam = re.sub(r"\.\d+\.", ".N.", str(key)).rsplit(".weight", 1)[0]
            row = seen.setdefault(fam, [0, (int(up[0]), int(down[1])), tuple(int(d) for d in target)])
            row[0] += 1
    return [(fam, n, produced, target) for fam, (n, produced, target) in seen.items()]




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


# Hardware that passes BETWEEN THE LEGS, which is a place on the body the rest of
# this file has no name for. held_part() reads the hardware's noun and answers with a
# limb -- neck, ankles, waist, body, and wrists for everything it does not recognise --
# so duct tape wound round the waist and through the crotch came back "wrists", the
# same as a pair of handcuffs, and the hold clause it produced said the tape stayed
# "tied and holding as it was put on" without ever saying WHERE.
#
# Meanwhile the bare clause went on naming the groin uncovered, because it suppresses
# only for a GARMENT the sheet lists and tape is hardware. So the shot said: tape
# exists somewhere, and the genitals are bare and in plain view. Both sentences were
# true to the node and only one of them could be drawn.
# Names the region outright, so it needs no second half: a chastity belt and a crotch
# strap are not ambiguous about where they sit.
_CROTCH_NAMED = re.compile(
    r"\bchastity\s+(?:belt|device)s?\b"
    r"|\b(?:crotch|groin)\s*(?:strap|rope|chain|cord|band|piece|panel)s?\b"
    r"|\bthrough\s+(?:the\s+)?(?:crotch|groin)\b", re.I)
# "between the legs" is a position for a hand, a knee, a bag or a camera far more
# often than it is hardware, so on its own it fires nothing -- the waist half has to
# be there too. A possessive NAME counts: "around Mara's waist" is the common wording.
_BETWEEN_LEGS = re.compile(
    r"\bbetween\s+(?:her|his|their|the|\w+'s)\s+legs\b", re.I)
_ROUND_WAIST = re.compile(
    r"\b(?:round|around|about)\s+(?:her|his|their|the|\w+'s)\s+"
    r"(?:waist|hips|middle|belly)\b"
    r"|\bwaist\s*(?:band|belt|chain|strap|rope)s?\b", re.I)


def crotch_seal(text, items=()):
    """The hardware this text closes over the groin, as written. "" when none.

    TWO HALVES REQUIRED, except for a chastity belt, which is one word for both. A
    beat has to put the thing AROUND the waist and BETWEEN the legs before this
    fires, because "between her legs" on its own is a position for a hand, a knee or
    a camera far more often than it is a strap, and a false positive here seals a
    region the author left open.

    Returns the hardware's own name where the beat gives one, so the clause built
    from it says the author's word -- duct tape stays duct tape -- and falls back to
    the generic where the naming is elsewhere in the sentence."""
    t = str(text or "")
    # ...or any wording the engine reads as fastened there, with hardware named --
    # several common ones named only one half, sealed nothing, and the piece was gone
    # by the next beat. Never a wording that takes it OFF.
    _named_seal = bool(engine.groin_sealed(t) and hardware_named(t)
                       and not _GROIN_OFF.search(t))
    if not (_CROTCH_NAMED.search(t) or _named_seal
            or (_BETWEEN_LEGS.search(t) and _ROUND_WAIST.search(t))):
        return ""
    named = hardware_named(t) or ""
    if not named:
        for item in items or ():
            if hardware_named(str(item)):
                named = str(item).strip()
                break
    if not named:
        m = _CROTCH_NAMED.search(t)
        if m:
            named = m.group(0).strip()
    return named or "the hardware"


# WHAT COUNTS AS ASKING FOR IT OFF. restraint_coming_off() reads "unlocks" and little
# else -- it was written for a lock, and duct tape is cut, peeled or unwrapped. A seal
# that outlives the beat removing it is the same bug as one that vanishes early, from
# the other side, so the wording this accepts is deliberately wide: any of these verbs
# in the same sentence as the thing itself.
_SEAL_COMES_OFF = re.compile(
    r"\b(?:unlocks?|unlocking|unlocked|removes?|removing|removed|unfastens?|"
    r"unbuckles?|unclips?|unwraps?|unwrapping|unwrapped|frees?|freeing|freed|"
    r"releases?|releasing|released|undoes|undoing|undone|unties?|untying|untied)\b"
    r"|\b(?:cuts?|cutting|peels?|peeling|peeled|takes?|taking|took|pulls?|pulling|"
    r"pulled|strips?|stripping|stripped|rips?|ripping|ripped|tears?|tearing|tore|"
    r"slices?|slicing|sliced|snips?|snipping|snipped)\b[^.]{0,48}?"
    r"\b(?:off|away|free|loose|open)\b", re.I)


_GROIN_OFF = re.compile(
    r"\bfrom\s+between\s+(?:her|his|their|[\w’']+['’]s)\s+(?:legs|thighs)\b"
    r"|\b(?:off|from)\s+(?:her|his|their|the|[\w’']+['’]s)\s+"
    r"(?:crotch|groin|vagina|pussy|vulva|labia|hips|waist)\b", re.I)
_TAKES_FROM = re.compile(
    r"\b(?:cuts?|cutting|peels?|peeling|peeled|pulls?|pulling|pulled|rips?|ripping|"
    r"ripped|tears?|tearing|tore|strips?|takes?|taking|took|removes?|removed|removing|"
    r"unwraps?|unwrapped|unwinds?|unwound|slices?|sliced|snips?|snipped)\b", re.I)
# Off ANOTHER part: "rips the tape off her mouth" is the gag, not the seal.
_OTHER_PART_OFF = re.compile(
    r"\b(?:off|from|on|over|around|round)\s+(?:her|his|their|the|[\w’']+['’]s)\s+"
    r"(?:mouth|lips|face|eyes|head|wrists?|ankles?|hands?|arms?|neck|throat|knees?|feet)\b",
    re.I)
# A strip torn off the roll, which is tape being got ready.
_STRIP_OFF_ROLL = re.compile(
    r"\b(?:tears?|tore|rips?|ripped|pulls?|pulled|cuts?|takes?|took|peels?|peeled)\s+off\s+"
    r"(?:a|an|another|some|one|two|the)\s+(?:\w+\s+)?"
    r"(?:strip|piece|length|bit|section|square|tear)s?\s+of\b", re.I)


def seal_comes_off(beat, item):
    """Does this beat take that sealed hardware off? Both halves, in one sentence.

    NOT where the sentence takes a piece off ANOTHER part: tape taken off her mouth is
    not this tape, and used to take it too. Not a strip torn off the roll, either,
    which is the piece being made."""
    b, it = str(beat or ""), str(item or "").strip()
    if not b or not it:
        return False
    head = it.split()[-1]          # "duct tape" is cut as "the tape" as often as not
    for sentence in re.split(r"(?<=[.!?])\s+", b):
        sentence = _STRIP_OFF_ROLL.sub(" ", sentence)
        _from_groin = bool(_GROIN_OFF.search(sentence) and _TAKES_FROM.search(sentence))
        if not (_SEAL_COMES_OFF.search(sentence) or _from_groin):
            continue
        if (_OTHER_PART_OFF.search(sentence) and not _from_groin
                and not engine.groin_sealed(sentence)):
            continue
        if (re.search(r"\b" + re.escape(it) + r"\b", sentence, re.I)
                or re.search(r"\b" + re.escape(head) + r"\b", sentence, re.I)
                or re.search(r"\b(?:it|them)\b", sentence, re.I)):
            return True
    return False


# On in the first half and held for the rest, hands clear at the last frame -- the
# same timing as RESTRAINT_GOING_ON, for the same reported fault.
SEALED_ON = (" The {item} goes around the waist and between the legs during this shot, "
             "covering the groin completely: it is on in the first half of the shot and "
             "lies flat against the skin there, sealing it, for the rest of the shot, and "
             "at the last frame it is in plain view and the hands that put it on have let "
             "go and are clear of it.")
SEALED_HOLD = (" The {item} runs around the waist and passes between the legs, "
               "covering the groin completely and lying flat against the skin there, "
               "and it stays exactly so for the whole shot.")


# What the hardware is MADE OF, when the author has not said.
#
# FORM_HOLD has always ended ", the same object in the same material" -- which holds
# the material steady across shots without ever saying what it is. An unspecified
# attribute is filled from the prior, the lesson this file already recorded for a
# bare region and for anatomy at the hip, and the prior for restraint hardware is
# black: black-finished cuffs, black chain. Reported as handcuffs that render black
# instead of steel.
#
# Metal only. Rope, tape and leather come back the way they are written, and a
# default on those would be the node inventing a colour the author did not ask for.
# SAYS THE MATERIAL, NOT THE FINISH. The first version of this read "bright bare
# steel, the metal polished and catching the light" -- which is a mirror finish, and
# it went on every pair of handcuffs in every film, so there was exactly one kind of
# cuff. That is the same fault as the black it was written to fix, from the other
# end: a default filling in more than the gap.
#
# What has to be said is that the metal is METAL and not a black coating. Everything
# past that -- polished, brushed, satin, dulled, worn -- is the finish, and leaving it
# open is what lets one film's cuffs differ from another's. Write it into the sheet
# ("Mara: she, 26, brushed steel handcuffs") and this stands down entirely.
_HARDWARE_MATERIAL = (
    (r"\b(?:handcuffs?|manacles?|leg\s+irons?|irons?)\b", "bare unpainted steel"),
    (r"\b(?:shackles?|cuffs?)\b", "bare unpainted steel"),
    (r"\b(?:chains?)\b", "bare unpainted steel, the links uncoated metal"),
)
# The author having already said. _HARDWARE_NOUN carries an adjective group, so a
# beat writing "black steel cuffs" or "leather cuffs" keeps its own word and this
# stands down -- it only fills a gap, it never argues with the text.
_MATERIAL_SAID = re.compile(
    r"\b(?:steel|stainless|chrome|chromed|nickel|nickelled|silver|iron|brass|"
    r"copper|alloy|metal|metallic|leather|nylon|plastic|rubber|canvas|webbing|"
    r"rope|hemp|cotton|black|blackened|blued|dark|matte|matt|gunmetal|bronze|"
    r"gold|golden|painted|coated|anodi[sz]ed|powder-?coated|white|red|blue|green|"
    r"pink|purple|grey|gray|brown)\b", re.I)

# "The handcuffs is" -- a pair of cuffs, a set of irons and a chain do not agree,
# and the hardware names in these beats are plural as often as not.
_PLURAL_HARDWARE = re.compile(r"\b(?:cuffs?|handcuffs|manacles|shackles|irons|"
                              r"chains|ropes|cords|straps)\b$", re.I)
HARDWARE_MATERIAL_CLAUSE = " The {item} {verb} {material}."


def hardware_material_clause(item, material):
    """The material sentence, agreeing with a plural piece of hardware."""
    if not item or not material:
        return ""
    plural = bool(re.search(r"(?:cuffs|handcuffs|manacles|shackles|irons|chains|"
                            r"ropes|cords|straps)\s*$", str(item), re.I))
    return HARDWARE_MATERIAL_CLAUSE.format(
        item=str(item).strip(), verb="are" if plural else "is", material=material)


def hardware_material(items, said=""):
    """(item, material) to state, or ("", "") when the author already said or it is
    not metal.

    `said` is every word the shot carries about this hardware -- the beat and the
    sheet entry -- because the author can name the material in either. One material
    sentence per shot, for the first piece of metal that needs one: a run naming
    cuffs and a chain gets the cuffs, and the chain's own clause already says its
    links are bare metal."""
    for item in items or ():
        text = str(item or "")
        if not text.strip():
            continue
        for pat, material in _HARDWARE_MATERIAL:
            if not re.search(pat, text, re.I):
                continue
            if _MATERIAL_SAID.search(text):
                return ("", "")           # written into the item's own name
            near = " ".join(s for s in re.split(r"(?<=[.!?])\s+", str(said or ""))
                            if re.search(pat, s, re.I))
            if near and _MATERIAL_SAID.search(near):
                return ("", "")           # written into the sentence that names it
            return (text.strip(), material)
    return ("", "")


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


# WHICH WAY UP A LYING BODY IS. The posture table has one entry for every way of
# being down -- "lies", "lays", "sprawls", and even "rolls onto her stomach", which
# names the facing in the beat and then discards it on the way in. Everything after
# that knows only `lying down`.
#
# So a woman laid face down and cuffed was described in the next shot as lying, with
# no side named, and an unspecified attribute is filled from the prior. Reported as
# her flipping onto her back in the following beat, unasked.
#
# And the one sentence there was said the wrong thing anyway: "The shoulder and the
# hip take the weight of the body" is a body on its SIDE, and it was being said about
# every lying body, prone and supine alike.
_FACE_DOWN = re.compile(
    r"\bface[-\s]?down\b|\bprone\b|\bfront[-\s]?down\b"
    r"|\bon\s+(?:her|his|their|its)\s+(?:stomach|belly|front|face)\b"
    r"|\bonto\s+(?:her|his|their)\s+(?:stomach|belly|front)\b", re.I)
_FACE_UP = re.compile(
    r"\bface[-\s]?up\b|\bsupine\b|\bback[-\s]?down\b"
    r"|\bon\s+(?:her|his|their)\s+back\b|\bonto\s+(?:her|his|their)\s+back\b", re.I)
_ON_SIDE = re.compile(
    r"\bon\s+(?:her|his|their)\s+side\b|\bonto\s+(?:her|his|their)\s+side\b"
    r"|\bside[-\s]?lying\b|\bon\s+one\s+side\b", re.I)

# What is against the surface, for each. Said because it is what makes the facing
# legible in a frame: "face down" alone leaves the torso unplaced, and the prior puts
# it back over. Positively phrased, like everything here.
# For a lying body whose facing the author never wrote. True of every one of them,
# and it keeps the guarantee the weight sentence was added for -- that the region
# under a lying body is not left for the prior to fill -- without claiming a side.
POSE_LYING_WEIGHT = ("The whole length of the body is along the surface, which "
                     "takes the weight of the body")

# Each names what is against the surface, and each ends on the same guarantee the
# generic one carries -- the region under a lying body is never left unsaid.
LYING_FACING = {
    "face down": ("lying face down, the chest, the stomach and the hips flat to the "
                  "surface, which takes the weight of the body, the head turned to "
                  "one side"),
    "face up": ("lying face up, the back of the shoulders and the back of the hips "
                "against the surface, which takes the weight of the body"),
    "on the side": ("lying on one side, the shoulder and the hip against the surface, "
                    "which takes the weight of the body"),
}


def lying_facing(text):
    """"face down", "face up", "on the side", or "" where the text does not say.

    Read from the author's own words only. A lying body whose facing nobody wrote is
    left unwritten -- guessing one is how this went wrong in the other direction."""
    t = str(text or "")
    if _FACE_DOWN.search(t):
        return "face down"
    if _ON_SIDE.search(t):
        return "on the side"
    if _FACE_UP.search(t):
        return "face up"
    return ""


# A BODY LEFT LYING STAYS DOWN, WITH ITS ARMS PLACED. "Mara is lying down." was the
# only thing a later shot said about it -- one late, short sentence -- and nothing at
# all about the arms. The prior for a person on a bed with somebody working over them
# is up on the elbows or the hands, so while Dan cuffed her or taped her mouth she
# pushed herself up off the mattress. REPORTED: she should be lying flat on the bed.
# Said with the arms placed whenever nothing else places them -- an unplaced arm is the
# one the prior props her up on -- and only on the shots she is being worked on (see
# handles_person), in the guard list: every other shot says "Mara is lying down.".
_LYING_SURFACE = re.compile(
    r"\b(?:on|onto|across|to|in)\s+(?:the|a|an|her|his|their)\s+((?:\w+\s+)?"
    r"(?:bed|mattress|floor|floorboards|ground|carpet|rug|cot|bunk|futon|tiles|concrete|"
    r"sofa|couch|bench|table))\b", re.I)


# HANDS ON HER: somebody else's verb with her, or a part of her, as its object.
_HANDLING = (
    r"push(?:es|ed|ing)?|shov(?:e|es|ed|ing)|pull(?:s|ed|ing)?|drag(?:s|ged|ging)?|"
    r"roll(?:s|ed|ing)?|flip(?:s|ped|ping)?|turn(?:s|ed|ing)?|lift(?:s|ed|ing)?|"
    r"carr(?:y|ies|ied|ying)|grab(?:s|bed|bing)?|seiz(?:e|es|ed|ing)|grip(?:s|ped|ping)?|"
    r"hold(?:s|ing)?|held|pin(?:s|ned|ning)?|press(?:es|ed|ing)?|strok(?:e|es|ed|ing)|"
    r"caress(?:es|ed|ing)?|touch(?:es|ed|ing)?|straddl(?:e|es|ed|ing)|"
    r"shak(?:e|es|ing)|slap(?:s|ped|ping)?|position(?:s|ed|ing)?|spread(?:s|ing)?|"
    r"ties|tied|tying|tap(?:es|ed|ing)|(?:hand)?cuff(?:s|ed|ing)?|bind(?:s|ing)?|"
    r"gag(?:s|ged|ging)?|blindfold(?:s|ed|ing)?|unties|untied|uncuff(?:s|ed)?|"
    r"unlock(?:s|ed)?|frees|freed|releas(?:e|es|ed)|"
    r"climb(?:s|ed|ing)?\s+(?:on|onto|over|on\s+top\s+of)|"
    r"kneel(?:s|ing)?\s+(?:on|over|astride)|knelt\s+(?:on|over|astride)|"
    r"sits?\s+(?:on|astride)|lies\s+(?:on|across)|lays?\s+(?:on|over|across)")
_HANDLED_PART = (r"(?:hair|face|cheeks?|arms?|wrists?|legs?|ankles?|shoulders?|back|"
                 r"hips?|waist|head|neck|throat|body|chin|hands?|thighs?|knees?|feet|"
                 r"foot|stomach|belly|side|chest|mouth|lips|jaw|elbows?)")
_OBJECT_TAIL = (r"(?=\s*(?:[,.;!?]|$)|\s+(?:down|up|over|onto|into|on|off|back|away|to|"
                r"towards?|across|by|and|then|with|from|against|out|in|closer|tight|"
                r"tighter|flat|close|around|round|forward|upright|aside|still|face)\b)")


def handles_person(acted, name, sheet="", described=()):
    """Does somebody ELSE put their hands on `name` in this beat -- "Dan pushes her
    down", "Dan strokes her hair", "Dan rolls Mara onto her stomach"? Her own verbs
    ("Mara rolls onto her side") are not handling, and neither is a beat about his
    phone. Used to keep the lying hold to the shots she is being worked on."""
    t = str(acted or "")
    if not name or not t:
        return False
    rows = dict((n, ln) for n, ln in sheet_lines(sheet) if n)
    pron = sheet_pronoun(rows.get(name, ""))
    others = [n for n in (described or []) if n and n != name]
    objs = [re.escape(name)]
    _shared = any(sheet_pronoun(rows.get(o, "")) == pron for o in others) if pron else False
    if pron in ("she", "he") and not _shared:
        objs.append({"she": "her", "he": "him"}[pron])
    elif not pron and others:
        objs += ["her", "him"]
    poss = [re.escape(name) + r"['’]s"]
    if pron in ("she", "he") and not _shared:
        poss.append({"she": "her", "he": "his"}[pron])
    elif not pron and others:
        poss += ["her", "his"]
    rx = re.compile(
        r"\b(?:" + _HANDLING + r")\s+(?:[\w'’]+\s+){0,2}?"
        r"(?:(?:" + "|".join(poss) + r")\s+(?:\w+\s+)?" + _HANDLED_PART + r"\b"
        r"|(?:" + "|".join(objs) + r")\b" + _OBJECT_TAIL + r")", re.I)
    _self = {"she": "she", "he": "he"}.get(pron, "")
    for m in rx.finditer(t):
        sentence = re.split(r"[.;!?]", t[:m.start()])[-1]
        subj = [(x.start(), x.group(0)) for x in re.finditer(
            r"\b(?:" + "|".join(re.escape(n) for n in [name] + others)
            + r"|she|he)\b(?!['’]s)", sentence, re.I)]
        if not subj:
            continue
        last = subj[-1][1]
        if last == name or (_self and last.lower() == _self):
            continue
        if last.lower() in ("she", "he") and not pron:
            continue                # with no pronoun on the sheet it may be her
        return True
    return False


def lying_stays(who="", surface=""):
    """The sentence keeping a RESTRAINED lying body lying for the whole shot.

    Short and plain on purpose. It said "her weight down on it the whole time ...
    while it is done to her", which a video model reads as intimate staging -- REPORTED
    as every scene drifting that way -- and it fired for anybody lying down at all, a
    woman in bed with a fever included. It is for the body being restrained, which is
    what it was asked for, and says only that she stays down. Not where her arms are:
    "her arms resting at her sides" placed arms no beat mentioned. REPORTED."""
    subj = who or "The body"
    return (f" {subj} stays lying flat{f' on the {surface}' if surface else ''} "
            f"through the whole shot.")


def facing_clause(facing):
    """The sentence for a facing, or "" for one that was never named."""
    said = LYING_FACING.get(str(facing or "").strip().lower(), "")
    return f" The body is {said}." if said else ""


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


def pose_clause(position, lying=False, legs="", facing=""):
    """One sentence describing the BODY a limb position makes. "" when unknown.

    `lying` adds what is under it, and `facing` decides WHICH sentence that is --
    prone, supine and on the side do not rest on the same parts. `legs` adds where the
    legs are held, which is a second fact and not an alternative: a hogtie has its
    arms behind the back AND its ankles drawn to them, and the one this file knew
    how to say was the arms."""
    key = str(position or "").strip().lower()
    said = _POSE_OF_POSITION.get(key, "")
    legs_said = _POSE_OF_LEGS.get(str(legs or "").strip().lower(), "")
    if not said and not legs_said:
        return ""
    if said and lying and key == "behind the back":
        # THE OLD WORDING WAS A BODY ON ITS SIDE -- "The shoulder and the hip take the
        # weight" -- asserted over a prone one and a supine one alike. The facing the
        # author wrote decides it now; one they did not write gets the sentence that
        # is true of any lying body, because the guarantee this was added for is that
        # the region UNDER a lying body is not left for the prior to fill.
        _said_facing = LYING_FACING.get(str(facing or "").strip().lower(), "")
        said = (f"{said}. The body is {_said_facing}" if _said_facing
                else f"{said}. {POSE_LYING_WEIGHT}")
    return "".join(f" {part}." for part in (said, legs_said) if part)


def pose_of(pose, who, described):
    """pose_clause's sentences said about WHOSE body, when somebody else is in the shot.

    "Both arms are behind the body, wrists together at the small of the back" names
    nobody; with the man who cuffed her beside her it is an instruction about whoever
    is on screen, and he stands with his arms behind his back too -- the cuffs
    wandering onto whichever body the model put the pose on. Same defect as own_body,
    in the sentence that leads the shot."""
    names = [n for n in (who or []) if n]
    if not pose or not names or len(described or []) < 2 \
            or all(n in names for n in described):
        return pose
    subject = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    out = re.sub(r"(?<![\w'])Both (arms|legs|ankles) are",
                 lambda m: f"Both of {subject}'s {m.group(1)} are", pose)
    out = re.sub(r"(?<![\w'])The whole length of the body",
                 f"The whole length of {subject}'s body", out)
    return re.sub(r"(?<![\w'])The body is", f"{subject}'s body is", out)


_LYING_SENTENCE = re.compile(
    r"\s*(?:The whole length of (?:the|[\w'’-]+['’]s) body|(?:The|[\w'’-]+['’]s) body is "
    r"lying)[^.]*\.")


def split_lying(pose):
    """(pose without its lying sentences, those sentences) -- for the shot whose fall
    has to come before the landing. See pose_clause and pose_of."""
    lying = "".join(" " + m.group(0).strip() for m in _LYING_SENTENCE.finditer(pose or ""))
    return _LYING_SENTENCE.sub("", pose or ""), lying


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


_PART_AT = {"neck": "round the neck", "throat": "round the throat", "mouth": "over the mouth",
            "eyes": "over the eyes", "head": "over the head", "waist": "round the waist",
            "body": "round the body", "chest": "round the chest",
            "groin": "between the legs, over the groin"}


def hardware_where(restraints):
    """{item: where it holds}, from the state's own records of each piece.

    The hold named every item and placed none of them -- "The leather collar, steel
    handcuffs and duct tape stay closed and fastened" -- so nothing said the cuffs are
    on the WRISTS, the collar round the NECK, the tape over the MOUTH. An unstated
    attribute is the model's to choose, and it chose: REPORTED as handcuffs on the
    ankles, a collar gone, the tape off by the next beat. Where one item holds two
    parts -- rope on the wrists and the ankles -- both are said."""
    out = {}
    for r in restraints or ():
        item, part = getattr(r, "item", ""), getattr(r, "part", "")
        if not item or not part:
            continue
        at = (part, getattr(r, "anchor", "") or "")
        out.setdefault(item, [])
        if at not in out[item]:
            out[item].append(at)
    return out


def anchors_placed(where):
    """True when some item in `where` carries its own anchor -- see _where_of."""
    return any(a for v in (where or {}).values() for _p, a in v)


def _where_of(item, where, who=""):
    """Where a (possibly merged) item name holds, as English -- on `who`'s body when a
    single wearer is named ("on Ana's wrists"), else on "the" part. "" if unknown."""
    if not where:
        return ""
    at = where.get(item)
    if at is None:
        head = item.split()[-1].lower() if item.split() else ""
        at = next((v for k, v in where.items()
                   if k.split() and k.split()[-1].lower() == head), None)
    if not at:
        return ""
    said, fast = [], []
    # The PART, and what it is fastened TO -- but not where the arms are: that is the
    # pose sentence's to say, and it does. The anchor is read per item from the state,
    # so the straps on her ankles are "fast to the chair" -- where the one anchor clause
    # for the whole hold said "holding the wrists fast at the chair" for ankle straps.
    for part, anchor in at:
        if who:
            prep = "over" if part in ("mouth", "eyes", "head") else "on"
            s = f"{prep} {who}'s {part}"
        else:
            s = _PART_AT.get(part, f"on the {part}")
        if s not in said:
            said.append(s)
        if anchor and anchor not in fast:
            fast.append(anchor)
    out = " and ".join(said)
    if fast:
        out += ", fast at the " + " and the ".join(fast) + ","
    return out


def restraint_sentence(item, wearers, described, anchor="", rigid=False, posed=False,
                       part="", where=None):
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
    # Each piece where it is -- see hardware_where. With ONE wearer named the place
    # carries the name ("the steel handcuffs on Ana's wrists"); the plain names still
    # decide the grammar and the material below.
    _one = who if (who and len(wearers) == 1) else ""
    _placed = [f"{i} {_where_of(i, where, _one)}".strip() for i in items]
    _subject_item = (", ".join(_placed[:-1]) + " and " + _placed[-1]) if len(_placed) > 1 \
        else (_placed[0] if _placed else item)
    _named_in_place = bool(_one) and any(f"{_one}'s" in p for p in _placed)
    if item:
        subject = (f"The {_subject_item}" if (not who or _named_in_place)
                   else f"The {_subject_item} on {who}")
        verb = "stay" if plural else "stays"
    else:
        subject = f"Every restraint on {who}" if who else "Every restraint"
        verb = "stays"
    it, was = ("they", "were") if plural else ("it", "was")
    _soft_word = re.compile(r"\b(?:rope|ropes|cord|cords|twine|string|strap|straps|"
                            r"tape|scarf|scarves|belt|stocking|stockings|tights|necktie|"
                            r"neckties|ties|bandanas?|sheets?|zip\s*ties?|"
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
        # A CHAIN IS DRAWN TO ITS FULL LENGTH. A pair of cuffs has no length to draw
        # -- it has two rings a fixed distance apart -- and telling the model metal is
        # at full length between two wrists is telling it to draw a chain there.
        # THE POSITION, NOT A STRUGGLE. Both forms ended "and the body strains against it
        # while the fastenings hold" -- written to keep a posed body from freezing, and
        # read as a direction: every later shot of her kneeling asked her to fight the
        # cuffs. REPORTED as characters doing things the beat never wrote. That the
        # position keeps is the hold; what the body does in it is the beat's.
        if item and _CUFF_FORM.search(item) and not re.search(r"\bchain", item, re.I):
            out += ("; the rings are locked where they are and that spacing does not "
                    "change, so the position it fixes is the position that keeps")
        else:
            out += (f"; {_stuff} {_drawn} already drawn to {'their' if _stuff == 'they' else 'its'} "
                    "full length, so the position it fixes is the position that keeps")
    elif rigid:
        out += rigid_tail(item, part or held_part([item] if item else []), plural,
                          where=where)
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
    body = re.sub(r"^Everything worn\b",
                  f"Everything {subject} {'are' if len(names) > 1 else 'is'} wearing", body)
    # Nobody else to hold to their entry when everyone in the shot is undressing.
    if all(n in names for n in (described or [])):
        return " " + body
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
# Set down on a piece of furniture: "a pair of handcuffs lies on the nightstand" is an
# object in the room, not a restraint on anybody. REPORTED as holds before any cuffing.
_RESTING_ON = re.compile(
    r"\b(?:lies?|lay|lying|rests?|rested|resting|sits?|sat|sitting|waits?|waiting)\s+"
    r"(?:\w+\s+){0,2}?(?:on|in|across|beside|by|inside|atop|under|next\s+to)\s+"
    r"(?:the|a|an|his|her|their)\s+(?:\w+\s+)?(?:nightstand|bedside\s+table|table|desk|"
    r"dresser|counter|shelf|drawer|floor|tray|chair|bench|stool|bag|box|case|sofa|couch|"
    r"cabinet|cupboard|mantelpiece|mantel|windowsill|sill|bed|mattress|ground|rug|carpet)s?\b",
    re.I)


# Said only where the BODY turns (see rotates_in), and without "as the view comes
# round", which asked the camera for an orbit nobody wrote.
TURN_HOLD = (" What is on the body now is all that is on it, front, side and behind, and "
             "whatever is fastened stays fastened and closed.")

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


# A WHOLE BODY TURNING, and nothing less. TURN_HOLD fired on any "turn": the page, the
# key, the light off, her head, "turns to him", a body carried across a room. Each
# added a sentence about every side of the body to a beat that showed none of them.
# REPORTED as the beat losing its share of the prompt to clauses it never asked for.
_ROTATES = re.compile(
    r"\b(?:turn(?:s|ed|ing)?|spin(?:s|ning)?|spun|swivel(?:s|led|ling)?|"
    r"rotat(?:es?|ed|ing)|whirl(?:s|ed|ing)?)\s+(?:right\s+|slowly\s+|quickly\s+)?"
    # ...round on the spot, not "turns around the corner": that is a walk. REPORTED.
    r"(?:a)?round\b(?!\s+(?:the|a|an)\s+(?:corner|bend|block|building|car|truck|van|"
    r"table|room|desk|counter|bar|back|side|front))"
    r"|\b(?:turn(?:s|ed|ing)?|roll(?:s|ed|ing)?|flip(?:s|ped|ping)?)\s+(?:right\s+)?over\b"
    r"|\bturn(?:s|ed|ing)?\s+(?:to\s+face\s+)?away\b"
    r"|\bturn(?:s|ed|ing)?\s+(?:her|his|their)\s+back\b"
    # A body spinning, not a thing spun: "twirls her hair", "spins the bottle".
    # REPORTED.
    r"|\b(?:spin(?:s|ning)?|spun|pirouett(?:e|es|ed|ing)|twirl(?:s|ed|ing)?)\b"
    r"(?!\s+(?:the|a|an|her|his|their|its|some|this|that)\b)"
    r"|\broll(?:s|ed|ing)?\s+(?:over\s+)?onto\s+(?:her|his|their)\s+(?:side|back|front|"
    r"stomach|belly|face)\b"
    r"|\b(?:roll(?:s|ed|ing)?|flip(?:s|ped|ping)?|turn(?:s|ed|ing)?|spin(?:s|ning)?|spun|"
    r"twirl(?:s|ed|ing)?)\s+(?:her|him|them|"
    r"(?-i:[A-Z][\w'’-]+))\s+(?:over|onto|around|round|face\s+down|face\s+up)\b"
    r"|\bfrom\s+behind\b|\bback\s+to\s+the\s+camera\b"
    r"|\bshows?\s+(?:her|his|their)\s+back\b|\bfaces?\s+(?:away|the\s+wall)\b", re.I)


def rotates_in(text):
    """Does this beat turn a whole body round -- see _ROTATES?"""
    return bool(_ROTATES.search(text or ""))


FALL_HOLD = (" A bound body falls as one piece: the fastened limbs stay fastened and travel "
             "with it, the arms staying in the hold, the shoulder, hip or side takes "
             "the landing, and the legs fold together under the body.")

FALL_HOLD_FREE = (" The body falls as one piece: the arms stay with it and the shoulder, "
                  "hip or side takes the landing, the legs folding together under it.")

# WHERE THE HANDS ARE, SAID FOR THE FALL ITSELF. FALL_HOLD says "the arms staying in
# the hold" -- a hold the sentence never places -- and sat at the END of the shot with
# the other guards, while the strongest prior in a falling body is a pair of hands
# thrown out to catch it. REPORTED as bound characters breaking their falls with
# their hands. The pose sentence placed the wrists, but as a standing fact; nothing
# tied it to the fall, and the fall won. This names the position the pose already
# holds, for the whole way down, and leads the shot beside the beat -- the same move
# that fixed cuffs drawn in front.
_FALL_ARMS = {
    "behind the back": "locked together behind the back",
    "in front of the body": "locked together and held in against the front of the body",
    "at the waist": "locked together and held in at the waist",
}
_FALL_LEGS = {
    "together": "The bound ankles stay together and both legs go down as one.",
    "ankles to the wrists": "The ankles stay drawn up to the wrists and the whole body "
                            "goes down as one.",
}


def bound_fall_clause(arms="", legs="", who=""):
    """The fall guard for a restrained body, with WHERE its hands are. FALL_HOLD when
    nothing says where. `who` names whose hands, for a shot with somebody else in it."""
    held = _FALL_ARMS.get(str(arms or "").strip().lower(), "")
    if not held:
        return FALL_HOLD
    whose = (who.capitalize() if who in ("her", "his", "their")
             else f"{who}'s" if who else "The")
    legs_said = _FALL_LEGS.get(str(legs or "").strip().lower(), "")
    return (f" {whose} hands stay {held} for the whole fall and the landing, carried "
            f"down with the body, so the shoulder, hip and side take the landing."
            + (f" {legs_said}" if legs_said else ""))


# Where a body comes down. A throw only counts when it puts somebody on one of these:
# "throws her onto the mattress" is a fall, "pushes her into the room" is not. A PUSH
# needs the floor itself -- "pushes her down onto the sofa" is somebody made to sit.
_FLOOR_WORDS = (r"floor|floorboards|ground|carpet|rug|deck|dirt|mud|grass|lawn|gravel|"
                r"sand|snow|ice|tiles?|concrete|pavement|asphalt|road|stairs|steps|earth")
_FLOOR_LIKE = r"(?:" + _FLOOR_WORDS + r")"
_LANDING = r"(?:" + _FLOOR_WORDS + r"|bed|mattress|sofa|couch|cot|bunk|futon)"
_THROWN_ON = (r"(?:(?:her|him|them|herself|himself|themselves|(?-i:[A-Z][\w-]+))\s+)?"
              r"(?:(?:down|back|backwards?|forwards?|hard|roughly|face[-\s]?(?:down|first))"
              r"\s+)?(?:on|onto|to|into|across)\s+(?:the|a|an|her|his|their)\s+(?:\w+\s+)?")

_FALL_CUE = re.compile(
    r"\b(?:falls?|fell|falling|drops?\s+to|dropped\s+to|collapse[sd]?|collapsing|"
    r"topple[sd]?|topples|tips?\s+over|tipped\s+over|keels?\s+over|goes\s+down|"
    r"went\s+down|slumps?|slumped|stumbles?|stumbled|overbalance[sd]?|"
    r"loses?\s+(?:her|his|their)\s+(?:balance|footing)|"
    r"lost\s+(?:her|his|their)\s+(?:balance|footing)|"
    # Falls that never say "fall". REPORTED as bound hands catching the body: every
    # one of these went out with no fall guard at all.
    r"trip(?:s|ped|ping)?\s+(?:over|on|up|and)|tumbl(?:e|es|ed|ing)|"
    # Sent sprawling, not sprawled: "sprawls on the sofa" is lying down -- the
    # posture table has it -- and was read as both a posture and a fall.
    r"(?:sends?|sent|goes|went|knocks?|knocked)\s+(?:(?:her|him|them|[A-Z][\w-]+)\s+)?"
    r"sprawling|pitch(?:es|ed|ing)?\s+(?:forwards?|backwards?|over|headlong)|"
    r"(?:knees|legs)\s+(?:buckle|buckled|give\s+way|gave\s+way|give\s+out|gave\s+out)|"
    r"crash(?:es|ed|ing)?\s+(?:down|(?:on|onto|to|into)\s+(?:the|a)\s+(?:\w+\s+)?"
    + _LANDING + r")|"
    r"lands?\s+(?:hard\s+)?(?:on|onto)\s+(?:(?:her|his|their)\s+(?:side|back|front|"
    r"face|stomach|belly|shoulder|knees)|(?:the|a)\s+(?:\w+\s+)?" + _LANDING + r")|"
    r"slam(?:s|med|ming)?\s+(?:down\s+)?(?:into|onto|on)\s+(?:the|a)\s+(?:\w+\s+)?"
    + _LANDING + r"|"
    r"(?:throw|throws|threw|thrown|hurl(?:s|ed)?|fling|flings|flung|toss(?:es|ed)?|"
    r"knock(?:s|ed)?)\s+" + _THROWN_ON + _LANDING + r"|"
    r"(?:shov(?:e|es|ed)|push(?:es|ed)?)\s+" + _THROWN_ON + _FLOOR_LIKE + r"|"
    # ...and a PERSON put down onto one by somebody else: "pushes her onto the bed",
    # "shoves Mara onto the sofa", "drops her on the floor". REPORTED as a bound body
    # landing with no fall sentence, so nothing kept the hands off the landing.
    r"(?:shov(?:e|es|ed)|push(?:es|ed)?|drops?|dropped|dumps?|dumped)\s+"
    r"(?:her|him|them|(?-i:[A-Z][\w-]+))\s+(?:(?:down|back|backwards?|hard|roughly|"
    r"face[-\s]?(?:down|first))\s+)?(?:on|onto|to|across|into)\s+"
    r"(?:the|a|an|her|his|their)\s+(?:\w+\s+)?" + _LANDING + r"|"
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


# Falls that are not a body going down: asleep, silent, in love, behind, apart; a
# voice dropping to a whisper; somebody stumbling over their words; going down the
# stairs or down on one knee.
_NOT_A_FALL = re.compile(
    # "Flat", "through", "under" and "away" only in their non-body senses: "the deal
    # falls through", "falls under his spell". "Mara falls flat on her back", "falls
    # through the ice" and "falls under the table" are bodies going down, and a thing
    # falling flat is no person anyway -- the subject test below sees to it. REPORTED
    # as the bound-fall hold lost.
    r"(?:falls?|fell|falling)\s+(?:\w+ly\s+)?(?:asleep|silent|quiet|still|apart|behind|"
    r"for\b|in\s+love|ill|short|open|into\s+(?:place|line|step|silence|a\s+rhythm|"
    r"conversation|a\s+doze|a\s+sleep|sleep)|in\s+with|on\s+deaf|"
    r"(?:through|away)(?=\s*(?:[.;,!?]|$))|under\s+(?:(?:his|her|their|the|a|an)\s+)?"
    r"(?:spell|suspicion|control|influence|sway|scrutiny|category|heading|"
    r"jurisdiction|command))"
    # Down on a knee is a kneel. REPORTED as "drops to one knee" staged as a full fall
    # onto the shoulder, hip or side. "Falls to her knees" too.
    r"|(?:drops?|dropped|dropping|sinks?|sank|goes|went|falls?|fell|falling)\s+"
    r"(?:down\s+)?(?:on)?to\s+"
    r"(?:one|a|her|his|their|both)\s+(?:knee|knees|crouch|squat)\b"
    r"|(?:stumbl\w*|trip\w*)\s+(?:over|through|on)\s+(?:(?:her|his|their|the|a|an)\s+)?"
    r"(?:own\s+|first\s+|next\s+)?(?:words?|lines?|apology|answer|sentence|name|reply|"
    r"speech|explanation|excuse|question|response|vows?)\b"
    r"|drops?\s+to\s+(?:a\s+)?(?:whisper|murmur|hush|mutter)"
    r"|(?:goes|went)\s+down\s+(?:on\b|the\s|a\s|to\s+the\s+(?:basement|cellar|lobby|"
    r"kitchen|beach|street|shop|bar|car))", re.I)
# What a person is called when they are the subject. A capital that is not a name --
# "Night falls", "Silence falls" -- is the thing coming down, not somebody.
_FALLER_HEAD = frozenset(
    "she he they her his their him them herself himself themselves body man woman "
    "men women boy girl guy person figure guard officer stranger driver".split())
_NOT_A_FALLER = frozenset(
    "night silence darkness dusk dawn evening rain snow sleep quiet mist fog light "
    "shadow shadows sun moon everything nothing something it the a an this that "
    "tears hair dust ash leaves".split())
_FALLER_SKIP = frozenset(
    "is was are were gets got get being been then suddenly finally slowly almost "
    "nearly just also both all".split())


def _faller_is_person(subject):
    """True for a person, False for a thing, None for no subject at all (elided)."""
    toks = re.findall(r"[A-Za-z][\w'’-]*", subject or "")
    while toks and (toks[-1].lower() in _FALLER_SKIP or toks[-1].lower().endswith("ly")):
        toks.pop()
    if not toks:
        return None
    head = re.sub(r"['’]s?$", "", toks[-1])
    low = head.lower()
    if low in _FALLER_HEAD:
        return True
    return bool(head[:1].isupper() and low not in _NOT_A_FALLER)


def falls_in(text):
    """Does a BODY go down in this beat? A dropped garment is not a fall.

    The fall guard tells the shot what takes the landing and what the legs do, so a
    match on something that is not a person aims all of that at the wrong subject
    and the shot puts a body on the floor to satisfy it.

    The subject is whatever sits between the start of the clause and the verb. An
    object there -- "it drops to the ground", "the belt falls to the floor" -- is
    the thing being let go of, not somebody going down.

    A PERSON, OR NOTHING. The subject was read and then ignored -- the loop returned
    True whatever it found -- so "Night falls over the town", "Mara falls asleep" and
    "Mara stumbles over her words" were each told what takes the landing and how the
    legs fold, a fall nobody wrote. REPORTED as characters doing things the beat never
    wrote. The subject now has to resolve to a person; an elided one ("trips and
    falls") is the sentence's own subject; the idioms in _NOT_A_FALL are no fall."""
    t = text or ""
    for m in _FALL_CUE.finditer(t):
        if _NOT_A_FALL.match(t, m.start()):
            continue
        # A person in the cue itself -- "sends him sprawling", "knocks Kate down",
        # "throws her onto the bed" -- is the one going down, whatever did it.
        if re.search(r"\s(?:her|him|them|herself|himself|themselves|(?-i:[A-Z][\w-]+))\s",
                     m.group(0) + " ", re.I) and not re.search(
                         r"\b(?:balance|footing)\b", m.group(0), re.I):
            return True
        head = t[:m.start()]
        cut = max((c.end() for c in
                   re.finditer(r"[.;!?]\s+|,\s*|\s+(?:and|but|then|so)\s+", head)),
                  default=0)
        subject = head[cut:]
        # ...and a subject joined by "and": "Mara and the chair fall over" is Mara going
        # down with the chair she is tied to. Cut at the "and", the subject was only
        # the chair, and the bound-fall hold was lost for the restrained body.
        _and = re.search(r"(?:^|[.;!?,]\s*)((?:[\w'’-]+\s+){0,2}?[\w'’-]+)\s+and\s+$",
                         head[:cut])
        if (_and and _faller_is_person(_and.group(1))
                and not _OBJECT_FALLER.search(_and.group(1))):
            return True
        if _OBJECT_FALLER.search(subject):
            continue                      # a thing came down, not a person
        person = _faller_is_person(subject)
        if person is None:
            # Elided: the sentence's own subject, or a fragment with none at all. Over a
            # description set off by commas, to the name in front of it: "Mara, cuffed,
            # falls", "Mara, tied to the chair, topples over". REPORTED as the bound-fall
            # hold lost for exactly the restrained subjects it is for.
            start = max((c.end() for c in re.finditer(r"[.;!?]\s+", head)), default=0)
            sent = head[start:cut] if cut > start else ""
            first = sent.split(",")[0]
            named = (len(re.findall(r"[A-Za-z][\w'’-]*", first)) <= 3
                     and "," in sent and _faller_is_person(first)
                     and not _OBJECT_FALLER.search(first))
            lead = re.match(r"\s*((?:[\w'’-]+\s+){0,2}?)(?:[a-z]+(?:s|es|ed)|is|was)\b",
                            first, re.I)
            person = (True if cut <= start else
                      bool(named or (lead and _faller_is_person(lead.group(1))
                                     and not _OBJECT_FALLER.search(lead.group(1)))))
        if person:
            return True
    return False


def fallers_in(text, names, pronouns=None):
    """Who goes down in this beat, as far as its own words say. An empty set where
    they do not say -- a pronoun two people answer to, or nobody at all.

    Read from the clause each fall sits in, so "Dan trips over the crate and falls"
    is Dan, and the woman cuffed beside him is not told her hands stay locked "for
    the whole fall" -- a fall she was never in, that the sentence would put her in."""
    t, out = text or "", set()
    by_word = {}
    for n in (names or []):
        p = (pronouns or {}).get(n)
        for w in {"she": ("she", "her"), "he": ("he", "him")}.get(p, ()):
            by_word.setdefault(w, []).append(n)

    def _people(span):
        """[(at, name)] for everybody `span` names, by name or by an unshared pronoun."""
        got = [(x.start(), n) for n in (names or [])
               for x in re.finditer(r"\b" + re.escape(n) + r"\b", span)]
        got += [(x.start(), who[0]) for w, who in by_word.items() if len(who) == 1
                for x in re.finditer(r"\b" + w + r"\b", span, re.I)]
        return sorted(got)

    for m in _FALL_CUE.finditer(t):
        if _NOT_A_FALL.match(t, m.start()):
            continue
        # THE PERSON PUT DOWN, not the one doing it: "Dan pushes HER onto the bed".
        obj = re.search(r"\s((?:her|him|them)|(?-i:[A-Z][\w-]+))\s", m.group(0) + " ", re.I)
        if obj and not re.search(r"\b(?:balance|footing)\b", m.group(0), re.I):
            got = _people(obj.group(1))
            if got:
                out.update(n for _a, n in got)
                continue
        head = t[:m.start()]
        cut = max((c.end() for c in
                   re.finditer(r"[.;!?]\s+|,\s*|\s+(?:and|but|then|so)\s+", head)),
                  default=0)
        stop = re.search(r"[.;!?,]|\s+(?:and|but|then|so)\s+", t[m.end():])
        clause = t[cut:m.end() + (stop.start() if stop else len(t) - m.end())]
        got = _people(clause)
        if not got and re.fullmatch(r"\s*(?:(?:\w+ly|then|also|suddenly|finally|just)\s+)*",
                                    head[cut:], re.I):
            # AN ELIDED SUBJECT IS THE ONE BEFORE IT: "Mara slips and falls", "tries to
            # run but falls". Read as nobody, the bound fall gave way to the free one
            # whenever somebody free shared the shot. REPORTED.
            start = max((c.end() for c in re.finditer(r"[.;!?]\s+", head)), default=0)
            got = _people(t[start:cut])[:1]
        out.update(n for _a, n in got)
    return out


CHAIN_HOLD = (" Every restraint stays closed and fastened as it was put on, its links "
              "keeping their size and the run between them taut") + FORM_HOLD

# ...and without the strain, for the reason restraint_sentence gives.
CHAIN_POSE_HOLD = (" Every restraint stays closed and fastened as it was put on; the metal "
                   "is already drawn to its full length, so the position it fixes is the "
                   "position that keeps") + FORM_HOLD

# A posture the body takes up itself, which no hardware forces: the end of a pose.
# Read as posture_in reads it, so a try ("tries to stand") or a denial ("cannot stand")
# is not an arrival.
_ATTEMPT = re.compile(r"\b(?:tr(?:y|ies|ied|ying)|struggl(?:e|es|ed|ing)|attempts?|"
                      r"cannot|can't|unable|fails?|failed)\b", re.I)


def _leaves_pose(acted, sheet, held=()):
    """Does somebody held in hardware stand, sit or walk of their own in this beat?"""
    t = str(acted or "")
    held = set(held or ())
    if not t or not held:
        return False
    cast = [n for n, _l in sheet_lines(sheet) if n]
    got = posture_in(t, cast)
    if any(n in held and p in ("standing", "sitting") for n, p in got.items()):
        return True
    return (not _ATTEMPT.search(t)
            and any(n in held for n in subjects_for(t, sheet, r"walks?|walked")))


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
    # ...and "cuffs Ana's wrists in front of her", with no body part after it.
    (_LIMB_EV + r"\s+(?:\w+\s+){0,3}?in\s+front(?:\s+of\s+(?:her|his|their)"
     r"(?:\s+(?:body|chest|waist)|(?!\s+(?:face|eyes|mouth|head|nose)\b)))?\b",
     "in front of the body"),
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


# limb_anchor's own positions. "at the waist" is one of them, not an object.
_LIMB_WHERE = frozenset(phrase for _pat, phrase in _LIMB_ANCHOR)


def limb_anchor_parts(position):
    """(where, point) of a limb_anchor phrase: "behind the back, at the headboard" gives
    ("behind the back", "at the headboard"), "at the pipe" ("", "at the pipe") and "at
    the waist" ("at the waist", ""). The point is the object the limbs are fastened to."""
    p = re.sub(r"\s+", " ", str(position or "")).strip()
    if ", at the " in p:
        where, point = p.split(", at the ", 1)
        return where.strip(), "at the " + point.strip()
    if p.lower().startswith("at the ") and p.lower() not in _LIMB_WHERE:
        return "", p
    return p, ""


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


# WHAT COUNTS AS THE AUTHOR PLACING THE CAMERA: a word that is about the camera in any
# sentence, or an ordinary verb with the camera as its subject. The bare words stood
# the hold down on prose that never mentions the camera -- "tilts her head", "pulls
# out a chair", "eggs in a pan", "a handheld radio", "circles around the table", "the
# drone of traffic", "looks into the camera" -- and one of them in the anchor freed
# the camera for the whole film. A lens or a film stock is the look, not a move, so
# the hold stays. ("dolly in" is left out: Dolly is a name.)
_CAMERA_VERBS = (r"moves?|moving|movement|motion|work|angle|position|shake|pans?|"
                 r"panning|tilts?|tilting|tracks?|tracking|follows?|following|pushes|"
                 r"pushing|pulls?|pulling|zooms?|zooming|dollies|dollying|cranes?|"
                 r"craning|booms?|trucks?|trucking|orbits?|orbiting|circles?|circling|"
                 r"arcs?|arcing|rises?|rising|drops?|dropping|lowers?|glides?|drifts?|"
                 r"swings?|rotates?|holds?|stays?|remains?|sits?|stands?|is")
_CAMERA_ASKED = re.compile(
    r"\bcameras?\s+(?:\w+ly\s+|then\s+|never\s+|always\s+)?(?:" + _CAMERA_VERBS + r")\b|"
    r"\bcameras?\s*:|"
    r"\b(?:static|still|fixed|locked[-\s]off|stationary|steady|hand-?held|moving|"
    r"tracking|shaky|overhead|tripod|slow)\s+(?:cameras?|shots?|frames?|takes?)\b|"
    r"\b(?:dolly|crane|jib|drone|aerial|orbit(?:ing)?|arc|follow|pov)\s+shots?\b|"
    r"\b(?:slow|quick|fast|gentle|smooth|slight|subtle|steady|gradual|whip|swish|crash|"
    r"snap)\s+(?:pans?|tilts?|dolly|zoom|push[-\s]?in|pull[-\s]?(?:back|out)|crane|orbit)\b|"
    r"\bzoom(?:s|ing|ed)?\s+(?:in|out)\b|\bpan(?:s|ning|ned)?\s+(?:left|right|across|over\s+to)\b|"
    r"\bhand-?held\b(?=\s*(?:[.,;:!?)]|$|and\b|throughout\b|footage\b|look\b|style\b))|"
    r"\bdolly\s+zoom\b|\bsteadicam\b|\bgimbal\b|\brack\s+focus\b|\bpov\b|"
    r"\bpoint[-\s]of[-\s]view\b|\blocked[-\s]off\b|\bdutch\s+(?:angle|tilt)\b", re.I)


_LIGHT_THING = (r"(?:lights?|lamps?|candles?|torch(?:es)?|flashlights?|fire(?:place)?|"
                r"bulbs?|neon|screens?|tv|television|headlights?|sun|moon)")
_LIGHT_CHANGES = re.compile(
    r"\b" + _LIGHT_THING + r"\b[^.;]{0,24}?\b(?:go(?:es)?\s+(?:out|off|on|dark|dim)|"
    r"went\s+(?:out|off|on)|comes?\s+on|came\s+on|dim(?:s|med|ming)?|"
    r"brighten(?:s|ed|ing)?|flicker(?:s|ed|ing)?|fades?|faded|dies|died|"
    r"blaze[sd]?|flare[sd]?|rises?|rose|sets?|setting|sinks?|sank)\b|"
    r"\b(?:turn(?:s|ed|ing)?|switch(?:es|ed|ing)?|flick(?:s|ed|ing)?|click(?:s|ed|ing)?|"
    r"shut(?:s|ting)?)\s+(?:on|off)\s+(?:the\s+|a\s+|her\s+|his\s+)?" + _LIGHT_THING + r"\b|"
    r"\b(?:turn(?:s|ed|ing)?|switch(?:es|ed|ing)?|flick(?:s|ed|ing)?|click(?:s|ed|ing)?)\s+"
    r"(?:the\s+|a\s+|her\s+|his\s+)?" + _LIGHT_THING + r"\s+(?:on|off)\b|"
    r"\b(?:lights?|blows?\s+out|blew\s+out|snuffs?\s+out)\s+(?:the\s+|a\s+)?"
    r"(?:candles?|fire|lamp|lantern|match)\b|"
    r"\b(?:open(?:s|ed|ing)?|clos(?:e|es|ed|ing)|draw(?:s|n|ing)?|pull(?:s|ed|ing)?)\s+"
    r"(?:the\s+|back\s+the\s+)?(?:curtains?|blinds?|shutters?|drapes?)\b|"
    r"\b(?:darkness|night|dusk|dawn)\s+(?:falls|fell|comes|came|breaks|broke|fills)\b|"
    r"\b(?:room|sky|screen|world)\s+(?:goes|went|turns?|turned|grows?|grew|fades?|faded)\s+"
    r"(?:dark|darker|black|white|bright|brighter|red|blue|orange)\b|"
    r"\bfades?\s+(?:to|into)\s+(?:black|white)\b|"
    r"\b(?:lightning|sunrise|sunset|explosion|muzzle\s+flash)\b", re.I)


def light_changes(beat):
    """Does this beat change the light on purpose?

    A shot's change of level across its length is otherwise the chain cooking, and is
    taken back out (see shot_grade). A lamp switched off is the author's, and keeps
    its darkness."""
    return bool(_LIGHT_CHANGES.search(str(beat or "")))


# What in _CAMERA_ASKED asks the camera NOT to move, or only names its style.
_CAMERA_STAYS = re.compile(r"hand-?held|static|still|fixed|locked|stationary|steady|"
                           r"tripod|\b(?:holds?|stays?|remains?|sits?|stands?)\b|"
                           r"^cameras?\s*:$", re.I)


def camera_moves(text):
    """Does this text ask the camera to MOVE -- a pan, a push-in, a follow, an orbit?

    "The camera is" is read with what follows it: "is tracking Mara" moves, "is still"
    and "is on Mara" do not. A bare "is" counted as staying, so a tracking camera was
    held. REPORTED."""
    t = str(text or "")
    for m in _CAMERA_ASKED.finditer(t):
        said = m.group(0).strip()
        if re.search(r"\bis$", said, re.I):
            said += " " + " ".join(t[m.end():].split()[:2])
            if not re.search(r"\bis\s+(?:\w+ly\s+)?\w+ing\b", said, re.I):
                continue
        if not _CAMERA_STAYS.search(said):
            return True
    return False


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


def frame_hold(beat, anchor="", people=1, outdoor=False, held=False):
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
    phrased: it says what the frame holds, never what it is not.

    EVERY SHOT WITH SOMEBODY IN IT, not only the ones staging whole-body action.
    A face acting, a line spoken, a person standing in cuffs -- each was left to the
    prior, which crops to the face, and a cropped body is a wardrobe and a set of
    restraints the model redraws from nothing when the frame widens again. REPORTED
    as clothing and bondage equipment not looking the same, or disappearing, when a
    character leaves the shot and comes back -- and asked for in so many words:
    both characters entirely in the shot, so nothing is missed. `people` is how many
    are described; 0 is an empty frame and gets nothing.

    `held` is a shot that opens on the previous shot's last frame. That frame already
    holds them whole -- the shot that composed it said so -- and "with the room around
    them" asks for a WIDER view than the frame it opens on, which the model settles by
    cutting to a side-on wide in a room drawn fresh. It used to say the bodies STAY
    whole there instead, "for the whole take", on every shot: a framing sentence the
    beat never asked for, which the keyframe and the camera hold already cover.
    REPORTED as the beat losing its share of the prompt. A held shot gets nothing.

    And nothing where the author asked for a camera move (_CAMERA_ASKED): a frame
    that moves is theirs to compose."""
    b = str(beat or "")
    if _FRAME_SIZE.search(b) or _FRAME_SIZE.search(str(anchor or "")):
        return ""
    if tight_framing(b) or tight_framing(str(anchor or "")):
        return ""
    if camera_moves(b) or camera_moves(str(anchor or "")):
        return ""
    if people is None:
        people = 1
    if int(people) < 1 or held:
        return ""
    place = "the surroundings" if outdoor else "the room"     # see outdoors()
    if int(people) > 1:
        return (f" The frame holds every body in it whole, head to feet, with {place} "
                "around them.")
    return (f" The frame holds the whole body, head to feet, with {place} around it.")


def entrance_clause(names):
    """Walk somebody the beat places into a frame that does not have them yet.

    The shot opens on the previous shot's last frame, and they are not in it. Said
    this way the frame stays the frame -- the camera, the room, the people already
    there -- and the newcomer comes into it, rather than the shot being recomposed
    around them. Positive, like every clause here: where they come from.

    Without ", and everything already in the frame stays where it is": a direction to
    every other body in the shot, which their own beat never gave. REPORTED as
    characters doing things the beat never wrote. The held camera and the keyframe
    already keep the frame. Never for somebody already there -- see
    already_in_position."""
    names = [n for n in (names or []) if n]
    if not names:
        return ""
    who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    verb = "comes" if len(names) == 1 else "come"
    return f" {who} {verb} into the frame from its edge as the shot begins."


# A beat that has somebody already THERE and in place: looking up from something,
# watching from somewhere, sitting at it, being at, on or in it, on the phone.
_IN_POSITION = re.compile(
    r"(?:is|are|was|were)\s+(?:already\s+|still\s+)?(?:at|on|in|by|behind|beside|near|"
    r"inside|sitting|seated|standing|lying|kneeling|waiting|asleep|perched|curled|"
    r"leaning|slumped|propped)\b"
    r"|,?\s*already\b"
    r"|(?:looks?|glances?|peers?)\s+up\s+from\b"
    r"|(?:watch(?:es)?|looks?|stares?|peers?)\s+(?:out\s+|on\s+)?from\b"
    r"|sits?\s+(?:at|on|in|behind|beside|by|across)\b"
    r"|stands?\s+(?:at|by|behind|beside|in|near|across)\b"
    r"|lies\s+(?:on|in)\b|waits?\s+(?:at|by|in|on|behind)\b"
    r"|(?:leans?|perch(?:es)?|lounges?|sprawls?)\s+(?:on|against|in|at|across)\b"
    r"|dials\b|answers\b|picks\s+up\s+the\s+(?:phone|receiver)\b"
    r"|takes\s+it\b|keeps?\s+(?:typing|reading|working|writing|talking)\b", re.I)


def already_in_position(acted, name, sheet=""):
    """Does the beat write `name` as already there and in place, rather than arriving?

    Read off the words straight after the name -- an adverb allowed between -- so
    "Dan looks up from his phone" is there and "Dan crosses to the desk" is not."""
    t = str(acted or "")
    if not t or not name:
        return False
    for m in re.finditer(r"\b" + re.escape(name) + r"\b(?!['’])", t):
        tail = t[m.end():]
        if re.match(r"\s*(?:\w+ly\s+)?(?:" + _IN_POSITION.pattern + r")", tail, re.I):
            return True
    return False


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
    # TWO PEOPLE ONLY. "Eyes on whoever is speaking, faces turned to them" turned a
    # whole room's heads on every line, a gaze nobody wrote. REPORTED as characters
    # doing things the beat never said. See faces_each_other for when two do.
    if n_people != 2:
        return ""
    return " They face each other, eyes on each other."


_NOT_FACE_TO_FACE = re.compile(
    r"\bto\s+(?:herself|himself|themselves|itself|no\s+one|nobody)\b"
    r"|\boff[\s-]?(?:screen|camera)\b"
    r"|\b(?:from|through|behind|across)\s+(?:the\s+)?(?:other\s+room|another\s+room|"
    r"next\s+room|hall|hallway|door|wall|window|landing|stairs|upstairs|downstairs|"
    r"outside|kitchen|bathroom)\b"
    r"|\b(?:at|to|into)\s+(?:the|her|his|their)\s+(?:tv|television|telly|screen|mirror|"
    r"camera|radio|laptop|monitor|phone|reflection|computer|webcam|microphone|mic)\b",
    re.I)


# A call that an earlier beat started and none has ended yet: "Lou dials a number."
# then 'Lou says, "You have one hour."' is a line down the phone.
_CALL_STARTS = re.compile(
    r"\b(?:dials?|dialled|dialed|dialing|dialling)\b"
    r"|\bphone\s+to\s+(?:her|his|their)\s+ear\b"
    r"|\b(?:answers?|picks?\s+up)\s+(?:the|her|his|their)\s+(?:phone|call)\b"
    r"|\bon\s+the\s+phone\b|\bvideo\s+call\b", re.I)
_CALL_ENDS = re.compile(
    r"\bhangs?\s+up\b|\bhung\s+up\b|\bends?\s+the\s+call\b"
    r"|\bputs?\s+(?:the|her|his|their)\s+phone\s+(?:down|away)\b", re.I)
# Side by side in a moving car, nobody turns to face anybody.
_AT_THE_WHEEL = re.compile(r"\b(?:drives?|driving|drove)\b|\b(?:at|behind)\s+the\s+wheel\b"
                           r"|\bgrips?\s+the\s+wheel\b", re.I)
# A listener the last beat left running, climbing or leaving has a task of their own.
_ON_THE_MOVE_SRC = (r"sprints?|runs?|ran|flees|fled|races?|dashes|bolts?|vaults?|"
                    r"climbs?|escapes?|walks?\s+(?:away|off|out)|storms?\s+(?:off|out)|"
                    r"leaves")


def faces_each_other(beat, acted, sheet, described, poses=None, facing="",
                     on_call=False, scene="", prev=""):
    """Is this a line said by one of two people TO the other, face to face?

    The dialogue eye-line is inferred, never written, so it is said only where nothing
    in the beat argues with it. Said down a phone (this beat's, or one an earlier beat
    picked up: `on_call`), through a door, to herself, at a television, from off
    screen, from the driving seat, to somebody lying face down, or to somebody with an
    action of their own -- in this beat, or still running from the last (`prev`) -- it
    was a turn of the head the author never wrote and sometimes could not have meant.
    REPORTED as characters doing things the beat never said."""
    people = [n for n in (described or []) if n]
    if len(people) != 2:
        return False
    b = str(beat or "")
    if (on_call or _REMOTE.search(b) or _NOT_FACE_TO_FACE.search(b)
            or _AT_THE_WHEEL.search(f"{scene} {acted}")):
        return False
    talkers = [n for n in (speakers_in(b, sheet) or []) if n in people]
    if not talkers:
        return False
    listeners = [n for n in people if n not in talkers] or people
    _moving = set(subjects_for(str(prev or ""), sheet, _ON_THE_MOVE_SRC))
    for n in listeners:
        if len(talkers) == 1 and (acts_in(acted, n, sheet, described) or n in _moving):
            return False
        if (poses or {}).get(n) == "lying down" and (
                facing == "face down" or lying_facing(acted) == "face down"):
            return False
    return True


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


# A BELT IS CLOTHING UNTIL SOMETHING SAYS OTHERWISE. "The radio on Bertrand's belt
# crackles" and "Bertrand tightens his belt" were each told "a belt closes around the
# waist and hips" as a piece of hardware. REPORTED. Somebody's own belt is never placed
# unless the beat binds or seals with it, and an unowned one only in a film with
# restraints or a seal in it (`gear`).
_BELT_CONTEXT = re.compile(
    r"\b(?:restrain\w*|strap(?:s|ped|ping)?|lock(?:s|ed|ing)?|padlock\w*|chastity|"
    r"seal\w*|bound|binds?|binding|tied|ties|tying|cuff\w*|chain\w*|shackl\w*)\b", re.I)
_OWN_BELT = re.compile(r"\b(?:his|her|their|my|your|(?-i:[A-Z][\w'’-]*)['’]s)\s+"
                       r"(?:\w+\s+)?belt", re.I)
# ...and what the beat takes off is not placed: "unties the blindfold" was told "a
# blindfold covers the eyes". REPORTED.
_TAKES_OFF = (r"\b(?:unties?|untied|untying|removes?|removed|removing|unbuckles?|"
              r"unbuckled|unclips?|unclipped|unfastens?|unfastened|unlocks?|unlocked|"
              r"cuts?|snips?|slices?|(?:takes?|took|pulls?|pulled|lifts?|lifted|slides?|"
              r"slid|slips?|slipped|peels?|peeled|rips?|ripped|tears?|tore|yanks?|"
              r"yanked)\s+off)\s+(?:[\w'’]+\s+){{0,3}}?(?:{pat})"
              r"|\b(?:takes?|took|pulls?|pulled|lifts?|lifted|slides?|slid|slips?|slipped|"
              r"peels?|peeled|rips?|ripped|tears?|tore|yanks?|yanked|gets?|got)\s+"
              r"(?:[\w'’]+\s+){{0,2}}?(?:{pat})\s+(?:\w+\s+)?off\b")


def unanchored_hardware(text, gear=True):
    """Phrases placing any hardware that is named with no body part beside it.

    A window of 60 characters either side counts as 'beside'. If the text already
    says where the thing goes, nothing is added -- what you wrote wins. Nothing the
    text takes off, and a belt only as gear -- see _BELT_CONTEXT."""
    out = []
    t = text or ""
    for pat, phrase in _HARDWARE_ANCHOR:
        if re.search(_TAKES_OFF.format(pat=pat), t, re.I):
            continue
        if phrase.startswith("a belt ") and not _BELT_CONTEXT.search(t) and (
                not gear or _OWN_BELT.search(t)):
            continue
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


# ...plus down on a knee, which is a kneel -- see _NOT_A_FALL.
_POSTURE_OF = tuple(engine._POSTURE_OF) + (
    ("kneeling", re.compile(r"\b(?:drops?|dropped|dropping|sinks?|sank|sinking|falls?|fell|"
                            r"falling)\s+(?:down\s+)?(?:on)?to\s+(?:one|a|her|his|their|both)\s+"
                            r"knees?\b", re.I)),)
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
                       and not engine.denied_posture(part, m.start())
                       # "a phone lies on the nightstand" lays nobody down
                       and not engine.posture_of_a_thing(part, m.start())),
                      key=lambda h: h[0])
        prev = 0
        for at, pose in hits:
            span = part[prev:at]
            tail = part[at:]
            # ...and "pins her TO the floor", "throws her ACROSS the bed", "pushes
            # her FACE down onto it": the object, before what puts it down.
            obj = re.match(r"\w+\s+(?:the\s+)?([\w'-]+)\s+"
                           r"(?:down|up|back|onto|into|on|in|to|across|flat|face|hard|"
                           r"roughly)\b", tail, re.I)
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


# THINGS A BODY DOES WITHOUT GETTING UP. "turns her head" read as travel, and
# "closes her eyes" read as handling an object, so a woman laid face down lost her
# posture on the next beat that had her do anything at all -- and the facing went
# with it, which is her rolling onto her back unasked.
#
# Narrow on purpose: one of these verbs, directly on one of these parts, and the part
# is her own. "lifts her to her feet" does not match -- the possessive is not on a
# part -- and neither does "opens the door" or "takes the cup". Feet, legs and knees
# are left out: those are how somebody gets up.
_STILL_LYING = re.compile(
    r"\b(?:turns?|turned|turning|lifts?|lifted|lifting|raises?|raised|raising|"
    r"drops?|dropped|dropping|lowers?|lowered|lowering|opens?|opened|opening|"
    r"closes?|closed|closing|shuts?|shutting|moves?|moved|moving|shifts?|shifted|"
    r"shifting|rests?|rested|resting|presses?|pressed|pressing|tilts?|tilted|"
    r"tilting|buries|buried|burying)\s+(?:her|his|their|the)\s+"
    r"(?:head|face|eyes?|eyelids?|chin|jaw|mouth|lips?|cheek|temple|forehead|"
    r"shoulders?|arms?|hands?|fingers?|gaze|breath|weight)\b", re.I)


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
        if _STILL_LYING.search(span):
            continue                  # done lying down as readily as standing
        if _real_travel(span):
            out.add(name)
        elif pose == "lying down" and _HANDLES.search(span):
            out.add(name)
    return out


# WHAT A BODY DOES WITHOUT LEAVING ITS POSTURE: speaking, looking, the face, the breath,
# waiting -- and dressing or undressing, which a squat was REPORTED not holding through.
_IN_PLACE_ACT = re.compile(
    r"(?:says?|said|asks?|asked|tells?|told|whispers?|whispered|repl(?:ies|ied)|"
    r"answers?|answered|shouts?|shouted|yells?|yelled|calls?|called|murmurs?|"
    r"mutters?|adds?|added|continues?|looks?|looked|watch(?:es|ed)|stares?|stared|"
    r"glances?|glanced|gazes?|peers?|smiles?|smiled|grins?|grinned|frowns?|"
    r"laughs?|laughed|giggles?|nods?|nodded|shakes|shook|sighs?|sighed|breathes?|"
    r"breathed|blinks?|cries|cried|sobs?|sobbed|weeps|waits?|waited|listens?|"
    r"listened|pauses?|paused|thinks?|hesitates?|swallows?|winces?|flinches?|"
    r"shivers?|trembles?|gasps?|moans?|groans?|whimpers?|blushes|stays?|stayed|"
    r"remains?|keeps?|holds?|"
    r"(?:takes?|took|pulls?|pulled|slips?|slipped|peels?|peeled)\s+(?:off|on|out\s+of)|"
    r"(?:puts?|pulls?)\s+on|removes?|removed|unbuttons?|unzips?|undresses|strips|"
    r"buttons?|zips?|ties|unties|(?:turns?|turned|lifts?|lifted|raises?|raised|"
    r"drops?|dropped|lowers?|lowered|tilts?|tilted|closes?|closed|opens?|opened|"
    r"shuts?|moves?|shifts?|rests?|rested)\s+(?:her|his|their)\s+(?:head|face|eyes?|"
    r"chin|gaze|mouth|lips?))\b", re.I)


# SOMEBODY ELSE MOVES THE BODY, or it moves itself somewhere: "Hobb pulls Lark to her
# feet", "walks Lark to the patrol car", "Lark ducks into the back seat". Lark went on
# "kneeling" through all of it, because only a posture verb of her own cleared it and
# a restrained body is not cleared by its own actions. REPORTED.
_MOVES_BODY = (
    r"(?:pull(?:s|ed)?|haul(?:s|ed)?|yank(?:s|ed)?|hoist(?:s|ed)?|heav(?:es|ed)|"
    r"help(?:s|ed)?|lift(?:s|ed)?|drag(?:s|ged)?|carr(?:ies|ied)|scoop(?:s|ed)?)\s+"
    r"{who}\b(?:\s+(?:up|to\s+(?:her|his|their)\s+feet|stand|onto|into|out|off|across|"
    r"along|through|toward|towards|away|back|down|over|from|to)\b)?"
    r"|(?:walk(?:s|ed)?|lead(?:s)?|led|march(?:es|ed)?|steer(?:s|ed)?|guid(?:es|ed)|"
    r"escort(?:s|ed)?|frogmarch(?:es|ed)?|push(?:es|ed)?|shov(?:es|ed))\s+{who}\s+"
    r"(?:\w+\s+)?(?:to|into|out|toward|towards|across|along|through|down|up|back|"
    r"outside|inside|over)\b")
_MOVES_SELF = (r"{who}\s+(?:\w+ly\s+)?(?:ducks?|ducked|climbs?|climbed|slides?|slid|"
               r"crawls?|crawled|steps?|stepped|gets?|got|clambers?|clambered|scrambles?|"
               r"scrambled)\s+(?:back\s+)?(?:into|in|out|onto|up|off|down|through)\b")


def carried_off(acted, name, sheet="", described=()):
    """Does this beat move `name` -- somebody else carrying, pulling or walking them,
    or them getting into or out of something? See _MOVES_BODY."""
    t = _outside_speech(str(acted or ""))
    who = [re.escape(name)]
    rows = dict((n, ln) for n, ln in sheet_lines(sheet) if n)
    pron = sheet_pronoun(rows.get(name, ""))
    if pron in ("she", "he") and sum(1 for n in (described or [])
                                     if sheet_pronoun(rows.get(n, "")) == pron) == 1:
        who.append({"she": "her", "he": "him"}[pron])
        subj = pron
    else:
        subj = ""
    alt = "(?:" + "|".join(who) + ")"
    if re.search(r"\b(?:" + _MOVES_BODY.format(who=alt) + r")", t, re.I):
        return True
    self_alt = "(?:" + "|".join([re.escape(name)] + ([subj] if subj else [])) + ")"
    return bool(re.search(r"\b" + _MOVES_SELF.format(who=self_alt), t, re.I))


def own_action(acted, name, sheet="", described=()):
    """Does the beat give `name` a body action of their own -- a verb straight after the
    name, or the pronoun only they answer to here -- that is not one of _IN_PLACE_ACT?

    "Mara sprints down the path", "Mara checks her phone": the beat is directing her,
    and a posture carried from an earlier beat argues with it. "Mara smiles",
    "Mara says ...", "Kate takes off her shirt" leave her where she was."""
    t = _outside_speech(str(acted or ""))
    who = [re.escape(name)]
    rows = dict((n, ln) for n, ln in sheet_lines(sheet) if n)
    pron = sheet_pronoun(rows.get(name, ""))
    if pron in ("she", "he") and sum(1 for n in (described or [])
                                     if sheet_pronoun(rows.get(n, "")) == pron) == 1:
        who.append(pron)
    for m in re.finditer(
            # A state is not an action, and an adverb is not the verb: "Mara has tears
            # in her eyes", "seems tired", "always smiles" each popped a carried kneel.
            # REPORTED.
            r"\b(?:" + "|".join(who) + r")\b(?:\s*,[^,.;]*,)?\s*,?\s*"
            r"(?:(?:and|then)\s+)?(?:(?:\w+ly|always|just|now|still|never|often|again|"
            r"also|only|even|almost|already|once|finally|soon)\s+)*"
            r"(?!(?:is|was|and|or|but|then|to|as|in|on|at|with|by|of|for|has|does|seems|"
            r"looks|feels|wears|needs|wants|appears|sounds|knows|remembers|hates|loves|"
            r"likes)\b)"
            r"(?-i:([a-z]+(?:s|es|ed)))\b", t, re.I):
        if not _IN_PLACE_ACT.match(t, m.start(1)):
            return True
    return False


def posture_hold(poses, described, upright=()):
    """One short sentence keeping people in the pose an earlier beat put them in.

    Only for people this shot DESCRIBES -- a pose belonging to somebody the text
    does not mention is a pose for nobody, and the model draws the person that
    sentence implies. Short on purpose: this is latched, so it lands in every shot
    after the one that stages it, and a long clause repeated is the guard bloat
    this node was rebuilt to escape.

    STANDING IS NOT SAID -- it is the default -- EXCEPT for somebody in restraints
    (`upright`). There the shot describes hardware holding the body, and standing is
    not what that implies: a woman standing
    when she was tied was on the floor with her legs spread by the next beat.
    REPORTED. For her it is said, and said as feet on the floor."""
    upright = set(upright or ())
    who = [(n, p) for n, p in (poses or {}).items()
           if n in set(described or []) and (p != "standing" or n in upright)]
    if not who:
        return ""
    if len(who) == 1 and who[0][1] == "standing":
        return f" {who[0][0]} is standing, upright on both feet."
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
_NOT_AN_OBJECT = engine.PLACE_NOT_AN_OBJECT
_GOES_TO = re.compile(r"\b(?:to|into|toward|towards|through\s+to|"
                      r"enters?|entered|entering|reaches|reached|arrives?\s+(?:at|in)|"
                      r"steps?\s+into|stepped\s+into)\s+"
                      + _DET_POSS + r"\s+" + _MOD
                      + r"(" + _PLACE + r")\b" + _NOT_AN_OBJECT, re.I)
# "down the hallway", "along the corridor" -- what it passes THROUGH.
_GOES_VIA = re.compile(r"\b(?:down|along|across|through|up|via|past)\s+"
                       + _DET_POSS + r"\s+" + _MOD
                       + r"(" + _PLACE + r")\b" + _NOT_AN_OBJECT, re.I)
# "from the living room", "out of the kitchen" -- where it STARTS.
_GOES_FROM = re.compile(r"\b(?:from|out\s+of|leaves?|leaving)\s+"
                        + _DET_POSS + r"?\s*" + _MOD
                        + r"(" + _PLACE + r")\b" + _NOT_AN_OBJECT, re.I)
# What makes "X room" a DIFFERENT room. Anything else in front of it -- "the room",
# "her room", "the dark room" -- is the room the shot is already in.
_ROOM_QUALIFIER = frozenset("""
other next adjoining neighbouring neighboring spare guest back front side hotel motel
box store storage boiler sitting drawing family games music panic reading computer
play sun throne war interview interrogation hospital operating recovery exam treatment
server control engine stock staff common living dining
""".split())


def _named_place(m, text):
    """The place match `m` names, or "" when it is only the word "room".

    "Dan walks across the room to the bed" read as travel to a place called "room":
    the shot was told it opens in the bedroom and arrives in the room, the next one
    that it takes place in "the room" instead of the bedroom the scene describes, and
    whoever the walk did not name was left behind in the room he had supposedly left.
    A bare room is the one they are in. A qualified one -- "the other room", "the
    spare room" -- is somewhere else, and keeps its qualifier."""
    got = re.sub(r"\s+", " ", m.group(1)).strip().lower()
    if got != "room":
        return got
    before = re.search(r"([A-Za-z][\w-]*)\s+$", str(text or "")[:m.start(1)])
    word = before.group(1).lower() if before else ""
    return f"{word} room" if word in _ROOM_QUALIFIER else ""
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
        return next((p for p in (_named_place(m, b) for m in rx.finditer(b)) if p), "")

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
    # A THRESHOLD IS NOT A ROOM. "Mara walks to the doorway" or "to the office door"
    # moved the whole shot into a room called the doorway, or the office: the next
    # shot, about Dan on the sofa she left him on, was told it "takes place in the
    # doorway", walls, floor and furniture -- a person who never moved relocated to
    # where somebody else walked. REPORTED as characters doing things the beat never
    # wrote. A walk to a door stays in the room it starts in.
    b = str(beat or "")
    frm, via, to = (
        "" if (not leg or leg in _THRESHOLDS
               or re.search(r"\b" + re.escape(leg) + r"\s+(?:door|doors|doorway|window|"
                            r"windows|gate)\b", b, re.I)) else leg
        for leg in (frm, via, to))
    return frm, via, to


_THRESHOLDS = frozenset(("doorway",))


_OUTDOOR = re.compile(
    r"\b(?:streets?|roads?|avenues?|boulevards?|alley(?:way)?s?|sidewalks?|pavements?|"
    r"crosswalks?|crossroads|intersections?|squares?|plazas?|parks?|gardens?|yards?|"
    r"backyards?|lawns?|forests?|woods|woodland|beach(?:es)?|shores?|seafront|"
    r"fields?|meadows?|desert|mountains?|hills?|hillside|cliffs?|car\s+parks?|"
    r"parking\s+lots?|rooftops?|bridges?|highways?|motorways?|outdoors|outside|"
    r"open\s+air|courtyards?|campsite|riverbank|river|lakeside|lake|docks?|piers?|"
    r"harbou?r|playground|countryside|trail|bus\s+stop|train\s+platform|"
    r"marketplace|market\s+square|town\s+square|village\s+green)\b"
    r"(?<!depth of field)", re.I)
_INDOOR = re.compile(
    r"\b(?:rooms?|bedrooms?|kitchens?|bathrooms?|showers?|living\s+rooms?|lounges?|"
    r"hallways?|corridors?|offices?|studios?|apartments?|flats?|basements?|cellars?|"
    r"garages?|attics?|lofts?|shops?|stores?|caf[eé]s?|bars?|pubs?|restaurants?|"
    r"clubs?|gyms?|classrooms?|wards?|cells?|warehouses?|barns?|sheds?|elevators?|"
    r"lifts?|churche?s?|halls?|hotels?|motels?|cabins?|indoors|inside|interior|"
    r"stairwells?|workshops?|diners?|lobb(?:y|ies)|changing\s+rooms?|locker\s+rooms?)\b",
    re.I)


def outdoors(here="", scene="", opening=""):
    """Is the place this shot is in OUTSIDE? True only on evidence of it.

    Every clause that holds a place still was written for a room -- "this room a
    moment earlier: the same walls, floor, furniture and light", "with the room around
    it". Said on a public street it is an instruction to draw walls and furniture,
    and the model did: REPORTED as a street scene turning into a house on the second
    beat, the one where the last frame rides along claimed as "this room".

    The shot's own place decides first; where it is not named, the scene -- or, with
    an anchor standing in for the scene, the opening beat -- and only when the text
    says outdoors and does not also name an interior: "a café on a busy street" is
    kept as a room, which is what these clauses already said."""
    h = str(here or "")
    if h.strip():
        if _OUTDOOR.search(h) and not _INDOOR.search(h):
            return True
        if _INDOOR.search(h):
            return False
    for text in (scene, opening):
        t = str(text or "")
        if _OUTDOOR.search(t) or _INDOOR.search(t):
            return bool(_OUTDOOR.search(t) and not _INDOOR.search(t))
    return False


# What holds a place still, said of a room and of anywhere else.
_SURROUND_IN = "walls, floor, furniture and light"
_SURROUND_OUT = "surroundings, ground and light"


def where_hold(here, scene, outdoor=False):
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
    if outdoor:
        return (f" This shot takes place in the {here}: the ground, surroundings and "
                f"light are the {here}'s throughout.")
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


def facing_phrase(beat="", movers=()):
    """"each body facing the way it goes", unless the beat walks backwards.

    ONLY THE ONES WHO MOVE. "Each body" is every body described, so "Mara walks to the
    workbench" walked Dan, still in the frame, to the workbench with her. REPORTED as
    characters doing things the beat never wrote. With `movers` -- see movers_in --
    they are named, and nobody else is moved."""
    if _GOES_BACKWARD.search(str(beat or "")):
        return ""
    if movers:
        return f" {_join_names(list(movers))} facing the way of travel"
    return " each body facing the way it goes"


def movers_in(acted, sheet, described):
    """The described people this beat moves, where that is not all of them. [] when the
    beat does not say, when everybody moves, or when a walk takes somebody along
    ("walks Dan to the door", "leads her out") -- "each body" is right for those."""
    described = [n for n in (described or []) if n]
    movers = [n for n in subjects_for(
        acted, sheet, _MOVES_OFF_SRC + r"|sprints?|jogs?|hurr(?:y|ies)|rush(?:es)?|"
        r"strides?|strolls?|wanders?|paces?|pads?|marche?s?") if n in described]
    if not movers or set(movers) >= set(described):
        return []
    others = [n for n in described if n not in movers]
    if re.search(r"\b(?:walks?|leads?|led|drags?|dragged|carr(?:y|ies|ied)|escorts?|"
                 r"guides?|pulls?|pushes|takes?|brings?|ushers?|marches|follows?)\s+"
                 r"(?:" + "|".join(re.escape(n) for n in others) + r"|her|him|them)\b",
                 str(acted or ""), re.I):
        return []
    return movers


def move_clause(dest, beat="", movers=()):
    """Perform an arrival the place list cannot name. "" when there is nowhere.

    The BODIES end nearer the destination. It was "the shot travels to the {dest}"
    with "the {dest} nearer at the last frame" -- the destination nearer the lens,
    which is a push-in, and the camera did it. The last-frame comparison stays,
    because it is what says which way the walk runs. `movers` names who -- see
    facing_phrase."""
    if not dest:
        return ""
    facing = facing_phrase(beat, movers)
    who = f" {_join_names(list(movers))}" if movers else " each body"
    return (f" The move to the {dest} plays out on screen from its first step to its "
            f"last,{facing + ' and' if facing else who} nearer the {dest} at "
            f"the last frame than at the first.")


_WALKS_THERE = re.compile(
    r"\b(?:walk(?:s|ed|ing)?|go(?:es|ing)?|went|run(?:s|ning)?|ran|(?<!the\s)steps?|"
    r"stepped|heads?|headed|cross(?:es|ed)|climb(?:s|ed|ing)?|carr(?:y|ies|ied)|"
    r"leads?|led|drags?|dragged|hurr(?:y|ies|ied)|strides?|strode|marche[sd])\b", re.I)


def travel_anchor(frm, via, to, here="", beat="", movers=()):
    """Say where the shot starts, what it passes, and where it ends. "" if nowhere.

    `here` is the room an earlier beat established, used when the beat names no
    origin -- a journey with only a destination is what renders as a cut.

    Short on purpose: this lands on travel beats, which already carry an action,
    and the node's whole balance problem is continuity crowding the beat out."""
    start = frm or here
    if not to or start == to:
        return ""
    facing = facing_phrase(beat, movers)
    walk = ("the walk between them played out on screen, every step in frame"
            + (f",{facing}." if facing else "."))
    # A FALL IS NOT A WALK. "Tumbles down the steps" arrives somewhere as surely as a
    # walk does, and was told "the walk between them, every step in frame" -- a
    # direction nobody wrote, and the body walked down the stairs it was meant to fall
    # down. REPORTED as characters self-directing. The journey still holds -- the shot
    # opens where the last one ended -- it just says what carries the body there.
    if falls_in(beat) and not _WALKS_THERE.search(beat or ""):
        walk = "the fall between them played out on screen, in frame."
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
                    + _MOD + r"(" + _PLACE + r")\b" + _NOT_AN_OBJECT, re.I)


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
        got = _named_place(m, text)
        if got:
            return got
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


def keyframe_claim(n, first=False):
    """Name the keyframe where it rides beside references.

    A keyframe is not only frame one. tokenize_with_weights labels EVERY image item
    `<Picture N>:`, and build_conditioning appends the handoff after the references so
    it disturbs no numbering -- so the encoder is shown it as a picture too. Alone it
    is <Picture 1> with nothing naming it, which is exactly how ComfyUI's own
    MiniMaxH3ImageToVideo hands over a first frame, and nothing needs saying.

    Beside a reference it is picture N+1 of N+1 with nothing naming it, and the rule
    for that is the node's oldest: a picture the prompt never names is ANOTHER
    subject. It shows the very people the references name, so what arrives is a copy
    of somebody already in the shot -- reported as the same person twice in one
    frame. handoff_claim fixed this for the same picture once it was demoted into the
    reference rows, and the keyframe was left unnamed on the reading that a first
    frame is not a subject; to the encoder it is one. On shot 1 it is the author's
    first_frame, with nothing earlier to be carried from."""
    if first:
        return (f" <Picture {n}> is the frame this shot opens on: the same place and "
                f"the people this shot describes, already in place rather than joined "
                f"by anybody new.")
    return handoff_claim(n)


_COUNT_ONE =" There is one person in the shot: one body, one face."
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


def room_claim(n, present, joining, outdoor=False):
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
    the shot is joined by nobody new, in the one case where it is.

    OUTDOORS it is "this place", held by its surroundings -- see outdoors()."""
    where = "this place" if outdoor else "this room"
    said = (f" <Picture {n}> is {where} a moment earlier: the same "
            f"{_SURROUND_OUT if outdoor else _SURROUND_IN}, from the same camera.")
    if present:
        said += (f" {' and '.join(present)} "
                 f"{'are the people' if len(present) > 1 else 'is the person'} there.")
    if joining:
        said += (f" {' and '.join(joining)} {'are' if len(joining) > 1 else 'is'} in "
                 f"{where} too, already in place at the first frame.")
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


def returning_room_claim(n, room, present, arriving, outdoor=False):
    """Claim a frame of a room the film showed before and has come back to.

    Its own claim, not room_claim's: that one says "a moment earlier", and this
    picture is from shots ago. It names who is in it, because an unclaimed person
    in a picture is another person."""
    said = (f" <Picture {n}> is the {room} as the film last showed it"
            f"{', where this shot arrives' if arriving else ''}: the same "
            f"{_SURROUND_OUT if (outdoor or outdoors(room)) else _SURROUND_IN}.")
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


def plate_claim(n, outdoor=False):
    """Claim shot 1's first_frame when it is carried as the SET rather than frame one.

    room_claim cannot serve here and saying so is the point: it calls the picture
    "this room a moment earlier" and names who was standing in it, and on shot 1
    there is no earlier and nobody was. A plate is a picture of a place with no
    people in it, and the claim has to say exactly that -- an unclaimed picture is
    read as another subject, and a picture of an empty room claimed as a person is
    how a figure gets invented to stand in it."""
    return (f" <Picture {n}> is the set this shot takes place in: the same "
            f"{_SURROUND_OUT if outdoor else _SURROUND_IN}, from the same camera. It is a picture of "
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
    rest = _RESTING_ON.search(part)
    moved = (_HANDLING_VERB.search(part) or (shown and _SHOWN_VERB.search(part))
             or (rest and engine.posture_of_a_thing(part, rest.start())))
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
    continuity rule about restraints. A carried thing is not hardware at all: see
    engine.carried_masked."""
    t = engine.carried_masked(text or "")
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
def displaced_garments(beat, scene):
    """engine.displaced_garments, kept to garments. "Drago lifts Calla into the trunk"
    read "lifts" plus whatever followed as a garment raised, and "The calla into the
    trunk stays on, pulled up" was carried three beats on. REPORTED. A capture has to
    name a garment, and never starts with somebody's name."""
    named = {n.lower() for n, _ in sheet_lines(scene or "") if n}
    named |= {w.lower() for w in re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][\w'’-]+)",
                                            str(beat or ""))}
    return [(g, how) for g, how in engine.displaced_garments(beat, scene)
            if engine.garment_words(g)
            and str(g).split()[0].lower().rstrip("'’s") not in named]
puts_it_back = engine.puts_it_back
restored_garments = engine.restored_garments


_DISPLACED_TO = re.compile(
    r"\b(?:down|up|off)\s+(?:to|around|round|past|below|over|at)\s+"
    r"(?:(?:her|his|their|the)\s+)?(mid[-\s]?thighs?|thighs?|knees?|ankles?|calves|shins|"
    r"hips|waist|feet|chest|ribs|belly|stomach|armpits|shoulders?)\b"
    r"|\b(?:down|up)\s+to\s+(mid[-\s]?thigh)\b", re.I)


def displaced_to(beat, garment):
    """How far the beat moved `garment` -- " to the thighs" -- or ""."""
    b = str(beat or "")
    head = str(garment or "").split()[-1] if str(garment or "").split() else ""
    at = re.search(r"\b" + re.escape(head) + r"\b", b, re.I) if head else None
    m = _DISPLACED_TO.search(b, at.end() if at else 0)
    if not m:
        return ""
    part = (m.group(1) or m.group(2) or "").lower()
    return f" to {part}" if part.startswith("mid") else f" to the {part}"


def under_displaced(garment, sheet, moved=()):
    """The undergarments a moved garment now shows: same wearer, same region, not moved.

    The sheet listed "denim shorts, a black thong" side by side and nothing said the
    thong is what the lowered shorts uncover -- so the thong was left to chance.
    REPORTED as shorts pulled down with no thong to be seen."""
    if engine.is_undergarment(garment):
        return []
    line = next((ln for n, ln in sheet_lines(sheet or "")
                 if re.search(r"\b" + re.escape(garment) + r"\b", ln, re.I)), "")
    regs = set(engine.regions_of(garment))
    moved_heads = {str(m).split()[-1].lower() for m in (moved or ()) if str(m).split()}
    return [g for g in engine.garments_in(line)
            if g != garment and engine.is_undergarment(g)
            and regs & set(engine.regions_of(g))
            and g.split()[-1].lower() not in moved_heads]


def off_now_clause(name, line_now, gone_items, beat="", after_removal=False,
                   pictured=False):
    """Say what somebody is wearing now, where something could put a removed garment back.

    The text already stops describing it, and that is not enough on its own: three
    things show the model the garment again --
      * a tagged picture of the person, taken in the full outfit, riding every shot;
      * the shot after a removal carrying the last frame as a reference, which is
        whatever state the removal reached;
      * the beat itself naming it -- "tosses her jacket onto the chair".
    REPORTED as clothing coming back once removed.

    POSITIVE, and it never names what came off: naming a garment is what puts it on,
    which is this file's standing rule (see the `revived` note). So it says what IS
    worn -- "Ana wears only her white tank top, black jeans and sneakers now" -- read
    from the entry as the shot will print it. Said only where one of the three is
    true; "" otherwise, and "" when the entry lists nothing worn."""
    if not name or not gone_items:
        return ""
    named = bool(beat) and any(
        re.search(r"\b" + re.escape(str(i).split()[-1]) + r"\b", beat, re.I)
        for i in gone_items if str(i).split())
    if not (pictured or after_removal or named):
        return ""
    # The entry's own items, as written ("a white tank top"), not the reader's
    # shortening of them ("top") -- the description is the continuity.
    # Tags out: a picture is claimed where its entry names it, once -- a second
    # <Picture N> here would be one more naming of it (see _PICTURE_TAG).
    body = _PICTURE_TAG.sub("", (line_now or "").split(":", 1)[-1])
    worn = [re.sub(r"^(?:a|an|the|some)\s+", "", piece.strip().rstrip("."), flags=re.I)
            for piece in body.split(",")
            if piece.strip() and engine.garment_words(piece)]
    if not worn:
        return ""
    what = worn[0] if len(worn) == 1 else ", ".join(worn[:-1]) + " and " + worn[-1]
    # "Zoe wear only their purple sweater" for an entry with no pronoun: the verb
    # agrees with the one name, and "wear" is only for a declared "they". REPORTED.
    _declared = sheet_pronoun(line_now) or ""
    pron = {"she": "her", "he": "his", "they": "their"}.get(_declared, "their")
    verb = "wear" if _declared == "they" else "wears"
    return f" {name} {verb} only {pron} {what} now."


def displaced_hold(items, dest=None, beneath=None):
    """Say where a moved garment now sits, so the next shot does not put it back.

    Without this the garment is described by the sheet in the state it was WORN, and
    the sheet is re-stamped into every shot -- so shorts pulled down are pulled back
    up by the next beat, or come back looking like a different pair.

    It was also not a sentence: "On the body and the denim shorts pulled down, left
    exactly where the beat put them" -- no verb, no destination, nothing about what
    the lowered garment uncovers. REPORTED as the shorts coming back up, and as shorts
    pulled down with no thong to be seen. Now: where it is, how far, and what shows."""
    if not items:
        return ""
    said = []
    for thing, how in items[:2]:
        plural = thing.endswith("s") and not thing.endswith("ss")
        it = "them" if plural else "it"
        s = (f"The {thing} {'stay' if plural else 'stays'} on, {how}"
             f"{(dest or {}).get(thing, '')}, where the beat left {it}")
        under = (beneath or {}).get(thing) or []
        if under:
            names = " and ".join(f"the {u}" for u in under[:2])
            s += (f", with {names} under {it} in view and still on")
        said.append(s)
    return " " + "; ".join(said) + "."


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


# A PART OF A GARMENT MOVED IS NOT THE GARMENT REMOVED, and a one-piece off a SHOULDER
# is lowered, not off: it is still on from the waist down. Both were read as the whole
# garment gone -- "slips the straps of her dress off her shoulders" came out as "the
# red dress comes off ... fully removed", and with nothing declared under it the bare
# clause filled everything the dress covered, genitals included. REPORTED as a clothed
# character exposing their genitals. A shirt or jacket slid off the shoulders is how
# those come off, so the shoulder rule is for one-pieces only.
_GARMENT_PART = (r"(?:shoulder\s+)?(?:straps?|sleeves?|hems?|collars?|necklines?|bodices?|"
                 r"cups?|waistbands?|hoods?)")
_PART_OF = (r"\b" + _GARMENT_PART + r"\s+of\s+(?:her|his|their|the|its)\s+"
            r"(?:[\w-]+\s+){0,2}?")
_OFF_SHOULDER = re.compile(
    r"^\s*(?:off\s+)?(?:(?:her|his|their|the|one|both)\s+)+shoulders?\b", re.I)
_ONE_PIECE = frozenset("""dress gown nightgown nightdress nightie slip sundress minidress
    maxidress jumpsuit romper playsuit bodysuit catsuit leotard swimsuit""".split())


def _lowered_not_off(word, span, after):
    """True where `word` is only a part of the garment moved, or a one-piece lowered."""
    if span and re.search(_PART_OF + re.escape(word) + r"\b", span, re.I):
        return True
    return bool(_OFF_SHOULDER.match(after or "")
                and (engine.singular_garment(word) or word).lower() in _ONE_PIECE)


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
    # A PERSON IS NOT A GARMENT. "Ana, in a red dress..." reads "Ana" as the head of a
    # wardrobe entry, so "Dan unzips Ana's dress and it falls to the floor" took off
    # "Ana" -- her own sheet line lost its name, and the shots after it said "Ana's she
    # came off earlier". Names, pronouns, and any capitalised word inside a sentence.
    _people = {n.lower() for n, _l in sheet_lines(scene) if n} | {
        "she", "he", "her", "him", "his", "hers", "they", "them", "their", "it"}

    def _a_person(word, at_text):
        w = word.lower()
        if w in _people or engine.singular_garment(w) in _people:
            return True
        return bool(word[:1].isupper() and not re.search(
            r"(?:^|[.!?]\s+)[\"“]?$", at_text))
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
        if cut and cut.group(0) == ",":
            more = _list_runs_on(tail[cut.start():], scene)
            if more:
                span = span + ", " + more
        _after = ""                 # what follows the "off": "her shoulders", or nothing
        if not (re.fullmatch(_UNDO_VERB, m.group(0), re.I)
                or re.search(r"\b(?:off|away|out\s+of|down)$", m.group(0), re.I)):
            part = re.search(r"\b(?:off|away)\b", span, re.I)
            if part:
                if _HAS_VERB.search(span[:part.start()]):
                    continue
                _after = span[part.end():]
                span = span[:part.start()]
        for _wm in re.finditer(r"\b[\w-]{3,}\b", span):
            word = _wm.group(0)
            low = engine.singular_garment(word)
            if not low or low in found:
                continue
            if _a_person(word, beat[:m.end()] + span[:_wm.start()]):
                continue
            if _lowered_not_off(word, span, _after):
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
    # The garment as the subject: "Kate's jacket comes off", "her dress drops to the
    # floor", "her jacket is removed". See engine.COMES_OFF.
    for m in engine.GARMENT_COMES_OFF.finditer(beat):
        if _in_a_request(beat, m.start()):
            continue
        word = m.group(1)
        low = engine.singular_garment(word)
        if not low or low in found or low in _NOT_A_GARMENT or _RESTRAINT_WORD.match(low):
            continue
        if _a_person(word, beat[:m.start(1)]):
            continue
        # "Her dress slips off her shoulders" -- lowered, not off. See _lowered_not_off.
        if (_lowered_not_off(word, "", beat[m.end():])
                or (re.search(r"\bshoulders?\b", m.group(0), re.I) and low in _ONE_PIECE)):
            continue
        if not _is_entry_head(low, scene):
            _kin = [k for k in _GARMENT_KIN.get(low, ()) if _is_entry_head(k, scene)]
            if len(_kin) != 1 or _kin[0] in found:
                continue
            low = _kin[0]
        if re.search(r"\b" + re.escape(word) + r"\b\s*(?:is|was|walks|stands|sits|=)",
                     scene, re.I):
            continue
        found.append(low)
    shown_off = exposed_by(beat, scene)
    return [f for f in found if f not in shown_off]


def _list_runs_on(rest, scene):
    """The rest of a garment LIST past a comma -- ", shirt and bra" -- or "".

    The object of a removal verb ended at the first comma, so "takes off her jacket,
    shirt and bra" took off the jacket and left the shirt and bra listed on every
    later shot. Only a list that CLOSES with "and"/"or" and names nothing but
    garments runs on: "her jacket, the shirt underneath" is an aside, and "her
    jacket, sits down" is the next action."""
    segs, pos = [], 0
    while True:
        m = re.match(r",\s*", rest[pos:])
        if not m:
            return ""
        start = pos + m.end()
        nxt = _OBJECT_END.search(rest, start)
        seg = rest[start:nxt.start() if nxt else len(rest)]
        words = re.findall(r"\b[\w-]{3,}\b", seg)
        if (_HAS_VERB.search(seg) or _REMOVAL_PROSE.search(seg)
                or not any(_is_entry_head(engine.singular_garment(w), scene)
                           for w in words)):
            return ""
        segs.append(seg.strip())
        if re.search(r"\b(?:and|or)\b", seg, re.I):
            return ", ".join(segs)
        if not nxt or nxt.group(0) != ",":
            return ""
        pos = nxt.start()


def removal_owner(beat, token, sheet):
    """Whose `token` this beat takes off, off the possessive in front of it. "" if none.

    "Dan takes off her shirt" is Kate's shirt, and "Dan takes off his shirt" after
    it is Dan's -- the same word, two garments. See engine.possessor_at."""
    people = [n for n, _ in sheet_lines(sheet) if n]
    head = str(token or "").split()[-1] if str(token or "").strip() else ""
    b = str(beat or "")
    if not people or not head:
        return ""
    m = re.search(r"\b" + re.escape(head) + r"(?:e?s)?\b", b, re.I)
    if not m:
        return ""
    pron = {n: sheet_pronoun(ln) for n, ln in sheet_lines(sheet) if n}
    who = engine.names_in(b, people)
    subject = engine._agent_before(b, m.start(), people) or (who[0] if who else "")
    return engine.possessor_at(b, m.start(), people, subject, pron)


garments_in = engine.garment_words
region_of = engine.region_of


_NAKED_CUE = engine.STRIPS_BARE


def strips_who(beat, cast, sheet=""):
    """Who this beat undresses. [] when it cannot tell.

    strips_bare only answers WHETHER somebody ends up with no clothes on. The
    wardrobe was then read off the whole shot sheet, so in a shot describing two
    people BOTH were stripped -- one character undressing made the other undress
    too. Reported as the second character mimicking the first.

    The subject is the name before the cue, the same reading posture_in uses -- unless
    somebody ELSE is being undressed: "Dan undresses Kate", "Dan strips her naked".
    See engine.undressed_object; `sheet` is what resolves the "her". With one person
    in the shot there is nobody else it can be."""
    people = [n for n in (cast or []) if n]
    b = str(beat or "")
    if not people or not b:
        return []
    if len(people) == 1:
        return people[:1]
    _pron = {n: sheet_pronoun(ln) for n, ln in sheet_lines(sheet) if n}
    obj = [n for n in engine.undressed_object(b, people, pronouns=_pron) if n in people]
    if obj:
        return obj
    m = _NAKED_CUE.search(b) or engine.STRIPS_TO.search(b)
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


_REMOVE_ITEM_SEP = re.compile(r"\s*[,;&+]\s*")
_REMOVE_ITEM_AND = re.compile(r"\s+(?:and|plus)\s+", re.I)


def removal_items(line):
    """The items of one `remove:` line. "jacket and shirt" is two; "black and white
    shirt" is one, because "black" is not a garment -- split, it would take the black
    jeans off as well."""
    out = []
    for piece in _REMOVE_ITEM_SEP.split(str(line or "")):
        parts = _REMOVE_ITEM_AND.split(piece)
        heads = [(re.findall(r"[\w-]+", p.lower()) or [""])[-1] for p in parts]
        if len(parts) > 1 and all(engine._WORD_ONE.match(h) or _RESTRAINT_WORD.match(h)
                                  or engine._PHRASE_ONE.search(p) or engine._HW_ONE.search(p)
                                  for h, p in zip(heads, parts)):
            out.extend(parts)
        else:
            out.append(piece)
    return [p for p in out if p.strip()]


def sheet_form(token, scene):
    """The sheet's own words for a `remove:` item it writes differently. Else as given.

    The scrub finds an item by its exact words, so "remove: blue jacket" found
    nothing in "blue denim jacket" and the jacket stayed on. Where exactly one entry
    has the item's head noun and every one of its words, that entry is the item."""
    t = str(token or "").strip()
    if (not t or not scene or " " not in t
            or re.search(r"\b" + re.escape(t) + r"\b", scene, re.I)):
        return t
    words = t.lower().split()
    hits = []
    for line in str(scene).split("\n"):
        for item in re.split(r"[,;.]", line.split(":", 1)[-1]):
            item = re.sub(r"<\s*picture[\s_\-]*\d+\s*>", " ", item, flags=re.I)
            have = re.findall(r"[\w-]+", item.lower())
            if words[-1] not in have or not all(w in have for w in words):
                continue
            name = engine.bare_name(re.sub(r"\s+", " ", item).strip())
            end = re.search(r"\b" + re.escape(words[-1]) + r"\b", name, re.I)
            name = name[:end.end()] if end else ""
            if name and name.lower() not in [h.lower() for h in hits]:
                hits.append(name)
    return hits[0].lower() if len(hits) == 1 else t


_REMOVE_ITEM_LEAD = re.compile(
    r"^(?:(?:a|an|the|her|his|their|its|my|your|our|some|both)\s+"
    r"|[A-Za-z][\w’'-]*['’]s\s+)+", re.I)


def removal_token(item):
    """One `remove:` item written the way the scrub can find it in the sheet.

    The scrub looks the item up in the sheet word for word, so "remove: her jacket"
    found nothing in "blue denim jacket" and the jacket stayed in every later shot,
    and the removal clause said "takes the her jacket off". So did "the jacket",
    "Kate's jacket", "jackets" against a sheet saying "jacket", and "Jacket." with
    its full stop. What is left is the garment: lower case, no article or
    possessive in front, the head noun as the vocabulary spells it."""
    t = re.sub(r"\s+", " ", str(item or "")).strip().strip(".!?:;\"'“”")
    t = _REMOVE_ITEM_LEAD.sub("", t).strip().lower()
    if not t:
        return ""
    words = t.split()
    return " ".join(words[:-1] + [engine.singular_garment(words[-1])])


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
        for t in removal_items(m.group(1)):
            t = removal_token(t)
            if t and t not in removed:
                removed.append(t)
        return ""

    def take_added(m):
        phrase = m.group(1).strip()
        if phrase:
            added.append(phrase)
        return ""

    body = _EXACT_LINE.sub("", _ADD_LINE.sub(take_added, _REMOVE_LINE.sub(take_removed, beat or "")))
    body = _HOLD_LINE.sub("", body)
    return re.sub(r"\n{2,}", "\n", body).strip(), removed, added


# HOLD: A PIECE THE AUTHOR DECLARES ON, WITHOUT ANY PROSE TO PARSE.
#
#     Dan grabs her by the arm.
#     hold: Mara, handcuffs behind her back; duct tape over her mouth
#
# Reported: restraints and gags written in wordings the reader does not know ("ties her
# hands", "puts a ball gag in her mouth") were never registered, so no later shot held
# them. A hold: line registers each piece on that person from this beat on -- the
# applying wording on this shot, the named hold after -- until a `remove:` line names it.
_HOLD_LINE = re.compile(r"^[ \t]*hold[ \t]*:[ \t]*(.*?)[ \t]*$", re.I | re.M)
_HOLD_WHO = re.compile(r"\s*([A-Za-z][\w'’-]*?)(?:['’]s)?(?:\s*[,:;–—-]\s*|\s+)(.*)$", re.S)
_HOLD_CUT = re.compile(
    r"\s*(?:,|\b(?:behind|over|across|on|onto|upon|in|into|inside|around|round|at|to|"
    r"between|above|below|under|beneath|through|from|with|against|chained|locked|"
    r"fastened|tied|taped|strapped|clipped|hooked|bound|cuffed|holding|keeping|binding|"
    r"tying|locking|fastening|pinning|joining|linking|running|leading|tight|tightly|"
    r"firmly|snug|snugly|now|still|that|which)\b)", re.I)
_HOLD_MEASURE = re.compile(
    r"^(?:(?:one|two|three|four|several|some|more)\s+)?"
    r"(?:(?:pair|set|length|strip|piece|roll|coil|loop|bit|band|layer|wrap)s?\s+of\s+)+", re.I)
_HOLD_PART_OF = {"hands": "wrists", "feet": "ankles"}
_HOLD_LIMBS = ("wrists", "ankles", "arms", "legs", "knees", "thighs", "elbows")
_HOLD_LEG_PARTS = ("ankles", "legs", "knees", "thighs", "feet")
_HOLD_ARM_PARTS = ("wrists", "arms", "hands", "elbows")
# A piece no hardware word names still goes on the part the line gives it: "a scarf over
# her eyes" is a blindfold in the author's words.
_HOLD_BY_PART = {"mouth": "gag", "eyes": "blindfold", "neck": "collar"}


def hold_lines(beat):
    """[the text of each `hold:` line] in this beat, in the order written."""
    return [m.group(1).strip() for m in _HOLD_LINE.finditer(beat or "") if m.group(1).strip()]


def strip_hold_lines(beat):
    """The beat with its `hold:` lines taken out: they are read, never sent."""
    return re.sub(r"\n{2,}", "\n", _HOLD_LINE.sub("", beat or "")).strip()


def hold_pieces(line, cast):
    """(person, [pieces]) for one `hold:` line. The person is the first word where it is a
    name on the sheet, else the only person on it; "" when neither."""
    text = str(line or "").strip()
    names = [n for n in (cast or []) if n]
    who, rest = "", text
    m = _HOLD_WHO.match(text)
    if m:
        hit = next((n for n in names if n.lower() == m.group(1).lower()), "")
        if hit:
            who, rest = hit, m.group(2)
    if not who and len(names) == 1:
        who = names[0]
    return who, [p.strip(" \t.,;") for p in rest.split(";") if p.strip(" \t.,;")]


def hold_restraints_of(piece):
    """[(canonical, part, item, position, anchor, legs)] for one `hold:` piece.

    Read with the engine's own hardware, part and position readers. No verb is needed:
    the line itself says the piece is on. The item keeps the author's words ("ball
    gag", "duct tape"); a part named in the piece wins over the hardware default, so
    "cuffs on her ankles" is on the ankles."""
    t = re.sub(r"\s+", " ", str(piece or "")).strip()
    if not t:
        return []
    lead = _HOLD_CUT.split(t, 1)[0]
    head = _REMOVE_ITEM_LEAD.sub("", lead).strip()
    head = _HOLD_MEASURE.sub("", head).strip().lower()
    pos = engine.position_in(t)
    anc = engine.anchor_in(t)
    if not anc:
        _at = engine._ANCHOR_AT.search(t)
        anc = re.sub(r"\s+", " ", _at.group(1).lower()) if _at else ""
    spans = engine.hardware_spans(t)
    own = engine.hardware_spans(head) if head else []
    # The part is read from the placement, never from the item's own name: "leg irons"
    # are on the ankles, as the hardware table has them.
    named = [_HOLD_PART_OF.get(p, p) for p, _at in engine.part_spans(t[len(lead):])]
    if not named and not own:
        named = [_HOLD_PART_OF.get(p, p) for p, _at in engine.part_spans(lead)]
    if not spans:
        if not named or not head:
            return []
        spans = [(_HOLD_BY_PART.get(named[0], "straps"), named[0], head, 0)]
        own = [spans[0]]
    out = []
    for canon, part, written, _at in spans:
        part = _HOLD_PART_OF.get(part, part)
        if (canon not in engine.PART_VARIES and named and part not in named
                and part in _HOLD_LIMBS and named[0] in _HOLD_LIMBS):
            part = named[0]
        item = head if (len(own) == 1 and own[0][0] == canon) else (written or canon)
        legs = ""
        if part in _HOLD_LEG_PARTS:
            legs = legs_anchor(t) or ("" if engine._is_rigid(item) or canon == "spreader bar"
                                      else "ankles together")
        out.append((canon, part, item or canon, pos, anc, legs))
    return out


def register_holds(state, changed, holds, cast, shot):
    """Put every piece this beat's `hold:` lines name on its person, as applied in this
    shot. ([(name, Restraint, legs, new)], [what could not be used]).

    A piece already on keeps its place in the order and takes any position or anchor
    the line gives it; a new one joins `changed["applied"]`, so everything that reads an
    application -- both ends on this shot, the named hold after -- reads it too."""
    got, problems = [], []
    applied = changed.setdefault("applied", [])
    for line in holds or []:
        who, pieces = hold_pieces(line, cast)
        if not who:
            problems.append(f"'hold: {line}' names nobody on the sheet")
            continue
        p = state.person(who)
        for piece in pieces:
            rows = hold_restraints_of(piece)
            if not rows:
                problems.append(f"'{piece}' names no restraint, gag or seal")
                continue
            for canon, part, item, pos, anc, legs in rows:
                key = next((k for k in p.hardware
                            if k[1] == part and engine._same_thing(k[0], canon)), (canon, part))
                old = p.hardware.get(key)
                # One gag, not two: the bare word beside the thing it is made of, in one
                # beat, is that thing -- the engine's own rule for prose.
                if part == "mouth" and canon in engine._MOUTH_PIECES and old is None:
                    twin = next((k for k, r in p.hardware.items() if k[1] == "mouth"
                                 and k[0] in engine._MOUTH_PIECES and r.applied_in == shot),
                                None)
                    if twin is not None and item == "gag":
                        continue
                    if twin is not None and p.hardware[twin].item == "gag":
                        _gone = p.hardware.pop(twin)
                        applied[:] = [(w, r) for w, r in applied if r is not _gone]
                if old is not None:
                    if len(item) > len(old.item or ""):
                        old.item = item
                        old.rigid = old.rigid or engine._is_rigid(item)
                    if pos and part in ("wrists", "arms"):
                        old.position = pos
                    if anc and part not in ("mouth", "eyes"):
                        old.anchor = anc
                    got.append((who, old, legs, False))
                    continue
                r = engine.Restraint(item, part, pos, anc, shot)
                p.hardware[key] = r
                applied.append((who, r))
                got.append((who, r, legs, True))
    return got, problems


def held_piece_named(state, tokens):
    """Does any `remove:` token name a piece the state holds on somebody? A hold: piece
    is in the state whatever the prose said, so its own name has to release it."""
    for q in getattr(state, "people", {}).values():
        for k, r in q.hardware.items():
            if any(names_any(r.item, [t]) or names_any(t, [k[0]]) for t in (tokens or [])):
                return True
    return False


def removal_parts(token):
    """The body parts a `remove:` token names, as the state files them: "duct tape over
    her mouth" takes the tape off the mouth and leaves tape anywhere else on."""
    rest = engine._HW_ONE.sub(" ", str(token or ""))       # "leg irons" names no part
    return {_HOLD_PART_OF.get(p, p) for p, _at in engine.part_spans(rest)}


def held_state(state, sealed=""):
    """[(name, [(item, part, position, anchor)])] for everybody the state has in a piece,
    plus ("", [(seal, "groin", "", "")]) for a seal no piece on a person carries."""
    out = []
    for n, q in getattr(state, "people", {}).items():
        rows = [(r.item, r.part, r.position, r.anchor) for r in q.hardware.values()]
        if rows:
            out.append((n, rows))
    if sealed and not any(pt == "groin" for _n, rows in out for _i, pt, _p, _a in rows):
        out.append(("", [(sealed, "groin", "", "")]))
    return out


def held_report(rows):
    """'shot 3: Mara -- handcuffs (wrists, behind the back), duct tape (mouth)' entries,
    one per person per shot, from (shot, held_state) rows. Shots before the first piece
    are left out; a shot after it with nothing on says so."""
    out, started = [], False
    for shot, people in rows:
        if not people:
            if started:
                out.append(f"shot {shot}: nothing held")
            continue
        started = True
        for name, pieces in people:
            said = ", ".join(
                f"{item} ({', '.join(x for x in (part, pos, ('fast to the ' + anc) if anc else '') if x)})"
                for item, part, pos, anc in pieces)
            out.append(f"shot {shot}: {name or 'sealed'} -- {said}")
    return out


_TIES_LIMB = re.compile(
    r"\b(?:ties|tied|tying|binds|bound|binding|restrains|restrained|restraining)\s+"
    r"(?:(?:her|his|their|him|them|[A-Z][\w'’-]*?(?:['’]s)?)\s+)?"
    r"(?:(?:\w+\s+){0,2}?(wrists?|hands?|arms?|ankles?|feet|legs?|knees?)\b|(up)\b)", re.I)


def unregistered_restraints(acted, state, changed):
    """The restraint and gag words this beat names that registered nothing: not put on,
    taken off or held on anybody once the beat is read. Reports; never acts."""
    def _canon(item):
        sp = engine.hardware_spans(str(item or ""))
        return sp[0][0] if sp else str(item or "")

    def _same(canon, part, k_canon, k_part):
        return (k_canon == canon or engine._same_thing(k_canon, canon)
                or (part in ("mouth", "eyes") and k_part == part))

    keys = [(k[0], k[1]) for q in getattr(state, "people", {}).values() for k in q.hardware]
    keys += [(_canon(r.item), r.part) for key in ("applied", "released")
             for _w, r in (changed or {}).get(key) or [] if r is not None]
    out = []
    for canon, part, written, _at in engine.hardware_spans(acted or ""):
        if not any(_same(canon, part, kc, kp) for kc, kp in keys):
            word = written or canon
            if word not in out:
                out.append(word)
    for m in _TIES_LIMB.finditer(acted or ""):
        limb = (m.group(1) or "").lower()
        group = (_HOLD_ARM_PARTS if re.match(r"wrist|hand|arm", limb)
                 else _HOLD_LEG_PARTS if limb else _HOLD_ARM_PARTS + _HOLD_LEG_PARTS)
        if not any(kp in group for _kc, kp in keys):
            word = re.sub(r"\s+", " ", m.group(0)).strip().lower()
            if word not in out:
                out.append(word)
    return out


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


# SOMETHING IN HER HANDS, while her wrists are held: "a phone in her hand", "holding a
# mug". REPORTED as a sheet line asking for a hand-held thing beside "both arms are
# behind the body" -- the model frees a hand to hold it.
_HAND_HELD = re.compile(
    r"\b(?:in\s+(?:her|his|their|one|each|both|either)\s+(?:\w+\s+)?hands?|in\s+hand|"
    r"holding|holds|clutching|clutches|gripping|grips|carrying|carries|cradling|cradles|"
    r"clasping|clasps|held\s+(?:in|up|out|against|in\s+front))\b", re.I)


def hands_free_line(line):
    """A sheet line without its hand-held things -- see _HAND_HELD. What it names as
    worn stays; a fragment that also fastens something is left as written."""
    head, sep, rest = str(line or "").partition(":")
    if not sep or not _HAND_HELD.search(rest):
        return line
    stop = rest.rstrip().endswith(".")
    kept = []
    for frag in rest.rstrip().rstrip(".").split(","):
        m = _HAND_HELD.search(frag)
        if not m or engine.hardware_spans(frag):
            kept.append(frag)
            continue
        # "grey sweater with a phone in her hand": the sweater stays.
        cuts = list(re.finditer(r"\s+(?:with|and|while)\s+", frag[:m.start()]))
        if cuts and engine.garments_in(frag[:cuts[-1].start()]):
            kept.append(frag[:cuts[-1].start()])
        tags = re.findall(r"<\s*picture\s+\d+\s*>", frag, re.I)
        if tags:
            kept.append(" " + " ".join(tags))
    return head + sep + ",".join(kept).rstrip(", ") + ("." if stop else "")


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
    # The same noise field at every beat, whatever each shot's length. See chain_noise.
    with _runtime_module._ChainNoise():
        (out,) = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                       cond, negative, latent, denoise=1.0)
    return out


# --------------------------------------------------------------------------------------
# POSE CONTROL. A second sampling pass, held to a skeleton video, on the shots where a
# restrained person's limbs need holding: cuffed wrists that come out to catch a fall,
# tied ankles that walk apart. pose_control.py holds the geometry, the detector and the
# patch install; this is the per-shot decision and the two passes. See the README.
# --------------------------------------------------------------------------------------
POSE_SHOT_MODES = ("repair broken shots", "every restrained shot", "bound falls only")
POSE_DRAWS = ("everyone", "everyone, thick lines", "bound person only")
_POSE_MODE_KEY = {"repair broken shots": "repair", "every restrained shot": "every",
                  "bound falls only": "falls"}
# The limb positions the skeleton can hold. "out to the sides" and "ankles to the neck"
# are left as pass 1 drew them.
POSE_ARMS = tuple(getattr(pose_control, "ARMS_POSITIONS", None)
                  or ("behind the back", "in front of the body", "at the waist",
                      "above the head"))
POSE_LEGS = tuple(getattr(pose_control, "LEGS_POSITIONS", None)
                  or ("ankles together", "held apart", "ankles to the wrists"))
POSE_DETECT_STRIDE = 2          # DWPose on every 2nd frame; the hint interpolates between
# How far apart "ankles together" lets the ankles sit, in torso lengths: a chain, bar or
# hobble between them holds them a stride's fraction apart; rope or tape holds them close.
POSE_GAP_LINKED = 0.6
POSE_GAP_TIED = 0.12
_POSE_LINKED_LEGS = re.compile(r"iron|shackle|chain|manacle|hobble", re.I)
# Latch mode, for a shot where a restraint goes on: the limbs it puts on are drawn as
# detected until they settle into the held shape, then held. Applying wording closes the
# piece by mid-shot, so the latch is searched from this share of the frames on.
POSE_LATCH_FROM = 0.4
POSE_LATCH_NEEDS_UPDATE = "latch mode needs the updated pose_control.py"


def pose_wearer_facts(arms, legs, anchored=False, leg_items=(), fall=False, latch=()):
    """One wearer's entry in Shot.bound_pose, from the planner's own words for the
    position. A position fastened to an object (", at the headboard", or "at the pipe"
    alone) is anchored; "at the waist" is a position, not an object. `latch` is the
    limbs ("arms", "legs") this shot puts on them."""
    where, point = limb_anchor_parts(arms)
    return {"arms": where,
            "legs": str(legs or "").strip(),
            "ankle_gap": (POSE_GAP_LINKED
                          if any(_POSE_LINKED_LEGS.search(str(i or "")) for i in leg_items or ())
                          else POSE_GAP_TIED),
            "anchored": bool(anchored or point),
            "fall": bool(fall),
            "latch_limbs": tuple(x for x in ("arms", "legs") if x in (latch or ()))}


def pose_held_limbs(facts):
    """The limbs of one bound_pose entry that the skeleton can hold."""
    f = facts or {}
    return tuple(x for x, ok in (("arms", f.get("arms") in POSE_ARMS),
                                 ("legs", f.get("legs") in POSE_LEGS)) if ok)


def restrains_in(beat, name):
    """Is `name` the one putting a restraint on in this beat: the name in front of an
    applying verb, in the -s forms restraint_going_on reads? Not a possessive ("Mara's
    wrists")."""
    if not name:
        return False
    b = _worn_masked(beat or "")
    return bool(re.search(
        r"\b" + re.escape(name) + r"\b(?!['’]s)" + _UP_TO_TWO_WORDS
        + r"\s+(?:" + _APPLY_NOW.pattern + r"|" + _APPLY_PHRASE.pattern + r")", b, re.I))


def pose_takes(fn_name, keyword):
    """Does the loaded pose_control.<fn_name> take `keyword`? Read off its signature, so
    an older pose_control.py is never called with a keyword it does not know."""
    fn = getattr(pose_control, fn_name, None)
    if fn is None:
        return False
    try:
        return keyword in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def pose_latch_ready():
    """Does the loaded pose_control.build_hint take latch_after?"""
    return pose_takes("build_hint", "latch_after")


def pose_describe(bound):
    """'Mara: arms behind the back, ankles together' for info, per bound person."""
    out = []
    for nm in sorted(bound or {}):
        b = bound[nm] or {}
        bits = [f"arms {b['arms']}" if b.get("arms") in POSE_ARMS else "",
                b.get("legs") if b.get("legs") in POSE_LEGS else "",
                "falling" if b.get("fall") else ""]
        out.append(f"{nm}: " + ", ".join(x for x in bits if x))
    return "; ".join(out)


def pose_candidate(shot, opening_cast, mode):
    """(bound, why) for one planned shot. bound is {name: facts} for build_hint when the
    shot gets pose control; otherwise {} and why it does not -- "" for a shot with no
    held arm or leg position in it, which says nothing.

    A shot whose limb hardware comes off is left alone: the limbs are free for part of
    it. One where it goes on is a candidate in latch mode -- the limbs it puts on
    (latch_limbs) are drawn as detected until they settle, then held; build_hint finds
    that frame. Left alone too: a body fastened to an object (the skeleton cannot draw
    it), a wearer the beat has doing the restraining (Shot.restrainers: the planner can
    read the captor as the one cuffed when "her" fits two people), and a wearer who is
    not in the opening frame (identification reads the first frames)."""
    facts = getattr(shot, "bound_pose", None) or {}
    held = {nm: dict(f) for nm, f in facts.items()
            if pose_held_limbs(f) or (f or {}).get("anchored")}
    if not held:
        return {}, ""
    if getattr(shot, "limbs_off", False):
        return {}, "restraints come off in it"
    held = {nm: f for nm, f in held.items() if not f.get("anchored")}
    if not held:
        return {}, "the restrained person is fastened to an object"
    doers = set(getattr(shot, "restrainers", None) or ())
    held = {nm: f for nm, f in held.items() if nm not in doers}
    if not held:
        return {}, "the one doing the restraining is read as the restrained person"
    cast = [n for n in (opening_cast or []) if n]
    if cast:
        held = {nm: f for nm, f in held.items() if nm in cast}
        if not held:
            return {}, "the restrained person is not in the opening frame"
    if _POSE_MODE_KEY.get(mode, "repair") == "falls" and not getattr(shot, "bound_fall", False):
        return {}, "no bound fall in it"
    for f in held.values():
        f["latch_limbs"] = tuple(x for x in (f.get("latch_limbs") or ())
                                 if x in pose_held_limbs(f))
    if any(f["latch_limbs"] for f in held.values()) and not pose_latch_ready():
        return {}, POSE_LATCH_NEEDS_UPDATE
    return held, ""


def pose_latched(bound):
    """Does any bound person in this candidate latch?"""
    return any((f or {}).get("latch_limbs") for f in (bound or {}).values())


# comfy's KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS, for a comfy that does not say.
_DISCARD_PENULTIMATE = ("dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2")


def pose_shot_schedule(model, sigmas, scheduler, steps, sampler_name=None,
                       soft_landing=False, shift_video=None, shift_audio=None):
    """The sigmas of this shot's STEPS, the schedule pose_end is a share of: the wired (or
    Hyperflow) schedule, else the scheduler's own -- built as sample_shot and
    landing_schedule build it. None when none can be read.

    The soft landing is left out on purpose. It splices one extra model call between the
    last step and 0 (insert_audio_landing) and changes no sigma before it, so the window
    read here holds at the same sigmas on the landed schedule. Counted on the landed
    schedule instead, 0.6 of 8 steps came out as 6 of 9 calls -- one more than the "first
    5 of 8 steps" info reports. At pose_end 1.0 the window has no lower bound, so the
    landing call is held too.

    A shot that goes through common_ksampler (no landing spliced in) runs on KSampler's
    schedule, which for dpm_2 and uni_pc is built one step longer with the penultimate
    sigma dropped (comfy/samplers.py, KSampler.calculate_sigmas); read the same way here,
    or the window is a step off."""
    if sigmas is not None and len(sigmas):
        return sigmas
    try:
        import comfy.samplers as _cs
        ms = model.get_model_object("model_sampling")
        landed = bool(soft_landing) and landing_schedule(
            model, scheduler, steps, shift_video, shift_audio) is not None
        drop = set(getattr(getattr(_cs, "KSampler", None), "DISCARD_PENULTIMATE_SIGMA_SAMPLERS",
                           None) or _DISCARD_PENULTIMATE)
        if landed or str(sampler_name or "") not in drop:
            return _cs.calculate_sigmas(ms, str(scheduler), int(steps))
        sched = _cs.calculate_sigmas(ms, str(scheduler), int(steps) + 1)
        return torch.cat([sched[:-2], sched[-1:]])
    except Exception:
        return None


def _pose_latent_shape(latent):
    """The VIDEO latent's shape (1, 24, T, h/16, w/16): the hint must encode to it."""
    s = latent["samples"]
    if getattr(s, "is_nested", False):
        s = s.unbind()[0]
    return tuple(int(x) for x in s.shape)


def _pose_interrupted(e):
    """comfy's interrupt is an Exception; it always propagates."""
    return type(e).__name__ == "InterruptProcessingException"


def _pose_device():
    try:
        return mm.get_torch_device()
    except Exception:
        return None


def pose_sample_shot(model, cond, negative, latent, seed, steps, cfg, sampler_name,
                     scheduler, sigmas, shift_video, shift_audio, soft_landing, *,
                     pose_cn, vae, detector, bound, mode, draw, strength, pose_end,
                     frame_count, w, h, tiled, carry=None, cast_count=None,
                     keep_decoded=False, audio_vae=None, latch_after=None,
                     carry_appearance=None):
    """(out, decoded pass-1 frames or None, report) for one candidate shot.

    Pass 1 is sample_shot, as every shot runs. Its frames are decoded at the sampled size
    and read by DWPose; build_hint finds the restrained person, rewrites their limbs into
    the held shape in their own torso frame and draws everybody as a skeleton video;
    pass 2 samples again with the same seed, conditioning and latent (so the same noise
    field) on a clone carrying comfy's H3 Fun control patch, held to the skeleton for the
    first pose_end of the steps.

    Nothing after pass 1 ends a render except an interrupt: any failure keeps pass 1 and
    says why. The decoded frames come back only when pass 1 is kept and `keep_decoded`,
    so the caller can skip its own decode of the same latent.

    `latch_after` (a frame index) is for a shot where a restraint goes on: build_hint
    holds the limbs in each person's latch_limbs only from the first analysed frame at or
    after it where they settle into the held shape. Passed only when set, so an older
    build_hint is never handed a keyword it does not take. `carry_appearance` ({name:
    appearance vector} from the shot before) is passed the same way, so a carried name
    only lands on a person who looks like them.

    report: {"outcome": "repaired" | "held" | "checked" | "skipped" | "oom" | "failed",
    "why", "boxes" {name: torso box at the last analysed frame}, "broken", "window"
    (s_start, s_end), "off" (True: stop pose control for the rest of the run), "notes",
    "latched" {name: frame index or None}, "latch" (latch_after was given),
    "appearance" {name: appearance vector at the last analysed frame}}."""
    report = {"outcome": "skipped", "why": "", "boxes": {}, "broken": False,
              "window": None, "off": False, "notes": [], "t_pass1": 0.0,
              "latched": {}, "latch": latch_after is not None, "appearance": {}}
    _t0 = time.perf_counter()
    out = sample_shot(model, cond, negative, latent, seed, steps, cfg, sampler_name,
                      scheduler, sigmas, shift_video, shift_audio, soft_landing)
    report["t_pass1"] = time.perf_counter() - _t0
    shot_model = None
    try:
        # A second pass may follow: the DiT's host copy and the audio VAE are kept, so
        # the RAM guard frees them last, not first. The DiT may still leave the card for
        # the decode, as at the shot's own decode -- it comes back from RAM, not disk.
        ensure_host_ram(_decode_ram(vae, out, tiled), keep=(vae, audio_vae, model),
                        what="the pose check's decode")
        frames = _decode_video(vae, out, tiled, free_first=model, keep=(vae, audio_vae))
        try:
            det = detector.detect(frames, stride=POSE_DETECT_STRIDE)
        except Exception as e:
            if _pose_interrupted(e):
                raise
            report.update(off=True, why=f"the pose estimator failed ({type(e).__name__}: "
                                        f"{e}); pose control is off for the rest of the run")
            return out, (frames if keep_decoded else None), report
        finally:
            try:
                detector.close()        # back to the CPU: the DiT needs the card again
            except Exception:
                pass
        _kw = {"cast_count": cast_count}
        if latch_after is not None:
            _kw["latch_after"] = int(latch_after)
        if carry_appearance and pose_takes("build_hint", "carry_appearance"):
            _kw["carry_appearance"] = dict(carry_appearance)
        try:
            hint, rep = pose_control.build_hint(
                det, int(frame_count), int(h), int(w), bound, carry=carry,
                mode=_POSE_MODE_KEY.get(mode, "repair"), draw=draw, **_kw)
        except TypeError as e:
            if latch_after is None or "latch_after" not in str(e):
                raise
            report["why"] = POSE_LATCH_NEEDS_UPDATE
            return out, (frames if keep_decoded else None), report
        rep = rep or {}
        report["boxes"] = dict(rep.get("boxes_last") or {})
        report["broken"] = bool(rep.get("broken"))
        report["notes"] = [str(n) for n in (rep.get("notes") or [])]
        report["latched"] = dict(rep.get("latched") or {})
        report["appearance"] = dict(rep.get("appearance_last") or {})
        if hint is None:
            if rep.get("skipped"):
                report["why"] = str(rep["skipped"])
            else:
                report["outcome"] = "checked"
            return out, (frames if keep_decoded else None), report
        del frames
        shape = _pose_latent_shape(latent)
        hint_latent = pose_control.encode_hint(vae, hint, shape)
        del hint
        _deep_cleanup()
        if hint_latent is None:
            report["why"] = "the skeleton video did not encode to the shot's latent shape"
            return out, None, report
        sched = pose_shot_schedule(model, sigmas, scheduler, steps, sampler_name,
                                   soft_landing, shift_video, shift_audio)
        if sched is None or len(sched) < 2:
            report["why"] = "the shot's sigma schedule could not be read"
            return out, None, report
        s_start, s_end = pose_control.pose_sigma_window(sched, pose_end)
        report["window"] = (float(s_start), float(s_end))
        # On a clone of the model as it stands here: after the schedule patch, the stamp
        # and FastH3's VSA, so the Fun block patch wraps VSA's as `previous` (the other
        # order lets VSA overwrite the control on those blocks).
        shot_model = pose_control.install_pose_control(model, pose_cn, vae, hint_latent,
                                                       shape, strength, s_start, s_end)
        del hint_latent
    except Exception as e:
        if _pose_interrupted(e):
            raise
        _deep_cleanup()
        report["why"] = ("ran out of memory building the skeleton video" if _is_oom(e)
                         else f"pose control failed on this shot ({type(e).__name__}: {e})")
        return out, None, report
    failed, oom = "", False
    try:
        _evict_all_but(model, latent)
        out2 = sample_shot(shot_model, cond, negative, latent, seed, steps, cfg,
                           sampler_name, scheduler, sigmas, shift_video, shift_audio,
                           soft_landing)
    except Exception as e:
        if _pose_interrupted(e):
            raise
        oom, failed = _is_oom(e), f"{type(e).__name__}: {e}"
    finally:
        shot_model = None
    if failed:
        # Cleaned up here, past the except: the exception's frames held the failed
        # pass's tensors while it was alive.
        if oom:
            _pose_oom_cleanup()
            report.update(outcome="oom", off=True,
                          why="the pose pass ran out of VRAM; kept the uncontrolled "
                              "render, and pose control is off for the rest of the run")
        else:
            _deep_cleanup()
            report.update(outcome="failed", off=True,
                          why=f"the pose pass failed ({failed}); kept the uncontrolled "
                              f"render, and pose control is off for the rest of the run")
        return out, None, report
    _deep_cleanup()
    report["outcome"] = "repaired" if report["broken"] else "held"
    return out2, None, report


def _pose_oom_cleanup():
    """After an out-of-memory error in the pose pass, before the render goes on in the same
    node run (comfy cleans up only when a node ends): what the failed pass left is
    collected, then the CUDA caches go -- comfy's soft_empty_cache, via _deep_cleanup, the
    node's cleanup after any OOM."""
    import gc
    gc.collect()
    _deep_cleanup()


def pose_handoff_boxes(detector, frame, boxes, w, h, appearance=None):
    """{name: torso box} of the restrained people in the handoff frame -- the frame the
    next shot opens on -- on the w x h grid the next shot is sampled on, or {}.

    `boxes` are this shot's last analysed torso boxes by name; the handoff frame's people
    are matched to them with the carry margins, so a name is carried only where it is
    clear. `appearance` ({name: appearance vector}) also has to agree when the loaded
    pose_control can compare it. One frame of DWPose."""
    if detector is None or frame is None or not boxes:
        return {}
    try:
        try:
            det = detector.detect(frame, stride=1)
        finally:
            try:
                detector.close()
            except Exception:
                pass
        people = list(((det or {}).get("people") or [[]])[-1] or [])
        fh, fw = int(frame.shape[1]), int(frame.shape[2])
        if fw != int(w) or fh != int(h):
            sx, sy = float(w) / max(1, fw), float(h) / max(1, fh)
            scaled = []
            for p in people:
                q = p.copy()
                q[:, 0] *= sx
                q[:, 1] *= sy
                scaled.append(q)
            people = scaled
        _kw = {}
        looks = list(((det or {}).get("appearance") or [[]])[-1] or [])
        if (appearance and looks and len(looks) == len(people)
                and pose_takes("identify_by_boxes", "carry_appearance")):
            _kw = {"appearance": looks, "carry_appearance": dict(appearance)}
        ids = pose_control.identify_by_boxes(people, boxes, **_kw)
        return dict(pose_control.torso_boxes(people, ids) or {})
    except Exception as e:
        if _pose_interrupted(e):
            raise
        return {}


def pose_shot_line(n, report, bound):
    """The info line for one analysed shot, with what the check saw after it -- which
    limbs broke and in which frames, or how much of the shot held -- since every
    threshold behind it is an estimate to be tuned on real renders."""
    outcome, why = report.get("outcome"), report.get("why") or ""
    seen = "; ".join(str(x) for x in (report.get("notes") or []) if x)
    seen = f" -- {seen}" if seen else ""
    latched = {nm: f for nm, f in (report.get("latched") or {}).items() if f is not None}
    if outcome in ("repaired", "held") and report.get("latch") and latched:
        at = (f"at frame {next(iter(latched.values()))}" if len(latched) == 1 else
              "at frames " + ", ".join(f"{f} ({nm})" for nm, f in sorted(latched.items())))
        return f"shot {n}: pose latched ({pose_describe(bound)}) {at}{seen}"
    if outcome == "repaired":
        return f"shot {n}: pose repaired ({pose_describe(bound)}){seen}"
    if outcome == "held":
        return f"shot {n}: pose held ({pose_describe(bound)}){seen}"
    if outcome == "checked":
        return f"shot {n}: pose checked, nothing broken{seen}"
    if outcome == "oom":
        return (f"shot {n}: pose pass ran out of VRAM, kept the uncontrolled render; pose "
                f"control is off for the rest of the run")
    return f"shot {n}: pose skipped -- {why or 'nothing to hold'}"



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
    "pose_strength": (1.0, 0.0, 2.0, float),
    "pose_end": (0.6, 0.1, 1.0, float),
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
                               "what happens. `exactly:` and `verbatim:` do the same thing.\n\n"
                               "A RESTRAINT, GAG OR SEAL THAT MUST STAY ON goes on its own line "
                               "under the beat that puts it on:\n"
                               "  hold: Mara, handcuffs behind her back; duct tape over her mouth\n\n"
                               "The person, then each piece with where it is, split by ';'. From "
                               "that beat on each piece is registered on that person whatever "
                               "the beat's wording: that shot is told it goes on, every later "
                               "shot that it stays. 'remove: duct tape' under a later beat takes "
                               "that piece off and leaves the rest on. With one person on the "
                               "sheet the name can be left out. The line is read, never sent. "
                               "info lists what is registered on each shot, and names any shot "
                               "whose beat mentions a restraint or gag that registered nothing."}),
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
                    "tooltip": "One seed for the whole chain. Its noise is drawn frame by "
                               "frame, so every shot shares one noise field whatever its "
                               "length."}),
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
                    "steps and scheduler are then only for the progress bar. Leave it "
                    "unwired for a Hyperflow LoRA: the node reads the grid and shifts from "
                    "the LoRA file and applies them itself."}),
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
                               "'fixed' gives every shot shot_seconds. Either way the "
                               "chain shares one noise field: the seed's noise is drawn "
                               "frame by frame, so a shot's length does not change it.\n\n"
                               "The estimate leans short on purpose: a shot that ends before "
                               "its action does hands a mid-motion frame to the next shot, "
                               "which the chain continues from. A shot that outlasts its "
                               "action has to invent the rest. Except a shot that puts a "
                               "restraint, gag or seal on: it gets one more action's time "
                               "and is not leaned short, because its last frame is what the "
                               "next shot opens on and has to show the piece in place."}),
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
                "keep_frame_after_removal": ("BOOLEAN", {"default": True,
                    "tooltip": "After a shot that takes something off, the NEXT shot still "
                               "opens on that shot's last frame -- the same camera, the same "
                               "place, the same restraints, carried as a picture.\n\n"
                               "Off, it restarts there instead: the frame rides only as a "
                               "reference (regular H3) and the shot re-derives its framing, "
                               "so the camera angle changes and whatever the text does not "
                               "restate is re-imagined. That is insurance against a garment "
                               "the model did not finish taking off being inherited through "
                               "the keyframe -- and it cost a cut and the scene's memory at "
                               "every removal. REPORTED as camera angles changing between "
                               "beats and beats losing what was in the one before, so it is "
                               "on by default now. (Renamed from restart_after_removal with "
                               "its meaning flipped, so a saved workflow's old 'true' "
                               "restores as 'keep the frame'.) FastH3 always keeps the "
                               "frame: it cannot read a reference, so a restart there is a "
                               "start from nothing."}),
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
                    "tooltip": "Take the burn the chain adds to itself back out of every "
                               "shot.\n\n"
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
                               "Measured per colour channel from each shot alone, in two "
                               "parts: the handoff against the model's reproduction of it at "
                               "frame one, where nothing was asked to change, and frame one "
                               "against the LAST frame, which is where a distill cooks a few "
                               "percent over every take. The shot's own frames are graded on "
                               "a ramp from frame one to the last, so the last frame -- which "
                               "is the next shot's handoff -- carries the whole correction: "
                               "nothing is left to compound, and there is no step at the "
                               "cut.\n\n"
                               "It never aims at a target and never compares a shot to shot "
                               "1. A beat that changes the light on purpose -- a lamp "
                               "switched off, curtains drawn, the sun setting -- or travels "
                               "to another place, or moves the camera, keeps what its take "
                               "did, and so does any change too large to be cooking.\n\n"
                               "1.0 takes all of each shot's own drift out; lower leaves a "
                               "share of it in, and that share still compounds slowly. 0 is "
                               "off. Watch the contrast line in info: if it still says UP, "
                               "raise this. It cannot undo clipping already baked in, and it "
                               "corrects levels only -- not softening, and nothing "
                               "spatial."}),
                # APPENDED. Saved workflows restore widget values by position.
                # APPENDED. Saved workflows restore widget values by position.
                # APPENDED: pose control. The socket takes no widget slot; the four
                # widgets after it follow hold_levels in this order.
                "pose_controlnet": ("MODEL_PATCH", {
                    "tooltip": "Holds a restrained character's arms and legs in place with "
                               "a skeleton. Wire a Load Model Patch node here with the "
                               "MiniMax H3 Fun ControlNet "
                               "(minimax_h3_fun_controlnet_union_pruned_int8_convrot, in "
                               "models/model_patches). Unwired, nothing changes.\n\n"
                               "It works only with the hybrid b25-49 checkpoint "
                               "(minimax_h3_hybrid_fl2va_ref2va_b25-49): the controlnet is "
                               "built for its 8-wide timestep table, and on any other base "
                               "the node turns it off and says why in info. It also needs "
                               "the DWPose files of comfyui_controlnet_aux, and "
                               "hold_restraints on.\n\n"
                               "On a shot where someone is restrained, the shot renders "
                               "as usual, DWPose reads the people in it, the restrained "
                               "person's elbows, wrists, knees and ankles are redrawn in "
                               "the held position, and the shot renders a second time "
                               "with the same seed, following that skeleton for its first "
                               "steps. Only shots that render twice cost twice the "
                               "time."}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0,
                    "step": 0.05,
                    "tooltip": "How strongly the second pass follows the skeleton. 1.0 is "
                               "the controlnet's own scale; raise it if the arms keep "
                               "moving, lower it if the body stiffens. 0 turns pose control "
                               "off even with the controlnet wired."}),
                "pose_end": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0,
                    "step": 0.05,
                    "tooltip": "The share of the steps that follow the skeleton, from the "
                               "first: 0.6 of 8 steps is the first 5. The remaining steps "
                               "run free, so hands, cuffs and faces get their detail from "
                               "the model rather than from the stick figure. 1.0 holds "
                               "every step."}),
                "pose_shots": (list(POSE_SHOT_MODES), {"default": POSE_SHOT_MODES[0],
                    "tooltip": "Which shots get the second pass.\n\n"
                               "repair broken shots: every shot with a restrained person "
                               "renders once and is checked. Only a shot where the held "
                               "limbs come apart -- wrists out to catch a fall, a reach, "
                               "ankles stepping apart -- or where a bound body falls, "
                               "renders again with the skeleton. A shot that holds keeps "
                               "its first render.\n\n"
                               "every restrained shot: every such shot renders twice.\n\n"
                               "bound falls only: only shots where a restrained body "
                               "falls.\n\n"
                               "Shots where the restraints go on or come off, and bodies "
                               "fastened to an object, are left as rendered. info lists "
                               "what happened on each shot."}),
                "pose_draw": (list(POSE_DRAWS), {"default": POSE_DRAWS[0],
                    "tooltip": "Who is drawn in the skeleton.\n\n"
                               "everyone: the restrained person in the held position, and "
                               "everyone else as the first render moved them, so the "
                               "second render keeps their motion.\n\n"
                               "everyone, thick lines: the same with heavier lines, for "
                               "when the skeleton is followed too loosely.\n\n"
                               "bound person only: only the restrained person; the others "
                               "move freely."}),
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
            keep_frame_after_removal=True, auto_remove=True, anchor="", character_memory="",
            pace=1.0,
            ambient_audio=None, ambient_level=0.25, foley_level=0.35,
            speech_lead_seconds=0.5, speech_tail_seconds=2.0,
            hold_levels=0.8, pose_controlnet=None, pose_strength=1.0, pose_end=0.6,
            pose_shots=POSE_SHOT_MODES[0], pose_draw=POSE_DRAWS[0],
            graph=None, restart_after_removal=None,
            **_removed):

        self._frames = None
        # The old input, by name (an API prompt), still means what it said.
        restart_after_removal = (not keep_frame_after_removal
                                 if restart_after_removal is None
                                 else bool(restart_after_removal))
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
            hold_levels=hold_levels, pose_controlnet=pose_controlnet,
            pose_strength=pose_strength, pose_end=pose_end, pose_shots=pose_shots,
            pose_draw=pose_draw, graph=graph,
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
                self._release_pose_detector()
        return prepared

    def _release_pose_detector(self):
        """Drop the DWPose models a render loaded, however the render ended."""
        _d, self._pose_detector = getattr(self, "_pose_detector", None), None
        if _d is not None:
            try:
                (getattr(_d, "release", None) or _d.close)()
            except Exception:
                pass                    # teardown must not mask what ended the render

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
            hold_levels=0.8, pose_controlnet=None, pose_strength=1.0, pose_end=0.6,
            pose_shots=POSE_SHOT_MODES[0], pose_draw=POSE_DRAWS[0], graph=None,
            **_removed):

        notes = []
        _bad = misaligned_widgets(
            dict(resolution=resolution, sampler_name=sampler_name, scheduler=scheduler,
                 shot_length=shot_length, upscale=upscale, latent_upscale=latent_upscale,
                 upscale_model=upscale_model, pose_shots=pose_shots, pose_draw=pose_draw),
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
            speech_tail_seconds=speech_tail_seconds, hold_levels=hold_levels,
            pose_strength=pose_strength, pose_end=pose_end))
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
        pose_strength, pose_end = _fixed["pose_strength"], _fixed["pose_end"]
        notes.extend(_fixnotes)
        # FASTH3: a distill runs as it was distilled. See fast_h3.
        _fast = fast_h3(model)
        # A restart carries the frame as a REFERENCE, which FastH3 cannot read -- so
        # there it was a start from nothing: no room, no restraints, no clothes but
        # what the text restates, and a new camera. Never on FastH3.
        restart_after_removal = bool(restart_after_removal) and not _fast
        if _fast:
            _fast_said = []
            if float(shift_video) == 12.0:       # the node's default, i.e. untouched
                shift_video = FAST_H3_SHIFT_VIDEO
                _fast_said.append(f"sigma shift set to {FAST_H3_SHIFT_VIDEO:g} video / "
                                  f"{float(shift_audio):g} audio, FastH3's own")
            elif (float(shift_video), float(shift_audio)) != (FAST_H3_SHIFT_VIDEO,
                                                               FAST_H3_SHIFT_AUDIO):
                _fast_said.append(f"your sigma shift is {float(shift_video):g}/"
                                  f"{float(shift_audio):g}; FastH3 is distilled at "
                                  f"{FAST_H3_SHIFT_VIDEO:g}/{FAST_H3_SHIFT_AUDIO:g}")
            if int(steps) != FAST_H3_STEPS:
                _fast_said.append(f"steps is {int(steps)}; FastH3 V2 is distilled for "
                                  f"exactly {FAST_H3_STEPS}, and any other count degrades it")
            if (sampler_name, scheduler) != ("res_multistep", "simple"):
                _fast_said.append(f"it runs on res_multistep with the simple scheduler, "
                                  f"not {sampler_name}/{scheduler}")
            notes.append(
                "FastH3 model detected (FastVideo's 8-step distill). It distilled first/"
                "last-frame generation only -- ref2va was not distilled -- so a reference "
                "row is a picture it was never taught to read, and it can draw one as "
                "another person. The pictures YOU tag with <Picture N> are still sent, "
                "because that is your choice; every picture the node would add on its "
                "own is not -- no recovered face, evened face, returning room or frame "
                "carried across a cut, and an untagged picture is not claimed onto "
                "anybody. A shot that would have carried the last frame as a reference "
                "starts fresh, and the handoff always stays a keyframe. If a tagged "
                "person still doubles, the tag is the cause -- for reference-driven "
                "likeness render with a base or hybrid H3. No extra audio step is spliced "
                "into the schedule either -- it changes "
                "the step count the model was distilled for. It runs with the VSA "
                "attention it was trained on (keep 10%, from 20% of the schedule): the "
                "node applies ComfyUI's own at render time, unless a Model Sparse "
                "Attention node is already wired in front of it"
                + ("; " + "; ".join(_fast_said) if _fast_said else ""))
        # HYPERFLOW: the grid it was distilled on, shifted as it was trained, with the
        # shifts stamped where the DiT reads them. See hyperflow_lora.
        _hyper = hyperflow_lora(model, graph)
        if _hyper and _fast:
            notes.append("a Hyperflow LoRA is on a FastH3 checkpoint. Both are 8-step "
                         "distills of different models, and FastH3's schedule is the one "
                         "kept; Hyperflow's grid is not applied. Use Hyperflow on a base H3")
            _hyper = None
        if _hyper:
            _grid = _hyper["sigmas"]
            _n = len(_grid) - 1
            _hf_said = []
            if sigmas is not None and len(sigmas):
                _yours = [round(float(x), 4) for x in sigmas]
                _raw = [round(float(x), 4) for x in _grid]
                _hf_said.append(
                    "your own `sigmas` input is wired, so it drives the sampler and "
                    "Hyperflow's grid is not applied"
                    + (" -- and what is wired is the grid UNSHIFTED, so the video runs "
                       "most of its steps at the bottom of the noise. Unwire it and the "
                       "node runs the grid shifted, as the LoRA was trained"
                       if _yours == _raw else ""))
            else:
                if (float(shift_video), float(shift_audio)) not in (
                        (12.0, 3.0), (_hyper["shift_video"], _hyper["shift_audio"])):
                    _hf_said.append(f"your sigma shift {float(shift_video):g}/"
                                    f"{float(shift_audio):g} is replaced by the "
                                    f"{_hyper['shift_video']:g}/{_hyper['shift_audio']:g} "
                                    f"it was distilled at")
                shift_video = _hyper["shift_video"]
                shift_audio = _hyper["shift_audio"]
                sigmas = hyperflow_sigmas(_grid, shift_video)
                model = stamp_h3_shift(model, shift_video, shift_audio)
                if int(steps) != _n:
                    _hf_said.append(f"steps is {_n}, the grid's own, not {int(steps)}")
                steps = _n
            notes.append(
                f"Hyperflow LoRA detected (read from {_hyper['source']}). It is distilled "
                f"onto a fixed {_n}-step grid, so every shot samples on that grid through "
                f"the video shift {_hyper['shift_video']:g} -- "
                + ", ".join(f"{float(x):.3f}" for x in hyperflow_sigmas(
                    _grid, _hyper["shift_video"]))
                + f" -- with the audio branch on shift {_hyper['shift_audio']:g}, both "
                f"stamped into the model so the DiT derives the audio timesteps from "
                f"the same numbers. scheduler only labels the progress bar now, and no "
                f"extra audio step is spliced in: it would change the grid"
                + ("; " + "; ".join(_hf_said) if _hf_said else ""))
            # EULER, ALWAYS. Each grid step is one jump the LoRA learned to make, and its
            # endpoint is the next grid point: a multistep sampler blends in the jump
            # before it, and a two-stage one evaluates between grid points, where no
            # step was ever distilled.
            if sampler_name != HYPERFLOW_SAMPLER:
                notes.append(f"sampler set to {HYPERFLOW_SAMPLER} for Hyperflow, from "
                             f"{sampler_name}: each step is one jump to the next grid "
                             f"point, and that is the only sampler that makes exactly those")
                sampler_name = HYPERFLOW_SAMPLER
            _hyper["two_time"], _tt_said = hyperflow_two_time_plan(model, _hyper)
            notes.append(_tt_said)
        # POSE CONTROL: whether it can run on this model at all, said here so plan_only
        # says it too. Unwired or strength 0 is off and silent. See pose_status.
        _pose_ok, _pose_note = False, ""
        if pose_controlnet is not None and float(pose_strength) > 0.0:
            if not hold_restraints:
                _pose_note = ("pose control off: hold_restraints is off, so no restraint is "
                              "held and there is no position to draw")
            elif pose_control is None:
                _pose_note = (f"pose control off: pose_control.py did not load "
                              f"({_POSE_LOAD_ERROR})")
            else:
                try:
                    _pose_ok, _pose_note = pose_control.pose_status(
                        model, pose_controlnet, pose_strength,
                        bool(_hyper and _hyper.get("two_time")))
                except Exception as e:      # a setup check never takes a render down
                    _pose_ok, _pose_note = False, (f"pose control off: the setup check "
                                                   f"failed ({type(e).__name__}: {e})")
        if _pose_ok:
            # Counted as pose_shot_schedule counts: a wired (or Hyperflow) schedule's own
            # steps, else the steps widget.
            _pose_n = (len(sigmas) - 1 if (sigmas is not None and len(sigmas) > 1)
                       else int(steps))
            _pose_k = max(1, min(_pose_n,
                                 int(math.ceil(float(pose_end) * _pose_n - 1e-9))))
            notes.append(
                f"pose control: on; strength {float(pose_strength):.2f}; the first {_pose_k} "
                f"of {_pose_n} steps follow the skeleton; {pose_shots}; draw {pose_draw}"
                + (f"; {_pose_note}" if _pose_note else ""))
        elif _pose_note:
            notes.append(_pose_note)
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
        # AS SHIPPED. Both of these were widgets defaulting to True and 4, and
        # both were briefly replaced by a measurement of free VRAM. Tiling is not a
        # memory question this node gets to re-answer: the whole-clip decode is the
        # largest allocation in a run, the project defaulted to tiled for that
        # reason, and the decoded frames are what the NEXT shot's keyframe is taken
        # from -- so changing how the decode runs changes the chain that carries a
        # room from one beat to the next. Reported as scenery rearranging between
        # beats: a TV gone, a door arrived.
        tiled_decode = True
        upscale_batch = 4
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
        # REPORTED, NOT ACTED ON. This briefly decided apply_model_sampling: an
        # upstream stamp meant the node stood down and let the other node's shifts run
        # instead of its own. That is the node changing the SCHEDULE on its own
        # initiative, which is the same class of mistake as the shift correction that
        # sat below this and broke every distilled LoRA. So it patches unconditionally,
        # the way it shipped, and the detection only says what it found.
        apply_model_sampling = True
        _upstream = upstream_h3_shift(model)
        if _upstream is not None:
            notes.append(
                f"the H3 schedule is ALSO SET UPSTREAM (video {_upstream[0]:g}"
                + (f"/audio {_upstream[1]:g}" if _upstream[1] else "")
                + f"), and this node applies its own {shift_video:g}/{shift_audio:g} over "
                  f"it -- to the sampler's schedule AND to the stamp the DiT reads its "
                  f"shifts from, so the two always agree and the node's numbers are the "
                  f"ones that run. (They used to disagree: the sampler took the node's "
                  f"shifts while the DiT kept the upstream ones.) Read from the stamp "
                  f"comfy's MiniMaxH3SigmaShift leaves in transformer_options. Set "
                  f"shift_video/shift_audio here rather than upstream")
        # SHIFT IS NOT THIS NODE'S TO CORRECT. What stood here solved shift_video
        # down from H3's 12 so the final step cleared less noise -- 4.66 at 8 steps,
        # 2.0 at 4 -- on the reasoning that a schedule leaving 0.63 for one evaluation
        # cannot resolve structure in it.
        #
        # THAT REASONING IS WRONG IN FRONT OF A DISTILLED LORA, which is the only place
        # the correction ever fired. A turbo LoRA distilled at 8 steps is TRAINED to
        # cross that 0.63 in one evaluation; the big final jump is not a defect in the
        # schedule, it is the thing distillation buys. Moving shift_video to 4.66 took
        # the schedule away from the one the LoRA learned and left the LoRA solving a
        # trajectory it was never trained on. Reported as third legs and body parts
        # that render half -- on some LoRAs and not others, because how far a given
        # LoRA is from the schedule it was trained on differs.
        #
        # shift_video and shift_audio are what you typed. The LoRA knows what schedule
        # it wants better than an analysis of the sigma curve does.
        _lora_steps = lora_step_targets(graph)
        if _lora_steps:
            _targets = sorted({n for n, _ in _lora_steps})
            if len(_targets) > 1:
                notes.append(
                    "two or more LoRAs in this workflow state DIFFERENT step counts ("
                    + "; ".join(f"{n} from {nm}" for n, nm in sorted(_lora_steps))
                    + "). A distilled LoRA collapses the denoising trajectory onto the "
                      "step count it was trained for, so stacking two that disagree asks "
                      "the model for both at once and it renders neither. steps is "
                      f"currently {steps}")
            elif int(steps) != _targets[0]:
                notes.append(
                    f"{_lora_steps[0][1]} is built for {_targets[0]} steps and steps is "
                    f"{steps}. Read out of the FILE NAME, which is the only place a LoRA "
                    f"states it. Running a distilled LoRA off its own step count "
                    f"denoises past or short of where its trajectory lands")
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
        # hold: lines are read, never sent -- see register_holds. One under the scene
        # paragraph belongs to the first beat, and a paragraph holding nothing else to
        # the beat above it, so it never becomes an empty shot.
        _hold_all = [hold_lines(b) for b in beats]
        beats = [strip_hold_lines(b) if _h else b for b, _h in zip(beats, _hold_all)]
        if hold_lines(scene) and _hold_all:
            _hold_all[0] = hold_lines(scene) + _hold_all[0]
            scene = strip_hold_lines(scene)
        for _i in range(len(beats) - 1, -1, -1):
            if _hold_all[_i] and not beats[_i] and not _exact_all[_i] and len(beats) > 1:
                _to = _i - 1 if _i else 1
                _hold_all[_to] = (_hold_all[_to] + _hold_all[_i] if _to < _i
                                  else _hold_all[_i] + _hold_all[_to])
                del beats[_i], _hold_all[_i], _exact_all[_i]
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
        _guessed = [(n, sheet_pronoun(ln)) for n, ln in sheet_lines(sheet)
                    if n and pronoun_is_a_guess(ln)]
        if _guessed:
            notes.append(
                "pronoun read off a possessive -- "
                + "; ".join(f"{n} as '{p}'" for n, p in _guessed)
                + ". The entry declares none, and the only one in it is a 'his' or a "
                  "'her', which is as often somebody else's ('wearing his hoodie'). The "
                  "pronoun decides the body every bare shot describes, so write it into "
                  "the entry ('Kate: she, 25, ...') if that is wrong")
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
        # The opening paragraph's staging, withheld after shot 1 -- see scene_staging.
        _scene_staged = scene_staging(scene, sheet)
        # Who the opening paragraph lays down, until a beat gets them up -- see
        # withhold_staging.
        scene_lying = {n for _st, _who, _pos in _scene_staged.values()
                       if _pos == "lying down" for n in _who}
        # Only the people the paragraph PLACES -- a subject, or somebody beside one --
        # not anybody it mentions: "Mara waits for Dan to come home" has Dan away, and
        # reading him as there put him in a keyframe that never held him, with no
        # entrance. REPORTED.
        _opening_names = {n for n, _ in sheet_lines(sheet) if n and re.search(
            r"(?:^|[.!?;:]\s+|,\s*|\b(?:and|while|as|when|where|but|with|beside|"
            r"near|behind|opposite|next\s+to|across\s+from)\s+)"
            + re.escape(n) + r"(?![\w'’-])(?!['’]s)", scene or "", re.M)}
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
        removed_in = {}             # 0-based shot -> what came off in it -- see off_now_clause
        offnow_shots = []           # shots told a garment is off, over a picture or a mention
        _extras_seen = False        # the film has staged people the sheet does not name
        untracked_strip = []        # (shot, items) a group removal the sheet cannot hold
        inferred_sound = []         # shots given one derived from their action
        restrained = posed = rigid_latched = False
        beat_said_posture = False
        restrained_who = set()    # who is actually in the hardware
        anchored = ""             # where fastened limbs are held
        sealed = ""               # hardware closed over the groin, until it comes off
        worn_item = ""            # the hardware, in the author's words
        worn_items = []           # ...each piece of it, in order
        displaced = {}            # garment -> how it was moved
        displaced_dest = {}       # garment -> how far: " to the thighs" -- see displaced_to
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
        lying_on = {}             # name -> what they were laid on, where the beat said
        lying_shots = []          # shots told a body stays lying flat
        facing = ""               # which way up a lying body is, until it gets up
        on_call = False           # a phone call an earlier beat started -- see faces_each_other
        prev_acted = ""           # the last beat's acted text, for the same
        here = place_named(scene) or first_place(scene)
        _opening_room = here         # where the opening paragraph puts people
        _present_shots = {}          # 0-based shot -> newcomers the beat has already there
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
        gag_shots = []            # shots told the gag stays over the mouth
        device_shots = []         # shots whose line belongs to a machine
        applied_shots = []        # shots that put the hardware on
        applying_shots = set()    # 1-based shots sized for a piece going on -- see plan_lengths
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
        own_grade_shots = set()     # 0-based shots whose change of level over the take is theirs
        outdoor_shots = set()       # 0-based shots whose place is outside -- see outdoors()
        shot_rooms = {}             # 0-based shot -> (room it opens in, room it ends in)
        hardware_changed = set()    # 1-based shots that put hardware on or take it off
        _undescribed = []           # rooms the film enters that the prompt never describes
        open_moves = []             # (shot, where) moves to a place the list cannot name
        frame_shots = []            # shots told what the frame holds
        legs_held = ""              # where a beat or the sheet fastened the legs
        arms_of, legs_of = {}, {}   # ...per wearer, from the beat that fastened them
        _pose_limbs_had = {}        # name -> {"arms"/"legs": position held}, for pose control's latch
        _pose_doubted = set()       # wearers read off a beat whose pronoun cannot be them
        limbs_freed = False         # the last piece on a limb came off -- see below
        arms_freed = legs_freed = False   # ...counted per pair of limbs
        stayed_on = []              # (shot, who) kept described because still in frame
        addressed_on = []           # (shot, who) described because a line is spoken to them
        exact_shots = []            # shots carrying an exact: line of the author's
        camera_shots = []           # shots told the camera holds still
        named_often = []            # (shot, name, times named, times this node named them)
        contact_shots = []          # shots told which body is with which
        held_over = []              # (shot, people kept in frame by their hardware)
        partners_held = []          # (shot, people kept in frame as a sex-scene partner)
        _intimate = set()           # who a sex scene in this room involves, until a cut
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
        # Who is held as each shot opens, and from which shot -- see the end of the beat.
        held_shots = {}             # 1-based shot -> {name: shot their restraint or gag went on}
        held_items = {}             # 1-based shot -> {name: (items, where)} held through it
        _held_since = {}            # name -> shot a beat first put something on them
        _held_keys = None           # name -> what holds them after the last beat
        _hold_by_beat = False       # the hold was registered by a beat, not a prop
        _hold_named = []            # what those beats named it
        _hold_was_on = False        # the hold's flag after the last beat
        _first_is_plate = False
        guard_words = beat_words = total_words = sound_words = 0
        _state = engine.SceneState(place=engine.place_in(scene or ""))
        # Dated by what a beat ACTS, not what it says: "I'll cuff you" in beat 2 does
        # not put the sheet's cuffs on two beats before the cuffing.
        _staged_at = engine.staged_applications(
            [engine.acted_text(extract_directives(b)[0]) for b in beats])
        _sheet_hw = {c for c, _p, _w, _a in engine.hardware_spans(sheet or "")}
        _pron_of = {n: sheet_pronoun(ln) for n, ln in sheet_lines(sheet) if n}
        _scene_hw = scene_restraints(static, [n for n, _ in sheet_lines(sheet) if n], _pron_of)
        _hold_cast = [n for n, _ in sheet_lines(sheet) if n] or list(cast)
        held_rows = []              # (shot, what the state holds on whom once it is read)
        unheld_words = []           # (shot, restraint words its beat names that registered nothing)
        _static_wear = static_wardrobe(static, [n for n, _ in sheet_lines(sheet) if n])
        for b in beats:
            body, toks, adds = extract_directives(b)
            toks = [sheet_form(t, scene) for t in toks]
            _said = _exact_all[len(plan)] if len(plan) < len(_exact_all) else []
            _exact = (" " + " ".join(terminate_lines(x) for x in _said)) if _said else ""
            if _said:
                exact_shots.append(len(plan) + 1)
            _marked = mark_dialogue(body)
            if _marked != body:
                dialogue_marked.append(len(plan) + 1)
                body = _marked
            # WHAT THIS SHOT ACTS OUT -- the beat without its speech, its orders and its
            # intentions. Every reading of what HAPPENS below takes this, never `body`:
            # see engine.acted_text. `body` still goes to the model word for word, and
            # still drives everything about the voice.
            _acted = engine.acted_text(body)
            _later_for_state = {c for c, at in _staged_at.items()
                                if at > len(plan) + 1}
            for _n, _line in sheet_lines(sheet):
                if _n:
                    # What the anchor dresses them in is worn too. See static_wardrobe.
                    _wear = [g for g in _static_wear.get(_n, [])
                             if not names_any(g, [x for x in gone if x not in restored])]
                    _state.declare(_n, _line + "".join(f", {g}" for g in _wear),
                                   staged_later=_later_for_state)
            # ...and a scene sentence that has somebody already in hardware, on them.
            for _n, _s in _scene_hw:
                if _n or not sheet_lines(sheet):
                    _state.declare(_n, _s, staged_later=_later_for_state, hardware_only=True)
            # What was on BEFORE this beat, so a piece that goes on in it can be told
            # from one that was already there. See _new_on below.
            _hw_before = {(_n, _k) for _n, _q in _state.people.items()
                          for _k in _q.hardware}
            _ch = _state.read(_acted, cast=[n for n, _ in sheet_lines(sheet) if n],
                              shot=len(plan) + 1, pronouns=_pron_of)
            _held_line, _hold_bad = register_holds(
                _state, _ch, _hold_all[len(plan)] if len(plan) < len(_hold_all) else [],
                _hold_cast, len(plan) + 1)
            for _bad in _hold_bad:
                notes.append(f"shot {len(plan) + 1}: a hold: line registered nothing -- "
                             f"{_bad}. Write it as 'hold: Name, item and where it is; "
                             f"next item'")
            if _ch.get("applied") or _ch.get("released"):
                hardware_changed.add(len(plan) + 1)
            # Where the shot is comes before who is in it. The rules below keep a person
            # described only within one room, and read the cut before it was worked out
            # -- the previous beat's, so a cut let them through and the shot after it
            # did not.
            _frm, _via, _to = travel_legs(_acted)
            _is_travel = bool(travel_anchor(_frm, _via, _to, here, body))
            _room_before = here
            _named_place = place_named(_acted)
            if _named_place in _THRESHOLDS:
                _named_place = ""        # standing in a doorway is not a room -- see travel_legs
            _place_now = _to or _frm or _named_place or here
            _opens_in = _frm or (_room_before if _is_travel else _place_now)
            _is_cut = bool(len(plan) and _opens_in and _room_before
                           and _opens_in != _room_before)
            # ...and a walk into another room leaves the room as surely as a cut does.
            _leaves_room = _is_cut or bool(_to and _to != _room_before)
            _was = list(active)
            _still_there = shot_frames.get(len(plan) - 1, ([], []))[1] if plan else []
            if _leaves_room:
                _intimate = set()
            _intimate -= set(leaves_in(_acted, sheet, _still_there))
            _back_cands = []
            _carried_on, _carried = [], []
            if character_guard:
                shot_sheet, active = sheet_for_beat(sheet, body, active)
                _staged_now = engine.staged_text(body)
                if _SEXUAL_STAGING.search(_staged_now) or _INTIMATE.search(_staged_now):
                    _intimate |= set(active) | set(_still_there)
                # THE PERSON THE HARDWARE GOES ON IS IN THE SHOT. "Dan handcuffs her
                # wrists behind her back" names Dan alone, and with no pronoun on the
                # sheet "her" named nobody here -- so the shot described one person, the
                # cuffs had no body to be on, and the next beat's tape went on a woman
                # the text had never placed in cuffs. REPORTED as the handcuffs breaking
                # when the duct tape goes on. The state knows whose they are.
                # ...but only somebody it is genuinely put ON or taken OFF: a restraint
                # the beat describes as already worn ("Jade sits tied to a chair") is
                # nobody's work in this shot, and reading it as one pulled whoever the
                # reader guessed into the frame. REPORTED with the rope on the wrong
                # person. See staged_on_now.
                _worked_on = [n for n, _l in sheet_lines(sheet)
                              if n and n not in (active or [])
                              and any(_w == n for _w, _r in
                                      [(_w, _r) for _w, _r in (_ch.get("applied") or [])
                                       if staged_on_now(_acted, getattr(_r, "item", ""))]
                                      + list(_ch.get("released") or []))]
                _worked_on += [n for n in dict.fromkeys(_n for _n, _r, _l, _nw in _held_line
                                                        if _nw)
                               if n not in (active or []) and n not in _worked_on]
                if _worked_on and not _leaves_room:
                    _keep = set(list(active or []) + _worked_on)
                    active = [n for n, _l in sheet_lines(sheet) if n in _keep]
                    shot_sheet = "\n".join(ln for n, ln in sheet_lines(sheet)
                                           if n in _keep)
                    held_over.append((len(plan) + 1, list(_worked_on)))
                # A PERSON IN HARDWARE STAYS IN THE SHOT.
                #
                # sheet_for_beat keeps whoever the beat names and drops the rest, and
                # for ordinary beats that is right: describing everybody in every shot
                # puts everybody in every shot. It is wrong about somebody FASTENED.
                # "Dan stops and sits back" names one of the two, so the woman cuffed
                # to the bed beside him left the shot -- and her entry went with her,
                # so the cuffs, the collar and everything else she wore stopped being
                # described. Reported as items going missing and the restraint
                # changing, on the beats that name one person and lean on continuity
                # for the other, which is most of a sex scene.
                #
                # Three conditions, all required. She was HERE last shot; she is in
                # hardware NOW; and nobody LEAVES in this beat -- "Jon walks out and
                # shuts the door" takes the camera with him, and being fastened is the
                # reason she did not follow, not a reason the shot stayed with her.
                # Not across a cut to another room either, for the same reason.
                # ...and she has to still BE there: "Dan leads Ana out", then "Dan comes
                # back alone" kept her -- cuffs, collar and all -- in a room she had been
                # taken out of, because only her being in the last beat was checked, not
                # her being in its last frame. The partner rule below always checked both.
                _held_on = [n for n, _l in sheet_lines(sheet)
                            if n and n in (restrained_who or set())
                            and n in (_was or []) and n in _still_there
                            and n not in (active or [])]
                # "Walks away" is not "walks out": with the camera held he goes across the
                # frame and she is still in it, chained where she was. Dropped, her entry
                # and her hardware left the shot, and the next shot opened on a frame
                # without them. REPORTED as equipment dropped when the camera follows the
                # action. Only an exit -- out, leaves, through the door -- lets her go.
                _walks_off = bool(
                    re.search(r"\b(?:walks?|walked|steps?|stepped|moves?|moved|wanders?|"
                              r"strolls?|turns?|turned|backs?|drifts?|paces?)\s+"
                              r"(?:\w+\s+)?(?:away|off)\b", body or "", re.I)
                    and not re.search(r"\b(?:out|outside|leaves?|left|exits?|door|doorway)\b",
                                      body or "", re.I))
                if (_held_on and not _leaves_room
                        and (_walks_off or not leaves_in(_acted, sheet, _was))):
                    _keep = set(list(active or []) + _held_on)
                    active = [n for n, _l in sheet_lines(sheet) if n in _keep]
                    shot_sheet = "\n".join(ln for n, ln in sheet_lines(sheet)
                                           if n in _keep)
                    held_over.append((len(plan) + 1, list(_held_on)))
                # A PARTNER STAYS IN THE SHOT, on the same three conditions, while the
                # scene is a sex scene. "Kate arches her back" names one of two bodies
                # in contact, and dropping the other told the shot "There is one person
                # in the shot" over a first frame holding both -- so the model moved or
                # removed him to agree, and he came back from somewhere else on the
                # next beat that named him. Reported as positions resetting mid-scene.
                _partners = [n for n, _l in sheet_lines(sheet)
                             if n and n in _intimate and n in (_was or [])
                             and n in _still_there and n not in (active or [])]
                if (_partners and any(n in _intimate for n in (active or []))
                        and not _leaves_room and not leaves_in(_acted, sheet, _was)):
                    _keep = set(list(active or []) + _partners)
                    active = [n for n, _l in sheet_lines(sheet) if n in _keep]
                    shot_sheet = "\n".join(ln for n, ln in sheet_lines(sheet)
                                           if n in _keep)
                    partners_held.append((len(plan) + 1, list(_partners)))
                # EVERYBODY STILL IN THE FRAME STAYS IN THE SHOT -- the two rules above,
                # for everyone. The shot opens on the last frame, and whoever is in it is
                # in the picture this shot starts from; leaving them out of the text told
                # it "There is one person in the shot" over a frame holding two, and the
                # model removed one to agree -- who came back from nowhere on the next
                # beat that named them, while the other vanished in turn. REPORTED as
                # beats losing what was in the beat before, and asked for in so many
                # words: both characters entirely in the shot. Nobody is kept past an
                # exit, a cut to another room, a beat that leaves somebody alone, or a
                # framing the author set themselves.
                _stays = [n for n, _l in sheet_lines(sheet)
                          if n and n in (_was or []) and n in _still_there
                          and n not in (active or [])]
                if _stays and not _leaves_room and not (
                        _FRAME_SIZE.search(body or "") or tight_framing(body or "")
                        or _ALONE.search(engine.staged_text(body) or "")):
                    _out_now = set(leaves_in(_acted, sheet, _was))
                    _stays = [n for n in _stays if n not in _out_now]
                    if _stays:
                        _keep = set(list(active or []) + _stays)
                        active = [n for n, _l in sheet_lines(sheet) if n in _keep]
                        # Without their <Picture N>: the frame the shot opens on already
                        # pictures them, and a second picture of one person is how a
                        # second one is drawn. Their own tag rides the shots that NAME
                        # them, as it always did.
                        shot_sheet = "\n".join(
                            (untagged(ln) if n in _stays else ln)
                            for n, ln in sheet_lines(sheet) if n in _keep)
                        stayed_on.append((len(plan) + 1, list(_stays)))
                # THE PERSON SPOKEN TO IS THERE. "Dan says, 'Take off your shirt.'" names
                # only Dan, so only Dan was described -- and on the next beat Ana, who
                # was being spoken to all along, walked into the frame from its edge: an
                # action nobody wrote. Only on her first appearance or while she is
                # still in the frame, never a question ("where are you?" is absence),
                # and never down a phone or through a door.
                _spoken_to = addressed_in(body, sheet)
                if (_spoken_to and _spoken_to not in (active or [])
                        and (_spoken_to not in _seen_before or _spoken_to in _still_there)
                        and not _leaves_room):
                    _keep = set(list(active or []) + [_spoken_to])
                    active = [n for n, _l in sheet_lines(sheet) if n in _keep]
                    shot_sheet = "\n".join(ln for n, ln in sheet_lines(sheet) if n in _keep)
                    addressed_on.append((len(plan) + 1, _spoken_to))
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
                # A plate rides as a reference, which FastH3 cannot read; there the
                # first_frame is what its model supports -- frame one.
                if (_new and not arrives_in(_acted) and not plan and not _fast
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
                if _new and not arrives_in(_acted) and plan:
                    _placed_shots[len(plan)] = list(_new)
                    # ALREADY THERE IS NOT AN ENTRANCE. "Dan looks up from his phone",
                    # "Dan is already sitting on the crate", or a Dan the opening
                    # paragraph stood by the window of this very room, was walked in
                    # from the edge of the frame -- an entrance nobody wrote. REPORTED
                    # as characters doing things the beat never wrote. They get no
                    # entrance, and the opening paragraph's sentence about them rides
                    # this shot. Freeing the camera to find them was tried and kept
                    # out: the camera hold is a reported fix of its own. Routing
                    # the shot through the soft cut instead was weighed and refused:
                    # that is the restart this used to be -- reported as angles
                    # changing between beats -- and on FastH3 a start from nothing.
                    _present_shots[len(plan)] = [
                        n for n in _new
                        if already_in_position(_acted, n, sheet)
                        or (n in _opening_names
                            and (_opens_in or "") == (_opening_room or ""))]
                    _walk_in = [n for n in _new if n not in _present_shots[len(plan)]]
                    _there = _present_shots[len(plan)]
                    notes.append(
                        f"shot {len(plan) + 1} introduces {', '.join(_new)} in "
                        f"position rather than arriving. The shot still opens on the "
                        f"previous shot's last frame -- same camera, same room, everyone "
                        f"and everything in it where they were"
                        + (f" -- and {'they come' if len(_walk_in) > 1 else _walk_in[0] + ' comes'}"
                           f" into it from the edge of the frame as it begins"
                           if _walk_in else "")
                        + (f"; {_join_names(_there)} "
                           f"{'are' if len(_there) > 1 else 'is'} written as already "
                           f"there, so no entrance is added and the beat's own words "
                           f"place {'them' if len(_there) > 1 else _there[0]}"
                           if _there else "")
                        + ". It used to restart "
                        "there instead, which cost the camera angle and everything the "
                        "last frame remembered (on FastH3, all of it). Write the entrance "
                        "yourself -- 'walks in', 'steps through' -- to say how")
                _back_cands = [n for n in active if n not in _was and n in _seen_before]
                _seen_before.update(active)
            else:
                shot_sheet = sheet
            _prev_stays = shot_frames.get(len(plan) - 1, ([], []))[1]
            # Somebody held since an earlier shot is pictured by that frame as they are
            # now, so their portrait is what gives way -- as at render.
            _no_carry = not _cond_module.may_carry_frame(
                _prev_stays, active,
                {n for n, ln in sheet_lines(sheet) if n and picture_tags(ln)}
                - {n for n in _held_since if n in _prev_stays})
            _fresh = (_is_cut
                      or (restart_after_removal and (len(plan) - 1) in stripped_shots
                          and _no_carry)
                      or bool(_ALONE.search(engine.staged_text(body))))
            _kept = [] if _fresh else list(_in_frame)
            # Kept in the last shot only because they were still in its frame is not
            # the beat putting them there: "Dan walks in" while he sits in the frame is
            # still a second Dan walking in, and still starts fresh.
            _only_kept = {n for k, ws in stayed_on if k == len(plan) for n in ws}
            _again = [n for n in comes_in(_acted, sheet)
                      if n in _kept and (n not in _was or n in _only_kept)
                      ] if (plan and not _is_travel) else []
            if _again:
                reentry_shots[len(plan)] = _again
                _kept = []
            _carry = [n for n in _kept if n not in active]
            # Removals are per PERSON. "shirt" off Kate is not "shirt" off Dan, and a
            # token already gone used to swallow every later removal of the same
            # word: his shirt stayed on his sheet line while the shot called his
            # chest bare. See removal_owner.
            _taken_from = {}            # token -> who it comes off in THIS beat
            _gone_before = list(gone)
            _gone_by_before = {t: set(w) for t, w in gone_by.items()}
            if auto_remove:
                inferred = []
                for t in infer_removals(_acted, scene):
                    if t in toks:
                        continue
                    _o = removal_owner(_acted, t, sheet)
                    if _o:
                        _taken_from[t] = [_o]
                    if t in gone and not (_o and gone_by.get(t) and _o not in gone_by[t]):
                        continue
                    inferred.append(t)
                if hold_restraints and restraint_coming_off(_acted):
                    for _n, _ln in sheet_lines(sheet if sheet_lines(sheet) else scene):
                        for _hw in restraint_words(_ln):
                            _named = re.search(r"\b" + re.escape(_hw) + r"\b",
                                               body or "", re.I)
                            # "...and slaps IT over her mouth" is the tape going on, not
                            # the cuffs coming off: not where this beat puts anything on.
                            _pron = (len(restraint_words(_ln)) == 1
                                     and not _ch.get("applied")
                                     and re.search(r"\b(?:them|it)\b", body or "", re.I))
                            if (_named or _pron) and _hw not in toks \
                                    and _hw not in gone and _hw not in inferred:
                                inferred.append(_hw)
                if inferred:
                    toks = list(toks) + inferred
                    notes.append(f"shot {len(plan) + 1}: read '{', '.join(inferred)}' as "
                                 f"coming off, from the beat's own wording")
            bare = auto_remove and strips_bare(_acted)
            _down_to = engine.strips_to(_acted) if auto_remove and not bare else None
            if bare or _down_to is not None:
                _strippers = strips_who(_acted, active if character_guard and active
                                        else [n for n, _ in sheet_lines(shot_sheet) if n],
                                        shot_sheet)
                # ...and what the anchor dresses them in, which is worn as much as the
                # sheet's: "Kate and Dan undress" left her bra and his boxers in the
                # anchor text of every later shot that called them bare.
                _their_wear = ", ".join(
                    g for n in (_strippers or [n for n, _ in sheet_lines(shot_sheet) if n])
                    for g in _static_wear.get(n, []))
            if _down_to is not None and _strippers:
                # A PARTIAL strip: everything on the sheet comes off except what the
                # beat keeps. "underwear" keeps whatever the sheet has underneath.
                _their_sheet = "\n".join(
                    ln for n, ln in sheet_lines(shot_sheet) if n in set(_strippers))
                _kept_on = [g for g in garments_in(_their_sheet + "\n" + _their_wear)
                            if g in _down_to
                            or ("underwear" in _down_to and is_undergarment(g))]
                _taken = [g for g in garments_in(_their_sheet + "\n" + _their_wear)
                          if g not in _kept_on and g not in toks
                          and (g not in gone or (gone_by.get(g)
                                                 and not set(_strippers) <= gone_by[g]))]
                for g in _taken:
                    _taken_from[g] = list(_strippers)
                if _taken:
                    toks = list(toks) + _taken
                    notes.append(
                        f"shot {len(plan) + 1} reads as {', '.join(_strippers)} stripping "
                        f"down to {', '.join(_kept_on) or ', '.join(_down_to)} -- so the rest "
                        f"of the wardrobe on the character sheet was taken off: "
                        f"{', '.join(_taken)}")
            if bare:
                _their_sheet = "\n".join(
                    ln for n, ln in sheet_lines(shot_sheet) if n in set(_strippers)
                ) or shot_sheet
                stripped = [g for g in garments_in(_their_sheet + "\n" + _their_wear)
                            if g not in toks
                            and (g not in gone or (_strippers and gone_by.get(g)
                                                   and not set(_strippers) <= gone_by[g]))]
                for g in stripped:
                    if _strippers:
                        _taken_from[g] = list(_strippers)
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
            # A BELT GOES WITH WHAT IT HOLDS UP. It is not on the garment list -- a steel
            # or chastity belt is hardware -- so a full strip left "a leather belt" on the
            # sheet of a naked man, every shot after, and a belt round the hips is a
            # waistband the model has to hang something from. REPORTED as clothing
            # restored over the genitals in sex scenes.
            if toks or bare:
                for _bn, _bl in sheet_lines(sheet):
                    if not _bn or "belt" in toks or not worn_belt(_bl):
                        continue
                    _lower = [t for t in toks if _LOWER_OUTER.search(str(t))
                              and re.search(r"\b" + re.escape(str(t)) + r"\b", _bl, re.I)
                              and _bn in (_taken_from.get(t)
                                          or [removal_owner(body, t, sheet)] or [_bn])]
                    if (bare and (not _strippers or _bn in _strippers)) or _lower:
                        toks = list(toks) + ["belt"]
                        _taken_from["belt"] = [_bn]
            revived = [t for t in gone if names_any(body, [t]) and t not in toks]
            if revived:
                notes.append(
                    f"shot {len(plan) + 1} names {', '.join(revived)} in its own text, and "
                    f"that came off earlier. Beats are sent to the model word for word, so "
                    f"naming it puts it back on -- the scene no longer mentions it, but this "
                    f"beat does. Reword the beat if it should stay off")
            if toks:
                stripped_shots.add(len(plan))
                removed_in[len(plan)] = list(toks)
                gone.extend(t for t in toks if t not in gone)
                if extras_in(body):
                    untracked_strip.append((len(plan) + 1, list(toks)))
                _took = strippers_in(_acted, shot_sheet if shot_sheet else sheet)
                for _t in toks:
                    _wears = [n for n, _wl in sheet_lines(sheet)
                              if n and (re.search(r"\b" + re.escape(_t) + r"\b",
                                                  _wl or "", re.I)
                                        or names_any(", ".join(_static_wear.get(n, [])),
                                                     [_t]))]
                    # Whose it is, when the beat says -- "Dan takes off HER shirt" is
                    # hers, whoever else wears one -- before who is doing it.
                    _whose = [n for n in (_taken_from.get(_t)
                                          or [removal_owner(_acted, _t, sheet)])
                              if n in _wears]
                    _whose = _whose or [n for n in _took if n in _wears] or _wears
                    _taken_from[_t] = _whose
                    gone_by.setdefault(_t, set()).update(_whose)
                _retired = [a for a in shown if names_any(a, toks)]
                if _retired:
                    shown = [a for a in shown if a not in _retired]
                    notes.append(f"shot {len(plan) + 1} takes off something an earlier "
                                 f"'add:' had put on, so that line retires with it: "
                                 + "; ".join(_retired))
                notes.append(f"removed from the scene from shot {len(plan) + 1} on: "
                             + ", ".join(scene_name_for(t, scene) or t for t in toks))
            maybe = missing_removals(_acted, scene, gone) if not auto_remove else []
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
                    if not (names_any(engine.garment_masked(body), [_head])
                            and beat_stages_wearing(_acted, _head)):
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
                                 if any(beat_stages_wearing(_acted, g) for g in _back)
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
            # Without a keyframe the removal shot keeps describing what comes off in
            # it -- but only on the person it comes off. Somebody who lost the same
            # garment earlier stays without it.
            visible = gone if has_keyframe else [g for g in gone
                                                 if g not in toks or g in _gone_before]
            _gone_by_here = {t: (_gone_by_before.get(t) if not has_keyframe and t in toks
                                 else gone_by.get(t)) for t in gone}
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
            _moved_now = {g for g, _h in displaced_garments(_acted, shot_sheet or sheet)}
            _back_now = set(restored_garments(_acted, shot_sheet or sheet))
            if puts_it_back(_acted) and len(displaced) == 1:
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
            # Cleared here too: the latch below never runs on a beat that takes it
            # off, so without this the seal outlives the removal that asked for it.
            # A REMOVAL BEAT NAMES THE THING IT REMOVES, so crotch_seal() fires on
            # "Dan unlocks the chastity belt" exactly as it does on the beat that put
            # it on. Read the removal FIRST and off the beat's own naming, not off
            # whatever happens to be latched -- checking `sealed` first meant a belt
            # the SHEET declared (never latched, because no beat applied it) got
            # latched by the beat taking it off, and then held for the rest of the run.
            _sealed_before = sealed
            _seal_named = crotch_seal(_acted)
            _seal_off = (seal_comes_off(_acted, _seal_named or sealed)
                         or (bool(sealed) and names_any(sealed, toks)
                             and not _OTHER_PART_OFF.search(_acted or "")))
            if _seal_off:
                sealed = ""
            elif _seal_named:
                sealed = _seal_named
            # DELIBERATELY NOT READ FROM THE SHEET. A sheet-declared belt can be
            # UNDER something -- defer_tag_for and hidden_layers exist for exactly
            # that, and hold the belt's words back on every shot where a garment
            # covers it, because a reference is an instruction to draw the thing.
            # Seeding the seal from the sheet named the belt in those shots and
            # undid it. What this clause is for is hardware a BEAT puts on, which is
            # the case that was losing it between shots; a covered belt already has
            # an owner and it is not this.
            _one_groin = ("" if sealed else
                          (groin_of(_one_pron, _one_age) if len(active or []) == 1 else ""))
            # Each line with what the anchor still has that person wearing, so a thong
            # written there keeps the hips covered when the jeans over it come off.
            _off_now = [g for g in gone if g not in restored]
            _worn_lines = [(n, ln + "".join(f", {g}" for g in _static_wear.get(n, [])
                                             if not names_any(g, _off_now)))
                           for n, ln in sheet_lines(shot_sheet)]
            _bare_sheet = ("\n".join(ln for n, ln in _worn_lines if n and names_any(ln, toks))
                           or "\n".join(ln for _n, ln in _worn_lines) or shot_sheet)
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
                              groin=("" if sealed else groin_of(*_pron_age(shot_sheet, _n))))
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
                          if not (_who and _gone_by_here.get(t) and _who not in
                                  {str(x).lower() for x in _gone_by_here[t]})]
                # Nothing in the hands while the wrists are held -- see hands_free_line.
                if any(_n.lower() == _who and any(r.part in ("wrists", "hands", "arms")
                                                  for r in _q.hardware.values())
                       for _n, _q in _state.people.items() if _n):
                    _ln = hands_free_line(_ln)
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
                _w = next((n for n in (_taken_from.get(_t) or [])), None) or next(
                    (n for n, ln in sheet_lines(_who_sheet) if n and names_any(ln, [_t])),
                    _wearer)
                _a = removal_agent(body, _cast_here, _w, _t)
                _by_agent.setdefault((_a, _w), []).append(_t)
            # Everyone a full strip undresses, not the first of them: "Kate and Dan
            # undress" told Dan to keep on what his entry lists, in the shot where
            # he takes it off.
            tail = (own_body(BARE_HOLD, (_strippers if (bare and _strippers) else None)
                             or _wearer or (active[:1] if active else []),
                             active if character_guard else
                             [n for n, _ in sheet_lines(_who_sheet) if n])
                    if (bare and toks)
                    else "".join(off_by_last_frame(
                        _items, _a, scene, body,
                        wearer_sheet="\n".join(ln for n, ln in sheet_lines(sheet)
                                               if n and n == _w))
                        for (_a, _w), _items in _by_agent.items()))
            _was_restrained = restrained
            _shown_hw = {x for _c, _p, _w, _a in engine.hardware_spans(_scene_for_state)
                         for x in (_c, _w)}
            if hold_restraints:
                _clears = bool(names_any(RESTRAINT_HOLD_KEY, toks)
                               or held_piece_named(_state, toks)
                               or any(restraint_present(t) for t in toks)
                               or (restraint_coming_off(_acted)
                                   and any(_RESTRAINT_WORD.match(str(t)) for t in toks)))
                # ONLY WHAT CAME OFF, OFF WHOEVER IT CAME OFF. A removal that looked like
                # hardware cleared every restraint on everybody: "Dan removes his belt"
                # uncuffed, uncollared and ungagged the woman beside him. REPORTED as
                # restraints breaking and coming undone. The state knows who wears what;
                # the named piece leaves its wearer, and the rest stays fastened.
                if _clears and any(_q.hardware for _q in _state.people.values()):
                    for _t in toks:
                        _all = bool(re.fullmatch(r"(?:all\s+)?(?:the\s+)?(?:restraints?|"
                                                 r"bindings?|bonds|hardware|everything)",
                                                 str(_t).strip(), re.I))
                        for _n in (_taken_from.get(_t) or list(_state.people)):
                            _q = _state.people.get(_n)
                            if _q is None:
                                continue
                            for _k in [k for k, r in _q.hardware.items()
                                       if (_all or names_any(r.item, [_t])
                                           or names_any(_t, [k[0]]))
                                       and (not removal_parts(_t)
                                            or k[1] in removal_parts(_t))]:
                                _ch.setdefault("released", []).append(
                                    (_n, _q.hardware.pop(_k)))
                                hardware_changed.add(len(plan) + 1)
                    if any(_q.hardware for _q in _state.people.values()):
                        _clears = False
                        restrained_who = {n for n, _q in _state.people.items()
                                          if _q.hardware}
                        if (sealed and names_any(sealed, toks)
                                and not _OTHER_PART_OFF.search(_acted or "")):
                            sealed = ""
                if _clears:
                    restrained = posed = rigid_latched = False
                    anchored = ""
                    sealed = ""
                    worn_item = ""
                    worn_items = []
                    restrained_who = set()
                elif (_ch.get("released") and restrained
                      and not any(_q.hardware for _q in _state.people.values())):
                    # A real release took the last piece off: the flag follows the state,
                    # or the hold goes on saying "every restraint stays closed" over
                    # hardware that is gone. NOT the seal: it comes off by its own
                    # removal, and unlocking her cuffs used to take it with them.
                    restrained = posed = rigid_latched = False
                    anchored = ""
                    worn_item = ""
                    worn_items = []
                    restrained_who = set()
                # ARMED BY A PIECE WITH A WEARER, which is what the state holds -- not by
                # hardware words anywhere in the scene text. "A pair of handcuffs lies on
                # the nightstand" and "keys clipped to his belt" held restraints on
                # nobody, and "Dan cuffs Mara" registered on her and armed nothing.
                # REPORTED. The beat's own words still arm it where the state has no
                # piece to name, and so do the scene's ("hogtied", "restrained") where
                # they name no piece at all. A piece only the sheet gives counts while
                # the shot's text still shows it: one covered by a garment is held back.
                elif (any(_r.applied_in or _r.item in _shown_hw or _k[0] in _shown_hw
                          for _q in _state.people.values()
                          for _k, _r in _q.hardware.items())
                      or restraint_present(_acted)
                      or ((not _shown_hw or any(not _n for _n, _s in _scene_hw))
                          and restraint_present(_scene_for_state))):
                    restrained = True
                    if not _was_restrained or restraint_going_on(_acted):
                        _staged_here = (engine.applies_hardware(_acted)
                                        or restraint_going_on(_acted)
                                        or (restraint_present(_acted)
                                            and not restraint_present(_scene_for_state)))
                        # WHO IT WENT ON is what the state recorded, read clause by clause.
                        # wearer_of reads the whole beat at once and gives up past four
                        # words, so "Dan pulls off a strip of duct tape and slaps it over
                        # her mouth" made DAN the restrained one -- and every shot after
                        # held his arms behind his back and left hers free. REPORTED as
                        # the cuffs breaking the beat the tape went on.
                        _put_on = {_n for _n, _r in (_ch.get("applied") or []) if _n}
                        _w = (engine.wearer_of(_acted, [n for n, _ in sheet_lines(sheet) if n])
                              if (_staged_here and not _put_on) else "")
                        _new = (_put_on or ({_w} if _w else set())
                                or set(restraint_wearers(sheet))
                                or restrained_by_beat(_acted, active))
                        restrained_who |= (_new if _new else set(active))
            # ...and from then on, whoever the state says is in hardware IS the set. It
            # only ever grew before, so one wrong guess -- the man doing the taping --
            # stayed "restrained" for the rest of the run, and the arms the pose placed
            # were his.
            if hold_restraints and _held_line:
                restrained = True
                if any(_r.rigid for _n, _r, _l, _nw in _held_line):
                    rigid_latched = True
            _state_held = {n for n, _q in _state.people.items() if _q.hardware}
            if restrained and _state_held:
                restrained_who = set(_state_held)
            _named_item = hardware_named(_acted) if restrained else ""
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
                             and restraint_going_on(_acted))
            # A SECOND PIECE GOING ON. _applying is the FIRST restraint of a run, so
            # once the cuffs were on, the tape pressed over her mouth two beats later
            # was told it "stays closed and fastened as it was put on" in the shot that
            # puts it on: on at the first frame, before anybody reached for it. The
            # model could not have it both ways, and the tape came and went. The state
            # knows which pieces are new; those get both ends of the change.
            # Not a piece the beat calls HERS: "chains her ankles to her collar" is a
            # collar she already has, and telling the shot it goes on now would stage
            # a collaring nobody wrote.
            def _possessed(_item):
                _head = re.escape(str(_item).split()[-1].rstrip("s"))
                # One describing word at most -- "her steel collar" -- and never a body
                # part or a preposition: "ties HER ankles WITH rope" is new rope.
                return bool(re.search(
                    r"\b(?:her|his|their|[A-Z][\w-]*['’]s)\s+"
                    r"(?:(?!(?:with|using|by|and|to|from|in|on|over|across|round|around|"
                    r"wrists?|ankles?|hands?|arms?|legs?|feet|neck|throat|mouth|lips|eyes|"
                    r"face|head|waist|knees?)\b)\w+\s+)?" + _head + r"s?\b", _acted or ""))
            # ...and only what the beat PUTS on. New to the state is not new to the
            # body: "her wrists cuffed to the headboard" is the first anybody hears of
            # the cuffs, and they have been on since before the shot. See staged_on_now.
            _new_on = {(_n, _r.item) for _n, _q in _state.people.items()
                       for _k, _r in _q.hardware.items()
                       if (_n, _k) not in _hw_before and _r.item
                       and not _possessed(_r.item)
                       and staged_on_now(_acted, _r.item)}
            _new_on |= {(_n, _r.item) for _n, _r, _l, _nw in _held_line if _nw}
            if (not early_hardware and restraint_going_on(_acted)
                    and restraint_present(_scene_for_state)):
                early_hardware.append(len(plan) + 1)
            if restrained and rigid_hardware(f"{_acted} {shot_scene}"):
                rigid_latched = True
            # A POSE ENDS WHEN THE BODY LEAVES IT. "Posed" latched on the kneel and held
            # for the rest of the film, so after "Mara stands up" the cuffs still fixed
            # "the position that keeps" -- a kneel she had got up out of. REPORTED as
            # characters held in poses the beat did not ask for. A posture of her own
            # that no hardware forces clears it; a forced one in the same beat wins.
            _unposed = (posed and not forced_pose(_acted)
                        and _leaves_pose(_acted, shot_sheet, restrained_who))
            if _unposed:
                posed = False
            elif rigid_latched and forced_pose(f"{_acted} {shot_scene}"):
                posed = True
            # A SHOT THAT PUTS A PIECE ON IS SIZED FOR THE HOLD AFTER IT -- see
            # plan_lengths. The same reading the applying wording comes from: the first
            # restraint, a piece new to the state that this beat puts on, or a seal that
            # was not there before. It is told when the piece goes on, so it is not also
            # told to finish its action on the last frame.
            _seal_going_on = bool(not _seal_off and not _sealed_before
                                  and crotch_seal(_acted, worn_items))
            if _applying or (restrained and _new_on) or _seal_going_on:
                applying_shots.add(len(plan) + 1)
            _applies_here = (len(plan) + 1) in applying_shots
            _have = plan_lengths([body], ceiling,
                                 shot_length == "from the beat", pace,
                                 applying=(1,) if _applies_here else ())[0][0] / H3_FPS
            _pace = ("" if _applies_here
                     else pace_clause(beat_seconds(body), _have, beat=body))
            if _pace:
                paced_shots.append(len(plan) + 1)
            _movers = movers_in(_acted, shot_sheet, active or [])
            _travel = travel_anchor(_frm, _via, _to, here, body, movers=_movers)
            _journey = bool(_travel)     # between places: the camera has to go too
            if _travel:
                travel_shots.append(len(plan) + 1)
            else:
                _open_to = moved_to(_acted, active)
                _travel = move_clause(_open_to, body, movers=_movers)
                if _travel:
                    open_moves.append((len(plan) + 1, _open_to))
            here = _place_now
            if _is_cut:
                cut_shots.add(len(plan))
            shot_rooms[len(plan)] = (_opens_in or "", here or "")
            _outside = outdoors(here, scene, beats[0] if beats else "")
            if _outside:
                outdoor_shots.add(len(plan))
            _carry = [n for n in _kept if n not in active]
            _shows = list(active) + _carry
            # A walk to another room leaves behind whoever it does not describe.
            _ends_with = list(active) + ([] if (_to and _to != _room_before) else _carry)

            _back = [n for n in _back_cands if n not in _kept]
            if _back:
                _returns.append((len(plan) + 1, list(_back)))
            _gone = leaves_in(_acted, sheet, _shows)
            _in_frame = [n for n in _ends_with if n not in _gone]
            _lone = left_alone(_acted, sheet, _in_frame)
            if _lone:
                _in_frame = [_lone]
            shot_frames[len(plan)] = (_shows, list(_in_frame))
            if here and here not in _described_rooms and here not in _undescribed:
                _undescribed.append(here)
            _where = where_hold(here, scene, outdoor=_outside) if not _travel else ""
            if _where:
                where_shots.append(len(plan) + 1)
            _room_now = (room_tone(here) or _room) if (auto_sound and _where) else _room
            _bed_now = ((scene_ambient(here) or ambient_bed)
                        if (auto_sound and _where) else ambient_bed)
            if _where and auto_sound and (_room_now != _room or _bed_now != ambient_bed):
                acoustic_shots.append((len(plan) + 1, here))
            # NOBODY ON SCREEN, NOBODY DESCRIBED. A beat of the clock, the rain, the
            # empty yard, still carried the last shot's people in full -- a sheet line
            # each and a body count -- which is how a face turns up in a shot of a
            # clock. Withheld for people who wear nothing to keep and have had nothing
            # taken off: a restraint, a seal or a removal is still described, so it
            # holds in the frame it opens on.
            _bare_frame = bool(character_guard and not beat_puts_somebody_on_screen(body, sheet))
            # ...but only where the frame is composed afresh, or the last one held
            # nobody. A take that opens on the previous last frame still shows its
            # people, and stripping their lines and count left the held keyframe's
            # bodies undescribed. REPORTED.
            _k_fresh = len(plan)
            _composed_fresh = (not plan or _k_fresh in cut_shots
                               or _k_fresh in reentry_shots
                               or (restart_after_removal
                                   and (_k_fresh - 1) in stripped_shots))
            _plain_people = (_bare_frame and not sealed
                             and (_composed_fresh or not _still_there) and not any(
                (n in (restrained_who or ()))
                or (n in _state.people and _state.people[n].hardware)
                or any(n in (gone_by.get(t) or ()) for t in gone)
                for n in (active or [])))
            if _plain_people and not plan:
                # The opening shot of an empty place: whoever the guard fell back on was
                # never drawn, so the next beat that names them introduces them.
                _seen_before.difference_update(active or [])
                _in_frame = []
                shot_frames[len(plan)] = ([], [])
            _pose_now = posture_in(_acted, active if character_guard and active
                                   else [n for n, _ in sheet_lines(_who_sheet) if n])
            _cleared_now = posture_cleared(_acted, poses)
            for _gone_pose in _cleared_now:
                poses.pop(_gone_pose, None)
                lying_on.pop(_gone_pose, None)
                facing = ""          # up off the floor is no longer facing anywhere
            # HER OWN ACTION TAKES OVER FROM A CARRIED POSTURE. "Mara kneels to tie her
            # shoe", then "Mara sprints down the path" -- and the shot was told "Mara is
            # kneeling" beside the sprint, because sprinting is not on the travel list.
            # REPORTED as characters held in poses the beat did not ask for. Somebody
            # free whose beat gives them an action of their own is directed by it; the
            # open-ended carry stays for a restrained body and a body lying down.
            _held_bodies = set(_hw_by_wearer) | set(restrained_who or ())
            for _n, _p in list(poses.items()):
                if ((_p != "lying down" and _n not in _held_bodies
                     and own_action(_acted, _n, _who_sheet, active or [_n]))
                        or carried_off(_acted, _n, _who_sheet, active or [_n])):
                    poses.pop(_n, None)
                    lying_on.pop(_n, None)
            _poses_held = {n: p for n, p in poses.items() if n not in _pose_now}
            _posture = ("" if (not hold_scene_state or _plain_people)
                        else posture_hold({n: p for n, p in poses.items()
                                           if n not in _pose_now},
                                          active if character_guard else
                                          [n for n, _ in sheet_lines(_who_sheet) if n],
                                          upright=set(_hw_by_wearer) | set(restrained_who or ())))
            if _posture:
                posture_shots.append(len(plan) + 1)
            poses.update(_pose_now)
            for _n, _p in _pose_now.items():
                _on = _LYING_SURFACE.search(engine.staged_text(_acted) or "")
                if _p != "lying down":
                    lying_on.pop(_n, None)
                elif _on:
                    lying_on[_n] = _on.group(1).lower()
            _anchor_now = limb_anchor(_acted) if restrained else ""
            if _anchor_now:
                anchored = _anchor_now
            _legs_now = legs_anchor(_acted) if restrained else ""
            if _legs_now:
                legs_held = _legs_now
            # PER WEARER, from that wearer's own piece. One latched position was put on
            # every restrained person in the shot, so Ana, cuffed in front, was said to
            # have her arms behind her like Mara, and Mara's free ankles were said to
            # be tied together like Ana's. REPORTED.
            for _n, _r in (_ch.get("applied") or []):
                _pt = getattr(_r, "part", "")
                if _pt in ("wrists", "arms", "hands", "elbows"):
                    arms_of[_n] = (_anchor_now.split(", at the")[0].strip()
                                   or getattr(_r, "position", "") or "")
                elif _pt in ("ankles", "legs", "knees", "thighs", "feet"):
                    legs_of[_n] = _legs_now
                # ...and whatever else the same beat places for the one it binds: a
                # hogtie is one cable on the wrists that draws the ankles up too.
                if _legs_now and _n not in legs_of:
                    legs_of[_n] = _legs_now
                if _anchor_now and _n not in arms_of:
                    arms_of[_n] = _anchor_now.split(", at the")[0].strip()
            # ANYTHING CLOSED OVER THE GROIN STAYS UNTIL A REMOVAL NAMES IT. Latched
            # like anchored above and cleared in the same place, so it outlives the
            # beat that applied it -- the reported failure was the tape being there
            # in one shot and the genitals bare in the next.
            # NOT ON THE BEAT THAT TAKES IT OFF. "Dan unlocks the chastity belt"
            # names the belt, so the detector fires on the removal too and would
            # re-latch the thing that was just removed, one line after the clear
            # above wiped it. The removal wins: it is the author asking.
            _sealed_now = "" if _seal_off else crotch_seal(_acted, worn_items)
            if _sealed_now:
                sealed = _sealed_now
            for _n, _r, _l, _nw in _held_line:
                if _r.part in ("wrists", "arms", "hands", "elbows") and _r.position:
                    arms_of[_n] = _r.position
                elif _r.part in ("ankles", "legs", "knees", "thighs", "feet") and _l:
                    legs_of[_n] = _l
            held_rows.append((len(plan) + 1, held_state(_state, sealed)))
            _unheld = unregistered_restraints(_acted, _state, _ch)
            if _unheld:
                unheld_words.append((len(plan) + 1, _unheld))
            _holding = bool(restrained and anchored and not _anchor_now)
            if _holding:
                anchored_shots.append(len(plan) + 1)
            if _holding and (_anchor_tight or tight_framing(body)):
                tight_shots.append(len(plan) + 1)
            turn = TURN_HOLD if (rotates_in(_acted)
                                 and (gone or shown or restrained)) else ""
            _falls = falls_in(_acted)
            fall = (FALL_HOLD if (restrained and _falls)
                    else FALL_HOLD_FREE if _falls else "")
            _bound_fall = bool(restrained and _falls)   # placed with the pose, below
            if fall:
                fall_shots.append(len(plan) + 1)
            rigid = restrained and rigid_latched
            chain = (CHAIN_POSE_HOLD if (rigid and posed)
                     else CHAIN_HOLD if rigid else "")
            _off_here = (restraint_coming_off(_acted)
                         or names_any(RESTRAINT_HOLD_KEY, toks)
                         or any(restraint_present(t) for t in toks))
            anchors = ("" if (_off_here or hardware_handled(_acted))
                       else anchor_clause(unanchored_hardware(
                           _acted, gear=bool(restrained or sealed))))
            if anchors:
                notes.append(f"shot {i_shot + 1} names hardware with no body part beside "
                             f"it, so the shot says where it sits: "
                             f"{anchors.split(': ', 1)[1].rstrip('.')}")
            # The opening paragraph's staging is the opening shot's -- see
            # scene_staging. Every shot that opens on the last frame has it withheld;
            # a cut, a re-entry or a restart composes afresh and gets it whole, and the
            # shot that first shows somebody the paragraph placed keeps their sentence.
            # Only the text the model is told: every reader in this file still has all
            # of shot_scene.
            _k_now = len(plan)
            _opens_on_frame = bool(_k_now and _k_now not in cut_shots
                                   and _k_now not in reentry_shots
                                   and not (restart_after_removal
                                            and (_k_now - 1) in stripped_shots))
            # LYING, TRACKED, NOT ASSUMED. Anyone with no recorded pose counted as
            # lying -- including somebody a walk had just got up -- so the opening
            # "Mara lies on the bed reading a paperback" went on being stamped after
            # she crossed to the window. REPORTED. Seeded from the paragraph; a posture
            # cleared or changed, an action of their own or a move ends it.
            scene_lying -= set(_cleared_now)
            scene_lying -= {n for n, _p in _pose_now.items() if _p != "lying down"}
            scene_lying -= set(_movers or ())
            scene_lying -= {n for n in list(scene_lying)
                            if own_action(_acted, n, _who_sheet, active or [n])}
            _scene_text = (withhold_staging(
                shot_scene, _scene_staged,
                keep=(set(_present_shots.get(_k_now, [])) | set(restrained_who or ())
                      | {n for n, _q in _state.people.items() if _q.hardware}),
                lying=scene_lying)
                if (_opens_on_frame and _scene_staged) else shot_scene)
            if _plain_people:
                _cast_names = [n for n, _ in sheet_lines(sheet) if n]
                _scene_text = "\n".join(
                    ln for ln in _scene_text.split("\n")
                    if not any(re.match(r"\s*" + re.escape(n) + r"\s*:", ln)
                               for n in _cast_names))
            _scene_sent, _held_rooms, _held_blocked, _held_text = scene_for_here(
                _scene_text, here, anchor,
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
                _moves = state_changes(_acted)
                _acting = [_state_key(t) for t, _ in _moves]
                _pairs = [(t, s) for t, s in stated_states(line)
                          if _state_key(t) not in state_acted and _state_key(t) not in _acting]
            _turn = direction_anchor(_moves)
            _state_clause = state_hold(_pairs[:max(0, 2 - _turn.count("first frame"))]) + _turn
            if _pairs:
                stated_shots.append(len(plan) + 1)
            if _turn:
                turned_shots.append(len(plan) + 1)
            if _pairs and exits_vehicle(_acted) and any(
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
            # WHICH WAY UP, held like the posture itself. The beat that lays her
            # down names it; every beat after it does not, and an unnamed facing is
            # one the prior picks -- which is how a woman laid face down came back
            # over onto her back in the following shot.
            _face_now = lying_facing(_acted)
            if _face_now:
                facing = _face_now
            _lying_now = any(_p == "lying down" for _p in poses.values())
            if engine.posture_in(_acted):
                beat_said_posture = True
            if not _lying_now and restrained and not beat_said_posture:
                _watch = restrained_who or set()
                _free = (not any(n in poses for n in _watch)) if _watch else (not poses)
                # ONLY A PERSON LIES DOWN, and only where the scene paragraph says so: "a
                # phone lies on the nightstand" and a sheet's "hair that lies loose" laid
                # her down from the first shot while she stood. REPORTED.
                _cast_rx = [re.escape(n) for n, _ in sheet_lines(sheet) if n]
                _para = "\n".join(
                    ln for ln in str(_scene_for_state or "").split("\n")
                    if not (_cast_rx and re.match(r"\s*(?:" + "|".join(_cast_rx) + r")\s*:", ln)))
                _names = ([n for n, _ in sheet_lines(sheet) if n]
                          or sorted(n for n in _watch if n))
                _down_here = (any(p == "lying down" for n, p in posture_in(_para, _names).items()
                                  if not _watch or n in _watch)
                              if _names else engine.posture_in(_para) == "lying down")
                if _free and _down_here:
                    _lying_now = True
            # UNCUFFED IS UNCUFFED. Once the last piece on a limb comes off, the arms
            # are not "behind the body, wrists together" any more -- that was read off
            # the beat that cuffed them, and it outlived the cuffs. Held until a new
            # piece goes on a limb.
            _LIMB = ("wrists", "ankles", "arms", "hands", "legs", "elbows", "knees",
                     "thighs", "feet")
            # ...PER PAIR OF LIMBS. Counted over every limb at once, uncuffing her wrists
            # while her ankles stayed roped freed nothing, and every shot after went on
            # holding her uncuffed arms "behind the body, wrists together". The arms go
            # when nothing holds an arm, the legs when nothing holds a leg.
            _ARM_PARTS = ("wrists", "arms", "hands", "elbows")
            _LEG_PARTS = ("ankles", "legs", "knees", "thighs", "feet")

            def _held_on(parts):
                return any(r.part in parts for _q in _state.people.values()
                           for r in _q.hardware.values())

            def _changed(key, parts):
                return any(getattr(r, "part", "") in parts for _n, r in (_ch.get(key) or []))

            def _limb_of(part):
                return "arms" if part in _ARM_PARTS else "legs" if part in _LEG_PARTS else ""
            # For pose control: which limbs go ON whom in this shot, by the beat or by a
            # hold: line, and whether limb hardware comes OFF. More is added below, once
            # the beat's own limb positions and the held keys are read. See
            # pose_candidate.
            _limbs_on = {}
            for _n, _r in (_ch.get("applied") or []):
                if _limb_of(getattr(_r, "part", "")):
                    _limbs_on.setdefault(_n, set()).add(_limb_of(_r.part))
            for _n, _r, _l, _nw in _held_line:
                if _nw and _limb_of(getattr(_r, "part", "")):
                    _limbs_on.setdefault(_n, set()).add(_limb_of(_r.part))
            _limbs_off = _changed("released", _ARM_PARTS + _LEG_PARTS)
            if _changed("released", _ARM_PARTS) and not _held_on(_ARM_PARTS):
                arms_freed = True
            if _changed("applied", _ARM_PARTS):
                arms_freed = False
            if _changed("released", _LEG_PARTS) and not _held_on(_LEG_PARTS):
                legs_freed = True
            if _changed("applied", _LEG_PARTS):
                legs_freed = False
            limbs_freed = arms_freed and legs_freed
            if arms_freed:
                anchored = ""
                posed = False
                _anchor_now = ""
            if legs_freed:
                legs_held = ""
                _legs_now = ""
            _pose_pos = ("" if arms_freed else
                         (_anchor_now or anchored
                          or (limb_anchor(_scene_for_state) if restrained else "")))
            _legs_pos = ("" if legs_freed else
                         (_legs_now or legs_held
                          or (legs_anchor(_scene_for_state) if restrained else "")))
            _arms_pos = _pose_pos.split(", at the")[0].strip()
            if not _arms_pos and _legs_pos == "ankles to the wrists":
                _arms_pos = "behind the back"
            _pose = pose_clause(_arms_pos, lying=_lying_now, legs=_legs_pos,
                                facing=facing)
            # WHERE THE LIMBS ARE GOES IN THE OPENING TOKENS, beside the beat, not
            # eight sentences down with the other continuity clauses.
            #
            # This file already measured the rule and built beat_leads on it: what
            # LEADS a prompt decides its composition, anatomy in the opening tokens is
            # what a distilled model settles the frame on, and at cfg 1 no later
            # sentence outvotes it. The limb position was then left sitting after the
            # sheet, the body count, the hardware clause and the chain clause -- a
            # later sentence, by the file's own finding.
            #
            # Reported, and not fixed by any amount of saying it: wrists cuffed in
            # FRONT of the body on every shot that uses handcuffs, while the text said
            # behind the back five times over. Five late sentences lose to the opening
            # tokens; one early sentence is the thing the rule says wins. It is moved,
            # not duplicated -- the guard list drops its copy, so the words are the
            # same words and only their position changed, which is the one lever here
            # that has ever moved composition.
            _mat_item, _mat = (hardware_material(worn_items, f"{body} {shot_sheet}")
                               if restrained else ("", ""))
            _material = hardware_material_clause(_mat_item, _mat)
            # pose_clause carries the facing where there IS an arm position; this
            # is for a lying body that has none, so the facing is still said.
            _facing = (facing_clause(facing)
                       if (_lying_now and not _arms_pos) else "")
            _seal = SEALED_HOLD.format(item=sealed) if sealed else ""
            _seal_led = False
            _pose_led = ""
            _hw_text = " ".join(worn_items or [])
            try:
                _worn_where = hardware_where([_r for _p in _state.people.values()
                                              for _r in _p.hardware.values()])
            except Exception:
                _worn_where = None
            _rigid_tail = ("" if not rigid else
                           cuff_rigid_sentence(cuff_part(worn_items, _worn_where)
                                               or held_part(worn_items))
                           if (_hw_text and _CUFF_FORM.search(_hw_text)
                               and not re.search(r"\bchain", _hw_text, re.I))
                           else CHAIN_RIGID_TAIL)
            hold = (RESTRAINT_GOING_ON + _rigid_tail + _ends_at
                    if _applying
                    else chain if chain else (RESTRAINT_HOLD if restrained else ""))
            if _applying:
                applied_shots.append(len(plan) + 1)

            _staged_here = displaced_garments(_acted, shot_scene)
            if _staged_here:
                staging_shots.add(len(plan) + 1)
            for _g, _how in _staged_here:
                _prev_state = displaced.get(_g, "")
                # Put back up again is a restore, not a new displacement.
                if _prev_state == "pulled down" and _how in ("pulled up", "pulled back"):
                    displaced.pop(_g, None)
                else:
                    displaced[_g] = _how
                    displaced_dest[_g] = displaced_to(_acted, _g)
            for _g in [g for g in displaced if names_any(g, toks)]:
                displaced.pop(_g, None)
            for _g in restored_garments(_acted, shot_scene):
                _head = str(_g).lower().split()[-1]
                for _k in [k for k in displaced
                           if str(k).lower().split()[-1] == _head]:
                    displaced.pop(_k, None)
            if len(displaced) == 1 and puts_it_back(_acted):
                displaced.clear()
            _body_low = (body or "").lower()
            _moved = displaced_hold(
                [(g, h) for g, h in displaced.items()
                 if not re.search(r"\b" + re.escape(g.split()[-1]) + r"\b", _body_low)],
                dest=displaced_dest,
                beneath={g: under_displaced(g, shot_sheet or sheet, displaced)
                         for g in displaced})
            # ...and on the shot that moves it, what it uncovers -- unless the beat says.
            for _g, _how in _staged_here:
                for _u in under_displaced(_g, shot_sheet or sheet, displaced):
                    if re.search(r"\b" + re.escape(_u.split()[-1]) + r"\b", _body_low):
                        continue
                    _pl = _g.endswith("s") and not _g.endswith("ss")
                    _moved += (f" The {_u} under the {_g} comes into view as "
                               f"{'they move' if _pl else 'it moves'}, and stays on.")
            if _moved:
                moved_shots.append(len(plan) + 1)

            # ...and only somebody the state has in a piece: the sheet line that names
            # a strap or a chain is not always one wearing it. See the latch above.
            _wearers = [n for n in restraint_wearers(shot_sheet)
                        if (not character_guard or n in active)
                        and (not _hw_by_wearer or n in _hw_by_wearer)]
            _described = (active if character_guard else
                         [n for n, _ in sheet_lines(shot_sheet) if n])
            if extras_in(body, singular=False):
                _extras_seen = True
            elif extras_dismissed(body):
                _extras_seen = False

            _look_now = (look_target(_acted, shot_sheet, _described)
                         if hold_gaze else "")
            _lookers = (subjects_for(_acted, shot_sheet, _LOOK_VERB_SRC)
                        if hold_gaze else [])
            _look_is_person = bool(_look_now) and any(
                _look_now == _n for _n, _ in sheet_lines(shot_sheet))
            if _look_now:
                for _n in (_lookers or (_described or [])):
                    looking_at[_n] = (_look_now, _look_is_person)
            elif (looks_somewhere(_acted) or arrives_in(_acted) or falls_in(_acted)
                  or turns_in(_acted, cast) or _MOVES_OFF.search(_acted or "")):
                _ends = (_lookers
                         or subjects_for(_acted, shot_sheet, _MOVES_OFF_SRC))
                for _n in (_ends or list(looking_at)):
                    looking_at.pop(_n, None)
            _gone_now = (set(leaves_in(_acted, sheet, _shows))
                         | set(subjects_for(_acted, shot_sheet, _MOVES_OFF_SRC)))
            if _gone_now:
                for _n, (_t, _is_person) in list(looking_at.items()):
                    if _is_person and _t in _gone_now:
                        looking_at.pop(_n, None)
            if _ALONE.search(engine.staged_text(body)):
                looking_at.clear()
            _carried = [n for n in _was
                        if n not in set(_described or []) and looking_at.get(n)
                        and n not in set(subjects_for(_acted, sheet, _MOVES_OFF_SRC))]
            # ONLY ON THE SHOT THAT WRITES THE LOOK. A look was held until something
            # ended it, so "Mara looks at the door" put her eyes on the door through
            # every later shot -- kissing, being cuffed, walking out -- and a looker who
            # had left the beat was named into the shot to keep looking. REPORTED as
            # characters doing things the beat never wrote. The beat that stages a look
            # gets it said back; the shots after it are left to their own beats.
            _gazers = ([n for n in dict.fromkeys(list(_lookers or []) + list(_described or []))
                        if n in set(_described or []) | set(_lookers or [])
                        and (looking_at.get(n) or ("",))[0] == _look_now]
                       if _look_now else [])
            _gaze = ""
            _faces = ""     # the eye-line inferred for a dialogue shot
            _contact = (contact_hold(contact_pairs(_acted, _described))
                        if len(_described or []) > 2 else "")
            if _contact:
                contact_shots.append(len(plan) + 1)
            # NOT WHERE THE OPENING FRAME ALREADY IS THE FRAME. A shot that opens on the
            # last shot's final frame, with the camera held, has its framing fixed twice
            # over; "head to feet, with the room around them" on top of that asks the
            # held camera for a wider view than the frame it opens on, and a take cannot
            # be both. The model settles it by cutting -- to the wide profile that shows
            # two whole bodies, in a room drawn fresh because the keyframe never showed
            # that much of it. Reported as sex scenes turning into side-angle shots in a
            # different location: climbs, lifts, pulls and pushes are all whole-body verbs
            # here. Only where the camera IS held: a journey's camera goes with them and
            # the keyframe pins only where they set off, and an author's own camera move
            # is theirs to frame.
            # A move inside the room keeps the hold: a still camera can watch somebody
            # cross to the fireplace, and freeing it there was the camera wandering.
            _camera = camera_hold(body, anchor, moving=_journey) if hold_camera else ""
            if _camera:
                camera_shots.append(len(plan) + 1)
            # A light changing, a journey, or the author's own camera move changes the
            # picture's levels over the take on purpose. Anything else is the chain
            # cooking, and the render takes it back out. See shot_grade.
            if _journey or light_changes(_acted) or _CAMERA_ASKED.search(body or ""):
                own_grade_shots.add(len(plan))
            _k = len(plan)
            _on_keyframe = bool(
                (first_frame is not None and not _first_is_plate) if _k == 0 else
                (_k not in cut_shots and _k not in reentry_shots
                 and not (restart_after_removal and (_k - 1) in stripped_shots)))
            # Everyone described, whole -- see frame_hold. Only where the frame is
            # composed afresh: shot 1, a cut, a restart, a journey. A shot opening on
            # the last frame already has its framing, and the camera hold keeps it.
            # Nobody, where the beat puts nobody on screen.
            # ...and everyone the shot describes, whether or not the beat names them:
            # a restrained woman held in a shot of the rain is a body in the frame.
            # REPORTED as the whole-body hold lost on such a shot.
            _bodies = (len(_described) if (_described and not _plain_people) else
                       (1 if beat_puts_somebody_on_screen(body, sheet) else 0))
            _frame = frame_hold(body, anchor, _bodies, outdoor=_outside,
                                held=bool(_on_keyframe and not _journey))
            if _frame:
                frame_shots.append(len(plan) + 1)
            _place_words = 0            # words the hold spends saying where -- see _placed_hold
            _wearer_here = (not restrained_who
                            or not character_guard
                            or not (_described or [])
                            or bool(restrained_who & set(_described or [])))
            # THE GAG, AND ANY PIECE GOING ON NOW, each in words of its own -- see
            # gag_hold and newly_on_clause. Both leave the hardware sentence below,
            # which says "closed and fastened as it was put on": true of cuffs, and of
            # neither a strip of tape nor a rope going on in this very shot.
            _gag_on = (gagged_in(_state, _described or list(_state.people))
                       if (restrained and _wearer_here) else {})
            _mouth_raw = {_r.item for _q in _state.people.values()
                          for _r in _q.hardware.values() if _r.part == "mouth"}
            _fresh = [i for _n, i in sorted(_new_on)
                      if i not in _mouth_raw and (_n in (_described or []) or not _described)]
            # PER PERSON: Kate's cuffs going on now are not Mara's, which have been on
            # since shot 1 -- skipped by name, the second cuffing took the first
            # woman's hold out of the shot.
            _skip_by = {}
            for _n, _i in _new_on:
                if _i not in _mouth_raw:
                    _skip_by.setdefault(_n, set()).add(_i)
            for _n in _gag_on:
                _skip_by.setdefault(_n, set()).update(
                    _r.item for _r in _state.people[_n].hardware.values()
                    if _r.part == "mouth")
            # ...and the sealed piece, which SEALED_HOLD says in full.
            _groin_raw = {_r.item for _q in _state.people.values()
                          for _r in _q.hardware.values() if _r.part == "groin"}
            if sealed and _groin_raw:
                _fresh = [i for i in _fresh if i not in _groin_raw]
                for _n, _q in _state.people.items():
                    _g = {_r.item for _r in _q.hardware.values() if _r.part == "groin"}
                    if _g:
                        _skip_by.setdefault(_n, set()).update(_g)
            _skip = set().union(*_skip_by.values()) if _skip_by else set()

            def _shown(items, who=None):
                # One wearer's items and skips are the same raw names, so they match
                # exactly: by substring, the duct tape on her mouth also took the tape
                # on her wrists out of the hold. Reported as a restraint breaking.
                _s = _skip if who is None else _skip_by.get(who, set())
                return [i for i in (items or [])
                        if not any(s == i or (who is None and (s in i or i in s))
                                   for s in _s)]
            _hw_shown = {_n: _shown(_v, _n) for _n, _v in _hw_by_wearer.items()}
            if not _wearer_here:
                hold = ""
                _pose = ""
                _facing = ""
                if (len(plan) + 1) in anchored_shots:
                    anchored_shots.remove(len(plan) + 1)
                absent_hold.append(len(plan) + 1)
            elif not _applying and restrained:
                _here_items = merge_hardware_names(
                    [i for n in (_described or []) for i in _hw_shown.get(n, [])]
                ) or _shown(worn_items)
                _here_item = ", ".join(_here_items)
                _here_rigid = bool(rigid) and (rigid_hardware(_here_item)
                                               if _here_item else True)
                # ONE SENTENCE PER WEARER WHERE THEY WEAR DIFFERENT THINGS. Pooled,
                # two restrained people in one shot read "The steel handcuffs and
                # leather collar on Ana and Mara stay closed", which says both pieces
                # and both bodies and never which goes on which -- and hardware named
                # on a body that is not wearing it is hardware the model draws there.
                # It used to be rare, because a shot naming one person described only
                # her; now that a fastened person stays in frame it is the ordinary
                # case, so the sentence has to carry the attribution.
                _own = [n for n in (_described or []) if _hw_shown.get(n)]
                # ...and whose, beside a piece going on somebody: "The cuffs stay
                # closed" next to "the cuffs go on Kate's wrists" reads as hers.
                _split = ((len(_own) > 1
                           and len({tuple(merge_hardware_names(_hw_shown[n]))
                                    for n in _own}) > 1)
                          or bool(_fresh and _own and len(_described or []) >= 2))

                def _placed_on(names):
                    return hardware_where([_r for _n in names
                                           for _r in (_state.people[_n].hardware.values()
                                                      if _n in _state.people else ())])

                def _placed_hold(*a, where=None, **k):
                    """The hold with each piece's place, and how many words the places
                    cost -- given back to the shot's guard budget below, so saying
                    WHERE never pushes another guard out."""
                    with_place = restraint_sentence(*a, where=where, **k)
                    plain = restraint_sentence(*a, **k)
                    return with_place, max(0, len(with_place.split()) - len(plain.split()))
                if _split:
                    _parts = [
                        _placed_hold(
                            ", ".join(merge_hardware_names(_hw_shown[n])),
                            [n], _described,
                            anchor=("" if (_anchor_now or anchors_placed(_placed_on([n])))
                                    else anchored),
                            rigid=bool(rigid) and rigid_hardware(
                                " ".join(_hw_shown[n])),
                            posed=bool(posed),
                            part=held_part(merge_hardware_names(_hw_shown[n])),
                            where=_placed_on([n]))
                        for n in _own]
                    # EVERYONE ELSE ONCE, AND ONLY WHERE IT SAYS SOMETHING. Each wearer's
                    # sentence ended on "Everyone else in the shot has on exactly what
                    # their own entry lists", so two wearers said it twice -- and a split
                    # made only because a piece goes on now said it beside the one
                    # sentence that already names her, about somebody whose entry is the
                    # piece going on. REPORTED as gag shots getting heavier. Said once,
                    # after the sentences, where two or more wearers share the frame with
                    # somebody wearing none of it.
                    hold = "".join(h.replace(OTHERS_UNCHANGED, "") for h, _w in _parts)
                    _fresh_names = {_n for _n, i in _new_on if i in _fresh}
                    if len(_own) >= 2 and any(
                            n not in _own and n not in _fresh_names
                            for n in (_described or [])):
                        hold += OTHERS_UNCHANGED
                    _place_words = sum(w for _h, w in _parts)
                elif not _here_items and _skip:
                    hold = ""       # all of it is a gag or going on now -- said below
                else:
                    hold, _place_words = _placed_hold(
                        # NAMED EVEN WHEN THE BEAT NAMES IT. "Ana strains against the
                        # straps" blanked the item -- and with it where it is and what
                        # it is fastened to -- to "every restraint". REPORTED as the
                        # item and its placement dropped from the beats.
                        _here_item,
                        _wearers, _described,
                        # Each item carries its own anchor where the state has one --
                        # see _where_of -- and the one clause for the whole hold goes.
                        anchor=("" if (_anchor_now or anchors_placed(
                            _placed_on(_described or list(_state.people)))) else anchored),
                        rigid=_here_rigid, posed=bool(posed),
                        part=held_part(_here_items or ([_here_item] if _here_item else [])),
                        where=_placed_on(_described or list(_state.people)))
                if _here_item:
                    named_shots.append(len(plan) + 1)
                if _fresh:
                    _fresh_who = [n for n in (_described or []) if n in
                                  {_n for _n, i in _new_on if i in _fresh}]
                    hold += newly_on_clause(
                        _fresh, where=_placed_on(_described or list(_state.people)),
                        posed=bool(posed),
                        who=(_fresh_who[0] if len(_fresh_who) == 1
                             and len(_described or []) >= 2 else ""))
                    applied_shots.append(len(plan) + 1)
            else:
                hold = own_hold(hold, _wearers, _described)
            # HOISTED HERE, BELOW THE GATE ABOVE, and that is the whole of the fix.
            #
            # _wearer_here clears the pose and the hold for a shot the restrained
            # person is not described in -- a body that is not in the frame does not
            # get its arms placed. The hoist that puts the position in the opening
            # tokens was written ABOVE it, so the sentence was already spliced into
            # `line` by the time the gate cleared the variable, and clearing it no
            # longer removed anything. A shot reading "Dan: he, 40" and "There is one
            # person in the shot" also read "Both arms are behind the body, wrists
            # together at the small of the back" -- her position, on him. Reported as
            # the restraint changing mid-scene, and worst where beats name one of two
            # people and lean on continuity for the other.
            _pose_known = bool(_arms_pos or _legs_pos)
            _held_here = [n for n in (_described or []) if n in (restrained_who or ())]
            _bound_pose_now = {}        # name -> (arms, legs), on-screen wearers: pose control
            if _held_here and _wearer_here and (arms_of or legs_of):
                # Each wearer's own arms and legs; never one body's pose said of two
                # who differ. A wearer with no record of their own takes the shot's
                # latched one only where nobody here has a record either.
                def _has(_n, parts):
                    _q = _state.people.get(_n)
                    return _q is None or any(r.part in parts for r in _q.hardware.values())
                _line_of = dict(sheet_lines(shot_sheet))
                _recorded = any(n in arms_of or n in legs_of for n in _held_here)
                _groups = {}
                for _n in _held_here:
                    _a = (arms_of.get(_n) if _n in arms_of else
                          (limb_anchor(_line_of.get(_n, "")).split(", at the")[0].strip()
                           or ("" if _recorded else _arms_pos)))
                    _l = (legs_of.get(_n) if _n in legs_of else
                          (legs_anchor(_line_of.get(_n, ""))
                           or ("" if _recorded else _legs_pos)))
                    # A tie that draws the ankles to the wrists or the neck is held by
                    # whatever is still on the body, not only by a piece on the legs.
                    _tied_up = _l in ("ankles to the wrists", "ankles to the neck")
                    _any = (_state.people.get(_n) is None
                            or bool(_state.people[_n].hardware))
                    _a = _a if ((_has(_n, ("wrists", "arms", "hands", "elbows"))
                                 or (_tied_up and _any)) and not arms_freed) else ""
                    _l = _l if ((_has(_n, ("ankles", "legs", "knees", "thighs", "feet"))
                                 or (_tied_up and _any)) and not legs_freed) else ""
                    if not _a and _l == "ankles to the wrists":
                        _a = "behind the back"
                    _groups.setdefault((_a, _l), []).append(_n)
                    if _a or _l:
                        _bound_pose_now[_n] = (_a, _l)
                _pose = "".join(
                    pose_of(pose_clause(_a, lying=_lying_now, legs=_l, facing=facing),
                            _names, _described)
                    for (_a, _l), _names in _groups.items())
                _pose_known = bool(_pose)
            elif _pose and restrained_who:
                _pose = pose_of(_pose, _held_here, _described)
                if _wearer_here and (_arms_pos or _legs_pos):
                    _bound_pose_now = {_n: (_arms_pos, _legs_pos) for _n in _held_here}
            def _whose_in(_n):
                """How a lead sentence says whose: nothing with one person in the
                shot, a pronoun where only _n answers to it -- the pose sentence beside
                it has already named them, and a third naming is how a second one is
                drawn -- else the name."""
                if len(_described or []) < 2:
                    return ""
                _rows = dict(sheet_lines(shot_sheet))
                _mine = sheet_pronoun(_rows.get(_n, ""))
                if _mine in ("she", "he") and not any(
                        sheet_pronoun(_rows.get(_o, "")) == _mine
                        for _o in _described if _o != _n):
                    return {"she": "her", "he": "his"}[_mine]
                return _n
            # THE GAG LEADS, beside the pose and for the same reason: a mouth is the
            # first thing a face is drawn with, and a late sentence does not outvote it.
            _gag_talk = set(speakers_in(body, _who_sheet)) if has_speech(body) else set()
            _gag_voc = ({_n for _n, _p in vocal_sources_in(body, shot_sheet)}
                        if voice_in(body) else set())
            _gag_second = ""
            if voice_in(body) and len(_gag_on) == 1 and not (_gag_talk | _gag_voc):
                _gag_second = second_vocal(body, next(iter(_gag_on)), _described or [])
            _gag_loose = bool(voice_in(body) and not (_gag_talk | _gag_voc)
                              and (unpinned_vocal(body) or _gag_second)
                              and len(_gag_on) == 1)
            _muffled = {_n for _n in _gag_on
                        if _n in _gag_talk or _n in _gag_voc or _gag_loose}
            _gag = "".join(
                gag_hold(_it, who=_whose_in(_n),
                         new=any(_w == _n and _i in _mouth_raw for _w, _i in _new_on),
                         muffled=_n in _muffled)
                for _n, _it in _gag_on.items())
            if _gag:
                gag_shots.append(len(plan) + 1)
            for _n in sorted(_gag_on):
                if _n in _gag_talk:
                    notes.append(
                        f"shot {len(plan) + 1} gives {_n} a spoken line while the "
                        f"{_gag_on[_n]} covers {_n}'s mouth. A line is lip-synced, and "
                        f"lips that move to words are lips with nothing over them -- "
                        f"the shot is told the words come out muffled behind it, but a "
                        f"model asked for both usually drops the gag. If it should "
                        f"stay on, write the sound instead: '{_n} makes a muffled "
                        f"sound behind the tape'")
            # A BOUND BODY FALLING, with the hands placed for the fall itself, in the
            # opening tokens. See bound_fall_clause.
            _fall_led = False
            _down = (fallers_in(_acted, _described, _pron_of)
                     if _bound_fall else set())
            # ...and only a fall the bound body is IN: named, or -- where the beat names
            # nobody it can resolve -- the bound people here. NEVER THE FREE FALL FOR A
            # BODY IN LIMB HARDWARE: "Mara slips and falls" beside Dan read as nobody,
            # and the free sentence left her hands to catch the landing. REPORTED.
            _limbed = {n for n, _q in _state.people.items()
                       if any(r.part in _LIMB for r in _q.hardware.values())}
            _bound_down = {n for n in (_down or set(_described or []) or set(restrained_who or ()))
                           if n in (restrained_who or ()) and n in _limbed}
            if _bound_fall and not _bound_down:
                # A gag or a collar binds no limb: her hands are free, and a sentence
                # keeping "the arms in the hold" puts them in one nobody fastened. And
                # when the one falling is somebody else, the bound body is not falling.
                fall, _bound_fall = FALL_HOLD_FREE, False
            if _bound_fall and _wearer_here:
                _fallers = [n for n in (_described or []) if n in _bound_down] or sorted(_bound_down)
                _arm_held = not arms_freed and any(
                    r.part in _ARM_PARTS for _f in _fallers
                    for r in (_state.people[_f].hardware.values()
                              if _f in _state.people else ()))
                _fall_arms = (arms_of.get(_fallers[0], "")
                              if (_arm_held and len(_fallers) == 1) else "") or _arms_pos
                # CUFFS WITH NO POSITION STILL PLACE THE HANDS for the fall: "the arms
                # staying in the hold" placed nothing, and the hands went out to catch
                # it. Behind the back, unless the beat puts them in front.
                if not _fall_arms and _arm_held:
                    _fall_arms = ("in front of the body"
                                  if re.search(r"\bin\s+front\b", _acted or "", re.I)
                                  else "behind the back")
                fall = bound_fall_clause(
                    _fall_arms, _legs_pos,
                    who=(_whose_in(_fallers[0]) if len(_fallers) == 1 else ""))
            # A BODY LEFT LYING, held down -- see lying_stays.
            _down_lead, _lying_led = "", set()
            for _n in (_described or []):
                if _poses_held.get(_n) != "lying down":
                    continue
                # ONLY A RESTRAINED BODY: anybody else lying down keeps the plain
                # posture hold ("Mara is lying down."), as before.
                _going_on_her = any(_w == _n for _w, _i in _new_on)
                if _n not in (restrained_who or ()) and not _going_on_her:
                    continue
                # ...AND ONLY WHILE SHE IS BEING WORKED ON: a restraint going on her or
                # coming off her, or somebody's hands on her. It sat straight after the
                # beat on every later shot -- "Dan checks his phone" pinned her flat to
                # the bed through the whole shot -- the strongest place in the prompt,
                # spent on a beat that never mentions her. REPORTED as characters held
                # in poses the beat did not ask for. The reported case it exists for is
                # being cuffed or taped while lying; any other shot keeps the plain
                # posture hold.
                _coming_off_her = any(_w == _n for _w, _r in (_ch.get("released") or []))
                if not (_going_on_her or _coming_off_her
                        or handles_person(_acted, _n, shot_sheet, _described)):
                    continue
                _subj = _whose_in(_n)       # "her"/"his", the name, or "" when alone
                _who = ({"her": "She", "his": "He"}.get(_subj, _subj) if _subj else "")
                _down_lead += lying_stays(_who, lying_on.get(_n, ""))
                _lying_led.add(_n)
            if _down_lead:
                lying_shots.append(len(plan) + 1)
                # ...and not said a second time, late and short, by the posture hold.
                _posture = ("" if not hold_scene_state
                            else posture_hold({n: p for n, p in _poses_held.items()
                                               if n not in _lying_led},
                                              active if character_guard else
                                              [n for n, _ in sheet_lines(_who_sheet) if n],
                                              upright=set(_hw_by_wearer)
                                              | set(restrained_who or ())))
                if not _posture and (len(plan) + 1) in posture_shots:
                    posture_shots.remove(len(plan) + 1)
            # THE FRAME NO LONGER LEADS. It led for the reason the pose does -- as guard
            # 15 of 15 it was the first thing a crowded shot dropped -- but it now goes
            # only on a shot composed afresh (see frame_hold), where it sits mid-list at
            # priority 8, behind the beat's own words instead of ahead of them.
            _speaks = has_speech(body)
            _in_shot = (_described if character_guard else
                        [n for n, _ in sheet_lines(_who_sheet) if n])
            # Only the person an order or an intention is about, and only for an action
            # the shot could stage early -- see deferred_holds and told_hold.
            _defer = deferred_holds(body, _in_shot, _acted, _who_sheet,
                                    speakers=speakers_in(body, _who_sheet) if _speaks else ())
            # Not where the shot DOES something -- a removal or a fastening, written
            # or directed: "asks Dan to take the belt off" with "remove: belt" is Dan
            # taking it off, now, and the ask is how the author said so.
            if toks or _applying:
                _defer = []
            _rows_told = dict(sheet_lines(_who_sheet))
            _told = told_hold([n for n, _k in _defer],
                              pronouns={n: sheet_pronoun(_rows_told.get(n, ""))
                                        for n, _k in _defer},
                              wearing={n for n, _k in _defer if _k == "wear"})
            if _told:
                told_shots.append(len(plan) + 1)
            # ONE OR THE OTHER for a person: "Dan stays where he is" beside "Dan comes
            # into the frame from its edge" asks for both. REPORTED. The entrance is
            # what the frame needs; the stay goes.
            _enter_names = [n for n in (_placed_shots.get(len(plan)) or [])
                            if n not in _present_shots.get(len(plan), [])]
            if _told and any(n in _enter_names for n, _k in _defer):
                _defer = [(n, k) for n, k in _defer if n not in _enter_names]
                _told = told_hold([n for n, _k in _defer],
                                  pronouns={n: sheet_pronoun(_rows_told.get(n, ""))
                                            for n, _k in _defer},
                                  wearing={n for n, _k in _defer if _k == "wear"})
                if not _told and (len(plan) + 1) in told_shots:
                    told_shots.remove(len(plan) + 1)
            _entering = entrance_clause(_enter_names)
            _told_led = False
            # THE SEAL, said on the right body and at the right time. The shot that puts
            # it on is told both ends -- off at the first frame, on and staying on once
            # it is on -- like any other piece going on; the old sentence said it was
            # there all shot, before anybody had wrapped it.
            if sealed and _seal:
                _sealers = [n for n in (_described or []) if n in _state.people
                            and any(_r.part == "groin"
                                    for _r in _state.people[n].hardware.values())] \
                    or [n for n in (_described or []) if n in (restrained_who or ())]
                _sw = _whose_in(_sealers[0]) if len(_sealers) == 1 else ""
                _sp = _sw if _sw in ("her", "his") else (f"{_sw}'s" if _sw else "the")
                _seal = (SEALED_ON if not _sealed_before else SEALED_HOLD).format(item=sealed)
                if _sp != "the":
                    for _w in ("waist", "legs", "groin"):
                        _seal = _seal.replace(f" the {_w}", f" {_sp} {_w}")
            _gag_led = False
            _fall_lead = fall if (_bound_fall and _wearer_here) else ""
            if (_pose or _seal or _facing or _entering or _told or _gag
                    or _fall_lead) and body:
                _at = line.find(body)
                if _at >= 0:
                    _cut = _at + len(body)
                    # What the line only ASKS for stays out of the shot: said right
                    # after it, where it cannot be crowded out -- see told_hold.
                    # The legs lead as the arms do: bound ankles with free hands are
                    # still a pose, and late in the shot it was the one that gave.
                    # Not the lying hold: it goes in the guard list, behind the
                    # beat's own words -- see lying_stays above. Nor the frame: see
                    # frame_hold.
                    # THE FALL BEFORE THE LANDING. On the shot that puts her down, the
                    # lying sentence went ahead of the fall, so the shot opened on a body
                    # already flat and the fall read as an afterthought. It follows it.
                    _pose_lead, _lying_lead = (_pose if _pose_known else ""), ""
                    if _fall_lead and _pose_lead:
                        _pose_lead, _lying_lead = split_lying(_pose_lead)
                    _lead = (_told + _entering + _pose_lead
                             + _fall_lead + _lying_lead + _gag + _facing + _seal)
                    _told_led = bool(_told)
                    line = (line[:_cut] + _lead + line[_cut:]).strip()
                    _pose_led = _pose if _pose_known else ""
                    _seal_led = bool(_seal)
                    _gag_led = bool(_gag)
                    _fall_led = bool(_fall_lead)
            _speaks = has_speech(body)
            _own = sound_described(body)
            if not _own and not _speaks and _BREATH_PREP.search(body):
                _breath_shots.append(len(plan) + 1)
            _voiced = voice_in(body)     # a voice opens the branch; effort alone does not
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
            # Held per the engine state: binding hardware on a described person -- see
            # duress_face. A collar alone holds nobody.
            _held_now = [n for n in (_described or []) if n in _state.people
                         and any(_BOUND_HARDWARE.search(f"{_k} {getattr(_r, 'item', '')}")
                                 for _k, _r in _state.people[n].hardware.items())]
            _duress = (duress_face(
                body, _described, sheet=shot_sheet, held=_held_now,
                applied_to={_n for _n, _i in _new_on}) if hold_gaze else "")
            # A face under a gag acts ABOVE it. "Played in the eyes and the mouth" is
            # a mouth the tape is over, and the face won.
            if _duress and _gag_on:
                _faces_of = ([_n for _n, _w in emotion_pairs(body, _described, shot_sheet)]
                             or _held_now or list(_described or []) or list(_gag_on))
                for _gagged_face in [_n for _n in _faces_of if _n in _gag_on][:2]:
                    _duress = eyes_above(_duress, _gag_on[_gagged_face], who=_gagged_face)
            if _duress:
                duress_shots.append(len(plan) + 1)
            _mouth_busy = bool(mouth_performs(body) or emotion_in(body))
            # NOT BESIDE THE MACHINE'S VOICE. device_voice_clause already closes the
            # room's mouths ("their own mouths closed"), so the shot said it twice.
            # REPORTED as guards crowding the beat. The silence pin is untouched: a
            # device line opens the branch for the device, as it always did.
            _mouth = MOUTH_HOLD if (mouths_shut_when_no_line and _has_people
                                    and not _speaks
                                    and not _voiced and not _mouth_busy
                                    and not _MOUTH_EATS.search(_acted or "")) else ""
            # A MOUTH HELD OPEN BY ITS GAG -- a ball, a bit, a ring, a stuffed cloth -- is
            # not told to stay closed: the model can only do that by drawing no gag.
            # REPORTED. Every other mouth in the shot keeps the line.
            _held_open = [n for n, _it in _gag_on.items() if engine.mouth_held_open(_it)]
            if _mouth and _held_open:
                _mouth = mouths_closed_but(
                    _held_open, [n for n in (_described or []) if n not in _held_open])
            _mouth_from_silence = bool(_mouth)
            _vocal_src = vocal_sources_in(body, shot_sheet) if _voiced else []
            if _voiced and not _vocal_src and _gag_second:
                _vocal_src = [(next(iter(_gag_on)), _gag_second)]
            # A sound behind a gag is that sound MUFFLED: "the screaming is Mara's"
            # opened the mouth the tape was over, and the tape went.
            _vocal_src = [(_n, f"muffled {_p}" if _n in _muffled else _p)
                          for _n, _p in _vocal_src]
            _voicers = [n for n, _ in _vocal_src]
            _vocal_word = _vocal_src[0][1] if _vocal_src else ""
            # A busy mouth no longer stands the voice guard down: whoever does NOT
            # have the line is still told so, as silent rather than closed. See
            # MOUTH_SILENT_REST.
            # A SHOUTED LINE HAS A SPEAKER. "Dan shouts, ..." is a voice verb, and with
            # no vocal source credited the guard stood down -- no "Only Dan speaks", no
            # language, and the listener's mouth left free. REPORTED. It stands down
            # only where nobody at all can be credited.
            _talkers_pre = speakers_in(body, shot_sheet) if _speaks else []
            if (not _mouth and mouths_shut_when_no_line and (_speaks or _voicers)
                    and not _device_line
                    and not (_voiced and not _voicers and not _talkers_pre)):
                _talkers = _talkers_pre
                _open = set(_talkers) | set(_voicers)
                _here_too = [n for n in _was
                             if n not in set(_described or [])
                             and n not in set(subjects_for(_acted, sheet,
                                                           _MOVES_OFF_SRC))]
                _silent = [n for n in list(_described or []) + _here_too
                           if n not in _open]
                if _voiced and unpinned_vocal(body):
                    _silent = []        # "she moans" may be any of them -- see unpinned_vocal
                _mouth = voice_sources(_talkers, _vocal_word, _voicers, _silent,
                                       pairs=_vocal_src,
                                       rest=MOUTH_SILENT_REST if (
                                           _mouth_busy or set(_silent) & set(_held_open))
                                       else None)
                for _n in _muffled & set(_talkers):
                    _mouth = _mouth.replace(
                        f"nly {_n} speaks",
                        f"nly {_n} speaks, muffled behind the "
                        f"{_gag_on[_n].split()[-1]}")
                if _mouth and _voicers:
                    vocal_shots.append(len(plan) + 1)
                # With no sheet the node knows nobody by name, and "more than one
                # described" can never be true -- so a two-person dialogue with no
                # character_memory had no voice guard at all. Said to one person it
                # holds nobody else; said to two it is the whole point.
                # ...and the people still in frame from the last shot count too: a beat
                # naming only the listener ("Dan holds her") is two people.
                if (not _mouth and _speaks and not _talkers and not _voicers
                        and (len(set(_described or []) | set(_here_too)) > 1
                             or (not _described and _has_people))):
                    _mouth = ONE_VOICE_BUSY if _mouth_busy else ONE_VOICE
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
                     if (_speaks and (not _voiced or (_talkers_pre and not _voicers)))
                     else "")
            if _lang and _shot_lang not in _langs_used:
                _langs_used.append(_shot_lang)
            if _lang:
                language_shots.append(len(plan) + 1)
            _said_words = len(engine.spoken_text(body).split())
            if _said_words:
                _spoken_words[len(plan) + 1] = _said_words
            # People in the room by the scene or the carried cast count, not only by
            # the beat: "The smart speaker says ..." in a living room the paragraph
            # fills with two people had no line saying the voice is the machine's.
            # REPORTED.
            _room_people = bool(_has_people or (_described or []) or any(
                re.search(r"(?<![\w'’-])" + re.escape(n) + r"(?![\w'’-])", _scene_text or "")
                for n, _ in sheet_lines(sheet) if n))
            _device = device_voice_clause(body) if (_device_line and _room_people) else ""
            if _device:
                device_shots.append(len(plan) + 1)
            # A SPEAKING SHOT GETS THE ROOM AND NOTHING ELSE. An inferred door, chain or
            # footstep beside a line is an action the beat never wrote, competing with
            # the voice. REPORTED as characters doing things the beat never wrote. The
            # bed and the room tone are added below as before.
            heard = ([] if (not auto_sound or _own or _speaks)
                     else sounds_for(_acted, held=[_state_key(t) for t, _ in _pairs]))
            # The beat's own vocal goes into the list whether or not the beat also reads
            # as a written sound. Only when it did was it put back, so "cries out" and
            # "grunts" were heard as the room alone: an exclusive list, true of nothing,
            # on a branch the vocal had just opened. See named_vocals_in.
            _vocals_here = named_vocals_in(body)
            if _own or _vocals_here:
                heard = [v for v in _vocals_here if v not in heard] + heard
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
            _own_unsaid = bool(_own) and not _vocals_here
            _sound = sound_clause(wordless(heard), only=not _speaks, written=_own_unsaid)
            if _muffled:
                # Heard through the gag. "Wordless screaming" is an open mouth.
                _sound = _sound.replace("wordless ", "muffled, wordless ", 1)
            if hold_gaze and _gazers:
                _g = _gazers[0]
                _target, _is_person = looking_at[_g]
                _elsewhere = " ".join([
                    hold, _posture, _pose, _travel, _where, _told, turn, _duress,
                    _mouth, _revealed, _under, _bare, _wearing, tail, _moved,
                    anchors, _state_clause, _device, _sound, _pace, fall, _gag,
                    _down_lead])

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
            if _leaves_room:
                on_call = False     # a call does not follow a cut into another room
            if _CALL_STARTS.search(_acted or ""):
                on_call = True
            _on_call_now = on_call
            if _CALL_ENDS.search(_acted or ""):
                on_call = False
            _prev_for_gaze, prev_acted = prev_acted, _acted
            # NOT OVER SOMETHING THE BEAT OR THE STATE ALREADY HAS THEM DOING: a written
            # occupation ("Dex works the bag"), a gaze of their own, or a hold -- lying,
            # bound, gagged. "They face each other" on those was the node's staging, not
            # the beat's. REPORTED.
            _occupied = any(
                n in (restrained_who or ()) or n in (_gag_on or {})
                or poses.get(n) == "lying down" or looking_at.get(n)
                or own_action(_acted, n, shot_sheet, _described)
                for n in (_described or []))
            if (hold_gaze and not _gaze and _speaks and not _look_now
                    and not _device_line and not _occupied
                    and faces_each_other(body, _acted, shot_sheet, _described,
                                         poses=poses, facing=facing,
                                         on_call=_on_call_now, scene=shot_scene,
                                         prev=_prev_for_gaze)):
                _faces = dialogue_gaze(len(_described))
                if _faces:
                    dialogue_gaze_shots.append(len(plan) + 1)
            # What came off stays off, where a picture or the beat could put it back.
            _offnow = ""
            _prev_off = set(removed_in.get(len(plan) - 1, ()))
            _now_lines = dict(sheet_lines(shot_scene or ""))
            _orig_lines = dict(sheet_lines(sheet))
            for _n in [n for n in (_described or []) if n in _now_lines]:
                _items = [t for t in gone
                          if _n in gone_by.get(t, ()) and t not in (toks or [])]
                _offnow += off_now_clause(
                    _n, _now_lines[_n], _items, beat=body,
                    after_removal=any(_n in gone_by.get(t, ()) for t in _prev_off),
                    pictured=bool(picture_tags(_orig_lines.get(_n, ""))))
            if _offnow:
                offnow_shots.append(len(plan) + 1)
            # SOMETHING TAKEN OFF EARLIER, NAMED AGAIN. "Dan pulls her panties aside" after
            # they came off names them, and a named garment is a drawn one -- on the body,
            # where the prior puts it. Give it somewhere else to be. Not where this beat
            # puts it back on, and not for a mention inside a line of dialogue.
            _named_here = engine.garment_masked(_acted)
            _off_again = [t for t in gone
                          if t not in toks and t not in restored
                          and not any(names_any(a, [t]) for a in (adds or []))
                          and names_any(_named_here, [t])
                          and set(gone_by.get(t) or ()) & set(_described or [])]
            _revived = "".join(
                f" {(_who[0] + chr(39) + 's ') if len(_who) == 1 else 'The '}"
                f"{scene_name_for(_t, scene) or _t} came off earlier and "
                f"{'stay' if plural_item(_t) else 'stays'} off the body, wherever the "
                f"beat puts {'them' if plural_item(_t) else 'it'}."
                for _t in _off_again[:2]
                for _who in [[n for n in (_described or []) if n in (gone_by.get(_t) or ())]])
            _guards = [
                (1, "removal", tail),        # the beat's own action, completing
                (2, "revived", _revived),    # ...and what came off before, still off
                (1, "wearing", _wearing),    # ...and its mirror, a garment going on
                (2, "offnow", _offnow),      # ...and what came off, still off
                (2, "revealed", _revealed),  # what shows where it was
                (2, "under", _under),         # ...and what is underneath, still on
                (2, "bare", _bare),          # ...or that nothing does
                (3, "hold", hold),           # hardware coming open is not a drift
                (4, "fall", "" if _fall_led else fall),   # a landing; hoisted when bound
                (3, "gag", "" if _gag_led else _gag),     # hoisted, see gag_hold
                (3, "down", _down_lead),   # see lying_stays
                (4, "travel", _travel),      # a journey needs both its ends
                (4, "where", _where),        # ...and later shots need the new room
                (5, "pace", _pace),          # ...and a short action needs the whole shot
                (5, "device", _device),      # a voice that is not hers
                (6, "moved", _moved),        # a garment left where it was put
                (7, "anchors", anchors),     # hardware with nowhere to sit
                (10, "state", _state_clause),
                (9, "posture", _posture),   # where the last beat left the body
                (3, "pose", "" if _pose_led else _pose),   # hoisted ahead of the sheet
                (3, "material", _material),  # what the metal IS, when nobody said
                (11, "gaze", _gaze),
                (12, "duress", _duress),
                (12, "mouth", _mouth),
                (12, "language", _lang),   # ...and in which language
                (13, "contact", _contact),   # who with whom; the beat says what
                (15, "faces", _faces),
                (8, "frame", _frame),        # a fresh composition only -- see frame_hold
                (13, "camera", _camera),
                (6, "told", "" if _told_led else _told),   # hoisted, see above
                (13, "turn", turn),
                (14, "sound", _sound),
            ]
            _floor = RESTRAINT_FLOOR_WORDS if (hold or _pose or anchors) else None
            if fall:
                _floor = (_floor or GUARD_FLOOR_WORDS) + FALL_FLOOR_WORDS
            if hold and _place_words:
                _floor = (_floor or GUARD_FLOOR_WORDS) + _place_words   # see _placed_hold
            _kept, _dropped = fit_guards(_guards, len(body.split()), floor=_floor)
            if _dropped:
                crowded.append((len(plan) + 1, _dropped))
            for _gone in _dropped:
                _tracker = {
                    "wearing": wearing_shots, "fall": fall_shots, "offnow": offnow_shots,
                    "gag": gag_shots, "down": lying_shots,
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
            # NOT FOR NOBODY: a beat that puts nobody on screen describes nobody -- see
            # _plain_people. A beat somebody walks into or out of keeps its count: a
            # name with no body counted is the random person it was REPORTED as.
            _cast_hold = ("" if _plain_people else
                          cast_hold(list(_described or []) + _also_named, body,
                                    _extras_seen))
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
            _events = (list(sounds_for(_acted, held=[_state_key(t)
                                                   for t, _ in _pairs]))
                       if auto_sound else [])
            # WHO IS HELD, AND SINCE WHICH SHOT, for the pictures at render: a portrait
            # or a frame from before a restraint or gag went on shows the person without
            # it. Read from the state the hold reads, and from the hold itself where it
            # registered a restraint from the beat that the state did not name -- such a
            # shot changes what they wear as much as one the state saw. Reported as tape
            # and cuffs gone in the shot after they went on.
            _n_now = len(plan) + 1
            _keys_now = {_nm: frozenset(_q.hardware) for _nm, _q in _state.people.items()
                         if _q.hardware}
            # ...the hold's own test for a beat adding wearers, so a prop the scene
            # mentions is never a restraint put on anybody.
            _hold_by_beat = bool(restrained and (_hold_by_beat or (
                restraint_present(_acted)
                and (not _hold_was_on or restraint_going_on(_acted)))))
            _hold_was_on = restrained
            _hold_named = (merge_hardware_names(_hold_named + [hardware_named(_acted)])
                           if _hold_by_beat else [])
            if _hold_by_beat:
                for _nm in (restrained_who or ()):
                    _keys_now.setdefault(_nm, frozenset({("held", "")}))
            if _held_keys is None:
                # On before the first beat: what the sheet declares, less what that
                # beat itself stages.
                _held_keys = {}
                for _nm, _k in _hw_before:
                    if _staged_at.get(_k[0] if isinstance(_k, tuple) else _k) != _n_now:
                        _held_keys[_nm] = _held_keys.get(_nm, frozenset()) | {_k}
            held_shots[_n_now] = dict(_held_since)
            _key_moves = {}             # name -> (keys gained, keys lost): pose control
            for _nm in set(_keys_now) | set(_held_keys):
                _was_k = _held_keys.get(_nm, frozenset())
                _now_k = _keys_now.get(_nm, frozenset())
                if _now_k == _was_k:
                    continue
                _key_moves[_nm] = (_now_k - _was_k, _was_k - _now_k)
                hardware_changed.add(_n_now)
                if not _now_k:
                    _held_since.pop(_nm, None)
                elif _now_k - _was_k:
                    _held_since.setdefault(_nm, _n_now)
            _held_keys = _keys_now
            held_items[_n_now] = {
                _nm: ([_r.item for _r in _state.people[_nm].hardware.values()],
                      hardware_where(list(_state.people[_nm].hardware.values())))
                if _nm in _state.people and _state.people[_nm].hardware
                else (list(_hold_named), {})
                for _nm in held_shots[_n_now] if _nm in _keys_now}
            # Nobody described is nobody pictured: a recovered face is never taken from
            # a shot of the clock -- see _plain_people.
            plan.add(shot_text,
                     list(active) if (character_guard and not _plain_people) else [],
                     _speaks, (_own and not _mute_written) or _voiced,
                     _voiced and not _own, _events)
            # What pose control needs to know about this shot -- see pose_candidate.
            _pose_falls = bool(_bound_fall and _wearer_here)
            _pose_fallers = list(_fallers) if _pose_falls else []
            # Fastened TO an object: the point part of limb_anchor's phrase ("..., at the
            # headboard", or "at the pipe" alone). A position by itself, "at the waist"
            # included, is not an anchor.
            _pose_anchor = bool(limb_anchor_parts(_pose_pos)[1])
            # Who does the restraining in this beat is never the one held, and nor is a
            # wearer recorded off a beat whose pronoun cannot be them ("Dan handcuffs
            # her wrists" with two women in the sheet records the cuffs on Dan).
            _going_on = bool(_ch.get("applied") or restraint_going_on(_acted)
                             or any(_nw for _n, _r, _l, _nw in _held_line))
            _doers = {_n for _n in _bound_pose_now
                      if _going_on and restrains_in(_acted, _n)}
            _beat_groups = {g for w, g in (("her", "she"), ("him", "he"), ("his", "he"),
                                           ("them", "they"), ("their", "they"))
                            if re.search(r"\b" + w + r"\b", _acted or "", re.I)}
            for _n in _limbs_on:
                if (restrains_in(_acted, _n) and _pron_of.get(_n) and _beat_groups
                        and _pron_of[_n] not in _beat_groups
                        and not re.search(r"\b(?:herself|himself|themselves|themself|own)\b"
                                          r"|\b" + re.escape(_n) + r"['’]s\b", _acted or "",
                                          re.I)):
                    _pose_doubted.add(_n)
            # Limbs going on by the word, with no piece for the state to record: "Dan
            # hogties Mara" when she is already cuffed. The limbs the beat itself places,
            # on the held people it names (all of them when it names none).
            _words_on = set()
            if restraint_going_on(_acted):
                if limb_anchor_parts(_anchor_now)[0] or _legs_now == "ankles to the wrists":
                    _words_on.add("arms")
                if _legs_now:
                    _words_on.add("legs")
            _took = [_n for _n in _bound_pose_now if _n not in _doers]
            _named = [_n for _n in _took if re.search(r"\b" + re.escape(_n) + r"\b",
                                                      _acted or "")]
            for _n in (_named or _took) if _words_on else ():
                _limbs_on.setdefault(_n, set()).update(_words_on)
            # ...and the held keys: a piece the sheet stages here, or the word-only hold
            # going on or off.
            for _n, (_gained, _lost) in _key_moves.items():
                for _k in _gained:
                    _pt = _k[1] if isinstance(_k, tuple) and len(_k) > 1 else ""
                    if _limb_of(_pt):
                        _limbs_on.setdefault(_n, set()).add(_limb_of(_pt))
                    elif _k == ("held", "") or not _pt:
                        _limbs_on.setdefault(_n, set()).update(_words_on or {"arms", "legs"})
                for _k in _lost:
                    _pt = _k[1] if isinstance(_k, tuple) and len(_k) > 1 else ""
                    if _k == ("held", "") and _keys_now.get(_n):
                        continue            # the word-only hold, now a piece the state names
                    if _limb_of(_pt) or not _pt:
                        _limbs_off = True
            # A limb the shot puts on latches -- unless it was held already, in the same
            # position, from an earlier shot.
            def _latch_of(_n, _a, _l):
                _had = _pose_limbs_had.get(_n, {})
                _before = {_limb_of(_k[1]) for _nm, _k in _hw_before
                           if _nm == _n and isinstance(_k, tuple) and len(_k) > 1}
                _now = {"arms": limb_anchor_parts(_a)[0], "legs": str(_l or "").strip()}
                return tuple(x for x in ("arms", "legs")
                             if x in _limbs_on.get(_n, ()) and _now[x]
                             and (_had.get(x, _now[x]) != _now[x]
                                  or (x not in _had and x not in _before)))
            plan.shots[-1].bound_pose = {
                _n: pose_wearer_facts(
                    _a, _l, anchored=_pose_anchor,
                    leg_items=[_r.item for _r in (_state.people[_n].hardware.values()
                                                  if _n in _state.people else ())
                               if getattr(_r, "part", "") in _LEG_PARTS],
                    fall=_pose_falls and (_n in _pose_fallers or not _pose_fallers),
                    latch=_latch_of(_n, _a, _l))
                for _n, (_a, _l) in _bound_pose_now.items()}
            plan.shots[-1].bound_fall = _pose_falls
            plan.shots[-1].fallers = _pose_fallers
            plan.shots[-1].limbs_on = bool(any(_limbs_on.values()))
            plan.shots[-1].limbs_off = bool(_limbs_off)
            plan.shots[-1].restrainers = sorted(
                _doers | {_n for _n in _bound_pose_now if _n in _pose_doubted})
            # What each held person's limbs were in, for the next shot's latch.
            for _n, _f in plan.shots[-1].bound_pose.items():
                _d = _pose_limbs_had.setdefault(_n, {})
                for _x in ("arms", "legs"):
                    if _f[_x]:
                        _d[_x] = _f[_x]
            for _n, _r in (_ch.get("released") or []):
                _x = _limb_of(getattr(_r, "part", ""))
                if _x and _n in _pose_limbs_had and not any(
                        _limb_of(_q.part) == _x
                        for _q in getattr(_state.people.get(_n), "hardware", {}).values()):
                    _pose_limbs_had[_n].pop(_x, None)
            for _n in list(_pose_limbs_had):
                if arms_freed:
                    _pose_limbs_had[_n].pop("arms", None)
                if legs_freed:
                    _pose_limbs_had[_n].pop("legs", None)
                if _n not in _keys_now or not _pose_limbs_had[_n]:
                    _pose_limbs_had.pop(_n, None)
            _pose_doubted &= set(_keys_now)

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
            # NOT CALLED ON A RENDER. lora_patch_mismatches needs each patched
            # weight's shape, and it got them from model_state_dict() -- which walks
            # every parameter of the model. The build this node was fast on never
            # called it, and under --enable-dynamic-vram the model is being streamed,
            # so walking all of it before sampling is not free in a way that can be
            # shown by reading. Reported as disk reads thrashing and a render running
            # four times slower. The function stays, and is tested, for use off the
            # render path; the node touches the model exactly as that build did.
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

        lens, len_note = plan_lengths(beats, ceiling, shot_length == "from the beat", pace,
                                      applying=applying_shots)
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
        # ...and the hold an applying shot is sized for counts as one -- see plan_lengths.
        if shot_length == "from the beat":
            _clauses += len(applying_shots)
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
        if held_over:
            notes.append(
                "kept in frame by their hardware -- "
                + "; ".join(f"shot {n}: {_join_names(w)}" for n, w in held_over)
                + ". The beat named somebody else and the guard would have dropped "
                  "them, which drops their sheet entry and every restraint on it with "
                  "it. A fastened person does not leave because the text stopped "
                  "mentioning them. They are let go by a beat that takes somebody out "
                  "of the room, by a cut to another room, or by the hardware coming "
                  "off -- write them out and they go")
        if stayed_on:
            notes.append(
                "kept in the shot because they are still in the frame it opens on -- "
                + "; ".join(f"shot {n}: {_join_names(w)}" for n, w in stayed_on[:12])
                + ". Each shot starts from the last one's final frame, so whoever is in "
                  "it is in the picture; describing only the people a beat names told "
                  "the shot there were fewer, and the model took the others out to "
                  "agree. They stay described, whole and counted, until a beat takes "
                  "them out -- an exit, being led out, a cut to another room, 'alone' -- "
                  "or you frame the shot yourself (a close-up, a medium shot)")
        if partners_held:
            notes.append(
                "kept in frame as a partner in a sex scene -- "
                + "; ".join(f"shot {n}: {_join_names(w)}" for n, w in partners_held)
                + ". The beat named only the other person, and the guard would have "
                  "dropped them and counted one body over a first frame holding two. "
                  "They are let go by a beat that takes them out of the room or by a "
                  "cut to another room -- write them out and they go")
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
                f"shot(s) {', '.join(str(n) for n in frame_shots)} are told what the "
                f"frame HOLDS: every person described in them whole, head to feet. Only "
                f"where the frame is composed afresh -- shot 1, a cut, a restart, a "
                f"journey: a shot that opens on the last one's final frame already has its "
                f"framing, the camera hold keeps it, and widening a held take is how it "
                f"cuts to a new angle in a new place. Nothing is said on a shot whose beat "
                f"puts nobody on screen, or where you move the camera yourself. An "
                f"attribute a prompt does not state is left to "
                f"the model's PRIOR, and the prior for a named, described person is a "
                f"portrait: cropped to the face, with the clothes and restraints below it "
                f"out of the picture and redrawn from nothing when they come back into "
                f"view. Reported as clothing and bondage equipment not looking the same, or "
                f"disappearing, when a character leaves the shot and comes back. Your "
                f"camera always wins: write any framing in the beat or the anchor -- a "
                f"close-up, a medium shot, waist-up -- and this stands down for it")
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
                f"took something off -- that is keep_frame_after_removal turned off, and it is what "
                f"stops a garment being inherited back through the keyframe. It costs "
                f"a visible cut at each of those points. Turn keep_frame_after_removal back "
                f"on to keep the chain unbroken and accept the risk")
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
        _held_said = held_report(held_rows)
        if _held_said:
            notes.append(
                "restraints registered per shot -- what the node holds on each person "
                "once that shot's beat is read, from the beats, the sheet and any hold: "
                "lines: " + "; ".join(_held_said)
                + ". A piece missing here is not held by any shot's text; a hold: line "
                  "under the beat that puts it on registers it")
        for _n, _words in unheld_words:
            _named = ", ".join(f"'{w}'" for w in _words)
            notes.append(f"shot {_n} names {_named} but nothing was registered -- add a "
                         f"hold: line if it should stay on")
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
        # No mood line any more -- see strain_face. The note says where the tone goes.
        if _film_mood == "grim" or _film_duress:
            notes.append(
                "this film reads as one of duress, and the node writes no mood line for "
                "it: a film's tone is the author's, set in the anchor, which goes into "
                "every shot as written. A face is told the strain only on a shot where "
                "a described person is held -- in hardware, or bound by the beat -- and "
                "named, so the captor alone in a frame is left to the beat")
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
                f"face is doing, because a described person in them is held -- in "
                f"hardware, or bound by the beat -- or the beat names a feeling and "
                f"whose it is. Reported as somebody smiling at the "
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
                f"describe a sound IN THE BEAT or give somebody a VOICE -- a moan, a "
                f"gasp, a laugh, a scream -- so their audio is left free to make it: "
                f"writing the sound is asking for audio on purpose. Effort alone -- "
                f"struggling, trembling, gripping, thrusting -- no longer opens it, "
                f"because a free branch with nothing to say was inventing words on "
                f"beats that were only actions. "
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
        _portrait_slots = {}        # 0-based shot -> {name: their pictures' places in it}
        _slot_owner = {t: n for n, ln in sheet_lines(sheet) if n for t in picture_tags(ln)}
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
            if _fast and not _tagged:
                # FastH3: a picture YOU tagged is your call and rides as tagged (below).
                # An untagged one is not claimed onto anybody for you -- that is the
                # node guessing, on a model that never learned to read the guess.
                plan.shots[_i].refs = []
                continue
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
                    _portrait_slots[_i] = {_here[0]: [1]}
                    _claimed_untagged.append(_i + 1)
                else:
                    plan.shots[_i].refs = []
                    _held_untagged.append(_i + 1)
                continue
            # Whose portrait each of this shot's pictures is, by its place in the shot.
            _live = [t for t in picture_tags(_s) if 1 <= t <= len(refs_all)]
            for _k, _t in enumerate(_live, 1):
                if _t in _slot_owner:
                    _portrait_slots.setdefault(_i, {}).setdefault(
                        _slot_owner[_t], []).append(_k)
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
                f"disturbs no numbering, and is NAMED there as the frame the shot opens "
                f"on -- the encoder is shown it as a picture too, and an unnamed picture "
                f"beside a reference is a second copy of the person it shows. Expect the NUMBER in script to differ from the "
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
        if _hyper and sigmas is not None and len(sigmas) > 1:
            # On Hyperflow's grid, not on a scheduler it never runs.
            _last_a = audio_sigma_of(float(sigmas[-2]), shift_video, shift_audio)
        # A SCHEDULER CAN END THIS OUTRIGHT, and this note used to deny it.
        _alt_sched = scheduler_that_finishes_audio(steps, shift_audio, shift_video,
                                                   scheduler)
        _fix_a = min(shift_audio_for(steps), float(shift_audio or 0.0) or 1.0)
        _soft_landing = bool(apply_model_sampling and not _fast
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
        if _pose_ok:
            # Which shots pose control will look at, and why the others are left alone.
            _pose_plan = []
            for _k, _shot in enumerate(plan.shots):
                _c = [n for n in _shot.cast if n]
                _bound, _why = pose_candidate(_shot, list(shot_frames.get(_k, (_c, _c))[0]),
                                              pose_shots)
                if _bound:
                    _pose_plan.append(
                        f"shot {_k + 1}: "
                        + ("checked after its first render"
                           if _POSE_MODE_KEY.get(pose_shots) == "repair"
                           else "rendered a second time")
                        + (", latched as the restraint goes on" if pose_latched(_bound)
                           else "")
                        + f" ({pose_describe(_bound)})")
                elif _why:
                    _pose_plan.append(f"shot {_k + 1}: pose skipped -- {_why}")
            notes.append("pose control plan -- " + ("; ".join(_pose_plan) if _pose_plan
                         else "no shot has a restrained person in a held position"))
            if any(p.endswith(POSE_LATCH_NEEDS_UPDATE) for p in _pose_plan):
                notes.append("pose control: " + POSE_LATCH_NEEDS_UPDATE)
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
            tiled_decode=tiled_decode, upscale_batch=upscale_batch,
            speech_lead_seconds=speech_lead_seconds, speech_tail_seconds=speech_tail_seconds, hold_levels=hold_levels, staging_shots=staging_shots, steps=steps,
            stripped_shots=stripped_shots, cut_shots=cut_shots,
            shot_rooms=shot_rooms, hardware_changed=hardware_changed, shot_frames=shot_frames,
            held_shots=held_shots, held_items=held_items, portrait_slots=_portrait_slots,
            reentry_shots=reentry_shots, own_grade_shots=own_grade_shots,
            refs_ok=not _fast, outdoor_shots=outdoor_shots,
            fast_h3=bool(_fast), hyperflow=_hyper,
            pose_controlnet=pose_controlnet if _pose_ok else None, pose_ok=bool(_pose_ok),
            pose_note=_pose_note, pose_strength=float(pose_strength),
            pose_end=float(pose_end), pose_shots=pose_shots, pose_draw=pose_draw,
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
        own_grade = prepared.own_grade_shots or set()
        # Every picture this loop adds on its own -- the frame carried across a cut,
        # a recovered or evened face, a returning room -- is a reference row. Off where
        # the model reads none: see fast_h3.
        refs_ok = prepared.refs_ok is not False
        outdoor_shots = prepared.outdoor_shots or set()
        hardware_changed = prepared.hardware_changed or set()
        held_shots = prepared.held_shots or {}
        held_items = prepared.held_items or {}
        portrait_slots = prepared.portrait_slots or {}
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
        # FastH3 reads a keyframe and not a reference row, so the handoff is never
        # demoted into one: below the safe aug it would be. See fast_h3.
        if (not refs_ok and ref_noise_aug is not None
                and float(ref_noise_aug) < KEYFRAME_SAFE_AUG):
            ref_noise_aug = KEYFRAME_SAFE_AUG
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
        tiled_decode = prepared.tiled_decode
        trim_seam = prepared.trim_seam
        upscale = prepared.upscale
        upscale_batch = prepared.upscale_batch
        upscale_model = prepared.upscale_model
        upscale_target_short_edge = prepared.upscale_target_short_edge
        vae = prepared.vae
        w = prepared.w
        # POSE CONTROL, per shot, on a clone made after the model prep below. See
        # pose_sample_shot.
        pose_cn = getattr(prepared, "pose_controlnet", None)
        pose_on = bool(getattr(prepared, "pose_ok", False) and pose_cn is not None
                       and pose_control is not None)
        pose_mode = getattr(prepared, "pose_shots", POSE_SHOT_MODES[0])
        pose_draw = getattr(prepared, "pose_draw", POSE_DRAWS[0])
        pose_strength = float(getattr(prepared, "pose_strength", 1.0))
        pose_end = float(getattr(prepared, "pose_end", 0.6))
        _pose_lines = []            # one per shot pose control looked at
        _pose_passes = 0            # second passes that ran
        t_pose = 0.0                # pose time beyond pass 1, inside t_sample
        _pose_boxes = {}            # name -> torso box at the last shot's last analysed frame
        _pose_looks = {}            # name -> appearance vector there
        self._release_pose_detector()

        def _frame_cast(k, last=False):
            _c = [n for n in plan.shots[k].cast if n]
            return list(_shot_frames.get(k, (_c, _c))[1 if last else 0])

        if apply_model_sampling:
            model, ms_note = apply_h3_model_sampling(model, shift_video, shift_audio)
            notes.append(ms_note)
        # After the schedule patch, both of these: VSA reads its start point off the
        # model's own sigmas, and the endpoint wrapper reads the stamp.
        if getattr(prepared, "fast_h3", False):
            model, _vsa_note = apply_fast_h3_vsa(model)
            if _vsa_note:
                notes.append(_vsa_note)
        _hf = getattr(prepared, "hyperflow", None)
        if _hf and _hf.get("two_time"):
            try:
                model = install_hyperflow_two_time(model, _hf)
            except Exception as e:
                notes.append(f"Hyperflow's endpoint conditioning could not be installed "
                             f"({type(e).__name__}: {e}), so it runs one-time")
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
        _held_framed_at = []        # (shot, name) whose portrait gave way to the opening frame
        _held_swapped = []          # (shot, name, source shot) given a frame of them held
        _held_noted = []            # (shot, name) kept their portrait with a note of the hold
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
            # A portrait of somebody whose restraint or gag went on in an earlier shot
            # shows them before it. The last frame shows them as they are now, so where
            # it pictures them it is their picture, a cut included, and the portrait
            # gives way. Reported as tape and cuffs gone in the shot after.
            _slots = portrait_slots.get(i) or {}
            _held_in = {n for n in (held_shots.get(i + 1) or {}) if _slots.get(n)}
            _held_framed = {n for n in _held_in if n in _prev_people}
            _carry_ok = bool(refs_ok and i and handoff is not None and i not in reentry_shots
                             and (_cond_module.may_carry_room if i in cut_shots
                                  else _cond_module.may_carry_frame)(
                                 _prev_people, plan.shots[i].cast,
                                 set(_tagged_names or ()) - _held_framed))
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
            # A person introduced in position no longer breaks the chain: the shot
            # opens on the last frame and the text walks them into it (see
            # entrance_clause). Restarting cost the camera angle and every state the
            # last frame carried -- REPORTED as angles changing between beats and beats
            # losing what was in the one before.

            # ...and the portraits of the held. Opening on a frame of them, that frame is
            # their picture. Otherwise a frame of them alone from after the change takes
            # the portrait's place, and failing that the portrait stays with the hold
            # named beside it.
            _drop_slots, _note_for, _gave_way = [], [], set()
            _refs_now = list(shot.refs)
            for _n in sorted(_held_in):
                _ks = list(_slots.get(_n) or [])
                _since = held_shots[i + 1][_n]
                if shot_handoff is not None and _n in _held_framed:
                    _drop_slots += _ks
                    _gave_way.add(_n)
                    _held_framed_at.append((i + 1, _n))
                elif (refs_ok and _captured.get(_n) is not None
                        and _captured_gen.get(_n) == _wardrobe_gen
                        and _captured_from.get(_n, 0) >= _since):
                    _refs_now[_ks[0] - 1] = _captured[_n]
                    _drop_slots += _ks[1:]
                    _held_swapped.append((i + 1, _n, _captured_from.get(_n, 0)))
                elif _n in (held_items.get(i + 1) or {}):
                    _note_for.append((_n, _ks[0]))
            if _drop_slots:
                shot_prompt, _refs_now = drop_portraits(shot_prompt, _refs_now, _drop_slots)
            for _n, _k in _note_for:
                _k -= sum(1 for d in set(_drop_slots) if d < _k)
                _items, _where = held_items[i + 1][_n]
                _said = held_picture_note(_k, _n, _items, _where)
                _at = re.search(r"<Picture " + str(_k) + r">[^.]*\.", shot_prompt)
                shot_prompt = (shot_prompt[:_at.end()] + _said + shot_prompt[_at.end():]
                               if _at else shot_prompt + _said)
                _held_noted.append((i + 1, _n))
            shot.refs = _refs_now

            _extra = []
            _evened_who = ""            # who the evening-up frame below pictures
            _cast = plan.shots[i].cast
            _returning = {w for n, ws in _returns if n == i + 1 for w in ws}
            _who = _cond_module.recoverable_subject(
                _cast, _tagged_names, _returning,
                {k: v for k, v in _captured.items()
                 if _captured_gen.get(k) == _wardrobe_gen})
            if not refs_ok:
                _who = ""
            # Not for anybody the shot OPENS on: whether the last frame rides as the
            # keyframe, as a carried reference or demoted, it is already a picture of
            # them, and a second picture of one person is how a second one is drawn.
            if _who and shot_handoff is not None and _who in _prev_people:
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
            elif (refs_ok and _tagged_names and len(_cast) > 1
                  and any(n in _tagged_names and n not in _gave_way for n in _cast)):
                _short = [n for n in _cast
                          if n and n not in _tagged_names
                          and _captured.get(n) is not None
                          and _captured_gen.get(n) == _wardrobe_gen]
                # Not for anybody the shot opens on either, the recovered face's rule:
                # a plain keyframe of them is a picture of them too, and the same frame
                # went out twice -- once as the keyframe, once as their face.
                if (len(_short) == 1 and f"{_short[0]}:" in shot_prompt
                        and not (shot_handoff is not None
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
            if not refs_ok:
                pass                    # a returning room is a reference row too
            elif (i in cut_shots or i in reentry_shots) and _opens in _room_frames:
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
                            len(shot.refs) + len(_extra), _back, _in_it, _arriving,
                            outdoor=i in outdoor_shots)
                        _room_returns.append((i + 1, _back, _from))
                        break
            _shot_refs = list(shot.refs) + _extra
            if _handoff_ref and _plate_on == i + 1:
                # A SET, not a room a moment earlier. See plate_claim.
                shot_prompt = shot_prompt + plate_claim(len(_shot_refs) + 1,
                                                        outdoor=i in outdoor_shots)
                _handoff_claimed.append(i + 1)
            elif _carry_rooms is not None:
                _was_room, _now_room = _carry_rooms
                if _was_room and _now_room and _was_room != _now_room:
                    shot_prompt = shot_prompt + carried_people_claim(
                        len(_shot_refs) + 1, _prev_people, _was_room, _now_room)
                else:
                    shot_prompt = shot_prompt + room_claim(len(_shot_refs) + 1,
                                                           _prev_people, [],
                                                           outdoor=i in outdoor_shots)
                shot_prompt = recount_with_claim(shot_prompt, plan.shots[i].cast,
                                                 _prev_people)
                _handoff_claimed.append(i + 1)
            elif _handoff_ref:
                _was, _join = next(((w, j) for s, w, j in _carried if s == i + 1),
                                   ([], []))
                shot_prompt = shot_prompt + room_claim(len(_shot_refs) + 1, _was, _join,
                                                       outdoor=i in outdoor_shots)
                shot_prompt = recount_with_claim(shot_prompt, plan.shots[i].cast, _was)
                _handoff_claimed.append(i + 1)
            elif handoff_rides_as_ref(shot_handoff, _shot_refs, ref_noise_aug):
                shot_prompt = shot_prompt + handoff_claim(len(_shot_refs) + 1)
                _handoff_claimed.append(i + 1)
                _aug_claimed.append(i + 1)
            elif shot_handoff is not None and any(r is not None for r in _shot_refs):
                # Still a keyframe, but the encoder is shown it as the picture after
                # the references, and an unnamed one is a second copy of whoever the
                # references name. See keyframe_claim.
                shot_prompt = shot_prompt + keyframe_claim(
                    sum(1 for r in _shot_refs if r is not None) + 1, first=(i == 0))
                _handoff_claimed.append(i + 1)
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
            _pose_bound, _pose_why = (pose_candidate(shot, _frame_cast(i), pose_mode)
                                      if pose_on else ({}, ""))
            _pose_rep, _pose_reuse = None, None
            # The identity carried across the cut is good only where the shot opens on the
            # very frame it was read from -- so the handoff frame is read here, once that
            # is known, and not on a cut, a re-entry, a restart or a demoted keyframe.
            _pose_keyed = bool(shot_handoff is not None and not _handoff_ref and not demoted)
            _evict_all_but(model, latent)
            try:
                _t0 = time.perf_counter()
                if _pose_bound:
                    if getattr(self, "_pose_detector", None) is None:
                        self._pose_detector = pose_control.PoseDetector(_pose_device())
                    _pose_carry = (pose_handoff_boxes(self._pose_detector, shot_handoff,
                                                      _pose_boxes, w, h,
                                                      appearance=_pose_looks)
                                   if (_pose_keyed and _pose_boxes) else {})
                    out, _pose_reuse, _pose_rep = pose_sample_shot(
                        model, cond, negative, latent, seed, steps, cfg, sampler_name,
                        scheduler, sigmas, shift_video, shift_audio, _soft_landing,
                        pose_cn=pose_cn, vae=vae, detector=self._pose_detector,
                        bound=_pose_bound, mode=pose_mode, draw=pose_draw,
                        strength=pose_strength, pose_end=pose_end, frame_count=fc,
                        w=w, h=h, tiled=tiled_decode,
                        carry=(dict(_pose_carry) if _pose_carry else None),
                        cast_count=len(_frame_cast(i)) or None,
                        keep_decoded=not (latent_upscale and latent_upscale != "off"),
                        audio_vae=audio_vae,
                        latch_after=(int(math.ceil(POSE_LATCH_FROM * int(fc)))
                                     if pose_latched(_pose_bound) else None),
                        carry_appearance=({n: v for n, v in _pose_looks.items()
                                           if n in _pose_carry} if _pose_carry else None))
                    t_pose += max(0.0, time.perf_counter() - _t0
                                  - float(_pose_rep.get("t_pass1") or 0.0))
                else:
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
            if _pose_rep is not None:
                if _pose_rep.get("outcome") in ("repaired", "held"):
                    _pose_passes += 1
                if _pose_rep.get("off"):
                    pose_on = False
                _pose_lines.append(pose_shot_line(i + 1, _pose_rep, _pose_bound))
            elif _pose_why:
                _pose_lines.append(f"shot {i + 1}: pose skipped -- {_pose_why}")

            try:
                parts = out["samples"].unbind() if hasattr(out["samples"], "unbind") else None
            except Exception:
                parts = None

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
            # RAM for this shot's frames BEFORE they are decoded. Every shot's frames
            # land in the headroom the weights left, and nothing between nodes runs to
            # give it back mid-chain -- see ensure_host_ram. The DiT and the text
            # encoder are done with until the next shot; the VAEs are next.
            if _pose_reuse is not None and pre_up is None:
                imgs = _pose_reuse      # pass 1 kept, and its frames were decoded already
            else:
                ensure_host_ram(_decode_ram(vae, out, shot_tiled), keep=(vae, audio_vae),
                                what=f"shot {i + 1}'s decode")
                imgs = _decode_video(vae, out, shot_tiled, free_first=model,
                                     keep=(vae, audio_vae))
            _pose_reuse = None
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
            _keyed = bool(shot_handoff is not None and not demoted)
            try:
                if (_keyed and imgs is not None
                        and imgs.shape[0] > 1 and hand_src is not None and hand_src.shape[0]):
                    _levels.observe(shot_handoff, imgs[0], imgs[-1], hand_src[-1])
            except Exception:
                pass
            # THE SHOT'S OWN COOKING COMES OUT OF ITS FRAMES, on a ramp from frame one to
            # the last, and the last frame IS the handoff. The correction before this
            # graded the handoff alone, by the median drift across cuts, and let what a
            # shot did over its length through almost untouched -- which is where the
            # burn builds. See shot_grade. The captured face is taken from these frames
            # after grading, so it needs no grade of its own.
            _grade = None
            try:
                if hold_levels > 0 and imgs is not None and imgs.shape[0] > 1:
                    _pre = (hand_src is not imgs and hand_src is not None
                            and hand_src.shape[0] > 0)
                    _sg = shot_grade(shot_handoff if _keyed else None, imgs[0], imgs[-1],
                                     hold_levels, own_change=(i in own_grade),
                                     pipe=(imgs[-1], hand_src[-1]) if _pre else None)
                    if _sg is not None:
                        grade_frames(imgs, *_sg)
                        _eg, _eo = torch.exp(_sg[1][0]), _sg[1][1]
                        if _pre:
                            hand_src = apply_levels(hand_src[-1:], _eg, _eo)
                        _levels.note(_eg, _eo)   # recorded for the end-of-run report
            except Exception:
                pass
            handoff = hand_src[-1:].detach().clamp(0.0, 1.0).to("cpu", copy=True)
            # Who is who at this shot's end, for the next shot to match in its opening
            # frame (read there, when it is a candidate that opens on this frame).
            _pose_boxes = (dict(_pose_rep["boxes"]) if (pose_on and _pose_rep is not None
                                                         and _pose_rep.get("boxes")) else {})
            _pose_looks = (dict(_pose_rep.get("appearance") or {}) if _pose_boxes else {})
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
            # ...AND FROM THE LAST FRAME, whenever the shot ENDS with one person alone in
            # it. A whole solo shot is rare in a scene with company, so most people never
            # had a frame to come back with -- but "Dan walks out" leaves her alone by the
            # last frame, and that frame is exactly how she looks now: her clothes, her
            # restraints. On a shot that put hardware on, it shows the hardware ON, so it
            # belongs to the wardrobe that starts after this shot. REPORTED as clothing and
            # bondage equipment not looking the same when a character comes back.
            try:
                _alone = [n for n in _frame_cast(i, last=True) if n]
                # Only somebody the shot's TEXT describes: where the camera went with
                # somebody leaving she is not described, and whatever the last frame
                # holds is not a picture of her to claim as one.
                if (len(_alone) == 1 and _alone[0] in set(plan.shots[i].cast or [])
                        and _wardrobe_normal and hand_src is not None
                        and hand_src.shape[0]):
                    _captured[_alone[0]] = hand_src[-1:].detach().clamp(0.0, 1.0).to(
                        "cpu", copy=True)
                    _captured_from[_alone[0]] = i + 1
                    _captured_gen[_alone[0]] = _wardrobe_gen + (1 if _n in hardware_changed
                                                                else 0)
            except Exception:
                pass
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

        self._release_pose_detector()
        if _pose_lines:
            notes.extend(_pose_lines)
        if getattr(prepared, "pose_ok", False):
            notes.append(f"pose control: {_pose_passes} extra pass"
                         f"{'' if _pose_passes == 1 else 'es'}, pose time {t_pose:.0f}s "
                         f"(inside the sampling time below)")
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
                                             upscale_target_short_edge, upscale_batch)
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
        if _held_framed_at or _held_swapped or _held_noted:
            _said = []
            if _held_framed_at:
                _said.append("left out where the shot opens on a frame that shows them: "
                             + ", ".join(f"{who} on shot {n}" for n, who in _held_framed_at))
            if _held_swapped:
                _said.append("replaced by a frame of them alone from after it went on: "
                             + ", ".join(f"{who} on shot {n}, from shot {src}"
                                         for n, who, src in _held_swapped))
            if _held_noted:
                _said.append("kept with a sentence naming what is on them now, no frame "
                             "of them since being available: "
                             + ", ".join(f"{who} on shot {n}" for n, who in _held_noted))
            notes.append(
                "portraits of people whose restraint or gag went on in an earlier "
                "shot -- " + "; ".join(_said) + ". The portrait shows them before it "
                "went on, and a near-clean reference asks the model to draw what it "
                "shows, so it put the free hands and the bare mouth back. Their face on "
                "those shots comes from the frames this run rendered of them")
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
                f"with nothing carried. Turn keep_frame_after_removal on to keep the "
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
