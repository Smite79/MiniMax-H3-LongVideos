# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import copy
import glob
import importlib.util as _ilu
import inspect
import json
import math
import os
import re
import sys
import time

import torch
import nodes
import comfy.samplers
import comfy.nested_tensor
import comfy.model_management as mm


def _load_local(name, filename):
    spec = _ilu.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))
    module = _ilu.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = _load_local("h3_runtime", "runtime.py")
au = _load_local("h3_audio", "audio.py")
cnd = _load_local("h3_conditioning", "conditioning.py")
up = _load_local("h3_upscale", "upscale.py")
snd = _load_local("h3_sound", "sound.py")
rst = _load_local("h3_restraints", "restraints.py")
wrd = _load_local("h3_wardrobe", "wardrobe.py")
try:
    pose = _load_local("h3_pose_control", "pose_control.py")
    POSE_IMPORT_ERROR = ""
except Exception as _e:
    pose = None
    POSE_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

NATIVE_RES = {"16:9": (1344, 768), "9:16": (768, 1344), "4:3": (1024, 768), "3:4": (768, 1024),
              "1:1": (768, 768), "21:9": (1536, 672), "9:21": (672, 1536)}
RES_MULTIPLE = 32
SPEECH_LEAD = 0.5
LEVELS = 0.8
POSE_LATCH_FROM = 0.4
HANDOFF_LATENT_TAIL = 8
SHOT_LENGTHS = ["from the beat", "fixed"]
BEAT_BASE_SEC = 0.8
SECONDS_PER_ACTION = 2.2
WORDS_PER_SEC = 2.5
MIN_AUTO_FRAMES = 73

FAST_H3_SHIFT_VIDEO = 10.0
FAST_H3_VSA_KEEP = 0.10
FAST_H3_VSA_START = 0.20
HYPERFLOW_SIGMAS = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0)
HYPERFLOW_SHIFT_VIDEO = 12.0
HYPERFLOW_SHIFT_AUDIO = 3.0
HYPERFLOW_SAMPLER = "euler"
HYPERFLOW_ENDPOINT_KEY = "transformer.endpoint_time_embedder.linear_1.lora_A.weight"
HYPERFLOW_WRAPPER_KEY = "h3_longvideos_hyperflow_two_time"
HYPERFLOW_TE_KEY = "diffusion_model.time_embedder.proj_in.weight"
_HYPERFLOW_DELTAS = {}

_PICTURE = re.compile(r"(\(\s*)?<\s*picture[\s_\-]*(\d+)\s*>(\s*\))?", re.I)
_SPEECH = re.compile(r"\"[^\"]*\"|“[^”]*”|<\s*d\s*>.*?<\s*/\s*d\s*>", re.S | re.I)
_DIRECTIVE = re.compile(r"^\s*(hold|release|remove|wear|seconds|exit)\s*:\s*(.*?)\s*$", re.I)
_CUT = re.compile(r"^\s*cut\s*:?\s*$", re.I)
_INLINE = re.compile(r"(?<=[.!?\"”])\s+(?=(?:hold|release|remove|wear|seconds|exit)\s*:)", re.I)
_WHO = re.compile(r"\s*([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)?)\s*(?:[,:–—-]\s*|\s+(?=[a-z])|$)(.*)$", re.S)
_SHEET = re.compile(r"^\s*([A-Z][\w'’-]{0,24}(?:\s+[A-Z][\w'’-]{0,24}){0,2})\s*:\s*\S")
_CLAIM = re.compile(r"\b([A-Z][\w'’-]{0,24})[\s,:(\[]*(?i:<\s*picture[\s_\-]*\d+\s*>)")
_EXIT = re.compile(
    r"\b(?:leaves?|left|leaving)(?=\s*(?:[.,;:!?]|$)|\s+(?:again|together|without|through|by|via|for|with|and|"
    r"then|now|quietly|alone)\b|\s+(?:the|this|that|his|her|their|our)\s+(?:[\w-]+\s+)?(?:room|cell|house|building|"
    r"apartment|flat|office|kitchen|bedroom|bathroom|basement|hall|hallway|corridor|car|van|scene|shot|frame|home)\b)"
    r"|\bexit(?:s|ed|ing)?\b"
    r"|\b(?:walk|goes|going|gone|went|heads|headed|step|run|ran|storm|hurr|slip|wander|strid|strode|march|rush|back|"
    r"driv|drove|sneak|snuck|dash|bolt|stomp|limp)\w*\s+(?:\w+ly\s+)?(?:out\b(?!\s+of\b)|out\s+of\s+"
    r"(?:the\s+|this\s+|his\s+|her\s+|their\s+)?(?:room|cell|house|building|door|doorway|flat|apartment|office|"
    r"kitchen|bedroom|bathroom|car|van|frame|shot|view|sight)\b|off\b(?!\s+(?:the|a|an|his|her|their)\b)"
    r"|away\b(?!\s+from\b)|outside\b|home\b)"
    r"|\b(?:disappear|vanish)(?:s|es|ed|ing)?\b"
    r"|\bout\s+of\s+(?:(?:the|this|that|his|her|their)\s+)?(?:frame|shot|view|sight)\b", re.I)
_NOT_NAMES = {"She", "He", "They", "It", "Her", "His", "Him", "Their", "The", "A", "An", "This", "That", "With",
              "Like", "As", "And", "Picture", "Setting", "Location", "Place", "Scene", "Style", "Look", "Camera",
              "Lighting", "Light", "Mood", "Time", "Note", "Notes", "Shot", "Audio", "Sound", "Music"}
_BOUNDARY = re.compile(r"[,;:]|\b(?:as|while|when|after|before|until|but|then|so)\b", re.I)
_EXTRAS = re.compile(r"\b(?:crowd|people|someone|somebody|others|strangers?|bystanders?|onlookers?|"
                     r"(?:a|an|another|the|two|three|several|some)\s+(?:\w+\s+)?(?:man|woman|men|women|guards?|nurses?|"
                     r"doctors?|officers?|police|person|girl|boy|child|children|kids?|figure|soldiers?)"
                     r"|(?:her|his|their)\s+(?:boyfriend|girlfriend|husband|wife|partner|friend|mother|father|sister|"
                     r"brother|captor|boss))\b", re.I)
_CAPS = re.compile(r"\b([A-Z][a-z]+)(?:['’]s)?\b")
_COMMON_CAPS = {"She", "He", "They", "It", "Her", "His", "Him", "Their", "Them", "We", "You", "The", "A", "An", "This",
                "That", "These", "Those", "There", "Then", "Now", "Here", "When", "While", "As", "After", "Before",
                "Suddenly", "Slowly", "Finally", "Meanwhile", "Later", "Inside", "Outside", "Behind", "Above", "Below",
                "Under", "Over", "Into", "With", "Without", "In", "On", "At", "To", "From", "For", "Of", "By", "Up",
                "Down", "Off", "Out", "And", "But", "Or", "So", "Still", "Just", "Only", "Again", "Back", "Away",
                "Nobody", "Nothing", "Every", "All", "Both", "Each", "No", "Not", "Picture", "Setting", "Location",
                "Scene", "Style", "Camera", "Shot", "Close", "Wide", "Night", "Day", "Morning", "Evening"}
_CLAUSE = re.compile(r"(?:[.!?;]+|,?\s+(?:and then|then|and|before|after|while|as|until)\s+"
                     r"|,\s+(?=\w+(?:ing|es|s|ed)\b))")

_ARM_POSITIONS = (("behind the back", r"behind\s+(?:her|his|their|the)\s+back"),
                  ("above the head", r"(?:above|over)\s+(?:her|his|their|the)\s+head"),
                  ("at the waist", r"at\s+(?:her|his|their|the)\s+waist"),
                  ("in front of the body", r"in\s+front"))
_ARMS = re.compile(r"\b(?:wrists?|hands?|arms?|elbows?)\b|cuff|behind\s+\w+\s+back", re.I)
_LEGS = re.compile(r"\b(?:ankles?|feet|knees)\b", re.I)
_HOGTIE = re.compile(r"hog-?tie|ankles?\s+(?:\w+\s+)?to\s+(?:her|his|their|the)\s+wrists", re.I)
_ANCHOR = re.compile(r"\bto\s+(?:the|a|an)\s+\w", re.I)
_LINKED = re.compile(r"chain|shackle|iron|manacle|hobble|spreader", re.I)


def frame_size(choice, megapixels):
    w, h = NATIVE_RES.get(choice, NATIVE_RES["16:9"])
    if megapixels and megapixels > 0:
        s = math.sqrt(megapixels * 1024 * 1024 / float(w * h))
        w = max(RES_MULTIPLE, int(round(w * s / RES_MULTIPLE)) * RES_MULTIPLE)
        h = max(RES_MULTIPLE, int(round(h * s / RES_MULTIPLE)) * RES_MULTIPLE)
    return w, h


def align_nearest(n):
    n = max(5, int(n))
    lo = n - ((n - 5) % 17)
    return min(rt.MAX_FRAMES, lo if (n - lo) <= (lo + 17 - n) else lo + 17)


def beat_seconds(text):
    spoken = sum(len(re.sub(r"</?\s*d\s*>|[\"“”]", " ", q).split()) for q in _SPEECH.findall(text))
    clauses = [p for p in _CLAUSE.split(_SPEECH.sub(" ", text)) if p and len(p.split()) >= 2]
    action = BEAT_BASE_SEC + SECONDS_PER_ACTION * len(clauses) if clauses else 0.0
    return max(action, spoken / WORDS_PER_SEC + 1.0 if spoken else 0.0)


def shot_frames(text, ceiling, applying):
    need = beat_seconds(text)
    if applying:
        need = max(need, MIN_AUTO_FRAMES / rt.H3_FPS) + SECONDS_PER_ACTION
    want = align_nearest(round(need * rt.H3_FPS)) if need else MIN_AUTO_FRAMES
    return min(max(MIN_AUTO_FRAMES, want), ceiling)


def parse_paragraph(par):
    out = {"lines": [], "hold": [], "release": [], "remove": [], "wear": [], "exit": [], "seconds": None, "cut": False,
           "unread": []}
    for line in (piece for raw in par.splitlines() for piece in _INLINE.split(raw)):
        m = _DIRECTIVE.match(line)
        if m is None:
            if _CUT.match(line):
                out["cut"] = True
            elif line.strip():
                out["lines"].append(line.strip())
            continue
        kind, arg = m.group(1).lower(), m.group(2)
        if kind == "exit":
            out["exit"] += [w.strip() for w in arg.split(",") if w.strip()]
        elif kind == "seconds":
            try:
                out["seconds"] = float(arg.lower().rstrip("s ").strip())
            except ValueError:
                out["unread"].append(line.strip())
        else:
            who = _WHO.match(arg)
            items = [i.strip() for i in who.group(2).split(";") if i.strip()] if who else []
            if who is None or (kind != "release" and not items):
                out["unread"].append(line.strip())
            else:
                out[kind].append((who.group(1).strip(), items))
    out["text"] = "\n".join(out["lines"])
    return out


def paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]


def parse_script(text, all_beats=False):
    merged, early, cut_next = [], [], False
    for p in map(parse_paragraph, paragraphs(text)):
        if p["text"]:
            for q in early:
                for k in ("hold", "release", "remove", "wear", "exit", "unread"):
                    p[k] = q[k] + p[k]
            p["cut"], early, cut_next = p["cut"] or cut_next, [], False
            merged.append(p)
        elif merged and any(p[k] for k in ("hold", "release", "remove", "wear", "exit", "unread", "seconds")):
            for k in ("hold", "release", "remove", "wear", "exit", "unread"):
                merged[-1][k] += p[k]
            merged[-1]["seconds"] = p["seconds"] or merged[-1]["seconds"]
            cut_next = cut_next or p["cut"]
        else:
            early += [p] if not merged else []
            cut_next = cut_next or p["cut"]
    if not merged:
        return None, []
    scene = merged.pop(0) if len(merged) > 1 and not all_beats else None
    return scene, merged


def _person(state, who):
    return next((k for k in state if k.lower() == who.lower()), None)


_FILLER = {"the", "a", "an", "her", "his", "their", "from", "off", "on", "over", "around", "in", "of"}


def _matches(release, item):
    words = [w for w in re.findall(r"[\w'’-]+", release.lower()) if w not in _FILLER]
    return bool(words) and all(w in item.lower() for w in words)


def apply_holds(state, para):
    new = {k: list(v) for k, v in state.items()}
    released, added = {}, {}
    for who, items in para["release"]:
        key = _person(new, who)
        if key is None:
            continue
        gone = [it for it in new[key] if not items or any(_matches(r, it) for r in items)]
        released.setdefault(key, []).extend(gone)
        new[key] = [it for it in new[key] if it not in gone]
        if not new[key]:
            del new[key]
    for who, items in para["hold"]:
        key = _person(new, who) or _person(added, who) or who
        have = new.setdefault(key, [])
        for it in items:
            if it.lower() not in (x.lower() for x in have):
                have.append(it)
                added.setdefault(key, []).append(it)
    return new, released, added


def held_line(state):
    return " ".join(f"{who}: {'; '.join(items)}." for who, items in state.items() if items)


_STAY_ARMS = {"behind the back": "stay locked behind the back", "in front of the body": "stay bound in front",
              "above the head": "stay above the head", "at the waist": "stay at the waist"}
_STAY_LEGS = {"ankles together": "ankles stay bound together", "ankles to the wrists": "ankles stay tied to the wrists"}


def held_notes(during):
    if not during:
        return []
    out = [held_line(during), "All of it stays on for the whole shot."]
    for who, items in during.items():
        facts = [limb_facts(it) for it in items]
        arms = next((a for a, _, anchored, _ in facts if a and not anchored), "")
        legs = next((g for _, g, anchored, _ in facts if g in _STAY_LEGS and not anchored), "")
        out += [f"{who}'s wrists {_STAY_ARMS[arms]} the whole time."] if arms else []
        out += [f"{who}'s {_STAY_LEGS[legs]} the whole time."] if legs else []
    return out


def limb_facts(item):
    arms = ""
    if _ARMS.search(item):
        arms = next((pos for pos, rx in _ARM_POSITIONS if re.search(rx, item, re.I)), "")
    legs = ""
    if _HOGTIE.search(item):
        legs = "ankles to the wrists"
    elif _LEGS.search(item):
        legs = "held apart" if re.search(r"\b(?:apart|spread)\b", item, re.I) else "ankles together"
    return arms, legs, bool(_ANCHOR.search(item)), bool(_LINKED.search(item))


def pose_plan(during, released, added):
    bound = {}
    for who in list(during) + [k for k in added if k not in during]:
        if any(a or b for a, b, _, _ in map(limb_facts, released.get(who, []))):
            continue
        facts = {"arms": "", "legs": "", "ankle_gap": 0.12, "anchored": False, "fall": False}
        latch = []
        for it, new in [(x, False) for x in during.get(who, [])] + [(x, True) for x in added.get(who, [])]:
            arms, legs, anchored, linked = limb_facts(it)
            if anchored and (arms or legs):
                facts["anchored"] = True
            if arms and not facts["arms"]:
                facts["arms"] = arms
                latch += ["arms"] if new else []
            if legs and not facts["legs"]:
                facts["legs"] = legs
                latch += ["legs"] if new else []
                facts["ankle_gap"] = 0.6 if linked else 0.12
        if facts["anchored"] or not (facts["arms"] or facts["legs"]):
            continue
        facts["latch_limbs"] = tuple(latch)
        bound[who] = facts
    return bound


def _mentions(text, names):
    return [n for n in names if re.search(rf"\b{re.escape(n)}\b", text or "")]


_PEOPLE = re.compile(r"\b(?:she|he|they|her|him|his|their|them|herself|himself|themselves|man|woman|men|women|girl|boy|"
                     r"child|kid|person|people|figure|someone|somebody|anyone|everyone|crowd|guards?|nurse|doctor|officer|"
                     r"cop|maid|waiter|waitress|driver|stranger|intruder|visitor)s?\b", re.I)


def referred(text, names, gender, around=()):
    out = []
    for m in re.finditer(r"\b(she|her|herself|he|him|his|himself)\b", text or "", re.I):
        g = "f" if m.group(1).lower() in ("she", "her", "herself") else "m"
        hits = [n for n in names if gender.get(n) == g] or [n for n in around if gender.get(n, g) == g]
        if len(hits) == 1 and hits[0] not in out:
            out.append(hits[0])
    return out


def nobody(text, names):
    return not (_PEOPLE.search(text or "") or extras_in(text or "")
                or set(_CAPS.findall(text or "")) - set(names) - _COMMON_CAPS)


def roster(sheets, texts, paras):
    names = [m.group(1) for t in sheets for m in map(_SHEET.match, t.splitlines()) if m]
    names += [n for t in texts for n in _CLAIM.findall(t)]
    names += [who for p in paras for who, _ in p["hold"] + p["release"]] + [n for p in paras for n in p["exit"]]
    return [n for n in dict.fromkeys(names) if n not in _NOT_NAMES]


def leavers(text, names):
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        for m in _EXIT.finditer(sentence):
            before = sentence[:m.start()]
            spots = sorted((k.start(), k.end(), n) for n in names for k in re.finditer(rf"\b{re.escape(n)}\b", before))
            if not spots:
                continue
            if re.fullmatch(r"\s+(?:\w+ly\s+)?", before[spots[-1][1]:]):
                who, j = [spots[-1][2]], len(spots) - 1
                while j > 0 and re.fullmatch(r"\s*(?:,|and|,\s*and)\s*", before[spots[j - 1][1]:spots[j][0]], re.I):
                    j -= 1
                    who.insert(0, spots[j][2])
            else:
                clauses = [c for c in _BOUNDARY.split(before) if _mentions(c, names)]
                who = [min(((k.start(), n) for n in names for k in re.finditer(rf"\b{re.escape(n)}\b", clauses[-1])))[1]]
            out += [n for n in who if n not in out]
    return out


def extras_in(text):
    return any(not re.search(r"(?:\b(?:is|was|are|were|as|like|being)|[:,(])\s*$", text[:m.start()], re.I)
               for m in _EXTRAS.finditer(text))


def cast_line(cast, names, *texts):
    joined = " ".join(t for t in texts if t)
    if extras_in(joined) or set(_CAPS.findall(joined)) - set(names) - _COMMON_CAPS:
        return ""
    if len(cast) == 1:
        return "There is one person in the shot: one body, one face."
    if len(cast) == 2:
        return "There are two people in the shot, with one body for each person."
    return ""


def without_absent(text, absent, names):
    here, out = [n for n in names if n not in absent], []
    for line in text.splitlines():
        m = _SHEET.match(line)
        if m and m.group(1) in names:
            out += [line] if m.group(1) not in absent else []
            continue
        out.append(" ".join(s for s in re.split(r"(?<=[.!?])\s+", line)
                            if not (_mentions(s, absent) and not _mentions(s, here))))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def shot_pictures(text, refs, held_names):
    claimed = set()
    for name in held_names:
        rx = rf"\b{re.escape(name)}\b[\s,:(\[]*<\s*picture[\s_\-]*(\d+)\s*>"
        claimed |= {int(m.group(1)) for m in re.finditer(rx, text, re.I)}
    wanted = sorted({int(m.group(2)) for m in _PICTURE.finditer(text)})
    live = [n for n in wanted if n <= len(refs) and refs[n - 1] is not None and n not in claimed]
    renum = {old: new for new, old in enumerate(live, 1)}

    def sub(m):
        n = int(m.group(2))
        if n in renum:
            return f"{m.group(1) or ''}<Picture {renum[n]}>{m.group(3) or ''}"
        return "" if (m.group(1) and m.group(3)) else (m.group(1) or "") + (m.group(3) or "")
    text = _PICTURE.sub(sub, text)
    text = re.sub(r"[ \t]+([,.;:])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip(), [refs[n - 1] for n in live]


def clothes(para, people, gender, described, undressed):
    off, on, unclear = wrd.read(para["text"], people, gender, described)
    told = {who for who, _ in para["remove"] + para["wear"]}
    off = {w: g for w, g in off.items() if w not in told}
    on = {w: g for w, g in on.items() if w not in told}
    for who, items in para["remove"]:
        off.setdefault(who, []).extend(i.lower() for i in items)
    for who, items in para["wear"]:
        on.setdefault(who, []).extend(i.lower() for i in items)
    new = {k: list(v) for k, v in undressed.items()}
    for who, gone in off.items():
        new.setdefault(who, []).extend(g for g in gone if g not in new.get(who, []))
    for who, back in on.items():
        new[who] = [g for g in new.get(who, []) if g not in back]
    return {k: v for k, v in new.items() if v}, off, on, unclear


def with_reading(para, people, gender, held):
    holds, releases, unclear = rst.read(para["text"], people, gender, held)
    told = {who for who, _ in para["hold"] + para["release"]}
    return dict(para, hold=[(w, i) for w, i in holds.items() if w not in told] + para["hold"],
                release=[(w, i) for w, i in releases.items() if w not in told] + para["release"],
                unread=para["unread"] + unclear)


def plan_shots(prompt, shot_seconds, refs, has_first_frame, memory="", anchor="", from_beat=False):
    scene, paras = parse_script(prompt, all_beats=bool((anchor or "").strip()))
    setting = [parse_paragraph(p) for p in paragraphs(anchor)] + ([scene] if scene else [])
    lead = setting + [parse_paragraph(p) for p in paragraphs(memory)]
    texts = [p["text"] for p in lead + paras]
    people = list(dict.fromkeys(roster([], [], lead + paras) + rst.agents(texts, _COMMON_CAPS | _NOT_NAMES)))
    gender = rst.genders(people, texts)
    state, undressed = {}, {}
    for k, para in enumerate(lead):
        lead[k] = with_reading(para, _mentions(para["text"], people), gender, state)
        state, _, _ = apply_holds(state, lead[k])
        undressed, _, _, _ = clothes(dict(lead[k], text=""), people, gender, {}, undressed)
    scene_text = "\n\n".join(p["text"] for p in lead if p["text"])
    described = wrd.descriptions(scene_text, people)
    names = roster([p["text"] for p in lead[len(setting):]], [p["text"] for p in lead], lead + paras)
    around, present = _mentions(" ".join(p["text"] for p in setting), names), []
    seen = _mentions(" ".join(p["text"] for p in lead), people)
    solo_seen, shots = set(), []
    for i, para in enumerate(paras):
        seen = list(dict.fromkeys(seen + _mentions(para["text"], people)))
        para = with_reading(para, seen, gender, state)
        dressed = wrd.undress(scene_text, undressed, people)
        undressed, off, on, unclear = clothes(para, seen, gender, described, undressed)
        para = dict(para, unread=para["unread"] + unclear)
        start = state
        state, released, added = apply_holds(start, para)
        during = {k: [it for it in v if it not in released.get(k, [])] for k, v in start.items()}
        during = {k: v for k, v in during.items() if v}
        keyed = (i > 0 and not para["cut"]) or (i == 0 and has_first_frame)
        named = list(dict.fromkeys(_mentions(para["text"], names) + [who for who, _ in para["hold"]]
                                   + referred(para["text"], names, gender, around)))
        cast = (named or present) if para["cut"] else list(dict.fromkeys(present + named))
        present = [n for n in cast if n not in leavers(para["text"], names) + para["exit"]]
        absent = [n for n in names if n not in cast]
        recover = [n for n in cast if n in solo_seen] if not keyed else []
        speech = bool(_SPEECH.search(para["text"]))
        text = "\n\n".join(x for x in (without_absent(dressed, absent, names), para["text"]) if x)
        drop = absent + (list(during) if keyed else []) + recover
        text, shot_refs = shot_pictures(text, refs, drop)
        sound_text, sounded = snd.sound_line(para["text"], scene_text, speech)
        notes = held_notes({who: items for who, items in during.items() if who in cast or who not in names})
        notes += [f"By the last frame, {who} has {' and '.join(items)} in plain view, and whoever put it on has let go."
                  for who, items in added.items()]
        if speech or snd.vocal(para["text"]):
            notes += [snd.muffled(who, snd.mouth_item(during.get(who, []))) for who in cast
                      if snd.mouth_item(during.get(who, []))]
        empty = not cast and nobody(para["text"], names)
        if empty:
            notes.append("Nobody is in the shot.")
        elif not speech and not snd.vocal(para["text"]):
            gags = [snd.mouth_item(during.get(who, [])) for who in cast]
            notes.append("Nobody speaks." if any(snd.held_open(g) for g in gags) else
                         "Nobody speaks, and every mouth stays closed.")
        notes += [sound_text, cast_line(cast, names, without_absent(scene_text, absent, names), para["text"])]
        if keyed:
            notes.append(f"<Picture {len(shot_refs) + 1}> is the frame this shot opens on: the same place and the same "
                         f"people, one moment earlier, carried forward rather than joined by anybody new.")
        notes += [f"<Picture {len(shot_refs) + k}> shows {who} as they look now." for k, who in enumerate(recover, 1)]
        text = "\n\n".join(x for x in (text, " ".join(n for n in notes if n)) if x)
        frames = rt.align_frame_count(round((para["seconds"] or shot_seconds) * rt.H3_FPS))
        if from_beat and not para["seconds"]:
            frames = shot_frames(para["text"], frames, bool(added))
        bound = pose_plan(during, released, added)
        latch = any(f["latch_limbs"] for f in bound.values())
        solo = present[0] if len(present) == 1 and present[0] in cast else ""
        solo_seen |= {solo} if solo else set()
        shots.append({"n": i + 1, "prompt": text, "refs": shot_refs, "frames": frames, "keyed": keyed,
                      "cut": para["cut"], "wordless": not speech, "sounded": sounded, "held": held_line(during),
                      "cast": cast, "solo": solo, "recover": recover, "bound": bound,
                      "added": held_line(added), "released": held_line(released),
                      "clothes_off": held_line(off), "clothes_on": held_line(on),
                      "unread": para["unread"] + ([u for q in lead for u in q["unread"]] if i == 0 else []),
                      "latch_after": int(math.ceil(POSE_LATCH_FROM * frames)) if latch else None})
    return shots


def _is_audio_vae(v):
    ur = getattr(v, "upscale_ratio", None)
    if isinstance(ur, (tuple, list)):
        return False
    if getattr(v, "audio_sample_rate", None) or getattr(v, "audio_sample_rate_output", None):
        return True
    if isinstance(ur, (int, float)) and getattr(v, "latent_dim", None) == 2:
        return True
    return None


def check_vaes(vae, audio_vae):
    if _is_audio_vae(audio_vae) is False:
        raise RuntimeError("audio_vae is a video VAE: wire the MiniMax-H3 audio VAE into audio_vae.")
    if _is_audio_vae(vae) is True:
        raise RuntimeError("vae is the audio VAE: the video and audio VAE inputs are swapped.")


def is_fast_h3(model):
    try:
        return getattr(model.model.diffusion_model.blocks[0].attn, "to_gate_compress", None) is not None
    except Exception:
        return False


def set_shift(model, shift_video, shift_audio):
    m = model.clone()
    try:
        ms = copy.deepcopy(m.get_model_object("model_sampling"))
        params = inspect.signature(ms.set_parameters).parameters
        kw = {k: float(v) for k, v in (("shift", shift_video), ("audio_shift", shift_audio)) if k in params}
        if kw:
            ms.set_parameters(**kw)
            m.add_object_patch("model_sampling", ms)
    except Exception:
        pass
    if isinstance(getattr(m, "model_options", None), dict):
        to = m.model_options["transformer_options"] = dict(m.model_options.get("transformer_options") or {})
        to["minimax_h3_sigma_shift_video"] = float(shift_video)
        to["minimax_h3_sigma_shift_audio"] = float(shift_audio)
    return m


def audio_sigma_of(video_sigma, shift_video, shift_audio):
    v, a, s = float(shift_video), float(shift_audio), float(video_sigma)
    base = s / (v + s * (1.0 - v))
    return a * base / (1.0 + (a - 1.0) * base)


def video_sigma_for_audio(target, shift_video, shift_audio):
    v, a, t = float(shift_video), float(shift_audio), float(target)
    base = t / (a - t * (a - 1.0))
    return base * v / (1.0 - base + base * v)


def insert_audio_landing(sigmas, shift_video, shift_audio, target=0.03, coarse=0.10):
    if len(sigmas) < 3 or sigmas[-1] != 0.0 or sigmas[-2] <= 0.0:
        return sigmas
    if audio_sigma_of(sigmas[-2], shift_video, shift_audio) <= coarse:
        return sigmas
    land = video_sigma_for_audio(target, shift_video, shift_audio)
    return sigmas[:-1] + [land, 0.0] if 0.0 < land < sigmas[-2] else sigmas


def _safetensors_meta(path):
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if not 0 < n < 64 << 20:
                return set(), {}
            h = json.loads(f.read(n))
        return set(h) - {"__metadata__"}, (h.get("__metadata__") or {})
    except Exception:
        return set(), {}


def _hyperflow_grid(value):
    try:
        grid = [float(x) for x in (json.loads(value) if isinstance(value, str) else value)]
    except (TypeError, ValueError):
        return None
    if len(grid) < 2 or abs(grid[0] - 1.0) > 1e-6 or abs(grid[-1]) > 1e-6 or any(b >= a for a, b in zip(grid, grid[1:])):
        return None
    return tuple(grid)


def hyperflow_lora(model, graph=None):
    try:
        get = getattr(model, "get_attachment", None)
        meta = get("lora_metadata") if callable(get) else (getattr(model, "attachments", None) or {}).get("lora_metadata")
    except Exception:
        meta = None
    if isinstance(meta, dict) and str(meta.get("hyperflow", "")).strip().lower() == "true":
        def num(key, default):
            try:
                return float(meta.get(key, default))
            except (TypeError, ValueError):
                return default
        return {"sigmas": _hyperflow_grid(meta.get("hyperflow_sigmas")) or HYPERFLOW_SIGMAS,
                "shift_video": num("hyperflow_video_shift", HYPERFLOW_SHIFT_VIDEO),
                "shift_audio": num("hyperflow_audio_shift", HYPERFLOW_SHIFT_AUDIO),
                "gate": num("hyperflow_gate", 0.0), "source": "LoRA metadata"}
    for node in (graph.values() if isinstance(graph, dict) else ()):
        inputs = node.get("inputs") if isinstance(node, dict) else None
        for key, value in (inputs.items() if isinstance(inputs, dict) else ()):
            if "lora_name" in str(key).lower() and isinstance(value, str) and "hyperflow" in value.lower():
                return {"sigmas": HYPERFLOW_SIGMAS, "shift_video": HYPERFLOW_SHIFT_VIDEO,
                        "shift_audio": HYPERFLOW_SHIFT_AUDIO, "gate": 0.0, "source": value}
    return None


def hyperflow_sigmas(grid, shift_video):
    s = float(shift_video)
    return torch.tensor([s * x / (1.0 + (s - 1.0) * x) for x in grid], dtype=torch.float32)


def hyperflow_endpoint_file():
    cands = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hyperflow_endpoint_*.safetensors")))
    try:
        import folder_paths
        cands += [folder_paths.get_full_path("loras", n) for n in folder_paths.get_filename_list("loras")
                  if "hyperflow" in str(n).lower()]
    except Exception:
        pass
    return next((p for p in cands if p and HYPERFLOW_ENDPOINT_KEY in _safetensors_meta(p)[0]), "")


def hyperflow_endpoint_deltas(path, strength):
    key = (path, os.path.getmtime(path), float(strength))
    if key not in _HYPERFLOW_DELTAS:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
            try:
                scale = float(meta.get("lora_alpha", 1)) / float(meta.get("lora_rank", 1))
            except (TypeError, ValueError, ZeroDivisionError):
                scale = 1.0

            def delta(module, linear):
                b = f.get_tensor(f"transformer.{module}.{linear}.lora_B.weight").float()
                return b @ f.get_tensor(f"transformer.{module}.{linear}.lora_A.weight").float()
            s = float(strength) * scale
            out = tuple(s * (delta("endpoint_time_embedder", ln) - delta("time_embedder", ln))
                        for ln in ("linear_1", "linear_2"))
        _HYPERFLOW_DELTAS.clear()
        _HYPERFLOW_DELTAS[key] = out
    return _HYPERFLOW_DELTAS[key]


def _shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def hyperflow_two_time_wrapper(gate, d_in, d_out):
    cache = {}

    def on(delta, like):
        k = (id(delta), like.device, like.dtype)
        if k not in cache:
            if len(cache) > 4:
                cache.clear()
            cache[k] = delta.to(device=like.device, dtype=like.dtype)
        return cache[k]

    def wrapper(executor, x, timestep, context, transformer_options={}, *args, **kwargs):
        dm = getattr(executor, "class_obj", None)
        te = getattr(dm, "time_embedder", None)
        ss = (transformer_options or {}).get("sample_sigmas")
        if te is None or getattr(dm, "use_adaln_curves", False) or ss is None or not gate:
            return executor(x, timestep, context, transformer_options, *args, **kwargs)
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", 12.0))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", 3.0))
        sigma = max(float(timestep.flatten()[0]) / 1000.0, 1e-6)
        grid = [float(v) for v in torch.as_tensor(ss).flatten().tolist()]
        nxt = max((v for v in grid if v < sigma - 1e-6), default=0.0)
        t_v, t_a = 1.0 - sigma, 1.0 - _shift_sigma(sigma, shift_v, shift_a)
        r_v, r_a = 1.0 - nxt, 1.0 - _shift_sigma(nxt, shift_v, shift_a)
        had = te.__dict__.get("forward")
        inner = te.forward

        def endpoint(r):
            def add(delta):
                return lambda _m, inp, out: out + torch.nn.functional.linear(inp[0].to(out.dtype), on(delta, out))
            hooks = (te.proj_in.register_forward_hook(add(d_in)), te.proj_out.register_forward_hook(add(d_out)))
            try:
                return inner(r)
            finally:
                for hk in hooks:
                    hk.remove()

        def two_time(t):
            e_t = inner(t)
            tt = t.to(torch.float32)
            r = torch.where((tt - t_a).abs() < 1e-5, torch.full_like(tt, r_a), tt)
            r = torch.where((tt - t_v).abs() < 1e-5, torch.full_like(tt, r_v), r)
            return e_t + float(gate) * (endpoint(r.to(t.dtype)).to(e_t.dtype) - e_t)
        te.forward = two_time
        try:
            return executor(x, timestep, context, transformer_options, *args, **kwargs)
        finally:
            if had is not None:
                te.forward = had
            else:
                te.__dict__.pop("forward", None)
    return wrapper


def hyperflow_two_time(model, hyper):
    try:
        dm = model.get_model_object("diffusion_model")
    except Exception:
        dm = None
    if dm is not None and (getattr(dm, "use_adaln_curves", False) or getattr(dm, "time_embedder", "absent") is None):
        return model, False, "Hyperflow endpoint conditioning off: this checkpoint has no time embedder"
    if any("hyperflow" in str(k).lower() and k != HYPERFLOW_WRAPPER_KEY
           for t in (getattr(model, "wrappers", None) or {}).values() for k in (t or {})):
        return model, False, "Hyperflow endpoint conditioning is already applied upstream"
    try:
        entries = (getattr(model, "patches", None) or {}).get(HYPERFLOW_TE_KEY) or []
        strength = float(entries[0][0]) if entries else None
    except Exception:
        strength = None
    if strength is None:
        return model, False, "Hyperflow endpoint conditioning off: the LoRA's time embedder is not on the model"
    path = hyperflow_endpoint_file()
    if not path:
        return model, False, ("Hyperflow endpoint conditioning off: put hyperflow_endpoint_v1.0.safetensors in the "
                              "node folder or the original Hyperflow LoRA in models/loras")
    gate = float(hyper.get("gate") or 0.0) or float(_safetensors_meta(path)[1].get("hyperflow_gate", 0.0) or 0.0)
    if not gate:
        return model, False, "Hyperflow endpoint gate is 0"
    import comfy.patcher_extension as pe
    d_in, d_out = hyperflow_endpoint_deltas(path, strength)
    m = model.clone()
    m.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, HYPERFLOW_WRAPPER_KEY,
                           hyperflow_two_time_wrapper(gate, d_in, d_out))
    return m, True, f"Hyperflow endpoint conditioning on (gate {gate:g})"


def apply_fast_h3_vsa(model):
    tops = (getattr(model, "model_options", None) or {}).get("transformer_options") or {}
    if (tops.get("patches_replace") or {}).get("dit"):
        return model, "FastH3 VSA left to the attention patch already on the model"
    if "cudamallocasync" in str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")).lower():
        return model, "FastH3 VSA not applied: restart ComfyUI with --disable-cuda-malloc"
    try:
        from comfy_extras.nodes_sparse_attention import apply_block_sparse_attention, parse_block_list
        m = apply_block_sparse_attention(model, tau=1.3, topk_ratio=FAST_H3_VSA_KEEP, vsa=True,
                                         start_percent=FAST_H3_VSA_START, end_percent=1.0, min_tokens=12288,
                                         dense_blocks=parse_block_list(""), sink_conditioning="exact_kv_and_rows",
                                         extra_tokens=0, verbose=False)
    except Exception as e:
        return model, f"FastH3 VSA could not be applied ({type(e).__name__}: {e})"
    return m, "FastH3 VSA on"


def prepare_model(model, steps, sampler_name, scheduler, sigmas, shift_video, shift_audio, graph):
    notes = []
    fast = is_fast_h3(model)
    hyper = None if fast else hyperflow_lora(model, graph)
    if fast and float(shift_video) == 12.0:
        shift_video = FAST_H3_SHIFT_VIDEO
    if fast:
        notes.append(f"FastH3: shift {float(shift_video):g}/{float(shift_audio):g}, {steps} steps")
    grid = hyper is not None and sigmas is None
    if grid:
        shift_video, shift_audio = hyper["shift_video"], hyper["shift_audio"]
        sigmas = hyperflow_sigmas(hyper["sigmas"], shift_video)
        steps, sampler_name = len(sigmas) - 1, HYPERFLOW_SAMPLER
        notes.append(f"Hyperflow ({hyper['source']}): {steps}-step grid, shift {shift_video:g}/{shift_audio:g}, {sampler_name}")
    model = set_shift(model, shift_video, shift_audio)
    two_time = False
    if grid:
        model, two_time, note = hyperflow_two_time(model, hyper)
        notes.append(note)
    if fast:
        model, note = apply_fast_h3_vsa(model)
        notes.append(note)
    window = sigmas
    if sigmas is None:
        window = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).float().cpu()
        if not fast:
            base = [float(x) for x in window]
            landed = insert_audio_landing(base, shift_video, shift_audio)
            if len(landed) != len(base):
                sigmas = torch.tensor(landed, dtype=torch.float32)
                notes.append("one audio landing step added")
    return model, steps, sampler_name, sigmas, window, two_time, notes


def sample(model, cond, negative, latent, seed, steps, sampler_name, scheduler, sigmas):
    if sigmas is not None:
        return rt._sample_on_sigmas(model, seed, 1.0, sampler_name, cond, negative, latent, sigmas)
    with rt._ChainNoise():
        return nodes.common_ksampler(model, seed, steps, 1.0, sampler_name, scheduler, cond, negative,
                                     latent, denoise=1.0)[0]


def foley_pass(model, cond, negative, latent, out, seed, steps, sampler_name, scheduler, sigmas):
    video, audio = out["samples"].unbind()
    lat = dict(latent, samples=comfy.nested_tensor.NestedTensor((video, torch.zeros_like(audio))),
               noise_mask=comfy.nested_tensor.NestedTensor((torch.zeros_like(video[:, :1]), torch.ones_like(audio[:, :1]))))
    return sample(model, cond, negative, lat, seed, steps, sampler_name, scheduler, sigmas)


def decode(vae, audio_vae, model, out, tiled=False):
    keep = (vae, audio_vae)
    try:
        rt.ensure_host_ram(rt._decode_ram(vae, out, tiled), keep=keep, what="the decode")
        imgs = rt._decode_video(vae, out, tiled, free_first=model, keep=keep)
    except Exception as e:
        if tiled or not rt._is_oom(e):
            raise
        rt._deep_cleanup()
        imgs = rt._decode_video(vae, out, True, free_first=model, keep=keep)
    return imgs, rt._decode_audio(audio_vae, out)


def _interrupted(e):
    return isinstance(e, getattr(mm, "InterruptProcessingException", ())) or "Interrupt" in type(e).__name__


def pose_pass(detector, imgs, shot, carry, model, pose_cn, vae, latent, strength, pose_end, window, w, h):
    try:
        det = detector.detect(imgs, stride=2)
    finally:
        detector.close()
    kw = {}
    if carry and carry[1]:
        kw["carry_appearance"] = carry[1]
    if shot["latch_after"] is not None:
        kw["latch_after"] = shot["latch_after"]
    hint, rep = pose.build_hint(det, int(imgs.shape[0]), h, w, shot["bound"], carry=carry[0] if carry else None,
                                mode="repair", draw="everyone", **kw)
    if hint is None:
        return None, rep
    shape = tuple(latent["samples"].unbind()[0].shape)
    hint_latent = pose.encode_hint(vae, hint, shape)
    if hint_latent is None:
        rep["skipped"] = "the skeleton video did not encode"
        return None, rep
    s0, s1 = pose.pose_sigma_window(window, pose_end)
    return pose.install_pose_control(model, pose_cn, vae, hint_latent, shape, strength, s0, s1), rep


def shot_line(shot):
    bits = [f"shot {shot['n']}: {shot['frames']} frames"]
    if shot["cut"]:
        bits.append("cut")
    if shot["cast"]:
        bits.append("cast: " + ", ".join(shot["cast"]))
    if shot["held"]:
        bits.append(f"held: {shot['held']}")
    if shot["added"]:
        bits.append(f"goes on: {shot['added']}")
    if shot["released"]:
        bits.append(f"comes off: {shot['released']}")
    if shot["clothes_off"]:
        bits.append(f"clothes off: {shot['clothes_off']}")
    if shot["clothes_on"]:
        bits.append(f"clothes on: {shot['clothes_on']}")
    if shot["bound"]:
        mode = "latch" if shot["latch_after"] is not None else "repair"
        bits.append(f"pose {mode}: " + ", ".join(f"{k} {v['arms'] or ''} {v['legs'] or ''}".strip()
                                               for k, v in shot["bound"].items()))
    return ", ".join(bits)


class H3LongVideos:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "resolution": (list(NATIVE_RES), {"default": "16:9"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "shot_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
                "sigmas": ("SIGMAS",),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "silence_wordless": ("BOOLEAN", {"default": True}),
                "plan_only": ("BOOLEAN", {"default": False}),
                "pose_controlnet": ("MODEL_PATCH",),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "pose_end": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05}),
                "anchor": ("STRING", {"multiline": True, "default": ""}),
                "character_memory": ("STRING", {"multiline": True, "default": ""}),
                "latent_upscale": (up.latent_models(), {"default": "off"}),
                "latent_upscale_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05}),
                "upscale": (up.FRAME_MODES, {"default": "off"}),
                "upscale_model": (up.frame_models(), {"default": "none"}),
                "upscale_target_short_edge": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
                "shot_length": (SHOT_LENGTHS, {"default": "from the beat"}),
                "ambient_audio": ("AUDIO",),
                "ambient_level": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "hidden": {"graph": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("images", "audio", "info", "script", "frames_per_shot", "total_frames", "shots", "video_seconds")
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"

    def run(self, model, clip, vae, audio_vae, prompt, resolution, megapixels, shot_seconds, steps, sampler_name,
            scheduler, seed, first_frame=None, ref_image_1=None, ref_image_2=None, ref_image_3=None,
            ref_image_4=None, sigmas=None, shift_video=12.0, shift_audio=3.0, silence_wordless=True,
            plan_only=False, pose_controlnet=None, pose_strength=1.0, pose_end=0.6, anchor="", character_memory="",
            latent_upscale="off", latent_upscale_scale=2.0, upscale="off", upscale_model="none",
            upscale_target_short_edge=0, shot_length="from the beat", ambient_audio=None, ambient_level=0.25,
            graph=None, **legacy):
        t0 = time.perf_counter()
        check_vaes(vae, audio_vae)
        w, h = frame_size(resolution, megapixels)
        shots = plan_shots(prompt, shot_seconds, [ref_image_1, ref_image_2, ref_image_3, ref_image_4],
                           first_frame is not None, character_memory, anchor, shot_length != "fixed")
        if not shots:
            raise ValueError("the prompt has no beats")
        script = "\n\n".join(f"[shot {s['n']}]\n{s['prompt']}" for s in shots)
        info = [f"{w}x{h}, {len(shots)} shots"]
        if legacy:
            info.append("ignored inputs from an older version of this node: " + ", ".join(sorted(legacy)))
        unread = [u for s in shots for u in s["unread"]]
        if unread:
            info.append("not read (write a hold: or release: line for these, as in 'hold: Mara, handcuffs behind "
                        "her back'): " + " / ".join(unread))
        if plan_only:
            info += [shot_line(s) for s in shots]
            return (torch.zeros((1, h, w, 3)), {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 44100},
                    " | ".join(info), script, shots[0]["frames"], 0, len(shots), 0.0)

        model, steps, sampler_name, sigmas, window, two_time, notes = prepare_model(
            model, steps, sampler_name, scheduler, sigmas, shift_video, shift_audio, graph)
        info += notes
        pose_ok = False
        if pose_controlnet is not None and any(s["bound"] for s in shots):
            if pose is None:
                info.append(f"pose control off: {POSE_IMPORT_ERROR}")
            else:
                pose_ok, note = pose.pose_status(model, pose_controlnet, pose_strength, two_time)
                info.append(note or "pose control on")
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

        acc = rt.FrameAccumulator(sum(s["frames"] for s in shots), rt._image_out_dtype(), True)
        audio_parts, sr = [], 44100
        handoff = first_frame[:1] if first_frame is not None else None
        detector, carry, size, captured = None, None, None, {}
        try:
            for i, shot in enumerate(shots):
                given = handoff if shot["keyed"] else None
                refs = shot["refs"] + [captured[n] for n in shot["recover"] if n in captured]
                cond, latent, fc, _ = cnd.build_conditioning(
                    clip, vae, audio_vae, shot["prompt"], w, h, shot["frames"], handoff=given, refs=refs,
                    silent=shot["wordless"] and silence_wordless,
                    lead_seconds=0.0 if shot["wordless"] else SPEECH_LEAD)
                rt._evict_all_but(model, latent)
                out = sample(model, cond, negative, latent, seed, steps, sampler_name, scheduler, sigmas)
                imgs = wav = None
                line = shot_line(shot)
                shot_carry, carry = (carry if shot["keyed"] else None), None
                if pose_ok and shot["bound"]:
                    imgs, wav = decode(vae, audio_vae, model, out)
                    if detector is None:
                        detector = pose.PoseDetector()
                    try:
                        patched, rep = pose_pass(detector, imgs, shot, shot_carry, model, pose_controlnet, vae,
                                                 latent, pose_strength, pose_end, window, w, h)
                    except Exception as e:
                        if _interrupted(e):
                            raise
                        patched, rep = None, {"skipped": f"pose check failed ({type(e).__name__}: {e})"}
                    carry = (rep["boxes_last"], rep.get("appearance_last") or None) if rep.get("boxes_last") else None
                    verdict = rep.get("skipped") or "nothing broken"
                    if patched is not None:
                        rt._evict_all_but(model, latent)
                        try:
                            out = sample(patched, cond, negative, latent, seed, steps, sampler_name, scheduler, sigmas)
                            imgs = wav = None
                            verdict = "repaired" if rep.get("broken") else "held"
                        except Exception as e:
                            if _interrupted(e):
                                raise
                            rt._deep_cleanup()
                            pose_ok = False
                            verdict = f"second pass failed ({type(e).__name__}); kept the first; pose control off"
                        patched = None
                    line += f", pose {verdict}"
                if shot["wordless"] and shot["sounded"] and silence_wordless:
                    rt._evict_all_but(model, latent)
                    out = foley_pass(model, cond, negative, latent, out, seed, steps, sampler_name, scheduler, sigmas)
                    wav = None
                    line += ", sound made for the finished picture"
                info.append(line)
                pre = None
                if latent_upscale != "off" and float(latent_upscale_scale) > 1.0:
                    video_lat, audio_lat = out["samples"].unbind()
                    upv, note = up.upscale_latent(video_lat, latent_upscale, latent_upscale_scale)
                    if note and note not in info:
                        info.append(note)
                    if upv is not video_lat:
                        pre, imgs = video_lat, None
                        out = dict(out, samples=comfy.nested_tensor.NestedTensor((upv, audio_lat)))
                if imgs is None:
                    imgs, wav = decode(vae, audio_vae, model, out, tiled=pre is not None)
                elif wav is None:
                    wav = rt._decode_audio(audio_vae, out)
                del out
                hand = imgs[-1:]
                if pre is not None:
                    tail = rt._decode_video(vae, {"samples": pre[:, :, -HANDOFF_LATENT_TAIL:].contiguous()}, True)
                    hand = tail[-1:]
                    del tail
                size = size or (int(imgs.shape[2]), int(imgs.shape[1]))
                imgs = up.fit(imgs, *size)
                grade = rt.shot_grade(given, imgs[0], imgs[-1], LEVELS,
                                      pipe=(imgs[-1], hand[-1]) if pre is not None else None) if imgs.shape[0] > 1 else None
                if grade is not None:
                    rt.grade_frames(imgs, *grade)
                    if pre is not None:
                        hand = rt.grade_frames(hand.clone(), grade[1], grade[1])
                handoff = (hand if pre is not None else imgs[-1:]).detach().clamp(0.0, 1.0).to("cpu", copy=True)
                if shot["solo"]:
                    captured[shot["solo"]] = handoff
                sr = wav["sample_rate"]
                wave = wav["waveform"]
                if given is not None and i > 0:
                    imgs = imgs[1:]
                    wave = wave[..., round(sr / rt.H3_FPS):]
                want = round(int(imgs.shape[0]) * sr / rt.H3_FPS)
                if wave.shape[-1] >= want:
                    wave = wave[..., :want]
                else:
                    wave = torch.cat([wave, wave.new_zeros(tuple(wave.shape[:-1]) + (want - wave.shape[-1],))], -1)
                acc.add(imgs)
                audio_parts.append(wave.to("cpu", copy=True))
                del imgs, wav, wave
                rt._deep_cleanup()
        except BaseException:
            acc.release()
            raise
        finally:
            if detector is not None:
                detector.release()
        video = acc.finish()
        video, note = up.upscale_frames(video, upscale, upscale_model, upscale_target_short_edge)
        if note:
            info.append(note)
        total = int(video.shape[0])
        audio, note = au.mix_ambient(torch.cat(audio_parts, dim=-1), sr, ambient_audio, ambient_level)
        if note:
            info.append(note)
        info.append(f"{total} frames, {total / rt.H3_FPS:.1f}s, rendered in {time.perf_counter() - t0:.0f}s")
        return (video, {"waveform": audio, "sample_rate": sr}, " | ".join(info), script,
                shots[0]["frames"], total, len(shots), total / rt.H3_FPS)


_NODE_IDS = ("H3LongVideos", "H3LongVideosFL2VA", "H3LongVideosV1", "H3LongVideosREF2VA")
NODE_CLASS_MAPPINGS = {name: H3LongVideos for name in _NODE_IDS}
NODE_DISPLAY_NAME_MAPPINGS = {name: "H3-LongVideos" for name in _NODE_IDS}
