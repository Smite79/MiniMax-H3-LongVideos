# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import copy
import re

try:
    from . import bonds, clothes, voice
    from .reader import FACE, POSE, WHERE, Reader
    from .script import STATE, beats, names_in, read_memory, split_args, unquoted
    from .shot import Shot
    from .words import FPS, GARMENT, LOWER, PIC, SKIRT, UNDER, align, has, join, sentences, zone
except ImportError:
    import bonds
    import clothes
    import voice
    from reader import FACE, POSE, WHERE, Reader
    from script import STATE, beats, names_in, read_memory, split_args, unquoted
    from shot import Shot
    from words import FPS, GARMENT, LOWER, PIC, SKIRT, UNDER, align, has, join, sentences, zone

__all__ = ["FPS", "join", "plan"]
SURE = {"handcuffs", "gag", "blindfold", "collar", "shackles", "leash", "hogtie", "zip ties"}
CARRIED = re.compile(r"\b(?:(?<!chastity[\s-])belt|hip|holster|pocket|hands?|bag|hold(?:s|ing)?|carr(?:y|ies|ying)|dangl\w*|hang\w*|"
                     r"tattoo\w*|necklace|bracelet)\b", re.I)


class World:
    def __init__(self, people):
        self.people = people
        self.held, self.off, self.worn, self.lifted, self.skin, self.pose, self.doors = {}, {}, {}, {}, {}, {}, {}

    def wearing(self, n):
        per = self.people.get(n)
        off = self.off.get(n, [])
        base = [] if "*" in off else clothes.garments(per["desc"]) if per else []
        return [g for g in base + self.worn.get(n, []) if g not in off]

    def bare_under(self, n, going):
        return [g for g in self.wearing(n) if zone(g) == {"lower"} and (has(UNDER + "|tights|pantyhose", g) or
                                                                         going and has(LOWER, g) and not has(SKIRT, g))]


def restrained(seg):
    return any(re.search(r"(?:ed|bound)$", m.group(0), re.I) and (k in SURE or bonds.spots(seg))
               for k, rx in bonds.VERB.items() for m in rx.finditer(seg))


def gear_items(text):
    out = []
    for _, seg in clothes.pieces(text):
        seg = re.sub(r"^\W*(?:(?:she|he|they)\s+(?:is|are)\s+)?(?:in|wearing|wears|with|has\s+on|naked\s+(?:except|but)\s+for)\s+"
                     r"(?:only\s+|just\s+)?", "", seg.strip(" ."), flags=re.I)
        if out and not GARMENT.search(seg) and bonds.spots(seg) and not restrained(seg) and \
                not any(rx.search(seg) for rx in bonds.NOUN.values()):
            out[-1] = f"{out[-1]} and {seg}"
            continue
        out.append(seg)
    return [i for seg in out if not GARMENT.search(seg) and (restrained(seg) or any(rx.search(seg) for rx in bonds.NOUN.values())
                                                              and not CARRIED.search(seg)) for i in bonds.parse(seg)]


def pose_of(text):
    for name, rx in POSE:
        if re.search(rf"\b(?:{rx})\b", text, re.I) or re.search(rf"\b{name}\b", text, re.I):
            where, face = WHERE.search(text), FACE.search(text)
            way = ("face down" if face and (face["f"] == "down" or face["s"] in ("front", "stomach", "belly")) else
                   "on her back" if face and (face["s"] == "back" or face["f"] == "up") else "on her side" if face and face["s"] == "side"
                   else "") if name == "lying" else ""
            return {"pose": name, "where": where["w"].lower() if where and name != "standing" else "", "way": way}
    return None


def tag_events(b, people):
    out = []
    for kind in STATE:
        for arg in b[kind]:
            who, rest = split_args(arg, people)
            if kind == "hold":
                items = [i for part in rest.split(";") if part.strip() for i in bonds.parse(part)]
            elif kind == "release":
                items = [i for part in re.split(r"[;,]", rest) if part.strip() for i in (bonds.parse(part) or [
                    {"gear": None, "spot": s, "pose": None, "to": None} for s in bonds.spots(part)])] or ["*"]
            elif kind in ("remove", "tear", "wear"):
                items = [clothes.plain(x) for x in re.split(r",\s*|\s+and\s+", rest) if x.strip()] or ["*"]
            elif kind == "lift":
                m = re.match(r"(.*?)[\s,]+(up|down|aside|open)\b(.*)$", rest, re.I)
                items = [(clothes.plain(m.group(1)), (m.group(2) + m.group(3)).strip().lower())] if m else [(clothes.plain(rest), None)]
            elif kind == "pose":
                items = [pose_of(rest) or {"pose": rest.strip(), "where": "", "way": ""}]
            else:
                items = [rest] if rest else []
            out.append({"kind": kind, "who": who, "items": items, "doer": None, "sent": f"{kind}: {arg}"})
    return out


def plan(prompt, memory="", anchor="", shot_seconds=10.0, from_beat=True, first_frame=False, pictures=(), silence=True):
    people, extra, start = read_memory(memory)
    paras, warn = beats(prompt), []
    anchor = (anchor or "").strip()
    scenery = lambda b: not anchor and not names_in(unquoted(b["text"]), people) and not voice.QUOTE.search(b["text"]) \
        and not any(b[k] for k in STATE + ("exit",))
    scene = ([extra] if extra else []) + ([paras.pop(0)["text"]] if paras and scenery(paras[0]) else [])
    lead, out = ([anchor] if anchor else []) + scene, []
    for k, b in enumerate(paras):
        if not (b["cut"] and k + 1 < len(paras) and scenery(b)):
            out.append(dict(b, cut=True, scene=paras[k - 1]["text"]) if k and paras[k - 1]["cut"] and scenery(paras[k - 1]) else b)
    paras = out
    if not paras:
        return [], ["the prompt has no beats"]
    owner = {v["head"]: n for n, v in people.items() if v["head"]}
    belt_pic = next((int(m.group(1) or m.group(2)) for t in [memory or "", anchor] + [p["text"] for p in paras]
                     for m in re.finditer(rf"chastity[\s-]+belts?[^.\n<]{{0,40}}{PIC}|{PIC}[^.\n<]{{0,40}}?chastity[\s-]+belts?", t, re.I)
                     if int(m.group(1) or m.group(2)) not in owner), None)
    w = World(people)
    for n, v in people.items():
        w.held[n] = bonds.put([], gear_items(v["desc"]))
    for e in tag_events(start, people):
        start_state(w, e)
    if anchor:
        reader = Reader(people, list(people))
        for e in reader.read(anchor)[0]:
            if e["kind"] in ("have", "hold"):
                w.held[e["who"]] = bonds.put(w.held.get(e["who"], []), e["items"])
            elif e["kind"] == "pose":
                w.pose[e["who"]] = e["items"][0]
    for n in people:
        if any(i["gear"] == "chastity belt" for i in w.held.get(n, [])):
            w.off[n] = w.off.get(n, []) + w.bare_under(n, False)
    reader, shots, present, prev, beside, last = Reader(people, [], dress=lambda n: w.wearing(n)), [], [], [], {}, None
    ceiling = align(round(float(shot_seconds) * FPS))
    paras = [dict(b, beat=k + 1) for k, b in enumerate(paras)]

    def attempt(b):
        i, text = len(shots), b["text"]
        tagged = any(b[k] for k in STATE)
        tags_on = [split_args(a, people)[0] for k in STATE for a in b[k]]
        cast = list(dict.fromkeys(([] if b["cut"] or i == 0 else present) + names_in(unquoted(text), people)
                                  + [n for n in tags_on if n in people]))
        reader.cast, reader.held = cast, w.held
        events, missed = (tag_events(b, people), []) if tagged else reader.read(text, not b.get("part"))
        warn.extend(f"shot {i + 1}: \"{s[:60]}\" was not read: {why}" for s, why in missed)
        warn.extend(f"shot {i + 1}: '{e['who']}' has no line in character_memory, so the {e['kind']}: tag changed nothing"
                    for e in events if e["who"] not in people)
        shot = Shot(w, people, cast, i, warn)
        for e in events:
            if e["who"] in people:
                shot.apply(e)
        keyed = (i > 0 and not b["cut"]) or (i == 0 and bool(first_frame))
        coming = [n for n in cast if keyed and i > 0 and n not in prev]
        leaving = [split_args(a, people)[0] for a in b["exit"]]
        built = shot.build(b, lead, text, keyed, coming, leaving, owner, belt_pic, pictures, ceiling, from_beat, silence, beside, last)
        return shot, built, cast, leaving

    def tried(b):
        nonlocal w, beside
        saved = (copy.deepcopy(w, {id(people): people}), dict(reader.__dict__), len(warn), copy.deepcopy(beside))
        shot = attempt(b)[0]
        w, beside = saved[0], saved[3]
        reader.__dict__.update(saved[1])
        del warn[saved[2]:]
        return shot

    k = 0
    while k < len(paras):
        b = paras[k]
        if b.get("scene"):
            lead = ([extra] if extra else []) + [b["scene"]]
            w.pose.clear()
        said = sentences(b["text"])
        if len(said) > 1 and not b["seconds"] and not any(b[x] for x in STATE):
            shot = tried(b)
            if shot.moves and shot.need > ceiling / FPS + 0.5:
                j = next((j for j in range(len(said) - 1, 1, -1) if tried(dict(b, text=" ".join(said[:j]), exit=[])).need <=
                          ceiling / FPS + 0.5), 1)
                paras[k:k + 1] = [dict(b, text=" ".join(said[:j]), exit=[], split=True),
                                  dict(b, text=" ".join(said[j:]), cut=False, part=True, split=True, scene=None)]
                continue
        shot, built, cast, leaving = attempt(b)
        shots.append(built)
        last = shot.voice_last
        warn += shot.warn_after
        present, prev = [n for n in cast if n not in leaving], cast
        k += 1
    return shots, warn


def start_state(w, e):
    n = e["who"]
    if e["kind"] == "hold":
        w.held[n] = bonds.put(w.held.get(n, []), e["items"])
    elif e["kind"] in ("remove", "tear"):
        w.off[n] = w.off.get(n, []) + e["items"]
    elif e["kind"] == "wear":
        w.worn[n] = w.worn.get(n, []) + e["items"]
    elif e["kind"] == "lift":
        w.lifted[n] = w.lifted.get(n, []) + [(g, way or "up") for g, way in e["items"]]
    elif e["kind"] == "mark":
        w.skin[n] = w.skin.get(n, []) + e["items"]
    elif e["kind"] == "pose":
        w.pose[n] = e["items"][0]
