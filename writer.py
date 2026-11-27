# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79
# SPDX-License-Identifier: GPL-3.0-or-later. See LICENSE.

import re

try:
    from . import bonds, clothes
    from .words import LOOSE, LOWER, WHOLE, cap, covers, has, join, many, pron, zone, UNDER
except ImportError:
    import bonds
    import clothes
    from words import LOOSE, LOWER, WHOLE, cap, covers, has, join, many, pron, zone, UNDER

COUNT = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
BELT = re.compile(rf"(?:,\s*|\s+and\s+|\s+)(?:(?:wearing|with|in|locked\s+in)\s+)?(?:a\s+|an\s+)?(?:[\w-]+\s+)?chastity[\s-]+"
                  rf"(?:belts?|devices?)(?:\s+(?:locked\s+)?(?:under(?:neath)?|beneath)\s+(?:(?:her|his|their)\s+)?(?:[\w-]+\s+){{0,3}}?"
                  rf"(?:{LOWER}|{WHOLE}|clothes|it|them)\b)?", re.I)
ONE_OF = re.compile(r"\b(?:the|a|an)\s+(?:[\w-]+\s+)?(bed|sink|stove|desk|counter|table|couch|sofa|bathtub|ladder|sawhorse|pillory|"
                    r"car|chair|cage|cross|bench)\b", re.I)


def labels(cast, people, pictured):
    hairs = [people[n]["hair"] for n in cast if n in people and people[n]["hair"]]
    return {n: n if n in pictured or len(cast) < 2 or n not in people or not people[n]["hair"] or hairs.count(people[n]["hair"]) > 1
            else f"{people[n]['hair']}-haired {n}" for n in cast}


RIDE = ("crouching", "bent over", "on all fours")


def belt_line(per, pic, on, start, kept):
    p, o, s = pron(per), pron(per, 2), pron(per, 0)
    tag = f" <Picture {pic}>" if pic else ""
    none = {"her": "panties or any other ", "his": "briefs or any other "}.get(p, "")
    bare = f"right on {p} bare skin, and {s} {'wear' if s == 'they' else 'wears'} no {none}underwear with it."
    if not on:
        return f"{cap(p)} chastity belt{tag} goes on {o} in this shot" + (f" underneath {p} {join(start)}" if start else "") + f", {bare}"
    if kept:
        return ""
    belt = f"chastity belt{tag} is locked on {o}, its band around {p} waist and its shield between {p} legs over {p} crotch, {bare}"
    them = "they are" if len(start) > 1 or start and many(start[-1]) else "it is"
    return f"Under {p} {join(start)}, {p} {belt} It comes into view only once {them} out of the way." if start else f"{cap(p)} {belt}"


def loose(kept):
    return bool(kept) and all(has(LOOSE, g) for g in kept)


def resting(where):
    return (f"bunched {where[8:]}" if where.startswith("down to just") else f"bunched around {where[8:]}" if where.startswith("down to ")
            else f"gathered up at {where[6:]}" if where.startswith("up to ") else f"pushed {where}" if where.startswith("up ")
            else f"hanging {where}")


def sheet(name, per, pic, worn_now, off, worn, lifted, held, added, skin, belt_pic, start, kept, ride):
    p, s = pron(per), pron(per, 0)
    hide = clothes.hidden(worn_now, off, lifted)
    moved = [(g, w) for g, w in lifted if w != "up into place"]
    desc = clothes.undress(per["desc"], list(off) + [g for g, _ in moved], hide)
    if start:
        desc = BELT.sub("", desc)
    look = [desc] + [f"{name} also wears {p} {g}." for g in worn if g not in off and g not in {x for x, _ in moved}]
    look += [f"{cap(p)} {g} {'are' if many(g) else 'is'} {resting(w)}, and {'they stay' if many(g) else 'it stays'} there."
             for g, w in moved]
    belt = next((i for i in held + added if i["gear"] == "chastity belt"), None)
    belted = belt in held
    vis = [g for g in worn_now if g not in off and g not in hide]
    outer = kept if belted else [g for g in vis if "lower" in covers(g) and g not in {x for x, _ in lifted}]
    beneath = [] if not (ride and loose(outer)) else [bonds.name(belt) + (f" <Picture {belt_pic}>" if belt_pic else "")] if belted \
        else [g for g in hide if "lower" in zone(g)]
    if belted or beneath or any("lower" in zone(g) for g in off if g != "*"):
        low = [f"{p} {g}" for g in vis if "lower" in zone(g) and not (belted and has(UNDER, g))]
        low += [f"{p} chastity belt"] if belted and not start else []
        under = f", and under {'them' if many(outer[-1]) else 'it'} nothing but {p} {join(beneath)}" if beneath else ""
        look += [f"Below the waist, {s} {'wear' if s == 'they' else 'wears'} only {join(low)}{under}."] if low else []
    look += [belt_line(per, belt_pic, belt in held, start, kept)] if belt else []
    look += [f"{cap(p)} skin shows {join(skin)}."] if skin else []
    text = " ".join(x.strip() if x.strip().endswith((".", "!", "?")) else x.strip() + "." for x in look if x and x.strip())
    return f"{name}{f' <Picture {pic}>' if pic else ''}: {text}"


def posture(who, per, call, pose, changed):
    if not pose:
        return []
    p = pron(per)
    where = f" {'in' if re.search(r'chair|stool|seat', pose['where']) else 'on'} {pose['where']}" if pose["where"] else ""
    if pose["pose"] == "bent over":
        where = f" over {pose['where']}" if pose["where"] else ""
    way = f" {pose['way'].replace('her', p)}" if pose.get("way") else ""
    rest = " Bent over it, her chest rests on it.".replace("her", p) if pose["pose"] == "bent over" and re.search(
        r"bed|desk|table|bench|counter", pose["where"]) else ""
    return [f"{cap(call(who))} ends the shot {pose['pose']}{way}{where}.{rest}" if changed else
            f"{cap(call(who))} stays {pose['pose']}{way}{where} for the whole shot, holding that position from the first frame to the "
            f"last.{rest}"]


def penetration(ev, call, people, pose):
    d = ev["items"][0]
    into, by = ev["who"], d["by"]
    p = pron(people.get(into, {}))
    tool = {"penis": f"{call(by)}'s penis", "fingers": f"{call(by)}'s fingers"}.get(d["tool"], f"the {d['tool']}")
    way = d["way"] or {"lying": "top", "bent over": "behind", "on all fours": "behind", "kneeling": "standing"}.get(
        (pose or {}).get("pose"), "standing")
    if (pose or {}).get("pose") == "lying" and (pose or {}).get("way") == "face down":
        way = "behind"
    line = {"behind": f"{call(by)} is right behind {call(into)}, his hips against her buttocks, and {tool} goes into her {d['hole']} "
                      f"level, pointing into her toward her head.",
            "top": f"{call(by)} is on top of {call(into)}, between her thighs as she lies on her back, and {tool} goes into her "
                   f"{d['hole']} level, in line with her body.",
            "astride": f"{call(into)} sits astride {call(by)}'s hips, facing him, and {tool} points straight up into her {d['hole']}, "
                       f"in line with her spine.",
            "side": f"{call(by)} lies behind {call(into)} on their sides, and {tool} goes into her {d['hole']} from behind, level.",
            "standing": f"{tool} goes into {call(into)}'s {d['hole']} from below, vertical and parallel to her thighs, at the center of "
                        f"her crotch."}[way]
    line = line.replace("her", p).replace("his", pron(people.get(by, {}))) if p != "her" else line
    return [cap(line) + f" It stays lined up with {p} body the whole time."]


def speech(cues, said, cast, call, people):
    spoken = [ln for ln in said if not ln["gag"] and not ln["thought"] and ln["who"]]
    muffled = [ln for ln in said if ln["gag"] and ln["who"]]
    out = []
    talkers = list(dict.fromkeys(ln["who"] for ln in spoken))
    if len(talkers) == 1:
        n, p, s = talkers[0], pron(people.get(talkers[0], {})), pron(people.get(talkers[0], {}), 0)
        out.append(f"Only {call(n)} speaks in this shot, in {p} own voice, with natural, expressive intonation and real emotion, the way "
                   f"people really talk: {p} lips move only while {s} {'say' if s == 'they' else 'says'} the line, and every word of it "
                   f"is {'theirs' if s == 'they' else 'hers' if s == 'she' else 'his'}.")
    elif talkers:
        out.append(f"{join([call(n) for n in talkers])} each speak only their own lines, in their own voices: each one's lips move only "
                   f"while they say their own words, and nobody else's lips move.")
    listeners = [n for n in cast if n not in talkers and n not in {ln["who"] for ln in muffled}]
    if talkers and listeners:
        out.append(f"{cap(join([call(n) for n in listeners]))} {'listen' if len(listeners) > 1 else 'listens'} with closed lips.")
    for n in dict.fromkeys(ln["who"] for ln in muffled):
        out.append(f"{cap(call(n))} makes only a muffled, closed-mouth mmmph through the gag, with no words.")
    out.append("Apart from the line, everyone stays silent." if spoken else "Apart from the muffled sounds, everyone stays silent."
               if muffled else "Everyone stays silent, with their lips closed.")
    out.append("All the sound comes from the scene itself: its natural ambience and the real sounds of what happens in it.")
    return out


def presence(cast, extras, text, lead, call, camera):
    out = []
    if cast and not extras:
        out.append("There is one person in the shot: one body, one face." if len(cast) == 1 else
                   f"There are {COUNT.get(len(cast), len(cast))} people in the shot, with one body for each person.")
    things = list(dict.fromkeys(m.group(1).lower() for m in ONE_OF.finditer(" ".join([text] + lead))))
    out += [f"There is only one {t} here, always in the same spot." for t in things
            if not re.search(rf"\b{t}s\b|\b(?:two|three|other|second|another)\s+(?:[\w-]+\s+)?{t}", " ".join([text] + lead), re.I)]
    if cast and not camera:
        out.append(f"The camera keeps {call(cast[0])}'s face in frame for the whole shot." if len(cast) == 1 else
                   f"The camera keeps the faces of {join([call(n) for n in cast])} in frame for the whole shot.")
    return out
