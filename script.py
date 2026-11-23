# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import re

try:
    from .words import PIC, sentences
except ImportError:
    from words import PIC, sentences

STATE = ("hold", "release", "remove", "wear", "lift", "tear", "mark", "unmark", "pose")
TAG = re.compile(rf"(?:^|(?<=[.!?\"”]\s))\s*(?P<k>{'|'.join(STATE)}|exit|seconds)\s*:\s*(?P<a>[^\n]*?)\s*(?=$|\s(?:{'|'.join(STATE)}|exit|"
                 rf"seconds)\s*:)", re.I | re.M)
NAME = r"[A-ZÀ-ÞŁŻŹŚĆŃ][\w'’À-ÿłżźśćńęąó-]*(?:\s+[A-ZÀ-ÞŁŻŹŚĆŃ][\w'’À-ÿłżźśćńęąó-]*)?"
LINE = re.compile(rf"^[\s\-*•>#]*\**(?P<name>{NAME})\**\s*(?:\(\s*{PIC}\s*\)|{PIC})?\s*(?::|,|\s-\s|\s–\s)\s*(?P<desc>.*)$|"
                  rf"^[\s\-*•>#]*\**(?P<only>{NAME})\**\s*{PIC}\s*$", re.I)
WOMAN = re.compile(r"\b(?:woman|women|girl|lady|female|wife|mother|mom|sister|daughter|girlfriend|she|her|herself|actress|queen|"
                   r"princess|nurse|maid|bride|mistress|blonde|brunette|redhead)\b", re.I)
MAN = re.compile(r"\b(?:man|men|guy|boy|male|husband|father|dad|brother|son|boyfriend|he|him|his|himself|actor|king|prince|"
                 r"gentleman|groom|master)\b", re.I)
HAIR = re.compile(r"\b(?!(?:her|his|their|the|a|an|and|of|with|no|long|short|curly|wavy|straight|messy|thick|thin)\b)"
                  r"([a-z]+(?:-[a-z]+)?)\s+(?:hair|curls|bob|ponytail|braids?)\b", re.I)


def person(desc, pic):
    sex = "f" if WOMAN.search(desc) and not (MAN.search(desc) and MAN.search(desc).start() < WOMAN.search(desc).start()) \
        else "m" if MAN.search(desc) else None
    hair = HAIR.search(desc)
    return {"desc": re.sub(rf"\s*{PIC}", "", desc, flags=re.I).strip(), "sex": sex, "hair": hair.group(1).lower() if hair else "",
            "head": int(pic) if pic else None}


def tags(text):
    found, rest = {k: [] for k in STATE + ("exit",)}, text
    seconds, cut = None, False
    for m in TAG.finditer(text):
        k, a = m["k"].lower(), m["a"].strip().rstrip(".")
        if k == "seconds":
            seconds = float(re.sub(r"[^\d.]", "", a) or 0) or None
        else:
            found[k].append(a)
    rest = TAG.sub("", text)
    lines = [x for x in rest.split("\n") if x.strip()]
    cut = any(re.fullmatch(r"\W*cut\W*", x, re.I) for x in lines)
    return " ".join(x.strip() for x in lines if not re.fullmatch(r"\W*cut\W*", x, re.I)), found, seconds, cut


def read_memory(memory):
    people, extra, start = {}, [], {k: [] for k in STATE + ("exit",)}
    for raw in (memory or "").split("\n"):
        line, found, _, _ = tags(raw)
        for k, v in found.items():
            start[k] += v
        m = LINE.match(line) if line.strip() and not re.match(rf"\s*{PIC}", line, re.I) else None
        name = (m["name"] or m["only"]) if m else None
        if name and not re.fullmatch(r"(?:the|a|an|scene|setting|place|location|note|style|camera)", name, re.I):
            name = name if not name.isupper() or len(name) < 3 else name.title()
            pic = re.search(PIC, line, re.I)
            people[name] = person(m["desc"] or "", pic.group(1) if pic else None)
        elif line.strip():
            extra.append(line.strip())
    return people, " ".join(extra), start


def beats(prompt):
    out = []
    for block in re.split(r"\n\s*\n", (prompt or "").strip()):
        text, found, seconds, cut = tags(block)
        if not text and out and not cut:
            for k, v in found.items():
                out[-1][k] += v
            out[-1]["seconds"] = seconds or out[-1]["seconds"]
            continue
        if not text and not any(found.values()):
            if cut and out:
                out[-1]["cut_after"] = True
            continue
        out.append({"text": text, "seconds": seconds, "cut": cut, **found})
    for k, b in enumerate(out):
        b["cut"] = b["cut"] or bool(k and out[k - 1].pop("cut_after", False))
    return out


def names_in(text, people):
    found = []
    for n in sorted(people, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w'’-]){re.escape(n)}(?![\w-])", text):
            if not any(a <= m.start() < b for _, a, b in found):
                found.append((n, m.start(), m.end()))
    return [n for n, _, _ in sorted(found, key=lambda x: x[1])]


def unquoted(text):
    return re.sub(r"[\"“][^\"”]*[\"”]", lambda m: " " * len(m.group(0)), text)


def split_args(arg, people):
    m = re.match(r"\s*([^,:;]+?)\s*(?:,|;|:|$)\s*(.*)$", arg)
    who = m.group(1).strip() if m else arg.strip()
    hit = next((n for n in people if n.lower() == who.lower()), None) or next(
        (n for n in people if who.lower().startswith(n.lower() + " ")), None)
    rest = m.group(2) if m else ""
    if hit and hit.lower() != who.lower():
        rest = (who[len(hit):].strip() + (", " + rest if rest else "")).strip()
    return hit or who, rest.strip()


def first_sentence(text):
    s = sentences(text)
    return s[0] if s else text
