# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79
# SPDX-License-Identifier: GPL-3.0-or-later. See LICENSE.

import re

try:
    from .words import EXTRA, GARMENT, HOSE, LOOSE, LOWER, SHOES, UNDER, UPPER, WHOLE, covers, has, many, zone
except ImportError:
    from words import EXTRA, GARMENT, HOSE, LOOSE, LOWER, SHOES, UNDER, UPPER, WHOLE, covers, has, many, zone

SPLIT = re.compile(rf"(,\s*(?:and\s+)?|\s+and\s+|\.\s*|\s+(?:over|under(?:neath)?|beneath|with)\s+(?=(?:[\w-]+\s+){{0,4}}?"
                   rf"{GARMENT.pattern}))", re.I)
BY = re.compile(r"\s*(?:over|under(?:neath)?|beneath|with)\s*", re.I)
TAIL = re.compile(r"\s+(?:underneath|beneath|under\s+(?:it|them))$", re.I)
LEAD = re.compile(r"^(?:(?:she|he|they)\s+(?:is|are)\s+)?(?:in|wearing|wears|dressed\s+in|clad\s+in|with|has\s+on)\s+|"
                  r"^(?:she|he|they)\s+(?:wears?|has\s+on|is\s+wearing)\s+", re.I)
ART = re.compile(r"^(?:a|an|the|some|her|his|their|a\s+pair\s+of|pair\s+of)\s+", re.I)
GENERIC = {"top": UPPER, "tops": UPPER, "bottoms": LOWER, "bottom": LOWER, "underwear": UNDER, "lingerie": UNDER, "shoes": SHOES,
           "pants": LOWER, "trousers": LOWER}
LEAVE = ", and {s} steps out of {it} one foot at a time"
ORDER = ((EXTRA, "takes off {w} {g}"),
         (r"jackets?|blazers?|(?:rain|over|pea|top|fur)?coats?|cardigans?|robes?|bathrobes?|kimonos?|parkas?|ponchos?|shawls?",
          "slides {w} {g} off {p} shoulders and down {p} arms"),
         (SHOES, "slips {w} {g} off one foot at a time"),
         (r"corsets?|bustiers?", "unlaces {w} {g} and takes {it} off"),
         (r"blouses?|shirts?", "unbuttons {w} {g} from the top down and slides {it} off {p} shoulders and down {p} arms"),
         (WHOLE, "slides {w} {g} off {p} shoulders and down {p} body to {p} ankles" + LEAVE),
         (UPPER, "pulls {w} {g} up over {p} head and off {p} arms"),
         (r"(?:mini)?skirts?", "opens {w} {g} at the waist and slides {it} down {p} legs to {p} ankles" + LEAVE),
         (LOWER, "opens {w} {g} at the waist and pulls {it} down {p} legs to {p} ankles" + LEAVE),
         (HOSE, "rolls {w} {g} down {p} legs and off {p} feet"),
         (r"bras?|bralettes?", "unhooks {w} {g} and slides the straps off {p} shoulders and down {p} arms"),
         (r"diapers?|nappy|nappies", "unfastens the tabs on {w} {g} and takes {it} off"),
         (UNDER, "slides {w} {g} down {p} legs to {p} ankles" + LEAVE))


def pieces(desc):
    parts = SPLIT.split(desc or "")
    return [(parts[k - 1] if k else "", parts[k]) for k in range(0, len(parts), 2)]


def plain(seg):
    return TAIL.sub("", ART.sub("", LEAD.sub("", seg.strip()).strip()).strip(" ."))


def garments(desc):
    return [plain(s) for _, s in pieces(desc) if GARMENT.search(s)]


def fit(word, worn):
    word = plain(word).lower()
    if re.fullmatch(r"(?:all\s+)?(?:(?:her|his|their)\s+)?(?:clothes|clothing|everything|outfit|all)", word):
        return list(worn)
    exact = [g for g in worn if re.search(rf"\b{re.escape(word)}\b", g, re.I)]
    if exact:
        return exact[:1] if len(exact) > 1 and any(g.lower() == word for g in exact) else exact
    rx = GENERIC.get(word.split()[-1]) if word else None
    hits = [g for g in worn if rx and has(rx, g)] or [g for g in worn if word and word.split()[-1].rstrip("s") in g.lower()]
    return hits[:1]


def undress(desc, off=(), hide=()):
    gone = {g.lower() for g in list(off) + list(hide)}
    out, lead, carry = [], "", None
    for joint, seg in pieces(desc):
        if GARMENT.search(seg) and plain(seg).lower() in gone:
            m = LEAD.match(seg.strip())
            lead, carry = lead or (m.group(0) if m else ""), (None if BY.fullmatch(joint) else joint) if carry is None else carry
            if seg.rstrip().endswith(".") and out:
                out[-1] = (out[-1][0], out[-1][1].rstrip(" .") + ".")
            continue
        joint, carry = (carry if carry is not None and GARMENT.search(seg) else joint), None
        if lead and GARMENT.search(seg) and not LEAD.match(seg.strip()):
            seg = lead + seg.strip()
        lead = ""
        out.append((joint, seg))
    text = ""
    for k, (joint, seg) in enumerate(out):
        nxt = k + 1 < len(out) and GARMENT.search(out[k + 1][1])
        if k and re.match(r"\s*,", joint) and GARMENT.search(seg) and not nxt and GARMENT.search(out[k - 1][1]):
            joint = " and "
        elif k and joint.strip() == "and" and nxt:
            joint = ", "
        text += (joint if k else "") + seg
    text = re.sub(r",\s*(?=[.]|$)", "", text.strip())
    text = re.sub(r"\s+(?:in|wearing|and)\s*\.?$", "", text).strip(" ,")
    return text if not desc.strip().endswith(".") or text.endswith(".") else text + "."


def hidden(worn, off=(), lifted=()):
    on = [g for g in worn if g not in off]
    up = {g for g, _ in lifted}
    cover = set().union(*[covers(g) for g in on if g not in up]) if on else set()
    return [g for g in on if has(UNDER, g) and zone(g) <= cover and g not in up]


def surface(worn, lifted=()):
    up = {g for g, wh in lifted if wh != "up into place"}
    on = [g for g in worn if g not in up and "lower" in zone(g)]
    fitted = sorted((g for g in on if not has(LOOSE, g)),
                    key=lambda g: next((k for k, rx in enumerate((LOWER, WHOLE, HOSE)) if has(rx, g)), 3))
    return next(iter(fitted), None), next((g for g in on if has(LOOSE, g)), None)


def rank(g):
    return next((k for k, (rx, _) in enumerate(ORDER) if has(rx, g)), len(ORDER))


def strip_step(g, down, x, p, s, by, floor):
    text = ("pulls {w} {g} the rest of the way down {p} " + ("legs" if zone(g) == {"lower"} else "body") + " to {p} ankles" + LEAVE) \
        if down else next((t for rx, t in ORDER if has(rx, g)), "takes off {w} {g}")
    text += f", and {by + ' ' if by and LEAVE in text else ''}drops {{it}} to the {floor}" if floor else ""
    return text.format(w=x, g=g, p=p, s=s, it="them" if many(g) else "it")


def owned(g, p):
    return f"{p} {g}"
