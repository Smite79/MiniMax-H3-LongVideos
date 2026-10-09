# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import re
from h3_restraints import sentences, resolve

GARMENT = (r"(?:t-?shirt|shirt|blouse|crop\s+top|tank\s+top|top|sweater|jumper|hoodie|cardigan|jacket|coat|blazer|vest|"
           r"dress|gown|skirt|shorts|trousers|pants|jeans|leggings|tights|stockings|socks|shoes|boots|heels|sneakers|"
           r"gloves|hat|cap|scarf|belt|robe|bathrobe|pajamas|pyjamas|nightgown|nightie|lingerie|bra|bralette|panties|"
           r"knickers|underwear|briefs|boxers|thong|g-string|bikini\s+(?:top|bottoms?)|bikini|swimsuit|corset|"
           r"bodysuit|uniform|apron)")
_TAKE = (r"takes?|took|taking|pulls?|pulled|pulling|slips?|slipped|slides?|slid|peels?|peeled|strips?|stripped|tugs?|"
         r"tugged|yanks?|yanked|rips?|ripped|tears?|tore|cuts?|kicks?|kicked|shrugs?|shrugged|wriggles?|wriggled|"
         r"lowers?|lowered|works?|worked|eases?|eased")
_OFF = re.compile(rf"\b(?:remov(?:e|es|ed|ing)|sheds?)\s+(?P<pre>(?:[\w'’-]+\s+){{0,4}}?)(?P<g>{GARMENT})\b"
                  rf"|\b(?:{_TAKE})\s+(?P<off>off\s+)?(?P<pre2>(?:[\w'’-]+\s+){{0,4}}?)(?P<h>{GARMENT})\b"
                  rf"(?P<tail>\s+(?:off|away|down|free)\b)?", re.I)
_SHEET = re.compile(r"^\s*([A-Z][\w'’-]{0,24}(?:\s+[A-Z][\w'’-]{0,24}){0,2})\s*:\s*\S")
_ON = re.compile(rf"\b(?:puts?|putting|pulls?|pulled|slips?|slipped|slides?|slid|gets?|got|tugs?)\s+"
                 rf"(?P<pre>(?:[\w'’-]+\s+){{0,3}}?)(?P<g>{GARMENT})\s+(?:back\s+)?on\b"
                 rf"|\b(?:puts?|putting|pulls?|slips?)\s+(?:back\s+)?on\s+(?P<pre2>(?:[\w'’-]+\s+){{0,3}}?)(?P<h>{GARMENT})\b", re.I)


def garment_of(word):
    return re.sub(r"\s+", " ", word.lower())


def mentions(text, garment):
    return bool(re.search(rf"\b{re.escape(garment)}s?\b", text or "", re.I))


_DET = {"a", "an", "her", "his", "their", "the", "my", "its", "some", "one"}
_STOP = {"wears", "wear", "wearing", "worn", "is", "are", "was", "were", "in", "with", "and", "or", "only", "just", "but",
         "of", "on", "under", "over", "into", "to", "from", "she", "he", "they", "still", "now", "also", "nothing"}


_TRAIL = re.compile(rf"(?:\s+(?:underwear|panties|briefs|bottoms?|knickers))?(?:\s*,?\s+(?:made\s+(?:of|from)|of|in|with|"
                    rf"cut|trimmed|lined|edged|covered|that|which|stretched|clinging|pulled|riding|sitting|hugging)\b"
                    rf".*?(?=\s*(?:[,;.!?]|\band\s+(?:a|an|the|her|his|their)\b|$)|\s+(?:a|an|the|her|his|their|matching)\s+"
                    rf"(?:[\w-]+\s+){{0,2}}?(?:{GARMENT})\b))?", re.I)
_ITS = re.compile(r"^\s*It(?:['’]s|s|\s+(?:is|was|has))\b", re.I)


def strip(text, garment):
    while True:
        m = re.search(rf"\b{re.escape(garment)}s?\b", text, re.I)
        if m is None:
            break
        start = m.start()
        for w in reversed(list(re.finditer(r"[\w'’-]+", text[:start]))[-5:]):
            low = w.group(0).lower()
            if low in _STOP or (w.group(0)[0].isupper() and low not in _DET):
                break
            start = w.start()
            if low in _DET:
                break
        before, after = text[:start], text[m.end():]
        after = after[_TRAIL.match(after).end():]
        if re.search(r",\s*$", before):
            before = re.sub(r",\s*$", "", before)
        elif re.search(r"\s+and\s*$", before):
            before = re.sub(r"\s+and\s*$", "", before)
        elif re.match(r"\s*,\s*", after):
            after = re.sub(r"^\s*,\s*", " ", after)
        elif re.match(r"\s+and\s+", after):
            after = re.sub(r"^\s+and\s+", " ", after)
        text = before + after
    text = re.sub(r"\b(?:wears|wearing|is\s+wearing|dressed\s+in|in|with)(?:\s+only|\s+just|\s+nothing\s+but)?\s*(?=[,.;:!?]|$)",
                  "", text, flags=re.I)
    text = re.sub(r"\b(wears|wearing|in)\s+(?:over|under|on\s+top\s+of|with)\s+", r"\1 ", text, flags=re.I)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*([.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return "" if re.fullmatch(r"(?:[A-Z][\w'’-]*|she|he|they)?\s*[.!?]?", text, re.I) else text


def _owners(text, people, last=None):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        named = [n for n in people if re.search(rf"\b{re.escape(n)}\b", s)]
        last = named[0] if len(named) == 1 else (None if named else last)
        out.append((s, named or ([last] if last else [])))
    return out


def descriptions(text, people):
    out = {}
    for line in (text or "").splitlines():
        sheet = _SHEET.match(line)
        for s, who in _owners(line, people, sheet.group(1) if sheet and sheet.group(1) in people else None):
            for n in who:
                out[n] = (out.get(n, "") + " " + s).strip()
    return out


def undress(text, undressed, people):
    if not undressed:
        return text
    lines = []
    for line in text.splitlines():
        sheet = _SHEET.match(line)
        kept, gone = [], False
        for s, who in _owners(line, people, sheet.group(1) if sheet and sheet.group(1) in people else None):
            if gone and _ITS.match(s):
                continue
            gone = False
            for n in who:
                for g in undressed.get(n, []):
                    gone = gone or mentions(s, g)
                    s = strip(s, g)
            kept.append(s)
        lines.append(" ".join(k for k in kept if k))
    return "\n".join(lines)


def read(text, people, gender, described):
    off, on, unclear = {}, {}, []
    for sentence in sentences(text):
        for rx, out in ((_OFF, off), (_ON, on)):
            for m in rx.finditer(sentence):
                if rx is _OFF and m.group("h") and not (m.group("off") or m.group("tail")):
                    continue
                garment = garment_of(m.group("g") or m.group("h"))
                own = re.search(r"\b(her|his|their)\b|\b([A-Z][a-z]+)['’]s\b",
                                (m.group("pre") if m.group("g") else m.group("pre2")) or "")
                who = resolve(own.group(1) or own.group(2), people, gender) if own else ""
                owners = [n for n in people if mentions(described.get(n, ""), garment)]
                who = who or (owners[0] if len(owners) == 1 else "")
                if not who:
                    unclear.append(f"whose {garment} in '{sentence.strip()}'")
                    continue
                if garment not in out.get(who, []):
                    out.setdefault(who, []).append(garment)
    return off, on, unclear
