# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import re

KINDS = (
    ("handcuffs", r"(?:hand)?cuffs|manacles", r"(?:hand)?cuff(?:s|ed|ing)?|manacl(?:es|ed|ing)", "wrists", "cuffs"),
    ("zip ties", r"zip[\s-]?ties|cable\s+ties", r"zip[\s-]?ti(?:es|ed|ing)", "wrists", "zip"),
    ("shackles", r"shackles|leg\s+irons", r"shackl(?:es|ed|ing)", "ankles", "shackles"),
    ("chains", r"chains?", r"chain(?:s|ed|ing)", "wrists", "chain"),
    ("duct tape", r"(?:duct\s+|gaffer\s+|packing\s+)?tape", r"tap(?:es|ed|ing)", "mouth", "tape"),
    ("rope", r"ropes?|cords?|twine", r"ti(?:es|ed|ing)|binds?|binding|bound", "wrists", "rope"),
    ("ball gag", r"ball[\s-]?gag", r"(?!)", "mouth", "gag"),
    ("gag", r"gag|cloth|rag", r"gag(?:s|ged|ging)", "mouth", "gag"),
    ("blindfold", r"blindfold", r"blindfold(?:s|ed|ing)", "eyes", "blindfold"),
    ("collar", r"collar", r"collar(?:s|ed)", "neck", "collar"),
)
PARTS = (("hips", r"hips?|waist|crotch|groin|pelvis|between\s+(?:her|his|their|[A-Z][a-z]+['’]s)\s+(?:legs|thighs)"),
         ("wrists", r"wrists?|hands|arms"), ("ankles", r"ankles?|feet|legs|knees"), ("mouth", r"mouth|lips|face|cheeks|jaw"),
         ("eyes", r"eyes"), ("neck", r"neck|throat"))
APPLY = (r"puts?|putting|snaps?|locks?|clicks?|fastens?|clamps?|slaps?|places?|clips?|wraps?|wrapping|winds?|presses?|"
         r"sticks?|stuffs?|shoves?|forces?|pushes?|ties?|tied|tying|binds?|loops?|secures?|straps?|buckles?|slips?")
REMOVE = r"remov(?:e|es|ed|ing)|unlock(?:s|ed|ing)?|unfasten(?:s|ed|ing)?|unbuckl(?:e|es|ed|ing)|unwrap(?:s|ped|ping)?"
STRIP = (r"takes?|took|taking|pulls?|pulled|pulling|rips?|ripped|ripping|peels?|peeled|peeling|tears?|tore|tearing|"
         r"yanks?|yanked|strips?|stripped|cuts?|cutting|slices?|sliced|snips?|snipped|loosens?|loosened|works?|worked")
CUT = r"cuts?|cutting|slices?|sliced|snips?|snipped|saws?|sawed"
OWNED = r"(?:the|her|his|their|[A-Z][a-z]+['’]s)\s+(?:[\w'’-]+\s+){0,2}?"
UNDO = (("uncuff", "handcuffs"), ("unti", "rope"), ("unbind", "rope"), ("unbound", "rope"), ("ungag", "gag"),
        ("unblindfold", "blindfold"), ("unchain", "chains"), ("unshackl", "shackles"), ("untap", "duct tape"))
PERSON = r"her|him|them|his|their|she|he|they"
_POS = re.compile(r"\b(?:behind\s+(?:her|his|their|the)\s+back|in\s+front(?:\s+of\s+(?:her|him|them|the\s+body))?|"
                  r"(?:above|over)\s+(?:her|his|their|the)\s+head|at\s+(?:her|his|their|the)\s+waist)\b", re.I)
_ANCHOR = re.compile(r"\bto\s+(?:the|a|an)\s+[\w-]+(?:\s+(?:frame|post|rail|bar|pipe|ring|hook|leg))?\b", re.I)
_AUX = re.compile(r"\b(?:is|are|was|were|gets?|got|been|being|remains?|stays?|still)\s+(?:\w+ly\s+)?$", re.I)
_FEMALE = r"woman|girl|lady|wife|mother|sister|daughter"
_MALE = r"man|boy|guy|husband|father|brother|son"
_AGENT = re.compile(r"\b([A-Z][a-z]+)(?:['’]s\b|\s+(?:\w+ly\s+)?[a-z]+(?:s|ed)\b)")


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def agents(texts, stop):
    return list(dict.fromkeys(n for t in texts for n in _AGENT.findall(t or "") if n not in stop))


def genders(names, texts):
    out = {}
    for t in texts:
        for s in sentences(t):
            named = [n for n in names if re.search(rf"\b{re.escape(n)}\b", s)]
            for n in named:
                said = re.search(rf"\b{re.escape(n)}\b(?:['’]s)?\s*(?::|,|\bis\b|\bwas\b)\s*(?:\w+\s+){{0,3}}?"
                                 rf"(?:({_FEMALE})|({_MALE}))\b", s, re.I)
                if said:
                    out.setdefault(n, "f" if said.group(1) else "m")
            if len(named) == 1 and re.search(r"\bherself\b", s, re.I):
                out.setdefault(named[0], "f")
            if len(named) == 1 and re.search(r"\bhimself\b", s, re.I):
                out.setdefault(named[0], "m")
    return out


def owner(items):
    found = [m.group(1).lower() for it in items for m in re.finditer(r"\b(her|his)\b", it, re.I)]
    return found[0] if found and all(f == found[0] for f in found) else ""


def tokens(text, people):
    names = "|".join(re.escape(n) for n in people) or "(?!)"
    spans = [m.span() for m in _POS.finditer(text)]
    return [(m.start(), m.group(1) or m.group(2)) for m in re.finditer(rf"\b({names})(?:['’]s)?\b|\b((?i:{PERSON}))\b", text)
            if not any(a <= m.start() < b for a, b in spans)]


def resolve(token, people, gender, exclude=()):
    if token in people:
        return token if token not in exclude else ""
    want = {"her": "f", "she": "f", "him": "m", "his": "m", "he": "m"}.get(token.lower())
    pool = [n for n in people if n not in exclude]
    known = [n for n in pool if want and gender.get(n) == want]
    pool = known or [n for n in pool if want is None or gender.get(n, want) == want]
    return pool[0] if len(pool) == 1 else ""


def parts_in(text):
    out, rest = [], _POS.sub(" ", text)
    for name, rx in PARTS:
        m = re.search(rf"\b(?:{rx})\b", rest, re.I)
        if m:
            out.append((name, rx, m.group(0)))
            rest = re.sub(rf"\b(?:{rx})\b", " ", rest, flags=re.I)
    return out


def phrase(kind, part, pos, poss, anchor, scope=""):
    if part == "hips":
        word = "waist" if re.search(r"\bwaist\b", scope, re.I) else "hips"
        around = f"around {poss} {word}" + (f" and between {poss} legs" if re.search(r"\bbetween\b", scope, re.I) else "")
        return f"{'duct tape' if kind == 'duct tape' else kind} {around}"
    tail = (f" {pos}" if pos and part == "wrists" else "") + (f" {anchor}" if anchor else "")
    if kind == "handcuffs":
        return (f"handcuffs {pos}" if pos else f"handcuffs on {poss} {part}") + (f" {anchor}" if anchor else "")
    if kind == "duct tape":
        return (f"duct tape over {poss} {part}" if part in ("mouth", "eyes") else f"duct tape around {poss} {part}") + tail
    if kind == "rope":
        return f"rope around {poss} {part}" + tail
    if kind in ("ball gag", "gag"):
        return f"a {kind} in {poss} mouth"
    if kind == "blindfold":
        return f"a blindfold over {poss} eyes"
    if kind == "collar":
        return f"a collar around {poss} neck" + (f" {anchor}" if anchor else "")
    return f"{kind} on {poss} {part}" + tail


def _wearer(sentence, toks, m, mode, people, gender):
    before, after = sentence[:m.start()], sentence[m.end():]
    near_before = next((t for p, t in reversed(toks) if p < m.start()), None)
    obj = next((t for p, t in toks if m.end() <= p <= m.end() + (25 if mode == "verb" else 45)), None)
    passive = mode == "state" or (mode == "verb" and m.group(0).lower().endswith(("ed", "bound"))
                                  and (obj is None or _AUX.search(before) is not None))
    if mode == "state":
        token, actor = re.search(rf"\b(her|his|their|[A-Z][a-z]+)(?=['’]s\b|\s)", m.group(0).split(None, 1)[-1]).group(1), ""
    elif passive:
        token, actor = near_before, ""
    elif obj is None:
        return None, ""
    else:
        actor = resolve(near_before, people, gender) if near_before else ""
        token = obj
    if token is None:
        return None, ""
    who = resolve(token, people, gender, exclude=(actor,) if actor else ())
    if not who and token.lower() in PERSON.split("|"):
        who = next((t for p, t in reversed(toks) if p < m.start() and t in people and t != actor), "")
    return who, {"her": "her", "she": "her", "him": "his", "his": "his", "he": "his"}.get(token.lower(), "")


def read(text, people, gender, held):
    holds, releases, unclear = {}, {}, []
    for sentence in sentences(text):
        toks, taken = tokens(sentence, people), []
        for kind, noun, verb, default, key in KINDS:
            found = ([(m, "noun") for m in re.finditer(rf"\b(?:{APPLY})\s+(?:[\w'’-]+\s+){{0,3}}?(?:{noun})\b", sentence, re.I)]
                     + [(m, "verb") for m in re.finditer(rf"(?<![\w-])(?:{verb})\b", sentence, re.I)]
                     + [(m, "state") for m in re.finditer(
                         rf"\b(?:{noun})\s+(?:[\w'’-]+\s+){{0,2}}?(?:over|across|around|on|in|between|covers?|covering|seals?|sealing)\s+"
                         rf"(?:her|his|their|[A-Z][a-z]+['’]s)\s+"
                         rf"(?:mouth|lips|eyes|neck|wrists|ankles|hips|waist|legs)\b", sentence, re.I)])
            for m, mode in found:
                if any(a < m.end() and m.start() < b for a, b in taken):
                    continue
                taken.append(m.span())
                before, after = sentence[:m.start()], sentence[m.end():]
                if re.search(r"un$", before, re.I) or re.match(r"\s+(?:off|from|away)\b", after, re.I):
                    continue
                if mode == "verb" and re.search(r"\b(?:the|a|an|her|his|their|some|of|pair)\s*$", before, re.I):
                    continue
                if kind == "rope" and mode == "verb" and not re.search(
                        r"\b(?:wrists?|hands|arms|ankles?|feet|legs|knees|up|together)\b|\b(?:rope|cord)s?\b", after, re.I):
                    continue
                if kind == "gag" and mode == "noun" and "gag" not in m.group(0).lower() and not re.search(r"\bmouth\b", after, re.I):
                    continue
                who, said = _wearer(sentence, toks, m, mode, people, gender)
                if who is None:
                    continue
                if not who:
                    unclear.append(f"who wears the {kind} in '{sentence.strip()}'")
                    continue
                pos, anchor = _POS.search(after) or _POS.search(sentence), _ANCHOR.search(after)
                scope = re.split(r"[,;]\s+(?=(?:then|and then|while|as|his|her|their|he|she|they|[A-Z][a-z]+)\b)",
                                 sentence[m.start():] if mode == "state" else after)[0]
                known = owner(held.get(who, []) + holds.get(who, [])) or {"f": "her", "m": "his"}.get(gender.get(who), "")
                for part, rx, spot in parts_in(scope) or [(default, dict(PARTS)[default], "")]:
                    have = held.get(who, []) + holds.get(who, [])
                    if any(key in h.lower() and (part in h.lower() or kind == "handcuffs") for h in have):
                        continue
                    own = re.search(r"\b(her|his|their)\b", spot, re.I) or \
                        re.search(rf"\b(her|his|their)\s+(?:\w+\s+)?(?:{rx})\b", scope, re.I)
                    if own and (said or known) and own.group(1).lower() != (said or known):
                        continue
                    poss = own.group(1).lower() if own else said or known or f"{who}'s"
                    item = phrase(kind, part, pos.group(0) if pos else "", poss, anchor.group(0) if anchor else "", scope)
                    holds.setdefault(who, []).append(item)
                    if owner([item]) and who not in gender:
                        gender[who] = "f" if owner([item]) == "her" else "m"
        applied = {k for who in holds for k in (r[4] for r in KINDS) if any(k in h.lower() for h in holds[who])}
        for kind, noun, _verb, _default, key in KINDS:
            undo = [w for w, k in UNDO if k == kind]
            pats = [rf"\b(?:{REMOVE})\s+{OWNED}(?:{noun})\b",
                    rf"\b(?:{STRIP})\s+(?:off|away|loose|free)\s+{OWNED}(?:{noun})\b",
                    rf"\b(?:{STRIP})\s+{OWNED}(?:{noun})\s+(?:off|away|loose|free)\b",
                    rf"\b(?:{CUT})\s+(?:through\s+|away\s+)?{OWNED}(?:{noun})\b(?!\s+(?:onto|around|over|across|into)\b)"]
            pats += [rf"\b(?:{'|'.join(undo)})\w*"] if undo else []
            for m in (m for pat in pats for m in re.finditer(pat, sentence, re.I)):
                if key in applied and not m.group(0).lower().startswith(("remov", "unwrap", "un")):
                    continue
                after = re.split(r"[,;]\s+(?=(?:then|and then|while|as|his|her|their|he|she|they|[A-Z][a-z]+)\b)",
                                 sentence[m.start():])[0]
                place = next(((n, rx) for n, rx, _ in parts_in(after)), None)
                tok = next((t for p, t in toks if p >= m.start() + 2), None)
                holders = [n for n, items in held.items() if any(key in i.lower() for i in items)]
                who = resolve(tok, people, gender) if tok else ""
                who = who if who in holders else (holders[0] if len(holders) == 1 else "")
                if not who:
                    continue
                items = [i for i in held[who] if key in i.lower()
                         and (place is None or re.search(rf"\b(?:{place[1]})\b", i, re.I))]
                if len(items) == 1:
                    releases.setdefault(who, []) if items[0] in releases.get(who, []) else \
                        releases.setdefault(who, []).append(items[0])
                elif len(items) > 1:
                    unclear.append(f"which {kind} comes off in '{sentence.strip()}'")
    return holds, releases, unclear
