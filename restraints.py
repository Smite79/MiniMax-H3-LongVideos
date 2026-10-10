# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import re

KINDS = (
    ("handcuffs", r"(?:hand)?cuffs|manacles", r"(?:hand)?cuff(?:s|ed|ing)?|manacl(?:es|ed|ing)", "wrists", "cuffs"),
    ("zip ties", r"zip[\s-]?ties|cable\s+ties", r"zip[\s-]?ti(?:es|ed|ing)", "wrists", "zip"),
    ("shackles", r"shackles|leg\s+irons", r"shackl(?:es|ed|ing)", "ankles", "shackles"),
    ("chains", r"chains?(?!\s+leash)", r"chain(?:s|ed|ing)", "wrists", "chain"),
    ("duct tape", r"(?:duct\s+|gaffer\s+|packing\s+)?tape", r"tap(?:es|ed|ing)", "mouth", "tape"),
    ("rope", r"ropes?|cords?|twine", r"ti(?:es|ed|ing)|binds?|binding|bound", "wrists", "rope"),
    ("ball gag", r"ball[\s-]?gag", r"(?!)", "mouth", "gag"),
    ("gag", r"gag|cloth|rag", r"gag(?:s|ged|ging)", "mouth", "gag"),
    ("blindfold", r"blindfold", r"blindfold(?:s|ed|ing)", "eyes", "blindfold"),
    ("collar", r"collar|(?:(?:chain|leather|metal|steel)\s+)?leash", r"collar(?:s|ed)|leash(?:es|ed)", "neck", "collar"),
)
_OWN = r"(?:her|his|their|[A-Z][a-z]+['’]s)"
PARTS = (("hips", r"hips?|waist|crotch|groin|pelvis|between\s+(?:her|his|their|[A-Z][a-z]+['’]s)\s+(?:legs|thighs)"),
         ("torso", rf"(?:around|across|over|about|off|from)\s+{_OWN}\s+(?:(?:entire|whole|upper)\s+)?(?:body|torso|chest|breasts?|"
                   rf"bust|stomach|belly|midriff|ribs|middle)(?:\s+and\s+(?:{_OWN}\s+)?arms)?|(?:pinn(?:ing|ed)|pins?)\s+{_OWN}\s+arms(?:\s+(?:to|against|at)\s+"
                   rf"{_OWN}\s+sides)?|arms\s+(?:to|against|at)\s+{_OWN}\s+sides"),
         ("wrists", r"wrists?|hands|arms"), ("ankles", r"ankles?|feet|legs|knees"), ("mouth", r"mouth|lips|face|cheeks|jaw"),
         ("eyes", r"eyes"), ("neck", r"neck|throat"))
APPLY = (r"puts?|putting|snaps?|snapped|locks?|locked|locking|clicks?|clicked|fastens?|fastened|clamps?|clamped|slaps?|"
         r"slapped|places?|placed|clips?|clipped|wraps?|wrapped|wrapping|winds?|presses?|pressed|sticks?|stuffs?|stuffed|"
         r"shoves?|forces?|pushes?|ties?|tied|tying|binds?|bound|binding|loops?|secures?|secured|securing|straps?|strapped|"
         r"buckles?|slips?|restrains?|restrained|restraining|pins?|pinned|pinning|cinch(?:es|ed)?|uses?|used|using|"
         r"attach(?:es|ed|ing)?|hooks?|hooked")
REPORT = {"handcuffs", "zip ties", "shackles", "ball gag", "blindfold"}
REMOVE = (r"remov(?:e|es|ed|ing)|unlock(?:s|ed|ing)?|unfasten(?:s|ed|ing)?|unbuckl(?:e|es|ed|ing)|unwrap(?:s|ped|ping)?|"
          r"unclip(?:s|ped|ping)?|unhook(?:s|ed|ing)?|detach(?:es|ed|ing)?")
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
_CLAUSE = re.compile(r"[,;]\s+(?=(?:then|and then|while|as|his|her|their|he|she|they|[A-Z][a-z]+)\b)")
_PART = r"mouth|lips|eyes|neck|wrists|hands|arms|ankles|feet|hips|waist|legs"
_WHO = r"(?P<who>her|his|their|(?-i:[A-Z])[a-z]+(?=['’]s))(?:['’]s)?"
_BIT = (r"(?:a|an|the|another|one|two|three|more)\s+(?:\w+\s+)?(?:strip|piece|length|band|loop|coil|section|layer|"
        r"turn)s?")
WRAPS = ("duct tape", "rope", "chains")
_AROUND = re.compile(r"\b(?:around|about)\s+(?:her|him|them|(?-i:[A-Z])[a-z]+)\b(?!['’])", re.I)
_SEX = {"her": "f", "she": "f", "him": "m", "his": "m", "he": "m"}
_BEHIND = re.compile(r"\b((?:wrists?|hands|arms|them|(?:hand)?cuff(?:s|ed)?|tied|bound|taped|zip[\s-]?tied|chained|shackled)\s+"
                     r"(?:together\s+)?)behind\s+(her|him|them)\b(?!\s+back\b)", re.I)
_BACK = {"her": "her", "him": "his", "them": "their"}
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
    want = _SEX.get(token.lower())
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
        around = f"wound all the way around {poss} {word}, front and back" + (
            f", and between {poss} legs" if re.search(r"\bbetween\b", scope, re.I) else "")
        return f"{kind} {around}"
    if part == "torso":
        pin = f", pinning {poss} arms to {poss} sides" if re.search(r"\barms\b", scope, re.I) else ""
        return f"{kind} wound all the way around {poss} torso, front and back{pin}"
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
        leash = re.search(r"\b((?:chain|leather|metal|steel)\s+)?leash", scope, re.I)
        if leash:
            return f"a collar around {poss} neck, with a {(leash.group(1) or '').lower()}leash clipped to it" + (
                f" and tied {anchor}" if anchor else "")
        return f"a collar around {poss} neck" + (f" {anchor}" if anchor else "")
    return f"{kind} on {poss} {part}" + tail


def _subject(sentence, toks, m):
    cands = [(p, t) for p, t in reversed(toks) if p < m.start() and t.lower() not in ("her", "his", "their", "him", "them")
             and sentence[p + len(t):p + len(t) + 2] not in ("'s", "’s")]
    near = [t for p, t in cands if re.fullmatch(r"\s+(?:(?:\w+ly|also|now|just)\s+)?", sentence[p + len(t):m.start()])]
    lead = [t for p, t in cands if re.search(r"(?:^|[,;:]|\b(?:and|as|while|when|then|but|before|after|until|once))\s*$",
                                             sentence[:p], re.I)]
    return (near + lead + [t for _, t in cands] + [None])[0]


def _described(mode, m, lead, before, after, verb):
    if mode == "worn":
        return not re.search(rf"\b(?:{APPLY}|{verb})\b", lead, re.I)
    if mode == "state":
        return bool(re.fullmatch(r"(?:.*(?:\bwith|,))?\s*(?:the|a|an)?\s*", lead, re.I | re.S)) and not re.search(
            rf"\b(?:{APPLY}|{verb}|is|are|was|were|gets?|got|being|goes|went)\b", m.group("gap"), re.I)
    return mode == "verb" and not re.search(r"\b(?:is|are|was|were|gets?|got|getting|being|been)\s+(?:\w+ly\s+)?$", before,
                                            re.I) and not re.match(r"[^,;.]*\bby\s+(?:[A-Z]|him\b|her\b|them\b)", after)


def _wearer(sentence, toks, m, mode, people, gender):
    before = sentence[:m.start()]
    near_before = next((t for p, t in reversed(toks) if p < m.start()), None)
    first = m.group(0).split()[0].lower()
    start = m.end() if mode == "verb" else m.start() + len(first)
    obj = next(((p, t) for p, t in toks if start <= p <= m.end() + (25 if mode == "verb" else 45)), None)
    if mode == "verb" and obj is not None and sentence[m.end():obj[0]].strip():
        obj = None
    obj = obj[1] if obj else None
    passive = mode in ("state", "worn") or (mode in ("verb", "noun") and first.endswith(("ed", "bound"))
                                            and (obj is None or _AUX.search(before) is not None))
    if mode in ("state", "it"):
        token, actor = m.group("who"), ""
    elif passive:
        token, actor = near_before, ""
    elif obj is None:
        return None, "", False
    else:
        subject = _subject(sentence, toks, m)
        actor = resolve(subject, people, gender) if subject else ""
        token = obj
    if token is None:
        return None, "", False
    limb = [x for x in re.finditer(r"\b(her|his|their|[A-Z][a-z]+(?=['’]s))(?:['’]s)?\s+(?:\w+\s+)?(?:wrists?|hands|arms|ankles?|"
                                   r"feet|legs)\b", before)]
    if token.lower() in ("them", "it") and limb:
        token = limb[-1].group(1)
    who, want = resolve(token, people, gender, exclude=(actor,) if actor else ()), _SEX.get(token.lower())
    if not who and token.lower() in PERSON.split("|"):
        who = next((t for p, t in reversed(toks) if p < m.start() and t in people and t != actor
                    and (want is None or gender.get(t, want) == want)), "")
    return who, {"f": "her", "m": "his"}.get(want, ""), passive


def read(text, people, gender, held):
    holds, releases, unclear, worn = {}, {}, [], {}
    for sentence in sentences(text):
        sentence = _BEHIND.sub(lambda b: f"{b.group(1)}behind {_BACK[b.group(2).lower()]} back", sentence)
        toks, taken = tokens(sentence, people), []
        for kind, noun, verb, default, key in KINDS:
            found = ([(m, "noun") for m in re.finditer(rf"\b(?:{APPLY})\s+(?:[\w'’-]+\s+){{0,3}}?(?:{noun})\b", sentence, re.I)]
                     + [(m, "worn") for m in re.finditer(rf"\b(?:in|with|wears?|wearing|wore|against)\s+(?:a\s+pair\s+of\s+|a\s+set\s+of\s+|"
                                                          rf"the\s+|her\s+|his\s+|their\s+|some\s+|an?\s+)?(?:(?!(?:and|or|with|in|"
                                                          rf"on)\b)[\w-]+\s+){{0,2}}?(?:{noun})\b", sentence, re.I)]
                     + [(m, "verb") for m in re.finditer(rf"(?<![\w-])(?:{verb})\b", sentence, re.I)]
                     + [(m, "state") for m in re.finditer(
                         rf"\b(?:{noun})\s+(?P<gap>(?:(?!(?:and|then|or|but|while|as)\b)[\w'’-]+\s+){{0,2}}?)(?:over|across|"
                         rf"around|on|onto|in|into|between|covers?|covering|seals?|sealing|holds?|holding|keeps?|{APPLY})\s+"
                         rf"{_WHO}\s+(?:{_PART})\b", sentence, re.I)]
                     + [(m, "it") for m in re.finditer(
                         rf"\b(?:{noun})\b[^.;]{{0,60}}?(?P<act>\b(?:{APPLY})\s+(?:it|them|{_BIT})\s+(?:\w+\s+)?(?:over|across|around|"
                         rf"on|onto|to|into|between|behind)\s+{_WHO}\s+(?:{_PART}|back)\b)", sentence, re.I)])
            for m, mode in found:
                span = m.span("act") if mode == "it" else m.span()
                if any(a < span[1] and span[0] < b for a, b in taken):
                    continue
                before, after = sentence[:m.start()], sentence[m.end():]
                lead = _CLAUSE.split(before)[-1]
                if re.search(r"un$", before, re.I) or re.match(r"\s+(?:off|from|away)\b", after, re.I):
                    continue
                if mode == "verb" and re.search(r"\b(?:the|a|an|her|his|their|some|of|pair)\s*$", before, re.I):
                    continue
                if kind == "rope" and mode == "verb" and not re.search(
                        r"\b(?:wrists?|hands|arms|ankles?|feet|legs|knees|up|together)\b|\b(?:rope|cord)s?\b", after, re.I):
                    continue
                if kind == "gag" and mode == "noun" and "gag" not in m.group(0).lower() and not re.search(r"\bmouth\b", after, re.I):
                    continue
                if mode == "worn" and m.group(0)[:4].lower() == "with" and not parts_in(lead):
                    continue
                who, said, passive = _wearer(sentence, toks, m, mode, people, gender)
                if who is None and key == "collar" and re.search(r"leash", m.group(0), re.I):
                    on = list(dict.fromkeys(n for n, items in list(held.items()) + list(holds.items())
                                            if any("collar" in i.lower() for i in items)))
                    who, said, passive = (on[0], "", False) if len(on) == 1 else (None, "", False)
                if who is None:
                    continue
                taken.append(span)
                if not who:
                    unclear.append(f"who wears the {kind} in '{sentence.strip()}'")
                    continue
                pos, anchor = _POS.search(after) or _POS.search(sentence), _ANCHOR.search(after)
                scope = _CLAUSE.split(lead + sentence[m.start():] if mode == "worn" else
                                      sentence[span[0]:] if mode in ("state", "it") else after)[0]
                scope = m.group(0) + scope if key == "collar" and mode in ("noun", "verb") else scope
                already = passive and _described(mode, m, lead, before, after, verb)
                known = owner(held.get(who, []) + holds.get(who, [])) or {"f": "her", "m": "his"}.get(gender.get(who), "")
                fallback = ("torso" if kind in WRAPS and _AROUND.search(scope) else default)
                for part, rx, spot in parts_in(scope) or [(fallback, dict(PARTS)[fallback], "")]:
                    have = held.get(who, []) + holds.get(who, [])
                    same = [h for h in have if key in h.lower() and (part in h.lower() or kind == "handcuffs")]
                    if same and not (pos and part == "wrists" and not any(_POS.search(h) for h in same)) and not (
                            key == "collar" and re.search(r"\bleash", scope, re.I) and not any("leash" in h for h in same)):
                        continue
                    own = re.search(r"\b(her|his|their)\b", spot, re.I) or \
                        re.search(rf"\b(her|his|their)\s+(?:\w+\s+)?(?:{rx})\b", scope, re.I)
                    if own and (said or known) and own.group(1).lower() != (said or known):
                        continue
                    poss = own.group(1).lower() if own else said or known or f"{who}'s"
                    item = phrase(kind, part, pos.group(0) if pos else "", poss, anchor.group(0) if anchor else "", scope)
                    for old in same:
                        if old in holds.get(who, []):
                            holds[who].remove(old)
                            if old in worn.get(who, []):
                                already = True
                                worn[who].remove(old)
                        elif old not in releases.get(who, []):
                            releases.setdefault(who, []).append(old)
                    holds.setdefault(who, []).append(item)
                    if already:
                        worn.setdefault(who, []).append(item)
                    if owner([item]) and who not in gender:
                        gender[who] = "f" if owner([item]) == "her" else "m"
        for kind, noun, verb, _default, key in KINDS:
            if kind in REPORT and re.search(rf"\b(?:{noun})\b|(?<![\w-])(?:{verb})\b", sentence, re.I) and \
                    not any(key in i.lower() for items in list(held.values()) + list(holds.values()) for i in items) and \
                    not any(kind in u and sentence.strip() in u for u in unclear):
                unclear.append(f"the {kind} in '{sentence.strip()}' were not read" if kind.endswith("s")
                               else f"the {kind} in '{sentence.strip()}' was not read")
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
                after = _CLAUSE.split(sentence[m.start():])[0]
                place = next(((n, rx) for n, rx, _ in parts_in(after)), None)
                tok = next((t for p, t in toks if p >= m.start() + 2), None)
                holders = [n for n, items in held.items() if any(key in i.lower() for i in items)]
                who = resolve(tok, people, gender) if tok else ""
                who = who if who in holders else (holders[0] if len(holders) == 1 else "")
                if not who:
                    continue
                items = [i for i in held[who] if key in i.lower()
                         and (place is None or re.search(rf"\b(?:{place[1]})\b", i, re.I))]
                leash = key == "collar" and "leash" in m.group(0).lower()
                items = [i for i in items if "leash" in i] if leash else items
                if len(items) == 1:
                    releases.setdefault(who, []) if items[0] in releases.get(who, []) else \
                        releases.setdefault(who, []).append(items[0])
                    if leash:
                        plain = re.sub(r",\s+with\s+an?\s+.*$", "", items[0])
                        holds.setdefault(who, []).append(plain)
                        worn.setdefault(who, []).append(plain)
                elif len(items) > 1:
                    unclear.append(f"which {kind} comes off in '{sentence.strip()}'")
    return holds, releases, unclear, worn
