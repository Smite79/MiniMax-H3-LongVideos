# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79
# SPDX-License-Identifier: GPL-3.0-or-later. See LICENSE.

import re

try:
    from .words import cap, join, sentences
except ImportError:
    from words import cap, join, sentences

PER_ACTION = 2.2
WORDS_PER_SEC = 2.8
SAY_SLACK = 0.6
SENTENCE_PAUSE = 0.35
MUFFLE_SOUND = 1.5
CUE_GAP = 0.4
SPEECH_LEAD = 0.5
SPEECH_TAIL = 0.8
SAYS = r"says?|said|asks?|asked|whispers?|whispered|shouts?|shouted|yells?|yelled|screams?|screamed|replies|replied|answers?|" \
       r"answered|mutters?|muttered|murmurs?|murmured|growls?|growled|orders?|ordered|commands?|commanded|tells?|told|calls?|called|" \
       r"cries|cried|snaps?|snapped|laughs?|laughed|adds?|added|continues?|continued|begs?|begged|pleads?|pleaded|demands?|" \
       r"demanded|warns?|warned|moans?|moaned|sighs?|sighed|gasps?|gasped|purrs?|purred|teases?|teased|explains?|explained|" \
       r"insists?|insisted|breathes|breathed|hisses|hissed|giggles?|giggled|sobs?|sobbed|groans?|groaned|mumbles?|mumbled"
THINK = r"thinks?|thought|wonders?|wondered"
QUOTE = re.compile(r"[\"“](?P<q>[^\"”]+)[\"”]")
MUFFLE = re.compile(r"\b(?:screams?|moans?|whimpers?|mumbles?|cries|sobs?|groans?|squeals?|yelps?|grunts?|pleads?|begs?|"
                    r"protests?|shrieks?)\b(?:\s+(?:\w+\s+){0,3}?(?:into|through|behind|against)\s+(?:the|her|his)\s+"
                    r"(?:[\w-]+\s+)?(?:gag|tape|cloth|panties|sock))?", re.I)
CLAUSE = re.compile(r"[.;!?]+|,\s+(?:and\s+|then\s+)")


class Voices:
    def __init__(self, people, cast, gagged, resolve, last=None):
        self.people, self.cast, self.gagged, self.resolve, self.last = people, cast, gagged, resolve, last

    def name(self, word, pre=""):
        word = word.strip()
        hit = next((n for n in self.cast if n.lower() == word.lower() or n.upper() == word), None)
        if hit or word.lower() not in ("she", "he"):
            return hit
        sex = "f" if word.lower() == "she" else "m"
        near = [n for s in sentences(QUOTE.sub("", pre)) for n in self.cast if (self.people.get(n) or {}).get("sex") == sex
                and re.match(rf"\W*(?:[\w-]+-haired\s+)?{re.escape(n)}\b", s)]
        return near[-1] if near else self.resolve(word)

    def subject(self, pre, names):
        acts = sentences(QUOTE.sub("", pre))
        hit = re.match(rf"\W*(?P<n>{names}|she|he)\b", acts[-1], re.I) if acts else None
        return self.name(hit["n"], pre) if hit else None

    def lines(self, text, before=""):
        out, names = [], "|".join(re.escape(n) for n in sorted(self.cast, key=len, reverse=True)) or "(?!)"
        lead, script, after = tags(names)
        for m in QUOTE.finditer(text):
            pre, post = text[:m.start()], text[m.end():]
            if re.search(r"\b(?:writes?|wrote|scrawls?|draws?|paints?|spells?|marks?|carves?|engraves?|types?|reads?|words?|sign)\s*$",
                         pre, re.I) or re.match(r"\s*(?:on|across|onto|into|over)\s+(?:(?-i:[A-Z])\w*['’]s|her|his|their|the)\b", post,
                                                re.I):
                continue
            tag, tail = lead.search(pre) or script.search(pre), after.match(post)
            cont = out and re.fullmatch(r"\s*[,.]?\s*", pre[pre.rfind("”" if "”" in pre else '"') + 1:])
            verb = None
            if tag:
                who, verb = self.name(tag["n"], pre) if tag["n"] else self.subject(pre, names), tag.groupdict().get("v")
            elif tail:
                who, verb = self.name(tail["n"] or tail["n2"], pre), tail["v"] or tail["v2"]
            elif cont:
                who = out[-1]["who"]
            else:
                who = self.subject(pre, names)
                if not who and not pre.strip() and len(self.cast) == 2 and QUOTE.search(before) and \
                        before.rstrip().endswith(("?", "?\"", "?”")):
                    who = next((n for n in self.cast if n != self.last), None)
            who = who or (self.cast[0] if len(self.cast) == 1 else None)
            thought = bool(verb and re.fullmatch(THINK, verb, re.I))
            out.append({"q": m["q"].strip(), "who": who, "thought": thought, "gag": who in self.gagged, "at": m.start(),
                        "verb": verb or ""})
            self.last = who or self.last
        return out

    def noises(self, text):
        bare = QUOTE.sub(lambda m: " " * len(m.group(0)), text)
        out = []
        for m in MUFFLE.finditer(bare):
            if re.search(r"[\"”][,.!?]?\s*(?:(?:[\w-]+\s+){0,2})?$", text[:m.start()]):
                continue
            sent = bare[max(0, bare.rfind(".", 0, m.start()) + 1):m.start()]
            who = next((n for n in reversed(self.cast) if re.search(rf"\b{re.escape(n)}\b", sent)), None) or \
                (self.name("she" if re.search(r"\bshe\b", sent, re.I) else "he", text[:m.start()])
                 if re.search(r"\b(?:she|he)\b", sent, re.I) else next(iter(self.gagged), None))
            if who in self.gagged:
                out.append({"q": "", "who": who, "thought": False, "gag": True, "at": m.start(), "verb": m.group(0)})
        return out


def actions(text):
    return len([c for c in CLAUSE.split(QUOTE.sub(" ", text)) if len(c.split()) >= 2 and not re.fullmatch(
        rf"\W*(?:[\w'’-]+\s+){{0,3}}?(?:{SAYS})\W*", c.strip(), re.I)])


def duration(line):
    if line["gag"]:
        return MUFFLE_SOUND
    parts = [p for p in re.split(r"(?<=[.!?])\s+", line["q"]) if p]
    return len(line["q"].split()) / WORDS_PER_SEC + SAY_SLACK + SENTENCE_PAUSE * max(0, len(parts) - 1)


def timeline(text, said):
    cues, t = [], SPEECH_LEAD
    for ln in sorted(said, key=lambda x: x["at"]):
        if ln["thought"]:
            continue
        start = max(t, SPEECH_LEAD + actions(text[:ln["at"]]) * PER_ACTION)
        end = start + duration(ln)
        cues.append((round(start, 2), round(end, 2), "muffle" if ln["gag"] else "line"))
        t = end + CUE_GAP
    return cues


def talk_time(cues):
    return cues[-1][1] + SPEECH_TAIL if cues else 0.0


CLINK = {"handcuffs": "The handcuffs ratchet shut with sharp metallic clicks", "chains": "the chains clink and rattle",
         "shackles": "the shackles' heavy chain clanks", "leash": "the chain links clink", "collar": "the collar's buckle clicks",
         "chastity belt": "the chastity belt's lock clicks shut", "spreader bar": "the spreader bar's locks click",
         "crotch chain": "the chain clinks", "rope": "the rope creaks as it is pulled tight", "tape": "the tape rips off the roll",
         "zip ties": "the zip ties zip tight", "ball gag": "the gag's strap buckle clicks", "straps": "the straps' buckles click"}


def audible(text):
    return " ".join(x for x in (WHY.sub("", s) for s in sentences(text)) if not INNER.search(x))


def sound(scene, action, cast, gear, muffled, music=False):
    action = audible(action)
    loud = [[n for n in cast if re.search(rf"\b{re.escape(n)}\b", s)] for s in sentences(action) if VOCAL.search(s)]
    quiet = [] if [] in loud else [n for n in cast if n not in {m for m, _ in muffled} and not any(n in x for x in loud)]
    clinks = [CLINK[g] for g in dict.fromkeys(gear) if g in CLINK]
    air = "A light breeze moves through the open air" if OUTSIDE.search(scene) else "The quiet room tone of the place"
    soundscape = [f"{air}. Clothes rustle softly, and every step, touch and movement makes its own sound as it happens."]
    soundscape += [cap(", ".join(clinks)) + "."] if clinks else []
    soundscape += [f"{n} lets out muffled, closed-mouth mmmph sounds through {g}." for n, g in muffled]
    head = " ".join(x for x in (scene, action) if x).strip()
    still = f"{cap(join(quiet))} {'stays' if len(quiet) == 1 else 'stay'} silent." if quiet else ""
    return f"integrated_multimodal_description: [Shot 1] {head} {still}".rstrip() + "\n\noverall_soundscape: " + " ".join(soundscape) + \
        f"\n\nnon_diegetic_music: {'as the scene describes' if music else 'N/A'}"


WHY = re.compile(r"\s*,?\s+(?:due\s+to|because|since|so\s+that|in\s+order\s+to|as\s+if|as\s+though|for\s+fear|to\s+(?:show|reveal|expose|"
                 r"display|humiliate|embarrass|shame|tease|prove)|showing|revealing|exposing)\b[^.!?]*", re.I)
INNER = re.compile(r"\b(?:tries|tried|trying|attempts?|wants?|refuses?|decides?|hopes?|feels?|thinks?|knows?|realizes?|wonders?|seems?|"
                   r"appears?|humiliat\w*|embarrass\w*|ashamed|shame\w*|afraid|scared|frightened|nervous|angry|upset|excited|aroused)\b",
                   re.I)
OUTSIDE = re.compile(r"\b(?:outside|outdoors?|park|parking|street|road|alley|sidewalk|garden|yard|backyard|field|meadow|forest|woods|beach|"
                     r"shore|lake|river|mountains?|hills?|camp\w*|patio|balcony|rooftop|square|market|playground|courtyard|driveway|trail|"
                     r"desert|farm|dock|pier|bridge)\b", re.I)
VOCAL = re.compile(r"\b(?:screams?|shouts?|yells?|cries|moans?|groans?|whimpers?|laughs?|giggles?|sobs?|gasps?|sings?|hums?|sighs?|"
                   r"grunts?|squeals?|shrieks?|pants?|whistles?)\b", re.I)


def tags(names):
    who = rf"(?P<n>{names}|she|he)(?:\s*\([^)]*\))?"
    verb = rf"(?:\w+ly\s+)?(?P<v>{SAYS}|{THINK})\b(?:\s+(?:(?:to|at)\s+(?:the\s+)?[\w-]+|her|him|them|everyone|{names}))?(?:\s+\w+ly)?"
    return (re.compile(rf"(?:^|(?<=[.!?\n\"”])|,|\s(?:and|then))\s*(?:{who}\s+)?{verb}\s*[:,]?\s*$", re.I),
            re.compile(rf"(?:^|(?<=[.!?\n]))\s*{who}\s*:\s*$", re.I),
            re.compile(rf"\s*,?\s*(?:{who}\s+{verb}|(?P<v2>{SAYS})\s+(?P<n2>{names})\b)", re.I))


def voiced(text, said, call, cast, drop=False):
    lead, script, after = tags("|".join(re.escape(call(n)) for n in sorted(cast, key=lambda n: len(call(n)), reverse=True)) or "(?!)")
    left, out, at, last = [ln for ln in said if ln["q"]], "", 0, None
    for m in QUOTE.finditer(text):
        ln = next((x for x in left if x["q"] == m["q"].strip()), None)
        left = [x for x in left if x is not ln]
        spoken = bool(ln and ln["who"] and not ln["thought"])
        if not drop and not spoken:
            continue
        pre, tail = text[at:m.start()], after.match(text, m.end())
        tag = (lead.search(pre) or script.search(pre)) if spoken else None
        if spoken:
            pre = re.sub(r"[\s,:;]+$", "", pre[:tag.start()] if tag else pre)
            pre += "." if pre.strip() and pre[-1] not in ".!?…\"”" else ""
        rest = re.compile(r"[^.!?\"“]*[.!?]?").match(text, tail.end()) if tail else None
        keep = bool(rest and spoken and not drop and rest.group(0).startswith(",") and rest.group(0).strip(", "))
        who = cap(call(ln["who"])) if spoken else ""
        if drop or not spoken:
            line = ""
        elif ln["gag"]:
            line = f"{who} tries to speak through the gag, making only muffled sounds" + ("" if keep else ".")
        else:
            parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", m["q"].strip().rstrip(",")) if p.strip()]
            line = f"{who} {'continues' if who == last and not pre.strip() else 'says'}: " + \
                f" {who} continues: ".join(f"<d>{p}</d>" for p in parts) + ("" if keep or parts and parts[-1][-1:] in ".!?" else ".")
        out += pre + " " + line + " "
        at, last = tail.end() if keep else rest.end() if rest else m.end(), who
    return re.sub(r"\s+([.,!?])", r"\1", re.sub(r"\s+", " ", out + text[at:])).strip()
