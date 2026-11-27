# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79
# SPDX-License-Identifier: GPL-3.0-or-later. See LICENSE.

import re

try:
    from . import bonds, clothes
    from .words import FIXTURE, GARMENT, cut, many, sentences
    from .script import names_in, unquoted
except ImportError:
    import bonds
    import clothes
    from words import FIXTURE, GARMENT, cut, many, sentences
    from script import names_in, unquoted

ON = re.compile(r"\b(?:(?:hand)?cuffs|manacles|shackles|chains|ties|binds|bound|gags|blindfolds|tapes|collars|leashes|straps|"
                r"zip[\s-]?ties|hog[\s-]?ties|locks|padlocks|clips|buckles|fits|fastens|puts|places|slips|wraps|winds|loops|secures|"
                r"restrains|pushes|stuffs|attaches|hooks|cinches)\b", re.I)
OFF = re.compile(r"\b(?:unlocks|unties|unbuckles|uncuffs|unchains|unclips|unhooks|unfastens|unwraps|ungags|frees|releases|removes|"
                 r"(?:takes|pulls|peels|lifts|slides|eases|unwinds)\s+(?!(?:her|him|them|(?-i:[A-Z])[\w'’-]*)\s+(?:off|out)\b)"
                 r"(?:[\w'’-]+\s+){0,3}?(?:off|out(?=\s*(?:[.,;!?]|$)|\s+of\s+(?:(?:her|his|their|(?-i:[A-Z])[\w'’-]*['’]s)\s+)?"
                 r"(?:mouth|eyes)\b))|cuts\s+(?:[\w'’-]+\s+){0,3}?"
                 r"(?:off|through|loose|away)|lets\s+(?:her|him|them)\s+go)\b",
                 re.I)
STATE = re.compile(r"\b(?:is|are|was|sits?|kneels?|lies|stands?|stays?|remains?|waits?)\b[^.;]*\b(?:(?:hand)?cuffed|chained|"
                   r"tied|bound|gagged|blindfolded|taped|collared|shackled|strapped|locked|leashed|hog[\s-]?tied)\b|"
                   r"(?:^|,\s*|\bwith\s+)(?:her|his|their)\s+(?:wrists|hands|ankles|feet|legs|knees|mouth|eyes|neck|collar)\s+(?:\w+\s+)?"
                   r"(?:(?:hand)?cuffed|chained|tied|bound|gagged|taped|shackled|strapped|locked)\b|"
                   r"\b(?:in|wearing|wears|has\s+on)\s+(?:a\s+|an\s+|her\s+|his\s+)?(?:[\w-]+\s+){0,2}?(?:handcuffs|cuffs|shackles|"
                   r"chains|(?:ball[\s-]?)?gag|blindfold|collar|leash|chastity[\s-]+belt|crotch[\s-]+(?:chain|rope)|spreader\s+bar)\b",
                   re.I)
STRIP = re.compile(r"\b(?:takes?|pulls?|slides?|peels?|slips?|yanks?|tugs?|shrugs?|kicks?|steps?)\s+(?:[\w'’-]+\s+){0,4}?"
                   r"(?:off|out\s+of)\b|"
                   r"\b(?:removes?|undresses|strips?)\b", re.I)
TEAR = re.compile(r"\b(?:rips?|tears?|cuts?|shreds?)\s+(?:[\w'’-]+\s+){0,4}?(?:off|open|away|apart)\b|\b(?:rips|tears|cuts)\b", re.I)
WEAR = re.compile(r"\b(?:puts?|pulls?|slips?|gets?)\s+(?:[\w'’-]+\s+){0,4}?(?:back\s+)?on\b|\bdresses\b|\bslips?\s+into\b", re.I)
LIFT = re.compile(r"\b(?:pulls?|lifts?|hikes?|pushes?|yanks?|rolls?|tugs?|drags?)\s+(?:[\w'’-]+\s+){0,4}?(?P<way>up|down|aside|open)\b",
                  re.I)
NAKED = re.compile(r"\b(?:strips?|undresses)\s+(?:\w+\s+)?(?:naked|bare|completely)\b|\bstrips?\s+(?:her|him|them|(?-i:[A-Z])\w+)\s*"
                   r"(?:[.,;]|$)|\b(?:takes?|pulls?)\s+(?:off\s+)?(?:all\s+)?(?:her|his|their)\s+clothes\b", re.I)
POSE = (("kneeling", r"kneel(?:s|ing)?|knelt|(?:on|to|onto)\s+(?:her|his|their)\s+knees|gets?\s+down\s+on\s+(?:her|his|their)\s+knees"),
        ("on all fours", r"on\s+all\s+fours|on\s+(?:her|his|their)\s+hands\s+and\s+knees"),
        ("bent over", r"bend(?:s|ing)?\s+(?:\w+\s+)?over|bent\s+over|leans?\s+over"),
        ("lying", r"lies|lying|lay|lays?\s+(?:down|back)|laid|lie\s+down|(?:throws?|pushes?|shoves?|pins?|lays?|flips?|rolls?)\s+"
                  r"(?:\w+\s+)?(?:down\s+)?(?:on|onto)\s+(?:the\s+)?(?:bed|floor|ground|mattress|couch|sofa|table)|face\s+down"),
        ("sitting", r"sits?|sitting|sat|seats?(?=\s+(?:her|him|them|herself|himself|themselves|(?-i:[A-Z])))|seated|"
                    r"(?:pushes?|shoves?|puts?|sets?|drops?|forces?|lowers?)\s+(?:[\w'’-]+\s+)?(?:down\s+)?"
                    r"(?:in|into|onto|on)\s+(?:a|the|her|his)\s+(?:\w+\s+)?(?:chair|stool|seat|armchair)|"
                    r"(?:gets?|climbs?|slides?|hops?|crawls?)\s+(?:in|into|onto)\s+(?:the\s+)?(?:\w+\s+)?seat"),
        ("crouching", r"crouch(?:es|ing)?|squats?|squatting"),
        ("standing", r"stands?\s+up|gets?\s+(?:up|to\s+(?:her|his|their)\s+feet)|rises|stands|standing|(?:pulls?|hauls?|drags?|lifts?|"
                     r"helps?)\s+(?:her|him|them|(?-i:[A-Z])[\w'’-]+)\s+(?:back\s+)?(?:up|to\s+(?:her|his|their)\s+feet)"))
ORDER = re.compile(r"\b(?:tells?|told|orders?|ordered|asks?|asked|commands?|commanded|instructs?|instructed|wants?|wanted|begs?|"
                   r"begged|urges?|urged|invites?|invited|motions?|motioned|gestures?|gestured|signals?|signall?ed|waves?|waved|"
                   r"beckons?|beckoned)\s+(?:(?:her|him|them|(?-i:[A-Z])[\w'’-]+)\s+)?to\s+(?:\w+ly\s+)?$", re.I)
SHOWING = re.compile(r"\s*,?\s+(?:to\s+(?:show|reveal|expose|display)|showing|revealing|exposing|baring)\b.*$", re.I)
WHERE = re.compile(rf"\b(?:on|onto|in|into|at|over|across|against|beside|by|on\s+top\s+of)\s+(?P<w>(?:the|a|an|her|his|their)\s+"
                   rf"(?:[\w'’-]+\s+){{0,2}}?(?:{FIXTURE}|couch|sofa|mattress|chair|stool|floor|ground|rug|carpet|grass|stairs|"
                   rf"table|desk|counter|sink|bathtub|tub|car|hood|wall|door|seat))\b", re.I)
FACE = re.compile(r"\bface\s+(?P<f>down|up)\b|\bon\s+(?:her|his|their)\s+(?P<s>back|side|front|stomach|belly)\b", re.I)
WRITE = re.compile(r"\b(?:writes?|scrawls?|draws?|paints?|marks?)\s+(?P<q>[\"“][^\"”]+[\"”])\s+(?:on|across|onto|over)\s+"
                   r"(?P<o>(?-i:[A-Z])[\w'’-]*['’]s|her|his|their)\s+(?:\w+\s+)?(?P<part>[a-z]+)|"
                   r"\bbrands?\s+(?P<bo>(?-i:[A-Z])[\w'’-]*['’]s|her|his|their)\s+(?:\w+\s+)?(?P<bp>[a-z]+)", re.I)
WASH = re.compile(r"\b(?:washes|wipes|scrubs|rubs)\s+(?:\w+\s+){0,4}?(?:off|away|clean)\b", re.I)
TOOL = r"(?:(?:his|her|the|a|an|two|three)\s+(?:[\w-]+\s+)?(?:cock|penis|dick|shaft|dildo|strap-?on|vibrator|vibe|plug|toy|fingers?)\s+)?"
SEX = re.compile(rf"\b(?:fucks?|fucking|enters?|penetrates?|pounds?|rides?|riding|(?:slides?|pushes?|thrusts?|eases?|sinks?|works?)\s+"
                 rf"{TOOL}(?:deep\s+)?(?:into|inside)\s+(?:her|him|(?-i:[A-Z])[\w'’-]+)(?!\s+(?:room|car|chair|seat|arms|lap)\b)|"
                 rf"(?:has|have|having)\s+sex|takes?\s+(?:her|him)\s+from\s+behind|sinks?\s+(?:down\s+)?onto|"
                 rf"lowers?\s+(?:herself|himself)\s+onto|bounces?\s+on|fingers?)\b", re.I)
TOY = re.compile(r"\b(?P<t>strap-?on|dildo|vibrator|vibe|(?:butt\s+|anal\s+)?plug|fingers?)\b", re.I)


class Reader:
    def __init__(self, people, cast, held=None, dress=None):
        self.people, self.cast, self.held = people, list(cast), held or {}
        self.dress = dress or (lambda n: clothes.garments((people.get(n) or {}).get("desc", "")))
        self.last, self.other, self.topic, self.cloth, self.part, self.said = None, None, None, None, None, None

    def sex(self, n):
        return (self.people.get(n) or {}).get("sex")

    def resolve(self, word, actor=None):
        sex = "f" if word.lower() in ("she", "her", "herself") else "m" if word.lower() in ("he", "him", "his", "himself") else None
        if word.lower() in ("herself", "himself", "themselves"):
            return actor
        pool = [n for n in self.cast if self.sex(n) == sex and n != actor] if word.lower() not in ("she", "he") else \
            [n for n in self.cast if self.sex(n) == sex]
        pool = pool if sex else [n for n in self.cast if n != actor]
        if len(pool) == 1:
            return pool[0]
        if word.lower() in ("she", "he") and self.last in pool:
            return self.last
        if word.lower() not in ("she", "he") and self.other in pool:
            return self.other
        return None

    def actor(self, s):
        m = re.match(rf"\W*(?:then|and|but|so|now|finally|slowly|suddenly|meanwhile)?\W*(?:(?:[\w-]+-haired)\s+)?(?P<n>"
                     rf"{'|'.join(map(re.escape, sorted(self.cast, key=len, reverse=True))) or '(?!)'}|she|he)\b(?!['’])", s, re.I)
        if not m:
            return None
        n = m["n"]
        return next((c for c in self.cast if c.lower() == n.lower()), None) or self.resolve(n)

    def target(self, s, actor, after=0):
        rest = s[after:]
        own = re.search(rf"(?<![\w'’-])({'|'.join(map(re.escape, sorted(self.cast, key=len, reverse=True))) or '(?!)'})['’]s?\b", rest)
        if own:
            return own.group(1)
        named = [n for n in names_in(rest, {c: 1 for c in self.cast}) if n != actor]
        if named:
            return named[0]
        pro = re.search(r"\b(her|him|his|them|their|herself|himself|themselves)\b", rest, re.I)
        if pro:
            hit = self.resolve(pro.group(1), actor)
            return hit
        return None

    def read(self, text, fresh=True):
        events, missed, self.said = [], [], None if fresh else self.said
        for s in sentences(text):
            for c in re.split(r";|,?\s+(?:and\s+)?then\s+|\.\s+", unquoted(s)):
                if c.strip():
                    self.clause(c.strip(), s, events, missed)
        return events, missed

    def clause(self, c, s, events, missed):
        actor = self.actor(c) or (self.last if re.match(r"\W*(?:and\s+)?(?:then\s+)?[a-z]+s\b", c) else None)
        add = lambda kind, who, items, doer=actor: events.append({"kind": kind, "who": who, "items": items, "doer": doer, "sent": s}) or \
            setattr(self, "topic", next((i["gear"] for i in items if isinstance(i, dict) and i.get("gear")), self.topic))
        ring = re.search(r"\bits\s+(?:front\s+)?ring\b", c, re.I)
        wearer = [n for n in self.cast if any(i["gear"] == "collar" for i in self.held.get(n, []))] + \
            [e["who"] for e in events if any(i.get("gear") == "collar" for i in e["items"] if isinstance(i, dict))]
        ring = ring or (re.search(r"\bit\b(?=\s+(?:to|onto)\b)", c) if self.topic == "collar" else None)
        if ring and wearer:
            c = c[:ring.start()] + f"{wearer[-1]}'s collar" + c[ring.end():]
        c = self.refer(c)
        noun = min((m for rx in bonds.NOUN.values() for m in [rx.search(c)] if m), key=lambda m: m.start(), default=None)
        it = re.search(r"\bit\b", c) if not noun and self.said and ON.search(c) and self.target(c, self.actor(c)) else None
        c = c[:it.start()] + f"the {self.said}" + c[it.end():] if it else c
        kind = next((k for k, rx in bonds.NOUN.items() if noun and (m := rx.search(c)) and m.start() == noun.start()), None)
        self.said = f"{bonds.look(c, kind)} {noun.group(0)}".strip() if noun else self.said
        them = re.search(r"\bthem\b", c, re.I)
        if them and self.part and not bonds.spots(c) and any(rx.search(c) for rx in list(bonds.NOUN.values()) + list(bonds.VERB.values())):
            c = c[:them.start()] + self.part + c[them.end():]
        names = '|'.join(map(re.escape, self.cast)) or '(?!)'
        part = re.search(rf"(?P<o>{names}['’]s|\bher|\bhis)\s+(?P<p>wrists|hands|arms|ankles|feet|legs)\b"
                         rf"(?P<w>\s+(?:behind\s+(?:her|his|their)\s+back|in\s+front\s+of\s+(?:her|him|them)))?", c, re.I)
        owner = (part["o"][:-2] if part["o"][-2:] in ("'s", "’s") else self.resolve(part["o"], actor)) if part else None
        self.part = f"{owner}'s {part['p']}{part['w'] or ''}" if owner else self.part
        gear = bonds.parse(c) if ON.search(c) or any(rx.search(c) for rx in list(bonds.NOUN.values()) + list(bonds.VERB.values())) else []
        gear = [g for g in gear if not (ring and wearer and g["gear"] == "collar")]
        first = min((m for rx in bonds.NOUN.values() for m in [rx.search(c)] if m), key=lambda m: m.start(), default=None)
        if gear and first and not OFF.search(c) and all(any(bonds.same(g, x) for xs in self.held.values() for x in xs) for g in gear) \
                and re.search(r"\b(?:her|his|their|its|the|(?-i:[A-Z])[\w'’-]*['’]s)\s+(?:[\w-]+\s+){0,2}$", c[:first.start()], re.I):
            gear = []
        verb = ON.search(c) or next((m for rx in bonds.VERB.values() for m in [rx.search(c)] if m), None)
        state = STATE.search(c)
        loose = re.search(rf"\b(?:unties|frees|releases|unchains|uncuffs|unbinds)\s+(?:{'|'.join(map(re.escape, self.cast)) or '(?!)'}|"
                          rf"her|him|them)\b(?!['’])(?!\s+(?:wrists|ankles|hands|feet|legs|arms|mouth))", c, re.I)
        if loose and not gear:
            gear = [{"gear": "handcuffs", "spot": None, "pose": None, "to": None}] if "uncuffs" in loose.group(0).lower() else ["*"]
        if OFF.search(c) and (gear or re.search(r"\b(?:wrists|ankles|hands|feet|mouth|eyes|neck)\b|\buncuffs\b", c, re.I)) \
                and not GARMENT.search(c):
            who = self.target(c, actor, OFF.search(c).start()) or (actor if re.search(r"\b(?:herself|himself|her\s+own|his\s+own)\b",
                                                                                         c, re.I) else None)
            led = any(isinstance(x, dict) and x["gear"] == "leash" for x in gear) and re.search(
                r"\b(?:from|off)\s+(?:of\s+)?(?:her|his|their|the|(?-i:[A-Z])[\w'’-]*['’]s)\s+(?:[\w-]+\s+)?collar", c, re.I)
            gear = [g for g in gear if not (led and isinstance(g, dict) and g["gear"] == "collar")]
            spec = gear or [{"gear": None, "spot": s_, "pose": None, "to": None} for s_ in bonds.spots(c)] or \
                [{"gear": "handcuffs", "spot": "wrists", "pose": None, "to": None}]
            if who:
                add("release", who, spec)
            else:
                missed.append((s, "could not tell whose restraint comes off; name the person, as in \"release: Sarah, handcuffs\""))
            self.note(actor, who)
        elif gear and state and not (ON.search(c) and actor and ON.search(c).start() < state.start() and
                                     self.target(c, actor, ON.search(c).end()) not in (None, actor)):
            who = next((n for n in names_in(c[:state.end()], {x: 1 for x in self.cast})), None) or actor or next(
                (self.resolve(m.group(0)) for m in [re.search(r"\b(?:her|his|she|he)\b", c, re.I)] if m), None)
            if who:
                add("have", who, gear, None)
                self.note(who, None)
            else:
                missed.append((s, f"could not tell who has the {bonds.name(gear[0])}; name the person, as in \"hold: Sarah, "
                                  f"{bonds.name(gear[0])}\""))
        elif gear and verb:
            who = actor if re.search(r"\b(?:herself|himself|themselves)\b", c, re.I) else \
                self.target(c, actor, verb.end() if ON.search(c) else 0) or self.target(c, actor)
            if who and (who != actor or re.search(r"\b(?:herself|himself|themselves|her\s+own|his\s+own)\b", c, re.I)):
                add("hold", who, gear)
                self.note(actor, who)
            elif not re.search(r"\b(?:picks?|grabs?|holds?|takes?|brings?|shows?|dangles?|drops?|throws?|tosses?|puts?\s+(?:\w+\s+)?"
                               r"(?:away|down|back)|sets?\s+(?:\w+\s+)?down|hands?)\b", c, re.I):
                many_ = gear[0]["gear"] in ("handcuffs", "chains")
                missed.append((s, f"could not tell who the {bonds.name(gear[0])} {'go' if many_ else 'goes'}"
                                  f" on; name the person, as in \"hold: Sarah, {bonds.name(gear[0])}\""))
        elif GARMENT.search(c) or NAKED.search(c) or re.search(r"\b(?:clothes|clothing|naked|everything)\b", c, re.I):
            self.garments(c, s, actor, add, missed)
        w = WRITE.search(s) if re.search(r"\b(?:writes?|scrawls?|draws?|paints?|marks?|brands?)\b", c, re.I) else None
        if w and not any(e["kind"] == "mark" and e["sent"] == s for e in events):
            who = self.owner(w["o"] or w["bo"], actor)
            text = f'the word {w["q"]} written across {"her" if self.sex(who) == "f" else "his"} {w["part"]}' if w["q"] else \
                f"a brand on {'her' if self.sex(who) == 'f' else 'his'} {w['bp']}"
            if who:
                add("mark", who, [text])
            else:
                missed.append((s, "could not tell whose skin it goes on; name the person, as in \"mark: Sarah, a brand on her hip\""))
        if WASH.search(c) and re.search(r"\b(?:writing|word|ink|marker|words)\b", c, re.I):
            who = self.target(c, actor, WASH.search(c).start()) or actor
            add("unmark", who, ["writing"]) if who else None
        if SEX.search(c) and not re.search(r"\b(?:rides?|riding)\s+(?:a|the|her|his)\s+(?:bike|horse|car|bus|train|elevator)\b", c, re.I):
            self.penetration(c, s, actor, add)
        self.posture(c, s, actor, add)
        self.note(actor, None)

    def mention(self, c, end):
        last = list(GARMENT.finditer(c, 0, end))
        if not last:
            return None
        names = "|".join(map(re.escape, self.cast)) or "(?!)"
        own = re.search(rf"(?:\b(?:{names})['’]s|\b(?:her|his|their))\s+(?:[\w-]+\s+){{0,3}}$", c[:last[-1].start()], re.I)
        return c[own.start() if own else last[-1].start():last[-1].end()]

    def refer(self, c):
        gear = [m.start() for rx in bonds.NOUN.values() for m in rx.finditer(c)]
        verbs = [m for rx in (STRIP, TEAR, LIFT, WEAR) for m in rx.finditer(c)]
        pro = next((p for p in re.finditer(r"\b(?:it|them)\b", c, re.I) if any(m.start() <= p.start() <= m.end() + 1 for m in verbs)),
                   None)
        if pro:
            last = [g.end() for g in GARMENT.finditer(c, 0, pro.start())]
            thing = None if any((last or [0])[-1] <= x < pro.start() for x in gear) else \
                self.mention(c, pro.start()) if last else self.cloth
            if thing and many(GARMENT.findall(thing)[-1]) == (pro.group(0).lower() == "them"):
                c = c[:pro.start()] + thing + c[pro.end():]
        other = any(not GARMENT.search(m.group(1)) for m in re.finditer(r"\b(?:her|his|their|its|the|a|an)\s+((?:[\w-]+\s+){0,2}[\w-]+)",
                                                                         c, re.I))
        self.cloth = self.mention(c, len(c)) or (None if gear or other else self.cloth)
        return c

    def owner(self, w, actor):
        m = re.match(r"(.+?)['’]s$", w.strip())
        return next((n for n in self.cast if m and n.lower() == m.group(1).lower()), None) if m else self.resolve(w, actor)

    def note(self, actor, target):
        self.last = actor or self.last
        self.other = target if target and target != self.last else self.other

    def garments(self, c, s, actor, add, missed):
        c = SHOWING.sub("", c)
        kind = "tear" if TEAR.search(c) else "remove" if STRIP.search(c) or NAKED.search(c) else "lift" if LIFT.search(c) else \
            "wear" if WEAR.search(c) else None
        if not kind:
            return
        at = (TEAR.search(c) or STRIP.search(c) or NAKED.search(c) or LIFT.search(c) or WEAR.search(c)).start()
        rest = c[at:]
        first, cast = GARMENT.search(rest), "|".join(map(re.escape, sorted(self.cast, key=len, reverse=True))) or "(?!)"
        named = next((m for m in re.finditer(rf"(?<![\w'’-])({cast})(?:['’]s)?\b", rest)
                      if not first or m.start() < first.start() or re.search(r"\b(?:off|from|of)\s+$", rest[:m.start()])), None)
        words = [clothes.plain(m.group(0)) for m in GARMENT.finditer(rest if kind != "lift" else c)]
        if named and (named.group(1) != actor or named.group(0) != named.group(1)):
            who = named.group(1)
        elif re.search(r"\b(?:her|his|their)\s+own\b|\b(?:herself|himself)\b", c, re.I) or kind == "wear" or not re.search(
                r"\b(?:her|his|him)\b", rest, re.I):
            who = actor
        else:
            sex = "f" if re.search(r"\bher\b", rest, re.I) else "m"
            own = actor and self.sex(actor) == sex and any(clothes.fit(x, self.dress(actor)) for x in words or ["clothes"])
            others = [n for n in self.cast if n != actor and self.sex(n) == sex and any(clothes.fit(x, self.dress(n))
                                                                                          for x in words or ["clothes"])]
            who = actor if own and not (others and re.search(r"\b(?:him|her)\b(?!\s+(?:[\w-]+\s+)?(?:" + GARMENT.pattern[2:-2] + "))",
                                                             rest, re.I)) else others[0] if len(others) == 1 else actor if own else None
        if not who:
            missed.append((s, "could not tell whose clothes these are; name the person, as in \"remove: Sarah, blouse\""))
            return
        if NAKED.search(c) and kind == "remove":
            add("remove", who, ["*"])
            self.note(actor, who if who != actor else None)
            return
        lift = LIFT.search(c) if kind == "lift" else None
        to = lift and re.match(r"\s+(?:to|around|past)\s+(?:her|his|their)\s+(?:ankles|knees|thighs|hips|waist|feet|calves|chest|neck|"
                               r"shoulders)\b", c[lift.end():], re.I)
        way = f"{lift['way']} {' '.join(to.group(0).split())}" if to else lift["way"] if lift else None
        if words:
            add(kind, who, [(w, way) if kind == "lift" else w for w in words])
            self.note(actor, who if who != actor else None)

    def penetration(self, c, s, actor, add):
        target = self.target(c, actor, SEX.search(c).start())
        tool = TOY.search(c)
        tool = (tool["t"].lower().replace("vibe", "vibrator").replace("finger", "fingers").replace("fingerss", "fingers")
                if tool else "penis")
        rides = re.search(r"\b(?:rides?|riding|sinks?\s+(?:down\s+)?onto|lowers?\s+(?:herself|himself)\s+onto|bounces?\s+on)\b", c, re.I)
        doer, into = (target, actor) if rides and tool == "penis" else (actor, target if target != actor else actor)
        if not doer or not into:
            return
        hole = "anus" if re.search(r"\b(?:ass|anus|anal|butt)\b", c, re.I) or self.sex(into) == "m" else "vagina"
        way = "behind" if re.search(r"\b(?:from\s+behind|doggy|bent\s+over|on\s+all\s+fours)\b", c, re.I) else \
            "astride" if rides else "top" if re.search(r"\b(?:missionary|on\s+top\s+of\s+her|between\s+her\s+legs)\b", c, re.I) else \
            "side" if re.search(r"\bspoon", c, re.I) else \
            "standing" if re.search(r"\bagainst\s+the\s+wall\b|\bstanding\b", c, re.I) else None
        add("sex", into, [{"by": doer, "tool": tool, "hole": hole, "way": way}], doer)

    def posture(self, c, s, actor, add):
        for name_, rx in POSE:
            m = next((m for m in re.finditer(rf"\b(?:{rx})\b", c, re.I) if not ORDER.search(c[:m.start()])), None)
            if not m:
                continue
            forced = re.search(r"\b(?:throws?|pushes?|shoves?|pins?|lays?|forces?|bends?|puts?|sets?|sits?|drags?|pulls?|hauls?|lifts?|"
                               r"lowers?|flips?|rolls?|knocks?)\s+(?P<o>(?-i:[A-Z])[\w'’-]+|her|him|them)\b", c, re.I)
            names = "|".join(map(re.escape, self.cast)) or "(?!)"
            own = re.search(rf"\b(?P<o>{names}|she|he)\s+(?:\w+ly\s+)?$", c[:m.start()], re.I)
            hit = forced if forced and forced.start() <= m.start() + 3 else own
            who = (next((n for n in self.cast if n.lower() == hit["o"].lower()), None) or self.resolve(hit["o"], actor)) if hit else actor
            if not who:
                return
            where = WHERE.search(c[m.start():]) or WHERE.search(c)
            face = FACE.search(c)
            way = ("face down" if face and (face["f"] == "down" or face["s"] in ("front", "stomach", "belly")) else
                   "on her back" if face and (face["s"] == "back" or face["f"] == "up") else "on her side" if face and face["s"] == "side"
                   else "") if name_ == "lying" else ""
            add("pose", who, [{"pose": name_, "where": where["w"].lower() if where and name_ != "standing" else "", "way": way}],
                actor if who != actor else None)
            return


def quote(e):
    return f"{e['kind']}: {e['who']} (\"{cut(e['sent'], 60)}\")"
