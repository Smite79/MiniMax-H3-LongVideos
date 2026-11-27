# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79
# SPDX-License-Identifier: GPL-3.0-or-later. See LICENSE.

import re

try:
    from . import bonds, car, clothes, voice, writer
    from .words import FPS, GARMENT, MIN_FRAMES, PIC, SKIRT, UNDER, align, cap, covers, cut, has, join, many, pron, strip_pics, \
        zone
except ImportError:
    import bonds
    import car
    import clothes
    import voice
    import writer
    from words import FPS, GARMENT, MIN_FRAMES, PIC, SKIRT, UNDER, align, cap, covers, cut, has, join, many, pron, strip_pics, \
        zone

BEAT_BASE = 0.8
SETTLE = 1.5
RUSH = 1.15
FETCH = re.compile(r"\b(?:takes?|pulls?|gets?|grabs?|fishes|digs|produces?|picks?)\s+(?:up\s+|out\s+)?(?:[\w'’-]+\s+){0,6}?"
                   r"(?:out\s+of|from)\s+(?:the|a|an|his|her|their)\s+(?:[\w-]+\s+)?(?:bag|duffle|duffel|pocket|drawer|box|case|"
                   r"backpack|briefcase|trunk|cabinet|closet|cupboard|shelf|table|desk|nightstand)\b|"
                   r"\b(?:takes?|pulls?)\s+out\s+(?:a|an|the|some|his|her)\b", re.I)
FETCH_SECS = 3
GARB_SECS = (2, 4, 3, 7, 6, 6, 4, 4, 6, 5, 4, 3, 4)
MOVES = r"walks?|runs?|steps?|moves?|crawls?|climbs?|gets?\s+up|stands?\s+up|turns?|leaves?|goes|comes|follows?|paces?|kneels?|" \
        r"sits?|lies|stands?|rises|kicks?|jumps?|rolls?|leans?|bends?|lowers?"
IDLE = r"watch(?:es|ing)?|looks?|stares?|smiles?|grins?|laughs?|waits?|stands?|steps?\s+back|says?|nods?|admires?|smirks?"
STRANGER = re.compile(r"\b(?:someone|somebody|anyone|a\s+crowd|people)\b|\b(?:a|an|the|another|two|three|some|other)\s+(?:[\w-]+\s+)?"
                      r"(?:man|woman|men|women|guy|guys|girl|boy|stranger|guard|guards|officer|nurse|doctor|waiter|waitress|driver|"
                      r"customer|neighbou?r|friend|figure|person)\b", re.I)
CAMERA = re.compile(r"\b(?:camera|close[\s-]?up|zooms?|pans?|wide\s+shot)\b", re.I)
MUSIC = re.compile(r"\b(?:music|song|sings?|singing|radio|melody|tune)\b", re.I)
KEEP = re.compile(rf"\b(?:throws?|tosses?|flings?|hands?|gives?|puts?|stuffs?|shoves?|keeps?|holds?|pockets?|folds?|hangs?|drops?|"
                  rf"carries|wads?|lays?|sets?)\s+(?:[\w'’-]+\s+){{0,3}}?(?:it|them|{GARMENT.pattern[2:-2]})\b", re.I)


class Shot:
    def __init__(self, world, people, cast, i, warn):
        self.w, self.people, self.cast, self.i, self.warn, self.warn_after = world, people, cast, i, warn, []
        self.was = {n: list(world.held.get(n, [])) for n in cast}
        self.off0 = {n: list(world.off.get(n, [])) for n in cast}
        self.worn0 = {n: list(world.worn.get(n, [])) for n in cast}
        self.lifted0 = {n: list(world.lifted.get(n, [])) for n in cast}
        self.skin0 = {n: list(world.skin.get(n, [])) for n in cast}
        self.wear0 = {n: world.wearing(n) for n in cast}
        self.added, self.gone, self.stripped, self.raised, self.dressed, self.inked, self.wiped = ({} for _ in range(7))
        self.posed, self.sex, self.moves, self.read, self.sided, self.voice_last = {}, [], [], [], [], None
        self.rode = {n: world.pose[n]["pose"] for n in cast if (world.pose.get(n) or {}).get("pose") in writer.RIDE}
        self.pose0 = {n: world.pose.get(n) for n in cast}

    def note(self, msg):
        self.warn.append(f"shot {self.i + 1}: {msg}")

    def apply(self, e):
        n, d, kind = e["who"], e["doer"], e["kind"]
        self.read.append((kind, e))
        w = self.w
        if kind in ("hold", "have"):
            if any(i["gear"] == "chastity belt" for i in e["items"]) and not any(i["gear"] == "chastity belt" for i in w.held.get(n, [])):
                under = w.bare_under(n, kind == "hold")
                if kind == "have":
                    w.off[n] = w.off.get(n, []) + under
                    self.off0[n] = self.off0.get(n, []) + under
                    self.wear0[n] = w.wearing(n)
                for g in under if kind == "hold" else ():
                    self.strip(n, g, d, False)
            if kind == "have":
                self.was[n] = bonds.put(self.was.get(n, []), e["items"])
            for it in e["items"]:
                it = dict(it, by=d) if it["gear"] == "leash" and kind == "hold" and not it["to"] and not it["by"] and d and d != n else it
                had = next((x for x in w.held.get(n, []) if bonds.same(x, it)), None)
                w.held[n] = bonds.put(w.held.get(n, []), [it])
                new = next(x for x in w.held[n] if bonds.same(x, it))
                if had is not None and had in self.added.get(n, []):
                    self.added[n] = [new if x is had else x for x in self.added[n]]
                    self.moves = [(k, m, new, dd, bonds.secs(new)) if t is had else (k, m, t, dd, sec) for k, m, t, dd, sec in self.moves]
                elif had is not None and kind == "hold" and bonds.linked(new) and not bonds.linked(had):
                    self.moves.append(("link", n, new, d, 4))
                if had is None and kind == "hold":
                    self.added.setdefault(n, []).append(it)
                    self.moves.append(("hold", n, it, d, bonds.secs(it)))
                    self.sided += [(d, n)] if it["to"] and not bonds.linked(it) and d and d != n and (d, n) not in self.sided else []
        elif kind == "release":
            mine = w.held.get(n, [])
            specs = [s for s in e["items"] if isinstance(s, dict)]
            score = lambda s, x: bonds.fits(s, x, mine)
            pick = mine if "*" in e["items"] else [x for x in mine if any(0 < score(s, x) == max(score(s, y) for y in mine)
                                                                          for s in specs)]
            if not pick:
                self.note(f"\"{cut(e['sent'])}\": {n} has nothing like that on, so nothing came off")
            for it in pick:
                w.held[n] = [x for x in w.held[n] if x is not it]
                self.gone.setdefault(n, []).append(it)
                self.moves.append(("release", n, it, d, bonds.secs(it, False)))
            if not any(x["spot"] == "wrists" for x in w.held.get(n, [])):
                w.held[n] = [dict(x, to=None) if bonds.linked(x) else x for x in w.held.get(n, [])]
        elif kind in ("remove", "tear"):
            worn = w.wearing(n)
            for word in e["items"]:
                hits = worn if word == "*" else clothes.fit(word, worn)
                if not hits:
                    gone = clothes.fit(word, self.off0.get(n, []) + clothes.garments((self.people.get(n) or {}).get("desc", "")))
                    belted = any(i["gear"] == "chastity belt" for i in w.held.get(n, []))
                    self.note(f"{n}'s {word} {'are' if many(word) else 'is'} already off" + (
                        f": nothing is worn under {pron(self.people.get(n))} chastity belt" if belted and has(UNDER, word) else "")
                              if gone else f"{kind}: '{word}' is not in {n}'s description in character_memory, so nothing came off; "
                                           f"use the word the description uses")
                for g in sorted(hits, key=clothes.rank):
                    self.strip(n, g, d, kind == "tear")
        elif kind == "lift":
            for word, way in e["items"]:
                hit = clothes.fit(word, w.wearing(n))
                if not hit:
                    self.note(f"lift: '{word}' is not in {n}'s description, so nothing was pulled")
                    continue
                g, p = hit[0], pron(self.people.get(n))
                where = self.place(g, way, p)
                w.lifted[n] = [x for x in w.lifted.get(n, []) if x[0] != g] + [(g, where)]
                self.raised.setdefault(n, []).append((g, where))
                self.moves.append(("lift", n, (g, where), d, 2))
        elif kind == "wear":
            belted = any(i["gear"] == "chastity belt" for i in w.held.get(n, []))
            for word in e["items"]:
                if belted and has(UNDER + "|tights|pantyhose", word) and zone(word) <= {"lower"}:
                    self.note(f"{n}'s {(clothes.fit(word, w.off.get(n, [])) or [word])[0]} cannot go on while "
                              f"{pron(self.people.get(n))} chastity belt is locked on: nothing is worn "
                              f"under it or with it; release: the belt in an earlier beat")
                    continue
                up = [g for g, _ in w.lifted.get(n, []) if clothes.fit(word, [g])]
                back = clothes.fit(word, w.off.get(n, []))
                if up:
                    w.lifted[n] = [x for x in w.lifted[n] if x[0] not in up]
                    g = up[0]
                elif back:
                    w.off[n] = [x for x in w.off[n] if x not in back]
                    g = back[0]
                else:
                    g = clothes.plain(word)
                    w.worn[n] = w.worn.get(n, []) + [g]
                self.dressed.setdefault(n, []).append((g, bool(up or back)))
                self.moves.append(("wear", n, (g, bool(up or back)), d, 4))
        elif kind == "mark":
            w.skin[n] = w.skin.get(n, []) + e["items"]
            self.inked.setdefault(n, []).extend(e["items"])
        elif kind == "unmark":
            gone = [x for x in w.skin.get(n, []) if not e["items"] or any(word.lower() in x.lower() or (
                word.lower() in ("writing", "words", "word") and "written" in x) for word in e["items"])]
            w.skin[n] = [x for x in w.skin.get(n, []) if x not in gone]
            self.wiped.setdefault(n, []).extend(gone)
        elif kind == "pose":
            w.pose[n] = e["items"][0]
            self.posed[n] = e["items"][0]
            self.rode[n] = self.rode.get(n) or (e["items"][0]["pose"] if e["items"][0]["pose"] in writer.RIDE else None)
        elif kind == "sex":
            self.sex.append(e)

    def place(self, g, way, p):
        if way and way not in ("up", "down", "aside", "open"):
            return way
        if "upper" in zone(g) and not has(SKIRT, g) and zone(g) != {"upper", "lower"}:
            return "open and down off her shoulders".replace("her", p) if way == "open" else "up above her breasts".replace("her", p)
        if zone(g) == {"upper", "lower"} or has(SKIRT, g) and way != "down":
            return f"up to {p} waist" if way != "down" else f"down to {p} waist"
        return f"down to {p} knees" if way != "up" else "up into place"

    def strip(self, n, g, d, torn):
        w, p = self.w, pron(self.people.get(n))
        held = w.held.get(n, [])
        wrists = any(i["spot"] in ("wrists", "link") for i in held)
        bottoms = r"pants|trousers|jeans|leggings|shorts|tights|pantyhose|panties|knickers|underwear|briefs|boxers|(?:mini)?skirts?"
        legs = next((i for i in held if i["spot"] in ("ankles", "knees") and has(bottoms, g)), None)
        if not torn and "upper" in zone(g) and wrists and zone(g) != {"lower"}:
            where = self.place(g, "open" if re.search(r"jacket|blazer|coat|cardigan|robe|shirt|blouse", g) else "up", p)
            self.note(f"{n}'s {g} cannot come off past {p} bound wrists, so {'they are' if many(g) else 'it is'} pulled {where} instead; "
                      f"release: the restraint in an earlier beat, or tear: it off")
            w.lifted[n] = [x for x in w.lifted.get(n, []) if x[0] != g] + [(g, where)]
            self.raised.setdefault(n, []).append((g, where))
            self.moves.append(("lift", n, (g, where), d, 2))
            return
        if not torn and legs:
            where = f"down to just above the {bonds.name(legs)} on {p} {legs['spot']}"
            self.note(f"{n}'s {g} cannot come off past {p} bound {legs['spot']}, so {'they are' if many(g) else 'it is'} pulled {where}")
            w.lifted[n] = [x for x in w.lifted.get(n, []) if x[0] != g] + [(g, where)]
            self.raised.setdefault(n, []).append((g, where))
            self.moves.append(("lift", n, (g, where), d, 3))
            return
        down = any(x == g and wh.startswith("down") for x, wh in w.lifted.get(n, []))
        w.off[n] = w.off.get(n, []) + [g]
        w.lifted[n] = [x for x in w.lifted.get(n, []) if x[0] != g]
        self.raised[n] = [x for x in self.raised.get(n, []) if x[0] != g]
        self.raised = {k: v for k, v in self.raised.items() if v}
        self.stripped.setdefault(n, []).append((g, torn))
        self.moves.append(("tear" if torn else "remove", n, (g, down), d, GARB_SECS[min(clothes.rank(g), len(GARB_SECS) - 1)]))

    def call_map(self, refs, owner):
        shown = {owner[k] for k in refs if k in owner}
        names = writer.labels(self.cast, self.people, shown)
        return lambda n: names.get(n, n) if n else "someone"

    def over(self, n):
        return *clothes.surface(self.w.wearing(n), self.w.lifted.get(n, [])), any(
            i["spot"] == "wrists" and i["pose"] == "behind" and not i["to"] for i in self.w.held.get(n, []))

    def steps(self, call):
        out, run, secs = [], [], 0
        for kind, n, thing, d, s in self.moves:
            secs += s
            if not d:
                continue
            per = self.people.get(n, {})
            p, o = pron(per), pron(per, 2)
            first = not any(x[1] == n and x[3] == d for x in run)
            x = p if d == n else (f"{call(n)}'s" if first else p)
            who = None if d == n else call(n) if first else o
            bound = bonds.behind(self.w.held.get(n, [])) is not None
            if kind in ("hold", "release"):
                text = bonds.step(thing, x, p, o, who, kind == "hold", self.over(n), bound)
                text += f", and drops {'them' if many(bonds.name(thing)) else 'it'} to the {self.floor}" \
                    if kind == "release" and thing["gear"] not in ("leash", "collar") else ""
            elif kind == "link":
                text = bonds.link_step(thing, who or o, p)
            elif kind in ("remove", "tear"):
                drop = None if self.keep else self.floor
                text = clothes.strip_step(*thing, x, p, pron(per, 0), call(d) if d != n else "", drop) if kind == "remove" else \
                    f"rips {x} {thing[0]} off" + (f", and drops {'them' if many(thing[0]) else 'it'} to the {drop}" if drop else "")
            elif kind == "lift":
                text = f"pulls {x} {thing[0]} {thing[1]}"
            else:
                text = f"puts {x} {thing[0]} {'back ' if thing[1] else ''}on"
            run.append((kind, n, thing, d, text))
        for k, (kind, n, thing, d, text) in enumerate(run):
            if k and run[k - 1][3] == d:
                out[-1] += f", then {text}"
            else:
                out.append(f"{cap(call(d))} {text}")
        return [x + "." for x in out], secs

    def ends(self, call):
        out = []
        for n, items in self.added.items():
            per = self.people.get(n, {})
            p, o = pron(per), pron(per, 2)
            keeps = {i["by"] for i in self.w.held.get(n, []) if i["gear"] == "leash"}
            by = list(dict.fromkeys(d for k, m, t, d, _ in self.moves if k == "hold" and m == n and d and d != n and d not in keeps))
            hidden = [i for i in items if i["pose"] == "behind" or i["gear"] == "chastity belt" and covered(self, n)
                      or i["gear"] in bonds.HARNESS and self.over(n)[1]]
            shown = [i for i in items if i not in hidden]
            parts = [f"{join([bonds.describe(i, p, o, call, self.over(n)) for i in shown])} in plain view" if shown else "",
                     join([bonds.describe(i, p, o, call, self.over(n)) for i in hidden])]
            out.append(f"By the last frame, {call(n)} has {join([x for x in parts if x])}" +
                       (f", and {join([call(d) for d in by])} {'have' if len(by) > 1 else 'has'} let go" if by else "") + ".")
        for n, items in self.gone.items():
            out.append(f"By the last frame, {call(n)} no longer has {pron(self.people.get(n, {}))} "
                       f"{join([bonds.name(i) for i in items])} on.")
        for n, items in self.stripped.items():
            per = self.people.get(n, {})
            left = [g for g in self.w.wearing(n) if g not in clothes.hidden(self.w.wearing(n), [], self.w.lifted.get(n, []))]
            cover = set().union(*[covers(g) for g in left if g not in {x for x, _ in self.w.lifted.get(n, [])}]) if left else set()
            bare = [z for z in ("upper", "lower") if z not in cover]
            undies = list(dict.fromkeys(g for z in bare for g in left if has(UNDER, g) and z in zone(g)))
            naked = [z for z in bare if not any(has(UNDER, g) and z in zone(g) for g in left)]
            state = ("completely naked" if len(naked) == 2 else f"naked from the waist {'up' if naked == ['upper'] else 'down'}"
                     if naked else "") + (", " if naked and undies else "") + (
                f"in only {pron(per)} {join(undies)}" + (f" {'above' if bare == ['upper'] else 'below'} the waist" if len(bare) == 1 else "")
                if undies else "")
            body = [bonds.describe(i, pron(per), pron(per, 2), call, self.over(n)) for i in self.w.held.get(n, [])
                    if i["spot"] in ("crotch", "waist")]
            gs = [g for g, _ in items]
            lie = "" if self.keep else f"{join(gs)} {'stay' if len(gs) > 1 or many(gs[0]) else 'stays'} on the {self.floor} where " \
                                       f"{'they' if len(gs) > 1 or many(gs[0]) else 'it'} fell, off {pron(per, 2)}"
            if state:
                out.append(f"Once {'they are' if len(gs) > 1 or many(gs[0]) else 'it is'} off, {call(n)} is {state} for the rest of the "
                           f"shot" + (f", apart from {join(body)}" if body else "") + (f", and {pron(per)} {lie}" if lie else "") + ".")
            elif lie:
                out.append(f"{cap(call(n))}'s {lie}.")
            belt = any(i["gear"] == "chastity belt" for i in self.w.held.get(n, []) if i in self.was.get(n, []))
            out += [f"{cap(call(n))}'s chastity belt stays locked on {pron(per, 2)} the whole time {pron(per)} {join(gs)} "
                    f"{'come' if len(gs) > 1 or many(gs[0]) else 'comes'} off."] if belt else []
        for n, items in self.raised.items():
            out += [f"By the last frame, {call(n)}'s {g} {'are' if many(g) else 'is'} pulled {where} and {'stay' if many(g) else 'stays'} "
                    f"there, still on {pron(self.people.get(n), 2)}." for g, where in items]
            bare = [g for g in clothes.hidden(self.wear0[n], self.off0[n], self.lifted0[n])
                    if g not in clothes.hidden(self.w.wearing(n), [], self.w.lifted.get(n, [])) and g in self.w.wearing(n)]
            way = items[-1][1].split()[0]
            out += [f"Once {'they are' if len(items) > 1 or many(items[0][0]) else 'it is'} {way}, {pron(self.people.get(n))} {join(bare)} "
                    f"{'show' if len(bare) > 1 or many(bare[0]) else 'shows'}{'' if way == 'down' else ' underneath'}."] if bare else []
        for n, items in self.dressed.items():
            out += [f"By the last frame, {call(n)}'s {g} {'are' if many(g) else 'is'} {'back in place' if back else 'on'}."
                    for g, back in items]
        for n, items in self.inked.items():
            out.append(f"By the last frame, {call(n)} has {join(items)} in plain view.")
        for n, items in self.wiped.items():
            out.append(f"By the last frame, {join(items)} {'is' if len(items) == 1 else 'are'} gone from {call(n)}'s skin.")
        return out

    def build(self, b, lead, text, keyed, coming, leaving, owner, belt_pic, pictures, ceiling, from_beat, silence, beside, last):
        cast, people, w = self.cast, self.people, self.w
        have = {k + 1 for k, img in enumerate(pictures) if img is not None}
        belted = any(i["gear"] == "chastity belt" for n in cast for i in w.held.get(n, []) + self.gone.get(n, []))
        refs = sorted({people[n]["head"] for n in cast if n in people and people[n]["head"] in have} |
                      ({belt_pic} if belt_pic in have and belted else set()))
        slot = {k: j for j, k in enumerate(refs, 1)}
        call = self.call_map(refs, owner)
        sheets = []
        for n in cast:
            if n not in people:
                continue
            per = people[n]
            s, p = pron(per, 0), pron(per)
            gone = [g for g, _ in self.stripped.get(n, [])]
            moved = [x for x in self.raised.get(n, []) if x[0] not in gone]
            up = [x for x in self.lifted0[n] if x[0] not in gone and x[0] not in {g for g, _ in moved}] + moved
            first = [f"At the start of the shot {s} still has {p} {join(gone)} on, until "
                     f"{'they come' if len(gone) > 1 or many(gone[0]) else 'it comes'} off."] if gone else []
            first += [f"At the start of the shot {p} {g} {'are' if many(g) else 'is'} still in place, until "
                      f"{'they are' if many(g) else 'it is'} pulled {wh}." for g, wh in moved]
            through = [i for i in self.was.get(n, []) if i in w.held.get(n, [])]
            look = writer.sheet(n, per, slot.get(per["head"]), self.wear0[n], self.off0[n] + gone, self.worn0[n], up, self.was[n],
                                self.added.get(n, []), self.skin0[n], slot.get(belt_pic), start_cover(self, n), covered(self, n),
                                self.rode.get(n))
            sheets.append(" ".join(x for x in [look, bonds.inventory(through, per, call, covered(self, n), self.over(n))] + first if x))
        gagged = {n for n in cast if any(i["spot"] == "mouth" for i in self.was.get(n, []) + self.added.get(n, []))}
        reader_res = lambda word: next(iter([n for n in cast if (people.get(n) or {}).get("sex") == ("f" if word.lower() in (
            "she", "her") else "m")][:1]), None) if len([n for n in cast if (people.get(n) or {}).get("sex") == (
                "f" if word.lower() in ("she", "her") else "m")]) == 1 else None
        v = voice.Voices(people, cast, gagged, reader_res, last)
        said = v.lines(text) + v.noises(text)
        self.voice_last = v.last
        cues = voice.timeline(text, said)
        self.floor = "ground" if voice.OUTSIDE.search(" ".join(lead)) else "floor"
        self.keep = bool(KEEP.search(unquoted_text(text)))
        steps, secs = self.steps(call)
        secs += FETCH_SECS * len(FETCH.findall(unquoted_text(text))) if steps else 0
        notes = steps
        notes += [f"It all happens at natural, real-life speed, one step after another, over about {max(2, round(secs))} seconds: every "
                  f"part is handled by hand and checked before the next one starts, no step is skipped, sped up or cut short, no "
                  f"restraint or piece of clothing appears, vanishes or changes by itself, and every reaction comes only after what "
                  f"causes it."] if self.moves else [
            "Everything happens at natural, real-life speed, in the order written, and every reaction comes only after what causes it."]
        notes += self.ends(call)
        for d, n in self.sided:
            per_d = people.get(d, {})
            notes.append(f"{cap(call(d))} does all of this from {call(n)}'s left side: {pron(per_d, 0)} stays on that side of "
                         f"{pron(people.get(n), 2)} the whole shot, nobody is at {pron(people.get(n))} right side, and {call(d)} appears "
                         f"only once in every frame.")
            beside[n] = d
        for n, d in list(beside.items()):
            if (d, n) in self.sided:
                continue
            tied = any(i["to"] for i in w.held.get(n, []))
            busy = any(x[3] == d for x in self.moves) or re.search(rf"\b{re.escape(d)}\b(?!['’])\s+(?:\w+ly\s+)?(?!(?:{IDLE})\b)"
                                                                   rf"[a-z]+s\b", unquoted_text(text), re.I)
            if not tied or busy or d not in cast or n not in cast:
                beside.pop(n, None) if not tied or busy else None
                continue
            notes.append(f"{cap(call(d))} stays at {call(n)}'s left side, the only {call(d)} in the frame, and nobody is at "
                         f"{pron(people.get(n))} right side.")
        close = bonds.near(text, cast, people, w.held)
        taut = bool(re.search(r"\b(?:tugs?|pulls?|yanks?|strains?|taut|tight)\b", text, re.I))
        turns = bonds.turning(text, cast, people, w.held)
        moving = turns | {n for n in cast if re.search(rf"\b{re.escape(n)}\b[^.]*?\b(?:{MOVES})\b", unquoted_text(text), re.I)}
        for n in cast:
            items = w.held.get(n, [])
            fresh = [i for i in items if i in self.added.get(n, [])]
            items = [dict(i, by=None) if i["gear"] == "leash" and i["by"] not in cast else i for i in items]
            pose1 = w.pose.get(n) if n in self.posed or n not in moving else None
            notes += bonds.notes(n, items, people.get(n, {}), call, self.pose0.get(n), pose1, n in self.posed, close.get(n),
                                 bool(fresh) and not any(i in self.was.get(n, []) for i in items if i["spot"] == "wrists"), self.over(n),
                                 taut, any(i["gear"] == "leash" for i in fresh), n in turns)
        for n in bonds.struggling(text, cast, people, w.held):
            notes += bonds.struggle(n, w.held[n], people.get(n, {}), call, w.pose.get(n))
        for e in self.sex:
            notes += writer.penetration(e, call, people, w.pose.get(e["who"]))
        for n in cast:
            pose = w.pose.get(n)
            held = w.held.get(n, [])
            if pose and (n in self.posed or n not in moving) and not bonds.behind(held) and not any(i["spot"] == "link" for i in held):
                notes += writer.posture(n, people.get(n, {}), call, pose, n in self.posed)
        for n, items in ((n, w.skin.get(n, [])) for n in cast):
            fresh = self.inked.get(n, [])
            kept = [x for x in items if x not in fresh]
            notes += [f"{cap(call(n))} has {join(kept)}. {'It stays' if len(kept) == 1 else 'They stay'} exactly as "
                      f"{'it is' if len(kept) == 1 else 'they are'} on {pron(people.get(n))} skin for the whole shot."] if kept else []
        notes += writer.speech(cues, said, cast, call, people)
        notes += car.notes(unquoted_text(text), " ".join(strip_pics(x) for x in lead), w.doors)
        notes += writer.presence(cast, bool(STRANGER.search(unquoted_text(text))), text, lead, call, bool(CAMERA.search(text)))
        notes += [f"{cap(call(n))} leaves during this shot." for n in leaving if n in cast]
        k = len(refs) + 1
        if keyed:
            there = [call(n) for n in cast if n not in coming]
            notes.append(f"<Picture {k}> is the frame this shot opens on." if self.i == 0 else
                         f"<Picture {k}> is the frame this shot opens on: the same place one moment earlier"
                         + (f", with only {join(there)} in view" if there else "") + f"; {join([call(n) for n in coming])} "
                         f"{'comes' if len(coming) == 1 else 'come'} into view during this shot." if coming else
                         f"<Picture {k}> is the frame this shot opens on: the same place and the same people one moment earlier, "
                         f"carried forward.")
        notes += [f"<Picture {slot[people[n]['head']]}> is for {call(n)}'s face and hair only: {pron(people[n])} clothes"
                  f"{' and restraints' if self.was.get(n) or w.held.get(n) else ''} are as this shot describes them."
                  for n in cast if n in people and people[n]["head"] in slot]
        named = relabel(text, cast, call)
        shown = voice.voiced(named, said, call, cast)
        retag = lambda t: re.sub(PIC, lambda m: f"<Picture {slot[int(m.group(1))]}>" if int(m.group(1)) in slot else "", t,
                                 flags=re.I).replace("  ", " ")
        prompt = "\n\n".join(x for x in [retag(t) for t in lead] + ["\n".join(sheets), retag(shown), " ".join(notes)]
                             if x.strip())
        act = max(voice.actions(text) * voice.PER_ACTION, secs)
        self.need = BEAT_BASE + act + (SETTLE if self.moves or self.inked or self.wiped else 0)
        talk = voice.talk_time(cues)
        changed = bool(self.moves or self.inked or self.wiped)
        fit = ""
        if b["seconds"]:
            frames = align(round(b["seconds"] * FPS))
        elif from_beat:
            frames = min(max(MIN_FRAMES, align(round((max(BEAT_BASE + act, talk) + (SETTLE if changed else 0)) * FPS), True)), ceiling)
        else:
            frames = ceiling
        if round(talk * FPS) > frames and not b["seconds"]:
            frames, fit = align(round(talk * FPS)), "lengthened to fit its line"
        cues = [(a, min(e, round(frames / FPS, 2)), kd) for a, e, kd in cues if a < frames / FPS]
        if act > (frames / FPS - BEAT_BASE) * RUSH:
            self.note(f"what happens in it takes about {BEAT_BASE + act:.0f}s at real-life speed, more than its {frames / FPS:.1f}s, so it "
                      f"may look rushed; {'give it more seconds:' if b['seconds'] else 'raise shot_seconds or split the beat'}")
        spoken = [ln for ln in said if not ln["thought"]]
        self.warn_after += [f"shot {self.i + 1}: the line \"{cut(ln['q'], 40)}\" has no speaker; name who says it, as in '\"…,\" "
                            f"{cast[0] if cast else 'Dana'} says'" for ln in spoken if not ln["who"] and ln["q"]]
        gear = [i["gear"] for x in (self.added, self.gone) for v in x.values() for i in v]
        muffled = [(ln["who"], "the " + next((bonds.name(i) for i in w.held.get(ln["who"], []) + self.gone.get(ln["who"], [])
                                             if i["spot"] == "mouth"), "gag")) for ln in said if ln["gag"] and ln["who"]]
        sound = voice.sound(" ".join(strip_pics(x) for x in lead), voice.voiced(named, said, call, cast, True), [call(n) for n in cast],
                            gear, list(dict.fromkeys((call(n), g) for n, g in muffled)), bool(MUSIC.search(text)))
        info = self.info(frames, fit, b, cast, cues, said, refs, leaving, call)
        return {"n": self.i + 1, "beat": b.get("beat", self.i + 1),
                "prompt": prompt, "frames": frames, "keyed": keyed, "cut": b["cut"], "cast": cast,
                "refs": [k - 1 for k in refs], "faces": [k in owner for k in refs],
                "silent": silence and not any(kd == "line" for _, _, kd in cues), "cues": cues, "changed": changed, "coming": coming,
                "sound": sound, "mute": (silence or bool(cues)) and not (MUSIC.search(text) or voice.VOCAL.search(unquoted_text(text))
                                                                         and not cues), "line": info,
                "read": [(k, f"{e['who']}, " + "; ".join(item_text(x, e["who"], self.people) for x in e["items"]))
                         for k, e in self.read if e["doer"] is not None or e["sent"] and ":" not in e["sent"][:9]],
                "said": [(ln["who"], ln["q"], bool(ln["gag"])) for ln in spoken if ln["q"]]}

    def info(self, frames, fit, b, cast, cues, said, refs, leaving, call):
        listed = lambda state: " ".join(f"{n}: {'; '.join(v)}." for n, v in state.items() if v)
        per = lambda n: (pron(self.people.get(n)), pron(self.people.get(n), 2))
        bits = [f"shot {self.i + 1}: {frames} frames ({frames / FPS:.1f}s)"] + [x for x in (
            fit, "after a cut" * b["cut"], f"part of beat {b.get('beat')}, split to fit" * bool(b.get("split"))) if x]
        heard = [e for k, e in self.read if not e["sent"].startswith(tuple(f"{x}:" for x in
                                                                            ("hold", "release", "remove", "wear", "lift", "tear", "mark",
                                                                             "unmark", "pose")))]
        bits += ["read from the beat: " + "; ".join(f"{e['kind']}: {e['who']}, " + "; ".join(item_text(x, e["who"], self.people)
                                                                                            for x in e["items"]) +
                                                    f" (\"{cut(e['sent'])}\")" for e in heard)] if heard else []
        through = {n: [bonds.describe(i, *per(n)) for i in self.was.get(n, []) if i in self.w.held.get(n, [])] for n in cast}
        bits += [f"{label}: {text}" for label, text in (
            ("cast", ", ".join(cast)), ("on", listed(through)),
            ("goes on", listed({n: [bonds.describe(i, *per(n)) for i in v] for n, v in self.added.items()})),
            ("comes off", listed({n: [bonds.describe(i, *per(n)) for i in v] for n, v in self.gone.items()})),
            ("clothes off", listed({n: [g for g, _ in v] for n, v in self.stripped.items()})),
            ("pulled", listed({n: [f"{g} {wh}" for g, wh in v] for n, v in self.raised.items()})),
            ("puts on", listed({n: [g for g, _ in v] for n, v in self.dressed.items()})),
            ("marks added", listed(self.inked)), ("marks off", listed(self.wiped)),
            ("pose", listed({n: [item_text(v, n, self.people)] for n, v in self.posed.items()})),
            ("leaves", ", ".join(leaving)),
            ("speaks", "; ".join(f"{ln['who'] or '?'}{' (thinks)' if ln['thought'] else ''}{' (muffled)' if ln['gag'] else ''}: "
                                 f"\"{cut(ln['q'] or ln['verb'], 40)}\"" for ln in said)),
            ("pictures", ", ".join(f"ref_image_{k}" for k in refs))) if text]
        return ", ".join(bits)


def covered(shot, n):
    if not any(i["gear"] == "chastity belt" for i in shot.w.held.get(n, [])):
        return []
    stays = shot.w.wearing(n)
    up = {g for g, _ in shot.w.lifted.get(n, [])}
    return [g for g in shot.wear0.get(n, []) if g in stays and "lower" in covers(g) and g not in up and
            g not in {x for x, _ in shot.lifted0.get(n, [])}]


def start_cover(shot, n):
    if not any(i["gear"] == "chastity belt" for i in shot.w.held.get(n, []) + shot.gone.get(n, [])):
        return []
    return [g for g in shot.wear0.get(n, []) if "lower" in covers(g) and g not in {x for x, _ in shot.lifted0.get(n, [])}]


def item_text(x, who, people):
    per = people.get(who, {})
    if isinstance(x, dict) and "gear" in x and x.get("look") is not None:
        return bonds.describe(x, pron(per), pron(per, 2))
    if isinstance(x, dict) and "pose" in x:
        return " ".join(v for v in (x["pose"], x.get("way", ""), f"on {x['where']}" if x.get("where") else "") if v)
    if isinstance(x, dict) and "tool" in x:
        return f"{x['tool']} into {x['hole']}" + (f", {x['way']}" if x["way"] else "")
    if isinstance(x, dict):
        return x.get("gear") or x.get("spot") or "restraint"
    if isinstance(x, tuple):
        return " ".join(v for v in x if v)
    return str(x)


def unquoted_text(text):
    return re.sub(r"[\"“][^\"”]*[\"”]", " ", text)


def relabel(text, cast, call):
    out = text
    for n in sorted(cast, key=len, reverse=True):
        if call(n) != n:
            parts = re.split(r"([\"“][^\"”]*[\"”])", out)
            out = "".join(x if x[:1] in "\"“" else re.sub(rf"(?<![\w-])(?<!-haired ){re.escape(n)}(?![\w-])", call(n), x) for x in parts)
    return re.sub(r"(^|[.!?]\s+|\n)([a-z])(?=[\w-]*-haired\b)", lambda m: m.group(1) + m.group(2).upper(), out)
