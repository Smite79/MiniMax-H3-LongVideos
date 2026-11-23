# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import re

try:
    from .words import ABOVE, BED, FIXTURE, HUE, SOFT, WIDE, cap, many, pron, sentences
except ImportError:
    from words import ABOVE, BED, FIXTURE, HUE, SOFT, WIDE, cap, many, pron, sentences

GEAR = (("crotch chain", r"crotch[\s-]+chains?", "", "crotch"),
        ("crotch rope", r"crotch[\s-]+ropes?", "", "crotch"),
        ("chastity belt", r"chastity[\s-]+(?:belts?|devices?)", "", "crotch"),
        ("spreader bar", r"spreader(?:[\s-]+bars?)?", "", "ankles"),
        ("ball gag", r"ball[\s-]?gags?", "", "mouth"),
        ("zip ties", r"zip[\s-]?ties?|cable[\s-]+ties?", r"zip[\s-]?ti(?:es|ed)", "wrists"),
        ("leash", r"leash(?:es)?|(?:chain|leather|dog|metal|steel)[\s-]+leads?", r"leash(?:es|ed)", "neck"),
        ("collar", r"collars?", r"collar(?:s|ed)", "neck"),
        ("shackles", r"shackles?|leg[\s-]+irons|fetters|ankle[\s-]+chains?", r"shackle[sd]?|fettered", "ankles"),
        ("handcuffs", r"handcuffs?|manacles?|(?:wrist|ankle|leg)?[\s-]*cuffs", r"(?:hand)?cuff(?:s|ed)|manacle[sd]", "wrists"),
        ("chains", r"chains?", r"chain(?:s|ed)|padlock(?:s|ed)", "wrists"),
        ("rope", r"ropes?|cords?", r"ti(?:es|ed)|bound|binds", "wrists"),
        ("tape", r"(?:duct[\s-]+)?tape", r"(?:duct[\s-]?)?tap(?:es|ed)", "mouth"),
        ("straps", r"straps?|belts?", r"strap(?:s|ped)|buckle[sd]", "wrists"),
        ("blindfold", r"blindfolds?", r"blindfold(?:s|ed)", "eyes"),
        ("gag", r"gags?", r"gag(?:s|ged)", "mouth"),
        ("hogtie", r"hog[\s-]?ties?", r"hog[\s-]?ti(?:es|ed)", "wrists"))
NOUN = {k: re.compile(rf"\b(?:{n})\b", re.I) for k, n, _, _ in GEAR}
VERB = {k: re.compile(rf"\b(?:{v})\b", re.I) for k, _, v, _ in GEAR if v}
SPOT = (("wrists", r"wrists?|hands|arms"), ("ankles", r"ankles?|feet|legs"), ("knees", r"knees"), ("thighs", r"thighs"),
        ("elbows", r"elbows"), ("neck", r"neck|throat"), ("mouth", r"mouth|lips"), ("eyes", r"eyes"),
        ("waist", r"waist|hips|belly"), ("crotch", r"crotch|between\s+(?:her|his|their)\s+legs"))
LINK = re.compile(r"\b(?:to|onto)\s+(?:the\s+(?:chain|links?|middle|center)\s+(?:of|between)\s+)?(?:her|his|their|the|"
                  r"(?-i:[A-Z])[\w'’-]*['’]s)\s+(?:[\w-]+\s+)?(?:(?:hand)?cuffs|(?:cuffed\s+)?(?:wrists|hands))\b", re.I)
TO = re.compile(rf"\b(?:to|onto|from)\s+(?P<to>(?:the|a|an|her|his|their)\s+(?:(?!(?:to|from|onto)\b)[\w'’-]+\s+){{0,3}}?"
                rf"{FIXTURE}(?:\s+{FIXTURE})*(?:\s+(?:in|on|at|of)\s+the\s+{FIXTURE})?)\b", re.I)
WORN = {"ball gag", "gag", "blindfold", "collar", "chastity belt", "crotch chain", "crotch rope"}
HARNESS = {"crotch chain": ("chain", "locked", "unlocks the padlock", "locks it{t} with a padlock"),
           "crotch rope": ("rope", "tied", "unties the knot", "ties it off{t} in a firm knot")}
OBJ = {"her": "her", "his": "him", "their": "them"}
LOW = ("sitting", "kneeling", "crouching", "lying", "bent over", "on all fours")
SEAT = r"chair|stool|seat|armchair|bench|couch|sofa"
ON_SECS = {"handcuffs": 6, "shackles": 7, "rope": 12, "chains": 9, "zip ties": 4, "straps": 8, "tape": 8, "collar": 6, "leash": 3,
           "chastity belt": 11, "crotch chain": 11, "crotch rope": 13, "spreader bar": 8, "ball gag": 7, "gag": 5, "blindfold": 5}
OFF_SECS = {"handcuffs": 5, "shackles": 5, "rope": 8, "chains": 6, "zip ties": 3, "straps": 5, "tape": 5, "collar": 4, "leash": 2,
            "chastity belt": 6, "crotch chain": 7, "crotch rope": 8, "spreader bar": 5, "ball gag": 4, "gag": 3, "blindfold": 3}
ON = {"handcuffs": "opens one cuff, closes it around {x} left {one} and ratchets it snug with a click, then closes the other around {p} "
                   "right {one} the same way and checks that both are tight",
      "shackles": "closes one shackle around {x} left ankle and locks it, then closes the other around {p} right ankle and locks it",
      "rope": "wraps the rope around {x} {s} several times, pulling each loop snug, cinches it between them and ties it off in a firm "
              "knot",
      "chains": "wraps the chain around {x} {s} twice, pulls it snug and locks the two ends together with a padlock",
      "zip ties": "loops a zip tie around {x} {s} and pulls it tight with a ratcheting zip",
      "straps": "wraps a strap around {x} {s}, pulls it tight and fastens the buckle",
      "tape": "winds the tape around {x} {s} several times, smoothing each layer down, and tears it off the roll",
      "collar": "wraps the {u} around {x} neck, threads the strap through the buckle, pulls it snug and fastens it",
      "leash": "clips the {u} to the ring at the front of {x} collar and checks that the clip is closed",
      "chastity belt": "fits the chastity belt around {x} bare waist, brings its shield down between {p} legs, right against {p} "
                       "bare crotch, settles it until it sits flat, and locks it shut",
      "spreader bar": "spreads {x} feet apart and locks one end of the spreader bar around each ankle in turn",
      "ball gag": "pushes the ball into {x} mouth behind {p} teeth, pulls the strap around behind {p} head and buckles it",
      "gag": "fits the gag into {x} mouth and fastens it behind {p} head",
      "blindfold": "lays the blindfold over {x} eyes and ties it behind {p} head"}
OFF = {"handcuffs": "unlocks the {u} with a small key and takes them off {x} {s}", "collar": "unbuckles the collar from {x} neck",
       "leash": "unclips the {u} from {x} collar", "ball gag": "unbuckles the gag and takes it out of {x} mouth",
       "gag": "takes the gag out of {x} mouth", "blindfold": "unties the blindfold and lifts it off {x} eyes",
       "chastity belt": "unlocks the chastity belt and takes it off {x} waist", "spreader bar": "unlocks the spreader bar from {x} ankles"}
METAL = re.compile(r"steel|metal|iron|chrome|brass|silver|gold", re.I)
STRUGGLE = re.compile(r"\b(?:struggl\w*|squirm\w*|writh\w*|thrash\w*|wriggl\w*|flail\w*|fights?|fighting|fought|resists?|resisting|"
                      r"(?:tugs?|pulls?|strains?|twists?|jerks?|yanks?)\s+(?:at|against)\s+(?:her|his|their|the)\s+(?:[\w-]+\s+)?"
                      r"(?:restraints|bonds|cuffs|handcuffs|chains?|ropes?|leash|collar|tape)|(?:break|get|wriggle|twist|pull)\s+"
                      r"(?:herself\s+|himself\s+|themselves\s+)?free)\b", re.I)
WHO = lambda k: rf"(?P<o{k}>her|him|them|(?-i:[A-Z])[\w'’-]+)"
BEHIND = re.compile(rf"\b(?:behind\s+{WHO(1)}(?!['’]s)(?!\s+back)|from\s+behind|"
                    rf"(?:against|at)\s+(?:her|his|their|(?-i:[A-Z])[\w'’-]+['’]s)\s+back|spoons?|"
                    rf"(?:pushes|shoves|nudges|marches|steers|prods|propels)\s+{WHO(2)}\s+(?:forward|ahead|along|on\b|toward|towards|into|"
                    rf"out|through|across)|pulls?\s+{WHO(3)}\s+back\s+(?:against|into))", re.I)
PLACED = re.compile(r"\b(?:the|a|an|his|her|their)\s+(?:[\w'’-]+\s+){1,3}?(?:up\s+|right\s+|just\s+|down\s+|back\s+)?$", re.I)
WALL = re.compile(r"\bagainst\s+(?:the|a)\s+(?P<w>wall|door|counter|desk|table|cabinet|car|fridge|bookcase|shelf|window|tree)\b", re.I)
FRONT = re.compile(rf"\b(?:hugs?|hugging|embraces?|embracing|kiss(?:es)?|kissing|holds?\s+{WHO(1)}\s+close|"
                   rf"pulls?\s+{WHO(2)}\s+(?:close|in\b|into\s+(?:his|her|their)\s+arms)|face\s+to\s+face|in\s+front\s+of\s+{WHO(3)}|"
                   rf"faces\s+{WHO(4)}|confronts?\s+{WHO(5)}|(?:walks?|steps?|comes?)\s+up\s+to\s+{WHO(6)}|approaches\s+{WHO(7)}|"
                   rf"(?:presses|pushes|pins)\s+(?:himself|herself|themselves)\s+against\s+{WHO(8)})", re.I)
HOLD = re.compile(rf"\b(?:(?:wraps?|puts?|throws?|slides?|slips?|locks?)\s+(?:his|her|their|both)\s+arms?\s+(?:tight(?:ly)?\s+)?"
                  rf"around\s+{WHO(1)}|holds?\s+{WHO(2)}\s+(?:tight(?:ly)?|still|in\s+place|in\s+(?:his|her|their)\s+arms|"
                  rf"against\s+(?:him|her|them|himself|herself|themselves|his|her|their)|"
                  rf"by\s+(?:the|her|his|their)\s+(?:waist|arms?|shoulders?|hips?))|"
                  rf"cuddles?\s+{WHO(3)}|grabs?\s+{WHO(4)}|restrains?\s+{WHO(5)}|holds?\s+(?:on\s+to|onto)\s+{WHO(6)}|"
                  rf"holds?\s+{WHO(8)}(?=\s*(?:[.,;!]|$|and\b))|"
                  rf"pulls?\s+{WHO(7)}\s+(?:back\s+)?(?:against|into)\s+(?:him|her|them|his|her|their))", re.I)
TURN = re.compile(rf"(?:\b{WHO(1)}\s+)?\b(?:turns?|turned|turning|spins?|spun|spinning|whirls?|whirled|pivots?|pivoted|swivels?|"
                  rf"twirls?|wheels?)\s+(?:{WHO(2)}\s+)?(?:(?:slowly|quickly|fully|halfway|half|all\s+the\s+way|back)\s+)?(?:around|"
                  rf"round|away|to\s+face|toward|towards|(?:her|his|their)\s+back|to\s+the\s+(?:wall|door|window|corner))\b", re.I)


def spots(text):
    text = re.sub(rf"\b(?:{SPOT[-1][1]})\b", "crotch", text, flags=re.I)
    return [k for k, rx in SPOT if re.search(rf"\b(?:{rx})\b", text, re.I)]


def look(text, gear):
    noun = next(n for k, n, _, _ in GEAR if k == gear)
    m = re.search(rf"\b((?:(?:{HUE})[\s,]+(?:and\s+)?){{1,3}})(?:{noun})\b", text, re.I)
    return " ".join(w for w in re.split(r"[\s,]+", m.group(1).lower()) if w and w != "and") if m else ""


def pose(text):
    return "behind" if re.search(r"\bbehind\s+(?:her|his|their|[\w'’-]+['’]s)\s+back\b|\bbehind\s+(?:her|him|them)\b", text,
                                 re.I) else \
        "front" if re.search(r"\bin\s+front\b", text, re.I) else \
        "above" if re.search(r"\b(?:above|over)\s+(?:her|his|their)\s+head\b|\boverhead\b|\bfrom\s+the\s+ceiling\b", text, re.I) \
        else None


def item(gear, text="", spot=None, **kw):
    spot = spot or next((s for s in spots(text) if s not in ("waist",) or gear in ("chains", "rope", "tape")), None) or \
        next(d for k, _, _, d in GEAR if k == gear)
    to = kw.pop("to", None)
    to = to if to is not None else next((m.group("to").lower() for m in reversed(list(TO.finditer(text))) if not re.match(
        r"\s+(?:on|of|at|in)\s+(?:the\s+front\s+of\s+)?(?:her|his|their|the|(?-i:[A-Z])[\w'’-]*['’]s)\s+(?:[\w-]+\s+)?collar\b",
        text[m.end():], re.I) and re.search(
        r"\b(?:chain|lock|tie|tied|fasten|padlock|clip|hook|(?:hand)?cuff|strap|tape|attach|secure|bind|bound|shackle|tether|anchor|"
        r"leash)\w*\s+(?:[\w'’,-]+\s+){0,5}$", text[:m.start()], re.I)), None)
    to = re.sub(r"^(?:a|an|her|his|their)\s+", "the ", to) if to else None
    by = re.search(r"\b(?:held\s+by|in)\s+((?-i:[A-Z])[\w'’-]+)(?:['’]s\s+hand)?\b", text)
    it = {"gear": gear, "look": look(text, gear) if text else "", "spot": spot,
          "pose": (pose(text) or ("behind" if not to else None)) if spot in ("wrists", "elbows") else None, "to": to,
          "by": by.group(1) if by else None, "chain": False}
    it.update(kw)
    return it


def parse(text):
    link = LINK.search(text)
    if link and re.search(r"\bcrotch\b|\bbetween\s+(?:her|his|their)\s+legs\b", text, re.I):
        out = parse(text[:link.start()] + text[link.end():])
        if any(x["gear"] in HARNESS for x in out):
            return [dict(x, to="handcuffs") if x["gear"] in HARNESS else x for x in out]
    found = [(k, m.start()) for k, rx in NOUN.items() for m in [rx.search(text)] if m]
    found = [(k, a) for k, a in found if not any(k2 != k and a2 <= a and NOUN[k2].search(text[a2:]).end() > a - a2 + 1
                                                 for k2, a2 in found if k2 in ("crotch chain", "crotch rope", "chastity belt",
                                                                               "ball gag", "spreader bar", "zip ties"))]
    found = [k for k, _ in sorted(found, key=lambda x: x[1])]
    found = [k for k in found if not ({"gag": "ball gag", "chains": "crotch chain", "rope": "crotch rope", "straps": "chastity belt"}
                                      .get(k) in found)]
    if not found:
        found = [k for k, rx in VERB.items() if rx.search(text)][:1]
    if "hogtie" in found:
        base = next((k for k in found if k != "hogtie"), "rope")
        return [item(base, text, "wrists", pose="behind", to=""), item(base, text, "ankles", to=""), item(base, text, "link", to="")]
    if "collar" in found and ({"leash", "chains"} & set(found) or TO.search(text)):
        lead = item("leash", text, "neck", chain="leash" not in found or bool(re.search(r"\bchain\b", text, re.I)))
        lead["look"] = "" if lead["chain"] else lead["look"]
        return [item("collar", re.sub(r"\b(?:leash|chains?)\b.*$", "", text, flags=re.I), "neck", to=""), lead]
    if found and found[0] in ("chains", "leash") and (re.search(r"\b(?:collar|neck|throat)\b", text, re.I) or "leash" in found):
        return [item("leash", text, "neck", chain=found[0] == "chains" or bool(re.search(r"\bchain\b", text, re.I)))]
    out = []
    for k in found[:1] if found and found[0] in WORN else found:
        here = [s for s in spots(text) if k not in WORN]
        band = k in ("chains", "rope") and "crotch" in here
        rest = [s for s in here if s not in ("waist", "crotch")] if band else here
        out += [item(f"crotch {k[:5]}", text, "crotch")] if band else []
        out += [item(k, text, s) for s in (rest if band or len(here) > 1 and k in ("handcuffs", "rope", "chains", "tape", "straps",
                                                                                   "zip ties") else [None])]
    return [x for x in out if not (x["gear"] in ("chains", "straps") and any(y["gear"] in ("handcuffs", "shackles") and
                                                                              y["spot"] == x["spot"] for y in out))]


def soft(it):
    return bool(SOFT.search(it["look"])) and it["gear"] == "handcuffs"


def name(it):
    g = "cuffs" if soft(it) or it["gear"] == "handcuffs" and it["spot"] != "wrists" else it["gear"]
    g = f"{'chain' if it['chain'] else 'leash'}" if it["gear"] == "leash" else g
    return f"{it['look']} {g}".strip()


def article(word):
    return "" if re.search(r"(?:s|cuffs|tape|(?<!crotch\s)rope)$", word) else "an " if word[:1] in "aeiou" else "a "


def where(it, p, o):
    return {"behind": f"behind {p} back", "front": f"together in front of {o}", "above": f"above {p} head"}.get(it["pose"], "together")


def middle(to):
    return f"the middle of {to}" if to and WIDE.search(re.split(r"\s+(?:in|on|at)\s+the\s+", to)[0]) and not re.search(
        r"\b(?:left|right|corner|post|leg|end|middle|top|bottom|side)s?\b", to) else to


def same(a, b):
    return a["gear"] == b["gear"] and a["spot"] == b["spot"] or a["spot"] == b["spot"] and a["spot"] in ("mouth", "eyes", "crotch") \
        or a["spot"] == b["spot"] == "neck" and a["gear"] == b["gear"]


def linked(it):
    return it["gear"] in HARNESS and it["to"] == "handcuffs"


def behind(items):
    return next((i for i in items if i["spot"] == "wrists" and i["pose"] == "behind" and not i["to"]), None)


def groin(p):
    return "vagina" if p == "her" else "groin"


def bed(p, over):
    fitted, loose = (over or (None, None))[:2]
    return f" under {p} {loose}" if loose else "", f"the outside of {p} {fitted}" if fitted else None if loose else f"{p} {groin(p)}"


def describe(it, p, o, call=str, over=None):
    n, s = name(it), it["spot"]
    a = article(n)
    if it["gear"] == "leash":
        end = f", its other end {'padlocked' if it['chain'] else 'tied'} to {middle(it['to'])}" if it["to"] else \
            f", its other end in {call(it['by'])}'s hand" if it["by"] else ""
        return f"{a}{n} clipped to the ring at the front of {p} collar{end}"
    if s == "link":
        return f"a separate short {'chain' if it['gear'] in ('handcuffs', 'chains', 'shackles') else it['gear']} joining " \
               f"{p} wrists to {p} ankles behind {o}"
    tie = f", {'chained' if it['gear'] in ('handcuffs', 'chains', 'shackles') else 'tied'} to {middle(it['to'])}" if it["to"] else ""
    if it["gear"] in HARNESS:
        (c, fixed), (under, onto) = HARNESS[it["gear"]][:2], bed(p, over)
        return f"{a}{n} {fixed} tight around {p} waist{under}, its center {c} pulled tight under {p} crotch" + \
            (f" and over {onto}" if onto else "") + f" from the front of {p} waist to the small of {p} back" + ("" if linked(it) else tie)
    text = {"collar": f"{a}{n} locked around {p} neck",
            "chastity belt": f"{a}{n} locked around {p} waist, its shield between {p} legs over {p} crotch",
            "spreader bar": f"{a}{n} locked between {p} ankles, holding {p} feet apart",
            "ball gag": f"{a}{n} strapped in {p} mouth", "gag": f"{a}{n} in {p} mouth", "blindfold": f"{a}{n} over {p} eyes",
            "shackles": f"{n} locked on {p} ankles, joined by their own short chain"}.get(it["gear"])
    if text:
        return text + tie
    if it["gear"] == "tape" and s in ("mouth", "eyes"):
        return f"{n} over {p} {s}"
    if it["gear"] == "tape" and s == "crotch":
        return f"{n} sealed over {p} crotch, between {p} legs{tie}"
    verb = {"handcuffs": "locking", "chains": "locked around", "rope": "tied around", "tape": "wound around",
            "straps": "buckled around", "zip ties": "pulled tight around"}.get(it["gear"], "on")
    place = f"{p} {s}" + (f" {where(it, p, o)}" if s == "wrists" else " together" if s in ("ankles", "knees", "thighs") else "")
    return f"{a}{n} {verb} {place}{tie}"


def secs(it, on=True):
    if it["spot"] == "link":
        return 4
    if it["gear"] == "tape" and it["spot"] in ("mouth", "eyes"):
        return 3 if on else 2
    return (ON_SECS if on else OFF_SECS).get(it["gear"], 6 if on else 4) + 2 * (it["pose"] == "behind") + 3 * linked(it) + \
        2 * (it["spot"] in ("ankles", "knees") and it["gear"] != "spreader bar") + 2 * bool(it["to"] and not linked(it) and on)


def put(have, new):
    for it in new:
        old = next((x for x in have if same(x, it)), None)
        if old is not None:
            it = {**it, "look": it["look"] or old["look"], "to": it["to"] if it["to"] is not None else old["to"]}
            have = [it if x is old else x for x in have]
        else:
            have = have + [it]
    return have


def fits(spec, it, mine):
    fam = lambda g: {"ball gag": "gag", "chains": "bind", "rope": "bind", "zip ties": "bind", "straps": "bind", "tape": "bind",
                     "handcuffs": "bind", "leash": "collar"}.get(g, g)
    unit = {"crotch chain": "chains", "crotch rope": "rope"}.get(it["gear"])
    if spec.get("gear") == "leash":
        return 2 * (it["gear"] == "leash")
    if spec.get("gear") and spec.get("spot") and spec["spot"] == it["spot"] and spec["spot"] in ("mouth", "eyes", "neck", "crotch"):
        return 2
    if unit:
        return 2 * bool(spec.get("gear") in (None, unit) and spec.get("spot") in (None, "crotch", "waist")
                        or spec.get("gear") == unit and not any(x["gear"] == unit for x in mine))
    gear_ok = not spec.get("gear") or fam(spec["gear"]) == fam(it["gear"])
    spot_ok = not spec.get("spot") or spec["spot"] == it["spot"] or spec.get("gear") and spec["spot"] == item(spec["gear"])["spot"]
    return (1 + (spec.get("gear") in (None, it["gear"]))) * bool(gear_ok and spot_ok)


def harness(it, x, p, on, over, doer):
    (c, _, opens, close), (under, onto), u = HARNESS[it["gear"]], bed(p, over), name(it)
    center = f"centers it tight over {onto}" if onto else "pulls it tight and centered"
    back = f"steps behind {OBJ.get(p, p)}, " if doer else ""
    end = f"{'padlocks' if c == 'chain' else 'ties'} its end through the short links that join {p} two cuffs" if linked(it) else \
        close.format(t=f" to the back of the waist {c}")
    if on:
        return f"takes a separate {u}, wraps its waist {c} tight around {x} waist{under} and " \
               f"{close.format(t=f' at the front, below {p} navel,')}, runs its center {c} down from the front of the waist {c} under " \
               f"{p} crotch, {center}, then {back}draws the center {c} up between {p} buttocks to the small of {p} back and {end}"
    free = f"{'unlocks' if c == 'chain' else 'unties'} the end of the center {c} from the short links between {x} two cuffs" \
        if linked(it) else f"{opens} at the small of {x} back"
    return f"{free}, draws the center {c} out from between {p} legs, then {opens} at the front of {p} waist and takes the {u} off " \
           f"{p} waist"


def action(it, x, p, on, over, doer):
    s = it["spot"]
    if it["gear"] in HARNESS:
        return harness(it, x, p, on, over, doer)
    text = (ON if on else OFF).get(it["gear"]) or ("fastens the {u} around {x} {s}" if on else "takes the {u} off {x} {s}")
    if it["gear"] == "tape" and on and s in ("mouth", "eyes"):
        text = "presses a strip of tape over {x} {s} and smooths it down"
    if it["gear"] == "tape" and on and s == "crotch":
        text = "runs the tape from the front of {x} waist down between {p} legs and up to the back, flat over {p} crotch"
    if it["gear"] == "collar" and METAL.search(it["look"]):
        text = "opens the {u}, fits it around {x} neck, closes it and locks it shut with a click" if on else \
            "unlocks the {u} and takes it off {x} neck"
    one = {"wrists": "wrist", "ankles": "ankle", "knees": "knee", "elbows": "elbow"}.get(s, s)
    text = text.format(x=x, p=p, s=s, one=one, u=name(it))
    if on and it["to"] and it["gear"] not in ("leash", *HARNESS):
        text += f", then {'chains' if it['gear'] in ('handcuffs', 'chains', 'shackles') else 'ties'} them to {middle(it['to'])}"
    if on and it["gear"] == "leash" and it["to"]:
        text += f" and padlocks its other end to {middle(it['to'])}"
    return text


def step(it, x, p, o, who=None, on=True, over=None, bound=False):
    s = it["spot"]
    if not who:
        return action(it, x, p, on, over, False)
    if s == "wrists" and it["pose"] == "behind" and not it["to"]:
        return f"steps behind {who}, takes both of {p} wrists and draws {p} arms back behind {o}, then " \
               f"{action(it, p, p, True, over, True)}, there at the small of {p} back, keeping {p} hands there the whole time, " \
               f"hidden from the front by {p} body" if on else \
            f"steps behind {who} and, at the small of {p} back, {action(it, p, p, False, over, True)}, while {p} hands stay behind " \
            f"{p} back until {p} wrists are free"
    if s == "wrists" and it["pose"] == "front" and not it["to"]:
        return f"stands in front of {who}, takes both of {p} wrists together in front of {o}, then {action(it, p, p, True, over, True)}" \
            if on else f"stands in front of {who} and {action(it, p, p, False, over, True)}"
    if s in ("ankles", "knees") and it["gear"] != "spreader bar" or it["gear"] == "spreader bar":
        return f"crouches down at {x} feet, then {action(it, p, p, on, over, True)}" if on else \
            f"crouches down at {x} feet and {action(it, p, p, False, over, True)}"
    if it["gear"] in HARNESS and (linked(it) or bound) and not on:
        return f"steps behind {who} and {action(it, p, p, False, over, True)}"
    if bound and s not in ("wrists", "elbows", "link"):
        return f"stands in front of {who} and {action(it, p, p, on, over, True)}"
    return action(it, x, p, on, over, True)


def link_step(it, who, p):
    c = HARNESS[it["gear"]][0]
    return f"steps behind {who} and unhooks the end of the center {c} from the back of the waist {c} and " \
           f"{'padlocks' if c == 'chain' else 'ties'} it through the short links that join {p} two cuffs, at the small of {p} back"


def inventory(items, per, call, covered, over=None):
    p, o, s = pron(per), pron(per, 2), pron(per, 0)
    band = next((i for i in items if linked(i)), None)
    tied = f", and the end of the center {HARNESS[band['gear']][0]} of {p} {name(band)} is padlocked through the short links " \
           f"between them, at the small of {p} back" if band else ""
    lines = [f"{cap(s)} {'have' if s == 'they' else 'has'} {describe(it, p, o, call, over)}"
             f"{tied if it['spot'] == 'wrists' and it['pose'] == 'behind' else ''}." for it in sorted(
        items, key=lambda i: 0 if i["gear"] == "collar" else 1 if i["gear"] == "leash" else 3 if i["spot"] == "wrists" else 2)
        if not (it["gear"] == "chastity belt" and covered)]
    return " ".join(lines) + " All of it stays on, exactly as described, for the whole shot." if lines else ""


def refers(word, n, people, bound):
    per = people.get(n) or {}
    return bool(word) and (word.lower() == n.lower() or word.lower() in (pron(per, 0), pron(per, 2)) and
                           sum((people.get(b) or {}).get("sex") == per.get("sex") for b in bound) == 1)


def near(text, cast, people, held):
    bound = [n for n in cast if behind(held.get(n, []))]
    out = {}
    for s in sentences(re.sub(r"[\"“][^\"”]*[\"”]", " ", text)):
        names = [n for n in cast if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", s)]
        for n in bound:
            mine = n in names or any(refers(w, n, people, bound) for w in re.findall(r"\b(?:she|he|her|him|they|them)\b", s, re.I))
            others = [y for y in names if y != n] or [y for y in cast if y != n][:1] if len(cast) == 2 else [y for y in names if y != n]
            if not mine or not others:
                continue
            hit = [(side, m) for side, rx in (("behind", BEHIND), ("wall", WALL), ("front", FRONT), ("hold", HOLD)) for m in rx.finditer(s)
                   if not (side == "behind" and m.group(0).lower().startswith("behind") and PLACED.search(s[:m.start()]))
                   and all(refers(v, n, people, bound) for k, v in m.groupdict().items() if k[0] == "o" and v)]
            if hit:
                side, m = hit[0]
                out[n] = (others[0], side, m.groupdict().get("w"))
    return out


def turning(text, cast, people, held):
    bound = [n for n in cast if behind(held.get(n, []))]
    named = rf"\b(?:{'|'.join(map(re.escape, cast))}|she|he|her|him|they|them)\b" if cast else r"(?!)"
    out = set()
    for s in sentences(re.sub(r"[\"“][^\"”]*[\"”]", " ", text)):
        for m in TURN.finditer(s):
            said = (m["o2"], m["o1"], *reversed(re.findall(named, s[:m.start()], re.I)))
            who = next((w for w in said if w and re.fullmatch(named, w, re.I)), None)
            out |= {n for n in bound if refers(who, n, people, bound)}
    return out


def hands(who, it, per, call, pose0, pose1, changed, close, fresh, tied, turns):
    p, o, s, name_ = pron(per), pron(per, 2), pron(per, 0), cap(call(who))
    plural = {"does": "do", "stays": "stay", "goes": "go", "rises": "rise", "makes": "make"}
    does = lambda verb: plural.get(verb, verb) if s == "they" else verb
    at, way, spot = (pose1 or {}).get("pose") or "standing", (pose1 or {}).get("way", ""), (pose1 or {}).get("where", "")
    where_ = f" {'in' if re.search(SEAT, spot) else 'over' if at == 'bent over' else 'on'} {spot}" if spot else ""
    when = f"from the moment they are drawn back there, all through the {name(it)} going on" if fresh else "the whole shot"
    lock = "held together and then locked together" if fresh else "locked together"
    link = f", held there by the end of the center {HARNESS[tied['gear']][0]} of {p} {name(tied)}, padlocked through the short links " \
           f"between {p} two cuffs" if tied else ""
    out = []
    bent = f"{p} elbows bent and pointing back and {p} locked hands held up together in the middle of {p} back"
    if at == "standing":
        out += [f"{name_} ends the shot standing upright."] if changed else []
        out.append(f"{name_}'s wrists stay {lock} behind {o} {when}, with {bent}{link}." if turns else
                   f"{name_}'s arms stay pulled behind {p} back {when}, with {p} wrists {lock} behind {o}: {p} shoulders are "
                   f"drawn back, both arms run at their full length from {p} shoulders down behind {o}, and {p} hands rest together at "
                   f"the small of {p} back{link}. From the front, both arms angle back from {p} shoulders and {p} body hides {p} "
                   f"hands{' and that padlock' if tied else ''}; seen from the side or from behind, both arms and both hands are in plain "
                   f"view, whole and locked together.")
    else:
        flat = at in ("bent over", "on all fours") or at == "lying" and way == "face down"
        state = {"on all fours": "kneeling bent forward", "bent over": "bent forward",
                 "lying": f"lying {way.replace('her', p)}".strip()}.get(at, at)
        body = f"{p} arms lie along {p} back and {p} wrists stay locked together, {p} hands resting on top of the small of {p} back, " \
               f"in plain view from above and behind" if flat else \
            f"{p} arms are beneath {o} and {p} wrists stay locked together under the small of {p} back" if way == "on her back" else \
            f"{p} arms stay behind {o} and {p} wrists stay locked together, {p} hands resting at the small of {p} back" + (
                f", between {p} back and the back of the {re.search(SEAT, spot).group(0)}" if at == "sitting" and re.search(SEAT, spot)
                else "")
        out.append(f"{name_} ends the shot {state}{where_}: {body}{link}." if changed else
                   f"{name_} {does('stays')} {state}{where_} for the whole shot: {body}{link}.")
    up = lambda q: ((q or {}).get("pose") or "standing") not in LOW
    seat = changed and up(pose0) and at == "sitting" and re.search(SEAT, spot)
    if seat:
        out.append(f"{name_} goes down into {spot} from standing close in front of it with {p} back to it, straight down onto the seat, "
                   f"both arms staying behind {o} the whole way.")
    elif changed and up(pose0) != up(pose1):
        out.append(f"{cap(s)} {does('goes') + ' down' if up(pose0) else does('rises')} with both arms staying behind {o} the whole way.")
    if turns or seat:
        out.append(f"Every turn {call(who)} {does('makes')} in this shot is one slow, continuous turn on the spot in small steps, "
                   f"{p} whole body turning together as one piece, with {bent} the whole way round: side-on, {p} bent arm and elbow "
                   f"show behind {o}; from the front, {p} elbows show behind {p} waist on both sides; from behind, {p} locked hands show "
                   f"in the middle of {p} back.")
    if close:
        y, side, wall = call(close[0]), *close[1:]
        out.append({"behind": f"{cap(y)} is close behind {o}, and {p} locked hands stay behind {o} at the small of {p} back, between "
                              f"{p} back and {y}'s body.",
                    "front": f"{cap(y)} is close in front of {o}, and {p} locked hands stay behind {p} back, on the far side of {p} "
                             f"body from {y}.",
                    "hold": f"{cap(y)} is close against {o}, and {p} locked hands stay behind {o} at the small of {p} back.",
                    "wall": f"{cap(p)} back is to the {wall}, and {p} locked hands stay behind {o}, between {p} back and the "
                            f"{wall}."}[side])
    out.append(f"Whatever {call(who)} {does('does')} in this shot, {s} {does('does')} it with {p} hands locked behind {p} back.")
    return out


def notes(who, items, per, call, pose0, pose1, changed, close, fresh, over, taut, leashed, turns):
    p, o, s, name_ = pron(per), pron(per, 2), pron(per, 0), cap(call(who))
    out = leash(who, items, per, call, taut, leashed)
    wrist = next((i for i in items if i["spot"] == "wrists"), None)
    when = "from the moment they are on" if fresh else "the whole shot"
    hog = any(i["spot"] == "link" for i in items)
    if hog:
        out.append(f"{name_} lies face down {when}, {p} ankles drawn up behind {o} to {p} wrists, and {p} hands lie on top of the small "
                   f"of {p} back.")
    elif wrist and not wrist["to"] and wrist["pose"] == "behind":
        out += hands(who, wrist, per, call, pose0, pose1, changed, close, fresh, next((i for i in items if linked(i)), None), turns)
    elif wrist and not wrist["to"] and wrist["pose"] == "front":
        out.append(f"{name_}'s locked wrists rest together in front of {o} at the height of {p} belly {when}.")
    elif wrist and not wrist["to"] and wrist["pose"] == "above":
        out.append(f"{name_}'s arms stay raised above {p} head {when}, {p} wrists locked together.")
    elif wrist and wrist["to"]:
        high = wrist["pose"] == "above" or (ABOVE.search(wrist["to"]) and wrist["pose"] is None)
        out.append(f"{name_}'s wrists are {'chained' if wrist['gear'] in ('handcuffs', 'chains') else 'tied'} to "
                   f"{middle(wrist['to'])} at one fixed point on it, {f'above {p} head' if high else 'at waist height'}, the same point "
                   f"{when}.")
    if wrist and wrist["gear"] == "handcuffs" and not soft(wrist):
        out.append(f"{name_}'s handcuffs are two rigid steel cuffs joined to each other by their own three short, rigid steel links, "
                   f"which hold them together, side by side, every moment, so {p} wrists stay a few centimetres apart at most.")
    elif wrist and soft(wrist):
        out.append(f"{name_}'s cuffs are {wrist['look']}, buckled around {p} wrists and clipped together, never rigid steel handcuffs.")
    ankle = next((i for i in items if i["spot"] in ("ankles", "knees") and i["gear"] != "spreader bar"), None)
    if ankle:
        out.append(f"{name_}'s {ankle['spot']} stay bound together {when}: whenever {s} {'move' if s == 'they' else 'moves'}, both legs "
                   f"move together as one pair, never apart.")
        out += [f"{cap(p)} {ankle['spot']} are fastened to {middle(ankle['to'])} at one fixed point on it."] if ankle["to"] else []
    out += harness_note(who, items, per, call, over)
    return out


def harness_note(who, items, per, call, over):
    p, o, name_ = pron(per), pron(per, 2), cap(call(who))
    band = next((i for i in items if i["gear"] in HARNESS), None)
    if not band:
        return []
    (c, fixed), (under, onto), (fitted, loose) = HARNESS[band["gear"]][:2], bed(p, over), (over or (None, None))[:2]
    stay = lambda g, place: f" {cap(p)} {g} {'stay' if many(g) else 'stays'} on {place}." if g else ""
    ends = f"fixed to the front of the waist {c} below {p} navel. From there the center {c} runs under {p} crotch, pulled tight" \
           f"{f' over {onto}' if onto else ''}, and up between {p} buttocks to the small of {p} back, where its end is padlocked " \
           f"behind {o}" if linked(band) else \
        f"{fixed} to the front of the waist {c} below {p} navel and to the back of the waist {c} at the small of {p} back. Between " \
        f"those two points the center {c} runs under {p} crotch, pulled tight{f' over {onto}' if onto else ''}, and up between {p} " \
        f"buttocks"
    out = [f"{name_}'s {name(band)} has two parts: a waist {c} {fixed} tight around {p} waist like a belt{under}, and a center {c} "
           f"{ends}, so all of it stays at {p} hips. It stays taut and centered the whole time and never hangs loose, slips to one side "
           f"or sags." + stay(fitted, f"under it the whole time, covering {o}") + stay(loose, "over all of it the whole time")]
    if linked(band):
        out.append(f"{name_}'s handcuffs and {p} {name(band)} are separate restraints, and the only link between them is the padlock at "
                   f"the end of the center {c}, closed through the short links between {p} two cuffs, behind {o} at the small of {p} back.")
    elif len(items) > 1:
        out.append(f"{cap(p)} {name(band)} is a piece of its own, complete in itself, with nothing else fastened to it anywhere.")
    return out


def leash(who, items, per, call, taut, fresh):
    p, o, s = pron(per), pron(per, 2), pron(per, 0)
    it = next((i for i in items if i["gear"] == "leash"), None)
    if not it:
        return []
    unit = "chain" if it["chain"] else "leash"
    out = f"{cap(call(who))}'s leash is one single {unit} of a fixed length, and one end of it is clipped to the ring at the front of " \
          f"{p} collar at {p} throat."
    if it["to"]:
        to = middle(it["to"])
        out += f" Its other end is {'padlocked' if it['chain'] else 'tied'} to {to} at one fixed point there" + (
            ", from the moment it is fastened." if fresh else ", the same point as in the frame this shot opens on.")
        side = " left" if BED.search(it["to"]) or WIDE.search(it["to"]) else ""
        if taut:
            out += f" Pulled taut, the {unit} runs in a straight line from {p} throat to {to}: straight out in front of {o} when {s} " \
                   f"{'face' if s == 'they' else 'faces'} it, over {p}{side} shoulder when {p} back is to it."
        elif ABOVE.search(it["to"]):
            out += f" Between them the {unit} hangs down from {to} to {p} throat in a slack curve."
        elif BED.search(it["to"]):
            out += f" Between them the {unit} hangs slack in one loose curve from the ring at {p} throat to {to}: out in front of {o} " \
                   f"when {s} {'face' if s == 'they' else 'faces'} it, and around the left side of {p} neck and back over {p} left " \
                   f"shoulder when {p} back is to it, its slack lying loose on the bed at {p} left side."
        else:
            out += f" Between them the {unit} hangs slack in one loose curve: from the ring at {p} throat it runs across {p} chest to " \
                   f"the side of {o} nearest {to}, drops beside {o} to the floor and lies loose along the floor to {to}."
        out += f" Its only fixed points are the ring at {p} collar and {to}."
    elif it["by"]:
        out += f" Its other end is in {call(it['by'])}'s hand the whole shot, and the {unit} runs from {p} throat straight to that hand. " \
               f"Its only fixed points are the ring at {p} collar and {call(it['by'])}'s hand."
    else:
        out += f" Its other end hangs free: the {unit} runs from {p} throat across {p} chest to one side of {o} and drops beside {o} " \
               f"to the floor."
    return [out]


def struggling(text, cast, people, held):
    bound = [n for n in cast if held.get(n) and n in people]
    out = []
    for s in sentences(re.sub(r"[\"“][^\"”]*[\"”]", " ", text)):
        if not STRUGGLE.search(s):
            continue
        out += [n for n in bound if re.search(rf"\b{re.escape(n)}\b", s)] or [
            m[0] for word in re.findall(r"\b(?:she|her|he|him|his)\b", s, re.I)
            for m in [[n for n in bound if people[n].get("sex") == ("f" if word.lower() in ("she", "her") else "m")]] if len(m) == 1]
    return list(dict.fromkeys(out))


def struggle(who, items, per, call, pose1):
    p, o, s, name_ = pron(per), pron(per, 2), pron(per, 0), cap(call(who))
    wrist = next((i for i in items if i["spot"] == "wrists"), None)
    ankle = next((i for i in items if i["spot"] in ("ankles", "knees")), None)
    lead = next((i for i in items if i["gear"] == "leash"), None)
    if wrist:
        target = f"the {name(wrist)} " + (f"holding {p} wrists to {middle(wrist['to'])}" if wrist["to"] else
                                         where(wrist, p, o).replace("together ", ""))
        hold = f"The {name(wrist)} hold {p} wrists " + (f"to {middle(wrist['to'])}" if wrist["to"] else "locked together")
    elif ankle:
        target, hold = f"the {name(ankle)} on {p} {ankle['spot']}", f"The {name(ankle)} hold {p} {ankle['spot']} together"
    elif lead:
        target, hold = f"{p} leash", f"{cap(p)} leash holds"
    else:
        target, hold = f"{p} {name(items[0])}", f"{cap(p)} {name(items[0])} {'hold' if many(name(items[0])) else 'holds'}"
    at = (pose1 or {}).get("pose")
    place = {"lying": f"twisting and arching where {s} {'lie' if s == 'they' else 'lies'}",
             "sitting": f"staying in {p} seat", "kneeling": f"staying on {p} knees"}.get(
        at, f"{p} feet braced where {s} {'stand' if s == 'they' else 'stands'}")
    return [f"{name_} struggles for real, desperate to get free: in short, hard jerks with pauses between them, {s} "
            f"{'twist' if s == 'they' else 'twists'} {p} shoulders and {'wrench' if s == 'they' else 'wrenches'} against {target}, "
            f"{place}, {p} face tight with strain. Between tries {s} {'stop' if s == 'they' else 'stops'} for a moment, breathing hard, "
            f"then {'try' if s == 'they' else 'tries'} again. {hold} through every pull."]
