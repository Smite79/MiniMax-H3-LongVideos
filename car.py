# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79
# SPDX-License-Identifier: GPL-3.0-or-later. See LICENSE.

import re

try:
    from .words import sentences
except ImportError:
    from words import sentences

CAR = re.compile(r"\b(?:car|sedan|saloon|hatchback|suv|jeep|coupe|vehicle|van|minivan|taxi)\b", re.I)
VAN = re.compile(r"\b(?:van|minivan)\b", re.I)
HATCH = re.compile(r"\b(?:hatchback|hatch|suv|tailgate|liftgate)\b", re.I)
OWN = r"(?:['’]?s)?"
DOOR = re.compile(rf"(?P<rear>\b(?:back|rear)(?:\s+passenger)?(?:[\s-]+side)?\s+doors?|\b(?:back|rear)[\s-]+seat\s+doors?|"
                  rf"\bsliding(?:\s+side)?\s+door)|"
                  rf"(?P<driver>\bdriver{OWN}(?:[\s-]+side)?(?:\s+front)?\s+door|\bdoor\s+on\s+the\s+driver{OWN}\s+side)|"
                  rf"(?P<passenger>\b(?:front\s+)?passenger{OWN}(?:[\s-]+side)?\s+door|\bdoor\s+on\s+the\s+passenger{OWN}\s+side)|"
                  rf"(?P<trunk>\btrunk\b|\b(?:the|car['’]s|its)\s+boot\b|\bboot\s+of\s+the\b|\bhatch\b|\btailgate\b|\bliftgate\b)|"
                  rf"(?P<door>\b(?:car|van)\s+door\b|\b(?:side\s+)?door\s+of\s+the\s+(?:car|van)\b|"
                  rf"\bthe\s+(?:side\s+)?door\b)", re.I)
SEAT = (("rear", r"back\s*seat|rear\s+seat|back\s+of\s+the\s+(?:car|van)|into\s+the\s+back\b"),
        ("passenger", r"passenger\s+seat|front\s+seat"), ("driver", rf"driver{OWN}\s+seat|behind\s+the\s+wheel|steering\s+wheel"))
INTO = re.compile(r"\b(?:gets?|climbs?|slides?|hops?|steps?)\s+(?:in|into|out)\b|\b(?:seat|seats|wheel)\b", re.I)
OPEN = re.compile(r"\b(?:opens?|pops?)\b|\b(?:pulls?|slides?|swings?|throws?|yanks?|flings?|is|stands?|stays?|hangs?)\s+(?:\w+\s+)?"
                  r"open\b", re.I)
SHUT = re.compile(r"\b(?:closes?|shuts?|slams?)\b|\b(?:is|stays?)\s+(?:shut|closed)\b", re.I)
AWAY = re.compile(r"\bdrives?\s+(?:off|away)\b", re.I)
GOES = re.compile(r"\b(?:into|out\s+of|from)\s+the\s+(?P<k>back|rear|front|passenger|driver['’]?s?)\s+seat\b", re.I)
SEATED = {"back": "rear", "rear": "rear", "front": "passenger", "passenger": "passenger"}
LAYOUT = {"car": "The car has four side doors, two on each side: a front door beside each front seat and a rear door right behind it "
                 "beside the back seat. Each side door is set in the side of the car, hinged at its front edge, and swings straight out "
                 "from the side.",
          "van": "The van has a front door on each side, beside the driver's seat and the front passenger seat, and a sliding door in "
                 "its side right behind the front passenger door."}
PLACE = {"driver": "The driver's door is the front door on the same side as the steering wheel, beside the driver's seat.",
         "passenger": "The front passenger door is the front door on the side away from the steering wheel, beside the front passenger "
                      "seat.",
         "rear": "The back door is the rear door on the side nearest them, right behind the front door on that side, and it opens onto "
                 "the back seat.",
         "side": "The {v} door is the front door on the side nearest them, set in the side of the {v}.",
         "trunk": "The trunk is at the very back of the car, behind the back seat, under a lid hinged at its front edge that lifts "
                  "straight up.",
         "van rear": "The sliding door is in the side of the van, right behind the front passenger door, and it slides back along the "
                     "side of the van to open onto the back.",
         "hatch": "The hatch is the whole back of the car, hinged along the back edge of the roof, and it lifts up to open onto the "
                  "space behind the back seat.",
         "van trunk": "The van's back doors are a pair at the very back of the van, each hinged at its outer edge, and they swing out "
                      "to the sides."}
NAME = {"driver": "the driver's door", "passenger": "the front passenger door", "rear": "the back door", "side": "the {v} door",
        "trunk": "the trunk lid", "van rear": "the sliding door", "hatch": "the hatch", "van trunk": "the van's back doors"}


def kind(k, v, hatch):
    return f"van {k}" if v == "van" and k in ("rear", "trunk") else "hatch" if k == "trunk" and hatch else k


def said(before, after):
    verbs = [(x.start(), "open") for x in OPEN.finditer(before)] + [(x.start(), "shut") for x in SHUT.finditer(before)]
    return "shut" if re.match(r"\s+(?:\w+\s+)?(?:shut|closed)\b", after, re.I) else "open" if re.match(
        r"\s+(?:\w+\s+)?open\b", after, re.I) else max(verbs)[1] if verbs else None


def notes(text, scene, doors):
    v = "van" if VAN.search(scene + " " + text) else "car"
    if not CAR.search(scene + " " + text):
        return []
    if AWAY.search(text):
        doors.clear()
    left = [k for k, st in doors.items() if st == "open"]
    hatch, used, last = bool(HATCH.search(scene + " " + text)), [], left[0] if len(left) == 1 else None
    for s, lead in [(x, True) for x in sentences(scene)] + [(x, False) for x in sentences(text)]:
        end = 0
        for m in DOOR.finditer(s):
            k = m.lastgroup
            if k == "door":
                if not (CAR.search(s) or INTO.search(s) or last):
                    continue
                k = "rear" if v == "van" and "side" in m.group(0).lower() else last or next(
                    (x for x, rx in SEAT if re.search(rf"\b(?:{rx})", text, re.I)), "side")
            k = last = kind(k, v, hatch)
            used += [k] if k not in used else []
            tail = re.match(r"\s+(?:\w+\s+)?(?:shut|closed|open)\b", s[m.end():], re.I)
            state = said(s[max(end, m.start() - 40):m.start()], tail.group(0) if tail else "")
            doors[k] = (doors.get(k) or state) if lead else state or doors.get(k)
            end = m.end() + (tail.end() if tail else 0)
        seat = GOES.search(s) if not end and not lead else None
        if seat:
            last = kind(SEATED.get(seat["k"].lower(), "driver"), v, hatch)
            used += [last] if last not in used else []
        it = re.search(rf"(?:{OPEN.pattern}|{SHUT.pattern})(?=\s+it\b)", s[end:], re.I) if last and end and not lead else None
        doors[last] = said(s[end:end + it.end()], "") if it else doors.get(last)
    out = ([LAYOUT[v]] if any(k not in ("trunk", "hatch", "van trunk") for k in used) else []) + [PLACE[k].format(v=v) for k in used]
    named = list(dict.fromkeys(used + [k for k, st in doors.items() if st == "open"]))
    out += [f"Apart from {' and '.join(NAME[k].format(v=v) for k in named)}, the {v} stays closed the whole shot, with every other "
            f"door shut."] if named else []
    out += [f"{NAME[k].format(v=v)[0].upper() + NAME[k].format(v=v)[1:]} {'is' if k != 'van trunk' else 'are'} still standing open, "
            f"exactly as before." for k, st in doors.items() if st == "open" and k not in used]
    out += [f"By the last frame, {NAME[k].format(v=v)} {'is' if k != 'van trunk' else 'are'} {doors[k]}." for k in used if doors.get(k)]
    return out
