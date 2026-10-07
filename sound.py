# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import re

MAX_SOUNDS = 3
_ON = r"\s+(?:at\s+|on\s+|against\s+|in\s+|with\s+)?(?:[\w'’-]+\s+){0,3}?"
_NAME = r"(?:her|him|them|(?-i:[A-Z][\w'’-]+))"
_GARMENTS = (r"(?:coat|jacket|shirt|t-shirt|dress|skirt|shorts|trousers|pants|jeans|leggings|tights|socks|boots|"
             r"shoes|gloves|top|vest|jumper|sweater|hoodie|blouse|cardigan|scarf|hat|belt|bra|panties|knickers|"
             r"underwear|briefs|boxers|thong|lingerie|stockings|bikini)s?")

SOUND_CUE = re.compile(
    r"\b(?:sounds?|noises?|echo(?:e?s|ing)?|rattl(?:e|es|ing)|clank(?:s|ing)?|clink(?:s|ing)?|creak(?:s|ing)?|"
    r"scrap(?:e|es|ing)|thud(?:s|ding)?|bang(?:s|ing)?|slam(?:s|ming)?|clatter(?:s|ing)?|jingl(?:e|es|ing)|"
    r"squeak(?:s|ing)?|footsteps?|breath(?:s|es|ing)?|pant(?:s|ing)?|gasp(?:s|ing)?|sigh(?:s|ing)?|"
    r"whimper(?:s|ing)?|moan(?:s|ing)?|groan(?:s|ing)?|sob(?:s|bing)?|scream(?:s|ing)?|shout(?:s|ing)?|"
    r"whisper(?:s|ing)?|laugh(?:s|ing|ter)?|hum(?:s|ming)?|buzz(?:es|ing)?|hiss(?:es|ing)?|drip(?:s|ping)?|"
    r"rustl(?:e|es|ing)|click(?:s|ing)?|snap(?:s|ping)?|zip(?:s|ping)?|rings?|ringing|wind|rain|thunder|"
    r"traffic|music|hollow|muffled|reverb|loud(?:ly)?|quietly|faintly|audible|noisy|deafening|"
    r"scuff(?:s|ing|ed)?|crunch(?:es|ing|ed)?|thump(?:s|ing|ed)?|patter(?:s|ing)?|whirr?(?:s|ing)?|"
    r"whine(?:s|d)?|whining|rumbl(?:e|es|ing)|growl(?:s|ing)?|roar(?:s|ing)?|chime(?:s|d)?|ticking|"
    r"knock(?:s|ing)?|tap(?:s|ping)?|whoosh(?:es|ing)?|sizzl(?:e|es|ing))\b", re.I)
_BREATH = re.compile(r"\bbreath(?:s|es|ing)?\b|\bbreathe[sd]?\b", re.I)
_BREATH_TAKEN = re.compile(
    r"\b(?:takes?|took|taking|draws?|drew|drawing|catch(?:es)?|caught|suck(?:s|ed)?|pull(?:s|ed)?|lets?\s+out|"
    r"releases?)\s+(?:in\s+)?(?:a|an|her|his|their|one|another|deep|long|slow|sharp|\s)*breath\b"
    r"|\bwith\s+a\s+breath\b|\ba\s+(?:deep\s+|long\s+|slow\s+|sharp\s+)?breath\b", re.I)

VOCALS = (
    (r"\bwhimper(?:s|ing|ed)?\b", "whimpering"),
    (r"\bsob(?:s|bing|bed)?\b", "sobbing"),
    (r"\bmoan(?:s|ing|ed)?\b", "moaning"),
    (r"\bgroan(?:s|ing|ed)?\b", "groaning"),
    (r"\bscream(?:s|ing|ed)?\b", "screaming"),
    (r"\bwhin(?:e|es|ing|ed)\b", "whining"),
    (r"\bcr(?:y|ies|ied|ying)\s+out\b", "crying out"),
    (r"\bgrunt(?:s|ing|ed)?\b", "grunting"),
    (r"\bsigh(?:s|ing|ed)?\b", "sighing"),
    (r"\bgasp(?:s|ing|ed)?\b", "gasping"),
    (r"\bpant(?:s|ing|ed)?\b", "panting"),
)
_VOCAL_NAMES = {phrase for _, phrase in VOCALS}

FOLEY = (
    (r"\b(?:walk(?:s|ed|ing)?|step(?:s|ped|ping)?|pace[sd]?|enters?|runs?|approach(?:es|ed)?|creep(?:s|ing)?|crept|"
     r"sneak(?:s|ing)?|shuffl(?:e|es|ing)|stumbl(?:e|es|ing)|stagger(?:s|ing)?)\b"
     r"(?!\s+(?:her|his|their)\s+(?:fingers?|hands?|tongue|eyes))", "footsteps"),
    (r"\b(?:open(?:s|ed|ing)?|clos(?:e|es|ed|ing)|shut(?:s|ting)?|slam(?:s|med|ming)?|push(?:es|ed|ing)?|"
     r"pull(?:s|ed|ing)?|kick(?:s|ed|ing)?|swing(?:s|ing)?|swung|bang(?:s|ed|ing)?|knock(?:s|ed|ing)?\s+on|"
     r"yank(?:s|ed|ing)?)\s+(?:open\s+|shut\s+)?(?:the|a|an|her|his|their|its|both|one|that|this)\s+"
     r"(?:[\w'’-]+\s+){0,2}?doors?\b"
     r"|\bdoors?\s+(?:[\w'’-]+\s+)?(?:opens?|closes?|shuts?|slams?|swings?|creaks?|bangs?)\b"
     r"|\b(?:comes?|came|walks?|steps?|bursts?|goes|went|leaves|left)\s+(?:in\s+|out\s+)?through\s+the\s+"
     r"(?:\w+\s+)?door\b", "a door on its hinges"),
    (r"\b(?:drag|rattl|pull|yank|tug|jerk|lift|drop|wrap|loop|unlock|shak|thrash|struggl|strain|fasten|"
     r"padlock|clip)\w*" + _ON + r"chains?\b|\bchains?\s+(?:rattl|clank|clink|drag|jangl|swing|go(?:es)?\s+taut)\w*",
     "chain links dragging"),
    (r"\b(?:snap|lock|click|clos|ratchet|tighten|clamp|squeez)\w*" + _ON + r"(?:hand)?cuffs?\b"
     r"|\b(?:hand)?cuffs?\s+(?:[\w'’-]+\s+){0,2}?(?:snap|lock|click|close|ratchet|tighten)\w*"
     r"|\b(?:hand)?cuff(?:s|ed)\s+" + _NAME + r"\b", "cuffs ratcheting closed"),
    (r"\b(?:rattl|shak|pull|tug|yank|jerk|twist)\w*" + _ON + r"(?:hand)?cuffs?\b"
     r"|\b(?:shackl|manacl)(?:es|ed)\s+" + _NAME + r"\b", "cuffs knocking"),
    (r"\bbolt(?:s|ed|ing)\s+(?:the\s+)?(?:door|gate|window|hatch)\b|\blatch(?:es|ed|ing)\s+(?:the\s+)?\w+"
     r"|\b(?:locks|locking|padlocks|padlocked|padlocking|locked)\s+(?!(?:eyes|gaze|onto)\b)"
     r"(?:it|them|the|a|her|his|their|up)\b", "a lock snapping shut"),
    (r"\b(?:drag(?:s|ged|ging)?|haul(?:s|ed|ing)?|shov(?:e|es|ing)|slid(?:e|es|ing))\b",
     "something dragging on the floor"),
    (r"\b(?:un)?buckl(?:es|ed|ing)\s+(?:up\b|(?:the|her|his|their|it|a|him|them)\b)"
     r"|\bstrap(?:s|ped|ping)\s+" + _NAME + r"\b"
     r"|\b(?:fasten|tighten|clasp|unfasten)\w*\s+(?:the|her|his|their|a)\s+(?:\w+\s+)?"
     r"(?:buckle|strap|harness|belt|collar)s?\b", "a buckle and leather creaking"),
    (r"\b(?:pour(?:s|ed|ing)?|splash(?:es|ed|ing)?)\b|\b(?:runs?|ran|turns?\s+on)\s+the\s+(?:tap|taps|water|bath|shower)\b"
     r"|\bfills?\s+(?:the|a)\s+(?:\w+\s+)?(?:glass|bath|sink|kettle|bucket|tub)\b", "water"),
    (r"\b(?:start(?:s|ed)?|rev(?:s|ved)?|driv(?:e|es|ing)|drove|park(?:s|ed)?)\s+(?:[\w'’-]+\s+){0,2}?"
     r"(?:van|car|engine|truck|motor)\b|\b(?:van|car|engine|truck|motor)\s+(?:[\w'’-]+\s+)?(?:starts?|revs?|roars?|"
     r"idles?|pulls?\s+(?:up|away|in|out|off)|drives?\s+(?:off|away|up|in))\b|\bdrives?\s+(?:off|away)\b",
     "an engine outside"),
    (r"\b(?:tear|tore|rip|peel|pull|yank|unroll|stretch|wrap|wind|wound|press|smooth)\w*\s+(?:[\w'’-]+\s+){0,3}?"
     r"(?:duct\s+|gaffer\s+|packing\s+|masking\s+)?tape\b|\btap(?:es|ed|ing)\s+(?:up\s+)?" + _NAME + r"\b",
     "tape tearing"),
    (r"\b(?:pull|tug|yank|tighten|cinch|knot|ties|tied|tying|wrap|loop|wind|wound|haul|strain|thrash|struggl|"
     r"jerk)\w*" + _ON + r"(?:rope|cord|twine|zip\s?tie)s?\b|\b(?:rope|cord)s?\s+(?:creak|tighten|go(?:es)?\s+tight|"
     r"bite|dig)\w*", "rope creaking as it goes tight"),
    (r"\b(?:pull|tug|take|took|slip|peel|unbutton|button|tear|tore|rip|yank|drop|throw|threw|fold|shrug|puts?|"
     r"remov|straighten|smooth|adjust|lift|kick)\w*\s+(?:[\w'’-]+\s+){0,3}?" + _GARMENTS + r"\b", "fabric rustling"),
    (r"\b(?:jingl|rattl|fumbl|drop|pull|take|took|turn|hand|toss|throw|threw|pocket|grab|fish|dangl|pick)\w*\s+"
     r"(?:[\w'’-]+\s+){0,3}?keys?\b|\bkeys?\s+(?:jingl|rattl|turn|clink)\w*", "keys on a ring"),
    (r"\b(?:drops?|dropped|throw(?:s|n)?|threw|toss(?:es|ed)?)\b", "something landing"),
    (r"\b(?:smack(?:s|ed)?|slap(?:s|ped)?|hits?|strikes?|struck)\b", "a sharp impact"),
    (r"\b(?:thrash|struggl|writh|strain|pull|tug|yank|twist|jerk|fight)\w*\s+(?:against|at|in|on)\s+"
     r"(?:the|her|his|their)\s+(?:\w+\s+)?(?:cuffs?|handcuffs?|shackles?|manacles?|chains?|ropes?|cords?|straps?|"
     r"restraints?|bindings?|ties|tape|harness|collar)\b", "restraints pulling taut"),
    (r"\b(?:un)?zip(?:s|ped|ping)\b|\b(?:pull|tug|yank|draw|run|slid)\w*\s+(?:[\w'’-]+\s+){0,2}?zipper\b",
     "a zip running"),
    (r"\b(?:rip|tear|tore|pull|open|undo|undid)\w*\s+(?:[\w'’-]+\s+){0,2}?velcro\b|\bvelcro\s+(?:straps?\s+)?(?:rips?|tears?)\b",
     "velcro tearing open"),
    (r"\b(?:thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|writh(?:e|es|ing|ed)|strain(?:s|ing|ed)?|"
     r"trembl(?:e|es|ing|ed)|shiver(?:s|ed|ing)?|wakes?\s+up|woke|pant(?:s|ing)?|breath(?:es|ing)?)\b", "breathing"),
)
_SUPERSEDES = {"cuffs ratcheting closed": ("cuffs knocking",)}

AMBIENT = (
    (r"\brain(?:ing|y)?\b|\bdownpour\b|\bdrizzl", "rain against the glass"),
    (r"\bstorm|\bthunder", "a storm somewhere outside"),
    (r"\bwind(?:y)?\b|\bgale\b", "wind against the building"),
    (r"\bbeach\b|\bsea\b|\bocean\b|\bshore\b", "the sea a long way off"),
    (r"\bforest\b|\bwoods?\b", "wind in the trees"),
    (r"\bgarden\b|\byard\b|\bpark\b", "birdsong"),
    (r"\bstreet\b|\broad\b|\btraffic\b|\bcity\b|\bpavement\b", "traffic somewhere off the street"),
    (r"\bcar\b|\bvan\b|\btruck\b|\bdriving\b", "an engine idling"),
    (r"\bkitchen\b", "a fridge humming"),
    (r"\bbathroom\b|\bshower\b", "water moving in the pipes"),
    (r"\bnursery\b|\bbaby\b|\bcrib\b", "a clock ticking"),
    (r"\bbedroom\b", "the quiet of a bedroom"),
    (r"\boffice\b|\bstudy\b", "a computer fan"),
    (r"\bworkshop\b|\bgarage\b|\bfactory\b", "a strip light humming"),
    (r"\bbasement\b|\bcellar\b|\bboiler\b", "a low hum off the strip light"),
    (r"\bhospital\b|\bward\b|\bclinic\b", "a monitor somewhere down the corridor"),
    (r"\bcafe\b|\bbar\b|\brestaurant\b|\bpub\b", "cutlery and moving chairs"),
    (r"\bschool\b|\bclassroom\b", "a corridor beyond the door"),
    (r"\bchurch\b|\bhall\b", "the air of a large empty room"),
    (r"\bstairs?\b|\bstairwell\b|\bhallway\b|\bcorridor\b", "the hollow quiet of a hallway"),
    (r"\bnight\b|\blate evening\b", "the quiet of a night"),
    (r"\bhome\b|\bhouse\b|\bflat\b|\bapartment\b|\bliving room\b|\blounge\b|\bindoors?\b|\broom\b",
     "the quiet of a house"),
)

_MOUTH = re.compile(r"\bgag(?:ged)?\b|\bmouth\b|\blips\b|\bmuzzle\b", re.I)
_HELD_OPEN = re.compile(r"\b(?:ball|ring|bit|spider|o-ring)[\s-]*gag\b|\bstuffed\b|\b(?:rag|cloth|sock|wad)\b", re.I)
_GAG_NOUN = re.compile(r"\b(tape|gag|cloth|rag|sock|scarf|bandana|muzzle|wad|hand)\b", re.I)


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def described(text):
    hits = [m.group(0) for m in SOUND_CUE.finditer(text or "")]
    return bool(hits) and not (all(_BREATH.fullmatch(h) for h in hits) and _BREATH_TAKEN.search(text))


def vocal(text):
    return any(re.search(p, text or "", re.I) for p, _ in VOCALS)


def foley(text):
    out = []
    for pat, phrase in VOCALS + FOLEY:
        if len(out) < MAX_SOUNDS and phrase not in out and re.search(pat, text or "", re.I):
            out.append(phrase)
    for keep, gone in _SUPERSEDES.items():
        if keep in out:
            out = [p for p in out if p not in gone]
    if any(p in _VOCAL_NAMES for p in out):
        out = [p for p in out if p != "breathing"]
    return [] if all(p in _VOCAL_NAMES for p in out) else out


def ambience(*texts):
    joined = " ".join(t for t in texts if t)
    return next((phrase for pat, phrase in AMBIENT if re.search(pat, joined, re.I)), "")


_MUSIC = re.compile(r"\b(?:music(?:al)?|songs?|sings?|singing|sang|radio|stereo|piano|guitar|violin|drums?|band|"
                    r"orchestra|melod(?:y|ies|ic)|tunes?|humm(?:ed|ing)|hums? a|whistl(?:e|es|ed|ing)|jukebox|"
                    r"concert|playlist)\b", re.I)
NO_MUSIC = "There is no background music."


def sound_line(beat, scene, speech):
    quiet = "" if _MUSIC.search(f"{beat} {scene}") else NO_MUSIC
    if speech:
        return quiet, False
    amb = ambience(scene) or ambience(beat)
    if described(beat):
        line = (f"The only sounds are the ones this beat describes, with {amb} under them." if amb
                else "The only sounds are the ones this beat describes.")
        return f"{line} {quiet}".strip(), True
    heard = foley(beat)
    if not heard:
        return quiet, False
    heard += [amb] if amb else []
    return f"The only sound{'s are' if len(heard) > 1 else ' is'} {_join(heard)}. {quiet}".strip(), True


def held_open(item):
    return bool(_HELD_OPEN.search(item or ""))


def mouth_item(items):
    return next((it for it in items if _MOUTH.search(it)), "")


def muffled(who, item):
    m = _GAG_NOUN.search(item)
    noun = m.group(1).lower() if m else "gag"
    if held_open(item):
        return f"Every sound from {who} comes out muffled, the mouth held open around the {noun}."
    return f"Every sound from {who} comes out muffled, the lips held shut under the {noun}."
