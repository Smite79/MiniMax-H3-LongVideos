# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""The prompt engine: scene state, beat by beat.

WHY THIS REPLACED WHAT WAS HERE BEFORE
--------------------------------------
The old engine was about sixty independent readers, each searching the beat for
its own thing and each appending its own sentence to the shot. `limb_anchor`
found "behind the back", `held_part` found "neck", and neither could see the
other, so the shot went out saying "holding the neck behind the back" -- a neck
behind a back. `hardware_named` returned one item, so a beat that cuffed the
wrists and locked on a collar recorded the collar and the handcuffs were never
mentioned again; hardware nobody mentions is hardware the model stops drawing,
and that read as her breaking out of them. `_PLACE` contained "door", so "Ana
looks at the door" moved the camera into a door.

Every one of those is the same failure: a clause derived on its own, with nothing
holding the facts together and nothing able to notice a contradiction.

So the shape here is different. A beat is parsed ONCE into events. Events update
one explicit state. The state renders ONE paragraph. There is exactly one place
that knows what is on whom, one that knows where anybody is, and one that turns
that into English -- so a contradiction is a bug in a value you can print, not an
emergent property of sixty regexes that never met.

WHAT THE STATE GUARANTEES
    - hardware is remembered per person, with the part it holds, the position it
      holds that part in, and what it is anchored to. All four together or not at
      all, so they cannot disagree.
    - every piece of hardware on a person is named in every shot until something
      takes it off. Not the newest, not the most specific: all of it.
    - a garment is on, off, or displaced, and the shot that changes it says both
      ends of the change.
    - a place is a room somebody can be in. A door is not a room.
"""

import re


HARDWARE = (
    (r"hand\s?cuffs?|handcuffed", "handcuffs", "wrists"),
    (r"leg\s?irons?", "leg irons", "ankles"),
    (r"ankle\s?irons?", "ankle irons", "ankles"),
    (r"wrist\s?irons?", "wrist irons", "wrists"),
    (r"tethers?|tethered", "tether", "wrists"),
    (r"(?:braided\s+)?(?:steel|wire)\s+cables?|(?:steel|baling)\s+wire", "steel cable", "wrists"),
    (r"hobbles?|hobbled", "hobble", "ankles"),
    (r"(?:bike|bicycle)\s+locks?|[ud]-?locks?", "bike lock", "wrists"),
    (r"cling\s?film|plastic\s+wrap", "cling film", "wrists"),
    (r"chastity\s+belts?", "chastity belt", "hips"),
    (r"(?:steel|metal|iron|chrome|brass|locking|lockable|restraint)\s+belts?",
     "steel belt", "waist"),
    (r"ankle\s+(?:cuffs?|chains?|straps?)", "ankle cuffs", "ankles"),
    (r"shackles?|shackled", "shackles", "ankles"),
    (r"manacles?|manacled", "manacles", "wrists"),
    (r"zip\s?ties?|cable\s?ties?", "zip ties", "wrists"),
    (r"collars?|chokers?|collared", "collar", "neck"),
    (r"leash(?:es)?|leashed", "leash", "neck"),
    (r"gags?|gagged", "gag", "mouth"),
    (r"blindfolds?|blindfolded", "blindfold", "eyes"),
    (r"harness(?:es)?", "harness", "body"),
    (r"spreader\s+bars?", "spreader bar", "ankles"),
    (r"straitjackets?", "straitjacket", "arms"),
    (r"ropes?|cords?|twine", "rope", "wrists"),
    (r"straps?", "straps", "wrists"),
    (r"chains?", "chain", "wrists"),
    (r"cuffs?|cuffed", "cuffs", "wrists"),
    (r"tapes?|taped|taping", "tape", "wrists"),
)
_ADJ = (r"(?:[A-Za-z]+-[A-Za-z]+|"
        # materials
        r"steel|stainless|iron|metal|metallic|nickel|chrome|chromed|brass|"
        r"bronze|copper|pewter|gunmetal|titanium|alumini?um|gold|golden|silver|"
        r"platinum|leather|pleather|suede|velvet|satin|silk|lace|mesh|nylon|"
        r"plastic|rubber|latex|silicone|neoprene|vinyl|pvc|canvas|denim|cotton|"
        r"wool|woollen|linen|rope|wood|wooden|ceramic|glass|resin|"
        # finishes
        r"mirrored|mirror|polished|brushed|burnished|hammered|plated|anodi[sz]ed|"
        r"matte|matt|gloss|glossy|shiny|dull|tempered|hardened|welded|riveted|"
        r"studded|spiked|lined|padded|quilted|ribbed|textured|smooth|"
        # colours
        r"black|white|red|blue|green|grey|gray|brown|pink|purple|tan|cream|navy|"
        r"crimson|scarlet|ivory|amber|olive|"
        # size and build
        r"heavy|light|thin|thick|wide|narrow|short|long|small|large|broad|slim|"
        r"duct|packing|electrical|zip)")

PARTS = (
    (r"wrists?", "wrists"),
    (r"ankles?", "ankles"),
    (r"necks?|throats?", "neck"),
    (r"mouths?", "mouth"),
    (r"eyes?", "eyes"),
    (r"elbows?", "elbows"),
    (r"knees?", "knees"),
    (r"thighs?", "thighs"),
    (r"waists?", "waist"),
    (r"arms?", "arms"),
    (r"legs?", "legs"),
    (r"hands?", "hands"),
    (r"feet|foot", "feet"),
)
# Things that are not restraints until somebody ties with them: "ties her wrists to
# the bedpost WITH silk scarves". Read only as the instrument of a tying verb, so a
# scarf somebody wears stays a garment. Unread, the scarves -- and the bedpost with
# them -- were never tracked and dropped out of every later shot. REPORTED.
SOFT_TIES = (
    (r"scar(?:f|ves)", "scarves"), (r"stockings?|pantyhose|tights", "stockings"),
    (r"belts?", "belt"), (r"neck\s?ties?|ties?", "ties"), (r"(?:bed\s?)?sheets?", "sheet"),
    (r"bandann?as?", "bandana"), (r"(?:shoe)?laces?", "laces"),
)
_RESTRAINABLE = re.compile(r"\b(?:wrists?|ankles?|hands?|arms?|legs?|feet|foot|knees?|"
                           r"thumbs?|elbows?|thighs?|neck|throat|mouth|eyes)\b", re.I)
_SOFT_TIE = re.compile(
    r"\b(?:with|using)\s+(?:(?:a|an|the|her|his|their|two|some|a\s+pair\s+of|pairs\s+of|"
    r"silk|satin|leather|nylon|cotton|lace|black|red|white|long|torn|knotted)\s+){0,3}("
    + "|".join(p for p, _c in SOFT_TIES) + r")\b", re.I)

PART_VARIES = frozenset({"scarves", "stockings", "belt", "ties", "sheet", "bandana", "laces",
                         "chain", "rope", "straps", "tape", "steel cable", "tether",
                         "steel belt", "cling film"})

REGION_OF = (
    (r"shorts|trousers|jeans|slacks|chinos|skirt|kilt|leggings|joggers|tights|"
     r"pantyhose|jeggings|culottes|tracksuit\s+bottoms|sweatpants|pants|"
     r"dress|gown|dressing-?gown|robe|kimono|sari|nightdress|nightie|nightgown|"
     r"jumpsuit|romper|playsuit|catsuit|bodysuit|leotard|onesie|"
     r"overalls|dungarees|boilersuit|coveralls|swimsuit|"
     r"panties|knickers|thong|g-?string|briefs|boxers|underwear|undies|"
     r"jockstrap|loincloth", "legs",
     "The legs are bare from the hip down"),
    (r"socks|stockings|hold-?ups|boots|shoes|trainers|sneakers|sandals|heels|"
     r"slippers|loafers|brogues|clogs|espadrilles|flats|moccasins|pumps|wedges",
     "feet", "The feet and ankles are bare"),
    (r"top|shirt|blouse|t-?shirt|tee|jumper|sweater|sweatshirt|hoodie|cardigan|"
     r"jacket|coat|tunic|bra|bralette|camisole|vest|"
     r"pullover|overcoat|raincoat|peacoat|windbreaker|blazer|anorak|parka|gilet|"
     # ...the whole-outfit garments again, which cover this half too.
     r"dress|gown|dressing-?gown|robe|kimono|sari|nightdress|nightie|nightgown|"
     r"jumpsuit|romper|playsuit|catsuit|bodysuit|leotard|onesie|"
     r"overalls|dungarees|boilersuit|coveralls|swimsuit",
     "torso",
     "The chest, shoulders and arms are bare skin"),
    (r"gloves|mittens", "hands", "The hands are bare"),
    (r"hat|cap|beanie|beret|headscarf|headband|hood", "head",
     "The head is bare, the hair uncovered"),
)
NUDITY = (
    (r"topless|bare-?chested|bare-?breasted|shirtless|"
     r"stripped\s+to\s+the\s+waist|strips\s+to\s+the\s+waist", ("torso",)),
    (r"bottomless|bare\s+from\s+the\s+waist\s+down", ("legs",)),
    (r"naked(?!\s+(?:eye|flame))|nude|in\s+the\s+nude|wearing\s+nothing|"
     r"with\s+no\s+clothes|stark\s+naked", ("torso", "legs", "feet")),
)

POSITIONS = (
    (r"behind\s+(?:her|his|their|the)\s+backs?", "behind the back"),
    (r"(?:above|over)\s+(?:her|his|their|the)\s+heads?|overhead", "above the head"),
    (r"in\s+front\s+of\s+(?:her|his|their)\s+(?:body|chest|waist)",
     "in front of the body"),
    (r"(?:out\s+)?to\s+the\s+sides?|spread\s+wide", "out to the sides"),
    (r"at\s+(?:her|his|their|the)\s+waists?", "at the waist"),
)

ANCHORS = (r"walls?|floors?|grounds?|ceilings?|pillars?|columns?|posts?|rails?|"
           r"railings?|bars?|rings?|hooks?|pipes?|radiators?|beams?|girders?|"
           r"struts?|stakes?|eye\s?bolts?|brackets?|cages?|fences?|grates?|"
           r"grilles?|bed\s?frames?|bed\s?posts?|headboards?|bedsteads?|beds?|"
           r"bunks?|benches?|chairs?|tables?|desks?|ladders?|anchors?|loops?|"
           # The furniture a scene is BUILT round. Missing, "straps her to the cross"
           # fastened her to nothing, and every later shot dropped where she was held.
           # REPORTED as bondage equipment and its placement dropped from the beats.
           r"(?:st\.?\s+andrew'?s\s+)?crosse?s?|x-?frames?|frames?|poles?|racks?|"
           r"stocks|pillor(?:y|ies)|sawhorses?|trees?|door\s?handles?|doors?|"
           r"handles?|banisters?|bannisters?|stair\s?rails?|trellis(?:es)?")

_DET = (r"(?<!\bthe\s)(?<!\ba\s)(?<!\ban\s)(?<!\bthese\s)(?<!\bthose\s)"
        r"(?<!\btwo\s)(?<!\bsome\s)(?<!\bmore\s)(?<!\bhis\s)(?<!\bher\s)"
        r"(?<!\bof\s)(?<!\bpair\s)(?<!\bset\s)")
APPLY_VERB = (
    r"(?:handcuffed|cuffed|chained|shackled|manacled|locked|padlocked|fastened|"
    r"secured|tethered|bound|tied|strapped|clipped|hooked|bolted|attached|"
    r"anchored|leashed|roped|gagged|blindfolded|collared|taped|trussed|lashed|"
    r"buckled|fettered|"
    r"hog-?(?:ties|tie|tied|tying|cuffs|cuffed|chains|chained)|"
    r"truss(?:es|ing)|zip[-\s]?(?:ties?|tied|tying)|cable[-\s]?(?:ties?|tied|tying)|"
    # A verb only where no determiner makes it the noun: "strains against THE manacles"
    # was read as manacles going on, and the new bare record replaced the iron manacles
    # chained to the beam -- anchor and all.
    + _DET + r"manacles|hobbl(?:ed|ing)|" + _DET + r"hobbles|" + _DET + r"fetters|"
    r"pinion(?:s|ed|ing)?|"
    r"restrain(?:s|ed|ing)|immobili[sz]e[sd]?|"
    r"handcuffing|cuffing|chaining|locking|fastening|securing|tethering|tying|"
    r"strapping|clipping|bolting|attaching|gagging|blindfolding|collaring|"
    r"taping|buckling|binding|shackling|"
    r"(?:puts?|putting|slips?|slipped|snaps?|snapped|clicks?|clicked|clamps?|"
    r"clamped)(?:\s+\S+){0,4}?\s+(?:on|onto|around|shut|closed)"
    # ...but not onto furniture: "puts the rope on the table" restrains nobody.
    r"(?!\s+(?:the|a|an)\s+(?:table|floor|bed|chair|shelf|counter|desk|bench|ground|"
    r"dresser|nightstand|sofa|couch|stool|tray|side|top)\b)|"
    # Tape is PRESSED, STUCK or SLAPPED on, and it goes OVER or ACROSS a mouth. None of
    # that was an applying verb, so "presses a strip of duct tape over her mouth" put
    # nothing on anybody as far as the state knew -- and with nothing tracked, the
    # next shot said "every restraint stays fastened" and never named the tape. It came
    # off by the next beat. REPORTED.
    r"(?:puts?|putting|places?|placed|placing|presses|pressed|pressing|sticks?|stuck|"
    r"sticking|slaps?|slapped|slapping|smooths?|smoothed|smoothing|plasters?|"
    r"plastered|plastering)(?:\s+\S+){0,5}?\s+(?:over|across|on|onto)\s+"
    r"(?:[\w'’]+\s+){0,2}?(?:mouth|lips|eyes|face|wrists?|ankles?|hands?)|"
    r"(?:tapes|cuffs|chains|straps|binds|ties|locks|shackles|clips|hooks|wraps)\s+"
    r"(?:\w+\s+){0,2}?(?:her|his|their|the)\s+(?:\w+\s+){0,2}?"
    r"(?:wrists?|ankles?|hands?|feet|legs?|arms?|neck|throat|waist|knees?|thumbs?)|"
    r"(?:loops?|looped|looping|wraps?|wrapped|wrapping|winds?|wound|winding|"
    r"coils?|coiled|coiling|threads?|threaded|threading|passes|passed|passing|"
    r"runs|ran|running|cinch(?:es|ed|ing)?|knots?|knotted|laces?|laced|"
    r"tighten(?:s|ed|ing)?|"
    r"hitch(?:es|ed|ing)?|slings?|slung)(?:\s+\S+){0,5}?\s+"
    r"(?:around|round|through|under|over|about|behind|between)"
    r"(?:\s+\S+){0,3}?\s+(?:" + "|".join(p for p, _n in PARTS)
    + r"|backs?|hips?|shoulders?|chests?|torsos?|heads?|thumbs?|bod(?:y|ies)|"
      r"her|him|them|herself|himself|themselves)\b|"
    r"closes?\s+around|clicks?\s+shut)"
    r"|" + _DET + r"(?:handcuffs|cuffs|chains|shackles|locks|padlocks|fastens|"
    r"secures|tethers|ties|straps|clips|hooks|bolts|attaches|anchors|leashes|"
    r"ropes|gags|blindfolds|collars|tapes|buckles|binds)")
RELEASE_VERB = (
    r"(?:unlocks?|unlocked|unlocking|uncuffs?|uncuffed|unbinds?|unbound|"
    r"unties?|untied|untying|unbuckles?|unbuckled|unstraps?|unstrapped|"
    r"unclips?|unclipped|unfastens?|unfastened|unshackles?|unshackled|"
    r"ungags?|ungagged|unchains?|unchained|releases?|released|releasing|"
    r"frees?|freed|freeing|cuts?\s+(?:off|away|free)|cut\s+(?:off|away|free)|"
    r"slips?\s+off|slipped\s+off|takes?\s+off|took\s+off|pulls?\s+off|"
    r"lifts?\s+(?:off|away)|removes?|removed|undoes|undid|opens?\s+the)")

PLACES = (r"hallway|hall|corridor|passage|landing|stairwell|staircase|stairs|"
          r"steps|bedroom|bathroom|washroom|kitchen|living\s+room|lounge|"
          r"dining\s+room|locker\s+rooms?|changing\s+rooms?|dressing\s+rooms?|"
          r"waiting\s+rooms?|utility\s+rooms?|gymnasium|gym|classroom|library|"
          r"cafeteria|canteen|reception|laundry|pantry|sauna|balcony|terrace|"
          r"rooftop|elevator|court|pool|showers|shower|store|shop|studio|"
          r"study|office|garage|basement|cellar|attic|loft|porch|"
          r"veranda|garden|yard|driveway|street|alley|car\s?park|lobby|foyer|"
          r"doorway|cell|warehouse|barn|shed|van|truck|room")
PLACE_ALSO_A_VERB = {"steps", "landing", "study", "lounge", "garage", "porch",
                     "court", "pool", "shower", "showers", "store", "shop",
                     "studio", "reception"}
_ROOM_MOD = (r"(?:(?!(?:of|the|an?|and|or|to|in|into|from|with|on|at|by|for|her|"
             r"his|their|its|my|our|your)\b)[A-Za-z][A-Za-z-]*\s+){0,3}?")
# A place word naming an OBJECT: "the laundry basket", "an office chair", "a pool of
# light", "her studio flat". Nobody is IN that room, and reading one as a room cut
# the shot to it -- a fresh start, the scenery redrawn, in a scene that never moved.
# A door or a table belongs to its room: "at the kitchen door" still puts them there.
PLACE_NOT_AN_OBJECT = (r"(?!\s+(?:baskets?|hampers?|bags?|piles?|heaps?|chairs?|stools?|"
                       r"towels?|lamps?|lights?|phones?|keys?|flat|apartment|shoes|"
                       r"shorts|clothes|gear|kit|hose|gel|cues?|sale|party)\b)"
                       r"(?!(?<=pool)\s+of\b)")

GARMENT_PHRASES = (r"chastity[\s-]*(?:belts?|devices?|cages?)|g[\s-]?strings?|"
                   r"boxer[\s-]+shorts?|sports?[\s-]+bras?|body[\s-]?suits?|"
                   r"suspender[\s-]+belts?|garter[\s-]+belts?")
GARMENT_WORDS = (r"shirt|blouse|top|t-shirt|tshirt|dressing-?gown|vest|waistcoat|"
                 r"gilet|jumper|sweater|"
                 r"sweatshirt|hoodie|cardigan|jacket|blazer|coat|anorak|parka|"
                 r"poncho|cloak|dress|gown|skirt|kilt|sari|kimono|trousers|pants|"
                 r"jeans|slacks|chinos|shorts|leggings|joggers|tracksuit|tights|"
                 r"stockings|socks|shoes|boots|trainers|sneakers|sandals|heels|"
                 r"pullover|overcoat|raincoat|peacoat|windbreaker|sweatpants|tee|"
                 r"loafers|brogues|clogs|espadrilles|flats|moccasins|pumps|wedges|"
                 r"beret|"
                 r"culottes|hold-?ups|jeggings|pantyhose|tunic|headscarf|headband|hood|"
                 r"nightgown|playsuit|catsuit|bodysuit|leotard|onesie|boilersuit|"
                 r"coveralls|"
                 r"slippers|gloves|mittens|scarf|hat|cap|beanie|tie|apron|"
                 r"overalls|dungarees|uniform|robe|pyjamas|pajamas|nightdress|"
                 r"nightie|swimsuit|bikini|trunks|romper|jumpsuit|"
                 r"bra|bralette|brassiere|camisole|undershirt|corset|bustier|"
                 r"slip|lingerie|knickers|panties|thong|briefs|boxers|underwear|"
                 r"undies|undercloth|jockstrap|loincloth|nappy|diaper|"
                 r"clothes|clothing|outfit")
_GARMENT = GARMENT_PHRASES + r"|" + GARMENT_WORDS

_GAP = r"(?:\s+\S+){0,4}?\s+"
# Shared with the removal readers in sampler.py: the sampler imports this one.
_STRIP_VERB = (r"take[sn]?|took|taking|pull(?:s|ed|ing)?|peel(?:s|ed|ing)?|"
               r"strip(?:s|ped|ping)?|cut(?:s|ting)?|rip(?:s|ped|ping)?|tear[s]?|tore|"
               r"slip(?:s|ped)?|shrug(?:s|ged)?|yank(?:s|ed)?|tug(?:s|ged)?|"
               r"toss(?:es|ed)?|throw[s]?|threw|"
               r"kick(?:s|ed|ing)?|step(?:s|ped|ping)?|lift(?:s|ed|ing)?|"
               r"slide[s]?|slid|wriggle[sd]?|wiggle[sd]?|work(?:s|ed)?")
_TRAILING_VERB = (r"take[sn]?|took|taking|pull(?:s|ed|ing)?|peel(?:s|ed|ing)?|"
                  r"strip(?:s|ped|ping)?|cut(?:s|ting)?|rip(?:s|ped|ping)?|tear[s]?|"
                  r"tore|slip(?:s|ped)?|shrug(?:s|ged)?|yank(?:s|ed)?|tug(?:s|ged)?|"
                  r"toss(?:es|ed)?|throw[s]?|threw|kick(?:s|ed|ing)?|"
                  r"work(?:s|ed)?|"
                  r"slide[s]?|slid|wriggle[sd]?|wiggle[sd]?")
TAKES_OFF = (r"(?:" + _TRAILING_VERB + r"|gets?|got)" + _GAP + r"(?:off|out\s+of)\b"
             r"|\b(?:steps?|stepped|stepping|lifts?|lifted|lifting|"
             r"works?|worked|working)\s+(?:off|out\s+of)\b"
             r"|\b(?:removes?|removed|removing|discards?|discarded|sheds?|shedding|"
             r"undresses|undressed)")
PUTS_ON = (r"(?:puts?|putting|pulls?|pulled|slips?|slipped|tugs?|tugged|"
           r"draws?|drew|drawing|"
           r"steps?|stepped|climbs?|climbed|gets?|got|wriggles?)" + _GAP +
           r"(?:on|into|back\s+on)\b"
           r"|\b(?:dresses?\s+in|dressed\s+in|buttons?|zips?\s+up|fastens?)")
DISPLACES = (r"(?:pulls?|pulled|pushes?|pushed|tugs?|tugged|hikes?|hiked|"
             r"rolls?|rolled|lifts?|lifted|yanks?|yanked|shoves?|shoved)"
             + _GAP + r"(?:aside|up|down|open)\b"
             r"|\b(?:unzips?|unzipped|unbuttons?|unbuttoned|unfastens?|unfastened|"
             r"undoes|undid)\b")


def _rx(pattern):
    return re.compile(pattern, re.I)


_SPOKEN_SPAN = re.compile(r"<d>.*?</d>|[\"“][^\"“”]{1,400}?[\"”]",
                          re.S)


def spoken_text(text):
    """Only what people SAY, with the markers stripped. "" when nobody speaks."""
    said = []
    for m in _SPOKEN_SPAN.finditer(text or ""):
        s = m.group(0)
        s = s[3:-4] if s.startswith("<d>") else s[1:-1]
        if s.strip():
            said.append(s.strip())
    return " ".join(said)


_BY_SCRIPT = (
    # Kana before Han: Japanese uses both, so Han alone is what makes it Chinese.
    ("Japanese", r"[぀-ヿ]"),
    ("Korean", r"[가-힯ᄀ-ᇿ]"),
    ("Chinese", r"[一-鿿㐀-䶿]"),
    ("Greek", r"[Ͱ-Ͽἀ-῿]"),
    ("Hebrew", r"[֐-׿]"),
    ("Arabic", r"[؀-ۿݐ-ݿ]"),
    ("Hindi", r"[ऀ-ॿ]"),
    ("Thai", r"[฀-๿]"),
    ("Ukrainian", r"[ЄЇєіїґ]|\b(?:що|це|ти|але|дуже|треба|дякую|немає)\b"),
    ("Russian", r"[Ѐ-ӿ]"),
)
_BY_SCRIPT_RX = tuple((n, re.compile(p)) for n, p in _BY_SCRIPT)
_BY_WORDS = (
    ("English", "the and is are you that not it to of in for with but what have"),
    ("Spanish", "el la los las que de y no se es por con para pero muy sí está"),
    ("French", "le la les des que de et ne pas est vous je pour avec au ça"),
    ("German", "der die das und nicht ist ich du sie wir mit für auf ein aber"),
    ("Italian", "il lo la che di non è sono per con questo come più sei ma"),
    ("Portuguese", "os as que de não é para com você isso mais está eu sou"),
    ("Dutch", "de het een en niet is ik je dat van voor met maar hij zijn"),
    ("Polish", "nie jest to się na że do co jak ale jestem tak mnie"),
    ("Turkish", "bir bu ve için ne değil çok ben sen var yok ama beni"),
    ("Swedish", "och att det är inte jag du en för med men han hon"),
)
_BY_WORDS_SET = tuple((n, frozenset(w.split())) for n, w in _BY_WORDS)


_LANG_NAMES = tuple(n for n, _ in _BY_SCRIPT) + tuple(n for n, _ in _BY_WORDS)
_LANG_ADJ = (r"(?:fluent|broken|perfect|rapid|halting|accented|flawless|bad|"
             r"basic|simple|quiet|loud|slow|fast)\s+")
_NAMED_LANG = re.compile(
    r"\b(?:in|into|speaks?|speaking|spoke|spoken|"
    r"switch(?:es|ed|ing)?\s+to|repl(?:y|ies|ied)\s+in|answers?\s+in|"
    r"says?\s+in|said\s+in|ask(?:s|ed)?\s+in)\s+"
    r"(?:" + _LANG_ADJ + r")?"
    r"(" + "|".join(_LANG_NAMES) + r")\b", re.I)


def language_named(text):
    """The language the TEXT ITSELF says is being spoken, or ''.

    The author's own statement, read from the stage direction rather than voted
    for out of the line. See _NAMED_LANG for why a speech frame is required."""
    m = _NAMED_LANG.search(str(text or ""))
    if not m:
        return ""
    said = m.group(1).lower()
    return next((n for n in _LANG_NAMES if n.lower() == said), "")


def language_of(text, fallback="English", named=""):
    """The language `text` is written in, or `fallback` when it cannot tell.

    Conservative on purpose: naming the WRONG language is worse than naming the
    one the author most likely wanted, so a Latin-alphabet guess has to win by a
    clear margin before it displaces the fallback."""
    t = str(text or "")
    if not t.strip():
        return fallback
    for name, rx in _BY_SCRIPT_RX:
        if rx.search(t):
            return name
    words = set(re.findall(r"[^\W\d_]+", t.lower(), re.UNICODE))
    if not words:
        return fallback
    scores = sorted(((len(words & ws), n) for n, ws in _BY_WORDS_SET), reverse=True)
    best, runner = scores[0], scores[1]
    if best[0] >= 2 and best[0] > runner[0]:
        return best[1]
    if named:
        return named
    return fallback


def _outside_speech(text):
    """The beat with everything anybody SAYS taken out.

    What a character says is not stage direction. The commonest thing to talk
    about is something that is NOT in the room -- "McKenna where are you?" is how
    absence gets written -- and reading a spoken name as a staged one put a full
    description of the missing person into the shot, so the model drew her.

    Both markers, because both exist in the pipeline: <d> once mark_dialogue has
    run, plain quotes before it."""
    return _SPOKEN_SPAN.sub(" ", text or "")


def staged_text(text):
    """What a beat STAGES: not what anybody says, and not what the narration asks.

    A narrated question is the same kind of thing as a line of speech. "Maya waits by
    the door. Will he come?" asks whether he will, which is to say he is not there --
    and the "he" read as him being present, so Will was described into the shot of
    her waiting for him. Removed for deciding who is in the shot, as speech is."""
    staged = _outside_speech(text)
    return " ".join(s for s in re.split(r"(?<=[.!?])\s+", staged) if not s.rstrip().endswith("?"))


# Where an order, a request or an intention stops: the end of its clause, or an "and"
# / "then" that starts a new subject's action -- "tells her to kneel AND SHE KNEELS".
_CLAUSE_TAIL = (r"(?:(?!\b(?:and|then|but|while|before|after|as|until)\s+"
                r"(?:she|he|they|it|(?-i:[A-Z][\w'’-]*))\b|\bthen\b"
                # "...and GRABS her arm": a verb of its own is the subject acting again.
                r"|\band\s+(?!(?:his|this|is|was|has|as|us|yes|less|always|perhaps)\b)"
                r"[a-z]+(?:s|ed)\b)[^,;.!?])*")
# "is going to THE kitchen" is going there now; "is going to TIE her" is not yet.
_TO_A_VERB = (r"to\s+(?!(?:the|a|an|her|his|their|my|your|our|its|this|that|these|those|"
              r"bed|work|school|church|town|sleep)\b)")
_NOT_YET = re.compile(
    # "Dan tells Ana to take off her shorts" -- an order, reported.
    r"\b(?:tells?|told|telling|orders?|ordered|ordering|asks?|asked|asking|commands?|"
    r"commanded|instructs?|instructed|begs?|begged|begging|pleads?\s+with|pleaded\s+with|"
    r"urges?|urged|warns?|warned|dares?|dared|invites?|invited|motions?\s+for|"
    r"signals?\s+for|gestures?\s+for|beckons?\s+for|waits?\s+for|wants?|wanted|"
    r"expects?|expected|needs?|needed|would\s+like|'d\s+like)\s+"
    r"(?:(?!to\b)[\w'’-]+\s+){0,3}?(?:not\s+)?to\s+" + _CLAUSE_TAIL
    # "Dan threatens to gag her", "is going to tie her up", "is about to", "plans to".
    + r"|\b(?:threatens?|threatened|promises?|promised|plans?|planned|intends?|intended|"
    r"offers?|offered|decides?|decided|prepares?|prepared|hopes?|hoped|means?|meant)\s+"
    r"to\s+" + _CLAUSE_TAIL
    + r"|\b(?:is|are|was|were|am|'s|'re)\s+(?:going|about|planning|preparing|getting\s+"
    r"ready|ready|set)\s+" + _TO_A_VERB + _CLAUSE_TAIL
    # "Dan will cuff her later", "she could run" -- not now.
    + r"|\b(?:will|would|shall|could|should|might|may|must|won't|wouldn't|"
    r"can't|cannot|couldn't|shouldn't)\s+" + _CLAUSE_TAIL, re.I)


# "top" that is not a garment: on top, top of, top half, the top drawer...
_NOT_A_TOP = re.compile(
    r"\b(?:on|onto|to|at|from|over|off|up)\s+(?:the\s+|its\s+)?(top)\b"
    r"|\b(top)(?=\s+(?:of|half|drawer|shelf|step|stairs?|bunk|floor|lip|button|end|edge|"
    r"corner|sheet|layer|speed|gear|priority|deck|bed|mattress)\b)"
    r"|\b(tops|topped|topping)\b(?=\s+(?:up|off|out|it|him|her|them|the)\b)", re.I)


def garment_masked(text):
    """The text with words that only LOOK like garments blanked, position for position.

    "Dan climbs on top of Ana" named "top" beside "climbs ... on", which is how putting
    a garment on is written -- so a crop top taken off before the sex scene went back
    on in the middle of it: "White crop top is off the body as the shot opens and fully
    on by the last frame". REPORTED as clothing restored to cover the body in sex
    scenes. Same length out as in, so offsets found in one hold in the other."""
    t = str(text or "")
    out = list(t)
    for m in _NOT_A_TOP.finditer(t):
        for g in (1, 2, 3):
            if m.group(g):
                for i in range(m.start(g), m.end(g)):
                    out[i] = "#"
    return "".join(out)


def acted_text(text):
    """What a beat ACTS OUT, now, in this shot: staged_text, with the orders, requests
    and intentions taken out as well.

    Everything that decides what HAPPENS in a shot -- a restraint going on, a garment
    coming off, somebody kneeling, leaving, arriving -- has to read this, not the beat.
    Read off the whole beat, "Dan tells Ana to take off her shorts" took the shorts off
    in the shot that only asks for it, and a line saying "I will gag you" put the gag
    on in the shot where it is threatened -- a beat early each time. REPORTED as
    actions happening before they are supposed to take place, and dialogue being used
    as an action. The words still go to the model exactly as written; only this node's
    reading of what they stage changes."""
    return re.sub(r"\s{2,}", " ", _NOT_YET.sub(" ", staged_text(text))).strip()


_HW_ONE = _rx(r"\b(" + _ADJ + r"(?:\s+" + _ADJ + r"){0,2}\s+)?("
              + "|".join(p for p, _n, _pt in HARDWARE) + r")\b")
_PART_ONE = _rx(r"\b(" + "|".join(p for p, _n in PARTS) + r")\b")
_NOUN_BEFORE = _rx(r"(?:\b(?:a|an|the|her|his|its|their|my|your|our|this|that|"
                   r"these|those|one|two|three|several|more|another|in|with|by|"
                   r"of|on|from)\b|[,;:(])\s*(?:" + _ADJ + r"\s+){0,3}$")
_REACHES = _rx(r"\b(?:around|round|through|under|over|behind|between|across|to|onto)"
               r"\b(?:\s+[\w']+){0,2}\s*$"
               r"|\band\s+(?:her|his|their|its|the|a|an|both|each)?\s*$")
_POSITION = [(_rx(r"\b" + p + r"\b"), name) for p, name in POSITIONS]
ANCHOR_DET = (r"(?:the|a|an|her|his|its|their|one|another|each|either|that|"
              r"this|both)\s+(?:(?:other|second|third|first|far|near|nearest|"
              r"opposite|left|right|upper|lower|top|bottom|same|nearby|steel|"
              r"iron|metal|heavy|small|large|wooden|old|thick)\s+){0,2}")
# Two anchor nouns together are ONE anchor -- "the ceiling hook", "the wall ring" --
# not the first of them: the hook came back as "the ceiling".
_MOTION = _rx(r"\b(?:walks?|walked|walking|goes|went|runs?|ran|steps?|stepped|heads?|"
               r"headed|moves?|moved|crosses|crossed|comes?|came|returns?|returned|"
               r"turns?|turned|backs?|wanders?|strides?|strode|hurries|hurried|paces|"
               r"paced|drifts?|leads?|led)\b")
_FASTENS = _rx(r"\b(?:chains?|cuffs?|ties?|straps?|locks?|clips?|hooks?|fastens?|"
               r"secures?|binds?|tethers?|attaches?|bolts?|padlocks?|shackles?|lashes?|"
               r"tapes?|leashes?|anchors?)\b")
_ANCHOR_AT = _rx(r"\bto\s+" + ANCHOR_DET + r"((?:(?:" + ANCHORS + r")\s+)?(?:" + ANCHORS
                 + r"))\b")
_APPLY = _rx(r"\b(?:" + APPLY_VERB + r")\b")
_RELEASE = _rx(r"\b(?:" + RELEASE_VERB + r")\b")
_PLACE_IN = _rx(r"\b(?:in|into|inside|through|down|along|across|to|onto|at)\s+"
                r"(?:the|a|an|her|his|their)\s+(" + _ROOM_MOD + r"(?:" + PLACES
                + r"))\b" + PLACE_NOT_AN_OBJECT)
_PLACE_WORD = _rx(r"\b(?:" + PLACES + r")\b")
_GARMENT_ONE = _rx(r"\b(" + _ADJ + r"(?:\s+" + _ADJ + r"){0,2}\s+)?("
                   + _GARMENT + r"s?)\b")
_TAKES_OFF = _rx(r"\b" + TAKES_OFF + r"\b")
_PUTS_ON = _rx(r"\b" + PUTS_ON + r"\b")
_DISPLACES = _rx(r"\b" + DISPLACES + r"\b")
_OPENS_GARMENT = _rx(r"\b(?:unzips?|unzipped|unbuttons?|unbuttoned|unfastens?|unfastened|"
                     r"undoes|undid|unhooks?|unhooked|unclasps?|unclasped)\b")
_COMPLETES_OFF = _rx(r"\b(?:off|out\s+of|away|lets?\s+(?:it|them)\s+(?:fall|drop|slide)|"
                     r"drops?\s+(?:it|them)|falls?\s+(?:to|down|away|off))\b")
_MOVES = _rx(r"\b(?:walks?|walked|walking|goes|go|went|going|runs?|ran|running|"
             r"steps?|stepped|stepping|moves?|moved|moving|enters?|entered|"
             r"leaves?|left|leaving|crosses|crossed|crossing|climbs?|climbed|"
             r"heads?|headed|returns?|returned|arrives?|arrived)\b")


def hardware_spans(text):
    """Every piece of hardware named, as (canonical, part, as-written, at).

    `at` is where it sits in the text, and it is not decoration: a beat that
    cuffs the wrists BEHIND THE BACK and locks a collar CHAINED TO THE WALL has
    two modifiers and two items, and attaching either modifier to both gives
    handcuffs chained to a wall they were never near. Modifiers bind to the
    nearest item, which needs positions to work out.

    ALL of it, too. The old reader returned only the longest single match, so a
    beat that put on cuffs and a collar recorded one and lost the other for the
    rest of the film."""
    text = text or ""
    parts = part_spans(text)
    raw, verb_ats = [], []
    for m in _HW_ONE.finditer(text):
        adj, noun = (m.group(1) or "").strip(), m.group(2)
        canon, part = next((n, pt) for p, n, pt in HARDWARE
                           if re.fullmatch(p, noun, re.I))
        if canon == "chain" and not noun.lower().endswith("ed") \
                and not _NOUN_BEFORE.search(text[:m.start()]):
            verb_ats.append(m.start())
            continue
        written = f"{adj} {noun}".strip().lower()
        if noun.lower().endswith(("ed", "ing")) or (
                noun.lower() != canon and not canon.endswith("s")):
            written = f"{adj} {canon}".strip().lower()
        raw.append([canon, part, written, m.start()])
    # Soft ties, as the instrument of a tying verb only -- see SOFT_TIES.
    for m in _SOFT_TIE.finditer(text):
        if not _APPLY.search(re.split(r"[.;!?]", text[:m.start()])[-1]):
            continue
        # ...and only on a LIMB, or to something: "ties her hair back with a scarf" and
        # "ties the bag with a belt" restrain nobody.
        _clause = (re.split(r"[.;!?]", text[:m.start()])[-1]
                   + re.split(r"[.;!?]", text[m.start():])[0])
        if not (_RESTRAINABLE.search(_clause) or anchor_spans(_clause)):
            continue
        if any(r[3] == m.start(1) for r in raw):
            continue
        noun = m.group(1)
        canon = next(c for p, c in SOFT_TIES if re.fullmatch(p, noun, re.I))
        lead = re.sub(r"^(?:with|using)\s+(?:(?:a|an|the|her|his|their|two|some)\s+"
                      r"|(?:a\s+)?pairs?\s+of\s+)*", "", m.group(0), flags=re.I)
        raw.append([canon, "wrists", re.sub(r"\s+", " ", lead).strip().lower(), m.start(1)])
    _ats = [(c, at) for c, _pt, _w, at in raw]
    _tether = []
    for row in raw:
        if row[0] not in PART_VARIES:
            continue
        _pt = (_nearest_part(parts, row[3], _ats)
               or _instrument_part(text, row[3], parts, [r for r in raw if r is not row]))
        if _pt:
            row[1] = _pt
        elif any(c != row[0] for c, _a in _ats) and _runs_to(text, row[3]):
            _tether.append(row)
    _spread = []
    for row in raw:
        if row[0] not in PART_VARIES:
            continue
        _stop = min([a for c, a in _ats if a > row[3]]
                    + [m.start() for m in re.finditer(r"[.;!?]", text)
                       if m.start() > row[3]],
                    default=len(text))
        _mine = [(pt, at) for pt, at in parts
                 if row[3] < at < _stop and pt != row[1]]
        for _pt, _at in dict.fromkeys(_mine).keys():
            if not _REACHES.search(text[row[3]:_at]):
                continue
            if any(r[0] == row[0] and r[1] == _pt for r in _spread):
                continue
            _spread.append([row[0], _pt, row[2], row[3]])
    raw += _spread
    raw = [r for r in raw if r not in _tether]
    for _vat in verb_ats:
        _pt = _nearest_part(parts, _vat, _ats)
        if _pt and not any(c == "chain" and pt == _pt for c, pt, _w, _a in raw):
            raw.append(["chain", _pt, "chain", _vat])
    if verb_ats and not raw and anchor_in(text):
        raw.append(["chain", "body", "chain", verb_ats[0]])
    out, seen = [], {}
    for canon, part, written, at in raw:
        key = (canon, part)
        if key in seen:
            i = seen[key]
            if len(written) > len(out[i][2]):
                out[i] = (canon, part, written, out[i][3])
            continue
        seen[key] = len(out)
        out.append((canon, part, written, at))
    return out


_REGION_RX = tuple((_rx(r"\b(?:" + p + r")\b"), region, said)
                   for p, region, said in REGION_OF)
_NUDITY_RX = tuple((_rx(r"\b(?:" + p + r")\b"), regions) for p, regions in NUDITY)

# "down to her underwear" keeps the underwear, so it is not a full strip. "Down to
# her skin", "down to nothing" and "down to get in the bath" keep nothing.
_KEEPS = (r"\s+to\s+(?:just\s+|only\s+|nothing\s+but\s+)?"
          r"(?:(?:her|his|their)\s+(?!(?:bare\s+)?skin\b|birthday\s+suit\b)|(?:a|an)\s+|"
          r"(?:" + GARMENT_PHRASES + r"|" + GARMENT_WORDS + r")s?\b)")
STRIPS_BARE = _rx(
    r"\bnaked\b(?!\s+(?:eye|flame))"
    r"|\bnude\b|\bin\s+the\s+nude\b"
    r"|\bundress(?:es|ed|ing)?\b(?!(?:\s+(?:right\s+)?down)?" + _KEEPS + r")"
    r"|\b(?:strips?|stripp(?:ed|ing))\s+(?:naked|bare)\b"
    r"|\b(?:strips?|stripp(?:ed|ing))\s+down\b(?!" + _KEEPS + r")"
    r"|\b(?:strips?|stripp(?:ed|ing))\s+(?:out\s+of|off)\b"
    r"(?=\s*(?:[.,;!?]|$)|\s+(?:and|then|while|as)\b|\s+(?:everything|it\s+all|all\s+of\s+it)\b"
    r"|\s+(?:(?:his|her|their|all\s+(?:his|her|their))\s+)?(?:clothes|clothing|garments|things|kit|outfit|gear)\b)"
    r"|\btakes?\s+(?:everything|it\s+all|all\s+of\s+it|the\s+lot)\s+off\b"
    r"|\b(?:takes?|took|taking|pulls?|pulled|peels?|peeled|sheds?|shed|"
    r"removes?|removed|gets?|got|slips?|slipped|strips?|stripped|stripping)\b"
    r"(?:\s+(?:off|out\s+of))?\s+(?:his|her|their|its|the|all\s+(?:his|her|their))?"
    r"\s*(?:clothes|clothing|garments|things|kit|outfit|gear)\b"
    r"(?:\s+off)?"
    r"|\bstrips?\b(?=\s*[.,;!?]|\s*$)"
    r"|\bstripp(?:ed|ing)\b(?=\s*[.,;!?]|\s*$)"
    r"|\bwearing\s+nothing\b|\bwith\s+no\s+clothes\b|\bbare\s+skin\b")


def region_of(garment):
    """The region a garment covers, or "" when it cannot be placed.

    The FIRST of them where a garment covers more than one. See regions_of, which
    is what anything latching a bare region should be asking."""
    for rx, region, _said in _REGION_RX:
        if rx.search(str(garment or "")):
            return region
    return ""


def regions_of(garment):
    """EVERY region a garment covers. [] when it cannot be placed.

    A dress is two of them, and a table that could only answer with one left the
    other unspecified -- which is the region the model fills from its own prior."""
    out = []
    for rx, region, _said in _REGION_RX:
        if rx.search(str(garment or "")) and region not in out:
            out.append(region)
    return out


def nudity_in(text):
    """The regions a beat says are bare BY DESCRIPTION, widest match first."""
    out = []
    for rx, regions in _NUDITY_RX:
        if rx.search(text or ""):
            for r in regions:
                if r not in out:
                    out.append(r)
    return out


def _bare_on(p, regions):
    for r in ([regions] if isinstance(regions, str) else regions):
        if r and r not in p.bare:
            p.bare.append(r)


def _bare_off(p, regions):
    for r in ([regions] if isinstance(regions, str) else regions):
        if r in p.bare:
            p.bare.remove(r)


_FURNITURE_BEFORE = _rx(r"\b(?:chair|table|bed|desk|bench|stool|sofa|couch|armchair|"
                        r"frame|cross|rack)s?\s*$")


def part_spans(text):
    """Every body part named, as (name, at).

    Not a piece of furniture's: "straps her ankles to the chair legs" named her
    ankles and the CHAIR's legs, and the straps were recorded on both."""
    out = []
    for m in _PART_ONE.finditer(text or ""):
        if _FURNITURE_BEFORE.search((text or "")[:m.start()]):
            continue
        out.append((next(n for p, n in PARTS
                         if re.fullmatch(p, m.group(1), re.I)), m.start()))
    return out


_RUNS_TO = _rx(r"^\s*\w*\s*(?:to|onto|from)\b")


def _runs_to(text, at):
    """Does the item at `at` run TO something -- is it a tether?

    Read just past the word, so "a chain to her collar" and "a chain running to
    the ring" both answer yes and "duct tape" answers no."""
    return bool(_RUNS_TO.search(text[at:][_first_gap(text[at:]):])) \
        or bool(anchor_in(text))


def _first_gap(s):
    """Index just past the first word of `s`."""
    m = re.search(r"\s", s)
    return m.start() if m else len(s)


def _nearest_part(parts, at, ats):
    """The part belonging to the item at `at`, or "".

    English puts it after: "a chain around her ankles", "chains her ankles
    together". So the first part named AFTER this item wins, unless another
    item is named in between -- that one owns it instead."""
    later = [(p, q) for p, q in parts if q > at]
    for name, q in later:
        if any(at < other < q for _c, other in ats):
            break
        return name
    return ""


_WITH_BEFORE = _rx(r"\b(?:with|using)\s+(?:(?:a|an|the|some|more|her|his|their|two|"
                   r"silk|satin|leather|nylon|cotton|black|red|white|long|"
                   r"(?:a\s+)?(?:strips?|lengths?|pieces?|coils?|roll|bits?|pairs?)\s+of)\s+)*$")


def _instrument_part(text, at, parts, rows):
    """The part an item introduced by WITH belongs to, or "".

    English puts the part FIRST there: "ties her ankles with rope", "tapes her mouth
    shut with duct tape". _nearest_part looks only forward and found nothing, so the
    rope took the hardware table's default -- the wrists -- and the tape became a
    second restraint on the wrists beside the one on her mouth. REPORTED as
    restraints turning up on the wrong limb.

    The part named before it in the same sentence, with no other item between; failing
    that, the part of the item it belongs to ("gags her with duct tape" -- the tape IS
    the gag)."""
    if not _WITH_BEFORE.search(text[:at]):
        return ""
    start = max([m.end() for m in re.finditer(r"[.;!?]", text[:at])], default=0)
    earlier = [(p, q) for p, q in parts if start <= q < at]
    for name, q in reversed(earlier):
        if any(q < r[3] < at for r in rows):
            break
        return name
    prior = [r for r in rows if start <= r[3] < at]
    return prior[-1][1] if prior else ""


def position_spans(text):
    """Every limb position named, as (name, at)."""
    out = []
    for rx, name in _POSITION:
        m = rx.search(text or "")
        if m:
            out.append((name, m.start()))
    return sorted(out, key=lambda x: x[1])


def position_in(text):
    """Where the arms are held. '' when the text does not say."""
    got = position_spans(text)
    return got[0][0] if got else ""


def anchor_spans(text):
    """Every fixed thing hardware is fastened to, as (name, at).

    The VERB is required. "to the <thing>" on its own is movement -- "she sinks
    to the floor", "he walks to the table" -- and reading those as fastenings
    latched a restraint over furniture somebody merely walked towards."""
    t, out = text or "", []
    for m in _ANCHOR_AT.finditer(t):
        clause = re.split(r"[.;!?]", t[:m.start()])[-1]
        # ...and in ITS OWN part of the sentence: "locks the cuffs on her and walks to
        # the door" fastens nothing to the door. A motion verb after the last "and" or
        # comma, with no fastening verb beside it, is somebody going somewhere.
        _sub = re.split(r"[,;]|\b(?:and|then|while)\b", clause)[-1]
        if _MOTION.search(_sub) and not _FASTENS.search(_sub) and not _APPLY.search(_sub):
            continue
        if _APPLY.search(clause) or re.search(
                r"\b(?:chains?|ropes?|cords?|cables?|leash(?:es)?|straps?|"
                r"tethers?|links?|lines?)\s+(?:\S+\s+){0,4}?"
                r"(?:runs?|holds?|leads?|stretch(?:es)?|extends?|goes|hangs?)\b",
                clause, re.I):
            out.append((re.sub(r"\s+", " ", m.group(1).lower()), m.start()))
    return out


def anchor_in(text):
    """What hardware is fastened to. '' when the text fastens nothing."""
    got = anchor_spans(text)
    return got[0][0] if got else ""


def place_in(text):
    """The room this text puts the shot in. '' when it names none.

    Behind a preposition, so a room has to be somewhere somebody IS. "Ana looks
    at the door" names no room -- and a door is not on the list in any case."""
    text = text or ""
    m = _PLACE_IN.search(text)
    if not m:
        return ""
    clause = re.split(r"[.;!?]", text[:m.end()])[-1]
    present = re.search(r"\b(?:is|are|was|were|stands?|sits?|waits?|lies?|remains?)\b",
                        clause, re.I)
    if not _MOVES.search(clause) and not present:
        return ""
    got = re.sub(r"\s+", " ", m.group(1).lower()).strip()
    return "" if got == "room" else got


def garments_in(text):
    """Garments named, as written, in order."""
    out, seen = [], set()
    for m in _GARMENT_ONE.finditer(text or ""):
        adj, noun = (m.group(1) or "").strip(), m.group(2)
        phrase = f"{adj} {noun}".strip().lower()
        key = noun.lower().rstrip("s")
        if key not in seen:
            seen.add(key)
            out.append(phrase)
    return out


_CLAUSE_BOUNDARY = re.compile(r"[,;.!?]|\b(?:and|while)\b", re.I)
# From one applied item to the next with nothing but its object, a part and "and"
# between -- no verb of its own, no sentence end. See SceneState.read.
_SHARED_OBJECT = re.compile(
    r"^[^.;!?]{0,80}?\band\s+(?:(?:a|an|the|some|two|her|his|their|pair\s+of|set\s+of|"
    r"steel|metal|iron|leather|chrome|heavy|black|silver|rubber|nylon|padded|thick|"
    r"thin|matching|locking)\s+)*$", re.I)   # no verb of its own: "and PUTS the rope" is not


_ONLY_A_DETERMINER = re.compile(
    r"\s*(?:(?:her|his|their|its|the|a|an|my|your|our|both|two|all)\s+)?\s*", re.I)


def _clause_at(text, at, boundaries=None):
    """Return the clause containing character offset `at` and its start offset."""
    if boundaries is None:
        boundaries = list(_CLAUSE_BOUNDARY.finditer(text or ""))
    lo = max((m.end() for m in boundaries if m.end() <= at), default=0)
    hi = min((m.start() for m in boundaries if m.start() > at), default=len(text))
    return text[lo:hi], lo


_NOT_CLOTHING = re.compile(
    r"^(?:handcuffs?|cuffs?|shackles?|manacles?|chains?|ropes?|cords?|straps?|"
    r"collars?|gags?|blindfolds?|restraints?|bindings?|tape|ties?|harness|"
    r"straitjacket|spreader|hogtie|clamps?|clips?)$", re.I)
_PHRASE_ONE = _rx(r"\b(?:" + GARMENT_PHRASES + r")\b")
_WORD_ONE = _rx(r"^(?:" + GARMENT_WORDS + r")s?$")
_WORD_EXACT = _rx(r"^(?:" + GARMENT_WORDS + r")$")


def singular_garment(word):
    """A garment word as the VOCABULARY spells it, so two readers cannot disagree.

    Reported: several women take their skirts off and the skirts are back in the next
    beat. "their skirts" yields the token "skirts" while the sheet says "a denim
    skirt", and every reader downstream looks the token up in the sheet -- the scrub
    by pattern, infer_removals by entry head -- so a plural garment matched nothing
    and the removal silently did nothing at all. One woman undressing wrote "her
    skirt" and worked; the moment the subject went plural so did the garment.

    A trailing "s" comes off only when the stem is ITSELF a garment word, which is an
    exact test here and not a guess: the vocabulary lists inherently plural garments
    only in the plural (boots, jeans, shorts, panties, tights, leggings, socks,
    knickers, trousers, gloves) and the rest only in the singular."""
    low = str(word or "").lower().strip("-")
    if low.endswith("s") and _WORD_EXACT.match(low[:-1]):
        return low[:-1]
    # "their dresses": the stem is "dress", not "dresse".
    if low.endswith("es") and _WORD_EXACT.match(low[:-2]):
        return low[:-2]
    return low


def garment_words(text):
    """Every garment named, as HEAD WORDS, restraints excluded.

    What the wardrobe logic tracks and compares: ["jeans", "chastity belt"]. Its
    sister garments_in keeps the adjectives, because a shot has to SAY "blue
    jeans" while the tracking only has to know they are jeans.

    Multi-word entries are read first and their words removed, so the single-word
    pass cannot also find "belt" inside "chastity belt" and record the item twice
    under two names."""
    text = text or ""
    out = []
    for m in _PHRASE_ONE.finditer(text):
        phrase = re.sub(r"[\s-]+", " ", m.group(0)).strip().lower()
        if phrase not in out:
            out.append(phrase)
    text = _PHRASE_ONE.sub(" ", text)
    for word in re.findall(r"\b[\w-]{3,}\b", text):
        low = word.lower().strip("-")
        if _NOT_CLOTHING.match(low):
            continue
        if not _WORD_ONE.match(low):
            continue
        low = singular_garment(low)
        if low not in out:
            out.append(low)
    return out

_POSTURE_DENIED = _rx(
    r"\b(?:cannot|can\s*not|can['\u2019]?t|could\s*not|could\s*n['\u2019]?t|"
    r"unable|never|not\s+able|no\s+longer\s+able|"
    r"does\s*n['\u2019]?t|does\s+not|do\s*n['\u2019]?t|did\s*n['\u2019]?t|did\s+not|"
    r"will\s+not|wo\s*n['\u2019]?t|fail(?:s|ed|ing)?|"
    r"tr(?:y|ies|ied|ying)|attempt(?:s|ed|ing)?|struggl(?:e|es|ed|ing)|"
    r"strain(?:s|ed|ing)?|fight(?:s|ing)?|want(?:s|ed)?|need(?:s|ed)?|"
    r"told|ordered|asked|begs?|begged)\b")


def denied_posture(text, at):
    """Is the posture verb at `at` negated, or only attempted, by what precedes it?"""
    before = str(text or "")[:max(0, int(at))]
    cut = 0
    for _m in re.finditer(r"[,;:.!?]|\b(?:so|and|but|then|yet|while|as|before|after)\b",
                          before, re.I):
        cut = _m.end()
    window = " ".join(re.findall(r"[\w'\u2019]+", before[cut:])[-5:])
    return bool(_POSTURE_DENIED.search(window))


def posture_in(text):
    """The posture this beat puts a body in. '' when it does not.

    Reads the one table. sampler.posture_in answers a different question -- which
    PERSON each posture belongs to, clause by clause -- and keeps its own reader
    for that, but off the same vocabulary."""
    t = text or ""
    hits = sorted((m.start(), name) for name, rx in _POSTURE_OF
                  for m in rx.finditer(t) if not denied_posture(t, m.start()))
    return hits[0][1] if hits else ""


DET_POSS = r"(?:the|her|his|their|its|a|an|\w+['’]s)"
_DET_POSS = DET_POSS


_UNDO_VERB = (r"remove[sd]?|removing|undress(?:es|ed)?|shed(?:s|ding)?|unzip(?:s|ped)?|"
              r"unbutton(?:s|ed)?|unhook(?:s|ed)?|unclasp(?:s|ed)?|unfasten(?:s|ed)?|"
              r"unlock(?:s|ed)?|unbuckle[sd]?|unclip(?:s|ped)?|unstrap(?:s|ped)?|"
              r"unlace[sd]?|untie[sd]?|unties|unwrap(?:s|ped)?|"
              r"undo(?:es)?|undid")


FLOOR = (r"floor|ground|tiles?|tiling|lino|mat|bath\s*mat|rug|carpet|deck|boards|"
         r"concrete|grass|sand|bed|sofa|couch|chair|seat|stool|bench|basket|"
         r"hamper|laundry|pile|heap")
TO_THE_FLOOR = (r"(?:to|on|onto|into|in)\s+(?:the|a|an|her|his|their)?\s*"
                r"(?:" + FLOOR + r")\b")
_LANDS_OFF = _rx(r"\s*(?:fall(?:s|ing)?|drop(?:s|ping)?|land(?:s|ing)?)?\s*"
                 + TO_THE_FLOOR)

_DISPLACE_WAY = (r"back\s+up|back\s+down|down|up|aside|open|back|"
                 r"off\s+(?:one|her|his|their)\s+shoulders?")
_DISPLACE = re.compile(
    r"\b(?:" + _STRIP_VERB + r"|push(?:es|ed|ing)?|shove[sd]?|roll(?:s|ed|ing)?|"
    r"hitch(?:es|ed)?|hike[sd]?|open(?:s|ed)?|undo(?:es)?|undid|unzip(?:s|ped)?|"
    r"unbutton(?:s|ed)?|unfasten(?:s|ed)?|unhook(?:s|ed)?|unclasp(?:s|ed)?|"
    r"lift(?:s|ed|ing)?|raise[sd]?|rais(?:es|ed|ing)|hoist(?:s|ed|ing)?|"
    r"hold(?:s|ing)?|held|gather(?:s|ed|ing)?|bunch(?:es|ed|ing)?)\s+"
    r"(?:(" + _DISPLACE_WAY + r")\s+)?"
    r"(" + _DET_POSS + r"\s+)?([\w][\w\- ]{0,28}?)"
    r"(?:\s+(" + _DISPLACE_WAY + r"))?"
    r"(?=[.,;:!?]|\s+(?:and|to|so|while|as|then)\b|$)", re.I)


_CLAUSE_BEFORE_GARMENT = _rx(r"\b(?:wears?|wearing|dressed(?:\s+in)?|in|is|are|was|were|"
                             r"has\s+on|puts?\s+on|only|just|a|an|the|her|his|their|its)"
                             r"\s+")


def scene_name_for(head, scene):
    """The sheet's OWN full name for a garment, found by its head noun. "" if absent.

    A beat calls a thing whatever is convenient -- "the shorts" for what the sheet
    dressed her in as "blue jeans shorts". Anything the node then says about it has
    to use the SHEET's words: a shot carrying both names is a shot describing two
    garments, and the model draws the bare one however it likes. That is a garment
    invented out of the node's own text, which is the worst kind.

    The entry is read back from the sheet: its modifiers are the words before the
    head noun in the same comma-separated item, and no further -- a name from the
    entry before it would attach one garment's colour to another."""
    head = (head or "").strip().lower()
    if not head or not scene:
        return ""
    best = ""
    for line in str(scene).split("\n"):
        # Only the wardrobe side of "Name: she, 22, blue jeans shorts".
        line = line.split(":", 1)[-1]
        for item in re.split(r"[,;.]", line):
            item = re.sub(r"<\s*picture\s+\d+\s*>", " ", item, flags=re.I)
            item = re.sub(r"\s+", " ", item).strip()
            for part in re.split(r"\s+(?:over|under|beneath|underneath|on\s+top\s+of|and)\s+",
                                 item, flags=re.I):
                part = re.split(r"\s+with\s+", part, flags=re.I)[0].strip()
                if not part or part.split()[-1].lower() != head:
                    continue
                # Drop a leading article or possessive; they are not description.
                part = bare_name(part)
                # ...and the clause in front of it when the scene is PROSE: "Kate
                # wears a red skirt" named the skirt "Kate wears a red skirt", and the
                # removal said "The Kate wears a red skirt comes off".
                part = _CLAUSE_BEFORE_GARMENT.split(part)[-1].strip()
                if len(part) > len(best):
                    best = part
    return best


def displaced_garments(beat, scene):
    """[(garment, how)] this beat MOVES without taking off. [] when none.

    Same two conditions infer_removals uses, and for the same reason: the beat has
    to stage it, and the scene has to already say the thing is worn. A displacement
    invented for something nobody is wearing describes a garment into existence."""
    if not beat or not scene:
        return []
    out, seen = [], set()
    low = scene.lower()
    for m in _DISPLACE.finditer(beat):
        way = (m.group(1) or m.group(4) or "").lower().strip()
        thing = re.sub(r"\s+", " ", (m.group(3) or "")).strip().lower()
        if not way and re.match(r"\s*(?:lift|rais|hoist|gather|bunch)", m.group(0),
                                re.I):
            way = "up"
        if not way and re.match(r"\s*(?:unzip|unbutton|unfasten|unhook|unclasp|undo|undid)",
                                m.group(0), re.I):
            way = "open"
        if not way or not thing or thing in seen:
            continue
        head = thing.split()[-1]
        if len(head) < 3 or head not in low:
            continue
        seen.add(thing)
        thing = scene_name_for(head, scene) or thing
        # "back up" and "back down" say the direction in their second word.
        way = re.sub(r"^back\s+", "", re.sub(r"\s+", " ", way))
        out.append((thing, "pulled " + way if way in ("down", "up", "aside", "back")
                    else way))
    return out


_RESTORE_VERB = (r"(?:let(?:s|ting)?(?:\s+go\s+of)?|drop(?:s|ped|ping)?|"
                 r"lower(?:s|ed|ing)?|smooth(?:s|ed|ing)?|straighten(?:s|ed|ing)?|"
                 r"fix(?:es|ed|ing)?|rearrang(?:e|es|ed|ing)|"
                 r"replac(?:e|es|ed|ing)|put(?:s|ting)?|tidy|tidies|tidied|"
                 r"cover(?:s|ed|ing)?\s+(?:herself|himself|themselves|up)|"
                 r"pull|tug|hitch|hike|yank|push)")
# The old pronoun form, plus the new verbs, still with no garment named.
_PUT_BACK = re.compile(
    r"\b(?:pull|tug|hitch|hike|yank|push)(?:s|ed|ing)?\s+"
    r"(?:it|them|these|those)\s+(?:back\s+)?(?:up|down|closed|shut|together)\b"
    r"|\b(?:pull|tug|hitch|hike|yank|push)(?:s|ed|ing)?\s+"
    r"(?:it|them)\s+back\b"
    r"|\b" + _RESTORE_VERB + r"(?:s|ed|ing)?\s+"
    r"(?:it|them|these|those)\s+(?:fall|drop|go|back|down|straight)\b"
    r"|\blet(?:s|ting)?\s+(?:it|them)\s+fall\b"
    r"|\b(?:cover(?:s|ed|ing)?\s+(?:herself|himself|themselves)\s+(?:back\s+)?up)\b",
    re.I)
_PUT_BACK_NAMED = re.compile(
    r"\b" + _RESTORE_VERB + r"\s+"
    + _DET_POSS + r"\s+([\w][\w\- ]{0,28}?)"
    r"(?:\s+(?:fall|drop|down|back|straight|up|closed|shut|together))?"
    r"(?=[.,;:!?]|\s+(?:and|to|so|while|as|then|over|again)\b|$)", re.I)


def puts_it_back(beat):
    """Does this beat put a displaced garment right without naming it?"""
    return bool(_PUT_BACK.search(beat or ""))


def restored_garments(beat, scene):
    """[garment] this beat puts back, by name. [] when it names none.

    Same two conditions as displaced_garments: the beat has to stage it, and the
    sheet has to already dress them in the thing. The sheet's own name is what
    comes back, so a restore keyed on "her skirt" clears a displacement stored as
    "long grey skirt"."""
    if not beat or not scene:
        return []
    out, low = [], scene.lower()
    for m in _PUT_BACK_NAMED.finditer(beat):
        if _LANDS_OFF.match(beat[m.end():]):
            continue
        thing = re.sub(r"\s+", " ", (m.group(1) or "")).strip().lower()
        if not thing:
            continue
        if re.search(r"\b(?:on|onto|to|into|in|at|over|under|from|with|and)\b", thing):
            continue
        head = thing.split()[-1]
        if len(head) < 3 or head not in low:
            continue
        name = scene_name_for(head, scene) or thing
        if name not in out:
            out.append(name)
    return out


_UNDER_BY_REGION = {
    "lower": (r"panties|knickers|thong|g-?string|briefs|boxers|boxer\s+shorts|"
              r"underwear|undies|jockstrap|loincloth|"
              r"chastity[\s-]*(?:belts?|devices?|cages?)"),
    "upper": (r"bra|bralette|brassiere|camisole|undershirt|vest|corset|bustier"),
}
_OUTER_BY_REGION = {
    "lower": (r"shorts|trousers|jeans|slacks|chinos|skirt|kilt|leggings|joggers|"
              r"tights|pantyhose|jeggings|culottes|"
              r"tracksuit\s+bottoms|dungarees|overalls|dress|gown|robe"),
    "upper": (r"top|shirt|blouse|t-?shirt|tee|jumper|sweater|sweatshirt|hoodie|"
              r"cardigan|jacket|coat|dress|gown|robe|dungarees|overalls|tunic"),
}


def implied_layers(scene):
    """{under: over} for underwear the scene lists beneath outer clothes it also lists.

    Only where BOTH are named: underwear with nothing over it is on show, and saying
    it is hidden would be describing away something the author dressed them in."""
    covers = {}
    for line in (scene or "").split("\n"):
        text = line.strip()
        if not text:
            continue
        for region, unders in _UNDER_BY_REGION.items():
            over = None
            for m in re.finditer(r"\b(?:" + _OUTER_BY_REGION[region] + r")\b",
                                 text, re.I):
                over = m.group(0).lower()
            if not over:
                continue
            for m in re.finditer(r"\b(?:" + unders + r")\b", text, re.I):
                under = re.sub(r"\s+", " ", m.group(0).lower())
                if under != over:
                    covers.setdefault(under, over)
    return covers


def is_undergarment(item):
    """Is this one of the things that is ALWAYS worn under clothes?

    Panties, knickers, thongs, briefs, boxers, underwear, bras, corsets, and
    chastity belts, devices and cages -- the _UNDER_BY_REGION list, which is why
    it is read off that list rather than a second copy that could drift from it.

    These are named and placed rather than deleted. A locket under a coat is a
    different thing: it genuinely cannot be seen, nothing is lost by waiting for
    the coat to come off, and it keeps the older behaviour."""
    t = str(item or "").lower()
    return any(re.search(r"\b(?:" + pat + r")\b", t, re.I)
               for pat in _UNDER_BY_REGION.values())


def hidden_layers(covers, gone, moved=()):
    """Garments still underneath something that is still covering them.

    `moved` is outer garments the beats have DISPLACED -- pulled down, pushed
    aside. Those are still worn, so `gone` never learns about them, and the
    layer beneath stayed hidden while the beat was busy showing it off: "pulls
    her shorts down to show the thong" described the thong in that one shot,
    from the author's own words, and hid it again in the next."""
    aside = {str(m).lower().split()[-1] for m in (moved or ()) if str(m).strip()}
    return [u for u, o in (covers or {}).items()
            if o not in gone and u not in gone
            and str(o).lower().split()[-1] not in aside]


_POSTURE_OF = (
    ("sitting", re.compile(r"\b(?:sits?|sat|sitting|seats?\s+(?:her|him|them)self|"
                           r"is\s+seated|takes?\s+a\s+seat|perch(?:es|ed)?)\b", re.I)),
    ("kneeling", re.compile(r"\b(?:kneels?|knelt|kneeling|"
                            r"(?:goes?|got|gets?)\s+down\s+on\s+(?:her|his|their)\s+knees)\b",
                            re.I)),
    ("lying down", re.compile(r"\b(?:lies?|lay|lays?|laid|lying|laying|"
                              r"stretches?\s+out|sprawls?|sprawled)\b", re.I)),
    ("standing", re.compile(r"\b(?:stands?|stood|standing|"
                            r"(?:gets?|got)\s+(?:up|to\s+(?:her|his|their)\s+feet)|"
                            r"rises?|rose|risen)\b", re.I)),
)
# ...down onto something a body lies on. Not a sofa or a chair: "pushes her down onto
# the sofa" is as often a woman made to sit.
_DOWN_ON = (r"(?:\s+(?:down|back|backwards|flat|hard|roughly|face[-\s]?(?:down|first|up)))*"
            r"\s+(?:on|onto|across|into|to)\s+(?:the|a|an|her|his|their)\s+(?:\w+\s+)?"
            r"(?:bed|mattress|floor|floorboards|ground|carpet|rug|cot|bunk|futon|tiles|"
            r"concrete)\b")
_NOT_A_BODY = re.compile(r"\b(?:it|chair|table|box|case|bag|door|house|room|"
                         r"building|tree|bottle|glass|book|light|lamp)\s+\w{0,8}?\s*"
                         r"(?:stands?|lies?|sits?)\b", re.I)
_POSTURE_OF = _POSTURE_OF + (
    ("squatting", _rx(r"\b(?:squats?|squatting|squatted)\b")),
    ("crouching", _rx(r"\b(?:crouch(?:es|ing|ed)?)\b")),
    ("bent over", _rx(r"\b(?:bends?\s+over|bent\s+over|leans?\s+over|"
                      r"leaned\s+over|doubles?\s+over)\b")),
    ("curled up", _rx(r"\b(?:curled\s+up|curls?\s+up|foetal|fetal)\b")),
    ("lying down", _rx(r"\broll(?:s|ed|ing)?\s+(?:over\s+)?(?:on)?to\s+"
                       r"(?:her|his|their)\s+"
                       r"(?:side|back|front|stomach|belly)\b")),
    # PUT DOWN, rather than lying down. "Dan pushes her down onto the bed" left no
    # posture at all, so the shots after it -- the cuffing, the taping -- said
    # nothing about how she lay, and the model sat her up on her arms. REPORTED as
    # her supporting herself on her arms when she should be lying flat on the bed.
    # The verb is the one at the match start, so sampler.posture_in hands the pose
    # to its OBJECT, the way it already does for "lays her on the bed".
    ("lying down", _rx(r"\b(?:push(?:es|ed)?|shov(?:e|es|ed)|throws?|threw|thrown|"
                       r"pins?|pinned|forc(?:e|es|ed)|lowers?|lowered|tips?|tipped|"
                       r"knocks?|knocked|flings?|flung|toss(?:es|ed)?|press(?:es|ed)?|"
                       r"holds?|held|puts?|rolls?|rolled)\s+"
                       r"(?:her|him|them|(?-i:[A-Z][\w-]+))" + _DOWN_ON)),
    ("lying down", _rx(r"\b(?:falls?|fell|collapses?|collapsed|flops?|flopped|"
                       r"tumbles?|tumbled)" + _DOWN_ON)),
)


_SENTENCE_START_ALSO = frozenset("""
may will can must might shall should would could does did was were are is
let get go come take put look stop wait now then there here this that these
those one some all any no yes so but and or if when while after before both
say tell keep hold turn open close pull push move step down
""".split())


def _alias_at(word, staged):
    """Where a one-word stand-in for a longer name appears, or None.

    A capital at the start of a sentence is free, so a word that is ordinary
    English there has to earn its match somewhere else in the beat."""
    fallback = None
    for m in re.finditer(r"\b" + re.escape(word) + r"\b", staged):
        opens = re.search(r"(?:^|[.!?;:]\s*|[\"'“]\s*)$", staged[:m.start()])
        if not opens:
            return m.start()
        if word.lower() not in _SENTENCE_START_ALSO and fallback is None:
            fallback = m.start()
    return fallback


_AUX_FOLLOWER = re.compile(r"\s+(?:I|you|he|she|we|they|it|there|this|that|anyone|someone|"
                           r"everyone|anybody|somebody)\b")


def names_in(beat, cast):
    """Names this beat STAGES, in the order the sentence puts them.

    Case-SENSITIVE, and that is not fussiness: prose capitalises a name, and
    matching without case makes the word "will" find a character called Will and
    "grace" find Grace. The sampler had that fixed and this file did not, which
    is what two copies of one idea buys you.

    Speech-stripped, for the same reason it is everywhere else -- "McKenna, where
    are you?" is how absence gets written, and reading it as presence put a whole
    sheet entry into a shot the person is not in."""
    staged = staged_text(beat or "")
    names = [str(n) for n in (cast or []) if n]
    hits, found = [], set()
    for n in names:
        for m in re.finditer(r"\b" + re.escape(n) + r"\b", staged):
            _opens = re.search(r"(?:^|[.!?]\s*[\"'\u201c]?)\s*$", staged[:m.start()])
            if _opens and _AUX_FOLLOWER.match(staged[m.end():]):
                continue
            hits.append((m.start(), n))
            found.add(n)
            break
    for n in names:
        if n in found or " " not in n:
            continue
        for w in n.split():
            if len(w) < 3 or not w[:1].isupper():
                continue
            if any(w == o or w in o.split() for o in names if o != n):
                continue
            m = _alias_at(w, staged)
            if m is not None:
                hits.append((m, n))
                found.add(n)
                break
    return [n for _at, n in sorted(hits)]


class Restraint:
    """One piece of hardware on one person.

    part, position and anchor travel TOGETHER, and that is the whole point. The
    old engine read the part off the item list and the position off the beat, in
    two functions that never met, and emitted "holding the neck behind the back".
    A position describes where the ARMS are, so a collar cannot carry one -- the
    constructor drops it rather than trusting the caller."""

    __slots__ = ("item", "part", "position", "anchor", "applied_in", "rigid")

    def __init__(self, item, part, position="", anchor="", applied_in=0,
                 rigid=False):
        self.item = item
        self.part = part
        self.position = position if part in ("wrists", "arms") else ""
        # ...and a gag or a blindfold is not fastened TO anything. "Presses tape over
        # her mouth, then drags her to the chair" put the tape "fast at the chair", a
        # sentence about a mouth taped to furniture that the model could not draw and
        # resolved by drawing no tape. REPORTED as the gag vanishing after the drag.
        self.anchor = anchor if part not in ("mouth", "eyes") else ""
        self.applied_in = applied_in
        self.rigid = bool(rigid) or _is_rigid(item)

    def phrase(self):
        """Where this one holds, as English."""
        bits = [f"the {self.part}"]
        if self.position:
            bits.append(self.position)
        if self.anchor:
            bits.append(f"fast to the {self.anchor}")
        return " ".join(bits)

    def __repr__(self):
        return (f"Restraint({self.item!r},{self.part!r},"
                f"{self.position!r},{self.anchor!r})")


# WHAT FREES SOMEBODY, AND WHAT ONLY LOOKS LIKE IT.
_ATTEMPT = re.compile(
    r"\b(?:try|tries|tried|trying|attempts?|attempted|attempting|struggles?|struggled|"
    r"struggling|strains?|strained|straining|fights?|fought|fighting|tugs?|tugged|"
    r"tugging|can't|cannot|can\s+not|couldn't|could\s+not|fails?|failed|failing|"
    r"unable|begs?|begged|begging|pleads?|pleaded|pleading|wants?|wanted|wishes|"
    r"hopes?|refuses?|refused|won't|doesn't|does\s+not|never|without|almost|nearly)\b",
    re.I)
_TAKE_AWAY = re.compile(
    r"\b(?:takes?|took|taking|pulls?|pulled|pulling|slips?|slipped|slipping|lifts?|"
    r"lifted|lifting|gets?|got|getting|eases?|eased|easing|peels?|peeled|peeling|rips?|"
    r"ripped|ripping|tears?|tore|tearing|yanks?|yanked|yanking|unwinds?|unwound|"
    r"unwraps?|unwrapped|cuts?|snips?|snipped|works?|worked|unhooks?|unhooked|"
    r"unclasps?|unclasped|drags?|dragged|tugs?|tugged)\b", re.I)


def _off_after(beat, at, canon):
    """Is the item at `at` followed, inside its clause, by off / out / away / loose?"""
    pat = next((p for p, n, _pt in HARDWARE if n == canon), None)
    m = re.compile(r"\b(?:" + pat + r")\b", re.I).search(beat, at) if pat else None
    end = m.end() if m else at
    return bool(re.match(r"^[^,;.!?]{0,30}?\b(?:off|out|away|loose)\b", beat[end:], re.I))


# One pair of handcuffs written two ways -- "cuffs her wrists" and "the steel
# handcuffs" -- is one thing to take off.
_SAME_THINGS = (frozenset(("cuffs", "handcuffs")),)


def _same_thing(a, b):
    return a == b or any(a in g and b in g for g in _SAME_THINGS)


# A GAG IS WHATEVER IS OVER THE MOUTH. "Gags her with duct tape" is one strip of tape
# written two ways, and it was recorded twice -- a gag AND tape -- so "rips the tape
# off her mouth" took the tape and left a gag nobody had put on, held in every shot
# after. The other way round, "removes her gag" found no gag over tape pressed on.
_MOUTH_PIECES = frozenset(("gag", "tape"))


def _same_piece(key, canon, part=None):
    """Does taking off `canon` (named at `part`) take off the piece stored as `key`?"""
    if _same_thing(key[0], canon):
        return True
    return (part == "mouth" and key[1] == "mouth"
            and key[0] in _MOUTH_PIECES and canon in _MOUTH_PIECES)


_FREE_VERB_ITEMS = (
    (r"uncuff", ("cuffs", "handcuffs", "manacles", "ankle cuffs", "wrist irons")),
    (r"ungag", ("gag",)),
    (r"unchain", ("chain",)),
    (r"unshackl", ("shackles", "leg irons", "ankle irons")),
    (r"unstrap", ("straps",)),
    (r"unblindfold", ("blindfold",)),
    (r"unti|untying|unbind|unbound", ("rope", "tape", "straps", "zip ties", "scarves",
                                     "stockings", "belt", "ties", "sheet", "bandana",
                                     "laces", "cling film", "steel cable", "tether")),
)
_FREE_PART = {"wrist": "wrists", "wrists": "wrists", "hand": "wrists", "hands": "wrists",
              "ankle": "ankles", "ankles": "ankles", "feet": "ankles", "foot": "ankles",
              "leg": "ankles", "legs": "ankles", "neck": "neck", "throat": "neck",
              "mouth": "mouth", "eyes": "eyes", "arms": "arms", "arm": "arms"}
_FREE_VERBS = (r"unlocks?|unlocked|unlocking|unties?|untied|untying|unbinds?|unbound|"
               r"unbinding|unchains?|unchained|unchaining|uncuffs?|uncuffed|uncuffing|"
               r"unshackles?|unshackled|unstraps?|unstrapped|ungags?|ungagged|"
               r"unblindfolds?|unblindfolded|releases?|released|releasing|frees|freed|"
               r"freeing|to\s+free")


def person_releases(beat, cast, pronouns=None, subject="", state=None):
    """[(name, part or None, only these canons or None)] a beat frees a PERSON of.

    The object has to be the person, and what follows it decides how much: nothing
    (or "from ...") is all of it, a restrainable part is what is on that part, and
    any other noun is not a restraint at all -- "releases her ARM" is a grip let go,
    "unties her HAIR" is hair. A try is not a release, and neither is "free" as an
    adjective: "pulls her hands free" is her straining, not him unlocking anything.
    A verb that names its item frees only that item: "uncuffs" leaves the collar."""
    b = str(beat or "")
    people = [n for n in (cast or []) if n]
    held = [n for n in people
            if state is not None and n in state.people and state.people[n].hardware]
    if not held:
        return []
    name_alt = "|".join(re.escape(n) for n in sorted(people, key=len, reverse=True))
    obj = r"(her|him|them|his|their" + (r"|(?-i:" + name_alt + r")(?:['’]s)?" if name_alt else "") + r")"
    out = []
    pat = re.compile(r"\b(" + _FREE_VERBS + r")\s+" + obj + r"(?![\w'’])"
                     r"(?:\s+(?:own\s+)?(\w+))?", re.I)
    passive = re.compile(r"\b(she|he|they" + (r"|(?-i:" + name_alt + r")" if name_alt else "")
                         + r")\s+(?:is|was|gets|got|has\s+been|is\s+being|finally\s+is)\s+"
                         r"(?:\w+ly\s+)?(released|freed|untied|uncuffed|unchained|"
                         r"unshackled|unstrapped|ungagged|unbound|set\s+free|cut\s+free|"
                         r"cut\s+loose)\b", re.I)
    sets_free = re.compile(r"\b(sets?|set|cuts?|cut)\s+" + obj + r"\s+(?:free|loose)\b", re.I)

    def _who(word, possessive):
        w = word.lower().rstrip("'’s") if word.lower().endswith(("'s", "’s")) else word.lower()
        for n in people:
            if word.startswith(n):
                return n
        group = {"her": "she", "him": "he", "them": "they", "his": "he",
                 "their": "they", "she": "she", "he": "he", "they": "they"}.get(w)
        if not group:
            return ""
        declared = any((pronouns or {}).get(n) == group for n in people)
        if declared:
            fits = [n for n in people if (pronouns or {}).get(n) == group]
            if possessive and subject in fits:
                return subject
            others = [n for n in fits if n != subject]
            return others[0] if len(others) == 1 else (fits[0] if len(fits) == 1 else "")
        # Nobody declared a pronoun: the one person in hardware, for an object.
        return held[0] if (len(held) == 1 and not possessive) else ""

    def _only(verb):
        v = verb.lower()
        for stem, items in _FREE_VERB_ITEMS:
            if re.match(stem, v):
                return set(items)
        return None

    for m in pat.finditer(b):
        clause_start = max(b.rfind(c, 0, m.start()) for c in ".;!?,") + 1
        if _ATTEMPT.search(b[clause_start:m.start()]):
            continue
        verb, word, nxt = m.group(1), m.group(2), (m.group(3) or "")
        possessive = word.lower() in ("his", "their") or word.endswith(("'s", "’s"))
        part = None
        if nxt and nxt.lower() not in ("from", "of", "and", "then", "so", "before",
                                       "after", "as", "with", "at", "in", "to", "while"):
            part = _FREE_PART.get(nxt.lower())
            if part is None:
                continue                    # her hair, her sneakers: not a restraint
            # "releases her arm" / "her hand" is a grip let go.
            if re.match(r"releas", verb, re.I) and nxt.lower() in ("arm", "arms", "hand", "hands"):
                continue
            possessive = True
        elif word.lower() == "her" and not nxt:
            possessive = False
        elif word.lower() in ("his", "their"):
            continue                        # "unlocks his" + nothing: not a person freed
        who = _who(word, possessive)
        if who and who in held:
            out.append((who, part, _only(verb)))
    for m in passive.finditer(b):
        clause_start = max(b.rfind(c, 0, m.start()) for c in ".;!?,") + 1
        if _ATTEMPT.search(b[clause_start:m.start()]):
            continue
        who = _who(m.group(1), False) if m.group(1).lower() in ("she", "he", "they") \
            else next((n for n in people if m.group(1) == n), "")
        if who and who in held:
            out.append((who, None, _only(m.group(2))))
    for m in sets_free.finditer(b):
        clause_start = max(b.rfind(c, 0, m.start()) for c in ".;!?,") + 1
        if _ATTEMPT.search(b[clause_start:m.start()]):
            continue
        who = _who(m.group(2), False)
        if who and who in held:
            out.append((who, None, None))
    return out


class Person:
    __slots__ = ("name", "hardware", "worn", "removed", "displaced",
                 "posture", "place", "bare")

    def __init__(self, name):
        self.name = name
        self.hardware = {}      # (canonical, part) -> Restraint
        self.worn = []          # garments on the body, as written
        self.removed = []       # garments taken off
        self.displaced = []     # pulled aside but still on
        self.posture = ""
        self.place = ""
        self.bare = []

    def restrained(self):
        return bool(self.hardware)

    def kinds(self):
        """What is on, by canonical name, in the order it went on.

        The key is a (name, part) pair so that a chain on the ankles and a chain
        on the wrists can both exist. Almost nothing cares about the part, so it
        asks here instead of unpacking the key."""
        return [c for c, _pt in self.hardware]

    def hw(self, canon, part=None):
        """One restraint by name, or None. Give a part to pick between two."""
        for (c, pt), r in self.hardware.items():
            if c == canon and (part is None or pt == part):
                return r
        return None

    def __repr__(self):
        return (f"Person({self.name!r},hw={list(self.hardware)},"
                f"posture={self.posture!r})")


class SceneState:
    """What is true right now, and what each shot is told because of it."""

    def __init__(self, place=""):
        self.place = place
        self.opened_in = place
        self.people = {}
        self.shot = 0

    def person(self, name):
        if name not in self.people:
            self.people[name] = Person(name)
        return self.people[name]

    def declare(self, name, description, staged_later=()):
        """Take what a character sheet already says as read.

        A sheet entry is a STATE, not an event: "Kate: she, 30, coat, handcuffs"
        says the handcuffs are already on before any beat puts them there. The
        engine read only beats at first, so a scene that opened with somebody
        already restrained had no hardware in it at all until a beat happened to
        mention some -- and a hold that never fires is hardware the model is free
        to leave off.

        `staged_later` is what stops that becoming its own bug. A sheet says WHAT
        somebody has and never says WHEN, so a sheet reading "McKenna: she, 27,
        green dress, handcuffs" beside a script that cuffs her in beat 3 declared
        them on from shot 1 -- and shot 1 went out saying "The handcuffs stay
        closed and fastened AS THEY WERE PUT ON", two shots before anybody put
        them on. Reported as a handcuff on her arm before she is handcuffed.

        Where the script stages the fastening, the script decides the moment. The
        sheet still supplies the description; it just does not get to start the
        clock. Declared, never applied: nothing here is a change, so no shot is
        told anything goes on during it."""
        p = self.person(name)
        for canon, part, written, _at in hardware_spans(description or ""):
            if canon in staged_later:
                continue
            if (canon, part) not in p.hardware:
                p.hardware[(canon, part)] = Restraint(written or canon, part,
                                              position_in(description or ""),
                                              anchor_in(description or ""), 0)
        _off = [_garment_key(x) for x in p.removed]
        for g in garments_in(description or ""):
            key = _garment_key(g)
            if key not in [_garment_key(x) for x in p.worn] and key not in _off:
                p.worn.append(g)
        return p

    def _holders_of(self, canon, beat, at, named, subject, cast, pronouns, part=None):
        """Who is wearing the `canon` a beat takes off -- the names in its clause that
        hold one, else the possessive in front of it, else the only person in one."""
        holders = [n for n, q in self.people.items()
                   if any(_same_piece(k, canon, part) for k in q.hardware)]
        if not holders:
            return []
        hit = [n for n in (named or []) if n in holders]
        if hit:
            return hit
        owner = possessor_at(beat, at, list(cast or self.people), subject, pronouns)
        if owner in holders:
            return [owner]
        return holders if len(holders) == 1 else []

    # -- reading a beat ----------------------------------------------------
    def read(self, beat, cast=(), shot=0, pronouns=None):
        """Update the state from one beat, and report what CHANGED.

        The change matters separately from the result: the shot that puts the
        cuffs on has to say both ends of that, and every shot after it says only
        the result.

        `pronouns` is {name: "she"|"he"|"they"} off the sheet, so "her" in "Dan
        takes off her shirt" can find Kate. Without it a possessive stays with the
        subject."""
        self.shot = shot
        beat = beat or ""
        changed = {"applied": [], "released": [], "moved_to": "", "posture": {},
                   "removed": [], "worn": [], "displaced": []}

        here = place_in(beat)
        if here:
            self.place = here
            changed["moved_to"] = here

        who = names_in(beat, cast)
        subject = who[0] if who else next(iter(list(self.people) or list(cast)
                                               or [""]))

        spans = hardware_spans(beat)
        garments = list(_GARMENT_ONE.finditer(garment_masked(beat)))
        boundaries = list(_CLAUSE_BOUNDARY.finditer(beat)) if spans or garments else []
        applying = bool(spans) and bool(_APPLY.search(beat))
        releasing = bool(_RELEASE.search(beat))

        if spans:
            _last = None            # (wearer, at) of the last item this beat applied
            for canon, part, written, at in spans:
                clause, lo = _clause_at(beat, at, boundaries)
                item_at = at - lo
                apply_at = max((m.start() for m in _APPLY.finditer(clause)
                                if m.start() <= item_at), default=-1)
                release_at = max((m.start() for m in _RELEASE.finditer(clause)
                                  if m.start() <= item_at), default=-1)
                # ONE VERB, TWO OBJECTS: "locks leg irons on her ankles AND steel cuffs
                # on her wrists". The second clause has no verb of its own, so it was
                # skipped and the cuffs were never on her. An object joined by "and"
                # straight after an applied one, with no verb between, shares it.
                # "Dan takes the collar off", "pulls the gag out of her mouth": the
                # verb and its particle either side of the item. Only "takes off" in one
                # piece was a release, so these left the collar and the gag on for good.
                if release_at < 0 and _TAKE_AWAY.search(clause[:item_at]) \
                        and _off_after(beat, at, canon):
                    release_at = max(m.start() for m in _TAKE_AWAY.finditer(clause[:item_at]))
                # A TRY IS NOT A RELEASE: "tries to slip the cuffs off" leaves them on.
                if release_at >= 0 and _ATTEMPT.search(clause[:release_at]):
                    release_at = -1
                # The verb NEAREST the item is the one acting on it -- but not the item's
                # own word: "unlocks Bea's HANDCUFFS" is a release, even though
                # "handcuffs" is also a verb.
                if release_at >= 0 and apply_at > release_at and apply_at != item_at:
                    release_at = -1
                _shares = bool(apply_at < 0 and release_at < 0 and _last is not None
                               and _SHARED_OBJECT.match(beat[_last[1]:at]))
                if apply_at < 0 and release_at < 0 and not _shares:
                    continue
                local_who = names_in(clause, cast)
                if release_at >= 0:
                    # OFF THE PERSON WEARING IT, not the one doing the taking: "Dan
                    # removes her gag" looked for a gag on Dan, found none, and the gag
                    # stayed on her. REPORTED as restraints that would not stay put.
                    for _n in self._holders_of(canon, beat, at, local_who, subject,
                                               cast, pronouns, part):
                        _q = self.people[_n]
                        for key in [k for k in list(_q.hardware)
                                    if _same_piece(k, canon, part)]:
                            changed["released"].append((_n, _q.hardware.pop(key)))
                    continue
                wearer = (_last[0] if (_shares and not local_who)
                          else _wearer(clause, local_who or who, subject, cast,
                                       context=beat))
                p = self.person(wearer)
                # ONE GAG, NOT TWO: the bare word "gag" beside the thing it is made of,
                # in one beat, is that thing. The specific name wins either way round.
                if part == "mouth" and canon in _MOUTH_PIECES:
                    _twin = next((k for k, r in p.hardware.items()
                                  if k[1] == "mouth" and k[0] in _MOUTH_PIECES
                                  and k[0] != canon and r.applied_in == shot), None)
                    if _twin is not None:
                        if (written or canon) == "gag":
                            _last = (wearer, at)
                            continue            # the tape already says it
                        if p.hardware[_twin].item == "gag":
                            p.hardware.pop(_twin)
                            changed["applied"] = [(w, r) for w, r in changed["applied"]
                                                  if not (w == wearer and r.item == "gag")]
                # A MENTION NEVER STRIPS A RECORD. The same piece named again -- "checks
                # the cuffs are locked" -- keeps the fuller name it went on under and
                # what it is fastened to, unless this text gives new ones.
                _old = p.hardware.get((canon, part))
                _item = written or canon
                _pos = _nearest(position_spans(beat), at, spans)
                _anc = _nearest(anchor_spans(beat), at, spans)
                if _old is not None:
                    if _old.item and len(_old.item) > len(_item) \
                            and _item.split()[-1] in _old.item:
                        _item = _old.item
                    _pos = _pos or _old.position
                    _anc = _anc or _old.anchor
                p.hardware[(canon, part)] = Restraint(_item, part, _pos, _anc, shot)
                changed["applied"].append((wearer, p.hardware[(canon, part)]))
                _last = (wearer, at)
        # A PERSON FREED, rather than an item: "the guard releases her", "Dan unties
        # Ana's wrists", "Ana is uncuffed". This used to be any release word and any
        # "her" anywhere in the beat -- so "Dan releases her ARM", "unties her HAIR",
        # "pulls off her SNEAKERS" and "Ana tries to pull her hands free" each took
        # every restraint she had off at once. REPORTED as restraints breaking and
        # coming undone when nothing in the script undid them. See person_releases.
        for _n, _part, _only in person_releases(beat, cast, pronouns, subject, self):
            _q = self.people.get(_n)
            if _q is None:
                continue
            _keys = [k for k in list(_q.hardware)
                     if (_part is None or k[1] == _part)
                     and (_only is None or k[0] in _only)]
            # A verb that names a FACE item frees that part of the face and nothing
            # else. "Dan ungags her" over tape pressed on her mouth found no "gag",
            # fell through to the line below, and took off the handcuffs as well.
            # REPORTED as restraints coming undone when nothing undid them.
            _face = {"gag": "mouth", "blindfold": "eyes"}
            _face_part = next((_face[c] for c in (_only or ()) if c in _face), None)
            if not _keys and _face_part:
                _keys = [k for k in list(_q.hardware) if k[1] == _face_part]
            elif not _keys and _only is not None:
                # "Dan unties her" -- or her wrists -- when she is cuffed, not tied: the
                # author freed her, or that part of her, whatever the verb suits.
                _keys = [k for k in list(_q.hardware) if _part is None or k[1] == _part]
            for key in _keys:
                changed["released"].append((_n, _q.hardware.pop(key)))

        # What a partial strip keeps is named after "takes off", and is not coming off.
        _keeps = strips_to(beat)
        _kept_keys = {_garment_key(k) for k in (_keeps or [])}
        if subject:
            for m in garments:
                g = f"{(m.group(1) or '').strip()} {m.group(2)}".strip().lower()
                key = _garment_key(g)
                if key in _kept_keys:
                    continue
                clause, lo = _clause_at(beat, m.start(), boundaries)
                item_at = m.start() - lo
                actions = [(x.start(), "off") for x in _TAKES_OFF.finditer(clause)
                           if x.start() <= item_at]
                actions += [(x.start(), "on") for x in _PUTS_ON.finditer(clause)
                            if x.start() <= item_at]
                actions += [(x.start(), "aside") for x in _DISPLACES.finditer(clause)
                            if x.start() <= item_at]
                action = max(actions, default=(-1, ""))[1]
                if not action and _ONLY_A_DETERMINER.fullmatch(clause[:item_at]):
                    _lo = lo
                    for _ in range(6):
                        _b = next((x for x in boundaries if x.end() == _lo), None)
                        if not _b or _b.group(0).lower() not in ("and", ",", ";"):
                            break
                        prev, _plo = _clause_at(beat, max(0, _b.start() - 1), boundaries)
                        back = [(x.start(), "off") for x in _TAKES_OFF.finditer(prev)]
                        back += [(x.start(), "on") for x in _PUTS_ON.finditer(prev)]
                        back += [(x.start(), "aside") for x in _DISPLACES.finditer(prev)]
                        action = max(back, default=(-1, ""))[1]
                        if action or _plo >= _lo:
                            break
                        _lo = _plo
                if action == "aside" and _OPENS_GARMENT.search(clause[:item_at]):
                    if _COMPLETES_OFF.search(re.split(r"[.;!?]", beat[m.end():])[0]):
                        action = "off"
                if not action and _COMES_OFF_HERE.match(beat[m.end():]):
                    action = "off"
                local_who = names_in(clause, cast)
                # Off or aside, "her" says whose it is. Going ON it may be the giver's:
                # "Ana puts her coat on Bea".
                wearer_g = ((possessor_at(beat, m.start(), list(cast or self.people),
                                          subject, pronouns)
                             if action in ("off", "aside") else "")
                            or _wearer(clause, local_who, subject))
                p = self.person(wearer_g)
                if action == "off":
                    if key not in [_garment_key(x) for x in p.removed]:
                        p.removed.append(g)
                        changed["removed"].append((wearer_g, g))
                    p.worn = [x for x in p.worn if _garment_key(x) != key]
                    p.displaced = [x for x in p.displaced
                                   if _garment_key(x) != key]
                    _bare_on(p, regions_of(g))
                elif action == "on":
                    if key not in [_garment_key(x) for x in p.worn]:
                        p.worn.append(g)
                        changed["worn"].append((wearer_g, g))
                    p.removed = [x for x in p.removed if _garment_key(x) != key]
                    p.displaced = [x for x in p.displaced
                                   if _garment_key(x) != key]
                    _bare_off(p, regions_of(g))
                elif action == "aside":
                    if key not in [_garment_key(x) for x in p.displaced]:
                        p.displaced.append(g)
                        changed["displaced"].append((wearer_g, g))

        _strip = STRIPS_BARE.search(beat or "")
        _nude = nudity_in(beat) or (["torso", "legs", "feet"] if _strip else [])
        if _nude:
            nude_at = min([m.start() for rx, _regions in _NUDITY_RX
                           for m in rx.finditer(beat)]
                          + ([_strip.start()] if _strip else []),
                          default=len(beat))
            located = [(abs(beat.find(n) - nude_at), n) for n in who if beat.find(n) >= 0]
            # "Kate and Dan undress" is BOTH of them. The nearest name to the cue was
            # Dan alone, so every later shot described his body and said nothing of
            # hers -- and a bare region nobody describes is drawn from the prior.
            # Only for a VERB: "Ana looks at topless Bea" is still the nearest name.
            _doers = (_subjects_before(beat, _strip.start(), who)
                      if _strip and _strip.start() == nude_at
                      and _STRIPPING.match(_strip.group(0)) else [])
            owners = (undressed_object(beat, list(cast or self.people), subject, pronouns)
                      or _doers
                      or ([min(located)[1]] if located else ([subject] if subject else [])))
            for n in owners:
                q = self.person(n)
                _bare_on(q, _nude)
                for g in list(q.worn):
                    if any(r in _nude for r in regions_of(g)):
                        q.worn.remove(g)
                        if _garment_key(g) not in [_garment_key(x)
                                                   for x in q.removed]:
                            q.removed.append(g)
                            changed["removed"].append((n, g))

        if _keeps is not None and not _nude:
            _to = STRIPS_TO.search(staged_text(beat))
            owners = (undressed_object(beat, list(cast or self.people), subject, pronouns)
                      or [_agent_before(staged_text(beat), _to.start() if _to else 0, who)
                          or subject])
            for n in [o for o in owners if o]:
                q = self.person(n)
                keep = [g for g in q.worn if _garment_key(g) in _kept_keys
                        or ("underwear" in _keeps and is_undergarment(g))]
                off = [g for g in q.worn if g not in keep]
                for g in off:
                    q.worn.remove(g)
                    if _garment_key(g) not in [_garment_key(x) for x in q.removed]:
                        q.removed.append(g)
                        changed["removed"].append((n, g))
                covered = {r for g in keep for r in regions_of(g)}
                _bare_on(q, [r for g in off for r in regions_of(g) if r not in covered])

        pose = posture_in(beat)
        if pose and subject:
            self.person(subject).posture = pose
            changed["posture"][subject] = pose

        return changed

    # -- writing the shot --------------------------------------------------
    def continuity(self, described=(), changed=None):
        """ONE paragraph, rendered from state. Every fact said once.

        The order is fixed -- hardware, what it holds, posture -- so a reader of
        the output can tell at a glance whether something is missing. The old
        engine emitted clauses in whatever order its readers happened to fire,
        which is why nobody noticed the handcuffs had stopped appearing."""
        changed = changed or {}
        out = []
        names = [n for n in described if n in self.people] or list(self.people)
        applied_now = {r.item for _w, r in changed.get("applied", [])}

        for name in names:
            p = self.people.get(name)
            if not p or not p.hardware:
                continue
            items = [r.item for r in p.hardware.values()]
            subject = _join(items)
            many = len(items) > 1 or _plural(items[0])
            if applied_now & set(items):
                out.append(
                    f"The {subject} {'go' if many else 'goes'} on during this "
                    f"shot: open and off the body at the first frame, closed on "
                    f"it by the last.")
            else:
                who = f" on {name}" if len(names) > 1 else ""
                soft = all(not r.rigid for r in p.hardware.values())
                shut = "tied and holding" if soft else "closed and fastened"
                out.append(
                    f"The {subject}{who} {'stay' if many else 'stays'} {shut} "
                    f"as {'they were' if many else 'it was'} put on, the same "
                    f"object in the same material.")
            holds = [r.phrase() for r in p.hardware.values() if r.phrase()]
            if holds:
                out.append(f"{'They hold' if many else 'It holds'} {_join(holds)}.")
            if any(r.rigid for r in p.hardware.values()):
                out.append("The links keep their size and the run between them "
                           "stays taut.")

        off_now = {g for _w, g in changed.get("removed", [])}
        on_now = {g for _w, g in changed.get("worn", [])}
        aside_now = {g for _w, g in changed.get("displaced", [])}
        for name in names:
            p = self.people.get(name)
            if not p:
                continue
            for g in p.removed:
                if g in off_now:
                    out.append(f"The {g} is on the body as the shot opens and "
                               f"fully off it by the last frame, taken off "
                               f"during this shot.")
                else:
                    out.append(f"The {g} is off the body and stays where it "
                               f"was put.")
            for g in p.worn:
                if g in on_now:
                    out.append(f"The {g} is off the body as the shot opens and "
                               f"fully on by the last frame, put on during this "
                               f"shot.")
            for g in p.displaced:
                if g in aside_now:
                    out.append(f"The {g} is moved aside during this shot and "
                               f"stays on the body.")
                else:
                    out.append(f"The {g} is still pulled aside, still on the body.")

        for name in names:
            p = self.people.get(name)
            if p and p.posture and name not in (changed.get("posture") or {}):
                who = name if len(names) > 1 else "The body"
                out.append(f"{who} is still {p.posture}.")

        if self.place and self.place != self.opened_in and not changed.get("moved_to"):
            out.append(f"The shot is in the {self.place}, "
                       f"not the room the scene text names.")

        return " ".join(out)


def _garment_key(g):
    """Two wordings of the same garment are one garment: "blue shorts" and
    "shorts" must not both be tracked, or the shot lists a spare pair."""
    return re.sub(r"\s+", " ", (g or "").lower()).split()[-1].rstrip("s")


_RIGID = _rx(r"\b(?:handcuffs?|cuffs?|chains?|shackles?|manacles?|leg\s?irons?|"
             r"spreader\s+bars?|steel|iron|metal|padlock|chrome|brass)\b")
_SOFT = _rx(r"\b(?:rope|ropes|cord|cords|twine|string|tape|scarf|stocking|"
            r"stockings|zip\s?ties?|cable\s?ties?|laces?)\b")


def _is_rigid(item):
    """Does this thing hold its shape? Soft goods never do, whatever else the
    phrase says -- "steel" in a sentence about rope does not make rope steel."""
    t = item or ""
    return bool(_RIGID.search(t)) and not _SOFT.search(t)


def _nearest(mods, at, spans):
    """The modifier belonging to the item at `at`, or ''.

    A modifier belongs to the last item mentioned before it -- English puts the
    qualifier after the thing it qualifies. So with items at 10 and 60 and a
    modifier at 75, the modifier is the second item's; a modifier at 20 is the
    first item's."""
    starts = sorted(s for _c, _p, _w, s in spans)
    mine = ""
    for name, where in mods:
        owner = max((s for s in starts if s <= where), default=starts[0])
        if owner == at:
            mine = mine or name
    return mine


_HARDWARE_VERB = (r"cuffs|ties|chains|straps|tapes|binds|locks|padlocks|shackles|"
                  r"manacles|hobbles|leashes|collars|gags|blindfolds|trusses|"
                  r"hog-?ties|restrains|fetters|pinions|"
                  r"tightens|cinches|buckles|fastens|clips|snaps")
_APPLY_ANY = re.compile(r"\b(?:" + APPLY_VERB + r"|" + _HARDWARE_VERB + r")\b", re.I)
_APPLIED_TO_PRONOUN = re.compile(
    r"\b(?:" + APPLY_VERB + r"|" + _HARDWARE_VERB + r")\b"
    r"(?:\s+\S+){0,3}?\s+(?:her|him|them)\b"
    r"|\b(?:" + APPLY_VERB + r"|" + _HARDWARE_VERB + r")\b[^.;!?]{0,40}?"
    r"\b(?:" + "|".join(p for p, _n, _pt in HARDWARE) + r")\b"
    r"(?:\s+\S+){0,2}?\s+(?:on|onto|around|round|about|over|under|to|behind|"
    r"between)\s+(?:her|him|them)\b"
    # "Presses duct tape over HER mouth": APPLY_VERB swallows the whole phrase, part
    # and all, so neither form above ever saw the "her" -- and the tape went on the
    # man pressing it. REPORTED as the gag disappearing: it was never on her at all.
    r"|\b(?:presses|pressed|pressing|sticks?|stuck|slaps?|slapped|smooths?|smoothed|"
    r"plasters?|plastered|puts?|places?|placed|wraps?|wrapped|winds?|wound)\b"
    r"[^.;!?]{0,60}?\b(?:over|across|on|onto|around|round)\s+(?:her|his|their)\s+"
    r"(?:mouth|lips|eyes|face|wrists?|ankles?|hands?|neck|throat)\b", re.I)


def _wearer(beat, who, fallback, cast=(), context=""):
    """Who the hardware goes ON. The agent is not the wearer.

    "The guard cuffs Ana" puts them on Ana; "Ana is cuffed by the guard" puts
    them on Ana too, and naive sentence order gets the second one backwards.
    Describing hardware on somebody the text never put it on gives the model
    wrists belonging to nobody, which is how a second figure gets invented to
    own them."""
    if len(who) < 2:
        hit = _APPLIED_TO_PRONOUN.search(beat or "") if who else None
        if hit and re.match(r"\s+own\b", (beat or "")[hit.end():]):
            hit = None                      # "her OWN mouth" is the doer's
        if hit:
            name = re.search(r"\b" + re.escape(who[0]) + r"\b", beat or "")
            between = (beat or "")[name.end():hit.start()] if name else ""
            # ...or named in an EARLIER clause: "Dan ties her wrists and tapes her
            # mouth" hands this "tapes her mouth", with no Dan in it to find, and the
            # tape went on him. The object is still the other person -- when the name
            # is the DOER over there, plainly, before the first fastening verb. Not
            # "handcuffs ANA'S wrists and locks a collar around her neck": she is the
            # one it is done to, and so is the "her".
            _doer = False
            if name is None and context:
                _first = _APPLY_ANY.search(context)
                _at = re.search(r"\b" + re.escape(who[0]) + r"\b(?!['’]s)", context)
                _doer = bool(_first and _at and _at.start() < _first.start())
            if _doer or (name and name.start() < hit.start()
                        and len(between.split()) <= 4 and not re.search(r"[,;:]", between)):
                others = [n for n in (cast or []) if n and n != who[0]]
                if len(others) == 1:
                    return others[0]
        return who[0] if who else fallback
    passive = re.search(r"\bby\s+(?:the\s+)?(\w+)", beat or "", re.I)
    agent = None
    if passive:
        agent = next((n for n in who
                      if n.lower() == passive.group(1).lower()), None)
    if agent is None:
        hit = None
        for m in _APPLY_ANY.finditer(beat or ""):
            before = (beat or "")[:m.start()].rstrip().split()
            if before and re.fullmatch(r"(?:\w+'s|her|his|their|its|our|my|your|the|a|an)",
                                       before[-1], re.I):
                continue
            hit = m
            break
        nearest = None
        if hit:
            for name in who:
                for m in re.finditer(r"\b" + re.escape(name) + r"\b", beat or ""):
                    if m.start() < hit.start() and (nearest is None or m.start() > nearest[0]):
                        nearest = (m.start(), name)
        agent = nearest[1] if nearest else who[0]
    return next((n for n in who if n != agent), fallback)


_PRONOUN_GROUP = {"her": "she", "hers": "she", "him": "he", "his": "he",
                  "them": "they", "their": "they", "theirs": "they"}


def _by_pronoun(word, cast, pronouns, subject, reflexive):
    """The one person in `cast` a pronoun can mean here, or "".

    `reflexive` is the possessive reading -- "Kate takes off HER shirt" is her own
    shirt -- so the subject wins whenever the pronoun fits them. The object reading
    -- "Dan undresses HER" -- is never the subject; that would be "herself".

    `pronouns` is {name: "she"|"he"|"they"} off the sheet. Where nobody declared
    the pronoun, the possessive stays with the subject, which is how this read
    before it knew any pronouns, and the object is the only other person or nobody."""
    group = _PRONOUN_GROUP.get(str(word or "").lower())
    people = [n for n in (cast or []) if n]
    if not group or not people:
        return ""
    fits = [n for n in people if (pronouns or {}).get(n) == group]
    if not fits:
        if group == "they":
            # Nobody's own pronoun: it is plural -- "McKenna and Tess take off
            # THEIR shirts" -- and no one person is the answer.
            return ""
        if reflexive:
            return subject or ""
        others = [n for n in people if n != subject]
        return others[0] if len(others) == 1 else ""
    if reflexive and subject in fits:
        return subject
    others = [n for n in fits if n != subject]
    return others[0] if len(others) == 1 else ""


_POSSESSIVE = re.compile(r"\b(?:([A-Z][\w’'-]*?)['’]s|((?i:her|his|their)))\s+")
_OWNER_BREAK = _rx(r"[.;:!?]|\b(?:the|a|an|this|that|these|those|is|are|was|were|has|have|"
                   r"had|then|puts?|putting|wears?|wearing|" + _STRIP_VERB + r"|"
                   + _UNDO_VERB + r")\b")


def possessor_at(text, at, cast, subject="", pronouns=None):
    """Whose garment the one named at `at` is, off the possessive in front of it.

    "" when nothing in front of it says. "Dan takes off her jacket and shirt" is two
    of HER garments, and reading the wearer off the sentence's subject instead put
    them on Dan -- who wears a shirt too, so his came off the sheet, the shot called
    his chest bare, and hers stayed listed in every shot after. The possessive
    reaches over a list ("her jacket and shirt") but not over a verb or a new
    determiner, which would make it a different phrase's."""
    before = str(text or "")[:max(0, int(at))]
    m = None
    for m in _POSSESSIVE.finditer(before):
        pass
    if not m:
        return ""
    between = before[m.end():]
    if len(between.split()) > 5 or _OWNER_BREAK.search(between):
        return ""
    if m.group(1):
        return next((n for n in (cast or []) if n and n == m.group(1)), "")
    return _by_pronoun(m.group(2), cast, pronouns, subject, reflexive=True)


_UNDRESSES = _rx(r"\b(?:undress(?:es|ed|ing)?|strip(?:s|ped|ping)?)\s+([\w’'-]+)"
                 r"(?:\s+([\w-]+))?")
_HELPS_UNDRESS = _rx(r"\bhelp(?:s|ed|ing)?\s+([\w’'-]+)\s+(?:to\s+)?"
                     r"(?:undress|strip|get\s+undressed)\b")
_CLOTHES = r"(?:clothes|clothing|garments|things|kit|outfit|gear)"
_TAKES_CLOTHES = _rx(r"\b(?:undress|strip|take|took|pull|peel|remove|get|got|slip|shed)\w*"
                     r"(?:\s+(?:off|out\s+of))?\s+(?:all\s+(?:of\s+)?)?"
                     r"(her|his|their|[A-Z][\w’'-]*?['’]s)\s+" + _CLOTHES + r"\b")
_CLOTHES_WORD = _rx(r"^(?:" + _CLOTHES + r"|" + GARMENT_PHRASES + r"|" + GARMENT_WORDS
                    + r")s?$")


_STRIPPING = _rx(r"(?:undress|strip|take|took|pull|peel|shed|remove|get|got|slip)")


def _subjects_before(text, at, who):
    """Everyone the clause names in front of `at`: "Kate and Dan undress" is both.

    The same reading sampler.strips_who makes, so the two halves of the node agree
    about who ended up with nothing on."""
    before = str(text or "")[:max(0, int(at))]
    cut = max((m.end() for m in re.finditer(
        r"[.;!?]\s+|,\s*|\s+(?:as|while|and\s+then|then|but)\s+", before)), default=0)
    span = before[cut:]
    return [n for n in (who or []) if re.search(r"\b" + re.escape(n) + r"\b", span)]


def _agent_before(text, at, people):
    """The person the sentence names last before `at`. "" when it names nobody."""
    sent = re.split(r"[.;!?]", str(text or "")[:at])[-1]
    hits = [(m.start(), n) for n in people
            for m in re.finditer(r"\b" + re.escape(n) + r"\b", sent)]
    return max(hits)[1] if hits else ""


def undressed_object(beat, cast, subject="", pronouns=None):
    """[who] a beat undresses when somebody ELSE does the undressing. [] otherwise.

    "Dan undresses Kate" and "Dan strips her naked" put Dan in front of the cue, and
    the reader that takes the name before the cue undressed him: Kate stayed in her
    clothes for the rest of the film and every later shot described Dan as bare.
    The person after the verb is the one it happens to. A possessive -- "takes off
    her clothes" -- names them too, but reads as the subject's own when it fits."""
    staged = staged_text(beat or "")
    people = [n for n in (cast or []) if n]
    for rx in (_HELPS_UNDRESS, _UNDRESSES, _TAKES_CLOTHES):
        for m in rx.finditer(staged):
            word = re.sub(r"['’]s$", "", m.group(1)) if rx is _TAKES_CLOTHES \
                else m.group(1)
            name = next((n for n in people if n == word), "")
            if name:
                return [name]
            low = word.lower()
            if low not in _PRONOUN_GROUP:
                continue
            nxt = (m.group(2) or "") if rx is _UNDRESSES else ""
            possessive = (rx is _TAKES_CLOTHES or low in ("his", "their")
                          or (low == "her" and bool(_CLOTHES_WORD.match(nxt))))
            agent = _agent_before(staged, m.start(), people) or subject
            got = _by_pronoun(low, people, pronouns, agent, possessive)
            if got:
                return [got]
    return []


# The garment as the SUBJECT of its own removal: "Kate's jacket comes off", "her
# dress drops to the floor", "the shirt is pulled off". Every removal reader here
# starts at a verb and reads its object, so a garment that comes off by itself, or
# in the passive, came off nobody and stayed listed on every later shot. Off a THING
# -- "off the rack", "off its hanger" -- is not off the body.
_OFF = r"(?:off|away)\b(?!\s+(?:the|a|an|its|this|that)\b)"
COMES_OFF = (r"(?:\s+(?:slowly|then|finally|now|quickly|gently|easily|\w+ly))*\s+(?:"
             r"(?:comes?|came|coming)\s+(?:right\s+|straight\s+)?" + _OFF +
             r"|(?:falls?|fell|falling|drops?|dropped|dropping|slides?|slid|sliding|"
             r"slips?|slipped|slipping|slithers?|slithered|tumbles?|tumbled|pools?|"
             r"pooled|puddles?|puddled|crumples?|crumpled)\s+"
             r"(?:(?:right\s+|straight\s+)?" + _OFF +
             r"|down\s+(?:her|his|their)\s+(?:legs?|body|hips|thighs)\b"
             r"|(?:to|around|at|about)\s+(?:her|his|their)\s+(?:ankles?|feet)\b"
             r"|" + TO_THE_FLOOR + r")"
             r"|(?:is|are|was|were|gets?|got|has\s+been|have\s+been|had\s+been)\s+"
             r"(?:\w+ly\s+)?(?:(?:removed|discarded|shed)\b|" + _OFF +
             r"|(?:taken|pulled|stripped|peeled|cut|torn|ripped|yanked|tugged|slipped|"
             r"eased|worked|wriggled|kicked|thrown|tossed|shrugged)\s+"
             r"(?:" + _OFF + r"|aside\b)))")
GARMENT_COMES_OFF = _rx(r"\b([\w-]{3,})" + COMES_OFF)
_COMES_OFF_HERE = _rx(r"^" + COMES_OFF)

_UNDER_WORD = _rx(r"\b(?:underwear|underclothes|underclothing|undies|lingerie|smalls)\b")
_STRIP_ANY = _rx(r"\b(?:" + _STRIP_VERB + r"|" + _UNDO_VERB + r")\b")
STRIPS_TO = _rx(
    r"\b(?:strip(?:s|ped|ping)?|undress(?:es|ed|ing)?)(?:\s+(?:right\s+)?down)?"
    r"(?=" + _KEEPS + r")\s+to\s+([^.;!?]+)"
    r"|\b(?:everything|all\s+(?:of\s+)?(?:her|his|their)\s+(?:clothes|clothing))"
    r"(?:\s+(?:off|away))?\s+(?:but|except(?:\s+for)?|apart\s+from|save\s+for)\s+"
    r"([^.;!?]+)")
_KEPT_FILLER = frozenset("""
her his their a an the and or just only nothing but pair pairs of matching
""".split())
_KEPT_STOP = _rx(r"^(?:" + _STRIP_VERB + r"|" + _UNDO_VERB + r"|puts?|sits?|stands?|lies?|"
                 r"climbs?|walks?|turns?|looks?|goes|heads?|moves?|kneels?|lays?|folds?|"
                 r"drops?|hangs?|leaves?|waits?|shivers?|shivering|then|before|while|as|"
                 r"so|until|into|onto|in|on|at|with)$")
_ADJ_ONE = _rx(r"^" + _ADJ + r"$")


def strips_to(beat):
    """The garment words a PARTIAL strip keeps on, or None when there is none.

    "strips to her bra and panties", "strips down to his boxers", "takes off
    everything but her socks". Read as nothing coming off at all, the sweater and
    jeans stayed in every later shot; read as a full strip, so did nothing -- the
    underwear went too. `underwear` is in the list when the beat keeps the
    underwear without naming it, for the caller to match against the sheet."""
    staged = staged_text(beat or "")
    for m in STRIPS_TO.finditer(staged):
        tail = m.group(1) if m.group(1) is not None else m.group(2)
        if m.group(2) is not None:
            sent = re.split(r"[.;!?]", staged[:m.start()])[-1]
            if not _STRIP_ANY.search(sent):
                continue
        kept, unknown = [], 0
        for tok in re.findall(r"[\w'’-]+|[,;]", tail):
            low = tok.lower()
            if _KEPT_STOP.match(low):
                break
            words = garment_words(tok)
            if words or _UNDER_WORD.search(low):
                kept += [w for w in words if w not in kept]
                if _UNDER_WORD.search(low) and "underwear" not in kept:
                    kept.append("underwear")
                unknown = 0
                continue
            if low in _KEPT_FILLER or tok in ",;" or _ADJ_ONE.match(low):
                continue
            unknown += 1
            if unknown > 2:
                break
        if kept:
            return kept
    return None


_LEADING_ARTICLE = re.compile(r"^(?:a|an|the|her|his|their|its)\s+", re.I)


def bare_name(text):
    """A garment or object name with any leading article or possessive taken off."""
    return _LEADING_ARTICLE.sub("", str(text or "").strip())


def applies_hardware(beat):
    """Does this beat FASTEN something onto somebody?

    The same test SceneState.read makes before it records anything: hardware named,
    an applying verb, and not a release. Exposed because the sampler had its own
    hand-written list of applying verbs that knew 22 of them, so a beat that hogtied
    somebody was not "hardware going on" as far as that half of the node was
    concerned -- and the reader that decides WHOSE it is was never consulted."""
    text = beat or ""
    return bool(hardware_spans(text)) and bool(_APPLY.search(text)) and not _RELEASE.search(text)


def wearer_of(beat, cast=()):
    """Who this beat puts hardware ON, by the same reading the state uses.

    "" when the sentence does not settle it. Exposed because the sampler has its own
    reading of who is restrained, and the two disagreed: a beat naming only the person
    DOING it recorded the restraint on them, so every shot about the person actually
    in it was told the hold belonged to nobody present. See _wearer."""
    who = names_in(beat or "", cast)
    if not who:
        return ""
    return _wearer(beat or "", who, who[0], cast)


def held_part_of(items):
    """The body part these items hold, as a plural noun for a sentence.

    A limb position describes the ARMS, so this answers "wrists" for anything
    that holds them and defers to the item otherwise -- a collar's position is
    never a limb position, and the constructor already refuses to give it one.

    Each item answers for itself, and a limb wins over a neck: this used to return the
    first part that was not the wrists, so steel handcuffs beside a leather collar came
    out "the neck are already behind the back when the hardware closes"."""
    parts = []
    for it in items or []:
        for piece in re.split(r",|\s+and\s+", str(it or "")):
            for pat, _n, part in HARDWARE:
                if re.search(r"\b(?:" + pat + r")\b", piece, re.I):
                    parts.append(part)
                    break
    if not parts or "wrists" in parts:
        return "wrists"
    limbs = [p for p in parts if p in ("ankles", "arms")]
    return limbs[0] if limbs else parts[0]


def staged_applications(beats):
    """{canonical hardware -> the 1-based beat that first puts it on}.

    Read once, before anything renders, because a sheet cannot say WHEN. Where
    the script stages a fastening, no earlier shot may be told that thing is
    already fastened -- that is a cuff on a wrist two shots before the cuffing."""
    out = {}
    for i, b in enumerate(beats or [], 1):
        b = b or ""
        if not _APPLY.search(b) or _RELEASE.search(b):
            continue
        for canon, _part, _w, at in hardware_spans(b):
            if re.search(r"\bto\s+(?:the|a|an|her|his|their)\s*$", b[:at], re.I):
                continue
            out.setdefault(canon, i)
    return out


def _plural(item):
    return item.endswith("s") and not item.endswith("ss")


def _join(items):
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]
