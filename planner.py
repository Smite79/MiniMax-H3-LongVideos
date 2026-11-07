# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import re

FPS = 24
MAX_FRAMES = 362
MIN_FRAMES = 124
BEAT_BASE = 0.8
PER_ACTION = 2.2
HOBBLE = 6.0
SETTLE = 1.5
RUSH = 1.15
WORDS_PER_SEC = 2.8
SAY_SLACK = 0.6
SENTENCE_PAUSE = 0.35
MUFFLE_MAX = 2.5
MUFFLE_SOUND = 1.5
CUE_GAP = 0.4
SPEECH_LEAD = 0.5
SPEECH_TAIL = 0.8

_UP = "A-ZÀ-ÖØ-ÞĀ-ɏ"
_LOW = "a-zß-öø-ÿĀ-ɏ"
_WORD = rf"[{_UP}](?:['’][{_UP}])?[{_LOW}]+(?:[{_UP}][{_LOW}]+)*(?:-[{_UP}][{_LOW}]+(?:[{_UP}][{_LOW}]+)*)*"
NAME = rf"(?-i:{_WORD}(?:\s+{_WORD}){{0,2}})"
_PIC = r"<\s*picture[\s_\-]*(\d+)\s*>"
STATE_TAGS = ("hold", "release", "wear", "remove", "tear", "lift")
_TAG = re.compile(r"^\s*(hold|release|wear|remove|tear|lift|pull|exit|seconds)\s*:\s*(.*?)\s*$", re.I)
_INLINE = re.compile(r"(?<=[.!?\"”])\s+(?=(?:hold|release|wear|remove|tear|lift|pull|exit|seconds)\s*:)", re.I)
_CUT = re.compile(r"^\s*cut\s*[:.!]?\s*$", re.I)
_HEAD = re.compile(rf"^\s*(?:[-*•>#]+\s*|\d+[.)]\s+)?(?:\*\*|__)?(?P<name>{NAME}|[{_UP}]{{2,}}(?:\s+[{_UP}]{{2,}}){{0,2}})"
                   rf"(?:\*\*|__)?(?:\s*\([^)<]*\))?(?:\s*,?\s*\(?\s*(?i:<\s*picture[\s_\-]*(?P<pic>\d+)\s*>)\s*\)?)?(?:\s*\([^)<]*\))?"
                   rf"(?:\*\*|__)?\s*(?P<sep>[:=]|[-–—]\s)?\s*(?P<rest>.*)$")
_TAGGED = re.compile(r"(\(\s*)?<\s*picture[\s_\-]*(\d+)\s*>(\s*\))?", re.I)
_SPEECH = re.compile(r"\"[^\"]*\"|“[^”]*”|<\s*d\s*>.*?<\s*/\s*d\s*>"
                     r"|(?<![\w'’])['‘](?=[^\s'‘’])(?:[^'‘’\n]|(?<=\w)['’](?=\w))+?(?<=[^\s'‘’])['’](?![\w'’])", re.S | re.I)
_STOP = re.compile(r"(?<=[.!?])(?<!\.\.\.)(?<!Mr\.)(?<!Ms\.)(?<!Dr\.)(?<!St\.)(?<!Mrs\.)\s+(?=[\"“‘']?[A-Z0-9])")
_EDGES = re.compile(r"^\s*(?:<\s*d\s*>|[\"“'‘])|(?:<\s*/\s*d\s*>|[\"”'’])\s*$", re.I)
_SAYS = (r"says?|said|asks?|asked|chants?|chanted|whispers?|whispered|murmurs?|murmured|mutters?|muttered|growls?|growled|hiss(?:es)?|hissed|"
         r"snaps?|snapped|shouts?|shouted|yells?|yelled|screams?|screamed|cries|cried|sobs?|sobbed|pleads?|pleaded|begs?|begged|"
         r"demands?|demanded|replies|replied|answers?|answered|adds?|added|continues?|continued|purrs?|purred|breathes?|breathed|"
         r"laughs?|laughed|giggles?|giggled|sighs?|sighed|moans?|moaned|groans?|groaned|mumbles?|mumbled|stammers?|stammered|"
         r"warns?|warned|orders?|ordered|commands?|commanded|tells?|told|calls?|called|teases?|teased|taunts?|taunted|coos?|cooed|"
         r"croons?|crooned|gasps?|gasped|whimpers?|whimpered|squeals?|squealed|insists?|insisted|admits?|admitted|promises?|"
         r"promised|repeats?|repeated|announces?|announced|declares?|declared|responds?|responded|retorts?|retorted|protests?|"
         r"protested|panted|chokes?|choked|grunts?|grunted|blurts?|blurted|sneers?|sneered|scoffs?|scoffed|jokes?|joked|"
         r"counters?|countered|goes\s+on|went\s+on|tries|tried|manages?|managed|starts?|started|begins?|began|croaks?|croaked|"
         r"rasps?|rasped|wheezes?|wheezed|squeaks?|squeaked|yelps?|yelped|whines?|whined|mimics?|mimicked|mocks?|mocked|"
         r"(?:shoots?|shot|fires?|fired|snaps?|snapped)\s+back|(?:calls?|called|cries|cried)\s+(?:out|after)")
_TRIES = re.compile(r"tries|tried|manages?|managed|starts?|started|begins?|began|continues?|continued", re.I)
_SOUNDS = re.compile(r"laughs?|laughed|giggles?|giggled|sighs?|sighed|sneers?|sneered|scoffs?|scoffed|grunts?|grunted|"
                     r"gasps?|gasped|sobs?|sobbed|moans?|moaned|groans?|groaned|chokes?|choked", re.I)
_MEEKSAY = re.compile(r"\b(?:mumbles?|mumbled|whimpers?|whimpered|pleads?|pleaded|begs?|begged|sobs?|sobbed|moans?|moaned|groans?|"
                      r"groaned|protests?|protested|gasps?|gasped|cries|cried|squeals?|squealed|chokes?|choked|panted|yelps?|yelped|"
                      r"croaks?|croaked|whines?|whined|squeaks?|squeaked|wheezes?|wheezed|rasps?|rasped)\b", re.I)
_NOISE = (r"screams?|screamed|shrieks?|shrieked|yells?|yelled|shouts?|shouted|cries|cried|sobs?|sobbed|moans?|moaned|groans?|"
          r"groaned|whimpers?|whimpered|whines?|whined|squeals?|squealed|yelps?|yelped|grunts?|grunted|mumbles?|mumbled|murmurs?|"
          r"murmured|hums?|hummed|wails?|wailed|howls?|howled|squeaks?|squeaked|mewls?|mewled|pleads?|pleaded|begs?|begged|"
          r"protests?|protested")
_VOICE = re.compile(r"\bvoice\s*(?:is|:)\s*([^.;\n]+)|\b(?:a|an|her|his|their)\s+((?:[\w-]+,?\s+){1,3}?)voice\b", re.I)
_LABELS = {"name", "names", "character", "characters", "looks", "look", "hair", "eyes", "outfit", "outfits", "clothes",
           "clothing", "wears", "wearing", "age", "height", "build", "body", "face", "skin", "personality", "voice", "role",
           "description", "appearance", "gender", "sex", "race", "species", "type", "class", "details", "features",
           "accessories", "shoes", "makeup", "tattoos", "piercings", "background", "traits", "nationality", "ethnicity",
           "occupation", "job", "relationship", "she", "he", "they", "her", "his", "their", "it", "the"}
_SCENES = {"setting", "location", "scene", "mood", "lighting", "light", "time", "place", "weather", "music", "audio", "sound",
           "camera", "shot", "style", "note", "notes"}
_SHE = re.compile(r"\b(?:woman|women|girl|lady|female|wife|mother|sister|daughter|girlfriend|she|her|herself)\b", re.I)
_HE = re.compile(r"\b(?:man|men|boy|guy|male|husband|father|brother|son|boyfriend|he|his|him|himself)\b", re.I)
_HAIR = re.compile(r"\b(?!(?:her|his|their|the|a|an|and|of|with|no)\b)([a-z]+(?:-[a-z]+)?)\s+hair\b", re.I)
_CAMERA = re.compile(r"\b(?:camera|close[\s-]?ups?|zoom(?:s|ed|ing)?|pans?|panning|tilts?|framed|framing|focus(?:es|ed)?\s+on|"
                     r"point\s+of\s+view|pov|over\s+(?:her|his|their)\s+shoulder|we\s+see|view\s+of|the\s+view|reveal(?:s|ed|ing)?|"
                     r"shot\s+of|wide\s+shot|full[\s-]body|angle)\b", re.I)
_FOLK = tuple((w.split("/")[0], (w.split("/") + [w + "s"])[1], x) for x, words in (
    ("m", "man/men guy boy gentleman/gentlemen policeman/policemen waiter salesman/salesmen watchman/watchmen doorman/doormen "
          "fireman/firemen postman/postmen repairman/repairmen handyman/handymen gunman/gunmen henchman/henchmen "
          "businessman/businessmen cameraman/cameramen fisherman/fishermen workman/workmen landlord butler priest monk"),
    ("f", "woman/women girl lady/ladies policewoman/policewomen waitress/waitresses saleswoman/saleswomen "
          "businesswoman/businesswomen landlady/landladies hostess/hostesses maid nun"),
    ("", "person/people stranger guard nurse doctor officer cop detective soldier bartender barista customer clerk cashier "
         "conductor receptionist janitor driver bouncer thug kidnapper intruder captor attacker agent worker bodyguard prisoner "
         "inmate child/children kid teenager neighbor figure onlooker passerby/passersby crowd host server vendor pilot teacher "
         "chef valet porter courier dealer mechanic paramedic medic firefighter sheriff deputy ranger sergeant captain sailor "
         "student tourist someone/somebody")) for w in words.split())
_FOLKS = "|".join(w for row in _FOLK for w in row[:2])
_FOLK_RX = re.compile(rf"\b(?:(?P<det>(?<!\bis )(?<!\bwas )(?<!\bas )(?<!\blike )(?:a|an|another|one|some|two|three|four|five|several|"
                      rf"a\s+few|a\s+group\s+of|a\s+pair\s+of|a\s+couple\s+of|the|this|that|these|those|both))\s+"
                      rf"(?:(?!(?:and|or|of|with|in|on|at|to|from)\b)[\w-]+\s+){{0,2}}?)?(?P<noun>{_FOLKS})\b", re.I)
_DET = r"(?:(?:a|an|another|one|the|this|that|these|those|both|two|three|four|five|some|several)\s+(?:[\w-]+\s+){0,2}?)?"
_AFTER = r"(?:\s+(?:in|with|by|beside|behind|near|at|from|wearing|holding)\s+(?:[\w'’-]+\s+){0,3}?[\w'’-]+)"
_TAGGED_AS = r"(?:\s+(?:in|on|over|with|by|beside|behind|near|at|from|wearing|holding)\s+(?:[\w'’-]+\s+){0,3}?[\w'’-]+)"
_AWAY = re.compile(r"\b(?:calls?|called|phones?|phoned|texts?|texted|emails?|emailed|thinks?\s+(?:of|about)|thought\s+(?:of|about)|"
                   r"remembers?|remembered|dreams?\s+(?:of|about)|mentions?|mentioned|talks?\s+about|talked\s+about|asks?\s+(?:about|for)|"
                   r"asked\s+(?:about|for)|waits?\s+for|waited\s+for|looks?\s+for|looked\s+for|searches\s+for|wants|needs|misses|"
                   r"imagines?|imagined|pictures?|pictured|like|than|about|(?:texts?|messages?|emails?|letters?|notes?|calls?|gifts?|"
                   r"packages?|photos?|pictures?)\s+from|(?:story|picture|photo|photograph|painting|drawing|image|memory|thought|dream|"
                   r"sketch|portrait|statue|video|recording|poster)\s+of)\s+$", re.I)
_THINGS = re.compile(r"(?:\s+(?:car|cars|rail|rails|dog|dogs|house|station|uniform|uniforms|badge|post|tower|booth|room|desk|seat|"
                     r"license|licence|door|gate|cap|hat|coat|jacket|shirt|boots|shoes|show|movie|film|magazine|costume|outfit)\b|"
                     r"['’]s\s+(?:office|house|home|room|car|apartment|place|bed|desk|shop|store|clinic))", re.I)
_SEEN = r"screens?|monitors?|computers?|laptops?|tablets?|tvs?|televisions?|webcams?|cameras?|feeds?|footage|recordings?|videos?|" \
        r"livestreams?|streams?|broadcasts?|news|(?:video|zoom|facetime|skype|teams)\s+(?:calls?|chats?)"
_HEARD = r"phones?|cell\s*phones?|smartphones?|radios?|intercoms?|speakerphones?|calls?|line"
_PICTURE = r"photos?|photographs?|pictures?|paintings?|portraits?|posters?|billboards?|magazines?|drawings?|sketch(?:es)?|mirrors?|reflections?"
_DEVICE = rf"(?:[\w-]+\s+)?(?:{_SEEN}|{_HEARD}|{_PICTURE})"
_ON_DEVICE = re.compile(rf"^(?:\s+[\w'’-]+){{0,3}}?\s+(?:on|in|over|through|via|from|inside)\s+(?:(?P<whose>the|a|an|her|his|their|its|"
                        rf"[\w-]+['’]s)\s+)?(?P<dev>{_DEVICE})\b", re.I)
_DEVICE_ON = re.compile(rf"(?:\b(?:on|in)\s+(?:the|a|an|her|his|their)\s+({_DEVICE})\s*,?\s*(?:with\s+|(?:\w+ing|and\s+\w+s)\s+"
                        rf"(?:\w+\s+)?(?:to|with|at)\s+)?|\b({_DEVICE})\s+(?:shows?|showed|displays?|"
                        rf"displayed|plays?|played)\s+|\bon\s+(?:a|the)\s+((?:[\w-]+\s+)?call)\s+with\s+|"
                        rf"\b(?:video[\s-]?calls?|facetimes?|skypes?|video[\s-]?chats?\s+with)\s+)$", re.I)
_TALKS = re.compile(r"\b(?:talks?|talking|talked|speaks?|speaking|spoke|chats?|chatting|chatted|argues?|arguing|argued|listens?|"
                    r"listening|listened|watches|watching|watched|looks?|looking|looked|stares?|staring|stared|glances?|waves?|nods?)\s+"
                    r"(?:\w+\s+)?(?:to|with|at)\s+$", re.I)
_HANGUP = re.compile(r"\b(?:hangs?\s+up|hung\s+up|ends?\s+the\s+(?:video\s+)?call|(?:closes?|shuts?|slams?)\s+(?:the|her|his|their)\s+"
                     r"laptop|turns?\s+off\s+(?:the|her|his|their)\s+(?:tv|television|computer|monitor|screen|phone|radio)|"
                     r"screen\s+goes\s+(?:dark|black|blank))\b", re.I)
_ITEMS = r"seat|side|door|window|license|licence|office|house|home|room|car|truck|van|apartment|flat|place|bed|desk|shop|store|" \
         r"clinic|area|section|quarters|station|booth|lounge|entrance|bathroom|restroom|toilet|locker|cabin|cell|bunk|chair|table|bench|" \
         r"post|position|spot|uniform|costume|outfit|clothes|name|number|photo|picture|grave|funeral|wedding|party|jacket|coat|shirt|" \
         r"hat|cap|shoes|boots|gloves|scarf|tie|wallet|keys|bag|purse|phone|watch|ring|necklace|glasses|money|card|wife|husband|" \
         r"daughter|son|mother|father|girlfriend|boyfriend|friend|family"
_GOES = re.compile(r"(?:\s+(?:in|on|with|wearing|holding)\s+(?:[\w'’-]+\s+){0,3}?[\w'’-]+)?(?:\s+\w+ly)?\s+(?:leaves|walks\s+(?:out|away|off|past|by)|runs\s+(?:out|off|away|past|by)|hurries\s+(?:out|off|"
                   r"away|past|by)|drives\s+(?:off|away|past|by)|passes(?:\s+by)?|disappears|exits|goes\s+(?:out|away))\b", re.I)
_MUSIC = re.compile(r"\b(?:music|song|sings?|singing|(?:live|rock|jazz|brass|wedding|marching|cover)\s+band|band\s+play(?:s|ing)?|"
                    r"radio|playlist|piano|guitar|melody)\b", re.I)
_VOCAL = re.compile(rf"\b(?:{_NOISE}|screaming|shrieking|yelling|shouting|crying|sobbing|moaning|groaning|whimpering|whining|squealing|"
                    r"yelping|grunting|mumbling|murmuring|humming|wailing|howling|pleading|begging|laugh(?:s|ed|ing)?|"
                    r"giggl(?:e|es|ed|ing)|chuckl(?:e|es|ed|ing)|sigh(?:s|ed|ing)?|gasp(?:s|ed|ing)?|cough(?:s|ed|ing)?|"
                    r"whistl(?:e|es|ed|ing))\b", re.I)
_TONES = re.compile(r"\b(?:bark(?:s|ed|ing)?|meow(?:s|ed|ing)?|purr(?:s|ed|ing)?|chirp(?:s|ed|ing)?|tweet(?:s|ed|ing)?|birdsong|"
                    r"sirens?|horns?|honk(?:s|ed|ing)?|bells?|chimes?|alarms?|beep(?:s|ed|ing)?|rings|ringing|ringtone|kettle|"
                    r"whistles?)\b", re.I)
_CROTCH = r"crotch|groin|vagina|pussy|vulva|labia|genitals|privates|penis"
_BETWEEN = re.compile(r"\bbetween\s+[\w'’-]+\s+(?:legs|thighs)\b", re.I)
_BODY = (("mouth", r"mouth|lips|jaw"), ("eyes", r"eyes"), ("wrists", r"wrists?|hands|arms"), ("ankles", r"ankles?|feet|foot|legs|knees"),
         ("neck", r"neck|throat"), ("body", rf"hips|waist|{_CROTCH}|chest|torso|body|thighs|stomach|breasts?|shoulders"))
_KIND = (("mouth", r"gag|muzzle|tape"), ("eyes", r"blindfold"), ("wrists", r"(?:hand)?cuffs|manacles|zip\s*ties?|rope|chains?"),
         ("ankles", r"shackles|leg\s+irons"), ("neck", r"collar|leash"))
_FREE = {"mouth": "mouth is empty and uncovered", "eyes": "eyes are uncovered", "wrists": "hands are free", "neck": "neck is bare"}
_LOW_COVER = r"pants|trousers|jeans|leggings|shorts|skirts?|miniskirts?|joggers|sweatpants|slacks|bottoms|capris|culottes"
_SHOES = r"shoes|boots|heels|sneakers|sandals|slippers|trainers|flats|pumps|loafers"
_UP_COVER = r"shirts?|t-shirts?|blouses?|tops?|sweaters?|jumpers?|hoodies?|sweatshirts?|tees?|cardigans?|jackets?|blazers?|" \
            r"(?:rain|over|pea|top|house|fur)?coats?|vests?|camisoles?|corsets?|bustiers?|halters?|ponchos?|shawls?|parkas?|anoraks?|" \
            r"windbreakers?"
_ALL_COVER = r"dress(?:es)?|nightdress(?:es)?|gowns?|nightgowns?|nighties|nightshirts?|robes?|bathrobes?|kimonos?|jumpsuits?|" \
             r"overalls|onesies?|pajamas|pyjamas|uniforms?|suits?|costumes?|sundress(?:es)?|minidress(?:es)?|babydolls?|negligees?|" \
             r"chemises?|teddies|teddy|rompers?|playsuits?|sarongs?"
_UNDER = r"panties|knickers|thongs?|g-strings?|underwear|briefs|boxers|bras?|lingerie"
_LOWER = _LOW_COVER + r"|panties|knickers|thongs?|g-strings?|underwear|briefs|boxers|tights|stockings|pantyhose"
_UPPER = _UP_COVER + r"|bras?"
_WHOLE = _ALL_COVER + r"|bodysuits?|leotards?|swimsuits?|bikinis?|catsuits?|lingerie"
_GENERIC = {"bottoms": _LOW_COVER, "bottom": _LOW_COVER, "top": _UP_COVER, "tops": _UP_COVER, "underwear": _UNDER,
            "shoes": _SHOES,
            "lingerie": _UNDER}
_ALL = re.compile(r"^(?:all\s+)?(?:(?:her|his|their)\s+)?(?:clothes|clothing|everything|outfit|all|naked|nude)$", re.I)
_PAIRS = r"shorts|pants|jeans|leggings|tights|trousers|boxers|briefs|panties|knickers|pajamas|pyjamas|overalls|bottoms|" \
         r"stockings|slacks|joggers|sweatpants|pantyhose|glasses"
_SHEER = re.compile(r"\b(?:sheer|see-?through|mesh|fishnet|open|unbuttoned|unzipped)\b", re.I)
_OPENS = re.compile(r"\b(?:jackets?|blazers?|cardigans?|coats?|robes?|bathrobes?|kimonos?)\b", re.I)
_SKIRT = re.compile(r"\b(?:mini)?skirts?\b", re.I)
_OUTER = re.compile(r"\b(?:jackets?|blazers?|coats?|cardigans?|vests?|robes?|bathrobes?|kimonos?|parkas?|anoraks?|windbreakers?|"
                    r"ponchos?|shawls?)\b", re.I)
_LOOSE_NAME = re.compile(r"(?:(?:her|his|their|the|a|an)\s+)?(?:(?!tank|crop|tube|halter|night|sun|mini|maxi|sweat|track|yoga|cargo|"
                         r"dress|pajama|pyjama|sleep|lounge|bikini|swim|board|gym|bike|hot)[\w-]+\s+)?(?:pants|trousers|slacks|"
                         r"bottoms?|tops?|shirts?|underwear|panties|knickers|briefs|undies|dress)", re.I)
_UPWARD = re.compile(r"\b(?:breasts?|chest|shoulders?|neck|head|armpits?)\b", re.I)
_JOIN = re.compile(r"(,\s*and\s+|,\s*|;\s*|\s+and\s+|(?<=[.!?])\s+|(?<!dressed)(?<!clad)\s+(?=(?:in|wearing|with|over|under|underneath|"
                   r"beneath|dressed\s+in|clad\s+in)\s))", re.I)
_LEAD = re.compile(r"^(?:(?:she|he|they)(?:\s+is|\s+are|['’]s)?\s+)?(?:in|wearing|wears|with|has\s+on|dressed\s+in|clad\s+in)\s+",
                   re.I)
_REL = re.compile(r"^(?:over|under|underneath|beneath|with)\s+", re.I)
_WHERE = re.compile(r"\s+(?:(?:pulled|pushed|lifted|hiked|yanked|tugged|rolled|slid)\s+)?"
                    r"((?:up|down|open|aside|to|above|over|below|around|past|off)\b.*)$", re.I)
_ARMS = ((r"behind\s+(?:her|his|their|the)\s+back|behind\s+(?:her|him|them)\b", "behind {p} back"),
         (r"in\s+front\b(?!\s+of\s+(?:the|a|an|this|that|(?-i:[A-Z])))", "in front of {o}"),
         (r"(?:above|over)\s+(?:her|his|their|the)\s+head", "above {p} head"), (r"at\s+(?:her|his|their|the)\s+waist", "at {p} waist"),
         (r"(?:at|by|to)\s+(?:her|his|their)\s+sides", "at {p} sides"),
         (r"spread[\s-]?eagled?|spread\s+(?:wide|out|apart)|(?:wide\s+)?apart", "spread apart"),
         *((rf"between\s+(?:her|his|their|the)\s+{w}", f"between {{p}} {w}") for w in ("legs", "thighs", "knees")))
_FIXTURES = r"bed\s?posts?|head\s?boards?|foot\s?boards?|bed\s?frames?|bed\s?rails?|door\s?(?:handles?|knobs?)|steering\s+wheels?|" \
            r"tables?|beds?|benches?|cots?|gurneys?|stretchers?|altars?|slabs?|mattress(?:es)?|floors?|ground|desks?|" \
            r"workbench(?:es)?|counters?|racks?|frames?|ceilings?|beams?|rafters?|hooks?|rings?|walls?|posts?|poles?|pillars?|" \
            r"columns?|pipes?|radiators?|bars?|railings?|rails?|trees?|doors?|doorknobs?|bolts?|sinks?|toilets?|fences?|gates?|" \
            r"banisters?|cages?|stocks|(?:arm)?chairs?|stools?|armrests?"
_SIDES = r"corners?|legs?|sides?|ends?|edges?|arms?|backs?|foot|feet|heads?|tops?"
_BEDPART = re.compile(r"\b(?:bed\s?posts?|head\s?boards?|foot\s?boards?|bed\s?frames?|bed\s?rails?)\b", re.I)
_SEAT = re.compile(r"\b(?:(?:arm)?chairs?|stools?)$", re.I)
_ARMREST = re.compile(r"\barm(?:rest)?s?\b", re.I)
_LIE_ON = re.compile(r"\b(?:tables?|beds?|benches?|cots?|gurneys?|stretchers?|altars?|slabs?|mattress(?:es)?|racks?)\b", re.I)
_FLOOR = re.compile(r"\b(?:floors?|ground|desks?|workbench(?:es)?|counters?)\b", re.I)
_OVERHEAD = re.compile(r"\b(?:ceilings?|beams?|rafters?)\b", re.I)
_ANCHOR = re.compile(rf"\b(?P<prep>to|onto|on|across|from)\s+(?P<a>(?:the|a|an|her|his|their)\s+(?:[\w-]+\s+){{0,2}}?(?:{_FIXTURES})\b"
                     rf"(?:\s+(?:{_SIDES})\b)?(?:\s+(?:of|on|in|at)\s+(?:the|a|an)\s+(?:[\w-]+\s+)?(?:{_FIXTURES})\b)?|"
                     rf"(?:the|its|each|both|all\s+four|four|two)\s+(?:[\w-]+\s+)?(?:{_SIDES})\s+of\s+(?:the|a|an)\s+(?:[\w-]+\s+)?"
                     rf"(?:{_FIXTURES})\b|(?:its|each|both|all\s+four|four|two)\s+(?:[\w-]+\s+)?(?:{_SIDES})\b|it\b|them\b)", re.I)
_TIED_TO = re.compile(r",\s+(?:locked|chained|tied|taped|strapped|fastened)\s+(?P<lie>face\s+down\s+|down\s+)?to\s+(?P<to>.+?)"
                      r"(?P<link>\s+by\s+a\s+separate\s+short\s+(?:chain|rope))?\s*$", re.I)
_WRISTS = r"(?:her|his|their) handcuffs|the [\w ]+? around (?:her|his|their) wrists"
_ONTO = re.compile(r"\b(?:to|through|onto|from)\s+(?:the\s+(?:short\s+)?(?:chain|links?)\s+(?:between|of|on|joining|linking)\s+)?"
                   r"(?:her|his|their|the|[\w-]+['’]s)\s+(?:own\s+)?(?:steel\s+)?(?:(?:hand)?cuffs|(?:cuffed|bound|tied|chained)\s+"
                   r"(?:wrists|hands))\b(?:\s+(?:behind\s+(?:her|his|their)\s+back|in\s+front\s+of\s+(?:her|him|them)))?", re.I)
_RING = re.compile(r"\b(?:to|onto|under|through|from)\s+(?:the\s+(?:back|front|middle)\s+of\s+)?"
                   r"(?:(?:the|a|an|her|his|their|[\w-]+['’]s)\s+(?:[\w-]+\s+)?"
                   r"(?:waist|belly|crotch)\s+(?:chains?|ropes?)|(?:the|her|his|their|[\w-]+['’]s)\s+(?:[\w-]+\s+)?(?:chains?|ropes?)\s+"
                   r"(?:around|on|at|between)\s+(?:the\s+small\s+of\s+(?:her|his|their)\s+back|(?:her|his|their|[\w-]+['’]s)\s+"
                   r"(?:waist|hips|belly|legs)))\b", re.I)
_REGIONS = (("wrists", r"wrists?|hands?|arms|(?:hand)?cuffs|manacles"),
            ("ankles", r"ankles?|feet|legs|knees|thighs|shackles|(?:leg\s+)?irons|fetters|(?:ankle|leg)\s+(?:cuffs|chains?|shackles)"),
            ("neck", r"necks?|collars?|throats?"))
_OWNED = r"(?:her|his|their|[\w-]+['’]s)\s+(?:(?:own|left|right|two|bound|cuffed|tied|chained|shackled|steel|metal)\s+){0,2}"
_ZONES = "|".join(rx for _, rx in _REGIONS)
_LINK = re.compile(rf"\bhog[\s-]?ti(?:e|es|ed|ing)\b|\b{_OWNED}(?P<a>{_ZONES})\b(?:(?!,|\b(?:then|and|before|after|while|as|so|but)\b)"
                   rf"[^.;!?]){{0,60}}?\b(?:to|onto|through|from)\s+{_OWNED}(?!(?:feet|knees)\b)(?P<b>{_ZONES})\b|"
                   rf"\bbetween\s+{_OWNED}(?P<c>{_ZONES})\s+and\s+(?:{_OWNED})?(?P<d>{_ZONES})\b", re.I)
_HARNESS = {"chains": ("locked", "unlocks the padlock", "hooks"), "rope": ("tied", "unties the knot", "ties")}
_GARB = {"remove": "remove", "tear": "remove", "lift": "lift", "wear": "wear"}
_TIES = {"handcuffs": ("locked", "locks"), "chains": ("chained", "chains"), "shackles": ("chained", "chains"), "rope": ("tied", "ties"),
         "a spreader bar": ("locked", "locks"), "a chastity belt": ("locked", "locks"),
         "zip ties": ("tied", "ties"), "tape": ("taped", "tapes"), "duct tape": ("taped", "tapes"), "straps": ("strapped", "straps")}
_BANDS = ("waist", "hips", "chest", "breasts", "torso")
_LIMBS = ("wrists", "ankles", "arms", "legs", "knees", "thighs")
_BIND = {"handcuffs": (5, 4, "closes one cuff around {x} left {one} with a click, then the other around {p} right {one}",
                       "unlocks the handcuffs with a small key and takes them {off}", ""),
         "zip ties": (4, 3, "loops a zip tie around {x} {s} and pulls it tight", "cuts the zip tie {off}", "zips it shut{t}"),
         "rope": (8, 6, "wraps the rope around {x} {s} several times, cinches it between them and ties it off in a firm knot",
                  "unties the knot and unwinds the rope {fr}", "ties it off{t} in a firm knot"),
         "chains": (6, 4, "wraps the chain around {x} {s} and locks it with a padlock", "unlocks the chain and takes it {off}",
                    "locks it{t} with a padlock"),
         "shackles": (6, 5, "locks one shackle around {x} left {one}, then the other around {p} right {one}",
                      "unlocks the shackles and takes them {off}", "locks it{t}"),
         "straps": (6, 4, "buckles a strap around each of {x} {s} and pulls it tight", "unbuckles the straps {fr}", "buckles it{t}"),
         "a spreader bar": (6, 5, "spreads {x} feet apart and locks one end of the spreader bar around each ankle",
                            "unlocks the spreader bar from {x} {s} and takes it away", ""),
         "a collar": (4, 3, "buckles the collar around {x} neck", "unbuckles the collar and takes it off {x} neck", ""),
         "a chastity belt": (6, 4, "fits the chastity belt around {x} waist, brings its shield down between {p} legs over {p} crotch and "
                                   "locks it shut", "unlocks the chastity belt and takes it off {x} waist", ""),
         "a ball gag": (5, 4, "pushes the ball into {x} mouth behind {p} teeth, pulls the strap around behind {p} head and buckles it",
                        "unbuckles the gag and takes it out of {x} mouth", ""),
         "a gag": (5, 4, "fits the gag into {x} mouth and fastens it behind {p} head", "unfastens the gag and takes it out of {x} mouth", ""),
         "a blindfold": (4, 3, "lays the blindfold over {x} eyes and ties it behind {p} head", "unties the blindfold and lifts it off {x} eyes",
                         ""),
         "duct tape": (7, 5, "", "peels the tape off {x} {s}", "")}
_BOUND = (5, 4, "fastens the {g} around {x} {s}", "takes the {g} {off}", "fastens it{t}")
_FEET = {"handcuffs": "closes one cuff around each {one} with a click, the short chain between them",
         "shackles": "locks one shackle around each {one}, the heavy chain hanging between them",
         "rope": "wraps the rope several times around both of them together, cinches it between them and ties it off in a firm knot",
         "zip ties": "loops a zip tie around both of them together and pulls it tight",
         "straps": "buckles a strap around both of them together and pulls it tight",
         "chains": "wraps the chain around both of them together and locks it with a padlock",
         "tape": "winds the tape around and around both of them together"}
_UNDRESS = ((r"gloves|hats?|caps?|scarf|scarves|glasses", 2, "takes off {w} {g}", ""),
            (r"jackets?|blazers?|(?:rain|over|pea|top|house|fur)?coats?|cardigans?|vests?|robes?|bathrobes?|kimonos?|parkas?|"
             r"anoraks?|windbreakers?|ponchos?|shawls?", 4, "slides {w} {g} off {p} shoulders and down {p} arms", ""),
            (_SHOES, 3, "takes off {w} {g} one foot at a time",
             "slips {w} {g} off, {p} feet staying together"),
            (r"corsets?|bustiers?", 7, "unlaces {w} {g} and takes {it} off", ""),
            (r"blouses?|button[\w-]*\s+[\w-]+", 6, "unbuttons {w} {g} from the top down and slides {it} off {p} shoulders and down {p} arms",
             ""),
            (_WHOLE, 6, "slides {w} {g} off {p} shoulders and all the way down {p} body, and off over {p} feet",
             "pulls {w} {g} up over {p} head and off {p} arms"),
            (_UP_COVER, 4, "pulls {w} {g} up over {p} head and off {p} arms", ""),
            (r"(?:mini)?skirts?|sarongs?", 4, "opens {w} {g} at the waist and slides {it} down {p} legs and off over {p} feet", ""),
            (_LOW_COVER, 6, "opens {w} {g} at the waist and pulls {it} down {p} legs and off over {p} feet", ""),
            (r"socks|stockings|tights|pantyhose", 5, "rolls {w} {g} down {p} legs and off {p} feet", "pulls {w} {g} off {p} feet"),
            (r"bras?|bralettes?", 4, "unhooks {w} {g} and slides the straps off {p} shoulders and down {p} arms", ""),
            (r"panties|knickers|thongs?|g-strings?|underwear|briefs|boxers", 3, "slides {w} {g} down {p} legs and off over {p} feet", ""))
_OBJ = {"his": "him", "their": "them"}
_HANDY = re.compile(r"\b(?:wipes?|reach(?:es)?|grabs?|clutch(?:es)?|hugs?|waves?|claps?|types?|writes?|dials?|catch(?:es)?|"
                    r"picks?\s+up|holds?\s+(?:up|out)|covers?\s+(?:her|his|their)\s+(?:face|eyes|mouth|ears|breasts|chest)|"
                    r"rubs?\s+(?:her|his|their)\s+(?:eyes|face)|(?:brushes|tucks|fixes|smooths)\s+(?:her|his|their)\s+hair)\b", re.I)
_GARMENT = re.compile(rf"\b(?:{_LOWER}|{_UPPER}|{_WHOLE}|shoes|socks|heels|boots|sneakers|sandals|slippers|gloves|hats?|caps?|"
                      rf"scarf|scarves|glasses)\b", re.I)
_READ = {"have": "on", "tear": "torn off"}
_COUNT = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
_SPOTS = rf"mouths?|lips|teeth|eyes|wrists?|hands?|arms?|ankles?|feet|(?<!\ba )(?<!\bone )(?<!\bon )foot(?!\s+of\b)|" \
         rf"leg(?=\s+(?:irons|chains?|cuffs|shackles)\b)|legs|knees|thighs|hips|waists?|(?<!\bon )(?<!\bon her )(?<!\bon his )belly|" \
         rf"necks?|throats?|chests?|breasts|torsos?|{_CROTCH}"
_GEAR = (("handcuffs", r"handcuff(?:s|ed|ing)?|manacle[sd]?|(?:ankle|leg)\s+cuffs", r"cuff(?:s|ed|ing)?", "wrists"),
         ("zip ties", r"zip[\s-]?ti(?:es|ed|e)|cable\s+ties", "", "wrists"),
         ("a ball gag", r"ball[\s-]?gags?", "", "mouth"),
         ("a gag", "", r"gag(?:s|ged|ging)?", "mouth"),
         ("a blindfold", r"blindfold(?:s|ed|ing)?", "", "eyes"),
         ("duct tape", r"tap(?:e|es|ed|ing)", "", ""),
         ("rope", r"ropes?|roped|cords?", r"ti(?:e|es|ed)|tying|bind(?:s|ing)?|bound", "wrists"),
         ("a collar", r"collar(?:s|ed)?", "", "neck"),
         ("chains", r"chains?", r"chained", "wrists"),
         ("a spreader bar", r"spreader(?:\s+bars?)?", "", "ankles"),
         ("a chastity belt", r"chastity[\s-]+(?:belts?|devices?)", "", "crotch"),
         ("shackles", r"shackle[sd]?|leg\s+irons|fetter(?:s|ed)?|(?:ankle|leg)\s+(?:chains?|irons|shackles)|hobble\s+chains?|"
                      r"(?<=\bin )irons", "", "ankles"),
         ("straps", rf"strapp(?:ed|ing)|straps?(?=\s+(?:(?:her|him|them|(?-i:[A-Z])[\w'’-]*)\s+(?:down|in|into|onto|to)\b|"
                    rf"(?:(?:around|across)\s+)?(?:her|his|their|(?-i:[A-Z])[\w-]*['’]s)\s+(?:{_SPOTS})\b))|"
                    r"(?:leather|restraint|velcro|nylon|canvas|buckled)\s+straps?", "", "wrists"))
_KIT = "|".join(rx for _, a, b, _ in _GEAR for rx in (a, b) if rx)
_REACH = r"(?:\s+(?!(?:and|then|while|as|before|after|so|but|when|until)\b)(?![\w-]+ing\b)" \
         r"(?!(?:ropes?|cords?|chains?|straps?|tape|belts?)\b)[\w'’,-]+){0,4}?\s+"
_AT = "|".join(rx for rx, _ in _ARMS)
_POSE = re.compile(rf"\b(?:wrists?|hands?|arms?|together|(?P<kit>{_KIT}))\b{_REACH}(?(kit)(?!between\b))(?P<at>{_AT})", re.I)
_LAP = re.compile(rf"\b(?:wrists?|hands?|arms?|together)\b{_REACH}(?P<at>{_AT})", re.I)
_SPOT_AS = {"lips": "mouth", "teeth": "mouth", "wrist": "wrists", "hand": "wrists", "hands": "wrists", "arm": "arms", "ankle": "ankles",
            "feet": "ankles", "foot": "ankles", "leg": "ankles", "belly": "waist", "throat": "neck", "throats": "neck", "mouths": "mouth",
            "waists": "waist", "necks": "neck", "chests": "chest", "torsos": "torso", **{w: "crotch" for w in _CROTCH.split("|")}}
_LEADS = r"(?:then|now|next|slowly|quickly|finally|gently|roughly|suddenly|carefully|firmly|tightly|again|still)"
_APPLY = (r"puts?|snaps?|snapped|locks?|locked|locking|clicks?|clicked|slaps?|slapped|wraps?|wrapped|wrapping|winds?|wound|"
          r"winding|presses?|pressed|pressing|sticks?|stuck|places?|placed|plasters?|plastered|seals?|sealed|sealing|stuffs?|"
          r"stuffed|shoves?|shoved|fastens?|fastened|secures?|secured|loops?|looped|clamps?|clamped|buckles?|buckled|straps?|"
          r"strapped|strapping|forces?|forced|applies|applied|tightens?|tightened|adds?|added|smooths?|smoothed|lays?|laid|binds?|"
          r"binding|ties|tied|tying|tapes|taped|taping|cuffs|cuffed|cuffing|handcuffs|handcuffed|manacles|manacled|gags|gagged|"
          r"gagging|blindfolds|blindfolded|chains|chained|chaining|shackles|shackled|fetters|fettered|collars|collared|"
          r"zip[\s-]?ties|zip[\s-]?tied|ropes|roped|cinches|cinched|lashes|lashed|tethers|tethered|hogties|hogtied|slips?|slipped|"
          r"slides?|slid|pushes?|pushed|jams?|jammed|crams?|crammed|fits?|fitted|knots?|knotted|padlocks?|padlocked|clips?|"
          r"clipped|hooks?|hooked|attaches|attached|anchors?|anchored|bolts?|bolted|cuff|gag|blindfold|handcuff|tie|chain|shackle|"
          r"press|seal|strap|wrap|lock|fasten|secure|buckle|stuff|covers?|covered|covering|closes|closed|closing|snapping|"
          r"clicking|slapping|stuffing|shoving|pushing|jamming|cramming|fastening|securing|slipping|sliding|placing|forcing|"
          r"looping|clamping|buckling|cinching|lashing|knotting|putting|tape(?=\s+(?:her|his|their|them|him|(?-i:[A-Z])))")
_DOES = re.compile(rf"(?<!\bthe )(?<!\bher )(?<!\bhis )(?<!\btheir )(?<!\bits )(?<!\ba )(?<!\bof )(?<!\bmore )(?<!\bsome )(?<!['’]s )"
                   rf"(?<!\bin )(?<!\bwith )(?<!\bcrotch )(?<!\bwaist )(?<!\bcenter )(?<!\bleg )(?<!\bankle )(?<!\bhobble )(?<!\bbelly )"
                   rf"(?<!\bzip )(?<!\bshort )(?<!\blong )(?<!\bheavy )(?<!\bsteel )(?<!\biron )(?<!\bmetal )(?<!\bthin )(?<!\bthick )"
                   rf"(?<!\bchain )(?<!\bchains )(?<!\bcuffs )(?<!\brope )(?<!\btape )(?<!\bstraps )(?<!\bshackles )(?<!\birons )"
                   rf"\b(?:{_APPLY})\b", re.I)
_VERBS = _APPLY + r"|uncuffs?|unties?|ungags?|frees?|releases?|strips?|stripped|undress(?:es|ed)?|dress(?:es|ed)?"
_OBJECT = r"(?=\s*(?:[,.;!?]|$)|\s+(?:to|on|onto|in|into|down|up|with|by|against|behind|and|tight|tightly|securely|firmly|again|too|" \
          r"so|then|at|over|around|across|from)\b)"
_ROUTES = re.compile(r"\b(?:pass(?:es|ed)?|thread(?:s|ed)?|feeds?|fed|stretch(?:es|ed)?|runs?|ran|pulls?|pulled|draws?|drew|layers?|"
                     r"layered|criss-?cross(?:es|ed)?|works?|worked|guides?|guided|brings?|brought|tugs?|tugged)\b(?!\s+(?:(?:a|an|her|"
                     r"his|their|one|the|both)\s+)?(?:fingers?|hands?|thumbs?|palms?|nails?|fingertips?|tongue|lips|knuckles?)\b)", re.I)
_PLACE = (r"roll|reel|spool|dispenser|table|nightstand|drawer|bag|belt|shelf|desk|counter|box|kit|pocket|floor|dresser|tray|cabinet|"
          r"cupboard|case|holster|purse|backpack|toolbox|ground|hook|wall|couch|sofa|pillow|mattress|chair|bed(?!\s*(?:frame|post|"
          r"rail|head|leg))")
_SOURCES = re.compile(rf"\b(?:off|from|out\s+of|away\s+from)\s+(?:(?:the|a|an|its|her|his|their|[\w-]+['’]s)\s+)?(?:[\w-]+\s+)?"
                      rf"(?:{_PLACE})s?\b", re.I)
_PLACED = re.compile(rf"\b(?:on|onto|in|into|inside|beside|next\s+to|under)\s+(?:the|a|an|her|his|their|its)\s+(?:[\w-]+\s+)?"
                     rf"(?:{_PLACE})s?\b", re.I)
_TAKE = (r"tak(?:e|es|ing)|took|taken|pull(?:s|ed|ing)?|rip(?:s|ped|ping)?|tear(?:s|ing)?|tore|torn|yank(?:s|ed|ing)?|"
         r"peel(?:s|ed|ing)?|slip(?:s|ped|ping)?|slid(?:e|es|ing)?|cut(?:s|ting)?|snip(?:s|ped|ping)?|work(?:s|ed|ing)?|"
         r"get(?:s|ting)?|got|tug(?:s|ged|ging)?|eas(?:e|es|ed|ing)|unwind(?:s|ing)?|unwound|lift(?:s|ed|ing)?|strip(?:s|ped|ping)?|"
         r"wrench(?:es|ed|ing)?|pr(?:y|ies|ied|ying)|shak(?:e|es|en|ing)|shook|snap(?:s|ped|ping)?|wriggl(?:e|es|ed|ing)|"
         r"twist(?:s|ed|ing)?")
_UNDO = re.compile(r"\bun(?:cuff|lock|ti(?:e|es|ed)\b|tying|bind|bound\b|gag|blindfold|wrap|tap(?:e|es|ed|ing)\b|chain|shackl|buckl|"
                   r"strap|hook|clip|clasp|knot|fasten|wind|wound|do(?:es|ing|ne)?\b|did\b)", re.I)
_LOOSE = re.compile(rf"\b(?:{_TAKE})\b(?P<mid>(?:\s+[\w'’-]+){{0,6}}?)\s+(?:(?:off|away|free|loose|open|undone|apart)\b|"
                    rf"out\b(?=\s*(?:[,.;!?]|$|of\s+(?:her|his|their|(?-i:[A-Z]))))|from\b(?=\s+(?:between\s+|around\s+)?(?:her|his|"
                    rf"their|(?-i:[A-Z]))))|\bremov(?:e|es|ed|ing)\b|\b(?:frees?|freed|freeing|releases?|released|"
                    rf"releasing)\b(?!\s+(?:her|his|their)\s+(?:grip|hold|own)\b)(?!\s+(?:one|a|the|her|his)\s+hand\b)|"
                    rf"\b(?:cut(?:s|ting)?|snip(?:s|ped|ping)?|slic(?:e|es|ed|ing)|saw(?:s|ed|ing)?)\s+(?:through|apart)\b|"
                    rf"\b(?:cut(?:s|ting)?|snip(?:s|ped|ping)?|slic(?:e|es|ed|ing))\s+(?:the|her|his|their|[\w-]+['’]s)\s+(?:\w+\s+)?"
                    rf"(?:zip[\s-]?ties?|ropes?|cords?|bonds|restraints)\b", re.I)
_FALLS = re.compile(rf"\b(?:{_KIT}|it|they)\s+(?:\w+\s+)?(?:comes?|came|falls?|fell|drops?|dropped|snaps?|snapped|springs?|sprang|pops?|"
                    rf"popped|clicks?|clicked|swings?|swung|slips?|slipped|slides?|slid|peels?|peeled|gives?|gave)\s+(?:\w+\s+)?"
                    rf"(?:off|away|open|loose|free|undone|apart|way)\b", re.I)
_STATE = r"handcuffed|cuffed|manacled|gagged|blindfolded|duct[\s-]?taped|taped|tied|bound|chained|shackled|fettered|collared|" \
         r"zip[\s-]?tied|stuffed|strapped|hog[\s-]?tied"
_WORN = r"\b(?:in|wears?|wearing|wore)\s+(?:(?:a\s+(?:pair|set)\s+of|a|an|her|his|their|the|steel|iron|heavy|leather|metal|rusty)\s+)*" \
        r"(?:(?:hand|ankle|leg)\s*)?(?:cuffs|chains|shackles|irons|restraints|fetters|manacles)\b|" \
        r"\b(?:in|wears?|wearing|wore)\s+(?:a|her|his|their|the)\s+(?:(?:steel|metal|leather|locked|locking)\s+)?" \
        r"(?:crotch\s+(?:chain|rope)|spreader\s+bar|chastity[\s-]+belt)\b"
_MEEK = re.compile(r"\b(?:squirms?|struggles?|whimpers?|moans?|gasps?|writhes?|wriggles?|pleads?|begs?|sobs?|shivers?|trembles?|"
                   r"flinches?|freezes?|winces?|groans?|mumbles?|screams?|cries?|whines?|thrashes?|kicks?|jerks?|tenses?|shakes?|"
                   r"squeals?|stares?|blinks?|watches?|waits?|sits?|lies|kneels?|falls?|fell|stumbles?|stumbled|collapses?|"
                   r"collapsed|topples?|toppled|tumbles?|tumbled|lands?|landed|sprawls?|sprawled)\b", re.I)
_TRYING = re.compile(r"\b(?:tr(?:y|ies|ied|ying)|struggl\w*|attempt\w*|fail\w*|can(?:no|’|')?t|cannot|unable|wants?|wanting|strain\w*|"
                     r"threat\w*|(?:going|about|plans?|promis\w*)\s+to|"
                     r"tests?|tested|testing|fights?|fighting)\b|\b(?:pull|tug|yank|claw|pick|paw|jerk|scratch)\w*\s+at\b", re.I)
_PREP = re.compile(r"\b(?:strip|piece|length|roll|bit|stretch|pair|coil)s?\s+of\b|\b(?:a|an|another|the|one|two|three|second|third|"
                   r"fresh|new|long|short|wide|thin|thick|single|final|last|first|next|more)\s+(?:[\w-]+\s+)?(?:strip|piece|length|"
                   r"bit|coil)s?\b|\bstrip\s+after\s+strip\b", re.I)
_KEPT = r"tied|bound|taped|chained|strapped|cuffed|handcuffed|shackled|locked|fastened|secured|roped|crossed"
_TOOL = re.compile(rf"\b(?:with|using|uses?|used)\s+(?:the\s+(?:heel|palm|tips?|flat|edge|side|back)\s+of\s+)?(?:her|his|their|both|one|"
                   rf"[\w-]+['’]s)\s+(?:own\s+)?(?:hands?|fingers?|fingertips|thumbs?|palms?|fists?|teeth|knees?|lips|mouth)\b"
                   rf"(?!{_REACH}(?:{_AT}))(?!\s+(?:\w+ly\s+)?(?:\w+-)?(?:{_KEPT}|together)\b)|"
                   r"\b(?:in|into)\s+(?:her|his|their|[\w-]+['’]s)\s+(?:hands?|palms?|fists?)\b|\bbetween\s+(?:her|his|their)\s+"
                   r"fingers\b|\b(?:a|one|another|the)\s+(?:\w+\s+)?(?:hand|palm|finger|thumb|fingertip|knuckle)s?\b|"
                   r"\b(?:arms?|legs?|back|foot|feet|head|neck)\s+of\s+(?:the|a|an)\s+\w+|"
                   r"\bits\s+(?:[\w-]+\s+)?(?:arms?|legs?|back|feet|head)\b", re.I)
_MOVED = re.compile(r"\b(?:down|up|aside|away|off|loose)\b(?:\s+[\w'’-]+){0,3}?\s+(?:around|under|onto|to|on|over|against)\s+"
                    r"(?:her|his|their|[\w-]+['’]s)\s+(?:neck|chin|forehead|head|throat|hair|collarbones?)\b|\bsp(?:its?|at|itting)\s+"
                    r"(?:\w+\s+){0,2}?(?:out|onto|away|free|aside)\b|\b(?:hangs?|hanging|hung|dangles?|dangling|falls?|fell|drops?|dropped|"
                    r"slips?|slipped|sags?|sagged|slides?|slid)\s+(?:loose\s+|down\s+)?(?:around|at|from|to|onto|under)\s+"
                    r"(?:her|his|their|[\w-]+['’]s)\s+(?:neck|chin|throat|collarbones?)\b", re.I)
_REDO = re.compile(r"\bre-?(?:cuff|tie|tape|bind|chain|gag|blindfold|strap|wrap|lock|fasten|position)\w*|\binstead\b|"
                   r"\b(?:moves?|moved|switch(?:es|ed)?)\b", re.I)
_CLOTH = re.compile(rf"\b(?:scarf|scarves|bandanas?|cloths?|rags?|sashes?|neckties?|socks?|stockings?|handkerchiefs?|hank(?:y|ies)|"
                    rf"towels?|napkins?|{_UNDER})\b", re.I)
_THING = re.compile(rf"(?P<c>{_CLOTH.pattern}|{_GARMENT.pattern}|(?:(?<=\bhis )|(?<=\bher )|(?<=\bown )|(?<=\bthe )|(?<=\ba ))tie\b)|"
                    rf"\b(?:{_KIT})\b", re.I)
_RAG = re.compile(rf"\bwith\s+(?:[\w'’-]+\s+){{0,3}}?(?P<a>{_CLOTH.pattern})|(?P<b>{_CLOTH.pattern})(?=(?:\s+[\w'’-]+){{0,8}}?\s+"
                  rf"(?:in|into|inside|between|past|over|across|around)\s+(?:her|his|their|[\w-]+['’]s)\s+(?:\w+\s+)?"
                  rf"(?:mouth|lips|teeth|eyes|face|head)\b)", re.I)
_TIED = re.compile(r"\b(?:scarf|scarves|bandanas?|sashes?|neckties?|towels?|stockings?)\b", re.I)
_INMOUTH = re.compile(r"\b(?:in|into|inside|between|past)\s+(?:her|his|their|[\w-]+['’]s)\s+(?:\w+\s+)?(?:mouth|lips|teeth)\b|"
                      r"\b(?:stuff|cram|fill|pack|plug)\w*\s+(?:her|his|their|[\w-]+['’]s)\s+mouth\b", re.I)
_OVER = re.compile(r"\b(?:over|across|around)\s+(?:her|his|their|[\w-]+['’]s)\s+(?:\w+\s+)?(?:mouth|lips|eyes|face|head)\b", re.I)
_PUTAWAY = re.compile(r"\b(?:puts?|sets?|lays?|places?|tosses?|drops?|throws?|stows?|pockets?|hangs?)\s+(?:[\w'’-]+\s+){0,3}?"
                      r"(?:back|away|down|aside)\b(?!\s+(?:on|over|around|across|between)\b)", re.I)
_SHOWN = re.compile(rf"^(?:[^,]{{0,60}},\s+)?\W*(?:(?:{_LEADS}|\w+ly)\W+)*(?:(?-i:[A-Z])[\w'’-]*|she|he|they)\s+(?:\w+ly\s+)?"
                    rf"(?:(?:holds?|held|picks?|picked)\s+(?:[\w'’-]+\s+){{0,4}}?(?:up|out)\b|(?:shows?|showed|dangles?|dangled|"
                    rf"waves?|waved|produces?|produced|brandish(?:es|ed)|reach(?:es|ed)\s+for)\b)", re.I)
_SOON = re.compile(r"\b(?:moments?|seconds?|instant)\s+later\b", re.I)
_STILL = re.compile(rf"\b(?:is|are|was|were|been|sits?|sat|lies|lay|lays|stands?|stood|kneels?|knelt|remains?|stays?|stayed|ends?\s+up)"
                    rf"\b(?:\s+[\w'’,-]+){{0,4}}?\s+(?:{_STATE})\b|\b(?:with\s+)?(?:her|his|their)\s+(?:wrists|hands|ankles|feet|mouth|"
                    rf"eyes|arms|legs|knees)(?:\s+and\s+(?:(?:her|his|their)\s+)?(?:wrists|hands|ankles|feet|arms|legs|knees))?\s+"
                    rf"(?:\w+\s+)?(?:{_STATE})\b|\b(?:her|his|their|[\w-]+['’]s)\s+(?:\w+\s+)?(?:{_STATE})\s+"
                    rf"(?:{_SPOTS})\b|\bwith\s+(?:[\w'’-]+\s+){{1,3}}?(?:{_STATE})\b|(?:{_CLOTH.pattern}|\b(?:gag|tape))\s+(?:{_STATE})\s+"
                    rf"(?:in|into|inside|over|around|across|between)\b|\bwith\s+(?:a|an|the|her|his|their|some)\s+(?:[\w-]+\s+){{0,2}}?"
                    rf"(?:{_KIT})\s+(?:around|on|over|in|between)\s+(?:her|his|their)\s+(?:{_SPOTS})\b|"
                    rf"^\W*(?:still\s+)?(?:{_STATE})\b(?:\s+down\b)?"
                    rf"(?:\s+(?:at|by)\s+(?:the|her|his|their)\s+(?:{_SPOTS})\b)?(?:\s+(?:with|in)\s+(?:[\w-]+\s+){{0,2}}?(?:{_KIT})\b)?"
                    rf"(?:\s*(?:,|and|,\s*and)\s*"
                    rf"(?:{_STATE})\b)*\s*(?:,|[.!?]?\s*$)|"
                    rf"\b(?:leav(?:es|ing)|left|keep(?:s|ing)|kept)\b(?:\s+[\w'’-]+){{0,5}}?\s+(?:on|in\s+place|locked|tied|taped|cuffed|"
                    rf"buckled|where)\b|,\s+(?:\w+ly\s+)?(?:{_STATE})\s+(?:at|by)\s+(?:the|her|his|their)\s+(?:{_SPOTS})\b|{_WORN}", re.I)
_SAME = re.compile(r"\b(?:does|did|do|doing)\s+the\s+same\b", re.I)
_GVERBS = (("tear", r"cut(?:s|ting)?|snip(?:s|ped|ping)?|slic(?:e|es|ed|ing)|rip(?:s|ped|ping)?|tear(?:s|ing)?|tore|torn|"
                    r"shred(?:s|ded|ding)?"),
           ("open", r"unbutton(?:s|ed|ing)?|unzip(?:s|ped|ping)?|open(?:s|ed|ing)?|unfasten(?:s|ed|ing)?|undo(?:es|ing|ne)?|undid|"
                    r"unhook(?:s|ed|ing)?|unclasp(?:s|ed|ing)?|unti(?:e|es|ed)|untying|unsnap(?:s|ped|ping)?|unlac(?:e|es|ed|ing)|"
                    r"unwrap(?:s|ped|ping)?|part(?:s|ed|ing)?"),
           ("remove", r"remov(?:e|es|ed|ing)|strip(?:s|ped|ping)?|shed(?:s|ding)?|discard(?:s|ed|ing)?|unwind(?:s|ing)?|unwound"),
           ("raise", r"lift(?:s|ed|ing)?|rais(?:e|es|ed|ing)|hik(?:e|es|ed|ing)|bunch(?:es|ed|ing)?|ruck(?:s|ed|ing)?|"
                     r"hitch(?:es|ed|ing)?|rid(?:e|es|ing)|rode|ridden|scrunch(?:es|ed|ing)?"),
           ("fix", r"fix(?:es|ed|ing)?|straighten(?:s|ed|ing)?|adjust(?:s|ed|ing)?|smooth(?:s|ed|ing)?|rearrang(?:e|es|ed|ing)|"
                   r"tid(?:y|ies|ied|ying)|tuck(?:s|ed|ing)?|zip(?:s|ped|ping)?|button(?:s|ed|ing)?|fasten(?:s|ed|ing)?|"
                   r"clos(?:e|es|ed|ing)"),
           ("put", r"put(?:s|ting)?|dress(?:es|ed|ing)?|don(?:s|ned|ning)?|step(?:s|ped|ping)?|climb(?:s|ed|ing)?|"
                   r"throw(?:s|ing|n)?|threw|drap(?:e|es|ed|ing)|wrap(?:s|ped|ping)?"),
           ("move", r"pull(?:s|ed|ing)?|tug(?:s|ged|ging)?|yank(?:s|ed|ing)?|drag(?:s|ged|ging)?|push(?:es|ed|ing)?|"
                    r"slid(?:e|es|ing)?|slip(?:s|ped|ping)?|roll(?:s|ed|ing)?|draw(?:s|n|ing)?|drew|work(?:s|ed|ing)?|"
                    r"eas(?:e|es|ed|ing)|shov(?:e|es|ed|ing)|inch(?:es|ed|ing)?|wrestl(?:e|es|ed|ing)|wriggl(?:e|es|ed|ing)|"
                    r"shimm(?:y|ies|ied|ying)|kick(?:s|ed|ing)?|tak(?:e|es|en|ing)|took|get(?:s|ting)?|got|peel(?:s|ed|ing)?|"
                    r"shrug(?:s|ged|ging)?|help(?:s|ed|ing)?|mov(?:e|es|ed|ing)|shift(?:s|ed|ing)?|flip(?:s|ped|ping)?|"
                    r"hook(?:s|ed|ing)?|wrench(?:es|ed|ing)?|jerk(?:s|ed|ing)?|haul(?:s|ed|ing)?|hoist(?:s|ed|ing)?|comes?|came|"
                    r"falls?|fell|drops?|dropped|hangs?|hung|pool(?:s|ed|ing)?|sags?|sagged|snap(?:s|ped|ping)?|"
                    r"settl(?:e|es|ed|ing)|spring(?:s|ing)?|sprang|stuff(?:s|ed|ing)?|cram(?:s|med|ming)?|jam(?:s|med|ming)?"))
_HANDLE = "|".join(rx for _, rx in _GVERBS)
_GPART = r"off|up|down|on|into|in|out\s+of|aside|open|apart|away|back(?:\s+(?:on|into|in|up|down|over))?"
_GPOST = re.compile(r"^(?:\s+(?:back|all\s+the\s+way|halfway|partway|right|further|again|still|\w+ly))*\s+(off|up|down|aside|open|"
                    r"apart|on|away|over|past|to|above|below|around|into|in\s+place|free|loose|out|closed|shut)\b", re.I)
_HANDS_ON = re.compile(r",\s+with\s+(?:her|his|their|[\w-]+['’]s)\s+hands\s+still\s+[^,.;!?]*")
_GONE = re.compile(r"(?:\s+(?:all\s+the\s+way|completely|right))?(?:\s+(?:down|off|out|away)(?:\s+(?:over\s+)?(?:her|his|their|"
                   r"[\w-]+['’]s)\s+(?:legs|hips|body|thighs|waist|feet|head|arms|shoulders))?)*(?:\s+and\s+(?:all\s+the\s+way\s+)?off\b"
                   r"(?:\s+over\s+(?:her|his|their)\s+(?:feet|head))?)?", re.I)
_GBE = re.compile(rf"^\s+(?:(?:is|are|was|were|gets?|got|has\s+been|have\s+been|stays?|stayed|remains?|still|now|already|\w+ly)\s+)*"
                  rf"(?P<v>{_HANDLE})\b", re.I)
_TOWARD = re.compile(r"\b(?:(?:up|down)\s+)?(?:to|above|over|past|around|below)\s+(?:her|his|their)\s+(?:breasts|chest|waist|hips|"
                     r"knees|ankles|thighs|neck|shoulders|collarbones?|armpits|calves|feet)\b|\b(?:aside|open)\b", re.I)
_PRONOUN = re.compile(r"\b(?:her|him|his|them|their)\b", re.I)
_SHE_LEADS = re.compile(rf"^\W*(?:(?:\w+ly\s+)?[a-z]+ing\b(?P<front>[^,]{{0,60}}),\s+)?(?:{_LEADS}\W+)*(?P<pro>she|he|they)\b", re.I)
_OBJ_NEXT = re.compile(r"\s*(?:[^\w\s'’]|$|(?:a|an|the|this|that|these|those|some|any|one|to|toward|towards|into|onto|on|in|at|by|"
                       r"with|without|from|off|out|up|down|over|under|against|around|across|through|behind|beside|near|and|or|but|as|"
                       r"while|when|until|before|after|so|then|too|again|close|closer|tight|tighter|away|aside|forward|free|loose|"
                       r"still|here|there|now|something|nothing|everything|what|how|why|where|if|whether|it|him|her|them|me|us|you|"
                       r"alone|go|upside|flat|awake|steady|naked|nude)\b)", re.I)
_CAUSE = re.compile(r"\b(?:let|lets|letting|make|makes|made|making|help|helps|helped|helping|watch|watches|watched|watching|see|sees|"
                    r"saw|hear|hears|heard|feel|feels|felt|have|has|had)\s+$", re.I)
_HUSH = "Nobody speaks, and every mouth stays closed."
_WORDLESS = "Nobody says a word."
_SOUND = "All the sound comes from the scene itself: its natural ambience and the real sounds of what happens in it."
_PRONE = re.compile(r"\bface[\s-]?down\b|\bon(?:to)?\s+(?:her|his|their)\s+(?:stomach|front|belly)\b|\bprone\b", re.I)
_GAZE = re.compile(r"\b(?:(?:look|turn|talk|call|shout|yell|whisper|laugh|wink)(?:s|ed|ing)?|(?:star|glanc|smil|wav)(?:e|es|ed|ing)|"
                   r"grin(?:s|ned|ning)?|nod(?:s|ded|ding)?|speak(?:s|ing)?|spoke|say(?:s|ing)?|said)\s+(?:\w+ly\s+)?(?:back\s+|over\s+|"
                   r"up\s+|down\s+)?(?:at|to|toward|towards)\s+$|\b(?:watch(?:es|ed|ing)?|eye[sd]?|tells?|told|asks?|asked|thanks?|"
                   r"thanked)\s+$", re.I)
_CUT_AT = re.compile(r"[,;:—–]|\b(?:and|but|then|while|as|when|until|before|after|because|so|or|once|if|though|although|since|"
                     r"whereas|who|which)\b", re.I)
_AIMED = re.compile(rf"(?<![\w'’-])(?!(?:{_LEADS}|\w+ly|and|but|so|or|nor|yet|as|while|whilst|when|because|after|before|until|once|"
                    rf"if|though|although|since|where|whereas)\b)[a-z]+\s+$")
_ROUGH = re.compile(r"\b(?:grabs?|grabbed|seizes?|seized|pushes|pushed|shoves?|shoved|pulls?|pulled|drags?|dragged|yanks?|yanked|"
                    r"holds?|held|pins?|pinned|restrains?|restrained|ties|tied|binds?|bound|(?:hand)?cuffs?|(?:hand)?cuffed|tapes?|taped|"
                    r"gags?|gagged|slaps?|slapped|hits?|punch(?:es|ed)?|kicks?|kicked|chokes?|choked|squeezes?|squeezed|twists?|twisted|"
                    r"forces?|forced|presses|pressed|lifts?|lifted|throws?|threw|shakes?|shook|slams?|slammed|clamps?|clamped|grips?|"
                    r"gripped|chains?|chained)\b", re.I)
_PLEAD = re.compile(r"\b(?:let\s+me\s+go|let\s+go\s+of\s+me|get\s+(?:off|away\s+from)\s+me|leave\s+me\s+alone|"
                    r"don['’]?t\s+(?:hurt|touch)\s+me|(?:you['’]re|you\s+are)\s+hurting\s+me|untie\s+me|help\s+me(?!\s+[a-z])|"
                    r"please\s+(?:don['’]?t|stop|no))\b", re.I)
_REPLY = re.compile(r"\W*(?:(?:yes|yeah|yep|yup|no|nope|nah|maybe|perhaps|sure|fine|okay|ok)\b\W|(?:somewhere|nowhere|nothing|nobody|"
                    r"no\s+one|because)\b(?!\s+(?:is|are|was|were|will|would|can|could|has|have|had|knows?|cares?|comes?|gets?|goes)\b))",
                    re.I)
_DETS = {"her", "his", "their", "the", "a", "an", "its", "s", "off", "up", "down", "and", "of", "with", "to", "over", "on", "into", "out",
         "from", "back", "some", "more", "my", "your", "aside", "open", "it", "them", "him", "own", "both", "this", "that"}
_CUES = (rf"{_APPLY}|takes?|took|pulls?|pulled|rips?|ripped|tears?|tore|yanks?|yanked|slips?|slipped|lifts?|lifted|raises?|raised|"
         rf"hikes?|hiked|pushes?|pushed|tugs?|tugged|removes?|removed|peels?|peeled|strips?|stripped|undress(?:es)?|unbuttons?|unzips?|"
         rf"opens?|uncuffs?|unties?|frees?|releases?|cuts?|drags?|slides?|slid|rolls?|rolled|grabs?|grabbed|reaches?|holds?|kneels?|"
         rf"leans?|watches?|smiles?|steps?|walks?|turns?|sits?|stands?|gets?|got|helps?|helped|keeps?|kept|tries|tried|struggles?|"
         rf"struggled|squirms?|squirmed|yelling|screaming|begging|pleading|crying|sobbing")
_CLAUSE = re.compile(rf",?\s+(?:and\s+)?then\s+|;\s*|,\s+but\s+|(?<!\bbound)(?<!\btied)(?<!\bcuffed)(?<!\bgagged)(?<!\btaped)(?<!\bchained)"
                     rf"(?<!\bblindfolded)(?<!\bshackled),?\s+and\s+(?=(?:{_LEADS}\s+)?(?:\w+ly\s+)?(?:re-?)?(?:{_CUES})\b)|"
                     rf",\s+(?!(?:(?:hand)?cuffs|chains|shackles|gags|ropes|straps|blindfolds|collars|manacles)\s*(?:,|and\b|or\b|[.;!?]|$))"
                     rf"(?=(?:{_CUES}|leaving|keeping)\b)|"
                     rf",\s+and\s+(?=(?:(?-i:[A-Z])[\w'’-]*|she|he|they)\s+(?:\w+ly\s+)?(?:{_CUES}|wears?|wore)\b)|"
                     rf",\s+(?:and\s+)?(?=(?:her|his|their|(?-i:[A-Z])[\w'’-]*['’]s)\s+(?:[\w-]+\s+)?(?:{_SPOTS})\s+"
                     rf"(?:is|are|was|were|now|"
                     rf"still|already|stays?|remains?)\b)", re.I)
_DIRS = r"to|toward|towards|into|in|out|over|across|up|down|back|away|forward|forwards|closer|along|through|around|round|past|after|" \
        r"off|inside|outside|ahead|aside|home|onward|onwards"
_FLIGHT = r"(?:the\s+)?(?:[\w-]+\s+)?(?:stairs|steps|staircase|stairway|ladder)\b"
_STRIDE = r"walk(?:s|ed|ing)?|(?<!\bthe )(?<!\ba )(?<!\bher )(?<!\bhis )(?<!\btheir )(?<!\bone )(?<!\bfew )(?<!\btiny )(?<!\bsmall )" \
          r"(?<!\bshort )(?<!\bbottom )(?<!\bfirst )(?<!\blast )step(?:s|ped|ping)?|(?:takes?|took|taking)\s+(?:[\w,-]+\s+){0,3}?steps?|" \
          r"shuffl(?:e|es|ed|ing)|hobbl(?:e|es|ed|ing)|" \
          r"stagger(?:s|ed|ing)?|limp(?:s|ed|ing)|wander(?:s|ed|ing)?|pac(?:es|ed|ing)|march(?:es|ed|ing)?|hurr(?:y|ies|ied|ying)|" \
          r"rush(?:es|ed|ing)?|retreat(?:s|ed|ing)?|approach(?:es|ed|ing)?|cross(?:es|ed|ing)?(?!\s+(?:her|his|their)\b)|" \
          r"follow(?:s|ed|ing)?|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|leav(?:es|ing)|tiptoe(?:s|d|ing)?|trudg(?:e|es|ed|ing)|" \
          r"creep(?:s|ing)?|crept|sneak(?:s|ing)?|snuck|lurch(?:es|ed|ing)?|waddl(?:e|es|ed|ing)|led|descend(?:s|ed|ing)?|" \
          r"climb(?:s|ed|ing)?(?!\s+(?:onto|into|on|in|over|out|off|through|aboard|inside)\b)"
_HEADS = r"go(?:es|ing)?|went|mov(?:e|es|ed|ing)|com(?:e|es|ing)|came|head(?:s|ed|ing)|backs|backed|backing|run(?:s|ning)?|ran|" \
         r"stumbl(?:e|es|ed|ing)|mak(?:e|es|ing)\s+(?:her|his|their)\s+way|made\s+(?:her|his|their)\s+way"
_FOOT = (("hop", r"hop(?:s|ped|ping)?|jump(?:s|ed|ing)?(?!\s+over)"), ("kick", r"kick(?:s|ed|ing)?(?!\s+(?:off|away)\b)"),
         ("struggle", r"struggl(?:e|es|ed|ing)(?!\s+to\b)|squirm(?:s|ed|ing)?|writh(?:e|es|ed|ing)|thrash(?:es|ed|ing)?|"
                      r"wriggl(?:e|es|ed|ing)|flail(?:s|ed|ing)?|crawl(?:s|ed|ing)?|scoot(?:s|ed|ing)?|inch(?:es|ed|ing)?(?=\s+"
                      r"(?:back|away|forwards?|backwards?|along|across|toward|towards)\b)"),
         ("kneel", r"kneel(?:s|ing)?|knelt|(?:on|onto|to)\s+(?:her|his|their)\s+knees|(?:sits?|sitting|sat)\s+(?:back\s+)?on\s+"
                   r"(?:her|his|their)\s+heels"), ("sit", r"sit(?:s|ting)?|sat|seated"),
         ("lie", r"lies|lying|(?:lay|lays|laying|lie)\s+(?:down|back|flat|still|on|in|face)|curl(?:s|ed|ing)?\s+up|sleep(?:s|ing)|slept|"
                 r"(?:rolls?|rolled|rolling|turns?|turned)\s+(?:over\s+)?(?:onto|on)\s+(?:her|his|their)\s+(?:side|back|front|stomach|belly)|"
                 r"(?:falls?|fell|collaps(?:es|ed)|drop(?:s|ped)|sinks?|sank|slump(?:s|ed)|tumbl(?:es|ed)|pushed|thrown|shoved|tossed|laid)"
                 r"\s+(?:back\s+|down\s+|flat\s+|face[\s-]?(?:down|up|first)\s+|facedown\s+)?(?:onto|on|across|into)\s+"
                 r"(?:the|a|an|her|his|their)\s+(?:[\w-]+\s+)?"
                 r"(?:bed|floor|ground|mattress|cot|couch|sofa)\b|(?:falls?|fell)\s+flat"),
         ("stand", r"stand(?:s|ing)?|stood|(?:gets?|got|getting)\s+up|(?:to|on)\s+(?:her|his|their)\s+feet"),
         ("walk", rf"(?:{_STRIDE})\b|(?:{_HEADS})\s+(?:\w+ly\s+)?(?:{_DIRS})\b"))
_WIDE = re.compile(rf"\b(?:(?:spread(?:s|ing)?|open(?:s|ed|ing)?|part(?:s|ed|ing)?)\s+(?:her|his|their)\s+(?:legs|knees|thighs)|"
                   rf"(?:her|his|their)\s+legs\s+(?:spread|apart|wide)|cross(?:es|ed|ing)?\s+(?:her|his|their)\s+(?:legs|ankles)|"
                   rf"cross-legged|straddl(?:e|es|ed|ing)|wrap(?:s|ped|ping)?\s+(?:her|his|their)\s+legs|"
                   rf"(?:step|climb|leap|jump)(?:s|ed|ped|ing|ping|t)?\s+over|sprint(?:s|ed|ing)?|"
                   rf"(?:run(?:s|ning)?|ran|dash(?:es|ed|ing)?|bolt(?:s|ed|ing)?|jog(?:s|ged|ging)?|rac(?:es|ed|ing))\s+(?:\w+ly\s+)?"
                   rf"(?:{_DIRS}|for)\b)", re.I)
_LED = r"leads?|led|walk(?:s|ed)?|march(?:es|ed)?|frogmarch(?:es|ed)?|escort(?:s|ed)?|guided?|guides|steer(?:s|ed)?|push(?:es|ed)?|" \
       r"shove[sd]?|pull(?:s|ed)?|yank(?:s|ed)?|tug(?:s|ged)?|takes?|took|brings?|brought|haul(?:s|ed)?|help(?:s|ed)?|gets?|got|" \
       r"move[sd]?|sends?|sent|force[sd]?|nudge[sd]?|herd(?:s|ed)?|rolls?|rolled|turns?|turned|flips?|flipped|lays?|laid"
_MAKE = r"makes?|made|forces?|forced|orders?|ordered|tells?|told|lets?|helps?|helped|has|had|gets?|got|wants?|wanted|" \
        r"watch(?:es|ed)?|sees?|saw|hears?|heard|feels?|felt"
_STEPS = {"walk": "stand", "stairs": "stand", "hop": "stand", "stand": "stand", "kneel": "kneel", "sit": "sit", "lie": "lie"}
_STRIDES = {"stairs": "stairs", "walk": "walk", "wide": "wide", "hop": "move", "kick": "move", "struggle": "move", "legs": "move"}
_LIMB = r"legs?|knees?|feet|foot|ankles?|thighs?|calves|calf|shins?|toes|heels?"
_PAIR = {"legs": "leg", "knees": "knee", "feet": "foot", "ankles": "ankle", "thighs": "thigh", "calves": "calf", "shins": "shin"}
_GRIP = r"grab|grip|grasp|seiz|catch|caught|tak|took|hold|held|clutch|snatch"
_BINDS = re.compile(rf"\b(?:hand)?cuff|\bshackl|\b(?:{_KEPT})\b|\b(?:{_KIT})\b\s+(?:around|on|between|binding|holding)\s+"
                    rf"(?:her|his|their|[\w-]+['’]s)\s+(?:ankles|knees|legs|feet)\b", re.I)
_PARTICLE = re.compile(r"(?:\s+(?:and\s+)?(?:up|down|out|back|forth|away|high|higher|aside)"
                       r"(?!\s+(?:the|a|an|her|his|their|its|this|that)\b))*", re.I)
_SELF = r"bend|bent|stretch|straighten|flex|point|wiggl|kick|stamp|stomp|tap|swing|swung|curl|tuck|draw|drew|rub|squeez|clench|" \
        r"shak|shook|bounc|rock|shift|plant|brac|scrap|pedal|pump|scissor|paddl|flutter|tens|strain|thrash|flail|cross"
_GO = rf"(?:{_SELF}|ris|rose|lift|rais|com|came|go|went|jerk|twitch|trembl|buckl|fl[iy]|flew|shoot|shot|thump|pound|bang|scrabbl|" \
      r"slid|drag|drop|fall|fell|squirm|wriggl|writh|struggl|dangl|wav|press|push|pull|dig|dug|lower|bring|brought|roll|twist|turn|" \
      r"rotat|hug|fold|hook|jiggl|heav|hoist|haul|throw|threw|toss|yank|tug|wrench|mov)\w*"
_RISE = re.compile(rf"(?:gets?|got|getting|stands?|standing|stood|sits?|sitting|sat|kneels?|kneeling|knelt|lies|lying|lay|lays|laying|"
                   rf"laid)\s+(?:up|down)\b|(?:to|onto)\s+(?:her|his|their)\s+(?:feet|knees|side|back|front|stomach|belly)\b|"
                   rf"curl|roll|turn|"
                   rf"falls?\b|fell\b|collaps|drop|sink|sank|slump|tumbl|flop|(?:{_LED})\b", re.I)
_UNTIL = re.compile(r"\s*[,;:—–.!?]|\s+(?P<conj>(?:and|or)\s+(?!(?:down|up|forth|back|out|in|again|over|apart|together|onto|into|on|"
                    r"to|off|under|across|through|around|toward|towards|against|along|past|from|with|behind|between)\b)|"
                    r"(?:then|but|while|as(?!\s+(?:one|far|high|hard|much|fast|long|well|if|though|wide|close)\b)|before|after|until|"
                    r"till|so(?!\s+(?:high|far|hard|much|fast|wide|close)\b)|when|whenever|because|whilst|where|unless)\b)", re.I)


def _remote(before, after, sex=""):
    on, by = _ON_DEVICE.match(after), _DEVICE_ON.search(before)
    mine = on and (on.group("whose") or "").lower() in {"m": ("his", "their"), "f": ("her", "their")}.get(sex, ("their",))
    face = re.match(rf"['’]s\s+(?:face|image|picture)\s+(?:\w+\s+){{0,2}}?(?:on|in|fills?)\s+(?:the|a|an|her|his|their|[\w-]+['’]s)\s+"
                    rf"({_DEVICE})\b", after, re.I)
    dev = (on.group("dev") if on and not (mine and not _TALKS.search(before)) else "") or (
        next((g for g in by.groups() if g), "video call") if by else "") or (face.group(1) if face else "")
    dev = re.sub(r"^(?:the|a|an|her|his|their|its)\s+", "", dev.lower())
    if not dev:
        return None
    kind = "mirror" if re.search(r"\b(?:mirrors?|reflections?)$", dev) else "seen" if re.search(rf"\b(?:{_SEEN})$", dev) else \
        "heard" if re.search(rf"\b(?:{_HEARD})$", dev) else "still"
    return f"{'in' if kind in ('still', 'mirror') else 'on'} the {dev}", kind


def folk(text, people, cast, carried=()):
    told, found, gone = narration(text), {k: dict(v, here=False) for k, v in dict(carried).items()}, set()
    names = "|".join(re.escape(n) for n in sorted(cast, key=len, reverse=True)) or "(?!)"
    for m in _FOLK_RX.finditer(told):
        word, det = m.group("noun").lower(), " ".join((m.group("det") or "").lower().split())
        sing, plural, sex = next(r for r in _FOLK if word in r[:2])
        before, after = told[:m.start()], told[m.end():]
        if not det and sing not in ("someone", "person", "stranger", "onlooker", "passerby") or (
                not det and word == sing and sing != "someone") or re.search(rf"(?-i:\b(?:{names}))\s*,\s*$", before) or \
                _AWAY.search(before) or _THINGS.match(after) or re.match(rf"-\w|['’]s?\s+(?:{_ITEMS})s?\b", after, re.I):
            continue
        known = det.split(" ")[0] in ("the", "this", "that", "these", "those", "both")
        far = _remote(before, after, sex)
        if far and far[1] == "mirror":
            continue
        voice = not far and bool(re.match(r"['’]s\s+voice\b", after)) and (sing not in found or not known)
        dev = bool(far)
        key = f"{sing}@" if dev or voice or known and sing not in found and f"{sing}@" in found else sing
        if key == sing and known and sing not in found and not re.match(_AFTER, after, re.I) and any(
                re.search(rf"\b(?:{sing}|{plural})\b", people[n]["desc"], re.I) for n in cast if n in people):
            continue
        words = m.group(0).lower().split()
        k = next((v for w, v in (("two", 2), ("both", 2), ("pair", 2), ("couple", 2), ("three", 3), ("four", 4), ("five", 5))
                  if w in words), None)
        count = 1 if sing == "someone" else k or (None if word == plural or sing == "crowd" else 1)
        if key in found:
            was = found[key]["count"]
            count = (was + 1 if was else None) if det == "another" else count if count and was and count > was else was
        kind = found[key]["kind"] if key in found else far[1] if far else "heard" if voice else ""
        where = found[key]["where"] if key in found else far[0] if far else "off screen" if voice else ""
        found[key] = {"noun": sing, "sex": sex, "count": count, "here": True, "where": where, "kind": kind}
        if _GOES.match(after):
            gone.add(key)
    gone |= {k for k, v in found.items() if v["where"] and _HANGUP.search(told)}
    for v in found.values():
        sing, plural = v["noun"], next(r[1] for r in _FOLK if r[0] == v["noun"])
        base = "someone" if sing == "someone" else "the crowd" if sing == "crowd" else f"the {sing}" if v["count"] == 1 else \
            f"the {_COUNT.get(v['count'], v['count'])} {plural}" if v["count"] else f"the {plural}"
        v["label"] = f"{base} {v['where']}".strip()
    return found, gone


def _folk_ref(found, people=None, cast=()):
    found = found or {}
    rows = [(next(r for r in _FOLK if r[0] == v["noun"]), v["label"], bool(v["where"])) for v in found.values()]
    rows += [(r, who[0], False) for r in _FOLK if all(v["noun"] != r[0] for v in found.values()) and r[0] not in (
        "man", "woman", "guy", "girl", "boy", "lady", "gentleman", "person")
             for who in [[n for n in cast if n in (people or {}) and re.search(rf"\b(?:{r[0]}|{r[1]})\b", people[n]["desc"], re.I)]]
             if len(who) == 1]
    rx = "|".join(rf"{_DET}(?:{s}|{p})\b{_TAGGED_AS}?" for (s, p, _), _, _ in rows) or "(?!)"

    def label(phrase):
        hits = [(far, lab) for (s, p, _), lab, far in rows if re.search(rf"\b(?:{s}|{p})\b", phrase, re.I)]
        on = bool(re.search(rf"\b(?:{_SEEN}|{_HEARD}|{_PICTURE}|voice)\b", phrase, re.I))
        return next((lab for far, lab in hits if far == on), hits[0][1] if hits else None)
    return rx, label


def paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]


def narration(text):
    return _SPEECH.sub(lambda m: " " * len(m.group(0)), text or "")


def _sentences(text):
    told, out, at = narration(text), [], 0
    for m in list(re.finditer(r"(?<=[.!?])\s+|\n+", told)) + [None]:
        end = m.start() if m else len(told)
        out += [(at, end)] if text[at:end].strip() else []
        at = m.start() + len(re.match(r"\s*", text[m.start():]).group(0)) if m else end
    return out


def _say(q):
    return SAY_SLACK + len(q.split()) / WORDS_PER_SEC + SENTENCE_PAUSE * len(_STOP.findall(q))


def timeline(lines, marks=(), packed=False):
    cues, end, after = [], SPEECH_LEAD - CUE_GAP, 0
    for ln in lines:
        wait = 0.0 if packed else sum(s for a, b, s in marks if after <= a and b <= ln["at"])
        after = max(after, ln["end"])
        if ln["thought"]:
            continue
        long = ln.get("long") or _say(ln["q"])
        start = max(end + CUE_GAP, (end if cues else SPEECH_LEAD) + wait)
        cues.append((round(start, 2), round(start + (min(MUFFLE_MAX, long) if ln["gag"] else long), 2),
                     "muffle" if ln["gag"] else "line"))
        end = cues[-1][1]
    return cues


def voices(text, cast, people, start, events, prior=None, crowd=None, before="", ahead=""):
    told, spans, ago = narration(text), list(_SPEECH.finditer(text or "")), narration(before) + "\n"
    heads = [s.start() for s in spans[1:]] + ([len(told)] if _SPEECH.match(ahead.lstrip()) else [])
    names = "|".join(re.escape(n) for n in sorted(cast, key=len, reverse=True)) or "(?!)"
    caps = "|".join(re.escape(n.upper()) for n in sorted(cast, key=len, reverse=True)) or "(?!)"
    refer, extra = _folk_ref(crowd, people, cast)
    ones = {v["label"]: v["sex"] for v in (crowd or {}).values() if v["count"] == 1}
    here = [v["label"] for v in (crowd or {}).values() if v["count"] == 1 and v["here"]]
    speaker = lambda raw: (lambda lab: lab if lab in ones or lab in cast or not lab else f"one of {lab}")(extra(raw))
    who = rf"(?:(?-i:{names})|she|he|they|{refer})"
    lines, said, edge = [], prior if prior in cast else None, 0
    query = _EDGES.sub("", (_SPEECH.findall(before or "") or [""])[-1]).strip()
    asks = {"q": query, "who": said, "post": False} if said and query.endswith("?") else None
    sex = lambda n: people.get(n, {}).get("sex", "") if n in people else ones.get(n, "")

    def gag(n, at):
        on = next((x for x in start.get(n, []) if "mouth" in parts(x)), None)
        for k, arg, sent in events:
            w, items = split_tag(arg, people)
            for x in items if w == n and told.find(sent) <= at and "mouth" in parts(arg) else ():
                on = (x if k in ("hold", "have") else None if k == "release" else on) if "mouth" in parts(x) else on
        return on

    def acted(at, fit):
        told_ = told[:at] if re.search(rf"(?-i:\b(?:{names}))\b|\b(?:she|he|they)\b|\b(?:{refer})", told[:at], re.I) else ago + told[:at]
        found = [(m.start(), m.group(1)) for m in re.finditer(rf"(?-i:\b({names}))\b(?!['’])\s+(?:\w+ly\s+)?(?-i:[a-z]+(?:s|ed))\b",
                                                              told_)]
        found += [(m.start(1), m.group(1)) for m in re.finditer(rf"(?:^|[.!?]\s+|\n)\W*(?:{_LEADS}\W+)*(?-i:({names}))['’]s\b",
                                                                told_)]
        found += [(m.start(), speaker(m.group(1))) for m in re.finditer(rf"\b({refer})(?:['’]s\s+voice)?(?!['’])\s+(?:\w+ly\s+)?"
                                                                          rf"(?-i:[a-z]+(?:s|ed))\b", told_, re.I)]
        for m in re.finditer(r"\b(she|he)\s+(?:\w+ly\s+)?(?-i:[a-z]+(?:s|ed))\b", told_, re.I):
            one = [n for n in cast + list(ones) if sex(n) == {"she": "f", "he": "m"}[m.group(1).lower()]]
            found += [(m.start(), one[0])] if len(one) == 1 else []
        found += [(m.start(), None) for m in re.finditer(r"\b(?:they|both\s+of\s+them|the\s+two(?:\s+of\s+them|\s+women|\s+men)?)\s+"
                                                         r"(?:\w+ly\s+|just\s+)?[a-z]+\b", told_, re.I)]
        for _, n in sorted(found, reverse=True):
            if n is None or n in fit:
                return n or False
        return said if said in fit else None

    for k, m in enumerate(spans):
        q = _EDGES.sub("", m.group(0)).strip().rstrip(",").strip()
        ends = [told.rfind(c, 0, m.start()) + len(c) for c in (". ", "! ", "? ", "\n") if told.rfind(c, 0, m.start()) >= 0]
        lo = max(ends + [edge])
        hi = min([i for i in (told.find(c, m.end()) for c in (". ", "! ", "? ", "\n")) if i >= 0]
                 + [spans[k + 1].start() if k + 1 < len(spans) else len(told)])
        before, after = told[lo:m.start()], told[m.end():hi]
        tag = re.match(rf"\s*[,.!?;:—–-]*\s*(?:(?P<a>{who})\s+(?:\w+ly\s+)?(?P<va>{_SAYS}|thinks?|thought|wonders?|wondered)|"
                       rf"(?P<vb>{_SAYS})\s+(?P<b>(?-i:{names})|{refer}))\b", after, re.I)
        verb_after = (tag.group("va") or "") if tag else ""
        to_do = bool(tag) and bool(re.match(rf"\s+to\s+(?!(?:{_SAYS}|speak|talk)\b)", after[tag.end():], re.I))
        reacts = bool(tag) and k < len(heads) and bool(re.search(r"(?<!\.)\.$", q)) and bool(
            re.fullmatch(r"\s*[^.!?]+[.!?]\s*", told[m.end():heads[k]]))
        tag = None if tag and (_TRIES.fullmatch(verb_after) and to_do or _SOUNDS.fullmatch(verb_after) and reacts) else tag
        script = re.fullmatch(rf"[^a-z]*?(?-i:\b({names}|{caps}))\s*(?:\(([^()]*)\)\s*)?[:—–-]\s*", before)
        verb = ([v for v in re.finditer(rf"\b(?:{_SAYS})\b", before, re.I)] or [None])[-1]
        aim = re.fullmatch(rf"[\s,:—–-]*(?:\w+ly\s*)?(?:(?P<o>him|her|them|(?-i:{names}))\b(?!['’])\s*(?:\w+ly\s*)?)?"
                           rf"(?:(?:to|at|into|through|against|in|from|toward|towards)\s+(?P<t>[\w'’ -]+?))?[\s,:—–-]*"
                           rf"(?:(?:something|what)\s+(?:that\s+)?(?:sounds?|sounded)\s+like|something\s+like)?[\s,:—–-]*",
                           before[verb.end():], re.I) if verb else None
        near = [t for t in re.finditer(rf"\b{who}\b(?!['’])", before[:verb.start()], re.I)
                if len(before[t.end():verb.start()].split()) <= 16] if aim else []
        lead = ([t for t in near if not _AIMED.search(before[:t.start()]) or re.fullmatch(r",?\s+who\s+(?:\w+ly\s+)?",
                                                                                          before[t.end():verb.start()])] or [None])[-1]
        heard, post = told[edge:m.start()], bool(tag) and not (script or lead)
        hearer = rf"\s+(?:to\s+|at\s+)?(?:(?-i:({names}))\b(?!['’])|({refer})(?![\w'’])|(?:him|her|them)\b(?![\w'’]|\s+[a-z]))"
        listener = re.match(hearer, after[tag.end():], re.I) if post else \
            re.fullmatch(rf"(?-i:({names}))", aim.group("o") or aim.group("t") or "") if lead and not script else None
        said_to = listener.group(listener.lastindex) if listener and listener.lastindex else None
        said_to = said_to if said_to in cast else speaker(said_to) if said_to else None
        if script:
            raw = next((n for n in cast if n.upper() == script.group(1).upper()), script.group(1))
            v, edit = "says" + (f" ({script.group(2).strip()})" if script.group(2) else ""), [(lo + script.start(1), m.end())]
        elif lead:
            raw, v, edit = lead.group(0), verb.group(0), [(lo + lead.start(), lo + lead.end()), (lo + verb.start(), m.end())]
        elif tag:
            raw, v, edit = tag.group("a") or tag.group("b"), tag.group("va") or tag.group("vb"), [
                (m.start(), m.end() + tag.end() + (listener.end() if listener else 0))]
        else:
            raw, v, edit = None, "says", [(m.start(), m.end())]
        mum = {n for n in cast if re.search(rf"(?-i:\b{re.escape(n)})\b[^.!?]*?\b(?:(?:clamps?|presses?|purses?|seals?|keeps?)\s+"
                                            rf"(?:her|his|their)\s+(?:lips|mouth)\s+(?:shut|closed|together|tight)|says?\s+nothing|"
                                            rf"stays?\s+(?:silent|quiet)|refuses?\s+to\s+(?:speak|answer|talk))", heard, re.I)}
        aimed = {t.group(0) for t in near if t.group(0) in cast and lead and t.start() > lead.start() and _AIMED.search(before[:t.start()])}
        skip = set(re.findall(rf"(?-i:\b({names}))\b(?!\s+(?:I|you|we|he|she|they|it)\b)", q)) | mum | aimed | (
            {said_to} if said_to else set())
        words = re.findall(r"[a-z’']+", q.lower())
        sound = bool(words) and bool(re.fullmatch(r"[mnhgpfk]*[aeiou]?[mnhgpfk]+", words[0]) and re.search(
            r"mph|mf|ngh|mmm|nnn|mmh|hmph|nnh|mmn", words[0]))
        plea = re.search(r"\b(?:i|me|my|mine|i'm|i’m)\b", q, re.I) and not re.search(r"\byou\b", q, re.I)
        meek = bool(_MEEKSAY.search(v))
        gagged = [n for n in cast if gag(n, m.start())]
        if raw and raw.lower() not in ("she", "he", "they"):
            name = speaker(raw) or raw
        else:
            gender = {"she": "f", "he": "m"}.get((raw or "").lower())
            fit = [n for n in cast + (list(ones) if gender else here) if n not in skip and (not gender or sex(n) in (gender, ""))]
            muted = [n for n in fit if n in gagged]
            ok = fit if meek or sound else [n for n in fit if n not in gagged] or fit
            named_ = re.search(rf"(?-i:\b(?:{names}))\b|\b(?:she|he|they)\b|\b(?:{refer})", heard, re.I)
            prev = None if named_ else lines[-1] if lines else asks
            if sound and len(muted) == 1:
                name = muted[0]
            elif raw is None and prev and prev["who"]:
                asked = (prev["q"].rstrip().endswith("?") or lines and q.endswith("?") and re.search(r"\byou\b", q, re.I)) and not (
                    prev["post"] and not heard.strip(" ,.;:—–-") and not _REPLY.match(q) and lines[-2:][0]["who"] == prev["who"])
                rest = [n for n in ok if n != prev["who"] and not gag(n, m.start())]
                called = [n for n in re.findall(rf"(?-i:\b({names}))\b", prev["q"]) if n in rest]
                name = (called[0] if called else rest[0] if len(rest) == 1 else acted(m.start(), rest) or None if rest
                        else prev["who"]) if asked else prev["who"]
                name = (rest[0] if len(rest) == 1 else None) if name and gag(name, m.start()) and not (sound or plea) else name
            else:
                last = acted(m.start(), fit)
                scene = re.split(r"(?<=[.!?])\s+", told[:m.start()].strip())[-1]
                hurt = [n for n in ok if n != last and re.search(rf"(?-i:\b{re.escape(n)})\b|\b{'her' if sex(n) == 'f' else 'hi[ms]'}\b",
                                                                 scene, re.I)] if last and _ROUGH.search(scene) and _PLEAD.search(q) else []
                name = hurt[0] if len(hurt) == 1 else last if last and last in muted and plea else ok[0] if len(ok) == 1 else \
                    acted(m.start(), ok)
                freed = [n for n in ok if any(k == "release" and "mouth" in parts(arg) and split_tag(arg, people)[0] == n
                                              and 0 <= told.find(sent) < m.start() for k, arg, sent in events)]
                name = name or (freed[0] if len(freed) == 1 and name is None else None)
        name = name or (cast[0] if len(cast) == 1 else None)
        muffle = re.search(r"\b(?:through|into|behind|against)\s+(?:the|her|his|their)\s+(?:[\w-]+\s+){0,2}?(tape|gag|cloth|rag|"
                           r"socks?|panties|scarf)\b", before[verb.end():] if lead and not script else after[:tag.end() + 40]
                           if post else "", re.I)
        lines.append({"q": q, "who": name, "verb": v, "edit": edit, "gag": (gag(name, m.start()) or (muffle.group(1) if muffle else None))
                      if name else None, "post": post,
                      "thought": bool(re.fullmatch(r"thinks?|thought|wonders?|wondered", v, re.I)), "at": lo, "end": edit[-1][1]})
        said, edge = name or said, edit[-1][1]
    noises = []
    for m in re.finditer(rf"\b(?:{_NOISE})\b", told, re.I):
        lo = max([told.rfind(c, 0, m.start()) + len(c) for c in (". ", "! ", "? ", "\n") if told.rfind(c, 0, m.start()) >= 0] + [0])
        subject = re.search(rf"(?:(?-i:\b({names}))['’]s|\b(her|his))\s+(?:(?!(?:and|or|then|but)\b)\w+\s+){{0,2}}$",
                            told[lo:m.start()], re.I) or \
            ([None] + list(re.finditer(rf"(?-i:\b({names}))\b(?!['’])|\b(she|he)\b", told[lo:m.start()], re.I)))[-1]
        if subject is None or any(a <= m.start() < b for ln in lines for a, b in ln["edit"]):
            continue
        g = {"she": "f", "her": "f", "he": "m", "his": "m"}.get((subject.group(2) or "").lower())
        fit = [subject.group(1)] if subject.group(1) else [n for n in cast if gag(n, m.start()) and sex(n) in (g, "")]
        item = gag(fit[0], m.start()) if len(fit) == 1 else None
        noises += [{"q": "", "who": fit[0], "verb": m.group(0), "edit": [], "gag": item, "thought": False, "at": lo, "end": m.end(),
                    "long": MUFFLE_SOUND}] if item else []
    return lines, noises


def gagword(item):
    return re.sub(r"^(?:a|an|the)\s+", "", re.sub(r"\s+(?:stuffed\s+in|tied\s+over|over|in)\s+.*$", "", item))


def _her(pre, post):
    if re.match(r"\s+back\b", post, re.I):
        return "poss" if re.search(r"\b(?:behind|on|onto|across|down|up|along|over|against|to|from|at|around|of|in|with|into)\s+$", pre,
                                   re.I) else "obj"
    if re.match(r"\s+own\b", post, re.I):
        return "own"
    nxt = re.match(r"\s+(\w+)\b(.*)", post, re.S)
    if re.match(rf"\s+(?:(?!(?:onto|on|into|in|to|from|off|over|under|at|by|with|her|his|their|him|them|down|up|across|along|"
                rf"through|toward|towards|around|out|away|the|a|an)\b)[\w-]+\s+){{0,2}}?"
                rf"(?:{_SPOTS}|{_GARMENT.pattern}|hair|face|head|body|shoulders?|skin|cheeks?|chin|forehead|ears?|"
                rf"nose|neck|fingers?|toes?|nails?|elbows?|belly|stomach|back|side|eyes?|voice|name|phone|bag|keys?|room|bed|seat|"
                rf"chair|car|hand)\b", post, re.I):
        return "poss"
    if nxt and nxt.group(1).lower().endswith("ly") and nxt.group(1).lower() != "only":
        return "obj" if _OBJ_NEXT.match(nxt.group(2)) else "poss"
    if _OBJ_NEXT.match(post):
        return "obj"
    return None if _CAUSE.search(pre) or nxt and nxt.group(1).lower().endswith("ing") else "poss"


def named(text, read, cast, people, call, crowd=None, before="", lines=(), held=()):
    xsex = {v["label"]: v["sex"] for v in (crowd or {}).values() if v["count"] == 1}
    xall = {v["label"]: v["sex"] for v in (crowd or {}).values()}
    strangers = set(xsex) | {f"one of {v['label']}" for v in (crowd or {}).values() if v["count"] != 1}
    told, out, spoken = narration(text), {}, {}
    sex = lambda n: people.get(n, {}).get("sex", "") if n in people else xall.get(n, "")
    fits = lambda pro: [n for n in cast + list(xsex) if sex(n) in ({"her": "f", "she": "f", "his": "m", "him": "m", "he": "m"}.get(
        pro.lower(), sex(n)), "")]
    names = "|".join(re.escape(n) for n in sorted(cast, key=len, reverse=True))
    if not names:
        return []
    refer, extra = _folk_ref(crowd, people, cast)
    for k, arg, sent, doer in read:
        who, items = split_tag(arg, people)
        at = told.find(sent)
        if at < 0 or who not in cast or doer not in cast and doer not in strangers or doer == who:
            continue
        if k in ("remove", "tear", "lift", "wear"):
            own = rf"(?P<g>{_GARMENT.pattern})"
        else:
            groups = set().union(set(), *map(parts, items))
            own = "|".join(rx for p, rx in _BODY if p in groups) + ("|head|face" if groups & {"mouth", "eyes"} else "") + \
                ("|legs" if "body" in groups else "")
            own = rf"(?:{own})\b" if own else "(?!)"
        for m in re.finditer(rf"\b(?P<p>her|his|their)\s+(?!own\b)(?:[\w-]+\s+){{0,2}}?{own}|\b(?:{_VERBS})\s+(?:\w+ly\s+)?"
                             rf"(?P<o>her|him)\b{_OBJECT}", sent, re.I):
            pro = m.group("p") or m.group("o")
            last = list(re.finditer(rf"(?-i:\b({names}))\b(?!['’])(?P<v>\s+(?:\w+ly\s+)?[a-z]+(?:s|ed)\b)?", sent[:m.start()]))
            if last and last[-1].group(1) == who and last[-1].group("v"):
                continue
            if who in fits(pro) and len(fits(pro)) > 1 and (not m.groupdict().get("g") or any(
                    worn_rx(m.group("g")).search(y) for y in items)) and not (pro.lower() == "their" and len(
                    {split_tag(a, people)[0] for _, a, s_, _ in read if s_ == sent}) > 1):
                s = at + m.start("p" if m.group("p") else "o")
                out[(s, s + len(pro))] = (who, f"{call(who)}'s" if m.group("p") else call(who))
        lead = _SHE_LEADS.match(sent)
        if lead and len(fits(lead["pro"])) > 1 and doer in fits(lead["pro"]) and (doer not in cast and not re.search(
                rf"\b(?:{refer})", sent, re.I) or doer in cast and not re.search(
                rf"(?-i:\b(?:{names}))\b(?!['’])\s+(?:\w+ly\s+)?(?-i:[a-z]+(?:s|ed))\b", sent)):
            out[(at + lead.start("pro"), at + lead.end("pro"))] = (doer, call(doer))
    for ln in lines:
        for a, b in ln["edit"] if ln["who"] else ():
            spoken.update({(a + m.start(), a + m.end()): ln["who"] for m in re.finditer(r"\b(?:she|he|her|him|his)\b", told[a:b], re.I)})
    known = {g: [n for n in cast if sex(n) == g] for g in "fm"}
    fit = lambda w, g: isinstance(w, str) and (sex(w) == g or not sex(w) and (w not in cast or not known[g]))
    tok = re.compile(rf"(?P<name>(?-i:\b(?:{names}))\b)(?P<ns>['’]s\b)?|(?P<x>\b(?:{refer}))(?P<xs>['’]s\b)?|"
                     rf"\b(?P<pro>she|he|her|him|his|herself|himself)\b|(?P<cut>{_CUT_AT.pattern})", re.I)
    lead_in = re.compile(rf"\W*(?:(?:{_LEADS}|\w+ly)\W+)*(?:[\w-]+-haired\s+)?", re.I)
    pair = re.compile(rf"\s+and\s+(?:[\w-]+-haired\s+)?(?:(?-i:\b(?:{names}))\b|{refer})", re.I)
    scopes = []

    def pick(g, now, skip, subject=False, meek=False):
        for refs, top in [(now, None)] + [(sc["refs"], sc["subj"]) for sc in reversed(scopes)]:
            cands = list(dict.fromkeys(w for w in refs if fit(w, g) and w not in skip))
            if any(isinstance(w, tuple) and w[1] == g for w in refs):
                return None
            if not cands:
                continue
            if len(cands) == 1:
                return cands[0]
            if refs is now or not subject:
                return None
            if meek:
                rest = [w for w in cands if w != top] if top in cands else [w for w in cands if w in held]
                return rest[0] if len(rest) == 1 else None
            return top if top in cands else None
        named_ = [n for n in cast if fit(n, g) and n not in skip]
        return named_[0] if len(named_) == 1 and not any(fit(x, g) and x not in skip for x in xall) else None

    for sent in (m.group(0) for m in re.finditer(r"[^.!?\n]+[.!?\"”]*", narration(before))):
        refs, top = [], None
        for m in tok.finditer(sent):
            who = m.group("name") or (extra(m.group("x")) if m.group("x") else None)
            if who:
                refs.append(who)
                top = top if top is not None or m.group("ns") or m.group("xs") or not lead_in.fullmatch(sent[:m.start()]) else \
                    "" if pair.match(sent[m.end():]) else who
        scopes.append({"refs": refs, "subj": top})
    for base, sent in ((m.start(), m.group(0)) for m in re.finditer(r"[^.!?\n]+[.!?\"”]*", told)):
        refs, top, local, cut, toks, near, fronted = [], None, None, 0, list(tok.finditer(sent)), [], set()
        later = lambda k: {w for t in toks[k + 1:] for w in [t.group("name") or (extra(t.group("x")) if t.group("x") else None)] if w}
        lead_ing = lambda at, end: re.match(rf"\W*(?:\w+ly\s+)?[a-z]+ing\b[^,]*{end}", sent[:at], re.I)
        for k, m in enumerate(toks):
            if m.group("cut"):
                cut, near = m.end(), []
                continue
            s, e = base + m.start(), base + m.end()
            start = bool(lead_in.fullmatch(sent[cut:m.start()]))
            if not m.group("pro"):
                who = m.group("name") or extra(m.group("x"))
                if not who:
                    continue
                refs.append(who)
                near.append(who)
                if start and not (m.group("ns") or m.group("xs")) or _CAUSE.search(sent[:m.start()]):
                    local = "" if pair.match(sent[m.end():]) else who
                    top = local if top is None else top
                continue
            pro = m.group("pro").lower()
            if pro.endswith("self"):
                continue
            g = "f" if pro in ("she", "her") else "m"
            kind = "subj" if pro in ("she", "he") else "obj" if pro == "him" else "poss" if pro == "his" else \
                _her(sent[:m.start()], sent[m.end():])
            fixed = out.get((s, e), (spoken.get((s, e)),))[0]
            if fixed:
                who = fixed
            elif kind == "subj":
                cata = re.match(rf"\W*(?:as|when|while|before|after|once|if|because|since|though|although|until)\b[^,]*,\s*"
                                rf"(?:[\w-]+-haired\s+)?((?-i:{names}))\b", sent, re.I)
                meek = bool(re.match(rf"\s+(?:\w+ly\s+)?{_MEEK.pattern}", sent[m.end():], re.I))
                again = re.search(rf"(?:(?-i:\b(?:{names}))|\b(?:{refer}))\s*[,—–-]\s*$", sent[:m.start()], re.I)
                who = cata.group(1) if cata and m.start() < sent.find(",") and fit(cata.group(1), g) else \
                    None if again else pick(g, refs, later(k) | (fronted if lead_ing(m.start(), r",\s*$") else set()), True, meek)
            elif kind == "obj":
                same = local and fit(local, g)
                ctrl = re.search(r"\b(?:tells?|told|asks?|asked|orders?|ordered|wants?|wanted|begs?|begged|forces?|forced|lets?|"
                                 r"makes?|made|helps?|helped|watch(?:es|ed)?|sees|saw|hears?|heard|gets?|got|allows?|allowed|"
                                 r"needs?|needed)\b", sent[cut:m.start()], re.I)
                who = None if isinstance(local, tuple) and local[1] == g or same and ctrl else pick(g, refs, {local} if local else set())
                fronted |= {who} if who and lead_ing(m.start(), "$") else set()
            elif kind == "poss":
                if local and (fit(local, g) or isinstance(local, tuple) and local[1] == g):
                    refs.append(local)
                    continue
                who = pick(g, refs, {local} if local else set())
            else:
                continue
            refs.append(who or ("?", g))
            if kind == "poss" and near and near[-1] == who:
                continue
            near.append(who)
            if kind == "subj":
                local = who or ("?", g)
                top = local if top is None else top
            elif kind == "obj" and _CAUSE.search(sent[:m.start()]):
                local = who or ("?", g)
            if who in cast and (s, e) not in out and (s, e) not in spoken:
                out[(s, e)] = (who, f"{call(who)}'s" if kind == "poss" else call(who))
        scopes.append({"refs": refs, "subj": top})
    return [(s, e, x) for (s, e), (_, x) in sorted(out.items())]


def voiced(text, lines, call, names=()):
    text, edits = text or "", []
    for ln in lines:
        w, q, item, last = ln["who"], ln["q"], ln["gag"], len(ln["edit"]) - 1
        verb = "says" if re.fullmatch(r"tries|tried|manages?|managed|starts?|started|begins?|began|tells?|told", ln["verb"], re.I) \
            else ln["verb"]
        say = f"thinks, silently and with closed lips: {q}" if ln["thought"] else \
            f"lets out a muffled, closed-mouth mmmph through the {gagword(item)}" if item else \
            f"{verb}: " + f" {w} continues: ".join(f"<d>{x}</d>" for x in _STOP.split(q))
        for j, (s, e) in enumerate(ln["edit"]):
            gone = re.match(rf"\s+(?:in|into|through|behind|against|around|past|under)\s+(?:the|her|his|their)\s+(?:[\w-]+\s+){{0,2}}?"
                            rf"(?:(?:tape|gag)\b|{_CLOTH.pattern})", text[e:], re.I) if item and j == last else \
                re.match(r"\.(?!\.)", text[e:]) if j == last and re.search(r"[.!?]$", q) else None
            e = e + gone.end() if gone else e
            piece = (call(w) if last and not j else say if last else f"{call(w)} {say}") if w else \
                f"<d>{q}</d>" if j == last else text[s:e]
            piece += "." if item and j == last and re.search(r"[.!?]$", q) and re.match(r"\s+[\"“‘']?(?-i:[A-Z])", text[e:]) else ""
            edits.append((s, e, piece))
    edits += [x for x in names if not any(a < x[1] and x[0] < b for a, b, _ in edits)]
    out, at = [], 0
    for s, e, piece in sorted(edits):
        head = not text[:s].strip() or bool(re.search(r"(?:[.!?][\"”’']?\s+|\n\s*)$", text[:s]))
        out += [text[at:s], cap(piece) if head else piece]
        at = e
    return "".join(out) + text[at:]


def align(n, nearest=False):
    n = max(5, int(n))
    lo = n - (n - 5) % 17
    return min(MAX_FRAMES, lo if lo == n or (nearest and n - lo <= lo + 17 - n) else lo + 17)


def actions(text):
    told = narration(text)
    return sum(1 for a, b in _sentences(text) for c in _CLAUSE.split(told[a:b]) if c and len(c.split()) >= 2 and not re.fullmatch(
        rf"\W*(?:(?:[\w'’-]+\s+){{1,3}}?(?:\w+ly\s+)?(?:{_SAYS})|(?:{_SAYS})\s+(?:[\w'’-]+\s*){{1,3}})[\W\s]*", c, re.I))


def _kind(m):
    return {"pull": "lift"}.get(m.group(1).lower(), m.group(1).lower())


def read_par(par):
    out = {"lines": [], "exit": [], "seconds": None, "cut": False, "unread": [], **{k: [] for k in STATE_TAGS}}
    for line in (piece for raw in par.splitlines() for piece in _INLINE.split(raw)):
        m = _TAG.match(line)
        if m is None:
            if _CUT.match(line):
                out["cut"] = True
            elif line.strip():
                out["lines"].append(line.strip())
            continue
        kind, arg = _kind(m), m.group(2)
        if kind == "exit":
            out["exit"] += [w.strip() for w in arg.split(",") if w.strip()]
        elif kind == "seconds":
            try:
                out["seconds"] = float(arg.lower().rstrip("s ").strip())
            except ValueError:
                out["unread"].append(line.strip())
        else:
            out[kind].append(arg)
    out["text"] = "\n".join(out["lines"])
    return out


def beats(prompt):
    keys = STATE_TAGS + ("exit", "unread")
    out, early, cut = [], [], False
    for p in map(read_par, paragraphs(prompt)):
        if p["text"]:
            for q in early:
                for k in keys:
                    p[k] = q[k] + p[k]
            p["cut"], early, cut = p["cut"] or cut, [], False
            out.append(p)
        elif p["cut"] and p["seconds"] is None and not any(p[k] for k in keys):
            cut = True
        elif out:
            for k in keys:
                out[-1][k] += p[k]
            out[-1]["seconds"] = p["seconds"] or out[-1]["seconds"]
            cut = cut or p["cut"]
        else:
            early.append(p)
            cut = cut or p["cut"]
    return out


def header(line):
    m = _HEAD.match(line)
    if not m:
        return "", "", None, ""
    name, rest = m.group("name"), m.group("rest").strip()
    name = name.title() if name.isupper() else name
    if name.lower() in _SCENES:
        return "scene", "", None, ""
    if name.lower() in _LABELS:
        named = name.lower() in ("name", "character") and re.fullmatch(NAME, rest)
        return ("person", rest, None, "") if named else ("label", "", None, "")
    if m.group("sep") or not rest:
        return "person", name, int(m.group("pic")) if m.group("pic") else None, rest
    return "", "", None, ""


def read_memory(memory):
    people, scene, tags, who = {}, [], {k: [] for k in STATE_TAGS}, None
    for raw in (memory or "").splitlines():
        line = raw.strip()
        if not line:
            who = None
            continue
        t = _TAG.match(line)
        if t and _kind(t) in tags:
            tags[_kind(t)].append(t.group(2))
            continue
        kind, name, pic, rest = header(line)
        if kind == "person":
            who = name
            person = people.setdefault(who, {"desc": "", "head": None})
            person["head"] = person["head"] or pic
            person["desc"] = f"{person['desc']} {rest}".strip()
        elif kind == "scene" or not who:
            who = None if kind == "scene" else who
            scene.append(line)
        else:
            people[who]["desc"] = f"{people[who]['desc']} {line}".strip()
    for p in people.values():
        she, he = bool(_SHE.search(p["desc"])), bool(_HE.search(p["desc"]))
        p["sex"] = "f" if she and not he else "m" if he and not she else ""
        hair = _HAIR.search(p["desc"])
        p["hair"] = hair.group(1).lower() if hair else ""
    return people, " ".join(scene), tags


def owners(people, texts):
    out = {}
    for n, p in people.items():
        for k in [p["head"]] + [int(x) for x in re.findall(_PIC, p["desc"], re.I)]:
            if k:
                out.setdefault(k, n)
    for text in texts:
        last = None
        for sent in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
            hit = [n for n in people if re.search(rf"(?<![\w'’-]){re.escape(n)}(?![\w-])", sent)]
            owner = hit[0] if len(hit) == 1 else None if hit else last
            for m in re.finditer(_PIC, sent, re.I):
                near = [n for n in hit if re.search(rf"{re.escape(n)}(?:['’]s)?[\s,:(\[\-–—=]*$", sent[:m.start()])
                        or re.match(rf"[\s:,\-–—=)]*{re.escape(n)}(?![\w-])", sent[m.end():])]
                if near or owner:
                    out.setdefault(int(m.group(1)), near[0] if near else owner)
            last = owner if hit or re.search(_PIC, sent, re.I) else last
    return out


def retag(text, slot):
    def sub(m):
        k = int(m.group(2))
        if k in slot:
            return f"{m.group(1) or ''}<Picture {slot[k]}>{m.group(3) or ''}"
        return "" if m.group(1) and m.group(3) else (m.group(1) or "") + (m.group(3) or "")
    text = re.sub(r"[ \t]+([,.;:])", r"\1", _TAGGED.sub(sub, text))
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"([:,])\s*,", r"\1", text)).strip()


def mentions(text, people):
    told = narration(text)
    return [n for n in people if re.search(rf"(?<![\w'’-]){re.escape(n)}(?![\w-])", told)]


def split_tag(arg, people, garments=False):
    text = arg.strip()
    sep = r"(?:\s*[,:;–—]\s*|\s+-\s+|\s+|$)"
    who = next((p for p in sorted(people, key=len, reverse=True) if re.match(rf"{re.escape(p)}{sep}", text, re.I)), None)
    if who:
        rest = re.sub(rf"^{re.escape(who)}{sep}", "", text, count=1, flags=re.I)
    else:
        who, rest = (re.split(r"\s*[,:;–—]\s*|\s+-\s+", text, maxsplit=1) + [""])[:2]
    items = [i.strip() for i in re.split(r"\s*[;,]\s*|\s+and\s+" if garments else r"\s*;\s*", rest) if i.strip()]
    if garments in ("remove", "lift"):
        items = ["*" if garments == "remove" and _ALL.match(i) else re.sub(r"^(?:her|his|their|the|a|an|some)\s+", "", i, flags=re.I)
                 for i in items]
    return who.strip(), items


def worn_rx(item):
    g = item.lower()
    if g == "*":
        return _GARMENT
    return re.compile(rf"\b(?:{_GENERIC[g]})\b" if g in _GENERIC else rf"\b{re.escape(_stem(g))}(?:e?s)?\b", re.I)


def arms(item, p):
    if "wrists" not in parts(item):
        return ""
    tied = _TIED_TO.search(item)
    head = item[:tied.start()] if tied else item
    return next((say.format(p=p, o=_OBJ.get(p, p)) for rx, say in _ARMS if re.search(rx, head, re.I)), "" if tied else f"behind {p} back")


def _tied(item):
    m = _TIED_TO.search(item or "")
    if not m or m["link"]:
        return None
    to = m.group("to")
    of = re.search(rf"\bof\s+(the\s+(?:[\w-]+\s+)?(?:{_FIXTURES}))$", to, re.I)
    return {"to": to, "on": "the bed" if _BEDPART.search(to) else of.group(1) if of else "the chair" if _ARMREST.search(to) else to,
            "face": bool(m.group("lie")) and m.group("lie").lower().startswith("face"), "lies": bool(m.group("lie")),
            "up": bool(_OVERHEAD.search(to))}


def _held_at(item, p):
    t = _tied(item)
    return f"{_TIES.get(next(iter(_gears(item)), ''), ('fastened',))[0]} to {t['to']}" if t else f"locked together {arms(item, p)}"


def _anchor(clause, upto, context="", gear=""):
    whole, found = f"{context} {upto}", None
    for m in _ANCHOR.finditer(clause):
        prep, a, lead = m.group("prep").lower(), m.group("a"), clause[:m.start()]
        after = lead[max([0] + [k.end() for k in re.finditer(rf"\b(?:{_STATE}|{_APPLY}|{_KIT})\b", lead, re.I)]):]
        if prep in ("on", "onto", "across") and not re.search(rf"\b(?!hog)(?:{_STATE}|lashed|secured|fastened|locked|pinned|held|"
                                                            rf"(?<!face )(?<!face-)down|"
                                                            rf"spread[\s-]?eagled?)\s+(?:[\w-]+\s+){{0,2}}?$", lead, re.I) or \
                prep == "from" and not (_OVERHEAD.search(a) and re.search(r"\b(?:hang|hung|suspend|dangl)", lead, re.I)) or \
                re.search(r"\b(?:next|close|up)\s+$", lead, re.I) or re.search(rf"\b(?:{_STRIDE}|{_HEADS}|hop(?:s|ped|ping)?)\b", after, re.I) \
                or re.search(rf"\b(?:{_STATE})\b[^,]*,\s*$", lead, re.I):
            continue
        if not re.search(rf"\b(?:{_FIXTURES})\b", a, re.I):
            seen = list(re.finditer(rf"\b(?:the|a|an)\s+(?:[\w-]+\s+){{0,2}}?(?:{_FIXTURES})\b",
                                    f"{context} {upto[:max(0, len(upto) - len(clause)) + m.start()]}", re.I))
            if not seen:
                continue
            thing = re.sub(r"^(?:a|an)\s+", "the ", seen[-1].group(0), flags=re.I)
            a = thing if re.fullmatch(r"it|them", a, re.I) else re.sub(r"^its\s+", "the ", a, flags=re.I) + f" of {thing}"
        found = (re.sub(r"\b(?:a|an)\s+", "the ", a, count=2, flags=re.I).lower(), prep, lead)
        break
    if not found and (re.search(rf"\b(?:{_STATE}|{_APPLY})\s+(?:(?:her|him|them|(?-i:[A-Z])[\w'’-]*)\s+)?down\b", clause, re.I)
                      or gear in ("chains", "straps") and not _POSE.search(upto)
                      and not re.search(r"\b(?:together|to\s+each\s+other)\b", clause, re.I)):
        lay = list(re.finditer(rf"\b(?:lies|lying|lay|laid|lays|down)\s+(?:[\w'’-]+\s+){{0,3}}?(?:on|onto|across)\s+((?:the|a|an)\s+"
                               rf"(?:[\w-]+\s+){{0,2}}?(?:{_FIXTURES}))\b", whole, re.I))
        found = (re.sub(r"^(?:a|an)\s+", "the ", lay[-1].group(1), flags=re.I).lower(), "on", "down ") if lay else None
    if not found:
        return None
    to, prep, lead = found
    head = ["bed"] if _BEDPART.search(to) else re.findall(rf"(?:{_FIXTURES})\b", to, re.I)
    cues = [m.group(1).lower() for m in re.finditer(
        rf"\b(sits?|sitting|seated|sat|lies|lying|lay|laid|lays|kneels?|kneeling|knelt|stands?|standing|stood|spread[\s-]?eagled?)\b"
        rf"(?:\s+[\w'’-]+){{0,4}}?\s+(?:on|onto|across|at|in|against|by|beside)\s+(?:the|a|an|her|his|their|its)\s+(?:[\w-]+\s+){{0,2}}?"
        rf"{re.escape(_stem(head[-1]))}", whole, re.I)] if head else []
    flat = bool(cues) and bool(re.match(r"l|spread", cues[-1]))
    lies = not _OVERHEAD.search(to) and ((bool(re.search(r"\bdown\s+$", lead, re.I)) or prep != "to")
                                         and bool(_LIE_ON.search(to) or _FLOOR.search(to)) or bool(_LIE_ON.search(to) or _BEDPART.search(to))
                                         and (flat or not cues) or bool(_FLOOR.search(to)) and flat)
    face = lies and bool(_PRONE.search(whole))
    pos = "spread apart" if re.search(r"\bspread|\bcorners?\b|\bbed\s?posts\b", f"{clause} {to}", re.I) else \
        "above {p} head" if re.search(r"head\s?board", to) or _OVERHEAD.search(to) else "at {p} sides" if lies else \
        "behind {p} back" if _SEAT.search(to) and not _ARMREST.search(to) else \
        "in front of {o}" if re.search(r"\b(?:tables?|desks?|counters?)$", to) else ""
    return to, lies, face, pos


def _lap(text):
    return _LAP.sub(lambda m: m.group(0)[:m.start("at") - m.start()] + " " * len(m.group("at")), text)


def parts(item):
    s = _lap(_TIED_TO.sub("", item.lower()))
    found = {"body"} if _BETWEEN.search(s) else set()
    s = _BETWEEN.sub(" ", s)
    found |= {p for p, rx in _BODY if re.search(rf"\b(?:{rx})\b", s)}
    return found or {p for p, rx in _KIND if re.search(rf"\b(?:{rx})\b", s)}


def _stem(item):
    words = re.findall(r"[\w'’-]+", item.lower())
    w = words[-1] if words else ""
    return w[:-1] if w.endswith("s") and not w.endswith("ss") and not re.fullmatch(_PAIRS, w) else w


def _same(a, b):
    return bool(_stem(a)) and _stem(a) == _stem(b)


def _has(rx, text):
    return bool(re.search(rf"\b(?:{rx})\b", text, re.I))


def _zone(text):
    return ({"lower", "upper"} if _has(_WHOLE, text) else set()) | ({"lower"} if _has(_LOWER, text) else set()) \
        | ({"upper"} if _has(_UPPER, text) else set())


def _covers(seg):
    if _SHEER.search(seg):
        return set()
    return ({"lower", "upper"} if _has(_ALL_COVER, seg) else set()) | ({"lower"} if _has(_LOW_COVER, seg) else set()) \
        | ({"upper"} if _has(_UP_COVER, seg) else set())


def _beneath(seg):
    if not _has(_UNDER, seg) or _has(f"{_LOW_COVER}|{_UP_COVER}|{_ALL_COVER}", seg):
        return set()
    return {"lower", "upper"} if _has("lingerie", seg) else {"upper"} if _has("bras?", seg) else {"lower"}


def _pieces(desc):
    parts = _JOIN.split(desc)
    return list(zip([""] + parts[1::2], parts[0::2]))


def _ends(text):
    return text.rstrip().endswith((".", "!", "?"))


def _fit(segs, g, loose=True):
    segs = [s for s in segs if _GARMENT.search(s)]
    hit = [s for s in segs if worn_rx(g).search(s)]
    if len(hit) == 1 or not _zone(g) or not hit and not (loose and _LOOSE_NAME.fullmatch(g.strip())):
        return hit
    under, outer = _has(_UNDER, g), bool(_OUTER.search(g))
    pool = [s for s in hit or segs if (hit or _zone(s) == _zone(g) and _has(_UNDER, s) == under) and bool(_OUTER.search(s)) == outer] \
        or (hit if outer else [])
    pool = ([s for s in pool if "lower" in _zone(s)] or pool) if under and len(pool) > 1 else pool
    return pool if len(pool) == 1 else hit


def undress(desc, off=(), hide=()):
    if not off and not hide:
        return desc
    out, carry, anded, last = [], None, False, ""
    for j, seg in _pieces(desc):
        attached = bool(j) and not j.strip() and not _ends(last)
        last = seg
        if seg in hide or (_GARMENT.search(seg) and any(worn_rx(g).search(seg) for g in off)) or (
                carry and attached and not _GARMENT.search(seg)):
            if attached and not carry and _REL.match(seg):
                if _ends(seg) and out and not _ends(out[-1][1]):
                    out[-1][1] = out[-1][1].rstrip() + seg.rstrip()[-1]
                continue
            lead = _LEAD.match(seg)
            carry, anded = carry or (j, lead.group(0) if lead else ""), anded or bool(re.search(r"\band\b", j, re.I))
            if _ends(seg):
                if out and not _ends(out[-1][1]):
                    out[-1][1] = out[-1][1].rstrip() + seg.rstrip()[-1]
                    _and(out, anded)
                carry, anded = None, False
            continue
        if carry:
            rel = _REL.match(seg) if attached else None
            seg = seg[rel.end():] if rel else seg
            if rel or carry[1] or not carry[0].strip():
                j = carry[0]
                seg = carry[1] + seg if carry[1] and _GARMENT.search(seg) and not _LEAD.match(seg) else seg
        out.append([j, seg])
        carry, anded = None, False
    if carry:
        _and(out, anded)
    text = re.sub(r"^\s*(?:,|;|and\b)\s*", "", "".join(j + s for j, s in out)).strip()
    text = re.sub(r"(?:^|(?<=[.!?;]))\s*(?:she|he|they)(?:\s+(?:is|are|was|were|has|have|also))*(?:\s+(?:wearing|wears|wear|dressed|"
                  r"clad))?\s*(?:[.!?;]|$)", "", text, flags=re.I).strip()
    return text + ("." if text and not _ends(text) else "")


def _and(out, anded):
    if anded and out and out[-1][0].startswith(",") and _GARMENT.search(out[-1][1]) and not _LEAD.match(out[-1][1]):
        out[-1][0] = " and "


def clean(seg, article=True):
    seg = re.sub(r"^[^\W\d_]+\s*:\s*", "", seg.strip())
    text = re.sub(r"\s+(?:underneath|beneath)$", "", _REL.sub("", _LEAD.sub("", seg)).rstrip(" .!?;,"), flags=re.I)
    return text if article else re.sub(r"^(?:her|his|their|the|a|an|some)\s+", "", text, flags=re.I)


def _owned(seg, person):
    text = clean(seg)
    return text if re.match(r"(?:her|his|their|the|a|an|some)\s", text, re.I) else f"{pron(person, 1)} {text}"


def _on(desc, off):
    return [s for _, s in _pieces(desc) if _GARMENT.search(s) and not any(worn_rx(g).search(s) for g in off)]


def _raised(seg, lifted):
    return any(worn_rx(g).search(seg) for g, _ in lifted)


def hidden(desc, off=(), worn=(), lifted=()):
    segs, covered = _on(desc, off), set()
    for s in segs + list(worn):
        up = next((w for g, w in lifted if worn_rx(g).search(s)), None)
        covered |= _covers(s) - (_bares(clean(s, False), up) if up is not None else set())
    return [s for s in segs if _beneath(s) and _beneath(s) <= covered and not _raised(s, lifted)]


def _many(items):
    return len(items) > 1 or any(g.endswith("s") and not g.endswith("ss") for g in items)


def _chest(person):
    return "breasts" if person.get("sex") == "f" else "chest"


def pulled(item, person, way=None):
    m = _WHERE.search(item)
    g = re.sub(r"^(?:her|his|their|the|a|an|some)\s+", "", (item[:m.start()] if m else item).strip(), flags=re.I)
    zone, p = _zone(g), pron(person)
    down = zone == {"lower"} and not _SKIRT.search(g)
    where = m.group(1) if m else way or ("open" if _OPENS.search(g) else f"up to {p} waist" if _SKIRT.search(g) or len(zone) > 1
                                         else f"down to {p} knees" if down else f"up above {p} {_chest(person)}" if zone
                                         else "aside")
    low = re.search(r"\b(?:knees|ankles|thighs|calves|feet|floor)\b", where, re.I)
    high = re.search(r"\b(?:breasts|chest|neck|shoulders|armpits|collarbones?|head)\b", where, re.I)
    return g, where if re.match(r"(?:up|down|open|aside|off)\b", where, re.I) else \
        ("down " if low or down and not high else "up ") + where


def _bares(g, where):
    zone = _zone(g)
    if len(zone) > 1 and not _OPENS.search(g):
        return zone if _UPWARD.search(where) else {"upper"} if where.lower().startswith("down") else {"lower"}
    return zone


def _leaving(person, zones, desc, off, worn, lifted, bonds=()):
    if not zones or any(_zone(s) & zones and not _raised(s, lifted) for s in _on(desc, off) + list(worn)):
        return ""
    p, o = pron(person), pron(person, 2)
    left = f", leaving {p} {_chest(person)} and everything below {p} waist bare" if zones >= {"lower", "upper"} else \
        f", leaving {p} {_chest(person)} bare" if "upper" in zones else f", leaving {o} bare below the waist"
    return left + _except(bonds, zones)


_NAKED = {"completely naked": {"lower", "upper"}, "naked from the waist down": {"lower"}, "naked from the waist up": {"upper"}}


def bare(desc, off, worn):
    left = set().union(*map(_zone, _on(desc, off) + list(worn)))
    took = {"lower", "upper"} if "*" in off else set().union(*map(_zone, off))
    gone = took - left
    return "completely naked" if gone >= {"lower", "upper"} else "naked from the waist down" if "lower" in gone else \
        "naked from the waist up" if "upper" in gone else ""


def cap(text):
    return text[:1].upper() + text[1:]


def pron(person, k=1):
    return {"f": ("she", "her", "her"), "m": ("he", "his", "him")}.get(person.get("sex", ""), ("they", "their", "them"))[k]


def join(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def labels(names, people, shown):
    hair = {n: people[n]["hair"] for n in names if n in people and people[n]["hair"] and n not in shown}
    tag = {n: f"{h}-haired {n}" for n, h in hair.items() if list(hair.values()).count(h) == 1}
    return tag if len(names) > 1 else {}


def _cut(text, n=70):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def _hits(text):
    found = [(g, d, m.start(), k == 1, m.group(0)) for g, sure, loose, d in _GEAR for k, rx in ((1, sure), (0, loose)) if rx
             for m in re.finditer(rf"\b(?:un|re)?(?:{rx})\b", text, re.I)]
    sure = [h for h in found if h[3]]
    tool = any(re.search(r"\b(?:with|using)\s+(?:[\w'’-]+\s+){0,3}$", text[:o[2]], re.I) for o in sure)
    spots = [(m.start(), _SPOT_AS.get(m.group(1).lower(), m.group(1).lower())) for m in re.finditer(
        rf"\b({_SPOTS})\b", _BETWEEN.sub(lambda m: "crotch".ljust(len(m.group(0))), _lap(text)), re.I)]
    near = lambda at: min(spots, key=lambda s: abs(s[0] - at))[1] if spots else ""
    keep = sure + [h for h in found if not h[3] and not tool and not any(
        o[3] and near(o[2]) == near(h[2]) and (h[4].lower().endswith("ed") or -len(o[4]) < o[2] - h[2] <= 30) for o in found)]
    return sorted(((g, d or next((o[1] for o in sorted(found, key=lambda o: abs(o[2] - at)) if not o[3] and o[1]), ""), at)
                   for g, d, at, _, _ in keep), key=lambda h: h[2])


def _kinds(text):
    return list({g: d for g, d, _ in reversed(_hits(text))}.items())[::-1]


def _gears(text):
    return {{"a ball gag": "a gag"}.get(g, g) for g, _ in _kinds(_TIED_TO.sub("", text))}


def _spots(text):
    return list(dict.fromkeys(_SPOT_AS.get(w.lower(), w.lower())
                              for w in re.findall(rf"\b({_SPOTS})\b", _BETWEEN.sub(" crotch ", _lap(text)), re.I)))


def _places(item):
    item = _TIED_TO.sub("", item)
    return set(_spots(item)) or {d for _, d in _kinds(item) if d}


def _frees(r, x, mine=()):
    if any(r.lower() == y.lower() for y in mine):
        return r.lower() == x.lower()
    kinds, spots = _gears(r), set(_spots(r))
    if kinds or spots:
        return (not kinds or bool(kinds & _gears(x))) and (not spots or bool(spots & _places(x)))
    word = re.sub(r"^(?:her|his|their|the|a|an|some)\s+", "", r.strip().lower()).split(" ")[0]
    return bool(word) and bool(re.search(rf"\b{re.escape(_stem(word))}", x, re.I))


def _spans(item):
    s = _lap(item.lower())
    return ({"lower"} if re.search(rf"\b(?:{_CROTCH}|waist|hips)\b", s) or _BETWEEN.search(s) else set()) | \
        ({"upper"} if re.search(r"\b(?:chest|breasts|torso)\b", s) else set())


def _except(items, zones):
    on = [re.sub(r"^(?:a|an)\s+", "", x) for x in items if _spans(x) & zones]
    return f", apart from the {join(on)}" if on else ""


def _meekness(verb):
    return 0 if not _MEEK.fullmatch(verb) else 1 if re.fullmatch(r"watches|waits|stares|blinks", verb, re.I) else \
        2 if re.fullmatch(r"sits|lies|kneels", verb, re.I) else 3


def _acts(text, spots=()):
    now = lambda m: not re.search(r"(?:ed|wound|stuck|laid|bound|fed|drew|brought|slid|ran)$", m.group(0), re.I)
    return any(map(now, _DOES.finditer(text))) or bool(spots) and any(map(now, _ROUTES.finditer(text)))


def _loosens(text):
    asked = lambda m: bool(re.search(r"\b(?:for|asks?|asked|asking|begs?|begged|begging|pleads?|pleaded|pleading|tells?|told|telling|"
                                     r"orders?|ordered|wants?|wanted|needs?|needed)\s+(?:(?-i:[A-Z])[\w'’-]*|her|him|them)\s+to\s+$",
                                     text[:m.start()], re.I))
    for m in _LOOSE.finditer(text):
        mid = m.group("mid")
        if asked(m):
            continue
        if mid is None or not mid.strip() or re.search(rf"\b(?:{_KIT}|it|them|her|him|his|their|{_SPOTS}|strips?|pieces?|ends?|"
                                                       rf"layers?|gag)\b|{_CLOTH.pattern}|(?-i:\b[A-Z])", mid, re.I) \
                and not re.search(r"\b(?:hand|finger|thumb|palm|head|face|hair|lips|mouth)\b", mid, re.I):
            return True
    return any(not asked(m) for m in _UNDO.finditer(text)) or bool(_FALLS.search(text))


def _gear_item(gear, spot, clause, p, duct=False, cloth="", upto="", context="", had=""):
    gear, spot = "tape" if gear == "duct tape" and not duct else gear, "wrists" if gear == "handcuffs" and spot == "arms" else spot
    if cloth and spot == "mouth" and (_INMOUTH.search(clause) or not _OVER.search(clause) and not _TIED.search(cloth)):
        return f"{cloth} stuffed in {p} mouth as a gag"
    if cloth and spot in ("mouth", "eyes"):
        return f"{cloth} tied over {p} {spot} as a {'gag' if spot == 'mouth' else 'blindfold'}"
    if gear in ("a gag", "a ball gag"):
        return f"{gear} in {p} mouth"
    if spot in ("mouth", "eyes"):
        return f"{gear} over {p} {spot}"
    if gear == "a spreader bar":
        return f"a spreader bar locked between {p} ankles"
    if gear == "a chastity belt":
        return f"a chastity belt locked around {p} waist, its shield between {p} legs over {p} crotch"
    if spot == "crotch" and gear in _HARNESS:
        u = _unit(gear)
        return f"a crotch {u} {_HARNESS[gear][0]} tight around {p} waist, its center {u} pulled tight under {p} crotch and over {p} " \
               f"{'vagina' if p == 'her' else 'groin'} from the front of {p} waist to the small of {p} back"
    o = _OBJ.get(p, p)
    fix = _anchor(clause, upto or clause, context, gear)
    one = spot == "ankles" and fix and not re.search(r"\b(?:ankles|feet|legs)\b", clause, re.I) and re.search(
        r"\b(?:(?P<side>left|right)\s+)?ankle\b(?!\s+(?:chains?|cuffs|shackles|irons|restraints))", clause, re.I)
    if one:
        return f"{gear} {'on' if gear == 'handcuffs' else 'around'} {p} {(one['side'] or 'left').lower()} ankle, " \
               f"{_TIES.get(gear, ('fastened',))[0]} to {fix[0]}"
    tie = f", {_TIES.get(gear, ('fastened',))[0]} {'face down ' if fix[2] else 'down ' if fix[1] else ''}to {fix[0]}" if fix else ""
    if spot == "crotch":
        return f"{gear} {'sealed' if gear.endswith('tape') else 'pulled tight'} over {p} crotch, between {p} legs{tie}"
    pose = list(_POSE.finditer(upto or clause)) if spot in ("wrists", "arms") else []
    pos = next(say.format(p=p, o=o) for rx, say in _ARMS if re.fullmatch(rx, pose[-1].group("at"), re.I)) if pose else had if had else \
        fix[3].format(p=p, o=o) if fix and (spot in ("wrists", "arms") or fix[3] == "spread apart") else \
        f"behind {p} back" if spot == "wrists" else ""
    if gear == "handcuffs" and pos and not fix and not _spots(pos):
        return f"handcuffs {pos}"
    return f"{gear} {'on' if gear == 'handcuffs' else 'around'} {p} {spot}" + (f" {pos}" if pos else "") + tie


def _garb(g):
    return next(((k, t, step, bound) for k, (rx, t, step, bound) in enumerate(_UNDRESS) if re.search(rf"\b(?:{rx})$", g, re.I)),
                (len(_UNDRESS), 4, "takes {w} {g} off", ""))


def _stuck(g, bonds, per):
    p, legs = pron(per), next((x for x in bonds if _places(x) & {"ankles", "feet", "legs", "knees", "thighs"}), None)
    if _zone(g) == {"lower"} and _has(_UNDER + "|tights|pantyhose", g) and any(_gear(x) == "a chastity belt" for x in bonds):
        return f"on under {p} chastity belt"
    if "upper" in _zone(g) and any("wrists" in parts(x) for x in bonds):
        return f"open and down off {p} shoulders" if _OPENS.search(g) else f"up above {p} {_chest(per)}"
    if _zone(g) != {"lower"} or not legs:
        return ""
    gear = re.sub(r"^an?\s+", "", _gear(legs))
    spot = next(s for s in ("thighs", "knees", "legs", "ankles", "feet") if s in _places(legs) or s == "feet")
    return f"down to just above the {gear} {'on' if gear in ('handcuffs', 'shackles', 'spreader bar') else 'around'} {p} {spot}"


def _cant(who, keep, per, where):
    one, p, on = not _many(keep), pron(per), where.startswith("on ")
    past = f"{p} chastity belt" if on else f"{p} bound {'wrists' if where.startswith(('up', 'open')) else 'legs'}"
    return f"{who}'s {join(keep)} cannot come off past {past}, so {'it' if one else 'they'} " \
           f"{('stays' if one else 'stay') + ' on' if on else ('is' if one else 'are') + ' pulled out of the way instead'}; release: " \
           f"the restraint in an earlier beat, or rip or cut {'it' if one else 'them'} off"


def _belt_look(on, per, pic):
    p, o, tag = pron(per), pron(per, 2), f" <Picture {pic}>" if pic else ""
    return f"{cap(p)} chastity belt{tag} is locked on {o}, its band around {p} waist and its shield between {p} legs over {p} crotch." \
        if on else f"{cap(p)} chastity belt{tag} goes on {o} in this shot." if tag else ""


def _pic_of(words, texts, taken=()):
    for rx in (rf"\b(?:{words})\b[^.\n<]{{0,40}}{_PIC}", rf"{_PIC}[^.\n<]{{0,40}}?\b(?:{words})\b"):
        for t in texts:
            k = next((int(m.group(1)) for m in re.finditer(rx, t or "", re.I) if int(m.group(1)) not in taken), None)
            if k:
                return k
    return None


def _gear(x):
    x = _TIED_TO.sub("", x)
    return next((g for g in _BIND if re.match(rf"{g}\b", x, re.I)), "") or next((g for g, _ in _kinds(x)), "restraints")


def _unit(g):
    return g[:-1] if g.endswith("s") else g


def _band(x):
    return _gear(x) in _HARNESS and bool(_places(x) & {"waist", "hips"})


def _crotch(x):
    return _band(x) and "crotch" in _places(x)


def _link(x):
    m = _TIED_TO.search(x or "")
    return (m["to"], m["link"].split()[-1]) if m and m["link"] else None


def _lashed(x):
    m = _band(x) and re.fullmatch(rf"(?P<to>{_WRISTS}) at the (?:small of \w+ back|front of \w+ (?P<front>waist))",
                                  (_link(x) or ("",))[0])
    return (m["to"], bool(m["front"])) if m else None


def _hogged(x):
    m = bool(x) and "ankles" in parts(x) and re.fullmatch(rf"(?P<to>{_WRISTS})(?: behind \w+ back| in front of \w+)?",
                                                           (_link(x) or ("",))[0])
    return m["to"] if m else None


def _links(text):
    zone = lambda w: next(k for k, rx in _REGIONS if re.fullmatch(rx, w, re.I))
    return [{"wrists", "ankles"} if not (m["a"] or m["c"]) else {zone(m["a"] or m["c"]), zone(m["b"] or m["d"])}
            for m in _LINK.finditer(text)]


def _joined(text):
    return any(len(z) > 1 for z in _links(text))


def _tether(x, wrist, p):
    to, at = f"{p} handcuffs" if "handcuffs" in _gears(wrist) else f"the {_unit(_gear(wrist))} around {p} wrists", arms(wrist, p)
    at = (f"at the front of {p} waist" if at.startswith("in front") else f"at the small of {p} back") if _band(x) else at
    return f"{_TIED_TO.sub('', x)}, {_TIES.get(_gear(x), ('fastened',))[0]} to {to}{' ' + at if at else ''} by a separate short " \
           f"{'rope' if 'rope' in (_gear(x), _gear(wrist)) else 'chain'}"


def _part(x, p):
    g, spot = _gear(x), "wrists" if "wrists" in parts(x) else "ankles"
    if _band(x):
        return f"{p} crotch {_unit(g)}" if _crotch(x) else f"the {_unit(g)} around {p} waist"
    return f"{p} handcuffs" if g == "handcuffs" and spot == "wrists" else f"{p} ankle cuffs" if g == "handcuffs" else \
        f"{p} {_unit(spot)} shackles" if g == "shackles" else "the spreader bar" if g == "a spreader bar" else \
        f"the {_unit(g)} around {p} {spot}"


def _point(x, p, front=False):
    return f"the {'front' if front else 'back'} of {_part(x, p)}" if _band(x) else f"the chain between {_part(x, p)}" \
        if _gear(x) in ("handcuffs", "shackles") else "the middle of the spreader bar" if _gear(x) == "a spreader bar" else _part(x, p)


def _linking(x, w, p, on=True):
    (to, front), strand = _lashed(x) or (_hogged(x), False), _link(x)[1]
    if not on:
        return f"{'unlocks' if strand == 'chain' else 'unties'} the separate short {strand} that joins " \
               f"{re.sub(r'\b(?:her|his|their)\b', w, to, count=1)} to {_point(x, p, front)} and takes it away"
    lead = "" if _band(x) else f"bends {w} knees and lifts {_low(x, p)} together up " + \
        ("toward " if "in front of" in _link(x)[0] else f"behind {_OBJ.get(p, p)} toward ") + f"{p} hands, then "
    to = re.sub(r"\b(?:her|his|their)\b", p if lead else w, to, count=1)
    return f"{lead}{'padlocks' if strand == 'chain' else 'ties'} a separate short {strand} from " \
           f"{f'the chain between {to}' if to.endswith('handcuffs') else to} to {_point(x, p, front)}"


def _swapped(x, items):
    base = lambda y: _TIED_TO.sub("", y)
    return next((y for y in items if y != x and _gear(x) == _gear(y) and (
        _band(x) and _band(y) and (base(x) == base(y) or _crotch(x) and not _crotch(y))
        or base(x) == base(y) and bool(_hogged(x)) != bool(_hogged(y)))), None)


def _put(items, adds):
    return [y for y in items if not any(_swapped(x, [y]) for x in adds)] + [x for x in adds if x not in items]


def _settle(items):
    free = not any("wrists" in parts(y) for y in items)
    return [_TIED_TO.sub("", x) if free and _link(x) else x for x in items]


def _takes(kind, x, items=()):
    was = _swapped(x, items) if kind == "hold" else None
    if was:
        return 4 + 2 * bool(_lashed(x)) if _crotch(x) and not _crotch(was) else 3
    if kind in ("hold", "release"):
        spots, gear = _places(x), _gear(x)
        tape = (3, 3) if spots <= {"mouth", "eyes"} else (6, 4) if "crotch" in spots else (8, 5) if spots & set(_BANDS) else \
            _BIND["duct tape"][:2]
        on, off = (4, 3) if " stuffed in " in x else (6, 4) if " tied over " in x else tape if gear == "duct tape" else \
            (_BIND[gear][0] + 1, _BIND[gear][1] + 1) if _crotch(x) else (4, 3) if "crotch" in spots else _BIND.get(gear, _BOUND)[:2]
        return on + 2 * bool(_tied(x) or _link(x)) if kind == "hold" else off + 2 * bool(_tied(x) or _link(x))
    if kind in ("remove", "wear"):
        return _garb(x)[1] * (1.3 if kind == "wear" else 1)
    return {"tear": 3, "lift": 3, "back": 2}.get(kind, 0)


def _still(p, flat):
    return f"{p} legs straight and flat where they lie" if flat else "both feet resting where they are"


def _taped(x, w, p, sent, flat=False):
    spot = (_spots(x) or ["mouth"])[0]
    if spot in ("mouth", "eyes") and not re.search(r"\baround\s+(?:her|his|their|[\w-]+['’]s)\s+head\b", sent, re.I):
        return f"tears a strip off the roll of duct tape and presses it flat over {w} {spot}", False
    pos = f" {arms(x, p)}" if spot == "wrists" and arms(x, p) else ""
    return (f"runs the duct tape from the front of {w} waist down between {p} legs and up to the back, flat over {p} crotch"
            if spot == "crotch" else
            f"presses the loose end of the duct tape flat against {w} {spot} and carries the roll all the way around {p} body, "
            f"passing it from one hand to the other behind {p} back, so the tape lies flat around {p} {spot}"
            if spot in _BANDS else
            f"carries the roll of duct tape around {w} head, over {p} {spot}" if spot in ("mouth", "eyes") else
            f"holds {w} {spot} pressed together, {_still(p, flat)}, and winds the duct tape around and around them"
            if spot in ("ankles", "knees", "legs") and not _tied(x) else
            f"winds the duct tape around {w} {spot}{pos}, turning the roll around them"
            + (f", then tapes them to {_tied(x)['to']}" if _tied(x) else "")), True


def _bind_step(x, nm, w, p, dp, on=True, laid=False, flat=False):
    gear, spot = _gear(x), (sorted(_places(x)) or ["wrists"])[0]
    thing = "the " + re.sub(r"^(?:a|an|the)\s+", "", re.split(r"\s+(?:stuffed\s+in|tied\s+over)\s+", x)[0])
    if " stuffed in " in x:
        return (f"pushes {thing} into {w} mouth with {dp} fingers until {p} mouth is full and {p} lips close around "
                f"{'them' if _many([thing]) else 'it'}" if on else f"pulls {thing} out of {w} mouth"), laid
    if " tied over " in x:
        return (f"ties {thing} over {w} {spot} and knots it behind {p} head" if on else f"unties {thing} and takes it off {w} {spot}"), laid
    words, unit, tied = _BIND.get(gear, _BOUND), _unit(gear), _tied(x)
    one = re.search(r"\b(?:left|right) ankle\b", x)
    spot = one.group(0) if one else spot
    lock, do, crotch = words[4], _TIES.get(gear, ("fastened", "fastens"))[1], spot == "crotch" and bool(words[4])
    if not on:
        at = f"under {w} crotch" if crotch else f"{w} {spot}"
        return words[3].format(x=w, s=spot, g=gear, off=("out from " if crotch else "off ") + at, fr="from " + at), laid
    if crotch:
        return f"runs the {unit} from the front of {w} waist under {p} crotch and up to the back, pulls it tight and " \
               f"{lock.format(t=' there')}" + (f", then {do} it to {tied['to']}" if tied else ""), laid
    arm = spot in ("wrists", "arms")
    pos, limbs = arms(x, p) or ("spread apart" if "spread apart" in x else ""), "arms" if arm else "legs"
    if tied and tied["lies"]:
        lay = "" if laid else f"lays {nm} down on {p} {'front' if tied['face'] else 'back'} on {tied['on']}"
        o = p if lay else w
        put = {f"at {p} sides": f"lays {o} arms flat on it at {p} sides", f"above {p} head": f"stretches {o} arms out above {p} head",
               "spread apart": f"spreads {o} {limbs} out wide", f"behind {p} back": f"keeps {o} arms behind {p} back"}.get(pos, "")
        return join([s for s in (lay, put, f"{do} {p if lay or put else w} {spot} to {tied['to']}") if s]), True
    if spot in ("ankles", "knees", "legs") and not tied and not pos and gear != "a spreader bar":
        put = _FEET.get(gear, "fastens the {g} around both of them together").format(g=gear, one=_PAIR[spot])
        return f"holds {w} {spot} pressed together, {_still(p, flat)}, and {put}", laid
    bring = {f"at {p} sides": f"arms down to {p} sides", f"at {p} waist": f"hands to {p} waist"}.get(
        pos, f"hands down {pos}" if pos.startswith("between") else f"arms {pos}")
    lead = f"spreads {w} {limbs} apart and " if pos == "spread apart" else f"brings {w} {bring} and " if pos and arm else ""
    x_ = p if lead else w
    return lead + (f"{do} {x_} {spot} to {tied['to']}" if tied else (words[2] if spot in _LIMBS or not lock else
                   f"wraps the {unit} around {{x}} {{s}} and {lock}").format(x=x_, p=p, s=spot, one=spot.rstrip("s"), g=gear, t="")), laid


def _harness_step(x, w, p, on=True, was=None, freed=False, dp="their"):
    gear, crotch, link = _gear(x), _crotch(x), _lashed(x)
    u, (_, opens, hook), close = _unit(gear), _HARNESS[gear], _BIND[gear][4]
    if not on:
        q = p if freed and link else w
        off = f"{opens} at the small of {q} back, pulls the center {u} out from under {p} crotch and takes the crotch {u} off {p} waist" \
            if crotch else _BIND[gear][3].format(x=q, s="waist", g=gear, off=f"off {q} waist", fr=f"from {q} waist")
        return f"{_linking(x, w, p, False)}, then {off}" if freed and link else off
    if was and _crotch(was) == crotch:
        return _linking(x if link else was, w, p, bool(link))
    lead = (f"{hook} a center {u} to the front of the {u} around {w} waist, runs it under {p} crotch" if was else
            f"takes a separate crotch {u} from {dp} pocket, wraps its waist {u} tight around {w} waist, runs its center {u} from the "
            f"front of the waist {u} under {p} crotch") + f", centers it tight over {p} {'vagina' if p == 'her' else 'crotch'}, " \
        f"brings it up between {p} buttocks to the small of {p} back and " \
        f"{close.format(t=f' to the back of the waist {u}')}" if crotch else \
        f"wraps the {u} tight around {w} waist and {close.format(t='')}"
    return lead + (f", then {_linking(x, p, p)}" if link and not (was and _lashed(was)) else "")


def _dress_step(kind, g, w, p, nm="", sent="", hobbled=False, hands="", pose=None):
    if kind == "wear":
        return f"puts {g} on {nm}" if nm else f"puts on {g}"
    if kind == "lift":
        done = f"{'pulls' if nm or not hands else 'pushes'} {w} {g[0]} {g[1]}"
    else:
        cut = "cuts" if re.search(r"\b(?:cut|snip|slic|scissors|knife|blade)", sent, re.I) else "rips"
        _, _, step, bound, it = *_garb(g), "them" if _many([g]) else "it"
        done = {"back": f"pulls {w} {g} back into place", "tear": f"{cut} {w} {g} open and pulls the pieces away"}.get(kind) or \
            (hobbled and bound or step).format(w=w, p=p, g=g, it=it)
    if not hands or kind != "remove":
        return done
    off = re.compile(rf"\s+and\s+off(?:\s+over)?\s+{p}\s+feet$")
    if not nm:
        done = f"pushes {w} {g} down over {p} hips and wriggles {it} down {p} legs and off over {p} feet" if off.search(done) else \
            f"pushes {w} {g} off with {p} feet, one heel against the other" if re.search(rf"\b(?:{_SHOES})$", g, re.I) else done
    return off.sub(f" to {p} ankles, and {nm + ' ' if nm else ''}steps out of {it} one foot at a time", done) \
        if pose == "stand" else done


def _doing(changes, people, call, held=None, stance=None):
    held, stance, seq, rolls, stuffed, total = held or {}, stance or {}, [], {}, [], 0.0
    laid = {n for n, v in held.items() if any((_tied(x) or {}).get("lies") for x in v)}
    links, on = {n: [x for x in v if _link(x)] for n, v in held.items()}, {n: list(v) for n, v in held.items()}
    for kind, who, x, doer, sent in changes:
        feet = _ankle_item(on.get(who, [])) if kind == "hold" and _band(x) and "ankles" not in parts(x) else None
        on[who] = on.get(who, []) + [x] if kind == "hold" else [y for y in on.get(who, []) if y != x]
        was = _swapped(x, held.get(who, [])) if kind == "hold" else None
        cut = [y for y in links.get(who, []) if kind == "release" and (y == x or "wrists" in parts(x))]
        links[who] = [y for y in links.get(who, []) if y not in cut and y != was] + ([x] if kind == "hold" and _link(x) else [])
        total += _takes(kind, x if isinstance(x, str) else x[0], held.get(who, []))
        mine, tape, held_by = doer == who, kind == "hold" and "duct tape" in _gears(x), []
        if doer is None:
            if tape:
                rolls.setdefault((None, who), False)
            continue
        actor, p = who if mine else doer, pron(people.get(who, {}))
        w = p if mine else f"{call(who)}'s"
        pose = "lie" if who in laid else stance.get(who)
        low = _stoop(x, w, call(who), mine, pose, stance.get(actor) == "kneel") if kind in ("hold", "release") else ""
        q = w if not low else p if mine or pron(people.get(actor, {})) != p else f"{who}'s" if who in people else w
        bend = lambda s: s if not low else f"{low}, {s}" if " and " in s else f"{low} and {s}"
        if tape:
            step, rolled = _taped(x, q, p, sent, pose == "lie")
            step = bend(step)
            rolls[(actor, who)] = rolls.get((actor, who), False) or rolled
        elif kind in ("hold", "release") and _band(x) and (_crotch(x) or _lashed(x) or was):
            step = _harness_step(x, w, p, kind == "hold", was, x in cut, pron(people.get(actor, {})))
            down = _low(feet, p if mine or pron(people.get(actor, {})) != p else w) if feet and not was else ""
            step = ", ".join(([f"stands up behind {call(who)}"] if down and stance.get(actor) == "kneel"
                              and stance.get(who) in (None, "stand") else []) + [step]) + (
                f", while {down} " + ("stays just as it is" if down.startswith("the ") else "stay just as they are") if down else "")
        elif was:
            step = _linking(x if _link(x) else was, w, p, bool(_link(x)))
        elif kind in ("hold", "release"):
            step, done = _bind_step(x, call(who), p if cut else q, p, pron(people.get(actor, {})), kind == "hold", who in laid,
                                    pose == "lie")
            step = ", then ".join([_linking(y, w, p, False) for y in cut] + [step if cut else bend(step)]
                                  + ([_linking(x, p, p)] if kind == "hold" and _link(x) else []))
            laid |= {who} if done else set()
            stuffed += [actor] if kind == "hold" and " stuffed in " in x and actor not in stuffed else []
        else:
            cuffs = next((y for y in on[who] if "wrists" in parts(y)), None)
            step = _dress_step(kind, x, w, p, "" if mine else call(who), sent, bool(_ankle_item(on[who])),
                               _held_at(cuffs, p) if cuffs else "", stance.get(who))
            held_by = [f"with {p if mine else call(who) + chr(39) + 's'} hands still {_held_at(cuffs, p)}"] if cuffs and kind != "tear" \
                else []
        if not (seq and seq[-1][0] == actor):
            seq.append((actor, [], [], []))
        seq[-1][1].append(step)
        seq[-1][2].extend(held_by)
        seq[-1][3].append(bool(low) and step.startswith(low))
    notes = [f"{cap(call(a))} " + ", then ".join(steps + (["lets go and stands back up"] if downs[-1] else []))
             + "".join(f", {h}" for h in dict.fromkeys(hs)) + "." for a, steps, hs, downs in seq]
    notes += [f"The roll of duct tape stays whole and outside {call(n)}'s body the whole time, and the tape lies flat where it goes."
              if a is None else
              f"The roll stays whole in {a}'s hand and outside {n}'s body the whole time; when the tape is on, {a} tears it off the roll "
              f"and puts the roll down." if rolled else
              f"{cap(call(a))} keeps the roll whole in {pron(people.get(a, {}))} other hand, outside {n}'s body."
              for (a, n), rolled in rolls.items()]
    notes += [f"{cap(call(a))}'s own mouth stays empty." for a in stuffed]
    gear = any(kind in ("hold", "release") for kind, *_ in changes) or any(held.values())
    notes += [f"It all happens at natural, real-life speed, one step after another, over about {max(1, round(total))} seconds: no "
              f"step is skipped, sped up or cut short, and no {'restraint or ' * gear}piece of clothing appears, vanishes or changes "
              f"by itself."] if total else []
    return notes


def _said(events, people, held=None):
    out = []
    for kind, arg, _, doer in events:
        if kind == "have":
            continue
        who, items = split_tag(arg, people, _GARB.get(kind, False))
        p, mine, does, bound = pron(people.get(who, {})), doer == who, [], kind in ("hold", "release")
        d, w = (who, p) if mine else (doer, f"{who}'s")
        them = {"her": "herself", "his": "himself"}.get(p, "themselves") if mine else who
        for x in items if bound else [join(items)] if items else []:
            spot = (sorted(_places(x)) or ["wrists"])[0] if bound else ""
            gear, cloth = _gear(x), " stuffed in " in x or " tied over " in x
            was, link, unit = _swapped(x, (held or {}).get(who, [])) if kind == "hold" else None, bound and _link(x), _unit(gear)
            tie = (" and " if _band(x) else ", then ") + _linking(x, p, p) if link else ""
            thing = "the " + re.sub(r"^(?:a|an|the)\s+", "", re.split(r"\s+(?:stuffed\s+in|tied\s+over)\s+", x)[0] if cloth else gear)
            does.append(f"runs a {unit} under {w} crotch{tie}" if was and _crotch(x) and not _crotch(was) else
                        _linking(x if link else was, w, p, bool(link)) if was else
                        f"puts a crotch {unit} on {them}{tie}" if kind == "hold" and _crotch(x) else
                        f"takes the crotch {unit} off {them}" if kind == "release" and _crotch(x) else
                        f"stuffs {thing} into {w} mouth" if kind == "hold" and " stuffed in " in x else
                        f"ties {thing} over {w} {spot}" if kind == "hold" and cloth else
                        f"cuffs {w} wrists {x.split(' ', 1)[1]}" if kind == "hold" and x.startswith("handcuffs ") and not _spots(x) else
                        f"{'cuffs' if gear == 'handcuffs' else _TIES.get(gear, ('', 'fastens'))[1]} {w} {spot} together{tie}"
                        if kind == "hold" and spot in ("ankles", "knees", "legs") and gear != "a spreader bar" and not _tied(x) else
                        "puts " + re.sub(r"\b(?:her|his|their)\b", w, _TIED_TO.sub("", x) if link else x, count=1) + tie
                        if kind == "hold" else
                        f"takes {thing} off {w} {spot}" if kind == "release" else f"takes off {w} {x}" if kind == "remove" else
                        f"tears {w} {x} off" if kind == "tear" else f"pulls {w} {x}" if kind == "lift" else
                        f"puts on {x}" if mine else f"puts {x} on {who}")
        cuffs = None if bound or kind == "tear" else next((y for y in (held or {}).get(who, []) if "wrists" in parts(y)), None)
        tail = f", with {p if mine else w} hands still {re.sub(r' of (?:her|him|them)$', '', _held_at(cuffs, p))}" if cuffs else ""
        out += [f"{cap(d)} " + ", then ".join(does) + tail + "."] if does else []
    return " ".join(out)


def _merged(atoms):
    out = []
    for k, who, x, sent, doer in atoms:
        if out and out[-1][:2] == [k, who] and out[-1][3] == doer:
            out[-1][2].append(x)
        else:
            out.append([k, who, [x], doer, sent])
    return [(k, f"{who}, {('; ' if k in ('hold', 'have', 'release') else ', ').join(xs)}", sent, doer) for k, who, xs, doer, sent in out]


def _pace(text, events, took, slow=None):
    told, out = narration(text), []
    spans = [(a, b, told[a:b].strip()) for a, b in _sentences(text)]
    home = [next((k for k, (*_, s) in enumerate(spans) if e[2] == s), None) for e in events]
    bare = [k for k in range(len(spans)) if k not in home] or [len(spans) - 1]
    home = [bare[min(sum(h is None for h in home[:j]), len(bare) - 1)] if h is None else h for j, h in enumerate(home)]
    for k, (a, b, s) in enumerate(spans):
        busy = [t for e, h in zip(events, home) if h == k for t in took(e[0], e[1]) if t]
        talk = sum(SAY_SLACK + len(m.group(0).split()) / WORDS_PER_SEC + CUE_GAP for m in _SPEECH.finditer(text[a:b]))
        s = _HANDS_ON.sub("", s)
        cs = [c for c in _CLAUSE.split(s) if c and len(c.split()) >= 2]
        tied = lambda c: bool(_DOES.search(c) or _ROUTES.search(c))
        more = sum(tied(c) and tied(d) and bool(re.search(r"\b(?:it|them)\b", d, re.I)) and not re.search(r"(?-i:[A-Z])", d)
                   and not _PLACED.search(d) and not _PUTAWAY.search(d) for c, d in zip(cs, cs[1:]))
        out.append((a, b, sum(busy) + PER_ACTION * max(0, actions(s) - more - len(busy)) + (slow(s) if slow else 0), talk))
    return out


def _rest(text, done, keep, people):
    seps = list(_CLAUSE.finditer(text))
    spans = list(zip([0] + [m.end() for m in seps], [m.start() for m in seps] + [len(text)]))
    said = lambda k: text[spans[k][0]:spans[k][1]]
    about = lambda c, t: bool(set(_spots(c)) & _places(t[2]) if _spots(c) else _gears(c) & _gears(t[2])) if t[0] in ("hold", "release") \
        else bool(re.search(rf"\b{re.escape(_stem(pulled(t[2], {})[0] if t[0] == 'lift' else t[2]))}", c, re.I))
    kept = lambda c: any(about(c, t) for t in keep)
    more = lambda c: bool(re.search(r"\b(?:it|them)\b", c, re.I) and (_DOES.search(c) or _ROUTES.search(c))) and not re.search(
        r"(?-i:[A-Z])", _HANDS_ON.sub("", c))
    gone, left = [], list(done)
    for k in range(len(spans)):
        hit = [] if kept(said(k)) else [t for t in left if about(said(k), t)]
        if hit or gone and gone[-1] == k - 1 and more(said(k)) and not kept(said(k)):
            gone.append(k)
            left = [t for t in left if t not in hit]
    stay, (who, doer) = [k for k in range(len(spans)) if k not in gone], (keep[0][1], keep[0][4]) if keep else (None, None)
    if left or not gone or not stay or not doer or _SPEECH.search(text) or stay[0] and re.match(r"\w+ing\b", said(stay[0])):
        return text
    end = re.search(r"[.!?…]+[\"”’]*$", text.strip())
    bare = stay[0] and re.match(r"\W*(?:her|his|their|the|both|(?-i:[A-Z])[\w-]*['’]s)\s", said(stay[0]), re.I)
    first = _said(_merged(keep), people).rstrip(".") + (end.group(0) if end and stay[0] == len(spans) - 1 else "") if bare else ""
    rest = "".join((text[spans[k - 1][1]:spans[k][0]] if j else "") + (first or said(k) if not j else said(k))
                   for j, k in enumerate(stay)).strip()
    rest += "" if stay[-1] == len(spans) - 1 else end.group(0) if end else "."
    g = _gear(keep[0][2])
    g = re.sub(r"^an?\s+", "", g) if g.startswith("a ") or g in ("handcuffs", "shackles", "duct tape") else _unit(g)
    rest = re.sub(r"\b(?:it|them)\b", f"the {g}", rest, count=1) if stay[0] and re.match(r"\W*(?:\w+ly\s+)?\w+\s+(?:it|them)\b", rest,
                                                                                          re.I) else rest
    lead = "" if not stay[0] or re.match(r"(?-i:[A-Z])|(?:she|he|they)\b", rest) else f"{cap(doer)} "
    p = pron(people.get(who, {}))
    m = re.search(rf"\b(?:{p}|{_OBJ.get(p, p)})\b", rest) if re.search(rf"(?-i:\b{re.escape(who)})\b", text) and \
        not re.search(rf"(?-i:\b{re.escape(who)})\b", rest) else None
    own = m and m.group(0) == p and not re.match(_OBJECT, rest[m.end():])
    return lead + (rest[:m.start()] + (f"{who}'s" if own else who) + rest[m.end():] if m else rest)


def _parts(text, read, room, people, took, garb, held=None, slow=None):
    told, units = narration(text), []
    fresh = lambda k, who, x: k == "hold" and _hogged(x) and _TIED_TO.sub("", x) not in (held or {}).get(who, [])
    for a, b, secs, talk in _pace(text, read, took, slow):
        evs = [e for e in read if e[2] == told[a:b].strip()]
        atoms = [(k, who, y, sent, doer) for k, arg, sent, doer in evs for who, items in [split_tag(arg, people, _GARB.get(k, False))]
                 for x in items or (garb(who) if k == "remove" else [])
                 for y in ([_TIED_TO.sub("", x), x] if fresh(k, who, x) else [x])]
        span = lambda t: _takes(t[0], t[2], (held or {}).get(t[1], []))
        free = [t for t in atoms if t[4] and span(t)]
        hog = {t[2] for t in free if fresh(*t[:3])}
        if not hog and (secs + talk <= room or len(free) < 2):
            units.append((text[a:b], evs, secs + talk, False))
            continue
        chunks, used = [[t for t in atoms if t[0] == "have"]], 0.0
        for t in free:
            if used and (used + span(t) > room or t[2] in hog):
                chunks, used = chunks + [[]], 0.0
            chunks[-1].append(t)
            used += span(t)
        last = [t for t in atoms if t not in free and t[0] != "have" or t in chunks[-1]]
        left, said = secs + talk - sum(span(t) for t in atoms if t not in last), \
            _rest(text[a:b], [t for c in chunks[:-1] for t in c if t[0] != "have"], last, people)
        said = re.sub(rf"\s+with\s+(?:another|a\s+second|a|an|the|her|his|their)\s+(?:pair\s+of\s+|set\s+of\s+)?(?:{_KIT})\b", "", said,
                      flags=re.I) if hog else said
        units += [(_said(_merged(c), people, held), _merged(c), sum(map(span, c)), False) for c in chunks[:-1]]
        units.append((said, _merged([t[:3] + (said.strip(), t[4]) for t in last] if said != text[a:b] else last), left, bool(hog)))
    if sum(u[2] for u in units) <= room * RUSH and not any(u[3] for u in units):
        return []
    parts, used = [[]], 0.0
    for u in units:
        if parts[-1] and (used + u[2] > room or u[3]):
            parts, used = parts + [[]], 0.0
        parts[-1].append(u)
        used += u[2]
    return [(" ".join(u[0].strip() for u in part), [e for u in part for e in u[1]]) for part in parts] if len(parts) > 1 else []


def _footwork(text, n, cast, people):
    sex = lambda m: pron(people.get(m, {}), 0)
    sole, o, p = [sex(m) for m in cast].count(sex(n)) == 1 and sex(n) != "they", pron(people.get(n, {}), 2), pron(people.get(n, {}))
    names = "|".join(re.escape(m) for m in sorted(cast, key=len, reverse=True)) or "(?!)"
    anyone = rf"(?:[\w-]+-haired\s+)?(?-i:(?:{names}))\b(?!['’])"
    whose = rf"both|one|either|each|her|his|their|(?:[\w-]+-haired\s+)?(?-i:(?:{names}))(?:['’]s|(?=\s+by\s+(?:the|both)\s))"
    limbs = re.compile(rf"(?<![\w'’-])(?!(?:un|re)?(?:{_KIT})\b)(?P<verb>[a-z]+(?:s|ed|ing)|{_GO}|held|kept|thrust)\s+"
                       rf"(?:(?:up|out|back|down|in|together|apart|free|both|of|\w+ly)\s+){{0,3}}(?P<w>{whose})\s+(?:[\w-]+\s+){{0,2}}?"
                       rf"(?:{_LIMB})\b|"
                       rf"\bwith\s+(?P<u>{whose})\s+(?:[\w-]+\s+){{0,2}}?(?:{_LIMB})\b(?!\s+(?:\w+ly\s+)?(?:\w+-)?(?:{_KEPT}|together|"
                       rf"apart|side|bent|straight|tucked|drawn|pressed|spread|stretched|curled|folded|up|out|wide|in|still|flat|bare|"
                       rf"free|dangling|hanging|resting|planted)\b)|"
                       rf"(?<!\bwith\s)(?<!\baround\s)(?<!\bon\s)(?<!\bover\s)(?<!\bunder\s)(?<!\bbetween\s)(?<!\bat\s)(?<!\bof\s)"
                       rf"(?<!\bto\s)\b(?P<v>{whose})\s+(?:[\w-]+\s+){{0,2}}?(?:{_LIMB})\s+(?:\w+ly\s+)?(?!(?:\w+-)?(?:{_KEPT})\b)"
                       rf"(?:{_GO}|(?:are|is|were|was|gets?|got)\s+(?:\w+ly\s+)?(?!(?:\w+-)?(?:{_KEPT})\b)[a-z]+ed)\b", re.I)
    mine = lambda w, pro: bool(re.fullmatch(rf"(?:[\w-]+-haired\s+)?(?-i:{re.escape(n)})", w) or sole and w.lower() == pro
                               or w.lower() == "they" and len(cast) < 3)
    by_ = lambda w, at: mine(w, o) or w.lower() == o and [m for m in cast if pron(people.get(m, {}), 2) == o and m not in [
        h[3] for h in subj if h[0] < at][-1:]] == [n]
    heads = (re.compile(rf"(?:^|[,;:—–]|\b(?:and|then|but|while|as|before|after|when|until|so|once|where|now))\W*"
                        rf"(?:(?:{_LEADS}|\w+ly)\W+)*(?P<w>{anyone}|she\b|he\b|they\b)", re.I),
             re.compile(rf"(?P<w>{anyone})(?=\s+(?:\w+ly\s+)?(?!(?:towards|always|afterwards|backwards|forwards|sideways|upstairs|"
                        rf"downstairs|perhaps|unless|besides)\b)(?:[a-z]+(?:s|ed)|went|came|ran|knelt|sat|stood|lay|got)\b)", re.I))
    make = re.compile(rf"\b(?:{_MAKE})\s+(?P<w>{anyone}|her\b|him\b|them\b)\s+(?:to\s+)?(?:\w+ly\s+)?$", re.I)
    led = re.compile(rf"\b(?P<v>{_LED}|sits?|sat|seats?|seated)\s+(?P<w>{anyone}|her\b|him\b|them\b)\s+(?:\w+ly\s+)?"
                     rf"(?:(?P<f>face[\s-]?(?:down|up)|facedown)\b\s*|face[\s-]?first\s+)?(?:(?:down\s+|over\s+)?(?:on|onto|to)\s+"
                     rf"(?:her|his|their)\s+(?P<k>knees|feet|side|back|front|stomach|belly)\b|(?P<d>{_DIRS}|on|onto)\b(?P<to>.{{0,40}})|"
                     rf"(?(f)|(?!)))", re.I)
    tried = re.compile(r"\b(?:tr(?:y|ies|ied|ying)|attempts?|attempted|struggles?|struggled|wants?|wanted|fights?)\s+to\s+(?:\w+ly\s+)?$|"
                       r"\b(?:can(?:no|['’])?t|cannot|unable\s+to)\s+(?:\w+ly\s+)?$", re.I)
    out, plain = [], narration(text)
    for a, b in _sentences(text):
        sent, subj = plain[a:b], []
        for m in (m for rx in heads for m in rx.finditer(sent)):
            pair = re.search(rf"({anyone})\s+and\s+$", sent[:m.start("w")])
            subj.append((m.start("w"), mine(m.group("w"), sex(n)) or bool(pair) and mine(pair.group(1), sex(n)),
                         bool(pair) or m.group("w").lower() == "they", re.sub(r"^[\w-]+-haired\s+", "", m.group("w"))))
        subj.sort()
        found, taken = [], []
        for kind, rx in _FOOT + (("wide", _WIDE.pattern),):
            for m in re.finditer(rf"\b(?:{rx})\b", sent, re.I):
                if kind != "wide" and any(a < m.end() and m.start() < b for a, b in taken):
                    continue
                by, first, last = make.search(sent[:m.start()]), m.group(0).split()[0].lower(), [h for h in subj if h[0] < m.start()][-1:]
                bare = not re.search(r"(?:s|ed|ing)$", first) and first not in (
                    "went", "came", "ran", "knelt", "sat", "stood", "crept", "snuck", "led", "lay", "laid", "fell", "sank", "slept", "got",
                    "on", "onto", "to", "her", "his",
                    "their", "cross-legged")
                own = by_(by.group("w"), m.start()) if by else bool(last) and last[0][1] and not (bare and not last[0][2] and not re.search(
                    r"\bto\s+(?:\w+ly\s+)?$", sent[:m.start()], re.I))
                run = kind == "wide" and re.match(r"run|ran|sprint|dash|bolt|jog|rac", first)
                tags = (["walk"] if run else []) + ([] if kind == "wide" and tried.search(sent[:m.start()]) else [kind])
                for tag in tags if own else ():
                    found.append((m.start(), m.end(), "stairs" if tag == "walk" and (first.startswith(("climb", "descend")) or re.search(
                        rf"\b(?:up|down)\s+{_FLIGHT}", sent[m.start():m.end() + 40], re.I)) else tag))
                taken += [(m.start(), m.end())] if own else []
        for m in led.finditer(sent):
            if by_(m.group("w"), m.start()) and not any(a < m.end() and m.start() < b for a, b in taken):
                to = m.group("to") or ""
                kind = "lie" if m.group("f") else {"knees": "kneel", "feet": "stand"}.get((m.group("k") or "").lower(),
                                                                                          "lie" if m.group("k") else None) or (
                    "sit" if re.search(r"\b(?:(?:arm)?chair|stool|bench|couch|sofa|seat|lap)\b", to, re.I) else
                    "lie" if re.search(r"\b(?:bed|floor|ground|mattress|cot|rug|carpet)\b", to, re.I) else
                    "stairs" if re.match(rf"(?:up|down)\s+{_FLIGHT}", f"{m.group('d')}{to}", re.I) else
                    "walk" if re.match(r"lead|led|walk|march|frogmarch|escort|guid|steer|herd|take|took|bring|brought|help|send|sent",
                                       m.group("v"), re.I) or re.fullmatch(r"to|toward|towards|into|across|through|along|out|away|forwards?|"
                                                                           r"past|inside|outside|ahead", m.group("d"), re.I) else None)
                found += [(m.start(), m.end("d") if m.group("d") else m.end(), kind)] if kind else []
        for m in limbs.finditer(sent):
            w, last = re.sub(r"['’]s$", "", m.group("w") or m.group("u") or m.group("v")), [h for h in subj if h[0] < m.start()][-1:]
            other = [q for q in cast if pron(people.get(q, {})) == w.lower() and q not in [h[3] for h in last]] == [n] and bool(last) \
                and m.group("w") and not re.match(_SELF, m.group("verb"), re.I)
            posed = m.group("u") and re.search(r"\b(?:sits?|sitting|sat|seated|lies|lying|lay|stands?|standing|stood|kneels?|kneeling|"
                                               r"knelt|leans?|leaning|waits?|waiting|rests?|resting|is|are|was|were|stays?|remains?|"
                                               r"hangs?|hanging|hung|slump(?:s|ed)?|crouch(?:es|ed|ing)?|squat(?:s|ting)?)\b",
                                               re.split(r"[,;]|\b(?:and|then|but|while|as|before|after|when)\b", sent[:m.start()])[-1],
                                               re.I) and not re.match(rf"\s+(?:\w+ly\s+)?(?:{_GO})", sent[m.end():], re.I)
            if (mine(w, p) or w.lower() in ("both", "one", "either", "each", p) and bool(last) and last[0][1] or other) and not posed \
                    and not any(x < m.end() and m.start() < y for x, y in taken):
                found.append((m.start(), m.end(), "legs"))
        out += [(k, sent, a + s, a + e) for s, e, k in sorted(found)]
    return out


def _stance(found):
    return next((_STEPS[k] for k, *_ in reversed(found) if k in _STEPS), None)


def _stoop(x, w, nm, mine, pose, kneels=False):
    if "ankles" not in parts(x) or (_tied(x) or {}).get("lies"):
        return ""
    low = "kneels" if kneels or pose == "lie" else "crouches"
    return "bends down" if mine else f"{low} behind {nm}" if pose == "kneel" else f"{low} at {w} feet"


def _ankle_item(items):
    ankles = [x for x in items if "ankles" in parts(x)]
    return next((x for x in ankles if _tied(x)), None) or next((x for x in ankles if "ankles" in _places(x)), ankles[0] if ankles else None)


def _feet_look(x, per):
    p, g, t, one, link = pron(per), _gear(x), _tied(x), re.search(r"\b(?:left|right) ankle\b", x), _link(x)
    spot = one.group(0) if one else next((q for q in ("knees", "legs") if q in _places(x)), "ankles")
    done = _TIES.get(g, ("fastened",))[0]
    if t:
        return f"{cap(p)} {spot} {'is' if one else 'are'} {done} to {t['to']}."
    held = {"handcuffs": "handcuffed together", "shackles": "shackled together by their own heavy chain about forty centimetres long",
            "a spreader bar": "locked to the two ends of a spreader bar"}.get(g, f"{done} together" + {
                "rope": " with rope", "duct tape": " with duct tape", "tape": " with tape", "zip ties": " with zip ties"}.get(g, ""))
    tail = f" and held to {link[0]} by a separate short {link[1]}, {_drawn(x, p, pron(per, 2))}" if link else \
        f" that holds {p} feet apart" if g == "a spreader bar" else ""
    return f"{cap(p)} {'ankles' if g in ('handcuffs', 'shackles', 'a spreader bar') else spot} are {held}{tail}."


def _low(x, p):
    g = _gear(x)
    return f"the spreader bar between {p} ankles" if g == "a spreader bar" else f"{p} feet" if "ankles" not in _places(x) else \
        f"{p} {'handcuffed' if g == 'handcuffs' else 'shackled' if g == 'shackles' else _TIES.get(g, ('bound',))[0]} ankles"


def _drawn(x, p, o, stay=False):
    bent, up = ("stay bent", "stay raised together") if stay else ("bent", "raised together")
    return f"{p} knees {bent} and {_low(x, p)} {up} " + (f"toward {p} hands" if "in front of" in _link(x)[0] else
                                                      f"behind {o}, close to {p} hands")


def _since(text, sent):
    words, spans = set(re.findall(r"\w+", sent.lower())), _sentences(text)
    return max(spans, key=lambda s: len(words & set(re.findall(r"\w+", text[s[0]:s[1]].lower()))))[1] if words and spans else len(text)


def _legs(text, gait, hobbled, call, cast, people, before):
    plain, spans, subs, ends, swaps = narration(text), _sentences(text), {}, {}, []
    limb = re.compile(rf"(?P<both>\bboth\s+(?:of\s+)?)?(?P<q>\b(?:one|either|each|both|her|his|their)\b|(?:[\w-]+-haired\s+)?"
                      rf"[^\s,;:.!?]+['’]s|(?:[\w-]+-haired\s+)?\S+(?=\s+by\s+(?:the|both)\s))\s+(?P<mid>(?:by\s+(?:the|both)\s+)?"
                      rf"(?:(?:left|right|other|bare|bound|cuffed|handcuffed|tied|taped|shackled|chained|long|own|two)\s+){{0,2}})"
                      rf"(?P<l>{_LIMB})\b(?P<pt>{_PARTICLE.pattern})", re.I)
    more = re.compile(rf"\s+(?:and|or)\s+(?:(?:her|his|their)\s+)?(?:{_LIMB}|arms?|wrists?|hands?|hips?|waist|shoulders?|head|hair|"
                      rf"neck|body|torso|elbows?)\b", re.I)
    claims, tight = {}, lambda n, at: n in hobbled and at >= hobbled[n][1]
    for n in cast:
        for k, _, at, end in gait.get(n, []):
            claims.setdefault(at, []).append(n)
    for n in [c for c in cast if c in hobbled]:
        per, x = people.get(n, {}), hobbled[n][0]
        p, g = pron(per), _gear(x)
        whose = f"{call(n)}'s" if [pron(people.get(m, {})) for m in cast].count(p) > 1 else p
        kind = "bar" if g == "a spreader bar" else "irons" if g == "shackles" else "cuffs" if g == "handcuffs" else "tied"
        spot = next((q for q in ("knees", "legs") if q in _places(x)), "ankles")
        tie = {"cuffs": "{w} ankles handcuffed together", "irons": "{w} ankles joined by the heavy chain of the shackles",
               "bar": "{w} ankles locked to the two ends of the spreader bar"}.get(
            kind, f"{{w}} {spot} {_TIES.get(g, ('fastened',))[0]} together")
        lead = {"walk": {"bar": "waddling with both legs held wide apart", "irons": "in short, hobbled steps"}.get(
                    kind, "in tiny shuffling steps"),
                "stairs": "one stair at a time" + ("" if kind in ("bar", "irons") else ", both feet together on each stair")}
        mine = re.compile(rf"(?:one|either|each|both|{p}|{pron(per, 2)}|(?:[\w-]+-haired\s+)?(?-i:{re.escape(n)})(?:['’]s)?)", re.I)
        done, was = {}, before.get(n)
        for k, _, at, end in gait.get(n, []):
            turn, was = _STEPS.get(k) not in (None, was) and was is not None, _STEPS.get(k) or was
            if not tight(n, at):
                continue
            a, b = next(s for s in spans if s[0] <= at < s[1])
            m = limb.search(plain, at) if k == "legs" else limb.match(plain, end + 1) if k == "kick" else None
            m = m if m and (k == "kick" or m.end("l") <= end) else None
            if m and m.group("q").lower() in ("her", "his") and any(tight(c, at) for c in claims[at] if c != n):
                continue
            listed = more.match(plain, m.end("l")) if m else None
            cut = _UNTIL.search(plain, listed.end() if listed else end, b)
            to = cut.start() if cut else b
            form = "v" if m and m.start() == at else "with" if m and plain.startswith("with", at) else "verb"
            verb = re.findall(r"[a-z]+", plain[at:end].lower())[-1 if form == "v" else 0] if m else ""
            act = bool(m) and (form == "with" or bool(re.match(rf"(?:{_GO}|{_GRIP})", verb))) and not listed \
                and not re.search(r"(?:\band|,)\s*$", plain[at:m.start()])
            word, q = (m.group("l").lower(), m.group("q")) if act else ("", "")
            by = act and m.group("mid").lower().startswith("by")
            held = by or act and bool(re.match(rf"(?:{_GRIP})", verb))
            single = act and word in _PAIR.values() and kind != "irons" and form != "v" and bool(mine.fullmatch(q)) and not (
                held and q.lower() in ("one", "either", "each", "both"))
            if single or act and word in _PAIR and q.lower() == p and whose != p:
                q = q[:1].lower() + q[1:] if re.match(r"(?:one|either|each|both|her|his|their)\b|[\w-]+-haired\s", q, re.I) else q
                adj = re.sub(r"\b(?:by\s+(?:the|both)|left|right|other|two|long|own)\s+", "", m.group("mid"), flags=re.I)
                own = q if q not in ("one", "either", "each", "both", p) else p if whose == p or re.search(
                    rf"(?-i:\b{re.escape(n)})\b", plain[a:m.start()]) else whose
                pair = word if word in _PAIR else next(k for k, v in _PAIR.items() if v == word)
                new = f"{q} by {p} {adj}{pair}" if by else f"{own} {adj}{pair}"
                subs[(m.start(), m.end("l"), 0)] = cap(new) if not plain[a:m.start()].strip() else new
                swaps += [(n, text[m.start():m.end("l")], subs[(m.start(), m.end("l"), 0)])] if single else []
            if act and kind != "bar" and (single or word in _PAIR) and not held and not re.match(r"rub|squeez|tap|strok|massag", verb) \
                    and not re.search(r"\b(?:together|apart)\b", plain[at:to], re.I):
                after = _PARTICLE.match(plain, end).end() if form == "v" else m.end()
                subs[(after, after, 0)] = " together"
            fam = _STRIDES.get(k) or ("move" if turn or _RISE.match(plain, at) else None)
            if fam and fam not in done.get(a, ()):
                said = [y for y in (lead.get(fam, ""), "" if a in done or _BINDS.search(plain, at, to) else tie) if y]
                done.setdefault(a, set()).add(fam)
                if said:
                    ends.setdefault(to, []).append((n, whose, p, said, bool(cut and cut.group("conj"))))
    for to, here in ends.items():
        bits, many = [], len({h[0] for h in here}) > 1
        for n, whose, p, said, _ in here:
            first = next((j for j, y in enumerate(said) if "{w}" in y), None)
            bits += [y.format(w=whose if many or j == first else p, p=p) for j, y in enumerate(said)]
        subs[(to, to, 1)] = ", " + ", ".join(dict.fromkeys(bits)) + ("," if here[0][4] else "")
    for s, e, r in sorted(subs, key=lambda k: (k[0], k[2]), reverse=True):
        text = text[:s] + subs[(s, e, r)] + text[e:]
    return text, list(dict.fromkeys(swaps))


def _ankles(name, x, w, per, did, pose):
    p, o, s = pron(per), pron(per, 2), pron(per, 0)
    v = lambda word: word if s != "they" else {"does": "do"}.get(word, word[:-1])
    g, t, S, one = _gear(x), _tied(x), cap(s), re.search(r"\b(?:left|right) ankle\b", x)
    spot, done, many = next((q for q in ("knees", "legs") if q in _places(x)), "ankles"), _TIES.get(g, ("fastened",))[0], _many([g])
    if t:
        seat = re.search(r"\b((?:arm)?chair|stool)s?\b", t["to"])
        return [f"{name}'s {one.group(0)} stays {done} to {t['to']} {w}, and {p} other leg moves freely: {s} {v('moves')} as far as the "
                f"{g} {'reach, and they stay' if many else 'reaches, and it stays'} locked around {p} {one.group(0)}." if one else
                f"{name} sits on the {seat.group(1)}, {p} {spot} {done} to {t['to']} {w}: {p} knees bend over the front of the seat and {p} "
                f"lower legs run straight down, {p} ankles held against {t['to']}." if seat else
                f"{name}'s {spot} stay {done} to {t['to']} {w}: {p} feet stay right there, held to it."]
    kind = "bar" if g == "a spreader bar" else "irons" if g == "shackles" else "cuffs" if g == "handcuffs" else "tied"
    does = v("does")
    low = f" just above {p} feet" if spot == "ankles" else ""
    wound = {"rope": "tied", "zip ties": "pulled", "tape": "wound", "duct tape": "wound", "chains": "wrapped and locked",
             "straps": "buckled"}.get(g, "fastened")
    out = [{"cuffs": f"{name}'s ankles stay handcuffed together {w}: a pair of rigid steel cuffs is locked around them{low}, joined by "
                     f"their own short chain of a few links that never stretches, bends or gets longer, so {p} ankles stay a few "
                     f"centimetres apart at most and {p} legs stay pressed together and always move together, both at once.",
            "irons": f"{name}'s ankles stay shackled together {w}: a pair of steel shackles is locked around them{low}, joined by their "
                     f"own heavy chain, about forty centimetres long, that never stretches or gets longer, so whatever {s} {does}, {p} "
                     f"feet stay within that chain's length of each other.",
            "bar": f"{name}'s ankles stay locked to the two ends of a rigid steel spreader bar {w}, so whatever {s} {does}, {p} feet stay "
                   f"the bar's length apart, about sixty centimetres, and {p} legs move together with the bar.",
            "tied": f"{name}'s {spot} stay {done} together {w}: the {g} {'are' if many else 'is'} {wound} tight around them{low}, so "
                    f"{p} legs stay pressed together and always move together, both at once."}[kind]]
    k = max((j for j, q in enumerate(did) if q in ("kneel", "sit", "lie", "stand")), default=-1)
    rest = None if set(_STRIDES) & set(did[k + 1:]) else did[k] if k >= 0 else pose
    stay = {"irons": {"kneel": f"{S} {v('kneels')} with {p} feet behind {o}, the chain between {p} ankles lying on the floor.",
                      "sit": f"{S} {v('sits')} with {p} feet in front of {o}, the chain slack between {p} ankles.",
                      "lie": f"{S} {v('lies')} with {p} legs close together, the chain between {p} ankles.",
                      "stand": f"{S} {v('stands')} with {p} feet a little apart, the chain between {p} ankles slack on the floor."},
            "bar": {"kneel": f"{S} {v('kneels')} with {p} knees wide apart, the bar lying across the floor behind {o} between {p} ankles.",
                    "sit": f"{S} {v('sits')} with {p} legs held wide apart by the bar.",
                    "lie": f"{S} {v('lies')} with {p} legs spread apart, the bar between {p} ankles.",
                    "stand": f"{S} {v('stands')} with {p} feet wide apart, the bar between {p} ankles just above the floor."}}.get(kind, {
        "kneel": f"{S} {v('kneels')} with {p} knees together and {p} feet behind {o}.",
        "sit": f"{S} {v('sits')} with {p} knees and feet together.",
        "lie": f"{S} {v('lies')} with {p} legs together.",
        "stand": f"{S} {v('stands')} with {p} feet together."})
    hog, more = _hogged(x), stay[rest] if rest and not _hogged(x) else ""
    out += [f"A separate short {_link(x)[1]} runs from {_point(x, p)} to {f'the chain between {hog}' if hog.endswith('handcuffs') else hog}"
            f", so {_drawn(x, p, o, True)}, the whole time."] if hog else []
    return out + ([f"Once {w.split('from the moment ', 1)[-1]}, {more[:1].lower()}{more[1:]}" if w != "the whole shot" else more]
                  if more else [])


def _pose(nm, per, was, new, did=(), pose=None, joined=False, face=False):
    p, o, s = pron(per), pron(per, 2), pron(per, 0)
    items, out, name, behind = _settle(was + new), [], cap(nm), False
    gear = lambda x: next(iter(_gears(x)), "restraints")
    done = lambda x: _TIES.get(gear(x), ("fastened",))[0]
    run = lambda k: "run" if _many([k]) or " and " in k else "runs"

    def when(hit, it="they"):
        return "the whole shot" if any(map(hit, was)) else f"from the moment {it} {'are' if it == 'they' or _many([it]) else 'is'} on"

    wrists = [x for x in items if "wrists" in parts(x)]
    wrist, ankle = next((x for x in wrists if _tied(x)), wrists[0] if wrists else None), _ankle_item(items)
    hold, leg = _tied(wrist), _tied(ankle)
    pos = next((arms(x, p) for x in wrists if arms(x, p)), "")
    ring = next((y for y in items if _lashed(y)), None)
    lashed = (f"the crotch {_unit(gear(ring))}" if _crotch(ring) else f"the {_unit(gear(ring))} around {p} waist") if ring else ""
    flat = [x for x, t in ((wrist, hold), (ankle, leg)) if t and t["lies"]]
    if flat:
        bed = _tied(flat[0])
        on, chain = bed["on"], join(list(dict.fromkeys(gear(x) for x in flat)))
        to = lambda t: "to it there" if t["to"] == on else f"to {t['to']}"
        bits = [{f"at {p} sides": f"both arms lie flat on it at {p} sides, {p} wrists {done(wrist)} {to(hold)}",
                 f"above {p} head": f"both arms are stretched out above {p} head and rest on it, {p} wrists {done(wrist)} {to(hold)}",
                 "spread apart": f"both arms are spread out wide and rest on it, {p} wrists {done(wrist)} {to(hold)}",
                 f"behind {p} back": f"both arms are pulled behind {p} back, {p} wrists locked together under {o} and {done(wrist)} "
                                     f"{to(hold)}"}.get(pos, f"{p} wrists are {done(wrist)} {to(hold)}")] \
            if wrist in flat else [] if wrist else ["both arms are free"]
        bits.append(f"both legs are spread out wide on it, {p} ankles {done(ankle)} {to(leg)}" if ankle in flat and "spread apart" in ankle
                    else f"both legs lie straight along it, {p} ankles {done(ankle)} {to(leg)}" if ankle in flat else
                    f"both legs lie straight along it, {p} ankles bound together" if ankle else "both legs lie along it at their full length")
        ends = list(dict.fromkeys(_tied(x)["to"] for x in flat))
        start = when(lambda x: x in flat, f"the {chain}")
        whole = start == "the whole shot"
        out.append(f"{'' if whole else cap(start) + ', '}{name if whole else nm} lies {'face down' if bed['face'] else f'on {p} back'} "
                   f"on {on}{' the whole shot' if whole else ''}, {p} whole body resting on it: {'; and '.join(bits)}. The {chain} "
                   f"{run(chain)} from {p} {join([w for w, x in (('wrists', wrist), ('ankles', ankle)) if x in flat])} straight "
                   f"{f'down to {on}' if all(t == on or t.endswith(f' of {on}') for t in ends) else f'to {join(ends)}'}, so {p} "
                   f"{join([w for w, x in (('arms', wrist), ('legs', ankle)) if x in flat])} stay down on {on} and never rise into the air.")
    if wrist and wrist not in flat:
        w = when(lambda x: "wrists" in parts(x), f"the {gear(wrist)}")
        seat = hold and _SEAT.search(hold["to"])
        if hold and hold["up"] or pos == f"above {p} head":
            out.append(f"{name}'s arms stay raised above {p} head {w}, {p} wrists {_held_at(wrist, p) if hold else 'locked together there'}: "
                       f"both arms run at their full length from {p} shoulders up to {p} wrists.")
        elif lashed and not pos.startswith("behind"):
            out.append(f"{name}'s wrists stay locked together in front of {o} {w}, held to the front of {lashed} by a separate short "
                       f"{_link(ring)[1]}: both arms bend at the elbows so {p} hands rest at the front of {p} waist, and {p} hands never "
                       f"come apart and never come free.")
        elif hold and _ARMREST.search(hold["to"]):
            them = "them" if re.search(r"\barm(?:rest)?s\b", hold["to"], re.I) else "it"
            out.append(f"{name} sits on {hold['on']}, {p} wrists {done(wrist)} to {hold['to']} {w}: {p} forearms rest along {hold['to']}, "
                       f"so {p} hands stay down on {them} and never rise into the air.")
        elif pos.startswith("behind"):
            behind = True
            around = f" around {hold['to']}" if hold and re.search(r"\b(?:posts?|poles?|pillars?|columns?|trees?|pipes?)$", hold["to"]) else ""
            also = f" and held to the back of {lashed} by a separate short {_link(ring)[1]}" if lashed else \
                f" and {done(wrist)} to {hold['to']}" if hold and not seat and not around else ""
            k = max((j for j, q in enumerate(did) if q in ("kneel", "sit", "lie", "stand")), default=-1)
            lying = (did[k] if k >= 0 else pose) == "lie" and not seat and not around
            out.append((f"{name} sits on {hold['to']}, {done(wrist)} to it {w}. " if seat else "")
                       + f"{name}'s arms stay pulled behind {p} back{around} {w}, with {p} wrists locked together behind {o}{also}: "
                       + (f"lying face down, both {p} arms lie along {p} back at their full length, and {p} hands stay locked together "
                          f"on top of the small of {p} back, in plain view from above and from either side. " if lying and face else
                          f"lying on {p} back, {s} {'lie' if s == 'they' else 'lies'} on {p} arms, both arms run down behind {o} under "
                          f"{p} body, and {p} hands stay locked together under the small of {p} back, hidden beneath {o}. " if lying else
                          f"{p} shoulders are drawn back, both arms run at their full length from {p} shoulders down behind {o}, and {p} "
                          f"hands rest together at the small of {p} back. From the front, {p} body hides {p} hands; seen from the side or "
                          f"from behind, both arms and both hands are in plain view, whole and locked together. ")
                       + f"Whatever {nm} does in this shot, {s} {'do' if s == 'they' else 'does'} it with {p} hands locked behind "
                         f"{p} back.")
        elif hold and re.search(r"\b(?:tables?|desks?|counters?)$", hold["to"]):
            out.append(f"{name} sits at {hold['to']} {w}, {p} forearms resting on top of it in front of {o} and {p} wrists {done(wrist)} "
                       f"to it: the {gear(wrist)} {run(gear(wrist))} straight down from {p} wrists to it, so {p} hands stay down on it.")
        elif hold:
            out.append(f"{name}'s wrists stay {done(wrist)} to {hold['to']} {w}: the {gear(wrist)} {run(gear(wrist))} straight from {p} "
                       f"wrists to it, and both arms run at their full length from {p} shoulders to {p} wrists.")
        elif pos in (f"at {p} sides", "spread apart"):
            out.append(f"{name}'s arms stay {'down at ' + p + ' sides' if 'sides' in pos else 'spread apart'} {w}, {p} wrists "
                       f"{done(wrist)} there: both arms run at their full length from {p} shoulders to {p} wrists.")
        elif pos:
            reach = f"both arms bend at the elbows so {p} hands rest at {p} waist" if pos == f"at {p} waist" else \
                f"both arms run at their full length from {p} shoulders to {p} wrists"
            out.append(f"{name}'s wrists stay locked together {pos} {w}: {reach}, and {p} hands never come apart and never come free.")
    hobbled = ankle if ankle and not flat and not leg else None
    pose = "sit" if hold and (_SEAT.search(hold["to"]) or _ARMREST.search(hold["to"])) else pose
    if ankle and not flat:
        out += _ankles(name, ankle, when(lambda x: "ankles" in parts(x), f"the {gear(ankle)}"), per, did, pose)
    if {"wrists", "ankles", "body"} & set().union(*map(parts, items)):
        out.append(f"Whenever the camera shows any part of {nm}, {s} {'are' if s == 'they' else 'is'} whole and in natural proportion: "
                   f"{p} head, shoulders, both arms and hands, torso, hips, both legs and feet are all there at their natural size, each "
                   f"joined to {p} body the natural way and placed as this shot describes.")
    pair = bool(wrist and ankle) and ankle not in wrists and gear(wrist) == gear(ankle) == "handcuffs"
    cuffs = [x for x in items if "handcuffs" in _gears(x) and x != hobbled]
    loose = sorted(set().union(*(_places(x) for x in cuffs if not _tied(x))))
    at = next((_held_at(x, p).replace("locked together", "locked") for x in cuffs if arms(x, p) or _tied(x)), "")
    whose = f"The handcuffs on {nm}'s wrists" if pair and cuffs == [wrist] else f"{name}'s handcuffs"
    out += [f"{whose} are rigid steel{f', {at}' if at else ''}: two solid cuffs joined by their own short chain of a few links that "
            f"never stretches, bends or gets longer" + (f", so {p} {join(loose)} stay a few centimetres apart at most" if loose else "")
            + "."] if cuffs else []
    crotch = next((x for x in items if _crotch(x)), None)
    if crotch:
        u, fixed = _unit(_gear(crotch)), _HARNESS[_gear(crotch)][0]
        out.append(f"{name}'s crotch {u} has two parts: a waist {u} {fixed} tight around {p} waist like a belt, and a center {u} "
                   f"{fixed} to the front of the waist {u} below {p} navel and to the back of the waist {u} at the small of {p} back. "
                   f"Between those two points the center {u} runs under {p} crotch, pulled tight over {p} "
                   f"{'vagina' if p == 'her' else 'crotch'}, and up between {p} buttocks, so all of it stays at {p} hips. It stays taut "
                   f"and centered the whole time and never hangs loose, slips to one side or sags.")
    chained = [g for g in ("chains", "shackles") if any(g in _gears(x) and not _band(x) and not (x == hobbled and g == "shackles")
                                                        for x in items)]
    out += [f"The {join(chained)} on {nm} are solid steel links of a fixed length that never stretch, bend or get longer."] if chained else []

    def piece(x):
        g, one = gear(x), re.search(r"\b(?:left|right) ankle\b", x)
        if _band(x):
            return f"{p} crotch {_unit(g)}" if _crotch(x) else f"the {_unit(g)} around {p} waist"
        if g == "a chastity belt":
            return f"{p} chastity belt"
        if x == wrist:
            return f"the handcuffs on {p} wrists" if pair else f"{p} handcuffs" if g == "handcuffs" else f"the {g} on {p} wrists"
        return f"the spreader bar between {p} ankles" if g == "a spreader bar" else \
            f"the {g} on {p} {one.group(0) if one else next((q for q in ('knees', 'legs', 'thighs') if q in _places(x)), 'ankles')}"

    pieces = [piece(x) for x in dict.fromkeys([wrist, ankle, next((x for x in items if _band(x)), None),
                                               next((x for x in items if _gear(x) == "a chastity belt"), None)]) if x]
    pieces[:1] = [cap(re.sub(rf"\b{p}\b", f"{nm}'s", x, count=1)) for x in pieces[:1]]
    ties = [f"a separate short {_link(y)[1]} from {f'the chain between {to}' if to.endswith('handcuffs') else to} to "
            f"{_point(y, p, front)}" for y in items for to, front in [_lashed(y) or (_hogged(y), False)] if to]
    them, many = "them" if len(pieces) == 2 else "any of them", len(ties) > 1
    out += [f"{join(pieces)} are separate restraints, " + (f"and the only {'links' if many else 'link'} between {them} "
                                                           f"{'are' if many else 'is'} {join(ties)}." if ties else
                                                           "each on its own with nothing joining it to another.")] \
        if len(pieces) > 1 and (ties or not joined) else []
    return out, behind


def _pair(text, kinds, spots, gear):
    hits = [h for h in _hits(text) if h[0] in {g for g, _ in kinds}]
    if len({h[0] for h in hits}) < 2:
        return [(gear[0], s) for s in spots]
    t = _BETWEEN.sub(lambda m: "crotch".ljust(len(m.group(0))), _lap(_TOOL.sub(lambda m: " " * len(m.group(0)), text)))
    at = {}
    for m in re.finditer(rf"\b({_SPOTS})\b", t, re.I):
        at.setdefault(_SPOT_AS.get(m.group(1).lower(), m.group(1).lower()), m.start())
    home = {g: d for g, _, _, d in _GEAR}
    pairs = [(min([h for h in hits if home.get(h[0]) == s] or hits, key=lambda h: abs(h[2] - at.get(s, 0)))[0], s) for s in spots]
    return pairs + [(g, home[g]) for g in dict.fromkeys(h[0] for h in hits) if g in ("handcuffs", "shackles", "a spreader bar")
                    and g not in {q for q, _ in pairs} and home[g] not in spots]


def _clothes(clause, names):
    out = []
    for m in _GARMENT.finditer(clause):
        before, word = clause[:m.start()], m.group(0).lower()
        if word in ("top", "tops") and re.match(r"\s+of\b", clause[m.end():]) or word in ("dress", "dresses") and re.search(
                rf"(?:\b(?:she|he|they|who)|(?-i:\b(?:{names})))\s+(?:\w+ly\s+)?$", before, re.I):
            continue
        words = list(re.finditer(r"[\w-]+", before))
        mods = []
        for w in reversed(words[-2:]):
            if w.group(0).lower() in _DETS or re.fullmatch(names, w.group(0)) or re.fullmatch(_HANDLE, w.group(0), re.I):
                break
            mods.insert(0, w)
        k = len(words) - len(mods)
        art = words[k - 1].group(0).lower() + " " if k and words[k - 1].group(0).lower() in ("a", "an") else ""
        out.append((mods[0].start() if mods else m.start(), m.end(), " ".join([w.group(0) for w in mods] + [m.group(0)]), art))
    return out


def _runs(clause, found):
    runs = []
    for f in found:
        if runs and re.fullmatch(r"\s*,?\s*(?:(?:and|or|&)\s+)?(?:(?:the|her|his|their|its|a|an|both|[\w-]+['’]s)\s+)?",
                                 clause[runs[-1][-1][1]:f[0]], re.I):
            runs[-1].append(f)
        else:
            runs.append([f])
    return runs


def _pulls(verb):
    if re.match(r"(?:pull|tug|yank|drag|slid|slip|roll|peel|work|eas|inch|wriggl|shimm|wrestl|push|shov|jerk|wrench|haul|draw|drew|com|"
                r"came|fall|fell|drop)", verb, re.I):
        return verb
    v = verb.lower()
    return "pulling" if v.endswith("ing") else "pulled" if re.search(r"(?:ed|en|took|got|shed)$", v) else "pulls" if v.endswith("s") \
        else "pull"


def _change(verb, part, post, of=""):
    vc = next((k for k, rx in _GVERBS if re.fullmatch(rx, verb, re.I)), "move")
    q = _GPOST.match(post)
    q = q.group(1).lower() if q else ""
    part = re.sub(r"\s+", " ", (part or "").lower())
    back = part.startswith("back") or bool(re.match(r"\s+back\b", post, re.I))
    way = part.replace("back", "").strip() or q
    dest = re.match(r"(?:\s+\w+ly)*(?:\s+back)?(?:\s+(?:up|down))?\s+(?:in|into|inside|onto|on|at|to|under|beside|across|over|"
                    r"between|past)\s+(?:the|a|an|his|her|their|its|(?-i:[A-Z])[\w'’-]*['’]s)\s+(?:open\s+)?(?!place\b)([\w-]+)", post, re.I)
    if re.match(r"(?:\s+\w+ly)*\s+(?:out\s+of|from)\s+(?:her|his|their|[\w-]+['’]s)\s+(?:\w+\s+)?(?:mouth|lips|teeth)\b", post, re.I):
        return "", ""
    if dest and re.fullmatch(r"mouth|lips|teeth", dest.group(1), re.I) and not re.match(r"(?:\s+\w+ly)*\s+(?:over|across)\b", post,
                                                                                        re.I):
        return ("tear" if vc == "tear" else "remove"), ""
    if (back or vc in ("put", "fix")) and (dest and not re.fullmatch(
            rf"(?:{_SPOTS}|body|shoulders?|head|hips|bottom|ass|butt|legs|breasts|chest|arms|feet|skin|own|collar|chin|neckline)",
            dest.group(1),
            re.I) and not re.match(r"(?:floor|ground|rug|carpet)$", dest.group(1), re.I) or re.match(r"(?:\s+\w+ly)*\s+back\s+at\b",
                                                                                                    post, re.I)):
        return "", ""
    if vc == "fix" or q in ("closed", "shut"):
        return "back", ""
    if vc == "tear" and re.fullmatch(r"straps?", of or ""):
        return "tear", ""
    if vc == "put" and re.match(r"(?:\s+\w+ly)*\s+(?:over|around)\s+(?:her|his|their|(?-i:[A-Z])[\w'’-]*['’]s)\s+(?:shoulders|body|"
                                r"head)\b", post, re.I):
        return "wear", ""
    if re.search(r"\b(?:to|onto|on|into)\s+the\s+(?:floor|ground|rug|carpet|corner|hamper|basket)\b", post[:60], re.I):
        return ("tear" if vc == "tear" else "remove"), ""
    if vc in ("put", "move") and (part.replace("back", "").strip() in ("on", "into", "in") or q == "on" and not re.match(
            r"(?:\s+(?:back|\w+ly))*\s+on\s+(?:the|a|an|top|to|his|her|their)\b", post, re.I)):
        return "wear", ""
    if q == "into" and not re.match(r"(?:\s+(?:back|\w+ly))*\s+into\s+place\b", post, re.I):
        return "", ""
    if way in ("off", "out of", "away", "out") and not re.match(r"\s+(?:\w+\s+)?(?:her|his|their)\s+shoulders?\b", post, re.I) or \
            re.match(r"(?:\s+\w+){0,3}?\s+and\s+off\b", post, re.I) or \
            re.match(r"(?:\s+[\w'’]+){0,4}?\s+over\s+(?:her|his|their)\s+head\b", post, re.I):
        return ("tear" if vc == "tear" else "remove"), ""
    if re.search(r"\b(?:off|from|down)\s+(?:her|his|their|[\w-]+['’]s)\s+shoulders?\b", post[:60], re.I):
        return "lift", "down off her shoulders"
    if vc == "remove":
        return "remove", ""
    if back:
        return "back", ""
    if vc == "tear" or vc == "open" and way in ("", "open", "apart"):
        return "lift", "open"
    if vc in ("raise", "move") and re.match(r"(?:\s+\w+){0,2}?\s+to\s+(?:the|one)\s+side\b", post, re.I):
        return "lift", "aside"
    if vc == "raise" or vc == "move" and way in ("up", "down", "aside", "open", "apart", "over", "past", "to", "above", "below",
                                                  "around"):
        toward = _TOWARD.search(f"{part} {post}"[:80])
        return "lift", toward.group(0) if toward else "open" if way == "apart" else way if way in ("up", "down", "aside", "open") else ""
    return "", ""


def _named(clause, names, words):
    m = re.search(rf"(?-i:\b({names}))['’]s\s+(?:[\w-]+\s+){{0,3}}?(?:{words})\b", clause, re.I)
    if m:
        return m.group(1)
    m = re.search(rf"\b(?:{_VERBS}|on|onto|around|from|off)\s+(?:\w+ly\s+)?(?-i:({names}))\b(?!['’])", clause, re.I)
    return m.group(1) if m else None


def _thing(text, last=""):
    hits = list(_THING.finditer(text or ""))
    return (hits[-1].group("c") or "") if hits else last


def read_beat(text, cast, people, held, off, worn, lifted, before="", crowd=None, story=""):
    names = "|".join(re.escape(n) for n in sorted(cast, key=len, reverse=True))
    got, missed, have, edits, told = [], [], {n: list(held.get(n, [])) for n in cast}, [], narration(text)
    if not names:
        return [], missed, text
    gone, raised, fresh, freed_now = {n: list(off.get(n, [])) for n in cast}, {n: list(lifted.get(n, [])) for n in cast}, set(), set()
    cut_now = set()
    duct, actor, meek, topic, last, prev = bool(re.search(r"\bduct\b", text or "", re.I)), None, None, None, None, []
    said_verb = None
    seen, watchers = _thing(narration(before)), set()
    ends = [narration(before)[a:b] for a, b in _sentences(before or "")]
    carry = [[(g, a) for _, _, g, a in r] for r in _runs(ends[-1], _clothes(ends[-1], names))][-1:] if ends else []
    carry = carry[0] if carry else None
    refer, extra = _folk_ref(crowd, people, cast)
    ones = {v["label"]: v["sex"] for v in (crowd or {}).values() if v["count"] == 1}
    xsex = {v["label"]: v["sex"] for v in (crowd or {}).values()}
    sex_of = lambda n: people.get(n, {}).get("sex", "") if n in cast else xsex.get(re.sub(r"^one of ", "", n or ""), "")
    as_doer = lambda lab: lab if not lab or lab in cast or lab in ones else f"one of {lab}"
    acts = r"(?!['’])\s+(?:\w+ly\s+)?(?-i:[a-z]+(?:s|ed))\b"
    hits = [(m.start(), m.group(1)) for m in re.finditer(rf"(?-i:\b({names}))\b{acts}", narration(before))]
    hits += [(m.start(), as_doer(extra(m.group(1)))) for m in re.finditer(rf"\b({refer}){acts}", narration(before), re.I)]
    latest, past, owned = max(hits)[1] if hits else None, "", None
    anyone = rf"(?:(?-i:{names})|her|him|them|herself|himself|themselves)"
    path = re.compile(rf"^(?:\s+\w+ly)?(?:\s+{anyone}(?!['’]))?(?:\s+(?P<part>{_GPART}))?(?:\s+hold\s+of)?(?:\s+(?:the|her|his|their|"
                      rf"(?-i:{names})['’]s)(?:\s+[\w-]+)?\s+(?P<of>straps?|hem|waistband|front|buttons?|zipper|zip|belt|sash|ties?|"
                      rf"sleeves?|neckline|edge|cups?|clasp|hooks?|laces?|elastic|band)\s+(?:of|on))?(?:\s+(?:the|a|an|her|his|their|"
                      rf"its|this|that|these|those|both|own|(?-i:{names})['’]s))*\s*$", re.I)
    target = re.compile(rf"\b(?:{_VERBS})\s+(?:\w+ly\s+)?(?:(?-i:{names})\b(?!['’])|(?:her|him|them)\b{_OBJECT})", re.I)
    strip_to = re.compile(rf"\b(?:strip(?:s|ped|ping)?|undress(?:es|ed|ing)?)(?:\s+({anyone}))?\s+(?:all\s+the\s+way\s+)?(?:down\s+)?to\s+"
                          rf"(?:her|his|their|(?-i:{names})['’]s)\s+(?:only\s+)?$", re.I)
    example = next((n for n in cast if n in people), cast[0])
    family = lambda x: {{"a ball gag": "a gag"}.get(g, g) for g in _gears(x)}

    def bound(n):
        return "wrists" in set().union(*map(parts, have.get(n, [])))

    def other(doer, sure, sex="", near=(), far=()):
        if doer and bound(doer):
            return doer
        fit = [n for n in cast if not sex or people.get(n, {}).get("sex", "") in (sex, "")]
        others = [n for n in fit if n != doer]
        others = ([n for n in others if n in near] or [n for n in others if n not in far] or others) if doer and doer not in cast \
            else others
        tied = [n for n in (others if sure else fit) if have.get(n)]
        return others[0] if sure and doer and len(others) == 1 else tied[0] if len(tied) == 1 else \
            others[0] if doer and len(others) == 1 else None

    def gendered(c):
        m = re.search(rf"\b(her|his|him)\s+(?:own\s+)?(?:[\w-]+\s+)?(?:{_SPOTS}|{_KIT})\b|\b(?:{_VERBS})\s+(?:\w+ly\s+)?(her|him)\b",
                      c, re.I)
        sex = ("f" if (m.group(1) or m.group(2)).lower() == "her" else "m") if m else ""
        fit = [n for n in cast if sex and people.get(n, {}).get("sex") == sex]
        return fit[0] if len(fit) == 1 else None

    def is_off(n, g):
        return any(x == "*" or _same(g, x) for x in gone.get(n, []))

    def wears(n, g, loose=False):
        return n in people and bool(_fit(_on(people[n]["desc"], gone.get(n, [])) + [w for w in worn.get(n, []) if not is_off(n, w)], g,
                                         loose))

    def real(n, g, loose=True):
        fit = _fit((_on(people[n]["desc"], []) if n in people else []) + worn.get(n, []), g, loose)
        return clean(fit[0], False) if len(fit) == 1 else g

    def cloth(g):
        fit = {clean(f[0], False) for n in cast if n in people for f in [_fit(_on(people[n]["desc"], []) + worn.get(n, []), g, False)]
               if len(f) == 1}
        name = fit.pop() if len(fit) == 1 else re.sub(r"^(?:her|his|their|the|a|an)\s+", "", g)
        return name if re.search(r"(?:s|wear|lingerie)$", name, re.I) else ("an " if name[:1].lower() in "aeiou" else "a ") + name

    def restraint(sent, c, doer, sure, force=None, base=-1):
        nonlocal topic, last, prev
        pair = None if force else re.search(rf"(?-i:\b({names}))(?:['’]s)?\s+and\s+(?-i:({names}))['’]s\s+(?:[\w-]+\s+)?(?:{_SPOTS})\b",
                                            c) or (None if re.search(rf"(?-i:\b(?:{names}))['’]s", c) or not re.search(
            rf"\btheir\s+(?:[\w-]+\s+)?(?:{_SPOTS})\b", c, re.I) else re.search(
            rf"(?-i:\b({names}))\s+and\s+(?-i:({names}))\b(?!['’])", sent[:sent.find(c) + len(c)] if c in sent else c))
        for n in pair.groups() if pair else ():
            restraint(sent, c, doer, sure, n, base)
        if pair:
            return
        low = {"waist", "hips", "crotch"}
        belly = low & set(_spots(c)) or re.search(r"\b(?:it|them)\b", c, re.I) and low & set(prev)
        onto = _ONTO.search(c) or belly and re.search(r"\b(?:to|through|onto)\s+(?:her|his|their|[\w-]+['’]s)\s+(?:wrists|hands)\b"
                                                     r"(?:\s+behind\s+(?:her|his|their)\s+back)?", c, re.I)
        ring, was, plain, hog = _RING.search(c), topic, c, {"wrists", "ankles"} in _links(c)
        for m in (onto, ring):
            plain = plain[:m.start()] + " " * len(m.group(0)) + plain[m.end():] if m else plain
        bare = _TOOL.sub(" ", re.sub(r"\b(?:between|through)\s+(?:her|his|their|[\w-]+['’]s)\s+(?:hand)?cuffs\b", "around her wrists", c,
                                     flags=re.I))
        flat = _TOOL.sub(" ", plain) if onto or ring else bare
        spots = own = _spots(flat)
        kinds = [k for k in _kinds(plain) if not (k[0] == "a collar" and (_GARMENT.search(c) or topic and topic[0] == "clothes" and
                                                                          re.search(r"\bthe\s+collar\b", c, re.I) and not _acts(c))
                                                  or k[0] == "rope" and not spots and (_GARMENT.search(c) or
                                                                                       re.search(r"\bhair\b", c, re.I)))]
        if [k[0] for k in kinds] == ["rope"] and not re.search(r"\b(?:ropes?|roped|cords?)\b", c, re.I) and topic and \
                topic[0] == "gear" and topic[1][0] != "rope" and (not topic[1][1] or topic[1][1] in (spots or prev) or not spots and re.search(
                    r"\b(?:it|them)\b", c, re.I)):
            kinds = [topic[1]]
        ref = re.search(r"\b(?:it|them|the\s+other|(?:the|that|this|each|another|more|extra|two|three|several|a\s+second)\s+(?:\w+\s+)?"
                        r"(?:roll|strips?|pieces?|ends?|layers?|ball))\b", c, re.I)
        rag = _RAG.search(c)
        rag = (rag.group("a") or rag.group("b")) if rag else seen if ref and re.fullmatch(r"it|them", ref.group(0), re.I) and (
            _INMOUTH.search(c) or _OVER.search(c)) and _acts(c, spots) else ""
        if (rag or _CLOTH.search(c) and _loosens(c)) and {"eyes", "mouth"} & set(spots) and not {g for g, _ in kinds} & {"duct tape", "a ball gag",
                                                                                                         "a blindfold", "a gag"}:
            kinds = [("a blindfold", "eyes")] if "eyes" in spots else [("a gag", "mouth")]
        rag = cloth(rag) if rag and {g for g, _ in kinds} & {"a gag", "a blindfold"} and not {g for g, _ in kinds} & {
            "duct tape", "a ball gag"} else ""
        if not kinds and _CLOTH.search(c) and _MOVED.search(c):
            kinds = [("a blindfold", "eyes")] if re.search(r"\beyes\b", c, re.I) else [("a gag", "mouth")]
        if "chains" in {g for g, _ in kinds} and not spots and re.search(r"\bcollar\b", c, re.I):
            kinds, spots = [k for k in kinds if k[0] != "a collar"], ["neck"]
        taped = duct or re.search(r"\btap(?:e|es|ed|ing)\b", text or "", re.I) or any("duct tape" in _gears(x) for v in have.values()
                                                                                    for x in v)
        if not kinds and taped and (ref and re.search(r"\b(?:roll|strips?|pieces?|layers?)\b", ref.group(0), re.I) and not (
                topic and topic[0] == "gear") or _PREP.search(c) and re.search(r"\b(?:strip|piece)s?\b", c, re.I)):
            topic = ("gear", ("duct tape", ""), sent)
        freed = re.search(rf"\b(?:frees?|freed|releases?|released)\s+(?:\w+ly\s+)?(?:{anyone}|(?-i:{names})['’]s|his|their)\b"
                          rf"(?!\s+(?:grip|hold|own|hand)\b)", c, re.I)
        same = _SAME.search(c)
        it = topic and topic[0] == "gear" and (ref or same)
        elided = bool(topic and topic[0] == "gear" and topic[2] == sent and own and not _kinds(plain) and re.fullmatch(
            rf"\W*(?:(?:then|and|also|next)\W+)*(?:her|his|their|[\w-]+['’]s)\s+(?:[\w-]+\s+)?(?:{_SPOTS})(?:\s+(?:too|as\s+well))?\W*", c,
            re.I))
        it = it or elided or topic and topic[0] == "gear" and own and not _kinds(plain) and (re.search(
            r"\b(?:pulled|stretched|drawn|wrapped|wound|smoothed|pressed|layered)\s+(?:tight(?:ly)?|taut|snug(?:ly)?|firm(?:ly)?|flat|"
            r"hard)\b|\b(?:seal|cover|pinn|wrapp|bind|tap|gagg|lock|strapp|press|fasten)ing\b", c, re.I) or topic[2] == sent and _acts(c))
        if not (kinds or freed or it or hog):
            return
        gear = kinds[0] if kinds else topic[1] if it else ("rope", "wrists") if hog else None
        topic = ("gear", gear, sent) if gear else topic
        spots = spots or (prev if ref or same else []) or (["mouth"] if gear and gear[0] == "duct tape" and re.search(
            r"\baround\s+(?:her|his|their|[\w-]+['’]s)\s+head\b", c, re.I) else ["wrists", "ankles"] if hog or re.search(
            r"\bspread[\s-]?eagle", c, re.I) else [])
        if not own and not target.search(c) and (_PREP.search(c) and not freed or _SOURCES.search(c) and not _UNDO.search(c)
                                                 or _PUTAWAY.search(c) or _SHOWN.search(c)):
            return
        loose = bool(freed) or _loosens(c)
        if loose and ref and re.search(r"\broll\b", ref.group(0), re.I) and not own:
            return
        if gear and gear[1] in ("mouth", "eyes") and _MOVED.search(c):
            loose, spots = True, [gear[1]]
        if loose and re.search(r"\bone\s+(?:of\s+(?:the|her|his|their)\s+)?(?:cuffs?|wrists?|ankles?|sides?|hands?|shackles?)\b|"
                               r"\b(?:left|right)\s+(?:wrist|ankle|hand|cuff|arm|leg)\b", c, re.I):
            return
        prev = spots or prev
        acting = _acts(c, spots) or bool(same) or bool(_SOON.search(c)) or elided
        still = not loose and not acting and bool(_STILL.search(c) or (last and last[1] == "have" and not _PRONOUN.search(c)
                                                                       and not re.search(rf"(?-i:\b(?:{names}))\b", c)))
        if not (loose or still or acting or _DOES.search(c) or spots and _ROUTES.search(c)) or \
                not (loose or still or spots or target.search(c)) and _PLACED.search(c):
            return
        kit = set() if hog and loose else {{"a ball gag": "a gag"}.get(g, g) for g, _ in kinds} or (
            {{"a ball gag": "a gag"}.get(gear[0], gear[0])} if gear and not freed and (kinds or it) else set())
        thing = gear[0].replace("a ", "") if gear and not freed and (kinds or it) and not (hog and loose) else "restraints"
        who = force or _named(bare, names, f"{_SPOTS}|{_KIT}|head|face")
        if not who and re.search(rf"\b(?:{refer})['’]s\s+(?:[\w-]+\s+){{0,2}}?(?:{_SPOTS}|{_KIT}|head|face)\b|\b(?:{_VERBS})\s+"
                                 rf"(?:\w+ly\s+)?(?:{refer})(?!['’])", c, re.I):
            return
        who = who or (owned if ref and re.fullmatch(r"it|them", ref.group(0), re.I) and not own else None)
        named = bool(who)
        if not who and re.search(rf"\b(?:{_VERBS}|on|onto|around)\s+(?:\w+ly\s+)?(?:herself|himself|themselves)\b", c, re.I):
            who = doer
        who = who or gendered(bare)
        if not who and still:
            state = re.search(rf"\b(?:{_STATE})\b|{_WORN}", c, re.I)
            near = re.findall(rf"(?-i:\b({names}))\b(?!['’])", c[:state.start()] if state else "")[-1:] or \
                re.findall(rf"(?-i:\b({names}))\b(?!['’])", c)[:1] or \
                re.findall(rf"(?-i:\b({names}))\s*,\s*$", sent[:sent.find(c)] if c in sent else "")
            tied = [n for n in cast if have.get(n)]
            who = near[0] if near else tied[0] if len(tied) == 1 else actor
        if not who and re.search(r"\b(?:her|his|their)\s+own\b", c, re.I):
            who = doer
        pro = re.search(r"\b(?:her|him|his|their)\b" + ("" if it else r"|\bthem\b"), bare, re.I)
        if not who and pro:
            at = sent.find(c) + (re.search(rf"\b{pro.group(0)}\b", c, re.I) or pro).start() if c in sent else 0
            g = {"her": "f", "him": "m", "his": "m"}.get(pro.group(0).lower(), "")
            marks = [(m.end(), n) for n in cast for m in re.finditer(rf"(?-i:\b{re.escape(n)})\b", sent[:at])]
            marks += [(m.end(), extra(m.group(0))) for m in re.finditer(rf"\b(?:{refer})", sent[:at], re.I)]
            marks = [(e, n) for e, n in marks if n and n != doer and (not g or (people.get(n, {}).get("sex", "") if n in cast else
                                                                                 xsex.get(n, "")) in (g, ""))]
            if marks and max(marks)[1] not in cast:
                return
            near = [n for n in cast if n not in watchers and re.search(rf"(?-i:\b{re.escape(n)})\b", sent[:at])]
            near = near or [max([(m.start(), n) for n in cast if n != doer and n not in watchers and (not g or sex_of(n) in (g, ""))
                                 for m in re.finditer(rf"(?-i:\b{re.escape(n)})\b", past)] or [(0, None)])[1]] if not near else near
            far = watchers | {n for n in cast if n not in near and re.search(rf"(?-i:\b{re.escape(n)})\b", sent[at:])}
            who = other(doer, sure, {"her": "f", "him": "m", "his": "m"}.get(pro.group(0).lower(), ""), near, far)
        holders = [n for n in cast if (any(family(x) & kit for x in have.get(n, [])) if kit else have.get(n))]
        if loose and not named and len(holders) == 1:
            who = holders[0]
        if not who and gear and gear[0] == "chains" and re.search(r"\bcollar\b", c, re.I):
            collared = [n for n in cast if any("a collar" in _gears(x) for x in have.get(n, []))]
            who = collared[0] if len(collared) == 1 else None
        if not who and not loose and c in sent:
            who = _named(sent[:sent.find(c)], names, f"{_SPOTS}|head|face")
        if not who and last and not re.search(rf"(?-i:\b(?:{names}))\b", c):
            who = last[0]
        if not who and not loose and holders:
            return
        if who and who not in cast or not who and loose and not holders and any(e[0] == "release" and e[3] == sent for e in got):
            return
        if not who:
            say = ("come off" if loose else "these are") if _many([thing]) else ("comes off" if loose else "this is")
            missed.append((sent, f"could not tell whose {thing} {say}; name the person, as in "
                                 f"\"{example}'s wrists\""))
            return
        p, mine = pron(people.get(who, {})), have.get(who, [])
        band = next((y for y in mine if _band(y)), None)
        hooked = ring or band and (re.search(rf"\b(?:to|onto|under|through|from)\s+(?:the\s+(?:back|front|middle)\s+of\s+)?the\s+"
                                             rf"{_unit(_gear(band))}s?\b", c, re.I) or re.search(
            r"\b(?:to|from)\s+(?:it|them)\b", c, re.I) and was and was[0] == "gear" and was[1][0] == _gear(band))
        cuffs = re.search(r"\b(?:(?:hand)?cuff\w*|wrists?|hands)\b", plain, re.I)
        if loose:
            undone = [x for x in mine if _lashed(x) and (onto or hooked and cuffs) or _hogged(x) and hog] \
                if not re.search(r"\boff\b", c, re.I) else []
            if undone:
                have[who] = [_TIED_TO.sub("", x) if x in undone else x for x in mine]
                got.append(["hold", who, [_TIED_TO.sub("", x) for x in undone], sent, doer])
                return
            whole = (freed and not kinds and not spots) or re.search(
                rf"\b(?:un\w+|frees?|freed|releases?|released)\s+(?:\w+ly\s+)?(?:(?-i:{names})|her|him|them)\b(?!['’])"
                rf"(?!\s+(?:{_SPOTS}|own)\b)", c, re.I)
            pick = [x for x in mine if (not kit or family(x) & kit) and (not spots or set(spots) & _places(x))] or [
                x for x in mine if (not kit or family(x) & kit) and spots and parts(" ".join(spots)) & parts(x) & {"ankles", "wrists"}] or [
                x for x in mine if kit and family(x) & kit and _places(x) & {"mouth", "eyes"} and not set(spots) & {"mouth", "eyes"}]
            if not pick:
                if who not in freed_now:
                    missed.append((sent, f"nothing to take off: {who} has no {thing}" + (f" on {p} {join(spots)}" if spots else "")))
            elif not spots and not whole and len({frozenset(_places(x)) for x in pick}) > 1:
                missed.append((sent, f"could not tell which {thing} {'come' if _many([thing]) else 'comes'} off: {who} has {join(pick)}; "
                                     f"name the spot, as in "
                                     f"\"off {who}'s {sorted(_places(pick[0]))[0]}\""))
            else:
                have[who] = [x for x in mine if x not in pick]
                freed_now.add(who)
                got.append(["release", who, pick, sent, doer])
            return
        doer = doer if sure or acting or still else next(
            (n for n in reversed(re.findall(rf"(?-i:\b({names}))\s+(?:\w+ly\s+)?[a-z]+s\b", c)) if n != who), doer)
        still = still or doer is None and (not acting or bool(re.match(rf"\W*(?:the|a|an|her|his|their)\s+(?:[\w-]+\s+)?(?:{_KIT})\b", c,
                                                                       re.I))) or not acting and doer == who and bound(who)
        near = [] if still else [k for k in kinds if any(h[0] == k[0] and m.start() <= h[2] <= m.end() + 25 for h in _hits(plain)
                                                          for m in _DOES.finditer(plain))]
        spots = spots or [s for y in mine if (onto or hooked) and gear and _gear(y) == gear[0] and _band(y) for s in sorted(_places(y))]
        pairs = _pair(flat, kinds, spots, gear) if spots else [(g, d) for g, d in (near or kinds or [gear]) if d] or \
            ([(gear[0], "wrists")] if _anchor(c, c) else [])
        pairs = [(g, s) for g, s in pairs if not (g in _HARNESS and s in ("waist", "hips") and (g, "crotch") in pairs)]
        pairs += [(gear[0], s) for s in ("wrists", "ankles")
                  if hog and s not in {d for _, d in pairs} and not any(s in parts(y) for y in mine)]
        if not pairs:
            missed.append((sent, f"could not tell where the {thing} {'go' if _many([thing]) else 'goes'}; name the spot, as in "
                                 f"\"around {who}'s wrists\""))
            return
        last, new = (who, "have" if still else "hold"), []
        upto = sent[:sent.find(c) + len(c)] if c in sent else c
        cx = re.sub(r"\b(?:to|from)\s+(?:it|them)\b", " ", c, flags=re.I) if hooked else c
        item = lambda g, s, q: _gear_item(g, s, cx, q, duct, rag if g in ("a gag", "a blindfold") else "", upto,
                                          f"{narration(story or before)} {past}",
                                          next((arms(x, q) for x in have.get(who, []) if s in ("wrists", "arms") and arms(x, q)), ""))
        made = list(dict.fromkeys(_gear_item(g, s, "", p) if hog and s in ("wrists", "ankles") else item(g, s, p) for g, s in pairs))
        cuffs = cuffs or any("wrists" in parts(x) for x in made)
        on = next((y for y in mine if "wrists" in parts(y)), None)
        made = [x for x in made if not (on and (onto or hooked or hog) and "wrists" in parts(x))]
        wrist = on or next((x for x in made if "wrists" in parts(x)), None)
        tied = next((x for x in made if _band(x)), None) if onto else band if hooked and cuffs else None
        if tied:
            if not on and (not wrist or re.search(r"\b(?:cuffed|bound|tied|chained)\s+(?:wrists|hands)\b", c, re.I)):
                wrist = wrist or f"handcuffs {'in front of ' + _OBJ.get(p, p) if 'front' in onto.group(0) else 'behind ' + p + ' back'}"
                made = [x for x in made if x != wrist]
                have.setdefault(who, []).append(wrist)
                got.append(["have", who, [wrist], sent, doer])
            made = [x for x in made if x != tied] + [_tether(tied, wrist, p)]
        legs = hog and _ankle_item(mine)
        made = [x for x in made if not (legs and "ankles" in parts(x))]
        legs = hog and (legs or next((x for x in made if "ankles" in parts(x)), None))
        if legs and wrist and not _hogged(legs):
            made = [x for x in made if x != legs] + [_tether(legs, wrist, p)]
        link = _link(made[-1]) if made and (tied or legs and wrist) else None
        if link and base >= 0 and not re.search(r"\bseparate\b", c, re.I):
            at = (onto or ring or hooked) if tied else next((m for m in _LINK.finditer(c) if m["a"] or m["c"]), None)
            end = base + len(re.sub(r"[\s.!?…\"”’]+$", "", c))
            edits.extend(([] if re.match(r"\s+(?:with|using)\b", c[at.end():], re.I) else
                          [(base + at.end(), base + at.end(), 0, f" with a separate short {link[1]}")]) if isinstance(at, re.Match) else
                         [] if tied else [(end, end, 0, f", {p} ankles held to {link[0]} by a separate short {link[1]}")])
        for x in made:
            twin = next((y for y in have.get(who, []) if (still or family(x) & family(y)) and _places(x) & _places(y)), None)
            x = _tether(x, wrist, p) if twin and wrist and _link(twin) and not _link(x) and _swapped(x, [twin]) else x
            put = twin and any(e[0] == "hold" and e[1] == who and twin in e[2] for e in got)
            if twin is None:
                new.append(x)
            elif _swapped(x, [twin]) and put and still:
                have[who] = [x if y == twin else y for y in have[who]]
                for entry in got:
                    entry[2] = [x if y == twin else y for y in entry[2]] if entry[1] == who else entry[2]
                fresh.add(x)
            elif _swapped(x, [twin]):
                have[who] = [y for y in have[who] if y != twin]
                for entry in got if put else ():
                    entry[2] = [y for y in entry[2] if y != twin] if entry[1] == who else entry[2]
                got[:] = [e for e in got if e[2]]
                new.append(x)
            elif twin not in fresh and twin != x and not still and _REDO.search(c):
                have[who] = [y for y in have[who] if y != twin] + [x]
                got.append(["release", who, [twin], sent, doer])
                fresh.add(x)
                new.append(x)
                have[who].remove(x)
            elif twin in fresh and _tied(x) and not _tied(twin):
                have[who] = [x if y == twin else y for y in have[who]]
                for entry in got:
                    entry[2] = [x if y == twin else y for y in entry[2]] if entry[1] == who else entry[2]
                fresh.add(x)
        pair = re.search(rf"(?-i:\b({names}))\s+and\s+(?-i:({names}))\b", c) if still and re.search(r"\b(?:both|each|they)\b", c, re.I) \
            else None
        for n in [pair.group(1) if pair.group(1) != who else pair.group(2)] if pair and who in pair.groups() else ():
            more = [item(g, s, pron(people.get(n, {}))) for g, s in pairs]
            more = [x for x in more if not any(_places(x) & _places(y) for y in have.get(n, []))]
            have.setdefault(n, []).extend(more)
            got.extend([[last[1], n, more, sent, doer]] if more else [])
        if new:
            have.setdefault(who, []).extend(new)
            fresh.update(new)
            got.append([last[1], who, new, sent, doer])

    def owner(pre, post, g, kind, doer, sure, before=""):
        m = re.search(rf"(?-i:\b({names}))['’]s\s+(?:own\s+)?$", pre) or \
            re.search(rf"\b(?:{_HANDLE})\s+(?:\w+ly\s+)?(?-i:({names}))\b(?!['’])", pre, re.I) or \
            re.match(rf"(?:\s+[\w'’-]+){{0,3}}?\s+(?:on|onto|off|from|over|around)\s+(?-i:({names}))\b", post, re.I)
        if m:
            return [m.group(1)]
        if re.search(r"\b(?:her|his|their)\s+own\s+$", pre, re.I):
            return [doer] if doer else []
        mine = re.search(r"\b(her|his|their)\s+$", pre, re.I)
        sex = {"her": "f", "his": "m"}.get(mine.group(1).lower(), "") if mine else ""
        if sex and doer and doer not in cast and ones.get(doer, sex) in (sex, ""):
            return []
        fits = lambda n: not sex or people.get(n, {}).get("sex") in (sex, "", None)
        own = [n for n in cast if wears(n, g) and fits(n)] or [n for n in cast if kind in ("remove", "tear", "lift") and wears(n, g, True)
                                                               and fits(n)]
        said = before + pre
        group = [n for n in own if re.search(rf"(?-i:\b{re.escape(n)})\b", said)]
        if mine and not sex and (len(group) > 1 or len(own) > 1 and re.search(r"\b(?:they|both)\b", said, re.I)):
            return group if len(group) > 1 else own
        onto = re.search(r"\b(?:on|onto|over|around)\s+(her|him|his|them)\b", post, re.I)
        to = {"her": "f", "him": "m", "his": "m"}.get(onto.group(1).lower(), "") if onto else ""
        across = [n for n in cast if n != doer and to and people.get(n, {}).get("sex") == to]
        if kind in ("wear", "back") and doer in people and to and people[doer].get("sex") != to and len(across) == 1:
            return across
        if kind in ("wear", "back") and doer and doer not in cast and not onto and not re.search(
                rf"\b(?:on|onto|over|around)\s+(?-i:{names})\b", post, re.I):
            return []
        if kind in ("wear", "back"):
            had = [n for n in cast if (is_off(n, g) or any(_same(g, x) for x, _ in raised.get(n, []))) and fits(n)]
            if len(had) == 1:
                return had
            if not had and not own:
                return [x for x in [other(doer, sure) if onto else doer] if x]
        pick = [n for n in own if have.get(n) and n != doer]
        hit = [n for n in own if n != doer and any(not _GAZE.search(said[:x.start()]) for x in
                                                   re.finditer(rf"(?-i:\b{re.escape(n)})\b(?!['’])", said))]
        return own[:1] if len(own) == 1 else pick if len(pick) == 1 and doer else hit if len(hit) == 1 else \
            [doer] if doer in own else []

    def on_now(who, skip=()):
        return sorted((clean(s, False) for s in _on(people[who]["desc"], gone[who]) + [w for w in worn.get(who, []) if not is_off(who, w)]
                       if not any(worn_rx(k).search(s) for k in skip)), key=lambda g: _garb(g)[0])

    def strip(who, drop, sent, doer, tear=False):
        per = people.get(who, {})
        way = {} if tear else {g: _stuck(g, have.get(who, []), per) for g in drop}
        keep, free = [g for g in drop if way.get(g)], [g for g in drop if not way.get(g)]
        torn, lift = [g for g in free if tear and _zone(g)], [g for g in keep if not way[g].startswith("on ")]
        gone[who] += free
        raised[who] = [x for x in raised.get(who, []) if not any(_same(x[0], y) for y in free + lift)] + \
            [pulled(f"{y} {way[y]}", per) for y in lift]
        got.extend([[k, who, xs, sent, doer] for k, xs in (("remove", [g for g in free if g not in torn]), ("tear", torn)) if xs])
        got.extend([["lift", who, [f"{y} {way[y]}" for y in lift], sent, doer]] if lift else [])
        groups = {}
        for y in keep:
            groups.setdefault(way[y], []).append(y)
        missed.extend((None, _cant(who, gs, per, w)) for w, gs in groups.items())
        return groups

    def left_on(groups, p):
        kept = lambda w: "left" if w.startswith("on ") else "pulled"
        return f", {' and '.join(f'{p} {join(gs)} {kept(w)} {w}' for w, gs in groups.items())}, where " \
               f"{'it stays' if not _many([y for gs in groups.values() for y in gs]) else 'they stay'}"

    def clothes(sent, c, doer, sure, base):
        nonlocal topic, said_verb
        stop, marked = base + len(re.sub(r"[\s.!?…\"”’]+$", "", c)), set()

        def hands(who):
            cuffs = next((x for x in have.get(who, []) if "wrists" in parts(x)), None)
            if cuffs and who not in marked and not re.search(r"\b(?:hands?|wrists?|arms?|cuff\w*)\b", c, re.I):
                marked.add(who)
                p = pron(people.get(who, {}))
                edits.append((stop, stop, 1, f", with {p if who == doer else who + chr(39) + 's'} hands still "
                                             f"{re.sub(r' of (?:her|him|them)$', '', _held_at(cuffs, p))}"
                                             + ("," if re.match(r"\s+\w", told[stop:]) else "")))
        runs = [(r[0][0], r[-1][1], [(g, a) for _, _, g, a in r]) for r in _runs(c, _clothes(c, names))]
        pro = list(re.finditer(rf"\b(?:{_HANDLE})\b(?:\s+\w+ly)?(?:\s+{anyone}(?!['’]))?(?:\s+(?:{_GPART}))?\s+(it|them)\b|"
                               rf"\blet(?:s|ting)?\s+(it|them)\s+(?={_HANDLE})", c, re.I)) or \
            ([] if runs else list(re.finditer(r"^\W*(?:then\s+)?(it|they)\b", c, re.I)))
        runs = sorted(runs + [(m.start(m.lastindex), m.end(m.lastindex), None) for m in pro], key=lambda r: r[0])
        if not runs or all(r[2] is None for r in runs) and not (topic and topic[0] in ("clothes", "theirs")) and not carry:
            bare = re.search(rf"(?:(?:(?-i:\b(?:{names}))|\b(?:she|he|they))\s+(?:\w+ly\s+)?(?:strips?|stripped|undress(?:es|ed)?)"
                             rf"(?:\s+{anyone})?|\b(?:strips?|stripped|undress(?:es|ed)?)\s+{anyone})(?:\s+(?:off|naked|nude|bare|down|"
                             rf"completely|fully))*(?=\s*(?:[,.!?]|$))", c, re.I)
            rid = None if bare else re.search(rf"\b(?P<v>{_HANDLE})(?:\s+\w+ly)?(?P<a>\s+off)?\s+(?:all\s+(?:of\s+)?)?(?P<o>her|his|their|"
                                              rf"(?-i:{names})['’]s)\s+(?P<own>own\s+)?(?:clothes|clothing)\b(?P<b>\s+off)?", c, re.I)
            rid = rid if rid and (rid.group("a") or rid.group("b") or re.fullmatch(_GVERBS[2][1], rid.group("v"), re.I)) else None
            if bare or rid:
                sex = {"her": "f", "his": "m"}.get(rid.group("o").lower(), "") if rid else ""
                mine = not re.search(r"\b(?:strips?|stripped|undress(?:es|ed)?)\s+(?:her|him|them)\b", c, re.I) if bare else \
                    rid.group("own") or not sex or sex_of(doer) == sex
                who = _named(c, names, r"clothes|clothing") or (doer if mine else other(doer, sure, sex))
                if who in cast and who in people:
                    groups = strip(who, on_now(who), sent, doer, bool(rid and re.fullmatch(_GVERBS[0][1], rid.group("v"), re.I)))
                    edits.extend([(base + (bare or rid).end(), base + (bare or rid).end(), 0, left_on(groups, pron(people[who])))]
                                 if groups else [])
                    hands(who)
                elif who in cast:
                    got.append(["remove", who, [], sent, doer])
                elif not who:
                    missed.append((sent, f"could not tell whose clothes come off; name the person, as in \"strips {example}\""))
            return
        for start, end, items in runs:
            it = items is None
            held = topic[2] if it and topic and topic[0] == "clothes" and len(topic) > 2 else []
            items = items or (topic[1] if topic and (topic[0] == "clothes" or topic[0] == "theirs" and doer in cast) else carry)
            pre, post = c[:start], c[end:]
            if not items or re.search(rf"\b(?:{_FOLKS})\s+(?:in|wearing|dressed\s+in)\s+(?:(?:a|an|the|her|his|their)\s+)?"
                                      rf"(?:[\w-]+\s+){{0,2}}$", pre, re.I):
                continue
            topic, g = ("clothes", items), items[0][0]
            to = strip_to.search(pre)
            if to:
                obj = to.group(1) or ""
                who = obj if re.fullmatch(names, obj) else doer if not obj or obj.lower().endswith("self") else other(doer, sure)
                groups = strip(who, on_now(who, [k for k, _ in items]), sent, doer) if who in people else {}
                edits.extend([(base + end, base + end, 0, left_on(groups, pron(people[who])))] if groups else [])
                hands(who) if who in people else None
                continue
            vm = next((v for v in reversed(list(re.finditer(rf"\b(?:{_HANDLE})\b", pre, re.I))) if path.match(pre[v.end():])
                       and not (re.search(r"(?:ed|torn|shorn|undone)$", v.group(0), re.I) and re.search(
                           r"\b(?:a|an|the|her|his|their|its|this|that|[\w-]+['’]s)\s+$", pre[:v.start()], re.I))), None)
            of, be, again = None, None, None
            if vm:
                verb, part, of = vm.group(0), path.match(pre[vm.end():]).group("part"), path.match(pre[vm.end():]).group("of")
            elif said_verb and said_verb[0] == sent and re.fullmatch(
                    rf"\W*(?:(?:and\s+)?then\s+|and\s+|or\s+)?(?:her|his|their|(?-i:{names})['’]s)\s+(?:own\s+)?", pre, re.I):
                again = said_verb
            else:
                be = _GBE.match(post)
                if not be:
                    continue
                verb, part, post = be.group("v"), None, post[be.end():]
            verb, kind, where = again[1:] if again else (verb, *_change(verb, part, post, of))
            said_verb = (sent, verb, kind, where) if vm and kind else said_verb
            torn = re.fullmatch(_GVERBS[0][1], verb, re.I)
            if kind == "lift" and torn:
                cut_now.add(_stem(g))
            kind = "tear" if kind == "remove" and _stem(g) in cut_now else kind
            if not kind or kind in ("remove", "tear") and any(is_off(n, g) for n in cast) and not re.search(
                    rf"(?:\b(?:her|his|their)|(?-i:\b(?:{names}))['’]s)\s+(?:own\s+)?$", pre, re.I):
                continue
            whos = held if held and kind not in ("wear", "back") else owner(pre, post, g, kind, doer, sure, sent[:max(0, sent.find(c))])
            whos = [w for w in whos if w in cast]
            topic, many = ("clothes", items, whos), "are" if _many([g]) else "is"
            if not whos and doer and doer not in cast:
                topic = ("theirs", items)
                continue
            if kind in ("remove", "tear") and _INMOUTH.search(c[end:]) and not any(wears(w, g) for w in whos):
                continue
            if not whos:
                if doer in cast + [None] and (not any(is_off(n, g) for n in cast) or kind in ("wear", "back")):
                    missed.append((sent, f"could not tell whose {g} {'these are' if many == 'are' else 'this is'}; name the person, "
                                         f"as in \"{example}'s {g}\""))
                continue
            asked, wanted = kind, where
            for j, who in enumerate(whos):
                kind, where = asked, wanted
                p, up = pron(people.get(who, {})), [w for x, w in raised.get(who, []) if _same(x, g)]
                names_ = [real(who, x, kind != "wear") for x, _ in items]
                if kind in ("wear", "back"):
                    if wears(who, g) and not up or kind == "back" and not up and not is_off(who, g):
                        continue
                    kind = "wear"
                elif is_off(who, g) and not wears(who, g):
                    if kind == "lift" and not it:
                        missed.append((sent, f"{who}'s {g} {many} already off"))
                    continue
                if kind in ("remove", "tear"):
                    groups = strip(who, names_, sent, doer, kind == "tear")
                    hands(who) if kind == "remove" else None
                    said = (lambda gs: c[start:end]) if it else (lambda gs: join([a + x for (x, a), y in zip(items, names_) if y in gs]))
                    free = [y for y in names_ if not any(y in gs for gs in groups.values())]
                    if groups and vm and not j:
                        mid = pre[vm.end():start]
                        pm, tail = path.match(mid), _GONE.match(c, end).end()
                        bare = re.sub(r"\s+", " ", mid[:pm.start("part")] + mid[pm.end("part"):]) if pm.group("part") else mid
                        det = re.search(r"(?:\b(?:her|his|their)|[\w-]+['’]s)\s+$", bare)
                        det = det.group(0) if det else ""
                        act = lambda w: {"pulls": "leaves", "pulled": "left", "pulling": "leaving", "pull": "leave"}.get(
                            _pulls(verb), "leaves") if w.startswith("on ") else _pulls(verb)
                        edits.append((base + vm.start(), base + tail, 0, (f"{verb}{mid}{said(free)}{c[end:tail]} and " if free else "")
                                      + " and ".join(f"{act(w)}{bare if not k else ' ' + det}{said(gs)} {w}"
                                                     for k, (w, gs) in enumerate(groups.items()))))
                    elif groups and be and not j and not next(iter(groups)).startswith("on "):
                        edits.append((base + end + be.start("v"), base + _GONE.match(c, end + be.end()).end(), 0,
                                      f"{_pulls(verb)} {next(iter(groups))}"))
                    continue
                if kind == "lift":
                    zone, way = _zone(g), where.split(" ")[0]
                    lower = zone == {"lower"} and not _SKIRT.search(g)
                    if up and way in ("up", "down") and not up[0].startswith(way) and (where in ("up", "down") or lower and way == "up"):
                        kind = "wear"
                    elif lower and way == "up" or up and (where in ("", "up", "down") or re.search(r"\bstill\b", c, re.I)):
                        continue
                    else:
                        said, low = where, ("thighs", "knees", "calves", "ankles", "feet", "floor")
                        where = f"down to {p} waist" if where == "down" and "upper" in zone else "" if where in ("up", "down") else \
                            re.sub(r"\b(?:her|his|their)\b", p, where)
                        bar, deep = _stuck(g, have.get(who, []), people.get(who, {})), re.findall(rf"\b({'|'.join(low)})\b", where)
                        if bar.startswith("on "):
                            missed.append((None, _cant(who, names_, people.get(who, {}), bar)))
                            edits.extend([(stop, stop, 0, f", but {p} {g} {'stay' if _many([g]) else 'stays'} {bar}")] if not j else [])
                            continue
                        if bar.startswith("down") and deep and max(map(low.index, deep)) >= low.index(
                                {"legs": "knees"}.get(bar.split()[-1], bar.split()[-1])):
                            at, where = c.find(said, end), bar
                            edits.extend([(base + at, base + at + len(said), 0, bar)] if at >= 0 and not j else [])
                        if up and pulled(f"{g} {where}".strip(), people.get(who, {}))[1] == up[0]:
                            continue
                if kind == "wear" and not j and re.fullmatch(_GVERBS[5][1], verb, re.I) and doer and doer != who and wears(doer, g) \
                        and re.search(r"\b(?:her|his|their)\s+(?:own\s+)?$", pre, re.I):
                    names_ = [real(doer, x) for x, _ in items]
                    gone.setdefault(doer, []).extend(names_)
                    got.append(["remove", doer, names_, sent, doer])
                if kind == "wear":
                    gone[who] = [x for x in gone.get(who, []) if not any(_same(x, y) for y in names_)]
                    raised[who] = [x for x in raised.get(who, []) if not any(_same(x[0], y) for y in names_)]
                else:
                    raised[who] = [x for x in raised.get(who, []) if not any(_same(x[0], y) for y in names_)] + \
                        [pulled(f"{y} {where}".strip(), people.get(who, {})) for y in names_]
                hands(who)
                got.append([kind, who, [f"{y} {where}".strip() for y in names_] if kind == "lift" else
                            [(a or ("" if _many([y]) else "an " if y[:1].lower() in "aeiou" else "a ")) + y
                             for y, (_, a) in zip(names_, items)] if kind == "wear" else names_, sent, doer])

    for a, b in _sentences(text):
        sent = told[a:b].strip()
        if not sent:
            continue
        cur = told.find(sent, a)
        last, prior, helped = None, None, re.search(rf"\bhelp(?:s|ed|ing)?\s+(?-i:({names}))\b(?!['’])", sent)
        watchers = set(re.findall(rf"(?-i:\b({names}))\b(?!['’])\s+(?:\w+ly\s+)?(?:watch(?:es|ed)?|looks?\s+on|looked\s+on|stares?|"
                                  rf"stared|waits?|waited|stands?\s+by|stood\s+by)\b", sent, re.I))
        for c in (x.strip() for x in _CLAUSE.split(sent) if x and x.strip()):
            cur = told.find(c, cur)
            lead = re.match(rf"^(?:[^,]{{0,60}},\s+)?\W*(?:{_LEADS}\W+)*(?-i:({names}))\b(?!['’])", c, re.I)
            them = re.match(rf"^(?:[^,]{{0,60}},\s+)?\W*(?:{_LEADS}\W+)*({refer})(?!['’])\s", c, re.I)
            pro = _SHE_LEADS.match(c)
            g = {"she": "f", "he": "m"}.get(pro["pro"].lower(), "") if pro else ""
            pro = pro if g else None
            only = [x for x, s in ones.items() if s in (g, "")] if g and not any(people.get(n, {}).get("sex", "") in (g, "")
                                                                                  for n in cast) else []
            named_doer = re.search(rf"(?-i:\b({names}))\s+(?:\w+ly\s+)?(?:(?:can|could|will|would|must|then)\s+)?(?:{_VERBS}|{_HANDLE})\b",
                                   c, re.I)
            picked = extra(them.group(1)) if them else only[0] if len(only) == 1 else None
            role = picked if picked in cast and not (lead or named_doer) else None
            stranger = None if lead or named_doer or role else as_doer(picked) or (
                latest if pro and latest and latest not in cast and sex_of(latest) == g else None)
            free_ = not (lead or named_doer or pro or role or them)
            role = role or (helped.group(1) if helped and free_ and sent.find(c) > helped.start() else None)
            stranger = prior = None if role else stranger or (prior if free_ else None)
            if lead or role:
                actor = role or lead.group(1)
            elif _SHE_LEADS.match(c) and not stranger:
                free, there = [n for n in cast if not bound(n)], [n for n in cast if re.search(rf"(?-i:\b{re.escape(n)})\b", c)]
                near = [n for n in re.findall(rf"(?-i:\b({names}))\b", sent[:max(0, sent.find(c))])
                        if people.get(n, {}).get("sex", "") in (g, "")]
                sexed = [n for n in cast if people.get(n, {}).get("sex", "") in (g, "")]
                reacts = bool(pro) and bool(re.match(rf"\s+(?:\w+ly\s+)?{_MEEK.pattern}", c[pro.end("pro"):], re.I))
                rest = [meek] if reacts and meek in sexed else [n for n in sexed if n != actor] if pro and (reacts or re.search(
                    rf"\b{'her' if g == 'f' else 'him'}\b", pro["front"] or "", re.I)) else []
                actor = rest[0] if len(rest) == 1 else sexed[0] if g and len(sexed) == 1 else \
                    next(n for n in cast if n not in there) if len(cast) == 2 and len(there) == 1 else \
                    (actor if actor in near else near[-1]) if near else \
                    free[0] if len(cast) == 2 and len(free) == 1 else \
                    next(n for n in cast if n != meek) if len(cast) == 2 and meek == actor and meek else actor
            doer = named_doer.group(1) if named_doer else lead.group(1) if lead else role or stranger or actor
            sure = bool(named_doer or lead or role or stranger)
            main, *sub = re.split(r"\b(?:while|whilst|as)\b", c, maxsplit=1)
            aside = re.match(rf"\s*(?:{_LEADS}\s+)?(?-i:({names}))\b(?!['’])\s", sub[0]) if sub else None
            for part, by, firm in ([] if _TRYING.search(main) else [(main.strip() if aside else c, doer, sure)]) + \
                    ([(sub[0].strip(), aside.group(1), True)] if aside and not _TRYING.search(sub[0]) else []):
                restraint(sent, part, by, firm, base=told.find(part, cur))
                if _GARMENT.search(part) or topic and topic[0] in ("clothes", "theirs") or carry or re.search(
                        r"\b(?:strip|undress|cloth)", part, re.I):
                    clothes(sent, part, by, firm, told.find(part, cur))
            subjects = sorted((_meekness(m.group(2)), k, m.group(1)) for k, m in enumerate(re.finditer(
                rf"\b({names})\s+(?:\w+ly\s+)?(?:can\s+|will\s+|could\s+)?([a-z]+(?:s|ed))\b", c))
                if not re.search(r"\b(?!(?:as|whereas|is|was)\s)[a-z]+(?:s|ed)\s+$", c[:m.start()]))
            acts = bool(_DOES.search(c))
            meek = (subjects[-1][2] if subjects[-1][0] and not acts else None) if subjects else \
                (role or lead.group(1)) if (lead or role) and _MEEK.search(c) and not acts else None if acts and meek == doer else meek
            actor = min(subjects, key=lambda s: (s[0], -s[1]))[2] if subjects else actor
            prev = _spots(_TOOL.sub(" ", c)) or prev
            owned = _named(_TOOL.sub(" ", c), names, _SPOTS) or (None if _spots(_TOOL.sub(" ", c)) else owned)
            seen = _thing(c, seen)
            latest = named_doer.group(1) if named_doer else lead.group(1) if lead else role or stranger or (actor if pro else latest)
            cur, carry = cur + len(c), None
        past += sent + " "
    moved = {}
    for a, b in _sentences(text) if edits else ():
        new = told[a:b]
        for x, y, _, r in sorted((e for e in edits if a <= e[0] < b), key=lambda e: e[:3], reverse=True):
            new = new[:x - a] + r + new[y - a:]
        moved[told[a:b].strip()] = new.strip()
    for x, y, _, r in sorted(edits, key=lambda e: e[:3], reverse=True):
        text = text[:x] + r + text[y:]
    for e in got:
        e[3] = moved.get(e[3], e[3])
    missed = [(moved.get(s, s), why) for s, why in missed]
    final, whole = {}, {i for i, (k, _, x, *_) in enumerate(got) if k == "remove" and not x}
    for i, (k, w, x, *_) in enumerate(got):
        for y in x if k in ("remove", "tear", "lift", "wear") else ():
            final[(w, _stem(pulled(y, {})[0] if k == "lift" else y))] = (i, y)
    for i, (k, w, x, *_) in enumerate(got):
        if k in ("remove", "tear", "lift", "wear"):
            got[i][2] = [y for y in x if final.get((w, _stem(pulled(y, {})[0] if k == "lift" else y))) == (i, y) and not (
                k == "wear" and w in people and worn_rx(y).search(undress(people[w]["desc"], off.get(w, [])))
                and not any(_same(y, z) for z, _ in lifted.get(w, [])))]
    return [(k, f"{w}, {('; ' if k in ('hold', 'have', 'release') else ', ').join(x)}" if x else w, s, d)
            for i, (k, w, x, s, d) in enumerate(got) if x or i in whole], list(dict.fromkeys(missed)), text


def _after(off, worn, lifted, stripped, raised, back, dressed):
    off, worn, lifted = ({k: list(v) for k, v in d.items()} for d in (off, worn, lifted))
    for who, items in stripped.items():
        off.setdefault(who, []).extend(items)
        worn[who] = [w for w in worn.get(who, []) if not any(g == "*" or _same(g, w) for g in items)]
        lifted[who] = [x for x in lifted.get(who, []) if not any(g == "*" or _same(g, x[0]) for g in items)]
    for who, items in raised.items():
        lifted[who] = [x for x in lifted.get(who, []) if not any(_same(g, x[0]) for g, _ in items)] + items
    for who, items in back.items():
        lifted[who] = [x for x in lifted.get(who, []) if x[0] not in items]
    for who, items in dressed.items():
        had = off.get(who, [])
        off[who] = [g for g in had if not any(_same(g, x) for x in items)]
        worn.setdefault(who, []).extend(x for x in items if not any(_same(g, x) for g in had))
    return off, worn, lifted


def plan(prompt, memory="", anchor="", shot_seconds=10.0, from_beat=True, first_frame=False, pictures=(), silence=True):
    people, extra, start = read_memory(memory)
    paras, warn = beats(prompt), []
    scene = [extra] if extra else []
    if not (anchor or "").strip() and len(paras) > 1 and people and not mentions(paras[0]["text"], people) \
            and not _SPEECH.search(paras[0]["text"]):
        first = paras.pop(0)
        scene.append(first["text"])
        for k in STATE_TAGS:
            start[k] += first[k]
    lead = [t for t in [(anchor or "").strip()] + scene if t]
    owner = owners(people, lead + [p["text"] for p in paras])
    belt = _pic_of(r"chastity[\s-]+belts?", [memory or "", anchor or ""] + lead + [p["text"] for p in paras],
                   {v["head"] for v in people.values() if v["head"]})
    have = {k + 1 for k, img in enumerate(pictures) if img is not None}
    held, off, worn, lifted, known = {}, {}, {}, {}, list(people)
    sample = {"hold": "duct tape over her mouth", "wear": "a red dress", "lift": "shirt"}

    def resolve(kind, args, garments=False):
        out = []
        for arg in args:
            who, items = split_tag(arg, people, garments)
            if people and who not in people:
                warn.append(f"{kind}: '{arg}' names {who}, who has no line in character_memory")
            if not who or (kind in sample and not items):
                warn.append(f"{kind}: '{arg}' needs a name and {'what is pulled' if kind == 'lift' else 'what goes on'}, as in "
                            f"'{kind}: Sarah, {sample[kind]}'")
                continue
            out.append((who, items))
            known.extend([who] if who not in known else [])
        return out

    def wearing(who):
        return _on(people[who]["desc"], off.get(who, [])) + [w for w in worn.get(who, [])
                                                             if not any(_same(w, g) for g in off.get(who, []))] \
            if who in people else []

    def absent(n, kind, who, items):
        return [f"shot {n}: {kind}: '{g}' is not in {who}'s description in character_memory, so nothing was "
                f"{'pulled' if kind == 'lift' else 'taken out of it'}; use the word the description uses" for g in items
                if who in people and g != "*" and not _fit(wearing(who), g)]

    for who, items in resolve("hold", start["hold"]):
        held.setdefault(who, []).extend(x for x in items if x not in held.get(who, []))
    for who, items in resolve("wear", start["wear"], "wear"):
        worn.setdefault(who, []).extend(items)
    for who, items in resolve("remove", start["remove"] + start["tear"], "remove"):
        off.setdefault(who, []).extend(items or ["*"])
    for who, items in resolve("lift", start["lift"], "lift"):
        lifted.setdefault(who, []).extend(pulled(x, people.get(who, {})) for x in items)
    def garb(who):
        return sorted((clean(s, False) for s in wearing(who)), key=lambda g: _garb(g)[0])

    def lasts(kind, arg):
        who, items = split_tag(arg, people, _GARB.get(kind, False))
        return [_takes(kind, x, held.get(who, [])) for x in (garb(who) if kind in ("remove", "tear") and (not items or "*" in items)
                                                             else items)]

    shots, present, prev, voice, ceiling, crowd, told, i = [], [], None, None, align(round(float(shot_seconds) * FPS)), {}, "", 0
    for k, q in enumerate(paras):
        q["beat"] = k + 1
    while i < len(paras):
        p = paras[i]
        keyed = (i > 0 and not p["cut"]) or (i == 0 and first_frame)
        tagged = [split_tag(a, people)[0] for k in STATE_TAGS for a in p[k]]
        cast = list(dict.fromkeys(([] if p["cut"] or i == 0 else present) + mentions(p["text"], known) + tagged))
        extras, gone_by = folk(p["text"], people, cast, {**(crowd if keyed else {}), **p.get("extras", {})})
        strangers = {v["label"] if v["count"] == 1 else f"one of {v['label']}": {"sex": v["sex"], "desc": "", "hair": "", "head": None}
                     for v in extras.values()}
        pal = {**people, **strangers}
        story = lead + [q["text"] for q in paras[max([k for k in range(i + 1) if paras[k]["cut"]] + [0]):i]]
        read, missed, p["text"] = (p["read"], p["missed"], p["text"]) if "read" in p else \
            read_beat(p["text"], cast, people, held, off, worn, lifted, paras[i - 1]["text"] if i else "", extras, " ".join(story))
        read, missed = ([], []) if tagged else (read, missed)
        hobbled = [n for n in cast if any("ankles" in parts(x) and not _tied(x) for x in held.get(n, []) + [
            y for a in [e[1] for e in read if e[0] in ("hold", "have")] + p["hold"] for w_, ys in [split_tag(a, people)] if w_ == n
            for y in ys])]
        slow = lambda s: sum(HOBBLE - PER_ACTION for n in hobbled for k, *_ in _footwork(s, n, cast, people)
                             if k in ("walk", "stairs", "hop"))
        split = [] if "read" in p or tagged or p["seconds"] else \
            _parts(p["text"], read, ceiling / FPS - BEAT_BASE, people, lasts, garb, held, slow)
        if split:
            heard = {ln["q"]: (ln["who"], ln["gag"]) for ln in voices(p["text"], cast, people, held, [e[:3] for e in read], voice, extras,
                                                                      paras[i - 1]["text"] if i else "")[0]}
            paras[i:i + 1] = [{**p, **{k: [] for k in STATE_TAGS}, "text": t, "read": r, "missed": missed if not k else [],
                               "cut": p["cut"] and not k, "exit": p["exit"] if k == len(split) - 1 else [], "heard": heard,
                               "unread": p["unread"] if not k else [], "part": (k + 1, len(split)), "extras": extras}
                              for k, (t, r) in enumerate(split)]
            continue
        hands = {}
        for j, (kind, arg, sent, doer) in enumerate(read):
            who, items = split_tag(arg, people, _GARB.get(kind, False))
            if kind == "have":
                held[who] = _put(held.get(who, []), items)
            else:
                p[kind].append(arg)
            for key in items if kind in ("hold", "have", "release") else [_stem(pulled(x, {})[0]) for x in items] or ["*"]:
                hands.setdefault((who, key), (doer, sent, j))

        def hand(who, x, kind):
            key = x if kind in ("hold", "release") else _stem(x[0] if isinstance(x, tuple) else x)
            return hands.get((who, key)) or (hands.get((who, "*")) if kind not in ("hold", "release") else None) or (None, "", len(read))
        warn += [f"shot {i + 1}: " + (f"\"{_cut(s)}\" was not read: {why}" if s else why) for s, why in missed]
        holds, frees = resolve("hold", p["hold"]), resolve("release", p["release"])
        wears, strips = resolve("wear", p["wear"], "wear"), resolve("remove", p["remove"], "remove")
        tears, lifts = resolve("tear", p["tear"], "remove"), resolve("lift", p["lift"], "lift")
        cast = list(dict.fromkeys(cast + [w for w, _ in holds + frees + wears + strips + tears + lifts]))
        leaving = [split_tag(w, people)[0] for w in p["exit"]]
        coming = [n for n in cast if prev is not None and keyed and n not in prev]
        gone = {}
        for who, items in frees:
            gone.setdefault(who, []).extend(x for x in held.get(who, []) if (not items or any(_frees(r, x, held[who]) for r in items))
                                            and x not in gone.get(who, []))
        added = {}
        for who, items in holds:
            new = [x for x in items if x not in held.get(who, []) and x not in added.get(who, [])]
            if new:
                added.setdefault(who, []).extend(new)
        through = {n: [x for x in held.get(n, []) if x not in gone.get(n, []) and not any(_swapped(y, [x]) for y in added.get(n, []))]
                   for n in cast}
        through = {n: v for n, v in through.items() if v}
        stripped, raised, back, dressed, ripped = {}, {}, {}, {}, {}
        for who, items in lifts:
            warn += absent(i + 1, "lift", who, [pulled(x, {})[0] for x in items])
            for g, where in (pulled(x, people.get(who, {})) for x in items):
                fit = _fit(wearing(who), g) if who in people else [g]
                raised.setdefault(who, []).extend([(clean(fit[0], False), where)] if len(fit) == 1 else [])
        for who, items, torn in [(w, x, False) for w, x in strips] + [(w, x, True) for w, x in tears]:
            warn += absent(i + 1, "remove", who, items)
            per, on = people.get(who, {}), sorted(wearing(who), key=lambda g: _garb(clean(g, False))[0])
            items = list(dict.fromkeys(clean(x, False) for g in items or ["*"] for x in (on if g == "*" else _fit(on, g)))) \
                if who in people else items or ["*"]
            way = {} if torn else {g: _stuck(g, through.get(who, []), per) for g in items}
            stuck = [g for g in items if way.get(g)]
            raised.setdefault(who, []).extend(pulled(g, per, way[g]) for g in stuck if not way[g].startswith("on "))
            warn += [f"shot {i + 1}: {_cant(who, [g for g in stuck if way[g] == w], per, w)}" for w in dict.fromkeys(way[g] for g in stuck)]
            stripped.setdefault(who, []).extend(g for g in items if g not in stuck and g not in stripped.get(who, []))
            ripped.setdefault(who, []).extend(g for g in items if torn)
        for who, items in wears:
            up = [g for g, _ in lifted.get(who, [])]
            back.setdefault(who, []).extend(g for g in up if any(_same(x, g) for x in items))
            dressed.setdefault(who, []).extend(x for x in items if not any(_same(x, g) for g in up))
        stripped = {k: v for k, v in stripped.items() if v}
        back, dressed = {k: v for k, v in back.items() if v}, {k: v for k, v in dressed.items() if v}
        off2, worn2, lifted2 = _after(off, worn, lifted, stripped, raised, back, dressed)
        sheets = []
        for n in cast:
            if n not in people:
                continue
            per, desc = people[n], people[n]["desc"]
            hid = hidden(desc, off.get(n, []), worn.get(n, []), lifted.get(n, []))
            look = [undress(desc, off.get(n, []), hid)]
            look += [f"{n} also wears {_owned(w, per)}." for w in worn.get(n, [])]
            ups = lifted.get(n, [])
            look += [f"{cap(pron(per))} {g} {'are' if _many([g]) else 'is'} pulled {where}"
                     f"{_leaving(per, _bares(g, where) - set().union(*(_bares(*u) for u in ups[k + 1:])), desc, off.get(n, []),
                                 worn.get(n, []), ups, held.get(n, []))}."
                     for k, (g, where) in enumerate(ups)]
            vis = [s for s in _on(desc, off.get(n, [])) if s not in hid] + worn.get(n, [])
            look += [f"{side} the waist, {pron(per, 0)} {'wear' if pron(per, 0) == 'they' else 'wears'} only "
                     f"{join([_owned(s, per) for s in vis if zone in _zone(s)])}."
                     for zone, side in (("upper", "Above"), ("lower", "Below"))
                     if any(zone in _zone(g) for g in off.get(n, []) if g != "*") and any(zone in _zone(s) for s in vis)]
            feet = _ankle_item(through.get(n, []))
            look += [_feet_look(feet, per)] if feet else []
            look += [_belt_look(x in through.get(n, []), per, belt) for x in through.get(n, []) + added.get(n, [])
                     if _gear(x) == "a chastity belt"][:1]
            look = [x.strip() for x in look if x and x.strip()]
            text = " ".join(x if k == len(look) - 1 or _ends(x) else f"{x}." for k, x in enumerate(look))
            mine = sorted(k for k, who in owner.items() if who == n)
            pic = per["head"] or (mine[0] if mine and not re.search(_PIC, desc, re.I) else None)
            head = f" <Picture {pic}>" if pic else ""
            sheets += [f"{n}{head}: {text}"] if text or head else []
        refs = [k for k in sorted({int(x) for t in lead + sheets + [p["text"]] for x in re.findall(_PIC, t, re.I)}) if k in have]
        slot = {k: j for j, k in enumerate(refs, 1)}
        shown = {owner[k] for k in refs if k in owner}
        name = labels(cast, people, shown)
        call = lambda n: name.get(n, n)
        events = [(k, a, s) for k, a, s, _ in read] + [(k, f"{w}, {'; '.join(x or held.get(w, []))}", "")
                                                       for k, done in (("hold", holds), ("release", frees)) for w, x in done if tagged]
        nxt = paras[i + 1] if i + 1 < len(paras) and paras[i + 1]["beat"] == p["beat"] else {}
        lines, noises = voices(p["text"], cast, people, held, events, voice, extras, paras[i - 1]["text"] if i else "", nxt.get("text", ""))
        for ln in lines:
            ln["who"], ln["gag"] = p.get("heard", {}).get(ln["q"], (ln["who"], ln["gag"]))
        sounds = sorted(lines + noises, key=lambda ln: (ln["at"], ln["end"]))
        spoken = [ln for ln in lines if not ln["gag"] and not ln["thought"]]
        talkers = list(dict.fromkeys(ln["who"] for ln in spoken if ln["who"]))
        hushed = {ln["who"]: ln["gag"] for ln in sounds if ln["gag"] and not ln["thought"] and ln["who"] not in talkers}
        hairs = [people[n]["hair"] for n in cast if n in people and people[n]["hair"]]
        talker = lambda n: call(n) if call(n) != n or len(cast) < 2 else f"{people[n]['hair']}-haired {n}" \
            if n in people and people[n]["hair"] and hairs.count(people[n]["hair"]) == 1 else n
        told = voiced(p["text"], lines, talker, named(p["text"], read, cast, people, call, extras, told if keyed else "", lines,
                                                     set(through) | set(added)))
        gait = {n: _footwork(told, n, cast, people) for n in cast}
        before = {n: _stance(_footwork(" ".join(story), n, cast, people)) for n in cast}
        notes = []
        if through:
            notes.append(" ".join(f"{cap(call(n))}: {'; '.join(v)}." for n, v in through.items())
                         + " All of it stays on, exactly as described, for the whole shot.")
        moves = [("hold", n, x) for n, v in added.items() for x in v] + [("release", n, x) for n, v in gone.items() for x in v] \
            + [("tear" if g in ripped.get(n, []) else "remove", n, g) for n, v in stripped.items() for g in v] \
            + [("lift", n, u) for n, v in raised.items() for u in v] + [("back", n, g) for n, v in back.items() for g in v] \
            + [("wear", n, g) for n, v in dressed.items() for g in v]
        moves = [(k, n, x, d if d in cast or d in strangers else None, s)
                 for k, n, x in sorted(moves, key=lambda m: hand(m[1], m[2], m[0])[2]) for d, s, _ in [hand(n, x, k)]]
        notes += _doing(moves, pal, call, held, {n: _stance(gait[n]) or before[n] for n in cast}) if moves \
            else ["Everything happens at natural, real-life speed."]
        for n, v in added.items():
            by = list(dict.fromkeys(hand(n, x, "hold")[0] for x in v))
            let = "" if by == [n] else f", and {join(by)} {'have' if len(by) > 1 else 'has'} let go" \
                if all(d and d != n and (d in cast or d in strangers) for d in by) else ", and whoever put it on has let go"
            hid = [x for x in v if arms(x, pron(people.get(n, {}))).startswith("behind")]
            seen = [x for x in v if x not in hid]
            notes.append(f"By the last frame, {call(n)} has " + " and ".join(x for x in (f"{join(seen)} in plain view" if seen else "",
                                                                                        join(hid) if hid else "") if x)
                         + f"{let}.")
        notes += [f"By the last frame, {call(n)} no longer has {join(v)}." for n, v in gone.items() if v]
        bound = {n: set().union(*map(parts, through.get(n, []) + added.get(n, []))) for n in cast
                 if through.get(n) or added.get(n)}
        tied = set().union(*bound.values()) if bound else set()
        for n in cast:
            free = [_FREE[q] for q in _FREE if q in tied and q not in bound.get(n, set())]
            if free:
                notes.append(f"{cap(call(n))}'s " + f" and {pron(people.get(n, {}))} ".join(free) + ".")
        notes += [f"{cap(call(n))} cannot open {pron(people.get(n, {}))} mouth or say a word the whole shot."
                  for n in through if "mouth" in set().union(*map(parts, through[n]))]
        joined = _joined(" ".join([memory or ""] + lead + [q["text"] for q in paras[:i + 1]]))
        for n in dict.fromkeys(list(through) + list(added)):
            per, was = people.get(n, {}), through.get(n, [])
            p_, o_, s_ = pron(per), pron(per, 2), pron(per, 0)
            kept = [x for x in added.get(n, []) if any(_swapped(x, [y]) for y in held.get(n, []))]
            seen = " ".join(story + [told])
            down = [m.end() for m in _PRONE.finditer(seen)]
            up = [m.end() for m in re.finditer(r"\bon(?:to)?\s+(?:her|his|their)\s+back\b|\bface[\s-]?up\b", seen, re.I)]
            posed, behind = _pose(call(n), per, was + kept, [x for x in added.get(n, []) if x not in kept],
                                  [k for k, *_ in gait.get(n, [])], before.get(n), joined, bool(down) and (not up or down[-1] > up[-1]))
            notes += posed
            one = [pron(people.get(m, {}), 0) for m in cast].count(s_) == 1
            subject = rf"(?-i:\b{re.escape(n)})\b(?!['’])" + (rf"|^\W*(?:{_LEADS}\W+)*{s_}\b" if one else "")
            warn += [f"shot {i + 1}: \"{_cut(t)}\" needs {n}'s hands, but {p_} wrists are bound behind {p_} back, so the model "
                     f"may bring them forward to do it; free {o_} first or write it without {p_} hands"
                     for t in re.split(r"(?<=[.!?])\s+|\n+", narration(p["text"]))
                     if behind and any(arms(x, p_).startswith("behind") for x in was)
                     and re.search(rf"(?:{subject})(?:\s+\w+ly)?\s+{_HANDY.pattern}", t, re.I)]
            legs = next((x for x in was if "ankles" in parts(x) and not re.search(r"\b(?:left|right) ankle\b|spread apart", x)
                         and _gear(x) != "a spreader bar"), None)
            fixed = (_tied(legs) or {}).get("to") if legs else None
            warn += [f"shot {i + 1}: \"{_cut(t)}\" needs {n}'s legs apart, but {p_} ankles are bound, so the model may break the "
                     f"restraint or draw an extra leg; free {p_} ankles first or write it without that" if k == "wide" else
                     f"shot {i + 1}: \"{_cut(t)}\" needs {n} to walk, but {p_} ankles are fastened to {fixed}, so the model may break "
                     f"the restraint or draw an extra leg; free {p_} ankles first or write it without that"
                     for k, t in dict.fromkeys((k, t) for k, t, *_ in gait.get(n, []))
                     if legs and (k == "wide" or fixed and k in ("walk", "stairs", "hop"))]
        for n in cast:
            if n not in people:
                continue
            per, desc = people[n], people[n]["desc"]
            p_, o_ = pron(per), pron(per, 2)
            took, up, put = stripped.get(n, []), raised.get(n, []), back.get(n, [])
            wrists = next((_held_at(x, p_) for x in through.get(n, []) + added.get(n, []) if arms(x, p_) or _tied(x)
                           and "wrists" in parts(x)), "")
            bonds, ends = through.get(n, []), through.get(n, []) + added.get(n, [])
            if took:
                notes.append(f"By the last frame, {call(n)}'s {'clothes' if '*' in took else join(took)} "
                             f"{'are' if '*' in took or _many(took) else 'is'} "
                             f"{'torn ' if set(took) <= set(ripped.get(n, [])) else ''}off.")
            for k, (g, where) in enumerate(up):
                one = not _many([g])
                tie = (f"; {p_} wrists stay {wrists} the whole time {'it moves' if one else 'they move'}, and "
                       f"{'it never goes' if one else 'they never go'} over {p_} head or down {p_} arms") \
                    if wrists and "upper" in _zone(g) else ""
                leave = _leaving(per, _bares(g, where) - set().union(*(_bares(*u) for u in up[k + 1:])), desc, off2.get(n, []),
                                 worn2.get(n, []), lifted2.get(n, []), ends)
                notes.append(f"By the last frame, {call(n)}'s {g} {'is' if one else 'are'} pulled {where} and "
                             f"{'stays' if one else 'stay'} there, still on {o_}{leave}{tie}.")
            was = hidden(desc, off.get(n, []), worn.get(n, []), lifted.get(n, []))
            now = hidden(desc, off2.get(n, []), worn2.get(n, []), lifted2.get(n, []))
            shown_now = [clean(s) for s in was if s not in now and s in _on(desc, off2.get(n, []))
                         and not any(worn_rx(g).search(s) for g, _ in up)]
            if shown_now:
                moved = took + [g for g, _ in up]
                notes.append(f"Under {'them' if '*' in took or _many(moved) else 'it'}, {call(n)} has {join(shown_now)} on, "
                             f"now in plain view.")
            if took:
                many = "*" in took or _many(took)
                after = bare(desc, off2.get(n, []), worn2.get(n, []))
                notes += [f"Once {'they are' if many else 'it is'} off, {call(n)} is {after} for the rest of the shot"
                          f"{_except(bonds, _NAKED[after])}."] \
                    if after and after != bare(desc, off.get(n, []), worn.get(n, [])) else []
                locked = [x for x in through.get(n, []) if parts(x) & {"wrists", "ankles"}]
                notes += [f"{cap(call(n))}'s {join(locked)} stay locked the whole time {p_} "
                          f"{'clothes' if '*' in took else join(took)} {'come' if many else 'comes'} off: nothing on {p_} wrists "
                          f"or ankles is unlocked, untied or loosened."] if locked else []
                notes += [f"{cap(call(n))}'s chastity belt stays locked on {o_} the whole time {p_} "
                          f"{'clothes' if '*' in took else join(took)} {'come' if many else 'comes'} off."
                          for x in through.get(n, []) if _gear(x) == "a chastity belt"][:1]
            notes += [f"{cap(call(n))}'s {g} {'stay' if _many([g]) else 'stays'} pulled {where} the whole shot."
                      for g, where in lifted.get(n, []) if (g, where) in lifted2.get(n, [])]
            notes += [f"By the last frame, {call(n)}'s {g} {'are' if _many([g]) else 'is'} pulled back into place." for g in put]
            pic = next((slot[k] for k in refs if owner.get(k) == n), None)
            notes += [f"<Picture {pic}> is for {call(n)}'s face and hair only: {p_} clothes are as this shot describes them."] \
                if pic and (took or up or put or off.get(n) or worn.get(n) or lifted.get(n) or dressed.get(n)) else []
            if dressed.get(n):
                notes.append(f"By the last frame, {call(n)} wears {join(dressed[n])}.")
            state = bare(desc, off.get(n, []), worn.get(n, []))
            late = [re.sub(r"^(?:a|an)\s+", "", x) for x in added.get(n, []) if state and _spans(x) & _NAKED[state]]
            notes += [f"{cap(call(n))} is {state} the whole shot{_except(bonds, _NAKED[state])}" + (
                f"{' and' if _except(bonds, _NAKED[state]) else ', apart from'} the {join(late)} once "
                f"{'they are' if _many([_gear(x) for x in late]) else 'it is'} on" if late else "") + "."] if state and not dressed.get(n) else []
        def own(n):
            m, p_ = _VOICE.search(pal.get(n, {}).get("desc", "")), pron(pal.get(n, {}))
            return f"{p_} own {m.group(2).strip(' ,')} voice" if m and m.group(2) else \
                f"{p_} own voice, {m.group(1).strip()}" if m else f"{p_} own voice"
        marks = [(a, b, t) for a, b, t, _ in _pace(p["text"], read, lasts, slow)]
        act = sum(t for _, _, t in marks) + (sum(t for k in STATE_TAGS for a in p[k] for t in lasts(k, a)) if tagged else 0)
        cues = timeline(sounds, marks)
        talk = cues[-1][1] + SPEECH_TAIL if cues else 0.0
        same = len({pal.get(n, {}).get("sex", "") for n in talkers}) == 1
        one = pal.get(talkers[0], {}) if talkers else {}
        unseen = {v["label"] for v in extras.values() if v["kind"] == "heard"}
        live = "with natural, expressive intonation and real emotion, the way people really talk"
        quiet = [n for n in cast if n not in talkers and n not in hushed]
        hush = f" {cap(join([talker(n) for n in quiet]))} {'listens' if len(quiet) == 1 else 'listen'} with closed lips." if quiet else ""
        notes += [f"Only {talker(talkers[0])} speaks in this shot, in {own(talkers[0])}, {live}: nobody in the shot moves their lips "
                  f"with {'it' if len(spoken) == 1 else 'them'}."] if len(talkers) == 1 and talkers[0] in unseen else \
            [f"Only {talker(talkers[0])} speaks in this shot, in {own(talkers[0])}, {live}: {pron(one)} lips move only while "
             f"{pron(one, 0)} {'says' if pron(one, 0) != 'they' else 'say'} {'the line' if len(spoken) == 1 else 'the lines'}, "
             f"and every word of {'it' if len(spoken) == 1 else 'them'} is "
             f"{ {'her': 'hers', 'his': 'his'}.get(pron(one, 1), 'theirs')}.{hush}"] if len(talkers) == 1 else \
            [f"{cap(join([talker(n) for n in talkers]))} speak in this shot, each in "
             f"{pron(pal.get(talkers[0], {})) if same else 'their'} own voice, {live}, and each saying only "
             f"{pron(pal.get(talkers[0], {})) if same else 'their'} own lines.{hush}"] if talkers else []
        notes += [f"{cap(talker(n))} makes only a muffled, closed-mouth mmmph through the {gagword(g)}, with no words."
                  for n, g in hushed.items()]
        heard = [x for x in ("the line" if len(spoken) == 1 else "the lines" if spoken else "",
                             "the muffled sounds" if hushed else "") if x]
        scene = " ".join(lead + [p["text"]])
        music, vocal = bool(_MUSIC.search(scene)), bool(_VOCAL.search(scene))
        notes.append(f"Apart from {' and '.join(heard)}, everyone stays silent." if heard else _WORDLESS if vocal else _HUSH)
        notes += [] if music else [_SOUND]
        xs = [v for v in extras.values() if v["here"] and not v["where"]]
        apart = [v for v in extras.values() if v["here"] and v["where"]]
        counts = [None if v["noun"] == "someone" or not v["here"] else v["count"] for v in extras.values() if not v["where"]]
        total = len(cast) + sum(c or 0 for c in counts)
        if people and not total and not counts and not apart:
            notes.append("Nobody is in the shot.")
        elif total and all(counts):
            notes.append("There is one person in the shot: one body, one face." if total == 1 else
                         f"There are {_COUNT.get(total, total)} people in the shot, with one body for each person"
                         + (f": {join([call(n) for n in cast] + [v['label'] for v in xs])}." if xs else "."))
        named_ = [call(n) for n in cast]
        for v in apart:
            be, x0, x1 = ("is", pron(v, 0), pron(v)) if v["count"] == 1 and pron(v, 0) != "they" else ("are", "they", "their")
            apart_from = f", a different {'person' if be == 'is' else 'people'} from {join(named_)} with " \
                         f"{'a face' if be == 'is' else 'faces'} of {x1} own" if named_ else ""
            notes.append(f"{cap(v['label'])} {be} only a voice{apart_from}: {x0} {be} never seen in the shot." if v["kind"] == "heard" else
                         f"{cap(v['label'])} {be} only a picture{apart_from}: {x0} {be} not in the room and never "
                         f"{'moves' if be == 'is' else 'move'}." if v["kind"] == "still" else
                         f"{cap(v['label'])} {be} only seen there{apart_from}: {x0} {be} not in the room and never "
                         f"{'appears' if be == 'is' else 'appear'} anywhere else in the shot"
                         + (f", and {join(named_)} {'is' if len(named_) == 1 else 'are'} in the room, never on "
                            f"{v['where'].split(' ', 1)[1]}." if named_ else "."))
        if cast and xs:
            one = len(xs) == 1 and xs[0]["count"] == 1
            x0, x1, x2 = (pron(xs[0], 0), pron(xs[0]), pron(xs[0], 2)) if one else ("they", "their", "them")
            sing = one and x0 != "they"
            whose = " or ".join(n + "'s" for n in named_)
            notes.append(f"{cap(join([v['label'] for v in xs]))} {'is a different person' if one else 'are different people'} from "
                         f"{join(named_)}: {x0} {'moves' if sing else 'move'} only as this shot describes, "
                         f"{'in ' + x1 if one else 'each in their'} own way and at {x1} own pace, and never "
                         f"{'copies' if sing else 'copy'} {whose} movements, actions or words; nothing done to "
                         f"{' or '.join(named_)} is done to {x2}, and nothing done to {x2} is done to {' or '.join(named_)}.")
        staying = [n for n in cast if n not in leaving]
        if staying and not _CAMERA.search(p["text"]):
            faces = f"{call(staying[0])}'s face" if len(staying) == 1 else "the faces of " + join([call(n) for n in staying])
            feet = [(n, y) for n in staying for y in [_ankle_item(through.get(n, []) + added.get(n, []) + gone.get(n, []))] if y]
            low = list(dict.fromkeys(_low(y, pron(people.get(n, {})) if len(feet) == 1 else "their") for n, y in feet))
            far = "" if not feet else ", and stays far enough back to keep all of " + (
                pron(people.get(feet[0][0], {}), 2) if len(feet) == 1 == len(staying) else join([call(n) for n, _ in feet])) + \
                " in frame too, down to " + (low[0] if len(low) == 1 else join([_low(y, f"{call(n)}'s") for n, y in feet]))
            notes.append(f"The camera keeps {faces} in frame for the whole shot{far}.")
        notes += [f"{cap(call(n))} leaves during this shot." for n in leaving if n in cast]
        if keyed:
            k = len(refs) + 1
            if i == 0:
                notes.append(f"<Picture {k}> is the frame this shot opens on.")
            elif coming:
                there = [call(n) for n in cast if n not in coming]
                notes.append(f"<Picture {k}> is the frame this shot opens on: the same place one moment earlier"
                             + (f", with only {join(there)} in view" if there else "") + f"; {join([call(n) for n in coming])} "
                             f"{'comes' if len(coming) == 1 else 'come'} into view during this shot.")
            else:
                notes.append(f"<Picture {k}> is the frame this shot opens on: the same place and the same people one moment "
                             f"earlier, carried forward.")
        hob = {n: (y, 0 if y in through.get(n, []) else _since(told, hand(n, y, "hold")[1]))
               for n in cast for y in [_ankle_item(through.get(n, [])) or _ankle_item(added.get(n, []))]
               if y and not _tied(y) and not re.search(r"\b(?:left|right) ankle\b", y)}
        shown_beat, swaps = _legs(told, gait, hob, call, cast, people, before)
        warn += [f"shot {i + 1}: \"{a}\" became \"{b}\": {n}'s ankles are bound together, so both legs move together"
                 for n, a, b in swaps]
        text = "\n\n".join(t for t in [retag(x, slot) for x in lead] + ["\n".join(retag(x, slot) for x in sheets)]
                           + [retag(shown_beat, slot), " ".join(notes)] if t)
        changes = bool(holds or frees or wears or strips or tears or lifts)
        if p["seconds"]:
            frames, fit = align(round(p["seconds"] * FPS)), ""
        elif from_beat:
            need = max(BEAT_BASE + act, talk) + (SETTLE if changes else 0.0)
            frames, fit = min(max(MIN_FRAMES, align(round(need * FPS), nearest=True)), ceiling), ""
        else:
            frames, fit = ceiling, ""
        over = talk - frames / FPS
        if over > 0:
            cues = [(round(s, 2), round(s + b - a, 2), k) for (a, b, k), (t, _, _) in zip(cues, timeline(sounds, packed=True))
                    for s in [max(t, a - over)]]
            talk = cues[-1][1] + SPEECH_TAIL
            if round(talk * FPS) > frames and not p["seconds"]:
                frames, fit = align(round(talk * FPS)), "lengthened to fit its line"
            cues = [(a, min(b, round(frames / FPS, 2)), k) for a, b, k in cues if a < frames / FPS]
        warn += [f"shot {i + 1}: what happens in it takes about {BEAT_BASE + act:.0f}s at real-life speed, more than its "
                 f"{frames / FPS:.1f}s, so it may look rushed; {'give it more seconds:' if p['seconds'] else 'raise shot_seconds'}"] \
            if act > (frames / FPS - BEAT_BASE) * RUSH else []
        for who, items in gone.items():
            held[who] = [x for x in held.get(who, []) if x not in items]
        for who, items in added.items():
            held[who] = _put(held.get(who, []), items)
        for who in held:
            held[who] = _settle(held[who])
        off, worn, lifted = off2, worn2, lifted2
        listed = lambda state: " ".join(f"{n}: {'all' if v == ['*'] else '; '.join(v)}." for n, v in state.items() if v)
        spans = iter(cues)
        timed = [(ln, None if ln["thought"] else next(spans, None)) for ln in sounds]
        bits = [f"shot {i + 1}: {frames} frames ({frames / FPS:.1f}s)"] + [b for b in (
            fit, "after a cut" * p["cut"], f"beat {p['beat']}, part {p['part'][0]} of {p['part'][1]}" if "part" in p else "") if b]
        bits += ["read from the beat: " + "; ".join(f"{_READ.get(k, k)}: {a} (\"{_cut(s, 60)}\")" for k, a, s, _ in read)] \
            if read else []
        bits += [f"{label}: {text}" for label, text in (("cast", ", ".join(cast)), ("on", listed(through)),
                                                       ("goes on", listed(added)), ("comes off", listed(gone)),
                                                       ("clothes off", listed(stripped)),
                                                       ("pulled", listed({n: [f"{g} {w}" for g, w in v] for n, v in raised.items()})),
                                                       ("pulled back", listed(back)), ("puts on", listed(dressed)),
                                                       ("leaves", ", ".join(leaving)),
                                                       ("unnamed", ", ".join(v["label"] for v in extras.values())),
                                                       ("speaks", "; ".join(f"{ln['who'] or '?'}{' (thinks)' if ln['thought'] else ''}"
                                                                            f"{' (muffled)' if ln['gag'] and not ln['thought'] else ''}"
                                                                            f"{f' {t[0]:.1f}-{t[1]:.1f}s' if t else ''}: "
                                                                            f"\"{_cut(ln['q'] or ln['verb'], 40)}\"" for ln, t in timed)),
                                                       ("pictures", ", ".join(f"ref_image_{k}" for k in refs))) if text]
        shots.append({"n": i + 1, "beat": p["beat"], "prompt": text, "frames": frames, "keyed": keyed, "cut": p["cut"], "cast": cast,
                      "refs": [k - 1 for k in refs], "faces": [k in owner for k in refs],
                      "silent": silence and not any(k == "line" for _, _, k in cues), "cues": cues, "changed": changes, "coming": coming,
                      "hush": "\n\n".join([y for y in (re.sub(_PIC, "", x, flags=re.I).strip() for x in lead) if y]
                                           + [_HUSH + ("" if music else " " + _SOUND)]) if cues else "",
                      "mute": (silence or bool(cues)) and not (music or vocal and not cues or _TONES.search(scene)),
                      "line": ", ".join(bits), "read": [(k, a) for k, a, *_ in read],
                      "said": [(ln["who"], ln["q"], bool(ln["gag"])) for ln in lines if not ln["thought"]]})
        warn += [f"shot {i + 1}: '{u}' was not read" for u in p["unread"]]
        warn += [f"shot {i + 1}: the line \"{_cut(ln['q'], 40)}\" has no speaker; name who says it, as in '\"…,\" "
                 f"{cast[0] if cast else 'Dana'} says'" for ln in lines if not ln["who"]]
        voice = next((ln["who"] for ln in reversed(lines) if ln["who"]), voice)
        present, prev, crowd = [n for n in cast if n not in leaving], cast, {k: v for k, v in extras.items() if k not in gone_by}
        i += 1
    return shots, warn
