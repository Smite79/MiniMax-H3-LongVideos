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
GEAR_TAGS = ("hold", "release", "wear", "remove", "tear", "lift")
MARK_TAGS = ("mark", "unmark")
STATE_TAGS = GEAR_TAGS + MARK_TAGS
_TAG = re.compile(r"^\s*(hold|release|wear|remove|tear|lift|mark|unmark|pull|exit|seconds)\s*:\s*(.*?)\s*$", re.I)
_INLINE = re.compile(r"(?<=[.!?\"”])\s+(?=(?:hold|release|wear|remove|tear|lift|mark|unmark|pull|exit|seconds)\s*:)", re.I)
_CUT = re.compile(r"^\s*cut\s*[:.!]?\s*$", re.I)
_HEAD = re.compile(rf"^\s*(?:[-*•>#]+\s*|\d+[.)]\s+)?(?:\*\*|__)?(?P<name>{NAME}|[{_UP}]{{2,}}(?:\s+[{_UP}]{{2,}}){{0,2}})"
                   rf"(?:\*\*|__)?(?:\s*\([^)<]*\))?(?:\s*,?\s*\(?\s*(?i:<\s*picture[\s_\-]*(?P<pic>\d+)\s*>)\s*\)?)?(?:\s*\([^)<]*\))?"
                   rf"(?:\*\*|__)?\s*(?P<sep>[:=]|[-–—]\s)?\s*(?P<rest>.*)$")
_TAGGED = re.compile(r"(\(\s*)?<\s*picture[\s_\-]*(\d+)\s*>(\s*\))?", re.I)
_SPEECH = re.compile(r"\"[^\"]*\"|“[^”]*”|<\s*d\s*>.*?<\s*/\s*d\s*>"
                     r"|(?<![\w'’])['‘](?=[^\s'‘’])(?:[^'‘’\n]|(?<=\w)['’](?=\w))+?(?<=[^\s'‘’])['’](?![\w'’])", re.S | re.I)
_INKED = re.compile(r"«[^»]*»")
_WROTE = r"writ(?:e|es|ing)|wrote|scrawl(?:s|ed|ing)?|scribbl(?:e|es|ed|ing)|inscrib(?:e|es|ed|ing)|etch(?:es|ed|ing)?"
_DRAWN = r"stamp(?:s|ed|ing)?|print(?:s|ed|ing)?|spell(?:s|ed|ing)?|ink(?:s|ed|ing)?|paint(?:s|ed|ing)?|draw(?:s|ing)?|drew|" \
         r"mark(?:s|ed|ing)?"
_SKIN = r"foreheads?|cheeks?|faces?|necks?|throats?|collarbones?|shoulder\s+blades?|shoulders?|chests?|breasts?|stomachs?|bellies|belly|" \
        r"abdomens?|ribs|ribcage|sides?|backs?|hips?|pelvis|buttocks?|butt|ass|bottoms?|thighs?|legs?|calves|calf|knees?|ankles?|feet|" \
        r"foot|forearms?|arms?|biceps?|wrists?|hands?|palms?|fingers?|body|skin"
_ON = r"on|onto|across|all\s+over|over|along|down|around|into|against|to"
_TOOL_INK = re.compile(r"\b(?:with|using|in)\s+(?:a|an|the|some|her|his|their)?\s*(?:[\w-]+\s+){0,2}?"
                       r"(?:marker|pen|sharpie|lipstick|lip\s*liner|eyeliner|paint|ink|chalk|crayon|pencil|brush|charcoal|blood|glitter)\b",
                       re.I)
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
_KIND = (("mouth", r"gag|muzzle|tape"), ("eyes", r"blindfold"),
         ("wrists", r"(?:hand)?cuffs|manacles|zip\s*ties?|rope(?!\s+leash)|chains?(?!\s+leash)"),
         ("ankles", r"shackles|leg\s+irons"), ("neck", r"collar|leash"))
_FREE = {"mouth": "mouth is empty and uncovered", "eyes": "eyes are uncovered", "wrists": "wrists are unbound", "neck": "neck is bare"}
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
            r"(?:step\s?)?ladders?|rungs?|sawhorses?|trestles?|" \
            r"columns?|pipes?|radiators?|bars?|railings?|rails?|trees?|doors?|doorknobs?|bolts?|sinks?|toilets?|fences?|gates?|" \
            r"banisters?|cages?|stocks|(?:arm)?chairs?|stools?|armrests?"
_SIDES = r"corners?|legs?|sides?|ends?|edges?|arms?|backs?|foot|feet|heads?|tops?"
_BEDPART = re.compile(r"\b(?:bed\s?posts?|head\s?boards?|foot\s?boards?|bed\s?frames?|bed\s?rails?)\b", re.I)
_SEAT = re.compile(r"\b(?:(?:arm)?chairs?|stools?)$", re.I)
_ARMREST = re.compile(r"\barm(?:rest)?s?\b", re.I)
_LIE_ON = re.compile(r"\b(?:tables?|beds?|benches?|cots?|gurneys?|stretchers?|altars?|slabs?|mattress(?:es)?|racks?)\b", re.I)
_FLOOR = re.compile(r"\b(?:floors?|ground|desks?|workbench(?:es)?|counters?)\b", re.I)
_SURFACE = r"beds?|floors?|ground|mattress(?:es)?|rugs?|carpets?|couch(?:es)?|sofas?|tables?|benches?|cots?|desks?|counters?|altars?"
_OVERHEAD = re.compile(r"\b(?:ceilings?|beams?|rafters?)\b", re.I)
_ANCHOR = re.compile(rf"\b(?P<prep>to|onto|on|across|from)\s+(?P<a>(?:the|a|an|her|his|their)\s+(?:[\w-]+\s+){{0,2}}?(?:{_FIXTURES})\b"
                     rf"(?:\s+(?:{_SIDES})\b)?(?:\s+(?:of|on|in|at)\s+(?:the|a|an)\s+(?:[\w-]+\s+)?(?:{_FIXTURES})\b)?|"
                     rf"(?:the|its|each|both|all\s+four|four|two)\s+(?:[\w-]+\s+)?(?:{_SIDES})\s+of\s+(?:the|a|an)\s+"
                     rf"(?:[\w-]+\s+of\s+(?:the|a|an)\s+)?(?:[\w-]+\s+)?(?:{_FIXTURES})\b|"
                     rf"(?:its|each|both|all\s+four|four|two)\s+(?:[\w-]+\s+)?(?:{_SIDES})\b|it\b|them\b)", re.I)
_TIED_TO = re.compile(r",\s+(?:locked|chained|tied|taped|strapped|fastened)\s+(?P<lie>face\s+down\s+|down\s+)?to\s+(?P<to>.+?)"
                      r"(?P<link>\s+by\s+a\s+separate\s+short\s+(?P<strand>chain|rope|strap|zip tie|strip of tape))?\s*$", re.I)
_STRAND = {"rope": "rope", "straps": "strap", "zip ties": "zip tie", "tape": "strip of tape", "duct tape": "strip of tape"}
_JOINS = {"chain": ("padlocks", "unlocks"), "rope": ("ties", "unties"), "strap": ("buckles", "unbuckles"), "zip tie": ("fastens", "cuts"),
          "strip of tape": ("runs", "peels off")}
_HELD_BY = re.compile(r",\s+its\s+other\s+end\s+held\s+by\s+(?P<by>[^,;]+?)\s*$", re.I)
_TAUT = re.compile(r"\b(?:strain|pull|tug|yank|jerk|lung|struggl|tighten|shorten)\w*\s+(?:\w+\s+){0,2}?(?:against|at|on|in)\s+(?:the|her|"
                   r"his|their|[\w'’-]+['’]s)\s+(?:\w+\s+)?(?:chain|leash)\b|\b(?:chain|leash)\s+(?:\w+\s+){0,3}?(?:taut|tight)\b|"
                   r"\b(?:at|to)\s+the\s+(?:full\s+)?end\s+of\s+(?:the|her|his|their)\s+(?:chain|leash)\b|\bas\s+far\s+as\s+(?:the|her|"
                   r"his|their)\s+(?:chain|leash)", re.I)
_LTAKE = r"holds?|held|grabs?|grabbed|grips?|gripped|seizes?|seized|catch(?:es)?|caught|takes?(?:\s+hold\s+of)?|took(?:\s+hold\s+of)?|" \
         r"picks?\s+up|picked\s+up|keeps?\s+hold\s+of|kept\s+hold\s+of|wraps?|wrapped|winds?|wound|loops?|looped|coils?|coiled|" \
         r"(?:tugs?|tugged|pulls?|pulled|yanks?|yanked|jerks?|jerked)(?:\s+(?:on|at))?|shortens?|shortened|gathers?\s+up|gathered\s+up"
_LBY = r"leads?|led|pulls?|pulled|drags?|dragged|tugs?|tugged|walks?|walked|guides?|guided|hauls?|hauled|yanks?|yanked|jerks?|" \
       r"jerked|draws?|drew|steers?|steered|parades?|paraded|takes?|took|holds?|held|keeps?|kept|brings?|brought"
_LGIVE = r"hands?|handed|gives?|gave|passes?|passed|tosses?|tossed|throws?|threw"
_LDROP = r"drops?|dropped|lets?\s+go\s+of|let\s+go\s+of|releases?|released"
_LFREE = r"unties|untied|unhooks?|unhooked|unclips?|unclipped|unfastens?|unfastened|unlocks?|unlocked|unchains?|unchained|detaches|" \
         r"detached|takes?|took|lifts?|lifted|pulls?|pulled"
_LFASTEN = re.compile(r"\b(?:ties|tied|loops|looped|wraps|wrapped|winds|wound|hooks|hooked|fastens|fastened|clips|clipped|locks|locked|"
                      r"padlocks|padlocked|chains|chained|secures|secured|attaches|attached|knots|knotted|snaps|snapped)\b", re.I)
_LHAND = re.compile(r"\b(?:around|round|in|into)\s+(?:his|her|their|(?-i:[A-Z])[\w'’-]*['’]s)\s+(?:own\s+)?(?:left\s+|right\s+)?"
                    r"(?:hands?|fists?|palms?|wrists?|knuckles)\b", re.I)
_WRISTS = r"(?:her|his|their) (?:hand|wrist )cuffs|the [\w ]+? around (?:her|his|their) wrists"
_JOINED = "joined to each other by their own three short, rigid steel links, which hold them together, side by side, every moment"
_ONE_EACH = "holding each piece in one hand and {} it with the other"
_CHORES = (("washing", r"wash(?:es|ed|ing)?|(?:does|do|doing|did|done)\s+the\s+(?:dishes|washing[\s-]up)", _ONE_EACH.format("scrubbing")),
           ("scrubbing", r"scrub(?:s|bed|bing)?", _ONE_EACH.format("scrubbing")),
           ("rinsing", r"rins(?:e|es|ed|ing)", "turning each piece under the water with both hands"),
           ("drying", r"dr(?:ies|ied|ying)|dry(?=\s+(?:the|a|an|each|every|it|them|off|up)\b)",
            "holding each piece in one hand and drying it with a cloth in the other"),
           ("wiping", r"wip(?:e|es|ed|ing)", "steadying things with one hand and wiping with the other"),
           ("polishing", r"polish(?:es|ed|ing)?", _ONE_EACH.format("polishing")),
           ("folding", r"fold(?:s|ed|ing)?", "taking each piece in both hands to fold it"),
           ("kneading", r"knead(?:s|ed|ing)?", "pressing and turning the dough with both hands"),
           ("chopping", r"chop(?:s|ped|ping)?", "holding the food steady with one hand and the knife in the other"),
           ("peeling", r"peel(?:s|ed|ing)?(?!\s+(?:[\w'’-]+\s+){0,3}?(?:off|away|back)\b)", _ONE_EACH.format("peeling")),
           ("sweeping", r"sweep(?:s|ing)?|swept", "both hands on the broom handle"),
           ("mopping", r"mop(?:s|ped|ping)?", "both hands on the mop handle"),
           ("wringing", r"wring(?:s|ing)?|wrung", "twisting it with both hands"),
           ("ironing", r"ironing|iron(?:s|ed)(?=\s+(?:the|a|an|his|her|their|it|them)\b)",
            "the iron in one hand while the other smooths the cloth"),
           ("typing", r"typ(?:es|ed|ing)|type(?=\s+(?:a|an|the|it|out|on|in)\b)", "both hands on the keys"))
_SPARE = re.compile(r"\b(?:(?P<soft>[\w-]+)\s+)?(?P<n>handcuffs|cuffs|manacles|shackles|leg\s+irons|fetters)\b(?!\s+of\b)", re.I)
_SHAFT = r"cock|dick|penis|shaft|erection|manhood"
_TOY = r"strap-?on(?:\s+dildo)?|dildo|vibrator|vibe|(?:butt\s+|anal\s+)?plug|(?:anal\s+)?beads|toy"
_HOLE = r"pussy|vagina|cunt|slit|(?:ass|arse|butt)(?:hole)?|anus|backside|rear"
_REAR = re.compile(r"(?:ass|arse|butt|anus|backside|rear)", re.I)
_FUCK = r"fuck(?:s|ed|ing)?|penetrat(?:es|ed|ing)|screw(?:s|ed|ing)|pound(?:s|ed|ing)|enter(?:s|ed|ing)|mount(?:s|ed|ing)|" \
        r"finger(?:s|ed|ing)|(?:has|have|had|having)\s+sex\s+with|mak(?:es|ing)\s+love\s+to|made\s+love\s+to|tak(?:es|ing)|took"
_DRIVE = r"thrust(?:s|ing)?|push(?:es|ed|ing)?|slid(?:es?|ing)?|slips?|slipped|slipping|sinks?|sank|sinking|drives?|drove|driving|" \
         r"plung(?:es|ed|ing)|rams?|rammed|ramming|eas(?:es|ed|ing)|guid(?:es|ed|ing)|forc(?:es|ed|ing)|works?|worked|working|" \
         r"bur(?:ies|ied|ying)|insert(?:s|ed|ing)?|lower(?:s|ed|ing)?|settl(?:es|ed|ing)|sits?|sat|sitting|impal(?:es|ed|ing)|" \
         r"slam(?:s|med|ming)?|bounc(?:es|ed|ing)|grind(?:s|ing)?"
_WAYS = (("side", r"spoon\w*|on\s+(?:their|her|his)\s+sides?"),
         ("bent", r"doggy(?:[\s-]?style)?|bent\s+over|bend(?:s|ing)?\s+(?:[\w'’-]+\s+)?over|face[\s-]?down|prone|"
                  r"on(?:to)?\s+(?:all\s+fours|(?:her|his|their)\s+hands\s+and\s+knees)"), ("behind", r"from\s+behind"),
         ("astride", r"straddl\w*|astride|cowgirl"),
         ("top", r"missionary|(?:climbs?|climbed|lies|lay)\s+on\s+top|"
                 r"between\s+(?:her|his|their|[\w-]+['’]s)\s+(?:spread\s+)?(?:legs|thighs)"),
         ("standing", r"against\s+(?:the|a)\s+wall|lifts?\s+(?:her|him)\s+(?:up|off)"))
_LAID = {"lie": ("top",), "prone": ("bent", "behind"), "bend": ("bent", "behind"), "fours": ("bent", "behind"), "kneel": ("behind",)}
_GOING = re.compile(r"\b(?:thrust\w*|pound\w*|pump\w*|faster|harder|deeper|(?:keeps?|kept)\s+(?:going|fucking|thrusting|riding|"
                    r"pumping|moving)|rid(?:es|ing)|bounc\w*|grind\w*|in\s+and\s+out)\b", re.I)
_PULLOUT = re.compile(rf"\b(?:pulls?|pulled|slides?|slid|slips?|slipped|takes?|took|draws?|drew)\s+(?:(?:the|his|her|their|it)\s+)?"
                      rf"(?:(?:[\w-]+\s+)?(?:{_SHAFT}|{_TOY}|fingers?)\s+)?out\b|\bwithdr[ae]w\w*|"
                      rf"\b(?:climbs?|climbed|gets?|got|rolls?|rolled)\s+off\b", re.I)
_APART = re.compile(r"\b(?:step|mov|back|pull|walk|turn|spin|spun|break|broke|draw|drew|slip|wriggl|twist)\w*\s+(?:\w+\s+){0,2}?"
                    r"(?:away|back|off|around|round|aside|free)\b(?!\s+(?:against|to|into|onto|toward)\b)|\blets?\s+(?:\w+\s+)?go\b|"
                    r"\breleas\w*|\bleav(?:es|ing)\b|\bexit\w*|\bfac(?:e|es|ing)\s+(?:him|her|them|each\s+other|(?-i:[A-Z]))", re.I)
_ALONE = r"(?!\s+(?!(?:\w+ly|and|as|while|with|from|again|deep|deeper|hard|harder|until|then|but|so|for|before|after|over|on|in|at|" \
         r"to|up|down|onto|into|inside|doggy|anally|against|across|beside|near|under|beneath|standing|there|here|now)\b)[a-z]+)"
_SURF = r"wall|door|counter(?:top)?|table|desk|fridge|refrigerator|post|pole|pillar|column|tree|car|cabinet|cupboard|wardrobe|" \
        r"book(?:shelf|case)|shel(?:f|ves)|sink|bar|railing|headboard|mirror|window|locker|stove|dresser"
_FACING = re.compile(r"\bfrom\s+behind\b|\bface[\s-]?(?:first|forward)\b|\bfac(?:e|es|ed|ing)\s+(?:the|a|an)\b|\b(?:her|his|their|"
                     r"[\w-]+['’]s)\s+(?:face|chest|front|breasts?|cheek|stomach|belly|forehead)\s+(?:\w+\s+){0,2}?against\b|"
                     r"\bbent\s+over\b|\bbend\w*\s+(?:\w+\s+)?over\b", re.I)
_ONTO = re.compile(r"\b(?:to|through|onto|from)\s+(?:the\s+(?:short\s+)?(?:chain|links?)\s+(?:between|of|on|joining|linking)\s+)?"
                   r"(?:her|his|their|the|[\w-]+['’]s)\s+(?:own\s+)?(?:steel\s+)?(?:(?:hand)?cuffs|(?:cuffed|bound|tied|chained)\s+"
                   r"(?:wrists|hands))\b(?:\s+(?:behind\s+(?:her|his|their)\s+back|in\s+front\s+of\s+(?:her|him|them)))?", re.I)
_RING = re.compile(r"\b(?:to|onto|under|through|from)\s+(?:the\s+(?:back|front|middle)\s+of\s+)?"
                   r"(?:(?:the|a|an|her|his|their|[\w-]+['’]s)\s+(?:[\w-]+\s+)?"
                   r"(?:waist|belly|crotch)\s+(?:chains?|ropes?)|(?:the|her|his|their|[\w-]+['’]s)\s+(?:[\w-]+\s+)?(?:chains?|ropes?)\s+"
                   r"(?:around|on|at|between)\s+(?:the\s+small\s+of\s+(?:her|his|their)\s+back|(?:her|his|their|[\w-]+['’]s)\s+"
                   r"(?:waist|hips|belly|legs)))\b", re.I)
_REGIONS = (("wrists", r"wrists?|hands?|arms|(?:hand|wrist\s+)?cuffs|manacles"),
            ("ankles", r"ankles?|feet|legs|knees|thighs|shackles|(?:leg\s+)?irons|fetters|(?:ankle|leg)\s+(?:cuffs|chains?|shackles)"),
            ("neck", r"necks?|collars?|throats?"))
_OWNED = r"(?:her|his|their|[\w-]+['’]s)\s+(?:(?:own|left|right|two|bound|cuffed|tied|chained|shackled|steel|metal)\s+){0,2}"
_ZONES = "|".join(rx for _, rx in _REGIONS)
_HOG = r"\bhog[\s-]?ti(?:e|es|ed|ing)\b"
_LINK = re.compile(rf"{_HOG}|\b{_OWNED}(?P<a>{_ZONES})\b(?:(?!,|\b(?:then|and|before|after|while|as|so|but)\b)"
                   rf"[^.;!?]){{0,60}}?\b(?:to|onto|through|from)\s+{_OWNED}(?!(?:feet|knees)\b)(?P<b>{_ZONES})\b|"
                   rf"\bbetween\s+{_OWNED}(?P<c>{_ZONES})\s+and\s+(?:{_OWNED})?(?P<d>{_ZONES})\b", re.I)
_HARNESS = {"chains": ("locked", "unlocks the padlock", "hooks"), "rope": ("tied", "unties the knot", "ties")}
_GARB = {"remove": "remove", "tear": "remove", "lift": "lift", "wear": "wear"}
_TIES = {"handcuffs": ("locked", "locks"), "chains": ("chained", "chains"), "shackles": ("chained", "chains"), "rope": ("tied", "ties"),
         "a spreader bar": ("locked", "locks"), "a chastity belt": ("locked", "locks"),
         "zip ties": ("tied", "ties"), "tape": ("taped", "tapes"), "duct tape": ("taped", "tapes"), "straps": ("strapped", "straps"),
         "a leash": ("fastened", "fastens")}
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
         "a leash": (3, 2, "clips the {u} to the ring at the front of {x} collar", "unclips the {u} from {x} collar", ""),
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
         "shackles": "locks one shackle around each {one}, the heavy steel links between them hanging loose",
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
            (r"(?:mini)?skirts?", 4, "opens {w} {g} at the waist and slides {it} down {p} legs and off over {p} feet",
             "opens {w} {g} at the waist and slides {it} off {p} hips and away, leaving {p} bound legs exactly as they are"),
            (_LOW_COVER, 6, "opens {w} {g} at the waist and pulls {it} down {p} legs and off over {p} feet", ""),
            (r"socks|stockings|tights|pantyhose", 5, "rolls {w} {g} down {p} legs and off {p} feet", "pulls {w} {g} off {p} feet"),
            (r"bras?|bralettes?", 4, "unhooks {w} {g} and slides the straps off {p} shoulders and down {p} arms", ""),
            (r"panties|knickers|thongs?|g-strings?|underwear|briefs|boxers", 3, "slides {w} {g} down {p} legs and off over {p} feet", ""))
_OBJ = {"his": "him", "their": "them"}
_HANDY = re.compile(r"\b(?:wipes?|reach(?:es)?|grabs?|clutch(?:es)?|hugs?|waves?|claps?|types?|writes?|dials?|catch(?:es)?|"
                    r"picks?\s+up|holds?\s+(?:up|out)|covers?\s+(?:her|his|their)\s+(?:face|eyes|mouth|ears|breasts|chest)|"
                    r"rubs?\s+(?:her|his|their)\s+(?:eyes|face)|(?:brushes|tucks|fixes|smooths)\s+(?:her|his|their)\s+hair|"
                    r"strokes?|fondles?|caress(?:es)?|fingers?|jerks?|plays?\s+with|touch(?:es)?\s+(?:herself|himself|themselves)|"
                    r"spreads?\s+(?:her|his|their)\s+(?:cheeks|buttocks|lips|labia|pussy)|"
                    rf"(?:slides?|pushes?|inserts?|works?|eases?|guides?|holds?)\s+(?:a|an|the|her|his|their)\s+(?:[\w-]+\s+)?"
                    rf"(?:{_TOY}))\b", re.I)
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
         ("rope", r"ropes?(?!\s+leash)|roped|cords?(?!\s+leash)", r"ti(?:e|es|ed)|tying|bind(?:s|ing)?|bound", "wrists"),
         ("a collar", r"collar(?:s|ed)?", "", "neck"),
         ("a leash", r"leash(?:es|ed)?", "", "neck"),
         ("chains", r"chains?(?!\s+leash)", r"chained", "wrists"),
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
_LEGGED = re.compile(r"\b(?:tables?|chairs?|beds?|stools?|benches?|desks?|sofas?|couch(?:es)?|cots?|(?:step\s?)?ladders?)\s+"
                     r"(?P<at>legs?|arms?|feet|foot)\b", re.I)
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
_HUSH = "Everyone stays silent, with their lips closed."
_THINKS = "thinks, silently and with closed lips"
_THOUGHT = r"thinks?|thought|wonders?|wondered"
_MUFFLE = re.compile(r"\b(?:through|into|behind|against)\s+(?:the|her|his|their)\s+(?:[\w-]+\s+){0,2}?(tape|gag|cloth|rag|socks?|panties|"
                     r"scarf)\b", re.I)
_HUE = r"red|black|white|pink|blue|green|purple|violet|yellow|orange|grey|gray|silver|gold|golden|brown|beige|tan|clear|transparent|" \
       r"neon|dark|light|bright|shiny|glossy|matte|leather|rubber|latex|silicone|steel|metal|iron|chrome|brass|wooden|silk|satin|velvet|" \
       r"lace|nylon|cotton|hemp|jute|padded|fur|furry|fluffy|velcro|plastic|pvc|vinyl|heavy|thick|thin|wide|studded|spiked|locking|" \
       r"soft|coarse|braided"
_SOFT = re.compile(r"\b(?:leather|padded|fur|furry|fluffy|velcro|nylon|rubber|latex|silk|satin|velvet|cotton|lace)\b", re.I)
_TALKED = re.compile(r"\b(?:says?|said|saying|words?|asks?|asked|tells?|told|whispers?|whispered|mutters?|muttered|"
                     r"mumbles?|mumbled|murmurs?|murmured|answer(?:s|ed)?(?!\s+(?:the|her|his|their)\s+(?:phone|door|call))|replies|"
                     r"replied|orders?|ordered|commands?|commanded|demands?|demanded|warns?|warned|threatens?|threatened|promises?|"
                     r"promised|explains?|explained|insists?|insisted|jokes?|joked|talks?|talking|talked|speaks?|speaking|spoke|chats?|"
                     r"chatting|chatted|argues?|arguing|argued|chants?|chanted|taunts?|taunted|stammers?|stammered|begs?|begged|pleads?|"
                     r"pleaded|protests?|protested|calls?\s+(?:out|for|to)|called\s+(?:out|for|to))\b", re.I)
_CLINK = {"handcuffs", "chains", "shackles", "a spreader bar", "a chastity belt"}
_LATCH = {"handcuffs": ("the handcuffs ratchet shut with sharp metallic clicks", "the handcuffs click open"),
          "shackles": ("the shackles clank shut", "the shackles clank open"),
          "chains": ("the chains rattle as they are fastened", "the chains rattle loose"),
          "zip ties": ("the zip ties rasp tight", "the zip ties snap as they are cut"),
          "duct tape": ("duct tape rips off the roll", "the tape peels away with a sticky rip"),
          "rope": ("the rope creaks as it is pulled tight", "the rope slithers loose"),
          "straps": ("the buckles clink as the straps are pulled tight", "the buckles clink as the straps come loose"),
          "a leash": ("the leash's clip snaps onto the collar ring", "the leash's clip snaps open")}
_OUTDOOR = re.compile(r"\b(?:outdoors?|outside|forest|woods|jungle|park|street|road|alley|beach|shore|field|meadow|garden|yard|backyard|"
                      r"lake|river|mountains?|hills?|desert|rooftop|parking\s+lot|campsite|clearing|porch|balcony|cliff|trail|"
                      r"countryside|farm|dock|pier|courtyard)\b", re.I)
_INDOOR = re.compile(r"\b(?:indoors|inside|room|bedroom|bathroom|kitchen|basement|cellar|office|hallway|corridor|apartment|house|"
                     r"cabin|garage|warehouse|cell|attic|studio|motel|hotel|bar|pub|(?:night)?club|restaurant|cafe|diner|shop|"
                     r"store|library|car|van|truck|elevator|lobby|dungeon|bunker|barn|shed|church|gym|classroom|hospital|prison|"
                     r"closet|stairwell|shower|theat(?:er|re)|study|den|lounge|parlou?r|loft|dorm|suite|chamber|vault|crypt|tent)\b", re.I)
_TETHER = {"chained": "chains", "strapped": "straps"}
_WORDLESS = "Every sound anyone makes is wordless."
_SOUND = "All the sound comes from the scene itself: its natural ambience and the real sounds of what happens in it."
_PRONE = re.compile(r"\bface[\s-]?down\b|\bon(?:to)?\s+(?:her|his|their)\s+(?:stomach|front|belly)\b|\bprone\b", re.I)
_SUPINE = re.compile(r"\bon(?:to)?\s+(?:her|his|their)\s+back\b|\bface[\s-]?up\b", re.I)
_FLANK = re.compile(r"\bon(?:to)?\s+(?:her|his|their)\s+side\b(?!\s+of\b)", re.I)
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
_CUES = (rf"(?:un)?(?:{_APPLY})|takes?|took|pulls?|pulled|rips?|ripped|tears?|tore|yanks?|yanked|slips?|slipped|lifts?|lifted|"
         rf"raises?|raised|hikes?|hiked|pushes?|pushed|tugs?|tugged|removes?|removed|peels?|peeled|strips?|stripped|undress(?:es)?|"
         rf"unbuttons?|unzips?|opens?|frees?|releases?|cuts?|drags?|slides?|slid|rolls?|rolled|grabs?|grabbed|reaches?|holds?|kneels?|"
         rf"leans?|watches?|smiles?|steps?|walks?|turns?|sits?|stands?|gets?|got|helps?|helped|keeps?|kept|tries|tried|struggles?|"
         rf"struggled|squirms?|squirmed|yelling|screaming|begging|pleading|crying|sobbing")
_KITS = r"(?!(?:(?:hand)?cuffs|chains|shackles|gags|ropes|straps|blindfolds|collars|manacles)\s*(?:,|and\b|or\b|[.;!?]|$))"
_CLAUSE = re.compile(rf",?\s+(?:and\s+)?then\s+|;\s*|,\s+but\s+|(?<!\bbound)(?<!\btied)(?<!\bcuffed)(?<!\bgagged)(?<!\btaped)(?<!\bchained)"
                     rf"(?<!\bblindfolded)(?<!\bshackled),?\s+and\s+{_KITS}(?=(?:{_LEADS}\s+)?(?:\w+ly\s+)?(?:re-?)?(?:{_CUES})\b)|"
                     rf",\s+{_KITS}(?=(?:{_CUES}|leaving|keeping)\b)|"
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
         ("fours", r"on(?:to)?\s+(?:all\s+fours|(?:her|his|their)\s+hands\s+and\s+knees)"),
         ("bend", r"bend(?:s|ing)?\s+(?:forward\s+|down\s+)?over|bent\s+(?:forward\s+)?over|(?:bends?|bending|bent)\s+forward"),
         ("crouch", r"crouch(?:es|ed|ing)?|squat(?:s|ted|ting)?"),
         ("turn", r"(?:turns?|turned|turning|spins?|spun|spinning|whirls?|whirled)\s+(?:back\s+)?(?:a?round|to\s+face)"),
         ("kneel", r"kneel(?:s|ing)?|knelt|(?:on|onto|to)\s+(?:her|his|their)\s+knees|(?:sits?|sitting|sat)\s+(?:back\s+)?on\s+"
                   r"(?:her|his|their)\s+heels"), ("sit", r"sit(?:s|ting)?|sat|seated|(?:rid(?:es|ing)|rode)(?=\s+(?:him|her|his|them|"
                                                          r"(?-i:[A-Z])))"),
         ("lie", r"lies|lying|(?:lay|lays|laying|lie)\s+(?:down|back|flat|still|on|in|face)|curl(?:s|ed|ing)?\s+up|sleep(?:s|ing)|slept|"
                 r"(?:rolls?|rolled|rolling|turns?|turned)\s+(?:over\s+)?(?:onto|on)\s+(?:her|his|their)\s+(?:side|back|front|stomach|belly)|"
                 r"(?:falls?|fell|collaps(?:es|ed)|drop(?:s|ped)|sinks?|sank|slump(?:s|ed)|tumbl(?:es|ed)|pushed|thrown|shoved|tossed|laid|"
                 r"pinned|flung|hurled|dumped|dragged|slammed|wrestled|knocked|lowered|placed|carried|lifted|held)"
                 r"\s+(?:back\s+|down\s+|flat\s+|face[\s-]?(?:down|up|first)\s+|facedown\s+)?(?:onto|on|across|into|to)\s+"
                 r"(?:the|a|an|her|his|their)\s+(?:[\w-]+\s+)?"
                 r"(?:bed|floor|ground|mattress|cot|couch|sofa)\b|(?:falls?|fell)\s+flat"),
         ("stand", r"stand(?:s|ing)?|stood|(?:gets?|got|getting)\s+up|(?:to|on)\s+(?:her|his|their)\s+feet|ris(?:es|ing)|"
                   r"straighten(?:s|ed|ing)?\s+up"),
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
_DROP = r"throws?|threw|toss(?:es|ed)?|hurls?|hurled|flings?|flung|drops?|dropped|dumps?|dumped|slams?|slammed|pins?|pinned|" \
        r"drags?|dragged|carr(?:y|ies|ied)|lifts?|lifted|puts?|plac(?:es|ed)|sets?|wrestles?|wrestled|knocks?|knocked|holds?|held|" \
        r"press(?:es|ed)|eases?|eased|lowers?|lowered|tackles?|tackled|tips?|tipped|trips?|tripped|sprawls?|sprawled"
_MAKE = r"makes?|made|forces?|forced|orders?|ordered|tells?|told|lets?|helps?|helped|has|had|gets?|got|wants?|wanted|" \
        r"watch(?:es|ed)?|sees?|saw|hears?|heard|feels?|felt"
_STEPS = {"walk": "stand", "stairs": "stand", "hop": "stand", "stand": "stand", "kneel": "kneel", "sit": "sit", "lie": "lie"}
_BASE = {"kneel": "kneeling", "sit": "sitting", "lie": "lying down", "fours": "on all fours", "crouch": "crouching", "bend": "bent over"}
_UPRIGHT = ("stand", "walk", "stairs", "hop")
_ORDERS = (("bend", r"bend\s+(?:forward\s+|down\s+)?over|bend\s+forward"),
           ("fours", r"(?:on|onto)\s+(?:all\s+fours|your\s+hands\s+and\s+knees)"),
           ("wide", r"spread\s+(?:your|those|them|'em)(?:\s+(?:legs|knees|thighs))?|open\s+your\s+legs|legs\s+apart"),
           ("kneel", r"kneel|(?:on|onto)\s+your\s+knees"), ("lie", r"lie\s+(?:down|back|flat|on|face)"), ("sit", r"sit(?:\s+down)?"),
           ("crouch", r"crouch|squat"), ("stand", r"stand\s+up|get\s+up|on\s+your\s+feet"))
_STOPS = r"on|onto|in|into|with|from|to|off|out|against|at|toward|towards|under|over|across|and|then|while|as|before|after|behind|" \
         r"beside|next|near|up|down|back|away|for|so|but|until|inside|between|of|it|them|her|his|their|its|open|shut|closed|ajar|aside|" \
         r"tight|tightly|wide|still|steady|firmly|loosely|high|low|close|there|here"
_TAIL = rf"(?:\s+(?:with|of)\s+(?:(?:a|an|the|some)\s+)?(?:(?!(?:{_STOPS})\b)[\w'’-]+\s+){{0,3}}?[a-z]+)?" \
        rf"(?=\s*[,.;:!?]|\s*$|\s+(?:{_STOPS})\b)"
_NP = rf"(?:a|an|the|some|two|three|several|her|his|their)\s+(?:(?!(?:{_STOPS})\b)[\w'’-]+\s+){{0,4}}?(?P<h>[a-z]+){_TAIL}"
_CARRIED = re.compile(rf"\b(?:(?P<take>brings?|brought|carr(?:y|ies|ied|ying)|picks?\s+up|picked\s+up|takes?\s+(?:out|off)|"
                      rf"took\s+(?:out|off)|takes?|took|grabs?|grabbed|holds?|held|holding|pulls?\s+(?:out|off)|pulled\s+(?:out|off)|"
                      rf"produces?|produced|lifts?|lifted|raises?|raised|fetch(?:es|ed)|retriev(?:es|ed)|unwraps?|unwrapped|removes?|"
                      rf"removed|unclips?|unclipped|unstraps?|unstrapped|uses?|used|wields?|wielded|swings?|swung|brandish(?:es|ed)|"
                      rf"presses?|pressed|rubs?|rubbed|runs?|ran|switch(?:es|ed)\s+on|turns?\s+on|turned\s+on)|"
                      rf"(?P<put>sets?|puts?|places?|placed|lays?|laid|drops?|dropped|tosses?|tossed|leaves?|left|rests?|rested|"
                      rf"pockets?|pocketed|throws?|threw|hurls?|hurled|smash(?:es|ed)|breaks?|broke|clips?|clipped|attach(?:es|ed)|"
                      rf"fastens?|fastened|straps?|strapped|fits?|fitted|slips?|slipped|slides?|slid|inserts?|inserted|pushes?|pushed|"
                      rf"pulls?|pulled|tapes?|taped|ties|tied|buckles?|buckled|hangs?|hung|wraps?|wrapped)|"
                      rf"(?P<give>hands?|handed|gives?|gave|passes?|passed))\s+"
                      rf"(?:(?P<who>(?-i:[A-Z])[\w'’-]*)\s+(?=(?:a|an|the|some|two|her|his|their)\b))?(?:(?:up|out|down|back)\s+)?"
                      rf"(?:(?P<it>it|them)\b|(?P<np>{_NP}))", re.I)
_WIELDED = re.compile(rf"\b(?:with|using)\s+(?P<np>{_NP})", re.I)
_BARE = re.compile(r"\b(?:clips?|clipped|attach(?:es|ed)|fastens?|fastened|puts?|places?|placed|straps?|strapped|fits?|fitted|slips?|"
                   r"slipped)\s+(?P<np>(?:[a-z-]+\s+)?(?P<h>[a-z]+s)(?:\s+with\s+(?:a|an|the)\s+[\w'’-]+(?:\s+[\w'’-]+)?)?)"
                   r"(?:\s+between\s+them)?(?=\s+(?:to|on|onto|over|around|between|inside|into)\s+(?:her|his|their|[\w'’-]+['’]s)\b)",
                   re.I)
_LEFT_AT = re.compile(r"\s+(?:(?P<with>with\s+(?:a|an|the|some)\s+[\w'’-]+(?:\s+[\w'’-]+)?)(?:\s+between\s+them)?\s+)?"
                      r"(?:back\s+|down\s+)?(?P<prep>on\s+top\s+of|onto|on|into|inside|in|beside|next\s+to|under|over|around|to|"
                      r"between|against|across)\s+(?P<to>(?:the|a|an|her|his|their|[\w'’-]+['’]s)\s+[\w'’-]+(?:\s+(?!(?:and|"
                      r"then|while|as|before|after|with|so|but|when|until)\b)[\w'’-]+){0,3}|(?-i:[A-Z])[\w'’-]+|her|him|them)", re.I)
_SETS_IN = re.compile(rf"(?<!\bthe )(?<!\ba )(?<!\ban )\b(?!(?:{_STOPS}|is|was|has|does|goes|comes|turns|looks|stands|sits|lies|"
                      rf"walks|steps|stays|waits)\b)[a-z]+(?:s|ed)\s+(?:\w+ly\s+)?(?:(?P<it>it|them)\b|(?P<np>{_NP}))", re.I)
_SPOT_IN = ("on", "in", "inside", "under", "beside", "next to", "on top of", "against", "across")
_HATCH = r"drawers?|doors?|cabinets?|cupboards?|wardrobes?|closets?|lockers?|lids?|trunks?|box(?:es)?|suitcases?|gates?|windows?|" \
         r"fridge|refrigerator|freezer|oven|dishwasher|hatch(?:es)?|safe|nightstands?|dressers?"
_BIN = r"drawers?|cabinets?|cupboards?|wardrobes?|closets?|lockers?|box(?:es)?|suitcases?|trunks?|fridge|refrigerator|freezer|safe"
_DOOR = r"(?:the|a|an|one|her|his|their|(?-i:[A-Z])[\w'’-]*['’]s)\s+(?P<d>(?:[\w-]+\s+){0,2}?(?P<n>%s))\b"
_SWING = r"pulls?|pulled|slides?|slid|yanks?|yanked|throws?|threw|swings?|swung|flings?|flung|kicks?|kicked|pushes?|pushed|eases?|eased"
_OPENED = tuple(re.compile(rx, re.I) for rx in (
    rf"\b(?:opens?|opened|opening|unlatch(?:es|ed)?|(?:{_SWING})\s+open)\s+(?:\w+ly\s+)?{_DOOR % _HATCH}",
    rf"\b(?:{_SWING})\s+{_DOOR % _HATCH}\s+open\b", rf"\b(?:out\s+of|from)\s+{_DOOR % _BIN}",
    rf"{_DOOR % _HATCH}\s+(?:is|was|stands?|stood|hangs?|hung|swings?|swung|slides?|slid|flies|flew|falls?|fell|creaks?|creaked|"
    rf"bangs?|banged|lies|lay|left)\s+open\b",
    rf"\b(?:the|an|a|her|his|their)\s+(?:wide\s+)?open\s+(?P<d>(?:[\w-]+\s+){{0,2}}?(?P<n>{_HATCH}))\b"))
_SHUT = tuple(re.compile(rx, re.I) for rx in (
    rf"\b(?:closes?|closed|closing|shuts?|shutting|slams?|slammed|latch(?:es|ed))\s+(?:\w+ly\s+)?(?:{_DOOR % _HATCH}|(?P<it>it|them)\b)",
    rf"\b(?:{_SWING}|bumps?|bumped|nudges?|nudged)\s+(?:{_DOOR % _HATCH}|(?P<it>it|them))\s+(?:shut|closed)\b",
    rf"{_DOOR % _HATCH}\s+(?:closes|closed|shuts|slams|slammed|(?:swings|swung|clicks|clicked|slides|slid|falls|fell)\s+shut)\b"))
_SITED = re.compile(r"\b(?:on\s+top|astride|straddl\w*|between\s+(?:her|his|their)\s+(?:legs|thighs)|behind|in\s+front|climb\w*|"
                    r"(?:kneel|kneels|kneeling|sits?|sitting|lies|lying)\s+on|(?:right|left|other)\s+side|around)\b", re.I)
_IDLE = r"watch(?:es|ing)?|looks?|stares?|smiles?|grins?|laughs?|waits?|stands?|steps\s+back|says?|nods?|admires?|smirks?"
_ONE_OF = ("sink", "stove", "oven", "cooker", "fridge", "refrigerator", "dishwasher", "microwave", "bed", "desk", "fireplace", "bathtub",
           "tub", "shower", "toilet", "sofa", "couch", "dresser", "wardrobe", "counter", "worktop", "kitchen island", "piano", "altar",
           "cage", "workbench", "ladder", "sawhorse", "pillory")
_PHASE = re.compile(r"\b(?:dirty|clean|wet|dry|soapy|greasy|filthy|spotless|washed|unwashed|rinsed|empty|full|half-empty|half-full|hot|"
                    r"cold|warm|steaming|used|unused|fresh)\s+", re.I)
_ACROSS = re.compile(r"\b(?:beds?|desks?|tables?|benches?|couch(?:es)?|sofas?|counters?|worktops?|bars?|barrels?|horses?|sawhorses?|"
                     r"stools?|ottomans?|chairs?|arms?|armrests?|laps?|knees?|hoods?|bonnets?|trunks?|altars?|mattress(?:es)?|cots?|"
                     r"pillows?|cushions?)\b", re.I)
_FIXED = re.compile(r"(?:rooms?|doors?|windows?|walls?|floors?|ceilings?|beds?|chairs?|tables?|desks?|dressers?|couch(?:es)?|sofas?|"
                    r"cabinets?|shel(?:f|ves)|counters?|sinks?|tubs?|baths?|showers?|closets?|drawers?|stairs?|hallways?|cars?|vans?|"
                    r"trucks?|seats?|steps?|breaths?|moments?|looks?|glances?|turns?|sips?|photos?|pictures?|times?|ways?|chances?|places?|"
                    r"hands?|hair|faces?|heads?|mouths?|lips|arms?|legs?|fingers?|shoulders?|necks?|chins?|cheeks?|backs?|feet|foot|"
                    r"knees?|waists?|hips?|body|bodies|skin|eyes?|ears?|noses?|chests?|breasts?|thighs?|wrists?|ankles?|belly|stomachs?|"
                    r"men|man|women|woman|girls?|boys?|guards?|officers?|strangers?|someone|people|persons?|nightstands?|bedposts?|"
                    r"headboards?|pipes?|rings?|radiators?|posts?|rails?|railings?|lamps?|lights?|curtains?|blinds?|rugs?|carpets?|"
                    r"mattress(?:es)?|pillows?|sheets?|blankets?|end|ends|side|sides|front|corner|edge|middle|top|bottom|kiss|hug|"
                    r"shower|bath|nap|walk|seat)", re.I)
_TURNS = re.compile(r",?\s+(?:and\s+)?then\s+|;\s*|,\s+(?:and\s+|but\s+|as\s+|while\s+)?(?=(?-i:[A-Z])|(?:she|he|they)\b)|"
                    r"\s+(?:and|as|while)\s+(?=(?-i:[A-Z])[\w'’-]*\s+(?:\w+ly\s+)?[a-z]+(?:s|ed)\b)")
_OBEY = r"obeys|obeyed|complies|complied|(?:does|did)\s+(?:so|it|that|as\s+(?:she|he|they)\s+(?:is|are|was|were)\s+told|" \
        r"what\s+(?:she|he|they)\s+(?:is|are|was|were)\s+told)"
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
    return _INKED.sub(lambda m: " " * len(m.group(0)), _SPEECH.sub(lambda m: " " * len(m.group(0)), text or ""))


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
        wait = 0.0 if packed else sum(s for a, b, s in marks if after <= a and b <= ln["by"])
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
        tag = re.match(rf"\s*[,.!?;:—–-]*\s*(?:(?P<a>{who})\s+(?:\w+ly\s+)?(?P<va>{_SAYS}|{_THOUGHT})|"
                       rf"(?P<vb>{_SAYS})\s+(?P<b>(?-i:{names})|{refer}))\b", after, re.I)
        verb_after = (tag.group("va") or "") if tag else ""
        to_do = bool(tag) and bool(re.match(rf"\s+to\s+(?!(?:{_SAYS}|speak|talk)\b)", after[tag.end():], re.I))
        reacts = bool(tag) and k < len(heads) and bool(re.search(r"(?<!\.)\.$", q)) and bool(
            re.fullmatch(r"\s*[^.!?]+[.!?]\s*", told[m.end():heads[k]]))
        tag = None if tag and (_TRIES.fullmatch(verb_after) and to_do or _SOUNDS.fullmatch(verb_after) and reacts) else tag
        script = re.fullmatch(rf"[^a-z]*?(?-i:\b({names}|{caps}))\s*(?:\(([^()]*)\)\s*)?[:—–-]\s*", before)
        verb = ([v for v in re.finditer(rf"\b(?:{_SAYS}|{_THOUGHT})\b", before, re.I)] or [None])[-1]
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
        muffle = _MUFFLE.search(before[verb.end():] if lead and not script else after[:tag.end() + 40] if post else "")
        lines.append({"q": q, "who": name, "verb": v, "edit": edit, "gag": (gag(name, m.start()) or (muffle.group(1) if muffle else None))
                      if name else None, "post": post,
                      "thought": bool(re.fullmatch(_THOUGHT, v, re.I)), "at": lo, "end": edit[-1][1],
                      "by": _upto(told, edit[0][0], edit[-1][1])})
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
        noises += [{"q": "", "who": fit[0], "verb": m.group(0), "edit": [], "gag": item, "thought": False, "at": m.start(), "end": m.end(),
                    "by": _upto(told, m.start(), m.end()), "long": MUFFLE_SOUND}] if item else []
    return lines, noises


def gagword(item):
    return re.sub(r"^(?:a|an|the)\s+", "", re.sub(r"\s+(?:stuffed\s+in|tied\s+over|over|in)\s+.*$", "", item))


def _noise(item):
    tie = re.search(r",\s+(chained|strapped)\s+(?:face\s+down\s+|down\s+)?to\s", item, re.I)
    kinds = _gears(item) | ({_TETHER[tie[1].lower()]} if tie else set()) | ({"chains"} if re.search(r"\bchain\s+leash", item) else set())
    kinds = kinds - {"handcuffs"} | {"straps"} if "handcuffs" in kinds and _SOFT.search(item) else kinds
    return kinds if re.search(r"\b(?:ropes?|cords?|twine)\b", item, re.I) else kinds - {"rope"}


def _unspoken(sent):
    talk = _TALKED.search(sent)
    if not talk or _MUFFLE.search(sent):
        return sent
    cuts = list(re.finditer(r",?\s+(?:and|while|then|so|but|as|before|after|until|when|because)\s+|[,;]\s+", sent))
    head = sent[:max([m.start() for m in cuts if m.end() <= talk.start()], default=0)]
    tail = sent[min([m.end() for m in cuts if m.start() >= talk.end()], default=len(sent)):]
    return head + "." if len(head.split()) >= 3 else _unspoken(cap(tail)) if tail.strip() else ""


def _sound(place, told, through, added, gone, muffled, music, quiet):
    acts = [" ".join(_INKED.sub("", re.sub(_PIC, "", told[a:b], flags=re.I)).split()) for a, b in _sentences(told)]
    acts = [x for x in (_unspoken(t) for t in acts if t and not re.search(rf"</?\s*d\s*>|[\"“”]|{_THINKS}", t)) if x]
    loud = [t for t in [place] + acts if _VOCAL.search(t)]
    criers = [max([(m.start(), n) for n in quiet for m in re.finditer(rf"\b{re.escape(n)}\b", t[:_VOCAL.search(t).start()])],
                  default=(0, None))[1] for t in loud if t != place]
    on = lambda kinds: list(dict.fromkeys(n for d in (through, added) for n, v in d.items() if any(_noise(x) & kinds for x in v)))
    heard = lambda d, k: list(dict.fromkeys(_LATCH[g][k] for v in d.values() for x in v for g in sorted(_noise(x)) if g in _LATCH))
    gags = {n: x for d in (through, added) for n, v in d.items() for x in v if loud and n not in muffled and "mouth" in parts(x)}
    metal, rope, straps = on(_CLINK), on({"rope"}), on({"straps"})
    clatter = heard(added, 0) + heard(gone, 1) + ([f"the metal restraints on {join(metal)} clink and rattle with every movement"]
                                                  if metal else []) \
        + ([f"the rope on {join(rope)} creaks softly with every movement"] if rope else []) \
        + ([f"the leather straps on {join(straps)} creak softly with every movement"] if straps else [])
    clatter += ["the vibrator hums with a low electric buzz"] if re.search(r"\b(?:vibrators?|wands?|vibes?)\b", " ".join(acts),
                                                                           re.I) else []
    mouths = [f"{n} lets out muffled, closed-mouth mmmph sounds through the {gagword(g)}" for n, g in muffled.items()] \
        + [f"{n}'s sounds are muffled by the {gagword(g)}" for n, g in gags.items()]
    out, inside = _OUTDOOR.search(place), _INDOOR.search(place)
    air = bool(out) and not (inside and inside.start() < out.start())
    room = "the open-air ambience of the place, with moving air and distant sounds" if air else \
        "the quiet room tone of the place" if inside else "the natural ambience of the place"
    scape = [cap(room) + (", with the music playing there." if music else "."),
             "Clothes rustle softly, and every step, touch and movement makes its own sound as it happens."]
    scape += [cap(join(x)) + "." for x in (clatter, mouths) if x]
    still = [] if None in criers else [n for n in quiet if n not in muffled and n not in criers]
    told = [place] * bool(place) + acts + ([f"{join(still)} {'stays' if len(still) == 1 else 'stay'} silent."] if still else [])
    return "\n\n".join([f"integrated_multimodal_description: [Shot 1] {' '.join(told)}", f"overall_soundscape: {' '.join(scape)}",
                        "non_diegetic_music: N/A"]), bool(loud)


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
    head = lambda at: not text[:at].strip() or bool(re.search(r"(?:[.!?][\"”’']?\s+|\n\s*)$", text[:at]))
    for ln in lines:
        w, q, item, last = ln["who"], ln["q"], ln["gag"], len(ln["edit"]) - 1
        verb = "says" if re.fullmatch(r"tries|tried|manages?|managed|starts?|started|begins?|began|tells?|told", ln["verb"], re.I) \
            else ln["verb"]
        to = bool(ln["edit"]) and bool(re.search(r"\bto\s+$", text[:ln["edit"][-1][0]]))
        words = f"closed-mouth mmmph through the {gagword(item)}" if item else \
            f" {w} continues: ".join(f"<d>{x}</d>" for x in _STOP.split(q))
        say = f"{_THINKS}: {q}" if ln["thought"] else f"{'let' if to else 'lets'} out a muffled, {words}" if item else f"{verb}: {words}"
        for j, (s, e) in enumerate(ln["edit"]):
            gone = re.match(rf"\s+(?:in|into|through|behind|against|around|past|under)\s+(?:the|her|his|their)\s+(?:[\w-]+\s+){{0,2}}?"
                            rf"(?:(?:tape|gag)\b|{_CLOTH.pattern})", text[e:], re.I) if item and j == last else \
                re.match(r"\.(?!\.)", text[e:]) if j == last and re.search(r"[.!?]$", q) else None
            e = e + gone.end() if gone else e
            piece = (call(w) if last and not j else say if last else words if not (head(s) or ln["thought"]) else f"{call(w)} {say}") \
                if w else f"<d>{q}</d>" if j == last else text[s:e]
            piece += "." if item and j == last and re.search(r"[.!?]$", q) and re.match(r"\s+[\"“‘']?(?-i:[A-Z])", text[e:]) else ""
            edits.append((s, e, piece))
    edits += [x for x in names if not any(a < x[1] and x[0] < b for a, b, _ in edits)]
    out, at = [], 0
    for s, e, piece in sorted(edits):
        out += [text[at:s], cap(piece) if head(s) else piece]
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


def _lettered(text):
    def inked(m):
        if m.group(0)[0] not in "\"“'‘":
            return m.group(0)
        seg = re.split(r"[.!?\"“”]\s*", text[:m.start()])[-1][-80:]
        verb = list(re.finditer(rf"\b(?:{_WROTE})\b", seg, re.I))
        wrote = bool(verb) and not re.search(rf"[,;]|\b(?:and|then|{_SAYS})\b", seg[verb[-1].end():], re.I)
        after = text[m.end():]
        placed = re.match(r"\s*(?:(?:is|are|was|were|being)\s+)?(?:written|scrawled|scribbled|inscribed|printed|etched|stamped|inked|"
                          r"painted|drawn|lettered|branded)\b", after, re.I) or re.match(
            rf"\s*(?:in\s+[\w -]{{1,25}}?\s+)?(?:{_ON})\s+(?:the\s+)?(?:her|his|their|[\w'’-]+['’]s)\s+(?:[\w-]+\s+){{0,2}}?"
            rf"(?:{_SKIN})\b", after, re.I)
        return "«" + m.group(0)[1:-1] + "»" if wrote or placed else m.group(0)
    return _SPEECH.sub(inked, text)


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
    out["text"] = _lettered("\n".join(out["lines"]))
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
    return {"to": to, "on": "the bed" if _BEDPART.search(to) else of.group(1) if of else "the chair" if _ARMREST.search(to) else
            re.sub(rf"\s+(?:{_SIDES})$", "", to),
            "face": bool(m.group("lie")) and m.group("lie").lower().startswith("face"), "lies": bool(m.group("lie")),
            "up": bool(_OVERHEAD.search(to)), "sides": _sides(to) if "spread apart" in item else None}


def _sides(to):
    m = re.match(r"(the\s+)((?:[\w-]+\s+)*?)(corners|legs|bed\s?posts|posts|sides|ends|rings|hooks|feet|arms|rails|uprights)\b(.*)$",
                 to, re.I)
    return tuple(f"{m[1]}{side} {m[2]}{'foot' if m[3] == 'feet' else m[3][:-1]}{m[4]}" for side in ("left", "right")) if m else None


def _held_at(item, p):
    t = _tied(item)
    return f"{_TIES.get(next(iter(_gears(item)), ''), ('fastened',))[0]} to {t['to']}" if t else f"locked together {arms(item, p)}"


def _holder(x):
    m = _HELD_BY.search(x or "")
    return m["by"] if m else None


def _leash(x, to=None, by=None):
    return _HELD_BY.sub("", _TIED_TO.sub("", x)) + (f", fastened to {to}" if to else f", its other end held by {by}" if by else "")


def _leash_step(x, was, w, actor, dp, call=str):
    u, t, by, off = "chain" if re.search(r"\bchain\b", x) else "leash", _tied(x), _holder(x), _tied(was) if was else None
    mine = by is not None and by == actor
    if not was:
        return _BIND["a leash"][2].format(x=w, u=u) + (
            f" and {'padlocks' if u == 'chain' else 'ties'} its other end to {t['to']}" if t else
            f" and keeps its other end in {dp} hand" if mine else
            f" and hands its other end to {call(by)}" if by else ", and its other end drops to the floor")
    end = f"the other end of {w} leash"
    if t:
        return f"{'padlocks' if u == 'chain' else 'ties'} {end} to {t['to']}"
    it = "it" if off else end
    return (f"unfastens {end} from {off['to']} and " if off else "") + (
        f"takes {it} in {dp} hand" if mine else f"hands {it} to {call(by)}" if by else
        f"lets {it} drop to the floor" if off else f"lets go of {end}, and it drops to the floor")


def _apart(step, x, dp):
    g = _gear(x)
    kit = "a separate length of chain" if g == "chains" else \
        f"a separate pair of {'shackles' if g == 'shackles' else 'cuffs' if _SOFT.search(x) else 'small steel handcuffs'}"
    return f"takes {kit} in {dp} hand, {step}"


def _spare(text, worn, kept=()):
    found = {m["n"].lower(): "two steel shackles linked by a few heavy steel links" if re.match(r"shackles|leg|fetters", m["n"], re.I) else
             "two buckled cuffs clipped together by a single short link" if m["soft"] and _SOFT.search(m["soft"]) else
             f"two rigid steel cuffs {_JOINED}" for m in _SPARE.finditer(text)}
    return [(n, d) for n, d in (found or dict(kept)).items()
            if ("shackles" if re.match(r"shackles|leg|fetters", n) else "handcuffs") not in worn]


def _dressed(x, p):
    kinds, spots = _kinds(x), _spots(x)
    pairs = (_pair(x, kinds, spots, kinds[0]) if spots else [(g, d) for g, d in kinds if d]) if kinds else []
    pairs = [(g, s) for g, s in pairs if not (g in _HARNESS and s in ("waist", "hips") and (g, "crotch") in pairs)]
    return list(dict.fromkeys(_gear_item(g, s, x, p, "duct" in x.lower()) for g, s in pairs)) or [x]


def _restraints(desc, p):
    segs, out = [], []
    for _, s in _pieces(desc):
        s = re.sub(r"^\W*(?:(?:she|he|they)\s+(?:is|are)\s+)?(?:(?:in|wearing|wears|with|has\s+on|dressed\s+in|clad\s+in|naked\s+"
                   r"(?:except|but)\s+for)\s+)?(?:only\s+|just\s+|nothing\s+but\s+)?", "", s.strip(" ."), flags=re.I)
        segs[-1:] = [f"{segs[-1]} and {s}"] if segs and _gears(segs[-1]) and _spots(s) and not _gears(s) else segs[-1:] + [s]
    for s in segs:
        kit = _gears(s)
        if kit and "a leash" not in kit and not _GARMENT.search(s) and not re.search(
                r"\b(?:belt|hip|holster|pocket|hands?|shoulder|bag|hold(?:s|ing)?|carr(?:y|ies|ying)|dangl\w*|hang\w*|tattoo\w*)\b",
                s, re.I) \
                and (kit & {"shackles", "a spreader bar", "a chastity belt", "a gag", "a blindfold", "zip ties"}
                     or {"crotch"} & set(_spots(s)) or "handcuffs" in kit and _spots(s)
                     or re.search(rf"\b(?:{_STATE}|locked|padlocked|handcuffs?|manacles?|(?:wrist|ankle|leg)\s+cuffs)\b", s, re.I)):
            out += _dressed(s, p)
    return out


def _unpack(x, p, names=()):
    if not re.search(r"\bleash", x, re.I) or re.search(r"\bleash clipped to the collar around\b", x):
        return [x] if re.search(r",|\b(?:her|his|their)\b", x, re.I) else _dressed(x, p)
    m = re.match(r"(?P<a>.*?\bcollar)\s+(?:with|and)\s+(?P<b>.*)$", x, re.I)
    by = re.search(rf"\b(?:held\s+by|in)\s+({'|'.join(map(re.escape, names)) or '(?!)'})\b", x)
    item = _gear_item("a leash", "neck", m["b"] if m else x, p)
    return ([m["a"]] if m else []) + [_leash(item, None, by.group(1)) if by else item]


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
            pre = f"{context} {upto[:max(0, len(upto) - len(clause)) + m.start()]}"
            seen = list(re.finditer(rf"\b(?:the|a|an)\s+(?:[\w-]+\s+){{0,2}}?(?:{_FIXTURES})\b", pre, re.I))
            kin = list(re.finditer(r"\b(?:the|a|an|her|his|their)\s+(?:[\w-]+\s+)?(?:collars?|(?:hand)?cuffs|shackles|rings?|"
                                   r"harness(?:es)?)\b", pre, re.I))
            if not seen or re.fullmatch(r"it|them", a, re.I) and kin and kin[-1].start() > seen[-1].start():
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
        rf"\b(sits?|sitting|seated|sat|lies|lying|lay|laid|lays|kneels?|kneeling|knelt|stands?|standing|stood|spread[\s-]?eagled?|"
        rf"bends?|bent|bending|leans?|leaning|leaned|draped?)\b(?:\s+[\w'’-]+){{0,4}}?\s+"
        rf"(?:on|onto|across|over|at|in|against|by|beside)\s+(?:the|a|an|her|his|their|its)\s+(?:[\w-]+\s+){{0,2}}?"
        rf"{re.escape(_stem(head[-1]))}", whole, re.I)] if head else []
    flat = bool(cues) and bool(re.match(r"l|spread", cues[-1]))
    lies = not _OVERHEAD.search(to) and ((bool(re.search(r"\bdown\s+$", lead, re.I)) or prep != "to")
                                         and bool(_LIE_ON.search(to) or _FLOOR.search(to)) or bool(_LIE_ON.search(to) or _BEDPART.search(to))
                                         and (flat or not cues) or bool(_FLOOR.search(to)) and flat)
    face = lies and bool(_PRONE.search(whole))
    pos = "spread apart" if re.search(r"\bspread|\bcorners?\b|\bbed\s?posts\b", f"{clause} {to}", re.I) else \
        "above {p} head" if _OVERHEAD.search(to) else "" if re.search(r"head\s?board", to) else "at {p} sides" if lies else \
        "behind {p} back" if _SEAT.search(to) and not _ARMREST.search(to) else \
        "in front of {o}" if re.search(r"\b(?:tables?|desks?|counters?)$", to) else ""
    return to, lies, face, pos


def _lap(text):
    for rx in (_LAP, _LEGGED):
        text = rx.sub(lambda m: m.group(0)[:m.start("at") - m.start()] + " " * len(m.group("at")), text)
    return text


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
        o[3] and near(o[2]) == near(h[2]) and (h[4].lower().endswith("ed") or -len(o[4]) < o[2] - h[2] <= 30 and not re.search(
            r",|\b(?:and|or)\b", text[h[2]:o[2]], re.I)) for o in found)]
    return sorted(((g, d or next((o[1] for o in sorted(found, key=lambda o: abs(o[2] - at)) if not o[3] and o[1]), ""), at)
                   for g, d, at, _, _ in keep), key=lambda h: h[2])


def _kinds(text):
    return list({g: d for g, d, _ in reversed(_hits(text))}.items())[::-1]


def _gears(text):
    kinds = {{"a ball gag": "a gag"}.get(g, g) for g, _ in _kinds(_TIED_TO.sub("", text))}
    return {"a leash"} if "a leash" in kinds else kinds


def _spots(text):
    text = re.sub(r"(?<!\bher )(?<!\bhis )(?<!\btheir )(?<!\bboth )(?<!['’]s )\bhands(?=\s+(?:the|a|an|it|them|him|her|over|back)\b)",
                  "     ", text or "", flags=re.I)
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


def _look(gear, text):
    rx = {"duct tape": r"duct\s+tape|tape", "tape": r"duct\s+tape|tape"}.get(gear) or "|".join(
        x for g, a, b, _ in _GEAR if g == gear for x in (a, b) if x)
    m = re.search(rf"\b((?:(?:{_HUE})(?:\s*,\s*|\s+and\s+|\s+)){{0,3}}(?:{_HUE})\s+)(?:{rx})\b", text or "", re.I) if rx else None
    return " ".join(w for w in re.split(r"[\s,]+", m.group(1).lower()) if w and w != "and") if m else ""


def _gear_item(gear, spot, clause, p, duct=False, cloth="", upto="", context="", had=""):
    item, look = _gear_shape(gear, spot, clause, p, duct, cloth, upto, context, had), _look(gear, clause) or _look(gear, upto)
    gear = "tape" if gear == "duct tape" and not duct else gear
    if not look or gear == "a leash" or cloth:
        return item
    if gear == "handcuffs":
        return re.sub(r"\bhandcuffs\b", f"{look} {'cuffs' if _SOFT.search(look) else 'handcuffs'}", item, count=1)
    word = re.sub(r"^(?:a|an)\s+", "", gear)
    m = re.search(rf"\b(?:(a|an)\s+)?{re.escape(word)}\b", item)
    art = ("an " if look[0] in "aeiou" else "a ") if m and m.group(1) else ""
    return item[:m.start()] + f"{art}{look} {word}" + item[m.end():] if m else item


def _gear_shape(gear, spot, clause, p, duct=False, cloth="", upto="", context="", had=""):
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
    chained = gear == "chains" and spot == "neck" and re.search(r"\bcollars?\b", f"{clause} {upto}", re.I)
    if gear == "a leash" or chained:
        loop = re.search(r"\b(?:loops?|looped|wraps?|wrapped|ties?|tied|knots?|knotted|hooks?|hooked)\s+(?:the\s+(?:other\s+)?end|it)\s+"
                         r"(?:of\s+the\s+\w+\s+)?(?:around|round|to|onto|on)\s+(the\s+[\w'’-]+(?:\s+[\w'’-]+){0,2}?)"
                         r"(?=\s*[.,;!?]|\s+(?:and|then|while|so)\b|$)", clause, re.I)
        fix = fix or ((loop.group(1),) if loop else None)
        kind = re.search(r"\b(chain|leather|rope|nylon|steel|metal)\s+leash", f"{clause} {upto} {context} {had}", re.I)
        kind = "chain" if chained else kind.group(1).lower() if kind else ""
        return f"a {kind + ' ' if kind else ''}leash clipped to the collar around {p} neck" + (f", fastened to {fix[0]}" if fix else "")
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
    if _zone(g) != {"lower"} or not legs or _SKIRT.search(g):
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
    return (m["to"], m["strand"]) if m and m["link"] else None


def _lashed(x):
    m = _band(x) and re.fullmatch(rf"(?P<to>{_WRISTS}) at the (?:small of \w+ back|front of \w+ (?P<front>waist))",
                                  (_link(x) or ("",))[0])
    return (m["to"], bool(m["front"])) if m else None


def _hogged(x):
    m = bool(x) and "ankles" in parts(x) and re.fullmatch(rf"(?P<to>{_WRISTS})(?: behind \w+ back| in front of \w+)?",
                                                           (_link(x) or ("",))[0])
    return m["to"] if m else None


def _hogtied(items):
    return any(_hogged(x) and "in front of" not in _link(x)[0] for x in items)


def _facing(text):
    face, hog = False, re.compile(_HOG, re.I)
    for _, k in sorted(((m.end(), k) for k, rx in ((True, _PRONE), (False, _SUPINE), ("side", _FLANK), (None, hog))
                        for m in rx.finditer(text)), key=lambda t: t[0]):
        face = (face == "side" and face) or True if k is None else k
    return face


def _links(text):
    zone = lambda w: next(k for k, rx in _REGIONS if re.fullmatch(rx, w, re.I))
    return [{"wrists", "ankles"} if not (m["a"] or m["c"]) else {zone(m["a"] or m["c"]), zone(m["b"] or m["d"])}
            for m in _LINK.finditer(text)]


def _joined(text):
    return any(len(z) > 1 for z in _links(text))


def _cuffs(x, p, spot="wrists"):
    return f"{p} ankle cuffs" if spot != "wrists" else f"{p} {'wrist cuffs' if _SOFT.search(_TIED_TO.sub('', x)) else 'handcuffs'}"


def _end(to):
    return f"the chain between {to}" if to.endswith("handcuffs") else f"the link between {to}" if to.endswith("wrist cuffs") else to


def _tether(x, wrist, p):
    to, at = _cuffs(wrist, p) if "handcuffs" in _gears(wrist) else f"the {_unit(_gear(wrist))} around {p} wrists", arms(wrist, p)
    at = (f"at the front of {p} waist" if at.startswith("in front") else f"at the small of {p} back") if _band(x) else at
    return f"{_TIED_TO.sub('', x)}, {_TIES.get(_gear(x), ('fastened',))[0]} to {to}{' ' + at if at else ''} by a separate short " \
           f"{_STRAND.get(_gear(x)) or _STRAND.get(_gear(wrist), 'chain')}"


def _part(x, p):
    g, spot = _gear(x), "wrists" if "wrists" in parts(x) else "ankles"
    if _band(x):
        return f"{p} crotch {_unit(g)}" if _crotch(x) else f"the {_unit(g)} around {p} waist"
    return _cuffs(x, p, spot) if g == "handcuffs" else \
        f"{p} {_unit(spot)} shackles" if g == "shackles" else "the spreader bar" if g == "a spreader bar" else \
        f"the {_unit(g)} around {p} {spot}"


def _point(x, p, front=False):
    soft = _SOFT.search(_TIED_TO.sub("", x))
    return f"the {'front' if front else 'back'} of {_part(x, p)}" if _band(x) else \
        f"the {'link' if soft else 'chain'} between {_part(x, p)}" if _gear(x) in ("handcuffs", "shackles") else \
        "the middle of the spreader bar" if _gear(x) == "a spreader bar" else _part(x, p)


def _linking(x, w, p, on=True, lay=""):
    (to, front), strand = _lashed(x) or (_hogged(x), False), _link(x)[1]
    if not on:
        return f"{_JOINS[strand][1]} the separate short {strand} that joins " \
               f"{re.sub(r'\b(?:her|his|their)\b', w, to, count=1)} to {_point(x, p, front)} and takes it away"
    lead = "" if _band(x) else f"{lay}{', ' * bool(lay)}bends {w} knees and lifts {_low(x, p)} together up " + \
        ("toward " if "in front of" in _link(x)[0] else f"behind {_OBJ.get(p, p)} toward ") + f"{p} hands, then "
    to = re.sub(r"\b(?:her|his|their)\b", p if lead else w, to, count=1)
    return f"{lead}{_JOINS[strand][0]} a separate short {strand} from " \
           f"{_end(to)} to {_point(x, p, front)}"


def _swapped(x, items):
    base = lambda y: _TIED_TO.sub("", y)
    return next((y for y in items if y != x and _gear(x) == _gear(y) and (
        _gear(x) == "a leash" or _band(x) and _band(y) and (base(x) == base(y) or _crotch(x) and not _crotch(y))
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
    return {"tear": 3, "lift": 3, "back": 2, "mark": 5, "unmark": 4}.get(kind, 0)


def _still(p, flat):
    return f"{p} legs straight and flat where they lie" if flat else "both feet resting where they are"


def _behind(nm, p, what):
    return f"steps behind {nm}, takes both of {p} wrists and draws {p} arms back behind {_OBJ.get(p, p)}, then {what}, there at the " \
           f"small of {p} back, where {p} body hides {p} hands from the front"


def _taped(x, w, p, sent, flat=False, nm="", by=False):
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
            _behind(nm, p, f"winds the duct tape around {w} wrists, turning the roll around them")
            if by and spot == "wrists" and pos.startswith(" behind") and not flat and not _tied(x) else
            f"winds the duct tape around {w} {spot}{pos}, turning the roll around them"
            + (f", then tapes them to {_tied(x)['to']}" if _tied(x) else "")), True


def _bind_step(x, nm, w, p, dp, on=True, laid=False, flat=False, by=False, drop=False):
    gear, spot = _gear(x), (sorted(_places(x)) or ["wrists"])[0]
    thing = "the " + re.sub(r"^(?:a|an|the)\s+", "", re.split(r"\s+(?:stuffed\s+in|tied\s+over)\s+", x)[0])
    if " stuffed in " in x:
        return (f"pushes {thing} into {w} mouth with {dp} fingers until {p} mouth is full and {p} lips close around "
                f"{'them' if _many([thing]) else 'it'}" if on else f"pulls {thing} out of {w} mouth"), laid
    if " tied over " in x:
        return (f"ties {thing} over {w} {spot} and knots it behind {p} head" if on else f"unties {thing} and takes it off {w} {spot}"), laid
    words, unit, tied = _BIND.get(gear, _BOUND), _unit(gear), _tied(x)
    soft = gear == "handcuffs" and bool(_SOFT.search(x))
    words = (*words[:2], "buckles one cuff around {x} left {one}, then the other around {p} right {one}, and clips them together",
             "unclips the cuffs and unbuckles them {off}", words[4]) if soft else words
    if gear == "a leash":
        return (_leash_step(x, None, w, None, dp) if on else words[3].format(x=w, u="chain" if re.search(r"\bchain\b", x) else "leash")), \
            laid
    one = re.search(r"\b(?:left|right) ankle\b", x)
    spot = one.group(0) if one else spot
    lock, do, crotch = words[4], _TIES.get(gear, ("fastened", "fastens"))[1], spot == "crotch" and bool(words[4])
    if not on:
        hid = spot in ("wrists", "arms") and arms(x, p).startswith("behind")
        own = f"{nm}'s" if by and dp == p else p if hid else w
        at = f"under {own} crotch" if crotch else f"{own} {spot}"
        did = words[3].format(x=own, s=spot, g=gear, off=("out from " if crotch else "off ") + at, fr="from " + at)
        stay = f"while {own} hands stay behind {p} back until {p} {spot} are free"
        kit = {"zip ties": "zip tie", "chains": "chain", "duct tape": "tape"}.get(gear, "cuffs" if soft else re.sub(r"^an?\s+", "", gear))
        did = (did if not hid else f"{did} by feel behind {_OBJ.get(p, p)}, at the small of {p} back, {stay}" if not by else
               f"{'leans over' if flat else 'steps behind'} {nm} and, there at the small of {own} back, {did}, {stay}")
        return did + (f", then drops the {kit} to the floor" if drop and spot in _LIMBS and not did.endswith("away") else ""), laid
    if crotch:
        return f"runs the {unit} from the front of {w} waist under {p} crotch and up to the back, pulls it tight and " \
               f"{lock.format(t=' there')}" + (f", then {do} it to {tied['to']}" if tied else ""), laid
    arm = spot in ("wrists", "arms")
    pos, limbs = arms(x, p) or ("spread apart" if "spread apart" in x else ""), "arms" if arm else "legs"
    each = lambda o: f"{do} {o} left {spot[:-1]} to {tied['sides'][0]}, then reaches across and {do} {p} right {spot[:-1]} to " \
                     f"{tied['sides'][1]}" if tied["sides"] else f"{do} {o} {spot} to {tied['to']}"
    if tied and tied["lies"]:
        lay = "" if laid else f"lays {nm} down on {p} {'front' if tied['face'] else 'back'} on {tied['on']}"
        o = p if lay else w
        put = {f"at {p} sides": f"lays {o} arms flat on it at {p} sides", f"above {p} head": f"stretches {o} arms out above {p} head",
               "spread apart": f"spreads {o} {limbs} out wide", f"behind {p} back": f"keeps {o} arms behind {p} back"}.get(pos, "")
        return join([s for s in (lay, put, each(p if lay or put else w)) if s]), True
    if spot in ("ankles", "knees", "legs") and not tied and not pos and gear != "a spreader bar":
        put = ("buckles one cuff around each {one} and clips them together" if soft else
               _FEET.get(gear, "fastens the {g} around both of them together")).format(g=gear, one=_PAIR[spot])
        return f"holds {w} {spot} pressed together, {_still(p, flat)}, and {put}", laid
    bring = {f"at {p} sides": f"arms down to {p} sides", f"at {p} waist": f"hands to {p} waist"}.get(
        pos, f"hands down {pos}" if pos.startswith("between") else f"arms {pos}")
    back = by and arm and pos.startswith("behind") and not tied and not flat
    lead = f"spreads {w} {limbs} apart and " if pos == "spread apart" else f"brings {w} {bring} and " if pos and arm and not back else ""
    x_ = f"{nm}'s" if by and dp == p else p if lead or back else w
    height = " at waist height" if tied and arm and pos != "spread apart" and not _above(tied["to"], True) and not re.search(
        r"\b(?:floors?|ground)\b", tied["to"]) else ""
    did = f"{each(x_)}{height}" if tied else (words[2] if spot in _LIMBS or not lock else
                                                                   f"wraps the {unit} around {{x}} {{s}} and {lock}").format(
        x=x_, p=p, s=spot, one=spot.rstrip("s"), g=gear, t="")
    return _behind(nm, p, did) if back else lead + did, laid


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


def _doing(changes, people, call, held=None, stance=None, face=False, lying=()):
    held, stance, seq, rolls, stuffed, waits, sided, total = held or {}, stance or {}, [], {}, [], [], [], 0.0
    laid = {n for n, v in held.items() if any((_tied(x) or {}).get("lies") for x in v)}
    links, on = {n: [x for x in v if _link(x)] for n, v in held.items()}, {n: list(v) for n, v in held.items()}
    for kind, who, x, doer, sent in changes:
        feet = _ankle_item(on.get(who, [])) if kind == "hold" and _band(x) and "ankles" not in parts(x) else None
        on[who] = on.get(who, []) + [x] if kind == "hold" else [y for y in on.get(who, []) if y != x]
        was = _swapped(x, held.get(who, [])) if kind == "hold" else None
        cut = [y for y in links.get(who, []) if kind == "release" and (y == x or "wrists" in parts(x))]
        links[who] = [y for y in links.get(who, []) if y not in cut and y != was] + ([x] if kind == "hold" and _link(x) else [])
        total += _takes(kind, x if isinstance(x, str) else x[0], held.get(who, []))
        mine, tape = doer == who, kind == "hold" and "duct tape" in _gears(x) and not was
        if doer is None:
            if tape:
                rolls.setdefault((None, who), False)
            p = pron(people.get(who, {}))
            waits += [(who, p)] if kind == "release" and arms(x, p).startswith("behind") and who not in [n for n, _ in waits] else []
            continue
        actor, p = who if mine else doer, pron(people.get(who, {}))
        w = p if mine else f"{call(who)}'s"
        pose = "lie" if who in laid else stance.get(who)
        at = re.search(rf"\bon(?:to)?\s+(?P<s>(?:the|a|an|her|his|their)\s+(?:[\w-]+\s+)?(?:{_SURFACE}))\b", sent or "", re.I)
        lay = "" if kind != "hold" or not _hogtied([x]) or pose == "lie" and face else \
            f"rolls {'over' if mine else call(who)} onto {p} front" if pose == "lie" else \
            f"{'lies' if mine else 'lays ' + call(who)} face down" + (f" on {at['s']}" if at else " on the floor" if pose in (None, "stand")
                                                                     else "")
        low = _stoop(x, w, call(who), mine, pose, stance.get(actor) == "kneel") if kind in ("hold", "release") else ""
        q = w if not low else p if mine or pron(people.get(actor, {})) != p else f"{who}'s" if who in people else w
        bend = lambda s: s if not low else f"{low}, {s}" if " and " in s else f"{low} and {s}"
        if tape:
            step, rolled = _taped(x, q, p, sent, pose == "lie", call(who), not mine)
            step = bend(step) + (f", then {_linking(x, p, p, True, lay)}" if _link(x) else "")
            rolls[(actor, who)] = rolls.get((actor, who), False) or rolled
        elif kind in ("hold", "release") and _band(x) and (_crotch(x) or _lashed(x) or was):
            step = _harness_step(x, w, p, kind == "hold", was, x in cut, pron(people.get(actor, {})))
            down = _low(feet, p if mine or pron(people.get(actor, {})) != p else w) if feet and not was else ""
            step = ", ".join(([f"stands up behind {call(who)}"] if down and stance.get(actor) == "kneel"
                              and stance.get(who) in (None, "stand") else []) + [step]) + (
                f", while {down} " + ("stays just as it is" if down.startswith("the ") else "stay just as they are") if down else "")
        elif kind == "hold" and _gear(x) == "a leash":
            dp = pron(people.get(actor, {}))
            step = _leash_step(x, was, w, actor, dp, call) if was else bend(_leash_step(x, None, q, actor, dp, call))
        elif was:
            step = _linking(x if _link(x) else was, w, p, bool(_link(x)), lay)
        elif kind in ("hold", "release"):
            dp, leash = pron(people.get(actor, {})), next((y for y in reversed(on[who]) if _gear(y) == "a leash"), None)
            keeps = _PUTAWAY.search(sent) or _SHOWN.search(sent) or any(k == "hold" and _gear(y) == _gear(x) for k, _, y, *_ in changes) \
                or re.search(r"\b(?:hands?|gives?|passes?|keeps?|holds?|tosses?|throws?|drops?|pockets?|puts?|sets?|places?|lays?)\s+"
                             r"(?:them|it)\b", sent, re.I)
            lain = who in laid or who in lying and bool((_tied(x) or {}).get("face")) == (face is True)
            step, done = _bind_step(x, call(who), p if cut else q, p, dp, kind == "hold", lain, pose == "lie", not mine,
                                    kind == "release" and not keeps)
            step = _apart(step, x, dp) if kind == "hold" and leash and _gear(x) in ("handcuffs", "shackles", "chains") else step
            step = ", then ".join([_linking(y, w, p, False) for y in cut] + [step if cut else bend(step)]
                                  + ([_linking(x, p, p, True, lay)] if kind == "hold" and _link(x) else []))
            laid |= {who} if done else set()
            sided += [(actor, who)] if not mine and _tied(x) and not _SITED.search(sent or "") and (actor, who) not in sided else []
            stuffed += [actor] if kind == "hold" and " stuffed in " in x and actor not in stuffed else []
        else:
            cuffs = next((y for y in on[who] if "wrists" in parts(y)), None)
            step = _dress_step(kind, x, w, p, "" if mine else call(who), sent, bool(_ankle_item(on[who])),
                               _held_at(cuffs, p) if cuffs else "", stance.get(who))
        if not (seq and seq[-1][0] == actor):
            seq.append((actor, [], []))
        seq[-1][1].append(step)
        seq[-1][2].append(bool(low) and step.startswith(low))
    notes = [f"{cap(call(a))} " + ", then ".join(steps + (["lets go and stands back up"] if downs[-1] else []))
             + "." for a, steps, downs in seq]
    notes += [f"The roll of duct tape stays whole and outside {call(n)}'s body the whole time, and the tape lies flat where it goes."
              if a is None else
              f"The roll stays whole in {a}'s hand and outside {n}'s body the whole time; when the tape is on, {a} tears it off the roll "
              f"and puts the roll down." if rolled else
              f"{cap(call(a))} keeps the roll whole in {pron(people.get(a, {}))} other hand, outside {n}'s body."
              for (a, n), rolled in rolls.items()]
    notes += [f"{cap(call(a))}'s own mouth stays empty." for a in stuffed]
    notes += [f"{cap(call(n))}'s hands stay behind {p} back, at the small of {p} back, until {p} wrists are free." for n, p in waits]
    notes += [f"{cap(call(a))} does all of this from {call(n)}'s left side: {pron(people.get(a, {}), 0)} stays on that side of "
              f"{pron(people.get(n, {}), 2)} the whole shot, nobody is at {pron(people.get(n, {}))} right side, and {call(a)} appears "
              f"only once in every frame." for a, n in sided]
    gear = any(kind in ("hold", "release") for kind, *_ in changes) or any(held.values())
    notes += [f"It all happens at natural, real-life speed, one step after another, over about {max(1, round(total))} seconds: no "
              f"step is skipped, sped up or cut short, no {'restraint or ' * gear}piece of clothing appears, vanishes or changes by "
              f"itself, and every reaction comes only after what causes it."] if total else []
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
            does.append(_leash_step(x, was, w, d, pron(people.get(d, {}))) if kind == "hold" and gear == "a leash" else
                        f"runs a {unit} under {w} crotch{tie}" if was and _crotch(x) and not _crotch(was) else
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
                        f"takes {thing} off {w} {spot}" + (f" {arms(x, p)}" if arms(x, p).startswith("behind") else "")
                        if kind == "release" else f"takes off {w} {x}" if kind == "remove" else
                        f"tears {w} {x} off" if kind == "tear" else f"pulls {w} {x}" if kind == "lift" else
                        f"puts on {x}" if mine else f"puts {x} on {who}")
        out += [f"{cap(d)} " + ", then ".join(does) + "."] if does else []
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


def _spread(text, marks):
    told, out = narration(text), []
    for a, b, t in marks:
        cuts = [a] + [a + m.end() for m in _TURNS.finditer(told[a:b])] + [b]
        parts = [(x, y, len(told[x:y].split())) for x, y in zip(cuts, cuts[1:]) if y > x]
        total = sum(w for *_, w in parts)
        out += [(x, y, t * w / total) for x, y, w in parts] if total else [(a, b, t)]
    return out


def _upto(told, at, end):
    stop = min([i for i in (told.find(c, end) for c in (". ", "! ", "? ", "\n")) if i >= 0] + [len(told)])
    return stop if re.search(r"\b(?:as|when|after|once)\s+(?!if\b|though\b|well\b|soon\b|much\b|long\b)", told[end:stop], re.I) else at


def _raising(clause, done):
    return bool(re.search(r"\b(?:lifts?|raises?|bends?|folds?|draws?)\b", clause, re.I)) and not _gears(clause) and bool(
        set(_spots(clause)) & {"ankles", "knees", "legs", "thighs"}) and any(t[0] == "hold" and _hogged(t[2]) for t in done)


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
    gone += [k for k in range(len(spans) - 1) if k not in gone and k + 1 in gone and _raising(said(k), done)]
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
        first = {("hold", t[1], _TIED_TO.sub("", t[2])) for t in free if fresh(*t[:3])}
        left, said = secs + talk - sum(span(t) for t in atoms if t not in last), \
            _rest(text[a:b], [t for c in chunks[:-1] for t in c if t[0] != "have" and t[:3] not in first], last, people)
        said = re.sub(rf"\s+with\s+(?:another|a\s+second|a|an|the|her|his|their)\s+(?:pair\s+of\s+|set\s+of\s+)?(?:{_KIT})\b", "", said,
                      flags=re.I) if hog else said
        seen = {w: list(v) for w, v in (held or {}).items()}
        for c in chunks[:-1]:
            units.append((_said(_merged(c), people, seen), _merged(c), sum(map(span, c)), False))
            for k, who, y, *_ in c:
                seen[who] = _put(seen.get(who, []), [y]) if k == "hold" else seen.get(who, [])
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
    chain = re.compile(rf"\b(?:{_MAKE})\s+(?P<w>{anyone}|her\b|him\b|them\b)\s+to\s+[^.;!?]*?\band\s+(?:\w+ly\s+)?$", re.I)
    stoop = re.compile(rf"\b(?:bends?|bent|bending|push(?:es|ed|ing)?|shov(?:es|ed|ing)|forc(?:es|ed|ing)|press(?:es|ed|ing)|pins?|"
                       rf"pinned|pinning|holds?|held|holding|drap(?:es|ed|ing)|throws?|threw|leans?|leaned)\s+(?P<w>{anyone}|her\b|him\b|"
                       rf"them\b)\s+(?:\w+ly\s+)?(?:face[\s-]?down\s+|forward\s+|down\s+)?(?:over|across)\s+"
                       rf"(?=(?:the|a|an|her|his|their)\b(?!\s+shoulders?\b))", re.I)
    led = re.compile(rf"\b(?P<v>{_LED}|{_DROP}|sits?|sat|seats?|seated)\s+(?P<w>{anyone}|her\b|him\b|them\b)\s+(?:\w+ly\s+)?"
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
                if kind != "wide" and any(a < m.end() and m.start() < b for a, b in taken) or kind == "kneel" and re.match(
                        r"on|to", m.group(0), re.I) and _GARMENT.search(re.split(r"[,;]|\b(?:and|then)\b", sent[:m.start()])[-1]) or \
                        kind in _BASE and re.match(rf"\s+(?:{anyone}|her|him|them)\b(?!['’])\s+(?:down|up|on|onto|in|into|against|over|"
                                                   rf"across|back|to|at|beside)\b", sent[m.end():], re.I):
                    continue
                first, last = m.group(0).split()[0].lower(), [h for h in subj if h[0] < m.start()][-1:]
                bare = not re.search(r"(?:s|ed|ing)$", first) and first not in (
                    "went", "came", "ran", "knelt", "sat", "stood", "bent", "crept", "snuck", "led", "lay", "laid", "fell", "sank", "slept",
                    "thrown", "flung", "held", "got", "on", "onto", "to", "her", "his", "their", "cross-legged")
                by = make.search(sent[:m.start()]) or (chain.search(sent[:m.start()]) if bare else None)
                own = by_(by.group("w"), m.start()) if by else bool(last) and last[0][1] and not (bare and not last[0][2] and not re.search(
                    r"\bto\s+(?:\w+ly\s+)?$", sent[:m.start()], re.I))
                legs = re.search(r"\b(her|his|their)\s+(?:legs|knees|thighs)\b", m.group(0), re.I) if kind == "wide" and not by else None
                if legs and last and ({"she": "her", "he": "his", "they": "their"}.get(last[0][3].lower())
                                      or pron(people.get(last[0][3], {}))) != legs[1].lower():
                    fit = [q for q in cast if pron(people.get(q, {})) == legs[1].lower() and q != last[0][3]]
                    own = fit == [n] if fit else own
                run = kind == "wide" and re.match(r"run|ran|sprint|dash|bolt|jog|rac", first)
                tags = (["walk"] if run else []) + ([] if kind == "wide" and tried.search(sent[:m.start()]) else [kind])
                for tag in tags if own else ():
                    found.append((m.start(), m.end(), "stairs" if tag == "walk" and (first.startswith(("climb", "descend")) or re.search(
                        rf"\b(?:up|down)\s+{_FLIGHT}", sent[m.start():m.end() + 40], re.I)) else tag))
                taken += [(m.start(), m.end())] if own else []
        found += [(m.start(), m.end(), "bend") for m in stoop.finditer(sent) if by_(m.group("w"), m.start())
                  and not any(a < m.end() and m.start() < b for a, b in taken)]
        for m in led.finditer(sent):
            if by_(m.group("w"), m.start()) and not any(a < m.end() and m.start() < b for a, b in taken):
                to = re.split(r"[,;]|\b(?:and|then|but|while|as|before|after|when|until)\b", m.group("to") or "")[0]
                where = f"{m.group('d') or ''}{to}"
                at = re.sub(r"\b(?:off|from|out\s+of|away\s+from)\s+(?:the|a|an|her|his|their)\s+(?:[\w-]+\s+)?[\w-]+", "", where,
                            flags=re.I)
                seat, bed = (lambda t, rx=rx: re.search(rf"\b(?:{rx})\b", t, re.I) for rx in (
                    r"(?:arm)?chair|stool|bench|couch|sofa|seat|lap", r"bed|floor|ground|mattress|cot|rug|carpet"))
                kind = "lie" if m.group("f") else {"knees": "kneel", "feet": "stand"}.get((m.group("k") or "").lower(),
                                                                                          "lie" if m.group("k") else None) or (
                    "sit" if seat(at) else
                    "lie" if bed(at) or re.match(r"la(?:y|id)", m.group("v"), re.I) and re.fullmatch(
                        r"down|on|onto|back|across", m.group("d") or "", re.I) else
                    "stand" if seat(where) or bed(where) else
                    "stairs" if re.match(rf"(?:up|down)\s+{_FLIGHT}", f"{m.group('d')}{to}", re.I) else
                    None if re.fullmatch(_DROP, m.group("v"), re.I) else
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


def _shape(kinds, start=None):
    for k in kinds:
        start = "stand" if k == "turn" and start in ("bend", "fours") else _STEPS.get(k, k) if k in _BASE or k in _UPRIGHT else start
    return start


def _obeyed(text, n, cast, people):
    s = pron(people.get(n, {}), 0)
    who = rf"(?-i:\b{re.escape(n)})\b" + (rf"|\b{s}\b" if [pron(people.get(m, {}), 0) for m in cast].count(s) == 1 else "")
    out = []
    for m in re.finditer(rf"(?:{who})\s+(?:\w+ly\s+)?(?:{_OBEY})\b", text, re.I):
        for q in list(re.finditer(r"\"[^\"]*\"|“[^”]*”|<d>.*?</d>", text[max(0, m.start() - 300):m.start()]))[-1:]:
            at = max(0, m.start() - 300) + q.start()
            out += [(k, q.group(0), at + h.start(), at + h.end()) for k, rx in _ORDERS
                    for h in re.finditer(rf"\b(?:{rx})\b", q.group(0), re.I)]
    return sorted(out, key=lambda f: f[2])


def _props(text, lead, cast, people=None):
    names = "|".join(re.escape(n) for n in sorted(cast, key=len, reverse=True)) or "(?!)"
    found, last, doer = {}, None, None
    for a, b in _sentences(text):
        sent = narration(text)[a:b]
        held = [*_CARRIED.finditer(sent), *_BARE.finditer(sent)]
        used = [w for w in _WIELDED.finditer(sent) if not any(m.start() < w.start() < m.end() for m in held)]
        moved = [g for g in _SETS_IN.finditer(sent) if not any(m.start() <= g.start() < m.end() for m in held + used)]
        for m in sorted(held + used + moved, key=lambda m: m.start()):
            who = [x.group(1) for x in re.finditer(rf"(?:^|[,;:]\s*|\b(?:and|then|while|as|before|after|when|so|but)\s+)\W*"
                                                  rf"(?:[\w-]+-haired\s+)?(?-i:({names}))\b(?!['’])(?=\s+(?:\w+ly\s+)?[a-z]+(?:s|ed)\b)",
                                                  sent) if x.end() <= m.start()]
            doer, np, h = who[-1] if who else doer, m.group("np"), (m.group("h") or "").lower()
            key = (np.split()[-1].lower() if h in ("pair", "set", "couple") else h) if np else last
            key = next((k for k, x in found.items() if key and key in _named_by(k, x[0])), key)
            core = re.split(r"\s+with\s+", np or "")[0] + " " + (key or "")
            if not key or np and (_FIXED.fullmatch(key) or re.search(rf"\b(?:{_KIT})\b", core, re.I) or _GARMENT.search(core)
                                  or _CLOTH.search(core) or re.search(rf"(?-i:\b(?:{names}))\b", np)):
                continue
            kind = "take" if m.re is _WIELDED else "put" if m.re in (_BARE, _SETS_IN) else next(
                k for k in ("take", "put", "give") if m.group(k))
            spot, to = _LEFT_AT.match(sent, m.end()), re.match(rf"\s+to\s+(?-i:({names}))\b", sent[m.end():])
            where = {"onto": "on", "to": "on", "into": "in"}.get(spot["prep"].lower(), spot["prep"].lower()) + " " + spot["to"] \
                if spot and kind == "put" else None
            if m.re is _SETS_IN and (key not in found or not where or not where.startswith(tuple(f"{x} " for x in _SPOT_IN))
                                     or not re.match(r"(?:the|a|an)\s", spot["to"], re.I)):
                continue
            rich = [x.group(0) for x in re.finditer(rf"\b(?:a|an|the|some)\s+(?:(?!(?:{_STOPS})\b)[\w'’-]+\s+){{0,4}}?"
                                                    rf"{re.escape(key)}{_TAIL}", lead, re.I)]
            desc = _PHASE.sub("", re.sub(r"^(?:a|an|the|some|her|his|their)\s+", "", max([np or ""] + rich + [found.get(key, ("",))[0]],
                                                                                         key=len), flags=re.I))
            desc += f" {spot['with']}" if spot and spot["with"] and spot["with"] not in desc else ""
            holder = (to.group(1) if to else m.group("who") if m.group("who") in cast else None) if kind == "give" else \
                doer if kind == "take" else None
            hands = holder or (doer if kind == "put" else None)
            found = {k: (d, w, None if hands and h_ == hands else h_, x) for k, (d, w, h_, x) in found.items()}
            found[key], last = (desc, where, holder, doer), key
        found = _filled(found, _coupling(sent, cast, people), people) if people is not None else found
    return found


def _filled(found, acts, people):
    for _, r, tool, hole, _ in acts:
        inside = rf"in(?:side)?\s+(?:(?-i:{re.escape(r)})['’]s|{pron(people.get(r, {}))})\s+(?:[\w-]+\s+)?(?P<h>{_HOLE})"
        found = {k: x for k, x in found.items() if k in tool or not (m := re.fullmatch(inside, x[1] or "", re.I))
                 or bool(_REAR.match(m["h"])) != (hole == "anus")}
    return found


def _ajar(text, was):
    out, now, shut = dict(was), {}, {}
    for _, opens, m in sorted([(m.start(), True, m) for rx in _OPENED for m in rx.finditer(text)]
                              + [(m.start(), False, m) for rx in _SHUT for m in rx.finditer(text)], key=lambda h: h[0]):
        n, d = (m.groupdict().get("n") or "").lower(), (m.groupdict().get("d") or "").lower()
        if opens:
            out[n] = now[n] = d
            continue
        n = next((k for k in reversed(out) if not n or k == n or k in d.split() or n in out[k].split()), None)
        shut.update({n: out.pop(n)} if n else {})
    acts = {n: "Opening and closing" if n in now and n in shut else "Opening" if n in now else "Closing" for n in {**now, **shut}}
    return [(d, acts.get(n)) for n, d in {**was, **now, **shut}.items() if n in acts or n in out], out


def _chores(text, cast, people):
    names, out = "|".join(re.escape(n) for n in sorted(cast, key=len, reverse=True)) or "(?!)", {}
    body = rf"(?:[\w'’-]+\s+)?(?:her|his|their|its|[\w-]+['’]s)\s+(?:own\s+)?(?:\w+\s+)?(?:{_SKIN}|hands?|arms?|hair|tears|eyes|body|" \
           rf"brow)\b"
    for sent in re.split(r"(?<=[.!?])\s+", text):
        for g, rx, how in _CHORES:
            for m in re.finditer(rf"\b(?:{rx})\b(?!\s+{body})", sent, re.I):
                hit = [*re.finditer(rf"(?-i:\b({names}))\b(?!['’])|\b(she|he)\b|\b(her|him)(?=\s+(?:to\s+)?$)", sent[:m.start()], re.I)]
                pro = (hit[-1][2] or hit[-1][3] or "").lower() if hit else ""
                sex = {"she": "f", "her": "f", "he": "m", "him": "m"}.get(pro, "")
                fits = [n for n in cast if people.get(n, {}).get("sex") == sex] if sex else []
                who = hit[-1][1] if hit and hit[-1][1] else fits[0] if len(fits) == 1 else cast[0] if len(cast) == 1 and people.get(
                    cast[0], {}).get("sex", "") in ("", sex or people.get(cast[0], {}).get("sex", "")) else None
                out.setdefault(who, (g, how)) if who else None
    return out


def _actors(sent, cast, people):
    names = "|".join(re.escape(n) for n in sorted(cast, key=len, reverse=True)) or "(?!)"
    sex = lambda n: people.get(n, {}).get("sex", "")

    def who(tok, other=None, busy=()):
        t = re.sub(r"['’]s$", "", tok or "")
        if t in cast:
            return t
        g = {"she": "f", "her": "f", "he": "m", "him": "m", "his": "m"}.get(t.lower())
        fit = [n for n in cast if sex(n) == g] if g else []
        fit, rest = [n for n in fit if n != other and n not in busy] if len(fit) > 1 else fit, [n for n in cast if n != other]
        return fit[0] if len(fit) == 1 else rest[0] if g and not fit and len(rest) == 1 and not sex(rest[0]) else None
    heads = [(h.start(), who(h["w"])) for h in re.finditer(rf"(?:^|[,;:]\s*|\b(?:and|then|while|as|before|after|when|so|but|until|"
                                                          rf"once|now)\s+)\W*(?:\w+ly\s+)?(?:[\w-]+-haired\s+)?"
                                                          rf"(?P<w>(?-i:{names})|she|he)\b(?!['’])", sent, re.I)]
    return heads, who


def _close(text, cast, people):
    names = "|".join(re.escape(n) for n in sorted(cast, key=len, reverse=True)) or "(?!)"
    per, obj = rf"(?-i:{names})(?:['’]s)?|her|him|his|them|their", rf"(?-i:{names})|her|him|them"
    at = rf"against\s+(?:the|a|an)\s+(?:[\w-]+\s+)?(?P<s>{_SURF})\b"
    rxs = (("behind", rf"\b(?:(?:stand|step|walk|mov|com|get|kneel|press|slip|sneak|creep|crouch|sit|lean|wait|stay|settl|position)\w*|"
                      rf"came|knelt|sat|got|is|was)\s+(?:[\w-]+\s+){{0,3}}?behind\s+(?P<x>{obj})\b(?!['’]){_ALONE}"),
           ("behind", rf"\b(?P<x>{per})(?:\s+(?!(?:{per})\b)[\w'’-]+){{0,3}}?\s+from\s+behind\b(?!\s+(?:the|a|an|his|her|their|its|this|"
                      rf"that)\b)(?:[^.;!?]{{0,30}}?\bby\s+(?P<y>{per})\b)?"),
           ("side", rf"\bspoon(?:s|ed|ing)?\s+(?P<x>{per})\b"),
           ("behind", rf"\bagainst\s+(?P<x>{per})\s+back\b"),
           ("behind", rf"\bback\s+against\s+(?P<y>{per})\b(?!['’])"),
           ("wall", rf"\b(?:pin|push|press|shov|slam|back|hold|held|trap|corner|lean|leant|put|set|stand|stood|prop)\w*\s+(?P<x>{obj})\s+"
                    rf"(?:\w+\s+){{0,2}}?{at}"),
           ("wall", rf"\b(?:lean|leant|stand|stood|rest|sit|sat|slump|press|back|flatten)\w*\s+(?:\w+\s+){{0,2}}?{at}"),
           ("wall", rf"\bwith\s+(?:her|his|their)\s+back\s+(?:against|to)\s+(?:the|a|an)\s+(?:[\w-]+\s+)?(?P<s>{_SURF})\b"),
           ("hug", rf"\b(?:hug|embrac|cuddl|snuggl)\w*\s+(?P<x>{obj})\b(?!['’])"),
           ("hug", rf"\b(?:wrap|put|throw|threw|slide|slid|slip)\w*\s+(?:his|her|their|both)\s+arms\s+(?:a?round|about)\s+(?P<x>{obj})\b"
                   rf"(?:['’]s)?"),
           ("hug", rf"\b(?:hold|held|pull|draw|drew|press|clasp|crush|gather)\w*\s+(?P<x>{obj})\s+(?:\w+\s+)?(?:close|closer|tight|tightly|"
                   rf"(?:in|into)\s+(?:his|her|their)\s+arms|(?:against|to)\s+(?:his|her|their)\s+(?:chest|body|breasts))\b"),
           ("front", rf"\b(?:kiss|grind|nuzzl)\w*\s+(?P<x>{obj})\b(?!['’])(?!\s+(?:on\s+)?(?:the\s+)?(?:back|neck|nape|shoulder)s?\b)"),
           ("front", rf"\b(?:stand|stood|step|mov|kneel|knelt|position|plant|place|stop|halt|wait|pause|remain|stay|settl)\w*\s+"
                     rf"(?:\w+\s+){{0,2}}?in\s+front\s+of\s+(?P<x>{obj})\b(?!['’])"),
           ("front", rf"\b(?:(?:walk|step|come|came|mov|go|went|stride|strode)\w*\s+(?:right\s+)?up\s+to|approach\w*|fac(?:e|es|ing)|"
                     rf"confront\w*)\s+(?P<x>{obj})\b(?!['’])"),
           ("front", rf"\bpress\w*\s+(?:himself|herself|themselves|(?:his|her|their)\s+(?:body|hips|chest))\s+(?:\w+\s+)?against\s+"
                     rf"(?P<x>{obj})\b(?!['’])"))
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        heads, who = _actors(sent, cast, people)
        for kind, rx in rxs:
            for m in re.finditer(rx, sent, re.I) if kind != "wall" or not _FACING.search(sent) else ():
                doer, d = next((n for at_, n in reversed(heads) if at_ < m.start()), None), m.groupdict()
                x = who(d["x"], doer) if d.get("x") else doer
                y = f"the {d['s'].lower()}" if d.get("s") else who(d["y"], x) if d.get("y") else doer
                out += [(y, x, kind)] if x and y and x != y and (y, x, kind) not in out else []
    return out


def _coupling(text, cast, people, shapes=None, kit=""):
    names = "|".join(re.escape(n) for n in sorted(cast, key=len, reverse=True)) or "(?!)"
    sex, shapes, out = (lambda n: people.get(n, {}).get("sex", "")), shapes or {}, []
    per, pos = rf"(?-i:{names})(?!['’])|her|him|herself|himself", rf"(?-i:{names})['’]s|her|his|their"
    hole = rf"(?P<po>{pos})\s+(?:[\w-]+\s+)?(?P<h>{_HOLE})\b|(?P<o>{per})\b{_ALONE}"
    rxs = (("fuck", rf"\b(?P<v>{_FUCK})\s+(?:\w+ly\s+)?(?:{hole})"),
           ("fuck", rf"\b(?:{_DRIVE})\s+(?:(?:himself|herself|deep|deeper|hard|harder|back|\w+ly)\s+){{0,3}}(?:into|inside)\s+(?:{hole})"),
           ("ride", rf"\b(?:rid(?:es|ing)|rode)\s+(?:\w+ly\s+)?(?:(?:{pos}|the|a|an)\s+(?:[\w-]+\s+)?(?P<t>{_SHAFT}|{_TOY})\b|"
                    rf"(?P<o>{per})\b{_ALONE})"),
           ("ride", rf"\b(?:{_DRIVE})\s+(?:(?:herself|himself|down|back|\w+ly)\s+){{0,3}}(?:onto|on)\s+(?:{pos}|the|a|an)\s+"
                    rf"(?:[\w-]+\s+)?(?P<t>{_SHAFT}|{_TOY})\b"),
           ("tool", rf"\b(?P<t>{_SHAFT}|{_TOY}|fingers?)\b[^.;!?]{{0,60}}?\b(?:into|inside|in\s+and\s+out\s+of|in|up)\s+(?:{hole})"),
           ("both", rf"\b(?:they|(?-i:{names})\s+and\s+(?-i:{names}))\s+(?:\w+ly\s+)?(?:fuck|have\s+sex|make\s+love)\b"),
           ("taken", rf"\b(?:is|are|was|were|gets?|got|being|been)\s+(?:\w+ly\s+)?(?:fucked|penetrated|pounded|screwed|mounted|"
                     rf"taken(?=\s+(?:from\s+behind|doggy|anally)))\b(?:[^.;!?]{{0,40}}?\bby\s+(?P<o>{per})\b)?"))

    for sent in re.split(r"(?<=[.!?])\s+", text):
        hits = sorted(((m.start(), k, m) for k, rx in rxs for m in re.finditer(rx, sent, re.I) if not re.match(
            r"tak|took", m.groupdict().get("v") or "", re.I) or re.match(r"\s+(?:from\s+behind|doggy|anally|up\s+the\s+ass|in\s+the\s+ass)",
                                                                       sent[m.end():], re.I)), key=lambda h: h[0])
        if not hits:
            continue
        _, k, m = hits[0]
        heads, who = _actors(sent, cast, people)
        d, doer = m.groupdict(), next((n for at, n in reversed(heads) if at < m.start()), None)
        busy = {n for _, n in heads} - {doer}
        own = lambda w: doer if (w or "").lower() in ("herself", "himself") else None
        aim, word = d.get("po") or d.get("o"), (d.get("t") or "").lower()
        tw = re.search(rf"\b(?:{pos})(?=\s+(?:[\w-]+\s+)?$)", sent[:m.start("t")], re.I) if word else None
        tw = tw.group(0) if tw else None
        if k == "both":
            men, women = [n for n in cast if sex(n) == "m"], [n for n in cast if sex(n) == "f"]
            g, r, tool = (men[0], women[0], "penis") if len(men) == len(women) == 1 else (None, None, None)
        elif k in ("ride", "taken"):
            r = doer
            g = own(aim) or who(aim, r, busy) if aim else who(tw, r, busy) if tw else None
            tool = "penis" if not word or re.fullmatch(_SHAFT, word) else word
        else:
            with_ = re.search(rf"\bwith\s+(?:(?:a|an|the|his|her|their|two|three|[\w-]+['’]s)\s+)?(?:[\w-]+\s+)?(?P<t>{_TOY}|fingers?)\b",
                              sent[m.end():], re.I) if k == "fuck" else None
            word = word or (with_["t"].lower() if with_ else "fingers" if re.match(r"finger", d.get("v") or "", re.I) else "")
            g = who(tw, who(aim, None, busy), busy) if tw else doer
            r = own(aim) or who(aim, g, busy)
            tool = "penis" if not word or re.fullmatch(_SHAFT, word) else "fingers" if word.startswith("finger") else word
        tool = "strap-on" if tool and tool.startswith("strap") else tool
        if len(cast) == 2 and (g or r) and not (g and r) and not (k == "ride" and tool not in ("penis", "strap-on")):
            g, r = (g, next(n for n in cast if n != g)) if g else (next(n for n in cast if n != r), r)
        g = g or (r if len(cast) == 1 and k != "ride" else None)
        if tool == "penis" and sex(g) == "f":
            strap = re.search(r"\bstrap-?on", kit, re.I)
            g, r, tool = (r, g, tool) if sex(r) == "m" else (g, r, "strap-on") if strap else (None, None, None)
        if not r or not tool or tool in ("penis", "strap-on") and (not g or g == r):
            continue
        h = d.get("h") or ""
        top = re.search(rf"\bon\s+top\s+of\s+(?P<w>{per})", sent, re.I)
        way = None if tool not in ("penis", "strap-on") else "astride" if k == "ride" else (
            {g: "astride", r: "top"}.get(who(top["w"], g if top["w"].lower() in ("her", "him") else None)) if top else None) or next(
            (w for w, rx in _WAYS if re.search(rf"\b(?:{rx})\b", sent, re.I)), None) or _LAID.get(shapes.get(r), (None,))[0] or (
            "astride" if shapes.get(g) == "lie" else None)
        act = (g, r, tool, "anus" if _REAR.match(h) or not h and sex(r) == "m" else "vagina", way)
        out += [act] if act not in out else []
    return out


def _coupled(act, call, people, body=None, free=True):
    g, r, tool, hole, way = act
    pg, pr = people.get(g, {}), people.get(r, {})
    rp, ro, rs, gp, go = pron(pr), pron(pr, 2), pron(pr, 0), pron(pg), pron(pg, 2)
    v = lambda s, w: w if s != "they" else w[:-1]
    R, whose = call(r), pron(pr) if g == r else f"{call(r)}'s"
    flat = way in ("top", "side", "bent") or way not in ("astride", "standing") and body in ("lie", "prone", "bend", "fours")
    up = not g or not flat and (way in ("astride", "standing", "behind") or body in (None, "stand", "kneel", "crouch"))
    legs = g and body in (None, "stand", "kneel") and way != "astride"
    into = lambda what: f"straight up into {what}" if up else f"into {what} toward {rp} head"
    line, run = (f"vertical, parallel to {rp} {'thighs' if legs else 'spine'}", f"up into {ro} and back down along that vertical line") \
        if up else (f"level, in line with {rp} body", "in and out along that level line") if flat else \
        (f"in line with the length of {rp} body", "in and out along that line")
    low = f"low between {whose} thighs, below {rp} crotch" if up else f"between {whose} thighs, at the center of {rp} crotch"
    base = "pointing straight down at the floor" if up else f"pointing away from {rp} head"
    if not g:
        return [f"The {tool} stands upright under {R}'s crotch, between {rp} thighs, {line}: its top end is inside {rp} {hole} and its "
                f"base rests below {ro}, and {rs} {v(rs, 'rises')} and {v(rs, 'sinks')} on it, so it slides {run}."]
    if g == r and not free:
        return [] if tool == "fingers" else [f"The {tool} is inside {rp} {hole}, {line}: only its base shows, between {rp} thighs, {base}."]
    G = cap(call(g))
    if tool == "fingers":
        return [f"{G}'s hand is {low}{', palm up' if up else ''}, and {gp} fingers point {into(f'{rp} {hole}')}, {line}, and slide {run}."]
    if tool not in ("penis", "strap-on"):
        tip = f"its top end inside {rp} {hole}" if up else f"its tip pointing into {rp} {hole} toward {rp} head"
        return [f"{G} holds the {tool} by its base, {gp} hand {low}: the {tool} is {line}, {tip} and its base {base}, and it slides {run}."]
    place = {"behind": f"{G} is right behind {R}, facing the same way {rs} {v(rs, 'faces')}, {gp} hips pressed against {rp} buttocks.",
             "top": f"{cap(R)} lies on {rp} back with {rp} legs apart, and {call(g)} lies on top of {ro} between {rp} thighs, face to face "
                    f"with {ro}, {gp} hips pressed against {rp} hips.",
             "astride": f"{cap(R)} sits astride {call(g)}'s hips, facing {go}, with one knee on each side of {go}.",
             "side": f"{G} lies on {gp} side right behind {R}, both facing the same way, {gp} hips pressed against {rp} buttocks.",
             "standing": f"{G} stands facing {R}, pressed close against {ro}, {gp} hips against {rp} hips."}
    shaft = "penis" if tool == "penis" else f"strap-on dildo, worn on a harness at the front of {gp} hips,"
    went = f"{G}'s {shaft} goes into {R}'s {hole} {'from below, ' if up else ''}at the center of {rp} crotch, between {rp} thighs"
    return [x for x in (place.get("behind" if way == "bent" else way), f"{went}: it points {into(ro)}, {line}, and slides {run}.") if x]


def _fixtures(text):
    text = re.sub(r"\bstep(?=ladders?\b)", "step ", text, flags=re.I)
    found = [(m.start(), n) for n in _ONE_OF for m in [re.search(rf"\b{n}\b", text, re.I)] if m and not re.search(
        rf"\b{n}e?s\b|\b(?:two|twin|double|both|second|another|other|each|every)\s+(?:[\w-]+\s+)?{n}\b", text, re.I)]
    return [n for _, n in sorted(found)]


def _named_by(key, desc):
    return {key} | {w for w in re.findall(r"[a-z]+", desc.lower()) if len(w) > 3 and not re.fullmatch(rf"{_HUE}|pair|with|of", w)}


def _kept(found, told, call, cast):
    out, busy = [], {x[2] or x[3] for k, x in _props(told, "", cast).items()
                     if not any(k in _named_by(f, y[0]) for f, y in found.items())}
    for h, (desc, where, holder, _) in list(found.items())[-5:]:
        be, spot = "are" if _many([h]) else "is", where or (f"in {call(holder)}'s hand" if holder in cast and holder not in busy else "")
        bits = [f"the same {desc} as before" if desc.lower() != h else "", f"{spot} as the shot begins" if spot else ""]
        if any(re.search(rf"\b{w}\b", told, re.I) for w in _named_by(h, desc)):
            out += [f"The {h} {be} {', '.join(x for x in bits if x)}."] if any(bits) else []
        elif spot:
            inside = spot.startswith(("in ", "inside ")) and re.search(rf"\b(?:{_HOLE})$", spot, re.I)
            deep = ": only its base shows, right at the opening, and the rest of it is inside, along the length of the body"
            out.append(f"The {desc} {be} still {spot}, looking exactly as {'they' if be == 'are' else 'it'} did{deep if inside else ''}.")
    return out


def _posture(found, text, p, bound=False, hands=None, opened=False):
    base = wide = None
    for k, sent, a, b in found:
        if k == "turn" and base and base[0] in ("bend", "fours"):
            base, wide = None, None
        elif k in _BASE or k in _UPRIGHT:
            wide, base = wide if wide == sent else None, (k, b, a) if k in _BASE or k == "stand" else None
        elif k == "wide" and re.match(r"(?:spread|open|part)|(?:(?:her|his|their)\s+)?legs\s+(?:spread|apart|wide)", text[a:b], re.I):
            wide = sent
    back = bool(hands) and hands[1].startswith("behind")
    upright = next((k in _UPRIGHT for k, *_ in reversed(found) if k in _BASE or k in _UPRIGHT), False)
    if not (base and base[0] in _BASE or wide and not bound):
        return (("standing upright" if upright else "in the same position as in the frame this shot opens on"),
                [f"{p} shoulders stay back", f"{p} {hands[0]} hands stay locked {hands[1]}"]) if back and (upright or opened) else None
    k = base[0] if base else "stand"
    lain = re.match(r"[^.;!?]*", text[base[2]:]).group(0) if k in ("lie", "sit", "kneel") else ""
    rx = rf"\b(on|in|to)(?:to)?\s+((?:the|a|an)\s+(?:[\w-]+\s+)?(?:{_SURFACE}|(?:arm)?chairs?|stools?))\b"
    bed = re.search(rx, lain, re.I) or ([None] + list(re.finditer(rx, re.split(r"[.;!?]", text[:base[2]])[-1], re.I)))[-1] if lain else None
    spot = f" {'in' if bed[1].lower() == 'in' else 'on'} {re.sub(r'^(?:a|an)\s', 'the ', bed[2])}" if bed else ""
    way = " face down" if _PRONE.search(lain) or re.search(_HOG, lain, re.I) else f" on {p} side" if _FLANK.search(lain) else \
        f" on {p} back" if _SUPINE.search(lain) else "" if bed else " down"
    over = re.match(r"\s*((?:the|a|an|her|his|their)\s+[\w'’-]+(?:\s+(?!(?:and|then|with|while|as|so|but|when|until|before|after|for|"
                    r"to|while)\b)[\w'’-]+){0,3})", text[base[1]:]) if k == "bend" else None
    word = f"bent over {re.sub(r'^(?:a|an)\s', 'the ', over.group(1))}" if over else "bent forward at the waist" if k == "bend" else \
        f"kneeling bent forward with {p} chest down" if k == "fours" and back else \
        f"lying{way}{spot}" if k == "lie" else _BASE.get(k, "standing") + spot
    word += f" with {p} {'knees' if k in ('kneel', 'fours', 'crouch') else 'legs'} spread apart" if wide and not bound else ""
    return word, ([f"{p} chest and shoulders rest on it"] if over and _ACROSS.search(over.group(1)) else []) + (
        [f"{p} {hands[0]} hands stay locked {hands[1]}"] if hands else [])


def _stoop(x, w, nm, mine, pose, kneels=False):
    if "ankles" not in parts(x) or (_tied(x) or {}).get("lies"):
        return ""
    low = "kneels" if kneels or pose == "lie" else "crouches"
    return "bends down" if mine else f"{low} behind {nm}" if pose == "kneel" else \
        f"{low} at {w} left side" if (_tied(x) or {}).get("sides") else f"{low} at {w} feet"


def _ankle_item(items):
    ankles = [x for x in items if "ankles" in parts(x)]
    return next((x for x in ankles if _tied(x)), None) or next((x for x in ankles if "ankles" in _places(x)), ankles[0] if ankles else None)


def _feet_look(x, per):
    p, g, t, one, link = pron(per), _gear(x), _tied(x), re.search(r"\b(?:left|right) ankle\b", x), _link(x)
    spot = one.group(0) if one else next((q for q in ("knees", "legs") if q in _places(x)), "ankles")
    done = _TIES.get(g, ("fastened",))[0]
    if t:
        return f"{cap(p)} {spot} {'is' if one else 'are'} {done} to {t['to']}."
    held = {"handcuffs": "handcuffed together", "shackles": "shackled together by about forty centimetres of heavy steel links",
            "a spreader bar": "locked to the two ends of a spreader bar"}.get(g, f"{done} together" + {
                "rope": " with rope", "duct tape": " with duct tape", "tape": " with tape", "zip ties": " with zip ties"}.get(g, ""))
    tail = f" and held to {link[0]} by a separate short {link[1]}, {_drawn(x, p, pron(per, 2))}" if link else \
        f" that holds {p} feet apart" if g == "a spreader bar" else ""
    return f"{cap(p)} {'ankles' if g in ('handcuffs', 'shackles', 'a spreader bar') else spot} are {held}{tail}."


def _low(x, p):
    g = _gear(x)
    return f"the spreader bar between {p} ankles" if g == "a spreader bar" else f"{p} feet" if "ankles" not in _places(x) else \
        f"{p} {'cuffed' if _SOFT.search(_TIED_TO.sub('', x)) else 'handcuffed' if g == 'handcuffs' else 'shackled' if g == 'shackles'
               else _TIES.get(g, ('bound',))[0]} ankles"


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
        tie = {"cuffs": "{w} ankles handcuffed together", "irons": "{w} ankles joined by the heavy steel links of the shackles",
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


def _ankles(name, x, w, per, did, pose, linked=False):
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
                f"{name}'s legs stay spread apart {w}, {p} left ankle {done} to {t['sides'][0]} and {p} right ankle to {t['sides'][1]}: "
                f"each foot stays right there, held to its own side." if t["sides"] else
                f"{name}'s {spot} stay {done} to {t['to']} {w}: {p} feet stay right there, held to it."]
    kind = "bar" if g == "a spreader bar" else "irons" if g == "shackles" else "cuffs" if g == "handcuffs" else "tied"
    does = v("does")
    low = f" just above {p} feet" if spot == "ankles" else ""
    wound = {"rope": "tied", "zip ties": "pulled", "tape": "wound", "duct tape": "wound", "chains": "wrapped and locked",
             "straps": "buckled"}.get(g, "fastened")
    soft = re.search(r"([\w\s-]*?)\s*cuffs\b", x) if kind == "cuffs" and _SOFT.search(x) else None
    out = [{"cuffs": f"{name}'s ankles stay cuffed together {w}: a pair of {soft.group(1).strip()} cuffs is buckled around them{low}, "
                     f"clipped to each other by a single short metal link that never stretches or gets longer, so {p} ankles stay a few "
                     f"centimetres apart at most and {p} legs stay pressed together and always move together, both at once." if soft else
                     f"{name}'s ankles stay handcuffed together {w}: a pair of rigid steel cuffs is locked around them{low}, " + (
                         _JOINED if linked else "joined by their own short "
                         "chain of a few links that never stretches, bends or gets longer") + f", so {p} ankles stay a few "
                     f"centimetres apart at most and {p} legs stay pressed together and always move together, both at once.",
            "irons": f"{name}'s ankles stay shackled together {w}: a pair of steel shackles is locked around them{low}, joined by about "
                     f"forty centimetres of heavy steel links that never stretch or get longer, so whatever {s} {does}, {p} feet stay "
                     f"within that length of each other.",
            "bar": f"{name}'s ankles stay locked to the two ends of a rigid steel spreader bar {w}, so whatever {s} {does}, {p} feet stay "
                   f"the bar's length apart, about sixty centimetres, and {p} legs move together with the bar.",
            "tied": f"{name}'s {spot} stay {done} together {w}: the {g} {'are' if many else 'is'} {wound} tight around them{low}, so "
                    f"{p} legs stay pressed together and always move together, both at once."}[kind]]
    k = max((j for j, q in enumerate(did) if q in ("kneel", "sit", "lie", "stand")), default=-1)
    rest = None if set(_STRIDES) & set(did[k + 1:]) else did[k] if k >= 0 else pose
    stay = {"irons": {"kneel": f"{S} {v('kneels')} with {p} feet behind {o}, the steel links between {p} ankles lying on the floor.",
                      "sit": f"{S} {v('sits')} with {p} feet in front of {o}, the steel links slack between {p} ankles.",
                      "lie": f"{S} {v('lies')} with {p} legs close together, the steel links between {p} ankles.",
                      "stand": f"{S} {v('stands')} with {p} feet a little apart, the steel links between {p} ankles slack on the floor."},
            "bar": {"kneel": f"{S} {v('kneels')} with {p} knees wide apart, the bar lying across the floor behind {o} between {p} ankles.",
                    "sit": f"{S} {v('sits')} with {p} legs held wide apart by the bar.",
                    "lie": f"{S} {v('lies')} with {p} legs spread apart, the bar between {p} ankles.",
                    "stand": f"{S} {v('stands')} with {p} feet wide apart, the bar between {p} ankles just above the floor."}}.get(kind, {
        "kneel": f"{S} {v('kneels')} with {p} knees together and {p} feet behind {o}.",
        "sit": f"{S} {v('sits')} with {p} knees and feet together.",
        "lie": f"{S} {v('lies')} with {p} legs together.",
        "stand": f"{S} {v('stands')} with {p} feet together."})
    hog, more = _hogged(x), stay[rest] if rest and not _hogged(x) else ""
    out += [f"A separate short {_link(x)[1]} runs from {_point(x, p)} to {_end(hog)}"
            f", so {_drawn(x, p, o, True)}, the whole time."] if hog else []
    return out + ([f"Once {w.split('from the moment ', 1)[-1]}, {more[:1].lower()}{more[1:]}" if w != "the whole shot" else more]
                  if more else [])


def _above(to, bed=False):
    return bool(re.search(r"\b(?:above|high|overhead|top|ceilings?|rafters?|beams?)\b", to)) or bed and bool(_BEDPART.search(to))


def _pose(nm, per, was, new, did=(), pose=None, joined=False, face=False, call=str, spare=(), ajar=(), taut=False, opening=False,
          shape=None, first=False):
    p, o, s = pron(per), pron(per, 2), pron(per, 0)
    items, out, name, behind = _settle(was + new), [], cap(nm), False
    gear = lambda x: next(iter(_gears(x)), "restraints")
    kit = lambda x: "cuffs" if gear(x) == "handcuffs" and _SOFT.search(x) else re.sub(r"^(?:a|an)\s+", "", gear(x))
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
                 f"in front of {o}": f"both arms lie on it, {p} wrists locked together in front of {o} and {done(wrist)} {to(hold)}",
                 f"behind {p} back": f"both arms are pulled behind {p} back, {p} wrists locked together under {o} and {done(wrist)} "
                                     f"{to(hold)}"}.get(pos, f"{p} wrists are {done(wrist)} {to(hold)}")] \
            if wrist in flat else [] if wrist else ["both arms are free"]
        bits.append(f"both legs are spread out wide on it, {p} ankles {done(ankle)} {to(leg)}" if ankle in flat and "spread apart" in ankle
                    else f"both legs lie straight along it, {p} ankles {done(ankle)} {to(leg)}" if ankle in flat else
                    f"both legs lie straight along it, {p} ankles bound together" if ankle else "both legs lie along it at their full length")
        ends = list(dict.fromkeys(_tied(x)["to"] for x in flat))
        slack = not pos and any(re.search(r"head\s?board", t) for t in ends)
        start = when(lambda x: x in flat, f"the {chain}")
        whole = start == "the whole shot"
        out.append(f"{'' if whole else cap(start) + ', '}{name if whole else nm} lies {'face down' if bed['face'] else f'on {p} back'} "
                   f"on {on}{' the whole shot' if whole else ''}, {p} whole body resting on it: {'; and '.join(bits)}. The {chain} "
                   f"{run(chain)} from {p} {join([w for w, x in (('wrists', wrist), ('ankles', ankle)) if x in flat])} "
                   f"{'' if slack else 'straight '}"
                   f"{f'down to {on}' if all(t == on or t.endswith(f' of {on}') for t in ends) else f'to {join(ends)}'}, so {p} "
                   f"{join([w for w, x in (('arms', wrist), ('legs', ankle)) if x in flat])} stay down on {on} and never rise into the air.")
    if wrist and wrist not in flat:
        w = when(lambda x: "wrists" in parts(x), f"the {kit(wrist)}")
        seat = hold and _SEAT.search(hold["to"])
        fixed = pos.startswith(("behind", "in front"))

        def stays(where):
            if not hold:
                return ""
            line = "rope" if gear(wrist) in ("rope", "zip ties", "tape", "duct tape") else "strap" if gear(wrist) == "straps" else "chain"
            keep = f"{p} wrists stay at waist height, below {p} shoulders, {where}."
            down = re.search(r"\b(?:floors?|ground)\b", hold["to"])
            return f" A long slack {line} hangs loose from {p} wrists to {hold['to']}, so {keep}" if _above(hold["to"], True) else \
                f" A short {line} runs down from {p} wrists to {hold['to']}, and {keep}" if down else \
                f" Fastened to {hold['to']} at the height of {p} waist, {keep}"
        if hold and hold["up"] and not fixed or pos == f"above {p} head":
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
            then = shape or pose
            hogged = _hogtied(items) and not seat and not around
            now = "sit" if seat else "lie" if face == "side" or hogged else _shape(did, then)
            lying, bowed = now == "lie" and not seat and not around, now in ("bend", "fours") and not seat and not around
            prone = lying and face != "side" and bool(face or hogged)
            rise = now == "stand" and then in ("lie", "bend", "fours", "kneel", "sit", "crouch") \
                or now in ("sit", "kneel") and then == "lie"
            lower = now in ("lie", "bend", "fours", "kneel", "sit", "crouch") and then == "stand" \
                or now == "lie" and then in ("sit", "kneel")
            they = s == "they"
            moves = [f"{s} {'draw' if they else 'draws'} both arms back behind {o} in the first moment"] * (first and wrist in was) + [
                f"{s} {'rise' if they else 'rises'} with both arms staying behind {o} the whole way"] * rise + [
                (f"{s} {'go' if they else 'goes'} down onto {p} front" if prone else
                 f"{s} {'lower' if they else 'lowers'} {'themselves' if they else o + 'self'}")
                + f" with both arms staying behind {o} the whole way"] * lower
            lead = ", then ".join(moves) + ", and then " if moves else ""
            out.append((f"{name} sits on {hold['to']}, {done(wrist)} to it {w}. " if seat else "")
                       + f"{name}'s arms stay pulled behind {p} back{around} {w}, with {p} wrists locked together behind {o}{also}: "
                       + lead
                       + (f"lying on {p} side, both {p} arms run down behind {o} along {p} back, and {p} hands stay locked together at "
                          f"the small of {p} back, in plain view from behind. " if lying and face == "side" else
                          f"lying face down, both {p} arms lie along {p} back at their full length, and {p} hands stay locked together "
                          f"on top of the small of {p} back, in plain view from above and from either side. " if prone else
                          f"lying on {p} back, {s} {'lie' if s == 'they' else 'lies'} on {p} arms, both arms run down behind {o} under "
                          f"{p} body, and {p} hands stay locked together under the small of {p} back, hidden beneath {o}. " if lying else
                          f"bent forward, both {p} arms lie along {p} back at their full length, and {p} hands stay locked together on top "
                          f"of the small of {p} back, in plain view from above and from behind. " if bowed else
                          f"{p} shoulders are drawn back, both arms run at their full length from {p} shoulders down behind {o}, and {p} "
                          f"hands rest together at the small of {p} back. From the front, both arms angle back from {p} shoulders and {p} "
                          f"body hides {p} hands; seen from the side or "
                          f"from behind, both arms and both hands are in plain view, whole and locked together. ")
                       + f"Whatever {nm} does in this shot, {s} {'do' if s == 'they' else 'does'} it with {p} hands locked behind "
                         f"{p} back." + stays(f"behind {p} back"))
        elif hold and re.search(r"\b(?:tables?|desks?|counters?)$", hold["to"]):
            out.append(f"{name} sits at {hold['to']} {w}, {p} forearms resting on top of it in front of {o} and {p} wrists {done(wrist)} "
                       f"to it: the {gear(wrist)} {run(gear(wrist))} straight down from {p} wrists to it, so {p} hands stay down on it.")
        elif pos in (f"at {p} sides", "spread apart"):
            out.append(f"{name}'s arms stay {'down at ' + p + ' sides' if 'sides' in pos else 'spread apart'} {w}, " + (
                f"{p} left wrist {done(wrist)} to {hold['sides'][0]} and {p} right wrist to {hold['sides'][1]}"
                if hold and hold["sides"] else
                f"{p} wrists {done(wrist)} {'to ' + hold['to'] if hold else 'there'}") + f": both arms run at their full length from {p} "
                f"shoulders to {p} wrists.")
        elif pos == f"in front of {o}" or hold and not pos:
            fast = f" and {done(wrist)} to {hold['to']}" if hold else ""
            out.append(f"{name}'s wrists stay locked together in front of {o} {w}{fast}: {p} shoulders are relaxed and down, both upper "
                       f"arms hang straight down along {p} sides, {p} elbows are bent so both forearms point forward, and {p} locked hands "
                       f"rest together at the height of {p} belly, in plain view below {p} chest. Whatever {nm} does in this shot, {s} "
                       f"{'do' if s == 'they' else 'does'} it with {p} hands locked together in front of {o} at belly height."
                       + stays(f"in front of {o}"))
        elif pos:
            reach = f"both arms bend at the elbows so {p} hands rest at {p} waist" if pos == f"at {p} waist" else \
                f"both arms run at their full length from {p} shoulders to {p} wrists"
            fast = f" and {done(wrist)} to {hold['to']}" if hold else ""
            out.append(f"{name}'s wrists stay locked together {pos} {w}{fast}: {reach}, and {p} hands never come apart and never come "
                       f"free." + stays(pos))
    hobbled = ankle if ankle and not flat and not leg else None
    pose = "sit" if hold and (_SEAT.search(hold["to"]) or _ARMREST.search(hold["to"])) else pose
    if ankle and not flat:
        out += _ankles(name, ankle, when(lambda x: "ankles" in parts(x), f"the {kit(ankle)}"), per, did, pose,
                       any(_gear(x) == "a leash" and re.search(r"\bchain\b", x) for x in items))
    if {"wrists", "ankles", "body"} & set().union(*map(parts, items)):
        out.append(f"Whenever the camera shows any part of {nm}, {s} {'are' if s == 'they' else 'is'} whole and in natural proportion: "
                   f"{p} head, shoulders, both arms and hands, torso, hips, both legs and feet are all there at their natural size, each "
                   f"joined to {p} body the natural way and placed as this shot describes.")
    pair = bool(wrist and ankle) and ankle not in wrists and gear(wrist) == gear(ankle) == "handcuffs"
    cuffs = [x for x in items if "handcuffs" in _gears(x) and x != hobbled]
    loose = sorted(set().union(*(_places(x) for x in cuffs if not _tied(x))))
    at = next((_held_at(x, p).replace("locked together", "locked") for x in cuffs if arms(x, p) or _tied(x)), "")
    soft = next((re.search(r"([\w\s-]*?)\s*cuffs\b", x) for x in cuffs if _SOFT.search(x)), None)
    noun = f"{soft.group(1).strip()} cuffs" if soft else "handcuffs"
    whose = f"The {noun} on {nm}'s wrists" if pair and cuffs == [wrist] else f"{name}'s {noun}"
    linked = any(_gear(x) == "a leash" and re.search(r"\bchain\b", x) for x in items)
    out += [f"{whose} are {'buckled around each wrist' if soft else 'rigid steel'}{f', {at}' if at else ''}: "
            + ("two soft cuffs clipped to each other by a single short metal link that never stretches or gets longer" if soft else
               f"two solid cuffs {_JOINED}" if linked else
               "two solid cuffs joined by their own short chain of a few links that never stretches, bends or gets longer")
            + (f", so {p} {join(loose)} stay a few centimetres apart at most" if loose else "") + "."] if cuffs else []
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
    leash = next((x for x in items if _gear(x) == "a leash"), None)
    unit, end, by = ("chain" if re.search(r"\bchain\b", leash) else "leash", _tied(leash), _holder(leash)) if leash else (None,) * 3
    lock, knot = ("padlocked", "padlock hangs on its own") if unit == "chain" else ("tied", "knot holds on its own")
    fixed = end["to"] if end else f"{call(by)}'s hand" if by else ""
    out = [f"{name}'s leash is one single {unit} of a fixed length, and one end of it is clipped to the ring at the front of {p} collar "
            f"at {p} throat. " + (
                f"Its other end is {lock} to {end['to']} at one fixed point on it, " + (
                    f"the same point as in the frame this shot opens on, where the {knot} the whole shot. " if opening and leash in was
                    else f"where the {knot} the whole shot. " if leash in was else f"and from the moment it is {lock}, the {knot} "
                                                                                    f"there. ") + (
                    f"Pulled taut, the {unit} runs in a straight line from {p} throat to {end['to']}: straight out in front of {o} when "
                    f"{s} {'face' if s == 'they' else 'faces'} it, over {p} shoulder when {p} back is to it." if taut else
                    f"Between them the {unit} hangs down from {end['to']} to {p} throat in a slack curve." if _above(end["to"]) else
                    f"Between them the {unit} hangs slack in one loose curve: from the ring at {p} throat it runs down across {p} chest "
                    f"to the side of {o} nearest {end['to']}, drops beside {o} to the floor, clear of {p} clothes, and lies loose along "
                    f"the floor, or along whatever {s} {'are' if s == 'they' else 'is'} on, all the way to {end['to']}. Anyone who comes "
                    f"near it steps over it or nudges it aside with a foot, and the {unit} only slides and settles.") if end else
                f"Its other end is in {call(by)}'s hand the whole shot, and the {unit} runs from {p} throat straight to that hand."
                if by else f"Its other end hangs free: the {unit} runs from {p} throat across {p} chest to one side of {o} and drops "
                           f"beside {o} to the floor, where its loose end lies.")
            + (f" Its only fixed points are the ring at {p} collar and {fixed}." if fixed else
               f" The ring at {p} collar is its only fixed point.")
            + "".join(f" The {n} in this shot are a separate, small pair of their own: {d}." for n, d in spare)
            + "".join((f" {act} the {d} moves only the {d}" if act else f" The {d} stays open and moves nothing else")
                      + f": {nm}'s leash stays one unbroken {unit} from the ring at {p} collar to "
                      + (f"{end['to']}" if end else f"{call(by)}'s hand")
                      + f", and if the {d} meets it, it only pushes the slack aside." for d, act in ajar if end or by)] * bool(leash) + out

    def piece(x):
        g, one = gear(x), re.search(r"\b(?:left|right) ankle\b", x)
        if _band(x):
            return f"{p} crotch {_unit(g)}" if _crotch(x) else f"the {_unit(g)} around {p} waist"
        if g == "a chastity belt":
            return f"{p} chastity belt"
        if x == wrist:
            return f"{p} handcuffs" if kit(x) == "handcuffs" and not pair else f"the {kit(x)} on {p} wrists"
        return f"the spreader bar between {p} ankles" if g == "a spreader bar" else \
            f"the {kit(x)} on {p} {one.group(0) if one else next((q for q in ('knees', 'legs', 'thighs') if q in _places(x)), 'ankles')}"

    pieces = [piece(x) for x in dict.fromkeys([wrist, ankle, next((x for x in items if _band(x)), None),
                                               next((x for x in items if _gear(x) == "a chastity belt"), None)]) if x]
    pieces[:1] = [cap(re.sub(rf"\b{p}\b", f"{nm}'s", x, count=1)) for x in pieces[:1]]
    ties = [f"a separate short {_link(y)[1]} from {_end(to)} to "
            f"{_point(y, p, front)}" for y in items for to, front in [_lashed(y) or (_hogged(y), False)] if to]
    them, many = "them" if len(pieces) == 2 else "any of them", len(ties) > 1
    out += [f"{join(pieces)} are separate restraints, " + (f"and the only {'links' if many else 'link'} between {them} "
                                                           f"{'are' if many else 'is'} {join(ties)}." if ties else
                                                           "each on its own with nothing joining it to another.")] \
        if len(pieces) > 1 and (ties or not joined) else []
    band = next((x for x in items if _crotch(x) and not _lashed(x)), None)
    out += [f"Nothing else is clipped, hooked or locked onto {p} crotch {_unit(gear(band))}: no other chain, cuff, link or leash "
            f"touches it at any point, front or back" + (f", and {p} locked hands rest just above the back of it, free of it"
                                                         if wrist and arms(wrist, p).startswith("behind") else "") + "."] \
        if band and len(items) > 1 else []
    out += [f"Each of {p} {'wrists' if x == wrist else 'ankles'} has its own {kit(x)}, {done(x)} on its own to {_tied(x)['sides'][0]} "
            f"and {_tied(x)['sides'][1]}, one each, at a fixed point that stays the same for the whole shot."
            for x in (wrist, ankle) if x and _tied(x) and _tied(x)["sides"]]
    out += [f"{cap(piece(x))} {'are' if _many([gear(x)]) else 'is'} {done(x)} to {_tied(x)['to']} at one fixed point on it" + (
        ", the same point as in the frame this shot opens on, for the whole shot." if opening and x in was else " for the whole shot."
        if x in was else f", and that point stays the same from the moment {'they are' if _many([gear(x)]) else 'it is'} on.")
            for x in (wrist, ankle) if x and _tied(x) and not _tied(x)["sides"]]
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


def _brand(item):
    return bool(re.search(r"\bbrand", item, re.I))


def _wiped(select, item):
    return not _brand(item) if select == "writing" else bool(re.search(re.escape(select), item, re.I))


def _whose(token, sent, at, cast, people, subject=False):
    name = re.sub(r"['’]s$", "", token)
    if name in cast:
        return name
    sex = {"her": "f", "his": "m"}.get(token.lower(), "")
    fit = [n for n in cast if not sex or people.get(n, {}).get("sex", "") in (sex, "")]
    did = [n for n in fit if re.search(rf"(?-i:\b{re.escape(n)})\b(?!['’])", sent[:at])]
    pool = (did or [n for n in fit if n not in did]) if subject else ([n for n in fit if n not in did] or did)
    return pool[0] if len(pool) == 1 else None


def read_marks(text, cast, people, skin=None):
    out, missed = [], []
    if not cast:
        return out, missed
    names = "|".join(re.escape(n) for n in sorted(cast, key=len, reverse=True))
    word = r"(?!(?:on|at|in|the|a|an|of|with|to|from|by|and)\b)[\w-]+\s+"
    mine = rf"(?:both\s+of\s+|each\s+of\s+|all\s+of\s+)?(?P<who>(?-i:{names})['’]s|her|his|their)\s+(?P<adj>(?:{word}){{0,2}}?)" \
           rf"(?P<spot>{_SKIN})\b"
    tail = r"(?=(?P<rest>[^.;!?]*))"
    write = re.compile(rf"\b(?P<verb>{_WROTE}|{_DRAWN})\b(?P<mid>[^.;!?]*?)\b(?P<prep>{_ON})\s+{mine}{tail}", re.I)
    done = re.compile(rf"(?P<mid>[^.;!?]*?)\b(?:(?:is|are|was|were|being|been)\s+)?(?P<verb>written|scrawled|scribbled|inscribed|printed|"
                      rf"etched|stamped|inked|painted|drawn|lettered)\s+(?:(?:in|with|using)\s+(?:[\w-]+\s+){{1,3}}?)?(?P<prep>{_ON})\s+"
                      rf"{mine}{tail}", re.I)
    burn = re.compile(rf"\b(?P<verb>brand(?:s|ed|ing)?)\b(?P<mid>[^.;!?]*?)(?:\b(?:{_ON}|at|above|below|near)\s+)?"
                      rf"(?:(?P<the>the)\s+(?P<tadj>(?:{word}){{0,2}}?)(?P<tspot>{_SKIN})\b|{mine})", re.I)
    for a, b in _sentences(text):
        sent, raw = narration(text)[a:b], text[a:b]
        tool = _TOOL_INK.search(raw)
        said = f" {re.sub(r'^using', 'with', tool.group(0), flags=re.I)}" if tool else ""
        passive = list(done.finditer(sent))
        found = list(write.finditer(sent)) or passive
        for m in found:
            mid, rest = raw[m.start("mid"):m.end("mid")], raw[m.start("rest"):m.end("rest")]
            words = re.search(r"«([^»]*)»", mid) or re.search(r"«([^»]*)»", rest)
            caps = re.search(r"\b[A-Z][A-Z0-9'’&!?.-]+(?:\s+[A-Z0-9][A-Z0-9'’&!?.-]*)*", mid)
            thing = re.sub(r"^(?:with|using)\s+", "", " ".join(_TOOL_INK.sub(" ", mid).split()).strip(" ,"), flags=re.I)
            bare = bool(thing) and not re.match(r"(?:a|an|the|her|his|their|its|some|my|your|our)\b", thing, re.I)
            letters = (words.group(1) if words else caps.group(0) if caps else thing if bare else "").strip()
            if re.fullmatch(rf"{_DRAWN}|printed|stamped|inked|painted|drawn|lettered", m["verb"], re.I) and not (tool or letters):
                continue
            who = _whose(m["who"], sent, m.start("who"), cast, people, "own" in (m["adj"] or "") or m in passive)
            if not who:
                missed.append((raw.strip(), f"could not tell whose body this is; name the person, as in \"{cast[0]}'s {m['spot']}\""))
                continue
            what = f'the word{"s" if len(letters.split()) > 1 else ""} "{letters}"' if letters else thing
            part = "drawn" if re.match(r"dr[ae]w", m["verb"], re.I) else "painted" if re.match(r"paint", m["verb"], re.I) else "written"
            adj = re.sub(r"\bown\s+", "", m["adj"] or "")
            where = f"{m['prep'].lower()} {pron(people.get(who, {}))} {adj}{m['spot']}{said}"
            out.append(("mark", who, f"{what} {part} {where}" if what else f"writing {where}", raw.strip()))
        if found:
            continue
        hot = re.search(r"\bbrand\b", sent, re.I) and re.search(r"\b(?:branding|iron|burn\w*|sear\w*|scar\w*|glowing|red-hot|heated)\b",
                                                              sent, re.I)
        m = burn.search(sent) if re.search(r"\bbrand(?:s|ed|ing)\b", sent, re.I) or hot else None
        if m:
            owner = m["who"] or next(iter(re.findall(rf"(?-i:{names})|\b(?:her|him|his|them)\b", m["mid"] or "")), "")
            who = _whose(re.sub(r"^him$", "his", owner, flags=re.I), sent, m.start(), cast, people) if owner else None
            spot, adj = (m["tspot"], m["tadj"]) if m["the"] else (m["spot"], m["adj"])
            if not who:
                missed.append((raw.strip(), f"could not tell whose body is branded; name the person, as in "
                                            f"\"{cast[0]}'s {spot or 'hip'}\""))
                continue
            form = re.search(r"\b(?:in\s+the\s+shape\s+of|shaped\s+(?:like|as)|in\s+the\s+form\s+of)\s+((?:a|an|the)\s+[\w' -]{1,25}?)"
                             r"(?=[,.;!?]|\s+(?:on|onto|across|into|over|against|to|at)\b|$)", raw, re.I)
            sign = re.search(r"\b(the\s+(?:letter|initials?|symbol|mark|sign|number)\s+\S{1,12})", raw, re.I)
            words = re.search(r"«([^»]*)»", raw)
            shape = f" in the shape of {form.group(1)}" if form else f" of {sign.group(1).rstrip(',.;!?')}" if sign else \
                f' reading "{words.group(1)}"' if words else ""
            at = f"{(adj or '') + spot}" if spot else "skin"
            out.append(("mark", who, f"a brand mark{shape} on {pron(people.get(who, {}))} {at}", raw.strip()))
            continue
        wipe = re.search(r"\b(?:wash(?:es|ed|ing)?|wip(?:es|ed|ing)|scrub(?:s|bed|bing)?|rub(?:s|bed|bing)?|clean(?:s|ed|ing)?|"
                         r"lick(?:s|ed|ing)?|scrap(?:es|ed|ing)|sponges?)\b[^.;!?]*?\b(?:writing|words?|letters?|lettering|marker|ink|"
                         r"lipstick|paint|message|scrawl|sharpie|chalk|crayon)\b", sent, re.I)
        if wipe and re.search(r"\b(?:off|away|clean|out)\b", sent, re.I):
            t = re.search(mine, sent, re.I)
            marked = [n for n in cast if (skin or {}).get(n)]
            who = _whose(t["who"], sent, t.start(), cast, people) if t else marked[0] if len(marked) == 1 else None
            out += [("unmark", who, "writing", raw.strip())] if who else []
    return out, missed


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

    def reend(who, x, new, sent, doer):
        have[who] = [new if y == x else y for y in have[who]]
        fresh.update([new] if x in fresh else [])
        entry = next((e for e in got if e[0] == "hold" and e[1] == who and x in e[2]), None)
        if entry:
            entry[2] = [new if y == x else y for y in entry[2]]
        else:
            got.append(["hold", who, [new], sent, doer])

    def leading(sent, c, doer):
        nonlocal topic
        on = [n for n in cast if n != doer and any(_gear(x) == "a leash" for x in have.get(n, []))]
        if doer not in cast or not on:
            return False
        it = "|it" if topic and topic[0] == "gear" and topic[1][0] in ("a leash", "chains") and topic[2] == sent else ""
        owned = rf"(?:the|her|his|their|(?-i:{names})['’]s)\s+(?:[\w-]+\s+)?"
        ref = rf"(?:{owned}leash|the\s+(?:other\s+|loose\s+|free\s+)?end(?:\s+of\s+{owned}(?:leash|chain)|(?!\s+of\b)){it})\b" \
              rf"(?!\s+(?:off|away|out)\b)"
        free = re.search(rf"\b(?:{_LFREE})\s+(?:\w+ly\s+)?(?:{ref}|(?P<chain>{owned}chain\b))\s+(?:from|off)\s+(?:of\s+)?"
                         rf"(?P<at>(?:the|a|an)\s+(?:[\w-]+\s+){{0,2}}?(?:{_FIXTURES}))\b", c, re.I)
        there = [n for n in on if free and free["at"].split()[-1].lower() in
                 (_tied(next(y for y in have[n] if _gear(y) == "a leash")) or {}).get("to", "").lower().split()]
        free, on = (free, there) if free and free["chain"] and there else (None if free and free["chain"] else free, on)
        give = re.search(rf"\b(?:{_LGIVE})\s+(?:\w+ly\s+)?(?:{ref}\s+(?:over\s+|back\s+)?to\s+(?-i:({names}))|(?-i:({names}))\s+{ref})",
                         c, re.I)
        drop = re.search(rf"\b(?:{_LDROP})\s+(?:\w+ly\s+)?{ref}|\blets?\s+{ref}\s+(?:fall|drop|go|hang|dangle)\b", c, re.I)
        by = re.search(rf"\b(?:{_LBY})\s+(?:\w+ly\s+)?(?:(?-i:({names}))|her|him|them)\b(?!['’])(?:\s+[\w'’-]+){{0,6}}?\s+by\s+{owned}"
                       rf"(?:leash|chain)\b", c, re.I)
        take = re.search(rf"\b(?:{_LTAKE})\s+(?:\w+ly\s+)?{ref}|\b(?:leads?|led)\s+(?:\w+ly\s+)?(?-i:({names}))\b(?!['’])"
                         rf"(?![^.;]*\bby\s+(?:the|her|his|their)\s+(?:hand|arm|hair|wrist|elbow|collar|shoulder|neck)\b)", c, re.I)
        m = free or give or drop or by or take
        if not m:
            return False
        own = re.search(rf"(?-i:\b({names}))['’]s\s+(?:[\w-]+\s+)?leash\b", m.group(0)) or (m is by or m is take) and m.group(1) and m
        sex = re.search(r"\b(her|his|him)\s+(?:[\w-]+\s+)?(?:leash|chain)\b|\b(?:" + _LBY + r")\s+(?:\w+ly\s+)?(her|him)\b",
                        m.group(0), re.I)
        sex = {"her": "f"}.get((sex.group(1) or sex.group(2)).lower(), "m") if sex else ""
        who = [own.group(1)] if own else [n for n in on if not sex or people.get(n, {}).get("sex", "") in (sex, "")]
        if len(who) != 1 or who[0] not in on:
            return False
        x = next(y for y in have[who[0]] if _gear(y) == "a leash")
        holder = (give.group(1) or give.group(2)) if m is give else None if m is drop else doer
        new = x if _tied(x) and m not in (free, give) or holder == who[0] else _leash(x, None, holder)
        if new != x:
            reend(who[0], x, new, sent, doer)
        topic = ("gear", ("a leash", "neck"), sent)
        return not {g for g, _ in _kinds(c)} - {"a leash", "chains", "a collar"}

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
        leashed = "a leash" in {g for g, _ in kinds} or ref and topic and topic[0] == "gear" and topic[1][0] in ("a collar", "a leash")
        if {"chains", "a leash"} & {g for g, _ in kinds} and not spots and (re.search(r"\bcollar\b", c, re.I) or leashed):
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
        kinds = kinds or [("straps", "wrists")] * bool(hog and re.search(r"\bstraps?\b", plain, re.I))
        if not (kinds or freed or it or hog):
            return
        gear, aside = kinds[0] if kinds else topic[1] if it else ("rope", "wrists") if hog else None, topic
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
        dressed = None if loose or force else re.search(
            rf"(?:^|[,;:]\s*|\b(?:and|then|but|while|as|when)\s+)\W*(?:[\w-]+-haired\s+)?(?-i:\b({names}))\b(?!['’])"
            rf"(?:\s+(?!(?:{_VERBS}|{_KIT})\b)[a-z]+s\b(?:(?!\b(?:{_VERBS}|{_KIT}|and|then|but|him|them|(?-i:{names}))\b)[^,;])*?)?"
            rf"(?:\s*,\s*|\s+with\s+|\s+(?:wears?|wearing|has)\s+)"
            rf"(?:(?:a|an|the|only|just|pair\s+of|[\w-]+)\s+){{0,3}}?(?:{_KIT})\b(?=\s+(?:on|around|over|in|locked|between|under)\b|"
            rf"\s*[,.;!?]|\s*$)(?![^.;]*\b(?:belt|hip|holster|pocket|hands?|bag)\b)", (sent[:sent.find(c)] if c in sent else "") + c)
        still = bool(dressed) or not loose and not acting and bool(_STILL.search(c) or (
            last and last[1] == "have" and not _PRONOUN.search(c) and not re.search(rf"(?-i:\b(?:{names}))\b", c)))
        if not (loose or still or acting or _DOES.search(c) or spots and _ROUTES.search(c)) or \
                not (loose or still or spots or target.search(c)) and _PLACED.search(c) or \
                it and not spots and _PLACED.search(c) and any(e[0] == "release" and e[3] == sent for e in got):
            return
        kit = set() if hog and loose else {{"a ball gag": "a gag"}.get(g, g) for g, _ in kinds} or (
            {{"a ball gag": "a gag"}.get(gear[0], gear[0])} if gear and not freed and (kinds or it) else set())
        thing = gear[0].replace("a ", "") if gear and not freed and (kinds or it) and not (hog and loose) else "restraints"
        who = force or (dressed[1] if dressed else _named(bare, names, f"{_SPOTS}|{_KIT}|head|face"))
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
        if not who and gear and gear[0] in ("chains", "a leash") and re.search(r"\bcollar\b", c, re.I):
            collared = [n for n in cast if any("a collar" in _gears(x) for x in have.get(n, []))]
            who = collared[0] if len(collared) == 1 else None
        if not who and not loose and c in sent:
            who = _named(sent[:sent.find(c)], names, f"{_SPOTS}|head|face")
        if not who and last and not re.search(rf"(?-i:\b(?:{names}))\b", c):
            who = last[0]
        if not who and not loose and len(holders) == 1 and ({d for _, d in kinds} == {"wrists"} or kit == {"a leash"}) and _anchor(c, c):
            who = holders[0]
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
        worn = next((y for y in mine if "wrists" in parts(y)), None) or _ankle_item(mine)
        if hog and not kinds and not it:
            ties = (*_STRAND, "handcuffs", "chains", "shackles")
            told = [g for g, _, _ in _hits(narration(story or before)) if g in ties]
            gear = (_gear(worn), "wrists") if worn else aside[1] if aside and aside[0] == "gear" and aside[1][0] in ties else \
                (told[-1], "wrists") if told else gear
            topic = ("gear", gear, sent)
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
            twice = [re.sub(r"^an?\s+", "", k or thing) for k in kit or [None]
                     if len({frozenset(_places(x)) for x in pick if not k or k in family(x)}) > 1][:1]
            if not pick:
                if who not in freed_now:
                    missed.append((sent, f"nothing to take off: {who} has no {thing}" + (f" on {p} {join(spots)}" if spots else "")))
            elif not spots and not whole and twice:
                missed.append((sent, f"could not tell which {twice[0]} {'come' if _many(twice) else 'comes'} off: {who} has {join(pick)}; "
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
                                          " ".join(x for x in have.get(who, []) if _gear(x) == "a leash") if s == "neck" else
                                          next((arms(x, q) for x in have.get(who, []) if s in ("wrists", "arms") and arms(x, q)), ""))
        look = lambda g: f"{_look(g, cx) or _look(g, worn or '')} {g}"
        made = list(dict.fromkeys(_gear_item(g, s, cx if s == "wrists" else look(g), p) if hog and s in ("wrists", "ankles") else
                                  item(g, s, p) for g, s in pairs))
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
        stop = base + len(re.sub(r"[\s.!?…\"”’]+$", "", c))
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
                fasten = bool(_LFASTEN.search(part)) and not _LHAND.search(part)
                if fasten or not leading(sent, part, by):
                    n0 = len(got)
                    restraint(sent, part, by, firm, base=told.find(part, cur))
                    for who, x in [(e[1], y) for e in got[n0:] if e[0] == "hold" and by in cast and by != e[1] for y in e[2]
                                   if _gear(y) == "a leash" and not _tied(y) and not _holder(y)]:
                        reend(who, x, _leash(x, None, by), sent, by)
                    if fasten:
                        leading(sent, part, by)
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
    held, off, worn, lifted, skin, known = {}, {}, {}, {}, {}, list(people)
    sample = {"hold": "duct tape over her mouth", "wear": "a red dress", "lift": "shirt",
              "mark": 'the word "SLAVE" written across her stomach'}

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
            out.append((who, [y for x in items for y in _unpack(x, pron(people.get(who, {})), people)] if kind == "hold" else items))
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
    for n, v in people.items():
        m = re.search(r"[^,;.]*\b(?:collar\b[^,;.]*\bleash|leash(?:ed)?\b[^,;.]*\bcollar)\b[^,;.]*", v["desc"], re.I)
        if m and not re.search(r"\b(?:hold(?:s|ing)?|carr(?:y|ies|ying))\b", m.group(0), re.I) and \
                not any(_gear(x) == "a leash" for x in held.get(n, [])):
            held.setdefault(n, []).append(_unpack(m.group(0).strip(), pron(v), people)[-1])
    for n, v in people.items():
        held.update({n: held.get(n, []) + [x for x in _restraints(v["desc"], pron(v)) if x not in held.get(n, [])]} if v["desc"] else {})
    names, key = "|".join(re.escape(n) for n in sorted(people, key=len, reverse=True)) or "(?!)", lambda y: (_gear(y), frozenset(parts(y)))
    for who, items in [split_tag(a, people) for k, a, *_ in (read_beat(anchor, list(people), people, {}, {}, {}, {})[0]
                                                             if (anchor or "").strip() else []) if k in ("have", "hold")] + [
            (m[1], _restraints(m["t"], pron(people[m[1]]))) for m in re.finditer(
                rf"(?-i:\b({names}))(?:['’]s\b|\s*:|\s+(?:wears|is|are|has|had)\b)(?P<t>[^.;!?]*?)(?=(?-i:\b(?:{names})\b)|[.;!?]|$)",
                anchor or "")]:
        kit = {key(y) for y in held.get(who, [])}
        held.update({who: held.get(who, []) + [x for x in items if key(x) not in kit]} if who in people else {})
    for who, items in resolve("wear", start["wear"], "wear"):
        worn.setdefault(who, []).extend(items)
    for who, items in resolve("remove", start["remove"] + start["tear"], "remove"):
        off.setdefault(who, []).extend(items or ["*"])
    for who, items in resolve("lift", start["lift"], "lift"):
        lifted.setdefault(who, []).extend(pulled(x, people.get(who, {})) for x in items)
    for who, items in resolve("mark", start["mark"]):
        skin.setdefault(who, []).extend(x for x in items if x not in skin.get(who, []))
    def garb(who):
        return sorted((clean(s, False) for s in wearing(who)), key=lambda g: _garb(g)[0])

    def lasts(kind, arg):
        who, items = split_tag(arg, people, _GARB.get(kind, False))
        return [_takes(kind, x, held.get(who, [])) for x in (garb(who) if kind in ("remove", "tear") and (not items or "*" in items)
                                                             else items)]

    shots, present, prev, voice, ceiling, crowd, told, i = [], [], None, None, align(round(float(shot_seconds) * FPS)), {}, "", 0
    spares, opened, coupled, near, beside = [], {}, [], [], {}
    for k, q in enumerate(paras):
        q["beat"] = k + 1
    while i < len(paras):
        p = paras[i]
        keyed = (i > 0 and not p["cut"]) or (i == 0 and first_frame)
        tagged = [split_tag(a, people)[0] for k in GEAR_TAGS for a in p[k]]
        cast = list(dict.fromkeys(([] if p["cut"] or i == 0 else present) + mentions(p["text"], known) + tagged))
        held.update({n: [_leash(x) if _holder(x) not in (None, *cast) else x for x in held[n]] for n in cast if n in held})
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
            paras[i:i + 1] = [{**p, **{k: [] for k in STATE_TAGS}, **({k: p[k] for k in MARK_TAGS} if k == len(split) - 1 else {}),
                               "text": t, "read": r, "missed": missed if not k else [],
                               "cut": p["cut"] and not k, "exit": p["exit"] if k == len(split) - 1 else [], "heard": heard,
                               "unread": p["unread"] if not k else [], "part": (k + 1, len(split)), "extras": extras}
                              for k, (t, r) in enumerate(split)]
            continue
        going = [split_tag(w, people)[0] for w in p["exit"]]
        read += [("hold", f"{n}, {_leash(x)}", f"exit: {_holder(x)}", _holder(x)) for n in cast for x in held.get(n, [])
                 if _holder(x) in going and not any(e[0] == "hold" and split_tag(e[1], people)[0] == n and "leash" in e[1] for e in read)]
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
        ink, wipe, seen_marks = resolve("mark", p["mark"]), resolve("unmark", p["unmark"]), []
        if not (ink or wipe):
            seen_marks, short = read_marks(p["text"], cast, people, skin)
            ink = [(w, [x]) for k, w, x, _ in seen_marks if k == "mark"]
            wipe = [(w, [x]) for k, w, x, _ in seen_marks if k == "unmark"]
            warn += [f"shot {i + 1}: \"{_cut(s)}\" was not read: {why}" for s, why in short]
        cast = list(dict.fromkeys(cast + [w for w, _ in holds + frees + wears + strips + tears + lifts + ink + wipe]))
        marked_off, marked_on = {}, {}
        for who, items in wipe:
            marked_off.setdefault(who, []).extend(x for x in skin.get(who, []) if x not in marked_off.get(who, [])
                                                  and (not items or any(_wiped(r, x) for r in items)))
        for who, items in ink:
            marked_on.setdefault(who, []).extend(x for x in items if x not in skin.get(who, []) and x not in marked_on.get(who, []))
        on_skin = {n: [x for x in skin.get(n, []) if x not in marked_off.get(n, [])] for n in cast}
        on_skin = {n: v for n, v in on_skin.items() if v}
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
        trail = {n: sorted(_footwork(" ".join(story), n, cast, people) + _obeyed(" ".join(story), n, cast, people), key=lambda f: f[2])
                 for n in cast}
        flat, hog = _facing(" ".join(story + [told])), {n for n in cast if _hogtied(held.get(n, []))}
        before = {n: "lie" if n in hog else _stance(trail[n]) for n in cast}
        shapes = {n: "lie" if n in hog else _shape([k for k, *_ in trail[n]]) for n in cast}
        notes = []
        if through:
            notes.append(" ".join(f"{cap(call(n))}: {x}." for n, v in through.items() for x in v)
                         + " All of it stays on, exactly as described, for the whole shot.")
        moves = [("hold", n, x) for n, v in added.items() for x in v] + [("release", n, x) for n, v in gone.items() for x in v] \
            + [("tear" if g in ripped.get(n, []) else "remove", n, g) for n, v in stripped.items() for g in v] \
            + [("lift", n, u) for n, v in raised.items() for u in v] + [("back", n, g) for n, v in back.items() for g in v] \
            + [("wear", n, g) for n, v in dressed.items() for g in v]
        moves = [(k, n, x, d if d in cast or d in strangers else None, s)
                 for k, n, x in sorted(moves, key=lambda m: hand(m[1], m[2], m[0])[2]) for d, s, _ in [hand(n, x, k)]]
        notes += _doing(moves, pal, call, held, {n: _stance(gait[n]) or before[n] for n in cast},
                        _facing(" ".join(story + [re.sub(_HOG, "", told, flags=re.I)])),
                        {n for n in cast if before[n] == "lie"}) if moves \
            else ["Everything happens at natural, real-life speed, in the order written, and every reaction comes only after what causes "
                  "it."]
        for n, v in added.items():
            keep = {_holder(x) for x in v} - {None}
            by = [d for d in dict.fromkeys(hand(n, x, "hold")[0] for x in v) if d not in keep]
            let = "" if not by or by == [n] else f", and {join(by)} {'have' if len(by) > 1 else 'has'} let go" \
                if all(d and d != n and (d in cast or d in strangers) for d in by) else ", and whoever put it on has let go"
            hid = [x for x in v if arms(x, pron(people.get(n, {}))).startswith("behind")]
            low = [x for x in v if x not in hid and "wrists" in parts(x) and _tied(x) and not _above(_tied(x)["to"])
                   and not _tied(x)["lies"]]
            seen = [x for x in v if x not in hid and x not in low]
            notes.append(f"By the last frame, {call(n)} has " + " and ".join(x for x in (
                f"{join(seen)} in plain view" if seen else "", f"{join(low)}, low at waist height" if low else "",
                join(hid) if hid else "") if x) + f"{let}.")
        notes += [f"By the last frame, {call(n)} no longer has {join(v)}." for n, v in gone.items() if v]
        for n, v in on_skin.items():
            one = len(v) == 1
            notes.append(f"{cap(call(n))} has {join(v)}. {'It stays' if one else 'They stay'} exactly as {'it is' if one else 'they are'} "
                         f"on {pron(people.get(n, {}))} skin for the whole shot: the same marks in the same places, in plain view, never "
                         f"fading, smudging, moving or changing."
                         + (" Every line and letter stays clear and exactly the same." if any(not _brand(x) for x in v) else ""))
        notes += [f"By the last frame, {call(n)} has {join(v)} in plain view." for n, v in marked_on.items() if v]
        notes += [f"By the last frame, {call(n)} no longer has {join(v)}." for n, v in marked_off.items() if v]
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
        kits = {g for v in (*through.values(), *added.values(), *gone.values()) for x in v for g in _gears(x)}
        spare = _spare(narration(" ".join(lead + sheets + [told])), kits, [] if p["cut"] else spares)
        spares = [] if _PUTAWAY.search(told) else spare
        ajar, opened = _ajar(narration(told), {} if p["cut"] else opened)
        plain, backed = narration(p["text"]), set()
        hog |= {n for n in cast if _hogtied(added.get(n, []))}
        bodies = {n: "prone" if x == "lie" and (flat is True or n in hog and not flat) else x
                  for n in cast for x in [_shape([k for k, *_ in gait[n]], shapes[n])]}
        acts = _coupling(plain, cast, people, bodies, " ".join(lead + sheets + story + [told]))
        moving = {n for n in cast for k, *_ in gait[n] if k in ("walk", "turn", "lie", "sit", "stairs", "hop")}
        fresh = [(g, r, "front" if w in ("standing", "astride") else w) for g, r, *_, w in acts
                 if w in ("behind", "bent", "side", "standing", "astride")] + _close(plain, cast, people)
        kept = [] if p["cut"] or _APART.search(plain) else [(y, x, k) for y, x, k in near if x in cast and x not in moving and (
            k == "wall" or y in cast and y not in moving) and not any({y, x} == {g, r} for g, r, *_ in acts)]
        slots = {}
        for y, x, k in sorted(fresh + kept, key=lambda t: ("hug", "front").index(t[2]) + 1 if t[2] in ("hug", "front") else 0):
            slots.setdefault((x, "wall" if k == "wall" else y), (y, x, k))
        near = list(slots.values())
        sides = {x for _, x, k in near if k == "side"} | {n for n in cast if flat == "side" and (bodies[n] == "lie" or n in hog)}
        for n in dict.fromkeys(list(through) + list(added)):
            per, was = people.get(n, {}), through.get(n, [])
            p_, o_, s_ = pron(per), pron(per, 2), pron(per, 0)
            kept = [x for x in added.get(n, []) if any(_swapped(x, [y]) for y in held.get(n, []))]
            posed, behind = _pose(call(n), per, was + kept, [x for x in added.get(n, []) if x not in kept],
                                  [k for k, *_ in gait.get(n, [])], before.get(n), joined,
                                  "side" if n in sides else flat is True, call, spare, ajar,
                                  bool(_TAUT.search(narration(told))), keyed, shapes.get(n) or ("stand" if i else None),
                                  i == 0 and first_frame)
            notes += posed
            backed |= {n} if behind else set()
            one = [pron(people.get(m, {}), 0) for m in cast].count(s_) == 1
            subject = rf"(?-i:\b{re.escape(n)})\b(?!['’])" + (rf"|^\W*(?:{_LEADS}\W+)*{s_}\b" if one else "")
            warn += [f"shot {i + 1}: \"{_cut(t)}\" needs {n}'s hands, but {p_} wrists are bound behind {p_} back, so the model "
                     f"may bring them forward to do it; free {o_} first or write it without {p_} hands"
                     for t in re.split(r"(?<=[.!?])\s+|\n+", plain)
                     if behind and any(arms(x, p_).startswith("behind") for x in was)
                     and (re.search(rf"(?:{subject})(?:\s+\w+ly)?\s+{_HANDY.pattern}", t, re.I) or n in _chores(t, cast, people))]
            legs = next((x for x in was if "ankles" in parts(x) and not re.search(r"\b(?:left|right) ankle\b|spread apart", x)
                         and _gear(x) != "a spreader bar"), None)
            fixed = (_tied(legs) or {}).get("to") if legs else None
            warn += [f"shot {i + 1}: \"{_cut(t)}\" needs {n}'s legs apart, but {p_} ankles are bound, so the model may break the "
                     f"restraint or draw an extra leg; free {p_} ankles first or write it without that" if k == "wide" else
                     f"shot {i + 1}: \"{_cut(t)}\" needs {n} to walk, but {p_} ankles are fastened to {fixed}, so the model may break "
                     f"the restraint or draw an extra leg; free {p_} ankles first or write it without that"
                     for k, t in dict.fromkeys((k, t) for k, t, *_ in gait.get(n, []))
                     if legs and (k == "wide" or fixed and k in ("walk", "stairs", "hop"))]
        chores = _chores(plain, cast, people)
        for n in cast:
            g, how = chores.get(n, (None, None))
            notes += [f"{cap(call(n))} does the {g} with both hands, {how}."] if g and "wrists" not in set().union(
                *map(parts, through.get(n, []) + added.get(n, []))) else []
        coupled = acts or ([] if p["cut"] else coupled)
        notes += [x for a in (acts or coupled * bool(_GOING.search(plain)))
                  for x in _coupled(a, call, people, bodies.get(a[1]), a[1] not in backed)]
        for x in dict.fromkeys(x for _, x, _ in near if x in backed):
            xp, xo = pron(people.get(x, {})), pron(people.get(x, {}), 2)
            arm = lambda y, k: f", under {pron(people.get(y, {}))} forearms" if k == "hug" else ""
            spots = [f"between {xp} back and {y}" if k == "wall" else f"between {xp} back and {call(y)}'s body" if k not in ("hug", "front")
                     else f"on the far side of {xp} body from {call(y)}{arm(y, k)}" for y, z, k in near if z == x]
            notes.append(f"{cap(call(x))}'s locked hands stay behind {xo} at the small of {xp} back, {', '.join(spots)}.")
        coupled, near = ([], [t for t in near if t[2] == "wall"]) if _PULLOUT.search(plain) else (coupled, near)
        for n in cast:
            if n not in people:
                continue
            per, desc = people[n], people[n]["desc"]
            p_, o_ = pron(per), pron(per, 2)
            took, up, put = stripped.get(n, []), raised.get(n, []), back.get(n, [])
            bonds, ends = through.get(n, []), through.get(n, []) + added.get(n, [])
            if took:
                notes.append(f"By the last frame, {call(n)}'s {'clothes' if '*' in took else join(took)} "
                             f"{'are' if '*' in took or _many(took) else 'is'} "
                             f"{'torn ' if set(took) <= set(ripped.get(n, [])) else ''}off.")
            for k, (g, where) in enumerate(up):
                one = not _many([g])
                leave = _leaving(per, _bares(g, where) - set().union(*(_bares(*u) for u in up[k + 1:])), desc, off2.get(n, []),
                                 worn2.get(n, []), lifted2.get(n, []), ends)
                notes.append(f"By the last frame, {call(n)}'s {g} {'is' if one else 'are'} pulled {where} and "
                             f"{'stays' if one else 'stay'} there, still on {o_}{leave}.")
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
                locked = [x for x in through.get(n, []) if "ankles" in parts(x) and "wrists" not in parts(x)]
                notes += [f"{cap(call(n))}'s {join(locked)} stay locked the whole time {p_} "
                          f"{'clothes' if '*' in took else join(took)} {'come' if many else 'comes'} off."] if locked else []
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
        marks = _spread(p["text"], [(a, b, t) for a, b, t, _ in _pace(p["text"], read, lasts, slow)])
        act = sum(t for _, _, t in marks) + (sum(t for k in GEAR_TAGS for a in p[k] for t in lasts(k, a)) if tagged else 0) \
            + sum(t for k in MARK_TAGS for a in p[k] for t in lasts(k, a))
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
        fresh = {w: d for k, a, _, d in read if k == "hold" and d for w, xs in [split_tag(a, people)] if w != d and any(map(_tied, xs))}
        idle = lambda d: not _SITED.search(p["text"]) and not re.search(
            rf"(?:(?-i:\b{re.escape(d)})|\b{pron(people.get(d, {}), 0)})\b(?!['’])\s+(?:\w+ly\s+)?(?!(?:{_IDLE})\b)[a-z]+",
            p["text"], re.I)
        notes += [f"{cap(call(d))} stays at {call(n)}'s left side, the only {call(d)} in the frame" + (
            f", and nobody is at {pron(people.get(n, {}))} right side" if len(cast) == 2 else "") + "."
                  for n, d in beside.items() if n not in fresh and {n, d} <= set(cast) and d not in leaving
                  and any(map(_tied, through.get(n, []))) and idle(d)]
        beside = {**{n: d for n, d in beside.items() if idle(d)}, **fresh}
        things = _fixtures(" ".join(story + [p["text"]]))
        notes += [f"There is only {join(['one ' + n for n in things])} here, {'each ' if len(things) > 1 else ''}always in the same "
                  f"spot."] if things else []
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
        moved = {n for n in cast if any(k in _BASE or k in _UPRIGHT or k in ("wide", "turn")
                                        for k, *_ in gait[n] + _obeyed(told, n, cast, people))} | {
            r for _, r, _, _, way in acts if way and way not in _LAID.get(bodies.get(r), ())}
        cuffs = lambda n: next((x for x in through.get(n, []) + added.get(n, []) if "wrists" in parts(x)), None)
        posed = lambda n, found, text, opened=False: _posture(found, text, pron(people.get(n, {})), any(
            "ankles" in parts(x) for x in through.get(n, []) + added.get(n, [])), cuffs(n) and (
            "cuffed" if "handcuffs" in _gears(cuffs(n)) else "bound", arms(cuffs(n), pron(people.get(n, {}))) or "together"), opened)
        poses = {n: posed(n, trail[n], " ".join(story), keyed) for n in staying if n not in moved}
        ends = {n: posed(n, gait[n] + _obeyed(told, n, cast, people), told) for n in staying if n in moved}
        bowed = [n for n in staying if (poses.get(n) or ends.get(n) or ("",))[0].startswith(("bent", "on all fours", "kneeling bent"))]
        how = lambda x: f": {join(x[1])}." if x[1] else "."
        if staying and not _CAMERA.search(p["text"]):
            shown = [n for n in staying if n not in bowed]
            faces = join([x for x in ((f"{call(shown[0])}'s face" if len(shown) == 1 else "the faces of " + join([call(n) for n in shown]))
                                      + " in frame" if shown else "", f"all of {join([call(n) for n in bowed])} in view" if bowed else "")
                          if x])
            feet = [(n, y) for n in staying for y in [_ankle_item(through.get(n, []) + added.get(n, []) + gone.get(n, []))] if y]
            low = list(dict.fromkeys(_low(y, pron(people.get(n, {})) if len(feet) == 1 else "their") for n, y in feet))
            far = "" if not feet else ", and stays far enough back to keep all of " + (
                pron(people.get(feet[0][0], {}), 2) if len(feet) == 1 == len(staying) else join([call(n) for n, _ in feet])) + \
                " in frame too, down to " + (low[0] if len(low) == 1 else join([_low(y, f"{call(n)}'s") for n, y in feet]))
            notes.append(f"The camera keeps {faces} for the whole shot{far}.")
        notes += [f"{cap(call(n))} stays {x[0]} for the whole shot, holding that position from the first frame to the last{how(x)}"
                  for n, x in poses.items() if x]
        notes += [f"{cap(call(n))} ends the shot {x[0]}{how(x)}" for n, x in ends.items() if x and (cuffs(n) or n in bowed)]
        notes += _kept(_filled(_props(" ".join(story[len(lead):]), " ".join(lead), cast, people), acts, people), told, call, cast)
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
        changes = bool(holds or frees or wears or strips or tears or lifts or ink or wipe)
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
        for who, items in marked_off.items():
            skin[who] = [x for x in skin.get(who, []) if x not in items]
        for who, items in marked_on.items():
            skin[who] = skin.get(who, []) + items
        off, worn, lifted = off2, worn2, lifted2
        listed = lambda state: " ".join(f"{n}: {'all' if v == ['*'] else '; '.join(v)}." for n, v in state.items() if v)
        spans = iter(cues)
        timed = [(ln, None if ln["thought"] else next(spans, None)) for ln in sounds]
        bits = [f"shot {i + 1}: {frames} frames ({frames / FPS:.1f}s)"] + [b for b in (
            fit, "after a cut" * p["cut"], f"beat {p['beat']}, part {p['part'][0]} of {p['part'][1]}" if "part" in p else "") if b]
        bits += ["read from the beat: " + "; ".join(f"{_READ.get(k, k)}: {a} (\"{_cut(s, 60)}\")" for k, a, s, _ in read)] \
            if read else []
        bits += ["marks read: " + "; ".join(f"{k}: {w}, {x} (\"{_cut(s, 60)}\")" for k, w, x, s in seen_marks)] if seen_marks else []
        bits += [f"{label}: {text}" for label, text in (("cast", ", ".join(cast)), ("on", listed(through)),
                                                       ("goes on", listed(added)), ("comes off", listed(gone)),
                                                       ("clothes off", listed(stripped)),
                                                       ("pulled", listed({n: [f"{g} {w}" for g, w in v] for n, v in raised.items()})),
                                                       ("pulled back", listed(back)), ("puts on", listed(dressed)),
                                                       ("marks", listed(on_skin)), ("marks added", listed(marked_on)),
                                                       ("marks off", listed(marked_off)),
                                                       ("leaves", ", ".join(leaving)),
                                                       ("unnamed", ", ".join(v["label"] for v in extras.values())),
                                                       ("speaks", "; ".join(f"{ln['who'] or '?'}{' (thinks)' if ln['thought'] else ''}"
                                                                            f"{' (muffled)' if ln['gag'] and not ln['thought'] else ''}"
                                                                            f"{f' {t[0]:.1f}-{t[1]:.1f}s' if t else ''}: "
                                                                            f"\"{_cut(ln['q'] or ln['verb'], 40)}\"" for ln, t in timed)),
                                                       ("pictures", ", ".join(f"ref_image_{k}" for k in refs))) if text]
        sound, loud = _sound(" ".join(y for y in (re.sub(_PIC, "", x, flags=re.I).strip() for x in lead) if y), told, through, added,
                             gone, hushed, music, cast)
        shots.append({"n": i + 1, "beat": p["beat"], "prompt": text, "frames": frames, "keyed": keyed, "cut": p["cut"], "cast": cast,
                      "refs": [k - 1 for k in refs], "faces": [k in owner for k in refs],
                      "silent": silence and not any(k == "line" for _, _, k in cues), "cues": cues, "changed": changes, "coming": coming,
                      "sound": sound, "mute": (silence or bool(cues)) and not (music or loud and not cues or _TONES.search(scene)),
                      "line": ", ".join(bits), "read": [(k, a) for k, a, *_ in read],
                      "said": [(ln["who"], ln["q"], bool(ln["gag"])) for ln in lines if not ln["thought"]]})
        warn += [f"shot {i + 1}: '{u}' was not read" for u in p["unread"]]
        warn += [f"shot {i + 1}: the line \"{_cut(ln['q'], 40)}\" has no speaker; name who says it, as in '\"…,\" "
                 f"{cast[0] if cast else 'Dana'} says'" for ln in lines if not ln["who"]]
        voice = next((ln["who"] for ln in reversed(lines) if ln["who"]), voice)
        present, prev, crowd = [n for n in cast if n not in leaving], cast, {k: v for k, v in extras.items() if k not in gone_by}
        i += 1
    return shots, warn
