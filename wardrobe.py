# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import re
from h3_restraints import sentences, resolve, stand_ins, tokens

GARMENT = (r"(?:t-?shirt|shirt|blouse|crop\s+top|tank\s+top|top|sweater|jumper|hoodie|cardigan|jacket|coat|blazer|vest|"
           r"(?:mini|micro|maxi|midi)?(?:dress|skirt)|gown|shorts|trousers|(?:hot|sweat|track)?pants|jeans|leggings|tights|"
           r"stockings|socks|shoes|boots|heels|sneakers|gloves|hat|cap|scarf|belt|robe|bathrobe|pajamas|pyjamas|nightgown|"
           r"nightie|lingerie|bra|bralette|panties|knickers|underwear|briefs|boxers|thong|g-string|bikini\s+(?:top|bottoms?)|"
           r"bikini|swimsuit|corset|bodysuit|catsuit|jumpsuit|playsuit|romper|leotard|camisole|negligee|chemise|babydoll|"
           r"diapers?|nappies|nappy|pampers|huggies|goodnites|pull-?ups|bloomers|plastic\s+pants|underpants|pantyhose|fishnets|onesies?|towels?|"
           r"nightdress|kimono|sarong|bodystocking|slippers|sandals|mittens|uniform|apron|bottoms|undies|slacks|joggers|"
           r"trunks|capris|culottes|girdle|speedos?|jockstrap|overalls|dungarees)")
_FUSED = r"(?:mini|micro|maxi|midi|hot|sweat|track)"
_LOWER = (r"skirt|shorts|trousers|pants|jeans|leggings|tights|panties|knickers|underwear|briefs|boxers|thong|g-string|"
          r"bikini bottoms?|diaper|pull-ups|bloomers|plastic pants|underpants|pantyhose|bottoms|undies|slacks|joggers|trunks|"
          r"capris|culottes|girdle|speedos?|jockstrap")
_UPPER = (r"t-?shirt|shirt|blouse|crop top|tank top|top|sweater|jumper|hoodie|cardigan|jacket|coat|blazer|vest|bra|bralette|"
          r"bikini top|camisole|corset")
_WHOLE = (r"dress|gown|robe|bathrobe|pajamas|pyjamas|nightgown|nightie|lingerie|bikini|swimsuit|bodysuit|catsuit|jumpsuit|"
          r"playsuit|romper|leotard|negligee|chemise|babydoll|uniform|onesie|towel|nightdress|kimono|sarong|bodystocking|"
          r"clothes|overalls|dungarees")
_TAKE = (r"takes?|took|taking|pulls?|pulled|pulling|slips?|slipped|slides?|slid|peels?|peeled|strips?|stripped|tugs?|"
         r"tugged|yanks?|yanked|rips?|ripped|tears?|tore|cuts?|kicks?|kicked|shrugs?|shrugged|wriggles?|wriggled|"
         r"lowers?|lowered|works?|worked|eases?|eased|lifts?|lifted|push(?:es)?|pushed|gets?|got|getting")
_OFF_RX = (r"\b(?:remov(?:e|es|ed|ing)|sheds?|drops?|dropped|dropping|untap(?:e|es|ed|ing))\s+"
           r"(?P<pre>(?:[\w'’-]+\s+){{0,4}}?)(?P<g>{g})\b"
           r"|\b(?:steps?|stepped|stepping|wriggles?|wriggled|wiggles?|wiggled|slips?|slipped|climbs?|climbed|shimmies|"
           r"shimmied|gets?|got|getting)\s+out\s+of\s+(?P<pre3>(?:[\w'’-]+\s+){{0,4}}?)(?P<i>{g})\b"
           r"|\blets?\s+(?P<pre4>(?:[\w'’-]+\s+){{0,4}}?)(?P<j>{g})\s+(?:fall|drop|slip|slide|pool)\b"
           r"|\b(?:{take})\s+(?P<off>(?:off|of|down)\s+)?(?P<pre2>(?:[\w'’-]+\s+){{0,4}}?)(?P<h>{g})\b"
           r"(?P<tail>\s+(?:off|away|down|free)\b|\s+over\s+(?:her|his|their)\s+head\b)?"
           r"|\b(?P<pre5>(?:her|his|their|the|(?-i:[A-Z])[a-z]+['’]s)\s+(?:[\w'’-]+\s+){{0,2}}?)(?P<k>{g})\s+"
           r"(?:(?:comes?|came)\s+(?:off|away|down)|(?:falls?|fell|drops?|dropped|slips?|slipped|slides?|slid|pools?|"
           r"pooled)\s+(?:off|away|down|free|(?:to|onto)\s+the\s+(?:floor|ground)|(?:to|around)\s+(?:her|his|their)\s+"
           r"(?:ankles|feet))\b)")
_OFF = re.compile(_OFF_RX.format(g=GARMENT, take=_TAKE), re.I)
_NAKED = re.compile(r"\b(?:(?P<act>strips?|stripped|stripping|gets?|got|getting)|is|was|are|stands?|stood|kneels?|knelt|"
                    r"lies|lay|lying|sits?|sat|waits?|remains?|stays?)\s+(?:down\s+)?(?:[\w'’-]+\s+){0,3}?(?:completely\s+|"
                    r"fully\s+|totally\s+|stark\s+)?(?P<n>naked|nude|topless|bottomless)\b"
                    r"|\b(?:completely|fully|totally|stark)\s+(?P<n2>naked|nude)\b"
                    r"|\b(?P<act2>undress(?:es|ed|ing)?)(?:\s+(?P<obj>her|him|(?-i:[A-Z])[a-z]+)\b)?"
                    r"|\b(?P<act3>takes?|took|taking|strips?|stripped|pulls?|pulled|tears?|tore|rips?|ripped)\s+(?:off\s+)?"
                    r"(?:all\s+(?:of\s+)?)?(?P<own>her|his|(?-i:[A-Z])[a-z]+['’]s)\s+clothes\b", re.I)
_SHEET = re.compile(r"^\s*([A-Z][\w'’-]{0,24}(?:\s+[A-Z][\w'’-]{0,24}){0,2})\s*:\s*\S")
_ON_RX = (r"\b(?:puts?|putting|pulls?|pulled|slips?|slipped|slides?|slid|gets?|got|tugs?)\s+"
          r"(?P<pre>(?:[\w'’-]+\s+){{0,3}}?)(?P<g>{g})\s+(?:back\s+)?on\b(?!\s+(?:the|a|an|its|this|that|these|those|"
          r"some|top|to|onto)\b|\s+(?:his|her|their|my|your|our)\s+(?!(?:and|as|while|then|before|after|again|so|but)\b)"
          r"[a-z]|\s+(?-i:[A-Z])[a-z]+['’]s\b)"
          r"|\b(?:puts?|putting|pulls?|slips?)\s+(?:back\s+)?on\s+(?P<pre2>(?:[\w'’-]+\s+){{0,3}}?)(?P<h>{g})\b"
          r"|\b(?:slips?|slipped|steps?|stepped|wriggles?|wriggled|gets?|got|climbs?|climbed|changes?|changed|shimmies|"
          r"shimmied)\s+(?:back\s+)?into\s+(?P<pre3>(?:[\w'’-]+\s+){{0,3}}?)(?P<i>{g})\b"
          r"|\b(?:fasten(?:s|ed|ing)?|tap(?:e|es|ed|ing)|secur(?:e|es|ed|ing)|straps?|strapped|wraps?|wrapped|ties?|tied)\s+"
          r"(?P<pre4>(?:[\w'’-]+\s+){{0,3}}?)(?P<j>{g})\s+(?:(?:back\s+)?(?:on|around|onto)\s+{body}|(?:shut|closed)\b)"
          r"|\b(?P<pre5>(?:her|his|their|the|(?-i:[A-Z])[a-z]+['’]s)\s+(?:[\w'’-]+\s+){{0,2}}?)(?P<k>{g})\s+(?:is|are)\s+"
          r"(?:now\s+)?(?:(?:secured|fastened|taped|strapped|firmly|snugly|back)\s+)*on\s+{body}")
_BODY = (r"(?:(?:her|him|them)\b(?!\s+[a-z])|(?:her|his|their|(?-i:[A-Z])[a-z]+['’]s)\s+(?:body|hips|waist|bottom|butt|"
         r"crotch|chest|legs|feet|head)\b|(?-i:[A-Z])[a-z]+\b(?!['’]))")
_DRESSED = re.compile(r"\b(?:gets?|got|getting)\s+(?:back\s+)?dressed\b|\b(?:puts?|putting|pulls?|slips?)\s+(?:all\s+)?"
                      r"(?P<own>her|his|their|(?-i:[A-Z])[a-z]+['’]s)\s+clothes\s+(?:back\s+)?on\b|\b(?:puts?|putting)\s+"
                      r"(?:back\s+)?on\s+(?:all\s+)?(?P<own2>her|his|their|(?-i:[A-Z])[a-z]+['’]s)\s+clothes\b"
                      r"|\b(?:gets?|got|slips?|climbs?)\s+(?:back\s+)?into\s+(?P<own3>her|his|their)\s+clothes\b"
                      r"|\bdresses\s+(?P<obj>her|him|(?-i:[A-Z])[a-z]+)\b(?!['’])", re.I)
_ON = re.compile(_ON_RX.format(g=GARMENT, body=_BODY), re.I)
_WEARS = re.compile(r"\b(?:wears?|wearing|wore|dressed\s+in|clad\s+in)\s+(?:only\s+|just\s+|nothing\s+but\s+)?"
                    r"(?P<list>[^.;:!?]+)", re.I)
_NOT_WORN = {"smile", "frown", "grin", "look", "expression", "makeup", "perfume", "lipstick", "nothing", "anything", "everything",
             "black", "white", "red", "blue", "green", "pink", "purple", "grey", "gray", "brown", "yellow", "orange", "silver",
             "gold", "leather", "latex", "lace", "silk", "cotton", "denim", "satin", "velvet", "rubber", "pvc", "vinyl", "tight",
             "loose", "short", "long", "thick", "thin", "only", "just", "hair", "glasses", "contacts", "tattoos", "piercings",
             "ring", "rings", "earrings", "necklace", "bracelet", "watch", "jewelry", "jewellery"}


def learned(described):
    out = []
    for text in (described or {}).values():
        for m in _WEARS.finditer(text or ""):
            for part in re.split(r",|\band\b|\bwith\b|\bover\b|\bunder\b", m.group("list"), flags=re.I):
                words = re.findall(r"[a-z][a-z-]{2,}", part.lower())
                if words and words[-1] not in _NOT_WORN and not re.fullmatch(rf"{GARMENT}", words[-1], re.I):
                    out.append(words[-1])
    return list(dict.fromkeys(out))


def _rx(extra):
    g = GARMENT if not extra else rf"(?:{GARMENT}|{'|'.join(map(re.escape, extra))})"
    return re.compile(_OFF_RX.format(g=g, take=_TAKE), re.I), re.compile(_ON_RX.format(g=g, body=_BODY), re.I), g


_SINGLE = {"nappies": "diaper", "nappy": "diaper", "diapers": "diaper", "pampers": "diaper", "huggies": "diaper",
           "goodnites": "diaper", "onesies": "onesie", "towels": "towel",
           "pullups": "pull-ups", "pull ups": "pull-ups"}


def garment_of(word):
    word = re.sub(rf"^{_FUSED}(?=(?:dress|skirt|pants)$)", "", re.sub(r"\s+", " ", word.lower()))
    return _SINGLE.get(word, word)


def _garment_rx(garment):
    if garment == "diaper":
        return r"\b(?:diapers?|napp(?:y|ies)|pampers|huggies|goodnites)\b"
    stem = re.escape(garment).replace(r"\-", r"[\s-]?")
    return rf"\b{_FUSED}?" + (rf"{stem[:-1]}(?:y|ies)\b" if garment.endswith("y") else rf"{stem}s?\b")


def mentions(text, garment):
    return bool(re.search(_garment_rx(garment), text or "", re.I))


def unsay(text, gone, zones=(), extra=()):
    found = []
    covers = lambda g: any(re.fullmatch(rf"{_LOWER if z == 'lower' else _UPPER}|{_WHOLE}", g) for z in zones)
    rx = re.compile(rf"\b{_rx(extra)[2]}(?:e?s)?\b", re.I)
    words = lambda s: re.findall(r"[\w'’-]+", s)

    def named(sentence):
        hits = [w for w in (m.group(0) for m in rx.finditer(sentence))
                if any(covers(garment_of(w[:len(w) - k])) for k in (0, 1, 2)) or any(mentions(w, g) for g in gone)]
        found.extend(garment_of(w) for w in hits)
        return bool(hits)

    def line(d):
        said = re.split(r"(?<=[.!?])\s+", d.group(1).strip())
        dirty = [named(s) for s in said]
        if not any(dirty):
            return d.group(0)
        pool = [w for s in said[:dirty.index(True)] for w in words(s)] or [
            w for s, x in zip(said, dirty) if not x for w in words(s)]
        if not pool:
            return ""
        fill = lambda n: " ".join(pool[k % len(pool)] for k in range(n))
        return "<d>" + " ".join(s if not x else (lambda f: f[:1].upper() + f[1:] + ".")(fill(len(words(s))))
                                for s, x in zip(said, dirty)) + "</d>"
    return re.sub(r"<d>(.*?)</d>", line, text or "", flags=re.S), list(dict.fromkeys(found))


ACCESSORY = (r"(?:bows?|ribbons?|necklaces?|bracelets?|anklets?|rings?|earrings?|crowns?|tiaras?|bonnets?|headbands?|hairbands?|"
             r"hair\s*clips?|barrettes?|scrunchies|scrunchie|flowers?|garlands?|leis?|glasses|sunglasses|masks?|pacifiers?|"
             r"dummies|dummy|bibs?|wristbands?|watch(?:es)?|chokers?|pendants?|lockets?|ties|tie|badges?|brooch(?:es)?|bandages?|"
             r"plasters?|wigs?|veils?|headphones|earplugs|sash(?:es)?|capes?|cloaks?|shawls?|garters?|pins?|clips?|zippers?|patch(?:es)?)")
_PART = (r"hair|head|neck|throat|wrists?|ankles?|fingers?|thumbs?|ears?|face|eyes|mouth|nose|lips|waist|hips|chest|breasts?|"
         r"arms?|legs?|thighs?|feet|foot|toes?|cheeks?|forehead|shoulders?|belly|tummy|body|back|torso|stomach|crotch|"
         r"bottom")
_PUT = (r"puts?|putting|places?|placed|placing|slips?|slipped|clips?|clipped|ties?|tied|fastens?|fastened|pins?|pinned|hangs?|"
        r"hung|straps?|strapped|buckles?|buckled|sets?|loops?|looped|drapes?|draped|pops?|popped|slides?|slid|wraps?|wrapped|"
        r"attach(?:es|ed|ing)?|sews?|sewed|sewn|sewing")
_NP = rf"(?P<np>(?:a|an|the|some|his|her|their|(?-i:[A-Z])[a-z]+['’]s)\s+(?:[\w'’-]+\s+){{{{0,3}}}}?(?P<item>{{g}}))"
_ADDS = (rf"\b(?:{_PUT})\s+{_NP}\s+(?P<prep>on|onto|in|into|around|over|to|through)\s+(?P<own>her|his|their|"
         rf"(?-i:[A-Z])[a-z]+['’]s)\s+(?P<part>{_PART}|{GARMENT})\b",
         rf"\b(?:{_PUT})\s+on\s+{_NP}",
         rf"\b(?:{_PUT})\s+{_NP}\s+(?:back\s+)?(?:on|onto)\s+(?P<obj>her|him|them|(?-i:[A-Z])[a-z]+)\b(?!['’]|\s+[a-z])",
         rf"\b(?:{_PUT})\s+{_NP}\s+(?:back\s+)?on\b(?=\s*(?:[,.;!?]|and\b|$))",
         rf"{_NP}\s+(?:is|are|was|were|gets?|got)\s+(?:being\s+)?(?:{_PUT})\s+(?P<prep>on|onto|in|into|around|over|to)\s+"
         rf"(?P<own>her|him|his|them|their|(?-i:[A-Z])[a-z]+(?:['’]s)?)(?:\s+(?P<part>{_PART}|{GARMENT}))?\b")
_TAKES = (rf"\b(?:removes?|removed|removing|unclips?|unclipped|unties?|untied|unfastens?|unfastened|unbuckles?|unbuckled)\s+{_NP}",
          rf"\b(?:takes?|took|taking|pulls?|pulled|slips?|slipped|plucks?|plucked|tugs?|tugged)\s+{_NP}\s+(?:off|out|away|from)\b",
          rf"\b(?:takes?|took|taking|pulls?|pulled|slips?|slipped)\s+(?:off|out)\s+{_NP}")


def _head(word):
    return garment_of(re.sub(r"(?<=[^s])e?s$", "", word.lower()) if word.lower() not in ("glasses", "sunglasses") else word)


def additions(text, people, gender, described, extras, here=(), extra=()):
    new, put, took = {k: list(v) for k, v in extras.items()}, {}, {}
    item = rf"(?:{ACCESSORY}|{_rx(extra)[2]})"
    pos = lambda who: {"f": "her", "m": "his"}.get(gender.get(who), "their")
    for sentence in sentences(text):
        subject = lambda at: next((t for _, t in reversed(tokens(sentence[:at], people))), "")
        for k, pat in enumerate(_ADDS):
            for m in re.finditer(pat.format(g=item), sentence, re.I):
                d = m.groupdict()
                actor = subject(m.start())
                tok = re.sub(r"['’]s$", "", d.get("own") or d.get("obj") or "")
                who = (resolve(tok, people, gender, exclude=[actor] if tok.lower() in ("her", "his", "him", "their", "them")
                               else (), here=here) if tok else resolve(actor, people, gender, here=here) if actor else "")
                if not who or (re.fullmatch(GARMENT, d["item"], re.I) and mentions(described.get(who, ""), d["item"])):
                    continue
                np = re.sub(r"^(?:the|some)\s+(\S)", lambda x: ("" if re.search(r"[^s]s$", d["item"], re.I) else "an "
                            if x.group(1).lower() in "aeiou" else "a ") + x.group(1), d["np"], flags=re.I)
                np = re.sub(r"^An?\b", lambda x: x.group(0).lower(), np)
                prep = {"to": "on", "onto": "on", "into": "in"}.get(d.get("prep", "").lower(), d.get("prep"))
                phrase = np + (f" {prep} {pos(who)} {d['part']}" if d.get("part") else "")
                if not any(_head(d["item"]) == _head(re.findall(r"[\w'’-]+", x)[-1]) or _head(d["item"]) in x
                           for x in new.get(who, [])):
                    new.setdefault(who, []).append(phrase)
                    put.setdefault(who, []).append(phrase)
        for pat in _TAKES:
            for m in re.finditer(pat.format(g=item), sentence, re.I):
                head = _head(m.group("item"))
                for who, have in new.items():
                    gone = [x for x in have if re.search(rf"\b{re.escape(m.group('item'))}\b", x, re.I) or
                            any(_head(w) == head for w in re.findall(r"[\w'’-]+", x))]
                    new[who] = [x for x in have if x not in gone]
                    took.setdefault(who, []).extend(gone) if gone else None
    return {k: v for k, v in new.items() if v}, put, took


def worn(text, extra=()):
    return [garment_of(m.group(0)) for m in re.finditer(rf"\b{_rx(extra)[2]}\b", text or "", re.I)]


_UNDER = (r"bra|bralette|panties|knickers|underwear|briefs|boxers|thong|g-string|diaper|pull-ups|bloomers|plastic pants|"
          r"underpants|undies|girdle|jockstrap")
_SHEER = r"lingerie|tights|pantyhose|stockings|fishnets|bodystocking"


def covered(described, gone, coming=()):
    left = [g for g in dict.fromkeys(worn(described, learned({"": described}))) if not any(mentions(g, x) for x in gone)]
    outer = [g for g in left if g not in coming and not re.fullmatch(rf"{_UNDER}|{_SHEER}", g)]
    zone = lambda g: _LOWER if re.fullmatch(_LOWER, g) else _UPPER if re.fullmatch(_UPPER, g) else None
    return {g: next(o for o in outer if re.fullmatch(rf"{zone(g)}|{_WHOLE}", o)) for g in left
            if re.fullmatch(_UNDER, g) and zone(g) and any(re.fullmatch(rf"{zone(g)}|{_WHOLE}", o) for o in outer)}


def bare(described, gone):
    left = [g for g in worn(described, learned({"": described})) if not any(mentions(g, x) for x in gone)]
    covers = lambda g, zone: re.fullmatch(rf"{zone}|{_WHOLE}", g) is not None
    return [name for zone, name in ((_LOWER, "lower"), (_UPPER, "upper"))
            if any(covers(g, zone) for g in gone) and not any(covers(g, zone) for g in left)]


_DET = {"a", "an", "her", "his", "their", "the", "my", "its", "some", "one"}
_STOP = {"wears", "wear", "wearing", "worn", "is", "are", "was", "were", "in", "with", "and", "or", "only", "just", "but",
         "of", "on", "under", "over", "into", "to", "from", "she", "he", "they", "still", "now", "also", "nothing"}


_TRAIL = re.compile(rf"(?:\s+(?:underwear|panties|briefs|bottoms?|knickers))?(?P<clause>\s*,?\s+(?P<head>made\s+(?:of|from)|of|"
                    rf"in|with|cut|trimmed|lined|edged|covered|that|which|stretched|clinging|pulled|riding|hugging)\b"
                    rf"(?P<rest>.*?)(?=\s*(?:[,;.!?]|\band\s+(?:a|an|the|her|his|their)\b|$)|\s+(?:a|an|the|her|his|their|"
                    rf"matching)\s+(?:[\w-]+\s+){{0,2}}?(?:{GARMENT})\b))?", re.I)
_AROUND = re.compile(r"^\s*(?:(?:the|this|that|these|those|her|his|their|my|your|our)\b|(?-i:[A-Z])[\w-]*['’]s\b)", re.I)
_LEFT = re.compile(r"\b(?:is\s+wearing|wears|wearing|wore|dressed\s+in|clad\s+in|in|with)(?:\s+only|\s+just|"
                   r"\s+nothing\s+but)?\s*$", re.I)
_ITS = re.compile(r"^\s*It(?:['’]s|s|\s+(?:is|was|has))\b", re.I)


def strip(text, garment, extra=(), names=()):
    more = re.compile(rf"\s*(?:(?:over|under|on\s+top\s+of|with)\s+)?(?:(?!(?:by|at|on|in|into|onto|near|behind|beside|"
                      rf"against|while|as|with|to|from)\b)[\w'’-]+\s+){{0,5}}?{_rx(extra)[2]}\b", re.I)
    rx = _garment_rx(garment)
    while True:
        m = re.search(rf"{rx}(?:\s+{rx})*", text, re.I)
        if m is None:
            break
        start = m.start()
        chunk = max(text.rfind(c, 0, start) for c in ",;:.!?()[]<>\n") + 1
        for w in reversed(list(re.finditer(r"[\w'’-]+", text[chunk:start]))[-8:]):
            low = w.group(0).lower()
            if low in _STOP or w.group(0) in names or re.fullmatch(r"(?-i:[A-Z])[\w-]*['’]s", w.group(0)):
                break
            start = chunk + w.start()
            if low in _DET:
                break
        before, after = text[:start], text[m.end():]
        t = _TRAIL.match(after)
        kept = t.group("clause") and t.group("head").lower() in ("of", "in", "with") and _AROUND.search(t.group("rest"))
        after = after[t.start("clause") if kept else t.end():]
        loose = kept or re.match(r"\s*(?:[,.;:!?]|$)", after)
        if re.search(r",\s*$", before):
            before = re.sub(r",\s*$", "", before)
        elif re.search(r"\s+and\s*$", before):
            before = re.sub(r"\s+and\s*$", "", before)
        elif re.match(r"\s*,\s*", after):
            after = re.sub(r"^\s*,\s*", " ", after)
        elif re.match(r"\s+and\s+", after):
            after = re.sub(r"^\s+and\s+", " ", after)
        if _LEFT.search(before):
            before = _LEFT.sub("", before) if loose and not more.match(after) else before
            after = re.sub(r"^\s*and\s+", " ", after)
        text = before + after
    text = re.sub(r"\b(wears|wearing|in)\s+(?:over|under|on\s+top\s+of|with)\s+", r"\1 ", text, flags=re.I)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*([.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return "" if re.fullmatch(r"(?:[A-Z][\w'’-]*|she|he|they)?\s*[.!?]?", text, re.I) else text


def _owners(text, people, last=None):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        named = [n for n in people if re.search(rf"\b{re.escape(n)}\b", s)]
        last = named[0] if len(named) == 1 else (None if named else last)
        out.append((s, named or ([last] if last else [])))
    return out


LABELS = {"name", "names", "character", "characters", "looks", "look", "hair", "eyes", "outfit", "outfits", "clothes",
          "clothing", "wears", "wearing", "age", "height", "build", "body", "face", "skin", "personality", "voice", "style",
          "role", "description", "appearance", "notes", "note", "gender", "sex", "race", "species", "type", "class",
          "details", "features", "accessories", "shoes", "makeup", "tattoos", "piercings", "background", "traits",
          "nationality", "ethnicity", "occupation", "job", "relationship"}
SCENE = {"setting", "location", "scene", "mood", "lighting", "light", "time", "place", "weather", "music", "audio", "sound",
         "camera", "shot", "style", "note", "notes"}
_HEAD = re.compile(r"^\s*(?:[-*•>#]+\s*|\d+[.)]\s+)?(?:\*\*|__)?(?:(?:U\.?S\.?S\.?|HMS|RRS)\s+)?"
                   r"(?P<name>(?-i:[A-Z])[\w'’-]{0,24}(?:\s+(?![A-Z]{1,4}-\d)(?-i:[A-Z])[\w'’-]{0,24}){0,2})(?:\*\*|__)?"
                   r"(?:\s+\(?[A-Z]{1,4}-\d[\w-]*\)?)?"
                   r"(?:\s*\([^)]*\))?(?:\s*<\s*picture[\s_\-]*\d+\s*>)?(?:\s*\([^)]*\))?(?:\*\*|__)?"
                   r"\s*(?P<sep>[:=]|[-–—]\s)?\s*(?P<rest>.*)$", re.I)


def plain(name):
    return name.title() if name.isupper() and len(name) > 1 else name


def header(line):
    m = _HEAD.match(line or "")
    if not m:
        return None
    name, rest = m.group("name"), m.group("rest").strip()
    if name.lower() in LABELS:
        if name.lower() in ("name", "character") and re.fullmatch(r"(?:[A-Z][\w'’-]{0,24}\s*){1,3}", rest):
            return plain(rest.strip())
        return ""
    return plain(name) if m.group("sep") or not rest else None


def line_owners(text, names):
    out, owner = [], None
    for line in (text or "").splitlines():
        found = header(line) if line.strip() else None
        head = _HEAD.match(line)
        if not line.strip() or head and head.group("name").lower() in SCENE:
            owner = None
        elif found:
            owner = found if found in names else None
        out.append((line, owner))
    return out


_FIELD = re.compile(rf"^\s*(?:[-*•>#]+\s*|\d+[.)]\s+)?(?:(?:{'|'.join(sorted(LABELS))})\s*:\s*)?", re.I)


def attributed(text, names):
    lines = []
    for line, owner in line_owners(text, names):
        if owner:
            head = _HEAD.match(line)
            body = head.group("rest") if head and header(line) else line
            line = f"{owner}: {_FIELD.sub('', body).strip()}"
        lines.append(line)
    return "\n".join(lines)


def descriptions(text, people):
    out = {}
    for line, owner in line_owners(text, people):
        sheet = _SHEET.match(line)
        for s, who in _owners(line, people, owner or (sheet.group(1) if sheet and sheet.group(1) in people else None)):
            for n in who:
                out[n] = (out.get(n, "") + " " + s).strip()
    return out


def undress(text, undressed, people):
    if not undressed:
        return text
    lines, extra = [], learned({"": text})
    for line, owner in line_owners(text, people):
        sheet = _SHEET.match(line)
        kept, gone = [], False
        for s, who in _owners(line, people, owner or (sheet.group(1) if sheet and sheet.group(1) in people else None)):
            if gone and _ITS.match(s):
                continue
            gone = False
            for n in who:
                for g in undressed.get(n, []):
                    gone = gone or mentions(s, g)
                    s = strip(s, g, extra, people)
            kept.append(s)
        lines.append(" ".join(k for k in kept if k))
    return "\n".join(lines)


def nudity(sentence, people, gender, here=(), unclear=None, extra=()):
    out = []
    for m in _NAKED.finditer(sentence):
        if re.search(r"\b(?:not|never|half|no\s+longer)\b|n['’]t\b", m.group(0), re.I):
            continue
        tok = (m.group("obj") or re.sub(r"['’]s$", "", m.group("own") or "")) or next(
            (t for _, t in reversed(tokens(sentence[:m.start()], people))), "")
        who = resolve(tok, people, gender, here=here) if tok else ""
        if not who:
            if unclear is not None:
                unclear.append(f"who is naked in '{sentence.strip()}'")
            continue
        word, after = (m.group("n") or m.group("n2") or "").lower(), sentence[m.end():]
        zones = (["upper"] if word == "topless" else ["lower"] if word == "bottomless" else
                 ["lower"] if re.match(r"\s+from\s+the\s+waist\s+down\b", after, re.I) else
                 ["upper"] if re.match(r"\s+from\s+the\s+waist\s+up\b", after, re.I) else ["lower", "upper"])
        out.append((who, [{"lower": _LOWER, "upper": _UPPER}[z] for z in zones],
                    not (m.group("act") or m.group("act2") or m.group("act3"))))
    return out


def stated(text, people, gender, described, here=()):
    out, extra = {}, learned(described)
    for sentence in sentences(text):
        for who, zones, state in nudity(sentence, people, gender, here):
            if state:
                gone = [g for g in worn(described.get(who, ""), extra)
                        if len(zones) == 2 or any(re.fullmatch(rf"{z}|{_WHOLE}", g) for z in zones)]
                out.setdefault(who, []).extend(g for g in gone or (["clothes"] if len(zones) == 2 else [])
                                               if g not in out.get(who, []))
    return out


def read(text, people, gender, described, here=()):
    off, on, unclear, blocked = {}, {}, [], stand_ins(text)
    here = [n for n in people if n in here or re.search(rf"\b{re.escape(n)}\b", text)]
    extra = learned(described)
    off_rx, on_rx, _ = _rx(extra)
    last = ""
    for sentence in sentences(text):
        named = list(re.finditer(rf"(?:\b(?P<det>her|his|their|the|(?-i:[A-Z])[a-z]+['’]s)\s+(?:[\w'’-]+\s+){{0,2}}?)?"
                                 rf"\b(?P<g>{_rx(extra)[2]})\b", sentence, re.I))
        sentence = re.sub(r"\b(?:it|them)\b", lambda m: next(
            (f"{n.group('det') or 'the'} {n.group('g')}" for n in reversed(named) if n.end() <= m.start()), last) or m.group(0),
            sentence)
        last = f"{named[-1].group('det') or 'the'} {named[-1].group('g')}" if named else ""
        for who, zones, _ in nudity(sentence, people, gender, here, unclear):
            gone = [g for g in worn(described.get(who, ""), extra)
                    if len(zones) == 2 or any(re.fullmatch(rf"{z}|{_WHOLE}", g) for z in zones)]
            for garment in gone or (["clothes"] if len(zones) == 2 else []):
                if garment not in off.get(who, []):
                    off.setdefault(who, []).append(garment)
        for m in _DRESSED.finditer(sentence):
            tok = re.sub(r"['’]s$", "", next((m.group(k) for k in ("own", "own2", "own3", "obj") if m.group(k)), "") or "")
            tok = tok if tok.lower() not in ("their",) else ""
            tok = tok or next((t for _, t in reversed(tokens(sentence[:m.start()], people))), "")
            who = resolve(tok, people, gender, here=here) if tok else ""
            if who and "clothes" not in on.get(who, []):
                on.setdefault(who, []).append("clothes")
        for rx, out in ((off_rx, off), (on_rx, on)):
            for m in rx.finditer(sentence):
                if rx is off_rx and m.group("h") and not (m.group("off") or m.group("tail")):
                    continue
                key = next(k for k in ("g", "h", "i", "j", "k") if k in rx.groupindex and m.group(k))
                garment = garment_of(m.group(key))
                own = re.search(r"\b((?i:her|his|their))\b|\b([A-Z][a-z]+)['’]s\b",
                                m.group({"g": "pre", "h": "pre2", "i": "pre3", "j": "pre4", "k": "pre5"}[key]) or "")
                if own and own.group(1) and ({"her": "f", "his": "m"}.get(own.group(1).lower(), "fm") in blocked or
                                             own.group(1).lower() == "their" and blocked):
                    unclear.append(f"whose {garment} in '{sentence.strip()}'")
                    continue
                want = {"her": "f", "his": "m"}.get((own.group(1) or "").lower()) if own else None
                owners = [n for n in people if mentions(described.get(n, ""), garment)
                          and (want is None or gender.get(n, want) == want)]
                who = (own.group(2) if own and own.group(2) in people else "") or (owners[0] if len(owners) == 1 else "") or (
                    resolve(own.group(1) or own.group(2), people, gender, here=here) if own else "")
                if not who:
                    unclear.append(f"whose {garment} in '{sentence.strip()}'")
                    continue
                if garment not in out.get(who, []):
                    out.setdefault(who, []).append(garment)
    return off, on, unclear
