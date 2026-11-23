# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.

import re

FPS = 24
MIN_FRAMES = 124
MAX_FRAMES = 362
PIC = r"<\s*picture[\s_\-]*(\d+)\s*>"

LOWER = r"pants|trousers|jeans|leggings|shorts|(?:mini)?skirts?|joggers|sweatpants|slacks|bottoms|capris|culottes"
UPPER = r"(?:t-|tank\s+|crop\s+|tube\s+)?(?:shirts?|tops?)|blouses?|sweaters?|jumpers?|hoodies?|sweatshirts?|tees?|cardigans?|" \
        r"jackets?|blazers?|(?:rain|over|pea|top|fur)?coats?|vests?|camisoles?|corsets?|bustiers?|halters?|ponchos?|shawls?|parkas?"
WHOLE = r"(?:summer\s+|night|mini|sun)?dress(?:es)?|(?:night)?gowns?|nighties|robes?|bathrobes?|kimonos?|jumpsuits?|overalls|" \
        r"onesies?|pajamas|pyjamas|uniforms?|suits?|costumes?|babydolls?|negligees?|chemises?|rompers?|playsuits?|sarongs?|" \
        r"bodysuits?|leotards?|swimsuits?|bikinis?|catsuits?"
UNDER = r"panties|knickers|thongs?|g-strings?|underwear|briefs|boxers|bras?|bralettes?|lingerie|diapers?|nappy|nappies|pull-?ups"
HOSE = r"tights|pantyhose|stockings|socks"
SHOES = r"shoes|boots|heels|sneakers|sandals|slippers|trainers|flats|pumps|loafers"
EXTRA = r"gloves|hats?|caps?|scarf|scarves|glasses"
SKIRT = r"(?:mini)?skirts?"
LOOSE = rf"{SKIRT}|(?:summer\s+|night|mini|sun)?dress(?:es)?|(?:night)?gowns?|nighties|(?:bath)?robes?|kimonos?|babydolls?|" \
        r"negligees?|chemises?|sarongs?"
GARMENT = re.compile(rf"\b(?:{UPPER}|{LOWER}|{WHOLE}|{UNDER}|{HOSE}|{SHOES}|{EXTRA})\b", re.I)
SHEER = re.compile(r"\b(?:sheer|see-?through|mesh|fishnet|open|unbuttoned|unzipped)\b", re.I)
HUE = r"red|black|white|pink|blue|green|purple|yellow|orange|grey|gray|silver|gold|golden|brown|beige|tan|clear|dark|light|" \
      r"shiny|glossy|leather|rubber|latex|silicone|steel|metal|iron|chrome|brass|wooden|silk|satin|velvet|lace|nylon|cotton|hemp|" \
      r"jute|padded|fur|furry|velcro|plastic|pvc|vinyl|heavy|thick|thin|wide|studded|locking|soft|braided|rigid"
SOFT = re.compile(r"\b(?:leather|padded|fur|furry|velcro|nylon|rubber|latex|silk|satin|velvet|cotton|lace|soft)\b", re.I)
FIXTURE = r"(?:(?:left|right|top|bottom|middle|base|foot|head|end|corners?|legs?|frame)\s+(?:of\s+the\s+)?)*" \
          r"(?:bed\s?posts?|head\s?boards?|foot\s?boards?|bed\s?frames?|bed\s?rails?|beds?|tables?|benches?|desks?|chairs?|" \
          r"stools?|counters?|radiators?|pipes?|poles?|posts?|pillars?|beams?|rafters?|hooks?|rings?|walls?|floors?|ceilings?|" \
          r"railings?|rails?|bars?|trees?|door\s?(?:handles?|knobs?)|doors?|sinks?|toilets?|fences?|gates?|cages?|crosses|" \
          r"(?:step\s?)?ladders?|rungs?|sawhorses?|trestles?|frames?|racks?|stocks|pillory|anchors?|eye\s?bolts?|bolts?)"
WIDE = re.compile(r"\b(?:head\s?boards?|foot\s?boards?|beds?|tables?|desks?|benches?|(?:step\s?)?ladders?|sawhorses?|cross|walls?|"
                  r"counters?|frames?|racks?)$", re.I)
ABOVE = re.compile(r"\b(?:ceilings?|beams?|rafters?|overhead|hooks?\s+(?:above|in\s+the\s+ceiling)|top\s+of)\b", re.I)
BED = re.compile(r"\b(?:head\s?boards?|foot\s?boards?|bed\s?(?:posts?|frames?|rails?)|beds?)\b", re.I)


def align(n, nearest=False):
    n = max(5, int(n))
    lo = n - (n - 5) % 17
    return min(MAX_FRAMES, lo if lo == n or (nearest and n - lo <= lo + 17 - n) else lo + 17)


def pron(person, case=1):
    sex = (person or {}).get("sex")
    return {"f": ("she", "her", "her"), "m": ("he", "his", "him")}.get(sex, ("they", "their", "them"))[case]


def cap(s):
    return s[:1].upper() + s[1:] if s else s


def join(items):
    items = [x for x in items if x]
    return "" if not items else items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def many(word):
    return bool(re.search(r"(?<!s)s$|\b(?:jeans|pants|shorts|tights|panties|pajamas|pyjamas|leggings|trousers)$", word.strip(), re.I))


def has(rx, text):
    return bool(re.search(rf"\b(?:{rx})\b", text or "", re.I))


def zone(g):
    return ({"upper", "lower"} if has(WHOLE, g) or has("lingerie", g) else set()) | ({"lower"} if has(LOWER, g) or (
        has(UNDER, g) and not has("bras?|bralettes?", g)) or has("tights|pantyhose", g) else set()) | (
        {"upper"} if has(UPPER, g) or has("bras?|bralettes?", g) else set())


def covers(g):
    return set() if SHEER.search(g) or has(UNDER, g) else zone(g) - ({"lower"} if has(r"tights|pantyhose", g) else set())


def sentences(text):
    text = text or ""
    flat = re.sub(r"[\"“][^\"”]*[\"”]", lambda m: re.sub(r"\s", "_", m.group(0)), text)
    cuts = [0] + [x for m in re.finditer(r"(?<=[.!?…])[\"”’]?(\s+)(?=[\"“A-Z*(])", flat) for x in m.span(1)] + [len(text)]
    return [s.strip() for s in (text[a:b] for a, b in zip(cuts[::2], cuts[1::2])) if s.strip()]


def strip_pics(text):
    return re.sub(rf"\s*{PIC}", "", text or "", flags=re.I).strip()


def cut(text, n=60):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"
