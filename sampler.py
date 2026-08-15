"""
H3-LongVideos-V1  (all-in-one: one prompt + one length -> long video+audio)
===========================================================================
You give it a prompt (first paragraph = the look/character kept across the
whole video; each later paragraph = a scene beat), a total length in seconds,
and a resolution from the VRAM-appropriate list. It spreads the seconds across
your paragraphs, splits them into shots that fit H3's 15s ceiling and your
VRAM, chains each shot from the previous one's last frame, and returns the
finished video + audio.

Resolution choices are built at node-load from your card's TOTAL VRAM and are
grouped by ratio (16:9 / 9:16 / 4:3 / 3:4 / 1:1). 'custom (may OOM)' lets you
enter width/height yourself and warns in the info output. The runtime
auto-budget (tiled decode -> lower resolution on real OOM) is the actual
no-crash guarantee; the list is a curated shortlist by card size.

Requirements: patch the model with ModelSamplingMiniMaxH3 upstream. H3 is
CFG-free (cfg 1) and needs no negative prompt — the node makes an empty one
internally, so there's no negative input to wire. denoise is fixed at 1.0
internally: a partial denoise starts sampling from a lower sigma on the joint
AV latent and desyncs the audio schedule, so there's deliberately no denoise
input to get wrong.

Verified against ComfyUI core (comfy_extras/nodes_minimax_h3.py, model_base.py,
sd.py).
"""

import gc
import re
import torch

import nodes
import comfy.utils
import comfy.samplers
import comfy.nested_tensor
import comfy.model_management as mm
import node_helpers

AUDIO_LATENT_FPS = 40
GB = 1024 ** 3
H3_MAX_FRAMES = 362
MIN_SHOT_FRAMES = 124          # internal VRAM floor (~5s @24fps)


# --- H3 geometry -----------------------------------------------------------
def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(fc):
    return 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2


def temporal_shape(length, fps=24):
    fc = align_frame_count(max(5, length))
    return fc, video_latent_t(fc), round(fc / fps * AUDIO_LATENT_FPS)


def res_down(w, h, factor=0.85, mult=32):
    snap = lambda v: max(mult, round(v * factor / mult) * mult)
    return snap(w), snap(h)


# --- native 768p canvas per ratio (H3-Base renders at 768 short edge) ------
# H3-Base's native/trained resolution is 768 on the short edge; rendering below
# it softens the whole frame (faces worst). So resolution is ALWAYS kept native
# and never traded down for VRAM -- when the card is tight, SHOT LENGTH shrinks
# instead (see estimate_shot_frames). The 768*1344 area cap means very wide
# ratios (21:9) land just under 768 short edge natively.
NATIVE_RES = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3":  (1024, 768),
    "3:4":  (768, 1024),
    "1:1":  (768, 768),
    "21:9": (1536, 672),
}
# 512-short-edge "fast" tier: ~4x fewer pixels than native, for the generate-low-
# then-upscale (LTX 2.3) workflow. Best for close/medium shots -- H3 distorts faces
# on WIDE shots at any resolution, so keep faces reasonably large in frame.
FAST_RES = {
    "16:9": (896, 512),
    "9:16": (512, 896),
    "4:3":  (704, 512),
    "3:4":  (512, 704),
    "1:1":  (512, 512),
    "21:9": (1184, 512),
}
# 640-short-edge "balanced" tier: a middle ground between fast 512 and native 768.
MID_RES = {
    "16:9": (1152, 640),
    "9:16": (640, 1152),
    "4:3":  (864, 640),
    "3:4":  (640, 864),
    "1:1":  (640, 640),
    "21:9": (1504, 640),
}
NATIVE_PIXELS = 1344 * 768        # ~1MP reference for the VRAM/length budget
# Shot-length budget fit (see estimate_shot_frames). Measured anchors on a 16GB
# card: at 1344x768 with the pruned NVFP4 DiT, 243f fits (~2.7GB spare) and 362f
# overflowed by ~4.3GB -> slope (362-243)/(7.0-2.7) ~= 27.7 frames per GB. The
# baseline absorbs the part of the latent that fits in already-counted space.
# Refit against BOTH measured points at once: 243f must be reachable at the 640p /
# 13.6GB case (~1.1GB scaled spare) and 362f must NOT be until ~7GB. That gives
# slope (362-243)/(7.0-1.12) ~= 20.2 f/GB with a 10.91GB baseline. The native
# NVFP4 case then lands at 260f -- above the 243f measured safe and below the 362f
# measured overflow, i.e. consistent with both rather than fitted to either.
FRAMES_PER_GB = 20.2
# Fraction of free VRAM held back for transient activation peaks during sampling
# (the steady-state latent is not the high-water mark). Prevents the node from
# picking a length that fits on paper but spills into shared memory mid-shot.
SPIKE_RESERVE = 0.12
FRAMES_BASELINE_GB = 10.91


def resolution_options():
    """All-preset, all-multiple-of-32 resolution list -- three short-edge tiers per
    ratio: native 768 (best detail), balanced 640, and fast 512 (generate-then-
    upscale). No custom entry: every option is a valid H3 size, so nothing to snap
    or mis-type. The length budget is resolution-aware, so lower tiers unlock
    longer shots."""
    opts  = [f"{r} - {w}x{h} (native)" for r, (w, h) in NATIVE_RES.items()]
    opts += [f"{r} - {w}x{h} (balanced)" for r, (w, h) in MID_RES.items()]
    opts += [f"{r} - {w}x{h} (fast, upscale later)" for r, (w, h) in FAST_RES.items()]
    return opts


def parse_resolution(choice):
    """Read WxH from a preset label. All presets are valid multiples of 32, so
    there's no custom path to snap. Falls back to 16:9 native if unrecognized."""
    import re
    m = re.search(r"(\d+)\s*x\s*(\d+)", choice or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    return NATIVE_RES["16:9"]


# --- prompt parsing + auto time distribution -------------------------------
def split_paragraphs(text, delimiter):
    raw = text.replace("\r\n", "\n").strip()
    if not raw:
        return []
    import re
    raw = re.sub(r"(?m)^\s*" + re.escape(delimiter) + r"\s*$", "\n\n", raw)
    return [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]


def _split_items(s):
    """Split a description into attribute items on commas AND sentence ends.

    A sheet written naturally ends clauses with a period -- "wearing a black t-shirt
    and jeans. Mouth closed." -- and treating that as ONE item drags a whole sentence
    into the inline parenthetical, which then reads as its own statement about a
    person rather than an attribute of the pronoun. Splitting on '.' as well keeps
    each item a short attribute."""
    import re
    return [i.strip(" .;") for i in re.split(r"[,.;]", s or "") if i.strip(" .;")]


def _norm_name(name):
    """Normalize a person key: trim whitespace and a trailing ':' so 'Kristy:'
    and 'Kristy' are the same person (the colon is natural to type because it's
    how the sheet renders back)."""
    return name.strip().rstrip(":").strip()


def _split_name(part):
    """Split 'Name = items' or 'Name: items' into (name, items_str). The name is
    bound by '=' or a leading 'Name:' (the token before ':' must have no comma,
    so a plain clothing list like 'grey shorts, red jacket' stays unnamed).
    Returns ('', part) when there's no name binder."""
    if "=" in part:
        name, desc = part.split("=", 1)
        return _norm_name(name), desc
    if ":" in part:
        head, tail = part.split(":", 1)
        if "," not in head:                      # a name won't contain a comma
            return head.strip(), tail
    return "", part


def _entries(text):
    """Split a wardrobe sheet into per-person entries. Accepts BOTH ';' and
    NEWLINES as separators -- character_memory is a multiline box, so one person
    per line is the natural way to write it, and silently mis-parsing that (folding
    the next person into the previous one's item list) breaks name lookup and makes
    removals fail to match. Also tolerates a leading '-' bullet per line."""
    import re
    parts = []
    for chunk in re.split(r"[;\n\r]+", text or ""):
        chunk = chunk.strip().lstrip("-*\u2022 ").strip()
        if chunk:
            parts.append(chunk)
    return parts


def parse_wardrobe(text):
    """Parse an INITIAL wardrobe sheet into an ordered {name: [items]} dict, so
    people are tracked independently and individual garments can be added or
    removed later. Entries split on ';' OR newlines; each is 'Name = a, b' OR
    'Name: a, b' (colon works too). An entry with no name binder is the single
    unnamed subject under '' (one-person, backward-compat)."""
    out = {}
    for part in _entries(text):
        name, desc = _split_name(part)
        out[name] = _split_items(desc)
    return out


def apply_wardrobe_change(active, text):
    """Apply a per-beat 'wardrobe:' directive so you DON'T restate the whole
    outfit to change one thing. Entries split on ';'; each targets one person
    (or the unnamed subject) with an operator:
        Name = a, b     replace that person's whole outfit
        Name += c, d    ADD items
        Name -= jacket  REMOVE items whose text contains any given token
    The Name may be written with or without a trailing colon ('Maya' or 'Maya:'
    both work). Bare forms (no Name) target the single unnamed subject: '= a,b',
    '+= hat', '-= jacket'. Names not mentioned are left untouched. So dropping a
    jacket is just 'wardrobe: Maya -= jacket' -- one token, nothing re-typed."""
    active = {k: list(v) for k, v in active.items()}
    for part in _entries(text):
        if "+=" in part:
            name, val, op = (*part.split("+=", 1), "+")
        elif "-=" in part:
            name, val, op = (*part.split("-=", 1), "-")
        elif "=" in part:
            name, val, op = (*part.split("=", 1), "=")
        else:
            name, val = _split_name(part); op = "="   # bare or 'Name: items' -> replace
        name = _norm_name(name)
        items = _split_items(val)
        cur = active.get(name, [])
        if op == "=":
            active[name] = items
        elif op == "+":
            active[name] = cur + [i for i in items if i.lower() not in (c.lower() for c in cur)]
        elif op == "-":
            toks = [t.lower() for t in items]
            active[name] = [it for it in cur if not any(t in it.lower() for t in toks)]
    return active


_PRO = {"she": "she", "her": "she", "hers": "she",
        "he": "he", "him": "he", "his": "he",
        "they": "they", "them": "they", "their": "they", "theirs": "they"}
_GENDER = {"woman": "she", "women": "she", "female": "she", "girl": "she", "lady": "she",
           "man": "he", "male": "he", "boy": "he", "guy": "he", "gentleman": "he"}


def _pronoun_of(items):
    """A person's pronoun, from an explicit token in their sheet ('she') or a
    gender word in their description ('woman'). None if undeclared/undetectable."""
    import re
    for it in items:
        if it.strip().lower() in _PRO:
            return _PRO[it.strip().lower()]
    for it in items:
        for w in re.findall(r"[a-z]+", it.lower()):
            if w in _GENDER:
                return _GENDER[w]
    return None


_PERSON_NOUNS = (r"woman|women|man|men|girl|boy|guy|lady|gentleman|person|people|"
                 r"female|male|figure|character|adult|teen|teenager")


def _deposition(desc, name=None):
    """Turn a description into pure ATTRIBUTES, removing any subject-introducing
    noun phrase.

    A description written naturally -- "a woman with silver hair", "a young woman",
    "Kristy is a tall woman" -- renders inline as `She (a woman with silver hair)`.
    That is TWO subject nouns in one clause ("She" and "a woman"), which
    text-to-video reads as two people: character duplication, visible from the very
    first shot and independent of resolution. Attributes alone -- "silver hair" --
    bind to the pronoun instead of competing with it.

    Strips: a leading article + optional adjectives + person noun (keeping any
    following "with/in ..." attributes), a copula phrase ("Kristy is a tall
    woman"), and a bare repeat of the character's own name."""
    import re
    d = (desc or "").strip()
    if not d:
        return d
    if name:
        d = re.sub(r"\b" + re.escape(name) + r"\b\s*(?:is|,)?\s*", "", d, flags=re.I)
    # "a young woman with silver hair" -> "silver hair"; "a tall woman" -> "tall"
    m = re.match(r"^\s*(?:an?|the)\s+((?:[\w\-]+\s+){0,3}?)(?:" + _PERSON_NOUNS + r")\b"
                 r"(?:\s+(?:with|in|wearing)\s+)?(.*)$", d, re.I)
    if m:
        adjectives, rest = m.group(1).strip(), m.group(2).strip()
        d = (rest if rest else adjectives) or adjectives
    # "wearing a black t-shirt" -> "black t-shirt": inside a parenthetical the verb
    # reads as a separate predicate about a subject, not an attribute of the pronoun
    d = re.sub(r"^\s*(?:wearing|dressed in|dressed|clad in|in)\s+", "", d, flags=re.I)
    d = re.sub(r"^\s*(?:an?|the)\s+", "", d)
    # a bare person noun left on its own carries no attribute -> drop it
    if re.fullmatch(r"\s*(?:an?|the)?\s*(?:" + _PERSON_NOUNS + r")\s*", d, re.I):
        return ""
    return re.sub(r"\s{2,}", " ", d).strip(" ,")


# Mouth/lip state items in a character sheet ("mouth closed", "lips together").
# Users add these to force mouths shut on action shots, which works -- but they are
# re-stamped into EVERY shot, so on a beat with real quoted dialogue the prompt
# tells the model to keep the mouth closed AND to speak. Dropped on speaking shots
# only; kept everywhere else so the forced-closed behaviour is preserved.


def _is_mouth_state(item):
    it = (item or "").strip().lower()
    if not it:
        return False
    return bool(re.search(r"\b(?:mouth|lips|jaw)\b", it) and
                re.search(r"\b(?:closed|shut|together|still|sealed|not\s+talking|no\s+talking)\b", it))


def _clean_items(items, name=None, drop_mouth_state=False):
    """Drop bare pronoun tokens, de-position any noun-phrase descriptions so a
    parenthetical never introduces a second subject, and -- on shots that contain
    real quoted dialogue -- drop mouth-state items so the sheet does not order a
    closed mouth in the same breath as a spoken line."""
    out = []
    for it in items:
        if it.strip().lower() in _PRO:
            continue
        if drop_mouth_state and _is_mouth_state(it):
            continue
        d = _deposition(it, name)
        if d:
            out.append(d)
    return out



def _pron_map(active):
    """{pronoun: [names]} for resolving a bare 'she'/'he' to a person."""
    out = {}
    for n, items in active.items():
        if not n:
            continue
        p = _pronoun_of(items)
        if p:
            out.setdefault(p, []).append(n)
    return out


def _resolve_subject(word, names, pron_map, single):
    """Map a subject token (a name or a pronoun) to a tracked person, or None if
    ambiguous. In a one-person scene any pronoun maps to that person."""
    wl = word.lower()
    for n in names:
        if n.lower() == wl:
            return n
    if wl in _PRO:
        want = _PRO[wl]
        cands = pron_map.get(want, [])
        if len(cands) == 1:
            return cands[0]
        if cands:
            return None                      # ambiguous: two people share this pronoun
        # No candidate with this pronoun. Only fall back to the lone remaining
        # person if their pronoun is UNDECLARED -- never map 'he' onto a declared
        # 'she' (which happened once the 'he' character had left the scene).
        if single and names and not any(names[0] in v for v in pron_map.values()):
            return names[0]
        return None
    return None


def _item_mentioned(item, window):
    """Does `window` refer to this wardrobe item? Matches the whole phrase, or the
    item's head noun, tolerant of singular/plural ('boots' vs 'boot') -- the strict
    exact-substring test missed 'takes off her boots' when the sheet said 'boots'
    and vice versa. Ignores generic colour/size adjectives so 'red jacket' is still
    matched by 'her jacket'."""
    import re
    w = window.lower()
    il = item.lower().strip()
    if not il:
        return False
    if il in w:
        return True
    words = [x for x in re.findall(r"[a-z\-]+", il)]
    if not words:
        return False
    head = words[-1]
    for form in {head, head.rstrip("s"), head + "s", head + "es"}:
        if form and re.search(r"\b" + re.escape(form) + r"\b", w):
            return True
    return False


def auto_wardrobe_removals(active, body):
    """Infer clothing REMOVALS from a beat's own action text, so you don't have
    to write a 'wardrobe:' line at all -- "she takes off her jacket" drops the
    jacket by itself.

    SAFE BY DESIGN: a removal only fires on an item the character is ALREADY
    wearing. Non-garment objects match nothing, so "the plane takes off down the
    runway" removes nothing. The subject can be a NAME or a PRONOUN: with two
    people, declare a pronoun per person ('Maya = she, ...; Jon = he, ...') and
    'she takes off her jacket' attributes to Maya. In a one-person scene any
    pronoun maps to that person. If the subject is ambiguous (two same-pronoun
    people, no name), the item is dropped from whoever wears it. Explicit
    'wardrobe: -=' always overrides."""
    import re
    if not body:
        return active
    text = " " + body.lower() + " "
    active = {k: list(v) for k, v in active.items()}
    names = [n for n in active if n]
    pron_map = _pron_map(active)
    single = len(names) == 1

    remove_cue = re.compile(
        # verb ... off / out of / aside / away   (covers "takes off", "slips out of")
        r"\b(?:takes?|took|taken|taking|pulls?|pulled|peels?|peeled|strips?|stripped|"
        r"slips?|slipped|shrugs?|shrugged|tears?|tore|yanks?|yanked|casts?|kicks?|"
        r"throws?|threw|tosses|tossed|hangs?|hung|drops?|dropped|sets?|set|puts?|put)\b"
        r"[\w\s\']{0,20}?\b(?:off|out of|aside|away|down)\b"
        # standalone removal verbs
        r"|\b(?:removes?|removed|removing|sheds?|shed|shedding|discards?|discarded|"
        r"ditch(?:es|ed)?|doffs?|doffed|unbuttons?|unzips?|unzipped|unbuckles?|"
        r"hangs? up|hung up)\b"
        # "<garment> is off / are off"
        r"|\bis off\b|\bare off\b"
        # put-away phrasings: "hangs her jacket on a hook", "drapes it over a chair"
        r"|\b(?:hangs?|hung|drapes?|draped|slings?|slung|drops?|dropped|tosses|tossed|"
        r"throws?|threw|leaves?|left|sets?|set|lays?|laid|places?|placed)\b[\w\s\']{0,20}?"
        r"\b(?:on|over|across)\s+(?:a|an|the)\b")

    subj_tokens = [re.escape(n.lower()) for n in names] + list(_PRO.keys())
    subj_re = re.compile(r"\b(" + "|".join(subj_tokens) + r")\b") if subj_tokens else None

    def nearest_subject(pos):
        best, bp = None, -1
        if subj_re:
            for mm in subj_re.finditer(text):
                if 0 <= mm.start() < pos and mm.start() > bp:
                    person = _resolve_subject(mm.group(1), names, pron_map, single)
                    if person is not None:
                        bp, best = mm.start(), person
        return best

    # Donning phrases must never trigger a removal ("pulls on a jacket", "puts on
    # her boots", "slips into her coat") -- the wrong direction is far worse than a
    # miss, since it would strip clothing the character just put ON.
    don_re = re.compile(r"\b(?:pulls?|puts?|slips?|throws?|shrugs?|zips?|buttons?|laces?|"
                        r"pulled|put|slipped|threw|shrugged)\b\s+(?:on|into)\b")
    don_spans = [(d.start(), d.end() + 45) for d in don_re.finditer(text)]

    for m in remove_cue.finditer(text):
        if any(a <= m.start() <= b for a, b in don_spans):
            continue
        # look forward from the verb, and a little BACKWARD too: some phrasings put
        # the garment first ("her jacket is off now", "the jacket, now removed").
        window = text[max(0, m.start() - 28):m.end() + 40]
        tgt = nearest_subject(m.start())
        for name in ([tgt] if tgt else list(active.keys())):
            for it in list(active.get(name, [])):
                if it.strip().lower() in _PRO:      # never treat the pronoun token as a garment
                    continue
                if _item_mentioned(it, window):
                    active[name] = [x for x in active[name] if x != it]
    return active




def _scrub_removed(text, removed):
    """Delete phrases for removed garments from PERSISTENT text (the anchor), so a
    garment written into the anchor prose -- e.g. 'A woman in a red flight jacket'
    -- can't re-apply itself on every shot after the character takes it off. Removes
    the item phrase plus a leading connector ('in a', 'wearing a', 'with a') and
    tidies the leftover punctuation. Case-insensitive; leaves everything else alone."""
    import re
    if not text or not removed:
        return text
    changed = False
    for item in sorted(removed, key=len, reverse=True):
        item = item.strip()
        if not item:
            continue
        pat = (r"(?:,\s*)?\b(?:wearing|dressed in|in|with)?\s*(?:a|an|the|her|his|their)?\s*"
               + re.escape(item) + r"\b")
        new_text = re.sub(pat, "", text, flags=re.I)
        if new_text != text:
            changed = True
            text = new_text
    if not changed:
        # Nothing was actually scrubbed from this text, so leave it EXACTLY as the
        # user wrote it. The tidy-up below repairs punctuation left by a removal;
        # running it unconditionally silently rewrote untouched prose (e.g.
        # "hangar and airfield" -> "hangar, airfield") on every shot after any
        # unrelated garment removal.
        return text.strip()
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    text = re.sub(r",\s*\.", ".", text)
    # tidy connectors left dangling by a removed phrase: "silver hair and , in a
    # hangar" / "silver hair and hangar" -> "silver hair, in a hangar"
    text = re.sub(r"\s+and\s*,", ",", text)
    text = re.sub(r"\s+and\s+(in|at|on|with|under|beside)\b", r", \1", text)
    text = re.sub(r"\s+and\s+(?=[a-z]+\s*,)", ", ", text)
    text = re.sub(r"\s+and\s*$", "", text)
    text = re.sub(r"^\s*(?:and|,)\s+", "", text)          # leading dangling connector
    text = re.sub(r"^\s*and\b", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    return text.strip(" ,")


def _strip_people_from_anchor(anchor_id, active):
    """Remove tracked people from the ANCHOR prose.

    The anchor is stamped into every shot, so if it also introduces a character --
    by name ("Kristy stands by the plane") or by description ("a woman with silver
    hair in a red jacket") -- that character is asserted TWICE per shot: once by the
    anchor and once by the beat's own inline binding. Text-to-video reads two
    introductions of one subject as two subjects, which is the character-duplication
    users see. The anchor should carry scene and style only; who is present is
    decided per beat.

    Removes (a) any tracked NAME plus its clause, and (b) a person-phrase whose
    description overlaps a tracked person's items (gender word + shared descriptors).
    Leaves everything else -- setting, lighting, lens, mood -- untouched."""
    import re
    if not anchor_id:
        return anchor_id
    txt = anchor_id
    for name in [n for n in active if n]:
        # drop a whole sentence that names this person, else just the name token
        sentences = re.split(r'(?<=[.!?])\s+', txt)
        kept = [c for c in sentences if not re.search(r"\b" + re.escape(name) + r"\b", c, re.I)]
        if len(kept) != len(sentences):
            txt = " ".join(kept)
        else:
            txt = re.sub(r"\b" + re.escape(name) + r"\b", "", txt, flags=re.I)
    # description overlap: "a woman with silver hair in a red jacket"
    for name, items in active.items():
        descs = [i for i in _clean_items(items, name) if len(i.split()) <= 4]
        if not descs:
            continue
        pron = _pronoun_of(items)
        nouns = {"she": r"(?:woman|women|girl|lady)", "he": r"(?:man|men|boy|guy)"}.get(pron, r"(?:person|figure)")
        # match the person phrase up to a sentence end / conjunction, so trailing
        # "with X in a Y" clauses go with it instead of leaving fragments behind
        pat = re.compile(r"(?:,\s*)?\b(?:a|an|the)\s+(?:[\w\-]+\s+){0,3}" + nouns +
                         r"(?:(?!\.|\band\b).)*", re.I)
        for mm_ in list(pat.finditer(txt)):
            phrase = mm_.group(0)
            if sum(1 for d in descs if d.lower() in phrase.lower()) >= 1:
                txt = txt.replace(phrase, "")
    # Generic, UNNAMED person references in the anchor ("the camera follows the
    # subject", "moves toward the person", "tracks the figure") are stamped into
    # every shot alongside the named cast, so the model renders an extra body that
    # matches nobody -- the phantom third person. Camera-direction wording is the
    # usual way these creep in. Rewrite them to refer to the framing, not a body.
    txt = re.sub(r"\b(?:the|a|an)\s+(?:main\s+|central\s+)?"
                 r"(?:subject|person|figure|character|model|individual|protagonist)\b",
                 "the scene", txt, flags=re.I)
    txt = re.sub(r"\b(?:the|a|an)\s+(?:subjects|people|figures|characters)\b",
                 "the scene", txt, flags=re.I)
    txt = re.sub(r"\s{2,}", " ", txt)
    txt = re.sub(r"\s+([,.])", r"\1", txt)
    txt = re.sub(r"(,\s*){2,}", ", ", txt)
    txt = re.sub(r"\.{2,}", ".", txt)          # "Warm light.." -> "Warm light."
    txt = re.sub(r"^[\s.,]+", "", txt)          # leading ". " left by a removed clause
    txt = re.sub(r"^\s*(?:and|,)\s+", "", txt)
    txt = re.sub(r"\s+and\s*$", "", txt)
    txt = re.sub(r"\s+\.", ".", txt)
    return txt.strip(" ,.").strip() + ("." if txt.strip(" ,") else "")


def compose_persistent(body, active, anchor_id, removed=None, departed=None,
                       count_subjects=False, speaking=False, front_load=False):
    """Assemble one shot's text WITHOUT duplicating subjects.

    Each present person's description is injected as a parenthetical at the FIRST
    reference to them in the beat -- whether that reference is their NAME or a
    resolvable PRONOUN ('she'/'he'). So 'she takes off her jacket' becomes 'she
    (silver hair, grey shorts) takes off her jacket': described once, no name, no
    duplicate subject. A person not referenced at all (by name or pronoun) is
    omitted from that shot. The unnamed single subject is prepended as before.

    Pronoun tokens declared in a person's sheet ('Maya = she, ...') are used to
    resolve 'she'/'he' but are stripped from the shown description. Keep the
    anchor to scene/style with NO names."""
    import re
    count_prefix = ""          # set when the count clause is front-loaded (LoRA runs)
    departed = set(departed or ())
    # A character who has LEFT the scene is never described again -- not even if a
    # later pronoun could resolve to them. This is what stops an exited character
    # being silently re-summoned into a later shot.
    active = {k: v for k, v in active.items() if k not in departed}
    named = [n for n in active if n and active[n]]
    unnamed = active.get("", [])
    anchor_id = _scrub_removed(anchor_id, removed)
    # Keep tracked people OUT of the always-on anchor: the beat binds them inline,
    # so leaving them here too introduces each character twice per shot.
    anchor_id = _strip_people_from_anchor(anchor_id, active)
    prefix_bits = [x for x in (anchor_id, ", ".join(unnamed) if unnamed else "") if x]

    if named:
        names = list(named)
        pron_map = _pron_map(active)
        single = len(names) == 1
        low = body.lower()

        # first reference position for each present person (name first, else pronoun)
        refs = {}
        for n in names:
            m = re.search(r"\b" + re.escape(n.lower()) + r"\b", low)
            if m:
                refs[n] = m.end()
        for m in re.finditer(r"\b(she|he|they|her|him|them|his|their)\b", low):
            person = _resolve_subject(m.group(1), names, pron_map, single)
            if person and person not in refs:
                refs[person] = m.end()

        if refs:
            # inject from rightmost position first so earlier indices stay valid
            for n in sorted(refs, key=lambda k: refs[k], reverse=True):
                desc = ", ".join(_clean_items(active[n], n, drop_mouth_state=speaking))
                if desc:
                    pos = refs[n]
                    body = body[:pos] + f" ({desc})" + body[pos:]
            # An EXPLICIT SUBJECT COUNT is the strongest prompt-side defence against
            # the model rendering a character twice. Duplication gets much more
            # likely below the model's native resolution: fewer pixels per subject
            # pushes the sample away from the training distribution and the figure
            # gets tiled. Stating the count (and "no other people") gives the model
            # a hard target instead of leaving the number implicit.
            if count_subjects:
                n_people = len(refs)
                word = {1: "one", 2: "two", 3: "three", 4: "four",
                        5: "five", 6: "six"}.get(n_people, str(n_people))
                noun = "person" if n_people == 1 else "people"
                clause = (f"Exactly {word} {noun} in this shot, no duplicates, "
                          f"no other people in frame, no extra bodies, "
                          f"no repeated figures, no crowd. ")
                if front_load:
                    # A distilled LoRA settles composition in its first step or two,
                    # so the count must be the FIRST thing in the prompt -- ahead of
                    # scene and style -- not buried after the anchor.
                    count_prefix = clause
                    clause = ""
                else:
                    count_prefix = ""
                body = clause + body
        # If NOBODY is referenced by name or pronoun, this is a scenery/cutaway beat
        # ("the hangar doors roll open"). Emit no people at all: the old grouped
        # 'Kristy: ... Jon: ...' prefix both re-introduced names (the duplication
        # pattern) and forced absent characters into shots they don't belong in.

    prefix = " ".join(prefix_bits)
    out = (prefix + " " + body).strip() if prefix else body.strip()
    return (count_prefix + out).strip()


def extract_wardrobe(body):
    """Pull a 'wardrobe: ...' directive line out of a beat body. Returns
    (clean_body, wardrobe_or_None). The directive is a whole line starting with
    'wardrobe:' (case-insensitive), placed INSIDE a beat (not as its own blank-
    line-separated paragraph, which would become its own shot). It's removed
    from the body so the literal 'wardrobe:' text isn't stamped as an action."""
    import re
    kept, wardrobe = [], None
    for ln in body.split("\n"):
        if re.match(r"\s*wardrobe\s*:", ln, re.I):
            wardrobe = ln.split(":", 1)[1].strip()
        else:
            kept.append(ln)
    return "\n".join(kept).strip(), wardrobe


def has_speech(body):
    """True only if a beat contains ACTUAL scripted speech -- double-quoted words
    or an explicit <d>...</d> tag. Bare speech VERBS ('calls out', 'tells', 'says'
    with no quoted line) deliberately do NOT count: unscripted speech is exactly
    what H3 fills with gibberish, so those beats get silenced too. If you want
    someone to speak, quote the line: She says, "Ready for departure."
    Apostrophes/single quotes never count (they'd false-fire on "she's")."""
    import re
    if not body:
        return False
    if re.search(r"<d>.*?</d>", body, re.S):
        return True
    if re.search(r'["\u201c\u201d].+?["\u201c\u201d]', body):     # double/curly quotes only
        return True
    return False


# Leading, physically-described directive. A trailing "no dialogue" sentence is the
# weakest position in a prompt; H3 follows described PHYSICAL STATE far better than
# an appended negation -- so the silence is stated as a mouth state, up front, and
# repeated once at the end as a hard constraint.
LIPS_CLOSED_LEAD = ("Everyone in this shot is silent with their mouth closed and lips together, "
                    "jaw still, not talking. ")
LIPS_CLOSED_TAIL = " No speech, no dialogue, no lip movement, no mouth movement."


def removed_phrase_items(body, anchor_id):
    """Garments named in a REMOVAL phrase in this beat that also appear in the
    anchor prose. Covers the case where the item was never in the wardrobe channel
    at all -- e.g. the anchor says 'a woman in a red flight jacket' and the beat
    says 'she takes off her jacket'. Without this the anchor would keep re-applying
    it forever. Returns the anchor phrases to scrub."""
    import re
    if not body or not anchor_id:
        return []
    verb = re.compile(r"\b(takes?|took|taking|pulls?|pulled|peels?|peeled|strips?|stripped|"
                      r"slips?|slipped|shrugs?|shrugged|removes?|removed|sheds?|shed|discards?|"
                      r"ditch(?:es|ed)?|doffs?|unbuttons?|unzips?)\b", re.I)
    stop = {"off", "out", "of", "aside", "away", "and", "the", "her", "his", "their", "then",
            "over", "onto", "with", "from", "a", "an", "it", "she", "he", "they", "up", "down"}
    out = []
    for m in verb.finditer(body):
        tail = body[m.end():m.end() + 50]
        for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", tail):
            if w.lower() in stop:
                continue
            # take the noun's surrounding phrase out of the anchor (adjectives included)
            am = re.search(r"((?:[A-Za-z\-]+\s+){0,3}" + re.escape(w) + r")\b", anchor_id, re.I)
            if am:
                out.append(am.group(1).strip())
            break
    return out


def extract_directive(body, key):
    """Pull a '<key>: ...' line out of a beat body. Returns (clean_body, value|None)."""
    import re
    kept, val = [], None
    for ln in body.split("\n"):
        if re.match(r"\s*" + key + r"\s*:", ln, re.I):
            val = ln.split(":", 1)[1].strip()
        else:
            kept.append(ln)
    return "\n".join(kept).strip(), val


def detect_exits(body, active, departed):
    """Names of characters who LEAVE in this beat, so they don't reappear later.
    Matches an exit phrase ('leaves', 'walks out', 'exits', 'drives off', 'steps
    out of frame', 'is gone') attributed to the nearest preceding subject (name or
    resolvable pronoun). Gated on tracked people, so 'the plane leaves' -- not a
    tracked person -- departs nobody."""
    import re
    if not body:
        return []
    text = " " + body.lower() + " "
    names = [n for n in active if n and n not in departed]
    if not names:
        return []
    pron_map = _pron_map({k: v for k, v in active.items() if k not in departed})
    single = len(names) == 1

    exit_cue = re.compile(
        r"\b(?:leaves?|left|leaving|exits?|exited|departs?|departed|"
        r"walks? (?:out|off|away)|walked (?:out|off|away)|steps? (?:out|off|away)|"
        r"stepped (?:out|off|away)|drives? (?:off|away)|drove (?:off|away)|"
        r"rides? (?:off|away)|runs? (?:out|off)|ran (?:out|off)|"
        r"disappears?|vanishes?|is gone|are gone|out of frame|off screen|off-screen)\b")

    subj_tokens = [re.escape(n.lower()) for n in names] + list(_PRO.keys())
    subj_re = re.compile(r"\b(" + "|".join(subj_tokens) + r")\b")

    out = []
    for m in exit_cue.finditer(text):
        best, bp = None, -1
        for sm in subj_re.finditer(text):
            if 0 <= sm.start() < m.start() and sm.start() > bp:
                person = _resolve_subject(sm.group(1), names, pron_map, single)
                if person is not None:
                    bp, best = sm.start(), person
        if best:
            out.append(best)
    return out


def departed_phrase_people(body, anchor_id):
    """Anchor phrases for people who LEAVE in this beat but were never declared in
    the character channel -- e.g. the anchor says 'a woman with silver hair and a
    bald man in navy overalls' and the beat says 'he walks out'. Without this the
    anchor keeps re-asserting them into every later shot.

    Resolves the departing subject from the pronoun/noun before the exit cue, then
    finds the matching person-phrase in the anchor by gender word ('man'/'woman'/
    'boy'/'girl'/etc.) and returns that whole phrase (with its trailing
    prepositional clause, e.g. 'a bald man in navy overalls') for scrubbing.
    Returns [] when nothing matches, so non-person exits ('the plane leaves')
    remove nobody."""
    import re
    if not body or not anchor_id:
        return []
    exit_cue = re.compile(
        r"\b(?:leaves?|left|leaving|exits?|exited|departs?|departed|"
        r"walks? (?:out|off|away)|walked (?:out|off|away)|steps? (?:out|off|away)|"
        r"stepped (?:out|off|away)|drives? (?:off|away)|drove (?:off|away)|"
        r"rides? (?:off|away)|runs? (?:out|off)|ran (?:out|off)|"
        r"disappears?|vanishes?|is gone|are gone|out of frame|off screen|off-screen)\b", re.I)
    want = {"she": ("woman", "women", "girl", "lady", "female"),
            "he":  ("man", "men", "boy", "guy", "gentleman", "male")}
    out = []
    for m in exit_cue.finditer(body):
        head = body[:m.start()]
        pm = None
        for p in re.finditer(r"\b(she|he|her|him|his|the\s+\w+)\b", head, re.I):
            pm = p.group(1).lower()
        if not pm:
            continue
        key = _PRO.get(pm.split()[-1])
        nouns = want.get(key, ())
        if not nouns:
            continue
        for noun in nouns:
            # the person phrase: optional article/adjectives + noun + an immediate
            # clothing clause only ('a bald man in navy overalls'). The clause must
            # not run past a comma, so a following scene phrase ('..., in a hangar')
            # is left intact.
            am = re.search(r"((?:a|an|the)\s+(?:[\w\-]+\s+){0,3}" + noun +
                           r"(?:\s+(?:in|with|wearing)\s+(?:a\s+|an\s+|the\s+)?"
                           r"(?:[\w\-]+\s+){0,2}[\w\-]+)?)(?=\s*(?:,|\.|$|\band\b))",
                           anchor_id, re.I)
            if am:
                out.append(am.group(1).strip().rstrip(","))
                break
    return out


# Natural speech runs ~2.3-2.8 words/sec in film dialogue; 2.5 is a safe middle.
# Used only to WARN that a line looks too long for the shot it sits in.
WORDS_PER_SEC = 2.5


def dialogue_fit_warnings(beats, seconds_per_shot):
    """Flag beats whose quoted dialogue is unlikely to fit the shot length.

    The VRAM budget shortens SHOTS (never resolution, since rendering below native
    softens the frame). That is the right trade for picture quality, but it is blind
    to dialogue: a line written for a 10s shot gets cut off mid-sentence in a 7s one.
    Audio cannot span the handoff either -- each shot generates its own -- so a
    truncated line is simply lost, not continued.

    Returns a list like ["shot 3: ~6.4s of dialogue in a 5.2s shot"] so the user can
    shorten the line, or choose a lower resolution tier to buy the duration back."""
    import re
    out = []
    for i, b in enumerate(beats or [], 1):
        body, _ = extract_wardrobe((b or "").strip())
        words = 0
        for q in re.findall(r'["\u201c]([^"\u201d]+)["\u201d]', body):
            words += len(q.split())
        if not words:
            continue
        need = words / WORDS_PER_SEC
        if need > seconds_per_shot * 0.92:      # leave a little room to breathe
            out.append(f"shot {i}: ~{need:.1f}s of dialogue in a {seconds_per_shot:.1f}s shot")
    return out


def speech_flags(beats):
    """Per-beat: does it contain scripted (quoted) dialogue? Same rule the prompt
    builder uses to decide silencing, exposed so the renderer can also MUTE the
    audio of non-speech shots -- a deterministic fix when H3 vocalizes anyway."""
    out = []
    for b in (beats if beats else [""]):
        body, _ = extract_wardrobe((b or "").strip())
        out.append(has_speech(body))
    return out


def distribute_generations(anchor, beats, gs, music="", char_memory="", auto_wardrobe=True,
                           auto_silence_nonspeech=True, count_subjects=False, front_load=False):
    """One beat = one shot. Stamp the permanent identity into each beat. Total
    video length is (number of shots) x (per-shot length), computed by the
    caller -- never divided out of a total, so beat count always equals shot count.

    WARDROBE LIVES IN ONE MUTABLE, PER-PERSON CHANNEL so it can be changed or
    removed, and so multiple people are tracked independently. The channel is
    seeded from character_memory, or from a 'wardrobe:' line in the anchor;
    whatever anchor prose REMAINS after pulling that line is permanent identity,
    stamped every shot. Clothing must NEVER be baked into the permanent anchor
    prose (the anchor is immutable and would re-assert a garment you tried to
    remove) -- keep identity in the prose, all clothing in this channel.

    auto_wardrobe (default on): removals are inferred from each beat's own action
    text, so "she takes off her jacket" drops the jacket with no directive. It's
    gated on tracked items, so non-garment objects ("the plane takes off") never
    fire. Additions/swaps still use an explicit 'wardrobe: += ...' line, which
    also overrides the auto-detection.

    Multi-person syntax: 'wardrobe: Maya = grey shorts, red jacket; Jon = navy
    overalls' (a colon works too). A per-beat 'wardrobe:' line updates only the
    names it mentions; one unnamed subject works as before.

    The two audio sections are appended after the visual timeline, in H3's
    documented field order:
      * `overall_soundscape:`  -- ambient/environmental sound (rain, room tone).
      * `non_diegetic_music:`  -- background score not part of the scene.
    Both are global (stamped on every shot). Dialogue and diegetic sound belong
    in the beat body / timeline, NOT in either of these."""
    beats = beats if beats else [""]
    anchor_id, anchor_wardrobe = extract_wardrobe((anchor or "").strip())
    seed = (char_memory or "").strip() or (anchor_wardrobe or "")
    active = parse_wardrobe(seed)            # {name: [items]}, mutable, per-person
    removed = []                             # garments taken off -> also scrubbed from the anchor
    departed = set()                         # characters who left the scene -> never reappear
    blocks = []
    for gi, b in enumerate(beats, 1):
        body, wardrobe_change = extract_wardrobe((b or "").strip())
        body, exit_directive = extract_directive(body, "exit")     # explicit 'exit: Jon'
        body, enter_directive = extract_directive(body, "enter")   # explicit 'enter: Jon' (undo)
        if enter_directive:
            for nm in _entries(enter_directive):
                departed.discard(_norm_name(nm))
        if wardrobe_change is not None:
            before = {k: list(v) for k, v in active.items()}
            active = apply_wardrobe_change(active, wardrobe_change)   # explicit: takes effect THIS shot
            for k, v in before.items():
                removed += [it for it in v if it not in active.get(k, [])]
        body = body or "continue the action, same subject"
        persistent = compose_persistent(body, active, anchor_id, removed, departed, count_subjects,
                                        speaking=has_speech(body), front_load=front_load)
        # Silence non-speech shots: a shot with no scripted dialogue gets an explicit
        # lips-closed / no-speech clause, so H3 doesn't animate a mouth or fill it with
        # gibberish before (or between) actual dialogue. Shots WITH quoted dialogue are
        # left alone so the speech renders.
        if auto_silence_nonspeech and not has_speech(body):
            persistent = LIPS_CLOSED_LEAD + persistent.rstrip(". ") + "." + LIPS_CLOSED_TAIL
        block = f"[Generation {gi}] {persistent}".strip()
        if gs and "soundscape:" not in block.lower():
            block += f"\noverall_soundscape: {gs}"
        # Music is OPT-IN: a blank field emits the spec's silence token N/A on every
        # shot, so H3 doesn't improvise a score. (Soundscape is NOT forced to N/A --
        # per the spec it takes N/A only when total silence is explicitly wanted, so a
        # blank soundscape still lets H3 provide ambient sound.)
        if "non_diegetic_music:" not in block.lower():
            block += f"\nnon_diegetic_music: {music if music else 'N/A'}"
        blocks.append(block.strip())
        # Auto-removal is DEFERRED to the next shot: the garment stays worn in the
        # shot that SHOWS it coming off ("she takes off her jacket"), and is gone
        # from the following shot onward -- which reads correctly. Gated on tracked
        # items, so "the plane takes off" removes nothing.
        # Exits are DEFERRED like removals: the character is still in the shot that
        # SHOWS them leaving, and absent from every shot after.
        if exit_directive:
            for nm in _entries(exit_directive):
                departed.add(_norm_name(nm))
        departed.update(detect_exits(body, active, departed))
        # Plain-text case: a person described only in the anchor prose (never in the
        # character channel) can't be "departed" by name -- scrub their phrase from
        # the anchor instead, exactly as removed garments are scrubbed.
        removed += departed_phrase_people(body, anchor_id)
        if auto_wardrobe:
            before = {k: list(v) for k, v in active.items()}
            active = auto_wardrobe_removals(active, body)
            for k, v in before.items():
                removed += [it for it in v if it not in active.get(k, [])]
            # Also catch a garment that lives ONLY in the anchor prose (never in the
            # wardrobe channel): the removal phrase names it, so scrub it from the
            # anchor from the next shot on, or the anchor re-applies it forever.
            removed += removed_phrase_items(body, anchor_id)
    return blocks


# --- VRAM helpers ----------------------------------------------------------
def vram_gb(device=None):
    try:
        dev = device or mm.get_torch_device()
        total, free = mm.get_total_memory(dev) / GB, mm.get_free_memory(dev) / GB
        if total > 0:
            return round(total, 2), round(free, 2)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            fb, tb = torch.cuda.mem_get_info()
            return round(tb / GB, 2), round(fb / GB, 2)
    except Exception:
        pass
    return 0.0, 0.0


def _deep_cleanup():
    """Release VRAM + RAM between shots so a long chain doesn't accumulate and OOM.
    Runs a Python GC pass (frees dereferenced tensors / CPU buffers), then hands
    ComfyUI its aggressive cache purge, then empties the CUDA allocator's cached
    blocks and IPC handles. Cheap relative to sampling; called once per beat."""
    gc.collect()
    try:
        mm.soft_empty_cache(True)      # aggressive (unload_all_models path)
    except TypeError:
        mm.soft_empty_cache()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def dit_resident_gb(model):
    """Loaded model weight size in GB, using ComfyUI's OWN accounting so the
    figure matches how ComfyUI itself budgets VRAM and counts packed/quantized
    dtypes correctly. The old raw numel*element_size sum over DiT params
    over-counts NVFP4/FP8 (it reads unpacked shapes + scale tensors) -- that's
    what produced the impossible 61.7GB reading. Falls back progressively."""
    # 1) ModelPatcher.model_size() -- authoritative, same call ComfyUI budgets with
    try:
        sz = model.model_size()
        if sz and sz > 0:
            return round(sz / GB, 2)
    except Exception:
        pass
    # 2) model_management.module_size on the inner model
    try:
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(mm, "module_size"):
            sz = mm.module_size(inner)
            if sz and sz > 0:
                return round(sz / GB, 2)
    except Exception:
        pass
    # 3) last resort: raw param sum (over-counts quant, but non-zero)
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    if dm is not None and hasattr(dm, "parameters"):
        try:
            return round(sum(p.numel() * p.element_size() for p in dm.parameters()) / GB, 2)
        except Exception:
            pass
    return 0.0


def estimate_shot_frames(total_gb, resident_gb, headroom_gb, pixels=None, free_gb=None):
    """Largest grid-aligned shot length the card can attempt.

    Budgets from CARD CAPACITY minus measured weight size -- deliberately NOT from
    instantaneous free VRAM. Free VRAM is read at one moment during graph execution,
    and whatever is resident right then (the checkpoint, the text encoder, a LoRA's
    adapters, another node's leftovers) makes it read far lower than the memory
    actually available across the render. That produced a real failure: a 13.6GB
    checkpoint at 640p floored to 124f/5s even though a forced 10s shot ran fine,
    peaking at 15.2GB on a 15.9GB card and settling at 11.2GB. Capacity minus
    weights is stable regardless of when the node happens to run.

    Model-agnostic: the only model-dependent input is resident_gb (ComfyUI's own
    accounting), so NVFP4 / FP8 / INT8 / GGUF / BF16 all flow through the same
    arithmetic -- a heavier checkpoint leaves less room and yields shorter shots.
    The rest is the latent + activations, which scale with pixels x frames, so
    `pixels` normalizes any resolution back to the native reference.

    Continuous fit, anchored to MEASURED points on a 16GB card:
        1344x768, ~11.7GB NVFP4  -> 243f fits; 362f overflowed by ~4.3GB
        640p,     ~13.6GB HQ     -> 10s (243f) fits, peak 15.2GB
    free_gb is still accepted (callers pass it) but is used only as a sanity floor:
    if the card is genuinely almost full right now, don't promise a long shot."""
    floor = align_frame_count(MIN_SHOT_FRAMES)
    if total_gb <= 0:
        return floor
    avail = total_gb - resident_gb - headroom_gb
    if avail <= 0:
        # Weights alone exceed the card: no resolution helps, and scaling a negative
        # value would perversely make lower resolutions look worse than higher ones.
        return floor
    if pixels and pixels > 0:
        avail *= NATIVE_PIXELS / float(pixels)     # lower res -> effectively more room
    frames = FRAMES_PER_GB * (avail + FRAMES_BASELINE_GB)
    # Sanity floor from a LIVE reading: capacity-minus-weights is the right basis
    # (see above), but if the card is genuinely almost empty right now -- another
    # app holding VRAM, a model that failed to unload -- do not promise a long
    # shot on paper. Only ever REDUCES the estimate; it can never raise it, so a
    # momentarily low reading during model load can't floor the budget the way
    # budgeting from free_gb directly used to.
    if free_gb is not None and free_gb > 0:
        live = FRAMES_PER_GB * ((free_gb * (1.0 - SPIKE_RESERVE)) + FRAMES_BASELINE_GB)
        if pixels and pixels > 0:
            live = FRAMES_PER_GB * (((free_gb * (1.0 - SPIKE_RESERVE))
                                     * (NATIVE_PIXELS / float(pixels))) + FRAMES_BASELINE_GB)
        frames = min(frames, live)
    frames = max(MIN_SHOT_FRAMES, min(H3_MAX_FRAMES, int(frames)))
    return max(floor, align_frame_count(min(H3_MAX_FRAMES, frames)))


def resolve_shot_frames(shot_seconds, fps, total_gb, resident_gb, headroom_gb,
                        allow_oversize=False, pixels=None, free_gb=None):
    """Returns (frames, note).

    Auto mode (shot_seconds <= 0): frames = the VRAM budget estimate (resolution-
    scaled). Forced mode: the requested length is clamped DOWN to the budget
    unless allow_oversize is set. When VRAM is unknown the request is honored."""
    budget = estimate_shot_frames(total_gb, resident_gb, headroom_gb, pixels, free_gb)
    if not (shot_seconds and float(shot_seconds) > 0):
        return budget, ""
    requested = align_frame_count(min(H3_MAX_FRAMES, max(5, round(float(shot_seconds) * fps))))
    if total_gb <= 0 or requested <= budget:
        return requested, ""
    if allow_oversize:
        return requested, (f"OVERSIZE: {requested}f requested vs {budget}f budget -- honoring it; "
                           f"may spill to system RAM (slow) or OOM")
    return budget, (f"requested {requested}f (~{requested/max(1,fps):.1f}s) exceeds the ~{budget}f VRAM "
                    f"budget -- clamped to {budget}f (~{budget/max(1,fps):.1f}s). Set allow_oversize_shots to override")


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


# --- conditioning + decode -------------------------------------------------
def _resize(image, width, height, crop):
    s = image[..., :3].movedim(-1, 1)
    s = comfy.utils.common_upscale(s, width, height, "lanczos", crop)
    return s.movedim(1, -1)


def _empty_av_latent(width, height, length, fps, batch_size=1):
    fc, lt, at = temporal_shape(length, fps)
    video = torch.zeros([batch_size, 24, lt, height // 16, width // 16], device=mm.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, at], device=mm.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, fc


def _decode_video(vae, out_latent, tiled, free_first=None, tile_t=None, tile_xy=None):
    """Decode the video latent. If `free_first` is the diffusion model, unload it
    first: sampling is finished, and the ~5GB video VAE needs the room. Leaving the
    DiT (plus resident bypass-LoRA adapters) on the card while the VAE loads is a
    second ratchet -- ComfyUI would otherwise evict reactively, after spilling."""
    if free_first is not None:
        try:
            mm.free_memory(1e30, mm.get_torch_device(), keep_loaded=[])
        except Exception:
            pass
    latent = out_latent["samples"]
    if latent.is_nested:
        latent = latent.unbind()[0]
    if tiled:
        # Temporal + spatial tiling. Without tile_t the VAE expands the WHOLE latent
        # clip at once, which on a 243-frame 1344x768 shot is the single largest
        # allocation in the run -- and on an unpruned checkpoint that is already
        # streaming, it is what tips the card over. Decoding in temporal chunks
        # trades a little speed for a much lower peak; None keeps ComfyUI's defaults.
        args = {}
        if tile_t:
            args["tile_t"] = int(tile_t)
            args["overlap_t"] = max(1, int(tile_t) // 8)
        if tile_xy:
            args["tile_x"] = int(tile_xy)
            args["tile_y"] = int(tile_xy)
        try:
            imgs = vae.decode_tiled(latent, **args) if args else vae.decode_tiled(latent)
        except TypeError:
            imgs = vae.decode_tiled(latent)      # older signature without tile_t
    else:
        imgs = vae.decode(latent)
    if len(imgs.shape) == 5:
        imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
    return imgs


def _decode_audio(audio_vae, out_latent):
    latent = out_latent["samples"]
    if latent.is_nested:
        latent = latent.unbind()[-1]
    audio = audio_vae.decode(latent).movedim(-1, 1)
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std
    sr = getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 44100))
    return {"waveform": audio, "sample_rate": sr}


def _build_shot_conditioning(clip, vae, prompt, width, height, length, fps, handoff):
    latent, fc = _empty_av_latent(width, height, length, fps)
    images, keyframes = [], []
    if handoff is not None:
        img = _resize(handoff[:1], width, height, "disabled")
        images.append(img)
        keyframes.append({"resolved_frame_index": 0, "image": img})
    tokens = clip.tokenize(prompt, images=images)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if keyframes:
        for kf in keyframes:
            kf["latent"] = vae.encode(kf.pop("image"))
        cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes, "minimax_frame_count": fc})
    return cond, latent


def _find_h3_sampling_node():
    """Locate the H3 sigma-shift node under ANY registered name. It was renamed
    to 'ModelSamplingMiniMaxH3' in a later patch (kijai PR #15243); older 0.30.x
    builds register it under a different id, so exact-key lookup misses it. Try
    the known names, then fuzzy-scan all node mappings for the H3 model-sampling
    node. Returns (class, key) or (None, None)."""
    maps = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    for key in ("ModelSamplingMiniMaxH3", "ModelSamplingMinimaxH3", "ModelSamplingMinimax", "ModelSamplingH3"):
        if key in maps:
            return maps[key], key
    for k, v in maps.items():
        kl = k.lower()
        if "sampl" in kl and (("minimax" in kl and "h3" in kl) or ("h3" in kl and "shift" in kl)):
            return v, k
    for k, v in maps.items():
        kl = k.lower()
        if ("minimax" in kl or "h3" in kl) and ("shift" in kl or "sampling" in kl):
            return v, k
    return None, None


def _call_node(cls, model, shift_video, shift_audio):
    """Call the H3 sampling node whether it uses the V1 (INPUT_TYPES/FUNCTION) or
    V3 (define_schema/execute) API, mapping the shift args by name."""
    inst = cls()
    # V1 API
    if hasattr(cls, "INPUT_TYPES") and getattr(cls, "FUNCTION", None):
        req = cls.INPUT_TYPES().get("required", {})
        kwargs = {}
        for name in req:
            low = name.lower()
            if low == "model":
                kwargs[name] = model
            elif "video" in low:
                kwargs[name] = float(shift_video)
            elif "audio" in low:
                kwargs[name] = float(shift_audio)
        out = getattr(inst, cls.FUNCTION)(**kwargs)
        return out[0] if isinstance(out, (tuple, list)) else out
    # V3 API: an execute()/patch() classmethod taking model + shift kwargs
    fn = None
    for cand in ("execute", "patch", "apply"):
        if hasattr(inst, cand):
            fn = getattr(inst, cand); break
    if fn is None:
        raise RuntimeError("unknown node API")
    out = fn(model=model, shift_video=float(shift_video), shift_audio=float(shift_audio))
    out = getattr(out, "result", out)                 # V3 NodeOutput
    return out[0] if isinstance(out, (tuple, list)) else out


def _direct_model_sampling(model, shift_video, shift_audio):
    """Fallback that sets the shift on the model's own model_sampling object
    without any node -- version-tolerant and V3-proof, since it uses model-level
    APIs (get_model_object / set_parameters / add_object_patch) rather than
    calling a node. Copies the sampling object so the base model isn't mutated,
    and applies audio_shift only if the installed set_parameters accepts it."""
    import inspect, copy
    m = model.clone()
    ms = copy.copy(m.get_model_object("model_sampling"))
    sig = inspect.signature(ms.set_parameters)
    kwargs = {}
    if "shift" in sig.parameters:
        kwargs["shift"] = float(shift_video)
    if "audio_shift" in sig.parameters:
        # NOTE: on ComfyUI 0.31 the audio latent is carried on the video schedule
        # scaled by audio_scale = shift_video / shift_audio (12/3 = 4.0), applied in
        # process_latent_in and undone in process_latent_out. Forcing that ratio to
        # 1.0 (audio_shift == shift_video) as a "legacy 0.30" emulation produces
        # SILENT output -- the model needs the scaling -- so it is not offered.
        kwargs["audio_shift"] = float(shift_audio)
    if not kwargs:
        raise RuntimeError("set_parameters takes no shift")
    ms.set_parameters(**kwargs)
    m.add_object_patch("model_sampling", ms)
    return m


def apply_h3_model_sampling(model, shift_video, shift_audio):
    """Apply H3's dual video/audio flow schedule from INSIDE the node so a missing
    upstream patch can't silently gibberish the audio.

    On ComfyUI 0.31+ the H3 nodes are V3-schema and don't live in the legacy
    NODE_CLASS_MAPPINGS the old way -- AND the model already defaults to the correct
    FLOW_AV schedule (12/3) at load. So the reliable path here is a DIRECT model-
    level patch (works regardless of node API); the node call is only a secondary.
    Order: direct model_sampling patch -> node under any name (V1/V3) -> give up with
    an informative, non-alarming note. Shifts aren't hardcoded (12/3 base, ~8 video
    for low-step MXFP8, ~4-6 audio for turbo)."""
    try:
        return _direct_model_sampling(model, shift_video, shift_audio), \
               f"model_sampling video {shift_video:g}/audio {shift_audio:g} (direct)"
    except Exception:
        pass
    cls, key = _find_h3_sampling_node()
    if cls is not None:
        try:
            return _call_node(cls, model, shift_video, shift_audio), \
                   f"model_sampling video {shift_video:g}/audio {shift_audio:g} (via {key})"
        except Exception:
            pass
    return model, (f"model_sampling not explicitly set (video {shift_video:g}/audio {shift_audio:g}); "
                   "on ComfyUI 0.30+ the model already defaults to the correct schedule, so this is "
                   "usually harmless -- only set shift_video/audio explicitly if you're on a low-step "
                   "MXFP8/turbo profile and the audio sounds wrong")


def _upscale_model_list():
    """Filenames in models/upscale_models, plus 'none'. Read fresh at INPUT_TYPES
    time so newly-added models show up on a graph reload."""
    try:
        import folder_paths
        return ["none"] + list(folder_paths.get_filename_list("upscale_models"))
    except Exception:
        return ["none"]


def _invoke_node(cls, **kwargs):
    """Call a registered ComfyUI node (V1 FUNCTION or V3 execute) with kwargs and
    return its first output. Used to reuse ComfyUI's own upscale-model loader/apply
    so we don't reimplement spandrel loading or tiled scaling."""
    inst = cls()
    fn = None
    if getattr(cls, "FUNCTION", None) and hasattr(inst, cls.FUNCTION):
        fn = getattr(inst, cls.FUNCTION)
    else:
        for cand in ("execute", "upscale", "load_model", "load"):
            if hasattr(inst, cand):
                fn = getattr(inst, cand); break
    if fn is None:
        raise RuntimeError("no callable entrypoint")
    out = fn(**kwargs)
    out = getattr(out, "result", out)
    return out[0] if isinstance(out, (tuple, list)) else out


def _find_node(substrings):
    """Find a registered node whose key contains all of `substrings` (lowercased)."""
    maps = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    for k, v in maps.items():
        kl = k.lower()
        if all(s in kl for s in substrings):
            return v
    return None


def _resize_short_edge(frames, target, method="lanczos"):
    """Resize a [B,H,W,C] frame batch so its short edge == target (keeping aspect,
    snapped to /32). Plain high-quality resize -- enlarges, doesn't add detail."""
    b, h, w, c = frames.shape
    if min(h, w) == target:
        return frames
    if h <= w:
        nh = target; nw = max(32, int(round(target * w / h / 32) * 32))
    else:
        nw = target; nh = max(32, int(round(target * h / w / 32) * 32))
    s = frames.movedim(-1, 1)
    s = comfy.utils.common_upscale(s, nw, nh, method, "disabled")
    return s.movedim(1, -1)


def _upscale_frames(frames, mode, model_name, target_short_edge, batch=4):
    """Optional post-pass upscale of the finished frames (on CPU).
      mode 'model'   : run a ComfyUI upscale model (Real-ESRGAN/UltraSharp class)
                       via the registered loader+apply nodes, chunked with cleanup
                       so 2000+ frames don't OOM; then fit to target short edge.
      mode 'rtx'     : NVIDIA RTX Video Super Resolution (Tensor Cores; fastest,
                       best quality for video -- needs Nvidia_RTX_Nodes_ComfyUI).
      mode 'lanczos' : plain high-quality resize to the target short edge.
    Any failure falls back to lanczos (or the raw frames), so it never breaks a
    render. Returns (frames, note). NOTE: this SHARPENS/ENLARGES; it does not
    reconstruct video detail the way a second-model (LTX 2.3) pass does."""
    if mode == "off" or frames is None or getattr(frames, "shape", [0])[0] == 0:
        return frames, ""
    note = ""
    if mode == "rtx":
        # NVIDIA RTX Video Super Resolution (Comfy-Org/Nvidia_RTX_Nodes_ComfyUI).
        # Runs on RTX Tensor Cores -- far faster than ESRGAN-class models and
        # generally cleaner on video, though like them it enhances/enlarges rather
        # than reconstructing detail (an LTX 2.3 re-generation does that).
        try:
            rtx = (_find_node(["rtx", "video", "super"]) or _find_node(["rtxvideosuperresolution"])
                   or _find_node(["rtx", "upscale"]))
            if rtx is None:
                raise RuntimeError("RTX node not installed (Nvidia_RTX_Nodes_ComfyUI)")
            scale = 2
            if target_short_edge and int(target_short_edge) > 0:
                cur = min(frames.shape[1], frames.shape[2])
                if cur > 0:
                    scale = max(1, min(4, int(round(int(target_short_edge) / cur))))
            out = []
            n = frames.shape[0]
            step = max(1, int(batch))
            for st in range(0, n, step):
                part = frames[st:st + step]
                res = None
                for kw in ({"image": part, "scale": scale}, {"images": part, "scale": scale},
                           {"image": part, "scale_factor": scale}, {"image": part}):
                    try:
                        res = _invoke_node(rtx, **kw); break
                    except TypeError:
                        continue
                if res is None:
                    raise RuntimeError("RTX node signature not recognized")
                out.append(res.detach().to("cpu"))
                del res, part
                _deep_cleanup()
            frames = torch.cat(out, dim=0)
            note = f"RTX Video Super Resolution x{scale}"
            if target_short_edge and int(target_short_edge) > 0:
                frames = _resize_short_edge(frames, int(target_short_edge))
                note += f"; fit to {int(target_short_edge)}px short edge"
            return frames, note
        except Exception as e:
            mode = "model"
            note = f"RTX upscale unavailable ({e}); fell back to model/lanczos"
    if mode == "model" and model_name and model_name != "none":
        try:
            loader = _find_node(["upscale", "model", "load"]) or _find_node(["loadupscalemodel"])
            applier = _find_node(["imageupscale", "model"]) or _find_node(["upscaleimageusingmodel"])
            if loader is None or applier is None:
                raise RuntimeError("upscale-model nodes not found")
            up_model = _invoke_node(loader, model_name=model_name)
            out = []
            n = frames.shape[0]
            for s in range(0, n, max(1, int(batch))):
                part = frames[s:s + max(1, int(batch))]
                res = _invoke_node(applier, upscale_model=up_model, image=part)
                out.append(res.detach().to("cpu"))
                del res, part
                _deep_cleanup()
            frames = torch.cat(out, dim=0)
            note = f"upscaled with {model_name}"
        except Exception as e:
            mode = "lanczos"
            note = f"model upscale unavailable ({e}); used lanczos"
    if target_short_edge and int(target_short_edge) > 0:
        try:
            frames = _resize_short_edge(frames, int(target_short_edge))
            note = (note + "; " if note else "") + f"fit to {int(target_short_edge)}px short edge"
        except Exception as e:
            note = (note + "; " if note else "") + f"resize failed ({e})"
    elif mode == "lanczos" and not note:
        note = "lanczos selected but no target set -> unchanged"
    return frames, note


def lora_active(model):
    """True if a LoRA is applied to this model, by either mechanism.

    Stock LoraLoaderModelOnly folds deltas in as ModelPatcher weight *patches*;
    bypass LoRAs (turbo packs) register *injections* / wrappers instead. Detecting
    both matters because a distilled LoRA compresses ~20 steps into 4-8, so the
    model commits to global composition -- including HOW MANY PEOPLE are in frame --
    within the first step or two and then reinforces that choice rather than
    revising it. That is why turbo LoRAs duplicate subjects even when the prompt is
    clean, and why the subject-count guard has to be forced on for them regardless
    of resolution."""
    try:
        if getattr(model, "patches", None):
            return True
        for attr in ("injections", "wrappers"):
            d = getattr(model, attr, None) or {}
            if any(len(v) for v in d.values()):
                return True
    except Exception:
        pass
    return False


def lora_overhead_gb(model):
    """Extra VRAM a bypass-LoRA holds resident during sampling.

    A bypass LoRA (e.g. the MiniMax-H3 Turbo LoRA) does NOT fold into the weights:
    it keeps every low-rank A/B pair live in bf16 and adds lora(x) in activation
    space each forward. With ~208 adapters plus per-adapter activations that is a
    real, measurable chunk the budget must not spend on frames -- otherwise the
    node picks a shot length that fits the base model and then overflows once the
    adapters and their activations land. Returns an estimate in GB (0 if none)."""
    try:
        injections = getattr(model, "injections", None) or {}
        n_inj = sum(len(v) for v in injections.values())
        n_hooks = 0
        for v in injections.values():
            for inj in v:
                n_hooks += len(getattr(inj, "hooks", ()) or ())
        wrappers = getattr(model, "wrappers", None) or {}
        n_wrap = sum(len(w) for w in wrappers.values())
        if not (n_inj or n_hooks or n_wrap):
            return 0.0
        # low-rank deltas are small individually; the cost that matters is the
        # per-adapter activation working set during the forward pass.
        return round(max(0.6, 0.004 * max(n_hooks, 1)), 2)
    except Exception:
        return 0.0


# Fingerprint of the model used by the previous run, so a checkpoint swap can be
# detected between queue executions. Module-level: it must outlive the node
# instance, which ComfyUI recreates per execution.
_LAST_MODEL_FP = {"fp": None}


def model_fingerprint(model):
    """A cheap, stable identity for the loaded DiT: (quant format, layer count,
    weight bytes, class name). Changes whenever the checkpoint changes -- a
    different quant, a pruned-vs-full build, or a different model entirely -- while
    staying identical across shots of the same run. Deliberately avoids hashing
    weights, which would cost more than the flush it guards."""
    try:
        dm = getattr(getattr(model, "model", None), "diffusion_model", None)
        fmts, n = {}, 0
        if dm is not None and hasattr(dm, "modules"):
            for mod in dm.modules():
                n += 1
                f = getattr(mod, "quant_format", None)
                if f:
                    fmts[f] = fmts.get(f, 0) + 1
        top = max(fmts.items(), key=lambda kv: kv[1])[0] if fmts else "none"
        size = 0
        try:
            size = int(model.model_size())
        except Exception:
            pass
        cls = type(dm).__name__ if dm is not None else "unknown"
        return (top, n, size, cls)
    except Exception:
        return None


def flush_for_model_change(model):
    """Detect a checkpoint swap since the last run and, if one happened, hard-flush
    GPU state before doing anything else.

    Why this matters: ComfyUI keeps previously-loaded models in current_loaded_models
    and only evicts reactively. Swapping checkpoints mid-session (e.g. NVFP4 -> FP8 ->
    MXFP8 while comparing quality) leaves the OLD DiT resident alongside the new one,
    plus any hooks/injections a previous LoRA installed and stale cached allocator
    blocks sized for the old model's layers. The result is a card that is already
    half full before the first shot samples -- which looks exactly like the node
    over-spilling, when in fact the budget was computed against memory the previous
    checkpoint never released.

    Returns a note for `info` when a change was detected (empty string otherwise)."""
    fp = model_fingerprint(model)
    prev = _LAST_MODEL_FP.get("fp")
    _LAST_MODEL_FP["fp"] = fp
    if prev is None or fp is None or prev == fp:
        return ""
    try:
        mm.unload_all_models()          # drop every resident model, not just the cache
    except Exception:
        pass
    # Never let a cleanup failure abort the run: the flush is best-effort hygiene,
    # and a partially-flushed card is still better than raising here.
    for _ in range(2):                  # 2nd pass frees blocks released by the 1st
        try:
            _deep_cleanup()
        except Exception:
            pass
    old_fmt, _n, old_sz, _c = prev
    new_fmt = fp[0]
    return (f"model changed since last run ({old_fmt} ~{old_sz / GB:.1f}GB -> {new_fmt} "
            f"~{fp[2] / GB:.1f}GB): flushed all resident models and VRAM caches")


def quant_accel_note(model):
    """Report the loaded DiT's quant format and whether THIS card can run it on
    tensor cores natively -- so a silent fall back to emulated/upcast math shows up
    in `info` instead of just looking like slow output.

    The node itself never sets dtypes, never autocasts and never rebuilds modules:
    it delegates sampling to ComfyUI's common_ksampler, and its only model patch is
    a schedule-object patch (add_object_patch on 'model_sampling'). So NVFP4/MXFP8
    tensor-core acceleration is entirely ComfyUI's dispatch on the quantized layers
    -- which is what we want: nothing here can disturb it. This is a read-only
    check."""
    try:
        import comfy.model_management as _mm
        dm = getattr(getattr(model, "model", None), "diffusion_model", None)
        fmts = {}
        if dm is not None and hasattr(dm, "modules"):
            for mod in dm.modules():
                f = getattr(mod, "quant_format", None)
                if f:
                    fmts[f] = fmts.get(f, 0) + 1
        if not fmts:
            return ""
        top = max(fmts.items(), key=lambda kv: kv[1])[0]
        native = None
        if "nvfp4" in top:
            native = getattr(_mm, "supports_nvfp4_compute", lambda: None)()
        elif "mxfp8" in top:
            native = getattr(_mm, "supports_mxfp8_compute", lambda: None)()
        if native is True:
            return f"{top}: native tensor-core compute"
        if native is False:
            return (f"WARNING {top}: this card/torch cannot run it natively -- weights are being "
                    f"upcast, so you pay full-precision compute with none of the speedup")
        return f"{top} weights"
    except Exception:
        return ""




def _evict_all_but(keep_model):
    """Unload every model EXCEPT the diffusion model from the GPU.

    This is the fix for VRAM ratcheting across a long chain. soft_empty_cache()
    only drops the CUDA allocator's cached blocks -- it does NOT unload models, so
    ComfyUI keeps the Qwen3-VL text encoder (~14.6GB) and both VAEs resident in
    current_loaded_models alongside the DiT. Each shot re-encodes the prompt
    (text encoder), encodes the handoff keyframe (video VAE), then samples (DiT),
    so all three compete for the card; ComfyUI only evicts reactively, i.e. AFTER
    it has already spilled. With a bypass LoRA also holding 208 bf16 adapters
    resident there is no room left, and every shot leaves the card fuller.

    Freeing them explicitly, right after conditioning is built and before
    sampling, keeps only what the sampler actually needs on the GPU."""
    try:
        keep = []
        for lm in list(getattr(mm, "current_loaded_models", [])):
            try:
                if lm.model is keep_model or getattr(lm, "model", None) is getattr(keep_model, "model", None):
                    keep.append(lm)
            except Exception:
                pass
        mm.free_memory(1e30, mm.get_torch_device(), keep_loaded=keep)
    except Exception:
        try:
            mm.soft_empty_cache(True)
        except Exception:
            pass






class H3LongVideosV1:
    CATEGORY = "sampling/minimax"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("images", "audio", "info", "script", "frames_per_shot", "total_frames", "shots", "video_seconds")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default":
                    "A woman with short silver hair and a scar over her left eyebrow. Warm "
                    "late-afternoon light, cinematic, 2K.\n"
                    "wardrobe: weathered red flight jacket, grey cargo shorts, black boots\n\n"
                    "walks across the tarmac toward a small propeller plane.\n\n"
                    "climbs in and flips the switches; the propeller spins.\n\n"
                    "taxis down the grass runway, the tail lifting.\n\n"
                    "the plane leaves the ground; wide shot banking against the sky.",
                    "tooltip": "This IS the integrated_multimodal_description (the visual/action "
                               "timeline). First paragraph = PERMANENT IDENTITY kept across the whole "
                               "video (hair, face, build) -- put NO clothing in this prose, or it can't "
                               "be changed later. Put clothing on a 'wardrobe:' line (in the first "
                               "paragraph and/or the character_memory field); it's the only channel that "
                               "can be changed/removed mid-chain. Each later paragraph = one scene beat. "
                               "Put dialogue and 'lips closed' beats in the beat bodies."}),
                "resolution": (resolution_options(), {
                    "tooltip": "Preset, all multiples of 32. Three short-edge tiers per ratio: native 768 "
                               "(best detail), balanced 640, fast 512 (generate-then-upscale). Lower tiers "
                               "render faster, free VRAM, and unlock longer shots (budget is res-aware)."}),
                # Base H3 (NVFP4/FP8, no distill LoRA) needs ~20 steps with res_multistep+simple.
                # 6-8 steps only makes sense WITH a working 4-step distill/turbo LoRA or an MXFP8
                # checkpoint tuned for low steps -- at 6-8 on the bare base model the frame comes
                # out soft/under-formed (faces worst). Default is the safe base value.
                "steps": ("INT", {"default": 20, "min": 1, "max": 200,
                    "tooltip": "Base H3 wants ~20 (res_multistep + simple). Drop to 6-8 ONLY with a "
                               "working distill/turbo LoRA or a low-step MXFP8 checkpoint -- on the "
                               "bare base model, low steps are the #1 cause of soft output."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "plan_only": ("BOOLEAN", {"default": False,
                    "tooltip": "Preview the shot split WITHOUT rendering. Uses THIS node's own settings (no "
                               "second node, no duplicate entry): returns the plan in 'info' and the "
                               "shots/frames/seconds outputs near-instantly. Turn off to render for real."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                "global_soundscape": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "AMBIENT/environmental sound only (rain, room tone, footsteps, engines). "
                               "Appended to every shot as overall_soundscape. NOT for dialogue -- speech "
                               "and lip timing live in the prompt beats. Leave blank for no ambient bed."}),
                "non_diegetic_music": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Background SCORE only -- genre, mood, instrumentation, tempo -- music that "
                               "is NOT part of the scene. Music is OPT-IN: leave this BLANK and the node "
                               "emits 'non_diegetic_music: N/A' on every shot so H3 adds no score (fixes "
                               "unwanted music). Fill it in to request a specific score. Not for music a "
                               "character plays/hears (that's diegetic; put it in the beat)."}),
                "apply_model_sampling": ("BOOLEAN", {"default": True,
                    "tooltip": "Patch ModelSamplingMiniMaxH3 (the dual video/audio schedule) inside the "
                               "node so you don't have to wire it upstream. Without it, H3's audio comes "
                               "out as gibberish. Turn OFF only if you patch it yourself upstream."}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 32.0, "step": 0.5,
                    "tooltip": "Video flow shift. 12 = base H3 (correct default). A low-step MXFP8 "
                               "checkpoint wants ~8. Only used when apply_model_sampling is on."}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 16.0, "step": 0.5,
                    "tooltip": "Audio flow shift. 3 = base H3. A 4-step distill/turbo LoRA setup wants "
                               "~4-6. Only used when apply_model_sampling is on."}),
                "trim_seam": ("BOOLEAN", {"default": True}),
                "vary_seed_per_shot": ("BOOLEAN", {"default": True}),
                "handoff_offset": ("INT", {"default": 0, "min": 0, "max": 12, "step": 1,
                    "tooltip": "End each shot this many frames early and hand THAT frame to the next "
                               "shot instead of the literal last frame. Set 2-4 if chained shots open "
                               "with moving/talking mouths -- it avoids seeding the next shot with a "
                               "mid-word open-mouth pose. Trims the matching audio tail too. 0 = last frame."}),
                "shot_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 15.1, "step": 0.5,
                    "tooltip": "Length of EACH shot in seconds. 0 = auto (largest that fits at native res). "
                               "One paragraph = one shot, so total video = (paragraph count) x this. Max ~15s."}),
                "allow_oversize_shots": ("BOOLEAN", {"default": False,
                    "tooltip": "OFF (default): a forced shot_seconds that won't fit VRAM is clamped DOWN to "
                               "what fits, and the clamp is reported in info. ON: honor the requested length "
                               "even if it exceeds the budget -- the render may spill into system RAM (slow) "
                               "or OOM. Only affects forced shot_seconds, not auto."}),
                "vram_headroom_gb": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 32.0, "step": 0.5}),
                "allow_res_backoff": ("BOOLEAN", {"default": True,
                    "tooltip": "If VRAM is tight, step resolution down instead of failing."}),
                "mute_nonspeech_audio": ("BOOLEAN", {"default": False,
                    "tooltip": "DETERMINISTIC gibberish fix: FULLY silence the audio of any shot that has no "
                               "scripted dialogue (no double-quoted line). Prompt-level silencing asks H3 "
                               "not to babble; this guarantees it. TRADE-OFF: it also removes that shot's "
                               "generated ambience/SFX, so lay a continuous ambient bed under the video in "
                               "post. Shots WITH quoted dialogue keep their audio untouched."}),
                "mute_fade_ms": ("INT", {"default": 40, "min": 0, "max": 500, "step": 10,
                    "tooltip": "Fade applied to the AUDIBLE shots that border a silenced one, so audio "
                               "doesn't cut to digital silence with a click. The silenced shots keep NO "
                               "original audio at all -- fading the muted shot itself would leave this many "
                               "ms of the gibberish audible at each end of every muted shot."}),
                "decode_tile_frames": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Temporal tiling for the VAE decode (tile_t). 0 = ComfyUI default, which "
                               "expands the WHOLE clip at once -- the single largest allocation in a run, "
                               "and the usual point where a big checkpoint tips into shared memory. Try 8-16 "
                               "if you spill during decode rather than sampling. Lower = less peak VRAM, "
                               "slightly slower."}),
                "decode_tile_size": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 32,
                    "tooltip": "Spatial tile size for the VAE decode (tile_x/tile_y). 0 = ComfyUI default. "
                               "Try 256 on a tight card at 1344x768."}),
                "cleanup_between_shots": ("BOOLEAN", {"default": True,
                    "tooltip": "Between beats, move each shot's decoded video+audio to system RAM and run "
                               "a full VRAM+RAM purge (GC + CUDA cache), so a long chain doesn't accumulate "
                               "on the GPU and OOM. Recommended on 16GB. Turn off only on a big card where "
                               "you want to skip the per-shot cleanup cost."}),
                "upscale": (["off", "rtx", "model", "lanczos"], {"default": "off",
                    "tooltip": "Optional post-pass on the finished frames. 'rtx' = NVIDIA RTX Video Super "
                               "Resolution (Tensor Cores -- fastest and best for video; needs the "
                               "Nvidia_RTX_Nodes_ComfyUI pack, falls back automatically if absent). 'model' = "
                               "a Real-ESRGAN/UltraSharp upscale model from upscale_model. 'lanczos' = plain "
                               "resize. All of these ENHANCE/ENLARGE; for true detail reconstruction from a "
                               "low-res render, use a separate LTX 2.3 upscale pass."}),
                "upscale_model": (_upscale_model_list(), {
                    "tooltip": "Upscale model from models/upscale_models (used when upscale = model)."}),
                "upscale_target_short_edge": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32,
                    "tooltip": "Fit the result's short edge to this many px (0 = keep the model's native "
                               "factor / no resize). E.g. generate 512 fast, set 768 to land at native size."}),
                "upscale_batch": ("INT", {"default": 4, "min": 1, "max": 64,
                    "tooltip": "Frames per chunk for the model upscale (lower = less VRAM, slower)."}),
                "anchor_override": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Set the persistent look explicitly instead of using the first paragraph."}),
                "auto_wardrobe": ("BOOLEAN", {"default": True,
                    "tooltip": "Read clothing REMOVALS straight from your beat prose -- 'she takes off her "
                               "jacket' drops the jacket with no 'wardrobe:' line needed. Safe: only fires "
                               "on items the character is already wearing, so 'the plane takes off' does "
                               "nothing. Additions/swaps still use 'wardrobe: += ...' (which overrides). "
                               "Turn OFF to control wardrobe only via explicit 'wardrobe:' lines."}),
                "subject_count_guard": (["auto", "on", "off"], {"default": "auto",
                    "tooltip": "Anti-duplication: prepend an explicit subject count to each shot "
                               "(\"Exactly two people in this shot, no duplicates, no other people in "
                               "frame\"). Character duplication gets much more likely BELOW the model's "
                               "native 768 short edge -- fewer pixels per subject pushes the sample out of "
                               "the training distribution and the figure gets tiled. A LoRA causes it too: a "
                               "distilled LoRA fixes composition (including how many people are in frame) "
                               "in its first step or two, so it duplicates even at native size -- there the "
                               "count is moved to the FRONT of the prompt so it binds before the scene. "
                               "'auto' = on when the short edge is under 768 OR a LoRA is applied; "
                               "'on' always; 'off' never."}),
                "auto_silence_nonspeech": ("BOOLEAN", {"default": True,
                    "tooltip": "Stop mouths moving / gibberish audio on shots with no dialogue. Any beat "
                               "with no scripted speech gets an explicit 'lips closed, no dialogue' clause, "
                               "so H3 doesn't animate or vocalize a mouth before real dialogue. Beats with "
                               "quoted dialogue (\"...\") are left alone. To make someone speak, put the "
                               "words in double quotes. Turn OFF to manage lip state yourself."}),
                "character_memory": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Optional dedicated wardrobe channel (same role as a 'wardrobe:' line in "
                               "the first paragraph -- use whichever you prefer; this field wins if both "
                               "are set). Re-stamped into every shot so clothing holds even when the "
                               "camera crops it out. IMPORTANT: this is the ONLY place clothing should "
                               "live -- keep it out of the anchor prose, or a removal won't stick because "
                               "the immutable anchor keeps re-adding it. To change/remove an item "
                               "mid-chain, put 'wardrobe: <new full sheet>' inside the beat where it "
                               "changes; omit the removed item from the new sheet and it stays gone. "
                               "WRITE ATTRIBUTES, NOT NOUN PHRASES: 'silver hair, 27, red jacket' -- NOT "
                               "'a woman with silver hair'. A noun phrase renders as 'She (a woman with...)', "
                               "i.e. two subjects in one clause, which causes character duplication. The node "
                               "strips them automatically, but writing attributes directly is cleaner. "
                               "ONE-TOKEN EDITS (no restating the outfit): 'wardrobe: -= jacket' removes "
                               "the jacket, 'wardrobe: += sunglasses' adds one. TWO+ PEOPLE: name them -- "
                               "'Maya = grey shorts, red jacket; Jon = navy overalls', then edit one at a "
                               "time: 'wardrobe: Maya -= jacket' leaves Jon untouched."}),
            },
        }

    def _render(self, model, clip, vae, audio_vae, negative, prompt, w, h, ln, fps, tiled, sa,
                handoff, decode_tile_frames=0, decode_tile_size=0):
        positive, latent = _build_shot_conditioning(clip, vae, prompt, w, h, ln, fps, handoff)
        seed, steps, cfg, sn, sch, denoise = sa
        # Conditioning is built, so the text encoder and VAEs are dead weight for the
        # whole sampling loop -- evict them and keep only the DiT on the card.
        _evict_all_but(model)
        (out,) = nodes.common_ksampler(model, seed, steps, cfg, sn, sch, positive, negative,
                                       latent, denoise=denoise)
        video = _decode_video(vae, out, tiled, free_first=model,
                              tile_t=decode_tile_frames, tile_xy=decode_tile_size)
        audio = _decode_audio(audio_vae, out)
        del out, positive, latent
        _deep_cleanup()
        return video, audio

    def run(self, model, clip, vae, audio_vae, prompt, resolution,
            steps, cfg, sampler_name, scheduler, seed,
            first_frame=None, fps=24, plan_only=False,
            global_soundscape="", non_diegetic_music="", apply_model_sampling=True,
            shift_video=12.0, shift_audio=3.0, trim_seam=True, vary_seed_per_shot=True,
            handoff_offset=0, vram_headroom_gb=1.5, allow_res_backoff=True,
            decode_tile_frames=0, decode_tile_size=0,
            cleanup_between_shots=True,
            anchor_override="", shot_seconds=0.0, allow_oversize_shots=False,
            character_memory="", auto_wardrobe=True, auto_silence_nonspeech=True,
            subject_count_guard="auto",
            upscale="off", upscale_model="none",
            upscale_target_short_edge=0, upscale_batch=4,
            mute_nonspeech_audio=False, mute_fade_ms=40):

        # FIRST: detect a checkpoint swap since the previous execution and hard-flush.
        # A stale resident model from a different checkpoint would otherwise poison
        # every VRAM measurement below (and leave old hooks/allocator blocks behind),
        # so this must run before the schedule patch and before vram_gb().
        swap_note = flush_for_model_change(model)

        fps = max(1, int(fps))
        w, h = parse_resolution(resolution)
        # H3 is CFG-free (cfg 1): the sampler skips the negative, but common_ksampler
        # still needs a conditioning object, so build an empty one from the clip.
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

        # Patch the dual video/audio schedule onto the model here, so a missing
        # upstream ModelSamplingMiniMaxH3 can't silently produce gibberish audio.
        # Shifts come from the widgets (12/3 base default; MXFP8/turbo differ).
        ms_note = ""
        if apply_model_sampling:
            model, ms_note = apply_h3_model_sampling(model, shift_video, shift_audio)

        paras = split_paragraphs(prompt, "##")
        if anchor_override.strip():
            anchor, beats = anchor_override.strip(), paras
        elif paras:
            anchor, beats = paras[0], paras[1:]
        else:
            anchor, beats = "", []

        total_gb, free_gb = vram_gb()
        resident_gb = dit_resident_gb(model)
        # Weights larger than the card means ComfyUI must stream them: NO shot
        # length or resolution avoids spilling into shared/system memory, so say so
        # plainly rather than letting it look like a tuning problem.
        streaming = total_gb > 0 and resident_gb > 0 and resident_gb > total_gb
        lora_gb = lora_overhead_gb(model)
        eff_headroom = vram_headroom_gb + lora_gb
        ln, ln_note = resolve_shot_frames(shot_seconds, fps, total_gb, resident_gb,
                                          eff_headroom, allow_oversize_shots, w * h, free_gb)
        if lora_gb:
            ln_note = ((ln_note + " ") if ln_note else "") + (
                f"reserved ~{lora_gb:.1f}GB for bypass-LoRA adapters (they stay resident in bf16 "
                f"rather than folding into the weights)")
        accel_note = quant_accel_note(model)
        if streaming:
            ln_note = ((ln_note + " ") if ln_note else "") + (
                f"NOTE: weights (~{resident_gb:.1f}GB) exceed VRAM (~{total_gb:.1f}GB), so they stream "
                f"from system RAM -- some shared-memory spill is unavoidable at ANY resolution or shot "
                f"length. Use a checkpoint that fits (pruned) to eliminate it")
        tiled = total_gb > 0 and (total_gb - resident_gb) < 20

        # Sub-native renders duplicate subjects far more often, so default the guard on
        # there and leave native renders alone (the extra clause costs prompt budget).
        lora_on = lora_active(model)
        # 'auto' fires below native resolution AND whenever a LoRA is applied: a
        # distilled LoRA fixes the subject count in its first step or two, so the
        # count has to be stated even at native size.
        count_subjects = (subject_count_guard == "on" or
                          (subject_count_guard == "auto" and (min(w, h) < 768 or lora_on)))
        fit_warnings = dialogue_fit_warnings(beats, ln / fps)
        gens = distribute_generations(anchor, beats, global_soundscape.strip(),
                                      non_diegetic_music.strip(), character_memory.strip(),
                                      auto_wardrobe, auto_silence_nonspeech, count_subjects,
                                      lora_on)

        if plan_only:
            # Preview the split using THIS node's own settings -- no render, near-instant.
            shots = len(gens)
            sps = round(ln / fps, 2)
            total = round(shots * sps, 2)
            vram_str = f"{total_gb:.1f}GB total / {resident_gb:.1f}GB weights" if total_gb else "VRAM unknown"
            fitw = dialogue_fit_warnings(beats, ln / fps)
            plan = (("DIALOGUE MAY BE CUT OFF -- " + "; ".join(fitw) + ". ") if fitw else "") + \
                   (f"PLAN (no render): {shots} shot(s) x {ln}f (~{sps:g}s each) = ~{total:g}s at {w}x{h}. "
                    f"{len(beats) or 1} beat(s). decode {'tiled' if tiled else 'full'}. {vram_str}."
                    + (f" {ln_note}." if ln_note else ""))
            ph_img = torch.zeros((1, 64, 64, 3))
            ph_audio = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 44100}
            return (ph_img, ph_audio, plan, "\n---\n".join(gens), ln, shots * ln, shots, total)

        spk = speech_flags(beats)          # which shots have real (quoted) dialogue
        vram_trace = []                    # free VRAM after each shot
        muted_flags = []                   # which shots were audio-silenced
        hoff = max(0, int(handoff_offset))
        backoff, video_chunks, audio_chunks = [], [], []
        handoff, sr = first_frame, None
        if cleanup_between_shots:
            _deep_cleanup()          # start the first (heaviest) shot with max free VRAM

        for i, gen_prompt in enumerate(gens):
            # denoise is fixed at 1.0 (partial denoise desyncs the joint AV schedule).
            sa = (seed + i if vary_seed_per_shot else seed, steps, cfg, sampler_name, scheduler, 1.0)
            if i == 0:
                while True:
                    try:
                        frames, audio = self._render(model, clip, vae, audio_vae, negative, gen_prompt, w, h, ln, fps, tiled, sa, handoff, decode_tile_frames, decode_tile_size)
                        break
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if not _is_oom(e):
                            raise
                        mm.soft_empty_cache(True)
                        if not tiled:
                            tiled = True; backoff.append("tiled decode")
                        elif allow_res_backoff and min(w, h) > 384:
                            nw, nh = res_down(w, h); backoff.append(f"res->{nw}x{nh}"); w, h = nw, nh
                        else:
                            raise RuntimeError("H3 Long Videos: not enough VRAM even at the smallest size. "
                                               "Pick a smaller resolution, close other GPU apps, or use a smaller quant.")
            else:
                try:
                    frames, audio = self._render(model, clip, vae, audio_vae, negative, gen_prompt, w, h, ln, fps, tiled, sa, handoff, decode_tile_frames, decode_tile_size)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if not _is_oom(e) or tiled:
                        raise
                    mm.soft_empty_cache(True); tiled = True; backoff.append(f"shot {i+1}: tiled")
                    frames, audio = self._render(model, clip, vae, audio_vae, negative, gen_prompt, w, h, ln, fps, tiled, sa, handoff, decode_tile_frames, decode_tile_size)

            sr = audio["sample_rate"]; wav = audio["waveform"]

            # End the shot `hoff` frames early so the frame handed to the NEXT shot
            # isn't the literal last frame (which may catch an open, mid-word mouth
            # and make the next shot start "talking"). Drop the matching audio tail
            # so this shot's A/V stays aligned. Skipped if the shot is too short.
            if hoff and frames.shape[0] > hoff + 1:
                cut = round(hoff * sr / fps)
                frames = frames[:-hoff]
                if cut:
                    wav = wav[..., :max(0, wav.shape[-1] - cut)]

            # Keep only a CPU copy of the handoff keyframe (re-encoded next shot),
            # and move this shot's decoded video+audio to CPU/RAM immediately so
            # they DON'T pile up in VRAM across the chain -- the main long-run OOM.
            if cleanup_between_shots:
                handoff = frames[-1:].detach().contiguous().to("cpu", copy=True)
            else:
                handoff = frames[-1:].clone()
            if trim_seam and i > 0:
                frames = frames[1:]; wav = wav[..., max(0, round(sr / fps)):]

            # Deterministic gibberish fix: a non-dialogue shot is silenced COMPLETELY.
            #
            # The earlier version faded the first/last `mute_fade_ms` from full
            # volume, which left ~20ms of the original audio audible at BOTH ends of
            # every muted shot -- on a 10-shot chain that is 20 short bursts of the
            # very gibberish the setting exists to remove. The fade belongs on the
            # NEIGHBOURING audible shots instead (applied after the loop), not on the
            # silent one, so nothing of the muted shot survives.
            muted_this_shot = bool(mute_nonspeech_audio and i < len(spk) and not spk[i])
            if muted_this_shot:
                wav = torch.zeros_like(wav)
            muted_flags.append(muted_this_shot)

            if cleanup_between_shots:
                # .contiguous() forces a real copy: after trim_seam / handoff_offset
                # these are SLICES of the decoded GPU tensor, and a view keeps the
                # whole parent allocation alive even after .to("cpu"). Without it the
                # previous shot's full decode is pinned while the next shot samples,
                # which is the VRAM ratchet across a long chain.
                frames_out = frames.detach().contiguous().to("cpu", copy=True)
                wav_out = wav.detach().contiguous().to("cpu", copy=True)
                video_chunks.append(frames_out); audio_chunks.append(wav_out)
                # drop every GPU reference from this shot, then purge VRAM + RAM
                del frames, wav, audio, frames_out, wav_out
                _deep_cleanup()
            else:
                video_chunks.append(frames); audio_chunks.append(wav)
                mm.soft_empty_cache()
            # trace free VRAM after each shot: a falling series means something is
            # still accumulating; a flat one means the chain is stable.
            vram_trace.append(round(vram_gb()[1], 2))

        # Fade the EDGES OF AUDIBLE chunks that border a silenced one, so audio does
        # not cut to digital silence with a click. The silenced shots stay fully
        # silent; only the audible neighbours are ramped.
        if mute_nonspeech_audio and any(muted_flags):
            fade = max(0, int(sr * int(mute_fade_ms) / 1000)) if sr else 0
            for idx, chunk in enumerate(audio_chunks):
                if idx >= len(muted_flags) or muted_flags[idx] or not fade:
                    continue
                n_s = chunk.shape[-1]
                if n_s <= 2 * fade:
                    continue
                prev_muted = idx > 0 and muted_flags[idx - 1]
                next_muted = idx + 1 < len(muted_flags) and muted_flags[idx + 1]
                if prev_muted:
                    ramp = torch.linspace(0.0, 1.0, fade, device=chunk.device, dtype=chunk.dtype)
                    chunk[..., :fade] *= ramp
                if next_muted:
                    ramp = torch.linspace(1.0, 0.0, fade, device=chunk.device, dtype=chunk.dtype)
                    chunk[..., n_s - fade:] *= ramp

        all_frames = torch.cat(video_chunks, dim=0)
        all_audio = torch.cat(audio_chunks, dim=-1)

        # Optional post-pass upscale of the finished frames (safe: any failure
        # falls back to lanczos / raw frames and never breaks the render).
        up_note = ""
        if upscale != "off":
            _deep_cleanup()
            all_frames, up_note = _upscale_frames(all_frames, upscale, upscale_model,
                                                  upscale_target_short_edge, upscale_batch)

        script = "\n---\n".join(gens)
        actual = all_frames.shape[0] / fps
        per_shot_s = ln / fps
        vram_str = f"{total_gb:.1f}GB total / {resident_gb:.1f}GB weights" if total_gb else "VRAM unknown"
        hoff_str = f" handoff -{hoff}f." if hoff else ""
        info = (f"{len(gens)} shot(s) x {ln}f (~{per_shot_s:.1f}s each) = ~{len(gens)*per_shot_s:.1f}s "
                f"at {w}x{h}; {all_frames.shape[0]} frames (~{actual:.1f}s actual). "
                f"decode {'tiled' if tiled else 'full'}. {vram_str}.{hoff_str}"
                + (" DIALOGUE MAY BE CUT OFF -- " + "; ".join(fit_warnings)
                   + ". Shorten the line, or pick a lower resolution tier to keep the duration."
                   if fit_warnings else "")
                + (" subject-count guard ON (sub-native resolution)."
                   if count_subjects and min(w, h) < 768 else "")
                + (f" {swap_note}." if swap_note else "")
                + (f" free VRAM/shot: {vram_trace}." if len(vram_trace) > 1 else "")
                + (f" {accel_note}." if accel_note else "")
                + (f" {ms_note}." if ms_note else "")
                + (f" {ln_note}." if ln_note else "")
                + (f" {up_note}." if up_note else "")
                + (f" Adjusted: {'; '.join(backoff)}." if backoff else ""))
        return (all_frames, {"waveform": all_audio, "sample_rate": sr}, info, script,
                ln, all_frames.shape[0], len(gens), round(actual, 2))


NODE_CLASS_MAPPINGS = {"H3LongVideosV1": H3LongVideosV1}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LongVideosV1": "H3 Long Videos V1"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
