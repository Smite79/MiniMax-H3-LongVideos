"""
H3 Long Videos  (one prompt + one length -> long video+audio)
=============================================================
One node covering both of H3's conditioning tasks:

  * FL2VA -- a first frame (or the previous shot's last frame) anchors the shot.
             This is what drives the chain: each shot continues the one before it.
  * REF2VA -- reference images condition the shot on what a character LOOKS like,
             independent of any frame.

Connect nothing to ref_image_* and it behaves exactly as the FL2VA node always
did. Connect a reference and `ref_mode` decides which shots use it.

THE ONE RULE: a shot carries EITHER references or the last-frame handoff, never
both. They are two task conditionings competing for the same cond_video_latents
slot in comfy/model_base.py -- the refs branch overwrites what the keyframe branch
wrote, while the packed layout still reserves rows for both, so a shot given both
hands the DiT fewer latents than it has condition rows. `ref_mode` chooses:

  first shot                -- refs establish the cast in shot 1, every later shot
                               uses the handoff. Continuity unbroken.
  every shot                -- strongest identity; no handoff, so beats meet as
                               CUTS rather than as one continuous take.
  every shot + handoff ref  -- refs every shot, plus the previous shot's last
                               frame as one more reference. Continuity returns as
                               a soft signal.

You give it a prompt (first paragraph = the look/character kept across the whole
video; each later paragraph = a scene beat), a shot length, and a resolution from
the VRAM-appropriate list. It splits the beats into shots that fit H3's ceiling
and your VRAM, chains them, and returns the finished video + audio.

Requirements: H3 is CFG-free (cfg 1) and needs no negative prompt -- the node
makes an empty one internally. denoise is fixed at 1.0: a partial denoise desyncs
the joint audio/video schedule.

Verified against ComfyUI core (comfy_extras/nodes_minimax_h3.py, model_base.py,
ldm/minimax/model.py, text_encoders/minimax.py, sd.py).
"""

import gc
import math
import os
import re
import torch

import nodes
import comfy.utils
import comfy.samplers
import comfy.nested_tensor
import comfy.model_management as mm
import node_helpers

try:
    from . import overlay as _overlay
except ImportError:      # loaded as a bare file (test_prompt_logic.py), not as a package
    import importlib.util as _ilu
    import os as _os
    _spec = _ilu.spec_from_file_location(
        "h3_overlay", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "overlay.py"))
    _overlay = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_overlay)

AUDIO_LATENT_FPS = 40
GB = 1024 ** 3
H3_MAX_FRAMES = 362
# H3's temporal grid is FIXED at 24 fps -- comfy_extras/nodes_minimax_h3.py hard-codes
# FPS = 24, and the audio latent length is derived from frame_count / 24. The model
# emits 24 fps content no matter what any node asks for, so every seconds<->frames
# conversion here MUST use 24. Treating it as a variable is what made a requested
# 10s shot render 124 frames (~5.2s of real time) when the widget said 12.
H3_FPS = 24
MIN_SHOT_FRAMES = 124          # internal VRAM floor (~5s @24fps)


# --- H3 geometry -----------------------------------------------------------
def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def align_frame_count_nearest(n):
    """Nearest 17k+5 grid point, not the next one up.

    align_frame_count always rounds UP, which is right when honoring a length that
    was ASKED for -- never give back less than requested. It is wrong for an
    ESTIMATE: the grid steps 17 frames (~0.7s), so rounding up added up to 0.7s to
    every content-sized shot and pushed a 9.5s estimate out to 10.1s. Pacing leans
    short on purpose; rounding should not quietly lean the other way."""
    n = max(5, int(n))
    lo = n - ((n - 5) % 17)
    hi = lo + 17
    return lo if (n - lo) <= (hi - n) else hi


def video_latent_t(fc):
    return 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2


def temporal_shape(length, fps=H3_FPS):
    """`fps` is accepted for call-site compatibility but deliberately IGNORED: the
    audio latent must line up with 24 fps video or the shot's sound is stretched
    against its picture."""
    fc = align_frame_count(max(5, length))
    return fc, video_latent_t(fc), round(fc / H3_FPS * AUDIO_LATENT_FPS)


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
    "9:21": (672, 1536),
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
    """ASPECT RATIOS only. `megapixels` decides the size.

    Shape and size are independent, and the widgets now say so. This list used to
    carry three short-edge tiers per ratio (native 768 / balanced 640 / fast 512),
    which baked a SIZE into every label -- and once megapixels existed those labels
    lied whenever it was on. Nothing is lost: the tiers were three points on the
    megapixel axis (~0.98 / ~0.70 / ~0.44MP), and a continuous control reaches them
    and everything between. MID_RES and FAST_RES are kept only as the reference
    anchors documented on them.

    NATIVE_RES still supplies each ratio's exact shape, which is what makes 1.00MP
    reproduce H3's own sizes: the ratio NAMES are approximations -- 1344x768 is
    1.750, i.e. 7:4, NOT 16:9 (1.778) -- so scaling runs from the real dimensions
    rather than the nominal ratio."""
    return list(NATIVE_RES)


def parse_resolution(choice):
    """The chosen ratio's reference dimensions, which `megapixels` then scales.

    Accepts a bare ratio ("16:9") and ALSO the old "16:9 - 1344x768 (native)" form,
    so a workflow saved before this list was simplified still resolves to the right
    shape instead of silently falling back to the first entry. Unrecognized input
    gives 16:9."""
    text = (choice or "").strip()
    if text in NATIVE_RES:
        return NATIVE_RES[text]
    m = re.search(r"(\d+)\s*x\s*(\d+)", text)          # legacy label carried its size
    if m:
        return int(m.group(1)), int(m.group(2))
    for r in NATIVE_RES:                                # legacy label led with the ratio
        if text.startswith(r):
            return NATIVE_RES[r]
    return NATIVE_RES["16:9"]


# --- sizing by pixel budget -------------------------------------------------
# Cost and training-distribution match are functions of TOKEN COUNT --
# (h/16)*(w/16)*frames -- which tracks total pixels, not the short edge. A
# short-edge target makes two aspect ratios look comparable when they are not:
#
#   1:1  768x768   short edge 768, reads native      ->  0.56 MP, 43% under
#   21:9 1536x672  short edge 672, reads sub-native  ->  0.98 MP, full budget
#
# Scaling from the PRESET's own dimensions (rather than from a nominal ratio) is
# what makes 1.00MP reproduce each preset's native size exactly. That distinction
# is real: 1344x768 is 1.750, i.e. 7:4 -- NOT 16:9, which is 1.778 -- and
# 1536x672 is 16:7, not 21:9. Computing from a nominal 16:9 lands on 1376x768 and
# never reproduces the native.
#
# This does NOT change the sigma schedule. H3's shift is a fixed 12.0 in its model
# config with no resolution-dependent term, unlike Flux/SD3 dynamic shifting.
MP_UNIT = 1024 * 1024          # 1 MP == 1024x1024, matching ComfyUI's own convention
RES_MULTIPLE = 32              # every shipped preset is a multiple of 32


def scale_to_megapixels(w, h, mp, multiple=RES_MULTIPLE):
    """Resize (w, h) to hit `mp` megapixels, keeping the aspect ratio.

    Constant-area square root, snapped to `multiple` on both axes -- which is what
    H3's patchified latent grid needs. Snapping moves the real area off the request
    slightly, so callers report the ACHIEVED value: a readout of what was asked for
    hides what was produced. mp <= 0 means "leave the preset alone"."""
    if not mp or mp <= 0 or w <= 0 or h <= 0:
        return int(w), int(h)
    multiple = max(1, int(multiple))
    scale = math.sqrt((float(mp) * MP_UNIT) / float(w * h))
    nw = max(multiple, int(round(w * scale / multiple)) * multiple)
    nh = max(multiple, int(round(h * scale / multiple)) * multiple)
    return nw, nh


# --- prompt parsing + auto time distribution -------------------------------
def split_paragraphs(text, delimiter):
    raw = text.replace("\r\n", "\n").strip()
    if not raw:
        return []
    raw = re.sub(r"(?m)^\s*" + re.escape(delimiter) + r"\s*$", "\n\n", raw)
    return [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]


# Widgets added after the node's original 36-widget layout. Kept LAST in
# INPUT_TYPES so a workflow saved before they existed still maps its stored values
# onto the right widgets (ComfyUI matches them by position, not by name).
# APPEND to this tuple when adding a widget; never insert into the middle.
ADDED_WIDGETS = (
    "beat_split", "per_beat_length",
    "watermark_text", "watermark_position", "watermark_size", "watermark_opacity",
    "watermark_margin", "intro_text", "intro_position", "intro_seconds",
    "intro_fade", "intro_size", "overlay_font", "overlay_stroke",
    "ref_mode", "ref_image_size", "ref_noise_aug", "auto_props", "prevent_nudity",
    "exposed_terms", "anatomy_guard", "lock_restraints",
)

NL = "\n"
# Lines that CONFIGURE a beat rather than being one. They attach to the beat that
# follows them, so a line-split never turns "wardrobe: ..." into its own shot.
DIRECTIVE_KEYS = ("wardrobe", "seconds", "duration", "exit", "enter",
                  "overall_soundscape", "non_diegetic_music", "soundscape", "music")


def is_directive_line(line):
    return bool(re.match(r"\s*(" + "|".join(DIRECTIVE_KEYS) + r")\s*:", line or "", re.I))


def expand_beats(paras, mode="auto"):
    """Turn the prompt's beat PARAGRAPHS into the final beat list. Returns
    (beats, note).

    Beats are separated by a BLANK line (or a '##' line). That is unambiguous, but
    it is also the single easiest thing to get wrong in a textarea: six beats typed
    on six consecutive lines are one paragraph, so they render as ONE shot with six
    actions crammed into it -- which reads as characters moving at triple speed, not
    as a splitting problem.

    mode:
      'auto'      -- blank lines first; any paragraph still holding more than one
                     content line is then split one beat per line, and says so.
      'each line' -- every content line is its own beat. Same result as 'auto'; kept
                     so the intent can be stated explicitly.

    There is deliberately no strict blank-lines-only mode any more. It was the ONE
    setting that could silently lose beats: six beats typed as two blocks of three
    rendered as two shots, with no note to say why, because the split note is only
    written when a paragraph is actually split. Nothing else on the node can change
    the beat count, so removing that option removes the whole failure class. A
    workflow that still stores 'blank line' falls through to 'auto' below.

    Directive lines ('wardrobe:', 'seconds:', 'exit:' ...) are never beats of their
    own: they attach to the next content line, or to the previous beat if they
    trail the paragraph."""
    # Any unrecognized mode means AUTO, never "do nothing". An earlier version fell
    # through an if/elif with no else and silently DROPPED every multi-line paragraph
    # -- six beats arrived as two shots with four beats simply gone. A stale value on
    # this widget (including 'blank line' from a workflow saved before it was removed)
    # is enough to trigger it, so the safe branch has to be the default.
    if mode not in ("auto", "each line"):
        mode = "auto"
    out, split_from = [], 0
    for p in paras:
        lines = [ln for ln in (p or "").splitlines() if ln.strip()]
        content = [ln for ln in lines if not is_directive_line(ln)]
        if len(content) <= 1:
            out.append(p)
        else:
            split_from += 1
            pending = []
            for ln in lines:
                if is_directive_line(ln):
                    # Hold it for the NEXT content line: a directive reads as a header
                    # for the beat it introduces ("wardrobe: -= jacket" / "she shrugs
                    # it off"). Only if nothing follows does it fall back to the beat
                    # above, handled after the loop.
                    pending.append(ln)
                    continue
                out.append(NL.join(pending + [ln]))
                pending = []
            if pending:                                     # directives with no beat after them
                if out:
                    out[-1] = out[-1] + NL + NL.join(pending)
                else:
                    out.append(NL.join(pending))
    note = ""
    if split_from and mode == "auto":
        note = (f"{split_from} paragraph(s) held several lines and were split one beat per LINE "
                f"-> {len(out)} beats. Separate beats with a BLANK line (or a '##' line) to control "
                f"this yourself")
    return out, note


def _garment_side(side):
    """Is `side` one garment, for the purpose of splitting an 'A and B' entry?

    Qualifies when its head noun is a garment the zone tables know, or when it is
    long enough to be a described item rather than a bare adjective. The zone
    vocabulary is looked up at call time because it is defined further down."""
    words = side.split()
    if not words:
        return False
    vocab = _ZONE_LOWER | _ZONE_UPPER | _ZONE_BOTH
    return _item_head(side) in vocab or len(words) >= 2


def _split_conjoined(item):
    """Split 'a black jacket and blue jeans' into two tracked garments.

    A sheet entry joined by 'and' was kept as ONE item, and _item_name() truncates
    at 'and' -- so "small white t-shirt and shiny white lace thong" became the item
    "small white t-shirt" with head `t-shirt`, and the thong was not tracked at all.
    "pulls down the thong" then matched nothing, the removal silently did not fire,
    and the whole compound string -- thong included -- was re-stamped into the
    character's parenthetical on every later shot. The prompt kept saying she was
    wearing it, so the model kept putting it back.

    Only split when BOTH sides read as garments and at least one is a garment the
    zone tables recognise. That last condition is what keeps colour pairs and
    ordinary attributes intact: "black and white dress" has 'black' on the left,
    which is neither a known garment nor two words, and "blonde hair and blue eyes"
    has no garment on either side. Not splitting is always safe -- the left-hand
    garment stays tracked through _item_name()'s truncation either way -- so this
    is purely additive: it can only start tracking a garment that was invisible."""
    parts = [p.strip(" .;") for p in re.split(r"\s+and\s+", item or "", flags=re.I)]
    parts = [p for p in parts if p]
    if len(parts) < 2 or not all(_garment_side(p) for p in parts):
        return [item]
    vocab = _ZONE_LOWER | _ZONE_UPPER | _ZONE_BOTH
    if not any(_item_head(p) in vocab for p in parts):
        return [item]
    return parts


def _split_items(s):
    """Split a description into attribute items on commas AND sentence ends.

    A sheet written naturally ends clauses with a period -- "wearing a black t-shirt
    and jeans. Mouth closed." -- and treating that as ONE item drags a whole sentence
    into the inline parenthetical, which then reads as its own statement about a
    person rather than an attribute of the pronoun. Splitting on '.' as well keeps
    each item a short attribute.

    Garments joined by 'and' are then separated so each is independently removable;
    see _split_conjoined()."""
    out = []
    for i in re.split(r"[,.;]", s or ""):
        i = i.strip(" .;")
        if i:
            out += _split_conjoined(i)
    return out


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


# Words that start a DETAIL trailing off a garment rather than continuing its name:
# "red leather jacket WITH silver zippers", "boots WITH steel buckles".
# NOTE: "down" is deliberately absent -- it is a material ("a puffy down jacket"),
# not a position, and cutting there would leave "puffy" as the garment name.
_ITEM_DETAIL = re.compile(
    r"\b(?:with|without|featuring|showing|bearing|in|on|over|under|across|that|which|"
    r"and|plus|sporting|carrying|covered|around|about|at|through|along|behind|"
    r"beneath|beside|near)\b")


def _item_name(item):
    """The part of a wardrobe item that NAMES the garment, without its detail.

    "red leather jacket with a white circular chest patch" -> "red leather jacket".
    Used both to match a removal and to refer to the garment in generated prose:
    the detail belongs in the description that is stamped every shot, not in a
    sentence whose only job is to say the thing came off."""
    il = (item or "").strip()
    cut = _ITEM_DETAIL.search(il.lower())
    name = il[:cut.start()].strip(" ,") if cut and cut.start() > 0 else il
    # Drop a leading determiner: sheets are written both ways ("a diaper", "diaper"),
    # and the generated prose supplies its own article -- "the a diaper underneath".
    name = re.sub(r"^(?:a|an|the|her|his|their|its|my|your)\s+", "", name, flags=re.I)
    return name or il


def _item_head(item):
    """The garment's own head noun, ignoring any detail trailing off it.

    Taking the LAST word was wrong the moment an item carried detail: the head of
    "red leather jacket with silver zippers" came out as `zippers`, of "bomber
    jacket with a white logo on the chest" as `chest`. "takes off her red jacket"
    then matched nothing and the removal SILENTLY did not fire -- the garment stayed
    on the sheet and got re-stamped into every later shot. Detailed wardrobe entries
    are normal (logos, zippers, torn knees), so the head has to be read from the
    part of the phrase that names the garment."""
    words = re.findall(r"[a-z\-]+", _item_name(item).lower())
    if not words:
        words = re.findall(r"[a-z\-]+", (item or "").lower())
    return words[-1] if words else ""


def _item_mentioned(item, window):
    """Does `window` refer to this wardrobe item? Matches the whole phrase, or the
    item's head noun, tolerant of singular/plural ('boots' vs 'boot') -- the strict
    exact-substring test missed 'takes off her boots' when the sheet said 'boots'
    and vice versa. Ignores generic colour/size adjectives so 'red jacket' is still
    matched by 'her jacket', and trailing detail so 'red jacket with silver zippers'
    is still matched by 'her jacket'."""
    w = window.lower()
    il = item.lower().strip()
    if not il:
        return False
    if il in w:
        return True
    head = _item_head(il)
    if not head:
        return False
    for form in {head, head.rstrip("s"), head + "s", head + "es"}:
        if form and re.search(r"\b" + re.escape(form) + r"\b", w):
            return True
    return False


# Where a removal verb's OBJECT ends. Anything past one of these belongs to a
# different phrase -- what was revealed, where the garment was put, what happened
# next -- and must not be treated as something that also came off.
_OBJECT_STOP = re.compile(
    r"[,.;:!?]"
    r"|\b(?:revealing|reveals?|showing|shows?|exposing|exposes?|leaving|leaves?|wearing|"
    r"wears?|underneath|beneath|under|over|on|onto|in|into|to|from|at|by|beside|near|"
    r"next|behind|against|while|as|before|after|then|toward|towards)\b"
    r"|\b(?:puts?|pulls?|slips?|throws?|zips?|buttons?|laces?)\s+(?:on|into)\b")


def _removal_object_spans(text, m):
    """(forward, backward) -- the two places a removal verb's object can sit.

    FORWARD is the normal case ("takes off her red jacket"). It runs from the verb
    to the first phrase boundary and INCLUDES the matched cue, because the put-away
    patterns ("hangs her jacket on a hook") carry the garment inside the match. It
    deliberately does not extend past a boundary: a fixed ~68-character window used
    to sweep up any garment sitting near the removal, so "drops it on the bench next
    to her boots" removed the boots and "over her black tank top" removed the top.

    BACKWARD covers the phrasings that put the garment FIRST -- "her jacket slips
    off her shoulders", "her dress falls to the ground", "the jacket is off now".
    It is tried only when the forward span turns up nothing, and it stops at the
    previous clause boundary so it reaches the subject and no further."""
    tail = text[m.end():m.end() + 60]
    cut = _OBJECT_STOP.search(tail)
    forward = m.group(0) + (tail[:cut.start()] if cut else tail)
    head = text[max(0, m.start() - 45):m.start()]
    bcut = None
    for b in re.finditer(r"[,.;:!?]|\b(?:and|then|while|as|before|after)\b", head):
        bcut = b
    backward = head[bcut.end():] if bcut else head
    return forward, backward


# Items that STAY ON until something explicitly says otherwise.
#
# A garment coming off by itself is usually what the prose meant. A restraint is
# not: it is a plot state that the scene establishes and only the scene ends. Left
# to the ordinary removal detector these came off far too easily, and often by
# accident -- "steps out of her jacket and the chain falls away" removed the ankle
# chain as a side effect of a beat about a jacket, because the removal window
# reaches any tracked item near the cue.
#
# Deliberately narrow. Only nouns that are unambiguously a restraint: `chain`,
# `collar`, `strap` and `belt` are all left OUT, because they are jewellery, a
# shirt part, a dress part and a garment at least as often. A compound like
# "ankle chain" or "chain restraint" is caught by its qualifier instead.
_RESTRAINT_HEADS = {
    "handcuff", "handcuffs", "cuff", "cuffs", "shackle", "shackles",
    "manacle", "manacles", "fetter", "fetters", "iron", "irons",
    "restraint", "restraints", "binding", "bindings", "bond", "bonds",
    "gag", "blindfold", "hood", "muzzle", "harness", "leash", "hobble",
    "strait-jacket", "straitjacket", "spreader", "zip-tie", "ziptie",
}
# ...and the qualifiers that turn an ambiguous noun into a restraint. Body parts
# and binding participles ONLY.
#
# Materials are deliberately absent. "leather" would make a leather belt a
# restraint, "steel" a steel watch strap. And a word may not appear in BOTH lists:
# `chain` and `rope` were in each at first, so they qualified themselves and a bare
# "gold chain" came out a restraint.
_RESTRAINT_QUALIFIER = re.compile(
    r"\b(?:ankle|wrist|leg|arm|thumb|neck|waist|"
    r"chained|shackled|cuffed|bound|tied|locked|padlocked|restraining)\b", re.I)
_RESTRAINT_NOUN = re.compile(
    r"\b(?:chain|chains|rope|ropes|cord|cords|tie|ties|strap|straps|collar|collars|"
    r"band|bands|belt|belts|cuff|cuffs)\b", re.I)


def is_restraint(item):
    """True when this wardrobe item is a physical restraint rather than clothing.

    Either an unambiguous restraint noun on its own ("handcuffs", "shackles"), or
    an ambiguous one carrying a restraint qualifier ("ankle chain", "leather wrist
    straps"). A bare "chain" or "collar" is NOT a restraint -- it is jewellery or a
    shirt part far more often, and a false positive here means a garment that can
    never be taken off."""
    name = _item_name(item or "").lower()
    if not name:
        return False
    if _item_head(item) in _RESTRAINT_HEADS:
        return True
    return bool(_RESTRAINT_NOUN.search(name) and _RESTRAINT_QUALIFIER.search(name))


# What a restraint DOES, once it is on. Keeping the item in the wardrobe list only
# says it exists; nothing there says the body cannot move freely, so H3 renders a
# cuffed character walking with their arms swinging. The restraint is present and
# doing nothing -- which reads as it having broken.
#
# Stated as a POSITIVE physical state, the same as the mouth state and the limb
# count. "cannot move her arms" is a negation and a weak cue; "her wrists stay
# together in front of her" describes a pose the model can actually render.
#
# Keyed by the body region the restraint binds, so two wrist restraints produce one
# clause rather than two competing ones.
_RESTRAINT_EFFECT = {
    "wrists": "the wrists stay bound close together, the arms moving as one and never "
              "swinging apart",
    "ankles": "the ankles stay bound close together, steps short and shuffling, the legs "
              "never striding apart",
    "mouth":  "the mouth stays covered and the jaw still",
    "eyes":   "the eyes stay covered, the head turning toward sound rather than sight",
    "body":   "the body stays held by the restraint, movement limited and tethered",
}
# Which region each restraint binds. Checked against the item's full name, so
# "ankle chain" and "leg irons" both reach `ankles`.
_RESTRAINT_REGION = (
    ("mouth",  re.compile(r"\b(?:gag|gagged|muzzle|muzzled)\b", re.I)),
    ("eyes",   re.compile(r"\b(?:blindfold|blindfolded|hood|hooded)\b", re.I)),
    ("ankles", re.compile(r"\b(?:ankle|ankles|leg|legs|hobble|feet|foot)\b", re.I)),
    ("wrists", re.compile(r"\b(?:handcuff|handcuffs|cuff|cuffs|wrist|wrists|manacle|"
                          r"manacles|thumb|arm|arms)\b", re.I)),
)


def restraint_regions(items):
    """Body regions currently bound, for everything this person is wearing.

    A restraint with no region of its own ("shackles", "fetters", "restraints")
    falls back to `body`, which says movement is limited without claiming to know
    which limb. Guessing a specific limb there would be worse than saying less."""
    out = []
    for it in items or []:
        if not is_restraint(it):
            continue
        name = _item_name(it)
        for region, rx in _RESTRAINT_REGION:
            if rx.search(name):
                if region not in out:
                    out.append(region)
                break
        else:
            if "body" not in out:
                out.append("body")
    return out


def restraint_clause(active, body, lock_restraints=True):
    """State what each restrained person's body cannot do, for the people in shot.

    Only for someone actually referenced in the beat -- describing a bound body in
    a shot that person is not in would summon them into it, which is the same
    failure the mouth state and the limb count are both gated against."""
    if not lock_restraints:
        return ""
    bits = []
    for name, items in (active or {}).items():
        regions = restraint_regions(items)
        if not regions:
            continue
        if name and not person_in_shot(body, name, active):
            continue
        subj = _subject_term(name, active) if name else "the subject"
        # `their` rather than a repeated name: naming someone twice in a shot is
        # what renders them twice.
        effects = "; ".join(_RESTRAINT_EFFECT[r] for r in regions)
        bits.append(f"{subj} is physically restrained -- {effects}")
    if not bits:
        return ""
    return " " + ". ".join(b[0].upper() + b[1:] for b in bits) + "."


def auto_wardrobe_removals(active, body, lock_restraints=True):
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
    if not body:
        return active
    # Quoted speech is an INSTRUCTION, not an action. 'Mom says: "take off your
    # thong"' used to strip the garment in the shot where it is merely asked for,
    # one shot before the character does it -- so the shot that performs the
    # removal no longer knew the garment was on, and the removal never got its
    # direction clause. Negation made it worse: "do not take off your jacket"
    # removed the jacket. Detect removals from narration only; the beat that
    # actually stages it is unquoted.
    text = " " + _mask_quotes(body)[0].lower() + " "
    active = {k: list(v) for k, v in active.items()}
    names = [n for n in active if n]
    pron_map = _pron_map(active)
    single = len(names) == 1

    remove_cue = re.compile(
        # verb ... off / out of / aside / away   (covers "takes off", "steps out of")
        r"\b(?:takes?|took|taken|taking|pulls?|pulled|peels?|peeled|strips?|stripped|"
        r"slips?|slipped|shrugs?|shrugged|tears?|tore|yanks?|yanked|casts?|kicks?|"
        r"throws?|threw|tosses|tossed|hangs?|hung|drops?|dropped|sets?|set|puts?|put|"
        # ...and the ones you get OUT OF rather than take off
        r"steps?|stepped|stepping|wriggles?|wriggled|wiggles?|wiggled|squirms?|squirmed|"
        r"slides?|slid|sliding|climbs?|climbed|works?|worked|eases?|eased|shakes?|shook)\b"
        r"[\w\s\']{0,20}?\b(?:off|out of|aside|away|down)\b"
        # the garment itself is the subject: "her dress falls to the ground",
        # "the jacket pools at her feet". Matched backward to the subject.
        r"|\b(?:falls?|fell|falling|slides?|slid|slips?|slipped|drops?|dropped|"
        r"pools?|pooled|tumbles?|tumbled|crumples?|crumpled)\s+"
        r"(?:to|onto|on to|down|off|open|away|around|at)\b"
        # "lets her jacket fall", "lets it drop"
        r"|\blets?\b[\w\s\']{0,20}?\b(?:fall|falls|drop|drops|slide|slides|slip|slips)\b"
        # standalone removal verbs
        r"|\b(?:removes?|removed|removing|sheds?|shed|shedding|discards?|discarded|"
        r"ditch(?:es|ed)?|doffs?|doffed|unbuttons?|unzips?|unzipped|unbuckles?|"
        r"undoes|undid|undone|unlaces?|unlaced|unhooks?|unhooked|unfastens?|unfastened|"
        r"unclasps?|unclasped|unsnaps?|unsnapped|unties?|untied|unwraps?|unwrapped|"
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
        forward, backward = _removal_object_spans(text, m)
        tgt = nearest_subject(m.start())
        # Try the verb's forward object first; only if that names nothing tracked do
        # we look BACK to the subject, which is where "her jacket falls to the
        # ground" and "her jacket slips off her shoulders" put the garment.
        for window in (forward, backward):
            if not window.strip():
                continue
            hit = False
            for name in ([tgt] if tgt else list(active.keys())):
                for it in list(active.get(name, [])):
                    if it.strip().lower() in _PRO:
                        continue
                    # A restraint is a plot state, not a garment. Prose never takes
                    # one off; only an explicit 'wardrobe: Name -= handcuffs' does.
                    if lock_restraints and is_restraint(it):
                        continue
                    if _item_mentioned(it, window):
                        active[name] = [x for x in active[name] if x != it]
                        hit = True
            if hit:
                break
    return active




def _scrub_removed(text, removed):
    """Delete phrases for removed garments from PERSISTENT text (the anchor), so a
    garment written into the anchor prose -- e.g. 'A woman in a red flight jacket'
    -- can't re-apply itself on every shot after the character takes it off. Removes
    the item phrase plus a leading connector ('in a', 'wearing a', 'with a') and
    tidies the leftover punctuation. Case-insensitive; leaves everything else alone."""
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


# Pronoun by grammatical case: (subject, object, possessive).
_PRON_CASES = {"she": ("she", "her", "her"),
               "he": ("he", "him", "his"),
               "they": ("they", "them", "their")}
# A name right after one of these is an OBJECT ("walks over to Dan"), so it takes the
# object form. Anything else mid-sentence is treated as an object too, since that is
# where a bare name usually lands ("hands Dan a wrench", "asks Dan").
_OBJECT_PREPS = ("to", "with", "at", "for", "from", "toward", "towards", "behind",
                 "beside", "near", "of", "on", "onto", "into", "over", "under", "past",
                 "by", "about", "around", "beneath", "against", "alongside", "opposite",
                 "between", "upon", "across", "after", "before", "beyond", "through")
# ...and after one of these (or a sentence end) it is a SUBJECT ("and Dan takes it").
_SUBJECT_LEADS = ("and", "then", "but", "so", "as", "while", "when", "until", "because",
                  "if", "though", "although", "where", "who")


def _mask_quotes(text):
    """Hide double-quoted spans behind placeholders so a rewrite cannot touch the
    spoken words. Returns (masked_text, spans)."""
    spans = []

    def grab(m):
        spans.append(m.group(0))
        return "\x00%d\x00" % (len(spans) - 1)

    return re.sub(r'["“][^"”]*["”]', grab, text), spans


def _unmask_quotes(text, spans):
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)


def dedupe_person_mentions(body, active):
    """Replace the SECOND and later mentions of a tracked person's name inside one
    beat with the right pronoun.

    Naming one person twice in a shot is the single most reliable way to make
    text-to-video render them twice -- "Kristy finds Dan ... she walks over to Dan"
    puts two Dans in frame. Binding the description once (compose_persistent) fixes
    the description, not the name itself, so the bare repeat still duplicates.

    Only fires where the result is unambiguous:
      * the person's pronoun must be known (declared in their sheet, or a gender word
        in their description) -- an undeclared person is left exactly as written;
      * no OTHER person in the shot may share that pronoun, or 'he' could not be
        traced back to the right one;
      * words inside double quotes are never touched -- a name in a spoken line is
        dialogue ("Kristy, over here"), not a second reference to stage.
    The FIRST mention always survives, so the description still has a name to bind to
    and the reader can still tell who the shot is about."""
    if not body:
        return body
    present = [n for n in active if n and active[n]]
    if not present:
        return body
    by_pron = {}
    for n in present:
        p = _pronoun_of(active[n])
        if p:
            by_pron.setdefault(p, []).append(n)

    masked, spans = _mask_quotes(body)
    for n in present:
        p = _pronoun_of(active[n])
        if not p or len(by_pron.get(p, [])) != 1:
            continue                        # undeclared pronoun, or two people share it
        subj, obj, poss = _PRON_CASES[p]
        hits = list(re.finditer(r"\b" + re.escape(n) + r"(?:'s|’s)?\b", masked, re.I))
        if len(hits) < 2:
            continue
        for m in reversed(hits[1:]):        # right to left, so earlier offsets stay valid
            token = m.group(0)
            raw_before = masked[:m.start()]
            before = raw_before.rstrip()
            prev = re.search(r"([A-Za-z']+)\s*$", before)
            prev = prev.group(1).lower() if prev else ""
            if token.endswith("s") and ("'" in token or "’" in token):
                rep = poss
            elif prev in _OBJECT_PREPS:
                rep = obj
            elif (not before) or before[-1] in ".!?;:" or prev in _SUBJECT_LEADS:
                rep = subj
            else:
                rep = obj
            # capitalize only at a real sentence/line start
            if (not before) or before[-1] in ".!?" or raw_before.rstrip(" \t").endswith("\n"):
                rep = rep.capitalize()
            masked = masked[:m.start()] + rep + masked[m.end():]
    return _unmask_quotes(masked, spans)


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
    count_prefix = ""          # set when the count clause is front-loaded (LoRA runs)
    departed = set(departed or ())
    # A character who has LEFT the scene is never described again -- not even if a
    # later pronoun could resolve to them. This is what stops an exited character
    # being silently re-summoned into a later shot.
    active = {k: v for k, v in active.items() if k not in departed}
    # Collapse repeat NAME mentions to pronouns before anything is measured or bound:
    # naming one person twice in a shot renders them twice, and the refs below must be
    # computed against the text that will actually be emitted.
    body = dedupe_person_mentions(body, active)
    named = [n for n in active if n and active[n]]
    unnamed = active.get("", [])
    anchor_id = _scrub_removed(anchor_id, removed)
    # Keep tracked people OUT of the always-on anchor: the beat binds them inline,
    # so leaving them here too introduces each character twice per shot.
    anchor_id = _strip_people_from_anchor(anchor_id, active)
    # An UNNAMED sheet is emitted as a bare list in front of the beat, so it has to
    # be closed off as its own sentence. Without the period it ran straight into the
    # action -- "...red jacket, blue jeans, black boots Kristy walks around the
    # garage" -- where "black boots Kristy" reads as one noun phrase. A NAMED sheet
    # never had this problem: it binds as a parenthetical at the person's first
    # mention instead of being prepended.
    unnamed_txt = ", ".join(unnamed) if unnamed else ""
    if unnamed_txt and unnamed_txt[-1] not in ".!?":
        unnamed_txt += "."
    prefix_bits = [x for x in (anchor_id, unnamed_txt) if x]

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

        # Nobody bound individually, but the beat addresses the cast in the plural:
        # bind everyone with a roll-call in FRONT of the beat rather than rewriting
        # the sentence. Prepending keeps the author's prose exactly as written --
        # 'they' can mean a subset, and expanding it in place would assert a cast
        # list the author did not write.
        roll_call = ""
        if not refs and len(names) > 1 and _PLURAL_CAST.search(body):
            bits = []
            for n in names:
                desc = ", ".join(_clean_items(active[n], n, drop_mouth_state=speaking))
                bits.append(f"{n} ({desc})" if desc else n)
            roll_call = ((", ".join(bits[:-1]) + " and " + bits[-1])
                         + (" are both in this shot." if len(bits) == 2
                            else " are all in this shot."))
            refs = {n: 0 for n in names}     # for the subject count below

        if refs:
            # inject from rightmost position first so earlier indices stay valid
            if not roll_call:
                for n in sorted(refs, key=lambda k: refs[k], reverse=True):
                    desc = ", ".join(_clean_items(active[n], n, drop_mouth_state=speaking))
                    if desc:
                        pos = refs[n]
                        body = body[:pos] + f" ({desc})" + body[pos:]
            else:
                prefix_bits.append(roll_call)
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
    kept, wardrobe = [], None
    for ln in body.split("\n"):
        if re.match(r"\s*wardrobe\s*:", ln, re.I):
            wardrobe = ln.split(":", 1)[1].strip()
        else:
            kept.append(ln)
    return "\n".join(kept).strip(), wardrobe


# --- anchor hazards ---------------------------------------------------------
# The anchor is stamped into EVERY shot, so anything in it has to be true of every
# shot. Four kinds of thing are not, and each fails in its own way.
_ANCHOR_PERSON = re.compile(
    r"\b(skin|pores?|complexion|freckles?|stubble|face|facial|eyes?|lips?|mouth|"
    r"hairs?|figure|portrait|subject|person|people|man|woman|men|women|girl|boy|"
    r"he|she|him|her|his|hers|they|them|their)\b", re.I)
_ANCHOR_APPARATUS = re.compile(
    r"\b(camera|camcorder|lens|sensor|tripod|gimbal|steadicam|dolly|crane|drone|"
    r"handheld|hand-held|iphone|phone|gopro|dslr|webcam|filming|filmed|crew|"
    r"operator|documentary|selfie|pov|point of view|shot on|taken with)\b", re.I)
_ANCHOR_FRAMING = re.compile(
    r"\b(medium shot|close-?ups?|wide shot|long shot|full shot|two shot|"
    r"over the shoulder|low angle|high angle|aerial|overhead shot|establishing shot)\b", re.I)


def anchor_warnings(anchor):
    """Things in the anchor that will misfire because it repeats on every shot.

    Pure text, no model. Each of these has cost a real render: face words put a face
    in an empty establishing frame, apparatus words render the equipment, framing
    pins every shot to one size, and clothing here is immutable so a removal can
    never stick."""
    a = (anchor or "").strip()
    if not a:
        return []
    out = []
    def found(rx):
        return sorted({m.group(0).lower() for m in rx.finditer(a)})
    p = found(_ANCHOR_PERSON)
    if p:
        out.append(f"person/face words in the anchor ({', '.join(p)}) -- the anchor is stamped on "
                   f"EVERY shot, so these arrive in shots with nobody in them and can render a "
                   f"face in an empty frame; move them to character_memory, which is only emitted "
                   f"where that person appears")
    q = found(_ANCHOR_APPARATUS)
    if q:
        out.append(f"camera/apparatus words in the anchor ({', '.join(q)}) -- naming the equipment "
                   f"can render the equipment, or someone holding it; describe the IMAGE instead "
                   f"('shallow depth of field' rather than '35mm lens', 'fine grain' rather than "
                   f"'sensor grain')")
    f = found(_ANCHOR_FRAMING)
    if f:
        out.append(f"framing in the anchor ({', '.join(f)}) -- this pins every shot to that size; "
                   f"put framing in the beats so it can change shot to shot")
    garments = sorted({w.lower() for w in re.findall(r"[A-Za-z][\w\-]*", a)
                       if garment_zones(w)})
    if garments:
        out.append(f"clothing in the anchor ({', '.join(garments)}) -- the anchor is immutable, so "
                   f"it re-applies the garment on every shot and a removal cannot stick; put "
                   f"clothing in character_memory, the only channel that can change mid-chain")
    return out


def anchor_contributes_nothing(anchor, char_memory=""):
    """True when the paragraph about to be consumed as the identity anchor would add
    NOTHING to any shot -- i.e. taking it as the anchor silently DELETES it.

    The anchor is stamped into every shot, so _strip_people_from_anchor removes any
    sentence that names a tracked character (otherwise that character is introduced
    twice per shot and the model renders them twice). A first paragraph that is
    *itself* an action beat about a tracked person -- "Kristy walks around in a garage
    looking for engine parts." -- is therefore stripped to nothing: the user loses that
    shot AND the only scene text they wrote, with just a mild note to say so.

    Returns False whenever the paragraph carries something real: a 'wardrobe:' line (it
    seeds the wardrobe channel), or prose that survives the strip. So a normal
    identity/scene anchor is never touched, and with no character_memory nothing is
    tracked, nothing is stripped, and this cannot fire."""
    anchor_id, anchor_wardrobe = extract_wardrobe((anchor or "").strip())
    if anchor_wardrobe:                       # seeds the wardrobe channel -> it matters
        return False
    if not anchor_id.strip():
        return False
    active = parse_wardrobe((char_memory or "").strip())
    return not _strip_people_from_anchor(anchor_id, active).strip(" .,")


# A sentence that STAGES something: a name or pronoun subject followed by a verb
# ("Kristy walks", "She finds", "Dan answers"). An anchor is scene and style -- noun
# phrases and lists ("An open 4 bay car garage.", "natural lighting, flat lighting") --
# and does not match.
_ACTION_SENT = re.compile(r"^\s*(?:He|She|They|[A-Z][a-z]+)"
                          r"(?:\s+and\s+(?:[A-Z][a-z]+|he|she|they))?"
                          r"\s+[a-z]+(?:s|ed|ing)\b")


def anchor_is_action_beat(anchor, later_paras=()):
    """True when the paragraph about to be consumed as the anchor is plainly a BEAT.

    anchor_contributes_nothing() only catches this when the character is tracked in
    character_memory -- with no sheet, nothing is tracked, nothing is stripped, and an
    action paragraph sails through to become the anchor. That is the common case: a
    prompt written as three beats, no character sheet, renders as two shots with the
    first beat demoted to a header stamped on the other two.

    Fires only when EVERY sentence stages an action AND the subject recurs later, so a
    mixed paragraph ("Kristy stands by the plane. A cinematic hangar, warm light.")
    keeps its scene text and stays an anchor, and a style list never matches at all. A
    pronoun subject is accepted on its own -- 'She walks in.' cannot be scene text."""
    body, wardrobe = extract_wardrobe((anchor or "").strip())
    if wardrobe:                              # seeds the wardrobe channel -> it matters
        return False
    body = body.strip()
    if not body:
        return False
    sents = [s for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    if not sents or not all(_ACTION_SENT.match(s) or has_speech(s) for s in sents):
        return False
    m = re.match(r"\s*([A-Za-z]+)", body)
    if not m:
        return False
    subj = m.group(1).lower()
    if subj in ("he", "she", "they"):
        return True
    # A NAME is only a beat subject if the prompt goes on using it. This is what keeps
    # a one-word style lead ("Cinematic lighting, warm tones.") from reading as an
    # action: 'cinematic' never comes back as a subject in the beats.
    later = " ".join(later_paras or ()).lower()
    return bool(re.search(r"\b" + re.escape(subj) + r"\b", later))


# --- props: objects that must survive the shot boundary ---------------------
# Nouns that are never a prop worth carrying: parts of the frame, parts of a body,
# and abstractions. Binding these would produce "the same ground from the previous
# shot", which is noise at best.
_NOT_A_PROP = {
    "ground", "air", "sky", "floor", "ceiling", "background", "foreground", "distance",
    "camera", "frame", "shot", "scene", "screen", "view", "angle", "light", "lighting",
    "shadow", "sun", "moment", "time", "day", "night", "morning", "evening", "way",
    # anatomy -- a body part is never a prop to carry between shots. Without these,
    # "reveals a nipple" was tracked as an object and a later mention got the full
    # continuity treatment: "exactly one nipple in this shot, the same nipple from
    # the previous shot, in the same place".
    "head", "face", "eyes", "eye", "hand", "hands", "arm", "arms", "leg", "legs",
    "body", "hair", "mouth", "jaw", "lips", "shoulder", "shoulders", "chest", "torso",
    "waist", "hip", "hips", "knee", "knees", "foot", "feet", "ankle", "wrist", "neck",
    "throat", "stomach", "belly", "navel", "skin", "thigh", "thighs", "breast",
    "breasts", "nipple", "nipples", "genitals", "genitalia", "penis", "vagina",
    "vulva", "groin", "crotch", "buttock", "buttocks", "backside", "bottom",
    # frame-relative words and bare determiners/pronouns, which are never objects
    "back", "front", "side", "top", "edge", "middle", "end", "left", "right",
    "centre", "center", "the", "a", "an", "it", "this", "that", "these", "those",
    "they", "them", "her", "his", "him", "its", "one", "other", "another", "something",
}
# Where a prop's NAME stops and its circumstance begins: "a van PARKED in the bay",
# "a barn WITH a red roof". Same idea as _ITEM_DETAIL for garments.
_PROP_TAIL = re.compile(
    r"\b(?:parked|standing|sitting|leaning|lying|resting|waiting|stopped|covered|"
    r"filled|loaded|painted|marked|in|on|at|by|near|beside|behind|under|over|with|"
    r"and|that|which|down|across|toward|towards|from|"
    # a VERB ends the noun phrase too: "a bare breast CATCHES the light" was being
    # read four words deep and keyed on the trailing "the".
    r"is|are|was|were|has|have|had|catches|catch|presses|press|hangs|hang|rests|"
    r"sits|sit|stands|stand|lies|lie|falls|fall|moves|move|shows|show|reveals|reveal|"
    r"appears|appear|becomes|become|looks|look|seems|seem|glints|glows|shines)\b")


def introduced_props(text):
    """{noun: phrase} for objects this text introduces INDEFINITELY -- "a white van"
    -> {"van": "white van"}. These are the things a later shot can refer back to."""
    out = {}
    body = _PICTURE_TAG.sub(" ", text or "")
    # Scan from each article WITHOUT consuming what follows it: a greedy match over
    # "a van and a truck" would swallow the truck's own article and lose it.
    for m in re.finditer(r"\b(?:a|an)\s+", body):
        ahead = body[m.end():]
        ahead = re.split(r"[,.;:!?]", ahead)[0]
        words_ahead = re.findall(r"[A-Za-z][\w\-]*", ahead)[:4]
        phrase = " ".join(words_ahead)
        cut = _PROP_TAIL.search(phrase.lower())
        if cut and cut.start() > 0:
            phrase = phrase[:cut.start()].strip()
        elif cut:
            continue                      # starts with a tail word -- not a name
        words = phrase.split()
        if not words:
            continue
        noun = words[-1].lower()
        if noun in _NOT_A_PROP or len(noun) < 3:
            continue
        out.setdefault(noun, phrase.strip())
    return out


def bind_props(body, props):
    """Rewrite the FIRST definite reference to each carried prop so it names the
    object instead of assuming one. Returns (body, [nouns bound]).

    "the van" in a later shot has no antecedent -- each shot is its own generation,
    and nothing in that prompt describes a van. The model invents one, which is how
    a second van appears in the frame while the first is still there. Naming it and
    asserting it is the SAME one is what binds the two shots together."""
    if not props or not body:
        return body, []
    masked, spans = _mask_quotes(body)
    bound = []
    for noun, phrase in props.items():
        pat = re.compile(r"\bthe\s+(" + re.escape(noun) + r")\b", re.I)
        m = pat.search(masked)
        if not m:
            continue
        masked = masked[:m.start()] + f"the same {phrase}" + masked[m.end():]
        bound.append(noun)
    return _unmask_quotes(masked, spans), bound


def dedupe_prop_mentions(body, nouns):
    """Collapse repeat mentions of the SAME object within one beat.

    "drives a van ... gets out of the van ... walks to the back of the van" is three
    vans in one prompt, and repetition is how a video model ends up rendering three.
    The node already does exactly this for people -- naming someone twice in a beat
    is the most reliable way to get two of them -- and an object is no different.

    Only fires when a single carried object is in play, so "it" cannot be ambiguous,
    and never inside quoted speech."""
    if not body or len(nouns) != 1:
        return body, 0
    noun = nouns[0]
    masked, spans = _mask_quotes(body)
    hits = list(re.finditer(r"\b(?:the|that|this)\s+" + re.escape(noun) + r"\b", masked, re.I))
    if len(hits) < 2:
        return body, 0
    # keep the first definite mention; the rest become pronouns
    for m in reversed(hits[1:]):
        masked = masked[:m.start()] + "it" + masked[m.end():]
    return _unmask_quotes(masked, spans), len(hits) - 1


def repeated_props(body, own):
    """Objects a beat introduces and then refers back to definitely, in the SAME
    beat -- the case a cross-shot carry never sees."""
    out = []
    for noun in own:
        if re.search(r"\b(?:the|that|this)\s+" + re.escape(noun) + r"\b", body or "", re.I):
            out.append(noun)
    return out


def prop_count_clause(nouns):
    """Positive count for objects, matching the subject-count guard's shape."""
    if not nouns:
        return ""
    bits = [f"exactly one {n} in this shot" for n in nouns]
    s = ", ".join(bits)
    return " " + s[0].upper() + s[1:] + "."


def prop_continuity_clause(bound, props):
    """One short sentence pinning a carried prop to the previous shot's object.

    Re-describing it is not enough on its own: "a white van" in shot 1 and "a white
    van" in shot 2 are two white vans. The clause states identity with the previous
    shot, and states the COUNT positively.

    It deliberately does NOT say "no second van". Naming the unwanted thing is how
    "she is no longer wearing the red jacket" put the jacket back on: to a video
    model a mention is a presence cue and a negation is weak, so "no second van"
    puts a second van in the text. The subject-count guard already had this right
    for people -- it leads with "Exactly one person" -- and props follow the same
    shape."""
    if not bound:
        return ""
    bits = []
    for noun in bound:
        phrase = props.get(noun, noun)
        bits.append(f"exactly one {noun} in this shot, the same {phrase} from the previous shot, "
                    f"in the same place")
    s = "; ".join(bits)
    return " " + s[0].upper() + s[1:] + "."


# Cues that the quoted words are PRINTED in the scene rather than spoken aloud.
# Kept to things that are unambiguously written surfaces or reading/marking verbs.
_WRITTEN_CUE = re.compile(
    r"\b(?:reads?|reading|marked|labell?ed|titled|captioned|headlined|written|"
    r"printed|engraved|stamped|embroidered|scrawled|painted|spells?|spelled|"
    r"signs?|posters?|banner|placard|plaque|sticker|label|headline|"
    r"caption|graffiti|screen|display|monitor|billboard|tattoo|note|letter|"
    r"envelope|book|page|menu|ticket|receipt|badge|nameplate)\b", re.I)

# ...and the verbs that mean someone said it out loud.
_SPOKEN_CUE = re.compile(
    r"\b(?:says?|said|saying|asks?|asked|asking|replies|replied|answers?|answered|"
    r"shouts?|shouted|yells?|yelled|calls?|called|whispers?|whispered|murmurs?|"
    r"murmured|mutters?|muttered|adds?|added|tells?|told|cries|cried|barks?|"
    r"barked|snaps?|snapped|breathes?|offers?|insists?|repeats?|begins?|continues?|"
    r"declares?|announces?|responds?|responded|urges?|warns?|pleads?|laughs?)\b",
    re.I)


def has_speech(body):
    """True only if a beat contains ACTUAL scripted speech -- double-quoted words
    or an explicit <d>...</d> tag. Bare speech VERBS ('calls out', 'tells', 'says'
    with no quoted line) deliberately do NOT count: unscripted speech is exactly
    what H3 fills with gibberish, so those beats get silenced too. If you want
    someone to speak, quote the line: She says, "Ready for departure."
    Apostrophes/single quotes never count (they'd false-fire on "she's").

    WRITTEN text in quotes is not speech. A sign, a label, a headline -- 'reads the
    sign marked "EXIT"' -- used to make the whole shot count as dialogue, so it got
    neither the lips-closed clause nor the no-voice soundscape and the characters
    stood there opening their mouths. Nothing in the beat was ever spoken."""
    if not body:
        return False
    if re.search(r"<d>.*?</d>", body, re.S):
        return True
    for m in re.finditer(r'["\u201c\u201d].+?["\u201c\u201d]', body, re.S):
        # Look at what introduces this quote, and let the NEAREST cue decide.
        # "Mara reads the sign, then says 'we go left'" is speech: 'says' sits
        # closer to the quote than 'reads' does. Comparing presence rather than
        # position got that backwards.
        lead = body[max(0, m.start() - 60):m.start()].lower()
        written = [x.end() for x in _WRITTEN_CUE.finditer(lead)]
        spoken = [x.end() for x in _SPOKEN_CUE.finditer(lead)]
        if written and (not spoken or written[-1] > spoken[-1]):
            continue                       # printed in the scene, nobody said it
        return True
    return False


# Silence is stated as a described PHYSICAL STATE, which H3 follows far better than
# an appended negation -- but NOT in the leading position it used to occupy.
#
# Opening every silent shot with "mouth closed, lips together, jaw still" put face
# anatomy in the first tokens the model reads, and a distilled LoRA fixes
# composition in its first step or two. The result was a face rendered at the start
# of shots -- including scenery shots with nobody in them at all, which is where it
# was unmistakable. The mouth state now follows the action instead of preceding it,
# and it is skipped entirely on a shot with no people, where "everyone is silent
# with their mouth closed" describes nobody and only invites a face.
#
# The audio half of the babble fix does not depend on this: the no-voice soundscape
# line and mute_nonspeech_audio both still apply.
# Extra limbs, duplicated hands, a third arm. There is exactly one lever for this:
# H3 is CFG-FREE at cfg 1, and comfy/samplers.py:610 sets uncond_ = None at that
# scale, so the negative prompt is NEVER EVALUATED. "extra limbs, deformed hands"
# in a negative does nothing at all on this model.
#
# So it has to be said POSITIVELY, in the same shape as the subject count clause
# that already stops duplicate people. Stating a number gives the model a target;
# negating one just puts the word in the prompt, and on this model a mention is a
# presence cue -- which is why this says what the body HAS and never what it lacks.
#
# Placed per-shot, never in the anchor. Anchor body words are what burned a face
# into the opening frames of every shot, found by bisection, and limb words there
# would carry the same risk on every shot including scenery.
ANATOMY_STATE = (" Each person has one head, two arms, two hands with five fingers each, "
                 "and two legs, in correct human proportion.")

LIPS_CLOSED_STATE = (" Everyone in this shot is silent with their mouth closed and lips together, "
                     "jaw still, not talking.")
LIPS_CLOSED_TAIL = " No speech, no dialogue, no lip movement, no mouth movement."

# The lips-closed clause constrains the PICTURE only. H3 generates audio from its
# own fields, and an ABSENT `overall_soundscape:` leaves that branch unconditioned
# -- which is exactly when it invents speech-like babble under a silent shot. So a
# silenced shot always gets a soundscape line, and it says no voices outright.
NO_VOICE_SOUNDSCAPE = ("ambient background sound and room tone only, no voices, no speech, "
                       "no talking, no whispering, no singing, no vocal sounds")
NO_VOICE_CLAUSE = ", no voices, no speech, no talking, no vocal sounds"


# A beat that refers to the cast only in the PLURAL ("they face each other") used
# to bind nobody: _resolve_subject() maps a pronoun to ONE person, and 'they' with
# two people resolves to neither. The shot then silently described no one -- losing
# both characters' descriptions and, after a removal, their exposure markers, so a
# stripped character quietly went back to being unmarked.
#
# Bare 'them'/'their' are deliberately NOT here. They refer to objects at least as
# often as to people -- "she steps out of them" is a garment, "light floods through
# them" is a pair of doors -- and this fires only when nobody was bound by name or
# singular pronoun, which is exactly the scenery-beat case that must stay empty.
_PLURAL_CAST = re.compile(
    r"\b(?:they|themselves|both|each other|one another|"
    r"the two of them|all of them)\b", re.I)


def person_referenced(body, name, active):
    """Is this person actually in the beat -- by name, or by a pronoun that resolves
    to them? Used to keep a wardrobe statement out of a shot they aren't in: saying
    "she is no longer wearing the jacket" in a shot about someone else SUMMONS her
    into it, which is the duplication failure the whole builder exists to avoid."""
    low = (body or "").lower()
    if name and re.search(r"\b" + re.escape(name.lower()) + r"\b", low):
        return True
    names = [n for n in active if n]
    pron_map = _pron_map(active)
    single = len(names) == 1
    for m in re.finditer(r"\b(she|he|they|her|him|them|his|their)\b", low):
        if _resolve_subject(m.group(1), names, pron_map, single) == name:
            return True
    return False


def person_in_shot(body, name, active, departed=()):
    """Is this person IN this shot -- by name, by a resolvable pronoun, or as part
    of a cast addressed in the plural?

    The single presence test. person_referenced() alone is not it: it resolves a
    pronoun to ONE person, so 'they' and 'both of them' answer False for everybody,
    and any clause gated on it silently skips a beat that binds the whole cast.

    That exact bug has now been written twice -- once in the mouth-state gate
    (a plural beat got no lips-closed clause, so those shots babbled) and again in
    the restraint clause (a plural beat dropped the physical constraint, so the
    restraints appeared to break). Both are gated on this function now, so a third
    caller cannot rediscover it."""
    if person_referenced(body, name, active):
        return True
    present = [n for n in (active or {}) if n and n not in (departed or ())]
    return len(present) > 1 and bool(_PLURAL_CAST.search(body or ""))


def _subject_term(name, active):
    """How to refer to a person in a generated clause: their declared PRONOUN when
    it identifies them uniquely, otherwise their name. Pronoun-first is the rule the
    whole builder follows -- a bare name is a fresh introduction, and introducing
    someone twice in a shot is what makes the model render them twice."""
    pron = _pronoun_of(active.get(name, []))
    if pron:
        holders = [n for n in active if n and _pronoun_of(active[n]) == pron]
        if len(holders) == 1:
            return pron
    return name


# Which part of the body a garment covers. Only two zones matter here, and the
# question each answers is strictly "is this part of the body still COVERED?" -- a
# removal that empties a zone is the one that renders as nudity.
#
# That question, not "is this clothing?", decides what belongs in these sets. A
# garter belt, stockings, hold-ups, socks, gloves and a scarf are all clothing and
# all deliberately absent: they leave the zone bare, so counting them as cover
# would SUPPRESS the exposure warning exactly when it is needed. Anything not
# listed maps to no zone, which is the safe default -- it can never mask a warning,
# it can only fail to volunteer an under-layer.
_ZONE_LOWER = {
    # trousers and their families
    "pants", "trousers", "jeans", "denims", "slacks", "chinos", "khakis", "cargos",
    "cords", "corduroys", "joggers", "sweatpants", "sweats", "trackpants",
    "breeches", "jodhpurs", "capris", "culottes", "bloomers", "harems",
    # skirts
    "skirt", "miniskirt", "midiskirt", "maxiskirt", "kilt", "sarong", "lungi", "dhoti",
    # shorts
    "shorts", "boardshorts", "trunks", "speedos", "hotpants",
    # legwear that DOES cover the pelvis
    "leggings", "jeggings", "treggings", "tights", "pantyhose", "pantihose",
    # underwear and lingerie bottoms
    "briefs", "boxers", "boxershorts", "panties", "knickers", "underpants",
    "underwear", "undies", "drawers", "thong", "g-string", "gstring", "tanga",
    "boyshorts", "boyshort", "hipsters", "jockstrap", "loincloth", "bottoms",
    # nappies
    "diaper", "diapers", "nappy", "nappies", "pull-ups", "pullups", "pull-up",
}
_ZONE_UPPER = {
    # shirts and tops
    "shirt", "t-shirt", "tshirt", "tee", "top", "crop-top", "croptop", "tanktop",
    "tank", "blouse", "jersey", "polo", "henley", "turtleneck", "flannel", "smock",
    "tunic", "kurta", "halter", "bandeau", "tube-top",
    # knitwear and outerwear
    "sweater", "jumper", "hoodie", "sweatshirt", "pullover", "cardigan", "jacket",
    "coat", "blazer", "vest", "waistcoat", "gilet", "anorak", "parka", "windbreaker",
    "bomber", "peacoat", "trenchcoat", "raincoat", "poncho", "shawl", "cape", "cloak",
    "thermal", "thermals",
    # lingerie and underlayers for the torso
    "bra", "brassiere", "bralette", "bustier", "corset", "basque", "camisole", "cami",
    "singlet", "undershirt", "brasiere",
}
_ZONE_BOTH = {
    # one-piece garments covering torso AND pelvis
    "dress", "gown", "frock", "pinafore", "jumpsuit", "romper", "playsuit", "catsuit",
    "unitard", "leotard", "bodysuit", "bodystocking", "onesie", "overalls",
    "dungarees", "coveralls", "boilersuit", "snowsuit", "wetsuit", "drysuit",
    "robe", "bathrobe", "housecoat", "dressinggown", "kimono", "kaftan", "caftan",
    "abaya", "sari", "saree", "toga",
    # sleepwear and lingerie one-pieces
    "nightgown", "nightie", "nightdress", "negligee", "babydoll", "teddy", "slip",
    "chemise", "pyjamas", "pajamas", "pjs",
    # swimwear counted as a set
    "swimsuit", "swimming-costume", "maillot", "bikini", "tankini", "monokini",
    # a suit is jacket + trousers
    "suit", "tracksuit",
}


def garment_zones(item):
    """The body zones a garment covers: {'lower'}, {'upper'}, both, or empty.

    Empty means it is not body covering at all (hat, boots, a scar, hair colour), so
    removing it can never expose anything."""
    head = _item_head(item)
    for form in (head, head.rstrip("s"), head + "s"):
        if form in _ZONE_BOTH:
            return {"upper", "lower"}
        if form in _ZONE_LOWER:
            return {"lower"}
        if form in _ZONE_UPPER:
            return {"upper"}
    return set()


# Once a zone has been stripped, its state has to be STATED in every later shot.
# Deleting the garment is only a silence, and a video model's default prior is a
# clothed person, so silence gets them dressed again a shot or two later -- the same
# reason "no longer wearing the red jacket" was not enough on its own. These read as
# a physical description, not as a negation, and they live in the wardrobe channel
# so they persist and clear exactly like a garment.
_BARE_MARK = {"lower": "bare below the waist", "upper": "bare chest"}
# The upper-zone default has to follow the person. "bare chest" on a woman is both
# odd phrasing and a weak cue -- it describes a male torso, and H3 renders roughly
# what the words describe. The lower default stays neutral: it is a position on the
# body, not an anatomy, and naming anatomy there is exactly what exposed_terms is
# for. A person with no declared pronoun keeps the neutral wording.
_BARE_MARK_BY_PRON = {
    "she": {"lower": "bare below the waist", "upper": "bare breasts"},
    "he": {"lower": "bare below the waist", "upper": "bare chest"},
}


# A character can START bare rather than becoming bare. Until this existed the
# exposure marker only fired on a REMOVAL, so someone naked from shot 1 was never
# marked and exposed_terms never applied to them.
#
# This has to be DECLARED, never inferred. Absence of clothing in a sheet means the
# author did not enumerate it -- "Jon = he, 35, bald" is an ordinary
# under-specified sheet, not a naked man -- and inferring nudity from a short sheet
# would put it in scenes nobody asked for. Only these explicit tokens count.
_DECLARED_BARE = {
    "nude": {"lower", "upper"}, "naked": {"lower", "upper"},
    "fully nude": {"lower", "upper"}, "fully naked": {"lower", "upper"},
    "completely nude": {"lower", "upper"}, "completely naked": {"lower", "upper"},
    "undressed": {"lower", "upper"}, "unclothed": {"lower", "upper"},
    "bottomless": {"lower"}, "topless": {"upper"},
    "waist down nude": {"lower"}, "nude below the waist": {"lower"},
    "bare chested": {"upper"}, "barechested": {"upper"}, "bare-chested": {"upper"},
}


def declared_bare_zones(items):
    """Zones a person's sheet says are bare from the outset, and the tokens saying so.

    Returns (zones, tokens). The tokens are returned so the caller can drop them
    from the description: they are replaced by the exposure marker, and leaving both
    in would state the same fact twice in one parenthetical."""
    zones, tokens = set(), []
    for it in items or []:
        key = _item_name(it).strip().lower()
        z = _DECLARED_BARE.get(key)
        if z is None:
            z = _DECLARED_BARE.get(re.sub(r"[^a-z ]", "", (it or "").strip().lower()))
        if z:
            zones |= z
            tokens.append(it)
    return zones, tokens


def parse_exposed_terms(text):
    """Per-person text for a stripped zone: {key: {zone: phrase}}.

    Written like the wardrobe sheet, so there is one syntax to learn:

        she = visible vulva, mvagina
        he  = visible penis, mpenis
        Mara upper = bare breasts

    A key is a PRONOUN (applies to everyone declaring it) or a NAME (which wins
    over the pronoun). Without a trailing 'upper' the entry describes the lower
    zone. Empty means the generic wording is used."""
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if not key or not val:
            continue
        zone = "lower"
        parts = key.split()
        if len(parts) > 1 and parts[-1].lower() in ("upper", "lower"):
            zone = parts[-1].lower()
            key = " ".join(parts[:-1])
        out.setdefault(_norm_name(key).lower(), {})[zone] = val
    return out


def exposed_mark(zone, name, items, terms):
    """The phrase to stamp for a stripped `zone` on this person.

    Name beats pronoun, pronoun beats the person's own default, which beats the
    neutral one -- so one setup covers a whole cast, and a single character can
    still be given their own wording."""
    pron = _pronoun_of(items or [])
    if terms:
        by_name = terms.get((name or "").strip().lower(), {})
        if zone in by_name:
            return by_name[zone]
        if pron:
            by_pron = terms.get(pron.lower(), {})
            if zone in by_pron:
                return by_pron[zone]
    # Configuring only the lower zone is the common case ('she = vagina'), and the
    # upper zone still has to be worded for the right body rather than defaulting
    # to a male torso.
    return _BARE_MARK_BY_PRON.get(pron or "", _BARE_MARK)[zone]


def bare_state_items(items, stripped_zones, marks=None):
    """Markers to add / remove so a stripped zone keeps saying it is stripped.

    Returns (add, drop). A zone that something covers again -- because a garment was
    put back on -- drops its marker, which is what "unless requested" means."""
    marks = marks or dict(_BARE_MARK)
    # Every phrase this function could have stamped, so a marker is recognised for
    # removal even if the configured wording changed between runs -- including the
    # per-pronoun defaults, or a marker stamped as 'bare breasts' on one run could
    # not be dropped on the next.
    known = set(_BARE_MARK.values()) | set(marks.values())
    for d in _BARE_MARK_BY_PRON.values():
        known |= set(d.values())
    add, drop = [], []
    for zone in _BARE_MARK:
        mark = marks.get(zone, _BARE_MARK[zone])
        present = mark in items
        covered = bool(remaining_cover([i for i in items if i not in known], {zone}))
        if zone in stripped_zones and not covered and not present:
            add.append(mark)
        elif (covered or zone not in stripped_zones) and present:
            drop.append(mark)
    return add, drop


def remaining_cover(items, zones):
    """Items still worn that cover any of `zones` -- what is underneath."""
    return [i for i in items if garment_zones(i) & zones]


def _is_plural_garment(item):
    """Garments that take a plural verb: overalls, jeans, boots, gloves, shorts.
    A head noun ending in a DOUBLE s (dress, harness) is singular, which is what
    separates them from a real plural.

    Read from the garment's own head, not the item's last word: "red jacket with
    silver zippers" ends on a plural detail while the garment itself is singular,
    which produced "the red jacket ... ARE off and she is not wearing THEM"."""
    head = _item_head(item)
    return bool(head) and head.endswith("s") and not head.endswith("ss")


def takes_off_clause(pairs, active=None):
    """The DIRECTION of a removal, stated in the shot that performs it.

    A removal is the one wardrobe change with a failure mode of its own: the motion
    is symmetric. The same frames played backwards are a person putting the garment
    ON, and both readings satisfy "takes off her red jacket" equally well. The model
    picks whichever the rest of the conditioning supports -- and when the shot's own
    description still listed the garment as worn, backwards was the reading that
    matched. The removal rendered in reverse and the jacket came back.

    So the end state is stated explicitly, and the reverse is ruled out by name.
    Said ONCE, in the removal shot only: every later shot simply describes what the
    person is wearing now, and never names the garment again -- to a video model a
    mention is a presence cue, and a negation is a weak one, so "no longer wearing
    the red jacket" in the NEXT shot was itself enough to put it back on."""
    active = active or {}
    by = {}
    for name, item in pairs:
        item = (item or "").strip()
        if item and item not in by.setdefault(name or "", []):
            by[name or ""].append(item)
    bits, still = [], []
    for name, items in by.items():
        # Refer to the garment by NAME, not by its full sheet entry. The detail
        # (logo, zippers, torn knee) is already stamped in the description every
        # shot; repeating it twice inside a sentence that only has to say the thing
        # came off buries the instruction in 22 words of wardrobe.
        what = " and ".join(_item_name(i) for i in items)
        # "the navy overalls IS off" reads as a mistake to the encoder that has to
        # parse this. Garments like overalls/jeans/boots are grammatically plural,
        # as is any list of more than one.
        plural = len(items) > 1 or any(_is_plural_garment(i) for i in items)
        # `pron` is the OBJECT form ("takes them off"), `subj_pron` the SUBJECT form
        # ("they are off") -- the impersonal branch needs the latter.
        verb, pron, subj_pron = ("are", "them", "they") if plural else ("is", "it", "it")
        subj = _subject_term(name, active) if name else ""
        # Name the garment ONCE. It was named twice here and again in the beat prose,
        # which made it the most-referenced thing in the shot -- and a garment
        # referenced that often gets rendered as a prominent object, picked up and
        # handled by whoever is nearby. Same rule as people: repeat the reference,
        # get the thing repeated. Saying where it ENDS UP is what stops it lingering
        # in someone's hands.
        if subj:
            bits.append(f"{subj} takes the {what} off during this shot; by the last frame "
                        f"{subj_pron} {verb} off, dropped away out of frame, and {subj.lower()} is "
                        f"no longer wearing {pron}")
        else:
            bits.append(f"the {what} {'come' if plural else 'comes'} off during this shot; by the "
                        f"last frame {subj_pron} {verb} off and dropped away out of frame")
        # SAY WHAT IS STILL ON. The clause is five statements about clothing coming
        # off; without this, nothing in it says the body is still covered, and the
        # model completes the obvious continuation -- shorts worn UNDER trousers were
        # listed once in a distant parenthetical and simply not rendered. Naming the
        # under-layer here, in the same breath as the removal, is what keeps it on.
        zones = set()
        for i in items:
            zones |= garment_zones(i)
        under = remaining_cover(active.get(name, []), zones) if zones else []
        if under:
            worn = " and ".join(_item_name(u) for u in under)
            who = (_subject_term(name, active).lower() if name else "the character")
            still.append(f"the {worn} underneath {'stay' if len(under) > 1 or _is_plural_garment(under[0]) else 'stays'} "
                         f"on and {who} is still wearing {'them' if len(under) > 1 or _is_plural_garment(under[0]) else 'it'}")
    if not bits:
        return ""
    s = "; ".join(bits)
    if still:
        s += ". " + ("; ".join(still)).capitalize()
    # The anti-reverse instruction is the point of the clause, so it is not left
    # implicit in the end-state description. Worded without a pronoun so it needs no
    # agreement with whatever came off.
    return (s[0].upper() + s[1:]
            + ". The motion runs one way only: the clothing comes off and is never put back on, "
              "never re-worn, and the action never plays in reverse.")


# Words that can never be part of the garment phrase itself.
_GARMENT_LEAD = {"off", "out", "of", "aside", "away", "down", "up", "the", "a", "an",
                 "her", "his", "their", "its", "it", "them", "then"}
# Words that END a garment phrase: a conjunction, a new preposition, or a new
# article all start something that is no longer the garment.
_GARMENT_END = {"and", "or", "but", "then", "on", "onto", "over", "into", "in", "to",
                "from", "at", "by", "with", "under", "beside", "as", "while", "before",
                "after", "a", "an", "the", "she", "he", "they", "her", "his", "their"}
# A person noun is never part of a garment phrase -- scrubbing one deletes the
# CHARACTER from the anchor and leaves the clothing behind.
_PERSON_NOUN = {"woman", "women", "man", "men", "girl", "boy", "guy", "lady", "person",
                "people", "figure", "child", "kid", "teen", "teenager", "male", "female"}


def removed_phrase_items(body, anchor_id):
    """Garments named in a REMOVAL phrase in this beat that also appear in the
    anchor prose. Covers the case where the item was never in the wardrobe channel
    at all -- e.g. the anchor says 'a woman in a red flight jacket' and the beat
    says 'she takes off her jacket'. Without this the anchor would keep re-applying
    it forever. Returns the anchor phrases to scrub.

    The phrase is read to its HEAD NOUN, not to the first word after the verb. The
    earlier version stopped at the first non-stop word, so "takes off her red
    jacket" yielded 'red' -- and matching 'red' with its preceding words in the
    anchor produced 'A woman in a red', which scrubbed the PERSON out of
    'A woman in a red jacket' and left 'jacket'. The garment survived, the
    character vanished, and clothing removal looked completely broken."""
    if not body or not anchor_id:
        return []
    verb = re.compile(r"\b(takes?|took|taking|pulls?|pulled|peels?|peeled|strips?|stripped|"
                      r"slips?|slipped|shrugs?|shrugged|removes?|removed|sheds?|shed|discards?|"
                      r"ditch(?:es|ed)?|doffs?|unbuttons?|unzips?)\b", re.I)
    out = []
    for m in verb.finditer(body):
        # Stop at punctuation: "shrugs off his overalls, a flight suit underneath"
        # must not drag the second clause into the garment.
        tail = re.split(r"[,.;:!?]", body[m.end():m.end() + 60])[0]
        words = re.findall(r"[A-Za-z][A-Za-z\-]*", tail)
        i = 0
        while i < len(words) and words[i].lower() in _GARMENT_LEAD:
            i += 1
        phrase = []
        while i < len(words) and words[i].lower() not in _GARMENT_END and len(phrase) < 4:
            phrase.append(words[i])
            i += 1
        if not phrase:
            continue
        head = phrase[-1]
        if head.lower() in _PERSON_NOUN:            # "takes off after the man" -- not clothing
            continue
        # Take the head noun with its adjectives out of the anchor, then trim any
        # leading word that belongs to the SENTENCE rather than to the garment.
        am = re.search(r"((?:[A-Za-z\-]+\s+){0,2}" + re.escape(head) + r")\b", anchor_id, re.I)
        if not am:
            continue
        toks = am.group(1).split()
        while len(toks) > 1 and toks[0].lower() in (_GARMENT_END | _PERSON_NOUN | _GARMENT_LEAD):
            toks.pop(0)
        if toks and toks[-1].lower() not in _PERSON_NOUN:
            out.append(" ".join(toks))
    return out


def extract_directive(body, key):
    """Pull a '<key>: ...' line out of a beat body. Returns (clean_body, value|None)."""
    kept, val = [], None
    for ln in body.split("\n"):
        if re.match(r"\s*" + key + r"\s*:", ln, re.I):
            val = ln.split(":", 1)[1].strip()
        else:
            kept.append(ln)
    return "\n".join(kept).strip(), val


# "walks out OF THE BARN" is emerging INTO the scene, not leaving it -- and a false
# exit is the expensive error: the character is stripped from every later shot and
# only an explicit 'enter:' brings them back. So "out of <somewhere>" is never an
# exit unless the somewhere is the frame itself.
_EMERGENCE_TAIL = re.compile(
    r"^\s+of\s+(?:the\s+|a\s+|an\s+|his\s+|her\s+|their\s+|its\s+)?"
    r"(?!frame\b|shot\b|view\b|screen\b|scene\b|camera\b|sight\b|there\b|here\b)\w+", re.I)


def _is_emergence(text, m):
    """True when this exit cue is really someone coming OUT OF a place into view.

    The cue match ends at "out"/"off", so what decides it is what FOLLOWS: "walks
    out | of the barn" is emergence, "walks out | of frame" is departure, and a
    bare "walks out" has nothing after it and stays an exit."""
    return bool(_EMERGENCE_TAIL.match(text[m.end():]))


# One list of departure phrasings for both readers of it: detect_exits(), which
# decides WHO left, and departed_phrase_people(), which scrubs their description
# out of the anchor. It was written out twice; the two copies had to agree or a
# character could be marked departed while the anchor kept describing them.
_EXIT_CUE = re.compile(
    r"\b(?:leaves?|left|leaving|exits?|exited|departs?|departed|"
    r"walks? (?:out|off|away)|walked (?:out|off|away)|steps? (?:out|off|away)|"
    r"stepped (?:out|off|away)|drives? (?:off|away)|drove (?:off|away)|"
    r"rides? (?:off|away)|runs? (?:out|off)|ran (?:out|off)|"
    r"disappears?|vanishes?|is gone|are gone|out of frame|off screen|off-screen)\b", re.I)


def detect_exits(body, active, departed):
    """Names of characters who LEAVE in this beat, so they don't reappear later.
    Matches an exit phrase ('leaves', 'walks out', 'exits', 'drives off', 'steps
    out of frame', 'is gone') attributed to the nearest preceding subject (name or
    resolvable pronoun). Gated on tracked people, so 'the plane leaves' -- not a
    tracked person -- departs nobody."""
    if not body:
        return []
    text = " " + body.lower() + " "
    names = [n for n in active if n and n not in departed]
    if not names:
        return []
    pron_map = _pron_map({k: v for k, v in active.items() if k not in departed})
    single = len(names) == 1

    subj_tokens = [re.escape(n.lower()) for n in names] + list(_PRO.keys())
    subj_re = re.compile(r"\b(" + "|".join(subj_tokens) + r")\b")

    out = []
    for m in _EXIT_CUE.finditer(text):
        if _is_emergence(text, m):
            continue
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
    if not body or not anchor_id:
        return []
    want = {"she": ("woman", "women", "girl", "lady", "female"),
            "he":  ("man", "men", "boy", "guy", "gentleman", "male")}
    out = []
    for m in _EXIT_CUE.finditer(body):
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


# A spoken line needs a beat of air before and after it inside the same shot --
# the mouth opens late and the last syllable must not land on the cut.
SPEECH_PAD_SEC = 1.0
# ...and a two-hander needs a hand-off between turns. Two people trading three
# lines is not the same screen time as one person saying all three back to back:
# the camera/mouth has to switch subject between each.
TURN_GAP_SEC = 0.5


# --- content-aware shot length ---------------------------------------------
# A beat's screen time is estimated from how many ACTIONS it stages, not from its
# word count. Word count measures how wordy you were; clause count measures how
# much has to happen.
#
# The estimate is deliberately biased SHORT, because the two errors are not
# symmetric. A shot that ends before the action finishes hands a mid-motion frame
# to the next shot, which is exactly what the handoff chain is built to continue.
# A shot that outlasts its action leaves the model seconds it was told nothing
# about, and the cheapest filler for a symmetric action (taking a jacket off, a
# door opening, sitting down) is to run it BACKWARDS -- which returns to the start
# state and makes the clip loopable. Too long is unrecoverable; too short is not.
BEAT_BASE_SEC = 2.0          # setup/settle time every shot needs regardless of content
SECONDS_PER_ACTION = 2.5     # screen time for one staged action clause
MIN_CONTENT_FRAMES = 73      # ~3.0s: the shortest shot that can hold one action
# Clause separators: a new coordinated verb phrase starts a new action.
_CLAUSE_SPLIT = (r"(?:[.!?;]+|,?\s+(?:and then|then|and|before|after|while|as|until)\s+"
                 r"|,\s+(?=[a-z]+ing\b))")


def action_clauses(beat):
    """How many distinct staged actions a beat contains.

    "takes off her red jacket and drops it on the workbench" is two; "walks the
    length of the garage, checking every bench, then stops at the far wall" is
    three. Quoted speech is excluded -- that time is counted by dialogue_seconds."""
    body, _ = extract_wardrobe((beat or "").strip())
    body = re.sub(r'["“][^"”]*["”]', " ", body)
    body = " ".join(ln for ln in body.splitlines() if not is_directive_line(ln))
    parts = [p.strip() for p in re.split(_CLAUSE_SPLIT, body) if p and p.strip()]
    # A fragment of one word is a leftover ("it", "her"), not an action of its own.
    return sum(1 for p in parts if len(p.split()) >= 2)


def estimate_beat_seconds(beat):
    """Screen time this beat needs, from its own content. 0.0 when it has none.

    Action and dialogue OVERLAP rather than add -- people talk while they move --
    so the estimate is the larger of the two, not their sum."""
    n = action_clauses(beat)
    action = (BEAT_BASE_SEC + SECONDS_PER_ACTION * n) if n else 0.0
    return max(action, dialogue_seconds(beat))


def dialogue_spans(beat):
    """Word count of each double-quoted span in a beat, in order. Length of the
    returned list is the number of speaking TURNS -- the multi-character case."""
    body, _ = extract_wardrobe((beat or "").strip())
    return [len(q.split()) for q in re.findall(r'["\u201c]([^"\u201d]+)["\u201d]', body) if q.split()]


def dialogue_words(beat):
    """Words inside double quotes in a beat -- the only speech H3 actually renders."""
    return sum(dialogue_spans(beat))


def dialogue_seconds(beat, pad=True):
    """Screen time this beat's dialogue needs, 0.0 when the beat has none.

    Counts every turn, so a two-character exchange is sized from the WHOLE
    exchange plus a gap between turns -- not from the longest single line.
    `pad` controls only the head/tail air; turn gaps are always counted because
    they are time the shot genuinely has to contain."""
    spans = dialogue_spans(beat)
    if not spans:
        return 0.0
    return (sum(spans) / WORDS_PER_SEC
            + TURN_GAP_SEC * (len(spans) - 1)
            + (SPEECH_PAD_SEC if pad else 0.0))


def beat_seconds_directive(beat):
    """Explicit per-beat length: a 'seconds: 8' (or 'duration: 8') line in the beat.
    Returns the float, or None when the beat doesn't set one."""
    for key in ("seconds", "duration"):
        _, val = extract_directive((beat or ""), key)
        if val:
            m = re.search(r"([0-9]*\.?[0-9]+)", val)
            if m:
                try:
                    v = float(m.group(1))
                except ValueError:
                    continue
                if v > 0:
                    return v
    return None


def plan_beat_frames(beats, fps, budget, per_beat=True):
    """Per-beat shot lengths in frames. Returns (lengths, notes).

    `budget` is the CEILING -- the VRAM budget, or a forced shot_seconds already
    clamped to it. Per-beat sizing can only ever make a shot shorter than that
    ceiling, never longer. Priority per beat:

      1. an explicit 'seconds: N' line in the beat -- always honored, down to
         H3's real 5-frame minimum, because you stated a duration outright;
      2. its own content -- action clauses and quoted dialogue (see
         estimate_beat_seconds), floored at MIN_CONTENT_FRAMES so a shot always
         has room for one action;
      3. with per_beat off, the ceiling, exactly as before.

    Why estimate at all, when action prose has no *reliable* duration? Because the
    alternative is not "no guess" -- it is "guess the maximum", which is what giving
    every beat the ceiling does. A 3-second action in a 12-second shot leaves nine
    seconds the model was told nothing about, and it fills them by repeating or
    REVERSING the action. Leaning short costs an unfinished action that the next
    shot continues from the handoff frame; leaning long costs a jacket that takes
    itself off and puts itself back on."""
    beats = beats if beats else [""]
    # MIN_SHOT_FRAMES is the floor of the *VRAM budget* -- the shortest shot the node
    # falls back to when it has to guess with no information at all. It must not raise
    # a length that came from you or from the beat's own content: `max(floor, ...)`
    # silently turned every request below ~5.2s into 124f, so 1s/2s/3s/4s all rendered
    # identically and both the widget and the `seconds:` directive looked broken.
    cap = max(5, int(budget))
    content_floor = align_frame_count(MIN_CONTENT_FRAMES)
    out, notes = [], []
    fps = max(1, int(fps))
    for i, b in enumerate(beats, 1):
        want, src, floor = beat_seconds_directive(b), "seconds:", 5
        snap = align_frame_count            # a stated length is never rounded DOWN
        if want is None:
            want = estimate_beat_seconds(b) if per_beat else 0.0
            src, floor, snap = "content", content_floor, align_frame_count_nearest
        if want <= 0:                       # no signal -> the ceiling
            out.append(cap)
            continue
        n = min(cap, max(floor, snap(int(round(want * fps)))))
        out.append(n)
        if n != cap:
            notes.append(f"shot {i}: {n}f (~{n / fps:.1f}s, from {src})")
    return out, notes


def pacing_warnings(beats, lengths, fps):
    """Beats whose content is far too thin for the length they were given.

    Pure arithmetic, no model involved: it cannot know that "walks across the
    tarmac" is 2s or 12s, but it can see 12 words sitting in a 12-second shot and
    say so BEFORE the render, instead of leaving you to discover it as an action
    that repeats or plays backwards."""
    out = []
    fps = max(1, int(fps))
    for i, (b, n) in enumerate(zip(beats or [], lengths or []), 1):
        if beat_seconds_directive(b):        # you stated it; not the node's business
            continue
        need = estimate_beat_seconds(b)
        have = n / fps
        if need and have > need * 1.8 and have - need >= 3.0:
            out.append(f"shot {i}: ~{need:.1f}s of content in a {have:.1f}s shot "
                       f"({action_clauses(b)} action(s), {dialogue_words(b)} spoken words)")
    return out


def dialogue_fit_warnings(beats, seconds_per_shot):
    """Flag beats whose quoted dialogue is unlikely to fit the shot length.

    The VRAM budget caps SHOTS (never resolution, since rendering below native
    softens the frame). That is the right trade for picture quality, but it is blind
    to dialogue: a line written for a 10s shot gets cut off mid-sentence in a 7s one.
    Audio cannot span the handoff either -- each shot generates its own -- so a
    truncated line is simply lost, not continued.

    seconds_per_shot takes a single value or a per-shot list (per-beat sizing).
    Returns a list like ["shot 3: ~6.4s of dialogue in a 5.2s shot"] so the user can
    shorten the line, or choose a lower resolution tier to buy the duration back."""
    out = []
    for i, b in enumerate(beats or [], 1):
        if isinstance(seconds_per_shot, (list, tuple)):
            if i > len(seconds_per_shot):
                break
            sec = seconds_per_shot[i - 1]
        else:
            sec = seconds_per_shot
        need = dialogue_seconds(b, pad=False)
        if not need:
            continue
        if need > sec * 0.92:      # leave a little room to breathe
            out.append(f"shot {i}: ~{need:.1f}s of dialogue in a {sec:.1f}s shot")
    return out


def continuity_warnings(gens):
    """Shots that describe NOBODY while people are still in the story.

    The chain hands each shot the previous one's last decoded frame. A scenery beat
    describes no one, so the frame it produces has no one in it -- and the next
    shot has to re-establish every character from an empty room. That is a cohesion
    break, and it is invisible in the prompt: the text of both shots is individually
    correct, which is why chains lose their people in the middle rather than
    degrading steadily.

    Only flagged when people appear on BOTH sides. A scenery beat that opens or
    closes a chain hands its frame to nobody, so it costs nothing."""
    if len(gens or []) < 3:
        return []
    # A bound person shows up as an inline parenthetical of real description.
    peopled = []
    for g in gens:
        body = (g or "").split("\n")[0]
        body = re.sub(r"^\[Generation \d+\]\s*", "", body)
        body = re.sub(r"^Exactly [a-z]+ (?:person|people)[^.]*\.\s*", "", body)
        peopled.append(bool(re.search(r"\([^)]{6,}\)", body)))
    out = []
    for i in range(1, len(peopled) - 1):
        if not peopled[i] and peopled[i - 1] and any(peopled[i + 1:]):
            out.append(
                f"shot {i + 1} describes nobody, between shots that do -- it hands shot "
                f"{i + 2} a frame with no people in it, so every character has to be "
                f"re-established from an empty room. Give it someone ('Dom watches from "
                f"the doorway'), or move it to the start or end of the chain")
    return out


def dialogue_filler_warnings(beats, seconds_per_shot):
    """Dialogue shots with far more time than their line, which H3 fills with speech.

    dialogue_fit_warnings covers the opposite error -- a line too long for its shot,
    which gets truncated. This is the one that produces BABBLE: a two-second line in
    a ten-second shot leaves eight seconds of audio the model was told nothing
    about, and the audio branch keeps talking to fill them. mute_nonspeech_audio
    cannot help, because a shot with a scripted line is deliberately left audible.

    Same vacuum as an over-long action beat, one channel across."""
    out = []
    for i, b in enumerate(beats or [], 1):
        sec = (seconds_per_shot[i - 1] if isinstance(seconds_per_shot, (list, tuple))
               else seconds_per_shot) if not isinstance(seconds_per_shot, (list, tuple)) \
            or i <= len(seconds_per_shot) else None
        if sec is None:
            break
        spoken = dialogue_seconds(b, pad=False)
        if not spoken:
            continue
        gap = sec - spoken
        if gap >= 3.0 and sec > spoken * 2:
            out.append(f"shot {i}: {spoken:.1f}s of dialogue in a {sec:.1f}s shot -- {gap:.1f}s of "
                       f"unscripted audio the model will fill with more speech")
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
                           auto_silence_nonspeech=True, count_subjects=False, front_load=False,
                           notes_out=None, auto_props=True, prevent_nudity=True,
                           exposed_terms="", strip_out=None, anatomy_guard=False,
                           lock_restraints=True):
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
    props = {}                               # objects introduced so far -> their phrase
    stripped = {}                            # person -> body zones stripped so far
    exposed = parse_exposed_terms(exposed_terms)
    # Every person key that existed at any point, so an entry naming someone who is
    # only introduced later by a 'wardrobe: Name = ...' directive is not called
    # unmatched. Checked after the loop, once the full cast is known.
    seen_names = {k for k in active if k}
    blocks = []
    for gi, b in enumerate(beats, 1):
        body, wardrobe_change = extract_wardrobe((b or "").strip())
        body, _ = extract_directive(body, "seconds")               # shot length, not prose
        body, _ = extract_directive(body, "duration")              # ditto (alias)
        body, exit_directive = extract_directive(body, "exit")     # explicit 'exit: Jon'
        body, enter_directive = extract_directive(body, "enter")   # explicit 'enter: Jon' (undo)
        if enter_directive:
            for nm in _entries(enter_directive):
                departed.discard(_norm_name(nm))
        # Naming a departed character again is intent to have them BACK. Without
        # this they stayed departed, so the beat carried their bare NAME with no
        # description while everyone else kept theirs -- and the described character
        # absorbed the action. A PRONOUN still cannot re-summon anyone: "he waves"
        # after someone left is ambiguous, a name is not. Use 'exit: Name' again to
        # send them back out.
        if departed:
            named_here, _spans = _mask_quotes(body)
            for nm in list(departed):
                if nm and re.search(r"\b" + re.escape(nm) + r"\b", named_here, re.I):
                    departed.discard(nm)
        body = body or "continue the action, same subject"
        # Props introduced in an EARLIER beat: bind the first definite reference to
        # them, so "the van" in shot 2 means the van from shot 1 instead of an
        # invented one standing next to it. Garments are excluded -- they have their
        # own channel, and "the same red jacket" would fight a removal.
        worn_nouns = {_item_head(i) for v in active.values() for i in v}
        own_props = {n: p for n, p in introduced_props(body).items() if n not in worn_nouns}
        carried = {n: p for n, p in props.items()
                   if n not in worn_nouns and n not in own_props}
        body, bound_props = bind_props(body, carried) if auto_props else (body, [])
        # An object introduced AND referred back to inside one beat never reaches the
        # cross-shot carry -- but that is the case that duplicates hardest, because
        # the repetition is all in one prompt. Collapse the repeats the way repeated
        # NAMES are collapsed, and state the count.
        here_again = repeated_props(body, own_props) if auto_props else []
        if here_again:
            body, _ = dedupe_prop_mentions(body, here_again)
        off_now = []                         # (person, garment) coming off in THIS shot

        def _drop(before, after):
            """Record what `after` no longer has, for both removal paths."""
            for k, v in before.items():
                gone = [it for it in v if it not in after.get(k, [])]
                removed.extend(gone)
                off_now.extend((k, it) for it in gone)
            return after

        if wardrobe_change is not None:
            before = {k: list(v) for k, v in active.items()}
            # explicit: takes effect THIS shot
            active = _drop(before, apply_wardrobe_change(active, wardrobe_change))
        # Auto-removals are resolved BEFORE the shot is composed, so the garment is
        # already out of the person's description in the very shot that takes it off.
        #
        # It used to be deferred to the next shot, on the reasoning that the shot
        # SHOWING the removal should still show the garment. That produced a shot
        # whose description says "wearing a red jacket" while its verb says "takes off
        # her red jacket" -- and the cheapest way for the model to satisfy both is to
        # run the motion the OTHER way, ending with the jacket on. The video played
        # the removal in reverse.
        #
        # The start state does not need the description: for every shot after the
        # first it is pinned by the handoff keyframe, which shows the garment still
        # worn. So the keyframe carries the START state and the prompt carries the
        # END state, and the direction between them is stated outright below.
        if auto_wardrobe:
            before = {k: list(v) for k, v in active.items()}
            active = _drop(before, auto_wardrobe_removals(active, body, lock_restraints))
            # A garment that lives ONLY in the anchor prose (never in the wardrobe
            # channel): the removal phrase names it, so scrub it from the anchor or
            # the anchor re-applies it forever.
            anchor_gone = removed_phrase_items(body, anchor_id)
            # Same rule on the anchor side: a restraint named in the anchor prose is
            # not scrubbed by a removal phrase either, or it would vanish from every
            # later shot without anything having asked for it.
            if lock_restraints:
                anchor_gone = [it for it in anchor_gone if not is_restraint(it)]
            removed += anchor_gone
            # Voice the anchor-side removal only when the channel didn't already cover
            # it, or the same jacket is announced twice.
            if anchor_gone and not off_now:
                off_now += [("", it) for it in anchor_gone]
        # Record which zones this person has been stripped in, then keep the state
        # STATED in every later shot. Deleting the garment is only a silence, and a
        # video model's default is a clothed person -- so silence puts the clothes
        # back on a shot or two later. The marker clears by itself if a garment
        # covering that zone is put back on, which is the "unless requested" half.
        for nm, it in off_now:
            z = garment_zones(it)
            if z:
                stripped.setdefault(nm, set()).update(z)
        # A sheet can DECLARE a zone bare from the outset ('Jon = he, 35, nude'),
        # which is the start-naked case: there is no removal to trigger on, so
        # without this the marker never fires and exposed_terms never reaches them.
        # The token itself is swapped out for the marker so the fact is stated once.
        declared = {}
        for nm in list(active):
            zones, tokens = declared_bare_zones(active.get(nm, []))
            if zones:
                declared[nm] = zones
                stripped.setdefault(nm, set()).update(zones)
                for tok in tokens:
                    active[nm].remove(tok)
        for nm in list(active):
            marks = {z: exposed_mark(z, nm, active.get(nm, []), exposed)
                     for z in ("lower", "upper")}
            add, drop = bare_state_items(active.get(nm, []), stripped.get(nm, set()), marks)
            # prevent_nudity gates the ASSERTION, not the removal. Deleting a garment
            # only leaves the zone undescribed, and a video model's default prior is a
            # clothed person -- so it dresses them again. It is this marker that makes
            # the prompt SAY the body is bare, which is the thing that renders. The
            # garment still comes off either way; without the marker the model simply
            # covers what nobody described. `info` still reports the empty zone.
            #
            # Filling in exposed_terms IS the intent, so it overrides the guard for the
            # people it names. Requiring both switches was a footgun: the terms sat
            # there looking configured and did nothing.
            # Declaring nudity in the sheet is as explicit as filling in
            # exposed_terms, so it overrides the guard for that person the same way.
            if prevent_nudity and not exposed and nm not in declared:
                add = []
            stripped_here = any(n == nm for n, _ in off_now)
            for mark in add:
                active[nm].append(mark)
                # This shot newly bared a zone by REMOVING something. The NEXT shot
                # must not continue from its last frame: that frame is the removal in
                # progress, and a picture of the garment still being worn beats any
                # sentence saying it is off. A zone that was declared bare from the
                # start has no such frame -- nothing came off -- so it must NOT cost
                # the next shot its handoff.
                if stripped_here and strip_out is not None and gi not in strip_out:
                    strip_out.append(gi)
            for mark in drop:
                active[nm].remove(mark)
        persistent = compose_persistent(body, active, anchor_id, removed, departed, count_subjects,
                                        speaking=has_speech(body), front_load=front_load)
        # State the DIRECTION of the change, in the shot that performs it. Only for
        # people actually in this shot; an anchor-prose garment is stated
        # impersonally, so it summons nobody.
        speak_off = [(n, it) for n, it in off_now
                     if not n or person_referenced(body, n, active)]
        off_clause = takes_off_clause(speak_off, active)
        prop_clause = (prop_continuity_clause(bound_props, carried)
                       + prop_count_clause([n for n in here_again if n not in bound_props]))
        # A removal that leaves a body zone with NOTHING on it is the one the node
        # cannot write its way out of: there is no under-layer to name, so the model
        # renders bare skin. Say so before the render rather than after it.
        if notes_out is not None:
            for nm, it in off_now:
                zones = garment_zones(it)
                if zones and not remaining_cover(active.get(nm, []), zones):
                    who = nm or "the character"
                    notes_out.append(
                        f"shot {gi}: removing the {_item_name(it)} leaves {who} with nothing on the "
                        f"{'/'.join(sorted(zones))} body -- H3 will render bare skin there. Add an "
                        f"under-layer to character_memory (e.g. 'grey shorts') if that is not intended")
        if off_clause:
            persistent = persistent.rstrip(". ") + ". " + off_clause
        if prop_clause:
            persistent = persistent.rstrip(". ") + "." + prop_clause
        # Silence non-speech shots: a shot with no scripted dialogue gets an explicit
        # lips-closed / no-speech clause, so H3 doesn't animate a mouth or fill it with
        # gibberish before (or between) actual dialogue. Shots WITH quoted dialogue are
        # left alone so the speech renders.
        # Two different silences, and they are NOT the same condition:
        #   no_speech  -> the AUDIO constraint. A shot with no scripted line must not
        #                 be given an unconditioned audio branch, whether or not
        #                 anyone is on screen; an empty room still babbles.
        #   mouth_state-> the PICTURE constraint. Only meaningful when someone is
        #                 there to have a mouth. On a scenery beat it describes
        #                 nobody and can only invite a face into an empty frame.
        no_speech = bool(auto_silence_nonspeech and not has_speech(body))
        # Must agree with compose_persistent()'s binding, including the PLURAL case.
        # It did not: person_referenced() resolves a pronoun to one person, so
        # 'they'/'both of them' answered False for everyone, and a beat that the
        # roll-call had just described in full counted as having nobody in it. Those
        # shots got no mouth constraint at all -- two people on screen, nothing
        # saying their lips are closed -- which is exactly a shot that opens mouths
        # at random.
        present_names = [n for n in active if n and n not in departed]
        people_here = (bool(active.get(""))
                       or any(person_in_shot(body, n, active, departed)
                              for n in present_names))
        if no_speech and people_here:
            persistent = persistent.rstrip(". ") + "." + LIPS_CLOSED_STATE + LIPS_CLOSED_TAIL
        # Same gate as the mouth state, and for the same reason: describing a body
        # in a frame that has none can only invite one in.
        if anatomy_guard and people_here:
            persistent = persistent.rstrip(". ") + "." + ANATOMY_STATE
        # What the restraints DO. Placed after the anatomy state so the limb count
        # is established before the limits on it, and gated on the same presence
        # test -- a bound body described in a shot with nobody in it invites one.
        if lock_restraints and people_here:
            rc = restraint_clause(active, body, lock_restraints)
            if rc:
                persistent = persistent.rstrip(". ") + "." + rc
        silent_shot = no_speech
        block = f"[Generation {gi}] {persistent}".strip()
        # A silenced shot ALWAYS gets a soundscape line. Leaving the field out is
        # what let H3 improvise a voice track under a shot whose picture was already
        # told to keep its mouth shut -- the babble the lips-closed clause cannot
        # reach, because it only constrains the frames.
        if "soundscape:" not in block.lower():
            if gs:
                block += f"\noverall_soundscape: {gs}{NO_VOICE_CLAUSE if silent_shot else ''}"
            elif silent_shot:
                block += f"\noverall_soundscape: {NO_VOICE_SOUNDSCAPE}"
        # Music is OPT-IN: a blank field emits the spec's silence token N/A on every
        # shot, so H3 doesn't improvise a score. (Soundscape is NOT forced to N/A --
        # per the spec it takes N/A only when total silence is explicitly wanted, so a
        # blank soundscape still lets H3 provide ambient sound.)
        if "non_diegetic_music:" not in block.lower():
            block += f"\nnon_diegetic_music: {music if music else 'N/A'}"
        blocks.append(block.strip())
        # Exits stay DEFERRED, unlike removals: a character has to be visible in the
        # shot that shows them leaving, and the frame they leave in is the shot's own
        # subject -- there is no reverse-motion trap, because "walks out" ending with
        # them present would contradict the beat itself, not just a description.
        if exit_directive:
            for nm in _entries(exit_directive):
                departed.add(_norm_name(nm))
        departed.update(detect_exits(body, active, departed))
        # Plain-text case: a person described only in the anchor prose (never in the
        # character channel) can't be "departed" by name -- scrub their phrase from
        # the anchor instead, exactly as removed garments are scrubbed.
        removed += departed_phrase_people(body, anchor_id)
        # Anything this beat introduces indefinitely becomes referable later.
        for _n, _ph in introduced_props(body).items():
            props.setdefault(_n, _ph)
        seen_names |= {k for k in active if k}
    # An exposed_terms key that matches neither a character nor a usable pronoun
    # never fires, and it fails SILENTLY: the lookup falls through to the pronoun,
    # then to the default wording, so a typo'd name looks configured and does
    # nothing. Only the three canonical pronouns work as pronoun keys, because
    # _pronoun_of() normalizes to those -- an object form like 'her' is dead
    # config for the same reason and is worth the same warning.
    if exposed and notes_out is not None:
        known = {n.lower() for n in seen_names} | set(_PRO.values())
        for key in exposed:
            if key in known:
                continue
            hint = ""
            if key in _PRO:                       # 'her'/'him'/'his'/'them'...
                hint = f" -- use '{_PRO[key]}' for the pronoun form"
            elif seen_names:
                hint = " -- known characters: " + ", ".join(sorted(seen_names))
            # Echo the key as the user typed it: parse_exposed_terms() lowercases,
            # and quoting something back in different case than they wrote reads
            # like a different entry.
            m = re.search(r"^\s*(" + re.escape(key) + r")\b", exposed_terms or "",
                          re.I | re.M)
            notes_out.append(
                f"exposed_terms entry '{m.group(1) if m else key}' matches no character "
                f"and no pronoun, so it never applies{hint}")
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
    if resident_gb >= total_gb:
        # STREAMING REGIME. model_size() reports the whole checkpoint, but a checkpoint
        # larger than the card is never all resident: ComfyUI streams it, so the weight
        # figure is NOT what occupies VRAM and cannot be subtracted from capacity. Doing
        # that arithmetic anyway drove the budget deeply negative and floored every shot
        # to 124f/~5s on a card that was demonstrably not running out -- a 44.3GB MXFP8
        # build on a 15.9GB card sampled 243f at 768x768 without exceeding VRAM.
        #
        # There is no meaningful "capacity minus weights" here, so budget from the LIVE
        # free reading instead: it measures what is actually unoccupied right now, which
        # in this regime is the only number that means anything. Without a reading there
        # is nothing to go on, so fall back to the floor.
        if not free_gb or free_gb <= 0:
            return floor
        avail = max(0.0, free_gb * (1.0 - SPIKE_RESERVE) - headroom_gb)
    # avail <= 0 here means the weights FIT but the safety headroom eats what is left.
    # That is not the same thing, and it used to floor every shot to 124f/~5s no matter
    # what -- including at the fast 512 tier, where a frame costs a quarter as much.
    # Two dialogue beats came out at ~5s each on a card that could hold far more. The
    # baseline term below already represents the latent that fits in space the weight
    # accounting has covered, so let the arithmetic run instead of bailing out.
    if avail > 0 and pixels and pixels > 0:
        # Lower res -> effectively more room. Only ever applied to a POSITIVE surplus:
        # a deficit is weights that do not fit, which no resolution can shrink, and
        # scaling it would perversely make lower resolutions look worse.
        avail *= NATIVE_PIXELS / float(pixels)
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


# --- ref2va reference conditioning ----------------------------------------
# H3's reference pipeline encodes a reference image at up to a 2048 short edge.
# Reference rows ride through EVERY sampling step, so this is also the setting
# that decides how much the references cost per step.
REF_IMAGE_SHORT_EDGE = 2048
CANVAS_MULTIPLE = 32


def ref_image_canvas(w, h, gen_w, gen_h, mode="match"):
    """Pure: the (width, height) a reference image is encoded at.

    'match' scales it (DOWN only, aspect kept) to the generation's pixel area, so a
    reference costs about as much as one frame of the shot. 'max' goes to the
    reference pipeline's 2048 short edge for the best identity fidelity, which on a
    long chain is several times slower because the rows are re-attended every step
    of every shot. Never upscales: a small reference stays small."""
    w, h = max(1, int(w)), max(1, int(h))
    if mode == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    else:
        scale = min(1.0, math.sqrt((int(gen_w) * int(gen_h)) / float(w * h)))
    snap = lambda v: max(CANVAS_MULTIPLE, round(v * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return snap(w), snap(h)


def _build_ref_images(vae, images, gen_w, gen_h, mode="match"):
    """(tokenizer items, DiT blocks) for a list of reference IMAGE tensors.

    The tokenizer labels each one `<Picture N>:` itself, in the order given here --
    so the roster the prompt refers to is decided by input order, not by anything
    written in the prompt."""
    items, blocks = [], []
    for img in images:
        if img is None:
            continue
        h, w = int(img.shape[1]), int(img.shape[2])
        tw, th = ref_image_canvas(w, h, gen_w, gen_h, mode)
        resized = _resize(img[:1], tw, th, "disabled")
        items.append({"type": "image", "data": resized})
        blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                       "latent": vae.encode(resized)})
    return items, blocks


def _build_shot_conditioning(clip, vae, prompt, width, height, length, fps, handoff,
                             ref_images=None, ref_image_size="match", ref_noise_aug=None):
    latent, fc = _empty_av_latent(width, height, length, fps)
    refs = [r for r in (ref_images or []) if r is not None]
    if refs:
        # ref2va: this shot is reference-conditioned rather than keyframe-conditioned,
        # and run() decides which per shot. A tagged shot is handed the previous
        # frame as an extra REFERENCE so a tag never reads as a cut.
        #
        # WHY it is either/or is now historical. On ComfyUI 0.30 the two channels
        # could not ride together: model_base.py filled ONE `cond_video_latents`
        # list and the refs branch OVERWROTE whatever the keyframe branch had put
        # there, while PackedLayout still laid out rows for both -- so the row count
        # and the latent count disagreed, landing as a shape error deep in the DiT
        # or feeding keyframe rows a reference's latent.
        #
        # ComfyUI 0.31+ concatenates instead:
        #     payload["cond_video_latents"] = payload.get("cond_video_latents", []) + [...]
        # and PackedLayout appends keyframe segments before ref segments, so the two
        # orders agree and both channels coexist. A tagged shot therefore takes its
        # references AND a real keyframe: the keyframe ANCHORS the first frame,
        # which is what continuity needs, while a reference only supplies identity.
        # Passing the handoff as a reference (the 0.30 workaround) asked the model to
        # look like the previous frame rather than to start from it.
        items, blocks = _build_ref_images(vae, refs, width, height, ref_image_size)
        tokens = clip.tokenize(prompt, minimax_ref_items=items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        vals = {}
        if blocks:
            vals["minimax_refs"] = blocks
            # How CLEAN the reference is presented as. The DiT both blends the
            # condition latent with noise at (1 - aug) and labels those rows with a
            # timestep of max(t_video, aug) -- so the model default of 0.999 hands it
            # a finished, noise-free image, which is an invitation to REPRODUCE it
            # rather than to take an identity from it. Lower says "approximate".
            #
            # It is a payload-level value, so it reaches the keyframe rows too. That
            # is why the handoff stays out of the ref channel: a keyframe softened to
            # 0.90 would stop anchoring, and continuity is exactly what it is for.
            if ref_noise_aug is not None:
                vals["minimax_visual_cond_noise_aug"] = float(ref_noise_aug)
        # Enforced here as well as in run(): one aug covers every cond latent, so a
        # softened reference would soften the anchor. Refusing at the source means no
        # caller can assemble that combination by accident.
        if handoff is not None and keyframe_rides_with_refs(ref_noise_aug):
            kf = {"resolved_frame_index": 0,
                  "latent": vae.encode(_resize(handoff[:1], width, height, "disabled"))}
            vals["minimax_keyframes"] = [kf]
            vals["minimax_frame_count"] = fc
        if vals:
            cond = node_helpers.conditioning_set_values(cond, vals)
        return cond, latent
    # (fall through to the keyframe-only path below)
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


# Below this, softening the references would soften the handoff KEYFRAME with them.
# visual_cond_noise_aug is a single payload value applied to every cond video latent
# (ldm/minimax/model.py:502-510) and it labels both segments the same way (:584:
# "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug)). There is no per-channel
# control, so a lowered ref_noise_aug would blend the anchor frame with noise and
# push its timestep up -- destroying exactly the continuity the keyframe is for.
KEYFRAME_SAFE_AUG = 0.99


def keyframe_rides_with_refs(ref_noise_aug):
    """Can a tagged shot carry its handoff as a real KEYFRAME alongside references?

    Yes at the default aug, where the keyframe passes through essentially untouched.
    No once the user has softened the references, because the same value would soften
    the keyframe -- there the handoff falls back to riding as an extra reference,
    which is weaker for continuity but leaves nothing to compromise."""
    return ref_noise_aug is None or float(ref_noise_aug) >= KEYFRAME_SAFE_AUG


# <Picture 1>, <picture_1>, <PICTURE 1> -- all the ways people write the tag.
_PICTURE_TAG = re.compile(r"<\s*picture[\s_\-]*(\d+)\s*>", re.I)


def picture_tags(text):
    """The reference slots a shot's text asks for, in ascending order."""
    return sorted({int(m.group(1)) for m in _PICTURE_TAG.finditer(text or "")})


def resolve_tagged_refs(text, ref_list):
    """(rewritten text, images, dropped) for the <Picture N> tags in ONE shot.

    The tokenizer numbers references by their position in the list it is handed, so
    a shot that uses only <Picture 2> would receive that image labelled
    <Picture 1> and the text would point at nothing. The tags are therefore
    RENUMBERED per shot to match what that shot actually carries: slot 2 alone
    becomes <Picture 1>, slots 2 and 4 become <Picture 1> and <Picture 2>.

    A tag naming a slot with no image connected refers to nothing at all, so it is
    removed from the text rather than left to confuse the encoder, and reported."""
    wanted = picture_tags(text)
    live = [n for n in wanted if 1 <= n <= len(ref_list or [])]
    dropped = [n for n in wanted if n not in live]
    renumber = {old: new for new, old in enumerate(live, 1)}

    def sub(m):
        n = int(m.group(1))
        return f"<Picture {renumber[n]}>" if n in renumber else ""

    out = _PICTURE_TAG.sub(sub, text or "")
    if dropped:                       # tidy the gap a removed tag leaves behind
        out = re.sub(r"\s+([,.;:])", r"\1", out)
        out = re.sub(r"(,\s*){2,}", ", ", out)
        out = re.sub(r"\s{2,}", " ", out)
    return out.strip(), [ref_list[n - 1] for n in live], dropped


def shot_references(ref_list, ref_mode, shot_index, handoff):
    """Pure: which reference images shot `shot_index` is conditioned on, or [] when
    the shot should use the keyframe handoff instead.

    The three modes trade identity against continuity, and they trade because a
    shot cannot carry both channels:

      'first shot'  -- references establish the cast in shot 1; every later shot
                       uses the last-frame handoff. Continuity is unbroken and the
                       look propagates down the chain, but only through the frames.
      'every shot'  -- every shot is ref-conditioned. Strongest identity, and no
                       handoff at all, so shots meet as CUTS rather than as one
                       continuous take.
      'every shot + handoff ref'
                    -- every shot is ref-conditioned AND the previous shot's last
                       frame is appended as one more reference. Continuity comes
                       back as a soft signal (the model is shown where the last
                       shot ended rather than told to start exactly there), and it
                       stays a single ref2va task, so nothing conflicts."""
    if not ref_list:
        return []
    if ref_mode == "first shot":
        return list(ref_list) if shot_index == 0 else []
    if ref_mode == "every shot":
        return list(ref_list)
    if ref_mode == "every shot + handoff ref":
        return list(ref_list) + ([handoff] if handoff is not None else [])
    return list(ref_list) if shot_index == 0 else []       # unknown value -> safest


# --- text-encoder / DiT compatibility -------------------------------------
# H3's DiT accepts text conditioning at exactly two widths (comfy/ldm/minimax/
# model.py, preprocess_text_embeds):
#   * text_dim   -- raw encoder states, projected by condition_proj (5120 on
#                   stock H3: Qwen3-VL-32B truncated to 50 layers)
#   * hidden_size-- states already refined to DiT width (5376), passed through
# Anything else dies deep inside ComfyUI as a bare "mat1 and mat2 shapes cannot
# be multiplied (156x6144 and 5120x5376)", which reads like a bug in this node.
# Check the width up front and name what is actually wrong.
_TE_HIDDEN = {
    5120: "Qwen3-VL-32B truncated to 50 layers -- the H3 text encoder",
    5376: "text embeds already refined to DiT width",
    4096: "Qwen3-VL-8B / T5-XXL -- not an H3 encoder",
    3584: "Qwen2.5-VL-7B -- not an H3 encoder",
    2560: "Qwen3-VL-4B -- not an H3 encoder",
    2048: "Qwen3-VL-2B / Qwen3-30B-A3B -- not an H3 encoder",
}


def _te_name(dim):
    return _TE_HIDDEN.get(dim, "not a width any H3 encoder produces")


def text_encoder_mismatch_note(got, accepted):
    """Pure: message for conditioning of width `got` fed to a DiT that accepts
    the widths in `accepted`, or None if it fits / nothing is known. Torch-free
    so the tests can drive it."""
    ok = sorted({int(a) for a in (accepted or ()) if a})
    if not got or not ok or int(got) in ok:
        return None
    got = int(got)
    return (
        f"H3 Long Videos: the CLIP input does not match this diffusion model. Its "
        f"conditioning is {got}-dim ({_te_name(got)}), but this H3 DiT only accepts "
        + " or ".join(f"{a} ({_te_name(a)})" for a in ok) + ". Check, in this order: "
        f"(1) the CLIPLoader feeding 'clip' is set to the MiniMax-H3 type -- the same "
        f"file loaded under another type gives a different width; (2) the encoder file "
        f"is the H3 one that shipped with your H3 checkpoint, not another Qwen3-VL; "
        f"(3) no upstream node replaced the conditioning between the encoder and this "
        f"node. Nothing was rendered."
    )


def _dit_text_widths(model):
    """The text widths this DiT accepts: (condition_proj.in_features,
    hidden_size). Reads module attributes, not weight.shape -- a quantized or
    packed weight has a misleading shape and would fake a mismatch. Missing
    values are dropped, so a model this can't introspect yields ()."""
    m = getattr(model, "model", model)
    dm = getattr(m, "diffusion_model", None)
    proj = getattr(dm, "condition_proj", None)
    out = []
    for n in (getattr(proj, "in_features", None), getattr(dm, "hidden_size", None)):
        if isinstance(n, int) and n > 0:
            out.append(n)
    return tuple(out)


def _cond_embed_dim(cond):
    """Width of an encoded conditioning's embedding tensor, or None."""
    try:
        return int(cond[0][0].shape[-1])
    except Exception:
        return None


# --- quantization kernels ---------------------------------------------------
# The kernels are NOT something this node installs or calls. comfy_kitchen (imported
# as `ck` in comfy/quant_ops.py) is a compiled package shipped with ComfyUI, and
# comfy/ops.py routes every quantized Linear through it -- ck.int8_linear() and
# friends -- whenever the loaded weights carry a quant_format. Sampling runs the
# model, so the node gets that path for free and cannot opt in or out of it.
#
# What the node CAN do is notice when the path is not there, because that failure
# is silent. If the CUDA backend is disabled (torch built against CUDA < 13
# triggers ck.registry.disable("cuda")) or comfy_kitchen fails to import, ComfyUI
# logs one line at startup and then quietly runs a slower, lower-fidelity route.
# Nothing errors, and the first sign is soft output -- which is indistinguishable
# from a dozen other causes unless something says so.
#
# Every format the checkpoint might use, mapped to the comfy-kitchen capability
# that serves it. The names come from ck.list_backends()[...]["capabilities"].
_QUANT_CAPABILITY = {
    "int8_tensorwise": "int8_linear",
    "int8_tensorwise+convrot": "int8_linear",
    "convrot_w4a4": "convrot_w4a4_linear",
    "asym_w4a8_int8": "w4a8_int8_linear",
    "nvfp4": "scaled_mm_nvfp4",
    "mxfp8": "scaled_mm_mxfp8",
}


def quant_format_of(model):
    """The dominant quant format of the loaded DiT, or "" when it is unquantized.

    Same detection the inspector node uses: a module's `quant_format` tag, with
    int8 split by whether its packed weight carries convrot."""
    try:
        dm = getattr(getattr(model, "model", None), "diffusion_model", None)
        if dm is None or not hasattr(dm, "modules"):
            return ""
        counts = {}
        for m in dm.modules():
            fmt = getattr(m, "quant_format", None)
            if fmt is None:
                continue
            if fmt == "int8_tensorwise":
                params = getattr(getattr(m, "weight", None), "_params", None)
                if getattr(params, "convrot", False):
                    fmt = "int8_tensorwise+convrot"
            counts[fmt] = counts.get(fmt, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0] if counts else ""
    except Exception:
        return ""


def kernel_backend_note(model):
    """Warn when the loaded checkpoint's quant format has no accelerated backend.

    Silent on an unquantized checkpoint (nothing to accelerate) and silent when a
    capable backend is present, so it only speaks when something is actually
    wrong."""
    fmt = quant_format_of(model)
    if not fmt:
        return ""
    want = _QUANT_CAPABILITY.get(fmt)
    try:
        import comfy_kitchen as ck
    except Exception as e:
        return (f"the checkpoint is {fmt} but comfy_kitchen failed to import "
                f"({type(e).__name__}) -- ComfyUI is running the slow dequantize "
                f"fallback, which is lower fidelity as well as slower")
    try:
        backends = ck.list_backends()
    except Exception:
        return ""                       # cannot introspect; never block on that
    live = [name for name, b in backends.items()
            if b.get("available") and not b.get("disabled")
            and (want is None or want in (b.get("capabilities") or ()))]
    if live:
        return ""
    disabled = [f"{name} ({b.get('unavailable_reason') or 'disabled'})"
                for name, b in backends.items() if not b.get("available") or b.get("disabled")]
    return (f"the checkpoint is {fmt} but no comfy-kitchen backend offers "
            f"'{want or fmt}' -- " + ("; ".join(disabled) if disabled else "none available")
            + ". ComfyUI falls back to a dequantize path: slower, and lower fidelity. "
              "Check the startup log for 'Found comfy_kitchen backend', and that torch "
              "is built against CUDA 13+ (cu130), which is what keeps the CUDA backend "
              "enabled")


# --- flow-shift vs step count ----------------------------------------------
# The shift maps timesteps onto sigmas, and the 'simple' scheduler then samples
# that curve at evenly spaced INDICES. At a high shift the curve is steep at the
# low-sigma end, so the last interval swallows most of the run:
#
#     4 steps, shift 12   ->  2.7% / 5.0% / 12.3% / 80.0%
#     4 steps, shift 3    -> 10.0% / 15.0% / 25.0% / 50.0%
#    20 steps, shift 12   ->  worst single step 38.7%
#
# 12 is the right default -- it is what H3's own model config declares, and at ~20
# steps it is well balanced. It only misbehaves when a distill LoRA drops the step
# count under it, and then it does so invisibly: nothing errors, the picture just
# comes back soft and painterly because one enormous final jump cannot resolve fine
# detail. Cheap to detect, so detect it.
def flow_step_shares(shift, steps, timesteps=1000):
    """Fraction of total denoising each sampler step performs.

    Mirrors comfy.model_sampling.ModelSamplingDiscreteFlow (time_snr_shift) plus
    comfy.samplers.simple_scheduler, which indexes the precomputed sigma table
    linearly. Returns [] when the inputs cannot form a schedule."""
    steps = int(steps)
    if steps < 2 or shift is None or float(shift) <= 0:
        return []
    a = float(shift)
    table = [a * t / (1.0 + (a - 1.0) * t)
             for t in ((i + 1) / timesteps for i in range(timesteps))]
    stride = len(table) / steps
    sig = [table[-(1 + int(x * stride))] for x in range(steps)] + [0.0]
    deltas = [sig[i] - sig[i + 1] for i in range(steps)]
    total = sum(deltas)
    if total <= 0:
        return []
    return [d / total for d in deltas]


# On ComfyUI 0.31+ the two shifts are COUPLED, and that is new. ModelSamplingAV
# carries the audio latent on the VIDEO schedule scaled by
#
#     audio_scale = shift_video / shift_audio        (12 / 3 = 4.0 by default)
#
# and that ratio drives process_latent_in, process_latent_out, the minimax payload
# and the DiT forward (comfy/model_base.py 2141/2144/2181, ldm/minimax/model.py:530).
# Collapse it to 1.0 by setting the two shifts equal and the audio pipeline loses
# the scaling it is built around -- which comes back as babble or silence.
#
# Before 0.31 the audio velocity was scaled by a derivative instead, so the two
# shifts were effectively independent and lowering shift_video alone was harmless.
# It is not harmless now.
H3_AUDIO_SCALE = 4.0            # the model's own 12/3


def audio_scale_note(shift_video, shift_audio):
    """Warn when the video/audio shift RATIO has drifted from what H3 expects."""
    try:
        sv, sa = float(shift_video), float(shift_audio)
    except (TypeError, ValueError):
        return ""
    if sa <= 0 or sv <= 0:
        return ""
    ratio = sv / sa
    if ratio >= 1.75:
        return ""
    return (f"shift_video {sv:g} / shift_audio {sa:g} gives audio_scale {ratio:.2f}, "
            f"against the {H3_AUDIO_SCALE:g} this model is built around. The audio latent "
            f"rides the VIDEO schedule scaled by that ratio, so flattening it toward 1.0 "
            f"breaks the audio branch -- babble or silence. Keep the two shifts about "
            f"{H3_AUDIO_SCALE:g}:1 apart: for shift_video {sv:g}, use shift_audio "
            f"{max(0.25, sv / H3_AUDIO_SCALE):g}")


def schedule_balance_note(shift, steps, scheduler, worst_allowed=0.55):
    """Warn when one sampler step carries most of the denoising.

    Only for the 'simple' scheduler, because that is the curve this reproduces --
    reporting these numbers for a scheduler that spaces sigmas differently would be
    making them up. The suggestion is searched rather than guessed: the lowest
    shift whose worst step falls under the threshold."""
    if str(scheduler) != "simple":
        return ""
    shares = flow_step_shares(shift, steps)
    if not shares:
        return ""
    worst = max(shares)
    if worst <= worst_allowed:
        return ""
    better = ""
    for cand in (6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0):
        cand_shares = flow_step_shares(cand, steps)
        if cand_shares and max(cand_shares) <= worst_allowed:
            # shift_audio has to come down WITH it. The two are coupled on 0.31+:
            # audio_scale = shift_video / shift_audio, and lowering only the video
            # shift flattens that ratio toward 1.0, which breaks the audio branch.
            # Suggesting a bare shift_video here is what produced babble.
            better = (f"; shift_video {cand:g} spreads it to "
                      + "/".join(f"{s * 100:.0f}%" for s in cand_shares)
                      + f" -- lower shift_audio to {max(0.25, cand / H3_AUDIO_SCALE):g} "
                        f"at the same time, or the audio breaks")
            break
    return (f"shift_video {float(shift):g} at {int(steps)} steps puts "
            + "/".join(f"{s * 100:.0f}%" for s in shares)
            + f" of the denoising into each step -- one step doing {worst * 100:.0f}% "
            f"cannot resolve fine detail, which renders as soft, painterly output"
            + better)


def check_text_encoder(model, cond):
    """Raise a readable RuntimeError when clip and model disagree. Silent when
    either side can't be read -- never block a run on a failed introspection."""
    note = text_encoder_mismatch_note(_cond_embed_dim(cond), _dit_text_widths(model))
    if note:
        raise RuntimeError(note)


def _is_audio_vae(v):
    """True when v looks like the H3 audio VAE (DAC/BigVGAN), False when it looks
    like a video/image VAE, None when it can't be told. The video VAEs carry a
    3-tuple upscale_ratio (t, y, x); the audio VAE carries a scalar and reports
    latent_dim 2 with an audio_sample_rate."""
    ur = getattr(v, "upscale_ratio", None)
    if isinstance(ur, (tuple, list)):
        return False
    if getattr(v, "audio_sample_rate", None) or getattr(v, "audio_sample_rate_output", None):
        return True
    if isinstance(ur, (int, float)) and getattr(v, "latent_dim", None) == 2:
        return True
    return None


def check_audio_vae_loaded(audio_vae):
    """Catch an UNCONVERTED audio VAE checkpoint.

    comfy/ldm/minimax/audio_vae.py loads a checkpoint whose weight-norm has been
    folded into plain "*.weight" tensors. Feed it the raw upstream file (172
    weight_g/weight_v pairs, no latents_mean/latents_std) and load_state_dict
    reports the misses as a WARNING, not an error: every weight-normed conv keeps
    its random init and the two normalization buffers stay torch.empty(), i.e.
    uninitialized memory. Decoding then multiplies the latents by garbage and the
    audio comes out as noise -- with nothing in the log at render time to say why.

    latents_std is the cheapest tell: it is a real per-channel scale, so a
    non-finite or absurd value means the buffer was never filled."""
    m = getattr(audio_vae, "first_stage_model", None)
    mean, std = getattr(m, "latents_mean", None), getattr(m, "latents_std", None)
    if mean is None or std is None:
        return
    try:
        bad = (not torch.isfinite(mean).all() or not torch.isfinite(std).all()
               or float(std.min()) <= 0.0 or float(std.max()) > 1e3
               or float(mean.abs().max()) > 1e3)
    except Exception:
        return                       # never block a render on a failed introspection
    if bad:
        raise RuntimeError(
            "the audio VAE loaded but its weights are NOT initialized -- this is the raw "
            "upstream MiniMax-H3 audio checkpoint (weight_g/weight_v weight-norm pairs, no "
            "latents_mean/latents_std). ComfyUI's loader needs the CONVERTED file, with "
            "weight-norm folded into plain '*.weight' tensors. Look for the 'Missing VAE keys' "
            "warning in the log when the VAE loaded. Download the repackaged H3 audio VAE from "
            "the Comfy-Org release; rendering with this one produces noise, not speech.")


def check_vae_wiring(vae, audio_vae):
    """Catch the commonest miswire -- the video VAE dropped into BOTH VAE inputs.
    Without this the run samples a whole shot, decodes the video fine, then dies
    deep inside comfy/sd.py with 'IndexError: tuple index out of range' when the
    video memory estimator indexes shape[4] of the 4-D audio latent."""
    if _is_audio_vae(audio_vae) is False:
        raise RuntimeError(
            "audio_vae is a video/image VAE, not the H3 audio VAE. Load the audio "
            "autoencoder (the DAC/BigVGAN one shipped with MiniMax-H3, e.g. "
            "minimax_h3_audio_vae.safetensors) in its own VAELoader and wire that "
            "into 'audio_vae'; the video VAE belongs on 'vae' only.")
    check_audio_vae_loaded(audio_vae)
    if _is_audio_vae(vae) is True:
        raise RuntimeError(
            "vae is the H3 audio VAE -- the video and audio VAE inputs are swapped. "
            "Wire the video VAE into 'vae' and the audio VAE into 'audio_vae'.")


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
        # A V3 node exposes INPUT_TYPES and a truthy FUNCTION ('EXECUTE_NORMALIZED')
        # for compatibility, so this branch runs on 0.31+ too -- and there it returns
        # a NodeOutput, not a tuple. Without the unwrap the caller got the wrapper
        # object where a MODEL belonged. Unreachable today because the direct patch
        # succeeds first, which is exactly why it went unnoticed.
        out = getattr(out, "result", out)
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
    # deepcopy, not copy: model_sampling is an nn.Module, and a SHALLOW copy shares
    # its `_buffers` dict with the original. set_parameters() re-registers `sigmas`
    # into that shared dict, so a shallow copy silently rewrites the BASE model's
    # sigma table -- the very thing this copy exists to prevent. Our own run reads
    # the patched object either way, but ComfyUI caches the model across queue
    # runs, so the damage outlives this execution and reaches anything else holding
    # that model. The buffer is ~1000 floats; the deepcopy is free.
    ms = copy.deepcopy(m.get_model_object("model_sampling"))
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


# --- SLA (sparse-linear attention) pairing ----------------------------------
# An SLA LoRA is a turbo LoRA fine-tuned WITH sparse attention in the loop, so the
# weights have already adapted to the approximation. That is the whole benefit: you
# get the sparse-attention speedup without the quality collapse. The two halves are
# a matched pair and only work together --
#
#   sparse attention ON  + ordinary LoRA -> the model sees an attention map it was
#       never trained on. Long-range coherence is what sparsity drops first, and in
#       a video DiT that shows up as the SAME PERSON RENDERED TWICE.
#   SLA LoRA + sparse attention OFF -> the weights are pre-compensating for
#       sparsity that isn't there. You pay its quality cost and get no speedup.
#
# Neither half is inferable from the model object. Verified against the actual
# file: minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors carries
# NO SLA marker -- not in its 624 tensor names, not in its 9 metadata keys. It is
# byte-shape-identical to any un-resized rank-128 H3 turbo LoRA. The only place the
# word appears is the FILENAME, so that is what has to be read.
#
# Matches 'sla' as a DELIMITED token, so 'slack', 'translate', 'isla' and 'SLAYER'
# do not fire. A LoRA that ends in '_sla' for some unrelated reason would warn
# spuriously -- the cost is one wrong line in `info`, never a changed render.
_SLA_NAME = re.compile(r"(?:^|[^a-z])sla(?:[^a-z]|$)", re.I)


def sparse_attention_active(model):
    """True when an attention-override patch (Sol-Attn and friends) is on `model`.

    ComfyUI carries these in model_options['transformer_options'], which our node
    receives already patched because the patch node ran upstream of us."""
    try:
        opts = getattr(model, "model_options", None) or {}
        tro = opts.get("transformer_options", {}) or {}
        return tro.get("optimized_attention_override") is not None
    except Exception:
        return False


def upstream_lora_names(graph, node_id, _seen=None):
    """LoRA filenames on the MODEL chain feeding this node, nearest last.

    Walks the workflow graph backwards from our own `model` input. Reading the
    filename is the only way to identify an SLA LoRA (see above), and the graph is
    the only place the filename survives -- ComfyUI stashes a LoRA's safetensors
    metadata on the patcher but never its name.

    Deliberately follows ONLY model-carrying inputs, so a LoRA wired into some
    other branch of the workflow is not mistaken for one that is affecting us."""
    if graph is None or node_id is None:
        return []
    _seen = set() if _seen is None else _seen
    nid = str(node_id)
    if nid in _seen:
        return []
    _seen.add(nid)
    try:
        node = graph.get_node(nid)
    except Exception:
        return []
    names, inputs = [], (node.get("inputs") or {})
    for key, val in inputs.items():
        # a link is [upstream_node_id, output_slot]; anything else is a widget value
        if isinstance(val, (list, tuple)) and len(val) == 2 and not isinstance(val[1], (list, dict)):
            if "model" in str(key).lower():
                names += upstream_lora_names(graph, val[0], _seen)
        elif isinstance(val, str) and "lora" in str(key).lower() and val.strip():
            names.append(val)
    return names


def sla_pairing(model, graph, node_id):
    """(sla_lora_name|None, sparse_on, note) -- how the two halves line up.

    `note` is the warning when they are mismatched, and it is worth a warning
    rather than a silent default because both mismatches cost a full render."""
    sparse = sparse_attention_active(model)
    loras = upstream_lora_names(graph, node_id)
    sla = next((n for n in reversed(loras) if _SLA_NAME.search(os.path.basename(str(n)))), None)
    note = ""
    if sla and not sparse:
        note = (f"SLA LoRA '{os.path.basename(str(sla))}' is loaded but NO sparse-attention patch "
                f"is active -- it was fine-tuned WITH sparse attention, so on dense attention you "
                f"pay its quality cost and get none of its speed. Add the Sol-Attn patch node "
                f"between the loader and this node, or load the non-SLA turbo LoRA instead")
    elif sparse and loras and not sla:
        note = (f"sparse attention is ON but the LoRA on this chain "
                f"({os.path.basename(str(loras[-1]))}) is not an SLA build -- the model was never "
                f"trained against the approximated attention map, and the first thing sparsity "
                f"drops is long-range coherence, which renders as the SAME PERSON TWICE. Load the "
                f"'_sla_' turbo LoRA, or turn the sparse-attention patch off")
    elif sparse and not loras:
        note = ("sparse attention is ON with no LoRA on this chain -- base H3 was not trained "
                "against an approximated attention map, so expect duplicated subjects. Pair it "
                "with an SLA turbo LoRA")
    return sla, sparse, note


# --- what a LoRA actually declares about itself ------------------------------
# Verified against all six LoRAs in this install before any of it was written,
# because most of what a "LoRA scanner" sounds like it should do is not in the
# files. What IS there:
#
#   base model     metadata: 'base_model' (Comfy-Org turbo builds) or
#                  'ss_base_model_version' (ai-toolkit builds, e.g. 'minimax_h3')
#   strength       metadata: 'training_scale' / 'baked_scale', both 1.0
#   step count     FILENAME only ('..._turbo_4step_...') -- no metadata field
#   training res   FILENAME only ('..._768p_...'), plus 'ss_resolution' on kohya
#                  builds, which none of these are
#
# What is NOT there, checked file by file:
#
#   trigger words  no 'ss_tag_frequency' in any of them. The ai-toolkit LoRAs carry
#                  nine metadata keys and none is a caption or tag list, so a
#                  trigger like 'mpenis' cannot be discovered -- it stays a manual
#                  exposed_terms entry.
#   sampler/cfg/   no field for these exists in any LoRA metadata standard, and
#   scheduler      none of these files has one. Reporting them would be invention.
#
# Everything below therefore either quotes metadata or says the filename said it.
# Nothing here overrides a widget: a render must stay reproducible from what the
# graph shows.
_LORA_STEPS = re.compile(r"(?:^|[^a-z0-9])(\d{1,2})[ _-]?step", re.I)
_LORA_RES = re.compile(r"(?:^|[^a-z0-9])(\d{3,4})p(?:[^a-z0-9]|$)", re.I)


def lora_declared(model):
    """The nearest LoRA's own metadata dict, as ComfyUI stashed it, or {}.

    comfy.sd.load_lora_for_models() calls set_attachments('lora_metadata', ...) on
    the patcher, so this is the LoRA's real safetensors header -- not a guess. With
    several stacked, the attachment holds the last one applied."""
    try:
        get = getattr(model, "get_attachment", None)
        return (get("lora_metadata") if get else None) or {}
    except Exception:
        return {}


def lora_hint_notes(model, graph, node_id, steps, short_edge):
    """Warnings where a LoRA's declared training disagrees with this run.

    Each note says WHERE the number came from, because the two sources are not
    equally trustworthy: metadata is what the trainer wrote, a filename is a naming
    convention that anyone can break by renaming the file."""
    notes = []
    names = upstream_lora_names(graph, node_id)
    if not names:
        return notes
    md = lora_declared(model)

    # --- base model: the one check that is pure metadata ---
    # ComfyUI's set_attachments('lora_metadata', ...) is overwritten by each loader
    # in turn, so on a stack this describes the LAST one applied only. Attributed to
    # that file by name rather than to "your LoRA", so the note cannot be read as a
    # claim about the others.
    base = str(md.get("base_model") or md.get("ss_base_model_version") or "")
    nearest = os.path.basename(str(names[-1]))
    if base and "minimax" not in base.lower() and "h3" not in base.lower():
        notes.append(f"'{nearest}' declares base_model '{base}', which is not MiniMax-H3 -- "
                     f"it will apply as noise on an H3 DiT")

    # Filename-derived checks run over EVERY LoRA on the chain. Only the nearest
    # one's metadata is reachable, but every one of their names is -- and on a stack
    # the step-count LoRA is usually NOT the nearest, so checking one file silently
    # skipped the very LoRA whose step count the sampler has to match.
    for raw in names:
        name = os.path.basename(str(raw))

        # --- step count: filename convention only ---
        m = _LORA_STEPS.search(name)
        if m:
            want = int(m.group(1))
            if want and int(steps) != want:
                notes.append(f"'{name}' is named as a {want}-step LoRA but steps={int(steps)} "
                             f"(from the filename, not metadata)" +
                             ("; a distill LoRA run past its step count re-noises a composition it "
                              "already settled" if int(steps) > want else
                              "; under its step count the distill has not finished resolving"))

        # --- training resolution: filename convention only ---
        m = _LORA_RES.search(name)
        if m and short_edge:
            want = int(m.group(1))
            if short_edge > want * 1.34:
                notes.append(f"'{name}' is named for {want}p but the short edge here is "
                             f"{int(short_edge)}px (from the filename, not metadata); well above a "
                             f"LoRA's training resolution H3 tends to tile the figure")
            elif short_edge * 1.34 < want:
                notes.append(f"'{name}' is named for {want}p but the short edge here is "
                             f"{int(short_edge)}px (from the filename, not metadata); well below "
                             f"it the LoRA's detail work has nothing to land on")

    # --- stacking a distill/turbo LoRA with others ---
    # The genuine conflict, and the one that reads as "a LoRA fight". A distill LoRA
    # rewrites the sampling trajectory: it settles global composition in the first
    # step or two of a 4-8 step schedule. A subject LoRA trained against BASE H3 at
    # a normal step count contributes deltas calibrated for a schedule that no
    # longer exists, and at full strength those land in exactly the steps the
    # distill is using to fix composition. Order does not matter -- ComfyUI sums the
    # patches -- so strength is the only lever.
    distill = [os.path.basename(str(n)) for n in names
               if _LORA_STEPS.search(os.path.basename(str(n)))
               or re.search(r"turbo|distill", os.path.basename(str(n)), re.I)]
    if distill and len(names) > 1:
        others = [os.path.basename(str(n)) for n in names
                  if os.path.basename(str(n)) not in distill]
        if others:
            notes.append(
                f"{len(names)} LoRAs on this chain, one of them a distill/turbo build "
                f"('{distill[0]}'). It settles composition in its first step or two, and the "
                f"deltas from {', '.join(f'{o!r}' for o in others[:3])} land in those same "
                f"steps -- lower the SUBJECT LoRA strengths first (0.6-0.8), not the distill's. "
                f"Stacking order does not matter; ComfyUI sums the patches")
    return notes


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






class H3LongVideos:
    CATEGORY = "sampling/minimax"
    FUNCTION = "run"
    # fps is emitted as BOTH types on purpose: ComfyUI does not coerce between them,
    # and the nodes that want a frame rate are split -- CreateVideo / SaveWEBM /
    # VHS Video Combine take a FLOAT, while plenty of utility nodes take an INT.
    # Wiring the wrong one is a red link, not a runtime error, so both are offered.
    # LATENT is APPENDED, never inserted: ComfyUI stores a link by output SLOT
    # INDEX, so adding at the end leaves every existing wire pointing at the same
    # output. Inserting mid-list would silently re-target them.
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "INT", "INT", "INT", "FLOAT", "FLOAT", "INT",
                    "LATENT")
    RETURN_NAMES = ("images", "audio", "info", "script", "frames_per_shot", "total_frames",
                    "shots", "video_seconds", "fps", "fps_int",
                    "latent")

    @classmethod
    def IS_CHANGED(cls, plan_only=False, **kwargs):
        """Force a re-run for the PLAN, leave a real render cacheable.

        Without an IS_CHANGED, ComfyUI keys this node's cache on its inputs alone, so
        re-queueing with the same widgets returns the previous outputs untouched -- and
        `info` is an output. That reads as "info doesn't update on each run", and it is
        actively misleading here, because both the info AND the chosen shot length now
        depend on LIVE FREE VRAM, which is not an input: the cached answer describes a
        card state that may no longer exist.

        plan_only is near-instant, so it always recomputes -- a stale plan is worse than
        no plan. A real render still respects the cache (returning NaN there would
        re-sample for minutes every time the graph is queued); change the seed, or any
        widget, to force one."""
        if plan_only:
            return float("nan")      # NaN != NaN -> never matches the cached signature
        return False

    @classmethod
    def INPUT_TYPES(cls):
        schema = {
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
                    "tooltip": "ASPECT RATIO only -- `megapixels` decides the size. The two are "
                               "independent, so changing shape does not change cost. Each ratio uses "
                               "H3's own dimensions as its reference, which is why 1.00MP reproduces "
                               "the model's native sizes exactly (16:9 -> 1344x768, 21:9 -> 1536x672)."}),
                # No off-switch any more: `resolution` is a bare ratio, so there are no
                # preset dimensions to fall back to. The floor keeps every result legal.
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 4.0, "step": 0.01,
                    "tooltip": "Pixel BUDGET, applied to the preset's aspect ratio. 1.00 = 1024x1024 "
                               "worth of pixels, the same convention as ComfyUI's Scale Image to Total "
                               "Pixels. START at 1.00: every NATIVE preset reproduces its own size there "
                               "(1344x768, 1536x672, ...), then step down -- 0.83 gives a 704 short edge, "
                               "0.65 gives 640 -- for speed, VRAM and longer shots. Cost and training fit "
                               "track TOTAL PIXELS, not the short edge: 1:1 768x768 reads as native by "
                               "short edge but is only 0.56MP, while 21:9 1536x672 reads as sub-native at "
                               "a full 0.98MP. Snapped to multiples of 32; `info` reports the size and MP "
                               "actually produced. Set 0 to use the preset's own dimensions instead."}),
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
                # ref2va inputs. Order matters and is the ONLY thing that decides the
                # roster: the tokenizer labels these <Picture 1>..<Picture 4> in the
                # order they appear here, then appends the prompt. Refer to them by
                # those tags in the prompt if you want a reference bound to a named
                # character ("Kristy, <Picture 1>, walks in").
                "ref_image_1": ("IMAGE", {"tooltip": "Reference image <Picture 1> -- identity/appearance "
                    "carried into the shots. ref2va conditioning REPLACES the keyframe handoff on any "
                    "shot it applies to (the model can carry one or the other, never both), so which "
                    "shots get it is set by ref_mode."}),
                "ref_image_2": ("IMAGE", {"tooltip": "Reference image <Picture 2>."}),
                "ref_image_3": ("IMAGE", {"tooltip": "Reference image <Picture 3>."}),
                "ref_image_4": ("IMAGE", {"tooltip": "Reference image <Picture 4>."}),
                "plan_only": ("BOOLEAN", {"default": False,
                    "tooltip": "Preview the shot split WITHOUT rendering. Uses THIS node's own settings (no "
                               "second node, no duplicate entry): returns the plan in 'info' and the "
                               "shots/frames/seconds outputs near-instantly. Turn off to render for real."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60,
                    "tooltip": "DISPLAY ONLY -- H3 always renders 24 fps. The model's frame grid and its "
                               "audio latent are both defined against 24, so this node computes every "
                               "duration at 24 regardless of what you set here. Set your video-save node "
                               "to 24 as well, or the clip plays at the wrong speed."}),
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
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.25, "max": 16.0, "step": 0.25,
                    "tooltip": "Audio flow shift. 3 = base H3. COUPLED to shift_video on ComfyUI "
                               "0.31+: the audio latent rides the video schedule scaled by "
                               "audio_scale = shift_video / shift_audio (12/3 = 4). Flattening that "
                               "ratio toward 1.0 breaks the audio branch -- babble or silence -- so "
                               "if you lower shift_video, lower this by the same factor. Only used "
                               "when apply_model_sampling is on."}),
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
                # ON by default: the prompt-side clauses ASK H3 not to vocalize (and now
                # condition the soundscape field too), but asking is not a guarantee --
                # babble under a silent shot was the one artifact that survived both.
                # Muting is the only deterministic answer, so it is the default and the
                # trade-off (that shot's ambience goes too) is stated in `info`.
                "mute_nonspeech_audio": ("BOOLEAN", {"default": True,
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
                "watermark_text": ("STRING", {"default": "",
                    "tooltip": "Composited with PIL onto every finished frame -- NOT rendered by the "
                               "model and NOT added to the prompt. White glyphs on a transparent layer, "
                               "alpha-blended over the video, so only the letters land on the picture. "
                               "Applied AFTER any upscale, so the text is crisp at final resolution. "
                               "Leave empty for none."}),
                "watermark_position": (["bottom-right", "bottom-left", "bottom-center",
                                        "top-right", "top-left", "top-center", "center"],
                    {"default": "bottom-right"}),
                "watermark_size": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 40.0, "step": 0.5,
                    "tooltip": "Cap height as a percentage of FRAME HEIGHT, so the mark keeps its "
                               "relative size at any resolution or upscale factor."}),
                "watermark_opacity": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Multiplies the white text alpha. 1.0 = solid white; 0.75 reads as a "
                               "watermark without burying the picture under it."}),
                "watermark_margin": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 25.0, "step": 0.5,
                    "tooltip": "Inset from the frame edge, as a percentage of the SHORT edge."}),
                "intro_text": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Title composited over the OPENING frames -- white on transparent, so the "
                               "first shot plays underneath it rather than being replaced by a card. "
                               "Multi-line is centered as a block. Holds for intro_seconds, then fades "
                               "out over intro_fade. Also PIL, never the model."}),
                "intro_position": (["center", "lower-third", "top-center", "bottom-center"],
                    {"default": "center"}),
                "intro_seconds": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.5,
                    "tooltip": "How long the title stays at full opacity before the fade starts."}),
                "intro_fade": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Linear fade-out length after the hold. 0 = hard cut."}),
                "intro_size": ("FLOAT", {"default": 9.0, "min": 0.5, "max": 40.0, "step": 0.5,
                    "tooltip": "Title cap height as a percentage of frame height."}),
                "overlay_font": ("STRING", {"default": "arial.ttf",
                    "tooltip": "TrueType font for BOTH overlays: a bare name resolved against the system "
                               "font folder (arial.ttf, arialbd.ttf, segoeui.ttf) or a full path to a "
                               ".ttf/.otf file. Falls back to the first font that loads if this one fails."}),
                "overlay_stroke": ("INT", {"default": 0, "min": 0, "max": 20,
                    "tooltip": "Black outline thickness in pixels around the white text. 0 keeps it pure "
                               "white as asked; 2-3 makes it survive a bright sky or a white wall."}),
                "ref_mode": (["where tagged", "first shot", "every shot", "every shot + handoff ref"],
                    {"default": "where tagged",
                     "tooltip": "Which shots the ref_image inputs condition. A shot carries EITHER "
                                "references or the last-frame handoff, never both. 'where tagged' "
                                "(default): write <Picture 1> in the beat where that character "
                                "appears and ONLY that shot gets the reference -- every other shot "
                                "keeps its handoff. This is the precise option: the other modes go by "
                                "shot NUMBER and are blind to who is actually in the shot, so a "
                                "character who first appears in shot 2 gets nothing while an empty "
                                "establishing shot 1 gets a portrait pushed into it. Tags are "
                                "renumbered per shot, so <Picture 2> alone still resolves. With refs "
                                "connected but no tags anywhere, falls back to first shot rather than "
                                "silently doing nothing. 'first shot' / 'every shot' / 'every shot + "
                                "handoff ref' go purely by position. Ignored when no ref_image is "
                                "connected."}),
                "ref_noise_aug": ("FLOAT", {"default": 0.999, "min": 0.50, "max": 1.0, "step": 0.005,
                    "tooltip": "How CLEAN each reference is presented to the model. 0.999 (H3's own "
                               "default) hands it a finished, noise-free image -- which invites the "
                               "model to REPRODUCE the reference in the opening frames instead of just "
                               "taking an identity from it. Lower values blend the condition with "
                               "noise and label it as approximate, so it informs the face without "
                               "being copied: try 0.95, then 0.90. Too low (below ~0.8) and the "
                               "reference stops holding identity at all. Applies ONLY to "
                               "ref-conditioned shots -- the last-frame handoff is never weakened, or "
                               "continuity would break."}),
                "ref_image_size": (["match", "max"], {"default": "match",
                    "tooltip": "How large each reference is encoded. 'match' scales it down to the "
                               "generation's pixel area -- a reference then costs about one frame per "
                               "step. 'max' uses the reference pipeline's 2048 short edge for the best "
                               "identity fidelity, but reference rows are re-attended EVERY step of "
                               "EVERY ref-conditioned shot, so on a long chain it is several times "
                               "slower. Neither ever upscales a small reference."}),
                "beat_split": (["auto", "each line"], {"default": "auto",
                    "tooltip": "How the prompt box becomes beats. Beats are meant to be separated by a "
                               "BLANK line (or a '##' line) -- but six beats typed on six consecutive "
                               "lines are ONE paragraph, so they would render as one shot with six actions "
                               "crammed into it, which looks like everyone is moving at triple speed. "
                               "auto (default): blank lines first, then any paragraph still holding "
                               "several lines is split one beat per LINE, and the info output says so. "
                               "'each line': every line is its own beat -- same result, stated explicitly. "
                               "Neither can lose a beat. (The old strict 'blank line' option was REMOVED: "
                               "it was the only setting that could silently collapse beats, and a stored "
                               "value of it now reads as 'auto'.) Directive lines (wardrobe:, seconds:, exit:) "
                               "are never beats -- they attach to the beat that follows them."}),
                "anchor_override": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Set the persistent look explicitly instead of using the first paragraph. "
                               "When this is filled in, EVERY paragraph of the prompt box is a beat/shot -- "
                               "nothing is consumed as the identity anchor. Put the permanent identity here "
                               "(hair, face, build, age) and the clothing in character_memory."}),
                "per_beat_length": ("BOOLEAN", {"default": True,
                    "tooltip": "PACING. Size each shot from what its beat actually stages, instead of giving "
                               "every shot the same length. ON (default): a beat's time is ~2s of setup plus "
                               "~2.5s per action clause, or its spoken line, whichever is longer -- so 'she "
                               "takes off her jacket and drops it on the bench' gets ~7s and a three-part "
                               "beat gets more. OFF: every shot gets the full ceiling. WHY IT MATTERS: a 3s "
                               "action in a 12s shot leaves 9 seconds the model was told nothing about, and "
                               "it fills them by repeating or REVERSING the action -- which is why clothing "
                               "comes off and goes back on. The estimate leans SHORT on purpose: an "
                               "unfinished action is continued by the next shot from the handoff frame, "
                               "while an overlong one is unrecoverable. Never exceeds the ceiling "
                               "(shot_seconds or the VRAM budget) and always lands on the 17n+5 grid. "
                               "Override any single beat with 'seconds: 8' on its own line inside that "
                               "paragraph -- that wins over everything, including this toggle."}),
                "lock_restraints": ("BOOLEAN", {"default": True,
                    "tooltip": "Physical restraints stay ON until something explicitly removes them. "
                               "Handcuffs, shackles, manacles, fetters, irons, gags, blindfolds, "
                               "harnesses, leashes, plus qualified forms like 'ankle chain' or "
                               "'leather wrist straps'. Without this they come off like any garment, "
                               "and often by ACCIDENT -- 'steps out of her jacket and the chain falls "
                               "away' removed the ankle chain as a side effect of a jacket beat. To "
                               "take one off, say so directly: 'wardrobe: Mara -= handcuffs'. Bare "
                               "'chain', 'collar', 'strap' and 'belt' are NOT treated as restraints; "
                               "they are jewellery, a shirt part, a dress part and a garment at least "
                               "as often."}),
                "anatomy_guard": (["off", "auto", "on"], {"default": "auto",
                    "tooltip": "State each person's limb COUNT positively, to stop spare arms and "
                               "duplicated hands. H3 is CFG-free at cfg 1, so a NEGATIVE prompt is "
                               "never evaluated -- 'extra limbs' in a negative does nothing. Naming "
                               "a number gives the model a target instead; negating one only puts "
                               "the word in the prompt. Added per-shot and only where people are "
                               "actually present, never in the anchor (anchor body words are what "
                               "burn a face into every opening frame). 'auto' = on below 768 short "
                               "edge OR when a LoRA is applied -- the conditions where anatomy "
                               "breaks down. Costs ~18 tokens on shots with people."}),
                "exposed_terms": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "What a stripped body zone is CALLED, per character, so it persists "
                               "automatically instead of being typed into every beat. Same syntax as "
                               "character_memory -- a PRONOUN covers everyone who declares it, a NAME "
                               "overrides it, and a trailing 'upper' targets the chest. "
                               "For example -- 'she = visible vulva, mvagina' / "
                               "'he = visible penis, mpenis' / 'Mara upper = bare breasts'. "
                               "Once a removal empties that zone the phrase is stamped into every "
                               "later shot that person is in, and clears by itself when something "
                               "covers the zone again. Put LoRA trigger words here too. Requires "
                               "prevent_nudity OFF; empty falls back to 'bare below the waist'."}),
                "prevent_nudity": ("BOOLEAN", {"default": True,
                    "tooltip": "Never let the prompt ASSERT that a body is bare. A removal still "
                               "happens either way -- this gates the sentence, not the garment. "
                               "Deleting the last item covering a zone only leaves it undescribed, "
                               "and a video model's default is a clothed person, so it covers what "
                               "nobody described. With this OFF the node states the state outright "
                               "('bare below the waist') and keeps stating it until something covers "
                               "that zone again, which is what makes a strip actually stick. ON is "
                               "the safe default; turn it OFF only when nudity is intended. Either "
                               "way info reports which zone a removal left uncovered."}),
                "auto_props": ("BOOLEAN", {"default": True,
                    "tooltip": "Carry OBJECTS across shots. Each shot is a separate generation, so "
                               "'the van' in shot 2 has no antecedent -- nothing in that prompt "
                               "describes a van, and the model invents one, which is how a second van "
                               "appears while the first is still in frame. With this on, an object "
                               "introduced indefinitely ('a white van') is bound on its first definite "
                               "reference in any later beat ('the van' -> 'the same white van') and a "
                               "short clause pins it to the previous shot: one van only, no second van. "
                               "Only the FIRST mention per shot is expanded, quoted dialogue is never "
                               "rewritten, worn garments are excluded (they have the wardrobe channel), "
                               "and frame/body nouns (the ground, the light, the hand) are never "
                               "carried."}),
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
            # Read-only graph access, for SLA-LoRA detection: a LoRA's filename is
            # the only thing that identifies an SLA build, and the graph is the only
            # place it survives. Named 'graph'/'node_id' rather than the usual
            # 'prompt' because this node already has a `prompt` widget -- ComfyUI
            # passes hidden inputs by parameter name, so "prompt": "PROMPT" would
            # overwrite the user's text with the workflow dict. Hidden inputs carry
            # no widget, so they cannot shift saved widget positions.
            "hidden": {"graph": "DYNPROMPT", "node_id": "UNIQUE_ID"},
        }
        # ComfyUI restores a saved graph's widget values POSITIONALLY, from a flat
        # widgets_values array. A widget inserted in the MIDDLE therefore shifts every
        # value after it onto the wrong widget in every workflow saved before it
        # existed -- silently, with no error. So widgets added after v1 are forced to
        # the END here, leaving the original order byte-for-byte intact.
        opt = schema["optional"]
        for name in ADDED_WIDGETS:
            if name in opt:
                opt[name] = opt.pop(name)      # re-insert at the end, value unchanged
        return schema

    def _render(self, model, clip, vae, audio_vae, negative, prompt, w, h, ln, fps, tiled, sa,
                handoff, decode_tile_frames=0, decode_tile_size=0,
                refs=None, ref_image_size="match", ref_noise_aug=None):
        positive, latent = _build_shot_conditioning(clip, vae, prompt, w, h, ln, fps, handoff,
                                                    ref_images=refs, ref_image_size=ref_image_size,
                                                    ref_noise_aug=ref_noise_aug)
        seed, steps, cfg, sn, sch, denoise = sa
        # Conditioning is built, so the text encoder and VAEs are dead weight for the
        # whole sampling loop -- evict them and keep only the DiT on the card.
        _evict_all_but(model)
        (out,) = nodes.common_ksampler(model, seed, steps, cfg, sn, sch, positive, negative,
                                       latent, denoise=denoise)
        # Keep a CPU copy of the sampled latent BEFORE decoding, for the `latent`
        # output. Latents are ~1000x smaller than the frames they decode to (a
        # 1344x768 124f shot is ~1.5MB against ~1.5GB), so carrying one per shot for
        # the whole chain is free. Detached and moved off the card immediately, for
        # the same reason the decoded frames are.
        raw = out.get("samples") if isinstance(out, dict) else None
        shot_latent = None
        if raw is not None:
            try:
                parts = raw.unbind() if hasattr(raw, "unbind") else None
                shot_latent = ([t.detach().to("cpu", copy=True) for t in parts]
                               if parts else raw.detach().to("cpu", copy=True))
            except Exception:
                shot_latent = None      # never fail a render for the sake of an output
        video = _decode_video(vae, out, tiled, free_first=model,
                              tile_t=decode_tile_frames, tile_xy=decode_tile_size)
        audio = _decode_audio(audio_vae, out)
        del out, positive, latent
        _deep_cleanup()
        return video, audio, shot_latent

    def run(self, model, clip, vae, audio_vae, prompt, resolution,
            steps, cfg, sampler_name, scheduler, seed,
            megapixels=1.0,
            first_frame=None, fps=24, plan_only=False,
            global_soundscape="", non_diegetic_music="", apply_model_sampling=True,
            shift_video=12.0, shift_audio=3.0, trim_seam=True, vary_seed_per_shot=True,
            handoff_offset=0, vram_headroom_gb=1.5, allow_res_backoff=True,
            decode_tile_frames=0, decode_tile_size=0,
            cleanup_between_shots=True,
            anchor_override="", shot_seconds=0.0, allow_oversize_shots=False,
            per_beat_length=True, beat_split="auto",
            character_memory="", auto_wardrobe=True, auto_props=True, prevent_nudity=True,
            exposed_terms="", anatomy_guard="auto", lock_restraints=True,
            auto_silence_nonspeech=True,
            subject_count_guard="auto",
            upscale="off", upscale_model="none",
            upscale_target_short_edge=0, upscale_batch=4,
            mute_nonspeech_audio=False, mute_fade_ms=40,
            watermark_text="", watermark_position="bottom-right", watermark_size=4.0,
            watermark_opacity=0.75, watermark_margin=3.0,
            intro_text="", intro_position="center", intro_seconds=3.0, intro_fade=0.6,
            intro_size=9.0, overlay_font="arial.ttf", overlay_stroke=0,
            ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None,
            ref_mode="where tagged", ref_image_size="match", ref_noise_aug=0.999,
            graph=None, node_id=None):

        # FIRST: detect a checkpoint swap since the previous execution and hard-flush.
        # A stale resident model from a different checkpoint would otherwise poison
        # every VRAM measurement below (and leave old hooks/allocator blocks behind),
        # so this must run before the schedule patch and before vram_gb().
        swap_note = flush_for_model_change(model)

        # Cheap wiring preflight: a video VAE on the audio_vae socket only blows up
        # after a full shot has been sampled and decoded, so reject it up front.
        check_vae_wiring(vae, audio_vae)

        # H3 renders 24 fps, always. Honor the widget only as a warning: a lower value
        # used to silently shorten every shot (10s -> 124f -> 5.2s of real time).
        fps_note = ("" if int(fps) == H3_FPS else
                    f"fps widget is {int(fps)} but H3 always renders {H3_FPS} fps -- all durations "
                    f"computed at {H3_FPS}; set your video-save node to {H3_FPS} too")
        fps = H3_FPS
        w, h = parse_resolution(resolution)
        # A pixel budget overrides the preset's SIZE while keeping its aspect ratio,
        # so the dropdown chooses the shape and this chooses how big. Scaling from
        # the preset's own dimensions is what makes 1.00MP reproduce each native
        # size exactly -- the preset NAMES are approximations (1344x768 is 7:4, not
        # 16:9), so computing from a nominal ratio would not.
        #
        # Reported as ACHIEVED, not requested: snapping to the 32-grid moves the
        # real area, and echoing the asked-for figure would print a number the
        # render never used.
        mp_note = ""
        if megapixels and float(megapixels) > 0:
            nw, nh = scale_to_megapixels(w, h, float(megapixels))
            if (nw, nh) != (w, h):
                mp_note = (f"megapixels {float(megapixels):.2f} -> {nw}x{nh} "
                           f"({nw * nh / MP_UNIT:.3f}MP actual; preset was {w}x{h} "
                           f"@ {w * h / MP_UNIT:.3f}MP)")
                w, h = nw, nh
            else:
                mp_note = (f"megapixels {float(megapixels):.2f} -> {w}x{h} "
                           f"({w * h / MP_UNIT:.3f}MP), the preset's own size")
        # H3 is CFG-free (cfg 1): the sampler skips the negative, but common_ksampler
        # still needs a conditioning object, so build an empty one from the clip.
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        # Cheapest possible preflight: this empty encode already went through the
        # text encoder, so compare its width to the DiT's before anything expensive.
        check_text_encoder(model, negative)

        # Patch the dual video/audio schedule onto the model here, so a missing
        # upstream ModelSamplingMiniMaxH3 can't silently produce gibberish audio.
        # Shifts come from the widgets (12/3 base default; MXFP8/turbo differ).
        ms_note = ""
        if apply_model_sampling:
            model, ms_note = apply_h3_model_sampling(model, shift_video, shift_audio)

        paras = split_paragraphs(prompt, "##")
        if anchor_override.strip():
            anchor, beat_paras = anchor_override.strip(), paras
        elif paras:
            anchor, beat_paras = paras[0], paras[1:]
        else:
            anchor, beat_paras = "", []
        # A first paragraph that would be stripped to nothing is not an anchor -- it is an
        # action beat about a tracked character, and consuming it deletes that shot
        # outright (the sentence names the character, so it gets removed from the always-on
        # anchor to avoid introducing them twice). Keep it as a BEAT and say so loudly,
        # rather than losing a shot and the scene text along with it.
        anchor_note = ""
        if (not anchor_override.strip()) and paras and \
                (anchor_contributes_nothing(anchor, character_memory.strip())
                 or anchor_is_action_beat(anchor, paras[1:])):
            preview = " ".join(anchor.split())[:60]
            anchor, beat_paras = "", paras
            anchor_note = (
                f'WARNING: paragraph 1 ("{preview}...") reads as an action beat about a tracked '
                f'character, not an identity anchor -- consuming it would have deleted that shot '
                f'entirely, so it was KEPT AS A BEAT. There is now no persistent scene text: put '
                f'the setting and style (with NO character names) in anchor_override.')
        # The anchor repeats on every shot, so what is IN it matters more than its
        # length. These have each cost a render: face words putting a face in an empty
        # establishing frame, apparatus words rendering the equipment (or someone
        # holding it), framing pinning every shot, clothing that no removal can strip.
        anchor_hazards = anchor_warnings(anchor)
        # Anchor extraction happens on PARAGRAPHS first, so a line-split can never
        # eat into the identity block; only the beat paragraphs are expanded.
        beats, split_note = expand_beats(beat_paras, beat_split)
        beats_note = (f"{len(beats)} beat(s) -> {len(beats)} shot(s) from {len(paras)} paragraph(s)"
                      + ("" if anchor_override.strip() else
                         "; paragraph 1 was consumed as the identity anchor (fill anchor_override "
                         "to make EVERY paragraph a beat)")
                      + (f". {split_note}" if split_note else ""))

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
        # Hitting the internal floor means the budget arithmetic gave up, and every shot
        # comes out ~5s regardless of what the beats need. That looked like the node
        # ignoring the prompt; say what actually ran out and what moves the number.
        if ln <= align_frame_count(MIN_SHOT_FRAMES) and total_gb > 0 and not (
                shot_seconds and float(shot_seconds) > 0):
            ln_note = ((ln_note + " ") if ln_note else "") + (
                f"SHOT LENGTH IS AT THE {ln}f (~{ln / fps:.1f}s) FLOOR -- every shot will be this "
                f"long whatever the beat asks for. "
                + (f"No live free-VRAM reading was available, so there was nothing to budget from "
                   f"(weights ~{resident_gb:.1f}GB stream and cannot be subtracted from the "
                   f"{total_gb:.1f}GB card)."
                   if streaming else
                   f"Weights ~{resident_gb:.1f}GB + headroom ~{eff_headroom:.1f}GB leave nothing of "
                   f"the {total_gb:.1f}GB card for the latent.")
                + f" Free right now: ~{free_gb:.1f}GB. Lower vram_headroom_gb, drop to the "
                  f"balanced/fast resolution tier, or close other GPU apps")
        accel_note = quant_accel_note(model)
        if streaming:
            ln_note = ((ln_note + " ") if ln_note else "") + (
                f"weights (~{resident_gb:.1f}GB) exceed VRAM (~{total_gb:.1f}GB), so they stream rather "
                f"than sitting on the card -- that figure is NOT subtracted from the budget, which is "
                f"built from the ~{free_gb:.1f}GB actually free instead")
        tiled = total_gb > 0 and (total_gb - resident_gb) < 20

        # Sub-native renders duplicate subjects far more often, so default the guard on
        # there and leave native renders alone (the extra clause costs prompt budget).
        lora_on = lora_active(model)
        # An SLA LoRA and a sparse-attention patch are a matched pair; either one
        # alone costs a full render to discover. Computed here so plan_only reports
        # it too -- that is the point of catching it, before anything is sampled.
        sla_name, sparse_on, sla_note = sla_pairing(model, graph, node_id)
        # Same idea, wider: where a LoRA's own declared training disagrees with this
        # run. Reports only -- never overrides a widget, so the render stays
        # reproducible from what the graph shows.
        hint_notes = lora_hint_notes(model, graph, node_id, steps, min(w, h))
        # The flow shift is right for base H3 at ~20 steps and wrong for a distill
        # at 4-8, and it fails silently -- soft output, no error. Computed here so
        # plan_only reports it before anything is sampled.
        sched_note = (schedule_balance_note(shift_video, steps, scheduler)
                      if apply_model_sampling else "")
        # The quantization kernels are ComfyUI's, not this node's -- but losing them
        # is silent, and the symptom (soft output) looks like a dozen other causes.
        kernel_note = kernel_backend_note(model)
        audio_ratio_note = (audio_scale_note(shift_video, shift_audio)
                            if apply_model_sampling else "")
        # One ordered list, emitted once per output. These used to be six separate
        # `+ (f" X -- {note}." if note else "")` fragments repeated at BOTH the plan
        # and the render site -- twelve lines that had to stay in sync by hand. They
        # did not: `audio_note` collided with the mute-reporting variable of the same
        # name, so the shift-ratio warning reached plan_only and never a real render,
        # while the mute note was printed twice under the wrong label.
        # 'auto' fires below native resolution AND whenever a LoRA is applied: a
        # distilled LoRA fixes the subject count in its first step or two, so the
        # count has to be stated even at native size.
        anatomy_on = (anatomy_guard == "on" or
                      (anatomy_guard == "auto" and (min(w, h) < 768 or lora_on)))
        count_subjects = (subject_count_guard == "on" or
                          (subject_count_guard == "auto" and (min(w, h) < 768 or lora_on)))
        # `ln` is the CEILING (VRAM budget, or a forced shot_seconds). Each beat gets
        # its own length UNDER that ceiling -- including when shot_seconds is forced,
        # which now means "no shot longer than this" rather than "every shot exactly
        # this". Forcing a length used to DISABLE per-beat sizing entirely, which is
        # why a plan made with a forced length disagreed with the auto render: two
        # different code paths for the same question.
        lens, len_notes = plan_beat_frames(beats, fps, ln, per_beat=bool(per_beat_length))
        secs = [n / fps for n in lens]
        if len_notes:
            n_short = sum(1 for n in lens if n < ln)
            ln_note = ((ln_note + " ") if ln_note else "") + (
                f"per-beat pacing sized {n_short} of {len(lens)} shot(s) under the {ln}f "
                f"(~{ln / fps:.1f}s) ceiling from their own content: " + "; ".join(len_notes)
                + ". Turn per_beat_length OFF to give every shot the full ceiling")
        # With pacing OFF, every beat gets the ceiling whether it has anything to fill
        # it with or not -- so say which beats are too thin for the length they got.
        # This is the failure that reads as an action repeating or playing backwards.
        pace_warnings = pacing_warnings(beats, lens, fps)
        if pace_warnings:
            ln_note = ((ln_note + " ") if ln_note else "") + (
                "THIN BEATS -- the model must invent the remaining time, which it fills by "
                "repeating or REVERSING the action: " + "; ".join(pace_warnings)
                + ". Add a second clause to the beat, set 'seconds:' on it, or turn "
                  "per_beat_length ON to size shots from their content")
        fit_warnings = dialogue_fit_warnings(beats, secs)
        # The opposite error, and the one that babbles: far more shot than line.
        filler_warnings = dialogue_filler_warnings(beats, secs)
        if filler_warnings:
            ln_note = ((ln_note + " ") if ln_note else "") + (
                "BABBLE RISK -- " + "; ".join(filler_warnings)
                + ". Turn per_beat_length ON to size these shots from their line, or set "
                  "'seconds:' on the beat")
        wardrobe_notes = []
        strip_shots = []      # shots that newly bared a zone -> the NEXT shot starts fresh
        gens = distribute_generations(anchor, beats, global_soundscape.strip(),
                                      non_diegetic_music.strip(), character_memory.strip(),
                                      auto_wardrobe, auto_silence_nonspeech, count_subjects,
                                      lora_on, notes_out=wardrobe_notes, auto_props=auto_props,
                                      prevent_nudity=prevent_nudity,
                                      exposed_terms=exposed_terms, strip_out=strip_shots,
                                      anatomy_guard=anatomy_on,
                                      lock_restraints=lock_restraints)

        # A scenery beat mid-chain hands the next shot a frame with no people in
        # it. Both prompts are individually correct, so this is invisible without
        # looking at the sequence -- which is why chains lose their cast in the
        # middle rather than degrading steadily.
        cohesion_notes = continuity_warnings(gens)
        preflight = [("SLA", sla_note),
                     ("LORA HINTS", "; ".join(hint_notes)),
                     ("", mp_note),
                     ("SCHEDULE", sched_note),
                     ("KERNELS", kernel_note),
                     ("AUDIO", audio_ratio_note),
                     ("CONTINUITY", "; ".join(cohesion_notes))]
        preflight_txt = "".join(f"{(lbl + ' -- ') if lbl else ''}{txt}. "
                                for lbl, txt in preflight if txt)

        if plan_only:
            # Preview the split using THIS node's own settings -- no render, near-instant.
            shots = len(gens)
            plan_lens = (lens + [ln] * shots)[:shots]
            total = round(sum(plan_lens) / fps, 2)
            uniform = len(set(plan_lens)) == 1
            shape = (f"{shots} shot(s) x {plan_lens[0]}f (~{plan_lens[0] / fps:g}s each)" if uniform
                     else f"{shots} shot(s), {sum(plan_lens)}f total: "
                          + ", ".join(f"{n}f/~{n / fps:.1f}s" for n in plan_lens))
            vram_str = f"{total_gb:.1f}GB total / {resident_gb:.1f}GB weights / {free_gb:.1f}GB free" if total_gb else "VRAM unknown"
            # Same dialogue/audio accounting the render reports, so the plan says up
            # front which shots will come back silent instead of surprising you after.
            n_silent = sum(1 for f in speech_flags(beats) if not f)
            plan_audio = (f" {n_silent} of {shots} shot(s) have no quoted dialogue -> "
                          + ("AUDIO-MUTED (ambience goes too)" if mute_nonspeech_audio
                             else "prompt/soundscape silencing only")) if n_silent else ""
            # Same reference accounting the render reports: which shots lose the
            # handoff is a composition decision, so it belongs in the preview.
            n_refs = len([r for r in (ref_image_1, ref_image_2, ref_image_3, ref_image_4)
                          if r is not None])
            plan_ref = ""
            if n_refs:
                on = [n + 1 for n in range(shots)
                      if shot_references([1] * n_refs, ref_mode, n, 1 if n else None)]
                plan_ref = (f" ref2va: {n_refs} reference image(s) at '{ref_image_size}' on shot(s) "
                            f"{','.join(str(n) for n in on) or 'none'} (ref_mode '{ref_mode}') -> "
                            f"those shots drop the last-frame handoff")
            plan = ((anchor_note + " ") if anchor_note else "") + \
                   preflight_txt + \
                   (("DIALOGUE MAY BE CUT OFF -- " + "; ".join(fit_warnings) + ". ") if fit_warnings else "") + \
                   (f"PLAN (no render): {shape} = ~{total:g}s at {w}x{h}. "
                    f"{len(beats) or 1} beat(s). decode {'tiled' if tiled else 'full'}. {vram_str}."
                    + (f" {beats_note}." if beats_note else "")
                    + (" ANCHOR: " + "; ".join(anchor_hazards) + "."
                       if anchor_hazards else "")
                    + (f"{plan_audio}." if plan_audio else "")
                    + (" EXPOSURE -- " + "; ".join(wardrobe_notes) + "."
                       if wardrobe_notes else "")
                    + (f"{plan_ref}." if plan_ref else "")
                + (f" {fps_note}." if fps_note else "")
                    + (f" {ln_note}." if ln_note else ""))
            ph_img = torch.zeros((1, 64, 64, 3))
            ph_audio = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 44100}
            # plan_only samples nothing, so there is no latent to hand out. Emit a
            # correctly-SHAPED empty one rather than None: a downstream LATENT input
            # would choke on None, and this keeps the preview wireable exactly like
            # a real run.
            return (ph_img, ph_audio, plan, "\n---\n".join(gens), max(plan_lens),
                    sum(plan_lens), shots, total, float(fps), int(fps),
                    _empty_av_latent(w, h, 5, fps)[0])

        spk = speech_flags(beats)          # which shots have real (quoted) dialogue
        vram_trace = []                    # free VRAM after each shot
        muted_flags = []                   # which shots were audio-silenced
        hoff = max(0, int(handoff_offset))
        backoff, video_chunks, audio_chunks = [], [], []
        latent_chunks = []                 # per-shot sampled latents, pre-decode
        handoff, sr = first_frame, None
        ref_list = [r for r in (ref_image_1, ref_image_2, ref_image_3, ref_image_4) if r is not None]
        ref_shots = []                     # which shots ended up ref-conditioned
        ref_missing = []                   # <Picture N> tags naming an unconnected slot
        ref_carried = []                   # tagged shots that kept continuity as an extra ref
        ref_keyframed = []                 # tagged shots that kept it as a real keyframe
        # 'where tagged' reads the prompt instead of counting shots. If references are
        # connected but nothing is tagged anywhere, fall back to first-shot placement
        # rather than silently conditioning nothing at all.
        tag_driven = bool(ref_list) and ref_mode == "where tagged" and any(
            picture_tags(g) for g in gens)
        if ref_list and ref_mode == "where tagged" and not tag_driven:
            ref_mode = "first shot"
        if cleanup_between_shots:
            _deep_cleanup()          # start the first (heaviest) shot with max free VRAM

        shot_lens = (lens + [ln] * len(gens))[:len(gens)]
        for i, gen_prompt in enumerate(gens):
            # denoise is fixed at 1.0 (partial denoise desyncs the joint AV schedule).
            sa = (seed + i if vary_seed_per_shot else seed, steps, cfg, sampler_name, scheduler, 1.0)
            ln_i = shot_lens[i]        # this beat's own length (<= the VRAM ceiling)
            # Which conditioning channels this shot carries is decided here; see
            # _build_shot_conditioning for how they are packed. On ComfyUI 0.31+ a
            # shot may carry BOTH references and a keyframe.
            carry_keyframe = False       # tagged shot keeps its handoff as a keyframe
            if tag_driven:
                # The prompt itself says where each reference belongs: the shot whose
                # text names <Picture N> gets image N, renumbered to match what that
                # shot actually carries. Every untagged shot keeps its handoff.
                gen_prompt, shot_refs, dropped = resolve_tagged_refs(gen_prompt, ref_list)
                for n in dropped:
                    if n not in ref_missing:
                        ref_missing.append(n)
                # A tagged shot keeps its continuity, but by which channel depends on
                # ref_noise_aug. On ComfyUI 0.31+ refs and keyframes coexist, so the
                # handoff can be a REAL keyframe -- it anchors the first frame, which
                # is what continuity means. Once references are softened that same
                # aug would soften the keyframe too, so there it falls back to riding
                # as an extra reference (the pre-0.31 workaround): weaker, but it
                # leaves no anchor to compromise. Appended AFTER the tagged images, so
                # their <Picture N> numbers are untouched.
                if shot_refs and handoff is not None:
                    if keyframe_rides_with_refs(ref_noise_aug):
                        carry_keyframe = True
                        ref_keyframed.append(i + 1)
                    else:
                        shot_refs = shot_refs + [handoff]
                        ref_carried.append(i + 1)
            else:
                shot_refs = shot_references(ref_list, ref_mode, i, handoff)
            # A shot that follows a strip starts FRESH. Continuing from a frame that
            # still shows the garment is how it reappears -- the picture outvotes the
            # text every time. Costs a cut exactly where the state changes, which is
            # where a cut belongs anyway.
            after_strip = i in strip_shots          # strip_shots is 1-based, i is 0-based
            shot_handoff = (None if after_strip
                            else handoff if (carry_keyframe or not shot_refs) else None)
            if shot_refs:
                ref_shots.append(i + 1)
            if i == 0:
                while True:
                    try:
                        frames, audio, shot_latent = self._render(model, clip, vae, audio_vae, negative, gen_prompt, w, h, ln_i, fps, tiled, sa, shot_handoff, decode_tile_frames, decode_tile_size,
                                                     shot_refs, ref_image_size, ref_noise_aug)
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
                    frames, audio, shot_latent = self._render(model, clip, vae, audio_vae, negative, gen_prompt, w, h, ln_i, fps, tiled, sa, shot_handoff, decode_tile_frames, decode_tile_size,
                                                     shot_refs, ref_image_size, ref_noise_aug)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if not _is_oom(e) or tiled:
                        raise
                    mm.soft_empty_cache(True); tiled = True; backoff.append(f"shot {i+1}: tiled")
                    frames, audio, shot_latent = self._render(model, clip, vae, audio_vae, negative, gen_prompt, w, h, ln_i, fps, tiled, sa, shot_handoff, decode_tile_frames, decode_tile_size,
                                                     shot_refs, ref_image_size, ref_noise_aug)

            if shot_latent is not None:
                latent_chunks.append(shot_latent)
            sr = audio["sample_rate"]; wav = audio["waveform"]

            # End the shot `hoff` frames early so the frame handed to the NEXT shot
            # isn't the literal last frame (which may catch an open, mid-word mouth
            # and make the next shot start "talking"). Drop the matching audio tail
            # so this shot's A/V stays aligned. Skipped if the shot is too short.
            # ...but ONLY when there IS a next shot. On the final shot the trim hands its
            # frames to nobody, so it just deletes the tail of the finished video -- on a
            # single-shot run that is the whole point of handoff_offset applied to the one
            # thing it cannot help (243f requested came back as 231 frames).
            if hoff and i < len(gens) - 1 and frames.shape[0] > hoff + 1:
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

        # --- the `latent` output ------------------------------------------------
        # The sampled latents, joined on the temporal axis. This is NOT the latent
        # form of `images`, and the difference is not cosmetic:
        #
        #   * trim_seam and handoff_offset cut DECODED frames. H3 compresses time,
        #     so one pixel frame is not one latent step and those cuts have no exact
        #     latent equivalent -- the seam frames trim_seam removes are still here.
        #   * the overlap fade and any post-pass upscale are pixel-space too.
        #
        # So decoding this yourself gives a slightly longer video with the seams
        # intact. On a SINGLE-shot run none of those apply and it is exact, which is
        # the case that matters for testing a latent upscaler.
        latent_note = ""
        latent_out = {"samples": _empty_av_latent(w, h, 5, fps)[0]["samples"]}
        if latent_chunks:
            try:
                if all(isinstance(c, list) and len(c) == 2 for c in latent_chunks):
                    vids = [c[0] for c in latent_chunks]
                    auds = [c[1] for c in latent_chunks]
                    # A mid-chain resolution backoff makes the shots un-concatenable.
                    if len({tuple(v.shape[1:2] + v.shape[3:]) for v in vids}) == 1:
                        latent_out = {"samples": comfy.nested_tensor.NestedTensor(
                            (torch.cat(vids, dim=2), torch.cat(auds, dim=-1)))}
                        if len(latent_chunks) > 1:
                            latent_note = (f" latent: {len(latent_chunks)} shot(s) joined on the "
                                           f"time axis -- PRE-trim, so it holds the seam frames "
                                           f"trim_seam drops from `images` and is longer by "
                                           f"{len(latent_chunks) - 1} frame(s)")
                        else:
                            latent_note = " latent: single shot, exact match for `images`"
                    else:
                        latent_out = {"samples": comfy.nested_tensor.NestedTensor(
                            (vids[-1], auds[-1]))}
                        latent_note = (" latent: shots differ in size after a resolution backoff, "
                                       "so only the LAST shot's latent is emitted")
            except Exception as e:
                latent_note = f" latent: could not be assembled ({type(e).__name__})"

        # Optional post-pass upscale of the finished frames (safe: any failure
        # falls back to lanczos / raw frames and never breaks the render).
        up_note = ""
        if upscale != "off":
            _deep_cleanup()
            all_frames, up_note = _upscale_frames(all_frames, upscale, upscale_model,
                                                  upscale_target_short_edge, upscale_batch)

        # Text overlays LAST -- after the upscale, so glyphs are rasterized at the
        # final pixel size instead of being interpolated up along with the picture.
        all_frames, ov_note = _overlay.apply_overlays(
            all_frames, fps, watermark_text, watermark_position, watermark_size,
            watermark_opacity, watermark_margin, intro_text, intro_seconds,
            intro_fade, intro_size, intro_position, overlay_font, overlay_stroke)

        script = "\n---\n".join(gens)
        actual = all_frames.shape[0] / fps
        uniform_len = len(set(shot_lens)) == 1
        shape_str = (f"{len(gens)} shot(s) x {shot_lens[0]}f (~{shot_lens[0] / fps:.1f}s each) "
                     f"= ~{sum(shot_lens) / fps:.1f}s" if uniform_len else
                     f"{len(gens)} shot(s), per-beat "
                     + ", ".join(f"{n}f/~{n / fps:.1f}s" for n in shot_lens)
                     + f" = ~{sum(shot_lens) / fps:.1f}s")
        vram_str = f"{total_gb:.1f}GB total / {resident_gb:.1f}GB weights / {free_gb:.1f}GB free" if total_gb else "VRAM unknown"
        # Say how many shots the trim actually touched: on a single-shot run it is none,
        # which explains the frame count instead of leaving it looking like a shortfall.
        hoff_str = (f" handoff -{hoff}f on {max(0, len(gens) - 1)} of {len(gens)} shot(s)"
                    f"{' (last shot keeps its tail)' if len(gens) else ''}." if hoff else "")
        # Say what was done about babble on non-dialogue shots, and what it cost. Both
        # states need reporting: muting is silent about the ambience it removes, and
        # NOT muting is silent about the babble it may leave in.
        n_silent = sum(1 for f in spk if not f)
        n_muted = sum(1 for f in muted_flags if f)
        if n_muted:
            audio_note = (f" {n_muted} of {len(gens)} shot(s) have no quoted dialogue and were AUDIO-MUTED "
                          f"(mute_nonspeech_audio) -- that also removes their generated ambience, so lay an "
                          f"ambient bed under the video in post, or untick it to keep H3's own")
        elif n_silent:
            audio_note = (f" {n_silent} of {len(gens)} shot(s) have no quoted dialogue: silenced in the prompt "
                          f"and soundscape only. If any of them still vocalize, tick mute_nonspeech_audio "
                          f"for a guaranteed fix")
        else:
            audio_note = ""
        # Which shots actually took the reference channel, and what they gave up for
        # it. Silence here would leave "why did shot 2 cut instead of continuing?"
        # unanswerable from the output alone.
        if ref_missing:
            ref_note_missing = (f" <Picture {','.join(str(n) for n in ref_missing)}> named in the prompt "
                                f"but no image is connected to that ref_image input -- the tag(s) were "
                                f"dropped from the text")
        else:
            ref_note_missing = ""
        if ref_list and ref_shots:
            kept = [n for n in range(1, len(gens) + 1) if n not in ref_shots]
            ref_note = (f" ref2va: {len(ref_list)} reference image(s) at '{ref_image_size}' on shot(s) "
                        f"{','.join(str(n) for n in ref_shots)} "
                        f"({'placed by <Picture N> tags' if tag_driven else f"ref_mode '{ref_mode}'"})"
                        + (f", ref_noise_aug {ref_noise_aug:.3f}" if ref_noise_aug is not None
                           and float(ref_noise_aug) < 0.999 else "")
                        + (f"; shot(s) {','.join(str(n) for n in kept)} keep the handoff" if kept
                           else "")
                        + (f"; shot(s) {','.join(str(n) for n in ref_keyframed)} carry the previous "
                           f"frame as a real KEYFRAME alongside their references, so a tag anchors "
                           f"rather than cuts" if ref_keyframed else "")
                        + (f"; the previous frame rides along as an extra reference on shot(s) "
                           f"{','.join(str(n) for n in ref_carried)} -- weaker than a keyframe, but "
                           f"ref_noise_aug below {KEYFRAME_SAFE_AUG:g} would soften a keyframe too "
                           f"(one aug covers every cond latent)" if ref_carried else "")
                        + ("" if (ref_keyframed or ref_carried or kept)
                           else ", so every cut between beats is a CUT, not a continuous take")
                        + ref_note_missing)
        elif ref_list:
            ref_note = (f" ref2va: {len(ref_list)} reference image(s) connected but ref_mode "
                        f"'{ref_mode}' applied them to no shot")
        else:
            ref_note = ""
        info = ((anchor_note + " ") if anchor_note else "") + \
               (f"{shape_str} at {w}x{h}; {all_frames.shape[0]} frames (~{actual:.1f}s actual). "
                f"decode {'tiled' if tiled else 'full'}. {vram_str}.{hoff_str}"
                + (" DIALOGUE MAY BE CUT OFF -- " + "; ".join(fit_warnings)
                   + ". Shorten the line, or pick a lower resolution tier to keep the duration."
                   if fit_warnings else "")
                + ((" subject-count guard ON ("
                    + ("sub-native resolution" if min(w, h) < 768 else "")
                    + ("; " if min(w, h) < 768 and lora_on else "")
                    + ("LoRA active -- count front-loaded so it binds before the scene"
                       if lora_on else "")
                    + ").") if count_subjects else "")
                + ((" " + preflight_txt.strip()) if preflight_txt else "")
                + (f"{latent_note}." if latent_note else "")
                + (f" SLA LoRA '{os.path.basename(str(sla_name))}' paired with sparse attention."
                   if sla_name and sparse_on else "")
                + (f" {beats_note}." if beats_note else "")
                + (f"{audio_note}." if audio_note else "")
                + (" EXPOSURE -- " + "; ".join(wardrobe_notes) + "."
                   if wardrobe_notes else "")
                + (f"{ref_note}." if ref_note else "")
                + (f" {fps_note}." if fps_note else "")
                + (f" {swap_note}." if swap_note else "")
                + (f" free VRAM/shot: {vram_trace}." if len(vram_trace) > 1 else "")
                + (f" {accel_note}." if accel_note else "")
                + (f" {ms_note}." if ms_note else "")
                + (f" {ln_note}." if ln_note else "")
                + (f" {up_note}." if up_note else "")
                + (f" {ov_note}." if ov_note else "")
                + (f" Adjusted: {'; '.join(backoff)}." if backoff else ""))
        # frames_per_shot is a single INT for a now-variable series: report the LONGEST
        # shot, which is what a downstream consumer must be able to hold.
        return (all_frames, {"waveform": all_audio, "sample_rate": sr}, info, script,
                max(shot_lens), all_frames.shape[0], len(gens), round(actual, 2),
                float(fps), int(fps), latent_out)


# REF2VA registers under its OWN key. The FL2VA pack one directory up keeps
# "H3LongVideosFL2VA" and the legacy "H3LongVideosV1" alias; ComfyUI builds one
# flat registry, so repeating either here would silently overwrite that node --
# same name in the search, no way to tell which copy a workflow is running.
# ONE node, three registration keys. ComfyUI stores the key verbatim in every saved
# workflow, so all three must keep resolving or existing graphs load as red "missing
# node" boxes: "H3LongVideosV1" is the original name, "H3LongVideosFL2VA" the rename,
# and "H3LongVideosREF2VA" the separate reference node that has now been folded in.
# They are aliases onto the same class -- there is no second implementation.
NODE_CLASS_MAPPINGS = {
    "H3LongVideos": H3LongVideos,
    "H3LongVideosFL2VA": H3LongVideos,        # do not remove
    "H3LongVideosV1": H3LongVideos,           # do not remove
    "H3LongVideosREF2VA": H3LongVideos,       # do not remove
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LongVideos": "H3 Long Videos (FL2VA + REF2VA)",
    "H3LongVideosFL2VA": "H3 Long Videos (FL2VA + REF2VA)",
    "H3LongVideosV1": "H3 Long Videos (FL2VA + REF2VA)",
    "H3LongVideosREF2VA": "H3 Long Videos (FL2VA + REF2VA)",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
