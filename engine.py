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

# ---------------------------------------------------------------------------
# Vocabulary. One table per KIND of thing, and a word appears in exactly one of
# them. The old engine had "chain" as a noun in one list and a verb in another,
# which let a chain-link fence satisfy both halves of a rule by itself.
# ---------------------------------------------------------------------------

# Hardware, and the part each kind holds. The part is a property of the ITEM --
# this is the table whose absence produced "holding the neck behind the back".
HARDWARE = (
    (r"hand\s?cuffs?", "handcuffs", "wrists"),
    (r"leg\s?irons?", "leg irons", "ankles"),
    (r"ankle\s+(?:cuffs?|chains?|straps?)", "ankle cuffs", "ankles"),
    (r"shackles?", "shackles", "ankles"),
    (r"manacles?", "manacles", "wrists"),
    (r"zip\s?ties?|cable\s?ties?", "zip ties", "wrists"),
    (r"collars?|chokers?", "collar", "neck"),
    (r"leash(?:es)?", "leash", "neck"),
    (r"gags?", "gag", "mouth"),
    (r"blindfolds?", "blindfold", "eyes"),
    (r"harness(?:es)?", "harness", "body"),
    (r"spreader\s+bars?", "spreader bar", "ankles"),
    (r"straitjackets?", "straitjacket", "arms"),
    (r"ropes?|cords?|twine", "rope", "wrists"),
    (r"straps?", "straps", "wrists"),
    (r"chains?", "chain", "wrists"),
    (r"cuffs?", "cuffs", "wrists"),
    (r"tape", "tape", "wrists"),
)
# Material and colour survive because they decide what the thing looks like:
# "steel collar" must not come back as "collar" two shots later.
_ADJ = (r"(?:steel|iron|metal|leather|nylon|plastic|rubber|rope|chrome|brass|"
        r"black|silver|white|red|brown|padded|heavy|thin|short|long|thick|"
        r"duct|packing|electrical|zip)")

# Where a limb is held. These all describe the ARMS -- that is why a limb
# position may never be attached to a collar.
POSITIONS = (
    (r"behind\s+(?:her|his|their|the)\s+backs?", "behind the back"),
    (r"(?:above|over)\s+(?:her|his|their|the)\s+heads?|overhead", "above the head"),
    (r"in\s+front\s+of\s+(?:her|his|their)\s+(?:body|chest|waist)",
     "in front of the body"),
    (r"(?:out\s+)?to\s+the\s+sides?|spread\s+wide", "out to the sides"),
    (r"at\s+(?:her|his|their|the)\s+waists?", "at the waist"),
)

# Fixed things hardware can be anchored to. A thing you cannot pick up and walk
# away with.
ANCHORS = (r"walls?|floors?|grounds?|ceilings?|pillars?|columns?|posts?|rails?|"
           r"railings?|bars?|rings?|hooks?|pipes?|radiators?|beams?|girders?|"
           r"struts?|stakes?|eye\s?bolts?|brackets?|cages?|fences?|grates?|"
           r"grilles?|bed\s?frames?|bed\s?posts?|headboards?|bedsteads?|beds?|"
           r"bunks?|benches?|chairs?|tables?|desks?|ladders?|anchors?|loops?")

# Verbs, as VERBS. Participles and -ing forms are unambiguous. The -s forms are
# also plural nouns, so they carry a lookbehind: "the guard chains her collar" is
# a verb and "the chains on the floor" is not.
_DET = (r"(?<!\bthe\s)(?<!\ba\s)(?<!\ban\s)(?<!\bthese\s)(?<!\bthose\s)"
        r"(?<!\btwo\s)(?<!\bsome\s)(?<!\bmore\s)(?<!\bhis\s)(?<!\bher\s)")
APPLY_VERB = (
    r"(?:handcuffed|cuffed|chained|shackled|manacled|locked|padlocked|fastened|"
    r"secured|tethered|bound|tied|strapped|clipped|hooked|bolted|attached|"
    r"anchored|leashed|roped|gagged|blindfolded|collared|taped|trussed|lashed|"
    r"buckled|fettered|"
    r"handcuffing|cuffing|chaining|locking|fastening|securing|tethering|tying|"
    r"strapping|clipping|bolting|attaching|gagging|blindfolding|collaring|"
    r"taping|buckling|binding|shackling|"
    # The particle can sit four words from its verb, exactly as it can for
    # garments: "puts the cuffs on her" is the ordinary way to write it, and
    # requiring "puts on" adjacent read that as no application at all.
    r"(?:puts?|putting|slips?|slipped|snaps?|snapped|clicks?|clicked|clamps?|"
    r"clamped)(?:\s+\S+){0,4}?\s+(?:on|onto|around|shut|closed)|"
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

# Rooms. A place is somewhere a scene can BE. "door" is not on this list and must
# not go back on it: a door is a thing inside a room, and putting it here moved
# the camera into a door whenever anybody looked at one.
PLACES = (r"hallway|hall|corridor|passage|landing|stairwell|staircase|stairs|"
          r"bedroom|bathroom|washroom|kitchen|living\s+room|lounge|dining\s+room|"
          r"study|office|garage|basement|cellar|attic|loft|porch|veranda|"
          r"garden|yard|driveway|street|alley|car\s?park|lobby|foyer|doorway|"
          r"cell|corridor|warehouse|barn|shed|van|car|truck|room")
# A room is usually described, not just named -- "the tiled bathroom", "the long
# hallway". Up to three adjectives, non-greedy so the NEAREST room still wins.
_ROOM_MOD = (r"(?:(?!(?:of|the|an?|and|or|to|in|into|from|with|on|at|by|for|her|"
             r"his|their|its|my|our|your)\b)[A-Za-z][A-Za-z-]*\s+){0,3}?")

_GARMENT = (r"shirt|blouse|top|t-?shirt|vest|jumper|sweater|hoodie|cardigan|"
            r"jacket|coat|dress|skirt|trousers|pants|jeans|shorts|leggings|"
            r"tights|socks|stockings|shoes|boots|heels|gloves|scarf|hat|cap|"
            r"bra|knickers|panties|underwear|briefs|nappy|diaper|robe|gown|"
            r"apron|uniform|overalls|dungarees|belt")

# Garment verbs. A garment has three states -- on, off, pulled aside -- and the
# shot that CHANGES one has to say both ends of the change, or the model is free
# to open the shot with it already done. Reported as a diaper turning into shorts
# a beat before the beat that put the shorts on.
# THE PARTICLE CAN BE FOUR WORDS AWAY. English puts the object between the verb
# and its particle as happily as after it: "takes off her shorts" and "takes her
# blue shorts off" are the same act, and "puts her blue shorts back on" has four
# words in the gap. Requiring them adjacent read that last one as no change at
# all, so the garment silently stayed off.
_GAP = r"(?:\s+\S+){0,4}?\s+"
TAKES_OFF = (r"(?:takes?|took|taking|pulls?|pulled|peels?|peeled|strips?|"
             r"stripped|shrugs?|slips?|slipped|steps?|gets?|got|kicks?|"
             r"kicked)" + _GAP + r"(?:off|out\s+of)\b"
             r"|\b(?:removes?|removed|removing|discards?|discarded|"
             r"undresses|undressed|unbuttons?|unzips?|unzipped)")
PUTS_ON = (r"(?:puts?|putting|pulls?|pulled|slips?|slipped|tugs?|tugged|"
           r"steps?|stepped|climbs?|climbed|gets?|got|wriggles?)" + _GAP +
           r"(?:on|into|back\s+on)\b"
           r"|\b(?:dresses?\s+in|dressed\s+in|buttons?|zips?\s+up|fastens?)")
DISPLACES = (r"(?:pulls?|pulled|pushes?|pushed|tugs?|tugged|hikes?|hiked|"
             r"rolls?|rolled|lifts?|lifted|yanks?|yanked|shoves?|shoved)"
             + _GAP + r"(?:aside|up|down|open)\b")

POSTURES = (
    (r"kneels?|kneeling|knelt|on\s+(?:her|his|their)\s+knees", "kneeling"),
    (r"sits?|sitting|sat|seated", "sitting"),
    (r"lies?|lying|lay|laid\s+(?:down|out)|on\s+(?:her|his|their)\s+back",
     "lying down"),
    (r"stands?|standing|stood|gets?\s+up|got\s+up|rises?|rose", "standing"),
    (r"crouch(?:es|ing|ed)?|squats?|squatting", "crouching"),
    (r"bent\s+over|bends?\s+over|leans?\s+over", "bent over"),
    (r"curled\s+up|foetal|fetal", "curled up"),
)


def _rx(pattern):
    return re.compile(pattern, re.I)


_SPOKEN_SPAN = re.compile(r"<d>.*?</d>|[\"“][^\"“”]{1,400}?[\"”]",
                          re.S)


def _outside_speech(text):
    """The beat with everything anybody SAYS taken out.

    What a character says is not stage direction. The commonest thing to talk
    about is something that is NOT in the room -- "McKenna where are you?" is how
    absence gets written -- and reading a spoken name as a staged one put a full
    description of the missing person into the shot, so the model drew her.

    Both markers, because both exist in the pipeline: <d> once mark_dialogue has
    run, plain quotes before it."""
    return _SPOKEN_SPAN.sub(" ", text or "")


_HW_ONE = _rx(r"\b(" + _ADJ + r"(?:\s+" + _ADJ + r")?\s+)?("
              + "|".join(p for p, _n, _pt in HARDWARE) + r")\b")
_POSITION = [(_rx(r"\b" + p + r"\b"), name) for p, name in POSITIONS]
_ANCHOR_AT = _rx(r"\bto\s+(?:the|a|an|her|his|their)\s+(" + ANCHORS + r")\b")
_APPLY = _rx(r"\b(?:" + APPLY_VERB + r")\b")
_RELEASE = _rx(r"\b(?:" + RELEASE_VERB + r")\b")
_PLACE_IN = _rx(r"\b(?:in|into|inside|through|down|along|across|to|onto|at)\s+"
                r"(?:the|a|an|her|his|their)\s+(" + _ROOM_MOD + r"(?:" + PLACES
                + r"))\b")
_PLACE_WORD = _rx(r"\b(?:" + PLACES + r")\b")
_GARMENT_ONE = _rx(r"\b(" + _ADJ + r"(?:\s+" + _ADJ + r")?\s+)?(" + _GARMENT + r"s?)\b")
_POSTURE = [(_rx(r"\b(?:" + p + r")\b"), name) for p, name in POSTURES]
_TAKES_OFF = _rx(r"\b" + TAKES_OFF + r"\b")
_PUTS_ON = _rx(r"\b" + PUTS_ON + r"\b")
_DISPLACES = _rx(r"\b" + DISPLACES + r"\b")
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
    out, seen = [], {}
    for m in _HW_ONE.finditer(text or ""):
        adj, noun = (m.group(1) or "").strip(), m.group(2)
        canon, part = next((n, pt) for p, n, pt in HARDWARE
                           if re.fullmatch(p, noun, re.I))
        written = f"{adj} {noun}".strip().lower()
        if canon in seen:
            # Same thing, described better the second time: keep the fuller
            # wording. "collar" then "steel collar" is one collar.
            i = seen[canon]
            if len(written) > len(out[i][2]):
                out[i] = (canon, part, written, out[i][3])
            continue
        seen[canon] = len(out)
        out.append((canon, part, written, m.start()))
    return out


def hardware_in(text):
    """Every piece of hardware named, as (canonical, part, as-written)."""
    return [(c, p, w) for c, p, w, _at in hardware_spans(text)]


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
        # The fastening verb has to be in THIS clause, not somewhere earlier in
        # the paragraph.
        clause = re.split(r"[.;!?]", t[:m.start()])[-1]
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
    m = _PLACE_IN.search(text or "")
    if not m:
        return ""
    got = re.sub(r"\s+", " ", m.group(1).lower()).strip()
    # A BARE "room" NAMES NOWHERE. "Ana walks into the room" says she goes
    # inside, not which room -- and taking it as a place produced "The shot is
    # in the room, not the room the scene text names", which contradicts itself
    # in one sentence. Modified, it is a real place: "the far room", "the back
    # room" and "the next room" all distinguish themselves from where we were.
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


def posture_in(text):
    """The posture this beat puts a body in. '' when it does not."""
    for rx, name in _POSTURE:
        if rx.search(text or ""):
            return name
    return ""


# ---------------------------------------------------------------------------
# STATE. One object knows what is true, and everything a shot says is rendered
# from it -- so two clauses cannot contradict each other, because there is only
# one place a fact lives.
# ---------------------------------------------------------------------------

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
        self.anchor = anchor
        self.applied_in = applied_in
        # RIGID metal keeps its shape. Steel decoded and re-encoded once a shot
        # has nothing in the text holding its links to a size, and it creeps --
        # a chain grows slack, cuffs turn into bracelets. Soft goods do not need
        # this and must not be given it: rope is tied, not held rigid.
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


class Person:
    __slots__ = ("name", "hardware", "worn", "removed", "displaced",
                 "posture", "place")

    def __init__(self, name):
        self.name = name
        self.hardware = {}      # canonical -> Restraint
        self.worn = []          # garments on the body, as written
        self.removed = []       # garments taken off
        self.displaced = []     # pulled aside but still on
        self.posture = ""
        self.place = ""

    def restrained(self):
        return bool(self.hardware)

    def __repr__(self):
        return (f"Person({self.name!r},hw={list(self.hardware)},"
                f"posture={self.posture!r})")


class SceneState:
    """What is true right now, and what each shot is told because of it."""

    def __init__(self, place=""):
        self.place = place
        # The room the SCENE paragraph names. A shot only needs telling where it
        # is once the film has moved somewhere else.
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
            if canon not in p.hardware:
                # applied_in = 0, so this never reads as "goes on during this
                # shot" -- shots are numbered from 1.
                p.hardware[canon] = Restraint(written or canon, part,
                                              position_in(description or ""),
                                              anchor_in(description or ""), 0)
        for g in garments_in(description or ""):
            if _garment_key(g) not in [_garment_key(x) for x in p.worn]:
                p.worn.append(g)
        return p

    # -- reading a beat ----------------------------------------------------
    def read(self, beat, cast=(), shot=0):
        """Update the state from one beat, and report what CHANGED.

        The change matters separately from the result: the shot that puts the
        cuffs on has to say both ends of that, and every shot after it says only
        the result."""
        self.shot = shot
        beat = beat or ""
        changed = {"applied": [], "released": [], "moved_to": "", "posture": {},
                   "removed": [], "worn": [], "displaced": []}

        here = place_in(beat)
        if here:
            self.place = here
            changed["moved_to"] = here

        # IN SENTENCE ORDER, not cast order. Ordering by the sheet put "Ana"
        # before "Guard" in "The guard handcuffs Ana", so the cuffs went on the
        # guard -- the agent wearing what he is applying, which is the invented
        # second figure all over again.
        #
        # ...and read from the STAGED half only. A name inside a line of dialogue
        # is being said, not staged: "Dan says: 'McKenna, put the cuffs on'"
        # would otherwise hand McKenna hardware in a shot she is not in.
        staged = _outside_speech(beat)
        who = sorted(
            (n for n in cast
             if re.search(r"\b" + re.escape(n) + r"\b", staged, re.I)),
            key=lambda n: re.search(r"\b" + re.escape(n) + r"\b", staged,
                                    re.I).start())
        subject = who[0] if who else next(iter(list(self.people) or list(cast)
                                               or [""]))

        hw = hardware_in(beat)
        applying = bool(hw) and bool(_APPLY.search(beat)) and not _RELEASE.search(beat)
        releasing = bool(_RELEASE.search(beat))

        if applying:
            wearer = _wearer(beat, who, subject)
            p = self.person(wearer)
            # MODIFIERS BIND TO THE NEAREST ITEM. "handcuffs her wrists behind
            # her back and locks a steel collar around her neck, chained to the
            # wall" carries two modifiers and two items; giving both modifiers
            # to both items produced handcuffs chained to a wall they were never
            # near, and a collar held behind a back.
            spans = hardware_spans(beat)
            for canon, part, written, at in spans:
                p.hardware[canon] = Restraint(
                    written or canon, part,
                    _nearest(position_spans(beat), at, spans),
                    _nearest(anchor_spans(beat), at, spans), shot)
                changed["applied"].append((wearer, p.hardware[canon]))
            anc = anchor_in(beat)
            # A chain named beside another item is that item's TETHER, not a
            # second restraint holding its own wrists. Fold it in, or the shot
            # says a chain holds the wrists while a collar holds the neck and
            # the model has two things to draw where there is one.
            if anc and len(p.hardware) > 1 and "chain" in p.hardware:
                for canon, r in p.hardware.items():
                    if canon != "chain" and not r.anchor:
                        r.anchor = anc
                p.hardware.pop("chain", None)
                changed["applied"] = [(w, r) for w, r in changed["applied"]
                                      if r.item != "chain"]
        elif releasing:
            # Whoever is actually wearing it. "The guard unlocks the handcuffs"
            # names only the agent, and taking the subject there tried to
            # release hardware from the man holding the key.
            held = [n for n, q in self.people.items() if q.restrained()]
            wearer = next((n for n in who if n in held),
                          held[0] if len(held) == 1 else subject)
            p = self.person(wearer)
            named = [c for c, _pt, _w in hw if c in p.hardware]
            if named:
                for canon in named:
                    changed["released"].append((wearer, p.hardware.pop(canon)))
            elif re.search(r"\b(?:them|it|her|him|everything|all\s+of\s+it)\b",
                           beat, re.I):
                # "the guard releases her" names no item, so all of it comes off.
                while p.hardware:
                    changed["released"].append((wearer, p.hardware.popitem()[1]))

        # Garments. The verb decides which way the change runs, and the item has
        # to be named -- a bare "she undresses" says nothing about which garment,
        # and guessing is how a garment came off a beat before the beat that
        # took it off.
        wearer_g = _wearer(beat, who, subject) if len(who) > 1 else subject
        if wearer_g:
            p = self.person(wearer_g)
            for g in garments_in(beat):
                key = _garment_key(g)
                if _TAKES_OFF.search(beat):
                    if key not in [_garment_key(x) for x in p.removed]:
                        p.removed.append(g)
                        changed["removed"].append((wearer_g, g))
                    p.worn = [x for x in p.worn if _garment_key(x) != key]
                    p.displaced = [x for x in p.displaced
                                   if _garment_key(x) != key]
                elif _PUTS_ON.search(beat):
                    if key not in [_garment_key(x) for x in p.worn]:
                        p.worn.append(g)
                        changed["worn"].append((wearer_g, g))
                    p.removed = [x for x in p.removed if _garment_key(x) != key]
                    p.displaced = [x for x in p.displaced
                                   if _garment_key(x) != key]
                elif _DISPLACES.search(beat):
                    if key not in [_garment_key(x) for x in p.displaced]:
                        p.displaced.append(g)
                        changed["displaced"].append((wearer_g, g))

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
                # WHOSE. With more than one person described, a hold that does
                # not say whose hardware it is describes cuffs on wrists
                # belonging to nobody -- and the model draws a body to own them.
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

        # GARMENTS. The shot that changes one says BOTH ENDS of the change --
        # where it starts and where it finishes -- because a shot told only the
        # result is free to open with the result already true, which is a
        # garment coming off a beat before the beat that takes it off.
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

        # WHERE. A journey moves the film, and every shot after it is in the new
        # room -- the scene paragraph still names the old one, and without this
        # the walk down the corridor arrives back in the room it left.
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


def _wearer(beat, who, fallback):
    """Who the hardware goes ON. The agent is not the wearer.

    "The guard cuffs Ana" puts them on Ana; "Ana is cuffed by the guard" puts
    them on Ana too, and naive sentence order gets the second one backwards.
    Describing hardware on somebody the text never put it on gives the model
    wrists belonging to nobody, which is how a second figure gets invented to
    own them."""
    if len(who) < 2:
        return who[0] if who else fallback
    passive = re.search(r"\bby\s+(?:the\s+)?(\w+)", beat or "", re.I)
    agent = None
    if passive:
        agent = next((n for n in who
                      if n.lower() == passive.group(1).lower()), None)
    if agent is None:
        agent = who[0]          # active voice: the one doing it comes first
    return next((n for n in who if n != agent), fallback)


def held_part_of(items):
    """The body part these items hold, as a plural noun for a sentence.

    A limb position describes the ARMS, so this answers "wrists" for anything
    that holds them and defers to the item otherwise -- a collar's position is
    never a limb position, and the constructor already refuses to give it one."""
    text = " ".join(items or [])
    for pat, _n, part in HARDWARE:
        if re.search(r"\b(?:" + pat + r")\b", text, re.I) and part not in ("wrists",):
            return part
    return "wrists"


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
        for canon, _part, _w, _at in hardware_spans(b):
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
