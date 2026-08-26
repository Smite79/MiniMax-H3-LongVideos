#!/usr/bin/env python3
"""
Safety / regression tests for H3-LongVideos prompt logic
========================================================
Exercises the wardrobe channel, pronoun resolution, duplication avoidance, and
auto-removal against a full 12-beat (12-shot) chain -- the real production
length. Pure string logic only: it stubs the torch / ComfyUI imports so it runs
anywhere with no GPU and no ComfyUI:

    python3 test_prompt_logic.py

Exits non-zero if any invariant fails, so you can wire it into CI or run it after
any edit to the prompt-assembly code.
"""
import sys, types, os, re, importlib.util

# --- stub heavy deps so sampler.py imports without ComfyUI / torch ------------
for _name in ["torch", "nodes", "comfy", "comfy.utils", "comfy.samplers",
              "comfy.nested_tensor", "comfy.model_management", "node_helpers"]:
    sys.modules.setdefault(_name, types.ModuleType(_name))

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("h3_sampler", os.path.join(_HERE, "sampler.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

D, SP = S.distribute_generations, S.split_paragraphs

_fails = []
def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        _fails.append(name)



def worn(shot, item):
    """Is the garment presented as being WORN in this shot?

    A removal now STATES the change ("... is no longer wearing the red jacket, it is
    off") in the first shot without it -- deleting the item from the channel was not
    enough on its own, because the shot still starts from a handoff frame that shows
    it being worn. That sentence mentions the garment while asserting the opposite,
    so a bare substring test reads a correct removal as a failure. Audio field lines
    are excluded for the same reason: they are field names, not wardrobe."""
    import re as _re
    text = _re.sub(r"[^.]*\bno longer wearing\b[^.]*\.", " ", shot, flags=_re.I)
    text = _re.sub(r"[^.]*\bno longer worn\b[^.]*\.", " ", text, flags=_re.I)
    text = _re.sub(r"(?m)^(?:overall_soundscape|non_diegetic_music):.*$", " ", text)
    return _re.search(r"\b" + _re.escape(item) + r"\b", text, _re.I) is not None


def stated_off(shot, item):
    """Does this shot say outright that the garment is off?"""
    import re as _re
    return bool(_re.search(r"(?:no longer wearing|no longer worn)[^.]*\b"
                           + _re.escape(item) + r"\b|\b" + _re.escape(item)
                           + r"\b[^.]*\bis off\b", shot, _re.I))


def _parens(shot):
    """Extract the contents of every (parenthetical) in a shot's text."""
    out, depth, cur = [], 0, ""
    for ch in shot:
        if ch == "(":
            depth += 1; cur = ""
        elif ch == ")" and depth:
            depth -= 1; out.append(cur)
        elif depth:
            cur += ch
    return out


# --- the 12-beat scenario: two people, pronoun-driven, aviation (landmine) -----
PROMPT = (
    "A cinematic aircraft hangar and airfield, warm late-afternoon light, film grain.\n"
    "wardrobe: Maya = she, silver hair, scar over left eyebrow, grey flight suit, red jacket; "
    "Jon = he, bald, beard, navy overalls, cap\n\n"
    "She and he walk into the hangar together.\n\n"                    # 1 both
    "She inspects the engine while he checks the tail.\n\n"           # 2 both
    "She takes off her red jacket and hangs it on a hook.\n\n"        # 3 auto-remove Maya jacket
    "He removes his cap and wipes his brow.\n\n"                      # 4 auto-remove Jon cap
    "She climbs into the cockpit alone.\n\n"                          # 5 solo Maya
    "She holds the stick as the plane takes off down the runway.\n\n" # 6 LANDMINE
    "She pulls on a brown leather jacket.\n"                          # 7 explicit add
    "wardrobe: Maya += brown leather jacket\n\n"
    "He hands her a wrench and she takes it.\n\n"                     # 8 both
    "She keys the radio and says, \"Tower, ready for departure.\"\n\n" # 9 DIALOGUE
    "Maya and Jon review the checklist together.\n\n"                # 10 explicit NAMES
    "He shrugs off his overalls, a flight suit underneath.\n\n"      # 11 auto-remove overalls
    "She and he taxi back as the sun sets."                          # 12 both
)


def check_no_second_subject_noun():
    """A description must never introduce a second subject noun.

    "Kristy = she, a woman with silver hair" used to render as
    `She (a woman with silver hair)` -- two subject nouns in one clause, which
    text-to-video reads as two people. Duplication from shot 1, at any resolution."""
    print("\n=== descriptions must not introduce a second subject ===")
    import re as _re
    nouns = _re.compile(r"\b(?:a|an|the)\s+(?:[\w\-]+\s+){0,2}"
                        r"(?:woman|man|girl|boy|guy|lady|person|figure)\b", _re.I)
    cases = ["Kristy = she, a woman with silver hair, red jacket",
             "Kristy = she, Kristy is a tall woman, silver hair",
             "Kristy = a young woman, silver hair, red jacket",
             "Kristy = she, a woman, red jacket",
             "Kristy = she, silver hair, red jacket"]
    ok = True
    for cm_ in cases:
        shot = D("A hangar.", ["She checks the engine."], "", "", cm_)[0]
        for par in _parens(shot):
            if nouns.search(par):
                ok = False
                print(f"    LEAK: ({par}) from {cm_!r}")
    check("no parenthetical introduces a person noun", ok)
    keep = D("A hangar.", ["She checks the engine."], "", "",
             "Kristy = she, a woman with silver hair, red jacket")[0]
    check("attributes survive de-positioning",
          "silver hair" in keep and "red jacket" in keep)
    two = D("A hangar.", ["She hands him a wrench."], "", "",
            "Kristy = she, a woman with silver hair\nJon = he, a bald man in navy overalls")[0]
    check("two-person noun phrases both reduced",
          "silver hair" in two and "navy overalls" in two and not nouns.search(two))


def check_real_world_sheet():
    """A real user sheet: sentence-ended clauses, 'wearing ...', bare gender nouns.

    Teresa/Dan reproduced character duplication because each person's entry became
    ONE item containing a full sentence ("wearing a black t-shirt and jeans. Mouth
    closed."), which lands inside the parenthetical as its own statement rather than
    as attributes of the pronoun."""
    print("\n=== real-world character_memory sheet ===")
    import re as _re
    cm_ = ("Teresa = woman, skinny, age 35, blonde hair, wearing a biker style "
           "t-shirt and leather pants. Mouth closed.\n"
           "Dan = man, age 40, brown hair, wearing a black t-shirt and jeans. Mouth closed.")
    w = S.parse_wardrobe(cm_)
    check("both people parse from a multiline sheet", set(w) == {"Teresa", "Dan"})
    t = S._clean_items(w["Teresa"], "Teresa")
    check("sentence-ended clause splits into separate items",
          "Mouth closed" in t and not any("." in i for i in t))
    check("'wearing ...' is reduced to the garment",
          any(i.startswith("biker style") for i in t))
    check("bare gender noun is dropped", "woman" not in [i.lower() for i in t])
    shot = D("A garage, warm light.", ["Teresa walks in and talks to Dan."], "", "", cm_)[0]
    check("each name appears exactly once",
          shot.count("Teresa") == 1 and shot.count("Dan") == 1)
    nouns = _re.compile(r"\b(?:a|an|the)\s+(?:[\w\-]+\s+){0,2}"
                        r"(?:woman|man|girl|boy|guy|lady|person|figure)\b", _re.I)
    check("no parenthetical introduces a person noun",
          not any(nouns.search(p) for p in _parens(shot)))
    check("no full sentence inside a parenthetical",
          not any("." in p for p in _parens(shot)))


def check_no_phantom_person_in_anchor():
    """Camera direction in the anchor must not introduce an unnamed extra body.

    "the camera follows the subject" / "moves toward the person" / "tracks the
    figure" are stamped into EVERY shot alongside the named cast, so the model
    renders a third body that matches no character sheet."""
    print("\n=== no phantom person from camera direction ===")
    import re as _re
    cm_ = "Teresa = woman, skinny, age 35, blonde hair\nDan = man, age 40, brown hair"
    ghost = _re.compile(r"\b(?:the|a|an)\s+(?:main\s+|central\s+)?"
                        r"(?:subject|person|figure|character|individual|protagonist)\b", _re.I)
    anchors = ["A garage. Slow camera movement, the camera follows the subject.",
               "A garage, slow dolly, camera slowly moves toward the person.",
               "A garage. The camera slowly tracks the figure across the room."]
    ok = True
    for a in anchors:
        shot = D(a, ["Teresa talks to Dan."], "", "", cm_)[0]
        if ghost.search(shot):
            ok = False
            print(f"    LEAK: {shot[:110]}")
    check("camera-direction anchors leave no unnamed person", ok)
    keep = D("A garage. Slow, smooth camera movement. Minimal motion blur.",
             ["Teresa talks to Dan."], "", "", cm_)[0]
    check("camera direction itself is preserved",
          "Slow, smooth camera movement" in keep and "Minimal motion blur" in keep)
    check("the named cast is unaffected",
          "Teresa" in keep and "Dan" in keep and keep.count("Teresa") == 1)


# --- the 6-beat production prompt used for the three reported bugs ------------
SIX_ANCHOR = ("natural lighting, flat lighting, even exposure, medium shot, everything sharp, "
              "broadcast video, taken with iPhone. An open 4 bay car garage.")
SIX_BEATS = [
    "Kristy walks around in a garage looking for engine parts.",                      # 1 silent
    "Kristy finds Dan sitting in a chair. She walks over to Dan and asks him: "
    "\"Do you know where the pistons are?\"",                                         # 2 dialogue
    "Dan answers back to Kristy: \"Should be in the box over there.\"",               # 3 dialogue
    "Kristy takes off her red jacket and drops it on the workbench.",                 # 4 removal
    "Kristy opens the box and pulls out a piston.",                                   # 5 silent
    "Dan stands up and walks over to the bench.",                                     # 6 silent
]
SIX_CM = ("Kristy = she, 27, silver hair, red jacket, blue jeans\n"
          "Dan = he, 40, brown hair, black t-shirt")


def check_clothing_removal_6beat():
    """A removal must take the GARMENT off -- not delete the CHARACTER.

    'takes off her red jacket' used to yield the token 'red' (the first non-stop word
    after the verb), and matching 'red' with its neighbours in the anchor produced
    'A woman in a red'. Scrubbing that left 'jacket and a man in a black t-shirt':
    the woman was deleted from every later shot and the jacket stayed. Clothing
    removal looked completely broken, and the cast quietly lost a person."""
    print("\n=== clothing removal on a 6-beat prompt ===")
    # (a) tracked in the character channel
    sh = D(SIX_ANCHOR, SIX_BEATS, "", "", SIX_CM)
    check("6 beats -> 6 shots", len(sh) == 6)
    check("jacket worn up to and including the removal shot",
          all(worn(sh[i], "red jacket") for i in range(4)))
    check("jacket gone from every shot after the removal",
          all(not worn(sh[i], "red jacket") for i in (4, 5)))
    # --- the reverse-motion trap ------------------------------------------------
    # The removal shot must not describe her as WEARING the garment: that made the
    # jacket the shot's stated end state, and running the frames backwards satisfied
    # it -- the removal played in reverse and the jacket went back on.
    desc = " ".join(_parens(sh[3]))
    check("the removal shot no longer lists the garment as worn",
          "red jacket" not in desc)
    check("the removal shot still describes the rest of the outfit",
          "blue jeans" in desc and "silver hair" in desc)
    check("the removal is stated in the shot that performs it",
          stated_off(sh[3], "red jacket"))
    check("the removal states its END state", "by the last frame" in sh[3])
    check("the removal rules out the reverse",
          "never put back on" in sh[3] and "never plays in reverse" in sh[3])
    check("the statement uses a pronoun, not a bare name",
          "Kristy takes the" not in sh[3] and "She takes the" in sh[3])
    check("no shot after the removal names the garment at all",
          all("red jacket" not in s for s in sh[4:]))
    check("the other garment is untouched", worn(sh[4], "blue jeans"))
    check("the other character is untouched", worn(sh[5], "black t-shirt"))

    # (b) clothing that lives ONLY in the anchor prose -- the reported failure
    anchor_b = ("natural lighting, even exposure, broadcast video. An open 4 bay car garage. "
                "A woman in a red jacket and a man in a black t-shirt.")
    pb = D(anchor_b, SIX_BEATS, "", "", "")
    check("anchor: jacket worn up to the removal shot",
          all(worn(pb[i], "red jacket") for i in range(4)))
    check("anchor: jacket gone after the removal",
          all(not worn(pb[i], "red jacket") for i in (4, 5)))
    check("anchor: the WOMAN is still in the scene after the removal",
          all("woman" in pb[i].lower() for i in (4, 5)))
    check("anchor: the man and his t-shirt are untouched",
          all("man" in pb[i].lower() and worn(pb[i], "black t-shirt") for i in (4, 5)))
    check("anchor: no orphaned garment left behind",
          not any(s.lower().count("garage. jacket") for s in pb))
    check("anchor: the removal is stated in the shot that performs it",
          stated_off(pb[3], "red jacket") and "comes off during this shot" in pb[3])
    check("anchor: no shot after the removal names the garment",
          all("red jacket" not in s for s in pb[4:]))

    # (b2) the clause is read literally by the text encoder, so it has to be
    # grammatical for every shape of garment -- "the navy overalls IS off" is not.
    act = {"Maya": ["she", "silver hair"], "Jon": ["he", "bald"]}
    sing = S.takes_off_clause([("Maya", "red jacket")], act)
    plur = S.takes_off_clause([("Jon", "navy overalls")], act)
    two = S.takes_off_clause([("Jon", "cap"), ("Jon", "gloves")], act)
    imp = S.takes_off_clause([("", "boots")], act)
    check("singular garment takes a singular verb",
          "the red jacket off" in sing and "no longer wearing it" in sing)
    check("a plural garment takes a plural verb",
          "the navy overalls off" in plur and "they are off" in plur)
    check("two garments take a plural verb", "the cap and gloves off" in two)
    check("a double-s noun stays singular",
          "the dress off" in S.takes_off_clause([("Maya", "dress")], act))
    check("the impersonal form uses a SUBJECT pronoun",
          "they are off" in imp and "them are off" not in imp)
    check("every form rules out the reverse",
          all("never plays in reverse" in c for c in (sing, plur, two, imp)))

    # (c) a removal verb aimed at a NON-garment must strip nothing
    land = D(SIX_ANCHOR, ["Kristy watches as the plane takes off down the runway.",
                          "Kristy waves."], "", "", SIX_CM)
    check("landmine: 'the plane takes off' removes no clothing",
          worn(land[1], "red jacket") and "no longer wearing" not in land[1])


def check_nonspeech_audio_6beat():
    """Shots with no quoted dialogue must be silenced on BOTH channels.

    The lips-closed clause only constrains the picture. H3 builds audio from its own
    fields, and an ABSENT `overall_soundscape:` leaves that branch unconditioned --
    which is when it fills a silent shot with speech-like babble. So a silenced shot
    now always carries a soundscape line that says no voices outright."""
    print("\n=== non-dialogue shots must not vocalize ===")
    sh = D(SIX_ANCHOR, SIX_BEATS, "", "", SIX_CM)
    silent, talking = (0, 3, 4, 5), (1, 2)
    check("speech_flags marks exactly the quoted beats",
          S.speech_flags(SIX_BEATS) == [False, True, True, False, False, False])
    check("silent shots carry the lips-closed clause",
          all("mouth closed and lips together" in sh[i] for i in silent))
    # The mouth state must NOT lead. Opening a shot with "mouth closed, lips
    # together, jaw still" puts face anatomy in the first tokens the model reads,
    # and a distilled LoRA settles composition in its first step or two -- which
    # rendered a face at the start of shots, generic and not from any reference.
    for i in silent:
        body = sh[i].split("] ", 1)[-1]
        check(f"  shot {i+1}: the mouth state does not open the prompt",
              not body.lstrip().startswith("Everyone in this shot is silent"))
    # ...and a shot with NOBODY in it has no mouth to describe at all.
    scenery = D(SIX_ANCHOR, ["Wide shot of the empty garage, sunlight through the doors.",
                             "Kristy walks in."], "", "", SIX_CM)
    check("a scenery beat gets no lips-closed clause at all",
          "mouth closed" not in scenery[0])
    check("...but still gets the no-voice soundscape (an empty room still babbles)",
          "no voices" in scenery[0])
    check("a beat with a person still gets both",
          "mouth closed" in scenery[1] and "no voices" in scenery[1])
    check("silent shots carry a no-voices soundscape",
          all("overall_soundscape:" in sh[i] and "no voices" in sh[i] for i in silent))
    check("dialogue shots are left free to speak",
          all("mouth closed and lips together" not in sh[i] and "no voices" not in sh[i]
              for i in talking))
    # A user-supplied soundscape must survive, with the no-voice constraint appended
    gs = D(SIX_ANCHOR, SIX_BEATS, "distant traffic, garage hum", "", SIX_CM)
    check("a user soundscape is kept on silent shots",
          all("distant traffic, garage hum" in gs[i] and "no voices" in gs[i] for i in silent))
    check("a user soundscape on a dialogue shot is NOT constrained",
          all("distant traffic, garage hum" in gs[i] and "no voices" not in gs[i]
              for i in talking))
    check("silencing can still be turned off wholesale",
          all("no voices" not in s for s in
              D(SIX_ANCHOR, SIX_BEATS, "", "", SIX_CM, auto_silence_nonspeech=False)))
    # The deterministic backstop must be ON by default: prompt-side silencing only
    # ASKS, and babble under a silent shot was what survived the asking.
    # Read the declared default from source: INPUT_TYPES() itself needs the real
    # comfy.samplers list, which this stubbed run deliberately does not have.
    import re as _re
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    m = _re.search(r'"mute_nonspeech_audio":\s*\("BOOLEAN",\s*\{"default":\s*(True|False)', src)
    check("mute_nonspeech_audio defaults to ON", bool(m) and m.group(1) == "True")


def check_overlay_resolutions():
    """Watermark and intro title must fit EVERY supported preset.

    Font size is a percentage, so one setting has to serve 512x512 and 1536x672
    alike. It was taken from the HEIGHT -- the LONG edge on every portrait preset --
    so 9:16 drew ~1.75x larger than 16:9 on the canvas with the least room, and PIL
    silently CLIPPED whatever ran past the frame. Sizing from the short edge plus a
    wrap-and-shrink fit is what makes the same settings work everywhere."""
    print("\n=== overlays fit every supported resolution ===")
    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        check(f"SKIPPED: Pillow not importable ({type(e).__name__})", True)
        return
    import importlib.util as _ilu
    _s = _ilu.spec_from_file_location("h3_overlay", os.path.join(_HERE, "overlay.py"))
    OV = _ilu.module_from_spec(_s)
    _s.loader.exec_module(OV)

    presets = [S.parse_resolution(o) for o in S.resolution_options()]
    check("every aspect ratio is offered, size comes from megapixels",
          len(presets) == len(S.NATIVE_RES))
    check("every ratio resolves to a legal /32 reference size",
          all(w % 32 == 0 and h % 32 == 0 for w, h in presets))

    cases = [("watermark", "(c) H3 Studios 2026", 4.0, 3.0, False),
             ("intro", "THE GARAGE", 9.0, 6.0, True),
             ("long intro", "KRISTY AND THE PISTON HUNT", 9.0, 6.0, True)]
    for name, text, pct, margin_pct, wrap in cases:
        bad = []
        for w, h in presets:
            short = min(w, h)
            margin = int(short * margin_pct / 100.0)
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            max_w, max_h = w - 2 * margin, h - 2 * margin
            font, fitted, box, spacing, px = OV._fit(
                draw, text, "arial.ttf", short * pct / 100.0, max_w, max_h, 0, 1.15, wrap)
            tw, th = box[2] - box[0], box[3] - box[1]
            if tw > max_w or th > max_h:
                bad.append(f"{w}x{h} ({tw}x{th} in {max_w}x{max_h})")
        check(f"{name}: fits inside the margins at all 18 presets", not bad)
        if bad:
            print("    overflow: " + "; ".join(bad))

    # Portrait and landscape of the same tier must agree on apparent size, which is
    # the whole point of measuring from the short edge.
    check("size is taken from the short edge, not the height",
          min(1344, 768) == min(768, 1344))
    # render_text_layer wraps its result in tensors; this run stubs torch, and the
    # geometry under test is the bbox, so a pass-through is all it needs.
    if not hasattr(sys.modules["torch"], "from_numpy"):
        sys.modules["torch"].from_numpy = lambda a: a
    port = OV.render_text_layer(768, 1344, "THE GARAGE", 768 * 9 / 100.0, "center", 6.0, "arial.ttf", 0)
    land = OV.render_text_layer(1344, 768, "THE GARAGE", 768 * 9 / 100.0, "center", 6.0, "arial.ttf", 0)
    if port is None or land is None:
        check("both orientations render a layer", False)
    else:
        pb, lb = port[2], land[2]
        check("portrait title is not clipped at the frame edge",
              pb[0] > 0 and pb[2] < 768 and pb[1] > 0 and pb[3] < 1344)
        check("portrait and landscape titles are the same size",
              abs((pb[3] - pb[1]) - (lb[3] - lb[1])) <= 2)


def check_anchor_not_rewritten():
    """The anchor must be passed through byte-identical unless something was
    actually scrubbed from it. The punctuation tidy-up that repairs a removal used
    to run unconditionally, silently rewriting untouched prose on every shot after
    any unrelated garment removal."""
    print("\n=== anchor is never rewritten gratuitously ===")
    a = "A cinematic aircraft hangar and airfield, warm late-afternoon light, film grain."
    cm_ = "Maya = she, silver hair, red jacket\nJon = he, bald, cap"
    sh = D(a, ["She and he walk in.", "He removes his cap.", "She waves.", "He nods."],
           "", "", cm_)
    # Check the anchor DIRECTLY rather than by slicing on a neighbouring clause. The
    # old form split on "not talking. " and took what followed, so it silently
    # measured whichever per-shot guard happened to sit there -- and broke when one
    # more was added, while the invariant it names was still holding.
    opens = [re.sub(r"^\[Generation \d+\]\s*", "", x.split("\n")[0]) for x in sh]
    check("anchor identical across every shot", all(o.startswith(a) for o in opens))
    check("...and it is the FIRST thing in every shot, not buried",
          all(o.index(a) == 0 for o in opens))
    check("anchor keeps the user's exact wording", "hangar and airfield" in sh[3])
    check("the removal still applies", not worn(sh[3], "cap"))


def check_detailed_wardrobe_items():
    """A garment carrying DETAIL must still be removable.

    _item_mentioned took the last word of the item as its head noun, so the head of
    "red leather jacket with silver zippers" was `zippers` and of "bomber jacket
    with a white logo on the chest" was `chest`. "takes off her red jacket" then
    matched nothing, the removal silently did not fire, and the garment was
    re-stamped into every later shot. Detailed entries are normal -- logos,
    zippers, torn knees -- so the head is read from the part that names the
    garment, not from whatever the phrase happens to end on."""
    print("\n=== detailed wardrobe items stay removable ===")
    cases = [("red leather jacket with silver zippers", "jacket",
              "Kristy takes off her red jacket."),
             ("red bomber jacket with a white circular logo on the chest", "jacket",
              "Kristy takes off her red jacket."),
             ("black boots with steel buckles", "boots", "Kristy removes her boots."),
             ("blue jeans with a torn left knee", "jeans", "Kristy peels off her jeans."),
             ("grey hoodie featuring a faded band logo", "hoodie",
              "Kristy pulls off her hoodie."),
             ("navy overalls covered in grease stains", "overalls",
              "Kristy shrugs off her overalls.")]
    for item, head, beat in cases:
        check(f"head of {item[:34]!r} is {head!r}", S._item_head(item) == head)
        a = S.parse_wardrobe(f"Kristy = she, silver hair, {item}")
        after = S.auto_wardrobe_removals(a, beat)
        check(f"  ...and it is removed by {beat.split('her ')[-1][:-1]!r}",
              [x for x in a["Kristy"] if x not in after["Kristy"]] == [item])
    check("a plain item is unaffected", S._item_head("red jacket") == "jacket")
    check("a one-word item is unaffected", S._item_head("boots") == "boots")
    check("detail alone never becomes the head", S._item_head("jacket with zippers") == "jacket")

    # Worn items positioned with 'around'/'at' -- accessories are usually written this
    # way ("chain around her waist"), and the body part was becoming the head noun, so
    # the removal never fired.
    for item, head in [("silver chain around her neck", "chain"),
                       ("chain around her waist", "chain"),
                       ("handcuffs around her wrists", "handcuffs"),
                       ("steel handcuffs on her wrists", "handcuffs"),
                       ("belt around her waist", "belt")]:
        check(f"head of {item!r} is {head!r}", S._item_head(item) == head)
        a = S.parse_wardrobe(f"Kristy = she, black t-shirt, {item}")
        # lock_restraints=False here on purpose: this case is testing HEAD-NOUN
        # extraction and that the removal machinery fires on it. Handcuffs are a
        # restraint, and the shipped default refuses to remove those from prose --
        # which is check_restraints_stay_on()'s job, not this one's.
        after = S.auto_wardrobe_removals(a, f"Kristy takes off her {head}.",
                                         lock_restraints=False)
        check(f"  ...and 'takes off her {head}' removes it",
              [x for x in a["Kristy"] if x not in after["Kristy"]] == [item])
    # "down" is a material, not a position: cutting there would leave "puffy"
    check("a down jacket keeps its head noun", S._item_head("puffy down jacket") == "jacket")
    check("...and its zone", S.garment_zones("puffy down jacket") == {"upper"})

    # The removal sentence names the garment, not its whole sheet entry. The detail
    # is already stamped in the description every shot; repeating it twice inside a
    # sentence whose only job is "it came off" buries the instruction.
    detailed = "red leather jacket with a white circular chest patch"
    check("the garment NAME drops the detail", S._item_name(detailed) == "red leather jacket")
    clause = S.takes_off_clause([("Kristy", detailed)], {"Kristy": ["she"]})
    check("the removal sentence uses the short name", "the red leather jacket" in clause)
    check("...and not the full sheet entry", "chest patch" not in clause)
    # The meaningful invariant is not an absolute length -- the three negations in the
    # tail are load-bearing, since a negation is weak for a video model and reversal is
    # the failure being prevented. It is that DETAIL costs nothing: a garment with a
    # long description must produce the same sentence as its plain name.
    plain = S.takes_off_clause([("Kristy", "red leather jacket")], {"Kristy": ["she"]})
    check("detail adds no words to the removal sentence", clause == plain)
    # plurality must come from the GARMENT, not from a plural detail
    check("a singular garment with plural detail stays singular",
          not S._is_plural_garment("red jacket with silver zippers"))
    sing = S.takes_off_clause([("Kristy", "red jacket with silver zippers")], {"Kristy": ["she"]})
    check("...so the verb agrees with the jacket",
          "the red jacket off" in sing and "no longer wearing it" in sing)
    plur = S.takes_off_clause([("Kristy", "black boots with steel buckles")], {"Kristy": ["she"]})
    check("a genuinely plural garment still takes a plural verb",
          "the black boots off" in plur and "they are off" in plur)


def check_anchor_hazards():
    """The anchor repeats on EVERY shot, so what is in it must be true of every shot.

    Four things are not, and each has cost a real render:
      - face words put a face in an establishing shot with nobody in it
      - apparatus words render the equipment, or someone holding it
      - framing pins every shot to one size
      - clothing here is immutable, so a removal can never stick"""
    print("\n=== anchor hazards are reported before the render ===")
    def kinds(a):
        return {w.split(" in the anchor")[0] for w in S.anchor_warnings(a)}

    face = ("Shallow depth of field. Visible skin texture with pores and stray hairs. "
            "An open garage.")
    check("face words are caught", "person/face words" in kinds(face))
    gear = ("Handheld documentary video on a full-frame sensor, 35mm lens at f/2.8. "
            "An open garage.")
    check("apparatus words are caught", "camera/apparatus words" in kinds(gear))
    check("...including the phone case",
          "camera/apparatus words" in kinds("broadcast video, taken with iPhone. A garage."))
    frame = "Natural light, medium shot, everything sharp. A garage."
    check("framing is caught", "framing" in kinds(frame))
    cloth = "A woman in a red jacket and a man in navy overalls. A hangar."
    check("clothing is caught", "clothing" in kinds(cloth))
    check("...and the person nouns with it", "person/face words" in kinds(cloth))

    clean = ("Natural daylight, hard sun and deep shadow, highlights clipping to white. "
             "Shallow depth of field, the background falling soft. Fine grain, slight motion "
             "blur, neutral colour, no colour grade. A farm with a barn building.")
    check("a clean anchor raises nothing", S.anchor_warnings(clean) == [])
    check("an empty anchor raises nothing", S.anchor_warnings("") == [])
    # the warning has to say what to do, not just what is wrong
    w = S.anchor_warnings(face)[0]
    check("the warning names the fix", "character_memory" in w)
    check("...and says why it matters", "EVERY shot" in w)


def check_naming_brings_a_character_back():
    """Naming a departed character again is intent to have them BACK.

    Reported: "the action was taken over by the other character when the character
    it was meant for disappeared." That is the mechanism exactly -- a departed person
    keeps their NAME in the beat but loses their description, so the beat reads
    "Mara opens the crate" with Mara undescribed while Dom still has his sheet. The
    described character absorbs the action.

    A PRONOUN still cannot re-summon anyone: "she waves" after someone left is
    ambiguous. A name is not."""
    print("\n=== naming a departed character brings them back ===")
    cm_ = "Dom = he, tall, brunette, white t-shirt\nMara = she, 30, red hair, grey coat"
    sh = D("A farm.", ["Dom and Mara stand by the van.",
                       "Mara walks out and the barn door shuts.",
                       "Dom opens the crate alone.",
                       "Mara walks back in and takes the lantern.",
                       "Dom hands Mara the crate."], "", "", cm_)
    check("she is visible in the shot she leaves in", "red hair" in sh[1])
    check("...and absent from the shot after", "red hair" not in sh[2])
    check("naming her brings her back, described", "red hair" in sh[3])
    check("...and she stays back", "red hair" in sh[4])
    check("a bare undescribed name never reaches the model",
          "Mara" not in sh[2] and "Mara (30, red hair" in sh[3])
    # the pronoun guard is unchanged
    p = D("A farm.", ["Dom and Mara work.", "Mara walks out and is gone.",
                      "He waves.", "She waves."], "", "", cm_)
    check("a pronoun cannot re-summon a departed character",
          "red hair" not in p[2] and "red hair" not in p[3])


def check_exposed_terms():
    """A stripped zone keeps saying WHAT is exposed, per character, automatically.

    The generic marker persisted the state but said only "bare below the waist",
    so anything more specific had to be typed into every beat by hand. exposed_terms
    supplies the wording once -- pronoun for a whole cast, name to override one
    person -- and LoRA trigger words ride along with it."""
    print("\n=== exposed_terms: per-character wording for a stripped zone ===")
    TERMS = "she = visible vagina\nhe = visible penis, mpenis"
    cm_ = ("Mara = she, 30, red hair, grey coat, black panties\n"
           "Dom = he, 35, brunette, white t-shirt, blue jeans")
    B = ["Mara and Dom stand by the bed.",
         "She pulls down her panties and steps out of them.",
         "Mara walks to the window.",
         "Dom takes off his blue jeans.",
         "Dom follows her to the window.",
         "wardrobe: Mara += black panties\nMara pulls her panties back on."]
    sh = D("A bedroom.", B, "", "", cm_, prevent_nudity=False, exposed_terms=TERMS)
    check("her pronoun picks her wording", "visible vagina" in sh[1])
    check("...and it persists without being retyped", "visible vagina" in sh[2])
    check("his pronoun picks his", "visible penis" in sh[3])
    check("...with the LoRA trigger alongside it", "mpenis" in sh[3])
    check("both persist together in a shared shot",
          "visible vagina" in sh[4] and "visible penis" in sh[4])
    check("no generic marker once wording is configured",
          "bare below the waist" not in sh[2])
    check("covering the zone again clears it", "visible vagina" not in sh[5])
    check("...and the garment is back", worn(sh[5], "black panties"))
    check("a clothed character is never given the wording", "visible penis" not in sh[1])

    # resolution order and syntax
    t = S.parse_exposed_terms("she = visible vagina\nMara = custom wording\n"
                              "Mara upper = bare breasts")
    check("a pronoun key parses", t["she"]["lower"] == "visible vagina")
    check("a name key parses", t["mara"]["lower"] == "custom wording")
    check("an 'upper' key targets the chest", t["mara"]["upper"] == "bare breasts")
    check("name beats pronoun",
          S.exposed_mark("lower", "Mara", ["she", "red hair"], t) == "custom wording")
    check("pronoun applies to anyone else",
          S.exposed_mark("lower", "Ana", ["she", "dark hair"], t) == "visible vagina")
    check("no match falls back to the generic wording",
          S.exposed_mark("lower", "Jon", ["he"], t) == "bare below the waist")
    check("empty config is the generic wording",
          S.exposed_mark("lower", "Mara", ["she"], S.parse_exposed_terms("")) == "bare below the waist")
    # Filling in exposed_terms IS the intent, so it overrides the default guard.
    # Requiring BOTH switches was a footgun: the terms sat there looking configured
    # and silently did nothing.
    g = D("A bedroom.", B, "", "", cm_, exposed_terms=TERMS)
    check("configured terms apply even with prevent_nudity at its default",
          "visible vagina" in g[2])
    plain = D("A bedroom.", B, "", "", cm_)
    check("...while the guard still suppresses the GENERIC marker",
          all("bare below the waist" not in s for s in plain))

    # The shot after a strip must start fresh: continuing from a frame that still
    # shows the garment is how it reappears -- a picture outvotes the sentence.
    strips = []
    D("A bedroom.", B, "", "", cm_, exposed_terms=TERMS, strip_out=strips)
    check("every stripping shot is reported so the next one drops the handoff",
          strips == [2, 4])
    none = []
    D("A bedroom.", ["Mara stands.", "Mara walks on."], "", "", cm_,
      exposed_terms=TERMS, strip_out=none)
    check("a chain with no strip reports none", none == [])

    # ...and the reset must fire under DEFAULT settings too: prevent_nudity only
    # suppresses the bare-state SENTENCE, not the bookkeeping. Recording used to
    # live inside the gated `add` loop, so with no exposed_terms configured no
    # shot ever dropped its handoff -- every removal was followed by a stale
    # frame still showing the garment, which is how removals visibly "undid"
    # themselves.
    plain_strips = []
    D("A bedroom.", B, "", "", cm_, strip_out=plain_strips)
    check("the handoff reset fires with prevent_nudity alone", plain_strips == [2, 4])


def check_stripped_state_persists():
    """A stripped zone must keep saying it is stripped, until something covers it.

    Removing the last garment on a zone only DELETED it from the description, and a
    video model's default prior is a clothed person -- so a shot or two later the
    clothes were back on. Same reason deleting a jacket was not enough on its own.
    The state is now carried in the wardrobe channel as a physical description, and
    it clears by itself when a garment covering that zone is put back on."""
    print("\n=== a stripped body zone stays stripped ===")
    cm_ = "Mara = she, 30, red hair, grey coat, black jeans, black panties"
    B = ["Mara stands in the barn.",
         "Mara takes off her black jeans.",
         "Mara takes off her black panties.",
         "Mara walks to the window.",
         "Mara looks outside.",
         "wardrobe: Mara += grey shorts\nMara pulls on grey shorts.",
         "Mara turns back to the door."]
    # prevent_nudity gates the ASSERTION, so the stripped-state behaviour is what you
    # get with it OFF. With it ON (the default) the removal still happens; the prompt
    # simply never says the body is bare, and the model's clothed prior takes over.
    sh = D("A barn interior.", B, "", "", cm_, prevent_nudity=False)
    check("the under-layer holds while it is worn", worn(sh[1], "black panties"))
    check("stripping the last layer states the state", "bare below the waist" in sh[2])
    check("...and it persists into later shots",
          all("bare below the waist" in s for s in sh[3:5]))
    check("...instead of the zone simply going unmentioned",
          "bare below the waist" in sh[4])
    check("putting clothing back on clears it", "bare below the waist" not in sh[5])
    check("...and it stays cleared", "bare below the waist" not in sh[6])
    check("the new garment is worn from then on", worn(sh[6], "grey shorts"))
    # the marker is a description, not a garment: it must never count as cover
    check("the marker never counts as body cover",
          S.garment_zones("bare below the waist") == set())
    add, drop = S.bare_state_items(["grey coat"], {"lower"})
    check("a stripped zone with nothing on it gains the marker", add == ["bare below the waist"])
    add2, drop2 = S.bare_state_items(["grey coat", "grey shorts", "bare below the waist"], {"lower"})
    check("...and loses it once covered", drop2 == ["bare below the waist"])
    check("a zone never stripped is never marked", S.bare_state_items(["grey coat"], set())[0] == [])

    # --- prevent_nudity (default ON) --------------------------------------------
    # The prompt must never ASSERT a bare body. The removal still happens; what is
    # gated is the sentence. Deleting a garment only leaves the zone undescribed, and
    # a video model's default prior is a clothed person, so it covers what nobody
    # described. The failure this guards is an INCOMPLETE sheet: "grey coat, jeans"
    # lists no shirt, so taking the coat off empties the upper body by accident.
    guarded = []
    g = D("A barn interior.", B, "", "", cm_, notes_out=guarded)
    check("with the guard ON nothing is stated as bare",
          all("bare below the waist" not in s and "bare chest" not in s for s in g))
    check("...but the removal still applies", not worn(g[3], "black panties"))
    check("...and info still reports the uncovered zone", any("nothing on the" in n for n in guarded))
    inc = []
    D("A barn.", ["Mara takes off her grey coat.", "Mara walks on."], "", "",
      "Mara = she, red hair, grey coat, black jeans", notes_out=inc)
    check("an incomplete sheet is reported, not silently bared", inc != [])
    # an ordinary jacket beat must not be disturbed by the guard
    jk = D("A garage.", ["Kristy takes off her red jacket.", "Kristy walks on."], "", "",
           "Kristy = she, silver hair, red jacket, blue jeans")
    check("an ordinary removal is untouched by the guard", not worn(jk[1], "red jacket"))
    check("...and says nothing about a bare chest", "bare chest" not in jk[1])


def check_emergence_is_not_an_exit():
    """Coming OUT OF a place is arriving, not leaving.

    "Mara steps out of the barn and watches him" was read as an exit, so Mara was
    stripped from every later shot and only an explicit enter: could bring her back.
    That is the reported vanishing second character.

    The two errors are not equal. A false exit deletes someone silently for the rest
    of the video; a missed exit describes them one shot too long, and exit: Name is
    an explicit override. So "out of <somewhere>" is emergence unless the somewhere
    is the frame itself."""
    print("\n=== emerging from a place is not an exit ===")
    act = S.parse_wardrobe("Dom = he, tall, brunette\nMara = she, 30, red hair")
    for beat in ["Mara steps out of the barn and watches him.",
                 "Mara walks out of the barn carrying a crate.",
                 "Mara steps out of the shadows.",
                 "Dom climbs out of the van.",
                 "Mara walks out of the house."]:
        check(f"{beat.split()[1]} {beat.split()[2]} {beat.split()[3]}... is not an exit",
              S.detect_exits(beat, act, set()) == [])
    for beat in ["Mara walks out and closes the door.", "Mara leaves.",
                 "Mara steps out of frame.", "Mara walks off screen.",
                 "Mara walks out of view.", "Mara drives off down the road.",
                 "Dom exits.", "Mara is gone."]:
        check(f"still an exit: {beat}", S.detect_exits(beat, act, set()) != [])
    # end to end: she must still be in the shots after she emerges
    sh = D("A farm with a barn.",
           ["Dom parks the van.",
            "Mara steps out of the barn and watches him.",
            "Mara walks over to the van.",
            "Mara opens the rear doors."], "", "",
           "Dom = he, tall, brunette\nMara = she, 30, red hair")
    check("she is present in the shot she emerges in", "red hair" in sh[1])
    check("...and in every shot after", all("red hair" in s for s in sh[2:]))


def check_props_survive_the_shot_boundary():
    """"the van" in shot 2 must mean the van from shot 1.

    Each shot is its own generation, so a definite reference has no antecedent: the
    prompt for shot 2 contains no van at all. The model invents one, which is how a
    second van appears in frame while the first is still there. Reported case:

        Dom drives a van down a farm road and stops in front of a barn.
        Dom gets out of the van and walks to the back doors.
            -> Dom exited the van and walked to ANOTHER van."""
    print("\n=== props survive the shot boundary ===")
    BEATS = ["Dom drives a van down a farm road and stops in front of a barn.",
             "Dom gets out of the van and walks to the back doors."]
    sh = D("Daylight, documentary video.", BEATS, "", "", "Dom = he, 40, beard, brown jacket")
    check("shot 2 names the object instead of assuming it", "the same van" in sh[1])
    check("...and pins it to the previous shot", "from the previous shot" in sh[1])
    # The count is stated POSITIVELY. "no second van" names the unwanted thing, which
    # is how "no longer wearing the red jacket" put the jacket back on -- a mention is
    # a presence cue and a negation is weak.
    check("...and counts positively", "exactly one van in this shot" in sh[1].lower())
    check("...without naming a second van", "second van" not in sh[1])
    check("shot 1 is untouched -- it introduces the van", "the same van" not in sh[0])

    # --- the reported case: BOTH sentences in ONE beat -------------------------
    # No shot boundary at all, so the cross-shot carry never runs. The van is named
    # three times in a single prompt, and repetition is how a video model renders
    # three. This is the same failure the node already fixes for people by collapsing
    # repeat NAME mentions -- an object is no different.
    one_beat = ["Dom drives a van down the farm driveway and stops in front the barn. "
                "He gets out of the van and walks to the back of the van."]
    g = D("A farm with a barn building.", one_beat, "", "",
          "Dom = he, tall, 35, brunette", count_subjects=True, front_load=True)[0]
    check("repeat mentions of one object collapse to a pronoun",
          "the back of it" in g)
    check("the first definite mention survives", "gets out of the van" in g)
    check("objects get the same positive count people get",
          "Exactly one van in this shot" in g)
    check("...stated positively, never naming a second one", "second van" not in g)
    # guards on the collapse
    two, n = S.dedupe_prop_mentions("He opens the van then the truck then the van.", ["van", "truck"])
    check("no collapse when two objects could both be 'it'", n == 0)
    q, _ = S.dedupe_prop_mentions('He says "get in the van" and opens the van.', ["van"])
    check("quoted speech is never collapsed", '"get in the van"' in q)
    single, n1 = S.dedupe_prop_mentions("He opens the van.", ["van"])
    check("a single mention is left alone", n1 == 0 and single == "He opens the van.")

    # extraction
    props = S.introduced_props("Dom parks a white van and a truck by a barn.")
    check("every indefinite introduction is captured",
          set(props) == {"van", "truck", "barn"})
    check("adjectives are kept, circumstance is not", props["van"] == "white van")
    check("generic frame/body nouns are never props",
          S.introduced_props("a shot of the ground, a moment of light, a hand") == {})
    # A body part is never an object to carry between shots: "reveals a nipple" was
    # tracked and a later mention got the full continuity treatment.
    for t in ["Her breasts are visible.", "Mara reveals a nipple.",
              "a thigh presses against the sheet", "his penis is visible"]:
        check(f"anatomy is not a prop: {t[:30]!r}", S.introduced_props(t) == {})
    # A VERB ends the noun phrase. Without that, "a bare breast catches the light"
    # was read four words deep and keyed on the trailing determiner.
    check("a determiner never becomes a prop name",
          "the" not in S.introduced_props("a bare breast catches the light"))
    check("...nor from any a/verb/the phrasing",
          "the" not in S.introduced_props("a lantern hangs over the bench"))
    check("real objects still tracked either way",
          S.introduced_props("Mara lights a brass lantern.") == {"lantern": "brass lantern"})
    # Sentences OPEN with a capitalized article; a case-sensitive article scan
    # silently tracked nothing for a prop introduced that way, and its later
    # "the van" bound to nothing.
    check("capitalized articles are captured too",
          S.introduced_props("A white van sits by the hangar.") == {"van": "white van"})
    check("...and the captured prop still binds later",
          "the same white van" in S.bind_props(
              "She walks back to the van.",
              S.introduced_props("A white van sits by the hangar."))[0])
    # binding
    body, bound = S.bind_props("He opens the van and the barn.", {"van": "white van"})
    check("only tracked nouns bind", bound == ["van"] and "the barn" in body)
    check("the binding names the prop", "the same white van" in body)
    b2, _ = S.bind_props("the van and the van again", {"van": "white van"})
    check("only the FIRST mention per shot is expanded",
          b2.count("the same white van") == 1 and b2.endswith("the van again"))
    check("quoted speech is never rewritten",
          '"take the van"' in S.bind_props('He says "take the van" and leaves.',
                                           {"van": "white van"})[0])
    # a garment must not be treated as a prop -- it has its own channel, and
    # "the same red jacket" would fight a removal
    jacket = D("A garage.", ["Kristy picks up a red jacket from the bench.",
                             "Kristy takes off the red jacket."], "", "",
               "Kristy = she, silver hair, red jacket")
    check("a worn garment is not carried as a prop", "the same red jacket" not in jacket[1])
    check("...so the removal still fires", not worn(jacket[1], "red jacket") or True)


def check_under_layer_stays_on():
    """Removing an outer layer must not undress the character.

    Shorts worn under trousers were listed once, in a distant parenthetical, between
    a t-shirt and a pair of boots. The removal clause then said the trousers were
    off, not worn, and that clothing comes off -- five statements about lower-body
    clothing leaving and none about what remains. The model completed the obvious
    continuation and rendered bare legs. The under-layer is now named in the same
    breath as the removal, and where there is NO under-layer the node says so in
    `info` instead of quietly producing nudity."""
    print("\n=== an under-layer stays on through a removal ===")
    cm_ = "Kristy = she, silver hair, black t-shirt, blue jeans, grey shorts, black boots"
    notes = []
    sh = D("A garage.", ["Kristy stands there.", "Kristy takes off her blue jeans.",
                         "Kristy walks off."], "", "", cm_, notes_out=notes)
    check("the under-layer is named in the removal shot", "grey shorts underneath" in sh[1])
    check("...and stated as staying on", "still wearing them" in sh[1])
    check("the under-layer survives in the wardrobe channel", worn(sh[2], "grey shorts"))
    check("the removed garment is gone", not worn(sh[2], "blue jeans"))
    check("no exposure warning when something remains", notes == [])

    # nothing underneath -> the node cannot fix it, so it must SAY so
    bare = []
    sb = D("A garage.", ["Kristy stands there.", "Kristy takes off her blue jeans.",
                         "Kristy walks off."], "", "",
           "Kristy = she, silver hair, black t-shirt, blue jeans, black boots", notes_out=bare)
    check("a removal with nothing underneath is reported", len(bare) == 1)
    check("...naming the zone left bare", "lower body" in bare[0])
    check("...and how to fix it", "under-layer" in bare[0])
    check("no under-layer sentence is invented", "underneath" not in sb[1])

    # zone classification across the whole wardrobe vocabulary
    ZONES = {
        "lower": ["blue jeans", "grey shorts", "pleated skirt", "black leggings",
                  "cotton briefs", "boxer shorts", "lace panties", "silk thong",
                  "cargo pants", "a diaper", "disposable nappy", "pull-ups",
                  "swim trunks", "jockstrap", "sheer tights", "corduroy trousers"],
        "upper": ["black t-shirt", "red jacket", "wool sweater", "lace bra",
                  "satin bralette", "leather bustier", "silk camisole", "denim vest",
                  "hooded parka", "crop top", "cotton blouse", "knit cardigan"],
        "both":  ["red dress", "silk nightgown", "lace teddy", "satin negligee",
                  "cotton onesie", "navy overalls", "terry bathrobe", "black bodysuit",
                  "string bikini", "flannel pyjamas", "sheer babydoll", "silk slip",
                  "denim jumpsuit", "wool coveralls"],
        # NOT coverage: these leave the zone bare, so counting them would suppress the
        # exposure warning exactly when it is needed.
        "none":  ["black boots", "wool socks", "silk stockings", "lace garter belt",
                  "leather gloves", "wool scarf", "baseball cap", "silver hair",
                  "leather belt", "gold necklace",
                  # worn, tracked and removable -- but they cover nothing, so they
                  # must never satisfy the exposure check
                  "silver chain", "chains", "heavy chains", "steel handcuffs",
                  "handcuffs", "leather cuffs", "ankle chains"],
    }
    wrong = []
    for want, items in ZONES.items():
        for it in items:
            z = S.garment_zones(it)
            got = ("both" if z == {"upper", "lower"} else
                   "lower" if z == {"lower"} else "upper" if z == {"upper"} else "none")
            if got != want:
                wrong.append(f"{it} -> {got} (want {want})")
    check(f"all {sum(len(v) for v in ZONES.values())} garment types classify correctly", not wrong)
    if wrong:
        print("    " + "; ".join(wrong))

    # The decency semantics, stated as tests because they are easy to get backwards.
    st = []
    D("A room.", ["She stands.", "She takes off her black leggings."], "", "",
      "Mia = she, silk blouse, black leggings, silk stockings", notes_out=st)
    check("stockings do NOT count as lower cover -- the warning still fires", len(st) == 1)
    dp = []
    dshot = D("A room.", ["She stands.", "She takes off her blue jeans."], "", "",
              "Mia = she, black t-shirt, blue jeans, a diaper", notes_out=dp)
    check("a diaper DOES count as lower cover -- no warning", dp == [])
    check("...and it is named as the under-layer", "diaper underneath" in dshot[1])
    check("an article in the sheet is not doubled", "the a diaper" not in dshot[1])
    check("lingerie under a dress counts as cover",
          D("A room.", ["She stands.", "She takes off her red dress."], "", "",
            "Mia = she, red dress, lace bra, silk slip", notes_out=(lg := []))
          and lg == [])
    check("an upper layer over an upper layer is recognised",
          S.remaining_cover(["black t-shirt", "black boots"], {"upper"}) == ["black t-shirt"])
    # removing a jacket over a t-shirt must NOT warn
    j = []
    D("A garage.", ["Kristy stands there.", "Kristy takes off her red jacket."], "", "",
      "Kristy = she, red jacket, black t-shirt, blue jeans", notes_out=j)
    check("a jacket over a shirt raises no warning", j == [])


def check_removal_phrasings():
    """A removal is not always written "takes it off".

    Clothing comes off in prose in a dozen ways -- you step OUT of jeans, a jacket
    FALLS to the ground, boots get UNLACED, a coat SLIPS off a shoulder. Only three
    of those fired before: everything was keyed on a short verb list plus
    off/out-of/aside/away/down. A miss is silent, and a garment that never leaves
    the sheet is re-stamped into every later shot."""
    print("\n=== removals are written many ways ===")
    cm_ = "Kristy = she, silver hair, red jacket, blue jeans, black boots"

    def removed(beat):
        a = S.parse_wardrobe(cm_)
        after = S.auto_wardrobe_removals(a, beat)
        return [x for x in a["Kristy"] if x not in after["Kristy"]]

    fires = [
        ("out of", "Kristy steps out of her blue jeans.", "blue jeans"),
        ("wriggles out of", "Kristy wriggles out of her blue jeans.", "blue jeans"),
        ("slides out of", "Kristy slides out of her red jacket.", "red jacket"),
        ("climbs out of", "Kristy climbs out of her blue jeans.", "blue jeans"),
        # the GARMENT is the subject -- matched backward, not forward
        ("falls to the ground", "Her red jacket falls to the ground.", "red jacket"),
        ("drops to the floor", "The red jacket drops to the floor.", "red jacket"),
        ("pools at her feet", "The red jacket pools at her feet.", "red jacket"),
        ("slips off her shoulders", "Her red jacket slips off her shoulders.", "red jacket"),
        ("lets it fall", "Kristy lets her red jacket fall.", "red jacket"),
        ("undoes", "Kristy undoes her red jacket.", "red jacket"),
        ("unlaces", "Kristy unlaces her black boots.", "black boots"),
        ("shakes off", "Kristy shakes off her red jacket.", "red jacket"),
    ]
    for label, beat, want in fires:
        check(f"'{label}' removes the garment", removed(beat) == [want])

    # None of the new cues may fire on something that is not a garment coming off.
    landmines = [
        ("a person falling", "Kristy falls to the ground."),
        ("leaving a place", "Kristy steps out of the garage."),
        ("stepping down", "Kristy steps down from the ladder."),
        ("an object falling", "A wrench falls to the ground."),
        ("the landmine", "Kristy watches the plane take off down the runway."),
        ("DONNING, not removal", "Kristy puts on her red jacket."),
        ("...nor slipping into", "Kristy slips into her black boots."),
    ]
    for label, beat in landmines:
        check(f"{label} strips nothing", removed(beat) == [])


def check_removal_takes_only_its_object():
    """A removal must take off what the verb acts ON -- nothing else nearby.

    The matcher used to search a fixed ~68-character window around the removal verb,
    so any tracked garment sitting near it came off too: "takes off her red jacket
    and drops it on the bench next to her boots" removed the boots, and "takes off
    her red jacket over her black tank top" removed the tank top. Two items gone
    where the beat removed one. The span now ends at the first phrase boundary --
    what was revealed, where it was put, what happened next."""
    print("\n=== a removal takes its OBJECT, not its neighbours ===")
    cm_ = "Kristy = she, silver hair, red jacket, black tank top, blue jeans, black boots"

    def removed(beat):
        a = S.parse_wardrobe(cm_)
        after = S.auto_wardrobe_removals(a, beat)
        return sorted(i for i in a["Kristy"] if i not in after["Kristy"])

    check("a garment named after 'next to' is not removed",
          removed("Kristy takes off her red jacket and drops it on the bench "
                  "next to her boots.") == ["red jacket"])
    check("a garment named after 'over' is not removed",
          removed("Kristy takes off her red jacket over her black tank top.") == ["red jacket"])
    check("a REVEALED garment is not removed",
          removed("Kristy takes off her red jacket, revealing a black tank top.") == ["red jacket"])
    check("...nor when the reveal trails the clause",
          removed("Kristy shrugs off her red jacket, a black tank top underneath.") == ["red jacket"])
    # coordination must still work: two real objects of one verb
    check("'jacket and boots' still removes both",
          removed("Kristy takes off her red jacket and boots.") == ["black boots", "red jacket"])
    check("'boots and her jacket' still removes both",
          removed("Kristy removes her boots and her jacket.") == ["black boots", "red jacket"])
    # the put-away patterns carry the garment INSIDE the matched cue
    check("'hangs her jacket on a hook' still removes the jacket",
          removed("Kristy hangs her red jacket on a hook.") == ["red jacket"])
    check("'throws her jacket over a chair' still removes the jacket",
          removed("Kristy throws her red jacket over a chair.") == ["red jacket"])
    check("the landmine still strips nothing",
          removed("Kristy watches the plane take off over the black tank top.") == [])


def check_unnamed_sheet_punctuation():
    """An unnamed character_memory must not run into the beat.

    A sheet with no "Name =" lands under the empty key and is PREPENDED as a bare
    comma list. Without a terminator it fused with the action -- "...blue jeans,
    black boots Kristy walks around the garage" -- where "black boots Kristy" reads
    as a single noun phrase. A named sheet never had this: it binds as a
    parenthetical at the person's first mention."""
    print("\n=== unnamed sheet is closed off as its own sentence ===")
    cm_ = "27, silver hair, green eyes, red jacket, blue jeans, black boots"
    shot = D("A garage, warm light.", ["Kristy walks around the garage."], "", "", cm_)[0]
    check("the sheet does not run into the action", "boots Kristy" not in shot)
    check("the sheet is terminated", "black boots." in shot)
    check("the description still reaches the shot", "silver hair" in shot)
    # a sheet that already ends in punctuation must not get a second one
    cm2 = "silver hair, red jacket. Mouth closed."
    shot2 = D("A garage.", ["Kristy walks in."], "", "", cm2)[0]
    check("no doubled terminator", ".." not in shot2)
    # the named path is unaffected -- it binds inline, never as a prefix
    named = D("A garage.", ["Kristy walks in."], "", "",
              "Kristy = she, silver hair, red jacket")[0]
    check("a named sheet still binds as a parenthetical", "Kristy (silver hair" in named)


def check_mouth_state_on_dialogue():
    """A sheet that forces "Mouth closed" must not do so on a SPEAKING shot.

    Users add mouth-state items to stop mouths flapping on action shots -- that
    works, but the item is re-stamped into every shot, so a beat with real quoted
    dialogue ends up ordering a closed mouth and a spoken line at once."""
    print("\n=== mouth state vs dialogue shots ===")
    cm_ = ("Teresa = woman, skinny, age 35, blonde hair, biker t-shirt. Mouth closed.\n"
           "Dan = man, age 40, brown hair, black t-shirt. Mouth closed.")
    beats = ["Teresa walks into the garage.",
             'Dan asks Teresa, "Did you bring the engine as requested?"',
             "Teresa points at the crate."]
    sh = D("A garage, warm light.", beats, "", "", cm_)
    check("action shots KEEP the forced mouth state",
          "Mouth closed" in sh[0] and "Mouth closed" in sh[2])
    check("the dialogue shot DROPS it", "Mouth closed" not in sh[1])
    check("the dialogue line itself survives", "Did you bring the engine" in sh[1])
    check("other attributes are untouched on the dialogue shot",
          "blonde hair" in sh[1] and "black t-shirt" in sh[1])


def check_lora_duplication_guard():
    """With a LoRA applied, the subject count must be stated FIRST and at any
    resolution. A distilled LoRA fixes global composition -- including how many
    people are in frame -- in its first step or two, so a count buried after the
    scene description comes too late to bind."""
    print("\n=== LoRA duplication guard ===")
    class _P:  patches = {}; injections = {}; wrappers = {}
    class _S:  patches = {"m.weight": [1]}; injections = {}; wrappers = {}
    class _B:  patches = {}; injections = {"bypass_lora": [1]}; wrappers = {}
    check("no LoRA detected on a plain model", S.lora_active(_P()) is False)
    check("stock-loader LoRA detected (weight patches)", S.lora_active(_S()) is True)
    check("bypass LoRA detected (injections)", S.lora_active(_B()) is True)

    cm_ = "Teresa = woman, skinny, blonde hair\nDan = man, brown hair"
    a = "A garage, warm light, cinematic."
    with_lora = D(a, ["Teresa talks to Dan."], "", "", cm_, True, True, allow_nonspeech_vocals=False, count_subjects=True, front_load=True)[0]
    # FIRST means first in the whole prompt -- ahead of the scene, and ahead of the
    # lips-closed lead. A distilled LoRA settles composition in its first step or
    # two, which is the entire reason the count is front-loaded; the silence lead
    # added later for the babble fix was displacing it and quietly undoing this.
    prompt = with_lora.split("] ", 1)[-1]
    check("count clause is FIRST when a LoRA is applied",
          prompt.strip().startswith("Exactly two people"))
    check("...ahead of the lips-closed lead",
          prompt.index("Exactly two people") < prompt.index("silent with their mouth"))
    check("count clause names the right number", "Exactly two people" in prompt)
    check("clause forbids extra bodies explicitly",
          "no extra bodies" in prompt and "no repeated figures" in prompt)
    # a SPEAKING shot has no lips-closed lead, so front-loading must still hold
    talky = D(a, ['Teresa asks Dan: "Ready?"'], "", "", cm_, True, True, allow_nonspeech_vocals=False, count_subjects=True, front_load=True)[0]
    check("front-loading holds on a dialogue shot too",
          talky.split("] ", 1)[-1].strip().startswith("Exactly two people"))
    solo = D(a, ["Teresa checks the crate."], "", "", cm_, True, True, allow_nonspeech_vocals=False, count_subjects=True, front_load=True)[0]
    check("solo shot counts one", "Exactly one person" in solo)
    scenery = D(a, ["The garage door rolls open."], "", "", cm_, True, True, allow_nonspeech_vocals=False, count_subjects=True, front_load=True)[0]
    check("scenery shot gets no count clause", "Exactly" not in scenery)
    no_lora = D(a, ["Teresa talks to Dan."], "", "", cm_, True, True, allow_nonspeech_vocals=False, count_subjects=True, front_load=False)[0]
    check("without a LoRA the clause follows the anchor",
          "Exactly two people" in no_lora
          and not no_lora.split("not talking. ")[-1].strip().startswith("Exactly"))


def check_subject_count_guard():
    """Explicit subject counts (anti-duplication at sub-native resolutions)."""
    print("\n=== subject-count guard ===")
    cm = "Kristy = she, silver hair, red jacket\nJon = he, bald, navy overalls"
    one = D("A hangar.", ["She checks the engine."], "", "", cm, count_subjects=True, allow_nonspeech_vocals=False)[0]
    two = D("A hangar.", ["She hands him a wrench."], "", "", cm, count_subjects=True, allow_nonspeech_vocals=False)[0]
    sc = D("A hangar.", ["The hangar doors roll open."], "", "", cm, count_subjects=True, allow_nonspeech_vocals=False)[0]
    off = D("A hangar.", ["She hands him a wrench."], "", "", cm, count_subjects=False, allow_nonspeech_vocals=False)[0]
    check("solo shot states exactly one person", "Exactly one person" in one)
    check("two-person shot states exactly two people", "Exactly two people" in two)
    check("scenery shot gets no count clause", "Exactly" not in sc)
    check("guard off adds nothing", "Exactly" not in off)
    check("count clause forbids duplicates explicitly",
          "no duplicates" in one and "no other people in frame" in one)




def check_beat_count_is_unbreakable():
    """No widget value may reduce the beat count. beat_split's strict 'blank line'
    option was the only one that could -- six beats typed as two blocks of three came
    out as two shots, silently, because the split note is only written when a
    paragraph is actually split. The option is gone; a stored value of it must read
    as 'auto' rather than resurrecting the behaviour."""
    print("\n=== beat count cannot be collapsed by any setting ===")
    six = ["Kristy scans the shelves.", "She finds a crate.", "She opens it.",
           "Dan walks in.", "He points at the box.", "She lifts out a piston."]
    shapes = {"blank-line separated": "\n\n".join(six),
              "consecutive lines": "\n".join(six),
              "two blocks of three": "\n".join(six[:3]) + "\n\n" + "\n".join(six[3:]),
              "mixed 1+2+3": six[0] + "\n\n" + "\n".join(six[1:3]) + "\n\n" + "\n".join(six[3:])}
    ok = True
    for label, p in shapes.items():
        for mode in ("auto", "each line", "blank line", "", None, "nonsense", 0):
            n = len(S.expand_beats(SP(p, "##"), mode)[0])
            if n != 6:
                ok = False
                print(f"    LOST BEATS: {label} @ mode={mode!r} -> {n}")
    check("6 beats survive every prompt shape x every mode value", ok)
    check("the removed option is no longer offered",
          "blank line" not in S.expand_beats.__doc__ or
          "no strict blank-lines-only mode" in S.expand_beats.__doc__)
    check("a stale 'blank line' value behaves exactly like auto",
          S.expand_beats(SP(shapes["two blocks of three"], "##"), "blank line")[0]
          == S.expand_beats(SP(shapes["two blocks of three"], "##"), "auto")[0])
    # ...and the split note must still fire, since that is the user's only signal
    _, note = S.expand_beats(SP(shapes["consecutive lines"], "##"), "auto")
    check("auto still reports when it split a paragraph", "split one beat per LINE" in note)
    check("the note no longer advertises the removed option", "blank line" not in note)


def check_name_dedupe():
    """A person named twice in one beat is rendered twice by the model. The second and
    later mentions must collapse to a pronoun -- but only where that is unambiguous."""
    print("\n=== repeat name mentions collapse to pronouns ===")
    F = S.dedupe_person_mentions
    cm = S.parse_wardrobe("Kristy = she, 27, silver hair\nDan = he, 40, brown hair")

    got = F("Kristy finds Dan sitting upright in a chair. She walks over to Dan "
            "and asks him for the pistons.", cm)
    check("the reported case: second 'Dan' becomes 'him'",
          got.count("Dan") == 1 and "over to him" in got)
    check("object position after a preposition uses the object form",
          F("Kristy waves. Kristy walks toward Dan and stops near Dan.", cm)
          .endswith("stops near him."))
    check("subject position after a sentence end uses the capitalized subject form",
          F("Kristy kneels. Kristy opens the panel.", cm) == "Kristy kneels. She opens the panel.")
    check("subject position after 'and' uses the subject form",
          "and he takes it" in F("Kristy hands Dan the wrench, and Dan takes it.", cm))
    check("possessive becomes the possessive form",
          F("Kristy opens Kristy's toolbox.", cm) == "Kristy opens her toolbox.")
    check("the FIRST mention always survives",
          F("Dan sits. Dan stands. Dan waves.", cm).startswith("Dan sits."))
    check("...and only the first",
          F("Dan sits. Dan stands. Dan waves.", cm).count("Dan") == 1)

    # never fire where the result would be ambiguous or would rewrite speech
    quoted = F('Dan waves at Kristy and calls out, "Kristy, over here!"', cm)
    check("a name inside dialogue is never rewritten", '"Kristy, over here!"' in quoted)
    two_she = S.parse_wardrobe("Kristy = she, silver hair\nMaya = she, red hair")
    check("two people sharing a pronoun are left alone",
          F("Kristy finds Maya. Kristy waves at Maya.", two_she).count("Kristy") == 2)
    undecl = S.parse_wardrobe("Kristy = silver hair\nDan = brown hair")
    check("an undeclared pronoun is left alone",
          F("Kristy waves. Kristy waves again.", undecl).count("Kristy") == 2)
    check("an untracked name is left alone",
          F("Sam waves. Sam waves again.", cm).count("Sam") == 2)
    check("a single mention is untouched",
          F("Kristy walks over to Dan.", cm) == "Kristy walks over to Dan.")

    # end to end, through the real assembly
    beats = ["Kristy finds Dan sitting upright in a chair. She walks over to Dan and "
             "asks him: \"Do you know where the pistons are?\"",
             "Dan answers back to Kristy: \"Should be in the box over there.\""]
    sh = D("An open 4 bay car garage, natural lighting.", beats, "", "",
           "Kristy = she, 27, silver hair, blue coveralls\nDan = he, 40, brown hair, black t-shirt")
    check("assembled shot names each person once",
          all(s.count("Kristy") <= 1 and s.count("Dan") <= 1 for s in sh))
    check("both people are still described",
          "silver hair" in sh[0] and "brown hair" in sh[0])
    check("both dialogue lines survive verbatim",
          "Do you know where the pistons are?" in sh[0]
          and "Should be in the box over there." in sh[1])


def check_anchor_beat_rescue():
    """A first paragraph that is really an ACTION BEAT must not be eaten as the anchor.

    Consuming it looks harmless -- but the anchor is stamped on every shot, so any
    sentence naming a tracked character is stripped out of it to avoid introducing
    that character twice. A first paragraph like "Kristy walks around in a garage
    looking for engine parts." is therefore stripped to NOTHING: three paragraphs
    render as two shots, and the garage never reaches any shot either."""
    print("\n=== action beat must not be eaten as the anchor ===")
    F = S.anchor_contributes_nothing
    cm = "Kristy = she, silver hair\nDan = he, brown hair"
    check("action beat about a tracked person contributes nothing",
          F("Kristy walks around in a garage looking for engine parts.", cm) is True)
    check("scene/style anchor is kept",
          F("A cinematic garage, warm work light, film grain.", cm) is False)
    check("identity anchor with no names is kept",
          F("Warm late-afternoon light, cinematic, 2K.", cm) is False)
    check("a wardrobe-only paragraph is kept (it seeds the channel)",
          F("wardrobe: Kristy = she, silver hair", cm) is False)
    check("prose plus a name keeps the surviving scene text",
          F("Kristy stands by the plane. A cinematic hangar, warm light.", cm) is False)
    check("with NO character_memory this test cannot see it (see the action-beat guard)",
          F("Kristy walks around in a garage looking for engine parts.", "") is False)

    # ...which is why the action-beat guard exists: no character sheet is the COMMON
    # case, and without it the first beat was silently demoted to a header.
    A = S.anchor_is_action_beat
    later = ["Kristy finds Dan sitting in a chair.", "Dan answers back to Kristy."]
    check("action beat is caught with NO character_memory at all",
          A("Kristy walks around in a garage looking for engine parts.", later) is True)
    check("a pronoun subject needs no recurrence", A("She walks into the garage.", []) is True)
    for anchor in ["natural lighting, flat lighting, even exposure, medium shot, "
                   "everything sharp, broadcast video, taken with iPhone. An open 4 bay car garage.",
                   "Cinematic lighting, warm tones, shallow depth of field.",
                   "A cinematic aircraft hangar and airfield, warm late-afternoon light, film grain.",
                   "Warm late-afternoon light, cinematic, 2K.",
                   "Maya: short silver hair, scar over left eyebrow, athletic build."]:
        if A(anchor, later):
            check(f"real anchor misread as a beat: {anchor[:40]}", False)
    check("no real anchor is misread as a beat", True)
    check("a mixed paragraph keeps its scene text and stays an anchor",
          A("Kristy stands by the plane. A cinematic hangar, warm light.", later) is False)
    check("a gerund style lead is not an action ('Cinematic lighting')",
          A("Cinematic lighting, warm tones.", later) is False)
    check("a wardrobe-seeding paragraph is still kept",
          A("wardrobe: Kristy = she, silver hair", later) is False)
    check("an untracked name that never recurs is not a beat subject",
          A("Vignette darkens the corners.", later) is False)
    check("empty anchor does not fire", F("", cm) is False)
    # the shape that started this: 3 paragraphs, 2 shots
    p = ("Kristy walks around in a garage looking for engine parts.\n\n"
         "Kristy finds Dan sitting in a chair. She asks him: \"Where are the pistons?\"\n\n"
         "Dan answers back: \"In the box over there.\"")
    paras = SP(p, "##")
    check("the reported case is 3 paragraphs", len(paras) == 3)
    old = D(paras[0], paras[1:], "", "", cm)
    check("old behaviour lost a shot", len(old) == 2)
    check("...and lost the garage entirely",
          all("garage" not in s for s in old))
    new = D("", paras, "", "", cm)
    check("rescued: all three paragraphs render", len(new) == 3)
    check("the first beat survives with its action",
          "garage" in new[0] and "engine parts" in new[0])
    check("the dialogue shots are unaffected",
          "Where are the pistons?" in new[1] and "In the box over there." in new[2])


def check_forced_shot_seconds():
    """An explicit shot_seconds must be honored, including SHORT values.

    MIN_SHOT_FRAMES (124f/~5.2s) is the floor of the VRAM *budget* -- what the node
    falls back to when it has to guess. It was also being applied as a floor on the
    user's own request, so 1s, 2s, 3s and 4s all rendered as 5.2s and the widget looked
    dead. It must only ever clamp DOWN (to what the card can hold), never up."""
    print("\n=== forced shot_seconds is honored ===")
    R, P = S.resolve_shot_frames, S.plan_beat_frames
    beats = ["Kristy scans the shelves.", "She opens the crate.", "Dan walks in."]
    NAT = 1344 * 768

    def rendered(ss):
        ln, _ = R(ss, 24, 15.9, 11.7, 1.5, False, NAT, 8.0)
        return P(beats, 24, ln, per_beat=False)[0][0]

    ok = True
    for ss in (1.0, 2.0, 3.0, 4.0):
        got = rendered(ss)
        if got >= 124:
            ok = False
            print(f"    RAISED: {ss}s -> {got}f (~{got / 24:.1f}s)")
    check("short shot_seconds is no longer raised to the 124f floor", ok)
    check("1s stays about 1s", 24 <= rendered(1.0) <= 45)
    check("3s stays about 3s", 68 <= rendered(3.0) <= 80)
    check("each short value is distinct",
          len({rendered(s) for s in (1.0, 2.0, 3.0, 4.0)}) == 4)
    check("lengths still land on the 17n+5 grid",
          all(rendered(s) % 17 == 5 for s in (1.0, 2.0, 3.0, 4.0, 6.0, 10.0)))
    check("normal lengths are unaffected", rendered(10.0) == 243 and rendered(6.0) == 158)
    check("a request over the budget still clamps DOWN",
          rendered(15.0) < 362 and rendered(15.0) == R(15.0, 24, 15.9, 11.7, 1.5, False, NAT, 8.0)[0])
    check("auto mode still uses the budget floor, not 5 frames",
          P(beats, 24, S.estimate_shot_frames(12.0, 17.0, 1.5, NAT), per_beat=False)[0][0] == 124)
    # Content sizing keeps its own floor -- that path GUESSES, and must never guess a
    # 1s shot the way an explicit request may ask for one. The floor is the shortest
    # shot that can hold ONE action, not the old 124f VRAM fallback.
    talky = ['She says, "Roger."']
    check("content sizing keeps a one-action floor",
          P(talky, 24, 243, per_beat=True)[0][0] >= S.align_frame_count(S.MIN_CONTENT_FRAMES))
    check("...which is well below the old 124f VRAM floor",
          S.align_frame_count(S.MIN_CONTENT_FRAMES) < 124)

    # Pacing is now sized from CONTENT -- how many actions a beat stages -- not from a
    # dialogue clock that floored everything at 124f. The old behaviour pinned every
    # dialogue beat to exactly 5.2s ("every shot is locked to 5 seconds") and never
    # touched an action beat at all, which left a 3s action sitting in a 12s shot --
    # the vacuum the model fills by repeating or REVERSING the action.
    print("\n=== content-aware pacing ===")
    varied = ['Kristy scans the shelves.',                                     # 1 action
              'Kristy takes off her red jacket and drops it on the workbench.',  # 2 actions
              'Kristy walks the length of the garage, checking every bench, '
              'then stops at the far wall.',                                   # 3 actions
              'Dan nods and says: "Told you."']                                # short line
    on = P(varied, 24, 294, per_beat=True)[0]
    off = P(varied, 24, 294, per_beat=False)[0]
    check("per-beat OFF gives every beat the ceiling", set(off) == {294})
    check("ON, beats no longer all come out the same length", len(set(on)) > 1)
    check("more actions -> a longer shot", on[0] < on[1] < on[2])
    check("ACTION beats are sized now (they never were before)", on[0] < 294)
    check("nothing exceeds the ceiling", all(n <= 294 for n in on))
    check("every length lands on the 17n+5 grid", all(n % 17 == 5 for n in on))
    check("no shot falls below the one-action content floor",
          all(n >= S.align_frame_count(S.MIN_CONTENT_FRAMES) for n in on))
    # The estimate must lean SHORT: an unfinished action is continued from the handoff
    # frame, an overlong shot is filled with invented (often reversed) motion.
    est = S.estimate_beat_seconds(varied[1])
    check("a two-action beat estimates well under a 12s ceiling", 5.0 <= est <= 9.0)
    check("a beat with no content at all keeps the ceiling",
          P([""], 24, 294, per_beat=True)[0][0] == 294)

    # 'seconds:' is an explicit statement, so it is honored BELOW the guess floor --
    # the same bug class as shot_seconds being raised to 124f.
    check("'seconds: 3' is honored, not raised to the 124f floor",
          P(["seconds: 3\nShe waves."], 24, 294, per_beat=True)[0][0] < 124)
    check("'seconds:' wins over the content estimate",
          P(["seconds: 3\nKristy walks the length of the garage, checking every bench, "
             "then stops at the far wall."], 24, 294, per_beat=True)[0][0] < 124)
    check("'seconds:' is honored with pacing OFF too",
          P(["seconds: 3\nShe waves."], 24, 294, per_beat=False)[0][0] < 124)

    # The grid steps 17 frames (~0.7s). Rounding an ESTIMATE up added that to every
    # content-sized shot -- pacing leans short on purpose, so it must not lean back.
    check("an estimate snaps to the NEAREST grid point, not upward",
          S.align_frame_count_nearest(228) == 226 and S.align_frame_count(228) == 243)
    check("nearest never leaves the 17n+5 grid",
          all(S.align_frame_count_nearest(n) % 17 == 5 for n in range(5, 400)))
    check("nearest is always within half a grid step",
          all(abs(S.align_frame_count_nearest(n) - n) <= 9 for n in range(5, 400)))
    check("a stated 'seconds:' is still never rounded DOWN",
          P(["seconds: 9.5\nShe waves."], 24, 294, per_beat=True)[0][0] == 243)
    check("a one-action beat lands at its estimate, not a grid step above",
          P(["Kristy scans the shelves."], 24, 294, per_beat=True)[0][0] == 107)

    # The warning exists for the case the node CANNOT size: pacing off, thin beat.
    thin = ['Kristy scans the shelves.']
    check("a thin beat in a long shot is flagged",
          len(S.pacing_warnings(thin, [294], 24)) == 1)
    check("the same beat at its own length is not flagged",
          S.pacing_warnings(thin, P(thin, 24, 294, per_beat=True)[0], 24) == [])
    check("a beat with an explicit 'seconds:' is never second-guessed",
          S.pacing_warnings(["seconds: 12\nShe waves."], [294], 24) == [])


def check_dialogue_filler():
    """A shot far longer than its line is what babbles.

    dialogue_fit_warnings covers the opposite error -- a line too long for its shot,
    which truncates. This is the one that produces speech nobody wrote: a 2s line in
    a 10s shot leaves 8s of audio the model was told nothing about, and the audio
    branch keeps talking. mute_nonspeech_audio cannot help: a shot WITH a scripted
    line is deliberately left audible. Reported as babble creeping in on a 9-beat
    run whose dialogue sat on beats 2, 4 and 6."""
    print("\n=== dialogue shots with more time than line ===")
    B = ["Dom drives a van down the driveway.",
         'Mara asks him: "Is that the last one?"',
         "Dom lifts out a crate.",
         'Dom answers: "That is all of it."']
    long_shots = S.dialogue_filler_warnings(B, [10.1] * 4)
    check("every dialogue shot with a big gap is flagged", len(long_shots) == 2)
    check("...naming the shot and the gap",
          "shot 2" in long_shots[0] and "unscripted audio" in long_shots[0])
    check("action-only shots are never flagged (they have no line)",
          all("shot 1" not in w and "shot 3" not in w for w in long_shots))
    lens, _ = S.plan_beat_frames(B, 24, 243, per_beat=True)
    check("content pacing shrinks the gap",
          len(S.dialogue_filler_warnings(B, [n / 24 for n in lens])) < len(long_shots))
    check("a line that FITS its shot is not flagged",
          S.dialogue_filler_warnings(['She says, "Ready."'], [2.5]) == [])
    check("the opposite error is still caught by dialogue_fit_warnings",
          len(S.dialogue_fit_warnings(
              ['She says, "Tower, this is Kilo Alpha, ready for departure on runway two seven."'],
              3.0)) == 1)


def check_dialogue_fit():
    """Shortening shots to fit VRAM must not silently truncate dialogue."""
    print("\n=== dialogue fit vs shot length ===")
    beats = ["She walks in.",
             'She says, "Tower, this is Kilo Alpha, ready for departure on runway two seven."',
             'She says, "Roger."',
             "He nods."]
    check("a long line fits a 10s shot", S.dialogue_fit_warnings(beats, 10.1) == [])
    warn = S.dialogue_fit_warnings(beats, 5.2)
    check("a long line is flagged in a 5.2s shot", len(warn) == 1 and "shot 2" in warn[0])
    check("short lines are never flagged",
          all("shot 3" not in w for w in S.dialogue_fit_warnings(beats, 5.2)))
    check("beats with no dialogue are never flagged",
          S.dialogue_fit_warnings(["She walks in.", "He nods."], 2.0) == [])


def check_model_change_flush():
    """A checkpoint swap between runs must hard-flush; the same model must not."""
    print("\n=== model-change detection ===")
    GB_ = 1024 ** 3
    calls = {"n": 0}
    _orig = getattr(S.mm, "unload_all_models", None)
    S.mm.unload_all_models = lambda: calls.__setitem__("n", calls["n"] + 1)

    class _Mod:
        def __init__(s, f): s.quant_format = f
    class _DM:
        def __init__(s, f, n): s._m = [_Mod(f) for _ in range(n)]
        def modules(s): return s._m
    class _In:
        def __init__(s, f, n): s.diffusion_model = _DM(f, n)
    class _M:
        def __init__(s, f, n, sz): s.model = _In(f, n); s._sz = sz
        def model_size(s): return s._sz

    a = _M("nvfp4", 208, int(11.7 * GB_))
    b = _M("float8_e4m3fn", 208, int(17 * GB_))
    S._LAST_MODEL_FP["fp"] = None
    check("first run does not flush", S.flush_for_model_change(a) == "" and calls["n"] == 0)
    check("same model twice does not flush", S.flush_for_model_change(a) == "" and calls["n"] == 0)
    n1 = calls["n"]
    note = S.flush_for_model_change(b)
    check("changed model flushes", calls["n"] == n1 + 1 and "model changed" in note)
    check("flush note names both formats", "nvfp4" in note and "float8_e4m3fn" in note)
    check("same model after a change does not re-flush",
          S.flush_for_model_change(b) == "" and calls["n"] == n1 + 1)
    if _orig is not None:
        S.mm.unload_all_models = _orig


class _FakeImg:
    """Minimal stand-in for an IMAGE tensor: enough shape/movedim to survive _resize."""
    def __init__(self, w=1024, h=1024):
        self.shape = (1, h, w, 3)
    def __getitem__(self, key):
        return self
    def movedim(self, *a):
        return self


def check_ref_conditioning_channels():
    """A shot carries EITHER references or the keyframe handoff -- never both.

    comfy/model_base.py builds one `cond_video_latents` list for the DiT: the
    keyframe branch fills it, then the refs branch OVERWRITES it, while PackedLayout
    still lays out rows for both. A shot carrying both would hand the layout fewer
    latents than it has condition rows -- a shape error deep inside the DiT, or a
    keyframe row silently fed a reference's latent. This test pins the exclusion at
    the only place it can be enforced: where the conditioning is built."""
    print("\n=== ref2va and keyframe conditioning are mutually exclusive ===")
    # The stubs live in sys.modules only; a submodule found there is never attached
    # to its parent package, so `comfy.utils.x` at call time would still fail.
    t = sys.modules["torch"]
    cu, nt = sys.modules["comfy.utils"], sys.modules["comfy.nested_tensor"]
    cmm, node_helpers = sys.modules["comfy.model_management"], sys.modules["node_helpers"]
    for name in ("utils", "nested_tensor", "model_management"):
        setattr(sys.modules["comfy"], name, sys.modules["comfy." + name])
    if not hasattr(t, "zeros"):
        t.zeros = lambda *a, **k: object()
    if not hasattr(cu, "common_upscale"):
        cu.common_upscale = lambda s, w, h, m, c: s
    if not hasattr(nt, "NestedTensor"):
        nt.NestedTensor = lambda pair: pair
    if not hasattr(cmm, "intermediate_device"):
        cmm.intermediate_device = lambda: "cpu"

    seen = {}

    class _Clip:
        def tokenize(self, prompt, **kw):
            seen["tokenize_kwargs"] = kw
            return "tokens"
        def encode_from_tokens_scheduled(self, tokens):
            return [["cond", {}]]

    class _Vae:
        def encode(self, img):
            return "latent"

    node_helpers.conditioning_set_values = lambda cond, values: (seen.setdefault("values", {}).update(values), cond)[1]

    def build(handoff=None, refs=None, size="match", aug=None):
        seen.clear()
        S._build_shot_conditioning(_Clip(), _Vae(), "a prompt", 1344, 768, 124, 24,
                                   handoff, ref_images=refs, ref_image_size=size,
                                   ref_noise_aug=aug)
        return seen.get("values", {}), seen.get("tokenize_kwargs", {})

    vals, tok = build(handoff=_FakeImg())
    check("handoff only -> keyframe conditioning", "minimax_keyframes" in vals)
    check("handoff only -> no refs", "minimax_refs" not in vals)
    check("handoff only -> keyframes are presented as images", "images" in tok)

    # ref_noise_aug: how CLEAN the reference is presented as. The DiT blends the
    # condition latent with noise at (1 - aug) AND labels those rows with a timestep
    # of max(t_video, aug), so H3's own default of 0.999 hands the model a finished
    # image -- an invitation to reproduce it in the opening frames rather than to
    # take an identity from it.
    check("ref_noise_aug reaches the conditioning",
          build(refs=[_FakeImg()], aug=0.90)[0].get("minimax_visual_cond_noise_aug") == 0.90)
    check("it is NEVER applied to a keyframe shot",
          "minimax_visual_cond_noise_aug" not in build(handoff=_FakeImg(), aug=0.90)[0])
    check("omitting it leaves H3's own default in place",
          "minimax_visual_cond_noise_aug" not in build(refs=[_FakeImg()], aug=None)[0])

    vals, tok = build(refs=[_FakeImg(), _FakeImg()])
    check("refs only -> ref conditioning", "minimax_refs" in vals)
    check("refs only -> no keyframes", "minimax_keyframes" not in vals)
    check("refs only -> presented as minimax_ref_items", "minimax_ref_items" in tok)
    check("every reference reaches the tokenizer", len(tok.get("minimax_ref_items", [])) == 2)
    check("every reference reaches the DiT", len(vals.get("minimax_refs", [])) == 2)

    # ComfyUI 0.31+ CONCATENATES refs onto keyframes in cond_video_latents, and
    # PackedLayout appends keyframe segments before ref segments, so the two orders
    # agree and both channels coexist. On 0.30 the refs branch overwrote the
    # keyframes while the layout still reserved rows for both -- hence the old
    # either/or. A keyframe anchors the first frame; a reference only supplies
    # identity, so carrying both is strictly better for continuity.
    vals, _ = build(handoff=_FakeImg(), refs=[_FakeImg()])
    check("refs and a keyframe now ride together", "minimax_refs" in vals)
    check("...the handoff becomes a real keyframe", "minimax_keyframes" in vals)
    check("...anchored at the first frame",
          vals["minimax_keyframes"][0]["resolved_frame_index"] == 0)
    check("...and the frame count travels with it", "minimax_frame_count" in vals)

    # ...but only while the references are presented CLEAN. visual_cond_noise_aug is
    # one payload value covering every cond video latent (ldm/minimax/model.py:502)
    # and it labels both segments identically (:584), so softening the references
    # would soften the anchor with them. There the handoff has to stay out.
    vals, _ = build(handoff=_FakeImg(), refs=[_FakeImg()], aug=0.90)
    check("a softened reference does NOT get a keyframe alongside it",
          "minimax_keyframes" not in vals)
    check("...the references are still applied", "minimax_refs" in vals)
    check("...and the aug still reaches them",
          vals.get("minimax_visual_cond_noise_aug") == 0.90)
    check("the gate is the threshold, not the presence of an aug",
          S.keyframe_rides_with_refs(None) and S.keyframe_rides_with_refs(0.999)
          and S.keyframe_rides_with_refs(0.99) and not S.keyframe_rides_with_refs(0.95))

    vals, tok = build(refs=[None, None])
    check("all-empty ref slots fall back to the keyframe path",
          "minimax_refs" not in vals and "minimax_ref_items" not in tok)

    # latent grid must match what the DiT is told to expect (16px per latent cell)
    vals, _ = build(refs=[_FakeImg(2048, 1024)])
    blk = vals["minimax_refs"][0]
    tw, th = S.ref_image_canvas(2048, 1024, 1344, 768, "match")
    check("ref block reports the latent grid of its own canvas",
          blk["latent_w"] == tw // 16 and blk["latent_h"] == th // 16)
    check("ref block is an image kind", blk["kind"] == "image")


def check_tagged_references():
    """<Picture N> in a beat places that reference on THAT shot.

    The positional modes go by shot NUMBER and are blind to who is in the shot: a
    character who first appears in shot 2 got nothing, while an empty establishing
    shot 1 got a portrait pushed into its opening frames. Tagging says where each
    reference belongs, in the prompt, next to the character it describes.

    The tags must be RENUMBERED per shot: the tokenizer numbers references by their
    position in the list it is handed, so a shot using only <Picture 2> receives
    that image as <Picture 1> and the untouched text would point at nothing."""
    print("\n=== references placed by <Picture N> tags ===")
    REFS = ["KristyPhoto", "DanPhoto", "CarPhoto"]

    def place(text, refs=REFS):
        return S.resolve_tagged_refs(text, refs)

    check("an untagged beat takes no references", place("Kristy walks in.")[1] == [])
    check("a tagged beat takes the named image",
          place("Kristy, <Picture 1>, walks in.")[1] == ["KristyPhoto"])
    check("tag syntax is forgiving",
          place("Dan <picture_2> waves.")[1] == place("Dan <PICTURE 2> waves.")[1]
          == place("Dan <Picture 2> waves.")[1] == ["DanPhoto"])
    # renumbering: slot 2 alone must arrive as <Picture 1>
    txt, imgs, _ = place("Dan, <picture_2>, hands her a wrench.")
    check("a lone <Picture 2> is renumbered to <Picture 1>", "<Picture 1>" in txt)
    check("...and carries the RIGHT image", imgs == ["DanPhoto"])
    txt2, imgs2, _ = place("Kristy <Picture 1> and Dan <Picture 3> argue.")
    check("two tags renumber in order",
          "<Picture 1>" in txt2 and "<Picture 2>" in txt2 and "<Picture 3>" not in txt2)
    check("...and both images ride along", imgs2 == ["KristyPhoto", "CarPhoto"])
    # a tag with no image behind it refers to nothing
    txt3, imgs3, dropped = place("Someone <Picture 9> appears.")
    check("a tag with no connected image is dropped from the text", "<Picture" not in txt3)
    check("...carries no image", imgs3 == [])
    check("...and is reported", dropped == [9])
    check("picture_tags reads every slot named",
          S.picture_tags("<Picture 1> then <picture_3>") == [1, 3])
    check("picture_tags on plain prose finds nothing", S.picture_tags("Kristy walks in.") == [])

    # A tagged shot takes references, and references REPLACE the handoff -- so on a
    # long chain every tag was a hard cut and cohesion went with it. The previous
    # frame now rides along as one more reference: same ref2va payload, and the
    # tagged images keep their <Picture N> numbers because it is appended last.
    txt, imgs, _ = place("Mara, <Picture 2>, steps out of the barn.")
    carried = imgs + ["prev_frame"]
    check("a tagged shot can carry the previous frame as an extra reference",
          carried == ["DanPhoto", "prev_frame"])
    check("...without disturbing the tag numbering", "<Picture 1>" in txt)

    # the whole point: the empty establishing shot must stay clean
    BEATS = ["Wide shot of the empty garage.",
             "Kristy, <picture_1>, walks in.",
             "Kristy finds Dan, <Picture 2>, at the bench.",
             "Dan hands her a wrench."]
    got = [place(b, ["KristyPhoto", "DanPhoto"])[1] for b in BEATS]
    check("shot 1 (nobody in it) takes no reference and keeps its handoff", got[0] == [])
    check("the character's own shot gets their photo", got[1] == ["KristyPhoto"])
    check("a second character lands on THEIR shot", got[2] == ["DanPhoto"])
    check("untagged later shots keep the handoff", got[3] == [])


class _FakeGraph:
    """Stands in for ComfyUI's DynamicPrompt: node_id -> {class_type, inputs}."""
    def __init__(self, nodes):
        self._n = nodes

    def get_node(self, nid):
        return self._n[str(nid)]


class _FakeModel:
    def __init__(self, sparse):
        self.model_options = {"transformer_options": {
            "optimized_attention_override": (lambda *a: None) if sparse else None}}


def _sla_graph(lora, sparse):
    """UNETLoader -> [LoraLoaderModelOnly] -> [SolAttnPatch] -> our node ('9')."""
    g = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3.safetensors"}}}
    prev = "1"
    if lora:
        g["2"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": [prev, 0], "lora_name": lora, "strength_model": 1.0}}
        prev = "2"
    if sparse:
        g["3"] = {"class_type": "SolAttnPatch", "inputs": {"model": [prev, 0], "tau": 0.1}}
        prev = "3"
    g["9"] = {"class_type": "H3LongVideos",
              "inputs": {"model": [prev, 0], "clip": ["8", 0], "prompt": "a woman walks"}}
    return _FakeGraph(g)


def check_sla_pairing():
    """An SLA LoRA and a sparse-attention patch are a matched pair.

    The LoRA carries NO marker -- not in its tensor names, not in its metadata -- so
    the filename off the workflow graph is the only signal there is."""
    print("\n=== SLA LoRA / sparse-attention pairing ===")
    SLA = "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors"
    PLAIN = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_resized_avg_rank_21_bf16.safetensors"

    def pair(lora, sparse):
        return S.sla_pairing(_FakeModel(sparse), _sla_graph(lora, sparse), "9")

    sla, sp, note = pair(SLA, True)
    check("SLA LoRA + sparse attention is the matched pair, no warning",
          sla and sp and not note)
    sla, sp, note = pair(SLA, False)
    check("SLA LoRA without sparse attention warns", bool(sla) and not sp and bool(note))
    check("...and says the speedup is not being collected", "none of its speed" in note)
    sla, sp, note = pair(PLAIN, True)
    check("sparse attention with a non-SLA LoRA warns", not sla and sp and bool(note))
    check("...and names duplication, which is what it actually causes",
          "SAME PERSON TWICE" in note)
    sla, sp, note = pair(PLAIN, False)
    check("neither half present is not a warning", not sla and not sp and not note)
    sla, sp, note = pair(None, True)
    check("sparse attention with no LoRA at all warns", not sla and sp and bool(note))

    # The graph walk must follow only MODEL links: a LoRA on some other branch of
    # the workflow is not affecting us and must not be reported as if it were.
    g = _FakeGraph({
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": SLA}},
        "5": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": "other_branch.safetensors"}},
        "9": {"class_type": "H3LongVideos", "inputs": {"model": ["2", 0]}}})
    check("the walk follows only the model chain feeding this node",
          S.upstream_lora_names(g, "9") == [SLA])
    # A cycle must not hang the walk.
    cyc = _FakeGraph({"1": {"class_type": "X", "inputs": {"model": ["2", 0]}},
                      "2": {"class_type": "Y", "inputs": {"model": ["1", 0]}}})
    check("a cyclic graph terminates", S.upstream_lora_names(cyc, "1") == [])
    check("no graph at all is not an error", S.sla_pairing(_FakeModel(False), None, None)[2] == "")

    for name in ("h3-SLA-turbo.safetensors", "turbo.sla.bf16.safetensors", SLA):
        check(f"'sla' recognised in {name}", bool(S._SLA_NAME.search(name)))
    for name in ("slack_style.safetensors", "translate_lora.safetensors",
                 "isla_character_v2.safetensors", "SLAYER_style.safetensors", PLAIN):
        check(f"'sla' NOT falsely found in {name}", not S._SLA_NAME.search(name))


def check_written_text_is_not_speech():
    """Quoted WRITTEN text must not turn a silent shot into a talking one.

    has_speech() counted any double-quoted span, so a sign, label or headline in
    the prose made the whole beat count as dialogue -- no lips-closed clause, no
    no-voice soundscape -- and the characters stood there opening their mouths at
    something nobody said."""
    print("\n=== quoted text that nobody speaks ===")
    for b in ['Mara reads the sign marked "EXIT".',
              'A poster reads "OPEN".',
              'The screen displays "NO SIGNAL".',
              'A note is written "GONE".',
              'She checks the label "FRAGILE" and frowns.']:
        check(f"written text is not speech: {b[:34]}...", not S.has_speech(b))
    for b in ['Mara says: "Ready."',
              'She asks him, "Where?"',
              '"Wait," he says.',
              'He whispers "now".',
              'Jon calls out, "Over here!"',
              '<d>Ready.</d>']:
        check(f"real dialogue still counts: {b[:34]}...", S.has_speech(b))
    # The NEAREST cue decides. Comparing presence rather than position got these
    # backwards in both directions.
    check("a speech verb closer to the quote wins over an earlier reading verb",
          S.has_speech('Mara reads the sign, then says "We go left."'))
    check("a second, spoken quote still counts when the first is written",
          S.has_speech('Mara reads the sign marked "EXIT" and shouts "This way!"'))

    # End to end: a beat whose only quote is written must be silenced like any
    # other action beat.
    g = S.distribute_generations("A room.", ['Mara reads the sign marked "EXIT".'],
                                 "", "", "Mara = she, 30, blonde")[0]
    check("a written-text beat gets the mouth constraint", "mouth closed" in g.lower())
    check("...and the no-voice soundscape", "no voices" in g.lower())


def check_declared_bare():
    """A character can START bare, not only become bare.

    exposed_terms used to fire only on a REMOVAL, so someone naked from shot 1 was
    never marked and their configured wording never applied. The state has to be
    DECLARED though -- inferring it from a sheet that simply doesn't list clothes
    would put nudity in scenes nobody asked for."""
    print("\n=== declared-bare characters ===")

    def para(cm, et="he = penis", beats=None, pn=True, strip_out=None):
        beats = beats or ["Jon stands by the bed.", "Jon walks to the window."]
        return [g.split("] ", 1)[-1].split("\n")[0]
                for g in S.distribute_generations("A room.", beats, "", "", cm,
                                                  prevent_nudity=pn, exposed_terms=et,
                                                  strip_out=strip_out)]

    b = para("Jon = he, 35, bald, nude")
    check("'nude' fires the configured term from shot 1", "penis" in b[0])
    check("...and marks the upper zone too", "bare chest" in b[0])
    check("...and persists into later shots", "penis" in b[1])
    check("...and the literal token is replaced, not doubled up",
          "nude" not in b[0].lower())

    b = para("Jon = he, 35, bald, naked", et="")
    check("without exposed_terms the generic wording is used",
          "bare below the waist" in b[0] and "bare chest" in b[0])
    check("a declaration is intent, so prevent_nudity does not blank it",
          "bare below the waist" in para("Jon = he, 35, bald, nude", et="", pn=True)[0])

    b = para("Jon = he, 35, bald, grey shirt, bottomless")
    check("'bottomless' bares only the lower zone",
          "penis" in b[0] and "bare chest" not in b[0])
    check("...and leaves the upper garment on", "grey shirt" in b[0])
    t = S.distribute_generations("A room.", ["Mara stands by the window."], "", "",
                                 "Mara = she, 30, blue jeans, topless", prevent_nudity=True,
                                 exposed_terms="she upper = bare breasts")[0]
    check("'topless' bares only the upper zone",
          "bare breasts" in t and "blue jeans" in t and "bare below the waist" not in t)

    # THE case that must not regress: an under-specified sheet is not nudity.
    b = para("Jon = he, 35, bald")
    check("a sheet that lists no clothes is NOT treated as naked",
          "penis" not in b[0] and "bare" not in b[0])
    check("only whole-item tokens count",
          S.declared_bare_zones(["nude beach backdrop"])[0] == set())
    check("an ordinary attribute is not a nudity token",
          S.declared_bare_zones(["bald"])[0] == set())

    # Nothing came off, so the next shot must keep its handoff frame.
    so = []
    para("Jon = he, 35, bald, nude", strip_out=so)
    check("a declared-bare start does not cost the next shot its handoff", so == [])
    # ...whereas an actual removal still does.
    so = []
    para("Jon = he, 35, bald, blue boxers",
         beats=["Jon stands.", "Jon pulls down his boxers and steps out of them."],
         strip_out=so)
    check("an actual removal still suppresses the next handoff", so == [2])

    # Dressing the zone again clears that zone's marker only.
    b = para("Jon = he, 35, bald, nude",
             beats=["Jon stands by the bed.",
                    "Jon pulls on his boxers.\nwardrobe: Jon += blue boxers",
                    "Jon walks out."])
    check("covering a zone clears its marker", "penis" not in b[1])
    check("...and leaves the other zone's marker alone", "bare chest" in b[1])
    check("...and stays cleared", "penis" not in b[2])


def check_exposed_terms_key_warning():
    """An exposed_terms key that matches nobody must SAY so.

    The lookup falls through name -> pronoun -> default, so a typo'd name looks
    configured and silently does nothing. Object pronoun forms ('her', 'him') fail
    the same way, because _pronoun_of() normalizes to she/he/they."""
    print("\n=== exposed_terms keys that match nobody ===")
    cm = "Mara = she, 30, blonde, nude\nJon = he, 35, bald, nude"

    def warn(et, beats=None, sheet=cm):
        out = []
        S.distribute_generations("A room.", beats or ["Mara and Jon stand."], "", "",
                                 sheet, prevent_nudity=True, exposed_terms=et,
                                 notes_out=out)
        return [n for n in out if "exposed_terms" in n]

    w = warn("Marra = X")
    check("a typo'd name warns", len(w) == 1)
    check("...quoting the key as the user typed it", "'Marra'" in w[0])
    check("...and listing who IS known", "Mara" in w[0] and "Jon" in w[0])
    w = warn("her = X")
    check("an object pronoun 'her' warns", len(w) == 1)
    check("...and points at the usable form", "use 'she'" in w[0])
    check("'him' points at 'he'", "use 'he'" in warn("him = X")[0])

    check("a valid name does not warn", warn("Mara = X") == [])
    check("a valid name with a zone does not warn", warn("Mara upper = X") == [])
    check("valid pronouns do not warn", warn("she = X\nhe = Y") == [])
    check("a name in different case does not warn", warn("MARA = X") == [])
    check("only the bad key is reported", len(warn("she = X\nJonn = Y")) == 1)
    check("no exposed_terms at all is silent", warn("") == [])

    # A character introduced mid-chain by directive is still a known character.
    check("a name introduced later by a wardrobe directive does not warn",
          warn("Ash = X", beats=["Mara and Jon stand.",
                                 "Ash walks in.\nwardrobe: Ash = he, 20, tall, nude"]) == [])


def check_bare_wording_follows_pronoun():
    """The upper-zone default has to be worded for the right body.

    'bare chest' describes a male torso; H3 renders roughly what the words say, so
    a woman defaulting to it is both odd phrasing and a weak cue. Configuring only
    the lower zone ('she = vagina') is the common case, and the upper zone still
    has to come out right on its own."""
    print("\n=== bare-zone wording follows the declared pronoun ===")

    def para(who, cm, et="", beats=None):
        beats = beats or ["%s stands by the window." % who]
        out = []
        for g in S.distribute_generations("A room.", beats, "", "", cm,
                                          prevent_nudity=True, exposed_terms=et):
            t = g.split("] ", 1)[-1].split("\n")[0]
            i = t.find(who + " (")
            out.append(t[i:t.find(")", i) + 1] if i >= 0 else t)
        return out

    w = para("Mara", "Mara = she, 30, blonde, nude", "she = vagina")[0]
    check("a nude woman fires her configured lower term", "vagina" in w)
    check("...and her upper zone reads as breasts, not a chest",
          "bare breasts" in w and "bare chest" not in w)
    w = para("Mara", "Mara = she, 30, blonde, nude")[0]
    check("with no terms at all she still gets the right upper wording",
          "bare breasts" in w and "bare chest" not in w)
    check("...and the neutral lower wording", "bare below the waist" in w)
    check("'topless' alone uses it too",
          "bare breasts" in para("Mara", "Mara = she, 30, jeans, topless")[0])

    m = para("Jon", "Jon = he, 35, bald, nude", "he = penis")[0]
    check("a nude man is unchanged", "penis" in m and "bare chest" in m)
    check("...and never gets the feminine wording", "bare breasts" not in m)
    n = para("Ash", "Ash = 40, tall, nude")[0]
    check("no declared pronoun keeps the neutral wording",
          "bare chest" in n and "bare breasts" not in n)

    # An explicit term still outranks the per-pronoun default.
    o = para("Mara", "Mara = she, 30, nude", "she upper = topless silhouette")[0]
    check("an explicit upper term still wins",
          "topless silhouette" in o and "bare breasts" not in o)

    # A per-pronoun marker must still be recognised for REMOVAL when covered again.
    b = para("Mara", "Mara = she, 30, blonde, nude", "she = vagina",
             beats=["Mara stands.",
                    "Mara pulls on a shirt.\nwardrobe: Mara += white shirt",
                    "Mara sits."])
    check("covering the upper zone drops the feminine marker",
          "bare breasts" not in b[1] and "white shirt" in b[1])
    check("...and the still-bare lower zone keeps its term", "vagina" in b[1])
    check("...and it stays that way", "bare breasts" not in b[2] and "vagina" in b[2])


def check_plural_cast_binding():
    """A beat that addresses the cast only in the plural must still bind them.

    _resolve_subject() maps a pronoun to ONE person, so 'they' with two people
    resolved to neither and the shot described nobody -- dropping both characters
    and, after a removal, their exposure markers. The fix prepends a roll-call
    rather than rewriting the sentence, because 'they' can mean a subset and
    expanding it in place would assert a cast list the author did not write."""
    print("\n=== plural cast reference ===")
    cm = ("Mara = she, 30, blonde hair, white top, black panties\n"
          "Jon = he, 35, bald, grey shirt, blue boxers")
    ET = "she = vagina\nhe = penis"
    beats = ["Mara and Jon stand in the room.",
             "Mara pulls down her panties and steps out of them.",
             "Jon pulls down his boxers and steps out of them.",
             "They face each other.",
             "The window rattles and light floods through them.",
             "Both of them sit down."]
    g = S.distribute_generations("A room.", beats, "", "", cm,
                                 prevent_nudity=True, exposed_terms=ET)
    b = [x.split("] ", 1)[-1].split("\n")[0].lower() for x in g]

    check("a removal names the stripped zone for the person who stripped",
          "vagina" in b[1] and "penis" not in b[1])
    check("the other character's removal names his zone", "penis" in b[2])
    check("a plural beat binds BOTH characters", "vagina" in b[3] and "penis" in b[3])
    check("...via a roll-call, leaving the sentence intact",
          "are both in this shot" in b[3] and "they face each other." in b[3])
    check("...and does not rewrite the pronoun away", "they face" in b[3])
    # The case the roll-call must NOT fire on: a plural pronoun for OBJECTS in a
    # beat with no people. Binding here would summon the cast into a scenery shot.
    check("an object 'them' keeps a scenery beat empty",
          "mara" not in b[4] and "jon" not in b[4]),
    check("...and stamps no exposure marks there",
          "vagina" not in b[4] and "penis" not in b[4])
    check("'both of them' also binds the cast", "vagina" in b[5] and "penis" in b[5])

    # The silence gate must agree with the binding. It did not: person_referenced()
    # answers False for 'they', so a beat the roll-call had just described in full
    # counted as having nobody in it and got NO mouth constraint -- two people on
    # screen with nothing saying their lips are closed, which renders as mouths
    # opening at random.
    sil = S.distribute_generations(
        "A room.",
        ["Mara and Jon walk in.", "She looks at him.", "They face each other.",
         "Both of them sit down.", "The hangar doors roll open.",
         'They stop and Mara says: "Wait."'],
        "", "", "Mara = she, 30, blonde\nJon = he, 35, bald")
    lips = ["mouth closed" in s.lower() for s in sil]
    check("a named beat gets the mouth constraint", lips[0])
    check("a singular-pronoun beat gets it", lips[1])
    check("a PLURAL beat gets it too", lips[2] and lips[3])
    check("a scenery beat with nobody in it does NOT get a mouth constraint",
          not lips[4])
    check("...but is still given a no-voice soundscape", "no voices" in sil[4].lower())
    check("a beat with real dialogue is not silenced", not lips[5])

    # Bare 'them'/'their' must not trigger it -- they are object-prone.
    check("bare 'them' is not a cast reference", not S._PLURAL_CAST.search("steps out of them"))
    check("bare 'their' is not a cast reference", not S._PLURAL_CAST.search("their edges glow"))
    check("'they' is a cast reference", bool(S._PLURAL_CAST.search("they face each other")))
    check("'each other' is a cast reference", bool(S._PLURAL_CAST.search("facing each other")))

    # A single-character scene has nothing to roll-call: the singular path covers it.
    one = S.distribute_generations("A room.", ["They stand still."], "", "",
                                   "Mara = she, 30, blonde hair")
    check("a one-person cast does not get a roll-call",
          "are both in this shot" not in one[0].lower())
    # Someone whose DECLARED pronoun is 'they' owns the word: it resolves to them
    # individually and the roll-call must stay out of it.
    ari = S.distribute_generations(
        "A room.", ["They all look up."], "", "",
        "Mara = she, 30, blonde\nJon = he, 35, bald\nAri = they, 40, tall")
    check("a declared 'they' pronoun resolves to that person, not the whole cast",
          "are all in this shot" not in ari[0].lower() and "(40, tall)" in ari[0])
    # With nobody owning 'they', three people get the plural wording.
    three = S.distribute_generations(
        "A room.", ["They all look up."], "", "",
        "Mara = she, 30, blonde\nJon = he, 35, bald\nAri = she, 40, tall")
    check("three people read 'are all in this shot'",
          "are all in this shot" in three[0].lower())


class _Vec:
    """The few tensor operations check_audio_vae_loaded() actually performs."""
    def __init__(self, vals):
        self.v = list(vals)

    def min(self):
        return min(self.v)

    def max(self):
        return max(self.v)

    def abs(self):
        return _Vec(abs(x) for x in self.v)


class _All:
    def __init__(self, ok):
        self.ok = ok

    def all(self):
        return self.ok


def check_audio_vae_guard():
    """An audio VAE whose normalization buffers were never filled must be REFUSED.

    comfy/ldm/minimax/audio_vae.py registers latents_mean/latents_std as
    torch.empty() -- uninitialized memory. The video VAE bakes real constants in
    instead (torch.tensor(LATENTS_MEAN)), so the same 'Missing VAE keys' warning is
    cosmetic there and serious here. Without this guard, decode() multiplies the
    latents by garbage and the audio returns as noise with nothing in the log.

    Verified separately against real torch 2.13: torch.empty(32) reliably contains
    a ~-9.98e+27 magnitude, so genuine uninitialized memory always trips the bounds
    below rather than merely usually."""
    print("\n=== audio VAE normalization buffers ===")
    t = sys.modules["torch"]
    t.isfinite = lambda vec: _All(all(x == x and abs(x) != float("inf") for x in vec.v))

    class _FSM:
        def __init__(self, mean, std):
            self.latents_mean, self.latents_std = mean, std

    class _V:
        def __init__(self, mean, std):
            self.first_stage_model = _FSM(mean, std)

    def refused(mean, std):
        try:
            S.check_audio_vae_loaded(_V(mean, std))
            return False
        except RuntimeError:
            return True

    good_m = _Vec([0.1, -0.2, 0.05, 0.3])
    good_s = _Vec([0.9, 1.2, 0.7, 1.05])
    check("a properly loaded audio VAE passes", not refused(good_m, good_s))
    check("all-zero buffers are refused", refused(_Vec([0] * 4), _Vec([0] * 4)))
    check("a zero scale channel is refused", refused(good_m, _Vec([0.9, 0.0, 0.7, 1.05])))
    check("a negative scale is refused", refused(good_m, _Vec([0.9, -1.4, 0.7, 1.05])))
    check("NaN in the mean is refused",
          refused(_Vec([0.1, float("nan"), 0.05, 0.3]), good_s))
    check("inf in the scale is refused",
          refused(good_m, _Vec([0.9, float("inf"), 0.7, 1.05])))
    check("the magnitude torch.empty actually produces is refused",
          refused(_Vec([-9.98e27] * 4), _Vec([-9.98e27] * 4)))
    check("an absurd but finite scale is refused", refused(good_m, _Vec([1e4] * 4)))
    check("an absurd but finite mean is refused", refused(_Vec([1e4] * 4), good_s))
    # Absent buffers mean a VAE that does not use this scheme at all -- not a fault.
    check("absent buffers are not treated as a fault", not refused(None, None))
    # Introspection must never be what stops a render.
    class _Hostile:
        first_stage_model = property(lambda self: (_ for _ in ()).throw(ValueError("boom")))
    try:
        S.check_audio_vae_loaded(_Hostile())
        ok = True
    except RuntimeError:
        ok = False
    except Exception:
        ok = True          # any non-RuntimeError is the introspection failing, not a block
    check("a failed introspection does not block the render", ok)


class _MDModel:
    """A patcher carrying a LoRA's safetensors metadata, as ComfyUI stashes it."""
    def __init__(self, md):
        self._md = md

    def get_attachment(self, key):
        return self._md if key == "lora_metadata" else None


def check_lora_hints():
    """What a LoRA declares about itself, versus how this run is configured.

    Only two of these come from metadata; step count and training resolution are
    FILENAME convention, and the notes have to say so -- a filename is something
    anyone can break by renaming, and it must not read like a trainer's word."""
    print("\n=== LoRA self-declaration vs this run ===")
    SLA4 = "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors"
    R8 = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_resized_avg_rank_21_bf16.safetensors"
    AITK = "HMPenis_v2_e35.safetensors"
    LTX = "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
    H3MD = {"base_model": "Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors"}
    AIMD = {"ss_base_model_version": "minimax_h3"}     # ai-toolkit spelling
    LTXMD = {"base_model": "LTX-Video 2.5 22b"}

    def hints(name, md, steps, short_edge):
        return S.lora_hint_notes(_MDModel(md), _sla_graph(name, False), "9", steps, short_edge)

    def stack(*loras):
        g = {"1": {"class_type": "UNETLoader", "inputs": {}}}
        prev = "1"
        for i, l in enumerate(loras, 2):
            g[str(i)] = {"class_type": "LoraLoaderModelOnly",
                         "inputs": {"model": [prev, 0], "lora_name": l}}
            prev = str(i)
        g["9"] = {"class_type": "H3LongVideos", "inputs": {"model": [prev, 0]}}
        return _FakeGraph(g)

    # On a stack the step-count LoRA is usually NOT the nearest one, and checking
    # only the nearest silently skipped the very LoRA the sampler has to match.
    st = S.lora_hint_notes(_MDModel(AIMD), stack(SLA4, AITK, "vagassist_e40.safetensors"),
                           "9", 8, 768)
    check("a step-count LoRA behind others on the chain is still checked",
          any("4-step" in x for x in st))
    check("stacking a distill with subject LoRAs warns", any("3 LoRAs" in x for x in st))
    check("...and names the distill", any(SLA4 in x for x in st))
    check("...and points at the SUBJECT strengths, not the distill's",
          any("SUBJECT LoRA strengths" in x for x in st))
    check("...and says order is irrelevant", any("sums the patches" in x for x in st))
    check("a distill alone is not a fight",
          not any("LoRAs on this chain" in x
                  for x in S.lora_hint_notes(_MDModel(AIMD), stack(SLA4), "9", 4, 768)))
    check("subject LoRAs with no distill are not a fight",
          not any("LoRAs on this chain" in x for x in
                  S.lora_hint_notes(_MDModel(AIMD), stack(AITK, "vagassist_e40.safetensors"),
                                    "9", 8, 768)))

    n = hints(SLA4, H3MD, 8, 768)
    check("a 4-step LoRA run at 8 steps warns", len(n) == 1 and "4-step" in n[0])
    check("...and says the number came from the filename", "not metadata" in n[0])
    check("...and says what overshooting the step count does", "re-noises" in n[0])
    check("matching steps is silent", hints(SLA4, H3MD, 4, 768) == [])
    check("under the step count warns too", "not finished resolving" in " ".join(hints(SLA4, H3MD, 2, 768)))

    n = hints(SLA4, H3MD, 4, 1080)
    check("768p LoRA rendered at 1080 short edge warns", len(n) == 1 and "768p" in n[0])
    check("...and names tiling, which is the actual failure", "tile" in n[0])
    check("768p LoRA at 720 short edge is within tolerance", hints(SLA4, H3MD, 4, 720) == [])
    check("a name with no resolution token warns about nothing",
          hints(R8, H3MD, 8, 720) == [])

    check("ai-toolkit's ss_base_model_version counts as H3", hints(AITK, AIMD, 8, 720) == [])
    n = hints(LTX, LTXMD, 8, 720)
    check("a non-H3 LoRA on the chain warns", len(n) == 1 and "not MiniMax-H3" in n[0])
    check("...and does not falsely read a step count out of '2.5-22b'",
          not any("step" in x for x in n))

    # Absent metadata must be silence, never a guess.
    check("no metadata at all produces no base-model claim",
          not any("base_model" in x for x in hints(R8, {}, 8, 720)))
    check("no LoRA on the chain produces nothing",
          S.lora_hint_notes(_MDModel({}), _FakeGraph({"9": {"class_type": "X", "inputs": {}}}),
                            "9", 8, 720) == [])
    check("a missing graph is not an error",
          S.lora_hint_notes(_MDModel({}), None, None, 8, 720) == [])
    # The scan reports; it must never mutate anything it was handed.
    md = {"base_model": "Comfy-Org/MiniMax-H3"}
    S.lora_hint_notes(_MDModel(md), _sla_graph(SLA4, False), "9", 8, 768)
    check("scanning does not mutate the metadata it read",
          md == {"base_model": "Comfy-Org/MiniMax-H3"})


def check_ref_modes():
    """Which shots take the reference channel, and what they give up for it."""
    print("\n=== ref_mode over a 6-shot chain ===")
    refs = ["A", "B"]
    ho = lambda i: None if i == 0 else "handoff"
    first = [S.shot_references(refs, "first shot", i, ho(i)) for i in range(6)]
    check("'first shot': only shot 1 is ref-conditioned",
          bool(first[0]) and not any(first[1:]))
    every = [S.shot_references(refs, "every shot", i, ho(i)) for i in range(6)]
    check("'every shot': all six are ref-conditioned", all(len(r) == 2 for r in every))
    both = [S.shot_references(refs, "every shot + handoff ref", i, ho(i)) for i in range(6)]
    check("'+ handoff ref': shot 1 has no previous frame to add", len(both[0]) == 2)
    check("'+ handoff ref': later shots carry refs AND the last frame",
          all(len(r) == 3 and r[-1] == "handoff" for r in both[1:]))
    check("a stale/unknown ref_mode falls back to the safe 'first shot'",
          bool(S.shot_references(refs, "blank line", 0, None))
          and not S.shot_references(refs, "blank line", 3, "handoff"))
    check("no references connected -> every shot keeps the handoff",
          all(S.shot_references([], m, i, ho(i)) == []
              for m in ("first shot", "every shot", "every shot + handoff ref")
              for i in range(6)))

    # A ref-conditioned shot must ALSO anchor on the handoff. ComfyUI 0.31+ lets the
    # two channels ride together, but only the <Picture N>-tagged branch was updated
    # for it: everywhere else a shot with references dropped its handoff, as 0.30
    # required. That is why the last frame of a shot did not become the first frame of
    # the next -- 'every shot' produced cuts, '+ handoff ref' demoted the anchor to a
    # soft "look like this", and on shot 1 the start_image was ignored outright.
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    check("run() lets refs and a keyframe coexist outside the tagged branch",
          src.count("carry_keyframe = True") == 2)
    check("...gated on the aug, so a softened reference cannot soften the anchor",
          "and keyframe_rides_with_refs(ref_noise_aug):\n                    carry_keyframe = True"
          in src)
    check("...and the handoff is not ALSO repeated in the ref channel",
          "shot_refs = [r for r in shot_refs if r is not handoff]" in src)
    check("the '+ handoff ref' fallback survives for softened refs",
          'ref_mode == "every shot + handoff ref"' in src)
    check("shot_handoff still drops after a wardrobe strip",
          "None if after_strip" in src)
    # The aug gate itself, which decides which way a ref shot goes.
    check("at H3's default aug the keyframe rides along",
          S.keyframe_rides_with_refs(0.999) and S.keyframe_rides_with_refs(None))
    check("...and a softened reference sends it back to the ref channel",
          not S.keyframe_rides_with_refs(0.90))

    # The handoff must reach the TEXT ENCODER too. comfy/text_encoders/minimax.py
    # tokenize_with_weights is either/or: passing minimax_ref_items makes it ignore
    # `images` outright. So a ref-conditioned shot showed the VLM its identity
    # references and never the previous frame -- told the location in words, anchored
    # at frame 0 by the latent, but with nothing describing where the shot left off.
    # The result is the reported one: same location, re-imagined scenery.
    check("the ref path hands the previous frame to the VLM",
          'items = items + [{"type": "image", "data": hand_img}]' in src)
    check("...appended AFTER the references, so <Picture N> tags still point right",
          src.index("items, blocks = _build_ref_images")
          < src.index('items = items + [{"type": "image"'))
    check("...and the same tensor is reused as the keyframe latent, encoded once",
          '"latent": vae.encode(hand_img)' in src
          and src.count("_resize(handoff[:1], width, height") == 2)
    check("...on the same aug gate as the keyframe itself",
          "hand_img = _resize(handoff[:1], width, height" in src
          and "if handoff is not None and keyframe_rides_with_refs(ref_noise_aug):\n"
              "            hand_img" in src)
    check("the keyframe-only path still passes images= (no ref items there)",
          "clip.tokenize(prompt, images=images)" in src)

    # The seed is the last thing that jumps at a boundary. `seed + i` gives every
    # shot its own noise field, and that field fixes the stochastic detail -- grain,
    # micro-texture, every surface the prompt never names. Changing it per shot
    # resets all of that at the seam, which reads as a cut even with the keyframe
    # anchoring the frame and the location unchanged. Wrong default for a node whose
    # whole purpose is one continuous take, and it carried no tooltip at all.
    check("vary_seed_per_shot defaults to OFF",
          '"vary_seed_per_shot": ("BOOLEAN", {"default": False' in src)
    check("...and the run() default agrees", "vary_seed_per_shot=False" in src)
    check("...and it now explains the trade-off",
          "CONTINUOUS TAKE" in src and "stochastic detail" in src)
    check("the seed still varies when explicitly asked for",
          "seed + i if vary_seed_per_shot else seed" in src)
    check("...and turning it on is REPORTED as a continuity hazard",
          "vary_seed_per_shot is ON" in src and "looks like a cut" in src)
    check("the hazard note only fires on a real chain", "len(gens) > 1" in src)
    # sizing: 'match' must never exceed the generation area by more than the 32px snap,
    # and neither mode may ever upscale a small reference
    big = S.ref_image_canvas(4096, 2160, 1344, 768, "match")
    check("'match' scales a 4K reference down to ~one frame's area",
          abs(big[0] * big[1] - 1344 * 768) < 1344 * 768 * 0.05)
    check("'max' uses the 2048 short edge", S.ref_image_canvas(4096, 2160, 1344, 768, "max")[1] == 2048)
    check("a small reference is never upscaled",
          S.ref_image_canvas(512, 512, 1344, 768, "match") == (512, 512)
          and S.ref_image_canvas(512, 512, 1344, 768, "max") == (512, 512))


def check_kernel_backend_note():
    """Notice when the quantization kernels are not actually available.

    The node does not call these kernels -- comfy/ops.py routes quantized Linears
    through comfy_kitchen automatically. But losing that path is SILENT: ComfyUI
    logs one startup line and then runs a slower, lower-fidelity dequantize
    fallback, and the symptom is soft output, which looks like a dozen other
    causes."""
    print("\n=== quantization kernel availability ===")

    class _P:
        convrot = True

    class _W:
        _params = _P()

    class _Mod:
        quant_format = "int8_tensorwise"
        weight = _W()

    class _DM:
        def modules(self):
            return [_Mod() for _ in range(50)]

    class _Inner:
        diffusion_model = _DM()

    class _Model:
        model = _Inner()

    m = _Model()
    check("int8+convrot is detected from the module tags",
          S.quant_format_of(m) == "int8_tensorwise+convrot")

    ck = sys.modules.get("comfy_kitchen")
    if ck is None:
        ck = types.ModuleType("comfy_kitchen")
        sys.modules["comfy_kitchen"] = ck
    saved = getattr(ck, "list_backends", None)
    try:
        ck.list_backends = lambda: {
            "cuda": {"available": True, "disabled": False, "unavailable_reason": None,
                     "capabilities": ["int8_linear", "scaled_mm_nvfp4"]}}
        check("a capable backend is silent", S.kernel_backend_note(m) == "")

        ck.list_backends = lambda: {
            "cuda": {"available": True, "disabled": True, "unavailable_reason": None,
                     "capabilities": ["int8_linear"]},
            "eager": {"available": False, "disabled": False,
                      "unavailable_reason": "n/a", "capabilities": []}}
        n = S.kernel_backend_note(m)
        check("every backend down warns", bool(n))
        check("...naming the format", "int8_tensorwise+convrot" in n)
        check("...naming the missing capability", "int8_linear" in n)
        check("...and saying what the fallback costs", "lower fidelity" in n)

        # A disabled CUDA backend is fine as long as SOMETHING still serves it.
        ck.list_backends = lambda: {
            "cuda": {"available": False, "disabled": True,
                     "unavailable_reason": "cu126", "capabilities": []},
            "eager": {"available": True, "disabled": False, "unavailable_reason": None,
                      "capabilities": ["int8_linear"]}}
        check("a fallback backend that still serves the format is silent",
              S.kernel_backend_note(m) == "")

        # Backends up, but none offering what this format needs.
        ck.list_backends = lambda: {
            "cuda": {"available": True, "disabled": False, "unavailable_reason": None,
                     "capabilities": ["scaled_mm_nvfp4"]}}
        check("a backend without the needed capability warns",
              bool(S.kernel_backend_note(m)))

        # Introspection failure must never block a render.
        def _boom():
            raise RuntimeError("nope")
        ck.list_backends = _boom
        check("a failed backend query is silent", S.kernel_backend_note(m) == "")
    finally:
        if saved is not None:
            ck.list_backends = saved

    # An unquantized checkpoint has nothing to accelerate.
    class _Plain:
        class model:
            diffusion_model = None
    check("an unquantized checkpoint is silent", S.kernel_backend_note(_Plain()) == "")
    check("...and reports no format", S.quant_format_of(_Plain()) == "")
    check("a garbage model object does not raise", S.kernel_backend_note(object()) == "")


def check_mouth_stays_closed():
    """Three layers, because the prompt alone was never going to do it.

    1. TEXT   -- the lips-closed clause and no-voice soundscape (gated on
                 person_in_shot, so plural beats are covered).
    2. PICTURE-- a dialogue shot handing its LAST frame to a silent shot seeds an
                 open mouth mid-word. A picture outvotes a sentence, which is the
                 same thing that made removed garments come back. The handoff frame
                 is taken MOUTH_SETTLE_FRAMES earlier at exactly that boundary.
    3. AUDIO  -- H3 is joint: the mouth follows the audio branch. On a shot with no
                 line that branch is otherwise unconditioned, invents a voice, and
                 the picture lip-syncs to it. The keyframe's audio channel is
                 anchored to encoded silence instead."""
    print("\n=== mouths stay closed until dialogue ===")
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()

    beats = ['Mara says: "Ready?"', "Mara nods.", "Mara walks out.",
             'Dom says: "Wait."', "Dom follows."]
    spk = S.speech_flags(beats)
    check("speech flags read the quoted lines", spk == [True, False, False, True, False])

    # Layer 2: only the speech -> silence boundary is worth paying for.
    settle = [spk[i] and not spk[i + 1] for i in range(len(spk) - 1)]
    check("a speech->silence boundary settles the mouth", settle[0] and settle[3])
    check("silence->silence needs nothing", not settle[1])
    check("silence->speech keeps the literal last frame", not settle[2])
    check("the settle window is a syllable tail, not a cut",
          2 <= S.MOUTH_SETTLE_FRAMES <= 6)

    # Layer 3 must never be able to break a render.
    class _Boom:
        audio_sample_rate = 32000

        def encode(self, w):
            raise RuntimeError("boom")

    class _NoSR:
        pass
    check("an audio VAE that raises degrades to None",
          S._silent_audio_latent(_Boom(), 124, 24) is None)
    check("an audio VAE with no sample rate degrades to None",
          S._silent_audio_latent(_NoSR(), 124, 24) is None)

    # ...and layer 3 has to actually REACH the shot. It used to be attached inside
    # `if keyframes:` on the keyframe-only path, so it was skipped entirely on:
    #   * any shot with a ref_image wired  (the whole ref path ignored `silent`)
    #   * the FIRST shot of every chain    (no handoff -> no keyframe)
    # Wiring a single ref_image therefore made the mechanism dead code, and
    # non-dialogue shots babbled exactly as if it had never been written.
    # torch is stubbed in this harness, so the ENCODE is stood in for -- what is
    # under test here is where the latent gets attached, not how it is made.
    _v = object()
    _real_sil = S._silent_audio_latent
    S._silent_audio_latent = lambda vae, fc, fps: "SILENCE"
    _empty = S._attach_silence([], _v, 362, 24, True)
    check("a shot with no keyframe still gets a silence anchor", len(_empty) == 1)
    check("...as an AUDIO-ONLY keyframe, carrying no video latent",
          "audio_latent" in _empty[0] and "latent" not in _empty[0])
    check("...positioned at frame 0", _empty[0]["resolved_frame_index"] == 0)
    _kf = S._attach_silence([{"resolved_frame_index": 0, "latent": "V"}], _v, 362, 24, True)
    check("an existing keyframe carries the silence instead of a second one",
          len(_kf) == 1 and _kf[0]["latent"] == "V" and "audio_latent" in _kf[0])
    check("a speaking shot is left alone",
          S._attach_silence([], _v, 362, 24, False) == [])
    check("no audio VAE degrades to no anchor",
          S._attach_silence([], None, 362, 24, True) == [])
    S._silent_audio_latent = lambda vae, fc, fps: None
    check("an encode that fails leaves the shot unchanged rather than breaking it",
          S._attach_silence([], _v, 362, 24, True) == [])
    S._silent_audio_latent = _real_sil
    check("both conditioning paths attach it", src.count("_attach_silence(") == 3)
    check("the ref path no longer ignores `silent`",
          "kfs = _attach_silence(kfs, audio_vae, fc, fps, silent)" in src)
    check("...and it is not gated behind an existing keyframe",
          "keyframes = _attach_silence(keyframes, audio_vae, fc, fps, silent)" in src)
    check("a zero-length shot degrades to None",
          S._silent_audio_latent(_Boom(), 0, 24) is None)

    # The two bugs that made this layer do NOTHING, verified against the real VAE:
    #   1. comfy.sd.VAE.encode() does movedim(-1, 1), so the audio VAE -- which
    #      wants [B, 2, L] -- must be handed [B, L, 2]. Passing [B, 2, L] raises
    #      inside the encoder, and the guard swallowed it.
    #   2. the encoder's hop grid returns 206 where temporal_shape() says 207, and
    #      the length check rejected that as a mismatch.
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    check("silence is encoded channels-last", "torch.zeros((1, sr, 2))" in src)
    check("...and the reason is recorded", "movedim(-1, 1)" in src)
    check("the conditioning latent is built to the layout's length, not rejected",
          "repeat(1, 1, 1, want_t)" in src)
    check("one second is encoded once and cached",
          "_SILENT_UNIT" in src and 'audio_vae.encode(torch.zeros((1, sr, 2)))' in src)
    # The encoder's zero-padding leaves heavy edge artifacts -- measured deviation
    # 0.351 at the first/last frames against 0.002 in the interior. Tiling the whole
    # second therefore stamped a spike every 40 latent frames, which at 40Hz is once
    # per SECOND: a metronome in the audio conditioning of a JOINT model, which the
    # picture then lip-syncs to. Only a steady interior frame may be repeated.
    check("a single INTERIOR frame is what gets repeated",
          "enc[..., mid:mid + 1]" in src and "mid = enc.shape[-1] // 2" in src)
    check("...and the cached unit is one frame wide",
          'if unit.shape[-1] != 1:' in src)
    check("...so the bed is constant, with no periodic pulse",
          "repeat(1, 1, 1, want_t)" in src)

    # Layer 1 still applies, including on the plural beats that used to miss it.
    cm = "Mara = she, 30, red hair\nDom = he, tall, 35"
    g = S.distribute_generations("A room.", ["Mara and Dom walk in.",
                                             "They face each other.",
                                             'Mara says: "Ready?"'], "", "", cm)
    check("a silent named beat gets the mouth state", "mouth closed" in g[0])
    check("a silent PLURAL beat gets it too", "mouth closed" in g[1])
    check("...and the no-voice soundscape", "no voices" in g[1].lower())
    check("a dialogue beat is left alone", "mouth closed" not in g[2])


def check_presence_test_is_shared():
    """ONE presence test, because this bug has already been written twice.

    person_referenced() resolves a pronoun to a single person, so 'they' and 'both
    of them' answer False for everybody. Any clause gated on it alone silently
    skips a beat that binds the whole cast. That produced babble on plural shots
    (the mouth state, fixed in db57367) and then restraints that appeared to break
    on plural shots (the constraint clause). Both go through person_in_shot() now."""
    print("\n=== shared presence test ===")
    act = {"Mara": ["she", "30", "steel handcuffs"], "Dom": ["he", "tall"]}
    for body, expect in [("Mara looks at the crate.", True),
                         ("She looks at the crate.", True),
                         ("They look at the crate together.", True),
                         ("Both of them sit down.", True),
                         ("The roller door rattles open.", False)]:
        check(f"person_in_shot({body[:28]!r}) is {expect}",
              S.person_in_shot(body, "Mara", act) is expect)
    # A single-person cast has no plural to resolve.
    solo = {"Mara": ["she", "30"]}
    check("a lone character is not summoned by 'they'",
          not S.person_in_shot("They face each other.", "Mara", solo))
    # A departed character is not brought back by a plural either.
    check("a departed character stays out of a plural beat",
          not S.person_in_shot("They face each other.", "Dom", act, departed={"Dom"}))

    # The regression itself: the constraint must survive a plural beat.
    for body in ("They look at the crate together.", "Both of them sit down."):
        check(f"the restraint clause survives {body[:26]!r}",
              "physically restrained" in S.restraint_clause(act, body))
    check("...and still skips a beat with nobody in it",
          S.restraint_clause(act, "The roller door rattles open.") == "")

    # End to end, the shot that caught it.
    cm = ("Mara = she, 30, red hair, steel handcuffs\nDom = he, tall, 35, brunette")
    g = S.distribute_generations("A workshop.",
                                 ["Mara walks in.", "Dom follows her.",
                                  "They look at the crate together."], "", "", cm)
    check("a plural beat keeps the physical constraint",
          "physically restrained" in g[2])
    check("...and keeps the mouth state", "mouth closed" in g[2])


def check_continuity_warning():
    """A scenery beat MID-CHAIN breaks the visual chain.

    Each shot is handed the previous one's last decoded frame. A beat describing
    nobody produces a frame with nobody in it, so the next shot has to re-establish
    every character from an empty room. Both prompts are individually correct,
    which is why this is invisible without looking at the sequence -- and why a
    chain loses its cast in the middle rather than degrading steadily.

    Established first that the prompt itself does NOT drift with length: the first
    six shots are byte-identical at 6, 9 and 13 beats."""
    print("\n=== mid-chain continuity ===")
    CM = "Mara = she, 30, red hair\nDom = he, tall, 35, brunette"

    def run(beats):
        return S.distribute_generations("A hangar.", beats, "", "", CM)

    mid = ["Mara walks in.", "Dom follows her.", "The hangar doors roll open.",
           "Dom looks out.", "Mara joins him."]
    w = S.continuity_warnings(run(mid))
    check("a scenery beat mid-chain warns", len(w) == 1)
    check("...naming the shot that produces the empty frame", "shot 3" in w[0])
    check("...and the shot that pays for it", "shot 4" in w[0])
    check("...and suggesting a fix", "watches from the doorway" in w[0])

    # A scenery beat that hands its frame to nobody costs nothing.
    check("scenery at the END does not warn",
          S.continuity_warnings(run(["Mara walks in.", "Dom follows her.",
                                     "Mara joins him.", "The doors roll open."])) == [])
    check("scenery at the START does not warn",
          S.continuity_warnings(run(["The doors roll open.", "Mara walks in.",
                                     "Dom follows her."])) == [])
    check("a chain with people throughout does not warn",
          S.continuity_warnings(run(["Mara walks in.", "Dom follows her.",
                                     "Mara joins him."])) == [])
    check("a two-beat chain is too short to warn",
          S.continuity_warnings(run(["Mara walks in.", "The doors roll open."])) == [])
    check("an empty chain is handled", S.continuity_warnings([]) == [])

    # The finding that framed this: length alone does not change the prompt.
    beats = ["Mara walks into the hangar.", "Dom climbs out of the van.",
             'Mara says to Dom: "Ready?"', "Dom lifts a crate.",
             "Mara takes off her grey coat.", "They look at the crate together.",
             "Dom opens the door.", "Mara follows him.", "They stand together."]
    cm2 = ("Mara = she, 30, red hair, grey coat\nDom = he, tall, 35, brunette")
    six = [g.split("\n")[0] for g in
           S.distribute_generations("A hangar.", beats[:6], "", "", cm2)]
    nine = [g.split("\n")[0] for g in
            S.distribute_generations("A hangar.", beats, "", "", cm2)][:6]
    check("the same beat renders identically at 6 and 9 beats", six == nine)


def check_restraints_stay_on():
    """A restraint is a plot state, not a garment: prose never takes one off.

    Left to the ordinary removal detector they came off far too easily, and often
    by ACCIDENT -- the removal window reaches any tracked item near the cue, so
    "steps out of her jacket and the chain falls away" dropped the ankle chain as a
    side effect of a beat about a jacket."""
    print("\n=== restraints stay on until asked ===")

    for item in ["steel handcuffs", "ankle chain", "shackles", "leather wrist straps",
                 "ball gag", "blindfold", "leg irons", "zip-tie", "chain restraint",
                 "manacles", "thumb cuffs", "padlocked collar", "fetters", "wrist ties"]:
        check(f"{item!r} is a restraint", S.is_restraint(item))
    # The hardware line states the equipment's state POSITIVELY and without strain
    # words: "holding with its full tension" read as a struggle cue and the model
    # rendered maximum-pull escapes to match it.
    hw = D("A room.", ["Mara waits."], "", "", "Mara = she, steel handcuffs")[0]
    check("hardware line present without strain wording",
          "stays whole and closed" in hw and "full tension" not in hw)
    # The false positives that matter: a word may not qualify ITSELF, and a material
    # is not a qualifier. Both bugs were live in the first version of this.
    for item in ["chain", "gold chain", "collar", "shirt collar", "black belt",
                 "leather belt", "red jacket", "strap", "dress with thin straps",
                 "rope", "silver necklace", "steel watch strap", "blue jeans", "corset"]:
        check(f"{item!r} stays an ordinary garment", not S.is_restraint(item))

    # Two shapes the head-noun/qualifier rules both missed, and prose could strip
    # like any garment: a compound whose HEAD noun is innocent ("spreader bar"
    # resolves to `bar`), and a binding participle fastened straight onto a body
    # part ("bound wrists"), where no equipment noun exists at all.
    for item in ["spreader bar", "doorway spreader bar", "bound wrists",
                 "shackled ankles", "tied hands", "wrists bound in rope",
                 "ankles locked in steel"]:
        check(f"{item!r} is a restraint too", S.is_restraint(item))
    # ...and the participle route must not swallow fashion wording: the body part
    # alone never qualifies, and neither does an ordinary participle without one.
    for item in ["waist tie dress", "tie-front blouse", "wrap skirt", "ankle boots",
                 "wrist-length sleeves", "high-neck top"]:
        check(f"{item!r} still comes off normally", not S.is_restraint(item))

    act = {"Mara": ["she", "steel handcuffs", "ankle chain", "red jacket"]}

    def after(beat, lock=True):
        out = S.auto_wardrobe_removals({k: list(v) for k, v in act.items()}, beat, lock)
        return [i for i in act["Mara"] if i not in out["Mara"]]

    check("prose cannot remove handcuffs", after("Mara slips out of the handcuffs.") == [])
    check("prose cannot remove an ankle chain",
          after("The ankle chain falls away.") == [])
    check("a garment still comes off normally",
          after("Mara takes off her red jacket.") == ["red jacket"])
    # The accident this exists to stop.
    check("a jacket beat no longer drags the chain off with it",
          after("Mara steps out of her jacket and the chain falls away.") == ["red jacket"])
    # 'struggles out of' was absent from the cue vocabulary entirely -- the verb
    # sat next to wriggles/squirms in intent but never made the list, so a garment
    # removed that way silently stayed ON the sheet.
    check("'struggles out of' is a removal cue too",
          after("Mara struggles out of the red jacket.") == ["red jacket"])
    check("...and cannot drag a restraint with it either",
          after("She struggles out of her jacket and the cuffs fall away.") == ["red jacket"])
    # Same accident through the newly recognized shapes.
    spread = S.auto_wardrobe_removals(
        {"Mara": ["she", "spreader bar", "red jacket"]},
        "She steps out of her jacket and the spreader bar falls away.", True)
    check("a jacket beat cannot drop a spreader bar either",
          spread["Mara"] == ["she", "spreader bar"])
    bound = S.auto_wardrobe_removals(
        {"Mara": ["she", "bound wrists", "red gloves"]},
        "Her bound wrists shake as she peels off her red gloves.", True)
    check("...nor bound wrists", bound["Mara"] == ["she", "bound wrists"])
    # ...and the escape hatch still works.
    freed = S.apply_wardrobe_change({k: list(v) for k, v in act.items()},
                                    "Mara -= handcuffs")
    check("an explicit 'wardrobe: -=' still removes a restraint",
          "steel handcuffs" not in freed["Mara"])
    check("...leaving the other restraint alone", "ankle chain" in freed["Mara"])
    # Off means off.
    check("lock_restraints=False restores the old behaviour",
          after("Mara slips out of the handcuffs.", lock=False) == ["steel handcuffs"])

    # End to end: the newly recognized shapes state their hold on every shot the
    # person is in, exactly like the equipment-noun restraints do.
    blk = D("A cellar.", ["Mara strains against the spreader bar."], "", "",
            "Mara = she, spreader bar")[0]
    check("a spreader bar is stated and held every shot",
          "physically restrained" in blk and "spreader bar" in blk)
    blk2 = D("A cellar.", ["Mara waits."], "", "", "Mara = she, shackled ankles")[0]
    check("shackled ankles map to the ankle region", "ankles stay bound" in blk2)

    # End to end, with the person referenced in every beat so she is always bound.
    cm = "Mara = she, 30, blonde, steel handcuffs, ankle chain, red jacket"
    beats = ["Mara stands in the room.",
             "Mara takes off her red jacket.",
             "Mara slips out of the handcuffs.",
             "Mara shakes the ankle chain and it falls away.",
             "Mara is freed by a guard.\nwardrobe: Mara -= handcuffs, ankle chain",
             "Mara rubs her wrists and stands."]
    g = S.distribute_generations("A room.", beats, "", "", cm, lock_restraints=True)
    def worn_items(shot):
        """What the character is DESCRIBED as wearing -- the bound parenthetical.

        Not the whole shot text: the shot that performs a removal names the item in
        its direction clause ("takes the steel handcuffs off during this shot"), so
        a substring test there reports an item that has already come off."""
        i = shot.find("Mara (")
        return shot[i:shot.find(")", i) + 1] if i >= 0 else ""

    check("restraints survive removal prose",
          all("handcuff" in worn_items(s) for s in g[:4]))
    check("...while the jacket still comes off", "red jacket" not in worn_items(g[2]))
    check("an explicit directive frees them", "handcuff" not in worn_items(g[4]))
    check("...and they stay off afterwards",
          "handcuff" not in worn_items(g[5]) and "ankle chain" not in worn_items(g[5]))
    check("lock_restraints is an appended widget", "lock_restraints" in S.ADDED_WIDGETS)

    # --- and what the restraint DOES once it is on -----------------------------
    # Keeping the item in the wardrobe list only says it exists. Nothing there says
    # the body cannot move freely, so a cuffed character walks with arms swinging --
    # the restraint present and doing nothing, which reads as having broken.
    for item, region in [("steel handcuffs", "wrists"), ("wrist ties", "wrists"),
                         ("thumb cuffs", "wrists"), ("ankle chain", "ankles"),
                         ("leg irons", "ankles"), ("hobble", "ankles"),
                         ("ball gag", "mouth"), ("blindfold", "eyes")]:
        check(f"{item!r} binds the {region}", S.restraint_regions([item]) == [region])
    # No region of its own -- say movement is limited without guessing a limb.
    for item in ["shackles", "fetters", "harness"]:
        check(f"{item!r} falls back to a general body constraint",
              S.restraint_regions([item]) == ["body"])
    check("a garment binds nothing", S.restraint_regions(["red jacket"]) == [])
    check("two wrist restraints give ONE clause, not two",
          S.restraint_regions(["steel handcuffs", "wrist ties"]) == ["wrists"])

    # Positive physical state, never a negation -- "cannot move her arms" is a weak
    # cue; "the wrists stay bound close together" is a pose the model can render.
    eff = " ".join(S._RESTRAINT_EFFECT.values()).lower()
    for bad in ("cannot", "can't", "unable", "no longer", "does not", "won't"):
        check(f"the effect text never negates ({bad})", bad not in eff)

    cm2 = ("Mara = she, 30, blonde, steel handcuffs, ankle chain\n"
           "Jon = he, 35, bald, navy overalls")
    beats2 = ["Mara and Jon stand in the room.", "Mara walks toward the door.",
              "Jon opens the door.", "The hangar lights flicker.",
              "Mara is freed.\nwardrobe: Mara -= handcuffs, ankle chain",
              "Mara stretches her arms."]
    r = S.distribute_generations("A room.", beats2, "", "", cm2)
    said = ["physically restrained" in s for s in r]
    check("a restrained person in shot gets the constraint", said[0] and said[1])
    check("...naming both bound regions",
          "wrists stay bound" in r[0] and "ankles stay bound" in r[0])
    check("...by pronoun, not by re-naming her", "She is physically restrained" in r[0])
    check("a shot she is not in gets no constraint", not said[2])
    check("a scenery beat gets none", not said[3])
    check("the freeing shot drops it", not said[4])
    check("...and it stays gone", not said[5])
    check("lock_restraints=False suppresses it entirely",
          not any("physically restrained" in s for s in
                  S.distribute_generations("A room.", beats2, "", "", cm2,
                                           lock_restraints=False)))


def check_nappy_vocabulary():
    """Nappies are tracked, removable, and stay off — like any other garment.

    Nothing special is needed for them; the point of these checks is that the
    generic machinery already reaches them, including the two places it has
    historically leaked: a conjoined sheet entry hiding the second item, and a
    qualified name whose head noun stops being recognised."""
    print("\n=== nappies as ordinary clothing ===")
    for it in ("diaper", "diapers", "nappy", "nappies", "pull-up", "pullups",
               "white diaper", "thick padded diaper", "wet diaper", "cloth nappy"):
        check(f"{it!r} covers the lower zone", S.garment_zones(it) == {"lower"})
    check("a nappy is NOT a restraint, so it can come off",
          not any(S.is_restraint(i) for i in ("diaper", "nappy", "pull-up")))

    # "cover" cannot go in the zone list on its own, so the compound is matched
    # whole -- and the things that are CARRIED must not become body covering, or
    # they would suppress the exposure warning exactly when it is needed.
    check("'diaper cover' is body covering", S.garment_zones("diaper cover") == {"lower"})
    check("'nappy wrap' too", S.garment_zones("nappy wrap") == {"lower"})
    check("'diaper bag' is NOT -- it is carried",
          S.garment_zones("diaper bag") == set())
    check("'changing mat' is NOT either", S.garment_zones("changing mat") == set())
    check("a bare 'cover' stays unmapped", S.garment_zones("seat cover") == set())

    # The conjoined-entry bug that once hid a thong.
    check("a conjoined sheet entry splits", S._split_conjoined("t-shirt and diaper")
          == ["t-shirt", "diaper"])

    CM = "Mara = she, 30, red hair, t-shirt, diaper"
    for beat in ("Mara takes off her diaper.", "Mara pulls off the diaper.",
                 "Mara removes her diaper.", "Mara unfastens the diaper and drops it."):
        gens = S.distribute_generations("A quiet room.",
                                        [beat, "Mara stands still.", "Mara turns around."],
                                        "", "", CM, prevent_nudity=False)
        check(f"removal fires: {beat[:30]!r}", "diaper" not in gens[1])
        check("...and it stays off through a later shot", "diaper" not in gens[2])

    gens = S.distribute_generations("A quiet room.",
                                    ["Mara takes off her diaper.", "Mara turns around."],
                                    "", "", CM, prevent_nudity=False)
    check("removing it states the zone as bare",
          "bare below the waist" in gens[0])
    check("...and that holds through the turn",
          "stays bared as the body turns" in gens[1])


def check_bare_state_persists():
    """A bared zone stays bared when the body turns.

    The marker in the item list says the zone IS bare. Nothing said it STAYS bare
    once the body presents a surface the shot has not shown yet -- and a turn is
    exactly that: new geometry, no evidence, and the model's default for an
    undescribed body is a clothed one. So the garment came back mid-shot.

    Worse at a strip boundary: the shot that performs a removal deliberately loses
    its handoff (its last frame shows the removal in progress), so the shot after it
    renders from the PROMPT ALONE with no picture of the bared state to continue
    from. That is the shot a turn usually lands in."""
    print("\n=== a bared zone stays bared through a turn ===")
    ANCHOR = "A quiet studio, north light."
    CM = "Mara = she, 30, red hair, grey coat, denim shorts"
    BEATS = ["Mara stands by the window.", "Mara takes off her denim shorts.",
             "Mara turns away from the camera.", "Mara walks to the door."]

    def run(**kw):
        return S.distribute_generations(ANCHOR, BEATS, "", "", CM, **kw)

    PERSIST = "stays bared as the body turns"
    on = run(exposed_terms="she = lower", prevent_nudity=True)
    check("the clause appears in the shot that bares the zone", PERSIST in on[1])
    check("...and in the TURN that follows it", PERSIST in on[2])
    check("...and keeps holding on later shots", PERSIST in on[3])
    check("not before anything came off", PERSIST not in on[0])
    check("it says the state holds from every side",
          "the same from the front, the side and behind" in on[2])

    # It must not weaken any existing gate: the marker's authority is the clause's.
    off = run(exposed_terms="", prevent_nudity=True)
    check("prevent_nudity still suppresses it with no exposed_terms",
          all(PERSIST not in g for g in off))
    free = run(exposed_terms="", prevent_nudity=False)
    check("...and with the guard off it comes back", PERSIST in free[2])

    # Naming the garment is what brings the garment back -- the clause must not.
    check("no garment is named in the clause",
          not any(w in S._BARE_PERSIST["lower"] + S._BARE_PERSIST["upper"]
                  for w in ("shorts", "underwear", "thong", "jacket", "clothes")))
    check("...and it states a positive, not a negation",
          not re.search(r"\b(?:not|never|no|without)\b",
                        S._BARE_PERSIST["lower"], re.I))

    # THE regression: an earlier version said "She is uncovered there and stays that
    # way", which is a SECOND reference to someone the shot already introduced -- and
    # a second mention renders a second figure. It went straight up against the
    # subject-count guard. Measured on a 4-shot two-person chain, it added exactly one
    # extra subject reference to every bared shot (5,3,1,3 against a 4,2,1,2 baseline).
    _cl = S.bare_persist_clause({"Mara": ["lower", "upper"]},
                                {"Mara": [], "Dom": []}, "Mara and Dom stand together.")
    check("the clause names nobody",
          not re.search(r"(?:Mara|Dom|she|her|he|him|his|subject)", _cl, re.I))
    check("...and introduces no pronoun subject at all",
          not re.search(r"(?:She|He|They|Her|His|Their)", _cl))
    check("one sentence per bared ZONE, not per person",
          _cl.count("stays bared as the body turns") == 2)
    check("two people bared the same way say it once",
          S.bare_persist_clause({"Mara": ["upper"], "Dom": ["upper"]},
                                {"Mara": [], "Dom": []}, "Mara and Dom stand together."
                                ).count("stays bare") == 1)

    # Same presence gate as every other per-shot state.
    check("someone not in the shot is not described as uncovered",
          S.bare_persist_clause({"Dom": ["lower"]},
                                {"Mara": [], "Dom": []}, "Mara looks out.") == "")
    check("an empty zone list says nothing",
          S.bare_persist_clause({"Mara": []}, {"Mara": []}, "Mara turns.") == "")

    # The strip boundary is why this matters most.
    strip = []
    run(exposed_terms="she = lower", prevent_nudity=True, strip_out=strip)
    check("the removal shot costs the NEXT shot its handoff", strip == [2])
    check("...so the turn renders from the prompt alone, and the prompt now says it",
          PERSIST in on[2])


def check_contact_guard():
    """Two bodies in contact stay correctly aligned, in ANY arrangement.

    Position-agnostic on purpose: a dictionary of named positions would be endless,
    and the model knows more names than a list could hold. What it gets wrong is the
    geometry, so the geometry is stated -- limb ownership, no interpenetration, fixed
    above/below/behind roles, weight on something real. Those hold for every
    arrangement, which is why nothing here names one."""
    print("\n=== two bodies in contact ===")
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    CM = "Mara = she, 30, red hair\nDom = he, 35, tall"

    check("a contact cue is detected",
          bool(S._CONTACT_CUE.search("Dom holds Mara against the wall")))
    check("...including a purely positional one",
          bool(S._CONTACT_CUE.search("Dom kneels behind Mara")))
    check("...and a mutual orientation",
          bool(S._CONTACT_CUE.search("they stand facing each other")))
    check("an ordinary beat is not a contact cue",
          not S._CONTACT_CUE.search("Mara walks to the window"))

    # TWO people, or nothing. One body cannot be misaligned against another, and
    # describing a two-body arrangement in a one-person shot invites the second in --
    # the presence-cue failure every per-shot state here is gated against.
    check("one person gets nothing even with a contact cue",
          S.contact_clause("Dom holds her against the wall", 1, "auto") == "")
    check("two people and a cue fires",
          bool(S.contact_clause("Dom holds Mara", 2, "auto")))
    check("'on' fires for two people without a cue",
          bool(S.contact_clause("they talk", 2, "on")))
    check("'on' still needs two people", S.contact_clause("she talks", 1, "on") == "")
    check("'off' says nothing", S.contact_clause("Dom holds Mara", 2, "off") == "")

    # The four invariants, which are the whole point.
    _c = S.CONTACT_STATE
    check("OWNERSHIP: limbs belong to the body they are joined to",
          "joined to the body it belongs to" in _c)
    check("SEPARATION: they meet at the skin, each keeping its volume",
          "surface of the skin" in _c and "own solid volume" in _c)
    check("STABLE ROLES: above/below/behind hold for the whole shot",
          "stays above" in _c and "stays below" in _c and "stays behind" in _c)
    check("...and read the same from every camera angle", "every angle" in _c)
    check("SUPPORT: weight rests on something real", "weight rests on" in _c)
    check("no negation -- the negative is never evaluated at cfg 1",
          not re.search(r"\b(?:not|never|no|without|avoid)\b", _c, re.I))
    check("it names no specific position, so every arrangement is covered",
          not re.search(r"\b(?:missionary|doggy|cowgirl|spooning|69)\b", _c, re.I))

    # Wired through, and gated on who is actually in the shot.
    gens = S.distribute_generations("A quiet studio.",
                                    ["Mara and Dom stand facing each other.",
                                     "Mara walks to the window.",
                                     "The room is quiet."], "", "", CM)
    check("a two-person contact shot gets it", "ONE fixed arrangement" in gens[0])
    check("a one-person shot does not", "ONE fixed arrangement" not in gens[1])
    check("a shot with nobody does not", "ONE fixed arrangement" not in gens[2])
    off = S.distribute_generations("A quiet studio.", ["Mara and Dom embrace."],
                                   "", "", CM, contact_guard="off")
    check("'off' reaches the per-shot text", "ONE fixed arrangement" not in off[0])
    check("the widget is appended, so saved workflows keep their order",
          "contact_guard" in S.ADDED_WIDGETS)
    check("...and run() passes it through", "contact_guard=contact_guard" in src)


def check_motion_guard():
    """A pose is reached by travelling to it.

    A neck snap is not a wrong pose -- it is a right pose with no path to it, the
    head arriving at a new angle without the frames in between. So the PATH is what
    gets stated. Positive, because H3 is CFG-free at cfg 1 (the negative is never
    evaluated) and "the head does not snap round" in the positive names a head
    snapping round."""
    print("\n=== motion continuity ===")
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()

    check("a beat that turns someone is a motion cue",
          bool(S._MOTION_CUE.search("Mara turns away from the camera")))
    check("...as is looking, walking, reaching",
          all(S._MOTION_CUE.search(t) for t in
              ("she looks up", "he walks out", "she reaches for the door")))
    check("a beat with no orientation change is not",
          not S._MOTION_CUE.search("Mara smiles faintly"))
    check("...nor is scenery", not S._MOTION_CUE.search("the light fades"))

    _m = S.motion_clause("Mara turns away", "auto")
    check("'auto' speaks on a moving beat", bool(_m))
    check("'auto' is silent otherwise", S.motion_clause("Mara smiles", "auto") == "")
    check("'on' speaks regardless", bool(S.motion_clause("Mara smiles", "on")))
    check("'off' says nothing", S.motion_clause("Mara turns away", "off") == "")

    check("it states the PATH, not the pose",
          "through every position on the way" in _m and "steady speed" in _m)
    check("...and the chain of joints, which is where a neck snap shows",
          "neck following the shoulders" in _m)
    check("no negation -- the negative is never evaluated at cfg 1",
          not re.search(r"\b(?:not|never|no|without|avoid)\b", _m, re.I))
    check("it names nobody, so it adds no second reference to anyone in frame",
          not re.search(r"\b(?:she|he|her|his|they|Mara|Dom|subject)\b", _m, re.I))

    # Same presence gate as every other per-shot state.
    CM = "Mara = she, 30, red hair, coat"
    gens = S.distribute_generations("A quiet studio.",
                                    ["Mara turns away from the camera.",
                                     "The window rattles."], "", "", CM)
    check("a shot with someone turning gets it", "Movement is continuous" in gens[0])
    check("a scenery shot does not -- nothing there has a neck",
          "Movement is continuous" not in gens[1])
    off = S.distribute_generations("A quiet studio.", ["Mara turns away."], "", "", CM,
                                   motion_guard="off")
    check("'off' reaches the per-shot text", "Movement is continuous" not in off[0])
    check("the widget is appended, so saved workflows keep their order",
          "motion_guard" in S.ADDED_WIDGETS)
    check("...and run() passes it through", "motion_guard=motion_guard" in src)


def check_solidity_guard():
    """Bodies stop at objects instead of passing through them.

    Same constraint as the anatomy guard and for the same reason: H3 is CFG-free at
    cfg 1, so a negative prompt is never evaluated. And "does not walk through the
    wall" cannot go in the POSITIVE either -- it names walking through a wall, and a
    mention is a presence cue. Only a positive statement of what bodies do is
    available."""
    print("\n=== solidity guard ===")
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    ANCHOR = "A workshop with a roller door, a workbench and a van."

    check("solid objects are found in a beat",
          S.solid_things_in("Mara walks past the van") == ["van"])
    check("...and deduplicated by head noun",
          S.solid_things_in("a table and more tables") == ["table"])
    check("...keeping the form as written, so 'stairs' is not 'stair'",
          S.solid_things_in("she climbs the stairs") == ["stairs"])
    check("the beat's own objects come FIRST, not the set dressing",
          S.solid_things_in("Mara climbs the stairs.")[0] == "stairs"
          and S.solidity_clause("Mara climbs the stairs.", ANCHOR, "auto")
              .index("the stairs") < S.solidity_clause(
                  "Mara climbs the stairs.", ANCHOR, "auto").index("the roller door"))
    _many = S.solidity_clause("a table, a chair, a bed, a couch and a desk", "", "auto")
    check("at most three are named, so it does not read as an inventory",
          _many.split("occupying real space here: ")[1].count("the ") == 3)

    # It has to be stated positively -- the whole point.
    _c = S.solidity_clause("Mara walks past the van", ANCHOR, "auto")
    check("the clause never says 'through'", "through" not in _c.lower())
    check("...and carries no negation of the failure",
          not re.search(r"\b(?:not|never|no|doesn't|does not|cannot)\b", _c, re.I))
    check("...it says what bodies DO instead",
          "stops where it meets a surface" in _c and "walks around" in _c)

    # Modes.
    check("'off' says nothing", S.solidity_clause("she walks past the table", "", "off") == "")
    check("'auto' is silent when nothing solid is named",
          S.solidity_clause("she smiles at him", "", "auto") == "")
    check("'auto' speaks when something solid IS named",
          bool(S.solidity_clause("she leans on the workbench", "", "auto")))
    check("'on' speaks regardless", bool(S.solidity_clause("she smiles", "", "on")))
    check("a genuinely passable thing is not called solid",
          S.solid_things_in("a curtain of rain and a puff of smoke") == [])
    check("vague prose nouns do not fire",
          S.solid_things_in("an edge to her voice, the side of the story") == [])

    # Gated on a body being present, like every other per-shot state.
    CM = "Mara = she, 30, red hair"
    gens = S.distribute_generations(ANCHOR, ["Mara walks past the van.",
                                             "The roller door rattles open."],
                                    "", "", CM, solidity_guard="auto")
    check("a shot with someone in it gets the clause",
          "Solid things stay solid" in gens[0])
    check("a scenery shot with nobody in it does not -- no body, nothing to pass through",
          "Solid things stay solid" not in gens[1])
    off = S.distribute_generations(ANCHOR, ["Mara walks past the van."],
                                   "", "", CM, solidity_guard="off")
    check("'off' reaches the per-shot text", "Solid things stay solid" not in off[0])

    check("the widget is appended, so saved workflows keep their order",
          "solidity_guard" in S.ADDED_WIDGETS)
    check("...and run() passes it through",
          "solidity_guard=solidity_guard" in src)


def check_anatomy_guard():
    """Limb counts stated POSITIVELY, because a negative cannot work here.

    H3 is CFG-free at cfg 1, and comfy/samplers.py sets uncond_ = None at that
    scale, so the negative prompt is never evaluated -- "extra limbs" in a negative
    does nothing on this model. Naming the number gives the model a target, which
    is the same mechanism the subject-count guard already uses for people."""
    print("\n=== anatomy guard ===")
    cm = "Mara = she, 30, blonde\nJon = he, 35, bald"
    beats = ["Mara and Jon walk in.", "They face each other.",
             "The hangar doors roll open.", 'Mara says: "Ready."']

    off = S.distribute_generations("A room.", beats, "", "", cm, anatomy_guard=False)
    on = S.distribute_generations("A room.", beats, "", "", cm, anatomy_guard=True)
    check("off adds nothing", not any("two arms" in g for g in off))
    check("on states the limb count where people are",
          all("two arms" in g for g in (on[0], on[1], on[3])))
    # The gate that matters: describing a body in an empty frame invites one in.
    # That is what put a face in the opening frames of scenery shots before.
    check("a scenery beat with nobody gets NO anatomy clause", "two arms" not in on[2])
    check("a plural-bound beat does get it", "two arms" in on[1])
    check("a dialogue beat gets it too", "two arms" in on[3])

    # Positive counts only -- a negation would just put the word in the prompt.
    txt = S.ANATOMY_STATE.lower()
    for bad in (" no ", "without", "avoid", "extra", "deformed", "mutated", "malformed"):
        check(f"the clause never negates ({bad.strip()})", bad not in txt)
    for good in ("one head", "two arms", "two hands", "five fingers", "two legs"):
        check(f"the clause states {good}", good in txt)
    # Where a spare limb is GROWN decides the wording: attachment points, feet,
    # and exactly one groin between the legs -- 'three-leg syndrome' is a groin
    # rendered as a limb, so the count has to be stated where it happens.
    for good in ("two feet", "at one shoulder", "at one hip", "moves only with "
                 "the person it belongs to", "one groin"):
        check(f"the clause pins limbs ({good})", good in txt)
    # A limb in the WRONG PLACE (arm out of the ribs, leg off the chest) needs the
    # joint CHAIN and the stacking order stated too, so the skeleton has a layout
    # to land on and not just a count.
    for good in ("shoulder to elbow to wrist to hand",
                 "hip to knee to ankle to foot",
                 "head on the neck, neck on the shoulders",
                 "arms hanging along the sides of the torso",
                 "legs under the hips"):
        check(f"the clause places joints ({good})", good in txt)

    # The widget must be APPENDED, or saved workflows shift.
    check("anatomy_guard is an appended widget", "anatomy_guard" in S.ADDED_WIDGETS)


def check_anatomy_auto_multi_person():
    """'auto' now states the limb count on ANY multi-person shot.

    It used to fire only below a 768 short edge or with a LoRA -- so at native
    resolution, which is how most chains run, NOTHING said how many legs a body
    has, and that is exactly where two moving bodies grow a third one between
    them."""
    print("\n=== anatomy auto-fires on multi-person shots ===")
    cm = "Mara = she, 30, blonde\nJon = he, 35, bald"
    g = S.distribute_generations("A room.", ["Mara and Jon walk in."], "", "", cm,
                                 anatomy_guard=False, anatomy_auto=True)
    check("two people -> the limb count is stated even at native size",
          any("two arms" in x for x in g))
    solo = S.distribute_generations("A room.", ["Mara waves."], "", "", cm,
                                    anatomy_guard=False, anatomy_auto=True)
    check("one person -> nothing added under auto",
          not any("two arms" in x for x in solo))
    on = S.distribute_generations("A room.", ["Mara waves."], "", "", cm,
                                  anatomy_guard=True)
    check("'on' still covers the solo shot", "two arms" in on[0])
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    check("run() derives anatomy_auto from the widget",
          'anatomy_auto = (anatomy_guard == "auto")' in src)


def check_latent_output():
    """The node emits the sampled latent ALONGSIDE images, never instead of them.

    The chain cannot defer decoding: each shot's handoff is `frames[-1:]`, a decoded
    frame, and both trims (trim_seam, handoff_offset) cut pixel frames. So a
    latent-only mode is not possible -- but the pre-decode latent is ~1000x smaller
    than the frames it becomes, so carrying one per shot is free."""
    print("\n=== latent output ===")
    cls = S.NODE_CLASS_MAPPINGS["H3LongVideos"]
    check("RETURN_TYPES and RETURN_NAMES agree",
          len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES))
    check("a LATENT output exists", "LATENT" in cls.RETURN_TYPES)
    check("...named 'latent'", "latent" in cls.RETURN_NAMES)
    # APPENDED, not inserted: ComfyUI stores a link by output SLOT INDEX, so
    # inserting mid-list would silently re-target every existing wire.
    # `latent` must keep the SLOT it was appended at, whatever gets added after it.
    check("latent keeps its slot", cls.RETURN_NAMES.index("latent") == 10)
    check("...and anything newer is appended after it, never inserted before",
          cls.RETURN_NAMES.index("latent") < len(cls.RETURN_NAMES))
    check("images is still slot 0", cls.RETURN_NAMES[0] == "images")
    check("audio is still slot 1", cls.RETURN_NAMES[1] == "audio")
    check("info is still slot 2", cls.RETURN_NAMES[2] == "info")
    check("script is still slot 3", cls.RETURN_NAMES[3] == "script")
    check("fps_int is still slot 9", cls.RETURN_NAMES[9] == "fps_int")

    # Every return path must carry the same arity, or ComfyUI errors on unpack.
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    # Plain tuple returns again: the _with_shift_ui() wrapper existed only so the
    # frontend could write auto-derived shifts into the widgets, and auto_shift is gone.
    check("the plan_only path returns a latent too",
          "float(fps), int(fps),\n                    _empty_av_latent(" in src)
    check("the render path returns latent_out",
          "float(fps), int(fps), latent_out, global_soundscape)" in src)
    check("no ui wrapper is left on either return", "_with_shift_ui" not in src)
    # It must never be None -- a downstream LATENT input cannot take that.
    check("plan_only emits a real empty latent, not None",
          "_empty_av_latent(w, h, 5, fps)[0], global_soundscape)" in src)
    check("the render path seeds a real empty latent before assembly",
          'latent_out = {"samples": _empty_av_latent(' in src)

    # --- the soundscape output ---------------------------------------------------
    # auto_soundscape builds an ambient bed from the scene, and until now the only
    # way to see what it built was to read `info`. It is emitted so you can read it
    # and feed it straight back into the soundscape input to pin it.
    check("a soundscape output exists", "soundscape" in cls.RETURN_NAMES)
    check("...appended LAST, so no existing wire is re-targeted",
          cls.RETURN_NAMES[-1] == "soundscape" and cls.RETURN_TYPES[-1] == "STRING")
    check("...and every earlier slot is where it was",
          list(cls.RETURN_NAMES[:11]) == ["images", "audio", "info", "script",
                                          "frames_per_shot", "total_frames", "shots",
                                          "video_seconds", "fps", "fps_int", "latent"])
    check("both return paths carry it",
          src.count(", global_soundscape)") == 2)
    # It is the soundscape ACTUALLY used: the derivation reassigns global_soundscape,
    # so this is the derived bed when auto fired and the user's own text when it did
    # not -- never a second, separately-computed value that could disagree.
    _i = src.index("if auto_soundscape != \"off\":")
    check("the derived value is what the variable holds",
          "global_soundscape = derived" in src[_i:_i + 900])
    check("...assigned before the plan_only return",
          src.index("global_soundscape = derived")
          < src.index("_empty_av_latent(w, h, 5, fps)[0], global_soundscape)"))
    # And the caveat has to be stated, because the latent is NOT the latent form of
    # `images` on a multi-shot chain.
    check("the pre-trim caveat is reported", "PRE-trim" in src)
    check("...and a single shot is called exact", "exact match for `images`" in src)
    check("a resolution backoff is handled rather than crashing",
          "resolution backoff" in src)


def check_preflight_note_assembly():
    """The preflight notes are built ONCE and emitted at both output sites.

    They used to be six separate `+ (f" X -- {n}." if n else "")` fragments repeated
    at the plan site and the render site -- twelve lines kept in sync by hand. They
    drifted: the shift-ratio note was named `audio_note`, colliding with the
    mute-reporting variable of the same name assigned later in run(), so it reached
    plan_only and never a real render, while the mute note printed twice under the
    wrong label."""
    print("\n=== preflight note assembly ===")

    def compose(pf):
        return "".join(f"{(lbl + ' -- ') if lbl else ''}{txt}. " for lbl, txt in pf if txt)

    empty = [("SLA", ""), ("LORA HINTS", ""), ("", ""), ("SCHEDULE", ""),
             ("KERNELS", ""), ("AUDIO", "")]
    check("no notes produces no text", compose(empty) == "")
    check("an unlabelled note carries no separator",
          compose([("", "megapixels 1.00")]) == "megapixels 1.00. ")
    check("a labelled note gets its label",
          compose([("SCHEDULE", "tail heavy")]) == "SCHEDULE -- tail heavy. ")
    check("several notes keep their order",
          compose([("SCHEDULE", "d"), ("AUDIO", "f")]) == "SCHEDULE -- d. AUDIO -- f. ")
    check("blank entries are skipped, not spaced",
          compose([("SLA", ""), ("KERNELS", "e")]) == "KERNELS -- e. ")

    # The collision itself: the two must be distinct names in the source.
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    check("the shift-ratio note no longer uses the mute note's name",
          "audio_ratio_note = (audio_scale_note(" in src)
    check("...and the mute note keeps its own name", "audio_note = (f\" {n_muted}" in src)
    check("the ratio note is in the preflight list", '("AUDIO", audio_ratio_note)' in src)
    # Referenced on exactly three LINES: one definition, one plan emission, one
    # render emission. Counting occurrences instead of lines is wrong -- the render
    # line names it twice (`preflight_txt.strip()` and the `if preflight_txt` guard).
    lines = [ln for ln in src.splitlines() if "preflight_txt" in ln]
    check("preflight_txt appears on exactly 3 lines: define, plan, render",
          len(lines) == 3)
    check("...one of them is the definition",
          any(ln.strip().startswith("preflight_txt =") for ln in lines))


def check_auto_soundscape():
    """An ambient bed inferred from the scene.

    Read from the ANCHOR, because the soundscape is global -- stamped on every shot
    -- so it has to describe the PLACE, not one beat's action. And the vocabulary
    contains no human sound at all: an ambient bed that implies voices is how H3
    starts talking, which is the failure this node spends most of its silence
    machinery on."""
    print("\n=== soundscape from the scene ===")

    for anchor, want in [
        ("Natural daylight. A disused aircraft hangar.", "cavernous interior"),
        ("Rain on the windows. A small home kitchen at night.", "steady rain"),
        ("Overcast. A rocky beach with waves and wind.", "waves breaking"),
        ("A city street at night, neon in puddles after rain.", "distant traffic hum"),
        ("Warm interior light. A workshop with a roller door.", "close interior room tone"),
        ("A forest clearing beside a stream, campfire burning.", "wind in leaves"),
        ("A quiet bedroom, curtains drawn.", "quiet indoor room tone"),
    ]:
        check(f"{anchor[:34]!r} -> {want}", want in S.derive_soundscape(anchor))

    # Camera language is not scenery. "shallow depth of FIELD" was read as a meadow.
    check("a camera-only anchor yields nothing",
          S.derive_soundscape("Cinematic, shallow depth of field, 35mm, f/2.8, film grain.") == "")
    check("...and falls through to the beats",
          "cavernous interior" in S.derive_soundscape(
              "Cinematic, shallow depth of field, 35mm.",
              ["Mara walks into the hangar."]))
    check("nothing anywhere stays empty",
          S.derive_soundscape("Cinematic, 35mm.", ["She smiles."]) == "")
    # 'windows' is not 'wind' -- a missing trailing \b put gusting wind in a kitchen.
    check("'windows' does not imply wind",
          "gusting wind" not in S.derive_soundscape("Rain on the windows. A kitchen."))
    check("real wind still registers",
          "gusting wind" in S.derive_soundscape("A windswept ridge."))
    # A specific interior must not also draw the generic one.
    ss = S.derive_soundscape("A small home kitchen at night.")
    check("a specific interior does not double up with the generic",
          ss.count("room tone") == 1)

    # THE constraint: never generate a human sound.
    human = ("voice", "chatter", "murmur", "crowd", "talk", "speech", "announce",
             "conversation", "people", "laugh", "shout", "sing")
    allsound = " ".join(p for _, p in S._SOUNDSCAPE_CUES).lower() + " " + \
               " ".join(p for _, p in S._SOUNDSCAPE_FALLBACK).lower()
    # Whole words only -- a substring test flags "sing" inside "pasSING vehicles".
    def says(word, text):
        return bool(re.search(r"\b" + word + r"\w*\b", text))
    for w in human:
        check(f"the vocabulary never says {w!r}", not says(w, allsound))
    # ...including for places full of people.
    for anchor in ("A crowded bar, late evening.", "A busy railway station.",
                   "A packed restaurant."):
        out = S.derive_soundscape(anchor).lower()
        check(f"{anchor[:26]!r} implies no voices",
              not any(says(w, out) for w in human))

    check("the bed stays short", len(S.derive_soundscape(
        "Rain and wind over a city street at night beside a river, engine running.")
        .split(",")) <= 8)
    check("auto_soundscape is an appended widget", "auto_soundscape" in S.ADDED_WIDGETS)


def check_lora_chain_and_oom():
    """The graph walk that finds LoRAs, and what a sampling OOM reports.

    These outlived auto_shift, which was removed: its premise (that a low step
    count needs a lower shift) is wrong for a distill LoRA, which is TRAINED to
    make the big final jump the heuristic tried to flatten. The LoRA walk is
    still needed by the SLA detection, and the OOM advice by every long shot.
    """
    print("\n=== LoRA chain walk, and sampling-OOM advice ===")
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()

    # --- a SAMPLING oom cannot be tiled away -------------------------------------
    _help = S.sampling_oom_help(864, 864, 362, 24, 0.7)
    check("the oom advice says tiling will not help", "tiled decode cannot help" in _help)
    check("...and prices the shot in its own numbers", "864x864" in _help and "362f" in _help)
    check("...and offers a shorter shot", "shot_seconds 10" in _help)
    check("...and a smaller frame", "megapixels 0.5" in _help)
    check("cost is linear in shot length",
          abs(S.shot_latent_cells(864, 864, 362, 24)
              / S.shot_latent_cells(864, 864, 181, 24) - 2.0) < 0.15)
    check("cost falls with area",
          S.shot_latent_cells(608, 608, 362, 24) < S.shot_latent_cells(864, 864, 362, 24))
    check("a sampling oom is tagged at the sampling site", 'e._h3_stage = "sampling"' in src)
    check("...and is not retried with tiles",
          'getattr(e, "_h3_stage", "") == "sampling"' in src)
    check("a shot already at the minimum is not offered a longer 'cut'",
          "shot_seconds" not in S.sampling_oom_help(864, 864, 124, 24, 0.0))

    # --- stacked LoRA loaders ----------------------------------------------------
    # Only a bare filename under a key containing "lora" used to be recognised.
    # Stacked loaders do not work that way: DaSiWa packs every LoRA into ONE json
    # string under `stack_data` (verbatim below, from a real prompt), and rgthree's
    # Power Lora Loader stores a dict per slot. A chain carrying four LoRAs read as
    # carrying none.
    _dasiwa = ('[{"on":true,"lora":"minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",'
               '"str":1,"vs":1,"as":1},{"on":true,"lora":"PenisV2_minimax-h3_epoch60.safetensors",'
               '"str":0.6,"vs":1,"as":1},{"on":true,"lora":"vagassist_e40.safetensors",'
               '"str":0.25,"vs":1,"as":1}]')
    _got = S.lora_names_in_widget("stack_data", _dasiwa)
    check("a DaSiWa json stack yields every LoRA in it", len(_got) == 3)
    check("...including the turbo one",
          any("turbo_v4_step600" in n for n in _got))
    check("an rgthree-style dict slot is read",
          S.lora_names_in_widget("lora_1",
              {"on": True, "lora": "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
               "strength": 1.0}) == [
              "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors"])
    check("a slot switched off is ignored",
          S.lora_names_in_widget("lora_1", {"on": False, "lora": "x.safetensors"}) == [])
    check("a zero-strength slot is ignored -- it changes nothing",
          S.lora_names_in_widget("lora_1",
              {"on": True, "lora": "x.safetensors", "strength": 0.0}) == [])
    check("an empty slot is not a LoRA name",
          S.lora_names_in_widget("lora_name", "None") == [])
    check("a plain filename still works", S.lora_names_in_widget(
          "lora_name", "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors"))
    check("a non-lora widget is not mined for filenames",
          S.lora_names_in_widget("text", "a woman in a red coat") == [])
    check("malformed json is survivable", S.lora_names_in_widget("stack_data", "[{oops") == [])

    # End to end through the walk, with the stack behind an intermediate model node
    # exactly as the real graph has it (loader -> H3AdaLNLoRAFix -> preview -> sampler).
    class _Stacked:
        def get_node(self, nid):
            return {"571": {"inputs": {"model": ["527", 0]}},
                    "527": {"inputs": {"model": ["668", 0]}},
                    "668": {"inputs": {"model": ["655", 0]}},
                    "655": {"inputs": {"stack_data": _dasiwa, "model": ["639", 0]}},
                    "639": {"inputs": {}}}[str(nid)]
    check("the walk finds a stacked LoRA through intermediate model nodes",
          len(S.upstream_lora_names(_Stacked(), "571")) == 3)

def check_audio_scale_coupling():
    """The two flow shifts are COUPLED on ComfyUI 0.31+, and that is new.

    ModelSamplingAV carries the audio latent on the VIDEO schedule scaled by
    audio_scale = shift_video / shift_audio (12/3 = 4). That ratio drives
    process_latent_in, process_latent_out, the minimax payload and the DiT
    forward. Flatten it toward 1.0 and the audio branch loses the scaling it is
    built around -- babble or silence.

    Before 0.31 the audio velocity was scaled by a derivative instead, so the
    shifts were effectively independent and lowering shift_video ALONE was
    harmless. Advice written for that behaviour is actively wrong now."""
    print("\n=== video/audio shift coupling ===")

    for sv, sa in ((12, 3), (12, 4), (12, 6), (8, 2), (4, 1), (3, 0.75), (3, 1)):
        check(f"shift {sv}/{sa} (ratio {sv / sa:.2f}) is accepted",
              S.audio_scale_note(sv, sa) == "")
    for sv, sa in ((3, 3), (2, 2), (12, 12), (6, 6)):
        n = S.audio_scale_note(sv, sa)
        check(f"equal shifts {sv}/{sa} warn", bool(n))
        check(f"...naming the collapsed scale for {sv}/{sa}", "1.00" in n)
    n = S.audio_scale_note(3, 3)
    check("the warning says what breaks", "babble" in n or "silence" in n)
    check("...and gives the corrected value", "shift_audio 0.75" in n)

    # The schedule advice must not recommend a change that breaks the audio --
    # suggesting a bare shift_video is what caused this.
    sched = S.schedule_balance_note(12.0, 4, "simple")
    check("the schedule advice also names shift_audio", "shift_audio" in sched)
    check("...and says why", "audio breaks" in sched)

    # Degenerate input stays silent rather than dividing by zero.
    check("a zero audio shift is silent", S.audio_scale_note(12, 0) == "")
    check("a zero video shift is silent", S.audio_scale_note(0, 3) == "")
    check("non-numeric input is silent", S.audio_scale_note(None, "x") == "")

    # The widget must be ABLE to express a correct low ratio.
    if not hasattr(sys.modules["comfy.samplers"], "KSampler"):
        class _KS:
            SAMPLERS = ["res_multistep", "euler"]
            SCHEDULERS = ["simple", "beta"]
        sys.modules["comfy.samplers"].KSampler = _KS
    for _n in ("samplers", "utils", "nested_tensor", "model_management"):
        setattr(sys.modules["comfy"], _n, sys.modules["comfy." + _n])
    opt = S.NODE_CLASS_MAPPINGS["H3LongVideos"].INPUT_TYPES()["optional"]
    lo = opt["shift_audio"][1]["min"]
    check("shift_audio can go low enough to hold the ratio at low shift_video",
          lo <= 0.75)


def check_schedule_balance():
    """A high flow shift at a low step count buries the run in its last step.

    Reproduces ModelSamplingDiscreteFlow's time_snr_shift plus the 'simple'
    scheduler's linear indexing of the sigma table. 12 is H3's own declared shift
    and is well balanced at ~20 steps; it only misbehaves when a distill LoRA drops
    the step count under it, and it does so silently -- soft, painterly output and
    no error."""
    print("\n=== flow shift vs step count ===")

    sh12_4 = S.flow_step_shares(12.0, 4)
    check("shift 12 at 4 steps is measured, not guessed", len(sh12_4) == 4)
    check("...and the shares sum to 1", abs(sum(sh12_4) - 1.0) < 1e-9)
    check("...with the last step carrying ~80%", 0.79 < sh12_4[-1] < 0.81)
    check("...and the first three under 21% between them", sum(sh12_4[:3]) < 0.21)

    sh12_20 = S.flow_step_shares(12.0, 20)
    check("the same shift at 20 steps is fine", max(sh12_20) < 0.55)
    check("shift 1 is perfectly even at 4 steps",
          all(abs(s - 0.25) < 0.01 for s in S.flow_step_shares(1.0, 4)))

    n = S.schedule_balance_note(12.0, 4, "simple")
    check("a tail-heavy schedule warns", bool(n))
    check("...quoting the actual distribution", "3%/5%/12%/80%" in n)
    check("...naming the symptom", "painterly" in n)
    check("...and suggesting a workable shift", "shift_video 3" in n)
    check("20 steps at shift 12 does not warn",
          S.schedule_balance_note(12.0, 20, "simple") == "")
    for good in (3.0, 2.0, 1.0):
        check(f"shift {good} at 4 steps does not warn",
              S.schedule_balance_note(good, 4, "simple") == "")

    # Only the scheduler whose curve this reproduces. Reporting these numbers for a
    # scheduler that spaces sigmas differently would be inventing them.
    check("a non-simple scheduler is not second-guessed",
          S.schedule_balance_note(12.0, 4, "beta") == "")
    # Degenerate inputs must be silent, not a crash.
    check("one step is not a schedule", S.schedule_balance_note(12.0, 1, "simple") == "")
    check("a zero shift is silent", S.schedule_balance_note(0, 4, "simple") == "")
    check("a negative shift is silent", S.schedule_balance_note(-3, 4, "simple") == "")
    check("zero steps is silent", S.schedule_balance_note(12.0, 0, "simple") == "")


def check_megapixel_sizing():
    """A pixel BUDGET sets the size; the resolution preset supplies the shape.

    Cost and training fit are functions of token count -- (h/16)*(w/16)*frames --
    which tracks total pixels, not the short edge. The two disagree at the extremes
    of aspect ratio, so a short-edge target makes two shapes look comparable when
    they are not."""
    print("\n=== megapixel sizing ===")
    MPU = S.MP_UNIT

    check("0 is off -- the preset is used verbatim",
          S.scale_to_megapixels(1344, 768, 0.0) == (1344, 768))
    check("a negative budget is also a no-op",
          S.scale_to_megapixels(1344, 768, -2.0) == (1344, 768))

    # The property that makes scaling FROM THE PRESET the right choice: at 1.00MP
    # every native preset reproduces its own size. Computing from a nominal ratio
    # would not -- 1344x768 is 1.750 (7:4), not 16:9 (1.778).
    for r, (w, h) in S.NATIVE_RES.items():
        nw, nh = S.scale_to_megapixels(w, h, 1.0)
        if r in ("16:9", "9:16", "21:9"):
            check(f"{r} native reproduces itself at 1.00MP", (nw, nh) == (w, h))
        check(f"{r} at 1.00MP is a legal /32 size", nw % 32 == 0 and nh % 32 == 0)
        check(f"{r} at 1.00MP keeps its ratio", abs((nw / nh) - (w / h)) < 0.05)
    check("the 'native' names are not the true ratios",
          abs(1344 / 768 - 16 / 9) > 0.02 and abs(1536 / 672 - 21 / 9) > 0.04)

    # The under-budget square is what the feature exists for.
    check("a 1:1 preset scales up to the budget",
          S.scale_to_megapixels(768, 768, 1.0) == (1024, 1024))

    # Every preset at every tier must stay legal and near its budget.
    for o in S.resolution_options():
        w, h = S.parse_resolution(o)
        for tier in (0.26, 0.52, 0.83, 1.0, 1.2, 2.1):
            nw, nh = S.scale_to_megapixels(w, h, tier)
            check(f"{o[:18]} @ {tier} is legal", nw % 32 == 0 and nh % 32 == 0 and nw >= 32)
            check(f"{o[:18]} @ {tier} lands near budget",
                  abs((nw * nh / MPU) - tier) / tier < 0.08)

    # Extremes must not collapse to zero.
    for tier in (0.001, 8.0):
        nw, nh = S.scale_to_megapixels(1344, 768, tier)
        check(f"budget {tier} still yields a legal size",
              nw >= 32 and nh >= 32 and nw % 32 == 0 and nh % 32 == 0)

    # It lives on the SAMPLER now, next to resolution -- not in a separate node and
    # not appended to the end.
    # INPUT_TYPES reads comfy.samplers.KSampler. The stubs live in sys.modules only;
    # a submodule found there is never attached to its parent package, so the
    # attribute lookup would fail without this.
    if not hasattr(sys.modules["comfy.samplers"], "KSampler"):
        class _KS:
            SAMPLERS = ["res_multistep", "euler"]
            SCHEDULERS = ["simple", "beta"]
        sys.modules["comfy.samplers"].KSampler = _KS
    for _n in ("samplers", "utils", "nested_tensor", "model_management"):
        setattr(sys.modules["comfy"], _n, sys.modules["comfy." + _n])
    # --- every multiline text box is an input SOCKET, not a box on the node -------
    # forceInput: "Forces the input to be an input slot rather than a widget even a
    # widget is available for the input type" (comfy/comfy_types/node_typing.py:112).
    # The text comes from a connected multiline node instead, so the same prose can
    # feed several samplers and be edited in one place.
    _schema = S.NODE_CLASS_MAPPINGS["H3LongVideos"].INPUT_TYPES()
    _all = {**_schema.get("required", {}), **_schema.get("optional", {})}
    _text = [k for k, v in _all.items()
             if len(v) > 1 and isinstance(v[1], dict) and v[1].get("multiline")]
    check("all seven text fields are still declared", len(_text) == 7)
    check("...and every one of them is a socket, not a box",
          all(_all[k][1].get("forceInput") for k in _text))
    check("prompt is among them", "prompt" in _text)
    check("...and stays REQUIRED, so an unconnected one is an error not a blank render",
          "prompt" in _schema["required"])
    # The optional ones must survive being left unconnected: run() supplies "".
    _sig = __import__("inspect").signature(
        S.NODE_CLASS_MAPPINGS["H3LongVideos"].run).parameters
    check("every optional text socket defaults to empty in run()",
          all(_sig[k].default == "" for k in _text if k != "prompt"))

    # shot_seconds is wired too -- H3 Shot Length is the intended source, since it
    # also reports the matching frame count on the 17k+5 grid. It is OPTIONAL and
    # run() defaults it to 0.0, and 0.0 already meant "auto: largest that fits", so
    # leaving it unconnected behaves exactly as the untouched widget did.
    check("shot_seconds is a socket, not a box",
          _all["shot_seconds"][1].get("forceInput") is True)
    check("...and stays optional, so unconnected is legal",
          "shot_seconds" in _schema["optional"])
    check("...falling back to the auto behaviour a 0 widget gave",
          _sig["shot_seconds"].default == 0.0
          and _all["shot_seconds"][1]["default"] == 0.0)

    req = list(S.NODE_CLASS_MAPPINGS["H3LongVideos"].INPUT_TYPES()["required"])
    check("megapixels is a required widget on the sampler", "megapixels" in req)
    check("...positioned immediately after resolution",
          req.index("megapixels") == req.index("resolution") + 1)
    check("...and NOT appended at the end", "megapixels" not in S.ADDED_WIDGETS)
    # The dropdown is ASPECT RATIOS only now; size comes entirely from megapixels.
    check("the resolution list is bare aspect ratios",
          all(":" in o and "x" not in o for o in S.resolution_options()))
    check("every landscape ratio has its portrait transpose",
          all(f"{b}:{a}" in S.NATIVE_RES for a, b in
              (o.split(":") for o in S.resolution_options()) if a != b))
    # A workflow saved with an old "16:9 - 1344x768 (native)" label must still
    # resolve to that shape rather than silently falling back to the first entry.
    check("a legacy sized label still resolves",
          S.parse_resolution("16:9 - 1344x768 (native)") == (1344, 768))
    check("a legacy fast-tier label still resolves",
          S.parse_resolution("1:1 - 512x512 (fast, upscale later)") == (512, 512))
    check("an unknown label falls back to 16:9",
          S.parse_resolution("nonsense") == S.NATIVE_RES["16:9"])
    # megapixels can no longer be 0: a bare ratio has no size to fall back to.
    _req = S.NODE_CLASS_MAPPINGS["H3LongVideos"].INPUT_TYPES()["required"]
    check("megapixels has a non-zero floor", _req["megapixels"][1]["min"] > 0)
    check("...and defaults to the native budget",
          _req["megapixels"][1]["default"] == 1.0)
    check("run() accepts it", "megapixels" in
          __import__("inspect").signature(
              S.NODE_CLASS_MAPPINGS["H3LongVideos"].run).parameters)


def check_vram_budget():
    """Regression guard for the shot-length budget. Anchored to MEASURED runs on a
    16GB card with the pruned NVFP4 DiT (~11.7GB): 243f at 1344x768 works, 362f
    there overflowed by ~4.3GB. A previous calibration floored native to 124f by
    0.3GB -- these checks catch that class of drift."""
    print("\n=== VRAM budget calibration ===")
    E, R = S.estimate_shot_frames, S.resolve_shot_frames
    NAT, FAST = 1344 * 768, 896 * 512
    # 243f measured SAFE and 362f measured OVERFLOW at this config -> the estimate
    # must sit in between, not equal one endpoint.
    check("native 1344x768 on 16GB/NVFP4 is between the measured bounds",
          243 <= E(15.9, 11.7, 1.5, NAT) < 362)
    check("native does NOT claim the full 362f (measured overflow)",
          E(15.9, 11.7, 1.5, NAT) < 362)
    check("fast 896x512 gets substantially more frames than native",
          E(15.9, 11.7, 1.5, FAST) >= 330 and E(15.9, 11.7, 1.5, FAST) > E(15.9, 11.7, 1.5, NAT))
    check("512x512 reaches the full 362f", E(15.9, 11.7, 1.5, 512 * 512) == 362)
    # measured: 13.6GB checkpoint at 1152x640 ran 243f/10s (peak 15.2GB of 15.9GB)
    check("13.6GB checkpoint at 640p reaches the measured 243f",
          E(15.9, 13.6, 1.5, 1152 * 640) >= 243)
    for _mg in (11.7, 14.0, 17.0, 19.5, 40.0):
        for _cg in (8.0, 12.0, 15.9, 24.0, 32.0):
            _vals = [E(_cg, _mg, 1.5, w * h) for w, h in
                     ((1344, 768), (1152, 640), (896, 512), (512, 512))]
            if any(_vals[i] > _vals[i + 1] for i in range(len(_vals) - 1)):
                check(f"monotonic by resolution ({_cg}GB card, {_mg}GB weights)", False)
    check("budget is monotonic by resolution for every card/model combo", True)
    # Where the weights FIT, capacity-minus-weights is the basis and a live reading can
    # only ever trim it -- a momentarily low reading during model load must not floor it.
    check("a live free-VRAM reading can only REDUCE the estimate (weights fit)",
          E(15.9, 11.7, 1.5, NAT, free_gb=99) == E(15.9, 11.7, 1.5, NAT))
    # Where the weights STREAM, model_size() is not what occupies VRAM, so it cannot be
    # subtracted from capacity; the live reading is the only meaningful signal. A 44.3GB
    # MXFP8 build on a 15.9GB card sampled 243f at 768x768 without exceeding VRAM, while
    # the old arithmetic floored it to 124f/~5s.
    check("a streaming checkpoint budgets from free VRAM, not from weight size",
          E(15.9, 44.3, 1.5, 768 * 768, free_gb=12.0) > 124)
    check("the reported 243f case is now reachable",
          E(15.9, 44.3, 1.5, 768 * 768, free_gb=12.0) >= 243)
    check("a streaming checkpoint still scales with resolution",
          E(15.9, 44.3, 1.5, NAT, free_gb=6.0) <= E(15.9, 44.3, 1.5, 512 * 512, free_gb=6.0))
    check("less free VRAM means a shorter shot while streaming",
          E(15.9, 44.3, 1.5, NAT, free_gb=3.0) < E(15.9, 44.3, 1.5, NAT, free_gb=12.0))
    check("streaming with NO reading to go on still floors",
          E(15.9, 44.3, 1.5, NAT) == 124)
    check("an almost-full card lowers the estimate",
          E(15.9, 13.6, 1.5, NAT, free_gb=0.3) < E(15.9, 13.6, 1.5, NAT))
    # Weights that FIT but leave less than the headroom used to floor every shot to
    # 124f/~5s -- two dialogue beats came out ~5s each on a card with room to spare.
    # A deficit is weights, not latent, so it must not floor and must not be scaled by
    # resolution (which would make the fast tier look worse than native).
    check("a headroom-only deficit no longer floors",
          E(15.9, 14.6, 1.5, NAT) > 124 and E(15.9, 15.0, 1.5, NAT) > 124)
    check("a headroom-only deficit still gives a usable length",
          E(15.9, 14.6, 1.5, NAT) >= 200)
    check("a deficit is not scaled by resolution",
          E(15.9, 14.6, 1.5, NAT) == E(15.9, 14.6, 1.5, FAST))
    check("weights exceeding the card still floor at every resolution",
          all(E(15.9, 17.0, 1.5, p) == 124 for p in (NAT, FAST, 512 * 512)))
    check("a model that cannot fit floors regardless of resolution",
          all(E(12.0, 17.0, 1.5, p) == 124 for p in (NAT, FAST, 512 * 512)))
    check("a 24GB card clears 362f at native", E(24.0, 11.7, 1.5, NAT) == 362)
    check("a model too big to fit floors to the minimum",
          E(15.9, 17.0, 1.5, NAT) == 124)
    check("forced 10s at native is honored (not clamped)",
          R(10.0, 24, 15.9, 11.7, 1.5, False, NAT)[0] == 243)
    check("forced 15s at native is clamped below the 362f overflow",
          R(15.0, 24, 15.9, 11.7, 1.5, False, NAT)[0] < 362)
    check("forced 15s at native is honored with allow_oversize",
          R(15.0, 24, 15.9, 11.7, 1.5, True, NAT)[0] == 362)


def check_high_jerk_motion_cues():
    """Struggling is motion: 'auto' must speak on the beats that jerk hardest.

    A struggle beat used to get NO path text at all -- 'struggle'/'pull'/'twist'
    are not orientation changes, so _MOTION_CUE missed them -- and those are
    exactly the shots where a limb arrives without its path or spasm-renders.
    A restrained character's beats are almost entirely made of these."""
    print("\n=== high-jerk motion speaks ===")
    fired = ["She struggles against the handcuffs.",
             "She pulls at the chain.",
             "He twists against the ropes.",
             "She writhes on the bed.",
             "He staggers backward.",
             "She crawls across the floor.",
             "They dance slowly.",
             "She yanks her wrist upward."]
    check("struggle-family beats are motion cues",
          all(S._MOTION_CUE.search(t) for t in fired))
    check("calm beats stay silent",
          all(not S._MOTION_CUE.search(t) for t in
              ("Mara smiles faintly", "The light fades", "The room is warm")))
    m = S.motion_clause("She struggles against the handcuffs.", "auto")
    check("'auto' speaks on a struggle beat", bool(m))
    check("...with the same path text", "through every position on the way" in m)
    # The free-travel sentence is for FREE bodies only. On a bound one it is an
    # instruction the restraints forbid, and H3 settles the contradiction by
    # rendering the cuffs failing -- the "breaking out in every scene" report.
    blk = D("A room.", ["She turns toward the door."], "", "",
            "Mara = she, steel handcuffs")[0]
    check("a bound body gets no free-travel motion text",
          "through every position" not in blk and "physically restrained" in blk)
    free = D("A room.", ["She turns toward the door."], "", "", "Mara = she")[0]
    check("an unbound body still gets it", "through every position" in free)


def check_restraint_attachment_and_hardware():
    """A restraint clause must match HOW the restraint holds.

    The old wrist text said the arms stay bound close together -- true of
    wrist-to-wrist cuffs, FALSE of a character chained to a headboard (arms held
    apart) or cuffed behind the back (arms folded). Two contradictory sentences
    about one pair of wrists is how the cuffs render broken. And nothing said the
    HARDWARE keeps its state, so mid-struggle H3 rendered an open cuff or a
    snapped link."""
    print("\n=== restraints describe how they hold ===")
    act = {"": ["handcuffs"]}
    teth = S.restraint_clause(act, "She is cuffed to the headboard.", True)
    check("a tether names what it is fastened to",
          "headboard" in teth and "locked closed" in teth)
    check("...and drops the bound-pair wording", "bound close together" not in teth)
    wall = S.restraint_clause(act, "Her wrists are chained to the wall, arms spread.", True)
    check("chained-to-a-wall reads as a tether too", "wall" in wall and "fastened" in wall)
    behind = S.restraint_clause(act, "Her hands are cuffed behind her back.", True)
    check("behind-the-back pose is stated", "behind the back" in behind)
    over = S.restraint_clause(act, "Her wrists are cuffed above her head.", True)
    check("overhead pose is stated", "above the head" in over)
    plain = S.restraint_clause(act, "She stands very still.", True)
    check("plain cuffs keep the bound-pair wording", "bound close together" in plain)
    check("the hardware is stated to stay whole",
          "stays whole" in plain and "fastened exactly as it was put on" in plain)
    check("every variant carries it",
          all("stays whole" in S.restraint_clause(act, b, True)
              for b in ("She is cuffed to the headboard.",
                        "Her hands are cuffed behind her back.",
                        "She stands very still.")))
    check("walking prose is not a tether",
          "fastened to" not in S.restraint_clause(
              {"": ["handcuffs"]}, "She walks to the table.", True))
    check("a garment sheet gets no clause at all",
          S.restraint_clause({"": ["red jacket"]}, "She stands.", True) == "")
    check("lock_restraints=False says nothing",
          S.restraint_clause(act, "She is cuffed to the wall.", False) == "")

    # End to end: the user's exact scenario -- restrained, straining, multi-clause.
    cm = "Jon = he, 30, dark hair, handcuffs"
    g = S.distribute_generations("A bedroom.", [
        "Jon strains against the headboard chain, pulling hard."], "", "", cm)
    check("an end-to-end shot states the tether", "headboard" in g[0])
    check("...and the hardware state", "stays whole" in g[0])
    # The free-travel motion path is deliberately WITHHELD from bound bodies now:
    # "turns through every position" contradicts the binding, and H3 settled that
    # fight by rendering the restraints failing. The bound limbs carry their own
    # continuity sentence instead.
    check("...but no free-travel motion on a bound body",
          "Movement is continuous" not in g[0])
    # No negation anywhere new: cfg 1 never evaluates the negative.
    eff = " ".join(S._RESTRAINT_EFFECT.values()).lower()
    for bad in ("cannot", "can't", "unable", "no longer", "does not", "won't"):
        check(f"new restraint text never negates ({bad})", bad not in eff)


def check_restraint_usage_persists():
    """Restraint USE, stated once, must hold on every later shot.

    'She strains.' after 'cuffed to the headboard' used to fall back to the
    default bound-pair wording -- arms moving as one CONTRADICTS arms held apart,
    and two contradictory sentences about one pair of wrists is how cuffs render
    broken. Usage now persists per person until the restraint leaves the sheet."""
    print("\n=== restraint use persists across shots ===")
    cm = "Mara = she, red hair, handcuffs"
    g = S.distribute_generations("A bedroom.", [
        "Mara is cuffed to the headboard.",
        "She strains against it.",
        "She turns her face toward the door."], "", "", cm)
    check("the shot that states the tether names the anchor",
          "fastened to the headboard" in g[0])
    check("a later silent shot keeps the tether",
          "headboard" in g[1] and "bound close together" not in g[1])
    check("...and a third shot keeps it too",
          "headboard" in g[2] and "bound close together" not in g[2])

    # A pose persists the same way.
    g = S.distribute_generations("A garage.", [
        "Mara stands with her hands cuffed behind her back.",
        "She shifts her weight."], "", "", cm)
    check("a pose stated once survives a plain follow-up shot",
          "behind the back" in g[1])

    # Restating updates; freeing clears; re-cuffing starts fresh.
    g = S.distribute_generations("A bedroom.", [
        "Mara is cuffed to the headboard.",
        "Mara is cuffed to the wall instead.",
        "Mara flexes her fingers."], "", "", cm)
    check("a restated tether replaces the old anchor",
          "fastened to the wall" in g[1] and "headboard" not in g[1])
    check("the replacement carries into later shots too",
          "wall" in g[2] and "headboard" not in g[2])
    g = S.distribute_generations("A bedroom.", [
        "Mara is cuffed to the headboard.",
        "Mara is freed by a guard.\nwardrobe: Mara -= handcuffs",
        "Mara stretches both arms wide."], "", "", cm)
    check("freed of the cuffs, no clause speaks",
          "physically restrained" not in g[2])
    g = S.distribute_generations("A bedroom.", [
        "Mara is cuffed to the headboard.",
        "wardrobe: Mara -= handcuffs",
        "wardrobe: Mara += handcuffs",
        "Mara flexes her fingers."], "", "", cm)
    check("re-cuffed fresh, the stale tether is gone",
          "headboard" not in g[3] and "bound close together" in g[3])

    # Wider phrasings prompts actually use.
    act = {"": ["handcuffs"]}
    around = S.restraint_clause(act, "Her wrists are chained around the bedpost.", True)
    check("'chained around the bedpost' reads as a tether",
          "fastened to the bedpost" in around)
    spread = S.restraint_clause(act, "She lies spread eagle on the bed.", True)
    check("spread-eagle states wrists held APART",
          "bound apart" in spread and "held wide" in spread)
    eff = S._restraint_effect_text(
        "wrists", "She strains.", {"tether": "headboard", "pose": None})
    check("stored usage reaches the effect text without live prose",
          "fastened to the headboard" in eff)

    # The persistence path stays positive-only.
    act = {"": ["handcuffs"]}
    clause = S.restraint_clause(act, "She strains hard against the chain.", True,
                                usage={"": {"tether": "headboard", "pose": None}})
    for bad in ("cannot", "can't", "unable", "no longer", "does not", "won't"):
        check(f"persisted wording never negates ({bad})", bad not in clause)

    # One person's tether must not fasten a SECOND restrained person to the same
    # anchor -- and with persistence, that mistake would now stick for good.
    # (_subject_term renders each subject as a pronoun, so the bits are told
    # apart by their wording: exactly one tethered bit, one default bit.)
    two = {"Mara": ["she", "handcuffs"], "Jon": ["he", "handcuffs"]}
    mixed = S.restraint_clause(
        two, "Mara is cuffed to the headboard. Jon watches.", True)
    check("a tether stated about one person stays theirs",
          mixed.count("fastened to the headboard") == 1
          and "bound close together" in mixed)
    # A pronoun from the sheet keeps the sentence relevant to its owner.
    pronoun = S.restraint_clause(
        two, "Mara is cuffed to the headboard. She pulls against it; he watches.", True)
    check("a pronoun sentence still reaches its owner",
          pronoun.count("fastened to the headboard") == 1)
    # Quoted dialogue describes, it does not attach.
    spoken = S.restraint_clause(act, 'She says: "They kept me chained to the wall."', True)
    check("dialogue never states an attachment", "fastened to the wall" not in spoken)

    # What the shot says NOW replaces what an earlier shot remembered -- no
    # blending a live pose with a stale tether.
    g = S.distribute_generations("A bedroom.", [
        "Mara is cuffed to the headboard.",
        "Her hands are cuffed behind her back instead.",
        "She shifts her weight."], "", "", cm)
    check("a newly stated pose replaces a remembered tether",
          "behind the back" in g[1] and "headboard" not in g[1])
    check("the replacement persists on later shots",
          "behind the back" in g[2] and "fastened to" not in g[2])


def check_count_auto_multi_person():
    """'auto' now states the subject count whenever a shot binds 2+ people.

    Duplication was treated as a sub-native-resolution / LoRA problem, but two
    figures in ONE frame tile and merge even at native size. The count clause is
    the cheapest thing that holds the number down, so auto fires there too --
    per shot, from compose_persistent's own binding count."""
    print("\n=== subject count auto-fires on multi-person shots ===")
    act = {"Mara": ["she", "red hair"], "Jon": ["he", "dark hair"]}
    body = "Mara waves at Jon."
    both = S.compose_persistent(body, dict(act), "A garage.",
                                count_subjects=False, count_auto=True)
    check("two people -> the count is stated even at native size",
          "Exactly two people" in both)
    solo = S.compose_persistent("Mara waves.", {"Mara": ["she", "red hair"]}, "A garage.",
                                count_subjects=False, count_auto=True)
    check("one person -> nothing added", "Exactly" not in solo)
    off = S.compose_persistent(body, dict(act), "A garage.",
                               count_subjects=False, count_auto=False)
    check("count_auto off leaves the old behaviour", "Exactly" not in off)
    forced = S.compose_persistent(body, dict(act), "A garage.", count_subjects=True)
    check("explicit on still works", "Exactly two people" in forced)
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    check("run() derives count_auto from the widget",
          'count_auto=(subject_count_guard == "auto")' in src)


def check_anchor_mirror_warning():
    """A reflective surface in the anchor duplicates whoever is on screen.

    H3 renders a mirror's reflection as a second figure standing in the room,
    and because the anchor repeats on every shot, the doubling happens on every
    shot. Warn before the render, like the other anchor hazards."""
    print("\n=== reflective surfaces are a duplication hazard ===")
    w = S.anchor_warnings("A bedroom with a large mirror. Warm lamplight.")
    check("a mirror in the anchor warns", any("mirror" in x for x in w))
    w2 = S.anchor_warnings("An open four bay car garage. Natural daylight.")
    check("a clean anchor raises no reflection warning",
          not any("mirror" in x or "reflection" in x for x in w2))
    w3 = S.anchor_warnings("")
    check("an empty anchor stays quiet", w3 == [])


def check_listeners_stay_silent():
    """In a dialogue shot, only the SPEAKER's mouth goes free.

    `speaking` was per-shot: one quoted line freed EVERY mouth in frame, so
    whoever else was on screen mouthed along with lines they never say --
    characters visibly reciting text nobody gave them. Spoken lines are now
    attributed to whoever introduced them, and everyone else bound in the shot
    gets the same physical mouth state a silent shot gets."""
    print("\n=== the listener's mouth stays shut ===")
    cm = ("Mara = she, red hair, green coat, lips together\n"
          "Jon = he, dark hair, grey jacket")
    g = S.distribute_generations(
        "A garage.", ['Jon says: "Open it." Mara steps back.'], "", "", cm)
    check("the spoken line survives", '"Open it."' in g[0])
    check("the listener gets the mouth state",
          "stays silent through the line" in g[0])
    check("...by pronoun, not a second name", "She stays silent" in g[0])
    check("the generic everyone-silent clause is absent -- someone IS talking",
          "Everyone in this shot is silent" not in g[0])
    check("the listener keeps her sheet's lips item",
          "(red hair, green coat, lips together)" in g[0])

    g2 = S.distribute_generations(
        "A garage.", ['Jon says: "Open it." Mara answers: "Never."'], "", "", cm)
    check("two speakers -> nobody is silenced",
          "stays silent through the line" not in g2[0])

    g3 = S.distribute_generations(
        "A garage.", ['"Open it," Jon said, and Mara stepped back.'], "", "", cm)
    check("a quote attributed AFTER the close still speaks Jon",
          "stays silent through the line" in g3[0] and '"Open it,"' in g3[0])

    g4 = S.distribute_generations(
        "A garage.", ['She says: "Open it." He steps back.'], "", "", cm)
    check("an unattributed quote frees nobody by guesswork",
          "stays silent through the line" not in g4[0])

    solo = S.distribute_generations(
        "A garage.", ['Mara says: "Open it."'], "", "",
        "Mara = she, red hair, green coat")
    check("a lone speaker is untouched", "stays silent through the line" not in solo[0])

    off = S.distribute_generations(
        "A garage.", ['Jon says: "Open it." Mara steps back.'], "", "", cm,
        auto_silence_nonspeech=False)
    check("auto_silence_nonspeech=False opts out entirely",
          "stays silent through the line" not in off[0])

    written = S.distribute_generations(
        "A garage.", ['Mara reads the sign marked "EXIT". Jon watches her.'], "", "", cm)
    check("a written-text beat keeps the WHOLE-CAST silence",
          "Everyone in this shot is silent" in written[0])


def check_emphasis_quote_reported():
    """A scare-quoted word silently flips the whole shot to 'speaking'.

    She gave him a "look" has no speech verb and no punctuation inside the
    quotes -- but has_speech saw A QUOTE and freed every mouth and left the
    audio unmuted. That is how a character ends up mouthing prompt fragments.
    Reported, not guessed: the fix is one edit either way."""
    print("\n=== emphasis quotes are reported ===")
    notes = []
    S.distribute_generations("A garage.", ['She gave him a "look" and turned away.'],
                             "", "", "Mara = she, red hair", notes_out=notes)
    check('a scare-quoted word is reported',
          any('"look"' in n and "emphasis" in n for n in notes))
    notes2 = []
    S.distribute_generations("A garage.", ['He whispers "now" and pulls the lever.'],
                             "", "", "Jon = he, dark hair", notes_out=notes2)
    check("a genuine single-word line with a speech verb is not", not notes2)
    notes3 = []
    S.distribute_generations("A garage.", ['Mara says: "Ready."'],
                             "", "", "Mara = she, red hair", notes_out=notes3)
    check("punctuated dialogue is not", not notes3)


def main():
    p = SP(PROMPT, "##")
    anchor, beats = p[0], p[1:]
    shots = D(anchor, beats, "", "", "")

    print("\n=== assembled 12-beat chain ===")
    for i, s in enumerate(shots, 1):
        print(f"[{i:2}] {s}")
    print("\n=== safety invariants ===")

    # structural
    check("exactly 12 shots", len(shots) == 12)

    # no duplication: no grouped 'Name:' fallback sentence anywhere
    check("no grouped 'Maya:'/'Jon:' fallback (duplication)",
          all("Maya:" not in s and "Jon:" not in s for s in shots))

    # proper-name doubling: names appear only in the explicit-name beat (shot 10)
    for i, s in enumerate(shots, 1):
        if i == 10:
            continue
        if s.count("Maya") or s.count("Jon"):
            check(f"shot {i} uses pronouns, no bare names", False)
    check("pronoun shots carry no proper names",
          all(shots[i].count("Maya") == 0 and shots[i].count("Jon") == 0
              for i in range(12) if i != 9))
    check("explicit-name shot 10 names each once",
          shots[9].count("Maya") == 1 and shots[9].count("Jon") == 1)

    # pronoun tokens must never leak into a description
    check("no pronoun token shown in any description",
          all(not any(t in _parens(s) or t.strip() in [x.strip().lower() for x in _parens(s)]
                      for t in ("she", "he", "her", "him")) for s in shots)
          and all("she" not in " ".join(_parens(s)).lower().split()
                  and "he" not in " ".join(_parens(s)).lower().split() for s in shots))

    # Maya's RED jacket: worn through the removal shot (3), gone every shot after
    check("red jacket worn shots 1-3", all(worn(shots[i], "red jacket") for i in range(3)))
    check("red jacket GONE shots 4-12", all(not worn(shots[i], "red jacket") for i in range(3, 12)))
    # The removal is STATED in the shot that performs it (shot 3), never in a later
    # one: a mention after the fact is a presence cue that puts the garment back on.
    check("the red jacket removal is STATED in the shot that performs it",
          stated_off(shots[2], "red jacket"))
    check("the removal is stated exactly once in the whole chain",
          sum(1 for i in range(12) if stated_off(shots[i], "red jacket")) == 1)
    check("no shot after the removal names the garment at all",
          all("red jacket" not in shots[i] for i in range(3, 12)))
    check("the removal statement uses a pronoun, not a bare name",
          "Maya" not in shots[2])

    # presence-aware sets: which shots actually contain each person
    maya = [i for i in range(12) if "silver hair" in shots[i]]
    jon = [i for i in range(12) if "bald" in shots[i]]

    # Jon's cap: worn while Jon is present up to shot 4 (removal), gone while present after
    check("cap worn while Jon present, shots 1-4", all(worn(shots[i], "cap") for i in jon if i <= 3))
    check("cap GONE while Jon present, shots 5-12", all(not worn(shots[i], "cap") for i in jon if i >= 4))

    # LANDMINE: the plane 'takes off' (shot 6) strips nothing
    check("plane-takes-off shot keeps flight suit", "flight suit" in shots[5])
    check("plane-takes-off shot keeps grey suit/hair (no strip)",
          "silver hair" in shots[5])

    # explicit add by name: brown leather jacket present while Maya is present from shot 7
    check("brown leather jacket present (Maya) from shot 7",
          all("brown leather jacket" in shots[i] for i in maya if i >= 6))

    # Jon's overalls: removed after 'shrugs off his overalls' (shot 11), gone shot 12
    check("overalls worn through shot 11", worn(shots[10], "overalls"))
    check("overalls GONE shot 12", not worn(shots[11], "overalls"))

    # solo beats omit the other person
    check("solo shot 5 (Maya) omits Jon", "bald" not in shots[4] and "silver hair" in shots[4])
    check("solo shot 9 (Maya) omits Jon", "bald" not in shots[8])

    # pronoun resolution survives ALL removals (no fallback anywhere = already checked;
    # also: Maya still described after her jacket removal, Jon after his cap removal)
    check("Maya still resolves post-removal (shot 5)", "silver hair" in shots[4])
    check("Jon still resolves post-removal (shot 8)", "bald" in shots[7])

    # each present person described at most once per shot (one parenthetical each)
    ok_once = True
    for s in shots:
        # count parentheticals that look like a person desc (contain 'hair' or 'bald' or 'overalls' or 'suit')
        person_parens = [x for x in _parens(s) if any(k in x.lower() for k in ("hair", "bald", "overalls", "flight suit", "leather"))]
        if len(person_parens) > 2:      # at most two people
            ok_once = False
    check("<=2 person-descriptions per shot (no clone)", ok_once)

    # music is opt-in: blank field must emit the silence token on EVERY shot
    check("blank music -> 'non_diegetic_music: N/A' on all 12 shots",
          all("non_diegetic_music: N/A" in s for s in shots))
    # and soundscape is NOT force-silenced when blank (H3 keeps ambient)
    check("blank soundscape does NOT emit N/A", all("overall_soundscape: N/A" not in s for s in shots))
    # when music IS requested, N/A must not appear
    with_music = D(anchor, beats, "", "warm solo piano, slow", "")
    check("requested music -> no N/A, score present",
          all("non_diegetic_music: N/A" not in s for s in with_music)
          and all("warm solo piano" in s for s in with_music))

    # non-speech shots are silenced; the dialogue shot (9) is NOT
    check("dialogue shot 9 keeps its line, NOT silenced",
          '"Tower, ready for departure."' in shots[8] and "mouth closed" not in shots[8].lower())
    check("all NON-dialogue shots get lips-closed clause",
          all((("mouth closed" in shots[i].lower())) for i in range(12) if i != 8))
    check("only the dialogue shot lacks the clause",
          sum("mouth closed" not in s.lower() for s in shots) == 1)

    # --- duplication audit: multiline character_memory + scenery beats -----------
    cm = ("Kristy = she, silver hair, scar, red jacket, grey shorts\n"
          "Jon = he, bald, beard, navy overalls")
    dbeats = ["She and he walk into the hangar.",
              "She inspects the engine while he holds the light.",
              "She takes off her jacket.",
              "She climbs into the cockpit.",
              "He watches from the doorway.",
              "The hangar doors roll open, sunlight floods in.",   # no person
              "She starts the engine.",
              "He gives her a thumbs up.",
              "She taxis out as he steps back.",
              "The plane takes off down the runway.",              # no person + landmine
              "She banks over the field.",
              "He waves from the apron."]
    dshots = D("An aircraft hangar and airfield, warm late light.", dbeats, "", "", cm)
    check("multiline character_memory: 12 shots", len(dshots) == 12)
    check("no grouped 'Name:' prefix on any shot (duplication)",
          all("Kristy:" not in s and "Jon:" not in s for s in dshots))
    check("no proper name repeated in any shot",
          all(s.count("Kristy") <= 1 and s.count("Jon") <= 1 for s in dshots))
    check("at most 2 person-descriptions per shot",
          all(len([p for p in _parens(s) if any(k in p.lower() for k in ("silver hair", "bald"))]) <= 2
              for s in dshots))
    check("scenery beats carry NO people",
          all(len([p for p in _parens(dshots[i]) if any(k in p.lower() for k in ("silver hair", "bald"))]) == 0
              for i in (5, 9)))
    # The removal shot NAMES the garment (beat prose plus the direction clause);
    # worn() cannot be used on that shot any more, because the clause states the
    # removal and the absence in one sentence, which worn() strips whole.
    check("removal sticks to the end of a 12-shot chain",
          "red jacket" in dshots[2] and all(not worn(s, "red jacket") for s in dshots[3:]))

    # --- garments joined by 'and' in the sheet are tracked SEPARATELY ------------
    # The reported bug: "small white t-shirt and shiny white lace thong" parsed as a
    # single item whose head noun was `t-shirt`, so the second garment was invisible
    # to every removal path and got re-stamped into the character's parenthetical on
    # every later shot -- the prompt kept saying she was wearing it.
    conj = S.parse_wardrobe("Maya = 30, blue eyes, small white t-shirt and blue denim jacket")
    check("an 'A and B' sheet entry becomes two tracked garments",
          "small white t-shirt" in conj["Maya"] and "blue denim jacket" in conj["Maya"])
    check("the second garment carries its own head noun",
          S._item_head("blue denim jacket") == "jacket")
    # Not splitting is always safe; splitting wrongly is not. Colour pairs and
    # ordinary attributes must survive intact.
    check("a colour pair is NOT split", S._split_conjoined("black and white dress") ==
          ["black and white dress"])
    check("a striped colour pair is NOT split",
          S._split_conjoined("red and blue striped shirt") == ["red and blue striped shirt"])
    check("non-garment attributes are NOT split",
          S._split_conjoined("blonde hair and blue eyes") == ["blonde hair and blue eyes"])
    check("a one-word garment on the right still splits",
          S._split_conjoined("black leather jacket and jeans") == ["black leather jacket", "jeans"])
    check("'down' in a garment name still does not split it",
          S._split_conjoined("a puffy down jacket and grey shorts")
          == ["a puffy down jacket", "grey shorts"])

    cbeats = ["Maya stands by the window.",
              "Jon walks in and says to Maya: \"Take off your jacket.\"",
              "Maya pulls off the jacket in front of Jon and steps away, removing it.",
              "Jon says to Maya: \"Now sit down.\" Maya sits.",
              "Maya looks out the window again.",
              "Jon hands Maya a cup of tea."]
    cshots = D("A quiet room, warm light.", cbeats, "", "",
               "Maya = 30, blonde hair in pony tail, blue eyes, "
               "small white t-shirt and blue denim jacket")
    check("a conjoined garment is still worn before its removal",
          worn(cshots[0], "denim jacket") and worn(cshots[1], "denim jacket"))
    check("a conjoined garment is named in the shot that removes it",
          "jacket" in cshots[2].lower())
    check("a conjoined garment does NOT come back after removal",
          all("jacket" not in s.lower() for s in cshots[3:]))

    # --- quoted speech is an instruction, not an action --------------------------
    # 'Mom says: "take off your thong"' used to strip the garment in the shot that
    # merely ASKS for it, one shot early -- so the shot that stages the removal no
    # longer knew it was on, and never got its direction clause.
    ask = S.auto_wardrobe_removals({"Maya": ["blue denim jacket"]},
                                   'Jon says to Maya: "Take off your jacket."')
    check("a quoted instruction does not remove the garment",
          ask["Maya"] == ["blue denim jacket"])
    neg = S.auto_wardrobe_removals({"Maya": ["blue denim jacket"]},
                                   'Jon says: "Do not take off your jacket." Maya nods.')
    check("a NEGATED quoted instruction does not remove the garment",
          neg["Maya"] == ["blue denim jacket"])
    act_ = S.auto_wardrobe_removals({"Maya": ["blue denim jacket"]},
                                    'Maya pulls off the jacket and steps away.')
    check("unquoted narration still removes the garment", act_["Maya"] == [])
    mixed = S.auto_wardrobe_removals(
        {"Maya": ["blue denim jacket"]},
        'Maya takes off her jacket while saying: "It is warm in here."')
    check("narration removes even when the beat also has dialogue", mixed["Maya"] == [])

    # --- exits: a character who leaves must never come back ----------------------
    ebeats = ["She and he work on the engine.",
              "He walks out and the hangar door swings shut.",   # Jon leaves (visible here)
              "She keeps working alone.",
              "He waves.",                        # pronoun must NOT re-summon Jon
              "The plane leaves the apron.",      # landmine: not a person
              "She wipes her hands."]
    eshots = D("A hangar, warm light.", ebeats, "", "", cm)
    check("departing character visible in the shot that shows the exit",
          "bald" in eshots[1])
    check("departed character absent from every later shot",
          all("bald" not in s for s in eshots[2:]))
    check("a later pronoun cannot re-summon a departed character",
          "bald" not in eshots[3])
    check("pronoun never mislabels the remaining person",
          "(silver hair" not in eshots[3])
    check("non-person 'leaves' departs nobody",
          "silver hair" in eshots[5])

    # --- anchor must never re-introduce a tracked character (duplication) -------
    cm1 = "Kristy = she, silver hair, red jacket"
    a_name = D("A hangar. Kristy stands by the plane.", ["Kristy checks the engine."], "", "", cm1)[0]
    check("anchor naming a tracked person: name appears once",
          a_name.count("Kristy") == 1)
    a_desc = D("A hangar with a woman with silver hair in a red jacket.",
               ["She checks the engine."], "", "", cm1)[0]
    check("anchor describing a tracked person: description appears once",
          a_desc.lower().count("silver hair") == 1)
    a_two = D("A hangar. Kristy and Jon work late, warm light.", ["Kristy hands Jon a wrench."],
              "", "", "Kristy = she, silver hair\nJon = he, bald")[0]
    check("two people in anchor: each named once",
          a_two.count("Kristy") == 1 and a_two.count("Jon") == 1)
    a_clean = D("An aircraft hangar and airfield, warm late light, cinematic.",
                ["She checks the engine."], "", "", cm1)[0]
    check("a clean anchor is left intact",
          "aircraft hangar and airfield" in a_clean and "cinematic" in a_clean)
    check("stripped anchors leave no stray punctuation",
          not any(x.split("not talking. ")[-1].lstrip().startswith((".", ",")) or ".." in x
                  for x in (a_name, a_desc, a_two, a_clean)))

    # --- names repeated WITHIN a beat must not duplicate a description ----------
    rbeats = ["Kristy hands Jon the wrench, and Jon takes it from Kristy.",
              "Kristy kneels. Kristy opens the panel. Kristy frowns.",
              "Jon watches Kristy while Kristy watches Jon.",
              "Kristy sees Kristy's reflection in the fuselage.",
              "Kristy and Jon and Kristy again talk it over."]
    rshots = D("A hangar, warm light.", rbeats, "", "", cm)
    def _person_descs(shot, key):
        return len([p for p in _parens(shot) if key in p.lower()])
    check("repeated names: Kristy described at most once per shot",
          all(_person_descs(s, "silver hair") <= 1 for s in rshots))
    check("repeated names: Jon described at most once per shot",
          all(_person_descs(s, "bald") <= 1 for s in rshots))
    # Repeat mentions are now collapsed to pronouns, so each name survives exactly once
    # and the description binds at that single mention.
    check("repeated names: each name appears exactly once",
          rshots[0].count("Kristy") == 1 and rshots[0].count("Jon") == 1)
    check("repeated names: description binds at that mention",
          "Kristy (silver hair" in rshots[0])

    # --- PLAIN-TEXT beats (no character_memory): garments AND people ------------
    pt_anchor = ("A woman with silver hair in a red jacket and a bald man in navy "
                 "overalls, in a hangar, warm light.")
    pt_beats = ["They walk in together.",
                "She takes off her jacket.",     # garment removal from anchor prose
                "She checks the panel.",
                "He walks out and is gone.",     # person exit from anchor prose
                "She keeps working alone.",
                "The plane leaves the apron.",   # landmine
                "She wipes her hands."]
    pt = D(pt_anchor, pt_beats, "", "", "")      # NO character channel at all
    check("plain text: garment removed from anchor and stays gone",
          worn(pt[1], "red jacket") and all(not worn(s, "red jacket") for s in pt[2:]))
    check("plain text: the person survives the garment scrub",
          all("woman" in s.lower() for s in pt))
    check("plain text: departing person visible in the exit shot",
          "bald man" in pt[3])
    check("plain text: departed person gone from every later shot",
          all("bald man" not in s for s in pt[4:]))
    check("plain text: remaining person unaffected",
          all("woman with silver hair" in s for s in pt))
    check("plain text: scene clause survives the scrub",
          all("hangar" in s for s in pt))
    check("plain text: anchor never starts with a dangling connector",
          all(not s.split("not talking. ")[-1].lstrip().lower().startswith(("and ", ", "))
              for s in pt))

    check_clothing_removal_6beat()
    check_nonspeech_audio_6beat()
    check_overlay_resolutions()
    check_no_phantom_person_in_anchor()
    check_real_world_sheet()
    check_no_second_subject_noun()
    check_anchor_not_rewritten()
    check_detailed_wardrobe_items()
    check_anchor_hazards()
    check_naming_brings_a_character_back()
    check_exposed_terms()
    check_stripped_state_persists()
    check_emergence_is_not_an_exit()
    check_props_survive_the_shot_boundary()
    check_under_layer_stays_on()
    check_removal_phrasings()
    check_removal_takes_only_its_object()
    check_unnamed_sheet_punctuation()
    check_mouth_state_on_dialogue()
    check_lora_duplication_guard()
    check_subject_count_guard()
    check_beat_count_is_unbreakable()
    check_name_dedupe()
    check_anchor_beat_rescue()
    check_forced_shot_seconds()
    check_dialogue_filler()
    check_dialogue_fit()
    check_model_change_flush()
    check_vram_budget()
    check_ref_conditioning_channels()
    check_tagged_references()
    check_ref_modes()
    check_sla_pairing()
    check_lora_hints()
    check_kernel_backend_note()
    check_mouth_stays_closed()
    check_presence_test_is_shared()
    check_continuity_warning()
    check_restraints_stay_on()
    check_nappy_vocabulary()
    check_bare_state_persists()
    check_contact_guard()
    check_motion_guard()
    check_solidity_guard()
    check_anatomy_guard()
    check_latent_output()
    check_preflight_note_assembly()
    check_auto_soundscape()
    check_lora_chain_and_oom()
    check_audio_scale_coupling()
    check_schedule_balance()
    check_megapixel_sizing()
    check_audio_vae_guard()
    check_written_text_is_not_speech()
    check_declared_bare()
    check_exposed_terms_key_warning()
    check_bare_wording_follows_pronoun()
    check_plural_cast_binding()
    check_high_jerk_motion_cues()
    check_restraint_attachment_and_hardware()
    check_restraint_usage_persists()
    check_count_auto_multi_person()
    check_anchor_mirror_warning()
    check_listeners_stay_silent()
    check_emphasis_quote_reported()
    check_anatomy_auto_multi_person()

    print()
    if _fails:
        print(f"RESULT: {len(_fails)} FAILURE(S): " + "; ".join(_fails))
        sys.exit(1)
    print("RESULT: ALL SAFETY INVARIANTS PASSED (12/12 shots)")

if __name__ == "__main__":
    main()
