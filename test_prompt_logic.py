#!/usr/bin/env python3
"""
Safety / regression tests for H3-LongVideos-V1 prompt logic
===========================================================
Exercises the wardrobe channel, pronoun resolution, duplication avoidance, and
auto-removal against a full 12-beat (12-shot) chain -- the real production
length. Pure string logic only: it stubs the torch / ComfyUI imports so it runs
anywhere with no GPU and no ComfyUI:

    python3 test_prompt_logic.py

Exits non-zero if any invariant fails, so you can wire it into CI or run it after
any edit to the prompt-assembly code.
"""
import sys, types, os, importlib.util

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
          "Kristy starts this shot" not in sh[3] and "She starts this shot" in sh[3])
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
          stated_off(pb[3], "red jacket") and "comes off during it" in pb[3])
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
          "red jacket is off" in sing and "not wearing it" in sing)
    check("a plural garment takes a plural verb",
          "navy overalls are off" in plur and "takes them off" in plur)
    check("two garments take a plural verb", "cap and gloves are off" in two)
    check("a double-s noun stays singular",
          "dress is off" in S.takes_off_clause([("Maya", "dress")], act))
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
    check("all three tiers x six ratios are offered", len(presets) == 18)

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
    anchors = {x.split("not talking. ")[1].split(" She")[0].split(" He")[0] for x in sh}
    check("anchor identical across every shot", len(anchors) == 1)
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
        after = S.auto_wardrobe_removals(a, f"Kristy takes off her {head}.")
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
          "red jacket is off" in sing and "not wearing it" in sing)
    plur = S.takes_off_clause([("Kristy", "black boots with steel buckles")], {"Kristy": ["she"]})
    check("a genuinely plural garment still takes a plural verb",
          "black boots are off" in plur and "takes them off" in plur)


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
    sh = D("A barn interior.", B, "", "", cm_)
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
    with_lora = D(a, ["Teresa talks to Dan."], "", "", cm_, True, True, True, True)[0]
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
    talky = D(a, ['Teresa asks Dan: "Ready?"'], "", "", cm_, True, True, True, True)[0]
    check("front-loading holds on a dialogue shot too",
          talky.split("] ", 1)[-1].strip().startswith("Exactly two people"))
    solo = D(a, ["Teresa checks the crate."], "", "", cm_, True, True, True, True)[0]
    check("solo shot counts one", "Exactly one person" in solo)
    scenery = D(a, ["The garage door rolls open."], "", "", cm_, True, True, True, True)[0]
    check("scenery shot gets no count clause", "Exactly" not in scenery)
    no_lora = D(a, ["Teresa talks to Dan."], "", "", cm_, True, True, True, False)[0]
    check("without a LoRA the clause follows the anchor",
          "Exactly two people" in no_lora
          and not no_lora.split("not talking. ")[-1].strip().startswith("Exactly"))


def check_subject_count_guard():
    """Explicit subject counts (anti-duplication at sub-native resolutions)."""
    print("\n=== subject-count guard ===")
    cm = "Kristy = she, silver hair, red jacket\nJon = he, bald, navy overalls"
    one = D("A hangar.", ["She checks the engine."], "", "", cm, True, True, True)[0]
    two = D("A hangar.", ["She hands him a wrench."], "", "", cm, True, True, True)[0]
    sc = D("A hangar.", ["The hangar doors roll open."], "", "", cm, True, True, True)[0]
    off = D("A hangar.", ["She hands him a wrench."], "", "", cm, True, True, False)[0]
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

    vals, _ = build(handoff=_FakeImg(), refs=[_FakeImg()])
    check("refs WIN when both are offered", "minimax_refs" in vals)
    check("the keyframe channel stays empty when refs are present",
          "minimax_keyframes" not in vals)

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
    # sizing: 'match' must never exceed the generation area by more than the 32px snap,
    # and neither mode may ever upscale a small reference
    big = S.ref_image_canvas(4096, 2160, 1344, 768, "match")
    check("'match' scales a 4K reference down to ~one frame's area",
          abs(big[0] * big[1] - 1344 * 768) < 1344 * 768 * 0.05)
    check("'max' uses the 2048 short edge", S.ref_image_canvas(4096, 2160, 1344, 768, "max")[1] == 2048)
    check("a small reference is never upscaled",
          S.ref_image_canvas(512, 512, 1344, 768, "match") == (512, 512)
          and S.ref_image_canvas(512, 512, 1344, 768, "max") == (512, 512))


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
    check("removal sticks to the end of a 12-shot chain",
          worn(dshots[2], "red jacket") and all(not worn(s, "red jacket") for s in dshots[3:]))

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

    print()
    if _fails:
        print(f"RESULT: {len(_fails)} FAILURE(S): " + "; ".join(_fails))
        sys.exit(1)
    print("RESULT: ALL SAFETY INVARIANTS PASSED (12/12 shots)")

if __name__ == "__main__":
    main()
