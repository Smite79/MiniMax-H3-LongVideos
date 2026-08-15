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
    check("the removal still applies", "cap" not in sh[3])


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
    body = with_lora.split("not talking. ")[-1]
    check("count clause is FIRST when a LoRA is applied",
          body.strip().startswith("Exactly two people"))
    check("count clause names the right number", "Exactly two people" in body)
    check("clause forbids extra bodies explicitly",
          "no extra bodies" in body and "no repeated figures" in body)
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
    check("a live free-VRAM reading can only REDUCE the estimate",
          E(15.9, 17.0, 1.5, NAT, free_gb=99) == E(15.9, 17.0, 1.5, NAT))
    check("an almost-full card lowers the estimate",
          E(15.9, 13.6, 1.5, NAT, free_gb=0.3) < E(15.9, 13.6, 1.5, NAT))
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
    check("red jacket worn shots 1-3", all("red jacket" in shots[i] for i in range(3)))
    check("red jacket GONE shots 4-12", all("red jacket" not in shots[i] for i in range(3, 12)))

    # presence-aware sets: which shots actually contain each person
    maya = [i for i in range(12) if "silver hair" in shots[i]]
    jon = [i for i in range(12) if "bald" in shots[i]]

    # Jon's cap: worn while Jon is present up to shot 4 (removal), gone while present after
    check("cap worn while Jon present, shots 1-4", all("cap" in shots[i] for i in jon if i <= 3))
    check("cap GONE while Jon present, shots 5-12", all("cap" not in shots[i] for i in jon if i >= 4))

    # LANDMINE: the plane 'takes off' (shot 6) strips nothing
    check("plane-takes-off shot keeps flight suit", "flight suit" in shots[5])
    check("plane-takes-off shot keeps grey suit/hair (no strip)",
          "silver hair" in shots[5])

    # explicit add by name: brown leather jacket present while Maya is present from shot 7
    check("brown leather jacket present (Maya) from shot 7",
          all("brown leather jacket" in shots[i] for i in maya if i >= 6))

    # Jon's overalls: removed after 'shrugs off his overalls' (shot 11), gone shot 12
    check("overalls worn through shot 11", "overalls" in shots[10])
    check("overalls GONE shot 12", "overalls" not in shots[11])

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
          "red jacket" in dshots[2] and all("red jacket" not in s for s in dshots[3:]))

    # --- exits: a character who leaves must never come back ----------------------
    ebeats = ["She and he work on the engine.",
              "He walks out of the hangar.",      # Jon leaves (visible here)
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
    check("repeated names: description lands on the FIRST mention",
          rshots[0].index("(silver hair") < rshots[0].rindex("Kristy"))

    # --- PLAIN-TEXT beats (no character_memory): garments AND people ------------
    pt_anchor = ("A woman with silver hair in a red jacket and a bald man in navy "
                 "overalls, in a hangar, warm light.")
    pt_beats = ["They walk in together.",
                "She takes off her jacket.",     # garment removal from anchor prose
                "She checks the panel.",
                "He walks out of the hangar.",   # person exit from anchor prose
                "She keeps working alone.",
                "The plane leaves the apron.",   # landmine
                "She wipes her hands."]
    pt = D(pt_anchor, pt_beats, "", "", "")      # NO character channel at all
    check("plain text: garment removed from anchor and stays gone",
          "red jacket" in pt[1] and all("red jacket" not in s for s in pt[2:]))
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

    check_no_phantom_person_in_anchor()
    check_real_world_sheet()
    check_no_second_subject_noun()
    check_anchor_not_rewritten()
    check_mouth_state_on_dialogue()
    check_lora_duplication_guard()
    check_subject_count_guard()
    check_dialogue_fit()
    check_model_change_flush()
    check_vram_budget()

    print()
    if _fails:
        print(f"RESULT: {len(_fails)} FAILURE(S): " + "; ".join(_fails))
        sys.exit(1)
    print("RESULT: ALL SAFETY INVARIANTS PASSED (12/12 shots)")

if __name__ == "__main__":
    main()
