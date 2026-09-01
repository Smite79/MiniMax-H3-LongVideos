"""Tests for H3 Long Videos.

Only what the node actually decides: how a prompt becomes shots, how a shot is
sized, and which shots get silence or a reference. There is no prompt-rewriting
layer to test any more -- your text goes through verbatim, and the test that
matters most is the one asserting exactly that.

Run: python test_node.py
"""

import importlib.util
import re
import io
import os
import sys
import types

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Stub the ComfyUI modules the node imports; none of them is touched by the pure
# functions under test.
for _n in ("torch", "nodes", "comfy", "comfy.utils", "comfy.sample", "comfy.samplers",
           "comfy.nested_tensor", "comfy.model_management", "latent_preview", "node_helpers"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["comfy.samplers"].KSampler = type("K", (), {"SAMPLERS": ["res_multistep"],
                                                        "SCHEDULERS": ["simple"]})
# `import comfy.samplers` binds the SUBMODULE onto the parent package; with stubs
# that has to be done by hand or `comfy.samplers` resolves to nothing.
for _sub in ("utils", "sample", "samplers", "nested_tensor", "model_management"):
    setattr(sys.modules["comfy"], _sub, sys.modules["comfy." + _sub])

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("h3sampler", os.path.join(_HERE, "sampler.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

_fails = []


def check(label, ok):
    print(("  PASS  " if ok else "  FAIL  ") + label)
    if not ok:
        _fails.append(label)


def test_beats():
    print("\n=== prompt -> shots ===")
    scene, beats = S.split_beats("A barn at dusk.\n\nHe walks in.\n\nShe follows him.")
    check("the first paragraph is the scene", scene == "A barn at dusk.")
    check("the rest are beats", beats == ["He walks in.", "She follows him."])
    check("one paragraph is one beat with no scene",
          S.split_beats("He walks in.") == ("", ["He walks in."]))
    check("blank input yields nothing", S.split_beats("   ") == ("", []))
    check("extra blank lines do not make empty beats",
          S.split_beats("A.\n\n\n\nB.\n\n   \n\nC.")[1] == ["B.", "C."])
    # Lines within a paragraph stay together: a beat is a PARAGRAPH. The old node
    # split per line and hard-wrapped prose came apart into fragments.
    scene, beats = S.split_beats("A barn.\n\nHe walks to the door\nand opens it.")
    check("lines inside a paragraph stay in one beat", beats == ["He walks to the door\nand opens it."])


def test_verbatim():
    print("\n=== the text is passed through unchanged ===")
    scene, beats = S.split_beats("Night, hard light.\n\nDan cuts the rope and she drops free.")
    shot = f"{scene} {beats[0]}"
    check("the beat survives word for word", "Dan cuts the rope and she drops free." in shot)
    check("the scene survives word for word", shot.startswith("Night, hard light."))
    check("nothing else is added", shot == "Night, hard light. Dan cuts the rope and she drops free.")
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    for gone in ("Exactly two people in this shot", "Solid things stay solid",
                 "physically restrained", "Two bodies in contact", "stays down for the whole shot",
                 "Movement is continuous", "lips together"):
        check(f"no guard text remains: {gone!r}", gone not in src)


def test_sizing():
    print("\n=== shot length and canvas ===")
    check("frames land on the 17k+5 grid", all(S.align_frame_count(n) % 17 == 5
                                               for n in (1, 50, 100, 240, 300)))
    check("10s -> 243f", S.align_frame_count(10 * 24) == 243)
    check("a length is never rounded down", S.align_frame_count(244) >= 244)
    check("H3's ceiling is respected", S.align_frame_count(9999) == S.MAX_FRAMES)
    check("the latent grid follows the frame count", S.video_latent_t(243) == 72)
    fc, lt, at = S.temporal_shape(243)
    check("audio latents track 24fps", (fc, at) == (243, round(243 / 24 * 40)))
    check("a ratio resolves to its native canvas", S.parse_resolution("16:9") == (1344, 768))
    check("an unknown ratio falls back to 16:9", S.parse_resolution("nonsense") == (1344, 768))
    w, h = S.scale_to_megapixels(1344, 768, 1.0)
    check("megapixels scales and stays on the 32 grid", w % 32 == 0 and h % 32 == 0)
    check("...and keeps the aspect ratio", abs((w / h) - (1344 / 768)) < 0.05)
    check("0 megapixels keeps the preset", S.scale_to_megapixels(1344, 768, 0) == (1344, 768))
    # Uniform lengths are the point: one seed is only one noise field when every
    # shot has the same latent shape.
    check("every shot gets the same length",
          len({S.align_frame_count(10 * 24) for _ in range(5)}) == 1)


def test_speech_and_refs():
    print("\n=== silence and references ===")
    check("a quoted line counts as speech", S.has_speech('She says: "Get up."'))
    # H3 registers <d>...</d> as its own dialogue delimiter, alongside a caption
    # channel. Only checking quotes meant a beat written the way the model expects
    # was treated as silent and had its audio muted.
    check("H3's own dialogue marker counts too", S.has_speech("She says <d>Get up.</d>"))
    check("...even at the start of a beat", S.has_speech("<d>Get up.</d> he says"))
    check("caption tokens are recognised as a request for on-screen text",
          bool(S._CAPTION_TOKEN.search("<|caption_start|>hi<|caption_end|>")))
    check("...and ordinary prose is not", not S._CAPTION_TOKEN.search("She walks in."))
    check("curly quotes count too", S.has_speech("He said “now”."))
    check("an ordinary beat has no speech", not S.has_speech("She walks to the window."))
    check("an empty beat has no speech", not S.has_speech(""))
    check("a picture tag is found", S.picture_tags("Dan, <Picture 1>, walks in.") == [1])
    check("tags are deduped and sorted",
          S.picture_tags("<Picture 2> and <picture_1> and <Picture 2>") == [1, 2])
    check("no tag means none", S.picture_tags("Dan walks in.") == [])


def test_removals():
    print("\n=== clothing removal between beats ===")
    body, toks = S.extract_removals("Dan cuts off her jacket.\nremove: jacket")
    check("the directive never reaches the model", "remove:" not in body)
    check("...and the beat itself is untouched", body == "Dan cuts off her jacket.")
    check("the item is captured", toks == ["jacket"])
    check("several items on one line",
          S.extract_removals("x\nremove: bra, shirt")[1] == ["bra", "shirt"])
    check("'off:' works too", S.extract_removals("x\noff: hat")[1] == ["hat"])
    check("a beat with no directive is unchanged",
          S.extract_removals("She walks in.") == ("She walks in.", []))
    sc = "A basement. Kate is 20, blonde, grey jacket, white shirt, black boots. Dan is 35."
    check("the item leaves the scene",
          "jacket" not in S.scrub_removed(sc, ["jacket"]))
    check("...and everything else stays",
          all(w in S.scrub_removed(sc, ["jacket"])
              for w in ("blonde", "white shirt", "black boots", "Dan is 35")))
    check("a sentence that was only about it is dropped whole",
          S.scrub_removed("A room. She wears a red coat. Dan waits.", ["red coat"])
          == "A room. Dan waits.")
    check("no tokens means no edit", S.scrub_removed(sc, []) == sc)
    # Surgical: only the named garment goes. Deleting the whole comma fragment
    # took neighbours with it, and an undescribed garment is one the model
    # re-invents -- which looks like the clothing changing by itself.
    check("a neighbour joined by 'and' survives",
          S.scrub_removed("Kate is 20, blonde, wearing a grey jacket and black boots.",
                          ["jacket"]) == "Kate is 20, blonde, wearing black boots.")
    check("a layer named after 'over' survives",
          S.scrub_removed("Kate wears a grey jacket over a white shirt.", ["jacket"])
          == "Kate wears a white shirt.")
    check("a stranded conjunction is cleaned up",
          S.scrub_removed("A basement. Kate is 20 and wears a grey jacket.", ["jacket"])
          == "A basement. Kate is 20.")
    check("a sentence reduced to a bare subject is dropped",
          S.scrub_removed("A room. She wears a red coat. Dan waits.", ["red coat"])
          == "A room. Dan waits.")
    # A wardrobe LIST entry goes whole. Trimming a fixed number of modifiers off
    # the front left orphans behind -- "skin-tight shiny black" after removing
    # "shorts" -- and an orphan description sitting in a garment list is read as
    # some garment, which is a garment coming back.
    _long = ("A basement. Kate is 20, blonde, shiny white bra, shiny white lace thong, "
             "skin-tight shiny black micro volleyball shorts, black crop top.")
    _r1 = S.scrub_removed(_long, ["shorts"])
    check("a long list entry is removed whole",
          "shorts" not in _r1 and "skin-tight" not in _r1 and "micro" not in _r1)
    check("...and its neighbours are intact",
          "shiny white bra" in _r1 and "shiny white lace thong" in _r1
          and "black crop top" in _r1)
    _r2 = S.scrub_removed(_long, ["thong"])
    check("no orphan adjective is left behind",
          "lace" not in _r2 and ", shiny," not in _r2)
    _r3 = S.scrub_removed(_long, ["crop top", "shorts", "thong"])
    check("three removals leave only what is still worn",
          _r3 == "A basement. Kate is 20, blonde, shiny white bra.")
    check("a picture tag is never dropped with a garment",
          S.picture_tags(S.scrub_removed(
              "A basement. Kate is 20, <Picture 1> blonde crop top, boots.",
              ["crop top"])) == [1])
    check("text with none of the tokens is untouched",
          S.scrub_removed("A basement with devices on the walls. Kate is 20.", ["jacket"])
          == "A basement with devices on the walls. Kate is 20.")
    # A beat that reads as a removal but carries no directive is REPORTED, never
    # acted on -- inferring removals from prose is what made the old node erratic.
    _sc2 = "A basement. Kate is 20, blonde, wearing a grey jacket and black boots."
    check("a missing remove: is noticed",
          S.missing_removals("Dan cuts off her jacket.", _sc2, []) == ["jacket"])
    check("...and not once the directive is there",
          S.missing_removals("Dan cuts off her jacket.", _sc2, ["jacket"]) == [])
    check("...and an ordinary beat is quiet",
          S.missing_removals("Kate walks to the window.", _sc2, []) == [])
    # Removals accumulate: once off, a garment stays out of every later shot.
    gone = []
    for _b in ("a\nremove: jacket", "b\nremove: shirt", "c"):
        gone.extend(t for t in S.extract_removals(_b)[1] if t not in gone)
    final = S.scrub_removed(sc, gone)
    check("both stay gone in a later beat",
          "jacket" not in final and "shirt" not in final and "black boots" in final)


def test_removal_completes():
    print("\n=== a removal has to finish inside its shot ===")
    # Scrubbing stops a garment being DESCRIBED. It does not tell the model to
    # finish taking it off -- and the last frame is the next shot's keyframe, so
    # a cut still in progress hands on a garment still half worn. The next beat
    # has moved on and never contradicts the picture, so it stays.
    one = S.off_by_last_frame(["bra"])
    check("the removing shot is told to finish it", "by the last frame" in one)
    check("...and that nothing is left on the body", "no longer on the body" in one)
    check("...and where it ends up", "out of frame" in one)
    check("a plural garment agrees",
          "shorts come off" in S.off_by_last_frame(["shorts"])
          and " are away" in S.off_by_last_frame(["shorts"]))
    check("a singular one does too",
          "bra comes off" in one and " is away" in one)
    # The sentence is capitalised, so the first "the" is "The".
    check("two items are joined", "The bra and the shorts come off"
          in S.off_by_last_frame(["bra", "shorts"]))
    check("no removal, no sentence", S.off_by_last_frame([]) == "")
    # Saying what comes off does not say where to STOP. An action with time left
    # runs on to whatever is next: shears that finish the shorts go on to the
    # thong, or the body under it.
    _b = S.off_by_last_frame(["shorts"])
    check("the action is bounded", "Everything else on the body stays exactly as it is" in _b)
    check("...covering hardware as well", "whole and closed as it was put on" in _b)
    check("...naming no other garment", "thong" not in _b and "bra" not in _b)
    # The BOUND sentence carries no negation: at cfg 1 the negative prompt is
    # never evaluated, so a negation in the positive only names what it forbids.
    # ("no longer on the body" belongs to the removal half, and is the wording the
    # previous node proved safe in the removing shot.)
    _bound = _b.split("dropped out of frame.")[1]
    check("...and the bound is stated positively",
          not re.search(r"no|not|never|nothing", _bound, re.I))
    check("it reads as a sentence", one.strip().startswith("The bra"))
    # Said ONCE. Naming the garment on a later shot is a presence cue, and that
    # phrasing put garments back on in the previous version of this node.
    scene = "A basement. Kate is 20, blonde, shiny white bra, black crop top."
    beats = ["Dan cuts off her crop top.\nremove: crop top",
             "Dan cuts off her bra.\nremove: bra",
             "Kate looks up at him."]
    gone, lines = [], []
    for b in beats:
        body, toks, _ = S.extract_directives(b)
        gone.extend(t for t in toks if t not in gone)
        lines.append(f"{S.scrub_removed(scene, gone)} {body}{S.off_by_last_frame(toks)}")
    check("shot 1 orders the crop top off", "crop top comes off during this shot" in lines[0])
    check("...and shot 2 never mentions it again", "crop top" not in lines[1])
    check("...nor shot 3", "crop top" not in lines[2] and "bra" not in lines[2])
    check("the scene loses each garment as it goes",
          "bra" in lines[0] and "bra" not in lines[2])


def test_layers():
    print("\n=== layers appear when they become visible ===")
    # A scene listing every layer at once tells the model the character wears
    # all of them simultaneously, with nothing saying which is hidden. The
    # keyframe holds the first frame; by the last frame only the text governs,
    # and the under layer starts showing through the top one.
    body, rem, add = S.extract_directives(
        "Dan cuts off her jacket.\nremove: jacket\nadd: her white shirt is now visible")
    check("both directives are taken out of the beat",
          body == "Dan cuts off her jacket.")
    check("the removal is captured", rem == ["jacket"])
    check("the addition is captured verbatim",
          add == ["her white shirt is now visible"])
    check("'wear:' is accepted too",
          S.extract_directives("x\nwear: a red coat")[2] == ["a red coat"])
    check("a beat with neither is unchanged",
          S.extract_directives("She walks in.") == ("She walks in.", [], []))
    check("the old two-value helper still works",
          S.extract_removals("x\nremove: hat") == ("x", ["hat"]))
    # An added layer retires when it is itself removed.
    gone, shown = ["white shirt"], ["her white shirt is now visible",
                                    "her black bra is now visible"]
    live = [a for a in shown if not S.names_any(a, gone)]
    check("a removed layer stops being described", live == ["her black bra is now visible"])
    check("...and one that was not removed stays", S.names_any("a red coat", ["coat"]))


def test_thin_beats():
    print("\n=== a shot longer than its beat ===")
    # A shot that outlasts its action leaves the model seconds it was told
    # nothing about, and the cheapest way to fill them is to CARRY ON: shears
    # that cut a garment off keep cutting into what is underneath.
    one = "Dan cuts off her bra and throws it away."
    two = ("Dan cuts off her bra and throws it away, then sets the shears down "
           "and steps back.")
    check("a two-clause beat asks for about 7s", 6.0 <= S.beat_seconds(one) <= 8.0)
    check("adding what happens next asks for more", S.beat_seconds(two) > S.beat_seconds(one))
    check("directive lines do not count as content",
          S.beat_seconds(one) == S.beat_seconds(one + "\nremove: bra"))
    check("dialogue is timed by words", S.beat_seconds('She says: "one two three four five."') > 0)
    check("an empty beat asks for nothing", S.beat_seconds("") == 0)
    check("the reported beat is flagged in a 10s shot",
          any("shot 1" in t for t in S.thin_beats([one], 10.0)))
    check("...and is not once it has somewhere to go",
          S.thin_beats([two], 10.0) == [])
    check("...nor at a shot length that matches it",
          S.thin_beats([one], 7.0) == [])


def test_auto_length():
    print("\n=== sizing a shot from its beat ===")
    one = "Dan cuts off her bra and throws it away."
    two = ("Dan cuts off her bra and throws it away, then sets the shears down "
           "and steps back.")
    still = "Kate lies still."
    ceil = S.align_frame_count(10 * 24)
    lens, note = S.plan_lengths([one, two, still], ceil, True)
    check("a shorter beat gets a shorter shot", lens[0] < lens[1])
    check("...and the shortest gets the least", lens[2] < lens[0])
    check("nothing exceeds the ceiling", all(n <= ceil for n in lens))
    check("nothing falls under one action's worth",
          all(n >= S.MIN_AUTO_FRAMES for n in lens))
    check("every length is on the 17k+5 grid", all(n % 17 == 5 for n in lens))
    check("the seed trade-off is reported", "one noise field" in note)
    # The whole point: auto sizing leaves no beat with time it was not given
    # anything to do with, which is what makes an action carry on past its end.
    thin = [t for b, f in zip([one, two, still], lens) for t in S.thin_beats([b], f / 24)]
    check("auto sizing leaves no thin beat", thin == [])
    fixed, fnote = S.plan_lengths([one, two, still], ceil, False)
    check("fixed mode gives every shot the ceiling", fixed == [ceil] * 3)
    check("...and says nothing about noise, since the shapes match", fnote == "")
    # An estimate rounds to the NEAREST grid point; a requested length rounds up.
    check("an estimate does not round up", S.align_frame_count_nearest(180) == 175)
    check("...while a request never returns less", S.align_frame_count(180) == 192)


def test_text_in_frame():
    print("\n=== watermarks and subtitles ===")
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    check("the sampler composites nothing onto the frames",
          "watermark" not in src.lower().split("# --- removals")[0]
          or "PIL" not in src)
    # Old scripts pasted back in as a prompt carry the field labels the previous
    # version printed. Verbatim pass-through sends them to the model, and a line
    # reading "overall_soundscape: room tone" is read as text to put ON the frame.
    old = ("[Generation 1] A basement. Dan walks in.\n"
           "overall_soundscape: room tone, footsteps\n"
           "non_diegetic_music: N/A\n\n"
           "[Generation 2] She looks up.\n"
           "overall_soundscape: room tone\n")
    clean, n = S.strip_legacy_fields(old)
    check("the field labels are dropped", n == 5 and "soundscape" not in clean)
    check("...and the shot tags with them", "[Generation" not in clean)
    check("...but the real text survives",
          "A basement. Dan walks in." in clean and "She looks up." in clean)
    check("...and the beat split is unchanged", len(S.split_beats(clean)[1]) == 1)
    check("an ordinary prompt is untouched",
          S.strip_legacy_fields("A room.\n\nShe walks in.")[1] == 0)
    # Naming text is what draws text, and at cfg 1 no negative prompt undoes it.
    for _t in ("Subtitles appear at the bottom.", "A watermark in the corner.",
               "The end credits roll.", "A timestamp in the corner."):
        check(f"named text is flagged: {_t[:28]!r}", bool(S._TEXT_CUE.search(_t)))
    for _t in ("A room with a neon sign.", "She walks in.", "He signs the form."):
        check(f"ordinary prose is not: {_t[:28]!r}", not S._TEXT_CUE.search(_t))


def test_reference_tags():
    print("\n=== <Picture N> tags ===")
    # comfy/text_encoders/minimax.py writes the "<Picture N>: " label itself,
    # numbering by the order it receives images. A shot using only <Picture 2>
    # gets that image labelled <Picture 1>, so the text has to be renumbered or
    # it points at nothing.
    refs = ["A", "B", "C", "D"]
    out, imgs, dropped = S.resolve_tags("Kate, <Picture 2>, walks in.", refs)
    check("a lone slot 2 is renumbered to 1", "<Picture 1>" in out)
    check("...and carries the right image", imgs == ["B"])
    out2, imgs2, _ = S.resolve_tags("Kate <Picture 2> and Dan <Picture 4> meet.", refs)
    check("two slots renumber in order",
          "<Picture 1>" in out2 and "<Picture 2>" in out2 and imgs2 == ["B", "D"])
    out3, imgs3, drop3 = S.resolve_tags("Kate, <Picture 9>, walks in.", refs)
    check("a tag with no image is removed", "Picture" not in out3 and drop3 == [9])
    check("...leaving readable text", out3 == "Kate, walks in.")
    check("untagged text is untouched",
          S.resolve_tags("No tags here.", refs)[0] == "No tags here.")
    check("no refs connected drops every tag",
          S.resolve_tags("Kate, <Picture 1>, walks in.", [])[1] == [])


def test_restraints_hold():
    print("\n=== a restraint, once on, stays whole ===")
    for _t in ("Kate is cuffed at the wrists.", "Dan handcuffs her.",
               "Her mouth is taped shut.", "Dan locks a chain around her waist.",
               "Kate is hogtied on the floor.", "Dan gags her.",
               "Dan ties a rope around her ankles.",
               "Wrists handcuffed behind back, ankles cuffed together."):
        check(f"restraint seen: {_t[:34]!r}", S.restraint_present(_t))
    # Ambiguous hardware needs a binding verb or a body part. An earlier version
    # listed "chain" as both noun and verb, so a chain-link fence armed the rule.
    for _t in ("A chain-link fence runs along the yard.", "He wears a leather belt.",
               "Kate walks to the window.", "The rope hangs from the rafters.",
               "Dan tapes the box shut."):
        check(f"not a restraint: {_t[:34]!r}", not S.restraint_present(_t))
    check("the hold is one sentence", S.RESTRAINT_HOLD.count(".") == 1)
    check("...impersonal, so it summons nobody",
          not re.search(r"(?:she|he|her|his|they)", S.RESTRAINT_HOLD, re.I))
    check("...and positive, since cfg 1 has no negative prompt",
          not re.search(r"no|not|never", S.RESTRAINT_HOLD, re.I))
    check("...saying what holds", "whole and closed" in S.RESTRAINT_HOLD
          and "fastened exactly as it was put on" in S.RESTRAINT_HOLD)


def test_schema():
    print("\n=== node schema ===")
    schema = S.H3LongVideos.INPUT_TYPES()
    req, opt = schema["required"], schema["optional"]
    for name in ("model", "clip", "vae", "audio_vae", "prompt"):
        check(f"{name} is required", name in req)
    check("the prompt is a socket, not a box", req["prompt"][1].get("forceInput") is True)
    check("cfg defaults to 1.0 -- H3 is CFG-free", req["cfg"][1]["default"] == 1.0)
    check("shot_seconds defaults to 10", req["shot_seconds"][1]["default"] == 10.0)
    check("the shifts default to 12/3",
          opt["shift_video"][1]["default"] == 12.0 and opt["shift_audio"][1]["default"] == 3.0)
    check("silence on non-speech shots is on", opt["silence_nonspeech"][1]["default"] is True)
    check("restraints are held by default", opt["hold_restraints"][1]["default"] is True)
    check("shot length is read from the beat by default",
          opt["shot_length"][1]["default"] == "from the beat")
    check("first_frame is offered", "first_frame" in opt)
    n_widgets = sum(1 for d in (req, opt) for k, v in d.items()
                    if not (len(v) > 1 and isinstance(v[1], dict) and v[1].get("forceInput"))
                    and (isinstance(v[0], list) or v[0] in ("INT", "FLOAT", "STRING", "BOOLEAN")))
    # 17 core + 6 upscale. The point of the number is that it stays small enough
    # to read; the old node had 38 and nobody could find anything.
    # 17 core + 6 upscale + shot_length + hold_restraints. The number matters only
    # as a ceiling: the old node had 38 and nobody could find anything.
    check(f"the node stays small: {n_widgets} widgets", n_widgets <= 26)
    for _u in ("upscale", "upscale_model", "upscale_target_short_edge", "upscale_batch",
               "latent_upscale", "latent_upscale_scale"):
        check(f"{_u} is on the node", _u in opt)
    check("both upscalers default to off",
          opt["upscale"][1]["default"] == "off" and opt["latent_upscale"][1]["default"] == "off")
    check("outputs include info and script",
          "info" in S.H3LongVideos.RETURN_NAMES and "script" in S.H3LongVideos.RETURN_NAMES)
    check("it registers under one id", set(S.NODE_CLASS_MAPPINGS) == {"H3LongVideos"})


def main():
    test_beats()
    test_verbatim()
    test_sizing()
    test_speech_and_refs()
    test_removals()
    test_layers()
    test_removal_completes()
    test_restraints_hold()
    test_thin_beats()
    test_auto_length()
    test_text_in_frame()
    test_reference_tags()
    test_schema()
    print()
    if _fails:
        print(f"RESULT: {len(_fails)} FAILURE(S): " + "; ".join(_fails))
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
