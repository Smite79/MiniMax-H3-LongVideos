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
          S.extract_removals("x\nremove: coat, shirt")[1] == ["coat", "shirt"])
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
    # A sentence's full stop lives on its LAST fragment. Removing the garment that
    # is listed last took the full stop with it and ran the sentence into the next
    # one -- "blue eyes Wrists cuffed behind back." In a strip sequence the item
    # coming off is usually the last one listed, so this fired on most removals.
    _end = "Maya: 27, blue eyes, grey scarf, black shorts. Wrists cuffed behind back."
    check("removing the last-listed item keeps the full stop",
          S.scrub_removed(_end, ["shorts"])
          == "Maya: 27, blue eyes, grey scarf. Wrists cuffed behind back.")
    check("...and so does removing the last two",
          S.scrub_removed(_end, ["scarf", "shorts"])
          == "Maya: 27, blue eyes. Wrists cuffed behind back.")
    check("...in either order",
          S.scrub_removed(_end, ["shorts", "scarf"])
          == S.scrub_removed(_end, ["scarf", "shorts"]))
    check("a sentence that keeps its last fragment is untouched",
          S.scrub_removed(_end, ["scarf"])
          == "Maya: 27, blue eyes, black shorts. Wrists cuffed behind back.")
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
    _long = ("A basement. Kate is 20, blonde, pale blue cotton shirt, long grey wool scarf, "
             "heavy black waxed canvas jacket, brown leather boots.")
    _r1 = S.scrub_removed(_long, ["jacket"])
    check("a long list entry is removed whole",
          "jacket" not in _r1 and "waxed" not in _r1 and "canvas" not in _r1)
    check("...and its neighbours are intact",
          "pale blue cotton shirt" in _r1 and "long grey wool scarf" in _r1
          and "brown leather boots" in _r1)
    _r2 = S.scrub_removed(_long, ["scarf"])
    check("no orphan adjective is left behind",
          "wool" not in _r2 and ", long," not in _r2)
    _r3 = S.scrub_removed(_long, ["scarf", "jacket", "boots"])
    check("three removals leave only what is still worn",
          _r3 == "A basement. Kate is 20, blonde, pale blue cotton shirt.")
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


def test_inferred_removals():
    print("\n=== a removal read out of the beat's own prose ===")
    # The scene a real run hit: two garments in one entry, and a restraint whose
    # entry mentions a body part. The first version of this accepted ANY word the
    # beat shared with the scene, so "cuts the tight top away from her back" took
    # off "tight", "her" and "back" -- and scrubbing "back" deleted the entry that
    # described the handcuffs, which took the hardware out of the prompt entirely.
    sc = ("A bare basement. Kate, 20, in a tight white crop top and black shorts, "
          "her wrists handcuffed behind her back.")
    check("the garment is read, and only the garment",
          S.infer_removals("Dan cuts the tight top away from her back.", sc) == ["top"])
    check("...not a modifier inside an entry",
          S.infer_removals("Dan pulls the tight crop top off.", sc) == ["top"])
    check("...not a pronoun",
          "her" not in S.infer_removals("Dan cuts off her shorts.", sc))
    check("...and the plain case still works",
          S.infer_removals("Dan cuts off her shorts.", sc) == ["shorts"])
    # Hardware is cleared by an explicit remove: and by nothing else. A beat that
    # cuts a rope must not silently unlock the cuffs.
    for _b in ("Dan cuts the rope from her wrists.", "Dan takes off her handcuffs."):
        check(f"no inferred restraint: {_b[:30]!r}", S.infer_removals(_b, sc) == [])
    check("an ordinary beat infers nothing",
          S.infer_removals("Kate lies still.", sc) == [])
    check("a beat cannot remove what is not worn",
          S.infer_removals("Dan cuts off her cape.", sc) == [])
    # And the scrub half: the restraint's entry survives a removal that merely
    # shares a word with it, because hardware absent from the text renders absent.
    for _t in (["top"], ["shorts"], ["top", "shorts"]):
        check(f"the cuffs survive removing {_t}",
              "handcuffed" in S.scrub_removed(sc, _t))
    check("...and an explicit remove: still clears them",
          "handcuffed" not in S.scrub_removed(sc, ["handcuffed"]))
    # The and-joined entry keeps its innocent half.
    _s = S.scrub_removed(sc, ["top"])
    check("the neighbour in the same entry stays", "black shorts" in _s)
    check("...and reads cleanly", ",black" not in _s and "  " not in _s)


def test_character_sheet():
    print("\n=== a character sheet is not a beat ===")
    # Reported: paragraph 2 of a script rendered as a whole shot of static
    # description. Worse than the wasted shot -- the wardrobe then lived in ONE
    # shot, so every later shot described no clothing, the model invented it, and
    # a removal had nothing to scrub because the garment was never in the scene.
    sheet = ("Maya: 27, silver hair, grey shorts, red jacket\n"
             "Jon: 34, navy overalls")
    check("a sheet is recognised", S.is_character_sheet(sheet))
    check("...with one person too", S.is_character_sheet("Maya: 27, red jacket"))
    check("...and an inner capital is fine",
          S.is_character_sheet("McKenna: 22, grey coat"))
    check("...and a participle that introduces attributes",
          S.is_character_sheet("Maya: wearing a red coat"))
    # A beat stages something; a line of dialogue stages something.
    for _p in ("Maya walks in.", 'Jon: "Hello."', "Maya: 27, red jacket\nJon walks in.",
               "A basement. Maya is 27.", "remove: jacket", ""):
        check(f"not a sheet: {_p[:30]!r}", not S.is_character_sheet(_p))
    # A LABELLED action is still an action. Getting this wrong is expensive in one
    # direction only: a sheet mistaken for a beat costs one visible shot, while a
    # beat mistaken for a sheet never renders AND has its words stamped onto every
    # other shot -- which reads as beats being absorbed into other beats.
    for _p in ("McKenna: thrashes in her restraints, trying to get free.",
               "Dan: walks in holding a pair of scissors.",
               "Camera: pushes in slowly on her face.",
               "Maya: turns to face him.",
               "Dan: is standing by the door."):
        check(f"a labelled action is a beat: {_p[:38]!r}", not S.is_character_sheet(_p))
    beats, pulled = S.pull_character_sheets(["Maya walks in.", sheet, "Jon follows."])
    check("the sheet leaves the beat list", beats == ["Maya walks in.", "Jon follows."])
    check("...and is kept", pulled == sheet)
    check("a script with no sheet is untouched",
          S.pull_character_sheets(["One.", "Two."]) == (["One.", "Two."], ""))
    # Reading order, and one string so a removal scrubs all of it.
    built = S.build_scene("Wide lens, night.", "A basement.", "Maya: red jacket", "Jon: 34")
    check("the anchor comes first", built.startswith("Wide lens, night."))
    check("...then the scene", built.index("A basement.") < built.index("Maya:"))
    check("...then the people", built.index("Maya:") < built.index("Jon:"))
    check("empty channels are skipped", S.build_scene("", "A basement.", "", "")
          == "A basement.")
    # The point of folding it in: a removal can now reach the wardrobe.
    check("a removal scrubs the sheet",
          "red jacket" not in S.scrub_removed(built, ["jacket"]))
    check("...and leaves the rest standing",
          "Wide lens, night." in S.scrub_removed(built, ["jacket"]))


def test_character_guard():
    print("\n=== only the people a beat involves are described ===")
    # Reported: the other character turning up in scenes they are not in. The sheet
    # has to be in every shot for clothing to hold -- but describing EVERYONE in
    # every shot puts everyone in every shot, because a described person is a person
    # the model draws.
    sheet = "Maya: 27, grey scarf, black jacket\nJon: 34, navy overalls"
    keep, who = S.sheet_for_beat(sheet, "Maya lies still on the floor.")
    check("a beat naming one person keeps one", who == ["Maya"])
    check("...and drops the other's line", "Jon" not in keep and "Maya" in keep)
    keep, who = S.sheet_for_beat(sheet, "Jon walks out and shuts the door.", ["Maya"])
    check("a beat naming the other keeps the other", who == ["Jon"])
    check("...and lets the first go", "Maya" not in keep)
    # A PRONOUN names someone too. Dropping Maya from "Jon takes her jacket off"
    # would leave the garment being removed undescribed in the shot removing it.
    keep, who = S.sheet_for_beat(sheet, "Jon takes her jacket off.", ["Maya"])
    check("a pronoun keeps whoever the last beat kept", sorted(who) == ["Jon", "Maya"])
    check("...so the garment coming off is still described", "grey scarf" in keep)
    # A beat naming nobody keeps the last beat's people rather than emptying the frame.
    keep, who = S.sheet_for_beat(sheet, "The camera pushes in.", ["Maya"])
    check("a beat naming nobody holds the last cast", who == ["Maya"])
    check("with no history it keeps everyone",
          sorted(S.sheet_for_beat(sheet, "The camera pushes in.")[1]) == ["Jon", "Maya"])
    # An unlabelled line belongs to the scene, not to a person, and never drops.
    keep, _ = S.sheet_for_beat("The room is cold.\nMaya: 27, grey scarf",
                               "Jon walks in.", ["Jon"])
    check("an unlabelled line is kept for everyone", "The room is cold." in keep)


def test_layers_from_prose():
    print("\n=== a layer stays out of the text until it is uncovered ===")
    # Reported: the under layer showing through the top one. A sheet lists every
    # layer at once, which says all of them are on show; nothing says which is
    # hidden, so the model draws the under layer through the one over it.
    sc = "Maya: 27, grey wool scarf, black quilted jacket, brown boots."
    check("what a removal exposes is read",
          S.exposed_by("Jon takes her jacket off to expose the scarf.", sc) == ["scarf"])
    check("...with 'exposing' too",
          S.exposed_by("Jon cuts off the jacket, exposing the scarf.", sc) == ["scarf"])
    check("...and nothing when nothing is exposed",
          S.exposed_by("Jon walks in.", sc) == [])
    check("...ignoring what is not worn",
          S.exposed_by("Jon takes her jacket off to expose the wall.", sc) == [])
    covers = S.infer_layers(["Jon takes her jacket off to expose the scarf."], sc)
    check("the script says what covers what", covers == {"scarf": "jacket"})
    # Hidden while covered, described again the moment the cover goes.
    check("covered while the jacket is on", S.hidden_layers(covers, []) == ["scarf"])
    check("...visible once it comes off", S.hidden_layers(covers, ["jacket"]) == [])
    check("...and not resurrected after it is removed itself",
          S.hidden_layers(covers, ["scarf"]) == [])
    check("no layers read, nothing hidden", S.hidden_layers({}, []) == [])


def test_opening_pose():
    print("\n=== shot 1 has no keyframe ===")
    # Shot 1 is the only shot with no previous frame to continue from, so its
    # opening pose comes from the text and nothing else. This reports; it does not
    # reorder the text, because what you write is what the shot gets.
    sc = ("A basement. Maya: 27, grey scarf, black jacket. Wrists cuffed behind back. "
          "She stays lying on her side on the floor.")
    note = S.posture_note(sc, False)
    check("a posture sentence is found", "opening pose" in note)
    check("...and its position reported", "4 of 4" in note)
    check("...pointing at the mechanism that pins it", "first_frame" in note)
    # first_frame pins the WHOLE frame. Telling someone with an identity portrait to
    # wire it there makes shot 1 a portrait -- worse than the pose they wanted.
    check("...saying it must be a composed frame", "composed frame" in note)
    check("...and where a portrait actually goes", "ref_image_1" in note)
    check("nothing said when a first_frame is wired", S.posture_note(sc, True) == "")
    check("...or when no posture is described",
          S.posture_note("A basement. Maya walks to the window.", False) == "")
    check("...or with no scene at all", S.posture_note("", False) == "")
    # ONE aug covers every visual conditioning row. At 0.999 a reference is handed
    # over near-clean, and near-clean means "reproduce this" -- framing included. A
    # portrait reference therefore pulls the shot towards portrait framing.
    rn = S.reference_note(1, 0.999, False)
    check("a near-clean reference is explained", "reproduce them" in rn)
    check("...naming framing as what carries over", "FRAMING included" in rn)
    check("...and that shot 1 has no anchor to protect", "no keyframe" in rn)
    check("with a first_frame, shot 1 does have one",
          "no keyframe" not in S.reference_note(1, 0.999, True))
    check("no references, nothing to say", S.reference_note(0, 0.999, False) == "")
    check("already softened, nothing to say", S.reference_note(1, 0.90, False) == "")


def test_removal_needs_a_particle():
    print("\n=== an ordinary action is not a removal ===")
    # Reported: a described garment rendering plain and pale. The verb pattern
    # fired on a BARE verb, so "pulls her crop top down" read as a removal and
    # scrubbed the entry -- leaving the garment still worn but undescribed, and
    # an undescribed garment is one the model invents. Colour and material are
    # lost with the entry, so the invented one comes back plain.
    sc = ("A bare basement. Kate, 20, blonde, black shiny latex crop top, "
          "white cotton shorts, brown leather boots, a grey coat.")
    for _b in ("Dan cuts off her shorts.", "Dan pulls off her boots.",
               "Dan pulls down her shorts.", "Dan removes her coat.",
               "Dan unzips her coat.", "Dan strips off her coat.",
               "Dan throws her coat away."):
        check(f"a removal still fires: {_b[:32]!r}", S.infer_removals(_b, sc))
    # The particle's POSITION settles the ambiguous case: straight after the verb
    # it removes, trailing after the object only "off" and "away" do.
    check("'takes her coat off' removes it",
          S.infer_removals("Dan takes her coat off.", sc) == ["coat"])
    check("...but 'pulls her crop top down' only adjusts it",
          S.infer_removals("Dan pulls her crop top down.", sc) == [])
    for _b in ("Dan cuts the rope from her wrists.", "Dan takes her hand.",
               "Dan pulls her closer.", "Dan throws the bag on the floor.",
               "Dan cuts the tape on the box.", "Dan takes a step back.",
               "Kate pulls at her sleeve.", "Dan straightens her coat."):
        check(f"not a removal: {_b[:34]!r}", S.infer_removals(_b, sc) == [])
    # The point of all of it: what is still worn keeps its full description.
    kept = S.scrub_removed(sc, S.infer_removals("Dan pulls her crop top down.", sc))
    check("the garment keeps its colour and material",
          "black shiny latex crop top" in kept)


def test_removal_completes():
    print("\n=== a removal has to finish inside its shot ===")
    # Scrubbing stops a garment being DESCRIBED. It does not tell the model to
    # finish taking it off -- and the last frame is the next shot's keyframe, so
    # a cut still in progress hands on a garment still half worn. The next beat
    # has moved on and never contradicts the picture, so it stays.
    one = S.off_by_last_frame(["coat"])
    check("the removing shot is told to finish it", "by the last frame" in one)
    check("...and that nothing is left on the body", "no longer on the body" in one)
    check("...and where it ends up", "out of frame" in one)
    check("a plural garment agrees",
          "boots come off" in S.off_by_last_frame(["boots"])
          and " are away" in S.off_by_last_frame(["boots"]))
    check("a singular one does too",
          "coat comes off" in one and " is away" in one)
    # The sentence is capitalised, so the first "the" is "The".
    check("two items are joined", "The coat and the boots come off"
          in S.off_by_last_frame(["coat", "boots"]))
    check("no removal, no sentence", S.off_by_last_frame([]) == "")
    # Saying what comes off does not say where to STOP. An action with time left
    # runs on to whatever is next: a hand that finishes one garment starts on the
    # next one, or on the body under it.
    _b = S.off_by_last_frame(["scarf"])
    check("the action is bounded", "Everything else worn stays exactly as it is" in _b)
    check("...covering hardware as well", "still fastened" in _b)
    # It bounds what is WORN, not the body. "Everything else on the body stays exactly
    # as it is for the whole shot" reads as an instruction to hold still, and enough
    # of those render a shot where nothing happens.
    check("...without telling the body to hold still",
          not re.search(r"\bbody stays\b|\bfor the whole shot\b|\bmotionless\b", _b, re.I))
    check("...naming no other garment", "jumper" not in _b and "coat" not in _b)
    # The BOUND sentence carries no negation: at cfg 1 the negative prompt is
    # never evaluated, so a negation in the positive only names what it forbids.
    # ("no longer on the body" belongs to the removal half, and is the wording the
    # previous node proved safe in the removing shot.)
    _bound = _b.split("dropped out of frame.")[1]
    check("...and the bound is stated positively",
          not re.search(r"\bno\b|\bnot\b|\bnever\b|\bnothing\b", _bound, re.I))
    check("it reads as a sentence", one.strip().startswith("The coat"))
    # Said ONCE. Naming the garment on a later shot is a presence cue, and that
    # phrasing put garments back on in the previous version of this node.
    scene = "A basement. Kate is 20, blonde, grey wool coat, black jumper."
    beats = ["Dan pulls off her coat.\nremove: coat",
             "Dan pulls off her jumper.\nremove: jumper",
             "Kate looks up at him."]
    gone, lines = [], []
    for b in beats:
        body, toks, _ = S.extract_directives(b)
        gone.extend(t for t in toks if t not in gone)
        lines.append(f"{S.scrub_removed(scene, gone)} {body}{S.off_by_last_frame(toks)}")
    check("shot 1 orders the coat off", "coat comes off during this shot" in lines[0])
    check("...and shot 2 never mentions it again", "coat" not in lines[1])
    check("...nor shot 3", "coat" not in lines[2] and "jumper" not in lines[2])
    check("the scene loses each garment as it goes",
          "jumper" in lines[0] and "jumper" not in lines[2])


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
                                    "her grey vest is now visible"]
    live = [a for a in shown if not S.names_any(a, gone)]
    check("a removed layer stops being described", live == ["her grey vest is now visible"])
    check("...and one that was not removed stays", S.names_any("a red coat", ["coat"]))


def test_thin_beats():
    print("\n=== a shot longer than its beat ===")
    # A shot that outlasts its action leaves the model seconds it was told
    # nothing about, and the cheapest way to fill them is to CARRY ON: an action
    # repeats itself on whatever is nearest.
    one = "Dan pulls off her coat and throws it away."
    two = ("Dan pulls off her coat and throws it away, then sets the hanger down "
           "and steps back.")
    # Two clauses at 2.2s each plus a small settle. It used to be 7s, which gave a
    # two-action beat well over twice the screen time its actions needed -- and the
    # surplus is spent performing them more slowly, not on anything new.
    check("a two-clause beat asks for about 5s", 4.5 <= S.beat_seconds(one) <= 5.5)
    check("adding what happens next asks for more", S.beat_seconds(two) > S.beat_seconds(one))
    check("directive lines do not count as content",
          S.beat_seconds(one) == S.beat_seconds(one + "\nremove: coat"))
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
    one = "Dan pulls off her coat and throws it away."
    two = ("Dan pulls off her coat and throws it away, then sets the hanger down "
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
    # A clamp applied to a body is hardware and has to stay on. Clamped to a bench it
    # is a tool -- only the context separates them, so it needs a fastening verb or a
    # body part like any other ambiguous item.
    for _t in ("Jon puts a steel clamp on her arm.", "Jon clamps a ring to her wrist.",
               "A steel clamp is clipped to her belt.",
               "Steel clamps fastened to her ankles."):
        check(f"a clamp on a body: {_t[:36]!r}", S.restraint_present(_t))
    for _t in ("A steel clamp holds the workpiece on the bench.",
               "Jon clamps the board to the workbench.",
               "Jon clips the coupon out of the paper."):
        check(f"a clamp on a thing: {_t[:36]!r}", not S.restraint_present(_t))
    # "clamps" is a noun here and a verb there, and a word in BOTH lists satisfies
    # both halves of the rule by itself -- which is how a chain-link fence used to
    # arm this, and how "clamps the board to the workbench" did.
    check("no word sits in both the noun list and the verb list",
          not (S._RESTRAINT_MAYBE.search("clamps") and S._BINDING_VERB.search("clamps")))
    check("...the same way tape is handled",
          not (S._RESTRAINT_MAYBE.search("tapes") and S._BINDING_VERB.search("tapes")))
    # A clamp is rigid, but it is not a chain: the chain clause speaks of links and
    # the run between fastenings, which is nonsense said of a clamp.
    check("a clamp does not earn the chain clause",
          not S.rigid_hardware("Jon puts a steel clamp on her arm."))
    check("...while a chain still does", S.rigid_hardware("a chain around her waist"))
    check("the hold is one sentence", S.RESTRAINT_HOLD.count(".") == 1)
    check("...impersonal, so it summons nobody",
          not re.search(r"\b(?:she|he|her|his|they)\b", S.RESTRAINT_HOLD, re.I))
    check("...and positive, since cfg 1 has no negative prompt",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", S.RESTRAINT_HOLD, re.I))
    check("...saying what holds", "whole and closed" in S.RESTRAINT_HOLD
          and "fastened exactly as it was put on" in S.RESTRAINT_HOLD)


def test_hardware_has_somewhere_to_go():
    print("\n=== hardware named with nowhere to sit ===")
    # Reported: a collar and leash being shown, and the collar ending up on top of
    # the head. A collar with no neck beside it is a band with no place to be, and a
    # model handed a band-shaped object and no anatomy puts it where bands sit most
    # often in its training data. Naming the part invents nothing: it is what the
    # object IS. It happens whether the item is being fastened or only held up.
    got = S.unanchored_hardware("Jon shows her a collar and leash.")
    check("a collar is put at the neck", "a collar closes around the neck" in got)
    check("...and the leash on the collar",
          any("leash clips to the collar" in g for g in got))
    check("a gag goes in the mouth",
          S.unanchored_hardware("Jon holds up a gag.") == ["a gag sits in the mouth"])
    check("handcuffs go on the wrists",
          "handcuffs close around the wrists"
          in S.unanchored_hardware("Jon shows her handcuffs."))
    # What you wrote wins. If the text already says where it goes, nothing is added.
    for _t in ("Jon buckles the collar around her neck.",
               "Jon fits the blindfold over her eyes.",
               "Jon clips the leash to the collar at her neck.",
               "Wrists handcuffed behind back."):
        check(f"already placed: {_t[:36]!r}", S.unanchored_hardware(_t) == [])
    check("no hardware, nothing to place",
          S.unanchored_hardware("Maya walks to the window.") == [])
    # A chastity belt is not a waistband. The generic belt phrase said nothing about
    # where it fastens, so the lock went where locks usually go on a strap -- behind.
    _cb = S.unanchored_hardware("Jon shows her a chastity belt.")
    check("a chastity belt gets its own placement", len(_cb) == 1)
    check("...with the lock at the front", "locks at the front" in _cb[0])
    check("...and the shield between the legs", "between the legs" in _cb[0])
    check("the plugged variant is the same belt",
          S.unanchored_hardware("Jon shows her a plugged chastity belt.") == _cb)
    # The plain belt entry must not ALSO match inside "chastity belt", or the shot
    # states two different placements for one object.
    check("...and the plain belt phrase does not double up",
          not any("closes around the waist" in p for p in _cb))
    check("an ordinary belt still gets the ordinary phrase",
          S.unanchored_hardware("Jon shows her a leather belt.")
          == ["a belt closes around the waist and hips"])
    check("naming the position yourself wins",
          S.unanchored_hardware("Jon locks the chastity belt at the front, over her hips.")
          == [])
    # The sentence itself.
    cl = S.anchor_clause(["a collar closes around the neck"])
    check("the clause reads as one sentence", cl.count(".") == 1)
    check("...and is impersonal",
          not re.search(r"\b(?:she|he|her|his|they)\b", cl, re.I))
    check("nothing to place, no sentence", S.anchor_clause([]) == "")


def test_pace():
    print("\n=== a shot longer than its action is filled by slowing it down ===")
    # Reported: the movement looks slow. A video model given more time than the
    # action needs does not invent more action -- it performs the same one more
    # slowly. Measured: "Maya walks to the window" is a few steps, under two seconds
    # of real movement, and the old constants gave it a 4.5s shot.
    _ceil = S.align_frame_count(10 * S.H3_FPS)
    one = "Maya walks to the window."
    two = "Jon walks in and takes her jacket off."
    check("a one-action beat no longer asks for four and a half seconds",
          S.beat_seconds(one) <= 3.2)
    check("...and a two-action beat is under six", S.beat_seconds(two) <= 5.5)
    # The base was the larger error: a chained shot opens mid-scene, continuing from
    # the previous frame, so there is nothing to set up.
    check("the settle allowance is small", S.BEAT_BASE_SEC < 1.0)
    # pace scales the whole estimate.
    slow = S.plan_lengths([two], _ceil, True, 1.5)[0][0]
    norm = S.plan_lengths([two], _ceil, True, 1.0)[0][0]
    fast = S.plan_lengths([two], _ceil, True, 0.6)[0][0]
    check("a lower pace shortens the shot", fast < norm)
    check("...and a higher one lengthens it", slow > norm)
    check("the floor still holds at one action's worth",
          S.plan_lengths([one], _ceil, True, 0.1)[0][0] == S.MIN_AUTO_FRAMES)
    check("the ceiling still holds", slow <= _ceil)
    check("a pace of 0 does not divide by zero or empty the shot",
          S.plan_lengths([two], _ceil, True, 0)[0][0] >= S.MIN_AUTO_FRAMES)
    check("'fixed' ignores pace entirely",
          S.plan_lengths([one, two], _ceil, False, 0.5)[0] == [_ceil, _ceil])


def test_av_grid_alignment():
    print("\n=== the audio grid does not land on the video grid ===")
    # H3's audio latent runs at 40/s against 24 fps video, so a shot's audio latent
    # count is round(frames / 24 * 40) -- exact only when the frame count divides by
    # 3. Every other length on the 17k+5 grid is up to 8.3 ms out, and with shots of
    # equal length the error carries the same sign every time and ACCUMULATES.
    worst, exact = 0.0, 0
    for k in range(0, 22):
        fc = 17 * k + 5
        if fc > S.MAX_FRAMES:
            break
        _f, _lt, at = S.temporal_shape(fc)
        drift = abs(at / S.AUDIO_LATENT_FPS - fc / S.H3_FPS) * 1000
        worst = max(worst, drift)
        exact += drift < 1e-9
    check("most grid lengths do not land exactly", exact < 8)
    check("...and the ones that miss, miss by ~8.3 ms", 8.0 < worst < 8.7)
    check("a length divisible by 3 is exact",
          abs(S.temporal_shape(39)[2] / S.AUDIO_LATENT_FPS - 39 / S.H3_FPS) < 1e-9)
    # The fix is per shot, not once at the end: correcting only the total would leave
    # every interior cut misaligned even with the final duration right.
    for fc in (73, 124, 226):
        want = int(round(fc * 44100 / S.H3_FPS))
        check(f"{fc}f wants {want} samples at 44.1k", want > 0)
    # temporal_shape must key the audio to 24 fps whatever fps is passed, or the
    # sound is stretched against the picture.
    check("the audio grid ignores a different fps",
          S.temporal_shape(73, 30) == S.temporal_shape(73, 24))


def test_chain_is_rigid():
    print("\n=== steel does not behave like rope ===")
    # A model with no reason to think otherwise draws a chain as a soft cord: it
    # sags, stretches to wherever a limb is going, and allows movement the hardware
    # does not allow. The restraint hold says the metal stays WHOLE -- it says
    # nothing about how it behaves while whole.
    for _t in ("Jon locks a chain around her waist.", "padlocked at the back",
               "Wrists handcuffed behind back.", "ankles shackled together",
               "steel cuffs", "a spreader bar", "hogcuffed on the floor"):
        check(f"rigid hardware: {_t[:32]!r}", S.rigid_hardware(_t))
    # Rope, tape and straps DO flex -- claiming they hold a straight line is wrong.
    for _t in ("a rope around her wrists", "her mouth taped shut",
               "a leather strap", "Maya lies still."):
        check(f"not rigid: {_t[:32]!r}", not S.rigid_hardware(_t))
    check("the clause is one sentence", S.CHAIN_HOLD.count(".") == 1)
    check("...it keeps the links the same size", "links keep their size" in S.CHAIN_HOLD)
    check("...holds the run straight", "straight and taut" in S.CHAIN_HOLD)
    check("...impersonal and positive",
          not re.search(r"\b(?:she|he|her|his|they|no|not|never)\b", S.CHAIN_HOLD, re.I))
    # It constrains the METAL. An earlier wording had the body reaching "only as far
    # as the metal allows before it stops" -- read plainly that is an instruction to
    # stop moving, and the holds stacked up to 64% of a shot whose beat was 11%.
    for _c, _n in ((S.CHAIN_HOLD, "chain"), (S.RESTRAINT_HOLD, "restraint"),
                   (S.TURN_HOLD, "turn"), (S.FALL_HOLD, "fall")):
        check(f"the {_n} clause does not tell the body to stop",
              not re.search(r"\bbefore it stops\b|\bthe body stays\b|\bholds? still\b|"
                            r"\bmotionless\b|\bdoes not move\b|\bstays put\b", _c, re.I))
    # It subsumes the restraint hold rather than joining it: two clauses saying "whole
    # and closed" is twice the stasis for one guarantee.
    check("the chain clause carries the restraint guarantee itself",
          "whole and closed" in S.CHAIN_HOLD and "fastened exactly as it was put on"
          in S.CHAIN_HOLD)
    # A position hardware was locked to enforce. Saying the metal keeps its shape is
    # not enough: a chain that keeps its shape can still be drawn with slack, and
    # slack is room to stand up out of a squat the chain was locked to hold.
    for _t in ("forcing her into a squat", "chained kneeling on the floor",
               "hogcuffed on the floor", "bent over the table", "spread-eagled",
               "locked crouching", "on her knees, chained to the wall"):
        check(f"a forced position: {_t[:34]!r}", S.forced_pose(_t))
    for _t in ("Maya walks to the window.", "Jon locks a chain around her waist.",
               "Maya lies still."):
        check(f"no position forced: {_t[:34]!r}", not S.forced_pose(_t))
    check("the pose clause says the metal is at full length",
          "drawn out to its full length" in S.CHAIN_POSE_HOLD)
    check("...that the position keeps", "the position that keeps" in S.CHAIN_POSE_HOLD)
    check("...and carries the restraint guarantee too",
          "whole and closed" in S.CHAIN_POSE_HOLD)
    # It must NOT buy the position by freezing the body -- straining against it is
    # exactly what should happen, and this is the clause most at risk of stasis.
    check("...while leaving the body free to act",
          "strains and pulls against it" in S.CHAIN_POSE_HOLD)
    check("...and telling it to hold still nowhere",
          not re.search(r"\bstill\b|\bmotionless\b|\bdoes not move\b|\bbefore it stops\b",
                        S.CHAIN_POSE_HOLD, re.I))
    check("...positively phrased", not re.search(r"\b(?:no|not|never|without)\b",
                                                 S.CHAIN_POSE_HOLD, re.I))


def test_saved_defaults():
    print("\n=== a preference outlives an edit to this file ===")
    # Every widget added changes INPUT_TYPES, and a node added afresh comes up with
    # the built-in defaults -- so a setting has to be put back by hand after every
    # edit. defaults.json is read at load and replaces them.
    def _schema():
        return {"required": {"steps": ("INT", {"default": 8}),
                             "sampler_name": (["euler", "res_multistep"], {}),
                             "model": ("MODEL",)},
                "optional": {"upscale": (["off", "lanczos"], {"default": "off"})}}
    sch = S.apply_saved_defaults(_schema(), {"steps": 6, "upscale": "lanczos"})
    check("a saved default replaces the built-in",
          sch["required"]["steps"][1]["default"] == 6)
    check("...in the optional group too",
          sch["optional"]["upscale"][1]["default"] == "lanczos")
    # It has to survive a widget being renamed or dropped, and a socket has no default.
    sch = S.apply_saved_defaults(_schema(), {"gone_widget": 1, "model": "x"})
    check("an unknown key changes nothing", sch["required"]["steps"][1]["default"] == 8)
    check("...and a socket is left alone", len(sch["required"]["model"]) == 1)
    # A combo can only default to one of its own choices: an upscale model that is no
    # longer installed must not become the default.
    sch = S.apply_saved_defaults(_schema(), {"sampler_name": "not_installed"})
    check("a combo rejects a choice it does not have",
          "default" not in sch["required"]["sampler_name"][1])
    sch = S.apply_saved_defaults(_schema(), {"sampler_name": "res_multistep"})
    check("...and accepts one it does",
          sch["required"]["sampler_name"][1]["default"] == "res_multistep")
    check("no saved file, no change",
          S.apply_saved_defaults(_schema(), {})["required"]["steps"][1]["default"] == 8)
    check("a malformed file is not fatal", isinstance(S.saved_defaults(), dict))
    # save_defaults captures what the node currently HAS, which beats transcribing
    # widget values by hand.
    vals = {"steps": 6, "scheduler": "beta", "seed": 42, "character_guard": True,
            "save_defaults": True, "model": object(), "prompt": "x", "clip": object()}
    keep = {k: v for k, v in vals.items() if k in S._SAVEABLE and v is not None}
    check("widgets are captured", keep["steps"] == 6 and keep["scheduler"] == "beta")
    # A saved True would arm every fresh node to re-save on its next run.
    check("save_defaults never saves itself", "save_defaults" not in keep)
    check("sockets are not saved",
          not any(k in keep for k in ("model", "clip", "prompt")))


def test_falling_bound():
    print("\n=== a bound body goes down without catching itself ===")
    # A falling body puts its hands out. With the hands fastened the model has to
    # resolve that, and freeing them is cheaper than landing on a shoulder -- so
    # the cuffs open or the chain snaps on the way down. The restraint hold does
    # not cover it: it speaks about the hardware, not about the fall.
    for _t in ("She falls forward onto the floor.", "Kate collapses.",
               "She loses her balance and goes down.", "Kate topples sideways.",
               "She stumbles and hits the floor.", "Kate slumps against the wall.",
               "Dan pushes her over.", "Dan knocks her down.",
               "Dan throws her to the ground.", "Dan pulls her down."):
        check(f"fall seen: {_t[:34]!r}", S.falls_in(_t))
    # Only a body going down counts. Light falls, gazes fall, and a dropped
    # object is not a dropped person.
    for _t in ("Kate walks to the window.", "She lies still.",
               "Dan drops the keys.", "Dan sets the crate down.",
               "Kate turns her head."):
        check(f"not a fall: {_t[:34]!r}", not S.falls_in(_t))
    check("the clause is one sentence", S.FALL_HOLD.count(".") == 1)
    check("...keeping the fastening through the fall",
          "fastened limbs stay fastened" in S.FALL_HOLD)
    check("...keeping the arms in the hold",
          "arms staying in the hold" in S.FALL_HOLD)
    # The hands are the whole problem, so the clause has to say what lands
    # INSTEAD of them. Saying "does not catch itself" would name catching, and
    # at cfg 1 there is no negative prompt to cancel it.
    check("...and naming what takes the landing",
          "shoulder, hip or side takes the landing" in S.FALL_HOLD)
    check("...impersonal and positive",
          not re.search(r"\b(?:she|he|her|his|they|no|not|never)\b",
                        S.FALL_HOLD, re.I))


def test_turning_around():
    print("\n=== turning shows a surface the keyframe never pinned ===")
    # The keyframe pins the FRONT. Once the body rotates, the model fills the
    # unseen side from its prior -- and its prior for an undescribed body is a
    # CLOTHED one. That is a removed garment coming back, often stacked wrongly,
    # and hardware on the far side re-invented as it rotates into view.
    for _t in ("She turned around.", "Kate turns to face him.",
               "She looks back over her shoulder.", "Kate rolls onto her side.",
               "The camera moves round to show her back."):
        check(f"turn seen: {_t[:32]!r}", S.turns_in(_t))
    for _t in ("Kate walks to the window.", "She lies still.", "Dan pulls off her coat."):
        check(f"no turn: {_t[:32]!r}", not S.turns_in(_t))
    # Being MOVED does the same damage as turning: the keyframe pinned one pose
    # from one side, and lifting or dragging someone puts the body where that
    # frame never showed it.
    _N = ["Kate", "Dan"]
    for _t in ("Dan lifts her onto the table.", "Dan drags her across the floor.",
               "Dan lays her down on the mat.", "Dan picks up Kate.",
               "Dan hauls her upright.", "Dan pulls her off the table."):
        check(f"moved body seen: {_t[:32]!r}", S.turns_in(_t, _N))
    # An object is not a body, a limb is not a body, and a garment is not a body.
    for _t in ("Dan lifts the crate.", "Dan picks up the scissors.",
               "Dan positions her legs behind her back.", "Dan grabs her ankles.",
               "Dan pulls her shorts off.", "Dan drops the keys."):
        check(f"not a moved body: {_t[:32]!r}", not S.turns_in(_t, _N))
    check("the clause covers what is worn", "all that is on it" in S.TURN_HOLD)
    check("...and what is fastened", "stays fastened and closed" in S.TURN_HOLD)
    check("...from every side", "front, side and behind" in S.TURN_HOLD)
    check("...as the view comes round", "as the view comes round" in S.TURN_HOLD)
    check("...naming no garment and no person",
          not re.search(r"\b(?:she|he|her|his|coat|top|shirt|jacket)\b",
                        S.TURN_HOLD, re.I))


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
    check("removals are read from the beat by default",
          opt["auto_remove"][1]["default"] is True)
    check("shot length is read from the beat by default",
          opt["shot_length"][1]["default"] == "from the beat")
    check("first_frame is offered", "first_frame" in opt)
    n_widgets = sum(1 for d in (req, opt) for k, v in d.items()
                    if not (len(v) > 1 and isinstance(v[1], dict) and v[1].get("forceInput"))
                    and (isinstance(v[0], list) or v[0] in ("INT", "FLOAT", "STRING", "BOOLEAN")))
    # 17 core + 6 upscale + shot_length, hold_restraints, restart_after_removal,
    # auto_remove + anchor, character_memory, character_guard. A ceiling, not a
    # target: the old node had 38 and nobody could find anything.
    check(f"the node stays small: {n_widgets} widgets", n_widgets <= 32)
    # Present, and in the order they were ADDED -- saved workflows restore widget
    # values by position with no names stored, so a widget inserted above an
    # existing one shifts every later value in every workflow already saved. New
    # ones go on the end, and stay in the order they arrived.
    for _w in ("anchor", "character_memory", "character_guard"):
        check(f"{_w} is offered", _w in opt)
    check("...and they sit at the end, in the order they were added",
          list(opt)[-5:] == ["anchor", "character_memory", "character_guard",
                             "save_defaults", "pace"])
    check("save_defaults is offered", "save_defaults" in opt)
    check("...and is off unless asked for", opt["save_defaults"][1]["default"] is False)
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
    test_inferred_removals()
    test_character_sheet()
    test_character_guard()
    test_layers_from_prose()
    test_opening_pose()
    test_removal_needs_a_particle()
    test_layers()
    test_removal_completes()
    test_restraints_hold()
    test_hardware_has_somewhere_to_go()
    test_pace()
    test_av_grid_alignment()
    test_chain_is_rigid()
    test_saved_defaults()
    test_falling_bound()
    test_turning_around()
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
