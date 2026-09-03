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
    # An OBJECT can carry a reference too -- "a silver locket <Picture 2>" is a
    # picture OF the locket -- and that tag has to come off with the object. Left
    # behind it keeps asserting the thing just removed, and a tag pointing at a
    # picture nothing accounts for is how a spare subject gets drawn.
    _obj = "Nora: <Picture 1>, 34, red hair, a silver locket <Picture 2>, green jacket."
    check("an object's tag leaves with the object",
          S.picture_tags(S.scrub_removed(_obj, ["locket"])) == [1])
    check("...and the person's stays", "<Picture 1>" in S.scrub_removed(_obj, ["locket"]))
    check("removing something else keeps both",
          S.picture_tags(S.scrub_removed(_obj, ["jacket"])) == [1, 2])
    check("removing both leaves only the person's",
          S.picture_tags(S.scrub_removed(_obj, ["locket", "jacket"])) == [1])
    # Ownership is decided by what stands immediately BEFORE the tag: a lowercase
    # noun means the object owns it, anything else -- a name, a colon, an age -- means
    # the person does. Erring towards the person, because losing an identity
    # reference costs the shot its face.
    for _t, _want in (("Nora: <Picture 1>, 34, she", ["1"]),
                      ("Nora <Picture 1> in a grey coat", ["1"]),
                      ("Kate is 20, <Picture 1> blonde crop top", ["1"]),
                      ("a silver locket <Picture 2>", []),
                      ("green canvas jacket <Picture 3>", [])):
        check(f"owner of {_t[:34]!r}", S.person_tags(_t) == _want)
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


def test_no_one_is_described_twice():
    print("\n=== one person, described once ===")
    # Reported: two heads in a shot. character_memory and a `Name:` paragraph in the
    # prompt are the same channel by two routes, and using both -- the natural thing
    # to do once the widget exists -- put the person in every shot TWICE. A model
    # told about one person twice renders two of them.
    merged, dupes = S.merge_sheets("Maya: 27, silver hair, grey coat.",
                                   "Maya: 27, silver hair, grey coat.")
    check("the second description is dropped", merged.count("Maya:") == 1)
    check("...and named", dupes == ["Maya"])
    # The earlier source wins, so character_memory overrides a sheet in the prompt.
    merged, _ = S.merge_sheets("Maya: 27, silver hair, grey coat.", "Maya: 27, red coat.")
    check("character_memory wins", "silver hair" in merged and "red coat" not in merged)
    # Different people are not duplicates.
    merged, dupes = S.merge_sheets("Maya: 27, grey coat", "Jon: 34, overalls")
    check("two people both survive", "Maya:" in merged and "Jon:" in merged)
    check("...and nothing is reported", dupes == [])
    check("one source alone is unchanged",
          S.merge_sheets("Maya: 27, grey coat")[0] == "Maya: 27, grey coat")
    check("nothing at all is empty", S.merge_sheets("", "") == ("", []))
    # An unlabelled line belongs to the scene, and is kept -- but not twice.
    merged, _ = S.merge_sheets("The room is cold.", "The room is cold.\nJon: 34")
    check("a repeated unlabelled line is said once", merged.count("The room is cold.") == 1)


def test_sheet_lines_are_terminated():
    print("\n=== a sheet line does not run into the beat ===")
    # The sheet is assembled ahead of the beat, so a line ending "grey coat" welds
    # onto it as "grey coat Maya lies still" -- and a name fused to the end of an
    # attribute list reads as one more item in the list, which is another person.
    check("a missing full stop is added",
          S.terminate_lines("Maya: 27, grey coat") == "Maya: 27, grey coat.")
    check("a trailing comma becomes one",
          S.terminate_lines("Maya: 27, grey coat,") == "Maya: 27, grey coat.")
    check("...and a semicolon", S.terminate_lines("Maya: 27;") == "Maya: 27.")
    check("an existing full stop is left alone",
          S.terminate_lines("Maya: 27, grey coat.") == "Maya: 27, grey coat.")
    check("...and so is a question mark", S.terminate_lines("Who?") == "Who?")
    check("every line gets one",
          S.terminate_lines("Maya: 27\nJon: 34") == "Maya: 27.\nJon: 34.")
    check("blank lines are dropped", S.terminate_lines("Maya: 27\n\n\nJon: 34")
          == "Maya: 27.\nJon: 34.")
    check("nothing in, nothing out", S.terminate_lines("") == "")


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
    _g = "Maya: 27, she, grey scarf, black jacket\nJon: 34, he, navy overalls"
    keep, who = S.sheet_for_beat(_g, "Jon takes her jacket off.", ["Maya"])
    check("a pronoun brings in who it refers to", sorted(who) == ["Jon", "Maya"])
    check("...so the garment coming off is still described", "grey scarf" in keep)
    # ...and only who it refers to. Adding the whole previous cast on ANY pronoun put
    # someone in a shot they were not in: "behind him" was read as evidence that
    # somebody else was present.
    check("a pronoun for the person already named brings in nobody else",
          S.sheet_for_beat(_g, "Jon walks out and shuts the door behind him.",
                           ["Maya"])[1] == ["Jon"])
    check("a lone 'she' finds the she-character",
          S.sheet_for_beat(_g, "She lies still.", ["Jon"])[1] == ["Maya"])
    check("...and a lone 'him' the he-character",
          S.sheet_for_beat(_g, "Maya looks up at him.", ["Maya"])[1] == ["Maya", "Jon"])
    check("the sheet's declaration is what resolves it",
          S.sheet_pronoun("Maya: 27, she, grey scarf") == "she"
          and S.sheet_pronoun("Jon: 34, he, overalls") == "he")
    check("...they is understood too",
          S.sheet_pronoun("Ash: 30, they, boots") == "they")
    check("...and a sheet declaring none says so",
          S.sheet_pronoun("Maya: 27, grey coat") is None)
    # With no declaration there is nothing to resolve against, so it falls back to
    # the last beat's people rather than guessing.
    check("no declared pronouns falls back to the last cast",
          S.sheet_for_beat("Maya: 27, grey coat\nJon: 34, overalls",
                           "She lies still.", ["Maya"])[1] == ["Maya"])
    # Names are matched CASE-SENSITIVELY. Prose capitalises a name, and matching
    # without case made the word "will" find a character called Will.
    _w = "Will: 30, he, grey coat\nGrace: 27, she, red jacket"
    check("an ordinary word is not a name",
          S.sheet_for_beat(_w, "She will walk to the window.", [])[1] == ["Grace"])
    check("...nor is a lowercase one", "Grace" not in
          S.sheet_for_beat(_w, "He says grace before eating.", [])[1])
    check("...while the capitalised name still matches",
          S.sheet_for_beat(_w, "Will walks to the window.", [])[1] == ["Will"])
    # A beat naming nobody keeps the last beat's people rather than emptying the frame.
    check("a beat naming nobody holds the last cast",
          S.sheet_for_beat(sheet, "The camera pushes in.", ["Maya"])[1] == ["Maya"])
    # ...but with NOTHING before it, describing everyone is the reported failure: the
    # whole character memory lands in a shot on the strength of not knowing who is in
    # it, and a person the text describes is a person the model draws.
    check("with no history, two on the sheet is a guess not worth making",
          S.sheet_for_beat(sheet, "The camera pushes in.")[1] == [])
    check("...and the same for a person the beat does not name",
          S.sheet_for_beat(sheet, "Someone knocks at the door.")[1] == [])
    check("...but a lone character is unambiguous and still resolves",
          S.sheet_for_beat("Maya: 27, grey coat", "The camera pushes in.")[1] == ["Maya"])
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
    # A near-clean reference is an invitation to REPRODUCE it, framing included, and
    # that is a matter of degree rather than a format error. This is the dial, and
    # shot 1 is where it shows: no keyframe there, so the reference is the only
    # picture and nothing competes with reproducing it.
    rn = S.reference_note(1, 0.999, False)
    check("a near-clean reference is explained", "REPRODUCE them" in rn)
    check("...naming framing as what carries over", "framing and background" in rn)
    check("...with the values to try", "0.95" in rn and "0.90" in rn)
    check("...and why shot 1 shows it", "only picture" in rn)
    check("with a first_frame, shot 1 has a keyframe to compete",
          "only picture" not in S.reference_note(1, 0.999, True))
    check("no references, nothing to say", S.reference_note(0, 0.999, False) == "")
    # Softened is the other branch: identity without copying, at the cost of the
    # handoff riding as a reference rather than anchoring.
    soft = S.reference_note(1, 0.90, False)
    check("a softened aug reads differently", "softened" in soft)
    check("...and states what it costs", "weaker continuity" in soft)


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


def test_undressing_completely():
    print("\n=== a beat that names no garment at all ===")
    # "strip out of their clothes, becoming naked" names nothing, so every other
    # removal path had nothing to take off -- and the scene went on listing the whole
    # wardrobe, re-stamped into every later shot, which is how the clothes came back.
    for _b in ("Nora undresses completely.", "Both of them strip out of their clothes.",
               "She strips off and gets in.", "He is naked by the window.",
               "She takes everything off.", "They are wearing nothing.",
               "She stands there nude.", "Stripped bare, she waits."):
        check(f"reads as undressing: {_b[:38]!r}", S.strips_bare(_b))
    # A naked eye is not a person, and stripping paint is not undressing.
    for _b in ("He examines it with the naked eye.", "A naked flame in the corner.",
               "She strips the paint off the door.", "He undoes his coat.",
               "She takes her coat off.", "He walks in."):
        check(f"not undressing: {_b[:38]!r}", not S.strips_bare(_b))
    # The beat says nothing about WHAT comes off, so it is read off the wardrobe.
    sheet = ("Kate: 27, she, grey coat, black jeans, brown boots, a white shirt, "
             "handcuffs on her wrists, a steel collar.")
    got = S.garments_in(sheet)
    check(f"every garment is found (got {got})",
          got == ["coat", "jeans", "boots", "shirt"])
    # Taking clothes off does not unlock anything: hardware is cleared by an explicit
    # 'remove:' and by nothing else.
    check("restraints are not clothing",
          not any(w in got for w in ("handcuffs", "collar")))
    check("...and neither is anything else in the line",
          not any(w in got for w in ("she", "wrists", "steel", "white")))
    check("nothing worn, nothing found", S.garments_in("Kate: 27, she, red hair.") == [])
    # Said once. Listing eight garments coming off is eight more mentions of clothing
    # in a shot whose point is that there is none.
    check("the clause finishes the removal inside the shot",
          "away by the last frame" in S.BARE_HOLD)
    check("...and says what is left", "bare skin" in S.BARE_HOLD)
    check("...while the hardware stays on", "stays fastened" in S.BARE_HOLD)
    check("...positively phrased",
          not re.search(r"\b(?:no|not|never|without|nothing)\b", S.BARE_HOLD, re.I))


def test_a_name_with_no_entry():
    print("\n=== somebody the sheet never describes ===")
    sheet = "Maya: she, 27, grey coat.\nJon: he, 35, jeans."
    beats = ["Maya walks in.",
             "Alex says hello to Maya.",
             "Alex walks to the window.",
             "Maya stands up and Alex takes her hand.",
             "Jon takes her coat off."]
    # The guard keeps the entries for the people a beat names. There is no entry to
    # keep for Alex, so those shots stage somebody the model is told nothing about.
    check("the undescribed person is found",
          S.unknown_people(beats, sheet) == {"Alex": [2, 3, 4]})
    # ...and the shot that has ONLY that person describes nobody: Alex is not on the
    # sheet, so there is no entry to keep, and Maya is not in the beat either.
    kept, who = S.sheet_for_beat(sheet, "Alex walks to the window.", ["Maya"])
    check("...which is why that shot describes the wrong person", who == ["Maya"])
    check("nobody on the sheet is reported", "Maya" not in S.unknown_people(beats, sheet))
    # A capitalised word is only a name once it has appeared MID-sentence. That
    # separates a name from an ordinary word opening a sentence, with no list of
    # ordinary words to keep.
    check("a word that only ever opens a sentence is not a name",
          S.unknown_people(["Alex walks in.", "Alex sits down."], sheet) == {})
    check("...and one appearance mid-sentence is enough",
          S.unknown_people(["Alex walks in.", "Maya greets Alex."], sheet)
          == {"Alex": [1, 2]})
    for _b in ("Maya walks in and pulls on her Nike leggings.",   # behind a determiner
               "She turns the TV off.",                           # all caps
               "Then she waits. Later she leaves.",                # sentence openers
               'Jon says: "Sure. Let us go."',                     # a quoted line
               "Maya walks into Jon's kitchen."):                  # possessive of a known name
        check(f"not reported as a person: {_b[:38]!r}",
              S.unknown_people([_b], sheet) == {})


def test_how_clothes_actually_come_off():
    print("\n=== the verbs people write removals in ===")
    # Reported as "clothing removals are bugged". These phrasings took the garment
    # off ON SCREEN while the scene kept saying it was worn -- and the scene is
    # re-stamped into every later shot, so it came back on and stayed on.
    sc = ("Kate: she, 27, white crop top, black leggings, grey jacket, black boots. "
          "A wooden chair, a bare light, a door, a table.")
    for _b, _want in (("Kate kicks off her boots.", "boots"),
                      ("Kate kicks her boots off.", "boots"),
                      ("Kate steps out of her leggings.", "leggings"),
                      ("Kate wriggles out of her leggings.", "leggings"),
                      ("Kate slides the jacket off.", "jacket"),
                      # Over a head is off. A garment goes over a head coming off or
                      # going on, and every strip verb is one-directional.
                      ("Mike lifts her top over her head.", "top")):
        check(f"{_b[:34]!r} -> {_want}", S.infer_removals(_b, sc) == [_want])
    # A particle belongs to the NEAREST verb before it. "kicks the chair and Mike
    # walks off" ends in "off", but it is the walking that is off -- and reading it
    # as a removal deleted the chair from the scene.
    for _b in ("Kate steps back and the light goes off.",
               "Kate kicks the chair and Mike walks off.",
               "Mike lifts the table and carries it off.",
               "Kate steps through the door.",
               "Kate lifts her chin and looks away.",
               "Kate slides the chair over.",
               "Mike works at the table until the light goes off."):
        check(f"not a removal: {_b[:36]!r}", S.infer_removals(_b, sc) == [])
    # ...and in the trailing form the particle ENDS the object, so the clause after
    # it is not part of what came off.
    check("what follows the particle is a new clause",
          S.infer_removals("Mike takes her jacket off and drops it on the chair.",
                           sc) == ["jacket"])
    # The exceptions to both rules: a verb that swallowed its particle, and one that
    # needs none. There the object follows the verb and the sentence runs on.
    check("'pulls off her boots' still reads",
          S.infer_removals("Mike pulls off her boots.", sc) == ["boots"])
    check("'unzips her jacket and pulls it off' still reads",
          S.infer_removals("Mike unzips her jacket and pulls it off.", sc) == ["jacket"])


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
    # The tag is the BINDING between an image and the subject the prompt describes,
    # and it belongs IN the prompt: comfy_extras/nodes_minimax_h3.py says "the prompt
    # refers to them as <Picture i>" and "Use the same tags when prompting". A picture
    # the prompt refers to is that subject; a picture it does not refer to is ANOTHER
    # subject. Stripping the tag does not remove a spare person, it creates one.
    refs = ["A", "B", "C", "D"]
    out, imgs, dropped = S.resolve_tags("Kate, <Picture 2>, walks in.", refs)
    check("the tag survives into the prompt", "<Picture 1>" in out)
    check("...renumbered to what the shot carries", "<Picture 2>" not in out)
    check("...and carrying the right image", imgs == ["B"])
    out2, imgs2, _ = S.resolve_tags("Kate <Picture 2> and Dan <Picture 4> meet.", refs)
    check("two slots renumber in order",
          "<Picture 1>" in out2 and "<Picture 2>" in out2 and imgs2 == ["B", "D"])
    # The shape it actually appears in: a character-sheet line naming who the
    # picture is.
    out4, imgs4, _ = S.resolve_tags("Kate: <Picture 1>, 22, she, blonde hair.", refs)
    check("a sheet line keeps its binding", out4 == "Kate: <Picture 1>, 22, she, blonde hair.")
    check("...and carries that image", imgs4 == ["A"])
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
    # A chastity belt gets NO placement clause: it is the item most likely to arrive
    # with its own <Picture N>, and a written description of where the shield and the
    # lock sit argues with the picture instead of adding to it.
    for _t in ("Jon shows her a chastity belt.",
               "Jon locks a chastity belt on her.",
               "Jon shows her a plugged chastity belt."):
        check(f"no clause for: {_t[:38]!r}", S.unanchored_hardware(_t) == [])
    # The plain belt entry must not reach inside "chastity belt" either, or the shot
    # is told where a waistband sits when the picture already shows the object.
    check("the plain belt phrase does not reach it",
          not any("closes around the waist" in p
                  for p in S.unanchored_hardware("Jon shows her a chastity belt.")))
    check("...while an ordinary belt still gets placed",
          S.unanchored_hardware("Jon holds up a belt.")
          == ["a belt closes around the waist and hips"])
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


def test_a_tape_gag_stays_tape():
    print("\n=== a tape gag is flat, and stays tape ===")
    # Reported: a duct tape gag that had become a mask by the later beats. Two
    # separate causes, both of them in the text.
    #
    # First, the placement clause was wrong for it. Tape lies flat against the
    # face; "a gag sits in the mouth" describes something with bulk, and that
    # clause was going onto every shot of the chain.
    for _t in ("Jon puts a duct tape gag on her.",
               "Jon gags her with duct tape.",
               "Jon shows her a tape gag."):
        check(f"tape lies flat: {_t[:34]!r}",
              S.unanchored_hardware(_t) == [S._TAPE_GAG_CLAUSE])
    # ...and it must not collect BOTH clauses, which disagree about the bulk.
    check("one clause, not two",
          S._GAG_CLAUSE not in S.unanchored_hardware("Jon puts a duct tape gag on her."))
    # A gag that really does have bulk keeps the phrase it had.
    check("a ball gag still sits in the mouth",
          S.unanchored_hardware("Jon holds up a ball gag.") == [S._GAG_CLAUSE])
    # Tape somewhere other than the mouth must not be sent to the mouth.
    check("tape at the wrists gets no mouth clause",
          S._TAPE_GAG_CLAUSE
          not in S.unanchored_hardware("Her wrists are bound with duct tape."))
    check("tape that is not a gag at all is left alone",
          S.unanchored_hardware("Jon tapes the box shut.") == [])
    # Second, the holds. They constrained the FASTENING and nothing else, so a
    # strip of tape decoded and re-encoded once a shot had nothing in the text
    # keeping it made of tape, and it drifted to the commoner object over a face.
    for _name in ("RESTRAINT_HOLD", "CHAIN_HOLD", "CHAIN_POSE_HOLD"):
        check(f"{_name} holds the material too",
              "keeps the material and shape" in getattr(S, _name))
    # It must stay positive: at cfg 1 a negative is never evaluated.
    check("the form hold is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", S.FORM_HOLD, re.I))
    # And short. These holds are already the longest thing a restrained shot carries.
    check("...and is one short sentence",
          S.FORM_HOLD.count(".") == 1 and len(S.FORM_HOLD.split()) <= 14)
    # It says nothing about the body, which is the beat's to direct.
    check("...and constrains no body",
          not re.search(r"\b(?:she|he|her|his|they|body|still)\b", S.FORM_HOLD, re.I))


def test_a_stated_state_is_not_an_event():
    print("\n=== a described state belongs at the first frame ===")
    # Reported on a 4-step distill LoRA: "stand behind a van with its doors closed"
    # rendered the doors OPEN and the characters closing them. The text named a
    # state and never said when it was true, and a video model asked for a door
    # renders what a door does. Fewer steps make it worse: the layout is committed
    # almost immediately and there are no later steps to argue it back.
    for _t in ("Mara and Dom stand behind a van with its doors closed.",
               "They stand by the closed doors of the van.",
               "The window is open.",
               "The curtains are still drawn."):
        check(f"state read: {_t[:38]!r}", S.stated_states(_t))
    check("the state and the thing come back together",
          S.stated_states("a van with its doors closed") == [("doors", "closed")])
    # A beat that WORKS the thing is asking for exactly the motion above, so it must
    # not be told the state holds. This is the one that would break the scene.
    for _t in ("Mara opens the van doors and climbs in.",
               "Dom slams the tailgate shut.",
               "Mara pulls the curtains.",
               "Dom locks the hatch.",
               "Mara closed the doors."):
        check(f"acted on: {_t[:34]!r}", S.state_acts(_t))
    # ...while the state word sitting straight in front of its noun is an adjective.
    check("'closed doors' is not an act", not S.state_acts("They pass the closed doors."))
    check("...but 'closed the doors' is", S.state_acts("Mara closed the doors."))
    # A character sheet goes into this same text. Boots are not a door.
    for _t in ("Mara: she, 30, red coat, brown boots.",
               "Dom looks back at the yard.",
               "He pulls his hood up."):
        check(f"nothing to hold: {_t[:34]!r}",
              not S.stated_states(_t) and not S.state_acts(_t))
    # The sentence itself: positive, because at cfg 1 a negative is never evaluated.
    cl = S.state_hold([("doors", "closed")])
    check("the clause is one sentence", cl.count(".") == 1)
    check("...and says when the state is true", "first frame" in cl)
    check("...and is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", cl, re.I))
    check("...and agrees with a plural", "The doors are already closed" in cl)
    check("...and with a singular",
          "The hatch is already shut" in S.state_hold([("hatch", "shut")]))
    # Two at most. Continuity that outgrows the beat is what the beat stops being about.
    many = S.state_hold([("doors", "closed"), ("gate", "open"), ("blinds", "drawn")])
    check("at most two states are held", many.count("first frame") == 2)
    check("nothing stated, nothing said", S.state_hold([]) == "")


def test_a_staged_change_names_both_ends():
    print("\n=== which end of the action is which ===")
    # Reported on the same 4-step LoRA: some distill LoRAs render an action
    # BACKWARDS -- the beat opens the doors and the shot closes them. A beat naming
    # one state names neither END, so the reverse is an equally good answer to it.
    # Saying both ends settles it, the way a removal already says "off during this
    # shot and away by the last frame".
    check("opening runs shut to open",
          S.direction_anchor(S.state_changes("Mara opens the van doors."))
          == " The doors are shut at the first frame and open by the last.")
    check("shutting runs open to shut",
          S.direction_anchor(S.state_changes("Dom slams the tailgate shut."))
          == " The tailgate is open at the first frame and shut by the last.")
    check("locking shuts", S.state_changes("Dom locks the hatch.") == [("hatch", "shut")])
    check("lifting opens", S.state_changes("Mara lifts the lid.") == [("lid", "open")])
    # A verb that genuinely goes either way gets NO anchor. Drawing the curtains
    # closes them and pulling a door can do either, and a wrong anchor is worse than
    # none: it asks for the reversal instead of merely allowing it.
    for _t in ("Mara pulls the curtains.", "Dom draws the blinds.",
               "Mara slides the door.", "Dom swings the gate."):
        _ch = S.state_changes(_t)
        check(f"no direction guessed: {_t[:30]!r}",
              _ch and _ch[0][1] is None and S.direction_anchor(_ch) == "")
    # ...but it still counts as having been WORKED, so the old state is not re-asserted
    # in a later shot. Not knowing the new state is a reason to say nothing, not a
    # reason to say the previous thing.
    check("an ambiguous verb still latches", S.state_acts("Dom draws the blinds.") == ["blind"])
    check("an adjective still does not", S.state_acts("They pass the closed doors.") == [])
    # The sentence itself.
    cl = S.direction_anchor([("doors", "open")])
    check("the clause is one sentence", cl.count(".") == 1)
    check("...and names both ends", "first frame" in cl and "by the last" in cl)
    check("...and is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", cl, re.I))
    check("...and agrees with a singular",
          S.direction_anchor([("hatch", "shut")]).startswith(" The hatch is open"))
    # Two at most, sharing a budget with the held states.
    many = S.direction_anchor([("doors", "open"), ("lid", "open"), ("gate", "shut")])
    check("at most two changes are anchored", many.count("first frame") == 2)
    check("nothing staged, nothing said", S.direction_anchor([]) == "")
    check("...and a directionless change says nothing",
          S.direction_anchor([("curtains", None)]) == "")


def test_sound_described():
    print("\n=== a beat that asks for a sound keeps its audio ===")
    # Silence is conditioned on encoded silence, which is not "no speech" but "no
    # sound at all" -- no footsteps, no chain, no room tone. It exists to stop an
    # unconditioned branch inventing a VOICE, and a beat asking for a sound is
    # asking for audio on purpose.
    for _t in ("The chain drags and rattles across the concrete.",
               "Jon's boots echo on the stone floor.",
               "She breathes hard through the gag.",
               "A low hum off the strip light.",
               "The door slams behind him.",
               "Rain on the window.",
               "She gasps."):
        check(f"sound asked for: {_t[:38]!r}", S.sound_described(_t))
    for _t in ("Maya lies still on the floor.", "Jon walks to the window.",
               "Maya looks up at him.", ""):
        check(f"no sound asked for: {_t[:38]!r}", not S.sound_described(_t))
    # A quoted line is speech, handled separately -- this is about everything else.
    check("speech and sound are separate questions",
          S.has_speech('He says: "Get up."') and not S.sound_described("He nods."))


def test_sound_is_derived_from_the_action():
    print("\n=== the sound a beat implies, without writing it twice ===")
    # H3 is joint, so the same prose conditions the audio branch -- and a beat that
    # says what happens has already said what it sounds like.
    check("walking gets footsteps",
          "footsteps" in S.sounds_for("Jon walks in holding a pair of scissors."))
    check("...scissors get blades",
          "blades through fabric" in S.sounds_for("Jon cuts off her coat."))
    check("...a chain gets links",
          "chain links dragging" in S.sounds_for("Maya thrashes against the chain."))
    check("...throwing gets something landing",
          "something landing" in S.sounds_for("He throws it away."))
    check("...a lock gets a lock", "a lock snapping shut" in S.sounds_for("He locks it."))
    check("a beat that stages nothing audible gets nothing",
          S.sounds_for("Maya lies still.") == [])
    for _t in ("Jon creeps across the floor.", "Jon drags the crate to the wall.",
               "He pours a glass of water.", "He unbuckles the harness.",
               "A van pulls up outside."):
        check(f"covered: {_t[:34]!r}", S.sounds_for(_t))
    # "locks eyes with her" is a look. It was giving the shot a padlock closing.
    check("a look is not a lock", S.sounds_for("He locks eyes with her.") == [])
    check("...while a padlock still is",
          "a lock snapping shut" in S.sounds_for("Jon locks the padlock shut."))
    # The table is a table: an action outside it is silent unless the beat names the
    # sound itself. That is the honest limit of deriving foley from prose.
    check("an action outside the table gets nothing",
          S.sounds_for("The camera pushes in on her face.") == [])
    # The SPACE, as opposed to the things in it. Read from the scene -- the one thing
    # that safely can be, because a room is hard in every shot whatever happens in it,
    # while a chain standing in the scene must not rattle where nobody moves.
    check("a concrete room is hard",
          S.room_tone("A cold concrete basement.") == "hard walls giving the sound back")
    check("...a carpeted one is not",
          S.room_tone("A carpeted bedroom.") == "a soft room with little echo")
    check("...outdoors has no walls",
          "open air" in S.room_tone("A field behind the house."))
    check("...tiles ring", "tiled" in S.room_tone("A tiled bathroom."))
    check("one room, one acoustic",
          S.room_tone("A tiled bathroom off a concrete hallway.").count(",") == 0)
    check("a scene naming no space gets none", S.room_tone("Two people talking.") == "")
    check("no scene at all is fine", S.room_tone("") == "")
    # An anchor describes the CAMERA, and every anchor written for this node says
    # "depth of field". That was read as a location, so an interior scene was told it
    # sounds like open air -- in every shot, since the room tone rides on all of them.
    lens = "Medium shadows, shallow depth of field. Medium focus."
    check("'depth of field' is a lens, not a location", S.room_tone(lens) == "")
    check("...so is 'field of view'", S.room_tone("Wide lens, deep field of view.") == "")
    check("a real field is still open air",
          "open air" in S.room_tone("They cross an open field towards the barn."))
    # With `anchor` set there is no scene PARAGRAPH, so the location is written in the
    # first beat and that is where the acoustic has to come from.
    check("the opening beat is the fallback",
          S.room_tone(lens, "A workshop with a long bench under a window.")
          == "a large room with a long tail")
    check("...and the scene still wins when it names a space",
          S.room_tone("A concrete basement.", "A workshop with a bench.")
          == "hard walls giving the sound back")
    # A cue, not an inventory: the shot has a word budget and the beat needs most of it.
    many = S.sounds_for("He walks in, unlocks the chain, cuts the tape, throws it down "
                        "and slams the door.")
    check("at most three sounds", len(many) <= S.MAX_SOUNDS)
    check("...and no repeats", len(many) == len(set(many)))
    # The sentence is prose. A label like "sound:" is read as text to DRAW.
    cl = S.sound_clause(["footsteps", "a door on its hinges"])
    check("it reads as a sentence", cl.strip().startswith("It sounds like"))
    check("...joining them properly", "footsteps and a door on its hinges" in cl)
    check("...ending as one", cl.count(".") == 1)
    check("three are joined with commas",
          "footsteps, blades through fabric and a sharp impact"
          in S.sound_clause(["footsteps", "blades through fabric", "a sharp impact"]))
    check("nothing heard, nothing said", S.sound_clause([]) == "")
    check("...and it is not a labelled line",
          not re.match(r"\s*\w+\s*:", S.sound_clause(["footsteps"])))


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


def test_sound_clause_closes_the_list():
    print("\n=== a free audio branch fills itself with a voice ===")
    # H3 is joint: the audio branch drives the face. On a shot with no line the
    # branch is free, so SOMETHING fills it -- and left loosely described it fills it
    # with speech, which the mouth then performs. Closing the list leaves nothing for
    # a voice to be.
    check("one sound, closed", S.sound_clause(["footsteps"], only=True)
          == " The only sound is footsteps.")
    check("two sounds, closed", S.sound_clause(["footsteps", "a door"], only=True)
          == " The only sounds are footsteps and a door.")
    check("three sounds, closed",
          S.sound_clause(["footsteps", "a door", "rain"], only=True)
          == " The only sounds are footsteps, a door and rain.")
    # A shot that speaks keeps the open form: closing it there would be telling the
    # model the line is not among the sounds.
    check("a speaking shot is left open",
          S.sound_clause(["footsteps"]) == " It sounds like footsteps.")
    check("nothing heard, nothing said", S.sound_clause([], only=True) == "")
    # Positively phrased. At cfg 1 H3 is CFG-free and no negative prompt is
    # evaluated, so "nobody speaks" is not a prohibition -- it is the word "speaks"
    # in the prompt.
    for _p in (True, False):
        cl = S.sound_clause(["footsteps", "a door"], only=_p)
        check(f"positively phrased (only={_p})",
              not re.search(r"\b(?:no|not|never|without|nobody|silent)\b", cl, re.I))


def test_hardware_belongs_to_somebody():
    print("\n=== a restraint hold names whose ===")
    # Reported: hardware locked onto one character turned up on the other, over their
    # clothes. The hold said "every restraint stays whole and closed" and named
    # nobody, which was fine while a shot meant one person -- put a second one in the
    # frame and it becomes an instruction about whoever is on screen.
    sheet = ("Nora: 34, she, red hair, a locked steel waist belt.\n"
             "Victor: he, 41, navy overalls, work boots.\n"
             "Kate: she, 20, grey coat")
    _wear = S.restraint_wearers(sheet)
    check(f"the wearer is read from the sheet entry (got {_wear})", _wear == ["Nora"])
    check("...not from anyone else's", "Victor" not in S.restraint_wearers(sheet))
    check("nobody wearing any, nobody named", S.restraint_wearers(
        "Nora: 34, she, red hair.\nVictor: he, 41, overalls") == [])
    two = S.own_hold(S.RESTRAINT_HOLD, ["Nora"], ["Nora", "Victor"])
    check("with two people the hold names the wearer", "restraint on Nora" in two)
    check("...and says whose the hardware is", "The hardware is Nora's" in two)
    check("...pinning the other to their own entry",
          "exactly what their own entry lists" in two)
    # Positively phrased: naming who wears it is what excludes everyone else, where
    # "nobody else is wearing one" asks the model to render an absence.
    check("...positively",
          not re.search(r"\b(?:no|not|never|nobody|without)\b", two, re.I))
    # One person in shot: no ambiguity, and the words would be budget spent on nothing.
    check("one person in shot is left alone",
          S.own_hold(S.RESTRAINT_HOLD, ["Nora"], ["Nora"]) == S.RESTRAINT_HOLD)
    check("nobody wearing hardware is left alone",
          S.own_hold(S.RESTRAINT_HOLD, [], ["Nora", "Victor"]) == S.RESTRAINT_HOLD)
    check("no hold, nothing to attribute", S.own_hold("", ["Nora"], ["Nora", "V"]) == "")
    # Every variant carries the same opening, so all three attribute.
    for _n, _h in (("chain", S.CHAIN_HOLD), ("chain+pose", S.CHAIN_POSE_HOLD)):
        check(f"{_n} attributes too",
              "restraint on Nora" in S.own_hold(_h, ["Nora"], ["Nora", "Victor"]))
    # Two wearers read as a list.
    both = S.own_hold(S.RESTRAINT_HOLD, ["Nora", "Kate"], ["Nora", "Kate", "Victor"])
    check("two wearers are both named", "on Nora and Kate" in both)


def test_one_pronoun_is_one_person():
    print("\n=== three characters, and a pronoun two of them answer to ===")
    # Reported with three characters defined and two in a shot: the third was pulled
    # in. Resolution walked the sheet ENTRIES and took everyone declaring "she" --
    # unambiguous with one woman on the sheet, a guess with two, and it took both.
    three = ("Nora: 34, she, red hair.\n"
             "Kate: 27, she, blonde.\n"
             "Dan: 41, he, dark hair")
    two = "Nora: 34, she, red hair.\nDan: 41, he, dark hair"
    for _sheet, _label, _beat, _prev, _want in (
            (three, "both named outright", "Dan hands Nora the spanner.", [],
             ["Nora", "Dan"]),
            # The scene continuing is the only evidence there is, so the one who was
            # in the last beat wins.
            (three, "the last beat narrows it", "Dan takes her coat off.", ["Nora"],
             ["Dan", "Nora"]),
            # Nothing to narrow with: add NOBODY. Naming a person the beat did not is
            # the failure; leaving them to the keyframe is recoverable.
            (three, "nothing narrows it", "Dan takes her coat off.", [], ["Dan"]),
            # Already accounted for by somebody the beat names outright.
            (three, "a named person answers it", "Nora and Dan look at her hands.", [],
             ["Nora", "Dan"]),
            (three, "only one man on the sheet", "Nora walks out behind him.", [],
             ["Nora", "Dan"]),
            (three, "she only, narrowed", "She walks to the window.", ["Kate"], ["Kate"]),
            # A two-hander is unambiguous and behaves exactly as before.
            (two, "two-hander, named + her", "Dan takes her coat off.", [],
             ["Dan", "Nora"]),
            (two, "two-hander, pronoun only", "She lies still.", [], ["Nora"])):
        _got = S.sheet_for_beat(_sheet, _beat, _prev)[1]
        check(f"{_label}: {_got}", sorted(_got) == sorted(_want))
    # ...and it is reported, because the fix is to write the name.
    _amb = S.unresolved_pronouns(three, "Dan takes her coat off.", [])
    check(f"the ambiguity is reported ({_amb})",
          _amb == [("she", ["Nora", "Kate"])])
    check("...but not once the last beat narrows it",
          S.unresolved_pronouns(three, "Dan takes her coat off.", ["Nora"]) == [])
    check("...nor when the beat names one of them",
          S.unresolved_pronouns(three, "Nora and Dan look at her hands.", []) == [])
    check("...nor with only one person declaring it",
          S.unresolved_pronouns(two, "Dan takes her coat off.", []) == [])


def test_a_tagged_object_can_be_taken_off():
    print("\n=== a tagged object is still a wardrobe entry ===")
    # An object carrying its own reference -- "a silver locket <Picture 2>," -- was
    # never the HEAD of a wardrobe entry, because the entry-end test looked for the
    # comma and found the tag instead. auto_remove could therefore never take a tagged
    # object off: it needed an explicit `remove:` line, while the identical untagged
    # object came off from the prose.
    tagged = "Nora: <Picture 1>, 34, red hair, a silver locket <Picture 2>, green jacket."
    plain = "Nora: <Picture 1>, 34, red hair, a silver locket, green jacket."
    check("a tagged object is an entry head", S._is_entry_head("locket", tagged))
    check("...same as an untagged one", S._is_entry_head("locket", plain))
    check("a tag at the end of the line is fine",
          S._is_entry_head("jacket", "Nora: 34, red hair, green jacket <Picture 2>."))
    for _sc, _label in ((tagged, "tagged"), (plain, "untagged")):
        check(f"the {_label} object comes off from the prose",
              S.infer_removals("Nora takes the silver locket off.", _sc) == ["locket"])
    check("a neighbour is unaffected",
          S.infer_removals("Nora takes her green jacket off.", tagged) == ["jacket"])
    # Hardware comes off by being UNDONE, and those verbs were missing entirely: a
    # beat saying "unlocks the belt" left it described as worn for the rest of the
    # film, because nothing read as a removal at all.
    _b = "Nora: <Picture 1>, 34, a steel chastity belt <Picture 3>, green jacket, boots."
    for _beat, _want in (
            ("Dan unlocks the chastity belt and takes it off.", ["belt"]),
            ("Dan unlocks the chastity belt.", ["belt"]),
            ("Dan unbuckles the belt.", ["belt"]),
            ("She unlaces the boots.", ["boots"]),
            ("He undoes the jacket and drops it.", ["jacket"])):
        check(f"undone: {_beat[:38]!r}", S.infer_removals(_beat, _b) == _want)
    for _beat in ("Dan unlocks the door and steps out.", "She unties her hair.",
                  "He looks at the belt.", "Dan tightens the belt."):
        check(f"not a removal: {_beat[:36]!r}", S.infer_removals(_beat, _b) == [])


def test_a_written_sound_is_recognised():
    print("\n=== a sound you wrote, in the words people write it in ===")
    # Writing the sound into a beat is what opens that shot's audio branch, and it is
    # the documented way to score a shot with no dialogue. The cue list was nouns --
    # "hum", "rattle", "footsteps" -- so a sound written with an ordinary noun and a
    # sound WORD went unrecognised, and the shot was silenced. "her boots loud on the
    # concrete" is the README's own example.
    for _t in ("her boots loud on the concrete", "a low hum off the strip light",
               "her boots scuff the floor", "gravel crunching under the tyres",
               "a knock at the door", "the engine roars",
               "rain drumming on the roof", "the fan whirring overhead",
               "the chain drags and rattles"):
        check(f"heard: {_t[:34]!r}", S.sound_described(_t))
    # Ordinary action still stages nothing audible of its own, which is what keeps a
    # walking shot silent.
    for _t in ("Nora walks to the window.", "Nora looks at the toolbox.",
               "Nora sits down on the bench.", "Nora picks up the spanner.",
               "Dan hands her the cable."):
        check(f"not a written sound: {_t[:32]!r}", not S.sound_described(_t))
    # Adverbs only where the bare adjective describes something else: "quietly closes
    # the door" is a sound being MADE, while these are the absence of one or nothing to
    # do with one. Opening the branch on an establishing beat is a free branch with no
    # line in the shot, which is where an invented voice comes from.
    for _t in ("The workshop is quiet, the roller door shut.", "She is quiet.",
               "A quiet street at night.", "She gives him a quiet look.",
               "The light is faint.", "A faint smile.",
               "He ticks a box on the form."):
        check(f"still silent: {_t[:36]!r}", not S.sound_described(_t))
    for _t in ("She quietly closes the door.", "The clock is ticking."):
        check(f"...but heard: {_t[:32]!r}", S.sound_described(_t))


def test_a_body_under_effort_has_a_voice():
    print("\n=== effort makes a sound, and it is a voice ===")
    # H3 is joint, so silence on the audio branch tells the model the person makes no
    # sound -- and a person making no sound is rendered still. A beat staging effort
    # was being silenced, which is the flat, unreacting face.
    for _b in ("McKenna thrashes on the bed.", "She writhes and arches under him.",
               "He shudders and grips the sheet.", "She strains against him."):
        check(f"voiced: {_b[:34]!r}",
              any("moans of effort" in s for s in S.sounds_for(_b)))
    # What you wrote wins. Those words are already sound cues, so the beat opens the
    # branch itself and nothing is added over the top of it.
    check("a beat naming the sound is left alone", S.sounds_for("She moans.") == [])
    check("...but it does count as asking for audio", S.sound_described("She moans."))
    for _w in ("gasps", "whimpers", "groans", "pants", "sobs"):
        check(f"{_w} is heard as a sound the author wrote", S.sound_described(f"She {_w}."))
    # Effort is read from the AUTHOR's verb, so it belongs with a quoted line and a
    # written sound -- not with the things this file infers, which may never open the
    # branch. That distinction is what keeps a walking shot silent.
    check("effort opens the branch", S.exertion_in("She writhes on the bed."))
    check("...and ordinary movement does not", not S.exertion_in("Maya walks to the window."))
    # Restraints need something to pull against. The verb alone was arming it, so a
    # bed became handcuffs.
    check("thrashing loose is not restraints",
          "restraints pulling taut" not in S.sounds_for("McKenna thrashes on the bed."))
    check("...and thrashing in cuffs is",
          "restraints pulling taut" in S.sounds_for("Kate thrashes against the handcuffs."))


def test_widget_values_are_usable():
    print("\n=== a widget value that is not a number ===")
    # Saved workflows restore widget values BY POSITION, with no names stored. Remove
    # or reorder a widget and every later value shifts up one, so a boolean can land
    # in a FLOAT slot -- which is where a widget reading NaN comes from, and a NaN
    # pace makes NaN shot lengths and a render that never starts.
    for _bad in (float("nan"), float("inf"), True, False, None, "", "abc"):
        out, notes = S.sane_widgets({"pace": _bad})
        check(f"unusable value repaired: {_bad!r}", out["pace"] == 1.0 and bool(notes))
    check("...and the cause is named",
          "by POSITION" in S.sane_widgets({"pace": float("nan")})[1][0])
    # Out of range is a value the user chose, so it is clamped rather than discarded.
    check("below the minimum is clamped", S.sane_widgets({"pace": 0.01})[0]["pace"] == 0.25)
    check("above the maximum is clamped", S.sane_widgets({"pace": 9.0})[0]["pace"] == 2.0)
    check("...and reported", "clamped" in S.sane_widgets({"pace": 9.0})[1][0])
    check("a good value is untouched and silent",
          S.sane_widgets({"pace": 1.0}) == ({"pace": 1.0}, []))
    # Ints stay ints: a float step count would index a sigma schedule wrongly.
    got = S.sane_widgets({"steps": 6.7})[0]["steps"]
    check("an int widget stays an int", isinstance(got, int) and got == 6)
    # Every numeric widget on the node is covered, and each entry agrees with the
    # schema it claims to restore -- otherwise the "default" put back is a fiction.
    sch = S.H3LongVideos.INPUT_TYPES()
    numeric = {n: sp for g in ("required", "optional") for n, sp in sch[g].items()
               if sp[0] in ("INT", "FLOAT")
               and not (len(sp) > 1 and sp[1].get("forceInput"))}
    missing = sorted(set(numeric) - {"seed"} - set(S._WIDGET_RANGE))
    check(f"every numeric widget is covered (missing: {missing})", not missing)
    for _n, (_d, _lo, _hi, _c) in S._WIDGET_RANGE.items():
        opts = numeric[_n][1]
        check(f"{_n} matches its widget: table {(_d, _lo, _hi)} vs "
              f"{(opts.get('default'), opts.get('min'), opts.get('max'))}",
              (opts["default"], opts["min"], opts["max"]) == (_d, _lo, _hi))


def test_schema():
    print("\n=== node schema ===")
    # INPUT_TYPES is the schema, full stop. It used to apply defaults.json on top --
    # the user's live settings file -- so this suite passed or failed on local
    # configuration, and did fail once defaults were saved. A test that moves with
    # the machine it runs on is worse than no test, and that file is gone.
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
    # auto_remove + anchor, character_memory, character_guard, pace, auto_sound,
    # hold_scene_state.
    # A ceiling, not a target: the old node had 38 and nobody could find anything.
    # Every one added since the rebuild answers a reported failure.
    check(f"the node stays small: {n_widgets} widgets", n_widgets <= 34)
    # Present, and in the order they were ADDED -- saved workflows restore widget
    # values by position with no names stored, so a widget inserted above an
    # existing one shifts every later value in every workflow already saved. New
    # ones go on the end, and stay in the order they arrived.
    for _w in ("anchor", "character_memory", "character_guard"):
        check(f"{_w} is offered", _w in opt)
    check("...and they sit at the end, in the order they were added",
          list(opt)[-6:] == ["anchor", "character_memory", "character_guard",
                             "pace", "auto_sound", "hold_scene_state"])
    check("hold_scene_state is offered, and on",
          "hold_scene_state" in opt and opt["hold_scene_state"][1]["default"] is True)
    # reference_mode is gone. It existed only because I had concluded fl2va could not
    # carry identity references, which was wrong: references and the keyframe ride
    # together, and always did. A switch whose "on" position was the bug is worse
    # than no switch.
    check("reference_mode is gone", "reference_mode" not in opt)
    # save_defaults was removed. A workflow saved with it still sends the value, so
    # run() swallows unknown keyword arguments rather than raising on load.
    check("save_defaults is gone", "save_defaults" not in opt and "save_defaults" not in req)
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
    test_no_one_is_described_twice()
    test_sheet_lines_are_terminated()
    test_character_guard()
    test_layers_from_prose()
    test_opening_pose()
    test_removal_needs_a_particle()
    test_how_clothes_actually_come_off()
    test_undressing_completely()
    test_a_name_with_no_entry()
    test_layers()
    test_removal_completes()
    test_restraints_hold()
    test_hardware_has_somewhere_to_go()
    test_a_tape_gag_stays_tape()
    test_a_stated_state_is_not_an_event()
    test_a_staged_change_names_both_ends()
    test_sound_described()
    test_sound_is_derived_from_the_action()
    test_pace()
    test_av_grid_alignment()
    test_chain_is_rigid()
    test_falling_bound()
    test_turning_around()
    test_thin_beats()
    test_auto_length()
    test_text_in_frame()
    test_reference_tags()
    test_sound_clause_closes_the_list()
    test_hardware_belongs_to_somebody()
    test_one_pronoun_is_one_person()
    test_a_tagged_object_can_be_taken_off()
    test_a_written_sound_is_recognised()
    test_a_body_under_effort_has_a_voice()
    test_widget_values_are_usable()
    test_schema()
    print()
    if _fails:
        print(f"RESULT: {len(_fails)} FAILURE(S): " + "; ".join(_fails))
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
