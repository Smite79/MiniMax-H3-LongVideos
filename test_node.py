"""Tests for H3-LongVideos.

Only what the node actually decides: how a prompt becomes shots, how a shot is
sized, and which shots get silence or a reference. There is no prompt-rewriting
layer to test any more -- your text goes through verbatim, and the test that
matters most is the one asserting exactly that.

Run: python test_node.py
"""

import importlib.util
import math
import re
import io
import os
import sys
import types
import weakref

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

for _n in ("torch", "nodes", "comfy", "comfy.utils", "comfy.sample", "comfy.samplers",
           "comfy.nested_tensor", "comfy.model_management", "latent_preview", "node_helpers"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["comfy.samplers"].KSampler = type("K", (), {"SAMPLERS": ["res_multistep"],
                                                        "SCHEDULERS": ["simple"]})
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
                 "Movement is continuous", "lips together",
                 "Everyone in this shot is silent",
                 "The line is spoken in"):
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
    check("every shot gets the same length",
          len({S.align_frame_count(10 * 24) for _ in range(5)}) == 1)


def test_speech_and_refs():
    print("\n=== silence and references ===")
    check("a quoted line counts as speech", S.has_speech('She says: "Get up."'))
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
    body, toks, _adds = S.extract_directives("Dan cuts off her jacket.\nremove: jacket")
    check("the directive never reaches the model", "remove:" not in body)
    check("...and the beat itself is untouched", body == "Dan cuts off her jacket.")
    check("the item is captured", toks == ["jacket"])
    check("several items on one line",
          S.extract_directives("x\nremove: coat, shirt")[1] == ["coat", "shirt"])
    check("'off:' works too", S.extract_directives("x\noff: hat")[1] == ["hat"])
    check("a beat with no directive is unchanged",
          S.extract_directives("She walks in.")[:2] == ("She walks in.", []))
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
    _obj = "Nora: <Picture 1>, 34, red hair, a silver locket <Picture 2>, green jacket."
    check("an object's tag leaves with the object",
          S.picture_tags(S.scrub_removed(_obj, ["locket"])) == [1])
    check("...and the person's stays", "<Picture 1>" in S.scrub_removed(_obj, ["locket"]))
    check("removing something else keeps both",
          S.picture_tags(S.scrub_removed(_obj, ["jacket"])) == [1, 2])
    check("removing both leaves only the person's",
          S.picture_tags(S.scrub_removed(_obj, ["locket", "jacket"])) == [1])
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
    _lead = "Mara: <Picture 1>, she, blue jeans, <Picture 2> a chastity belt."
    check("a leading tag goes with its object when the person is already tagged",
          S.picture_tags(S.scrub_removed(_lead, ["chastity belt"])) == [1])
    check("...and the trailing form still does",
          S.picture_tags(S.scrub_removed(
              "Mara: <Picture 1>, she, blue jeans, a chastity belt <Picture 2>.",
              ["chastity belt"])) == [1])
    check("...while an object tag nothing removes stays",
          S.picture_tags(S.scrub_removed(
              "Mara: <Picture 1>, she, a locket <Picture 3>, blue jeans.",
              ["jeans"])) == [1, 3])
    check("text with none of the tokens is untouched",
          S.scrub_removed("A basement with devices on the walls. Kate is 20.", ["jacket"])
          == "A basement with devices on the walls. Kate is 20.")
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
        gone.extend(t for t in S.extract_directives(_b)[1] if t not in gone)
    final = S.scrub_removed(sc, gone)
    check("both stay gone in a later beat",
          "jacket" not in final and "shirt" not in final and "black boots" in final)


def test_inferred_removals():
    print("\n=== a removal read out of the beat's own prose ===")
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
    for _b in ("Dan cuts the rope from her wrists.", "Dan takes off her handcuffs."):
        check(f"no inferred restraint: {_b[:30]!r}", S.infer_removals(_b, sc) == [])
    check("an ordinary beat infers nothing",
          S.infer_removals("Kate lies still.", sc) == [])
    check("a beat cannot remove what is not worn",
          S.infer_removals("Dan cuts off her cape.", sc) == [])
    for _t in (["top"], ["shorts"], ["top", "shorts"]):
        check(f"the cuffs survive removing {_t}",
              "handcuffed" in S.scrub_removed(sc, _t))
    check("...and an explicit remove: still clears them",
          "handcuffed" not in S.scrub_removed(sc, ["handcuffed"]))
    # The and-joined entry keeps its innocent half.
    _s = S.scrub_removed(sc, ["top"])
    check("the neighbour in the same entry stays", "black shorts" in _s)
    check("...and reads cleanly", ",black" not in _s and "  " not in _s)


def test_a_remove_line_is_read_as_the_garment():
    """REPORTED: a garment taken off is still listed in the beats that follow.

    A `remove:` item was looked up in the sheet word for word, so anything but the
    bare noun -- "her jacket", "the jacket", "jackets", "Jacket." -- matched nothing
    and the jacket stayed on for the rest of the film."""
    print("\n=== a remove: line is read as the garment ===")
    for line, want in (("her jacket", ["jacket"]), ("the jacket", ["jacket"]),
                       ("jackets", ["jacket"]), ("Jacket.", ["jacket"]),
                       ("Kate's jacket", ["jacket"]), ("jacket and shirt", ["jacket", "shirt"]),
                       ("jacket; shirt", ["jacket", "shirt"]), ("bra & panties", ["bra", "panties"]),
                       ("black and white shirt", ["black and white shirt"]),
                       ("cuffs and collar", ["cuffs", "collar"]),
                       ("coat, shirt", ["coat", "shirt"])):
        got = S.extract_directives("x\nremove: " + line)[1]
        check(f"remove: {line!r} -> {want}", got == want)
    sc = ("Kate: she, 25, blue denim jacket, black and white striped shirt, black jeans.\n"
          "Ana: she, 30, green jacket.")
    for tok, want in (("blue jacket", "blue denim jacket"),
                      ("black and white shirt", "black and white striped shirt"),
                      ("jacket", "jacket"), ("red jacket", "red jacket")):
        check(f"the sheet's own words for {tok!r}", S.sheet_form(tok, sc) == want)
    check("'takes the her jacket off' is gone",
          "the her" not in S.off_by_last_frame(
              S.extract_directives("x\nremove: her jacket")[1], "Kate", sc, "x"))


def test_every_way_a_garment_comes_off_is_read():
    """REPORTED with the one above. A removal written any way but verb-then-object
    was not read, so the garment stayed on the sheet line of every later shot."""
    print("\n=== every way a garment comes off is read ===")
    sc = "Kate: she, 25, blue denim jacket, white shirt, black jeans, red bra."
    for beat, want in (("Kate's jacket comes off.", ["jacket"]),
                       ("Her jacket is removed.", ["jacket"]),
                       ("Her jacket is quickly pulled off.", ["jacket"]),
                       ("Kate gets her jeans off.", ["jeans"]),
                       ("Kate takes off her jacket, shirt and bra.", ["jacket", "shirt", "bra"]),
                       ("Kate takes off her jacket, shirt, and jeans, then sits.",
                        ["jacket", "shirt", "jeans"]),
                       ("Kate takes off her jacket, sits down and sighs.", ["jacket"]),
                       ("Kate takes off her jacket, the white shirt underneath clinging to her.",
                        ["jacket"])):
        got = S.infer_removals(beat, sc)
        check(f"read: {beat!r}", got == want)
    for beat in ("Kate gets her jacket and heads out.", "Kate holds her jacket, shirt and bag.",
                 "The red bra comes off the rack.", "Her jacket is taken off the hook.",
                 "Is her jacket coming off?", "Kate asks Dan to take off her jacket."):
        check(f"not a removal: {beat!r}", S.infer_removals(beat, sc) == [])
    mem = "Kate: she, 25, white shirt.\nDan: he, 40, grey shirt."
    check("whose it is, by the possessive",
          S.removal_owner("Dan takes off her shirt.", "shirt", mem) == "Kate")
    check("...and his own", S.removal_owner("Dan takes off his shirt.", "shirt", mem) == "Dan")
    check("the one undressed, not the one undressing",
          S.strips_who("Dan undresses her.", ["Kate", "Dan"], mem) == ["Kate"])


def test_a_word_does_not_move_the_room():
    """REPORTED: the scenery changes when it should not, and positions reset.

    Every one of these cut the film to a room it was never in, or staged somebody
    already in the frame walking in -- and either one starts the shot fresh, with the
    room and the bodies redrawn."""
    print("\n=== a word does not move the room ===")
    for text in ("Kate lies on the bed in her room.", "Dan waits in the room.",
                 "Kate tosses her jeans in the laundry basket.",
                 "Kate sits in the office chair.", "Kate lies in the pool of light.",
                 "Kate lies back on the bed in the studio flat."):
        check(f"no room in {text!r}", S.place_named(text) == "")
    for text in ("Dan walks across the room to the bed.", "Dan walks into the room."):
        check(f"no journey in {text!r}", S.travel_legs(text) == ("", "", ""))
    check("the other room is somewhere else",
          S.travel_legs("Dan goes into the other room.")[2] == "other room")
    check("...and so is the spare room",
          S.travel_legs("Dan walks into the spare room.")[2] == "spare room")
    check("a door still belongs to its room",
          S.place_named("They stand at the kitchen door.") == "kitchen")
    check("...and a garden is still a garden",
          S.place_named("Maya sits in the garden.") == "garden")
    sheet = "Kate: she, 25, grey sweater.\nDan: he, 40, grey shirt."
    for text in ("Dan enters her.", "Dan comes inside her.", "Dan slips inside Kate.",
                 "Dan enters Kate.", "Dan slips into her."):
        check(f"into a person is not into the room: {text!r}",
              S.comes_in(text, sheet) == [] and not S.arrives_in(text))
    for text in ("Dan enters the room.", "Dan enters her bedroom.", "Dan enters Kate's room.",
                 "Dan enters Room 12.", "Dan walks in."):
        check(f"...but this is an entrance: {text!r}", S.comes_in(text, sheet) == ["Dan"])


def test_character_sheet():
    print("\n=== a character sheet is not a beat ===")
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
    sheet = "Maya: 27, grey scarf, black jacket\nJon: 34, navy overalls"
    keep, who = S.sheet_for_beat(sheet, "Maya lies still on the floor.")
    check("a beat naming one person keeps one", who == ["Maya"])
    check("...and drops the other's line", "Jon" not in keep and "Maya" in keep)
    keep, who = S.sheet_for_beat(sheet, "Jon walks out and shuts the door.", ["Maya"])
    check("a beat naming the other keeps the other", who == ["Jon"])
    check("...and lets the first go", "Maya" not in keep)
    _g = "Maya: 27, she, grey scarf, black jacket\nJon: 34, he, navy overalls"
    keep, who = S.sheet_for_beat(_g, "Jon takes her jacket off.", ["Maya"])
    check("a pronoun brings in who it refers to", sorted(who) == ["Jon", "Maya"])
    check("...so the garment coming off is still described", "grey scarf" in keep)
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
    check("no declared pronouns falls back to the last cast",
          S.sheet_for_beat("Maya: 27, grey coat\nJon: 34, overalls",
                           "She lies still.", ["Maya"])[1] == ["Maya"])
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


def test_two_person_cast_has_one_body_each():
    print("\n=== two named people have one body each ===")
    check("an exact pair gets a composition constraint",
          S.cast_hold(["Dan", "Crystal"]) ==
          " There are two people in the shot, with one body for each person.")
    check("duplicate names do not manufacture a pair",
          S.cast_hold(["Dan", "Dan"]) == " There is one person in the shot: one body, one face.")
    check("a solo shot gets a count too",
          S.cast_hold(["Dan"]) == " There is one person in the shot: one body, one face.")
    check("...and it stands down for staged extras",
          S.cast_hold(["Dan"], "A crowd watches him.") == "")
    check("a crowd shot is untouched", S.cast_hold(["Dan", "Crystal", "Mara"]) == "")


def test_extracted_planning_policies():
    print("\n=== extracted planning policies stay aligned ===")
    quiet = S.ShotAudio(False, False, False, True, 0.5, S.AUDIO_LATENT_FPS)
    check("a quiet shot is pinned", quiet.pinned)
    line = S.ShotAudio(True, True, False, True, 0.5, S.AUDIO_LATENT_FPS)
    check("dialogue gets a 20-frame lead", line.lead_frames == 20)
    tail = S.ShotAudio(True, True, False, True, 0.5, S.AUDIO_LATENT_FPS, 2.0, 2.0, 226)
    check("a short line in a long shot gets a silent tail", tail.tail_frames == 197)
    check("a six-argument ShotAudio has no tail", line.tail_frames == 0)
    check("no margin, no tail",
          S.ShotAudio(True, True, False, True, 0.5, S.AUDIO_LATENT_FPS, 2.0, 0.0, 226).tail_frames == 0)
    check("a wordless shot has no line to pin behind",
          S.ShotAudio(False, False, False, True, 0.5, S.AUDIO_LATENT_FPS, 0.0, 2.0, 226).tail_frames == 0)
    check("a line that fills its shot leaves nothing to pin",
          S.ShotAudio(True, True, False, True, 0.5, S.AUDIO_LATENT_FPS, 2.0, 2.0, 90).tail_frames == 0)
    check("a sliver under half a second is not worth clipping a word for",
          S.ShotAudio(True, True, False, True, 0.5, S.AUDIO_LATENT_FPS, 2.0, 1.5, 107).tail_frames == 0)
    check("a carried room cannot duplicate a tagged subject",
          not S._cond_module.may_carry_room(["Dan"], ["Dan", "Crystal"], {"Dan"}))
    check("an untagged room carry is safe when its cast remains",
          S._cond_module.may_carry_room(["Dan"], ["Dan", "Crystal"], set()))
    plan = S.ShotPlan()
    plan.add("one", ["Dan"], False, False, False, [])
    plan.set_frame_counts([73])
    check("one add keeps every field aligned", plan.validate() is plan)
    plan.add("two", ["Crystal"], True, True, False, [])
    try:
        plan.set_frame_counts([90])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched shot lengths accepted")
    check("an invalid update leaves shot lengths unchanged",
          [shot.frame_count for shot in plan.shots] == [73, 0])
    plan.set_frame_counts([73, 90])
    check("each shot owns its duration", plan.shots[1].frame_count == 90)

    frames = S.FrameAccumulator(4, torch.float32, True)
    frames.add(torch.full((3, 1, 1, 3), 1.0))
    frames.add(torch.full((2, 1, 1, 3), 2.0001))
    frames.add(torch.full((1, 1, 1, 3), 3.0))
    actual = frames.finish()[:, 0, 0, 0]
    check("overflow keeps later shots in order without reducing precision",
          torch.equal(actual, torch.tensor([1., 1., 1., 2.0001, 2.0001, 3.])))
    frames = S.FrameAccumulator(3, torch.float32, True)
    frames.add(torch.ones((3, 1, 1, 3)))
    finished = frames.finish()
    finished_ref = weakref.ref(finished)
    del finished
    check("finishing lets downstream code release the video buffer",
          finished_ref() is None)


def test_layers_from_prose():
    print("\n=== a layer stays out of the text until it is uncovered ===")
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
    sc = ("A basement. Maya: 27, grey scarf, black jacket. Wrists cuffed behind back. "
          "She stays lying on her side on the floor.")
    note = S.posture_note(sc, False)
    check("a posture sentence is found", "opening pose" in note)
    check("...and its position reported", "4 of 4" in note)
    check("...pointing at the mechanism that pins it", "first_frame" in note)
    check("...saying it must be a composed frame", "composed frame" in note)
    check("...and where a portrait actually goes", "ref_image_1" in note)
    check("nothing said when a first_frame is wired", S.posture_note(sc, True) == "")
    check("...or when no posture is described",
          S.posture_note("A basement. Maya walks to the window.", False) == "")
    check("...or with no scene at all", S.posture_note("", False) == "")
    rn = S.reference_note(1, 0.999, False)
    check("a near-clean reference is explained", "REPRODUCE them" in rn)
    check("...naming framing as what carries over", "framing and background" in rn)
    check("...with the values to try", "0.95" in rn and "0.90" in rn)
    check("...and why shot 1 shows it", "only picture" in rn)
    check("with a first_frame, shot 1 has a keyframe to compete",
          "only picture" not in S.reference_note(1, 0.999, True))
    check("no references, nothing to say", S.reference_note(0, 0.999, False) == "")
    soft = S.reference_note(1, 0.90, False)
    check("a softened aug reads differently", "softened" in soft)
    check("...and states what it costs", "weaker continuity" in soft)


def test_removal_needs_a_particle():
    print("\n=== an ordinary action is not a removal ===")
    sc = ("A bare basement. Kate, 20, blonde, black shiny latex crop top, "
          "white cotton shorts, brown leather boots, a grey coat.")
    for _b in ("Dan cuts off her shorts.", "Dan pulls off her boots.",
               "Dan removes her coat.", "Dan unzips her coat and pulls it off.",
               "Dan strips off her coat.", "Dan throws her coat away."):
        check(f"a removal still fires: {_b[:32]!r}", S.infer_removals(_b, sc))
    check("unzipping alone takes nothing off", S.infer_removals("Dan unzips her coat.", sc) == [])
    check("...it opens the coat", ("grey coat", "open") in S.engine.displaced_garments(
        "Dan unzips her coat.", sc))
    for _b in ("Dan pulls down her shorts.", "Dan pulls her shorts down.",
               "Dan pushes up her crop top."):
        check(f"down is displacement, not removal: {_b[:34]!r}",
              not S.infer_removals(_b, sc) and S.displaced_garments(_b, sc))
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
    check("restraints are not clothing",
          not any(w in got for w in ("handcuffs", "collar")))
    check("...and neither is anything else in the line",
          not any(w in got for w in ("she", "wrists", "steel", "white")))
    check("nothing worn, nothing found", S.garments_in("Kate: 27, she, red hair.") == [])
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
    check("the undescribed person is found",
          S.unknown_people(beats, sheet) == {"Alex": [2, 3, 4]})
    kept, who = S.sheet_for_beat(sheet, "Alex walks to the window.", ["Maya"])
    check("...which is why that shot describes the wrong person", who == ["Maya"])
    check("nobody on the sheet is reported", "Maya" not in S.unknown_people(beats, sheet))
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
    sc = ("Kate: she, 27, white crop top, black leggings, grey jacket, black boots. "
          "A wooden chair, a bare light, a door, a table.")
    for _b, _want in (("Kate kicks off her boots.", "boots"),
                      ("Kate kicks her boots off.", "boots"),
                      ("Kate steps out of her leggings.", "leggings"),
                      ("Kate wriggles out of her leggings.", "leggings"),
                      ("Kate slides the jacket off.", "jacket"),
                      ("Mike lifts her top over her head.", "top")):
        check(f"{_b[:34]!r} -> {_want}", S.infer_removals(_b, sc) == [_want])
    for _b in ("Kate steps back and the light goes off.",
               "Kate kicks the chair and Mike walks off.",
               "Mike lifts the table and carries it off.",
               "Kate steps through the door.",
               "Kate lifts her chin and looks away.",
               "Kate slides the chair over.",
               "Mike works at the table until the light goes off."):
        check(f"not a removal: {_b[:36]!r}", S.infer_removals(_b, sc) == [])
    check("what follows the particle is a new clause",
          S.infer_removals("Mike takes her jacket off and drops it on the chair.",
                           sc) == ["jacket"])
    check("'pulls off her boots' still reads",
          S.infer_removals("Mike pulls off her boots.", sc) == ["boots"])
    check("'unzips her jacket and pulls it off' still reads",
          S.infer_removals("Mike unzips her jacket and pulls it off.", sc) == ["jacket"])
    for _b in ("Kate asks Mike to take the jacket off.",
               "Kate begs Mike to unlock the jacket.",
               "Kate asks him to remove the jacket. He shakes his head.",
               "Kate pleads with him to take off the jacket.",
               "Kate wants him to take the jacket off.",
               "Mike tells her to take the jacket off.",
               "Kate asks for the jacket to come off.",
               "Kate whispers to him to take the jacket off."):
        check(f"asked for, not done: {_b[:38]!r}", S.infer_removals(_b, sc) == [])
    check("asked, then done in the next sentence",
          S.infer_removals("Kate asks him to unlock the jacket. He takes the "
                           "jacket off.", sc) == ["jacket"])
    check("asked, then done after a comma",
          S.infer_removals("Kate begs him to remove the jacket, and he removes "
                           "the jacket.", sc) == ["jacket"])
    check("'Before he takes...' is still a removal",
          S.infer_removals("Before he takes the jacket off, he pauses.", sc)
          == ["jacket"])
    check("'tore the jacket off' is still a removal",
          S.infer_removals("Mike tore the jacket off her.", sc) == ["jacket"])
    for _b in ('Kate walks up to Mike. "Will you take the jacket off?"',
               'Kate goes to Mike and asks, "Can you take the jacket off?"',
               "Kate asks if he will take the jacket off.",
               "Kate asks whether he can take the jacket off.",
               'Kate asks Mike: "Take the jacket off."',
               '"Would you take the jacket off?"',
               "<d>Please take the jacket off.</d>"):
        check(f"asked, not done: {_b[:40]!r}", S.infer_removals(_b, sc) == [])
    for _b in ('"Take the jacket off." Mike unlocks the jacket.',
               'Kate asks him to take it off. Mike takes the jacket off.',
               '"Will you take it off?" Mike pulls the jacket away.',
               '"Are you ready?" Mike takes the jacket off.',
               '"Hold still." Mike removes the jacket.'):
        check(f"asked and granted: {_b[:40]!r}", S.infer_removals(_b, sc) == ["jacket"])


def test_bare_region():
    """A garment coming off with nothing under it leaves the region UNSPECIFIED,
    and an unspecified region is filled by the model's own prior. For legs that
    prior is legwear: leggings and tights appeared that the prompt never asked
    for, and the keyframe carried them into every later shot."""
    check("shorts off, nothing under -> legs named bare",
          "legs are bare" in S.bare_clause(["shorts"], {}, "crop top, shorts"))
    check("jeans off -> legs named bare",
          "legs are bare" in S.bare_clause(["jeans"], {}, "t-shirt, jeans"))
    check("boots off -> feet named bare",
          "feet and ankles are bare" in S.bare_clause(["boots"], {}, "jeans, boots"))
    check("gloves off -> hands named bare",
          "hands are bare" in S.bare_clause(["gloves"], {}, "coat, gloves"))
    check("a t-shirt still covers the torso",
          S.bare_clause(["jacket"], {}, "jacket, t-shirt") == "")
    check("tights still on cover the legs",
          S.bare_clause(["shorts"], {}, "crop top, shorts, tights") == "")
    check("panties underneath -> reveal_clause speaks, not this",
          S.bare_clause(["shorts"], {"panties": "shorts"},
                        "crop top, panties, shorts") == "")
    check("a chastity belt underneath counts as a layer",
          S.bare_clause(["shorts"], {"chastity belt": "shorts"},
                        "crop top, chastity belt, shorts") == "")
    check("jewellery has no region", S.bare_clause(["locket"], {}, "dress, locket") == "")
    check("nothing removed, nothing said", S.bare_clause([], {}, "shorts") == "")
    for _g, _w in ((["shorts"], "crop top, shorts"), (["jeans"], "tee, jeans"),
                   (["boots"], "jeans, boots"), (["skirt"], "blouse, skirt")):
        _c = S.bare_clause(_g, {}, _w).lower()
        check(f"clause names no garment: {_g[0]!r}",
              not any(w in _c for w in ("leggings", "tights", "stockings",
                                        "panties", "underwear", "knickers")))
    # Two regions read as prose, not as two capitalised sentences spliced.
    check("two regions join grammatically",
          "and the feet" in S.bare_clause(["shorts", "boots"], {}, "shorts, boots"))


def test_the_addressee_is_not_the_speaker():
    """Verb-then-name is usually the ADDRESSEE, not the speaker.

    Introduced by the inverted-attribution fix and caught in a render: "She tells
    Dan to wait" credited Dan, so the shot said "Only Dan speaks; every other mouth
    closed" -- which holds the actual speaker's mouth shut and moves the listener's.
    The voice comes out of the wrong face, which is worse than crediting nobody."""
    sheet = "McKenna: she, 22, top.\nDan: he, 30, shirt."
    for _b in ('She tells Dan to wait. "Wait here."',
               'She asks Dan: "Can you take this off?"',
               'She begs Dan for help. "Please."'):
        check(f"the addressee is not credited: {_b[:32]!r}",
              S.speakers_in(_b, sheet) == ["McKenna"])
    # A real inversion follows a CLOSING QUOTE, which is what tells them apart.
    for _b in ('"Sure thing," says Dan.', '"Sure thing." says Dan.',
               "<d>Sure thing.</d> says Dan."):
        check(f"a real inversion still reads: {_b[:30]!r}",
              S.speakers_in(_b, sheet) == ["Dan"])
    check("a subject pronoun outranks a later name",
          S.speakers_in('She tells Dan to wait. "Wait."', sheet) == ["McKenna"])
    check("...and a name still wins when it comes first",
          S.speakers_in('In the living room, Dan looks up. "Sure thing."', sheet)
          == ["Dan"])
    check("...and he resolves the same way",
          S.speakers_in('He looks at her. "Fine."', sheet) == ["Dan"])
    _two = "Ann: she, 30, coat.\nBea: she, 31, coat."
    check("an ambiguous pronoun credits nobody",
          S.speakers_in('She waits. "Now?"', _two) == [])


def test_the_audio_branch_has_its_own_last_step():
    """Babble starting at step 3 of 4 -- which is the FINAL step of a 4-step run.

    The audio branch runs on its own shifted timeline. time_shift_sigma inverts the
    video shift and re-applies the audio one, so what is left for the last step
    depends on the STEP COUNT and shift_audio, and not at all on shift_video --
    which is the dial everybody reaches for.

    At 8 steps, shift_audio 3.0 leaves 0.30. At the 4 a distilled LoRA wants, the
    same 3.0 leaves 0.50: half the audio denoising in one jump, and a branch
    resolving that much at once invents whatever is easiest."""
    for _n, _a, _want in ((4, 3.0, 0.50), (8, 3.0, 0.30), (4, 1.0, 0.25),
                          (4, 1.5, 1.0 / 3.0), (8, 1.0, 0.125), (6, 3.0, 0.375)):
        _got = S.last_audio_sigma(_n, _a)
        check(f"{_n} steps at shift_audio {_a} -> {_want:.3f}",
              abs(_got - _want) < 1e-9)
    # Fewer steps is always steeper; more shift_audio is always steeper.
    check("fewer steps leaves more for the last one",
          S.last_audio_sigma(4, 3.0) > S.last_audio_sigma(8, 3.0))
    check("...and so does a bigger audio shift",
          S.last_audio_sigma(4, 3.0) > S.last_audio_sigma(4, 1.0))
    # Garbage in does not raise: this feeds a report, never the sampler.
    check("a bad step count is harmless", S.last_audio_sigma("x", 3.0) == 0.0)
    check("a bad shift is harmless", S.last_audio_sigma(4, None) == 0.0)


def test_the_babble_advice_points_the_right_way():
    """The report that fires when the audio branch is babbling told you to make it
    worse.

    sigma = a / (steps + a - 1) rises monotonically with a, so a low step count
    needs a SMALLER shift_audio. The note scaled it as 3.0 * 8 / steps, which at the
    4 steps a distilled LoRA wants advised 6.0 -- taking the last step from 0.50 to
    0.67, from half the audio denoising in one jump to two thirds. The suite covered
    last_audio_sigma, which was right, and never read the advice built on it."""
    for _n in (2, 3, 4, 6, 8, 12, 16):
        _a = S.shift_audio_for(_n)
        check(f"{_n} steps: advice is settable", 1.0 <= _a <= 20.0)
        check(f"{_n} steps: advice never raises the last step",
              S.last_audio_sigma(_n, _a) <= S.last_audio_sigma(_n, 3.0) + 1e-9
              or _n > 8)
    # It reproduces the default exactly where the default is what you are running.
    check("8 steps rounds back to 3.0", abs(S.shift_audio_for(8) - 3.0) < 1e-6)
    check("...and 4 steps asks for less, not more", S.shift_audio_for(4) < 3.0)
    check("...landing on the default's own last step",
          abs(S.last_audio_sigma(4, S.shift_audio_for(4))
              - S.DEFAULT_LAST_AUDIO_SIGMA) < 1e-6)
    # Fewer steps -> smaller shift. The direction is the whole point of the fix.
    check("the advice falls as steps fall",
          S.shift_audio_for(4) < S.shift_audio_for(6) < S.shift_audio_for(8))
    # Where the widget floor binds it stays honest rather than printing 0.4.
    check("clamped at the widget floor", S.shift_audio_for(2) == 1.0)
    check("a bad step count is harmless", S.shift_audio_for(None) == 0.0)
    check("an impossible target is harmless", S.shift_audio_for(8, 1.0) == 0.0)


def test_a_lora_states_its_step_count_in_its_name():
    """The file name is the ONLY place a distilled LoRA says what it was built for.

    Its safetensors metadata carries rank, alpha, baked_scale and conversion notes
    and no sigma, no shift and no step count -- and comfy keeps only that metadata
    dict, dropping the path, so the name survives nowhere but the workflow graph.

    THE TRAP IS step600. Two of the shipped H3 LoRAs carry a training checkpoint in
    the name, and reading it as a sampling target would advise 600 steps off a LoRA
    that wants 4. Digits BEFORE the word, never after."""
    def _g(*names):
        return {str(i): {"class_type": "LoraLoader", "inputs": {"lora_name": n}}
                for i, n in enumerate(names)}

    check("a 4step LoRA is read as 4",
          S.lora_step_targets(_g("minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_fro_v4.safetensors"))
          == [(4, "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_fro_v4.safetensors")])
    for _n, _want in (("minimax_h3_fl2v_turbo_8step_v1.0_comfyui_resized.safetensors", 8),
                      ("minimax_h3_taomate_fl2va_3step_ema_comfyui.safetensors", 3),
                      ("some_lora_6-step.safetensors", 6),
                      ("some_lora_12_step.safetensors", 12)):
        check(f"{_n} -> {_want}", S.lora_step_targets(_g(_n))[0][0] == _want)
    # The checkpoint trap, in both files that carry it.
    for _n in ("minimax_h3_fl2v_lightx2v_v0.1_dareties_v4_step600_comfy_fro.safetensors",
               "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"):
        check(f"a training checkpoint is NOT a step target: {_n[-24:]}",
              S.lora_step_targets(_g(_n)) == [])
    # LoRAs that simply do not say.
    for _n in ("Astro nsfw.safetensors", "LTX2.3_DMD_hybrid_v2.safetensors",
               "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
               "minimax_h3_fl2v_turbo_silver_dareties_comfy_full_v1.safetensors"):
        check(f"no claim, no reading: {_n[:28]}", S.lora_step_targets(_g(_n)) == [])
    # Two distill LoRAs that disagree is the one thing really worth calling a fight.
    _two = S.lora_step_targets(_g("a_4step.safetensors", "b_8step.safetensors"))
    check("two disagreeing LoRAs are both reported", sorted(_two) == [
        (4, "a_4step.safetensors"), (8, "b_8step.safetensors")])
    # The same LoRA on model and CLIP is one LoRA.
    check("the same file twice is one entry",
          len(S.lora_step_targets(_g("a_4step.safetensors", "a_4step.safetensors"))) == 1)
    # Stacker nodes number their slots; those are lora_name inputs too.
    check("a stacker's numbered slots are read",
          len(S.lora_step_targets({"1": {"class_type": "LoraStacker", "inputs": {
              "lora_name_1": "a_4step.safetensors",
              "lora_name_2": "b_8step.safetensors"}}})) == 2)
    # This runs off a hidden input that is absent on older frontends.
    for _bad in (None, {}, "nope", {"1": {"inputs": None}}, {"1": {}}, {"1": None}):
        check(f"a graph of {_bad!r} is harmless", S.lora_step_targets(_bad) == [])


def test_the_graph_says_whether_the_schedule_is_already_set():
    """apply_model_sampling asked the reader a question about their own graph.

    "Turn off only if you patch it upstream yourself" -- and a wrong answer either
    patches the schedule twice or leaves it unset. comfy's MiniMaxH3SigmaShift
    stamps what it applied into transformer_options, so the model carries the answer.

    THE MODEL_SAMPLING OBJECT IS NOT THE TEST. An H3 checkpoint loads with the right
    FLOW_AV 12/3 schedule already on it, so "is a shift set" is true before anybody
    has touched anything -- reading that instead would stand the node's own patch
    down on every clean run. Only the stamp separates a patch from a default."""
    class Stamped:
        model_options = {"transformer_options": {
            "minimax_h3_sigma_shift_video": 8.0, "minimax_h3_sigma_shift_audio": 2.0}}
    class VideoOnly:
        model_options = {"transformer_options": {"minimax_h3_sigma_shift_video": 6.0}}
    class Clean:
        model_options = {"transformer_options": {}}
    class Defaulted:          # a loaded H3 model: schedule set, nobody patched it
        model_options = {"transformer_options": {}}
        model_sampling = "ModelSamplingAV(shift=12, audio_shift=3)"
    class Bare:
        pass
    check("an upstream patch is seen, with its numbers",
          S.upstream_h3_shift(Stamped()) == (8.0, 2.0))
    check("...and a video-only stamp reads 0 for the audio",
          S.upstream_h3_shift(VideoOnly()) == (6.0, 0.0))
    check("an untouched graph is not a patch", S.upstream_h3_shift(Clean()) is None)
    check("a model that merely HAS a schedule is not a patch",
          S.upstream_h3_shift(Defaulted()) is None)
    for _bad in (Bare(), None, "model", 7):
        check(f"{type(_bad).__name__} is harmless", S.upstream_h3_shift(_bad) is None)
    class Junk:
        model_options = {"transformer_options": {"minimax_h3_sigma_shift_video": "eight"}}
    check("an unreadable stamp is not a patch", S.upstream_h3_shift(Junk()) is None)


def test_a_repeated_naming_is_spent_as_a_pronoun():
    """The node taking its own advice.

    Naming somebody three times in one shot draws a second copy of them, and every
    clause that owns a fact pays a naming to say whose it is. Dropping the clause does
    not work -- fit_guards records why -- so the naming is spent differently: the
    first one in the node's own clause text stands, the repeats become pronouns, and
    the fact stays where it was.

    Measured on guard-heavy shots, this takes 44 namings to 41 and empties the
    five-naming bucket. Modest, and it is the whole of what is safely available: the
    rest of the namings are in the author's words, which are never touched."""
    SHE, HE = [("Mara", "she")], [("Dan", "he")]
    TWO_SHE = [("Mara", "she"), ("Kate", "she")]
    MIX = [("Mara", "she"), ("Dan", "he")]

    def out(text, who, **kw):
        return S.pronoun_rewrite(text, who, **kw)[0]

    check("the first naming stands and the second goes",
          out(" Mara is lying down. Mara is still lying down.", SHE)
          == " Mara is lying down. She is still lying down.")
    check("...and a sentence-opening pronoun takes the capital",
          "She is still" in out(" Mara is lying down. Mara is still lying down.", SHE))
    check("a possessive before a noun becomes the determiner",
          out(" Only Mara speaks. Mara's mouth is shut.", SHE)
          == " Only Mara speaks. Her mouth is shut.")
    check("...and his, for a he",
          out(" Dan stands. Dan's coat is open.", HE)
          == " Dan stands. His coat is open.")
    check("a name after a preposition takes the object form",
          out(" Dan stands. Mara turns to Dan.", HE)
          == " Dan stands. Mara turns to him.")
    # THE ATTRIBUTION ITSELF IS LEFT ALONE. "the sobbing is her" is not English, and
    # the clause exists to say whose vocal it is so nobody else's mouth is opened.
    check("a predicate possessive keeps its name",
          out(" Only Mara sobs. The sobbing is Mara's.", SHE)
          == " Only Mara sobs. The sobbing is Mara's.")
    # The restraint that makes the whole thing safe.
    check("two people answering to 'she' means nothing is rewritten",
          out(" Mara is lying down. Kate looks at Mara.", TWO_SHE)
          == " Mara is lying down. Kate looks at Mara.")
    check("...while one of each is fine",
          out(" Mara is lying down. Dan looks at Mara. Mara waits.", MIX)
          == " Mara is lying down. Dan looks at her. She waits.")
    check("extras staged means nothing is rewritten",
          out(" Mara is lying down. Mara waits.", SHE, extras=True)
          == " Mara is lying down. Mara waits.")
    check("a single naming is left alone",
          out(" Mara is lying down.", SHE) == " Mara is lying down.")
    # It reports what it did, per person, so info can say so.
    check("the swaps are counted and named",
          S.pronoun_rewrite(" Mara waits. Mara sits. Mara stands.", SHE)[1]
          == [("Mara", 2)])
    for _bad in (None, "", " text with nobody in it"):
        check(f"{_bad!r} is harmless", S.pronoun_rewrite(_bad, SHE)[1] == [])
    for _who in (None, [], [("", "she")], [("Mara", None)], [("Mara", "it")]):
        check(f"cast {_who!r} is harmless",
              S.pronoun_rewrite(" Mara waits. Mara sits.", _who)[1] == [])


def test_the_dit_gets_the_pinned_pool_to_itself():
    """Every step as slow as the first, where it used to be slow once and then fast.

    Under --enable-dynamic-vram whatever part of the DiT does not fit the card streams
    from pinned host RAM every step. ComfyUI marks a model idle, and so lets its pins
    go, only between nodes -- and this node encodes and samples inside one, so the
    25 GB text encoder was still holding its pins beside a 32 GB DiT against a 49.6 GB
    limit. The DiT could not pin the rest of itself and re-read it every step.
    Asking free_memory for pins_required did not help: it evicts idle models only.

    REPORTED NEXT: the ComfyUI drive reading on every beat. The fix above dropped every
    other model's pins outright, every shot, so the whole text encoder came back off
    the drive at every encode and both VAEs at every decode. Only the DiT's SHORTFALL
    is released now, from the biggest holder first -- and nothing when RAM allows."""
    _rt = S._runtime_module
    GB = 1024 ** 3
    saved = {k: getattr(_rt.mm, k, None) for k in ("free_memory", "get_torch_device",
                                                   "soft_empty_cache",
                                                   "current_loaded_models",
                                                   "MAX_PINNED_MEMORY",
                                                   "TOTAL_PINNED_MEMORY")}
    saved_ram = _rt._ram_available
    calls, freed = [], {}

    class Patcher:
        def __init__(self, name, size, pinned, dynamic=True):
            self.name, self.dynamic, self.size = name, dynamic, size
            self.load_device = "cuda:0"
            self.pins = [pinned]
            self.model = type("M", (), {})()
            self.model.dynamic_pins = {"cuda:0": {"weights": (None, [], [-1], self.pins,
                                                              [0], {})}}
            self.model.memory_required = lambda s: 2 * GB
        def is_dynamic(self): return self.dynamic
        def model_size(self): return self.size
        def partially_unload_ram(self, n):
            got = min(n, self.pins[0])
            self.pins[0] -= got
            freed[self.name] = freed.get(self.name, 0) + got
            return got

    class Loaded:
        def __init__(self, patcher): self.model = patcher

    lat = {"samples": torch.zeros(1, 16, 8, 32, 32)}

    def run(ram, dit_pinned=0, cap=-1, total=0):
        calls.clear()
        freed.clear()
        dit = Patcher("dit", 32 * GB, dit_pinned)
        te, vae = Patcher("te", 25 * GB, 25 * GB), Patcher("vae", 1 * GB, 1 * GB)
        _rt.mm.current_loaded_models = [Loaded(te), Loaded(dit), Loaded(vae)]
        _rt.mm.MAX_PINNED_MEMORY, _rt.mm.TOTAL_PINNED_MEMORY = cap, total
        _rt._ram_available = lambda: ram
        _rt._evict_all_but(dit, lat)
        return freed

    try:
        _rt.mm.get_torch_device = lambda: "cuda:0"
        _rt.mm.soft_empty_cache = lambda *a, **k: calls.append("empty")
        _rt.mm.free_memory = lambda need, dev, keep_loaded=(): calls.append(
            [lm.model.name for lm in keep_loaded])
        got = run(ram=60 * GB)
        check("VRAM is freed around the DiT, as before", calls == [["dit"]])
        check("with RAM to spare nothing is unpinned -- the drive is not read", got == {})
        check("...nor once the DiT is already pinned",
              run(ram=4 * GB, dit_pinned=32 * GB) == {})
        got = run(ram=20 * GB)
        _short = 32 * GB + 2 * GB - 20 * GB + 256 * 1024 ** 2
        check("short of RAM, the text encoder gives up exactly the shortfall",
              got.get("te") == _short)
        check("...and the VAEs about to decode keep theirs", "vae" not in got)
        check("...and the DiT keeps its own", "dit" not in got)
        got = run(ram=40 * GB, cap=int(49.6 * GB), total=26 * GB)
        check("the pin cap is a shortfall too",
              got.get("te") == int(26 * GB + 32 * GB - int(49.6 * GB)) + 256 * 1024 ** 2)
        # A ComfyUI without dynamic VRAM has no pins to measure, and must still sample.
        freed.clear()
        _rt.mm.current_loaded_models = [Loaded(object())]
        _rt._evict_all_but(type("P", (), {"model": type("M", (), {
            "memory_required": staticmethod(lambda s: GB)})()})(), lat)
        check("an older ComfyUI unpins nothing and does not crash", freed == {})
    finally:
        _rt._ram_available = saved_ram
        for k, v in saved.items():
            if v is not None:
                setattr(_rt.mm, k, v)
            elif hasattr(_rt.mm, k):
                delattr(_rt.mm, k)


def test_no_report_touches_weight_data():
    """Reported: renders went several times slower.

    The render path was byte-identical, so nothing per-step had changed. What had
    changed was what ran just BEFORE it. A report measuring how much of a LoRA
    survived the compute dtype pulled six real weights up to float32, did the B@A
    matmul, added, cast back and subtracted -- about 800 MiB of transient float32 per
    sampled layer at H3's [5376, 7168], and half a second of it, immediately before
    sampling started. On a card already streaming a model that does not fit, that is
    resident weights evicted, and every step after it streams more.

    It is gone. So is the quantized-checkpoint note beside it, which described the
    merge path a streaming model never takes.

    WHAT IS LEFT MAY READ SHAPES AND NOTHING ELSE. A weight here raises the moment
    anything asks for its data, so a report that starts materialising one fails this
    rather than quietly costing a render."""
    class Landmine:
        """A weight that will say its shape and nothing else.

        Not a torch.Tensor on purpose: anything reaching for .to(), .detach(), a
        dtype or arithmetic gets an AttributeError, which is the failure this test
        is for. Reading .shape is all a report is allowed to do."""
        def __init__(self, *shape):
            self.shape = tuple(shape)

    class Ad:
        def __init__(self, out, r, inn):
            self.weights = [torch.zeros(out, r), torch.zeros(r, inn),
                            None, None, None, None]
    class Patcher:
        def __init__(self, ok=True):
            self.patches = {f"diffusion_model.blocks.{i}.attn.out_proj.weight":
                            [(1.0, Ad(5376, 21, 7168 if ok else 999))] for i in range(8)}
            self._sd = {k: Landmine(5376, 7168) for k in self.patches}
        def model_state_dict(self): return self._sd

    check("a fitting LoRA reports nothing, without reading a weight",
          S.lora_patch_mismatches(Patcher(True)) == [])
    bad = S.lora_patch_mismatches(Patcher(False))
    check("a mismatched one is still caught, from shapes alone", len(bad) == 1)
    check("...with the count and both shapes",
          bad[0][1] == 8 and bad[0][2] == (5376, 999) and bad[0][3] == (5376, 7168))
    # The two that DID touch weight data are gone, not merely unused.
    for _gone in ("lora_delta_survival", "lora_on_quantized",
                  "LORA_LANDS_FLOOR", "LORA_SURVIVAL_SAMPLE"):
        check(f"{_gone} is gone", not hasattr(S, _gone))
    _src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    for _gone in ("lora_delta_survival(", "lora_on_quantized("):
        check(f"no call site remains: {_gone!r}", _gone not in _src)


def test_a_lora_that_does_not_fit_is_reported_not_silent():
    """A LoRA built for the wrong variant of H3 half-loads, and nothing said so.

    comfy applies each pair as (B @ A).reshape(weight.shape) inside a bare try/except:
    a pair that will not reshape logs one ERROR line and the weight is handed back
    untouched, while every pair that DOES fit is applied. The LoRA is then half on.

    WHAT DOES NOT FIT IS THE PART THAT HOLDS STRUCTURE. H3's variants differ in the
    AdaLN input -- 2688 on the full fl2va, 8 on the pruned and on the hybrid this node
    recommends -- and are identical in attention and MLP. Measured on the shipped
    LoRAs: two of six drop exactly their 51 AdaLN pairs onto a hybrid and keep all 208
    of the rest. A distilled few-step trajectory on the attention stack with its
    timestep modulation missing is anatomy that does not resolve, on some LoRAs and
    not others, which is why it never pointed at the LoRA."""
    class Ad:                       # comfy's LoRAAdapter: weights[0] @ weights[1]
        def __init__(self, out, r, inn):
            self.weights = [torch.zeros(out, r), torch.zeros(r, inn), None, None, None, None]
    class Patcher:
        def __init__(self, patches, sd):
            self.patches, self._sd = patches, sd
        def model_state_dict(self):
            return self._sd

    HYBRID = {f"diffusion_model.blocks.{i}.adaln_proj.linear.weight": torch.zeros(96768, 8)
              for i in range(50)}
    HYBRID.update({f"diffusion_model.blocks.{i}.attn.out_proj.weight": torch.zeros(5376, 7168)
                   for i in range(50)})
    # A LoRA trained on the FULL model: AdaLN wants 2688 in, attention fits either way.
    full = {f"diffusion_model.blocks.{i}.adaln_proj.linear.weight": [(1.0, Ad(96768, 13, 2688))]
            for i in range(50)}
    full.update({f"diffusion_model.blocks.{i}.attn.out_proj.weight": [(1.0, Ad(5376, 16, 7168))]
                 for i in range(50)})
    got = S.lora_patch_mismatches(Patcher(full, HYBRID))
    check("the mismatch is caught", len(got) == 1)
    fam, n, produced, target = got[0]
    check("...reported per family, not once per block",
          n == 50 and fam.endswith("adaln_proj.linear") and ".N." in fam)
    check("...with both shapes, so it is diagnosable",
          produced == (96768, 2688) and target == (96768, 8))
    check("...and the layers that DO fit are not reported", "attn" not in fam)
    # The same LoRA converted for this variant says nothing.
    fitted = {f"diffusion_model.blocks.{i}.adaln_proj.linear.weight": [(1.0, Ad(96768, 13, 8))]
              for i in range(50)}
    check("a LoRA that fits is silent", S.lora_patch_mismatches(Patcher(fitted, HYBRID)) == [])
    # It runs on every render, so nothing here may raise.
    for _bad in (None, "model", 7, Patcher({}, HYBRID), Patcher({"k": [(1.0, None)]}, HYBRID),
                 Patcher({"k": [(1.0, Ad(1, 1, 1))]}, {}), Patcher({"k": []}, HYBRID)):
        check(f"{type(_bad).__name__} is harmless", S.lora_patch_mismatches(_bad) == [])
    class Dead:
        patches = {"k": [(1.0, Ad(1, 1, 1))]}
        def model_state_dict(self): raise RuntimeError("no state dict")
    check("a patcher that will not hand over a state dict is harmless",
          S.lora_patch_mismatches(Dead()) == [])


def test_silence_reports_what_happened():
    """The silence note reported the FLAG, not the result.

    _silent_audio_latent is defensive on purpose -- every failure returns None so a
    render never dies for a nicety. But the info then said a shot was "conditioned
    on real silence" when its audio branch was wide open, and a shot with no
    scripted line babbled with nothing in the report explaining why. H3 is joint:
    an unconditioned branch invents a voice and the picture lip-syncs to it."""
    check("a VAE with no sample rate yields no latent",
          S._silent_audio_latent(object(), 121, 24) is None)
    check("...and None from a missing VAE too",
          S._silent_audio_latent(None, 121, 24) is None)
    # The status the report reads.
    check("the status tracker exists",
          set(S._SILENCE_STATUS) == {"asked", "applied", "why"})
    S._SILENCE_STATUS.update(asked=4, applied=4, why="")
    check("nothing missed when all applied",
          S._SILENCE_STATUS["asked"] - S._SILENCE_STATUS["applied"] == 0)
    S._SILENCE_STATUS.update(asked=4, applied=1, why="no audio VAE is wired to the node")
    check("a shortfall is countable",
          S._SILENCE_STATUS["asked"] - S._SILENCE_STATUS["applied"] == 3)
    check("...and carries a reason", S._SILENCE_STATUS["why"])
    S._SILENCE_STATUS.update(asked=0, applied=0, why="")


def test_a_dropped_garment_is_not_a_fall():
    """A garment let go of falls. So does a belt, a key, a coat. The fall guard
    tells the shot what takes the landing and what the legs do, so aiming it at an
    object puts the PERSON on the floor to satisfy it -- reported as her falling to
    the ground when he took the belt off and it dropped.

    The subject is whatever sits between the start of the clause and the verb. The
    second half of _FALL_CUE already required a person; the first half -- falls,
    drops to, collapses -- required nothing at all."""
    for _b in ("Sam unlocks the belt. It drops to the ground.",
               "Sam takes the belt off and it falls to the floor.",
               "The belt falls to the floor.",
               "Kate takes off her crop top. It drops to the ground.",
               "The handcuffs drop to the floor.",
               "Her coat falls to the ground.",
               "Kate drops her coat on the ground.",
               "Kate pulls down her shorts."):
        check(f"not a fall: {_b[:40]!r}", not S.falls_in(_b))
    for _b in ("Kate falls to the floor.",
               "She falls to the floor.",
               "Kate collapses.",
               "Sam pushes her to the ground.",
               "Kate stumbles and goes down.",
               "She slumps against the wall.",
               "Kate hits the floor.",
               "Sam knocks Kate down."):
        check(f"still a fall: {_b[:40]!r}", S.falls_in(_b))
    # Both in one beat: the person is what the guard is for.
    check("a garment dropping does not mask a real fall",
          S.falls_in("Sam takes the belt off. It drops to the ground. "
                     "Kate falls to the floor."))


def test_a_short_action_gets_the_whole_shot():
    """Actions performed way ahead of schedule.

    thin_beats could always SEE this -- one action sitting in a ten second shot --
    and only ever reported it. The shot was told what happens and nothing about
    when, so the action was performed at once and the spare seconds filled by
    carrying on: the same movement repeated on whatever was nearest.

    A timing anchor, the shape the node already uses for a door ("open at the first
    frame and shut by the last") and a removal ("away by the last frame")."""
    check("a short action in a long shot is paced", S.pace_clause(3.0, 10.0))
    check("...and says when, not how fast",
          "even pace" in S.pace_clause(3.0, 10.0)
          and "slowly" not in S.pace_clause(3.0, 10.0))
    check("...naming both ends",
          "first frame" in S.pace_clause(3.0, 10.0)
          and "last" in S.pace_clause(3.0, 10.0))
    check("a shot that fits its beat is left alone", S.pace_clause(8.0, 10.0) == "")
    check("...and a small gap too", S.pace_clause(3.0, 5.0) == "")
    check("...and an equal one", S.pace_clause(3.0, 3.0) == "")
    check("a beat with no content asks for nothing", S.pace_clause(0.0, 10.0) == "")
    check("rubbish in, nothing out", S.pace_clause(None, "x") == "")
    # The clause and the report agree on which shots are thin.
    _thin = S.thin_beats(["Kate waits."], 10.0)
    check("thin_beats and pace_clause agree",
          bool(_thin) == bool(S.pace_clause(S.beat_seconds("Kate waits."), 10.0)))


def test_a_comma_separated_list_is_a_list_of_actions():
    """Scenes cut short: a beat's actions were undercounted, so the shot was sized
    for a fraction of what it stages and performed the rest inside that.

    A plain comma between verb phrases starts a new action, and it is the commonest
    way anybody writes a sequence. Only " and " used to split, so "walks in, drops
    her bag, takes off her coat, hangs it up..." counted as TWO actions and got 5.2
    seconds for ten."""
    dense = ("She walks in, drops her bag, takes off her coat, hangs it up, "
             "crosses the room, opens the window, looks out, turns back, sits "
             "down and picks up the remote.")
    check("a dense beat asks for real time", S.beat_seconds(dense) > 15.0)
    check("...and a one-action beat still does not",
          S.beat_seconds("She waits.") <= 3.0)
    check("a character sheet is not a list of actions",
          S.beat_seconds("McKenna: she, 22, Shiny white crop top, chastity belt, "
                         "blue jean shorts.") <= 3.0)
    # Sized shots follow, and the CEILING still holds.
    _ceil = S.align_frame_count(int(round(11.0 * S.H3_FPS)))
    _lens, _note = S.plan_lengths([dense], _ceil, True, 1.0)
    check("a dense beat is capped by shot_seconds", _lens[0] == _ceil)
    check("...and the report says it was capped",
          "stage more than shot_seconds allows" in _note)
    check("...naming what it wanted", "wants" in _note)
    # Nothing is capped when it fits, and the note stays quiet.
    _lens2, _note2 = S.plan_lengths(["She waits."], _ceil, True, 1.0)
    check("a short beat is not capped",
          "stage more than shot_seconds allows" not in _note2)
    # 'fixed' still gives every shot the ceiling exactly.
    _lens3, _ = S.plan_lengths([dense, "She waits."], _ceil, False, 1.0)
    check("fixed mode is unaffected", _lens3 == [_ceil, _ceil])


def test_one_person_undressing_is_one_person():
    """The second character copying the first.

    strips_bare only says WHETHER somebody ends up with no clothes on. The wardrobe
    was then read off the whole shot sheet, so a shot describing two people stripped
    BOTH -- one character undressing undressed the other. And the clauses that come
    with it named nobody: "Everything worn comes off during this shot" in a
    two-person shot is an instruction about whoever is on screen."""
    cast = ["McKenna", "Dan"]
    for _b, _want in (
            ("McKenna and Dan sit down. McKenna takes off her clothes.", ["McKenna"]),
            ("Dan takes off his clothes and gets on the bed.", ["Dan"]),
            ("McKenna watches as Dan undresses.", ["Dan"]),
            ("McKenna and Dan undress.", ["McKenna", "Dan"])):
        check(f"who undresses: {_b[:36]!r}",
              sorted(S.strips_who(_b, cast)) == sorted(_want))
    check("nobody undresses in a plain beat", S.strips_who("Dan waits.", cast) == [])
    check("one person in the shot is that person",
          S.strips_who("Somebody undresses.", ["Dan"]) == ["Dan"])
    # own_body names whose body it is, and only where there is more than one.
    _bare = (" The legs are bare from the hip down, the skin itself the outermost "
             "surface there.")
    check("the body is attributed with two people",
          S.own_body(_bare, "McKenna", cast).startswith(" McKenna's legs are bare"))
    check("...and everyone else is pinned to their own entry",
          "own entry lists" in S.own_body(_bare, "McKenna", cast))
    check("...and a one-person shot is left alone",
          S.own_body(_bare, "McKenna", ["McKenna"]) == _bare)
    check("the full-strip hold is attributed too",
          "Everything McKenna is wearing" in S.own_body(S.BARE_HOLD, "McKenna", cast))


def test_speech_is_marked_as_speech():
    """Quotation marks say nothing to the model. <d> and </d> do.

    They are special tokens H3 was trained with -- 151669 and 151670 in
    comfy/text_encoders/minimax.py -- and they mark a span as SPOKEN. A quoted
    instruction reached the model as an ordinary imperative sentence and was
    performed, often a beat before anybody said it. Refusing to STAGE it, which
    every reader here now does, did nothing about the model reading it."""
    check("a quoted line becomes a marked one",
          S.mark_dialogue('Dana says: "Take off your shorts and lie down."')
          == "Dana says: <d>Take off your shorts and lie down.</d>")
    check("every word survives, in order",
          "Take off your shorts and lie down."
          in S.mark_dialogue('Dana says: "Take off your shorts and lie down."'))
    check("a one-word line is still a line",
          "<d>Wait.</d>" in S.mark_dialogue('Dana says: "Wait." and steps back.'))
    check("a scare quote is left alone",
          S.mark_dialogue('She wears a "vintage" coat.')
          == 'She wears a "vintage" coat.')
    check("...and so is a quoted noun mid-sentence",
          S.mark_dialogue('He called it a "problem" and left.')
          == 'He called it a "problem" and left.')
    # Already marked, or nothing to mark.
    check("an already-marked beat is untouched",
          S.mark_dialogue("Dana turns. <d>Already marked.</d>")
          == "Dana turns. <d>Already marked.</d>")
    check("a beat with no speech is untouched",
          S.mark_dialogue("McKenna walks in.") == "McKenna walks in.")
    check("empty in, empty out", S.mark_dialogue("") == "")
    # The readers still see it as speech afterwards, so nothing it says is staged.
    _m = S.mark_dialogue('Dana says: "Take off your shorts and lie down."')
    check("the marked line is still refused by the removal reader",
          S.infer_removals(_m, "McKenna: she, 22, shorts.") == [])
    check("...and by the posture reader",
          S.posture_in(_m, ["McKenna", "Dana"]) == {})


def test_a_posture_told_is_not_a_posture_taken():
    """Being told to lie down is not lying down.

    'Dana says to McKenna: "Take off your shorts and lie down on the change
    table."' put McKenna down a beat early -- and Dana with her, since both names
    precede the verb. The removal reader has refused quoted speech since asking for
    a garment stopped removing it; this is the same rule for the body, and it uses
    the same reader."""
    cast = ["McKenna", "Dana"]
    for _b in ('Dana says to McKenna: "Take off your shorts and lie down on the '
               'change table."',
               "Dana asks McKenna to lie down.",
               "Dana tells her to sit.",
               "<d>Sit down.</d>",
               'Dana says: "Everyone sit down."'):
        check(f"asked for, not taken: {_b[:40]!r}", S.posture_in(_b, cast) == {})
    # ...and the moment it IS taken, it registers.
    check("told, then done",
          S.posture_in('Dana says: "Lie down." McKenna lies down on the table.',
                       cast) == {"McKenna": "lying down"})
    check("...and the speaker is not put down with her",
          "Dana" not in S.posture_in('Dana says: "Lie down." McKenna lies down '
                                     'on the table.', cast))
    check("a plain action still registers",
          S.posture_in("McKenna lies down on the change table.", cast)
          == {"McKenna": "lying down"})
    check("...and a transitive one still puts the object down",
          S.posture_in("Dana lays McKenna down on the change table.", cast)
          == {"McKenna": "lying down"})


def test_an_action_lets_go_of_a_posture():
    """A latched pose survives until another is staged -- and a beat can put
    somebody back on their feet without ever saying so. "Dana takes out a new
    nappy and places it on the change table" is not something anybody does lying
    down, but it names no posture, so "Dana is still lying down" went on being
    said in every later shot."""
    poses = {"McKenna": "lying down", "Dana": "lying down"}
    for _b, _want in (
            ("Dana takes out a new diaper and places it on the change table.",
             {"Dana"}),
            ("Dana walks to the drawer.", {"Dana"}),
            ("McKenna picks up her toy.", {"McKenna"})):
        check(f"the pose lets go: {_b[:38]!r}", S.posture_cleared(_b, poses) == _want)
    for _b in ("Dana looks at McKenna.",
               "McKenna cries.",
               "Dana strokes McKenna's hair.",
               "Dana waits."):
        check(f"the pose survives: {_b[:34]!r}", S.posture_cleared(_b, poses) == set())
    check("handling does not unseat a sitter",
          S.posture_cleared("Dana picks up the bottle.", {"Dana": "sitting"}) == set())
    check("...but walking does",
          S.posture_cleared("Dana walks to the door.", {"Dana": "sitting"}) == {"Dana"})
    check("...and handling does unseat somebody lying",
          S.posture_cleared("Dana picks up the bottle.",
                            {"Dana": "lying down"}) == {"Dana"})


def test_a_transitive_posture_puts_the_object_down():
    """"Dana lies McKenna down" puts MCKENNA down.

    The posture reader took the name before the verb, so the person doing the
    laying was latched as lying down -- and every later shot said she was still
    lying down while the beat had her up and working. It is the same subject/object
    confusion as crediting an addressee with somebody else's line."""
    cast = ["McKenna", "Dana"]
    for _b, _want in (("Dana lies McKenna down.", {"McKenna": "lying down"}),
                      ("Dana lays McKenna down on the change table.",
                       {"McKenna": "lying down"}),
                      ("Dana sits McKenna down in the chair.",
                       {"McKenna": "sitting"}),
                      ("Dana laid her down on the table.",
                       {"McKenna": "lying down"})):
        check(f"the object goes down: {_b[:36]!r}", S.posture_in(_b, cast) == _want)
    for _b, _want in (("McKenna is lying down on the change table.",
                       {"McKenna": "lying down"}),
                      ("McKenna lies on the bed.", {"McKenna": "lying down"}),
                      ("McKenna sits down and looks at Dana.", {"McKenna": "sitting"}),
                      ("Dana stands up.", {"Dana": "standing"}),
                      ("McKenna and Dana sit down.",
                       {"McKenna": "sitting", "Dana": "sitting"})):
        check(f"the subject goes down: {_b[:36]!r}", S.posture_in(_b, cast) == _want)
    # An active beat stages no posture at all, so nothing is latched from it.
    check("handling things is not a posture",
          S.posture_in("Dana takes out a diaper and places it on the table.",
                       cast) == {})
    # "lays" and "laid" are the transitive spellings and were matched by nothing.
    check("lays is read", "lying down" in S.posture_in("Dana lays her down.",
                                                       cast).values())


def _tone(secs, f, sr=44100, amp=0.4, ch=2):
    import math
    t = torch.arange(int(secs * sr), dtype=torch.float32) / sr
    return (torch.sin(2 * math.pi * f * t) * amp).unsqueeze(0).repeat(ch, 1)


def _centroid(y, sr=44100):
    m = y.mean(0)
    X = torch.fft.rfft(m).abs()
    f = torch.fft.rfftfreq(int(m.numel()), d=1.0 / sr)
    return float((X * f).sum() / X.sum())


def _above(y, split=1200.0, sr=44100):
    """Fraction of the ENERGY above `split`. The measure that tells a footstep from a
    heartbeat: a pulse has none at all up there, and anything striking a real surface
    has a contact transient that does. Measured against a synthesised heartbeat, and
    against the old recipe, both of which came back at 0.000."""
    m = y.mean(0) if y.dim() > 1 else y
    P = torch.fft.rfft(m.float()).abs().pow(2)
    f = torch.fft.rfftfreq(int(m.numel()), d=1.0 / sr)
    tot = float(P.sum())
    return float(P[f > split].sum()) / tot if tot > 0 else 0.0


def _duty(y, sr=44100, rel=0.10, ms=5.0):
    """Fraction of the time the sound is audibly present. ~1.0 is continuous; a train
    of discrete events with gaps between them is low, and low is what reads as
    tapping when the thing being built is water."""
    m = y.mean(0) if y.dim() > 1 else y
    w = max(1, int(sr * ms / 1000.0))
    k = (int(m.numel()) // w) * w
    fr = m[:k].reshape(-1, w).pow(2).mean(-1).sqrt()
    return float((fr > rel * fr.max()).float().mean())


def test_effort_verbs_open_the_branch_they_are_given_sound_for():
    """_EXERTION and the effort entry in _SOUND_FROM have to agree, and did not.

    The table gave nine verbs "unsteady breathing, with gasps and moans of effort".
    That is TEXT. Only _EXERTION opens the audio branch and exempts a shot from
    mouths_shut_when_no_line -- so those beats had the sound written into the prompt
    and were then pinned to silence with the mouth held closed: a body working in
    total silence behind a still face."""
    for v in ("She arches her back.", "She shudders.", "She bucks.",
              "She grinds against him.", "He thrusts.", "They rock together.",
              "She clutches the sheet.", "She grips his shoulder.", "She clenches."):
        check(f"effort opens the branch: {v}", S.exertion_in(v))
    # ...and the two lists still agree in the other direction.
    for v in ("She writhes.", "She strains.", "She thrashes.", "She moans."):
        check(f"still effort: {v}", S.exertion_in(v))
    check("a still scene is not effort", not S.exertion_in("She sits on the bed."))
    check("...nor is describing furniture", not S.exertion_in("The bed is made."))
    for v in ("She rocks the cradle.", "He grinds the coffee.",
              "She arches an eyebrow.", "He grips the railing and looks down.",
              "He rocks back on his heels.", "She clutches her bag and walks out.",
              "The truck rocks over the kerb.", "He thrusts the letter at her.",
              "She clenches her jaw and says nothing.",
              "She grips the wheel and drives off.", "He grips the door handle.",
              "She clutches the folder."):
        check(f"not effort: {v}", not S.exertion_in(v))
        check(f"...and no moans in its prompt: {v}",
              not [x for x in S.sounds_for(v) if "moans" in x])
    # The complement is what makes the difference, so the pairs have to split.
    for yes, no in (("She arches her back.", "She arches an eyebrow."),
                    ("They rock together.", "She rocks the cradle."),
                    ("She grinds against him.", "He grinds the coffee."),
                    ("She grips his shoulder.", "She grips the railing."),
                    ("She claws at his back.", "She claws the paint off.")):
        check(f"split: {yes!r} vs {no!r}",
              S.exertion_in(yes) and not S.exertion_in(no))


def test_furniture_under_movement_is_built():
    """The NON-VOCAL half, which the synthesiser can honestly make: a frame and a
    mattress working. Both conditions required and in either order, because a bed
    standing in the scene must not creak in a shot where nobody moves."""
    check("movement on a bed sounds",
          "a bed frame working" in S.sounds_for("They rock together on the bed."))
    check("...either order", "a bed frame working" in
          S.sounds_for("The bed shifts under them as they move together."))
    check("a bed nobody moves on is silent",
          "a bed frame working" not in S.sounds_for("She sits on the bed."))
    check("...and so is movement with no furniture",
          "a bed frame working" not in S.sounds_for("They rock together."))


def test_a_chain_can_be_fastened_to_a_wall():
    """Reported: a collar chained to a wall, and she took it off.

    _ANCHOR_POINT listed what hardware could be fastened to and the WALL was not on
    it -- nor the floor, the ceiling or a pillar. So "chains it to the wall" produced
    no anchor at all: the hold said the collar stays closed and nothing in any later
    shot said she was tethered. A shot that then has her cross the room holds a
    collar, no tether, and a beat saying she walks away, and the cheapest way for the
    model to make that agree is to take the collar off.

    The VERB is required with it. "to the <noun>" alone read movement as fastening --
    "he walks to the table" came back anchored at the table -- and adding wall and
    floor without fixing that would have made "she sinks to the floor" a chain."""
    for t in ("chains it to the wall", "chained to the wall",
              "chains her collar to the wall", "secures the chain to the floor",
              "bolted to the ceiling", "chained to a pillar",
              "tethered to a stake in the ground", "the leash is clipped to the ring",
              "locks the chain to the ring", "cuffed to the bed frame"):
        check(f"anchors: {t!r}", S.limb_anchor(t))
    check("the wall is named", S.limb_anchor("chained to the wall") == "at the wall")
    check("both halves still read",
          S.limb_anchor("handcuffed above her head to the bed frame")
          == "above the head, at the bed frame")
    # Movement is not fastening.
    for t in ("she sinks to the floor", "he walks to the table",
              "she is dragged to the bed", "she falls to the floor",
              "he crosses to the window", "she runs to the wall",
              "they move to the bed", "she looks to the door"):
        check(f"not an anchor: {t!r}", not S.limb_anchor(t))


def test_the_anchor_holds_the_part_the_hardware_is_on():
    """The clause said "holding the wrists" whatever the hardware was.

    A steel collar chained to a wall came out as WRISTS held at the wall, which is a
    different restraint -- and it leaves the neck free in the one shot whose whole
    point is that it is not. Two restraints to draw and a reason to drop one."""
    check("a collar holds the neck", S.held_part(["steel collar"]) == "neck")
    check("a leash too", S.held_part(["leash"]) == "neck")
    check("leg irons hold the ankles", S.held_part(["leg irons"]) == "ankles")
    check("shackles too", S.held_part(["shackles"]) == "ankles")
    check("a harness holds the body", S.held_part(["harness"]) == "body")
    check("cuffs still hold the wrists", S.held_part(["handcuffs"]) == "wrists")
    check("rope defaults to the wrists", S.held_part(["rope"]) == "wrists")
    check("...and so does nothing named", S.held_part([]) == "wrists")
    check("a collar among others still wins",
          S.held_part(["steel collar", "chain"]) == "neck")
    said = S.restraint_sentence("steel collar", [], [], anchor="at the wall")
    check("the sentence says neck", "holding the neck fast at the wall" in said)
    unnamed = S.restraint_sentence("", [], [], anchor="at the wall", part="neck")
    check("...even with no item named",
          "holding the neck fast at the wall" in unnamed)
    check("...and wrists remain the default",
          "holding the wrists fast at the wall"
          in S.restraint_sentence("", [], [], anchor="at the wall"))


def test_one_beat_can_put_on_two_things():
    """Reported as her breaking out of the handcuffs, and it was the prompt.

    hardware_named returns ONE item -- the most specific -- and the worn list was
    built by appending that single string. So the ordinary way to write it,

        The guard handcuffs Ana's wrists behind her back and locks a steel collar
        around her neck, chained to the wall.

    recorded the collar and dropped the handcuffs. From the next shot on the cuffs
    were not in the prompt at all, and hardware nobody mentions is hardware the model
    stops drawing: a woman with a collar and free hands, rendered exactly as told.

    The across-shots version of this bug was fixed long ago -- worn_item used to be
    overwritten by the next shot -- and the within-one-beat version was left."""
    b = ("The guard handcuffs Ana's wrists behind her back and locks a steel "
         "collar around her neck, chained to the wall.")
    got = S.hardware_all_named(b)
    check(f"both are recorded: {got}", len(got) >= 2)
    check("...the cuffs", any("cuff" in x for x in got))
    check("...and the collar", any("collar" in x for x in got))
    check("the material is kept", any("steel collar" == x for x in got))
    two = S.hardware_all_named("Dan cuffs her wrists and gags her with duct tape.")
    check(f"cuffs and tape: {two}",
          any("cuff" in x for x in two) and any("tape" in x for x in two))
    # The same thing said twice is one thing, however it is worded.
    one = S.hardware_all_named("He puts the collar on. The steel collar clicks shut.")
    check(f"one collar, not two: {one}",
          len([x for x in one if "collar" in x]) == 1)
    check("...and the longer wording wins", "steel collar" in one)
    check("nothing named is an empty list",
          S.hardware_all_named("Ana walks to the window.") == [])


def test_a_neck_is_not_behind_a_back():
    """limb_anchor merges a limb POSITION with a fixed POINT into one string, and
    with cuffs behind the back and a collar chained to a wall the sentence came out
    as "holding the neck behind the back, at the wall" -- a neck behind a back, in
    the clause whose whole job is to say plainly what holds what."""
    both = S.restraint_sentence("handcuffs, steel collar", [], [],
                                anchor="behind the back, at the wall", part="neck")
    check("no neck behind a back", "neck behind the back" not in both)
    check("the position is not buried in the hold",
          "holding the wrists behind the back" not in both)
    check("...and the point keeps its part", "holding the neck fast at the wall"
          in both)
    check("the pose is its own sentence",
          "Both arms are behind the body" in S.pose_clause("behind the back"))
    # A position on its own, and a point on its own, both still read.
    pos = S.restraint_sentence("handcuffs", [], [], anchor="behind the back")
    check("position alone leaves the hold clean",
          "behind the back" not in pos)
    pt = S.restraint_sentence("steel collar", [], [], anchor="at the wall",
                              part="neck")
    check("point alone", "holding the neck fast at the wall" in pt)


def test_a_door_is_not_a_room():
    """"door" was in _PLACE, so "Ana looks at the door" -- the most ordinary beat
    there is -- moved the whole shot: "This shot is in the door, not the room the
    scene text names." A door is a thing inside a room."""
    check("a door is not a place", not S.first_place("Ana looks at the door."))
    check("...nor when opened", not S.first_place("Ana opens the door."))
    check("...nor when walked to", not S.first_place("Ana walks to the door."))
    # A doorway is somewhere you can stand, and stays.
    check("a doorway still is", S.first_place("Ana waits in the doorway.") == "doorway")
    for room in ("kitchen", "basement", "hallway", "bathroom"):
        check(f"{room} still reads",
              S.first_place(f"Ana walks into the {room}.") == room)


def test_holding_an_item_back_never_costs_the_person():
    """hide_item is surgical where scrub_removed is not, and that distinction is
    the whole reason it exists.

    scrub_removed drops a whole comma-separated fragment, which is right for a
    garment that has come off. Used to hold a covered item back it took "green
    dress, steel collar" down to nothing and the PERSON's line with it, leaving
    shots with nobody described in them at all. That was reported, twice, as the
    node removing things from the character memory.

    This takes the item phrase and stops. The name always survives, whatever else
    the entry held, and so does every other garment in it."""
    for line, want in (
            ("Ana: <Picture 1>, she, 28, chastity belt, jeans.",
             "Ana: <Picture 1>, she, 28, jeans."),
            ("Ana: she, 28, a steel chastity belt, a skirt.",
             "Ana: she, 28, a skirt."),
            ("Ana: chastity belt, jeans.", "Ana: jeans."),
            ("Ana: a chastity belt.", "Ana.")):
        got = S.hide_item(line, ["chastity belt"])
        check(f"{line[:40]!r} -> {got!r}", got == want)
    # The name is never lost, even when the item was the only thing in the entry.
    for line in ("Ana: a chastity belt.", "Ana: chastity belt.",
                 "Ana: <Picture 1>, a chastity belt."):
        check(f"the name survives: {line[:34]!r}",
              S.hide_item(line, ["chastity belt"]).startswith("Ana"))
    # Everything else in the entry survives with it.
    full = S.hide_item("McKenna: she, <Picture 1>, 22, blonde hair, blue eyes, "
                       "mirrored steel collar with ring, chastity belt, skin "
                       "tight shiny black PVC mini-skirt.", ["chastity belt"])
    for keep in ("<Picture 1>", "blonde hair", "blue eyes", "steel collar",
                 "mini-skirt", "22"):
        check(f"kept: {keep}", keep in full)
    check("...and only the belt is gone", "chastity belt" not in full)
    # A line terminator is kept, or the next sheet line welds onto this one.
    check("the full stop survives", full.rstrip().endswith("."))
    # Nothing named, nothing changed.
    same = "Ana: she, 28, jeans."
    check("nothing to hide leaves it alone", S.hide_item(same, []) == same)


def test_a_chastity_belt_is_underwear():
    """Asked for directly: "treat the chastity belt as underwear".

    It was already layered as one -- always under the outer garment, whatever
    order the sheet lists them in -- but garments_in never saw it, because that
    reader works word by word and the item is two words. "chastity belt" is
    neither "chastity" nor "belt", and bare "belt" cannot go in the single-word
    list because a belt through trouser loops is not underwear.

    So it was the one thing in a sheet that nothing tracked as clothing: it could
    not be taken off by the prose, and auto_remove could never clear it."""
    for line, want in (
            ("Ana: she, 28, blue jeans, a steel chastity belt.", "chastity belt"),
            ("Ana: she, 28, a chastity device.", "chastity device"),
            ("Ana: she, 28, a chastity cage.", "chastity cage"),
            ("Ana: she, 28, a g-string.", "g string"),
            ("Ana: she, 28, boxer shorts.", "boxer shorts")):
        got = S.garments_in(line)
        check(f"{want} is a garment: {got}", want in got)
    # ...and it does not swallow the ordinary kind of belt.
    check("a leather belt is not underwear",
          "belt" not in " ".join(S.garments_in("Dan: he, 40, trousers and a "
                                               "leather belt.")))
    both = S.garments_in("Ana: she, 28, blue jeans, a steel chastity belt.")
    check(f"both garments are read: {both}",
          "jeans" in both and "chastity belt" in both)
    # It layers as underwear, which it already did, in either order.
    for line in ("Ana: she, 28, jeans, a chastity belt.",
                 "Ana: she, 28, a chastity belt, jeans."):
        check(f"under the jeans: {line[-28:]!r}",
              S.implied_layers(line).get("chastity belt") == "jeans")
    check("...and is recognised as an undergarment",
          S.is_undergarment("chastity belt") and S.is_undergarment("panties"))


def test_a_beat_names_the_sound_its_props_make():
    """The event sounds, which are a different system from the mixed ambient bed --
    that one builds TONE and cannot make a zipper. These go in the PROMPT, so the
    model makes them in sync with the picture."""
    def snd(b):
        return S.sounds_for(b)
    # Gaps found by listing the beats this is asked for and reading what came back.
    check("a zipper by name", "a zip running" in snd("Dan pulls the zipper down."))
    check("velcro", "velcro tearing open" in snd("Dan tears the velcro open."))
    check("rope going tight",
          "rope creaking as it goes tight" in snd("She pulls the rope tight."))
    check("lower-body garments rustle too",
          "fabric rustling" in snd("She takes off her shorts."))
    check("...and boots", "fabric rustling" in snd("She pulls off her boots."))
    # A bolt is not something dragging on the floor.
    check("a bolt is metal", snd("Dan slides the bolt across.")[0] == "a metal bolt sliding")
    check("...and a real drag still is",
          snd("Dan drags the crate across the floor.")[0] == "something dragging on the floor")
    # Cuffs being APPLIED are a ratchet; knocking is what they do afterwards.
    check("applying cuffs is a ratchet",
          "cuffs ratcheting closed" in snd("Dan clicks the cuffs shut."))
    check("...and it retires the general sound",
          "cuffs knocking" not in snd("Dan clicks the cuffs shut."))
    check("cuffs merely present still knock",
          snd("Dan cuffs her wrists behind her back.") == ["cuffs knocking"])
    check("hardware after the verb", "restraints pulling taut" in
          snd("She strains against the cuffs."))
    check("hardware before the verb", "restraints pulling taut" in
          snd("The cuffs hold her wrists as she strains."))
    check("...and for a chain", "restraints pulling taut" in
          snd("The chain is taut while she thrashes."))
    check("a ratchet reads either way", "cuffs ratcheting closed" in
          snd("The cuffs ratchet closed around her wrists."))
    # No regressions on the things that must stay silent.
    check("a look is not a padlock", snd("She locks eyes with him.") == [])
    check("thrashing alone arms no restraint",
          "restraints pulling taut" not in snd("McKenna thrashes on the bed."))
    check("a shot still gets a cue, not an inventory",
          len(snd("She walks in dragging the chain, cuffs knocking, "
                  "unzips her coat and drops it.")) <= S.MAX_SOUNDS)


def test_an_ambient_bed_is_mixed_not_conditioned():
    """Ambience laid UNDER the finished soundtrack, rather than derived in the prompt.

    Deriving it needs the audio branch left open, and an open branch on a joint model
    fills itself with a voice -- that is why ambience everywhere was reverted. A mix
    asks nothing of the model, has nothing to lip-sync to, and so cannot speak."""
    sr = 44100
    voice = _tone(4.0, 300.0, amp=0.6).unsqueeze(0)          # [1, 2, N]
    bed = {"waveform": _tone(2.0, 90.0).unsqueeze(0), "sample_rate": sr}
    out, note = S.mix_ambient(voice, sr, bed, 0.25)
    check("the soundtrack keeps its length", tuple(out.shape) == tuple(voice.shape))
    check("the bed is actually added", not torch.allclose(out, voice))
    check("...and reported", "ambient bed was laid under" in note)
    # A bed nobody asked for must cost nothing at all.
    check("level 0 is a no-op", S.mix_ambient(voice, sr, bed, 0.0) == (voice, ""))
    check("no bed is a no-op", S.mix_ambient(voice, sr, None, 0.5) == (voice, ""))
    # Wrong rate would play the bed at the wrong speed and pitch.
    b22 = {"waveform": _tone(2.0, 90.0, sr=22050).unsqueeze(0), "sample_rate": 22050}
    o2, n2 = S.mix_ambient(voice, sr, b22, 0.25)
    check("a different sample rate is resampled", "resampled from 22050 Hz" in n2)
    check("...to the right length", tuple(o2.shape) == tuple(voice.shape))
    mono = {"waveform": _tone(2.0, 90.0, ch=1).unsqueeze(0), "sample_rate": sr}
    check("a mono bed is spread to the channels",
          tuple(S.mix_ambient(voice, sr, mono, 0.25)[0].shape) == tuple(voice.shape))
    loud = {"waveform": _tone(2.0, 90.0, amp=1.0).unsqueeze(0), "sample_rate": sr}
    o4, n4 = S.mix_ambient(_tone(4.0, 300.0, amp=1.0).unsqueeze(0), sr, loud, 1.0)
    check("a mix that would clip is scaled", "stop it clipping" in n4)
    check("...and peaks at 1.0", float(o4.abs().max()) <= 1.0 + 1e-6)
    # Anything unusable leaves the render alone rather than failing it.
    check("a bed with no waveform is survivable",
          S.mix_ambient(voice, sr, {"sample_rate": sr}, 0.5)[0] is voice)


def test_the_loop_join_does_not_click():
    """A bed is looped to the length of the film, and a bed that clicks once per loop
    is the one thing a background bed must not do.

    The obvious construction is wrong and was caught by measuring: appending the
    crossfade to the END of a full-length unit leaves it finishing on x[fade-1]
    while the next repeat starts on x[0], which are not adjacent. Overlap the tail
    onto the HEAD and shorten the unit instead, so tiling steps between samples that
    were adjacent in the source."""
    sr = 44100
    n = int(9.5 * sr)
    for f in (97.3, 123.7, 211.11):
        src = _tone(2.0, f)
        loop = S._seamless_loop(src, n, sr)
        check(f"{f} Hz loops to the right length", int(loop.shape[-1]) == n)
        step_src = float((src[:, 1:] - src[:, :-1]).abs().max())
        step_loop = float((loop[:, 1:] - loop[:, :-1]).abs().max())
        reps = -(-n // int(src.shape[-1]))
        tiled = src.repeat(1, reps)[..., :n]
        step_tiled = float((tiled[:, 1:] - tiled[:, :-1]).abs().max())
        check(f"{f} Hz has no join bigger than the material itself",
              step_loop <= step_src * 1.5)
        # ...and this is not vacuous: plain tiling on the same material is far worse.
        check(f"{f} Hz plain tiling would click", step_tiled > step_src * 5)
    # A bed longer than the film is simply truncated.
    check("a long bed is cut to length",
          int(S._seamless_loop(_tone(20.0, 100.0), n, sr).shape[-1]) == n)


def test_a_garment_going_on_has_both_ends():
    """A removal is scrubbed from its staging shot AND told it FINISHES there --
    both ends, because the keyframe shows the garment on and the text has to carry
    it off. An `add:` had only the scrub's opposite: the phrase went into the same
    shot's scene block as a plain worn item.

    So a shot inheriting a last frame WITHOUT the garment was told flatly that it
    has it. That is a disagreement rather than a change, and the model settles it in
    the opening frames by turning whatever is on the body into the garment -- read
    as one thing instantly becoming another, a beat before the beat that puts it on,
    which is what those opening frames are."""
    check("a dressing is read", S.beat_stages_wearing("Kate puts her shorts back on.",
                                                      "shorts"))
    check("...with a preposition", S.beat_stages_wearing(
        "Kate pulls her shorts on over the leggings.", "shorts"))
    check("...and stepping into", S.beat_stages_wearing("Kate steps into her shorts.",
                                                        "shorts"))
    check("...and fastening", S.beat_stages_wearing("Kate zips up her jacket.", "jacket"))
    check("a reveal is not a dressing",
          not S.beat_stages_wearing("Dan cuts off her jacket and throws it away.",
                                    "shirt"))
    check("a garment merely mentioned is not put on",
          not S.beat_stages_wearing("Kate looks at her shorts on the bench.", "shorts"))
    check("...nor is something else going on a bench",
          not S.beat_stages_wearing("Kate puts her bag on the bench.", "shorts"))
    check("a removal is not a dressing",
          not S.beat_stages_wearing("Kate takes off her shorts.", "shorts"))
    c = S.wearing_clause(["her blue shorts"])
    check("the clause gives the opening frame", "as the shot opens" in c)
    check("...and the last", "by the last frame" in c)
    check("...and says it happens during the shot", "put on during this shot" in c)
    check("nothing to put on says nothing", S.wearing_clause([]) == "")


def test_a_lens_setting_is_not_a_room():
    """_PLACE is an alternation with no edges of its own, so searching it RAW matches
    inside words. "shallow depth of field" contains "hall" -- and every camera anchor
    ever written for this node says shallow. The film was put in a hallway it never
    had: stated on each shot, used as the origin of the first journey, and handed to
    room_tone, which gave a lens setting the acoustic of a cathedral."""
    check("shallow is not a hall", S.first_place("shallow depth of field") == "")
    check("...nor is it a room to correct to",
          S.where_hold("bedroom", "Shot on 35mm, shallow depth of field.") == "")
    # Other words that carry a place inside them.
    check("a doorstep is not a door", S.first_place("She waits on the doorstep.") == "")
    check("a hallmark is not a hall", S.first_place("A hallmark of the style.") == "")
    check("a bedroom is still a bedroom", S.first_place("A dimly lit bedroom.") == "bedroom")
    check("...and a hall is still a hall", S.first_place("The hall was empty.") == "hall")
    check("the travel reader was never fooled",
          S.travel_in("She walks to the shallow end.")[2] == "")


def test_the_sound_clause_spends_from_the_budget():
    """The sound clause was appended AFTER fit_guards, so it was the one piece of
    node-written text no cap could reach -- and the largest single contributor, 97
    words of 420 across a six-shot script. It is now ranked in, LAST, so it is cut
    before anything that traces to a report.

    Ranking it high was tried and measured: at the same budget it wins its words
    from the continuity holds, and the suites caught it costing the fall/landing
    guard. This asserts the ordering that survived, not the one that read best."""
    big = " " + " ".join(["word"] * (S.GUARD_FLOOR_WORDS - 5)) + "."
    # It is cuttable at all, which before this it was not: it never reached here.
    kept, dropped = S.fit_guards([(1, "keep", big), (14, "sound", big)], 0)
    check("sound can be cut", "sound" in dropped)
    # ...and it is cut BEFORE a guard that traces to a report, never instead of one.
    kept, dropped = S.fit_guards([(4, "fall", big), (14, "sound", big)], 0)
    check("the fall guard outranks it", "fall" not in dropped)
    check("...and it is the one that goes", "sound" in dropped)
    kept, _ = S.fit_guards([(14, "sound", " S."), (1, "first", " F.")], 99)
    check("the sentence order is the list order", kept.strip() == "S. F.")
    check("the floor is the measured one", S.GUARD_FLOOR_WORDS == 90)
    check("...and so is the ratio", S.GUARD_WORDS_PER_BEAT_WORD == 5)


def test_a_described_room_is_still_a_room():
    """A room is usually described, not just named: "the tiled bathroom", "the long
    hallway", "the master bedroom". Every place reader wanted the article and the
    room word ADJACENT, so one adjective made the whole journey invisible -- no ends
    named, `here` never updated, and the room hold that depends on it never fired.
    Silently, because nothing was found to warn about."""
    check("a described destination is read",
          S.travel_in("She walks him down the hallway to the tiled bathroom.")[2]
          == "bathroom")
    check("...and a described waypoint",
          S.travel_in("She walks him down the long hallway to the bedroom.")[1]
          == "hallway")
    check("...and a described origin",
          S.travel_in("She leaves the upstairs bedroom for the kitchen.")[0] == "bedroom")
    check("two modifiers still read",
          S.travel_in("She walks him to the second-floor landing.")[2] == "landing")
    check("a plain one is unaffected",
          S.travel_in("She walks him down the hallway to the bedroom.")[1:] ==
          ("hallway", "bedroom"))
    check("place_named reads a described room",
          S.place_named("Inside the small kitchen.") == "kitchen")
    check("the nearest place still wins",
          S.place_named("They stand at the kitchen door.") == "kitchen")
    # ...and a match cannot cross a preposition or a comma into the next clause.
    check("it does not cross a comma",
          S.travel_in("She walks to the sink, the bedroom dark behind her.")[2] != "bedroom")


def test_the_sound_follows_the_room():
    """The ambient bed and the room tone were read ONCE, before the shot loop, out of
    the scene. A film that walked into a tiled bathroom went on being told it sounds
    like the carpeted room it left -- H3 is joint, so that is the picture told one
    room and the audio told another inside the same conditioning."""
    check("a bathroom has its own acoustic", S.room_tone("bathroom") == "tiled walls ringing")
    check("...and its own bed", S.scene_ambient("bathroom") != "")
    check("...different from a living room's",
          S.room_tone("bathroom") != S.room_tone("A carpeted living room."))
    # A room with no sound of its own leaves the film's own bed standing.
    check("an unremarkable room changes nothing", S.room_tone("lobby") == "")


def test_the_room_follows_the_characters():
    """The scene paragraph is stamped into EVERY shot, so a script that walks from
    the living room to the bedroom goes on opening every later shot with "A living
    room." while the beat has them on the bed. The shot holds two places at once
    and settles on whichever the model weighs more, differently each time -- a
    scene that keeps changing and resetting."""
    check("a later room is stated",
          "bedroom" in S.where_hold("bedroom", "A living room."))
    check("...and says the scene disagrees",
          "takes place in the bedroom" in S.where_hold("bedroom",
                                                              "A living room."))
    # Silent where there is nothing to correct.
    check("the scene already naming it says nothing",
          S.where_hold("bedroom", "A bedroom with a low lamp.") == "")
    check("a scene naming no room says nothing",
          S.where_hold("bedroom", "Two people, late evening.") == "")
    check("no room known, nothing said", S.where_hold("", "A living room.") == "")
    check("no scene, nothing said", S.where_hold("bedroom", "") == "")
    check("a scene's room is found without a preposition",
          S.first_place("A living room.") == "living room")
    check("...and with one", S.first_place("Inside the kitchen, late.") == "kitchen")
    check("...and none where there is none", S.first_place("Two people.") == "")


def test_a_journey_has_two_ends():
    """A beat that walks somebody from one room to another is a staged change with
    two ends, exactly like a door opening. Told only where it FINISHES, the shot
    renders the destination and starts there -- the living room is the bedroom at
    the first frame and the hallway between them is never seen. Reported as scenes
    being cut short and missing their detail."""
    for _b, _want in (
            ("She walks him down the hallway to the bedroom.", ("", "hallway", "bedroom")),
            ("She leads him to the bedroom.", ("", "", "bedroom")),
            ("They go from the living room to the kitchen.", ("living room", "", "kitchen")),
            ("He walks out of the kitchen and up the stairs to the bedroom.",
             ("kitchen", "stairs", "bedroom"))):
        check(f"travel read: {_b[:38]!r}", S.travel_in(_b) == _want)
    # A place named without MOVEMENT is not a journey.
    for _b in ("She looks to the bedroom.",
               "She sits in the living room.",
               "The bedroom door is closed.",
               "He waits."):
        check(f"not travel: {_b[:34]!r}", S.travel_in(_b) == ("", "", ""))
    # Both ends, and the middle where the beat gives one.
    check("both ends are named",
          "opens in the living room" in S.travel_anchor("living room", "", "bedroom")
          and "arrives in the bedroom" in S.travel_anchor("living room", "", "bedroom"))
    check("...and the route between them",
          "along the hallway" in S.travel_anchor("living room", "hallway", "bedroom"))
    check("the established room is the origin",
          "opens in the living room"
          in S.travel_anchor("", "", "bedroom", here="living room"))
    _noorigin = S.travel_anchor("", "", "locker room")
    check("no origin anywhere still walks in",
          "enters the locker room" in _noorigin and "every step in frame" in _noorigin)
    check("...and names no origin it does not have", "opens in the" not in _noorigin)
    check("...and going nowhere says nothing",
          S.travel_anchor("bedroom", "", "bedroom") == "")
    # place_named is what latches the room when a beat only says where people are.
    check("a place is read from 'in the X'",
          S.place_named("McKenna finds Dan in the living room.") == "living room")
    check("...and nothing where none is named",
          S.place_named("McKenna waits.") == "")


def test_a_posture_carries_to_the_next_shot():
    """A beat that sits somebody down ends its shot with them seated.

    Reported as the end of one beat and the start of the next not matching: they
    are standing at the end of a shot and sitting at the start of the next, or the
    reverse. The scene-state reader tracks SCENERY -- doors, windows, drawers --
    and nothing about the body, so no later shot was ever told what pose the last
    beat left somebody in. The keyframe carries it as a picture, but the text is
    what the model reconciles that against, and text saying nothing loses to a
    reference saying something."""
    cast = ["Kate", "Sam"]
    for _b, _want in (("Kate sits down in the chair.", {"Kate": "sitting"}),
                      ("Kate takes a seat.", {"Kate": "sitting"}),
                      ("Kate kneels on the floor.", {"Kate": "kneeling"}),
                      ("Kate lies down on the bed.", {"Kate": "lying down"}),
                      ("Kate stands up.", {"Kate": "standing"}),
                      ("Kate gets to her feet.", {"Kate": "standing"}),
                      ("Kate and Sam sit down.",
                       {"Kate": "sitting", "Sam": "sitting"})):
        check(f"posture read: {_b[:32]!r}", S.posture_in(_b, cast) == _want)
    check("a second person doing something else is not seated",
          S.posture_in("Kate sits down and Sam stays by the door.", cast)
          == {"Kate": "sitting"})
    check("...and two postures in one beat land on the right people",
          S.posture_in("Kate sits down and Sam stands by the window.", cast)
          == {"Kate": "sitting", "Sam": "standing"})
    # Furniture is not a body.
    for _b in ("The chair stands in the corner.",
               "The case lies on the table.",
               "Sam looks at her.",
               "Kate walks to the door."):
        check(f"not a posture: {_b[:32]!r}", S.posture_in(_b, cast) == {})
    check("a pose for somebody not in the shot is not said",
          S.posture_hold({"Kate": "sitting"}, ["Sam"]) == "")
    check("...and is said for somebody who is",
          "Kate is sitting" in S.posture_hold({"Kate": "sitting"}, ["Kate"]))
    check("standing is never held",
          S.posture_hold({"Kate": "standing"}, ["Kate"]) == "")
    check("...while a sat pose is held, and named once",
          S.posture_hold({"Kate": "sitting", "Sam": "standing"},
                         ["Kate", "Sam"]) == " Kate is sitting.")


def test_the_wearer_is_in_the_shot():
    """A garment cannot be acted on without the person wearing it.

    "Dan unlocks the chastity belt" names only Dan, so the shot described only Dan
    -- and her sheet line went, taking BOTH her <Picture N> tags with it. The shot
    then unlocked her belt while carrying no reference at all: the belt had nothing
    to look like, and she was in frame undescribed and unpinned, which renders as
    somebody else. Reported as duplicates in a beat and a belt that stopped
    matching its image."""
    sheet = ("McKenna: <Picture 1>, she, 22, Shiny white crop top, "
             "chastity belt <Picture 2>, blue jean shorts.\n"
             "Dan: he, 30, black t-shirt, jeans.")
    for _b in ("Dan unlocks the chastity belt.",
               "Dan takes the crop top off.",
               "Dan picks up the shorts from the floor."):
        _, _who = S.sheet_for_beat(sheet, _b, ["Dan"])
        check(f"the wearer is kept: {_b[:34]!r}", sorted(_who) == ["Dan", "McKenna"])
    # His OWN things do not pull her in, and neither does scenery.
    for _b, _want in (("Dan takes off his jeans.", ["Dan"]),
                      ("Dan puts on his t-shirt.", ["Dan"]),
                      ("Dan looks out of the window.", ["Dan"]),
                      ("McKenna sits down.", ["McKenna"])):
        _, _who = S.sheet_for_beat(sheet, _b, ["Dan"])
        check(f"nobody extra: {_b[:30]!r}", _who == _want)
    check("a tagged entry still yields its head noun",
          "belt" in S.entry_heads("McKenna: <Picture 1>, she, 22, Shiny white "
                                  "crop top, chastity belt <Picture 2>, shorts."))
    check("...and the age and pronoun are not things",
          not ({"she", "22"} & set(S.entry_heads("K: she, 22, red coat."))))


def test_a_group_beat_keeps_the_group():
    """"They sit down" is two people, so both need their sheet line.

    "they" is in _PRONOUN_SET as a SINGULAR group -- the pronoun a nonbinary
    character declares -- so a plural "they" resolved to whoever the last beat
    happened to keep, and "both of them" / "the two of them" are not pronouns at
    all and matched nothing. One of the two people in the shot was left with no
    description, and a person the text does not describe is a person the model
    invents, clothes included. Reported as clothing invented for somebody who had
    been out of shot: they came back in a group beat and were never re-described."""
    sheet = "Kate: she, 30, blue coat.\nSam: he, 34, black shirt."
    for _b in ("They sit down.",
               "Both of them sit.",
               "The two of them wait.",
               "They look at each other.",
               "They walk out together.",
               "All of them turn to the door."):
        _, _who = S.sheet_for_beat(sheet, _b, ["Kate"])
        check(f"the group is kept: {_b[:30]!r}", _who == ["Kate", "Sam"], )
    for _b in ("Kate takes off her shorts and steps out of them.",
               "Kate picks up the boots and puts them by the door.",
               "Kate pulls the shorts down and kicks them away.",
               "Kate looks at their reflection."):
        _, _who = S.sheet_for_beat(sheet, _b, ["Kate"])
        check(f"an object pronoun is not the group: {_b[:34]!r}", _who == ["Kate"])
    for _b, _want in (("She sits down.", ["Kate"]),
                      ("Kate sits down.", ["Kate"]),
                      ("Sam waits by the door.", ["Sam"]),
                      ("Kate takes off her coat.", ["Kate"])):
        _, _who = S.sheet_for_beat(sheet, _b, ["Kate"])
        check(f"one person stays one: {_b[:28]!r}", _who == _want)
    # A character who DECLARES they/them is not a group.
    _nb = "Ash: they, 28, red coat.\nSam: he, 34, black shirt."
    _, _who = S.sheet_for_beat(_nb, "They pick up their bag.", ["Ash"])
    check("a declared they/them is one person", _who == ["Ash"])
    check("...and group_beat says so",
          not S.group_beat("They pick up their bag.", S.sheet_lines(_nb)))
    check("...while it is a group where nobody declares it",
          S.group_beat("They sit down.", S.sheet_lines(sheet)))


def test_generic_clothes_come_off_too():
    """"Takes off his clothes" is a removal. It names no garment the sheet lists,
    so every path that matches a garment word had nothing to take off: his wardrobe
    stayed in the scene text and was re-stamped into every later shot. Reported as
    her clothes coming off properly while his did not -- hers were named, his were
    "his clothes"."""
    for _b in ("Sam takes off his clothes.",
               "Sam takes off his clothes and gets on the bed.",
               "Sam takes his clothes off.",
               "Sam removes his clothing.",
               "Sam sheds his clothes.",
               "Sam slips out of his clothes.",
               "Sam strips.",
               "Sam undresses.",
               "Sam gets undressed."):
        check(f"undressing: {_b[:38]!r}", S.strips_bare(_b))
    # Clothes that are handled but not WORN, and the other senses of strip.
    for _b in ("Sam picks up his clothes from the floor.",
               "Sam folds the clothes.",
               "Kate looks at the clothes on the rail.",
               "She strips the paint off the door.",
               "Sam peels a strip of tape from the roll.",
               "Sam is stripping wire.",
               "Sam takes off his jumper.",
               "Sam waits."):
        check(f"not undressing: {_b[:38]!r}", not S.strips_bare(_b))


def test_the_shot_says_each_thing_once():
    """Three faults read off one real shot's prompt text.

    ONE: the removal was stated twice -- the beat says "she takes off her shorts
    and steps out of them", and the clause said the whole thing again, who and
    what included. Two statements of one action invite it being rendered twice.
    The clause now adds only what the beat leaves out: that it FINISHES here.

    TWO: the author's capitalisation was destroyed. "PVC" came back "pvc", which
    is a different token sequence than was written.

    THREE: a belt became restraint hardware because a body part appeared anywhere
    in the same text. "chastity belt" in the sheet and "she sits with her legs
    crossed" in a beat armed the restraint hold, which then latched over something
    that was never a restraint."""
    sc = ("McKenna: she, 22, Shiny white crop top, "
          "skin-tight black shiny PVC volleyball shorts.")
    # ONE -- the beat already stages it, so only the completion is added.
    _staged = S.off_by_last_frame(["shorts"], "McKenna",
                                  sc, "McKenna takes off her shorts.")
    check("a staged removal is not restated",
          "takes the" not in _staged and "own hands" not in _staged)
    check("...but it is still finished in this shot",
          "away by the last frame" in _staged and "fully removed" in _staged)
    _unstaged = S.off_by_last_frame(["shorts"], "McKenna", sc, "McKenna stands still.")
    check("an unstaged removal still names the hands",
          "McKenna takes the" in _unstaged and "own hands" in _unstaged)
    # ...and asking for it is not staging it.
    _asked = S.off_by_last_frame(["shorts"], "Dan", sc,
                                 'McKenna asks: "Can you take the shorts off?"')
    check("asking does not count as staging", "Dan takes the" in _asked)
    # TWO -- the author's capitalisation.
    check("PVC is not pvc",
          S.scene_name_for("shorts", sc) == "skin-tight black shiny PVC volleyball shorts")
    check("...and it reaches the clause", "PVC" in _staged)
    # THREE -- the qualifier has to be in the hardware's own clause.
    check("a belt is not armed by a distant body part",
          not S.restraint_present("K: she, 30, chastity belt, shorts. "
                                  "K sits with her legs crossed."))
    check("...nor a leather belt by a distant neck",
          not S.restraint_present("K: she, 30, leather belt. S rubs his neck."))
    check("a belt locked ON a body part still counts",
          S.restraint_present("K: she, 30, chastity belt locked on her hips."))
    check("...and a binding verb still counts",
          S.restraint_present("S locks the chastity belt."))
    check("plain hardware is unaffected",
          S.restraint_present("K sits with the handcuffs on."))
    # FOUR -- the speaker is the clause's SUBJECT, not the name nearest the verb.
    _sheet = "Kate: she, 30, coat.\nSam: he, 34, shirt."
    check("'Kate approaches Sam and asks' is Kate speaking",
          S.speakers_in('Kate approaches Sam and asks: "Can you help me?"',
                        _sheet) == ["Kate"])
    check("...and a plain attribution still reads",
          S.speakers_in('Sam says to Kate: "Sure."', _sheet) == ["Sam"])
    check("...and a quote after a name with no verb",
          S.speakers_in('Kate approaches Sam. "Can you help me?"', _sheet) == ["Kate"])
    check("...and a second sentence's speaker wins there",
          S.speakers_in('Kate walks in. Sam says: "Hello."', _sheet) == ["Sam"])


def test_a_removal_stays_on_one_person():
    """A modifier inside one character's garment is not another character's garment.

    "Dan pulls off her jeans shorts" names ONE garment. But "jeans" is also the head
    of Dan's own entry, so the reader matched it against his sheet line and took HIS
    trousers off too -- in a beat that never mentions him coming out of anything --
    and they stayed off, because a removal is permanent. Reported as his pants coming
    off automatically in the shot after."""
    sc = ("McKenna: she, 22, Shiny white crop top, blue jeans shorts.\n"
          "Dan: he, 40, t-shirt, jeans.")
    check("'her jeans shorts' is one garment",
          S.infer_removals("Dan pulls off her jeans shorts.", sc) == ["shorts"])
    check("...whoever is doing it",
          S.infer_removals("McKenna takes off her jeans shorts.", sc) == ["shorts"])
    check("...and the other person keeps his",
          "jeans" not in S.infer_removals("Dan pulls off her jeans shorts.", sc))
    # The opposite failure would be worse: a garment that IS his still comes off.
    check("he can still take his own off",
          S.infer_removals("Dan takes off his jeans.", sc) == ["jeans"])
    # Two garments genuinely coming off are still two.
    check("two real garments are still two",
          S.infer_removals("Dan takes off his jeans and his t-shirt.", sc)
          == ["jeans", "t-shirt"])
    check("...and a coordinated pair reads",
          sorted(S.infer_removals("McKenna takes off her top and her shorts.", sc))
          == ["shorts", "top"])
    # The layering reader shares the defect and the fix.
    check("the layer reader agrees",
          "jeans" not in S.infer_layers(["Dan pulls off her jeans shorts to show "
                                         "the thong."], sc))


def test_a_removal_names_it_the_way_the_sheet_does():
    """The removal clause uses the SHEET's words. infer_removals keys a garment by
    its head noun -- "shorts" -- which is right for matching and wrong for prose:
    the shot then read "the shorts come off" beside a sheet saying "blue jeans
    shorts", which is two garments described, and the one that came back was the
    bare one. Same defect as the displacement path, a different code path, and the
    first fix only reached the other one."""
    sc = ("McKenna: she, 22, Shiny white crop top, chastity belt, "
          "blue jeans shorts.")
    check("the removal clause carries the sheet's words",
          "The blue jeans shorts come off"
          in S.off_by_last_frame(["shorts"], "", sc))
    check("...and so does the agent form",
          "Dan takes the blue jeans shorts off"
          in S.off_by_last_frame(["shorts"], "Dan", sc))
    check("...for a two-word name too",
          "The chastity belt comes off" in S.off_by_last_frame(["belt"], "", sc))
    check("plural agreement follows the full name",
          " are away" in S.off_by_last_frame(["shorts"], "", sc))
    check("...and singular stays singular",
          " is away" in S.off_by_last_frame(["belt"], "", sc))
    _tagged = ("McKenna: <Picture 1>, she, 22, Shiny white crop top, "
               "chastity belt <Picture 2>, blue jeans shorts.")
    check("a picture tag is not part of the garment name",
          S.scene_name_for("belt", _tagged) == "chastity belt")
    check("...and the untagged ones are unaffected",
          S.scene_name_for("shorts", _tagged) == "blue jeans shorts"
          and S.scene_name_for("top", _tagged) == "Shiny white crop top")
    check("capitalisation is the author's",
          S.scene_name_for("shorts", "K: she, 22, skin-tight black shiny PVC "
                           "volleyball shorts.") == "skin-tight black shiny PVC "
          "volleyball shorts")
    check("...so the removal clause carries it",
          "The chastity belt" in S.off_by_last_frame(["belt"], "", _tagged)
          and "comes off" in S.off_by_last_frame(["belt"], "", _tagged))
    check("...and the picture the sheet gave it",
          "<Picture 2>" in S.off_by_last_frame(["belt"], "", _tagged))
    check("an untagged garment gets no tag",
          "<Picture" not in S.off_by_last_frame(["shorts"], "", _tagged))
    check("scene_tag_for finds the entry's own tag",
          S.scene_tag_for("belt", _tagged) == "<Picture 2>")
    check("...and none where the entry has none",
          S.scene_tag_for("shorts", _tagged) == "")
    check("a garment the sheet does not name still reads",
          "The cape comes off" in S.off_by_last_frame(["cape"], "", sc))
    check("no scene, no expansion", "The shorts come off"
          in S.off_by_last_frame(["shorts"], "", ""))


def test_the_sheet_names_the_garment():
    """Anything the node says about a garment uses the SHEET's words, not the
    beat's. A beat says "pulls the shorts back up" for what the sheet dressed her
    in as "blue jeans shorts"; the guard echoed the beat, so the shot carried a
    bare "the shorts" beside the sheet's full name. A model handed two differently
    named garments draws two, and the shorts came back a different colour and cut
    -- invented out of the node's own text."""
    sc = ("McKenna: she, 22, Shiny white crop top, blue jeans shorts, "
          "black leather boots.")
    check("the sheet's full name is recovered from the head noun",
          S.scene_name_for("shorts", sc) == "blue jeans shorts")
    check("...for a two-word modifier too",
          S.scene_name_for("boots", sc) == "black leather boots")
    check("a garment the sheet never names has no name",
          S.scene_name_for("skirt", sc) == "")
    check("modifiers do not cross a comma",
          "crop" not in S.scene_name_for("shorts", sc))
    check("an article is not description",
          S.scene_name_for("coat", "Kate: she, 30, the grey coat.") == "grey coat")
    # The displacement itself carries the sheet's name, whatever the beat called it.
    _d = dict(S.displaced_garments("McKenna pulls the shorts down.", sc))
    check("a displacement is stored under the sheet's name",
          "blue jeans shorts" in _d)
    _d2 = dict(S.displaced_garments("McKenna pulls her blue jeans shorts down.", sc))
    check("...and the full name still matches itself",
          "blue jeans shorts" in _d2)


def test_removal_completes():
    print("\n=== a removal has to finish inside its shot ===")
    one = S.off_by_last_frame(["coat"])
    check("the removing shot is told to finish it", "by the last frame" in one)
    check("...and that the body is clear of it", "clear of the body" in one)
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
    _b = S.off_by_last_frame(["scarf"])
    check("the action is bounded", "Everything else worn stays exactly as it is" in _b)
    check("...covering hardware as well", "untouched and fastened" in _b)
    check("...without telling the body to hold still",
          not re.search(r"\bbody stays\b|\bfor the whole shot\b|\bmotionless\b", _b, re.I))
    check("...naming no other garment", "jumper" not in _b and "coat" not in _b)
    _bound = _b.split("dropped out of frame.")[1]
    check("...and the bound is stated positively",
          not re.search(r"\bno\b|\bnot\b|\bnever\b|\bnothing\b", _bound, re.I))
    check("it reads as a sentence", one.strip().startswith("The coat"))
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
    # An added layer retires when it is itself removed.
    gone, shown = ["white shirt"], ["her white shirt is now visible",
                                    "her grey vest is now visible"]
    live = [a for a in shown if not S.names_any(a, gone)]
    check("a removed layer stops being described", live == ["her grey vest is now visible"])
    check("...and one that was not removed stays", S.names_any("a red coat", ["coat"]))


def test_a_function_word_is_never_a_character():
    print("\n=== 'The' is not somebody with no sheet entry ===")
    sheet = "McKenna: she, 22, a vest.\nDana: she, 35, a coat."
    for beats, label in (
            (['McKenna pulls at the cuffs. "Nearly there," The guard says.'], "The"),
            (['"Wait," She says. McKenna stops.'], "She"),
            (['McKenna looks up, Then she stands.'], "Then"),
            (['It is dark. "Go," It says.'], "It")):
        check(f"a function word is not a person: {label!r}",
              S.unknown_people(beats, sheet) == {})
    # ...and a REAL unknown name is still reported, which is the whole point.
    check("an unknown name is still named",
          S.unknown_people(["Dana walks in with Nora behind her."], sheet) == {"Nora": [1]})
    check("...including one that is also an ordinary word, once it is used as a name",
          S.unknown_people(["Dana nods at Grace, then Grace leaves."], sheet) == {"Grace": [1]})
    check("nobody unknown, nothing reported",
          S.unknown_people(["McKenna and Dana talk."], sheet) == {})
    for _n in ("grace", "will", "hope", "faith", "may", "mark", "rose"):
        check(f"{_n!r} stays available as a name", _n not in S._NEVER_A_NAME)
    check("the stopword list is a set", isinstance(S._NEVER_A_NAME, frozenset))


def test_thin_beats():
    print("\n=== a shot longer than its beat ===")
    one = "Dan pulls off her coat and throws it away."
    two = ("Dan pulls off her coat and throws it away, then sets the hanger down "
           "and steps back.")
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
    refs = ["A", "B", "C", "D"]
    out, imgs, dropped = S.resolve_tags("Kate, <Picture 2>, walks in.", refs)
    check("the tag survives into the prompt", "<Picture 1>" in out)
    check("...renumbered to what the shot carries", "<Picture 2>" not in out)
    check("...and carrying the right image", imgs == ["B"])
    out2, imgs2, _ = S.resolve_tags("Kate <Picture 2> and Dan <Picture 4> meet.", refs)
    check("two slots renumber in order",
          "<Picture 1>" in out2 and "<Picture 2>" in out2 and imgs2 == ["B", "D"])
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
    for _t in ("A chain-link fence runs along the yard.", "He wears a leather belt.",
               "Kate walks to the window.", "The rope hangs from the rafters.",
               "Dan tapes the box shut."):
        check(f"not a restraint: {_t[:34]!r}", not S.restraint_present(_t))
    for _t in ("Jon puts a steel clamp on her arm.", "Jon clamps a ring to her wrist.",
               "A steel clamp is clipped to her belt.",
               "Steel clamps fastened to her ankles."):
        check(f"a clamp on a body: {_t[:36]!r}", S.restraint_present(_t))
    for _t in ("A steel clamp holds the workpiece on the bench.",
               "Jon clamps the board to the workbench.",
               "Jon clips the coupon out of the paper."):
        check(f"a clamp on a thing: {_t[:36]!r}", not S.restraint_present(_t))
    check("no word sits in both the noun list and the verb list",
          not (S._RESTRAINT_MAYBE.search("clamps") and S._BINDING_VERB.search("clamps")))
    check("...the same way tape is handled",
          not (S._RESTRAINT_MAYBE.search("tapes") and S._BINDING_VERB.search("tapes")))
    check("a clamp does not earn the chain clause",
          not S.rigid_hardware("Jon puts a steel clamp on her arm."))
    check("...while a chain still does", S.rigid_hardware("a chain around her waist"))
    check("the hold is one sentence", S.RESTRAINT_HOLD.count(".") == 1)
    check("...impersonal, so it summons nobody",
          not re.search(r"\b(?:she|he|her|his|they)\b", S.RESTRAINT_HOLD, re.I))
    check("...and positive, since cfg 1 has no negative prompt",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", S.RESTRAINT_HOLD, re.I))
    check("...saying what holds", "stays closed and fastened as it was put on" in S.RESTRAINT_HOLD)


def test_hardware_has_somewhere_to_go():
    print("\n=== hardware named with nowhere to sit ===")
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
    for _t in ("Jon shows her a chastity belt.",
               "Jon locks a chastity belt on her.",
               "Jon shows her a plugged chastity belt."):
        check(f"no clause for: {_t[:38]!r}", S.unanchored_hardware(_t) == [])
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
    for _name in ("RESTRAINT_HOLD", "CHAIN_HOLD", "CHAIN_POSE_HOLD"):
        check(f"{_name} holds the material too",
              "same object in the same material" in getattr(S, _name))
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
    for _t in ("Mara and Dom stand behind a van with its doors closed.",
               "They stand by the closed doors of the van.",
               "The window is open.",
               "The curtains are still drawn."):
        check(f"state read: {_t[:38]!r}", S.stated_states(_t))
    check("the state and the thing come back together",
          S.stated_states("a van with its doors closed") == [("doors", "closed")])
    for _t in ("Mara opens the van doors and climbs in.",
               "Dom slams the tailgate shut.",
               "Mara pulls the curtains.",
               "Dom locks the hatch.",
               "Mara closed the doors."):
        check(f"acted on: {_t[:34]!r}", S.state_acts(_t))
    # ...while the state word sitting straight in front of its noun is an adjective.
    check("'closed doors' is not an act", not S.state_acts("They pass the closed doors."))
    check("...but 'closed the doors' is", S.state_acts("Mara closed the doors."))
    for _t in ("Dan walks out from behind a van with closed rear doors.",
               "They come out from behind the closed sliding door of the van.",
               "A van with shut cargo doors.",
               "Two people behind a van with closed back doors."):
        check(f"still an adjective: {_t[:38]!r}",
              S.stated_states(_t) and not S.state_changes(_t))
    for _t in ("Mara closed the rear doors.", "Mara closed the van's doors.",
               "Dom shut the two doors.", "Dom locked both doors."):
        check(f"still an act: {_t[:38]!r}",
              S.state_changes(_t) and not S.stated_states(_t))
    check("a possessive gap is still read",
          S.state_changes("Mara closed the van's doors.") == [("doors", "shut")])
    check("acting on it wins over stating it",
          not S.stated_states("Dom slams the tailgate shut.")
          and S.state_changes("Dom slams the tailgate shut.") == [("tailgate", "shut")])
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


def test_the_hardware_keeps_being_named():
    print("\n=== a restraint that is never named is not drawn ===")
    for _t, _want in (("Dan catches her and cuffs her wrists behind her back.", "cuffs"),
                      ("Dan locks steel handcuffs on her.", "steel handcuffs"),
                      ("Jon locks a chain around her waist.", "chain"),
                      ("Her wrists are bound with rope.", "rope"),
                      ("Dan buckles the leather collar on.", "leather collar"),
                      ("Dan fits a blindfold over her eyes.", "blindfold")):
        check(f"named: {_t[:36]!r} -> {S.hardware_named(_t)!r}",
              S.hardware_named(_t) == _want)
    check("nothing named, nothing latched",
          not S.hardware_named("Mara walks to the window."))
    for _t, _want in (("Dan puts a duct tape gag on her.", "duct tape gag"),
                      ("Dan puts a tape gag on her.", "tape gag"),
                      ("Dan tapes her mouth shut.", "tape"),
                      ("Dan pushes a ball gag into her mouth.", "ball gag")):
        check(f"material kept: {_t[:34]!r} -> {S.hardware_named(_t)!r}",
              S.hardware_named(_t) == _want)
    check("the material beats an earlier verb",
          S.hardware_named("Dan gags her with duct tape.") == "duct tape")
    cl = S.restraint_sentence("handcuffs", [], ["Mara"])
    check("the object is named", "The handcuffs" in cl)
    check("...in one sentence", cl.count(".") == 1)
    check("...positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", cl, re.I))
    check("singular agrees",
          S.restraint_sentence("collar", [], ["Mara"]).startswith(" The collar stays"))
    check("plural agrees",
          S.restraint_sentence("cuffs", [], ["Mara"]).startswith(" The cuffs stay"))
    check("rope is tied, not closed",
          "tied and holding" in S.restraint_sentence("rope", [], ["Mara"]))
    check("...and cuffs with tape are still closed",
          "closed and fastened" in S.restraint_sentence("cuffs, duct tape", [], ["Mara"]))


def test_memory_is_asked_for_honestly():
    print("\n=== freeing VRAM asks for what the shot needs, not for everything ===")
    class _Lat:
        shape = (1, 16, 10, 96, 128)
        dtype = "float16"
    class _VAE:
        vae_dtype = "float16"
        def memory_used_decode(self, shape, dtype):
            return 2500 * shape[-1] * shape[-2] * 2
    lat = _Lat()
    need = S._decode_headroom(_VAE(), lat)
    check(f"the decode asks for a real number ({need:.0f} bytes)", 0 < need < 1e29)
    check("...with headroom over the estimate", need > _VAE().memory_used_decode(lat.shape, None))
    class _NoEstimate: pass
    class _Raises:
        vae_dtype = "float16"
        def memory_used_decode(self, shape, dtype): raise RuntimeError("no")
    check("no estimator falls back to the old behaviour",
          S._decode_headroom(_NoEstimate(), lat) == 1e30)
    check("a failing estimator does too", S._decode_headroom(_Raises(), lat) == 1e30)
    calls = []
    class _MM:
        @staticmethod
        def get_torch_device(): return "cpu"
        @staticmethod
        def free_memory(req, dev, keep_loaded=None): calls.append(req)
        @staticmethod
        def soft_empty_cache(*a): pass
    class _Inner:
        def memory_required(self, shape):
            import math
            return 2.0 * math.prod(shape) * 4
    class _Model: model = _Inner()
    _orig = S._runtime_module.mm
    S._runtime_module.mm = _MM()
    try:
        S._evict_all_but(_Model(), {"samples": lat})
        check(f"sampling asks for a real number ({calls[-1]:.0f} bytes)",
              0 < calls[-1] < 1e29)
        S._evict_all_but(_Model(), None)
        check("no latent falls back", calls[-1] == 1e30)
        class _Broken: model = object()
        S._evict_all_but(_Broken(), {"samples": lat})
        check("a model that cannot size itself falls back", calls[-1] == 1e30)
    finally:
        S._runtime_module.mm = _orig


def test_the_hold_needs_its_wearer_on_screen():
    print("\n=== cuffs are not described in a shot with nobody wearing them ===")
    for _b, _cast, _want in (
            ("Dan walks in and cuffs her wrists behind her back.", ["Mara", "Dan"], {"Mara"}),
            ("Dan locks the cuffs on Mara.", ["Dan", "Mara"], {"Mara"}),
            ("Mara cuffs Dan to the pipe.", ["Mara", "Dan"], {"Dan"}),
            ("Mara is handcuffed to the rail.", ["Mara"], {"Mara"})):
        check(f"wearer read: {_b[:38]!r}", S.restrained_by_beat(_b, _cast) == _want)
    check("the beat may open on the victim",
          S.restrained_by_beat("Mara runs for the door. Dan catches her and cuffs "
                               "her wrists.", ["Mara", "Dan"]) == {"Mara"})
    check("no agent named, nobody is excluded",
          S.restrained_by_beat("She is cuffed to the rail.", ["Mara", "Dan"])
          == {"Mara", "Dan"})


def test_the_hold_names_its_wearer_once():
    print("\n=== attributing the hardware costs one mention, not two ===")
    out = S.own_hold(S.RESTRAINT_HOLD, ["Mara"], ["Mara", "Dan"])
    check("the wearer is named", "Every restraint on Mara" in out)
    check("...exactly once", len(re.findall(r"\bMara\b", out)) == 1, )
    check("no unattached body is introduced", "the body" not in out.lower())
    check("everyone else is still excluded",
          "Everyone else in the shot has on exactly what their own entry lists" in out)
    # One person described: no ambiguity to resolve, so no words spent on it.
    check("a solo shot is left alone",
          S.own_hold(S.RESTRAINT_HOLD, ["Mara"], ["Mara"]) == S.RESTRAINT_HOLD)
    check("nobody wearing it, nothing added",
          S.own_hold(S.RESTRAINT_HOLD, [], ["Mara", "Dan"]) == S.RESTRAINT_HOLD)
    check("no hold, nothing to attribute", S.own_hold("", ["Mara"], ["Mara", "Dan"]) == "")
    # Two wearers still read as English.
    two = S.own_hold(S.RESTRAINT_HOLD, ["Mara", "Kate"], ["Mara", "Kate", "Dan"])
    check("two wearers are joined properly", "Every restraint on Mara and Kate" in two)


def test_the_shot_that_puts_hardware_on():
    print("\n=== putting the cuffs on is not wearing them ===")
    for _b in ("Dan catches her and cuffs her wrists.", "Dan handcuffs her.",
               "Dan locks the cuffs on her wrists.", "Dan puts the handcuffs on her.",
               "Dan snaps the cuffs shut.", "Dan straps her ankles together.",
               "Dan buckles the collar on."):
        check(f"staged: {_b[:40]!r}", S.restraint_going_on(_b))
    for _b in ("Mara pulls against the cuffs.", "Mara strains at her cuffs.",
               "The chains hang from the beam.", "She twists in the straps.",
               "Mara stands by the wall, her wrists cuffed.",
               "Mara is handcuffed to the rail.",
               "Her wrists are chained above her head.",
               "Mara walks to the window."):
        check(f"not staged: {_b[:40]!r}", not S.restraint_going_on(_b))
    # The clause: both ends, one sentence, nothing about holding still.
    cl = S.RESTRAINT_GOING_ON
    check("it names both ends", "first frame" in cl and "by the last" in cl)
    check("...in one sentence", cl.count(".") == 1)
    check("...positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", cl, re.I))
    check("...and asks no body to hold still",
          not re.search(r"\b(?:still|motionless|frozen)\b", cl, re.I))
    # It must not claim the hardware is already fastened, which is the whole fault.
    check("...and does not assert it is already on",
          "still fastened at the last frame" not in cl)


def test_a_machines_line_is_not_the_actors_line():
    print("\n=== a voice out of a television is not hers ===")
    sheet = "Mara: she, 30.\nDan: he, 41."
    for _b in ('Mara sits on the sofa. The TV says: "Storms tonight."',
               'The radio announces: "Line four is delayed."',
               'The intercom crackles: "Come to the desk."',
               'The television plays: "...and back after this."',
               'Mara watches the screen. The TV goes: "Breaking news."',
               'The answerphone plays: "Mara, Dan called you back."',
               'Mara puts the kettle on. The radio says: "All units."'):
        check(f"the machine has it: {_b[:40]!r}", S.speech_is_a_devices(_b, sheet))
    for _b in ('Mara says: "Look at that."',
               'Mara sits by the TV. She says: "Turn it up."',
               'The TV says: "Storms tonight." Mara says: "Again?"',
               'The TV says: "Storms." She whispers: "No."',
               'Dan asks: "Is it on?"',
               '"Turn it off," Mara mutters at the television.',
               'Mara watches the TV. "I hate this."',
               'Mara grabs the phone and says: "Get someone here."',
               'Mara picks up the phone and says: "Get someone here."',
               'Mara slams the phone down and says: "Get someone here."',
               'Dan turns off the radio and says: "Enough."',
               'Mara crosses the room, picks up the phone, and says: "Now."'):
        check(f"the person keeps it: {_b[:40]!r}", not S.speech_is_a_devices(_b, sheet))
    # No quote at all is not a device line either -- there is no line to reassign.
    check("no line, nothing to attribute",
          not S.speech_is_a_devices("Mara watches the TV.", sheet))
    # The clause.
    cl = S.device_voice_clause('The TV says: "Storms tonight."')
    check("the machine is named", "the TV's" in cl)
    check("...as the author spelled it", "tv's" not in cl)
    check("...and the listeners are given something to do",
          "let it play" in cl and "listening" in cl)
    check("...without being frozen to do it",
          "still" not in cl and "mouths closed" in cl)
    check("...and it is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", cl, re.I))
    check("no machine, no clause", S.device_voice_clause("Mara says: 'Hello.'") == "")


def test_a_shifted_workflow_is_named_not_rendered():
    print("\n=== widget values out of position are caught, not guessed at ===")
    opts = S.combo_options(S.H3LongVideos.INPUT_TYPES())
    check("the choice widgets are found", "sampler_name" in opts and "resolution" in opts)
    bad = S.misaligned_widgets(
        {"sampler_name": "beta", "scheduler": 48, "resolution": 0.7}, opts)
    check("a scheduler in the sampler slot is caught",
          any(n == "sampler_name" for n, _, _ in bad))
    check("a seed in the scheduler slot is caught",
          any(n == "scheduler" for n, _, _ in bad))
    check("a number in the resolution slot is caught",
          any(n == "resolution" for n, _, _ in bad))
    # A healthy workflow must pass untouched, or this fires on everybody.
    good = {k: v[0] for k, v in opts.items()}
    check("valid choices raise nothing", S.misaligned_widgets(good, opts) == [])
    check("a widget that was not sent is not judged",
          S.misaligned_widgets({}, opts) == [])
    msg = S.alignment_error(bad)
    check("it names the cause", "restored by POSITION" in msg)
    check("...and the fix", "Fix node (recreate)" in msg)
    check("...and clears the model and the prompt of blame",
          "Nothing is wrong with the model or the prompt" in msg)
    check("nothing wrong, no message", S.alignment_error([]) == "")


def test_what_is_exposed_is_not_also_removed():
    print("\n=== a garment the beat reveals is not one it takes off ===")
    sc = "Mara: she, 22, a grey coat, a navy jumper."
    for _b in ("Mara pulls off her coat to show the jumper underneath.",
               "Mara pulls off her coat, showing the jumper underneath.",
               "Mara pulls off her coat to reveal the jumper."):
        check(f"only the coat comes off: {_b[:38]!r}",
              S.infer_removals(_b, sc) == ["coat"])
        check(f"...and the jumper is the one exposed: {_b[:26]!r}",
              S.exposed_by(_b, sc) == ["jumper"])
    # Both really coming off is still both coming off.
    check("two removals still read as two",
          S.infer_removals("Mara pulls off her coat and her jumper.", sc)
          == ["coat", "jumper"])
    # The sentence that tells the shot what fills the space.
    cl = S.reveal_clause(["panties"])
    check("the under layer is named", "The panties underneath" in cl)
    check("...as what is seen there", "what shows there now" in cl)
    check("...and as unchanged on the body", "on and unchanged" in cl)
    check("...in one sentence", cl.count(".") == 1)
    check("...positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", cl, re.I))
    check("singular agrees", "underneath is what shows" in S.reveal_clause(["jumper"]))
    check("nothing revealed, nothing said", S.reveal_clause([]) == "")
    # revealed_by is what feeds it: the cover has come off, so what was under shows.
    covers = {"panties": "shorts"}
    check("uncovered by the shorts coming off",
          S.revealed_by(covers, ["shorts"]) == ["panties"])
    check("...and not while they are still on", S.revealed_by(covers, []) == [])


def test_underwear_goes_under():
    print("\n=== underwear is not drawn through the clothes over it ===")
    got = S.implied_layers("Mara: she, 22, blue denim shorts, white top, panties, "
                           "a chastity belt.")
    check("panties go under the shorts", got.get("panties") == "shorts")
    check("...and so does the belt", got.get("chastity belt") == "shorts")
    # The outer half has to include what people actually write for legs.
    for _outer in ("jeans", "tights", "leggings", "trousers", "a skirt"):
        _sc = f"Mara: she, 22, {_outer}, a chastity belt, a top."
        check(f"a belt goes under {_outer!r}", S.implied_layers(_sc).get("chastity belt"))
    for _spelling in ("a chastity belt", "chastity-belt", "a chastity device",
                      "a chastity cage", "a steel chastity-belt"):
        _sc = f"Mara: she, 22, blue jeans, {_spelling}, a top."
        check(f"hidden however it is written: {_spelling!r}",
              any("chastity" in k for k in S.implied_layers(_sc)))
    check("an ordinary belt is not underwear",
          not S.implied_layers("Mara: she, 22, jeans, a belt."))
    check("his jeans do not cover her belt",
          S.implied_layers("McKenna: she, a chastity belt.\nDan: he, blue jeans.") == {})
    for _order in ("McKenna: she, chastity belt, blue jeans shorts.\nDan: he, blue jeans.",
                   "Dan: he, blue jeans.\nMcKenna: she, chastity belt, blue jeans shorts."):
        check(f"her own shorts, whichever line comes first: {_order[:18]!r}",
              S.implied_layers(_order) == {"chastity belt": "shorts"})
    check("the cover is the head noun",
          S.implied_layers("Mara: she, chastity belt, blue jeans shorts.")
          == {"chastity belt": "shorts"})
    check("a dress covers both bra and knickers",
          S.implied_layers("Mara: a summer dress, a bra and knickers underneath.")
          == {"knickers": "dress", "bra": "dress"})
    check("a skirt covers a thong",
          S.implied_layers("Mara: a skirt, a thong, boots.") == {"thong": "skirt"})
    check("trousers do not cover a bra",
          S.implied_layers("Mara: jeans, a bra, boots.") == {})
    check("a top does not cover panties",
          S.implied_layers("Mara: a white top, panties.") == {})
    check("underwear alone stays visible",
          S.implied_layers("Mara: she, 22, panties and a bra.") == {})
    check("no underwear listed, nothing to hide",
          S.implied_layers("Mara: jeans and a t-shirt.") == {})
    # hidden_layers is what acts on it: still under something that has not come off.
    covers = {"panties": "shorts"}
    check("hidden while the shorts are on", S.hidden_layers(covers, []) == ["panties"])
    check("...and back once they come off", S.hidden_layers(covers, ["shorts"]) == [])


def test_pulling_something_down_is_not_falling():
    print("\n=== 'pulls down her shorts' is not a body hitting the floor ===")
    for _t in ("She stands and pulls down her shorts.",
               "She pulls her shorts down to show the thong.",
               "Mara pulls down the blind.", "Dan pulls down the shutter.",
               "Dan pulls her jacket down off her shoulders.",
               "Dan throws the keys down.", "She drags the chair over.",
               "He pushes the door open."):
        check(f"not a fall: {_t[:40]!r}", not S.falls_in(_t))
    # What goes down has to be a person -- named, then the direction.
    for _t in ("Dan pushes her down onto the floor.", "Dan knocks him down.",
               "Dan throws her to the ground.", "Dan pulls her down.",
               "Dan shoves her over.", "Dan drags Mara down."):
        check(f"still a fall: {_t[:40]!r}", S.falls_in(_t))
    check("the passive still reads", S.falls_in("She is pushed to the floor."))
    # The body's own verbs are untouched by any of this.
    for _t in ("Mara trips and falls to the floor.", "Mara collapses.",
               "Mara loses her balance.", "Kate slumps against the wall."):
        check(f"own fall: {_t[:40]!r}", S.falls_in(_t))


def test_a_fall_says_what_takes_the_landing():
    print("\n=== a falling body is told what catches it ===")
    for _n in ("FALL_HOLD", "FALL_HOLD_FREE"):
        _h = getattr(S, _n)
        check(f"{_n} names what takes the landing",
              "shoulder, hip or side takes the landing" in _h)
        # The legs are the part that grew, and they had no job in the old clause.
        check(f"{_n} gives the legs something to do", "legs fold" in _h.lower())
        check(f"{_n} is one sentence", _h.count(".") == 1)
        check(f"{_n} is positively phrased",
              not re.search(r"\bno\b|\bnot\b|\bnever\b", _h, re.I))
        check(f"{_n} counts nothing",
              not re.search(r"\b(?:two|both|pair|exactly|only|single)\b", _h, re.I))
    # A bound fall keeps what it always said: the hold does not give way to break it.
    check("the bound clause still holds the hardware",
          "fastened limbs stay fastened" in S.FALL_HOLD
          and "arms staying in the hold" in S.FALL_HOLD)
    # The free clause must NOT claim the arms are held -- they are not.
    check("the free clause claims no hold",
          "hold" not in S.FALL_HOLD_FREE and "fastened" not in S.FALL_HOLD_FREE)
    check("...and is the shorter of the two",
          len(S.FALL_HOLD_FREE.split()) < len(S.FALL_HOLD.split()))


def test_the_look_goes_where_the_beat_says():
    print("\n=== the eyes go where the beat put them ===")
    for _t, _want in (("Mara sits on the sofa looking at the TV.", "TV"),
                      ("She stares at the television screen.", "television screen"),
                      ("Mara glances at the clock and stands up.", "clock"),
                      ("She is watching the TV.", "TV"),
                      ("He peers into the box.", "box"),
                      ("Mara looks over at the window, then back.", "window"),
                      ("She studies the map on the wall.", "map"),
                      ("Mara looks down at the phone in her hand.", "phone")):
        check(f"target read: {_t[:38]!r} -> {S.look_target(_t)!r}",
              S.look_target(_t) == _want)
    for _t in ("Mara looks tired.", "She is looking for the keys.",
               "A look of fear crosses her face.", "He looks up.",
               "It looks like rain.", "Mara walks to the window.",
               "She takes a long look around."):
        check(f"no target: {_t[:38]!r}", not S.look_target(_t))
    for _t in ("Mara looks at her.", "She watches him.", "He stares at them."):
        check(f"a pronoun is not a target: {_t[:32]!r}", not S.look_target(_t))
    # The sentence.
    cl = S.gaze_hold("TV")
    check("the clause is one sentence", cl.count(".") == 1)
    check("...and names the thing", "the TV" in cl)
    check("...and is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", cl, re.I))
    check("...and says nothing about the camera",
          not re.search(r"\bcamera|lens|frame\b", cl, re.I))
    check("...and names nobody",
          not re.search(r"\b(?:she|he|her|his|they|their)\b", cl, re.I))
    check("nothing named, nothing said", S.gaze_hold("") == "")


def test_an_object_tag_leaves_with_its_object():
    print("\n=== a scrubbed object takes its picture tag with it ===")
    for text, toks in (
        ("Mara: <Picture 1>, she, 30, wearing a chastity belt <Picture 2>, and a coat.",
         ["chastity belt"]),
        ("Nora: <Picture 1>, 34, she, wearing a silver locket <Picture 2> and boots.",
         ["locket"]),
        ("She is wearing a silver locket <Picture 2>.", ["locket"]),
        ("Nora: <Picture 1>, 34, she, a silver locket <Picture 2>, green jacket.",
         ["locket"]),
    ):
        out = S.scrub_removed(text, toks)
        check(f"the tag goes too: {toks[0]!r} in {text[:30]!r}", "<Picture 2>" not in out)
    for text, toks in (
        ("Mara: <Picture 1>, she, 30, wearing a chastity belt <Picture 2>, and a coat.",
         ["chastity belt"]),
        ("Nora: <Picture 1>, 34, she, green jacket.", ["jacket"]),
        ("Nora: <Picture 1> wearing a green jacket.", ["jacket"]),
    ):
        check(f"the person keeps theirs: {text[:34]!r}",
              "<Picture 1>" in S.scrub_removed(text, toks))
    # What is left has to read as English, or the leftovers describe something.
    check("no stranded conjunction at the front of a list",
          S.scrub_removed("Mara: <Picture 1>, she, 30, a belt <Picture 2>, and a coat.",
                          ["belt"]).strip()
          == "Mara: <Picture 1>, she, 30, a coat.")
    check("a sentence emptied to a subject and copula is dropped",
          S.scrub_removed("She is wearing a silver locket <Picture 2>.",
                          ["locket"]).strip() == "")
    check("...while a real sentence survives",
          "green jacket" in S.scrub_removed(
              "Nora: <Picture 1>, 34, she, a locket <Picture 2> and green jacket.",
              ["locket"]))


def test_scenery_does_not_move_the_wrists():
    print("\n=== a light fitting is not a limb ===")
    for _t in ("A bare concrete room, one bulb overhead. "
               "McKenna, wrists handcuffed behind her back.",
               "Strip lights overhead. She is cuffed behind her back.",
               "The cable is stretched up the wall. She is cuffed behind her back."):
        check(f"scenery does not win: {_t[:40]!r}",
              S.limb_anchor(_t) == "behind the back")
    # Scenery alone anchors nothing at all.
    for _t in ("One bulb overhead. She kneels.",
               "A lamp hangs over the table.",
               "The rope is stretched up to the beam."):
        check(f"no body, no anchor: {_t[:38]!r}", not S.limb_anchor(_t))
    # ...while the limb readings it was built for still read.
    check("a fastening participle still reads",
          S.limb_anchor("cuffed above her head") == "above the head")
    check("a limb still reads", S.limb_anchor("her wrists overhead") == "above the head")
    check("stretched arms still read",
          "above the head" in S.limb_anchor("Her arms are stretched up and locked to the rail."))
    check("and the attachment point comes with it",
          S.limb_anchor("handcuffed above her head to the bed frame")
          == "above the head, at the bed frame")
    # The pose the shot is actually given, which is what the model reads.
    check("the pose follows the hardware",
          "behind the body" in S.pose_clause(
              S.limb_anchor("one bulb overhead. cuffed behind her back")))
    for _t in ("Crates stacked to the sides.", "The doors spread wide.",
               "A rope at her waist.", "The door closes behind her back.",
               "A table in front of her body."):
        check(f"scenery anchors nothing: {_t[:34]!r}", not S.limb_anchor(_t))
    # Legs are not arms. This one moved the wrists to wherever the legs were.
    check("legs do not move the wrists", not S.limb_anchor("Her legs spread wide."))
    check("...even beside the real position",
          S.limb_anchor("Her legs spread wide. Her wrists are cuffed at her waist.")
          == "at the waist")
    # ...while every position still reads when a limb or a fastening is there.
    for _t, _want in (("Arms spread wide, chained to the wall.", "out to the sides"),
                      ("Her wrists are cuffed at her waist.", "at the waist"),
                      ("Her hands are cuffed in front of her body.", "in front of the body"),
                      ("Mara is cuffed behind her back.", "behind the back")):
        check(f"still reads {_want!r}",
              S.limb_anchor(_t).split(", at the")[0] == _want)


def test_fastened_limbs_keep_their_anchor():
    print("\n=== where the cuffs are held, not just that they are shut ===")
    for _t in ("Mara is handcuffed above her head to the bed frame.",
               "Her wrists are cuffed above her head.",
               "Mara is cuffed behind her back.",
               "Her arms are stretched up and locked to the rail.",
               "Cuffed to the radiator."):
        check(f"anchor read: {_t[:38]!r}", S.limb_anchor(_t))
    check("both halves when both are written",
          S.limb_anchor("handcuffed above her head to the bed frame")
          == "above the head, at the bed frame")
    check("the body-relative half alone",
          S.limb_anchor("cuffed above her head") == "above the head")
    check("the attachment point alone",
          S.limb_anchor("cuffed to the radiator") == "at the radiator")
    check("a plural anchor point is read",
          "at the posts" in S.limb_anchor("chained to the posts"))
    # A pose is not an anchor, and neither is an unrelated "to the".
    for _t in ("Mara kneels on the floor.", "Mara walks to the window.",
               "He hands her the keys to the car.", "She looks up at the ceiling."):
        check(f"no anchor: {_t[:34]!r}", not S.limb_anchor(_t))
    cl = S.restraint_sentence("cuffs", [], ["Mara"], anchor="above the head")
    check("the position is not buried in the hold",
          "above the head" not in cl)
    check("...and the pose says it as a body",
          "wrists together above the head" in S.pose_clause("above the head"))
    check("the clause is one sentence", cl.count(".") == 1)
    check("...and is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", cl, re.I))
    check("...and says nothing about the body holding still",
          not re.search(r"\b(?:motionless|frozen|does not move)\b", cl, re.I))
    check("nothing anchored, nothing added",
          "holding the wrists" not in S.restraint_sentence("cuffs", [], ["Mara"]))
    # Framing tight enough to lose the anchor, which is what the next shot inherits.
    for _t in ("A close shot of her face.", "Close-up on her hands.",
               "Tight on the lock.", "Her face fills the frame."):
        check(f"tight frame: {_t[:32]!r}", S.tight_framing(_t))
    for _t in ("Mara turns her head.", "A wide shot of the room.",
               "The camera pulls back."):
        check(f"not tight: {_t[:32]!r}", not S.tight_framing(_t))


def test_only_a_beat_with_a_person_gets_a_mouth():
    print("\n=== a mouth clause needs a mouth to be about ===")
    sheet = "Kate: she, 30, red coat.\nDan: he, 41."
    for _t in ("Kate walks to the window.", "She lies still.",
               "Dan and Kate stand by the crate.", "The man waits by the door.",
               "Somebody moves behind the glass."):
        check(f"a person is on screen: {_t[:36]!r}",
              S.beat_puts_somebody_on_screen(_t, sheet))
    for _t in ("Rain on the corrugated roof.", "The gate stands open.",
               "A low hum comes off the strip light.", "An empty yard.",
               "Wind through the fence wire."):
        check(f"nobody on screen: {_t[:36]!r}",
              not S.beat_puts_somebody_on_screen(_t, sheet))
    # A pronoun carries it without any sheet at all.
    check("a pronoun needs no sheet", S.beat_puts_somebody_on_screen("She lies still.", ""))
    check("an unlisted name is not guessed at",
          not S.beat_puts_somebody_on_screen("Kate walks to the window.", ""))
    # The sentence itself: appended, never leading, and positively phrased.
    check("the clause is one short sentence",
          S.MOUTH_HOLD.count(".") == 1 and len(S.MOUTH_HOLD.split()) <= 12)
    check("...and is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b|\bnobody\b", S.MOUTH_HOLD, re.I))
    check("...and starts with a space, so it appends",
          S.MOUTH_HOLD.startswith(" "))


def test_a_tag_names_the_socket_it_is_wired_to():
    print("\n=== <Picture N> means ref_image_N ===")
    check("no gap, nothing to do",
          S.renumber_reference_tags("Mara: <Picture 1>, Dom: <Picture 2>.", [1, 2])
          == "Mara: <Picture 1>, Dom: <Picture 2>.")
    check("a gap is closed up",
          S.renumber_reference_tags("Mara: <Picture 1>, a locket <Picture 3>.", [1, 3])
          == "Mara: <Picture 1>, a locket <Picture 2>.")
    check("one socket, not the first",
          S.renumber_reference_tags("Mara: <Picture 2>.", [2]) == "Mara: <Picture 1>.")
    check("the last socket alone",
          S.renumber_reference_tags("Mara: <Picture 4>.", [4]) == "Mara: <Picture 1>.")
    check("all four, out of a gap",
          S.renumber_reference_tags("<Picture 2> <Picture 4>", [2, 4])
          == "<Picture 1> <Picture 2>")
    check("a tag on an empty socket is left for the stripper",
          S.renumber_reference_tags("Mara: <Picture 3>.", [1]) == "Mara: <Picture 3>.")
    check("...and is named", S.unwired_reference_tags("Mara: <Picture 3>.", [1]) == [3])
    check("nothing wired, every tag is named",
          S.unwired_reference_tags("<Picture 1> <Picture 2>", []) == [1, 2])
    check("all wired, nothing to name",
          S.unwired_reference_tags("<Picture 1> <Picture 3>", [1, 3]) == [])
    # The spelling the rest of the file accepts.
    check("spacing and underscores are read the same",
          S.renumber_reference_tags("<picture_3> < Picture 3 >", [1, 3])
          == "<Picture 2> <Picture 2>")
    check("no tags, no change", S.renumber_reference_tags("Mara: she, 30.", [1, 3])
          == "Mara: she, 30.")
    check("empty text survives", S.renumber_reference_tags("", [1, 3]) == "")


def test_an_exit_and_a_shut_door_disagree():
    print("\n=== leaving the van, with the doors shut ===")
    for _t in ("Mara and Dom step out of the van with its rear doors closed.",
               "Mara and Dom get out of the van and stand behind it.",
               "They climb out of the back of the van.",
               "Dom exits the van.",
               "They pile out of the truck."):
        check(f"an exit: {_t[:40]!r}", S.exits_vehicle(_t))
    for _t in ("Mara and Dom walk out from behind a van with closed rear doors.",
               "Mara and Dom stand behind the van, its rear doors closed.",
               "Dom walks out of the warehouse.",
               "Mara steps out of the shower.",
               "Dom looks back at the yard."):
        check(f"not an exit: {_t[:40]!r}", not S.exits_vehicle(_t))


def test_a_held_thing_is_not_also_heard_moving():
    print("\n=== the shot is not told to hold a door and to sound like one swinging ===")
    b = "Mara and Dom stand behind a van with closed rear doors."
    check("a door mention alone is heard swinging",
          "a door on its hinges" in S.sounds_for(b))
    check("...but not while the shot is holding it shut",
          "a door on its hinges" not in S.sounds_for(b, held=["door"]))
    check("the other sounds survive", S.sounds_for(b, held=["door"]) == ["an engine outside"])
    # A beat that WORKS the door is asking for exactly that swing, and keeps it.
    check("a staged opening still sounds like one",
          "a door on its hinges" in S.sounds_for("Mara opens the van doors."))
    check("holding nothing changes nothing", S.sounds_for(b, held=[]) == S.sounds_for(b))
    check("an unrelated hold drops nothing",
          S.sounds_for(b, held=["curtain", "lid"]) == S.sounds_for(b))


def test_a_staged_change_names_both_ends():
    print("\n=== which end of the action is which ===")
    check("opening runs shut to open",
          S.direction_anchor(S.state_changes("Mara opens the van doors."))
          == " The doors are shut at the first frame and open by the last.")
    check("shutting runs open to shut",
          S.direction_anchor(S.state_changes("Dom slams the tailgate shut."))
          == " The tailgate is open at the first frame and shut by the last.")
    check("locking shuts", S.state_changes("Dom locks the hatch.") == [("hatch", "shut")])
    check("lifting opens", S.state_changes("Mara lifts the lid.") == [("lid", "open")])
    for _t in ("Mara pulls the curtains.", "Dom draws the blinds.",
               "Mara slides the door.", "Dom swings the gate."):
        _ch = S.state_changes(_t)
        check(f"no direction guessed: {_t[:30]!r}",
              _ch and _ch[0][1] is None and S.direction_anchor(_ch) == "")
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
    check("an action outside the table gets nothing",
          S.sounds_for("The camera pushes in on her face.") == [])
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
    lens = "Medium shadows, shallow depth of field. Medium focus."
    check("'depth of field' is a lens, not a location", S.room_tone(lens) == "")
    check("...so is 'field of view'", S.room_tone("Wide lens, deep field of view.") == "")
    check("a real field is still open air",
          "open air" in S.room_tone("They cross an open field towards the barn."))
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
    _ceil = S.align_frame_count(10 * S.H3_FPS)
    one = "Maya walks to the window."
    two = "Jon walks in and takes her jacket off."
    check("a one-action beat no longer asks for four and a half seconds",
          S.beat_seconds(one) <= 3.2)
    check("...and a two-action beat is under six", S.beat_seconds(two) <= 5.5)
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
    for fc in (73, 124, 226):
        want = int(round(fc * 44100 / S.H3_FPS))
        check(f"{fc}f wants {want} samples at 44.1k", want > 0)
    check("the audio grid ignores a different fps",
          S.temporal_shape(73, 30) == S.temporal_shape(73, 24))


def test_chain_is_rigid():
    print("\n=== steel does not behave like rope ===")
    for _t in ("Jon locks a chain around her waist.", "padlocked at the back",
               "Wrists handcuffed behind back.", "ankles shackled together",
               "steel cuffs", "a spreader bar", "hogcuffed on the floor"):
        check(f"rigid hardware: {_t[:32]!r}", S.rigid_hardware(_t))
    # Rope, tape and straps DO flex -- claiming they hold a straight line is wrong.
    for _t in ("a rope around her wrists", "her mouth taped shut",
               "a leather strap", "Maya lies still."):
        check(f"not rigid: {_t[:32]!r}", not S.rigid_hardware(_t))
    check("the clause is one sentence", S.CHAIN_HOLD.count(".") == 1)
    check("...it keeps the links the same size", "links keeping their size" in S.CHAIN_HOLD)
    check("...holds the run taut", "run between them taut" in S.CHAIN_HOLD)
    check("...impersonal and positive",
          not re.search(r"\b(?:she|he|her|his|they|no|not|never)\b", S.CHAIN_HOLD, re.I))
    for _c, _n in ((S.CHAIN_HOLD, "chain"), (S.RESTRAINT_HOLD, "restraint"),
                   (S.TURN_HOLD, "turn"), (S.FALL_HOLD, "fall")):
        check(f"the {_n} clause does not tell the body to stop",
              not re.search(r"\bbefore it stops\b|\bthe body stays\b|\bholds? still\b|"
                            r"\bmotionless\b|\bdoes not move\b|\bstays put\b", _c, re.I))
    check("the chain clause carries the restraint guarantee itself",
          "stays closed and fastened as it was put on" in S.CHAIN_HOLD and "as it was put on"
          in S.CHAIN_HOLD)
    for _t in ("forcing her into a squat", "chained kneeling on the floor",
               "hogcuffed on the floor", "bent over the table", "spread-eagled",
               "locked crouching", "on her knees, chained to the wall"):
        check(f"a forced position: {_t[:34]!r}", S.forced_pose(_t))
    for _t in ("Maya walks to the window.", "Jon locks a chain around her waist.",
               "Maya lies still."):
        check(f"no position forced: {_t[:34]!r}", not S.forced_pose(_t))
    check("the pose clause says the metal is at full length",
          "drawn to its full length" in S.CHAIN_POSE_HOLD)
    check("...that the position keeps", "the position that keeps" in S.CHAIN_POSE_HOLD)
    check("...and carries the restraint guarantee too",
          "stays closed and fastened as it was put on" in S.CHAIN_POSE_HOLD)
    check("...while leaving the body free to act",
          "strains against it" in S.CHAIN_POSE_HOLD)
    check("...and telling it to hold still nowhere",
          not re.search(r"\bstill\b|\bmotionless\b|\bdoes not move\b|\bbefore it stops\b",
                        S.CHAIN_POSE_HOLD, re.I))
    check("...positively phrased", not re.search(r"\b(?:no|not|never|without)\b",
                                                 S.CHAIN_POSE_HOLD, re.I))


def test_falling_bound():
    print("\n=== a bound body goes down without catching itself ===")
    for _t in ("She falls forward onto the floor.", "Kate collapses.",
               "She loses her balance and goes down.", "Kate topples sideways.",
               "She stumbles and hits the floor.", "Kate slumps against the wall.",
               "Dan pushes her over.", "Dan knocks her down.",
               "Dan throws her to the ground.", "Dan pulls her down."):
        check(f"fall seen: {_t[:34]!r}", S.falls_in(_t))
    for _t in ("Kate walks to the window.", "She lies still.",
               "Dan drops the keys.", "Dan sets the crate down.",
               "Kate turns her head."):
        check(f"not a fall: {_t[:34]!r}", not S.falls_in(_t))
    check("the clause is one sentence", S.FALL_HOLD.count(".") == 1)
    check("...keeping the fastening through the fall",
          "fastened limbs stay fastened" in S.FALL_HOLD)
    check("...keeping the arms in the hold",
          "arms staying in the hold" in S.FALL_HOLD)
    check("...and naming what takes the landing",
          "shoulder, hip or side takes the landing" in S.FALL_HOLD)
    check("...impersonal and positive",
          not re.search(r"\b(?:she|he|her|his|they|no|not|never)\b",
                        S.FALL_HOLD, re.I))


def test_turning_around():
    print("\n=== turning shows a surface the keyframe never pinned ===")
    for _t in ("She turned around.", "Kate turns to face him.",
               "She looks back over her shoulder.", "Kate rolls onto her side.",
               "The camera moves round to show her back."):
        check(f"turn seen: {_t[:32]!r}", S.turns_in(_t))
    for _t in ("Kate walks to the window.", "She lies still.", "Dan pulls off her coat."):
        check(f"no turn: {_t[:32]!r}", not S.turns_in(_t))
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
    check("one sound, closed", S.sound_clause(["footsteps"], only=True)
          == " The only sound is footsteps.")
    check("two sounds, closed", S.sound_clause(["footsteps", "a door"], only=True)
          == " The only sounds are footsteps and a door.")
    check("three sounds, closed",
          S.sound_clause(["footsteps", "a door", "rain"], only=True)
          == " The only sounds are footsteps, a door and rain.")
    check("a speaking shot is left open",
          S.sound_clause(["footsteps"]) == " It sounds like footsteps.")
    check("nothing heard, nothing said", S.sound_clause([], only=True) == "")
    for _p in (True, False):
        cl = S.sound_clause(["footsteps", "a door"], only=_p)
        check(f"positively phrased (only={_p})",
              not re.search(r"\b(?:no|not|never|without|nobody|silent)\b", cl, re.I))


def test_hardware_belongs_to_somebody():
    print("\n=== a restraint hold names whose ===")
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
    check("...once, not twice", len(re.findall(r"\bNora\b", two)) == 1)
    check("...and no unattached body is introduced", "the body" not in two.lower())
    check("...while everyone else is still excluded",
          "Everyone else in the shot has on exactly what their own entry lists" in two)
    check("...pinning the other to their own entry",
          "exactly what their own entry lists" in two)
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
    three = ("Nora: 34, she, red hair.\n"
             "Kate: 27, she, blonde.\n"
             "Dan: 41, he, dark hair")
    two = "Nora: 34, she, red hair.\nDan: 41, he, dark hair"
    for _sheet, _label, _beat, _prev, _want in (
            (three, "both named outright", "Dan hands Nora the spanner.", [],
             ["Nora", "Dan"]),
            (three, "the last beat narrows it", "Dan takes her coat off.", ["Nora"],
             ["Dan", "Nora"]),
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


def test_underwear_is_a_garment_with_a_place_on_the_body():
    """REPORTED: she undresses in the bathroom, gets in the shower, and her thong is
    back in the next beat.

    Three separate things had to be true for that, and all three were.

    One: region_of could not place ANY lower-body underwear. The torso row of the
    region table has listed a bra since the day it was written -- that is the report
    it exists for, "a bra coming back on somebody topless" -- and the leg row never
    got its counterpart. A garment with no region latches no bare region, so the
    hips had no sentence in that shot or in any shot after it, and an unspecified
    region is filled by the model's own prior.

    Two: a beat saying somebody is NAKED takes each worn garment off by looking its
    region up, so the one garment it could not place stayed "worn" in the state while
    the text said she was nude.

    Three: bare_hold went silent for any region the SHEET named a layer under --
    whether or not that layer had come off as well. A full strip is the case it
    matters most in, and it was the one case it could not speak in."""
    print("\n=== underwear has a place on the body ===")
    for _g in ("thong", "panties", "knickers", "a black g-string", "briefs",
               "boxers", "boxer shorts", "underwear", "undies", "jockstrap"):
        check(f"placed on the lower body: {_g!r}", S.region_of(_g) == "legs")
    # The torso half, which already worked, stays where it was.
    for _g in ("bra", "bralette", "camisole", "vest"):
        check(f"...and the upper body is unchanged: {_g!r}", S.region_of(_g) == "torso")
    check("a chastity belt is not given a region", S.region_of("chastity belt") == "")
    _covers = {"thong": "skirt"}
    check("silent while the thong is still on",
          S.bare_hold(["legs"], _covers, "", []) == "")
    check("...and says so once the thong has come off too",
          "legs are bare from the hip down" in S.bare_hold(["legs"], _covers, "",
                                                           ["skirt", "thong"]))
    check("a garment still WORN over the region keeps it quiet",
          S.bare_hold(["legs"], {}, "denim skirt", ["thong"]) == "")
    check("the bra half was suppressed the same way",
          "chest, shoulders and arms are bare" in S.bare_hold(
              ["torso"], {"bra": "shirt"}, "", ["shirt", "bra"]))
    _said = S.bare_hold(["legs"], {}, "", [], body="a woman's body")
    check("the clause carries no negation",
          not re.search(r"\b(?:no|nothing|not|never|without)\b", _said, re.I))
    check("...and still names the body", "a woman's body" in _said)


def test_how_underwear_actually_comes_off():
    """The other half of the same report: the removal that was never read at all.

    Eight of twenty ways people write underwear coming off did nothing -- the
    garment stayed in the sheet, the sheet is re-stamped into every shot, so it
    came back on and stayed on. One of them was worse than nothing: "lets the thong
    fall to the floor" matched the RESTORE vocabulary, so the node read the author's
    removal and enacted its opposite.

    The verbs that carry these are kept out of _STRIP_VERB on purpose -- that list
    also builds the displacement reader, where a bare "gets" or "pushes" reads "gets
    down on her knees" and "pushes the door open" as garments being moved."""
    print("\n=== how underwear actually comes off ===")
    _sc = ("A tiled bathroom with a wet floor.\n"
           "Kate: she, 28, a denim skirt, a black thong.")
    for _b in (
            "Kate hooks her thumbs in the thong and steps out of it.",
            "Kate tugs the thong down and steps clear of it.",
            "Kate gets out of the thong.",
            "Kate shimmies out of the thong.",
            "Kate wriggles out of the thong.",
            "Kate pushes the thong off her hips.",
            # down PAST the hips. Down on its own is still a displacement.
            "Kate hooks her thumbs into the waistband and pushes the thong down her legs.",
            "Kate slides the thong down to the floor.",
            # and onto the floor, whatever verb carried it there
            "Kate drops the thong on the floor.",
            "Kate lets the thong fall to the floor.",
            # ...and the ones that already worked, which must keep working
            "Kate takes off the thong.",
            "Kate peels off the thong.",
            "Kate steps out of her thong.",
            "Kate slides the thong down and off.",
            "Kate kicks the thong away.",
            "Kate slips the thong off and drops it on the tiles."):
        check(f"comes off: {_b[29:72]!r}", S.infer_removals(_b, _sc) == ["thong"])
    check("let fall to the floor is not a restore",
          S.restored_garments("Kate lets the thong fall to the floor.", _sc) == [])
    check("...and let fall on its own still puts a lifted skirt back",
          S.puts_it_back("Kate lets it fall."))
    check("...as does naming it",
          S.restored_garments("Kate lets the skirt fall.", _sc) == ["denim skirt"])
    check("a restore is never keyed to the room",
          S.restored_garments("Kate drops the thong on the floor.", _sc) == [])
    for _b in ("Kate pulls the skirt down.", "Kate eases the thong down her thighs.",
               "Kate pushes the skirt down over her hips.",
               "Kate lets the skirt fall back into place."):
        check(f"not a removal: {_b[5:48]!r}", S.infer_removals(_b, _sc) == [])
    check("...and pulling the skirt down is read as moved",
          S.displaced_garments("Kate pulls the skirt down.", _sc)
          == [("denim skirt", "pulled down")])
    _room = ("A tiled bathroom. A glass shower, a bath, a wooden stool, a mirror.\n"
             "Kate: she, 28, a denim skirt, a black thong.")
    for _b in ("Kate steps out of the shower.", "Kate climbs out of the bath.",
               "Kate gets off the stool.", "Kate kicks the stool away.",
               "Kate drops her towel onto the stool."):
        check(f"a fixture is not undressed: {_b[5:44]!r}",
              S.infer_removals(_b, _room) == [])
    check("...and the garment in the same room still comes off",
          S.infer_removals("Kate steps out of the thong.", _room) == ["thong"])
    check("a pronoun with two garments in the clause is left alone",
          S.infer_removals("Kate touches the skirt and the thong and steps out of it.",
                           _sc) == [])
    check("...and does not reach back past a full stop",
          S.infer_removals("Kate hangs up the skirt. She picks up the brush and "
                           "drops it on the floor.", _sc) == [])
    # Named in full one clause, by head the next: one garment, taken off once.
    _belt = "Nora: 34, a steel chastity belt, green jacket, boots."
    check("the same garment is not taken off twice",
          S.infer_removals("Dan unlocks the chastity belt and takes it off.",
                           _belt) == ["belt"])
    _two = "Kate: she, 28, a wool jumper, a denim skirt.\nDan: he, 40, a blue shirt."
    for _b, _want in (("Kate pulls off the jumper and she sits down.", ["jumper"]),
                      ("Kate pulls off the jumper and he looks away.", ["jumper"]),
                      ("Kate takes the skirt off and they both laugh.", ["skirt"]),
                      ("Dan pulls off the shirt and throws it down.", ["shirt"])):
        check(f"no pronoun comes off: {_b[5:46]!r}",
              S.infer_removals(_b, _two) == _want)


def test_the_age_on_the_sheet_reaches_the_body():
    """REPORTED: "Breast development should also be correct, given the age of a person."

    The age was inert. It went to the model inside the author's own words and nothing
    here read it, so every clause this file writes about a body said "a woman's body" --
    true of a woman of 22 and a woman of 62, which settles nothing between them. An
    attribute a prompt does not state is left to the PRIOR, and the prior is a woman in
    her twenties whatever the sheet says."""
    print("\n=== the age on the sheet reaches the body ===")
    for _line, _want in (
            ("Nora: <Picture 1>, she, 24, long dark hair.", 24),
            ("Maya: 27, silver hair, grey coat.", 27),
            ("Dan: he, aged 41, a grey coat.", 41),
            ("Sam: she, 38yo, red hair.", 38),
            ("Ana: she, 52 years old.", 52),
            ("Mara: she, a 29-year-old nurse.", 29),
            ("Eve: she, in her forties, dark hair.", 45),
            ("Liz: she, early thirties, freckles.", 32),
            ("Jo: she, late 20s, tattoos.", 28),
            ("Ivy: she, mid-50s, grey bob.", 55),
            ("Kate: she, 28.", 28),
            ("Jo: she, 33;", 33),
            ("Dan: he, 41", 41)):
        check(f"age read: {_line[:42]!r} -> {_want}", S.age_in(_line) == _want)
    for _line in ("Nora: <Picture 2>, she, long dark hair, size 10 boots.",
                  "Dan: he, 5'7\", a grey coat.", "Tess: she, tall, blue eyes.",
                  "Ann: she, a 9mm in her belt.", "Zoe: she, wears a number 7 shirt."):
        check(f"not an age: {_line[:44]!r}", S.age_in(_line) == 0)
    check("no age keeps the plain phrasing",
          S.body_of("she") == "a woman's body" and S.body_of("he") == "a man's body")
    check("an age is named", S.body_of("she", 45) == "the body of a woman of 45"
          and S.body_of("he", 41) == "the body of a man of 41")
    check("an undeclared pronoun still names nothing",
          S.body_of("they", 30) == "" and S.body_of("", 30) == "")
    _said = {a: S.figure_of("she", a) for a in (19, 24, 31, 42, 48, 63)}
    check(f"every adult decade says something", all(_said.values()))
    check("...and they differ from each other", len(set(_said.values())) >= 5)
    check("...each naming the age it was given",
          all(str(a) not in v for a, v in _said.items()))   # the body phrase carries it
    check("no age means no figure", S.figure_of("she") == "")
    check("a sheet that declares no 'she' gets none",
          S.figure_of("he", 41) == "" and S.figure_of("they", 30) == "")
    _f, _b = S.figure_of("she", 45), S.body_of("she", 45)
    check("the chest clause rides a chest sentence",
          "the breasts" in S.bare_hold(["torso"], {}, "", [], body=_b, figure=_f))
    check("...and not a legs-only one",
          "the breasts" not in S.bare_hold(["legs"], {}, "", [], body=_b, figure=_f))
    check("...nor one where the chest was capped out",
          "the breasts" not in S.bare_hold(["legs", "feet", "torso"], {}, "", [],
                                           body=_b, figure=_f))
    check("...and it comes after the body it is a fact about",
          S.bare_hold(["torso"], {}, "", [], body=_b, figure=_f).index(_b)
          < S.bare_hold(["torso"], {}, "", [], body=_b, figure=_f).index("the breasts"))
    for _a in (19, 24, 31, 42, 48, 63):
        check(f"the figure at {_a} does not say 'skin' twice",
              "skin" not in S.figure_of("she", _a))


def test_no_body_is_described_for_a_declared_minor():
    """The floor under the age reader, and it is not a softer description -- none.

    Reading an age to drive anatomy means the reader has to answer for the ages it was
    not meant for. A generator has no business composing a body for a child, so every
    clause built on body_of and figure_of goes silent below 18, and a film that declares
    a minor AND stages nudity or sex does not render at all."""
    print("\n=== no body is described for a declared minor ===")
    for _a in (0, 1, 9, 13, 16, 17):
        if _a:
            check(f"no body at {_a}", S.body_of("she", _a) == ""
                  and S.body_of("he", _a) == "")
            check(f"...and no figure at {_a}", S.figure_of("she", _a) == "")
    check(f"{S.ADULT_AGE} is where a body starts being named",
          S.body_of("she", S.ADULT_AGE) != ""
          and S.body_of("she", S.ADULT_AGE - 1) == "")
    check("the bare clause names no body for a minor",
          "body" not in S.bare_hold(["torso"], {}, "", [],
                                    body=S.body_of("she", 15),
                                    figure=S.figure_of("she", 15)))
    _adult = "Kate: she, 28, long hair."
    _child = "Kate: she, 28.\nSam: she, 15."
    _sexual = "A bedroom.\n\nKate undresses and lies down.\n\nShe moans."
    _plain = "A kitchen.\n\nSam eats breakfast.\n\nKate reads the paper."
    check("an adult film with sexual staging is not refused",
          S.minor_with_sexual_staging(_adult, _sexual) == "")
    check("a declared minor in an ordinary scene is not refused",
          S.minor_with_sexual_staging(_child, _plain) == "")
    _msg = S.minor_with_sexual_staging(_child, _sexual)
    check("a declared minor with sexual staging IS refused", _msg != "")
    check("...naming who was declared under age", "Sam" in _msg)
    check("...and saying nothing rendered", "nothing was rendered" in _msg)
    check("...and what to do if the age is a typo", "typo" in _msg)
    check("...and does not claim to know who it was about", "whichever" in _msg)
    for _word in ("naked", "nude", "sex", "fucking", "orgasm", "moans", "topless",
                  "undresses", "masturbating", "aroused", "nipples"):
        check(f"staging recognised: {_word!r}",
              S.minor_with_sexual_staging(_child, f"A room.\n\nShe is {_word}.") != "")
    # An age that is not declared cannot trip it -- there is nothing to read.
    check("no age declared anywhere is not refused",
          S.minor_with_sexual_staging("Kate: she, long hair.", _sexual) == "")


def test_a_written_sound_is_recognised():
    print("\n=== a sound you wrote, in the words people write it in ===")
    for _t in ("her boots loud on the concrete", "a low hum off the strip light",
               "her boots scuff the floor", "gravel crunching under the tyres",
               "a knock at the door", "the engine roars",
               "rain drumming on the roof", "the fan whirring overhead",
               "the chain drags and rattles"):
        check(f"heard: {_t[:34]!r}", S.sound_described(_t))
    for _t in ("Nora walks to the window.", "Nora looks at the toolbox.",
               "Nora sits down on the bench.", "Nora picks up the spanner.",
               "Dan hands her the cable."):
        check(f"not a written sound: {_t[:32]!r}", not S.sound_described(_t))
    for _t in ("The workshop is quiet, the roller door shut.", "She is quiet.",
               "A quiet street at night.", "She gives him a quiet look.",
               "The light is faint.", "A faint smile.",
               "He ticks a box on the form."):
        check(f"still silent: {_t[:36]!r}", not S.sound_described(_t))
    for _t in ("She quietly closes the door.", "The clock is ticking."):
        check(f"...but heard: {_t[:32]!r}", S.sound_described(_t))


def test_the_upscale_path_does_not_hold_the_chain_twice():
    print("\n=== the upscale path streams instead of concatenating ===")
    _put, _done = S._stream_chunks(0)
    check("nothing put -> nothing returned", _done() is None)
    for _total, _step in ((10, 3), (10, 10), (1, 4), (9, 4), (7, 1)):
        _pieces = [torch.rand(min(_step, _total - i), 4, 6, 3)
                   for i in range(0, _total, _step)]
        _put, _done = S._stream_chunks(_total)
        for _pc in _pieces:
            _put(_pc)
        check(f"streamed {_total} in steps of {_step} == torch.cat",
              torch.equal(_done(), torch.cat(_pieces, dim=0)))
    _put, _done = S._stream_chunks(4)
    _put(torch.rand(2, 40, 60, 3)); _put(torch.rand(2, 40, 60, 3))
    check("destination takes the chunk's own H/W", tuple(_done().shape) == (4, 40, 60, 3))

    _seen = []
    _real = S.comfy.utils.common_upscale if hasattr(S.comfy.utils, "common_upscale") else None
    def _fake(sm, wdt, hgt, method, crop):
        _seen.append(int(sm.shape[0]))
        return torch.nn.functional.interpolate(sm.float(), size=(hgt, wdt),
                                               mode="nearest").to(sm.dtype)
    S.comfy.utils.common_upscale = _fake
    try:
        _f = torch.rand(70, 64, 96, 3)
        _got = S._resize_short_edge(_f, 96)
        check("the resize is chunked, not one call", len(_seen) > 1)
        check(f"...no chunk exceeds RESIZE_CHUNK ({S.RESIZE_CHUNK})",
              max(_seen) <= S.RESIZE_CHUNK)
        check("...every frame is accounted for", sum(_seen) == 70)
        check("...and the result is the whole batch", tuple(_got.shape) == (70, 96, 128, 3))
        _seen.clear()
        _same = S._resize_short_edge(_f, 96)
        check("...chunking is deterministic", torch.equal(_got, _same))
        _seen.clear()
        _nop = torch.rand(4, 64, 64, 3)
        check("an already-correct size is a no-op",
              S._resize_short_edge(_nop, 64) is _nop and not _seen)
    finally:
        if _real is not None:
            S.comfy.utils.common_upscale = _real
        else:
            del S.comfy.utils.common_upscale


def test_a_vae_that_tiles_itself_is_not_asked_to():
    print("\n=== the tiled detour costs 3x on a VAE that owns its tiling ===")
    class _FSM:
        comfy_has_chunked_io = True

    class _Owns:
        handles_tiling = True
        first_stage_model = _FSM()
        def __init__(self): self.calls = []
        def decode(self, z):
            self.calls.append("decode"); return torch.rand(1, 3, 8, 16, 16)
        def decode_tiled(self, z, **kw):
            self.calls.append("decode_tiled"); return torch.rand(1, 3, 8, 16, 16)

    class _Plain(_Owns):
        handles_tiling = False
        first_stage_model = None

    _lat = {"samples": torch.zeros(1, 24, 4, 4, 4)}
    _v = _Owns(); S._decode_video(_v, _lat, True)
    check("owns its tiling: tiled=True still goes to decode()", _v.calls == ["decode"])
    _v = _Owns(); S._decode_video(_v, _lat, False)
    check("...and tiled=False is unchanged", _v.calls == ["decode"])
    _p = _Plain(); S._decode_video(_p, _lat, True)
    check("a plain VAE still gets decode_tiled", _p.calls == ["decode_tiled"])
    _p = _Plain(); S._decode_video(_p, _lat, False)
    check("...and its untiled path is unchanged", _p.calls == ["decode"])
    class _NoBuf(_Owns):
        class _F: comfy_has_chunked_io = False
        first_stage_model = _F()
    _n = _NoBuf(); S._decode_video(_n, _lat, True)
    check("tiling owned but no chunked IO -> old path kept", _n.calls == ["decode_tiled"])


def test_the_overlay_does_not_copy_a_chain_it_will_not_draw_on():
    print("\n=== H3 Overlay does not clone what it will not touch ===")
    import importlib.util as _u, os as _os
    _sp = _u.spec_from_file_location("h3_overlay_t",
                                     _os.path.join(_HERE, "overlay.py"))
    _ov = _u.module_from_spec(_sp); _sp.loader.exec_module(_ov)
    _N = _ov.NODE_CLASS_MAPPINGS["H3Overlay"]()
    _src = torch.rand(8, 32, 48, 3)
    _mark = _src.clone()

    _out, _note = _N.run(images=_src, fps=24)
    check("nothing to draw -> no copy is made", _out.data_ptr() == _src.data_ptr())
    check("...and it still says so", "frames unchanged" in _note)
    check("...and the upstream tensor is untouched", torch.equal(_src, _mark))
    check("...shape and dtype pass through",
          _out.shape == _src.shape and _out.dtype == _src.dtype)
    # Whitespace is not text. " " must take the same path as "".
    _ws, _ = _N.run(images=_src, fps=24, watermark_text="   ", intro_text="\n")
    check("whitespace is not something to draw", _ws.data_ptr() == _src.data_ptr())

    _out2, _note2 = _N.run(images=_src, fps=24, watermark_text="HELLO")
    check("something to draw -> a copy IS made", _out2.data_ptr() != _src.data_ptr())
    check("...the upstream tensor is STILL untouched", torch.equal(_src, _mark))
    check("...and the frames really were drawn on", not torch.equal(_out2, _src))
    check("...and it says what it did", "watermark" in _note2 or "overlays applied" in _note2)


def test_behind_the_back_is_read_however_it_is_written():
    print("\n=== the wrists are behind the back however that is typed ===")
    for _t in ("Dan handcuffs her wrists behind her back.",
               "Her hands are cuffed behind her.",
               "Her wrists are bound together behind her.",
               "Dan cuffs her hands together at the small of her back.",
               "Her hands are cuffed at the small of her back.",
               "She is cuffed, hands behind back.",
               "Her arms are pinned behind her.",
               "Her hands are bound behind her back with cuffs."):
        check(f"reads as behind the back: {_t[:44]!r}",
              S.limb_anchor(_t) == "behind the back")
    # ...and the pose clause then has something to say, which is the point.
    check("...and that gives the shot a pose to state",
          "wrists together" in S.pose_clause(S.limb_anchor("Her hands are cuffed behind her.")))
    for _t in ("Dan stands behind her.", "Dan steps in behind her and looks down.",
               "The van is parked behind her.", "He walks behind her to the barn.",
               "Dan closes the door behind her.", "She hears him behind her."):
        check(f"a person behind her is not her wrists: {_t[:38]!r}",
              not S.limb_anchor(_t))
    # The other positions are untouched.
    for _t, _want in (("Her wrists are cuffed above her head.", "above the head"),
                      ("Her hands are cuffed at her waist.", "at the waist"),
                      ("Her arms are out to the sides.", "out to the sides")):
        check(f"unchanged: {_want}", S.limb_anchor(_t) == _want)


def test_a_bound_body_lying_down_has_something_under_it():
    print("\n=== a bound body lying down rests on itself ===")
    _up = S.pose_clause("behind the back")
    _down = S.pose_clause("behind the back", lying=True)
    check("on her feet, the pose clause is unchanged", "take the weight" not in _up)
    # WORDING CHANGED, GUARANTEE DID NOT. "The shoulder and the hip take the
    # weight" is a body on its SIDE, and it was said about every lying body. What
    # must hold is that the region under one is never left unsaid.
    check("lying down, the weight is named", "the weight of the body" in _down)
    check("...without claiming a side nobody wrote", "shoulder and the hip" not in _down)
    _prone = S.pose_clause("behind the back", lying=True, facing="face down")
    check("a facing the author wrote is used", "face down" in _prone
          and "the weight of the body" in _prone)
    _side = S.pose_clause("behind the back", lying=True, facing="on the side")
    check("...and the side wording belongs to the side", "shoulder and the hip" in _side)
    check("...and the arms are still said to be behind", "behind the body" in _down)
    for _neg in (" no ", " not ", "nothing", "never", "without"):
        check(f"no negation in the clause: {_neg.strip()!r}", _neg not in _down.lower())
    for _pos in ("above the head", "in front of the body", "out to the sides",
                 "at the waist"):
        check(f"untouched when lying: {_pos}",
              S.pose_clause(_pos) == S.pose_clause(_pos, lying=True))
    check("an unknown position still says nothing",
          S.pose_clause("sideways", lying=True) == "")
    # ...and the posture that feeds it has to be readable off how people write it.
    for _b in ("McKenna rolls onto her side.", "She rolled onto her back.",
               "He rolls over onto his stomach.", "She lays down on her side."):
        check(f"reads as lying: {_b[:34]!r}", S.engine.posture_in(_b) == "lying down")
    for _b in ("The barrel rolls onto its side.", "The van rolls to a stop."):
        check(f"...and an object is not a body: {_b[:32]!r}", not S.engine.posture_in(_b))


def test_a_body_under_effort_has_a_voice():
    print("\n=== effort makes a sound, and it is a voice ===")
    for _b in ("McKenna thrashes on the bed.", "She writhes and arches under him.",
               "He shudders and grips the sheet.", "She strains against him."):
        check(f"voiced: {_b[:34]!r}",
              any("moans of effort" in s for s in S.sounds_for(_b)))
    check("a beat naming the sound is left alone", S.sounds_for("She moans.") == [])
    check("...but it does count as asking for audio", S.sound_described("She moans."))
    check("two named vocals, still nothing added", S.sounds_for("She moans and sobs.") == [])
    for _b, _want in (("She whimpers and thrashes in her restraints.", "whimpering"),
                      ("She sobs and pulls against the cuffs.", "sobbing"),
                      ("She screams and thrashes in her restraints.", "screaming"),
                      ("She moans and thrashes in her restraints.", "moaning")):
        _got = S.sounds_for(_b)
        check(f"named vocal survives a mixed list: {_want}", _want in _got)
        check(f"...and retires the inferred effort phrase: {_want}",
              not any("moans of effort" in _s for _s in _got))
    check("no vocal named, effort phrase still there",
          any("moans of effort" in _s for _s in S.sounds_for("She thrashes in her restraints.")))
    _dup = S.sounds_for("She wakes up and thrashes in her restraints.")
    check("no duplicate breathing in one clause",
          not ("breathing" in _dup and any("moans of effort" in _s for _s in _dup)))
    for _w in ("gasps", "whimpers", "groans", "pants", "sobs"):
        check(f"{_w} is heard as a sound the author wrote", S.sound_described(f"She {_w}."))
    check("effort opens the branch", S.exertion_in("She writhes on the bed."))
    check("...and ordinary movement does not", not S.exertion_in("Maya walks to the window."))
    check("thrashing loose is not restraints",
          "restraints pulling taut" not in S.sounds_for("McKenna thrashes on the bed."))
    check("...and thrashing in cuffs is",
          "restraints pulling taut" in S.sounds_for("Kate thrashes against the handcuffs."))


def test_widget_values_are_usable():
    print("\n=== a widget value that is not a number ===")
    for _bad in (float("nan"), float("inf"), True, False, None, "", "abc"):
        out, notes = S.sane_widgets({"pace": _bad})
        check(f"unusable value repaired: {_bad!r}", out["pace"] == 1.0 and bool(notes))
    check("...and the cause is named",
          "BY POSITION" in S.sane_widgets({"pace": float("nan")})[1][0])
    _many = S.sane_widgets({"pace": float("nan"), "steps": float("nan"),
                            "megapixels": float("nan")})[1]
    check("all of them in a single note", len([n for n in _many if "not usable" in n]) == 1)
    check("...naming each widget", all(w in _many[0] for w in ("pace", "steps", "megapixels")))
    check("...and that it returns until the node is recreated",
          "every restart" in _many[0] and "Fix node (recreate)" in _many[0])
    # Out of range is a value the user chose, so it is clamped rather than discarded.
    check("below the minimum is clamped", S.sane_widgets({"pace": 0.01})[0]["pace"] == 0.25)
    check("above the maximum is clamped", S.sane_widgets({"pace": 9.0})[0]["pace"] == 2.0)
    check("...and reported", "clamped" in S.sane_widgets({"pace": 9.0})[1][0])
    check("a good value is untouched and silent",
          S.sane_widgets({"pace": 1.0}) == ({"pace": 1.0}, []))
    # Ints stay ints: a float step count would index a sigma schedule wrongly.
    got = S.sane_widgets({"steps": 6.7})[0]["steps"]
    check("an int widget stays an int", isinstance(got, int) and got == 6)
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
    schema = S.H3LongVideos.INPUT_TYPES()
    req, opt = schema["required"], schema["optional"]
    for name in ("model", "clip", "vae", "audio_vae", "prompt"):
        check(f"{name} is required", name in req)
    check("the prompt is a socket, not a box", req["prompt"][1].get("forceInput") is True)
    check("shot_seconds defaults to 10", req["shot_seconds"][1]["default"] == 10.0)
    check("the shifts default to 12/3",
          opt["shift_video"][1]["default"] == 12.0 and opt["shift_audio"][1]["default"] == 3.0)
    check("restraints are held by default", opt["hold_restraints"][1]["default"] is True)
    check("removals are read from the beat by default",
          opt["auto_remove"][1]["default"] is True)
    check("shot length is read from the beat by default",
          opt["shot_length"][1]["default"] == "from the beat")
    check("first_frame is offered", "first_frame" in opt)
    n_widgets = sum(1 for d in (req, opt) for k, v in d.items()
                    if not (len(v) > 1 and isinstance(v[1], dict) and v[1].get("forceInput"))
                    and (isinstance(v[0], list) or v[0] in ("INT", "FLOAT", "STRING", "BOOLEAN")))
    check(f"the node stays small: {n_widgets} widgets", n_widgets <= 28)
    for _w in ("anchor", "character_memory"):
        check(f"{_w} is offered", _w in opt)
    check("...and they sit at the end, in the order they were added",
          list(opt)[-9:] == ["anchor", "character_memory", "pace",
                             "ambient_audio", "ambient_level", "foley_level",
                             "speech_lead_seconds", "speech_tail_seconds",
                             "hold_levels"])
    # SEVEN WIDGETS THAT WERE NOT CHOICES. Each had one right answer the node
    # could reach and the reader could not; each is now measured or pinned. They are
    # asserted GONE, the way reference_mode and save_defaults are, because a widget
    # that comes back is a question being asked again.
    for _gone, _why in (("cfg", "H3 is CFG-free; pinned to 1.0"),
                        ("apply_model_sampling", "read from the model's own stamp"),
                        ("silence_nonspeech", "already decided per shot"),
                        ("trim_seam", "the seam frame is a duplicate either way"),
                        ("tiled_decode", "measured against free VRAM"),
                        ("cleanup_between_shots", "pinned on"),
                        ("upscale_batch", "measured against free VRAM"),
                        # The continuity guards. Each answered a reported failure and
                        # turning one off returned it rather than trading it; verbatim
                        # turned off all seven at once, to tell the node's doing from
                        # the model's, and went at the reader's word.
                        ("character_guard", "the cast scoping always runs"),
                        ("hold_gaze", "pinned on"),
                        ("hold_scene_state", "pinned on"),
                        ("mouths_shut_when_no_line", "pinned on"),
                        ("hold_camera", "pinned on"),
                        ("auto_sound", "pinned on"),
                        ("beat_leads", "pinned on: the beat leads, measured better"),
                        ("verbatim", "removed; info still reports what each clause says")):
        check(f"{_gone} is gone -- {_why}", _gone not in opt and _gone not in req)
    check("reference_mode is gone", "reference_mode" not in opt)
    check("save_defaults is gone", "save_defaults" not in opt and "save_defaults" not in req)
    for _u in ("upscale", "upscale_model", "upscale_target_short_edge",
               "latent_upscale", "latent_upscale_scale"):
        check(f"{_u} is on the node", _u in opt)
    check("both upscalers default to off",
          opt["upscale"][1]["default"] == "off" and opt["latent_upscale"][1]["default"] == "off")
    check("outputs include info and script",
          "info" in S.H3LongVideos.RETURN_NAMES and "script" in S.H3LongVideos.RETURN_NAMES)
    aliases = {"H3LongVideos", "H3LongVideosFL2VA", "H3LongVideosV1",
               "H3LongVideosREF2VA"}
    check("all saved-workflow ids resolve to the sampler",
          set(S.NODE_CLASS_MAPPINGS) == aliases
          and all(v is S.H3LongVideos for v in S.NODE_CLASS_MAPPINGS.values()))


_SPARSE_ON = {"transformer_options": {"patches_replace": {"dit": {("double_block", 0): None}}}}
_SPARSE_OFF = {"transformer_options": {}}


def _alloc_model(model_options=None):
    m = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=None))
    if model_options is not None:
        m.model_options = model_options
    return m


def test_the_allocator_that_aborts_is_refused_before_sampling():
    print("\n=== the allocator that aborts instead of raising ===")
    saved = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
        why = S.sparse_attention_allocator_abort(_alloc_model(_SPARSE_ON))
        check("cudaMallocAsync WITH a sparse DiT patch is refused", "ABORT" in why)
        check("...naming the flag that fixes it", "--disable-cuda-malloc" in why)
        check("...and saying where the allocator came from, since nobody picked it",
              "CUDA 13" in why)
        check("the allocator alone is not refused -- it is only lethal with the patch",
              S.sparse_attention_allocator_abort(_alloc_model(_SPARSE_OFF)) == "")
        check("a model that cannot say is never refused on a guess",
              S.sparse_attention_allocator_abort(_alloc_model()) == "")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        check("another allocator config with the same patch is left alone",
              S.sparse_attention_allocator_abort(_alloc_model(_SPARSE_ON)) == "")
        del os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        check("no allocator config at all is left alone",
              S.sparse_attention_allocator_abort(_alloc_model(_SPARSE_ON)) == "")
    finally:
        if saved is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = saved
    check("the patch probe tells 'absent' apart from 'cannot tell'",
          S.sparse_dit_patched(_alloc_model(_SPARSE_OFF)) is False
          and S.sparse_dit_patched(_alloc_model()) is None
          and S.sparse_dit_patched(_alloc_model(_SPARSE_ON)) is True)


def main():
    test_beats()
    test_verbatim()
    test_sizing()
    test_speech_and_refs()
    test_removals()
    test_inferred_removals()
    test_a_remove_line_is_read_as_the_garment()
    test_every_way_a_garment_comes_off_is_read()
    test_a_word_does_not_move_the_room()
    test_character_sheet()
    test_no_one_is_described_twice()
    test_sheet_lines_are_terminated()
    test_character_guard()
    test_two_person_cast_has_one_body_each()
    test_extracted_planning_policies()
    test_layers_from_prose()
    test_opening_pose()
    test_removal_needs_a_particle()
    test_how_clothes_actually_come_off()
    test_undressing_completely()
    test_a_name_with_no_entry()
    test_layers()
    test_bare_region()
    test_the_sheet_names_the_garment()
    test_a_removal_names_it_the_way_the_sheet_does()
    test_a_removal_stays_on_one_person()
    test_the_shot_says_each_thing_once()
    test_generic_clothes_come_off_too()
    test_a_group_beat_keeps_the_group()
    test_the_wearer_is_in_the_shot()
    test_a_posture_carries_to_the_next_shot()
    test_a_journey_has_two_ends()
    test_the_room_follows_the_characters()
    test_effort_verbs_open_the_branch_they_are_given_sound_for()
    test_furniture_under_movement_is_built()
    test_a_chain_can_be_fastened_to_a_wall()
    test_the_anchor_holds_the_part_the_hardware_is_on()
    test_one_beat_can_put_on_two_things()
    test_a_neck_is_not_behind_a_back()
    test_a_door_is_not_a_room()
    test_holding_an_item_back_never_costs_the_person()
    test_a_chastity_belt_is_underwear()
    test_a_beat_names_the_sound_its_props_make()
    test_an_ambient_bed_is_mixed_not_conditioned()
    test_the_loop_join_does_not_click()
    test_a_garment_going_on_has_both_ends()
    test_a_lens_setting_is_not_a_room()
    test_the_sound_clause_spends_from_the_budget()
    test_a_described_room_is_still_a_room()
    test_the_sound_follows_the_room()
    test_a_transitive_posture_puts_the_object_down()
    test_an_action_lets_go_of_a_posture()
    test_a_posture_told_is_not_a_posture_taken()
    test_speech_is_marked_as_speech()
    test_one_person_undressing_is_one_person()
    test_a_comma_separated_list_is_a_list_of_actions()
    test_a_short_action_gets_the_whole_shot()
    test_a_dropped_garment_is_not_a_fall()
    test_the_babble_advice_points_the_right_way()
    test_silence_reports_what_happened()
    test_the_audio_branch_has_its_own_last_step()
    test_the_addressee_is_not_the_speaker()
    test_removal_completes()
    test_restraints_hold()
    test_hardware_has_somewhere_to_go()
    test_a_tape_gag_stays_tape()
    test_a_stated_state_is_not_an_event()
    test_the_hardware_keeps_being_named()
    test_memory_is_asked_for_honestly()
    test_the_hold_needs_its_wearer_on_screen()
    test_the_hold_names_its_wearer_once()
    test_the_shot_that_puts_hardware_on()
    test_a_machines_line_is_not_the_actors_line()
    test_a_shifted_workflow_is_named_not_rendered()
    test_what_is_exposed_is_not_also_removed()
    test_underwear_goes_under()
    test_pulling_something_down_is_not_falling()
    test_a_fall_says_what_takes_the_landing()
    test_the_look_goes_where_the_beat_says()
    test_an_object_tag_leaves_with_its_object()
    test_fastened_limbs_keep_their_anchor()
    test_scenery_does_not_move_the_wrists()
    test_only_a_beat_with_a_person_gets_a_mouth()
    test_a_tag_names_the_socket_it_is_wired_to()
    test_an_exit_and_a_shut_door_disagree()
    test_a_held_thing_is_not_also_heard_moving()
    test_a_staged_change_names_both_ends()
    test_sound_described()
    test_sound_is_derived_from_the_action()
    test_pace()
    test_av_grid_alignment()
    test_chain_is_rigid()
    test_falling_bound()
    test_turning_around()
    test_thin_beats()
    test_a_function_word_is_never_a_character()
    test_auto_length()
    test_text_in_frame()
    test_reference_tags()
    test_sound_clause_closes_the_list()
    test_hardware_belongs_to_somebody()
    test_one_pronoun_is_one_person()
    test_a_tagged_object_can_be_taken_off()
    test_the_age_on_the_sheet_reaches_the_body()
    test_no_body_is_described_for_a_declared_minor()
    test_underwear_is_a_garment_with_a_place_on_the_body()
    test_how_underwear_actually_comes_off()
    test_a_written_sound_is_recognised()
    test_the_upscale_path_does_not_hold_the_chain_twice()
    test_a_vae_that_tiles_itself_is_not_asked_to()
    test_the_overlay_does_not_copy_a_chain_it_will_not_draw_on()
    test_behind_the_back_is_read_however_it_is_written()
    test_a_bound_body_lying_down_has_something_under_it()
    test_a_body_under_effort_has_a_voice()
    test_a_lora_states_its_step_count_in_its_name()
    test_the_graph_says_whether_the_schedule_is_already_set()
    test_a_repeated_naming_is_spent_as_a_pronoun()
    test_the_dit_gets_the_pinned_pool_to_itself()
    test_no_report_touches_weight_data()
    test_a_lora_that_does_not_fit_is_reported_not_silent()
    test_widget_values_are_usable()
    test_the_allocator_that_aborts_is_refused_before_sampling()
    test_schema()
    print()
    if _fails:
        print(f"RESULT: {len(_fails)} FAILURE(S): " + "; ".join(_fails))
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
