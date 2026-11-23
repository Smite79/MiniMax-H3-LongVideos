import io
import json
import os
import re
import sys
import tempfile
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", ".."))]
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch
import comfy.nested_tensor
import comfy.patcher_extension as pe
import chain
import faces
import modelprep
import node
import overlay
import planner as P

MIXED = lambda text, *things: [t for t in re.split(r"(?<=[.!?])\s+", text) if re.search(r"\b(?:hands?|wrists?|cuffs?|handcuffs)\b", t)
                               and any(x in t for x in things)]

_fails = []
MEM = ("Sarah <Picture 1>: a young woman with long blonde hair, in a white blouse and grey leggings.\n"
       "Dana: a woman with short black hair, in a black suit.")
IMG = torch.rand(1, 64, 48, 3)


def check(label, ok, extra=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else f"   {extra}"))
    if not ok:
        _fails.append(label)


def notes(shot):
    return shot["prompt"].split("\n\n")[-1]


def plan(*beats, m=MEM, a="", scene="A bedroom with a bed.", **kw):
    return P.plan(scene + "\n\n" + "\n\n".join(beats), m, a, pictures=kw.pop("pictures", [IMG, IMG, IMG]), **kw)


def beat(shot):
    return shot["prompt"].split("\n\n")[-2]


def sheet(shot, name="Sarah"):
    return next(line for line in shot["prompt"].split("\n") if line.startswith(name))


def said(text, shots):
    return [s for s in shots if text in s["line"]]


def test_fetch():
    print("\n=== taking gear out of a bag puts nothing on or off, 'them' and 'it' find their owner, a chain locked to her cuffs is "
          "one crotch chain linked to them, and putting gear on takes its real time, split over shots when one is too short ===")
    mem = MEM.replace("Dana: a woman with short black hair, in a black suit.", "Paul: a man with short dark hair, in a black suit.")
    shots, warn = plan("Paul pulls Sarah's arms behind her back. He takes out a pair of handcuffs.",
                       "He secures them together using the handcuffs.",
                       "Paul takes out a crotch chain from the duffle bag. He tightly wraps it around Sarah's hips, then pulls the "
                       "remaining chain between her legs and locks it to the handcuffs.",
                       "Paul takes out a tight mirrored steel collar with ring out of the duffle bag and locks it around Sarah's neck. He "
                       "then takes a chain leash out of the duffle bag and attaches it to the ring on her collar.",
                       "Paul unlocks the crotch chain and takes it off Sarah.", m=mem, shot_seconds=15.0)
    alone = plan("Sarah waits. Paul walks in.", "He secures them together using the handcuffs.", m=mem)[0][1]
    find = lambda text: next((s for s in shots if text in s["prompt"]), {"prompt": "", "line": "", "frames": 0})
    cuffed, chained = find("closes it around her left wrist"), find("takes a separate crotch chain")
    collared, leashed, freed = find("opens the steel collar"), find("clips the chain to the ring"), find("unlocks the end of the center")
    check("nothing is read from gear taken out of a bag; 'secures them' cuffs the one other person's wrists behind her; the chain "
          "locked to the cuffs is one crotch chain whose center chain ends in a padlock through the cuffs' own links, never cuffs at "
          "her crotch and never a second chain from the cuffs; the leash clips to the collar's own ring and stays in his hand; a steel "
          "collar locks; that padlock comes off first",
          not warn and "read from the beat" not in shots[0]["line"]
          and "hold: Sarah, handcuffs locking her wrists behind her back" in cuffed["line"]
          and "hold: Sarah, handcuffs locking her wrists behind her back" in alone["line"]
          and "between her legs" not in chained["line"]
          and "stands in front of Sarah and takes a separate crotch chain, wraps its waist chain tight around her waist and locks it at "
              "the front, below her navel, with a padlock, runs its "
              "center chain down from the front of the waist chain under her crotch, centers it tight over the outside of her grey "
              "leggings, then steps behind her, draws the center chain up between her buttocks to the small of her back and padlocks "
              "its end through the short links that join her two cuffs." in notes(chained)
          and not any("separate short chain" in s["prompt"] or "chain between her handcuffs" in s["prompt"] for s in shots)
          and not [x for x in re.split(r"(?<=[.!?])\s+", collared["prompt"]) if "crotch chain" in x and "front" in x and "cuffs" in x]
          and "She has handcuffs locking her wrists behind her back, and the end of the center chain of her crotch chain is padlocked "
              "through the short links between them, at the small of her back." in sheet(collared)
          and "and her hands rest together at the small of her back, held there by the end of the center chain of her crotch chain, "
              "padlocked through the short links between her two cuffs. From the front, both arms angle back from her shoulders and "
              "her body hides her hands and that padlock;" in notes(collared)
          and "Sarah's handcuffs and her crotch chain are separate restraints, and the only link between them is the padlock at the end "
              "of the center chain, closed through the short links between her two cuffs, behind her at the small of her back."
              in notes(collared)
          and "Nothing else is clipped" not in notes(collared)
          and "stands in front of Sarah and opens the steel collar, fits it around her neck, closes it and locks it shut with a click."
              in notes(collared)
          and "a chain clipped to the ring at the front of her collar, its other end in dark-haired Paul's hand" in notes(leashed)
          and "padlocked" not in leashed["line"]
          and "steps behind Sarah and unlocks the end of the center chain from the short links between her two cuffs, draws the center "
              "chain out from between her legs, then unlocks the padlock at the front of her waist and takes the crotch chain off her "
              "waist" in notes(freed),
          (warn, [s["line"] for s in shots], alone["line"], [notes(s) for s in shots[2:]]))
    steel = plan("Paul takes out a tight mirrored steel collar with ring out of the duffle bag. He tightly locks it on Sarah's neck.",
                 m=mem)[0][-1]
    check("'it' after a collar taken out of the bag keeps what the collar is made of",
          "hold: Sarah, a steel collar locked around her neck" in steel["line"]
          and "opens the steel collar, fits it around Sarah's neck, closes it and locks it shut with a click" in notes(steel),
          steel["line"])
    check("each piece takes its real time, with fetching it from the bag counted, every part handled and checked before the next; a "
          "beat too long for one shot is split at a sentence into shots that each fit, and the second still knows what 'it' is",
          len(shots) == 7 and "over about 8 seconds: every part is handled by hand and checked before the next one starts" in notes(cuffed)
          and "part of beat 3, split to fit" in chained["line"] and "over about 14 seconds" in notes(chained)
          and chained["frames"] == 362 and "over about 9 seconds" in notes(collared) and "over about 6 seconds" in notes(leashed)
          and collared is not leashed and "part of beat 4, split to fit" in leashed["line"] and "over about 10 seconds" in notes(freed)
          and [s["beat"] for s in shots] == [1, 2, 3, 3, 4, 4, 5],
          ([s["line"][:120] for s in shots], [re.findall(r"over about \d+ seconds", s["prompt"]) for s in shots]))


def test_bonds():
    print("\n=== bound hands are placed for every pose, every body close to her and every step, and no sentence names two restraints ===")
    mem = MEM.replace("Dana: a woman with short black hair, in a black suit.", "Paul: a man with short dark hair, in a black suit.")
    cuffed = mem + "\nhold: Sarah, handcuffs behind her back"
    one = lambda *b: plan(*b, m=cuffed)[0]
    behind, push, hug, wall = (notes(one(b)[0]) for b in ("Paul stands behind Sarah and holds her shoulders.",
                                                           "Paul pushes Sarah forward toward the door.",
                                                           "Paul pulls Sarah close and kisses her.", "Paul pins Sarah against the wall."))
    cuffing = notes(plan("Paul cuffs Sarah's wrists behind her back.", m=mem)[0][0])
    chair = notes(one("Paul rolls the office chair up behind Sarah.")[0])
    check("cuffing her behind her back keeps her hands at the small of her back, hidden from the front, from the moment her arms are "
          "drawn back and all through the cuffs going on, so the cuffing never moves to her front",
          "there at the small of her back, keeping her hands there the whole time, hidden from the front by her body." in cuffing
          and "Sarah's arms stay pulled behind her back from the moment they are drawn back there, all through the handcuffs going on, "
              "with her wrists held together and then locked together behind her" in cuffing, cuffing)
    held = [notes(s) for s in one("Paul wraps his arms around Sarah.", "Paul embraces her from behind.", "Paul holds Sarah tight.")]
    check("someone holding her with no side named is close against her with her locked hands behind her, and 'from behind' is behind "
          "her, never in front",
          all("Dark-haired Paul is close against her, and her locked hands stay behind her at the small of her back." in held[i]
              for i in (0, 2)) and "Dark-haired Paul is close behind her" in held[1], held)
    check("someone close behind her, pushing her along, holding her close or pinning her to a wall gets a sentence saying where her "
          "locked hands are between the bodies, and hands 'behind her back' alone are not someone behind her",
          all("Dark-haired Paul is close behind her, and her locked hands stay behind her at the small of her back, between her back and "
              "dark-haired Paul's body." in x for x in (behind, push))
          and "Dark-haired Paul is close in front of her, and her locked hands stay behind her back, on the far side of her body from "
              "dark-haired Paul." in hug
          and "Her back is to the wall, and her locked hands stay behind her, between her back and the wall." in wall
          and "close behind" not in cuffing + chair, (behind, push, hug, wall, cuffing, chair))
    moves = one("Sarah sits down on the chair.", "Paul pushes Sarah down onto her knees.", "Sarah lies face down on the floor.",
                "Sarah gets up.")
    spin = ("Every turn Sarah makes in this shot is one slow, continuous turn on the spot in small steps, her whole body turning together "
            "as one piece, with her elbows bent and pointing back and her locked hands held up together in the middle of her back the "
            "whole way round: side-on, her bent arm and elbow show behind her;")
    check("sitting, kneeling, lying face down and getting up each say where her locked hands are, going down or rising keeps both "
          "arms behind her the whole way, and sitting down in a chair is done with her back to it after one slow whole-body turn with "
          "her elbows bent back",
          "Sarah ends the shot sitting in the chair: her arms stay behind her and her wrists stay locked together, her hands resting at "
          "the small of her back, between her back and the back of the chair." in notes(moves[0])
          and "Sarah goes down into the chair from standing close in front of it with her back to it, straight down onto the seat, both "
              "arms staying behind her the whole way." in notes(moves[0]) and spin in notes(moves[0])
          and spin not in notes(moves[1])
          and "Sarah ends the shot kneeling: her arms stay behind her" in notes(moves[1])
          and "Sarah ends the shot lying face down on the floor: her arms lie along her back and her wrists stay locked together, her "
              "hands resting on top of the small of her back" in notes(moves[2])
          and "Sarah ends the shot standing upright." in notes(moves[3])
          and "She rises with both arms staying behind her the whole way." in notes(moves[3]), [notes(s)[:500] for s in moves])
    turns = one("Sarah turns around.", "Paul turns Sarah around to face the wall.", "She slowly turns back toward him.",
                "Sarah turns her head toward the window.", "Paul turns around and looks at the door.", "Paul tells Sarah to sit down.")
    check("her turning around, being turned or turning back is one slow turn of her whole body with her elbows bent back and her locked "
          "hands held up in the middle of her back, never straight arms that read as arms at her sides; a head turn or someone else "
          "turning is not, and an order to sit is nobody's pose",
          all(spin in notes(s) and "Sarah's wrists stay locked together behind her the whole shot, with her elbows bent and pointing back"
              in notes(s) and "full length" not in notes(s) for s in turns[:3])
          and not any("Every turn" in notes(s) for s in turns[3:])
          and "pose:" not in turns[5]["line"], [(s["line"][:200], notes(s)[:300]) for s in turns])
    steps = one("Paul gags Sarah.", "Paul puts a collar on Sarah.", "Paul uncuffs Sarah.")
    check("with her hands cuffed behind her, a gag or collar goes on from in front of her, and uncuffing is done behind her with her "
          "hands kept behind her until they are free, the cuffs dropped to the floor",
          "Dark-haired Paul stands in front of Sarah and fits the gag into her mouth" in notes(steps[0])
          and "Dark-haired Paul stands in front of Sarah and wraps the collar around her neck" in notes(steps[1])
          and "Dark-haired Paul steps behind Sarah and, at the small of her back, unlocks the handcuffs with a small key and takes them "
              "off her wrists, while her hands stay behind her back until her wrists are free, and drops them to the floor."
              in notes(steps[2]) and "hands" not in notes(steps[2]).split("free, and drops")[1], [notes(s)[:400] for s in steps])
    gear = r"\b(?:handcuffs|cuffs|collar|leash|crotch chain|chastity belt|gag|blindfold|rope|shackles)\b"
    shots = one("Paul puts a collar and a chain leash on Sarah.", "Paul puts a crotch chain on Sarah.", "Paul gags Sarah.",
                "Paul cuffs Sarah's ankles.", "Paul leads Sarah to the door.")
    kinds = lambda x: {m.lower().replace("handcuffs", "cuffs") for m in re.findall(gear, x)}
    two = [x for s in shots for x in re.split(r"(?<=[.!?])\s+", sheet(s) + " " + notes(s)) if len(kinds(x)) > 1
           and "ring at the front of her collar" not in x]
    check("no sentence names two different restraints together", not two, two)


def test_scene():
    print("\n=== restraints stay in her description through a new scene, and a struggle is a real fight against them ===")
    mem = MEM + "\nhold: Sarah, a steel collar; handcuffs behind her back"
    moved, warn = plan("Sarah sits on the bed.", "cut\nA dark basement with a concrete floor.", "Dana pushes Sarah into the basement.",
                       m=mem)
    kept = ("Sarah <Picture 1>: a young woman with long blonde hair, in a white blouse and grey leggings. She has a steel collar "
            "locked around her neck. She has handcuffs locking her wrists behind her back. All of it stays on, exactly as described, "
            "for the whole shot.")
    check("a paragraph after a cut that names nobody is the new scene for the shots after it, never an empty shot; her restraints are "
          "in her description in every shot, her picture never overrides them, and her pose stays in the old place",
          len(moved) == 2 and moved[1]["prompt"].startswith("A dark basement with a concrete floor.\n")
          and "bedroom" not in moved[1]["prompt"]
          and moved[1]["cut"] and all(sheet(s) == kept for s in moved) and "Sarah: handcuffs" not in notes(moved[1])
          and "is for Sarah's face and hair only: her clothes and restraints are as this shot describes them." in notes(moved[1])
          and "sitting" not in notes(moved[1]) and not warn, ([sheet(s) for s in moved], [s["prompt"][:80] for s in moved], warn))
    fought = [plan("Dana watches Sarah.", b, m=mem)[0][1] for b in (
        "Sarah struggles against her restraints.", "Dana holds Sarah's arm while she tries to break free.",
        "Sarah sits in the chair and fights to get free.", "Dana struggles with the heavy box.")]
    check("a bound person struggling, named or as she, jerks hard against her cuffs in bursts with pauses, braced where she is, and "
          "the cuffs hold; someone free struggling with a box gets no such note",
          all("Sarah struggles for real, desperate to get free: in short, hard jerks with pauses between them, she twists her shoulders "
              "and wrenches against the handcuffs behind her back, " in notes(s) and "Between tries she stops for a moment, breathing "
              "hard, then tries again. The handcuffs hold her wrists locked together through every pull." in notes(s) for s in fought[:3])
          and "her feet braced where she stands, her face tight with strain." in notes(fought[0])
          and "staying in her seat, her face tight with strain." in notes(fought[2]) and "struggles for real" not in notes(fought[3]),
          [notes(s) for s in fought])


def test_tags():
    print("\n=== tag lines set restraints, clothes, poses and marks, and replace what the beat would be read as ===")
    shots, warn = plan("Dana walks in.\nhold: Sarah, red ball gag; handcuffs behind her back", "Sarah struggles.\nrelease: Sarah, gag",
                       "Dana leaves.\nexit: Dana", "Sarah waits.")
    check("hold: puts each item on the one it names, release: takes off only what it names, and exit: takes her out of the shots after",
          "goes on: Sarah: a red ball gag strapped in her mouth; handcuffs locking her wrists behind her back." in shots[0]["line"]
          and "comes off: Sarah: a red ball gag strapped in her mouth." in shots[1]["line"]
          and "handcuffs locking her wrists behind her back" in shots[2]["line"] and shots[3]["cast"] == ["Sarah"]
          and "Dana leaves during this shot." in notes(shots[2]), [s["line"] for s in shots])
    tagged = plan("Dana cuffs Sarah's ankles.\nhold: Sarah, a crotch chain")[0]
    check("a tag in a beat replaces what the beat would be read as, and a bare item goes where it belongs",
          "goes on: Sarah: a crotch chain" in tagged[0]["line"] and "ankles" not in tagged[0]["line"], tagged[0]["line"])
    start = plan("Sarah waits.", m=MEM + "\nhold: Sarah, a collar with a leash held by Dana\nmark: Sarah, a heart drawn on her cheek")[0]
    check("hold: and mark: lines in character_memory are on from the first shot",
          "on: Sarah: a collar locked around her neck; a leash clipped to the ring at the front of her collar" in start[0]["line"]
          and "a heart drawn on her cheek" in sheet(start[0]), (start[0]["line"], sheet(start[0])))
    posed = plan("Sarah waits.\npose: Sarah, kneeling on the floor", "Sarah looks up.")[0]
    check("pose: sets a pose that holds in the shots after it",
          "ends the shot kneeling on the floor" in notes(posed[0]) and "stays kneeling on the floor for the whole shot" in notes(posed[1]),
          [notes(s) for s in posed])
    _, warn = plan("Mia waits.\nhold: Mia, handcuffs")
    check("a tag naming someone without a character line is flagged", any("'Mia' has no line" in w for w in warn), warn)


def test_reading():
    print("\n=== the small reader: what it reads, and what it refuses to guess ===")
    cases = [("Dana cuffs Sarah's wrists behind her back.", "hold: Sarah, handcuffs locking her wrists behind her back"),
             ("Sarah sits down. Dana gags her with a red ball gag.", "hold: Sarah, a red ball gag strapped in her mouth"),
             ("Dana ties Sarah's ankles together with rope.", "hold: Sarah, rope tied around her ankles together"),
             ("Dana locks a chastity belt on Sarah.", "hold: Sarah, a chastity belt locked around her waist"),
             ("Dana hogties Sarah with rope on the floor.", "a separate short rope joining her wrists to her ankles behind her"),
             ("Sarah kneels on the bed, her wrists handcuffed behind her back.", "have: Sarah, handcuffs locking her wrists behind"),
             ("Dana takes off Sarah's blouse.", "remove: Sarah, blouse"),
             ("Dana pulls Sarah's leggings down.", "lift: Sarah, leggings down"),
             ("Dana strips Sarah naked.", "remove: Sarah, *"),
             ("Dana throws Sarah onto the bed.", "pose: Sarah, lying on the bed"),
             ("Sarah struggles. Dana pins her face down on the bed.", "pose: Sarah, lying face down on the bed"),
             ("Dana pushes Sarah into a chair.", "pose: Sarah, sitting on a chair"),
             ('Dana writes "MINE" across Sarah\'s stomach.', 'mark: Sarah, the word "MINE" written across her stomach')]
    got = [plan(t)[0][0]["line"] for t, _ in cases]
    check("each phrasing is read onto the right person, and info quotes what it read",
          all(want in line for (_, want), line in zip(cases, got)),
          [(t, line[:200]) for (t, want), line in zip(cases, got) if want not in line])
    quiet = ["Dana picks up the handcuffs.", "Sarah struggles against the bed.", "Dana smiles at Sarah.", "Sarah tugs at the sheets."]
    check("handling gear, struggling or touching changes nothing", all("read from the beat" not in plan(t)[0][0]["line"] for t in quiet),
          [plan(t)[0][0]["line"] for t in quiet])
    _, warn = plan("She takes off her jacket.", m=MEM.replace("Dana: a woman with short black hair, in a black suit.",
                                                              "Dana: a woman with short black hair, in a black jacket."))
    check("a she that could be either woman is not guessed: info says so and names the tag to use",
          any("could not tell whose clothes these are" in w and "remove: Sarah" in w for w in warn), warn)
    own = plan("Dana cuffs Sarah's wrists behind her back.", "Sarah puts her blouse back on.", "Dana unties Sarah.")[0]
    check("a plain 'her' before a garment is the actor's own, and 'unties Sarah' takes everything off her",
          "puts on: Sarah:" in own[1]["line"] or "clothes off" not in own[1]["line"] and "comes off: Sarah: handcuffs" in own[2]["line"],
          [s["line"] for s in own])


def test_crotch():
    print("\n=== a crotch chain is a waist chain and a center chain, put on, worn and taken off as one piece ===")
    tight = ("a crotch chain locked tight around her waist, its center chain pulled tight under her crotch and over her vagina from the "
             "front of her waist to the small of her back")
    cuff, bare = "Dana cuffs Sarah's wrists behind her back.", MEM.replace(" and grey leggings", "")
    chained = plan("Dana puts a crotch chain on Sarah.", "Dana unlocks the chain and takes it off Sarah.", m=bare)[0]
    cuffed = plan(cuff, "Dana puts a crotch chain on Sarah.", "Sarah walks to the door.", "Dana unlocks the chain and takes it off Sarah.",
                  m=bare)[0]
    same = [plan(b)[0][0]["line"] for b in ("Dana wraps a chain around Sarah's waist and between her legs and locks it with a padlock.",
                                            "Dana locks a chain around Sarah's waist and between her legs.")]
    roped = plan("Dana ties a rope around Sarah's waist and between her legs.")[0][0]
    wrists = plan("Dana chains Sarah's wrists behind her back.", "Dana puts a crotch chain on Sarah.",
                  "Dana unlocks the chain on Sarah's wrists.")[0][2]
    on = ("takes a separate crotch chain, wraps its waist chain tight around {x} waist and locks it at the front, below her navel, with "
          "a padlock, runs its center chain down from the front of the waist chain under her crotch, centers it tight over {c}, then "
          "steps behind her, draws the center chain up between her buttocks to the small of her back and locks it to the back of the "
          "waist chain with a padlock.")
    check("it goes on in one step: waist chain locked at the front, center chain under her crotch, then from behind her locked to the "
          "back; every shot says what its two parts are, 'the chain' takes it off and not her cuffs, and a chain around her waist and "
          "between her legs is one, never chains on her ankles",
          "Black-haired Dana " + on.format(x="Sarah's", c="her vagina") in notes(chained[0])
          and all(f"hold: Sarah, {tight} (" in x for x in [chained[0]["line"]] + same)
          and "Sarah's crotch chain has two parts: a waist chain locked tight around her waist like a belt, and a center chain locked to "
              "the front of the waist chain below her navel and to the back of the waist chain at the small of her back. Between those "
              "two points the center chain runs under her crotch, pulled tight over her vagina, and up between her buttocks, so all of "
              "it stays at her hips. It stays taut and centered the whole time and never hangs loose, slips to one side or sags."
              in notes(chained[0])
          and "Black-haired Dana unlocks the padlock at the small of Sarah's back, draws the center chain out from between her legs, "
              "then unlocks the padlock at the front of her waist and takes the crotch chain off her waist" in notes(chained[1])
          and "on: Sarah: handcuffs locking her wrists behind her back." in cuffed[3]["line"]
          and "By the last frame, Sarah no longer has her crotch chain on." in notes(chained[1])
          and "hold: Sarah, a crotch rope tied tight around her waist, its center rope pulled tight under her crotch and over her vagina "
              "from the front of her waist to the small of her back (" in roped["line"] and "ankles" not in roped["line"]
          and "comes off: Sarah: chains locked around her wrists behind her back., " in wrists["line"],
          (chained[0]["line"], notes(chained[0]), notes(chained[1]), same, roped["line"], wrists["line"], cuffed[3]["line"]))
    check("with her hands cuffed behind her, the one putting it on stands in front of her for the front and steps behind her only to "
          "lock the back, takes it off from behind, and no sentence names her hands or cuffs with it",
          "Black-haired Dana stands in front of Sarah and " + on.format(x="her", c="her vagina") in notes(cuffed[1])
          and "Black-haired Dana steps behind Sarah and unlocks the padlock at the small of her back, draws the center chain out from "
              "between her legs, then unlocks the padlock at the front of her waist and takes the crotch chain off her waist"
              in notes(cuffed[3])
          and "Her crotch chain is a piece of its own, complete in itself, with nothing else fastened to it anywhere." in notes(cuffed[2])
          and not MIXED(" ".join(s["prompt"] for s in cuffed), "crotch chain", "center chain", "waist chain"),
          [notes(s) for s in cuffed[1:]])
    legs = plan("Dana puts a crotch chain on Sarah.", "Sarah walks to the door.")[0]
    skirt = plan("Dana puts a crotch chain on Sarah.", m=MEM.replace("grey leggings", "a short black skirt and black panties"))[0][0]
    check("over clothes it lies on the outside of what covers her crotch, which stays on, a skirt stays over all of it, and her "
          "vagina is never named",
          "under her crotch, centers it tight over the outside of her grey leggings, then steps behind her" in notes(legs[0])
          and all("pulled tight over the outside of her grey leggings, and up between her buttocks" in notes(s)
                  and "Her grey leggings stay on under it the whole time, covering her." in notes(s) for s in legs)
          and "wraps its waist chain tight around Sarah's waist under her short black skirt and locks it at the front, below her navel, "
              "with a padlock, runs its center chain down from the front of the waist chain under her crotch, centers it tight over the "
              "outside of her black panties" in notes(skirt)
          and "Her black panties stay on under it the whole time, covering her. Her short black skirt stays on over all of it the whole "
              "time." in notes(skirt) and "in plain view" not in notes(skirt)
          and not any("vagina" in s["prompt"] for s in legs + [skirt]), [notes(s) for s in legs + [skirt]])


def test_tether():
    print("\n=== a collar chained to a fixture is one chain from its ring, never a chain on her wrists ===")
    lock = "Dana locks a steel collar around Sarah's neck."
    chain = "a chain clipped to the ring at the front of her collar, its other end padlocked to the middle of the headboard"
    said_ = [([lock + " She chains it to the headboard."], ""), ([lock + " She chains its ring to the headboard."], ""),
             ([lock, "Dana chains Sarah's collar to the headboard."], ""), ([lock, "Dana chains Sarah's collar ring to the headboard."], ""),
             (["Sarah kneels on the bed, her collar chained to the headboard."], ""),
             (["Sarah kneels on the bed."], "Sarah wears a steel collar chained to the headboard."),
             (["Sarah kneels on the bed."], "Sarah wears a steel collar with its ring chained to the headboard.")]
    lines = [plan(*b, a=a)[0][-1]["line"].split("cast: ")[-1] for b, a in said_]
    check("every way of chaining a collar to the headboard gives the collar and one chain from its ring, and nothing on her wrists",
          all(x.count(chain) == 1 and "collar locked around her neck" in x and "wrists" not in x for x in lines), lines)
    shots = plan("Sarah kneels on the bed.", "Sarah tugs at the chain.", "Dana watches her.",
                 a="Sarah wears a steel collar with its ring chained to the headboard, and her wrists are handcuffed behind her back.")[0]
    body = [notes(s) for s in shots[1:]]
    check("the chain runs from the middle of the headboard over her left shoulder, slack on the bed at her left side or taut when "
          "pulled, and she never lies down",
          all("Its only fixed points are the ring at her collar and the middle of the headboard." in b and "lies on her back" not in b
              for b in body) and "its slack lying loose on the bed at her left side" in body[2]
          and "Pulled taut, the chain runs in a straight line from her throat to the middle of the headboard" in body[1], body)
    cuffed = [plan(b, m="Sarah: a young woman with long blonde hair, in a white blouse.")[0][0]["line"] for b in
              ("Dana handcuffs Sarah to the radiator.", "Sarah is handcuffed to the bed post.")]
    check("handcuffing someone to a fixture without naming the wrists keeps what the cuffs are locked to",
          "chained to the radiator" in cuffed[0] and "chained to the bed post" in cuffed[1], cuffed)
    worn = [plan("Dana leads Sarah along the path.", m="Sarah: a young woman with long blonde hair, in a black top and a skirt, " + x
                 + ".")[0][0]["line"] for x in ("a steel collar with a chain leash, and her hands handcuffed behind her back",
                                                "her wrists cuffed behind her back", "and she is handcuffed behind her back",
                                                "her hair tied back in a ponytail", "a pair of handcuffs in her hands")]
    check("cuffs written as 'handcuffed' or 'cuffed' in a description are read without losing the collar and chain beside them, "
          "while hair tied back or cuffs carried in her hands are not restraints on her",
          all("handcuffs locking her wrists behind her back" in x for x in worn[:3]) and "a steel collar locked around her neck" in worn[0]
          and not any("on: " in x for x in worn[3:]), worn)
    mem_ = ("Sarah: a young woman with long blonde hair, in a black top, a steel collar with a chain leash.\nDana: a woman with short "
            "black hair.")
    pull = [plan("Dana leads Sarah along the path.", b, m=mem_)[0][1]["line"] for b in (
        "Sarah pulls back on her chain lead as Dana pushes her forward.", "Sarah pulls back on her chain leash, but Dana locks her grip "
        "on it.", "Dana unclips the chain leash from Sarah's collar.")]
    check("pulling on the leash she already has never puts a chain on anyone, a chain lead is her leash, and unclipping the leash "
          "from her collar leaves the collar on",
          all("goes on" not in x and "comes off" not in x for x in pull[:2]) and "on: Sarah: a steel collar locked around her neck., "
          "comes off: Sarah: a chain clipped to the ring at the front of her collar." in pull[2], pull)
    led = plan("Sarah waits.", m="Sarah: a young woman with long blonde hair, in a white blouse.\nDana: a woman with short black hair.\n"
                                 "hold: Sarah, a black collar; a chain leash held by Dana; handcuffs behind her back")[0][0]["line"]
    check("a chain leash is one leash from her collar, never a chain on her wrists as well",
          "a chain clipped to the ring at the front of her collar" in led and "chains locked around her wrists" not in led, led)


def test_car():
    print("\n=== a car door is named for what it is and where it is, and stays open or shut as left ===")
    road = "A quiet road with a black car parked at the edge."
    shots = plan("Dana opens the car door and Sarah climbs into the back seat.", "Dana slams the door shut and walks to the driver's "
                 "door.", "Dana pops the trunk, then shuts it.", scene=road)[0]
    car_ = [notes(s) for s in shots]
    check("a car door into the back seat is the rear door right behind the front door, its hinges and side are said, and it is "
          "open at the end of the shot",
          "a front door beside each front seat and a rear door right behind it beside the back seat" in car_[0]
          and "The back door is the rear door on the side nearest them, right behind the front door on that side, and it opens onto "
              "the back seat." in car_[0] and "By the last frame, the back door is open." in car_[0], car_[0])
    check("'the door' later is the one left open, the driver's door is on the steering wheel side, and a trunk opened then shut "
          "with 'it' ends shut",
          "By the last frame, the back door is shut." in car_[1] and "driver's door is the front door on the same side as the "
                                                                    "steering wheel" in car_[1]
          and "By the last frame, the driver's door" not in car_[1] and "The trunk is at the very back of the car" in car_[2]
          and "By the last frame, the trunk lid is shut." in car_[2] and "back door" not in car_[2], car_)
    lead = [notes(s) for s in plan("Dana pulls Sarah out of the back seat.", "Dana walks Sarah along the path.",
                                   scene="A park with a silver car parked beside the grass, its back door open.")[0]]
    check("a door in the scene's own text is placed too, a back door is the side door with the rest of the car closed, and it stays "
          "open in every shot until a beat shuts it",
          all("The back door is the rear door on the side nearest them" in x and "Apart from the back door, the car stays closed the "
              "whole shot, with every other door shut." in x and "By the last frame, the back door is open." in x for x in lead), lead)
    van = notes(plan("Dana slides open the side door of the van and pushes Sarah into the back.", scene="A parking lot with a white "
                                                                                                  "van.")[0][0])
    room = notes(plan("Dana opens the door and walks out.")[0][0])
    check("a van's side door is its sliding door, and a door in a room with no car gets no car notes",
          "The sliding door is in the side of the van, right behind the front passenger door" in van
          and "By the last frame, the sliding door is open." in van and "steering wheel" not in room and "car" not in room, (van, room))


def test_rules():
    print("\n=== the writing rules learned from renders ===")
    shots = plan("Dana cuffs Sarah's wrists behind her back.", "Dana locks a chastity belt on Sarah.",
                 "Dana wraps a crotch chain around Sarah's waist.", "Dana clips a leash to Sarah's collar and ties it to the bed post.",
                 m=MEM + "\nhold: Sarah, collar")[0]
    mixed = [t for s in shots for t in re.split(r"(?<=[.!?])\s+", s["prompt"]) if re.search(r"\b(?:hands?|wrists?|cuffs?|handcuffs)\b", t)
             and re.search(r"\b(?:chastity belt|crotch chain|leash|chain clipped)\b", t)]
    check("no sentence names her hands or cuffs together with the belt, the crotch chain or the leash", not mixed, mixed)
    side = plan("Dana chains Sarah's wrists to the headboard.", "Dana watches her.", "Dana walks to the door.")[0]
    check("whoever fastens her to a fixture does it from her left side, stays there while only watching, and the note drops when "
          "she moves",
          "does all of this from Sarah's left side" in notes(side[0]) and "stays at Sarah's left side" in notes(side[1])
          and "left side" not in notes(side[2]), [notes(s)[-600:] for s in side])
    cuffs = plan("Dana cuffs Sarah's wrists behind her back.", "Dana cuffs Sarah's ankles.")[0]
    check("cuffs go on from behind at the small of her back and are rigid, ankle cuffs go on crouching at her feet, her arms stay "
          "behind her, and no sentence names two restraints together",
          "steps behind Sarah, takes both of her wrists and draws her arms back behind her" in notes(cuffs[0])
          and "rigid steel" in notes(cuffs[0]) and "arms stay pulled behind her back" in notes(cuffs[1])
          and "crouches down at Sarah's feet, then opens one cuff, closes it around her left ankle" in notes(cuffs[1])
          and "separate restraints" not in notes(cuffs[1]), [notes(s) for s in cuffs])
    soft = plan("Dana puts black leather cuffs on Sarah's wrists in front of her.")[0]
    check("soft cuffs are leather and buckled, never rigid steel, and cuffs in front rest at belly height",
          "never rigid steel handcuffs" in notes(soft[0]) and "at the height of her belly" in notes(soft[0]), notes(soft[0]))


def test_clothes():
    print("\n=== clothes come off, go back on and stop at restraints ===")
    mem = ("Sarah <Picture 1>: a young woman with long blonde hair, in a white blouse, a red skirt and black panties.\n"
           "Dana: a woman with short black hair, in a black suit.")
    shots, warn = plan("Dana takes off Sarah's skirt.", "Sarah puts her skirt back on.", "Dana cuffs Sarah's wrists behind her back.",
                       "Dana takes off Sarah's blouse.", "Dana tears off Sarah's blouse.", m=mem)
    check("a garment comes off step by step, and the shot it comes off in already describes her without it, saying only that she "
          "starts the shot still in it, so a long shot never puts it back on; hidden underwear shows once uncovered, and it goes back on",
          "opens Sarah's red skirt at the waist and slides it down her legs" in notes(shots[0])
          and sheet(shots[0]) == "Sarah <Picture 1>: a young woman with long blonde hair, in a white blouse and black panties. Below the "
                                 "waist, she wears only her black panties. At the start of the shot she still has her red skirt on, until "
                                 "it comes off."
          and "in a white blouse and black panties" in sheet(shots[1]) and "puts on: Sarah: red skirt" in shots[1]["line"],
          [(sheet(s), s["line"]) for s in shots[:2]])
    check("a top cannot come off past cuffed wrists: it is pulled open instead and info says why; tearing it takes it off",
          "pulled: Sarah: white blouse open and down off her shoulders" in shots[3]["line"]
          and any("cannot come off past her bound wrists" in w for w in warn) and "clothes off: Sarah: white blouse" in shots[4]["line"],
          ([s["line"] for s in shots[3:]], warn))
    said = [plan(b, m=mem)[0][0] for b in ("Dana pulls Sarah's skirt down and takes it off.",
                                           "Dana pulls down Sarah's skirt. Sarah steps out of it and backs away from her.",
                                           "Dana takes off her suit and hands it to Sarah.")]
    check("'it' after a take-off verb is the garment named before it, a skirt pulled down then stepped out of ends up off, not still "
          "on her, and someone named after the garment is not its owner",
          "clothes off: Sarah: red skirt" in said[0]["line"] and "Once it is off, Sarah is in only her black panties" in notes(said[0])
          and "pulls Sarah's red skirt down to her knees. Sarah pulls her red skirt the rest of the way down her legs to her ankles, and "
              "she steps out of it one foot at a time, and drops it to the floor." in notes(said[1])
          and "still on her" not in notes(said[1]) and "clothes off: Dana: black suit" in said[2]["line"],
          [(s["line"], notes(s)) for s in said])
    road = plan("Dana takes off Sarah's skirt.", m=mem, scene="A quiet road beside a field.")[0][0]
    kept = plan("Dana takes off Sarah's skirt and throws it on the bed.", m=mem)[0][0]
    two = plan("Sarah pulls off her skirt.", m=mem + "\nAmy: a woman with red hair, in a blue skirt.")[0][0]
    check("a garment that comes off goes all the way: down to her ankles, she steps out of it, and it is dropped and stays where it "
          "fell, on the ground outdoors, unless the beat says where it goes; and two people in skirts do not stop the reading",
          "slides it down her legs to her ankles, and she steps out of it one foot at a time, and black-haired Dana drops it to the "
          "ground." in notes(road) and "and her red skirt stays on the ground where it fell, off her." in notes(road)
          and "drops it" not in notes(kept) and "where it fell" not in notes(kept) and "clothes off: Sarah: red skirt" in two["line"],
          [notes(road), notes(kept), two["line"]])
    down = plan("Dana yanks Sarah's skirt down to her ankles.", "Dana pulls Sarah out of the car by her cuffed wrists.",
                "Dana walks Sarah along the road.", m=mem + "\nhold: Sarah, handcuffs behind her back",
                scene="A quiet road with a black car parked at the edge.")[0]
    check("a skirt pulled down stays where it was left in later shots, said as where it rests and not as worn and then pulled, and "
          "pulling someone out of a car by her cuffed wrists leaves the cuffs on",
          sheet(down[1]).startswith("Sarah <Picture 1>: a young woman with long blonde hair, in a white blouse and black panties. Her "
                                    "red skirt is bunched around her ankles, and it stays there.")
          and "pulled down" not in sheet(down[2]) and "release" not in down[1]["line"]
          and "handcuffs locking her wrists behind her back" in down[2]["line"], [(sheet(s), s["line"]) for s in down])
    legs = plan("Dana ties Sarah's ankles together with rope.", "Dana pulls off Sarah's panties.", m=mem)[0]
    check("bottoms cannot come off past bound ankles, so they stop just above the rope",
          "pulled: Sarah: black panties down to just above the rope on her ankles" in legs[1]["line"], legs[1]["line"])


def test_diaper():
    print("\n=== a diaper is underwear: never named under clothes, named when a squat shows it, there once they come off ===")
    mem = "Sarah <Picture 1>: a young woman with long blonde hair, in a pink t-shirt, blue jeans and a white diaper."
    jeans = plan("Sarah walks to the couch.", "Sarah takes off her jeans.", m=mem)[0]
    check("under jeans the diaper is never in the picture text, and with the jeans off she is in it, not naked",
          "diaper" not in jeans[0]["prompt"] and "in a pink t-shirt and a white diaper" in sheet(jeans[1])
          and "At the start of the shot she still has her blue jeans on, until they come off." in sheet(jeans[1])
          and "Once they are off, Sarah is in only her white diaper below the waist for the rest of the shot, and her blue jeans stay on "
              "the floor where they fell, off her." in notes(jeans[1])
          and "naked" not in notes(jeans[1]), [s["prompt"] for s in jeans])
    skirt = plan("Sarah walks to the couch.", "Sarah squats down to pick up a cup.", m=mem.replace(
        "in a pink t-shirt, blue jeans and a white diaper", "wearing a short pleated skirt over a thick white diaper, and white "
                                                            "sneakers"))[0]
    check("a skirt over a diaper is two garments: the diaper is left out while she walks and named under the skirt when she squats",
          sheet(skirt[0]) == "Sarah <Picture 1>: a young woman with long blonde hair, wearing a short pleated skirt and white sneakers."
          and sheet(skirt[1]).endswith("Below the waist, she wears only her short pleated skirt, and under it nothing but her thick white "
                                       "diaper."), [sheet(s) for s in skirt])
    under = plan("Sarah walks to the couch.", m=mem.replace("in a pink t-shirt, blue jeans and a white diaper",
                                                            "wearing a diaper under her grey leggings and a hoodie, her hair falling over "
                                                            "her shoulders"))[0]
    check("'a diaper under her leggings' leaves just the leggings, and words like 'over her shoulders' are left alone",
          sheet(under[0]) == "Sarah <Picture 1>: a young woman with long blonde hair, wearing her grey leggings and a hoodie, her hair "
                             "falling over her shoulders.", sheet(under[0]))
    shown = plan("Dana lifts up the front of Sarah's skirt to show her diaper.", m=mem.replace(
        "in a pink t-shirt, blue jeans and a white diaper", "in a pink t-shirt and a short skirt over a white diaper"))[0]
    check("lifting a skirt to show the diaper lifts only the skirt, says the diaper shows once it is up, and does not stand anyone up",
          "lift: Sarah, skirt up (" in shown[0]["line"] and "diaper up" not in shown[0]["line"] and "pose:" not in shown[0]["line"]
          and "Once it is up, her white diaper shows underneath." in notes(shown[0]), (shown[0]["line"], notes(shown[0])))


def test_belt():
    print("\n=== a chastity belt is worn on bare skin, with no underwear, and under any clothes ===")
    mem = ("Sarah <Picture 1>: a young woman with long blonde hair, in a white blouse and black panties.\n"
           "Dana: a woman with short black hair, in a black suit.")
    bare = "right on her bare skin, and she wears no panties or any other underwear with it."
    fresh, warn = plan("Dana locks a chastity belt on Sarah.", "Sarah pulls her panties back on.", m=mem)
    check("locking a belt on takes her panties off first and never lets them back on while it is locked",
          "slides Sarah's black panties down her legs to her ankles, and she steps out of them one foot at a time, and black-haired Dana "
          "drops them to the floor, then fits the chastity belt around her bare waist"
          in notes(fresh[0]) and "Below the waist, she wears only her chastity belt." in sheet(fresh[1])
          and any("black panties cannot go on while her chastity belt is locked on" in w for w in warn), (notes(fresh[0]), warn))
    jeans = plan("Dana locks a chastity belt on Sarah.", m=mem.replace(" and black panties", ", blue jeans and pink panties"))[0]
    check("jeans over her panties come off first too", "clothes off: Sarah: blue jeans; pink panties." in jeans[0]["line"],
          jeans[0]["line"])
    shorts = plan("Sarah walks to the door.", "Dana takes off Sarah's shorts.",
                  m=mem.replace(" and black panties", ", black latex shorts and a steel chastity belt"))[0]
    check("a belt in her description is read and her panties are gone; under shorts that stay on, the picture never names it, and the "
          "shot they come off in says it is under them and shows once they are off",
          "on: Sarah: a steel chastity belt locked around her waist" in shorts[0]["line"]
          and sheet(shorts[0]) == "Sarah <Picture 1>: a young woman with long blonde hair, in a white blouse and black latex shorts. "
                                  "Below the waist, she wears only her black latex shorts."
          and "chastity" not in notes(shorts[0]) and "chastity" not in beat(shorts[0])
          and f"Under her black latex shorts, her chastity belt is locked on her, its band around her waist and its shield between her legs "
              f"over her crotch, {bare} It comes into view only once they are out of the way." in sheet(shorts[1])
          and "apart from a steel chastity belt" in notes(shorts[1]),
          [(sheet(s), notes(s)[:300]) for s in shorts])
    skirt = plan("Sarah walks to the shelf.", "Sarah squats down to pick up a book, then stands up.", "Sarah reads the book.",
                 m=mem.replace(" and black panties", ", a short black skirt and a steel chastity belt"))[0]
    check("under a skirt the belt is never named while she stands, and a squat, even one she stands up from, names it as all she "
          "wears under the skirt",
          "chastity" not in skirt[0]["prompt"] and "chastity" not in skirt[2]["prompt"]
          and sheet(skirt[1]).endswith("Below the waist, she wears only her short black skirt, and under it nothing but her steel "
                                       "chastity belt.")
          and "chastity" not in notes(skirt[1]), [sheet(s) for s in skirt])
    dress = plan("Sarah walks to the table.", "Sarah bends over the table.", m=mem.replace(
        " in a white blouse and black panties", " in a blue summer dress, wearing a steel chastity belt"))[0]
    check("a dress hangs loose the same way", "chastity" not in dress[0]["prompt"]
          and "only her blue summer dress, and under it nothing but her steel chastity belt." in sheet(dress[1]),
          [sheet(s) for s in dress])
    worn, warn = plan("Sarah, locked in a chastity belt, kneels on the bed.", "Dana pulls off Sarah's panties.", m=mem)
    check("being locked in one is read and leaves her panties off for good, and pulling them off again only warns",
          "black panties" not in sheet(worn[0]) and any("panties are already off: nothing is worn under her chastity belt" in w
                                                        for w in warn),
          (sheet(worn[0]), warn))


def test_speech():
    print("\n=== who says each line, and the sound of each shot ===")
    shots, warn = plan('Dana walks up to Sarah. "You are late. Sit down." she says.', '"Sorry," Sarah says.', 'Dana: "Good."',
                       'Sarah looks away. "He will never come," she thinks.', "Dana gags Sarah with a ball gag. Sarah screams into the gag.")
    check("a line goes to its speaker in H3's spoken form, a long line is split into its sentences, and a thought is not spoken",
          "Black-haired Dana says: <d>You are late.</d> Black-haired Dana continues: <d>Sit down.</d>" in beat(shots[0])
          and "Sarah says: <d>Sorry</d>." in beat(shots[1]) and "says: <d>Good.</d>" in beat(shots[2])
          and '"He will never come," she thinks.' in beat(shots[3]) and not shots[3]["cues"], [beat(s) for s in shots])
    check("each line gets its own window after the actions before it, the speaker alone moves her lips, and the shot is not silent",
          shots[0]["cues"][0][0] >= 2.5 and shots[0]["cues"][0][2] == "line" and not shots[0]["silent"]
          and "Only black-haired Dana speaks in this shot" in notes(shots[0]) and "listens with closed lips" in notes(shots[0]),
          (shots[0]["cues"], notes(shots[0])))
    check("a gagged scream is a muffled window with no words, the shot is rendered silent, and its sound names the muffled sounds",
          [k for _, _, k in shots[4]["cues"]] == ["muffle"] and shots[4]["silent"]
          and "muffled, closed-mouth mmmph sounds through the ball gag" in shots[4]["sound"], (shots[4]["cues"], shots[4]["sound"]))
    check("every sound prompt is in H3's sound form, with the place and what happens and no spoken words",
          all(s["sound"].startswith("integrated_multimodal_description: [Shot 1] A bedroom with a bed.") and "<d>" not in s["sound"]
              and "late" not in s["sound"] and s["sound"].endswith("non_diegetic_music: N/A") for s in shots), [s["sound"] for s in shots])
    writing = plan('Dana writes "MINE" across Sarah\'s stomach.')[0]
    check("a quoted word being written is not a spoken line", not writing[0]["cues"] and '"MINE"' in beat(writing[0]), beat(writing[0]))
    park = plan("Dana closes the car door. Dana pulls Sarah toward the park to show her off. Sarah tries to pull back, due to being "
                "embarrassed. Sarah struggles.", scene="A parking lot beside a busy park.")[0]
    check("the sound prompt keeps only what makes a sound: no motives, tries or feelings that H3 could read out, and an outdoor "
          "place gets a breeze",
          "closes the car door. Black-haired Dana pulls Sarah toward the park. Sarah struggles. Black-haired Dana and Sarah stay silent."
          in park[0]["sound"] and "embarrassed" not in park[0]["sound"] and "show her off" not in park[0]["sound"]
          and "A light breeze moves through the open air." in park[0]["sound"], park[0]["sound"])
    order = plan('Dana tugs the leash and she tells Sarah: "Kneel down. Look at me."', '"Stay," Dana says, pointing at the floor.')[0]
    check("a tag inside an action sentence is written once in H3's spoken form, an order is not read as a pose, a tag's trailing "
          "action stays, and the sound around the line keeps everyone silent with no dangling tag",
          "Black-haired Dana tugs the leash. Black-haired Dana says: <d>Kneel down.</d> Black-haired Dana continues: <d>Look at me.</d>"
          in beat(order[0]) and "tells" not in beat(order[0]) and "kneeling" not in notes(order[0])
          and "Black-haired Dana tugs the leash. Black-haired Dana and Sarah stay silent." in order[0]["sound"]
          and "Black-haired Dana says: <d>Stay</d>, pointing at the floor." in beat(order[1]),
          ([beat(s) for s in order], notes(order[0])[-300:], order[0]["sound"]))


def test_poses():
    print("\n=== a pose holds until a beat changes it ===")
    shots = plan("Dana throws Sarah onto the bed.", "Dana cuffs Sarah's wrists behind her back.", "Sarah gets on all fours.",
                 "Dana pulls Sarah to her feet.")[0]
    check("thrown onto the bed she stays lying there while cuffed, on all fours with cuffed hands she kneels bent forward, and pulled "
          "up she ends standing",
          "ends the shot lying on the bed" in notes(shots[0]) and "stays lying on the bed for the whole shot" in notes(shots[1])
          and "ends the shot kneeling bent forward" in notes(shots[2]) and "ends the shot standing" in notes(shots[3]),
          [notes(s)[-400:] for s in shots])
    sex = plan("Dana bends Sarah over the bed.", "Marcus fucks Sarah from behind.",
               m=MEM + "\nMarcus: a man with short dark hair, in a blue shirt.")[0]
    check("bent over a bed her chest rests on it, and penetration from behind lines up with her body",
          "chest rests on it" in notes(sex[0]) and "right behind Sarah, his hips against her buttocks" in notes(sex[1]),
          [notes(s)[-500:] for s in sex])


def test_pictures():
    print("\n=== pictures ===")
    forms = ["Sarah <Picture 1>: a woman.", "Sarah: <Picture 1> a woman.", "Sarah: a woman. <Picture 1>", "Sarah (<Picture 1>): a woman.",
             "**Sarah** <Picture 1>: a woman.", "- Sarah <Picture 1>: a woman.", "SARAH - a woman <Picture 1>"]
    check("the picture is Sarah's however the line is written, and goes in every shot she is in",
          all(all(s["refs"] == [0] and "Sarah <Picture 1>:" in s["prompt"] for s in plan("Sarah waits.", "Sarah sits.", m=f)[0])
              for f in forms), [plan("Sarah waits.", m=f)[0][0]["refs"] for f in forms])
    two = plan("Dana waits.", "Sarah walks in.", m=MEM.replace("Dana:", "Dana <Picture 3>:"))[0]
    check("pictures go in for the people in view, renumbered from 1, with the opening frame after them",
          two[0]["refs"] == [2] and "Dana <Picture 1>:" in two[0]["prompt"] and two[1]["refs"] == [0, 2]
          and "<Picture 3> is the frame this shot opens on" in two[1]["prompt"] and two[1]["coming"] == ["Sarah"],
          [(s["refs"], s["coming"]) for s in two])
    belt = plan("Dana locks the chastity belt <Picture 2> on Sarah.", "Sarah walks.",
                m=MEM.replace("leggings", "leggings and black panties"))[0]
    check("a picture named for the belt goes in with it, never cropped to a face",
          belt[1]["refs"] == [0, 1] and belt[1]["faces"] == [True, False] and "Her chastity belt <Picture 2>" in belt[1]["prompt"],
          [(s["refs"], s["faces"]) for s in belt])


def test_lengths():
    print("\n=== shot lengths ===")
    shots, _ = plan("Sarah waits.", "seconds: 12\nSarah sits.", "Dana talks to Sarah. " + '"' + " ".join(["word"] * 40) + '," she says.',
                    shot_seconds=6.0)
    check("every length is 17k+5 frames, short beats get the floor, seconds: sets it exactly, and a long line gets the time it needs",
          all((s["frames"] - 5) % 17 == 0 for s in shots) and shots[0]["frames"] == 124 and shots[1]["frames"] == P.align(12 * 24)
          and shots[2]["frames"] > P.align(6 * 24) and "lengthened to fit its line" in shots[2]["line"], [s["frames"] for s in shots])
    fixed = plan("Sarah waits.", "Sarah sits.", from_beat=False, shot_seconds=8.0)[0]
    check("fixed gives every shot shot_seconds", [s["frames"] for s in fixed] == [P.align(8 * 24)] * 2, [s["frames"] for s in fixed])
    crowd = plan("Sarah waits.", "A guard walks in and stops beside Sarah.")[0]
    check("a head count is given for named people only, and left out when someone unnamed is in the beat",
          "There is one person in the shot" in notes(crowd[0]) and "in the shot" not in notes(crowd[1]).split("There is only")[0][-200:]
          and "people in the shot" not in notes(crowd[1]), [notes(s)[-300:] for s in crowd])
    cut = plan("Sarah waits.", "cut\n\nDana drives.")[0]
    check("a cut starts fresh with its own cast, and a scene paragraph naming nobody is kept at the top of every shot",
          cut[1]["cut"] and not cut[1]["keyed"] and cut[1]["cast"] == ["Dana"] and all(s["prompt"].startswith("A bedroom") for s in cut),
          [(s["cut"], s["cast"]) for s in cut])


class Patched:
    def __init__(self, wrappers=None):
        self.wrappers, self.model_options, self.patches = wrappers or {}, {"transformer_options": {}}, []

    def clone(self):
        c = Patched(dict(self.wrappers))
        c.model_options = {"transformer_options": dict(self.model_options["transformer_options"])}
        return c

    def get_model_object(self, name):
        return SimpleNamespace(set_parameters=lambda shift=1.0, audio_shift=1.0: None, sigmas=torch.linspace(1, 0, 10))

    def add_object_patch(self, *a):
        self.patches.append(a)

    def add_wrapper_with_key(self, kind, key, fn):
        self.wrappers.setdefault(kind, {})[key] = fn


def test_models():
    print("\n=== model setup ===")
    heard = lambda x: 3 * (float(x) / (12 - 11 * float(x))) / (1 + 2 * float(x) / (12 - 11 * float(x)))
    pdd = Patched({pe.WrappersMP.DIFFUSION_MODEL: {modelprep.PDD_KEY: [None]}})
    m, steps, sampler, sigmas, fine, notes = modelprep.prepare(pdd, 8, "res_multistep", "simple", torch.linspace(1, 0, 9), 12.0,
                                                               3.0, {}, "9")
    tops = m.model_options["transformer_options"]
    check("PDD Acc samples on its own sigmas with euler at shift 12/3",
          sampler == "euler" and tops["minimax_h3_sigma_shift_video"] == 12.0 and "PDD Acc" in notes[0] and fine[0] is sigmas
          and torch.allclose(fine[1], sigmas[-len(fine[1]):]) and heard(fine[1][0]) <= 0.88 < heard(sigmas[-len(fine[1]) - 1]), notes)
    try:
        modelprep.prepare(pdd, 8, "euler", "simple", None, 12.0, 3.0, {}, "9")
        check("PDD Acc without its sigmas stops and says so", False)
    except ValueError as e:
        check("PDD Acc without its sigmas stops and says so", "sigmas" in str(e))
    graph = {"9": {"inputs": {"sigmas": ["14", 1], "model": ["11", 0]}}, "14": {"class_type": modelprep.PDD_NODE, "inputs": {}},
             "11": {"inputs": {"unet_name": "x.safetensors"}}}
    saved = (modelprep.loras, modelprep.meta, modelprep.checkpoint)
    try:
        modelprep.loras = lambda g, n=None: ["minimax_h3_fl2v_turbo_8step.safetensors"]
        modelprep.meta = lambda p: (set(), {})
        modelprep.checkpoint = lambda g, n: ("minimax_h3_hybrid_ref2va.safetensors", "")
        _, steps, sampler, sigmas, _, notes = modelprep.prepare(Patched(), 8, "res_multistep", "simple", torch.linspace(1, 0, 9),
                                                                12.0, 3.0, graph, "9")
        check("stray PDD sigmas are dropped, and a few-step LoRA is sampled with euler on the steps set",
              sampler == "euler" and steps == 8 and sigmas is None and "left out" in notes[0], notes)
        modelprep.loras = lambda g, n=None: ["flashgen_4step.safetensors"]
        _, steps, sampler, sigmas, fine, notes = modelprep.prepare(Patched(), 8, "res_multistep", "simple", None, 12.0, 3.0, {}, "9")
        unshift = lambda t: [round(float(x) / (12 - 11 * float(x)), 4) for x in t]
        check("FlashGen gets its own 4-step grid shifted at 12, and its sound passes twice as many steps, each spoken line said again "
              "over the last six",
              steps == 4 and abs(float(sigmas[1]) - 12 * 0.7 / (1 + 11 * 0.7)) < 1e-6
              and unshift(fine[0]) == [1.0, 0.85, 0.7, 0.55, 0.4, 0.275, 0.15, 0.075, 0.0] and unshift(fine[1]) == unshift(fine[0])[2:]
              and any("said again over the last 6" in n for n in notes), ([unshift(f) for f in fine], notes))
        check("a turbo LoRA or a T2V checkpoint takes no pictures, a ref2va checkpoint with a style LoRA does",
              not modelprep.takes_pictures(Patched(), {}, "9")[0]
              and (setattr(modelprep, "loras", lambda g, n=None: ["Astro nsfw.safetensors"]) or modelprep.takes_pictures(Patched(), {}, "9")[0]))
    finally:
        modelprep.loras, modelprep.meta, modelprep.checkpoint = saved
    head = json.dumps({"hyperflow_sigmas": {"dtype": "F32", "shape": [7], "data_offsets": [0, 28]}}).encode()
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        f.write(len(head).to_bytes(8, "little") + head + bytes(28))
    saved, flash = modelprep.full_path, {"9": {"inputs": {"model": ["11", 0]}}, "11": {"inputs": {"model": "flash.safetensors"}}}
    try:
        modelprep.full_path = lambda kind, name: f.name if kind == "diffusion_models" and name == "flash.safetensors" else ""
        got = {n: modelprep.prepare(Patched(), n, "res_multistep", "simple", None, 12.0, 3.0, flash, "9") for n in (4, 6, 8, 20)}
    finally:
        modelprep.full_path = saved
        os.unlink(f.name)
    check("a HyperFlow H3 Flash checkpoint, from any loader, is sampled with euler at shift 12/3 on its own 4-, 6- or 8-step grid, "
          "the 6-step one being the sigmas stored in the file",
          [g[1] for g in got.values()] == [4, 6, 8, 8] and {g[2] for g in got.values()} == {"euler"}
          and [round(float(x), 6) for x in got[6][3]] == [1.0, 0.982656, 0.914643, 0.674756, 0.385224, 0.159055, 0.0]
          and got[6][5] == ["HyperFlow H3 Flash: 6 steps, shift 12/3, euler"], {n: (g[1], g[5]) for n, g in got.items()})
    check("on Flash's 4-step grid the sound passes run on its 8-step grid and each spoken line is said again from its fourth step, "
          "while 6 or more steps keep their own schedule and say each line again from the same noise level",
          torch.allclose(got[4][4][0], modelprep.shifted(modelprep.FLASH[8], 12.0)) and torch.allclose(got[4][4][1], got[4][4][0][3:])
          and all(got[n][4][0] is got[n][3] and heard(got[n][4][1][0]) <= 0.88 for n in (6, 8, 20)), got[4][4])
    land = modelprep.landing([1.0, 0.8, 0.6, 0.0], 12.0, 3.0)
    check("a coarse last step gets one audio landing step, a fine one none", len(land) == 5 and 0 < land[-2] < 0.6
          and modelprep.landing([1.0, 0.5, 0.2, 0.0], 12.0, 3.0) == [1.0, 0.5, 0.2, 0.0], land)


class FakeClip:
    def tokenize(self, text, **kw):
        return {"text": text, "items": kw.get("minimax_ref_items")}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1), {"tokens": tokens}]]


class FakeVAE:
    def encode(self, img):
        return torch.full((1, 24, 1, int(img.shape[1]) // 16, int(img.shape[2]) // 16), float(img.mean()))


class FakeAudioVAE:
    audio_sample_rate = 1000

    def encode(self, wave):
        return torch.zeros(1, 32, 2, 80)


def test_flash():
    print("\n=== a HyperFlow H3 Flash checkpoint from ComfyUI's own loader is filled in and conditioned on each step's endpoint ===")
    from comfy.ldm.minimax.model import MLP, Attention, TimeEmbedder, time_shift_sigma
    from safetensors.torch import save_file
    gen = torch.Generator().manual_seed(5)
    rand = lambda *shape: torch.randn(*shape, generator=gen) * 0.2
    hidden, inner, ffn, rank = 16, 16, 24, 3
    factors = {f"blocks.0.attn.{c}_proj.{w}.weight": rand(*((rank, hidden) if w == "down" else (inner, rank))) for c in "qkv"
               for w in ("down", "up")}
    factors.update({"blocks.0.attn.out_proj.down.weight": rand(rank, inner), "blocks.0.attn.out_proj.up.weight": rand(hidden, rank),
                    "blocks.0.mlp.fc1.down.weight": rand(rank, hidden), "blocks.0.mlp.fc1.up.weight": rand(2 * ffn, rank),
                    "blocks.0.mlp.fc2.down.weight": rand(rank, ffn), "blocks.0.mlp.fc2.up.weight": rand(hidden, rank),
                    "endpoint_time_embedder.proj_in.weight": rand(32, 8), "endpoint_time_embedder.proj_in.bias": rand(32),
                    "endpoint_time_embedder.proj_out.weight": rand(4, 32), "endpoint_time_embedder.proj_out.bias": rand(4),
                    "hyperflow_gate": torch.tensor([0.25]), "hyperflow_sigmas": torch.linspace(1, 0, 7)})
    path = os.path.join(tempfile.mkdtemp(), "flash.safetensors")
    save_file(factors, path)
    block = torch.nn.Module()
    block.attn = Attention(hidden, 2, 8, 1e-6, operations=torch.nn)
    block.mlp = MLP(hidden, ffn, operations=torch.nn)
    for layer in (block.attn.qkv_proj, block.attn.out_proj, block.mlp.fc1, block.mlp.fc2):
        layer.weight = None
    net = torch.nn.Module()
    net.blocks = torch.nn.ModuleList([block])
    net.time_embedder = TimeEmbedder(8, 32, 4, operations=torch.nn)
    model = SimpleNamespace(get_model_object=lambda name: net, size=99)
    saved = modelprep.model_file
    try:
        modelprep.model_file = lambda g, n: ("flash.safetensors", path)
        modelprep.mend(model, {}, "9")
        modelprep.model_file = lambda g, n: ("other.safetensors", "")
        broken = Attention(hidden, 2, 8, 1e-6, operations=torch.nn)
        broken.qkv_proj.weight = None
        try:
            modelprep.mend(SimpleNamespace(get_model_object=lambda name: broken), {}, "9")
            refused = ""
        except RuntimeError as e:
            refused = str(e)
    finally:
        modelprep.model_file = saved
    dense = lambda c: factors[f"{c}.up.weight"] @ factors[f"{c}.down.weight"]
    ref = torch.nn.Module()
    ref.attn = Attention(hidden, 2, 8, 1e-6, operations=torch.nn)
    ref.mlp = MLP(hidden, ffn, operations=torch.nn)
    ref.attn.load_state_dict({"qkv_proj.weight": torch.cat([dense(f"blocks.0.attn.{c}_proj") for c in "qkv"]),
                              "out_proj.weight": dense("blocks.0.attn.out_proj"), "q_norm.weight": block.attn.q_norm.weight,
                              "k_norm.weight": block.attn.k_norm.weight})
    ref.mlp.load_state_dict({"fc1.weight": dense("blocks.0.mlp.fc1"), "fc2.weight": dense("blocks.0.mlp.fc2")})
    x = rand(5, hidden)
    with torch.no_grad():
        same = torch.allclose(block.attn(x), ref.attn(x), atol=1e-5) and torch.allclose(block.mlp(x), ref.mlp(x), atol=1e-5)
    check("the empty layers are filled from the checkpoint's low-rank factors, q|k|v fused in ComfyUI's order and the feed-forward "
          "running through ComfyUI's own swiglu, matching dense weights exactly, and a checkpoint that cannot fill them is refused",
          same and model.size == 0 and isinstance(block.attn.qkv_proj, modelprep.LowRank) and "load it with the loader" in refused,
          refused)
    end = net.endpoint_time_embedder
    t = torch.tensor([0.4, 0.7, 0.999])
    with torch.no_grad():
        plain = net.time_embedder(t)
        end.step = ((0.4, 0.6), (0.7, 0.8))
        mixed = net.time_embedder(t)
        want = plain + 0.25 * (end(torch.tensor([0.6, 0.8, 0.999])) - plain)
        end.step = None
    seen = []
    run = lambda *a, **k: seen.append(end.step) or "ran"
    tops = {"sample_sigmas": torch.tensor([1.0, 0.98, 0.9, 0.0]), "minimax_h3_sigma_shift_video": 12.0, "minimax_h3_sigma_shift_audio": 3.0}
    got = modelprep.stepped(end, run, None, torch.tensor([980.0]), None, tops)
    heard = lambda v: float(time_shift_sigma(torch.tensor(v), 12.0, 3.0))
    (tv, rv), (ta, ra) = seen[0]
    check("each distinct time t is blended with its endpoint r by the stored gate, the video and audio rows step to the next sigma "
          "on their own shifts, conditioning rows keep r = t, and nothing is left set after the step",
          torch.allclose(mixed, want, atol=1e-6) and got == "ran" and end.step is None and abs(tv - 0.02) < 1e-6
          and abs(rv - 0.1) < 1e-6 and abs(ta - (1 - heard(0.98))) < 1e-6 and abs(ra - (1 - heard(0.9))) < 1e-6, seen)

    class Mended(Patched):
        def clone(self):
            c = Mended(dict(self.wrappers))
            c.model_options = {"transformer_options": dict(self.model_options["transformer_options"])}
            return c

        def get_model_object(self, name):
            return SimpleNamespace(set_parameters=lambda shift=1.0, audio_shift=1.0: None, endpoint_time_embedder=end)
    m, steps, sampler, sigmas, _, notes = modelprep.prepare(Mended(), 6, "euler", "simple", torch.linspace(1, 0, 7), 12.0, 3.0, {},
                                                            "9")
    check("a mended model samples with the endpoint wrapper on",
          "h3_longvideos_endpoint" in m.wrappers.get(pe.WrappersMP.DIFFUSION_MODEL, {})
          and any("two-time conditioning filled in" in n for n in notes), notes)


def test_chain():
    print("\n=== conditioning, look and the hand-off ===")
    sr, gen = 16000, torch.Generator().manual_seed(3)
    t = torch.arange(5 * sr) / sr
    voice = sum(torch.sin(2 * torch.pi * 180 * k * t) / k for k in range(1, 6)) * 0.3
    room = torch.randn(5 * sr, generator=gen) * 0.01
    burst = lambda a, b: ((t >= a) & (t < b)).float()
    wave = (room + voice * (burst(1.0, 1.5) + burst(3.0, 3.5))).view(1, 1, -1).repeat(1, 2, 1)
    quiet = chain.unvoice(wave.clone(), sr, [(2.5, 4.0)])
    level = lambda x, a, b: float(x[0, 0, int(a * sr):int(b * sr)].pow(2).mean().sqrt())
    check("a voice in the gaps around a line is brought down to the room's own level, while the line and the room tone stay as they "
          "were", level(quiet, 1.1, 1.4) < 3 * level(wave, 0.2, 0.8) and abs(level(quiet, 3.1, 3.4) - level(wave, 3.1, 3.4)) < 1e-4
          and abs(level(quiet, 0.2, 0.8) - level(wave, 0.2, 0.8)) < 1e-4 and chain.gaps([(1.0, 2.0), (3.0, 4.0)], 5.0)
          == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)], [level(quiet, 1.1, 1.4), level(wave, 0.2, 0.8)])
    clinks = sum(torch.sin(2 * torch.pi * f * t) * 0.7 ** k for k, f in enumerate((1240.0, 2870.0, 4410.0))) * torch.exp(-(t % 0.5) * 18)
    thumps = sum(((t >= a) & (t < a + 0.2)).float() * torch.sin(2 * torch.pi * 90 * (t - a)) * torch.exp(-(t - a).clamp(min=0) * 40)
                 for a in (0.5, 1.1, 1.7, 2.3, 2.9, 3.5, 4.1))
    kept = [level(chain.unvoice(x.clone(), sr, []), 0, 5) / level(x, 0, 5)
            for x in ((room + 0.3 * clinks).view(1, 1, -1).repeat(1, 2, 1), (room + 0.25 * thumps).view(1, 1, -1).repeat(1, 2, 1))]
    check("chains and cuffs clinking and footsteps thudding keep their full level: only a voice is brought down",
          all(abs(k - 1) < 0.02 for k in kept), kept)
    hand, ref, mem = torch.full((1, 64, 96, 3), 0.5), torch.full((1, 32, 32, 3), 0.2), torch.full((1, 64, 96, 3), 0.7)
    cond, lat = chain.build(FakeClip(), FakeVAE(), FakeAudioVAE(), "x", 96, 64, 41, hand, [ref], mem, silent=True)
    vals = cond[0][1]
    check("pictures go in as references in order, the opening frame as the keyframe, the memory frame last",
          [round(float(i["data"].mean()), 1) for i in vals["tokens"]["items"]] == [0.2, 0.5, 0.7]
          and len(vals["minimax_refs"]) == 2 and abs(float(vals["minimax_keyframes"][0]["latent"].mean()) - 0.5) < 1e-6)
    check("a frame already at the shot's size goes in untouched, with no resampling or 8-bit rounding",
          chain.resize(hand, 96, 64) is not None and torch.equal(chain.resize(hand, 96, 64), hand[..., :3]))
    check("a silent shot pins all of its sound to silence", float(lat["noise_mask"].unbind()[1].max()) == 0.0)
    order, memo = [], {}
    vae = SimpleNamespace(encode=lambda img: order.append("vae") or FakeVAE().encode(img))
    clip = SimpleNamespace(tokenize=FakeClip().tokenize, encode_from_tokens_scheduled=lambda t: order.append("text") or [[torch.zeros(1), {}]])
    for _ in range(3):
        chain.build(clip, vae, FakeAudioVAE(), "x", 96, 64, 41, hand, [ref], mem, memo=memo)
    check("each reference picture and memory frame is encoded once per run, the opening frame every shot, and all of it "
          "before the text encoder runs", order == ["vae", "vae", "vae", "text", "vae", "text", "vae", "text"], order)
    store = chain.spool(6, (4, 5, 3), torch.float16)
    store[2:4].copy_(torch.full((2, 4, 5, 3), 0.25))
    maps = open("/proc/self/maps").read() if os.path.exists("/proc/self/maps") else ""
    check("finished frames go to a file-backed buffer whose temp file is already deleted, and read back exactly",
          float(store[2:4].float().mean()) == 0.25 and float(store[:2].float().abs().max()) == 0 and tuple(store.shape) == (6, 4, 5, 3)
          and store.dtype == torch.float16 and "h3longvideos-" in maps and "(deleted)" in maps
          and not [n for n in os.listdir(tempfile.gettempdir()) if n.startswith("h3longvideos-")],
          [line for line in maps.splitlines() if "h3longvideos" in line])
    _, lat = chain.build(FakeClip(), FakeVAE(), FakeAudioVAE(), "x", 96, 64, 41, cues=[(0.25, 0.5), (1.0, 1.25)])
    mask = lat["noise_mask"].unbind()[1][0, 0, 0]
    check("a spoken shot has sound only in each line's window and is silent before, between and after",
          float(mask[:10].max()) == 0 and float(mask[10:20].min()) == 1 and float(mask[20:40].max()) == 0
          and float(mask[40:50].min()) == 1 and float(mask[50:].max()) == 0, mask)
    seen, real = {}, chain.sample
    chain.sample = lambda m, c, n, lat, *a: seen.update(lat=lat) or dict(lat)
    try:
        sound = torch.rand(1, 32, 2, 68)
        chain.foley(None, [[None, {}]], None, {}, {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 2, 4, 6), sound))},
                    0, 8, "euler", "simple", None, 1.0, [(0.5, 1.0)])
    finally:
        chain.sample = real
    still, heard = seen["lat"]["noise_mask"].unbind()
    check("a muffled sound is laid into its own window, and the shot's sound everywhere else is kept",
          torch.equal(seen["lat"]["samples"].unbind()[1], sound) and float(still.max()) == 0 and float(heard[..., 20:40].min()) == 1
          and float(heard[..., :20].max()) == 0 and float(heard[..., 40:].max()) == 0)
    gen = torch.Generator().manual_seed(1)
    base = torch.rand(1, 96, 128, 3, generator=gen) * 0.6 + 0.2
    grey = lambda x, k: (x.mean(-1, keepdim=True) + (x - x.mean(-1, keepdim=True)) * k).clamp(0, 1)
    pale = torch.cat([grey(base, 0.6)] * 6)
    chain.apply(pale, chain.grade(base, pale[1], pale[-1]))
    check("a continued shot that opens washed out is graded back onto the frame it continues from",
          abs(chain.look(pale[3])[1] / chain.look(base[0])[1] - 1) < 0.05, (chain.look(pale[3]), chain.look(base[0])))
    soft = chain.soften(base, (1.1, 1.2))
    check("the hand-off is toned down by the boost H3 adds, never past a quarter, and left alone with no boost",
          abs(chain.look(soft[0])[1] * 1.2 / chain.look(base[0])[1] - 1) < 0.03 and chain.soften(base, None) is base
          and chain.soften(base, (0.9, 0.9)) is base)
    burn = lambda x, k: (((x - x.mean()) * k + x.mean())).clamp(0, 1)
    frames = torch.cat([burn(base, 1 + 0.4 * t / 23) for t in range(24)])
    want = chain.look(base[0])
    chain.hold(frames, want)
    check("a shot that burns as it runs ends back on the scene's look and starts where it was",
          abs(chain.look(frames[-1])[0] / want[0] - 1) <= 0.035 and abs(chain.look(frames[0])[0] / want[0] - 1) <= 0.035,
          (chain.look(frames[-1]), want))
    smooth = torch.nn.functional.interpolate(torch.rand(1, 3, 12, 16, generator=gen), size=(96, 128), mode="bicubic",
                                             align_corners=False).permute(0, 2, 3, 1).clamp(0, 1) * 0.6 + 0.2
    grain = (torch.rand(1, 96, 128, 1, generator=gen) - 0.5) * 0.04
    rough = torch.cat([(smooth + grain * (1 + 0.5 * t / 23)).clamp(0, 1) for t in range(24)])
    was, aim = rough.clone(), chain.crisp(rough[0])
    low = lambda x: chain._blur(x.permute(0, 3, 1, 2).reshape(-1, 1, 96, 128), 4.0)
    _, eased = chain.hold(rough, None, aim)
    over = lambda x: max(a / b for a, b in zip(chain.crisp(x), aim))
    check("a shot whose fine detail sharpens as it runs is eased back to the scene's sharpness, with its tones left alone",
          eased < 1 and over(rough[-1]) < 1.12 and over(was[-1]) > 1.3 and float((rough[0] - was[0]).abs().max()) < 2e-3
          and float((low(rough[-1:]) - low(was[-1:])).abs().max()) < 0.01, (eased, over(rough[-1]), over(was[-1])))
    back = torch.nn.functional.interpolate(base.permute(0, 3, 1, 2), size=(384, 672), mode="bilinear").permute(0, 2, 3, 1)
    opening = back.clone()
    opening[:, 20:100, 20:120] += 0.03 * torch.sign(torch.randn(1, 80, 100, 1, generator=gen))
    end = opening.clone()
    end[:, 150:300, 300:450] += 0.2 * torch.sign(torch.sin(torch.arange(150, dtype=torch.float32) * 0.785)).view(1, 1, -1, 1)
    end[:, 320:370, 500:600] = torch.tensor([0.55, 0.55, 0.58])
    opening, end = opening.clamp(0, 1), end.clamp(0, 1)
    fresh = chain.refresh(end, back, torch.cat([opening, end]))
    gap = lambda x, y, r, c: float((x[:, r, c] - y[:, r, c]).abs().mean())
    check("the hand-off gets its still background back, while what moved or changed is kept",
          gap(fresh, back, slice(30, 90), slice(30, 110)) < 0.003 and gap(fresh, end, slice(150, 300), slice(300, 450)) < 0.003
          and gap(fresh, end, slice(320, 370), slice(500, 600)) < 0.003 and chain.refresh(end, None, end) is end)
    gone, there = torch.full((1, 384, 672, 3), 0.015), torch.full((1, 384, 672, 3), 0.015)
    gone[:, 80:, 100:160], there[:, 80:, 400:460] = torch.tensor([0.02, 0.035, 0.06]), torch.tensor([0.02, 0.035, 0.06])
    kept = chain.refresh(there, gone, torch.cat([gone, there]))
    check("in a dim room, legs that moved away are never pasted back into the hand-off from the frame handed on before",
          gap(kept, there, slice(80, 384), slice(90, 170)) < 0.003 and gap(kept, there, slice(80, 384), slice(390, 470)) < 0.003,
          (gap(kept, there, slice(80, 384), slice(90, 170)), gap(kept, there, slice(80, 384), slice(390, 470))))
    wall = (back * 0.2 + 0.5).clamp(0, 1)
    bare, slack, pushed = wall.clone(), wall.clone(), wall.clone()
    slack[:, 200:202, 100:600] += 0.12
    pushed[:, 200:202, 100:300], pushed[:, 160:162, 300:600] = 0.12 + pushed[:, 200:202, 100:300], 0.12 + pushed[:, 160:162, 300:600]
    pushed[:, 160:202, 299:301] += 0.12
    grainy = (slack + 0.03 * torch.sign(torch.randn(slack.shape, generator=gen))).clamp(0, 1)
    fastened, moved, still = (chain.refresh(e, b, torch.cat([b, e, e])) for b, e in ((bare, slack), (slack, pushed), (grainy, slack)))
    found = (gap(fastened, slack, slice(198, 204), slice(100, 600)), gap(moved, pushed, slice(150, 210), slice(100, 600)),
             gap(still, grainy, slice(20, 360), slice(20, 650)))
    check("a thin chain fastened or pushed aside in a shot is kept whole in the hand-off and its old place is never pasted back over "
          "it, while grain on a still chain still gets the frame handed on before", max(found) < 0.003, found)
    box = faces.head_box([[50, 40, 1], [50, 60, 1]] + [[0, 0, 0]] * 12 + [[46, 38, 1], [54, 38, 1]], 100, 200)
    check("the face crop runs from above the hair to the chin and is squared on the head",
          box is not None and box[1] < 30 and 50 < box[3] < 60 and abs(box[2] - box[0] - box[3] + box[1]) <= 1, box)


def render(prompt, dwpose=False, counts=(), textured=False, blind=False, fine=(None, torch.tensor([0.9, 0.4, 0.0])), **kw):
    calls = {"build": [], "sample": [], "gate": []}
    saved = (chain.build, chain.sample, chain.decode, chain.unvoice, chain.make_room, chain.cleanup, modelprep.prepare,
             modelprep.check_vaes, modelprep.takes_pictures, modelprep.partial, faces.available, faces.count, faces.crop, faces.Detector)
    real = chain.build

    def build(clip, vae, avae, text, w, h, frames, handoff=None, refs=(), memory=None, silent=False, cues=(), memo=None):
        calls["build"].append({"text": text, "handoff": handoff, "refs": list(refs), "memory": memory, "silent": silent, "w": w, "h": h})
        cond, lat = real(clip, vae, avae, text, w, h, frames, handoff, refs, memory, silent, cues, memo)
        return cond, dict(lat, shot=len(calls["build"]), frames=frames)

    def sample(model, cond, neg, lat, seed, steps, sampler, scheduler, sigmas):
        calls["sample"].append({"shot": lat["shot"], "text": cond[0][1]["tokens"]["text"],
                                "audio_only": float(lat.get("noise_mask").unbind()[0].max()) == 0
                                if lat.get("noise_mask") is not None else False,
                                "heard": float(lat["noise_mask"].unbind()[1].mean()) if lat.get("noise_mask") is not None else 1.0,
                                "start": float(sigmas[0]) if sigmas is not None else 1.0,
                                "steps": len(sigmas) - 1 if sigmas is not None else steps})
        return dict(lat)

    def decode(vae, avae, out, keep=()):
        k, n = out["shot"], out["frames"]
        imgs = torch.full((n, 64, 96, 3), k / 10.0) + (torch.arange(n).float() / 1e4).view(n, 1, 1, 1)
        if textured:
            imgs = imgs * 0.5 + torch.linspace(0, 0.5, 96).view(1, 1, 96, 1) * torch.tensor([1.0, 0.6, 0.3])
        return imgs, torch.full((1, 2, round(n * 1000 / 24)), float(k)), 1000

    seen = iter(counts)
    chain.build, chain.sample, chain.decode = build, sample, decode
    chain.unvoice = lambda wave, sr, keep: calls["gate"].append((round(float(wave.mean())), keep)) or wave
    chain.make_room = chain.cleanup = lambda *a, **k: None
    modelprep.prepare = lambda m, st, sn, sc, sg, sv, sa, g, nid: (m, st, sn, None, fine, ["prepared"])
    modelprep.check_vaes = lambda v, a: None
    modelprep.takes_pictures = lambda m, g, n: (False, "x.safetensors was not trained on reference pictures") if blind else (True, "")
    modelprep.partial = lambda m, g, n: []
    faces.available = lambda: dwpose
    faces.count = lambda det, frame: next(seen, 2)
    faces.crop = lambda det, pic: pic[:, :8, :8]
    faces.Detector = lambda: SimpleNamespace(close=lambda: None)
    try:
        out = node.H3LongVideos().run("base", FakeClip(), FakeVAE(), FakeAudioVAE(), prompt, "16:9", kw.pop("megapixels", 0.01), 4.0, 8,
                                      "euler", "simple", 7, character_memory=kw.pop("memory", MEM), **kw)
    finally:
        (chain.build, chain.sample, chain.decode, chain.unvoice, chain.make_room, chain.cleanup, modelprep.prepare,
         modelprep.check_vaes, modelprep.takes_pictures, modelprep.partial, faces.available, faces.count, faces.crop,
         faces.Detector) = saved
    return out, calls


def anon_mb():
    for line in open("/proc/self/status") if os.path.exists("/proc/self/status") else ():
        if line.startswith("RssAnon"):
            return int(line.split()[1]) / 1024
    return None


def test_overlay():
    print("\n=== the overlay streams the video to a temp file and never copies it into RAM ===")
    gen = torch.Generator().manual_seed(3)
    src = chain.spool(40, (72, 128, 3), torch.float16)
    for s in range(0, 40, 8):
        src[s:s + 8].copy_(torch.rand(8, 72, 128, 3, generator=gen))
    kept = src.clone()
    out, info = overlay.H3Overlay().run(src, 24, watermark_text="mark", watermark_size=12.0, intro_text="Title", intro_seconds=1.0,
                                        intro_fade=0.5, intro_size=20.0, overlay_stroke=1)
    plain = kept.clone()
    overlay.apply_overlays(plain, 24, "mark", wm_size_pct=12.0, intro="Title", intro_seconds=1.0, intro_fade=0.5, intro_size_pct=20.0,
                           font_name="arial.ttf", stroke_px=1)
    check("the result is what compositing onto a plain copy of the frames gives, and the frames that came in are left as they were",
          torch.equal(out, plain) and not torch.equal(out, kept) and torch.equal(src, kept) and "watermark composited" in info
          and "intro title composited" in info, info)
    check("it comes back in a file-backed buffer of its own, not in the input's storage",
          out.untyped_storage().is_shared() and out.untyped_storage().data_ptr() != src.untyped_storage().data_ptr()
          and out.dtype == src.dtype and out.shape == src.shape)
    same, note = overlay.H3Overlay().run(src, 24)
    check("with no text the frames pass through unchanged and uncopied",
          same.untyped_storage().data_ptr() == src.untyped_storage().data_ptr() and "unchanged" in note, note)
    big = chain.spool(80, (720, 1280, 3), torch.float16)
    big.fill_(0.25)
    before = anon_mb()
    done, _ = overlay.H3Overlay().run(big, 24, watermark_text="mark", intro_text="Title")
    grew = None if before is None else anon_mb() - before
    mb = big.numel() * big.element_size() / 2 ** 20
    check("a long video is not copied into RAM: the process takes on a small fraction of the video's size",
          grew is None or grew < mb / 4, f"grew {grew} MB for a {mb:.0f} MB video")
    del big, done
    real_fallocate, real_free = os.posix_fallocate, chain.mm.get_free_memory

    def full(*a):
        raise OSError(28, "No space left on device")

    os.posix_fallocate = full
    try:
        chain.mm.get_free_memory = lambda *a, **k: 1000
        try:
            chain.spool(4, (64, 64, 3), torch.float16)
            refused = ""
        except MemoryError as e:
            refused = str(e)
        chain.mm.get_free_memory = lambda *a, **k: 1 << 40
        held = chain.spool(4, (64, 64, 3), torch.float16)
    finally:
        os.posix_fallocate, chain.mm.get_free_memory = real_fallocate, real_free
    check("with no room on disk the frames go to RAM only if they fit there, and otherwise it says so instead of being killed later",
          "No space left on device" in refused and "of RAM is free" in refused
          and tuple(held.shape) == (4, 64, 64, 3) and not held.untyped_storage().is_shared(), refused)


def test_render():
    print("\n=== the shot chain ===")
    prompt = "A room.\n\nDana waits.\n\nSarah walks in.\n\n\"Sit,\" Dana says.\n\ncut\n\nSarah drives."
    out, calls = render(prompt, dwpose=True, ref_image_1=IMG)
    video, audio, info = out[0], out[1], out[2]
    talk = [c["heard"] for c in calls["sample"] if c["shot"] == 3 and c["audio_only"]]
    check("one picture pass per beat, plus an audio-only pass for each wordless beat, and for a beat with a line one that gives "
          "the rest of it the room's own sound, one that says the line again over that sound, and one that gives the stretch after "
          "the line its sound again, now that the line has the room around it",
          [c["shot"] for c in calls["sample"] if not c["audio_only"]] == [1, 2, 3, 4]
          and [c["shot"] for c in calls["sample"] if c["audio_only"]] == [1, 2, 3, 3, 3, 4] and abs(sum(talk[:2]) - 1) < 0.02
          and 0 < talk[2] < talk[0], calls["sample"])
    check("every sound pass is given the shot's own sound, never the picture's text with its character sheets and notes, and only "
          "the line's own pass is given the picture's text",
          all(c["text"].startswith("integrated_multimodal_description: [Shot 1] A room.") and c["text"].endswith("non_diegetic_music: N/A")
              and " hair" not in c["text"] for c in calls["sample"] if c["audio_only"] and "<d>" not in c["text"])
          and [c["shot"] for c in calls["sample"] if c["audio_only"] and "<d>" in c["text"]] == [3]
          and all(" hair" in c["text"] for c in calls["sample"] if not c["audio_only"]), [c["text"][:70] for c in calls["sample"]])
    check("stray voices are taken out of every shot, around the line in a shot with one and from all of a shot with none, but never "
          "out of a shot that calls for a voice",
          [k for k, _ in calls["gate"]] == [1, 2, 3, 4] and [len(keep) for _, keep in calls["gate"]] == [0, 0, 1, 0]
          and not render("A room.\n\nSarah screams.")[1]["gate"], calls["gate"])
    check("each continued shot opens on the last frame of the one before, and a cut starts fresh",
          calls["build"][0]["handoff"] is None and calls["build"][3]["handoff"] is None
          and abs(float(calls["build"][1]["handoff"].mean()) - (0.1 + (P.align(96) - 1) / 1e4)) < 1e-3,
          [c["handoff"] is None for c in calls["build"]])
    check("the repeated opening frame of each continued shot is trimmed and the sound matches the picture",
          int(video.shape[0]) == 4 * P.align(96) - 2 and abs(int(audio["waveform"].shape[-1]) - int(video.shape[0]) * 1000 / 24) <= 4,
          (int(video.shape[0]), int(audio["waveform"].shape[-1])))
    info = render(prompt, textured=True)[0][2]
    check("each continued shot reports how it opened against the frame before and how it ends against shot 1",
          "opened at contrast" in info and "end against shot 1: contrast" in info and "sharpness" in info, info)
    check("the face crop goes in for Sarah, and info has a line per shot with how long each part of it took",
          tuple(calls["build"][1]["refs"][0].shape[1:3]) == (8, 8) and sum(1 for x in info.split(" | ") if x.startswith("shot ")) == 4
          and all(re.search(r"\d+s \(encode \d+s, sampling \d+s, decode \d+s, finish \d+s\)$", x)
                  for x in info.split(" | ") if x.startswith("shot ")), info)
    away = render("A room.\n\nDana and Sarah sit.\n\nDana looks away.\n\nSarah looks at Dana.", dwpose=True, counts=(2, 1, 2))[1]["build"]
    out, calls = render("A room.\n\nDana and Sarah sit.\n\nDana walks out.\nexit: Dana\n\nSarah waits alone.\n\nDana walks back in "
                        "and sits beside Sarah.", dwpose=True, counts=(2, 2, 1, 1))
    check("someone coming back into the scene gets the last frame with everyone's face in view, numbered after the opening frame and "
          "naming only them, while a face that only turns away gets nothing, so the scene never goes back to an earlier frame",
          [c["memory"] is not None for c in calls["build"]] == [False, False, False, True]
          and "<Picture 2> is this same place a little earlier, with Dana in view: Dana looks exactly as there on coming back into view."
          in calls["build"][3]["text"] and "an earlier frame with Dana in view went in" in out[2]
          and all(c["memory"] is None for c in away), ([c["memory"] is not None for c in calls["build"] + away], out[2]))
    new = render("A room.\n\nSarah waits alone.\n\nDana walks in and stops in front of Sarah.", dwpose=True, counts=(1, 1))[1]["build"]
    check("someone coming into the scene for the first time gets no earlier frame, since none of them has them in view",
          [c["memory"] is None for c in new] == [True, True] and "a little earlier" not in new[1]["text"], [c["memory"] for c in new])
    out, calls = render("A room.\n\nDana and Sarah sit.\n\nDana looks away.\n\nSarah looks at Dana.", dwpose=True, counts=(2, 1, 2),
                        blind=True, ref_image_1=IMG)
    check("on a model not trained on pictures every shot still gets the picture, and no earlier frame goes in",
          [len(c["refs"]) for c in calls["build"]] == [1, 1, 1] and all(c["memory"] is None for c in calls["build"])
          and "no earlier frame of the scene is fed back in" in out[2], ([len(c["refs"]) for c in calls["build"]], out[2]))
    native, node.NATIVE["16:9"] = node.NATIVE["16:9"], (96, 64)
    try:
        out, calls = render("A room.\n\nDana waits.", megapixels=2.0)
    finally:
        node.NATIVE["16:9"] = native
    sizes = {(r, m): node.frame_size(r, m) for r, m in (("1:1", 1.0), ("4:3", 1.0), ("16:9", 1.5), ("16:9", 0.5))}
    check("the model always renders inside H3's own canvas, short edge 768 and no more than 768x1344, and a bigger size asked for "
          "is reached by scaling the finished frames up, so the scene is never repeated or mirrored across a frame too big for it",
          sizes == {("1:1", 1.0): ((768, 768), (768, 768)), ("4:3", 1.0): ((1024, 768), (1024, 768)),
                    ("16:9", 1.5): ((1344, 768), (1632, 928)), ("16:9", 0.5): ((960, 544), (960, 544))}
          and all((c["w"], c["h"]) == (96, 64) for c in calls["build"]) and tuple(out[0].shape[1:3]) == (96, 128)
          and "96x64, scaled up to 128x96 after rendering" in out[2], (sizes, tuple(out[0].shape), out[2][:80]))
    out, calls = render("A room.\n\nDana tapes Sarah's mouth shut.\n\n\"Help!\" Sarah screams into the tape.")
    muffled = [c for c in calls["sample"] if c["shot"] == 2]
    check("a taped scream is rendered silent, given sound in its own short window, and the rest of the shot gets the room's own sound",
          len(muffled) == 3 and muffled[0]["heard"] == 0 and all(c["audio_only"] for c in muffled[1:]) and 0 < muffled[2]["heard"] < 0.5
          and abs(muffled[1]["heard"] + muffled[2]["heard"] - 1) < 0.02, muffled)
    check("the scream's own window and the rest of its shot share one sound, with her muffled sounds and none of her words",
          muffled[1]["text"] == muffled[2]["text"] and "Sarah lets out muffled, closed-mouth mmmph sounds through the" in muffled[1]["text"]
          and "Help" not in muffled[1]["text"], muffled[1]["text"])
    voice = torch.tensor([0.95, 0.7, 0.3, 0.0])
    out, calls = render("A room.\n\nDana waits.\n\n\"Sit,\" Dana says.", fine=(torch.linspace(1, 0, 9), voice))
    talk = [c for c in calls["sample"] if c["shot"] == 2]
    check("on a few-step schedule the sound passes run on the finer schedule from the start, and once the rest of the shot has its "
          "sound each spoken line is said again over its own window from part-way down it, with the shot's own text",
          [c["steps"] for c in calls["sample"] if c["shot"] == 1] == [8, 8] and [c["audio_only"] for c in talk] == [False, True, True, True]
          and talk[3]["start"] == 1.0 and talk[3]["text"] == talk[1]["text"] and 0 < talk[3]["heard"] < talk[1]["heard"]
          and abs(talk[2]["start"] - 0.95) < 1e-6 and talk[2]["steps"] == 3 and talk[2]["text"] == talk[0]["text"]
          and "<d>Sit</d>" in talk[2]["text"]
          and 0 < talk[2]["heard"] < 0.5 and talk[1]["start"] == 1.0 and talk[1]["steps"] == 8
          and abs(talk[1]["heard"] + talk[2]["heard"] - 1) < 0.02 and talk[1]["text"].startswith("integrated_multimodal_description"),
          talk)
    out, calls = render("A room.\n\nDana waits.", foley_resolution="off", plan_only=False)
    check("foley off renders each beat once", len(calls["sample"]) == 1)
    sent = []
    sys.modules["server"] = SimpleNamespace(PromptServer=SimpleNamespace(instance=SimpleNamespace(
        send_progress_text=lambda text, nid: sent.append((nid, text)))))
    try:
        render(prompt, unique_id="5")
    finally:
        del sys.modules["server"]
    check("the node shows the plan and the shot it is on, live, on itself",
          len(sent) == 4 and all(nid == "5" for nid, _ in sent) and "RENDERING SHOT 1 OF 4" in sent[0][1] and "shot 4:" in sent[0][1]
          and "RENDERING SHOT 4 OF 4" in sent[3][1] and "s), shot 3:" in sent[3][1].split("RENDERING")[0].replace("\n", ", "),
          [t for _, t in sent[-1:]])
    out, calls = render("A room.\n\nDana waits.", plan_only=True)
    check("plan_only samples nothing and returns the script", not calls["sample"] and "[shot 1]" in out[3])


def main():
    test_bonds()
    test_fetch()
    test_scene()
    test_tags()
    test_reading()
    test_crotch()
    test_tether()
    test_car()
    test_rules()
    test_clothes()
    test_diaper()
    test_belt()
    test_speech()
    test_poses()
    test_pictures()
    test_lengths()
    test_models()
    test_flash()
    test_chain()
    test_overlay()
    test_render()
    print("\nRESULT: " + (f"{len(_fails)} FAILURE(S): " + "; ".join(_fails) if _fails else "ALL PASSED"))


if __name__ == "__main__":
    main()
