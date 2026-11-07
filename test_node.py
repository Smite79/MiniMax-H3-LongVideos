import io
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
import planner as P

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


def test_tags():
    print("\n=== restraints and clothes come only from tag lines ===")
    shots, warn = P.plan("An office.\n\nDana sits at the desk.\n\nSarah walks in.\n\nDana tapes her mouth shut.\n"
                         "hold: Sarah, duct tape over her mouth\n\nDana smiles.\n\nDana peels the tape off.\nrelease: Sarah\n\n"
                         "Sarah gasps.", MEM, pictures=[IMG])
    check("the tag puts it on the person it names, and nobody else, and says it goes on in that shot",
          "By the last frame, Sarah has duct tape over her mouth in plain view" in notes(shots[2])
          and "Dana" not in shots[2]["line"].split("goes on:")[1] and "Black-haired Dana's mouth is empty and uncovered." in notes(shots[2]),
          shots[2]["line"])
    check("it stays on in the shots after, the free one is said to be free, and the taped one cannot speak",
          "Sarah: duct tape over her mouth. All of it stays on, exactly as described, for the whole shot." in notes(shots[3])
          and "Sarah cannot open her mouth or say a word the whole shot." in notes(shots[3])
          and "Black-haired Dana's mouth is empty and uncovered." in notes(shots[3]), notes(shots[3]))
    check("release: with no item takes everything off, and the shot after has none of it",
          "comes off: Sarah: duct tape over her mouth." in shots[4]["line"] and "duct tape" not in notes(shots[5]), shots[4]["line"])
    read, warn = P.plan("An office.\n\nDana and Sarah stand.\n\nDana tapes her mouth shut and cuffs her.\n\nDana smiles.", MEM)
    check("without a tag line the beat itself is read, onto the person it happens to, and info quotes the sentence",
          "goes on: Sarah: tape over her mouth; handcuffs behind her back." in read[1]["line"] and "read from the beat: hold: Sarah"
          in read[1]["line"] and "Sarah: tape over her mouth; handcuffs behind her back." in notes(read[2]), read[1]["line"])
    over, _ = P.plan("An office.\n\nDana and Sarah stand.\n\nDana tapes her mouth shut.\nhold: Sarah, a red gag", MEM)
    check("a tag line in a beat replaces what the beat would be read as", "goes on: Sarah: a red gag." in over[1]["line"]
          and "read from" not in over[1]["line"], over[1]["line"])
    wrong, warn = P.plan("A room.\n\nDana waits.\nhold: Dama, rope around her wrists\n\nDana sits.", MEM)
    check("a tag naming someone without a character line is applied as written and flagged",
          any("names Dama, who has no line in character_memory" in w for w in warn) and "Dama" in wrong[0]["line"], warn)
    early, _ = P.plan("A cell.\n\nSarah waits.\n\nDana walks in.", MEM + "\nhold: Sarah, handcuffs behind her back")
    check("a hold line in character_memory is on from the first shot",
          "Sarah: handcuffs behind her back." in notes(early[0]) and "goes on" not in early[0]["line"], early[0]["line"])
    inline, _ = P.plan("A room.\n\nSarah sits. Dana walks in. hold: Sarah, a red collar\n\nwear: Dana, a red dress\n\nDana waits.", MEM)
    check("tags written after a sentence on the same line, or in a paragraph of their own, belong to that beat",
          "goes on: Sarah: a red collar." in inline[0]["line"] and "puts on: Dana: a red dress." in inline[0]["line"]
          and len(inline) == 2, [s["line"] for s in inline])
    odd, _ = P.plan("A room.\n\nDana and Sarah stand.\n\nDana puts the restraints on Sarah.\nhold: Sarah, leather restraints on her "
                    "wrists", MEM)
    check("a restraint the node has no steps for still goes on, told as fastened",
          "goes on: Sarah: leather restraints on her wrists." in odd[1]["line"] and "has leather restraints on her wrists" in notes(odd[1]),
          (odd[1]["line"], notes(odd[1])))


def test_reading():
    print("\n=== restraints and clothes read from the beats ===")
    mem = ("Sarah <Picture 1>: a young woman with long blonde hair, in a white tank top, pink pajama bottoms and a black thong.\n"
           "Dana: a woman with short black hair, in a black suit and a grey jacket.")
    scene = "A bedroom at night.\n\nDana and Sarah stand by the bed.\n\n"
    said = lambda beat: [s["line"] for s in P.plan(scene + beat, mem, pictures=[IMG])[0]][-1]
    for beat, want in (("Dana grabs a roll of duct tape. She tightly wraps it around Sarah's mouth to keep it taped shut.",
                        "goes on: Sarah: duct tape over her mouth."),
                       ("Sarah watches as Dana tapes her mouth.", "goes on: Sarah: tape over her mouth."),
                       ("Dana cuffs Sarah's hands behind her back.", "goes on: Sarah: handcuffs behind her back."),
                       ("Sarah squirms. She wraps more tape around her hips.", "goes on: Sarah: tape around her hips."),
                       ("Dana pulls her tank top up.", "pulled: Sarah: white tank top up above her breasts."),
                       ("Dana pulls Sarah's pajama bottoms off.", "clothes off: Sarah: pink pajama bottoms."),
                       ("Dana takes off her jacket.", "clothes off: Dana: grey jacket."),
                       ("Sarah sits on the bed, bound and gagged.", "on: Sarah: rope around her wrists behind her back; a gag in her mouth.")):
        check(f"'{beat}' reads as {want}", want in said(beat), said(beat))
    check("a struggle, a touch or stripping a bed changes nothing",
          all("read from" not in said(b) for b in ("Sarah tries to pull free of the cuffs.", "Dana strips the bed.",
                                                   "Dana runs a finger over the tape on Sarah's mouth.")))
    kick = P.plan(scene + "Sarah kicks her legs as Dana pulls her pajama bottoms off.", mem)[0][-1]
    check("in 'Sarah kicks her legs as Dana pulls her pajama bottoms off' Dana takes them off, not Sarah",
          "clothes off: Sarah: pink pajama bottoms." in kick["line"]
          and "Black-haired Dana opens blonde-haired Sarah's pink pajama bottoms at the waist" in notes(kick), notes(kick))
    shy = P.plan(scene + "Dana cuffs Sarah's hands behind her back.\n\nSarah whimpers and shakes her head. Ignoring her, she drags "
                         "the tank top up to her collarbones.", mem)[0][-1]
    by = P.plan(scene + "Dana stands next to a stranger in a torn jacket.", mem)[0][-1]
    check("in 'Ignoring her, she drags the tank top up' she is the other woman, in the beat and the step alike, and a stranger's torn "
          "jacket is not Dana's",
          "Ignoring blonde-haired Sarah, black-haired Dana drags the tank top up to her collarbones" in shy["prompt"]
          and "Black-haired Dana pulls blonde-haired Sarah's white tank top up to her collarbones" in notes(shy)
          and "read from" not in by["line"], (shy["prompt"].split("\n\n")[-2], notes(shy), by["line"]))
    both = P.plan(scene + "Dana cuffs Sarah and tapes her mouth.\n\nDana peels the tape off Sarah's mouth and uncuffs her.", mem)[0]
    check("two things coming off in one beat both come off", "comes off: Sarah: tape over her mouth; handcuffs behind her back."
          in both[2]["line"] and " on: " not in both[2]["line"], both[2]["line"])
    check("gear picked up, put back or torn off the roll, and a person moved, change nothing",
          all("read from" not in said(b) for b in ("Dana takes the handcuffs off the nightstand.", "Dana rips a strip of tape off the roll.",
                                                   "Dana pushes Sarah off the bed.", "Sarah sits up in her pajama bottoms.",
                                                   "Dana lays Sarah back down on the bed.")))
    taped, warn = P.plan(scene + "Dana cuffs Sarah's hands behind her back.\n\nDana pulls off Sarah's pajama bottoms and her thong.\n\n"
                         "Dana wraps duct tape around Sarah's waist and between her legs.\n\nDana keeps doing this until Sarah's vagina "
                         "is tightly sealed in duct tape.\n\nDana tears off a strip and slaps it over Sarah's mouth.\n\n"
                         "Dana peels the tape off.\n\nDana peels the tape off Sarah's mouth.", mem, shot_seconds=15.0)
    check("tape between her legs is crotch tape that stays on, 'until' is no release, and naked lines keep the tape",
          "duct tape sealed over her crotch, between her legs" in taped[3]["line"] and not any("shot 5" in w for w in warn)
          and "ankles stay bound" not in taped[4]["prompt"] and "naked from the waist down the whole shot, apart from the duct tape "
          "around her waist and duct tape sealed over her crotch, between her legs" in notes(taped[4]), (warn, notes(taped[4])))
    check("tape going on says who holds the roll and how it goes around her, step by step, and the roll stays whole outside her body",
          "Black-haired Dana presses the loose end of the duct tape flat against blonde-haired Sarah's waist and carries the roll all "
          "the way around her body" in notes(taped[3])
          and ", then runs the duct tape from the front of blonde-haired Sarah's waist down between her legs and up to the back"
          in notes(taped[3]) and "The roll stays whole in Dana's hand and outside Sarah's body the whole time" in notes(taped[3])
          and "Black-haired Dana tears a strip off the roll of duct tape and presses it flat over blonde-haired Sarah's mouth. "
              "Black-haired Dana keeps the roll whole in her other hand, outside Sarah's body." in notes(taped[5]),
          (notes(taped[3]), notes(taped[5])))
    check("tape in several places comes off only where the beat says",
          any("shot 7" in w and "could not tell which duct tape comes off" in w for w in warn) and "comes off" not in taped[6]["line"]
          and "comes off: Sarah: tape over her mouth." in taped[7]["line"] and "crotch" in taped[7]["line"], (warn, taped[7]["line"]))
    check("handcuffs and chains are rigid", "handcuffs are rigid steel" in notes(taped[4]) and "never stretch" in notes(
        P.plan(scene + "Dana chains Sarah's ankles to the bed frame.", mem)[0][1]), notes(taped[4]))
    cut, warn = P.plan(scene + "Dana cuffs Sarah's hands behind her back.\n\nDana cuts Sarah's tank top off with scissors.", mem)
    check("a garment cut or torn off comes off past cuffed wrists", "torn off: Sarah, white tank top" in cut[2]["line"]
          and "clothes off: Sarah: white tank top." in cut[2]["line"] and not warn, (cut[2]["line"], warn))
    last = lambda *b: P.plan("A room.\n\nDana and Sarah stand by the bed.\n\n" + "\n\n".join(b), mem)[0][-1]
    check("a garment put away, a touch on taped skin and gear moved off a spot read as they should",
          last("Dana pulls Sarah's pajama bottoms off.", "Dana puts Sarah's pajama bottoms back in the drawer.")["read"] == []
          and last("Dana tapes Sarah's mouth shut.", "Dana rests a hand on Sarah's taped mouth.")["read"] == []
          and last("Dana blindfolds Sarah.", "Dana pulls the blindfold down around Sarah's neck.")["read"]
          == [("release", "Sarah, a blindfold over her eyes")], last("Dana blindfolds Sarah.", "Dana pulls the blindfold down "
                                                                     "around Sarah's neck.")["read"])
    crotch = last("Dana winds duct tape around Sarah's hips, then feeds the roll down between her thighs.",
                  "Dana slices through the tape between Sarah's legs and peels it away.")
    check("tape fed between her legs after ', then' is crotch tape, and cutting it there frees only that spot",
          crotch["read"] == [("release", "Sarah, duct tape sealed over her crotch, between her legs")]
          and "on: Sarah: duct tape around her hips." in crotch["line"], crotch["line"])
    chained = last("Dana cuffs Sarah's hands behind her back.", "Dana chains Sarah's cuffed wrists to the bed frame.", "Sarah waits.")
    check("chaining cuffed wrists to the bed adds the chains, fastened where the cuffs hold them, and rigid",
          "chains around her wrists behind her back, chained to the bed frame" in chained["line"]
          and "with her wrists locked together behind her and chained to the bed frame" in notes(chained)
          and "never stretch" in notes(chained), (chained["line"], notes(chained)))
    outfit = "Lena: a woman, in a denim jacket over a white t-shirt, blue jeans over black panties, and white sneakers.\nDana: a woman."
    top = P.plan("A room.\n\nLena stands.\n\nDana pulls Lena's top off.\n\nDana pulls Lena's pants off.", outfit)[0]
    check("'top' and 'pants' take off her one t-shirt and her jeans, and hidden layers keep the list's commas",
          "clothes off: Lena: white t-shirt." in top[1]["line"] and "clothes off: Lena: blue jeans." in top[2]["line"]
          and "over a white t-shirt, blue jeans, and white sneakers." in top[0]["prompt"], (top[1]["line"], top[2]["line"]))
    mixed = "Sarah: a young woman with long blonde hair, in a white tank top.\nJake: a man with short brown hair, in a grey hoodie."
    he = P.plan("A room.\n\nJake and Sarah stand.\n\nJake binds her wrists with rope. His ankles are free.", mixed)[0]
    check("his or her goes to the only man or woman in view", "goes on: Sarah: rope around her wrists behind her back." in he[1]["line"],
          he[1]["line"])


def test_speech():
    print("\n=== who says each line ===")
    mem = ("Sarah: a young woman with long blonde hair and a soft, breathy voice, in a white tank top.\n"
           "Dana: a woman with short black hair, in a black jacket. Voice: low and husky.")
    said = lambda *b: P.plan("A room.\n\nDana and Sarah stand by the bed.\n\n" + "\n\n".join(b), mem)[0][-1]
    shot = said('Dana leans in and whispers, "Hold still." She smiles. "This won\'t take long."')
    check("a line goes to its speaker, a line that carries on stays with her, and the prompt names her by look",
          shot["said"] == [("Dana", "Hold still.", False), ("Dana", "This won't take long.", False)]
          and "Black-haired Dana leans in and whispers: <d>Hold still.</d>" in shot["prompt"]
          and "Only black-haired Dana speaks in this shot, in her own voice, low and husky" in notes(shot), shot["prompt"])
    check("a tag after the line, a name and a colon, and an action right before the line each give the speaker",
          said('"Go to hell," Sarah snaps.')["said"] == [("Sarah", "Go to hell", False)]
          and said('Dana: "Now we wait."')["said"] == [("Dana", "Now we wait.", False)]
          and said('Sarah glares at her. "Go to hell."')["said"] == [("Sarah", "Go to hell.", False)]
          and "in her own soft, breathy voice" in notes(said('Sarah glares at her. "Go to hell."')))
    ends = said('"Are you still here?" Dana asks. Sarah nods. "Stop," she says.')
    check("a line that ends in its own ? ! or . takes no second full stop from the tag after it",
          "asks: <d>Are you still here?</d> Sarah nods." in ends["prompt"] and "says: <d>Stop</d>." in ends["prompt"], ends["prompt"])
    gagged = said("Dana tapes Sarah's mouth shut.", '"Please," Sarah mumbles into the tape.')
    check("a taped speaker makes a muffled sound with no spoken words, in a short window of its own, and the shot stays wordless",
          gagged["said"] == [("Sarah", "Please", True)] and "<d>" not in gagged["prompt"] and gagged["silent"]
          and gagged["cues"] == [(0.5, round(0.5 + P.SAY_SLACK + 1 / P.WORDS_PER_SEC, 2), "muffle")]
          and "Blonde-haired Sarah lets out a muffled, closed-mouth mmmph through the tape." in gagged["prompt"]
          and "Apart from the muffled sounds, everyone stays silent." in notes(gagged), gagged["prompt"])
    moans = said("Dana tapes Sarah's mouth shut.", 'Sarah lies on the bed and whimpers into the tape. "Mmmph!" Sarah squirms.')
    check("a sound from a taped mouth told in the beat, or quoted with no tag, gets a short muffled window too",
          moans["cues"] == [(0.5, 0.5 + P.MUFFLE_SOUND, "muffle"), (0.5 + P.MUFFLE_SOUND + P.CUE_GAP,
                                                                    round(0.5 + P.MUFFLE_SOUND + P.CUE_GAP + P.SAY_SLACK + 1 / P.WORDS_PER_SEC, 2), "muffle")]
          and moans["silent"] and "mmmph through the tape. Sarah squirms." in moans["prompt"]
          and said("Dana tapes Sarah's mouth shut.", "Dana strokes Sarah's hair and hums.")["cues"] == [], (moans["cues"], moans["prompt"]))
    timed = said('Dana walks to the bed. She sits down. "Hold still," she says. Sarah nods. "Okay," Sarah whispers.')
    check("each line gets a short window of its own, placed after the actions before it, with silence around it",
          [k for _, _, k in timed["cues"]] == ["line", "line"] and timed["cues"][0][0] > 0.5 + P.PER_ACTION
          and timed["cues"][1][0] >= timed["cues"][0][1] + P.PER_ACTION - 0.1
          and all(b - a < 1.5 for a, b, _ in timed["cues"]) and timed["cues"][-1][1] + P.SPEECH_TAIL <= timed["frames"] / 24
          and "Apart from the lines, everyone stays silent." in notes(timed), (timed["cues"], timed["frames"]))
    check("a plain 'says' never goes to the one whose mouth is taped",
          said("Dana tapes Sarah's mouth shut.", '"Much better," she says.')["said"] == [("Dana", "Much better", False)])
    both = said('Dana peels the tape off Sarah\'s mouth. "Thank you," Sarah gasps. "Don\'t thank me yet," Dana answers.')
    check("two speakers each keep their own lines", both["said"] == [("Sarah", "Thank you", False), ("Dana", "Don't thank me yet", False)]
          and "each in her own voice, with natural, expressive intonation and real emotion, the way people really talk, and each "
              "saying only her own lines." in notes(both), both["said"])
    check("gag sounds and a gagged plea stay with the gagged one; a line to her, a named listener or an answer goes to the other",
          said("Dana tapes Sarah's mouth shut.", 'Sarah thrashes on the bed. "Mmph! Let me go!"')["said"] == [("Sarah", "Mmph! Let me go!", True)]
          and said("Dana tapes Sarah's mouth shut.", 'Sarah whimpers into the tape. "Shh, nobody can hear you."')["said"]
          == [("Dana", "Shh, nobody can hear you.", False)]
          and said('"I\'m not scared of you," she tells Sarah.')["said"] == [("Dana", "I'm not scared of you", False)]
          and said('"Where is the key, Sarah?" Dana asks. "Somewhere you will never find it."')["said"][1][0] == "Sarah"
          and said('"Look at me, Sarah," she says.')["said"] == [("Dana", "Look at me, Sarah", False)])
    think = said('"I have to get out of here," Sarah thinks.')
    check("a quoted thought is not spoken", think["said"] == [] and think["silent"] and "<d>" not in think["prompt"], think["prompt"])
    long = said('Sarah glares at Dana. "Where were you? I waited for hours. You could have called."')
    check("each sentence of a long line is marked as the speaker's own, with time for the pauses between them, and the listener "
          "keeps her lips closed",
          "Blonde-haired Sarah says: <d>Where were you?</d> Sarah continues: <d>I waited for hours.</d> Sarah continues: <d>You could "
          "have called.</d>" in long["prompt"] and long["said"] == [("Sarah", "Where were you? I waited for hours. You could have called.",
                                                                   False)]
          and round(long["cues"][0][1] - long["cues"][0][0], 2) == round(P.SAY_SLACK + 11 / P.WORDS_PER_SEC + 2 * P.SENTENCE_PAUSE, 2)
          and "with natural, expressive intonation and real emotion" in notes(long)
          and "Black-haired Dana listens with closed lips." in notes(long), (long["prompt"], long["cues"]))
    quiet = said("Dana walks to the window.")
    check("the sound is said as what is heard, never as music that is not there",
          "All the sound comes from the scene itself" in notes(quiet) and "music" not in quiet["prompt"], notes(quiet))
    scream, radio, band = said("Sarah screams."), said("A radio plays music."), said("Dana puts a rubber band around Sarah's wrists.")
    check("voices are kept out of a shot with no lines unless it calls for a voice, music or a tone of its own, and a scream is not "
          "told to keep its mouth closed",
          quiet["mute"] and band["mute"] and "All the sound comes from the scene itself" in notes(band)
          and not scream["mute"] and "Nobody says a word." in notes(scream) and "every mouth stays closed" not in notes(scream)
          and not radio["mute"] and not said("A dog barks outside.")["mute"] and long["mute"] and said('"Get out!" Sarah screams.')["mute"]
          and not P.plan("A room.\n\nDana waits.", mem, silence=False)[0][0]["mute"]
          and P.plan('A room.\n\n"Wait," Dana says.', mem, silence=False)[0][0]["mute"],
          [(s["mute"], notes(s)) for s in (quiet, scream, radio, band)])


def test_hands():
    print("\n=== gags made of clothes, and where bound wrists go ===")
    mem = ("Sarah: a young woman with long blonde hair, in a white tank top and pink pajama bottoms over black panties.\n"
           "Dana: a woman with short black hair, in a black jacket and jeans over a red thong.")
    plan = lambda *b: P.plan("A bedroom.\n\nDana and Sarah stand by the bed.\n\n" + "\n\n".join(b), mem)
    shot = plan("Dana pulls off Sarah's pajama bottoms.", "Dana pulls off Sarah's panties and stuffs them into her mouth.")[0][-1]
    check("underwear stuffed into a mouth is a gag of that garment on the one it goes into, and the beat names her",
          shot["read"] == [("remove", "Sarah, black panties"), ("hold", "Sarah, black panties stuffed in her mouth as a gag")]
          and "stuffs them into blonde-haired Sarah's mouth" in shot["prompt"]
          and "Black-haired Dana slides blonde-haired Sarah's black panties down her legs and off over her feet, then pushes the black "
              "panties into blonde-haired Sarah's mouth with her fingers until her mouth is full" in notes(shot)
          and "Black-haired Dana's own mouth stays empty." in notes(shot)
          and "Black-haired Dana's mouth is empty and uncovered." in notes(shot), shot["prompt"])
    own = plan("Dana takes off her own thong and shoves it into Sarah's mouth.")[0][-1]
    check("the doer's own underwear comes off her and goes into the other one's mouth, and her own 'her' is left alone",
          own["read"] == [("remove", "Dana, red thong"), ("hold", "Sarah, a red thong stuffed in her mouth as a gag")]
          and "Dana takes off her own thong" in own["prompt"], own["read"])
    muffled = plan("Dana stuffs a rag into Sarah's mouth.", '"Mmph!" Sarah whimpers.')[0][-1]
    check("a muffled sound goes through what is stuffed in her mouth",
          "mmmph through the rag" in muffled["prompt"] and "Sarah: a rag stuffed in her mouth as a gag." in notes(muffled),
          muffled["prompt"])
    shown, warn = plan("Dana holds up a red ball gag.", "Dana picks a discarded sock up off the floor.",
                       "Dana fits the ball gag between Sarah's teeth.")
    check("holding up a gag or picking a discarded sock up off the floor is handling a prop, not putting it on anyone",
          not warn and not shown[1]["read"] and not shown[2]["read"] and shown[3]["read"] == [("hold", "Sarah, a ball gag in her mouth")],
          (warn, [s["read"] for s in shown]))
    where = lambda b: plan(b)[0][-1]["read"]
    check("cuffs go behind her back unless the beat puts her wrists in front, and where someone stands does not count",
          where("Dana stands in front of Sarah and cuffs her wrists.") == [("hold", "Sarah, handcuffs behind her back")]
          and where("Kneeling in front of her, Dana cuffs Sarah's hands.") == [("hold", "Sarah, handcuffs behind her back")]
          and where("Dana cuffs Sarah's wrists in front of her.") == [("hold", "Sarah, handcuffs in front of her")],
          where("Kneeling in front of her, Dana cuffs Sarah's hands."))
    shots, warn = plan("Dana cuffs Sarah's hands behind her back.", "Sarah wipes her eyes.")
    check("cuffs behind her back are never asked to be in plain view, the pose is spelled out, and a beat that needs her hands is "
          "flagged",
          "By the last frame, blonde-haired Sarah has handcuffs behind her back, and Dana has let go." in notes(shots[1])
          and "Black-haired Dana brings blonde-haired Sarah's arms behind her back and closes one cuff around her left wrist with a "
              "click, then the other around her right wrist." in notes(shots[1]) and "both arms run at their full length from her shoulders down behind her, and her hands rest together "
                                 "at the small of her back. From the front, her body hides her hands; seen from the side or from behind, "
                                 "both arms and both hands are in plain view" in notes(shots[2])
          and "out of sight" not in notes(shots[2]) and "rigid steel, locked behind her back" in notes(shots[2])
          and any("\"Sarah wipes her eyes.\" needs Sarah's hands" in w for w in warn), (notes(shots[1]), warn))
    down = plan("Dana cuffs Sarah's hands behind her back.", "Dana pushes Sarah down onto her stomach on the floor.",
                "Sarah lies still.")[0]
    pushed = plan("Dana pushes Sarah face down onto the floor and cuffs her wrists behind her back.", "Dana cuffs Sarah's ankles.")[0]
    check("lying face down, her cuffed hands stay on top of the small of her back, in plain view, and being pushed face down onto "
          "the floor lays her down",
          all("lying face down, both her arms lie along her back at their full length, and her hands stay locked together on top of "
              "the small of her back, in plain view from above and from either side." in notes(s) for s in down[2:4] + pushed[1:])
          and "Once the handcuffs are on, she lies with her legs together." in notes(pushed[2])
          and "her legs straight and flat where they lie" in notes(pushed[2]) and "stands with" not in notes(pushed[2]),
          [notes(s) for s in pushed[1:]])
    alone = P.plan("A cell.\n\nSarah sits on the bed, handcuffed.", mem)[0][0]
    check("a closing 'handcuffed.' is on the one the sentence is about", alone["read"] == [("have", "Sarah, handcuffs behind her back")],
          alone["read"])


def test_bodies():
    print("\n=== bound bodies stay whole, and fastened people stay where they are fastened ===")
    mem = ("Sarah: a young woman with long blonde hair, in a white tank top and jeans.\n"
           "Dana: a woman with short black hair, in a black suit.")
    last = lambda *b: P.plan("A basement.\n\n" + "\n\n".join(b), mem)[0][-1]
    down = last("Sarah and Dana are in the basement. Sarah lies on a metal table.", "Dana chains Sarah down to the table.", "Sarah waits.")
    check("chained down on a table, she lies on it with her arms down on it and the chains run down to it",
          "chains around her wrists at her sides, chained down to the table" in down["line"]
          and "Blonde-haired Sarah lies on her back on the table the whole shot, her whole body resting on it: both arms lie flat on it "
              "at her sides, her wrists chained to it there" in notes(down)
          and "The chains run from her wrists straight down to the table, so her arms stay down on the table and never rise into the air."
          in notes(down) and "above her head" not in notes(down) and "ceiling" not in notes(down), notes(down))
    check("'chained down on a table' with no 'to', and straps on a gurney, read the same way",
          last("Sarah is chained down on a table.")["read"] == [("have", "Sarah, chains around her wrists at her sides, chained down to the table")]
          and last("Sarah is strapped to the gurney.")["read"]
          == [("have", "Sarah, straps around her wrists at her sides, strapped down to the gurney")])
    check("sitting at a table, chained to it, her hands stay down on it",
          "Sarah sits at the table the whole shot, her forearms resting on top of it in front of her and her wrists chained to it"
          in notes(last("Sarah sits at the table, her wrists chained to it.")))
    corners = last("Sarah and Dana are in the basement.", "Sarah lies on the table, her wrists and ankles chained to its corners.")
    check("wrists and ankles chained to its corners are hers, spread out on the table",
          corners["read"] == [("have", "Sarah, chains around her wrists spread apart, chained down to the corners of the table; chains around "
                                       "her ankles spread apart, chained down to the corners of the table")]
          and "both arms are spread out wide and rest on it" in notes(corners) and "both legs are spread out wide on it" in notes(corners),
          corners["read"])
    check("'her ankles chained together' while she lies on the floor chains them to each other, not to the floor, and a closing "
          "'chained down.' on the bed is read",
          last("Sarah lies on the floor, her ankles chained together.")["read"] == [("have", "Sarah, chains around her ankles")]
          and last("Sarah lies on the bed, chained down.")["read"]
          == [("have", "Sarah, chains around her wrists at her sides, chained down to the bed")],
          (last("Sarah lies on the floor, her ankles chained together.")["read"], last("Sarah lies on the bed, chained down.")["read"]))
    check("arms go up only when the beat hangs her from above",
          "Sarah's arms stay raised above her head the whole shot, her wrists chained to the ceiling"
          in notes(last("Sarah hangs from the ceiling, her wrists chained above her head.")))
    cuffed = P.plan("A basement.\n\nSarah sits on a chair, her wrists handcuffed behind her back.\n\nThe camera moves down to her feet.\n\n"
                    "Dana ties Sarah's ankles together with rope. We see her legs and feet.", mem)[0]
    check("a bound person's whole body is spelled out from every side, ankles tied while she sits keep her knees together, and a beat "
          "that shows her legs drops the face line",
          all("Whenever the camera shows any part of" in notes(s) and "both legs and feet are all there at their natural size" in notes(s)
              for s in cuffed) and "Sarah's ankles stay tied together from the moment the rope is on: the rope is tied tight around "
                                   "them just above her feet, so her legs stay pressed together and always move together, both at "
                                   "once. Once the rope is on, she sits with her knees and feet together."
          in notes(cuffed[2])
          and "The camera keeps" not in notes(cuffed[2]) and "Whenever the camera shows" not in notes(last("Sarah waits.")),
          [notes(s) for s in cuffed])


def test_pronouns():
    print("\n=== she and he are the named person they stand for ===")
    mix = "Sarah <Picture 1>: a woman with long blonde hair.\nJake <Picture 2>: a man with short brown hair."
    two = "Sarah <Picture 1>: a woman with long blonde hair.\nDana <Picture 2>: a woman with short black hair."
    beat = lambda mem, *b: P.plan("A room.\n\n" + "\n\n".join(b), mem, pictures=[IMG, IMG])[0][-1]["prompt"].split("\n\n")[-2]
    check("she, he, her and his become the names they stand for, and her own hair stays hers",
          beat(mix, "Sarah sits on the couch.", "She picks up the phone.") == "Sarah picks up the phone."
          and beat(mix, "Sarah and Jake sit.", "He takes her hand and she smiles.") == "Jake takes Sarah's hand and Sarah smiles."
          and beat(mix, "Sarah walks in and she sits down. She brushes her hair.")
          == "Sarah walks in and Sarah sits down. Sarah brushes her hair.",
          beat(mix, "Sarah and Jake sit.", "He takes her hand and she smiles."))
    check("with two women, she is the one the sentence before was about, or the one acted on when she reacts",
          beat(two, "Sarah grabs Dana. She pushes her away.") == "Sarah grabs Dana. Sarah pushes Dana away."
          and beat(two, "Dana pushes Sarah onto the bed. She screams.") == "Dana pushes Sarah onto the bed. Sarah screams."
          and beat(two, "Dana walks in. She stares at Sarah.") == "Dana walks in. Dana stares at Sarah.")
    fell = P.plan("A room.\n\nSarah and Dana stand by the bed.\n\nDana pushes Sarah down. She falls, handcuffed.", two,
                  pictures=[IMG, IMG])[0][-1]
    check("the she who falls after being pushed is the one pushed, and the handcuffs the sentence gives her go on her, while a she "
          "who screams after struggling stays the one struggling",
          fell["prompt"].split("\n\n")[-2] == "Dana pushes Sarah down. Sarah falls, handcuffed."
          and fell["read"] == [("have", "Sarah, handcuffs behind her back")]
          and beat(two, "Sarah and Dana stand by the bed.", "Sarah struggles. She screams.") == "Sarah struggles. Sarah screams.",
          (fell["prompt"].split("\n\n")[-2], fell["read"]))
    check("'drags her down the bed' names her, not 'her bed', and in 'Ignoring her, she ...' she is the other woman",
          beat(two, "Sarah lies on the bed. Dana grabs her ankles and drags her down the bed.")
          == "Sarah lies on the bed. Dana grabs her ankles and drags Sarah down the bed."
          and beat(two, "Sarah and Dana sit on the bed. Sarah sobs. Ignoring her, she ties the rope.")
          == "Sarah and Dana sit on the bed. Sarah sobs. Ignoring Sarah, Dana ties the rope.")
    check("a she that could be either is left alone, and so is a he that is an unnamed man",
          beat(two, "Sarah and Dana sit on the couch. She laughs.") == "Sarah and Dana sit on the couch. She laughs."
          and beat(mix, "Jake sits.", "A man walks in. He sits beside Jake.") == "A man walks in. He sits beside Jake.")


def test_extras():
    print("\n=== unnamed people stay separate ===")
    mem = ("Sarah: a woman with long blonde hair, in a white tank top.\n"
           "Dana: a woman with short black hair, in a black jacket.")
    shots, warn = P.plan("A bar.\n\nSarah and Dana sit.\n\nA man walks in. \"Come with me,\" the man says.\n\nSarah looks at Dana."
                         "\n\nThe man leaves.\n\nSarah sighs.\n\ncut\n\nA guard opens the door.", mem)
    check("a line tagged with an unnamed man is his, and the prompt and notes say so",
          shots[1]["said"] == [("the man", "Come with me", False)] and "The man says: <d>Come with me</d>" in shots[1]["prompt"]
          and "Only the man speaks in this shot, in his own voice" in notes(shots[1]) and not warn, (shots[1]["prompt"], warn))
    check("an unnamed person is counted and kept apart in the beats that name him, and left out of the ones that do not",
          "There are three people in the shot, with one body for each person: blonde-haired Sarah, black-haired Dana and the man."
          in notes(shots[1]) and "The man is a different person from blonde-haired Sarah and black-haired Dana: he moves only as "
                                 "this shot describes" in notes(shots[1])
          and "never copies blonde-haired Sarah's or black-haired Dana's movements, actions or words" in notes(shots[1])
          and "the man" not in notes(shots[2]) and "people in the shot" not in notes(shots[2]) and "the man" in notes(shots[3])
          and "the man" not in notes(shots[4]) and "There is one person in the shot" in notes(shots[5]), [notes(s) for s in shots])
    call, _ = P.plan("An office.\n\nSarah sits at her desk.\n\nSarah talks to a man on her computer screen.\n\nThe man laughs.\n\n"
                     "Sarah closes the laptop.\n\nSarah stretches.", mem)
    check("the man on her computer screen is on the screen, a different person from Sarah, and she stays in the room",
          "There is one person in the shot: one body, one face." in notes(call[1])
          and "The man on the computer screen is only seen there, a different person from Sarah with a face of his own: he is not in "
              "the room" in notes(call[1]) and "Sarah is in the room, never on the computer screen." in notes(call[1])
          and "The man on the computer screen is only seen there" in notes(call[2]) and "the man" not in notes(call[4])
          and all("The camera keeps Sarah's face in frame" in notes(s) or "blonde-haired" in notes(s) for s in call),
          [notes(s) for s in call])
    video = P.plan("An office.\n\nSarah and Dana sit at the desk.\n\nSarah is on a video call with a man.", mem)[0][-1]
    check("on a video call with a man, the named people stay in the room and only the man is on the call",
          "There are two people in the shot, with one body for each person." in notes(video)
          and "The man on the video call is only seen there, a different person from blonde-haired Sarah and black-haired Dana" in notes(video)
          and "blonde-haired Sarah and black-haired Dana are in the room, never on the video call." in notes(video)
          and "Sarah is only" not in notes(video), notes(video))
    phone = P.plan("A kitchen.\n\nSarah paces.\n\nSarah talks to a man on the phone. \"I'll be there soon,\" the man says.", mem)[0][-1]
    voice = P.plan("A room.\n\nSarah and Dana sit.\n\nA man's voice calls from the hallway. \"Dinner is ready!\"", mem)[0][-1]
    car = P.plan("A car.\n\nSarah and Dana sit.\n\nSarah sits in the driver's seat by a man-made lake.", mem)[0][-1]
    check("a voice on the phone or from off screen is never seen and moves nobody's lips, and a driver's seat brings in nobody",
          phone["said"] == [("the man on the phone", "I'll be there soon", False)] and "only a voice" in notes(phone)
          and "nobody in the shot moves their lips" in notes(phone) and voice["said"][0][0] == "the man off screen"
          and "There are two people in the shot" in notes(voice) and "unnamed" not in car["line"], (notes(phone), voice["said"]))
    mixed = ("Sarah: a woman with long blonde hair, in a white tank top.\nJake: a man with short brown hair, in a grey hoodie.")
    asked = P.plan("A kitchen.\n\nSarah and Jake sit.\n\nA man walks in. \"Who are you?\" he asks. Jake stands up.", mixed)[0][-1]
    check("'he' after an unnamed man walks in is the man, not the named man", asked["said"] == [("the man", "Who are you?", False)],
          asked["said"])
    check("'the woman' with two named women in view is one of them, not someone new",
          "unnamed" not in P.plan("A room.\n\nSarah and Dana sit.\n\nThe woman smiles.", mem)[0][-1]["line"])
    cuffed, warn = P.plan("An alley.\n\nSarah and Dana stand.\n\nA man cuffs Sarah's hands behind her back.\n\n"
                          "Dana tapes the man's mouth shut.", mem)
    check("an unnamed man doing it is the one who does it, and what is done to him lands on nobody named",
          cuffed[1]["read"] == [("hold", "Sarah, handcuffs behind her back")]
          and "The man brings blonde-haired Sarah's arms behind her back and closes one cuff around her left wrist" in notes(cuffed[1])
          and "and the man has let go" in notes(cuffed[1]) and cuffed[2]["read"] == [] and not warn, (notes(cuffed[1]), warn))
    men = "Jake: a man with short brown hair, in a grey hoodie.\nTom: a man with a grey beard, in a flannel shirt."
    her = P.plan("A park.\n\nSarah and Dana sit.\n\nA woman grabs Sarah. She cuffs her hands behind her back while Dana watches.",
                 mem)[0][-1]
    him = P.plan("A pier.\n\nJake and Tom sit.\n\nA man grabs Jake. He cuffs his hands behind his back while Tom watches.", men)[0][-1]
    check("an unnamed woman or man is 'she' or 'he' right after acting, also beside named women or men, and it lands on the right one",
          her["read"] == [("hold", "Sarah, handcuffs behind her back")] and "The woman cuffs blonde-haired Sarah's hands" in her["prompt"]
          and him["read"] == [("hold", "Jake, handcuffs behind his back")], (her["read"], him["read"], her["prompt"]))
    girl = P.plan("A park.\n\nSarah and Dana sit.\n\nA girl walks in. \"Hi,\" she says. Sarah smiles.", mem)[0][-1]
    boy = P.plan("A pier.\n\nJake and Tom sit.\n\nA boy runs up to Jake. \"Mister, is this yours?\" he asks.", men)[0][-1]
    check("a girl's or a boy's line is theirs beside named women or men, and the note uses she or he",
          girl["said"] == [("the girl", "Hi", False)] and "The girl is a different person" in notes(girl) and "she moves" in notes(girl)
          and boy["said"] == [("the boy", "Mister, is this yours?", False)] and "he moves" in notes(boy), (girl["said"], boy["said"]))


def test_names():
    print("\n=== any name ===")
    for a, b in (("Zoë", "Chloé"), ("McKenzie", "Brooke"), ("Anne-Marie", "Jo"), ("O'Brien", "Siobhan"), ("Łucja", "Żaneta"),
                 ("Mary Jane", "Will")):
        mem = f"{a}: a woman with short dark hair.\n{b}: a woman with long red hair."
        shots, warn = P.plan(f"A room.\n\n{a} and {b} sit.\n\n{a} tapes her mouth.\nhold: {b}, duct tape over her mouth", mem)
        check(f"{a} and {b} are each one person and the tape goes only on {b}",
              shots[1]["cast"] == [a, b] and f"goes on: {b}: duct tape over her mouth." in shots[1]["line"] and not warn,
              (shots[1]["line"], warn))


def test_cast():
    print("\n=== who is in view ===")
    shots, _ = P.plan("A room.\n\nDana waits.\n\nSarah walks in.\n\nDana walks out.\nexit: Dana\n\nSarah sits.\n\n"
                      "Dana comes back in.\n\ncut\n\nSarah drives.", MEM, pictures=[IMG])
    check("people come in when a beat names them and stay until an exit: line, and a cut starts with the beat's own",
          [s["cast"] for s in shots] == [["Dana"], ["Dana", "Sarah"], ["Dana", "Sarah"], ["Sarah"], ["Sarah", "Dana"], ["Sarah"]],
          [s["cast"] for s in shots])
    check("the camera keeps everyone's face in frame except whoever leaves, and the shot says they leave",
          "The camera keeps Sarah's face in frame for the whole shot. Black-haired Dana leaves during this shot." in notes(shots[2])
          and "The camera keeps the faces of black-haired Dana and Sarah in frame" in notes(shots[1]), notes(shots[2]))
    check("someone coming in is named against who the opening frame holds",
          "with only Sarah in view; black-haired Dana comes into view during this shot." in notes(shots[4]), notes(shots[4]))
    check("names inside a spoken line do not bring anyone in",
          P.plan('A room.\n\nDana waits.\n\n"Where is Sarah?" Dana asks.', MEM)[0][1]["cast"] == ["Dana"])
    close, _ = P.plan("A room.\n\nDana and Sarah sit.\n\nThe camera moves in close on Dana.", MEM)
    check("a beat that directs the camera gets no face line", "The camera keeps" not in notes(close[1]), notes(close[1]))
    crowd, _ = P.plan("A bar.\n\nDana waits.\n\nA man walks past Dana.\n\nA crowd gathers around Dana.", MEM)
    check("a head count counts unnamed people too, and is left out when their number is not known",
          "There is one person in the shot" in notes(crowd[0])
          and "There are two people in the shot, with one body for each person: Dana and the man." in notes(crowd[1])
          and "people in the shot" not in notes(crowd[2]), [notes(s) for s in crowd])


def test_clothes():
    print("\n=== clothes ===")
    shots, _ = P.plan("A room.\n\nDana and Sarah stand.\n\nDana pulls down Sarah's leggings.\nremove: Sarah, leggings\n\n"
                      "Sarah shivers.\n\nSarah pulls on a red skirt.\nwear: Sarah, a red skirt\n\nSarah sits.", MEM, pictures=[IMG])
    check("a garment taken off is off by the last frame of its shot and gone from her description after it",
          "By the last frame, Sarah's grey leggings are off." in notes(shots[1])
          and "Sarah <Picture 1>: a young woman with long blonde hair, in a white blouse." in shots[2]["prompt"]
          and "Sarah is naked from the waist down the whole shot." in notes(shots[2]), shots[2]["prompt"])
    check("a garment put on is worn from then on and covers her again",
          "By the last frame, Sarah wears a red skirt." in notes(shots[3]) and "Sarah also wears a red skirt." in shots[4]["prompt"]
          and "naked" not in notes(shots[4]), shots[4]["prompt"])
    off, _ = P.plan("A room.\n\nSarah undresses.\nremove: Sarah\n\nSarah waits.", MEM)
    check("remove: with no item takes everything off",
          "Sarah: a young woman with long blonde hair." in off[1]["prompt"] and "completely naked" in notes(off[1]), off[1]["prompt"])
    mem = "Sarah: a young woman with long blonde hair, wearing a white top and black leggings.\nDana: a woman with short black hair."
    said = {}
    for tag in ("leggings", "leggings, thong", "bottoms", "clothes", "all her clothes", "top and leggings", "her top; her leggings"):
        got, warn = P.plan(f"A room.\n\nDana and Sarah stand.\n\nDana undresses Sarah.\nremove: Sarah, {tag}\n\nSarah waits.", mem)
        said[tag] = (next(l for l in got[2]["prompt"].split("\n") if l.startswith("Sarah:")), notes(got[2]))
    check("garments listed with commas or 'and', 'bottoms', and 'clothes' or 'all her clothes' all come off the description",
          all("leggings" not in d and "naked from the waist down" in n for d, n in (said[t] for t in ("leggings", "leggings, thong", "bottoms")))
          and all(d == "Sarah: a young woman with long blonde hair." and "completely naked" in n
                  for d, n in (said[t] for t in ("clothes", "all her clothes", "top and leggings", "her top; her leggings"))), said)
    loose, warn = P.plan("A room.\n\nDana and Sarah stand.\n\nSarah undresses.\nremove: Sarah leggings\n\nDana waits.\n"
                         "remove: Sarah, pants", mem)
    check("a tag with no comma after the name still reads, and a garment the description does not have is flagged",
          "clothes off: Sarah: black leggings." in loose[1]["line"] and any("'pants' is not in Sarah's description" in w for w in warn), warn)
    cuffed, _ = P.plan("A room.\n\nDana and Sarah stand.\n\nDana cuffs Sarah.\nhold: Sarah, handcuffs behind her back\n\n"
                       "Dana pulls down Sarah's leggings.\nremove: Sarah, leggings\n\nSarah waits.",
                       mem.replace("Sarah:", "Sarah <Picture 1>:"), pictures=[IMG])
    check("cuffed wrists are held behind her back from the moment they go on, and stay locked while clothes come off",
          "Sarah's arms stay pulled behind her back from the moment the handcuffs are on" in notes(cuffed[1])
          and "Sarah's arms stay pulled behind her back the whole shot" in notes(cuffed[3])
          and "Sarah's handcuffs behind her back stay locked the whole time her black leggings come off" in notes(cuffed[2])
          and "Once they are off, Sarah is naked from the waist down for the rest of the shot." in notes(cuffed[2]), notes(cuffed[2]))
    check("a dress is matched whole, not as 'dre'", P.undress("a woman in a red dress, with long hair", ["dress"])
          == "a woman with long hair." or P.undress("in a red dress, with long hair", ["dress"]) == "with long hair.",
          P.undress("in a red dress, with long hair", ["dress"]))
    twins = ("Sarah: a woman with long blonde hair, in a grey jacket, white t-shirt, blue jeans and white sneakers.\n"
             "Dana: a woman with short black hair, in a black jacket, red t-shirt, black jeans and boots.")
    pair = ("Sarah: a woman with long blonde hair, in a grey hoodie, white t-shirt, blue jeans and white sneakers.\n"
            "Marcus: a man with short dark hair, in a black jacket, grey t-shirt, blue jeans and black sneakers.")
    read = lambda b, m=twins: [x for s in P.plan("A room.\n\nDana, Marcus and Sarah stand.\n\n" + b, m)[0] for x in s["read"]]
    check("a garment both women wear goes to the one who takes it off, or to the one she acts on, or to both with 'their'",
          read("Dana takes off her jacket.") == [("remove", "Dana, black jacket")]
          and [x[1] for x in read("Dana grabs Sarah and pulls her jacket off.")] == ["Sarah, grey jacket"]
          and [x[1] for x in read("Dana looks at Sarah and takes off her jacket.")] == ["Dana, black jacket"]
          and [x[1] for x in read("Sarah and Dana take off their jackets.")] == ["Sarah, grey jacket", "Dana, black jacket"]
          and [x[1] for x in read("Sarah removes her t-shirt, then her jeans.")] == ["Sarah, white t-shirt", "Sarah, blue jeans"]
          and [x[1] for x in read("Dana takes her jacket off and hangs it up.")] == ["Dana, black jacket"],
          [read(b) for b in ("Dana takes off her jacket.", "Sarah and Dana take off their jackets.")])
    check("a garment taken off and put back on, or handed to someone, never strips the one who took it off",
          read("Marcus takes off his jacket and puts it back on.", pair) == []
          and [x[1] for x in read("Marcus takes off his jacket and drapes it over her shoulders.", pair)]
          == ["Marcus, black jacket", "Sarah, a jacket"],
          read("Marcus takes off his jacket and drapes it over her shoulders.", pair))


def test_layers():
    print("\n=== layers, lifted clothes and clothes against restraints ===")
    mem = ("Sarah <Picture 1>: a young woman with long blonde hair, in a white tank top, pink pajama bottoms and a black thong.\n"
           "Dana: a woman with short black hair, in a black suit.")
    sheet = lambda shot: next(l for l in shot["prompt"].split("\n") if l.startswith("Sarah"))
    shots, warn = P.plan("A bedroom.\n\nDana and Sarah stand.\n\nDana pulls Sarah's pajama bottoms off.\n"
                         "remove: Sarah, pajama bottoms\n\nSarah waits.", mem, pictures=[IMG])
    check("underwear under another garment is left out of the description until what covers it comes off, then shows",
          sheet(shots[0]).endswith("in a white tank top and pink pajama bottoms.") and "thong" not in shots[0]["prompt"]
          and "Under them, Sarah has a black thong on, now in plain view." in notes(shots[1])
          and "naked" not in notes(shots[1]) and sheet(shots[2]).endswith("in a white tank top and a black thong. Below the waist, "
                                                                          "she wears only a black thong."), sheet(shots[2]))
    check("a bra shows under nothing, hides under a shirt, and shows through a sheer one",
          P.hidden("in a black bra and jeans.") == [] and P.hidden("in a grey shirt and a black bra.") == ["a black bra."]
          and P.hidden("in a sheer grey shirt and a black bra.") == [], P.hidden("in a grey shirt and a black bra."))
    shots, warn = P.plan("A bedroom.\n\nDana and Sarah stand.\n\nDana cuffs Sarah.\nhold: Sarah, handcuffs behind her back\n\n"
                         "Dana lifts Sarah's tank top.\nlift: Sarah, tank top\n\nSarah squirms.\n\nDana pulls it down again.\n"
                         "wear: Sarah, tank top\n\nDana pulls her pajama bottoms down.\npull: Sarah, pajama bottoms", mem, pictures=[IMG])
    check("a lifted top stays on, pulled up, while cuffed wrists stay locked behind her back",
          "By the last frame, Sarah's white tank top is pulled up above her breasts and stays there, still on her, leaving her breasts "
          "bare; her wrists stay locked together behind her back the whole time it moves, and it never goes over her head or "
          "down her arms." in notes(shots[2]), notes(shots[2]))
    check("it stays pulled up in the shots after, until wear: puts it back",
          "Her white tank top is pulled up above her breasts, leaving her breasts bare." in sheet(shots[3])
          and "Sarah's white tank top stays pulled up above her breasts the whole shot." in notes(shots[3])
          and "By the last frame, Sarah's white tank top is pulled back into place." in notes(shots[4])
          and "pulled up" not in sheet(shots[5]), sheet(shots[3]))
    check("pull: bottoms go down to the knees and show what was hidden under them",
          "Sarah's pink pajama bottoms are pulled down to her knees and stay there, still on her." in notes(shots[5])
          and "Under them, Sarah has a black thong on, now in plain view." in notes(shots[5]), notes(shots[5]))
    shots, warn = P.plan("A bedroom.\n\nDana and Sarah stand.\n\nDana cuffs Sarah.\nhold: Sarah, handcuffs behind her back\n\n"
                         "Dana strips Sarah.\nremove: Sarah, clothes\n\nDana uncuffs her.\nrelease: Sarah\n\n"
                         "Dana takes the top off.\nremove: Sarah, tank top", mem, pictures=[IMG])
    check("clothes that cannot come off past cuffed wrists are pulled up instead and info says why; the rest come off",
          "clothes off: Sarah: pink pajama bottoms; black thong." in shots[2]["line"]
          and "pulled: Sarah: white tank top up above her breasts." in shots[2]["line"]
          and any("white tank top cannot come off past her bound wrists" in w for w in warn)
          and "Sarah's handcuffs behind her back stay locked the whole time her pink pajama bottoms and black thong come off"
          in notes(shots[2]), (shots[2]["line"], warn))
    check("once the cuffs are off, the top comes off", "clothes off: Sarah: white tank top." in shots[4]["line"]
          and "completely naked" in notes(shots[4]), shots[4]["line"])
    bound, warn = P.plan("A cell.\n\nSarah sits.\n\nDana pulls off Sarah's pajama bottoms.\nremove: Sarah, pajama bottoms",
                         mem + "\nhold: Sarah, rope around her ankles", pictures=[IMG])
    check("bottoms cannot come off past bound ankles, so the beat itself pulls them down to just above the rope",
          "pulled: Sarah: pink pajama bottoms down to just above the rope around her ankles." in bound[1]["line"]
          and bound[1]["prompt"].split("\n\n")[-2] == "Dana pulls Sarah's pajama bottoms down to just above the rope around her ankles.",
          (bound[1]["line"], bound[1]["prompt"].split("\n\n")[-2]))
    check("a removed garment that began the list hands its 'in' to the next, and a list keeps its 'and'",
          P.undress("a tall woman in a red dress and black heels.", ["dress"]) == "a tall woman in black heels."
          and P.undress("in a white tank top, pink pajama bottoms and a black thong.", [], ["a black thong."])
          == "in a white tank top and pink pajama bottoms.")
    office = "Dana: a woman in her thirties, short red hair, black blazer, blue shirt, grey trousers"
    blazer = P.plan("An office.\n\nDana takes off her blazer.\n\nDana waits.", office)[0]
    scarf = P.plan("An office.\n\nDana wraps a red scarf around her neck.\nwear: Dana, red scarf\n\nDana waits.", office)[0]
    dana = lambda shot: next(l for l in shot["prompt"].split("\n") if l.startswith("Dana"))
    check("a garment named without an article is hers, and a description without a full stop ends before the next sentence",
          dana(blazer[1]).endswith("blue shirt, grey trousers. Above the waist, she wears only her blue shirt.")
          and dana(scarf[1]).endswith("blue shirt, grey trousers. Dana also wears her red scarf."), (dana(blazer[1]), dana(scarf[1])))


def test_pictures():
    print("\n=== pictures ===")
    pics = [IMG, None, IMG, None]
    mem = MEM + "\nMia <Picture 3>: a woman with red hair."
    shots, _ = P.plan("A room.\n\nSarah waits.\n\nMia walks in.\n\nSarah and Mia talk.", mem, pictures=pics)
    check("pictures go in for the people in view, renumbered from 1, with the opening frame after them",
          shots[1]["refs"] == [0, 2] and "Sarah <Picture 1>:" in shots[1]["prompt"] and "Mia <Picture 2>:" in shots[1]["prompt"]
          and "<Picture 3> is the frame this shot opens on" in notes(shots[1]), (shots[1]["refs"], shots[1]["prompt"]))
    check("a person's picture goes in every shot they are in, not only the one they come into view in",
          [s["refs"] for s in shots] == [[0], [0, 2], [0, 2]] and "Mia <Picture 2>:" in shots[2]["prompt"], [s["refs"] for s in shots])
    check("people without a picture are told apart by their hair, people with one are not",
          "black-haired Dana" in notes(P.plan("A room.\n\nDana and Sarah wait.", MEM, pictures=[IMG])[0][0])
          and "blonde-haired Sarah" not in notes(P.plan("A room.\n\nDana and Sarah wait.", MEM, pictures=[IMG])[0][0]))
    forms = {"Sarah <Picture 1>: a woman.": 0, "Sarah: <Picture 1> a woman.": 0, "Sarah: a woman. <Picture 1>": 0,
             "Sarah (<Picture 1>): a woman.": 0, "**Sarah** <Picture 1>: a woman.": 0, "- Sarah <Picture 1>: a woman.": 0,
             "Sarah <Picture 1>\nA woman.\nHair: long blonde.": 0, "SARAH - a woman <Picture 1>": 0,
             "Sarah, <picture 1>: a woman.": 0, "Sarah: a woman.": 1}
    for mem, in_beat in forms.items():
        beat = "A room.\n\nDana waits.\n\nSarah " + ("<Picture 1> " if in_beat else "") + "walks in.\n\nSarah sits."
        got = P.plan(beat, mem + "\nDana: a woman.", pictures=[IMG])[0]
        check(f"the picture is Sarah's, and goes in every shot she is in, from '{mem.splitlines()[0]}'"
              + (" with the tag in a beat" if in_beat else ""),
              [s["refs"] for s in got] == [[], [0], [0]] and all(s["faces"] == [True] for s in got[1:])
              and got[1]["cast"] == ["Dana", "Sarah"] and "<Picture 1>" in got[2]["prompt"], [(s["cast"], s["refs"]) for s in got])
    first, _ = P.plan("A room.\n\nSarah waits.", MEM, first_frame=True, pictures=[IMG])
    check("a first_frame opens the first shot as the picture after the references",
          first[0]["keyed"] and "<Picture 2> is the frame this shot opens on." in notes(first[0]), notes(first[0]))


def test_lengths():
    print("\n=== shot lengths and speech ===")
    shots, _ = P.plan("A room.\n\nDana sits.\n\nDana stands, walks to the door, opens it and then looks back at the empty room.\n\n"
                      'Dana says "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen."\n\n'
                      "Dana waits.\nseconds: 12", MEM, shot_seconds=6.0)
    check("every length is 17k+5 frames, short beats get the floor, longer ones more up to shot_seconds",
          all((s["frames"] - 5) % 17 == 0 for s in shots) and shots[0]["frames"] == P.MIN_FRAMES
          and P.MIN_FRAMES < shots[1]["frames"] <= P.align(6 * 24), [s["frames"] for s in shots])
    check("a spoken line gets the time it needs past shot_seconds, its speech tags, a lead and quiet after it",
          shots[2]["frames"] > P.align(6 * 24) and "lengthened to fit its line" in shots[2]["line"] and "<d>one two" in shots[2]["prompt"]
          and not shots[2]["silent"] and shots[2]["cues"] == [(0.5, round(0.5 + P.SAY_SLACK + 18 / P.WORDS_PER_SEC, 2), "line")],
          (shots[2]["line"], shots[2]["cues"]))
    check("seconds: sets the length exactly, and a wordless shot is silent", shots[3]["frames"] == P.align(12 * 24)
          and shots[3]["silent"] and "Nobody speaks, and every mouth stays closed." in notes(shots[3]))
    fixed, _ = P.plan("A room.\n\nDana sits.\n\nDana stands.", MEM, shot_seconds=5.0, from_beat=False)
    check("fixed gives every shot shot_seconds", all(s["frames"] == P.align(120) for s in fixed))
    check("a first paragraph naming nobody is the scene, kept in every shot; with an anchor every paragraph is a beat",
          len(fixed) == 2 and all(s["prompt"].startswith("A room.") for s in fixed)
          and len(P.plan("A room.\n\nDana sits.", MEM, anchor="Night.")[0]) == 2)


def test_pace():
    print("\n=== actions at real-life speed ===")
    mem = ("Sarah: a young woman with long blonde hair, in a white blouse, a black skirt, a black bra, black panties and black heels.\n"
           "Dana: a woman with short black hair, in a black jacket and grey jeans.")
    plan = lambda b, secs=15.0: P.plan("A bedroom.\n\nDana and Sarah stand by the bed.\n\n" + b, mem, shot_seconds=secs)
    beat = lambda s: s["prompt"].split("\n\n")[-2]
    sat, cuffed, (stripped, warn) = plan("Dana sits on the bed.")[0], plan("Dana cuffs Sarah's hands behind her back.")[0], \
        plan("Dana strips Sarah naked.")
    check("a beat takes as long as its actions take in real life, and stripping her runs on into a second shot instead of being rushed",
          sat[1]["frames"] == P.MIN_FRAMES and cuffed[1]["frames"] > P.MIN_FRAMES and [s["beat"] for s in stripped] == [1, 2, 2]
          and "beat 2, part 1 of 2" in stripped[1]["line"] and stripped[1]["frames"] >= 13 * 24 and stripped[2]["frames"] >= 8 * 24
          and not warn,
          [(s["frames"], s["line"]) for s in stripped])
    check("the first part says only what happens in it, and the last keeps the beat as written",
          beat(stripped[1]) == "Dana takes off Sarah's black heels, white blouse and black skirt."
          and beat(stripped[2]) == "Dana strips Sarah naked." and "clothes off: Sarah: black heels; white blouse; black skirt." in
          stripped[1]["line"]
          and "blonde-haired Sarah is completely naked" in notes(stripped[2]), [beat(s) for s in stripped])
    check("every garment comes off step by step, the way it really comes off, at real-life speed",
          "Black-haired Dana takes off blonde-haired Sarah's black heels one foot at a time, then unbuttons blonde-haired Sarah's white "
          "blouse from the top down and slides it off her shoulders and down her arms, then opens blonde-haired Sarah's black skirt at "
          "the waist and slides it down her legs and off over her feet." in notes(stripped[1])
          and "unhooks blonde-haired Sarah's black bra and slides the straps off her shoulders and down her arms" in notes(stripped[2])
          and "It all happens at natural, real-life speed, one step after another, over about 13 seconds: no step is skipped, sped up or "
              "cut short, and no piece of clothing appears, vanishes or changes by itself." in notes(stripped[1]),
          notes(stripped[1]))
    freed = plan("Dana gags Sarah with a ball gag.\n\nDana unbuckles the gag and takes it out of Sarah's mouth.")[0]
    check("a restraint goes on and comes off step by step, and a shot where nothing changes still runs at real-life speed",
          "Black-haired Dana brings blonde-haired Sarah's arms behind her back and closes one cuff around her left wrist with a click, "
          "then the other around her right wrist." in notes(cuffed[1])
          and "Black-haired Dana pushes the ball into blonde-haired Sarah's mouth behind her teeth" in notes(freed[1])
          and "Black-haired Dana unbuckles the gag and takes it out of blonde-haired Sarah's mouth." in notes(freed[2])
          and "Everything happens at natural, real-life speed." in notes(cuffed[0]), notes(freed[2]))
    check("the speed line names restraints only in a shot that has them",
          "and no piece of clothing appears" in notes(stripped[1]) and "and no restraint or piece of clothing appears" in notes(freed[1]),
          (notes(stripped[1]), notes(freed[1])))
    gagged = plan('Dana holds up a ball gag. "Open wide." Sarah clamps her lips shut. "Fine. The hard way." Dana works the ball '
                  'between Sarah\'s teeth and buckles the strap. "Nnnh!"', 10.0)[0]
    said = [x for s in gagged if s["beat"] == 2 for x in s["said"]]
    check("a beat that runs over several shots keeps every line, each with the speaker the whole beat gives it",
          len(gagged) > 2 and said == [("Dana", "Open wide.", False), ("Dana", "Fine. The hard way.", False), ("Sarah", "Nnnh!", True)],
          said)
    near, fixed = plan("Dana pulls Sarah's blouse and skirt off.", 10.0), plan("Dana strips Sarah naked.\nseconds: 8")
    check("a beat a little over shot_seconds stays one shot, and a beat with seconds: is never split but is flagged as rushed",
          len(near[0]) == 2 and not near[1] and len(fixed[0]) == 2 and fixed[0][1]["frames"] == P.align(8 * 24)
          and any("may look rushed" in w for w in fixed[1]), (near, fixed[1]))
    tagged = plan("Dana pulls down Sarah's skirt.\nremove: Sarah, black skirt")[0][1]
    check("a change from a tag line names nobody doing it, so it gets no steps, but still runs at real-life speed",
          notes(tagged).startswith("It all happens at natural, real-life speed, one step after another, over about 4 seconds"),
          notes(tagged))
    fight = plan("Dana strips Sarah down to her panties while Sarah struggles.")[0]
    check("a strip while she struggles is still read, and comes off in a real order, heels first",
          "clothes off: Sarah: black heels; white blouse; black skirt." in fight[1]["line"] and "black bra" in fight[2]["line"],
          [s["line"] for s in fight])
    spread = plan("Sarah lies on the bed. Dana ties Sarah spread-eagle to the bed.")[0]
    check("someone already lying on the bed from the shot before is not laid down again",
          "Black-haired Dana lays blonde-haired Sarah down on her back on the bed" in notes(spread[1])
          and "Black-haired Dana spreads blonde-haired Sarah's legs out wide and ties her ankles to the bed." in notes(spread[2])
          and "lays blonde-haired Sarah down" not in notes(spread[2]), notes(spread[2]))
    tight = ("a crotch chain locked tight around her waist, its center chain pulled tight under her crotch and over her vagina from the "
             "front of her waist to the small of her back")
    chained = plan("Dana puts a crotch chain on Sarah.\n\nDana unlocks the chain and takes it off Sarah.")[0]
    same, roped = (plan(b)[0][1] for b in ("Dana wraps a chain around Sarah's waist and between her legs and locks it with a padlock.",
                                           "Dana ties a rope around Sarah's waist and between her legs."))
    check("a crotch chain is a waist chain and a center chain locked to it at the front and back, tight under her crotch over her "
          "vagina and all of it at her hips, on and off in one step",
          "Black-haired Dana takes a separate crotch chain from her pocket, wraps its waist chain tight around blonde-haired Sarah's "
          "waist, runs its center chain from the front of the waist chain under her crotch, centers it tight over her vagina, brings it "
          "up between her buttocks to the small of her back and locks it to the back of the waist chain with a padlock."
          in notes(chained[1]) and f"hold: Sarah, {tight} (" in chained[1]["line"] and f"hold: Sarah, {tight} (" in same["line"]
          and "Blonde-haired Sarah's crotch chain has two parts: a waist chain locked tight around her waist like a belt, and a "
              "center chain locked to the front of the waist chain below her navel and to the back of the waist chain at the small of "
              "her back. Between those two points the center chain runs under her crotch, pulled tight over her vagina, and up between "
              "her buttocks, so all of it stays at her hips. It stays taut and centered the whole time and never hangs loose, slips to "
              "one side or sags." in notes(chained[1])
          and not any(w in notes(chained[1]) for w in ("straight down", "solid steel links", "between her legs"))
          and "Black-haired Dana unlocks the padlock at the small of blonde-haired Sarah's back, pulls the center chain out from under "
              "her crotch and takes the crotch chain off her waist." in notes(chained[2])
          and "hold: Sarah, a crotch rope tied tight around her waist, its center rope pulled tight under her crotch and over her vagina "
              "from the front of her waist to the small of her back (" in roped["line"],
          (chained[1]["line"], notes(chained[1]), notes(chained[2]), roped["line"]))
    later = plan("Dana chains Sarah's waist.\n\nDana runs the chain between Sarah's legs and padlocks it.")[0]
    check("a chain run between her legs later turns the chain around her waist into a crotch chain",
          "Black-haired Dana hooks a center chain to the front of the chain around blonde-haired Sarah's waist, runs it under her crotch, "
          "centers it tight over her vagina, brings it up between her buttocks to the small of her back and locks it to the back of the "
          "waist chain with a padlock."
          in notes(later[2]) and f"goes on: Sarah: {tight}." in later[2]["line"] and "comes off" not in later[2]["line"],
          (later[2]["line"], notes(later[2])))
    bare, warn = plan("Dana strips Sarah naked and wraps a chain around her waist and between her legs.", 10.0)
    check("a name something is done to never does the next thing, and a crotch chain after a strip goes on whole in one shot",
          len([s for s in bare if f"hold: Sarah, {tight}" in s["line"]]) == 1 and not any("hold: Dana" in s["line"] for s in bare)
          and "Black-haired Dana takes a separate crotch chain from her pocket, wraps its waist chain tight around blonde-haired Sarah's "
              "waist"
          in notes(bare[-1])
          and not any("may look rushed" in w for w in warn), ([s["line"] for s in bare], warn))
    lap, waist, pull = (plan(b)[0][1] for b in ("Dana cuffs Sarah's hands between her legs.", "Dana chains Sarah's hands at her waist.",
                                                "Dana uses both hands to pull the chain between her legs."))
    check("hands between her legs or at her waist are where her bound hands go, never a second restraint on her crotch or waist",
          "hold: Sarah, handcuffs on her wrists between her legs (" in lap["line"] and "crotch" not in lap["line"]
          and "Black-haired Dana brings blonde-haired Sarah's hands down between her legs and closes one cuff around her left wrist "
              "with a click, then the other around her right wrist." in notes(lap)
          and "hold: Sarah, chains around her wrists at her waist (" in waist["line"]
          and "brings blonde-haired Sarah's hands to her waist and wraps the chain around her wrists" in notes(waist)
          and "both arms bend at the elbows so her hands rest at her waist" in notes(waist)
          and f"hold: Sarah, {tight} (" in pull["line"] and "wrists" not in pull["line"],
          (lap["line"], waist["line"], pull["line"]))
    strapped = plan("Dana straps Sarah's wrists and ankles to the bed frame.")[0][1]
    armed = plan("Dana pushes Sarah down onto a wooden chair. She ties her wrists to the arms of the chair with rope.")[0][1]
    check("straps on her wrists and ankles are read, and wrists tied to the arms of a chair rest on them, not behind her back",
          "hold: Sarah, straps around her wrists, strapped to the bed frame; straps around her ankles, strapped to the bed frame"
          in strapped["line"]
          and "hold: Sarah, rope around her wrists, tied to the arms of the chair (" in armed["line"]
          and "Blonde-haired Sarah sits on the chair, her wrists tied to the arms of the chair from the moment the rope is on: her "
              "forearms rest along the arms of the chair" in notes(armed) and "behind her back" not in notes(armed),
          (strapped["line"], notes(armed)))
    watched, pinned = (plan(b)[0][1] for b in ("Dana watches as Sarah struggles. She gags her.",
                                               "Dana pins Sarah down while Sarah struggles, and gags her."))
    check("she and her go to the one acting and the one struggling, whichever name came last",
          all("Black-haired Dana fits the gag into blonde-haired Sarah's mouth" in notes(s) for s in (watched, pinned)),
          (notes(watched), notes(pinned)))
    masked = plan("Two masked women grab Sarah, tie her wrists with rope and tape her mouth.", 10.0)[0]
    selfcuff = plan("Sarah snaps the handcuffs onto her own wrists behind her back.")[0][1]
    shown = plan("Sarah stumbles in with her hands cuffed in front of her and tape over her mouth.")[0][1]
    check("unnamed people stay in every part of a split beat, nobody who did it herself lets go, and cuffs and tape on two spots are "
          "both read",
          "One of the two women brings blonde-haired Sarah's arms behind her back and wraps the rope around her wrists" in notes(masked[1])
          and "and one of the two women has let go." in notes(masked[1]) and "the two women." in notes(masked[1])
          and "By the last frame, blonde-haired Sarah has handcuffs behind her back." in notes(selfcuff)
          and "on: Sarah, handcuffs in front of her; tape over her mouth (" in shown["line"],
          (notes(masked[1]), notes(selfcuff), shown["line"]))
    coat = plan("A stranger takes off his coat and hands it to Sarah, and she puts it on.")[0][1]
    later = plan("Dana cuffs Sarah's hands behind her back. Her ankles are bound with rope a moment later, and Dana steps back.")[0][1]
    check("after ', and she' the one named last can act, and a restraint said to happen a moment later goes on in the shot",
          "Blonde-haired Sarah puts on a coat." in notes(coat)
          and "then crouches at blonde-haired Sarah's feet, holds Sarah's ankles pressed together, both feet resting where they are, and "
              "wraps the rope several times around both of them together" in notes(later),
          (notes(coat), notes(later)))
    cuff = "Dana cuffs Sarah's hands behind her back.\n\n"
    belly, named, junk = (plan(cuff + b, 10.0) for b in (
        "Dana wraps chains around Sarah's waist and between her legs and chains them to her handcuffs behind her back.",
        "Dana puts a crotch chain on Sarah and locks it to her handcuffs behind her back.",
        "Dana wraps a chain around Sarah's waist, runs it between her legs and locks it to the handcuffs behind her back."))
    step = ("Black-haired Dana takes a separate crotch chain from her pocket, wraps its waist chain tight around blonde-haired Sarah's "
            "waist, runs its center chain from the front of the waist chain under her crotch, centers it tight over her vagina, brings it "
            "up between her buttocks to the small of her back and locks it to the back of the waist chain with a padlock, then padlocks a "
            "separate short chain from the chain between her handcuffs to the back of her crotch chain.")
    check("a crotch chain locked to her handcuffs goes on in one step in one shot, linked to her cuffs by a separate short chain of its "
          "own, which the beat names too",
          all(len(s) == 3 and not w and step in notes(s[2]) and f"hold: Sarah, {tight}, chained to her handcuffs at the small of her "
              f"back by a separate short chain (" in s[2]["line"] and "cuff around" not in notes(s[2])
              and "hold: Sarah, handcuffs" not in s[2]["line"] and "passes it through" not in notes(s[2])
              and beat(s[2]).endswith("behind her back with a separate short chain.") for s, w in (belly, named, junk))
          and "with her wrists locked together behind her and held to the back of the crotch chain by a separate short chain: "
              in notes(belly[0][2])
          and "and the only link between them is a separate short chain from the chain between her handcuffs to the back of her crotch "
              "chain." in notes(belly[0][2]), [(s[2]["line"], beat(s[2]), notes(s[2]), w) for s, w in (belly, named, junk)])
    onto = plan("Dana puts a crotch chain on Sarah.\n\nDana cuffs Sarah's hands behind her back to the crotch chain.")[0][2]
    relink, (uncuffed, rested) = plan(cuff + "Dana puts a crotch chain on Sarah.\n\nDana locks the crotch chain to Sarah's handcuffs.")[0][3], \
        plan(cuff + "Dana puts a crotch chain on Sarah and locks it to her cuffs.\n\nDana uncuffs Sarah.\n\nSarah rubs her wrists.")[0][3:5]
    check("cuffs put on to a crotch chain, a link made later, and uncuffing her all change only the link",
          "then the other around her right wrist, then padlocks a separate short chain from the chain between blonde-haired Sarah's "
          "handcuffs to the back of her crotch chain." in notes(onto)
          and "Black-haired Dana padlocks a separate short chain from the chain between blonde-haired Sarah's handcuffs to the back of "
              "her crotch chain." in notes(relink) and relink["frames"] == P.MIN_FRAMES
          and "Black-haired Dana unlocks the separate short chain that joins blonde-haired Sarah's handcuffs to the back of her crotch "
              "chain and takes it away, then unlocks the handcuffs with a small key and takes them off her wrists." in notes(uncuffed)
          and f"Blonde-haired Sarah: {tight}." in rested["prompt"], (notes(onto), notes(relink), notes(uncuffed), rested["prompt"]))
    split, warn = plan("Dana cuffs Sarah's hands behind her back and puts a crotch chain on her, locking it to her handcuffs.", 10.0)
    front = plan("Dana cuffs Sarah's hands in front of her.\n\nDana puts a crotch chain on Sarah and locks it to her handcuffs.")[0][2]
    walked = plan("Dana walks Sarah down the corridor in handcuffs.")[0][1]
    check("split over short shots the cuffs go on first and the whole crotch chain next, told as just that; cuffed in front it locks at "
          "the front of her waist; and 'in handcuffs' is worn",
          "beat 2, part 2 of 2" in split[2]["line"]
          and beat(split[2]) == "Dana puts a crotch chain on Sarah, locking it to her handcuffs with a separate short chain."
          and step in notes(split[2]) and not warn
          and "then padlocks a separate short chain from the chain between her handcuffs to the front of her crotch chain." in notes(front)
          and "held to the front of the crotch chain by a separate short chain: both arms bend at the elbows" in notes(front)
          and "on: Sarah, handcuffs behind her back (" in walked["line"],
          (beat(split[2]), notes(split[2]), warn, notes(front), walked["line"]))


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
    pdd = Patched({pe.WrappersMP.DIFFUSION_MODEL: {modelprep.PDD_KEY: [None]}})
    m, steps, sampler, sigmas, notes = modelprep.prepare(pdd, 8, "res_multistep", "simple", torch.linspace(1, 0, 9), 12.0, 3.0,
                                                         {}, "9")
    tops = m.model_options["transformer_options"]
    check("PDD Acc samples on its own sigmas with euler at shift 12/3",
          sampler == "euler" and tops["minimax_h3_sigma_shift_video"] == 12.0 and "PDD Acc" in notes[0], notes)
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
        _, steps, sampler, sigmas, notes = modelprep.prepare(Patched(), 8, "res_multistep", "simple", torch.linspace(1, 0, 9),
                                                             12.0, 3.0, graph, "9")
        check("stray PDD sigmas are dropped, and a few-step LoRA is sampled with euler on the steps set",
              sampler == "euler" and steps == 8 and sigmas is None and "left out" in notes[0], notes)
        modelprep.loras = lambda g, n=None: ["flashgen_4step.safetensors"]
        _, steps, sampler, sigmas, _ = modelprep.prepare(Patched(), 8, "res_multistep", "simple", None, 12.0, 3.0, {}, "9")
        check("FlashGen gets its own 4-step grid shifted at 12", steps == 4 and abs(float(sigmas[1]) - 12 * 0.7 / (1 + 11 * 0.7)) < 1e-6)
        check("a turbo LoRA or a T2V checkpoint takes no pictures, a ref2va checkpoint with a style LoRA does",
              not modelprep.takes_pictures(Patched(), {}, "9")[0]
              and (setattr(modelprep, "loras", lambda g, n=None: ["Astro nsfw.safetensors"]) or modelprep.takes_pictures(Patched(), {}, "9")[0]))
    finally:
        modelprep.loras, modelprep.meta, modelprep.checkpoint = saved
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
    box = faces.head_box([[50, 40, 1], [50, 60, 1]] + [[0, 0, 0]] * 12 + [[46, 38, 1], [54, 38, 1]], 100, 200)
    check("the face crop runs from above the hair to the chin and is squared on the head",
          box is not None and box[1] < 30 and 50 < box[3] < 60 and abs(box[2] - box[0] - box[3] + box[1]) <= 1, box)


def render(prompt, dwpose=False, counts=(), textured=False, blind=False, **kw):
    calls = {"build": [], "sample": [], "gate": []}
    saved = (chain.build, chain.sample, chain.decode, chain.unvoice, chain.make_room, chain.cleanup, modelprep.prepare,
             modelprep.check_vaes, modelprep.takes_pictures, modelprep.partial, faces.available, faces.count, faces.crop, faces.Detector)
    real = chain.build

    def build(clip, vae, avae, text, w, h, frames, handoff=None, refs=(), memory=None, silent=False, cues=(), memo=None):
        calls["build"].append({"text": text, "handoff": handoff, "refs": list(refs), "memory": memory, "silent": silent})
        cond, lat = real(clip, vae, avae, text, w, h, frames, handoff, refs, memory, silent, cues, memo)
        return cond, dict(lat, shot=len(calls["build"]), frames=frames)

    def sample(model, cond, neg, lat, seed, steps, sampler, scheduler, sigmas):
        calls["sample"].append({"shot": lat["shot"], "audio_only": float(lat.get("noise_mask").unbind()[0].max()) == 0
                                if lat.get("noise_mask") is not None else False,
                                "heard": float(lat["noise_mask"].unbind()[1].mean()) if lat.get("noise_mask") is not None else 1.0})
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
    modelprep.prepare = lambda m, st, sn, sc, sg, sv, sa, g, nid: (m, st, sn, None, ["prepared"])
    modelprep.check_vaes = lambda v, a: None
    modelprep.takes_pictures = lambda m, g, n: (False, "x.safetensors was not trained on reference pictures") if blind else (True, "")
    modelprep.partial = lambda m, g, n: []
    faces.available = lambda: dwpose
    faces.count = lambda det, frame: next(seen, 2)
    faces.crop = lambda det, pic: pic[:, :8, :8]
    faces.Detector = lambda: SimpleNamespace(close=lambda: None)
    try:
        out = node.H3LongVideos().run("base", FakeClip(), FakeVAE(), FakeAudioVAE(), prompt, "16:9", 0.01, 4.0, 8, "euler",
                                      "simple", 7, character_memory=kw.pop("memory", MEM), **kw)
    finally:
        (chain.build, chain.sample, chain.decode, chain.unvoice, chain.make_room, chain.cleanup, modelprep.prepare,
         modelprep.check_vaes, modelprep.takes_pictures, modelprep.partial, faces.available, faces.count, faces.crop,
         faces.Detector) = saved
    return out, calls


def test_ankles():
    print("\n=== bound legs move together and are said once as a pair; free legs are left exactly as written ===")
    mem = ("Sarah: a young woman with long blonde hair, in a white blouse and a black skirt.\n"
           "Dana: a woman with short black hair, in a black jacket and grey jeans.\n"
           "Luis: a young man with short brown hair, in a grey t-shirt and jeans.")
    plan = lambda *b: P.plan("A cellar with a steel table, a wooden chair, a bed and a staircase.\n\n" + "\n\n".join(b), mem)
    beat = lambda s: s["prompt"].split("\n\n")[-2]
    cuffed, warn = plan("Dana cuffs Sarah's ankles.", "Sarah walks up the stairs.", "Sarah crosses her legs.")
    check("ankle cuffs go on one around each ankle, held together with her feet resting, she climbs the stairs one stair at a time "
          "with her ankles cuffed together, in a longer shot, and legs crossed are flagged",
          "Black-haired Dana crouches at blonde-haired Sarah's feet, holds Sarah's ankles pressed together, both feet resting where they "
          "are, and closes one cuff around each ankle with a click, the short chain between them, then lets go and stands back up."
          in notes(cuffed[0])
          and "Sarah's ankles stay handcuffed together the whole shot: a pair of rigid steel cuffs is locked around them just above her "
              "feet, joined by their own short chain of a few links that never stretches, bends or gets longer, so her ankles stay a "
              "few centimetres apart at most and her legs stay pressed together and always move together, both at once."
          in notes(cuffed[1])
          and beat(cuffed[1]) == "Sarah walks up the stairs, one stair at a time, both feet together on each stair, blonde-haired "
                                 "Sarah's ankles handcuffed together."
          and beat(cuffed[2]) == "Sarah crosses her legs, blonde-haired Sarah's ankles handcuffed together."
          and cuffed[1]["frames"] > P.MIN_FRAMES and "handcuffs are rigid" not in notes(cuffed[1])
          and not any(w in s["prompt"] for s in cuffed for w in ("full stride", "one leg", "as one", "legs are free"))
          and any("shot 3" in w and "needs Sarah's legs apart" in w for w in warn), (beat(cuffed[1]), warn))
    lying = plan("Sarah lies on the floor.", "Dana kneels at Sarah's feet and handcuffs her ankles together.", "Sarah struggles and kicks.",
                 "Sarah lifts both legs into the air.", "Sarah lies still.")[0]
    check("bound ankles are in her description before the beat, every movement of her legs has both legs move together with her "
          "ankles cuffed together, and the lying pose is written only while her legs stay still",
          "Black-haired Dana kneels at blonde-haired Sarah's feet, holds Sarah's ankles pressed together, her legs straight and flat "
          "where they lie, and closes one cuff around each ankle with a click, the short chain between them, then lets go and stands "
          "back up." in notes(lying[1])
          and "Once the handcuffs are on, she lies with her legs together." in notes(lying[1])
          and "Her ankles are handcuffed" not in lying[1]["prompt"]
          and "a black skirt. Her ankles are handcuffed together.\n" in lying[2]["prompt"]
          and beat(lying[2]) == "Sarah struggles, blonde-haired Sarah's ankles handcuffed together, and kicks."
          and beat(lying[3]) == "Sarah lifts both legs together into the air, blonde-haired Sarah's ankles handcuffed together."
          and not any("lies with her legs" in notes(s) for s in lying[2:4])
          and "All of it stays on, exactly as described, for the whole shot." in notes(lying[3])
          and beat(lying[4]) == "Sarah lies still." and "She lies with her legs together." in notes(lying[4]),
          [s["prompt"] for s in lying[1:]])
    grab = plan("Sarah lies on the floor, her ankles handcuffed together.", "Dana grabs her ankles and lifts her legs onto the bed.",
                "Dana bends her knees and kneels beside Sarah.")[0]
    late = plan("Sarah lies on the floor.", "Sarah kicks and struggles as Dana handcuffs her ankles together.")[0]
    knees = plan("Sarah lies on the bed.", "Dana ties Sarah's ankles together with rope, then her knees.")[0]
    pair = plan("Sarah and Dana sit on the floor, their ankles handcuffed together.", "Sarah and Dana kick their legs.")[0]
    gagged = plan("Sarah and Dana sit back to back, their wrists tied with rope and tape over their mouths.")[0][0]
    check("a grab takes her cuffed ankles by name and lifts her legs together, the other woman's own knees are left as written, cuffs "
          "put on while she struggles are read, a split beat ties her ankles together and then her knees, and 'their ankles' after "
          "two names binds both, each by name",
          beat(grab[1]) == "Dana grabs blonde-haired Sarah's ankles, blonde-haired Sarah's ankles handcuffed together, and lifts "
                           "blonde-haired Sarah's legs together onto the bed."
          and beat(grab[2]) == "Dana bends her knees and kneels beside Sarah."
          and "hold: Sarah, handcuffs on her ankles" in late[1]["line"]
          and beat(late[1]) == "Sarah kicks and struggles as Dana handcuffs blonde-haired Sarah's ankles together."
          and [beat(s) for s in knees[1:]] == ["Dana ties Sarah's ankles together.", "Dana ties Sarah's knees together."]
          and "on: Sarah: handcuffs on her ankles. Dana: handcuffs on her ankles." in pair[1]["line"]
          and gagged["read"] == [("have", f"{n}, rope around her wrists behind her back; tape over her mouth") for n in ("Sarah", "Dana")]
          and beat(pair[0]) == "Sarah and Dana sit on the floor, their ankles handcuffed together."
          and beat(pair[1]) == "Sarah and Dana kick their legs together, blonde-haired Sarah's ankles handcuffed together, black-haired "
                               "Dana's ankles handcuffed together.",
          (beat(grab[1]), beat(grab[2]), late[1]["line"], [beat(s) for s in knees], [(s["line"], beat(s)) for s in pair]))
    held, warn = plan("Sarah lies on the floor, her ankles handcuffed together. Luis stands over her.",
                      "Luis grabs her by the ankle and lifts her leg up.", "Sarah lifts one leg and shifts her weight from foot to foot.",
                      "Sarah lies on the floor with her ankles handcuffed together.")
    check("with a man in the shot nothing says anyone's legs are free, one leg or one ankle becomes the pair and is flagged, and words "
          "that only describe her cuffed ankles or name a foot in passing are left alone",
          not any("legs are free" in s["prompt"] for s in held)
          and beat(held[1]) == "Luis grabs blonde-haired Sarah by her ankles, her ankles handcuffed together, and lifts blonde-haired "
                               "Sarah's legs up together."
          and beat(held[2]) == "Sarah lifts her legs together, her ankles handcuffed together, and shifts her weight from foot to foot."
          and beat(held[3]) == "Sarah lies on the floor with her ankles handcuffed together."
          and [w.split(":", 1)[1].split(":")[0] for w in warn] == [
              ' "blonde-haired Sarah by the ankle" became "blonde-haired Sarah by her ankles"',
              ' "blonde-haired Sarah\'s leg" became "blonde-haired Sarah\'s legs"',
              ' "one leg" became "her legs"'], ([beat(s) for s in held], warn))
    free = plan("Sarah lies on her back on the bed. Luis stands at the foot of the bed.", "Luis grabs her ankles and lifts her legs up.",
                "Sarah kicks her legs.", "Luis grabs her left ankle.")[0]
    two = plan("Sarah and Dana sit on the bed.", "Dana grabs her legs.")[0]
    check("free legs are left as written: a grab, a kick and one ankle get no added 'both' or 'one in each hand', and 'her legs' that "
          "could be either woman's is left as written",
          [beat(s) for s in free[1:]] == ["Luis grabs blonde-haired Sarah's ankles and lifts blonde-haired Sarah's legs up.",
                                          "Sarah kicks her legs.",
                                          "Luis grabs blonde-haired Sarah's left ankle."]
          and beat(two[1]) == "Dana grabs her legs.", [beat(s) for s in free + two])
    irons, warn = plan("Luis puts ankle chains on Sarah.", "Sarah tries to run for the stairs.", "Sarah stands by the table.",
                       "Sarah lifts both legs, then one leg.")
    both = plan("Luis shuffles in, in leg irons, his hands cuffed in front of him.")[0][0]
    check("ankle chains are leg irons: hobbled steps with her ankles joined by the chain, the chain slack when she stands, both legs "
          "moving together while one leg alone is left as written, and a man in leg irons with cuffed hands has both",
          "hold: Sarah, shackles around her ankles" in irons[0]["line"]
          and "a pair of steel shackles is locked around them just above her feet, joined by their own heavy chain, about forty "
              "centimetres long, that never stretches or gets longer" in notes(irons[1])
          and beat(irons[1]) == "Sarah tries to run for the stairs, in short, hobbled steps, her ankles joined by the heavy chain of the "
                                "shackles."
          and "She stands with her feet a little apart, the chain between her ankles slack on the floor." in notes(irons[2])
          and "Her ankles are shackled together by their own heavy chain about forty centimetres long." in irons[1]["prompt"]
          and beat(irons[3]) == "Sarah lifts both legs together, her ankles joined by the heavy chain of the shackles, then one leg."
          and not warn and "on: Luis, shackles around his ankles; handcuffs in front of him" in both["line"], (beat(irons[1]), warn))
    tied, warn = plan("Dana ties Sarah's ankles together with rope.", "Sarah hops to the door.", "Sarah falls across the mattress.",
                      "Dana forces her to her knees.", "Sarah runs for the stairs.")
    check("tied ankles hop, fall and are forced down with her ankles tied together, lie and kneel with her legs together, and a run "
          "is flagged but still told as tiny steps",
          "hold: Sarah, rope around her ankles (" in tied[0]["line"] and "locked to the door" not in tied[1]["line"]
          and beat(tied[1]) == "Sarah hops to the door, blonde-haired Sarah's ankles tied together."
          and beat(tied[2]) == "Sarah falls across the mattress, blonde-haired Sarah's ankles tied together."
          and "She lies with her legs together." in notes(tied[2])
          and beat(tied[3]) == "Dana forces her to her knees, blonde-haired Sarah's ankles tied together."
          and "She kneels with her knees together and her feet behind her." in notes(tied[3])
          and beat(tied[4]) == "Sarah runs for the stairs, in tiny shuffling steps, blonde-haired Sarah's ankles tied together."
          and any("shot 5" in w and "needs Sarah's legs apart" in w for w in warn) and len(warn) == 1, ([beat(s) for s in tied], warn))
    floor = plan("Sarah stands by the bed, her ankles handcuffed together. Dana watches her.", "Sarah lays on the floor.",
                 "Sarah lies on the floor.", "The camera moves in close on Sarah's face.")[0]
    side, alone = (plan(b)[0][0] for b in ("Sarah and Dana lie on the floor, their ankles tied together with rope. Luis watches them.",
                                           "Sarah lies on the floor in leg irons."))
    keep = "and stays far enough back to keep all of blonde-haired Sarah in frame too, down to her handcuffed ankles."
    check("lying on the floor keeps her bound ankles in frame and bound: 'lays on the floor' is lying, going down from standing says "
          "her ankles are cuffed right there, the camera stays far enough back to show her down to the cuffs, and a beat that moves "
          "the camera itself keeps its own framing",
          beat(floor[1]) == "Sarah lays on the floor, blonde-haired Sarah's ankles handcuffed together."
          and "She lies with her legs together." in notes(floor[1]) and "She stands" not in notes(floor[1])
          and beat(floor[2]) == "Sarah lies on the floor." and all(keep in notes(s) for s in floor[:3])
          and "The camera keeps" not in notes(floor[3]) and "down to their tied ankles." in notes(side)
          and "The camera keeps Sarah's face in frame for the whole shot, and stays far enough back to keep all of her in frame too, "
              "down to her shackled ankles." in notes(alone)
          and "far enough back" not in notes(plan("Sarah lies on the floor. Dana watches her.")[0][0]),
          [(beat(s), notes(s)) for s in floor + [side, alone]])
    down = [(b, plan("Sarah stands by the bed, her ankles handcuffed together. Dana watches her.", b)[0][1]) for b in (
        "Sarah fell onto the floor.", "Sarah laid on the floor.", "Sarah lays face down on the floor.", "Dana lays Sarah on the floor.",
        "Dana laid Sarah on the bed.", "Sarah slept on the floor.", "Dana sits Sarah on the wooden chair.")]
    check("every way of lying down or being laid down is read as lying (sat down on the chair as sitting), with her cuffed ankles "
          "said right where she goes down",
          all(beat(s) == b[:-1] + ", blonde-haired Sarah's ankles handcuffed together." and (
              "She sits with her knees and feet together." if "sits" in b else "She lies with her legs together.") in notes(s)
              for b, s in down), [(beat(s), notes(s)) for _, s in down])
    chair, tether, bar = (plan(b, "Sarah walks across the cellar.")[0] for b in (
        "Dana handcuffs Sarah's ankles to the legs of the chair.", "Luis chains Sarah's left ankle to the bed frame with a long chain.",
        "Luis locks Sarah's ankles to either end of a spreader bar."))
    check("ankles cuffed to a chair seat her on it, one ankle chained to the bed leaves the other leg free, and a spreader bar holds "
          "her feet apart",
          "Blonde-haired Sarah sits on the chair, her ankles locked to the legs of the chair" in notes(chair[0])
          and "chains around her left ankle, chained to the bed frame" in tether[0]["line"] and "lies" not in notes(tether[0])
          and "her other leg moves freely" in notes(tether[1]) and beat(tether[1]) == "Sarah walks across the cellar."
          and "hold: Sarah, a spreader bar locked between her ankles" in bar[0]["line"]
          and "spreads her feet apart and locks one end of the spreader bar around each ankle" in notes(bar[0])
          and beat(bar[1]) == "Sarah walks across the cellar, waddling with both legs held wide apart, her ankles locked to the two ends "
                              "of the spreader bar.", (notes(chair[0]), notes(tether[1]), beat(bar[1])))
    knee, taped, still, loose, irons = (plan(*b)[0] for b in (
        ("Sarah sits on the wooden chair with her knees tied together.", "She swings her legs."),
        ("Sarah lies on the floor with her ankles duct-taped together. Dana stands over her.",),
        ("Sarah sits on the floor with her ankles zip-tied. Luis waits by the stairs.",
         "Sarah lies on the bed with the rope around her ankles pulled tight."),
        ("Sarah sits on the chair with her ankles tied together.", "Luis unties her legs.", "She kicks her legs out."),
        ("Sarah lies on the floor in leg irons.", "Luis grabs her legs and arms and pulls her onto the bed.",
         "Luis lifts her by the feet and holds her upside down.")))
    check("'with her knees tied together' binds her knees, duct-taped ankles stay on the one lying there, ankles only described as "
          "bound are left alone, untying her legs frees her ankles, and 'legs and arms' and 'holds her upside down' read whole",
          knee[0]["read"] == [("have", "Sarah, rope around her knees")]
          and beat(knee[1]) == "Sarah swings her legs together, her knees tied together."
          and taped[0]["read"] == [("have", "Sarah, duct tape around her ankles")]
          and [beat(s) for s in still] == ["Sarah sits on the floor with her ankles zip-tied. Luis waits by the stairs.",
                                           "Sarah lies on the bed with the rope around her ankles pulled tight."]
          and loose[1]["read"] == [("release", "Sarah, rope around her ankles")]
          and beat(loose[2]) == "Blonde-haired Sarah kicks her legs out."
          and [beat(s) for s in irons[1:]] == [
              "Luis grabs blonde-haired Sarah's legs and arms, her ankles joined by the heavy chain of the shackles, and pulls "
              "blonde-haired Sarah onto the bed.",
              "Luis lifts blonde-haired Sarah by the feet, her ankles joined by the heavy chain of the shackles, and holds blonde-haired "
              "Sarah upside down."],
          [(s["read"], beat(s)) for s in knee + taped + still + loose + irons])
    seen = cuffed + lying + grab + late + knees + pair + held + free + two + tied + chair + bar + knee + taped + still + loose + irons
    split = re.compile(r"\bleft (?:leg|ankle|foot|knee|thigh|calf|shin)\b[^.]{0,40}?\bright (?:leg|ankle|foot|knee|thigh|calf|shin)\b|"
                       r"\bone in each hand\b|\bside by side\b|\bon each ankle\b|"
                       r"\bboth (?:her|his|their|[\w-]+-haired \w+['’]s) (?:legs|ankles|knees|feet)\b")
    check("a pair of legs, ankles, feet or knees is said as one pair: never a left one and a right one, one in each hand, side by "
          "side, on each ankle or an added 'both her legs'",
          not any(split.search(s["prompt"]) for s in seen), [m.group(0) for s in seen for m in split.finditer(s["prompt"])])
    read = lambda b: [r for s in plan(b)[0] for r in s["read"]]
    check("the ankles are read however they are put: hand and foot, a list of restraints, 'then her ankles', a cut zip tie, an "
          "appositive, and a walk to something is never a tie to it",
          read("Sarah lies on the floor, bound hand and foot.") == [("have", "Sarah, rope around her wrists behind her back; rope around "
                                                                      "her ankles")]
          and read("Dana puts Sarah in a belly chain, handcuffs and leg irons.") == [
              ("hold", "Sarah, chains around her waist"), ("hold", "Sarah, shackles around her ankles"),
              ("hold", "Sarah, handcuffs behind her back")]
          and read("Dana zip-ties Sarah's wrists, then her ankles.") == [("hold", "Sarah, zip ties around her wrists behind her back"),
                                                                         ("hold", "Sarah, zip ties around her ankles")]
          and plan("Dana zip-ties Sarah's ankles.", "Dana cuts the zip tie.")[0][1]["read"] == [
              ("release", "Sarah, zip ties around her ankles")]
          and read("Dana leads Sarah, shackled at the ankles, to the table.") == [("have", "Sarah, shackles around her ankles")]
          and read("Sarah walks to the post in handcuffs.") == [("have", "Sarah, handcuffs behind her back")],
          [read(b) for b in ("Dana puts Sarah in a belly chain, handcuffs and leg irons.", "Dana zip-ties Sarah's wrists, then her ankles.",
                             "Sarah walks to the post in handcuffs.")])


def test_undress_bound():
    print("\n=== clothes that cannot pass bound legs or wrists stop above the restraint, in the beat itself, and the restraint holds ===")
    mem = ("Sarah: a young woman with long blonde hair, in a white blouse, blue jeans, black panties and white sneakers.\n"
           "Dana: a woman with short black hair, in a red dress and black heels.\n"
           "Luis: a young man with short brown hair, in a grey t-shirt and jeans.")
    plan = lambda *b: P.plan("A cellar with a steel table, a wooden chair, a bed and a staircase.\n\n" + "\n\n".join(b), mem)
    beat = lambda s: s["prompt"].split("\n\n")[-2]
    lying = "Sarah lies on the bed, her ankles handcuffed together."
    shoes, warn = plan(lying, "Luis takes off her sneakers and her jeans.", "Sarah lies still.")
    dress = plan("Dana sits on the bed, her ankles handcuffed together.", "Dana takes off her dress.")[0][1]
    cuffs = plan("Sarah sits on the chair.", "Luis cuffs Sarah's ankles together and pulls her jeans off.")[0][1]
    knees = plan("Sarah lies on the bed, her knees tied together with rope.", "Luis pulls her jeans off.")[0][1]
    down = [plan(w, "Luis pulls her jeans down to her ankles.")[0][1] for w in (lying, "Sarah lies on the bed, her knees tied together.")]
    check("bound ankles keep the jeans on, pulled down to just above the cuffs in the beat, the notes and the next description; "
          "shoes slip off with her feet staying together; a dress goes over her head; cuffs put on first in the same beat count; a "
          "pull down past the cuffs or tied knees stops just above them",
          beat(shoes[1]) == "Luis takes off blonde-haired Sarah's sneakers and pulls blonde-haired Sarah's jeans down to just above the "
                            "handcuffs on her ankles."
          and "slips blonde-haired Sarah's white sneakers off, her feet staying together, then pulls blonde-haired Sarah's blue "
              "jeans down to just above the handcuffs on her ankles." in notes(shoes[1])
          and not any(w in s["prompt"] for s in shoes for w in ("one foot at a time", "off over her feet"))
          and "Her blue jeans are pulled down to just above the handcuffs on her ankles." in shoes[2]["prompt"]
          and any("Sarah's blue jeans cannot come off past her bound legs" in w for w in warn)
          and "Dana pulls her red dress up over her head and off her arms." in notes(dress)
          and beat(cuffs) == "Luis cuffs Sarah's ankles together and pulls blonde-haired Sarah's jeans down to just above the handcuffs on "
                             "her ankles."
          and beat(knees) == "Luis pulls blonde-haired Sarah's jeans down to just above the rope around her knees."
          and [beat(s) for s in down] == ["Luis pulls blonde-haired Sarah's jeans down to just above the handcuffs on her ankles.",
                                          "Luis pulls blonde-haired Sarah's jeans down to just above the rope around her knees."],
          (beat(shoes[1]), notes(shoes[1]), beat(cuffs), beat(knees), notes(dress), warn, [beat(s) for s in down]))
    naked = plan(lying, "Luis strips her naked.")[0]
    clothes = plan(lying, "Luis removes her clothes.")[0]
    ripped = plan(lying, "Luis rips her clothes off.")[0]
    wrists = plan("Sarah sits on the chair, her wrists handcuffed behind her back.", "Luis takes off her blouse.")[0][1]
    free = plan("Sarah sits on the chair.", "Luis strips her naked.")[0]
    check("stripping her naked or removing her clothes says in the beat what stays at her cuffed ankles, ripping takes the shoes off "
          "whole, a top stops above her breasts when her wrists are cuffed, and nothing changes when she is free",
          beat(naked[-1]) == "Luis strips blonde-haired Sarah naked, blonde-haired Sarah's blue jeans and black panties pulled down to just "
                             "above the handcuffs on blonde-haired Sarah's ankles, where they stay."
          and beat(clothes[-1]) == "Luis removes blonde-haired Sarah's clothes, blonde-haired Sarah's blue jeans and black panties pulled "
                                   "down to just above the handcuffs on blonde-haired Sarah's ankles, where they stay."
          and "remove: Sarah, white sneakers (" in ripped[1]["line"] and "torn off: Sarah, white blouse, blue jeans (" in ripped[1]["line"]
          and beat(wrists) == "Luis pulls blonde-haired Sarah's blouse up above her breasts, with Sarah's hands still locked together behind "
                              "her back."
          and beat(free[-1]) == "Luis strips blonde-haired Sarah naked."
          and "pulls them down her legs and off over her feet" in notes(free[-1]),
          ([beat(s) for s in naked + clothes], ripped[1]["line"], beat(wrists), [notes(s) for s in free]))
    standing = "Sarah stands by the table, her wrists handcuffed behind her back."
    off, own, shoes = (plan(standing, b)[0][1] for b in ("Luis pulls her jeans down and off.", "Sarah wriggles out of her jeans.",
                                                         "Sarah takes off her sneakers."))
    front = plan("Sarah stands by the table, her hands cuffed in front of her.", "Luis pulls down her jeans.")[0][1]
    lying = plan("Sarah lies on the bed, her wrists handcuffed behind her back.", "Luis pulls her jeans off.")[0][1]
    back = plan(standing, "Luis pulls her jeans down to her knees.", "Luis pulls them back up.")[0][2]
    check("with her wrists cuffed, every change to her clothes says in the beat that her hands stay locked; standing, she steps out "
          "of her jeans one foot after the other with her hands still locked; on her own she pushes them down and her shoes off "
          "without her hands; lying down she steps out of nothing; 'them' at the start of the next beat is the jeans",
          beat(off) == "Luis pulls blonde-haired Sarah's jeans down and off, with Sarah's hands still locked together behind her back."
          and "opens blonde-haired Sarah's blue jeans at the waist and pulls them down her legs to her ankles, and blonde-haired Sarah "
              "steps out of them one foot at a time, with blonde-haired Sarah's hands still locked together behind her back." in notes(off)
          and beat(own) == "Sarah wriggles out of her jeans, with her hands still locked together behind her back."
          and "Sarah pushes her blue jeans down over her hips and wriggles them down her legs to her ankles, and steps out of them one "
              "foot at a time, with her hands still locked together behind her back." in notes(own)
          and "Sarah pushes her white sneakers off with her feet, one heel against the other, with her hands still locked together "
              "behind her back." in notes(shoes)
          and beat(front) == "Luis pulls down blonde-haired Sarah's jeans, with Sarah's hands still locked together in front."
          and "pulls them down her legs and off over her feet, with blonde-haired Sarah's hands still locked together behind her back."
          in notes(lying) and "steps out" not in notes(lying)
          and "pulled back: Sarah: blue jeans." in back["line"]
          and beat(back) == "Luis pulls them back up, with Sarah's hands still locked together behind her back.",
          (beat(off), notes(off), beat(own), notes(own), notes(shoes), beat(front), notes(lying), back["line"], beat(back)))


def test_belt():
    print("\n=== a chastity belt stays on once it is on, with its picture in every shot, and nothing comes off from under it ===")
    mem = ("Sarah <Picture 1>: a young woman with long blonde hair, in a white blouse and black panties.\n"
           "Dana: a woman with short black hair, in a black jacket and grey jeans.")
    plan = lambda *b, m=mem: P.plan("A bedroom with a bed.\n\n" + "\n\n".join(b), m, pictures=[IMG, IMG, IMG])
    beat = lambda s: s["prompt"].split("\n\n")[-2]
    sheet = lambda s: next(line for line in s["prompt"].split("\n") if line.startswith("Sarah <Picture"))
    on = plan("Dana takes off Sarah's panties.", "Dana locks the chastity belt <Picture 3> on Sarah.", "Sarah lies on the bed.",
              "Dana strips Sarah naked.")[0]
    check("the chastity belt goes on and stays on: her description says so before every later beat, its picture goes in from the "
          "shot it goes on in, nothing says she is naked without it, and stripping her leaves it locked on",
          "goes on: Sarah: a chastity belt locked around her waist" in on[1]["line"]
          and "Her chastity belt <Picture 2> goes on her in this shot." in sheet(on[1]) and 2 in on[1]["refs"]
          and "naked from the waist down the whole shot, apart from the chastity belt locked around her waist, its shield between her "
              "legs over her crotch once it is on." in notes(on[1])
          and all("Her chastity belt <Picture 2> is locked on her, its band around her waist and its shield between her legs over "
                  "her crotch." in sheet(s) and 2 in s["refs"] for s in on[2:])
          and "apart from the chastity belt locked around her waist" in notes(on[2])
          and "Sarah's chastity belt stays locked on her the whole time her white blouse comes off." in notes(on[3]),
          [(sheet(s), s["refs"], notes(s)) for s in on[1:]])
    told = plan("Dana locks a chastity belt on Sarah.", "Sarah walks to the bed.", m=mem + "\n\n<Picture 3> is a steel chastity belt.")[0]
    worn, warn = plan("Sarah <Picture 1>, locked in a chastity belt, kneels on the bed.", "Dana pulls off Sarah's panties.",
                      "Dana unlocks Sarah's chastity belt and takes it off.", "Dana pulls off Sarah's panties.")
    check("a picture named for the belt anywhere goes with it and her own picture never does, being locked in one is read, her "
          "panties stay on under it until it is unlocked, and unlocking it takes it off with no false warning",
          all("Her chastity belt <Picture 2>" in sheet(s) and 2 in s["refs"] for s in told)
          and "Her chastity belt is locked on her" in sheet(worn[0]) and worn[0]["refs"] == [0]
          and beat(worn[1]) == "Dana leaves Sarah's panties on under her chastity belt."
          and "comes off: Sarah: a chastity belt" in worn[2]["line"] and "clothes off: Sarah: black panties." in worn[3]["line"]
          and not any("was not read" in w for w in warn) and any("black panties cannot come off past her chastity belt" in w for w in warn),
          ([sheet(s) for s in told], [(s["line"], beat(s)) for s in worn], warn))


def test_apart():
    print("\n=== restraints on different spots stay separate unless a beat joins them ===")
    mem = ("Sarah: a young woman with long blonde hair, in a white blouse and black panties.\n"
           "Dana: a woman with short black hair, in a black jacket and grey jeans.")
    plan = lambda *b: P.plan("A bedroom.\n\n" + "\n\n".join(b), mem)[0]
    apart = "are separate restraints"
    cuffed = plan("Dana cuffs Sarah's hands behind her back.", "Dana cuffs Sarah's ankles.", "Dana puts a crotch chain on Sarah.",
                  "Sarah stands still.")
    linked = plan("Dana cuffs Sarah's hands behind her back.", "Dana puts a crotch chain on Sarah and locks it to her handcuffs.",
                  "Dana ties Sarah's ankles with rope.")
    tied = plan("Dana ties Sarah's wrists behind her back.", "Dana ties Sarah's ankles together.",
                "Dana ties Sarah's wrists to her ankles.", "Sarah struggles.")
    check("handcuffs, ankle cuffs and a crotch chain are separate pieces with nothing joining them, one restraint alone gets no such "
          "line, and a link a beat makes, to her crotch chain or from her wrists to her ankles, is named as the one separate short chain "
          "or rope between them",
          apart not in notes(cuffed[0])
          and "The handcuffs on blonde-haired Sarah's wrists and the handcuffs on her ankles are separate restraints, each on its own "
              "with nothing joining it to another." in notes(cuffed[1])
          and all("The handcuffs on blonde-haired Sarah's wrists, the handcuffs on her ankles and her crotch chain are separate "
                  "restraints, each on its own with nothing joining it to another." in notes(s) for s in cuffed[2:])
          and "Blonde-haired Sarah's handcuffs and her crotch chain are separate restraints, and the only link between them is a "
              "separate short chain from the chain between her handcuffs to the back of her crotch chain." in notes(linked[1])
          and "Blonde-haired Sarah's handcuffs, the rope on her ankles and her crotch chain are separate restraints, and the only link "
              "between any of them is a separate short chain from the chain between her handcuffs to the back of her crotch chain."
          in notes(linked[2])
          and "each on its own with nothing joining it to another." in notes(tied[1])
          and all("The rope on blonde-haired Sarah's wrists and the rope on her ankles are separate restraints, and the only link between "
                  "them is a separate short rope from the rope around her wrists to the rope around her ankles." in notes(s)
                  for s in tied[2:]),
          [notes(s) for s in cuffed + linked + tied])
    joins = ("Dana chains Sarah's handcuffs to her ankle cuffs.", "Dana hogties Sarah.", "Her wrists are chained to her collar.",
             "a short chain between her wrists and her ankles")
    plain = ("Dana snaps the handcuffs onto Sarah's wrists.", "Dana cuffs Sarah's hands and pulls her to her feet.",
             "Dana cuffs Sarah's wrists, then moves to her ankles.", "Dana pulls Sarah's panties down her legs to her ankles.",
             "Dana puts a crotch chain on Sarah and locks it to her handcuffs.")
    check("wrists, ankles and neck joined to each other count as a link, and putting cuffs on, getting to her feet or moving on to her "
          "ankles do not", all(map(P._joined, joins)) and not any(map(P._joined, plain)),
          ([t for t in joins if not P._joined(t)], [t for t in plain if P._joined(t)]))


def test_links():
    print("\n=== every restraint keeps its own chain, and a link between two is a separate short chain of its own ===")
    mem = ("Sarah: a young woman with long blonde hair, in a white blouse and blue jeans.\n"
           "Dana: a woman with short black hair, in a black jacket and grey trousers.")
    plan = lambda *b: P.plan("A dim bedroom.\n\n" + "\n\n".join(b), mem)
    beat, sheet = (lambda s: s["prompt"].split("\n\n")[-2]), (lambda s: s["prompt"].split("\n\n")[1])
    lying = "Sarah lies face down on the bed, her wrists handcuffed behind her back and her ankles handcuffed together."
    link = "handcuffs on her ankles, locked to her handcuffs behind her back by a separate short chain"
    both = plan("Dana handcuffs Sarah's wrists behind her back.", "Dana handcuffs Sarah's ankles together.", "Dana puts a crotch chain on Sarah.",
                "Sarah kneels on the bed.")[0]
    check("wrist cuffs, ankle cuffs and a crotch chain each keep a chain of their own: rigid steel cuffs just above her feet with "
          "their own short chain that never stretches, the crotch chain under her crotch and all of it at her hips",
          all(t in notes(both[3]) for t in ("a pair of rigid steel cuffs is locked around them just above her feet, joined by their own "
                                             "short chain of a few links that never stretches, bends or gets longer",
                                             "two solid cuffs joined by their own short chain of a few links",
                                             "the center chain runs under her crotch, pulled tight over her vagina, and up between her "
                                             "buttocks, so all of it stays at her hips"))
          and "between her legs" not in both[3]["prompt"], notes(both[3]))
    tied, warn = plan(lying, "Dana padlocks Sarah's ankle cuffs to her handcuffs.", "Sarah struggles.",
                      "Dana unlocks the chain between Sarah's wrists and ankles.")
    check("lying face down in cuffs is not being locked to the bed; padlocking her ankle cuffs to her handcuffs joins the two cuffs' own "
          "chains by a separate short chain, said in the beat, the step, her description and every shot; unlocking it takes only it away",
          tied[0]["read"] == [("have", "Sarah, handcuffs behind her back; handcuffs on her ankles")]
          and tied[1]["read"] == [("hold", f"Sarah, {link}")]
          and beat(tied[1]) == "Dana padlocks Sarah's ankle cuffs to her handcuffs with a separate short chain."
          and "Black-haired Dana bends blonde-haired Sarah's knees and lifts her handcuffed ankles together up behind her toward her "
              "hands, then padlocks a "
              "separate short chain from the chain between her handcuffs to the chain between her ankle cuffs." in notes(tied[1])
          and "Her ankles are handcuffed together and held to her handcuffs behind her back by a separate short chain, her knees bent "
              "and her handcuffed ankles raised together behind her, close to her hands." in sheet(tied[2])
          and all("A separate short chain runs from the chain between her ankle cuffs to the chain between her handcuffs, so her knees "
                  "stay bent and her handcuffed ankles stay raised together behind her, close to her hands, the whole time." in notes(s)
                  and "and the only link between them is a separate short chain from the chain between her handcuffs to the chain "
                      "between her ankle cuffs." in notes(s) and "from the moment" not in notes(s) for s in tied[1:3])
          and beat(tied[2]) == "Sarah struggles, blonde-haired Sarah's ankles handcuffed together."
          and tied[3]["read"] == [("hold", "Sarah, handcuffs on her ankles")]
          and "Black-haired Dana unlocks the separate short chain that joins blonde-haired Sarah's handcuffs to the chain between her "
              "ankle cuffs and takes it away." in notes(tied[3]) and not warn,
          ([s["read"] for s in tied], [beat(s) for s in tied], [notes(s) for s in tied], warn))
    hog = plan(lying, "Dana hogties Sarah.", "Dana uncuffs Sarah's wrists.", "Sarah rolls over.")[0]
    rope = plan("Sarah lies on the floor, hogtied with rope.", "Dana unties the hogtie.")[0]
    cuffs = plan("Dana hogtie-cuffs Sarah, her wrists cuffed behind her back and padlocked to her ankle cuffs.")[0]
    check("'hogties' links the cuffs she has and the beat says how; uncuffing her wrists takes that chain off first and leaves her ankles "
          "cuffed; 'hogtied with rope' ties a separate short rope between her wrists and ankles that 'unties the hogtie' takes away; a "
          "later mention of her ankle cuffs keeps the link",
          hog[1]["read"] == [("hold", f"Sarah, {link}")]
          and beat(hog[1]) == "Dana hogties Sarah, blonde-haired Sarah's ankles held to her handcuffs behind her back by a separate short "
                              "chain."
          and "Black-haired Dana unlocks the separate short chain that joins blonde-haired Sarah's handcuffs to the chain between her ankle "
              "cuffs and takes it away, then unlocks the handcuffs with a small key and takes them off her wrists." in notes(hog[2])
          and hog[2]["line"].endswith("comes off: Sarah: handcuffs behind her back.") and hog[3]["line"].endswith(
              "on: Sarah: handcuffs on her ankles.")
          and rope[0]["read"] == [("have", "Sarah, rope around her wrists behind her back; rope around her ankles, tied to the rope "
                                           "around her wrists behind her back by a separate short rope")]
          and rope[1]["read"] == [("hold", "Sarah, rope around her ankles")]
          and "Black-haired Dana unties the separate short rope that joins the rope around blonde-haired Sarah's wrists to the rope around "
              "her ankles and takes it away." in notes(rope[1])
          and cuffs[-1]["read"] == [("hold", f"Sarah, {link}")],
          ([(s["read"], s["line"], beat(s)) for s in hog + rope + cuffs], [notes(s) for s in hog[2:3] + rope[1:]]))
    waist, hands = (plan("Sarah stands by the bed, her wrists handcuffed behind her back and a chain locked around her waist.", b)[0]
                    for b in ("Dana padlocks Sarah's handcuffs to the back of the waist chain.",
                              "Dana chains Sarah's cuffed hands to the back of her waist chain."))
    crotch = plan("Dana handcuffs Sarah's wrists behind her back.", "Dana puts a crotch chain on Sarah.",
                  "Dana padlocks Sarah's handcuffs to the back of her crotch chain.")[0]
    check("padlocking her handcuffs to the back of a waist chain or crotch chain is read, with no second set of chains on her wrists, "
          "and a separate short chain joins the chain between her handcuffs to the back of it",
          all(s[1]["read"] == [("hold", "Sarah, chains around her waist, chained to her handcuffs at the small of her back by a "
                                        "separate short chain")] for s in (waist, hands))
          and beat(waist[1]) == "Dana padlocks Sarah's handcuffs to the back of the waist chain with a separate short chain."
          and "Black-haired Dana padlocks a separate short chain from the chain between blonde-haired Sarah's handcuffs to the back of the "
              "chain around her waist." in notes(waist[1])
          and "with her wrists locked together behind her and held to the back of the chain around her waist by a separate short chain: "
              in notes(waist[1])
          and crotch[2]["read"][0][1].endswith("chained to her handcuffs at the small of her back by a separate short chain")
          and "Black-haired Dana padlocks a separate short chain from the chain between blonde-haired Sarah's handcuffs to the back of "
              "her crotch chain." in notes(crotch[2]),
          ([(s[-1]["read"], beat(s[-1])) for s in (waist, hands, crotch)], [notes(s[-1]) for s in (waist, crotch)]))
    other, _ = plan("Sarah lies face down on the floor, her wrists handcuffed behind her back.",
                    "Dana cuffs her ankles to her wrists with another pair of handcuffs.")
    check("a hogtie on free ankles cuffs them together in one shot with her legs flat, then lifts the cuffed ankles together and "
          "chains them in the next, and the tool already used is not named again",
          "beat 2, part 1 of 2" in other[1]["line"] and "beat 2, part 2 of 2" in other[2]["line"]
          and beat(other[1]) == "Dana cuffs Sarah's ankles together." and "lifts" not in notes(other[1])
          and "closes one cuff around each ankle with a click, the short chain between them, then lets go and stands back up"
              in notes(other[1])
          and beat(other[2]) == "Dana cuffs blonde-haired Sarah's ankles to her wrists."
          and "bends blonde-haired Sarah's knees and lifts her handcuffed ankles together up behind her toward her hands"
              in notes(other[2]),
          [(beat(s), s["line"][:60]) for s in other])


def test_render():
    print("\n=== the shot chain ===")
    prompt = "A room.\n\nDana waits.\n\nSarah walks in.\n\n\"Sit,\" Dana says.\n\ncut\n\nSarah drives."
    out, calls = render(prompt, dwpose=True, ref_image_1=IMG)
    video, audio, info = out[0], out[1], out[2]
    talk = [c["heard"] for c in calls["sample"] if c["shot"] == 3]
    check("one picture pass per beat, plus an audio-only pass for each wordless beat and one that gives the rest of a beat with a "
          "line the room's own sound",
          [c["shot"] for c in calls["sample"] if not c["audio_only"]] == [1, 2, 3, 4]
          and [c["shot"] for c in calls["sample"] if c["audio_only"]] == [1, 2, 3, 4] and abs(sum(talk) - 1) < 0.02, calls["sample"])
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
    out, calls = render("A room.\n\nDana and Sarah sit.\n\nDana looks away.\n\nSarah looks at Dana.", dwpose=True, counts=(2, 1, 2),
                        blind=True, ref_image_1=IMG)
    check("on a model not trained on pictures every shot still gets the picture, and no earlier frame goes in",
          [len(c["refs"]) for c in calls["build"]] == [1, 1, 1] and all(c["memory"] is None for c in calls["build"])
          and "no earlier frame of the scene is fed back in" in out[2], ([len(c["refs"]) for c in calls["build"]], out[2]))
    out, calls = render("A room.\n\nDana tapes Sarah's mouth shut.\n\n\"Help!\" Sarah screams into the tape.")
    muffled = [c for c in calls["sample"] if c["shot"] == 2]
    check("a taped scream is rendered silent, given sound in its own short window, and the rest of the shot gets the room's own sound",
          len(muffled) == 3 and muffled[0]["heard"] == 0 and all(c["audio_only"] for c in muffled[1:]) and 0 < muffled[1]["heard"] < 0.5
          and abs(muffled[1]["heard"] + muffled[2]["heard"] - 1) < 0.02, muffled)
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
    test_tags()
    test_reading()
    test_speech()
    test_hands()
    test_bodies()
    test_pronouns()
    test_extras()
    test_names()
    test_cast()
    test_clothes()
    test_layers()
    test_pictures()
    test_lengths()
    test_pace()
    test_models()
    test_chain()
    test_ankles()
    test_undress_bound()
    test_belt()
    test_apart()
    test_links()
    test_render()
    print("\nRESULT: " + (f"{len(_fails)} FAILURE(S): " + "; ".join(_fails) if _fails else "ALL PASSED"))


if __name__ == "__main__":
    main()
