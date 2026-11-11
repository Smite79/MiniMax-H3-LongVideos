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

MIXED = lambda text, *things: [t for t in re.split(r"(?<=[.!?])\s+", text) if re.search(r"\b(?:hands?|wrists?|cuffs|handcuffs)\b", t)
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
          in read[1]["line"] and "Sarah: tape over her mouth. Blonde-haired Sarah: handcuffs behind her back." in notes(read[2]),
          read[1]["line"])
    over, _ = P.plan("An office.\n\nDana and Sarah stand.\n\nDana tapes her mouth shut.\nhold: Sarah, a red gag", MEM)
    check("a tag line in a beat replaces what the beat would be read as, a bare item placed where it goes",
          "goes on: Sarah: a red gag in her mouth." in over[1]["line"]
          and "read from" not in over[1]["line"], over[1]["line"])
    wrong, warn = P.plan("A room.\n\nDana waits.\nhold: Dama, rope around her wrists\n\nDana sits.", MEM)
    check("a tag naming someone without a character line is applied as written and flagged",
          any("names Dama, who has no line in character_memory" in w for w in warn) and "Dama" in wrong[0]["line"], warn)
    early, _ = P.plan("A cell.\n\nSarah waits.\n\nDana walks in.", MEM + "\nhold: Sarah, handcuffs behind her back")
    check("a hold line in character_memory is on from the first shot",
          "Sarah: handcuffs behind her back." in notes(early[0]) and "goes on" not in early[0]["line"], early[0]["line"])
    inline, _ = P.plan("A room.\n\nSarah sits. Dana walks in. hold: Sarah, a red collar\n\nwear: Dana, a red dress\n\nDana waits.", MEM)
    check("tags written after a sentence on the same line, or in a paragraph of their own, belong to that beat",
          "goes on: Sarah: a red collar around her neck." in inline[0]["line"] and "puts on: Dana: a red dress." in inline[0]["line"]
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
    close = said('Dana pulls Sarah close and whispers, "Don\'t make a sound. Nobody can hear you."')
    split = said('"Did you really think you could run?" Dana asks. "Nobody is coming. Stop struggling."')
    check("a long line stays with the one who speaks it, not the one named before the speech verb, and a line split by its tag "
          "stays with its speaker",
          "Black-haired Dana pulls Sarah close and whispers: <d>Don't make a sound.</d> Dana continues: <d>Nobody can hear you.</d>"
          in close["prompt"] and [w for w, _, _ in split["said"]] == ["Dana", "Dana"]
          and [w for w, _, _ in said('Sarah turns to Dana and asks, "Why? I trusted you."')["said"]] == ["Sarah"]
          and [w for w, _, _ in said('Sarah says to Dana: "Listen to me. Please."')["said"]] == ["Sarah"]
          and [w for w, _, _ in said('Dana grabs Sarah\'s wrists. "Let go of me! You\'re hurting me!"')["said"]] == ["Sarah"]
          and [w for w, _, _ in said('Sarah asks, "Where were you?" Dana says, "At work."')["said"]] == ["Sarah", "Dana"]
          and [w for w, _, _ in said('"Hi," Sarah says, "how are you?"')["said"]] == ["Sarah", "Sarah"], (close["prompt"], split["said"]))
    turns = P.plan("A room.\n\nDana and Sarah stand by the bed.\n\nSarah faces Dana. \"I found the letters. I know about her. Don't lie to "
                   "me. Five years of lies. I want the truth.\" Dana sighs. \"It meant nothing. She meant nothing. I made a mistake. I was "
                   "lonely. Please forgive me.\"", mem)[0]
    spoken = [(w, "sighs: <d>" in sh["prompt"], "Dana sighs." in sh["prompt"]) for sh in turns for w, _, _ in sh["said"]]
    check("a sentence of reaction after a quote that ends in a full stop is not its speech tag, even when the beat is split into shots, "
          "so each of two long lines stays with its own speaker",
          spoken == [("Sarah", False, True), ("Dana", False, False)] and "Blonde-haired Sarah says: <d>I found the letters.</d>"
          in turns[-2]["prompt"], (spoken, [sh["prompt"][-900:] for sh in turns[-2:]]))
    asked = P.plan("A room.\n\nDana and Sarah stand by the bed.\n\nSarah asks, \"Where were you last night?\"\n\n"
                   "\"At work. I told you.\"", mem)[0]
    check("a quote on a paragraph of its own, straight after a question, is the answer of the other person",
          [w for sh in asked for w, _, _ in sh["said"]] == ["Sarah", "Dana"], [sh["said"] for sh in asked])
    label = said('SARAH: "Where were you? I waited for hours."'), said('Dana (whispering): "Hold still. Please."')
    check("a speaker's name in capitals, or with a direction in brackets, before a colon and a quote is read as the speaker",
          label[0]["said"][0][0] == "Sarah" and label[1]["said"][0][0] == "Dana"
          and "Black-haired Dana says (whispering): <d>Hold still.</d>" in label[1]["prompt"], (label[0]["said"], label[1]["prompt"]))
    quiet = said("Dana walks to the window.")
    check("the sound is said as what is heard, never as music that is not there",
          "All the sound comes from the scene itself" in notes(quiet) and "music" not in quiet["prompt"], notes(quiet))
    scream, radio, band = said("Sarah screams."), said("A radio plays music."), said("Dana puts a rubber band around Sarah's wrists.")
    check("voices are kept out of a shot with no lines unless it calls for a voice, music or a tone of its own, and a scream is not "
          "told to keep its mouth closed",
          quiet["mute"] and band["mute"] and "All the sound comes from the scene itself" in notes(band)
          and not scream["mute"] and "Every sound anyone makes is wordless." in notes(scream) and "lips closed" not in notes(scream)
          and not radio["mute"] and not said("A dog barks outside.")["mute"] and long["mute"] and said('"Get out!" Sarah screams.')["mute"]
          and not P.plan("A room.\n\nDana waits.", mem, silence=False)[0][0]["mute"]
          and P.plan('A room.\n\n"Wait," Dana says.', mem, silence=False)[0][0]["mute"],
          [(s["mute"], notes(s)) for s in (quiet, scream, radio, band)])
    bound = said("Dana snaps handcuffs on Sarah's wrists behind her back. Dana writes \"SLAVE\" across Sarah's stomach.")
    taped = said("Dana presses duct tape over Sarah's mouth.", "Sarah tries to scream \"Help!\" through the tape. \"Quiet,\" Dana says.")
    song = said("A radio plays music. Sarah thinks, \"I have to get out of here.\"")
    freed = said("Dana cuffs Sarah's hands behind her back.", "Dana unlocks the handcuffs while Sarah calls for an ambulance.")
    towel = said("Dana gags Sarah with a dish towel and asks Sarah to stay quiet.")
    told, heard = (lambda x: x["sound"].split("\n\n")[0]), (lambda x: x["sound"].split("\n\n")[1])
    begs, cries = said("Sarah begs Dana to let her go."), said("Sarah whimpers and pulls at the bedpost while Dana watches.")
    check("a beat whose only sound is talk with no words given keeps every voice out and everyone silent, while a whimper stays "
          "hers alone and the one watching stays silent",
          begs["mute"] and told(begs).endswith("A room. Sarah and Dana stay silent.") and not cries["mute"]
          and told(cries).endswith("Sarah whimpers and pulls at the bedpost while Dana watches. Dana stays silent."),
          [(x["mute"], told(x)) for x in (begs, cries)])
    chores = [[re.sub(r"^[\w-]+-haired ", "", t) for t in re.split(r"(?<=[.!?])\s+", P.plan("A kitchen.\n\nSarah walks in.\n\n" + b, mem)
                                                                                         [0][1]["prompt"]) if " does the " in t]
              for b in ("Sarah is washing the dishes while Dana watches.", "She picks up a plate and washes it in the sink.",
                        "Dana orders Sarah to do the dishes.", "Dana chops onions while Sarah folds the laundry.",
                        "Sarah dries her hands on a towel.", "Sarah folds her arms.", "Sarah peels off her top.", "Sarah waves at Dana.",
                        "Dana cuffs Sarah's hands behind her back. Sarah wipes the table.")]
    scrub = "Sarah does the washing with both hands, holding each piece in one hand and scrubbing it with the other."
    check("someone with free hands doing a two-handed chore, however it is worded, is shown how both hands do it, and a body part, an "
          "undressing or bound hands get nothing",
          chores == [[scrub], [scrub], [scrub], ["Sarah does the folding with both hands, taking each piece in both hands to fold it.",
                                            "Dana does the chopping with both hands, holding the food steady with one hand and the knife "
                                            "in the other."], [], [], [], [], []], chores)
    kiss, moan = said('Dana kisses Sarah\'s neck. "I want you," Sarah says.'), said('Dana spanks Sarah. Sarah moans. "Thank you," she '
                                                                                   'whispers.')
    check("the sound around a spoken line keeps its speaker silent too, so nothing is said in front of the line, while a moan the "
          "beat gives stays hers",
          kiss["cues"] and re.search(r"(?:Sarah and Dana|Dana and Sarah) stay silent\.$", told(kiss))
          and moan["cues"] and told(moan).endswith("Sarah moans. Dana stays silent."), [told(kiss), told(moan)])
    speech = re.compile(r"\b(?:speaks?|voices?|words?|whispers?|talk\w*|dialogue)\b", re.I)
    check("a shot's sound is written apart from its picture, as the place, what happens, who stays silent, the restraints' own "
          "sounds and the gag's muffled ones, with no word of speech in it and no character sheet, note, spoken line, thought or "
          "written word to be read out",
          told(bound).startswith("integrated_multimodal_description: [Shot 1] A room. ") and "snaps handcuffs on" in told(bound)
          and "writes across" in told(bound) and "SLAVE" in bound["prompt"] and "SLAVE" not in bound["sound"]
          and "tank top" in bound["prompt"] and "tank top" not in bound["sound"] and "stays on" not in bound["sound"]
          and "The handcuffs ratchet shut with sharp metallic clicks and the metal restraints on Sarah clink and rattle with every "
              "movement." in heard(bound) and heard(bound).startswith("overall_soundscape: The quiet room tone of the place.")
          and told(bound).endswith(" stay silent.") and bound["sound"].endswith("\n\nnon_diegetic_music: N/A")
          and not any(speech.search(x["sound"]) for x in (bound, taped, song))
          and "Sarah lets out muffled, closed-mouth mmmph sounds through the duct tape." in heard(taped)
          and "Help" not in taped["sound"] and "Quiet" not in taped["sound"] and "<d>Quiet</d>" in taped["prompt"]
          and "tries to let out a muffled, closed-mouth mmmph through the duct tape" in taped["prompt"]
          and "with the music playing there." in heard(song)
          and "get out of here" not in song["sound"]
          and "thinks, silently and with closed lips: I have to get out of here." in song["prompt"]
          and not song["cues"] and not song["said"], [bound["sound"], taped["sound"], song["sound"], taped["prompt"][-400:]])
    check("talk with no words given is cut out of the sound, leaving what is done in the same sentence, and a towel gag tied on "
          "does not creak like rope",
          "A room. Dana unlocks the handcuffs. " in told(freed) and "ambulance" not in freed["sound"]
          and "The handcuffs click open." in heard(freed) and "rattle" not in heard(freed)
          and "gags Sarah with a dish towel." in told(towel)
          and "asks" not in towel["sound"] and "rope" not in towel["sound"], [freed["sound"], towel["sound"]])


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
          and "Black-haired Dana steps behind blonde-haired Sarah, takes both of her wrists and draws her arms back behind her, then "
              "closes one cuff around blonde-haired Sarah's left wrist with a click, then the other around her right wrist, there at "
              "the small of her back, where her body hides her hands from the front." in notes(shots[1])
          and "both arms run at their full length from her shoulders down behind her, and her hands rest together at the small of her "
              "back. From the front, both arms angle back from her shoulders and her body hides her hands; seen from the side or from "
              "behind, both arms and both hands are in plain view" in notes(shots[2])
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


def test_sex():
    print("\n=== penetration lines up with the body, and cuffed hands stay behind her through sex ===")
    mf = (["Sarah", "Marcus"], {"Sarah": {"sex": "f"}, "Marcus": {"sex": "m"}})
    ff = (["Dana", "Sarah"], {"Dana": {"sex": "f"}, "Sarah": {"sex": "f"}})
    mm = (["Leo", "Max"], {"Leo": {"sex": "m"}, "Max": {"sex": "m"}})
    trio = (["Sarah", "Marcus", "Dana"], {"Sarah": {"sex": "f"}, "Marcus": {"sex": "m"}, "Dana": {"sex": "f"}})
    read = lambda who, t: P._coupling(t, *who, {}, t)
    sm = lambda tool="penis", hole="vagina", way=None: [("Marcus", "Sarah", tool, hole, way)]
    check("penetration is read however it is worded: who, into whom, with what, into which opening and in what position, and "
          "nothing else is",
          read(mf, "Marcus fucks Sarah from behind.") == sm(way="behind") and read(mf, "He fucks her hard against the wall.") == sm(
              way="standing") and read(mf, "Marcus fucks Sarah against the wall.") == sm(way="standing")
          and read(mf, "Marcus pounds her doggy style.") == sm(way="bent") and read(mf, "His cock slides in and out of her pussy.") == sm()
          and read(mf, "She takes his cock deep in her ass.") == sm(hole="anus") and read(mf, "Sarah lowers herself onto his cock.") == sm(
              way="astride") and read(mf, "Sarah climbs on top of Marcus and rides him.") == sm(way="astride")
          and read(mf, "Marcus lies on top of her and fucks her.") == sm(way="top") and read(mf, "Sarah is being fucked by Marcus.") == sm()
          and read(mf, "Sarah fucks Marcus.") == sm() and read(mf, "Marcus slides two fingers inside her.") == sm("fingers")
          and read(mf, "Marcus pushes the butt plug into her ass.") == sm("butt plug", "anus")
          and read(ff, "Dana straps on a strap-on and fucks Sarah.") == [("Dana", "Sarah", "strap-on", "vagina", None)]
          and read(mm, "Leo fucks Max from behind.") == [("Leo", "Max", "penis", "anus", "behind")]
          and read(trio, "Dana watches as Marcus fucks her.") == sm() and read(trio, "Dana pushes a dildo into Sarah's ass.") == [
              ("Dana", "Sarah", "dildo", "anus", None)]
          and read(mf, "Sarah pushes the vibrator inside herself.") == [("Sarah", "Sarah", "vibrator", "vagina", None)]
          and not any(read(mf, t) for t in ("Marcus fucks her mouth.", "Marcus fingers her clit.", "Marcus takes her hand.",
                                            "Marcus enters her room.", "He pushes her into the wall.", "She sits on his lap.",
                                            "He runs his fingers into her hair.", "Marcus puts the dildo in the drawer.",
                                            "Marcus rubs his cock against her pussy.")) and not read(ff, "Dana fucks her."),
          [read(mf, t) for t in ("She takes his cock deep in her ass.", "Sarah is being fucked by Marcus.")])
    mem = ("Sarah: a young woman with long blonde hair, naked.\nMarcus: a man with short dark hair, naked.\n"
           "hold: Sarah, handcuffs behind her back")
    shots, warn = P.plan("A kitchen with a long counter.\n\n" + "\n\n".join([
        "Marcus bends Sarah over the counter.", "Marcus fucks Sarah from behind.", "Marcus thrusts harder.",
        "Marcus pulls out, and Sarah stands up.", "Sarah turns to face Marcus.", "Sarah strokes Marcus's cock."]), mem)
    bent = ("bent forward, both her arms lie along her back at their full length, and her hands stay locked together on top of the "
            "small of her back, in plain view from above and from behind.")
    put = ("Dark-haired Marcus is right behind blonde-haired Sarah, facing the same way she faces, his hips pressed against her "
           "buttocks. Dark-haired Marcus's penis goes into blonde-haired Sarah's vagina at the center of her crotch, between her thighs: "
           "it points into her toward her head, level, in line with her body, and slides in and out along that level line.")
    rise = "she rises with both arms staying behind her the whole way, and then her shoulders are drawn back"
    check("bent over, her cuffed hands lie on top of the small of her back; sex from behind places him behind her with the penis "
          "going into her along her body, carried while he keeps thrusting and gone once he pulls out; standing up again her arms "
          "stay behind her the whole way, and a beat that needs her cuffed hands is flagged",
          all(bent in notes(s) for s in shots[:3]) and put in notes(shots[1]) and put in notes(shots[2])
          and all("penis" not in notes(s) for s in shots[3:]) and rise in notes(shots[3]) and rise not in notes(shots[4])
          and any("\"Sarah strokes Marcus's cock.\" needs Sarah's hands" in w for w in warn), [notes(s) for s in shots[1:4]])
    mem = "Emma: a woman with short red hair, naked.\nJake: a man with a beard, naked."
    shots = P.plan("A bedroom with a bed.\n\n" + "\n\n".join(["Jake lays Emma on her back on the bed and spreads her legs.",
                                                            "Jake pushes a dildo into Emma's pussy.", "Emma climbs on top of Jake "
                                                            "and rides him.", "Emma moans."]), mem)[0]
    check("spreading her legs spreads hers, not his; with her lying down a toy lies level between her thighs, in line with her body; "
          "riding sits her astride him with his penis vertical, the dildo is no longer in her once he is, and she stays sitting "
          "afterwards instead of lying",
          "Red-haired Emma stays lying on her back on the bed with her legs spread apart for the whole shot" in notes(shots[1])
          and "Jake stays standing" not in " ".join(map(notes, shots))
          and "Jake holds the dildo by its base, his hand between red-haired Emma's thighs, at the center of her crotch: the dildo is "
              "level, in line with her body, its tip pointing into her vagina toward her head and its base pointing away from her head, "
              "and it slides in and out along that level line." in notes(shots[1])
          and "Red-haired Emma sits astride Jake's hips, facing him, with one knee on each side of him. Jake's penis goes into "
              "red-haired Emma's vagina from below, at the center of her crotch, between her thighs: it points straight up into her, "
              "vertical, parallel to her spine, and slides up into her and back down along that vertical line." in notes(shots[2])
          and not any("The dildo is still" in notes(s) for s in shots[2:])
          and "lying down" not in notes(shots[2]) and "Red-haired Emma stays sitting for the whole shot" in notes(shots[3]),
          [notes(s) for s in shots])
    mem = ("Lena: a woman with long blonde hair, naked.\nDana: a woman with short black hair, naked.\n"
           "Marcus: a man with short dark hair, naked.")
    aim = lambda *beats: [notes(s) for s in P.plan("A kitchen with a counter and a bed.\n\n" + "\n\n".join(beats), mem)[0]]
    up = "vertical, parallel to her thighs"
    level = "level, in line with her body"
    toy = ("Black-haired Dana holds the vibrator by its base, her hand low between blonde-haired Lena's thighs, below her crotch: the "
           "vibrator is vertical, parallel to her thighs, its top end inside her vagina and its base pointing straight down at the floor, "
           "and it slides up into her and back down along that vertical line.")
    standing = [aim("Lena stands at the counter.", "Dana pushes a vibrator into Lena's pussy.", "Dana works it in and out faster."),
                aim("Dana slides two fingers inside Lena."), aim("Lena stands by the counter and pushes the dildo inside herself."),
                aim("Marcus fucks Lena from behind at the counter."), aim("Marcus fucks Lena against the wall."),
                aim("Lena kneels.", "Dana pushes the dildo into Lena's pussy.")]
    flat = [aim("Marcus pounds Lena doggy style.", "He thrusts harder."),
            aim("Lena lies on her back on the bed.", "Dana slides the dildo into Lena's pussy."),
            aim("Lena gets on all fours on the bed.", "Dana pushes the butt plug into Lena's ass."),
            aim("Marcus spoons Lena and slides into her.")]
    check("penetration lines up with her body: standing or kneeling it is vertical, parallel to her thighs, from below and straight "
          "up into her, with a toy held by its base low between her thighs and its base pointing at the floor; lying, bent over, on all "
          "fours or spooned it is level, along her body toward her head; doggy stays level while he keeps thrusting",
          toy in standing[0][1] and toy in standing[0][2] and all(up in s[-1] and level not in s[-1] for s in standing)
          and all(level in s[-1] and up not in s[-1] for s in flat) and level in flat[0][0]
          and "palm up, and her fingers point straight up into her vagina" in standing[1][0]
          and "Lena holds the dildo by its base, her hand low between her thighs, below her crotch" in standing[2][0]
          and "from below" in standing[3][0] and "stands facing blonde-haired Lena" in standing[4][0],
          [s[-1] for s in standing + flat])
    lv = (["Lena", "Victor"], {"Lena": {"sex": "f"}, "Victor": {"sex": "m"}})
    near = lambda who, t: [(y, x) for y, x, k in P._close(t, *who) if k in ("behind", "side")]
    check("someone close behind someone else is read however it is worded, and nothing else is",
          all(near(lv, t) == [("Victor", "Lena")] for t in (
              "Victor walks up behind Lena and wraps his arms around her waist.", "Lena stands at the counter. Victor stands close "
              "behind her.", "Victor gropes Lena's breasts from behind.", "Lena is grabbed from behind by Victor.",
              "Victor spoons Lena on the bed.", "Victor presses his chest against Lena's back.", "Lena leans back against Victor.",
              "Victor kneels behind Lena."))
          and near(ff, "Dana hugs Sarah from behind.") == [("Dana", "Sarah")]
          and not any(near(lv, t) for t in ("Lena stands with her hands behind her back.", "Victor stands behind Lena's chair.",
                                            "Lena walks in and the door closes behind her.", "Victor holds Lena tight against his chest.",
                                            "Victor looks at Lena from behind the counter.", "Lena leans back against the counter.",
                                            "Victor cuffs Lena's wrists behind her back.")),
          [near(lv, t) for t in ("Lena is grabbed from behind by Victor.", "Victor stands behind Lena's chair.")])
    mem = ("Lena: a woman with long blonde hair, in a black latex crop top and black latex shorts.\nVictor: a man with short dark hair, "
           "in a black shirt and black trousers.\nhold: Lena, handcuffs behind her back")
    shots, warn = P.plan("A kitchen with a sink.\n\n" + "\n\n".join([
        "Lena stands at the sink and washes the dishes.", "Victor walks up behind Lena and wraps his arms around her waist.",
        "Victor holds Lena tight against his chest.", "Victor steps back and Lena turns around."]), mem)
    hands = "Blonde-haired Lena's locked hands stay behind her at the small of her back, between her back and dark-haired Victor's body."
    check("with someone close behind her, her cuffed hands stay between her back and that body, carried until they part; from the "
          "front her arms angle back; and a chore that needs her cuffed hands is flagged",
          hands in notes(shots[1]) and hands in notes(shots[2]) and not any("locked hands stay" in notes(s) for s in (shots[0], shots[3]))
          and "From the front, both arms angle back from her shoulders" in notes(shots[1])
          and any("\"Lena stands at the sink and washes the dishes.\" needs Lena's hands" in w for w in warn),
          ([notes(s) for s in shots], warn))
    kinds = lambda t: [(y, k) for y, _, k in P._close(t, *lv)]
    check("someone face to face with her, hugging her, or her back against a wall or furniture is read, and facing the wall, "
          "kissing her neck or putting something against a wall is not",
          kinds("Victor wraps his arms around Lena's waist.") == [("Victor", "hug")] and kinds("Victor pulls Lena close.") == [
              ("Victor", "hug")] and kinds("Victor kisses Lena.") == [("Victor", "front")] and kinds("Victor stands in front of "
                                                                                                     "Lena.") == [("Victor", "front")]
          and kinds("Victor pins Lena against the wall.") == [("the wall", "wall")] and kinds("Lena leans back against the counter.")
          == [("the counter", "wall")] and kinds("Lena stands with her back to the door.") == [("the door", "wall")]
          and not any(kinds(t) for t in ("Victor pushes Lena's face against the wall.", "Lena faces the wall.", "Victor kisses Lena's "
                                         "neck.", "Victor sets the box against the wall.", "Victor bends Lena over the table.")),
          [kinds(t) for t in ("Victor wraps his arms around Lena's waist.", "Victor pushes Lena's face against the wall.")])
    mem = ("Lena: a woman with long blonde hair, naked.\nMarcus: a man with short dark hair, naked.\n"
           "hold: Lena, handcuffs behind her back")
    run = lambda *beats, **kw: [notes(s) for s in P.plan("A bedroom with a bed.\n\n" + "\n\n".join(beats), mem, **kw)[0]]
    wall = run("Marcus pins Lena against the wall and fucks her.", "He thrusts harder.", "Marcus pulls out.", "Lena walks to the bed.")
    hug = run("Marcus hugs Lena.", "Marcus kisses her.", "Marcus steps back.")
    stay = "Blonde-haired Lena's locked hands stay behind her at the small of her back, "
    check("pinned to a wall, hugged or face to face, her cuffed hands stay between her back and the wall, on the far side of her body "
          "from him, under his forearms when he holds her; the wall carries until she moves away, the person until they part",
          stay + "between her back and the wall, on the far side of her body from dark-haired Marcus." in wall[0]
          and stay + "between her back and the wall" in wall[2] and "locked hands stay" not in wall[3]
          and stay + "on the far side of her body from dark-haired Marcus, under his forearms." in hug[0]
          and stay + "on the far side of her body from dark-haired Marcus, under his forearms." in hug[1]
          and "locked hands stay" not in hug[2],
          wall + hug)
    moved = run("Lena stands by the bed.", "Marcus pushes Lena to her knees.", "Lena lies down on the bed.", "Lena gets up.",
                "Lena kneels.")
    first = run("Lena stands by the window.", "Lena looks at Marcus.", first_frame=True)
    down, up_ = "she lowers herself with both arms staying behind her the whole way, and then ", "she rises with both arms staying"
    check("going down onto her knees, onto the bed or from standing in a later shot, her cuffed arms stay behind her the whole way, as "
          "they do getting up; a scene that opens on a picture with her already cuffed starts by drawing her arms back behind her",
          down in moved[1] and down + "lying on her back" in moved[2] and up_ in moved[3] and down in moved[4] and down not in moved[0]
          and "she draws both arms back behind her in the first moment, and then her shoulders are drawn back" in first[0]
          and "first moment" not in first[1] and not any("first moment" in t for t in run("Lena stands by the window.")),
          moved + first)
    talk = run("Lena stands in the middle of the room.", "Marcus walks in and stops in front of Lena.",
               "Marcus says, \"You will wait right here. Do you understand?\"", "Lena sits on the bed.", first_frame=True)
    still = ("Blonde-haired Lena stays standing upright for the whole shot, holding that position from the first frame to the last: her "
             "shoulders stay back and her cuffed hands stay locked behind her back.")
    check("standing cuffed while someone stops in front of her and talks to her, she stays standing upright with her cuffed hands "
          "locked behind her back, on the far side of her body from him, carried into the talking shot; a scene that never says she "
          "stands gets no standing note",
          still in talk[1] and still in talk[2] and stay + "on the far side of her body from dark-haired Marcus." in talk[2]
          and "standing upright" not in talk[3] and not any("standing upright" in t for t in run("Marcus looks at Lena.")), talk)
    tied = mem + "\nhold: Lena, a chain leash from her collar, padlocked to the counter\nDana: a woman with long brown hair, in a corset."
    pick = [notes(s) for s in P.plan("A kitchen with a counter. A large black dildo lies on the counter.\n\nDana walks up beside Lena."
                                     "\n\nDana picks up the large black dildo from the counter.", tied, first_frame=True)[0]]
    said = re.split(r"(?<=[.!?])\s+", pick[1])
    check("with her leash padlocked beside her, the slack chain runs off to that side and drops beside her, never straight down her "
          "front, its only fixed points are her collar and the counter, it is never named with her hands, and while someone else "
          "picks something up she holds the position of the opening frame with her cuffed hands locked behind her back",
          "from the ring at her throat it runs down across her chest to the side of her nearest the counter, drops beside her to the "
          "floor" in pick[1] and "Its only fixed points are the ring at her collar and the counter." in pick[1]
          and "straight down" not in pick[1] and not MIXED(pick[1], "leash", "chain")
          and "Blonde-haired Lena stays in the same position as in the frame this shot opens on for the whole shot, holding that "
              "position from the first frame to the last: her shoulders stay back and her cuffed hands stay locked behind her back."
          in pick[1] and min(k for k, t in enumerate(said) if "leash is one single" in t)
          < min(k for k, t in enumerate(said) if "arms stay pulled" in t), pick)
    alone = run("Lena pushes the vibrator inside herself.", "Lena moans.")[0]
    check("cuffed behind her back, a toy she pushes into herself is never put in her hands: only its base shows between her thighs",
          "The vibrator is inside her vagina, vertical, parallel to her thighs: only its base shows, between her thighs, pointing "
          "straight down at the floor." in alone and "holds the vibrator" not in alone, alone)
    mem = ("Lena: a woman with long blonde hair, in a black latex crop top and a black latex skirt.\nDana: a woman with short black "
           "hair, in a black corset.")
    cuff = lambda s, b: notes(P.plan(f"A kitchen with a sink.\n\n{s}\n\n{b}", mem)[0][1])
    back = ("steps behind blonde-haired Lena, takes both of her wrists and draws her arms back behind her, then ", ", there at the "
            "small of her back, where her body hides her hands from the front.")
    behind = [cuff(s, b) for s, b in (("Lena stands by the counter.", "Dana cuffs Lena's hands behind her back."),
                                      ("Lena stands by the counter.", "Dana ties Lena's wrists behind her back with rope."),
                                      ("Lena stands by the counter.", "Dana tapes Lena's wrists behind her back with duct tape."),
                                      ("Lena washes the dishes.", "Dana handcuffs Lena behind her back."),
                                      ("Lena kneels on the floor.", "Dana cuffs Lena's hands behind her back."))]
    others = [cuff(s, b) for s, b in (("Lena lies face down on the floor.", "Dana cuffs Lena's hands behind her back."),
                                      ("Lena stands by the counter.", "Lena cuffs her own hands behind her back."),
                                      ("Lena stands by the counter.", "Dana cuffs Lena's wrists in front of her."))]
    check("whoever cuffs or ties someone's hands behind her back steps behind her, draws her arms back and closes them there, at "
          "the small of her back; face down, on her own or in front nobody steps behind her",
          all(back[0] in t and back[1] in t for t in behind) and not any("steps behind" in t for t in others), behind + others)
    mem = "Emma: a woman with short red hair, naked.\nJake: a man with a beard, naked.\nhold: Emma, rope tying her wrists behind her back"
    shots = P.plan("A bedroom with a bed.\n\nEmma and Jake lie down on the bed.\n\nJake fucks Emma from behind as they lie on their "
                   "sides.", mem)[0]
    check("spooning lays her on her side with her bound hands at the small of her back, between her and him, never on her back",
          "lying on her side, both her arms run down behind her along her back" in notes(shots[1]) and "on her back," not in notes(shots[1])
          and "Jake lies on his side right behind red-haired Emma" in notes(shots[1])
          and "between her back and Jake's body." in notes(shots[1]), notes(shots[1]))


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
          and "The man steps behind blonde-haired Sarah, takes both of her wrists and draws her arms back behind her, then closes one cuff "
              "around her left wrist" in notes(cuffed[1])
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
    check("cuffed wrists are held behind her back from the moment they go on and while clothes come off, and no sentence names her "
          "hands or cuffs with the clothes",
          "Sarah's arms stay pulled behind her back from the moment the handcuffs are on" in notes(cuffed[1])
          and "Sarah's arms stay pulled behind her back the whole shot" in notes(cuffed[2]) + notes(cuffed[3])
          and not MIXED(notes(cuffed[2]), "leggings")
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
    check("a lifted top stays on, pulled up, while cuffed wrists stay locked behind her back, and no sentence names her hands with it",
          "By the last frame, Sarah's white tank top is pulled up above her breasts and stays there, still on her, leaving her breasts "
          "bare." in notes(shots[2]) and "Sarah's arms stay pulled behind her back the whole shot" in notes(shots[2])
          and not MIXED(notes(shots[2]), "tank top"), notes(shots[2]))
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
          and not MIXED(notes(shots[2]), "bottoms", "thong", "tank top"), (shots[2]["line"], warn))
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
          and shots[3]["silent"] and "Everyone stays silent, with their lips closed." in notes(shots[3]))
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
              "cut short, no piece of clothing appears, vanishes or changes by itself, and every reaction comes only after what causes it."
              in notes(stripped[1]),
          notes(stripped[1]))
    freed = plan("Dana gags Sarah with a ball gag.\n\nDana unbuckles the gag and takes it out of Sarah's mouth.")[0]
    check("a restraint goes on and comes off step by step, and a shot where nothing changes still runs at real-life speed",
          "Black-haired Dana steps behind blonde-haired Sarah, takes both of her wrists and draws her arms back behind her, then closes "
          "one cuff around blonde-haired Sarah's left wrist with a click, then the other around her right wrist, there at the small of "
          "her back, where her body hides her hands from the front." in notes(cuffed[1])
          and "Black-haired Dana pushes the ball into blonde-haired Sarah's mouth behind her teeth" in notes(freed[1])
          and "Black-haired Dana unbuckles the gag and takes it out of blonde-haired Sarah's mouth." in notes(freed[2])
          and "Everything happens at natural, real-life speed, in the order written, and every reaction comes only after what causes it."
              in notes(cuffed[0]), notes(freed[2]))
    check("the speed line names restraints only in a shot that has them",
          "short, no piece of clothing appears" in notes(stripped[1])
          and "short, no restraint or piece of clothing appears" in notes(freed[1]),
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
          and "Black-haired Dana brings blonde-haired Sarah's hands down between her legs and closes one cuff around blonde-haired "
              "Sarah's left wrist with a click, then the other around her right wrist." in notes(lap)
          and "hold: Sarah, chains around her wrists at her waist (" in waist["line"]
          and "brings blonde-haired Sarah's hands to her waist and wraps the chain around blonde-haired Sarah's wrists" in notes(waist)
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
          "One of the two women steps behind blonde-haired Sarah, takes both of her wrists and draws her arms back behind her, then "
          "wraps the rope around blonde-haired Sarah's wrists" in notes(masked[1])
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
          "then the other around her right wrist, there at the small of her back, where her body hides her hands from the front, then "
          "padlocks a separate short chain from the chain between blonde-haired Sarah's handcuffs to the back of her crotch chain."
          in notes(onto)
          and "Black-haired Dana padlocks a separate short chain from the chain between blonde-haired Sarah's handcuffs to the back of "
              "her crotch chain." in notes(relink) and relink["frames"] == P.MIN_FRAMES
          and "Black-haired Dana unlocks the separate short chain that joins blonde-haired Sarah's handcuffs to the back of her crotch "
              "chain and takes it away, then steps behind blonde-haired Sarah and, there at the small of blonde-haired Sarah's back, "
              "unlocks the handcuffs with a small key and takes them off blonde-haired Sarah's wrists, while blonde-haired Sarah's hands "
              "stay behind her back until her wrists are free, then drops the handcuffs to the floor." in notes(uncuffed)
          and f"Blonde-haired Sarah: {tight}." in rested["prompt"], (notes(onto), notes(relink), notes(uncuffed), rested["prompt"]))
    off = [notes(plan(cuff + b)[0][-1]) for b in ("Dana uncuffs Sarah.", "Sarah unlocks her own handcuffs.",
                                                  "Sarah lies face down on the bed.\n\nDana unlocks Sarah's handcuffs.",
                                                  "Sarah stands still.\nrelease: Sarah, handcuffs",
                                                  "Dana unlocks Sarah's handcuffs and puts them on the bed.",
                                                  "Dana puts a steel collar on Sarah and clips a chain leash to it, padlocked to the bed."
                                                  "\n\nDana unlocks Sarah's handcuffs.")] \
        + [notes(plan(f"Dana {on}.\n\nDana {un}.")[0][-1]) for on, un in (("ties Sarah's wrists behind her back with rope",
                                                                         "unties Sarah's wrists"),
                                                                        ("cuffs Sarah's hands in front of her", "uncuffs Sarah"))]
    warn = plan(cuff + "Dana unlocks Sarah's handcuffs and puts them on the bed.")[1]
    stay = "while blonde-haired Sarah's hands stay behind her back until her wrists are free, then drops the"
    check("cuffs or rope taken off wrists bound behind her back come off there, at the small of her back, named as hers, and her hands "
          "stay behind her until her wrists are free; whoever took them off drops them to the floor unless the beat says where they go, "
          "and no loose pair is left by her leash; cuffs in front come off in front",
          "Black-haired Dana steps behind blonde-haired Sarah and, there at the small of blonde-haired Sarah's back, unlocks the handcuffs "
          f"with a small key and takes them off blonde-haired Sarah's wrists, {stay} handcuffs to the floor." in off[0]
          and "Blonde-haired Sarah unlocks the handcuffs with a small key and takes them off her wrists by feel behind her, at the small "
              "of her back, while her hands stay behind her back until her wrists are free, then drops the handcuffs to the floor."
          in off[1]
          and "Black-haired Dana leans over blonde-haired Sarah and, there at the small of blonde-haired Sarah's back, unlocks the"
          in off[2]
          and "Blonde-haired Sarah's hands stay behind her back, at the small of her back, until her wrists are free." in off[3]
          and "takes them off blonde-haired Sarah's wrists, while blonde-haired Sarah's hands stay behind her back until her wrists are "
              "free." in off[4] and "floor" not in off[4] and not warn
          and f"{stay} handcuffs to the floor." in off[5] and "separate, small pair" not in off[5]
          and "Black-haired Dana steps behind blonde-haired Sarah and, there at the small of blonde-haired Sarah's back, unties the knot "
              f"and unwinds the rope from blonde-haired Sarah's wrists, {stay} rope to the floor." in off[6]
          and "Black-haired Dana unlocks the handcuffs with a small key and takes them off blonde-haired Sarah's wrists, then drops the "
              "handcuffs to the floor." in off[7] and "small of her back" not in off[7], (off, warn))
    taped = notes(plan("Dana tapes Sarah's wrists behind her back with duct tape.")[0][-1])
    check("tape wound around her wrists behind her back by another woman names whose wrists they are",
          "then winds the duct tape around blonde-haired Sarah's wrists, turning the roll around them" in taped, taped)
    both = [plan(cuff + f"Dana {on}.\n\nDana {un}.")[0][-1]["read"] for on, un in (
        ("gags Sarah with a ball gag", "takes the gag out of Sarah's mouth and unlocks her handcuffs"),
        ("ties Sarah's ankles together with rope", "unties Sarah's ankles and unlocks her handcuffs"),
        ("gags Sarah with a ball gag", "removes Sarah's gag and handcuffs"))]
    check("'and unlocks her handcuffs' after taking something else off takes the handcuffs off too, and so does 'her gag and handcuffs'",
          both == [[("release", "Sarah, a ball gag in her mouth"), ("release", "Sarah, handcuffs behind her back")],
                   [("release", "Sarah, rope around her ankles"), ("release", "Sarah, handcuffs behind her back")],
                   [("release", "Sarah, handcuffs behind her back; a ball gag in her mouth")]], both)
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
    m, steps, sampler, sigmas, fine, notes = modelprep.prepare(pdd, 8, "res_multistep", "simple", torch.linspace(1, 0, 9), 12.0,
                                                               3.0, {}, "9")
    tops = m.model_options["transformer_options"]
    check("PDD Acc samples on its own sigmas with euler at shift 12/3",
          sampler == "euler" and tops["minimax_h3_sigma_shift_video"] == 12.0 and "PDD Acc" in notes[0] and fine is None, notes)
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
          "while 6 or more steps leave the sound as it is",
          torch.allclose(got[4][4][0], modelprep.shifted(modelprep.FLASH[8], 12.0)) and torch.allclose(got[4][4][1], got[4][4][0][3:])
          and all(got[n][4] is None for n in (6, 8, 20)), got[4][4])
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


def render(prompt, dwpose=False, counts=(), textured=False, blind=False, fine=None, **kw):
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
          and "a pair of steel shackles is locked around them just above her feet, joined by about forty centimetres of heavy steel "
              "links that never stretch or get longer" in notes(irons[1])
          and beat(irons[1]) == "Sarah tries to run for the stairs, in short, hobbled steps, her ankles joined by the heavy steel links of "
                                "the shackles."
          and "She stands with her feet a little apart, the steel links between her ankles slack on the floor." in notes(irons[2])
          and "Her ankles are shackled together by about forty centimetres of heavy steel links." in irons[1]["prompt"]
          and beat(irons[3]) == "Sarah lifts both legs together, her ankles joined by the heavy steel links of the shackles, then one "
                                "leg."
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
              "Luis grabs blonde-haired Sarah's legs and arms, her ankles joined by the heavy steel links of the shackles, and pulls "
              "blonde-haired Sarah onto the bed.",
              "Luis lifts blonde-haired Sarah by the feet, her ankles joined by the heavy steel links of the shackles, and holds "
              "blonde-haired Sarah upside down."],
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


def test_positions():
    print("\n=== a position someone is put into holds through the beats after it ===")
    mem = ("Lena: a young woman with long blonde hair, in a black lace bra and black lace panties.\n"
           "Dana: a woman with long brown hair, in a black corset and black leather trousers.")
    plan = lambda *b: P.plan("A dim bedroom with a wide bed and a table.\n\n" + "\n\n".join(b), mem)[0]
    thrown = plan("Dana throws Lena onto the bed.", "Dana handcuffs Lena's wrists behind her back.", "Dana cuffs Lena's ankles.")
    pinned = plan("Dana pins Lena face down on the bed and cuffs her hands behind her back.", "Dana chains Lena's ankles.")
    bent = plan("Dana bends Lena over the table and cuffs her wrists.", "Dana cuffs Lena's ankles to the table legs.")
    held = lambda *b: next((x for x in re.findall(r"Blonde-haired Lena (?:stays|ends the shot) ([^.:]+)", notes(plan(*b)[-1]))), "")
    others = [held("Lena is thrown onto the bed.", "Dana ties Lena's ankles together with rope."),
              held("Dana knocks Lena to the floor.", "Dana cuffs Lena's ankles."),
              held("Dana wrestles Lena to the ground and cuffs her.", "Dana shackles Lena's ankles."),
              held("Dana pushes Lena into a chair.", "Dana tapes Lena's ankles to the chair legs."),
              held("Dana throws Lena onto the bed.", "Dana pulls Lena off the bed and cuffs her hands behind her back.")]
    check("thrown or pinned down, she stays lying where she landed, face down when pinned face down, in every shot after it, the ankle "
          "cuffs go on with her legs flat where they lie and she is never laid down again; bent over a table she stays bent over it "
          "while her ankles are cuffed to its legs, which are never read as her legs; passive, knocked, wrestled, pushed into a chair "
          "and pulled off the bed each hold as written",
          all("Blonde-haired Lena stays lying on the bed for the whole shot" in notes(s) for s in thrown[1:])
          and "kneels at blonde-haired Lena's feet, holds Lena's ankles pressed together, her legs straight and flat where they lie"
              in notes(thrown[2])
          and "stays lying face down on the bed" in notes(pinned[1]) and " lays " not in notes(pinned[1])
          and len(bent) == 2 and bent[1]["read"] == [("hold", "Lena, handcuffs on her ankles, locked to the table legs")]
          and "stays bent over the table" in notes(bent[1]) and " lays " not in notes(bent[1])
          and others == ["lying on the bed for the whole shot, holding that position from the first frame to the last",
                         "lying on the floor for the whole shot, holding that position from the first frame to the last",
                         "lying on the ground for the whole shot, holding that position from the first frame to the last",
                         "sitting in the chair for the whole shot, holding that position from the first frame to the last",
                         "standing upright"],
          ([notes(s) for s in thrown[1:]], others))


def test_worn():
    print("\n=== a restraint she already wears is on from the start, wherever it is written, and stays on its own ===")
    mem = ("Lena: a young woman with long blonde hair, in a black lace bra and black lace panties.\n"
           "Dana: a woman with long brown hair, in a black corset and black leather trousers.")
    story = "A dim bedroom with a wide bed.\n\nLena stands by the bed.\n\nDana walks up to Lena and kisses her."
    chain = ("Lena: a crotch chain locked tight around her waist, its center chain pulled tight under her crotch and over her vagina from "
             "the front of her waist to the small of her back.")
    on = lambda m=mem, a="": P.plan(story, m, a)[0][-1]["line"]
    given = [on(a=t) for t in ("Lena wears a crotch chain.", "Lena has a crotch chain locked on.", "Lena: crotch chain.",
                               "Lena is naked except for a crotch chain.")] + [
        on(mem.replace("in a black lace bra and black lace panties", "naked except for a crotch chain")),
        on(mem + "\nhold: Lena, crotch chain")]
    plain = [on(mem.replace("in a black lace bra", d)) for d in ("with a thin gold chain around her neck, in a black lace bra",
                                                                 "in a shirt with rolled cuffs, a black lace bra",
                                                                 "with handcuffs on her belt, in a black lace bra")]
    people = P.read_memory(mem)[0]
    worn = [P.read_beat(t, list(people), people, {}, {}, {}, {})[0][0][:2] for t in (
        "Lena stands by the bed, handcuffs on her wrists.", "Lena waits, leather cuffs on her ankles.",
        "Dana cuffs Lena with steel handcuffs.")]
    cuffed = notes(P.plan("A dim bedroom.\n\nDana cuffs Lena's hands behind her back.", mem.replace(
        "in a black lace bra and black lace panties", "naked except for a crotch chain"))[0][0])
    leash = P.plan("A dim bedroom with a wide bed.\n\nDana puts a collar on Lena and clips a leash to it.", mem)[0][0]["read"]
    check("a crotch chain written in the anchor, her description or a bare hold tag gets its full description in every shot, while a "
          "necklace, shirt cuffs or cuffs on a belt stay clothing; worn cuffs belong to the one wearing them; nothing else is hooked onto "
          "a crotch chain; a leash clipped to \"it\" after a collar goes on the collar",
          all(chain in x for x in given) and all("on: " not in x for x in plain)
          and worn == [("have", "Lena, handcuffs behind her back"), ("have", "Lena, leather cuffs on her ankles"),
                       ("hold", "Lena, steel handcuffs behind her back")]
          and "Nothing else is clipped, hooked or locked onto her crotch chain: no other chain, cuff, link or leash touches it at any "
              "point, front or back, and her locked hands rest just above the back of it, free of it." in cuffed
          and leash[-1] == ("hold", "Lena, a leash clipped to the collar around her neck, its other end held by Dana"),
          (given, plain, worn, leash))
    mem = ("Sarah: a young woman with long red hair, in a white tank top and black shorts.\n"
           "Marcus: a man with short dark hair and a beard, in a black t-shirt and jeans.")
    ladder = P.plan("A dim garage with a tall wooden stepladder.\n\nMarcus cuffs Sarah's wrists above her head to the top of the "
                    "ladder. He spreads her legs wide and cuffs her ankles to the corners of the base of the ladder.\n\nSarah "
                    "struggles.\n\nMarcus kisses Sarah.\n\nMarcus watches her.", mem)[0]
    side = ("Dark-haired Marcus stays at red-haired Sarah's left side, the only dark-haired Marcus in the frame, and nobody is at her "
            "right side.")
    check("cuffed to a ladder, her wrists lock to its top and her spread ankles to its two corners one at a time, by one man who does "
          "it all from her left side and stays there while she is held and he only looks on, the ladder said to be the only one, never "
          "read as wide cuffs or as her legs held together",
          [s["read"] for s in ladder[:2]] == [[("hold", "Sarah, handcuffs on her wrists above her head, locked to the top of the ladder")],
                                          [("hold", "Sarah, handcuffs on her ankles spread apart, locked to the corners of the base of "
                                                    "the ladder")]]
          and "crouches at red-haired Sarah's left side, spreads her legs apart and locks her left ankle to the left corner of the base "
              "of the ladder, then reaches across and locks her right ankle to the right corner of the base of the ladder"
              in notes(ladder[1])
          and all("Dark-haired Marcus does all of this from red-haired Sarah's left side: he stays on that side of her the whole shot, "
                  "nobody is at her right side, and dark-haired Marcus appears only once in every frame." in notes(s) for s in ladder[:2])
          and [side in notes(s) for s in ladder[2:]] == [True, False, False]
          and all("There is only one ladder here, always in the same spot." in notes(s) for s in ladder)
          and "pressed together" not in notes(ladder[1]), [notes(s) for s in ladder])


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
          and beat(wrists) == "Luis pulls blonde-haired Sarah's blouse up above her breasts."
          and beat(free[-1]) == "Luis strips blonde-haired Sarah naked."
          and "pulls them down her legs and off over her feet" in notes(free[-1]),
          ([beat(s) for s in naked + clothes], ripped[1]["line"], beat(wrists), [notes(s) for s in free]))
    standing = "Sarah stands by the table, her wrists handcuffed behind her back."
    off, own, shoes = (plan(standing, b)[0][1] for b in ("Luis pulls her jeans down and off.", "Sarah wriggles out of her jeans.",
                                                         "Sarah takes off her sneakers."))
    front = plan("Sarah stands by the table, her hands cuffed in front of her.", "Luis pulls down her jeans.")[0][1]
    lying = plan("Sarah lies on the bed, her wrists handcuffed behind her back.", "Luis pulls her jeans off.")[0][1]
    back = plan(standing, "Luis pulls her jeans down to her knees.", "Luis pulls them back up.")[0][2]
    check("with her wrists cuffed, her clothes change in the beat and in the step without naming her hands, and her pose keeps her "
          "hands where they are cuffed; standing, she steps out of her jeans one foot after the other; on her own she pushes them "
          "down and her shoes off without her hands; lying down she steps out of nothing; 'them' at the start of the next beat is the "
          "jeans",
          beat(off) == "Luis pulls blonde-haired Sarah's jeans down and off."
          and "opens blonde-haired Sarah's blue jeans at the waist and pulls them down her legs to her ankles, and blonde-haired Sarah "
              "steps out of them one foot at a time." in notes(off)
          and "Blonde-haired Sarah's arms stay pulled behind her back the whole shot" in notes(off)
          and beat(own) == "Sarah wriggles out of her jeans."
          and "Sarah pushes her blue jeans down over her hips and wriggles them down her legs to her ankles, and steps out of them one "
              "foot at a time." in notes(own)
          and "Sarah pushes her white sneakers off with her feet, one heel against the other." in notes(shoes)
          and beat(front) == "Luis pulls down blonde-haired Sarah's jeans."
          and "pulls them down her legs and off over her feet." in notes(lying) and "steps out" not in notes(lying)
          and not any(MIXED(notes(x), "jeans", "sneakers") for x in (off, own, shoes, front, lying, back))
          and "pulled back: Sarah: blue jeans." in back["line"]
          and beat(back) == "Luis pulls them back up.",
          (beat(off), notes(off), beat(own), notes(own), notes(shoes), beat(front), notes(lying), back["line"], beat(back)))


def test_skirts():
    print("\n=== a skirt has no legs to pull down, so bound legs never stop it, and a beat split into shots says each thing once ===")
    mem = ("Sarah: a young woman with long blonde hair, in a white blouse, a short denim skirt, black panties and white sneakers.\n"
           "Dana: a woman with short black hair, in a red dress and black heels.")
    jeans = mem.replace("a short denim skirt", "blue jeans")
    room = "A cellar with a steel table, a wooden chair, a bed and a staircase.\n\n"
    cuffed = "Dana pushes Sarah face down onto the floor and cuffs her wrists behind her back.\n\n"
    hog = "Dana lifts both of Sarah's legs and hogties her with another set of handcuffs"
    story = lambda cast, *b: P.plan(room + "\n\n".join(b), cast)
    said = lambda shots: [s["prompt"].split("\n\n")[-2] for s in shots[1:]]
    ankles = "Sarah lies on the bed, her ankles handcuffed together."
    skirt, warn = story(mem, ankles, "Dana takes off Sarah's skirt.")
    pants, pwarn = story(jeans, ankles, "Dana takes off Sarah's jeans.")
    free = story(mem, "Sarah stands by the table.", "Dana takes off Sarah's skirt.")[0]
    check("a skirt comes off over the hips of a woman with cuffed ankles, her legs left exactly as they are and no warning, while "
          "jeans still stop above the cuffs and a free woman's skirt still slides down her legs",
          "opens blonde-haired Sarah's short denim skirt at the waist and slides it off her hips and away, leaving her bound legs "
          "exactly as they are." in notes(skirt[-1])
          and not any(w in skirt[-1]["prompt"] for w in ("down her legs", "just above the handcuffs")) and not warn
          and "pulls blonde-haired Sarah's blue jeans down to just above the handcuffs on her ankles" in notes(pants[-1])
          and any("blue jeans cannot come off past her bound legs" in w for w in pwarn)
          and "slides it down her legs and off over her feet" in notes(free[-1]), (notes(skirt[-1]), warn, notes(pants[-1]), pwarn))
    hogged = story(mem, cuffed + hog + ".", "Dana removes Sarah's skirt.")[0]
    check("a skirt comes off a hogtied woman without her bound legs being pulled on",
          "slides it off her hips and away, leaving her bound legs exactly as they are" in notes(hogged[-1])
          and not any(w in hogged[-1]["prompt"] for w in ("down her legs", "just above the handcuffs")), notes(hogged[-1]))
    first = said(story(mem, cuffed + "Dana removes Sarah's skirt, then lifts both of Sarah's legs and hogties her with another set "
                                     "of handcuffs.")[0])
    last = said(story(mem, cuffed + hog + ", then removes Sarah's skirt.")[0])
    check("a removal first in a beat with a hogtie is said once, before the ankles are cuffed, and the hogtie then goes on without it",
          len(first) == 2 and first[0].startswith("Dana takes off Sarah's short denim skirt") and first[0].endswith("Dana cuffs Sarah's "
          "ankles together.") and first[1].startswith("Dana lifts both of Sarah's legs and hogties") and "skirt" not in first[1], first)
    check("a removal last in a beat with a hogtie comes in a shot of its own after the hogtie, which neither cuffs the ankles a "
          "second time nor lifts the legs again",
          len(last) == 3 and last[0] == "Dana cuffs Sarah's ankles together." and last[1].startswith("Dana bends Sarah's knees")
          and "cuffs Sarah's ankles" not in last[1] and last[2].startswith("Dana removes Sarah's skirt")
          and not any(w in last[2] for w in ("lifts", "hogtie")), last)


def test_anchors():
    print("\n=== a restraint fastened to something keeps the arms where they were cuffed, said in terms of where they are ===")
    mem = ("Sarah: a woman with long blonde hair, in a white tank top.\nDana: a woman with short black hair, in a black jacket.")
    last = lambda *b: notes(P.plan("A cellar.\n\nDana and Sarah stand together.\n\n" + "\n\n".join(b), mem)[0][-1])
    lift = ("lifting", "rising", "reaching", "raised", "above her head", "full length from her shoulders to her wrists")
    front = last("Dana handcuffs Sarah's wrists in front of her, then chains her to the wall.", "Sarah waits.")
    check("wrists cuffed in front and chained to a wall rest at belly height with the wall fastening said to be at waist height, and "
          "nothing about lifting is said",
          "wrists stay locked together in front of her the whole shot and chained to the wall" in front
          and "locked hands rest together at the height of her belly, in plain view below her chest" in front
          and "Fastened to the wall at the height of her waist, her wrists stay at waist height, below her shoulders, in front of her."
          in front and not any(w in front for w in lift), front)
    behind = last("Dana handcuffs Sarah's wrists behind her back, then chains her to a hook in the ceiling.", "Sarah waits.")
    check("wrists cuffed behind her back stay behind it when the chain goes to a hook in the ceiling: a slack chain hangs from it, "
          "and nothing says raised",
          "arms stay pulled behind her back" in behind and "A long slack chain hangs loose from her wrists to the hook in the ceiling, "
          "so her wrists stay at waist height, below her shoulders, behind her back." in behind
          and not any(w in behind for w in lift), behind)
    both = last("Dana handcuffs Sarah's wrists in front of her, then chains her to a hook in the ceiling.", "Sarah waits.")
    check("wrists cuffed in front stay in front when the chain goes to the ceiling",
          "in front of her the whole shot and chained to the hook in the ceiling" in both and "A long slack chain hangs loose" in both
          and not any(w in both for w in lift), both)
    up = last("Dana chains Sarah's wrists to a hook in the ceiling.", "Sarah waits.")
    check("wrists chained overhead with no position given still hang raised, as written", "arms stay raised above her head" in up, up)
    bare = last("Dana handcuffs Sarah to the pipe.", "Sarah waits.")
    check("wrists fastened to a pipe with no position given rest in front of her at belly height, the pipe at waist height",
          "wrists stay locked together in front of her the whole shot and locked to the pipe" in bare
          and "Fastened to the pipe at the height of her waist" in bare and not any(w in bare for w in lift), bare)
    floor = last("Dana handcuffs Sarah's wrists in front of her, then chains her to a ring in the floor.", "Sarah waits.")
    check("a ring in the floor is reached by a short chain running down, with her wrists at waist height",
          "A short chain runs down from her wrists to the ring in the floor, and her wrists stay at waist height" in floor, floor)
    room = "A bedroom with a bed and a wooden headboard.\n\nDana and Sarah are in the room.\n\n"
    bed = lambda *b: P.plan(room + "\n\n".join(b), mem)[0][-1]
    lying = bed("Sarah lies on the bed.", "Dana cuffs Sarah's wrists to the headboard.", "Sarah waits.")
    check("a headboard puts nothing above her head: with no position given the wrists are just locked to it, in the item and in the shot",
          "above her head" not in lying["line"] and "above her head" not in lying["prompt"] and "stretched out" not in lying["prompt"]
          and "her wrists are locked to the headboard" in notes(lying) and "straight to the headboard" not in notes(lying),
          (lying["line"], notes(lying)))
    sitting = bed("Sarah sits on the bed.", "Dana cuffs Sarah's wrists to the headboard.", "Sarah waits.")
    check("sitting at a headboard she rests her hands at belly height, a long slack chain going to it",
          "above her head" not in sitting["prompt"] and "locked hands rest together at the height of her belly" in notes(sitting)
          and "A long slack chain hangs loose from her wrists to the headboard" in notes(sitting), notes(sitting))
    said = bed("Sarah lies on the bed.", "Dana cuffs Sarah's hands above her head to the headboard.", "Sarah waits.")
    check("above her head is kept when the beat says so", "stretched out above her head" in notes(said), notes(said))
    owner = bed("Dana cuffs Sarah's wrists behind her back.", "Dana lays her on the bed and chains the cuffs to the headboard.",
                "Sarah waits.")
    check("chaining 'the cuffs' to the headboard, with only one woman cuffed, is read onto her and keeps her wrists behind her back",
          "chains around her wrists behind her back, chained down to the headboard" in owner["line"]
          and "pulled behind her back" in notes(owner) and "above her head" not in owner["prompt"], (owner["line"], notes(owner)))
    plain = last("Dana cuffs Sarah's hands in front of her.", "Sarah waits.")
    check("cuffs in front with nothing fastened rest at belly height too, with no fastening said",
          "locked hands rest together at the height of her belly" in plain and "Fastened" not in plain
          and not any(w in plain for w in lift), plain)
    step = P.plan("A cellar.\n\nDana and Sarah stand together.\n\nDana handcuffs Sarah's wrists in front of her and chains her to "
                  "the pipe.", mem)[0]
    check("the step that fastens them says the wrists are fastened at waist height, and the end of the shot has them low there, "
          "not 'in plain view'", any("to the pipe at waist height" in s["prompt"] for s in step)
          and any(", low at waist height" in s["prompt"] and "locked to the pipe in plain view" not in s["prompt"] for s in step),
          [s["prompt"][-700:] for s in step[1:2]])


def test_marks():
    print("\n=== words written on a body and brands stay on it, shot after shot, until washed off ===")
    crew = ("Sarah: a woman with long blonde hair, in a white tank top.\nDana: a woman with short black hair, in a black jacket.\n"
            "Marcus: a man with short dark hair, in a black t-shirt.")
    plan = lambda *b, mem=crew: P.plan("A room.\n\nDana and Sarah stand together.\n\n" + "\n\n".join(b), mem)
    shots, warn = plan('Dana uncaps a black marker and writes "SLAVE" across Sarah\'s stomach.', "Sarah looks down.", "Dana laughs.")
    check("a quoted word being written is not read as a spoken line, and stays in the beat as written",
          shots[1]["said"] == [] and not shots[1]["cues"] and "writes «SLAVE» across Sarah's stomach" in shots[1]["prompt"]
          and "<d>" not in shots[1]["prompt"], (shots[1]["said"], shots[1]["prompt"]))
    check("it goes on in the shot that writes it, and is on her in every shot after",
          'has the word "SLAVE" written across her stomach in plain view' in notes(shots[1])
          and all('has the word "SLAVE" written across her stomach. It stays exactly as it is' in notes(s) for s in shots[2:])
          and "SLAVE" not in notes(shots[0]) and not warn, [notes(s) for s in shots])
    marks = plan("Dana brands Sarah's left hip with a glowing iron.", 'Dana writes "MINE" on Sarah\'s thigh with a red marker.',
                 "Sarah gasps.", "Dana washes the writing off Sarah's thigh.", "Sarah looks up.")[0]
    check("a brand goes on and stays, and a second mark goes on beside it",
          "has a brand mark on her left hip in plain view" in notes(marks[1])
          and 'has the word "MINE" written on her thigh with a red marker in plain view' in notes(marks[2])
          and "a brand mark on her left hip" in notes(marks[3]) and 'the word "MINE" written on her thigh' in notes(marks[3]),
          notes(marks[3]))
    check("washing the writing off takes only the writing off, and the brand is still there after",
          'no longer has the word "MINE"' in notes(marks[4]) and "a brand mark on her left hip" in notes(marks[4])
          and "MINE" not in notes(marks[5]) and "a brand mark on her left hip" in notes(marks[5]), (notes(marks[4]), notes(marks[5])))
    tagged = plan("Sarah sighs.\nmark: Sarah, the word \"OWNED\" written across her back in red lipstick", "Dana smiles.",
                  "Sarah nods.\nunmark: Sarah", "Dana nods.")[0]
    check("mark: puts it on from that shot and unmark: takes it off, with no guessing from the beat",
          'has the word "OWNED" written across her back in red lipstick in plain view' in notes(tagged[1])
          and 'written across her back in red lipstick. It stays' in notes(tagged[2])
          and "no longer has" in notes(tagged[3]) and "OWNED" not in notes(tagged[4]), [notes(s) for s in tagged])
    start = plan("Sarah sighs.", "Dana smiles.", mem=crew + "\nmark: Sarah, a brand mark on her left shoulder")[0]
    check("a mark: line in character_memory has it on from the first shot",
          all("Sarah has a brand mark on her left shoulder. It stays" in notes(s) for s in start), [notes(s) for s in start])
    guess, gwarn = plan('The word "SLUT" is written across her stomach.', "Sarah looks down.")
    check("when 'her' could be either woman nothing is guessed: it says so, and nothing is put on anyone",
          any("could not tell whose body this is" in w for w in gwarn) and not any("SLUT" in notes(s) for s in guess), gwarn)
    man, mwarn = plan('Marcus writes "MINE" across Sarah\'s chest.', "Marcus writes \"YES\" across her back.", mem=crew)
    check("a man writing on a woman by name is read, and 'her' with Dana also in the room is asked about, not guessed",
          'Sarah has the word "MINE" written across her chest' in notes(man[1]) and not mwarn == []
          and not any("YES" in notes(s) for s in man), (mwarn, [notes(s) for s in man]))
    quiet = ["Dana writes in her notebook.", "Dana draws her sword.", "Dana draws a heart on the whiteboard.",
             "Dana draws a line across Sarah's chest with her finger.", "Sarah wears a designer brand of sneakers on her feet.",
             "Dana paints Sarah's nails red."]
    check("writing in a notebook, drawing a sword, a finger across her chest, a shoe brand and painted nails put nothing on her skin",
          not any("marks" in plan(q, "Sarah smiles.")[0][1]["line"] for q in quiet), quiet)
    heard = plan('Dana writes a note and hands it over, "Read this."', "Sarah reads it.")[0]
    check("a spoken line after a clause that mentions writing is still spoken",
          [w for w, *_ in heard[1]["said"]] == ["Dana"], heard[1]["said"])


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
              "cuffs and takes it away, then leans over blonde-haired Sarah and, there at the small of blonde-haired Sarah's back, unlocks "
              "the handcuffs with a small key and takes them off blonde-haired Sarah's wrists, while blonde-haired Sarah's hands stay "
              "behind her back until her wrists are free, then drops the handcuffs to the floor." in notes(hog[2])
          and hog[2]["line"].endswith("comes off: Sarah: handcuffs behind her back.") and hog[3]["line"].endswith(
              "on: Sarah: handcuffs on her ankles.")
          and rope[0]["read"] == [("have", "Sarah, rope around her wrists behind her back; rope around her ankles, tied to the rope "
                                           "around her wrists behind her back by a separate short rope")]
          and rope[1]["read"] == [("hold", "Sarah, rope around her ankles")]
          and "Black-haired Dana unties the separate short rope that joins the rope around blonde-haired Sarah's wrists to the rope around "
              "her ankles and takes it away." in notes(rope[1])
          and cuffs[-1]["read"] == [("hold", f"Sarah, {link}")],
          ([(s["read"], s["line"], beat(s)) for s in hog + rope + cuffs], [notes(s) for s in hog[2:3] + rope[1:]]))
    put, back, side, front = (plan(*b)[0] for b in (
        ("Sarah stands by the bed.", "Dana hogties Sarah on the bed with handcuffs.", "Sarah struggles."),
        ("Sarah lies on the bed, her wrists handcuffed behind her back.", "Dana hogties Sarah."),
        ("Sarah lies on her side on the floor, hogtied with rope.",),
        ("Sarah lies on the floor, hogtied, her wrists cuffed in front of her.",)))
    down = ("lying face down, both her arms lie along her back at their full length, and her hands stay locked together on top of the "
            "small of her back")
    check("a hogtie lays her face down: whoever ties it lays her face down where the beat says, or rolls her onto her front from her "
          "back, before lifting her ankles, and she stays face down with her hands behind her back while it is on; lying on her side or "
          "cuffed in front stays so when the text says so",
          "Black-haired Dana lays blonde-haired Sarah face down on the bed, bends blonde-haired Sarah's knees and lifts her handcuffed "
          "ankles together up behind her toward her hands" in notes(put[3])
          and "she goes down onto her front with both arms staying behind her the whole way, and then " + down in notes(put[3])
          and down in notes(put[4]) and "goes down" not in notes(put[4])
          and "Black-haired Dana rolls blonde-haired Sarah onto her front, bends blonde-haired Sarah's knees" in notes(back[2])
          and down in notes(back[2]) and down in notes(plan("Sarah lies on the floor, hogtied with rope.")[0][0])
          and "lying on her side, both her arms run down behind her along her back" in notes(side[0])
          and front[0]["read"] == [("have", "Sarah, handcuffs in front of her; handcuffs on her ankles, locked to her handcuffs in "
                                            "front of her by a separate short chain")],
          [notes(s) for s in put[3:] + back[2:] + side] + [front[0]["read"]])
    ends = lambda *b: plan(*b)[0][-1]["read"][-1][1]
    worn, said, kit, fetched = (ends(*b) for b in (("Dana handcuffs Sarah's wrists behind her back.", "Dana hogties Sarah."),
                                                  ("Dana handcuffs Sarah and hogties her.",),
                                                  ("Dana chains Sarah's wrists behind her back.", "Dana hogties her."),
                                                  ("Dana takes a coil of chain from the shelf.", "Dana hogties Sarah.")))
    soft, strap, zip_, tape = (plan(f"Dana hogties Sarah with {g}.", "Dana unties the hogtie.")[0]
                               for g in ("leather cuffs", "straps", "zip ties", "duct tape"))
    check("a hogtie uses the restraint the beat names, else what is on her wrists or was brought out before, and rope only when "
          "nothing is; its ankle restraint keeps the same look, and the link between them is of the same kind",
          worn == said == "Sarah, handcuffs on her ankles, locked to her handcuffs behind her back by a separate short chain"
          and kit == fetched == "Sarah, chains around her ankles, chained to the chain around her wrists behind her back by a separate "
                                "short chain"
          and ends("Dana gags Sarah.", "Dana hogties her.").startswith("Sarah, rope around her ankles, tied to the rope")
          and ends("Dana hogties Sarah with steel chains.").startswith("Sarah, steel chains around her ankles")
          and soft[2]["read"] == [("hold", "Sarah, leather cuffs on her ankles, locked to her wrist cuffs behind her back by a separate "
                                           "short chain")]
          and "padlocks a separate short chain from the link between her wrist cuffs to the link between her ankle cuffs" in notes(soft[2])
          and "The cuffs on blonde-haired Sarah's wrists and the cuffs on her ankles are separate restraints" in notes(soft[2])
          and "then buckles a separate short strap from the strap around her wrists to the strap around her ankles" in notes(strap[2])
          and "then fastens a separate short zip tie from the zip tie around her wrists" in notes(zip_[1])
          and "cuts the separate short zip tie that joins" in notes(zip_[2])
          and "lifts her taped ankles together up behind her toward her hands, then runs a separate short strip of tape" in notes(tape[2])
          and "peels off the separate short strip of tape that joins" in notes(tape[3]) and "winds" not in notes(tape[3]),
          [worn, said, kit, fetched] + [(s["read"], notes(s)) for s in soft[2:] + strap[2:] + zip_[1:] + tape[2:]])
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


def test_holds():
    print("\n=== poses, leashes and objects hold from shot to shot ===")
    mem = ("Sarah: a woman in her twenties with long blonde hair, in a white tank top and denim shorts.\n"
           "Dana: a woman in her thirties with short black hair, in a black blazer and grey trousers.")
    plan = lambda *b, m="": P.plan("A dim bedroom.\n\n" + "\n\n".join(b), mem + m)[0]
    held = lambda shots: [next((t for t in re.split(r"(?<=[.!?])\s+", x["prompt"]) if "holding that position" in t), "") for x in shots]
    bent = held(plan('"Bend over and spread your legs," Dana says.', "Sarah bends over the bed and spreads her legs.",
                     "Dana walks slowly around her.", "Dana runs a hand down Sarah's back.", "Sarah stands up and turns to face Dana."))
    told = held(plan("Dana orders Sarah to bend over the desk and spread her legs.", "Dana picks up a riding crop."))
    obeys = held(plan('"Bend over and spread your legs," Dana orders.', "Sarah does as she is told.", "Dana walks around her."))
    check("a pose taken in one beat holds through the shots after it, whether it was told, ordered or obeyed, until the beat that "
          "changes it",
          bent[:2] == ["", ""] and bent[2] == bent[3] == "Blonde-haired Sarah stays bent over the bed with her legs spread apart for the "
          "whole shot, holding that position from the first frame to the last: her chest and shoulders rest on it."
          and bent[4] == "" and "stays bent over the desk with her legs spread apart" in told[1]
          and "stays bent forward at the waist with her legs spread apart" in obeys[2], [bent, told, obeys])
    plain = lambda *b: plan("Dana cuffs Sarah's hands behind her back.", *b)
    cuffed = [held(plain(b, "Dana walks around her."))[-1] for b in ("Dana bends Sarah over the bed.", "Dana pushes Sarah face down over "
              "the desk.", "Sarah gets on all fours.")]
    turned = held(plain("Dana bends Sarah over the bed.", "Dana walks around her.", "Sarah turns around to face Dana."))
    camera = [t for t in re.split(r"(?<=[.!?])\s+", plain("Dana bends Sarah over the bed.", "Dana walks around her.")[2]["prompt"])
              if t.startswith("The camera keeps")]
    check("a cuffed pose keeps the cuffs where they are: bent over, her chest rests on what she is bent over and her hands stay locked "
          "behind her, on all fours with her hands cuffed behind her she kneels bent forward instead, turning round ends the bend, and "
          "the camera keeps all of her in view rather than her face",
          cuffed[0] == "Blonde-haired Sarah stays bent over the bed for the whole shot, holding that position from the first frame to the "
          "last: her chest and shoulders rest on it and her cuffed hands stay locked behind her back."
          and "stays bent over the desk" in cuffed[1] and "stays kneeling bent forward with her chest down" in cuffed[2]
          and turned[-1] == "" and camera == ["The camera keeps black-haired Dana's face in frame and all of blonde-haired Sarah in view "
                                              "for the whole shot."], [cuffed, turned, camera])
    pulled = held(plan("Dana pulls Sarah's denim shorts down to her knees.", "Dana steps back."))
    knelt = held(plan("Sarah kneels on the floor.", "Dana circles her.", "Sarah gets on all fours.", "Dana watches her."))
    check("pulling clothes down to the knees is not kneeling, while kneeling and going on all fours each hold until changed",
          pulled == ["", ""] and "stays kneeling" in knelt[1] and knelt[2] == "" and "stays on all fours" in knelt[3], [pulled, knelt])
    leash = plan("Dana cuffs Sarah's hands behind her back.", "Dana clips a chain leash to Sarah's collar and ties the other end to "
                 "the pipe.", "Sarah turns around and faces the wall.", m="\nhold: Sarah, a black leather collar")
    runs = "Blonde-haired Sarah's leash is one single chain of a fixed length, and one end of it is clipped to the ring at the front " \
           "of her collar at her throat. Its other end is padlocked to the pipe at one fixed point on it, {} Between them the chain " \
           "hangs slack in one loose curve: from the ring at her throat it runs down across her chest to the side of her nearest the " \
           "pipe, drops beside her to the floor, clear of her clothes, and lies loose along the floor, or along whatever she is on, all " \
           "the way to the pipe. Anyone who comes near it steps over it or nudges it aside with a foot, and the chain only slides and " \
           "settles. Its only fixed points are the ring at her collar and the pipe."
    lines = lambda s: re.split(r"(?<=[.!?])\s+", s["prompt"])
    mixed = [t for s in leash for t in lines(s) if re.search(r"\bleash\b", t) and re.search(r"\bcuffs?\b|\bhandcuffs\b|\bwrists?\b", t)]
    first = [min(k for k, t in enumerate(lines(s)) if "leash is one single" in t)
             < min(k for k, t in enumerate(lines(s)) if t.startswith("Blonde-haired Sarah's arms stay")) for s in leash[1:]]
    check("a chain leash clipped to a collar and tied off is one leash from her collar to one fixed point on what it is padlocked "
          "to, kept at the point it holds in the frame each shot opens on, never a chain wrapped round her neck or a rope, slack along "
          "the floor whichever way she turns, and running on its own apart from her cuffs",
          "a chain leash clipped to the collar around her neck, fastened to the pipe" in leash[1]["line"]
          and "clips the chain to the ring at the front of blonde-haired Sarah's collar and padlocks its other end to the pipe"
              in leash[1]["prompt"] and runs.format("and from the moment it is padlocked, the padlock hangs on its own there.")
              in leash[1]["prompt"]
          and runs.format("the same point as in the frame this shot opens on, where the padlock hangs on its own the whole shot.")
              in leash[2]["prompt"]
          and "two solid cuffs joined to each other by their own three short, rigid steel links, which hold them together, side by "
              "side, every moment" in leash[2]["prompt"] and not mixed
          and first == [True, True]
          and not any(re.search(r"\brope\b|chains around her neck|wraps the chain", x["prompt"]) for x in leash)
          and "clink and rattle" in leash[2]["sound"], [x["line"] for x in leash])
    ring = plan("Dana buckles a leather collar around Sarah's neck, clips a chain leash to it and locks the other end of the leash to a "
                "steel ring in the wall.", "Dana clips a leash to Sarah's collar. Dana ties the other end of the leash to the bedpost.")
    check("a collar and its leash put on in one sentence are both read, and the same leash moved later stays one leash",
          "hold: Sarah, a leather collar around her neck" in ring[0]["line"]
          and "a chain leash clipped to the collar around her neck, fastened to the steel ring in the wall" in ring[0]["line"]
          and "goes on: Sarah: a chain leash clipped to the collar around her neck, fastened to the bedpost." in ring[1]["line"]
          and "steel ring" not in ring[1]["line"], [x["line"] for x in ring])
    box = plan("Dana walks in carrying a small wooden box with a brass lock and sets it on the dresser.",
               "Dana opens the box and takes out a black leather riding crop.", "Dana walks to the window and looks outside.",
               "Dana turns back to the dresser and taps the crop against the box.")
    kept = [[t for t in re.split(r"(?<=[.!?])\s+", x["prompt"]) if "looking exactly as it did" in t or "as the shot begins" in t]
            for x in box]
    check("an object brought in keeps its look and its place: named again, it is the same object where it was left, and out of the "
          "beat it stays where it was",
          kept[0] == [] and kept[1] == ["The box is the same small wooden box with a brass lock as before, on the dresser as the shot "
                                        "begins."]
          and "The small wooden box with a brass lock is still on the dresser, looking exactly as it did." in kept[2]
          and any(re.fullmatch(r"The black leather riding crop is still in (?:black-haired )?Dana's hand, looking exactly as it did\.", t)
                  for t in kept[2])
          and any(re.fullmatch(r"The crop is the same black leather riding crop as before, in (?:black-haired )?Dana's hand as the shot "
                               r"begins\.", t) for t in kept[3]), kept)
    check("clothes, restraints, body parts, people, furniture and doors held open are not objects to keep",
          not any(P._props(t, "", ["Sarah", "Dana"]) for t in (
              "Dana takes off her blazer and drapes it over the chair.", "Dana takes Sarah's hand.", "Dana takes a deep breath.",
              "Dana holds the door open.", "Dana picks up a roll of duct tape.", "Dana carries Sarah to the bed.",
              "Dana places a blindfold over Sarah's eyes.", "Sarah holds the edge of the table.", "Dana puts a hand on her hip."))
          and P._props("Dana hands Sarah a glass of water.", "", ["Sarah", "Dana"]) == {"glass": ("glass of water", None, "Sarah", "Dana")},
          P._props("Dana holds the door open.", "", ["Sarah", "Dana"]))


def test_leash_ends():
    print("\n=== where the other end of a leash is ===")
    mem = ("Sarah: a woman in her twenties with long blonde hair, in a white blouse and a short black pleated skirt.\n"
           "Marcus: a man in his thirties with short dark hair, in a black t-shirt and blue jeans.\n"
           "Mia: a woman in her thirties with short red hair, in a grey suit.\n")
    run = lambda *b, m="hold: Sarah, a black leather collar": P.plan("A dim bedroom.\n\n" + "\n\n".join(b), mem + m)
    note = lambda s: next((t for t in re.split(r"(?<=[.!?])\s+", s["prompt"]) if t.startswith("Its other end")), "")
    clip = run("Marcus clips a chain leash to Sarah's collar.", "Marcus pulls Sarah across the room by the leash.")[0]
    check("a leash clipped on with nothing to tie it to stays in the hand of whoever clipped it, and the picture is told so in every "
          "shot",
          "clips the chain to the ring at the front of blonde-haired Sarah's collar and keeps its other end in his hand"
              in clip[0]["prompt"]
          and all(note(s) == "Its other end is in dark-haired Marcus's hand the whole shot, and the chain runs from her throat straight to "
                             "that hand." for s in clip) and "let go" not in clip[0]["prompt"], [note(s) for s in clip])
    moves = [run(*b)[0] for b in (("Marcus clips a leash to Sarah's collar and hands the end to Mia.", "Mia leads Sarah to the window."),
                                  ("Marcus clips a chain leash to Sarah's collar.", "Marcus drops the leash and steps back."),
                                  ("Marcus clips a chain leash to Sarah's collar.", "Marcus walks out of the room.\nexit: Marcus",
                                   "Sarah waits by the door."),
                                  ("Marcus chains Sarah's collar to the ring in the wall.", "Marcus unhooks the leash from the ring and "
                                   "leads Sarah to the bed."),
                                  ("Marcus clips a chain leash to Sarah's collar and wraps it around his hand.", "Marcus holds the end of "
                                   "the bed."))]
    free = ("Its other end hangs free: the chain runs from her throat across her chest to one side of her and drops beside her to the "
            "floor, where its loose end lies.")
    check("the other end of a leash is handed over, dropped, let go of by whoever leaves and taken off a ring, and wrapping it round "
          "a hand never puts a leash on him",
          all(note(s).startswith("Its other end is in red-haired Mia's hand") for s in moves[0])
          and note(moves[1][1]) == free and "lets go of the other end of blonde-haired Sarah's leash" in moves[1][1]["prompt"]
          and "lets go of the other end of blonde-haired Sarah's leash" in moves[2][1]["prompt"] and note(moves[2][2]) == free
          and "unfastens the other end of blonde-haired Sarah's leash from the ring in the wall and takes it in his hand"
              in moves[3][1]["prompt"] and note(moves[3][1]).startswith("Its other end is in dark-haired Marcus's hand")
          and "Marcus: a chain leash" not in moves[4][0]["line"] and "his collar" not in moves[4][0]["prompt"]
          and all(note(s).startswith("Its other end is in dark-haired Marcus's hand") for s in moves[4]),
          [[note(s) for s in m] for m in moves])
    kept = run("Marcus pulls Sarah by the leash.", m="hold: Sarah, a black leather collar with a chain leash")
    sheet = P.plan("A dim bedroom.\n\nMarcus pulls Sarah by the leash.", mem.replace("pleated skirt.", "pleated skirt, a black leather "
                                                                                    "collar with a chain leash."))
    check("a collar with a leash, written in a hold line or in the character's own description, is a collar and a leash, and the leash "
          "is read when someone takes it",
          "on: Sarah: a black leather collar., goes on: Sarah: a chain leash clipped to the collar around her neck, its other end held by "
          "Marcus." in kept[0][0]["line"]
          and note(kept[0][0]).startswith("Its other end is in dark-haired Marcus's hand") and not kept[1]
          and note(sheet[0][0]).startswith("Its other end is in dark-haired Marcus's hand") and not sheet[1],
          [kept[0][0]["line"], kept[1], sheet[1]])
    tie = "Marcus clips a chain leash to Sarah's collar and fastens the other end to the radiator."
    cuffs = run(tie, "Marcus takes a pair of handcuffs out of the drawer.", "Marcus walks around Sarah.",
                "Marcus cuffs Sarah's wrists behind her back.", "Marcus unhooks the chain from the radiator.")[0]
    back = run(tie, "Marcus shows Sarah a pair of handcuffs and puts them back in the drawer.", "Marcus walks around Sarah.")[0]
    loose = "The handcuffs in this shot are a separate, small pair of their own: two rigid steel cuffs joined to each other by their own " \
            "three short, rigid steel links, which hold them together, side by side, every moment."
    check("handcuffs brought out near a fastened leash are their own separate pair until they go on or are put away, the leash stays "
          "fastened while they lock on, and unhooking the chain from where it is fastened takes it off there",
          [loose in s["prompt"] for s in cuffs + back] == [False, True, True, False, False, False, True, False]
          and "Dark-haired Marcus takes a separate pair of small steel handcuffs in his hand, steps behind blonde-haired Sarah, takes "
              "both of her wrists and draws her arms back behind her, then closes one cuff around her left wrist with a click, then the "
              "other around her right wrist, there at the small of her back, where her body hides her hands from the front."
              in cuffs[3]["prompt"]
          and all(note(s).startswith("Its other end is padlocked to the radiator") for s in cuffs[:4])
          and "unfastens the other end of blonde-haired Sarah's leash from the radiator" in cuffs[4]["prompt"]
          and note(cuffs[4]).startswith("Its other end is in dark-haired Marcus's hand"),
          [[note(s), loose in s["prompt"]] for s in cuffs + back])
    drawer = run(tie, "Marcus opens the top drawer of the nightstand and takes out a crop.", "Marcus walks around Sarah.",
                 "Marcus closes the drawer.", "Sarah kneels.")[0]
    wardrobe = run("Marcus clips a chain leash to Sarah's collar.", "Marcus opens the wardrobe.", "Marcus kicks the wardrobe door shut.")[0]
    moved = [next((t for t in re.split(r"(?<=[.!?])\s+", s["prompt"]) if " moves " in t), "") for s in drawer + wardrobe]
    path = "blonde-haired Sarah's leash stays one unbroken chain from the ring at her collar to the radiator, and if the top drawer " \
           "meets it, it only pushes the slack aside."
    check("a drawer, door or lid opened near a fastened or held leash moves only itself, while it opens, stays open and closes, and the "
          "leash stays one unbroken chain between its two ends",
          moved[:5] == ["", f"Opening the top drawer moves only the top drawer: {path}",
                        f"The top drawer stays open and moves nothing else: {path}",
                        f"Closing the top drawer moves only the top drawer: {path}", ""]
          and moved[6].startswith("Opening the wardrobe moves only the wardrobe: blonde-haired Sarah's leash stays one unbroken chain from "
                                  "the ring at her collar to dark-haired Marcus's hand")
          and moved[7].startswith("Closing the wardrobe") and moved[5] == "", moved)
    said = lambda s: next((t for t in re.split(r"(?<=[.!?])\s+", s["prompt"]) if t.startswith(("Between them", "Pulled taut"))), "")
    walk = run(tie, "Marcus walks slowly around Sarah.", "Sarah strains against the chain.", "Sarah kneels.")[0]
    hook = run("Marcus clips a chain leash to Sarah's collar and fastens the other end to a hook in the ceiling.")[0]
    check("a fastened chain leash lies slack along the floor for anyone to step over, runs taut only while the beat pulls it tight, "
          "and hangs down from a ceiling hook",
          [said(s).split(":")[0] for s in walk] == ["Between them the chain hangs slack in one loose curve"] * 2 + [
              "Pulled taut, the chain runs in a straight line from her throat to the radiator",
              "Between them the chain hangs slack in one loose curve"]
          and "steps over it" in walk[1]["prompt"] and "padlocks its other end to the radiator" in walk[0]["prompt"]
          and said(hook[0]) == "Between them the chain hangs down from the hook in the ceiling to her throat in a slack curve.",
          [said(s) for s in walk + hook])
    post = run("Marcus cuffs Sarah's wrists to the bedpost.", "Sarah tugs at the cuffs.", "cut\nSarah waits.")[0]
    point = [next((t for t in re.split(r"(?<=[.!?])\s+", s["prompt"]) if "fixed point" in t), "") for s in post]
    cuffed = "Her handcuffs are locked to the bedpost at one fixed point on it"
    check("wrists or ankles fastened to anything stay at one fixed point on it, the point it holds in the frame each continued shot "
          "opens on", point == [f"{cuffed}, and that point stays the same from the moment they are on.",
                                f"{cuffed}, the same point as in the frame this shot opens on, for the whole shot.",
                                f"{cuffed} for the whole shot."], point)


def test_kitchen():
    print("\n=== everyday objects follow what is done with them ===")
    mem = ("Sarah: a woman in her twenties with long blonde hair, in a white blouse and a short black pleated skirt.\n"
           "Dana: a woman in her thirties with short black hair, in a black blazer and grey trousers.\n")
    run = lambda *b: P.plan("A small kitchen with a sink under the window and a long counter beside it.\n\n" + "\n\n".join(b), mem)[0]
    said = lambda s, w: [t for t in re.split(r"(?<=[.!?])\s+", s["prompt"]) if w in t]
    plates = run("Sarah puts the dirty plates on the counter and washes them in the sink.", "Sarah dries her hands on a towel.")
    pan = run("Sarah picks up a dirty pan and scrubs it in the sink.", "Sarah sets the pan on the drying rack.", "Sarah wipes her hands.")
    handed = run("Sarah takes a plate from the counter and washes it in the sink.", "Sarah hands the plate to Dana.")
    sink = run("Sarah bends over the sink to wash the dishes.", "Dana watches from the doorway.")
    one = [said(x, "There is only") for x in plates + run("Sarah walks in.") + P.plan("A hotel bathroom with two sinks and a bathtub.\n\n"
                                                                                     "Sarah washes her face.", mem)[0]]
    check("the place's one sink, counter, bed or desk is said to be the only one, always in the same spot, and nothing written in the "
          "plural is", one == [["There is only one sink and one counter here, each always in the same spot."]] * 3
          + [["There is only one bathtub here, always in the same spot."]], one)
    check("an object goes where the beat puts it next, a washed thing is not kept dirty, an object the beat moves is only said to start "
          "where it was, and bending over a sink is not lying across it",
          said(plates[1], "plates") == ["The plates are still in the sink, looking exactly as they did."]
          and said(pan[1], "pan") == ["Sarah sets the pan on the drying rack.", "The pan is in the sink as the shot begins."]
          and said(pan[2], "pan") == ["The pan is still on the drying rack, looking exactly as it did."]
          and said(handed[1], "plate") == ["Sarah hands the plate to Dana.", "The plate is in the sink as the shot begins."]
          and said(sink[1], "bent") == ["Blonde-haired Sarah stays bent over the sink for the whole shot, holding that position from the "
                                        "first frame to the last."], [said(plates[1], "plates"), said(pan[1], "pan"),
                                                                      said(handed[1], "plate"), said(sink[1], "bent")])


def test_fixes5():
    print("\n=== leashes, timing, toys and the look of restraints ===")
    mem = ("Sarah: a woman in her twenties with long blonde hair, in a white tank top and denim shorts.\n"
           "Dana: a woman in her thirties with short black hair, in a black blazer and grey trousers.\n")
    last = lambda b, m="": P.plan("A dim basement.\n\nDana cuffs Sarah's hands behind her back.\n\n" + b + "\n\nSarah turns around.",
                                  mem + m)[0][-1]
    collar = "hold: Sarah, a black leather collar"
    leashes = [last(b, m) for b, m in (("Dana puts a collar on Sarah and chains it to the wall.", ""),
                                       ("Dana attaches a chain to Sarah's collar and padlocks it to the radiator.", collar),
                                       ("Dana hooks a chain leash onto Sarah's collar and loops the end around the bedpost.", collar),
                                       ("Dana fastens the leash to a hook on the wall.",
                                        collar + "\nhold: Sarah, a chain leash clipped to her collar"),
                                       ("Dana snaps a leash onto Sarah's collar and hands the end to Mia.", collar))]
    check("a chain from a collar is a leash from her collar, however it is written, and never a chain on her wrists or a collar round "
          "her wrists",
          [re.search(r"leash clipped to (?:the collar around her neck|her collar)(?:, fastened to ([^;.]+))?", x["line"]).group(1)
           for x in leashes] == ["the wall", "the radiator", "the bedpost", "the hook on the wall", None]
          and not any(re.search(r"chains around her wrists|collar around her wrists|The chains on", x["prompt"]) for x in leashes)
          and P.parts("a chain leash clipped to her collar") == {"neck"}, [x["line"] for x in leashes])
    gagged = mem + "hold: Sarah, a ball gag in her mouth"
    cues = [P.plan("A cellar.\n\n" + b, gagged)[0][-1]["cues"][0][0] for b in (
        "Dana raises the riding crop and brings it down across Sarah's thighs, and Sarah screams into her gag.",
        "Sarah screams into her gag as Dana brings the crop down.")]
    told = P.plan("A cellar.\n\nDana slaps Sarah across the face, and Sarah gasps, \"Please stop.\"", mem)[0][-1]
    check("a reaction waits for what causes it: a scream after a blow in the same sentence, or one written 'as' the blow lands, starts "
          "after the blow, and the picture is told every reaction comes after its cause",
          all(c > 1.5 for c in cues) and told["cues"][0][0] > 1.5
          and "every reaction comes only after what causes it" in told["prompt"], (cues, told["cues"]))
    toys = P.plan("A hotel room.\n\nDana takes a pink wand vibrator out of her bag and switches it on.\n\nDana presses the wand against "
                  "Sarah's thigh and Sarah gasps.\n\nDana clips a pair of nipple clamps with a chain between them to Sarah's nipples."
                  "\n\nDana steps back and watches.\n\nDana bends Sarah over the bed and picks up a wooden paddle.\n\nDana spanks "
                  "Sarah.", mem)[0]
    kept = [[t for t in re.split(r"(?<=[.!?])\s+", x["prompt"]) if "looking exactly as" in t or "as the shot begins" in t] for x in toys]
    check("toys and bondage gear keep their look and place: the wand is the same vibrator still in her hand, clamps stay on, and an "
          "object taken up after bending someone over is in the doer's hand",
          any(re.search(r"The vibrator is the same pink wand vibrator as before, in (?:black-haired )?Dana's hand as the shot begins", t)
              for t in kept[1])
          and "The pair of nipple clamps with a chain are still on Sarah's nipples, looking exactly as they did." in kept[3]
          and not any("vibrator" in t for t in kept[3])
          and any(re.search(r"wooden paddle is still in (?:black-haired )?Dana's hand", t) for t in kept[5]), kept)
    looks = {b: P.plan("A room.\n\n" + b, mem)[0][-1] for b in (
        "Dana pushes a red ball gag into Sarah's mouth and buckles it behind her head.", "Dana buckles black leather cuffs around Sarah's "
        "wrists behind her back.", "Dana ties Sarah's wrists with red rope.",
        "Dana fastens padded leather ankle cuffs around Sarah's ankles.")}
    items = [re.search(r"goes on: Sarah: (.*?)\.(?:,|$)", x["line"]).group(1) for x in looks.values()]
    soft = list(looks.values())
    check("a restraint keeps its colour and material, and soft cuffs are buckled leather, never rigid steel handcuffs",
          items == ["a red ball gag in her mouth", "black leather cuffs behind her back", "red rope around her wrists behind her back",
                    "padded leather cuffs on her ankles"]
          and "rigid steel" not in soft[1]["prompt"] and "rigid steel" not in soft[3]["prompt"]
          and "buckles one cuff around blonde-haired Sarah's left wrist, then the other around her right wrist, and clips them together"
              in soft[1]["prompt"],
          items)


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
    check("every audio-only pass is given the shot's own sound, never the picture's text with its character sheets and notes",
          all(c["text"].startswith("integrated_multimodal_description: [Shot 1] A room.") and c["text"].endswith("non_diegetic_music: N/A")
              and " hair" not in c["text"] for c in calls["sample"] if c["audio_only"])
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
          len(muffled) == 3 and muffled[0]["heard"] == 0 and all(c["audio_only"] for c in muffled[1:]) and 0 < muffled[1]["heard"] < 0.5
          and abs(muffled[1]["heard"] + muffled[2]["heard"] - 1) < 0.02, muffled)
    check("the scream's own window and the rest of its shot share one sound, with her muffled sounds and none of her words",
          muffled[1]["text"] == muffled[2]["text"] and "Sarah lets out muffled, closed-mouth mmmph sounds through the" in muffled[1]["text"]
          and "Help" not in muffled[1]["text"], muffled[1]["text"])
    voice = torch.tensor([0.95, 0.7, 0.3, 0.0])
    out, calls = render("A room.\n\nDana waits.\n\n\"Sit,\" Dana says.", fine=(torch.linspace(1, 0, 9), voice))
    talk = [c for c in calls["sample"] if c["shot"] == 2]
    check("on a few-step schedule each spoken line is said again over its own window from part-way down a finer schedule, with the "
          "shot's own text, and the sound passes run on the finer schedule from the start",
          [c["steps"] for c in calls["sample"] if c["shot"] == 1] == [8, 8] and [c["audio_only"] for c in talk] == [False, True, True]
          and abs(talk[1]["start"] - 0.95) < 1e-6 and talk[1]["steps"] == 3 and talk[1]["text"] == talk[0]["text"]
          and "<d>Sit</d>" in talk[1]["text"]
          and 0 < talk[1]["heard"] < 0.5 and talk[2]["start"] == 1.0 and talk[2]["steps"] == 8
          and abs(talk[1]["heard"] + talk[2]["heard"] - 1) < 0.02 and talk[2]["text"].startswith("integrated_multimodal_description"),
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
    test_tags()
    test_reading()
    test_speech()
    test_hands()
    test_sex()
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
    test_flash()
    test_chain()
    test_overlay()
    test_ankles()
    test_positions()
    test_worn()
    test_undress_bound()
    test_skirts()
    test_anchors()
    test_marks()
    test_belt()
    test_apart()
    test_links()
    test_holds()
    test_fixes5()
    test_leash_ends()
    test_kitchen()
    test_render()
    print("\nRESULT: " + (f"{len(_fails)} FAILURE(S): " + "; ".join(_fails) if _fails else "ALL PASSED"))


if __name__ == "__main__":
    main()
