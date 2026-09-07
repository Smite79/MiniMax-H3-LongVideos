"""Smoke test: run the whole render path with fake models.

Nothing here touches a GPU, loads a checkpoint or asks the real model for
anything -- the model, CLIP and both VAEs are stubs that return correctly shaped
tensors. What it exercises is the code the node runs on every render: the shot
loop, conditioning assembly, the keyframe handoff, decode, trim and concat.

This exists because a missing module-level name shipped once and only showed up
as a traceback on the user's first render. Parsing and unit-testing pure
functions did not catch it; running the path does.

Run: python test_smoke.py
"""

import importlib.util
import io
import os
import itertools
import re
import sys
import types

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# --- stub the ComfyUI surface the node imports ------------------------------
for _n in ("nodes", "comfy", "comfy.utils", "comfy.sample", "comfy.samplers",
           "comfy.nested_tensor", "comfy.model_management", "latent_preview", "node_helpers"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
_c = sys.modules["comfy"]
for _sub in ("utils", "sample", "samplers", "nested_tensor", "model_management"):
    setattr(_c, _sub, sys.modules["comfy." + _sub])

sys.modules["comfy.samplers"].KSampler = type("K", (), {"SAMPLERS": ["res_multistep"],
                                                        "SCHEDULERS": ["simple"]})
sys.modules["comfy.samplers"].sampler_object = lambda name: object()


class FakeNested:
    """Stands in for comfy.nested_tensor.NestedTensor: a (video, audio) pair."""
    is_nested = True

    def __init__(self, parts):
        self.parts = tuple(parts)

    def unbind(self):
        return self.parts


sys.modules["comfy.nested_tensor"].NestedTensor = FakeNested

_mm = sys.modules["comfy.model_management"]
_mm.intermediate_device = lambda: torch.device("cpu")
_mm.get_torch_device = lambda: torch.device("cpu")
_mm.free_memory = lambda *a, **k: None
_mm.soft_empty_cache = lambda *a, **k: None
_mm.unload_all_models = lambda *a, **k: None
_mm.loaded_models = lambda *a, **k: []

sys.modules["comfy.utils"].PROGRESS_BAR_ENABLED = False
sys.modules["comfy.utils"].common_upscale = (
    lambda s, w, h, method, crop: torch.nn.functional.interpolate(s, size=(h, w)))
sys.modules["latent_preview"].prepare_callback = lambda *a, **k: None
sys.modules["node_helpers"].conditioning_set_values = (
    lambda cond, vals: [[c[0], {**c[1], **vals}] for c in cond])

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("h3smoke", os.path.join(_HERE, "sampler.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

_fails = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        _fails.append(label)


# --- fakes ------------------------------------------------------------------

W, H, FRAMES = 128, 96, 39            # small, on the 17k+5 grid
# A scene and one wordless beat. Kept as a constant because it is checked twice, with
# the mouth guard on and off, and the two spellings must not drift apart.
TWO_LINE_ROOM = "A room.\n\nHe walks in."


class FakeCLIP:
    def __init__(self):
        self.seen = []

    def tokenize(self, prompt, minimax_ref_items=None):
        self.seen.append((prompt, list(minimax_ref_items or [])))
        return {"prompt": prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 8, 16), {}]]


class FakeVAE:
    upscale_ratio = (4, 8, 8)
    latent_dim = 3

    def __init__(self):
        self.encodes = 0

    def encode(self, image):
        self.encodes += 1
        n = image.shape[0]
        return torch.zeros(1, 24, max(1, n), H // 16, W // 16)

    def decode(self, latent):
        t = latent.shape[2] if latent.ndim == 5 else 1
        return torch.rand(t * 4, H, W, 3)

    def decode_tiled(self, latent, **kw):
        return self.decode(latent)


class FakeAudioVAE:
    upscale_ratio = 512
    latent_dim = 2
    audio_sample_rate = 32000
    audio_sample_rate_output = 44100

    class _M:
        latents_mean = torch.zeros(32)
        latents_std = torch.ones(32)

    first_stage_model = _M()

    def encode(self, wav):
        return torch.zeros(1, 32, 2, 16)

    def decode(self, latent):
        # Real layout is [B, L, C]; _decode_audio movedim's it to [B, C, L].
        n = latent.shape[-1] if latent.ndim >= 2 else 16
        return torch.rand(1, n * 800, 2)


class FakeModel:
    def clone(self):
        return self

    def get_model_object(self, name):
        class MS:
            def set_parameters(self, shift=1.0, **kw):
                self.shift = shift
        return MS()

    def add_object_patch(self, *a, **k):
        pass

    model = types.SimpleNamespace(model_config=types.SimpleNamespace(__class__=object))


def fake_ksampler(model, seed, steps, cfg, sn, sch, positive, negative, latent, denoise=1.0):
    """Return a latent shaped exactly as _empty_av_latent built it."""
    v, a = latent["samples"].unbind()
    out = dict(latent)
    out["samples"] = FakeNested((v.clone(), a.clone()))
    return (out,)


sys.modules["nodes"].common_ksampler = fake_ksampler


def run_node(prompt, **kw):
    node = S.H3LongVideos()
    args = dict(model=FakeModel(), clip=FakeCLIP(), vae=FakeVAE(), audio_vae=FakeAudioVAE(),
                prompt=prompt, resolution="4:3", megapixels=0.0,
                shot_seconds=FRAMES / S.H3_FPS, steps=2, cfg=1.0,
                sampler_name="res_multistep", scheduler="simple", seed=1,
                apply_model_sampling=False, tiled_decode=False)
    args.update(kw)
    # the preset is 1024x768; force the small canvas the fakes are built for
    S.NATIVE_RES["4:3"] = (W, H)
    return node.run(**args)


def test_plan():
    print("\n=== plan_only ===")
    imgs, audio, info, script, fps_shot, total, shots, secs = run_node(
        "A room.\n\nHe walks in.\n\nShe follows.", plan_only=True)
    check("it reports without rendering", "PLAN ONLY" in info)
    check("it counts the shots", shots == 2, f"{shots}")
    check("the script shows both shots", script.count("[Shot ") == 2)
    check("the scene is prepended to each", script.count("A room.") == 2)


def test_render():
    print("\n=== full render path ===")
    try:
        imgs, audio, info, script, per_shot, total, shots, secs = run_node(
            "A room.\n\nHe walks in.\n\nShe follows him and says: \"Wait.\"")
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("the render path runs end to end", False, f"{type(e).__name__}: {e}")
        return
    check("the render path runs end to end", True)
    check("frames come back", imgs.ndim == 4 and imgs.shape[0] > 0, str(tuple(imgs.shape)))
    check("audio comes back", audio["waveform"].ndim == 3)
    check("two shots were rendered", shots == 2, str(shots))
    check("the seam frame is trimmed from shot 2",
          imgs.shape[0] == total and total > 0)
    check("info is populated", bool(info))


def test_keyframe_handoff():
    print("\n=== the keyframe is encoded, once per boundary ===")
    vae = FakeVAE()
    run_node("A room.\n\nOne.\n\nTwo.\n\nThree.", vae=vae)
    # Shot 1 has no previous frame; shots 2-4 each encode the frame handed to them.
    # Passing the previous shot's own latent instead was tried and was wrong: a
    # keyframe is ONE pixel frame, which lands on H3's 5f grid point and encodes to
    # TWO latent frames, and a causal encoder's last latent is not a standalone
    # opening frame anyway. It degraded every shot after the first.
    # 3 shots = 2 boundaries; shot 1 has no previous frame and no first_frame here.
    check("one encode per boundary, none for shot 1", vae.encodes == 2,
          f"{vae.encodes} encodes")
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    check("no latent is smuggled in as a keyframe", "handoff_latent" not in src)


def test_references_and_silence():
    print("\n=== references and silence ===")
    clip = FakeCLIP()
    run_node("A room.\n\nOne.\n\nTwo.", clip=clip,
             ref_image_1=torch.rand(1, 64, 64, 3))
    # seen[0] is the empty negative prompt, encoded once before the loop.
    shots_seen = clip.seen[1:]
    # No <Picture N> tag anywhere, so placing by tag would place the reference on no
    # shot at all -- a connected input that silently does nothing. It falls back to
    # every shot instead. Shot 1 carries the reference alone (no previous frame yet);
    # shot 2 carries the reference AND its keyframe.
    check("an untagged reference still reaches every shot",
          [len(i) for _, i in shots_seen] == [1, 2],
          str([len(i) for _, i in shots_seen]))
    clip2 = FakeCLIP()
    # auto_sound OFF: with it on, an ambient bed read from the scene opens every
    # branch, and this is the silence path.
    run_node("A room.\n\nHe walks in.\n\nShe says: \"Now.\"", clip=clip2,
             auto_sound=False)
    check("both shots reached the encoder", len(clip2.seen) == 3)   # + the negative
    check("the beat text is what was sent",
          "He walks in." in clip2.seen[1][0] and "Now." in clip2.seen[2][0])
    # auto_sound appends a sound sentence -- "He walks in." implies footsteps. Your
    # words are still never rewritten; the node only ever adds after them, and every
    # addition has a switch.
    # The beat has no line and no sound of its own, so it is pinned silent and gets no
    # sound sentence: a clause would describe an acoustic the conditioning removes,
    # and a free branch there is what put a voice in the mouth.
    # It carries the mouth clause and nothing else: no sound sentence, because a
    # clause would describe an acoustic the conditioning removes. The mouth clause is
    # the picture half of the same guarantee and has its own switch.
    check("...with no sound sentence added to a silenced shot",
          clip2.seen[1][0] == "A room. He walks in." + S.MOUTH_HOLD, clip2.seen[1][0])
    clip2b = FakeCLIP()
    run_node(TWO_LINE_ROOM, clip=clip2b, mouths_shut_when_no_line=False,
             auto_sound=False)
    check("...and none at all with the mouth guard off",
          clip2b.seen[1][0] == "A room. He walks in.", clip2b.seen[1][0])
    # The shot that DOES speak gets the open form: closing the list there would be
    # telling the model the line is not in it.
    clip4 = FakeCLIP()
    run_node("A room.\n\nShe walks in and says: \"Now.\"", clip=clip4)
    # The GUARANTEE, not the exact sentence: the sound clause is the open form,
    # and the beat and its line are there untouched. Asserting the whole string
    # made this fail every time an unrelated clause was added beside it.
    check("a shot with a line is not told that is all there is",
          "It sounds like" in clip4.seen[1][0]
          and "footsteps" in clip4.seen[1][0]
          and "the only sound" not in clip4.seen[1][0]
          and "She walks in and says:" in clip4.seen[1][0]
          and "Now." in clip4.seen[1][0], clip4.seen[1][0])
    clip3 = FakeCLIP()
    run_node("A room.\n\nHe walks in.", clip=clip3, auto_sound=False)
    check("...and with that off only the mouth clause remains",
          clip3.seen[1][0] == "A room. He walks in." + S.MOUTH_HOLD, clip3.seen[1][0])


def test_first_frame():
    print("\n=== first_frame anchors shot 1 ===")
    vae = FakeVAE()
    run_node("A room.\n\nOne.\n\nTwo.", vae=vae, first_frame=torch.rand(1, H, W, 3))
    # 2 shots: shot 1 encodes the supplied first_frame, shot 2 encodes the handoff.
    check("a supplied first frame is encoded as shot 1's keyframe", vae.encodes == 2,
          f"{vae.encodes}")
    bare = FakeVAE()
    run_node("A room.\n\nOne.\n\nTwo.", vae=bare)
    check("...and without one, shot 1 encodes nothing", bare.encodes == 1, f"{bare.encodes}")


def test_aug_protects_the_keyframe():
    print("\n=== ref_noise_aug must not corrupt the keyframe ===")
    # comfy/ldm/minimax/model.py applies visual_cond_noise_aug to BOTH the
    # reference rows and the keyframe rows: below 1.0 it noises the keyframe
    # latent and labels it max(t_v, aug) instead of 0.999. Softening references
    # would therefore corrupt the anchor of every shot after the first, and that
    # shows up during SAMPLING.
    P = "A room.\n\nOne.\n\nTwo."
    ref = torch.rand(1, 64, 64, 3)
    vae_hi = FakeVAE()
    info_hi = run_node(P, vae=vae_hi, ref_image_1=ref, ref_noise_aug=0.999)[2]
    check("at a safe aug the handoff is a real keyframe",
          "riding as an extra reference" not in info_hi)
    check("...and it is encoded", vae_hi.encodes >= 1, f"{vae_hi.encodes}")
    # One aug covers every visual condition row. Below KEYFRAME_SAFE_AUG it would
    # noise the keyframe too, so there the handoff stops anchoring and rides as an
    # extra reference: weaker continuity, but nothing pretending to anchor.
    info_lo = run_node(P, ref_image_1=ref, ref_noise_aug=0.90)[2]
    check("a soft aug demotes the keyframe rather than noising it",
          "riding as an extra reference" in info_lo)
    check("...and the note says what it costs", "weaker continuity" in info_lo)
    # With nothing connected there is no reference to soften, so the keyframe is safe
    # whatever the widget says.
    info_noref = run_node(P, ref_noise_aug=0.90)[2]
    check("no references means the keyframe is never demoted",
          "riding as an extra reference" not in info_noref)


def test_beat_reviving_a_garment():
    print("\n=== a later beat that names a removed garment ===")
    # Beats go to the model word for word. The scene can be perfectly scrubbed
    # and the removal honoured, and then beat 5 says "her coat" and it is
    # back. Ten beats in, that reads as the removal randomly failing.
    P = ("A basement. Kate is 20, blonde, grey wool coat, black jumper." + "\n\n"
         "Dan pulls off her coat.\nremove: coat" + "\n\n"
         "Kate lies still." + "\n\n"
         "Dan pulls at her coat again." + "\n\n"
         "Kate turns her head.")
    info = M_info = run_node(P)[2]
    check("the reviving beat is named", "shot 3 names coat" in info)
    check("...with the reason", "word for word" in info)
    check("...and the removing beat itself is not flagged",
          "shot 1 names coat" not in info)
    clean = run_node("A basement. Kate is 20." + "\n\n" +
                     "Dan pulls off her coat.\nremove: coat" + "\n\n" +
                     "Kate lies still.")[2]
    check("a clean script raises nothing", "in its own text" not in clean)


def test_restart_after_removal():
    print("\n=== a shot after a removal starts fresh ===")
    # Every shot is anchored to the previous shot's last frame. If the model
    # does not finish taking the garment off inside its own shot, that frame
    # still shows it -- and a keyframe is a PICTURE, which outvotes any
    # sentence. Inherit it once and every later shot inherits it too.
    P = ("A basement. Kate is 20, blonde, grey wool coat, black jumper." + "\n\n"
         "Dan pulls off her coat.\nremove: coat" + "\n\n"
         "Kate lies still." + "\n\n"
         "Kate breathes.")
    vae_on = FakeVAE()
    info_on = run_node(P, vae=vae_on, restart_after_removal=True)[2]
    check("the shot after the removal is named", "shot(s) 2 start fresh" in info_on)
    check("...with the reason", "picture outvotes the text" in info_on)
    check("...and the cost", "costs a cut" in info_on)
    vae_off = FakeVAE()
    info_off = run_node(P, vae=vae_off, restart_after_removal=False)[2]
    check("off, nothing restarts", "start fresh" not in info_off)
    # A dropped keyframe is one fewer frame to encode.
    check("the fresh shot encodes no keyframe", vae_on.encodes < vae_off.encodes,
          f"{vae_on.encodes} vs {vae_off.encodes}")
    check("a script with no removals is unaffected",
          "start fresh" not in run_node("A room.\n\nOne.\n\nTwo.")[2])


def test_auto_removal():
    print("\n=== removals read from the beat, with no directives ===")
    P = ("A basement. Kate is 20, blonde, grey jumper, wool scarf, black boots."
         + "\n\n" +
         "Dan pulls off her boots and throws them away, showing the wool scarf."
         + "\n\n" +
         "Dan pulls off her scarf and throws it away." + "\n\n" +
         "Kate lies still.")
    script = run_node(P)[3]
    sh = script.split("\n---\n")
    # Shot 1 takes the boots off and has NO keyframe, so the text is the only thing
    # saying they were on to start with and it keeps saying so. Gone from shot 2.
    check("the removing shot still says the boots are on", "black boots" in sh[0])
    check("...and shot 2 has lost them", "black boots" not in sh[1])
    # The scarf is under the boots -- read from "showing the wool scarf" -- so it is
    # not described as visible until they come off. Only the scene text counts here;
    # the beat names it either way, because beats go to the model word for word.
    _scene1 = sh[0].split("Dan pulls")[0]
    check("the covered scarf is not described yet", "wool scarf" not in _scene1)
    check("...and is once the boots are off", "wool scarf" in sh[1])
    check("shot 3 keeps neither", "boots" not in sh[2] and "scarf" not in sh[2])
    check("...and keeps what was never taken off", "grey jumper" in sh[2])
    info = run_node(P)[2]
    check("info says what it read", "read 'boots' as coming off" in info)
    # Off, nothing is inferred and the warning comes back instead.
    info_off = run_node(P, auto_remove=False)[2]
    check("auto_remove off infers nothing", "as coming off" not in info_off)
    check("...and warns instead", "no 'remove:' line" in info_off)


def test_fall_keeps_the_hardware():
    print("\n=== a fall does not open the cuffs ===")
    P = ("A bare cellar. Kate, 24, in a grey coat.\n\n"
         "Dan cuffs her wrists behind her back.\n\n"
         "Kate loses her balance and falls onto the floor.\n\n"
         "Kate lies still while Dan walks out.")
    blocks = [b for b in re.split(r"(?=\[Shot )", run_node(P, plan_only=True)[3])
              if b.strip()]
    check("three shots planned", len(blocks) == 3)
    fall = S.FALL_HOLD.strip()
    # The hold latches: once a restraint goes on it is held for the rest of the run.
    # Cuffs are rigid hardware, so the chain clause carries the guarantee here -- it
    # says "whole and closed" itself, and emitting both would say it twice.
    #
    # Shot 1 is the shot that PUTS them on, and it gets both ends instead of the
    # standing hold. The standing hold asserts the cuffs are fastened as they were put
    # on and still fastened at the last frame, which read at frame 1 says they are
    # already closed -- so they close first and the catching happens around them.
    check("the applying shot is told both ends",
          S.RESTRAINT_GOING_ON.strip() in blocks[0], blocks[0][-90:])
    check("...and keeps the metal rigid while it goes on",
          S.CHAIN_RIGID_TAIL.strip() in blocks[0], "")
    check("...and is not also told it is already fastened",
          "closed and fastened as" not in blocks[0], "")
    for i, b in enumerate(blocks[1:], 2):
        check(f"shot {i} holds the restraint",
              "closed and fastened as" in b)
    # The fall clause is per-beat -- it only earns its tokens where a body goes down.
    check("the fall beat says what takes the landing", fall in blocks[1])
    check("...the beat that puts them on does not", fall not in blocks[0])
    check("...and neither does lying still afterwards", fall not in blocks[2])
    # No restraint anywhere in the prompt: a fall is just a fall, nothing to protect.
    loose = run_node("A bare cellar. Kate, 24.\n\nKate trips and falls.",
                     plan_only=True)[3]
    check("an unbound fall adds nothing", fall not in loose)


def test_anchor_is_the_scene():
    print("\n=== an anchor makes every paragraph a beat ===")
    # Reported as "the character memory is not being passed" and "the first line is
    # ignored". One cause: with the framing moved into the anchor, the prompt starts
    # with an ACTION -- and paragraph 1 was still being taken as the scene. That beat
    # was then prepended to every shot, repeated to the end of the film, never given
    # a shot of its own, and any removal in it could never stick because the scene
    # restated the garment on every later shot.
    P = "Jon walks in and takes her scarf off.\n\nMaya lies still.\n\nJon leaves."
    sh = [x for x in re.split(r"(?=\[Shot )", run_node(
        P, plan_only=True, anchor="Wide lens, night.",
        character_memory="Maya: 27, grey scarf, black boots.")[3]) if x.strip()]
    check("every paragraph gets its own shot", len(sh) == 3)
    check("the first action is not stolen as the scene",
          "Jon walks in" in sh[0] and "Jon walks in" not in sh[1])
    check("the anchor leads every shot", all("Wide lens, night." in s for s in sh))
    check("...and the character sheet follows it",
          all(s.index("Wide lens") < s.index("Maya: 27") for s in sh))
    # Shot 1 removes the scarf and has no keyframe, so it still says the scarf is on
    # -- otherwise the shot reads "it is not worn, take it off". Gone after that.
    check("the removal sticks", all("grey scarf" not in s for s in sh[1:]))
    check("...and what was never removed is still described",
          all("black boots" in s for s in sh))
    # With no anchor, paragraph 1 is the scene exactly as before.
    sh2 = [x for x in re.split(r"(?=\[Shot )", run_node(
        "A basement.\n\nJon walks in.\n\nMaya lies still.", plan_only=True)[3])
        if x.strip()]
    check("no anchor, no change", len(sh2) == 2)
    check("...and the scene still leads", all("A basement." in s for s in sh2))


def test_guard_and_layers_end_to_end():
    print("\n=== the guard and the layers, through the whole path ===")
    P = ("Medium shadows. A basement workshop.\n\n"
         "Maya: 27, blonde hair, grey wool scarf, black quilted jacket, brown boots. "
         "Wrists cuffed behind back. She stays lying on her side on the floor.\n"
         "Jon: 34, navy overalls.\n\n"
         "Maya lies still on the floor, eyes closed.\n\n"
         "Jon walks in and takes her jacket off to expose the scarf.\n\n"
         "Maya lies still.\n\n"
         "Jon walks out and shuts the door.")
    imgs, audio, info, script = run_node(P, plan_only=True)[:4]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("four beats, four shots", len(sh) == 4)
    check("Jon is absent from the shot he is not in", "Jon: 34" not in sh[0])
    check("...and present in the one he is", "Jon: 34" in sh[1])
    check("...and alone once he leaves her behind", "Maya: 27" not in sh[3])
    check("Maya is kept where a pronoun refers to her", "Maya: 27" in sh[1])
    # The scarf is under the jacket, read from the script's own wording.
    check("the covered layer is not described", "grey wool scarf" not in sh[0])
    check("...and appears once the jacket is off", "grey wool scarf" in sh[2])
    check("the jacket is described while it is on", "quilted jacket" in sh[0])
    check("...and gone after it comes off", "quilted jacket" not in sh[2])
    check("info names the layering", "scarf under jacket" in info)
    check("...and the opening-pose warning", "opening pose comes from the text" in info)
    # Off, every sheet line goes into every shot, as before.
    off = [x for x in re.split(r"(?=\[Shot )", run_node(P, plan_only=True,
           character_guard=False)[3]) if x.strip()]
    check("guard off puts everyone in every shot",
          all("Jon: 34" in s and "Maya: 27" in s for s in off))


def test_removing_shot_without_a_keyframe():
    print("\n=== a removal in a shot that has no keyframe ===")
    # Reported: the garment rendering half-off -- opened up, with the layer under it
    # showing. The scrub applies to the removing shot too, which is right when that
    # shot has a KEYFRAME: the picture already shows the garment on at the start, so
    # text saying it is worn would put it back at the end. Shot 1 has no keyframe.
    # Scrubbing there left the shot saying: it is not worn, take it off, and the
    # thing under it is already showing -- and the model rendered the contradiction.
    P = ("A basement.\n\n"
         "Maya: 27, grey wool scarf, black quilted jacket, brown boots.\n\n"
         "Jon cuts off her jacket to expose the scarf.\n\n"
         "Maya lies still.\n\n"
         "Jon pulls off her scarf.")
    sh = [x for x in re.split(r"(?=\[Shot )", run_node(P, plan_only=True)[3]) if x.strip()]
    check("the removing shot still says the garment is worn",
          "quilted jacket" in sh[0])
    check("...and the layer under it is not yet showing",
          "grey wool scarf" not in sh[0])
    # WHOSE HANDS, where the beat names them. An agentless "the jacket comes off"
    # describes a garment removing itself, which is what a belt dropping to the floor
    # on its own looked like.
    check("...and it is still told to come off",
          "jacket off during this shot" in sh[0] or "jacket comes off during this shot" in sh[0])
    # WHOSE HANDS -- and the limit on that. The agent can only be somebody the SHEET
    # describes, because that is the cast the guard resolves. Jon acts in this beat
    # but has no entry, so the only person in the shot is Maya and the clause names
    # her. Undressing herself is a defensible reading of a prompt that never says who
    # Jon is; naming a person the shot does not describe would be worse, since a
    # described person is a person the model draws.
    check("...by the hands of somebody the sheet describes",
          "Maya takes the black quilted jacket off during this shot" in sh[0],
          sh[0][-120:])
    check("the next shot has lost it", "quilted jacket" not in sh[1])
    check("...and shows what was under it", "grey wool scarf" in sh[1])
    check("info explains the exception", "no keyframe" in run_node(P, plan_only=True)[2])
    # A removing shot that DOES have a keyframe still scrubs in place -- the picture
    # carries the starting state there, so the text does not have to.
    check("a later removing shot still scrubs itself",
          "grey wool scarf" not in sh[2].split("Jon pulls")[0])
    # With a first_frame wired, shot 1 has a keyframe and behaves like the rest.
    sh_ff = [x for x in re.split(r"(?=\[Shot )", run_node(
        P, plan_only=True, first_frame=torch.rand(1, H, W, 3))[3]) if x.strip()]
    check("a wired first_frame restores the normal rule",
          "quilted jacket" not in sh_ff[0].split("Jon cuts")[0])


def test_hardware_anchor_end_to_end():
    print("\n=== a shown collar still belongs on a neck ===")
    P = ("A basement.\n\nMaya: 27, grey coat.\n\n"
         "Jon shows her a collar and leash.\n\n"
         "Jon buckles the collar around her neck.\n\n"
         "Maya walks to the window.")
    imgs, audio, info, script = run_node(P, plan_only=True)[:4]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("the shot that shows it says where it goes",
          "a collar closes around the neck" in sh[0])
    check("...and where the leash goes", "leash clips to the collar" in sh[0])
    check("the shot that places it is left alone",
          "closes around the neck" not in sh[1])
    check("an ordinary beat gets nothing",
          "sits where it belongs" not in sh[2])
    check("info says it stepped in", "no body part beside it" in info)


def test_chain_hold_end_to_end():
    print("\n=== a chain holds its shape through the whole run ===")
    P = ("A basement.\n\n"
         "Maya: 27, grey coat. Wrists cuffed behind back.\n\n"
         "Jon locks a chain around her waist and padlocks it at the back.\n\n"
         "Maya pulls against the chain.\n\n"
         "Maya lies still.")
    sh = [x for x in re.split(r"(?=\[Shot )", run_node(P, plan_only=True)[3]) if x.strip()]
    chain = "links keeping their size"
    check("every shot with the hardware holds it rigid", all(chain in s for s in sh))
    # It REPLACES the restraint hold instead of joining it -- both say "whole and
    # closed", and two clauses for one guarantee is twice the stasis in the prompt.
    check("...instead of repeating the restraint hold",
          all("the run between them" in s for s in sh))
    check("...while still carrying its guarantee",
          all("closed and fastened as it was put on" in s
              or "closed and fastened as they were put on" in s for s in sh))
    # Rope flexes. Saying it holds a straight line would be wrong, so it does not.
    soft = run_node("A basement.\n\nMaya: 27, a rope around her wrists.\n\n"
                    "Maya lies still.", plan_only=True)[3]
    check("rope is not claimed to be rigid", chain not in soft)
    # ...and it is TIED, not closed and fastened. restraint_sentence has said so
    # since it was written -- "rope is not closed and it is not fastened, and
    # saying so of a cord describes a mechanism that is not there" -- but the
    # rope was declared in the SHEET and nothing read the sheet, so no hardware
    # was found and the shot fell through to the generic wording this assertion
    # was pinned to. The engine reads the sheet, so the soft branch now fires.
    # The check keeps its intent: whatever the mechanism is called, it is whole.
    check("...but it is still held whole",
          "tied and holding as" in soft or "closed and fastened as" in soft, soft[-120:])
    check("...and a cord is tied rather than fastened shut",
          "tied and holding as" in soft, soft[-120:])
    # Steel locked on in shot 1 is still steel in shot 5. Tested per shot rather than
    # latched, the shot naming the chain got the rigid clause and every shot after it
    # fell back to the soft one -- which is where the slack came back from.
    later = [x for x in re.split(r"(?=\[Shot )", run_node(
        "A basement.\n\nMaya: 27, grey coat.\n\nJon locks a chain around her waist.\n\n"
        "Maya walks to the window.\n\nMaya looks down.", plan_only=True)[3]) if x.strip()]
    # The shot that locks it on gets both ends rather than the standing clause, but
    # the METAL is rigid throughout -- steel is steel while it is being closed, and
    # letting it go soft on the shot that introduces the object is where the model's
    # idea of the object gets set.
    # The content, not the exact sentence: the applying shot carries it as its own
    # sentence and the standing clause carries it after a semicolon.
    _rigid = "the run between them"
    check("the metal is rigid from the shot that names it",
          all(_rigid in s for s in later))
    check("rigidity latches past the shot that names it",
          all(chain in s for s in later[1:]))
    check("...and that first shot is the one putting it on",
          S.RESTRAINT_GOING_ON.strip() in later[0], later[0][-90:])
    check("...and the soft clause is not used instead",
          all("the run between them" in s for s in later))
    # A position the hardware enforces latches too: the chain that put a body in a
    # squat is still that length three shots later, so the squat is still the position.
    posed = [x for x in re.split(r"(?=\[Shot )", run_node(
        "A basement.\n\nMaya: 27, grey coat. Wrists cuffed behind back.\n\n"
        "Jon locks a chain from her ankles to her collar, forcing her into a squat.\n\n"
        "Maya strains against the chain, trying to stand.\n\nMaya breathes hard.",
        plan_only=True)[3]) if x.strip()]
    check("a forced position keeps for the rest of the run",
          all("drawn to its full length" in s for s in posed))
    check("...replacing the plain chain clause rather than joining it",
          all(chain not in s for s in posed))
    # No position forced: the plain clause, so an unposed chain does not freeze anyone.
    check("a chain with no position forced stays plain",
          all("drawn to its full length" not in s for s in later))
    # Hardware with nothing restrained by it is scenery, not a restraint.
    loose = run_node("A yard with a chain-link fence.\n\nMaya walks past it.",
                     plan_only=True)[3]
    check("scenery does not arm it", chain not in loose)


def test_every_paragraph_accounted_for():
    print("\n=== no beat quietly goes missing ===")
    # Two ways a beat disappears: it reads as a character sheet and is folded into
    # the scene, or it was never a separate paragraph. Both look like beats being
    # absorbed into other beats, and both were silent.
    P = ("A basement.\n\n"
         "Maya: 27, grey coat\nJon: 34, navy overalls\n\n"
         "Maya walks in.\nMaya sits down.\nMaya stands up.\n\n"
         "Jon: pushes the door shut.\n\n"
         "Maya leaves.")
    imgs, audio, info, script = run_node(P, plan_only=True)[:4]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("a labelled ACTION still gets its own shot", len(sh) == 3)
    check("...and is not folded into the scene",
          "pushes the door shut" in sh[1] and "pushes the door shut" not in sh[0])
    check("every paragraph is accounted for", "5 paragraph(s) in the prompt" in info)
    check("...naming what became shots", "3 rendered as shots" in info)
    check("...what was folded in", "2 folded in as character sheet(s)" in info)
    check("...and what became the scene", "1 kept as the scene" in info)
    # Lines joined by a single newline are ONE beat. That is not changed -- it is
    # reported, because three actions in one shot look like two were absorbed.
    check("a merged beat is flagged", "carry more than one line" in info)
    check("...naming the shot", "shot(s) 1 carry" in info)
    check("...and saying what to do", "put an empty line between them" in info)
    clean = run_node("A room.\n\nOne.\n\nTwo.", plan_only=True)[2]
    check("a clean script is not nagged", "carry more than one line" not in clean)


def test_person_described_once_end_to_end():
    print("\n=== two heads: one person described twice ===")
    P = "A basement.\n\nMaya: 27, silver hair, grey coat.\n\nMaya lies still on the floor."
    imgs, audio, info, script = run_node(
        P, plan_only=True, character_memory="Maya: 27, silver hair, grey coat.")[:4]
    shot = " ".join([x for x in re.split(r"(?=\[Shot )", script) if x.strip()][0].split())
    check("the person is described once", shot.count("Maya:") == 1)
    check("...and the duplicate is reported", "described more than once" in info)
    check("...naming who", "Maya described" in info)
    # No terminator on a sheet line welds it onto the beat: "grey coat Maya lies
    # still", which reads as one more item in the attribute list.
    s2 = run_node("A basement.\n\nMaya lies still on the floor.", plan_only=True,
                  character_memory="Maya: 27, silver hair, grey coat")[3]
    check("the sheet does not run into the beat", "grey coat Maya" not in s2)
    check("...it is ended properly", "grey coat. Maya" in s2)
    # Using one channel only is unaffected.
    s3 = run_node(P, plan_only=True)[3]
    check("one channel alone still describes the person", "Maya: 27" in s3)
    check("...once", " ".join(s3.split()).count("Maya: 27") == 1)


def test_undressing_completely_end_to_end():
    print("\n=== undressing takes ALL of it off, and only theirs ===")
    mem = ("Nora: 34, she, tall, red hair, green canvas jacket, grey wool jumper, "
           "white t-shirt, black jeans, brown leather boots.\n"
           "Victor: he, 41, dark hair, navy overalls, tan work shoes")
    P = "\n\n".join(["Nora walks in and sets a toolbox on the bench.",
                     "Nora undresses completely and steps into the shower.",
                     "Nora reaches for a towel.",
                     "Victor walks in carrying a coil of cable."])
    info, script = run_node(P, plan_only=True, anchor="A room.",
                            character_memory=mem)[2:4]
    sh = [" ".join(x.split()) for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("she is dressed to begin with", "green canvas jacket" in sh[0])
    for _g in ("jacket", "jumper", "t-shirt", "jeans", "boots"):
        check(f"{_g} is gone from the undressing shot", _g not in sh[1])
    # It has to STAY off: the scene is re-stamped into every later shot, so a garment
    # left in it is a garment back on.
    check("...and stays gone afterwards",
          not any(g in sh[2] for g in ("jacket", "jumper", "jeans", "boots")))
    # Scoped to the people the beat names. Undressing one person must not take the
    # other one's clothes off.
    check("the other character keeps his clothes", "navy overalls" in sh[3])
    # Said once, rather than reciting the wardrobe back at a shot whose point is that
    # there is none of it.
    check("one sentence, not five", sh[1].count("comes off during this shot") == 1)
    check("...and it is the bare one", "leaving bare skin" in sh[1])
    check("info names what it cleared", "read off the character sheet" in info)
    check("...and lists the garments", "jacket, jumper, t-shirt, jeans, boots" in info)


def test_a_tagged_object_comes_off_and_goes_back_on():
    print("\n=== an object's reference follows the object ===")
    # A <Picture N> can be attached to a THING, not only a person: "a silver locket
    # <Picture 2>" is a picture of the locket. It has to come off when the locket does
    # -- left behind, the reference keeps asserting what was just removed -- and come
    # back when an 'add:' puts it on again.
    mem = "Nora: <picture 1>, 34, she, red hair, a silver locket <picture 2>, green jacket"
    P = "\n\n".join([
        "Nora stands by the bench.",
        "Nora unclasps the silver locket and sets it down.\nremove: locket",
        "Nora looks out of the window.",
        "Nora picks it up again.\nadd: her silver locket <picture 2> is back around her neck",
        "Nora walks to the door."])
    script = run_node(P, plan_only=True, anchor="A workshop.", character_memory=mem,
                      ref_image_1=torch.rand(1, H, W, 3),
                      ref_image_2=torch.rand(1, H, W, 3))[3]
    tags = [re.findall(r"<Picture \d+>", " ".join(x.split()))
            for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("both references start on", tags[0] == ["<Picture 1>", "<Picture 2>"],
          str(tags[0]))
    check("the object's reference goes with the object",
          tags[2] == ["<Picture 1>"], str(tags[2]))
    check("...and the person's stays throughout",
          all("<Picture 1>" in t for t in tags), str(tags))
    # Putting it back on. This could not work before: the token stayed in `gone` for
    # the rest of the film, so an 'add:' naming it was suppressed by the very removal
    # it was undoing.
    check("an add puts the object and its reference back",
          tags[3] == ["<Picture 1>", "<Picture 2>"], str(tags[3]))
    check("...and it stays on after that",
          tags[4] == ["<Picture 1>", "<Picture 2>"], str(tags[4]))


def test_hardware_stays_on_its_owner():
    print("\n=== hardware does not spread to the other character ===")
    # Reported: a belt locked onto one character turned up on the other, over their
    # clothes. The hold named nobody, so in a two-person shot it was an instruction
    # about whoever was on screen.
    mem = ("Nora: 34, she, red hair, bare skin, a locked steel waist belt.\n"
           "Victor: he, 41, dark hair, navy overalls, work boots")
    P = ("Nora stands by the bench, the steel belt locked on her hips.\n\n"
         "Victor walks in through the side door and looks at her.")
    sh = [" ".join(x.split()) for x in
          re.split(r"(?=\[Shot )", run_node(P, plan_only=True, anchor="A workshop.",
                                            character_memory=mem)[3]) if x.strip()]
    check("the shot with one person keeps the plain hold",
          "Every restraint stays closed" in sh[0])
    check("...and spends no words naming whose", "Every restraint on Nora" not in sh[0])
    check("the shot with two names the wearer", "Every restraint on Nora" in sh[1])
    # Once. The second sentence saying the same thing cost another naming of her, and
    # a described person is a person the model draws -- reported as a second girl
    # appearing at the moment of cuffing.
    check("...naming her once, not twice",
          len(re.findall(r"\bNora\b", sh[1].split("Nora:")[-1])) == 1, sh[1][-90:])
    check("...pinning the other to his own entry",
          "exactly what their own entry lists" in sh[1])
    # And his entry is still there to be pinned to.
    check("the other character keeps his clothes described", "navy overalls" in sh[1])


def test_a_state_in_the_scene_is_not_reasserted():
    print("\n=== a state changed in one beat is not re-asserted later ===")
    # The van usually stands in the SCENE paragraph, which is prepended to every
    # shot. So the text saying "doors closed" is in shot 3 as much as shot 1 -- and
    # by shot 3 the doors have been opened. Asserting the written state there would
    # shut them again, which is the reported bug wearing the other shoe.
    P = ("Daylight. A yard, and a van with its doors closed.\n\n"
         "Mara and Dom stand behind the van.\n\n"
         "Mara opens the van doors and climbs in.\n\n"
         "Dom looks back at the yard.")
    shots = [s for s in run_node(P, plan_only=True)[3].split("---") if s.strip()]
    check("three shots", len(shots) == 3, "")
    check("shot 1 is told the doors are already closed",
          "already closed at the first frame" in shots[0], "")
    # The beat that opens them is asking for that motion. Holding the state here
    # would be the node arguing with the script. It gets the two ENDS of the change
    # instead, which is a different sentence and the subject of its own test.
    check("the shot that opens them is not told the state holds",
          "already closed" not in shots[1], "")
    check("...and the shot after is not told the old state",
          "first frame" not in shots[2], "")
    # The author's own words still reach the model verbatim -- the node adds nothing
    # and takes nothing away. It is only the ADDED sentence that stops.
    check("the scene text itself is untouched",
          all("doors closed" in s for s in shots), "")
    info = run_node(P, plan_only=True)[2]
    check("info names the shot", "shot(s) 1 describe scenery in a state" in info, "")
    check("...and names the switch", "hold_scene_state" in info, "")
    off = run_node(P, plan_only=True, hold_scene_state=False)[3]
    check("the switch turns it off", "first frame" not in off, "")
    check("...and changes nothing else", off.count("doors closed") == 3, "")


def _prompts_sent(P, **kw):
    """Every prompt build_conditioning actually received, in shot order."""
    seen = []
    orig = S.build_conditioning
    def spy(clip, vae, audio_vae, prompt, *a, **k):
        seen.append((prompt, len(k.get("refs") or [])))
        return orig(clip, vae, audio_vae, prompt, *a, **k)
    S.build_conditioning = spy
    try:
        out = run_node(P, **kw)
    finally:
        S.build_conditioning = orig
    return seen, out[3]


def test_the_cuffs_stay_in_the_picture():
    print("\n=== the hardware is named on every shot it is on ===")
    # Reported after the sheet stopped listing the item: the cuffs disappeared while
    # she still looked restrained. The sheet had been the thing naming the object in
    # every shot. Taking it off the sheet is right -- it put the cuffs in the shots
    # before they went on -- and naming it here is what that costs.
    mem = "Mara: she, 22, grey dress.\nDan: he, 41."
    P = ("A bare room.\n\nMara backs away from Dan.\n\n"
         "Dan catches her and cuffs her wrists behind her back.\n\n"
         "Mara sits on the crate.\n\nMara looks at the door.")
    sh = [s for s in run_node(P, plan_only=True, character_memory=mem)[3].split("---")
          if s.strip()]
    check("before it goes on, nothing is claimed", "The cuffs stay" not in sh[0], "")
    check("the applying shot says it in the beat", "cuffs" in sh[1].lower(), "")
    check("...and is not told it a second time", "The cuffs stay" not in sh[1], "")
    for i in (2, 3):
        check(f"shot {i + 1} names the hardware", "cuffs" in sh[i].lower(), sh[i][-70:])
        check(f"...as the object, not just a category", "The cuffs stay" in sh[i], "")
    # A removal lets go of it, like every other latch here.
    P2 = ("A bare room.\n\nDan cuffs her wrists behind her back.\n\nMara sits.\n\n"
          "remove: cuffs\nDan takes the cuffs off.\n\nMara stands up.\n\nMara walks out.")
    sh2 = [s for s in run_node(P2, plan_only=True, character_memory=mem)[3].split("---")
           if s.strip()]
    check("held while they are on", "The cuffs stay" in sh2[1], "")
    check("let go after the removal",
          all("The cuffs stay" not in s for s in sh2[2:]), "")
    # Nothing restrained anywhere: this must not fire on an ordinary scene.
    plain = run_node("A room.\n\nMara waits.\n\nMara walks to the window.",
                     plan_only=True, character_memory=mem)[3]
    check("an unrestrained scene is untouched", "The cuffs stay" not in plain, "")


def test_a_sheet_that_claims_hardware_too_early():
    print("\n=== the sheet listing cuffs she has not been put in yet ===")
    # Why the applying fix does not reach a scene written this way: the sheet lists
    # the hardware, the sheet goes into EVERY shot, so she is restrained from shot 1
    # and the cuffing shot is told the restraint is already fastened. That renders as
    # restrained first and caught afterwards. The sheet is the author's standing
    # description and the beat is the author's action -- the node reports the clash
    # rather than picking a winner.
    mem = "Mara: she, 22, grey dress, handcuffs on her wrists.\nDan: he, 41."
    P = ("A bare room.\n\nMara backs away from Dan.\n\n"
         "Dan catches her and cuffs her wrists behind her back.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    check("the clash is reported", "already lists it as worn" in info, "")
    check("...naming the shot that stages it", "shot(s) 2 stage hardware going ON" in info, "")
    # The sheet really is in the shot before it happens -- that is the point.
    sh = [s for s in script.split("---") if s.strip()]
    check("the cuffs are described before they go on", "handcuffs" in sh[0].lower(), "")
    # A clean sheet gets the applying clause and no complaint.
    clean = run_node(P, plan_only=True, character_memory="Mara: she, 22.\nDan: he, 41.")
    check("a clean sheet raises nothing", "already lists it as worn" not in clean[2], "")
    check("...and gets both ends on the applying shot",
          "hardware goes on during this shot" in clean[3], "")
    # Reported once. It is one authoring decision, not one per shot.
    many = run_node(P + "\n\nDan locks the cuffs tighter.\n\nDan checks them again.",
                    plan_only=True, character_memory=mem)[2]
    check("said once, not per shot", many.count("already lists it as worn") == 1, "")


def test_the_audit_findings_stay_fixed():
    print("\n=== nine defects found by audit, each held down ===")
    mem3 = "Mara: she, 22.\nKate: she, 20.\nDan: he, 41."
    def sh(P, **kw):
        return [s for s in run_node(P, plan_only=True, **kw)[3].split("---") if s.strip()]

    # 1. restrained_who latched only at the FIRST transition, so a second person
    #    cuffed later was never recorded and their hardware went undescribed forever.
    got = sh("A room.\n\nDan cuffs Mara's wrists.\n\nDan cuffs Kate's wrists too.\n\n"
             "Kate sits alone.\n\nMara looks up.", character_memory=mem3)
    check("a second person cuffed later is held too",
          all("closed and fastened" in s or "hardware goes on" in s for s in got),
          str([("closed and fastened" in s) for s in got]))

    # 2. worn_item was one string, so a second item overwrote the first and the cuffs
    #    stopped being named from the gag onward.
    # Two people, not three: with two women on the sheet "her" is ambiguous, the guard
    # resolves nobody, and that is a separate problem the node already reports.
    last = sh("A room.\n\nDan cuffs her wrists.\n\nDan gags her with duct tape.\n\n"
              "Mara sits still.",
              character_memory="Mara: she, 22.\nDan: he, 41.")[-1].lower()
    check("both items stay named", "cuff" in last and "tape" in last, last[-90:])
    # Mixed hardware and soft goods is described as hardware: steel that is only
    # "tied and holding" is steel nobody has said is closed.
    check("...and cuffs with tape are still closed and fastened",
          "closed and fastened" in last, last[-90:])

    # 3. A displaced outer garment never uncovered what was under it: the beat showed
    #    the thong in its own words and the layering hid it again the next shot.
    m = "Mara: she, 22, denim shorts, a black thong."
    got = sh("A room.\n\nMara stands.\n\nMara pulls her shorts down to show the thong.\n\n"
             "Mara waits.\n\nMara pulls them back up.\n\nMara stands.", character_memory=m)
    check("displacement uncovers what is under it",
          "thong" in got[2].lower(), got[2][-80:])
    check("...and putting it back covers it again",
          "thong" not in got[4].lower(), got[4][-80:])

    # 4. `gone` only ever grew, so an add: could not re-cover.
    got = sh("A room.\n\nMara pulls off her shorts.\n\nMara waits.\n\n"
             "add: her denim shorts are back on\nMara pulls her shorts on.\n\nMara stands.",
             character_memory=m)
    check("an add: re-hides the layer under it", "thong" not in got[3].lower(), "")

    # 5. Stockings stop at the thigh and cover no waistband.
    check("stockings do not cover a belt",
          not S.implied_layers("Mara: she, stockings, a chastity belt."))
    check("...while tights do",
          S.implied_layers("Mara: she, tights, a chastity belt.").get("chastity belt"))

    # 6. One of two speaking made the shot a speaking shot, so the LISTENER's mouth
    #    was left free -- which is where invented lip-sync lands.
    one = sh('A room.\n\nMara says: "Wait." Dan listens.',
             character_memory="Mara: she, 22.\nDan: he, 41.")[0]
    check("the listener's mouth is closed", "Only Mara speaks" in one, one[-80:])
    both = sh('A room.\n\nMara says: "Wait." Dan says: "No."',
              character_memory="Mara: she, 22.\nDan: he, 41.")[0]
    check("...and nobody is closed when both speak", "Only" not in both, "")

    # 7. Rope is tied, not closed and fastened.
    rope = sh("A room.\n\nHer wrists are tied above her head with rope.\n\nMara pulls at it.",
              character_memory="Mara: she, 22.")[1]
    check("rope is tied, not fastened shut",
          "tied and holding" in rope and "closed and fastened" not in rope, rope[-90:])

    # 8. Whether the chain actually restarted was invisible.
    on = run_node("A room.\n\nMara pulls off her coat.\n\nMara waits.", plan_only=True,
                  restart_after_removal=True,
                  character_memory="Mara: she, 22, a grey coat, white top.")[2]
    check("a restart is reported", "start FRESH" in on, "")

    # 9. The gaze target was stated once and dropped, while every other state latches.
    got = sh("A room.\n\nMara watches the TV.\n\nMara sits still.\n\nMara walks out.",
             character_memory="Mara: she, 22.")
    check("the look is held while she stays put",
          "turned to the TV" in got[1], got[1][-70:])
    check("...and let go when she leaves", "eyes and the head" not in got[2], "")


def test_no_cuffs_described_without_their_wearer():
    print("\n=== a shot without the restrained person is not told about cuffs ===")
    # Reported as nasty duplicates. The hold latched for the whole film, so a shot
    # describing only the other character still said cuffs were closed on wrists --
    # and with nobody in the text to own those wrists, the model draws one.
    mem = "Mara: she, 22, grey dress.\nDan: he, 41, dark coat."
    P = ("A bare room.\n\nMara waits by the window.\n\n"
         "Dan walks in and cuffs her wrists behind her back.\n\n"
         "Mara sits on the crate.\n\nDan checks the cuffs.\n\nMara looks up.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    hw = ["yes" if ("closed and fastened" in s or "hardware goes on" in s) else "no"
          for s in sh]
    check("before it goes on, nothing", hw[0] == "no", str(hw))
    check("the applying shot has it", hw[1] == "yes", str(hw))
    check("the wearer's own shots have it", hw[2] == "yes" and hw[4] == "yes", str(hw))
    check("the shot with only the other man does NOT",
          hw[3] == "no", sh[3][-90:])
    check("info names that shot", "shot(s) 4 describe nobody who is wearing" in info, "")
    # It LATCHES: leaving it out of his shot must not lose it for hers.
    check("...and it is back when she is", "closed and fastened" in sh[4], "")


def test_caught_first_then_restrained():
    print("\n=== the shot that puts the cuffs on gets both ends ===")
    mem = "Mara: she, 30.\nDan: he, 41."
    def kinds(P):
        script = run_node(P, plan_only=True, character_memory=mem)[3]
        return [("APPLY" if "hardware goes on during this shot" in s
                 else "HOLD" if "closed and fastened as" in s else "-")
                for s in script.split("---") if s.strip()]
    got = kinds("A living room.\n\nMara runs for the door. Dan catches her and cuffs "
                "her wrists.\n\nMara stands by the wall.\n\nMara pulls against the cuffs.")
    check("the applying shot is told both ends", got[0] == "APPLY", str(got))
    check("...and every shot after gets the standing hold",
          got[1:] == ["HOLD", "HOLD"], str(got))
    # Already wearing it: the standing hold is right and "off at the first frame"
    # would be a lie about somebody who has been in cuffs since the scene began.
    worn = kinds("A room. Mara is handcuffed to the rail.\n\n"
                 "Mara pulls against the cuffs.\n\nMara looks at the door.")
    check("hardware already worn is never called new", "APPLY" not in worn, str(worn))
    check("...and still gets the standing hold", worn == ["HOLD", "HOLD"], str(worn))
    plain = kinds("A room.\n\nMara walks to the window.\n\nMara sits down.")
    check("no hardware, no clause either way", plain == ["-", "-"], str(plain))
    info = run_node("A living room.\n\nDan catches her and cuffs her wrists.",
                    plan_only=True, character_memory=mem)[2]
    check("info names the shot", "shot(s) 1 put the hardware ON" in info, "")


def test_a_television_keeps_its_own_voice():
    print("\n=== the line on the TV does not come out of her mouth ===")
    mem = "Mara: she, 30."
    P = ('A living room.\n\nMara sits on the sofa. The TV says: "Storms tonight."\n\n'
         'Mara says: "Again?"\n\nThe TV plays in the empty room.')
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    check("the voice is given back to the set", "the TV's" in sh[0], sh[0][-90:])
    check("...and the mouths are held closed", "Mouths in the shot" in sh[0], "")
    # The branch must STAY OPEN. The set is supposed to be heard -- silencing it
    # would trade one wrong thing for another. Checked on its own, because the empty
    # room later in this prompt IS silenced and should be.
    solo = run_node('A living room.\n\nMara sits on the sofa. '
                    'The TV says: "Storms tonight."',
                    plan_only=True, character_memory=mem)[2]
    check("the shot is not silenced", "conditioned on real silence" not in solo, "")
    # The WORDS reach the model unchanged. The quotation marks do not: they are
    # exchanged for H3's own <d>...</d>, which is the token pair that says a span
    # is spoken. Quote marks say nothing to the model, and a quoted imperative was
    # reaching it as an imperative sentence and being performed.
    check("...and the line reaches the model as written",
          "Storms tonight." in sh[0] and "<d>Storms tonight.</d>" in sh[0], "")
    # Her own line is untouched: she is speaking, and her mouth must move.
    check("her own line is left alone", "the TV's" not in sh[1], "")
    check("...and her mouth is not held shut", "Mouths in the shot" not in sh[1], "")
    # Nobody in the beat, nothing about mouths -- ca75672 again.
    check("an empty room is told nothing about mouths", "Mouths in the shot" not in sh[2], "")
    check("info names the shot", "shot(s) 1 have a spoken line that belongs" in info, "")
    off = run_node(P, plan_only=True, character_memory=mem,
                   mouths_shut_when_no_line=False)[3]
    check("the switch turns it off", "the TV's" not in off, "")


def test_a_shifted_workflow_stops_before_rendering():
    print("\n=== a slid workflow fails with an explanation, not a bad render ===")
    # The exact shape from the report: values slid up one slot after a widget was
    # converted to an input. Rendering anyway would use a scheduler as a sampler and
    # settings nobody picked, and the output would look like a broken model.
    try:
        run_node("A room.\n\nMara waits.", plan_only=True,
                 resolution=0.7, sampler_name="beta", scheduler=48,
                 shot_length=True, cfg=float("nan"))
        check("it refuses to render", False, "no error raised")
    except RuntimeError as e:
        msg = str(e)
        check("it refuses to render", True, "")
        check("...naming the widgets that are wrong",
              "sampler_name" in msg and "scheduler" in msg, "")
        check("...the cause", "restored by POSITION" in msg, "")
        check("...and the fix", "Fix node (recreate)" in msg, "")
    # It must not fire on a healthy graph, or nobody can render at all.
    ok = run_node("A room.\n\nMara waits.", plan_only=True)
    check("a healthy workflow is untouched", "PLAN ONLY" in ok[2], "")
    # A NaN in a NUMBER is still repaired rather than refused -- that one is
    # recoverable, and sane_widgets says so in info.
    num = run_node("A room.\n\nMara waits.", plan_only=True, pace=float("nan"))
    check("a NaN number is still repaired, not refused",
          "not usable numbers" in num[2], "")


def test_the_removal_shot_says_what_is_under():
    print("\n=== taking the shorts off shows the panties, not skin ===")
    # Reported: the shorts come off and the render goes straight to bare, past the
    # underwear the sheet named. The removal clause is emphatic and specific -- off
    # the body, dropped out of frame -- while the layer beneath is one entry in an
    # attribute list, and against a prior that says trousers coming off means bare
    # skin, a list entry does not compete.
    mem = "Mara: she, 22, blue denim shorts, white top, black panties."
    P = "A room.\n\nMara stands.\n\nMara pulls off her shorts.\n\nMara turns."
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    check("the removing shot is told what shows there",
          "what shows there now" in sh[1], sh[1][-90:])
    check("...naming the layer", "panties underneath" in sh[1].lower(), "")
    check("...and saying it stays on", "still on" in sh[1], "")
    check("said only on the shot that uncovers it",
          "what shows there now" not in sh[0] and "what shows there now" not in sh[2], "")
    check("info names the shot", "take off a garment that was covering" in info, "")
    # Nothing underneath: there is nothing to promise, and promising anyway would
    # dress her in something the sheet never gave her.
    bare = run_node("A room.\n\nMara pulls off her shorts.\n\nMara turns.",
                    plan_only=True,
                    character_memory="Mara: she, 22, blue denim shorts, white top.")[3]
    check("nothing underneath, nothing claimed", "what shows there now" not in bare, "")
    # A full strip takes the cover AND what was under it. Saying the panties show
    # would put back the garment the beat was most explicit about removing.
    strip = run_node("A room.\n\nMara undresses completely.\n\nMara turns.",
                     plan_only=True, character_memory=mem)[3]
    check("a full strip promises nothing", "what shows there now" not in strip, "")


def test_a_beat_can_name_what_the_layering_hid():
    print("\n=== a beat naming a covered garment puts it back, and is reported ===")
    # Three fixes into "the belt is on top of the jeans", with the sheet correct, the
    # tag correct and the picture correctly withheld. This is the route none of them
    # could touch: beats are passed through word for word and are never scrubbed, so
    # the layering takes the belt out of the sheet and the beat puts it straight back.
    # A described thing is a drawn thing, drawn over whatever is on top of it -- and
    # the next shot starts from this one's last frame, so it is carried forward from
    # there and looks permanent rather than like one bad shot.
    mem = "Mara: <Picture 1>, she, 22, blue jeans, a chastity belt <Picture 2>."
    P = ("A room.\n\nMara stands by the window.\n\n"
         "Mara runs a hand over the chastity belt under her jeans.\n\nMara waits.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    check("the sheet still hides it", "chastity belt" not in sh[0].lower(), "")
    check("the beat still says it", "chastity belt" in sh[1].lower(), "")
    check("...word for word, unedited",
          "runs a hand over the chastity belt under her jeans" in sh[1], "")
    check("the clash is reported", "says is covered" in info, "")
    check("...naming the shot and the garment",
          "shot 2: chastity belt" in info, "")
    # And it must not fire when the beat leaves it alone.
    quiet = run_node("A room.\n\nMara stands.\n\nMara waits.", plan_only=True,
                     character_memory=mem)[2]
    check("no mention, no warning", "says is covered" not in quiet, "")


def test_a_removal_says_whose_hands():
    print("\n=== a garment does not take itself off ===")
    # Reported: she asks to have the chastity belt taken off, and it drops off on its
    # own. The removal clause named no agent -- "the belt comes off during this shot
    # and is away by the last frame, dropped out of frame" is true of a belt falling
    # to the floor by itself, and that is what it rendered. The beat named the person;
    # the clause was simply not carrying it.
    mem = ("McKenna: she, 22, Shiny white crop top, chastity belt, blue jeans shorts.\n"
           "Dan: he, 41, dark coat.")
    P = ("A room.\n\nMcKenna stands.\n\n"
         "remove: shorts\nMcKenna takes off her shorts.\n\n"
         'McKenna asks Dan to take the chastity belt off. "Please take it off."\n'
         "remove: belt\n\nMcKenna waits.")
    sh = [s for s in run_node(P, plan_only=True, character_memory=mem)[3].split("---")
          if s.strip()]
    check("she undresses herself with her own hands",
          "McKenna takes off her shorts" in sh[1]
          and "away by the last frame" in sh[1], sh[1][-110:])
    # ASKING gives the hands to the other person. "She asks Dan to take it off" is
    # Dan's doing -- reading the asker as the agent is what put them back on her.
    check("asking hands it to the other person",
          "Dan takes the chastity belt off during this shot" in sh[2], sh[2][-110:])
    check("...and not to the one who asked",
          "McKenna takes the chastity belt off" not in sh[2], "")
    # One person in the shot is that person, with no ambiguity to resolve.
    solo = run_node("A room.\n\nremove: coat\nMara takes off her coat.", plan_only=True,
                    character_memory="Mara: she, 22, a grey coat, white top.")[3]
    check("a solo shot names her",
          "Mara takes off her coat" in solo
          and "away by the last frame" in solo, "")
    # The unit rule, directly.
    beat = 'McKenna asks Dan to take the chastity belt off.'
    check("the agent is read from the beat",
          S.removal_agent(beat, ["McKenna", "Dan"], "McKenna") == "Dan")
    check("...and without a wearer the first named acts",
          S.removal_agent(beat, ["McKenna", "Dan"], None) == "McKenna")
    check("no cast, no agent claimed", S.removal_agent(beat, []) == "")


def test_the_only_tag_being_on_a_covered_thing():
    print("\n=== the belt is the only tagged thing, and it is under the shorts ===")
    # The reported sheet, verbatim. Two faults met in it, and each hid the other.
    sheet = ("McKenna: she, 22, Shiny white crop top, chastity belt <picture 2>, "
             "blue jeans shorts.")
    # 1. "blue jeans shorts" is a pair of SHORTS. Taking the first outer match made the
    #    cover "jeans" while a removal names it "shorts", so the two never lined up and
    #    the belt was hidden correctly and then never uncovered.
    check("the cover is the head noun, not the first word",
          S.implied_layers(sheet) == {"chastity belt": "shorts"},
          str(S.implied_layers(sheet)))
    check("...so taking the shorts off uncovers it",
          S.hidden_layers(S.implied_layers(sheet), ["shorts"]) == [])
    # 2. The belt's tag is the ONLY tag. Layering scrubs the belt on every covered
    #    shot and the tag goes with it -- correctly -- and the node then saw no tags
    #    anywhere and fell back to "untagged references ride EVERY shot", sending the
    #    belt's picture into exactly the shots that had just hidden it.
    BELT = torch.full((1, H, W, 3), 0.80)
    rows = []
    ob = S.build_conditioning
    def spy(clip, vae, audio_vae, p, *a, **k):
        rows.append((p, len(k.get("refs") or [])))
        return ob(clip, vae, audio_vae, p, *a, **k)
    S.build_conditioning = spy
    try:
        run_node("A room.\n\nMcKenna stands.\n\nMcKenna waits.\n\n"
                 "remove: shorts\nMcKenna takes off her shorts.\n\nMcKenna turns.",
                 character_memory=sheet + "\nDan: he, 41.", ref_image_2=BELT)
    finally:
        S.build_conditioning = ob
    check("no picture while it is covered",
          rows[0][1] == 0 and rows[1][1] == 0, str([n for _, n in rows]))
    check("...and no words either",
          not any("chastity" in p.lower() for p, _ in rows[:2]), "")
    check("the picture arrives when the shorts come off", rows[2][1] == 1, "")
    check("...and stays after", rows[3][1] == 1, "")
    check("...described too", all("chastity" in p.lower() for p, _ in rows[2:]), "")


def test_an_object_tag_works_without_a_face_picture():
    print("\n=== a tagged object is placed even when nobody is tagged ===")
    # A picture is not always somebody's face. With the person untagged, the belt's
    # own tag was the only one in the entry -- and once the layering scrubbed the
    # belt's WORDS, ownership was decided on the leftover comma fragment, which has
    # nothing in front of it. So the tag read as the person's, survived, and went on
    # fetching the belt's picture into every shot where the belt was covered.
    FACE = torch.full((1, H, W, 3), 0.10)
    BELT = torch.full((1, H, W, 3), 0.80)
    which = lambda t: "FACE" if abs(float(t.mean()) - 0.10) < 0.01 else "BELT"
    def imgs(mem):
        rows = []
        ob = S.build_conditioning
        def spy(clip, vae, audio_vae, p, *a, **k):
            rows.append([which(r) for r in (k.get("refs") or [])])
            return ob(clip, vae, audio_vae, p, *a, **k)
        S.build_conditioning = spy
        try:
            run_node("A room.\n\nMara stands.\n\nMara waits.\n\nMara pulls off her jeans.",
                     character_memory=mem, ref_image_1=FACE, ref_image_2=BELT)
        finally:
            S.build_conditioning = ob
        return rows
    for label, mem in (
            ("nobody tagged, belt tag first",
             "Mara: she, 22, blue jeans, <Picture 2> a chastity belt."),
            ("nobody tagged, belt tag last",
             "Mara: she, 22, blue jeans, a chastity belt <Picture 2>.")):
        got = imgs(mem)
        check(f"{label}: withheld while covered",
              "BELT" not in got[0] and "BELT" not in got[1], str(got))
        check(f"{label}: sent when uncovered", "BELT" in got[2], str(got))
    # A face picture alongside it still behaves, and still travels every shot.
    got = imgs("Mara: <Picture 1>, she, blue jeans, a chastity belt <Picture 2>.")
    check("with a face too, the face is always there",
          all("FACE" in g for g in got), str(got))
    check("...and the belt only when uncovered",
          "BELT" not in got[0] and "BELT" in got[2], str(got))


def test_an_untagged_picture_defeats_the_layering():
    print("\n=== an untagged reference is sent even where its garment is covered ===")
    # Reported: the belt drawn on top of the jeans, with "chastity belt" spelled the
    # way the layering matches. The words were right -- the belt had stopped being
    # described -- and the PICTURE was still going in. An untagged reference rides
    # every shot, so an image of something currently underneath something else is
    # sent on the shots where it is covered, and the model draws what it is shown.
    img = lambda v: torch.full((1, H, W, 3), v)
    P = "A room.\n\nMara stands.\n\nMara pulls off her jeans."
    untagged = run_node(P, plan_only=True, ref_image_1=img(0.1), ref_image_2=img(0.8),
                        character_memory="Mara: she, 22, blue jeans, a chastity belt.")[2]
    check("the clash is reported",
          "not one <Picture N> tag anywhere" in untagged, "")
    check("...naming the layer it will override",
          "chastity belt under jeans" in untagged, "")
    check("...and saying what fixes it", "<Picture 2>" in untagged, "")
    # Tagged, the layering controls the image and there is nothing to warn about.
    tagged = run_node(P, plan_only=True, ref_image_1=img(0.1), ref_image_2=img(0.8),
                      character_memory="Mara: <Picture 1>, she, 22, blue jeans, "
                                       "a chastity belt <Picture 2>.")[2]
    check("a tagged wardrobe raises nothing",
          "not one <Picture N> tag anywhere" not in tagged, "")
    # No layers to defeat: an untagged reference is then just an untagged reference,
    # which the node already warns about separately.
    flat = run_node(P, plan_only=True, ref_image_1=img(0.1),
                    character_memory="Mara: she, 22, a white top and boots.")[2]
    check("no layers, no layer warning",
          "not one <Picture N> tag anywhere" not in flat, "")


def test_underwear_is_hidden_until_it_is_not():
    print("\n=== underwear stays out of the text while something is over it ===")
    mem = "Mara: she, 22, blue denim shorts, white top, panties, a chastity belt."
    P = ("A room.\n\nMara stands by the window.\n\n"
         "Mara pulls off her shorts.\n\nMara turns around.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s.lower() for s in script.split("---") if s.strip()]
    check("while the shorts are on, the panties are not described",
          "panties" not in sh[0], sh[0][-90:])
    # The belt goes under as well: worn beneath jeans it is COVERED, not forgotten,
    # and it comes back by the same route as any other layer -- the cover comes off,
    # the reveal clause names it, and it stays in the scene from then on.
    check("...nor the belt", "chastity belt" not in sh[0], sh[0][-90:])
    check("...while the shorts themselves are", "shorts" in sh[0], "")
    # The shot that takes them off is where both become visible, and it has to say so
    # or the reveal happens against a body the text says is bare.
    check("the reveal shot describes them", "panties" in sh[1] and "chastity belt" in sh[1], "")
    check("...and every shot after", "panties" in sh[2] and "chastity belt" in sh[2], "")
    check("info explains the layering", "read as layers" in info, "")
    # Nothing over it: underwear is on show and must not be described away.
    solo = run_node("A room.\n\nMara stands by the window.", plan_only=True,
                    character_memory="Mara: she, 22, panties and a bra.")[3]
    check("underwear alone is still described", "panties" in solo.lower(), "")


def test_a_garment_moved_is_not_a_garment_gone():
    print("\n=== shorts pulled down stay on, and stay described ===")
    # Reported: the shorts changed appearance in the next beat. "Pulls down her
    # shorts" was read as a full removal, so the shot was told they come off and are
    # "dropped out of frame", and the entry was scrubbed from the scene -- leaving
    # every later shot describing nothing where something still was. An undescribed
    # garment is one the model re-invents, which is the same pair coming back a
    # different pair.
    mem = "Mara: she, 22, blue denim shorts, white top."
    P = ("A room.\n\nMara stands and pulls down her shorts.\n\n"
         "Mara steps to the window.\n\nMara looks back.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    check("the shorts are not taken off",
          "come off during this shot" not in sh[0]
          and "comes off during this shot" not in sh[0], sh[0][-80:])
    check("...and are not scrubbed from the scene",
          all("shorts" in s.lower() for s in sh), "")
    check("the later shots say where they now sit",
          all("Still on the body" in s for s in sh[1:]), "")
    check("...and the staging shot is not told it twice",
          "Still on the body" not in sh[0], "")
    check("info names the shots", "MOVED rather than taken off" in info, "")
    # Put back up: the latch lets go, by name or by pronoun.
    for _put in ("Mara pulls her shorts back up.", "Mara pulls them back up."):
        back = run_node("A room.\n\nMara pulls down her shorts.\n\nMara waits.\n\n"
                        + _put + "\n\nMara walks out.",
                        plan_only=True, character_memory=mem)[3]
        got = ["yes" if "Still on the body" in s else "no"
               for s in back.split("---") if s.strip()]
        check(f"restored by {_put[:28]!r}", got == ["no", "yes", "no", "no"], str(got))
    # A real removal still empties the wardrobe and says so.
    off = run_node("A room.\n\nMara pulls off her shorts.\n\nMara waits.",
                   plan_only=True, character_memory=mem)[3]
    # The guarantee is that the removal FINISHES in the shot -- the last frame is
    # the next shot's keyframe. Not the exact sentence: where the beat already
    # stages the act, the clause says only the part the beat leaves out, because
    # restating who and what was the same removal written twice.
    check("a real removal still removes",
          "away by the last frame" in off and "fully removed" in off, "")
    check("...and stops describing them", "shorts" not in off.split("---")[1].lower(), "")


def test_undressing_does_not_drop_her():
    print("\n=== taking a garment off does not put a body on the floor ===")
    # Whole-path, because the unit test cannot show the clause reaching the shot. The
    # fall cue read "pulls down her shorts" as a body being put on the floor, and the
    # shot was then told what takes the landing and how the legs fold under it.
    mem = "Mara: she, 22, denim shorts.\nDan: he, 41."
    P = ("A room.\n\nMara stands and pulls down her shorts.\n\n"
         "Dan pushes her down onto the floor.")
    sh = [s for s in run_node(P, plan_only=True, character_memory=mem)[3].split("---")
          if s.strip()]
    check("the undressing shot is not told to fall",
          "falls as one piece" not in sh[0], sh[0][-80:])
    check("...and the real fall still is", "falls as one piece" in sh[1], "")
    # The removal itself must still work -- this only changes what reads as a FALL.
    check("the garment still comes off", "shorts" in sh[0].lower(), "")
    info = run_node(P, plan_only=True, character_memory=mem)[2]
    check("only the real fall is reported", "shot(s) 2 put a body down" in info, "")


def test_an_unbound_fall_is_told_what_catches_it():
    print("\n=== a fall with no hardware in it still names the landing ===")
    # FALL_HOLD only ever fired on a RESTRAINED fall -- the concern there was the
    # hold giving way when the hands came up to break it. A free body falling had
    # nothing said about it at all, and that is the shot the spare leg turned up on.
    mem = "Mara: she, 30."
    free = run_node("A room.\n\nMara trips and falls to the floor.",
                    plan_only=True, character_memory=mem)[3]
    check("a free fall is told what takes the landing",
          "The body falls as one piece" in free, free[-90:])
    check("...and it is not the bound wording",
          "A bound body falls" not in free, "")
    bound = run_node("A room.\n\nMara is handcuffed behind her back.\n\n"
                     "Mara falls to the floor.", plan_only=True, character_memory=mem)[3]
    last = [s for s in bound.split("---") if s.strip()][-1]
    check("a bound fall keeps the bound wording", "A bound body falls" in last, "")
    check("...and not both at once", "The body falls as one piece" not in last, "")
    # It must not fire on shots that are not a fall -- every clause costs the beat
    # some of the shot.
    still = run_node("A room.\n\nMara walks to the window.\n\nMara drops the keys.",
                     plan_only=True, character_memory=mem)[3]
    check("no fall, no clause",
          "falls as one piece" not in still and "bound body falls" not in still, "")
    info = run_node("A room.\n\nMara trips and falls to the floor.",
                    plan_only=True, character_memory=mem)[2]
    check("info names the shot", "shot(s) 1 put a body down" in info, "")


def test_a_named_look_target_is_restated():
    print("\n=== a beat that names something to look at gets it said twice ===")
    P = ("A living room.\n\nMara sits on the sofa looking at the TV.\n\n"
         "Mara looks at her.\n\nMara walks to the window.")
    info, script = run_node(P, plan_only=True, character_memory="Mara: she, 30.")[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    check("the named target is restated",
          "The eyes and the head are turned to the TV" in sh[0], sh[0][-70:])
    # It follows the beat rather than leading it -- ca75672: anatomy in the opening
    # tokens is what a distilled LoRA settles composition on.
    check("...after the beat, not before it",
          sh[0].index("looking at the TV") < sh[0].index("The eyes and the head"), "")
    check("a pronoun target adds nothing", "The eyes and the head" not in sh[1], "")
    check("a beat with no look adds nothing", "The eyes and the head" not in sh[2], "")
    check("info names the shot", "shot(s) 1 name something to look at" in info, "")
    off = run_node(P, plan_only=True, character_memory="Mara: she, 30.",
                   hold_gaze=False)[3]
    check("the switch turns it off", "The eyes and the head" not in off, "")
    check("...and leaves the beat exactly as written", "looking at the TV" in off, "")


def test_a_covered_object_does_not_send_its_picture():
    print("\n=== an object out of view does not carry its reference ===")
    # Reported: the object looked different when it came back into view. While it was
    # covered, the scrubber removed its WORDS and left its <Picture N> -- so the shot
    # still sent the reference, unclaimed, and whatever the model made of a picture
    # nothing accounted for became the keyframe the next shot was built on.
    FACE = torch.full((1, H, W, 3), 0.10)
    OBJ = torch.full((1, H, W, 3), 0.80)
    which = lambda t: "FACE" if abs(float(t.mean()) - 0.10) < 0.01 else "OBJ"
    mem = ("Mara: <Picture 1>, she, 30, wearing a silver locket <Picture 2>, "
           "and a long coat.")
    P = ("A room.\n\nMara stands by the window in her coat.\n\n"
         "Mara takes off the coat, showing the locket.\n\nMara walks to the door.")
    rows = []
    ob = S.build_conditioning
    def spy(clip, vae, audio_vae, prompt, *a, **k):
        rows.append((prompt, [which(r) for r in (k.get("refs") or [])]))
        return ob(clip, vae, audio_vae, prompt, *a, **k)
    S.build_conditioning = spy
    try:
        run_node(P, character_memory=mem, ref_image_1=FACE, ref_image_2=OBJ)
    finally:
        S.build_conditioning = ob
    check("the covered shot sends only the face", rows[0][1] == ["FACE"], str(rows[0][1]))
    check("...and names no picture it does not carry",
          sorted(set(re.findall(r"<Picture (\d+)>", rows[0][0]))) == ["1"], rows[0][0][:80])
    check("...and does not describe the covered object",
          "locket" not in rows[0][0].lower(), "")
    # Back in view: the words and the picture return together.
    check("the reveal brings the picture back", rows[1][1] == ["FACE", "OBJ"], str(rows[1][1]))
    check("...claimed by the text",
          sorted(set(re.findall(r"<Picture (\d+)>", rows[1][0]))) == ["1", "2"], "")
    check("...and it stays for the shot after", rows[2][1] == ["FACE", "OBJ"], str(rows[2][1]))
    # Every shot: as many references as the text names. The invariant this broke.
    bad = [i + 1 for i, (p, imgs) in enumerate(rows)
           if len(imgs) != len(set(re.findall(r"<Picture (\d+)>", p)))]
    check("no shot carries a picture its text never names", not bad, str(bad))


def test_the_anchor_survives_a_close_shot():
    print("\n=== cuffs above the head are still above the head next shot ===")
    P = ("A room.\n\nMara is handcuffed above her head to the bed frame.\n\n"
         "A close shot of her face.\n\nMara turns her head.\n\n"
         "remove: handcuffs\nMara sits up and rubs her wrists.")
    info, script = run_node(P, plan_only=True, character_memory="Mara: she, 30.")[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    # The staging shot has the author's own words and gets no second sentence about it.
    check("the staging shot is not argued with",
          "the wrists" not in sh[0], "")
    check("the next shot is told where they are",
          "above the head, at the bed frame" in sh[1], sh[1][-80:])
    # The one that matters: a close shot crops the anchor out of the picture, so the
    # text is the only thing still carrying it.
    check("...including the close shot", "the wrists" in sh[1], "")
    check("...and the shot after that", "the wrists" in sh[2], "")
    # It latches like the hardware and is released by the same `remove:`.
    check("a removal lets go of it", "the wrists" not in sh[3], sh[3][-70:])
    check("info names the held shots", "fastened limbs held in place on shot(s) 2, 3" in info, "")
    check("...and names the tight framing", "frame tight enough to crop" in info, "")
    # Nothing to anchor, nothing said -- this must not fire on ordinary shots.
    plain = run_node("A room.\n\nMara waits.\n\nMara walks to the window.",
                     plan_only=True, character_memory="Mara: she, 30.")[3]
    check("an unrestrained scene is untouched", "the wrists" not in plain, "")


def test_mouths_stay_shut_with_no_line():
    print("\n=== a shot with nobody speaking keeps its mouth closed ===")
    # H3 is joint: the face follows the audio branch. A shot with no line but a sound
    # the AUTHOR wrote -- "a low hum off the strip light" -- kept its branch open, and
    # an open branch invents a voice the picture lip-syncs to. Nobody is speaking and
    # the mouth moves anyway. That was the hole; a written sound was enough to open it.
    P = ('A workshop.\n\n'
         'Kate walks to the window.\n\n'
         'Kate says: "Wait there."\n\n'
         'A low hum comes off the strip light.\n\n'
         'Kate strains against the cuffs.')
    # With a sheet, because that is how the node knows Kate is a person. It does not
    # scan prose for capitals on purpose -- a sheet LABELS people and guessing from
    # capitalisation picks up place names. A beat with a bare pronoun still works
    # without one; a beat with only an unlisted name does not, and that is the
    # conservative direction: no clause rather than a clause about nobody.
    MEM = "Kate: she, 30, red coat."
    # auto_sound OFF: the trade this reports -- a written sound given up so the
    # mouth can be held -- only arises when the branch would otherwise be pinned.
    info, script = run_node(P, plan_only=True, character_memory=MEM,
                            auto_sound=False)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    mouth = [i + 1 for i, s in enumerate(sh) if "Mouths in the shot stay closed" in s]
    check("the wordless shot with a person is told to close", mouth == [1], str(mouth))
    check("the speaking shot is not", "Mouths in the shot stay closed" not in sh[1], "")
    # ca75672, which this must not undo: a mouth sentence on a beat with nobody in it
    # describes a person who is not there, and the only way to satisfy it is to draw a
    # face into an empty frame. The AUDIO half has no such limit -- an empty room still
    # babbles -- so shot 3 is silenced without being told anything about mouths.
    check("the scenery beat is told nothing about mouths",
          "Mouths in the shot stay closed" not in sh[2], "")
    check("...but is still silenced", "sound is" not in sh[2], "")
    # Effort is vocal and its mouth SHOULD be open. Silencing a straining body was a
    # bug once already -- it renders as a flat, unreacting face.
    check("a straining body is left alone", "Mouths in the shot stay closed" not in sh[3], "")
    # ...and its branch stays OPEN. With auto_sound off there is no sound sentence
    # to look for, so the guarantee is read from the report: shot 4 is one of the
    # shots whose audio is left free, which is what "keeps its audio" means.
    check("...and keeps its audio",
          "either describe a sound IN THE BEAT or stage EFFORT" in info
          and re.search(r"shot\(s\) [^|]*\b4\b[^|]*stage EFFORT", info) is not None,
          "")
    # The written sound on a wordless shot is given up, because conditioning the
    # branch is the only thing that actually settles the mouth.
    check("the wordless sound shot is silenced", "sound is" not in sh[2], "")
    check("info names the shots held closed", "mouths held closed on shot(s) 1" in info, "")
    check("...and names what it cost", "gave up the sound you wrote" in info, "")
    # The switch puts it all back.
    off = run_node(P, plan_only=True, character_memory=MEM,
                   mouths_shut_when_no_line=False)[3]
    check("off, nothing is told to close", "Mouths in the shot stay closed" not in off, "")
    check("off, the written sound comes back",
          "sound" in [s for s in off.split("---") if s.strip()][2].lower(), "")
    # Positively phrased: at cfg 1 no negative is evaluated, so an absence cannot be
    # asked for -- only a closed mouth can be drawn.
    check("the clause is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b|\bnobody\b", S.MOUTH_HOLD, re.I), "")
    check("...and is one short sentence",
          S.MOUTH_HOLD.count(".") == 1 and len(S.MOUTH_HOLD.split()) <= 14, "")


def test_script_is_what_was_sent():
    print("\n=== script reports the text the model was actually given ===")
    # script is documented as the exact per-shot text, and it is what the reader is
    # told to check when a shot renders somebody they did not ask for. It was built
    # before the render loop and never touched again, while the recovered-face claim
    # was written onto the loop's own copy -- so on exactly the shot most likely to
    # be under investigation, the model got "Dom: <Picture 1>, he, 41" and script
    # said "Dom: he, 41". Diagnosing a duplicate from that leads to the wrong fix.
    mem = "Mara: <Picture 1>, she, 30, red coat.\nDom: he, 41, grey jacket."
    P = ("Daylight. A yard.\n\nDom stands by the gate.\n\n"
         "Mara walks along the fence.\n\nDom looks at the sky.")
    sent, script = _prompts_sent(P, character_memory=mem,
                                 ref_image_1=torch.rand(1, H, W, 3))
    rep = [b.split("] ", 1)[1].strip() for b in script.split("\n---\n")]
    check("every shot matches, claim included",
          all(s.strip() == r for (s, _), r in zip(sent, rep)), "")
    check("the recovery shot really does carry a claim",
          "<Picture 1>" in sent[2][0] and "<Picture 1>" in rep[2], "")


def test_every_reference_is_claimed():
    print("\n=== no shot carries a picture its text never names ===")
    # The node's own rule, and the cause of every duplicate reported so far: a
    # picture the prompt refers to is that subject, and one it never mentions is
    # ANOTHER subject -- a second person with the same face and clothes. So the
    # count of references sent must equal the count of tags in the prompt, and the
    # numbers must run 1..n with no gaps, or a tag points at the wrong image.
    img = lambda: torch.rand(1, H, W, 3)
    cases = [
        ("one tagged, one untagged returning",
         "A yard.\n\nDom stands by the gate.\n\nMara walks along the fence.\n\n"
         "Dom looks at the sky.",
         dict(character_memory="Mara: <Picture 1>, she, 30, red coat.\nDom: he, 41.",
              ref_image_1=img())),
        ("two people, two tagged references",
         "A yard.\n\nMara waits.\n\nDom arrives.\n\nMara and Dom talk.",
         dict(character_memory="Mara: <Picture 1>, she, 30.\nDom: <Picture 2>, he, 41.",
              ref_image_1=img(), ref_image_2=img())),
        ("a reference nobody tags",
         "A yard.\n\nMara waits.\n\nDom arrives.",
         dict(character_memory="Mara: <Picture 1>, she, 30.\nDom: he, 41.",
              ref_image_1=img(), ref_image_2=img())),
        ("an object tag beside a person tag",
         "A yard.\n\nMara waits.\n\nMara holds the locket.\n\nDom arrives.",
         dict(character_memory="Mara: <Picture 1>, she, a silver locket <Picture 2>.\n"
                               "Dom: he, 41.",
              ref_image_1=img(), ref_image_2=img())),
    ]
    for name, P, kw in cases:
        sent, _ = _prompts_sent(P, **kw)
        bad = []
        for i, (p, n) in enumerate(sent, 1):
            tags = sorted({int(x) for x in re.findall(r"<Picture (\d+)>", p)})
            if n != len(tags) or tags != list(range(1, n + 1)):
                bad.append(f"shot {i}: {n} sent, tags {tags}")
        check(f"{name}: every picture is claimed", not bad, "; ".join(bad))


def _encoded_refs(P, **kw):
    """(prompt, number of images actually encoded as references) per shot.

    The demotion happens inside build_conditioning, so counting the refs handed TO
    it misses the handoff being added. This counts what reaches the encoder."""
    rows, box = [], {}
    ob, orr = S.build_conditioning, S._build_ref_images
    def spy_r(vae, imgs, w, h, size):
        box["n"] = len(imgs); return orr(vae, imgs, w, h, size)
    def spy_b(clip, vae, audio_vae, prompt, *a, **k):
        box["n"] = 0
        out = ob(clip, vae, audio_vae, prompt, *a, **k)
        rows.append((prompt, box["n"]))
        return out
    S._build_ref_images, S.build_conditioning = spy_r, spy_b
    try:
        run_node(P, **kw)
    finally:
        S._build_ref_images, S.build_conditioning = orr, ob
    return rows


def test_the_demoted_handoff_is_claimed():
    print("\n=== below the safe aug, the handoff is named too ===")
    # Reported as a duplicate Dan at the END of the video. Below KEYFRAME_SAFE_AUG
    # the handoff stops being a keyframe and is encoded as an extra reference -- and
    # it went in unclaimed, on the reasoning that a first frame is not a subject. In
    # the reference rows it is not a first frame any more: it is picture N of N, and
    # a picture the prompt never names is read as ANOTHER subject. The picture is the
    # previous shot, so the other subject wears that shot's face and clothes.
    #
    # Later shots only, because a shot needs BOTH a handoff and a reference to demote
    # anything -- which is exactly "at the end of the video".
    mem = "Dan: <Picture 1>, he, 41, grey jacket.\nMara: she, 30, red coat."
    P = "A yard.\n\nDan waits.\n\nMara arrives.\n\nDan and Mara talk.\n\nDan looks up."
    img = torch.rand(1, H, W, 3)
    for aug in (0.999, 0.98, 0.95, 0.90):
        rows = _encoded_refs(P, character_memory=mem, ref_image_1=img, ref_noise_aug=aug)
        bad = []
        for i, (p, enc) in enumerate(rows, 1):
            tags = {int(x) for x in re.findall(r"<Picture (\d+)>", p)}
            if enc != len(tags) or sorted(tags) != list(range(1, enc + 1)):
                bad.append(f"shot {i}: {enc} encoded, tags {sorted(tags)}")
        check(f"every picture claimed at aug {aug}", not bad, "; ".join(bad))
    # The claim must say it is the SAME people, or naming it invites a new one.
    rows = _encoded_refs(P, character_memory=mem, ref_image_1=img, ref_noise_aug=0.95)
    late = rows[2][0]
    check("the handoff is named as the opening frame",
          "is the frame this shot opens on" in late, "")
    check("...and as the same people, not new ones",
          "the same people" in late and "anybody new" in late, "")
    # At a safe aug nothing is demoted, so nothing extra is said.
    early = _encoded_refs(P, character_memory=mem, ref_image_1=img, ref_noise_aug=0.999)
    check("nothing added when the handoff stays a keyframe",
          all("opens on" not in p for p, _ in early), "")
    info = run_node(P, character_memory=mem, ref_image_1=img, ref_noise_aug=0.95)[2]
    check("info explains it", "encoded as a reference rather than a keyframe" in info, "")


def test_a_gapped_socket_still_sends_its_image():
    print("\n=== wiring ref_image_1 and ref_image_3 sends both ===")
    # Whole-path, because the unit test cannot show the image being dropped. Wire a
    # gap and the second picture used to vanish: its tag matched nothing in a roster
    # of two, so it was stripped, and a reference nothing claims is not sent.
    img = lambda: torch.rand(1, H, W, 3)
    P = "A yard.\n\nMara waits.\n\nMara walks."
    sent, _ = _prompts_sent(P, character_memory="Mara: <Picture 1>, she, a locket <Picture 3>.",
                            ref_image_1=img(), ref_image_3=img())
    p, n = sent[0]
    check("both images are sent", n == 2, f"{n}")
    check("...and both are claimed in the text",
          sorted(re.findall(r"<Picture (\d+)>", p)) == ["1", "2"], p[:90])
    # A single image on a socket that is not the first.
    sent, _ = _prompts_sent(P, character_memory="Mara: <Picture 2>, she, 30.",
                            ref_image_2=img())
    p, n = sent[0]
    check("a lone image on socket 2 is sent", n == 1 and "<Picture 1>" in p, f"{n} {p[:60]}")
    # Sockets filled from the top are untouched -- this must not move anybody's
    # working setup.
    sent, _ = _prompts_sent(P, character_memory="Mara: <Picture 1>, she, a locket <Picture 2>.",
                            ref_image_1=img(), ref_image_2=img())
    p, n = sent[0]
    check("no gap, nothing changes",
          n == 2 and sorted(re.findall(r"<Picture (\d+)>", p)) == ["1", "2"], f"{n}")
    # The renumbered tag must still fetch the RIGHT image. Two pictures that can be
    # told apart by value, so a mix-up shows up as the wrong face rather than as a
    # number that merely looks tidy. This is the check that matters: renumbering is
    # only safe if the image follows the number.
    A = torch.full((1, H, W, 3), 0.10)
    B = torch.full((1, H, W, 3), 0.70)
    val = lambda t: round(float(t.mean()), 2)
    seen = []
    orig = S.build_conditioning
    def spy(clip, vae, audio_vae, prompt, *a, **k):
        seen.append((prompt, [val(r) for r in (k.get("refs") or [])]))
        return orig(clip, vae, audio_vae, prompt, *a, **k)
    S.build_conditioning = spy
    try:
        run_node("A yard.\n\nMara waits.\n\nDom arrives.\n\nMara and Dom talk.",
                 character_memory="Mara: <Picture 1>, she, 30.\nDom: <Picture 3>, he, 41.",
                 ref_image_1=A, ref_image_3=B)
    finally:
        S.build_conditioning = orig
    check("the lone-reference shot renumbers to 1",
          re.findall(r"<Picture (\d+)>", seen[1][0]) == ["1"], str(seen[1][0][-60:]))
    check("...and still sends that person's own image",
          seen[1][1] == [0.7], str(seen[1][1]))
    check("the shared shot keeps both, in order",
          sorted(re.findall(r"<Picture (\d+)>", seen[2][0])) == ["1", "2"]
          and seen[2][1] == [0.1, 0.7], str(seen[2][1]))
    # And a tag on an empty socket is reported rather than silently dropped.
    info = run_node(P, plan_only=True, character_memory="Mara: <Picture 3>, she, 30.",
                    ref_image_1=img())[2]
    check("a tag with no image behind it is named",
          "names a socket with no image on it" in info, "")


def test_a_modified_state_is_not_read_as_an_act():
    print("\n=== 'closed rear doors' is not somebody closing them ===")
    # Reported after the state hold shipped: the doors were STILL opening and being
    # closed on camera. The node was doing it. A word between the state and its noun
    # made "closed" parse as the verb, and the shot was handed "the doors are open at
    # the first frame and shut by the last" -- the inverted guard, asking for exactly
    # the render it exists to prevent. Whole-path, because that is where it bit.
    for _beat in ("Mara and Dom walk out from behind a van with closed rear doors.",
                  "Mara and Dom step out from behind the van, its back doors closed.",
                  "Mara and Dom stand behind a van with shut cargo doors."):
        s = run_node("Daylight. A yard.\n\n" + _beat, plan_only=True)[3]
        check(f"held, not anchored: {_beat[36:60]!r}",
              "already" in s and "open at the first frame" not in s, "")
    # The one that really does stage it still gets its two ends.
    s = run_node("Daylight. A yard.\n\nMara closed the van's doors.", plan_only=True)[3]
    check("a possessive act is still an act",
          "The doors are open at the first frame and shut by the last." in s, "")
    # And the state is bounded: "stays closed" with no end on it is a state something
    # can happen to before the shot is out.
    s = run_node("Daylight. A yard.\n\nThey stand by a van with its doors closed.",
                 plan_only=True)[3]
    check("the held state is bounded", "for the whole shot" in s, "")
    # Whole-path: the auto sound must not ask for the swing the hold forbids. Needs a
    # line in the beat, because a silent shot gets no sound clause at all.
    P = ('Daylight. A yard.\n\nMara stands behind a van with closed rear doors. '
         'Mara says: "We wait here."')
    s = run_node(P, plan_only=True)[3]
    check("the held doors are not also heard swinging",
          "already closed" in s and "a door on its hinges" not in s, "")
    check("...while the shot still has its other sound", "an engine outside" in s, "")
    # The shot that stages the opening keeps the sound of one.
    s = run_node('Daylight. A yard.\n\nMara opens the van doors. Mara says: "Here."',
                 plan_only=True)[3]
    check("a staged opening still sounds like one", "a door on its hinges" in s, "")


def test_a_staged_change_gets_both_ends():
    print("\n=== a staged change is anchored at both ends ===")
    # The shot that WORKS the thing gets no held state -- it is asking for that
    # motion -- but it is exactly the shot a reversing LoRA renders backwards. It
    # gets the two ends instead.
    P = ("Daylight. A yard, and a van with its doors closed.\n\n"
         "Mara and Dom stand behind the van.\n\n"
         "Mara opens the van doors and climbs in.\n\n"
         "Dom slams the tailgate shut.")
    shots = [s for s in run_node(P, plan_only=True)[3].split("---") if s.strip()]
    check("shot 1 holds the standing state",
          "already closed at the first frame" in shots[0], "")
    check("shot 2 gets both ends of the opening",
          "The doors are shut at the first frame and open by the last." in shots[1], "")
    check("shot 3 gets both ends of the shutting",
          "The tailgate is open at the first frame and shut by the last." in shots[2], "")
    check("the working shot is not also told the state holds",
          "already" not in shots[1].split("climbs in.")[1], "")
    info = run_node(P, plan_only=True)[2]
    check("info names the anchored shots", "shot(s) 2, 3 stage a change" in info, "")
    check("...and says where reversal is likeliest", "shot 1, which has no previous" in info, "")
    off = run_node(P, plan_only=True, hold_scene_state=False)[3]
    check("the switch turns it off too", "by the last" not in off, "")
    # Continuity and beat share one shot. Two anchors plus two held states would be
    # four continuity sentences on a beat that is one line long.
    B = ("A yard. The gate is open and the blinds are drawn.\n\n"
         "Mara opens the van doors and Dom lifts the lid of the crate.")
    one = run_node(B, plan_only=True)[3]
    check("at most two frame sentences in a shot", one.count("first frame") == 2, "")
    check("...and the beat's own action wins the budget",
          "The doors are shut" in one and "The lid is shut" in one, "")


def test_dialogue_headroom():
    print("\n=== how much of a speaking shot the line does not fill ===")
    # The audio branch is open for the WHOLE shot once there is a line in it, so the
    # seconds the line does not cover are unconditioned audio in a shot the model
    # already knows has a voice. That is where speech carries on past the line and
    # turns into babble.
    P = "\n\n".join([
        'Dan says: "Wait."',
        'Dan says: "Wait there a moment, I need to check the cable before you '
        'start it up."',
        "Nora picks up the spanner."])
    fixed = run_node(P, plan_only=True, anchor="A workshop.",
                     shot_length="fixed", shot_seconds=15.0)[2]
    check("a one-word line in a fixed shot is flagged", "dialogue headroom" in fixed)
    check("...with the words and the seconds", "shot 1: 1 word(s), about 0.4s" in fixed)
    check("...and a long line in the same shot too",
          "shot 2: 15 word(s), about 6.0s" in fixed)
    check("...naming the two ways out", "longer line, or a shorter shot" in fixed)
    # Sized from the beat, the shot follows the line and there is no tail to report.
    beat = run_node(P, plan_only=True, anchor="A workshop.",
                    shot_length="from the beat", shot_seconds=15.0)[2]
    check("sizing from the beat leaves no headroom", "dialogue headroom" not in beat)
    # A shot with no line has no dialogue to outlast; silence covers it instead.
    quiet = run_node("Nora walks to the window.\n\nNora sits down.", plan_only=True,
                     anchor="A workshop.", shot_length="fixed", shot_seconds=15.0)[2]
    check("a shot with no line is not flagged", "dialogue headroom" not in quiet)


def test_introducing_somebody_already_in_position():
    print("\n=== a character introduced in position starts fresh ===")
    # Reported: a character is thrown into the shot and then moved to where they
    # belong. Every shot continues from the previous shot's LAST FRAME, and somebody
    # appearing for the first time is not in it -- so the beat says where they are and
    # the picture says they are nowhere. The picture wins, and they have to arrive out
    # of nothing and travel to the spot.
    mem = "Nora: 34, she, red hair.\nDan: 41, he, dark hair, navy overalls"
    tail = "\n\nNora picks up the spanner."

    def encodes(beat2):
        vae = FakeVAE()
        info = run_node("Nora sets a toolbox on the bench.\n\n" + beat2 + tail,
                        anchor="A workshop.", character_memory=mem, vae=vae)[2]
        return vae.encodes, info

    # In position: the shot cannot inherit a frame that has him in it, so it starts
    # fresh. Three shots, and only shot 3 encodes a keyframe.
    n_placed, info = encodes("Dan is already sitting on the crate, watching her.")
    check("an in-position introduction drops its keyframe", n_placed == 1, str(n_placed))
    check("...and the run says so", "introduces Dan in position" in info)
    check("...naming what it costs", "Costs a cut" in info)
    check("...and how to keep the join", "Write the entrance" in info)
    # Arriving is what the chain is FOR: he walks in from the frame before.
    n_arrive, info2 = encodes("Dan walks in through the side door and looks at her.")
    check("an arriving introduction keeps the chain", n_arrive == 2, str(n_arrive))
    check("...and claims nothing", "introduces Dan in position" not in info2)
    for _b, _want in (("Dan enters the workshop.", True),
                      ("Dan comes back in.", True),
                      ("Nora follows him in.", True),
                      ("Dan steps into the room.", True),
                      ("Dan is at the far wall, watching her.", False),
                      ("Dan stands at the bench.", False),
                      ("Dan waits by the roller door.", False)):
        check(f"arrival={_want}: {_b[:34]!r}", S.arrives_in(_b) == _want)


def test_back_after_a_shot_away():
    print("\n=== somebody back after a shot away ===")
    # Reported: a character's appearance is lost when they walk out of frame and
    # return. Every shot starts from the PREVIOUS shot's last frame, so somebody who
    # was not in that shot is not in the picture this one begins from -- their
    # appearance comes from the sheet text and nothing else, and text drifts where a
    # picture does not.
    P = "\n\n".join(["Nora sets a toolbox on the bench.",
                     "Nora walks out through the side entrance.",
                     "Victor walks in and kneels by the cable, alone in the workshop.",
                     "Nora comes back in and picks up the spanner."])
    mem = "Nora: 34, she, tall, red hair.\nVictor: he, 41, dark hair"
    info = run_node(P, plan_only=True, anchor="A room.", character_memory=mem)[2]
    check("the return is detected", "back after a shot away" in info)
    check("...naming the shot and who", "shot 4: Nora" in info)
    check("...and why the keyframe cannot carry them",
          "not in the picture this one begins from" in info)
    # A reference tag IS the picture that pins them, so the advice differs.
    check("with no tag, it says to add one", "no <Picture N> tag" in info)
    tagged = run_node(P, plan_only=True, anchor="A room.",
                      character_memory="Nora: <picture 1>, 34, she, red hair.\n"
                                       "Victor: he, 41, dark hair",
                      ref_image_1=torch.rand(1, H, W, 3))[2]
    check("with a tag, it says they are pinned", "All of them carry a reference" in tagged)
    check("...and does not ask for one", "no <Picture N> tag" not in tagged)
    # Nobody leaves, nobody returns.
    straight = run_node("Nora walks in.\n\nNora sits down.", plan_only=True,
                        anchor="A room.", character_memory=mem)[2]
    check("a chain nobody leaves reports nothing",
          "back after a shot away" not in straight)

    # And the fix: a frame from the middle of the last shot they WERE in, sent as a
    # reference on the shot they come back to. The middle because somebody walking
    # out is gone by the last frame and somebody walking in is missing from the first.
    seen = []
    orig = FakeCLIP.tokenize

    def spy(self, text, minimax_ref_items=None, **kw):
        seen.append(sum(1 for it in (minimax_ref_items or []) if it["type"] == "image"))
        return orig(self, text, minimax_ref_items=minimax_ref_items, **kw)

    FakeCLIP.tokenize = spy
    try:
        base = ["Nora sets a toolbox on the bench.",
                "Nora walks out through the side entrance.",
                "Victor walks in and kneels by the cable, alone in the workshop."]
        seen.clear()
        info = run_node("\n\n".join(base + ["Nora comes back in and picks up the spanner."]),
                        anchor="A room.", character_memory=mem)[2]
        # shot 1 no keyframe and no refs; 2 and 3 their keyframe; 4 keyframe + recovered.
        check("the return shot gets a second picture", seen[1:] == [0, 1, 1, 2],
              str(seen[1:]))
        check("...and the run says whose face and from where",
              "recovered a face for Nora on shot 4, from shot 2" in info)
        # NARROW: the recovered frame carries whoever else was on screen, so a return
        # into company is reported and left alone rather than risking a second person.
        seen.clear()
        multi = run_node("\n\n".join(
            base + ["Nora comes back in and Victor hands her the spanner."]),
            anchor="A room.", character_memory=mem)[2]
        check("a return into company is left alone", seen[1:] == [0, 1, 1, 1],
              str(seen[1:]))
        check("...and nothing is claimed for it", "recovered a face" not in multi)
        # A tagged character already has their own reference travelling with them.
        seen.clear()
        tagged = run_node("\n\n".join(base + ["Nora comes back in and picks up the spanner."]),
                          anchor="A room.",
                          character_memory="Nora: <picture 1>, 34, she, red hair.\n"
                                           "Victor: he, 41, dark hair",
                          ref_image_1=torch.rand(1, H, W, 3))[2]
        check("a tagged character is not given a second picture",
              "recovered a face" not in tagged)
        # THE RECOVERED FRAME HAS TO BE CLAIMED IN THE PROSE. A picture the prompt
        # refers to is that subject; one it never mentions is ANOTHER subject. Sent
        # unclaimed, a frame of somebody is read as a second person who looks exactly
        # like them -- same face, same clothes -- beside the one the beat asked for.
        tagseen = []
        _o = FakeCLIP.tokenize

        def _spy(self, text, minimax_ref_items=None, **kw):
            tagseen.append((sum(1 for it in (minimax_ref_items or [])
                                if it["type"] == "image"),
                            re.findall(r"<Picture \d+>", text), text))
            return _o(self, text, minimax_ref_items=minimax_ref_items, **kw)

        FakeCLIP.tokenize = _spy
        try:
            run_node("\n\n".join(["Dan walks into the workshop alone.",
                                  "Dan walks out through the side door.",
                                  "Nora kneels by the cable, alone.",
                                  "Dan comes back in and picks up the spanner."]),
                     anchor="A workshop.",
                     character_memory="Nora: 34, she, red hair.\n"
                                      "Dan: he, 41, dark hair, navy overalls")
        finally:
            FakeCLIP.tokenize = _o
        _pics, _tags, _txt = tagseen[4]          # [0] is the negative; shot 4
        check("the return shot carries the recovered frame and its keyframe",
              _pics == 2, str(_pics))
        check("...and the prose claims the recovered one", _tags == ["<Picture 1>"],
              str(_tags))
        check("...on the entry of the person it depicts",
              "Dan: <Picture 1>," in _txt, _txt[:80])
        # THE SOURCE has to be solo too, not just the destination. A frame is a
        # picture of everyone in it, so one captured from a shared shot brings the
        # other person back into a shot that does not call for them -- which is the
        # second character turning up uninvited.
        shared = ["Nora walks into the workshop.",
                  "Victor comes in and Nora hands him the spanner.",
                  "Victor walks in and kneels by the cable, alone.",
                  "Nora comes back in and picks up the toolbox."]
        info2 = run_node("\n\n".join(shared), anchor="A room.", character_memory=mem)[2]
        src = re.search(r"recovered a face for Nora on shot 4, from shot (\d+)", info2)
        check("the frame comes from a shot that was hers alone",
              bool(src) and src.group(1) == "1", src.group(1) if src else "none")
        # Never on screen alone: no clean frame exists, so nothing is sent. Leaving
        # the sheet text to carry her beats importing somebody who is not in the shot.
        only_shared = run_node("\n\n".join(shared[1:]), anchor="A room.",
                               character_memory=mem)[2]
        check("no solo frame anywhere means nothing is recovered",
              "recovered a face" not in only_shared)
    finally:
        FakeCLIP.tokenize = orig


def test_a_name_with_no_entry_end_to_end():
    print("\n=== a person the sheet never describes ===")
    P = "\n\n".join(["Maya walks in.", "Alex says hello to Maya.",
                     "Alex walks to the window.",
                     "Maya stands up and Alex takes her hand."])
    mem = "Maya: she, 27, grey coat.\nJon: he, 35, jeans."
    info, script = run_node(P, plan_only=True, anchor="A room.",
                            character_memory=mem)[2:4]
    check("the run reports the undescribed person", "Alex, who has no entry" in info)
    check("...naming the shots", "shot(s) 2, 3, 4 name Alex" in info)
    check("...and what to do about it", "use one name throughout" in info)
    # Nothing is acted on: whether Alex is Jon under another name or a third person
    # is not answerable from the text, and guessing would rewrite the script.
    shots = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("the script is not rewritten", "Alex says hello to Maya." in shots[1])
    check("...and no entry is invented for them", "Alex:" not in script)
    # A sheet that covers everyone says nothing.
    clean = run_node("Maya walks in.\n\nJon greets Maya.", plan_only=True,
                     anchor="A room.", character_memory=mem)[2]
    check("a complete sheet is not reported", "has no entry" not in clean)


def test_references_ride_with_the_keyframe():
    print("\n=== a reference and the keyframe ride together ===")
    # I broke this, then removed it, on the conclusion that fl2va could not carry an
    # identity reference at all. Wrong, and the old node's own comment said so:
    # ComfyUI packs both channels in AGREEING orders -- model_base.py:2183-2191 builds
    # cond_video_latents as keyframe latents then ref latents, and PackedLayout emits
    # keyframe "cond" segments then ref "ref_img" ones -- so a shot takes references
    # AND a real keyframe. The keyframe anchors the first frame; a reference only
    # says who somebody is. They were never alternatives.
    #
    # Which image is the first frame is decided by resolved_frame_index in
    # minimax_keyframes, NOT by a label's number. The labels only have to line up with
    # the <Picture N> tags in the prompt, so references keep slots 1..N and the
    # handoff is appended after them where it disturbs no numbering.
    seen = []
    orig = FakeCLIP.tokenize

    def spy(self, text, minimax_ref_items=None, **kw):
        n = sum(1 for it in (minimax_ref_items or []) if it["type"] == "image")
        seen.append((n, len(re.findall(r"<Picture \d+>", text))))
        return orig(self, text, minimax_ref_items=minimax_ref_items, **kw)

    FakeCLIP.tokenize = spy
    try:
        mem = "Kate: <picture 1>, 22, she, blonde hair.\nMike: he, 35, jeans"
        seen.clear()
        # Mike ARRIVES, so the chain is kept and this stays a test of the picture
        # roster rather than of the fresh-start rule for an in-position introduction.
        run_node("Kate walks in.\n\nMike walks in and Kate sits beside him.\n\n"
                 "Mike stands up.",
                 anchor="A room.", character_memory=mem,
                 ref_image_1=torch.rand(1, H, W, 3))
        shots = seen[1:]                     # seen[0] is the negative
        # shot 1: the reference only -- no previous frame to hand over yet.
        # shot 2: the reference AND the keyframe. This is the pairing I forbade.
        # shot 3: the guard drops Kate, so no tag, so no reference; keyframe only.
        check("the roster is ref + keyframe, not one or the other",
              [p for p, _ in shots] == [1, 2, 1], str([p for p, _ in shots]))
        check("the shot carrying both still names exactly one picture",
              shots[1] == (2, 1), str(shots[1]))
        check("a shot the guard trimmed carries no reference", shots[2][1] == 0)
        # The tag stays IN the prompt: it is the binding between the picture and the
        # person, which comfy_extras/nodes_minimax_h3.py tells you to write there.
        info, script = run_node("Kate walks in.\n\nKate sits down.", plan_only=True,
                                anchor="A room.", character_memory=mem,
                                ref_image_1=torch.rand(1, H, W, 3))[2:4]
        check("the binding survives into the script", "<Picture 1>" in script)
        check("...on the person it depicts", "Kate: <Picture 1>, 22" in script)
        check("the run explains the pairing", "ride alongside the keyframe" in info)
        # No reference connected: the handoff is the only picture, which is H3's own
        # first-frame shape.
        seen.clear()
        run_node("A room.\n\nOne.\n\nTwo.\n\nThree.")
        check("a chain with no reference keeps the keyframe picture",
              any(pics == 1 for pics, _ in seen))
    finally:
        FakeCLIP.tokenize = orig


def test_sound_survives_silencing():
    print("\n=== a described sound is not silenced away ===")
    # No space named, so no room tone -- this test is about the SILENCE path, and a
    # bed under every shot would mean nothing is silenced and there is nothing to see.
    P = ("Two people, late evening.\n\n"
         "Maya: 27, grey coat.\n\n"
         "Maya lies still.\n\n"
         "The chain drags and rattles beside her.\n\n"
         'Jon says: "Get up."')
    vae = FakeAudioVAE()
    info = run_node(P, audio_vae=vae, auto_sound=False)[2]
    check("the beat that describes a sound keeps its audio",
          "describe a sound IN THE BEAT" in info)
    # NAMED, not counted. "2 shot(s) have an open branch" tells a reader that two
    # of eleven can babble and gives no way to find out which two -- and the note's
    # own advice is that the beat's own sound wording opened it, which cannot be
    # acted on without knowing which beat.
    check("...and is named", "shot(s) 2 have no line but either describe a sound" in info)
    check("the beat with none is silenced",
          "shot(s) 1 have no quoted line and no sound" in info)
    check("...and the guidance says what silence actually is",
          "not 'no speech', it is 'no sound at all'" in info)
    check("...and how to score a scene", "DESCRIBE it in the prose" in info)
    check("...and warns off a label", "read as text to draw" in info)
    # With silencing off, nothing is silenced and nothing is claimed about it.
    off = run_node(P, silence_nonspeech=False)[2]
    check("silencing off silences nothing", "conditioned on real silence" not in off)


def test_auto_sound_end_to_end():
    print("\n=== sound generated from the prompt ===")
    # No space named: this test isolates the sound a beat's own ACTION implies, and
    # room tone would put a bed on every shot including the ones meant to be bare.
    #
    # Derived sound is TEXT ONLY and can never unsilence a shot, so it lands only on
    # shots whose audio branch is already open: ones with a line, or with a sound the
    # author wrote. A shot with neither stays pinned to silence -- the mouth follows
    # the audio, and an inference is not reason enough to let it move.
    P = ("Two people, late evening.\n\n"
         "Maya: 27, grey coat. Wrists cuffed behind back.\n\n"
         "Jon walks in holding a pair of scissors and says: \"Hold still.\"\n\n"
         "Jon walks to the bench and looks at the box.\n\n"
         "Maya lies still.\n\n"
         "The chain drags and rattles beside her.")
    imgs, audio, info, script = run_node(P, plan_only=True)[:4]
    sh = [" ".join(x.split()) for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    # Shot 1 speaks, so its branch is open anyway and the action's sound is added.
    check("walking is heard on the shot that speaks", "footsteps" in sh[0])
    check("...and the scissors", "blades through fabric" in sh[0])
    check("...in the open form, because it has a line", "It sounds like" in sh[0])
    # Shots 2 and 3 have no line and no sound of their own. With auto_sound OFF
    # they are pinned silent and told nothing about sound, since the clause would
    # describe an acoustic that is not there. This is the one that was babbling,
    # and it is still the behaviour with the switch off -- an ambient bed read from
    # the anchor is what opens them now, chosen deliberately.
    q_info, q_script = run_node(P, plan_only=True, auto_sound=False)[2:4]
    q_sh = [" ".join(x.split()) for x in re.split(r"(?=\[Shot )", q_script)
            if x.strip()]
    check("a shot with no line gets no derived sound (auto_sound off)",
          "footsteps" not in q_sh[1])
    check("...and no sound sentence at all",
          "sounds like" not in q_sh[1] and "only sound" not in q_sh[1])
    check("a beat staging nothing audible gets nothing", "sounds like" not in sh[2])
    # What you wrote wins: a beat describing its own sound is left alone AND stays open.
    check("a beat with its own sound is not overwritten", "It sounds like" not in sh[3])
    check("...and it still counts as asking for audio",
          "either describe a sound IN THE BEAT or" in info)
    check("info lists the shots it scored", "were given the sound" in info)
    check("...saying it can never unsilence one", "never unsilence a shot" in info)
    # Sound is counted apart from the continuity guards, which ask for the opposite
    # thing -- for something to stay as it is rather than to happen.
    check("the balance separates sound from guards", "sound " in info.split("balance")[1][:120])
    # Off, nothing is added and those shots go back to being silenced.
    off = run_node(P, plan_only=True, auto_sound=False)[3]
    check("auto_sound off adds nothing", "It sounds like" not in off)


def test_room_tone_under_every_shot():
    print("\n=== the room is heard even when nothing happens ===")
    # Real footage has a bed under the events. Digital silence between them is what
    # makes a scene sound staged rather than recorded.
    P = ("A cold concrete basement with bare walls.\n\n"
         "Maya: 27, grey coat.\n\n"
         "Jon walks in and says: \"Get up.\"\n\n"
         "Maya lies still.\n\n"
         "Maya breathes hard, the sound of it loud in the room.")
    imgs, audio, info, script = run_node(P, plan_only=True)[:4]
    sh = [" ".join(x.split()) for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("a shot that speaks carries the room too",
          "hard walls giving the sound back" in sh[0])
    check("the acting shot still gets its events", "footsteps" in sh[0])
    check("info names the acoustic", "room tone read from the scene" in info)
    # The ambient bed goes under shots whose audio branch is ALREADY open -- ones
    # with a line, or with a sound the author wrote. It is NOT put under a wordless
    # shot, and that reversal is the point.
    #
    # "Ambience everywhere" was tried and does not work. The bed was allowed to open
    # a branch, which is the one thing nothing inferred here may do: an open branch
    # on a joint model fills itself, and at 4-8 steps the last audio step clears
    # 50%-30% of the denoising in a single jump, so what it fills with is a voice.
    # Ambience everywhere and silence are mutually exclusive by construction -- the
    # silence latent IS the audio, and there is no room in it for a room tone.
    check("the speaking shot carries the bed", "low hum off the strip light" in sh[0])
    check("the wordless shot is left silent",
          "low hum off the strip light" not in sh[1])
    check("...and info says so", "ambient bed read from the anchor" in info)
    # THE CONDITIONING, not just the text. `sounded` decides whether the silence
    # latent is applied at render time, and the bed used to be counted into it --
    # so a wordless shot was reported as one that "describes a sound IN THE BEAT",
    # which it does not, and its branch was left open. plan_only shows the text
    # either way, so this reads the node's own account of which shots are pinned.
    _pin = re.search(r"shot\(s\) ([\d, ]+) have no quoted line and no sound described",
                     info)
    _open = re.search(r"shot\(s\) ([\d, ]+) have no line but either", info)
    check("the wordless shot is pinned to silence",
          _pin is not None and "2" in _pin.group(1), _pin.group(1) if _pin else "none")
    check("...and is not counted as describing its own sound",
          _open is None or "2" not in _open.group(1),
          _open.group(1) if _open else "none")

    # THE SILENCE PATH, which is still the path with auto_sound OFF.
    #
    # Reported twice: auto_sound was moving the mouth and babbling. H3 is joint, so
    # a free audio branch fills itself with a VOICE and the face lip-syncs to it. No
    # wording suppresses that -- only the silent keyframe does, and it pins the
    # whole shot. Scoring the film from the anchor re-opens that branch on purpose,
    # chosen on 2026-09-06 with the trade stated; auto_sound is the way back.
    q_info, q_script = run_node(P, plan_only=True, auto_sound=False)[2:4]
    q_sh = [" ".join(x.split()) for x in re.split(r"(?=\[Shot )", q_script)
            if x.strip()]
    check("auto_sound off: a wordless shot is silenced",
          "conditioned on real silence" in q_info)
    check("...and the room does not go under it",
          "hard walls" not in q_sh[1], q_sh[1][-70:])
    check("...and it is told nothing about sound",
          "sounds like" not in q_sh[1] and "only sound" not in q_sh[1])
    # A sound the AUTHOR wrote used to keep that shot open even with no line -- and
    # an open branch is exactly where the invented voice and the lip-sync came from.
    # mouths_shut_when_no_line, on by default, silences it instead. The old rule is
    # still the rule with the switch off, and that is the trade, stated both ways.
    # auto_sound stays ON here: the room tone this asserts comes from it, and the
    # point is the MOUTH guard, not the bed.
    off = run_node(P, plan_only=True, mouths_shut_when_no_line=False)[3]
    off_sh = [b for b in off.split("---") if b.strip()]
    check("a sound you wrote keeps the branch open, guard off",
          "hard walls giving the sound back" in off_sh[2])
    check("...in the closed form, because it has no line",
          "The only sound" in off_sh[2])
    check("...while on, that shot is silenced so the mouth cannot move",
          "The only sound" not in q_sh[2]
          and "Mouths in the shot stay closed" in q_sh[2])
    check("...and info explains the mouth",
          "so the mouths could be held shut" in q_info)
    # A scene naming no space gets no bed, and the silence guard still applies.
    plain = run_node("Two people talking.\n\nHe waits.\n\nShe waits.",
                     plan_only=True)[2]
    check("no space named, no room tone", "room tone read" not in plain)
    check("...and no ambient bed either", "ambient bed read" not in plain)


def test_the_decode_keeps_the_vae_it_is_about_to_use():
    print("\n=== eviction does not throw away the VAE it needs ===")
    # Reported: forced eviction costing 37% of wall-clock on a 3090. The half that is
    # wrong on EVERY card is this one -- free_memory ran with keep_loaded=[], which
    # unloads every resident model including the video VAE, and ComfyUI then reloads
    # it three lines later to run the decode. Peak VRAM is identical either way (the
    # VAE has to be resident to decode), so the round trip was pure cost.
    class _LM:                      # stands in for ComfyUI's LoadedModel
        def __init__(self, m):
            self.model = m

    calls = []
    _orig_free = _mm.free_memory
    model, vae, avae = FakeModel(), FakeVAE(), FakeAudioVAE()
    _mm.current_loaded_models = [_LM(model), _LM(vae), _LM(avae)]

    def spy(memory_required, device, keep_loaded=(), **kw):
        calls.append((memory_required, [lm.model for lm in keep_loaded]))

    _mm.free_memory = spy
    try:
        run_node("A room.\n\nOne.\n\nTwo.", model=model, vae=vae, audio_vae=avae)
    finally:
        _mm.free_memory = _orig_free
        del _mm.current_loaded_models

    check("eviction still runs", bool(calls))
    # Two call sites per shot: _evict_all_but before sampling, free_first before decode.
    pre_sample = [c for c in calls if model in c[1]]
    pre_decode = [c for c in calls if vae in c[1]]
    check("before sampling, the DiT is what is kept", bool(pre_sample))
    check("...and the VAEs are not", all(vae not in c[1] for c in pre_sample))
    check("before decode, the video VAE is kept", bool(pre_decode))
    check("...and the audio VAE with it, used on the next line",
          all(avae in c[1] for c in pre_decode))
    check("...while the DiT goes, which is what makes the decode fit",
          all(model not in c[1] for c in pre_decode))
    # Not changed, and deliberately: sizing the request needs real hardware.
    check("the request is still unsized", all(c[0] >= 1e29 for c in calls))
    # A model ComfyUI does not hold cannot be kept, and must not raise.
    _mm.current_loaded_models = []
    check("nothing resident, nothing kept", S._resident([model, vae]) == [])
    del _mm.current_loaded_models
    check("no current_loaded_models at all is survivable", S._resident([model]) == [])


def test_finished_shots_are_held_in_half_precision():
    print("\n=== the chain does not crowd the weights out of RAM ===")
    # ComfyUI offloads models to system RAM rather than discarding them, so a shot
    # boundary is a PCIe copy while that RAM is there and a disk read once it is not.
    # The finished chain is the largest thing this node holds and the one thing it can
    # shrink: a 107s chain at 1056x608 is 18.5GB as float32 and 9.3GB as float16,
    # against ~39GB of weights on a 64GB machine.
    imgs = run_node("A room.\n\nOne.\n\nTwo.\n\nThree.")[0]
    check("what comes out is still float32", imgs.dtype == torch.float32, str(imgs.dtype))
    check("...and still in range",
          float(imgs.min()) >= 0.0 and float(imgs.max()) <= 1.0)
    # Free, not a trade: fp16 resolves far finer than the 8 bits the output has.
    x = torch.rand(100000)
    err = (x - x.half().float()).abs().max().item()
    check(f"fp16 error {err:.1e} is inside one 8-bit step {1 / 255:.1e}", err < 1 / 255)
    # With cleanup off the frames stay where they were; nothing is converted, and the
    # concat must not trip over a dtype it did not expect.
    off = run_node("A room.\n\nOne.\n\nTwo.", cleanup_between_shots=False)[0]
    check("cleanup off still returns float32", off.dtype == torch.float32, str(off.dtype))


def test_detail_trend():
    print("\n=== the chain is measured for softening ===")
    # Every boundary decodes a shot, takes its LAST frame and re-encodes it as the
    # next shot's keyframe. That round trip is lossy and it runs on the model's own
    # output, so shot 11 is sampled from a picture that has been through ten
    # decode/encode cycles. The softening is invisible shot to shot and obvious end
    # to end, which is exactly the kind of thing to measure rather than argue about.
    sharp = torch.rand(48, 48, 3)
    soft = sharp.clone()
    for _ in range(4):
        soft[1:-1, 1:-1] = (soft[:-2, 1:-1] + soft[2:, 1:-1]
                            + soft[1:-1, :-2] + soft[1:-1, 2:]) / 4
    check("a blurred frame measures less detail",
          S.frame_detail(sharp)[0] > S.frame_detail(soft)[0])
    check("a flat frame measures no detail", S.frame_detail(torch.zeros(8, 8, 3))[0] == 0)
    check("a 1px frame does not divide by zero", S.frame_detail(torch.rand(1, 1, 3)) == (0.0, 0.0))
    # The report only claims a trend when there is one.
    falling = S.detail_report([(0.09, .2), (0.08, .2), (0.07, .2), (0.06, .2)])
    check("a falling chain is called out", "DOWN 33%" in falling)
    check("...with the cause named", "re-encodes it as the next" in falling)
    check("...and a way out", "restart_after_removal" in falling)
    check("a flat chain is not alarming",
          "flat within" in S.detail_report([(0.09, .2), (0.089, .2), (0.091, .2)]))
    check("one shot claims no trend", S.detail_report([(0.09, .2)]) == "")
    check("no shots, no line", S.detail_report([]) == "")
    check("the run reports it", "detail per shot" in run_node("A room.\n\nOne.\n\nTwo.")[2])
    check("...and plan_only does not",
          "detail per shot" not in run_node("A room.\n\nOne.\n\nTwo.", plan_only=True)[2])


def test_av_stays_in_sync():
    print("\n=== the sound is as long as the picture ===")
    # Reported: video and audio out of sync. Each shot's audio latent is
    # round(frames / 24 * 40), exact only when the frame count divides by 3, so most
    # lengths leave the sound up to 8.3 ms off its own picture -- and concatenated
    # with equal shot lengths that error has the same sign every time and adds up.
    for n_beats in (2, 3, 5):
        P = "A room.\n\n" + "\n\n".join(f"Beat {i}." for i in range(n_beats))
        imgs, audio, info, script, fps_shot, total, shots, secs = run_node(P)
        sr = audio["sample_rate"]
        v = total / S.H3_FPS
        a = audio["waveform"].shape[-1] / sr
        drift_ms = abs(a - v) * 1000
        check(f"{n_beats} beats: sound matches picture within a sample "
              f"({drift_ms:.3f} ms)", drift_ms < 1.0)
        check(f"...and the frame count is what the video reports",
              imgs.shape[0] == total)
    # It is corrected per shot, so every interior cut lands too -- not just the
    # total duration at the end.
    check("the correction is reported", "realigned to the picture" in
          run_node("A room.\n\nOne.\n\nTwo.")[2])


def test_nothing_wearable_is_ever_added():
    """No clothing is invented, WHATEVER the character memory says.

    The hand-written cases below it check specific wardrobes. This one is the
    general claim, and it is made two ways because either alone has a hole:

      1. A SWEEP over awkward wardrobes x beat orderings, asserting the script's
         garment vocabulary against the author's own. Catches a known garment word
         appearing where the prompt never asked for it.

      2. Every WORD the node adds that the prompt did not contain, checked for
         anything wearable. A garment word that is in no hand-written list -- the
         hole in (1) -- shows up here, because this asserts on what the node
         actually emitted rather than on a list somebody remembered to write.

    Reported repeatedly, and each earlier fix looked complete because the wardrobe
    that exposed the next one was not in the tests."""
    print("\n=== nothing wearable is ever added ===")
    garment = re.compile(
        r"\b(leggings|stockings|tights|pantyhose|hold-?ups|nylons|hosiery|socks|"
        r"panties|knickers|thong|g-?string|briefs|boxers|underwear|undies|"
        r"jockstrap|bra|bralette|brassiere|corset|bustier|camisole|undershirt|"
        r"shorts|trousers|jeans|slacks|chinos|skirt|kilt|joggers|jeggings|"
        r"culottes|dungarees|overalls|dress|gown|robe|jumper|sweater|sweatshirt|"
        r"hoodie|cardigan|jacket|coat|shirt|blouse|t-?shirt|tee|top|tunic|boots|"
        r"shoes|trainers|sneakers|sandals|heels|gloves|mittens|scarf|hat|belt|"
        r"apron|cape|poncho|shawl|swimsuit|bikini|leotard)\b", re.I)
    wardrobes = (
        # A tagged entry: the <Picture N> made the head noun "2" and broke the
        # name lookup for exactly the garment a reference was pinning.
        "McKenna: <Picture 1>, she, 22, Shiny white crop top, "
        "chastity belt <Picture 2>, blue jeans shorts.",
        "Kate: she, 30, blouse, panties, skirt.",
        "Mara: she, 25, red dress.",
        "Jon: he, 50.",
        "Bea: she, 28, corset, stockings, garter belt, heels.",
        "Ana: she, 19, a t-shirt, tights, ankle socks, trainers.",
        "Dee: she, 40, sundress.\nEve: she, 41, sundress.",
    )
    beats = (
        "{N} stands by the chair.",
        "{N} takes off her clothes.",
        "{N} pulls the skirt down.",
        "{N} falls to the floor.",
        "{N} sits, wrists bound above her head.",
        "{N} walks out of frame.",
        "{N} comes back in.",
        "{N} is stripped bare.",
        # Plain removals that leave a region with nothing named under it -- the
        # bare-region clause fires here, and it is the one guard that talks about
        # an uncovered part of the body, so it is the likeliest to name a garment.
        "{N} takes off the skirt.",
        "{N} takes off her tights.",
        "{N} takes off the dress.",
        "{N} takes off her boots.",
    )
    invented, added, runs = {}, {}, 0
    for w in wardrobes:
        n = w.split(":", 1)[0].strip()
        for combo in itertools.islice(itertools.permutations(beats, 3), 0, 12):
            P = w + "\n\n" + "\n\n".join(b.format(N=n) for b in combo)
            try:
                script = run_node(P, plan_only=True)[3]
            except Exception as e:
                invented.setdefault("RAISED " + repr(e)[:50], P[:60])
                continue
            runs += 1
            allowed = {x.lower().replace("-", "") for x in garment.findall(P)}
            for x in garment.findall(script):
                x = x.lower().replace("-", "")
                if x not in allowed:
                    invented.setdefault(x, P[:60])
            # ...and every word the node ADDS, so a garment word nobody listed
            # above still surfaces.
            src = {t for t in re.findall(r"[a-z][a-z-]{2,}", P.lower())}
            for word in re.findall(r"[a-z][a-z-]{2,}", script.lower()):
                if word not in src:
                    added.setdefault(word, P[:60])
    check(f"no garment invented across {runs} prompts", not invented,
          f"invented {sorted(invented)[:6]}")
    wearable = re.compile(
        r"(leggings|stocking|tight|pantyhose|nylon|hosiery|sock|panti|knicker|"
        r"thong|brief|boxer|underwear|undies|bra|corset|camisole|short|trouser|"
        r"jean|skirt|kilt|jogger|dress|gown|robe|jumper|sweater|hoodie|cardigan|"
        r"jacket|coat|shirt|blouse|tunic|boot|shoe|trainer|sneaker|sandal|heel|"
        r"glove|mitten|scarf|apron|cape|poncho|shawl|bikini|leotard)", re.I)
    hits = sorted(w for w in added if wearable.search(w))
    check("no wearable word is ever ADDED by the node", not hits, f"added {hits[:6]}")


def test_no_garment_is_ever_invented():
    """END TO END: no garment word reaches a prompt unless the author wrote it.

    Reported repeatedly -- legwear appearing in shots that never asked for it. The
    unit tests cover each builder alone; this one reads the finished script and
    asserts against the AUTHOR'S OWN vocabulary, which is the only check that
    catches a word introduced by a path nobody thought to test."""
    print("\n=== nothing is invented ===")
    garment = re.compile(
        r"\b(leggings|stockings|tights|pantyhose|hold-?ups|nylons|hosiery|socks|"
        r"panties|knickers|thong|briefs|boxers|underwear|undies|bra|bralette|"
        r"corset|camisole|shorts|trousers|jeans|slacks|chinos|skirt|kilt|joggers|"
        r"dress|gown|robe|jumper|sweater|sweatshirt|hoodie|cardigan|jacket|coat|"
        r"shirt|blouse|t-shirt|tee|top|tunic|boots|shoes|trainers|sneakers|"
        r"sandals|heels|gloves|mittens|scarf|hat|belt)\b", re.I)
    cases = (
        ("a request, nothing removed",
         "McKenna: she, 22, crop top, chastity belt, blue jeans shorts.\n"
         "Dan: he, 40, t-shirt.\n\n"
         "McKenna stands by the chair.\n\n"
         'McKenna approaches Dan. "Will you take the chastity belt off?"\n\n'
         "McKenna sits in the chair."),
        ("a strip with nothing underneath",
         "McKenna: she, 22, crop top, blue jeans shorts.\n\n"
         "McKenna stands.\n\n"
         "Dan pulls off her jeans shorts.\n\n"
         "McKenna sits."),
        ("a strip with a layer underneath",
         "Kate: she, 30, blouse, panties, skirt.\n\n"
         "Kate stands.\n\n"
         "Kate takes off her skirt.\n\n"
         "Kate walks away."),
        ("one garment only",
         "Mara: she, 25, red dress.\n\n"
         "Mara stands in the room.\n\n"
         "Mara takes off her dress.\n\n"
         "Mara sits down."),
        ("no garment at all",
         "Jon: he, 50.\n\n"
         "Jon walks in.\n\n"
         "Jon sits down."),
    )
    for name, P in cases:
        allowed = {w.lower() for w in garment.findall(P)}
        script = run_node(P, plan_only=True)[3]
        invented = {w.lower() for w in garment.findall(script)} - allowed
        check("nothing invented: %s" % name, not invented,
              "invented %s" % sorted(invented))


def test_a_garment_keeps_its_description():
    """END TO END: a displaced garment keeps the sheet's words, and a garment put
    back stops being described as displaced.

    Reported as shorts coming back "a different kind". They were not re-invented by
    the model out of nothing: the node itself named them twice, once fully from the
    sheet and once bare from the beat."""
    print("\n=== a garment keeps its description ===")
    sheet = ("McKenna: she, 22, Shiny white crop top, blue jeans shorts, "
             "black leather boots.\n\n")
    # The beat uses the SHORT name; the guard must still use the sheet's.
    s = run_node(sheet + "McKenna stands.\n\n"
                 "McKenna pulls the shorts down.\n\n"
                 "McKenna looks at the window.", plan_only=True)[3]
    check("the displacement carries the sheet's full name",
          "blue jeans shorts pulled down" in s)
    check("...and never a bare one beside it", "the shorts pulled" not in s)
    # Put back up, under a different name than it went down under.
    s2 = run_node(sheet + "McKenna stands.\n\n"
                  "McKenna pulls her blue jeans shorts down.\n\n"
                  "McKenna pulls the shorts back up.\n\n"
                  "McKenna sits.", plan_only=True)[3]
    last = s2.split("[Shot 4]")[-1]
    check("a garment put back is no longer displaced", "pulled down" not in last)
    check("...and is not both up and down at once",
          not ("pulled down" in last and "pulled up" in last))


def test_a_removal_always_names_hands():
    """END TO END: the removal clause names WHOSE hands, in a real prompt.

    Reported as the action happening twice -- she takes the shorts off herself and
    he takes them off as well. The beat named her; the clause named nobody, so the
    shot carried two removals and the model gave the unattributed one to the other
    person in the frame.

    The cause was upstream of the agent logic, which was right all along: a sheet
    paragraph folded into the scene never reaches pull_character_sheets -- it only
    ever sees the beat -- so `sheet` was "" for the whole run, and the wearer and
    the cast read off it came back empty for EVERY shot."""
    print("\n=== a removal names whose hands ===")
    two = ("McKenna: she, 22, Shiny white crop top, blue jean shorts.\n"
           "Dan: he, 40, t-shirt, jeans.\n\n")
    s1 = run_node(two + "McKenna takes off her jean shorts.", plan_only=True)[3]
    check("she undresses herself",
          "McKenna takes off her jean shorts" in s1
          and "away by the last frame" in s1)
    s2 = run_node(two + "Dan pulls off her jean shorts.", plan_only=True)[3]
    check("...and he does it when the beat says so",
          "Dan pulls off her jean shorts" in s2
          and "away by the last frame" in s2)
    # WHOSE HANDS has to be somewhere in the shot -- that is the guarantee, and
    # the beat supplies it here. The completion clause may then be agentless:
    # restating who and what is the same removal written twice, which rendered
    # it twice. What must never happen is a shot where NOBODY has the hands, so
    # a beat that does not stage the removal still gets the named clause.
    for _s, _who in ((s1, "McKenna"), (s2, "Dan")):
        check("the hands are named in the shot (%s)" % _who, _who in _s)
    _unstaged = run_node(two + "remove: shorts\nMcKenna stands by the door.",
                         plan_only=True)[3]
    check("a removal the beat does not stage still names the hands",
          "McKenna takes the blue jean shorts off during this shot" in _unstaged)
    # THE ACTION, not just the end state. This clause exists because scrubbing the
    # scene stops a garment being DESCRIBED and does not tell the model to complete
    # the removal -- the last frame is the next shot's keyframe, and a cut still in
    # progress hands on a garment half worn. A version that said only "the shorts
    # are away by the last frame" asserted the outcome and never the act, and
    # garments stopped coming off.
    for _s, _lbl in ((s1, "hers"), (s2, "his"), (_unstaged, "unstaged")):
        check("the clause says the removal HAPPENS (%s)" % _lbl,
              "off during this shot" in _s)
    # One person in the shot is still that person.
    solo = run_node("Mara: she, 25, red dress.\n\nMara takes off her dress.",
                    plan_only=True)[3]
    check("alone, it is still her hands",
          "Mara takes off her dress" in solo
          and "away by the last frame" in solo)


def test_hands_and_holds_follow_the_beat():
    """Two defects seen in one real script, on neutral text.

    1. ONE agent for the whole beat. A beat that takes a coat off and then asks
       about a scarf gave BOTH removals to the other person -- her own coat came
       off by his hands. Attribution is per garment now, on the garment's own
       clause, and the first-named-acts fallback reads that clause too.

    2. The restraint hold could only be cleared by a `remove:` line. auto_remove
       filters hardware out of infer_removals on purpose, so a script that unlocks
       the cuffs IN ITS PROSE never cleared it: every later shot went on saying they
       stay closed and fastened, over hardware the beat had put on the floor. And
       clearing the latch alone was not enough -- the sheet still listed the cuffs,
       so the next shot read them back out and latched again."""
    print("\n=== hands and holds follow the beat ===")
    mem = "Kate: she, 30, blue coat, wool scarf, grey jumper.\nSam: he, 34, shirt."
    s1 = run_node("A hallway.\n\n"
                  "Kate takes off her coat and hangs it up. She finds Sam and asks "
                  'him: "Can you get this scarf off?"\n\n'
                  "Sam unties the scarf. Kate takes off her jumper.",
                  plan_only=True, character_memory=mem)[3]
    sh = [x for x in s1.split("---") if x.strip()]
    check("her own coat is by her hands",
          "Kate takes off her coat" in sh[0]
          and "away by the last frame" in sh[0])
    check("...not the person she asks about something else",
          "Sam takes the blue coat off" not in sh[0])
    check("the garment she asked about is by his",
          "Sam takes the wool scarf off during this shot" in sh[1]
          or "Sam unties the scarf" in sh[1])
    check("...and the one she removes herself is hers",
          "Kate takes off her jumper" in sh[1]
          and "away by the last frame" in sh[1])

    hw = "Kate: she, 30, coat, handcuffs.\nSam: he, 34, shirt."
    for _beats, _held, _lbl in (
            ("Kate sits with the handcuffs on.\n\nSam looks at the handcuffs.\n\n"
             "Kate waits.", True, "a mention does not unlock them"),
            ("Kate sits with the handcuffs on.\n\nKate tugs at the handcuffs.\n\n"
             "Kate waits.", True, "nor does pulling at them"),
            ("Kate sits with the handcuffs on.\n\nSam checks the handcuffs are "
             "tight.\n\nKate waits.", True, "nor does checking them"),
            ("Kate sits with the handcuffs on.\n\nSam unlocks the handcuffs.\n\n"
             "Kate waits.", False, "unlocking them in the prose does"),
            ("Kate sits with the handcuffs on.\n\nSam unlocks them. They drop to "
             "the floor.\n\nKate waits.", False, "...by pronoun too"),
            ("Kate sits with the handcuffs on.\n\nSam unfastens the handcuffs.\n\n"
             "Kate waits.", False, "...and unfastening them"),
            ("Kate sits with the handcuffs on.\n\nremove: handcuffs\nSam takes them "
             "off.\n\nKate waits.", False, "a remove: line still works")):
        _s = run_node("A room.\n\n" + _beats, plan_only=True, character_memory=hw)[3]
        _last = [x for x in _s.split("---") if x.strip()][-1]
        _on = bool(re.search(r"Every restraint[^.]*\.|The handcuffs stay[^.]*\.", _last))
        check(_lbl, _on == _held, "held=%s want=%s" % (_on, _held))
    # ...and the hardware leaves the SHEET, or the next shot reads it back out.
    _s = run_node("A room.\n\nKate sits with the handcuffs on.\n\n"
                  "Sam unlocks the handcuffs. They drop to the floor.\n\n"
                  "Kate rubs her wrists.", plan_only=True, character_memory=hw)[3]
    check("the hardware leaves the sheet",
          "handcuffs" not in [x for x in _s.split("---") if x.strip()][-1])


def test_one_line_is_one_voice():
    """END TO END: when one person has the line, the other does not babble.

    The guard said only that the other MOUTHS stay closed -- the picture half. On a
    joint model the face follows the audio, so a second voice in the stream moves a
    second mouth whatever the prose says about jaws. The shot now hears how many
    voices there are, not just how many mouths, and the prose is what conditions
    the audio branch.

    And a line with NO name on it got no guard at all: whose mouth to hold was
    unknowable, so both were free. How many voices there are does not require
    knowing whose."""
    print("\n=== one line is one voice ===")
    mem = "Kate: she, 30, blue coat.\nSam: he, 34, black shirt."
    named = run_node('A room.\n\nKate and Sam stand together. Kate says: "Come here."',
                     plan_only=True, character_memory=mem)[3]
    check("the speaker is named", "Only Kate speaks" in named)
    check("...and the other jaws are held",
          "every other mouth in the shot stays closed" in named)
    # SHORT on purpose. Every word in this clause is speech vocabulary, and on a
    # joint model the prose conditions the audio branch: a longer version ("one
    # voice in the shot, the line said once, with room tone either side of it")
    # was added to stop a listener babbling and was reported as causing it.
    check("...and says no more than that",
          "room tone either side" not in named and "said once" not in named)
    # A line is a second or two in a shot of five to ten, and the branch is open for
    # all of it: told only that there is ONE voice, the model still had seconds to
    # fill on either side and invented more talking to fill them. Reported as babble
    # before the dialogue starts.
    # <d>...</d> is the same case.
    tag = run_node("A room.\n\nKate turns to Sam. <d>Come here.</d>",
                   plan_only=True, character_memory=mem)[3]
    check("a <d> line is attributed too", "Only Kate speaks" in tag)
    # No name on the line, two people who could be saying it.
    for _b in ('The two of them wait. "Now?"', "They face each other. <d>Now?</d>"):
        _out = run_node("A room.\n\n" + _b, plan_only=True, character_memory=mem)
        _info, _s = _out[2], _out[3]
        check(f"unattributed still gets one voice: {_b[:26]!r}",
              "Only the person speaking" in _s)
        check("...and says so in info", "names no speaker" in _info)
    # One person alone needs no guard: there is nobody else to babble, and a
    # sentence about other mouths implies other people.
    solo = run_node('A room.\n\nKate waits alone. "Now?"', plan_only=True,
                    character_memory=mem)[3]
    check("one person alone gets no mouth guard",
          "One voice in the shot" not in solo and "Only Kate speaks" not in solo)
    # The speaker is named ONCE. Naming a person twice in one shot is what put a
    # second copy of them in frame.
    check("the speaker is named once in the guard",
          named.split("Only Kate speaks")[1].split(".")[0].count("Kate") == 0)


def test_the_plan_says_whether_silence_can_be_applied():
    """A wrong VAE on audio_vae costs nothing at load time and silently unpins
    every line-free shot -- and the first anybody knows of it is a shot with no
    dialogue that babbles. Probed in the PLAN, so finding out does not cost a full
    render."""
    print("\n=== the plan says whether silence can be applied ===")
    P = "A room.\n\nKate looks around.\n\nKate sits down."
    mem = "Kate: she, 30, coat."
    ok = run_node(P, plan_only=True, character_memory=mem,
                  audio_vae=FakeAudioVAE())[2]
    check("a working audio VAE reports it can", "silence can be applied" in ok)
    check("...and does not cry wolf", "SILENCE CANNOT BE APPLIED" not in ok)
    none = run_node(P, plan_only=True, character_memory=mem, audio_vae=None)[2]
    check("a missing audio VAE is called out", "SILENCE CANNOT BE APPLIED" in none)
    check("...naming the input", "audio_vae input" in none)
    # The likeliest wiring mistake: the VIDEO vae on the audio input. It loads
    # without complaint and reports no sample rate.
    S._SILENT_UNIT["lat"] = None

    class NotAnAudioVae:
        def encode(self, x):
            raise RuntimeError("not an audio vae")

    wrong = run_node(P, plan_only=True, character_memory=mem,
                     audio_vae=NotAnAudioVae())[2]
    check("a VAE that cannot encode silence is called out",
          "SILENCE CANNOT BE APPLIED" in wrong)
    check("...and says which file the input wants",
          "minimax_h3_audio_vae" in wrong)
    S._SILENT_UNIT["lat"] = None


def test_the_silent_latent_looks_like_silence():
    """The conditioning has to look like what the encoder produces, not just decode
    quietly. The old build kept ONE interior frame and repeated it, on the argument
    that silence is homogeneous -- it is not, in latent space. Encoded silence has
    frame-to-frame variation (mean 0.002-0.004, max 0.021); a repeated frame has a
    delta of exactly zero, which is a signal no encoder makes, and a model handed
    conditioning outside its own distribution has every reason to disregard it.

    Checked with a stand-in whose encode has the same shape and the same edge
    artifacts as the real VAE, so the shape of the fix is tested without loading a
    checkpoint. The numbers in the docstring were measured against the real one.

    This lives in the smoke suite because test_node stubs torch, and a stand-in
    built from stub tensors cannot exercise a function that slices and flips."""
    print("\n=== the silent latent looks like silence ===")

    class EdgyVae:
        """Encodes to [B, 32, 2, T] with big deltas at the ends, like the real one."""
        audio_sample_rate = 32000

        def encode(self, x):
            n = x.shape[1] // 800
            t = torch.arange(n, dtype=torch.float32)
            base = 0.46 + 0.002 * torch.sin(t * 0.7)      # interior variation
            base[:4] += torch.tensor([0.22, 0.10, 0.05, 0.03])   # padded head
            base[-4:] += torch.tensor([0.03, 0.05, 0.10, 0.17])  # padded tail
            return base.reshape(1, 1, 1, n).repeat(1, 32, 2, 1)

    vae = EdgyVae()
    S._SILENT_UNIT["lat"] = None
    lat = S._silent_audio_latent(vae, 226, S.H3_FPS)
    check("a latent is produced", lat is not None)
    _, _, want_t = S.temporal_shape(226, S.H3_FPS)
    check("...of exactly the length the layout wants", lat.shape[-1] == want_t)
    check("...with the layout's channel count", lat.shape[1] == 32)
    d = (lat[..., 1:] - lat[..., :-1]).abs()
    check("it is NOT a flat signal", float(d.mean()) > 1e-5)
    # The edges the encoder pads are thrown away, so no join carries their spike.
    check("no seam spike from the padded ends", float(d.max()) < 0.05)
    # Ping-pong: every join repeats a frame, so plain tiling's seam is gone.
    S._SILENT_UNIT["lat"] = None
    long = S._silent_audio_latent(vae, 462, S.H3_FPS)
    dl = (long[..., 1:] - long[..., :-1]).abs()
    check("a longer shot has no seam either", float(dl.max()) < 0.05)
    check("...and is still the right length",
          long.shape[-1] == S.temporal_shape(462, S.H3_FPS)[2])
    # Still defensive: anything unexpected returns None rather than raising.
    S._SILENT_UNIT["lat"] = None

    class Broken:
        audio_sample_rate = 32000

        def encode(self, x):
            raise RuntimeError("nope")

    check("a failing encode returns None",
          S._silent_audio_latent(Broken(), 226, S.H3_FPS) is None)
    check("no sample rate returns None",
          S._silent_audio_latent(object(), 226, S.H3_FPS) is None)

    class TooShort:
        audio_sample_rate = 32000

        def encode(self, x):
            return torch.zeros((1, 32, 2, 3))

    S._SILENT_UNIT["lat"] = None
    check("an encode too short to trim returns None",
          S._silent_audio_latent(TooShort(), 226, S.H3_FPS) is None)
    S._SILENT_UNIT["lat"] = None


def test_a_softened_handoff_is_not_a_keyframe():
    """The removing shot keeps describing the garment when nothing anchors it.

    The scrub applies to the removing shot too, and the only reason that is safe is
    that the shot's KEYFRAME already shows the garment on at the start. Below
    KEYFRAME_SAFE_AUG the handoff stops being a keyframe and rides as an extra
    reference -- it says who somebody is, not what the opening frame holds -- so
    the premise is false and scrubbing took the belt out of the text of the very
    shot that removes it. It was gone a beat early with nothing holding it on."""
    print("\n=== a softened handoff is not a keyframe ===")
    # No outer garment over the belt: implied_layers would keep it out of the sheet
    # until that garment came off, which is a different rule and would mask what
    # this test is about.
    P = ("A room.\n\nMcKenna: she, 22, crop top, chastity belt.\n\n"
         "Dan: he, 30, shirt.\n\nMcKenna walks in.\n\nDan looks at her.\n\n"
         "Dan unlocks the chastity belt.\n\nMcKenna sits down.")
    soft = [x for x in re.split(r"(?=\[Shot )",
                                run_node(P, plan_only=True, ref_noise_aug=0.95)[3])
            if x.strip()]
    check("softened: the removing shot still says it is worn",
          "chastity belt" in soft[2].split("Dan unlocks")[0])
    check("...and the next shot does not", "chastity belt" not in soft[3])
    # At an anchoring aug the keyframe does hold the first frame, and scrubbing the
    # removing shot is right: text saying it is worn would put it back at the end.
    hard = [x for x in re.split(r"(?=\[Shot )",
                                run_node(P, plan_only=True, ref_noise_aug=0.999)[3])
            if x.strip()]
    check("anchored: the removing shot is scrubbed",
          "chastity belt" not in hard[2].split("Dan unlocks")[0])
    check("...and it still comes off there", "comes off during this shot" in hard[2])
    # The report has to name WHICH reason, or it cannot be acted on.
    info = run_node(P, plan_only=True, ref_noise_aug=0.95)[2]
    check("info names the aug", "ref_noise_aug 0.95 is below" in info)


def test_a_journey_reaches_the_shot():
    """END TO END: the travel clause lands in the shot, with the room the previous
    beat established as its origin.

    The unit tests exercise travel_anchor directly and passed with the clause
    disconnected from the render loop, which is the failure that matters: a
    journey given only its destination is one the model can satisfy by starting
    there."""
    print("\n=== a journey reaches the shot ===")
    mem = "McKenna: she, 22, top.\nDan: he, 30, shirt."
    P = ("A scene in a home.\n\n"
         "McKenna finds Dan in the living room.\n\n"
         "She takes his hand and walks him down the hallway to the bedroom.\n\n"
         "They sit on the bed.")
    out = run_node(P, plan_only=True, character_memory=mem)
    info, script = out[2], out[3]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("the travelling shot is told where it begins",
          "begins in the living room" in sh[1])
    check("...and where it ends", "ends in the bedroom" in sh[1])
    check("...and what it passes through", "along the hallway" in sh[1])
    check("...and that it is not a cut", "not a cut" in sh[1])
    # The origin came from the PREVIOUS beat: this one never names it.
    check("the origin is latched from the earlier beat",
          "living room" not in P.split("\n\n")[2])
    # A shot that goes nowhere gets nothing.
    check("a stationary shot carries no travel clause",
          "The shot begins in" not in sh[0] and "The shot begins in" not in sh[2])
    check("info names the travelling shots", "move between places" in info)


def test_a_line_is_spoken_in_one_language():
    """H3 is joint and multilingual. The prose conditions the audio branch, and a
    branch told a line is spoken but never told in WHAT will pick a language --
    fluent delivery in one nobody asked for sounds like babble to anybody expecting
    English."""
    print("\n=== a line is spoken in one language ===")
    mem = "Kate: she, 30, coat.\nSam: he, 34, shirt."
    P = ('A room.\n\nKate waits.\n\nKate says to Sam: "Come here."\n\n'
         "They sit down.")
    out = run_node(P, plan_only=True, character_memory=mem)
    info, script = out[2], out[3]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("the speaking shot is told its language",
          "spoken in English" in sh[1])
    check("...and the silent ones are not",
          "spoken in" not in sh[0] and "spoken in" not in sh[2])
    check("info names the shots", "told it is spoken in English" in info)
    # Positively phrased: naming the unwanted language would ask for it.
    check("no language is named but the wanted one",
          not any(w in sh[1] for w in ("not in", "Spanish", "French", "Chinese")))
    # Non-Latin characters are a strong language signal and easy to miss by eye.
    check("plain English is clean", S.non_latin_in('Kate says: "Hello."') == [])
    check("...and accented Latin is not flagged",
          S.non_latin_in("caf\u00e9 na\u00efve") == []
          and S.non_latin_in("cafe\u0301") == [])
    check("...nor curly quotes and dashes",
          S.non_latin_in("\u201cHello\u201d \u2014 she said\u2026") == [])
    check("a non-Latin script IS flagged",
          S.non_latin_in("\u041f\u0440\u0438\u0432\u0435\u0442") != [])
    odd = run_node('A room.\n\nKate says: "\u041f\u0440\u0438\u0432\u0435\u0442."',
                   plan_only=True, character_memory="Kate: she, 30, coat.")[2]
    check("...and reported", "not Latin text" in odd)
    check("...without being removed from the prompt", "NOT removed" in odd)


def test_undressing_does_not_spread():
    """END TO END: one character undressing leaves the other dressed."""
    print("\n=== undressing does not spread ===")
    mem = "McKenna: she, 22, crop top, shorts.\nDan: he, 30, shirt, jeans."
    P = ("A room.\n\nMcKenna and Dan stand.\n\n"
         "McKenna and Dan sit down. McKenna takes off her clothes.\n\nThey wait.")
    sh = [x for x in re.split(r"(?=\[Shot )",
                              run_node(P, plan_only=True, character_memory=mem)[3])
          if x.strip()]
    check("the other character keeps his wardrobe", "shirt, jeans" in sh[1])
    check("...and hers is gone", "crop top" not in sh[1].split("Dan:")[0])
    check("the clause says whose body it is",
          "Everything McKenna is wearing" in sh[1])
    check("...and pins everyone else to their own entry",
          "own entry lists" in sh[1])
    check("...once, not twice", sh[1].count("own entry lists") == 1)
    # It stays gone for him on later shots too.
    check("he is still dressed afterwards", "shirt, jeans" in sh[2])


def test_a_working_character_is_not_still_lying_down():
    """END TO END: the hold lets go when the beat contradicts it."""
    print("\n=== a working character is not still lying down ===")
    mem = "McKenna: she, 2, blonde.\nDana: she, 35, brunette, shirt, jeans."
    P = ("A nursery.\n\nDana lies down on the sofa with McKenna.\n\n"
         "Dana takes out a new blanket and places it on the change table.\n\n"
         "Dana waits.")
    sh = [x for x in re.split(r"(?=\[Shot )",
                              run_node(P, plan_only=True, character_memory=mem)[3])
          if x.strip()]
    check("the working shot does not say she is lying down",
          "Dana is still lying down" not in sh[1])
    check("...nor the shot after it", "Dana is still lying down" not in sh[2])
    # And a pose that is NOT contradicted is still held.
    P2 = ("A nursery.\n\nDana and McKenna sit down on the sofa.\n\n"
          "Dana and McKenna look at the window.\n\nDana and McKenna wait.")
    sh2 = [x for x in re.split(r"(?=\[Shot )",
                               run_node(P2, plan_only=True, character_memory=mem)[3])
           if x.strip()]
    check("a pose nothing contradicts is still held",
          "still sitting" in sh2[1])


def test_pacing_reaches_the_thin_shots():
    """END TO END: the pacing clause lands on shots that outlast their beat, and
    on no others."""
    print("\n=== pacing reaches the thin shots ===")
    mem = "Kate: she, 30, coat."
    P = ("A room.\n\nKate waits.\n\n"
         "Kate walks in, drops her bag, takes off her coat, hangs it up and "
         "crosses the room.\n\nKate looks at the window.")
    # 'fixed' gives every shot the same length regardless of content, which is
    # where a three second action lands in a ten second shot.
    out = run_node(P, plan_only=True, character_memory=mem,
                   shot_length="fixed", shot_seconds=10.0)
    info, script = out[2], out[3]
    paced = [i for i, b in enumerate(script.split("---"), 1)
             if "even pace across the whole shot" in b]
    check("the thin shots are paced", paced == [1, 3], str(paced))
    check("...and the full one is not", 2 not in paced)
    check("info names them", "stage less than their length" in info)
    # Sizing from the beat already matches content, so nothing is thin and the
    # clause stays out of the way.
    fitted = run_node(P, plan_only=True, character_memory=mem)[3]
    check("sizing from the beat needs no pacing",
          "even pace across the whole shot" not in fitted)


def test_an_instruction_is_not_the_action():
    """END TO END: a beat that TELLS somebody to undress and lie down does neither
    until the beat that does it."""
    print("\n=== an instruction is not the action ===")
    mem = "McKenna: she, 22, shorts.\nDana: she, 35, jeans."
    P = ("A public bathroom.\n\n"
         "McKenna and Dana walk in. Dana says to McKenna: "
         '"Take off your shorts and lie down on the change table." '
         "Dana puts the bag on the change table.\n\n"
         "McKenna takes off her shorts and lies down on the change table.\n\n"
         # She has to be IN the last shot for her pose to be stated: a pose for
         # somebody the text does not describe is a pose for nobody.
         "Dana opens the bag and looks at McKenna.")
    sh = [x for x in re.split(r"(?=\[Shot )",
                              run_node(P, plan_only=True, character_memory=mem)[3])
          if x.strip()]
    check("the shorts are still on while she is only told",
          "shorts" in sh[0].split("Dana says")[0])
    check("...and nothing is taken off yet",
          "off during this shot" not in sh[0])
    check("...and nobody is lying down yet", "still lying down" not in sh[0])
    # The beat that actually does it.
    check("the next beat takes them off", "off during this shot" in sh[1])
    # ...and the pose it sets is held afterwards, for her only.
    check("the pose is held after that", "McKenna is still lying down" in sh[2])
    check("...and the speaker is not lying down", "Dana is still lying" not in sh[2])
    # ...and with NOTHING happening after the instruction, so a wrongly latched
    # pose has nothing to clear it. Without the guard both of them are laid down
    # by a sentence Dana only spoke.
    idle = ("A public bathroom.\n\nMcKenna and Dana walk in. Dana says to McKenna: "
            '"Take off your shorts and lie down on the change table."\n\n'
            "McKenna and Dana wait.")
    idle_sh = [x for x in re.split(r"(?=\[Shot )",
                                   run_node(idle, plan_only=True,
                                            character_memory=mem)[3]) if x.strip()]
    # The words of the order are still in the shot -- they have to be, beats go to
    # the model verbatim -- so the LISTENER is given something to be doing, or the
    # instruction is the only thing in the frame about their body.
    check("the listener is given something to do",
          "McKenna listens" in sh[0])
    # Split defensively: with the clause absent this used to raise IndexError
    # rather than report a failure, which hides the result behind a traceback.
    _tail = sh[0].split("McKenna listens")
    check("...and is named once, not twice",
          len(_tail) > 1 and _tail[1].split(".")[0].count("McKenna") == 0)
    check("...and a line with no order in it adds nothing",
          "listens, still" not in run_node(
              'A room.\n\nDana says: "Hello there."', plan_only=True,
              character_memory=mem)[3])
    check("a spoken instruction latches nobody",
          "still lying down" not in idle_sh[1], idle_sh[1][-90:])


def test_the_scene_does_not_reset():
    """END TO END: after a move, later shots are in the new room, and the journey
    itself has both its ends."""
    print("\n=== the scene does not reset ===")
    mem = "Kate: she, 30, coat.\nSam: he, 34, shirt."
    P = ("A living room.\n\nKate and Sam sit on the sofa.\n\n"
         "Kate walks him down the hallway to the bedroom.\n\n"
         "They sit on the bed.\n\nKate looks at him.")
    out = run_node(P, plan_only=True, character_memory=mem)
    info, script = out[2], out[3]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    # The journey has an origin, which it only has because `here` is seeded from
    # the scene: stated as a destination alone it renders as a cut.
    check("the move begins where the scene says", "begins in the living room" in sh[1])
    check("...and ends in the new room", "ends in the bedroom" in sh[1])
    # ...and the shots after it stay there.
    check("the next shot is in the new room", "is in the bedroom" in sh[2])
    check("...and so is the one after", "is in the bedroom" in sh[3])
    check("info names them", "room the scene text does not name" in info)
    # A script that never moves is untouched.
    still = run_node("A living room.\n\nKate waits.\n\nKate looks up.",
                     plan_only=True, character_memory=mem)[3]
    check("a script that stays put says nothing", "This shot is in the" not in still)


def test_a_described_room_still_holds():
    """END TO END: the same journey as above, into a room the author DESCRIBED. One
    adjective used to switch the whole room tracker off, silently, and the acoustic
    stayed behind in the room they left."""
    print("\n=== a described room still holds ===")
    mem = "Kate: she, 30, coat.\nSam: he, 34, shirt."
    # The shots that check the acoustic carry a LINE, because the bed and the room
    # tone go under shots whose audio branch is already open. A wordless shot is
    # pinned to silence and is given no acoustic at all -- see
    # test_room_tone_under_every_shot for why ambience everywhere does not work.
    P = ("A carpeted living room, lamps low.\n\n"
         "Kate and Sam sit on the sofa and she says: \"Come with me.\"\n\n"
         "Kate walks him down the hallway to the tiled bathroom.\n\n"
         "They stand by the sink and she says: \"Wait here.\"\n\n"
         "Kate turns the tap on.")
    out = run_node(P, plan_only=True, character_memory=mem)
    info, script = out[2], out[3]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("the journey has both ends", "begins in the living room" in sh[1]
          and "ends in the bathroom" in sh[1])
    check("the next shot is in the new room", "is in the bathroom" in sh[2])
    check("...and so is the one after", "is in the bathroom" in sh[3])
    # The acoustic goes with them.
    check("the room tone follows", "tiled walls ringing" in sh[2])
    check("...and the ambient bed", "water moving in the pipes" in sh[2])
    check("the room they left keeps its own", "a soft room with little echo" in sh[0])
    check("...and is not given the new one", "tiled walls ringing" not in sh[0])
    check("info names the shots", "sound followed them into the new room" in info)
    # auto_sound off leaves the picture fix alone and takes only the sound.
    quiet = run_node(P, plan_only=True, character_memory=mem, auto_sound=False)[3]
    check("auto_sound off keeps the room hold", "is in the bathroom" in quiet)
    check("...and drops the acoustic", "tiled walls ringing" not in quiet)


def test_the_sound_clause_is_inside_the_budget():
    """END TO END, and testing the WIRING rather than the function.

    The unit test for this passed with the fix reverted, because it called
    fit_guards directly with clauses of its own -- which says nothing about whether
    the sound clause is actually in the list the node passes. It was appended after
    the call for a long time, so it was the one piece of node-written text no cap
    could reach.

    The shipped budget almost never binds, so this squeezes it deliberately and
    checks the node's own report names what went."""
    print("\n=== the sound clause is inside the budget ===")
    mem = "Kate: she, 30, coat, scarf.\nSam: he, 34, shirt."
    # Beats carry LINES, because a sound clause only exists on a shot whose audio
    # branch is open -- a wordless shot is pinned to silence and gets none, so there
    # would be nothing for the budget to drop.
    P = ("A living room.\n\nKate and Sam sit on the sofa and she says: \"Sit down.\"\n\n"
         "Kate takes off her scarf and says: \"It is warm in here.\"\n\n"
         "Kate walks him down the hallway to the tiled bathroom.\n\n"
         "They stand by the sink and she says: \"Wait here.\"\n\n"
         "Sam waits and says: \"All right.\"")
    floor, ratio = S.GUARD_FLOOR_WORDS, S.GUARD_WORDS_PER_BEAT_WORD
    try:
        S.GUARD_FLOOR_WORDS, S.GUARD_WORDS_PER_BEAT_WORD = 20, 1
        info = run_node(P, plan_only=True, character_memory=mem)[2]
    finally:
        S.GUARD_FLOOR_WORDS, S.GUARD_WORDS_PER_BEAT_WORD = floor, ratio
    check("a squeezed budget drops something", "guard clauses dropped for room" in info)
    # ...and the thing it drops is the sound clause, because it ranks last. If it
    # were still appended after the cap it could not be dropped at all.
    check("the sound clause is what gives way", re.search(r"dropped for room[^|]*sound",
                                                          info) is not None)
    # At the SHIPPED budget nothing is dropped: the cap catches runaways and does
    # not trim routinely, so no current render changes because of this.
    clean = run_node(P, plan_only=True, character_memory=mem)[2]
    check("the shipped budget drops nothing", "guard clauses dropped for room" not in clean)


def test_the_anchor_is_not_read_as_a_room():
    """END TO END: an anchor is the CAMERA. room_tone has always said so and reads the
    opening beat instead, but the room reader was handed the whole assembled string --
    anchor, scene and sheet -- so a lens line was being asked where everybody is.

    With "shallow depth of field" in the anchor, every shot was told it was in a hall,
    the first journey took the hall as its origin, and the acoustic became a large
    room with a long tail. None of it was in the script."""
    print("\n=== the anchor is not read as a room ===")
    anchor = ("Shot on 35mm, shallow depth of field, warm practical lighting, "
              "handheld, muted colour grade, film grain.")
    mem = "Kate: she, 30, coat.\nSam: he, 34, shirt."
    P = ("Kate and Sam sit on the sofa.\n\n"
         "Kate walks him down the hallway to the bedroom.\n\n"
         "They sit on the bed.\n\nSam waits.")
    out = run_node(P, plan_only=True, character_memory=mem, anchor=anchor)
    script = out[3]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("no shot is put in a phantom hall", "in the hall," not in script)
    check("...and no journey starts in one", "begins in the hall" not in script)
    check("...and the lens gets no cathedral", "a large room with a long tail" not in script)
    # The anchor itself is still carried on every shot -- that is what it is for.
    check("the anchor is on every shot",
          all("Shot on 35mm" in x for x in sh), f"{len(sh)} shots")
    check("...once per shot", all(x.count("Shot on 35mm") == 1 for x in sh))
    # The journey itself still works; it simply has no invented origin.
    check("the move still reaches the bedroom", "ends in the bedroom" in sh[1])
    # ...and a scene paragraph that DOES name a room still seeds it.
    named = run_node("A living room.\n\nKate waits.\n\n"
                     "Kate walks him down the hallway to the bedroom.",
                     plan_only=True, character_memory=mem)[3]
    check("a real scene room still seeds the origin", "begins in the living room" in named)
    # THE HALF THAT CAUGHT A WRONG FIX. Excluding the anchor from the room reader
    # looks right -- a lens line is not a location -- but the anchor is DOCUMENTED
    # to carry the location, and with one set there is no scene paragraph for the
    # room to live in instead. Excluded, this loses the origin of its first journey,
    # which is the destination-with-no-origin case that renders as a cut. The word
    # boundary was the whole fix; the anchor was never at fault.
    located = run_node(P, plan_only=True, character_memory=mem,
                       anchor="A carpeted living room. Shot on 35mm, "
                              "shallow depth of field, handheld.")[3]
    check("a location IN the anchor still seeds the origin",
          "begins in the living room" in located)
    check("...and still no phantom hall", "in the hall" not in located)


def test_a_garment_does_not_appear_at_the_first_frame():
    """END TO END: an outer garment comes off, several beats pass, and it goes back
    on. The shot that puts it on must describe the CHANGE, not the end state, and
    must not also list it as already worn -- one shot holding the garment in two
    states is the disagreement that made it appear at frame one."""
    print("\n=== a garment does not appear at the first frame ===")
    mem = "Kate: she, 30, brown hair, grey leggings, blue shorts."
    P = ("A changing room.\n\nKate stands by the bench.\n\n"
         "Kate takes off her shorts.\n\nKate sits down on the bench.\n\n"
         "Kate reaches for the bench.\n\n"
         "Kate puts her shorts back on.\nadd: her blue shorts\n\n"
         "Kate stands up and walks out.")
    out = run_node(P, plan_only=True, character_memory=mem)
    info, script = out[2], out[3]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    # Off in the shots between.
    check("the shorts are gone after the removal", "shorts" not in sh[2])
    check("...and stay gone", "shorts" not in sh[3])
    # The staging shot describes the CHANGE...
    check("the staging shot gives both ends",
          "off the body as the shot opens" in sh[4] and "by the last frame" in sh[4])
    # ...and does NOT also list them as already worn. The add phrase is held back.
    check("...and does not also list them as worn", "Her blue shorts." not in sh[4])
    # From the next shot they are simply worn, the way a removal scrubs from its own.
    check("the next shot wears them", "Her blue shorts." in sh[5])
    check("info names the shot", "put a garment back ON" in info)
    # The OTHER use of add: -- revealing a layer already underneath -- is untouched.
    rev = run_node("A yard.\n\nNora stands by the gate.\n\n"
                   "Dan cuts off her jacket and throws it away.\n"
                   "remove: jacket\nadd: her white shirt underneath\n\n"
                   "Nora turns to the gate.",
                   plan_only=True,
                   character_memory="Nora: she, 34, red hair, green jacket, white shirt.")[3]
    check("a reveal stages no dressing", "off the body as the shot opens" not in rev)
    check("...and is described in its own shot", "Her white shirt underneath." in
          [x for x in re.split(r"(?=\[Shot )", rev) if x.strip()][1])
    # ...and the case the beat_stages_wearing guard actually protects: the SAME
    # garment coming back, on a beat that does not stage anybody putting it on. It
    # went back on between shots, so there is no dressing to give both ends to, and
    # the shot should simply describe it as worn.
    off = run_node("A changing room.\n\nKate stands by the bench.\n\n"
                   "Kate takes off her shorts.\n\nKate walks to the door.\n\n"
                   "Kate is at the door.\nadd: her blue shorts\n\n"
                   "Kate opens the door.",
                   plan_only=True, character_memory=mem)[3]
    osh = [x for x in re.split(r"(?=\[Shot )", off) if x.strip()]
    check("a garment back with no dressing staged gets no clause",
          "off the body as the shot opens" not in off)
    check("...and is described as worn in that shot", "Her blue shorts." in osh[3])


def test_the_ambient_bed_reaches_the_soundtrack():
    """END TO END, and testing the WIRING. The unit tests call mix_ambient directly,
    which says nothing about whether the node ever calls it -- the same gap that let
    the sound-clause budget test pass with its fix reverted.

    It goes on ONCE, over the joined soundtrack, so the loop runs continuously
    through the cuts instead of restarting at every one."""
    print("\n=== the ambient bed reaches the soundtrack ===")
    import math
    P = ("A room.\n\nHe walks in.\n\nShe follows him and says: \"Wait.\"")
    # THE FAKE MODEL IS RANDOM: two identical runs differ by 0.69 peak, so nothing
    # here may compare one run's waveform against another's. An earlier version of
    # this test did, and "the bed is not the dry one" passed on that alone. What is
    # sound to assert is the node's own account -- the note comes from mix_ambient's
    # return value and from nowhere else, so it is present exactly when the call is.
    n = int(run_node(P)[1]["waveform"].shape[-1])
    sr = 44100
    tt = torch.arange(sr, dtype=torch.float32) / sr          # 1s, non-integer cycles
    bed = {"waveform": (torch.sin(2 * math.pi * 97.3 * tt) * 0.4
                        ).unsqueeze(0).repeat(2, 1).unsqueeze(0),
           "sample_rate": sr}
    wet_out = run_node(P, ambient_audio=bed, ambient_level=0.3)
    wet, info = wet_out[1]["waveform"], wet_out[2]
    check("the soundtrack keeps its length", int(wet.shape[-1]) == n)
    check("the node actually mixes the bed", "ambient bed was laid under" in info)
    check("...over the whole soundtrack", "under the whole soundtrack" in info)
    check("...and loops the short one", "looped with a crossfade" in info)
    # Level 0 is a wired bed that costs nothing until it is asked for.
    off = run_node(P, ambient_audio=bed, ambient_level=0.0)
    check("level 0 mixes nothing", "ambient bed was laid under" not in off[2])
    check("...and still renders", int(off[1]["waveform"].shape[-1]) > 0)
    # WITH NOTHING WIRED, the bed is BUILT from the scene -- no file, no second
    # model pass. That is the default, so it is the path that has to work.
    built = run_node("A tiled bathroom.\n\nHe walks in.\n\n"
                     "She follows him and says: \"Wait.\"", ambient_level=0.25)
    check("a bed is built with nothing wired", "built from the scene" in built[2])
    check("...naming what it read", "tiled walls ringing" in built[2])
    check("...and it is mixed", "ambient bed was laid under" in built[2])
    # A wired file OVERRIDES the built one rather than stacking with it.
    check("a wired bed is used instead of a built one",
          "built from the scene" not in info)
    # WHEN THE SHAPED BED WILL NOT BUILD, the plain one goes on instead -- never
    # nothing, and never "go and wire a file". Forced, because it should not happen
    # on its own.
    _real = S.synth_ambient
    try:
        S.synth_ambient = lambda *a, **k: None
        fb = run_node("A tiled bathroom.\n\nHe walks in.", ambient_level=0.25)[2]
    finally:
        S.synth_ambient = _real
    check("a failed shape still gets a bed", "ambient bed was laid under" in fb)
    check("...and says it fell back", "plain fallback" in fb)
    check("...without telling anyone to wire a file", "Wire a recording there" not in fb)
    # A description of EVENTS gets the room, and says so rather than disappointing.
    ev = run_node("A garden in the morning.\n\nHe walks in.", ambient_level=0.25)[2]
    check("an eventful description is honest about tone",
          "names EVENTS" in ev, ev[-200:] if "built from" in ev else "no bed built")
    # ...and the silent conditioning is untouched by any of it: the bed is a MIX and
    # must not re-open an audio branch the way an inferred bed did.
    check("wordless shots are still pinned",
          "have no quoted line and no sound described" in info)


def test_built_sound_lands_in_the_right_shot():
    """END TO END, and testing the PLACEMENT rather than the note. The note names the
    shot from the same loop that does the mixing, so it would read correctly even if
    the span arithmetic were wrong. foley_for is replaced with a flat marker, which
    shows up as a DC offset in exactly the samples it was written to."""
    print("\n=== built sound lands in the right shot ===")
    mem = "Ana: she, 28, grey coat.\nGuard: he, 40, uniform."
    P = ("A concrete room.\n\nAna stands by the table.\n\n"
         "The guard closes the cuffs around her wrists.\n\n"
         # The speaking shot STAGES A SOUND too, so leaving it alone is a real
         # assertion: its branch is open and already making that sound from the
         # same prose, and building over it would double every rattle.
         "The guard drags the chain and says: \"Sit down.\"")
    _real = S.foley_for
    try:
        S.foley_for = lambda phrase, n, sr, seed=0: torch.ones(int(n))
        out = run_node(P, character_memory=mem, ambient_level=0.0, foley_level=0.5)
    finally:
        S.foley_for = _real
    wav, info, per_shot = out[1]["waveform"], out[2], out[4]
    check("info names the shot", "sound built into the shot itself" in info)
    # The fake model is random with mean ~0, so a span carrying the marker has a
    # mean near +0.5 and every other span sits near 0.
    n_shots = out[6]
    span = int(wav.shape[-1]) // max(n_shots, 1)
    means = [float(wav[..., i * span:(i + 1) * span].mean()) for i in range(n_shots)]
    hot = [i + 1 for i, m in enumerate(means) if m > 0.25]
    check("shot 2 carries the built sound", 2 in hot, f"means {[round(m,2) for m in means]}")
    check("the speaking shot does not", 3 not in hot,
          f"means {[round(m,2) for m in means]}")
    check("the shot with nothing to sound does not", 1 not in hot,
          f"means {[round(m,2) for m in means]}")
    # The switches that turn it off.
    off = run_node(P, character_memory=mem, ambient_level=0.0, foley_level=0.0)[2]
    check("foley_level 0 builds nothing", "sound built into the shot" not in off)
    na = run_node(P, character_memory=mem, ambient_level=0.0, foley_level=0.5,
                  auto_sound=False)[2]
    check("auto_sound off builds nothing", "sound built into the shot" not in na)


def test_one_picture_two_people_is_reported():
    """Reported: a scene written for two people rendered the same woman twice, and
    the second character was not recognised as a separate person.

    A shot carrying a reference for one person and describing another who has none
    gives the model a photographed face and two faces to draw. A reference is the
    strongest identity signal in the prompt -- far stronger than "38, dark hair" --
    so the face that exists gets used twice.

    The node cannot fix it: nothing in the text outranks a photograph, and it is the
    model resolving a shot with more subjects than pictures. What it must do is SAY
    so, in terms of the fix, instead of leaving it to be discovered in a render."""
    print("\n=== one picture, two people ===")
    img = torch.rand(1, H, W, 3)
    P = ("A hotel room.\n\nKristy and Dan sit on the bed.\n\n"
         "Dan takes her hand.")
    one = run_node(P, plan_only=True, ref_image_1=img,
                   character_memory="Kristy: <Picture 1>, she, 26, blonde.\n"
                                    "Dan: he, 38, dark hair.")[2]
    check("the mismatch is reported",
          "carry a reference picture for one person" in one)
    check("...naming who has no picture", "describe Dan" in one)
    check("...and naming the shots", "shot(s) 1, 2" in one)
    check("...and saying what to do about it", "<Picture 2>" in one)
    # It must not cry wolf. Both tagged is the state it is asking for.
    both = run_node(P, plan_only=True, ref_image_1=img, ref_image_2=img,
                    character_memory="Kristy: <Picture 1>, she, 26, blonde.\n"
                                     "Dan: <Picture 2>, he, 38, dark hair.")[2]
    check("both tagged says nothing", "carry a reference picture for one" not in both)
    # ...nor when there is no picture in play at all, which is a different note.
    none = run_node(P, plan_only=True,
                    character_memory="Kristy: <Picture 1>, she, 26, blonde.\n"
                                     "Dan: he, 38, dark hair.")[2]
    check("no reference wired says nothing", "carry a reference picture for one" not in none)
    # ...nor on a scene with one person in it.
    solo = run_node("A room.\n\nKristy sits down.\n\nKristy stands up.",
                    plan_only=True, ref_image_1=img,
                    character_memory="Kristy: <Picture 1>, she, 26, blonde.")[2]
    check("a solo scene says nothing", "carry a reference picture for one" not in solo)


def test_built_sound_reaches_an_effort_shot():
    """Reported as hearing nothing, and it was self-inflicted.

    The foley mix ran only on shots PINNED TO SILENCE -- right, while the only way a
    branch opened was the author writing a sound, because then the branch is already
    making that sound out of the same prose and building over it doubles every
    footfall. Then the nine effort verbs were made _voiced. _voiced unpins the shot.
    So the same commit that added "a bed frame working" and "restraints pulling
    taut" stopped either of them ever being mixed, on precisely the beats they were
    written for: read from the text, put in the prompt, dropped from the audio.

    An effort branch opened to make a VOICE, and a voice is not a frame or a chain.
    Tested end to end on the SPAN, not the note -- the note is written by the loop
    that does the mixing and would read correctly even if the arithmetic were wrong."""
    print("\n=== built sound reaches an effort shot ===")
    P = ("A cell.\n\nShe sits on the bunk.\n\n"
         "She strains against the cuffs.\n\n"
         "They rock together on the bed.")
    _real = S.foley_for
    try:
        S.foley_for = lambda phrase, n, sr, seed=0: torch.ones(int(n))
        out = run_node(P, ambient_level=0.0, foley_level=0.5)
    finally:
        S.foley_for = _real
    wav, info, n_shots = out[1]["waveform"], out[2], out[6]
    span = int(wav.shape[-1]) // max(n_shots, 1)
    means = [float(wav[..., i * span:(i + 1) * span].mean()) for i in range(n_shots)]
    hot = [i + 1 for i, m in enumerate(means) if m > 0.25]
    shown = f"means {[round(m, 2) for m in means]}"
    check("the restrained effort shot sounds", 2 in hot, shown)
    check("the furniture effort shot sounds", 3 in hot, shown)
    check("a shot with nothing to sound stays quiet", 1 not in hot, shown)
    check("the note says why those two are different",
          "stage effort, so their branch IS open" in info)
    # The VOCAL phrase on those shots must not be built -- it has no recipe, and
    # that is the whole reason a noise shaper may run on an open branch at all.
    # Checked on the REAL builder: the stub above answers every phrase, including
    # the vocal one, so asserting this against the stubbed run tests the stub.
    check("the vocal phrase has no recipe and cannot be built",
          S.foley_for("unsteady breathing, with gasps and moans of effort",
                      44100, 44100, seed=0) is None)
    check("...and no recipe in the table is a voice",
          not [k for k in S._FOLEY if re.search(
              r"breath|gasp|moan|voice|speak|whimper|cry|sob|scream|pant", k, re.I)])
    real = run_node(P, ambient_level=0.0, foley_level=0.5)[2]
    # Guarded: with the fix reverted there is no note at all, and an IndexError
    # here reports as a crashed suite rather than as the failure it is.
    built = (real.split("sound built into the shot")[1].split(". Shot(s)")[0]
             if "sound built into the shot" in real else "(no foley note)")
    check("the real note builds the frame", "a bed frame working" in built)
    check("...and not the breathing beside it", "unsteady breathing" not in built)
    # An author-written sound still opens the branch and still suppresses the mix:
    # that shot's audio is already making it.
    W = ("A cell.\n\nThe chain drags on the concrete, and she says: \"Wait.\"")
    try:
        S.foley_for = lambda phrase, n, sr, seed=0: torch.ones(int(n))
        out2 = run_node(W, ambient_level=0.0, foley_level=0.5)
    finally:
        S.foley_for = _real
    w2, n2 = out2[1]["waveform"], out2[6]
    s2 = int(w2.shape[-1]) // max(n2, 1)
    m2 = [float(w2[..., i * s2:(i + 1) * s2].mean()) for i in range(n2)]
    check("a written sound still suppresses the mix", all(m < 0.25 for m in m2),
          f"means {[round(m, 2) for m in m2]}")
    # Silencing OFF says "pin nothing, let the model sound every shot" -- and then
    # there is no shot the model cannot make, which is the only reason anything is
    # built here. The effort shots kept their built layer while the model was also
    # sounding them from the same prose: the exact doubling the gate exists to stop.
    off = run_node(P, ambient_level=0.0, foley_level=0.5, silence_nonspeech=False)[2]
    check("nothing is built when the model sounds everything",
          "sound built into the shot" not in off)


def test_a_collar_chained_to_a_wall_stays_on():
    """Reported: her collar was chained to a wall and she took it off.

    END TO END, because the unit halves both passed while the film was still wrong.
    limb_anchor could read a wall and held_part could say "neck", and neither was
    reached: the wall was missing from _ANCHOR_POINT so nothing anchored, and the
    part was computed from an item name that is dropped whenever the beat already
    says it. Testing the pieces is what let this ship -- so this asserts the SHOTS."""
    print("\n=== a collar chained to a wall stays on ===")
    P = ("A basement.\n\n"
         "The guard locks a steel collar around Ana's neck and chains it to the "
         "wall.\n\n"
         "Ana sits on the floor.\n\n"
         "Ana pulls at the chain.\n\n"
         "Ana stands and walks to the far side of the room.\n\n"
         "Ana looks up at the door.")
    script = run_node(P, plan_only=True,
                      character_memory="Ana: she, 28, grey shirt.\n"
                                       "Guard: he, 40, uniform.")[3]
    shots = [b.split("]", 1)[1] for b in script.split("[Shot ")[1:]]
    check("every shot after the first is tethered",
          all("at the wall" in s for s in shots[1:]),
          f"{[('at the wall' in s) for s in shots]}")
    check("...and it is the NECK being held, not the wrists",
          all("holding the neck at the wall" in s for s in shots[1:]),
          f"{[('holding the neck' in s) for s in shots]}")
    check("no shot claims the wrists", not any("the wrists" in s for s in shots))
    check("the hardware stays fastened in every later shot",
          all("closed and fastened" in s for s in shots[1:]))
    # The shot that STAGES the chaining says it in the author's own words and must
    # not get the clause back a second time.
    check("the staging shot is not told twice", "at the wall" not in shots[0])
    # A scene with no anchor in it must not sprout one.
    plain = run_node("A room.\n\nThe guard cuffs Ana's wrists.\n\n"
                     "Ana sits down.", plan_only=True,
                     character_memory="Ana: she, 28.\nGuard: he, 40.")[3]
    check("cuffs with no anchor named stay unanchored",
          "at the wall" not in plain and "holding the neck" not in plain)


COLLAR_PHRASINGS = [
    "The guard locks a steel collar around Ana's neck and chains it to the wall.",
    "The guard chains her collar to the wall.",
    "Her collar is chained to the wall.",
    "She is chained to the wall by her collar.",
    "A chain runs from her collar to the wall.",
    "The collar at her throat is padlocked to a ring in the wall.",
    "The guard puts a collar on her and clips the chain to a ring in the wall.",
    "Ana is collared and chained to the wall.",
    "The chain from her collar is bolted to the wall.",
    "Her collar is attached to a chain fixed to the wall.",
    "The guard fastens the collar and secures the chain to the wall.",
    "She wears a steel collar chained to the wall.",
    "The collar around her neck is locked to a wall ring.",
    "A short chain holds her collar to the wall.",
]


def test_every_way_of_chaining_a_collar_to_a_wall():
    """Reported twice. The first fix -- adding the wall to _ANCHOR_POINT -- was
    shipped without a render and did not fix it, because the anchor reader is only
    consulted once a restraint is LATCHED, and six of these fourteen latched nothing.

    restraint_present needs an ambiguous noun (chain, collar) plus a binding verb or
    a body part in the same sentence. _BINDING_VERB holds participles almost
    exclusively -- "chains", "clips" and "holds" are absent and "bolted" is not in it
    at all -- and "collared" matches no noun pattern, so the whole latch stayed down
    and every later shot was free of hardware entirely.

    Hardware fastened to something that does not move IS a restraint, whatever verb
    form said so. Tested on the SHOTS, across every phrasing, because the unit halves
    passed while the film was wrong the first time."""
    print("\n=== every way of chaining a collar to a wall ===")
    for p in COLLAR_PHRASINGS:
        script = run_node("A basement.\n\n" + p + "\n\nAna sits on the floor.\n\n"
                          "Ana stands and walks across the room.",
                          plan_only=True,
                          character_memory="Ana: she, 28.\nGuard: he, 40.")[3]
        last = script.split("[Shot ")[-1]
        check(f"tethered: {p[:44]!r}",
              "at the wall" in last or "at the ring" in last)
        check(f"...and fastened: {p[:40]!r}",
              "closed and fastened" in last or "tied and holding" in last)


def test_moving_towards_something_is_not_being_chained_to_it():
    """The other half of the same rule, and the reason it cannot simply be loosened.

    restraint_present now accepts hardware plus an anchor point, and limb_anchor
    accepts a chain that "runs" or "holds" to a fixed thing. Both would read ordinary
    movement as a fastening if the verb were not constrained -- "she runs to the
    wall" is the exact shape of "a chain runs to the wall"."""
    print("\n=== movement is not fastening ===")
    for t_ in ("she sinks to the floor", "he walks to the table",
               "she is dragged to the bed", "she falls to the floor",
               "he crosses to the window", "she runs to the wall",
               "they move to the bed", "she looks to the door",
               "he runs to the fence", "a chain-link fence runs along the wall",
               "the rope runs to the mast", "she leans back against the wall",
               "the dog runs to the gate",
               # HARDWARE BEING MOVED, not fastened. These are the ones the first
               # attempt let through: _FASTEN_TO was written as bare stems with an
               # optional suffix, and "chain", "rope", "clip" and "lock" are nouns
               # as well as verbs, so the pattern matched its own hardware and called
               # it a fastening. With the restraint gate leaning on the anchor, each
               # of these then latched a hold over hardware lying on the ground.
               "She drops the rope to the floor.", "He throws the chain to the floor.",
               "She kicks the cuffs to the wall.", "The rope lies next to the bed.",
               "He carries the chain to the table.", "The clip fell to the floor.",
               "The chains hang next to the door."):
        check(f"no anchor: {t_!r}", not S.limb_anchor(t_))
        check(f"no restraint: {t_!r}", not S.restraint_present(t_))


def test_two_restraints_put_on_together_both_survive():
    """END TO END, and the unit half would have passed while the film stayed wrong.

    Cuffs and a collar applied in ONE beat recorded only the collar, because
    hardware_named returns a single item and the worn list appended that string. So
    from shot 2 the handcuffs were absent from the prompt entirely -- not held, not
    even named -- and the model drew free hands. Reported as her breaking out of the
    handcuffs, which is the model rendering exactly what it was given."""
    print("\n=== two restraints, one beat ===")
    P = ("A basement cell, bare concrete.\n\n"
         "The guard handcuffs Ana's wrists behind her back and locks a steel "
         "collar around her neck, chained to the wall.\n\n"
         "Ana sits against the wall.\n\n"
         "Ana pulls at the chain.\n\n"
         # This beat calls them "the cuffs" where shot 1 said "handcuffs". Same
         # pair, and the worn list has to know that.
         "Ana twists her wrists in the cuffs.\n\n"
         "Ana looks at the door.")
    script = run_node(P, plan_only=True,
                      character_memory="Ana: she, 28.\nGuard: he, 40.")[3]
    shots = [b.split("]", 1)[1] for b in script.split("[Shot ")[1:]]
    check("the cuffs are named in shot 2", "cuffs" in shots[1], shots[1][:220])
    check("...and the collar with them", "collar" in shots[1])
    check("both stay fastened in every later shot",
          all("closed and fastened" in x for x in shots[1:]))
    # The merged anchor used to read "holding the neck behind the back, at the
    # wall" -- a neck behind a back.
    check("no neck is behind a back",
          not any("neck behind the back" in x for x in shots))
    check("the wrists take the limb position",
          all("the wrists behind the back" in x for x in shots[1:]))
    check("the collar takes the fixed point",
          all("fast at the wall" in x for x in shots[1:]))
    # "Ana looks at the door" must not relocate the camera into a door.
    check("a door does not move the shot",
          not any("in the door," in x for x in shots))
    # ONE pair of handcuffs, however the beats word it. An exact-match dedupe on the
    # worn list produced "The handcuffs, steel collar, chain and cuffs stay closed",
    # which reads as four things and invites the model to draw a spare set.
    held = [x.split(" stay closed")[0].split(" stays closed")[0] for x in shots[1:]]
    check("the cuffs are listed once", all(h.count("cuffs") <= 1 for h in held),
          f"{[h[-90:] for h in held]}")


def test_a_name_that_is_only_spoken_is_not_in_the_shot():
    """Reported: a character turned up in a scene they were not supposed to be in.

        Dana opens the door and calls out: "McKenna where are you?"

    put McKenna's whole sheet line into that shot -- "McKenna: she, 27, green
    dress" -- so the model was handed a full description of her and drew her
    standing there. She is the one person the beat says is NOT in the room.

    Calling for somebody is the commonest way to write their absence, and it was
    reading as their presence. Presence is decided on the beat with its spoken
    spans removed; a name staged OUTSIDE the quote still counts."""
    print("\n=== a spoken name is not a staged one ===")
    mem = "Dana: she, 31, blue jacket.\nMcKenna: she, 27, green dress."

    def described(beat):
        s = run_node("A hallway.\n\n" + beat + "\n\nDana walks on.",
                     plan_only=True, character_memory=mem)[3]
        first = s.split("[Shot ")[1].split("]", 1)[1]
        return {n for n in ("Dana", "McKenna") if n + ":" in first}

    check("called for, and absent",
          described('Dana opens the door and calls out: "McKenna where are you? '
                    'I am looking for you."') == {"Dana"})
    check("spoken about, and absent",
          described('Dana says: "McKenna took the keys."') == {"Dana"})
    check("...even named twice in the line",
          described('Dana shouts: "McKenna! McKenna, answer me."') == {"Dana"})
    # Staged AND spoken still counts: the staging half is what puts her there.
    check("staged as well as spoken",
          described('Dana turns to McKenna and says: "McKenna, wait."')
          == {"Dana", "McKenna"})
    check("staged with no speech at all",
          described("Dana and McKenna walk together.") == {"Dana", "McKenna"})
    # And the same rule where hardware is concerned. Asserted on the SHOT, not on
    # restrained_by_beat: that one deliberately returns everybody when it cannot
    # tell who is applying what -- a hold that fires needlessly is a wasted
    # sentence, one that fails to fire is hardware that stops being described --
    # so the guarantee lives in who gets DESCRIBED, not in that set.
    hw = run_node('A cell.\n\nDan says: "McKenna, put the cuffs on."'
                  '\n\nDan waits.', plan_only=True,
                  character_memory="Dan: he, 40, uniform.\n"
                                   "McKenna: she, 27, green dress.")[3]
    first = hw.split("[Shot ")[1].split("]", 1)[1]
    check("a spoken instruction describes nobody new",
          "McKenna:" not in first, first[:200])
    check("...and puts no hardware on her",
          "cuffs stay" not in first and "cuffs are" not in first, first[:200])


def test_a_verb_is_not_a_room():
    """Found in the same shot as the bug above: "McKenna steps out of the far
    room" read as the flight of STEPS, and "Ana walks into the room" produced
    "This shot is in the room, not the room the scene text names" -- a sentence
    that contradicts itself in eight words.

    first_place searches free text with no preposition in front of it, so it
    cannot tell a noun from a verb. The words that collide do not get to win."""
    print("\n=== a verb is not a room ===")
    for t_ in ("McKenna steps out of the far room.", "She steps forward.",
               "He lounges on the sofa.", "They study the map."):
        check(f"not a place: {t_!r}", S.first_place(t_) not in
              ("steps", "lounge", "study"), repr(S.first_place(t_)))
    check("a bare room names nowhere", not S.first_place("Ana walks into the room."))
    # ...but a qualified one is somewhere.
    check("the back room does", S.first_place("Ana walks into the back room.")
          == "back room")
    check("the far room does", S.first_place("She steps out of the far room.")
          == "far room")
    for room, want in (("A kitchen.", "kitchen"), ("A long hallway.", "hallway"),
                       ("A basement.", "basement"), ("The corridor is dark.",
                                                     "corridor")):
        check(f"{want} still reads", S.first_place(room) == want)


def test_entering_a_room_is_a_journey_not_a_cut():
    """Reported: "Dana continues to walk through the home and enters McKenna's
    bedroom" cut instantly from the living room to the bedroom.

    travel_in returned ('', '', '') for it, so the journey guard said nothing and
    the shot was free to cut. Two reasons, both in the destination pattern:

      - ENTERING is a verb, and the pattern held prepositions only. "into the
        bedroom" read; "enters the bedroom" did not, and entering is how arrival
        is usually written.
      - "McKenna's bedroom" is the ordinary way to say whose room it is, and the
        determiner list was the/her/his/their/a, which no possessive name matches.

    The origin was never the problem: travel_anchor already falls back to the room
    the film is in. It just never learned there was a destination to go to."""
    print("\n=== entering a room is a journey ===")
    for beat, want_to in (
            ("Dana continues to walk through the home and enters McKenna's "
             "bedroom.", "bedroom"),
            ("Dana walks through the house and enters the bedroom.", "bedroom"),
            ("Dana enters McKenna's bedroom.", "bedroom"),
            ("Dana walks down the hall and steps into the kitchen.", "kitchen"),
            ("Dana walks from the living room into the bedroom.", "bedroom")):
        _f, _v, to = S.travel_in(beat)
        check(f"destination read: {beat[:44]!r} -> {to!r}", to == want_to)
    # A named place with no movement is still not a journey.
    check("looking somewhere is not travel",
          S.travel_in("She looks to the bedroom.") == ("", "", ""))
    check("...nor is standing in a room",
          S.travel_in("Dana stands in the bedroom.") == ("", "", ""))

    # END TO END: the walk is one move, and every shot after it is in the new room.
    P = ("A living room in a suburban home.\n\n"
         "Dana stands by the sofa.\n\n"
         "Dana continues to walk through the home and enters McKenna's bedroom.\n\n"
         "Dana looks around the bedroom.")
    script = run_node(P, plan_only=True,
                      character_memory="Dana: she, 31, blue jacket.")[3]
    sh = [b.split("]", 1)[1] for b in script.split("[Shot ")[1:]]
    check("the walk is one continuous move",
          "one continuous move, not a cut" in sh[1], sh[1][:220])
    check("...starting where she was", "begins in the living room" in sh[1])
    check("...and ending where she goes", "ends in the bedroom" in sh[1])
    # The room the film is in follows her, or the scene paragraph keeps saying
    # living room over a shot standing in the bedroom.
    check("the next shot is in the bedroom",
          "in the bedroom, not the room the scene text names" in sh[2], sh[2][:220])
    # The shot before the journey is untouched.
    check("the shot before it says nothing",
          "continuous move" not in sh[0] and "not the room" not in sh[0])


def test_timing_report():
    print("\n=== the timing breakdown ===")
    P = "A room.\n\nOne.\n\nTwo."
    info = run_node(P)[2]
    for want in ("sampling", "decode", "per shot"):
        check(f"info reports {want}", want in info)
    check("...with a wall-clock total", "rendered" in info and "s --" in info)
    # plan_only does no work, so it must not claim any timings.
    check("plan_only reports no timings", "per shot" not in run_node(P, plan_only=True)[2])


def test_upscale_paths():
    print("\n=== upscale wiring ===")
    # Neither pack is installed here, so both fall back. What is under test is that
    # the calls are wired correctly and a render still completes -- the failure mode
    # that shipped twice was a signature mismatch, not a bad upscale.
    P = "A room.\n\nOne.\n\nTwo."
    try:
        imgs, _a, info, _s, _p, total, _sh, _sec = run_node(
            P, latent_upscale="off", upscale="lanczos", upscale_target_short_edge=128)
        ok, detail = imgs.ndim == 4 and imgs.shape[0] == total, str(tuple(imgs.shape))
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    check("a pixel upscale pass runs and returns frames", ok, detail)
    try:
        imgs2 = run_node(P, latent_upscale="off")[0]
        check("no upscale leaves the frames alone", imgs2.ndim == 4)
    except Exception as e:
        check("no upscale leaves the frames alone", False, f"{type(e).__name__}: {e}")
    # The latent handoff must come from the SAMPLED latent, never the upscaled one,
    # or the chain inherits the upscaler's guess and it compounds.
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    # The chain must not inherit the upscaler's reinterpretation: the shot's own
    # frames stay upscaled, but the handoff is decoded from the SAMPLED latent.
    # Eleven boundaries of upscaled-then-downscaled frames compounds into colour
    # cast and mush.
    check("the handoff comes from the pre-upscale latent",
          "pre_up[:, :, -n:]" in src and "hand_src = tail" in src)
    check("...and is clamped before it is re-encoded",
          "clamp(0.0, 1.0)" in src)


def main():
    test_plan()
    test_render()
    test_keyframe_handoff()
    test_references_and_silence()
    test_first_frame()
    test_upscale_paths()
    test_aug_protects_the_keyframe()
    test_beat_reviving_a_garment()
    test_restart_after_removal()
    test_auto_removal()
    test_fall_keeps_the_hardware()
    test_anchor_is_the_scene()
    test_guard_and_layers_end_to_end()
    test_removing_shot_without_a_keyframe()
    test_hardware_anchor_end_to_end()
    test_chain_hold_end_to_end()
    test_every_paragraph_accounted_for()
    test_av_stays_in_sync()
    test_person_described_once_end_to_end()
    test_references_ride_with_the_keyframe()
    test_undressing_completely_end_to_end()
    test_a_tagged_object_comes_off_and_goes_back_on()
    test_hardware_stays_on_its_owner()
    test_a_state_in_the_scene_is_not_reasserted()
    test_the_cuffs_stay_in_the_picture()
    test_a_sheet_that_claims_hardware_too_early()
    test_the_audit_findings_stay_fixed()
    test_no_cuffs_described_without_their_wearer()
    test_caught_first_then_restrained()
    test_a_television_keeps_its_own_voice()
    test_a_shifted_workflow_stops_before_rendering()
    test_the_removal_shot_says_what_is_under()
    test_a_beat_can_name_what_the_layering_hid()
    test_a_removal_says_whose_hands()
    test_the_only_tag_being_on_a_covered_thing()
    test_an_object_tag_works_without_a_face_picture()
    test_an_untagged_picture_defeats_the_layering()
    test_underwear_is_hidden_until_it_is_not()
    test_a_garment_moved_is_not_a_garment_gone()
    test_undressing_does_not_drop_her()
    test_an_unbound_fall_is_told_what_catches_it()
    test_a_named_look_target_is_restated()
    test_a_covered_object_does_not_send_its_picture()
    test_the_anchor_survives_a_close_shot()
    test_mouths_stay_shut_with_no_line()
    test_script_is_what_was_sent()
    test_every_reference_is_claimed()
    test_the_demoted_handoff_is_claimed()
    test_a_gapped_socket_still_sends_its_image()
    test_a_modified_state_is_not_read_as_an_act()
    test_a_staged_change_gets_both_ends()
    test_dialogue_headroom()
    test_introducing_somebody_already_in_position()
    test_back_after_a_shot_away()
    test_a_name_with_no_entry_end_to_end()
    test_sound_survives_silencing()
    test_auto_sound_end_to_end()
    test_room_tone_under_every_shot()
    test_the_decode_keeps_the_vae_it_is_about_to_use()
    test_finished_shots_are_held_in_half_precision()
    test_detail_trend()
    test_timing_report()
    test_no_garment_is_ever_invented()
    test_nothing_wearable_is_ever_added()
    test_a_garment_keeps_its_description()
    test_a_removal_always_names_hands()
    test_hands_and_holds_follow_the_beat()
    test_one_line_is_one_voice()
    test_the_plan_says_whether_silence_can_be_applied()
    test_the_silent_latent_looks_like_silence()
    test_a_softened_handoff_is_not_a_keyframe()
    test_a_journey_reaches_the_shot()
    test_the_scene_does_not_reset()
    test_a_described_room_still_holds()
    test_the_sound_clause_is_inside_the_budget()
    test_the_anchor_is_not_read_as_a_room()
    test_a_garment_does_not_appear_at_the_first_frame()
    test_the_ambient_bed_reaches_the_soundtrack()
    test_built_sound_lands_in_the_right_shot()
    test_built_sound_reaches_an_effort_shot()
    test_one_picture_two_people_is_reported()
    test_a_collar_chained_to_a_wall_stays_on()
    test_every_way_of_chaining_a_collar_to_a_wall()
    test_moving_towards_something_is_not_being_chained_to_it()
    test_two_restraints_put_on_together_both_survive()
    test_a_name_that_is_only_spoken_is_not_in_the_shot()
    test_a_verb_is_not_a_room()
    test_entering_a_room_is_a_journey_not_a_cut()
    test_pacing_reaches_the_thin_shots()
    test_a_line_is_spoken_in_one_language()
    test_undressing_does_not_spread()
    test_a_working_character_is_not_still_lying_down()
    test_an_instruction_is_not_the_action()
    print()
    if _fails:
        print(f"RESULT: {len(_fails)} FAILURE(S): " + "; ".join(_fails))
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
