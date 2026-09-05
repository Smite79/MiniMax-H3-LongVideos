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
    run_node("A room.\n\nHe walks in.\n\nShe says: \"Now.\"", clip=clip2)
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
    run_node(TWO_LINE_ROOM, clip=clip2b, mouths_shut_when_no_line=False)
    check("...and none at all with the mouth guard off",
          clip2b.seen[1][0] == "A room. He walks in.", clip2b.seen[1][0])
    # The shot that DOES speak gets the open form: closing the list there would be
    # telling the model the line is not in it.
    clip4 = FakeCLIP()
    run_node("A room.\n\nShe walks in and says: \"Now.\"", clip=clip4)
    check("a shot with a line is not told that is all there is",
          clip4.seen[1][0] == 'A room. She walks in and says: "Now." '
                              'It sounds like footsteps.', clip4.seen[1][0])
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
    check("...and it is still told to come off",
          "jacket comes off during this shot" in sh[0])
    check("the next shot has lost it", "quilted jacket" not in sh[1])
    check("...and shows what was under it", "grey wool scarf" in sh[1])
    check("info explains the exception", "no keyframe" in run_node(P, plan_only=True)[2])
    # A removing shot that DOES have a keyframe still scrubs in place -- the picture
    # carries the starting state there, so the text does not have to.
    check("a later removing shot still scrubs itself", "grey wool scarf" not in sh[2])
    # With a first_frame wired, shot 1 has a keyframe and behaves like the rest.
    sh_ff = [x for x in re.split(r"(?=\[Shot )", run_node(
        P, plan_only=True, first_frame=torch.rand(1, H, W, 3))[3]) if x.strip()]
    check("a wired first_frame restores the normal rule",
          "quilted jacket" not in sh_ff[0])


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
    check("...but it is still held whole", "closed and fastened as" in soft)
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
    check("...and the line reaches the model as written",
          '"Storms tonight."' in sh[0], "")
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
          "not a usable number" in num[2], "")


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
    check("a real removal still removes", "off during this shot" in off, "")
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
          "holding the wrists" not in sh[0], "")
    check("the next shot is told where they are",
          "above the head, at the bed frame" in sh[1], sh[1][-80:])
    # The one that matters: a close shot crops the anchor out of the picture, so the
    # text is the only thing still carrying it.
    check("...including the close shot", "holding the wrists" in sh[1], "")
    check("...and the shot after that", "holding the wrists" in sh[2], "")
    # It latches like the hardware and is released by the same `remove:`.
    check("a removal lets go of it", "holding the wrists" not in sh[3], sh[3][-70:])
    check("info names the held shots", "fastened limbs held in place on shot(s) 2, 3" in info, "")
    check("...and names the tight framing", "frame tight enough to crop" in info, "")
    # Nothing to anchor, nothing said -- this must not fire on ordinary shots.
    plain = run_node("A room.\n\nMara waits.\n\nMara walks to the window.",
                     plan_only=True, character_memory="Mara: she, 30.")[3]
    check("an unrestrained scene is untouched", "holding the wrists" not in plain, "")


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
    info, script = run_node(P, plan_only=True, character_memory=MEM)[2:4]
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
    check("...and keeps its audio", "sound" in sh[3].lower(), "")
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
    info = run_node(P, audio_vae=vae)[2]
    check("the beat that describes a sound keeps its audio",
          "describe a sound IN THE BEAT" in info)
    check("...and is counted", "1 shot(s) have no line but either describe a sound" in info)
    check("the beat with none is silenced", "1 shot(s) have no quoted line and no sound"
          in info)
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
    # Shots 2 and 3 have no line and no sound of their own: pinned silent, and told
    # nothing about sound, since the clause would describe an acoustic that is not
    # there. This is the one that was babbling.
    check("a shot with no line gets no derived sound",
          "footsteps" not in sh[1])
    check("...and no sound sentence at all",
          "sounds like" not in sh[1] and "only sound" not in sh[1])
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
    # Reported twice: auto_sound was moving the mouth and babbling. H3 is joint, so a
    # free audio branch fills itself with a VOICE and the face lip-syncs to it. No
    # wording suppresses that -- only the silent keyframe does, and it pins the whole
    # shot. So nothing this node INFERS may open the branch: not room tone, and not a
    # sound worked out from the action either.
    check("a shot with no line and no sound of its own is silenced",
          "conditioned on real silence" in info)
    check("...and the room does not go under it", "hard walls" not in sh[1], sh[1][-70:])
    check("...and it is told nothing about sound",
          "sounds like" not in sh[1] and "only sound" not in sh[1])
    # A sound the AUTHOR wrote used to keep that shot open even with no line -- and
    # an open branch is exactly where the invented voice and the lip-sync came from.
    # mouths_shut_when_no_line, on by default, now silences it instead. The old rule
    # is still the rule with the switch off, and that is the trade, stated both ways.
    off = run_node(P, plan_only=True, mouths_shut_when_no_line=False)[3]
    off_sh = [b for b in off.split("---") if b.strip()]
    check("a sound you wrote keeps the branch open, guard off",
          "hard walls giving the sound back" in off_sh[2])
    check("...in the closed form, because it has no line",
          "The only sound" in off_sh[2])
    check("...while on, that shot is silenced so the mouth cannot move",
          "The only sound" not in sh[2] and "Mouths in the shot stay closed" in sh[2])
    check("...and info explains the mouth", "stops the mouth moving" in info)
    # A scene naming no space gets no bed, and the silence guard still applies.
    plain = run_node("Two people talking.\n\nHe waits.\n\nShe waits.", plan_only=True)[2]
    check("no space named, no room tone", "room tone read" not in plain)


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
    test_caught_first_then_restrained()
    test_a_television_keeps_its_own_voice()
    test_a_shifted_workflow_stops_before_rendering()
    test_the_removal_shot_says_what_is_under()
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
    print()
    if _fails:
        print(f"RESULT: {len(_fails)} FAILURE(S): " + "; ".join(_fails))
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
