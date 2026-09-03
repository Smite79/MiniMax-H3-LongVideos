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
    check("...with nothing added to a silenced shot",
          clip2.seen[1][0] == "A room. He walks in.", clip2.seen[1][0])
    # The shot that DOES speak gets the open form: closing the list there would be
    # telling the model the line is not in it.
    clip4 = FakeCLIP()
    run_node("A room.\n\nShe walks in and says: \"Now.\"", clip=clip4)
    check("a shot with a line is not told that is all there is",
          clip4.seen[1][0] == 'A room. She walks in and says: "Now." '
                              'It sounds like footsteps.', clip4.seen[1][0])
    clip3 = FakeCLIP()
    run_node("A room.\n\nHe walks in.", clip=clip3, auto_sound=False)
    check("...and with that off it is exactly what was written",
          clip3.seen[1][0] == "A room. He walks in.")


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
    for i, b in enumerate(blocks, 1):
        check(f"shot {i} holds the restraint",
              S.RESTRAINT_HOLD.strip() in b or S.CHAIN_HOLD.strip() in b)
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
    chain = S.CHAIN_HOLD.strip()
    check("every shot with the hardware holds it rigid", all(chain in s for s in sh))
    # It REPLACES the restraint hold instead of joining it -- both say "whole and
    # closed", and two clauses for one guarantee is twice the stasis in the prompt.
    check("...instead of repeating the restraint hold",
          all(S.RESTRAINT_HOLD.strip() not in s for s in sh))
    check("...while still carrying its guarantee",
          all("whole and closed" in s for s in sh))
    # Rope flexes. Saying it holds a straight line would be wrong, so it does not.
    soft = run_node("A basement.\n\nMaya: 27, a rope around her wrists.\n\n"
                    "Maya lies still.", plan_only=True)[3]
    check("rope is not claimed to be rigid", chain not in soft)
    check("...but it is still held whole", S.RESTRAINT_HOLD.strip() in soft)
    # Steel locked on in shot 1 is still steel in shot 5. Tested per shot rather than
    # latched, the shot naming the chain got the rigid clause and every shot after it
    # fell back to the soft one -- which is where the slack came back from.
    later = [x for x in re.split(r"(?=\[Shot )", run_node(
        "A basement.\n\nMaya: 27, grey coat.\n\nJon locks a chain around her waist.\n\n"
        "Maya walks to the window.\n\nMaya looks down.", plan_only=True)[3]) if x.strip()]
    check("rigidity latches past the shot that names it",
          all(chain in s for s in later))
    check("...and the soft clause is not used instead",
          all(S.RESTRAINT_HOLD.strip() not in s for s in later))
    # A position the hardware enforces latches too: the chain that put a body in a
    # squat is still that length three shots later, so the squat is still the position.
    posed = [x for x in re.split(r"(?=\[Shot )", run_node(
        "A basement.\n\nMaya: 27, grey coat. Wrists cuffed behind back.\n\n"
        "Jon locks a chain from her ankles to her collar, forcing her into a squat.\n\n"
        "Maya strains against the chain, trying to stand.\n\nMaya breathes hard.",
        plan_only=True)[3]) if x.strip()]
    check("a forced position keeps for the rest of the run",
          all(S.CHAIN_POSE_HOLD.strip() in s for s in posed))
    check("...replacing the plain chain clause rather than joining it",
          all(chain not in s for s in posed))
    # No position forced: the plain clause, so an unposed chain does not freeze anyone.
    check("a chain with no position forced stays plain",
          all(S.CHAIN_POSE_HOLD.strip() not in s for s in later))
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
          "Every restraint stays whole" in sh[0])
    check("...and spends no words naming whose", "hardware is Nora's" not in sh[0])
    check("the shot with two names the wearer", "Every restraint on Nora" in sh[1])
    check("...and says whose the hardware is", "The hardware is Nora's" in sh[1])
    check("...pinning the other to his own entry",
          "exactly what their own entry lists" in sh[1])
    # And his entry is still there to be pinned to.
    check("the other character keeps his clothes described", "navy overalls" in sh[1])


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
    # A sound the AUTHOR wrote is a request for audio, so that shot stays open -- and
    # with no line it is told these are the only sounds, which shapes a branch that is
    # legitimately free.
    check("a sound you wrote yourself keeps the branch open",
          "hard walls giving the sound back" in sh[2])
    check("...in the closed form, because it has no line", "The only sound" in sh[2])
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
