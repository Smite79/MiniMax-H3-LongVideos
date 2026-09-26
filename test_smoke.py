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


try:                                     # 3.11 renamed it; both are the same parser
    import re._parser as _re_parser
except ImportError:                      # pragma: no cover - older interpreters
    import sre_parse as _re_parser


def _spellings(pattern, cap=32):
    """Every literal string a small alternation can match.

    The two restraint vocabularies are checked against each other by walking the
    engine's HARDWARE table -- and walking its CANONICAL NAMES only passed while six
    alternate spellings inside those very patterns matched nothing in this file at
    all. A table entry is its pattern, not its label, so the test reads the pattern.

    Only the constructs those patterns use: alternation, an optional group, a
    literal, and a character class. Anything else contributes nothing, which is
    right for a sampler -- it yields fewer strings, never wrong ones."""
    def walk(seq):
        out = [""]
        for op, av in seq:
            kind = str(op)
            if kind.endswith("LITERAL"):
                out = [s + chr(av) for s in out]
            elif kind.endswith("BRANCH"):
                alts = [t for sub in av[1] for t in walk(sub)]
                out = [s + a for s in out for a in alts]
            elif kind.endswith("SUBPATTERN"):
                alts = walk(av[3])
                out = [s + a for s in out for a in alts]
            elif kind.endswith("MAX_REPEAT") or kind.endswith("MIN_REPEAT"):
                lo, _hi, sub = av
                alts = ([""] if lo == 0 else []) + walk(sub)
                out = [s + a for s in out for a in alts]
            elif kind.endswith("IN"):
                picks = [chr(a) for o, a in av if str(o).endswith("LITERAL")]
                out = [s + p for s in out for p in (picks or [" "])]
            if len(out) > cap * 8:
                out = out[:cap * 8]
        return out
    return sorted({s for s in walk(_re_parser.parse(pattern)) if s})[:cap]


# --- fakes ------------------------------------------------------------------

W, H, FRAMES = 128, 96, 39            # small, on the 17k+5 grid
TWO_LINE_ROOM = "A room.\n\nHe walks in."


class FakeCLIP:
    def __init__(self):
        self.seen = []

    def tokenize(self, prompt, minimax_ref_items=None):
        self.seen.append((prompt, list(minimax_ref_items or [])))
        return {"prompt": prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 8, 16), {}]]


def _vae_out_dtype():
    """What a real VAE.decode hands back: VAE.vae_output_dtype() IS
    model_management.intermediate_dtype() (comfy/sd.py). The stubs returned fp32
    unconditionally, so every assertion about what the node does with the VAE's
    dtype was passing on a fixture that could not produce the interesting case."""
    try:
        return _mm.intermediate_dtype()
    except Exception:
        return torch.float32


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
        return torch.rand(max(1, (t - 2) // 5 * 17 + 5), H, W, 3).to(_vae_out_dtype())

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
        return torch.rand(1, n * 800, 2).to(_vae_out_dtype())


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
                shot_seconds=FRAMES / S.H3_FPS, steps=2,
                sampler_name="res_multistep", scheduler="simple", seed=1)
    args.update(kw)
    # the preset is 1024x768; force the small canvas the fakes are built for
    S.NATIVE_RES["4:3"] = (W, H)
    return node.run(**args)


def test_independent_adult_arm_actions():
    print("\n=== separate adult arm actions stay separate across shots ===")
    memory = ("Maya: she, 38, green sweater, grey trousers.\n"
              "Owen: he, 42, blue shirt, brown trousers.")
    first = ("Maya raises her right hand to point at a painting. "
             "Owen keeps his hands at his sides.")
    second = ("Maya lowers her right hand to her side. "
              "Owen folds his arms across his chest.")
    clip = FakeCLIP()
    result = run_node("Maya and Owen stand in an art studio.\n\n"
                      + first + "\n\n" + second,
                      character_memory=memory, clip=clip)
    shots = result[3].split("\n---\n")
    check("both action beats reach rendering", len(shots) == 2)
    check("each character is described once in each shot",
          all(s.count("Maya:") == 1 and s.count("Owen:") == 1 for s in shots))
    check("raising and lowering stay in their own shots",
          first in shots[0] and second not in shots[0]
          and second in shots[1] and first not in shots[1])
    check("no restraint-derived arm pose is invented",
          all("Both arms are" not in s and "wrists together" not in s for s in shots))
    visual_inputs = [items for prompt, items in clip.seen if prompt.strip()]
    check("no first-shot picture and only one continuity picture on shot two",
          [len(items) for items in visual_inputs] == [0, 1])
    check("the reported script matches the text actually encoded",
          [s.split("] ", 1)[1] for s in shots]
          == [prompt for prompt, _ in clip.seen if prompt.strip()])


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


def test_a_cut_keeps_its_own_first_frame():
    print("\n=== a shot that opens on no keyframe keeps frame one ===")
    mem = "Mara: she, 30, a grey coat, white top."
    P = ("A room.\n\nMara stands by the window.\n\n"
         "Mara pulls off her coat.\nremove: coat\n\n"
         "Mara waits.")
    on = run_node(P, character_memory=mem, restart_after_removal=True, trim_seam=True)
    off = run_node(P, character_memory=mem, restart_after_removal=False, trim_seam=True)
    check("the kept frame is reported, not a silent change in the count",
          "kept their FIRST frame" in on[2], on[2][:160])
    check("...and a fully chained run says nothing about it",
          "kept their FIRST frame" not in off[2], "")
    check("a cut keeps the frame a keyframed shot would have dropped",
          on[5] == off[5] + 1, f"{on[5]} vs {off[5]}")
    check("the frames returned match the reported total",
          int(on[0].shape[0]) == on[5], f"{int(on[0].shape[0])} vs {on[5]}")

    def _per_frame(r):
        return r[1]["waveform"].shape[-1] / max(1, int(r[0].shape[0]))
    check("picture and sound are trimmed together",
          abs(_per_frame(on) - _per_frame(off)) < 2.0,
          f"{_per_frame(on):.2f} vs {_per_frame(off):.2f}")
    plain = run_node("A room.\n\nHe walks in.\n\nShe follows.", trim_seam=True)
    check("a fully chained script still trims every seam",
          "kept their FIRST frame" not in plain[2]
          and int(plain[0].shape[0]) == plain[5], str(plain[5]))


def test_a_room_change_with_no_walk_is_a_cut():
    print("\n=== a shot that opens in another room starts fresh ===")
    mem = "McKenna: she, 22, a grey t-shirt, black shorts."
    P = ("A small flat at night. Her bedroom has an unmade bed and a lamp.\n\n"
         "McKenna gets up and comes out of her bedroom.\n\n"
         "McKenna walks down the hallway to the living room.\n\n"
         "McKenna goes from the living room to the kitchen.\n\n"
         "McKenna sits on the sofa in the living room.")
    info = run_node(P, plan_only=True, character_memory=mem)[2]
    check("the shot that jumps rooms is cut, and only it",
          "shot(s) 4 CUT, because" in info, info[:160])
    # A WALK IS NOT THIS: a travel beat opens in the room it is leaving.
    walk = run_node("A flat.\n\nMcKenna walks from the bedroom to the hallway.\n\n"
                    "McKenna walks from the hallway to the kitchen.",
                    plan_only=True, character_memory=mem)[2]
    check("a journey keeps its keyframe", "OPEN IN A DIFFERENT ROOM" not in walk, "")
    # ...but a journey starting somewhere the last shot did not end is still a cut.
    jump = run_node("A flat.\n\nMcKenna is in the kitchen.\n\n"
                    "McKenna walks from the bedroom to the bathroom.",
                    plan_only=True, character_memory=mem)[2]
    check("a walk whose origin is not where we were is a cut",
          "OPEN IN A DIFFERENT ROOM" in jump, jump[:160])
    one = run_node("A kitchen with a white table.\n\nMcKenna fills the kettle.\n\n"
                   "McKenna sits down.", plan_only=True, character_memory=mem)[2]
    check("a one-room script is untouched",
          "OPEN IN A DIFFERENT ROOM" not in one and "never describes" not in one, "")
    # A room the prompt never describes is a room the model invents -- say so.
    check("an undescribed room is reported",
          "never describes" in info and "living room" in info, "")
    desc = run_node("A flat. The living room has a green sofa. The kitchen is small and white."
                    "\n\nMcKenna is in the living room.\n\nMcKenna is in the kitchen.",
                    plan_only=True, character_memory=mem)[2]
    check("a described room draws no warning", "never describes" not in desc, desc[:150])
    # The render path still runs when a shot opens on no keyframe.
    r = run_node(P, character_memory=mem)
    check("the render path runs through a room cut",
          int(r[0].shape[0]) == r[5] and r[5] > 0, str(r[5]))


def test_extras_are_not_forbidden_by_the_body_count():
    print("\n=== a beat that stages extras keeps them ===")
    check("a pair still gets the count",
          "one body for each person" in S.cast_hold(["Dan", "Crystal"]))
    check("...but not when the beat stages extras",
          S.cast_hold(["Dan", "Crystal"], "Two women dance behind them.") == "")
    check("...nor a crowd", S.cast_hold(["Dan", "Crystal"], "A crowd watches.") == "")
    check("a group cue about the cast is not extras",
          "one body for each person" in S.cast_hold(["Dan", "Crystal"], "They both sit down."))
    check("extras_in reads plural people, not body parts",
          S.extras_in("Other girls wait.") and not S.extras_in("He lowers his arms."))

    mem = "Dan: he, 40, a work coat.\nCrystal: she, 26, a red dress."
    sh = [" ".join(x.split()) for x in run_node(
        "A bar.\n\nDan and Crystal sit at the bar.\n\n"
        "Dan and Crystal talk while two women dance behind them.",
        plan_only=True, character_memory=mem)[3].split("---") if x.strip()]
    check("the pair-only shot keeps the count",
          "one body for each person" in sh[0], sh[0][-90:])
    check("...and the shot staging extras does not",
          "one body for each person" not in sh[1], sh[1][-120:])

    img = torch.rand(1, H, W, 3)
    P = "A bar.\n\nCrystal waits.\n\nCrystal turns."
    MEM1 = "Crystal: <Picture 1>, she, 26, a red dress."
    on = run_node(P, plan_only=True, character_memory=MEM1, ref_image_1=img)[2]
    check("the scoping guard is always on, so this stays quiet",
          "fixes the camera on one person" not in on, "")


def test_a_walk_into_an_unlisted_room_is_still_walked():
    print("\n=== heading for the locker room is walked, not cut to ===")
    check("a gym is a place", S.first_place("A school gym, hard light.") == "gym")
    check("a locker room is one room, not 'room'",
          S.travel_legs("They head to the locker room.")[2] == "locker room")
    check("a journey with no origin still walks in",
          "enters the locker room" in S.travel_anchor("", "", "locker room"))
    # ...and the hardware words this file needs are still NOT places.
    check("bars are restraints, not a room",
          S.travel_legs("She is chained to the bars.")[2] == "")
    check("lifting is not travelling",
          S.travel_legs("He lifts her to the bed.")[2] == "")

    mem = "Mia: she, 19, a blue kit.\nTess: she, 19, a red kit."
    P = ("A school gym, hard overhead light.\n\n"
         "Mia and Tess play volleyball with other girls.\n\n"
         "They head to the locker room.")
    out = run_node(P, plan_only=True, character_memory=mem)
    sh = [" ".join(x.split()) for x in out[3].split("---") if x.strip()]
    check("the move is performed as a walk",
          "opens in the gym and arrives in the locker room" in sh[1]
          and "every step in frame" in sh[1], sh[1][-150:])
    check("the extras are not forbidden on the shot that stages them",
          "one body for each person" not in sh[0], sh[0][-110:])
    check("...and the count stays down while they are still there",
          "one body for each person" not in sh[1]
          and "one person in the shot" not in sh[1], sh[1][-110:])
    _SH = "Dan: he, 40, a work coat.\nCrystal: she, 26, a red dress."
    back = [" ".join(x.split()) for x in run_node(
        "A bar.\n\nDan and Crystal sit with two women dancing behind them.\n\n"
        "The two women have gone.\n\nDan and Crystal sit at the bar together.",
        plan_only=True, character_memory=_SH)[3].split("---") if x.strip()]
    check("the count returns once the extras are dismissed",
          "one body for each person" in back[-1], back[-1][-110:])
    check("the room is reported by its real name",
          "enters locker room" in out[2] or "locker room" in out[2], "")
    check("...and not as bare 'room'", "the film enters room," not in out[2], "")


def test_a_move_to_any_place_is_performed():
    print("\n=== a move to a place no list names is still walked ===")
    for where in ("dungeon", "cargo bay", "stable", "chapel", "morgue", "greenhouse",
                  "tent", "boiler room", "cockpit", "sauna", "vault", "crypt",
                  "laundromat", "observatory", "infirmary", "barracks"):
        b = f"They head to the {where}."
        check(f"a {where} is somewhere to go", S.moved_to(b) == where, S.moved_to(b))
        check(f"...and it costs the walk: {where}", S.travel_spaces(b) == 2)
    # What must stay inside the room it is in.
    for thing in ("door", "window", "bench", "bed", "lockers", "table", "bars",
                  "girl", "van"):
        check(f"a {thing} is not somewhere to go",
              S.moved_to(f"She walks to the {thing}.") == "", thing)
    for phrase in ("her hand", "his shoulder", "other girls"):
        check(f"{phrase!r} is not somewhere to go",
              S.moved_to(f"She walks to {phrase}.") == "", phrase)
    # A place named without MOVEMENT is not a journey, exactly as before.
    for b in ("She looks at the dungeon.", "The dungeon is cold.",
              "He waits in the chapel."):
        check(f"not a move: {b[:26]!r}", S.moved_to(b) == "")

    # End to end, across scenes nothing in this file was written for.
    mem = "Mia: she, 19, a blue kit."
    for label, P, dest in (
            ("dungeon", "A stone dungeon lit by torches.\n\nMia stands by the wall."
                        "\n\nMia is led to the armoury.", "armoury"),
            ("freighter", "The cargo bay of a freighter.\n\nMia checks a crate."
                          "\n\nMia heads to the cockpit.", "cockpit"),
            ("farmyard", "A muddy farmyard at dawn.\n\nMia feeds the chickens."
                         "\n\nMia walks to the stable.", "stable")):
        out = run_node(P, plan_only=True, character_memory=mem)
        last = " ".join([x for x in out[3].split("---") if x.strip()][-1].split())
        check(f"{label}: the arrival is performed",
              f"move to the {dest}" in last and "first step to its last" in last,
              last[-120:])
        check(f"{label}: and it is reported",
              f"to the {dest}" in out[2] and "cannot name" in out[2], "")
    # An object destination adds nothing, so ordinary beats are not crowded.
    plain = run_node("A kitchen.\n\nMia fills the kettle.\n\nMia walks to the window.",
                     plan_only=True, character_memory=mem)[3]
    check("a move to an object inside the room adds no clause",
          "The move to the" not in plain, plain[-110:])


def test_an_unstated_frame_becomes_a_portrait():
    print("\n=== a shot that says nothing about the camera is told what the frame holds ===")
    check("a whole-body action gets a frame",
          "whole body" in S.frame_hold("McKenna serves the ball hard across the net."))
    check("...and so does a move", "whole body" in S.frame_hold("McKenna walks to the bench."))
    check("a face acting does not", S.frame_hold("McKenna smiles.") == "")
    # The author's camera always wins -- a close-up included, since somebody asked for it.
    check("a close-up in the beat stands it down",
          S.frame_hold("Close-up on her face as she serves.") == "")
    check("a stated shot size stands it down",
          S.frame_hold("McKenna serves the ball.", "Shot on 35mm, wide shots throughout.") == ""
          and S.frame_hold("McKenna serves the ball.", "Medium shots, eye level.") == ""
          and S.frame_hold("McKenna serves the ball.", "Close-ups throughout.") == "")
    for _a in ("Shot on 35mm, handheld, natural light.", "Cinematic, anamorphic lens.",
               "35mm film, shallow depth of field.", "Handheld, available light, grainy."):
        check(f"a camera note with no size still fires: {_a[:30]!r}",
              S.frame_hold("McKenna serves the ball.", _a) != "", _a)
    check("...and a wide shot named in the beat too",
          S.frame_hold("A wide shot as McKenna serves.") == "")

    mem = "McKenna: she, 22, tall, long blonde hair, blue eyes, a blue kit."
    P = ("A school gym, hard overhead light.\n\n"
         "McKenna serves the ball hard across the net.\n\n"
         "McKenna walks to the bench and picks up a towel.")
    out = run_node(P, plan_only=True, character_memory=mem)
    sh = [" ".join(x.split()) for x in out[3].split("---") if x.strip()]
    check("shot 1 is told what the frame holds",
          "whole body, head to feet" in sh[0], sh[0][-90:])
    # Shot 2 opens on shot 1's last frame with the camera held: that frame IS its frame.
    # Asking for "head to feet, with the room around it" as well is a wider view than
    # the take opens on, and the model cuts to get it -- REPORTED as sex scenes turning
    # into side-angle shots in a different location.
    check("a shot opening on a keyframe is not told to reframe",
          "head to feet" not in sh[1] and "one unbroken take" in sh[1], sh[1][-90:])
    check("the reason is reported", "frame HOLDS" in out[2], "")
    check("a destination stops at a conjunction",
          S.moved_to("McKenna walks to the bench and picks up a towel.") == "")
    check("...and a compound room is still a room",
          S.moved_to("They head to the locker room.") == "locker room"
          and S.moved_to("He walks into the engine room.") == "engine room")
    check("no gibberish reaches the shot", "and picks on screen" not in sh[1], sh[1][-90:])


def test_an_interrupt_releases_the_frame_buffer():
    print("\n=== stopping a run drops the frame buffer before it unwinds ===")
    acc = S.FrameAccumulator(8, torch.float32, True)
    acc.add(torch.zeros((4, 1, 1, 3)))
    acc.add(torch.zeros((4, 1, 1, 3)))
    check("the buffer holds frames", acc.tensor is not None and acc.used == 8)
    acc.release()
    check("release drops the tensor", acc.tensor is None)
    check("...and the overflow list", acc.overflow == [] and acc.used == 0)

    acc2 = S.FrameAccumulator(6, torch.float32, False)
    acc2.add(torch.zeros((3, 1, 1, 3)))
    acc2.add(torch.zeros((3, 1, 1, 3)))
    check("a correctly sized buffer never reaches overflow", acc2.overflow == [])

    # An interrupt out of the render path releases the buffer and is re-raised intact.
    node = S.H3LongVideos()
    held = S.FrameAccumulator(4, torch.float32, True)
    held.add(torch.zeros((4, 1, 1, 3)))
    import comfy.model_management as _mm
    _Interrupt = getattr(_mm, "InterruptProcessingException", None)
    if _Interrupt is None:
        class _Interrupt(BaseException):
            pass
    check("the interrupt is a BaseException, so no `except Exception` eats it",
          issubclass(_Interrupt, BaseException) and not issubclass(_Interrupt, Exception))

    def _boom(_self, _prepared):
        _self._frames = held
        raise _Interrupt()
    _orig = S.H3LongVideos._render
    S.H3LongVideos._render = _boom
    try:
        raised = None
        try:
            node.run(**_render_args())
        except BaseException as e:
            raised = e
        check("the interrupt is re-raised, never swallowed",
              isinstance(raised, _Interrupt), type(raised).__name__)
        check("...and the frame buffer was dropped on the way out",
              held.tensor is None and node._frames is None)
    finally:
        S.H3LongVideos._render = _orig


def _render_args():
    return dict(model=FakeModel(), clip=FakeCLIP(), vae=FakeVAE(), audio_vae=FakeAudioVAE(),
                prompt="A room.\n\nHe walks in.", resolution="4:3", megapixels=0.0,
                shot_seconds=FRAMES / S.H3_FPS, steps=2,
                sampler_name="res_multistep", scheduler="simple", seed=1)


def test_a_posture_denied_is_not_a_posture_taken():
    print("\n=== chained into a squat, she is not told she is standing ===")
    P = "Dan forces McKenna down into a squat. She cannot stand."
    check("the squat is what she is in, not standing",
          S.posture_in(P, ["McKenna"]) == {"McKenna": "squatting"},
          str(S.posture_in(P, ["McKenna"])))
    # An ATTEMPT is not an arrival -- the same error in a friendlier disguise.
    for b in ("McKenna tries to stand.", "McKenna struggles to get up.",
              "McKenna is unable to stand.", "McKenna is no longer able to stand."):
        check(f"not a posture taken: {b[:34]!r}", S.posture_in(b, ["McKenna"]) == {})
    # ...and a posture plainly stated is still read.
    for b, w in (("McKenna stands up.", "standing"), ("McKenna sits down.", "sitting"),
                 ("McKenna squats down.", "squatting")):
        check(f"still read: {b[:22]!r}", S.posture_in(b, ["McKenna"]) == {"McKenna": w})
    check("a cue does not cross a clause boundary",
          S.posture_in("McKenna cannot kneel, so she sits.", ["McKenna"]) == {"McKenna": "sitting"})
    check("...nor does a later strain undo a stated posture",
          S.posture_in("McKenna stands and strains against the chains.", ["McKenna"])
          == {"McKenna": "standing"})
    check("the engine reader agrees", S.engine.posture_in("She cannot stand.") == ""
          and S.engine.posture_in("She stands.") == "standing")

    mem = ("McKenna: she, 22, a grey vest, ankle chains, wrist cuffs.\n"
           "Dan: he, 40, a work coat.")
    SC = ("A bare cell, one bulb.\n\nDan chains her ankles to a ring in the floor.\n\n"
          "Dan forces McKenna down into a squat. She cannot stand.\n\n"
          "McKenna stays down, breathing hard.\n\nMcKenna tries to stand.")
    sh = [" ".join(x.split()) for x in
          run_node(SC, plan_only=True, character_memory=mem)[3].split("---") if x.strip()]
    # The squat is staged in shot 2 (index 1); every shot from there on.
    for i in (1, 2, 3):
        check(f"shot {i + 1} never says she is standing",
              "is still standing" not in sh[i], sh[i][-120:])
        check(f"...and shot {i + 1} keeps the chain at full length",
              "full length" in sh[i], sh[i][-120:])


def test_the_face_plays_the_feeling_the_author_named():
    print("\n=== a stated emotion is played, not clamped shut ===")
    check("an emotion is read in the author's word",
          S.emotion_in("Mia is delighted with the news.") == "delighted"
          and S.emotion_in("McKenna is terrified and shaking.") == "terrified")
    check("...and nothing is invented where none is named",
          S.emotion_in("McKenna walks to the door.") == "")
    check("the clause names the mouth, which the guard would have held shut",
          "mouth" in S.mood_face("delighted") and "delighted" in S.mood_face("delighted"))
    check("no feeling, no clause", S.mood_face("") == "")

    HAPPY = ("A sunny kitchen.\n\nMia and Tess laugh over breakfast.\n\n"
             "Mia is delighted with the news.\n\nMia hugs Tess, beaming.")
    HMEM = "Mia: she, 24, red hair.\nTess: she, 25, dark hair."
    sh = [" ".join(x.split()) for x in
          run_node(HAPPY, plan_only=True, character_memory=HMEM)[3].split("---") if x.strip()]
    check("a compound subject makes both of them the source",
          [n for n, _ in S.vocal_sources_in("Mia and Tess laugh over breakfast.", HMEM)]
          == ["Mia", "Tess"])
    check("...so neither of them is told her mouth stays closed",
          "mouth in the shot stays closed" not in sh[0]
          and "Mouths in the shot stay closed" not in sh[0], sh[0][-130:])
    check("delight is played, not clamped",
          "expression is delighted" in sh[1] and "Mouths in the shot stay closed" not in sh[1],
          sh[1][-130:])
    check("...and so is beaming",
          "expression is beaming" in sh[2] and "Mouths in the shot stay closed" not in sh[2],
          sh[2][-130:])

    DUR = ("A bare cell, one bulb.\n\nMcKenna is cuffed to the bench.\n\n"
           "McKenna is terrified and shaking.\n\nMcKenna stares at the door, desperate.")
    ds = [" ".join(x.split()) for x in
          run_node(DUR, plan_only=True, character_memory="McKenna: she, 22, a grey vest, wrist cuffs.")[3].split("---") if x.strip()]
    check("the author's word replaces the generic mood",
          "expression is terrified" in ds[1] and "the mouth set" not in ds[1], ds[1][-130:])
    check("...and again where she is desperate",
          "expression is desperate" in ds[2] and "the mouth set" not in ds[2], ds[2][-130:])
    check("a duress shot that names no feeling keeps the film's mood",
          "The mood is grim" in ds[0], ds[0][-130:])
    _sheet = "Dan: he, 40.\nMcKenna: she, 22."
    check("a new predicate after `and` credits only its own subject",
          [n for n, _ in S.vocal_sources_in("Dan holds the door and McKenna sobs.", _sheet)]
          == ["McKenna"])


def test_a_pronoun_pointing_away_keeps_the_person_it_means():
    print("\n=== 'Tess kneels beside her' keeps McKenna described ===")
    same = "McKenna: she, 22, long blonde hair.\nTess: she, 21, short dark hair."
    mixed = "McKenna: she, 22, long blonde hair.\nDan: he, 40, a work coat."
    three = same + "\nMara: she, 30, grey hair."
    prev = ["McKenna", "Tess"]

    def kept(sheet, beat):
        return sorted(S.sheet_for_beat(sheet, beat, previous=prev)[1])

    check("a pronoun aimed away keeps the person it means",
          kept(same, "Tess kneels beside her.") == ["McKenna", "Tess"])
    check("...and so does a look", kept(same, "Tess looks at her.") == ["McKenna", "Tess"])
    check("a door shut behind him is still nobody else",
          kept(mixed, "Dan walks out and shuts the door behind him.") == ["Dan"])
    # A possessive is not an object pronoun, and an ordinary beat is untouched.
    check("a possessive keeps its own reading", kept(same, "Tess unlocks her cuffs.") == ["Tess"])
    check("a beat with no pronoun is untouched",
          kept(same, "Tess watches from the door.") == ["Tess"])
    check("two named already leaves the pronoun alone",
          kept(same, "Tess and McKenna look at her hands.") == ["McKenna", "Tess"])
    check("somebody absent is not dragged in",
          S.sheet_for_beat(same, "Tess looks at her.", previous=["Tess"])[1] == ["Tess"])
    check("...nor when nothing is carried at all",
          S.sheet_for_beat(same, "Tess stares at her.", previous=[])[1] == ["Tess"])
    check("three on the sheet, two in the scene, resolved",
          sorted(S.sheet_for_beat(three, "Tess kneels beside her.",
                                  previous=["McKenna", "Tess"])[1]) == ["McKenna", "Tess"])
    # ...and NOTHING is guessed when two candidates are both present.
    check("two candidates present, no guess",
          S.sheet_for_beat(three, "Tess kneels beside her.",
                           previous=["McKenna", "Tess", "Mara"])[1] == ["Tess"])
    check("the reader itself is narrow",
          S.pronoun_points_away("Tess kneels beside her.")
          and not S.pronoun_points_away("shuts the door behind him.")
          and not S.pronoun_points_away("Tess looks at her hands."))

    mem = ("McKenna: she, 22, long blonde hair, large breasts, a grey vest, steel handcuffs.\n"
           "Tess: she, 21, short dark hair, a red top.")
    P = ("A bare cell, one bulb.\n\nDan cuffs McKenna's wrists behind her back.\n\n"
         "McKenna pulls at the cuffs.\n\nMcKenna sits down on the bench.\n\n"
         "Tess kneels beside her.")
    sh = [" ".join(x.split()) for x in
          run_node(P, plan_only=True, character_memory=mem)[3].split("---") if x.strip()]
    check("the shot that only names Tess still describes McKenna",
          "McKenna: she, 22, long blonde hair" in sh[-1], sh[-1][-150:])
    check("...and her restraints are held there too",
          "handcuffs" in sh[-1] or "cuffs" in sh[-1], sh[-1][-150:])


def test_a_feeling_belongs_to_the_face_the_beat_pins_it_on():
    print("\n=== one person's feeling is not worn by everybody in the shot ===")
    cast = ["McKenna", "Dan"]
    check("the owner is the person the beat puts in front of it",
          S.emotion_owner("McKenna is terrified as Dan steps closer.", cast, "terrified")
          == "McKenna")
    check("...and `and` opens a new predicate with its own subject",
          S.emotion_owner("Dan holds the door and McKenna is terrified.", cast, "terrified")
          == "McKenna")
    check("a feeling pinned on nobody has no owner",
          S.emotion_owner("The room is terrified.", cast, "terrified") == "")
    check("a named clause names them", "McKenna's face carries" in S.mood_face("terrified", "McKenna"))
    check("a solo clause stays impersonal", S.mood_face("terrified").startswith(" The face"))
    pairs = S.emotion_pairs("Dan is furious and McKenna is terrified.", cast)
    check("both feelings are pinned", pairs == [("Dan", "furious"), ("McKenna", "terrified")],
          str(pairs))
    check("...and said in one sentence",
          "Dan's face carries furious" in S.mood_faces(pairs)
          and "McKenna's carries terrified" in S.mood_faces(pairs))
    check("no pairs, nothing said", S.mood_faces([]) == "")

    mem = "McKenna: she, 22, long blonde hair, a grey vest.\nDan: he, 40, a work coat."
    P = ("A bare cell, one bulb.\n\nMcKenna is terrified as Dan steps closer.\n\n"
         "McKenna is delighted.\n\nDan is furious and McKenna is terrified.\n\n"
         "Dan holds the door and McKenna is terrified.")
    sh = [" ".join(x.split()) for x in
          run_node(P, plan_only=True, character_memory=mem)[3].split("---") if x.strip()]
    check("two in the shot: the terror is hers and is named",
          "McKenna's face carries" in sh[0] and "The face carries" not in sh[0], sh[0][-120:])
    check("one in the shot: nobody else it could be",
          "The face carries it: the expression is delighted" in sh[1], sh[1][-120:])
    check("two feelings, two faces", "Dan's face carries furious" in sh[2]
          and "McKenna's carries terrified" in sh[2], sh[2][-130:])
    check("a new predicate does not hand it to the other one",
          "McKenna's face carries" in sh[3] and "Dan's face" not in sh[3], sh[3][-120:])
    # The frame clause counts bodies too: "the whole body" of two people is one body.
    check("the frame clause is plural with two in the shot",
          "every body in it whole" in S.frame_hold("McKenna serves the ball.", "", 2))
    check("...and singular with one",
          "the whole body" in S.frame_hold("McKenna serves the ball.", "", 1))


def test_a_bare_region_says_whose_body_it_is():
    print("\n=== a bare region names the body, so the prior does not pick one ===")
    check("the declared pronoun is what names the body",
          S.body_of("she") == "a woman's body" and S.body_of("he") == "a man's body")
    # `they` is an UNDECLARED body, not a licence to guess one.
    check("an undeclared body is not guessed",
          S.body_of("they") == "" and S.body_of("") == "" and S.body_of(None) == "")
    check("the clause carries it",
          "on a woman's body" in S.bare_hold(["legs"], body="a woman's body"))
    check("...and says nothing extra without it",
          "body" not in S.bare_hold(["legs"]).replace("nothing else worn", ""))

    for mem, beat, want in (
            ("McKenna: she, 22, a grey vest, denim shorts.",
             "McKenna takes off her shorts.\nremove: shorts", "the body of a woman of 22"),
            ("Dan: he, 40, a work coat, jeans.",
             "Dan takes off his jeans.\nremove: jeans", "the body of a man of 40")):
        sh = [" ".join(x.split()) for x in run_node(
            f"A bare room.\n\n{beat}\n\nThey stand still.",
            plan_only=True, character_memory=mem)[3].split("---") if x.strip()]
        check(f"the uncovering shot names {want}", want in sh[0], sh[0][-120:])
        check(f"...and so does the shot after it ({want})", want in sh[1], sh[1][-120:])
    nb = [" ".join(x.split()) for x in run_node(
        "A bare room.\n\nAlex takes off their jeans.\nremove: jeans\n\nThey stand still.",
        plan_only=True, character_memory="Alex: they, 30, a coat, jeans.")[3].split("---") if x.strip()]
    check("an undeclared pronoun gets the region without a body",
          "bare from the hip down" in nb[0]
          and "on a woman" not in nb[0] and "on a man" not in nb[0]
          and "the body of a" not in nb[0],
          nb[0][-120:])


def test_a_plural_removal_still_comes_off():
    print("\n=== skirts come off and stay off ===")
    SC = "A changing room.\nMcKenna: she, 22, a grey vest, a denim skirt, a black thong."
    for b in ("McKenna takes off her skirt.", "McKenna takes off her skirts.",
              "McKenna and the other girls take off their skirts.",
              "The other girls take off their skirts.", "They take off their skirts."):
        check(f"read as coming off: {b[:40]!r}", S.infer_removals(b, SC) == ["skirt"],
              str(S.infer_removals(b, SC)))
    SC2 = ("A room.\nDan: he, 40, brown boots, blue jeans, white socks, a grey top.")
    for b, want in (("Dan kicks off his boots.", ["boots"]),
                    ("Dan steps out of his jeans.", ["jeans"]),
                    ("Dan peels off his socks.", ["socks"]),
                    ("Dan and Mia take off their boots.", ["boots"]),
                    ("Dan takes off his tops.", ["top"])):
        check(f"kept as written: {b[:34]!r}", S.infer_removals(b, SC2) == want,
              str(S.infer_removals(b, SC2)))
    check("the normaliser is exact, not a guess",
          S.engine.singular_garment("skirts") == "skirt"
          and S.engine.singular_garment("boots") == "boots"
          and S.engine.singular_garment("jeans") == "jeans"
          and S.engine.singular_garment("panties") == "panties"
          and S.engine.singular_garment("tops") == "top")
    check("...and garment_words agrees with it, off the one owner",
          S.engine.garment_words("take off their skirts") == ["skirt"]
          and S.engine.garment_words("kicks off her boots") == ["boots"])

    # END TO END: scrubbed from the sheet on the next shot and every shot after it.
    mem = "McKenna: she, 22, a grey vest, a denim skirt, a black thong."
    for label, beat in (("one woman", "McKenna takes off her skirt."),
                        ("several", "McKenna and the other girls take off their skirts."),
                        ("unnamed only", "The other girls take off their skirts.")):
        sh = [" ".join(x.split()) for x in run_node(
            f"A changing room.\n\n{beat}\n\nThey stand still.\n\nThey turn around.",
            plan_only=True, character_memory=mem)[3].split("---") if x.strip()]
        def _has_skirt(shot):
            m = re.search(r"McKenna: [^.]*\.", shot)
            return "skirt" in (m.group(0) if m else "")
        check(f"{label}: the skirt is gone from the next shot", not _has_skirt(sh[1]), sh[1][:130])
        check(f"{label}: ...and stays gone", not _has_skirt(sh[2]), sh[2][:130])


def test_a_removal_undresses_only_its_wearer():
    print("\n=== one woman taking her shirt off leaves the other's on ===")
    MEM = ("McKenna: she, 22, a white shirt, a black bra, steel handcuffs.\n"
           "Tess: she, 21, a white shirt, blue jeans.\nDan: he, 40, a work coat.")

    def last_shot(beat):
        sh = [" ".join(x.split()) for x in run_node(
            f"A room.\n\nMcKenna, Tess and Dan stand by the window.\n\n{beat}\n\nThey wait.",
            plan_only=True, character_memory=MEM, ref_noise_aug=0.999)[3].split("---")
            if x.strip()]
        return sh[-1]

    def entry(shot, who):
        m = re.search(who + r": [^.]*\.", shot)
        return m.group(0) if m else ""

    one = last_shot("McKenna takes off her shirt.")
    check("the remover loses her shirt", "shirt" not in entry(one, "McKenna"), entry(one, "McKenna"))
    check("...and the other woman keeps hers", "shirt" in entry(one, "Tess"), entry(one, "Tess"))
    check("...and nothing else of the remover's goes",
          "handcuff" in entry(one, "McKenna"), entry(one, "McKenna"))
    both = last_shot("McKenna and Tess take off their shirts.")
    check("a compound subject undresses both",
          "shirt" not in entry(both, "McKenna") and "shirt" not in entry(both, "Tess"),
          entry(both, "Tess"))
    other = last_shot("Tess takes off her shirt while McKenna watches.")
    check("the other way round too", "shirt" not in entry(other, "Tess")
          and "shirt" in entry(other, "McKenna"), entry(other, "McKenna"))

    hw = last_shot("Dan unlocks McKenna's handcuffs.")
    check("somebody else's removal still reaches the wearer",
          "handcuff" not in entry(hw, "McKenna"), entry(hw, "McKenna"))
    check("...and takes nothing else of hers", "shirt" in entry(hw, "McKenna"), entry(hw, "McKenna"))
    check("the remover reader keeps a compound subject",
          sorted(S.strippers_in("McKenna and Tess take off their shirts.", MEM))
          == ["McKenna", "Tess"])
    check("...and a new predicate after `and` is not a remover",
          S.strippers_in("Dan holds the door and McKenna takes off her shirt.", MEM)
          == ["McKenna"])


def test_a_pairing_the_beat_wrote_is_said_back():
    print("\n=== who is with whom, where it could be read wrong ===")
    CAST = ["Mia", "Tess", "Dan", "Jon"]
    check("two pairs in one beat are both read",
          S.contact_pairs("Mia kisses Dan while Tess kisses Jon.", CAST)
          == [("Mia", "Dan"), ("Tess", "Jon")])
    check("one pair is read", S.contact_pairs("Mia kisses Dan.", CAST) == [("Mia", "Dan")])
    check("furniture is not a partner",
          S.contact_pairs("Dan holds the door while Mia kisses Jon.", CAST) == [("Mia", "Jon")])
    check("...and a chain is not either", S.contact_pairs("Mia pulls the chain.", CAST) == [])
    # NOTHING IS GUESSED where the beat names nobody -- guessing which two is the bug.
    check("an unnamed pairing gets nothing", S.contact_pairs("They kiss.", CAST) == [])
    for b, w in (("Mia dances with Dan.", ("Mia", "Dan")),
                 ("Tess leans against Jon.", ("Tess", "Jon")),
                 ("Mia undresses Dan.", ("Mia", "Dan"))):
        check(f"read: {b[:26]!r}", S.contact_pairs(b, CAST) == [w])
    check("the clause names both sides",
          "Mia with Dan" in S.contact_hold([("Mia", "Dan")]))
    check("...and both pairs when there are two",
          "Mia with Dan" in S.contact_hold([("Mia", "Dan"), ("Tess", "Jon")])
          and "Tess with Jon" in S.contact_hold([("Mia", "Dan"), ("Tess", "Jon")]))
    check("no pair, no clause", S.contact_hold([]) == "")

    mem = ("Mia: she, 20, a red dress.\nTess: she, 21, a blue dress.\n"
           "Dan: he, 22, a work shirt.\nJon: he, 23, a hoodie.")
    sh = [" ".join(x.split()) for x in run_node(
        "A bar.\n\nMia, Tess, Dan and Jon stand together.\n\n"
        "Mia kisses Dan while Tess kisses Jon.\n\nThey all talk.",
        plan_only=True, character_memory=mem, ref_noise_aug=0.999)[3].split("---")
        if x.strip()]
    check("the contact shot names both pairs",
          "Mia with Dan" in sh[1] and "Tess with Jon" in sh[1], sh[1][-130:])
    check("...and a beat with no contact says nothing about it",
          "The contact is" not in sh[2], sh[2][-110:])
    two = [" ".join(x.split()) for x in run_node(
        "A bar.\n\nMia and Dan stand together.\n\nMia kisses Dan.",
        plan_only=True, character_memory="Mia: she, 20, a red dress.\nDan: he, 22, a shirt.",
        ref_noise_aug=0.999)[3].split("---") if x.strip()]
    check("two in the shot need no pairing said", "The contact is" not in two[-1], two[-1][-110:])


def test_keyframe_handoff():
    print("\n=== the keyframe is encoded, once per boundary ===")
    vae = FakeVAE()
    run_node("A room.\n\nOne.\n\nTwo.\n\nThree.", vae=vae)
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
    check("an untagged reference still reaches every shot",
          [len(i) for _, i in shots_seen] == [1, 2],
          str([len(i) for _, i in shots_seen]))
    clip2 = FakeCLIP()
    run_node("A room.\n\nHe walks in.\n\nShe says: \"Now.\"", clip=clip2,
             auto_sound=False)
    check("both shots reached the encoder", len(clip2.seen) == 3)   # + the negative
    check("the beat text is what was sent",
          "He walks in." in clip2.seen[1][0] and "Now." in clip2.seen[2][0])
    check("...with no sound sentence added to a silenced shot",
          "It sounds like" not in clip2.seen[1][0]
          and "the only sound" not in clip2.seen[1][0]
          and S.MOUTH_HOLD in clip2.seen[1][0]
          and "A room. He walks in." in clip2.seen[1][0], clip2.seen[1][0])
    clip2b = FakeCLIP()
    run_node(TWO_LINE_ROOM, clip=clip2b, mouths_shut_when_no_line=False)
    check("...and a stale mouths_shut=False does not take the guard away",
          S.MOUTH_HOLD in clip2b.seen[1][0]
          and "A room. He walks in." in clip2b.seen[1][0], clip2b.seen[1][0])
    clip4 = FakeCLIP()
    run_node("A room.\n\nShe walks in and says: \"Now.\"", clip=clip4)
    check("a shot with a line is not told that is all there is",
          "It sounds like" in clip4.seen[1][0]
          and "footsteps" in clip4.seen[1][0]
          and "the only sound" not in clip4.seen[1][0]
          and "She walks in and says:" in clip4.seen[1][0]
          and "Now." in clip4.seen[1][0], clip4.seen[1][0])
    clip3 = FakeCLIP()
    run_node("A room.\n\nHe walks in.", clip=clip3, auto_sound=False)
    check("...and with that off only the mouth clause remains",
          "It sounds like" not in clip3.seen[1][0]
          and "the only sound" not in clip3.seen[1][0]
          and S.MOUTH_HOLD in clip3.seen[1][0]
          and "A room. He walks in." in clip3.seen[1][0], clip3.seen[1][0])


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
    P = "A room.\n\nOne.\n\nTwo."
    ref = torch.rand(1, 64, 64, 3)
    vae_hi = FakeVAE()
    info_hi = run_node(P, vae=vae_hi, ref_image_1=ref, ref_noise_aug=0.999)[2]
    check("at a safe aug the handoff is a real keyframe",
          "riding as an extra reference" not in info_hi)
    check("...and it is encoded", vae_hi.encodes >= 1, f"{vae_hi.encodes}")
    info_lo = run_node(P, ref_image_1=ref, ref_noise_aug=0.90)[2]
    check("a soft aug demotes the keyframe rather than noising it",
          "riding as an extra reference" in info_lo)
    check("...and the note says what it costs", "weaker continuity" in info_lo)
    info_noref = run_node(P, ref_noise_aug=0.90)[2]
    check("no references means the keyframe is never demoted",
          "riding as an extra reference" not in info_noref)


def test_beat_reviving_a_garment():
    print("\n=== a later beat that names a removed garment ===")
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
    check("the removing shot still says the boots are on", "black boots" in sh[0])
    check("...and shot 2 has lost them", "black boots" not in sh[1])
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
    check("the applying shot is told both ends",
          S.RESTRAINT_GOING_ON.strip() in blocks[0], blocks[0][-90:])
    # Rigid while it goes on -- in the wording for what it IS. A pair of cuffs held
    # its shape by "its links keeping their size and the run between them taut",
    # which is a chain, and is what made cuffs render as chains.
    check("...and keeps it rigid while it goes on, as cuffs and not as a chain",
          S.CUFF_RIGID_TAIL.strip() in blocks[0], blocks[0][-140:])
    check("...and is not handed a chain's wording",
          S.CHAIN_RIGID_TAIL.strip() not in blocks[0], blocks[0][-140:])
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


def test_removing_shot_without_a_keyframe():
    print("\n=== a removal in a shot that has no keyframe ===")
    P = ("A basement.\n\n"
         "Maya: 27, grey wool scarf, black quilted jacket, brown boots.\n\n"
         "Jon cuts off her jacket to expose the scarf.\n\n"
         "Maya lies still.\n\n"
         "Jon pulls off her scarf.")
    sh = [x for x in re.split(r"(?=\[Shot )", run_node(P, plan_only=True)[3]) if x.strip()]
    check("the removing shot still says the garment is worn",
          "quilted jacket" in sh[0])
    check("...and the layer under it is not yet showing",
          "grey wool scarf" not in " ".join(sh[0].split()).split("There is")[0])
    check("...and it is still told to come off",
          "jacket off during this shot" in sh[0] or "jacket comes off during this shot" in sh[0])
    check("...by the hands of somebody the sheet describes",
          "Maya takes the black quilted jacket off during this shot" in sh[0],
          sh[0][-120:])
    check("the next shot has lost it", "quilted jacket" not in sh[1])
    check("...and shows what was under it", "grey wool scarf" in sh[1])
    check("info explains the exception", "no keyframe" in run_node(P, plan_only=True)[2])
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
    check("...instead of repeating the restraint hold",
          all("the run between them" in s for s in sh))
    check("...while still carrying its guarantee",
          all("closed and fastened as it was put on" in s
              or "closed and fastened as they were put on" in s for s in sh))
    # Rope flexes. Saying it holds a straight line would be wrong, so it does not.
    soft = run_node("A basement.\n\nMaya: 27, a rope around her wrists.\n\n"
                    "Maya lies still.", plan_only=True)[3]
    check("rope is not claimed to be rigid", chain not in soft)
    check("...but it is still held whole",
          "tied and holding as" in soft or "closed and fastened as" in soft, soft[-120:])
    check("...and a cord is tied rather than fastened shut",
          "tied and holding as" in soft, soft[-120:])
    later = [x for x in re.split(r"(?=\[Shot )", run_node(
        "A basement.\n\nMaya: 27, grey coat.\n\nJon locks a chain around her waist.\n\n"
        "Maya walks to the window.\n\nMaya looks down.", plan_only=True)[3]) if x.strip()]
    _rigid = "the run between them"
    check("the metal is rigid from the shot that names it",
          all(_rigid in s for s in later))
    check("rigidity latches past the shot that names it",
          all(chain in s for s in later[1:]))
    check("...and that first shot is the one putting it on",
          S.RESTRAINT_GOING_ON.strip() in later[0], later[0][-90:])
    check("...and the soft clause is not used instead",
          all("the run between them" in s for s in later))
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
    s2 = run_node("A basement.\n\nMaya lies still on the floor.", plan_only=True,
                  character_memory="Maya: 27, silver hair, grey coat")[3]
    check("the sheet does not run into the beat", "grey coat Maya" not in s2)
    check("...it is ended properly", "grey coat." in s2)
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
    check("...and stays gone afterwards",
          not any(g in sh[2] for g in ("jacket", "jumper", "jeans", "boots")))
    check("the other character keeps his clothes", "navy overalls" in sh[3])
    check("one sentence, not five", sh[1].count("comes off during this shot") == 1)
    check("...and it is the bare one", "leaving bare skin" in sh[1])
    check("info names what it cleared", "read off the character sheet" in info)
    check("...and lists the garments", "jacket, jumper, t-shirt, jeans, boots" in info)


def test_a_tagged_object_comes_off_and_goes_back_on():
    print("\n=== an object's reference follows the object ===")
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
    check("an add puts the object and its reference back",
          tags[3] == ["<Picture 1>", "<Picture 2>"], str(tags[3]))
    check("...and it stays on after that",
          tags[4] == ["<Picture 1>", "<Picture 2>"], str(tags[4]))


def test_hardware_stays_on_its_owner():
    print("\n=== hardware does not spread to the other character ===")
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
    check("...naming her once, not twice",
          len(re.findall(r"\bNora\b", sh[1].split("Nora:")[-1])) == 1, sh[1][-90:])
    check("...pinning the other to his own entry",
          "exactly what their own entry lists" in sh[1])
    # And his entry is still there to be pinned to.
    check("the other character keeps his clothes described", "navy overalls" in sh[1])


def test_a_state_in_the_scene_is_not_reasserted():
    print("\n=== a state changed in one beat is not re-asserted later ===")
    P = ("Daylight. A yard, and a van with its doors closed.\n\n"
         "Mara and Dom stand behind the van.\n\n"
         "Mara opens the van doors and climbs in.\n\n"
         "Dom looks back at the yard.")
    shots = [s for s in run_node(P, plan_only=True)[3].split("---") if s.strip()]
    check("three shots", len(shots) == 3, "")
    check("shot 1 is told the doors are already closed",
          "already closed at the first frame" in shots[0], "")
    check("the shot that opens them is not told the state holds",
          "already closed" not in shots[1], "")
    check("...and the shot after is not told the old state",
          "first frame" not in shots[2], "")
    check("the scene text itself is untouched",
          all("doors closed" in s for s in shots), "")
    info = run_node(P, plan_only=True)[2]
    check("info names the shot", "shot(s) 1 describe scenery in a state" in info, "")
    check("...and the state really is carried", info.count("shot(s) 1") >= 1, "")
    check("...and the scenery line is in every shot",
          run_node(P, plan_only=True)[3].count("doors closed") == 3, "")


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


def _shots(P, **kw):
    return [x for x in re.split(r"(?=\[Shot )", run_node(P, plan_only=True, **kw)[3])
            if x.strip()]


def test_what_comes_off_in_the_bathroom_stays_off():
    """REPORTED: she undresses in the bathroom, steps into the shower, and her thong
    is back in the next beat. It was supposed to stay off.

    The thong really was scrubbed out of the text -- that part worked. What brought it
    back was the SILENCE where it had been. The region table could not place any
    lower-body underwear, so nothing latched the hips as bare; and bare_hold, which
    says so on every shot after the removal, went quiet for any region the sheet named
    a layer under -- whether or not that layer had come off as well. A full strip is
    the case it matters most in, and it was the one case it could not speak in.

    So the shot said the chest was bare, the feet were bare, and nothing at all about
    the hips. An unspecified region is filled by the model's own prior, the prior for
    a hip is underwear, and the keyframe carried the invention into every shot after
    it. Nothing restored the thong. The prior drew one."""
    print("\n=== what comes off in the bathroom stays off ===")
    mem = "Kate: she, 28, a grey t-shirt, a denim skirt, a black thong, sandals."
    P = ("A tiled bathroom, warm light.\n\n"
         "Kate takes off her clothes and stands naked.\n\n"
         "Kate steps into the shower and the water runs over her.\n\n"
         "Kate turns under the water.\n\n"
         "Kate runs her hands through her wet hair.")
    sh = _shots(P, character_memory=mem)
    check("every beat reaches rendering", len(sh) == 4, str(len(sh)))
    # The text side, which already worked: the garment is out of the sheet.
    for _i in range(1, 4):
        check(f"shot {_i + 1} no longer describes the thong",
              "thong" not in sh[_i].lower(), sh[_i][:120])
    # ...and the half that did not: the space it left is SAID, on every shot after.
    for _i in range(1, 4):
        check(f"shot {_i + 1} says the hips and legs are bare",
              "legs are bare from the hip down" in sh[_i], sh[_i][-160:])
        check(f"...and the chest too", "chest, shoulders and arms are bare" in sh[_i],
              sh[_i][-160:])
    check("the bare clause names skin rather than what is absent",
          all("outermost surface there" in s for s in sh[1:]), sh[1][-90:])
    check("...and carries no negation",
          all(not re.search(r"\bnothing\b|\bno longer\b", s.split("] ", 1)[1])
              for s in sh[1:]), sh[1][-120:])
    bra = "Kate: she, 28, a grey t-shirt, a black bra, jeans."
    sh2 = _shots("A bedroom.\n\nKate pulls off the t-shirt and unhooks the bra.\n\n"
                 "Kate sits on the end of the bed.\n\nKate looks at the window.",
                 character_memory=bra)
    for _i in (1, 2):
        check(f"topless shot {_i + 1} says the chest is bare",
              "chest, shoulders and arms are bare" in sh2[_i], sh2[_i][-160:])
    sh3 = _shots("A bedroom.\n\nKate unzips the denim skirt and lets it fall.\n\n"
                 "Kate sits down.\n\nKate looks up.",
                 character_memory="Kate: she, 28, a denim skirt, a black thong.")
    check("the uncovering shot says what shows there now",
          "thong underneath is what shows" in sh3[0], sh3[0][-160:])
    check("...and never also calls that region bare",
          all("legs are bare from the hip down" not in s for s in sh3),
          sh3[1][-160:])
    check("...and the thong stays described, because it is still on",
          all("thong" in s.lower() for s in sh3[1:]), sh3[1][:140])


def test_the_thong_comes_off_however_it_is_written():
    """The other half: eight of twenty ways people write underwear coming off did
    nothing at all, so the garment stayed in the sheet -- and the sheet goes into
    every shot, which is the garment back on and staying on.

    Worst of them, "lets the thong fall to the floor", matched the RESTORE
    vocabulary: the node read the author's removal and enacted its opposite."""
    print("\n=== the thong comes off however the beat is written ===")
    mem = "Kate: she, 28, a denim skirt, a black thong."
    for _beat in ("Kate hooks her thumbs in the thong and steps out of it.",
                  "Kate pushes the thong down her legs.",
                  "Kate lets the thong fall to the floor.",
                  "Kate shimmies out of the thong.",
                  "Kate pushes the thong off her hips."):
        sh = _shots("A tiled bathroom with a wet floor.\n\n"
                    "Kate unzips the denim skirt and steps out of it.\n\n" + _beat
                    + "\n\nKate steps into the shower.\n\nKate turns under the water.",
                    character_memory=mem)
        check(f"off and stays off: {_beat[5:46]!r}",
              all("thong" not in s.lower() for s in sh[2:]), sh[2][:130])
        check(f"...and the hips are named instead: {_beat[5:34]!r}",
              all("legs are bare from the hip down" in s for s in sh[2:]),
              sh[2][-150:])
    room = ("A tiled bathroom. A glass shower, a bath, a wooden stool.\n\n"
            "Kate steps out of the shower.\n\nKate reaches for the towel.\n\n"
            "Kate looks in the mirror.")
    sh4 = _shots(room, character_memory=mem)
    for _i, _s in enumerate(sh4):
        check(f"shot {_i + 1} still has its shower", "shower" in _s.lower(), _s[:130])


def test_a_breath_does_not_hold_the_branch_open():
    """REPORTED: micro-babble at the start of a scene, as somebody goes to talk.

    "Dana takes a breath." is the beat people write immediately before a line,
    and it read as the author asking for a sound -- so the audio branch stayed
    open for the whole shot with half a second of breath in it. An open branch on
    a joint model fills itself, and at 4-8 steps the last audio step clears
    30-50% of the denoising in one jump, so what it fills with is a voice."""
    print("\n=== a breath does not hold the branch open ===")
    for beat in ("Dana takes a breath.", "Dana draws breath to answer.",
                 "Dana takes a deep breath and turns to her.",
                 "Dana catches her breath."):
        check(f"closed: {beat!r}", not S.sound_described(beat))
    for beat in ("She breathes hard through the gag.", "Dana is breathing hard.",
                 "Dana sighs.", "Dana gasps.", "Dana moans.", "The door slams."):
        check(f"open: {beat!r}", S.sound_described(beat))
    # ...and not when something else in the beat lasts.
    check("a breath beside a chain still opens it",
          S.sound_described("Dana takes a breath as the chain rattles."))
    info = run_node("A room.\n\nDana takes a breath.\n\nDana says: \"Listen.\"",
                    plan_only=True, character_memory="Dana: she, 35.")[2]
    check("the silenced breath is reported", "stage a breath" in info)
    check("...and says how to get it back", "in the same beat as the line" in info)
    quiet = run_node("A room.\n\nDana waits.", plan_only=True,
                     character_memory="Dana: she, 35.")[2]
    check("...and a beat with no breath says nothing", "stage a breath" not in quiet)


def test_appearing_is_not_arriving():
    """REPORTED: ghosting on a character introduction.

    A staged ARRIVAL keeps the previous frame as the keyframe, because somebody
    walking in through a door has a path into a frame that does not have them in
    it. "Appears", "shows up", "turns up" describe the result and not the
    movement -- there is no path, so the only way into that frame is to fade up
    inside it, which is what ghosting is."""
    print("\n=== appearing is not arriving ===")
    for beat, want in (("McKenna walks in through the door.", True),
                       ("McKenna enters the room.", True),
                       ("McKenna comes into the room.", True),
                       ("McKenna follows her in.", True),
                       ("McKenna appears in the doorway.", False),
                       ("McKenna shows up at the door.", False),
                       ("McKenna turns up beside her.", False),
                       ("McKenna appears.", False)):
        check(f"{beat!r} arrives={want}", S.arrives_in(beat) is want,
              str(S.arrives_in(beat)))
    check("a real entrance survives an opinion",
          S.arrives_in("McKenna walks in and appears calm.") is True)
    check("...and so does this one",
          S.arrives_in("McKenna enters and appears nervous.") is True)


def test_a_line_is_marked_however_it_is_punctuated():
    """REPORTED: dialogue duplication and words pronounced wrongly.

    '"Come here," Dana says.' is how half of written dialogue is punctuated, and
    only the text BEFORE a quote was consulted for a speech cue -- so that form
    was never wrapped in <d>...</d> at all. Quotation marks say nothing to the
    model; an unmarked line is one the audio branch was never told is spoken."""
    print("\n=== a line is marked however it is punctuated ===")
    for beat in ('"Come here," Dana says.', '"Come here," she says.',
                 '"Come here," Dana said quietly.'):
        check(f"marked: {beat!r}", "<d>" in S.mark_dialogue(beat),
              S.mark_dialogue(beat))
    check("the cue-first form still works",
          S.mark_dialogue('Dana says: "Come here."').count("<d>") == 1)
    # A scare quote is not a line, whichever side the words are on.
    for beat in ('She wore a "vintage" coat.', 'He called it a "problem" yesterday.',
                 'The sign said "exit" in red.'):
        check(f"not a line: {beat!r}", "<d>" not in S.mark_dialogue(beat),
              S.mark_dialogue(beat))
    mem = "Dana: she, 35."
    thin = run_node("A room.\n\nDana says: \"No.\"\n\nDana waits.",
                    plan_only=True, character_memory=mem, shot_seconds=6.0)[2]
    check("a line that does not fill its shot is reported",
          "no line in it" in thin, thin[-200:])
    full = run_node("A room.\n\nDana says: \"I told you last night that this was "
                    "going to happen and you did not listen to a word of it.\"",
                    plan_only=True, character_memory=mem, shot_seconds=6.0)[2]
    check("...and a line that does fill it is not", "no line in it" not in full)
    # Numbers and abbreviations have no single spoken form, and the model picks.
    hard = run_node('A room.\n\nDana says: "Dr. Vale gets here at 7:30."',
                    plan_only=True, character_memory=mem)[2]
    check("unsayable text in a line is reported", "no single way" in hard)
    check("...naming what it found", "7:30" in hard and "Dr." in hard)
    soft = run_node('A room.\n\nDana says: "Doctor Vale gets here at half seven."',
                    plan_only=True, character_memory=mem)[2]
    check("...and spelled-out dialogue is clean", "no single way" not in soft)
    # Only inside the quotes: narration is never spoken, so its digits are fine.
    narr = run_node('A room.\n\nDana checks the clock at 7:30.\n\n'
                    'Dana says: "You should have told me."',
                    plan_only=True, character_memory=mem)[2]
    check("...and narration is left alone", "no single way" not in narr)


def test_a_two_word_sheet_name_does_not_duplicate_her():
    """REPORTED: a duplicate Mistress, in the FIRST beat -- so nothing to do with
    keyframes or carried frames. sheet_lines took a one-word name, "Mistress
    Vale:" parsed as unlabelled, and an unlabelled line is global: her whole
    description went into every shot with no name attached."""
    print("\n=== a two-word sheet name does not duplicate her ===")
    mem = "Mistress Vale: she, 38, tall, dark hair, a black dress.\nAna: she, 24."
    got = _shots("A panelled study.\n\nThe Mistress stands at the window.\n\n"
                 "Ana kneels by the desk.", character_memory=mem)
    check("the sheet line is attributed, not global",
          "tall, dark hair" not in got[1], " ".join(got[1].split())[:170])
    check("...and she is described in her own shot",
          "tall, dark hair" in got[0], " ".join(got[0].split())[:170])
    # The old failure in one assertion: two descriptions of a woman in one shot.
    both = "Mistress: she, 38, a black dress.\nMistress Vale: she, 38, tall, dark hair."
    one = _shots("A panelled study.\n\nThe Mistress stands at the window.",
                 character_memory=both)
    check("one beat naming one woman describes one woman",
          one[0].count("she, 38") == 1, " ".join(one[0].split())[:200])


def test_dan_is_not_instantiated_twice():
    print("\n=== an exact pair gets one body each ===")
    mem = ("Dan: He, 35, black t-shirt, blue jeans, black shoes.\n\n"
           "Crystal: She, 35, white t-shirt, blue jeans, white shoes.")
    shot = _shots("In a home, Dan and Crystal are in the kitchen.",
                  character_memory=mem)[0]
    check("the exact cast is constrained",
          "There are two people in the shot" in shot, shot)
    check("the constraint asks for one body each",
          "one body for each person" in shot, shot)
    solo = _shots("In a home, Dan is in the kitchen.", character_memory=mem)[0]
    check("the constraint does not invent Crystal in a solo shot",
          "two people in the shot" not in solo, solo)


def test_the_camera_is_held_where_nothing_places_it():
    """REPORTED: the camera moves on its own, which breaks continuity and stops the
    last frame carrying into the next beat.

    Every shot opens on the previous shot's last frame, so a shot that drifts away
    from the viewpoint it started on hands the drifted one forward and the next shot
    adds its own. Nothing in the text ever said the camera stays put, and an unstated
    attribute is left to the model's prior -- which, for a video model, is movement."""
    print("\n=== the camera is held still where nothing places it ===")
    said = S.camera_hold("Maya pours tea.")
    check("a beat that says nothing about the camera gets the hold",
          "one unbroken take from one position, angle and distance" in said, said)
    check("...in one short sentence", said.count(".") == 1 and len(said.split()) <= 14,
          f"{len(said.split())} words")
    check("...phrased positively, like every other guard here",
          not re.search(r"\bnot\b|\bnever\b|\bno\b|\bwithout\b", said, re.I), said)
    check("...and names no camera", not re.search(r"camera|lens", said, re.I), said)
    check("a journey keeps its moving camera",
          S.camera_hold("Maya walks through to the hall.", moving=True) == "")
    for words in ("The camera pans across to Maya.", "Maya pours tea, handheld.",
                  "A slow push in as Maya pours tea.", "Close on her hands, rack focus."):
        check(f"the author's camera wins: {words[:28]!r}", S.camera_hold(words) == "", words)
    for anchor_text in ("Shot on 35mm, handheld.", "Locked-off camera, warm light.",
                        "Anamorphic lens, slow dolly."):
        check(f"...in the anchor too: {anchor_text[:24]!r}",
              S.camera_hold("Maya pours tea.", anchor_text) == "", anchor_text)
    check("a plain anchor does not stand it down",
          S.camera_hold("Maya pours tea.", "A warm kitchen at night.") != "")
    # REPORTED: the camera kept moving. Ordinary prose was read as the author's camera
    # and freed it -- in the anchor, for the whole film.
    for words in ("Maya tilts her head and smiles.", "Owen pulls out a chair and sits.",
                  "Maya fries eggs in a pan.", "Owen puts his handheld radio down.",
                  "Maya circles around the table.", "A drone of traffic outside.",
                  "Maya looks into the camera.", "Dolly in the kitchen pours tea."):
        check(f"prose is not a camera note: {words[:28]!r}", S.camera_hold(words) != "", words)
    check("a lens or a stock is the look, not a move",
          S.camera_hold("Maya pours tea.", "Shot on 35mm, anamorphic lens.") != "")
    for words in ("The camera slowly pushes in on Maya.", "Camera: locked off.",
                  "Maya pours tea, handheld and shaky.", "A drone shot of the farm.",
                  "Maya's point of view."):
        check(f"...while the camera's own words still count: {words[:28]!r}",
              S.camera_hold(words) == "", words)

    mem = "Maya: she, 30, green sweater.\nOwen: he, 34, blue shirt."
    P = ("A kitchen with white tiles.\n\nMaya pours tea.\n\nOwen sits at the table.\n\n"
         "Maya walks through to the living room.\n\nMaya sits on the sofa.")
    out = run_node(P, character_memory=mem, plan_only=True)
    shots = _shots_of(out)
    check("every ordinary shot carries it",
          all("one unbroken take" in shots[i] for i in (0, 1, 3)), shots[0][-120:])
    check("...and the travel shot does not",
          "one unbroken take" not in shots[2], shots[2][-160:])
    check("...and the run says which shots and why",
          "shot(s) 1, 2, 4 say nothing about the camera" in str(out[2])
          and "opens on the PREVIOUS shot's last frame" in str(out[2]), "")
    # A move inside the room is watched from where the camera already is.
    inside = _shots_of(run_node("A living room with a fireplace.\n\nMaya steps towards "
                                "the fireplace.", character_memory=mem, plan_only=True))[0]
    check("a move inside the room keeps the hold",
          "The move to the fireplace" in inside and "one unbroken take" in inside,
          inside[-200:])


class FakePatcher:
    """A ModelPatcher's LoRA bookkeeping, and nothing else.

    ComfyUI keeps `patches` as weight name -> [(strength, delta, strength_model,
    offset, function)], one entry per LoRA that touched that weight, and the last
    LoRA's safetensors metadata under the "lora_metadata" attachment."""

    def __init__(self, keys=4, strength=0.8, stacked=1, name="crystal_v3"):
        self.patches = {f"w{i}": [(strength, object(), 1.0, None, None)] * stacked
                        for i in range(keys)}
        self.patches_uuid = "original"
        self.attachments = {"lora_metadata": {"ss_output_name": name}} if name else {}

    def clone(self):
        twin = FakePatcher.__new__(FakePatcher)
        twin.patches = {k: list(v) for k, v in self.patches.items()}
        twin.patches_uuid = self.patches_uuid
        twin.attachments = dict(self.attachments)
        return twin


class LoraCLIP(FakeCLIP):
    def __init__(self, patcher=None):
        super().__init__()
        self.patcher = patcher if patcher is not None else FakePatcher()

    def clone(self):
        twin = LoraCLIP(self.patcher.clone())
        twin.seen = self.seen
        return twin


def test_the_count_counts_who_the_text_names():
    """REPORTED: randoms turning up in the scene again.

    A shot said "There is one person in the shot: one body, one face" while a clause
    inside it named a second person -- "Ana's legs are bare from the hip down" in a
    shot describing Ben. Those clauses are deliberate: a latched state goes on being
    said while the beat is about somebody else, because the keyframe still shows her
    and silence lets the prior re-dress her. But a name with no body to own it is a
    body the model adds.

    Counting the people the FRAME carries instead was tried and reverted: that
    asserts bodies the text cannot identify at all, which is worse. The line between
    them is what the shot's own words NAME."""
    print("\n=== the body count counts the people the text names ===")
    mem = "Ana: she, 30, a grey t-shirt, blue jeans.\nBen: he, 35, a black coat."
    for label, script, clause in (
            ("a look held past its owner's shot",
             "A workshop.\n\nAna looks at the window.\n\nBen walks in and puts a box down.",
             "Ana's eyes and head are turned"),
            ("a bare region held past its owner's shot",
             "A room.\n\nAna takes off her jeans.\nremove: jeans\n\nBen walks in.",
             "Ana's legs are bare")):
        shots = _shots_of(run_node(script, character_memory=mem, plan_only=True))
        second = " ".join(shots[1].split())
        check(f"{label}: the clause still speaks", clause in second, second[:200])
        check(f"...and the count includes her",
              "There are two people in the shot" in second, second[:200])
        check(f"...without describing her", "Ana:" not in second, second[:200])

    # A shot that names nobody extra counts what it describes, as before.
    plain = _shots_of(run_node("A kitchen.\n\nAna pours coffee.\n\nAna drinks it.",
                               character_memory=mem, plan_only=True))
    check("a solo shot still counts one",
          all("There is one person in the shot" in sh for sh in plain), plain[-1][-120:])
    pair = _shots_of(run_node("A kitchen.\n\nAna and Ben sit at the table.\n\nThey talk.",
                              character_memory=mem, plan_only=True))
    check("a two-hander still counts two",
          "There are two people in the shot" in pair[0], pair[0][-120:])
    carried = _shots_of(run_node("A kitchen.\n\nAna and Ben sit at the table.\n\nBen drinks.",
                                 character_memory=mem, plan_only=True))
    check("a carried person nothing says anything about is not counted",
          "There is one person in the shot" in carried[1], carried[1][-160:])


def test_a_length_goes_where_it_is_put():
    """REPORTED: a steel cable put round the neck and then round the ankles breaks and
    lets the legs drop, and the handcuffs sometimes vanish and are replaced by the
    cable.

    Three causes, all in the readers. A length goes on by being put AROUND -- loop,
    wrap, wind, thread, run -- and none of those were fastening verbs, so the cable
    was recorded on nobody and only the beat that staged it ever mentioned it: from
    the next shot on there was no cable in the text at all. The part it holds was
    read from a table that says a cable holds wrists, not from the neck and ankles
    the beat names. And one length named at two places recorded one of them."""
    print("\n=== a length goes where it is put, and stays there ===")
    E = S.engine
    def state(beat, sheet="Ana: she, 30, a grey t-shirt."):
        st = E.SceneState(place="workshop")
        st.declare("Ana", sheet)
        st.declare("Mara", "Mara: she, 41, overalls.")
        st.read(beat, cast=["Ana", "Mara"], shot=1)
        return {n: sorted((r.item, r.part) for r in p.hardware.values())
                for n, p in st.people.items() if p.hardware}

    got = state("Mara loops a steel cable around Ana's neck and down around her ankles.")
    check("one length at two places is recorded at both",
          got == {"Ana": [("steel cable", "ankles"), ("steel cable", "neck")]}, str(got))
    got = state("Mara tapes her wrists and her ankles.")
    check("...and the tape too, by its own name",
          got == {"Ana": [("tape", "ankles"), ("tape", "wrists")]}, str(got))
    for beat in ("Mara wraps a chain around her ankles.", "Mara winds rope around her wrists.",
                 "Mara threads a cable through her cuffs.", "Mara slings a strap around her waist."):
        check(f"put on by being put around: {beat[:38]!r}", bool(state(beat)), str(state(beat)))
    # ...and none of that reads an ordinary sentence as a restraint.
    for beat in ("She runs to the door.", "The cuffs are on the table.",
                 "The cable runs along the wall."):
        check(f"not a restraint: {beat!r}", not state(beat), str(state(beat)))

    # THE LEGS, held by a length rather than by a fastening verb.
    for text, want in (("a steel cable around her ankles", "ankles together"),
                       ("a cable looped around her ankles", "ankles together"),
                       ("Mara loops a steel cable around Ana's neck and down around her ankles.",
                        "ankles to the neck"),
                       ("a cable from her neck to her ankles", "ankles to the neck"),
                       ("she runs to the door", ""),
                       ("the lamp above her head", "")):
        check(f"legs_anchor {text[:40]!r}", S.legs_anchor(text) == want, S.legs_anchor(text))

    mem = ("Ana: she, 30, a grey t-shirt, steel handcuffs on her wrists behind her back.\n"
           "Mara: she, 41, overalls.")
    P = ("A workshop.\n\nAna kneels on the floor.\n\n"
         "Mara loops a steel cable around Ana's neck and down around her ankles.\n\n"
         "Ana strains against the cable.\n\nAna breathes.")
    shots = _shots_of(run_node(P, character_memory=mem, plan_only=True))
    check("the cuffs are still named once the cable is on",
          all("handcuff" in sh for sh in shots), "")
    check("...and the cable is named on every shot after it goes on",
          all("cable" in sh for sh in shots[1:]), shots[-1][-160:])
    check("...both in one hold sentence",
          "steel handcuffs and steel cable" in shots[-1], shots[-1][-200:])
    check("...and the legs are held up by the line to the neck",
          all("held there by the line running to the neck" in sh for sh in shots[1:]),
          shots[-1][-200:])


def test_any_restraint_holds_from_shot_to_shot():
    """VALIDATION: a person can be restrained any way the author writes it, with any
    hardware, and it holds from shot to shot.

    Walked as a matrix rather than a case, because the gaps were not in the ideas but
    in the vocabularies: this node keeps its own list of restraint words and the
    engine keeps another, and they had drifted. Irons of every kind, a tether, a
    steel cable, a bike lock, cling film -- the engine knew them and the node's own
    reader did not, so the hardware was tracked while the HOLD that keeps it fastened
    never fired."""
    print("\n=== any restraint, any way, shot to shot ===")
    HOLD = re.compile(r"stays?\s+(?:closed and fastened|tied and holding|shut)", re.I)
    HARDWARE = ["steel handcuffs", "zip ties", "rope", "duct tape", "a steel chain",
                "a steel braided tether", "leather cuffs", "shackles", "manacles",
                "a spreader bar", "a leather collar and leash", "a straitjacket",
                "thumb cuffs", "ankle irons", "wrist irons", "leg irons", "a bike lock",
                "a canvas strap", "cling film", "a hobble"]
    bad = []
    for hw in HARDWARE:
        mem = (f"Ana: she, 30, a grey t-shirt, {hw} holding her wrists behind her back."
               "\nMara: she, 41, overalls.")
        shots = _shots_of(run_node("A workshop.\n\nAna kneels on the floor.\n\n"
                                   "Mara looks at her.\n\nAna breathes.",
                                   character_memory=mem, plan_only=True))
        for i, sh in enumerate(shots, 1):
            if not HOLD.search(sh) or "Both arms are" not in sh:
                bad.append(f"{hw} shot {i}")
    check(f"every kind of hardware holds on every shot ({len(HARDWARE)} kinds)",
          not bad, "; ".join(bad[:4]))

    missed = [n for _p, n, _pt in S.engine.HARDWARE
              if not S.restraint_present(f"Ana: she, 30, {n} locked on her wrists.")]
    check("the engine's hardware is hardware to this file too", not missed, str(missed))

    gaps = [(n, form) for pat, n, part in S.engine.HARDWARE
            for form in _spellings(pat)
            if not S.restraint_present(
                f"Ana: she, 30, {form} locked on her {part or 'wrists'}.")]
    check(f"...in every spelling those patterns accept "
          f"({sum(len(_spellings(p)) for p, _n, _pt in S.engine.HARDWARE)} of them)",
          not gaps, str(gaps[:6]))

    deaf = [v for v in ("restrains", "immobilises", "immobilizes", "pinions",
                        "fetters", "collars", "hobbles", "trusses")
            if not S.restraint_present(f"Mara {v} Ana.")]
    check("a verb the engine applies is a restraint to this file too",
          not deaf, str(deaf))

    mem = "Ana: she, 30, a grey t-shirt.\nMara: she, 41, overalls."
    for beat, legs in (("Mara hogties her with a steel cable.", True),
                       ("Mara cuffs her wrists behind her back.", False),
                       ("Mara zip ties Ana's wrists behind her back.", False),
                       ("Mara hogties her.", True),
                       ("Mara trusses Ana up with rope.", True)):
        P = f"A workshop.\n\nAna stands.\n\n{beat}\n\nAna strains.\n\nAna breathes."
        shots = _shots_of(run_node(P, character_memory=mem, plan_only=True))
        check(f"holds after {beat[:34]!r}",
              all(HOLD.search(sh) for sh in shots[2:]), shots[-1][-140:])
        if legs:
            check(f"...and the legs are placed by {beat[:26]!r}",
                  all("Both legs are" in sh for sh in shots[2:]), shots[-1][-140:])

    for beat, cast, want in (("Mara hogties her with a steel cable.", ["Ana", "Mara"], "Ana"),
                             ("Mara cuffs her to the bed frame.", ["Ana", "Mara"], "Ana"),
                             ("The guard handcuffs Ana's wrists.", ["Ana", "Guard"], "Ana"),
                             ("Ana is cuffed by the guard.", ["Ana", "Guard"], "Ana"),
                             ("Mara runs for the door. Dan catches her and cuffs her wrists.",
                              ["Mara", "Dan"], "Mara"),
                             ("Ana unlocks Bea's handcuffs.", ["Ana", "Bea"], "Bea")):
        check(f"wearer of {beat[:34]!r}", S.engine.wearer_of(beat, cast) == want,
              S.engine.wearer_of(beat, cast))
    check("three people and a pronoun is not guessed at",
          S.engine.wearer_of("Mara hogties her.", ["Ana", "Mara", "Vic"]) in ("", "Mara"))
    check("a belt locked on her own hips stays hers",
          S.engine.wearer_of("Nora stands by the bench, the steel belt locked on her hips.",
                             ["Nora", "Victor"]) == "Nora")


def test_a_restraint_survives_the_shot_that_undresses_it():
    """REPORTED: a body chained wrist-to-ankle, and the moment the trousers are pulled
    down the chain breaks and the legs drop back into place.

    Two causes, both in this file. The clauses that hold a restraint shut and say
    where the limbs are fastened were DROPPED FOR ROOM on exactly that shot -- the
    removal sentence and the two bare-region sentences it brings rank above them and
    fill a 90-word budget on their own. And the limb table only ever described ARMS,
    so a hogtie was told where its wrists were and nothing about its legs; a leg the
    text does not place is a leg the model straightens."""
    print("\n=== a restraint survives the shot that undresses the body ===")
    mem = ("Ana: she, 30, grey t-shirt, black trousers, a steel chain linking her wrists "
           "and ankles behind her back.\nMara: she, 41, navy overalls.")
    P = ("A workshop.\n\nAna lies hogtied on the floor.\n\nMara kneels beside her.\n\n"
         "Mara pulls Ana's trousers down to her knees.\nremove: trousers\n\n"
         "Ana pulls against the chain.")
    out = run_node(P, character_memory=mem, plan_only=True)
    shots = _shots_of(out)
    for i, sh in enumerate(shots, 1):
        check(f"shot {i} keeps the hardware shut", "stays closed and fastened" in sh, sh[-160:])
        check(f"shot {i} says where the arms are held", "behind the body" in sh, sh[-160:])
        check(f"shot {i} says where the legs are held", "Both legs are bent back" in sh, sh[-160:])
    check("the removal still happens on the shot that stages it",
          "away by the last frame" in shots[2], shots[2][-200:])
    check("...and what it uncovers is still said", "bare from the hip down" in shots[2].lower()
          or "legs are bare" in shots[2].lower(), shots[2][-200:])

    # THE LEGS, read the ways people write them -- and not read off the scenery.
    for text, want in (("Ana is hogtied on the floor", "ankles to the wrists"),
                       ("hog-tied on the mat", "ankles to the wrists"),
                       ("her ankles chained to her wrists", "ankles to the wrists"),
                       ("her wrists and ankles chained behind her back", "drawn back"),
                       ("her ankles cuffed together", "ankles together"),
                       ("a spreader bar between her ankles", "held apart"),
                       ("her legs spread wide", "held apart"),
                       ("the curtains are drawn back", ""),
                       ("he stands behind her", ""),
                       ("the crates are stacked to the sides", ""),
                       ("she walks to the bed", "")):
        check(f"legs_anchor {text[:34]!r}", S.legs_anchor(text) == want, S.legs_anchor(text))
    # The arms and the legs are two facts, not alternatives.
    both = S.pose_clause("behind the back", legs="ankles to the wrists")
    check("a hogtie is told both halves",
          "Both arms are behind the body" in both and "Both legs are bent back" in both, both)
    check("...and a plain cuffing is unchanged",
          S.pose_clause("behind the back") ==
          " Both arms are behind the body, wrists together at the small of the back.")

    # THE BUDGET is what dropped them, so a shot with hardware gets the room for it.
    check("an ordinary shot keeps the ordinary floor",
          S.fit_guards([(1, "a", "one two three " * 40)], 4)[1] == [],
          str(S.fit_guards([(1, "a", "one two three " * 40)], 4)[1]))
    kept, dropped = S.fit_guards([(1, "a", "word " * 100), (3, "hold", "word " * 60)], 4)
    check("...and drops what will not fit in it", dropped == ["hold"], str(dropped))
    kept, dropped = S.fit_guards([(1, "a", "word " * 100), (3, "hold", "word " * 60)], 4,
                                 floor=S.RESTRAINT_FLOOR_WORDS)
    check("a restrained shot has room for both", dropped == [], str(dropped))


def test_one_photographed_face_and_two_people():
    """REPORTED: duplicates that happen when a picture was NOT used for a character.

    A shot carrying a reference for one person and describing another who has none is
    one photographed face and two people to draw. A reference is the strongest
    identity signal in a prompt -- far stronger than "35, dark hair" -- so the one
    that exists gets used for both bodies and the second character arrives as a copy
    of the first. This file's own note on it said the node could not stop it, because
    no sentence outranks a photo. There is no sentence, but there is a picture: a
    frame from a shot that held the other person ALONE, in the clothes they are
    wearing now, which is the same frame a returning face is recovered from."""
    print("\n=== one photographed face and two people to draw ===")
    mem = "Dan: <Picture 1>, he, 35, black t-shirt.\nCrystal: she, 35, white t-shirt."
    P = ("A kitchen.\n\nDan pours coffee.\n\nCrystal reads at the table alone.\n\n"
         "Dan and Crystal talk.\n\nDan laughs.")
    rows = _encoded_refs(P, character_memory=mem, ref_image_1=torch.rand(1, H, W, 3))
    counts = [n for _p, n in rows]
    tags = [re.findall(r"<Picture (\d+)>", p) for p, _n in rows]
    check("the mixed shot carries a picture for each of them", counts[2] == 2, str(counts))
    check("...each claimed on its own sheet entry",
          "Dan: <Picture 1>," in rows[2][0] and "Crystal: <Picture 2>," in rows[2][0],
          rows[2][0][:200])
    check("...numbered 1..n with nothing unclaimed",
          not _unnamed_pictures(_encoder_rows(P, character_memory=mem,
                                              ref_image_1=torch.rand(1, H, W, 3))),
          str(tags[2]))
    info = str(run_node(P, character_memory=mem, ref_image_1=torch.rand(1, H, W, 3))[2])
    check("the run says whose face it sent and where it came from",
          "shot 3 gave Crystal a face of their own, from shot 2" in info, info[-200:])
    check("...and that tagging her does the same from the first shot",
          "does the same thing from the first shot" in info)

    never = ("A kitchen.\n\nDan and Crystal talk.\n\nDan pours coffee.\n\nCrystal laughs.")
    quiet = str(run_node(never, character_memory=mem, ref_image_1=torch.rand(1, H, W, 3))[2])
    check("nothing is sent when no solo frame of her exists",
          "face of their own" not in quiet)
    check("...and the state is still reported as the hazard it is",
          "no <Picture N> of their own" in quiet, quiet[-200:])

    changed = ("A kitchen.\n\nDan pours coffee.\n\nCrystal reads at the table alone.\n\n"
               "Crystal takes off her jacket.\nremove: jacket\n\nDan and Crystal talk.")
    stale = str(run_node(changed, character_memory=mem + ", a denim jacket",
                         ref_image_1=torch.rand(1, H, W, 3))[2])
    check("a frame from before she changed clothes is not sent",
          "face of their own" not in stale, stale[-200:])

    # NOT BESIDE THE DEMOTED HANDOFF. Below the safe aug the handoff -- the frame she was
    # alone in -- rides in the reference rows as soon as any reference does, so her face
    # sent as well was a second picture of her in the same shot.
    low = _encoded_refs(P, character_memory=mem, ref_image_1=torch.rand(1, H, W, 3),
                        ref_noise_aug=0.95)
    check("below the safe aug she is pictured once: Dan's reference and the handoff",
          low[2][1] == 2, str([n for _p, n in low]))
    check("...and every picture is still named",
          not _unnamed_pictures(_encoder_rows(P, character_memory=mem,
                                              ref_image_1=torch.rand(1, H, W, 3),
                                              ref_noise_aug=0.95)))

    # NOT WHEN BOTH ARE TAGGED: neither is short of a picture.
    both = "Dan: <Picture 1>, he, 35, black t-shirt.\nCrystal: <Picture 2>, she, 35, white t-shirt."
    two = str(run_node(P, character_memory=both, ref_image_1=torch.rand(1, H, W, 3),
                       ref_image_2=torch.rand(1, H, W, 3))[2])
    check("two tagged people need no evening up", "face of their own" not in two)


def test_an_untagged_reference_is_claimed_or_held():
    """REPORTED: character duplicates that survive every guard here, on a setup where
    dropping a LoRA's strength to 0.5 changed nothing.

    A reference image connected with no <Picture N> tag rode EVERY shot with nothing
    in the text naming it -- this file's oldest rule broken in its commonest setup,
    because connecting a face to ref_image_1 without writing a tag is how most people
    wire one up. A picture the text names is that subject; one it never mentions is
    another subject standing beside them, and no wording here can argue with a second
    person that arrives as a picture."""
    print("\n=== an untagged reference is claimed, or it is not sent ===")
    img = lambda: torch.rand(1, H, W, 3)
    mem = "Dan: he, 35, black t-shirt.\nCrystal: she, 35, white t-shirt."
    P = "A kitchen.\n\nDan and Crystal sit at the table.\n\nCrystal laughs.\n\nDan pours coffee."

    rows = _encoded_refs(P, character_memory=mem, ref_image_1=img())
    counts = [n for _p, n in rows]
    tags = [sorted({int(x) for x in re.findall(r"<Picture (\d+)>", p)}) for p, _n in rows]
    check("the two-person shot is sent no reference at all", counts[0] == 0, str(counts))
    check("...and the solo shots get it, claimed on the person they describe",
          counts[1:] == [1, 1] and tags[1] == [1, 2] and tags[2] == [1, 2]
          and all("<Picture 2> is the frame this shot opens on" in p for p, _ in rows[1:]),
          f"{counts} {tags}")
    check("...on that person's own sheet entry", "Dan: <Picture 1>," in rows[2][0], rows[2][0][:160])
    info = str(run_node(P, character_memory=mem, ref_image_1=img())[2])
    check("the run says which shots claimed it", "claimed on the one person they describe" in info)
    check("...and which were sent none", "were sent NO reference" in info)
    check("...and what to do instead", "Tag the pictures" in info)

    # Two pictures and one person is still a guess about which picture is whom.
    two = [n for _p, n in _encoded_refs(P, character_memory=mem, ref_image_1=img(), ref_image_2=img())]
    check("two untagged pictures are never placed", two == [0, 0, 0], str(two))

    plate = [n for _p, n in _encoded_refs("A kitchen with white tiles.\n\nThe kettle boils.\n\n"
                                          "Steam rises from the spout.", ref_image_1=img())]
    check("a script with nobody in it keeps its reference", plate == [1, 1], str(plate))

    # A TAGGED reference is unchanged: it rides the shots that name its subject.
    tagged = _encoded_refs(P, character_memory="Dan: <Picture 1>, he, 35, black t-shirt.\n"
                                               "Crystal: she, 35, white t-shirt.",
                           ref_image_1=img())
    check("tagging it puts it back on every shot that names him, and only those",
          [n for _p, n in tagged] == [1, 0, 1], str([n for _p, n in tagged]))


def test_a_shared_pose_names_nobody():
    """Every clause that owns a fact pays a naming to say whose it is, and this file's
    own rule is that naming somebody twice in one shot draws a second copy. One pose
    shared by everybody needs no names at all."""
    print("\n=== a pose everybody shares is said once, impersonally ===")
    check("two people, one pose", S.posture_hold({"Dan": "sitting", "Crystal": "sitting"},
                                                 ["Dan", "Crystal"]) == " Both are sitting.")
    check("three of them", S.posture_hold({"Dan": "sitting", "Crystal": "sitting", "Mara": "sitting"},
                                          ["Dan", "Crystal", "Mara"]) == " Everyone in the shot is sitting.")
    check("poses that differ still name their owners",
          S.posture_hold({"Dan": "sitting", "Crystal": "kneeling"}, ["Dan", "Crystal"])
          == " Dan is sitting; Crystal is kneeling.")
    check("somebody described with no pose held keeps the naming",
          S.posture_hold({"Dan": "sitting"}, ["Dan", "Crystal"]) == " Dan is sitting.")
    check("standing is still never held", S.posture_hold({"Dan": "standing", "Crystal": "standing"},
                                                         ["Dan", "Crystal"]) == "")
    mem = "Dan: he, 35, black t-shirt.\nCrystal: she, 35, white t-shirt."
    shots = _shots_of(run_node("A kitchen.\n\nDan and Crystal sit at the table.\n\n"
                               "Dan and Crystal laugh together.", character_memory=mem, plan_only=True))
    check("the shot after they sit says it without names",
          "Both are sitting." in shots[1] and "Dan is sitting" not in shots[1], shots[1][-200:])
    # ...and the count of namings is reported, with how much of it is this node's.
    info = str(run_node("A kitchen.\n\nDan and Crystal sit at the table.\n\n"
                        "Crystal laughs and touches Dan's arm.\n\nCrystal looks at Dan.",
                        character_memory=mem, plan_only=True)[2])
    check("a person named three times in a shot is reported",
          "named more than twice in one shot" in info and "from this node" in info, info[:120])


def test_a_lora_is_reported():
    """A LoRA is the one input to a shot this node neither writes nor can read out of
    the text, and it was invisible here: two runs whose prompts are identical render
    differently and nothing else said why. Reporting it is all this does -- a lever
    that scaled it down was tried and taken out again, because dropping a strength to
    0.5 did not stop the duplicates it was built for."""
    print("\n=== what LoRA is attached is reported ===")
    patcher = FakePatcher(keys=6, strength=0.9, stacked=2)
    check("the stack, the weights and the strengths are read off the patcher",
          S.lora_facts(patcher) == (2, 6, [0.9]), str(S.lora_facts(patcher)))
    check("...nothing claimed for a model carrying none", S.lora_facts(object()) == (0, 0, []))
    check("the last one's name comes out of its metadata",
          S.lora_name_of(patcher) == "crystal_v3", S.lora_name_of(patcher))
    check("...and nothing is invented without metadata",
          S.lora_name_of(FakePatcher(name="")) == "")

    mem = "Dan: he, 35, black t-shirt.\nCrystal: she, 35, white t-shirt."
    P = ("A kitchen.\n\nDan and Crystal sit at the table.\n\nCrystal laughs.\n\n"
         "Dan walks out of the kitchen.\n\nCrystal reads.")
    said = str(run_node(P, character_memory=mem, clip=LoraCLIP())[2])
    check("the run says what is attached, and where",
          "1 on the TEXT ENCODER over 4 weights at strength 0.8" in said, said[:200])
    check("...naming the last one applied", "last one applied: crystal_v3" in said)
    check("...and why it is worth reporting at all",
          "neither writes nor can read out of your text" in said)

    # A run with no LoRA anywhere says nothing about LoRA at all.
    quiet = str(run_node(P, character_memory=mem)[2])
    check("a run with no LoRA reports none", "LoRA:" not in quiet)


def test_cuffs_are_not_described_as_a_chain():
    """Reported: handcuffs turn into chains in the shot that uses them.

    _RIGID_HARDWARE puts handcuffs, manacles, shackles and irons in the same bucket
    as chains and padlocks, which is right about the one thing it was asked -- none
    of them flex. The SENTENCE built from it was written for a chain and says so:
    "its links keep their size and the run between them stays taut". Links, a run
    between them, taut. Handed that about a pair of handcuffs, with nothing anywhere
    saying what handcuffs look like, the model draws the thing the words describe.

    A chain's rigidity is its LENGTH holding. A cuff's is two closed rings a fixed
    distance apart. The same guarantee, and it cannot be said in the same words. The
    distance is given, because a length that is not stated is a length the model
    picks, and the one it picks for metal between two wrists is a chain's."""
    print("\n=== cuffs are cuffs, not a length of chain ===")
    MEM = "Mara: she, 26, dark hair."
    sh = [" ".join(b.split("]", 1)[1].split()) for b in run_node(
        "A cell.\n\nDan cuffs Mara's wrists behind her back.\n\n"
        "Mara stands still.\n\nMara waits.", plan_only=True,
        character_memory=MEM)[3].split("[Shot ")[1:]]
    RING, CHAIN = "closed ring locked round each wrist", "run between them"
    check("the applying shot says what cuffs are", RING in sh[0], sh[0][-170:])
    check("...and not what a chain is", CHAIN not in sh[0], sh[0][-170:])
    check("...giving the spacing, so it is not left to the prior",
          "a hand's width apart" in sh[0])
    for i in (1, 2):
        check(f"shot {i + 1} holds the same shape", RING in sh[i], sh[i][-170:])
        check(f"...still not a chain", CHAIN not in sh[i])
    # A CHAIN IS STILL A CHAIN. This fixes what cuffs are told, not what chains are.
    ch = [" ".join(b.split("]", 1)[1].split()) for b in run_node(
        "A basement.\n\nJon locks a chain around her waist.\n\nMaya pulls against the chain.",
        plan_only=True, character_memory="Maya: 27, grey coat.")[3].split("[Shot ")[1:]]
    check("a chain keeps its links", all(CHAIN in s for s in ch), ch[-1][-170:])
    check("...and is not given rings", all("closed ring" not in s for s in ch))
    # Leg irons close round ankles, and the sentence has to say so.
    li = [" ".join(b.split("]", 1)[1].split()) for b in run_node(
        "A cell.\n\nDan locks leg irons on Mara's ankles.\n\nMara shuffles forward.",
        plan_only=True, character_memory=MEM)[3].split("[Shot ")[1:]]
    check("leg irons close round ankles, not wrists",
          all("round each ankle" in s for s in li), li[0][-170:])
    check("...and never say wrist", all("round each wrist" not in s for s in li))
    # The unit, both directions.
    check("cuffs get rings", "closed ring" in S.rigid_tail("handcuffs", "wrists", True))
    check("chains keep links", "links keeping" in S.rigid_tail("chain", "waist", False))
    check("a shot holding both keeps the chain wording",
          "links keeping" in S.rigid_tail("chain and cuffs", "wrists", True))
    check("the sentence form follows the part too",
          "each ankle" in S.cuff_rigid_sentence("ankles")
          and "each wrist" in S.cuff_rigid_sentence("wrists"))


def test_metal_hardware_is_told_what_it_is_made_of():
    """Reported: handcuffs render BLACK instead of steel.

    FORM_HOLD has always ended ", the same object in the same material" -- which
    holds the material steady from shot to shot without ever saying what it is. An
    unspecified attribute is filled from the prior, which is this file's own lesson
    for a bare region and for the anatomy at the hip, and the prior for restraint
    hardware is black: black-finished cuffs, black chain.

    Metal only. Rope, tape and leather come back the way they are written, and a
    default on those would be the node inventing a colour nobody asked for. And it
    only ever fills a GAP: an author who writes steel, or black leather, keeps their
    own word and this says nothing."""
    print("\n=== the metal is told what it is ===")
    STEEL = "bare unpainted steel"
    MEM = "Mara: she, 26, dark hair."
    sh = [" ".join(b.split("]", 1)[1].split()) for b in run_node(
        "A cell.\n\nDan cuffs Mara's wrists behind her back.\n\n"
        "Mara stands still.\n\nMara waits.", plan_only=True,
        character_memory=MEM)[3].split("[Shot ")[1:]]
    check("the applying shot says what the metal is", STEEL in sh[0], sh[0][:200])
    check("...and every shot holding them does too", all(STEEL in s for s in sh))
    check("...agreeing with a plural pair", "The cuffs are bare unpainted steel" in sh[0],
          sh[0][:200])
    # THE MATERIAL, NOT THE FINISH. Saying it is polished put a mirror finish on
    # every pair of cuffs in every film -- one kind of handcuff, everywhere. What has
    # to be excluded is a black coating; polished, brushed, satin or worn is the
    # author's to write, and the sheet saying so stands this clause down entirely.
    for _finish in ("polished", "catching the light", "mirror", "bright"):
        check(f"the finish is left open: no {_finish!r}",
              _finish not in sh[0], sh[0][:200])
    # The author's own word always wins -- this fills a gap, it does not argue.
    leather = " ".join(run_node(
        "A cell.\n\nDan puts black leather cuffs on Mara's wrists.\n\nMara stands still.",
        plan_only=True, character_memory=MEM)[3].split())
    check("black leather is left alone", STEEL not in leather, leather[:200])
    steel_sheet = " ".join(run_node(
        "A cell.\n\nMara stands still.\n\nMara waits.", plan_only=True,
        character_memory="Mara: she, 26, dark hair, steel handcuffs.")[3].split())
    check("a sheet that already says steel is not told again",
          "are bright bare steel" not in steel_sheet, steel_sheet[:200])
    # Not metal, not this clause's business.
    rope = " ".join(run_node(
        "A cell.\n\nDan ties Mara's wrists with rope.\n\nMara stands still.",
        plan_only=True, character_memory=MEM)[3].split())
    check("rope gets no metal sentence", STEEL not in rope, rope[:180])
    # The unit, straight.
    check("bare cuffs get a material", S.hardware_material(["handcuffs"])[1] == STEEL)
    check("...and nothing about how shiny", 
          not re.search(r"polish|shine|shiny|gleam|mirror|bright",
                        S.hardware_material(["handcuffs"])[1], re.I))
    check("...and a chain names its links",
          "links" in S.hardware_material(["chain"])[1])
    for said in ("steel handcuffs", "chrome cuffs", "blackened shackles"):
        check(f"{said!r} says it already", S.hardware_material([said]) == ("", ""))
    for quiet in ("rope", "duct tape", "leather straps", "zip ties"):
        check(f"{quiet!r} is not metal here", S.hardware_material([quiet]) == ("", ""))
    check("a beat naming the material counts too",
          S.hardware_material(["cuffs"], "Dan locks the chrome cuffs on her") == ("", ""))
    for bad in (None, [], [""], [None]):
        check(f"{bad!r} is harmless", S.hardware_material(bad) == ("", ""))


def test_hardware_closed_over_the_groin_stays_closed():
    """Reported: duct tape wound round the waist and between the legs is there in one
    beat and gone in the next, the genitals bare again.

    TWO SENTENCES, BOTH TRUE TO THE NODE, AND ONLY ONE DRAWABLE. held_part() reads a
    hardware noun and answers with a limb -- neck, ankles, waist, body, wrists for
    anything it does not know -- so tape through the crotch came back "wrists", the
    same as handcuffs, and the hold clause said it stayed "tied and holding as it was
    put on" without ever saying WHERE. Meanwhile the bare clause went on calling the
    groin uncovered, because it suppresses for a GARMENT on the sheet and tape is
    hardware. The shot said tape exists somewhere and the genitals are in plain view.

    Sealed until a removal asks, per the author: what is on the genitals stays on."""
    print("\n=== what is closed over the groin stays closed ===")
    MEM = "Mara: she, 26, dark hair, a cotton dress, underwear."
    P = ("A bare room.\n\nMara takes off her dress and underwear.\n\n"
         "Dan wraps duct tape around Mara's waist and between her legs.\n\n"
         "Mara stands still.\n\nMara pulls at the tape.\n\n"
         "Dan cuts the duct tape off.\n\nMara sits down.")
    sh = [" ".join(b.split("]", 1)[1].split()) for b in
          run_node(P, plan_only=True, character_memory=MEM)[3].split("[Shot ")[1:]]
    SEAL, BARE = "covering the groin completely", "genitals uncovered"
    check("before it goes on, the groin is bare as written", BARE in sh[0], sh[0][:150])
    for i in (1, 2, 3):
        check(f"shot {i + 1} says where the tape sits", SEAL in sh[i], sh[i][:170])
        check(f"...and stops calling the groin bare", BARE not in sh[i])
        check(f"...and says it in the opening tokens", sh[i].index(SEAL) < 230, sh[i][:230])
    # Struggling with it is not taking it off.
    check("pulling at it does not take it off", SEAL in sh[3], sh[3][:150])
    # The removal is the author asking, and it is obeyed.
    check("the beat that cuts it off ends the seal", SEAL not in sh[4], sh[4][:150])
    check("...and the groin is bare again", BARE in sh[4])
    check("...and it does not come back", SEAL not in sh[5] and BARE in sh[5])
    # A belt and a chain go the same way, and each is named in the author's own word.
    belt = [" ".join(b.split("]", 1)[1].split()) for b in run_node(
        "A bare room.\n\nDan locks a chastity belt on Mara.\n\nMara stands still.\n\n"
        "Mara walks out to the hallway.", plan_only=True,
        character_memory="Mara: she, 26, dark hair.")[3].split("[Shot ")[1:]]
    check("a chastity belt seals too", all(SEAL in s for s in belt), belt[-1][:150])
    check("...under the author's own word for it",
          all("chastity belt runs around the waist" in s for s in belt), belt[0][:170])
    chain = [" ".join(b.split("]", 1)[1].split()) for b in run_node(
        "A cell.\n\nDan runs a chain around Mara's waist and between her legs.\n\n"
        "Mara stands still.\n\nDan unlocks the chain.\n\nMara stretches.",
        plan_only=True, character_memory="Mara: she, 26, dark hair.")[3].split("[Shot ")[1:]]
    check("a waist-and-crotch chain seals", SEAL in chain[0] and SEAL in chain[1])
    check("...and unlocking it ends it", SEAL not in chain[2] and SEAL not in chain[3])
    # A hand, a knee or a bag between the legs is not hardware.
    for _not in ("Dan puts his hand between her legs", "Mara sits with her bag between her legs",
                 "Dan kneels between her legs"):
        check(f"not hardware: {_not[:34]!r}", S.crotch_seal(_not) == "")
    check("...and cuffs are not either", S.crotch_seal("Dan cuffs her wrists behind her back") == "")


def test_which_way_up_a_lying_body_is_holds():
    """Reported: laid face down and restrained, she is on her back in the next beat.

    The posture table has ONE entry for every way of being down. "lies", "lays",
    "sprawls" -- and "rolls onto her stomach", which names the facing in the beat and
    discards it on the way in. Everything downstream knew only `lying down`, so the
    shot after the one that laid her down said nothing about which way up she was,
    and an unspecified attribute is filled from the prior.

    The one sentence there was said the wrong thing anyway: "The shoulder and the hip
    take the weight of the body" is a body on its SIDE, and it was being said about
    prone and supine bodies alike.

    AND THE POSTURE ITSELF WAS BEING DROPPED. posture_cleared read "turns her head"
    as travel and "closes her eyes" as handling an object, so the latch let go on the
    next beat that had her do anything at all -- taking the facing with it."""
    print("\n=== which way up she is lying holds ===")
    MEM = "Mara: she, 26, dark hair."
    DOWN, UP, SIDE = "lying face down", "lying face up", "lying on one side"
    sh = [" ".join(b.split("]", 1)[1].split()) for b in run_node(
        "A cell.\n\nDan cuffs Mara's wrists behind her back and lays her face down "
        "on the bunk.\n\nMara lies still.\n\nMara turns her head.\n\n"
        "Dan watches her.\n\nMara closes her eyes.",
        plan_only=True, character_memory=MEM)[3].split("[Shot ")[1:]]
    check("the beat that lays her down says which way up", DOWN in sh[0], sh[0][:200])
    for i in (1, 2, 3, 4):
        check(f"shot {i + 1} still has her face down", DOWN in sh[i], sh[i][:200])
        check(f"...and never turns her over", UP not in sh[i])
    # Moving her own head or eyes is done lying down as readily as standing.
    check("turning her head does not stand her up", DOWN in sh[2], sh[2][:200])
    check("closing her eyes does not either", DOWN in sh[4], sh[4][:200])
    # The author can turn her over, and then it holds the other way.
    roll = [" ".join(b.split("]", 1)[1].split()) for b in run_node(
        "A cell.\n\nMara lies on her back on the bunk.\n\nMara rolls onto her "
        "stomach.\n\nMara lies still.", plan_only=True,
        character_memory=MEM)[3].split("[Shot ")[1:]]
    check("on her back is read as face up", UP in roll[0], roll[0][:200])
    check("rolling over is read, not discarded", DOWN in roll[1], roll[1][:200])
    check("...and the new facing holds", DOWN in roll[2] and UP not in roll[2])
    # Getting up ends it.
    stand = [" ".join(b.split("]", 1)[1].split()) for b in run_node(
        "A cell.\n\nDan lays Mara face down on the bunk.\n\nMara lies still.\n\n"
        "Mara stands up.\n\nMara walks to the door.", plan_only=True,
        character_memory=MEM)[3].split("[Shot ")[1:]]
    check("standing up lets the facing go", DOWN not in stand[2], stand[2][:200])
    check("...and it does not come back", DOWN not in stand[3])
    # The unit, both directions.
    for _t, _want in (("Dan lays her face down", "face down"),
                      ("Dan lays her on her stomach", "face down"),
                      ("He pushes her prone on the floor", "face down"),
                      ("She lies on her back", "face up"),
                      ("She rolls onto her side", "on the side"),
                      ("Dan cuffs her wrists behind her back", ""),
                      ("She lies down on the bed", "")):
        check(f"{_t!r} -> {_want!r}", S.lying_facing(_t) == _want)
    # The weight sentence belongs to the facing that has it.
    _side = S.pose_clause("behind the back", lying=True, facing="on the side")
    _prone = S.pose_clause("behind the back", lying=True, facing="face down")
    check("the side wording is the side's", "shoulder and the hip" in _side)
    check("...and prone gets its own", "chest" in _prone and "shoulder and the hip" not in _prone)
    check("an unwritten facing claims no side",
          "shoulder and the hip" not in S.pose_clause("behind the back", lying=True))
    check("...but still says what is under her",
          "the weight of the body" in S.pose_clause("behind the back", lying=True))


def test_a_body_not_in_the_shot_gets_no_position():
    """Her arms, placed on him.

    _wearer_here has always cleared `hold` and `_pose` for a shot the restrained
    person is not described in -- a body that is not in the frame does not get its
    arms put anywhere. The hoist that moved the limb position into the opening tokens
    was written ABOVE that gate, so the sentence was already spliced into the shot's
    line by the time the gate cleared the variable, and clearing it removed nothing.
    The shot read "Dan: he, 40", "There is one person in the shot", and "Both arms
    are behind the body, wrists together at the small of the back" -- her position,
    on the only body left in the frame.

    A fastened person now STAYS in the shot when the beat names somebody else, so
    reaching this gate takes a beat that really removes her: somebody leaving takes
    the camera with them. That is the case here, and the gate still has to hold."""
    print("\n=== a body that is not in the shot gets no position ===")
    MEM = ("Mara: she, 26, dark hair, steel handcuffs behind her back, a collar.\n"
           "Dan: he, 40, a work shirt.")
    P = ("A bedroom with a lamp on the table.\n\n"
         "Dan cuffs Mara's wrists behind her back and lays her on the bed.\n\n"
         "Mara lies still.\n\nDan stops and sits back.\n\n"
         "Dan walks out and shuts the door.")
    sh = [" ".join(b.split("]", 1)[1].split()) for b in
          run_node(P, plan_only=True, character_memory=MEM)[3].split("[Shot ")[1:]]
    POSE = "arms are behind the body"
    check("the shot that cuffs her places her arms", POSE in sh[0], sh[0][:170])
    check("...and so does the one she is in", POSE in sh[1], sh[1][:170])
    # She is fastened and nobody leaves, so she stays in his beat.
    check("a beat naming only him keeps her, because she is fastened",
          "Mara:" in sh[2], sh[2][:200])
    check("...with her arms still placed", POSE in sh[2], sh[2][:200])
    check("...and her cuffs still described", "cuffs" in sh[2], sh[2][:220])
    check("info says she was kept", "kept in frame by their hardware" in
          str(run_node(P, plan_only=True, character_memory=MEM)[2]))
    # He LEAVES, and the camera goes with him: she is not in that shot, and her
    # body must not be described on his.
    check("the shot he walks out of does not name her", "Mara:" not in sh[3], sh[3][:200])
    check("...and does NOT place her arms on him", POSE not in sh[3], sh[3][:220])
    check("...and says nothing about her cuffs", "cuffs" not in sh[3], sh[3][:220])

def test_the_limb_position_leads_the_shot():
    """Reported, and not fixed by saying it more: wrists cuffed in FRONT of the body
    on every shot using handcuffs, while the text said behind the back five times.

    This file already measured the rule and built beat_leads on it -- what LEADS a
    prompt decides its composition, anatomy in the opening tokens is what a distilled
    model settles the frame on, and at cfg 1 no later sentence outvotes it. The limb
    position was then left sitting after the sheet, the body count, the hardware
    clause and the chain clause: a later sentence, by this file's own finding. Five
    late sentences lose to the opening tokens.

    So it is MOVED, not repeated. The words are the same words; only their position
    changed, which is the one lever here that has ever moved composition."""
    print("\n=== where the limbs are leads the shot ===")
    MEM = "Mara: she, 26, dark hair, a grey t-shirt.\nDan: he, 40, a work coat."
    P = ("A cell.\n\nDan cuffs Mara's wrists behind her back.\n\n"
         "Mara stands against the wall.\n\nMara turns to face the door.")
    sh = [" ".join(b.split("]", 1)[1].split()) for b in
          run_node(P, plan_only=True, character_memory=MEM)[3].split("[Shot ")[1:]]
    POSE = "Both arms are behind the body, wrists together at the small of the back"
    for i, s in enumerate(sh, 1):
        check(f"shot {i} carries the position", POSE in s, s[:160])
        check(f"...and says it ONCE, not twice", s.count(POSE) == 1)
        # It must sit ahead of the sheet, which is what it used to sit behind.
        check(f"...ahead of the character sheet",
              s.index(POSE) < s.index("Mara: she, 26"), s[:200])
    # The beat is still first: your words lead, the position follows them.
    check("the beat still leads", sh[0].index("Dan cuffs") < sh[0].index(POSE), sh[0][:150])
    # And on the applying shot it precedes the hardware clause it used to trail.
    check("the position precedes the hardware clause",
          sh[0].index(POSE) < sh[0].index("The hardware goes on"), sh[0][:200])
    # A shot with no limb position must be untouched.
    plain = [" ".join(b.split("]", 1)[1].split()) for b in
             run_node("A kitchen.\n\nMara pours a glass of water.\n\nMara drinks it.",
                      plan_only=True, character_memory="Mara: she, 26, dark hair."
                      )[3].split("[Shot ")[1:]]
    check("a shot with no restraint gets no hoisted pose",
          all("Both arms are" not in s for s in plain), plain[0][:160])
    check("...and its sheet still follows its beat",
          plain[0].index("Mara pours") < plain[0].index("Mara: she, 26"), plain[0][:150])


def test_the_pronoun_swap_never_touches_your_words():
    """END TO END: the rewrite is confined to the clauses this node wrote.

    The unit test covers the forms. What it cannot cover is the boundary, which is
    the part that matters: your beat, your scene and your exact lines go to the model
    the way you typed them, and a swap that reached into them would be this node
    editing the author -- the one thing it does not do."""
    print("\n=== a repeated naming becomes a pronoun, in the node's words only ===")
    mem = "Kate: she, 30, blue coat, scarf.\nSam: he, 34, black shirt."
    beat = 'Kate takes off her scarf and says: "It is warm in here."'
    P = ('A living room.\n\nKate and Sam sit on the sofa and she says: "Sit down."\n\n'
         + beat + "\n\nKate walks him down the hallway to the tiled bathroom.")
    out = run_node(P, plan_only=True, character_memory=mem)
    sh = [" ".join(x.split()) for x in re.split(r"(?=\[Shot )", out[3]) if x.strip()]
    two, info = sh[1], str(out[2])
    check("your beat is in the shot exactly as written",
          "Kate takes off her scarf and says:" in two, two[:200])
    # The node names her once in its own clauses and points back for the rest.
    check("the first clause naming still names her", "Kate is sitting" in two, two[-260:])
    check("...and the mouth clause points back with a pronoun",
          "Only she speaks" in two, two[-260:])
    check("info says a naming was spent as a pronoun",
          "spent as a PRONOUN" in info and "Kate x1" in info)
    # Two women in the shot: the pronoun would not resolve, so nothing is rewritten.
    mem2 = "Kate: she, 30, blue coat.\nAna: she, 27, red dress."
    two = " ".join(run_node("A bar.\n\nKate sits and Ana looks at her. Kate waits.",
                            plan_only=True, character_memory=mem2)[3].split())
    check("two women means no swap: the clauses keep their names",
          "Only she speaks" not in two and " she is sitting" not in two.lower(), two[-200:])


def test_verbatim_sends_your_text_and_nothing_else():
    """VERBATIM IS GONE, and so are the seven switches it stood in for.

    It sent the prompt with no clause this node writes, to tell the node's doing from
    the model's. That is a real diagnosis and it has no replacement -- which is worth
    a test rather than a shrug, because the thing this asserts is a capability that
    was deliberately given up: there is now no way to render your text without the
    node's sentences over it.

    What is left is info, which still reports what every clause WOULD have said. The
    guards themselves are no longer switchable at all, so what this test guards is
    that none of them came back as a widget and that a stale workflow still sending
    one is absorbed rather than obeyed."""
    print("\n=== the guards are not switchable ===")
    _spec = S.H3LongVideos.INPUT_TYPES()
    _req, _opt = _spec["required"], _spec.get("optional", {})
    for _w in ("verbatim", "character_guard", "hold_gaze", "hold_scene_state",
               "mouths_shut_when_no_line", "hold_camera", "auto_sound", "beat_leads"):
        check(f"{_w} is not a widget", _w not in _opt and _w not in _req)
    mem = "Kate: she, 30, blue coat.\nSam: he, 34, black shirt."
    P = 'A room.\n\nKate and Sam stand together. Kate says: "Come here."'
    plain = run_node(P, plan_only=True, character_memory=mem)[3]
    # Every stale value a saved workflow could still be sending, all at once.
    stale = run_node(P, plan_only=True, character_memory=mem, verbatim=True,
                     character_guard=False, hold_gaze=False, hold_scene_state=False,
                     mouths_shut_when_no_line=False, hold_camera=False,
                     auto_sound=False, beat_leads=False)[3]
    check("a stale workflow turning all of them off changes nothing", stale == plain)
    check("...and the mouth guard really is in there", "Only Kate speaks" in plain)
    check("...and the camera take", "one unbroken take" in plain)
    info = run_node(P, plan_only=True, character_memory=mem)[2]
    check("VERBATIM says nothing, because there is no verbatim",
          "VERBATIM is on" not in info)
    check("...and the balance of the prompt is still reported", "prompt balance" in info)


def test_an_exact_line_is_yours_untouched():
    """Everything else in a shot is either the author's text put through a reader or a
    clause this file wrote, and on a short beat the node's own clauses were measured
    at 70% of the shot against the beat's 8%. An `exact:` line is neither: it is
    placed after the beat in the author's words and nothing reads, scopes, scrubs,
    reorders or drops it."""
    print("\n=== an exact: line reaches the model word for word ===")
    mem = "Ana: she, 29, grey jacket.\nMara: she, 41, navy uniform."
    P = ("A depot at night.\n\nMara walks Ana to the entrance.\n"
         "exact: Ana's wrists stay behind her back the whole way.\n"
         "exact: The light stays low\n\n"
         "Ana stops at the door.")
    out = run_node(P, plan_only=True, character_memory=mem)
    shots = _shots_of(out)
    check("the line is in the shot, word for word",
          "Ana's wrists stay behind her back the whole way." in shots[0], shots[0][:220])
    check("...a second one too, given the full stop it lacked",
          "The light stays low." in shots[0], shots[0][:220])
    check("...straight after the beat, ahead of the node's own clauses",
          shots[0].index("wrists stay behind") < shots[0].index("There are two people"), "")
    check("...and only in its own shot", "wrists stay behind" not in shots[1], shots[1][:120])
    check("the directive line itself never reaches the model",
          "exact:" not in shots[0].lower(), shots[0][:220])
    check("the run says which shots carry one",
          "shot(s) 1 carry an exact: line" in str(out[2]), "")
    # Counted against the BEAT in the balance report, because it is the author's text.
    def _share(info, what):
        m = re.search(what + r" (?:is )?(\d+)%", str(info))
        return int(m.group(1)) if m else -1
    bare = run_node("A depot at night.\n\nMara walks Ana to the entrance.\n\nAna stops at the door.",
                    plan_only=True, character_memory=mem)
    check("...and it is counted as the author's words, not as a guard",
          _share(out[2], "the beat") > _share(bare[2], "the beat")
          and _share(out[2], "continuity clauses") < _share(bare[2], "continuity clauses"),
          f"with {_share(out[2], 'the beat')}% vs without {_share(bare[2], 'the beat')}%")
    read = run_node("A depot at night.\n\nAna waits.\n"
                    "exact: Mara's van is parked behind her with its side door open.\n",
                    plan_only=True, character_memory=mem)
    solo = _shots_of(read)[0]
    check("a name in it puts nobody in the shot",
          "Mara: she, 41" not in solo and "one person in the shot" in solo, solo[:200])
    check("...a door in it stages no change", "first frame and open by the last" not in solo, solo)
    check("...and it is still in the text, whole",
          "Mara's van is parked behind her with its side door open." in solo, solo[:200])
    mood = run_node("A depot at night.\n\nAna waits.\nexact: Her hands are cuffed behind her back.\n\n"
                    "Ana looks up.", plan_only=True, character_memory="Ana: she, 29, grey jacket.")
    check("...and it does not set the mood of the film",
          "mood is grim" not in mood[3], mood[3][-160:])
    floor, ratio = S.GUARD_FLOOR_WORDS, S.GUARD_WORDS_PER_BEAT_WORD
    try:
        S.GUARD_FLOOR_WORDS, S.GUARD_WORDS_PER_BEAT_WORD = 6, 1
        tight = run_node(P, plan_only=True, character_memory=mem)
    finally:
        S.GUARD_FLOOR_WORDS, S.GUARD_WORDS_PER_BEAT_WORD = floor, ratio
    check("a squeezed budget drops guards", "guard clauses dropped for room" in str(tight[2]))
    check("...and never the exact line",
          "Ana's wrists stay behind her back the whole way." in _shots_of(tight)[0],
          _shots_of(tight)[0][:200])
    for word in ("exact", "exactly", "verbatim"):
        check(f"{word}: is a directive", S.exact_lines(f"{word}: hold the frame") == ["hold the frame"])
    check("say: is not", S.exact_lines('say: "wait here"') == [])
    check("a sentence mid-beat is not a directive",
          S.exact_lines("Mara says: \"Wait here.\"") == [])


def test_a_walk_is_not_its_own_reverse():
    """REPORTED: somebody escorted from a vehicle to a doorway turned round, walked
    the other way, and then walked BACKWARDS to the doorway.

    The same reversal the door anchor settles -- time-flip augmentation teaches a
    model that a clip and its reverse are the same clip -- and the travel clause
    could not settle it: everything it said was as true of the reversed walk as of
    the real one. It named the two ends in SPACE and left the direction open."""
    print("\n=== a walk says which way it goes ===")
    mem = "Ana: she, 29, grey jacket.\nMara: she, 41, navy uniform."
    said = S.move_clause("office door", "Mara walks Ana to the office door.")
    check("a move to an unlisted place says which way the bodies face",
          "each body facing the way it goes" in said, said)
    check("...and that the bodies get nearer the destination",
          "nearer the office door at the last frame than at the first" in said, said)
    check("...and never that the destination gets nearer the lens",
          "travels to" not in said and "door nearer at" not in said, said)
    between = S.travel_anchor("garage", "", "hallway", "", "Mara walks Ana to the hallway.")
    check("a walk between two rooms says it too",
          "each body facing the way it goes" in between, between)
    arriving = S.travel_anchor("", "", "hallway", "", "Ana walks into the hallway.")
    check("...and so does an arrival with no origin",
          "each body facing the way it goes" in arriving, arriving)
    for beat in ("Ana backs away to the office door.", "Ana walks backwards to the door.",
                 "Ana retreats to the office door."):
        check(f"the author's direction wins: {beat[:30]!r}",
              "facing the way it goes" not in S.move_clause("office door", beat), beat)
    for beat, want in (("Mara escorts Ana to the entrance.", True),
                       ("Mara ushers Ana to the gate.", True),
                       ("Mara marches Ana to the depot entrance.", True),
                       ("Mara brings Ana a cup of tea.", False),
                       ("Mara hands Ana the keys.", False)):
        shot = _shots_of(run_node("A depot at night.\n\n" + beat, plan_only=True,
                                  character_memory=mem))[0]
        check(f"travel clause={want}: {beat[:34]!r}",
              ("The move to the" in shot) == want, shot[-140:])


def test_a_thing_that_opens_itself_is_a_staged_change():
    """REPORTED: a van pulls up, its side door slides open, somebody gets out -- and
    halfway through the shot the door is closed again.

    That reversal is what the two-ended anchor settles, and it never ran. The reader
    wanted the verb in FRONT of its noun ("opens the door"), while a door that opens
    on its own is written the other way round; and "slides" is in the list of verbs
    that go either way and earn no anchor on their own. The beat said which way in
    the word right after the verb, and nothing read it."""
    print("\n=== a door that opens itself is a change with two ends ===")
    for text, want in (("A van pulls up and the side door slides open.", [("door", "open")]),
                       ("The side door slides shut.", [("door", "shut")]),
                       ("The gate swings shut behind her.", [("gate", "shut")]),
                       ("The curtains draw back.", [("curtains", "open")]),
                       ("The door is opening.", [("door", "open")]),
                       ("Ana pulls the door open.", [("door", "open")])):
        check(f"{text[:38]!r}", S.state_changes(text) == want, str(S.state_changes(text)))
    for text in ("The window is open.", "Mara and Dom stand behind a van with its doors closed.",
                 "The curtains are still drawn.", "They stand by the closed doors of the van."):
        check(f"still a state: {text[:34]!r}", S.state_changes(text) == []
              and bool(S.stated_states(text)), str(S.state_changes(text)))
    # END TO END: the shot gets both ends of the change.
    shots = _shots_of(run_node(
        "A loading yard at dawn.\n\nA van pulls up and the side door slides open; Ana jumps out."
        "\n\nAna walks to the shutter.", plan_only=True, character_memory="Ana: she, 29, grey jacket."))
    check("the shot is told which end is which",
          "The door is shut at the first frame and open by the last." in shots[0], shots[0][-200:])
    check("...and the shot after it is not told the old state",
          "door is shut" not in shots[1], shots[1][-160:])


def test_a_cut_carries_the_people_across():
    """REPORTED: shots cutting to a new scene, breaking character continuity.

    A shot after a removal, and a shot opening in another room, dropped the previous
    frame and carried NO picture at all: room, faces, hair and clothes were all
    re-imagined from the text. The frame now rides as a claimed reference instead."""
    print("\n=== a cut carries the people across as a reference ===")
    mem = "Maya: she, 30, green sweater, grey jeans.\nOwen: he, 34, blue shirt."
    P = ("A living room.\n\nMaya and Owen sit on the couch.\n\nMaya takes off her sweater.\n\n"
         "Owen laughs.")
    rows = _encoded_refs(P, character_memory=mem)
    info = run_node(P, character_memory=mem)[2]
    check("the shot after a removal carries the frame as a reference",
          [n for _, n in rows] == [0, 0, 1], str([n for _, n in rows]))
    check("...claimed as the room, with everyone still in it",
          "<Picture 1> is this room a moment earlier" in rows[2][0]
          and "Maya and Owen are the people there" in rows[2][0], rows[2][0][-220:])
    check("...and the run says so", "shot 3 (something came off in the shot before)" in info)
    check("...and it is not a fresh start", "start fresh" not in info)

    P = "A house.\n\nMaya stands in the hallway.\n\nIn the bedroom, Maya sits on the bed."
    rows = _encoded_refs(P, character_memory=mem)
    check("a cut to another room carries her as a reference",
          [n for _, n in rows] == [0, 1], str([n for _, n in rows]))
    check("...claimed as the person, naming both rooms",
          "<Picture 1> is Maya a moment earlier, in the hallway" in rows[1][0]
          and "This shot is in the bedroom." in rows[1][0], rows[1][0][-200:])

    # Somebody left behind in the hallway is not in the bedroom to be claimed.
    rows = _encoded_refs("A house.\n\nMaya and Owen stand in the hallway.\n\n"
                         "In the bedroom, Maya sits on the bed.", character_memory=mem)
    check("...but not with somebody left behind in it", rows[1][1] == 0, str(rows[1][1]))

    # A face captured before a change of clothes is a picture of the old clothes.
    P = ("A kitchen.\n\nMaya reads.\n\nMaya takes off her sweater and walks out of the kitchen."
         "\n\nOwen cooks.\n\nMaya comes back in.")
    info = run_node(P, character_memory=mem)[2]
    check("a face from before a change of clothes is not recovered",
          "back after a shot away" in info and "recovered a face" not in info, info[-300:])


def test_somebody_still_in_the_frame_is_not_back():
    """REPORTED, still: a second Dan.

    Who a keyframe shows was read off the TEXT -- the previous shot's described cast.
    A reaction shot describing only Crystal does not take Dan out of the picture it
    opens on, so the next beat about Dan read as him "back after a shot away" and the
    node sent a recovered frame of him while the keyframe still had him sitting there."""
    print("\n=== somebody still in the frame is not back after a shot away ===")
    mem = ("Dan: he, 35, black t-shirt, blue jeans.\n"
           "Crystal: she, 35, white t-shirt, blue jeans.")

    def pictures(P):
        clip = FakeCLIP()
        out = run_node(P, character_memory=mem, clip=clip)
        return [len(items) for p, items in clip.seen if p.strip()], str(out[2]), out[3]

    refs, info, _ = pictures("A kitchen.\n\nDan pours coffee.\n\n"
                             "Crystal sits down opposite Dan.\n\nCrystal laughs.\n\nDan smiles.")
    check("an undescribed person is still in the keyframe, so no second picture of him",
          refs == [0, 1, 1, 1], str(refs))
    check("...and he is not reported as back", "back after a shot away" not in info)
    check("...and nothing is recovered", "recovered a face" not in info)
    shots = _shots("A kitchen.\n\nDan pours coffee.\n\nCrystal sits down opposite Dan.\n\n"
                   "Crystal laughs.", character_memory=mem)
    check("the reaction shot counts the one person it describes",
          "There is one person in the shot" in shots[2]
          and "two people" not in shots[2], shots[2][-160:])
    check("...and does not describe the other", "Dan:" not in shots[2], shots[2][:120])

    # Control: a real exit, and the return gets its picture.
    refs, info, _ = pictures("A kitchen.\n\nDan pours coffee.\n\nCrystal sits down opposite Dan.\n\n"
                             "Dan walks out of the kitchen.\n\nCrystal laughs.\n\n"
                             "Dan comes back in and smiles.")
    check("somebody who walked out and comes back is recovered",
          "recovered a face for Dan on shot 5, from shot 1" in info, info[-300:])
    check("...and the shot after he left counts one person",
          "There is one person in the shot" in _shots(
              "A kitchen.\n\nDan pours coffee.\n\nCrystal sits down opposite Dan.\n\n"
              "Dan walks out of the kitchen.\n\nCrystal laughs.", character_memory=mem)[3])

    # A face is captured from a frame with ONE PERSON IN IT, not one person described.
    refs, info, _ = pictures("A kitchen.\n\nDan and Crystal sit at the table.\n\nCrystal laughs.\n\n"
                             "Crystal walks out of the kitchen.\n\nDan reads the paper.\n\n"
                             "Crystal comes back in.")
    check("a frame that also shows Dan is never recovered as Crystal's face",
          "recovered a face" not in info and refs[-1] == 1, f"{refs} {info[-200:]}")

    # Staged walking in while the frame still has him: the shot starts fresh.
    refs, info, script = pictures("A kitchen.\n\nDan sits at the table.\n\nCrystal walks in.\n\n"
                                  "Dan walks in and pours coffee.")
    check("walking in while still in the frame starts fresh",
          "START FRESH where somebody still in the frame is staged walking in -- shot 3: Dan"
          in info, info[-300:])
    last = script.split("\n---\n")[-1]
    check("...with at most one picture of him", refs[-1] <= 1
          and len(re.findall(r"<Picture \d+>", last)) == refs[-1], f"{refs} {last[:160]}")
    for P, why in (("A hallway.\n\nDan stands at the front door.\n\nDan walks in.",
                    "somebody the shot before staged at the door"),
                   ("A house.\n\nDan walks down the hallway.\n\nDan walks into the kitchen.",
                    "a walk between rooms")):
        info = run_node(P, plan_only=True, character_memory=mem)[2]
        check(f"...but not {why}", "still in the frame is staged walking in" not in info)

    # Who leaves, read off the subject of the leaving.
    both = ["Dan", "Crystal"]
    for beat, want in (("Crystal hands Dan the keys and leaves.", ["Crystal"]),
                       ("Crystal hands Dan the keys and he leaves.", ["Dan"]),
                       ("Crystal watches Dan walk away.", ["Dan"]),
                       ("They leave together.", both),
                       ("Dan steps out of the shower.", []),
                       ("Dan leaves the cup on the table.", []),
                       ("Dan steps away from the window.", []),
                       ('Crystal says: "Dan, leave."', [])):
        check(f"leaves_in {beat!r}", S.leaves_in(beat, mem, both) == want,
              str(S.leaves_in(beat, mem, both)))
    for beat, want in (("Dan comes back in with two mugs.", ["Dan"]),
                       ("Dan walks over to the sink.", []),
                       ("Dan returns to his seat.", [])):
        check(f"comes_in {beat!r}", S.comes_in(beat, mem) == want, str(S.comes_in(beat, mem)))


def test_a_carried_room_is_not_a_second_picture_of_somebody():
    """REPORTED: a duplicate Mistress.

    The room-carry sends the previous shot's last frame as a REFERENCE, so a shot
    introducing somebody in position keeps the set instead of re-imagining it. If
    the person in that frame already has a portrait on the sheet, the shot gets
    two pictures of her -- her <Picture 1> and the carried frame -- and two
    pictures of one person is how a second one gets drawn. The recovered-frame
    path skips tagged people for exactly this reason; this was written without
    that skip."""
    print("\n=== a carried room is not a second picture of somebody ===")
    sent = []
    _bc = S.build_conditioning

    def spy(clip, vae, avae, prompt, w, h, length, **kw):
        sent.append((len(kw.get("refs") or []), bool(kw.get("handoff_as_ref"))))
        return _bc(clip, vae, avae, prompt, w, h, length, **kw)

    P = ("A panelled study.\n\nThe Mistress stands at the window.\n\n"
         "Ana is already kneeling by the desk, watching the Mistress.\n\n"
         "Ana lowers her eyes.")
    S.build_conditioning = spy
    try:
        run_node(P, character_memory=("Mistress: she, 38, a black dress. <Picture 1>\n"
                                      "Ana: she, 24, a shirt."),
                 ref_image_1=torch.rand(1, H, W, 3))
    finally:
        S.build_conditioning = _bc
    check("the shot that introduces somebody carries no second picture of her",
          not any(as_ref for _n, as_ref in sent), str(sent))
    check("...and her own portrait still goes in", sent[1][0] == 1, str(sent))
    sent2 = []
    S.build_conditioning = lambda c, v, a, p, w, h, l, **kw: (
        sent2.append(bool(kw.get("handoff_as_ref"))) or _bc(c, v, a, p, w, h, l, **kw))
    try:
        run_node(P, character_memory="Mistress: she, 38, a black dress.\nAna: she, 24.")
    finally:
        S.build_conditioning = _bc
    check("with no portrait in play the room is still carried", any(sent2), str(sent2))


def test_a_bare_region_is_said_on_every_shot():
    """REPORTED: a bra comes back on somebody topless, and the sheet never had one.

    Nothing restored it -- the model INVENTED it, because the clause saying the
    chest is uncovered was only ever on the beat that uncovered it. From the next
    shot that region was unspecified, and an unspecified region is filled from the
    model's own prior.

    End to end on purpose. The state half of this lives in the engine and is
    tested there; the clause is written here, and an engine test passes whether
    or not the shot ever says it."""
    print("\n=== a bare region is said on every shot ===")
    mem = "Kate: she, 24, a shirt, jeans.\nSam: he, 40."
    got = _shots("A bare room.\n\nKate is topless, sitting on the crate.\n\n"
                 "Kate looks at the door.\n\nKate listens.", character_memory=mem)
    for i, s in enumerate(got, 1):
        check(f"shot {i} says the chest is bare", "chest" in s and "bare" in s,
              " ".join(s.split())[-120:])
    check("...and it names the chest, not just the shoulders",
          all("chest, shoulders and arms are bare" in s for s in got))
    # Undressing reaches the same place, and the sheet must not re-dress her.
    off = _shots("A bare room.\n\nKate takes off her shirt.\n\n"
                 "Kate looks at the door.\n\nKate listens.", character_memory=mem)
    check("a removal is held the same way",
          all("chest" in s for s in off), " ".join(off[-1].split())[-120:])
    other = _shots("A bare room with a crate.\n\n"
                   "Kate takes off her shirt and drops it on the crate.\n\n"
                   "Sam watches from the doorway.\n\n"
                   "Kate turns towards him.", character_memory=mem)
    check("a beat naming only the other person still says it",
          "chest" in other[1], " ".join(other[1].split())[-140:])
    check("...and says WHOSE chest it is", "Kate's chest" in other[1],
          " ".join(other[1].split())[-140:])
    check("...and names her when both are described",
          "Kate's chest" in other[2], " ".join(other[2].split())[-140:])
    # ...and dressing again stops it, or she is told her chest is bare over a shirt.
    back = _shots("A bare room.\n\nKate takes off her shirt.\n\n"
                  "Kate puts on her shirt.\n\nKate listens.", character_memory=mem)
    check("dressing again stops the clause", "chest" not in back[-1],
          " ".join(back[-1].split())[-120:])


def test_a_squat_survives_speech_and_undressing():
    """REPORTED: she does not stay squatting, she stands up on her own.

    posture_cleared read the whole beat INCLUDING quoted speech, so a line with a
    travel word in it cleared the pose -- "Someone is coming" is about somebody
    else. And "takes off her shirt" matched the travel list on "takes", so
    undressing cleared it too, which is why the two reports arrived together."""
    print("\n=== a squat survives speech and undressing ===")
    mem = "Kate: she, 24, a shirt, jeans.\nSam: he, 40."
    got = _shots("A bare room.\n\nKate squats down beside the crate.\n\n"
                 "Kate says: \"Someone is coming.\"\n\n"
                 "Kate takes off her shirt.\n\nKate listens.", character_memory=mem)
    check("the author's word is kept, not swapped for a crouch",
          "is squatting" in got[1], " ".join(got[1].split())[-120:])
    check("a spoken travel word does not stand her up",
          "is squatting" in got[2], " ".join(got[2].split())[-120:])
    check("...nor does taking a garment off",
          "is squatting" in got[3], " ".join(got[3].split())[-120:])
    walk = _shots("A bare room.\n\nKate squats down beside the crate.\n\n"
                  "Kate walks to the door.\n\nKate listens.", character_memory=mem)
    check("walking away does clear it", "is squatting" not in walk[2],
          " ".join(walk[2].split())[-120:])


def test_the_audit_findings_stay_fixed():
    print("\n=== nine defects found by audit, each held down ===")
    mem3 = "Mara: she, 22.\nKate: she, 20.\nDan: he, 41."
    def sh(P, **kw):
        return [s for s in run_node(P, plan_only=True, **kw)[3].split("---") if s.strip()]

    got = sh("A room.\n\nDan cuffs Mara's wrists.\n\nDan cuffs Kate's wrists too.\n\n"
             "Kate sits alone.\n\nMara looks up.", character_memory=mem3)
    check("a second person cuffed later is held too",
          all("closed and fastened" in s or "hardware goes on" in s for s in got),
          str([("closed and fastened" in s) for s in got]))

    last = sh("A room.\n\nDan cuffs her wrists.\n\nDan gags her with duct tape.\n\n"
              "Mara sits still.",
              character_memory="Mara: she, 22.\nDan: he, 41.")[-1].lower()
    check("both items stay named", "cuff" in last and "tape" in last, last[-90:])
    check("...and cuffs with tape are still closed and fastened",
          "closed and fastened" in last, last[-90:])

    m = "Mara: she, 22, denim shorts, a black thong."
    got = sh("A room.\n\nMara stands.\n\nMara pulls her shorts down to show the thong.\n\n"
             "Mara waits.\n\nMara pulls them back up.\n\nMara stands.", character_memory=m)
    check("displacement uncovers what is under it",
          "thong" in got[2].lower(), got[2][-80:])
    check("...and putting it back covers it again",
          "whole, opaque and unbroken" in got[4].lower(), got[4][-130:])

    # 4. `gone` only ever grew, so an add: could not re-cover.
    got = sh("A room.\n\nMara pulls off her shorts.\n\nMara waits.\n\n"
             "add: her denim shorts are back on\nMara pulls her shorts on.\n\nMara stands.",
             character_memory=m)
    check("an add: re-covers the layer under it",
          "whole, opaque and unbroken" in got[3].lower(), got[3][-130:])

    # 5. Stockings stop at the thigh and cover no waistband.
    check("stockings do not cover a belt",
          not S.implied_layers("Mara: she, stockings, a chastity belt."))
    check("...while tights do",
          S.implied_layers("Mara: she, tights, a chastity belt.").get("chastity belt"))

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
    # CHANGED BY THE PERSISTENCE RULE. "Dan checks the cuffs" is a shot with the
    # cuffs in it, so the woman wearing them is in it too -- she was here last beat,
    # she is fastened, and nobody leaves. The old expectation was that naming one
    # person dropped the other, which is what lost her cuffs mid-scene.
    check("the shot that only names him still has her hardware",
          hw[3] == "yes", str(hw))
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
    solo = run_node('A living room.\n\nMara sits on the sofa. '
                    'The TV says: "Storms tonight."',
                    plan_only=True, character_memory=mem)[2]
    check("the shot is not silenced", "conditioned on real silence" not in solo, "")
    check("...and the line reaches the model as written",
          "Storms tonight." in sh[0] and "<d>Storms tonight.</d>" in sh[0], "")
    # Her own line is untouched: she is speaking, and her mouth must move.
    check("her own line is left alone", "the TV's" not in sh[1], "")
    check("...and her mouth is not held shut", "Mouths in the shot" not in sh[1], "")
    # Nobody in the beat, nothing about mouths -- ca75672 again.
    check("an empty room is told nothing about mouths", "Mouths in the shot" not in sh[2], "")
    check("info names the shot", "shot(s) 1 have a spoken line that belongs" in info, "")


def test_a_shifted_workflow_stops_before_rendering():
    print("\n=== a slid workflow fails with an explanation, not a bad render ===")
    try:
        run_node("A room.\n\nMara waits.", plan_only=True,
                 resolution=0.7, sampler_name="beta", scheduler=48,
                 shot_length=True)
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
    num = run_node("A room.\n\nMara waits.", plan_only=True, pace=float("nan"))
    check("a NaN number is still repaired, not refused",
          "not usable numbers" in num[2], "")


def test_the_removal_shot_says_what_is_under():
    print("\n=== taking the shorts off shows the panties, not skin ===")
    mem = "Mara: she, 22, blue denim shorts, white top, black panties."
    P = "A room.\n\nMara stands.\n\nMara pulls off her shorts.\n\nMara turns."
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    check("the removing shot is told what shows there",
          "what shows there now" in sh[1], sh[1][-90:])
    check("...naming the layer", "panties underneath" in sh[1].lower(), "")
    check("...and saying it stays on", "on and unchanged" in sh[1], "")
    check("said only on the shot that uncovers it",
          "what shows there now" not in sh[0] and "what shows there now" not in sh[2], "")
    check("info names the shot", "take off a garment that was covering" in info, "")
    bare = run_node("A room.\n\nMara pulls off her shorts.\n\nMara turns.",
                    plan_only=True,
                    character_memory="Mara: she, 22, blue denim shorts, white top.")[3]
    check("nothing underneath, nothing claimed", "what shows there now" not in bare, "")
    strip = run_node("A room.\n\nMara undresses completely.\n\nMara turns.",
                     plan_only=True, character_memory=mem)[3]
    check("a full strip promises nothing", "what shows there now" not in strip, "")


def test_a_beat_can_name_what_the_layering_hid():
    print("\n=== a beat naming a covered garment puts it back, and is reported ===")
    mem = "Mara: <Picture 1>, she, 22, blue jeans, a chastity belt <Picture 2>."
    P = ("A room.\n\nMara stands by the window.\n\n"
         "Mara runs a hand over the chastity belt under her jeans.\n\nMara waits.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    check("the covered shot does not name it",
          "chastity belt" not in sh[0].lower(), sh[0][:200])
    # The clause names the COVER. The belt is named once, by the sheet.
    check("...and the jeans are described as covering it",
          "jeans cover the hips and waist completely" in sh[0].lower(),
          sh[0][:240])
    check("...and nothing else in her entry is touched",
          "blue jeans" in sh[0].lower(), sh[0][:240])
    check("...and the jeans are described as covering it",
          "whole, opaque and unbroken" in sh[0].lower(), sh[0][:260])
    check("...and its picture waits for the cover to come off",
          "<Picture 2>" not in sh[0], sh[0][:220])
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
    check("the cover is the head noun, not the first word",
          S.implied_layers(sheet) == {"chastity belt": "shorts"},
          str(S.implied_layers(sheet)))
    check("...so taking the shorts off uncovers it",
          S.hidden_layers(S.implied_layers(sheet), ["shorts"]) == [])
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
    check("the picture waits while it is covered",
          rows[0][1] == 0 and rows[1][1] == 0, str([n for _, n in rows]))
    check("...and the words wait with it",
          not any("chastity" in p.lower() for p, _ in rows[:2]),
          rows[0][0][-160:])
    check("...with the cover described as opaque over it",
          all("whole, opaque and unbroken" in p.lower() for p, _ in rows[:2]),
          rows[0][0][-200:])
    check("...with the cover described as opaque over it",
          all("whole, opaque and unbroken" in p.lower() for p, _ in rows[:2]),
          rows[0][0][-200:])
    check("the picture arrives when the shorts come off", rows[2][1] == 1, "")
    check("...and stays after", rows[3][1] == 1, "")
    check("...described too", all("chastity" in p.lower() for p, _ in rows[2:]), "")


def test_an_object_tag_works_without_a_face_picture():
    print("\n=== a tagged object is placed even when nobody is tagged ===")
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
        check(f"{label}: picture waits while covered",
              "BELT" not in got[0] and "BELT" not in got[1], str(got))
        check(f"{label}: and arrives when uncovered", "BELT" in got[2], str(got))
    # A face picture alongside it still behaves, and still travels every shot.
    got = imgs("Mara: <Picture 1>, she, blue jeans, a chastity belt <Picture 2>.")
    check("with a face too, the face is always there",
          all("FACE" in g for g in got), str(got))
    check("...and the belt from the shot that uncovers it",
          "BELT" not in got[0] and "BELT" in got[2], str(got))


def test_an_untagged_picture_defeats_the_layering():
    print("\n=== an untagged reference is sent even where its garment is covered ===")
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
    check("while the shorts are on, the shorts are the outermost layer",
          "whole, opaque and unbroken" in sh[0], sh[0][-150:])
    check("...and the panties are not named",
          "panties" not in sh[0], sh[0][-150:])
    check("...nor the belt", "chastity belt" not in sh[0], sh[0][-150:])
    check("...while the shorts themselves are", "shorts" in sh[0], "")
    check("the reveal shot describes them", "panties" in sh[1] and "chastity belt" in sh[1], "")
    check("...and every shot after", "panties" in sh[2] and "chastity belt" in sh[2], "")
    check("info explains the layering", "read as layers" in info, "")
    # Nothing over it: underwear is on show and must not be described away.
    solo = run_node("A room.\n\nMara stands by the window.", plan_only=True,
                    character_memory="Mara: she, 22, panties and a bra.")[3]
    check("underwear alone is still described", "panties" in solo.lower(), "")


def test_a_garment_moved_is_not_a_garment_gone():
    print("\n=== shorts pulled down stay on, and stay described ===")
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
          all("On the body and" in s for s in sh[1:]), "")
    check("...and the staging shot is not told it twice",
          "On the body and" not in sh[0], "")
    check("info names the shots", "MOVED rather than taken off" in info, "")
    # Put back up: the latch lets go, by name or by pronoun.
    for _put in ("Mara pulls her shorts back up.", "Mara pulls them back up."):
        back = run_node("A room.\n\nMara pulls down her shorts.\n\nMara waits.\n\n"
                        + _put + "\n\nMara walks out.",
                        plan_only=True, character_memory=mem)[3]
        got = ["yes" if "On the body and" in s else "no"
               for s in back.split("---") if s.strip()]
        check(f"restored by {_put[:28]!r}", got == ["no", "yes", "no", "no"], str(got))
    # A real removal still empties the wardrobe and says so.
    off = run_node("A room.\n\nMara pulls off her shorts.\n\nMara waits.",
                   plan_only=True, character_memory=mem)[3]
    check("a real removal still removes",
          "away by the last frame" in off and "fully removed" in off, "")
    check("...and stops describing them", "shorts" not in off.split("---")[1].lower(), "")


def test_undressing_does_not_drop_her():
    print("\n=== taking a garment off does not put a body on the floor ===")
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
    check("...after the beat, not before it",
          sh[0].index("looking at the TV") < sh[0].index("The eyes and the head"), "")
    check("a pronoun target adds nothing", "The eyes and the head" not in sh[1], "")
    check("a beat with no look adds nothing", "The eyes and the head" not in sh[2], "")
    check("info names the shot", "shot(s) 1 name something to look at" in info, "")
    kept = run_node(P, plan_only=True, character_memory="Mara: she, 30.")[3]
    check("the beat is left exactly as written", "looking at the TV" in kept, "")


def test_a_line_with_no_look_turns_the_faces_to_each_other():
    print("\n=== a dialogue shot that names no look faces the speakers to each other ===")
    mem = "Dan: he, 40, a grey jacket.\nMara: she, 30, a red coat."
    P = ("A kitchen.\n\nDan and Mara stand at the counter. Dan asks: \"Is that the last one?\"\n\n"
         "Mara looks at the window and says: \"Nearly.\"\n\n"
         "Dan waits alone.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    check("two people and a line, no look staged: they face each other",
          "face each other" in sh[0], sh[0][-90:])
    check("...said positively, naming no camera", "camera" not in sh[0].lower(), "")
    check("a look the beat stages wins, and this stands down",
          "turned to the window" in sh[1] and "face each other" not in sh[1], sh[1][-90:])
    check("no line, nothing to face", "face each other" not in sh[2], "")
    check("info names the shot", "shot(s) 1 carry a line and two or more people" in info, "")


def test_a_walk_along_a_place_arrives_in_it():
    print("\n=== a beat that travels along a place ends in it ===")
    mem = "McKenna: she, 22, a grey t-shirt, black shorts."
    P = ("A small flat at night. Her bedroom has an unmade bed and a lamp.\n\n"
         "McKenna gets up and comes out of her bedroom.\n\n"
         "McKenna walks down the hallway.\n\n"
         "McKenna goes into the kitchen.")
    script = run_node(P, plan_only=True, character_memory=mem)[3]
    sh = [" ".join(x.split()) for x in script.split("---") if x.strip()]
    check("the hallway shot is told it opens where she left off and arrives in the hallway",
          "opens in the bedroom and arrives in the hallway" in sh[1], sh[1][-160:])
    check("...with the walk played out in frame", "every step in frame" in sh[1], "")
    check("the kitchen shot opens in the HALLWAY, not the bedroom",
          "opens in the hallway and arrives in the kitchen" in sh[2], sh[2][-160:])
    check("...and is never sent back to the bedroom", "opens in the bedroom" not in sh[2], "")
    # A via WITH a destination is untouched: the destination still wins.
    both = run_node("A flat.\n\nMcKenna walks down the hallway to the kitchen.",
                    plan_only=True, character_memory=mem)[3]
    check("a via with a destination keeps both",
          "carries along the hallway" in both and "arrives in the kitchen" in both, both[-160:])


def test_a_rooms_description_waits_outside_that_room():
    print("\n=== the scene's description of a room waits outside that room ===")
    SCENE = "A small flat at night. Her bedroom has an unmade bed and a lamp."
    out, held, blocked, waited = S.scene_for_here(SCENE, "kitchen")
    check("another room's description is held", "unmade bed" not in out, out)
    check("...the film's own framing is kept", "small flat at night" in out, out)
    check("...the room is named for the report", held == ["bedroom"] and not blocked, str(held))
    check("...and the sentence that waited is quoted back",
          waited == ["Her bedroom has an unmade bed and a lamp."], str(waited))
    # A ROOM DESCRIBED WITH A COLON is the author describing a room, not a person.
    colon = S.scene_for_here("A small flat at night. Her bedroom: an unmade bed and a lamp.",
                             "kitchen", "", ["McKenna"])
    check("a room written with a colon is still held",
          "unmade bed" not in colon[0] and "at night" in colon[0], colon[0])
    # ...while a declared name with a colon is protected even naming another room.
    keepn = S.scene_for_here("McKenna: she, 22, asleep in the bedroom.", "kitchen", "", ["McKenna"])
    check("a declared name's line is protected", "McKenna: she, 22" in keepn[0], keepn[0])
    # A ROOM THE BEAT NAMES IS NEVER HELD -- `here` goes stale on an unlisted verb.
    stale = S.scene_for_here("Her bedroom has an unmade bed. The kitchen is small and white.",
                             "bedroom", "", ["McKenna"], "McKenna pads into the kitchen.")
    check("a room the beat names survives a stale tracked room",
          "small and white" in stale[0], stale[0])
    same, held2, _, _ = S.scene_for_here(SCENE, "bedroom")
    check("the room we are IN keeps its description", "unmade bed" in same, same)
    check("...and reports nothing held", held2 == [], str(held2))
    for txt, room in (("A kitchen. She is at the sink.", "kitchen"),
                      ("Shot on 35mm, shallow depth of field.", "bedroom"),
                      ("Two people, late evening.", "kitchen")):
        check(f"unchanged: {txt[:28]!r}", S.scene_for_here(txt, room)[0] == txt)
    check("no room known, nothing held", S.scene_for_here(SCENE, "")[1] == [])
    # A SHEET LINE IS NEVER TOUCHED, whatever it names.
    sheet = S.scene_for_here("A kitchen.\nMcKenna: she, 22, a grey t-shirt.", "bedroom")[0]
    check("a sheet entry survives a room it does not match",
          "McKenna: she, 22, a grey t-shirt." in sheet, sheet)
    # Two rooms named: each is kept only in its own.
    two = S.scene_for_here("Her bedroom has an unmade bed. The kitchen is small and white.",
                           "kitchen")[0]
    check("the other room goes and this one stays",
          "unmade bed" not in two and "small and white" in two, two)
    weld, weld_held, weld_blocked, _ = S.scene_for_here(
        "A small flat at night, her bedroom with an unmade bed.", "kitchen")
    check("a welded sentence is kept rather than losing the film's framing",
          "at night" in weld and weld_blocked and weld_held == ["bedroom"], weld)
    ANCH = "Shot on 35mm in a cramped kitchen"
    a_in = S.scene_for_here(ANCH + ".\nHer bedroom has an unmade bed.", "bedroom", ANCH)
    check("an anchor naming another room is kept", "35mm" in a_in[0] and a_in[1] == [], a_in[0])
    a_out = S.scene_for_here(ANCH + ".\nHer bedroom has an unmade bed.", "kitchen", ANCH)
    check("...and the bedroom still goes while it stays",
          "35mm" in a_out[0] and "unmade bed" not in a_out[0], a_out[0])

    # END TO END, the reported script, including the return.
    mem = "McKenna: she, 22, a grey t-shirt, black shorts."
    P = ("A small flat at night. Her bedroom has an unmade bed and a lamp.\n\n"
         "McKenna gets up and comes out of her bedroom.\n\n"
         "McKenna walks down the hallway to the living room.\n\n"
         "McKenna goes from the living room to the kitchen.\n\n"
         "McKenna goes back into her bedroom and lies down.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [" ".join(x.split()) for x in script.split("---") if x.strip()]
    check("shot 1 is in the bedroom and keeps the bed", "unmade bed" in sh[0], sh[0][:140])
    check("the shot ARRIVING in the living room has no bed", "unmade bed" not in sh[1], sh[1][:200])
    check("...and the living-room-to-kitchen shot has no bed", "unmade bed" not in sh[2], sh[2][:200])
    check("...and the bed comes back when she walks back in", "unmade bed" in sh[3], sh[3][:140])
    check("the film's framing is on every shot",
          all("small flat at night" in x for x in sh), "")
    check("nobody loses their sheet line", all("McKenna: she, 22" in x for x in sh), "")
    check("info reports which shots it waited on",
          "WAITS OUTSIDE it, on shot(s) 2, 3" in info, "")
    # A one-room script is untouched and says nothing about waiting.
    info1 = run_node("A kitchen.\n\nMcKenna fills the kettle.\n\nMcKenna opens a cupboard.",
                     plan_only=True, character_memory=mem)[2]
    check("a script that never leaves one room reports nothing",
          "WAITS OUTSIDE" not in info1, "")

    FLAT = "A small apartment. The living room has a red sofa; the kitchen has white tiles."
    lounge = S.scene_for_here(FLAT, "living room", "", ["Maya"], "")
    kitchen = S.scene_for_here(FLAT, "kitchen", "", ["Maya"], "")
    check("the living-room shot keeps the sofa and not the tiles",
          "red sofa" in lounge[0] and "white tiles" not in lounge[0], lounge[0])
    check("...and says which room waited", lounge[1] == ["kitchen"], str(lounge[1]))
    check("the kitchen shot keeps the tiles and not the sofa",
          "white tiles" in kitchen[0] and "red sofa" not in kitchen[0], kitchen[0])
    check("the film's own framing survives both", all(
        "A small apartment." in sent for sent in (lounge[0], kitchen[0])), "")
    check("...and a clause promoted out of a semicolon opens its sentence",
          "The kitchen has white tiles." in kitchen[0], kitchen[0])
    # A line nothing was held from is the author's, punctuation included.
    plain = S.scene_for_here("A cabin. Snow outside; a fire burning.", "kitchen", "", ["Maya"], "")
    check("a line that held nothing keeps its semicolons",
          plain[0] == "A cabin. Snow outside; a fire burning.", plain[0])
    # End to end: neither shot carries the other room.
    shots = _shots_of(run_node(
        FLAT + "\n\nMaya reads on the red sofa in the living room.\n\n"
        "Maya fills the kettle in the kitchen.", plan_only=True,
        character_memory="Maya: she, 30, green sweater."))
    check("no shot carries the room it is not in",
          "white tiles" not in shots[0] and "red sofa" not in shots[1],
          " | ".join(s[:80] for s in shots))


def test_a_covered_object_does_not_send_its_picture():
    print("\n=== an object out of view does not carry its reference ===")
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
        rows.append((prompt, [which(r) for r in (k.get("refs") or [])],
                     k.get("handoff") is not None))
        return ob(clip, vae, audio_vae, prompt, *a, **k)
    S.build_conditioning = spy
    try:
        run_node(P, character_memory=mem, ref_image_1=FACE, ref_image_2=OBJ)
    finally:
        S.build_conditioning = ob
    # A keyframe beside the references is named after them, as the opening frame.
    kf = lambda r: 1 if (r[2] and r[1]) else 0
    check("the covered shot sends only the face", rows[0][1] == ["FACE"], str(rows[0][1]))
    check("...and names no picture it does not carry",
          sorted(set(re.findall(r"<Picture (\d+)>", rows[0][0]))) == ["1"], rows[0][0][:80])
    check("...and does not describe the covered object",
          "locket" not in rows[0][0].lower(), "")
    # Back in view: the words and the picture return together.
    check("the reveal brings the picture back", rows[1][1] == ["FACE", "OBJ"], str(rows[1][1]))
    check("...claimed by the text",
          sorted(set(re.findall(r"<Picture (\d+)>", rows[1][0])))
          == [str(n) for n in range(1, 3 + kf(rows[1]))], "")
    check("...and it stays for the shot after", rows[2][1] == ["FACE", "OBJ"], str(rows[2][1]))
    # Every shot: as many pictures as the text names. The invariant this broke.
    bad = [i + 1 for i, r in enumerate(rows)
           if len(r[1]) + kf(r) != len(set(re.findall(r"<Picture (\d+)>", r[0])))]
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
    check("the next shot is told where the arms are",
          "wrists together above the head" in sh[1], sh[1][-120:])
    check("...and what they are fastened to",
          "fast at the bed frame" in sh[1], sh[1][-120:])
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


def test_a_grin_is_not_a_closed_mouth():
    print("\n=== the guard stops countermanding the beat ===")
    MEM = "Dana: she, 34, red coat."

    def held(beat):
        sc = run_node("A kitchen.\n\n" + beat, plan_only=True,
                      character_memory=MEM)[3]
        return "Mouths in the shot stay closed" in sc

    for _b in ("Dana smiles at him.",
               "Dana grins, wide and mean.",
               "Dana bites her lip.",
               "Dana yawns.",
               "Dana's mouth falls open.",
               "Dana sneers at the badge.",
               "Dana licks her lips.",
               "Dana mouths the words."):
        check(f"the beat keeps its mouth: {_b!r}", not held(_b), "")
    for _b in ("Dana stares at the door.",
               "Dana walks to the window.",
               "Dana finds her sister's body."):
        check(f"a shut mouth is still held: {_b!r}", held(_b), "")

    for _b in ("The dog bites the postman.", "She grinds the coffee.",
               "He licks the envelope.", "The engine spits and dies.",
               "She mouths off at the guard.", "The jaw of the vice opens."):
        check(f"ordinary English is quiet: {_b!r}", not S.mouth_performs(_b), "")
    check("...and a person spitting is not", S.mouth_performs("He spits on the floor."))

    info = run_node("A kitchen.\n\nDana grins, wide and mean.", plan_only=True,
                    character_memory=MEM, auto_sound=False)[2]
    check("...and a silent expression opens no audio branch",
          re.search(r"shot\(s\) [^|]*\b1\b[^|]*conditioned on real silence", info)
          is not None, info[:300])
    check("info says which shots kept their mouths",
          "the beat itself puts the mouth to work" in info, info[:300])

    for _c in (S.MOUTH_HOLD, S.ONE_VOICE):
        check(f"no stillness ordered: {_c[:34]!r}",
              not re.search(r"\bstill\b|\bmotionless\b|\bfrozen\b", _c, re.I), _c)
        check("...and the mouth is still shut", "clos" in _c)
    check("the face is given something to do", "expressions moving" in S.MOUTH_HOLD)


def test_nothing_tells_the_cast_to_hold_still():
    print("\n=== the guards stopped ordering stillness ===")
    STILL = re.compile(r"\bstill\b|\bmotionless\b|\bfrozen\b|\bunmoving\b|"
                       r"\brigid\b|\bstatic\b", re.I)
    _tv = 'The TV says: "x"'
    built = [
        ("told_hold", S.told_hold(["Mara"])),
        ("told_hold, two", S.told_hold(["Mara", "Dan"])),
        ("posture_hold", S.posture_hold({"Mara": "lying down"}, ["Mara"])),
        ("posture_hold, two", S.posture_hold(
            {"Mara": "lying down", "Dan": "kneeling"}, ["Mara", "Dan"])),
        ("device_voice_clause", S.device_voice_clause(_tv)),
        ("displaced_hold", S.displaced_hold([("shorts", "pulled down")])),
        ("pace_clause", S.pace_clause(4.0, 12.0)),
        ("reveal_clause", S.reveal_clause(["thong"])),
        ("off_by_last_frame", S.off_by_last_frame(["shorts"], "McKenna", "A van.", "")),
    ]
    for _n, _c in built:
        check(f"no stillness ordered by {_n}", not STILL.search(_c), _c)
    for _n in sorted(dir(S)):
        _v = getattr(S, _n, None)
        if (isinstance(_v, str) and _n.isupper() and len(_v.split()) >= 4
                and not re.search(r"\(\?|\\b|\|", _v)):
            check(f"no stillness ordered by {_n}", not STILL.search(_v), _v)

    check("the listener is still given something to do",
          "listens" in S.told_hold(["Mara"]))
    check("...and is still held to the sheet's wardrobe",
          "sheet already lists" in S.told_hold(["Mara"]))
    check("the latched pose is still named",
          "lying down" in S.posture_hold({"Mara": "lying down"}, ["Mara"]))
    check("the machine still owns the voice", "TV's" in S.device_voice_clause(_tv))
    check("...and the room's own mouths are still closed",
          "mouths closed" in S.device_voice_clause(_tv))
    check("...and it is still positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", S.device_voice_clause(_tv), re.I))
    check("one voice is still asserted",
          "Only the person speaking" in S.ONE_VOICE and "closed" in S.ONE_VOICE)
    check("the moved garment is still on the body",
          "On the body" in S.displaced_hold([("shorts", "pulled down")]))
    check("...and still where the beat left it",
          "where the beat put them" in S.displaced_hold([("shorts", "pulled down")]))
    check("the layer underneath is still unchanged",
          "unchanged" in S.reveal_clause(["thong"]))
    check("the rest of the wardrobe is still bound",
          "untouched" in S.off_by_last_frame(["shorts"], "McKenna", "A van.", ""))
    check("the beat still runs to the last frame",
          "last" in S.pace_clause(4.0, 12.0))


def test_a_face_under_duress_is_not_a_portrait():
    print("\n=== the face stops being left to the model's prior ===")
    MEM = "McKenna: she, 26, dark hair, a red jacket, handcuffs behind her back."
    PLAIN = "Kate: she, 30, a grey coat."

    def face(beat, mem=MEM, **kw):
        return run_node("A van interior, night.\n\n" + beat, plan_only=True,
                        character_memory=mem, **kw)[3]

    check("a restrained shot says what the face is doing",
          "shows the strain" in face("McKenna lies against the wheel arch."), "")
    # A beat with UNAMBIGUOUS duress in it does too, with no sheet hardware at all.
    check("...and so does a beat staging unambiguous duress",
          "shows the strain" in face("Kate is bound and gagged on the floor.",
                                     mem=PLAIN), "")
    check("...while one ambiguous beat alone does not",
          "shows the strain" not in face("Kate struggles and twists away.",
                                         mem=PLAIN), "")
    check("...unless the anchor says what the film is",
          "shows the strain" in face("Kate struggles and twists away.", mem=PLAIN,
                                     anchor="Handheld. A tense, grim abduction."), "")
    check("an ordinary scene is left alone",
          "shows the strain" not in face("Kate makes the coffee.", mem=PLAIN), "")
    check("...and so is an ordinary beat about a calm person",
          "shows the strain" not in face("Kate reads the letter.", mem=PLAIN), "")
    check("a collar alone does not stage duress",
          "shows the strain" not in face("McKenna sits on the bed.",
                                         mem="McKenna: she, 26, a leather collar."), "")
    check("...but cuffs do",
          "shows the strain" in face("McKenna sits on the bed.",
                                     mem="McKenna: she, 26, cuffs on her wrists."), "")
    check("...and so does rope",
          "shows the strain" in face("McKenna sits on the bed.",
                                     mem="McKenna: she, 26, rope around her wrists."), "")
    check("a written smile is not argued with",
          "shows the strain" not in face("McKenna smiles at him."), "")

    two = run_node("A van interior, night.\n\nMcKenna pulls at the cuffs while Dan "
                   "watches.", plan_only=True,
                   character_memory=MEM + "\nDan: he, 40, a work coat.")[3]
    check("two in the shot, and the clause still names nobody",
          "the face shows the strain" in two and "McKenna's face" not in two,
          two[-200:])

    info = run_node("A van interior, night.\n\nMcKenna lies against the wheel arch.",
                    plan_only=True, character_memory=MEM, auto_sound=False)[2]
    check("...and it opens no audio branch",
          "conditioned on real silence" in info, info[:200])
    check("info names the shots it spoke for",
          "left to the model's prior" in info, info[:200])

    cl = S.DURESS_FACE
    check("the clause is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b|\bun\w+ing\b", cl, re.I), cl)
    check("...and orders no stillness",
          not re.search(r"\bstill\b|\bmotionless\b|\bfrozen\b", cl, re.I), cl)
    check("...and names no camera", not re.search(r"camera|lens", cl, re.I), cl)
    check("...in one sentence", cl.count(".") == 1, cl)


def test_her_whimper_does_not_free_his_mouth():
    print("\n=== a vocal belongs to somebody ===")
    MEM = ("McKenna: she, 26, dark hair, handcuffs behind her back.\n"
           "Dan: he, 40, a work coat.")

    def shot(beat):
        return " ".join(run_node("A van interior, night.\n\n" + beat, plan_only=True,
                                 character_memory=MEM)[3].split())

    a = shot("McKenna sobs while Dan watches.")
    check("the vocal is given an owner", "The sobbing is McKenna's" in a, a[-200:])
    check("...and the other mouth is closed",
          "every other mouth in the shot stays closed" in a, a[-200:])

    b = shot('McKenna whimpers, and Dan says: "Nearly there."')
    check("the line is localized to the speaker", "Only Dan speaks" in b, b[-220:])
    check("...and the vocal to the other character",
          "the whimpering is McKenna's" in b, b[-220:])
    check("...and neither of their mouths is held shut",
          "Mouths in the shot stay closed" not in b, b[-220:])

    c = shot("McKenna whimpers behind the gag.")
    check("a lone vocaliser is not told to close her mouth",
          "Mouths in the shot stay closed" not in c, c[-200:])
    check("...and with nobody else in the beat, nothing is claimed",
          "every other mouth" not in c, c[-200:])

    d = shot("Somebody sobs in the dark while Dan waits.")
    check("an unattributed vocal holds nobody",
          "every other mouth in the shot stays closed" not in d, d[-200:])

    info = run_node("A van interior, night.\n\nMcKenna sobs while Dan watches.",
                    plan_only=True, character_memory=MEM, auto_sound=False)[2]
    check("the vocal still keeps its audio",
          re.search(r"shot\(s\) [^|]*\b1\b[^|]*stage EFFORT", info) is not None,
          info[:300])
    check("info names the shots whose vocal was attributed",
          "belongs to somebody" in info, info[:300])

    cl = S.voice_sources(["Dan"], "whimpering", ["McKenna"], ["Sam"])
    check("the clause is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", cl, re.I), cl)
    check("...and orders no stillness",
          not re.search(r"\bstill\b|\bmotionless\b|\bfrozen\b", cl, re.I), cl)


def test_her_look_does_not_land_on_him():
    print("\n=== a held look belongs to whoever is doing the looking ===")
    MEM = ("McKenna: she, 26, dark hair, handcuffs behind her back.\n"
           "Dan: he, 40, a work coat.")
    P = ("A white van parked facing away down the lane, rear doors open, night.\n\n"
         "Dan lifts McKenna into the back of the van.\n\n"
         "McKenna looks at the lane behind them.\n\n"
         "Dan closes one of the rear doors.")
    sh = [" ".join(x.split()) for x in
          re.split(r"(?=\[Shot )", run_node(P, plan_only=True,
                                            character_memory=MEM)[3]) if x.strip()]
    check("the beat that stages the look gets the clause",
          "turned to the lane" in sh[1], sh[1][-160:])
    check("...and the shot she is not in never claims it impersonally",
          "The eyes and the head are turned to the lane" not in sh[2], sh[2][-160:])

    P2 = ("A lane, night.\n\n"
          "McKenna looks at the treeline.\n\n"
          "Dan checks the wheel while McKenna waits.")
    sh2 = [" ".join(x.split()) for x in
           re.split(r"(?=\[Shot )", run_node(P2, plan_only=True,
                                             character_memory=MEM)[3]) if x.strip()]
    check("the held look is attributed once two people are in the shot",
          ("McKenna's eyes and head are turned to the treeline" in sh2[1]
           or "Her eyes and head are turned to the treeline" in sh2[1]), sh2[1][-200:])

    # The clause itself, both ways.
    check("impersonal with nobody named",
          S.gaze_hold("TV") == " The eyes and the head are turned to the TV.")
    check("...and named when it has to be",
          S.gaze_hold("TV", "Mara's") == " Mara's eyes and head are turned to the TV.")
    check("...or by pronoun where the name is already spent in this shot",
          S.gaze_hold("TV", "her") == " Her eyes and head are turned to the TV.")
    check("...and a person target takes no article",
          S.gaze_hold("Dan", "", True) == " The eyes and the head are turned to Dan.")
    check("nothing named, nothing said", S.gaze_hold("", "Mara's") == "")
    check("info says the look was attributed",
          "a look belongs to whoever" in run_node(P, plan_only=True,
                                                  character_memory=MEM)[2])


def test_an_anchor_says_when_it_has_taken_the_scenes_place():
    print("\n=== the scene paragraph that quietly became a shot ===")
    P = ("A white van parked facing away down the lane, rear doors open, night.\n\n"
         "Dan lifts her into the back.\n\n"
         "Dan closes one of the rear doors.")
    MEM = "Dan: he, 40, a work coat."
    info = run_node(P, plan_only=True, character_memory=MEM,
                    anchor="Handheld, night exterior.")[2]
    check("the node says the scene paragraph became a shot",
          "is being spent as shot 1" in info, info[:300])
    check("...and quotes it back so it can be recognised",
          "A white van parked facing away down the lane" in info, info[:300])
    check("...and says what that costs",
          "not carried into any other shot" in info, info[:300])

    P2 = ("Dan opens the rear doors.\n\nDan lifts her into the back.")
    q = run_node(P2, plan_only=True, character_memory=MEM,
                 anchor="Handheld, night exterior.")[2]
    check("an opening ACTION is not reported as lost scene text",
          "is being spent as shot 1" not in q, q[:300])
    # ...and never without an anchor, where the first paragraph IS the scene.
    r = run_node(P, plan_only=True, character_memory=MEM)[2]
    check("with no anchor there is nothing to report",
          "is being spent as shot 1" not in r, r[:300])
    check("...because the scene is carried instead",
          all("facing away down the lane" in b for b in
              [x for x in re.split(r"(?=\[Shot )",
                                   run_node(P, plan_only=True,
                                            character_memory=MEM)[3]) if x.strip()]))


def test_a_grim_film_is_grim_in_every_shot():
    print("\n=== the scene stops reading as a happy one ===")
    MEM = ("McKenna: she, 26, dark hair, handcuffs behind her back.\n"
           "Dan: he, 40, a work coat.")
    P = ("A van interior, night.\n\nMcKenna pulls at the cuffs.\n\n"
         "Dan watches her.\n\nDan closes the hatch.\n\nDan starts the engine.")
    sh = [" ".join(x.split()) for x in
          re.split(r"(?=\[Shot )", run_node(P, plan_only=True, character_memory=MEM,
                                            anchor="Handheld, tight interior.")[3])
          if x.strip()]
    for _i in (1, 2, 3, 4):
        check(f"shot {_i + 1} carries the film's mood",
              "The mood is grim" in sh[_i], sh[_i][-140:])
    check("the shot with the restrained person still gets her face",
          "shows the strain" in sh[1], sh[1][-140:])
    # She is cuffed in the back of the van and nobody leaves, so the persistence
    # rule keeps her in his shots -- and a shot with her in it gets her face.
    check("...and the captor's shot carries her face too, because she is in it",
          "The mood is grim" in sh[3] and "shows the strain" in sh[3], sh[3][-140:])

    PLAIN = "Kate: she, 30, a grey coat.\nSam: he, 33."
    Q = ("A kitchen, morning.\n\nKate makes the coffee.\n\n"
         "Sam reads the paper.\n\nKate sits down.")
    qs = [" ".join(x.split()) for x in
          re.split(r"(?=\[Shot )", run_node(Q, plan_only=True, character_memory=PLAIN,
                                            anchor="Wide, morning light.")[3])
          if x.strip()]
    check("an ordinary film gets no mood anywhere",
          not any("mood is grim" in x for x in qs), qs[0][-140:])

    check("the mood is read from the sheet's hardware",
          S.film_stages_duress(["Dan closes the hatch."],
                               "McKenna: she, 26, cuffs on her wrists."))
    check("...or from unambiguous duress in any beat",
          S.film_stages_duress(["Kate makes the coffee.",
                                "Kate is held captive in the cellar."], PLAIN))
    check("...but never from an ambiguous verb alone",
          not S.film_stages_duress(["Kate makes the coffee.", "Kate sobs."], PLAIN))
    # THE ANCHOR SETTLES IT, both ways, and outranks everything.
    check("a declared tone turns it on",
          S.film_stages_duress(["Kate makes the coffee."], PLAIN,
                               "Wide. A grim, claustrophobic film."))
    check("...and a declared light tone turns it off outright",
          not S.film_stages_duress(["Kate is bound and gagged in the cellar."], PLAIN,
                                   "Wide. A warm, comic short."))
    check("mood_declared reads both directions",
          S.mood_declared("A tense kidnapping, handheld") == "grim"
          and S.mood_declared("Warm, romantic, golden hour") == "light"
          and S.mood_declared("Handheld, 35mm, night") == "")
    check("...and a collar alone is still not duress",
          not S.film_stages_duress(["Kate makes the coffee."],
                                   "Kate: she, 30, a leather collar."))
    check("...and an ordinary film is not", not S.film_stages_duress(
        ["Kate makes the coffee.", "Sam reads the paper."], PLAIN))

    check("the mood clause is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", S.DURESS_MOOD, re.I), S.DURESS_MOOD)
    check("...and orders no stillness",
          not re.search(r"\bstill\b|\bmotionless\b|\bfrozen\b", S.DURESS_MOOD, re.I))
    check("the mood clause rides with the face it belongs to",
          "mood is grim" in run_node(P, plan_only=True, character_memory=MEM)[3])


def test_a_look_survives_the_next_beat():
    print("\n=== the look she was given does not evaporate ===")
    MEM = ("McKenna: she, 26, dark hair, handcuffs behind her back.\n"
           "Dan: he, 40, a work coat.")

    def two(second):
        sh = [" ".join(x.split()) for x in
              re.split(r"(?=\[Shot )",
                       run_node("A lane at night.\n\nMcKenna looks at the van.\n\n" + second,
                                plan_only=True, character_memory=MEM,
                                anchor="Handheld, night exterior.")[3]) if x.strip()]
        return sh[1], sh[2]

    _, b = two("McKenna watches him.")
    check("a look at a person is said, not dropped",
          "turned to Dan" in b, b[-200:])
    _, b2 = two("McKenna looks at Dan.")
    check("...and a look at a NAME is a look too", "turned to Dan" in b2, b2[-200:])
    THREE = MEM + "\nSam: he, 35, a hood."
    amb = " ".join(run_node("A lane at night.\n\nMcKenna watches him while Dan and "
                            "Sam wait.", plan_only=True, character_memory=THREE,
                            anchor="Handheld, night exterior.")[3].split())
    check("an ambiguous pronoun resolves to nobody",
          "turned to Dan" not in amb and "turned to Sam" not in amb, amb[-200:])

    for _second in ("Dan opens the driver's door.",
                    "Dan lifts the case into the back."):
        _, c = two(_second)
        # Named while she was only carried, pronoun now that she is described --
        # she is fastened, so the persistence rule keeps her in the shot. The fact
        # carried is the same fact.
        check(f"her look is carried past {_second[:22]!r}",
              ("McKenna's eyes and head are turned to the van" in c
               or "Her eyes and head are turned to the van" in c), c[-200:])
    sh = [" ".join(x.split()) for x in
          re.split(r"(?=\[Shot )",
                   run_node("A lane at night.\n\nMcKenna looks at the van.\n\n"
                            "Dan opens the driver's door.\n\nDan starts the engine.",
                            plan_only=True, character_memory=MEM,
                            anchor="Handheld, night exterior.")[3]) if x.strip()]
    check("...and is not still being claimed two beats later",
          "McKenna's eyes" not in sh[3], sh[3][-200:])
    # And a beat that walks her off ends it, as before.
    _, d = two("McKenna walks away down the lane.")
    check("walking her off still ends the look",
          "turned to the van" not in d, d[-200:])


def test_the_bed_survives_an_anchor():
    print("\n=== the room tone that was supposed to fill the lead-in ===")
    MEM = "McKenna: she, 26, dark hair.\nDan: he, 40, a work coat."
    P = ("A van interior, night. The engine is running.\n\n"
         "Dan looks back and says: \"Sit still.\"\n\n"
         "McKenna says: \"Where are we going?\"")

    def beds(**kw):
        sh = [" ".join(x.split()) for x in
              re.split(r"(?=\[Shot )", run_node(P, plan_only=True,
                                                character_memory=MEM, **kw)[3])
              if x.strip()]
        return [("sounds like" in x or "only sound" in x) for x in sh]

    withq = beds(anchor="Handheld, tight interior.")
    check("the speaking shots keep their bed with an anchor set",
          all(withq[1:]), str(withq))
    plain = beds()
    check("...exactly as they do without one", all(plain), str(plain))
    check("the bed itself is read from the opening beat",
          S.scene_ambient("Handheld, tight interior.") == ""
          and S.scene_ambient("Handheld, tight interior.",
                              "A van interior, night. The engine is running.")
          == "an engine idling")
    Q = ("A van interior, night. The engine is running.\n\nMcKenna waits.")
    q = [" ".join(x.split()) for x in
         re.split(r"(?=\[Shot )", run_node(Q, plan_only=True, character_memory=MEM,
                                           anchor="Handheld, tight interior.")[3])
         if x.strip()]
    check("a wordless shot still gets no bed and stays silent",
          not any("sounds like" in x for x in q), q[-1][-150:])
    check("info says where the bed was read from",
          "read from the opening beat" in run_node(P, plan_only=True,
                                                   character_memory=MEM,
                                                   anchor="Handheld, tight interior.")[2])


def test_the_audio_branch_gets_a_soft_landing():
    print("\n=== the audio branch stops landing from a great height ===")
    f = S.audio_sigma_of
    g = S.video_sigma_for_audio
    for want in (0.20, 0.10, 0.05, 0.03, 0.01):
        got = f(g(want, 12.0, 3.0), 12.0, 3.0)
        check(f"audio {want} round-trips through the video grid",
              abs(got - want) < 1e-9, f"{got}")
    check("the formula matches comfy's own for the known tail",
          abs(f(0.75, 12.0, 3.0) - 0.42857142857) < 1e-9, str(f(0.75, 12.0, 3.0)))

    # simple @ 5 steps, shift 12 -- the schedule this actually happens on.
    SIMPLE = [1.0, 0.9796, 0.9474, 0.8889, 0.75, 0.0]
    out = S.insert_audio_landing(SIMPLE, 12.0, 3.0)
    check("a coarse tail gets exactly one step inserted",
          len(out) == len(SIMPLE) + 1, str(out))
    check("...inserted before the zero, not after",
          out[-1] == 0.0 and out[-2] > 0.0, str(out))
    check("...and the schedule stays strictly decreasing",
          all(a > b for a, b in zip(out, out[1:])), str(out))
    check("...leaving every earlier sigma untouched",
          out[:len(SIMPLE) - 1] == SIMPLE[:-1], str(out))
    check("the audio's last jump drops from 43% to about 3%",
          abs(f(out[-2], 12.0, 3.0) - 0.03) < 1e-6, str(f(out[-2], 12.0, 3.0)))

    FINE = [1.0, 0.5, 0.154, 0.062, 0.0109, 0.0]
    check("a soft tail is not touched", S.insert_audio_landing(FINE, 12.0, 3.0) == FINE)
    for _bad in ([], [0.0], [1.0], [1.0, 0.5], None):
        check(f"malformed input is returned as-is: {_bad!r}",
              S.insert_audio_landing(_bad, 12.0, 3.0) == (_bad if _bad else _bad))
    check("a schedule with no trailing zero is left alone",
          S.insert_audio_landing([1.0, 0.75, 0.5], 12.0, 3.0) == [1.0, 0.75, 0.5])
    # Never twice, however coarse.
    once = S.insert_audio_landing(S.insert_audio_landing(SIMPLE, 12.0, 3.0), 12.0, 3.0)
    check("it is never inserted twice", len(once) == len(SIMPLE) + 1, str(once))


def test_a_kidnapping_reads_as_one_without_being_declared():
    print("\n=== the beats are allowed to say what the film is ===")
    MEM = "McKenna: she, 26, dark hair, a red jacket.\nDan: he, 40, a work coat."
    SCRIPT = ["Dan grabs McKenna from behind and covers her mouth.",
              "Dan drags McKenna towards the van.",
              "Dan forces McKenna into the back of the van.",
              "Dan ties McKenna's wrists behind her back.",
              "Dan puts a strip of tape over McKenna's mouth.",
              "Dan holds McKenna down.",
              "Dan shoves McKenna against the wheel arch.",
              "McKenna is bound and gagged in the back.",
              "Dan pulls a hood over McKenna's head.",
              "McKenna tries to get away."]
    check("the whole script reads as an abduction",
          S.film_stages_duress(SCRIPT, MEM), "")
    for _b in ("Dan ties McKenna's wrists behind her back.",
               "Dan puts a strip of tape over McKenna's mouth.",
               "McKenna is bound and gagged in the back.",
               "Dan pulls a hood over McKenna's head."):
        check(f"unambiguous: {_b[:40]!r}",
              S.beat_duress_strength(_b) == "strong", S.beat_duress_strength(_b))
    for _b in ("Dan grabs McKenna from behind and covers her mouth.",
               "Dan drags McKenna towards the van.",
               "Dan forces McKenna into the back of the van.",
               "Dan holds McKenna down.",
               "Dan shoves McKenna against the wheel arch.",
               "McKenna tries to get away."):
        check(f"seen, but ambiguous: {_b[:40]!r}",
              S.beat_duress_strength(_b) == "weak", S.beat_duress_strength(_b))
    for _b in ("She was grabbed from behind on the towpath.",
               "She is bundled into the back of the van.",
               "They were hauled out of the church one at a time.",
               "He was walked out of the building between two men.",
               "She is loaded into the back like freight.",
               "He is being held captive somewhere in the north of the city."):
        check(f"passive voice is seen: {_b[:40]!r}",
              S.beat_duress_strength(_b) != "", "")

    sh = [" ".join(x.split()) for x in
          re.split(r"(?=\[Shot )",
                   run_node("A lane at night.\n\nDan drags McKenna towards the van."
                            "\n\nDan opens the rear doors.", plan_only=True,
                            character_memory=MEM,
                            anchor="Handheld, night. A grim abduction.")[3])
          if x.strip()]
    check("the coercion beat gets the face, not just the tone",
          "shows the strain" in sh[1], sh[1][-170:])
    check("...and the beat without her still carries the film's mood",
          "The mood is grim" in sh[2], sh[2][-170:])

    for _b in ("Kate grabs a coffee and leaves.",
               "Kate drags the case to the door.",
               "Kate forces the window open.",
               "Kate ties her hair back.",
               "Kate tapes the box shut.",
               "Kate pulls the hood of the car open."):
        check(f"a prop is not a victim: {_b[:34]!r}",
              not S.film_stages_duress([_b], "Kate: she, 30, a grey coat."), "")
    check("an ordinary film is still left alone",
          not S.film_stages_duress(["Kate makes the coffee.", "Sam reads the paper."],
                                   "Kate: she, 30.\nSam: he, 33."))


def test_a_line_is_locked_to_the_person_who_says_it():
    print("\n=== the line belongs to one mouth and the others are shut ===")
    MEM = "Dan: he, 40, a work coat.\nMara: she, 33, a green scarf."

    def shot(second, first="Dan and Mara stand by the door."):
        sc = re.split(r"(?=\[Shot )",
                      run_node("A hallway.\n\n" + first + "\n\n" + second,
                               plan_only=True, character_memory=MEM,
                               anchor="Wide, day.")[3])
        return " ".join([x for x in sc if x.strip()][-1].split())

    a = shot('Dan says: "Wait here."')
    check("a lone speaking beat still locks the line",
          "Only Dan speaks" in a, a[-190:])
    check("...and closes the mouth of whoever else is in the room",
          "every other mouth in the shot stays closed" in a, a[-190:])

    b = shot('Dan says: "Wait." Mara says: "No."')
    check("two speakers are put in order",
          "Dan speaks first, then Mara" in b, b[-190:])

    check("saying nothing is not saying something",
          S.speakers_in('Dan says: "Wait here." Mara says nothing.', MEM) == ["Dan"],
          str(S.speakers_in('Dan says: "Wait here." Mara says nothing.', MEM)))
    for _b in ('Mara says nothing.', 'Mara said nothing at all.',
               'Mara does not say a word.', 'Mara never says a word.'):
        check(f"...{_b!r}", S.speakers_in(_b, MEM) == [], str(S.speakers_in(_b, MEM)))
    c = shot('Dan says: "Wait here." Mara says nothing.')
    check("...so the lock is not cancelled by it", "Only Dan speaks" in c, c[-190:])

    solo = " ".join([x for x in re.split(r"(?=\[Shot )",
                     run_node('A hallway.\n\nDan says: "Wait here."', plan_only=True,
                              character_memory="Dan: he, 40, a work coat.",
                              anchor="Wide, day.")[3]) if x.strip()][-1].split())
    check("one person in the whole film gets no lock clause",
          "every other mouth" not in solo, solo[-160:])
    # ...and an unattributable line still falls back to counting the voices.
    amb = shot('"Wait here."')
    check("an unattributed line still says how many voices there are",
          "Only the person speaking" in amb, amb[-190:])


def test_mouths_stay_shut_with_no_line():
    print("\n=== a shot with nobody speaking keeps its mouth closed ===")
    P = ('A workshop.\n\n'
         'Kate walks to the window.\n\n'
         'Kate says: "Wait there."\n\n'
         'A low hum comes off the strip light.\n\n'
         'Kate strains against the cuffs.')
    MEM = "Kate: she, 30, red coat."
    info, script = run_node(P, plan_only=True, character_memory=MEM,
                            auto_sound=False)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    mouth = [i + 1 for i, s in enumerate(sh) if "Mouths in the shot stay closed" in s]
    check("the wordless shot with a person is told to close", mouth == [1], str(mouth))
    check("the speaking shot is not", "Mouths in the shot stay closed" not in sh[1], "")
    check("the scenery beat is told nothing about mouths",
          "Mouths in the shot stay closed" not in sh[2], "")
    check("...but is still silenced", "sound is" not in sh[2], "")
    check("a straining body is left alone", "Mouths in the shot stay closed" not in sh[3], "")
    check("...and keeps its audio",
          "either describe a sound IN THE BEAT or stage EFFORT" in info
          and re.search(r"shot\(s\) [^|]*\b4\b[^|]*stage EFFORT", info) is not None,
          "")
    check("the wordless sound shot is silenced", "sound is" not in sh[2], "")
    check("info names the shots held closed", "mouths held closed on shot(s) 1" in info, "")
    check("...and names what it cost", "gave up the sound you wrote" in info, "")
    check("the clause is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b|\bnobody\b", S.MOUTH_HOLD, re.I), "")
    check("...and is one short sentence",
          S.MOUTH_HOLD.count(".") == 1 and len(S.MOUTH_HOLD.split()) <= 14, "")


def test_script_is_what_was_sent():
    print("\n=== script reports the text the model was actually given ===")
    mem = "Mara: <Picture 1>, she, 30, red coat.\nDom: he, 41, grey jacket."
    P = ("Daylight. A yard.\n\nDom stands by the gate.\n\nDom walks out through the gate.\n\n"
         "Mara walks along the fence.\n\nDom comes back and looks at the sky.")
    sent, script = _prompts_sent(P, character_memory=mem,
                                 ref_image_1=torch.rand(1, H, W, 3))
    rep = [b.split("] ", 1)[1].strip() for b in script.split("\n---\n")]
    check("every shot matches, claim included",
          all(s.strip() == r for (s, _), r in zip(sent, rep)), "")
    check("the recovery shot really does carry a claim",
          "<Picture 1>" in sent[3][0] and "<Picture 1>" in rep[3], sent[3][0][-200:])


def test_every_reference_is_claimed():
    print("\n=== no shot carries a picture its text never names ===")
    img = lambda: torch.rand(1, H, W, 3)
    cases = [
        ("one tagged, one untagged returning",
         "A yard.\n\nDom stands by the gate.\n\nDom walks out through the gate.\n\n"
         "Mara walks along the fence.\n\nDom comes back and looks at the sky.",
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
        bad = _unnamed_pictures(_encoder_rows(P, **kw))
        check(f"{name}: every picture is claimed", not bad, "; ".join(bad))


def _encoded_refs(P, **kw):
    """(prompt, number of images actually encoded as references) per shot.

    The demotion happens inside build_conditioning, so counting the refs handed TO
    it misses the handoff being added. This counts what reaches the encoder."""
    rows, box = [], {}
    ob, orr = S.build_conditioning, S._cond_module._build_ref_images
    def spy_r(vae, imgs, w, h, size):
        box["n"] = len(imgs); return orr(vae, imgs, w, h, size)
    def spy_b(clip, vae, audio_vae, prompt, *a, **k):
        box["n"] = 0
        out = ob(clip, vae, audio_vae, prompt, *a, **k)
        rows.append((prompt, box["n"]))
        return out
    S._cond_module._build_ref_images, S.build_conditioning = spy_r, spy_b
    try:
        run_node(P, **kw)
    finally:
        S._cond_module._build_ref_images, S.build_conditioning = orr, ob
    return rows


def _encoder_rows(P, **kw):
    """(prompt, pictures the ENCODER is shown, whether they must all be named) per shot.

    _encoded_refs counts the reference rows; this counts what tokenize receives,
    keyframe included. tokenize_with_weights labels every image item <Picture N>, so
    this is the roster the text has to name."""
    rows, box = [], {}
    ob, ot = S.build_conditioning, FakeCLIP.tokenize
    def spy_t(self, text, minimax_ref_items=None, **k):
        box["pics"] = sum(1 for it in (minimax_ref_items or []) if it["type"] == "image")
        return ot(self, text, minimax_ref_items=minimax_ref_items, **k)
    def spy_b(clip, vae, audio_vae, prompt, *a, **k):
        box["pics"] = 0
        out = ob(clip, vae, audio_vae, prompt, *a, **k)
        refs = sum(1 for r in (k.get("refs") or []) if r is not None)
        rows.append((prompt, box["pics"], bool(refs or k.get("handoff_as_ref"))))
        return out
    FakeCLIP.tokenize, S.build_conditioning = spy_t, spy_b
    try:
        run_node(P, **kw)
    finally:
        FakeCLIP.tokenize, S.build_conditioning = ot, ob
    return rows


def _unnamed_pictures(rows):
    """Shots whose text does not name exactly the pictures the encoder is shown.

    One exception, and it is ComfyUI's own format: a keyframe riding ALONE is
    <Picture 1> with nothing naming it, the way MiniMaxH3ImageToVideo sends a first
    frame. Beside a reference it is one more picture of the people that reference
    names, and unnamed it is a second copy of them."""
    bad = []
    for i, (p, pics, named) in enumerate(rows, 1):
        tags = sorted({int(x) for x in re.findall(r"<Picture (\d+)>", p)})
        if tags != (list(range(1, pics + 1)) if named else []):
            bad.append(f"shot {i}: {pics} shown, tags {tags}")
    return bad


def test_the_demoted_handoff_is_claimed():
    print("\n=== below the safe aug, the handoff is named too ===")
    mem = "Dan: <Picture 1>, he, 41, grey jacket.\nMara: she, 30, red coat."
    P = "A yard.\n\nDan waits.\n\nMara arrives.\n\nDan and Mara talk.\n\nDan looks up."
    img = torch.rand(1, H, W, 3)
    for aug in (0.999, 0.98, 0.95, 0.90):
        bad = _unnamed_pictures(_encoder_rows(P, character_memory=mem, ref_image_1=img,
                                              ref_noise_aug=aug))
        check(f"every picture claimed at aug {aug}", not bad, "; ".join(bad))
    # The claim must say it is the SAME people, or naming it invites a new one.
    rows = _encoded_refs(P, character_memory=mem, ref_image_1=img, ref_noise_aug=0.95)
    late = rows[2][0]
    check("the handoff is named as the opening frame",
          "is the frame this shot opens on" in late, "")
    check("...and as the same people, not new ones",
          "the same people" in late and "anybody new" in late, "")
    # At a safe aug nothing is demoted -- but beside a reference the keyframe is still a
    # picture to the encoder, and REPORTED: the same person twice in one frame while it
    # went unnamed. Named the same way; a keyframe riding alone is not.
    early = _encoder_rows(P, character_memory=mem, ref_image_1=img, ref_noise_aug=0.999)
    beside = [p for p, pics, named in early if named and pics > 1]
    check("a keyframe beside a reference is named as the opening frame",
          beside and all("is the frame this shot opens on" in p and "anybody new" in p
                         for p in beside), str(len(beside)))
    check("...and a keyframe riding alone is not",
          all("opens on" not in p for p, pics, named in early if not named), "")
    check("a chain with no reference names no keyframe",
          all("opens on" not in p for p, _, _ in _encoder_rows("A yard.\n\nOne.\n\nTwo.")),
          "")
    info = run_node(P, character_memory=mem, ref_image_1=img, ref_noise_aug=0.95)[2]
    check("info explains it", "encoded as a reference rather than a keyframe" in info, "")


def test_a_gapped_socket_still_sends_its_image():
    print("\n=== wiring ref_image_1 and ref_image_3 sends both ===")
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
    sent, _ = _prompts_sent(P, character_memory="Mara: <Picture 1>, she, a locket <Picture 2>.",
                            ref_image_1=img(), ref_image_2=img())
    p, n = sent[0]
    check("no gap, nothing changes",
          n == 2 and sorted(re.findall(r"<Picture (\d+)>", p)) == ["1", "2"], f"{n}")
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
    # Each shot also names its keyframe, after the references: that is the last tag.
    check("the lone-reference shot renumbers to 1",
          re.findall(r"<Picture (\d+)>", seen[1][0]) == ["1", "2"]
          and "<Picture 2> is the frame this shot opens on" in seen[1][0],
          str(seen[1][0][-60:]))
    check("...and still sends that person's own image",
          seen[1][1] == [0.7], str(seen[1][1]))
    check("the shared shot keeps both, in order",
          sorted(re.findall(r"<Picture (\d+)>", seen[2][0])) == ["1", "2", "3"]
          and seen[2][1] == [0.1, 0.7], str(seen[2][1]))
    # And a tag on an empty socket is reported rather than silently dropped.
    info = run_node(P, plan_only=True, character_memory="Mara: <Picture 3>, she, 30.",
                    ref_image_1=img())[2]
    check("a tag with no image behind it is named",
          "names a socket with no image on it" in info, "")


def test_a_modified_state_is_not_read_as_an_act():
    print("\n=== 'closed rear doors' is not somebody closing them ===")
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
    s = run_node("Daylight. A yard.\n\nThey stand by a van with its doors closed.",
                 plan_only=True)[3]
    check("the held state is bounded", "for the whole shot" in s, "")
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
    check("the two-ended anchor is always written",
          "by the last" in run_node(P, plan_only=True)[3], "")
    B = ("A yard. The gate is open and the blinds are drawn.\n\n"
         "Mara opens the van doors and Dom lifts the lid of the crate.")
    one = run_node(B, plan_only=True)[3]
    check("at most two frame sentences in a shot", one.count("first frame") == 2, "")
    check("...and the beat's own action wins the budget",
          "The doors are shut" in one and "The lid is shut" in one, "")


def test_dialogue_headroom():
    print("\n=== how much of a speaking shot the line does not fill ===")
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
    mem = "Nora: 34, she, red hair.\nDan: 41, he, dark hair, navy overalls"
    tail = "\n\nNora picks up the spanner."

    def encodes(beat2):
        vae = FakeVAE()
        info = run_node("Nora sets a toolbox on the bench.\n\n" + beat2 + tail,
                        anchor="A workshop.", character_memory=mem, vae=vae)[2]
        return vae.encodes, info

    n_placed, info = encodes("Dan is already sitting on the crate, watching her.")
    check("an in-position introduction still carries the frame",
          "carries the previous frame as a REFERENCE" in info)
    check("...and the run says so", "introduces Dan in position" in info)
    check("...naming what the room brings", "the room, the light and Nora come" in info)
    check("...and how to keep it as the anchor", "Write the entrance" in info)
    check("intentional room carry does not report a low-strength reference",
          "anchor would be noised" not in info)
    _s2 = re.split(r"\[Shot ", run_node(
        "Nora sets a toolbox on the bench.\n\nDan is already sitting on the crate, "
        "watching her." + tail, anchor="A workshop.", character_memory=mem)[3])[2]
    check("the carried frame is claimed as the room",
          "is this room a moment earlier" in _s2, _s2[-260:])
    check("...naming who was in it", "Nora is the person there" in _s2)
    check("...and who is already in place", "Dan is in this room too" in _s2)
    check("...without the claim that nobody new joins",
          "joined by anybody new" not in _s2)
    _run3 = run_node(
        "Nora and Ada set a toolbox on the bench.\n\nDan is already sitting on the "
        "crate, watching Ada.\n\nAda picks up the spanner.", anchor="A workshop.",
        character_memory=mem + "\nAda: 29, she, short hair")
    _s3 = re.split(r"\[Shot ", _run3[3])[2]
    check("somebody the beat does not name keeps the frame carried",
          "carries the previous frame as a REFERENCE" in _run3[2])
    check("...claimed with everyone in it", "Nora and Ada are the people there" in _s3,
          _s3[-220:])
    check("...and counted as the two it describes",
          "There are two people in the shot" in _s3, _s3[-220:])
    _info4 = run_node(
        "Nora and Ada set a toolbox on the bench.\n\nDan is already sitting on the "
        "crate, watching Ada.\n\nAda picks up the spanner.", anchor="A workshop.",
        character_memory=mem + "\nAda: <Picture 1>, 29, she, short hair",
        ref_image_1=torch.rand(1, H, W, 3))[2]
    check("a portrait in the frame keeps the fresh start",
          "carries the previous frame as a REFERENCE" not in _info4)
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
        check("...and the prose claims the recovered one, then the keyframe",
              _tags == ["<Picture 1>", "<Picture 2>"]
              and "<Picture 2> is the frame this shot opens on" in _txt, str(_tags))
        check("...on the entry of the person it depicts",
              "Dan: <Picture 1>," in _txt, _txt[:80])
        shared = ["Nora walks into the workshop.",
                  "Victor comes in and Nora hands him the spanner.",
                  "Victor walks in and kneels by the cable, alone.",
                  "Nora comes back in and picks up the toolbox."]
        info2 = run_node("\n\n".join(shared), anchor="A room.", character_memory=mem)[2]
        src = re.search(r"recovered a face for Nora on shot 4, from shot (\d+)", info2)
        check("the frame comes from a shot that was hers alone",
              bool(src) and src.group(1) == "1", src.group(1) if src else "none")
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
    shots = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("the script is not rewritten", "Alex says hello to Maya." in shots[1])
    check("...and no entry is invented for them", "Alex:" not in script)
    # A sheet that covers everyone says nothing.
    clean = run_node("Maya walks in.\n\nJon greets Maya.", plan_only=True,
                     anchor="A room.", character_memory=mem)[2]
    check("a complete sheet is not reported", "has no entry" not in clean)


def test_references_ride_with_the_keyframe():
    print("\n=== a reference and the keyframe ride together ===")
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
        run_node("Kate walks in.\n\nMike walks in and Kate sits beside him.\n\n"
                 "Mike stands up.",
                 anchor="A room.", character_memory=mem,
                 ref_image_1=torch.rand(1, H, W, 3))
        shots = seen[1:]                     # seen[0] is the negative
        check("the roster is ref + keyframe, not one or the other",
              [p for p, _ in shots] == [1, 2, 1], str([p for p, _ in shots]))
        # Both are pictures to the encoder, so both are named: the reference on Kate,
        # the keyframe as the frame the shot opens on.
        check("the shot carrying both names both pictures",
              shots[1] == (2, 2), str(shots[1]))
        check("a shot the guard trimmed carries no reference", shots[2][1] == 0)
        info, script = run_node("Kate walks in.\n\nKate sits down.", plan_only=True,
                                anchor="A room.", character_memory=mem,
                                ref_image_1=torch.rand(1, H, W, 3))[2:4]
        check("the binding survives into the script", "<Picture 1>" in script)
        check("...on the person it depicts", "Kate: <Picture 1>, 22" in script)
        check("the run explains the pairing", "ride alongside the keyframe" in info)
        seen.clear()
        run_node("A room.\n\nOne.\n\nTwo.\n\nThree.")
        check("a chain with no reference keeps the keyframe picture",
              any(pics == 1 for pics, _ in seen))
    finally:
        FakeCLIP.tokenize = orig


def test_audio_sigma_reads_the_scheduler():
    print("\n=== the last audio sigma follows the SCHEDULER, not just the formula ===")
    import comfy.samplers as _cs
    _ms_mod = sys.modules.setdefault("comfy.model_sampling",
                                     types.ModuleType("comfy.model_sampling"))
    setattr(sys.modules["comfy"], "model_sampling", _ms_mod)
    seen = []
    def _fake(model_sampling, scheduler, steps):
        seen.append((scheduler, steps))
        return [1.0, 0.669, 0.416, 0.201, 0.003, 0.0]   # kl_optimal-shaped tail
    class _FakeMS:
        def set_parameters(self, **kw): self.kw = kw
    _had = getattr(_cs, "calculate_sigmas", None)
    _had_ms = getattr(_ms_mod, "ModelSamplingDiscreteFlow", None)
    _cs.calculate_sigmas = _fake
    _ms_mod.ModelSamplingDiscreteFlow = _FakeMS
    try:
        got = S.last_audio_sigma(5, 3.0, scheduler="kl_optimal")
    finally:
        if _had is None:
            del _cs.calculate_sigmas
        else:
            _cs.calculate_sigmas = _had
        if _had_ms is None:
            del _ms_mod.ModelSamplingDiscreteFlow
        else:
            _ms_mod.ModelSamplingDiscreteFlow = _had_ms
    check("it asks the scheduler for the real ladder", seen == [("kl_optimal", 5)], str(seen))
    check("...and reports that, not the formula's 0.43", got < 0.05, f"{got:.4f}")


def test_audio_sigma_falls_back_to_the_closed_form():
    print("\n=== ...and without a real ComfyUI it still gives the `simple` answer ===")
    got = S.last_audio_sigma(5, 3.0)
    check("closed form still exact for simple", abs(got - 3.0 / 7.0) < 1e-6, f"{got:.4f}")
    check("...and 8 steps still gives the documented 0.30",
          abs(S.last_audio_sigma(8, 3.0) - 0.30) < 1e-6, f"{S.last_audio_sigma(8, 3.0):.4f}")

def test_a_hidden_garments_lettering_goes_with_it():
    print("\n=== the print on a covered garment is covered too ===")
    T = 'McKenna: she, 26, denim shorts, a black thong with "PRINCESS" across the front.'
    got = S.hide_item(T, ["a black thong"])
    check("the lettering leaves with the garment", "princess" not in got.lower(), got)
    check("...and the outer garment stays", "denim shorts" in got.lower(), got)
    check("...and the person keeps her line", "McKenna" in got, got)

    T2 = 'McKenna: she, 26, denim shorts, a "PRINCESS" lettered thong.'
    got2 = S.hide_item(T2, ["thong"])
    check("...also when the print is written first",
          "princess" not in got2.lower(), got2)

    T3 = "McKenna: she, 26, a thong and denim shorts."
    got3 = S.hide_item(T3, ["thong"])
    check("a fragment naming another garment keeps it",
          "denim shorts" in got3.lower(), got3)

    for label, T in (
            ("its own fragment",
             'McKenna: she, 26, a black thong, "BRAT" printed across the back, denim shorts.'),
            ("pointing back with a pronoun",
             "McKenna: she, 26, a black thong, the word BRAT across its back, denim shorts."),
            ("a bare capitalised word",
             "McKenna: she, 26, a black thong, BRAT across the back, denim shorts."),
            ("a trailing modifier that is not a print",
             "McKenna: she, 26, a black thong, with a bow at the hip, denim shorts."),
            ("a sentence of its own",
             "McKenna: she, 26, denim shorts. She wears a black thong. BRAT is printed across the back.")):
        got = S.hide_item(T, ["thong"])
        check(f"the print goes with the garment: {label}",
              "brat" not in got.lower() and "bow" not in got.lower()
              and "denim shorts" in got.lower() and "McKenna" in got
              and "She ." not in got, got)
    T5 = "McKenna: she, 26, a black thong, red lipstick, a tattoo across the lower back, denim shorts."
    got5 = S.hide_item(T5, ["thong"])
    check("an unrelated item after the garment stays",
          "red lipstick" in got5 and "tattoo across the lower back" in got5
          and "thong" not in got5, got5)

def test_camera_framing_is_read_from_the_anchor():
    print("\n=== a close frame written in the ANCHOR is still a close frame ===")
    check("the anchor's framing is seen", S.tight_framing("Close-up on her face."))
    check("...and a beat's still is", S.tight_framing("She leans in, tight on her eyes."))
    check("...and a wide anchor is not", not S.tight_framing("Wide shot, night."))

    mem = "McKenna: she, 26, denim shorts."
    P = ("McKenna lies on the bed, wrists cuffed behind her back.\n\n"
         "She turns her head.")
    info = run_node(P, plan_only=True, character_memory=mem,
                    anchor="Close-up on her face. 85mm lens.")[2]
    check("the tight-frame warning fires from the anchor alone",
          "frame tight enough to crop the anchor point out" in info,
          info[:200])
    wide = run_node(P, plan_only=True, character_memory=mem,
                    anchor="Wide shot, night.")[2]
    check("...and does not fire on a wide film",
          "frame tight enough to crop the anchor point out" not in wide)

def test_a_tight_frame_drops_wardrobe_it_cannot_contain():
    print("\n=== a close-up stops describing what is outside it ===")
    mem = "McKenna: she, 26, dark hair, a red jacket, denim shorts, black boots."
    P = "McKenna lies on the bed, wrists cuffed behind her back.\n\nShe turns her head."
    tight = run_node(P, plan_only=True, character_memory=mem,
                     anchor="Close-up on her face. 85mm lens.")[3]
    check("legs are out of a face frame", "denim shorts" not in tight.lower(), tight[:200])
    check("...and so are the feet", "black boots" not in tight.lower(), tight[:200])
    check("...but head-and-shoulders keeps the jacket",
          "red jacket" in tight.lower(), tight[:200])
    check("...and she is still herself", "McKenna" in tight)
    check("...and the restraint is STILL said",
          "cuff" in tight.lower(), tight[:260])
    check("...and where the limbs are held is still said",
          "small of the back" in tight.lower(), tight[:260])

    # A frame naming the HANDS keeps the gloves and drops the rest.
    hands = run_node(P, plan_only=True,
                     character_memory="McKenna: she, 26, gloves, denim shorts.",
                     anchor="Macro lens on her hands.")[3]
    check("a hand frame keeps the gloves", "gloves" in hands.lower(), hands[:200])
    check("...and drops the shorts", "denim shorts" not in hands.lower(), hands[:200])

    wide = run_node(P, plan_only=True, character_memory=mem, anchor="Wide shot, night.")[3]
    check("a wide film keeps everything", "denim shorts" in wide.lower(), wide[:200])
    bare = run_node(P, plan_only=True, character_memory=mem, anchor="Close-up. 85mm.")[3]
    check("a subjectless close-up changes nothing",
          "denim shorts" in bare.lower(), bare[:200])

def test_the_soundtrack_is_the_models_own():
    """REPORTED: "Just get rid of the ambient sounds all together. They sound horrid.
    Go back to the model's natural audio."

    The node used to build the non-vocal half of the soundtrack itself -- a room tone
    shaped from the scene's own wording, and 21 foley recipes laid into the shots whose
    audio branch is pinned to silence. The reasoning was sound: a pinned shot cannot
    get audio from the model at all, because prompt text never opens a branch, so
    auto_sound was writing sounds into prompts that could not make them.

    It still did not pass. Reported first as footsteps sounding like heartbeats and a
    bathroom that tapped; then, once both of those measured clean, as horrid anyway.
    Synthesis that measures right and sounds wrong is the end of that road.

    So the test is the absence: what comes out is what the model made, and neither
    level widget may move a single sample of it."""
    print("\n=== the soundtrack is the model's own ===")
    P = ("A tiled bathroom. The shower runs and water moves in the pipes.\n\n"
         "Kate walks across the tiles to the shower.\n\n"
         "Kate stands under the water.")

    def _run(**kw):
        torch.manual_seed(12345)
        return run_node(P, character_memory="Kate: she, 28, a towel.", **kw)[1]["waveform"]

    base = _run(ambient_level=0.0, foley_level=0.0)
    check("the run is reproducible with the seed pinned",
          torch.equal(base, _run(ambient_level=0.0, foley_level=0.0)))
    for _al, _fl in ((0.9, 0.9), (0.25, 0.35), (1.0, 0.0), (0.0, 1.0)):
        check(f"ambient_level={_al} foley_level={_fl} changes not one sample",
              torch.equal(base, _run(ambient_level=_al, foley_level=_fl)))
    for _n in ("foley_for", "synth_ambient", "plain_bed", "_FOLEY", "phrase_seed",
               "bed_recipe", "_BED_RECIPE", "_BED_EVENTFUL", "_BED_RMS", "_MODES",
               "_band", "_hits", "_even", "_contact", "_flow", "_creak",
               "_gait", "_walk", "_step_period", "_MOTION_TIMED", "motion_envelope"):
        check(f"no builder named {_n!r} survives", not hasattr(S, _n))
    check("ShotAudio no longer offers to accept built foley",
          not hasattr(S.ShotAudio(False, True, False, True, 0.0, S.AUDIO_LATENT_FPS),
                      "accepts_built_foley"))
    _src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    for _gone in ("foley_for(", "synth_ambient(", "plain_bed(", "motion_envelope(",
                  "_foley_on", "accepts_built_foley"):
        check(f"no call site remains: {_gone!r}", _gone not in _src)
    _opt = list(S.H3LongVideos.INPUT_TYPES()["optional"].keys())
    check("ambient_audio, ambient_level and foley_level keep their positions",
          _opt[23:26] == ["ambient_audio", "ambient_level", "foley_level"])
    check("...and the three after them have not shifted",
          _opt[26:29] == ["speech_lead_seconds", "speech_tail_seconds", "hold_levels"])
    _bed = {"waveform": torch.full((1, 2, 8000), 0.5), "sample_rate": 44100}
    _mixed, _note = S.mix_ambient(torch.zeros((1, 2, 16000)), 44100, _bed, 0.5)
    check("a wired bed reaches the soundtrack", float(_mixed.abs().max()) > 0.1)
    check("a wired bed at level 0 is a no-op",
          torch.equal(S.mix_ambient(torch.zeros((1, 2, 16000)), 44100, _bed, 0.0)[0],
                      torch.zeros((1, 2, 16000))))
    check("...as is no bed at all",
          torch.equal(S.mix_ambient(torch.zeros((1, 2, 16000)), 44100, None, 0.9)[0],
                      torch.zeros((1, 2, 16000))))
    _info = run_node(P, character_memory="Kate: she, 28.", foley_level=0.35)[2]
    check("info says the level does nothing now", "does nothing any more" in _info)
    check("...and says what that costs", "is SILENT" in _info)
    check("...and how to get sound back",
          "write the sound into that beat" in _info and "ambient_audio" in _info)


def test_shot_one_is_the_only_unpinned_shot():
    """REPORTED: "The girl doesn't look the same from the first to last beat... The
    remaining beats are fine. I even have image reference strength set to 0.999", with
    a hardware artefact in beat 1 alone -- her hair caught in a collar.

    Both halves are one fact, and it is structural rather than a bug: shot 1 is the
    only shot whose opening frame is pinned by nothing. Every later shot opens on the
    previous shot's last frame, which fixes pose, framing and the arrangement of
    everything on the body, so the chain agrees with itself and shot 1 is the one that
    can disagree. An arrangement no picture settles -- how hair sits against a collar
    -- is settled by the model, once, in the only shot with no picture.

    And ref_noise_aug cannot reach it, which is the part the report turned on.
    build_conditioning's own comment is plain: "the keyframe ANCHORS the first frame,
    which is what continuity needs, while a reference only supplies identity. They are
    not alternatives." A cleaner reference sharpens identity; there is no frame on
    shot 1 for it to sharpen."""
    print("\n=== shot 1 is the only unpinned shot ===")
    mem = "Nora: <Picture 1>, she, 24, long dark hair, a steel collar, a grey dress."
    P = ("A bare room, cold light.\n\nNora stands by the window.\n\n"
         "Nora turns to look at the door.\n\nNora sits on the crate.")
    ref = torch.rand((1, 64, 64, 3))

    def _pins(**kw):
        """What pins each shot's opening frame, as seen by build_conditioning."""
        got, orig = [], S.build_conditioning
        def spy(clip, vae, audio_vae, prompt, width, height, length,
                handoff=None, refs=None, ref_noise_aug=0.999, **k):
            got.append(("keyframe" if (handoff is not None and not k.get("handoff_as_ref"))
                        else "reference" if handoff is not None else "nothing",
                        len(refs or [])))
            return orig(clip, vae, audio_vae, prompt, width, height, length,
                        handoff=handoff, refs=refs, ref_noise_aug=ref_noise_aug, **k)
        S.build_conditioning = spy
        _p = kw.pop("prompt_override", P)
        try:
            run_node(_p, character_memory=mem, **kw)
        finally:
            S.build_conditioning = orig
        return got

    bare = _pins(ref_image_1=ref, ref_noise_aug=0.999)
    check(f"three shots measured: {[p for p, _ in bare]}", len(bare) == 3)
    check("shot 1's opening frame is pinned by NOTHING", bare[0][0] == "nothing")
    check("...while every later shot has a keyframe",
          all(p == "keyframe" for p, _ in bare[1:]))
    check("...and shot 1 does carry the reference anyway",
          all(n == 1 for _p, n in bare))
    for _aug in (0.90, 0.95, 0.999, 1.0):
        _p = _pins(ref_image_1=ref, ref_noise_aug=_aug)
        check(f"at ref_noise_aug {_aug} shot 1 is still unpinned", _p[0][0] == "nothing")
    _ff = _pins(ref_image_1=ref, first_frame=ref, ref_noise_aug=0.999)
    check("first_frame reaches shot 1 either way", _ff[0][0] != "nothing")
    _ent = _pins(ref_image_1=ref, first_frame=ref, ref_noise_aug=0.999,
                 prompt_override=("A bare room, cold light.\n\n"
                                  "Nora walks in through the side door.\n\n"
                                  "Nora turns to look at the door.\n\n"
                                  "Nora sits on the crate."))
    check("...and pins frame one when beat 1 stages an entrance",
          _ent[0][0] == "keyframe")
    # ...and info says all of that, including which dial is NOT this one.
    _info = run_node(P, plan_only=True, character_memory=mem, ref_image_1=ref,
                     ref_noise_aug=0.999)[2]
    _n = [x for x in _info.split(" | ") if "NO first_frame" in x]
    check("info reports the asymmetry", len(_n) == 1)
    _n = _n[0] if _n else ""
    check("...saying shot 1 alone is pinned by nothing",
          "only shot in this film whose opening frame is pinned by NOTHING" in _n)
    check("...that every other shot opens on the previous last frame",
          "previous shot's last frame" in _n)
    check("...that ref_noise_aug cannot fix it",
          "ref_noise_aug IS NOT THE DIAL FOR THIS" in _n)
    check("...naming the one-shot-only symptom that was reported",
          "hair sitting differently against a collar" in _n)
    check("...and first_frame as the fix", "Wire first_frame to fix it" in _n)
    # It must go quiet once the fix is applied, or it is noise.
    check("...and says nothing once first_frame is wired",
          "NO first_frame" not in run_node(P, plan_only=True, character_memory=mem,
                                          ref_image_1=ref, first_frame=ref)[2])
    check("with a reference wired it defers rather than repeating",
          "see the ref_noise_aug note above" in _n
          and "subject, pose, framing, background" not in _n)
    _no_ref = [x for x in run_node(P, plan_only=True, character_memory=mem)[2].split(" | ")
               if "NO first_frame" in x][0]
    check("...and carries the advice itself when there is no reference note",
          "subject, pose, framing, background" in _no_ref)


def test_a_first_frame_can_be_the_set_instead_of_frame_one():
    """REPORTED, after being told to wire first_frame: "No, this should be fixed using
    reference images from the start. I only use first_frame for the scene."

    Which is a workflow the node could not serve. first_frame was always taken as
    shot 1's opening frame, pinned whole -- and a picture of a SET has nobody in it,
    so the cast the beat places there has to appear out of nothing during shot 1. That
    is the reported artefact: beat 1 wrong, every later beat fine, because beat 2
    inherits the settled version from its handoff.

    The node already refuses exactly this on every OTHER shot, with its own reasoning
    written down -- "that frame does not have them in it, and a keyframe is a picture,
    so they would have to appear out of nothing and travel to the spot the beat
    describes". Shot 1 could not reach it: the guard read `and plan`, which is empty on
    the first shot. True before first_frame could supply a keyframe; wrong after.

    The set is told from an opening frame BY THE SCRIPT, never by looking at pixels:
    the beat has to place the cast rather than stage an entrance, and every one of them
    has to already have a portrait of their own. Then this picture has nothing left to
    say about who they are, only about where they are."""
    print("\n=== a first_frame can be the set instead of frame one ===")
    plate = torch.rand((1, 256, 256, 3))
    ref = torch.rand((1, 256, 256, 3))
    TAG = "Nora: <Picture 1>, she, 24, long dark hair, a steel collar."
    BARE = "Nora: she, 24, long dark hair, a steel collar."
    PLACED = "A bare room.\n\nNora stands by the window.\n\nNora turns to the door."
    ARRIVES = ("A bare room.\n\nNora walks in through the side door.\n\n"
               "Nora turns to the door.")

    def _first(mem, P, **kw):
        """(how shot 1's first_frame is used, refs passed, shot 1's prompt)."""
        got, orig = [], S.build_conditioning
        def spy(clip, vae, audio_vae, prompt, width, height, length,
                handoff=None, refs=None, ref_noise_aug=0.999, **k):
            got.append(("set" if (handoff is not None and k.get("handoff_as_ref"))
                        else "keyframe" if handoff is not None else "nothing",
                        len(refs or []), prompt))
            return orig(clip, vae, audio_vae, prompt, width, height, length,
                        handoff=handoff, refs=refs, ref_noise_aug=ref_noise_aug, **k)
        S.build_conditioning = spy
        try:
            run_node(P, character_memory=mem, **kw)
        finally:
            S.build_conditioning = orig
        return got[0]

    # THE REPORTED CASE. A plate, a beat that places her, a portrait of her.
    _how, _n, _pr = _first(TAG, PLACED, first_frame=plate, ref_image_1=ref)
    check("a plate with the cast placed and portrayed rides as the SET", _how == "set")
    check("...her portrait keeps <Picture 1>", "<Picture 1>, she, 24" in _pr)
    check("...the plate is claimed as the set, after her",
          "<Picture 2> is the set this shot takes place in" in _pr)
    check("...and claimed as having nobody in it",
          "a picture of the place only, with nobody in it" in _pr)
    check("...and not as a room a moment earlier", "a moment earlier" not in _pr)

    check("an entrance in beat 1 still pins frame one",
          _first(TAG, ARRIVES, first_frame=plate, ref_image_1=ref)[0] == "keyframe")
    check("no portrait means the frame is her only picture, so it stays frame one",
          _first(BARE, PLACED, first_frame=plate)[0] == "keyframe")
    check("...and a tag pointing at an empty socket portrays nobody",
          _first(TAG, PLACED, first_frame=plate)[0] == "keyframe")
    check("no first_frame at all is unchanged",
          _first(TAG, PLACED, ref_image_1=ref)[0] == "nothing")

    # TWO PEOPLE: every one of them has to be covered, not just the first.
    _two = ("Nora: <Picture 1>, she, 24, long dark hair.\n"
            "Dan: <Picture 2>, he, 41, a grey coat.")
    _one = "Nora: <Picture 1>, she, 24, long dark hair.\nDan: he, 41, a grey coat."
    _P2 = ("A bare room.\n\nNora stands by the window and Dan sits on the crate.\n\n"
           "Nora turns to the door.")
    check("both portrayed -> the plate is the set",
          _first(_two, _P2, first_frame=plate, ref_image_1=ref,
                 ref_image_2=ref)[0] == "set")
    check("one of them unportrayed -> it stays frame one",
          _first(_one, _P2, first_frame=plate, ref_image_1=ref)[0] == "keyframe")

    # ...and it is REPORTED, with what to do if the reading is wrong.
    _info = run_node(PLACED, plan_only=True, character_memory=TAG,
                     first_frame=plate, ref_image_1=ref)[2]
    check("info says the frame was read as the set",
          "first_frame is being read as the SET" in _info)
    check("...and that it is not discarded", "NOT discarded" in _info)
    check("...and how to pin frame one instead",
          "put the cast IN that frame" in _info and "write the entrance" in _info)
    check("...and says nothing when the frame is taken as frame one",
          "read as the SET" not in run_node(ARRIVES, plan_only=True, character_memory=TAG,
                                           first_frame=plate, ref_image_1=ref)[2])


def test_the_age_reaches_the_shot_and_the_floor_holds():
    """End to end for "Breast development should also be correct, given the age of a
    person" -- and for the floor that request makes necessary.

    An age driving anatomy has to answer for the ages it was not meant for, so the same
    reader that names a woman of 45 names nothing at all below 18."""
    print("\n=== the age reaches the shot, and the floor holds ===")
    P = "A bare room.\n\nKate takes off her vest.\nremove: vest\n\nKate stands still."

    def _chest(mem):
        sh = [" ".join(x.split()) for x in
              run_node(P, plan_only=True, character_memory=mem)[3].split("---")
              if x.strip()]
        m = re.search(r"(The chest[^.]*\.)", sh[1])
        return m.group(1) if m else ""

    _22 = _chest("Kate: she, 22, a grey vest, denim shorts.")
    _58 = _chest("Kate: she, 58, a grey vest, denim shorts.")
    _none = _chest("Kate: she, a grey vest, denim shorts.")
    check("the age reaches the shot", "the body of a woman of 22" in _22, _22[-90:])
    check("...and so does the chest at that age", "sitting high on the chest" in _22)
    check("a different age says something different",
          "the body of a woman of 58" in _58 and "hanging low" in _58, _58[-90:])
    check("...so two ages are not one description", _22 != _58)
    check("no age keeps the plain phrasing",
          "on a woman's body" in _none and "the breasts" not in _none, _none[-90:])
    mem = "Kate: she, 38, a grey vest, denim shorts.\nSam: he, 9, a school jumper."
    info, script = run_node(
        "A kitchen.\n\nSam eats breakfast.\n\nKate reads the paper.",
        plan_only=True, character_memory=mem)[2:4]
    check("a scene with a child in it still renders", "[Shot 1]" in script)
    check("...and no body is named for them anywhere",
          "the body of a man of 9" not in script and "a man's body" not in script)
    check("...and info says so rather than leaving it to be noticed",
          "declared under 18 on the sheet" in info and "NO body is described" in info)
    check("...and says the scene itself renders", "The scene renders" in info)
    check("...and that it would not have with sex staged in it",
          "nothing would have rendered at all" in info)
    check("a film with no minor says none of that",
          "declared under 18" not in run_node(
              P, plan_only=True,
              character_memory="Kate: she, 38, a grey vest, denim shorts.")[2])
    _raised = ""
    try:
        run_node("A bedroom.\n\nKate undresses and lies down.\n\nShe moans.",
                 character_memory="Kate: she, 28.\nSam: she, 15.")
    except RuntimeError as _e:
        _raised = str(_e)
    check("a declared minor plus sexual staging renders nothing", _raised != "")
    check("...saying so plainly", "REFUSED" in _raised and "nothing was rendered" in _raised)
    check("...and naming the entry that tripped it", "Sam" in _raised)
    # Both halves of that really are required, end to end.
    _ok = run_node("A bedroom.\n\nKate undresses and lies down.\n\nShe moans.",
                   plan_only=True, character_memory="Kate: she, 28.")
    check("the same script with no minor declared renders", "[Shot 1]" in _ok[3])
    _ok2 = run_node("A kitchen.\n\nSam eats breakfast.\n\nKate reads.",
                    plan_only=True, character_memory="Kate: she, 38.\nSam: she, 15.")
    check("a declared minor in an ordinary scene renders", "[Shot 1]" in _ok2[3])


def test_sound_survives_silencing():
    print("\n=== a described sound is not silenced away ===")
    P = ("Two people, late evening.\n\n"
         "Maya: 27, grey coat.\n\n"
         "Maya lies still.\n\n"
         "The chain drags and rattles beside her.\n\n"
         'Jon says: "Get up."')
    vae = FakeAudioVAE()
    info = run_node(P, audio_vae=vae)[2]
    check("the beat that describes a sound keeps its audio",
          "describe a sound IN THE BEAT" in info or "were given the sound" in info)
    check("...and every shot is given the room to be in",
          "ambient bed read from" in info)
    check("...and the written sound is not overwritten",
          "were given the sound" in info or "describe a sound IN THE BEAT" in info)
    # THE SWITCH IS GONE. Silencing is decided per shot -- it fires where there is
    # no quoted line -- so a stale workflow still sending silence_nonspeech=False is
    # absorbed and ignored rather than turning a correct decision off.
    off = run_node(P, silence_nonspeech=False)[2]
    check("a stale silence_nonspeech=False no longer disables it",
          "conditioned on real silence" in off)
    traded = run_node(P, audio_vae=FakeAudioVAE(), auto_sound=False)[2]
    check("the mouth guard gives the written sound up", "gave up the sound you wrote" in traded)
    check("...and the shot it took it from is really silenced",
          "shot(s) 1, 2 have no quoted line and no sound described" in traded, traded[-260:])


def test_auto_sound_end_to_end():
    print("\n=== sound generated from the prompt ===")
    P = ("Two people, late evening.\n\n"
         "Maya: 27, grey coat. Wrists cuffed behind back.\n\n"
         "Jon walks in holding a pair of scissors and says: \"Hold still.\"\n\n"
         "Jon walks to the bench and looks at the box.\n\n"
         "Maya lies still.\n\n"
         "The chain drags and rattles beside her.")
    imgs, audio, info, script = run_node(P, plan_only=True,
                                         mouths_shut_when_no_line=False)[:4]
    sh = [" ".join(x.split()) for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    # Shot 1 speaks, so its branch is open anyway and the action's sound is added.
    check("walking is heard on the shot that speaks", "footsteps" in sh[0])
    check("...and the scissors", "blades through fabric" in sh[0])
    check("...in the open form, because it has a line", "It sounds like" in sh[0])
    check("a beat staging nothing audible gets nothing", "sounds like" not in sh[2])
    # What you wrote wins: a beat describing its own sound is left alone AND stays open.
    check("a beat with its own sound is not overwritten", "It sounds like" not in sh[3])
    check("info lists the shots it scored", "were given the sound" in info)
    check("...saying it can never unsilence one", "never unsilence a shot" in info)
    check("the balance separates sound from guards", "sound " in info.split("balance")[1][:120])
    # The switch is gone: a stale auto_sound=False is absorbed, not obeyed.
    stale = run_node(P, plan_only=True, auto_sound=False,
                     mouths_shut_when_no_line=False)[3]
    check("a stale auto_sound=False still scores the film", stale == script)


def test_room_tone_under_every_shot():
    print("\n=== the room is heard even when nothing happens ===")
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
    check("the speaking shot carries the bed", "low hum off the strip light" in sh[0])
    check("the wordless shot is left silent",
          "low hum off the strip light" not in sh[1])
    check("...and info says so", "ambient bed read from the anchor" in info)
    _pin = re.search(r"shot\(s\) ([\d, ]+) have no quoted line and no sound described",
                     info)
    _open = re.search(r"shot\(s\) ([\d, ]+) have no line but either", info)
    check("the wordless shot is pinned to silence",
          _pin is not None and "2" in _pin.group(1), _pin.group(1) if _pin else "none")
    check("...and is not counted as describing its own sound",
          _open is None or "2" not in _open.group(1),
          _open.group(1) if _open else "none")

    q_info, q_script = run_node(P, plan_only=True, auto_sound=False)[2:4]
    q_sh = [" ".join(x.split()) for x in re.split(r"(?=\[Shot )", q_script)
            if x.strip()]
    check("auto_sound off: a wordless shot is silenced",
          "conditioned on real silence" in q_info)
    check("...and the room does not go under it",
          "hard walls" not in q_sh[1], q_sh[1][-70:])
    check("...and it is told nothing about sound",
          "sounds like" not in q_sh[1] and "only sound" not in q_sh[1])
    off = run_node(P, plan_only=True, mouths_shut_when_no_line=False)[3]
    off_sh = [b for b in off.split("---") if b.strip()]
    check("a stale mouths_shut=False does not reopen the branch",
          "The only sound" not in off_sh[2])
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


def test_a_vocal_shot_says_what_fills_the_gaps():
    print("\n=== between the moans ===")
    def _clause(_beat, _scene="Inside a van at night."):
        _sh = run_node(f"{_scene}\n\n{_beat}", plan_only=True)[3]
        _m = re.search(r"(The only sound[^.]*\.|It sounds like[^.]*\.)", _sh)
        return _m.group(1) if _m else ""

    for _beat, _word in (("She moans.", "moaning"),
                         ("He thrusts into her, and she moans.", "moaning"),
                         ("She whimpers and thrashes in her restraints.", "whimpering"),
                         ("She sobs quietly.", "sobbing"),
                         ("She screams.", "screaming"),
                         ("She groans.", "groaning"),
                         ("She whines.", "whining")):
        _c = _clause(_beat)
        check(f"the vocal is still named: {_word}", _word in _c, f"{_beat!r} -> {_c!r}")
        check(f"...and the gaps between them are too: {_word}",
              S._VOCAL_BETWEEN in _c, f"{_beat!r} -> {_c!r}")
        check(f"...with the author's word first: {_word}",
              _c.index(_word) < _c.index(S._VOCAL_BETWEEN), _c)
    check("the clause is still exclusive with no line spoken",
          _clause("She moans.").startswith("The only sound"), _clause("She moans."))
    check("a beat with no vocal gains no breath", _clause("He walks in.") == "")
    check("...nor does a written NON-vocal sound, which takes the muted path",
          _clause("The chain rattles against the frame.") == "")
    _both = _clause('McKenna moans, and Dan says: "Nearly there."')
    check("a vocal beside a line keeps an open list",
          _both.startswith("It sounds like"), _both)
    check("...and still names both the vocal and the breath",
          "moaning" in _both and S._VOCAL_BETWEEN in _both, _both)
    # The breath is the node's own existing vocabulary, not a new word invented here.
    check("the between-vocal sound is a phrase the node already used elsewhere",
          S._VOCAL_BETWEEN in S._VOCAL_RETIRES, S._VOCAL_BETWEEN)
    _sil = run_node("A quiet room.\n\nShe stands by the window.", plan_only=True)[3]
    check("a silenced shot gains no breath", S._VOCAL_BETWEEN not in _sil, _sil[-200:])


def test_an_exclusive_sound_clause_never_denies_the_beat():
    print("\n=== the closed list names the sound the author wrote ===")
    def _clause(_beat):
        _sh = run_node(f"Inside a van at night.\n\n{_beat}", plan_only=True)[3]
        _m = re.search(r"(The only sound[^.]*\.|It sounds like[^.]*\.)", _sh)
        return _m.group(1) if _m else ""

    for _beat, _word in (("She screams.", "screaming"),
                         ("She sobs quietly.", "sobbing"),
                         ("She starts whimpering and thrashes in her restraints.",
                          "whimpering"),
                         ("She moans and pulls against the cuffs.", "moaning")):
        _c = _clause(_beat)
        check(f"the clause names it: {_word}", _word in _c, f"{_beat!r} -> {_c!r}")
        check(f"...and no longer claims the bed is the only sound: {_word}",
              not re.match(r"The only sound is an? [a-z ]+\.$", _c), _c)
    _chain = "The chain rattles against the frame."
    check("a written non-vocal sound still takes the muted path", _clause(_chain) == "",
          _clause(_chain))
    _info = run_node(f"Inside a van at night.\n\n{_chain}", plan_only=True)[2]
    check("...and the trade is still reported",
          "gave up the sound you wrote" in _info)
    check("a silent beat gains no vocal", _clause("He walks in.") == "")
    # named_vocals_in is the author's own word, matched literally -- not an inference.
    check("named_vocals_in reads the beat", S.named_vocals_in("She screams.") == ["screaming"])
    check("...and finds nothing where nothing is named",
          S.named_vocals_in("He walks in.") == [])
    check("...and it is the same table sounds_for uses",
          {v for _p, v in S._VOCAL_FROM} == set(S._NAMED_VOCALS))


def test_the_chain_is_never_held_twice():
    print("\n=== the chain is one allocation from first shot to return ===")
    for _n in (2, 4, 8):
        _P = "A room.\n\n" + "\n\n".join(f"Beat {_i}." for _i in range(_n))
        _out = run_node(_P)
        _v = _out[0]
        _slack = _v.untyped_storage().nbytes() - _v.numel() * _v.element_size()
        _frame = _v[0].numel() * _v.element_size()
        # The join is a view of the one buffer, so the seam frames trim_seam dropped
        # are its tail -- never written, so never committed, so no RAM. Making this 0
        # took a second copy of the whole chain at the join, at the end of the run,
        # with RAM at its fullest: a server killed by the OOM killer.
        check(f"{_n} beats: the only slack is the trimmed seam frames, never written",
              _slack % _frame == 0 and _slack // _frame <= _n - 1,
              f"unused storage: {_slack} bytes = {_slack / _frame:g} frames")
    # ...and it is still the right pixels, in range, in the output dtype.
    _out = run_node("A room.\n\nOne.\n\nTwo.\n\nThree.")
    _v = _out[0]
    check("...and still float32 by default", _v.dtype == torch.float32, str(_v.dtype))
    check("...and still in range", float(_v.min()) >= 0.0 and float(_v.max()) <= 1.0)
    check("...and the frame count is unchanged", _v.shape[0] == _out[5])

    _real_decode = FakeVAE.decode
    try:
        FakeVAE.decode = lambda self, latent: torch.rand(
            max(1, (latent.shape[2] - 2) // 5 * 17 + 5) + 3, H, W, 3).to(_vae_out_dtype())
        _o = run_node("A room.\n\nOne.\n\nTwo.\n\nThree.")
        _vo = _o[0]
        check("a VAE that over-decodes still returns a chain", _vo.shape[0] > 0)
        check("...of the length it actually decoded", _vo.shape[0] == _o[5])
        check("...in range", float(_vo.min()) >= 0.0 and float(_vo.max()) <= 1.0)
        check("...and in the output dtype", _vo.dtype == torch.float32)
    finally:
        FakeVAE.decode = _real_decode


def test_the_position_may_only_be_written_once_in_the_scene():
    print("\n=== the wrists are placed even when only the scene says where ===")
    _p = ("Inside a van at night. McKenna lies in the back, wrists cuffed behind her back.\n"
          "McKenna: she, 26, dark hair.\n\n"
          "Dan gets into the van and looks at her.\n\n"
          "She lays down on her side.")
    _shots = run_node(_p)[3].split("\n---\n")
    check("the position reaches every shot from the scene alone",
          all("wrists together at the small of the back" in _s for _s in _shots))
    check("...and the lying shot is told what carries her weight",
          "the weight of the body" in _shots[-1])
    # A BEAT THAT MOVES THEM STILL WINS. The scene is only the fallback.
    _moved = run_node("Inside a van. McKenna sits, wrists cuffed behind her back.\n\n"
                      "Dan cuffs her wrists above her head.")[3]
    check("a beat that moves the wrists overrides the scene",
          "above the head" in _moved and "small of the back" not in _moved.split("Both arms")[-1])
    def _weight(_p):
        return ["the weight of the body" in _s for _s in run_node(_p)[3].split("\n---\n")]
    check("the scene alone can say she is lying",
          _weight("Inside a van at night. McKenna lies in the back, wrists cuffed "
                  "behind her back.\n\nDan gets into the van and looks at her.\n\n"
                  "He watches her.") == [True, True])
    check("...with a sheet too",
          _weight("Inside a van. McKenna lies in the back, wrists cuffed behind her "
                  "back.\nMcKenna: she, 26, dark hair.\n\nDan gets into the van."
                  "\n\nHe watches her.") == [True, True])
    check("a beat that stands her up ends it, for good",
          _weight("Inside a van. McKenna lies in the back, wrists cuffed behind her "
                  "back.\n\nShe gets to her feet.\n\nDan looks at her.") == [False, False])
    check("nobody in hardware, nothing said",
          _weight("A bedroom. Nora lies on the bed.\n\nShe looks at the ceiling.") == [False])
    check("restrained but nobody said lying, nothing said",
          _weight("A barn. McKenna stands, wrists cuffed behind her back.\n\n"
                  "Dan looks at her.") == [False])
    check("and the beat that lays her down still fires it",
          _weight("Inside a van. McKenna sits in the back, wrists cuffed behind her "
                  "back.\nMcKenna: she, 26, dark hair.\n\nShe lays down on her side.")
          == [True])


def test_finished_shots_are_held_in_half_precision():
    print("\n=== the chain does not crowd the weights out of RAM ===")
    imgs = run_node("A room.\n\nOne.\n\nTwo.\n\nThree.")[0]
    check("what comes out is still float32", imgs.dtype == torch.float32, str(imgs.dtype))
    check("...and still in range",
          float(imgs.min()) >= 0.0 and float(imgs.max()) <= 1.0)
    # Free, not a trade: fp16 resolves far finer than the 8 bits the output has.
    x = torch.rand(100000)
    err = (x - x.half().float()).abs().max().item()
    check(f"fp16 error {err:.1e} is inside one 8-bit step {1 / 255:.1e}", err < 1 / 255)
    off = run_node("A room.\n\nOne.\n\nTwo.", cleanup_between_shots=False)[0]
    check("a stale cleanup_between_shots=False still returns float32", off.dtype == torch.float32, str(off.dtype))
    check("no flag at all -> float32, as before", S._image_out_dtype() == torch.float32)
    _mm.intermediate_dtype = lambda: torch.float16
    try:
        check("--fp16-intermediates -> the chain follows it",
              S._image_out_dtype() == torch.float16)
        _h = run_node("A room.\n\nOne.\n\nTwo.\n\nThree.")[0]
        check("...and what comes out really is float16", _h.dtype == torch.float16,
              str(_h.dtype))
        check("...still in range", float(_h.min()) >= 0.0 and float(_h.max()) <= 1.0)
        _o = run_node("A room.\n\nOne.\n\nTwo.", cleanup_between_shots=False)[0]
        check("...stale cleanup flag too", _o.dtype == torch.float16, str(_o.dtype))
        _pair = run_node("A room.\n\nOne.\n\nTwo.")
        _w = _pair[1]["waveform"] if isinstance(_pair[1], dict) else _pair[1]
        check("images follow the flag, audio does not",
              _pair[0].dtype == torch.float16 and _w.dtype == torch.float32,
              f"{_pair[0].dtype}/{_w.dtype}")
        _mm.intermediate_dtype = lambda: torch.float32
        check("flag off -> float32 again", S._image_out_dtype() == torch.float32)
        check("...byte for byte what it returned before",
              run_node("A room.\n\nOne.\n\nTwo.")[0].dtype == torch.float32)
    finally:
        del _mm.intermediate_dtype


class DriftVAE(FakeVAE):
    """A VAE whose decode carries the grade of the last frame it was handed.

    THE STOCK FAKE CANNOT SHOW THIS BUG AND THAT IS WHY IT REACHED A USER. FakeVAE.decode
    returns fresh torch.rand, independent of anything encoded, so every shot's output is
    statistically identical by construction and a compounding grade defect is invisible to
    every test in this file. Closing the loop is the whole fixture: what the keyframe looked
    like decides what comes back, and each pass expands contrast about mid grey the way a
    4-step distill at cfg 1 does.

    Keeps FakeVAE's frame-count formula, because the chain preallocates from sum(lens) and
    a shot that decodes to the wrong length sends the whole suite down the overflow path."""

    GAIN = 1.06

    def __init__(self):
        super().__init__()
        self.seen = []           # (mean, std, clipped fraction) of every frame encoded
        self._last = None

    def encode(self, image):
        x = image.float()
        if x.dim() == 4:
            x = x[0]
        if x.dim() == 3 and int(x.shape[-1]) >= 3:
            self.seen.append((float(x.mean()), float(x[..., :3].std()),
                              float(((x <= 0.0) | (x >= 1.0)).float().mean())))
            self._last = x[..., :3].clone()
        return super().encode(image)

    def decode(self, latent):
        t = latent.shape[2] if latent.ndim == 5 else 1
        n = max(1, (t - 2) // 5 * 17 + 5)
        base = torch.rand(H, W, 3) if self._last is None else self._last
        burned = ((base - 0.5) * self.GAIN + 0.5).clamp(0.0, 1.0)
        return burned.unsqueeze(0).repeat(n, 1, 1, 1).to(_vae_out_dtype())


def test_the_chain_does_not_burn_in():
    print("\n=== the chain does not cook itself ===")
    P = "\n\n".join(["A kitchen at night."]
                     + [f"She takes a step to the left. Beat {i}." for i in range(1, 8)])

    off = DriftVAE()
    info_off = run_node(P, vae=off, hold_levels=0.0)[2]
    on = DriftVAE()
    info_on = run_node(P, vae=on, hold_levels=1.0)[2]

    s_off = [s for _, s, _ in off.seen]
    s_on = [s for _, s, _ in on.seen]
    check("the fixture really does cook the chain with the correction off",
          len(s_off) >= 3 and s_off[-1] > s_off[0] * 1.10, f"{[round(v, 4) for v in s_off]}")
    check("...and hold_levels holds it down",
          s_on[-1] < s_off[-1], f"on={[round(v, 4) for v in s_on]} off={[round(v, 4) for v in s_off]}")
    grow_off = s_off[-1] / s_off[0] if s_off[0] else 0.0
    grow_on = s_on[-1] / s_on[0] if s_on[0] else 0.0
    check("...by most of the growth, not a sliver of it",
          grow_on < 1.0 + (grow_off - 1.0) * 0.6, f"off x{grow_off:.3f} on x{grow_on:.3f}")
    c_off = [c for _, _, c in off.seen]
    c_on = [c for _, _, c in on.seen]
    check("clipping does not keep widening either",
          max(c_on) <= max(c_off) + 1e-6,
          f"on={max(c_on):.4f} off={max(c_off):.4f}")

    check("the contrast trend is reported at all", "contrast per shot" in info_off,
          info_off[-400:])
    check("...and a climb is named as cooking", "COOKING" in info_off, info_off[-400:])
    check("hold_levels says what it measured", "hold_levels:" in info_on, info_on[-400:])
    check("...and off it never claims to have corrected anything",
          "took it back out" not in info_off, info_off[-300:])

    # pure functions, no render
    lv = S.HandoffLevels()
    flat = torch.full((H, W, 3), 0.5)
    check("a flat frame is refused rather than divided by",
          lv.observe(flat, flat) is False)
    check("nothing measured means nothing applied",
          S.HandoffLevels().gains(1.0) == (None, None))
    g = S.HandoffLevels()
    lo = torch.rand(H, W, 3) * 0.4 + 0.3
    hi = ((lo - 0.5) * 1.08 + 0.5).clamp(0.0, 1.0)
    for _ in range(4):
        g.observe(lo, hi, hi, hi)
    gain, off_v = g.gains(1.0)
    check("a measured expansion comes back as a gain below 1",
          gain is not None and float(gain.max()) < 1.0, f"{gain}")
    check("...and strength 0 is off even with a measurement in hand",
          g.gains(0.0) == (None, None))
    rep = S.detail_report([(0.10, 0.08), (0.12, 0.10), (0.14, 0.13)])
    check("a rising chain is no longer called 'not softening'",
          "not softening" not in rep, rep)
    check("...the rise is named as cooking instead", "COOKING" in rep, rep)


class CookVAE(DriftVAE):
    """A VAE whose shot reproduces the frame it was handed EXACTLY, then cooks over the take.

    DriftVAE burns every frame of a shot by the same amount -- burn at the CUT, the one
    kind the first correction measured. This reproduces frame one faithfully and expands
    contrast frame by frame to the last, which is how the burn kept arriving after that
    correction was in: the cut measures nothing, and the last frame -- the handoff --
    goes out cooked for the next shot to reproduce faithfully and cook again."""

    GAIN = 1.08

    def decode(self, latent):
        t = latent.shape[2] if latent.ndim == 5 else 1
        n = max(1, (t - 2) // 5 * 17 + 5)
        base = (torch.rand(H, W, 3) * 0.5 + 0.25) if self._last is None else self._last
        g = (self.GAIN ** torch.linspace(0.0, 1.0, n)).view(n, 1, 1, 1)
        return ((base.unsqueeze(0) - 0.5) * g + 0.5).clamp(0.0, 1.0).to(_vae_out_dtype())


def test_the_take_does_not_cook_the_chain():
    """REPORTED after hold_levels was in: the scene still burning itself in, every beat."""
    print("\n=== a shot that cooks over its length does not hand the burn on ===")
    beats = [f"She takes a step to the left. Beat {i}." for i in range(1, 8)]
    P = "\n\n".join(["A kitchen at night."] + beats)
    off, on = CookVAE(), CookVAE()
    run_node(P, vae=off, hold_levels=0.0)
    out = run_node(P, vae=on, hold_levels=1.0)
    s_off = [s for _, s, _ in off.seen]
    s_on = [s for _, s, _ in on.seen]
    check("the fixture cooks the chain inside the take with the correction off",
          len(s_off) >= 3 and s_off[-1] > s_off[0] * 1.15, f"{[round(v, 4) for v in s_off]}")
    check("...and the handoff no longer carries it on",
          s_on[-1] < s_on[0] * 1.03, f"on={[round(v, 4) for v in s_on]}")
    v = out[0].float()
    check("...and the video's own last frame is not burned against its first",
          float(v[-1].std()) < float(v[0].std()) * 1.03,
          f"{float(v[0].std()):.4f} -> {float(v[-1].std()):.4f}")
    # A beat that changes the light keeps what its take did.
    lamp = list(beats)
    lamp[2] = "She switches off the lamp."
    kept = CookVAE()
    run_node("\n\n".join(["A kitchen at night."] + lamp), vae=kept, hold_levels=1.0)
    s_k = [s for _, s, _ in kept.seen]
    check("a shot that switches the lamp off is not graded back over its take",
          len(s_k) == len(s_on) and s_k[3] > s_on[3] * 1.04,
          f"lamp={[round(x, 4) for x in s_k]} plain={[round(x, 4) for x in s_on]}")

    # pure functions, no render
    check("a light changing is read as the author's",
          all(S.light_changes(b) for b in ("She switches off the lamp.", "The lights go out.",
                                           "He draws the curtains.", "The sun sets.",
                                           "She turns the lamp on.")))
    check("...and a light that is merely there is not",
          not any(S.light_changes(b) for b in ("A lamp glows on the desk.",
                                               "She lies in the warm lamp light.",
                                               "He sits by the window.")))
    first = torch.rand(H, W, 3) * 0.5 + 0.25
    last = ((first - 0.5) * 1.10 + 0.5).clamp(0.0, 1.0)
    sg = S.shot_grade(None, first, last, 1.0)
    check("a take that cooked is graded back at its end and not at its start",
          sg is not None and float(sg[0][0].abs().max()) < 1e-6
          and float(sg[1][0].max()) < 0.0, f"{sg}")
    check("...not when the beat changed the light itself",
          S.shot_grade(None, first, last, 1.0, own_change=True) is None)
    check("...nor for a change too large to be cooking",
          S.shot_grade(None, first, first * 0.4, 1.0) is None)
    check("strength 0 is off", S.shot_grade(None, first, last, 0.0) is None)
    fr = torch.stack([first, last]).clone()
    S.grade_frames(fr, *sg)
    check("grading puts the last frame back at the first frame's contrast",
          abs(float(fr[1].std()) - float(first.std())) < 0.01 * float(first.std())
          and torch.equal(fr[0], first), f"{float(first.std()):.4f} {float(fr[1].std()):.4f}")


def test_detail_trend():
    print("\n=== the chain is measured for softening ===")
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
    for _s, _who in ((s1, "McKenna"), (s2, "Dan")):
        check("the hands are named in the shot (%s)" % _who, _who in _s)
    _unstaged = run_node(two + "remove: shorts\nMcKenna stands by the door.",
                         plan_only=True)[3]
    check("a removal the beat does not stage still names the hands",
          "McKenna takes the blue jean shorts off during this shot" in _unstaged)
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
    check("...and says no more than that",
          "room tone either side" not in named and "said once" not in named)
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
    solo = run_node('A room.\n\nKate waits alone. "Now?"', plan_only=True,
                    character_memory=mem)[3]
    check("one person alone gets no mouth guard",
          "One voice in the shot" not in solo and "Only Kate speaks" not in solo)
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
    class OtherVae(EdgyVae):
        def encode(self, x):
            return super().encode(x) + 1.0

    other = S._silent_audio_latent(OtherVae(), 226, S.H3_FPS)
    check("changing audio VAE rebuilds the cached silence",
          float(other.mean()) > float(lat.mean()) + 0.5)
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


def test_silence_pins_the_generated_audio():
    print("\n=== silence pins the generated audio target ===")
    video = torch.randn((1, 24, 4, 3, 5))
    audio = torch.randn((1, 32, 2, 40))
    silence = torch.zeros_like(audio) + 0.125

    latent = {"samples": FakeNested((video, audio))}
    check("a full-shot silence mask is installed",
          S._pin_audio_silence(latent, silence, None))
    got_video, got_audio = latent["samples"].unbind()
    video_mask, audio_mask = latent["noise_mask"].unbind()
    check("the target audio is encoded silence", torch.equal(got_audio, silence))
    check("video remains free to denoise", bool(torch.all(video_mask == 1)))
    check("the whole silent shot is locked", bool(torch.all(audio_mask == 0)))
    check("the video target is untouched", torch.equal(got_video, video))

    latent = {"samples": FakeNested((video, audio))}
    check("a dialogue lead-in mask is installed",
          S._pin_audio_silence(latent, silence, 20))
    _, audio_mask = latent["noise_mask"].unbind()
    check("the first half-second is locked", bool(torch.all(audio_mask[..., :20] == 0)))
    check("speech can denoise after the lead-in", bool(torch.all(audio_mask[..., 20:] == 1)))

    latent = {"samples": FakeNested((video, audio))}
    check("a lead-in and a tail install together", S._pin_audio_silence(latent, silence, 20, 8))
    _, audio_mask = latent["noise_mask"].unbind()
    check("the opening is locked", bool(torch.all(audio_mask[..., :20] == 0)))
    check("the close is locked", bool(torch.all(audio_mask[..., -8:] == 0)))
    check("the line's span between them is free", bool(torch.all(audio_mask[..., 20:-8] == 1)))

    latent = {"samples": FakeNested((video, audio))}
    check("a tail alone is enough to install", S._pin_audio_silence(latent, silence, 0, 8))
    _, audio_mask = latent["noise_mask"].unbind()
    check("...and it locks only the close",
          bool(torch.all(audio_mask[..., -8:] == 0)) and bool(torch.all(audio_mask[..., :-8] == 1)))

    latent = {"samples": FakeNested((video, audio))}
    check("a tail that would overlap the lead is clipped to what is left",
          S._pin_audio_silence(latent, silence, 30, 20))
    _, audio_mask = latent["noise_mask"].unbind()
    check("...so the lead keeps its 30 and the tail takes the other 10", bool(torch.all(audio_mask == 0)))
    check("nothing to pin is reported, not counted as applied",
          not S._pin_audio_silence({"samples": FakeNested((video, audio))}, silence, 0, 0))

    bad = {"samples": FakeNested((video, audio))}
    check("a mismatched silent latent is rejected",
          not S._pin_audio_silence(bad, silence[..., :-1], None))
    check("a rejected mask does not partially mutate the latent", "noise_mask" not in bad)


def test_a_softened_handoff_is_not_a_keyframe():
    """The removing shot keeps describing the garment when nothing anchors it.

    The scrub applies to the removing shot too, and the only reason that is safe is
    that the shot's KEYFRAME already shows the garment on at the start. Below
    KEYFRAME_SAFE_AUG the handoff stops being a keyframe and rides as an extra
    reference -- it says who somebody is, not what the opening frame holds -- so
    the premise is false and scrubbing took the belt out of the text of the very
    shot that removes it. It was gone a beat early with nothing holding it on."""
    print("\n=== a softened handoff is not a keyframe ===")
    P = ("A room.\n\nMcKenna: she, 22, crop top, chastity belt.\n\n"
         "Dan: he, 30, shirt.\n\nMcKenna walks in.\n\nDan looks at her.\n\n"
         "Dan unlocks the chastity belt.\n\nMcKenna sits down.")
    soft = [x for x in re.split(r"(?=\[Shot )",
                                run_node(P, plan_only=True, ref_noise_aug=0.95)[3])
            if x.strip()]
    check("softened: the removing shot still says it is worn",
          "chastity belt" in soft[2])
    check("...and the next shot does not", "chastity belt" not in soft[3])
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
          "opens in the living room" in sh[1])
    check("...and where it ends", "arrives in the bedroom" in sh[1])
    check("...and what it passes through", "along the hallway" in sh[1])
    check("...and that the walk happens on screen",
          "played out on screen" in sh[1] and "every step in frame" in sh[1])
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
          "The language is English" in sh[1])
    check("...and the silent ones are not",
          "The language is" not in sh[0] and "The language is" not in sh[2])
    check("info names the shots", "told which language it is spoken in" in info)
    # Positively phrased: naming the unwanted language would ask for it.
    check("no language is named but the wanted one",
          not any(w in sh[1] for w in ("not in", "Spanish", "French", "Chinese")))
    def said_in(script_text, which=1):
        s = [x for x in re.split(r"(?=\[Shot )",
                                 run_node(script_text, plan_only=True)[3]) if x.strip()]
        m = re.search(r"The language is ([^.]*)\.", " ".join(s[which].split()))
        return m.group(1) if m else ""

    for _lang, _line in (
            ("Spanish", 'Kate dice: "No sé lo que estás haciendo con eso."'),
            ("French", 'Kate dit: "Je ne sais pas ce que vous faites."'),
            ("German", 'Kate sagt: "Ich weiss nicht was du da machst."'),
            ("Russian", 'Kate says: "Не знаю '
                        'что ты дела'
                        'ешь."'),
            ("Japanese", 'Kate says: "何をしているの？"'),
    ):
        got = said_in("A room.\n\nKate waits.\n\n" + _line)
        check(f"a line in {_lang} is called {_lang}", got == _lang, got)
    short = said_in('Una habitación.\n\nKate espera.\n\n'
                    'Kate dice: "No sé lo que estás haciendo con eso."\n\n'
                    'Kate dice: "Sí."', which=2)
    check("...and the short line follows the script", short == "Spanish", short)
    for _line in ('Klaus approaches her and says in German '
                  '"Oh, du siehst heute toll aus, Schatz!"',
                  'Klaus angrily stares at her and says in German '
                  '"OK, das reicht mir jetzt wirklich!"'):
        _got = said_in("A kitchen.\n\nKlaus waits.\n\n" + _line)
        check(f"one function word, but the author said German: {_line[42:62]!r}",
              _got == "German", _got)
    _mixed = ('A kitchen.\n\nMara says: "Where have you been all evening, exactly?"\n\n'
              'Klaus says in German "OK, das reicht mir jetzt wirklich!"\n\n'
              'Mara says: "That is not what I asked you and you know it."')
    # s[0] is Shot 1: the scene paragraph is not a shot, so the German line is s[1].
    check("the German line in an English script is German",
          said_in(_mixed, 1) == "German", said_in(_mixed, 1))
    check("...and the English ones stay English",
          said_in(_mixed, 0) == "English" and said_in(_mixed, 2) == "English",
          f"{said_in(_mixed, 0)}/{said_in(_mixed, 2)}")
    for _t in ("The German soldier looks up.", "A German car pulls in.",
               "She pets the German shepherd.", "Klaus is German.",
               "The Spanish tiles are cold.", "A French window stands open."):
        check(f"not a language: {_t[:34]!r}", not S.engine.language_named(_t))
    for _t, _w in (("She speaks German to him.", "German"),
                   ("He replies in broken German.", "German"),
                   ("She switches to Spanish.", "Spanish"),
                   ("He asked in Russian where the key was.", "Russian")):
        check(f"named: {_t[:34]!r}", S.engine.language_named(_t) == _w)
    check("the vote is not overruled by a stray mention",
          S.engine.language_of("Je ne sais pas ce que vous faites",
                               named="German") == "French")
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
          "Dana is lying down" not in sh[1])
    check("...nor the shot after it", "Dana is lying down" not in sh[2])
    # And a pose that is NOT contradicted is still held.
    P2 = ("A nursery.\n\nDana and McKenna sit down on the sofa.\n\n"
          "Dana and McKenna look at the window.\n\nDana and McKenna wait.")
    sh2 = [x for x in re.split(r"(?=\[Shot )",
                               run_node(P2, plan_only=True, character_memory=mem)[3])
           if x.strip()]
    check("a pose nothing contradicts is still held",
          bool(re.search(r"\b(?:is|are) sitting", sh2[1])), sh2[1][-160:])


def test_pacing_reaches_the_thin_shots():
    """END TO END: the pacing clause lands on shots that outlast their beat, and
    on no others."""
    print("\n=== pacing reaches the thin shots ===")
    mem = "Kate: she, 30, coat."
    P = ("A room.\n\nKate waits.\n\n"
         "Kate walks in, drops her bag, takes off her coat, hangs it up and "
         "crosses the room.\n\nKate looks at the window.")
    out = run_node(P, plan_only=True, character_memory=mem,
                   shot_length="fixed", shot_seconds=10.0)
    info, script = out[2], out[3]
    paced = [i for i, b in enumerate(script.split("---"), 1)
             if "even pace across the whole shot" in b]
    check("the thin shots are paced", paced == [1, 3], str(paced))
    check("...and the full one is not", 2 not in paced)
    check("info names them", "stage less than their length" in info)
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
         "Dana opens the bag and looks at McKenna.")
    sh = [x for x in re.split(r"(?=\[Shot )",
                              run_node(P, plan_only=True, character_memory=mem)[3])
          if x.strip()]
    check("the shorts are still on while she is only told",
          re.search(r"McKenna: she, 22,[^.]*shorts", sh[0]) is not None)
    check("...and nothing is taken off yet",
          "off during this shot" not in sh[0])
    check("...and nobody is lying down yet", "still lying down" not in sh[0])
    # The beat that actually does it.
    check("the next beat takes them off", "off during this shot" in sh[1])
    # ...and the pose it sets is held afterwards, for her only.
    check("the pose is held after that", "McKenna is lying down" in sh[2])
    check("...and the speaker is not lying down", "Dana is still lying" not in sh[2])
    idle = ("A public bathroom.\n\nMcKenna and Dana walk in. Dana says to McKenna: "
            '"Take off your shorts and lie down on the change table."\n\n'
            "McKenna and Dana wait.")
    idle_sh = [x for x in re.split(r"(?=\[Shot )",
                                   run_node(idle, plan_only=True,
                                            character_memory=mem)[3]) if x.strip()]
    check("the listener is given something to do",
          "McKenna listens" in sh[0])
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
    check("the move begins where the scene says", "opens in the living room" in sh[1])
    check("...and ends in the new room", "arrives in the bedroom" in sh[1])
    # ...and the shots after it stay there.
    check("the next shot is in the new room", "takes place in the bedroom" in sh[2])
    check("...and so is the one after", "takes place in the bedroom" in sh[3])
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
    P = ("A carpeted living room, lamps low.\n\n"
         "Kate and Sam sit on the sofa and she says: \"Come with me.\"\n\n"
         "Kate walks him down the hallway to the tiled bathroom.\n\n"
         "They stand by the sink and she says: \"Wait here.\"\n\n"
         "Kate turns the tap on.")
    out = run_node(P, plan_only=True, character_memory=mem)
    info, script = out[2], out[3]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("the journey has both ends", "opens in the living room" in sh[1]
          and "arrives in the bathroom" in sh[1])
    check("the next shot is in the new room", "takes place in the bathroom" in sh[2])
    check("...and so is the one after", "takes place in the bathroom" in sh[3])
    # The acoustic goes with them.
    check("the room tone follows", "tiled walls ringing" in sh[2])
    check("...and the ambient bed", "water moving in the pipes" in sh[2])
    check("the room they left keeps its own", "a soft room with little echo" in sh[0])
    check("...and is not given the new one", "tiled walls ringing" not in sh[0])
    check("info names the shots", "sound followed them into the new room" in info)
    # The switch is gone: the room hold and the acoustic now always travel together.
    quiet = run_node(P, plan_only=True, character_memory=mem, auto_sound=False)[3]
    check("a stale auto_sound=False keeps the room hold",
          "takes place in the bathroom" in quiet)
    check("...and no longer drops the acoustic", "tiled walls ringing" in quiet)


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
    check("the sound clause is what gives way", re.search(r"dropped for room[^|]*sound",
                                                          info) is not None)
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
    check("...and no journey starts in one", "opens in the hall" not in script)
    check("...and the lens gets no cathedral", "a large room with a long tail" not in script)
    # The anchor itself is still carried on every shot -- that is what it is for.
    check("the anchor is on every shot",
          all("Shot on 35mm" in x for x in sh), f"{len(sh)} shots")
    check("...once per shot", all(x.count("Shot on 35mm") == 1 for x in sh))
    # The journey itself still works; it simply has no invented origin.
    check("the move still reaches the bedroom", "arrives in the bedroom" in sh[1])
    # ...and a scene paragraph that DOES name a room still seeds it.
    named = run_node("A living room.\n\nKate waits.\n\n"
                     "Kate walks him down the hallway to the bedroom.",
                     plan_only=True, character_memory=mem)[3]
    check("a real scene room still seeds the origin", "opens in the living room" in named)
    located = run_node(P, plan_only=True, character_memory=mem,
                       anchor="A carpeted living room. Shot on 35mm, "
                              "shallow depth of field, handheld.")[3]
    check("a location IN the anchor still seeds the origin",
          "opens in the living room" in located)
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
    check("the next shot wears them", "Kate is wearing the blue shorts." in sh[5])
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
    off = run_node("A changing room.\n\nKate stands by the bench.\n\n"
                   "Kate takes off her shorts.\n\nKate walks to the door.\n\n"
                   "Kate is at the door.\nadd: her blue shorts\n\n"
                   "Kate opens the door.",
                   plan_only=True, character_memory=mem)[3]
    osh = [x for x in re.split(r"(?=\[Shot )", off) if x.strip()]
    check("a garment back with no dressing staged gets no clause",
          "off the body as the shot opens" not in off)
    check("...and is described as worn in that shot", "Kate is wearing the blue shorts." in osh[3])


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
          all("holding the neck fast at the wall" in s for s in shots[1:]),
          f"{[('holding the neck' in s) for s in shots]}")
    check("no shot claims the wrists", not any("the wrists" in s for s in shots))
    check("the hardware stays fastened in every later shot",
          all("closed and fastened" in s for s in shots[1:]))
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
         "Ana twists her wrists in the cuffs.\n\n"
         "Ana looks at the door.")
    script = run_node(P, plan_only=True,
                      character_memory="Ana: she, 28.\nGuard: he, 40.")[3]
    shots = [b.split("]", 1)[1] for b in script.split("[Shot ")[1:]]
    check("the cuffs are named in shot 2", "cuffs" in shots[1], shots[1][:220])
    check("...and the collar with them", "collar" in shots[1])
    check("both stay fastened in every later shot",
          all("closed and fastened" in x for x in shots[1:]))
    check("no neck is behind a back",
          not any("neck behind the back" in x for x in shots))
    # A limb position is a POSE sentence of its own now.
    check("the wrists take the limb position",
          all("wrists together at the small of the back" in x
              for x in shots[1:]))
    check("the collar takes the fixed point",
          all("fast at the wall" in x for x in shots[1:]))
    # "Ana looks at the door" must not relocate the camera into a door.
    check("a door does not move the shot",
          not any("in the door," in x for x in shots))
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
          "played out on screen" in sh[1], sh[1][:220])
    check("...starting where she was", "opens in the living room" in sh[1])
    check("...and ending where she goes", "arrives in the bedroom" in sh[1])
    check("the next shot is in the bedroom",
          "takes place in the bedroom" in sh[2], sh[2][:220])
    # The shot before the journey is untouched.
    check("the shot before it says nothing",
          "played out on screen" not in sh[0] and "takes place in the" not in sh[0])


NEGATION = re.compile(
    r"\b(?:not|never|no|none|nobody|nothing|without|cannot|can't|won't|"
    r"doesn't|isn't|aren't)\b", re.I)


def test_no_guard_sentence_carries_a_negation():
    """At cfg 1 H3 evaluates NO negative prompt, so a negation in the positive
    prompt is just the thing it names. This file says so in eleven places -- "no
    leggings" would be read as leggings -- and then shipped, in the clause
    written to prevent a cut:

        one continuous move, not a cut

    ...which puts the word CUT in the prompt of the only shot that must not cut.
    Reported twice as an instant cut across a house, the second time after the
    destination reader was fixed and the clause was demonstrably in the prompt.
    The wording was the bug, and no amount of reading the shot text catches it,
    because the clause looks exactly like what you meant.

    Two more went with it: "in the {room}, NOT the room the scene text names"
    pointed at the room being overridden, and "fully removed and NO LONGER on the
    body" names the body it is clearing.

    This sweeps what the node actually SENDS. A negation in a comment is fine and
    there are hundreds; one in an emitted sentence is a bug by construction."""
    print("\n=== no guard sentence carries a negation ===")
    mem = ("Ana: she, 28, blue coat, wool scarf, grey shirt, handcuffs.\n"
           "Dan: he, 40, uniform.")
    scenes = [
        "A living room.\n\nAna stands by the sofa.\n\n"
        "Ana walks into the bedroom.\n\nAna looks around.",
        "A cell.\n\nDan handcuffs Ana's wrists behind her back and chains her "
        "to the wall.\n\nAna pulls at the chain.\n\nDan walks out.",
        "A hallway.\n\nAna takes off her coat and hangs it up.\n\n"
        'Ana sits down.\n\nAna says: "Where is he?"',
        "A yard.\n\nRain on the corrugated roof.\n\nAna runs for the gate."
        "\n\nAna falls.",
        "A room.\n\nAna pulls her scarf aside.\n\nAna puts the scarf back on."
        "\n\nAna stands up.",
    ]
    scenes = [(sc, mem) for sc in scenes] + [
        ("A room.\n\nAna stands.\n\nAna takes off her jeans.\n\n"
         "Ana takes off the chastity belt.\n\nAna sits.",
         "Ana: she, 28, blue jeans, a steel chastity belt."),
    ]
    bad = {}
    for sc, sheet in scenes:
        script = run_node(sc, plan_only=True, character_memory=sheet)[3]
        beats = set(sc.split("\n\n"))
        for block in script.split("[Shot ")[1:]:
            body = block.split("]", 1)[1]
            for sent in re.split(r"(?<=[.!?])\s+", body):
                sent = sent.strip()
                if not sent or any(sent in b for b in beats):
                    continue
                if NEGATION.search(sent) and "<d>" not in sent:
                    bad[sent[:90]] = True
    check("every guard sentence is positively phrased: "
          + ("; ".join(sorted(bad)[:2]) or "all clear"), not bad)


def test_hardware_waits_for_the_beat_that_puts_it_on():
    """Reported: a handcuff on McKenna's arm before she is handcuffed.

    A sheet says WHAT somebody has and never WHEN. "McKenna: she, 27, green
    dress, handcuffs" beside a script that cuffs her in beat 3 put them on from
    shot 1 -- twice over. The engine declared them worn, so shot 1 went out
    saying "The handcuffs stay closed and fastened AS THEY WERE PUT ON" two shots
    before anybody put them on; and the sheet itself, which this node re-stamps
    into every shot, listed handcuffs on her from the opening frame. A described
    item is a drawn item.

    Where the script stages the fastening, the script owns the moment. The sheet
    still supplies the description -- it just does not start the clock."""
    print("\n=== hardware waits for its beat ===")
    P = ("A kitchen.\n\n"
         "McKenna stands by the counter.\n\n"
         "McKenna turns to face Dana.\n\n"
         "Dana handcuffs McKenna's wrists behind her back.\n\n"
         "McKenna pulls at the cuffs.")
    mem = "Dana: she, 31, blue jacket.\nMcKenna: she, 27, green dress, handcuffs."
    sh = [b.split("]", 1)[1] for b in
          run_node(P, plan_only=True, character_memory=mem)[3].split("[Shot ")[1:]]
    check("shot 1 keeps the author's sheet line",
          "handcuffs" in sh[0], sh[0][:200])
    check("...but claims nothing is fastened yet",
          "closed and fastened" not in sh[0], sh[0][:200])
    check("...and shot 2 likewise",
          "handcuffs" in sh[1] and "closed and fastened" not in sh[1], sh[1][:200])
    check("the applying shot has them", "cuff" in sh[2].lower())
    check("...with both ends of the change",
          "off the body at the first frame" in sh[2] or "open and off" in sh[2],
          sh[2][:240])
    check("and they stay after", "cuff" in sh[3].lower())
    check("...held, not re-applied",
          "closed and fastened" in sh[3] and "first frame" not in sh[3], sh[3][:240])

    worn = [b.split("]", 1)[1] for b in
            run_node("A cell.\n\nKate sits on the bunk.\n\nKate waits.",
                     plan_only=True,
                     character_memory="Kate: she, 30, handcuffs."
                     )[3].split("[Shot ")[1:]]
    # ...and the disagreement is REPORTED, not silently resolved.
    info = run_node(P, plan_only=True,
                    character_memory="Dana: she, 31, blue jacket.\n"
                                     "McKenna: she, 27, green dress, handcuffs.")[2]
    check("the sheet/script clash is reported",
          "already lists it as worn" in info, info[-400:])
    check("...and says the wording is untouched",
          "Your wording is never edited" in info)

    check("already-worn hardware holds from shot 1",
          "handcuffs" in worn[0] and "closed and fastened" in worn[0], worn[0][:220])
    check("...and keeps holding", "closed and fastened" in worn[1])


def test_the_applying_shot_says_where_the_limbs_finish():
    """Reported: "the handcuffs broke in the next beat -- her arms were at her
    sides and not handcuffed behind her back".

    Nothing broke. The going-on clause said what the HARDWARE does across the
    shot -- open at the first frame, closed by the last -- and nothing about the
    body, and the limb position was deliberately withheld there on the grounds
    that the author's own words sit right beside it. They do, but they describe
    the ACT, and the next shot does not inherit the act. It inherits the last
    frame. So the cuffs could close with the arms wherever they happened to be,
    and the following shot opened on a picture of somebody with their arms at
    their sides while its text insisted the wrists were behind the back.

    Text loses to an inherited picture, every time. The frame the next shot
    started from never had them behind her back."""
    print("\n=== the applying shot says where the limbs finish ===")
    P = ("A kitchen.\n\nMcKenna stands by the counter.\n\n"
         "Dana handcuffs McKenna's wrists behind her back.\n\n"
         "McKenna pulls at the cuffs.\n\nMcKenna turns around.")
    sh = [b.split("]", 1)[1] for b in
          run_node(P, plan_only=True,
                   character_memory="Dana: she, 31.\nMcKenna: she, 27, green dress."
                   )[3].split("[Shot ")[1:]]
    check("the applying shot names the end position",
          "By the last frame the wrists are behind the back" in sh[1], sh[1][:300])
    check("...as well as the hardware's two ends",
          "open and off the body at the first frame" in sh[1])
    check("the next shot still holds the position",
          "wrists together at the small of the back" in sh[2], sh[2][:240])
    check("...and the one after that",
          "wrists together at the small of the back" in sh[3])
    # A shot that stages no fastening must not claim a last-frame position.
    check("an ordinary shot says nothing about it",
          "By the last frame" not in sh[0], sh[0][:200])
    # A collar has no limb position, and must not be handed one.
    neck = [b.split("]", 1)[1] for b in
            run_node("A cell.\n\nDan locks a steel collar around Ana's neck.\n\n"
                     "Ana sits down.", plan_only=True,
                     character_memory="Dan: he, 40.\nAna: she, 28."
                     )[3].split("[Shot ")[1:]]
    check("a collar gets no limb position",
          "By the last frame the wrists" not in neck[0], neck[0][:240])


def test_a_collar_in_the_sheet_is_held_like_hardware():
    """Reported: the collar was missing from her neck when she was seen in her
    room. It was not the room -- the collar never latched at all.

    restraint_present needs an ambiguous noun plus a binding verb or a body part
    in the SAME line, and a sheet entry reading "green dress, steel collar" has
    neither. So no hold ever fired for it in any shot, and hardware nobody holds
    is hardware the model drops -- it just took a change of room for it to show.

    Bare "collar" stays ambiguous, because a shirt has one. The MATERIAL settles
    it: a shirt's collar is stiff or starched, never steel, and it does not
    lock."""
    print("\n=== a collar in the sheet is held like hardware ===")
    for line in ("McKenna: she, 27, green dress, steel collar.",
                 "McKenna: 27, a leather collar.",
                 "McKenna: 27, a locked collar.",
                 "McKenna: 27, a collar and a padlock.",
                 "McKenna: she, 27, a collar at her throat."):
        check(f"a restraint: {line[:44]!r}", S.restraint_present(line))
    for line in ("McKenna: 27, white shirt with a stiff collar.",
                 "Dan: 40, uniform with a starched collar.",
                 "Dan: 40, a jacket with a fur collar.",
                 "McKenna: she, 27, green dress."):
        check(f"clothing, not hardware: {line[:40]!r}",
              not S.restraint_present(line))
    # END TO END, and across a change of room, which is where it was noticed.
    sh = [b.split("]", 1)[1] for b in
          run_node("A living room.\n\nMcKenna stands by the sofa.\n\n"
                   "McKenna walks into her bedroom.\n\n"
                   "McKenna sits on the bed.", plan_only=True,
                   character_memory="McKenna: she, 27, green dress, steel collar."
                   )[3].split("[Shot ")[1:]]
    for i, s in enumerate(sh, 1):
        check(f"shot {i} names the collar", "collar" in s.lower(), s[:180])
        check(f"shot {i} holds it", "closed and fastened" in s, s[:180])


def test_an_undergarment_keeps_its_words_and_waits_for_its_picture():
    """The two halves of layering an under-garment, which are NOT the same
    question and were got wrong in both directions before this settled.

      WORDS   stay. Deleting the author's item is what put their own wording out
              of the prompt, reported three times, the last a chastity belt with
              a reference attached to it.
      PICTURE waits. A reference is an instruction to reproduce an image and at
              near-clean ref_noise_aug it outweighs any sentence about what is on
              top of what -- so a picture of the belt, in a shot where the belt is
              under the jeans, draws the belt. Reported as it poking through the
              clothes, one commit after the picture was made to travel.

    The tag goes with the picture rather than the image merely being withheld,
    because refs are routed BY the tags and a tag naming a picture the shot does
    not carry is its own bug.

    Nothing here comes from a beat: the sheet is the only place the belt is
    named, so this fails if the sheet stops carrying it."""
    print("\n=== an undergarment keeps its words, its picture waits ===")
    mem = "Ana: <Picture 1>, she, 28, blue jeans, a chastity belt <Picture 2>."
    P = ("A room.\n\nAna stands by the window.\n\n"
         "Ana takes off her jeans.\n\nAna waits.")
    img = torch.rand(1, H, W, 3)
    sh = [b.split("]", 1)[1] for b in
          run_node(P, plan_only=True, character_memory=mem,
                   ref_image_1=img, ref_image_2=img)[3].split("[Shot ")[1:]]
    covered, bare = sh[0], sh[2]
    sheet_line = [l for l in covered.split(chr(10)) if l.strip().startswith("Ana:")]
    check("the character memory line survives", bool(sheet_line), covered[:200])
    check("...with the rest of it intact",
          bool(sheet_line) and "blue jeans" in sheet_line[0].lower()
          and "28" in sheet_line[0], (sheet_line[0] if sheet_line else covered)[:200])
    check("...and only the belt held back",
          bool(sheet_line) and "chastity belt" not in sheet_line[0].lower(),
          (sheet_line[0] if sheet_line else covered)[:200])
    check("...and the jeans are described as covering it",
          "cover the hips and waist completely" in covered.lower(),
          covered[:260])
    check("...and describes the jeans as covering it",
          "whole, opaque and unbroken" in covered.lower(), covered[:260])
    check("...while its picture waits for the cover to move",
          "<Picture 2>" not in covered, covered[:200])
    check("...and the person's own picture does not",
          "<Picture 1>" in covered, covered[:200])
    check("the uncovered shot has both", "chastity belt" in bare.lower()
          and "<Picture 2>" in bare, bare[:220])
    check("...and stops calling it covered",
          "whole, opaque and unbroken" not in bare.lower(), bare[:220])


def test_an_under_layer_belongs_to_somebody():
    """Reported from a real shot: a scene with Dana in jeans and McKenna in a
    skirt and a chastity belt put

        The chastity belt is worn under the skirt...

    into a shot describing only DANA, who has neither. The belt does not go on
    Dana. A described garment is a drawn garment and it is drawn on whoever is in
    the frame.

    implied_layers was read off the WHOLE sheet at once, so it paired a garment
    with a cover and lost whose they were. A sheet line is one person, so the
    layers are read line by line and the owner kept. Per line the same sheet
    answers correctly -- Dana {}, McKenna {chastity belt: skirt} -- which is what
    made this findable."""
    print("\n=== an under-layer belongs to somebody ===")
    mem = ("Dana: she, 35, black t-shirt, blue jeans.\n"
           "McKenna: she, 27, a skirt, a chastity belt.")
    P = ("A home.\n\nA camera tracks Dana moving around the home.\n\n"
         "McKenna steps out of the back room.\n\nDana looks at her.")
    sh = [b.split("]", 1)[1] for b in
          run_node(P, plan_only=True, character_memory=mem)[3].split("[Shot ")[1:]]
    check("Dana's shot says nothing about a belt",
          "chastity belt" not in sh[0].lower(), sh[0][:220])
    check("...nor about a skirt she does not own",
          "skirt" not in sh[0].lower(), sh[0][:220])
    check("McKenna's shot describes the skirt covering it",
          "skirt covers the hips and waist" in sh[1].lower(), sh[1][:240])
    check("...and Dana's later shot is clean again",
          "chastity belt" not in sh[2].lower(), sh[2][:220])
    # Both in one shot: the belt has to say WHOSE, or it lands on either woman.
    two = run_node("A home.\n\nDana and McKenna stand in the hall.",
                   plan_only=True, character_memory=mem)[3]
    body = two.split("]", 1)[1]
    # Attribution moved to the cover, which is the garment now named.
    check("with both described, the cover is attributed",
          "McKenna's skirt covers" in body, body[:300])
    check("...and the name is not mangled", "Mckenna" not in body, body[:300])
    # One person, and no attribution needed.
    solo = run_node("A home.\n\nMcKenna waits.", plan_only=True,
                    character_memory="McKenna: she, 27, a skirt, a chastity belt.")[3]
    solo_body = solo.split("]", 1)[1]
    check("alone, it is just the skirt",
          "The skirt covers the hips and waist completely" in solo_body,
          solo_body[:240])


def test_the_picture_arrives_when_the_cover_moves():
    """The rule, given directly: "when the skirt has been lifted up to show the
    chastity belt, that's when it should be shown, or when the clothing on top has
    been removed."

    Two things stopped that. LIFTING was not a displacement at all -- lift, raise,
    hoist, gather and bunch were missing from the verb list, so the beat that
    uncovers the belt moved nothing. And the verbs that carry their own direction
    were thrown away for not stating one: "lifts her skirt" says which way by
    saying lift, and the pattern wanted an "up" that nobody writes.

    Then an off-by-one on top of it. The layering read `displaced` before the beat
    had been added to it, so even a recognised displacement only took effect on
    the NEXT shot -- the belt came out from under the skirt one shot late."""
    print("\n=== the picture arrives when the cover moves ===")
    mem = "McKenna: <Picture 1>, she, 27, a skirt, a chastity belt <Picture 2>."
    FACE, BELT = torch.rand(1, H, W, 3), torch.rand(1, H, W, 3)

    def refs(prompt):
        rows, ob = [], S.build_conditioning

        def spy(clip, vae, av, p, *a, **k):
            rows.append(len(k.get("refs") or []))
            return ob(clip, vae, av, p, *a, **k)
        S.build_conditioning = spy
        try:
            run_node(prompt, character_memory=mem, ref_image_1=FACE,
                     ref_image_2=BELT)
        finally:
            S.build_conditioning = ob
        return rows

    for how, beat in (("lifted", "McKenna lifts her skirt."),
                      ("raised", "McKenna raises her skirt."),
                      ("held up", "McKenna holds her skirt up."),
                      ("pulled aside", "McKenna pulls her skirt aside."),
                      ("removed", "McKenna takes off her skirt.")):
        got = refs("A home.\n\nMcKenna waits.\n\n" + beat + "\n\nMcKenna waits.")
        check(f"{how}: face only until the cover moves", got[0] == 1, str(got))
        check(f"{how}: and the belt from THAT shot, not the next",
              got[1] == 2, str(got))
    quiet = refs("A home.\n\nMcKenna waits.\n\nMcKenna sits.\n\nMcKenna stands.")
    check("untouched: the picture never arrives", quiet == [1, 1, 1], str(quiet))
    words = run_node("A home.\n\nMcKenna waits.\n\nMcKenna sits.", plan_only=True,
                     character_memory=mem)[3]
    check("...and the belt waits, in every covered shot",
          words.lower().count("chastity belt") == 0, words[:200])
    # A lift is not everything that moves: a chin is not a garment.
    check("lifting a chin displaces nothing",
          not S.displaced_garments("McKenna lifts her chin.",
                                   "McKenna: she, 27, a skirt, a chastity belt."))


def test_letting_the_cover_fall_puts_it_back():
    """A lifted skirt has to come back down, or it stays lifted for the rest of the
    film and whatever was under it stays on show.

    The only restores recognised were pull/tug/hitch/hike/yank/push with a PRONOUN
    and a direction -- "pulls them back up". What actually gets written is the
    opposite: a lifted skirt is LET FALL, DROPPED, LOWERED, SMOOTHED DOWN,
    STRAIGHTENED or LET GO of, and none of those carries a direction word. Nor
    could any of them name the garment; "lets the skirt fall" matched nothing."""
    print("\n=== letting the cover fall puts it back ===")
    sheet = "McKenna: she, 27, a long grey skirt, a chastity belt."
    for beat in ("McKenna lets the skirt fall.", "McKenna lets her skirt drop.",
                 "McKenna lets go of the skirt.", "McKenna smooths her skirt down.",
                 "McKenna lowers her skirt.", "McKenna straightens her skirt.",
                 "McKenna drops her skirt.", "McKenna puts her skirt back."):
        got = S.restored_garments(beat, sheet)
        check(f"restores: {beat[:38]!r}", got == ["long grey skirt"], str(got))
    for beat in ("McKenna lifts her skirt.", "McKenna lets the door close.",
                 "McKenna waits."):
        check(f"not a restore: {beat[:34]!r}",
              not S.restored_garments(beat, sheet), str(S.restored_garments(beat, sheet)))
    # Pronoun forms, which name nothing.
    for beat in ("McKenna lets it fall.", "McKenna lets them fall.",
                 "McKenna pulls them back up.", "McKenna covers herself up."):
        check(f"pronoun restore: {beat[:34]!r}", S.puts_it_back(beat))
    check("...and an ordinary beat is not one", not S.puts_it_back("McKenna waits."))

    # END TO END: the picture follows the cover, both ways, on the shot that moves it.
    mem = "McKenna: <Picture 1>, she, 27, a long grey skirt, a chastity belt <Picture 2>."
    FACE, BELT = torch.rand(1, H, W, 3), torch.rand(1, H, W, 3)

    def refs(prompt):
        rows, ob = [], S.build_conditioning

        def spy(clip, vae, av, p, *a, **k):
            rows.append(len(k.get("refs") or []))
            return ob(clip, vae, av, p, *a, **k)
        S.build_conditioning = spy
        try:
            run_node(prompt, character_memory=mem, ref_image_1=FACE, ref_image_2=BELT)
        finally:
            S.build_conditioning = ob
        return rows

    got = refs("A home.\n\nMcKenna waits.\n\nMcKenna lifts her skirt.\n\n"
               "McKenna lets it fall.\n\nMcKenna waits.")
    check(f"lifted, then covered again: {got}", got == [1, 2, 1, 1], str(got))
    stays = refs("A home.\n\nMcKenna waits.\n\nMcKenna lifts her skirt.\n\n"
                 "McKenna waits.\n\nMcKenna waits.")
    check(f"...and stays shown if nothing puts it back: {stays}",
          stays == [1, 2, 2, 2], str(stays))
    named = refs("A home.\n\nMcKenna waits.\n\nMcKenna lifts her skirt.\n\n"
                 "McKenna smooths her skirt down.\n\nMcKenna waits.")
    check(f"...and a named restore works too: {named}", named == [1, 2, 1, 1],
          str(named))


def test_somebody_else_can_lift_your_skirt():
    r"""Reported from a real shot: "Dana lifts up McKenna's skirt to check the
    chastity belt" left the belt covered and its picture withheld.

    The rule was right and the parse was not. _DISPLACE took a determiner of
    the/her/his/their/a/an, which matches no POSSESSIVE NAME, and the apostrophe
    is not in the [\w\- ] the garment itself is read with -- so the whole match
    failed and the lift was invisible. One person lifting another's clothing is
    the ordinary way this gets written.

    Same fault as "enters McKenna's bedroom" on travel, in a different reader, so
    both now use the same _DET_POSS rather than a second copy that can drift."""
    print("\n=== somebody else can lift your skirt ===")
    sheet = ("McKenna: she, 22, a chastity belt, a mini-skirt.\n"
             "Dana: she, 35, black t-shirt, blue jeans.")
    for beat in ("Dana lifts up McKenna's skirt to check the chastity belt.",
                 "Dana lifts McKenna's skirt.",
                 "Dana pulls McKenna's skirt aside."):
        got = S.displaced_garments(beat, sheet)
        check(f"displaced: {beat[:44]!r}", bool(got), str(got))
    for beat in ("Dana lets McKenna's skirt fall.",
                 "Dana smooths McKenna's skirt down."):
        check(f"restored: {beat[:44]!r}",
              bool(S.restored_garments(beat, sheet)),
              str(S.restored_garments(beat, sheet)))
    check("a chin is not displaced",
          not S.displaced_garments("Dana lifts McKenna's chin.", sheet))

    mem = ("McKenna: she, <Picture 1>, 22, a chastity belt <Picture 2>, "
           "a mini-skirt.\nDana: she, 35, black t-shirt, blue jeans.")
    FACE, BELT = torch.rand(1, H, W, 3), torch.rand(1, H, W, 3)
    rows, ob = [], S.build_conditioning

    def spy(clip, vae, av, p, *a, **k):
        rows.append((p, len(k.get("refs") or [])))
        return ob(clip, vae, av, p, *a, **k)
    S.build_conditioning = spy
    try:
        run_node("A home.\n\nDana lifts up McKenna's skirt to check the "
                 "chastity belt.\n\nMcKenna waits.", character_memory=mem,
                 ref_image_1=FACE, ref_image_2=BELT)
    finally:
        S.build_conditioning = ob
    check("the lifting shot carries the belt's picture", rows[0][1] == 2,
          str([n for _, n in rows]))
    check("...and does not call the skirt opaque over it",
          "whole, opaque" not in rows[0][0], rows[0][0][-200:])
    check("...and stays up while nothing puts it back",
          rows[1][1] == 2 and "whole, opaque" not in rows[1][0],
          str([n for _, n in rows]))


def test_timing_report():
    print("\n=== the timing breakdown ===")
    P = "A room.\n\nOne.\n\nTwo."
    info = run_node(P)[2]
    for want in ("sampling", "decode", "per shot"):
        check(f"info reports {want}", want in info)
    check("...with a wall-clock total", "rendered" in info and "s --" in info)
    # plan_only does no work, so it must not claim any timings.
    check("plan_only reports no timings",
          " -- sampling " not in run_node(P, plan_only=True)[2])


def test_upscale_paths():
    print("\n=== upscale wiring ===")
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
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    check("the handoff comes from the pre-upscale latent",
          "pre_up[:, :, -n:]" in src and "hand_src = tail" in src)
    check("...and is clamped before it is re-encoded",
          "clamp(0.0, 1.0)" in src)


def _shots_of(result):
    return [sh.split("] ", 1)[1] if "] " in sh else sh for sh in result[3].split("\n---\n")]


def test_the_opening_does_not_name_people_a_shot_leaves_out():
    """An opening that names people rode into every shot unscoped: "Maya and Owen wait
    in a train station." headed a shot the node had cut down to Owen alone, so the
    model was told two people stand there and given one person to draw."""
    print("\n=== the opening paragraph names nobody a shot leaves out ===")
    memory = "Maya: she, 38, green sweater.\nOwen: he, 42, blue shirt."
    shots = _shots_of(run_node("Maya and Owen wait in a train station.\n\n"
                               "Owen checks the departure board.\n\n"
                               "Maya reads a newspaper on a bench.",
                               character_memory=memory, plan_only=True))
    check("two solo shots", len(shots) == 2)
    check("Owen's shot does not name Maya", "Maya" not in shots[0])
    check("Maya's shot does not name Owen", "Owen" not in shots[1])
    check("...and both keep the station", all("train station" in sh for sh in shots))
    # A pronoun after a cut sentence goes with it.
    shots = _shots_of(run_node("Maya sits at her desk in an office. She types a report.\n\n"
                               "Owen knocks and walks in.",
                               character_memory=memory, plan_only=True))
    check("no stray 'She' about an absent Maya", "She types" not in shots[-1]
          and "Maya" not in shots[-1] and "office" in shots[-1])
    # The anchor is scoped the same way.
    shots = _shots_of(run_node("A kitchen.\n\nOwen chops onions.", character_memory=memory,
                               anchor="Maya and Owen are in a bright kitchen.", plan_only=True))
    check("an anchor naming an absent person does not reach his solo shot",
          "Maya" not in shots[-1] and "kitchen" in shots[-1])
    # Control: everyone the opening names is in the shot, so it is untouched.
    shots = _shots_of(run_node("Maya and Owen wait in a train station.\n\n"
                               "They look at the board together.",
                               character_memory=memory, plan_only=True))
    check("an opening about the whole cast is kept word for word",
          "Maya and Owen wait in a train station." in shots[0])


def test_a_sheet_written_first_is_a_sheet():
    """Opening a script with who is in it made the sheet the SCENE: every person
    described in every shot as prose, with no scoping, count or mouth guard."""
    print("\n=== a character sheet written first is a sheet, not the scene ===")
    shots = _shots_of(run_node("Maya: she, 38, green sweater.\nOwen: he, 42, blue shirt.\n\n"
                               "A park.\n\nMaya sits on a bench.\n\nOwen feeds the ducks.",
                               plan_only=True))
    check("the sheet is not a shot of its own", len(shots) == 2)
    check("Owen's shot does not describe Maya", "Maya:" not in shots[1] and "Owen:" in shots[1])
    check("...and counts the one person it describes",
          "There is one person in the shot" in shots[1], shots[1][-160:])
    check("the park is still the scene", all(sh.startswith("A park.") for sh in shots))
    # Control: a heading with a colon is not a person and stays the scene.
    shots = _shots_of(run_node("Interior: a kitchen at night.\n\nMaya: she, 38, green sweater.\n\n"
                               "Maya makes tea.", plan_only=True))
    check("'Interior: ...' stays the scene", shots[0].startswith("Interior: a kitchen at night."))


def test_a_two_word_name_makes_a_sheet():
    """"Mistress Vale:" did not match the sheet-line pattern, so the sheet rendered as
    a shot and every later shot described nobody."""
    print("\n=== a sheet entry with a two-word name is still a sheet ===")
    shots = _shots_of(run_node("A library.\n\nMistress Vale: she, 45, black dress.\n"
                               "Owen: he, 42, blue shirt.\n\nMistress Vale shelves a book.\n\n"
                               "Owen reads at a table.", plan_only=True))
    check("no shot is spent on the sheet", len(shots) == 2)
    check("Mistress Vale is described in her shot", "Mistress Vale: she, 45, black dress." in shots[0])
    check("Owen is described in his", "Owen: he, 42, blue shirt." in shots[1])
    check("'Both women: tired' is still not a sheet line",
          not S.is_character_sheet("Both women: tired"))


def test_one_person_under_two_names_is_described_once():
    """"Maya Brooks" in character_memory and "Maya:" in the prompt were two keys, so
    one woman went into every shot twice, in two different outfits."""
    print("\n=== one person under two forms of her name is described once ===")
    result = run_node("Maya: she, 38, red coat.\n\nA park.\n\nMaya sits on a bench.",
                      character_memory="Maya Brooks: she, 38, green sweater.", plan_only=True)
    shots = _shots_of(result)
    check("one entry for her", all(sh.count("she, 38") == 1 for sh in shots))
    check("character_memory's entry is the one kept", all("green sweater" in sh and "red coat" not in sh
                                                          for sh in shots))
    check("...and the author is told", any(isinstance(part, str) and "described more than once" in part
                                           for part in result))
    # Control: a shared word with a different age is somebody else.
    shots = _shots_of(run_node("A garden.\n\nMay: she, 24, yellow dress.\n\nMay and Aunt May pick apples.",
                               character_memory="Aunt May: she, 60, grey cardigan.", plan_only=True))
    check("May and Aunt May stay two people", "May: she, 24" in shots[-1]
          and "Aunt May: she, 60" in shots[-1] and "two people" in shots[-1])


def test_a_bare_chest_belongs_to_whoever_undressed():
    """Two entries listing a sweater: the first entry was taken as the wearer, so Lena
    taking hers off put the bare chest on Maya, still listed in her own sweater."""
    print("\n=== a bare region goes to the person who undressed, not the first entry ===")
    shots = _shots_of(run_node("A bedroom.\n\nMaya and Lena talk.\n\nLena takes off her sweater.",
                               character_memory="Maya: she, 38, green sweater.\n"
                                                "Lena: she, 30, yellow sweater.", plan_only=True))
    last = shots[-1]
    check("the bare chest is Lena's", "Lena's chest" in last)
    check("...not Maya's", "Maya's chest" not in last)
    check("Maya keeps her sweater", "Maya: she, 38, green sweater." in last)


def test_looking_at_a_place_does_not_go_there():
    """"Maya looks out of the window at the garden" put the shot IN the garden, started
    it fresh, and kept the film there: "Maya pours tea." two shots on was still in it."""
    print("\n=== looking at a place does not move the shot there ===")
    clip = FakeCLIP()
    result = run_node("A kitchen with white tiles and a copper kettle.\n\nMaya slices bread.\n\n"
                      "Maya looks out of the window at the garden.\n\nMaya pours tea.",
                      character_memory="Maya: she, 38, grey cardigan.", clip=clip)
    shots = _shots_of(result)
    check("no shot is relocated to the garden", not any("takes place in the garden" in sh for sh in shots))
    check("the glance keeps the keyframe", [len(items) for p, items in clip.seen if p.strip()] == [0, 1, 1])
    check("the place reader still finds a real presence", S.place_named("Maya sits in the garden.") == "garden")
    check("...and not a pointed-at room", S.place_named("Maya points at the kitchen.") == "")


def test_a_room_the_film_returns_to_is_carried():
    """A room shown earlier and returned to had no picture of itself: the words rebuilt
    it, and the rebuild was a different room."""
    print("\n=== a room the film comes back to is carried from when it was last shown ===")
    memory = "Maya: she, 38, grey cardigan.\nOwen: he, 42, blue shirt."
    def refs(clip):
        return [len(items) for p, items in clip.seen if p.strip()]
    clip = FakeCLIP()
    result = run_node("A small apartment. The living room has a red sofa; the kitchen has white tiles.\n\n"
                      "Maya reads on the red sofa in the living room.\n\nMaya fills the kettle in the kitchen.\n\n"
                      "Maya sits on the sofa in the living room.", character_memory=memory, clip=clip)
    last = _shots_of(result)[-1]
    check("a cut back to the living room carries its picture", refs(clip)[-1] == 1)
    check("...claimed as that room", "is the living room as the film last showed it" in last)
    check("...naming who is in it", "Maya is the person in it." in last)
    check("...and the author is told", "carried a room back" in str(result[2]))
    clip = FakeCLIP()
    result = run_node("A small apartment. The living room has a red sofa; the kitchen has white tiles.\n\n"
                      "Maya reads on the red sofa.\n\nMaya walks into the kitchen and fills the kettle.\n\n"
                      "Maya comes back to the living room and sits on the sofa.", character_memory=memory, clip=clip)
    check("a walk back by the same person carries no second picture of her",
          "carried a room back" not in str(result[2]) and refs(clip)[-1] == 1)
    # A change of clothes since retires the old frame.
    clip = FakeCLIP()
    result = run_node("A small apartment. The living room has a red sofa; the kitchen has white tiles.\n\n"
                      "Maya reads on the red sofa in the living room.\n\nMaya takes off her cardigan in the kitchen.\n\n"
                      "Maya fills the kettle in the kitchen.\n\nMaya sits on the sofa in the living room.",
                      character_memory="Maya: she, 38, grey cardigan over a white blouse.", clip=clip)
    check("a frame from before a costume change is not carried", "carried a room back" not in str(result[2]))


def _entry_in(shot, name):
    m = re.search(re.escape(name) + r": [^.]*\.", shot)
    return m.group(0) if m else ""


def test_a_layer_under_a_removed_garment_stays_on():
    """"long red coat over a grey sweater" was one entry: the coat came off and took the
    sweater with it, and the removal shot called her chest bare."""
    print("\n=== taking off an outer layer leaves the one under it ===")
    shots = _shots_of(run_node("A hallway.\n\nMaya takes off her coat.\n\nMaya checks her phone.",
                               character_memory="Maya: she, 38, long red coat over a grey sweater, "
                                                "black jeans, brown boots.", plan_only=True))
    check("the sweater is still described after the coat", "grey sweater" in _entry_in(shots[1], "Maya"))
    check("...the coat is not", "coat" not in _entry_in(shots[1], "Maya"))
    check("...and no shot calls her chest bare", not any("chest, shoulders and arms are bare" in sh
                                                        for sh in shots))
    shots = _shots_of(run_node("A hallway.\n\nMaya takes off her jacket.\n\nMaya checks her phone.",
                               character_memory="Maya: she, 38, denim jacket with rolled sleeves and a hood, "
                                                "white t-shirt, black jeans.", plan_only=True))
    check("a jacket's own hood goes with it", "hood" not in _entry_in(shots[1], "Maya")
          and "white t-shirt" in _entry_in(shots[1], "Maya"))


def test_how_a_garment_comes_off_is_read_right():
    """"sheds her coat" removed nothing; "strips off her coat" stripped her naked;
    "unzips his jacket" took the jacket off."""
    print("\n=== sheds, strips off and unzips mean what they say ===")
    memory = "Maya: she, 38, long red coat over a grey sweater, black jeans, brown boots."
    shots = _shots_of(run_node("A hallway.\n\nMaya sheds her coat.\n\nMaya checks her phone.",
                               character_memory=memory, plan_only=True))
    check("'sheds her coat' takes the coat off", "coat" not in _entry_in(shots[1], "Maya"))
    shots = _shots_of(run_node("A hallway.\n\nMaya strips off her coat.\n\nMaya checks her phone.",
                               character_memory=memory, plan_only=True))
    check("'strips off her coat' takes only the coat", _entry_in(shots[1], "Maya")
          == "Maya: she, 38, a grey sweater, black jeans, brown boots.")
    shots = _shots_of(run_node("An office.\n\nOwen unzips his jacket.\n\nOwen sits at his desk.",
                               character_memory="Owen: he, 42, navy jacket over a white shirt, grey trousers.",
                               plan_only=True))
    check("an unzipped jacket stays on", "navy jacket" in _entry_in(shots[1], "Owen"))
    check("...and the next shot keeps it open", "jacket open" in shots[1])
    check("'unbuttons his shirt' opens the shirt, not the jacket over it",
          S.engine.displaced_garments("Owen unbuttons his shirt.",
                                      "Owen: he, 42, navy jacket over a white shirt.") == [("white shirt", "open")])


def test_a_garment_put_back_on_in_prose_comes_back():
    """"puts her coat back on" did nothing without an add: line, so the coat never came
    back into her description."""
    print("\n=== a garment put back on in prose is described again ===")
    shots = _shots_of(run_node("A hallway.\n\nMaya takes off her coat.\n\nMaya checks her phone.\n\n"
                               "Maya puts her coat back on.\n\nMaya opens the front door.",
                               character_memory="Maya: she, 38, long red coat over a grey sweater, "
                                                "black jeans, brown boots.", plan_only=True))
    check("the put-back shot stages it going on", "off the body as the shot opens" in shots[2])
    check("the next shot has her wearing it", "Maya is wearing the long red coat." in shots[3])


def test_a_garment_called_by_its_familys_word_is_found():
    """"takes off her shoes" beside "brown leather boots" named nothing on the sheet,
    so the boots stayed on in the text while the beat took them off."""
    print("\n=== 'shoes' for boots, 'sweatshirt' for a hoodie ===")
    sheet = "Maya: she, 38, grey hoodie, black jeans, brown leather boots."
    check("'shoes' means the boots", S.infer_removals("Maya takes off her shoes.", sheet) == ["boots"])
    check("'sweatshirt' means the hoodie", S.infer_removals("Maya pulls off her sweatshirt.", sheet) == ["hoodie"])
    check("two candidates is not a guess",
          S.infer_removals("Maya takes off her top.", "Maya: she, 38, white shirt, blue blouse.") == [])


def _described_in(shot, names):
    return [n for n in names if f"{n}:" in shot]


def test_a_pronoun_in_a_description_is_not_the_declared_one():
    """"Owen: he, ... carries her photo" was read as "she": the groups were tried in a
    fixed order and "her" is in his description."""
    print("\n=== the declared pronoun wins over one in the description ===")
    check("a declared 'he' with 'her' in the description is he",
          S.sheet_pronoun("Owen: he, 42, blue shirt, carries her photo in his wallet.") == "he")
    check("a declared 'she' with 'his' in the description is she",
          S.sheet_pronoun("Maya: she, 38, wears his old watch.") == "she")
    check("with nothing declared, a person noun says it",
          S.sheet_pronoun("Maya: 38, a woman with red hair.") == "she"
          and S.sheet_pronoun("Owen: 42, a tall man, blue shirt.") == "he")
    check("...but a possessive is not the person", S.sheet_pronoun("Kit: 25, brother's jacket.") is None)
    shots = _shots_of(run_node("A kitchen.\n\nMaya pours coffee.\n\nHe sits down.\n\nShe smiles.",
                               character_memory="Maya: 38, a woman with red hair, green sweater.\n"
                                                "Owen: 42, a tall man, blue shirt.", plan_only=True))
    check("'He sits down' reaches the man the sheet calls a man", _described_in(shots[1], ["Maya", "Owen"]) == ["Owen"])
    check("...and 'She smiles' the woman", _described_in(shots[2], ["Maya", "Owen"]) == ["Maya"])


def test_a_name_used_as_a_word_stages_nobody():
    """"Will he come?" and "May I come in?" put Will and May into shots about waiting for
    them."""
    print("\n=== 'Will he come?' does not put Will in the shot ===")
    memory = "Will: he, 50, grey coat.\nMaya: she, 38, green sweater."
    shots = _shots_of(run_node("A porch.\n\nMaya waits by the door. Will he come?\n\nWill opens the gate.",
                               character_memory=memory, plan_only=True))
    check("the waiting shot describes only Maya", _described_in(shots[0], ["Will", "Maya"]) == ["Maya"])
    check("...and Will is there when he actually arrives", _described_in(shots[1], ["Will", "Maya"]) == ["Will"])
    shots = _shots_of(run_node("A doorway.\n\nOwen knocks. May I come in?",
                               character_memory="May: she, 30, blue dress.\nOwen: he, 42, blue shirt.", plan_only=True))
    check("'May I come in?' does not stage May", _described_in(shots[0], ["May", "Owen"]) == ["Owen"])


def test_an_undeclared_pronoun_is_reported():
    """With no pronoun on the sheet, "He sits down" cannot reach anybody and keeps the last
    shot's cast. That cannot be guessed; it is said."""
    print("\n=== a sheet with no pronouns is reported when the script uses them ===")
    result = run_node("A kitchen.\n\nMaya pours coffee.\n\nHe sits down.",
                      character_memory="Maya: 38, green sweater.\nOwen: 42, blue shirt.", plan_only=True)
    check("the author is told which entries have no pronoun",
          "Maya and Owen have no pronoun on the sheet" in str(result[2]))
    quiet = run_node("A kitchen.\n\nMaya pours coffee.\n\nOwen sits down.",
                     character_memory="Maya: 38, green sweater.\nOwen: 42, blue shirt.", plan_only=True)
    check("...and not when the script only uses names", "no pronoun on the sheet" not in str(quiet[2]))


def test_hardware_is_named_only_on_its_own_wearer():
    """Two people in two different restraints, and each shot names only what is on
    the body it describes.

    The hold read its list of hardware by flattening every restraint in the FILM into
    one list, so Ana's solo shot was told the leather collar stays closed -- Mara's
    collar, on Ana's neck, with the count in the same breath saying one body. The
    other direction put Ana's steel handcuffs on Mara. Hardware named on a body that
    is not wearing it is hardware the model draws there, or a second body to put it
    on: this file's standing rule."""
    print("\n=== hardware is named on its own wearer ===")
    mem = ("Ana: she, 30, a grey t-shirt, steel handcuffs locked on her wrists.\n"
           "Mara: she, 41, overalls, a leather collar locked on her neck.")
    shots = _shots_of(run_node(
        "A workshop.\n\nAna and Mara kneel side by side.\n\nAna looks up.\n\n"
        "Mara looks down.", character_memory=mem, plan_only=True))
    # BOTH ARE FASTENED AND BOTH STAY IN FRAME, so both pieces are in both shots --
    # and the sentence has to say which is on which, which is the rule this test was
    # written for. Pooled, it read "The steel handcuffs and leather collar on Ana and
    # Mara", naming both pieces and both bodies and pairing neither.
    check("Ana's handcuffs are attributed to Ana",
          "steel handcuffs on Ana" in shots[1], shots[1][-200:])
    check("...and Mara's collar to Mara", "leather collar on Mara" in shots[1])
    check("the other shot attributes them the same way",
          "steel handcuffs on Ana" in shots[2] and "leather collar on Mara" in shots[2])
    check("...and neither piece is put on the wrong body",
          "handcuffs on Mara" not in shots[2] and "collar on Ana" not in shots[2])
    # Both wearers in frame: both pieces belong in the sentence.
    check("the shot with both of them names both", "handcuffs" in shots[0]
          and "collar" in shots[0])


def test_a_soft_restraint_is_not_called_metal():
    """A leather collar is not metal, and is not told it is.

    The pose wording said "the metal is already drawn to its full length" whenever
    anything in the film was rigid -- so a woman in a leather collar was given a
    steel one, because somebody ELSE in the scene was in handcuffs. The material is
    the thing this node promises holds from shot to shot; naming the wrong one is the
    continuity break, not a wording nicety."""
    print("\n=== a soft restraint is not called metal ===")
    mem = ("Ana: she, 30, a grey t-shirt, steel handcuffs locked on her wrists.\n"
           "Mara: she, 41, overalls, a leather collar locked on her neck.")
    shots = _shots_of(run_node(
        "A workshop.\n\nAna and Mara kneel side by side.\n\nAna looks up.\n\n"
        "Mara looks down.", character_memory=mem, plan_only=True))
    # Ana is in the shot too now, and her handcuffs ARE metal, so the shot contains
    # the word. What must not happen is the COLLAR being called metal, so the check
    # is on the collar's own sentence rather than on the whole shot.
    _collar = next((s for s in shots[2].split(". ") if "leather collar on Mara" in s), "")
    check("the collar sentence exists", bool(_collar), shots[2][-200:])
    check("the collar is not called metal", "the metal" not in _collar, _collar)
    check("...but it is still held at its full length",
          "already drawn to its full length" in _collar, _collar)
    # Steel cuffs no longer say "the metal is drawn to its full length" -- a pair of
    # cuffs has no length to draw. They say what holds their shape instead, and the
    # leather collar still says what holds its.
    _cuffs = next((s for s in shots[1].split(". ") if "steel handcuffs on Ana" in s), "")
    check("the cuffs sentence exists", bool(_cuffs), shots[1][-200:])
    check("steel cuffs are held as rings, not as a run of chain",
          "rings are locked" in _cuffs and "full length" not in _cuffs, _cuffs)
    # Rope, all the way soft, is not metal on any shot of its own.
    rope = _shots_of(run_node(
        "A workshop.\n\nAna kneels on the floor.\n\nAna breathes.",
        character_memory="Ana: she, 30, a grey t-shirt, rope tying her wrists "
                         "behind her back.", plan_only=True))
    check("rope is not metal either", not any("the metal" in s for s in rope))


def test_one_garment_is_named_once_when_it_comes_off():
    """A garment two readers both find is still ONE garment coming off.

    The prose reader inferred "belt" from "Dan unlocks the chastity belt" and the
    hardware reader inferred it again off the sheet, and neither checked the other's
    list -- so the shot said "The chastity belt and the chastity belt come off",
    which is one garment described twice (two to draw) and a plural verb on a single
    item."""
    print("\n=== one garment, named once ===")
    P = ("A room.\n\nMcKenna: she, 22, crop top, chastity belt.\n\n"
         "Dan: he, 30, shirt.\n\nMcKenna walks in.\n\nDan looks at her.\n\n"
         "Dan unlocks the chastity belt.\n\nMcKenna sits down.")
    shot = _shots_of(run_node(P, plan_only=True, ref_noise_aug=0.999))[2]
    said = next((s for s in re.split(r"(?<=\.)\s+", " ".join(shot.split()))
                 if "comes off" in s or "come off" in s), "")
    check("the removal names the belt once",
          said.count("chastity belt") == 1, said or shot)
    check("...and as one thing, not two", "comes off during this shot" in shot)
    check("...and not as a pair", "belt and the chastity belt" not in shot)


def test_hardware_in_a_hand_is_not_hardware_on_a_body():
    """Carrying a pair of handcuffs does not put them on the person carrying them.

    The plain-noun branch counted the word alone, and "a pair OF handcuffs" put the
    article two words back where the determiner guard could not see it -- so the noun
    read as the VERB `handcuffs` as well. "Mara drops a pair of handcuffs into the
    toolbox" was a beat that cuffed Mara: the shot was told the hardware is open at
    the first frame and closed by the last, and every shot after it said the cuffs
    stay closed on her."""
    print("\n=== hardware in a hand ===")
    HOLD = re.compile(r"stays?\s+(?:closed and fastened|tied and holding)", re.I)
    for beat in ("Mara drops a pair of handcuffs into the toolbox.",
                 "Mara throws the cuffs on the bench.",
                 "Mara picks up a set of cuffs."):
        shots = _shots_of(run_node(f"A workshop.\n\n{beat}\n\nMara wipes her hands.",
                                   character_memory="Mara: she, 41, overalls.",
                                   plan_only=True))
        check(f"nothing is fastened by {beat[:30]!r}",
              not any(HOLD.search(sh) for sh in shots), shots[-1][-120:])
        check("...and nothing is told where it sits on her",
              "sits where it belongs" not in shots[0])
    # ...while a beat that carries it AND fastens it is a fastening.
    both = _shots_of(run_node(
        "A workshop.\n\nMara picks up the handcuffs and locks them on Ana's wrists."
        "\n\nAna kneels.", character_memory="Ana: she, 30, a grey t-shirt.\n"
                                            "Mara: she, 41, overalls.", plan_only=True))
    check("picking them up and locking them on is still a restraint",
          HOLD.search(both[-1]) is not None, both[-1][-120:])
    shown = _shots_of(run_node("A workshop.\n\nMara holds up the steel collar.\n\n"
                               "Mara turns.", character_memory="Mara: she, 41, overalls.",
                               plan_only=True))
    check("a collar held up is drawn as a collar", "sits where it belongs" in shown[0])
    check("...but nobody is wearing it", not any(HOLD.search(sh) for sh in shown))


def test_a_length_reaches_only_what_it_is_taken_around():
    """A length is recorded where it is PUT, not at every part named after it.

    Four ways this over-fired, all of them added while making lengths work at all:
    a run through a pulley was a cable locked on the person running it; every body
    part named after the item took a copy of it, so "straps Ana's wrists to the bench
    and wipes her own hands" put the straps on the hands as well; a neck word within
    sixty characters of a leg word hoisted a kneeling body off the floor; and the
    bare word "around" counted as evidence that a fastening had happened at all."""
    print("\n=== a length reaches what it is taken around ===")
    E = S.engine
    def parts(beat):
        return sorted((c, pt) for c, pt, _w, _a in E.hardware_spans(beat))

    check("a cable run through a pulley is on nobody",
          not E.applies_hardware("Mara runs the steel cable through the pulley."))
    check("...nor is a chain wound round a post",
          not E.applies_hardware("Mara winds the chain around the post."))
    check("a cable run around a neck is on somebody",
          E.applies_hardware("Mara loops the steel cable around Ana's neck."))

    check("a length holds the part it is taken around, and no other",
          parts("Mara straps Ana's wrists to the bench and wipes her own hands.")
          == [("straps", "wrists")],
          str(parts("Mara straps Ana's wrists to the bench and wipes her own hands.")))
    check("...and a second clause takes nothing with it",
          parts("Mara ties the rope around Ana's waist, then rests her hands "
                "on Ana's shoulders.") == [("rope", "waist")])
    check("...while a second object of the same verb does",
          parts("Mara tapes her wrists and her ankles.")
          == [("tape", "ankles"), ("tape", "wrists")])
    check("...and one length at two places is at both",
          parts("Mara loops a steel cable around her neck and down around her ankles.")
          == [("steel cable", "ankles"), ("steel cable", "neck")])

    for text, want in (("Mara loops a steel cable around her neck and down around "
                        "her ankles.", "ankles to the neck"),
                       ("a cable from her neck to her ankles", "ankles to the neck"),
                       ("Ana kneels, the collar at her neck, her legs folded under her.", ""),
                       ("Mara looks around the workshop while Ana stands with her "
                        "feet together.", ""),
                       ("Ana stands with her feet together.", ""),
                       ("Ana's ankles are cuffed together.", "ankles together")):
        check(f"legs of {text[:42]!r}", S.legs_anchor(text) == want, S.legs_anchor(text))


def test_the_shot_that_puts_it_on_says_so():
    """The applying shot must not be told the hardware is already closed.

    `restraint_going_on` keeps its own list of applying verbs, and the engine keeps a
    fuller one. Where they disagreed the shot that puts the manacles on fell through
    to the standing hold -- "Every restraint stays closed and fastened as it was put
    on" -- said of hardware that is open and in somebody's hands at the first frame,
    which is the one shot where that sentence is a lie.

    The lists are mirrored rather than merged, because this one answers a narrower
    question: the engine reads "her wrists cuffed" as hardware on, correctly, and
    here that must NOT read as the shot where it closes."""
    print("\n=== the shot that puts it on ===")
    staged = [v for v in ("manacles Ana's wrists", "blindfolds Ana", "leashes Ana",
                          "hogties Ana", "straitjackets Ana", "collars Ana",
                          "hobbles Ana", "shackles Ana's ankles",
                          "loops the rope around Ana's wrists",
                          "winds the chain around Ana's waist")
              if not S.restraint_going_on(f"Mara {v}.")]
    check("every way of putting it on is read as putting it on", not staged, str(staged))
    # ...and a state that already holds is not a shot that closes it.
    already = [b for b in ("Mara stands by the wall, her wrists cuffed behind her.",
                           "Mara is handcuffed to the rail.",
                           "The cable wound around her wrists holds.",
                           "Ana pulls against the cuffs.")
               if S.restraint_going_on(b)]
    check("...and what is already on is not put on again", not already, str(already))


def test_a_layer_the_author_shows_stops_waiting():
    """An under-layer waits while nothing has shown it. A beat that shows it ends the
    wait, the same way lifting the skirt does.

    "Her black thong shows above the waistband" puts the thong in that shot's picture,
    and the next shot opens on that frame -- so going back to holding it told the
    model the skirt is "the outermost layer there and the only one in view" one frame
    after the thong was visible in it. A picture outvotes a sentence, and what renders
    is the garment half there."""
    print("\n=== a layer the author shows stops waiting ===")
    mem = "Ana: she, 30, a denim skirt over a black thong, a grey t-shirt."
    shown = _shots_of(run_node(
        "A room.\n\nAna stands.\n\nHer black thong shows above the waistband.\n\n"
        "Ana waits.\n\nAna sits.", character_memory=mem, plan_only=True))
    check("it waits before the beat shows it",
          "thong" not in shown[0].split("There is")[0])
    for i, sh in enumerate(shown[1:], 2):
        head = " ".join(sh.split()).split("There is")[0]
        check(f"shot {i} keeps it once it has been shown", "thong" in head, head[-70:])
        check(f"...and shot {i} no longer calls the skirt the only thing in view",
              "only one in view" not in sh, sh[:170])
    quiet = _shots_of(run_node("A room.\n\nAna stands.\n\nAna waits.\n\nAna sits.",
                               character_memory=mem, plan_only=True))
    check("a thong nothing shows still waits",
          not any("thong" in sh.split("There is")[0] for sh in quiet))
    check("...and the cover is still described as unbroken",
          all("only one in view" in sh for sh in quiet), quiet[-1][:150])


def test_underwear_is_described_and_comes_all_the_way_off():
    """REPORTED: described underwear renders black whatever it was written as, and
    taking it off does not leave the body bare.

    Two causes, one at each end of the layer's life. The reveal clause named the
    garment by its identity KEY -- the head noun -- so a sheet saying "red lace
    panties" was answered with "The panties underneath are what shows there now": one
    garment named twice in a prompt, once with its description and once without, and
    the bare mention is the one the prior answers. And when the last layer finally
    came off, "The legs are bare from the hip down" named the legs and stopped, so the
    one part of that region underwear occupies was left unspecified -- which this
    file's own note beside REGION_OF says the prior fills with underwear."""
    print("\n=== underwear is described, and comes all the way off ===")
    mem = "Ana: she, 30, a denim skirt over red lace panties, a grey t-shirt."
    shots = _shots_of(run_node(
        "A room.\n\nAna stands by the bed.\n\nremove: skirt\nAna steps out of the "
        "skirt.\n\nremove: panties\nAna takes the panties off.\n\nAna lies down.",
        character_memory=mem, plan_only=True))
    check("the revealed layer keeps its colour and material",
          "red lace panties underneath" in " ".join(shots[1].split()),
          " ".join(shots[1].split())[-170:])
    check("...and is not also named bare",
          "The panties underneath" not in shots[1])
    check("the shot that takes it off says the body is bare there",
          "genitals uncovered" in shots[2], " ".join(shots[2].split())[-190:])
    check("...and every shot after it still does",
          all("genitals uncovered" in sh for sh in shots[3:]), shots[-1][-150:])

    covered = _shots_of(run_node(
        "A room.\n\nAna stands.\n\nremove: jeans\nAna steps out of the jeans.\n\n"
        "Ana lies down.",
        character_memory="Ana: she, 30, blue jeans over a black thong, a grey t-shirt.",
        plan_only=True))
    check("underwear still on keeps the region clothed",
          not any("genitals uncovered" in sh for sh in covered), covered[1][-150:])
    check("...and the thong is named as what shows, in the sheet's words",
          "black thong underneath" in " ".join(covered[1].split()))

    check("a declared adult gets the clause", S.groin_of("she", 30) != "")
    check("...and so does a man", S.groin_of("he", 40) != "")
    for pron, age in (("she", 17), ("she", 0), ("they", 30), ("", 30)):
        check(f"nothing is described for pronoun={pron!r}, age={age}",
              S.groin_of(pron, age) == "")


def test_a_restraint_stays_on_after_it_is_applied():
    """REPORTED: restraints disappear from the shots after they are applied.

    Every earlier restraint test declares the hardware on the SHEET -- already on --
    so the whole applying path went unwalked. Three things were wrong in it, and each
    one silently emptied every shot that followed:

      - "Dan locks a steel collar on her neck" put the collar on DAN. The reader that
        decides who wears it looks for the verb, then up to three words, then the
        pronoun -- and the ITEM and its preposition are four. With the wearer wrong,
        every shot describing McKenna had no wearer present and the hold was left out
        of all of them.
      - "Dan puts the cuffs on her" was not read as a restraint at all: "cuffs" is an
        ambiguous noun and that clause has neither a binding verb nor a body part, so
        the latch never armed even though the state recorded the cuffs on her.
      - A steel belt was in no vocabulary anywhere, and "tightens" was not a
        fastening verb."""
    print("\n=== a restraint stays on after it is applied ===")
    E = S.engine
    HELD = re.compile(r"stays? (?:closed and fastened|tied and holding)")
    mem = "McKenna: she, 22, a white crop top.\nDan: he, 30, a brown coat."
    beats = ["Dan handcuffs McKenna.",
             "Dan cuffs her wrists behind her back.",
             "Dan locks a steel collar on her neck.",
             "Dan buckles a leather collar around her throat.",
             "Dan puts the cuffs on her.",
             "Dan locks a steel belt around her waist.",
             "Dan tightens a strap around her thighs.",
             "Dan ties her wrists with rope.",
             "Dan hogties her."]
    for beat in beats:
        shots = _shots_of(run_node(
            f"A room.\n\nMcKenna stands.\n\n{beat}\n\nMcKenna breathes.\n\n"
            "McKenna waits.", character_memory=mem, plan_only=True))
        check(f"it is still on after {beat[:38]!r}",
              all(HELD.search(sh) for sh in shots[2:]), shots[-1][-130:])
        check("...and it went on the person it was applied to",
              E.wearer_of(beat, ["McKenna", "Dan"]) == "McKenna",
              E.wearer_of(beat, ["McKenna", "Dan"]))
    # ...and none of that makes a restraint out of an ordinary sentence.
    for beat in ("Dan locks the door behind her.", "Dan shuts the gate behind her.",
                 "Dan tapes the box shut.", "Dan puts the kettle on."):
        shots = _shots_of(run_node(
            f"A room.\n\nMcKenna stands.\n\n{beat}\n\nMcKenna breathes.",
            character_memory=mem, plan_only=True))
        check(f"nothing is fastened by {beat[:34]!r}",
              not any(HELD.search(sh) for sh in shots), shots[-1][-110:])


def test_the_budget_buys_as_many_guarantees_as_it_can():
    """fit_guards is a greedy fill, and that is deliberate.

    Stopping at the first refusal was tried -- the argument being that a ranking
    should decide what survives, not a price -- and measured on a real shot it was
    plainly worse. A restrained body being stripped spends 174 of its 200 words on
    the removal, the layers and the hardware hold, refuses the 38-word fall guard,
    and then throws away posture (3), duress (14), mouth (9) and camera (12) behind
    it. All four fit. One of them is the clause that keeps mouths shut on a wordless
    shot, which is the whole babble guard.

    A budget that binds has to buy as many guarantees as it can. The ranking chooses
    what goes first; it does not veto everything cheaper behind one refusal."""
    print("\n=== the budget buys what it can ===")
    clauses = [(1, "first", " one two three four five"),
               (5, "big", " " + " ".join(["word"] * 200)),
               (9, "cheap", " tiny clause here")]
    kept, dropped = S.fit_guards(clauses, 2)
    check("the top-ranked clause is kept", "first" not in dropped)
    check("...the one that does not fit is dropped", "big" in dropped, str(dropped))
    check("...and a cheap one behind it still fits", "cheap" not in dropped, str(dropped))
    check("the kept text holds both", "one two three four five" in kept
          and "tiny clause here" in kept, repr(kept))
    # An expensive FIRST clause is still kept: something always survives.
    only_big = S.fit_guards([(1, "big", " " + " ".join(["word"] * 200))], 2)
    check("a shot is never left with no guard", only_big[1] == [], str(only_big[1]))


def test_a_fall_keeps_its_landing_guard():
    """The clause that says what takes a landing is a body-integrity guard, not a
    continuity detail: without it the model frees the hands to break the fall, and on
    a body that cannot free them it grows a limb that can. Its own note records the
    report -- "a third leg on the shot where she fell".

    A restrained body being stripped is the worst case for it, and the worst case is
    where the budget ran out: removal, layers and hardware hold fill a 200-word floor
    on their own, so the one guard against an invented limb was the one thing
    refused. A fall buys its own room now."""
    print("\n=== a fall keeps its landing guard ===")
    mem = ("McKenna: she, 22, a white crop top over a lace bra, blue jeans over a "
           "black thong, boots, steel handcuffs locked on her wrists, ankle irons.\n"
           "Dan: he, 30, a brown coat.")
    shots = _shots_of(run_node(
        "A workshop.\n\nMcKenna kneels.\n\nremove: crop top\nremove: jeans\n"
        "remove: thong\nDan strips her and she falls.\n\nMcKenna strains.",
        character_memory=mem, plan_only=True))
    fell = shots[1]
    check("the falling shot says what takes the landing",
          "landing" in fell or "takes the weight" in fell, fell[-200:])
    check("...and still holds the hardware shut",
          "closed and fastened" in fell, fell[-200:])
    check("...and still keeps the mouths shut",
          "Mouths in the shot stay closed" in fell, fell[-200:])


def test_an_intimate_scene_holds_its_frame_and_its_voices():
    """REPORTED: in sex scenes moaning and sounds of pleasure came out as gibberish,
    and the scene kept turning into a side-angle shot in a different location."""
    print("\n=== an intimate scene holds its frame, and its voices stay wordless ===")
    mem = "Mara: she, 30, long dark hair.\nDan: he, 35, short brown hair."
    room = "A bedroom at night, warm lamp light.\n\n"
    sh = _shots_of(run_node(
        room + "Mara and Dan kiss on the bed.\n\n"
        "Dan lays Mara on her back and climbs on top of her.\n\n"
        "Mara climbs on top and rides Dan.\n\n"
        "Dan carries Mara to the shower.", character_memory=mem, plan_only=True))
    for i in (1, 2):
        check(f"a position change on a keyframe is not reframed wide: shot {i + 1}",
              "head to feet" not in sh[i] and "with the room around" not in sh[i], sh[i][-120:])
        check(f"...and keeps the held camera: shot {i + 1}", "one unbroken take" in sh[i])
    check("a journey keeps its frame, the camera going with them",
          "head to feet" in sh[3], sh[3][-120:])

    def _sound(beat):
        s = _shots_of(run_node(room + beat, character_memory=mem, plan_only=True))[0]
        m = re.search(r"(The only sound[^.]*\.|It sounds like[^.]*\.)", s)
        return s, (m.group(1) if m else "")

    for beat, word in (("Mara cries out in pleasure.", "wordless crying out"),
                       ("Mara makes sounds of pleasure.", "wordless moaning"),
                       ("Mara gasps with pleasure.", "wordless gasping"),
                       ("Dan grunts.", "wordless grunting"),
                       ("Mara moans.", "wordless moaning")):
        s, c = _sound(beat)
        check(f"the vocal is named, and wordless: {beat}", word in c, c)
    s, c = _sound("Mara cries out in pleasure.")
    check("...never an exclusive list of the room alone",
          "crying out" in c and not c.startswith("The only sounds are the quiet"), c)
    check("\"cries out\" is not read as crying", "The crying is" not in s, s[-160:])
    s, c = _sound("Dan grunts as he thrusts, and Mara makes sounds of pleasure.")
    check("two vocals are credited one each",
          "the grunting is dan's" in s.lower() and "the moaning is mara's" in s.lower(),
          s[-200:])
    check("...and neither mouth is closed on its own sound", "every other mouth" not in s)
    s, c = _sound("Dan grunts as he thrusts, and she moans.")
    check("a pronoun's vocal closes nobody's mouth", "every other mouth" not in s, s[-200:])
    check("the effort sound is wordless too", "wordless" in S.EFFORT_BREATH)
    check("wordless() folds the vocals into one phrase and keeps the order",
          S.wordless(["moaning", "breathing", "grunting", "a bed frame working"])
          == ["wordless moaning and grunting", "breathing", "a bed frame working"])
    check("...and leaves a list with no vocal alone",
          S.wordless(["breathing", "rain against the glass"])
          == ["breathing", "rain against the glass"])


def test_a_dropped_clause_is_not_reported_as_sent():
    """The per-shot trackers append where a clause is BUILT, before the budget runs,
    so when the budget binds the notes go on naming shots that never got it.

    A reader uses these notes to work out why a shot came out wrong; one that names
    the wrong shot costs more than the dropped clause did."""
    print("\n=== a dropped clause is not reported as sent ===")
    mem = ("Ana: she, 30, a grey t-shirt over a black bra, blue jeans, boots.\n"
           "Mara: she, 41, navy overalls.")
    # The removal is on shot 1: a later shot opens on a keyframe and is not given the
    # frame clause at all, so it is not crowded enough to drop one.
    info = str(run_node(
        "A workshop with white tiles.\n\nremove: t-shirt\nAna pulls it off.\n\n"
        "Ana looks at Mara.\n\nAna waits.", character_memory=mem, plan_only=True)[2])
    drop = next((n for n in info.split(" | ") if "dropped for room" in n), "")
    check("the run reports a shot whose clauses were dropped", "shot 1:" in drop, drop[:90])
    # Those exact clause names must not appear in their own notes for shot 1.
    for name, marker in (("frame", "told what the frame HOLDS"),
                         ("camera", "one unbroken TAKE")):
        if name in drop:
            said = next((n for n in info.split(" | ") if marker in n), "")
            shots = said.split("shot(s) ")[1].split(" ")[0] if "shot(s) " in said else ""
            check(f"...and the {name} note does not claim shot 1",
                  "1" not in shots.split(","), said[:110])


def test_a_promoted_clause_opens_in_upper_case():
    """A clause promoted out of a semicolon opens the sentence -- and the whole prompt.
    The capital was only restored when something survived in front of it."""
    print("\n=== a promoted clause opens in upper case ===")
    shots = _shots_of(run_node(
        "A kitchen; the kitchen has white tiles.\n\nAna fills a glass at the sink.\n\n"
        "Ana drinks.", character_memory="Ana: she, 30, a grey t-shirt.",
        plan_only=True))
    for i, sh in enumerate(shots, 1):
        body = sh.split("] ", 1)[-1].lstrip()
        check(f"shot {i} does not open in lower case",
              not body[:1].islower(), body[:60])


def test_the_reports_say_what_happened():
    """Three notes that stated something untrue.

    The balance note counted clauses that verbatim never sent, and drove its own
    guard share negative. The pacing note ran the clause splitter over the raw beat,
    so the clauses inside a quoted line counted as staged actions. And the
    recovered-face note said `script` does not show the claim tag, which it does."""
    print("\n=== the reports say what happened ===")
    mem = "Ana: she, 30, a grey t-shirt."
    for P in ("A workshop.\n\nAna screams as the drill whines.\n\nAna waits.",
              'A workshop.\n\nAna says: "Hold this."\n\nAna waits.'):
        info = str(run_node(P, character_memory=mem, plan_only=True)[2])
        bal = next((n for n in info.split(" | ") if "balance" in n), "")
        check("no negative share in the balance", "-" not in bal.split("clauses")[1][:6],
              bal[:130])
        check("...and the clauses really are counted, verbatim being gone",
              "continuity clauses 0%" not in bal, bal[:130])
    # The pacing number counts ACTIONS, and a spoken line is not four of them.
    spoken = str(run_node(
        'A workshop.\n\nAna puts the crate down and says: "Take it, then go, and '
        'do not come back."\n\nAna waits.', character_memory=mem, plan_only=True)[2])
    quiet = str(run_node(
        "A workshop.\n\nAna puts the crate down.\n\nAna waits.",
        character_memory=mem, plan_only=True)[2])
    def per(info):
        n = next((x for x in info.split(" | ") if x.startswith("pacing:")), "")
        return float(n.split("pacing: ")[1].split("s of")[0]) if n else 0.0
    check("a spoken line is not counted as staged actions",
          per(spoken) >= per(quiet) * 0.8, f"{per(spoken)} vs {per(quiet)}")


def test_a_line_is_counted_once():
    """A line marked the way this node's own note tells you to mark it counted DOUBLE.

    _QUOTED was defined twice -- once matching plain quotes, once matching those AND
    H3's <d> marker -- and the second silently won everywhere, because a function
    looks its globals up when it runs. Both places that count spoken words then added
    _DIALOGUE_TAG on top of a pattern that already matched it. A sixteen-word line was
    believed to take 12.8 seconds instead of 6.4: its shot was planned at twice the
    length, the dialogue-headroom warning never fired because the line already
    "fitted", and the tail silence pin started late."""
    print("\n=== a line is counted once ===")
    line = "Take the whole crate down to the yard and stack it by the blue door now"
    quoted = f'Ana says: "{line}"'
    marked = f"Ana says: <d>{line}</d>"
    check("sixteen words, counted as sixteen", S.spoken_words(quoted) == 16,
          str(S.spoken_words(quoted)))
    check("...however the line is marked", S.spoken_words(marked) == 16,
          str(S.spoken_words(marked)))
    check("...so both spellings plan the same shot",
          abs(S.beat_seconds(quoted) - S.beat_seconds(marked)) < 0.01,
          f"{S.beat_seconds(quoted)} vs {S.beat_seconds(marked)}")
    check("the delimiters are not words", S.spoken_words("Ana says: <d>Hold this.</d>") == 2,
          str(S.spoken_words("Ana says: <d>Hold this.</d>")))
    check("a beat with no line speaks nothing", S.spoken_words("Ana crosses the yard.") == 0)
    check("_QUOTED is defined once", sum(
        1 for ln in open(__file__.replace("test_smoke.py", "sampler.py"))
        if ln.startswith("_QUOTED = ")) == 1)


def test_the_ceiling_is_the_ceiling():
    """shot_seconds below one action's worth was silently ignored.

    The floor was applied AFTER the cap, so every shot came out at 3.0s however small
    the number asked for -- and the note said the shots had been CUT to it. Two
    statements in one report, one of them the opposite of what happened."""
    print("\n=== the ceiling is the ceiling ===")
    beats = ["Ana crosses the yard.", "Ana opens the gate."]
    for secs in (1.0, 2.0, 2.3):
        cf = S.align_frame_count_nearest(int(round(secs * S.H3_FPS)))
        lens, note = S.plan_lengths(beats, cf, True)
        check(f"shot_seconds {secs}s is honoured", all(n <= cf for n in lens),
              f"ceiling {cf}f, got {lens}")
        check("...and the run says the shots are shorter than one action",
              "below the" in note and "one staged action needs" in note, note[:90])
    # ...and above the floor nothing changed.
    cf = S.align_frame_count_nearest(int(round(6.0 * S.H3_FPS)))
    lens, note = S.plan_lengths(beats, cf, True)
    check("a roomy ceiling still sizes from the beat", all(n < cf for n in lens), str(lens))
    check("...and says nothing about a floor", "one staged action needs" not in note)


def test_the_machine_with_the_line_is_the_one_that_speaks():
    """The voice clause re-scanned the beat and took whichever machine came first,
    ignoring which one carries the speech verb -- so "Ana puts down the phone. The
    radio says: 'Storm warning.'" put the voice in the phone she had just put down."""
    print("\n=== the machine with the line ===")
    for beat, want in (
            ('Ana puts down the phone. The radio says: "Storm warning."', "radio"),
            ('Ana looks at the monitor while the tannoy announces: "Shift over."', "tannoy"),
            ('The TV says: "Rain later."', "TV"),
            ('The intercom buzzes and the speaker says: "Stand clear."', "speaker")):
        said = S.device_voice_clause(beat)
        got = re.search(r"the (\S+?)'s", said)
        check(f"the voice is the {want}'s", bool(got) and got.group(1) == want,
              said[:90])


def test_a_written_sound_is_not_denied():
    """The exclusive sound clause is what leaves nothing for an invented voice to
    fill, and it was closing the list around the node's OWN two inferences.

    "Mara strains against the vice as rain hammers the tin roof" came out as "The only
    sounds are a strip light humming and a large room with a long tail" -- the rain
    the author wrote and the straining body both excluded by a sentence claiming to
    name everything audible."""
    print("\n=== a written sound is not denied ===")
    mem = "Mara: she, 41, overalls."
    shot = _shots_of(run_node(
        "A workshop.\n\nMara strains against the vice as rain hammers the tin roof."
        "\n\nMara wipes her face.", character_memory=mem, plan_only=True,
        mouths_shut_when_no_line=False))[0]
    said = " ".join(shot.split())
    check("the list still closes, so no voice fills the gap", "The only sounds" in said,
          said[-160:])
    check("...but it does not deny what the beat describes",
          "the ones this beat describes" in said, said[-160:])
    # A written VOCAL is still named outright: that path was already right.
    vocal = " ".join(_shots_of(run_node(
        "A workshop.\n\nMara screams.\n\nMara breathes.", character_memory=mem,
        plan_only=True, mouths_shut_when_no_line=False))[0].split())
    check("a scream is still named as the sound", "screaming" in vocal, vocal[-160:])


def test_a_sound_given_up_is_really_silenced():
    """The shot paid the cost and got none of the benefit.

    With the mouth guard on, a wordless shot whose sound the author wrote has that
    sound stripped out so the mouths can be held shut -- and the run says so. But the
    plan recorded the shot as still having sound, so ShotAudio.pinned was false and
    the branch was never pinned to silence. An open branch on a joint model fills
    itself, and what it fills with is a voice: the one thing the whole exchange was
    meant to buy."""
    print("\n=== a sound given up is really silenced ===")
    P = "A workshop.\n\nRain hammers the tin roof.\n\nMara wipes her face."
    info = str(run_node(P, character_memory="Mara: she, 41, overalls.", plan_only=True,
                        mouths_shut_when_no_line=True, silence_nonspeech=True)[2])
    check("the run says the sound was given up", "gave up the sound you wrote" in info)
    check("...and the shot it was taken from is pinned",
          "shot(s) 1, 2 have no quoted line and no sound described" in info,
          info[-240:])


def test_a_carried_frame_does_not_add_an_uncounted_body():
    """REPORTED: duplicate characters. Two causes, both about pictures.

    A shot of McKenna alone counted ONE body and carried a picture of the shot before
    it, claimed as "Dan is the person there". The count is written a whole phase
    before the carry is decided -- the carry needs a frame to carry -- so it never saw
    the claim. One body counted, two people named, and a photograph of the one who is
    not counted.

    And a shot went out with THREE pictures for two people: McKenna's portrait, Dan's
    evening-up frame, and the room frame that also shows Dan. Two pictures of one
    person is this file's own recipe for drawing a second. The recovered-face path
    already skipped a person the carried frame pictures; the evening-up path did
    not."""
    print("\n=== a carried frame does not add an uncounted body ===")
    img = lambda: torch.rand(1, H, W, 3)
    mem = ("McKenna: she, <Picture 1>, 22, a white crop top, steel handcuffs locked "
           "on her wrists.\nDan: he, 30, a brown coat.")
    rows = _encoded_refs(
        "A workshop.\n\nDan walks to the bench.\n\nMcKenna strains against the chain."
        "\n\nDan kneels beside her.", character_memory=mem, ref_image_1=img())
    said = " ".join(rows[1][0].split())
    check("the shot naming two people counts two bodies",
          "Dan is the person there" not in said or "two people" in said, said[-190:])
    # THREE PICTURES FOR TWO PEOPLE, on the shape that produced them.
    rows2 = _encoded_refs(
        "A workshop.\n\nDan walks to the bench.\n\nDan kneels beside her.\n\n"
        "McKenna lies still.", character_memory=mem, ref_image_1=img())
    for i, (txt, n) in enumerate(rows2, 1):
        t = " ".join(txt.split())
        dan = t.count("Dan: <Picture") + len(re.findall(r"Dan is the person", t))
        check(f"shot {i} sends at most one picture of Dan", dan <= 1, t[-170:])
        people = len({x for x in ("McKenna", "Dan")
                      if re.search(rf"\b{x}:", t.split("There is")[0].split("There are")[0])
                      or re.search(rf"\b{x} is the person", t)})
        check(f"...and no more pictures than people it identifies",
              n <= max(1, people), f"{n} pictures, {people} identified")


def test_one_person_gets_one_picture():
    """Two pictures of one person in one shot is what draws a second copy of her.

    Every picture path checks the others -- except the frame sent to EVEN UP a
    two-hander, which was invisible to all of them. A shot went out carrying Mara's
    solo frame as one picture AND a returning room frame as another, both taken from
    the same earlier shot, with the text claiming her in both: "Mara: <Picture 2>"
    and "<Picture 3> is the kitchen ... Mara is the person in it"."""
    print("\n=== one person, one picture ===")
    img = lambda: torch.rand(1, H, W, 3)
    mem = ("Ana: <Picture 1>, she, 30, a grey apron.\nMara: she, 34, a blue shirt.\n"
           "Nils: he, 40, a brown coat.")
    P = ("A kitchen with white tiles.\n\nMara measures a dowel.\n\n"
         "In the yard, Nils stacks crates.\n\n"
         "In the kitchen, Ana is at the counter with Mara.")
    rows = _encoded_refs(P, character_memory=mem, ref_image_1=img())
    text, count = rows[2]
    check("two people in the shot are sent two pictures", count == 2, str(count))
    check("...one each, and Mara is claimed once",
          text.count("Mara is the person in") == 0 and "Mara: <Picture 2>" in text,
          " ".join(text.split())[-200:])
    check("...and no third picture rides with them",
          "<Picture 3>" not in text, " ".join(text.split())[-200:])


def test_an_untagged_picture_is_not_a_stranger():
    """A picture the prompt never mentions is read as ANOTHER person standing beside
    the ones it describes, and no sentence here can argue with a photograph.

    The claim-or-hold decision read the per-shot CAST LIST, which is empty in every
    shot when character_guard is off -- and the text still carries the whole sheet.
    So the picture rode untagged through the entire film in one of the two commonest
    setups there is."""
    print("\n=== an untagged picture is not a stranger ===")
    img = lambda: torch.rand(1, H, W, 3)
    mem = "Ana: she, 30, a grey apron.\nMara: she, 34, a blue shirt."
    P = "A kitchen.\n\nAna pours coffee.\n\nAna and Mara wash the cups.\n\nMara dries them."
    rows = _encoded_refs(P, character_memory=mem, character_guard=False,
                         ref_image_1=img())
    for i, (text, count) in enumerate(rows, 1):
        check(f"guard off, shot {i}: no picture rides unnamed",
              count == 0 or "<Picture" in text, f"encoded={count}")
    # A script with nobody described still keeps its plate, and is told why.
    plate = [n for _p, n in _encoded_refs(
        "A kitchen with white tiles.\n\nThe kettle boils.\n\nSteam rises from the spout.",
        ref_image_1=img())]
    check("a script with nobody in it still keeps its reference", plate == [1, 1],
          str(plate))
    info = str(run_node("A kitchen.\n\nAna pours coffee.\n\nAna sits down.",
                        plan_only=True, ref_image_1=img())[2])
    check("...and a sheetless script is told what an untagged picture costs",
          "riding with nothing in the text naming it" in info)


def test_a_beat_that_moves_a_garment_keeps_the_cast():
    """`_was` is the previous shot's cast. A loop moving a garment reused the name for
    the garment's old STATE, inside the same per-beat pass, and two clauses further
    down still read it as a list of names.

    So on any beat that displaces a garment, the carried gaze and the carried mouth
    guard iterated a STRING: either "" -- and a person standing in the keyframe lost
    the clause that keeps her mouth shut -- or the letters of "pulled up", a cast of
    p, u, l, l, e, d. Names invented out of a garment's state."""
    print("\n=== a garment moving does not eat the cast ===")
    mem = "Ana: she, 30, a grey t-shirt, blue jeans.\nMara: she, 41, navy overalls."
    shots = _shots_of(run_node(
        "A workshop.\n\nAna and Mara stand at the bench.\n\nAna looks at the window.\n\n"
        "Mara pulls her overalls down.\n\nMara says: \"Hold this.\"",
        character_memory=mem, plan_only=True))
    check("the person the frame carries is still counted",
          "two people" in shots[2], shots[2][:170])
    check("...and her latched look survives the beat",
          "Ana's eyes and head are turned to the window" in shots[2], shots[2][:200])
    check("...and the garment still moves", "pulled down" in shots[3], shots[3][-140:])


def test_a_carried_clause_and_the_count_agree():
    """Two shots carrying the same sentence must not be given opposite counts.

    The count is drawn from the people this shot's own words NAME -- but presence was
    read off the PREVIOUS shot's cast, one shot of memory, while a latched clause goes
    on naming somebody for as long as it holds. So two consecutive shots carried "The
    eyes and the head are turned to Mara." word for word and were told first that
    there are two bodies and then that there is one. The second names a person the
    same breath says is not there."""
    print("\n=== a carried clause and the count agree ===")
    mem = "Ana: she, 30, a grey t-shirt.\nMara: she, 27, a blue apron."
    shots = _shots_of(run_node(
        "A workshop.\n\nAna looks at Mara.\n\nAna sits down.\n\nAna stands up.",
        character_memory=mem, plan_only=True))
    for i, sh in enumerate(shots, 1):
        if "turned to Mara" in sh:
            check(f"shot {i} names Mara and counts her",
                  "two people" in sh, sh[:170])
    # ...and a look at somebody who LEAVES stops being said at all.
    gone = _shots_of(run_node(
        "A kitchen.\n\nAna looks at Mara.\n\nMara walks out.\n\nAna pours coffee.",
        character_memory=mem, plan_only=True))
    check("a look at somebody who left is dropped",
          not any("turned to Mara" in sh for sh in gone[2:]), gone[-1][:170])
    check("...and she is not counted either",
          "one person" in gone[2], gone[2][:170])


def test_a_pronoun_object_keeps_the_person_it_means():
    """"Mara hugs her" needs two bodies, and kept one.

    The rule read a pronoun after a PREPOSITION -- "kneels beside her" was right --
    and had nothing for a direct object, which is the commoner half. The woman "her"
    refers to lost her sheet line, and the shot was then told "There is one person in
    the shot: one body, one face" beside a verb whose own meaning needs two. A person
    in frame with no description is a person the model dresses out of nothing."""
    print("\n=== a pronoun object keeps its person ===")
    mem = "Ana: she, 30, a grey t-shirt.\nMara: she, 27, a blue apron."
    for beat in ("Mara hugs her.", "Mara joins her.", "Mara follows her.",
                 "Mara watches her.", "Mara hands her the tin.",
                 "Mara passes her the wrench.", "Mara kneels beside her.",
                 "Mara told her the news."):
        check(f"both women are kept by {beat!r}",
              len(S.sheet_for_beat(mem, beat, ["Ana", "Mara"])[1]) == 2,
              str(S.sheet_for_beat(mem, beat, ["Ana", "Mara"])[1]))
    # ...and a POSSESSIVE is still the subject's own.
    for beat in ("Mara takes her jacket off.", "Mara washes her hands.",
                 "Mara shuts the door behind her.", "Mara pulls her hair back."):
        check(f"...while {beat!r} stays one person",
              S.sheet_for_beat(mem, beat, ["Ana", "Mara"])[1] == ["Mara"],
              str(S.sheet_for_beat(mem, beat, ["Ana", "Mara"])[1]))


def test_a_people_word_is_not_always_a_crowd():
    """Staging extras stands the body-count guard down for the REST OF THE FILM, so a
    people-word used as a modifier cost the duplicate guard everywhere.

    "the staff room", "the men's overalls", "the women's section", "the customers'
    invoices", "the figures in the ledger" all read as crowds. And the release was as
    loose as the latch: "Ana picks up the empty box" dismissed a crowd staged one beat
    earlier, while the crowd is still in the keyframe the shot opens on -- and the
    same word clears the carried frame, so an empty box was emptying the room."""
    print("\n=== a people-word is not always a crowd ===")
    for beat in ("Ana walks into the staff room.", "Ana folds the men's overalls.",
                 "Ana checks the women's section.", "Ana files the customers' invoices.",
                 "Ana reads the figures in the ledger.", "The others have gone.",
                 "Ana hears people outside."):
        check(f"no crowd in {beat[:38]!r}", not S.extras_in(beat))
    for beat in ("Students fill the yard.", "A crowd of students waits.",
                 "The staff room fills with students.", "Two men wait by the van.",
                 "Onlookers press against the barrier.", "Dark figures wait by the gate."):
        check(f"a crowd in {beat[:38]!r}", S.extras_in(beat))
    for beat in ("Ana picks up the empty box.", "Ana empties the bin.",
                 "The crowd leaves."):
        check(f"{beat[:34]!r} dismisses nobody", not S.extras_dismissed(beat))
    for beat in ("The yard is empty.", "Ana is alone now.",
                 "Ana stands in the empty hall.", "Ana is by herself."):
        check(f"{beat[:34]!r} does dismiss them", S.extras_dismissed(beat))


def test_a_garment_set_down_is_not_a_garment_put_on():
    """Taking something off, and putting it down, is not putting it back on.

    Three ways one shot came to hold both clauses at once. "dress" is a garment as
    often as it is a verb, so the shot a dress came off was also told the dress goes
    on. "Ana puts the t-shirt on the bench" is the same six words as putting it on.
    And the gap between verb and particle reached across a whole clause, so "pulls
    Ana's t-shirt off and drops it on the bench" matched as "pulls ... off and drops
    it on". Every shot after opened "Ana is wearing the grey t-shirt", on a woman
    whose sheet entry had been scrubbed because she took it off."""
    print("\n=== set down is not put on ===")
    for beat, item, want in (
            ("Ana takes off her dress and drops it on the bench.", "dress", False),
            ("Ana puts the t-shirt on the bench.", "t-shirt", False),
            ("Mara pulls Ana's t-shirt off and drops it on the bench.", "t-shirt", False),
            ("Ana hangs the coat on the hook.", "coat", False),
            ("Ana puts her boots on the floor.", "boots", False),
            ("Ana folds her dresses and puts them in the drawer.", "dress", False),
            ("Ana sews a button on the shirt.", "shirt", False),
            # ...and the dressings that must survive all of that.
            ("Ana puts her coat on.", "coat", True),
            ("Ana pulls the sweater back on.", "sweater", True),
            ("Ana pulls her boots on.", "boots", True),
            ("Ana steps into the jeans.", "jeans", True),
            ("Ana zips up the jacket.", "jacket", True),
            ("Ana pulls off her jumper and puts on her coat.", "coat", True)):
        check(f"{'puts on' if want else 'does not put on'}: {beat[:40]!r}",
              S.beat_stages_wearing(beat, item) is want)
    # End to end: the removing shot says one thing about the garment, not two.
    shot = _shots_of(run_node(
        "A workshop.\n\nMara pulls Ana's t-shirt off and drops it on the bench.\n\n"
        "Ana breathes.",
        character_memory="Ana: she, 30, a grey t-shirt, blue jeans.\n"
                         "Mara: she, 34, a blue apron.", plan_only=True))[0]
    check("the removing shot does not also dress her",
          "fully on by the last frame" not in shot, shot[-140:])


def test_what_a_removal_uncovers_is_said_on_every_later_shot():
    """A region with nothing on it has to be described for as long as that is true.

    An unspecified region is one the model fills from its own prior, which is the
    report this machinery exists for: a bra coming back on a topless character whose
    sheet never had one. It was never restored, it was invented.

    Four ways the latch never caught. A compound removal recorded only the first
    garment, so "takes off her t-shirt and her jeans" left the legs unspecified from
    the next shot on. "Ana undresses completely." latched nothing at all while "Ana
    strips naked." latched everything -- two spellings of one act. And a dress, gown,
    robe or overalls sat in no region row, so the commonest full-body garment in any
    script came off and not one shot said anything about the body it left."""
    print("\n=== what a removal uncovers ===")
    E = S.engine

    def bare_of(beat, sheet="she, 30, a grey t-shirt, blue jeans, boots"):
        st = E.SceneState("a workshop")
        st.declare("Ana", sheet)
        st.read(beat, cast=["Ana"], shot=1)
        return st.people["Ana"].bare

    check("a compound removal records both garments",
          bare_of("Ana takes off her t-shirt and her jeans.") == ["torso", "legs"],
          str(bare_of("Ana takes off her t-shirt and her jeans.")))
    check("...and a list of three records all of them",
          sorted(bare_of("Ana takes off her t-shirt, her jeans and her boots."))
          == ["feet", "legs", "torso"])
    check("...while a second clause with a verb of its own inherits nothing",
          bare_of("Ana takes off her boots and sits on the bench.") == ["feet"])

    for spelling in ("Ana undresses completely.", "Ana strips naked.",
                     "Ana takes all her clothes off.", "Ana takes off her clothes."):
        check(f"{spelling!r} leaves the whole body bare",
              sorted(bare_of(spelling)) == ["feet", "legs", "torso"],
              str(bare_of(spelling)))
    check("...but taking off ONE named garment still takes off one",
          bare_of("Ana strips off her coat.") == ["torso"])
    check("...and stripping paint undresses nobody",
          bare_of("She strips the paint off the door.") == [])

    # A GARMENT THAT IS THE WHOLE OUTFIT leaves two regions, and both are said.
    for whole in ("gown", "dress", "robe", "overalls", "jumpsuit"):
        check(f"a {whole} leaves both halves of the body",
              sorted(E.regions_of(whole)) == ["legs", "torso"],
              str(E.regions_of(whole)))
    check("...and an apron leaves nobody bare", E.regions_of("apron") == [])
    shots = _shots_of(run_node(
        "A workshop.\n\nAna stands at the bench.\n\nremove: gown\nAna pulls the gown "
        "off.\n\nAna picks up a wrench.\n\nAna turns to the window.",
        character_memory="Ana: she, 30, a silk gown.", plan_only=True))
    check("the shots after a gown comes off describe the body it left",
          all("bare from the hip down" in sh and "bare skin" in sh
              for sh in shots[2:]), shots[-1][-150:])


def test_a_full_stop_ends_a_clause():
    """Two sentences in one beat are two clauses.

    The engine knew only "," ";" "and" and "while", and a beat is usually several
    sentences -- so "Ana takes off her boots. Mara hangs a coat on the hook." was ONE
    clause. The boots were credited to Mara, and the coat, which is on a hook and was
    never worn by anybody, was recorded as coming off her too: every later shot
    described Mara barefoot and topless. Writing the same beat with ", and" gave the
    right answer, which is the tell."""
    print("\n=== a full stop ends a clause ===")
    E = S.engine
    for beat in ("Ana takes off her boots. Mara hangs a coat on the hook.",
                 "Ana takes off her boots, and Mara hangs a coat on the hook."):
        st = E.SceneState()
        st.declare("Ana", "she, 30, a coat, boots, gloves")
        st.declare("Mara", "she, 41, overalls")
        ch = st.read(beat, cast=["Ana", "Mara"])
        check(f"the boots are Ana's in {beat[:34]!r}",
              ch["removed"] == [("Ana", "boots")], str(ch["removed"]))
        check("...and nobody else is undressed by it",
              st.people["Mara"].bare == [], str(st.people["Mara"].bare))


def test_a_layer_is_covered_where_it_actually_sits():
    """The clause that covers an under-layer has to name the right part of the body,
    and must not take the covering garment out of the sheet with it.

    It said "the hips and waist" whatever the garment was, so a bra under a t-shirt
    produced "The t-shirt covers the hips and waist completely ... the only one in
    view" -- on a character whose same entry lists blue jeans. And hiding the
    under-layer of "a denim skirt over black knickers" matched three words back and
    took the SKIRT out too: the model invented a skirt for the covered shots, which
    then changed into a denim one at the removal boundary."""
    print("\n=== a layer is covered where it sits ===")
    check("a t-shirt covers the chest",
          "the chest and stomach" in S.under_clause([("black bra", "t-shirt", "Ana")]))
    check("a skirt covers the hips",
          "the hips and waist" in S.under_clause([("knickers", "denim skirt", "Ana")]))
    check("a dress covers both",
          "the chest, stomach, hips and waist"
          in S.under_clause([("camisole", "silk dress", "Ana")]))
    check("boots cover the feet",
          "the feet and ankles" in S.under_clause([("socks", "boots", "Ana")]))

    check("hiding the under-layer keeps the garment over it",
          S.hide_item("Ana: she, 30, a denim skirt over black knickers, a grey t-shirt.",
                      ["knickers"])
          == "Ana: she, 30, a denim skirt, a grey t-shirt.",
          S.hide_item("Ana: she, 30, a denim skirt over black knickers, a grey t-shirt.",
                      ["knickers"]))
    check("...with an article in the way too",
          S.hide_item("Ana: she, 30, blue jeans over a black thong, a grey t-shirt.",
                      ["thong"]) == "Ana: she, 30, blue jeans, a grey t-shirt.")
    check("...and the adjectives still go with their own garment",
          S.hide_item("Ana: she, 30, a tight white crop top, blue jeans.",
                      ["crop top"]) == "Ana: she, 30, blue jeans.")


def test_another_persons_clothes_do_not_answer_for_this_body():
    """The bare clause is silent when something still worn covers the region -- and
    it was reading every person in the shot, so the OTHER character's clothes
    answered for this one's body.

    A woman whose trousers have just come off is told nothing about her legs because
    somebody kneeling beside her is wearing overalls. The region goes unspecified,
    and the model fills it from its own prior: legwear the prompt never asked for,
    carried into every later shot by the keyframe."""
    print("\n=== another person's clothes ===")
    P = ("A workshop.\n\nAna lies hogtied on the floor.\n\nMara kneels beside her.\n\n"
         "Mara pulls Ana's trousers down to her knees.\nremove: trousers\n\n"
         "Ana pulls against the chain.")
    for mara in ("navy overalls", "a navy shirt"):
        mem = ("Ana: she, 30, grey t-shirt, black trousers, a steel chain linking her "
               f"wrists and ankles behind her back.\nMara: she, 41, {mara}.")
        shot = _shots_of(run_node(P, character_memory=mem, plan_only=True))[2]
        check(f"...with Mara in {mara!r}, Ana's legs are still described",
              "bare from the hip down" in shot, shot[-150:])


def test_the_two_wardrobe_readers_agree():
    """The sampler and the engine must read the same sentence the same way.

    They keep separate vocabularies for removals, for dressing and for what counts as
    a garment, and where they disagree ONE HALF ACTS AND THE OTHER DOES NOT -- which
    is not a missed feature, it is a contradiction inside one prompt. The sampler
    takes the sweater off, scrubs it from the sheet and prints "the red sweater comes
    off during this shot"; SceneState never records it, so `p.bare` never gains the
    torso and no later shot says anything about her chest. An unspecified region is
    one the model fills from its own prior, which is the report REGION_OF exists for:
    a bra coming back on somebody topless, on a character whose sheet never had one.

    Walked as a matrix, because the gaps were never where anyone was looking: the
    removal lists had drifted by FORTY-TWO forms, the dressing lists by nine in both
    directions, and twenty-four garments the sampler's own families name could not be
    placed by the engine at all. Four of those were engine.REGION_OF disagreeing with
    engine.GARMENT_WORDS -- two lists in one file."""
    print("\n=== the two wardrobe readers agree ===")
    E = S.engine

    def engine_removes(beat):
        st = E.SceneState()
        st.declare("Ana", "she, 30, a red sweater, blue jeans")
        ch = st.read(beat, cast=["Ana"])
        return bool(ch.get("removed")) or bool(st.people["Ana"].bare)

    sheet = "Ana: she, 30, a red sweater, blue jeans."
    forms = sorted(set(_spellings(E._STRIP_VERB, cap=200)))
    split = [v for v in forms
             if bool(S.infer_removals(f"Ana {v} her sweater off.", sheet))
             != engine_removes(f"Ana {v} her sweater off.")]
    check(f"a removal is a removal to both readers ({len(forms)} verb forms)",
          not split, str(split[:8]))

    dressing = [v for v in ("puts", "pulls", "pulled", "slips", "slipped", "tugs",
                            "tugged", "steps", "stepped", "climbs", "climbed",
                            "gets", "got", "wriggles", "draws", "drew")
                if bool(S._PUTS_ON.search(f"Ana {v} the sweater on."))
                != bool(re.search(E.PUTS_ON, f"Ana {v} the sweater on.", re.I))]
    check("...and so is a dressing", not dressing, str(dressing))

    unplaceable = [w for fam in S._GARMENT_FAMILIES for w in fam
                   if not E.garments_in(f"Ana takes her white {w} off.")
                   or not E.region_of(w)]
    check("every garment the sampler names, the engine can place",
          not unplaceable, str(unplaceable[:8]))

    orphan = [w for pat, _r, _s in E.REGION_OF for w in _spellings(pat, cap=80)
              if not E.garments_in(f"Ana takes her white {w} off.")]
    check("...and every word with a region is a word the engine calls a garment",
          not orphan, str(orphan[:8]))

    # None of that turns an ordinary sentence into an undressing.
    innocent = [b for b in ("Ana cuts the bread.",
                            "Ana works late at the bench.",
                            "Ana steps back from the door.",
                            "Ana lifts the lid off the box.",
                            "Ana throws the switch.")
                if engine_removes(b)]
    check("...and no ordinary sentence undresses anybody", not innocent, str(innocent))

    # THE POINT OF ALL OF IT: the region is still described several shots later.
    for verb in ("yanks", "rips", "works", "tosses", "cuts"):
        shots = _shots_of(run_node(
            f"A workshop.\n\nAna stands.\n\nAna {verb} her sweater off.\n\n"
            "Ana picks up a spanner.\n\nAna waits.",
            character_memory="Ana: she, 30, a red sweater, blue jeans.",
            plan_only=True))
        check(f"...so {verb!r} leaves a chest the later shots still describe",
              all("are bare skin" in sh for sh in shots[2:]), shots[-1][-110:])


def test_a_beam_in_a_roof_is_not_a_smile():
    """"Beam" is a piece of a building in this node's own anchor list, and a broad
    smile in its emotion list. Chaining somebody to one made the face beam."""
    print("\n=== a beam is not a smile ===")
    check("chaining somebody to a beam names no feeling",
          S.emotion_in("Mara chains Ana to the beam.") == "")
    check("...nor does looking at one", S.emotion_in("Ana looks up at the beam.") == "")
    check("...nor do the beams holding a roof",
          S.emotion_in("The steel beams hold the roof.") == "")
    check("a person still beams", S.emotion_in("Ana beams at him.") == "beaming")
    shot = _shots_of(run_node("A barn.\n\nMara chains Ana to the beam.\n\nAna breathes.",
                              character_memory="Ana: she, 30, a grey t-shirt.\n"
                                               "Mara: she, 41, overalls.",
                              plan_only=True))[0]
    check("...and the shot that chains her says nothing about a smile",
          "expression is beam" not in shot)


def test_the_shot_that_takes_it_off_is_not_told_where_it_sits():
    """One sentence must not undo the other in the same breath.

    The clause that says where hardware belongs fires on any beat naming hardware
    with no body part beside it -- and "Mara unlocks the handcuffs" is exactly that
    shape. The removing shot read both "The handcuffs come off during this shot and
    are away by the last frame" and "handcuffs close around the wrists"."""
    print("\n=== taken off, not placed ===")
    mem = ("Ana: she, 30, a grey t-shirt, steel handcuffs locked on her wrists.\n"
           "Mara: she, 41, overalls.")
    shot = _shots_of(run_node("A workshop.\n\nAna kneels.\n\nMara unlocks the handcuffs."
                              "\n\nAna stands up.", character_memory=mem,
                              plan_only=True))[1]
    check("the removing shot says they come off", "come off during this shot" in shot)
    check("...and is not also told where they sit",
          "sits where it belongs" not in shot, shot)
    check("a plural item takes a plural verb", "handcuffs come off" in shot)
    check("...and a singular one does not", S.plural_item("chastity belt") is False)
    check("...and a dress is not a plural", S.plural_item("a red dress") is False)
    check("...while boots are", S.plural_item("black boots") is True)


def test_a_removal_stays_off_in_every_later_shot():
    """REPORTED: a garment taken off is still listed in the beats that follow.

    Run through the whole shot loop, because the failures were spread over it: the
    `remove:` line read word for word, the removal verb read only in front of its
    object, the wearer read off whoever was named first, and a garment word already
    gone for one person swallowing every later removal of it for anybody else."""
    print("\n=== a removal stays off in every later shot ===")
    mem = ("Kate: she, 25, blue denim jacket, white shirt, black jeans, white bra, "
           "black panties.\nDan: he, 40, grey shirt, black trousers.")
    rest = "\n\nKate sits on the bed. Dan sits beside her.\n\nKate and Dan talk quietly."

    def after(beat):
        shots = _shots_of(run_node("A small flat.\n\nKate and Dan stand in the bedroom."
                                   "\n\n" + beat + rest, plan_only=True,
                                   character_memory=mem))
        return shots[2:]

    def entry(shot, who):
        m = re.search(who + r": [^.\n]*\.", shot)
        return m.group(0) if m else ""

    for beat, gone in (("Kate shrugs.\nremove: her jacket", ["jacket"]),
                       ("Kate shrugs.\nremove: the jacket", ["jacket"]),
                       ("Kate shrugs.\nremove: Jacket.", ["jacket"]),
                       ("Kate shrugs.\nremove: jacket and shirt", ["jacket", "white shirt"]),
                       ("Kate shrugs.\nremove: blue jacket", ["jacket"]),
                       ("Kate's jacket comes off.", ["jacket"]),
                       ("Her jacket is removed.", ["jacket"]),
                       ("Kate takes off her jacket, shirt and jeans.",
                        ["jacket", "white shirt", "jeans"])):
        later = after(beat)
        check(f"stays off after {beat.splitlines()[-1]!r}",
              all(g not in entry(s, "Kate") for s in later for g in gone),
              entry(later[-1], "Kate"))
    later = after("Dan takes off her jacket and shirt.")
    check("her shirt comes off her", all("white shirt" not in entry(s, "Kate") for s in later),
          entry(later[-1], "Kate"))
    check("...and his stays on him", all("grey shirt" in entry(s, "Dan") for s in later),
          entry(later[-1], "Dan"))
    later = after("Dan undresses Kate.")
    check("the one undressed has nothing listed",
          all("jeans" not in entry(s, "Kate") for s in later), entry(later[-1], "Kate"))
    check("...and the one undressing keeps his clothes",
          all("trousers" in entry(s, "Dan") for s in later), entry(later[-1], "Dan"))
    later = after("Kate strips to her bra and panties.")
    check("a partial strip takes the rest off",
          all("jeans" not in entry(s, "Kate") for s in later), entry(later[-1], "Kate"))
    check("...and keeps what it names",
          all("white bra" in entry(s, "Kate") and "black panties" in entry(s, "Kate")
              for s in later), entry(later[-1], "Kate"))
    shots = _shots_of(run_node(
        "A small flat.\n\nKate and Dan stand in the bedroom.\n\nKate takes off her shirt."
        "\n\nDan takes off his shirt." + rest, plan_only=True, character_memory=mem))
    # Shot 3 follows a removal, so it starts fresh and still describes what comes off
    # in it -- on him. Hers came off a shot earlier and stays off.
    check("the same word off a second person comes off him too",
          all("grey shirt" not in entry(s, "Dan") for s in shots[3:]), entry(shots[-1], "Dan"))
    check("...named as his own", "grey shirt comes off" in shots[2].lower(), shots[2])
    check("...and hers does not come back while his comes off",
          "white shirt" not in entry(shots[2], "Kate"), entry(shots[2], "Kate"))


def test_a_scene_keeps_its_room_and_its_people():
    """REPORTED: the scenery resets when it should not, positions reset, and a sex
    scene resets the scenery.

    "across the room" was a journey to a room called "room", so the shot was told it
    arrived somewhere else and the woman it did not name was left behind in the
    bedroom it had supposedly left. "Dan enters her" was Dan walking in, so a shot of
    two people already in bed started fresh. And a beat naming one partner counted
    one body over a first frame holding two."""
    print("\n=== a scene keeps its room and its people ===")
    mem = "Kate: she, 25, grey sweater, blue jeans.\nDan: he, 40, grey shirt, black trousers."
    room = ("A small bedroom with a double bed, grey walls and a lamp on the nightstand. "
            "The kitchen has white tiles.")
    shots = _shots_of(run_node(
        room + "\n\nKate sits on the bed. Dan stands by the window.\n\n"
        "Dan walks across the room to the bed.\n\nDan sits beside Kate.",
        character_memory=mem, plan_only=True))
    check("across the room is not a journey",
          not any("arrives in the room" in s or "place in the room" in s for s in shots),
          shots[1][:200])
    check("...and the woman it did not name is still there", "Kate:" in shots[2], shots[2][:160])

    info, script = run_node(
        room + "\n\nKate and Dan kiss on the bed.\n\nKate lies back on the bed.\n\n"
        "Dan enters her.\n\nKate arches her back.\n\nDan comes inside her.\n\nKate lies still.",
        character_memory=mem, plan_only=True)[2:4]
    shots = _shots_of((None, None, info, script))
    check("nobody already in bed is staged walking in",
          "staged walking in" not in info, info[:200])
    for i, s in enumerate(shots, 1):
        check(f"shot {i} describes both partners and counts two",
              "Kate:" in s and "Dan:" in s and "two people" in s, s[:180])
    check("the partner rule says what it did", "kept in frame as a partner" in info)

    shots = _shots_of(run_node(
        room + "\n\nKate and Dan kiss on the bed.\n\nKate smiles.\n\n"
        "Kate walks into the kitchen.\n\nKate pours a glass of water.",
        character_memory=mem, plan_only=True))
    check("a partner does not follow her into another room",
          "Dan:" not in shots[2] and "Dan:" not in shots[3], shots[3][:160])
    shots = _shots_of(run_node(
        room + "\n\nKate and Dan kiss on the bed.\n\nKate smiles.\n\nDan walks out."
        "\n\nKate lies still.", character_memory=mem, plan_only=True))
    check("...or stay once he has walked out", "Dan:" not in shots[3], shots[3][:160])
    shots = _shots_of(run_node(
        room + "\n\nKate and Dan sit on the bed.\n\nKate reads a book.",
        character_memory=mem, plan_only=True))
    check("a scene with no contact in it is unchanged", "Dan:" not in shots[1], shots[1][:160])


def test_the_anchor_and_the_body_agree():
    """REPORTED: the wrong genitalia is used, and a thong written in the anchor is
    dropped from the prompt.

    "Kate and Dan undress" undressed Dan alone, so later shots described his body and
    nothing of hers; the genitals were nobody's in particular; a shared anchor sentence
    went whole from a shot without Dan; and a thong the anchor put on her did not count
    as worn, so the jeans coming off told the shot her groin was bare over it."""
    print("\n=== the anchor and the body agree ===")
    mem = "Kate: she, 25, grey sweater, blue jeans.\nDan: he, 40, grey shirt, black trousers."
    shots = _shots_of(run_node(
        "A small bedroom.\n\nKate and Dan stand by the bed.\n\nKate and Dan undress.\n\n"
        "Kate and Dan lie on the bed.", character_memory=mem, plan_only=True))
    check("both of them are bare after both undress",
          "a woman's genitals" in shots[2] and "a man's genitals" in shots[2], shots[2][-240:])
    shots = _shots_of(run_node(
        "Kate sits on the bed.\n\nKate stands up.",
        anchor="A bedroom at night. Kate wears a black thong and Dan wears boxers.",
        character_memory="Kate: she, 25, blonde.\nDan: he, 40, beard.", plan_only=True))
    check("a thong in a shared anchor sentence survives a shot without Dan",
          all("black thong" in s and "boxers" not in s for s in shots), shots[0][:160])
    shots = _shots_of(run_node(
        "Kate sits on the bed.\n\nKate takes off her jeans.\n\nKate stands up.",
        anchor="A bedroom at night. Kate wears a black thong.",
        character_memory="Kate: she, 25, blue jeans, grey sweater.", plan_only=True))
    check("the anchor's thong keeps the hips covered when the jeans come off",
          not any("genitals" in s for s in shots) and all("black thong" in s for s in shots),
          shots[1][-200:])
    shots = _shots_of(run_node(
        "Kate sits on the bed.\n\nKate takes off her skirt.",
        anchor="A bedroom at night. Kate wears a red skirt over a black thong.",
        character_memory="Kate: she, 25, blonde.", plan_only=True))
    check("a garment named in anchor prose comes off by its own name",
          "The red skirt comes off" in shots[1] and "Kate wears a red skirt comes" not in shots[1],
          shots[1][:260])
    # A full strip takes the anchor's clothes too, off everyone it undresses.
    shots = _shots_of(run_node(
        "Kate and Dan stand by the bed.\n\nKate and Dan undress.\n\nKate and Dan lie on the bed."
        "\n\nKate and Dan kiss.",
        anchor="A bedroom at night. Kate wears a white bra and a black thong. Dan wears grey boxers.",
        character_memory=mem, plan_only=True))
    check("the stripping shot undresses both of them",
          "Everything Kate and Dan are wearing comes off" in shots[1]
          and "keeps on exactly what" not in shots[1], shots[1][:300])
    for i in (2, 3):
        check(f"shot {i + 1} lists none of the anchor's underwear",
              not any(g in shots[i] for g in ("white bra", "black thong", "grey boxers")),
              shots[i][:160])


def test_one_seed_is_one_noise_field_at_any_length():
    """REPORTED: the seed seems to change drastically at every beat; the clothes do not
    stay the same.

    The seed never changed -- but ComfyUI draws noise in memory order, so with shot
    lengths sized from each beat, the same seed was an unrelated noise field in every
    shot, not even sharing its first frame. Drawn frame by frame, a frame's noise no
    longer depends on how long its shot is."""
    print("\n=== one seed is one noise field at any length ===")
    R = S._runtime_module

    def lat(frames, audio):
        return FakeNested([torch.zeros(1, 24, frames, 4, 4), torch.zeros(1, 32, 2, audio)])

    sv, sa = R.chain_noise(lat(7, 50), 123).unbind()
    lv, la = R.chain_noise(lat(12, 90), 123).unbind()
    check("a short and a long shot share their frames' video noise",
          torch.equal(sv, lv[:, :, :7]))
    check("...and their audio noise", torch.equal(sa, la[..., :50]))
    check("the same shot on the same seed is the same noise",
          torch.equal(R.chain_noise(lat(7, 50), 123).unbind()[0], sv))
    check("another seed is another field",
          not torch.equal(R.chain_noise(lat(7, 50), 124).unbind()[0], sv))
    check("it is still unit gaussian noise", abs(float(lv.std()) - 1.0) < 0.1)
    fallback = lambda img, seed, inds=None: "comfy"
    check("a batch_index gets ComfyUI's own draw",
          R.chain_noise(lat(7, 50), 1, [0], fallback=fallback) == "comfy")
    check("...and so does a latent that is not H3's",
          R.chain_noise(torch.zeros(1, 4, 8, 8, 8, 2), 1, fallback=fallback) == "comfy")
    had = getattr(S.comfy.sample, "prepare_noise", None)
    S.comfy.sample.prepare_noise = fallback
    try:
        with R._ChainNoise():
            swapped = S.comfy.sample.prepare_noise is not fallback
            inside = S.comfy.sample.prepare_noise(lat(7, 50), 123).unbind()[0]
        check("sampling draws the chain's noise", swapped and torch.equal(inside, sv))
        check("...and ComfyUI's is put back after", S.comfy.sample.prepare_noise is fallback)
    finally:
        if had is None:
            del S.comfy.sample.prepare_noise
        else:
            S.comfy.sample.prepare_noise = had


def main():
    test_independent_adult_arm_actions()
    test_plan()
    test_render()
    test_a_cut_keeps_its_own_first_frame()
    test_a_room_change_with_no_walk_is_a_cut()
    test_extras_are_not_forbidden_by_the_body_count()
    test_a_walk_into_an_unlisted_room_is_still_walked()
    test_a_move_to_any_place_is_performed()
    test_an_unstated_frame_becomes_a_portrait()
    test_an_interrupt_releases_the_frame_buffer()
    test_a_posture_denied_is_not_a_posture_taken()
    test_the_face_plays_the_feeling_the_author_named()
    test_a_pronoun_pointing_away_keeps_the_person_it_means()
    test_a_feeling_belongs_to_the_face_the_beat_pins_it_on()
    test_a_bare_region_says_whose_body_it_is()
    test_a_plural_removal_still_comes_off()
    test_a_removal_undresses_only_its_wearer()
    test_a_pairing_the_beat_wrote_is_said_back()
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
    test_a_line_with_no_look_turns_the_faces_to_each_other()
    test_a_walk_along_a_place_arrives_in_it()
    test_a_rooms_description_waits_outside_that_room()
    test_a_covered_object_does_not_send_its_picture()
    test_the_anchor_survives_a_close_shot()
    test_mouths_stay_shut_with_no_line()
    test_a_grin_is_not_a_closed_mouth()
    test_nothing_tells_the_cast_to_hold_still()
    test_a_face_under_duress_is_not_a_portrait()
    test_her_whimper_does_not_free_his_mouth()
    test_a_line_is_locked_to_the_person_who_says_it()
    test_her_look_does_not_land_on_him()
    test_a_look_survives_the_next_beat()
    test_the_bed_survives_an_anchor()
    test_the_audio_branch_gets_a_soft_landing()
    test_an_anchor_says_when_it_has_taken_the_scenes_place()
    test_a_grim_film_is_grim_in_every_shot()
    test_a_kidnapping_reads_as_one_without_being_declared()
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
    test_the_soundtrack_is_the_models_own()
    test_shot_one_is_the_only_unpinned_shot()
    test_a_first_frame_can_be_the_set_instead_of_frame_one()
    test_the_age_reaches_the_shot_and_the_floor_holds()
    test_sound_survives_silencing()
    test_auto_sound_end_to_end()
    test_room_tone_under_every_shot()
    test_the_decode_keeps_the_vae_it_is_about_to_use()
    test_an_exclusive_sound_clause_never_denies_the_beat()
    test_an_intimate_scene_holds_its_frame_and_its_voices()
    test_a_vocal_shot_says_what_fills_the_gaps()
    test_the_chain_is_never_held_twice()
    test_the_position_may_only_be_written_once_in_the_scene()
    test_finished_shots_are_held_in_half_precision()
    test_detail_trend()
    test_the_chain_does_not_burn_in()
    test_the_take_does_not_cook_the_chain()
    test_timing_report()
    test_no_garment_is_ever_invented()
    test_nothing_wearable_is_ever_added()
    test_a_garment_keeps_its_description()
    test_a_removal_always_names_hands()
    test_hands_and_holds_follow_the_beat()
    test_one_line_is_one_voice()
    test_the_plan_says_whether_silence_can_be_applied()
    test_the_silent_latent_looks_like_silence()
    test_silence_pins_the_generated_audio()
    test_a_softened_handoff_is_not_a_keyframe()
    test_a_journey_reaches_the_shot()
    test_the_scene_does_not_reset()
    test_a_described_room_still_holds()
    test_the_sound_clause_is_inside_the_budget()
    test_the_anchor_is_not_read_as_a_room()
    test_a_garment_does_not_appear_at_the_first_frame()
    test_one_picture_two_people_is_reported()
    test_a_collar_chained_to_a_wall_stays_on()
    test_every_way_of_chaining_a_collar_to_a_wall()
    test_moving_towards_something_is_not_being_chained_to_it()
    test_two_restraints_put_on_together_both_survive()
    test_a_name_that_is_only_spoken_is_not_in_the_shot()
    test_a_verb_is_not_a_room()
    test_entering_a_room_is_a_journey_not_a_cut()
    test_no_guard_sentence_carries_a_negation()
    test_hardware_waits_for_the_beat_that_puts_it_on()
    test_the_applying_shot_says_where_the_limbs_finish()
    test_a_collar_in_the_sheet_is_held_like_hardware()
    test_an_undergarment_keeps_its_words_and_waits_for_its_picture()
    test_an_under_layer_belongs_to_somebody()
    test_the_picture_arrives_when_the_cover_moves()
    test_letting_the_cover_fall_puts_it_back()
    test_somebody_else_can_lift_your_skirt()
    test_pacing_reaches_the_thin_shots()
    test_a_line_is_spoken_in_one_language()
    test_undressing_does_not_spread()
    test_a_working_character_is_not_still_lying_down()
    test_an_instruction_is_not_the_action()
    test_a_breath_does_not_hold_the_branch_open()
    test_camera_framing_is_read_from_the_anchor()
    test_a_tight_frame_drops_wardrobe_it_cannot_contain()
    test_a_hidden_garments_lettering_goes_with_it()
    test_audio_sigma_reads_the_scheduler()
    test_audio_sigma_falls_back_to_the_closed_form()
    test_appearing_is_not_arriving()
    test_a_line_is_marked_however_it_is_punctuated()
    test_a_two_word_sheet_name_does_not_duplicate_her()
    test_dan_is_not_instantiated_twice()
    test_somebody_still_in_the_frame_is_not_back()
    test_a_cut_carries_the_people_across()
    test_a_thing_that_opens_itself_is_a_staged_change()
    test_a_walk_is_not_its_own_reverse()
    test_an_exact_line_is_yours_untouched()
    test_cuffs_are_not_described_as_a_chain()
    test_metal_hardware_is_told_what_it_is_made_of()
    test_hardware_closed_over_the_groin_stays_closed()
    test_which_way_up_a_lying_body_is_holds()
    test_a_body_not_in_the_shot_gets_no_position()
    test_the_limb_position_leads_the_shot()
    test_the_pronoun_swap_never_touches_your_words()
    test_verbatim_sends_your_text_and_nothing_else()
    test_an_untagged_reference_is_claimed_or_held()
    test_one_photographed_face_and_two_people()
    test_a_restraint_survives_the_shot_that_undresses_it()
    test_any_restraint_holds_from_shot_to_shot()
    test_a_length_goes_where_it_is_put()
    test_the_count_counts_who_the_text_names()
    test_a_shared_pose_names_nobody()
    test_a_lora_is_reported()
    test_the_camera_is_held_where_nothing_places_it()
    test_a_carried_room_is_not_a_second_picture_of_somebody()
    test_a_bare_region_is_said_on_every_shot()
    test_a_squat_survives_speech_and_undressing()
    test_what_comes_off_in_the_bathroom_stays_off()
    test_the_thong_comes_off_however_it_is_written()
    test_the_opening_does_not_name_people_a_shot_leaves_out()
    test_a_sheet_written_first_is_a_sheet()
    test_a_two_word_name_makes_a_sheet()
    test_one_person_under_two_names_is_described_once()
    test_a_bare_chest_belongs_to_whoever_undressed()
    test_looking_at_a_place_does_not_go_there()
    test_a_room_the_film_returns_to_is_carried()
    test_a_layer_under_a_removed_garment_stays_on()
    test_how_a_garment_comes_off_is_read_right()
    test_a_garment_put_back_on_in_prose_comes_back()
    test_a_garment_called_by_its_familys_word_is_found()
    test_a_pronoun_in_a_description_is_not_the_declared_one()
    test_a_name_used_as_a_word_stages_nobody()
    test_an_undeclared_pronoun_is_reported()
    test_hardware_is_named_only_on_its_own_wearer()
    test_a_soft_restraint_is_not_called_metal()
    test_one_garment_is_named_once_when_it_comes_off()
    test_hardware_in_a_hand_is_not_hardware_on_a_body()
    test_a_length_reaches_only_what_it_is_taken_around()
    test_the_shot_that_puts_it_on_says_so()
    test_a_layer_the_author_shows_stops_waiting()
    test_underwear_is_described_and_comes_all_the_way_off()
    test_a_restraint_stays_on_after_it_is_applied()
    test_the_budget_buys_as_many_guarantees_as_it_can()
    test_a_fall_keeps_its_landing_guard()
    test_a_dropped_clause_is_not_reported_as_sent()
    test_a_promoted_clause_opens_in_upper_case()
    test_the_reports_say_what_happened()
    test_a_line_is_counted_once()
    test_the_ceiling_is_the_ceiling()
    test_the_machine_with_the_line_is_the_one_that_speaks()
    test_a_written_sound_is_not_denied()
    test_a_sound_given_up_is_really_silenced()
    test_a_carried_frame_does_not_add_an_uncounted_body()
    test_one_person_gets_one_picture()
    test_an_untagged_picture_is_not_a_stranger()
    test_a_beat_that_moves_a_garment_keeps_the_cast()
    test_a_carried_clause_and_the_count_agree()
    test_a_pronoun_object_keeps_the_person_it_means()
    test_a_people_word_is_not_always_a_crowd()
    test_a_garment_set_down_is_not_a_garment_put_on()
    test_what_a_removal_uncovers_is_said_on_every_later_shot()
    test_a_full_stop_ends_a_clause()
    test_a_layer_is_covered_where_it_actually_sits()
    test_another_persons_clothes_do_not_answer_for_this_body()
    test_the_two_wardrobe_readers_agree()
    test_a_beam_in_a_roof_is_not_a_smile()
    test_the_shot_that_takes_it_off_is_not_told_where_it_sits()
    test_a_removal_stays_off_in_every_later_shot()
    test_a_scene_keeps_its_room_and_its_people()
    test_the_anchor_and_the_body_agree()
    test_one_seed_is_one_noise_field_at_any_length()
    print()
    if _fails:
        print(f"RESULT: {len(_fails)} FAILURE(S): " + "; ".join(_fails))
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
