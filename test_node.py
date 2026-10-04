import io
import os
import sys
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
import sampler as S

_fails = []


def check(label, ok, extra=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else f"   {extra}"))
    if not ok:
        _fails.append(label)


SCRIPT = """A dim cell with a cot. Mara <Picture 1> is a tall woman in a grey dress. Dan <Picture 2> is a guard.

Mara sits on the cot.

Dan handcuffs her wrists behind her back.
hold: Mara, handcuffs behind her back

Dan wraps duct tape around her mouth.
hold: Mara, duct tape over her mouth
seconds: 6

Dan says "Quiet." Mara looks at him.

cut
A corridor. Dan drags Mara along.

Dan pulls the tape off.
release: Mara, duct tape"""

IMG1, IMG2 = torch.full((1, 64, 64, 3), 0.1), torch.full((1, 64, 64, 3), 0.2)


def test_plan():
    print("\n=== script plan ===")
    shots = S.plan_shots(SCRIPT, 10.0, [IMG1, IMG2, None, None], False)
    check("six beats, six shots", len(shots) == 6, len(shots))
    p = [s["prompt"] for s in shots]
    check("the scene leads every shot and each beat follows word for word",
          all(x.startswith("A dim cell with a cot.") for x in p) and "Mara sits on the cot." in p[0]
          and "Dan says \"Quiet.\" Mara looks at him." in p[3], p[0])
    check("no directive reaches a prompt",
          not any(k in x.lower() for x in p for k in ("hold:", "release:", "seconds:")) and "\ncut" not in p[4], p[4])
    check("nothing is held before anything goes on", [s["held"] for s in shots[:2]] == ["", ""])
    check("a held item shows from the shot after it goes on",
          shots[2]["held"] == "Mara: handcuffs behind her back."
          and shots[3]["held"] == "Mara: handcuffs behind her back; duct tape over her mouth.", shots[3]["held"])
    check("the held line ends the prompt", p[3].endswith("Mara: handcuffs behind her back; duct tape over her mouth."))
    check("a released item leaves the held line in the shot it comes off",
          shots[5]["held"] == "Mara: handcuffs behind her back.", shots[5]["held"])
    check("seconds: sets that shot's length on the H3 grid",
          shots[2]["frames"] == S.rt.align_frame_count(144) and shots[0]["frames"] == 243,
          [s["frames"] for s in shots])
    check("cut starts the shot without the previous frame; the rest continue",
          [s["keyed"] for s in shots] == [False, True, True, True, False, True], [s["keyed"] for s in shots])
    check("speech is told apart from wordless beats", [s["wordless"] for s in shots] == [True, True, True, False, True, True])
    check("before anything is held both portraits ride, in order", shots[1]["refs"] == [IMG1, IMG2]
          and "Mara <Picture 1>" in p[1] and "Dan <Picture 2>" in p[1])
    check("a held person's portrait is dropped on a continued shot and the rest renumbered",
          shots[2]["refs"] == [IMG2] and "Mara is a tall woman" in p[2] and "Dan <Picture 1>" in p[2], p[2])
    check("after a cut the portrait rides again", shots[4]["refs"] == [IMG1, IMG2])
    one = S.plan_shots("Mara waves.", 5.0, [None] * 4, True)
    check("a single paragraph is one shot, keyed on a first frame", len(one) == 1 and one[0]["keyed"]
          and one[0]["prompt"] == "Mara waves.")
    t, refs = S.shot_pictures("Ana (<Picture 3>) waits. <Picture 1> hums.", [IMG1, None, None], [])
    check("a tag with no image is removed with its brackets", t == "Ana waits. <Picture 1> hums." and refs == [IMG1], t)


def test_pose_plan():
    print("\n=== pose plan ===")
    shots = S.plan_shots(SCRIPT, 10.0, [None] * 4, False)
    b = [s["bound"] for s in shots]
    check("no pose before anything is held", b[0] == {})
    check("the cuffing shot latches the arms", b[1].get("Mara", {}).get("latch_limbs") == ("arms",)
          and b[1]["Mara"]["arms"] == "behind the back" and shots[1]["latch_after"] == 98, b[1])
    check("held cuffs are repaired from frame 0", b[2].get("Mara", {}).get("latch_limbs") == ()
          and shots[2]["latch_after"] is None, b[2])
    check("tape coming off the mouth leaves the cuffs held", b[5].get("Mara", {}).get("arms") == "behind the back")
    check("limb facts", S.limb_facts("wrists zip tied in front")[:2] == ("in front of the body", "")
          and S.limb_facts("ankles chained together")[1:] == ("ankles together", False, True)
          and S.limb_facts("duct tape over her mouth")[:2] == ("", "")
          and S.limb_facts("duct tape wrapped around her waist and legs")[:2] == ("", "")
          and S.limb_facts("ankles tied to her wrists")[1] == "ankles to the wrists"
          and S.limb_facts("handcuffed to the bed frame")[2])
    anchored = S.plan_shots("A room.\n\nMara waits.\nhold: Mara, wrists cuffed behind her back to the pipe\n\nMara waits.",
                            5.0, [None] * 4, False)
    check("a restraint fastened to an object gets no pose", anchored[1]["bound"] == {}, anchored[1]["bound"])
    off = S.plan_shots("A room.\nhold: Mara, handcuffs behind her back\n\nDan unlocks the cuffs.\nrelease: Mara, handcuffs",
                       5.0, [None] * 4, False)
    check("cuffs held in the scene are on from shot 1, and the shot they come off gets no pose",
          off[0]["held"] == "" and off[0]["bound"] == {}, off[0])


class FakeClip:
    def tokenize(self, text, **kw):
        return (text, kw.get("minimax_ref_items"))

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 4), {"tokens": tokens}]]


class FakeVAE:
    def encode(self, img):
        return torch.zeros(1, 24, 1, int(img.shape[1]) // 16, int(img.shape[2]) // 16)


class FakeAudioVAE:
    audio_sample_rate = 100

    def encode(self, wav):
        return torch.zeros(1, 32, 2, 80)


def test_conditioning():
    print("\n=== conditioning ===")
    hand = torch.rand(1, 64, 96, 3)
    cond, latent, fc, pinned = S.cnd.build_conditioning(FakeClip(), FakeVAE(), FakeAudioVAE(), "x", 96, 64, 41,
                                                       handoff=hand, refs=[IMG1], silent=True)
    vals = cond[0][1]
    items = vals["tokens"][1]
    check("refs then the handoff go to the text encoder as pictures", len(items) == 2 and items[1]["data"].shape == (1, 64, 96, 3))
    check("the handoff is pinned as the keyframe at frame 0", vals["minimax_keyframes"][0]["resolved_frame_index"] == 0)
    check("the reference rows ride with their noise aug", len(vals["minimax_refs"]) == 1
          and vals["minimax_visual_cond_noise_aug"] == 0.999)
    check("a wordless shot's audio is pinned to silence", pinned and float(latent["noise_mask"].unbind()[1].max()) == 0.0)
    _, latent, _, pinned = S.cnd.build_conditioning(FakeClip(), FakeVAE(), FakeAudioVAE(), "x", 96, 64, 41,
                                                    lead_seconds=0.5)
    m = latent["noise_mask"].unbind()[1]
    check("a spoken shot holds only its first half second", pinned and float(m[..., :20].max()) == 0.0
          and float(m[..., 20:].min()) == 1.0)


def test_helpers():
    print("\n=== sampling helpers ===")
    check("presets scale to megapixels on the 32 grid", S.frame_size("16:9", 1.0) == (1344, 768)
          and S.frame_size("16:9", 0) == (1344, 768) and S.frame_size("9:16", 0.5) == (544, 960))
    hf = SimpleNamespace(get_attachment=lambda k: {"hyperflow": "true", "hyperflow_gate": "0.25",
                                                   "hyperflow_sigmas": "[1.0, 0.5, 0.0]"} if k == "lora_metadata" else None)
    h = S.hyperflow_lora(hf)
    check("Hyperflow is read from the LoRA's metadata", h and h["sigmas"] == (1.0, 0.5, 0.0) and h["gate"] == 0.25, h)
    check("...or from the file name in the graph",
          S.hyperflow_lora(SimpleNamespace(), {"1": {"inputs": {"lora_name": "minimax_h3_hyperflow_8step.safetensors"}}}))
    check("no Hyperflow, no grid", S.hyperflow_lora(SimpleNamespace(), {}) is None)
    sig = S.hyperflow_sigmas(S.HYPERFLOW_SIGMAS, 12.0)
    check("the grid is shifted at 12 and keeps its ends", abs(float(sig[0]) - 1.0) < 1e-6 and float(sig[-1]) == 0.0
          and abs(float(sig[4]) - 12 * 0.5 / 6.5) < 1e-5)
    base = [1.0, 0.9, 0.7, 0.4, 0.0]
    landed = S.insert_audio_landing(base, 12.0, 3.0)
    check("an audio landing step goes in before 0 only when the audio lands coarse",
          len(landed) == 6 and 0.0 < landed[-2] < 0.4 and S.insert_audio_landing([1.0, 0.01, 0.0], 12.0, 3.0) == [1.0, 0.01, 0.0])


class FakeDet:
    made = 0

    def __init__(self, *a):
        FakeDet.made += 1
        self.released = False

    def close(self):
        pass

    def release(self):
        self.released = True


def render(script, pose_cn=None, pose_result=None, oom_pass2=False, plan_only=False, lat_up=None, frames_up=None,
           **run_kw):
    calls = {"cond": [], "sample": [], "pose": [], "decode": [], "tail": [], "frames_up": []}
    saved = (S.check_vaes, S.prepare_model, S.cnd.build_conditioning, S.sample, S.decode, S.rt._evict_all_but,
             S.rt._deep_cleanup, S.pose_pass, S.pose, S.up.upscale_latent, S.up.upscale_frames, S.rt._decode_video)
    S.check_vaes = lambda v, a: None
    S.prepare_model = lambda m, st, sn, sc, sg, sv, sa, g: (m, st, sn, None, torch.linspace(1, 0, st + 1), False, ["prepared"])

    def build(clip, vae, avae, prompt, w, h, length, handoff=None, refs=(), silent=False, lead_seconds=0.0):
        calls["cond"].append({"prompt": prompt, "handoff": handoff, "silent": silent, "lead": lead_seconds})
        return "cond", {"shot": len(calls["cond"]), "fc": length}, length, silent

    def sample(model, cond, neg, latent, seed, steps, sn, sc, sg):
        calls["sample"].append({"model": model, "shot": latent["shot"], "seed": seed})
        if oom_pass2 and model == "patched":
            raise RuntimeError("CUDA out of memory")
        lat = comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 4, 2, 2), torch.zeros(1, 32, 2, 8)))
        return {"samples": lat, "shot": latent["shot"], "fc": latent["fc"], "model": model}

    def decode(vae, avae, model, out, tiled=False):
        fc, k = out["fc"], out["shot"]
        lh, lw = out["samples"].unbind()[0].shape[-2:]
        calls["decode"].append({"shot": k, "tiled": tiled, "size": (int(lh) * 4, int(lw) * 4)})
        imgs = torch.full((fc, int(lh) * 4, int(lw) * 4, 3), k / 10.0) + (torch.arange(fc).float() / 1e4).view(fc, 1, 1, 1)
        if out["model"] == "patched":
            imgs += 0.05
        return imgs, {"waveform": torch.full((1, 2, round(fc * 1000 / 24)), float(k)), "sample_rate": 1000}

    def tail(vae, out, tiled, **kw):
        lat = out["samples"]
        calls["tail"].append({"frames": int(lat.shape[2]), "tiled": tiled})
        return torch.full((3, int(lat.shape[-2]) * 4, int(lat.shape[-1]) * 4, 3), 0.9)

    def pose_pass(det, imgs, shot, carry, model, cn, vae, latent, st, pe, window, w, h):
        calls["pose"].append({"shot": shot["n"], "carry": carry, "latch": shot["latch_after"]})
        return (pose_result or (lambda n: ("patched", {"broken": True, "boxes_last": {"Mara": n}})))(shot["n"])

    def frames_fake(frames, mode, name, target):
        calls["frames_up"].append({"mode": mode, "name": name, "target": target, "n": int(frames.shape[0])})
        return frames_up(frames, mode, name, target)

    S.cnd.build_conditioning, S.sample, S.decode, S.pose_pass = build, sample, decode, pose_pass
    S.rt._evict_all_but = lambda *a, **k: None
    S.rt._deep_cleanup = lambda: None
    S.rt._decode_video = tail
    S.pose = SimpleNamespace(pose_status=lambda m, cn, st, tt: (True, ""), PoseDetector=FakeDet)
    if lat_up is not None:
        S.up.upscale_latent = lat_up
    if frames_up is not None:
        S.up.upscale_frames = frames_fake
    try:
        out = S.H3LongVideos().run("base", FakeClip(), "vae", "avae", script, "16:9", 0.01, 2.0, 8, "euler", "simple",
                                   7, pose_controlnet=pose_cn, plan_only=plan_only, **run_kw)
    finally:
        (S.check_vaes, S.prepare_model, S.cnd.build_conditioning, S.sample, S.decode, S.rt._evict_all_but,
         S.rt._deep_cleanup, S.pose_pass, S.pose, S.up.upscale_latent, S.up.upscale_frames, S.rt._decode_video) = saved
    return out, calls


def test_render():
    print("\n=== render chain ===")
    out, calls = render(SCRIPT)
    video, audio, info = out[0], out[1], out[2]
    fc = [S.rt.align_frame_count(48)] * 2 + [S.rt.align_frame_count(144)] + [S.rt.align_frame_count(48)] * 3
    keyed_after_first = 4
    check("every shot sampled once with the node's seed", [c["shot"] for c in calls["sample"]] == [1, 2, 3, 4, 5, 6]
          and all(c["seed"] == 7 for c in calls["sample"]))
    check("frame one of each continued shot is trimmed", int(video.shape[0]) == sum(fc) - keyed_after_first,
          (int(video.shape[0]), sum(fc)))
    check("audio matches the picture", abs(int(audio["waveform"].shape[-1]) - int(video.shape[0]) * 1000 / 24) <= 6,
          (int(audio["waveform"].shape[-1]), int(video.shape[0])))
    hand = [c["handoff"] for c in calls["cond"]]
    check("each continued shot opens on the last frame of the one before", hand[0] is None and hand[4] is None
          and abs(float(hand[1].mean()) - (0.1 + (fc[0] - 1) / 1e4)) < 1e-4
          and abs(float(hand[5].mean()) - (0.5 + (fc[4] - 1) / 1e4)) < 1e-4, [None if x is None else float(x.mean()) for x in hand])
    check("wordless shots are silenced and the spoken one holds its lead",
          [c["silent"] for c in calls["cond"]] == [True, True, True, False, True, True] and calls["cond"][3]["lead"] == 0.5)
    check("no pose without the controlnet", calls["pose"] == [] and FakeDet.made == 0)
    check("info has a line per shot and the totals", sum(1 for x in info.split(" | ") if x.startswith("shot ")) == 6
          and "rendered in" in info, info)
    check("script holds what each shot was told", out[3].count("[shot ") == 6 and "Mara: handcuffs behind her back." in out[3])

    out, calls = render(SCRIPT, pose_cn="cn")
    check("the pose check runs on every shot with a held limb", [c["shot"] for c in calls["pose"]] == [2, 3, 4, 5, 6])
    check("the cuffing shot latches; later shots repair",
          calls["pose"][0]["latch"] == int(S.math.ceil(S.POSE_LATCH_FROM * fc[1])) and calls["pose"][1]["latch"] is None,
          [c["latch"] for c in calls["pose"]])
    check("who is who carries across continued shots and stops at a cut",
          [c["carry"] for c in calls["pose"]] == [None, ({"Mara": 2}, None), ({"Mara": 3}, None), None, ({"Mara": 5}, None)],
          [c["carry"] for c in calls["pose"]])
    passes = [(c["shot"], c["model"]) for c in calls["sample"]]
    check("a repaired shot samples a second time on the patched model, same seed",
          passes.count((3, "patched")) == 1 and passes.count((3, "base")) == 1 and all(c["seed"] == 7 for c in calls["sample"]))
    check("the repaired frames are the ones kept", abs(float(out[0][0].mean()) - 0.1) < 1e-3
          and any(abs(float(f.mean()) - 0.35) < 2e-3 for f in out[0]))
    check("info says what pose control did", out[2].count("pose repaired") == 5, out[2])

    out, calls = render(SCRIPT, pose_cn="cn", oom_pass2=True)
    check("a second pass that runs out of memory keeps the first and turns pose control off",
          [c["shot"] for c in calls["pose"]] == [2] and "second pass failed" in out[2]
          and int(out[0].shape[0]) == sum(fc) - keyed_after_first, ([c["shot"] for c in calls["pose"]], out[2]))

    out, calls = render(SCRIPT, pose_cn="cn", pose_result=lambda n: (None, {"skipped": "not guessing"}))
    check("a skipped check keeps the first pass", all(c["model"] == "base" for c in calls["sample"])
          and "pose not guessing" in out[2], out[2])

    out, calls = render(SCRIPT, plan_only=True)
    check("plan_only samples nothing and still returns the script", calls["sample"] == [] and out[3].count("[shot ") == 6
          and "pose latch: Mara behind the back" in out[2], out[2])


def test_scene_inputs():
    print("\n=== anchor and character memory ===")
    shots = S.plan_shots("Mara sits.\n\nMara stands.", 5.0, [None] * 4, False,
                         memory="Mara: a tall woman in a grey dress.\nhold: Mara, handcuffs behind her back",
                         anchor="Cinematic, 35mm, a dim cell.")
    check("with an anchor every paragraph is a beat", len(shots) == 2 and "Mara sits." in shots[0]["prompt"])
    check("the anchor leads, then the character memory, then the beat",
          shots[0]["prompt"].startswith("Cinematic, 35mm, a dim cell.\n\nMara: a tall woman in a grey dress.\n\nMara sits."),
          shots[0]["prompt"])
    check("a hold: in the character memory is on from the first shot",
          shots[0]["held"] == "Mara: handcuffs behind her back." and "hold:" not in shots[0]["prompt"], shots[0]["held"])
    plain = S.plan_shots("A cell.\n\nMara sits.", 5.0, [None] * 4, False, memory="Dan: a guard.")
    check("without an anchor the first paragraph is the scene, before the character memory",
          len(plain) == 1 and plain[0]["prompt"] == "A cell.\n\nDan: a guard.\n\nMara sits.", plain[0]["prompt"])
    out, _ = render(SCRIPT, plan_only=True, character_memory="Dan: a guard.", negative="x", ambient_level=0.2)
    check("inputs left over from an older workflow are named in info, not a crash",
          "ignored inputs from an older version of this node: ambient_level, negative" in out[2], out[2])
    check("...and character_memory reaches every shot", out[3].count("Dan: a guard.") == 6, out[3])


class V3Node:
    @classmethod
    def define_schema(cls):
        return None

    @classmethod
    def execute(cls, **kw):
        cls.seen.append(kw)
        return SimpleNamespace(result=(cls.answer(kw),))


def fake_v3(answer):
    return type("FakeV3", (V3Node,), {"seen": [], "answer": staticmethod(answer)})


class V1Node:
    FUNCTION = "go"

    def go(self, **kw):
        return (kw["x"] * 2,)


def test_upscale_module():
    print("\n=== upscalers ===")
    maps = S.up.nodes.NODE_CLASS_MAPPINGS
    saved = dict(maps)
    try:
        check("a V1 node is called through its FUNCTION", S.up.run_node(V1Node, x=21) == 42)
        lat = fake_v3(lambda kw: {"samples": torch.zeros(1, 24, 3, 8, 12)})
        maps["MinimaxH3LatentUpscaler3D"] = lat
        v = torch.zeros(1, 24, 3, 4, 6)
        upv, note = S.up.upscale_latent(v, "minimax_h3_latent_upscaler_3d_fp32.pth", 2.0)
        kw = lat.seen[-1] if lat.seen else {}
        check("the latent upscaler gets its current inputs (mode, align, chunking, device, precision)",
              upv.shape == (1, 24, 3, 8, 12) and note == "" and kw.get("enable_chunking") is True
              and kw.get("mode") == {"mode": "scale by multiplier", "scale": 2.0} and kw.get("align") == 32
              and kw.get("device") in ("cuda", "cpu") and kw.get("precision") in ("fp16", "fp32")
              and kw.get("model_name") == "minimax_h3_latent_upscaler_3d_fp32.pth", kw)
        check("off or a scale of 1 does nothing",
              S.up.upscale_latent(v, "off", 2.0) == (v, "") and S.up.upscale_latent(v, "m", 1.0) == (v, ""))
        maps["MinimaxH3LatentUpscaler3D"] = fake_v3(lambda kw: {"samples": torch.zeros(1, 24, 2, 8, 12)})
        same, note = S.up.upscale_latent(v, "m", 2.0)
        check("a changed frame count is refused", same is v and "unexpected shape" in note, note)
        maps["MinimaxH3LatentUpscaler3D"] = fake_v3(lambda kw: 1 / 0)
        same, note = S.up.upscale_latent(v, "m", 2.0)
        check("a failing upscaler keeps the latent and says why", same is v and "ZeroDivisionError" in note, note)
        del maps["MinimaxH3LatentUpscaler3D"]
        same, note = S.up.upscale_latent(v, "m", 2.0)
        check("without the pack the latent is kept and info says why", same is v and "node pack" in note, note)
        frames = torch.rand(6, 32, 48, 3)
        rtx = fake_v3(lambda kw: kw["images"].repeat_interleave(2, 1).repeat_interleave(2, 2))
        maps["RTXVideoSuperResolution"] = rtx
        out, note = S.up.upscale_frames(frames, "rtx", "none", 0)
        check("RTX gets images, a scale-by resize and a quality, in batches",
              out.shape == (6, 64, 96, 3) and len(rtx.seen) == 2 and rtx.seen[0]["quality"] == "ULTRA"
              and rtx.seen[0]["resize_type"] == {"resize_type": "scale by multiplier", "scale": 2.0} and "RTX" in note,
              (tuple(out.shape), note))
        out, note = S.up.upscale_frames(frames, "rtx", "none", 128)
        check("a target short edge picks the RTX factor and fits the result",
              rtx.seen[-1]["resize_type"]["scale"] == 4.0 and out.shape[1:3] == (128, 192), (rtx.seen[-1], tuple(out.shape)))
        loader = fake_v3(lambda kw: "model:" + kw["model_name"])
        apply = fake_v3(lambda kw: kw["image"].repeat_interleave(2, 1).repeat_interleave(2, 2))
        maps["UpscaleModelLoader"], maps["ImageUpscaleWithModel"] = loader, apply
        out, note = S.up.upscale_frames(frames, "model", "RealESRGAN_x2.pth", 0)
        check("an upscale model is loaded once and run in batches", out.shape == (6, 64, 96, 3) and len(loader.seen) == 1
              and apply.seen[0]["upscale_model"] == "model:RealESRGAN_x2.pth" and "RealESRGAN_x2" in note, note)
        out, note = S.up.upscale_frames(frames, "lanczos", "none", 64)
        check("lanczos fits the short edge on the 32 grid", out.shape == (6, 64, 96, 3) and note == "short edge 64px", note)
        del maps["RTXVideoSuperResolution"]
        out, note = S.up.upscale_frames(frames, "rtx", "none", 64)
        check("a missing RTX node still fits the target and says why", out.shape == (6, 64, 96, 3) and "not installed" in note,
              note)
        check("off does nothing", S.up.upscale_frames(frames, "off", "none", 64)[0] is frames)
    finally:
        maps.clear()
        maps.update(saved)


def test_upscale_render():
    print("\n=== upscaling in the chain ===")
    model = "minimax_h3_latent_upscaler_3d_fp32.pth"

    def doubled(video, name, scale):
        return torch.zeros(tuple(video.shape[:3]) + (video.shape[3] * 2, video.shape[4] * 2)), ""
    out, calls = render(SCRIPT, lat_up=doubled, latent_upscale=model)
    check("every shot is decoded from its upscaled latent, tiled",
          len(calls["decode"]) == 6 and all(d["tiled"] and d["size"] == (16, 16) for d in calls["decode"]), calls["decode"][:2])
    check("the video comes out at the upscaled size", tuple(out[0].shape[1:3]) == (16, 16), tuple(out[0].shape))
    hand = calls["cond"][1]["handoff"]
    check("the next shot opens on the last frame decoded at the sampled size",
          len(calls["tail"]) == 6 and all(t["tiled"] for t in calls["tail"])
          and tuple(hand.shape[1:3]) == (8, 8) and abs(float(hand.mean()) - 0.9) < 1e-6,
          (calls["tail"][:1], None if hand is None else tuple(hand.shape)))
    out, calls = render(SCRIPT, pose_cn="cn", lat_up=doubled, latent_upscale=model)
    check("the pose check reads the sampled size and the kept shot is the upscaled one",
          [d["size"] for d in calls["decode"] if d["shot"] == 3] == [(8, 8), (16, 16)]
          and tuple(out[0].shape[1:3]) == (16, 16), [d for d in calls["decode"] if d["shot"] == 3])

    def flaky(video, name, scale):
        flaky.n += 1
        return (doubled(video, name, scale) if flaky.n <= 2 else (video, "latent upscale failed (RuntimeError: boom)"))
    flaky.n = 0
    out, calls = render(SCRIPT, lat_up=flaky, latent_upscale=model)
    check("a shot whose upscale fails is fitted to the others' size, and info says so once",
          tuple(out[0].shape[1:3]) == (16, 16) and out[2].count("latent upscale failed") == 1, out[2])
    out, calls = render(SCRIPT, frames_up=lambda f, m, n, t: (f[:, ::2, ::2], "frames note"),
                        upscale="lanczos", upscale_target_short_edge=4)
    check("the finished video goes through the frame upscaler once, with its settings",
          len(calls["frames_up"]) == 1 and calls["frames_up"][0]["mode"] == "lanczos"
          and calls["frames_up"][0]["target"] == 4 and "frames note" in out[2] and tuple(out[0].shape[1:3]) == (4, 4),
          (calls["frames_up"], tuple(out[0].shape)))


def test_upscalers_real():
    print("\n=== the installed upscalers on the CPU (optional) ===")
    root = os.path.normpath(os.path.join(HERE, "..", ".."))
    maps = S.up.nodes.NODE_CLASS_MAPPINGS
    saved = dict(maps)
    try:
        try:
            import importlib.util
            path = os.path.join(root, "custom_nodes", "Comfyui_Minimax_h3_latent_Upscaler", "nodes",
                                "minimax_h3_latent_upscaler_3d.py")
            spec = importlib.util.spec_from_file_location("h3_latent_upscaler_3d", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            maps.update(mod.NODE_CLASS_MAPPINGS)
            name = next(n for n in S.up.latent_models() if n != "off")
        except Exception as e:
            print(f"  NOTE  the latent upscaler pack or its model is not available ({type(e).__name__}: {e}); skipped")
        else:
            upv, note = S.up.upscale_latent(torch.randn(1, 24, 2, 6, 8), name, 2.0)
            check(f"the installed latent upscaler runs with {name}", tuple(upv.shape) == (1, 24, 2, 12, 16) and note == "",
                  (tuple(upv.shape), note))
        try:
            import comfy_extras.nodes_upscale_model as U
            maps["UpscaleModelLoader"], maps["ImageUpscaleWithModel"] = U.UpscaleModelLoader, U.ImageUpscaleWithModel
            name = next(n for n in S.up.frame_models() if "x2" in n.lower())
        except Exception as e:
            print(f"  NOTE  ComfyUI's upscale-model nodes or an x2 model are not available ({type(e).__name__}: {e}); skipped")
        else:
            out, note = S.up.upscale_frames(torch.rand(2, 32, 48, 3), "model", name, 0)
            check(f"ComfyUI's upscale-model nodes run with {name}", tuple(out.shape) == (2, 64, 96, 3) and "failed" not in note,
                  (tuple(out.shape), note))
    finally:
        maps.clear()
        maps.update(saved)


def main():
    test_plan()
    test_pose_plan()
    test_conditioning()
    test_helpers()
    test_render()
    test_scene_inputs()
    test_upscale_module()
    test_upscale_render()
    test_upscalers_real()
    print("\nRESULT: " + ("ALL PASSED" if not _fails else f"{len(_fails)} FAILURE(S): " + "; ".join(_fails)))
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
