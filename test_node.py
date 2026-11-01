import io
import json
import os
import re
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
    check("the scene leads every shot and each beat follows word for word, its line in H3's speech tags",
          all(x.startswith("A dim cell with a cot.") for x in p) and "Mara sits on the cot." in p[0]
          and "Dan says <d>Quiet.</d> Mara looks at him." in p[3], p[3])
    check("no directive reaches a prompt",
          not any(k in x.lower() for x in p for k in ("hold:", "release:", "seconds:")) and "\ncut" not in p[4], p[4])
    check("nothing is held before anything goes on", [s["held"] for s in shots[:2]] == ["", ""])
    check("a held item shows from the shot after it goes on",
          shots[2]["held"] == "Mara: handcuffs behind her back."
          and shots[3]["held"] == "Mara: handcuffs behind her back; duct tape over her mouth.", shots[3]["held"])
    check("the held line opens the shot's closing notes",
          p[3].split("\n\n")[-1].startswith("Mara: handcuffs behind her back; duct tape over her mouth."), p[3])
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
    check("after a cut, someone last seen alone gets that look and the other portrait rides again",
          shots[4]["recover"] == ["Mara"] and shots[4]["refs"] == [IMG2], (shots[4]["recover"], len(shots[4]["refs"])))
    one = S.plan_shots("Mara waves.", 5.0, [None] * 4, True)
    check("a single paragraph is one shot, keyed on a first frame", len(one) == 1 and one[0]["keyed"]
          and one[0]["prompt"].startswith("Mara waves.\n\n"), one[0]["prompt"])
    t, refs, _ = S.shot_pictures("Ana (<Picture 3>) waits. <Picture 1> hums.", [IMG1, None, None], [])
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
    import folder_paths
    was = folder_paths.get_filename_list
    folder_paths.get_filename_list = lambda kind: ["minimax_h3_hyperflow_8step.safetensors"]
    try:
        check("...or from the file name in the graph",
              S.hyperflow_lora(SimpleNamespace(), {"1": {"inputs": {"lora_name": "minimax_h3_hyperflow_8step.safetensors"}}}))
    finally:
        folder_paths.get_filename_list = was
    check("no Hyperflow, no grid", S.hyperflow_lora(SimpleNamespace(), {}) is None)
    sig = S.hyperflow_sigmas(S.HYPERFLOW_SIGMAS, 12.0)
    check("the grid is shifted at 12 and keeps its ends", abs(float(sig[0]) - 1.0) < 1e-6 and float(sig[-1]) == 0.0
          and abs(float(sig[4]) - 12 * 0.5 / 6.5) < 1e-5)
    base = [1.0, 0.9, 0.7, 0.4, 0.0]
    landed = S.insert_audio_landing(base, 12.0, 3.0)
    check("an audio landing step goes in before 0 only when the audio lands coarse",
          len(landed) == 6 and 0.0 < landed[-2] < 0.4 and S.insert_audio_landing([1.0, 0.01, 0.0], 12.0, 3.0) == [1.0, 0.01, 0.0])


class Patched:
    def __init__(self, keys):
        self.wrappers = {S.pe.WrappersMP.DIFFUSION_MODEL: {k: [] for k in keys}}
        self.model_options = {"transformer_options": {}}

    def clone(self):
        return self

    def get_model_object(self, name):
        if name != "model_sampling":
            raise AttributeError(name)
        import comfy.model_sampling as ms
        return type("Flow", (ms.ModelSamplingDiscreteFlow, ms.CONST), {})()

    def add_object_patch(self, *a):
        pass

    def add_wrapper_with_key(self, *a):
        pass


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
           seconds=2.0, dwpose=False, check=None, auto=None, fast=False, **run_kw):
    calls = {"cond": [], "sample": [], "pose": [], "decode": [], "tail": [], "frames_up": [], "draft": []}
    saved = (S.check_vaes, S.prepare_model, S.cnd.build_conditioning, S.sample, S.decode, S.rt._evict_all_but,
             S.rt._deep_cleanup, S.pose_pass, S.pose, S.up.upscale_latent, S.up.upscale_frames, S.rt._decode_video,
             S.rt._decode_audio, S.pose_read, S.auto_pose_patch, S.is_fast_h3)
    calls["check"] = []
    S.is_fast_h3 = lambda m: fast
    S.check_vaes = lambda v, a: None
    S.prepare_model = lambda m, st, sn, sc, sg, sv, sa, g: (m, st, sn, None, torch.linspace(1, 0, st + 1), ["prepared"])

    def build(clip, vae, avae, prompt, w, h, length, handoff=None, refs=(), silent=False, lead_seconds=0.0, text=None):
        if text is not None:
            calls["draft"].append({"size": (w, h), "text": text, "handoff": handoff, "refs": list(refs)})
            return "small", {"shot": len(calls["cond"]), "fc": length, "draft": True}, length, silent
        calls["cond"].append({"prompt": prompt, "handoff": handoff, "silent": silent, "lead": lead_seconds,
                              "refs": list(refs), "size": (w, h)})
        return "cond", {"shot": len(calls["cond"]), "fc": length}, length, silent

    def sample(model, cond, neg, latent, seed, steps, sn, sc, sg):
        mask = latent.get("noise_mask")
        calls["sample"].append({"model": model, "shot": latent["shot"], "seed": seed, "foley": mask is not None,
                                "draft": latent.get("draft", False), "cond": cond,
                                "mask": None if mask is None else [float(m.max()) for m in mask.unbind()]})
        if oom_pass2 and model == "patched":
            raise RuntimeError("CUDA out of memory")
        if mask is not None:
            video = latent["samples"].unbind()[0]
            lat = comfy.nested_tensor.NestedTensor((video, torch.zeros(1, 32, 2, 8)))
            return {"samples": lat, "shot": latent["shot"], "fc": latent["fc"],
                    "model": "patched" if float(video.mean()) == 1.0 else "base"}
        video = torch.full((1, 24, 4, 2, 2), 1.0 if model == "patched" else 0.0)
        lat = comfy.nested_tensor.NestedTensor((video, torch.zeros(1, 32, 2, 8)))
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
        calls["pose"].append({"shot": shot["n"], "carry": carry, "latch": shot["latch_after"],
                              "size": tuple(imgs.shape[1:3])})
        return (pose_result or (lambda n: ("patched", {"broken": True, "boxes_last": {"Mara": n}})))(shot["n"])

    def frames_fake(frames, mode, name, target):
        calls["frames_up"].append({"mode": mode, "name": name, "target": target, "n": int(frames.shape[0])})
        return frames_up(frames, mode, name, target)

    S.cnd.build_conditioning, S.sample, S.decode, S.pose_pass = build, sample, decode, pose_pass
    S.rt._evict_all_but = lambda *a, **k: None
    S.rt._deep_cleanup = lambda: None
    S.rt._decode_video = tail
    S.rt._decode_audio = lambda avae, out: {"waveform": torch.full((1, 2, round(out["fc"] * 1000 / 24)), 9.0),
                                            "sample_rate": 1000}
    S.pose = SimpleNamespace(pose_status=lambda m, cn, st: (True, ""), PoseDetector=FakeDet,
                             dwpose_status=lambda: (dwpose, ""))

    def read_pose(det, imgs, shot, carry, w, h):
        calls["check"].append(shot["n"])
        return None, (check or (lambda n, k: {"broken": False}))(shot["n"], calls["check"].count(shot["n"]))
    S.pose_read = read_pose
    S.auto_pose_patch = lambda: auto or (None, "")
    if lat_up is not None:
        S.up.upscale_latent = lat_up
    if frames_up is not None:
        S.up.upscale_frames = frames_fake
    try:
        out = S.H3LongVideos().run("base", FakeClip(), "vae", "avae", script, "16:9", 0.01, seconds, 8, "euler", "simple",
                                   7, pose_controlnet=pose_cn, plan_only=plan_only, **run_kw)
    finally:
        (S.check_vaes, S.prepare_model, S.cnd.build_conditioning, S.sample, S.decode, S.rt._evict_all_but,
         S.rt._deep_cleanup, S.pose_pass, S.pose, S.up.upscale_latent, S.up.upscale_frames, S.rt._decode_video,
         S.rt._decode_audio, S.pose_read, S.auto_pose_patch, S.is_fast_h3) = saved
    return out, calls


def test_render():
    print("\n=== render chain ===")
    out, calls = render(SCRIPT)
    video, audio, info = out[0], out[1], out[2]
    fc = [S.rt.align_frame_count(48)] * 2 + [S.rt.align_frame_count(144)] + [S.rt.align_frame_count(67)] + \
        [S.rt.align_frame_count(48)] * 2
    keyed_after_first = 4
    check("every shot sampled once with the node's seed",
          [c["shot"] for c in calls["sample"] if not c["foley"]] == [1, 2, 3, 4, 5, 6] and all(c["seed"] == 7 for c in calls["sample"]))
    foley = [c for c in calls["sample"] if c["foley"]]
    check("a shot without dialogue but with sounds gets its sound made for the finished picture: video kept, audio made",
          [c["shot"] for c in foley] == [2, 3, 5, 6] and all(c["mask"] == [0.0, 1.0] and c["model"] == "base" for c in foley)
          and "sound made for the finished picture" in info, [(c["shot"], c["mask"]) for c in foley])
    check("frame one of each continued shot is trimmed", int(video.shape[0]) == sum(fc) - keyed_after_first,
          (int(video.shape[0]), sum(fc)))
    check("audio matches the picture", abs(int(audio["waveform"].shape[-1]) - int(video.shape[0]) * 1000 / 24) <= 6,
          (int(audio["waveform"].shape[-1]), int(video.shape[0])))
    hand = [c["handoff"] for c in calls["cond"]]
    check("each continued shot opens on the last frame of the one before", hand[0] is None and hand[4] is None
          and abs(float(hand[1].mean()) - (0.1 + (fc[0] - 1) / 1e4)) < 1e-4
          and abs(float(hand[5].mean()) - (0.5 + (fc[4] - 1) / 1e4)) < 1e-4, [None if x is None else float(x.mean()) for x in hand])
    check("every shot without dialogue renders its picture silent; the spoken one holds its lead",
          [c["silent"] for c in calls["cond"]] == [True, True, True, False, True, True] and calls["cond"][3]["lead"] == 0.5,
          [c["silent"] for c in calls["cond"]])
    check("no pose without the controlnet", calls["pose"] == [] and FakeDet.made == 0)
    check("info has a line per shot and the totals", sum(1 for x in info.split(" | ") if x.startswith("shot ")) == 6
          and "rendered in" in info, info)
    check("script holds what each shot was told", out[3].count("[shot ") == 6 and "Mara: handcuffs behind her back." in out[3])

    check("without pose control or retakes, info says held limbs are kept by the prompt alone and what would change that",
          "held arms or ankles in shot(s) 2, 3, 4, 5, 6 are kept by the prompt alone" in info
          and "pose control needs the H3 Fun controlnet" in info, info)
    out, calls = render(SCRIPT, pose_cn="cn")
    check("with pose control running there is no such warning", "kept by the prompt alone" not in out[2], out[2])
    check("the pose check runs on every shot with a held limb", [c["shot"] for c in calls["pose"]] == [2, 3, 4, 5, 6])
    check("the cuffing shot latches; later shots repair",
          calls["pose"][0]["latch"] == int(S.math.ceil(S.POSE_LATCH_FROM * fc[1])) and calls["pose"][1]["latch"] is None,
          [c["latch"] for c in calls["pose"]])
    check("who is who carries across continued shots and stops at a cut",
          [c["carry"] for c in calls["pose"]] == [None, ({"Mara": 2}, None), ({"Mara": 3}, None), None, ({"Mara": 5}, None)],
          [c["carry"] for c in calls["pose"]])
    passes = [(c["shot"], c["model"]) for c in calls["sample"] if not c["foley"]]
    check("a repaired shot samples a second time on the patched model, same seed",
          passes.count((3, "patched")) == 1 and passes.count((3, "base")) == 1 and all(c["seed"] == 7 for c in calls["sample"]))
    check("the repaired frames are the ones kept", abs(float(out[0][0].mean()) - 0.1) < 1e-3
          and any(abs(float(f.mean()) - 0.35) < 2e-3 for f in out[0]))
    check("info says what pose control did", out[2].count("pose repaired") == 5
          and "checked on a half-size draft" in out[2], out[2])
    full = calls["cond"][1]["size"]
    check("each held shot is drafted at half size from the same text, and only the draft feeds the pose check",
          [d["size"] for d in calls["draft"]] == [S.draft_size(*full)] * 5 and all(d["text"] == "cond" for d in calls["draft"])
          and [c["shot"] for c in calls["sample"] if c["draft"]] == [2, 3, 4, 5, 6]
          and all(c["cond"] == "small" and c["model"] == "base" for c in calls["sample"] if c["draft"])
          and all(c["size"] == (full[1], full[0]) for c in calls["pose"]), (calls["draft"], [c["size"] for c in calls["pose"]]))
    check("the draft costs a quarter of the picture: 1344x768 drafts at 672x384",
          S.draft_size(1344, 768) == (672, 384) and S.draft_size(768, 1344) == (384, 672))

    out, calls = render(SCRIPT, pose_cn="cn", oom_pass2=True)
    check("a controlled pass that runs out of memory renders the shot plainly at full size and turns pose control off",
          [c["shot"] for c in calls["pose"]] == [2] and "controlled pass failed" in out[2]
          and [(c["shot"], c["model"]) for c in calls["sample"] if not c["draft"] and not c["foley"] and c["shot"] == 2]
          == [(2, "patched"), (2, "base")] and int(out[0].shape[0]) == sum(fc) - keyed_after_first,
          ([c["shot"] for c in calls["pose"]], out[2]))

    out, calls = render(SCRIPT, pose_cn="cn", pose_result=lambda n: (None, {"skipped": "not guessing"}))
    finals = [c for c in calls["sample"] if not c["draft"] and not c["foley"]]
    check("a skipped check renders the shot plainly at full size, never keeping the draft",
          all(c["model"] == "base" for c in calls["sample"]) and [c["shot"] for c in finals] == [1, 2, 3, 4, 5, 6]
          and all(c["cond"] == "cond" for c in finals) and "pose not guessing" in out[2], out[2])

    out, calls = render(SCRIPT, plan_only=True)
    check("plan_only samples nothing and still returns the script", calls["sample"] == [] and out[3].count("[shot ") == 6
          and "pose latch: Mara behind the back" in out[2], out[2])


def test_speech():
    print("\n=== every word of a line is spoken ===")
    beat = "Dan grabs her arm. 'You aren't going anywhere tonight,' he says. Mara's eyes widen."
    shot = S.plan_shots("A cell.\n\n" + beat, 5.0, [None] * 4, False)[0]
    check("a line in single quotes is speech, and the apostrophes in aren't and Mara's are not quotes",
          not shot["wordless"] and "<d>You aren't going anywhere tonight,</d> he says. Mara's eyes widen." in shot["prompt"]
          and "Nobody speaks" not in shot["prompt"], shot["prompt"])
    for line, want in (('Dan says "Get in."', 'Dan says <d>Get in.</d>'), ("Dan says “Get in.”", "Dan says <d>Get in.</d>"),
                       ("Dan says ‘Don’t move.’", "Dan says <d>Don’t move.</d>"), ("Dan says <d>Get in.</d>", "Dan says <d>Get in.</d>")):
        check(f"{line} reaches H3 inside its own speech tags", S.dialogue(line) == want, S.dialogue(line))
    quiet = S.plan_shots("A cell.\n\nDan's hands shake. She doesn't look up.", 5.0, [None] * 4, False)[0]
    check("apostrophes alone are not a line", quiet["wordless"] and "<d>" not in quiet["prompt"], quiet["prompt"])
    long = "Dan says \"" + " ".join(["word"] * 60) + ".\""
    shot = S.plan_shots("A cell.\n\n" + long, 5.0, [None] * 4, False)[0]
    check("a line that needs more than one shot can hold gets the longest shot and says to split it",
          shot["frames"] == S.rt.MAX_FRAMES and "split it across beats" in S.shot_line(shot), S.shot_line(shot))
    ten = S.plan_shots("A cell.\n\nDan says \"one two three four five six seven eight nine ten.\"", 4.0, [None] * 4, False)[0]
    check("a ten-word line gets about 5.4 seconds, H3's own pace with room before and after, even when shot_seconds is 4",
          ten["frames"] == S.rt.align_frame_count(round((10 / 3.5 + 2.5) * 24))
          and "lengthened to fit its line" in S.shot_line(ten),
          (ten["frames"], S.shot_line(ten)))
    told = S.plan_shots("A cell.\n\nDan says \"one two three four five six seven eight nine ten.\"\nseconds: 3", 4.0,
                        [None] * 4, False)[0]
    check("seconds: is still obeyed, and info says the line needs more", told["frames"] == S.rt.align_frame_count(72)
          and "needs more than its seconds" in S.shot_line(told), S.shot_line(told))


def test_shot_length():
    print("\n=== shot length ===")
    script = ("A cell.\n\nMara sits on the cot.\n\nDan opens the door, walks in and sits down.\n\n"
              "Dan says \"Not a sound, or you will regret it tonight.\"\n\n"
              "Dan handcuffs her wrists behind her back.\nhold: Mara, handcuffs behind her back\n\n"
              "Mara waits.\nseconds: 12")
    twelve = S.rt.align_frame_count(288)
    beat = [s["frames"] for s in S.plan_shots(script, 15.0, [None] * 4, False, from_beat=True)]
    check("from the beat: one action, three actions, a spoken line, a restraint going on, an explicit length",
          beat == [73, 175, 124, 124, twelve], beat)
    capped = [s["frames"] for s in S.plan_shots(script, 5.0, [None] * 4, False, from_beat=True)]
    check("shot_seconds caps every estimate except a spoken line's, and seconds: still wins",
          capped == [73, 124, 124, 124, twelve], capped)
    fixed = [s["frames"] for s in S.plan_shots(script, 4.0, [None] * 4, False)]
    check("fixed gives every shot shot_seconds, unless its line needs longer", fixed == [107, 107, 124, 107, twelve], fixed)
    check("a beat with nothing to stage gets one action's worth", S.shot_frames("Silence.", 362, False) == 73)
    out, _ = render(script, plan_only=True, seconds=15.0)
    fixed_out, _ = render(script, plan_only=True, seconds=15.0, shot_length="fixed")
    saved, S._LOADED = S._LOADED, {k: 0.0 for k in S._LOADED}
    stale, _ = render(script, plan_only=True, seconds=15.0)
    S._LOADED = saved
    check("info says to restart ComfyUI when the node's code changed after it was loaded, and only then",
          "RESTART COMFYUI: " in stale[2] and "sampler.py" in stale[2] and "RESTART" not in out[2], stale[2][:200])
    check("the widget reaches the plan, from the beat by default",
          "shot 2: 175 frames" in out[2] and "shot 2: 362 frames" in fixed_out[2], (out[2], fixed_out[2]))


def test_presence():
    print("\n=== who is in each shot ===")
    script = ("A dim cell. Mara <Picture 1> sits on a cot. Dan <Picture 2> stands by the door.\n\n"
              "Dan walks out and slams the door.\n\nMara cries.\n\nDan comes back in and sits beside Mara.\n\nHe leaves.\nexit: Dan\n\n"
              "Mara waits.\n\ncut\nA garden. Mara walks along the path.")
    shots = S.plan_shots(script, 10.0, [IMG1, IMG2, None, None], False, memory="Mara: a tall woman.\nDan: a guard.\nSetting: a prison.")
    check("a named exit, a return when named, an exit: line, and a cut",
          [s["cast"] for s in shots] == [["Dan"], ["Mara"], ["Mara", "Dan"], ["Dan", "Mara"], ["Mara"], ["Mara"]],
          [s["cast"] for s in shots])
    p = shots[1]["prompt"]
    check("an absent person's picture, character line and scene sentence stay out",
          shots[1]["refs"] == [IMG1] and "Dan" not in p and "Mara: a tall woman." in p and "Setting: a prison." in p, p)
    check("the head count follows the cast", "There is one person in the shot: one body, one face." in p
          and "There are two people in the shot" in shots[2]["prompt"])
    check("after a cut a person seen alone gets that frame as their current look, in place of the portrait",
          shots[5]["recover"] == ["Mara"] and shots[5]["refs"] == [] and "<Picture 1> shows Mara as they look now." in shots[5]["prompt"]
          and "<Picture" not in shots[5]["prompt"].split("\n\n")[0], shots[5]["prompt"])
    check("names read from leaving lines",
          S.leavers("Dan walks out.", ["Mara", "Dan"]) == ["Dan"]
          and S.leavers("Mara watches Dan leave.", ["Mara", "Dan"]) == ["Dan"]
          and S.leavers("Dan kisses Mara and leaves.", ["Mara", "Dan"]) == ["Dan"]
          and S.leavers("Dan and Ana leave.", ["Mara", "Dan", "Ana"]) == ["Dan", "Ana"]
          and S.leavers("Mara screams and Dan storms off.", ["Mara", "Dan"]) == ["Dan"])
    check("...and lines that only sound like leaving",
          S.leavers("Dan leaves Mara on the bed.", ["Mara", "Dan"]) == []
          and S.leavers("Dan walks away from the bed.", ["Mara", "Dan"]) == []
          and S.leavers("Mara slips out of her cuffs.", ["Mara", "Dan"]) == []
          and S.leavers("Dan left the keys on the table.", ["Mara", "Dan"]) == [])
    taping = ("A dim room. Mara lies on the bed. Dan stands beside her.\n\n"
              "Dan wraps duct tape around her mouth.\nhold: Mara, duct tape over her mouth")
    loose = S.plan_shots(taping, 5.0, [None] * 4, False)[0]["prompt"]
    known = S.plan_shots(taping, 5.0, [None] * 4, False, memory="Mara: a woman.\nDan: a man.")[0]["prompt"]
    check("a man the beats show acting is counted in the head count though no memory declares him, and still looks nothing "
          "like her",
          "There are two people in the shot" in loose and "Everyone in the shot other than Mara looks nothing like her" in loose
          and "There are two people in the shot" in known, (loose, known))
    check("a person described as 'a tall woman' is not an extra; a man walking in is",
          not S.extras_in("Mara is a tall woman.") and not S.extras_in("Dan: a man.") and not S.extras_in("Mara, a nurse, waits.")
          and S.extras_in("A man walks in.") and S.extras_in("The guard watches.") and S.extras_in("Her boyfriend calls."))
    crowd = S.plan_shots("A street.\n\nMara walks past a crowd.", 5.0, [None] * 4, False, memory="Mara: a woman.")
    check("no head count when the beat brings in other people", "There is one person" not in crowd[0]["prompt"])
    odd = S.plan_shots("She is <Picture 1>. Style: noir.\n\nShe waits.", 5.0, [IMG1, None, None, None], False)
    check("pronouns and labels are not taken for characters", odd[0]["cast"] == [] and odd[0]["refs"] == [IMG1], odd[0])


def test_beats_decide_presence():
    print("\n=== only the beat puts people in a shot ===")
    memory = "Dan: a tall man.\nCrystal: a slim woman."
    script = ("A bathroom in a small apartment. Dan and Crystal live here.\n\n"
              "The camera views the interior of the bathroom. It pans over to the door, where it opens to the inside.\n\n"
              "Dan walks in.\n\nCrystal follows him.")
    shots = S.plan_shots(script, 10.0, [None] * 4, False, memory=memory)
    check("an empty establishing beat puts nobody in the shot and says so",
          shots[0]["cast"] == [] and shots[0]["prompt"] == "A bathroom in a small apartment.\n\nThe camera views the interior "
          "of the bathroom. It pans over to the door, where it opens to the inside.\n\nNobody is in the shot. There is no "
          "background music.", shots[0]["prompt"])
    check("people join from the beat that brings them in, not from the scene paragraph",
          [x["cast"] for x in shots] == [[], ["Dan"], ["Dan", "Crystal"]] and "There is one person" in shots[1]["prompt"],
          [x["cast"] for x in shots])
    pair = S.plan_shots("Ana and Lena sit on the couch talking.\n\nAna stands up and walks to the window.\n\nShe looks out at the "
                        "rain.\n\nLena joins her at the window and they look out together.\n\nThe lights flicker.\n\nAna reads a "
                        "message from Lena's boss.\n\nAna and Lena face each other and hug.\n\nAna sits alone, her wrists crossed "
                        "over each other in her lap.\n\nLena walks into the kitchen.", 5.0, [IMG1, None, None, None], False,
                        memory="Ana <Picture 1>: a woman with long red hair.\nLena: a woman with short blonde hair.",
                        anchor="A small living room in the evening.")
    check("someone named once is not carried into beats that do not name them, refer to them or bring everyone in",
          [sorted(x["cast"]) for x in pair] == [["Ana", "Lena"], ["Ana"], ["Ana"], ["Ana", "Lena"], ["Ana", "Lena"],
                                                ["Ana", "Lena"], ["Ana", "Lena"], ["Ana"], ["Lena"]]
          and "short blonde hair" not in pair[1]["prompt"]
          and "this shot frames only Ana (exactly as in <Picture 1>), and nobody else is in view." in pair[1]["prompt"],
          [x["cast"] for x in pair])
    table = S.plan_shots("A bathroom.\n\nSarah and Dana walk in.\n\nSarah lies down on the table.\n\nDana opens the bag."
                         "\n\nDana takes off Sarah's diaper.\n\nDana walks out.\n\nSarah sits up.\n\nDana comes back in.",
                         5.0, [IMG1, None, None, None], False, memory="Sarah <Picture 1>: a woman.\nDana: a woman.")
    check("someone still in the room stays in view through a beat about someone else when a later beat needs them where "
          "they are, so they never walk back in; someone who walks back in, or has left, comes into view as before",
          [sorted(x["cast"]) for x in table] == [["Dana", "Sarah"]] * 5 + [["Sarah"], ["Dana"]]
          and "comes into view" not in table[3]["prompt"] and "this shot frames only Dana" in table[6]["prompt"],
          [S.shot_line(x) for x in table])
    unnamed = S.plan_shots("A bathroom.\n\nA man opens the door.", 5.0, [None] * 4, False, memory=memory)
    check("a beat with someone in it, named or not, is not called empty", "Nobody is in the shot" not in unnamed[0]["prompt"])
    loose = S.plan_shots("A bathroom. Crystal is inside.\n\nShe looks at the mirror.", 5.0, [None] * 4, False)[0]["prompt"]
    known = S.plan_shots("A bathroom. Crystal is inside.\n\nShe looks at the mirror.", 5.0, [None] * 4, False,
                         memory="Crystal: a slim woman.")[0]
    check("a beat that says she is not empty, and she is the declared woman when there is one",
          "Nobody is in the shot" not in loose and "Crystal is inside." in loose and known["cast"] == ["Crystal"]
          and "Crystal: a slim woman." in known["prompt"], (loose, known["cast"]))


def test_continuity_lines():
    print("\n=== continuity between beats ===")
    shots = S.plan_shots(SCRIPT, 10.0, [IMG1, IMG2, None, None], False)
    claim = "is the frame this shot opens on: the same place and the same people, one moment earlier"
    check("every continued shot claims its opening frame, numbered after its pictures",
          f"<Picture 3> {claim}" in shots[1]["prompt"] and f"<Picture 2> {claim}" in shots[2]["prompt"]
          and claim not in shots[0]["prompt"] and claim not in shots[4]["prompt"], shots[2]["prompt"])
    check("a shot where something goes on ends with it in plain view",
          "By the last frame, Mara has handcuffs behind her back in plain view, and whoever put it on has let go."
          in shots[1]["prompt"], shots[1]["prompt"])
    check("what is held stays on for the whole shot, wrists in place, the other person free of it, when the next thing goes on",
          "Mara: handcuffs behind her back. All of it stays on for the whole shot. Dan wears none of it. Mara's wrists stay locked together behind the back "
          "the whole time. By the last frame, Mara has duct tape over her mouth in plain view" in shots[2]["prompt"], shots[2]["prompt"])
    legs = S.plan_shots("A room.\nhold: Mara, wrists zip tied in front; rope around her ankles; cuffs to the bed rail\n\nMara waits.",
                        5.0, [None] * 4, False)
    check("held wrists in front and ankles get their own line; a restraint fastened to an object adds none",
          "Mara's wrists stay bound together in front the whole time." in legs[0]["prompt"]
          and "Mara's ankles stay bound together the whole time." in legs[0]["prompt"]
          and legs[0]["prompt"].count("the whole time.") == 2, legs[0]["prompt"])
    one = S.plan_shots("Mara waves.", 5.0, [None] * 4, True)
    check("a first frame is claimed too", f"<Picture 1> {claim}" in one[0]["prompt"], one[0]["prompt"])


def test_strangers():
    print("\n=== extras are welcome and look like nobody in character memory ===")
    memory = "Crystal is a slim woman with long dark hair. She wears a grey dress.\nMike, a tall man in a black shirt."
    script = ("A dim bar.\n\nMike waits at the bar.\n\nA woman walks past Mike.\n\nCrystal and Mike sit down as a crowd watches.\n\n"
              "Jake walks over to Crystal.\n\nCrystal kneels.")
    shots = S.plan_shots(script, 5.0, [None] * 4, False, memory=memory)
    check("memory written as sentences declares its people, and leaves their lines out of shots they are not in",
          [x["cast"] for x in shots[:3]] == [["Mike"], ["Mike"], ["Mike", "Crystal"]]
          and "Crystal is a slim woman" not in shots[1]["prompt"] and "grey dress" not in shots[1]["prompt"]
          and "She wears a grey dress." in shots[2]["prompt"], [x["prompt"][:200] for x in shots])
    unlike = ("Everyone in the shot other than Mike and Crystal looks nothing like them: a different face, hair, body and "
              "clothes, and nothing that Mike and Crystal wear.")
    check("a shot with a woman, a crowd or an undeclared name says they look nothing like the declared characters",
          "Everyone in the shot other than Mike looks nothing like him: a different face, hair, body and clothes, and "
          "nothing that Mike wears." in shots[1]["prompt"] and unlike in shots[2]["prompt"]
          and "Everyone in the shot other than Crystal looks nothing like her: a different face, hair, body and clothes, and "
          "nothing that Crystal wears." in shots[3]["prompt"]
          and "looks nothing like" not in shots[0]["prompt"] + shots[4]["prompt"], [x["prompt"][-600:] for x in shots])
    check("a continued shot lets the extras in, and only a shot without any keeps everyone else out",
          "carried forward; the strangers this beat brings in come into view during this shot." in shots[1]["prompt"]
          and "carried forward; Crystal and the strangers this beat brings in come into view during this shot."
          in shots[2]["prompt"] and "but this shot frames only Crystal, and nobody else is in view." in shots[4]["prompt"]
          and "There is one person in the shot" in shots[4]["prompt"], [x["prompt"][-400:] for x in shots])
    looks = S.plan_shots("A room.\n\nCrystal waits.", 5.0, [None] * 4, False,
                         memory="Crystal: a slim Asian woman. Long dark hair. Blue eyes.")
    check("capitalised words that describe someone do not count as another person",
          "There is one person in the shot" in looks[0]["prompt"], looks[0]["prompt"])


def test_unnamed_pronouns():
    print("\n=== she or he next to an unnamed person never pulls in a declared character ===")
    memory = "Crystal <Picture 1>: a slim woman with long dark hair.\nMike <Picture 2>: a tall man in a black shirt."
    cast = lambda beat: [x["cast"] for x in S.plan_shots(f"A bar.\n\n{beat}", 5.0, [IMG1, IMG2, None, None], False,
                                                           memory=memory)]
    check("a waitress, a woman, a receptionist, a girl or a man is who the pronoun means, not the declared character",
          all(cast(f"Mike sits at the bar.\n\n{b}") == [["Mike"], ["Mike"]] for b in (
              "A waitress brings him a drink. She smiles.", "Mike talks to a woman at the bar. She laughs.",
              "The receptionist hands him a key and she points to the elevator.", "A girl sits beside him. He buys her a drink."))
          and cast("Crystal sits alone at home.\n\nA man knocks on her door. He waits.") == [["Crystal"], ["Crystal"]]
          and cast("Mike sits at the bar.\n\nShe walks in and sits beside him.") == [["Mike"], ["Mike", "Crystal"]])
    people, gender = ["Crystal", "Mike"], {"Crystal": "f", "Mike": "m"}
    holds, _, unclear, _ = S.rst.read("A woman walks in. Mike handcuffs her.", people, gender, {})
    off, _, why = S.wrd.read("A woman takes off her jacket.", people, gender, {"Crystal": "Crystal wears a jacket."})
    check("cuffs or a garment on an unnamed woman are not put on the declared one; the sentence is listed instead",
          holds == {} and any("who wears the handcuffs" in u for u in unclear) and off == {} and why, (holds, unclear, off, why))
    check("a description of a named character is not an unnamed person",
          S.rst.stand_ins("Crystal, a slim woman, sits down. Mike is a tall man.") == set()
          and S.rst.read("Crystal, a slim woman, sits down. Mike cuffs her wrists.", people, gender, {})[0]
          == {"Crystal": ["handcuffs on her wrists"]})


def test_mentioned_not_present():
    print("\n=== a name only mentioned, or someone who has gone, is not in the shot ===")
    memory = "Crystal <Picture 1>: a slim woman with long dark hair.\nMike <Picture 2>: a tall man in a black shirt."
    cast = lambda script, scene="A house.": [x["cast"] for x in S.plan_shots(f"{scene}\n\n{script}", 5.0, [IMG1, IMG2, None, None],
                                                                             False, memory=memory)]
    check("a letter from her, her photo, thinking of her, calling her or waiting for her does not put her in the shot",
          all(cast(f"Mike sits at home.\n\n{b}") == [["Mike"], ["Mike"]] for b in (
              "Mike reads a letter from Crystal.", "Mike looks at Crystal's photo on the wall.", "Mike thinks about Crystal.",
              "Mike calls Crystal on the phone.", "Mike waits for Crystal.")))
    check("but walking over to her, taking something from her, calling her over, her hand, or she arriving does",
          all(cast(f"Mike sits at home.\n\n{b}")[1] == ["Mike", "Crystal"] for b in (
              "Mike walks over to Crystal.", "Mike takes the keys from Crystal.", "Mike calls Crystal over.",
              "Crystal's hand touches his arm.", "Mike waits for Crystal. She walks in.")))
    gone = ("Crystal heads upstairs.", "Crystal goes to bed.", "Crystal goes into the bathroom.", "Crystal storms off to her room.",
            "Crystal disappears into the hallway.")
    check("heading upstairs, going to bed or into another room takes her out of the beats after",
          all(cast(f"Crystal and Mike talk in the kitchen.\n\n{b}\n\nThey sit down together.")[-1] == ["Mike"] for b in gone)
          and cast("Crystal and Mike talk.\n\nCrystal goes into the kitchen.\n\nThey sit down together.", "A small kitchen.")[-1]
          == ["Crystal", "Mike"])


def test_unstated_gender():
    print("\n=== a character with no stated gender is never swapped for one with a stated gender ===")
    memory = "Sarah <Picture 1>: a woman with long red hair and a black crop top.\nDana: short blonde hair, grey hoodie and jeans."
    first = lambda beat: S.plan_shots(f"{beat}\n\nDana sits.", 5.0, [IMG1, None, None, None], False, memory=memory,
                                      anchor="A small apartment.")[0]
    check("her, she, a mention, or someone away or only heard does not bring the other character in",
          all(first(b)["cast"] == ["Dana"] for b in (
              "Dana walks into her apartment and drops her bag.", "Dana sits on the couch. She checks her phone.",
              "Dana is waiting for Sarah to come home.", "Dana cleans up before Sarah arrives.", "Dana gets a text from Sarah.",
              "Dana is alone; Sarah is at work.", "Dana hears Sarah upstairs.", "Dana is thinking about Sarah.")))
    check("a name or pronoun inside spoken words, a voice-over, or screaming for her does not bring her in",
          all(first(b)["cast"] == ["Dana"] for b in (
              'Dana storms in. "Sarah, where are you?" she shouts.', 'Dana says, "I told Sarah to wait for me."',
              '“Sarah! Get down here!” Dana yells up the stairs.', 'Dana: "Sarah will be home soon."',
              'Sarah (V.O.): "I\'m upstairs."\nDana looks up.', 'Dana screams for Sarah.', 'Dana says "we belong together."'))
          and sorted(first('"Dana, stop!" Sarah begs as Dana ties her ankles.')["cast"]) == ["Dana", "Sarah"])
    check("her hoodie goes to the one who wears it, and her own wrists to the one cuffing them",
          first("Dana pulls off her hoodie.")["clothes_off"] == "Dana: hoodie."
          and first("Dana cuffs her own wrists.")["added"] == "Dana: handcuffs on her wrists.")
    pair = S.plan_shots("A small apartment.\n\nSarah and Dana talk.\n\nThe lights flicker.\n\nDana kneels before Sarah.", 5.0,
                        [IMG1, None, None, None], False, memory=memory)
    check("kneeling before her still puts her there, and info says why someone the beat does not name is in it",
          sorted(pair[2]["cast"]) == ["Dana", "Sarah"] and "Dana (from the shot before)" in S.shot_line(pair[1])
          and "(from" not in S.shot_line(pair[2]), [S.shot_line(x) for x in pair])
    opening = lambda first: S.plan_shots(f"{first}\n\nDana sits.\n\nSarah stands.", 5.0, [IMG1, None, None, None], False,
                                         memory=memory)
    check("a first paragraph where someone acts, speaks or the camera moves is the first shot, never repeated in the others",
          all(len(opening(f)) == 3 and f not in opening(f)[1]["prompt"] for f in (
              "Sarah and Dana sit on the couch talking.", "In a dim basement, Dana cuffs Sarah's wrists.", '"Wake up," Dana says.',
              "The camera pans across the bedroom.", "Sarah sits on the bed.", "Sarah is lying on the bed."))
          and all(len(opening(f)) == 2 for f in ("A dim basement.", "A bar. Sarah <Picture 1> sits at the counter.",
                                                 "A small apartment. Sarah wears a red dress and has long hair.")))
    check("a scene sentence with someone who is not in the shot is left out of it, a mention of their things is not",
          "live here" not in opening("A small apartment. Sarah and Dana live here.")[1]["prompt"]
          and S.without_absent("Sarah is tied to Dana's bed.", ["Dana"], ["Sarah", "Dana"]) == "Sarah is tied to Dana's bed.")
    check("a she or he opening a sentence in a character's own lines gives their gender, but quoted speech does not",
          S.rst.genders(["Dana"], [S.wrd.attributed("Dana:\nShort blonde hair.\nShe wears a grey hoodie.", ["Dana"])])
          == {"Dana": "f"} and S.rst.genders(["Dan"], ['Dan: "Where is she?"']) == {})


def test_memory_layouts():
    print("\n=== character memory is read the same however it is laid out ===")
    layouts = {
        "sheet": "Crystal: a slim woman with long dark hair and a red dress.\nMike: a tall man in a black shirt.",
        "header and lines": "Crystal:\nA slim woman with long dark hair. She wears a red dress.\n\nMike:\nA tall man in a black shirt.",
        "header, no blank line": "Crystal:\nA slim woman with long dark hair.\nShe wears a red dress.\nMike:\nA tall man in a black shirt.",
        "tag header": "Crystal <Picture 1>\nA slim woman with long dark hair, red dress.\n\nMike <Picture 2>\nA tall man in a black shirt.",
        "bullets": "- Crystal: a slim woman with long dark hair and a red dress.\n- Mike: a tall man in a black shirt.",
        "bold": "**Crystal**: a slim woman with long dark hair and a red dress.\n**Mike**: a tall man in a black shirt.",
        "age": "Crystal (25): a slim woman with long dark hair and a red dress.\nMike (30): a tall man in a black shirt.",
        "dash": "Crystal - a slim woman with long dark hair and a red dress.\nMike - a tall man in a black shirt.",
        "follow-on lines": "Crystal: a slim woman.\nLong dark hair.\nWears a red dress.\nMike: a tall man.\nWears a black shirt.",
        "all caps": "CRYSTAL: a slim woman with long dark hair and a red dress.\nMIKE: a tall man in a black shirt.",
        "prose": "Crystal is a slim woman with long dark hair. Her dress is red.\n\nMike is a tall man. His shirt is black.",
        "form": "Name: Crystal\nLooks: slim woman, long dark hair, red dress\n\nName: Mike\nLooks: tall man, black shirt",
        "header and bullets": "Crystal\n- slim woman\n- long dark hair\n- red dress\n\nMike\n- tall man\n- black shirt",
    }
    bad = []
    for k, mem in layouts.items():
        shots = S.plan_shots("A bar.\n\nMike sits at the bar.\n\nCrystal walks in and sits beside Mike.", 5.0, [None] * 4, False, memory=mem)
        one, two = shots[0]["prompt"], shots[1]["prompt"]
        ok = ([x["cast"] for x in shots] == [["Mike"], ["Mike", "Crystal"]]
              and not any(w in one for w in ("slim woman", "long dark hair", "red dress")) and "tall man" in one
              and "slim woman" in two and "tall man" in two
              and S.rst.genders(["Crystal", "Mike"], [S.wrd.attributed(mem, ["Crystal", "Mike"])]) == {"Crystal": "f", "Mike": "m"})
        bad += [] if ok else [k]
    check("every layout keeps an absent character's whole description out, its own in, and reads both genders", not bad, bad)
    tagged = S.plan_shots("A bar.\n\nMike sits at the bar.\n\nCrystal walks in and sits beside Mike.", 5.0, [IMG1, IMG2, None, None], False,
                          memory="CRYSTAL <Picture 1>: a slim woman.\nMIKE <Picture 2>: a tall man.\nSetting: a dim bar.")
    check("a picture belongs to its block, all-caps names match the beats, and a Setting line is never a character's",
          [S.shot_line(x).split("pictures: ")[-1] for x in tagged]
          == ["Mike (ref_image_2, face and hair)", "Crystal (ref_image_1, face and hair), Mike (ref_image_2, face and hair)"]
          and all("Setting: a dim bar." in x["prompt"] for x in tagged), [S.shot_line(x) for x in tagged])


def test_things():
    print("\n=== ships, vehicles and animals are not people ===")
    memory = ("Valiant <Picture 1>: a long white starship with twin blue engines.\n"
              "Raptor <Picture 2>: a black alien warship shaped like a crescent.")
    ships = S.plan_shots("Deep space, a nebula behind a ringed planet.\n\nValiant drifts past the planet.\n\nRaptor drops out of "
                         "hyperspace behind Valiant.", 5.0, [IMG1, IMG2, None, None], False, memory=memory)
    person = ("person in the shot", "people in the shot", "mouth", "same people", "speaks")
    check("a shot of things alone says there are no people in it and asks for no bodies, faces, mouths or speech",
          "There are no people anywhere in the shot, only Valiant." in ships[0]["prompt"]
          and "There are no people anywhere in the shot, only Valiant and Raptor." in ships[1]["prompt"]
          and not any(w in x["prompt"] for x in ships for w in person)
          and "the same place and the same things in it, one moment earlier, carried forward; Raptor (exactly as in <Picture 2>) "
              "comes into view during this shot, and nothing else joins." in ships[1]["prompt"], [x["prompt"][-500:] for x in ships])
    mixed = S.plan_shots("Deep space.\n\nValiant drifts past the planet.\n\nReyes stands on the bridge of Valiant.\n\n"
                         "Valiant's crew runs to their stations.", 5.0, [IMG1, None, None, None], False,
                         memory="Reyes: a stern woman in a grey uniform.\nValiant <Picture 1>: a long white starship.")
    check("a person aboard counts as the one person, and a crew is let in as extras",
          "There is one person in the shot: one body, one face." in mixed[1]["prompt"]
          and "the same people" in mixed[1]["prompt"] and "There is one person" not in mixed[2]["prompt"]
          and "the strangers this beat brings in come into view" in mixed[2]["prompt"]
          and "There are no people" not in mixed[1]["prompt"] + mixed[2]["prompt"], [x["prompt"][-400:] for x in mixed])
    check("a description decides by what it names first",
          S.things(["Valiant", "Kirk", "Rex", "Mara"], {"Valiant": "Valiant: a starship crewed by 400 officers.",
                                                        "Kirk": "Kirk: the captain of a starship.", "Rex": "Rex: a big dog.",
                                                        "Mara": "Mara: a woman with a car."}) == {"Valiant", "Rex"})


def test_named_ship_only():
    print("\n=== the ship a beat names is the one it gets ===")
    beats = "\n\nThe Excelsior glides out of spacedock as she powers up.\n\nThe Enterprise follows the Excelsior out."
    scene = "Spacedock above Earth, where the Enterprise sits in dry dock."
    for mem in ("USS Excelsior <Picture 1>: an Excelsior-class Federation starship.\nUSS Enterprise <Picture 2>: a "
                "Constitution-class Federation starship.",
                "USS Excelsior (NCC-2000) <Picture 1>: an Excelsior-class Federation starship.\nUSS Enterprise (NCC-1701-A) "
                "<Picture 2>: a Constitution-class Federation starship.",
                "USS Excelsior NCC-2000 <Picture 1>: an Excelsior-class Federation starship.\nUSS Enterprise NCC-1701-A "
                "<Picture 2>: a Constitution-class Federation starship.",
                "USS Excelsior NCC-2000: an Excelsior-class Federation starship.\nUSS Enterprise NCC-1701-A: a Constitution-class "
                "Federation starship."):
        shots = S.plan_shots(scene + beats, 5.0, [IMG1, IMG2, None, None], False, memory=mem)
        tagged = "<Picture" in mem
        check(f"beat 1 gets only Excelsior, keeps its place, and beat 2 brings Enterprise in ({mem[14:30]}...)",
              [x["cast"] for x in shots] == [["Excelsior"], ["Excelsior", "Enterprise"]]
              and [len(x["refs"]) for x in shots] == ([1, 2] if tagged else [0, 0])
              and "Enterprise" not in shots[0]["prompt"] and shots[0]["prompt"].startswith("Spacedock above Earth.\n")
              and scene in shots[1]["prompt"], [(x["cast"], len(x["refs"]), x["prompt"][:160]) for x in shots])
    layouts = ("USS {n} - <Picture {k}>: {c} Federation starship.", "USS {n} – <Picture {k}> – {c} Federation starship.",
               "<Picture {k}>: USS {n}, {c} Federation starship.", "<Picture {k}> USS {n}, {c} Federation starship.",
               "USS {n}, {c} Federation starship <Picture {k}>.", "USS {n}, {c} Federation starship (<Picture {k}>).",
               "USS {n} = <Picture {k}>, {c} Federation starship.",
               "The USS {n} is {c} Federation starship, shown in <Picture {k}>.",
               "USS {n} is {c} Federation starship. <Picture {k}>")
    fleet = (("Excelsior", 1, "an Excelsior-class"), ("Enterprise", 2, "a Constitution-class"), ("Reliant", 3, "a Miranda-class"))
    got = []
    for f in layouts:
        shots = S.plan_shots("Spacedock above Earth.\n\nThe Excelsior glides out of spacedock.\n\nThe Enterprise follows the Excelsior.", 5.0,
                             [IMG1, IMG2, IMG1, None], False,
                             memory="\n".join(f.format(n=n, k=k, c=c) for n, k, c in fleet))
        got.append((f, [S.shot_line(x) for x in shots]))
    check("a picture tag anywhere on a memory line belongs to the ship that line is about, and only that ship's picture goes in",
          all(lines == ["shot 1: 124 frames, cast: Excelsior, pictures: Excelsior (ref_image_1)",
                        "shot 2: 124 frames, cast: Excelsior, Enterprise, pictures: Excelsior (ref_image_1), Enterprise "
                        "(ref_image_2)"] for _, lines in got), [g for g in got if g[1][0].count("ref_image") != 1])
    check("a scene sentence about someone who is not there still goes whole",
          S.without_absent("A cell. Dan stands guard, with a rifle.", ["Dan"], ["Dan", "Mara"]) == "A cell.")


def test_hands_and_backs():
    print("\n=== who uses their hands, and what shows from behind ===")
    memory = "Crystal: a slim woman. She wears a steel chastity belt with a chain hanging from the back.\nMike: a tall man."
    script = ("A dim basement.\n\nCrystal stands with her wrists cuffed behind her back. Mike stands beside her.\n\n"
              "The camera moves behind Crystal, showing her back. Mike grabs the chain at the back of her chastity belt and "
              "pulls it.")
    shots = S.plan_shots(script, 5.0, [None] * 4, False, memory=memory)
    check("the free person does the hand work while the cuffed hands stay still",
          "It is Mike who grabs the chain at the back of her chastity belt and pulls it, with his own hands, while Crystal's "
          "hands stay still." in shots[1]["prompt"] and "It is" not in shots[0]["prompt"], shots[1]["prompt"])
    check("a shot from behind turns her whole body away, the back of what she wears showing, and she never twists",
          "Seen from behind, Crystal faces away from the camera from head to feet, so her back, the back of her head and the "
          "back of everything she wears show. Crystal's body never twists at the waist: her head, chest, hips and legs always "
          "face the same way and turn together." in shots[1]["prompt"] and "Seen from behind" not in shots[0]["prompt"]
          and shots[1]["prompt"].count("never twists") == 1, shots[1]["prompt"])
    cast, g, cuffed = ["Crystal", "Mike"], {"Crystal": "f", "Mike": "m"}, {"Crystal": ["handcuffs behind her back"]}
    check("a pronoun doer is named, the adverb kept, the next clause and the dialogue left out",
          S.hand_notes("He slowly pulls the chain, while Crystal gasps.", cast, g, cuffed)
          == ["It is Mike who slowly pulls the chain, with his own hands, while Crystal's hands stay still."]
          and S.hand_notes("Mike grabs the chain and says \"Move.\"", cast, g, cuffed)
          == ["It is Mike who grabs the chain, with his own hands, while Crystal's hands stay still."])
    check("no hand line when nobody's hands are held, when the held one acts, or when her hands are the point",
          S.hand_notes("Mike grabs the chain.", cast, g, {}) == []
          and S.hand_notes("She pulls at the chain. Mike watches.", cast, g, cuffed) == []
          and S.hand_notes("Crystal struggles. Mike grabs her cuffed wrists and pulls her up.", cast, g, cuffed)
          == ["It is Mike who grabs her cuffed wrists and pulls her up, with his own hands."])
    rear = lambda t: S.rear_notes(t, cast, g)
    check("a back view is read from the camera, a back to the camera, or facing away",
          all(rear(t) and "Crystal" in rear(t)[0] for t in ("The camera views Crystal from behind as Mike approaches.",
                                                             "Crystal stands with her back to the camera.",
                                                             "Crystal faces away from the camera.",
                                                             "A rear view of Crystal kneeling.")))
    check("someone standing behind her, cuffs behind her back or the back of a room is not a back view",
          all(rear(t) == [] for t in ("The camera watches as Mike steps behind her.", "Mike stands behind Crystal.",
                                      "Mike cuffs her wrists behind her back. The camera holds on her face.",
                                      "The camera shows her bottom lip trembling.", "The camera watches from the back of the room.",
                                      "Viewed from behind, Mike grabs the chain.")))


def test_quiet_mouths():
    print("\n=== no talking where nobody talks ===")
    shots = S.plan_shots(SCRIPT, 10.0, [None] * 4, False)
    line = "Nobody speaks, and every mouth stays closed."
    check("a shot without dialogue says nobody speaks and mouths stay closed",
          all(line in shots[k]["prompt"] for k in (0, 1, 2, 4, 5)) and line not in shots[3]["prompt"])
    ball = S.plan_shots("A room.\nhold: Mara, ball gag in her mouth\n\nMara waits.", 5.0, [None] * 4, False)
    check("a gag that holds the mouth open keeps it open", "Nobody speaks." in ball[0]["prompt"] and line not in ball[0]["prompt"],
          ball[0]["prompt"])
    loud = S.plan_shots("A room.\n\nMara screams.", 5.0, [None] * 4, False)
    pant = S.plan_shots("A room.\n\nMara pants.", 5.0, [None] * 4, False)
    check("a beat with its own vocal sound is not told to keep quiet",
          "Nobody speaks" not in loud[0]["prompt"] and "Nobody speaks" not in pant[0]["prompt"])
    video, audio = torch.ones(1, 24, 4, 2, 2), torch.ones(1, 32, 2, 8)
    seen = {}

    def fake_sample(model, cond, neg, latent, seed, steps, sn, sc, sg):
        seen.update(latent, cond=cond)
        v, a = latent["samples"].unbind()
        return {"samples": comfy.nested_tensor.NestedTensor((v * 0 + 5, a + 3)), "extra": 1}
    saved = S.sample
    S.sample = fake_sample
    cond = [["emb", {"minimax_keyframes": [1], "minimax_refs": [2], "minimax_token_tags": 3}]]
    try:
        res, scaled = S.foley_pass("m", cond, "n", {"shot": 1, "noise_mask": "old"},
                                   {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, 7, 8, "euler", "simple",
                                   None, 0.5)
        v, a = seen["samples"].unbind()
        mv, ma = seen["noise_mask"].unbind()
        rv, ra = res["samples"].unbind()
        check("the sound pass keeps the finished video and makes only the audio",
              bool((v == 1).all()) and float(a.abs().max()) == 0.0 and float(mv.max()) == 0.0 and float(ma.min()) == 1.0
              and seen["shot"] == 1 and bool((rv == 1).all()) and bool((ra == 3).all()) and res["extra"] == 1
              and not scaled and seen["cond"] is cond)
        big = torch.rand(1, 24, 7, 48, 84)
        res, scaled = S.foley_pass("m", cond, "n", {"shot": 2}, {"samples": comfy.nested_tensor.NestedTensor((big, audio))},
                                   7, 8, "euler", "simple", None, 0.5)
        v, _ = seen["samples"].unbind()
        mv, _ = seen["noise_mask"].unbind()
        check("on a full-size shot the sound pass listens to a half-size copy and hands back the full picture",
              scaled and tuple(v.shape) == (1, 24, 7, 24, 42) and tuple(mv.shape) == (1, 1, 7, 24, 42)
              and abs(float(v.mean()) - float(big.mean())) < 1e-3 and res["samples"].unbind()[0] is big
              and seen["cond"][0][1] == {"minimax_token_tags": 3})
        full, scaled = S.foley_pass("m", cond, "n", {"shot": 3}, {"samples": comfy.nested_tensor.NestedTensor((big, audio))},
                                    7, 8, "euler", "simple", None, 1.0)
        check("foley_resolution full keeps the old full-size pass", not scaled
              and tuple(seen["samples"].unbind()[0].shape) == (1, 24, 7, 48, 84) and seen["cond"] is cond)
        lead = {"samples": comfy.nested_tensor.NestedTensor((video, torch.zeros_like(audio)))}
        S.au.pin_audio_silence(lead, torch.full_like(audio, 2.0), 3)
        S.foley_pass("m", cond, "n", lead, {"samples": comfy.nested_tensor.NestedTensor((video, audio))},
                     7, 8, "euler", "simple", None, 0.5, pinned=True)
        _, a = seen["samples"].unbind()
        mv, ma = seen["noise_mask"].unbind()
        check("a spoken line's sound pass keeps the quiet half second before the line",
              float(mv.max()) == 0.0 and float(ma[..., :3].max()) == 0.0 and float(ma[..., 3:].min()) == 1.0
              and bool((a[..., :3] == 2.0).all()))
    finally:
        S.sample = saved


def test_directive_forms():
    print("\n=== hold lines written in other ways ===")
    tail = "\n\nDan wraps duct tape around her mouth."
    forms = {
        "no comma": "A room. Mara lies on the bed.\n\nDan handcuffs her wrists behind her back.\nhold: Mara handcuffs behind her back",
        "a dash": "A room. Mara lies on the bed.\n\nDan handcuffs her wrists behind her back.\nHold: Mara - handcuffs behind her back",
        "same line": "A room. Mara lies on the bed.\n\nDan handcuffs her wrists behind her back. hold: Mara, handcuffs behind her back",
        "own paragraph": "A room. Mara lies on the bed.\n\nDan handcuffs her wrists behind her back.\n\nhold: Mara, handcuffs behind her back",
    }
    for label, script in forms.items():
        shots = S.plan_shots(script + tail, 5.0, [None] * 4, False)
        check(f"a hold line with {label} is read and carried to the next shot",
              len(shots) == 2 and shots[1]["held"] == "Mara: handcuffs behind her back."
              and "hold" not in shots[0]["prompt"].lower().replace("whole", ""), [(s["held"], s["prompt"]) for s in shots])
    cut = S.plan_shots("A room.\n\nMara sits.\n\ncut\n\nA garden. Mara walks.", 5.0, [None] * 4, False)
    check("a cut written as its own paragraph cuts the next shot", [s["cut"] for s in cut] == [False, True], [s["cut"] for s in cut])
    early = S.plan_shots("hold: Mara, rope around her ankles\n\nA room. Mara sits.\n\nMara waits.", 5.0, [None] * 4, False)
    check("a hold before the scene is on from the first shot", len(early) == 1 and early[0]["held"] == "Mara: rope around her ankles.",
          [(s["held"], s["prompt"]) for s in early])
    out, _ = render("A room. Mara lies on the bed.\n\nDan cuffs her.\nhold: handcuffs behind her back\n\nMara waits.",
                    plan_only=True)
    check("a hold line with no name is reported, not silently dropped",
          "not read (write a hold:" in out[2] and "hold: handcuffs behind her back" in out[2], out[2])


def test_mumble():
    print("\n=== gagged speech ===")
    shots = S.plan_shots(SCRIPT, 10.0, [None] * 4, False)
    check("a gagged person in a shot with speech is muffled",
          "Every sound from Mara comes out muffled, the lips held shut under the tape." in shots[3]["prompt"], shots[3]["prompt"])
    check("no muffling in a quiet shot", "muffled" not in shots[4]["prompt"])
    ball = S.plan_shots("A room.\nhold: Mara, ball gag in her mouth\n\nMara whimpers.", 5.0, [None] * 4, False)
    check("a gag that holds the mouth open says so, and a vocal sound triggers it",
          "Every sound from Mara comes out muffled, the mouth held open around the gag." in ball[0]["prompt"], ball[0]["prompt"])


def test_sound_lines():
    print("\n=== foley and ambience ===")
    line, sounded = S.snd.sound_line("Dan walks to the door and opens the door.", "A rainy street.", False)
    check("actions bring their foley, the scene its ambience, and no music",
          sounded and line == "The only sounds are footsteps, a door on its hinges and rain against the glass. "
          "There is no background music.", line)
    line, sounded = S.snd.sound_line("Mara pours coffee into a mug.", "A small kitchen. A workshop. An office.", False)
    check("no steady hum or whir of a fridge, a strip light or a computer fan is added that the scene does not ask for",
          not re.search(r"hum|fan|whirr", line), line)
    line, sounded = S.snd.sound_line("Somewhere a dog barks loudly.", "", False)
    check("a beat that names its own sounds keeps them", sounded and line == "The only sounds are the ones this beat "
          "describes. There is no background music.", line)
    check("a spoken shot is told there is no background music",
          S.snd.sound_line("Dan walks in.", "", True) == ("There is no background music.", False))
    check("nothing to hear, nothing said, and no music", S.snd.sound_line("Mara thinks.", "", False)
          == ("There is no background music.", False))
    check("music the beat or the scene asks for is allowed",
          S.snd.sound_line("Dan turns the radio up.", "", True) == ("", False)
          and "no background music" not in S.snd.sound_line("Mara walks in.", "A piano bar.", False)[0]
          and "no background music" not in S.snd.sound_line("Mara sings softly.", "", False)[0])
    check("at most three sounds, and closing cuffs replace rattling ones",
          len(S.snd.foley("Dan walks in, slams the door, drags a chair, pours water and drops the keys.")) == 3
          and S.snd.foley("Dan handcuffs her and yanks the cuffs.") == ["cuffs ratcheting closed"])
    check("a vocal sound alone adds no foley", S.snd.foley("Mara whimpers.") == [])


def test_ambient_bed():
    print("\n=== ambient bed ===")
    track = torch.zeros(1, 2, 1000)
    bed = {"waveform": torch.ones(1, 1, 300) * 0.5, "sample_rate": 100}
    out, note = S.au.mix_ambient(track, 100, bed, 0.4)
    check("a bed is looped under the whole soundtrack at its level",
          out.shape == track.shape and abs(float(out[0, 0, 500]) - 0.2) < 1e-4 and abs(float(out[0, 1, 900]) - 0.2) < 1e-4
          and "ambient bed mixed at 0.40" in note, (out.shape, note))
    out, _ = S.au.mix_ambient(track, 200, bed, 0.4)
    check("a bed at another sample rate is resampled", out.shape == track.shape)
    out, _ = S.au.mix_ambient(torch.full((1, 2, 100), 0.9), 100, bed, 1.0)
    check("the mix is scaled down instead of clipping", float(out.abs().max()) <= 1.0 + 1e-6)
    check("level 0 or no bed leaves the soundtrack alone",
          S.au.mix_ambient(track, 100, bed, 0.0)[0] is track and S.au.mix_ambient(track, 100, None, 0.5)[0] is track)
    out, note = S.au.mix_ambient(track, 100, {"waveform": None}, 0.5)
    check("an empty bed says so", out is track and "no usable waveform" in note, note)
    out, calls = render(SCRIPT, ambient_audio={"waveform": torch.ones(1, 2, 50) * 0.1, "sample_rate": 1000}, ambient_level=0.5)
    check("the node mixes the bed into its soundtrack", "ambient bed mixed at 0.50" in out[2], out[2])


def test_recovered_look_render():
    print("\n=== a current look carried across a cut ===")
    script = "A cell. Mara <Picture 1> waits.\n\nMara paces.\n\ncut\nA garden. Mara walks."
    out, calls = render(script, ref_image_1=IMG1)
    check("the shot after a cut gets the last frame of the shot she was alone in",
          len(calls["cond"]) == 2 and calls["cond"][1]["handoff"] is None and len(calls["cond"][1]["refs"]) == 1
          and abs(float(calls["cond"][1]["refs"][0].mean()) - (0.1 + (S.rt.align_frame_count(48) - 1) / 1e4)) < 1e-4,
          [(c["handoff"] is None, len(c["refs"])) for c in calls["cond"]])


def test_pictures_by_checkpoint():
    print("\n=== reference pictures on every checkpoint ===")
    graph = {"9": {"class_type": "H3LongVideos", "inputs": {"model": ["7", 0], "clip": ["2", 0]}},
             "7": {"class_type": "LoraLoader", "inputs": {"model": ["5", 0], "clip": ["2", 0], "lora_name": "hyperflow.safetensors"}},
             "5": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3/10Eros_Max_H3_FL2VA-INT8-ConvRot-HQ.safetensors"}},
             "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors"}},
             "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl.safetensors"}}}
    check("the checkpoint is traced back through the LoRA loaders on the node's own model link",
          S.checkpoint_name(graph, "9") == "h3/10Eros_Max_H3_FL2VA-INT8-ConvRot-HQ.safetensors" and S.checkpoint_name({}, "9") == "")
    hybrid = dict(graph, **{"9": {"class_type": "H3LongVideos", "inputs": {"model": ["3", 0]}}})
    check("only a ref2va checkpoint takes pictures in every shot; an unknown loader is treated as one",
          S.takes_pictures("m", graph, "9")[0] is False and "10Eros_Max_H3_FL2VA" in S.takes_pictures("m", graph, "9")[1]
          and S.takes_pictures("m", hybrid, "9") == (True, "") and S.takes_pictures("m", None, None) == (True, ""))
    import folder_paths
    import tempfile
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory() as tmp:
        tensor = {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(1)}
        save_file({"diffusion_model.time_embedder.proj_in.lora_A.weight": torch.zeros(1)}, os.path.join(tmp, "hyperflow.safetensors"))
        save_file(tensor, os.path.join(tmp, "student.safetensors"), {"model": "4-step student, LoRA on the T2VA transformer"})
        save_file(tensor, os.path.join(tmp, "flow.safetensors"), {"tasks": '["t2va", "fl2va", "ref2va"]', "hyperflow": "true",
                                                                 "hyperflow_gate": "0.25"})
        save_file(tensor, os.path.join(tmp, "minimax_h3_fl2v_turbo_8step.safetensors"))
        save_file(tensor, os.path.join(tmp, "hybrid-4to8step-Turbo_REF2VAHybrid-T8.safetensors"))
        save_file(tensor, os.path.join(tmp, "dmad.safetensors"), {"sampling": "4 steps, video shift 12, audio shift 2, "
                                                                            "re-noise step rule, no CFG"})
        save_file(tensor, os.path.join(tmp, "style.safetensors"))
        was = folder_paths.get_full_path, folder_paths.get_filename_list
        folder_paths.get_full_path = lambda kind, name: os.path.join(tmp, name) if os.path.isfile(os.path.join(tmp, name)) else None
        folder_paths.get_filename_list = lambda kind: sorted(os.listdir(tmp))
        on = lambda curve: SimpleNamespace(get_model_object=lambda name: SimpleNamespace(use_adaln_curves=curve))
        stack = lambda *rows: {"9": {"inputs": {"model": ["8", 0]}}, "3": hybrid["3"], "8": {"inputs": {
            "model": ["3", 0], "stack_data": json.dumps([{"on": o, "lora": n, "str": st} for n, o, st in rows])}}}
        try:
            got = (S.partial_loras(on(True), graph, "9"), S.partial_loras(on(False), graph, "9"),
                   S.partial_loras(on(True), hybrid, "9"))
            blind = S.takes_pictures("m", stack(("student.safetensors", True, 1.0), ("style.safetensors", True, 1.0)), "9")
            named = S.takes_pictures("m", stack(("minimax_h3_fl2v_turbo_8step.safetensors", True, 0.8)), "9")
            fine = [S.takes_pictures("m", g, "9") for g in (
                stack(("flow.safetensors", True, 1.0), ("style.safetensors", True, 1.0)),
                stack(("student.safetensors", False, 1.0), ("minimax_h3_fl2v_turbo_8step.safetensors", True, 0)),
                {"9": {"inputs": {"model": ["7", 0]}}, "3": hybrid["3"],
                 "7": {"inputs": {"model": ["3", 0], "lora_name": "student.safetensors", "strength_model": 0}}})]
            flow = S.hyperflow_lora(SimpleNamespace(), stack(("flow.safetensors", True, 1.0)))
            core = lambda on: S.takes_pictures("m", {"9": {"inputs": {"model": ["8", 0]}}, "3": hybrid["3"], "8": {"inputs": {
                "model": ["3", 0], "loras.0.lora_name": "student.safetensors", "loras.0.strength": 1.0,
                "loras.0.enabled": on, "loras.1.lora_name": "style.safetensors", "loras.1.strength": 1.0,
                "loras.1.enabled": True}}}, "9")[0]
            rows = (core(False), core(True))
            hybrid_turbo = S.takes_pictures("m", stack(("hybrid-4to8step-Turbo_REF2VAHybrid-T8.safetensors", True, 1.0)), "9")
            dmad = S.prepare_model(Patched([]), 8, "res_multistep", "simple", None, 12.0, 3.0,
                                   stack(("dmad.safetensors", True, 1.0)))
            own = S.prepare_model(Patched([]), 8, "res_multistep", "simple", torch.tensor([1.0, 0.0]), 12.0, 3.0,
                                  stack(("dmad.safetensors", True, 1.0)))
            save_file(tensor, os.path.join(tmp, "minimax_h3_4step_lora_flashgen_v1.0.safetensors"))
            flash = S.prepare_model(Patched([]), 8, "res_multistep", "simple", None, 10.0, 2.0,
                                    stack(("minimax_h3_4step_lora_flashgen_v1.0.safetensors", True, 1.0)))
            turbo = S.prepare_model(Patched([]), 8, "res_multistep", "simple", None, 12.0, 3.0,
                                    stack(("style.safetensors", True, 0.25), ("minimax_h3_fl2v_turbo_8step.safetensors", True, 1.0)))
            plain = S.prepare_model(Patched([]), 30, "res_multistep", "simple", None, 12.0, 3.0,
                                    stack(("style.safetensors", True, 1.0)))
        finally:
            folder_paths.get_full_path, folder_paths.get_filename_list = was
    check("a LoRA trained without reference pictures, found in any loader by its file, keeps a person's picture out of a "
          "shot that opens on a frame they are in; one trained with them, a style LoRA, or one switched off does not",
          blind == (False, "student.safetensors was trained without reference pictures") and named[0] is False
          and all(f == (True, "") for f in fine), (blind, named, fine))
    check("ComfyUI's own LoRA Stack is read row by row, so a row switched off does not count", rows == (True, False), rows)
    tops = lambda r: r[0].model_options["transformer_options"]
    grid = [12 * x / (1 + 11 * x) for x in (1.0, 0.7, 0.4, 0.15, 0.0)]
    check("FlashGen, whose file carries no recipe, gets the one it was trained with: 4 euler steps on its own schedule "
          "shifted at 12/3",
          flash[1:3] == (4, "euler") and all(abs(a - b) < 1e-5 for a, b in zip(flash[3].tolist(), grid))
          and tops(flash)["minimax_h3_sigma_shift_video"] == 12.0 and tops(flash)["minimax_h3_sigma_shift_audio"] == 3.0,
          flash[1:])
    check("any few-step speed LoRA, stacked with a style LoRA or not, is sampled with euler on the steps set and no "
          "extra audio step, and without one the sampler set is kept",
          turbo[1:3] == (8, "euler") and turbo[3] is None and len(turbo[4]) == 9
          and "a multistep sampler overshoots" in " ".join(turbo[5])
          and plain[1:3] == (30, "res_multistep") and len(plain[4]) == 31, (turbo[1:], plain[2], len(plain[4])))
    check("a speed LoRA built for a ref2va checkpoint keeps the pictures, turbo or not", hybrid_turbo == (True, ""),
          hybrid_turbo)
    check("a LoRA whose metadata gives its sampling is sampled that way, steps, shifts and a re-noise rule as lcm, unless "
          "sigmas are wired",
          dmad[1:3] == (4, "lcm") and dmad[3] is None and tops(dmad)["minimax_h3_sigma_shift_audio"] == 2.0
          and len(dmad[4]) == 5 and "the way it was trained" in " ".join(dmad[5])
          and own[1:3] == (8, "res_multistep") and tops(own)["minimax_h3_sigma_shift_audio"] == 3.0, (dmad[1:], own[1:]))
    check("Hyperflow loaded through a LoRA stack is found, with its gate read from its own file",
          flow is not None and flow["gate"] == 0.25 and flow["source"] == "flow.safetensors", flow)
    check("a LoRA with timestep layers on a time-curve checkpoint is named as only partly loaded, and not on a checkpoint "
          "with a time embedder or without that LoRA", got == (["hyperflow.safetensors"], [], []), got)
    script = "A cell. Mara <Picture 1> waits. Dan <Picture 2> stands guard.\n\nMara paces.\n\nMara sits.\n\ncut\nA garden. Mara walks."
    out, calls = render(script, ref_image_1=IMG1, ref_image_2=IMG1, graph=graph, unique_id="9")
    check("on an FL2VA checkpoint like 10Eros the pictures go in except where she is already in the opening frame",
          [len(c["refs"]) for c in calls["cond"]] == [1, 0, 1] and "not trained on reference pictures" in out[2],
          [len(c["refs"]) for c in calls["cond"]])
    entry = "A cell. Mara <Picture 1> waits. Dan <Picture 2> is a guard.\n\nMara paces.\n\nDan walks in."
    _, calls = render(entry, ref_image_1=IMG1, ref_image_2=IMG2, graph=graph, unique_id="9")
    check("someone who walks into a continued shot still gets their picture, renumbered",
          [len(c["refs"]) for c in calls["cond"]] == [1, 1] and abs(float(calls["cond"][1]["refs"][0].mean()) - 0.2) < 1e-6
          and "Dan <Picture 1>" in calls["cond"][1]["prompt"] and "Mara <Picture" not in calls["cond"][1]["prompt"],
          ([len(c["refs"]) for c in calls["cond"]], calls["cond"][1]["prompt"][:200]))
    out, calls = render(script, ref_image_1=IMG1, ref_image_2=IMG1, graph=hybrid, unique_id="9")
    check("on the hybrid fl2va_ref2va the portraits still go in", [len(c["refs"]) for c in calls["cond"]] == [1, 1, 1]
          and "not trained on reference pictures" not in out[2], [len(c["refs"]) for c in calls["cond"]])
    memory = "Crystal <Picture 1>: a slim woman.\nMike <Picture 2>: a tall man.\nhold: Crystal, handcuffs behind her back"
    late = "A cell.\n\nMike waits by the door.\n\nMike watches as Crystal is pushed into the cell.\n\nMike grabs Crystal <Picture 1> by the arm."
    for pics in (True, False):
        shots = S.plan_shots(late, 5.0, [IMG1, IMG2, None, None], False, memory=memory, pictures=pics)
        check(f"someone already held keeps their picture in the shot they enter (ref model: {pics}), and it says they come in",
              [len(s["refs"]) for s in shots][1] == (2 if pics else 1) and "Crystal <Picture 1>" in shots[1]["prompt"]
              and "carried forward; Crystal (exactly as in <Picture 1>) comes into view during this shot, and nobody else joins."
              in shots[1]["prompt"]
              and "Crystal <Picture" not in shots[2]["prompt"] and "rather than joined by anybody new" in shots[2]["prompt"],
              [(len(s["refs"]), s["prompt"][-300:]) for s in shots])
        check("a restated action carries no picture tags", "It is Mike who grabs Crystal by the arm, with his own hands."
              in shots[2]["prompt"], shots[2]["prompt"])
    tagged = S.plan_shots("A bedroom.\n\nShe sits on the bed.\n\nHe walks in and cuffs her wrists.", 5.0, [IMG1, IMG2, None, None],
                          False, memory="Crystal <Picture 1>: a slim woman.\nMike (<Picture 2>): a tall man.")
    check("she and he find characters declared only in memory, even with a picture tag between name and colon",
          [(s["cast"], len(s["refs"])) for s in tagged] == [(["Crystal"], 1), (["Crystal", "Mike"], 2)]
          and tagged[1]["added"] == "Crystal: handcuffs on her wrists.", [(s["cast"], len(s["refs"]), s["added"]) for s in tagged])


def test_fast_h3_pictures():
    print("\n=== FastH3 gets pictures only where nobody can be drawn twice ===")
    script = "A cell. Mara <Picture 1> waits. Dan <Picture 2> stands guard.\n\nMara paces.\n\nMara sits.\n\ncut\nA garden. Mara walks."
    out, calls = render(script, fast=True, ref_image_1=IMG1, ref_image_2=IMG1)
    check("her portrait opens the video, is left out while she is in the opening frame, and her look returns after the cut",
          [len(c["refs"]) for c in calls["cond"]] == [1, 0, 1] and "Mara <Picture" not in calls["cond"][1]["prompt"],
          [len(c["refs"]) for c in calls["cond"]])
    check("the opening frame still carries the shot on as <Picture 1>",
          calls["cond"][1]["handoff"] is not None and "<Picture 1> is the frame this shot opens on" in calls["cond"][1]["prompt"]
          and calls["cond"][2]["handoff"] is None, [c["handoff"] is None for c in calls["cond"]])
    check("info says when pictures are left out", "distilled without reference pictures" in out[2]
          and "already in" in out[2], out[2])
    _, base = render(script, ref_image_1=IMG1, ref_image_2=IMG1)
    check("other checkpoints still get her portrait in every shot, and her current look after the cut",
          [len(c["refs"]) for c in base["cond"]] == [1, 1, 1] and float(base["cond"][2]["refs"][0].mean()) != 0.1,
          [len(c["refs"]) for c in base["cond"]])

def test_memory():
    print("\n=== running out of VRAM ===")
    check("ComfyUI's dynamic VRAM failure counts as out of memory",
          S.rt._is_oom(RuntimeError("VRAM grow failed: 272157696 bytes"))
          and S.rt._is_oom(torch.cuda.OutOfMemoryError("CUDA out of memory")) and not S.rt._is_oom(ValueError("bad"))
          and S.pose._is_oom(RuntimeError("VRAM grow failed: 1 bytes")))
    log = []

    def run(error):
        def go(*a, **k):
            log.append("sample")
            raise error
        log.clear()
        S.rt._sample_on_sigmas = go
        try:
            S.sample("m", "c", "n", {}, 7, 8, "euler", "simple", torch.ones(3))
        except Exception as e:
            return e
    saved = S.rt._sample_on_sigmas
    try:
        got = run(RuntimeError("VRAM grow failed: 1 bytes"))
        check("a pass that runs out of VRAM says what to lower, without retrying into a broken memory plan",
              isinstance(got, RuntimeError) and "lower megapixels or shot_seconds" in str(got) and log == ["sample"], (got, log))
        got = run(ValueError("bad input"))
        check("other errors pass through untouched", isinstance(got, ValueError) and log == ["sample"], (got, log))
    finally:
        S.rt._sample_on_sigmas = saved
    base = SimpleNamespace()
    seen = {}

    def executor(model, noise_shape, conds, **kw):
        seen["need"] = model.model.memory_required([1, 24, 37, 48, 84])
        seen["kw"] = kw
        return "loaded"
    model = SimpleNamespace(model=base)
    got = S.reserve_activations(executor, model, [1, 24, 37, 48, 84], [], force_offload=False)
    tokens = 37 * 24 * 42
    check("model loading reserves room for the shot's real activations, about 120 KB per token, then puts it back",
          got == "loaded" and seen["need"] == tokens * S.H3_TOKEN_BYTES + S.H3_FIXED_BYTES and "memory_required" not in vars(base)
          and seen["kw"] == {"force_offload": False}, seen)
    base.memory_required = "theirs"
    S.reserve_activations(executor, model, [1, 24, 37, 48, 84], [])
    check("an estimate someone else already set is restored afterwards", base.memory_required == "theirs")
    big = S.h3_activations([1, 24, 72, 48, 84])
    check("a 10 s shot at 1344x768 reserves about 9.4 GB for activations instead of ComfyUI's under 1 GB",
          9.0e9 < big < 10.0e9, big)
    packed = S.h3_activations([1, 1, 24 * 72 * 48 * 84 + 32 * 2 * 300])
    check("the same shot reserves the same when ComfyUI hands over its video and audio packed into one row",
          abs(packed - big) < 0.01 * big, (packed, big))

    pdd = S.prepare_model(Patched(["minimax_h3_pdd_acc"]), 8, "res_multistep", "simple", torch.tensor([1.0, 0.5, 0.0]),
                          10.0, 2.0, {})
    try:
        S.prepare_model(Patched(["minimax_h3_pdd_acc"]), 8, "res_multistep", "simple", None, 12.0, 3.0, {})
        unwired = ""
    except ValueError as e:
        unwired = str(e)
    check("a model with PDD Acc on it is sampled the way PDD was trained, euler at shift 12/3 on its own sigmas, and its "
          "sigmas left unwired stops the run with what to wire",
          pdd[2] == "euler" and pdd[0].model_options["transformer_options"]["minimax_h3_sigma_shift_video"] == 12.0
          and pdd[0].model_options["transformer_options"]["minimax_h3_sigma_shift_audio"] == 3.0
          and pdd[3].tolist() == [1.0, 0.5, 0.0] and "wire its sigmas output" in unwired, (pdd[1:], unwired))
    try:
        S.check_vaes(SimpleNamespace(first_stage_model=SimpleNamespace(decoder=1)), None)
        half = ""
    except RuntimeError as e:
        half = str(e)
    check("a decoder-only video VAE is refused with what to wire instead",
          "no encoder" in half and S.check_vaes(SimpleNamespace(first_stage_model=SimpleNamespace(decoder=1, encoder=1)),
                                                None) is None, half)


def test_grading():
    print("\n=== the look holds from shot to shot ===")
    rt = S.rt
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 96), torch.linspace(0, 1, 168), indexing="ij")
    blobs = sum(torch.exp(-((xx - torch.rand(1)) ** 2 + (yy - torch.rand(1)) ** 2) / 0.02) * torch.rand(3).view(3, 1, 1)
                for _ in range(12))
    orig = (0.25 + 0.5 * blobs / blobs.max() + 0.05 * torch.randn(3, 96, 168)).clamp(0, 1).permute(1, 2, 0)

    def burn(img, amount):
        y = rt._ycc(img)
        y[..., 0] = ((y[..., 0] - 0.5) * (1 + 0.6 * amount) + 0.5).clamp(0, 1) ** (1 + 0.4 * amount)
        y[..., 1:] = (y[..., 1:] - 0.5) * (1 + amount) + 0.5
        return rt._rgb(y).clamp(0, 1)
    sat = lambda img: float((rt._ycc(img)[..., 1:] - 0.5).abs().mean())
    con = lambda img: float(rt._ycc(img)[..., 0].std())
    creep = torch.stack([burn(orig, 0.25 * i / 11) for i in range(12)])
    start, anchor = creep[0].clone(), rt.look(orig)
    held = rt.hold_look(creep, anchor)
    end = rt.look(creep[-1])
    fading = torch.stack([burn(orig, -0.25 * i / 11) for i in range(12)])
    kept = fading.clone()
    big = torch.nn.functional.interpolate(orig.permute(2, 0, 1)[None], size=(288, 504), mode="bilinear")[0].permute(1, 2, 0)
    frame = big[None].clone()
    toned = rt.precompensate(frame, (1.02, 1.06))
    want = rt.look(frame)
    got = rt.look(toned)
    check("the frame a shot starts from is toned down by the boost H3 was seen to add, so the shot lands on the true "
          "look, and with no boost it is handed over as it is",
          abs(got[1] / want[1] - 1 / 1.06) < 0.01 and abs(got[0] / want[0] - 1 / 1.02) < 0.01
          and rt.precompensate(frame, None) is frame and rt.precompensate(frame, (0.98, 0.99)) is frame
          and torch.equal(frame, big[None]), (got, want))
    check("a shot that burns as it runs is eased back to the scene's look by its last frame, from nothing at its first "
          "frame, and one that only fades is left alone",
          held and torch.allclose(creep[0], start, atol=1e-5) and end[0] <= anchor[0] * 1.035 and end[1] <= anchor[1] * 1.035
          and not rt.hold_look(fading, anchor) and torch.equal(fading, kept),
          (anchor, end))
    gen = torch.Generator().manual_seed(7)
    noise = lambda s: rt._blur(torch.randn(1, 1, 288, 504, generator=gen), s)[0, 0, ..., None].expand(-1, -1, 3)
    clean = (torch.linspace(0.2, 0.7, 504).view(1, -1, 1).expand(288, -1, 3) + 0.02 * noise(1.0)).clone()
    clean[100:200, 150:300] += 0.25
    mottle = noise(2.5)
    crust = torch.stack([(clean + 0.06 * i / 11 * mottle).clamp(0, 1) for i in range(12)])
    first, grain = crust[0].clone(), rt.texture(clean)
    smooth = rt.hold_texture(crust, grain)
    step = lambda x: float(x[120:180, 160:290].mean() - x[120:180, 310:440].mean())
    calm = crust[:1].clone()
    rough = torch.stack([(clean + 0.06 * mottle).clamp(0, 1)] * 6)
    opened = rt.hold_texture(rough, grain)
    check("a shot whose smooth surfaces turn blotchy as it runs is eased back to the scene's texture by its last frame, "
          "a shot that opens blotchy is held from its first frame, edges keep their strength, and a shot no rougher "
          "than the scene is left alone",
          smooth and torch.allclose(crust[0], first, atol=1e-5)
          and rt.texture(crust[-1], grain[1])[0] <= grain[0] * 1.035 and abs(step(crust[-1]) / step(clean) - 1) < 0.02
          and opened and max(rt.texture(f, grain[1])[0] for f in rough) <= grain[0] * 1.035
          and not rt.hold_texture(calm, grain) and torch.equal(calm, crust[:1]),
          (grain, rt.texture(crust[-1], grain[1]), [rt.texture(f, grain[1]) for f in rough[:1]],
           step(crust[-1]) / step(clean)))
    def chain(frame):
        key, look = orig.clone(), None
        for n in range(12):
            f = torch.stack([frame(key, k) for k in range(24)])
            g = rt.shot_grade(None if n == 0 else key, f[0], f[-1])
            if g is not None:
                rt.grade_frames(f, g)
            key = f[-1].clone()
            look = key.clone() if n == 0 else look
        return key, look
    key, look = chain(lambda key, k: burn(key, 0.05))
    check("twelve shots that each open more saturated and contrasty than the frame they continue from keep the first shot's look",
          abs(sat(key) / sat(look) - 1) < 0.02 and abs(con(key) / con(look) - 1) < 0.02, (sat(key) / sat(look), con(key) / con(look)))
    key, look = chain(lambda key, k: burn(key, -0.15))
    check("twelve shots that each open a little paler than the frame they continue from do not wash the video out",
          abs(sat(key) / sat(look) - 1) < 0.02 and abs(con(key) / con(look) - 1) < 0.02, (sat(key) / sat(look), con(key) / con(look)))
    moved = torch.stack([orig.clone() for _ in range(24)])
    moved[-1, 30:70, 20:60] = 0.05
    before = float(rt.tone(moved[-1])[0][3])
    g = rt.shot_grade(orig, moved[1], moved[-1])
    after = float(rt.tone(rt.grade_frames(moved, g)[-1])[0][3]) if g is not None else before
    check("when someone dark steps into the end of a shot its shadows stay dark: it is never forced onto the opening's tones",
          abs(after - before) < 0.02, (before, after))
    first = torch.stack([burn(orig, 0.1 * k / 23) for k in range(24)])
    check("the first shot, and a shot after a cut, set the look and are left as rendered",
          rt.shot_grade(None, first[0], first[-1]) is None)
    f = torch.stack([burn(orig, 0.05) * (1 - 0.6 * k / 23) for k in range(24)])
    dark = float(f[-1].mean())
    rt.grade_frames(f, rt.shot_grade(orig, f[0], f[-1]))
    check("a real change inside a shot, like the lights going down, is kept", float(f[-1].mean()) < 0.6 * float(f[0].mean())
          and float(f[-1].mean()) < 1.2 * dark, (float(f[0].mean()), float(f[-1].mean()), dark))
    grows = torch.stack([burn(orig, 0.3 * k / 23) for k in range(24)])
    g = rt.shot_grade(orig, grows[1], grows[-1])
    check("how a shot develops after its opening is its own: nothing inside a shot is forced onto another frame's tones",
          g is None or float((rt.tone(rt.grade_frames(grows.clone(), g)[-1]) - rt.tone(grows[-1])).abs().max()) < 0.01)
    first = burn(orig, 0.2)
    g = rt.shot_grade(orig, first, first)
    graded = rt.grade_frames(first.unsqueeze(0).clone(), g)[0]
    check("a continued shot opens on the same look as the frame it continues from",
          float((rt.tone(graded) - rt.tone(orig)).abs().mean()) < 0.002 and float((rt.tone(first) - rt.tone(orig)).abs().mean()) > 0.01)
    black = torch.full((24, 96, 168, 3), 0.02)
    check("a black or faded opening is left alone instead of being stretched to the last shot's look",
          rt.shot_grade(orig, black[0], black[-1]) is None)
    other = torch.flip(orig, dims=[-1]).roll(1, dims=-1) * 0.4
    check("a shot that does not pick up where the last one ended is left as rendered", rt.shot_grade(orig, other, other) is None)
    check("nothing is touched when the shot already matches", rt.shot_grade(orig, orig, orig) is None)

    def washout(img, amount):
        y = rt._ycc(img)
        mid = y[..., 0].median()
        y[..., 0] = (y[..., 0] - mid) * (1 - 0.5 * amount) + mid
        y[..., 1:] = (y[..., 1:] - 0.5) * (1 - amount) + 0.5
        return rt._rgb(y).clamp(0, 1)
    pale = torch.stack([washout(orig, 0.7) for _ in range(24)])
    rt.grade_frames(pale, rt.shot_grade(orig, pale[1], pale[-1]))
    check("a continued shot that comes out badly washed out, same scene and brightness, is given its colour back",
          sat(pale[12]) / sat(orig) > 0.95 and con(pale[12]) / con(orig) > 0.95, (sat(pale[12]) / sat(orig), con(pale[12]) / con(orig)))
    punchy = torch.stack([burn(orig, 0.3)] + [orig.clone() for _ in range(23)])
    g = rt.shot_grade(orig, punchy[1], punchy[-1])
    old_way = rt.shot_grade(orig, punchy[0], punchy[-1])
    drained = rt.grade_frames(punchy.clone(), old_way) if old_way is not None else punchy
    check("an odd opening frame that is trimmed anyway does not drain the frames after it",
          g is None and sat(drained[6]) / sat(orig) < 0.97, (g is None, sat(drained[6]) / sat(orig)))
    check("the look readout gives contrast and colour, and nothing for a flat frame",
          rt.look(orig) is not None and rt.look(burn(orig, 0.3))[1] > rt.look(orig)[1] and rt.look(black[0]) is None)
    hand = rt.match_frame(burn(orig, 0.3).unsqueeze(0).clone(), rt.tone(orig))
    check("the handing-off frame can be put back on the shot's look on its own",
          float((rt.tone(hand) - rt.tone(orig)).abs().mean()) < 0.004)


def test_lora_refit():
    print("\n=== LoRA timestep layers re-fitted to the loaded checkpoint ===")
    from comfy.weight_adapter.lora import LoRAAdapter
    P = S.pose
    t = torch.linspace(0, 1, 4097)
    eros = torch.stack([torch.sin((k + 1) * 1.7 * t) * (0.5 + 0.1 * k) for k in range(16)], dim=1)
    hyb = eros @ torch.linspace(-1, 1, 128).reshape(16, 8) + torch.linspace(0.2, 0.9, 8)

    class Patcher:
        def __init__(self, patches):
            self.patches, self.patches_uuid = patches, 0
            lin = SimpleNamespace(in_features=16)
            self.dm = SimpleNamespace(use_adaln_curves=True, adaln_t_table=eros,
                                      blocks=[SimpleNamespace(adaln_proj=SimpleNamespace(linear=lin))])

        def get_model_object(self, name):
            return self.dm

        def clone(self):
            return Patcher({k: list(v) for k, v in self.patches.items()})
    torch.manual_seed(0)
    up, down, db = torch.randn(64, 4), torch.randn(4, 8), torch.randn(64) * 0.1
    key = "diffusion_model.blocks.0.adaln_proj.linear.weight"
    bias = key[:-6] + "bias"
    model = Patcher({key: [(1.0, LoRAAdapter({}, (up, down, 4.0, None, None, None)), 1.0, None, None)],
                     bias: [(1.0, ("diff", (db,)), 1.0, None, None)]})
    curves = S.pose.curve_table
    P.curve_table = lambda width: hyb if width == 8 else None
    try:
        out, note = S.pose.bridge_lora_timesteps(model)
    finally:
        P.curve_table = curves
    pts = torch.rand(500)
    rows = lambda table: S.pose._curve_rows(table, pts)
    want = rows(hyb) @ down.T @ up.T + db
    got = sum(rows(eros) @ e[1].weights[1].T @ e[1].weights[0].float().T * (1.0 if e[1].weights[2] is None else
                                                                          float(e[1].weights[2]) / e[1].weights[1].shape[0])
              for e in out.patches[key]) + sum(b[1][1][0] for b in out.patches[bias])
    check("an 8-wide LoRA's timestep change is re-fitted onto a 16-wide checkpoint and lands where it was meant to",
          float((got - want).norm() / want.norm()) < 1e-3 and "re-fitted" in note, (float((got - want).norm() / want.norm()), note))
    check("the incoming model's own patches are left untouched", model.patches[key][0][1].weights[1] is down
          and model.patches[bias][0][1][1][0] is db and out is not model)
    same = Patcher({key: [(1.0, LoRAAdapter({}, (up, torch.randn(4, 16), 4.0, None, None, None)), 1.0, None, None)]})
    check("a LoRA built for this checkpoint's width is left alone", S.pose.bridge_lora_timesteps(same) == (same, ""))
    P.curve_table = lambda width: None
    saved = S.pose.full_curve
    S.pose.full_curve = lambda: None
    try:
        _, note = S.pose.bridge_lora_timesteps(model)
    finally:
        S.pose.curve_table, S.pose.full_curve = curves, saved
    check("without the checkpoint it was built for, info says so", "could not be re-fitted" in note, note)


def test_reading():
    print("\n=== restraints read from the beats ===")
    script = ("A dim room. Mara lies on the bed. Dan stands beside her.\n\nDan handcuffs her wrists behind her back.\n\n"
              "Dan wraps duct tape around her mouth.\n\nDan sits down and watches her.\n\nDan pulls the tape off.")
    shots = S.plan_shots(script, 10.0, [None] * 4, False)
    check("cuffs go on, then stay on while the tape goes on, with no hold lines",
          shots[0]["added"] == "Mara: handcuffs behind her back." and shots[1]["held"] == "Mara: handcuffs behind her back."
          and shots[1]["added"] == "Mara: duct tape over her mouth."
          and "Mara's wrists stay locked together behind the back the whole time." in shots[1]["prompt"], [S.shot_line(x) for x in shots])
    check("pose control latches the cuffs as they go on and holds them after",
          shots[0]["bound"]["Mara"]["latch_limbs"] == ("arms",) and shots[1]["bound"]["Mara"]["latch_limbs"] == ()
          and all(x["bound"].get("Mara", {}).get("arms") == "behind the back" for x in shots[1:]))
    check("the tape comes off and the cuffs stay", shots[3]["released"] == "Mara: duct tape over her mouth."
          and shots[3]["held"] == "Mara: handcuffs behind her back.", S.shot_line(shots[3]))
    later = S.plan_shots("A cell. Mara waits. Dan stands guard.\n\nDan cuffs her.\n\nMara kneels with her hands cuffed behind her "
                         "back.\n\nDan waits.", 5.0, [None] * 4, False)
    check("cuffs read with no position take the one a later beat gives, and keep it",
          later[0]["added"] == "Mara: handcuffs on her wrists." and later[1]["held"] == "Mara: handcuffs behind her back."
          and later[1]["released"] == "Mara: handcuffs on her wrists." and later[2]["held"] == "Mara: handcuffs behind her back."
          and "Mara's wrists stay locked together behind the back the whole time." in later[1]["prompt"]
          and all(x["bound"]["Mara"]["arms"] == "behind the back" for x in later[1:] if "Mara" in x["cast"]),
          [S.shot_line(x) for x in later])
    placed = S.plan_shots("A cell.\nhold: Mara, handcuffs\n\nMara wears handcuffs behind her back.\n\nDan waits.", 5.0,
                          [None] * 4, False)
    check("a hold with no position takes the one the beat gives from that shot's first frame",
          placed[0]["held"] == "Mara: handcuffs behind her back." and placed[0]["added"] == ""
          and placed[1]["held"] == "Mara: handcuffs behind her back.", [S.shot_line(x) for x in placed])
    people, gender = ["Mara", "Dan"], {"Mara": "f", "Dan": "m"}
    read = lambda t, held=None: S.rst.read(t, people, gender, held or {})[0]
    leashed = "a collar around her neck, with a chain leash clipped to it"
    check("a leash goes with the collar it is clipped to, never as chains on her wrists",
          all(read(t) == {"Mara": [leashed]} for t in ("Dan locks a collar around her neck and clips a chain leash to it.",
                                                      "Dan clips a chain leash to her collar.", "Dan attaches a chain leash to Mara's collar.",
                                                      "Mara wears a collar with a chain leash.", "Mara kneels in a collar and chain leash."))
          and read("Dan clips a leash to it.", {"Mara": ["a collar around her neck"]})
          == {"Mara": ["a collar around her neck, with a leash clipped to it"]}
          and read("Mara kneels in a steel collar.") == {"Mara": ["a collar around her neck"]})
    off = S.rst.read("Dan unclips the leash.", people, gender, {"Mara": [leashed]})
    check("unclipping the leash leaves the collar on, already in place",
          off[0] == {"Mara": ["a collar around her neck"]} and off[1] == {"Mara": [leashed]}
          and off[3] == {"Mara": ["a collar around her neck"]}, off)
    check("rope carried in, chains on a wall and a grab by the collar put nothing on anyone",
          all(read(t) == {} for t in ("Dan enters with a coil of rope.", "Mara sits in a room with chains on the wall.",
                                      "Dan grabs her by the collar.")))
    memory = "Mara: a slim woman. She wears a steel chastity belt with a chain hanging from the back.\nDan: a tall man."
    leash = S.plan_shots("A cell.\n\nMara kneels with her wrists cuffed behind her back.\n\nDan locks a collar around her neck and "
                         "clips a chain leash to it.\n\nThe camera frames her face.\n\nDan pulls her up by the leash.\n\n"
                         "Dan unclips the leash.", 5.0, [None] * 4, False, memory=memory)
    check("the leash runs to the hand that holds it, from the shot it goes on until it comes off, and nobody else wears it",
          [("The leash runs from Mara's collar to Dan's hand, and its links never stretch." in x["prompt"]) for x in leash]
          == [False, True, False, True, False] and leash[2]["cast"] == ["Mara"]
          and "The leash and the chain on her steel chastity belt are two separate chains." in leash[3]["prompt"]
          and "Dan wears none of it." in leash[3]["prompt"]
          and "in plain view." in leash[1]["prompt"] and "has let go" not in leash[1]["prompt"], [x["prompt"][-900:] for x in leash])
    twist = "Mara's body never twists at the waist: her head, chest, hips and legs always face the same way and turn together."
    tight = ("Mara's chastity belt is worn the right way round. The front of the belt sits low on her belly, over her crotch, and "
             "its chain sits between her buttocks, up to the small of her back; it turns with her.")
    fast = "The chain on Mara's chastity belt stays fastened at both ends, short and snug against her; it never hangs loose, trails or stretches."
    side = lambda d: S.side_notes(f"Mara: a slim woman. {d}", ["Mara"], gender)
    check("a hanging chain sits on her backside, a fastened one between her buttocks, the front low on her belly, every shot",
          all("Mara's steel chastity belt is worn the right way round. The front of the belt sits low on her belly, over her "
              f"crotch, and its chain sits on her backside, at the small of her back; it turns with her. {twist}" in x["prompt"]
              for x in leash) and "stays fastened" not in leash[0]["prompt"], leash[0]["prompt"])
    check("the belt is found however it is written, across sentences and with no article, and a plain one still gets its line",
          all(side(d) == [tight, twist, fast] for d in (
              "She wears chastity belt with tight rear heavy duty security chain going through her ass.",
              "She wears a chastity belt. A chain is attached to the back of the belt.",
              "She wears a chastity belt that has a chain at the back.", "She wears a chastity belt with a rear chain.",
              "She wears a chastity belt. It has a chain at the back that keeps it locked on her.",
              "She is locked in a chastity belt, secured at the back with a chain."))
          and side("She wears a steel chastity belt.") == [
              "Mara's steel chastity belt is worn the right way round. The front of the belt sits low on her belly, over her "
              "crotch, and its back sits on her backside, at the small of her back; it turns with her.", twist]
          and side("She wears a chastity belt with a front shield and a chain at the back.")[0]
          == tight.replace("The front of the belt", "The shield at the front of the belt"),
          [side(d) for d in ("She wears chastity belt with tight rear heavy duty security chain going through her ass.",)])
    check("other worn things are pinned by place too, and hair down her back or a tattoo there pins nothing",
          side("She wears a leather collar with a ring at the front.")[0]
          == "Mara's leather collar is worn the right way round. The ring at the front of the collar sits at her throat, and its "
             "back sits at the nape of her neck; it turns with her."
          and "its zip sits between her shoulder blades" in side("She wears a dress with a zip down the back.")[0]
          and side("She has long dark hair down her back.") == [] and side("She wears a shirt. A tattoo on her back.") == []
          and S.side_notes("Mara kneels in a cell with a door at the back.", ["Mara"], gender) == [])
    out, _ = render("A cell.\n\nMara kneels.", character_memory="Mara: a slim woman, chastity belt with tight rear heavy duty "
                                                                "security chain going through her ass.")
    check("info names what is kept the right way round",
          "kept the right way round, front and back pinned to the body: Mara's chastity belt" in out[2], out[2])
    held = "Mara: a slim woman. She wears a steel chastity belt with a chain on the back that keeps it secured to her body."
    pulled = lambda t: S.pull_notes(t, held, ["Mara", "Dan"], gender, {"Mara": [leashed]})
    holds = ("The chain on Mara's steel chastity belt keeps its length when it is pulled: its links are solid metal and never "
             "stretch, it stays snug against her, and the pull moves her by the belt instead.")
    check("pulling that chain or her leash keeps its length and moves her; a chain elsewhere, or one the beat breaks, gets none",
          all(pulled(t) == [holds] for t in ("Dan grabs the chain at the back of her chastity belt and pulls it.",
                                             "Dan yanks her chain.", "Dan pulls her up by the chain."))
          and pulled("Dan pulls her up by the leash.") == ["The chain leash on Mara's collar keeps its length when it is pulled: "
                                                           "its links are solid metal and never stretch, it stays clipped to "
                                                           "the collar, and the pull moves her by the collar instead."]
          and pulled("Dan pulls the chain on the wall.") == [] and pulled("Dan pulls the chain and it snaps.") == [])
    yank = S.plan_shots("A cell.\n\nMara kneels.\n\nDan grabs the chain at the back of her chastity belt and pulls it.", 5.0,
                        [None] * 4, False, memory="Mara: a slim woman. She wears a steel chastity belt with a chain on the back "
                                                  "that keeps it secured to her body.\nDan: a tall man.")
    check("in the shot that pulls the chain, its pull line stands in for the fastened line",
          holds in yank[1]["prompt"] and "stays fastened at both ends" not in yank[1]["prompt"]
          and "stays fastened at both ends" in yank[0]["prompt"], yank[1]["prompt"])
    expect = {
        "Dan cuffs Mara.": {"Mara": ["handcuffs on her wrists"]},
        "Dan cuffs her hands behind her.": {"Mara": ["handcuffs behind her back"]},
        "Dan pulls her arms behind her and handcuffs her wrists.": {"Mara": ["handcuffs behind her back"]},
        "Mara stands with her wrists handcuffed behind her.": {"Mara": ["handcuffs behind her back"]},
        "Mara wears handcuffs behind her back.": {"Mara": ["handcuffs behind her back"]},
        "Dan stands behind her and cuffs her wrists.": {"Mara": ["handcuffs on her wrists"]},
        "Dan snaps the handcuffs on her wrists.": {"Mara": ["handcuffs on her wrists"]},
        "Dan tapes her wrists and ankles together.": {"Mara": ["duct tape around her wrists", "duct tape around her ankles"]},
        "Dan ties her ankles together with rope.": {"Mara": ["rope around her ankles"]},
        "Dan puts a ball gag in her mouth.": {"Mara": ["a ball gag in her mouth"]},
        "Dan gags her.": {"Mara": ["a gag in her mouth"]},
        "Dan blindfolds her.": {"Mara": ["a blindfold over her eyes"]},
        "Dan locks a collar around her neck.": {"Mara": ["a collar around her neck"]},
        "Dan chains her ankles to the bed frame.": {"Mara": ["chains on her ankles to the bed frame"]},
        "Dan zip ties her wrists in front of her.": {"Mara": ["zip ties on her wrists in front of her"]},
        "Mara lies on the bed, her wrists cuffed behind her back.": {"Mara": ["handcuffs behind her back"]},
        "Mara sits with duct tape over her mouth.": {"Mara": ["duct tape over her mouth"]},
        "He cuffs her.": {"Mara": ["handcuffs on her wrists"]},
        "Dan grabs Mara and cuffs her.": {"Mara": ["handcuffs on her wrists"]},
        "Dan turns Mara around and handcuffs her wrists behind her back.": {"Mara": ["handcuffs behind her back"]},
        "Dan restrains Mara, cuffing her wrists behind her back.": {"Mara": ["handcuffs behind her back"]},
        "Dan grabs Mara's wrists and cuffs them behind her back.": {"Mara": ["handcuffs behind her back"]},
        "Dan locks her wrists in handcuffs.": {"Mara": ["handcuffs on her wrists"]},
        "Mara sits in handcuffs.": {"Mara": ["handcuffs on her wrists"]},
        "The cuffs click shut around her wrists.": {"Mara": ["handcuffs on her wrists"]},
        "Handcuffs are snapped onto Mara's wrists.": {"Mara": ["handcuffs on her wrists"]},
        "Dan takes out the handcuffs and snaps them onto her wrists.": {"Mara": ["handcuffs on her wrists"]},
        "Dan uses handcuffs to lock her wrists behind her back.": {"Mara": ["handcuffs behind her back"]},
        "Dan takes duct tape and wraps it around her mouth.": {"Mara": ["duct tape over her mouth"]},
        "Dan takes out a roll of duct tape and presses a strip over her mouth.": {"Mara": ["duct tape over her mouth"]},
        "Dan comes back with rope and ties her ankles.": {"Mara": ["rope around her ankles"]},
    }
    got = {t: read(t) for t in expect}
    check("common restraint wording is read onto the right person", got == expect,
          {t: g for t, g in got.items() if g != expect[t]})
    check("whoever does the restraining is never the one restrained",
          all("Dan" not in read(t) for t in expect))
    quiet = ["Dan ties his shoes.", "Dan tapes the box shut.", "Mara gags at the smell.", "Dan grabs her by the collar.",
             "Dan walks in with handcuffs.", "Dan slides the key into the handcuffs.",
             "Dan unlocks the cuffs and slips them into his pocket."]
    check("wording that only sounds like a restraint is left alone", all(read(t) == {} for t in quiet),
          {t: read(t) for t in quiet})
    belt = S.rst.read("Dan carries handcuffs on his belt.", people, gender, {})
    check("cuffs it mentions but cannot place are listed as not read", belt[0] == {} and "were not read" in belt[2][0], belt)
    worn = S.plan_shots("A cell. Mara and Dan wait.\n\nMara kneels on the floor, her wrists cuffed behind her back.\n\n"
                        "Dan watches her.", 10.0, [None] * 4, False)
    check("cuffs a beat describes as already on are held for that whole shot, not put on during it",
          worn[0]["held"] == "Mara: handcuffs behind her back." and not worn[0]["added"]
          and worn[0]["bound"]["Mara"]["latch_limbs"] == () and worn[1]["held"] == "Mara: handcuffs behind her back.",
          [S.shot_line(x) for x in worn])
    held = {"Mara": ["duct tape over her mouth", "duct tape around her wrists", "handcuffs behind her back"]}
    off = S.rst.read("Dan removes the tape from her mouth.", people, gender, held)[1]
    state, _, _ = S.apply_holds(held, {"release": [(w, i) for w, i in off.items()], "hold": []})
    check("taking the tape off her mouth leaves the tape on her wrists",
          state["Mara"] == ["duct tape around her wrists", "handcuffs behind her back"], state)
    three = S.rst.read("Dan cuffs her.", ["Mara", "Ana", "Dan"], {"Mara": "f", "Ana": "f", "Dan": "m"}, {})
    check("when it cannot tell who, it says so instead of guessing", three[0] == {} and "who wears the handcuffs" in three[2][0],
          three)
    told = S.plan_shots("A room. Mara sits. Dan stands.\n\nDan cuffs her.\nhold: Mara, handcuffs in front of her\n\nMara waits.",
                        5.0, [None] * 4, False)
    check("a hold: line overrides what was read for that person", told[1]["held"] == "Mara: handcuffs in front of her.",
          told[1]["held"])
    check("genders come only from descriptions and reflexives, not from a pronoun in the next sentence",
          S.rst.genders(["Mara", "Dan"], ["Mara is a tall woman. Dan stands beside her.", "Dan grabs the keys. He leaves."])
          == {"Mara": "f"} and S.rst.genders(["Dan"], ["Dan steadies himself."]) == {"Dan": "m"})
    crystal = ("A dim room. Crystal lies on the bed. Max stands beside her.\n\nMax handcuffs her wrists behind her back.\n\n"
               "Max wraps duct tape around Crystal's mouth, his hands pressing it flat.")
    last = S.plan_shots(crystal, 10.0, [None] * 4, False)[-1]
    check("her tape is over her mouth: the possessive follows her own cuffs, not the hands of the man taping it",
          last["added"] == "Crystal: duct tape over her mouth." and last["held"] == "Crystal: handcuffs behind her back.",
          S.shot_line(last))
    bare = S.plan_shots("A room. Crystal lies on the bed.\n\nMax cuffs Crystal.\n\nMax tapes Crystal's mouth shut.",
                        10.0, [None] * 4, False)[-1]
    check("with no pronoun anywhere the item uses her name, never a guess",
          bare["added"] == "Crystal: duct tape over Crystal's mouth." and bare["held"] == "Crystal: handcuffs on Crystal's wrists.",
          S.shot_line(bare))


def test_wardrobe():
    print("\n=== clothing coming off and going back on ===")
    script = ("A bedroom.\n\nMax takes off her jacket.\n\nCrystal sits on the bed.\n\n"
              "Crystal puts her jacket back on.\n\nCrystal waits.")
    memory = "Crystal: a slim woman in a red top and a black jacket.\nMax: a tall man in a grey suit."
    shots = S.plan_shots(script, 10.0, [None] * 4, False, memory=memory)
    line = lambda k: next(ln for ln in shots[k]["prompt"].splitlines() if ln.startswith("Crystal:"))
    check("a garment taken off leaves her description from the next shot, and comes back after she puts it on",
          [line(k) for k in range(4)] == ["Crystal: a slim woman in a red top and a black jacket.",
                                          "Crystal: a slim woman in a red top.", "Crystal: a slim woman in a red top.",
                                          "Crystal: a slim woman in a red top and a black jacket."]
          and shots[0]["clothes_off"] == "Crystal: jacket." and shots[2]["clothes_on"] == "Crystal: jacket.",
          [line(k) for k in range(4)])
    people, gender = ["Crystal", "Max"], {"Crystal": "f"}
    desc = {"Crystal": "Crystal: a slim woman in a red top and a black thong."}
    read = lambda t: S.wrd.read(t, people, gender, desc)
    check("underwear such as a thong is a garment, read off whoever it is on",
          read("Max takes off her thong.")[0] == {"Crystal": ["thong"]}
          and read("Crystal pulls her thong down her legs.")[0] == {"Crystal": ["thong"]}
          and read("Crystal takes the thong off.")[0] == {"Crystal": ["thong"]}
          and read("Max removes Crystal's bra.")[0] == {"Crystal": ["bra"]})
    check("touching a garment is not taking it off", read("Max pulls her shirt.") == ({}, {}, []))
    check("whose garment, when it cannot tell, is reported",
          S.wrd.read("He takes off the hat.", people, {}, {})[2] == ["whose hat in 'He takes off the hat.'"])
    told = S.plan_shots("A bedroom.\n\nCrystal turns.\nremove: Crystal, black jacket\n\nCrystal sits."
                        "\nwear: Crystal, black jacket\n\nCrystal stands.", 10.0, [None] * 4, False, memory=memory)
    check("remove: and wear: lines take a garment off and put it back",
          "black jacket" not in told[1]["prompt"] and "black jacket" in told[2]["prompt"], [t["prompt"][:120] for t in told])
    strip = S.wrd.undress
    check("the description reads cleanly after a garment comes off",
          strip("Crystal lies on the bed. She wears only a black lace thong.", {"Crystal": ["thong"]}, people)
          == "Crystal lies on the bed."
          and strip("Crystal wears a black thong, a red top and boots.", {"Crystal": ["thong", "boots"]}, people)
          == "Crystal wears a red top."
          and strip("Crystal wears a jacket over her dress.", {"Crystal": ["jacket"]}, people) == "Crystal wears her dress."
          and strip("Max: a tall man in a grey suit. Crystal wears a red dress.", {"Crystal": ["dress"]}, people)
          == "Max: a tall man in a grey suit.")


    cases = {"Crystal wears a black latex thong.": "",
             "Crystal wears a thong made of shiny black latex.": "",
             "Crystal wears a thong of red latex with lace trim and a white bra.": "Crystal wears a white bra.",
             "Crystal wears black latex thong underwear.": "",
             "Crystal wears a black-latex thong, cut high on her hips.": "",
             "Crystal wears a latex thong and a white bra.": "Crystal wears a white bra.",
             "Crystal wears a red thong with a matching bra.": "Crystal wears a matching bra."}
    got = {d: S.wrd.strip(d, "thong") for d in cases}
    check("a thong that comes off takes its whole description with it, latex and all",
          got == cases, {d: g for d, g in got.items() if g != cases[d]})
    left = S.wrd.undress("Crystal wears a thong. It is black latex. She has long red hair.", {"Crystal": ["thong"]}, ["Crystal"])
    check("a sentence that only describes the removed thong goes with it", left == "She has long red hair.", left)

def test_undressed():
    print("\n=== clothes that come off stay off, and he stays inside her ===")
    people, gender = ["Sarah", "Dan"], {"Sarah": "f", "Dan": "m"}
    desc = {"Sarah": "Sarah: a woman wearing a black PVC crop top and a black PVC miniskirt."}
    read = lambda t: S.wrd.read(t, people, gender, desc)[0]
    check("a skirt comes off however it is written, typos and all, and a miniskirt is the same skirt",
          all(read(t) == {"Sarah": ["skirt"]} for t in (
              "Sarah slides off her skirt.", "Sarah slides her skirt off.", "Sarah pulls down her skirt.",
              "Sarah slides of her skirt.", "Sarah takes off her miniskirt.", "Sarah removes her skirt.",
              "Sarah steps out of her skirt.", "Sarah lets her skirt fall to the floor.", "Sarah drops her skirt.",
              "Dan slips her mini skirt off.")),
          {t: read(t) for t in ("Sarah pulls down her skirt.", "Sarah slides of her skirt.", "Sarah steps out of her skirt.")})
    check("stripping naked, being undressed or taking off all her clothes takes off everything she wears, and touching, "
          "lifting or not being naked takes off nothing",
          read("Sarah strips naked.") == read("Dan undresses her.") == read("Sarah takes off all her clothes.")
          == {"Sarah": ["crop top", "skirt"]} and read("Sarah is naked from the waist down.") == {"Sarah": ["skirt"]}
          and read("Sarah lifts her skirt.") == read("Dan pulls her skirt.") == read("Sarah is not naked.") == {})
    check("a skirt that comes off takes its whole description with it, PVC and all, and a crop top goes with a top",
          S.wrd.undress(desc["Sarah"], {"Sarah": ["skirt"]}, people) == "Sarah: a woman wearing a black PVC crop top."
          and S.wrd.undress(desc["Sarah"], {"Sarah": ["top"]}, people) == "Sarah: a woman wearing a black PVC miniskirt.")
    room = {"Sarah is a young woman in a white diaper, standing by the door.": "Sarah is a young woman standing by the door.",
            "Sarah, in a diaper with the toilet behind her against the wall, waits by the door.":
            "Sarah, with the toilet behind her against the wall, waits by the door.",
            "Sarah, in a diaper in the bathroom, waits.": "Sarah, in the bathroom, waits.",
            "Sarah wears a diaper, pulled up tight, and stands at the sink.": "Sarah stands at the sink.",
            "Sarah: a woman with a diaper and long hair.": "Sarah: a woman with long hair.",
            "Sarah wears a diaper with cartoon prints on the front.": "", "Sarah walks in.": "Sarah walks in."}
    got = {t: S.wrd.undress(t, {"Sarah": ["diaper"]}, people) for t in room}
    check("a garment that comes off takes only its own words: the room, where she is and what she does stay, and no "
          "stray 'in' or 'wearing' is left behind",
          got == room and S.wrd.undress("Sarah wears a red shirt and a blue yukata.", {"Sarah": ["shirt"]}, people)
          == "Sarah wears a blue yukata.", {t: g for t, g in got.items() if g != room[t]})
    worn = {"Sarah": "Sarah: a woman wearing a thick white diaper and a pink t-shirt.", "Dan": "Dan: a man wearing a blue yukata."}
    took = lambda t: S.wrd.read(t, people, gender, worn)[0]
    check("a diaper, nappy, pull-ups or towel comes off like any garment, a nappy is a diaper, and anything a description says "
          "they are wearing can come off too",
          all(took(t) == {"Sarah": ["diaper"]} for t in ("Sarah takes off her diaper.", "Dan removes her diaper.",
                                                         "Dan untapes her diaper.", "Sarah pulls her diaper down.",
                                                         "Sarah steps out of her nappy.", "Dan takes off her nappies."))
          and took("Sarah slides off her pull-ups.") == {"Sarah": ["pull-ups"]} and took("Sarah drops her towel.") == {"Sarah": ["towel"]}
          and took("Dan takes off his yukata.") == {"Dan": ["yukata"]} and took("Sarah takes off her smile.") == {}
          and S.wrd.undress(worn["Sarah"], {"Sarah": ["diaper"]}, people) == "Sarah: a woman wearing a pink t-shirt."
          and S.wrd.bare(worn["Sarah"], ["diaper"]) == ["lower"],
          {t: took(t) for t in ("Dan untapes her diaper.", "Sarah steps out of her nappy.", "Dan takes off his yukata.")})
    puts = lambda t: S.clothes(S.parse_paragraph(t), people, gender, worn, {"Sarah": ["diaper"]}, people)[2]
    check("spoken words never put clothes on or take them off, and putting it on the chair or the bed is not putting it on",
          all(puts(t) == {} for t in ('"Put your diaper back on," Dan says.', '"I am not putting my diaper back on!" Sarah shouts.',
                                      "Sarah puts her diaper on the chair.", "Sarah slips her diaper on the floor.",
                                      "Dan pulls her diaper on the bed.", "Sarah gets her diaper on Dan's lap."))
          and all(puts(t) == {"Sarah": ["diaper"]} for t in ("Sarah puts her diaper back on.", "Dan puts a fresh diaper on her.",
                                                             "Sarah puts on her diaper.", "Sarah slides her diaper on and smiles."))
          and S.clothes(S.parse_paragraph('"Strip naked," Dan orders.'), people, gender, worn, {}, people)[1] == {},
          {t: puts(t) for t in ('"Put your diaper back on," Dan says.', "Sarah puts her diaper on the chair.")})
    bare = lambda gone, d=desc["Sarah"]: S.wrd.bare(d, gone)
    check("she is bare below once nothing covers her there, above once nothing covers her there, and a dress covers both",
          bare(["skirt"]) == ["lower"] and bare(["top"]) == ["upper"] and bare(["skirt", "top"]) == ["lower", "upper"]
          and bare(["skirt"], "Sarah wears a skirt over black panties.") == []
          and bare(["dress"], "Sarah wears a red dress.") == ["lower", "upper"] and bare([]) == [])
    mem = "Sarah <Picture 1>: a woman with long blonde hair, wearing a black PVC crop top and a black PVC miniskirt.\n" \
          "Dan: a man with short dark hair."
    plan = lambda *beats: S.plan_shots("A dim living room with a leather couch.\n\n" + "\n\n".join(beats), 5.0,
                                       [IMG1, None, None, None], False, memory=mem)
    notes = lambda x: x["prompt"].split("\n\n")[-1]
    skirt = plan("Dan sits on the couch.", "Sarah stands up and slides off her skirt.", "Sarah turns around.",
                 "Sarah grinds against him.", "Sarah puts her skirt back on.", "Sarah smiles.")
    check("from the shot it comes off in, she stays naked from the waist down and her description loses the skirt, while "
          "her picture stays in every shot, for her face and hair only, until she puts it back on",
          "From the moment her skirt comes off, Sarah stays naked from the waist down" in notes(skirt[1])
          and all("Sarah is naked from the waist down the whole shot" in notes(x) and "miniskirt" not in x["prompt"]
                  and x["refs"] and "<Picture 1> is for Sarah's face and hair only: her clothes are as this shot describes "
                  "them." in notes(x) for x in skirt[2:4])
          and "naked" not in notes(skirt[4]) and "miniskirt" in skirt[5]["prompt"] and skirt[5]["refs"],
          [S.shot_line(x) for x in skirt])
    said = plan("Sarah stands by the bed.", "Sarah takes off her skirt.", '"I hate that skirt," Sarah says.',
                "Sarah looks at her skirt on the floor.", "Sarah smiles.")
    check("every shot gives only her face and hair from her picture, so her clothes come from the words alone; once the "
          "skirt is off the shot names it no more than the beat does, and lists no body part",
          all(notes(x).count("skirt") == 0 and "Sarah is naked from the waist down the whole shot." in notes(x)
              and not re.search(r"\b(?:bottom|butt|crotch|hips|breasts)\b", notes(x)) for x in said[2:])
          and [x["faces"] for x in said] == [["Sarah"]] * 5
          and all(x["refs"] and "face and hair" in S.shot_line(x) for x in said)
          and skirt[5]["faces"] == ["Sarah"] and "clothes on" not in S.shot_line(said[2]),
          [(S.shot_line(x), notes(x)) for x in said])
    check("a garment she took off and then names in her line is kept out of the picture's text, the real line kept for "
          "the sound, and narration that names it is left as it is",
          "<d>" not in said[2]["unspoken"] and "skirt" not in said[2]["unspoken"].split("\n\n")[2]
          and "<d>I hate that skirt,</d>" in said[2]["prompt"]
          and said[2]["unsaid"] == ["skirt"] and "spoken skirt kept out of the picture" in S.shot_line(said[2])
          and said[3]["unspoken"] == "" and not said[1]["unsaid"], [(x["unsaid"], x["unspoken"][:120]) for x in said])
    line = ('<d>Look at me. No diapers, panties, pants or underwear for me, just my bra and my dress. Mom can put a '
            'new diaper on me.</d> She points at her shoes.')
    out = S.wrd.unsay(line, ["diaper"], {"lower"})
    check("with her bare from the waist down, a sentence in her line that names anything worn there or over everything, "
          "or the garment she took off, is said in the picture as her own words before it, the same length, so nothing in "
          "it is acted out; a line that is only about it is left out of the picture",
          out == ("<d>Look at me. Look at me Look at me Look at me Look at me Look at. Look at me Look at me Look "
                  "at.</d> She points at her shoes.", ["diaper", "panties", "pants", "underwear", "dress"])
          and S.wrd.unsay(line, [], {"upper"})[1] == ["bra", "dress"] and S.wrd.unsay(line, [], set()) == (line, [])
          and S.wrd.unsay("<d>Where are my Skirts?</d> She looks.", [], {"lower"})[0] == " She looks.", out)
    out, calls = render('A bedroom.\n\nSarah takes off her skirt.\n\n"Look at me. I hate that skirt," Sarah says.',
                        ref_image_1=IMG1, character_memory=mem)
    built = [("<d>Look at me. Look at me Look.</d>" in c["prompt"], "skirt,</d>" in c["prompt"]) for c in calls["cond"]]
    sounds = [c for c in calls["sample"] if c["foley"] and c["shot"] == 2]
    check("that shot's picture is made from the line without the word, then its sound from the real line over the "
          "finished picture",
          built == [(False, False), (True, False), (False, True)] and len(sounds) == 1
          and sounds[0]["mask"] == [0.0, 1.0] and "the line's sound made over the finished picture" in out[2],
          (built, [(c["shot"], c["mask"]) for c in sounds]))
    still = "This shot is only the line being spoken: everyone stays where and as they are."
    talks = [S.talk_only(t) for t in ('Dana says to Sarah: "OK, take off your leggings."', '"Lift up your butt," Dana says.',
                                      'She whispers to him: "Lie down."', 'Dana says to Sarah: "Lift up." Sarah lifts up.',
                                      'Dana turns to Sarah and says: "Lie down."', 'Sarah lies down.')]
    check("a beat that is only someone talking tells H3 everyone stays where and as they are, and a beat with anything "
          "else in it keeps its actions",
          talks == [True, True, True, False, False, False]
          and still in S.plan_shots('A room.\n\nDana says to Sarah: "OK, take off your leggings."', 5.0, [None] * 4,
                                    False)[0]["prompt"], talks)
    heard = "Spoken words are only heard: the shot shows what it describes, not what anyone talks about."
    mixed = S.plan_shots('A room.\n\nSarah picks up her skirt and says: "I hate this."', 5.0, [None] * 4, False)[0]["prompt"]
    check("a spoken line with actions in its beat tells H3 the words are only heard, not shown; a beat that is only the "
          "line says everyone stays as they are; a shot without a line says neither",
          heard in mixed and still not in mixed and still in said[2]["prompt"] and still in said[2]["unspoken"]
          and heard not in said[2]["prompt"] and not any(n in said[1]["prompt"] + said[3]["prompt"] for n in (heard, still)),
          [notes(x) for x in said[1:4]] + [mixed])
    brand = ("Sarah: she, <Picture 1>, skin-tight shiny white spandex PVC crop top, skin-tight shiny rubber white Pampers "
             "diaper with Pampers logo.")
    film = S.plan_shots("A bathroom.\n\nSarah takes off her diaper.\n\nSarah sits down.", 5.0, [IMG1, None, None, None],
                        False, memory=brand)
    check("a brand goes with the garment it names, logo and all, the garment's words stop at a comma so the crop top "
          "stays, and Pampers or Huggies come off as a diaper",
          S.wrd.undress(brand, {"Sarah": ["diaper"]}, people)
          == "Sarah: she, <Picture 1>, skin-tight shiny white spandex PVC crop top."
          and S.wrd.undress("Sarah: a crop top, white diaper.", {"Sarah": ["diaper"]}, people) == "Sarah: a crop top."
          and S.wrd.undress("Sarah wears Huggies.", {"Sarah": ["diaper"]}, people) == ""
          and S.wrd.read("Sarah takes off her Pampers.", people, gender, {"Sarah": brand})[0] == {"Sarah": ["diaper"]}
          and "Pampers" not in film[1]["prompt"] and "crop top" in film[1]["prompt"],
          (S.wrd.undress(brand, {"Sarah": ["diaper"]}, people), film[1]["prompt"][:300]))
    layers = ("Sarah: she, <Picture 1>, white crop top, thick white diaper, black leggings with a metal zipper that runs "
              "between her legs from front to back.\nDana: She, blue jeans.")
    film = S.plan_shots("A bathroom.\n\nSarah and Dana walk in.\n\nSarah unzips her leggings and slides them down her legs."
                        "\n\nDana opens the bag next to Sarah and takes out a new Pampers diaper.\n\nDana removes Sarah's diaper."
                        "\n\nDana fastens the new diaper around Sarah's hips. The diaper is now secured on Sarah's body."
                        "\n\nSarah smiles.", 5.0, [IMG1, None, None, None], False, memory=layers)
    line = lambda x: next((ln for ln in x["prompt"].split("\n") if ln.startswith("Sarah:")), "")
    check("what is worn under something else stays out of sight until that comes off, the shot it comes off in says it "
          "covers it, a garment's own details go with it, a brand is not a stranger, and fastening a diaper on her or saying "
          "it is secured on her puts it back on",
          "diaper" not in line(film[0]) and "leggings" in line(film[0]) and "diaper" in line(film[1])
          and "leggings" in line(film[1]) and "Her leggings cover her diaper until they come off." in film[1]["prompt"]
          and line(film[2]).endswith("white crop top, thick white diaper.") and "looks nothing like" not in film[2]["prompt"]
          and "cast: Sarah, Dana" in S.shot_line(film[3])
          and "clothes on: Sarah: diaper." in S.shot_line(film[4]) and "naked" not in notes(film[4])
          and "diaper" in line(film[5]) and S.strangers("Dana takes out a new Pampers diaper and Nike shoes.", "", ["Dana"]) == []
          and S.strangers("Dana waves at Marco.", "", ["Dana"]) == ["Marco"]
          and all(S.wrd.read(t, people, gender, worn)[1] == {"Sarah": ["diaper"]} for t in (
              "Dan fastens a new diaper around her hips.", "Dan tapes the diaper shut.",
              "The diaper is now secured on Sarah's body."))
          and all(S.wrd.read(t, people, gender, worn)[1] == {} for t in ("Her diaper is on her bed.",
                                                                       "Sarah wraps her towel around the pole.")),
          [(S.shot_line(x), line(x)) for x in film])
    bow = S.plan_shots("A bedroom.\n\nDana puts a pink bow in Sarah's hair.\n\nDana walks to the window and opens it.\n\n"
                       "Sarah smiles.\n\nDana takes the bow out of Sarah's hair.\n\nSarah waits.", 5.0,
                       [IMG1, None, None, None], False, memory="Sarah <Picture 1>: a woman in a blue dress.\nDana: a woman.")
    worn_bow = "Sarah wears a pink bow in her hair."
    two = ["Sarah", "Dana"], {"Sarah": "f", "Dana": "f"}
    dress = {"Sarah": "Sarah: a woman in a blue dress."}
    reads = {}
    for t in ("Dana puts a bib on her.", "Sarah puts the glasses on.", "Sarah puts on a red hat.",
              "Dana clips a silver necklace around Sarah's neck.", "Dana puts her hand on Sarah's shoulder.",
              "Dana puts the cup on the table.", "Sarah puts her dress back on.", "Dana puts the wipe on Sarah's belly."):
        reads[t] = S.wrd.additions(t, *two, dress, {}, two[0])[0].get("Sarah", [])
    check("anything put on someone that their description does not have stays on them in every shot after, in view or not, "
          "until a beat takes it off; hands, cups, wipes and their own clothes are not additions",
          [worn_bow in x["prompt"] for x in bow] == [False, True, True, False, False]
          and reads == {"Dana puts a bib on her.": ["a bib"], "Sarah puts the glasses on.": ["glasses"],
                        "Sarah puts on a red hat.": ["a red hat"],
                        "Dana clips a silver necklace around Sarah's neck.": ["a silver necklace around her neck"],
                        "Dana puts her hand on Sarah's shoulder.": [], "Dana puts the cup on the table.": [],
                        "Sarah puts her dress back on.": [], "Dana puts the wipe on Sarah's belly.": []}
          and S.wrd.additions("Sarah takes off her hat.", *two, dress, {"Sarah": ["a red hat"]}, two[0])[0] == {},
          ([worn_bow in x["prompt"] for x in bow], reads))
    crop, was = IMG1[:, :8, :8, :].clone(), S.face_crop
    try:
        for found in (crop, None):
            S.face_crop = lambda det, pic: found
            out, calls = render("A bedroom.\n\nSarah stands by the bed.\n\nSarah takes off her skirt.\n\nSarah smiles.",
                                ref_image_1=IMG1, character_memory=mem, dwpose=True)
            sizes = [tuple(c["refs"][0].shape[1:3]) if c["refs"] else None for c in calls["cond"]]
            full = tuple(IMG1.shape[1:3])
            check("the picture the model gets is cropped to her face and hair in every shot"
                  if found is not None else "where no face is found, the whole picture goes in and info says so",
                  sizes == ([(8, 8)] * 3 if found is not None else [full] * 3)
                  and (found is not None) != ("Sarah: whole picture used, no face found to crop it to" in out[2]),
                  (sizes, out[2]))
    finally:
        S.face_crop = was
    box = S.pose.head_box([[50, 40, 1], [50, 60, 1]] + [[0, 0, 0]] * 12 + [[46, 38, 1], [54, 38, 1]], 100, 200)
    check("the face crop runs from above her hair to her chin, short of her shoulders, and is squared on her head",
          box is not None and box[1] < 30 and 50 < box[3] < 60 and abs(box[2] - box[0] - box[3] + box[1]) <= 1
          and S.pose.head_box([[50, 40, 1]] + [[0, 0, 0]] * 17, 100, 200) is None, box)
    back = plan("Dan sits on the couch.", "Sarah steps out of her skirt.", "Dan pulls out his phone.",
                "Sarah walks over, climbs onto his lap and rides him.")
    check("when she comes back into view after it is off, her picture is for her face and hair only",
          "Sarah (face and hair as in <Picture 1>) comes into view" in back[3]["prompt"], notes(back[3]))
    naked = plan("Sarah kneels on the bed, completely naked.", "Dan kneels behind her.")
    check("naked as she is in the beat holds the whole shot, from that shot on",
          all("Sarah is naked the whole shot." in notes(x)
              and "crop top" not in x["prompt"] for x in naked), [notes(x) for x in naked])
    sex = plan("Dan sits on the couch.", "Sarah slides off her skirt.",
               "Sarah turns around and sits down on Dan's waist, her legs spread, allowing his penis to enter her vagina.",
               "Sarah grinds on him.", "Dan pulls out halfway and slams back in.", "Dan grabs her hips.", "Sarah climbs off him.",
               "Sarah smiles.")
    held = ("Dan's penis is buried deep inside Sarah's vagina the whole shot: her crotch and his groin stay pressed tight "
            "together, so only the base of his penis shows, and their hips rock and grind together as they move.")
    check("once he enters her his penis stays inside her in every shot of them until she climbs off, and pulling out "
          "and going back in is no end",
          "Once Dan's penis is inside Sarah's vagina, it stays buried deep inside her for the rest of the shot: her crotch "
          "and his groin stay pressed tight together" in notes(sex[2]) and "sliding" not in notes(sex[2]) + notes(sex[3])
          and all(held in notes(x) for x in sex[3:6]) and not any("penis" in notes(x) for x in sex[6:])
          and sex[3]["inside"] == [("Dan", "Sarah")] and "inside: Dan in Sarah" in S.shot_line(sex[3]),
          [S.shot_line(x) for x in sex])
    starts = lambda b: [x["inside"] for x in plan("Dan sits on the couch.", "Sarah kneels in front of him.", b)][-1]
    check("riding him, sinking onto his cock, being fucked or taken from behind, sex and anal all put him inside her; his "
          "cock in her mouth or hand, or entering a city, does not",
          all(starts(b) == [("Dan", "Sarah")] for b in (
              "Sarah climbs onto his lap and rides him.", "Sarah sinks onto Dan's cock.", "Dan fucks her.",
              "Dan takes her from behind.", "Sarah has sex with Dan.", "Dan slides his cock into her ass.",
              "Sarah rides Dan's cock."))
          and all(starts(b) == [] for b in ("Sarah takes his cock into her mouth.", "Sarah strokes his cock in her hand.",
                                            "Sarah enters Paris."))
          and "inside Sarah's anus the whole shot: her bottom and his groin stay pressed tight together" in notes(
              plan("Sarah kneels on the bed.", "Dan slides his cock into her ass.", "Dan fucks her harder.")[2]),
          {b: starts(b) for b in ("Sarah sinks onto Dan's cock.", "Dan takes her from behind.")})
    gone = plan("Dan sits on the couch.", "Sarah rides him.", "Dan stands up.", "Sarah rides him again.")
    check("standing up ends it, and riding him again starts it again",
          [bool(x["inside"]) for x in gone] == [False, True, False, True]
          and "Once Dan's penis is inside" in notes(gone[3]), [S.shot_line(x) for x in gone])


GARMENTS = ("t-shirt", "shirt", "blouse", "crop top", "tank top", "top", "sweater", "jumper", "hoodie", "cardigan", "jacket", "coat",
            "blazer", "vest", "dress", "minidress", "skirt", "miniskirt", "gown", "shorts", "trousers", "pants", "hotpants",
            "sweatpants", "jeans", "leggings", "tights", "stockings", "socks", "shoes", "boots", "heels", "sneakers", "gloves", "hat",
            "cap", "scarf", "belt", "robe", "bathrobe", "pajamas", "pyjamas", "nightgown", "nightie", "lingerie", "bra", "bralette",
            "panties", "knickers", "underwear", "briefs", "boxers", "thong", "g-string", "bikini top", "bikini bottoms", "bikini",
            "swimsuit", "corset", "bodysuit", "catsuit", "jumpsuit", "playsuit", "romper", "leotard", "camisole", "negligee",
            "chemise", "babydoll", "diaper", "nappy", "pull-ups", "bloomers", "plastic pants", "underpants", "pantyhose", "fishnets",
            "onesie", "towel", "nightdress", "kimono", "sarong", "bodystocking", "slippers", "sandals", "mittens", "uniform", "apron",
            "bottoms", "undies", "slacks", "joggers", "trunks", "capris", "culottes", "girdle", "speedos", "jockstrap", "overalls",
            "dungarees")
LOWER = {"skirt", "miniskirt", "shorts", "trousers", "pants", "hotpants", "sweatpants", "jeans", "leggings", "tights", "panties",
         "knickers", "underwear", "briefs", "boxers", "thong", "g-string", "bikini bottoms", "diaper", "nappy", "pull-ups",
         "bloomers", "plastic pants", "underpants", "pantyhose", "bottoms", "undies", "slacks", "joggers", "trunks", "capris",
         "culottes", "girdle", "speedos", "jockstrap"}
UPPER = {"t-shirt", "shirt", "blouse", "crop top", "tank top", "top", "sweater", "jumper", "hoodie", "cardigan", "jacket", "coat",
         "blazer", "vest", "bra", "bralette", "camisole", "corset", "bikini top"}
WHOLE = {"dress", "minidress", "gown", "robe", "bathrobe", "pajamas", "pyjamas", "nightgown", "nightie", "lingerie", "bikini",
         "swimsuit", "bodysuit", "catsuit", "jumpsuit", "playsuit", "romper", "leotard", "negligee", "chemise", "babydoll",
         "uniform", "onesie", "towel", "nightdress", "kimono", "sarong", "bodystocking", "overalls", "dungarees"}


def test_every_garment():
    print("\n=== every garment comes off, stays off, and goes back on the same way ===")
    people, gender = ["Sarah", "Dan"], {"Sarah": "f", "Dan": "m"}
    off = ("Sarah takes off her {g}.", "Sarah takes her {g} off.", "Sarah removes her {g}.", "Sarah pulls off her {g}.",
           "Her {g} falls to the floor.", "Sarah's {g} comes off.", "Sarah touches her {g}. She takes it off.",
           "Sarah touches her {g} and wriggles out of it.", "Sarah pushes her {g} down to her ankles.",
           "Sarah gets her {g} off.",
           "Sarah slips out of her {g}.", "Dan pulls her {g} off.", "Sarah peels off her {g}.", "Sarah strips off her {g}.",
           "Dan rips her {g} off.", "Sarah wriggles out of her {g}.", "Sarah sheds her {g}.", "Sarah slides her {g} off.",
           "Dan takes Sarah's {g} off.", "Sarah drops her {g} to the floor.", "Sarah lets her {g} fall to the floor.",
           "Sarah quickly takes off her {g}.", "She takes off her red {g}.", "Sarah pulls her {g} over her head.",
           "Sarah gets out of her {g}.", "Dan lifts her {g} off.", "Sarah kicks off her {g}.", "Dan yanks down her {g}.")
    on = ("Sarah puts her {g} back on.", "Sarah puts on her {g}.", "Sarah pulls her {g} on.", "Sarah slips into her {g}.",
          "Dan puts her {g} back on her.", "Sarah steps into her {g}.", "Sarah gets back into her {g}.", "Sarah gets dressed.",
          "Sarah puts her clothes back on.")
    never = ('"Take off your {g}," Dan says.', '"Put your {g} back on," Dan says.', "Sarah adjusts her {g}.",
             "Sarah tugs at her {g}.", "Sarah smooths her {g}.", "Sarah lifts her {g}.", "Sarah puts her {g} on the bed.",
             "Sarah looks at her {g}.", "Sarah folds her {g}.", "Sarah unbuttons her {g}.", "Sarah pulls her {g} up.",
             "Sarah hangs her {g} on the door.", "Sarah buttons her {g}.", "Sarah throws her {g} at Dan.",
             "Sarah holds her {g}. She puts it on the bed.", "Her {g} falls just above her knees.", "Her {g} sags.",
             "Sarah touches her {g} and Dan pulls her close, taking it slow.", "Her {g} doesn't come off.")
    wrong = {}
    for g in GARMENTS:
        want = S.wrd.garment_of(g)
        desc = {"Sarah": f"Sarah: a woman wearing red {g} and a silver necklace.", "Dan": f"Dan: a man wearing red {g}."}
        read = lambda t: S.clothes(S.parse_paragraph(t.format(g=g)), people, gender, desc, {"Sarah": [want]}, people)
        bad = [t for t in off if read(t)[1] != {"Sarah": [want]}] + [t for t in on if read(t)[0].get("Sarah")]
        bad += [t for t in never if read(t)[1] or read(t)[2]]
        bad += ["Dan takes off his {g}."] if read("Dan takes off his {g}.")[1] != {"Dan": [want]} else []
        left = S.wrd.undress(desc["Sarah"], {"Sarah": [want]}, people)
        bad += ["strip"] if S.wrd.mentions(left, want) or "necklace" not in left else []
        zone = ["lower"] if g in LOWER else ["upper"] if g in UPPER else ["lower", "upper"] if g in WHOLE else []
        bad += ["zone"] if S.wrd.bare(desc["Sarah"], [want]) != zone else []
        said = lambda gone, zones: S.wrd.unsay(f"<d>Hello there. Where is my {g}?</d>", gone, zones)[0]
        hit = lambda x: x.startswith("<d>Hello there. Hello there Hello") and not S.wrd.mentions(x, want)
        bad += ["unsay"] if not hit(said([want], ())) or zone and not hit(said([], zone)) else []
        film = S.plan_shots(f"A bedroom.\n\nSarah stands by the bed. Dan sits on it.\n\nSarah takes off her {g}.\n\nSarah "
                            f"smiles at Dan.\n\nDan kisses her.\n\nSarah puts her {g} back on.\n\nSarah waits.", 5.0,
                            [IMG1, None, None, None], False, memory=f"Sarah <Picture 1>: a woman wearing red {g}.\nDan: a man.")
        line = lambda x: next((ln for ln in x["prompt"].splitlines() if ln.startswith("Sarah")), "")
        bad += ["film"] if any(S.wrd.mentions(line(x), want) or not x["refs"] or "face and hair only" not in x["prompt"]
                               or zone and "naked" not in x["prompt"]
                               for x in film[2:4]) or not S.wrd.mentions(line(film[5]), want) else []
        if bad:
            wrong[g] = bad
    check(f"all {len(GARMENTS)} garments: every way of taking one off is read, every way of putting it back on, nothing said "
          "aloud or done to it in passing counts, it leaves the description whole, bares its own part of the body, and stays "
          "off, its picture kept for the face and hair only, until it goes back on", not wrong, wrong)


def test_partner_named():
    print("\n=== everyone described in the memory, and anyone the beats treat as a person, is a character ===")
    sarah = "Sarah <Picture 1>: a woman with long blonde hair, wearing a black PVC miniskirt."
    beats = ("A dim living room with a leather couch.\n\nDan sits on the couch.\n\nSarah stands in front of him.\n\nSarah turns "
             "around and sits down on Dan's waist, allowing his penis to enter her vagina.")
    cast = lambda mem: [x["cast"] for x in S.plan_shots(beats, 5.0, [IMG1, IMG2, None, None], False, memory=mem)]
    forms = ("Dan: a muscular man.", "dan: a muscular man with short dark hair.", "Male: Dan, a muscular man.",
             "The man: Dan, muscular, short dark hair.", "Name: Dan\nGender: male", "Dan - a muscular man.",
             "**Dan**: a muscular man.", "Dan <Picture 2>: a muscular man.")
    check("a man described in the memory is a character however his entry is written, lower case, Male: or The man: included",
          all(cast(sarah + "\n\n" + f) == [["Dan"], ["Dan", "Sarah"], ["Dan", "Sarah"]] for f in forms),
          {f: cast(sarah + "\n\n" + f) for f in forms})
    undeclared = S.plan_shots("A dim living room with a leather couch.\n\nSarah stands in front of the couch.\n\nSarah slides off "
                              "her skirt.\n\nSarah mounts Dan's waist in reverse cowgirl, allowing his penis to enter vagina."
                              "\n\nShe gasps.\n\nSarah rides him.", 5.0, [IMG1, None, None, None], False, memory=sarah)
    check("a man the beats treat as a person is in the shot with her though the memory never names him, and he stays inside her",
          [x["cast"] for x in undeclared][2:] == [["Sarah", "Dan"]] * 3
          and all(x["inside"] == [("Dan", "Sarah")] for x in undeclared[2:]), [S.shot_line(x) for x in undeclared])
    check("a body part, sex, riding, sitting on or standing in front of someone, or doing what people do makes a name a "
          "person; rain, light, music, cities, days and cars do not",
          S.engaged(["Sarah sits down on Dan's waist.", "Sarah stands in front of Mike.", "Tom smiles.", "Sarah rides Jake.",
                     "Sarah has sex with Leo."], ["Sarah"]) == ["Dan", "Mike", "Tom", "Jake", "Leo"]
          and S.engaged(["Rain falls on the window.", "London sleeps under the rain.", "Candlelight dances over her skin.",
                         "Sarah looks out over London.", "Sarah meets him on Monday.", "Sarah flies to Paris and waits in Rome.",
                         "Music plays softly.", "Sarah rides the Metro.", "The Ferrari waits outside."], ["Sarah"]) == [])


def test_many():
    print("\n=== any number of people: each act, each pair and each head count on its own ===")
    mem = "Sarah <Picture 1>: a woman with long blonde hair.\nMia <Picture 2>: a woman with short red hair."
    plan = lambda *beats, m=mem: S.plan_shots("A dim bedroom.\n\n" + "\n\n".join(beats), 5.0, [IMG1, IMG2, None, None], False,
                                             memory=m)
    notes = lambda x: x["prompt"].split("\n\n")[-1]
    dp = plan("Sarah kneels on the bed. Dan and Jake stand beside her.",
              "Sarah climbs onto Dan and rides him while Jake takes her from behind.", "Sarah moans.", "Jake pulls out.",
              "Sarah keeps riding Dan.")
    check("two men in one woman are two acts, the second in her other hole, each held until it ends on its own, and "
          "everyone joined stays in the shot",
          [x["inside"] for x in dp[1:]] == [[("Dan", "Sarah"), ("Jake", "Sarah")]] * 2 + [[("Dan", "Sarah")]] * 2
          and "Jake's penis is buried deep inside Sarah's anus the whole shot" in notes(dp[2])
          and "Dan's penis is buried deep inside Sarah's vagina the whole shot" in notes(dp[3])
          and sorted(dp[3]["cast"]) == ["Dan", "Jake", "Sarah"], [S.shot_line(x) for x in dp])
    couples = plan("Dan and Jake sit on the couch.", "Sarah rides Dan.", "Jake fucks Mia from behind.", "Dan pulls out of Sarah.",
                   "Mia moans.")
    check("two couples are two pairs: each starts, holds and ends apart, and a shot of one keeps that one's partner",
          [x["inside"] for x in couples[1:]] == [[("Dan", "Sarah")], [("Dan", "Sarah"), ("Jake", "Mia")], [("Jake", "Mia")],
                                                 [("Jake", "Mia")]]
          and sorted(couples[4]["cast"]) == ["Jake", "Mia"], [S.shot_line(x) for x in couples])
    men = plan("Dan lies on the bed. Jake kneels behind him.", "Jake slides his cock into Dan.", "Dan groans.", m="")
    check("a man can be the one entered, in his anus, and the line says his",
          men[1]["inside"] == men[2]["inside"] == [("Jake", "Dan")]
          and "Jake's penis is buried deep inside Dan's anus the whole shot: his bottom" in notes(men[2]), [notes(x) for x in men])
    check("a head count says how many people a shot holds, three or more included",
          "There are three people in the shot, with one body for each person." in notes(dp[1])
          and S.cast_line(["A", "B", "C", "D"], ["A", "B", "C", "D"], "", "") == "There are four people in the shot, with "
          "one body for each person.")
    check("her in a shot with two women and a man who acts on her tells nobody's gender, and with one woman tells hers",
          S.contrast(["Dan watches her."], ["Sarah", "Mia", "Dan"], {"Sarah": "f"}) == {}
          and S.contrast(["Dan watches her."], ["Mia", "Dan"], {}) == {"Mia": "f"})


def test_hips():
    print("\n=== tape around the hips ===")
    people, gender = ["Crystal", "Dan"], {"Crystal": "f", "Dan": "m"}
    read = lambda t: S.rst.read(t, people, gender, {})[0]
    check("tape around the hips and between the legs is its own part, never the ankles",
          read("Dan wraps duct tape around her hips and between her legs.")
          == {"Crystal": ["duct tape wound all the way around her hips, front and back, and between her legs"]}
          and read("Crystal: a slim woman with duct tape wound around her waist and between her legs.")
          == {"Crystal": ["duct tape wound all the way around her waist, front and back, and between her legs"]})
    check("legs tied together are still the ankles, and a wrist position at the waist stays a position",
          read("Dan ties her legs together.") == {"Crystal": ["rope around her ankles"]}
          and read("Dan ties her wrists at her waist.") == {"Crystal": ["rope around her wrists at her waist"]})
    check("tape around the hips gets no pose", S.limb_facts("duct tape around her hips and between her legs")[:2] == ("", ""))
    wraps = {"Dan wraps duct tape around her entire body.":
             ["duct tape wound all the way around her torso, front and back"],
             "Dan wraps duct tape around her torso, pinning her arms to her sides.":
             ["duct tape wound all the way around her torso, front and back, pinning her arms to her sides"],
             "Dan winds duct tape around her chest and arms.":
             ["duct tape wound all the way around her torso, front and back, pinning her arms to her sides"],
             "Dan wraps tape around her several times.":
             ["duct tape wound all the way around her torso, front and back"],
             "Dan ties rope around her chest.": ["rope wound all the way around her torso, front and back"]}
    got = {t: read(t).get("Crystal") for t in wraps}
    check("a wrap around her body, torso or chest is a torso wrap that goes all the way round, never mouth or wrist tape",
          got == wraps, {t: g for t, g in got.items() if g != wraps[t]})
    check("her body or stomach mentioned in passing is not a wrap",
          read("Dan tapes her ankles as she lies on her stomach.") == {"Crystal": ["duct tape around her ankles"]}
          and read("Dan tapes her wrists, her chest heaving.") == {"Crystal": ["duct tape around her wrists"]})
    torso = wraps["Dan wraps duct tape around her entire body."][0]
    held = {"Crystal": [torso, "duct tape over her mouth"]}
    check("cutting the tape off her body takes the wrap off and leaves the mouth tape",
          S.rst.read("Dan cuts the tape off her body.", ["Crystal", "Dan"], {"Crystal": "f", "Dan": "m"}, held)[1]
          == {"Crystal": [torso]})
    check("a torso wrap gets no pose", S.limb_facts(wraps["Dan winds duct tape around her chest and arms."][0])[:2] == ("", ""))
    memory = "Dan: a tall man.\nCrystal: a slim woman with duct tape wound around her hips and between her legs."
    shots = S.plan_shots("A bedroom.\n\nDan looks down at her.\n\nCrystal turns her head.\n\n"
                         "Dan peels the tape off her hips.", 10.0, [None] * 4, False, memory=memory)
    check("on from the first shot as written, nothing on her ankles, and off when it is peeled off",
          shots[0]["held"] == "Crystal: duct tape wound all the way around her hips, front and back, and between her legs."
          and shots[0]["bound"] == {} and shots[2]["released"] == "Crystal: duct tape wound all the way around her hips, front "
          "and back, and between her legs.", [S.shot_line(x) for x in shots])
    check("a person referred to as her is in the shot when she is the only woman", shots[0]["cast"] == ["Dan", "Crystal"],
          shots[0]["cast"])
    away = S.plan_shots("A hallway.\nhold: Crystal, duct tape over her mouth\n\nDan walks.", 5.0, [None] * 4, False,
                        memory="Dan: a tall man.\nCrystal: a slim woman.")
    check("what someone wears is stated only while they are in the shot",
          away[0]["cast"] == ["Dan"] and "duct tape" not in away[0]["prompt"], away[0]["prompt"])


def test_no_false_removal():
    print("\n=== nothing comes off unless the beat takes it off ===")
    people, gender = ["Crystal", "Dan"], {"Crystal": "f", "Dan": "m"}
    hips = {"Crystal": ["duct tape around her hips and between her legs"]}
    both = {"Crystal": ["duct tape around her hips and between her legs", "duct tape over her mouth"]}
    off = lambda t, held: S.rst.read(t, people, gender, held)[1]
    check("tape stays on when the beat only mentions it, cuts more of it, or puts more on",
          all(off(t, hips) == {} for t in ("Dan removes her shirt and checks the tape.", "Dan cuts a strip of tape.",
                                            "Dan tears off a piece of tape and presses it over her mouth.")))
    check("tape coming off somewhere else does not take the hips tape",
          off("Dan rips the tape off her face.", hips) == {} and off("Dan rips the tape off her mouth.", both)
          == {"Crystal": ["duct tape over her mouth"]})
    check("taking it off by name or place takes off exactly that piece",
          off("Dan peels the tape off her hips.", both) == {"Crystal": ["duct tape around her hips and between her legs"]}
          and off("Dan removes the tape.", hips) == {"Crystal": ["duct tape around her hips and between her legs"]})
    unclear = S.rst.read("Dan pulls the tape off.", people, gender, both)
    check("with two pieces on and no place named, it asks instead of taking both",
          unclear[1] == {} and "which duct tape comes off" in unclear[2][0], unclear)
    script = ("A bedroom.\nhold: Crystal, duct tape around her hips and between her legs\n\nDan removes her shirt.\n\n"
              "Dan cuts a strip of tape.\n\nCrystal waits.")
    shots = S.plan_shots(script, 10.0, [None] * 4, False, memory="Crystal: a slim woman.\nDan: a tall man.")
    check("across a whole scene the tape stays held until something takes it off",
          all(x["held"] == "Crystal: duct tape around her hips and between her legs." and not x["released"] for x in shots),
          [S.shot_line(x) for x in shots])


def test_retakes():
    print("\n=== a broken restraint is rendered again ===")
    def once(n, k):
        return {"broken": True, "broken_frames": [1, 2, 3]} if (n == 3 and k == 1) else {"broken": False}
    out, calls = render(SCRIPT, dwpose=True, check=once)
    takes = [c["seed"] for c in calls["sample"] if c["shot"] == 3 and not c["foley"]]
    check("a shot whose cuffs break is rendered again on a new seed and the take that holds is kept",
          takes == [7, 8] and "shot 3:" in out[2] and "restraint broke: 1 retake(s), kept one where it held" in out[2]
          and out[2].count("restraint held") == 4, (takes, out[2]))
    check("info says the restraints are checked after each take", "are checked after each take" in out[2]
          and "kept by the prompt alone" not in out[2])
    always = lambda n, k: {"broken": True, "broken_frames": list(range(10 - 3 * k))} if n == 2 else {"broken": False}
    out, calls = render(SCRIPT, dwpose=True, check=always)
    takes = [c["seed"] for c in calls["sample"] if c["shot"] == 2 and not c["foley"]]
    check("when every take breaks, the one that breaks least is kept",
          takes == [7, 8, 9] and "kept the take with the fewest broken frames (1)" in out[2], (takes, out[2]))
    out, calls = render(SCRIPT, dwpose=True, check=always, pose_retries=0)
    check("pose_retries 0 renders each shot once", [c["shot"] for c in calls["sample"] if not c["foley"]] == [1, 2, 3, 4, 5, 6]
          and "kept by the prompt alone" in out[2])
    out, calls = render(SCRIPT, auto=("cn", "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"))
    check("on any checkpoint the node loads the pose controlnet itself and pose control runs",
          "pose controlnet minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors loaded by the node" in out[2]
          and [c["shot"] for c in calls["pose"]] == [2, 3, 4, 5, 6], out[2])


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
    plain = S.plan_shots("A cell.\n\nMara sits.", 5.0, [None] * 4, False, memory="Mara: a tall woman.")
    check("without an anchor the first paragraph is the scene, before the character memory",
          len(plain) == 1 and plain[0]["prompt"].startswith("A cell.\n\nMara: a tall woman.\n\nMara sits."), plain[0]["prompt"])
    out, _ = render(SCRIPT, plan_only=True, character_memory="Dan: a guard.", negative="x", foley_level=0.2)
    check("inputs left over from an older workflow are named in info, not a crash",
          "ignored inputs from an older version of this node: foley_level, negative" in out[2], out[2])
    check("...and character_memory reaches every shot its person is in", out[3].count("Dan: a guard.") == 5
          and "Dan: a guard." not in out[3].split("[shot 2]")[0], out[3])


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
    test_shot_length()
    test_speech()
    test_presence()
    test_beats_decide_presence()
    test_continuity_lines()
    test_hands_and_backs()
    test_strangers()
    test_things()
    test_memory_layouts()
    test_mentioned_not_present()
    test_unstated_gender()
    test_unnamed_pronouns()
    test_named_ship_only()
    test_quiet_mouths()
    test_directive_forms()
    test_reading()
    test_wardrobe()
    test_undressed()
    test_every_garment()
    test_partner_named()
    test_many()
    test_hips()
    test_no_false_removal()
    test_retakes()
    test_mumble()
    test_sound_lines()
    test_ambient_bed()
    test_recovered_look_render()
    test_fast_h3_pictures()
    test_pictures_by_checkpoint()
    test_memory()
    test_grading()
    test_lora_refit()
    test_scene_inputs()
    test_upscale_module()
    test_upscale_render()
    test_upscalers_real()
    print("\nRESULT: " + ("ALL PASSED" if not _fails else f"{len(_fails)} FAILURE(S): " + "; ".join(_fails)))
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
