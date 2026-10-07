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
    fc = [S.rt.align_frame_count(48)] * 2 + [S.rt.align_frame_count(144)] + [S.rt.align_frame_count(48)] * 3
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


def test_shot_length():
    print("\n=== shot length ===")
    script = ("A cell.\n\nMara sits on the cot.\n\nDan opens the door, walks in and sits down.\n\n"
              "Dan says \"Not a sound, or you will regret it tonight.\"\n\n"
              "Dan handcuffs her wrists behind her back.\nhold: Mara, handcuffs behind her back\n\n"
              "Mara waits.\nseconds: 12")
    twelve = S.rt.align_frame_count(288)
    beat = [s["frames"] for s in S.plan_shots(script, 15.0, [None] * 4, False, from_beat=True)]
    check("from the beat: one action, three actions, a spoken line, a restraint going on, an explicit length",
          beat == [73, 175, 107, 124, twelve], beat)
    capped = [s["frames"] for s in S.plan_shots(script, 5.0, [None] * 4, False, from_beat=True)]
    check("shot_seconds caps every estimate and seconds: still wins", capped == [73, 124, 107, 124, twelve], capped)
    fixed = [s["frames"] for s in S.plan_shots(script, 5.0, [None] * 4, False)]
    check("fixed gives every shot shot_seconds", fixed == [124, 124, 124, 124, twelve], fixed)
    check("a beat with nothing to stage gets one action's worth", S.shot_frames("Silence.", 362, False) == 73)
    out, _ = render(script, plan_only=True, seconds=15.0)
    fixed_out, _ = render(script, plan_only=True, seconds=15.0, shot_length="fixed")
    check("the widget reaches the plan, from the beat by default",
          "shot 2: 175 frames" in out[2] and "shot 2: 362 frames" in fixed_out[2], (out[2], fixed_out[2]))


def test_presence():
    print("\n=== who is in each shot ===")
    script = ("A dim cell. Mara <Picture 1> sits on a cot. Dan <Picture 2> stands by the door.\n\n"
              "Dan walks out and slams the door.\n\nMara cries.\n\nDan comes back in.\n\nHe leaves.\nexit: Dan\n\n"
              "Mara waits.\n\ncut\nA garden. Mara walks along the path.")
    shots = S.plan_shots(script, 10.0, [IMG1, IMG2, None, None], False, memory="Mara: a tall woman.\nDan: a guard.\nSetting: a prison.")
    check("a named exit, a return when named, an exit: line, and a cut",
          [s["cast"] for s in shots] == [["Dan"], ["Mara"], ["Mara", "Dan"], ["Mara", "Dan"], ["Mara"], ["Mara"]],
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
    check("no head count while someone in the shot is not a declared character",
          "one person" not in loose and "two people" not in loose and "There are two people in the shot" in known, (loose, known))
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
    check("what is held stays on for the whole shot, wrists in place, when the next thing goes on",
          "Mara: handcuffs behind her back. All of it stays on for the whole shot. Mara's wrists stay locked behind the back "
          "the whole time. By the last frame, Mara has duct tape over her mouth in plain view" in shots[2]["prompt"], shots[2]["prompt"])
    legs = S.plan_shots("A room.\nhold: Mara, wrists zip tied in front; rope around her ankles; cuffs to the bed rail\n\nMara waits.",
                        5.0, [None] * 4, False)
    check("held wrists in front and ankles get their own line; a restraint fastened to an object adds none",
          "Mara's wrists stay bound in front the whole time." in legs[0]["prompt"]
          and "Mara's ankles stay bound together the whole time." in legs[0]["prompt"]
          and legs[0]["prompt"].count("the whole time.") == 2, legs[0]["prompt"])
    one = S.plan_shots("Mara waves.", 5.0, [None] * 4, True)
    check("a first frame is claimed too", f"<Picture 1> {claim}" in one[0]["prompt"], one[0]["prompt"])


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
    print("\n=== reference pictures only for checkpoints trained on them ===")
    graph = {"9": {"class_type": "H3LongVideos", "inputs": {"model": ["7", 0], "clip": ["2", 0]}},
             "7": {"class_type": "LoraLoader", "inputs": {"model": ["5", 0], "clip": ["2", 0], "lora_name": "hyperflow.safetensors"}},
             "5": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3/10Eros_Max_H3_FL2VA-INT8-ConvRot-HQ.safetensors"}},
             "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors"}},
             "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl.safetensors"}}}
    check("the checkpoint is traced back through the LoRA loaders on the node's own model link",
          S.checkpoint_name(graph, "9") == "h3/10Eros_Max_H3_FL2VA-INT8-ConvRot-HQ.safetensors" and S.checkpoint_name({}, "9") == "")
    hybrid = dict(graph, **{"9": {"class_type": "H3LongVideos", "inputs": {"model": ["3", 0]}}})
    check("only a ref2va checkpoint takes reference pictures; an unknown loader keeps them",
          S.takes_pictures("m", graph, "9")[0] is False and "10Eros_Max_H3_FL2VA" in S.takes_pictures("m", graph, "9")[1]
          and S.takes_pictures("m", hybrid, "9") == (True, "") and S.takes_pictures("m", None, None) == (True, ""))
    script = "A cell. Mara <Picture 1> waits. Dan <Picture 2> stands guard.\n\nMara paces.\n\nMara sits.\n\ncut\nA garden. Mara walks."
    out, calls = render(script, ref_image_1=IMG1, ref_image_2=IMG1, graph=graph, unique_id="9")
    check("on an FL2VA checkpoint like 10Eros no shot gets a portrait or a recovered look, so nobody is drawn twice",
          all(c["refs"] == [] for c in calls["cond"]) and all("<Picture 2>" not in c["prompt"] for c in calls["cond"])
          and "not trained on reference pictures" in out[2], [len(c["refs"]) for c in calls["cond"]])
    out, calls = render(script, ref_image_1=IMG1, ref_image_2=IMG1, graph=hybrid, unique_id="9")
    check("on the hybrid fl2va_ref2va the portraits still go in", [len(c["refs"]) for c in calls["cond"]] == [1, 1, 1]
          and "not trained on reference pictures" not in out[2], [len(c["refs"]) for c in calls["cond"]])


def test_fast_h3_pictures():
    print("\n=== FastH3 gets only what it was distilled on ===")
    script = "A cell. Mara <Picture 1> waits. Dan <Picture 2> stands guard.\n\nMara paces.\n\nMara sits.\n\ncut\nA garden. Mara walks."
    out, calls = render(script, fast=True, ref_image_1=IMG1, ref_image_2=IMG1)
    check("no reference pictures and no picture tags for them, even after a cut",
          len(calls["cond"]) == 3 and all(c["refs"] == [] for c in calls["cond"])
          and all("<Picture 2>" not in c["prompt"] and "Mara <Picture" not in c["prompt"] for c in calls["cond"]),
          [(len(c["refs"]), c["prompt"][:120]) for c in calls["cond"]])
    check("the opening frame still carries the shot on as <Picture 1>",
          calls["cond"][1]["handoff"] is not None and "<Picture 1> is the frame this shot opens on" in calls["cond"][1]["prompt"]
          and calls["cond"][2]["handoff"] is None, [c["handoff"] is None for c in calls["cond"]])
    check("info says why the pictures were left out", "distilled without reference pictures" in out[2], out[2])
    _, base = render(script, ref_image_1=IMG1, ref_image_2=IMG1)
    check("other checkpoints still get her portrait, and her current look after the cut",
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
    key, look = chain(lambda key, k: burn(key, 0.05 + 0.05 * k / 23))
    check("twelve shots that each come out more saturated and contrasty than they started keep the first shot's look",
          abs(sat(key) / sat(look) - 1) < 0.02 and abs(con(key) / con(look) - 1) < 0.02, (sat(key) / sat(look), con(key) / con(look)))
    key, look = chain(lambda key, k: burn(key, -0.25 * max(0.0, 1 - k / 6) + 0.03 * k / 23))
    check("shots that open a little flat and brighten up do not wash the video out",
          abs(sat(key) / sat(look) - 1) < 0.03 and abs(con(key) / con(look) - 1) < 0.03, (sat(key) / sat(look), con(key) / con(look)))
    first = torch.stack([burn(orig, 0.1 * k / 23) for k in range(24)])
    check("the first shot, and a shot after a cut, set the look and are left as rendered",
          rt.shot_grade(None, first[0], first[-1]) is None)
    f = torch.stack([burn(orig, 0.05) * (1 - 0.6 * k / 23) for k in range(24)])
    dark = float(f[-1].mean())
    rt.grade_frames(f, rt.shot_grade(orig, f[0], f[-1]))
    check("a real change inside a shot, like the lights going down, is kept", float(f[-1].mean()) < 0.6 * float(f[0].mean())
          and float(f[-1].mean()) < 1.2 * dark, (float(f[0].mean()), float(f[-1].mean()), dark))
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
    check("the look readout gives contrast and colour, and nothing for a flat frame",
          rt.look(orig) is not None and rt.look(burn(orig, 0.3))[1] > rt.look(orig)[1] and rt.look(black[0]) is None)
    hand = rt.match_frame(burn(orig, 0.3).unsqueeze(0).clone(), rt.tone(orig))
    check("the handing-off frame can be put back on the shot's look on its own",
          float((rt.tone(hand) - rt.tone(orig)).abs().mean()) < 0.004)


def test_reading():
    print("\n=== restraints read from the beats ===")
    script = ("A dim room. Mara lies on the bed. Dan stands beside her.\n\nDan handcuffs her wrists behind her back.\n\n"
              "Dan wraps duct tape around her mouth.\n\nDan sits down and watches her.\n\nDan pulls the tape off.")
    shots = S.plan_shots(script, 10.0, [None] * 4, False)
    check("cuffs go on, then stay on while the tape goes on, with no hold lines",
          shots[0]["added"] == "Mara: handcuffs behind her back." and shots[1]["held"] == "Mara: handcuffs behind her back."
          and shots[1]["added"] == "Mara: duct tape over her mouth."
          and "Mara's wrists stay locked behind the back the whole time." in shots[1]["prompt"], [S.shot_line(x) for x in shots])
    check("pose control latches the cuffs as they go on and holds them after",
          shots[0]["bound"]["Mara"]["latch_limbs"] == ("arms",) and shots[1]["bound"]["Mara"]["latch_limbs"] == ()
          and all(x["bound"].get("Mara", {}).get("arms") == "behind the back" for x in shots[1:]))
    check("the tape comes off and the cuffs stay", shots[3]["released"] == "Mara: duct tape over her mouth."
          and shots[3]["held"] == "Mara: handcuffs behind her back.", S.shot_line(shots[3]))
    people, gender = ["Mara", "Dan"], {"Mara": "f", "Dan": "m"}
    read = lambda t, held=None: S.rst.read(t, people, gender, held or {})[0]
    expect = {
        "Dan cuffs Mara.": {"Mara": ["handcuffs on her wrists"]},
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
    script = ("A bedroom. Crystal stands by the bed.\n\nMax takes off her jacket.\n\nCrystal sits on the bed.\n\n"
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
    told = S.plan_shots("A bedroom. Crystal waits.\n\nCrystal turns.\nremove: Crystal, black jacket\n\nCrystal sits."
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


def test_hips():
    print("\n=== tape around the hips ===")
    people, gender = ["Crystal", "Dan"], {"Crystal": "f", "Dan": "m"}
    read = lambda t: S.rst.read(t, people, gender, {})[0]
    check("tape around the hips and between the legs is its own part, never the ankles",
          read("Dan wraps duct tape around her hips and between her legs.")
          == {"Crystal": ["duct tape around her hips and between her legs"]}
          and read("Crystal: a slim woman with duct tape wound around her waist and between her legs.")
          == {"Crystal": ["duct tape around her waist and between her legs"]})
    check("legs tied together are still the ankles, and a wrist position at the waist stays a position",
          read("Dan ties her legs together.") == {"Crystal": ["rope around her ankles"]}
          and read("Dan ties her wrists at her waist.") == {"Crystal": ["rope around her wrists at her waist"]})
    check("tape around the hips gets no pose", S.limb_facts("duct tape around her hips and between her legs")[:2] == ("", ""))
    memory = "Dan: a tall man.\nCrystal: a slim woman with duct tape wound around her hips and between her legs."
    shots = S.plan_shots("A bedroom. Dan stands by the bed.\n\nDan looks down at her.\n\nCrystal turns her head.\n\n"
                         "Dan peels the tape off her hips.", 10.0, [None] * 4, False, memory=memory)
    check("on from the first shot as written, nothing on her ankles, and off when it is peeled off",
          shots[0]["held"] == "Crystal: duct tape around her hips and between her legs." and shots[0]["bound"] == {}
          and shots[2]["released"] == "Crystal: duct tape around her hips and between her legs.", [S.shot_line(x) for x in shots])
    check("a person referred to as her is in the shot when she is the only woman", shots[0]["cast"] == ["Dan", "Crystal"],
          shots[0]["cast"])
    away = S.plan_shots("A hallway. Dan waits.\nhold: Crystal, duct tape over her mouth\n\nDan walks.", 5.0, [None] * 4, False,
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
    test_presence()
    test_beats_decide_presence()
    test_continuity_lines()
    test_quiet_mouths()
    test_directive_forms()
    test_reading()
    test_wardrobe()
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
    test_scene_inputs()
    test_upscale_module()
    test_upscale_render()
    test_upscalers_real()
    print("\nRESULT: " + ("ALL PASSED" if not _fails else f"{len(_fails)} FAILURE(S): " + "; ".join(_fails)))
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
