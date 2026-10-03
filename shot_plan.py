# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""Per-shot records passed from prompt planning to rendering."""

from dataclasses import dataclass, field


@dataclass
class Shot:
    prompt: str
    cast: list[str]
    speech: bool
    sounded: bool
    voiced_only: bool
    events: list[str]
    frame_count: int = 0
    refs: list[object] = field(default_factory=list)
    line_seconds: float = 0.0     # the planner's estimate of the spoken line, words / WORDS_PER_SEC
    # Pose control's facts, from the restraint planning (see sampler.pose_candidate).
    bound_pose: dict = field(default_factory=dict)   # {name: {arms, legs, ankle_gap, anchored, fall, latch_limbs}}, on-screen wearers
    bound_fall: bool = False      # a restrained body falls in this shot
    fallers: list[str] = field(default_factory=list)  # who falls in it, when bound_fall
    limbs_on: bool = False        # limb hardware goes on in this shot
    limbs_off: bool = False       # limb hardware comes off in this shot
    restrainers: list[str] = field(default_factory=list)  # wearers pose control leaves alone: they do the restraining

    @property
    def limb_change(self):
        """Limb hardware goes on or comes off in this shot."""
        return bool(self.limbs_on or self.limbs_off)


@dataclass
class ShotPlan:
    shots: list[Shot] = field(default_factory=list)

    @property
    def prompts(self):
        return [shot.prompt for shot in self.shots]

    def add(self, prompt, cast, speech, sounded, voiced_only, events):
        shot = Shot(prompt, list(cast or ()), bool(speech), bool(sounded),
                    bool(voiced_only), list(events or ()))
        self.shots.append(shot)

    def set_frame_counts(self, counts):
        counts = [int(n) for n in counts]
        if len(counts) != len(self.shots) or any(n <= 0 for n in counts):
            raise ValueError("frame counts must be positive and match the planned shots")
        for shot, count in zip(self.shots, counts):
            shot.frame_count = count

    def validate(self):
        if any(shot.frame_count <= 0 for shot in self.shots):
            raise ValueError("each shot needs a positive frame count before rendering")
        return self

    def __len__(self):
        return len(self.shots)


@dataclass
class PreparedVideo:
    """Resolved inputs consumed by the render stage; model/tensor handles are shared."""
    _placed_shots: object
    _first_is_plate: object
    _returns: object
    _soft_landing: object
    _tagged_names: object
    ambient_audio: object
    ambient_level: float
    apply_model_sampling: bool
    audio_vae: object
    auto_sound: bool
    bared_shots: object
    cfg: float
    cleanup_between_shots: bool
    clip: object
    first_frame: object
    foley_level: float
    h: int
    latent_upscale: str
    latent_upscale_scale: float
    megapixels: float
    model: object
    moved_shots: object
    negative: object
    notes: list[str]
    plan: ShotPlan
    ref_noise_aug: float | None
    restart_after_removal: bool
    revealed_shots: object
    sampler_name: str
    scheduler: str
    seed: int
    shift_audio: float
    shift_video: float
    sigmas: object
    silence_nonspeech: bool
    speech_lead_seconds: float
    speech_tail_seconds: float
    hold_levels: float
    staging_shots: object
    steps: int
    stripped_shots: object
    cut_shots: object
    tiled_decode: bool
    trim_seam: bool
    upscale: str
    upscale_batch: int
    upscale_model: str
    upscale_target_short_edge: int
    vae: object
    w: int
    shot_rooms: object = None         # {0-based shot: (room it opens in, room it ends in)}
    hardware_changed: object = None   # 1-based shots that put hardware on or take it off
    held_shots: object = None         # {1-based shot: {name: shot their restraint or gag went on}}, as it opens
    held_items: object = None         # {1-based shot: {name: (items, where)}} for those held through it
    portrait_slots: object = None     # {0-based shot: {name: 1-based places of their portraits in its refs}}
    shot_frames: object = None        # {0-based shot: (who its frames show, who is still there at its end)}
    reentry_shots: object = None      # {0-based shot: who walks in while the keyframe still has them}
    own_grade_shots: object = None    # {0-based shots whose change of level over the take is the author's}
    refs_ok: bool = True              # False on a model that reads no reference rows (FastH3)
    outdoor_shots: object = None      # {0-based shots whose place is outside}
    fast_h3: bool = False             # FastVideo's FastH3: VSA goes on at render
    hyperflow: object = None          # Hyperflow's settings, with "two_time" when it can run
    pose_controlnet: object = None    # the MODEL_PATCH wired to pose_controlnet
    pose_ok: bool = False             # pose_status passed: pose control runs this render
    pose_note: str = ""               # pose_status's note
    pose_strength: float = 1.0
    pose_end: float = 0.6             # share of the steps held to the skeleton
    pose_shots: str = "repair broken shots"
    pose_draw: str = "everyone"
