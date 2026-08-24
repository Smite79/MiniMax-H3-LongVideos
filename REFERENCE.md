# H3-LongVideos — full reference

The complete behaviour of the node, field by field. For a synopsis and quick
start, see [README.md](README.md).

## Install
Copy the `H3-LongVideos/` folder into `ComfyUI/custom_nodes/` and restart
ComfyUI (full server restart, not just a browser refresh).

## Node: **H3 Long Videos (FL2VA + REF2VA)**  (category: sampling/minimax)
*One node covers both H3 conditioning tasks. It registers under four keys --
`H3LongVideos`, `H3LongVideosFL2VA`, `H3LongVideosV1` and `H3LongVideosREF2VA` --
all aliases onto the same class, so every workflow saved under any previous name
keeps loading with nothing to re-wire. REF2VA was briefly a separate, duplicated
sampler; it is now folded in.*

<!-- AUTOGEN:FIELDS BEGIN -- edit gen_reference.py, not this block -->

There are **71** inputs and **12** outputs. Required inputs are marked **R**; everything else is optional and has a working default.

### Wiring

| field | | type | constraints |
|---|---|---|---|
| `model` | **R** | MODEL | — |
| `clip` | **R** | CLIP | — |
| `vae` | **R** | VAE | — |
| `audio_vae` | **R** | VAE | — |
| `first_frame` |  | IMAGE | — |
| `ref_image_1` |  | IMAGE | — |
| `ref_image_2` |  | IMAGE | — |
| `ref_image_3` |  | IMAGE | — |
| `ref_image_4` |  | IMAGE | — |

**`ref_image_1`** — Reference image <Picture 1> -- identity/appearance carried into the shots. Which shots receive it is set by ref_mode (or <Picture N> tags in the beats); a referenced shot ALSO carries the previous frame as its keyframe, so taking a reference never costs continuity.

**`ref_image_2`** — Reference image <Picture 2>.

**`ref_image_3`** — Reference image <Picture 3>.

**`ref_image_4`** — Reference image <Picture 4>.

### The scene

| field | | type | constraints |
|---|---|---|---|
| `prompt` | **R** | STRING | **input socket**, multiline |
| `character_memory` |  | STRING | **input socket**, multiline |
| `anchor_override` |  | STRING | **input socket**, multiline |
| `beat_split` |  | choice | `auto`, `each line` |
| `per_beat_length` |  | BOOLEAN | default `True` |

**`prompt`** — This IS the integrated_multimodal_description (the visual/action timeline). First paragraph = PERMANENT IDENTITY kept across the whole video (hair, face, build) -- put NO clothing in this prose, or it can't be changed later. Put clothing on a 'wardrobe:' line (in the first paragraph and/or the character_memory field); it's the only channel that can be changed/removed mid-chain. Each later paragraph = one scene beat. Put dialogue and 'lips closed' beats in the beat bodies.

**`character_memory`** — Optional dedicated wardrobe channel (same role as a 'wardrobe:' line in the first paragraph -- use whichever you prefer; this field wins if both are set). Re-stamped into every shot so clothing holds even when the camera crops it out. IMPORTANT: this is the ONLY place clothing should live -- keep it out of the anchor prose, or a removal won't stick because the immutable anchor keeps re-adding it. To change/remove an item mid-chain, put 'wardrobe: <new full sheet>' inside the beat where it changes; omit the removed item from the new sheet and it stays gone. WRITE ATTRIBUTES, NOT NOUN PHRASES: 'silver hair, 27, red jacket' -- NOT 'a woman with silver hair'. A noun phrase renders as 'She (a woman with...)', i.e. two subjects in one clause, which causes character duplication. The node strips them automatically, but writing attributes directly is cleaner. ONE-TOKEN EDITS (no restating the outfit): 'wardrobe: -= jacket' removes the jacket, 'wardrobe: += sunglasses' adds one. TWO+ PEOPLE: name them -- 'Maya = grey shorts, red jacket; Jon = navy overalls', then edit one at a time: 'wardrobe: Maya -= jacket' leaves Jon untouched.

**`anchor_override`** — Set the persistent look explicitly instead of using the first paragraph. When this is filled in, EVERY paragraph of the prompt box is a beat/shot -- nothing is consumed as the identity anchor. Put the permanent identity here (hair, face, build, age) and the clothing in character_memory.

**`beat_split`** — How the prompt box becomes beats. Beats are meant to be separated by a BLANK line (or a '##' line) -- but six beats typed on six consecutive lines are ONE paragraph, so they would render as one shot with six actions crammed into it, which looks like everyone is moving at triple speed. auto (default): blank lines first, then any paragraph still holding several lines is split one beat per LINE, and the info output says so. 'each line': every line is its own beat -- same result, stated explicitly. Neither can lose a beat. (The old strict 'blank line' option was REMOVED: it was the only setting that could silently collapse beats, and a stored value of it now reads as 'auto'.) Directive lines (wardrobe:, seconds:, exit:) are never beats -- they attach to the beat that follows them.

**`per_beat_length`** — PACING. Size each shot from what its beat actually stages, instead of giving every shot the same length. ON (default): a beat's time is ~2s of setup plus ~2.5s per action clause, or its spoken line, whichever is longer -- so 'she takes off her jacket and drops it on the bench' gets ~7s and a three-part beat gets more. OFF: every shot gets the full ceiling. WHY IT MATTERS: a 3s action in a 12s shot leaves 9 seconds the model was told nothing about, and it fills them by repeating or REVERSING the action -- which is why clothing comes off and goes back on. The estimate leans SHORT on purpose: an unfinished action is continued by the next shot from the handoff frame, while an overlong one is unrecoverable. Never exceeds the ceiling (shot_seconds or the VRAM budget) and always lands on the 17n+5 grid. Override any single beat with 'seconds: 8' on its own line inside that paragraph -- that wins over everything, including this toggle.

### Size and length

| field | | type | constraints |
|---|---|---|---|
| `resolution` | **R** | choice | `16:9`, `9:16`, `4:3`, `3:4`, `1:1`, `21:9`, `9:21` |
| `megapixels` | **R** | FLOAT | default `1.0`, range `0.1`–`4.0` |
| `shot_seconds` |  | FLOAT | **input socket**, range `0.0`–`15.1` |
| `allow_oversize_shots` |  | BOOLEAN | default `False` |
| `allow_res_backoff` |  | BOOLEAN | default `True` |
| `fps` |  | INT | default `24`, range `1`–`60` |

**`resolution`** — ASPECT RATIO only -- `megapixels` decides the size. The two are independent, so changing shape does not change cost. Each ratio uses H3's own dimensions as its reference, which is why 1.00MP reproduces the model's native sizes exactly (16:9 -> 1344x768, 21:9 -> 1536x672).

**`megapixels`** — Pixel BUDGET, applied to the preset's aspect ratio. 1.00 = 1024x1024 worth of pixels, the same convention as ComfyUI's Scale Image to Total Pixels. START at 1.00: every NATIVE preset reproduces its own size there (1344x768, 1536x672, ...), then step down -- 0.83 gives a 704 short edge, 0.65 gives 640 -- for speed, VRAM and longer shots. Cost and training fit track TOTAL PIXELS, not the short edge: 1:1 768x768 reads as native by short edge but is only 0.56MP, while 21:9 1536x672 reads as sub-native at a full 0.98MP. Snapped to multiples of 32; `info` reports the size and MP actually produced. Set 0 to use the preset's own dimensions instead.

**`shot_seconds`** — Length of EACH shot in seconds, taken from a connected input -- H3 Shot Length is the intended source, since it also reports the matching frame count on the 17k+5 grid. Leave it UNCONNECTED for auto: the largest shot that fits at the chosen size, which is what a 0 in the old widget did. One paragraph = one shot, so total video = (paragraph count) x this. Max ~15s.

**`allow_oversize_shots`** — OFF (default): a forced shot_seconds that won't fit VRAM is clamped DOWN to what fits, and the clamp is reported in info. ON: honor the requested length even if it exceeds the budget -- the render may spill into system RAM (slow) or OOM. Only affects forced shot_seconds, not auto.

**`allow_res_backoff`** — If VRAM is tight, step resolution down instead of failing.

**`fps`** — DISPLAY ONLY -- H3 always renders 24 fps. The model's frame grid and its audio latent are both defined against 24, so this node computes every duration at 24 regardless of what you set here. Set your video-save node to 24 as well, or the clip plays at the wrong speed.

### Sampling

| field | | type | constraints |
|---|---|---|---|
| `steps` | **R** | INT | default `20`, range `1`–`200` |
| `cfg` | **R** | FLOAT | default `1.0`, range `0.0`–`30.0` |
| `sampler_name` | **R** | choice | `euler`, `euler_cfg_pp`, `euler_ancestral`, `euler_ancestral_cfg_pp`, `heun`, `heunpp2`, `exp_heun_2_x0`, `exp_heun_2_x0_sde`, `dpm_2`, `dpm_2_ancestral`, `lms`, `dpm_fast`, `dpm_adaptive`, `dpmpp_2s_ancestral`, `dpmpp_2s_ancestral_cfg_pp`, `dpmpp_sde`, `dpmpp_sde_gpu`, `dpmpp_2m`, `dpmpp_2m_cfg_pp`, `dpmpp_2m_sde`, `dpmpp_2m_sde_gpu`, `dpmpp_2m_sde_heun`, `dpmpp_2m_sde_heun_gpu`, `dpmpp_3m_sde`, `dpmpp_3m_sde_gpu`, `ddpm`, `lcm`, `ipndm`, `ipndm_v`, `deis`, `res_multistep`, `res_multistep_cfg_pp`, `res_multistep_ancestral`, `res_multistep_ancestral_cfg_pp`, `gradient_estimation`, `gradient_estimation_cfg_pp`, `er_sde`, `seeds_2`, `seeds_3`, `sa_solver`, `sa_solver_pece`, `ddim`, `uni_pc`, `uni_pc_bh2` |
| `scheduler` | **R** | choice | `simple`, `sgm_uniform`, `karras`, `exponential`, `ddim_uniform`, `beta`, `normal`, `linear_quadratic`, `kl_optimal` |
| `seed` | **R** | INT | default `0`, range `0`–`18446744073709551615` |
| `vary_seed_per_shot` |  | BOOLEAN | default `False` |
| `apply_model_sampling` |  | BOOLEAN | default `True` |
| `shift_video` |  | FLOAT | default `12.0`, range `1.0`–`32.0` |
| `shift_audio` |  | FLOAT | default `3.0`, range `0.25`–`16.0` |

**`steps`** — Base H3 wants ~20 (res_multistep + simple). Drop to 6-8 ONLY with a working distill/turbo LoRA or a low-step MXFP8 checkpoint -- on the bare base model, low steps are the #1 cause of soft output.

**`vary_seed_per_shot`** — Give each shot its own seed (seed+1, seed+2, ...) instead of one seed for the whole chain. OFF by default, because this node builds a CONTINUOUS TAKE. The seed sets the noise field every shot is sampled from, and that field is what fixes the stochastic detail -- grain, micro-texture, the exact rendering of surfaces the prompt never names. Change it between shots and all of that resets at the boundary, which reads as a cut even when the keyframe anchors the frame and the location is unchanged: the same room, rendered afresh. Shots still differ with one seed -- each has its own beat text and its own handoff keyframe. Turn this ON only when you WANT the beats to look separately shot, or when repeated beats are coming out too alike.

**`apply_model_sampling`** — Patch ModelSamplingMiniMaxH3 (the dual video/audio schedule) inside the node so you don't have to wire it upstream. Without it, H3's audio comes out as gibberish. Turn OFF only if you patch it yourself upstream.

**`shift_video`** — Video flow shift. 12 = base H3 (correct default). A low-step MXFP8 checkpoint wants ~8. Only used when apply_model_sampling is on.

**`shift_audio`** — Audio flow shift. 3 = base H3. COUPLED to shift_video on ComfyUI 0.31+: the audio latent rides the video schedule scaled by audio_scale = shift_video / shift_audio (12/3 = 4). Flattening that ratio toward 1.0 breaks the audio branch -- babble or silence -- so if you lower shift_video, lower this by the same factor. Only used when apply_model_sampling is on.

### Chaining shots

| field | | type | constraints |
|---|---|---|---|
| `trim_seam` |  | BOOLEAN | default `True` |
| `handoff_offset` |  | INT | default `0`, range `0`–`12` |
| `cleanup_between_shots` |  | BOOLEAN | default `True` |
| `vram_headroom_gb` |  | FLOAT | default `1.5`, range `0.0`–`32.0` |

**`trim_seam`** — Drop the FIRST frame of every shot after the first. That frame is the model's own reproduction of the handoff -- the last frame of the previous shot, which it was anchored to. Keeping it shows the same moment twice and reads as a stutter. So at a working seam the last frame of one shot and the first of the next are NOT identical: they are one frame of normal motion apart, which is what continuous footage looks like. Turn this off only to inspect how closely the anchor was reproduced.

**`handoff_offset`** — End each shot this many frames early and hand THAT frame to the next shot instead of the literal last frame. Set 2-4 if chained shots open with moving/talking mouths -- it avoids seeding the next shot with a mid-word open-mouth pose. Trims the matching audio tail too. 0 = last frame.

**`cleanup_between_shots`** — Between beats, move each shot's decoded video+audio to system RAM and run a full VRAM+RAM purge (GC + CUDA cache), so a long chain doesn't accumulate on the GPU and OOM. Recommended on 16GB. Turn off only on a big card where you want to skip the per-shot cleanup cost.

### Audio

| field | | type | constraints |
|---|---|---|---|
| `global_soundscape` |  | STRING | **input socket**, multiline |
| `non_diegetic_music` |  | STRING | **input socket**, multiline |
| `auto_soundscape` |  | choice | `off`, `fill if blank`, `always` |
| `auto_silence_nonspeech` |  | BOOLEAN | default `True` |
| `allow_nonspeech_vocals` |  | BOOLEAN | default `False` |
| `mute_nonspeech_audio` |  | BOOLEAN | default `True` |
| `mute_fade_ms` |  | INT | default `40`, range `0`–`500` |

**`global_soundscape`** — AMBIENT/environmental sound only (rain, room tone, footsteps, engines). Appended to every shot as overall_soundscape. NOT for dialogue -- speech and lip timing live in the prompt beats. Leave blank for no ambient bed.

**`non_diegetic_music`** — Background SCORE only -- genre, mood, instrumentation, tempo -- music that is NOT part of the scene. Music is OPT-IN: leave this BLANK and the node emits 'non_diegetic_music: N/A' on every shot so H3 adds no score (fixes unwanted music). Fill it in to request a specific score. Not for music a character plays/hears (that's diegetic; put it in the beat).

**`auto_soundscape`** — Build the ambient bed from the scene instead of typing one. Reads the ANCHOR (the soundscape is global, so it must describe the PLACE, not one beat's action), falling back to the beats when the anchor is pure camera language. 'A disused aircraft hangar' -> cavernous interior, long reverb, distant metal ticks. Weather layers first: rain, wind, snow, fog. NO human sounds are ever generated -- no chatter, crowd or announcements -- because an ambient bed that implies voices is how H3 starts talking. 'fill if blank' derives one only when the global_soundscape input is unconnected or empty, so connecting your own text is enough to keep it. 'always' derives even when you HAVE connected one, overriding it -- deliberate, for comparing your bed against a derived one without unwiring. 'off' never derives. Whichever fires, the bed actually used comes out on the `soundscape` output, so you can read it and wire it back into the input to pin it.

**`auto_silence_nonspeech`** — Stop mouths moving / gibberish audio on shots with no dialogue. Any beat with no scripted speech gets an explicit 'lips closed, no dialogue' clause, so H3 doesn't animate or vocalize a mouth before real dialogue. Beats with quoted dialogue ("...") are left alone. To make someone speak, put the words in double quotes. Turn OFF to manage lip state yourself.

**`mute_nonspeech_audio`** — DETERMINISTIC gibberish fix: FULLY silence the audio of any shot that has no scripted dialogue (no double-quoted line). Prompt-level silencing asks H3 not to babble; this guarantees it. TRADE-OFF: it also removes that shot's generated ambience/SFX, so lay a continuous ambient bed under the video in post. Shots WITH quoted dialogue keep their audio untouched.

**`allow_nonspeech_vocals`** — Allow non-speech vocal sounds (screams, sobs, gasps, moans) on shots with no dialogue. When ON, the node skips the lips-closed clause and softens the no-voice soundscape so it bans speech, dialogue and singing but permits screams, sobs, gasps and other distress vocalizations. The audio branch is also left unmuted on those shots. Speech is still suppressed — only double-quoted lines count as speaking — so H3 does not invent chatter. Turn this ON when your scene contains distress sounds that H3 would otherwise suppress. Keep `auto_silence_nonspeech` ON for shots that should be truly silent.

**`mute_fade_ms`** — Fade applied to the AUDIBLE shots that border a silenced one, so audio doesn't cut to digital silence with a click. The silenced shots keep NO original audio at all -- fading the muted shot itself would leave this many ms of the gibberish audible at each end of every muted shot.

### Consistency guards

| field | | type | constraints |
|---|---|---|---|
| `subject_count_guard` |  | choice | `auto`, `on`, `off` |
| `anatomy_guard` |  | choice | `off`, `auto`, `on` |
| `solidity_guard` |  | choice | `off`, `auto`, `on` |
| `motion_guard` |  | choice | `off`, `auto`, `on` |
| `contact_guard` |  | choice | `off`, `auto`, `on` |
| `lock_restraints` |  | BOOLEAN | default `True` |
| `auto_wardrobe` |  | BOOLEAN | default `True` |
| `auto_props` |  | BOOLEAN | default `True` |
| `prevent_nudity` |  | BOOLEAN | default `True` |
| `exposed_terms` |  | STRING | **input socket**, multiline |

**`subject_count_guard`** — Anti-duplication: prepend an explicit subject count to each shot ("Exactly two people in this shot, no duplicates, no other people in frame"). Character duplication gets much more likely BELOW the model's native 768 short edge -- fewer pixels per subject pushes the sample out of the training distribution and the figure gets tiled. A LoRA causes it too: a distilled LoRA fixes composition (including how many people are in frame) in its first step or two, so it duplicates even at native size -- there the count is moved to the FRONT of the prompt so it binds before the scene. 'auto' = on when the short edge is under 768 OR a LoRA is applied, and also on ANY shot holding two or more people -- multi-figure frames are where duplication happens even at native size; 'on' always; 'off' never.

**`anatomy_guard`** — State each person's limb COUNT positively, to stop spare arms, duplicated hands and the third leg. H3 is CFG-free at cfg 1, so a NEGATIVE prompt is never evaluated -- 'extra limbs' in a negative does nothing. Naming a number gives the model a target instead; negating one only puts the word in the prompt. Added per-shot and only where people are actually present, never in the anchor (anchor body words are what burn a face into every opening frame). 'auto' = on below 768 short edge OR when a LoRA is applied, and also on ANY shot holding two or more people -- spare limbs are grown where bodies meet and move together. Costs ~90 tokens on shots with people.

**`solidity_guard`** — Keep bodies from passing through objects. States that the solid things in the shot occupy space and that bodies stop at surfaces. Stated POSITIVELY, and it has to be: H3 is CFG-free at cfg 1, so a negative prompt is never evaluated, and 'does not walk through the wall' in the positive names walking through a wall -- a mention is a presence cue. It says what bodies DO instead: stop at the surface, rest on the floor, press against what they touch, go around the furniture. 'auto' speaks only when the shot actually names something solid (walls, doors, tables, stairs, vehicles, crates, trees...), reading BOTH the beat and the identity block, since the set is usually described in the anchor. 'on' states it every shot. Only ever applied to a shot with someone in it -- a body is needed before one can pass through anything.

**`motion_guard`** — Stop poses being reached without the frames in between -- the head arriving at a new angle with no path to it (a 'neck snap'), a body teleporting between two positions. What is missing in a snap is the PATH, not the pose, so the path is what gets stated: movement travels through every position on the way, at one steady speed, the neck following the shoulders and the shoulders following the hips. Positive, because at cfg 1 H3 is CFG-free and the negative is never evaluated -- and 'the head does not snap round' in the positive names a head snapping round. 'auto' speaks only on a beat that actually moves someone (turns, looks, walks, leans, reaches... and the high-jerk ones -- struggles, pulls, twists, writhes -- where a limb most often arrives without its path), since a beat where nobody changes orientation has no path to describe. 'on' states it every shot. Names nobody, so it adds no second reference to anyone already in frame. A snap right after a cut is a different thing: that is the model leaving the keyframe pose. handoff_offset helps there.

**`contact_guard`** — Keep two bodies in contact correctly aligned -- any position, not a list of named ones. Position-agnostic on purpose. The model already knows more position names than a dictionary could hold; what it gets wrong is the GEOMETRY, so the geometry is what gets stated, and these hold for every arrangement: - OWNERSHIP: each person keeps their own head, two arms and two legs, each joined to the body it belongs to. Overlapping bodies is exactly when an arm gets reassigned to the wrong torso. - SEPARATION: they meet at the surface of the skin, each keeping its own volume, rather than passing into one another. - STABLE ROLES: whoever is above stays above, below stays below, behind stays behind, for the whole shot and from every camera angle. Positions morph mid-shot because nothing fixes them. - SUPPORT: weight rests on whatever is holding it, and the two bodies stay in proportion. Needs TWO people in the shot -- one body cannot be misaligned against another, and saying otherwise would invite a second person in. 'auto' fires on a contact cue in the beat; 'on' states it whenever two or more people are present. Describe the arrangement in the beat itself in RELATIVE terms (who is above, behind, facing whom) rather than by a position name alone -- this guard holds a stated arrangement together, it cannot infer one you did not state.

**`lock_restraints`** — Physical restraints stay ON until something explicitly removes them. Handcuffs, shackles, manacles, fetters, irons, gags, blindfolds, harnesses, leashes, plus qualified forms like 'ankle chain' or 'leather wrist straps'. Without this they come off like any garment, and often by ACCIDENT -- 'steps out of her jacket and the chain falls away' removed the ankle chain as a side effect of a jacket beat. To take one off, say so directly: 'wardrobe: Mara -= handcuffs'. Bare 'chain', 'collar', 'strap' and 'belt' are NOT treated as restraints; they are jewellery, a shirt part, a dress part and a garment at least as often.

**`auto_wardrobe`** — Read clothing REMOVALS straight from your beat prose -- 'she takes off her jacket' drops the jacket with no 'wardrobe:' line needed. Safe: only fires on items the character is already wearing, so 'the plane takes off' does nothing. Additions/swaps still use 'wardrobe: += ...' (which overrides). Turn OFF to control wardrobe only via explicit 'wardrobe:' lines.

**`auto_props`** — Carry OBJECTS across shots. Each shot is a separate generation, so 'the van' in shot 2 has no antecedent -- nothing in that prompt describes a van, and the model invents one, which is how a second van appears while the first is still in frame. With this on, an object introduced indefinitely ('a white van') is bound on its first definite reference in any later beat ('the van' -> 'the same white van') and a short clause pins it to the previous shot: one van only, no second van. Only the FIRST mention per shot is expanded, quoted dialogue is never rewritten, worn garments are excluded (they have the wardrobe channel), and frame/body nouns (the ground, the light, the hand) are never carried.

**`prevent_nudity`** — Never let the prompt ASSERT that a body is bare. A removal still happens either way -- this gates the sentence, not the garment. Deleting the last item covering a zone only leaves it undescribed, and a video model's default is a clothed person, so it covers what nobody described. With this OFF the node states the state outright ('bare below the waist') and keeps stating it until something covers that zone again, which is what makes a strip actually stick. ON is the safe default; turn it OFF only when nudity is intended. Either way info reports which zone a removal left uncovered.

**`exposed_terms`** — What a stripped body zone is CALLED, per character, so it persists automatically instead of being typed into every beat. Same syntax as character_memory -- a PRONOUN covers everyone who declares it, a NAME overrides it, and a trailing 'upper' targets the chest. For example -- 'she = visible vulva, mvagina' / 'he = visible penis, mpenis' / 'Mara upper = bare breasts'. Once a removal empties that zone the phrase is stamped into every later shot that person is in, and clears by itself when something covers the zone again. Put LoRA trigger words here too. Requires prevent_nudity OFF; empty falls back to 'bare below the waist'.

### Reference images

| field | | type | constraints |
|---|---|---|---|
| `ref_mode` |  | choice | `where tagged`, `first shot`, `every shot`, `every shot + handoff ref` |
| `ref_image_size` |  | choice | `match`, `max` |
| `ref_noise_aug` |  | FLOAT | default `0.999`, range `0.5`–`1.0` |

**`ref_mode`** — Which shots the ref_image inputs condition. A shot carries EITHER references or the last-frame handoff, never both. 'where tagged' (default): write <Picture 1> in the beat where that character appears and ONLY that shot gets the reference -- every other shot keeps its handoff. This is the precise option: the other modes go by shot NUMBER and are blind to who is actually in the shot, so a character who first appears in shot 2 gets nothing while an empty establishing shot 1 gets a portrait pushed into it. Tags are renumbered per shot, so <Picture 2> alone still resolves. With refs connected but no tags anywhere, falls back to first shot rather than silently doing nothing. 'first shot' / 'every shot' / 'every shot + handoff ref' go purely by position. Ignored when no ref_image is connected.

**`ref_image_size`** — How large each reference is encoded. 'match' scales it down to the generation's pixel area -- a reference then costs about one frame per step. 'max' uses the reference pipeline's 2048 short edge for the best identity fidelity, but reference rows are re-attended EVERY step of EVERY ref-conditioned shot, so on a long chain it is several times slower. Neither ever upscales a small reference.

**`ref_noise_aug`** — How CLEAN each reference is presented to the model. 0.999 (H3's own default) hands it a finished, noise-free image -- which invites the model to REPRODUCE the reference in the opening frames instead of just taking an identity from it. Lower values blend the condition with noise and label it as approximate, so it informs the face without being copied: try 0.95, then 0.90. Too low (below ~0.8) and the reference stops holding identity at all. Applies ONLY to ref-conditioned shots -- the last-frame handoff is never weakened, or continuity would break.

### Decode and upscale

| field | | type | constraints |
|---|---|---|---|
| `decode_tile_frames` |  | INT | default `0`, range `0`–`128` |
| `decode_tile_size` |  | INT | default `0`, range `0`–`1024` |
| `upscale` |  | choice | `off`, `rtx`, `model`, `lanczos` |
| `upscale_model` |  | choice | `none`, `GFPGANv1.4.pth`, `RealESRGAN_x2.pth`, `RealESRGAN_x4plus.pth` |
| `upscale_target_short_edge` |  | INT | default `0`, range `0`–`4096` |
| `upscale_batch` |  | INT | default `4`, range `1`–`64` |

**`decode_tile_frames`** — Temporal tiling for the VAE decode (tile_t). 0 = ComfyUI default, which expands the WHOLE clip at once -- the single largest allocation in a run, and the usual point where a big checkpoint tips into shared memory. Try 8-16 if you spill during decode rather than sampling. Lower = less peak VRAM, slightly slower.

**`decode_tile_size`** — Spatial tile size for the VAE decode (tile_x/tile_y). 0 = ComfyUI default. Try 256 on a tight card at 1344x768.

**`upscale`** — Optional post-pass on the finished frames. 'rtx' = NVIDIA RTX Video Super Resolution (Tensor Cores -- fastest and best for video; needs the Nvidia_RTX_Nodes_ComfyUI pack, falls back automatically if absent). 'model' = a Real-ESRGAN/UltraSharp upscale model from upscale_model. 'lanczos' = plain resize. All of these ENHANCE/ENLARGE; for true detail reconstruction from a low-res render, use a separate LTX 2.3 upscale pass.

**`upscale_model`** — Upscale model from models/upscale_models (used when upscale = model).

**`upscale_target_short_edge`** — Fit the result's short edge to this many px (0 = keep the model's native factor / no resize). E.g. generate 512 fast, set 768 to land at native size.

**`upscale_batch`** — Frames per chunk for the model upscale (lower = less VRAM, slower).

### Overlays

| field | | type | constraints |
|---|---|---|---|
| `watermark_text` |  | STRING | default `''` |
| `watermark_position` |  | choice | `bottom-right`, `bottom-left`, `bottom-center`, `top-right`, `top-left`, `top-center`, `center` |
| `watermark_size` |  | FLOAT | default `4.0`, range `0.5`–`40.0` |
| `watermark_opacity` |  | FLOAT | default `0.75`, range `0.0`–`1.0` |
| `watermark_margin` |  | FLOAT | default `3.0`, range `0.0`–`25.0` |
| `intro_text` |  | STRING | **input socket**, multiline |
| `intro_position` |  | choice | `center`, `lower-third`, `top-center`, `bottom-center` |
| `intro_seconds` |  | FLOAT | default `3.0`, range `0.0`–`30.0` |
| `intro_fade` |  | FLOAT | default `0.6`, range `0.0`–`10.0` |
| `intro_size` |  | FLOAT | default `9.0`, range `0.5`–`40.0` |
| `overlay_font` |  | STRING | default `'arial.ttf'` |
| `overlay_stroke` |  | INT | default `0`, range `0`–`20` |

**`watermark_text`** — Composited with PIL onto every finished frame -- NOT rendered by the model and NOT added to the prompt. White glyphs on a transparent layer, alpha-blended over the video, so only the letters land on the picture. Applied AFTER any upscale, so the text is crisp at final resolution. Leave empty for none.

**`watermark_size`** — Cap height as a percentage of the frame's SHORT edge, so the mark keeps its apparent size across portrait and landscape presets alike.

**`watermark_opacity`** — Multiplies the white text alpha. 1.0 = solid white; 0.75 reads as a watermark without burying the picture under it.

**`watermark_margin`** — Inset from the frame edge, as a percentage of the SHORT edge.

**`intro_text`** — Title composited over the OPENING frames -- white on transparent, so the first shot plays underneath it rather than being replaced by a card. Multi-line is centered as a block. Holds for intro_seconds, then fades out over intro_fade. Also PIL, never the model.

**`intro_seconds`** — How long the title stays at full opacity before the fade starts.

**`intro_fade`** — Linear fade-out length after the hold. 0 = hard cut.

**`intro_size`** — Title cap height as a percentage of the frame's SHORT edge.

**`overlay_font`** — TrueType font for BOTH overlays: a bare name resolved against the system font folder (arial.ttf, arialbd.ttf, segoeui.ttf) or a full path to a .ttf/.otf file. Falls back to the first font that loads if this one fails.

**`overlay_stroke`** — Black outline thickness in pixels around the white text. 0 keeps it pure white as asked; 2-3 makes it survive a bright sky or a white wall.

### Preview

| field | | type | constraints |
|---|---|---|---|
| `plan_only` |  | BOOLEAN | default `False` |

**`plan_only`** — Preview the shot split WITHOUT rendering. Uses THIS node's own settings (no second node, no duplicate entry): returns the plan in 'info' and the shots/frames/seconds outputs near-instantly. Turn off to render for real.

### Outputs

| slot | name | type |
|---|---|---|
| 0 | `images` | IMAGE |
| 1 | `audio` | AUDIO |
| 2 | `info` | STRING |
| 3 | `script` | STRING |
| 4 | `frames_per_shot` | INT |
| 5 | `total_frames` | INT |
| 6 | `shots` | INT |
| 7 | `video_seconds` | FLOAT |
| 8 | `fps` | FLOAT |
| 9 | `fps_int` | INT |
| 10 | `latent` | LATENT |
| 11 | `soundscape` | STRING |

Outputs are only ever **appended**. A workflow stores an output link by slot index, so inserting one would silently re-target every wire after it.

<!-- AUTOGEN:FIELDS END -->

One node. You set just two things:

1. **prompt** — this is the `integrated_multimodal_description` (the visual +
   action timeline). The first paragraph is the look/character kept across the
   whole video; each later paragraph is a scene beat. Put dialogue and any
   "lips closed / not speaking" beats here. (Blank line between paragraphs.)
2. **total_seconds** — how long the finished video should be.

The node reads the paragraphs, spreads `total_seconds` evenly across them,
splits that into shots that fit both H3's 15s ceiling and your VRAM, chains
each shot from the previous one's last frame, trims the seams, and outputs the
finished **images** + **audio** (plus **info** and the **script** it built).
Optionally it also upscales the result and composites a **watermark** and/or an
**intro title** onto the finished frames — see [Text overlays](#text-overlays-watermark-and-intro-title-overlaypy).

Also on the node: a **resolution** dropdown — a **native** 768-short-edge tier
per ratio (best detail) and a **fast** 512-short-edge tier per ratio (for the
generate-low-then-upscale workflow), plus a **balanced** 640 tier. Every option is a
valid multiple of 32 — there's no custom width/height to snap or mis-type. The
fast tier is ~4× fewer pixels, so it renders faster, frees VRAM, and
(because the length budget is resolution-aware) unlocks **longer shots** — on a
16GB card a 512 shot reaches the full 362f/15s where native only manages ~243f/
~10s. Best for close/medium shots; H3 distorts faces on *wide* shots at any
resolution, so keep faces reasonably large in frame. Pair the fast tier with an
external LTX 2.3 upscale pass (with correct sigmas) to bring finals back to high
resolution at near-native quality. Also: seed (with control-after-generate),
steps, cfg, sampler, scheduler. Everything else (fps, seam trim, **handoff offset**,
VRAM headroom, explicit anchor, ambient `global_soundscape`,
`non_diegetic_music`, `character_memory`) is optional with sensible defaults.

**Shot length is sized from what each beat stages** (`per_beat_length`, **on by
default**). A beat's time is ~2s of setup plus ~2.5s per action clause, or its
spoken line, whichever is longer — so "she takes off her jacket and drops it on
the bench" gets ~7s while a three-part beat gets more. `shot_seconds` and the VRAM
budget are the **ceiling**, not the length of every shot.

This exists because the opposite is a real failure, not an aesthetic one. A 3-second
action in a 12-second shot leaves nine seconds the model was told nothing about,
and it fills them by repeating or **reversing** the action — which is why clothing
came off and went back on. The estimate deliberately leans **short**: an unfinished
action is continued by the next shot from the handoff frame, while an overlong shot
is unrecoverable.

Override any single beat with `seconds: 8` on its own line inside that paragraph —
that wins over the estimate, over `shot_seconds`, and over the toggle, and it is
honored down to H3's 5-frame minimum. Turn `per_beat_length` **off** to give every
shot the full ceiling; `info` then warns about any beat too thin for the length it
got, since that is the case the node cannot size for you.

*Note for workflows saved before this:* ComfyUI stores widget values, so an existing
node still holds the old `per_beat_length` value and must be ticked by hand.

**A short `shot_seconds` used to be ignored.** Values below ~5.2s were raised to the
124-frame budget floor, so 1s, 2s, 3s and 4s all rendered identically. That floor is
the fallback for what the node *guesses*, and must never override what you asked for;
an explicit request is now honored down to H3's 5-frame minimum.

**`info` not updating between runs.** The node had no `IS_CHANGED`, so ComfyUI keyed
its cache on the inputs alone and re-queueing with unchanged widgets returned the
previous outputs — `info` among them. That is actively misleading here, because both
`info` and the chosen shot length depend on live free VRAM, which is not an input.
`plan_only` now always recomputes (it is near-instant, and a stale plan is worse than
none); a real render stays cacheable, so change the seed or any widget to force one.

**Every shot coming out ~5s (the 124f floor).** The length budget is card capacity
minus measured weight size minus `vram_headroom_gb`. If that came out at or below
zero it returned the internal 124-frame floor immediately — so a checkpoint that
*fits* but leaves less than the headroom (e.g. ~14.6GB of weights on a 16GB card)
pinned **every** shot to ~5.2s no matter what the beats asked for, and dropping to the
fast 512 tier changed nothing, because the early return skipped the resolution scaling
entirely. That case now runs the normal arithmetic: ~14.6GB weights on a 16GB card
gives 226f (~9.4s) instead of 124f (~5.2s). Weights that genuinely exceed the card
still floor at every resolution — that one is real, and no shot length fixes it.

A deficit is weights, not latent, so it is deliberately *not* scaled by resolution;
in that regime the tiers report the same length. Resolution only buys frames when
there is a surplus to scale. And when the floor is hit for real, `info` now says so
outright — which knob moved it, and by how much — instead of quietly handing back 5s
shots.

**Beat counts can no longer be collapsed by a setting.** `beat_split` used to offer a
strict `blank line` mode, and it was the only control on the node that could silently
lose beats: six beats typed as two blocks of three rendered as **two** shots, with
nothing in `info` to say why (the split note is only written when a paragraph is
actually split). That option is **removed**. `beat_split` now offers `auto` and
`each line`, which produce identical results — neither can drop a beat — and a
workflow that still stores `blank line` reads as `auto`. The widget itself is kept in
place rather than deleted, because removing a widget shifts every stored value after
it in already-saved graphs.

Nothing else on the node changes the beat count. The only other control that alters
it is `anchor_override`: leave it empty and paragraph 1 is consumed as the anchor
(see below) — intended behaviour, and now guarded against the case where that
silently deletes a shot.

**Repeat name mentions are collapsed automatically.** Naming one person twice in a
single beat — "Kristy finds Dan… she walks over to **Dan**" — is the most reliable
way to make H3 render that person twice, and binding the description once doesn't
fix it, because the bare repeated name is what duplicates. The node now rewrites the
second and later mentions to the right pronoun by grammatical case: subject
(`and Dan takes it` → `and he takes it`), object (`over to Dan` → `over to him`),
possessive (`Kristy's toolbox` → `her toolbox`). Write the beats however reads
naturally; you don't have to police your own repeats.

It only fires where the result is unambiguous. The person's pronoun must be known —
declared in their sheet (`Dan = he, …`) or inferable from a gender word — and no one
else in the shot may share it, or "he" couldn't be traced back. Words inside double
quotes are never touched, so a name in a spoken line (`"Kristy, over here!"`) stays
exactly as written. The first mention always survives, so the description still has
a name to bind to.

**A first paragraph that is really a beat is now rescued, not eaten** — including with
no `character_memory` set, which is the common case and the one the first version of
this guard could not see. A first paragraph is kept as a beat when every sentence in
it stages an action ("Kristy walks around in a garage looking for engine parts.") and
its subject recurs later in the prompt, or when it strips to nothing. A scene/style
anchor never matches; neither does a mixed paragraph that still carries scene text.
Without
`anchor_override`, paragraph 1 becomes the identity anchor — so three paragraphs
render as two shots. That is by design, but it turns into silent data loss when
paragraph 1 is itself an action beat naming a tracked character ("Kristy walks
around in a garage looking for engine parts."): the anchor is stamped on every
shot, so any sentence naming a tracked character is stripped out of it to stop
that character being introduced twice — which leaves nothing at all. You lose the
shot *and* the only scene text you wrote. The node now detects that exact case
(anchor strips to empty), keeps the paragraph as a **beat**, and puts a WARNING at
the front of `info`. It can't misfire on a normal anchor: a `wardrobe:` line, or
any prose that survives the strip, is left alone, and with no `character_memory`
nothing is tracked so nothing is stripped. The real fix is still to put the
setting and style — **with no character names** — in `anchor_override`.

**Character / wardrobe memory.** The keyframe handoff only carries what the
*last frame* showed — so if a shot ends zoomed in on the face, the pants aren't
in that frame and the next shot reinvents them. The fix is to keep wardrobe in
one **mutable text channel** that's re-stamped into every shot, independent of
framing — and, crucially, to keep clothing **out of the permanent anchor prose**,
because the anchor is immutable and would re-assert a garment you're trying to
remove.

The rule: the first paragraph's prose is **permanent identity only** (hair,
face, build). Clothing goes on a `wardrobe:` line — either in the first
paragraph or in the dedicated `character_memory` field (that field wins if both
are set). Example first paragraph:

    Maya: short silver hair, scar over left eyebrow, athletic build. Cinematic.
    wardrobe: grey cargo shorts, red flight jacket, black boots

**Automatic removal from your prose (no `wardrobe:` line needed).** With
`auto_wardrobe` on (the default), the node reads clothing *removals* straight
from a beat's own action text — "she takes off her jacket," "he sheds his coat,"
"she peels off her gloves" — and drops that item, with no directive to type.
It's safe by design: a removal only fires on an item the character is already
wearing, so non-garment phrases like "the plane takes off down the runway"
match nothing and change nothing. If a name precedes the action ("Maya takes
off…"), only that person is affected.

**The keyframe carries the start state; the prompt carries the end state.** From
the removal shot onward the garment is gone from the person's description, and
that shot alone states the change outright: *"She starts this shot wearing the red
jacket and takes it off during the shot; by the last frame the red jacket is off
and she is not wearing it. The motion runs one way only: the clothing comes off
and is never put back on, never re-worn, and the action never plays in reverse."*
It uses the person's **pronoun**, never a bare name (a bare name re-introduces
them, and re-introducing someone is what makes the model render them twice).

Both halves of that exist for a reason, and both were bugs first:

- **Listing the garment as worn in the shot that removes it made the video play
  backwards.** A removal is the one wardrobe change whose motion is symmetric —
  the same frames reversed are a person putting the garment *on*, and both
  readings satisfy "takes off her red jacket" equally. When the shot's own
  description still said "wearing a red jacket", backwards was the reading that
  matched the description, so the removal rendered in reverse and the jacket came
  back. The start state does not need the description: for every shot after the
  first, the handoff keyframe already shows the garment still on.
- **Naming the garment in a LATER shot put it back on.** To a video model a
  mention is a presence cue and a negation is a weak one, so "she is no longer
  wearing the red jacket" in the following shot was itself enough to re-dress her.
  No shot after the removal names the item at all now — they simply describe what
  she *is* wearing.

Plural garments get plural agreement ("the navy overalls **are** off, he is not
wearing **them**"), because the clause is read literally by the text encoder.

This works whether the garment lives in the wardrobe channel or **only in your
anchor prose** ("A woman in a red jacket and a man in a black t-shirt"): the
phrase is scrubbed from the anchor so it can't re-apply itself forever. The
garment's **head noun** is what gets matched — "takes off her red jacket" reads as
*jacket*, not *red* — because matching the adjective used to pull out "A woman in
a red" and scrub the **person** out of the scene while leaving the jacket behind.

Additions and swaps still use an explicit one-token line (`wardrobe: += mirrored
sunglasses`), which also overrides the auto-detection. Turn `auto_wardrobe` off
to drive wardrobe purely through explicit `wardrobe:` lines. Either way, the
`script` output shows the resolved wardrobe for every shot, so you can always
see exactly what each shot inherited.

**No gibberish / no mouths moving before dialogue.** H3 will animate — and
vocalize (as babble) — a mouth on any shot it thinks involves speech, which is
why action shots leading up to a line often show moving mouths or gibberish
audio. With `auto_silence_nonspeech` on (the default), any beat with **no
scripted dialogue** automatically gets a "lips closed, no dialogue" clause, so
mouths stay shut and silent until real speech. A beat counts as dialogue only if
it contains **double-quoted** words (or an `<d>…</d>` tag) — so to make someone
speak, put the words in double quotes: `She says, "Tower, ready for departure."`
Those shots are left alone; every other shot is silenced. Turn the toggle off to
manage lip state yourself. Pair it with `handoff_offset` if a dialogue shot still
hands a mid-word open mouth to the next shot.

Inside a dialogue shot the silence is now **per person**: a spoken line is
attributed to whoever introduced it (`Jon says: "…"`, or `…" said Jon`), and
everyone ELSE bound in that shot gets the lips-closed state by pronoun. Before,
one quoted line freed every mouth in frame, and non-speakers mouthed along.
A quote with no attributable speaker frees nobody by guesswork, and a
scare-quoted single word (she gave him a "look") — which still flips the shot to
speaking — is reported in `info` so you can drop the quotes or attribute the line.

Silencing now covers **both channels**. The lips-closed clause constrains the
picture only; H3 builds audio from its own fields, and an **absent**
`overall_soundscape:` leaves that branch unconditioned — which is exactly when it
fills a silent shot with speech-like babble. Every silenced shot therefore also
carries a soundscape line that says *no voices, no speech, no talking* outright.
If you supply a `global_soundscape`, it is kept and the no-voice constraint is
appended to it on those shots only.

**`mute_nonspeech_audio` is ON by default** — the deterministic backstop. Prompt
and soundscape clauses *ask* H3 not to vocalize; muting guarantees it by zeroing
the audio of every shot with no quoted line (neighbouring audible shots get a
short `mute_fade_ms` ramp so nothing clicks). The trade-off is real and `info`
reports it on every run: that shot's generated **ambience goes with the babble**,
so lay a continuous ambient bed under the video in post — or untick the widget to
keep H3's own sound and rely on the prompt-side silencing alone.

**A character who leaves comes back when you name them.** Naming a departed
character in a later beat is intent to have them present, so they return with their
description intact. A *pronoun* deliberately cannot do this -- "she waves" after
someone left is ambiguous -- and `exit: Name` sends them out again if you meant it.

This matters more than it sounds: a departed character keeps their NAME in the beat
but loses their description, so the beat reads "Mara opens the crate" with Mara
undescribed while everyone else still has their sheet. The described character then
absorbs the action.

Related: coming **out of** a place is arriving, not leaving. "Mara steps out of the
barn" no longer marks her as departed -- only "walks out", "out of frame", "off
screen", "leaves", "drives off" and the like do.

To change or remove an item mid-chain *explicitly*, put a `wardrobe:` line **inside** the
beat where it changes (not as its own paragraph). You have two ways, and you do
**not** have to restate the whole outfit:

- **One-token edit (easiest):** `wardrobe: -= jacket` removes any item matching
  "jacket"; `wardrobe: += mirrored sunglasses` adds one. Everything else the
  person is wearing carries forward untouched.
- **Full replace:** `wardrobe: grey cargo shorts, black tank top` sets the whole
  outfit (use when several things change at once).

Either way the change is sticky from that shot onward, and because clothing
lives only in this channel (never the anchor prose), a removal actually stays
gone.

**Two or more people.** Name each person, separated by `;`, and put their
**identity and clothing together** in the named channel:

    wardrobe: Maya = silver hair, scar, grey cargo shorts, red flight jacket; Jon = bald, bearded, navy overalls

Then use their names in the beats. The node binds each person's description
**inline at the single place their name appears** — `Maya (silver hair, grey
shorts, red jacket) greets Jon (bald, navy overalls)` — so each name occurs once
per shot. This matters: the earlier approach emitted a separate `Maya: clothes`
sentence *and* left `Maya walks…` in the beat, and two mentions of a name make
text-to-video render the person **twice**. Binding inline fixes that. A tracked
person you don't name in a beat is left out of that shot (not forced in).

Critical rule to avoid duplicates: **keep names out of the anchor.** The anchor
is stamped on every shot, so a name there is an extra mention that re-triggers
duplication. Anchor = scene and style only (no people); all per-person identity
and clothing go in the named channel; who's in a given shot is decided by which
names you use in that beat.

**Writing with pronouns (recommended for two people).** To keep prose natural
and cut duplication further, you can refer to people as "she"/"he"/"they" in the
beats instead of repeating names — but then declare each person's pronoun in
their sheet so the node knows who's who:

    wardrobe: Maya = she, silver hair, scar, grey shorts, red jacket; Jon = he, bald, navy overalls

Now "She takes off her jacket" attributes the removal to Maya, and each person's
description binds at the pronoun (`She (silver hair, grey shorts…) greets him
(bald, navy overalls)`) — no names, no doubling, and removals still stick. In a
one-person scene any pronoun maps to that person automatically (no declaration
needed). Two people of the **same** pronoun can't be told apart by "she" alone —
name them in the beat where it matters, or use an explicit `wardrobe: Maya -=`
line. The pronoun token itself is used only for resolution; it never shows in
the description.

Edit one at a time — `wardrobe: Maya -= jacket` drops Maya's jacket and
leaves Jon exactly as he was. The name can be written with or without a colon
(`Maya -= jacket` and `Maya: -= jacket` are the same, and you can seed with
`Maya: grey shorts, red jacket` too — whichever reads naturally). Names you
don't mention are untouched; a person with all items removed simply drops out of
the sheet. This is the node-side of multi-person; H3 itself is weakest at
multi-subject identity binding, attribute cross-wiring, and multi-speaker audio
— keep people visually distinct, add spatial cues ("Maya on the left"), and
prefer one speaker per shot for clean dialogue.

**Steps:** default is **20**, the right value for the base H3 model
(res_multistep + simple). Only drop to 6–8 if you have a working 4-step
distill/turbo LoRA or a low-step MXFP8 checkpoint — on the bare base model,
low steps are the main cause of soft/under-formed frames (faces worst).

**Handoff offset:** if chained shots open with moving or "talking" mouths, set
`handoff_offset` to 2–4. The node then ends each shot that many frames early
and hands *that* frame to the next shot instead of the literal last frame
(which can catch a mid-word open mouth), trimming the matching audio tail so
A/V stays aligned. 0 = use the last frame (original behavior).

**Audio, the three H3 sections.** You don't type any field labels — the node
assembles them:
- **Visual + dialogue** → the **prompt** box. It *is* the
  `integrated_multimodal_description`. Speech, lip state, and diegetic sound a
  character makes/hears all go here, in the beats.
- **Ambient bed** → the **global_soundscape** widget. Appended to every shot as
  `overall_soundscape:` — environmental sound only (rain, room tone, engines).
- **Score** → the **non_diegetic_music** widget. Music is **opt-in**: leave it
  blank and the node writes `non_diegetic_music: N/A` on every shot, so H3 adds
  no score (a blank field otherwise makes H3 improvise its own music). Fill it in
  to request a specific score (instrumentation, tempo). Music a character plays
  or hears is diegetic and belongs in the prompt beat.

Both audio widgets are global (stamped on every shot) so a bed/score stays
consistent across the whole video. Leave either blank to omit that section.

**VRAM:** it measures the model's size (via ComfyUI's own accounting, so
quantized checkpoints report correctly), picks the largest shot length your
card can attempt, and on a caught out-of-memory quietly backs off (tiled
decode → lower resolution) instead of crashing.

Forcing `shot_seconds` now still respects that budget: a forced length that
won't fit is **clamped down** to what fits and the clamp is reported in `info`
(e.g. 15s on a 16GB card clamps to ~10s/243f). Set `allow_oversize_shots` to
override and honor the requested length anyway — but then the render may spill
into system RAM (slow) or OOM.

Note on the reactive backoff: it can only fire on a *caught* OOM. If NVIDIA's
"CUDA – Sysmem Fallback Policy" is on (the default on Windows), an over-budget
run silently spills VRAM into system RAM instead of raising, so the backoff
never triggers and you get a slow, over-cap run. Set that policy to *Prefer No
Sysmem Fallback* (per-app is fine) so over-budget shots raise and the backoff
can do its job. The predictive clamp above is the guard when the fallback is on.

Between-shot cleanup (`cleanup_between_shots`, on by default): after each beat
the node moves that shot's decoded video+audio to system RAM and runs a full
VRAM+RAM purge (Python GC, ComfyUI's aggressive cache clear, and the CUDA
allocator's `empty_cache` + `ipc_collect`). Without this, every shot's frames
stay resident on the GPU and accumulate across the chain — the main reason a
long (12-shot) run OOMs partway through even when a single shot fits. The
handoff keyframe is kept off-GPU too and re-encoded next shot. Leave it on for
16GB; turn it off only on a large card if you want to skip the small per-shot
cleanup cost. (Trade-off: the finished frames now accumulate in system RAM
instead — expected, since the full video has to live somewhere — so for very
long high-res renders, watch RAM rather than VRAM.)

## Built-in upscale (optional post-pass)
Set `upscale` to enable a post-generation pass on the finished frames:
- **rtx** — NVIDIA **RTX Video Super Resolution**, running on RTX Tensor Cores.
  Fastest option by a wide margin (NVIDIA claims ~30× vs other local upscalers)
  and generally cleaner on video than UltraSharp-class models. Requires the
  `Nvidia_RTX_Nodes_ComfyUI` pack (`git clone https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI`
  plus `nvidia-vfx` from its requirements; it may not appear in ComfyUI Manager).
  If the pack isn't installed the node falls back automatically — it never breaks
  a render. On 16GB, keep `upscale_batch` low: long clips can exhaust system RAM.
- **model** — runs a Real-ESRGAN / UltraSharp-class upscale model from
  `models/upscale_models` (pick it in `upscale_model`), chunked with cleanup so
  a long clip doesn't OOM. Real per-frame sharpening/detail.
- **lanczos** — plain high-quality resize to `upscale_target_short_edge`
  (enlarges; adds no new detail).

`upscale_target_short_edge` fits the result's short edge to a target (0 = keep
the model's native factor). Typical use: generate on the **fast 512 tier** for
speed/length, then set the target to 768 to land back near native size.

**Honest ceiling:** this pass sharpens and enlarges; it does **not** reconstruct
video detail the way a second-model re-generation does. For true near-native
recovery from a low-res render, a separate **LTX 2.3** upscale pass is the gold
standard (it re-generates detail and holds lip-sync) — but it needs its own
model loaded, which is why it lives outside this node rather than in it. Use the
built-in model upscale for a quick, self-contained quality lift; use an external
LTX 2.3 pass (with correct sigmas) when you need the best possible result.

## Text overlays: watermark and intro title (`overlay.py`)
Two optional PIL overlays, **composited onto the finished frames — never asked of
the model and never added to the prompt.** H3, like every video diffusion model,
renders text as plausible-looking letterforms that drift, warp and re-spell
themselves frame to frame; a watermark that changes shape every frame is worse
than none. Compositing gives pixel-identical text on every frame at zero sampling
cost, and keeps the words out of the prompt where they'd otherwise steal
conditioning from the actual shot.

Both draw **white glyphs on a fully transparent layer** that is alpha-blended over
the video, so only the letters land on the picture and the image shows through
everywhere else.

- **`watermark_text`** — stamped on **every frame**. `watermark_position` (7
  anchors: the four corners, `top-center`, `bottom-center`, `center`),
  `watermark_size` (cap height as a **% of the short edge**, default 4.0, so the
  mark keeps its relative size at any resolution, ratio or upscale factor),
  `watermark_opacity` (default 0.75 — reads as a watermark without burying the
  picture), `watermark_margin` (inset as a % of the **short** edge).
- **`intro_text`** — a title over the **opening of the finished video**, not a
  replacement card: the first shot plays underneath it. Multi-line is centered as
  a block. It holds at full opacity for `intro_seconds` (default 3.0), then
  linearly fades over `intro_fade` (default 0.6; 0 = hard cut). `intro_position`
  offers `center`, `lower-third`, `top-center`, `bottom-center`; `intro_size`
  defaults to 9.0% of the short edge.
- **Fits every preset automatically.** Both sizes are measured from the **short
  edge**, and the block is word-wrapped and then shrunk until it sits inside the
  margins. Sizing from the *height* meant a portrait canvas (9:16, 3:4) drew the
  text ~1.75× larger on the canvas with the least room for it, and anything that
  ran past the frame was silently clipped by Pillow — no error, no note, just
  missing characters. A long title now wraps onto as many lines as it needs and
  reads the same at 512×896 as at 1536×672.
- **`overlay_font`** — TrueType face for **both** overlays: a bare name resolved
  against the system font folder (`arial.ttf`, `arialbd.ttf`, `segoeui.ttf`) or a
  full path to a `.ttf`/`.otf`. If it won't load, the node falls back through
  Arial → Segoe UI → DejaVu Sans → Liberation Sans, and finally to PIL's bitmap
  default (which ignores size — ugly, but never fatal).
- **`overlay_stroke`** — black outline in pixels around the white text. 0 keeps it
  pure white; **2–3 makes it survive a bright sky or a white wall.**

Leave a text field empty to skip that overlay; both are off by default.

**Applied last, after any upscale**, so glyphs are rasterized at the final pixel
size instead of being interpolated up along with the picture. Two consequences
worth knowing: the overlays run once on the **whole concatenated video**, so the
intro sits over the opening of the finished piece rather than the top of every
shot; and because it happens post-upscale, the text is crisp even on a
fast-512-then-upscale workflow.

Everything here is **best-effort by design** — any failure (missing Pillow, an
unloadable font, a bad position) returns the frames untouched and writes a note
into `info`, because a cosmetic overlay must never destroy a finished render.
Blending is chunked 64 frames at a time and cropped to the text's tight bounding
box, so a corner watermark on a 3000-frame chain doesn't blend 3000 full frames.

**Checkpoint swaps are detected and flushed.** ComfyUI keeps previously-loaded
models resident and only evicts reactively, so switching checkpoints mid-session
(e.g. NVFP4 → FP8 → MXFP8 while comparing quality) leaves the *old* DiT on the
card alongside the new one, plus any hooks a previous LoRA installed and stale
allocator blocks sized for the old layers. The card is then already half full
before the first shot samples — which looks like the node over-spilling, when in
fact the budget was measured against memory the previous checkpoint never
released. The node fingerprints the model (quant format, layer count, weight
size) and, when it changes between runs, calls `unload_all_models()` plus a
double VRAM/RAM purge **before** any measurement or patching. `info` reports it:
`model changed since last run (nvfp4 ~11.7GB -> mxfp8 ~19.5GB): flushed all
resident models and VRAM caches`. Identical models across runs are untouched, so
there's no cost to a normal chain.


**Character duplication at low resolution (`subject_count_guard`).** Duplication
gets markedly more likely *below* H3's native 768 short edge: fewer pixels per
subject pushes the sample away from the training distribution and the model tiles
the figure. The strongest prompt-side defence is an explicit count, so the node
can prepend one to each shot — `Exactly two people in this shot, no duplicates,
no other people in frame.` — counting only the characters actually referenced in
that beat. `auto` (default) enables it when the short edge is under 768 **or when a LoRA is
applied** — a distilled LoRA compresses ~20 steps into 4–8, so it fixes global
composition (including how many people are in frame) within the first step or two
and then reinforces that choice rather than revising it. That is why turbo LoRAs
duplicate subjects even at native resolution with a clean prompt. On LoRA runs the
count clause is also moved to the **front** of the prompt, ahead of scene and
style, so it binds before composition settles. Both stock-loader LoRAs (weight
patches) and bypass LoRAs (injections) are detected. `auto` also fires on ANY shot
that binds two or more people — multi-figure frames tile and merge even at native
size, and the count clause is the cheapest thing that holds the number down.
`on` always; `off` never.
Scenery beats with no characters never get the clause.

If duplication persists at native resolution, that's the model rather than the
prompt — keep subjects visually distinct, add spatial cues ("Kristy at the left
wing"), and avoid having two characters overlap in frame. A mirror or other
reflective surface in the anchor is a common hidden cause: H3 renders the
reflection as a second figure standing in the room, and since the anchor repeats
on every shot, the doubling happens on every shot — the anchor-hazards report now
flags it before the render.


**Write attributes, not noun phrases.** In `character_memory`, describe people as
bare attributes — `Kristy = she, 27, silver hair, red jacket` — not as noun
phrases like `a woman with silver hair`. A noun phrase renders inline as
`She (a woman with silver hair)`, which puts **two subject nouns in one clause**
("She" and "a woman") and reads to the model as two people: character duplication
from the very first shot, at any resolution. The node now strips these
automatically (`a young woman, silver hair` → `young, silver hair`, and a bare
`a woman` is dropped entirely), but writing attributes directly is clearer and
avoids relying on the cleanup.


**Spilling at native resolution (`decode_tile_frames` / `decode_tile_size`).** The
VAE decode — not sampling — is usually the peak allocation in a run: without
temporal tiling the video VAE expands the *entire* clip at once (a 243-frame
1344x768 shot is the largest single tensor the node ever creates). On a checkpoint
that already exceeds VRAM and streams, that decode is what tips the card into
shared memory.

Set `decode_tile_frames` to 8–16 to decode in temporal chunks, and
`decode_tile_size` to 256 for spatial tiles. Both default to 0 (ComfyUI's
defaults). Lower values mean lower peak VRAM and slightly slower decode. Try these
before dropping resolution — they cost speed, not picture quality.

**Structural limit:** if the weights alone exceed your VRAM (e.g. an unpruned
~19.5GB MXFP8 on a 16GB card), *some* spill is unavoidable no matter what the node
does — the model is streaming before a single frame is allocated. Tiling reduces
the peak on top of that baseline; only a checkpoint that fits removes it.


**Camera direction can summon a phantom person.** Writing motion guidance in the
anchor — "slow camera movement, the camera follows *the subject*", "moves toward
*the person*", "tracks *the figure*" — leaves an unnamed person reference in text
that is stamped into **every** shot, alongside your named cast. The model renders
it as an extra body matching no character sheet. The node now rewrites those
generic references ("the subject/person/figure/character") to refer to the scene
instead, keeping the camera direction intact. Safest is to phrase motion without
a person at all: `Slow, smooth camera movement. Minimal motion blur.`


## Per-beat directives (`key: value` lines)
A beat can carry directive lines that configure it rather than describe it. They
are **stripped out of the prose** before the prompt is built, and they are never
beats of their own — a directive attaches to the next content line, or to the
previous beat if it trails the paragraph. So a `wardrobe:` line inside a beat
won't accidentally become its own shot.

- **`wardrobe:`** — clothing and identity. Covered in full above.
- **`seconds: 8`** (alias **`duration: 8`**) — an explicit length for *this* beat,
  in seconds. This is the highest-priority length signal: it is **honored even
  when `per_beat_length` is off**, because you stated a duration outright. It is
  still clamped to the VRAM budget, which is a hard ceiling — a per-beat length
  can only make a shot **shorter** than the card allows, never longer. Use it to
  give one beat room ("`seconds: 10`") while the rest take the default.
- **`exit: Jon`** — Jon leaves the scene. Like auto-removals, exits are
  **deferred**: he is still present in the shot that *shows* him leaving, and
  absent from every shot after it. This stops a character the story has written
  out from wandering back in. Exits are also detected automatically from the
  action text; the directive is the explicit override.
- **`enter: Jon`** — undoes a previous exit, bringing him back into play.
- **`soundscape:` / `overall_soundscape:`** and **`music:` /
  `non_diegetic_music:`** — per-beat audio that overrides the global widgets for
  that shot only. A beat that sets its own skips the global stamp.

Without a `seconds:` line the length falls back to quoted dialogue (~2.5 words/sec
plus 1s of air, when `per_beat_length` is on) and otherwise to the full budget.
Action prose carries no reliable duration signal — "walks across the tarmac" is 2s
or 12s depending on the tarmac — so a silent beat is **never guessed short**.

## Preview the split without rendering (`plan_only`)
Set `plan_only` on the main node to **True** to see how a job will split —
shots, frames per shot, seconds, total length — near-instantly, with **no
render**. It uses the node's *own* settings (resolution, `shot_seconds`, fps,
prompt), so there's nothing to re-enter and nothing can drift out of sync. The
plan appears in the `info` output and the `shots` / `frames_per_shot` /
`video_seconds` outputs are populated. Turn it off to render for real. (This
replaces the old separate Plan node, which required duplicating settings by
hand.)



## Node: **H3 Shot Length**  (category: MiniMax-H3/utils)
Holds ONE shot length and emits it as both `seconds` and a valid H3 frame count
(17k+5 grid, capped at 362 unless you turn the cap off). Wire:

    H3 Shot Length (seconds) → H3 Long Videos FL2VA (shot_seconds)
    H3 Shot Length (frames)  → Model Preview Override (preview_frames)

One value entered once drives both, so they can't drift apart. It never reads
the model, so it can sit upstream of a preview override without creating a
wiring cycle. The main node also outputs `frames_per_shot` and `total_frames`
if you'd rather read back the value it chose.

## Optional node: **H3 Model Inspector**
Reports the base precision of a loaded model (BF16 / FP8 / INT8 / NVFP4 /
MXFP8) and whether your card runs it natively. Use it to confirm you're on a
base checkpoint (so 20 steps is right) vs a distill/low-step path.

## Reference images (the REF2VA part)

Connect up to four images to `ref_image_1` … `ref_image_4`. The tokenizer labels
them `<Picture 1>` … `<Picture 4>` **in input order** and appends your prompt
after them, so you can bind one to a character by name in the prompt —
`Kristy, <Picture 1>, walks around the garage` — or say nothing and let them work
as a general appearance anchor.

**A shot carries either references or the last-frame handoff — never both.** They
are two different task conditionings competing for the same `cond_video_latents`
slot inside ComfyUI's H3 model wrapper: the reference branch overwrites what the
keyframe branch wrote, while the packed layout still reserves rows for both, so a
shot given both would hand the DiT fewer latents than it has condition rows.
`ref_mode` decides where they land. The default reads your prompt:

| `ref_mode` | Where the reference goes |
|---|---|
| `where tagged` *(default)* | **The shot whose text names `<Picture N>`** — write the tag in the beat where that character appears. Every untagged shot keeps its handoff, and a tagged shot carries the previous frame as an extra reference so the tag is not a cut. |
| `first shot` | Shot 1, whoever is in it |
| `every shot` | All shots, no handoff anywhere |
| `every shot + handoff ref` | All shots, previous frame added as an extra reference |

**Why tagging is the precise option.** The positional modes go by shot *number*
and are blind to who is actually in the shot. A character who first appears in
shot 2 gets nothing, while an empty establishing shot 1 gets a portrait pushed
into its opening frames. Tagging puts the reference next to the character it
describes:

```
Wide shot of the empty garage, sunlight through the bay doors.

Kristy, <Picture 1>, walks in and looks around.

Kristy finds Dan, <Picture 2>, at the bench.

Dan hands her a wrench.
```

Shot 1 renders clean and keeps its handoff. Shot 2 carries Kristy's photo, shot 3
carries Dan's, shot 4 chains normally. Tags are **renumbered per shot** — a shot
using only `<Picture 2>` receives that image as `<Picture 1>` and its text is
rewritten to match, because the tokenizer numbers references by their position in
the list the shot actually carries. `<picture_1>`, `<Picture 1>` and `<PICTURE 1>`
all work. A tag naming an unconnected input is removed from the text and reported
in `info`. With references connected but nothing tagged anywhere, it falls back to
`first shot` rather than silently conditioning nothing.

The three positional modes still differ in what they trade:

| `ref_mode` | Shot 1 | Shots 2+ | Trade |
|---|---|---|---|
| `first shot` *(default)* | references | handoff | Continuity unbroken; identity is established once and then carried by the frames |
| `every shot` | references | references | Strongest identity; **no handoff**, so beats meet as cuts, not one continuous take |
| `every shot + handoff ref` | references | references **+ previous last frame as an extra reference** | Continuity returns as a soft signal — the model is *shown* where the last shot ended rather than told to start exactly there |

**Stopping the reference being copied into the opening frames.** `ref_noise_aug`
controls how *clean* the reference is presented as. H3's own default, **0.999**,
hands the model a finished, noise-free image — which is an invitation to
reproduce it at the start of the shot rather than to take an identity from it.
The DiT uses the value twice: it blends the condition latent with noise at
`1 - aug`, and it labels those rows with a timestep of `max(t_video, aug)`.

Lower it if the reference appears burned into the first frames: try **0.95**,
then **0.90**. Below about 0.8 the reference stops holding identity at all. It
applies **only to ref-conditioned shots** — the last-frame handoff is never
weakened, or continuity would break.

`ref_image_size` sets how large each reference is encoded. `match` scales it down
to the generation's pixel area, so a reference costs roughly one frame per step.
`max` uses the reference pipeline's 2048 short edge for the best identity
fidelity — but reference rows are re-attended **every step of every
ref-conditioned shot**, so on a long chain `max` is several times slower. Neither
mode ever upscales a small reference.

With no reference connected, the node behaves exactly like FL2VA.

`info` reports which shots took the reference channel and which kept the handoff,
and `plan_only` previews the same split before you spend a render on it.

## Character syntax: a worked example

Four fields, each with one job. This is the whole setup for a character driven by
a reference picture.

**1. The picture** — wired, never named in a text field:

```
Load Image ──(IMAGE)──> ref_image_1
ref_mode       = first shot
ref_image_size = match
```

A **head-and-shoulders** shot is the right choice: it spends the whole reference
on the face, which is what drifts, and it carries no clothing, so it cannot fight
a wardrobe removal later. A full-body reference wearing a garment you intend to
remove will keep re-asserting it on every ref-conditioned shot.

**2. `character_memory`** — who they are and what they wear:

```
Kristy = she, silver hair in a ponytail, scar over left eyebrow,
         red leather jacket with a white chest patch, blue jeans, grey shorts, black boots
```

- **`Name = `** — the named form. Without it the sheet is prepended to every beat
  as a bare list instead of binding to the character.
- **Pronoun first** (`she` / `he` / `they`) — drives pronoun resolution, repeat-name
  collapsing and removal attribution. It is stripped from what is rendered.
- **Attach detail with `with`, never a comma.** `red leather jacket with a white
  chest patch` is ONE garment; `red leather jacket, white chest patch` is two, and
  only the first would come off.
- **Include the under-layer** (`grey shorts`) whenever something is removed. That
  is what keeps a removal from rendering as nudity; `info` warns when a removal
  leaves a body zone bare.
- **4–7 items.** The sheet is re-stamped on every shot, so spend it on
  distinctive, renderable traits. `27` and `athletic build` mostly do not render;
  `scar over left eyebrow` does.

**3. The anchor (first paragraph)** — scene and style only:

```
Handheld phone video, 26mm lens, natural window light, mild lens distortion,
light compression noise, no color grade. An open 4 bay car garage.
```

**No names, no clothing.** Names here are stripped anyway (the beat binds people
inline), and clothing here is immutable — a removal cannot stick because the
anchor re-applies it every shot.

**4. The beats** — action, framing, and the tag if you want it:

```
Medium shot. Kristy, <Picture 1>, walks around the garage checking the benches,
then stops at the far wall.

Kristy finds Dan sitting in a chair and asks him: "Where are the pistons?"

Kristy takes off her red jacket and drops it on the workbench, then turns back
to the engine.
```

Framing belongs here, not in the anchor, so it can change shot to shot. Two
clauses per beat is a good ratio — a beat with one short action in a long shot
leaves the model seconds it was told nothing about.

### Where `<Picture N>` is safe

The tokenizer uses the same `<Picture N>` numbering for **keyframes**, so on a
shot with no references the tag points at the previous shot's last frame instead
of your photo:

| `ref_mode` | Tag in a beat | Tag in the anchor or `character_memory` |
|---|---|---|
| `first shot` *(default)* | **First beat only** | **No** — those are stamped on every shot, and shots 2+ have no references |
| `every shot` | Anywhere | Yes — every shot carries the references |
| `every shot + handoff ref` | Anywhere | Yes |

The tag is optional either way: the reference conditions the shot whether or not
you name it.

### What the model receives

```
shot 1  ... Kristy (silver hair in a ponytail, scar over left eyebrow, red leather
        jacket with a white chest patch, blue jeans, grey shorts, black boots),
        <Picture 1>, walks around the garage checking the benches ...

shot 3  ... Kristy (silver hair in a ponytail, blue jeans, grey shorts, black boots)
        takes off her red jacket ... by the last frame the red leather jacket is off
        and she is not wearing it. The grey shorts underneath stay on ... never put
        back on, never re-worn, and the action never plays in reverse.
```

The jacket leaves the description in the shot that removes it, the under-layer is
stated, and no later shot names the jacket at all.

## Props: objects that must survive the shot boundary

Each shot is a separate generation, so a definite reference has no antecedent. Write:

```
Dom drives a van down a farm road and stops in front of a barn.

Dom gets out of the van and walks to the back doors.
```

and the prompt for shot 2 contains **no van at all** — just the words "the van".
The model has nothing to resolve that against except the handoff frame, so it
invents one, and you get Dom stepping out of one van and walking to another.

With `auto_props` on (the default), an object introduced **indefinitely** is
carried forward and bound on its first definite reference in a later beat:

```
shot 2: ... gets out of the same van and walks to the back doors.
        The van is the same van as in the previous shot -- one van only, no second van.
```

Re-describing it is not enough by itself — "a white van" in two shots is two white
vans — so the clause also states it is the *same* object and forbids a second.

Scoped deliberately: only the **first** mention per shot is expanded (so prompts
don't bloat), quoted dialogue is never rewritten, **worn garments are excluded**
(they have the wardrobe channel, and "the same red jacket" would fight a removal),
and frame/body nouns — the ground, the light, the hand — are never carried.

**Repeat mentions inside one beat are collapsed too.** Naming the same object
three times in a single shot — "drives a van … gets out of the van … walks to the
back of the van" — is the most reliable way to get more than one of it, exactly as
naming a person twice duplicates them. The first definite mention survives and the
rest become "it", and the shot gets a positive count: *Exactly one van in this
shot.* This only fires when a single object is in play, so "it" can never be
ambiguous, and quoted dialogue is never touched.

**Introduce things indefinitely the first time.** `a rusted red toolbox` in beat 1
gives every later `the toolbox` something to bind to; starting with `the toolbox`
gives the node nothing to carry and the model something to invent.

## What belongs in the anchor (and what will bite you)

The anchor is stamped into **every** shot, so anything in it has to be true of
every shot. `plan_only` and `info` now scan it and report four hazards, each of
which has cost a real render:

| in the anchor | what happens |
|---|---|
| **person / face words** — `skin`, `pores`, `subject`, `them`, `hair` | they arrive in shots with nobody in them, and can render a face in an empty establishing frame. Put them in `character_memory`, which is only emitted where that person appears. |
| **apparatus words** — `camera`, `lens`, `sensor`, `handheld`, `iPhone`, `documentary` | the equipment gets rendered, sometimes with someone holding it. Describe the **image**, not the gear. |
| **framing** — `medium shot`, `close-up`, `wide shot` | pins every shot to that size. Put framing in the beats so it can change. |
| **clothing** — any garment noun | the anchor is immutable, so it re-applies the garment every shot and a removal can never stick. |

A working anchor holds **light**, **image properties** and **the location**:

```
Natural daylight, hard sun and deep shadow, highlights clipping to white. Shallow
depth of field, the background falling soft. Fine grain, slight motion blur,
neutral colour, no colour grade. A farm with a barn building.
```

Note what is *not* named: no lens, no sensor, no handheld, no skin, no subject. The
realism cues survive as descriptions of the picture — "shallow depth of field"
rather than "35mm at f/2.8", "fine grain" rather than "sensor grain". Per-person
realism (`weathered skin with visible pores`) belongs in `character_memory`, where
it only appears in shots containing that person.

**Babble on a shot that HAS dialogue.** `mute_nonspeech_audio` cannot help there —
a shot with a scripted line is deliberately left audible. The cause is the same
vacuum that makes an action repeat: a 2-second line in a 10-second shot leaves
8 seconds of audio the model was told nothing about, and the audio branch keeps
talking to fill them. `info` now reports it before the render:

    BABBLE RISK -- shot 2: 2.0s of dialogue in a 10.1s shot -- 8.1s of unscripted
    audio the model will fill with more speech

The fix is length, not silencing: leave `per_beat_length` on so the shot is sized
from its line, or put `seconds:` on that beat. (`dialogue_fit_warnings` covers the
opposite error — a line too long for its shot, which gets truncated.)

## Requirements
- ComfyUI 0.30+ with native MiniMax-H3 support.
- The node applies **ModelSamplingMiniMaxH3** (the video/audio flow schedule)
  internally by default via `apply_model_sampling`, so you no longer have to
  wire it upstream — a missing patch is the usual cause of gibberish audio.
  Turn it off if you patch upstream yourself. Shifts are exposed
  (`shift_video`/`shift_audio`): base H3 = 12/3 (the defaults), a low-step MXFP8
  checkpoint ≈ 8 video, a 4-step distill/turbo LoRA ≈ 4–6 audio.
- **Pillow (PIL)** only if you use the text overlays. ComfyUI already ships it, so
  this is effectively always satisfied; if it were missing, the overlays are
  skipped with a note in `info` and the render still completes.
- No negative prompt needed — H3 is CFG-free (cfg 1) and the node makes an empty one internally.
- No denoise input — it's fixed at 1.0 internally (partial denoise desyncs the joint audio/video schedule).

## Disclaimer

The owner of this repo will not be responsible for any copyright strikes
incurred because of use. You are responsible for your works. Use this node
responsibly and ethically.
