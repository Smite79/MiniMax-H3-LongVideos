# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""Audio policy shared by conditioning and soundtrack assembly."""

from dataclasses import dataclass
import math
import re

import torch
import comfy.nested_tensor
from h3_runtime import temporal_shape


@dataclass(frozen=True)
class ShotAudio:
    speech: bool
    sounded: bool
    voiced_only: bool
    silence_enabled: bool
    lead_seconds: float
    latent_fps: int
    # The tail. Everything after the line's expected end is pinned the way the lead
    # pins everything before its start. All three default off, so a ShotAudio built
    # the old way -- six positional arguments -- behaves exactly the old way.
    line_seconds: float = 0.0      # planner's estimate of the spoken line
    tail_seconds: float = 0.0      # free audio kept after that estimate; 0 = no tail pin
    frame_count: int = 0           # the shot in pixel frames; the audio T comes from it

    @property
    def pinned(self):
        return self.silence_enabled and not self.speech and not self.sounded

    @property
    def lead_frames(self):
        if not self.speech or self.lead_seconds <= 0:
            return 0
        return round(self.lead_seconds * self.latent_fps)

    @property
    def tail_frames(self):
        """Audio latent frames pinned at the END of a dialogue shot.

        The lead pins the opening so the line cannot start early; nothing pinned the
        close, and a 2s line in a 9s shot left 7s of open branch in a shot the model
        knows has a voice in it -- which is where speech carries on past the line, or
        doubles it. The free span is lead + the line's estimate + tail_seconds; the
        rest is held at encoded silence. The model chooses WHEN to speak, so the
        margin is the author's dial: a clipped word costs more than a second of babble.
        Off unless the shot speaks, the margin is set, and at least half a second would
        be pinned -- a sliver is not worth the risk of clipping."""
        if (not self.speech or self.tail_seconds <= 0 or self.line_seconds <= 0
                or self.frame_count <= 0):
            return 0
        total = temporal_shape(self.frame_count)[2]
        free = self.lead_frames + round((self.line_seconds + self.tail_seconds) * self.latent_fps)
        tail = total - free
        return tail if tail >= round(0.5 * self.latent_fps) else 0

    @property
    def needs_silence_latent(self):
        return self.pinned or self.lead_frames > 0 or self.tail_frames > 0

    @property
    def accepts_built_foley(self):
        return self.silence_enabled and (self.pinned or self.voiced_only)


_SILENT_UNIT = {"lat": None, "key": None}


_BED_EVENTFUL = ("birdsong", "cutlery", "monitor somewhere", "corridor beyond")


_BED_RMS = 0.08


_BED_RECIPE = (
    (r"\brain\b",            dict(tilt=0.8, cut=9000, hp=250, mod=(0.30, 0.18))),
    (r"\bstorm\b|\bthunder", dict(tilt=1.7, cut=700,          mod=(0.13, 0.40))),
    (r"\bwind\b|\btrees\b",  dict(tilt=1.2, cut=2600,         mod=(0.18, 0.42))),
    (r"\bsea\b|\bocean\b",   dict(tilt=1.3, cut=1700,         mod=(0.11, 0.50))),
    (r"\btraffic\b",         dict(tilt=1.7, cut=900,          mod=(0.07, 0.22))),
    (r"\bengine\b",          dict(tilt=1.5, cut=520, hum=(60.0, 0.30),
                                  mod=(0.09, 0.12))),
    # NOT 0.55 Hz AT 45% DEPTH. That is 33 swells a minute on the one bed a bathroom
    # gets, and a bed that heaves at pulse rate is the same defect the footsteps had
    # arriving from the other side -- the report heard both in the same scene. Water
    # in pipes does not surge, it hisses, with fine turbulence in it. Faster and far
    # shallower, and the cut opened up because 1250 Hz made it a rumble when the sound
    # of a pipe is mostly above that.
    (r"\bpipes\b|\bwater\b", dict(tilt=1.1, cut=3200,         mod=(2.70, 0.10))),
    # A clock at exactly 1.00 Hz is 60 a minute, which is a resting pulse, and the
    # tick was one soft click -- so the one bed in the table whose rate is RIGHT still
    # landed on a heartbeat. A mechanical clock does not tick, it ticks and TOCKS: two
    # unequal strikes to the second, which no pulse does.
    (r"\bclock\b|\bticking", dict(tilt=1.6, cut=800, tick=(1.0, 0.22), tock=0.62)),
    # The hum family: a fridge, a fan, a strip light, a monitor. Tonal, not noise.
    (r"\bhum(?:ming|s)?\b|\bfan\b|\bfridge\b|\bstrip light\b|\bmonitor\b",
                             dict(tilt=1.4, cut=1500, hum=(100.0, 0.22))),
    (r"\btiled\b|\bringing\b", dict(tilt=0.9, cut=6000, hp=180)),
    (r"\bhard walls\b|\bgiving the sound back\b", dict(tilt=1.6, cut=950)),
    (r"\bopen air\b|\bno walls close\b|\bbirdsong\b", dict(tilt=1.0, cut=7000)),
    (r"\bhollow quiet\b|\bhallway\b|\bcorridor\b|\blarge empty room\b|\blong tail\b",
                             dict(tilt=1.6, cut=700)),
    (r"\bcutlery\b|\bchairs\b", dict(tilt=1.1, cut=4500)),
    (r"\bsoft room\b|\blittle echo\b", dict(tilt=1.8, cut=520)),
    (r"\bnight\b|\bbedroom\b|\bhouse\b|\bquiet\b", dict(tilt=1.9, cut=380)),
)


def bed_recipe(phrase):
    """How to build the bed this phrase describes. The neutral room if none match."""
    p = str(phrase or "").lower()
    for pat, rec in _BED_RECIPE:
        if re.search(pat, p):
            return dict(rec)
    return dict(tilt=1.8, cut=420)


def synth_ambient(phrase, n, sr, seed=0, channels=2):
    """Build `n` samples of the ambience `phrase` describes. [C, n], or None.

    Shaped in the FREQUENCY domain -- white noise, an envelope, back again -- which
    gives exact spectral control in one pass and, unlike a per-sample filter, does
    not walk a million-sample loop in Python.

    Generated at the FULL length of the film, so unlike a wired file there is no
    loop and therefore no join to hide.

    Defensive like everything else on this path: any failure returns None and the
    soundtrack goes out as the model made it."""
    try:
        n, sr = int(n), int(sr)
        if n < 64 or sr <= 0:
            return None
        rec = bed_recipe(phrase)
        g = torch.Generator().manual_seed(int(seed) & 0x7fffffff)
        w = torch.randn((int(channels), n), generator=g)
        f = torch.fft.rfftfreq(n, d=1.0 / sr).clamp(min=1.0)
        # Amplitude goes as f^(-tilt/2), so POWER goes as f^-tilt: tilt 1 is pink,
        # 2 is brown. Then a gentle low-pass, and a high-pass where the recipe wants
        # the bottom out of it.
        env = f.pow(-float(rec.get("tilt", 1.8)) / 2.0)
        env = env / (1.0 + (f / float(rec.get("cut", 420))) ** 2)
        if rec.get("hp"):
            env = env * (f / (f + float(rec["hp"])))
        y = torch.fft.irfft(torch.fft.rfft(w, dim=-1) * env, n=n, dim=-1)
        t = torch.arange(n, dtype=torch.float32) / sr
        # Slow movement, so a bed does not sit perfectly still and read as a hiss.
        if rec.get("mod"):
            rate, depth = rec["mod"]
            y = y * (1.0 + float(depth) * torch.sin(2 * math.pi * float(rate) * t))
        # A tonal hum is a TONE, not noise: a fridge and a strip light are pitched.
        if rec.get("hum"):
            hz, amp = rec["hum"]
            hum = (torch.sin(2 * math.pi * float(hz) * t)
                   + 0.35 * torch.sin(2 * math.pi * float(hz) * 2 * t))
            y = y + float(amp) * hum.unsqueeze(0)
        if rec.get("tick"):
            rate, amp = rec["tick"]
            step = max(1, int(sr / max(float(rate), 0.01)))
            click = torch.zeros(n)
            idx = torch.arange(0, n, step)
            click[idx] = 1.0
            # TICK, TOCK. The offbeat strike, quieter than the beat, which is what
            # makes an escapement an escapement rather than a metronome -- and what
            # stops 60 strikes a minute reading as a pulse. See the recipe.
            if rec.get("tock"):
                off = idx[:-1] + step // 2
                click[off[off < n]] = float(rec["tock"])
            decay = torch.exp(-torch.arange(min(step, int(sr * 0.05)),
                                            dtype=torch.float32) / (sr * 0.004))
            click = torch.nn.functional.conv1d(
                click.view(1, 1, -1), decay.flip(0).view(1, 1, -1),
                padding=decay.numel() - 1)[0, 0, :n]
            y = y + float(amp) * (click * torch.randn(n, generator=g)).unsqueeze(0)
        # NORMALISE BY RMS, NOT PEAK. Peak-normalising made the loudness depend on
        # the recipe's crest factor rather than on the setting: measured across the
        # beds, a strip-light hum came out at -8.2 dBFS and a ticking clock at
        # -34.1, a 26 dB spread from one ambient_level. RMS puts them all at the
        # same subjective level, so the widget means the same thing in every room.
        rms = float(y.pow(2).mean().sqrt())
        if not (rms > 0.0) or not torch.isfinite(y).all():
            return None
        y = y * (_BED_RMS / rms)
        # ...then hold the peak down, because a peaky recipe (the clock) would
        # otherwise reach 2.8 at that RMS and clip before the mix even sees it.
        peak = float(y.abs().max())
        if peak > 0.95:
            y = y * (0.95 / peak)
        return y
    except Exception:
        return None                        # a bed is a nicety, a render is not


_MODES = ((1.00, 1.000, 1.00), (1.48, 0.270, 0.75),
          (2.13, 0.132, 0.55), (3.31, 0.060, 0.40))


def _band(x, sr, f0, q=4.0, order=3):
    """Resonant filter by spectral envelope: a mode cluster around f0, one pass.

    ORDER 3, which was measured. A single resonator's skirt falls off as 1/f, and
    against noise -- equal energy per Hz, spread over 20 kHz -- enough survives above
    the centre that the result is bright whatever f0 says: footsteps aimed at 130 Hz
    came back with a spectral centroid of 3.6 kHz, and every recipe sounded like the
    same hiss. Cubing the response is what makes f0 mean something.

    The f0/q interface is unchanged, so every recipe gets the mode cluster without
    being rewritten -- this is the one place all 21 of them pass through."""
    n = int(x.shape[-1])
    X = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(n, d=1.0 / sr).clamp(min=1.0)
    resp = torch.zeros_like(f)
    for ratio, gain, qs in _MODES:
        fc = float(f0) * ratio
        if fc >= sr * 0.45:            # past Nyquist is not a mode, it is aliasing
            continue
        qq = max(0.7, float(q) * qs)
        # BANDWIDTH COMPENSATION, and it is not optional. A resonator's absolute
        # bandwidth is fc/Q, so a mode an octave up passes twice the noise for the
        # same gain -- and these are excited by noise, which has equal energy per
        # Hz. Uncompensated, the cluster came out about 2x brighter across every
        # recipe and put a footstep at 428 Hz against the 130 it is aimed at, which
        # is the "a footstep is a hiss" failure the order-3 skirt was fixed for.
        # Energy through a mode goes as gain^2 * fc / Q, so scaling the gain by
        # sqrt(Q/fc) makes the numbers above mean the loudness they look like.
        g_i = float(gain) * math.sqrt(float(qs) / float(ratio))
        # ...and the cluster itself scales with Q, because Q IS how much the thing
        # rings. Metal at q 5-8 has strong upper modes; a footstep at q 1.6 is a
        # broadband thud on a floor and has almost none. Applied only above the
        # fundamental, so a low-Q recipe collapses back to the single resonator it
        # was tuned as -- which is what keeps a footstep at 130 Hz a footstep.
        if ratio > 1.0:
            g_i *= min(1.0, float(q) / 4.0)
        resp = resp + g_i * (1.0 / torch.sqrt(
            1.0 + (qq * (f / fc - fc / f)) ** 2)) ** int(order)
    return torch.fft.irfft(X * resp, n=n)


def _hits(n, sr, g, times, decay, amp=1.0):
    """Decaying noise bursts at the given times (seconds). The excitation for a
    click, a rattle, a footfall -- everything percussive here is this plus a band.

    EVERY HIT DIFFERS. They used to be identical -- same level, same decay, same
    everything -- and thirty-three identical clicks is not a chain, it is a machine.
    Nothing gives a synthetic sound away faster: the ear is far better at spotting a
    repeat than at judging a timbre, so a rattle whose links are all the same reads
    as fake even when each single link sounds right.

    Level varies about +/-5 dB and decay by about a third, which is the spread a
    real repeated contact has from hitting at a different point and angle."""
    x = torch.zeros(n)
    for t in times:
        i = int(t * sr)
        if i < 0 or i >= n:
            continue
        a = float(amp) * float(torch.exp((torch.rand(1, generator=g) - 0.5) * 1.1))
        d = float(decay) * float(1.0 + (torch.rand(1, generator=g) - 0.5) * 0.7)
        L = max(4, int(d * sr))
        m = min(L, n - i)
        env = torch.exp(-torch.arange(m, dtype=torch.float32)
                        / max(d * sr / 4.0, 1.0))
        x[i:i + m] += torch.randn(m, generator=g) * env * a
    return x


def _room(x, sr, secs=0.11, wet=0.16, seed=0):
    """A small room around the sound. Convolution with a decaying noise tail plus
    three early reflections.

    The dryness was the loudest tell. Every one of these was rendered anechoic --
    no reflections, no tail -- and nothing in the physical world sounds like that;
    the ear reads a bone-dry impact as "not in a place" before it judges anything
    else about it. The tail is rolled off above 2.2 kHz because a real room absorbs
    highs faster than lows, and a bright tail is its own kind of wrong.

    Linear convolution, not circular: the transform is padded past n + L so a tail
    cannot wrap round and appear before the hit that caused it."""
    n = int(x.shape[-1])
    L = max(8, int(float(secs) * sr))
    if n < 8 or not (float(wet) > 0.0):
        return x
    g = torch.Generator().manual_seed(int(seed) & 0x7fffffff)
    t = torch.arange(L, dtype=torch.float32)
    ir = torch.randn(L, generator=g) * torch.exp(-t / max(L / 5.0, 1.0))
    ir[0] = 0.0
    for d, a in ((0.0071, 0.50), (0.0133, 0.34), (0.0211, 0.23)):
        i = int(d * sr)
        if i < L:
            ir[i] += a
    m = 1
    while m < n + L:
        m <<= 1
    F = torch.fft.rfftfreq(m, d=1.0 / sr).clamp(min=1.0)
    IR = torch.fft.rfft(ir, n=m) / (1.0 + F / 2200.0)
    wet_sig = torch.fft.irfft(torch.fft.rfft(x, n=m) * IR, n=m)[:n]
    p, q = float(wet_sig.abs().max()), float(x.abs().max())
    if not (p > 0.0) or not torch.isfinite(wet_sig).all():
        return x
    wet_sig = wet_sig * (q / p)
    return x * (1.0 - float(wet)) + wet_sig * float(wet)


def _even(start, count, gap, jitter, g):
    """Click times, with a little jitter so a rattle is not a drum machine."""
    j = (torch.rand(int(count), generator=g) - 0.5) * 2.0 * float(jitter)
    return [float(start + i * gap + j[i]) for i in range(int(count))]


# ---------------------------------------------------------------------------
# NOT EVERYTHING IS AN IMPACT, and for a long time everything here was.
#
# Every recipe in this file was _hits -- a train of decaying noise bursts -- sent
# through one narrow resonator. That is exactly right for metal striking metal, and
# the cuffs, the keys, the bolt and the lock all measure like the real thing. It is
# wrong for three whole families, and measurement says how wrong:
#
#   footsteps            0.0% of their energy above 1.2 kHz, centroid 226 Hz
#   something landing    0.0%, centroid 175 Hz
#   a bed frame working  0.0%, centroid 395 Hz
#   a door on its hinges 3.4%
#   rope creaking        2.8%
#   water                duty 0.41 -- no water at all for 59% of the time
#
# A footstep with NO high frequency in it is not a footstep. The bright part is the
# contact -- sole against surface -- and it is the half that says what is stepping
# on what. Strip it out and what is left is a soft low thump, and at the walking
# cadence this recipe used (109 a minute) a soft low thump is a HEARTBEAT. Reported
# exactly that way: "footsteps sound like heartbeats".
#
# Water is not 14 impacts a second either. It is continuous, and a sound that stops
# 59% of the time is a tap: reported as "tapping sounds" in a bathroom.
#
# So there are three excitations now instead of one. _hits stays, unchanged, for the
# things that genuinely are impacts.
# ---------------------------------------------------------------------------


def _contact(n, sr, g, times, body=120.0, bright=2600.0, decay=0.085, amp=1.0,
             mix=0.55):
    """Something striking a SURFACE: a bright contact over a low body.

    Two bands at the same instants. The low one is the mass arriving, which is all
    this used to be; the bright one is the contact itself -- sole on tile, a box on
    boards -- and it is the half the ear identifies the sound by. A thump on its own
    says only that something heavy happened, and the listener's own prior supplies
    the rest: at a walking rate, from inside a body, that prior is a pulse.

    The two calls to _hits roll their own per-hit level and decay, so the balance
    between contact and body differs from step to step the way it does when a real
    foot lands at a slightly different angle. That is wanted, not tolerated."""
    low = _band(_hits(n, sr, g, times, decay, amp), sr, float(body), 1.6)
    tap = _band(_hits(n, sr, g, times, float(decay) * 0.13, amp), sr, float(bright), 0.9)
    lo_p, tp_p = float(low.abs().max()), float(tap.abs().max())
    if lo_p > 0:
        low = low / lo_p
    if tp_p > 0:
        tap = tap / tp_p
    return low * (1.0 - float(mix)) + tap * float(mix)


def _flow(n, sr, g, f0, q=1.1, rough=7.0, depth=0.5):
    """A CONTINUOUS sound: water running, something scraping, cloth moving.

    Band-limited noise whose level wanders, rather than a train of bursts with
    silence between them. The level never reaches zero -- that is the whole point,
    and it is what the duty measurement checks: water that stops is a tap, and a
    scrape that stops is a knock.

    Three incommensurate wander rates, at random phase, so the movement does not
    settle into a pulse of its own -- which is the failure being fixed, and it would
    be careless to rebuild it one layer up."""
    y = _band(torch.randn(n, generator=g), sr, float(f0), float(q))
    t = torch.arange(n, dtype=torch.float32) / sr
    env = torch.zeros(n)
    for hz, a in ((float(rough), float(depth)),
                  (float(rough) * 0.37, float(depth) * 0.55),
                  (float(rough) * 2.31, float(depth) * 0.30)):
        env = env + a * torch.sin(2 * math.pi * hz * t
                                  + float(torch.rand(1, generator=g)) * 6.2832)
    return y * (1.0 + env).clamp(min=0.28)


def _creak(n, sr, g, times, f0=420.0, secs=0.45, glide=1.7, slip=34.0, amp=1.0):
    """Stick-slip: a pitched squeal that GLIDES, broken up by the slipping.

    A hinge, a rope going tight, a bed frame taking weight. All of them were impulse
    trains through a resonator, which is a knock -- and a knock is what they sounded
    like. What makes a creak a creak is that the surfaces grip, release and grip
    again dozens of times a second while the load changes, so the pitch RISES through
    the event and the amplitude is chopped up at the slip rate.

    Harmonics matter: a squeal is a rich tone, and a pure sine reads as a test
    signal. The glide and the slip rate both vary per event for the same reason
    every _hits burst does -- identical repeats are what gives synthesis away."""
    x = torch.zeros(n)
    rub = torch.zeros(n)
    for t0 in times:
        i = int(float(t0) * sr)
        if i < 0 or i >= n:
            continue
        d = float(secs) * float(1.0 + (torch.rand(1, generator=g) - 0.5) * 0.5)
        L = min(max(8, int(d * sr)), n - i)
        if L < 8:
            continue
        t = torch.arange(L, dtype=torch.float32) / sr
        gl = float(glide) * float(1.0 + (torch.rand(1, generator=g) - 0.5) * 0.3)
        hz = float(f0) * (1.0 + (gl - 1.0) * (t / max(d, 1e-6)).clamp(max=1.0))
        ph = 2 * math.pi * torch.cumsum(hz, dim=0) / sr
        tone = torch.sin(ph) + 0.45 * torch.sin(2 * ph) + 0.22 * torch.sin(3 * ph)
        # The slipping. A rounded square at the slip rate, so the tone is chopped
        # rather than tremoloed -- a creak is intermittent contact, not vibrato.
        sl = float(slip) * float(1.0 + (torch.rand(1, generator=g) - 0.5) * 0.4)
        chop = (torch.sin(2 * math.pi * sl * t
                          + float(torch.rand(1, generator=g)) * 6.2832) * 3.0)
        chop = (0.55 + 0.45 * chop.clamp(-1.0, 1.0))
        env = (1.0 - torch.exp(-t / 0.012)) * torch.exp(-t / max(d / 2.2, 1e-6))
        a = float(amp) * float(torch.exp((torch.rand(1, generator=g) - 0.5) * 0.9))
        # THE SLIPPING IS A MICRO-IMPACT, dozens a second, and it is where a creak
        # gets its broadband content. Without it the sound is a tone and its two
        # harmonics and nothing else -- measured at 0.0% of the energy above 1.2 kHz
        # on the bed frame, which is the same empty high end that let a low thump at
        # pulse rate read as a heartbeat. Gated by the same chop, because the grit
        # happens AT the slip and nowhere between.
        rub[i:i + L] += torch.randn(L, generator=g) * chop * env * a
        x[i:i + L] += tone * chop * env * a
    # ...BAND-LIMITED TO THE STRUCTURE'S OWN RESONANCES, and collected so it costs
    # one pass rather than one per event. White grit made every creak measure at a
    # 3.1 kHz centroid -- a hiss, with a wooden bed frame as bright as a steel hinge.
    # A slip excites what it is slipping ON, so the grit sits a few multiples above
    # the squeal and the wood stays wooden.
    # Floored, because a low squeal's third harmonic is still low: at f0 205 the
    # grit landed at 656 Hz and the bed frame came back with 0.2% of its energy above
    # 1.2 kHz -- empty up top again. A slip is a tiny impact and a tiny impact is
    # broad whatever it lands on.
    return x + 0.55 * _band(rub, sr, max(float(f0) * 3.2, 850.0), 0.8)


# A WALK IS NOT A METRONOME, and a metronome at walking speed is a pulse.
#
# The old footfall times came from _even: a fixed 0.55 s gap with +/-9% of jitter,
# laid from 0.25 s to the end of the shot whatever the shot was doing. Three things
# wrong with that, and the report named two of them -- "sound like heartbeats" and
# "not properly timed with movement".
#
# A real gait is ASYMMETRIC. Left and right are not the same interval; one leg
# carries slightly longer, and the difference is a few percent and consistent within
# a walk. That alternation is most of what the ear uses to hear a walk as a walk
# rather than as a pulse, and it is free to put back.
def _gait(secs, g, step=0.52, start=0.18, count=None, asym=0.055, vary=0.045):
    """Footfall times for a walk. Alternating, varied, and it STOPS.

    `count` caps the number of steps; without it the walk fills `secs`, which is the
    old behaviour and is right only when the beat really does walk the whole shot."""
    secs = float(secs)
    if secs <= 0 or step <= 0:
        return []
    n = int(count) if count else max(2, int(secs / float(step)))
    out, t = [], float(start)
    for i in range(n):
        if t >= secs:
            break
        out.append(t)
        # Left, then right: one interval slightly longer than the other, plus the
        # step-to-step variation a real walk has from the floor and the stride.
        side = float(asym) if i % 2 else -float(asym)
        v = float((torch.rand(1, generator=g) - 0.5) * 2.0 * float(vary))
        t += float(step) * (1.0 + side + v)
    return out


# THE ONLY THING IN THE TABLE WITH A VISIBLE SYNC POINT.
#
# A chain rattling or cloth moving has no frame the ear can check it against, so
# building it blind costs nothing. A footfall does: the foot lands on screen, and a
# footstep train laid at a fixed interval regardless is guaranteed to disagree with
# the picture. That is the second half of the report -- "not properly timed with
# movement" -- and it is why this set has one member rather than being a flag on
# every recipe.
_MOTION_TIMED = frozenset({"footsteps"})


def _footfalls(n, sr, g, times):
    """The footstep VOICE, given the times. One definition, because there are two
    callers -- the blind recipe and the motion-timed path -- and a second copy of
    these numbers would drift until the same film had two kinds of foot in it."""
    return _contact(n, sr, g, times, body=125, bright=2200, decay=0.070, mix=0.62)

# A step is between these, or it is not a step. Below is a run's cadence at best and
# above is somebody stopping between paces; outside the range a peak in the motion
# envelope is something else moving and must not be read as a gait.
_STEP_MIN, _STEP_MAX = 0.26, 0.95


def _step_period(env, fps):
    """The step interval `env` implies, in seconds, or None when it is not periodic.

    Autocorrelation over the plausible step lags. This part is a MEASUREMENT: a walk
    is periodic, frame-to-frame change inherits that period, and the lag of the
    strongest correlation is it. Rejected unless the peak is a real fraction of the
    zero-lag energy, because noise autocorrelates too and a shot of somebody standing
    still would otherwise produce a confident cadence out of nothing.

    A full stride is two steps, so a dominant lag at twice the step rate is halved --
    which happens when the two legs are not equally visible to the camera and only one
    swing per stride shows up."""
    try:
        e = env.detach().to(torch.float32).flatten()
        if e.numel() < 8:
            return None
        e = e - e.mean()
        z = float(e.pow(2).sum())
        if not (z > 0):
            return None
        lo = max(1, int(_STEP_MIN * fps))
        # AT LEAST THREE CYCLES of whatever it claims to have found. One broad bump in
        # a short shot correlates with itself at half its own width, and that is not a
        # cadence -- it is a pan. Three is the fewest that distinguishes a repeating
        # thing from a thing that happened.
        hi = min(int(e.numel()) // 3, int(_STEP_MAX * 2.0 * fps))
        if hi <= lo:
            return None
        r = [float((e[:-k] * e[k:]).sum()) / z for k in range(lo, hi + 1)]
        best = max(r)
        # 0.35, measured rather than guessed. At 0.18 a shot of pure noise came back
        # with a confident 0.5 s cadence -- a short envelope has few lags, so noise
        # autocorrelates well by chance, and inventing a gait out of grain is worse
        # than laying one blind. A real walk correlates far above this: the ten test
        # periods all land near 1.0, and with noise at 15% of the swing they do not
        # move at all.
        if best < 0.35:
            return None
        # THE SHORTEST STRONG LAG, NOT THE STRONGEST. A periodic signal correlates
        # with itself at every multiple of its period, and the longer lags often
        # correlate BETTER because they catch more cycles -- so taking the maximum
        # picked multiples. Measured against known step periods it answered 0.875 s
        # for a 0.35 s step (x2.5), 0.667 for 0.44 (x1.5) and 0.583 for 0.78 (x0.75):
        # four of five wrong, all of them harmonically related to the truth.
        #
        # The fundamental is the first lag that is nearly as good as the best one, and
        # it has to be a local peak -- the rising shoulder of a later peak is not a
        # period. This is the standard remedy for the same error in pitch detection.
        at = None
        for i, v in enumerate(r):
            if v < 0.85 * best:
                continue
            if (i == 0 or r[i - 1] <= v) and (i + 1 >= len(r) or r[i + 1] <= v):
                at = lo + i
                break
        if at is None:
            at = lo + r.index(best)
        secs = at / float(fps)
        if secs > _STEP_MAX:
            secs = secs / 2.0            # the lag was a full stride, not a step
        return secs if _STEP_MIN <= secs <= _STEP_MAX else None
    except Exception:
        return None


def _walk(n, sr, g, env, fps, step=0.52):
    """Footfall times placed against the picture's own movement.

    Two different kinds of claim, and they are worth keeping apart:

    MEASURED -- the cadence, from _step_period, and WHEN there is movement at all.
    Both come straight out of the envelope.

    INFERRED -- which phase of the cycle the foot lands on. Frame difference peaks
    when a limb is travelling fastest, which is mid-swing, and falls at contact: the
    swing leg has decelerated to nothing and the body is on both feet. So the steps
    are placed on the envelope's MINIMA. That is a physical argument rather than a
    measurement, and it is right up to half a step -- which is still the difference
    between a gait that drifts against the picture all shot and one that does not.

    The gate is RELATIVE to the shot's own movement, which is what makes it safe on
    any scene: there is no absolute scale for "moving", and a shot where nobody moves
    normalises its own noise up and keeps its steps rather than going silent. What it
    catches is the case it was reported for -- walking in and then standing still --
    where the still half really is far below the walking half."""
    secs = n / float(sr)
    if env is None or int(env.numel()) < 8:
        return _gait(secs, g, step=step)
    k = _step_period(env, fps)
    e = env.detach().to(torch.float32).flatten()
    span = float(e.max() - e.min())
    if k is None or not (span > 0):
        # No cadence to read. Still worth gating: an even train over a motionless
        # stretch is the complaint, whatever the rate.
        return [t for t in _gait(secs, g, step=step)
                if _moving(e, t, fps, span, step)]
    kf = max(2, int(round(k * fps)))
    # The phase whose samples sit LOWEST in the envelope -- contact, see above.
    best, off = None, 0
    for p in range(kf):
        idx = torch.arange(p, int(e.numel()), kf)
        if idx.numel() < 2:
            continue
        v = float(e[idx].mean())
        if best is None or v < best:
            best, off = v, p
    out = []
    t = off / float(fps)
    while t < secs:
        if _moving(e, t, fps, span, k):
            out.append(t)
        t += k
    return out


def _moving(e, t, fps, span, period, gate=0.32):
    """Is the picture moving at time `t`? Relative to this shot's own range.

    THE WINDOW IS A FRACTION OF THE STEP, not a fixed slice of time, and that is not
    a refinement -- with a fixed 0.12 s it deleted every footstep in a walk at a
    half-second cadence, which is the commonest cadence there is.

    The reason is that these two mechanisms pull against each other by construction.
    Steps are placed on the envelope's MINIMA, because that is where the contact is;
    the gate then asks whether the picture is moving there -- and at the bottom of a
    trough it is not. A window narrow against the step period sees only the trough,
    answers no, and drops the very step it was asked about. Scaled to the period it
    always reaches the neighbouring peak, so the question it answers is the one
    intended: is the body moving AROUND here, not at this instant.

    The cost it could have is that a step just after somebody stops still sees the
    walking it came out of, letting one leak past the stop. Measured at three
    cadences, none leaks -- the window reaches backwards by half a step and the first
    step after a halt is a whole one past it."""
    i = int(t * fps)
    w = max(1, int(max(0.10, 0.55 * float(period)) * fps))
    lo, hi = max(0, i - w), min(int(e.numel()), i + w + 1)
    if hi <= lo:
        return True
    return bool(float(e[lo:hi].max()) >= float(e.min()) + gate * span)


_FOLEY = {
    "cuffs ratcheting closed":
        lambda n, sr, g, secs: _band(_hits(n, sr, g, _even(secs * 0.33, 9, 0.030, 0.004, g),
                                           0.020), sr, 3200, 6.0),
    "cuffs knocking":
        lambda n, sr, g, secs: _band(_hits(n, sr, g, _even(secs * 0.25, 4, 0.22, 0.06, g),
                                           0.035), sr, 2600, 5.0),
    "chain links dragging":
        lambda n, sr, g, secs: _band(_hits(n, sr, g,
                                           _even(0.05, max(4, int(secs * 11)), 0.09, 0.035, g),
                                           0.028), sr, 4200, 7.0),
    # A LOAD COMING ON IS A CREAK, NOT A KNOCK. Both of these were impulse trains
    # through a resonator and measured 2.7% of their energy above 1.2 kHz -- a dull
    # thump where the sound is a rising squeal. Webbing and rope going tight grip and
    # slip as the load builds, which is what _creak is.
    "restraints pulling taut":
        lambda n, sr, g, secs: _creak(n, sr, g, _even(secs * 0.3, 3, 0.38, 0.10, g),
                                      f0=540, secs=0.32, glide=1.9, slip=41.0),
    "rope creaking as it goes tight":
        lambda n, sr, g, secs: _creak(n, sr, g, _even(secs * 0.3, 3, 0.34, 0.09, g),
                                      f0=430, secs=0.38, glide=2.1, slip=29.0),
    "a lock snapping shut":
        lambda n, sr, g, secs: _band(_hits(n, sr, g, [secs * 0.5], 0.045), sr, 2100, 5.0),
    "a metal bolt sliding":
        lambda n, sr, g, secs: _band(_hits(n, sr, g, _even(secs * 0.4, 6, 0.035, 0.010, g),
                                           0.030), sr, 1800, 4.0),
    "keys on a ring":
        lambda n, sr, g, secs: _band(_hits(n, sr, g, _even(secs * 0.3, 7, 0.055, 0.025, g),
                                           0.030), sr, 5200, 8.0),
    "a zip running":
        lambda n, sr, g, secs: _band(_hits(n, sr, g, _even(secs * 0.35, 70, 0.0065, 0.0012, g),
                                           0.006), sr, 4800, 5.0),
    "velcro tearing open":
        lambda n, sr, g, secs: _band(_hits(n, sr, g, _even(secs * 0.35, 120, 0.004, 0.0015, g),
                                           0.005), sr, 3000, 1.6),
    "tape pulling off":
        lambda n, sr, g, secs: _band(_hits(n, sr, g, _even(secs * 0.3, 90, 0.007, 0.002, g),
                                           0.008), sr, 2400, 2.0),
    # Cloth moving is continuous while it moves -- 3.3 bursts a second reads as
    # somebody patting it. Rougher and faster than water, because a fold catches.
    "fabric rustling":
        lambda n, sr, g, secs: _flow(n, sr, g, 2900, q=0.85, rough=17.0, depth=0.75),
    "blades through fabric":
        lambda n, sr, g, secs: _band(_hits(n, sr, g, _even(secs * 0.3, 5, 0.18, 0.05, g),
                                           0.09), sr, 3600, 2.2),
    # A slow rhythm of frame creaks. Low and wooden, and the rate is deliberately
    # unhurried: the point is that the room is not silent, not that the shot has a
    # metronome in it.
    #
    # ...WHICH IS EXACTLY WHAT IT BECAME. An impulse train at 240 Hz with nothing
    # above 1.2 kHz, 97 of them a minute: a soft low thump at pulse rate, which is
    # the same defect the footsteps had and the same sound. The rate was never the
    # problem and it is unchanged -- wood creaking is what this is, so it creaks.
    "a bed frame working":
        lambda n, sr, g, secs: _creak(n, sr, g,
                                      _even(0.15, max(3, int(secs * 1.6)), 0.62,
                                            0.09, g),
                                      f0=205, secs=0.34, glide=1.45, slip=19.0),
    # Both halves, because the phrase names both: the buckle is metal and strikes,
    # the leather creaks. Built as one impact train it was all buckle.
    "a buckle and leather creaking":
        lambda n, sr, g, secs: (
            _band(_hits(n, sr, g, _even(secs * 0.3, 3, 0.22, 0.07, g), 0.030),
                  sr, 2800, 5.0) * 2.2
            + _creak(n, sr, g, _even(secs * 0.32, 3, 0.24, 0.08, g),
                     f0=330, secs=0.26, glide=1.5, slip=26.0)),
    # THE ONE THE REPORT WAS ABOUT. 130 Hz, nothing above 1.2 kHz at all, 109 evenly
    # spaced soft thumps a minute: every measurable property of a heartbeat. The
    # contact is back (that is _contact), and the metronome is a gait (that is
    # _gait). See both, and see _MOTION_TIMED for the timing half.
    "footsteps":
        lambda n, sr, g, secs: _footfalls(n, sr, g, _gait(secs, g)),
    # DRAGGING IS CONTINUOUS. Thirty-two bursts with gaps between them is something
    # being bumped along, not slid: a scrape is unbroken contact, and the gaps were
    # 63% of the sound.
    "something dragging on the floor":
        lambda n, sr, g, secs: _flow(n, sr, g, 950, q=0.65, rough=9.0, depth=0.55),
    # Landing is a contact too -- 110 Hz and nothing above 1.2 kHz was a thud with
    # no floor in it. Less contact in the mix than a footstep: a dropped thing is
    # mostly mass, where a shoe is mostly surface.
    "something landing":
        lambda n, sr, g, secs: _contact(n, sr, g, [secs * 0.5], body=105, bright=1500,
                                        decay=0.12, mix=0.40),
    "a sharp impact":
        lambda n, sr, g, secs: _contact(n, sr, g, [secs * 0.45], body=160, bright=2900,
                                        decay=0.065, mix=0.62),
    # A HINGE SQUEALS. Eight knocks at 780 Hz is a door being rapped, not one
    # swinging: the sound of a hinge is one long rising tone for the length of the
    # swing, and the metal gripping and releasing is what makes it waver.
    "a door on its hinges":
        lambda n, sr, g, secs: _creak(n, sr, g, [secs * 0.3], f0=620,
                                      secs=min(0.9, max(0.35, secs * 0.32)),
                                      glide=1.55, slip=23.0),
    # THE OTHER ONE THE REPORT WAS ABOUT. 14 discrete bursts a second at 1.4 kHz,
    # present only 41% of the time, with 43% jitter on the gaps: irregular mid-high
    # clicks, which is tapping. Reported from a bathroom, where it was the shower.
    # Water is a flow -- broad, bright and unbroken -- with the roughness of the
    # spray in it rather than the gaps of a dripping tap.
    "water":
        lambda n, sr, g, secs: _flow(n, sr, g, 2400, q=0.55, rough=13.0, depth=0.4),
}


def phrase_seed(phrase):
    """A stable number for a phrase. Python's own hash() is salted per process, so
    the same chain would sound like a different chain on every relaunch."""
    h = 0
    for ch in str(phrase or ""):
        h = (h * 131 + ord(ch)) & 0x7fffffff
    return h


def foley_for(phrase, n, sr, seed=0, motion=None, fps=24.0):
    """Build the sound `phrase` names, `n` samples long. None when there is no
    recipe -- which includes every vocal phrase, deliberately.

    `motion` is the shot's own movement envelope, one value per frame gap, from
    runtime.motion_envelope. Only footsteps use it, and only footsteps have a sync
    point worth the trouble -- see _MOTION_TIMED. Without it the gait is laid blind,
    which is what every recipe here did before."""
    try:
        n, sr = int(n), int(sr)
        make = _FOLEY.get(str(phrase or ""))
        if make is None or n < 64 or sr <= 0:
            return None
        # THE PROP'S VOICE IS THE PROP'S, and it does not change between shots.
        #
        # The seed was the film's seed plus the SHOT INDEX, and it drives the noise
        # texture, the event jitter and the room together -- so the same chain, named
        # in three consecutive beats, was built three times from three different
        # generators and came back as three different chains in three different rooms.
        # Reported as the sounds not being the same per beat, and it is the same
        # complaint the picture side has about a face with nothing holding it.
        #
        # Seeded from the PHRASE instead: one object, one voice, every time it is
        # named. Shot length still varies the event times, because the recipes lay
        # them out across `secs`, so two shots of a chain are not a copy of each other
        # unless they are the same length doing the same thing -- which is the one
        # case where being identical is right.
        g = torch.Generator().manual_seed((int(seed) + phrase_seed(phrase)) & 0x7fffffff)
        if motion is not None and str(phrase) in _MOTION_TIMED:
            times = _walk(n, sr, g, motion, float(fps) or 24.0)
            if not times:
                return None          # the picture never moves: no steps belong here
            y = _footfalls(n, sr, g, times)
        else:
            y = make(n, sr, g, n / float(sr))
        # The room goes on LAST and on everything, which is what a room does: it is
        # a property of the place, not of the prop. Applied here rather than in the
        # recipes so all 21 get it and none can forget it.
        #
        # ...and a property of the PLACE cannot be re-rolled per shot, which is what
        # the shot-index seed was doing: the reverb tail changed at every cut, so the
        # room itself sounded like it was being rebuilt between beats.
        y = _room(y, sr, seed=(int(seed) + 977) & 0x7fffffff)
        peak = float(y.abs().max())
        if not (peak > 0.0) or not torch.isfinite(y).all():
            return None
        return y * (0.7 / peak)
    except Exception:
        return None


def plain_bed(n, sr, seed=0, channels=2):
    """The last-resort bed: noise and a moving average, and nothing else.

    synth_ambient is defensive, so it can return None -- and a built bed that comes
    back empty leaves the output with no ambience at all. Wiring a file is NOT the
    remedy for that: the built bed is the feature, and a file is only ever an
    override for a real location. So there is a floor under it.

    Deliberately primitive. No FFT, no envelope, no recipe -- a cumulative-sum box
    filter over white noise, which is a rumble, and which cannot fail on any input
    the caller can hand it. It is not as good as the shaped bed and does not try to
    be; it is the difference between a quiet room and nothing at all."""
    try:
        n, sr, channels = int(n), int(sr), max(1, int(channels))
        if n < 8 or sr <= 0:
            return None
        g = torch.Generator().manual_seed(int(seed) & 0x7fffffff)
        y = torch.randn((channels, n), generator=g)
        # Box filter by cumulative sum: out[i] = mean(w[i-k:i]). k sets the corner.
        #
        # CASCADED THREE TIMES, which was measured rather than assumed. One pass is
        # a sinc, whose first sidelobe is only -13 dB -- against white noise, which
        # has equal energy per Hz, enough leaks through the whole top of the band to
        # put the spectral centroid at 3.3 kHz. That is a hiss, not the rumble this
        # is meant to be. Three passes is sinc^3, and the centroid lands where the
        # description says.
        k = max(2, min(n // 4, int(sr / 200)))          # ~200 Hz
        for _ in range(3):
            c = torch.cumsum(torch.nn.functional.pad(y, (k, 0)), dim=-1)
            y = (c[..., k:] - c[..., :-k])[..., :n] / float(k)
        rms = float(y.pow(2).mean().sqrt())
        if not (rms > 0.0) or not torch.isfinite(y).all():
            return None
        y = y * (_BED_RMS / rms)
        peak = float(y.abs().max())
        return y * (0.95 / peak) if peak > 0.95 else y
    except Exception:
        return None


def _seamless_loop(x, n, sr):
    """[C, M] -> [C, n], looped with a crossfade so the join does not click.

    Plain tiling puts a discontinuity at every repeat, once per loop length. In a
    bed that is meant to sit under everything unnoticed, a regular click is the one
    thing that gets noticed -- the same objection that made the silence latent
    ping-pong its interior rather than tile it. Here the material is real audio
    being PLAYED rather than a latent being conditioned on, so it cannot be
    reversed: a room tone read backwards is fine, but footsteps are not. Crossfade
    instead, which works on both."""
    m = int(x.shape[-1])
    if m <= 0:
        return None
    if m >= n:
        return x[..., :n]
    fade = min(int(0.25 * sr), m // 4)
    if fade < 1:
        reps = -(-n // m)
        return x.repeat(1, reps)[..., :n]
    # OVERLAP-ADD the tail onto the head, and shorten the unit by the overlap. The
    # unit then runs x[m-fade] .. x[m-fade-1], so tiling it steps between samples
    # that were adjacent in the source and there is no discontinuity anywhere.
    #
    # Measured, because the obvious construction is wrong: appending the crossfade
    # to the END of a full-length unit leaves it finishing on x[fade-1] while the
    # next repeat starts on x[0], which are not adjacent -- a 2s tone that does not
    # divide evenly gave a 64x jump at the join, worse than plain tiling's 41x.
    t = torch.linspace(0.0, 1.0, fade, dtype=x.dtype, device=x.device)
    head = x[..., :fade] * t + x[..., m - fade:] * (1.0 - t)
    unit = torch.cat([head, x[..., fade:m - fade]], dim=-1)
    if int(unit.shape[-1]) < 1:
        reps = -(-n // m)
        return x.repeat(1, reps)[..., :n]
    reps = -(-n // int(unit.shape[-1]))
    return unit.repeat(1, reps)[..., :n]


def mix_ambient(audio, sr, bed, level):
    """Lay an ambient bed UNDER a finished soundtrack. -> (waveform, note).

    The bed is PLAYED, not conditioned on: it is the file, at the level asked for,
    under whatever the model generated. That is the whole reason to do it here
    rather than in the sampler -- ambience needs no cooperation from a joint model,
    has nothing to lip-sync to, and so cannot put a voice in a wordless shot. The
    conditioning path can only steer the branch toward something bed-LIKE, and on a
    shot with a line it competes with the line.

    Defensive throughout, like the silence latent: any failure returns the audio
    untouched with a note saying so, because a bed is a nicety and a render is not.
    """
    try:
        if audio is None or bed is None or float(level or 0.0) <= 0.0:
            return audio, ""
        w = bed.get("waveform") if isinstance(bed, dict) else None
        if w is None or not int(getattr(w, "ndim", 0)):
            return audio, ("ambient_audio is wired but carries no waveform, so nothing "
                           "was laid under the soundtrack")
        w = w[0] if w.dim() == 3 else w              # [B, C, M] -> [C, M]
        if w.dim() != 2 or w.shape[-1] < 2:
            return audio, ("ambient_audio is too short to loop, so nothing was laid "
                           "under the soundtrack")
        w = w.detach().to(dtype=audio.dtype, device=audio.device)
        b_sr = int((bed.get("sample_rate") if isinstance(bed, dict) else 0) or 0)
        # RESAMPLE, or the bed plays at the wrong speed and pitch. Linear is coarse
        # for music and inaudible on a room tone, which is what this input is for.
        resampled = ""
        if b_sr > 0 and b_sr != int(sr):
            want = max(2, int(round(w.shape[-1] * float(sr) / float(b_sr))))
            w = torch.nn.functional.interpolate(
                w.unsqueeze(0), size=want, mode="linear", align_corners=False)[0]
            resampled = f", resampled from {b_sr} Hz"
        ch = int(audio.shape[1])
        if int(w.shape[0]) != ch:
            w = (w.mean(dim=0, keepdim=True).repeat(ch, 1) if int(w.shape[0]) > ch
                 else w[:1].repeat(ch, 1))
        n = int(audio.shape[-1])
        loop = _seamless_loop(w, n, int(sr))
        if loop is None:
            return audio, ""
        out = audio + loop.unsqueeze(0) * float(level)
        # NORMALISE rather than clip. Clipping a bed that pushed a loud line over
        # the top distorts the LINE, which is the thing worth keeping.
        peak = float(out.abs().max())
        gain = ""
        if peak > 1.0:
            out = out / peak
            gain = f", and the mix was scaled by {1.0 / peak:.2f} to stop it clipping"
        secs = w.shape[-1] / float(sr)
        return out, (f"an ambient bed was laid under the whole soundtrack at level "
                     f"{float(level):.2f} -- {secs:.1f}s of audio{resampled}, looped "
                     f"with a crossfade so the join does not click{gain}. It is your "
                     f"file, played under what the model generated: it conditions "
                     f"nothing, so it cannot put a voice in a wordless shot the way "
                     f"an inferred bed did. Shots pinned to silence keep their silent "
                     f"conditioning and get the bed on top, which is what makes a "
                     f"wordless shot sound like a room instead of a mute")
    except Exception as exc:
        return audio, (f"the ambient bed could not be mixed ({type(exc).__name__}), so "
                       f"the soundtrack is unchanged")


_SILENCE_STATUS = {"asked": 0, "applied": 0, "why": ""}


_SILENT_SECONDS = 2


_SILENT_EDGE = 4


def _silent_audio_latent(audio_vae, frame_count, fps):
    """A keyframe audio latent of actual SILENCE, or None if it cannot be made.

    H3 is a JOINT model: the mouth follows the audio branch. On a shot with no
    scripted line the branch is otherwise unconditioned, and an unconditioned audio
    branch invents a voice -- which the picture then lip-syncs to. The lips-closed
    sentence is arguing with a stream that has already decided someone is talking.

    REBUILT 2026-09-05, from measurements against the real VAE rather than from
    reasoning. The previous version encoded one second, kept a SINGLE interior
    frame and repeated it, on the argument that silence is homogeneous. It is not,
    in latent space: encoded silence has genuine frame-to-frame variation (delta
    mean 0.002-0.004, max 0.021), and a repeated frame has a delta of exactly
    0.000000. That is a flat signal no encoder produces, and a model handed
    conditioning outside its own distribution has every reason to disregard it --
    which is an audio branch back to inventing a voice, with the report saying
    silence went on.

    The fix that version was avoiding is real too: tiling the whole encoded second
    end to end leaves a 25x spike at each join (0.554 against 0.022), once per
    second, which is a metronome in the conditioning of a joint model.

    So: encode two seconds, drop the padded ends, and PING-PONG the interior --
    forward, reversed, forward. Every join repeats a frame, so there is no seam,
    and the interior statistics are the encoder's own. Measured over a 9s shot:

        one frame repeated   peak 0.000686   delta mean 0.000000   max 0.000000
        whole 2s tiled       peak 0.000314   delta mean 0.017451   max 0.554715
        interior ping-pong   peak 0.000566   delta mean 0.002039   max 0.021159

    where the encoder's own interior is mean 0.0021, max 0.0212. Decoded peak
    0.000566 on a +/-1.0 scale is about -65 dBFS: silence.

    Everything here stays defensive. Shapes are CHECKED against what the layout
    expects rather than assumed, and any failure returns None so the shot falls
    back to an unconditioned branch instead of breaking the render -- the caller
    reports when that happens, so it is no longer a silent failure.
    """
    try:
        sr = int(getattr(audio_vae, "audio_sample_rate", 0) or 0)
        if sr <= 0:
            return None
        _, _, want_t = temporal_shape(frame_count, fps)
        if want_t <= 0:
            return None
        key = (id(audio_vae), sr)
        block = _SILENT_UNIT.get("lat") if _SILENT_UNIT.get("key") == key else None
        if block is None:
            # CHANNELS LAST. comfy.sd.VAE.encode() does `pixel_samples.movedim(-1, 1)`
            # before handing off, so the audio VAE -- which wants [B, 2, L] -- must be
            # given [B, L, 2]. Passing [B, 2, L] raises inside the encoder, and an
            # early version did exactly that: swallowed by the guard below, so the
            # whole layer silently did nothing.
            #
            # Two seconds, encoded ONCE and cached. Encoding a full 15s shot instead
            # cost a VAE pass big enough to OOM mid-render on a 16GB card, where the
            # failure again degraded silently to no conditioning at all.
            enc = audio_vae.encode(torch.zeros((1, sr * _SILENT_SECONDS, 2)))
            if enc is None or enc.dim() != 4 or enc.shape[1] != 32:
                return None
            if enc.shape[-1] <= 2 * _SILENT_EDGE + 1:
                return None
            block = enc[..., _SILENT_EDGE:-_SILENT_EDGE].detach().to("cpu").clone()
            _SILENT_UNIT["lat"] = block
            _SILENT_UNIT["key"] = key
        n = block.shape[-1]
        if n < 1:
            return None
        # Forward, reversed, forward... Each join repeats a frame, so the seam that
        # plain tiling leaves is gone while the interior variation is the encoder's.
        pieces, have, i = [], 0, 0
        while have < want_t:
            piece = block if i % 2 == 0 else torch.flip(block, dims=[-1])
            pieces.append(piece)
            have += n
            i += 1
        out = torch.cat(pieces, dim=-1)[..., :want_t].clone()
        if out.shape[-1] != want_t:
            return None
        return out
    except Exception:
        return None                         # never fail a render for a nicety


def _pin_audio_silence(latent, silence, lead_frames=None, tail_frames=0):
    """Start target audio at encoded silence and preserve the requested span(s).

    lead_frames None pins the whole shot. Otherwise the first lead_frames and the
    last tail_frames are held at silence and the span between is left to the model
    -- that is where the line goes. The tail is clipped to what the lead leaves, so
    the two can never overlap. Nothing pinned at all is a no-op, reported as False
    so the caller does not count it as applied."""
    try:
        video, audio = latent["samples"].unbind()
        silence = silence.to(device=audio.device, dtype=audio.dtype)
        if silence.shape != audio.shape:
            return False
        audio_mask = torch.ones_like(audio[:, :1])
        if lead_frames is None:
            audio_mask.zero_()
        else:
            t = audio.shape[-1]
            n = min(t, max(0, int(lead_frames)))
            m = min(t - n, max(0, int(tail_frames or 0)))
            if n <= 0 and m <= 0:
                return False
            if n > 0:
                audio_mask[..., :n] = 0
            if m > 0:
                audio_mask[..., t - m:] = 0
        latent["samples"] = comfy.nested_tensor.NestedTensor((video, silence))
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video[:, :1]), audio_mask))
        return True
    except Exception:
        return False
