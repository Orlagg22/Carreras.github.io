#!/usr/bin/env python3
"""
Generates Noryx's soundtrack: 14 original tracks, synthesised from scratch (no samples, no recordings), so they are
free of any third-party copyright. Released as CC0.

  python3 scripts/generate-soundtrack.py            # all tracks -> public/music/<id>.mp3
  python3 scripts/generate-soundtrack.py redline    # only some

Needs: numpy, scipy and ffmpeg (with libmp3lame). Tracks are deterministic (seeded), so a re-run gives the same files.
The track list (titles, pools) is mirrored in src/lib/audio/music-catalog.ts.
"""
import math
import os
import subprocess
import sys

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter

SR = 44100
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "music")

SCALES = {
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
}
CHORDS = {"M": [0, 4, 7], "m": [0, 3, 7], "M9": [0, 4, 7, 14], "m9": [0, 3, 7, 14], "M7": [0, 4, 7, 11], "m7": [0, 3, 7, 10]}

# id, title, pool, style, bpm, root (MIDI of the key's tonic around C3), scale, progression [(semitones, quality)], bars, options
TRACKS = [
    # --- menus: calm, spacious ---
    dict(id="pit-lane-dawn", title="Pit Lane Dawn", pool="menu", style="menu", bpm=76, root=50, scale="lydian",
         prog=[(0, "M9"), (2, "M"), (7, "M9"), (9, "m7")], bars=48, seed=11),
    dict(id="quiet-paddock", title="Quiet Paddock", pool="menu", style="menu", bpm=70, root=45, scale="minor",
         prog=[(0, "m9"), (8, "M"), (3, "M9"), (10, "M")], bars=44, seed=12),
    dict(id="garage-light", title="Garage Light", pool="menu", style="menu", bpm=84, root=53, scale="major",
         prog=[(0, "M9"), (7, "M"), (9, "m7"), (5, "M9")], bars=52, seed=13),
    dict(id="slow-out-lap", title="Slow Out Lap", pool="menu", style="menu", bpm=68, root=52, scale="dorian",
         prog=[(0, "m9"), (5, "M"), (0, "m7"), (10, "M")], bars=44, seed=14),
    # --- racing: driving electronic ---
    dict(id="redline", title="Redline", pool="race", style="race", bpm=140, root=45, scale="minor",
         prog=[(0, "m"), (8, "M"), (3, "M"), (10, "M")], bars=80, bass="eighths", arp=[0, 2, 4, 2, 1, 3, 5, 3, 0, 2, 4, 2, 1, 3, 5, 4],
         lead="saw", seed=21),
    dict(id="apex-hunter", title="Apex Hunter", pool="race", style="race", bpm=132, root=50, scale="phrygian",
         prog=[(0, "m"), (1, "M"), (0, "m"), (10, "M")], bars=72, bass="gallop", arp=[0, 3, 1, 4, 2, 5, 3, 4, 0, 3, 1, 4, 2, 5, 4, 3],
         lead="saw", seed=22),
    dict(id="slipstream", title="Slipstream", pool="race", style="race", bpm=128, root=54, scale="minor",
         prog=[(0, "m"), (5, "m"), (8, "M"), (7, "M")], bars=72, bass="offbeat", arp=[0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3],
         lead="pluck", seed=23),
    dict(id="turbo-pulse", title="Turbo Pulse", pool="race", style="race", bpm=150, root=52, scale="minor",
         prog=[(0, "m"), (10, "M"), (8, "M"), (10, "M")], bars=80, bass="gallop", arp=[0, 4, 2, 4, 1, 5, 3, 5, 0, 4, 2, 4, 1, 5, 3, 4],
         lead="saw", seed=24),
    dict(id="chicane", title="Chicane", pool="race", style="race", bpm=124, root=48, scale="minor",
         prog=[(0, "m"), (3, "M"), (10, "M"), (5, "m")], bars=72, bass="offbeat", arp=[0, 2, 1, 3, 2, 4, 3, 5, 4, 2, 3, 1, 2, 0, 1, 2],
         lead="pluck", seed=25),
    dict(id="full-throttle", title="Full Throttle", pool="race", style="race", bpm=145, root=55, scale="minor",
         prog=[(0, "m"), (8, "M"), (10, "M"), (0, "m")], bars=80, bass="eighths", arp=[0, 2, 4, 5, 4, 2, 1, 3, 0, 2, 4, 5, 4, 2, 3, 1],
         lead="saw", seed=26),
    dict(id="late-braking", title="Late Braking", pool="race", style="race", bpm=136, root=47, scale="dorian",
         prog=[(0, "m"), (5, "M"), (10, "M"), (7, "m")], bars=80, bass="eighths", arp=[0, 3, 5, 3, 1, 4, 5, 4, 0, 3, 5, 3, 2, 4, 5, 3],
         lead="pluck", seed=27),
    # --- results: warm, uplifting ---
    dict(id="podium-glow", title="Podium Glow", pool="results", style="results", bpm=104, root=48, scale="major",
         prog=[(0, "M"), (7, "M"), (9, "m"), (5, "M")], bars=56, seed=31),
    dict(id="checkered-flag", title="Checkered Flag", pool="results", style="results", bpm=112, root=55, scale="major",
         prog=[(0, "M"), (5, "M"), (7, "M"), (5, "M")], bars=60, seed=32),
    dict(id="cool-down-lap", title="Cool Down Lap", pool="results", style="results", bpm=96, root=46, scale="major",
         prog=[(0, "M7"), (9, "m7"), (5, "M7"), (7, "M")], bars=52, seed=33),
]


def mtof(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def tvec(dur):
    return np.arange(int(dur * SR), dtype=np.float32) / SR


# ----------------------------------------------------------------------------------------------------------- synth
def tone(f, dur, kmax=16, tilt=1.0, tau=0.0, htau=0.0, atk=0.004, rel=0.03, cents=0.0, vib_hz=0.0, vib=0.0, odd=False):
    """Additive voice: harmonics 1..k with a 1/k^tilt spectrum, optional decay (faster in the highs), vibrato"""
    n = int((dur + rel) * SR)
    t = np.arange(n, dtype=np.float32) / SR
    kk = max(1, min(kmax, int(min(9000.0, 0.45 * SR) / f)))
    k = np.arange(1, kk + 1, dtype=np.float32)
    if odd:
        k = k[k % 2 == 1]
    fr = f * 2.0 ** (cents / 1200.0)
    ph = 2 * np.pi * fr * t
    if vib_hz:
        ph = ph + (2 * np.pi * fr * vib / (2 * np.pi * vib_hz)) * np.sin(2 * np.pi * vib_hz * t)
    h = np.sin(k[:, None] * ph[None, :]) / (k[:, None] ** tilt)
    if tau > 0:
        h *= np.exp(-t[None, :] * (1.0 + htau * (k[:, None] - 1.0)) / tau)
    y = h.sum(axis=0)
    env = np.minimum(1.0, t / max(atk, 1e-4))
    if rel > 0:
        tail = np.clip((dur + rel - t) / rel, 0.0, 1.0)
        env = env * tail
    return (y * env).astype(np.float32)


def bell(f, dur, tau=2.4):
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    y = np.zeros(n, np.float32)
    for r, a, d in ((1.0, 1.0, 1.0), (2.0, 0.45, 0.7), (2.76, 0.25, 0.5), (5.4, 0.12, 0.3), (8.93, 0.05, 0.2)):
        y += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / (tau * d))
    return (y * np.minimum(1.0, t / 0.003)).astype(np.float32)


def piano(f, dur, tau=1.1):
    return tone(f, dur, kmax=10, tilt=1.9, tau=tau, htau=1.6, atk=0.003, rel=0.12)


class Noise:
    def __init__(self, seed):
        self.buf = np.random.default_rng(seed).standard_normal(SR * 4).astype(np.float32)
        self.rng = np.random.default_rng(seed + 1)

    def get(self, n):
        i = int(self.rng.integers(0, len(self.buf) - n - 1))
        return self.buf[i:i + n].copy()


def kick(punch=1.0, length=0.45):
    t = tvec(length)
    fe = 46 + 120 * np.exp(-t / 0.028)
    ph = 2 * np.pi * np.cumsum(fe) / SR
    y = np.sin(ph) * np.exp(-t / 0.16) * punch
    y[:90] += np.linspace(0.5, 0, 90, dtype=np.float32)
    return y.astype(np.float32)


def snare(nz, soft=1.0):
    t = tvec(0.28)
    n = nz.get(len(t))
    n = np.diff(n, prepend=0).astype(np.float32)
    y = n * np.exp(-t / 0.055) * 0.55 + np.sin(2 * np.pi * 188 * t) * np.exp(-t / 0.05) * 0.5
    return (y * soft).astype(np.float32)


def hat(nz, open_=False):
    t = tvec(0.16 if open_ else 0.05)
    n = np.diff(np.diff(nz.get(len(t)), prepend=0), prepend=0).astype(np.float32)
    return (n * np.exp(-t / (0.07 if open_ else 0.014)) * 0.5).astype(np.float32)


def crash(nz, dur=1.8):
    t = tvec(dur)
    n = np.diff(nz.get(len(t)), prepend=0).astype(np.float32)
    return (n * np.exp(-t / 0.55) * 0.45).astype(np.float32)


def riser(nz, dur):
    t = tvec(dur)
    n = np.diff(nz.get(len(t)), prepend=0).astype(np.float32)
    return (n * (t / dur) ** 2.2 * 0.5).astype(np.float32)


def make_ir(rt60, seed):
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = np.random.default_rng(seed).standard_normal((n, 2)).astype(np.float32) * np.exp(-6.9 * t / rt60)[:, None]
    b, a = butter(1, 4200 / (SR / 2))
    ir = lfilter(b, a, ir, axis=0).astype(np.float32)
    ir[: int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR), dtype=np.float32)[:, None]
    return ir / np.sqrt((ir ** 2).sum(axis=0).mean())


# ----------------------------------------------------------------------------------------------------------- mix
class Mix:
    def __init__(self, seconds):
        self.n = int(seconds * SR) + SR * 4
        self.bus = {k: np.zeros((self.n, 2), np.float32) for k in ("drums", "bass", "arp", "pad", "lead")}
        self.kicks = []

    def add(self, bus, t0, sig, pan=0.0, gain=1.0):
        i = int(t0 * SR)
        if i < 0 or i >= self.n:
            return
        m = min(len(sig), self.n - i)
        a = (pan + 1) * math.pi / 4
        self.bus[bus][i:i + m, 0] += sig[:m] * math.cos(a) * gain
        self.bus[bus][i:i + m, 1] += sig[:m] * math.sin(a) * gain


def chord_pitches(root, ch):
    return [root + ch[0] + iv for iv in CHORDS[ch[1]]]


def sidechain(n, kicks, depth, rel=0.13):
    sc = np.ones(n, np.float32)
    L = int(0.4 * SR)
    seg = 1 - depth * np.exp(-np.arange(L, dtype=np.float32) / (SR * rel))
    for t in kicks:
        i = int(t * SR)
        m = min(L, n - i)
        if m > 0:
            sc[i:i + m] = np.minimum(sc[i:i + m], seg[:m])
    return sc


def pingpong(x, delay, fb, taps):
    mono = x.mean(axis=1)
    out = np.zeros_like(x)
    d = int(delay * SR)
    for k in range(1, taps + 1):
        s = k * d
        if s >= len(mono):
            break
        out[s:, k % 2] += mono[:-s] * (fb ** k)
    return out


def reverb(x, rt60, seed):
    ir = make_ir(rt60, seed)
    wet = np.stack([fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in (0, 1)], axis=1)
    return wet.astype(np.float32)


# ----------------------------------------------------------------------------------------------------------- melody
def build_melody(tr, rng, variant):
    """One melody for the progression: per chord, a list of (start in 16ths, length in 16ths, midi)"""
    sc = SCALES[tr["scale"]]
    lo, hi = tr["root"] + 19, tr["root"] + 38
    pool = sorted({tr["root"] + 12 * o + s for o in range(0, 4) for s in sc if lo <= tr["root"] + 12 * o + s <= hi})
    cells = [[4, 4, 4, 4], [6, 2, 4, 4], [3, 3, 2, 3, 3, 2], [2, 2, 4, 2, 2, 4], [8, 4, 4], [6, 6, 4], [4, 2, 2, 4, 4]]
    out = []
    prev = pool[len(pool) // 2]
    for ci, ch in enumerate(tr["prog"]):
        cpcs = {(tr["root"] + ch[0] + iv) % 12 for iv in CHORDS[ch[1]]}
        cell = cells[int(rng.integers(0, len(cells)))] if ci % 2 == 0 or variant else out_cell
        out_cell = cell
        pos = 0
        notes = []
        cands = [p for p in pool if p % 12 in cpcs]
        cur = min(cands, key=lambda p: abs(p - prev) + float(rng.random()) * 3)
        for j, d in enumerate(cell):
            if j > 0:
                idx = pool.index(cur) if cur in pool else len(pool) // 2
                step = int(rng.choice([-2, -1, -1, 0, 1, 1, 2]))
                if idx > len(pool) * 0.75:
                    step = -abs(step)
                if idx < len(pool) * 0.25:
                    step = abs(step)
                cur = pool[int(np.clip(idx + step, 0, len(pool) - 1))]
            rest = j > 0 and rng.random() < 0.12
            if j == len(cell) - 1:
                cur = min(cands, key=lambda p: abs(p - cur))
            if not rest:
                notes.append((pos, d, cur))
            pos += d
        prev = cur
        out.append(notes)
    return out


# ----------------------------------------------------------------------------------------------------------- render
def plan_race(bars):
    seq = [("intro", 8), ("build", 8), ("A", 16), ("break", 8), ("B", 16)]
    if bars >= 72:
        seq.append(("C", 8 if bars == 72 else 16))
    seq.append(("outro", 8))
    out = []
    for name, n in seq:
        out += [name] * n
    return out[:bars] + ["outro"] * max(0, bars - len(out))


def render_race(tr):
    rng = np.random.default_rng(tr["seed"])
    nz = Noise(tr["seed"] * 7)
    bpm = tr["bpm"]
    beat = 60.0 / bpm
    bar = beat * 4
    s16 = beat / 4
    bars = tr["bars"]
    secs = bars * bar
    mx = Mix(secs)
    plan = plan_race(bars)
    prog = tr["prog"]
    root = tr["root"]
    melA = build_melody(tr, np.random.default_rng(tr["seed"] + 100), False)
    melB = build_melody(tr, np.random.default_rng(tr["seed"] + 200), True)
    arp_pat = tr["arp"]
    cache = {}

    def cached(key, fn):
        if key not in cache:
            cache[key] = fn()
        return cache[key]

    kick_s, snare_s = kick(), snare(nz)
    for b in range(bars):
        sec = plan[b]
        pos_in_sec = b - plan.index(sec)
        sec_len = plan.count(sec)
        t0 = b * bar
        ch = prog[b % len(prog)]
        cp = chord_pitches(root, ch)
        drums = sec in ("A", "B", "C", "outro") or (sec == "build" and pos_in_sec >= 4)
        full = sec in ("A", "B", "C")
        # pad
        padamp = {"intro": 0.7, "build": 0.7, "break": 1.0}.get(sec, 0.5)
        for p in cp:
            f = mtof(p + 12)
            for cents, pan in ((-9, -0.7), (9, 0.7)):
                mx.add("pad", t0, cached(("pad", p, cents, round(bar, 3)), lambda: tone(f, bar, kmax=10, tilt=1.5, atk=0.7, rel=0.9, cents=cents)), pan, 0.16 * padamp)
        # arp
        if sec != "outro" or pos_in_sec < 5:
            tones = []
            for o in (0, 12, 24):
                tones += [p + 12 + o for p in cp]
            rate = 2 if sec in ("intro", "break") else 1
            amp = {"intro": 0.5, "build": 0.7, "break": 0.5}.get(sec, 0.8)
            if sec == "outro":
                amp *= max(0.0, 1 - pos_in_sec / 5)
            for i in range(0, 16, rate):
                idx = arp_pat[i] % len(tones)
                p = tones[idx]
                f = mtof(p)
                sig = cached(("arp", p, round(s16 * rate, 4)), lambda: tone(f, s16 * rate * 1.2, kmax=14, tilt=1.05, tau=0.13, htau=0.5, rel=0.04))
                mx.add("arp", t0 + i * s16, sig, (-0.55, 0.55)[(i // rate) % 2], 0.30 * amp)
        # bass
        bass_on = full or sec == "outro" or (sec == "build" and pos_in_sec >= 4) or sec == "break"
        if bass_on:
            r = root + ch[0] - 12
            f = mtof(r)
            style = tr["bass"]
            hits = []
            if sec == "break":
                hits = [(0, 16, 0)]
            elif style == "eighths":
                hits = [(i * 2, 2, 12 if i % 2 else 0) for i in range(8)]
            elif style == "gallop":
                pat = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15]
                hits = [(i, 1, 12 if i in (3, 11) else 0) for i in pat]
            else:  # offbeat: the bass sits between the kicks
                hits = [(i * 4 + 2, 2, 0) for i in range(4)]
            for st, ln, oct_ in hits:
                ff = mtof(r + oct_)
                sig = cached(("bass", r + oct_, ln, round(s16, 4)), lambda: tone(ff, s16 * ln * 0.95, kmax=8, tilt=1.3, tau=0.5 if ln > 4 else 0.2, htau=0.4, atk=0.004, rel=0.03))
                mx.add("bass", t0 + st * s16, sig, 0.0, 0.5 if sec != "break" else 0.35)
        # drums
        if drums:
            amp = 1.0
            if sec == "outro":
                amp = max(0.0, 1 - max(0, pos_in_sec - 2) / 6)
            for q in range(4):
                if sec == "build" and pos_in_sec < 4:
                    break
                tk = t0 + q * beat
                mx.add("drums", tk, kick_s, 0.0, 0.9 * amp)
                mx.kicks.append(tk)
            for q in (1, 3):
                if full or (sec == "build" and pos_in_sec >= 4):
                    mx.add("drums", t0 + q * beat, snare_s, 0.05, 0.55 * amp)
            for i in range(16):
                if i % 2 == 1 or (full and i % 4 == 2):
                    open_ = i % 4 == 2 and full
                    mx.add("drums", t0 + i * s16, hat(nz, open_), (-0.3, 0.3)[i % 4 // 2], (0.2 if i % 4 else 0.12) * amp)
        elif sec == "intro" and pos_in_sec >= 4:
            for i in range(2, 16, 4):
                mx.add("drums", t0 + i * s16, hat(nz, False), 0.2, 0.14)
        # fills, crash, risers
        if sec in ("A", "B", "C") and pos_in_sec == 0 or (sec == "outro" and pos_in_sec == 0):
            mx.add("drums", t0, crash(nz), 0.2, 0.5)
        if sec == "build" and pos_in_sec == sec_len - 1:
            for i in range(8, 16):
                mx.add("drums", t0 + i * s16, snare(nz, 0.35 + 0.08 * (i - 8)), 0.0, 0.5)
            mx.add("drums", t0, riser(nz, bar), 0.0, 0.5)
        if sec == "break" and pos_in_sec == sec_len - 1:
            mx.add("drums", t0, riser(nz, bar), 0.0, 0.6)
            for i in range(8, 16):
                mx.add("drums", t0 + i * s16, snare(nz, 0.3 + 0.07 * (i - 8)), 0.0, 0.45)
        # lead
        lead_on = sec in ("break", "B", "C")
        if lead_on and not (sec == "break" and pos_in_sec == sec_len - 1):
            mel = melA if sec == "break" else melB
            notes = mel[b % len(prog)]
            up = 12 if sec == "C" else 0
            for st, ln, p in notes:
                f = mtof(p + up - 12)
                kind = tr["lead"]
                dur = ln * s16 * (1.05 if kind == "saw" else 0.9)
                if kind == "saw":
                    sig = tone(f, dur, kmax=16, tilt=0.95, tau=1.0, htau=0.15, atk=0.008, rel=0.07, vib_hz=5.2, vib=0.004, cents=4)
                    sig += tone(f, dur, kmax=16, tilt=0.95, tau=1.0, htau=0.15, atk=0.008, rel=0.07, vib_hz=5.2, vib=0.004, cents=-4)
                    sig *= 0.5
                else:
                    sig = tone(f, dur, kmax=12, tilt=1.2, tau=0.22, htau=0.5, atk=0.004, rel=0.05)
                mx.add("lead", t0 + st * s16, sig, 0.1, 0.4 if sec == "break" else 0.46)
    return mix_down(mx, secs, drum_sc=0.55, wet=dict(pad=0.30, arp=0.20, lead=0.28), rt=1.9, delay=(beat * 0.75, 0.45, 4), fade_in=0.05, fade_out=7.0)


def render_menu(tr):
    rng = np.random.default_rng(tr["seed"])
    bpm = tr["bpm"]
    bar = 4 * 60.0 / bpm
    bars = tr["bars"]
    secs = bars * bar
    mx = Mix(secs)
    prog = tr["prog"]
    root = tr["root"]
    sc = SCALES[tr["scale"]]
    mel = build_melody(tr, np.random.default_rng(tr["seed"] + 50), False)
    beat = bar / 4
    for b in range(bars):
        t0 = b * bar
        ch = prog[(b // 2) % len(prog)]
        cp = chord_pitches(root, ch)
        first = b % 2 == 0
        if first:
            for p in cp:
                for cents, pan in ((-7, -0.8), (7, 0.8), (0, 0.0)):
                    mx.add("pad", t0, tone(mtof(p + 12), bar * 2, kmax=9, tilt=1.7, atk=1.6, rel=1.8, cents=cents), pan, 0.17)
            if b >= 6:
                mx.add("bass", t0, tone(mtof(root + ch[0] - 12), bar * 2, kmax=3, tilt=1.2, atk=0.4, rel=1.2), 0.0, 0.55)
        if b >= 3 and b < bars - 4:
            tones = [p + 24 for p in cp] + [p + 36 for p in cp]
            for i in range(8):
                if rng.random() < (0.55 if b < 12 else 0.7):
                    p = tones[int(rng.integers(0, len(tones)))]
                    mx.add("arp", t0 + i * beat / 2, bell(mtof(p), 3.0), float(rng.uniform(-0.7, 0.7)), 0.16)
        if 14 <= b < bars - 6:
            for st, ln, p in mel[(b // 2) % len(prog)] if b % 2 == 0 else []:
                f = mtof(p - 12)
                sig = tone(f, ln * beat / 4 * 1.5, kmax=5, tilt=1.8, tau=1.4, htau=0.8, atk=0.02, rel=0.4, vib_hz=4.6, vib=0.003)
                mx.add("lead", t0 + st * beat / 4 * 2, sig, 0.15, 0.30)
    return mix_down(mx, secs, drum_sc=0.0, wet=dict(pad=0.55, arp=0.55, lead=0.5), rt=3.6, delay=(beat * 1.5, 0.4, 3), fade_in=3.0, fade_out=9.0)


def render_results(tr):
    rng = np.random.default_rng(tr["seed"])
    nz = Noise(tr["seed"] * 5)
    bpm = tr["bpm"]
    beat = 60.0 / bpm
    bar = beat * 4
    bars = tr["bars"]
    secs = bars * bar
    mx = Mix(secs)
    prog = tr["prog"]
    root = tr["root"]
    mel = build_melody(tr, np.random.default_rng(tr["seed"] + 70), False)
    kick_s = kick(0.7)
    for b in range(bars):
        t0 = b * bar
        ch = prog[b % len(prog)]
        cp = chord_pitches(root, ch)
        outro = b >= bars - 6
        for p in cp:
            for cents, pan in ((-8, -0.6), (8, 0.6)):
                mx.add("pad", t0, tone(mtof(p + 12), bar, kmax=10, tilt=1.5, atk=0.5, rel=0.8, cents=cents), pan, 0.15)
        # piano arpeggio, eighths
        tones = [p + 12 for p in cp] + [p + 24 for p in cp]
        order = [0, 1, 2, 3, 2, 1, 2, 4] if len(cp) == 3 else [0, 1, 2, 3, 2, 1, 3, 2]
        for i in range(8):
            p = tones[order[i] % len(tones)]
            mx.add("arp", t0 + i * beat / 2, piano(mtof(p), beat), (-0.3, 0.3)[i % 2], 0.34)
        if b >= 4 and not outro:
            mx.add("bass", t0, tone(mtof(root + ch[0] - 12), beat * 2, kmax=7, tilt=1.4, tau=0.9, atk=0.01, rel=0.1), 0.0, 0.5)
            mx.add("bass", t0 + beat * 2, tone(mtof(root + ch[0] - 12 + (7 if ch[1][0] == "M" else 0)), beat * 2, kmax=7, tilt=1.4, tau=0.9, atk=0.01, rel=0.1), 0.0, 0.42)
        if 8 <= b and not outro:
            for q in (0, 2):
                tk = t0 + q * beat
                mx.add("drums", tk, kick_s, 0.0, 0.6)
                mx.kicks.append(tk)
            for i in range(8):
                mx.add("drums", t0 + i * beat / 2, hat(nz, False), (-0.35, 0.35)[i % 2], 0.12)
        if 16 <= b and not outro:
            for q in (1, 3):
                mx.add("drums", t0 + q * beat, snare(nz, 0.5), 0.0, 0.4)
        if 12 <= b < bars - 8:
            for st, ln, p in mel[b % len(prog)]:
                mx.add("lead", t0 + st * beat / 4, piano(mtof(p), ln * beat / 4 * 1.6, tau=1.3), 0.15, 0.36)
        if b % 8 == 0 and b > 0:
            mx.add("drums", t0, crash(nz, 2.4), 0.2, 0.3)
    return mix_down(mx, secs, drum_sc=0.25, wet=dict(pad=0.35, arp=0.3, lead=0.3), rt=2.4, delay=(beat * 0.75, 0.35, 3), fade_in=0.4, fade_out=7.0)


def mix_down(mx, secs, drum_sc, wet, rt, delay, fade_in, fade_out):
    n = mx.n
    sc = sidechain(n, mx.kicks, drum_sc) if drum_sc > 0 else np.ones(n, np.float32)
    scc = sc[:, None]
    pad, arp, bass, lead, drums = mx.bus["pad"] * scc, mx.bus["arp"] * (0.35 + 0.65 * scc), mx.bus["bass"] * scc, mx.bus["lead"], mx.bus["drums"]
    send = pad * wet["pad"] + arp * wet["arp"] + lead * wet["lead"]
    echo = pingpong(arp * 0.5 + lead * 0.6, *delay)
    rev = reverb(send + echo * 0.6, rt, 99)
    out = pad + arp + bass + lead + drums + echo * 0.5 + rev
    b, a = butter(2, 28 / (SR / 2), btype="high")
    out = lfilter(b, a, out, axis=0).astype(np.float32)
    total = int((secs + 1.5) * SR)
    out = out[:total]
    t = np.arange(len(out), dtype=np.float32) / SR
    env = np.minimum(1.0, t / max(fade_in, 1e-3)) * np.clip((secs + 1.5 - t) / fade_out, 0, 1) ** 1.5
    out *= env[:, None]
    out = np.tanh(out * 1.15) / np.tanh(1.15)
    peak = float(np.abs(out).max())
    return (out * (0.9 / max(peak, 1e-6))).astype(np.float32)


def encode(audio, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
           "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "44100", "-codec:a", "libmp3lame", "-b:a", "96k", path]
    subprocess.run(cmd, input=audio.tobytes(), check=True)


def main():
    want = set(sys.argv[1:])
    for tr in TRACKS:
        if want and tr["id"] not in want:
            continue
        fn = {"race": render_race, "menu": render_menu, "results": render_results}[tr["style"]]
        audio = fn(tr)
        path = os.path.join(OUT, tr["id"] + ".mp3")
        encode(audio, path)
        rms = float(np.sqrt((audio ** 2).mean()))
        print(f"{tr['id']:16s} {len(audio) / SR:6.1f}s  rms {rms:.3f}  nan={bool(np.isnan(audio).any())}  {os.path.getsize(path) / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
