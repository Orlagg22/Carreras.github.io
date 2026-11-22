#!/usr/bin/env python3
"""
Rebuilds the interior of the ported "Modern Cartoon Sports Car" (after scripts/cars/add_cartoon_interior.py and
scripts/cars/remove_cartoon_lid.py) so the cockpit and dash views read as a real car's, in the spirit of the F1 car's:

  - lighter, two-tone materials (the first interior's near-black boxes went navy-black under the game's lighting)
  - a dashboard with an instrument binnacle holding a textured cluster (speedometer, rev counter, trip screen), a centre
    screen, round air vents, a door card with armrest, handle and speaker, seats with red inserts, a console with a
    shifter, A-pillar trims, sun visors and an interior mirror
  - a leather steering wheel (own `STEER` node, column along local +Z, tilted 24 degrees, which the game turns about
    that axis) with a flat top and bottom, metal spokes and a hub badge. The driver's gloved hands and forearms are the game's own
    (src/lib/render/driver-arms.ts), switched on by the `driver` entry this script writes to the GLB's extras
  - the cockpit and dash cameras: the cockpit camera is the driver's eye, the dash camera sits just behind the wheel so
    the rim, the hands and the cluster fill the view

The pictures (cluster, screen, badge) are drawn here with PIL and embedded in the GLB as PNG. The script drops the
previous interior (every `cabin_*` primitive and the old STEER node) first, so it is safe to run on its own output.

  python3 scripts/cars/rebuild_cartoon_interior.py IN.glb OUT.glb
"""
import io
import json
import math
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC, OUT = sys.argv[1], sys.argv[2]
d = open(SRC, "rb").read()
jl = struct.unpack("<I", d[12:16])[0]
J = json.loads(d[20:20 + jl])
bl = struct.unpack("<I", d[20 + jl:24 + jl])[0]
BIN = bytearray(d[28 + jl:28 + jl + bl])

DRIVER_X = 0.38  # +X is the car's left; the driver sits on the left


# ------------------------------------------------------------------------------------------------- geometry helpers
def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_to(ax):
    """Rotation taking +Z to the unit vector ax"""
    z = np.array([0, 0, 1.0])
    ax = np.array(ax, float) / np.linalg.norm(ax)
    v = np.cross(z, ax)
    c = z @ ax
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


class Mesh:
    def __init__(self, textured=False):
        self.P, self.N, self.I, self.UV = [], [], [], []
        self.textured = textured

    def add(self, p, n, i, uv=None):
        o = sum(len(x) for x in self.P)
        self.P.append(np.asarray(p, float))
        self.N.append(np.asarray(n, float))
        self.I.append(np.asarray(i, np.uint32) + o)
        if self.textured:
            self.UV.append(np.asarray(uv if uv is not None else np.zeros((len(p), 2)), float))

    def arrays(self):
        P, N = np.vstack(self.P), np.vstack(self.N)
        I = np.concatenate(self.I).reshape(-1, 3).copy()
        # faces agree with their vertex normals
        fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
        vn = N[I[:, 0]] + N[I[:, 1]] + N[I[:, 2]]
        bad = (fn * vn).sum(1) < 0
        I[bad] = I[bad][:, [0, 2, 1]]
        UV = np.vstack(self.UV).astype("<f4") if self.textured else None
        return P.astype("<f4"), N.astype("<f4"), I.reshape(-1).astype("<u4"), UV

    def box(self, c, sz, R=np.eye(3)):
        c = np.array(c, float)
        h = np.array(sz, float) / 2
        for a in range(3):
            for sg in (1, -1):
                u, v, n = np.zeros(3), np.zeros(3), np.zeros(3)
                n[a] = sg
                u[(a + 1) % 3] = 1
                v[(a + 2) % 3] = 1
                hu, hv = h[(a + 1) % 3], h[(a + 2) % 3]
                q = [-u * hu - v * hv, u * hu - v * hv, u * hu + v * hv, -u * hu + v * hv]
                q = [x + n * h[a] for x in q]
                self.add([c + R @ x for x in q], [R @ n] * 4, [0, 1, 2, 0, 2, 3])

    def cyl(self, p0, p1, r, seg=18, caps=True):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        ax = p1 - p0
        R = rot_to(ax)
        L = np.linalg.norm(ax)
        ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        ring = np.c_[np.cos(ang) * r, np.sin(ang) * r, np.zeros(seg)]
        P, N, I = [], [], []
        for k in range(seg):
            P += [ring[k], ring[k] + [0, 0, L]]
            nn = np.array([math.cos(ang[k]), math.sin(ang[k]), 0])
            N += [nn, nn]
        for k in range(seg):
            a, b = 2 * k, 2 * ((k + 1) % seg)
            I += [a, b, a + 1, b, b + 1, a + 1]
        if caps:
            for z, nz in ((0, -1), (L, 1)):
                c0 = len(P)
                P.append([0, 0, z])
                N.append([0, 0, nz])
                for k in range(seg):
                    P.append(ring[k] + [0, 0, z])
                    N.append([0, 0, nz])
                for k in range(seg):
                    I += [c0, c0 + 1 + k, c0 + 1 + (k + 1) % seg]
        P, N = np.array(P), np.array(N)
        self.add((R @ P.T).T + p0, (R @ N.T).T, I)

    def torus(self, R_, r_, Rot, c, su=48, sv=12, squash=None):
        """Ring in the local XY plane; `squash(x, y)` may flatten the outline (returns the radius scale)"""
        P, N, I = [], [], []
        for i in range(su):
            u = 2 * np.pi * i / su
            cx = np.array([math.cos(u), math.sin(u), 0])
            sc = squash(cx[0], cx[1]) if squash else 1.0
            for j in range(sv):
                v = 2 * np.pi * j / sv
                P.append(cx * (R_ * sc + r_ * math.cos(v)) + np.array([0, 0, r_ * math.sin(v)]))
                N.append(cx * math.cos(v) + np.array([0, 0, math.sin(v)]))
        for i in range(su):
            for j in range(sv):
                a, b = i * sv + j, ((i + 1) % su) * sv + j
                c_, d_ = ((i + 1) % su) * sv + (j + 1) % sv, i * sv + (j + 1) % sv
                I += [a, b, c_, a, c_, d_]
        P, N = np.array(P), np.array(N)
        self.add((Rot @ P.T).T + c, (Rot @ N.T).T, I)

    def disc(self, c, n, r, seg=28, uv_axes=None):
        """Flat disc; with `uv_axes` = (right, up) unit vectors, planar UVs (the picture seen from the +n side flipped)"""
        n = np.array(n, float)
        R = rot_to(n)
        ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        P = np.array([[0, 0, 0]] + [[math.cos(a) * r, math.sin(a) * r, 0] for a in ang])
        I = []
        for k in range(seg):
            I += [0, 1 + k, 1 + (k + 1) % seg]
        W = (R @ P.T).T + np.array(c, float)
        uv = None
        if uv_axes is not None:
            right, up = np.array(uv_axes[0], float), np.array(uv_axes[1], float)
            rel = W - np.array(c, float)
            uv = np.c_[0.5 + (rel @ right) / (2 * r), 0.5 - (rel @ up) / (2 * r)]
        self.add(W, [n] * len(P), I, uv)

    def quad_z(self, x0, x1, y0, y1, z, flip=False):
        """A picture on a plane facing the driver (-Z): the viewer's left is +X, so x1 is the picture's left edge"""
        P = [[x1, y1, z], [x0, y1, z], [x0, y0, z], [x1, y0, z]]
        uv = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.add(P, [[0, 0, -1]] * 4, [0, 1, 2, 0, 2, 3], uv)


# ------------------------------------------------------------------------------------------------- pictures
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def font(size, mono=False):
    try:
        return ImageFont.truetype(MONO if mono else FONT, size)
    except OSError:
        return ImageFont.load_default()


def polar(cx, cy, r, deg):
    a = math.radians(deg)
    return cx + r * math.cos(a), cy - r * math.sin(a)


def dial(dr, cx, cy, r, vmax, step, minor, label, red_from=None, unit=""):
    dr.ellipse([cx - r - 10, cy - r - 10, cx + r + 10, cy + r + 10], fill=(26, 27, 30), outline=(150, 152, 158), width=5)
    dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(7, 8, 10))
    a0, a1 = 225, -45  # sweep, clockwise
    n_minor = int(vmax / minor)
    for k in range(n_minor + 1):
        v = k * minor
        deg = a0 + (a1 - a0) * v / vmax
        major = abs(v / step - round(v / step)) < 1e-6
        red = red_from is not None and v >= red_from
        col = (235, 40, 40) if red else (235, 236, 240)
        x0, y0 = polar(cx, cy, r - (30 if major else 18), deg)
        x1, y1 = polar(cx, cy, r - 6, deg)
        dr.line([x0, y0, x1, y1], fill=col, width=6 if major else 3)
        if major:
            tx, ty = polar(cx, cy, r - 56, deg)
            dr.text((tx, ty), str(int(v / (1000 if label == "rpm" else 1))), font=font(26), fill=col, anchor="mm")
    dr.text((cx, cy + r * 0.52), unit, font=font(22), fill=(150, 154, 162), anchor="mm")
    # needle at rest (zero), hub
    nx, ny = polar(cx, cy, r - 24, a0)
    dr.line([cx, cy, nx, ny], fill=(255, 90, 20), width=7)
    dr.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=(40, 42, 46), outline=(200, 202, 206), width=3)


def make_cluster():
    im = Image.new("RGB", (1024, 320), (8, 9, 11))
    dr = ImageDraw.Draw(im)
    dial(dr, 220, 165, 138, 320, 40, 20, "kmh", unit="km/h")
    dial(dr, 804, 165, 138, 9000, 1000, 500, "rpm", red_from=7500, unit="x1000 rpm")
    # trip screen between the dials
    dr.rounded_rectangle([392, 52, 632, 270], radius=18, fill=(10, 20, 30), outline=(70, 90, 110), width=3)
    dr.text((512, 100), "N", font=font(60), fill=(200, 255, 0), anchor="mm")
    dr.text((512, 160), "0", font=font(70, True), fill=(235, 240, 245), anchor="mm")
    dr.text((512, 205), "km/h", font=font(22), fill=(150, 160, 170), anchor="mm")
    dr.line([420, 232, 604, 232], fill=(60, 80, 100), width=2)
    dr.text((512, 252), "ODO 1 204 km   22 C", font=font(18), fill=(150, 160, 170), anchor="mm")
    return im


def make_screen():
    im = Image.new("RGB", (512, 320), (12, 16, 22))
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 512, 36], fill=(20, 26, 34))
    dr.text((14, 18), "NORYX NAV", font=font(18), fill=(200, 255, 0), anchor="lm")
    dr.text((498, 18), "18:56", font=font(18), fill=(220, 225, 232), anchor="rm")
    for k in range(7):  # a street grid
        dr.line([k * 90 - 20, 36, k * 90 + 60, 320], fill=(34, 44, 56), width=6)
        dr.line([0, 70 + k * 42, 512, 40 + k * 46], fill=(34, 44, 56), width=6)
    route = [(60, 300), (150, 230), (230, 220), (300, 150), (420, 120)]
    dr.line(route, fill=(0, 190, 255), width=10, joint="curve")
    dr.ellipse([404, 104, 436, 136], fill=(200, 255, 0), outline=(255, 255, 255), width=3)
    dr.polygon([(60, 318), (46, 292), (74, 292)], fill=(255, 255, 255))
    dr.rounded_rectangle([14, 48, 150, 92], radius=10, fill=(20, 26, 34))
    dr.text((82, 70), "RWANDA 2 km", font=font(16), fill=(235, 240, 245), anchor="mm")
    return im


def make_badge():
    im = Image.new("RGB", (256, 256), (14, 14, 16))
    dr = ImageDraw.Draw(im)
    dr.ellipse([8, 8, 248, 248], fill=(22, 22, 25), outline=(190, 192, 198), width=8)
    dr.ellipse([40, 40, 216, 216], outline=(200, 255, 0), width=8)
    dr.text((128, 132), "N", font=font(120), fill=(200, 255, 0), anchor="mm")
    return im


def png_bytes(im):
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return b.getvalue()


# ------------------------------------------------------------------------------------------------- glTF plumbing
def buffer_view(data, target=None):
    while len(BIN) % 4:
        BIN.append(0)
    v = {"buffer": 0, "byteOffset": len(BIN), "byteLength": len(data)}
    if target:
        v["target"] = target
    BIN.extend(data)
    J["bufferViews"].append(v)
    return len(J["bufferViews"]) - 1


def accessor(arr, typ, ctype, target, minmax=False):
    a = {"bufferView": buffer_view(arr.tobytes(), target), "componentType": ctype, "count": int(arr.shape[0]), "type": typ}
    if minmax:
        a["min"] = arr.min(0).astype(float).tolist()
        a["max"] = arr.max(0).astype(float).tolist()
    J["accessors"].append(a)
    return len(J["accessors"]) - 1


def add_texture(im):
    J.setdefault("images", []).append({"bufferView": buffer_view(png_bytes(im)), "mimeType": "image/png"})
    J.setdefault("textures", []).append({"sampler": 0, "source": len(J["images"]) - 1})
    return len(J["textures"]) - 1


MATS = {}


def mat(name, col, rough=0.7, metal=0.0, emis=None, tex=None):
    m = {"name": name, "doubleSided": True, "pbrMetallicRoughness": {"baseColorFactor": list(col) + [1.0], "metallicFactor": metal, "roughnessFactor": rough}}
    if tex is not None:
        m["pbrMetallicRoughness"]["baseColorFactor"] = [1, 1, 1, 1]
        m["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex}
        m["emissiveTexture"] = {"index": tex}
        m["emissiveFactor"] = list(emis or (0.9, 0.9, 0.9))
    elif emis:
        m["emissiveFactor"] = list(emis)
    J["materials"].append(m)
    MATS[name] = len(J["materials"]) - 1


def primitives(groups):
    out = []
    for k, m in groups.items():
        if not m.P:
            continue
        P, N, I, UV = m.arrays()
        attrs = {"POSITION": accessor(P, "VEC3", 5126, 34962, True), "NORMAL": accessor(N, "VEC3", 5126, 34962)}
        if UV is not None:
            attrs["TEXCOORD_0"] = accessor(UV, "VEC2", 5126, 34962)
        out.append({"attributes": attrs, "indices": accessor(I.reshape(-1, 1), "SCALAR", 5125, 34963), "material": MATS[k]})
    return out


# ------------------------------------------------------------------------------------------------- drop the old interior
body = next(n for n in J["nodes"] if n["name"] == "BODY")
body_mesh = J["meshes"][body["mesh"]]
body_mesh["primitives"] = [p for p in body_mesh["primitives"] if not J["materials"][p["material"]]["name"].startswith(("cabin_", "int_"))]
car = next(n for n in J["nodes"] if n["name"] == "CAR")
old_steer = [i for i, n in enumerate(J["nodes"]) if n["name"] == "STEER"]
car["children"] = [c for c in car["children"] if c not in old_steer]
for i in old_steer:  # keep node indices stable: the old node stays as an empty, unattached node
    J["nodes"][i] = {"name": "STEER_OLD"}

# ------------------------------------------------------------------------------------------------- materials
t_cluster, t_screen, t_badge = add_texture(make_cluster()), add_texture(make_screen()), add_texture(make_badge())
mat("int_dash_top", (0.105, 0.105, 0.115), 0.55)         # dash pad: soft black
mat("int_dash_low", (0.20, 0.20, 0.22), 0.75)            # lower dash and door cards: warm grey
mat("int_floor", (0.09, 0.09, 0.10), 0.95)               # carpet
mat("int_liner", (0.50, 0.49, 0.47), 0.95)               # headliner
mat("int_trim", (0.07, 0.07, 0.075), 0.6)                # pillar trims, console
mat("int_seat", (0.13, 0.13, 0.145), 0.6)                # leather
mat("int_red", (0.62, 0.04, 0.04), 0.55)                 # inserts and stitching
mat("int_metal", (0.62, 0.63, 0.66), 0.3, 0.9)           # vent rings, spokes, handles
mat("int_mirror", (0.70, 0.74, 0.78), 0.08, 0.95)
mat("int_wheel", (0.045, 0.045, 0.05), 0.5)              # wheel leather
mat("int_cluster", (1, 1, 1), 0.4, tex=t_cluster)
mat("int_screen", (1, 1, 1), 0.3, tex=t_screen)
mat("int_badge", (1, 1, 1), 0.4, tex=t_badge, emis=(0.35, 0.35, 0.35))
G = {k: Mesh(textured=k in ("int_cluster", "int_screen", "int_badge")) for k in MATS}

# ------------------------------------------------------------------------------------------------- the cabin (rig space)
# shell: floor, headliner, parcel shelf, door cards
G["int_floor"].box((0, 0.38, -0.10), (1.90, 0.04, 2.30))
G["int_liner"].box((0, 1.62, -0.35), (1.90, 0.03, 1.30))
G["int_trim"].box((0, 0.72, -1.10), (1.90, 0.68, 0.04))
G["int_trim"].box((0, 1.02, -1.00), (1.90, 0.06, 0.30))
for sx in (1, -1):
    G["int_dash_low"].box((sx * 0.955, 0.66, -0.15), (0.05, 0.46, 1.90))            # lower door card
    G["int_dash_top"].box((sx * 0.95, 0.98, -0.15), (0.06, 0.12, 1.90))             # sill and window cap
    G["int_red"].box((sx * 0.925, 0.77, -0.15), (0.02, 0.03, 1.60))                 # stitched stripe
    G["int_dash_top"].box((sx * 0.86, 0.80, 0.0), (0.16, 0.07, 0.62))               # armrest
    G["int_metal"].box((sx * 0.92, 0.90, 0.40), (0.03, 0.03, 0.16))                 # door pull
    G["int_trim"].cyl((sx * 0.925, 0.58, 0.55), (sx * 0.935, 0.58, 0.55), 0.09, 20)  # speaker
# A-pillar trims, inside the paint's pillars
for sx in (1, -1):
    a, b = np.array([sx * 0.93, 1.20, 1.10]), np.array([sx * 0.89, 1.66, 0.30])
    mid, ax = (a + b) / 2, b - a
    R = rot_to(ax / np.linalg.norm(ax))
    G["int_trim"].box(mid + [-sx * 0.025, 0, 0], (0.055, 0.05, np.linalg.norm(ax)), R)
# sun visors
for sx in (DRIVER_X, -DRIVER_X):
    G["int_liner"].box((sx, 1.585, 0.20), (0.42, 0.012, 0.20))

# dashboard: pad sloping to the glass, a lower grey fascia, the binnacle, vents, centre screen
G["int_dash_low"].box((0, 0.78, 0.94), (1.96, 0.16, 0.56))                   # lower fascia (to y 0.86)
G["int_dash_top"].box((0, 0.92, 0.94), (1.96, 0.12, 0.56))                   # upper body
G["int_dash_top"].box((0, 1.00, 0.92), (1.90, 0.05, 0.52), rotx(0.10))       # pad
G["int_red"].box((0, 0.86, 0.658), (1.80, 0.014, 0.012))                     # red seam between the two tones
G["int_dash_top"].box((DRIVER_X, 1.19, 0.83), (0.66, 0.03, 0.30))            # binnacle visor
for sx in (-1, 1):
    G["int_dash_top"].box((DRIVER_X + sx * 0.325, 1.095, 0.83), (0.03, 0.20, 0.30))  # binnacle sides
G["int_dash_top"].box((DRIVER_X, 1.005, 0.83), (0.66, 0.03, 0.30))
G["int_cluster"].quad_z(DRIVER_X - 0.30, DRIVER_X + 0.30, 1.025, 1.175, 0.70)
G["int_screen"].quad_z(-0.14, 0.14, 0.88, 1.04, 0.655)                       # centre screen
G["int_trim"].box((0, 0.96, 0.67), (0.30, 0.19, 0.03))                        # its bezel
G["int_trim"].box((0, 0.62, 0.25), (0.20, 0.10, 0.78), roty(0))
# round vents: a metal ring around a dark disc, facing the driver
for vx, vy in ((0.82, 0.94), (-0.82, 0.94), (0.28, 0.80), (-0.28, 0.80)):
    G["int_metal"].cyl((vx, vy, 0.655), (vx, vy, 0.66), 0.062, 24)
    G["int_trim"].cyl((vx, vy, 0.652), (vx, vy, 0.664), 0.050, 24)

# centre tunnel, shifter, cup recess, seats
G["int_trim"].box((0, 0.52, 0.10), (0.20, 0.28, 1.50))
G["int_dash_low"].box((0, 0.668, 0.0), (0.205, 0.012, 0.50))
G["int_metal"].cyl((0, 0.67, 0.22), (0, 0.80, 0.22), 0.014, 12)
G["int_trim"].box((0, 0.83, 0.22), (0.05, 0.05, 0.07))
for sx in (DRIVER_X, -DRIVER_X):
    tilt = rotx(-0.21)
    G["int_seat"].box((sx, 0.46, -0.10), (0.52, 0.12, 0.60))                  # cushion
    G["int_seat"].box((sx, 0.82, -0.40), (0.52, 0.62, 0.12), tilt)            # backrest
    G["int_red"].box((sx, 0.80, -0.335), (0.30, 0.46, 0.02), tilt)            # insert
    for ox in (-0.24, 0.24):
        G["int_seat"].box((sx + ox, 0.74, -0.34), (0.08, 0.50, 0.15), tilt)   # bolsters
    G["int_seat"].box((sx, 1.28, -0.46), (0.30, 0.22, 0.10), tilt)            # headrest
    G["int_trim"].box((sx, 0.40, -0.10), (0.14, 0.30, 0.50))
# interior mirror under the glass
Rm = roty(-0.18) @ rotx(-0.12)
G["int_trim"].box((0, 1.31, 0.80), (0.27, 0.078, 0.04), Rm)
G["int_mirror"].box(np.array([0, 1.31, 0.80]) + Rm @ np.array([0, 0, -0.0215]), (0.245, 0.056, 0.004), Rm)
G["int_trim"].cyl((0, 1.42, 0.84), (0, 1.34, 0.80), 0.011, 10)
G["int_trim"].box((0, 1.42, 0.855), (0.05, 0.03, 0.016))

# ------------------------------------------------------------------------------------------------- steering wheel + hands
SW = {k: Mesh(textured=(k == "int_badge")) for k in ("int_wheel", "int_metal", "int_red", "int_trim", "int_badge")}
ID = np.eye(3)
FLAT = 0.86  # the rim's flat top and bottom: radius scale where |y| would exceed it


def flat_rim(x, y):
    return 1.0 if abs(y) < FLAT else FLAT / abs(y)


SW["int_wheel"].torus(0.178, 0.021, ID, (0, 0, 0), su=64, sv=14, squash=flat_rim)
for ang in (180, 0):  # the two side spokes: broad metal bars
    a = math.radians(ang)
    SW["int_metal"].box((math.cos(a) * 0.09, -0.012, 0.0), (0.17, 0.034, 0.012))
SW["int_metal"].box((0, -0.10, 0.0), (0.05, 0.12, 0.012))                      # lower spoke
SW["int_trim"].cyl((0, 0, -0.026), (0, 0, 0.03), 0.058, 28)                    # hub
SW["int_badge"].disc((0, 0, -0.0275), (0, 0, -1), 0.05, 28, uv_axes=((-1, 0, 0), (0, 1, 0)))
SW["int_red"].box((0, 0.158, 0), (0.024, 0.030, 0.046))                         # 12 o'clock marker
SW["int_trim"].cyl((0, 0, 0.03), (0, 0, 0.34), 0.028, 14)                      # column
# ------------------------------------------------------------------------------------------------- write
body_mesh["primitives"] += primitives(G)
J["meshes"].append({"name": "STEER", "primitives": primitives(SW)})
a = math.radians(24)
J["nodes"].append({"name": "STEER", "mesh": len(J["meshes"]) - 1, "translation": [DRIVER_X, 0.93, 0.50], "rotation": [math.sin(a / 2), 0, 0, math.cos(a / 2)]})
car["children"].append(len(J["nodes"]) - 1)

# The game's driver arms (driver-arms.ts): hands on the rim at 9 and 3 o'clock, elbows inside the seat's reach
car["extras"]["driver"] = {
    "gripX": 0.178,
    "gripY": -0.01,
    "handScale": 1.4,
    "elbows": {"left": [DRIVER_X + 0.24, 0.75, 0.10], "right": [DRIVER_X - 0.24, 0.75, 0.10]},
    "glove": "#eab308",
    "pad": "#18181b",
    "sleeve": "#1d3a2f",
}
cams = car["extras"]["cameras"]
cams["cockpit"] = {"pos": [DRIVER_X, 1.22, -0.05], "pitchDeg": -6, "fov": 60}
cams["dash"] = {"pos": [DRIVER_X, 1.22, 0.14], "pitchDeg": -11, "fov": 58}
src = car["extras"]["source"]
if "interior rebuilt" not in src["changes"]:
    src["changes"] += "; interior rebuilt (textured cluster, screen and badge, two-tone dashboard, door cards, seats, steering wheel)"

J["buffers"][0]["byteLength"] = len(BIN)
js = json.dumps(J, separators=(",", ":")).encode()
while len(js) % 4:
    js += b" "
while len(BIN) % 4:
    BIN.append(0)
open(OUT, "wb").write(struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(BIN)) + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(BIN), 0x004E4942) + bytes(BIN))
print("ok", len(BIN) // 1024, "KB")
