#!/usr/bin/env python3
"""
Rebuilds the interior of the ported "Modern Cartoon Sports Car" (after scripts/cars/remove_cartoon_lid.py and, before
finish_cartoon_car.py, whichever order the GLB was last made in: the script drops the previous interior first, so it is safe
to run on its own output) so the cockpit and dash views read as a real car's:

  - real materials, tiled procedural textures drawn here: grained black leather (dash pad, sills, armrests), alcantara
    (lower dash, door cards, pillars), carbon twill (console, centre stack), brushed aluminium (spokes, vents, pedals),
    quilted perforated leather (seats), carpet; red piping and stitching
  - a dashboard of curved profiles (pad rolling down to the cowl, a seam in red piping, an alcantara fascia), an instrument
    binnacle with a curved visor over the cluster (`int_cluster`, the live dials of cartoon-dash.ts), a centre stack turned
    towards the driver with the map screen (`int_screen`) standing in front of its housing (so no two surfaces share a depth
    and nothing flickers), climate knobs, round louvred vents, a column shroud, pedals
  - door cards with a leather sill, alcantara and leather panels, carbon insert, armrest with window switches, pull, handle,
    speaker and scuff plate; bucket seats with bolsters, wings, a red insert and a headrest; belts; a tunnel with a
    selector, cupholder, drive-mode dial and start button
  - a leather steering wheel (own `STEER` node, column along local +Z, tilted 24 degrees, which the game turns about that
    axis) with a flat top and bottom, metal spokes, thumb buttons, paddles and a hub badge
  - the `driver` entry of the GLB's extras that switches on the game's own driver (src/lib/render/driver-arms.ts: hands,
    forearms, arms, torso and helmet)
  - the cockpit and dash cameras: the cockpit camera is the driver's eye, the dash camera sits just behind the wheel

At the end the GLB is re-packed without the materials, textures and bytes nothing uses any more.

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
    def __init__(self, textured=False, autouv=None):
        self.P, self.N, self.I, self.UV = [], [], [], []
        self.autouv = autouv  # metres per texture repeat: UVs projected on the axis a face looks along
        self.textured = textured or autouv is not None

    def add(self, p, n, i, uv=None):
        o = sum(len(x) for x in self.P)
        self.P.append(np.asarray(p, float))
        self.N.append(np.asarray(n, float))
        self.I.append(np.asarray(i, np.uint32) + o)
        if self.textured:
            if uv is None and self.autouv:
                P_, N_ = np.asarray(p, float), np.asarray(n, float)
                ax = np.abs(N_).argmax(1)
                uvs = np.where((ax == 0)[:, None], P_[:, [2, 1]], np.where((ax == 1)[:, None], P_[:, [0, 2]], P_[:, [0, 1]]))
                uv = uvs / self.autouv
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

    # ---- rounded box, extrusions and strips (all in rig space)
    def rbox(self, c, sz, r, R=np.eye(3)):
        """Box with rounded edges (radius r): each face is a grid whose outer ring is projected onto the corner arcs"""
        c = np.array(c, float)
        h = np.array(sz, float) / 2
        r = float(min(r, h.min() - 1e-4))
        inner = h - r

        def coords(hh, inn):
            e = [-hh, -(inn + 0.66 * r), -(inn + 0.33 * r), -inn, inn, inn + 0.33 * r, inn + 0.66 * r, hh]
            return np.array(sorted(set(np.round(e, 6))))

        for a in range(3):
            for sg in (1, -1):
                u, v = (a + 1) % 3, (a + 2) % 3
                Cu, Cv = coords(h[u], inner[u]), coords(h[v], inner[v])
                P, N = [], []
                for cu in Cu:
                    for cv in Cv:
                        q = np.zeros(3)
                        q[a], q[u], q[v] = sg * h[a], cu, cv
                        inn = np.clip(q, -inner, inner)
                        d = q - inn
                        ln = np.linalg.norm(d)
                        if ln > 1e-9:
                            n = d / ln
                        else:
                            n = np.zeros(3)
                            n[a] = sg
                        P.append(c + R @ (inn + n * r))
                        N.append(R @ n)
                I = []
                nv = len(Cv)
                for i in range(len(Cu) - 1):
                    for j in range(nv - 1):
                        k = i * nv + j
                        I += [k, k + 1, k + nv + 1, k, k + nv + 1, k + nv]
                self.add(P, N, I)

    def prism(self, outline, x0, x1, smooth_deg=50.0):
        """A convex (or star-shaped) outline given as (z, y) points, extruded from x0 to x1, with smooth sides"""
        pts = np.array(outline, float)
        n = len(pts)
        cen = pts.mean(0)
        en = []
        for i in range(n):
            e = pts[(i + 1) % n] - pts[i]
            nn = np.array([e[1], -e[0]])
            nn /= np.linalg.norm(nn)
            if nn @ ((pts[i] + pts[(i + 1) % n]) / 2 - cen) < 0:
                nn = -nn
            en.append(nn)
        cs = math.cos(math.radians(smooth_deg))

        def vnorm(i, side_edge):
            a, b = en[(i - 1) % n], en[i % n]
            if a @ b > cs:
                m = a + b
                return m / np.linalg.norm(m)
            return en[side_edge]

        def p3(x, zy):
            return [x, zy[1], zy[0]]

        for i in range(n):
            j = (i + 1) % n
            n0, n1 = vnorm(i, i), vnorm(j, i)
            P = [p3(x0, pts[i]), p3(x1, pts[i]), p3(x1, pts[j]), p3(x0, pts[j])]
            N = [[0, n0[1], n0[0]], [0, n0[1], n0[0]], [0, n1[1], n1[0]], [0, n1[1], n1[0]]]
            self.add(P, N, [0, 1, 2, 0, 2, 3])
        for x, nx in ((x0, -1), (x1, 1)):
            P = [p3(x, cen)] + [p3(x, p) for p in pts]
            self.add(P, [[nx, 0, 0]] * len(P), sum([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)], []))

    def shell(self, curve, th, x0, x1):
        """A thin curved slab: `curve` (z, y) points swept from x0 to x1, th thick, the curve's up side first"""
        pts = np.array(curve, float)
        n = len(pts)
        nr = []
        for i in range(n):
            a = pts[max(i - 1, 0)]
            b = pts[min(i + 1, n - 1)]
            e = b - a
            nn = np.array([-e[1], e[0]])
            nn /= np.linalg.norm(nn)
            nr.append(nn if nn[1] >= 0 else -nn)
        top = [pts[i] + nr[i] * th / 2 for i in range(n)]
        bot = [pts[i] - nr[i] * th / 2 for i in range(n)]

        def p3(x, zy):
            return [x, zy[1], zy[0]]

        for i in range(n - 1):
            for arr, sgn in ((top, 1), (bot, -1)):
                P = [p3(x0, arr[i]), p3(x1, arr[i]), p3(x1, arr[i + 1]), p3(x0, arr[i + 1])]
                N = [[0, sgn * nr[i][1], sgn * nr[i][0]]] * 2 + [[0, sgn * nr[i + 1][1], sgn * nr[i + 1][0]]] * 2
                self.add(P, N, [0, 1, 2, 0, 2, 3])
            for x, nx in ((x0, -1), (x1, 1)):
                P = [p3(x, top[i]), p3(x, top[i + 1]), p3(x, bot[i + 1]), p3(x, bot[i])]
                self.add(P, [[nx, 0, 0]] * 4, [0, 1, 2, 0, 2, 3])
        for i, sgn in ((0, -1), (n - 1, 1)):
            e = pts[1] - pts[0] if sgn < 0 else pts[-1] - pts[-2]
            nn = e / np.linalg.norm(e) * sgn
            P = [p3(x0, top[i]), p3(x1, top[i]), p3(x1, bot[i]), p3(x0, bot[i])]
            self.add(P, [[0, nn[1], nn[0]]] * 4, [0, 1, 2, 0, 2, 3])

    def stitches(self, p0, p1, count, size=(0.012, 0.0035, 0.0025)):
        """A row of small raised stitches from p0 to p1 (axis-aligned along the longest axis)"""
        p0, p1 = np.array(p0, float), np.array(p1, float)
        for t in np.linspace(0.0, 1.0, count):
            self.box(p0 + (p1 - p0) * t, size)

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


# ------------------------------------------------------------------------------------------------- tiling textures
def pnoise(n, cut_x, cut_y=None, seed=0):
    """Periodic Gaussian noise (so the tile repeats seamlessly), unit variance"""
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.normal(size=(n, n)))
    fy = np.fft.fftfreq(n)[:, None] * n
    fx = np.fft.fftfreq(n)[None, :] * n
    cut_y = cut_x if cut_y is None else cut_y
    F *= np.exp(-((fx / cut_x) ** 2 + (fy / cut_y) ** 2))
    a = np.real(np.fft.ifft2(F))
    return (a - a.mean()) / a.std()


def to_img(v, tint=(1.0, 1.0, 1.0)):
    v = np.clip(v, 0, 255)
    return Image.fromarray(np.stack([v * t for t in tint], -1).clip(0, 255).astype("uint8"), "RGB")


def tex_leather(base=40, seed=1):
    g1, g2 = pnoise(512, 70, seed=seed), pnoise(512, 22, seed=seed + 1)
    crease = (np.abs(g1) < 0.14) * 0.5
    return to_img(base * (1 + 0.08 * g1 + 0.05 * g2 - crease * 0.5), (1.0, 1.0, 1.04))


def tex_alcantara(base=56, seed=3):
    return to_img(base * (1 + 0.11 * pnoise(512, 150, seed=seed) + 0.05 * pnoise(512, 50, seed=seed + 1)), (1.0, 1.0, 1.03))


def tex_carpet(base=34, seed=5):
    return to_img(base * (1 + 0.20 * pnoise(512, 110, seed=seed) + 0.10 * pnoise(512, 36, seed=seed + 1)))


def tex_carbon(seed=7):
    n = 512
    y, x = np.mgrid[0:n, 0:n]
    cell = 16
    weave = ((x // cell + y // cell) % 2).astype(float)
    sheen = np.where(weave > 0, np.sin((x + y) * 0.55), np.sin((x - y) * 0.55))
    val = 38 + 26 * (0.5 + 0.5 * sheen) * (0.65 + 0.35 * weave) + 3 * pnoise(n, 120, seed=seed)
    return to_img(val, (1.0, 1.0, 1.05))


def tex_alu(seed=9):
    streaks = pnoise(512, 3.0, 220.0, seed=seed)
    return to_img(150 + 26 * streaks + 6 * pnoise(512, 90, seed=seed + 1))


def tex_seat(seed=11):
    """Quilted leather: diamond seams, pillowed centres and small perforations"""
    n, P = 512, 128
    y, x = np.mgrid[0:n, 0:n].astype(float)
    u, v = ((x + y) % P) / P, ((x - y) % P) / P
    d = np.maximum(np.abs(u - 0.5), np.abs(v - 0.5))  # 0 at a diamond's centre, 0.5 on its seam
    pillow = 1.0 + 0.35 * (1 - 2 * d)
    seam = np.clip((d - 0.455) / 0.045, 0, 1)
    base = 44 * pillow * (1 - 0.65 * seam)
    holes = (((x % 12) - 6) ** 2 + ((y % 12) - 6) ** 2 < 4.5) & (d < 0.30)
    base = np.where(holes, base * 0.35, base)
    base = base * (1 + 0.12 * pnoise(n, 70, seed=seed) + 0.06 * pnoise(n, 22, seed=seed + 1))
    return to_img(base, (1.0, 1.0, 1.04))

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


def add_texture(im, fmt="PNG"):
    if fmt == "JPEG":
        b = io.BytesIO()
        im.resize((256, 256), Image.LANCZOS).save(b, "JPEG", quality=82)  # tiling noise: 256 px is plenty at 0.2 m a repeat
        data, mime = b.getvalue(), "image/jpeg"
    else:
        data, mime = png_bytes(im), "image/png"
    J.setdefault("images", []).append({"bufferView": buffer_view(data), "mimeType": mime})
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
t_leather = add_texture(tex_leather(), "JPEG")
t_alc = add_texture(tex_alcantara(), "JPEG")
t_carpet = add_texture(tex_carpet(), "JPEG")
t_carbon = add_texture(tex_carbon(), "JPEG")
t_alu = add_texture(tex_alu(), "JPEG")
t_seat = add_texture(tex_seat(), "JPEG")
t_wheelc = add_texture(tex_leather(base=30, seed=21), "JPEG")


def tex_mat(name, tex, rough, metal=0.0):
    mat(name, (1, 1, 1), rough, metal)
    J["materials"][-1]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex}


tex_mat("int_dash_top", t_leather, 0.5)        # dash pad, door sills, armrests: grained black leather
tex_mat("int_dash_low", t_alc, 0.95)           # lower dash, door cards, pillars: dark alcantara
tex_mat("int_floor", t_carpet, 0.98)
tex_mat("int_trim", t_carbon, 0.28)            # console and trim: carbon twill
tex_mat("int_seat", t_seat, 0.55)              # quilted, perforated leather
tex_mat("int_metal", t_alu, 0.32, 0.55)        # brushed aluminium
tex_mat("int_wheel", t_wheelc, 0.5)            # wheel leather
mat("int_red", (0.42, 0.015, 0.015), 0.55)     # stitching and piping
mat("int_black", (0.012, 0.012, 0.014), 0.4)   # glossy black: vents' insides, buttons, speaker
mat("int_cluster", (1, 1, 1), 0.4, tex=t_cluster)
mat("int_screen", (1, 1, 1), 0.3, tex=t_screen)
mat("int_badge", (1, 1, 1), 0.4, tex=t_badge, emis=(0.35, 0.35, 0.35))
TEXTURED_UV = ("int_cluster", "int_screen", "int_badge")
G = {k: (Mesh(textured=True) if k in TEXTURED_UV else Mesh(autouv=0.20)) for k in MATS}
G["int_trim"].autouv = 0.16
G["int_black"].autouv = None
G["int_black"].textured = False
G["int_red"].autouv = None
G["int_red"].textured = False
for k in ("int_black", "int_red"):
    # plain colours: the Mesh only carries UVs for the textured materials
    pass

# ------------------------------------------------------------------------------------------------- the cabin (rig space)
FWD = np.array([0.0, 0.0, 1.0])
# floor, rear bulkhead and parcel shelf
G["int_floor"].rbox((0, 0.38, -0.10), (1.88, 0.05, 2.30), 0.02)
G["int_dash_low"].rbox((0, 0.72, -1.10), (1.88, 0.68, 0.05), 0.02)
G["int_dash_top"].rbox((0, 1.02, -1.00), (1.88, 0.07, 0.34), 0.03)

# door cards, each side: leather sill, alcantara upper, leather lower, carbon insert, armrest with switches, pull, speaker
for sx in (1, -1):
    G["int_dash_top"].rbox((sx * 0.925, 1.00, -0.15), (0.08, 0.075, 1.95), 0.03)              # sill / window cap
    G["int_dash_low"].rbox((sx * 0.945, 0.84, -0.15), (0.05, 0.27, 1.90), 0.02)               # upper card
    G["int_dash_top"].rbox((sx * 0.945, 0.56, -0.15), (0.05, 0.22, 1.90), 0.02)               # lower card
    G["int_trim"].rbox((sx * 0.915, 0.74, 0.05), (0.025, 0.11, 0.95), 0.01)                   # carbon insert
    G["int_red"].box((sx * 0.9185, 0.665, -0.15), (0.004, 0.006, 1.75))                       # piping under the card
    G["int_dash_top"].rbox((sx * 0.85, 0.86, 0.02), (0.18, 0.08, 0.64), 0.035)                # armrest
    G["int_red"].stitches((sx * 0.85, 0.9, -0.28), (sx * 0.85, 0.9, 0.32), 22, (0.003, 0.0025, 0.012))
    for k, zz in enumerate((0.15, 0.20, 0.25)):                                              # window switches
        G["int_metal"].rbox((sx * 0.9, 0.9, zz), (0.045, 0.012, 0.03), 0.004)
    G["int_metal"].rbox((sx * 0.905, 0.945, 0.46), (0.032, 0.03, 0.20), 0.012)                # door pull
    G["int_black"].rbox((sx * 0.905, 0.93, 0.46), (0.02, 0.02, 0.16), 0.008)
    G["int_metal"].rbox((sx * 0.9, 0.975, -0.34), (0.04, 0.025, 0.20), 0.01)                  # door handle
    G["int_black"].cyl((sx * 0.92, 0.52, 0.56), (sx * 0.935, 0.52, 0.56), 0.085, 28)          # speaker
    G["int_metal"].cyl((sx * 0.915, 0.52, 0.56), (sx * 0.92, 0.52, 0.56), 0.095, 28)
    G["int_metal"].rbox((sx * 0.93, 0.35, -0.1), (0.05, 0.03, 1.0), 0.01)                     # scuff plate on the sill
# A-pillar trims, inside the paint's pillars
for sx in (1, -1):
    a, b = np.array([sx * 0.93, 1.20, 1.10]), np.array([sx * 0.89, 1.66, 0.30])
    mid, ax = (a + b) / 2, b - a
    Rp = rot_to(ax / np.linalg.norm(ax))
    G["int_dash_low"].rbox(mid + [-sx * 0.025, 0, 0], (0.055, 0.05, np.linalg.norm(ax)), 0.015, Rp)

# ---- dashboard: leather pad curving down to the cowl, alcantara fascia under a red seam
G["int_dash_top"].prism([(0.655, 0.86), (0.655, 0.97), (0.70, 1.012), (0.78, 1.036), (0.95, 1.042), (1.10, 1.03), (1.22, 1.0), (1.22, 0.86)], -0.97, 0.97)
G["int_dash_low"].prism([(0.70, 0.60), (1.22, 0.55), (1.22, 0.86), (0.672, 0.86), (0.672, 0.70)], -0.97, 0.97)
G["int_red"].box((0, 0.862, 0.66), (1.90, 0.008, 0.014))                                     # piping on the seam
G["int_red"].stitches((-0.90, 0.925, 0.6535), (0.90, 0.925, 0.6535), 100, (0.011, 0.0035, 0.0025))
G["int_red"].stitches((-0.90, 0.99, 0.69), (0.90, 1.002, 0.76), 100, (0.011, 0.0025, 0.0035))
# the instrument binnacle: a curved visor over the cluster, side walls, a sill and a back plate
G["int_dash_top"].shell([(0.70, 1.19), (0.72, 1.21), (0.78, 1.226), (0.88, 1.233), (1.0, 1.222)], 0.03, DRIVER_X - 0.335, DRIVER_X + 0.335)
for sx in (-1, 1):
    G["int_dash_top"].rbox((DRIVER_X + sx * 0.335, 1.105, 0.86), (0.028, 0.22, 0.32), 0.012)
G["int_dash_top"].rbox((DRIVER_X, 1.006, 0.86), (0.70, 0.03, 0.32), 0.012)
G["int_black"].rbox((DRIVER_X, 1.105, 0.985), (0.66, 0.22, 0.03), 0.01)
G["int_cluster"].quad_z(DRIVER_X - 0.30, DRIVER_X + 0.30, 1.00, 1.15, 0.74)
# the centre stack: a rounded housing turned a little towards the driver, the screen on its face, knobs below it
yaw = roty(math.radians(-9))  # +X is the car's left, where the driver is: turn the face towards him
HC = np.array([0.0, 0.935, 0.675])
G["int_trim"].rbox(HC, (0.42, 0.34, 0.11), 0.035, yaw)
sw, sh, sz = 0.145, 0.095, HC[2] - 0.055 - 0.003
corners = [np.array([x, HC[1] + y, 0.0]) for x, y in ((sw, sh + 0.02), (-sw, sh + 0.02), (-sw, -sh + 0.02), (sw, -sh + 0.02))]
Psc = [HC + yaw @ (np.array([c[0], c[1] - HC[1], sz - HC[2]])) for c in corners]
G["int_screen"].add(Psc, [yaw @ np.array([0, 0, -1.0])] * 4, [0, 1, 2, 0, 2, 3], [[0, 0], [1, 0], [1, 1], [0, 1]])
for kx in (-0.12, 0.0, 0.12):                                                               # climate knobs
    G["int_metal"].cyl(HC + yaw @ np.array([kx, -0.14, -0.05]), HC + yaw @ np.array([kx, -0.14, -0.082]), 0.021, 24)
for bx in (-0.06, 0.06):
    G["int_black"].rbox(HC + yaw @ np.array([bx, -0.14, -0.05]), (0.04, 0.012, 0.012), 0.004, yaw)
G["int_red"].cyl(HC + yaw @ np.array([0.17, -0.12, -0.04]), HC + yaw @ np.array([0.17, -0.12, -0.062]), 0.012, 16)   # hazard
# round vents: metal ring, black bowl, three louvres, a thumb wheel
for vx, vy in ((0.84, 0.955), (-0.84, 0.955), (0.30, 0.955), (-0.30, 0.955)):
    G["int_metal"].cyl((vx, vy, 0.652), (vx, vy, 0.668), 0.066, 28)
    G["int_black"].cyl((vx, vy, 0.64), (vx, vy, 0.675), 0.053, 28)
    for lx in (-0.025, 0.0, 0.025):
        G["int_dash_top"].box((vx + lx, vy, 0.646), (0.006, 0.09, 0.008))
    G["int_metal"].cyl((vx, vy - 0.07, 0.65), (vx, vy - 0.07, 0.66), 0.012, 12)
# steering column shroud and the knee bolster
G["int_trim"].rbox((DRIVER_X, 0.855, 0.675), (0.17, 0.14, 0.14), 0.04)
G["int_dash_low"].rbox((DRIVER_X, 0.66, 0.66), (0.50, 0.20, 0.08), 0.03)
# pedals: aluminium brake and throttle, a dead pedal
for px, ph, pw in ((0.30, 0.13, 0.10), (0.46, 0.16, 0.05)):
    G["int_metal"].rbox((px, 0.40, 0.60), (pw, ph, 0.014), 0.006, rotx(-0.45))
G["int_metal"].rbox((0.72, 0.42, 0.55), (0.18, 0.12, 0.012), 0.005, rotx(-0.5))

# ---- centre tunnel: alcantara body, carbon deck, selector, cupholder, drive-mode dial, start button
G["int_dash_low"].rbox((0, 0.55, 0.10), (0.22, 0.34, 1.50), 0.05)
G["int_trim"].rbox((0, 0.718, 0.20), (0.20, 0.03, 0.98), 0.012)
G["int_black"].rbox((0, 0.738, 0.34), (0.095, 0.03, 0.15), 0.012)                           # boot
G["int_metal"].cyl((0, 0.74, 0.34), (0, 0.80, 0.34), 0.013, 14)
G["int_dash_top"].rbox((0, 0.835, 0.34), (0.06, 0.07, 0.07), 0.025)                          # selector knob
G["int_black"].cyl((0, 0.725, -0.16), (0, 0.745, -0.16), 0.04, 24)                           # cupholder
G["int_metal"].cyl((0, 0.717, -0.16), (0, 0.72, -0.16), 0.046, 24)
G["int_metal"].cyl((0.0, 0.735, 0.58), (0.0, 0.765, 0.58), 0.026, 24)                        # drive-mode dial
G["int_red"].cyl((0.075, 0.735, 0.58), (0.075, 0.75, 0.58), 0.015, 18)                       # start button
G["int_dash_top"].rbox((0, 0.80, -0.55), (0.20, 0.06, 0.30), 0.03)                           # armrest lid

# ---- seats: bucket cushions and backs with raised bolsters, a red centre insert, an integrated headrest, belts
tilt = rotx(-0.21)
for sx in (DRIVER_X, -DRIVER_X):
    G["int_seat"].rbox((sx, 0.46, -0.10), (0.54, 0.13, 0.64), 0.05)                          # cushion
    for ox in (-0.26, 0.26):
        G["int_seat"].rbox((sx + ox, 0.505, -0.11), (0.075, 0.18, 0.60), 0.03)              # cushion bolsters
    G["int_seat"].rbox((sx, 0.86, -0.40), (0.50, 0.62, 0.13), 0.05, tilt)                    # back
    for ox in (-0.255, 0.255):
        G["int_seat"].rbox((sx + ox, 0.88, -0.355), (0.075, 0.58, 0.18), 0.03, tilt)        # wings
    G["int_red"].rbox((sx, 0.84, -0.336), (0.20, 0.42, 0.012), 0.004, tilt)                  # insert
    G["int_seat"].rbox((sx, 1.30, -0.47), (0.31, 0.27, 0.115), 0.05, tilt)                   # headrest
    G["int_dash_top"].rbox((sx, 0.40, -0.10), (0.16, 0.30, 0.52), 0.03)                      # rail cover
# belts: from the shoulder, across the chest, to the buckle by the tunnel
for sx, anchor in ((DRIVER_X, (0.80, 1.34, -0.50)), (-DRIVER_X, (-0.80, 1.34, -0.50))):
    a = np.array(anchor, float)
    b = np.array([sx * 0.4, 0.66, -0.14])
    mid, ax = (a + b) / 2, b - a
    Rb = rot_to(ax / np.linalg.norm(ax))
    G["int_dash_low"].box(mid + [0, 0, 0.045], (0.05, 0.004, np.linalg.norm(ax)), Rb @ rotx(0.0))
    G["int_metal"].rbox(b + [0, 0.01, 0.03], (0.05, 0.02, 0.07), 0.006)

# ------------------------------------------------------------------------------------------------- steering wheel
SW = {k: (Mesh(textured=True) if k == "int_badge" else Mesh(autouv=0.25)) for k in ("int_wheel", "int_metal", "int_red", "int_trim", "int_badge", "int_black")}
SW["int_red"].autouv = None
SW["int_red"].textured = False
SW["int_black"].autouv = None
SW["int_black"].textured = False
ID = np.eye(3)
FLAT = 0.86


def flat_rim(x, y):
    return 1.0 if abs(y) < FLAT else FLAT / abs(y)


SW["int_wheel"].torus(0.178, 0.021, ID, (0, 0, 0), su=72, sv=16, squash=flat_rim)
for sx in (1, -1):
    SW["int_metal"].rbox((sx * 0.095, -0.012, 0.0), (0.16, 0.040, 0.014), 0.006)            # side spokes
    SW["int_black"].cyl((sx * 0.13, 0.012, -0.0075), (sx * 0.13, 0.012, -0.0105), 0.0105, 16)  # thumb buttons
    SW["int_black"].cyl((sx * 0.13, -0.014, -0.0075), (sx * 0.13, -0.014, -0.0105), 0.0105, 16)
SW["int_metal"].rbox((0, -0.105, 0.0), (0.055, 0.12, 0.014), 0.006)                          # lower spoke
SW["int_red"].cyl((-0.14, 0.0, -0.0075), (-0.14, 0.0, -0.011), 0.0105, 16)                  # start button on the spoke
SW["int_trim"].cyl((0, 0, -0.026), (0, 0, 0.03), 0.058, 32)                                   # hub
SW["int_badge"].disc((0, 0, -0.0275), (0, 0, -1), 0.05, 32, uv_axes=((-1, 0, 0), (0, 1, 0)))
SW["int_red"].box((0, 0.160, 0), (0.016, 0.030, 0.046))                                      # 12 o'clock marker
SW["int_red"].stitches((-0.07, 0.1535, 0.0), (0.07, 0.1535, 0.0), 12, (0.007, 0.002, 0.044))
for sx in (1, -1):                                                                           # paddle shifters behind the rim
    SW["int_metal"].rbox((sx * 0.155, 0.02, -0.055), (0.012, 0.09, 0.032), 0.005, rotx(0.0))
SW["int_trim"].cyl((0, 0, 0.03), (0, 0, 0.34), 0.028, 14)                                      # column

# ------------------------------------------------------------------------------------------------- write
body_mesh["primitives"] += primitives(G)
J["meshes"].append({"name": "STEER", "primitives": primitives(SW)})
a = math.radians(24)
J["nodes"].append({"name": "STEER", "mesh": len(J["meshes"]) - 1, "translation": [DRIVER_X, 0.93, 0.50], "rotation": [math.sin(a / 2), 0, 0, math.cos(a / 2)]})
car["children"].append(len(J["nodes"]) - 1)

# The game's driver arms (driver-arms.ts): hands on the rim at 9 and 3 o'clock, elbows inside the seat's reach
car["extras"]["driver"] = {
    "gripX": 0.178,
    "gripY": 0.0,
    "rimTube": 0.021,
    "handScale": 1.0,
    "elbows": {"left": [DRIVER_X + 0.20, 0.79, 0.16], "right": [DRIVER_X - 0.20, 0.79, 0.16]},
    "skin": "#d4a37e",
    "sleeve": "#1f2937",
    "cuff": "#e5e7eb",
    # torso against the seat's backrest, helmet just behind the eye point (the driver's-eye view leaves them out)
    "body": {"torso": [DRIVER_X, 0.84, -0.22], "head": [DRIVER_X, 1.21, -0.17], "suit": "#1f2937", "helmet": "#dc2626"},
}
cams = car["extras"]["cameras"]
cams["cockpit"] = {"pos": [DRIVER_X, 1.22, -0.05], "pitchDeg": -6, "fov": 60}
cams["dash"] = {"pos": [DRIVER_X, 1.22, 0.14], "pitchDeg": -11, "fov": 58}
src = car["extras"]["source"]
if "interior rebuilt" not in src["changes"]:
    src["changes"] += "; interior rebuilt (textured cluster, screen and badge, two-tone dashboard, door cards, seats, steering wheel)"

# ------------------------------------------------------------------------------------------------- drop what nothing uses
def compact():
    """Re-packs the GLB without the previous interiors' leftovers: unused materials, textures, images, accessors and bytes"""
    global BIN
    used_meshes = sorted({n["mesh"] for n in J["nodes"] if "mesh" in n})
    mesh_map = {o: n for n, o in enumerate(used_meshes)}
    J["meshes"] = [J["meshes"][o] for o in used_meshes]
    for n in J["nodes"]:
        if "mesh" in n:
            n["mesh"] = mesh_map[n["mesh"]]
    prims = [p for m in J["meshes"] for p in m["primitives"]]
    mats = sorted({p["material"] for p in prims if "material" in p})
    mmap = {o: n for n, o in enumerate(mats)}
    new_mats = [J["materials"][o] for o in mats]
    TEXKEYS = ("baseColorTexture", "metallicRoughnessTexture")

    def slots(m):
        out = [m["pbrMetallicRoughness"].get(k) for k in TEXKEYS] + [m.get(k) for k in ("normalTexture", "emissiveTexture", "occlusionTexture")]
        return [t for t in out if t]

    texs = sorted({t["index"] for m in new_mats for t in slots(m)})
    tmap = {o: n for n, o in enumerate(texs)}
    new_texs = [dict(J["textures"][o]) for o in texs]
    imgs = sorted({t["source"] for t in new_texs})
    imap = {o: n for n, o in enumerate(imgs)}
    new_imgs = [dict(J["images"][o]) for o in imgs]
    for t in new_texs:
        t["source"] = imap[t["source"]]
    for m in new_mats:
        for t in slots(m):
            t["index"] = tmap[t["index"]]
    for p in prims:
        if "material" in p:
            p["material"] = mmap[p["material"]]
    accs = sorted({a for p in prims for a in list(p["attributes"].values()) + [p["indices"]]})
    amap = {o: n for n, o in enumerate(accs)}
    new_accs = [dict(J["accessors"][o]) for o in accs]
    bvs = sorted({a["bufferView"] for a in new_accs} | {im["bufferView"] for im in new_imgs})
    bmap = {o: n for n, o in enumerate(bvs)}
    new_bin = bytearray()
    new_bvs = []
    for o in bvs:
        bv = dict(J["bufferViews"][o])
        while len(new_bin) % 4:
            new_bin.append(0)
        off = bv.get("byteOffset", 0)
        data = bytes(BIN[off:off + bv["byteLength"]])
        bv["byteOffset"] = len(new_bin)
        new_bin.extend(data)
        new_bvs.append(bv)
    for a in new_accs:
        a["bufferView"] = bmap[a["bufferView"]]
    for im in new_imgs:
        im["bufferView"] = bmap[im["bufferView"]]
    for p in prims:
        p["attributes"] = {k: amap[v] for k, v in p["attributes"].items()}
        p["indices"] = amap[p["indices"]]
    J["materials"], J["textures"], J["images"] = new_mats, new_texs, new_imgs
    J["accessors"], J["bufferViews"] = new_accs, new_bvs
    BIN = new_bin
compact()
J["buffers"][0]["byteLength"] = len(BIN)
js = json.dumps(J, separators=(",", ":")).encode()
while len(js) % 4:
    js += b" "
while len(BIN) % 4:
    BIN.append(0)
open(OUT, "wb").write(struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(BIN)) + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(BIN), 0x004E4942) + bytes(BIN))
print("ok", len(BIN) // 1024, "KB")
