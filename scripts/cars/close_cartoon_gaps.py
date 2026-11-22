#!/usr/bin/env python3
"""
Closes the gaps in the cartoon car's shell that show the road (or the sky) through the cabin (run after
line_cartoon_doors.py).

The body's paint was modelled as separate panels that do not quite meet at the A-pillar bases and beside the dashboard,
and the engine bay and wheel wells behind them are open, so from the driver's seat a few slivers looked straight through
the car. This looks for them the way the camera would: it shoots rays from the cockpit and dash camera eyes over the whole
field the player can look at, keeps those that leave the car without hitting any solid part and without crossing the
glass, finds where each squeezes through the shell (its closest approach to the solid geometry), and puts a small dark
alcantara patch there across the ray. It then casts again and adds patches for what is left, up to a few rounds.

  python3 scripts/cars/close_cartoon_gaps.py IN.glb OUT.glb          (needs numpy and scipy)
"""
import json
import struct
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import ConvexHull, cKDTree

SRC, OUT = sys.argv[1], sys.argv[2]
d = open(SRC, "rb").read()
jl = struct.unpack("<I", d[12:16])[0]
J = json.loads(d[20:20 + jl])
bl = struct.unpack("<I", d[20 + jl:24 + jl])[0]
BIN = bytearray(d[28 + jl:28 + jl + bl])

EYES = [(0.38, 1.38, -0.05), (0.38, 1.36, 0.14), (-0.38, 1.38, -0.05)]
YAW = (-80, 80)      # degrees either side of straight ahead, to look at
PITCH = (-55, 25)
STEP = 2.0
ROUNDS = 3
PATCH_MARGIN = 0.03  # m beyond the cluster of squeeze points
MAX_Y = 1.30         # only gaps below the window sills / pillar bases: higher up, seeing out past the pillars is what windows are for
CLUSTER_DIST = 0.08
MAX_HALF = 0.10      # a patch never grows past this, so one never hides the view out of a window
NAME_MAT = "int_dash_low"


def accessor(i):
    a = J["accessors"][i]
    bv = J["bufferViews"][a["bufferView"]]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
    dt = {5126: "<f4", 5123: "<u2", 5125: "<u4"}[a["componentType"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    return np.frombuffer(BIN, dtype=dt, count=a["count"] * n, offset=off).reshape(a["count"], n)


def add_accessor(arr, typ, ctype, target, minmax=False):
    while len(BIN) % 4:
        BIN.append(0)
    J["bufferViews"].append({"buffer": 0, "byteOffset": len(BIN), "byteLength": arr.nbytes, "target": target})
    BIN.extend(arr.tobytes())
    a = {"bufferView": len(J["bufferViews"]) - 1, "componentType": ctype, "count": int(arr.shape[0]), "type": typ}
    if minmax:
        a["min"] = arr.min(0).astype(float).tolist()
        a["max"] = arr.max(0).astype(float).tolist()
    J["accessors"].append(a)
    return len(J["accessors"]) - 1


body = next(n for n in J["nodes"] if n["name"] == "BODY")
mesh = J["meshes"][body["mesh"]]
mat_of = lambda p: J["materials"][p["material"]]["name"]
mesh["primitives"] = [p for p in mesh["primitives"] if "gap_patch" not in p.get("extras", {})]


def prim_tris(p):
    pos = accessor(p["attributes"]["POSITION"]).astype(np.float32)
    return pos[accessor(p["indices"]).reshape(-1, 3).astype(np.int64)]


def solid_and_glass():
    solid = [prim_tris(p) for p in mesh["primitives"] if mat_of(p) != "glass"]
    glass = [prim_tris(p) for p in mesh["primitives"] if mat_of(p) == "glass"]
    return np.concatenate(solid), np.concatenate(glass)


def first_hit(o, D, tris, chunk=40):
    v0 = tris[:, 0]
    e1, e2 = tris[:, 1] - v0, tris[:, 2] - v0
    out = np.full(len(D), np.inf, np.float32)
    sv = (o[None] - v0)
    q = np.cross(sv, e1)
    for s in range(0, len(D), chunk):
        dd = D[s:s + chunk].astype(np.float32)
        h = np.cross(dd[:, None, :], e2[None])
        a = np.einsum("nj,rnj->rn", e1, h)
        ok = np.abs(a) > 1e-9
        f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
        u = f * np.einsum("nj,rnj->rn", sv, h)
        v = f * np.einsum("rj,nj->rn", dd, q)
        t = f * np.einsum("nj,nj->n", e2, q)[None]
        m = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4)
        out[s:s + chunk] = np.where(m, t, np.inf).min(1)
    return out


def view_directions():
    yaw = np.radians(np.arange(YAW[0], YAW[1] + 1e-6, STEP))
    pit = np.radians(np.arange(PITCH[0], PITCH[1] + 1e-6, STEP))
    yy, pp = np.meshgrid(yaw, pit)
    # model axes: +Z ahead, +X to the left, +Y up
    return np.stack([-np.sin(yy) * np.cos(pp), np.sin(pp), np.cos(yy) * np.cos(pp)], -1).reshape(-1, 3)


DIRS = view_directions()


def find_leaks():
    solid, glass = solid_and_glass()
    verts = np.unique(np.round(solid.reshape(-1, 3), 3), axis=0)
    tree = cKDTree(verts)
    squeeze, rays = [], []
    for eye in EYES:
        o = np.array(eye, np.float32)
        hit_s = first_hit(o, DIRS, solid)
        leak = np.isinf(hit_s)
        if not leak.any():
            continue
        hit_g = first_hit(o, DIRS[leak], glass)
        leak_dirs = DIRS[leak][np.isinf(hit_g)]  # not through the glass: a real gap
        if not len(leak_dirs):
            continue
        ts = np.linspace(0.2, 3.0, 140)
        pts = o[None, None] + leak_dirs[:, None, :] * ts[None, :, None]
        dist, _ = tree.query(pts.reshape(-1, 3))
        k = dist.reshape(len(leak_dirs), -1).argmin(1)
        sq = pts[np.arange(len(leak_dirs)), k]
        low = sq[:, 1] < MAX_Y
        squeeze.append(sq[low])
        rays.append(leak_dirs[low])
    if not squeeze:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(squeeze), np.concatenate(rays)


patches = []  # (centre, ray, half-size)
for rnd in range(ROUNDS):
    S, R = find_leaks()
    print("round", rnd, "leaking rays:", len(S))
    if not len(S):
        break
    cl = fcluster(linkage(S, "single"), CLUSTER_DIST, "distance")
    for c in set(cl):
        m = cl == c
        P = S[m]
        ray = R[m].mean(0)
        ray /= np.linalg.norm(ray)
        ctr = P.mean(0)
        a = np.cross(ray, [0, 1, 0])
        if np.linalg.norm(a) < 1e-3:
            a = np.cross(ray, [1, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(ray, a)
        # the patch follows the outline of the gap (the hull of where the rays squeeze through), a little enlarged
        xy = np.c_[(P - ctr) @ a, (P - ctr) @ b]
        try:
            hull = xy[ConvexHull(xy).vertices]
        except Exception:  # a line or a point of rays: a small square
            hull = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.02
        hc = hull.mean(0)
        hull = hc + (hull - hc) * 1.0
        grow = hull - hc
        grow = grow / np.maximum(np.linalg.norm(grow, axis=1, keepdims=True), 1e-9) * PATCH_MARGIN
        hull = np.clip(hull + grow, -MAX_HALF, MAX_HALF)
        patches.append((ctr, a, b, hull))
        # the patch goes into the geometry now, so the next round sees it
        corners = np.array([ctr + h[0] * a + h[1] * b for h in hull], np.float32)
        n = np.repeat(ray[None], len(corners), axis=0).astype("<f4")
        uv = np.c_[corners[:, 2], corners[:, 1]].astype("<f4") / 0.20
        idx = np.array([[0, k, k + 1] for k in range(1, len(corners) - 1)], "<u4").reshape(-1, 1)
        mesh["primitives"].append({
            "attributes": {
                "POSITION": add_accessor(corners, "VEC3", 5126, 34962, True),
                "NORMAL": add_accessor(n, "VEC3", 5126, 34962),
                "TEXCOORD_0": add_accessor(uv, "VEC2", 5126, 34962),
            },
            "indices": add_accessor(idx, "SCALAR", 5125, 34963),
            "material": next(i for i, mm in enumerate(J["materials"]) if mm["name"] == NAME_MAT),
            "extras": {"gap_patch": True},
        })
    print("  patches so far:", len(patches))
S, R = find_leaks()
print("final leaking rays:", len(S))

car = next(nd for nd in J["nodes"] if nd["name"] == "CAR")
ch = car["extras"]["source"]["changes"]
if "gaps closed" not in ch:
    car["extras"]["source"]["changes"] = ch + "; gaps in the shell closed (gaps closed: nothing shows the road through the cabin)"
J["buffers"][0]["byteLength"] = len(BIN)
js = json.dumps(J, separators=(",", ":")).encode()
while len(js) % 4:
    js += b" "
while len(BIN) % 4:
    BIN.append(0)
open(OUT, "wb").write(
    struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(BIN)) + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(BIN), 0x004E4942) + bytes(BIN)
)
print("ok", len(patches), "patches ->", OUT)
