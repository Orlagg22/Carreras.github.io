#!/usr/bin/env python3
"""
Replaces the door lining (line_cartoon_doors.py) and the floating gap patches (close_cartoon_gaps.py) of the cartoon car
with one clean, opaque side wall per side of the cabin.

Why: the old lining was a copy of the paint triangles (ragged, with slivers between them, so the white paint back and the
sky showed through as grey/black shards) and the "gap patches" were loose flat squares that hung in the side windows.

How: for each side a regular (z, y) grid is shot with rays from the cabin outward; the first hit on the paint gives the
inner skin of the body at that point. Where the paint has a hole (A-pillar base, wheel arch, fender seams) there is no hit
and the wall is filled in smoothly from its neighbours, so nothing is left open. The wall sits a hair inside the skin (the
paint still hides it from outside), is dark alcantara like the door cards (`int_dash_low`) and stops under the side glass.

  python3 scripts/cars/fill_cartoon_sides.py IN.glb OUT.glb          (needs numpy and scipy)
"""
import json
import struct
import sys

import numpy as np
from scipy import ndimage

SRC, OUT = sys.argv[1], sys.argv[2]
d = open(SRC, "rb").read()
jl = struct.unpack("<I", d[12:16])[0]
J = json.loads(d[20:20 + jl])
bl = struct.unpack("<I", d[20 + jl:24 + jl])[0]
BIN = bytearray(d[28 + jl:28 + jl + bl])

OFFSET = 0.014            # m, wall sits this far inside the paint's inner skin
X_START = 0.62            # rays start this far from the centre line, going outward
X_MIN = 0.80              # paint hits nearer the centre than this are not the side skin
Z0, Z1 = -1.06, 1.75      # wall extent along the car (door ... A-pillar base ... dash side)
Y0 = 0.88                 # lower edge (below it the door cards already close the cabin)
Y_TOP_FRONT = 1.27        # upper edge where there is no side glass (A-pillar base)
DZ, DY = 0.025, 0.02
GLASS_GAP = 0.012
NAME = "int_dash_low"


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
# earlier lining / patches / walls are dropped first, so the script can be run on any version of the model
mesh["primitives"] = [p for p in mesh["primitives"] if not (set(p.get("extras", {})) & {"lining", "gap_patch", "side_wall"})]
prims = {mat_of(p): p for p in mesh["primitives"]}


def tris(p):
    pos = accessor(p["attributes"]["POSITION"]).astype(np.float64)
    return pos[accessor(p["indices"]).reshape(-1, 3).astype(np.int64)]


paint = tris(prims["paint"])
glass = tris(prims["glass"]).reshape(-1, 3)
side_glass = glass[np.abs(glass[:, 0]) > 0.7]


def glass_bottom(z):
    near = side_glass[np.abs(side_glass[:, 2] - z) < 0.10]
    return near[:, 1].min() if len(near) else np.nan


def shoot(sign, Y, Zs):
    """x of the first paint hit going outward from (sign*X_START, y, z); nan if none."""
    v0 = paint[:, 0]
    e1, e2 = paint[:, 1] - v0, paint[:, 2] - v0
    out = np.full(len(Y), np.nan)
    D = np.array([sign, 0.0, 0.0])
    h = np.cross(D, e2)
    a = e1 @ h.T if False else np.einsum("nj,nj->n", e1, h)
    ok = np.abs(a) > 1e-12
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    for k in range(len(Y)):
        o = np.array([sign * X_START, Y[k], Zs[k]])
        s = o - v0
        u = f * np.einsum("nj,nj->n", s, h)
        q = np.cross(s, e1)
        v = f * (q @ D)
        t = f * np.einsum("nj,nj->n", e2, q)
        m = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4)
        if m.any():
            x = o[0] + sign * t[m]
            x = x[np.abs(x) >= X_MIN]
            if len(x):
                out[k] = x[np.argmin(np.abs(x))]
    return out


zs = np.arange(Z0, Z1 + 1e-9, DZ)
nz = len(zs)
# upper edge per column: just under the side glass, smoothed; A-pillar base height where there is no side glass
top = np.array([glass_bottom(z) for z in zs])
top = np.where(np.isnan(top), Y_TOP_FRONT + GLASS_GAP, top)
top = np.minimum(ndimage.uniform_filter1d(top, 9, mode="nearest"), Y_TOP_FRONT + GLASS_GAP) - GLASS_GAP

all_P, all_N = [], []
for sign in (+1, -1):
    ny = int(round((top.max() - Y0) / DY)) + 1
    ys = np.linspace(Y0, top.max(), ny)
    ZZ, YY = np.meshgrid(zs, ys)  # (ny, nz)
    X = shoot(sign, YY.ravel(), ZZ.ravel()).reshape(ny, nz)
    X = np.abs(X)
    bad = np.isnan(X)
    med = ndimage.median_filter(np.where(bad, np.nanmedian(X), X), size=7, mode="nearest")
    bad |= np.abs(X - med) > 0.06            # a hit on some far part of the body is not the skin here
    print("side", sign, "grid", X.shape, "holes filled:", int(bad.sum()), "of", X.size)
    # fill: nearest valid value, then smooth only the filled part so it blends with its surroundings
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    F = X[tuple(idx)]
    F[bad] = X[tuple(idx)][bad]
    for _ in range(25):
        S = ndimage.uniform_filter(F, 3, mode="nearest")
        F = np.where(bad, S, F)
    W = (F - OFFSET) * sign
    # lower rows below the wall start are kept; cut the vertices above the column's top edge by squashing them onto it
    YC = np.minimum(YY, top[None, :])
    # a gently sloped wall never goes nearer the centre than the dash/door cards (0.78) so it cannot cut into them
    W = np.where(sign > 0, np.maximum(W, 0.78), np.minimum(W, -0.78))
    G = np.stack([W, YC, ZZ], -1)         # (ny, nz, 3)
    # smooth normals toward the cabin centre line from the grid
    dz_ = np.gradient(G, axis=1)
    dy_ = np.gradient(G, axis=0)
    N = np.cross(dz_, dy_)
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
    N = np.where((N[..., 0] * -sign > 0)[..., None], N, -N)
    base = sum(len(p) for p in all_P)
    all_P.append(G.reshape(-1, 3))
    all_N.append(N.reshape(-1, 3))
    ii = np.arange(ny * nz).reshape(ny, nz)
    # degenerate quads (top squashed) are harmless: zero-area triangles
    a_, b_, c_, d_ = ii[:-1, :-1], ii[:-1, 1:], ii[1:, 1:], ii[1:, :-1]
    all_idx = np.stack([a_, b_, c_, a_, c_, d_], -1).reshape(-1, 3)
    all_P[-1] = (G.reshape(-1, 3), all_idx)

P_list, N_list, I_list, off = [], [], [], 0
for (Pg, Ig), Ng in zip(all_P, all_N):
    P_list.append(Pg); N_list.append(Ng); I_list.append(Ig + off); off += len(Pg)
P = np.concatenate(P_list).astype("<f4")
NN = np.concatenate(N_list).astype("<f4")
IDX = np.concatenate(I_list).astype("<u4").reshape(-1, 1)
UV = (np.c_[P[:, 2], P[:, 1]] / 0.20).astype("<f4")   # the alcantara's tiling, as in the door cards
mesh["primitives"].append({
    "attributes": {
        "POSITION": add_accessor(P, "VEC3", 5126, 34962, True),
        "NORMAL": add_accessor(NN, "VEC3", 5126, 34962),
        "TEXCOORD_0": add_accessor(UV, "VEC2", 5126, 34962),
    },
    "indices": add_accessor(IDX, "SCALAR", 5125, 34963),
    "material": next(i for i, m in enumerate(J["materials"]) if m["name"] == NAME),
    "extras": {"side_wall": True},
})

car = next(nd for nd in J["nodes"] if nd["name"] == "CAR")
ch = car["extras"]["source"]["changes"]
if "side walls" not in ch:
    car["extras"]["source"]["changes"] = ch + "; solid side walls fill the door sides and A-pillar bases (side walls: no gaps, no loose patches)"
J["buffers"][0]["byteLength"] = len(BIN)
js = json.dumps(J, separators=(",", ":")).encode()
while len(js) % 4:
    js += b" "
while len(BIN) % 4:
    BIN.append(0)
open(OUT, "wb").write(
    struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(BIN)) + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(BIN), 0x004E4942) + bytes(BIN)
)
print("ok", len(BIN) // 1024, "KB ->", OUT)
