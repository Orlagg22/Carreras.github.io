#!/usr/bin/env python3
"""
Last pass over the ported "Modern Cartoon Sports Car" (run after rebuild_cartoon_interior.py and remove_cartoon_lid.py).

  1. Removes the interior rear-view mirror (the `int_mirror` glass plus its housing, stalk and roof mount, which are
     `int_trim` triangles around x 0, y 1.26-1.46, z 0.74-0.90).
  2. Removes the headliner / sun visors (`int_liner`): through the roof glass they showed from outside as a grey square.
  3. Removes the triangles of `details`, `glass`, `lamps` and `blinkers` that lie flush with, or up to 4 mm under, a
     paint triangle that covers them and faces the same way. They are never visible, but they share depth with the
     paint, so as the car moved the depth buffer picked between the two surfaces pixel by pixel: the grey / black
     flickering and diagonal hatching on the bonnet. Decals standing more than 0.05 mm above the paint stay.

  python3 scripts/cars/finish_cartoon_car.py IN.glb OUT.glb
"""
import json
import struct
import sys

import numpy as np
from scipy.spatial import cKDTree

SRC, OUT = sys.argv[1], sys.argv[2]
d = open(SRC, "rb").read()
jl = struct.unpack("<I", d[12:16])[0]
J = json.loads(d[20:20 + jl])
bl = struct.unpack("<I", d[20 + jl:24 + jl])[0]
BIN = bytearray(d[28 + jl:28 + jl + bl])

MIRROR_BOX = ((-0.20, 1.26, 0.74), (0.20, 1.46, 0.90))
MAX_GAP = 0.004          # m: flush .. 4 mm under the paint
SEARCH_R = 0.15
HIDE_IN = ("details", "glass", "lamps", "blinkers")


def accessor(i):
    a = J["accessors"][i]
    bv = J["bufferViews"][a["bufferView"]]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
    dt = {5126: "<f4", 5123: "<u2", 5125: "<u4"}[a["componentType"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    return np.frombuffer(BIN, dtype=dt, count=a["count"] * n, offset=off).reshape(a["count"], n)


def set_indices(prim, keep_idx):
    keep = np.ascontiguousarray(keep_idx.reshape(-1).astype("<u4"))
    while len(BIN) % 4:
        BIN.append(0)
    J["bufferViews"].append({"buffer": 0, "byteOffset": len(BIN), "byteLength": keep.nbytes, "target": 34963})
    BIN.extend(keep.tobytes())
    J["accessors"].append({"bufferView": len(J["bufferViews"]) - 1, "componentType": 5125, "count": int(keep.size), "type": "SCALAR"})
    prim["indices"] = len(J["accessors"]) - 1


body = next(n for n in J["nodes"] if n["name"] == "BODY")
mesh = J["meshes"][body["mesh"]]
mat_name = lambda p: J["materials"][p["material"]]["name"]


def tris(prim):
    pos = accessor(prim["attributes"]["POSITION"]).astype(np.float64)
    idx = accessor(prim["indices"]).reshape(-1, 3).astype(np.int64)
    return idx, pos[idx]


# ---- 1 + 2: mirror and headliner ---------------------------------------------------------------------------------------
before = len(mesh["primitives"])
mesh["primitives"] = [p for p in mesh["primitives"] if mat_name(p) not in ("int_mirror", "int_liner")]
print("primitives removed (mirror glass, headliner):", before - len(mesh["primitives"]))
for p in mesh["primitives"]:
    if mat_name(p) != "int_trim":
        continue
    idx, t = tris(p)
    c = t.mean(1)
    lo, hi = np.array(MIRROR_BOX[0]), np.array(MIRROR_BOX[1])
    gone = np.all((c >= lo) & (c <= hi), axis=1)
    print("mirror housing / stalk / mount triangles removed:", int(gone.sum()))
    if gone.any():
        set_indices(p, idx[~gone])

# ---- 3: triangles hidden under (or flush with) the paint ----------------------------------------------------------------
paint = next(p for p in mesh["primitives"] if mat_name(p) == "paint")
_, pt = tris(paint)
pc = pt.mean(1)
pn = np.cross(pt[:, 1] - pt[:, 0], pt[:, 2] - pt[:, 0])
pn /= np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-12)
tree = cKDTree(pc)


def covered_by_paint(point, normal, k_hits):
    """Is `point` inside a paint triangle that faces `normal` and lies 0.05 .. 4 mm above it (or flush)?"""
    for j in k_hits:
        off = float(np.dot(point - pc[j], pn[j]))
        if abs(off) > MAX_GAP or float(np.dot(normal, pn[j])) < 0.95:
            continue
        if off > 0.00005:  # standing more than 0.05 mm above the paint: a decal, kept
            continue
        a, b, c = pt[j]
        q = point - off * pn[j]  # onto the paint's plane
        v0, v1, v2 = b - a, c - a, q - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-15:
            continue
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        if v >= -0.02 and w >= -0.02 and v + w <= 1.02:
            return True
    return False


total = 0
for p in mesh["primitives"]:
    name = mat_name(p)
    if name not in HIDE_IN:
        continue
    idx, t = tris(p)
    c = t.mean(1)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    dist, hits = tree.query(c, k=12)
    gone = np.zeros(len(c), bool)
    for i in range(len(c)):
        close = [hits[i, k] for k in range(12) if dist[i, k] <= SEARCH_R]
        if close and covered_by_paint(c[i], n[i], close):
            gone[i] = True
    print(f"{name}: {int(gone.sum())} of {len(c)} hidden / flush triangles removed")
    total += int(gone.sum())
    if gone.any():
        set_indices(p, idx[~gone])

J["buffers"][0]["byteLength"] = len(BIN)
car = next(n for n in J["nodes"] if n["name"] == "CAR")
ch = car["extras"]["source"]["changes"]
for note in ("interior mirror and headliner removed", "triangles hidden under the paint removed (depth flicker)"):
    if note not in ch:
        ch += "; " + note
car["extras"]["source"]["changes"] = ch
js = json.dumps(J, separators=(",", ":")).encode()
while len(js) % 4:
    js += b" "
while len(BIN) % 4:
    BIN.append(0)
open(OUT, "wb").write(
    struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(BIN)) + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(BIN), 0x004E4942) + bytes(BIN)
)
print("removed", total, "hidden triangles ->", OUT)
