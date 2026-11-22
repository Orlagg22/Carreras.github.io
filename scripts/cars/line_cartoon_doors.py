#!/usr/bin/env python3
"""
Lines the cartoon car's doors from the inside (run after rebuild_cartoon_interior.py).

The body is a hollow shell: the door cards stand 15-20 cm inside the paint, and from the driver's seat everything above
them, up to the side glass, was the paint's own back, white, with the exterior's handle and creases, so the doors
looked see-through / unfinished. This copies the paint triangles of that band on both sides, moves each 12 mm toward the
cabin along its own normal (so the lining follows the skin's curves exactly and the paint still hides it from outside) and
gives them the dark alcantara of the door cards (`int_dash_low`). The band stops under the side glass.

  python3 scripts/cars/line_cartoon_doors.py IN.glb OUT.glb
"""
import json
import struct
import sys

import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
d = open(SRC, "rb").read()
jl = struct.unpack("<I", d[12:16])[0]
J = json.loads(d[20:20 + jl])
bl = struct.unpack("<I", d[20 + jl:24 + jl])[0]
BIN = bytearray(d[28 + jl:28 + jl + bl])

OFFSET = 0.012          # m, toward the cabin
X_MIN, X_MAX = 0.82, 1.22
Y_MIN = 0.90
Z_MIN, Z_MAX = -1.05, 0.98
GLASS_GAP = 0.012       # stop this far under the glass' lower edge
NAME = "int_door_lining"


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

# a lining from an earlier run is dropped first, so the script can be run again
mesh["primitives"] = [p for p in mesh["primitives"] if mat_of(p) != "int_dash_low" or "lining" not in p.get("extras", {})]

prims = {mat_of(p): p for p in mesh["primitives"]}


def tris(p):
    pos = accessor(p["attributes"]["POSITION"]).astype(np.float64)
    return pos[accessor(p["indices"]).reshape(-1, 3).astype(np.int64)]


paint = tris(prims["paint"])
glass = tris(prims["glass"]).reshape(-1, 3)
side_glass = glass[np.abs(glass[:, 0]) > 0.7]


def glass_bottom(z):
    near = side_glass[np.abs(side_glass[:, 2] - z) < 0.14]
    return near[:, 1].min() if len(near) else 1.22


c = paint.mean(1)
n = np.cross(paint[:, 1] - paint[:, 0], paint[:, 2] - paint[:, 0])
area2 = np.linalg.norm(n, axis=1)
n /= np.maximum(area2[:, None], 1e-12)

pick = (np.abs(c[:, 0]) > X_MIN) & (np.abs(c[:, 0]) < X_MAX) & (c[:, 1] > Y_MIN) & (c[:, 2] > Z_MIN) & (c[:, 2] < Z_MAX) & (area2 > 1e-10)
limit = np.array([glass_bottom(z) for z in c[:, 2]]) - GLASS_GAP
# the highest vertex too, so no triangle pokes up into the glass
pick &= (c[:, 1] < limit) & (paint[:, :, 1].max(1) < limit + 0.03)
T = paint[pick]
N = n[pick]
sgn = np.sign(T.mean(1)[:, 0])
# the normal pointing at the cabin's centre line
N = np.where((N[:, 0] * -sgn > 0)[:, None], N, -N)
T = T + N[:, None, :] * OFFSET
print("lining triangles:", len(T), "(of", int(((np.abs(c[:, 0]) > X_MIN)).sum()), "paint triangles on the door sides)")

P = T.reshape(-1, 3).astype("<f4")
NN = np.repeat(N, 3, axis=0).astype("<f4")
UV = np.c_[P[:, 2], P[:, 1]].astype("<f4") / 0.20  # the alcantara's tiling, as in the door cards
IDX = np.arange(len(P), dtype="<u4").reshape(-1, 1)
prim = {
    "attributes": {
        "POSITION": add_accessor(P, "VEC3", 5126, 34962, True),
        "NORMAL": add_accessor(NN, "VEC3", 5126, 34962),
        "TEXCOORD_0": add_accessor(UV, "VEC2", 5126, 34962),
    },
    "indices": add_accessor(IDX, "SCALAR", 5125, 34963),
    "material": next(i for i, m in enumerate(J["materials"]) if m["name"] == "int_dash_low"),
    "extras": {"lining": True},
}
mesh["primitives"].append(prim)

car = next(nd for nd in J["nodes"] if nd["name"] == "CAR")
ch = car["extras"]["source"]["changes"]
if "doors lined" not in ch:
    car["extras"]["source"]["changes"] = ch + "; doors lined from the inside (doors lined: the paint's back no longer shows)"
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
