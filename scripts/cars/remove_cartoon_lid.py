#!/usr/bin/env python3
"""
Removes the "lid" from the ported "Modern Cartoon Sports Car" (scripts/cars/port_cartoon_sports_car.py output).

The source model has no interior: under its glass a flat, opaque sheet (part of the `details` primitive, sloping from
y 1.28 at the rear seats to y 1.18 at the base of the windscreen) hides the empty cabin. Once add_cartoon_interior.py put
the driver's eye inside the cabin (y 1.15-1.22) that sheet sat over the camera: the dash and cockpit views looked at its
dark underside, which hid the road and the horizon (a black band across the screen, or a fully black view), and the
gauges, seats and wheel were all under it.

The sheet's triangles are the ones of `details` that face up or down, lie at cabin height (y 1.10-1.42), inside the
cabin's footprint (|x| < 1.05, z from the parcel shelf at -1.10 to the windscreen base at 1.30). The windscreen's black frame goes too (see FRAME_*). Anything else in
`details` (pillars, mirrors, vents, the rear deck behind the shelf) stays. It also lifts the cockpit camera 7 cm and sets
the dash anchor ahead of the instrument hood under the windscreen (the game's own dash view, onboard-views.ts, is placed
from the cockpit eye). The script is idempotent.

  python3 scripts/cars/remove_cartoon_lid.py IN.glb OUT.glb
"""
import io
import json
import struct
import sys

import numpy as np
from PIL import Image

SRC, OUT = sys.argv[1], sys.argv[2]
d = open(SRC, "rb").read()
jl = struct.unpack("<I", d[12:16])[0]
J = json.loads(d[20:20 + jl])
bl = struct.unpack("<I", d[20 + jl:24 + jl])[0]
BIN = bytearray(d[28 + jl:28 + jl + bl])

Y_RANGE = (1.10, 1.42)
Z_RANGE = (-1.10, 1.30)
X_MAX = 1.05
MIN_FLATNESS = 0.7  # |normal.y| of a lid triangle


# The windscreen's black frame (the rubber seal, A-pillar lining and wiper cowl, dark texels of the `details` atlas): a few
# centimetres thick, but with the eye 35 cm from the glass it filled the top of the cockpit view and a black band crossed
# the dash view. Removed from the glass's footprint (the exterior keeps the paint's own pillars and roof).
FRAME_Y_MIN = 1.15
FRAME_Z = (-0.35, 1.30)
DARK_LUMA = 0.10
# The dark wheel-arch liner under the bonnet's rear edge, in the gap between the windscreen's base and the bonnet: seen from
# the dash view as a black blob (the bonnet's paint covers it from outside)
LINER_Y_MIN = 0.95
LINER_Z = (0.60, 1.45)
LINER_X_MIN = 0.45


def accessor(i):
    a = J["accessors"][i]
    bv = J["bufferViews"][a["bufferView"]]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
    dt = {5126: "<f4", 5123: "<u2", 5125: "<u4"}[a["componentType"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    return np.frombuffer(BIN, dtype=dt, count=a["count"] * n, offset=off).reshape(a["count"], n)


body = next(n for n in J["nodes"] if n["name"] == "BODY")
removed = 0
for prim in J["meshes"][body["mesh"]]["primitives"]:
    if J["materials"][prim["material"]]["name"] != "details":
        continue
    pos = accessor(prim["attributes"]["POSITION"]).astype(np.float64)
    idx = accessor(prim["indices"]).reshape(-1, 3).astype(np.int64)
    tri = pos[idx]
    c = tri.mean(1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    uvs = accessor(prim["attributes"]["TEXCOORD_0"]).astype(np.float64)
    bt = J["materials"][prim["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    bvw = J["bufferViews"][J["images"][J["textures"][bt]["source"]]["bufferView"]]
    o = bvw.get("byteOffset", 0)
    tex = np.asarray(Image.open(io.BytesIO(bytes(BIN[o:o + bvw["byteLength"]]))).convert("RGB")).astype(np.float64) / 255
    th, tw, _ = tex.shape
    uv = uvs[idx].mean(1)
    luma = tex[(uv[:, 1] % 1 * th).astype(int).clip(0, th - 1), (uv[:, 0] % 1 * tw).astype(int).clip(0, tw - 1)].mean(1)
    frame = (luma < DARK_LUMA) & (c[:, 1] >= FRAME_Y_MIN) & (c[:, 2] >= FRAME_Z[0]) & (c[:, 2] <= FRAME_Z[1])
    liner = (luma < DARK_LUMA) & (c[:, 1] >= LINER_Y_MIN) & (c[:, 2] >= LINER_Z[0]) & (c[:, 2] <= LINER_Z[1]) & (np.abs(c[:, 0]) >= LINER_X_MIN)
    lid = frame | liner | (
        (np.abs(nrm[:, 1]) > MIN_FLATNESS)
        & (c[:, 1] >= Y_RANGE[0]) & (c[:, 1] <= Y_RANGE[1])
        & (c[:, 2] >= Z_RANGE[0]) & (c[:, 2] <= Z_RANGE[1])
        & (np.abs(c[:, 0]) <= X_MAX)
    )
    removed += int(lid.sum())
    if not lid.any():
        continue
    keep = np.ascontiguousarray(idx[~lid].reshape(-1).astype("<u4"))
    while len(BIN) % 4:
        BIN.append(0)
    J["bufferViews"].append({"buffer": 0, "byteOffset": len(BIN), "byteLength": keep.nbytes, "target": 34963})
    BIN.extend(keep.tobytes())
    J["accessors"].append({"bufferView": len(J["bufferViews"]) - 1, "componentType": 5125, "count": int(keep.size), "type": "SCALAR"})
    prim["indices"] = len(J["accessors"]) - 1

J["buffers"][0]["byteLength"] = len(BIN)
car = next(n for n in J["nodes"] if n["name"] == "CAR")
src = car["extras"]["source"]
cams = car["extras"]["cameras"]
cams["cockpit"] = {"pos": [0.38, 1.22, -0.05], "pitchDeg": -3, "fov": 60}
cams["dash"] = {"pos": [0.38, 1.27, 0.85], "pitchDeg": -2, "fov": 58}
if removed and "opaque sheet that hid the empty cabin" not in src["changes"]:
    src["changes"] += "; the opaque sheet that hid the empty cabin under the glass removed"
js = json.dumps(J, separators=(",", ":")).encode()
while len(js) % 4:
    js += b" "
while len(BIN) % 4:
    BIN.append(0)
open(OUT, "wb").write(struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(BIN)) + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(BIN), 0x004E4942) + bytes(BIN))
print("removed", removed, "triangles ->", OUT)
