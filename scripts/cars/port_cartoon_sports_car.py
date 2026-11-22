#!/usr/bin/env python3
"""
Ports "Modern Cartoon Sports Car" (RCC Design, CC BY 4.0, https://skfb.ly/oPytt) onto the car rig of the ported cars
(src/lib/ported-car-model.ts):

  CAR (extras: wheelCentres, tyreRadius, tyreHalfWidth, cameras, helmet, source)
    BODY                     paint, details, glass and lamps, one primitive per material
    WHEEL_LF/LR/RF/RR        hub pivot at the tread centre; children WHEEL_xx_TYRE and WHEEL_xx_RIM (rim + brake disc)

Changes from the source model: the fake shadow plane and the brake calipers are left out, the clearcoat and specular
extensions are dropped (the paint gets a glossy roughness instead), the 4096 px textures are downscaled to 1024-2048 px
and re-encoded as JPEG, and the nodes are rebuilt on the rig. The model is already in metres (+X left, +Y up, +Z forward).

  python3 scripts/cars/port_cartoon_sports_car.py SRC.glb OUT.glb
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
BIN = d[28 + jl:28 + jl + bl]

NP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def accessor(i):
    a = J["accessors"][i]
    bv = J["bufferViews"][a["bufferView"]]
    dt = np.dtype(NP[a["componentType"]]).newbyteorder("<")
    n = NC[a["type"]]
    stride = bv.get("byteStride") or dt.itemsize * n
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    if stride == dt.itemsize * n:
        arr = np.frombuffer(BIN, dtype=dt, count=a["count"] * n, offset=off).reshape(a["count"], n)
    else:
        arr = np.stack([np.frombuffer(BIN, dtype=dt, count=n, offset=off + k * stride) for k in range(a["count"])])
    return arr.astype(np.float32) if a["componentType"] == 5126 else arr.astype(np.int64)


def local_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T
    t = np.array(n.get("translation", [0, 0, 0]), dtype=np.float64)
    q = np.array(n.get("rotation", [0, 0, 0, 1]), dtype=np.float64)
    s = np.array(n.get("scale", [1, 1, 1]), dtype=np.float64)
    x, y, z, w = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = r * s
    m[:3, 3] = t
    return m


# ------------------------------------------------------------------------------------------------- source geometry
parts = []  # dict(name, material, pos, nrm, uv, idx) in world space
def walk(i, parent):
    n = J["nodes"][i]
    m = parent @ local_matrix(n)
    if "mesh" in n:
        for p in J["meshes"][n["mesh"]]["primitives"]:
            if p.get("mode", 4) != 4:
                continue
            pos = accessor(p["attributes"]["POSITION"])
            nrm = accessor(p["attributes"]["NORMAL"])
            uv = accessor(p["attributes"]["TEXCOORD_0"])
            idx = accessor(p["indices"]).reshape(-1)
            normal_m = np.linalg.inv(m[:3, :3]).T
            wp = pos @ m[:3, :3].T + m[:3, 3]
            wn = nrm @ normal_m.T
            wn /= np.maximum(np.linalg.norm(wn, axis=1, keepdims=True), 1e-9)
            parts.append(dict(name=n.get("name", ""), material=J["materials"][p["material"]]["name"], pos=wp.astype(np.float32),
                              nrm=wn.astype(np.float32), uv=uv.astype(np.float32), idx=idx))
    for c in n.get("children", []):
        walk(c, m)

for r in J["scenes"][J.get("scene", 0)]["nodes"]:
    walk(r, np.eye(4))

body_parts, wheel_parts = [], {"fl": {}, "fr": {}, "rl": {}, "rr": {}}
for p in parts:
    nm = p["name"].lower()
    if "shadow" in nm or nm.startswith("caliper_"):
        continue
    for corner in wheel_parts:
        if nm.endswith(f"_{corner}_488_wheels_0") or nm.startswith(("tyre_", "rim_", "bdisk_")) and f"_{corner}_" in nm:
            kind = "tyre" if nm.startswith("tyre_") else "rim"  # the brake disc turns with the rim
            wheel_parts[corner].setdefault(kind, []).append(p)
            break
    else:
        body_parts.append(p)

# ------------------------------------------------------------------------------------------------- glTF writer
bin_ = bytearray()
views, accs, images, textures, materials = [], [], [], [], []


def view(data, target=None):
    while len(bin_) % 4:
        bin_.append(0)
    v = {"buffer": 0, "byteOffset": len(bin_), "byteLength": len(data)}
    if target:
        v["target"] = target
    bin_.extend(data)
    views.append(v)
    return len(views) - 1


def acc(arr, ctype, typ, target, minmax=False):
    a = {"bufferView": view(arr.tobytes(), target), "componentType": ctype, "count": int(len(arr)), "type": typ}
    if minmax:
        a["min"] = [float(x) for x in arr.min(axis=0)]
        a["max"] = [float(x) for x in arr.max(axis=0)]
    accs.append(a)
    return len(accs) - 1


def source_image(i):
    im = J["images"][i]
    bv = J["bufferViews"][im["bufferView"]]
    raw = BIN[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
    return Image.open(io.BytesIO(raw))


tex_cache = {}
def tex(src_image, size, quality=84):
    key = (src_image, size)
    if key not in tex_cache:
        im = source_image(src_image).convert("RGB")
        if im.width > size:
            im = im.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
        images.append({"bufferView": view(buf.getvalue()), "mimeType": "image/jpeg"})
        textures.append({"sampler": 0, "source": len(images) - 1})
        tex_cache[key] = len(textures) - 1
    return {"index": tex_cache[key]}


src_mats = {m["name"]: m for m in J["materials"]}
paint = src_mats["488_PAINT"]["pbrMetallicRoughness"]
MAT = {}
def make_materials():
    MAT["488_PAINT"] = {"name": "paint", "doubleSided": True, "pbrMetallicRoughness": {
        "baseColorFactor": paint.get("baseColorFactor", [1, 1, 1, 1]), "metallicFactor": 0.2, "roughnessFactor": 0.32}}
    MAT["488_DETAILS"] = {"name": "details", "doubleSided": True, "pbrMetallicRoughness": {
        "baseColorTexture": tex(1, 2048), "metallicRoughnessTexture": tex(2, 1024)}}
    g = src_mats["488_GLASS"]["pbrMetallicRoughness"]["baseColorFactor"]
    MAT["488_GLASS"] = {"name": "glass", "doubleSided": True, "alphaMode": "BLEND", "pbrMetallicRoughness": {
        "baseColorFactor": [g[0], g[1], g[2], 0.32], "metallicFactor": 0.0, "roughnessFactor": 0.05}}
    MAT["488_LAMPS"] = {"name": "lamps", "doubleSided": True, "emissiveFactor": [1, 1, 1], "emissiveTexture": tex(5, 1024),
                        "pbrMetallicRoughness": {"baseColorTexture": tex(1, 2048), "metallicFactor": 0.0, "roughnessFactor": 0.4}}
    MAT["488_LAMPS_OFF"] = {"name": "blinkers", "doubleSided": True, "pbrMetallicRoughness": {
        "baseColorTexture": tex(1, 2048), "metallicFactor": 0.0, "roughnessFactor": 0.89}}
    MAT["488_WHEELS"] = {"name": "wheels", "doubleSided": True, "normalTexture": tex(8, 1024), "pbrMetallicRoughness": {
        "baseColorTexture": tex(6, 1024), "metallicRoughnessTexture": tex(7, 1024)}}
make_materials()
mat_index = {}
for k, m in MAT.items():
    materials.append(m)
    mat_index[k] = len(materials) - 1


def mesh_of(plist, origin=(0, 0, 0)):
    """One glTF mesh from parts, one primitive per material; positions made relative to `origin`"""
    by_mat = {}
    for p in plist:
        by_mat.setdefault(p["material"], []).append(p)
    prims = []
    for mname, ps in by_mat.items():
        pos, nrm, uv, idx, base = [], [], [], [], 0
        for p in ps:
            pos.append(p["pos"] - np.array(origin, dtype=np.float32))
            nrm.append(p["nrm"])
            uv.append(p["uv"])
            idx.append(p["idx"] + base)
            base += len(p["pos"])
        pos, nrm, uv, idx = (np.concatenate(x) for x in (pos, nrm, uv, idx))
        itype = (np.uint16, 5123) if len(pos) < 65535 else (np.uint32, 5125)
        prims.append({"attributes": {"POSITION": acc(pos.astype(np.float32), 5126, "VEC3", 34962, True),
                                     "NORMAL": acc(nrm.astype(np.float32), 5126, "VEC3", 34962),
                                     "TEXCOORD_0": acc(uv.astype(np.float32), 5126, "VEC2", 34962)},
                      "indices": acc(idx.astype(itype[0]).reshape(-1, 1), itype[1], "SCALAR", 34963),
                      "material": mat_index[mname]})
    return prims


meshes, nodes = [], []
CORNER_NODE = {"fl": "WHEEL_LF", "fr": "WHEEL_RF", "rl": "WHEEL_LR", "rr": "WHEEL_RR"}
centres, radius, halfw = {}, {}, {}
car_children = []

meshes.append({"name": "BODY", "primitives": mesh_of(body_parts)})
nodes.append({"name": "BODY", "mesh": 0})
car_children.append(0)

for corner, kinds in wheel_parts.items():
    name = CORNER_NODE[corner]
    tyre_pos = np.concatenate([p["pos"] for p in kinds["tyre"]])
    lo, hi = tyre_pos.min(axis=0), tyre_pos.max(axis=0)
    c = (lo + hi) / 2
    centres[name] = [round(float(x), 5) for x in c]
    radius[name] = round(float((hi[1] - lo[1]) / 2), 5)
    halfw[name] = round(float((hi[0] - lo[0]) / 2), 5)
    hub = len(nodes)
    nodes.append({"name": name, "translation": [float(x) for x in c], "children": [hub + 1, hub + 2]})
    for kind in ("tyre", "rim"):
        meshes.append({"name": f"{name}_{kind.upper()}", "primitives": mesh_of(kinds[kind], origin=c)})
        nodes.append({"name": f"{name}_{kind.upper()}", "mesh": len(meshes) - 1})
    car_children.append(hub)

body_pos = np.concatenate([p["pos"] for p in body_parts])
print("body bounds", np.round(body_pos.min(axis=0), 3), np.round(body_pos.max(axis=0), 3))
print("wheel centres", centres, "radius", radius, "half width", halfw)

CAM = json.loads(open(sys.argv[3]).read()) if len(sys.argv) > 3 else {}
extras = {"wheelCentres": centres, "tyreRadius": radius, "tyreHalfWidth": halfw, "liveryMask": False,
          "cameras": CAM.get("cameras", {}), "helmet": CAM.get("helmet", {"pos": [0.4, 1.2, -0.1]}),
          "source": {"title": "Modern Cartoon Sports Car", "author": "RCC Design", "url": "https://skfb.ly/oPytt", "license": "CC BY 4.0",
                     "changes": "Rebuilt on the car rig (body, wheels, rims, tyres); fake shadow plane and brake calipers removed; clearcoat and specular extensions dropped; textures downscaled and re-encoded as JPEG"}}
nodes.append({"name": "CAR", "children": car_children, "extras": extras})

gltf = {"asset": {"version": "2.0", "generator": "Noryx cartoon-sports-car converter"}, "scene": 0, "scenes": [{"name": "Scene", "nodes": [len(nodes) - 1]}],
        "nodes": nodes, "meshes": meshes, "materials": materials, "textures": textures, "images": images,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(bin_)}]}
js = json.dumps(gltf, separators=(",", ":")).encode()
js += b" " * (-len(js) % 4)
bin_ += b"\0" * (-len(bin_) % 4)
out = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(bin_)) + struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(bin_), b"BIN\0") + bytes(bin_)
open(OUT, "wb").write(out)
print(f"wrote {OUT}: {len(out) / 1e6:.2f} MB, {len(images)} textures")
