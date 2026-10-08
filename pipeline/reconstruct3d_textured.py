"""Full textured export for one ball: meshes WITH per-vertex H&E colours +
registered, ball-masked RGB slice textures for in-3D projection.

Usage: python reconstruct3d_textured.py <glom-number>
Writes into glom3d-viewer/data/: glom_22-184_g{N}.json (+ "slices", vertex "c")
and tex_g{N}/sec*.webp (RGBA, alpha = feathered ball mask).
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import openslide
from scipy import ndimage
from scipy.signal import savgol_filter
from skimage.measure import marching_cubes

from reconstruct3d_g1 import (CANVAS, DS, MPP, OFF, S, ZUM, b64, compose,
                              gray_patch, load_sections, poly_area, rasterise, sdf)
from reconstruct3d_g1_v2 import local_ecc
from reconstruct3d_g1_v3 import GRID, VXY, field, sdf_upsample_z

G = int(sys.argv[1])
REPO = Path("/data3/zhuy39/GLomSeg/glom3d-viewer/data")
TEXDIR = REPO / f"tex_g{G}"
TEXDIR.mkdir(parents=True, exist_ok=True)
NEW = Path("/data3/zhuy39/GLomSeg/new data/22-184")

secs = load_sections(G)
order = sorted(secs)
slides = {}

# ---- registration (v2 recipe) ----
warps = {}
base = np.array([[1, 0, OFF], [0, 1, OFF]], np.float32)
for i, s in enumerate(order):
    if i == 0:
        warps[s] = base
        continue
    prev = order[i - 1]
    pinfo, info = secs[prev], secs[s]
    p_c = pinfo["host"].mean(0) - np.array(pinfo["origin"])
    c_c = info["host"].mean(0) - np.array(info["origin"])
    W, ok = local_ecc(gray_patch(slides, pinfo), gray_patch(slides, info), p_c, c_c)
    if not ok:
        W = np.array([[1, 0, p_c[0] - c_c[0]], [0, 1, p_c[1] - c_c[1]]], np.float32)
    warps[s] = compose(warps[prev], W)
cents = []
for s in order:
    W3 = np.vstack([warps[s], [0, 0, 1]])
    c = secs[s]["host"].mean(0) - np.array(secs[s]["origin"])
    cents.append((W3 @ [c[0], c[1], 1.0])[:2])
cents = np.array(cents)
win = min(9, len(order) - (1 - len(order) % 2))
smooth = np.column_stack([savgol_filter(cents[:, k], win, 2) for k in (0, 1)])
for i, s in enumerate(order):
    d = cents[i] - smooth[i]
    warps[s] = compose(np.array([[1, 0, -d[0]], [0, 1, -d[1]]], np.float32), warps[s])

# ---- masks + registered RGB at GRID resolution ----
z0, z1 = order[0], order[-1]
nz = z1 - z0 + 1
classes = ["glom", "vessels", "mesangial"]
stacks = {c: np.zeros((nz, GRID, GRID), np.uint8) for c in classes}
rgb = {}           # real z -> GRID RGB (BGR)
cells = []
SC = np.array([[1 / DS, 0, 0], [0, 1 / DS, 0]], np.float32)
for s in order:
    info, W = secs[s], warps[s]
    z = s - z0
    for c, polys in [("glom", [info["host"]]), ("vessels", info["inner"]["vessels"]),
                     ("mesangial", info["inner"]["mesangial"])]:
        m = rasterise(polys, info["origin"], W).astype(np.float32)
        stacks[c][z] = (cv2.resize(m, (GRID, GRID), interpolation=cv2.INTER_AREA) > 0.5)
    sl = info["slide"]
    if sl not in slides:
        slides[sl] = openslide.OpenSlide(str(NEW / sl))
    img = np.array(slides[sl].read_region(info["origin"], 0, (S, S)).convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    Wsm = compose(SC, W)
    rgb[z] = cv2.warpAffine(img, Wsm, (GRID, GRID), flags=cv2.INTER_AREA,
                            borderValue=(245, 244, 246))
    W3 = np.vstack([W, [0, 0, 1]])
    for p in info["inner"]["cell"]:
        cc = p.mean(0) - np.array(info["origin"])
        cx, cy, _ = W3 @ np.array([cc[0], cc[1], 1.0])
        cells.append((z * ZUM, cx * MPP, cy * MPP,
                      max(math.sqrt(poly_area(p) / math.pi) * MPP, 1.0)))
cells = np.array(cells, np.float32)

have = sorted(rgb)
for c in classes:
    v = stacks[c]
    for z in range(nz):
        if z in have:
            continue
        lo = max(h for h in have if h < z)
        hi = min(h for h in have if h > z)
        t = (z - lo) / (hi - lo)
        v[z] = ((1 - t) * sdf(v[lo]) + t * sdf(v[hi]) > 0).astype(np.uint8)

# ---- meshes (v3 recipe) + per-vertex colour from nearest real section ----
spz = ZUM / 3
up = {c: sdf_upsample_z(stacks[c], 3) for c in stacks}
flds = {c: field(up[c], (1.0, 1.0, 1.0)) for c in up}
real_fine = np.array([h * 3 for h in have], float)
out_meshes = {}
for c, f in flds.items():
    if f.max() < 0.5:
        continue
    verts, faces, normals, _ = marching_cubes(f, 0.5, spacing=(spz, VXY, VXY))
    vxyz = verts[:, [2, 1, 0]].astype(np.float32)   # x,y,z um (z includes 1-slice pad)
    fine = vxyz[:, 2] / spz - 1
    near = np.array(have)[np.abs(real_fine[None, :] - fine[:, None]).argmin(1)]
    cols = np.clip((vxyz[:, 0] / VXY).round().astype(int), 0, GRID - 1)
    rows = np.clip((vxyz[:, 1] / VXY).round().astype(int), 0, GRID - 1)
    col = np.zeros((len(vxyz), 3), np.uint8)
    for zz in np.unique(near):
        sel = near == zz
        col[sel] = rgb[zz][rows[sel], cols[sel]][:, ::-1]   # BGR->RGB
    lo, hi = vxyz.min(0), vxyz.max(0)
    scale = (hi - lo) / 65535.0
    q = np.round((vxyz - lo) / np.where(scale == 0, 1, scale)).astype(np.uint16)
    out_meshes[c] = dict(v=b64(q), f=b64(faces.astype(np.uint32)),
                         lo=lo.tolist(), scale=scale.tolist(), nv=len(vxyz),
                         c=b64(col))
    print(c, len(vxyz), "verts +colors")

# ---- slice textures: crop to ball bbox + margin, feathered alpha ----
ys, xs = np.where(stacks["glom"].any(0))
M = 20
r0, r1 = max(ys.min() - M, 0), min(ys.max() + M, GRID)
c0, c1 = max(xs.min() - M, 0), min(xs.max() + M, GRID)
slice_items = []
for z in have:
    a = cv2.dilate(stacks["glom"][z], np.ones((9, 9), np.uint8)).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 3)
    tile = rgb[z][r0:r1, c0:c1]
    alpha = (np.clip(a[r0:r1, c0:c1], 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([tile, alpha])
    fn = f"sec{z + z0:02d}.webp"
    cv2.imwrite(str(TEXDIR / fn), rgba, [cv2.IMWRITE_WEBP_QUALITY, 88])
    slice_items.append(dict(f=fn, z=round(z * ZUM + spz, 3)))
print("textures:", len(slice_items), "->", TEXDIR.name)

out = dict(meta=dict(mouse="22-184", glom=f"g{G}", sections=[z0, z1], z_um=ZUM,
                     voxel_xy_um=VXY, version="v4-textured"),
           meshes=out_meshes,
           cells=dict(p=b64(cells[:, [1, 2, 0]]), r=b64(cells[:, 3]), n=len(cells)),
           slices=dict(dir=f"tex_g{G}", x0=round(c0 * VXY, 3), y0=round(r0 * VXY, 3),
                       w=round((c1 - c0) * VXY, 3), h=round((r1 - r0) * VXY, 3),
                       items=slice_items))
(REPO / f"glom_22-184_g{G}.json").write_text(json.dumps(out))
print("->", REPO / f"glom_22-184_g{G}.json")
