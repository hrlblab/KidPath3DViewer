"""Pixel-level z interpolation of the registered H&E slices (flow morphing).

For each pair of consecutive real sections: dense bidirectional optical flow
(OpenCV DIS) -> intermediate frames every 1 um by flow-guided warping + blend
(video-frame-interpolation style; no ghosting, unlike cross-fade).
Alpha comes from the SDF-upsampled glom mask, so texture and geometry agree.

Usage: python pixel_interp_textures.py <glom-number>
Rewrites glom3d-viewer/data/tex_g{N}/ (adds sec*i*.webp) and updates the
dataset JSON's slices.items. Also writes recon3d/cmp_pixel_interp_g{N}.jpg
(xz cut: nearest-repeat vs flow-interpolated).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import openslide
from scipy.signal import savgol_filter

from reconstruct3d_g1 import (DS, MPP, OFF, OUTDIR, S, ZUM, compose,
                              gray_patch, load_sections, rasterise, sdf)
from reconstruct3d_g1_v2 import local_ecc
from reconstruct3d_g1_v3 import GRID, VXY, sdf_upsample_z

G = int(sys.argv[1])
REPO = Path("/data3/zhuy39/GLomSeg/glom3d-viewer/data")
TEXDIR = REPO / f"tex_g{G}"
NEW = Path("/data3/zhuy39/GLomSeg/new data/22-184")

# ---- registration (identical to reconstruct3d_textured) ----
secs = load_sections(G)
order = sorted(secs)
slides, warps = {}, {}
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
sm = np.column_stack([savgol_filter(cents[:, k], win, 2) for k in (0, 1)])
for i, s in enumerate(order):
    d = cents[i] - sm[i]
    warps[s] = compose(np.array([[1, 0, -d[0]], [0, 1, -d[1]]], np.float32), warps[s])

z0 = order[0]
SC = np.array([[1 / DS, 0, 0], [0, 1 / DS, 0]], np.float32)
rgb, gmask = {}, {}
for s in order:
    info, W = secs[s], warps[s]
    z = s - z0
    sl = info["slide"]
    if sl not in slides:
        slides[sl] = openslide.OpenSlide(str(NEW / sl))
    img = np.array(slides[sl].read_region(info["origin"], 0, (S, S)).convert("RGB"))
    rgb[z] = cv2.warpAffine(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), compose(SC, W),
                            (GRID, GRID), flags=cv2.INTER_AREA, borderValue=(245, 244, 246))
    m = rasterise([info["host"]], info["origin"], W).astype(np.float32)
    gmask[z] = (cv2.resize(m, (GRID, GRID), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
have = sorted(rgb)

# gap-fill glom masks, then SDF-upsample x3 -> fine alpha source
nz = have[-1] + 1
gstack = np.zeros((nz, GRID, GRID), np.uint8)
for z in have:
    gstack[z] = gmask[z]
for z in range(nz):
    if z in have:
        continue
    lo = max(h for h in have if h < z)
    hi = min(h for h in have if h > z)
    t = (z - lo) / (hi - lo)
    gstack[z] = ((1 - t) * sdf(gstack[lo]) + t * sdf(gstack[hi]) > 0).astype(np.uint8)
gfine = sdf_upsample_z(gstack, 3)          # index f = z*3

# crop region (identical formula to reconstruct3d_textured)
ys, xs = np.where(gstack.any(0))
M = 20
r0, r1 = max(ys.min() - M, 0), min(ys.max() + M, GRID)
c0, c1 = max(xs.min() - M, 0), min(xs.max() + M, GRID)

dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
yy, xx = np.mgrid[0:GRID, 0:GRID].astype(np.float32)


def warp(img, flow, t):
    return cv2.remap(img, xx + t * flow[..., 0], yy + t * flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def alpha_of(fidx):
    a = cv2.dilate(gfine[min(fidx, gfine.shape[0] - 1)], np.ones((9, 9), np.uint8)).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 3)
    return (np.clip(a[r0:r1, c0:c1], 0, 1) * 255).astype(np.uint8)


def save(tile, alpha, fn):
    cv2.imwrite(str(TEXDIR / fn), np.dstack([tile, alpha]), [cv2.IMWRITE_WEBP_QUALITY, 88])


items = []
spz = ZUM / 3
for z in have:
    items.append(dict(f=f"sec{z + z0:02d}.webp", z=round(z * ZUM + spz, 3)))

fine_rgb = {z * 3: rgb[z] for z in have}    # for the comparison cut
n_interp = 0
for a, b in zip(have, have[1:]):
    ga = cv2.cvtColor(rgb[a], cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(rgb[b], cv2.COLOR_BGR2GRAY)
    fab = dis.calc(ga, gb, None)
    fba = dis.calc(gb, ga, None)
    n = (b - a) * 3
    for k in range(1, n):
        t = k / n
        I = ((1 - t) * warp(rgb[a], fab, t).astype(np.float32)
             + t * warp(rgb[b], fba, 1 - t).astype(np.float32)).astype(np.uint8)
        fidx = a * 3 + k
        fn = f"sec{a + z0:02d}i{k}.webp"
        save(I[r0:r1, c0:c1], alpha_of(fidx), fn)
        items.append(dict(f=fn, z=round(fidx * spz + spz, 3), i=1))
        fine_rgb[fidx] = I
        n_interp += 1
for z in have:   # rewrite real ones too (same crop)
    save(rgb[z][r0:r1, c0:c1], alpha_of(z * 3), f"sec{z + z0:02d}.webp")
items.sort(key=lambda d: d["z"])
print(f"g{G}: {len(have)} real + {n_interp} interpolated frames")

# ---- RGBA voxel atlas for volume rendering (1 um near-isotropic) ----
maxf0 = max(fine_rgb)
nzf = maxf0 + 1
tw, th = c1 - c0, r1 - r0
cols = int(np.ceil(np.sqrt(nzf)))
rows = int(np.ceil(nzf / cols))
atlas = np.zeros((rows * th, cols * tw, 4), np.uint8)
for f in range(nzf):
    ri, ci = divmod(f, cols)
    atlas[ri * th:(ri + 1) * th, ci * tw:(ci + 1) * tw] = np.dstack(
        [fine_rgb[f][r0:r1, c0:c1], alpha_of(f)])
cv2.imwrite(str(REPO / f"vol_g{G}.webp"), atlas, [cv2.IMWRITE_WEBP_QUALITY, 90])
vol_meta = dict(atlas=f"vol_g{G}.webp", nx=int(tw), ny=int(th), nz=int(nzf), cols=int(cols),
                x0=round(c0 * VXY, 3), y0=round(r0 * VXY, 3),
                w=round(tw * VXY, 3), h=round(th * VXY, 3),
                z0=round(spz, 3), dz=round(spz, 3))
print("volume atlas:", atlas.shape, "->", f"vol_g{G}.webp")

# update dataset json
jp = REPO / f"glom_22-184_g{G}.json"
data = json.loads(jp.read_text())
data["volume"] = vol_meta
data["slices"]["items"] = items
data["slices"]["x0"] = round(c0 * VXY, 3)
data["slices"]["y0"] = round(r0 * VXY, 3)
data["slices"]["w"] = round((c1 - c0) * VXY, 3)
data["slices"]["h"] = round((r1 - r0) * VXY, 3)
jp.write_text(json.dumps(data))

# xz comparison: nearest-repeat vs flow-interp
maxf = max(fine_rgb)
rowc = (r0 + r1) // 2
A = np.zeros((maxf + 1, c1 - c0, 3), np.uint8)   # nearest repeat
B = np.zeros_like(A)                              # flow interp
harr = np.array(have)
for f in range(maxf + 1):
    near = have[int(np.abs(harr * 3 - f).argmin())]
    A[f] = rgb[near][rowc, c0:c1]
    src = fine_rgb.get(f)
    B[f] = (src if src is not None else rgb[near])[rowc, c0:c1]
sc = 4
tiles = []
for img, t in [(A, "nearest repeat (before)"), (B, "flow-morph interp (after)")]:
    im = cv2.resize(img, ((c1 - c0) * sc, (maxf + 1) * sc), interpolation=cv2.INTER_NEAREST)
    im = cv2.copyMakeBorder(im, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(im, t, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    tiles.append(im)
cv2.imwrite(str(OUTDIR / f"cmp_pixel_interp_g{G}.jpg"), np.hstack(tiles))
print("cmp ->", OUTDIR / f"cmp_pixel_interp_g{G}.jpg")
