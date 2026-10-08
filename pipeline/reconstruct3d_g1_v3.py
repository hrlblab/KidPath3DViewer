"""v3: compare three anti-stair-step strategies and export the SDF-interpolated model.

  A  v2 baseline: binary stack @3um, gauss sigma_z=0.9
  B  stronger z smoothing: gauss sigma_z=1.8
  C  SDF shape interpolation: z upsampled 3um -> 1um by linear interpolation of
     per-slice signed distance fields (near-isotropic voxels), mild isotropic gauss

Outputs: recon3d/cmp_xz_sections.jpg (orthogonal xz cut through the centre for
A/B/C) and meshes_g1.json rebuilt from variant C.
"""
from __future__ import annotations

import base64
import json
import math

import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

from reconstruct3d_g1 import (CANVAS, DS, MPP, OFF, OUTDIR, ZUM, b64, compose,
                              gray_patch, load_sections, poly_area, rasterise, sdf)
from reconstruct3d_g1_v2 import local_ecc
from scipy.signal import savgol_filter

GRID = CANVAS // DS          # 400 px @ ~0.97 um
VXY = MPP * DS


def build_small_stacks(g=1):
    """Redo v2 registration, return per-class binary stacks @GRID + cells."""
    secs = load_sections(g)
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
    smooth = np.column_stack([savgol_filter(cents[:, k], win, 2) for k in (0, 1)])
    for i, s in enumerate(order):
        d = cents[i] - smooth[i]
        warps[s] = compose(np.array([[1, 0, -d[0]], [0, 1, -d[1]]], np.float32), warps[s])

    z0, z1 = order[0], order[-1]
    nz = z1 - z0 + 1
    classes = ["glom", "vessels", "mesangial"]
    stacks = {c: np.zeros((nz, GRID, GRID), np.uint8) for c in classes}
    cells = []
    for s in order:
        info, W = secs[s], warps[s]
        z = s - z0
        for c, polys in [("glom", [info["host"]]),
                         ("vessels", info["inner"]["vessels"]),
                         ("mesangial", info["inner"]["mesangial"])]:
            m = rasterise(polys, info["origin"], W).astype(np.float32)
            stacks[c][z] = (cv2.resize(m, (GRID, GRID), interpolation=cv2.INTER_AREA) > 0.5)
        W3 = np.vstack([W, [0, 0, 1]])
        for p in info["inner"]["cell"]:
            cc = p.mean(0) - np.array(info["origin"])
            cx, cy, _ = W3 @ np.array([cc[0], cc[1], 1.0])
            cells.append(((s - z0) * ZUM, cx * MPP, cy * MPP,
                          max(math.sqrt(poly_area(p) / math.pi) * MPP, 1.0)))
    have = sorted(s - z0 for s in order)
    for c in classes:
        v = stacks[c]
        for z in range(nz):
            if z in have:
                continue
            lo = max(h for h in have if h < z)
            hi = min(h for h in have if h > z)
            t = (z - lo) / (hi - lo)
            v[z] = ((1 - t) * sdf(v[lo]) + t * sdf(v[hi]) > 0).astype(np.uint8)
    return stacks, np.array(cells, np.float32), (z0, z1)


def sdf_upsample_z(stack, f=3):
    """Binary (nz,h,w) -> (nz*f-f+1,h,w) via per-pair SDF interpolation."""
    nz = stack.shape[0]
    sd = np.stack([sdf(stack[z]) for z in range(nz)])
    out = np.zeros(((nz - 1) * f + 1,) + stack.shape[1:], np.uint8)
    for z in range(nz - 1):
        for k in range(f):
            t = k / f
            out[z * f + k] = ((1 - t) * sd[z] + t * sd[z + 1]) > 0
    out[-1] = stack[-1]
    return out


def field(stack, sig, up=1):
    pad = np.zeros((stack.shape[0] + 2,) + stack.shape[1:], np.float32)
    pad[1:-1] = stack
    return ndimage.gaussian_filter(pad, sig)


def xz_cut(fields, spacing_z):
    """fields: {cls: float field}; render xz plane through centre, 1um/px."""
    cols = {"glom": (90, 90, 230), "vessels": (90, 200, 90), "mesangial": (210, 90, 210)}
    g = fields["glom"]
    yc = int(np.argmax((g > 0.5).sum((0, 2))))
    H = int(round((g.shape[0] - 1) * spacing_z))
    img = np.full((H, 420, 3), 255, np.uint8)
    for cls in ["glom", "vessels", "mesangial"]:
        m = (fields[cls][:, yc, :] > 0.5).astype(np.uint8)
        m = cv2.resize(m * 255, (420, H), interpolation=cv2.INTER_NEAREST)
        sel = m > 127
        col = np.array(cols[cls])
        img[sel] = (0.45 * img[sel] + 0.55 * col).astype(np.uint8)
    return img


def main():
    stacks, cells, (z0, z1) = build_small_stacks()

    variants = {}
    variants["A  gauss z=0.9 (v2)"] = ({c: field(stacks[c], (0.9, 1.3, 1.3)) for c in stacks}, ZUM)
    variants["B  gauss z=1.8"] = ({c: field(stacks[c], (1.8, 1.3, 1.3)) for c in stacks}, ZUM)
    up = {c: sdf_upsample_z(stacks[c], 3) for c in stacks}
    variants["C  SDF interp 3um->1um"] = ({c: field(up[c], (1.0, 1.0, 1.0)) for c in up}, ZUM / 3)

    tiles = []
    for name, (flds, spz) in variants.items():
        im = xz_cut(flds, spz)
        im = cv2.copyMakeBorder(im, 36, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(im, name, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
        tiles.append(im)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                value=(255, 255, 255)) for t in tiles]
    cv2.imwrite(str(OUTDIR / "cmp_xz_sections.jpg"), np.hstack(tiles))
    print("xz comparison saved")

    # export variant C
    out = {"meta": dict(mouse="22-184", glom="g1", sections=[z0, z1], z_um=ZUM,
                        voxel_xy_um=VXY, version="v3-sdf-interp-1um"), "meshes": {}}
    flds, spz = variants["C  SDF interp 3um->1um"]
    for c, f in flds.items():
        if f.max() < 0.5:
            continue
        verts, faces, normals, _ = marching_cubes(f, 0.5, spacing=(spz, VXY, VXY))
        out["meshes"][c] = dict(v=b64(verts[:, [2, 1, 0]].astype(np.float32)),
                                n=b64((-normals[:, [2, 1, 0]]).astype(np.float32)),
                                f=b64(faces.astype(np.uint32)),
                                nv=len(verts), nf=len(faces))
        print(c, len(verts), "verts", len(faces), "faces")
    out["cells"] = dict(p=b64(cells[:, [1, 2, 0]]), r=b64(cells[:, 3]), n=len(cells))
    json.dump(out, open(OUTDIR / "meshes_g1.json", "w"))
    print("meshes_g1.json <- variant C")


if __name__ == "__main__":
    main()
