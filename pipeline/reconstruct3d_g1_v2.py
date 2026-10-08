"""v2 reconstruction: ball-centred local ECC + trajectory smoothing.

Fixes the residual layer-to-layer jitter of v1 (whole-field ECC):
  A. register each consecutive pair on a 600px window centred on the BALL
     (euclidean: translation+rotation), so the ball itself is what aligns;
  B. chain into the first section's frame;
  C. smooth the per-slice ball-centroid trajectory (Savitzky-Golay) and apply
     the residual as a per-slice shift: high-frequency jitter removed, the real
     low-frequency anatomical tilt kept.
Outputs overwrite recon3d/meshes_g1.json + new QC images; prints jitter RMS.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from skimage.measure import marching_cubes

from reconstruct3d_g1 import (CANVAS, DS, MPP, OFF, OUTDIR, S, ZUM,
                              b64, compose, gray_patch, load_sections,
                              poly_area, rasterise, sdf)

WIN = 600  # local ECC window (px) ~146 um


def local_ecc(prev_g, cur_g, p_c, c_c):
    """Euclidean warp cur->prev patch coords, estimated on ball-centred crops."""
    def crop(img, c):
        x0 = int(min(max(c[0] - WIN // 2, 0), S - WIN))
        y0 = int(min(max(c[1] - WIN // 2, 0), S - WIN))
        return img[y0:y0 + WIN, x0:x0 + WIN], np.array([x0, y0], float)
    a, p0 = crop(prev_g, p_c)
    b, c0 = crop(cur_g, c_c)
    W = np.eye(2, 3, dtype=np.float32)
    ok = True
    try:
        _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_EUCLIDEAN,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6))
    except cv2.error:
        ok = False
    # full-patch warp: x_prev = p0 + W(x_cur - c0)
    T1 = np.array([[1, 0, p0[0]], [0, 1, p0[1]]], np.float32)
    T2 = np.array([[1, 0, -c0[0]], [0, 1, -c0[1]]], np.float32)
    return compose(T1, compose(W, T2)), ok


def main():
    secs = load_sections()
    order = sorted(secs)
    slides = {}

    # A+B: ball-centred chained registration
    warps = {}
    base = np.array([[1, 0, OFF], [0, 1, OFF]], np.float32)
    fails = []
    for i, s in enumerate(order):
        info = secs[s]
        if i == 0:
            warps[s] = base
            continue
        prev = order[i - 1]
        pinfo = secs[prev]
        p_c = pinfo["host"].mean(0) - np.array(pinfo["origin"])
        c_c = info["host"].mean(0) - np.array(info["origin"])
        W, ok = local_ecc(gray_patch(slides, pinfo), gray_patch(slides, info), p_c, c_c)
        if not ok:
            # fallback: pure centroid translation
            W = np.array([[1, 0, p_c[0] - c_c[0]], [0, 1, p_c[1] - c_c[1]]], np.float32)
            fails.append((prev, s))
        warps[s] = compose(warps[prev], W)
    print("local-ECC fallbacks:", fails or "none")

    # C: trajectory smoothing on registered ball centroids
    cents = []
    for s in order:
        info = secs[s]
        W3 = np.vstack([warps[s], [0, 0, 1]])
        c = info["host"].mean(0) - np.array(info["origin"])
        cents.append((W3 @ [c[0], c[1], 1.0])[:2])
    cents = np.array(cents)
    win = min(9, len(order) - (1 - len(order) % 2))
    smooth = np.column_stack([savgol_filter(cents[:, k], win, 2) for k in (0, 1)])
    resid = cents - smooth
    print(f"jitter RMS before smoothing: {np.sqrt((resid**2).sum(1)).mean()*MPP:.1f} um "
          f"(max {np.sqrt((resid**2).sum(1)).max()*MPP:.1f} um)")
    for i, s in enumerate(order):
        T = np.array([[1, 0, -resid[i, 0]], [0, 1, -resid[i, 1]]], np.float32)
        warps[s] = compose(T, warps[s])

    # rasterise + volume (same as v1)
    z0, z1 = order[0], order[-1]
    nz = z1 - z0 + 1
    classes = ["glom", "vessels", "mesangial"]
    vols = {c: np.zeros((nz, CANVAS, CANVAS), np.uint8) for c in classes}
    cells = []
    for s in order:
        info, W = secs[s], warps[s]
        z = s - z0
        vols["glom"][z] = rasterise([info["host"]], info["origin"], W)
        vols["vessels"][z] = rasterise(info["inner"]["vessels"], info["origin"], W)
        vols["mesangial"][z] = rasterise(info["inner"]["mesangial"], info["origin"], W)
        W3 = np.vstack([W, [0, 0, 1]])
        for p in info["inner"]["cell"]:
            c = p.mean(0) - np.array(info["origin"])
            cx, cy, _ = W3 @ np.array([c[0], c[1], 1.0])
            r = math.sqrt(poly_area(p) / math.pi) * MPP
            cells.append((z * ZUM, cx * MPP, cy * MPP, max(r, 1.0)))

    have = set(s - z0 for s in order)
    for c in classes:
        v = vols[c]
        for z in range(nz):
            if z in have:
                continue
            lo = max(h for h in have if h < z)
            hi = min(h for h in have if h > z)
            t = (z - lo) / (hi - lo)
            v[z] = ((1 - t) * sdf(v[lo]) + t * sdf(v[hi]) > 0).astype(np.uint8)

    qc = np.full((CANVAS, CANVAS, 3), 255, np.uint8)
    for i, s in enumerate(order):
        col = tuple(int(x) for x in cv2.applyColorMap(
            np.array([[int(255 * i / (len(order) - 1))]], np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0])
        cnts, _ = cv2.findContours(vols["glom"][s - z0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(qc, cnts, -1, col, 2)
    cv2.imwrite(str(OUTDIR / "qc_registration_g1_v2.jpg"), qc)

    out = {"meta": dict(mouse="22-184", glom="g1", sections=[z0, z1],
                        z_um=ZUM, voxel_xy_um=MPP * DS, version="v2-local-ecc+smooth"),
           "meshes": {}}
    for c in classes:
        small = np.stack([cv2.resize(sl.astype(np.float32), (CANVAS // DS, CANVAS // DS),
                                     interpolation=cv2.INTER_AREA) for sl in vols[c]])
        pad = np.zeros((small.shape[0] + 2, small.shape[1], small.shape[2]), np.float32)
        pad[1:-1] = small
        sm = ndimage.gaussian_filter(pad, (0.9, 1.3, 1.3))
        if sm.max() < 0.5:
            continue
        verts, faces, normals, _ = marching_cubes(sm, 0.5, spacing=(ZUM, MPP * DS, MPP * DS))
        out["meshes"][c] = dict(v=b64(verts[:, [2, 1, 0]].astype(np.float32)),
                                n=b64((-normals[:, [2, 1, 0]]).astype(np.float32)),
                                f=b64(faces.astype(np.uint32)),
                                nv=len(verts), nf=len(faces))
        print(c, len(verts), "verts")
    cells = np.array(cells, np.float32)
    out["cells"] = dict(p=b64(cells[:, [1, 2, 0]]), r=b64(cells[:, 3]), n=len(cells))
    json.dump(out, open(OUTDIR / "meshes_g1.json", "w"))
    print("done ->", OUTDIR / "meshes_g1.json")


if __name__ == "__main__":
    main()
