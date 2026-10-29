"""Generate demo/synthetic.json — a procedurally generated, glomerulus-like
demo dataset for KidPath3DViewer. Pure geometry (lumpy ball + winding tubes +
blobs + random cells); contains NO tissue-derived data, safe to publish.

Usage: python demo/make_synthetic.py
Deps : numpy scipy scikit-image
"""
import base64
import json
import os

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

rng = np.random.default_rng(7)
N, SP = 96, 1.75                      # grid size, um per voxel -> 168 um cube
c = N * SP / 2
zz, yy, xx = (np.mgrid[0:N, 0:N, 0:N].astype(np.float32) * SP)
r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2)

# ---- capsule: radius-60um ball with smooth lumps ----
noise = ndi.gaussian_filter(rng.standard_normal((N, N, N)).astype(np.float32), 7)
noise *= 8 / np.abs(noise).max()
ball = (60 + noise) - r               # >0 inside

# ---- capillary-like tubes: winding closed curves inside the ball ----
occ = np.zeros((N, N, N), bool)
for _ in range(7):
    t = np.linspace(0, 2 * np.pi, 600, dtype=np.float32)
    p = []
    for _ax in range(3):
        a = rng.uniform(0, 2 * np.pi, 3)
        w = rng.integers(1, 4, 3)
        amp = rng.uniform(18, 42, 3)
        amp *= 44 / amp.sum()
        p.append(c + sum(A * np.sin(k * t + ph) for A, k, ph in zip(amp, w, a)))
    px, py, pz = p
    keep = np.sqrt((px - c) ** 2 + (py - c) ** 2 + (pz - c) ** 2) < 50
    ii = (np.round(np.c_[pz, py, px][keep] / SP).astype(int)).clip(0, N - 1)
    occ[ii[:, 0], ii[:, 1], ii[:, 2]] = True
tube = (ndi.distance_transform_edt(~occ, sampling=SP) < 5.5) & (ball > 4)
tube_f = ndi.gaussian_filter(tube.astype(np.float32), 1.2)

# ---- mesangial-like blobs between tubes ----
blob = np.zeros((N, N, N), np.float32)
for _ in range(9):
    while True:
        q = rng.uniform(c - 40, c + 40, 3)
        if np.linalg.norm(q - c) < 42:
            break
    d2 = (xx - q[0]) ** 2 + (yy - q[1]) ** 2 + (zz - q[2]) ** 2
    blob += np.exp(-d2 / (2 * rng.uniform(6, 10) ** 2)).astype(np.float32)
blob *= (ball > 6)

# ---- cells: random points inside ----
pts = []
while len(pts) < 350:
    q = rng.uniform(c - 55, c + 55, 3)
    if np.linalg.norm(q - c) < 52:
        pts.append(q)
pts = np.array(pts, np.float32)
rad = rng.uniform(2.0, 3.6, len(pts)).astype(np.float32)


def mesh(vol, level):
    v, f, *_ = measure.marching_cubes(vol, level, spacing=(SP, SP, SP))
    v = v[:, ::-1].copy()             # (z,y,x) -> (x,y,z)
    return {"v": b64(v.astype("<f4")), "f": b64(f.astype("<u4"))}


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


out = {
    "meta": {"z_um": 3.0, "sections": [1, 20],
             "note": "synthetic demo — procedurally generated, no tissue data"},
    "meshes": {"glom": mesh(ball, 0.0),
               "vessels": mesh(tube_f, 0.5),
               "mesangial": mesh(blob, 0.55)},
    "cells": {"p": b64(pts.astype("<f4")), "r": b64(rad), "n": int(len(pts))},
}
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic.json")
with open(dst, "w") as fh:
    json.dump(out, fh)
print(f"wrote {dst}  ({os.path.getsize(dst)/1e6:.1f} MB)")
