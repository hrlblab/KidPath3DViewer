"""Render docs/demo.gif from demo/synthetic.json: the model rotates while a
cut plane sweeps along X, then Y, then Z (the viewer's X/Y/Z cut feature).

Usage: python docs/make_demo_gif.py
Deps : numpy matplotlib pillow
"""
import base64
import io
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "demo", "synthetic.json")
OUT = os.path.join(HERE, "demo.gif")
j = json.load(open(SRC))

CLASSES = (("mesangial", (0xd8/255, 0x65/255, 0xc8/255), 0.9),
           ("vessels",   (0x4f/255, 0xbd/255, 0x82/255), 1.0),
           ("glom",      (0xc9/255, 0xa1/255, 0x5a/255), 0.13))
LIGHT = np.array([0.5, 0.5, 1.0]) / np.linalg.norm([0.5, 0.5, 1.0])
BG = "#101114"

meshes = {}
for key, col, alpha in CLASSES:
    m = j["meshes"][key]
    v = np.frombuffer(base64.b64decode(m["v"]), "<f4").reshape(-1, 3)
    f = np.frombuffer(base64.b64decode(m["f"]), "<u4").reshape(-1, 3)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    shade = (0.35 + 0.65 * np.clip(n @ LIGHT, 0, 1))[:, None]
    cols = np.c_[np.clip(np.array(col)[None] * shade, 0, 1),
                 np.full(len(tri), alpha)]
    meshes[key] = (tri, tri.mean(1), cols)          # tris, centroids, rgba
cells = np.frombuffer(base64.b64decode(j["cells"]["p"]), "<f4").reshape(-1, 3)

allv = np.vstack([m[0].reshape(-1, 3) for m in meshes.values()])
lo, hi = float(allv.min()), float(allv.max())

# phases: (axis index or None, label); each sweep goes full->cut->full
FULL, SWEEP = 8, 16
phases = [(None, "")] + [(ax, f"{'XYZ'[ax]} cut" ) for ax in range(3)]
frames = []
nf = FULL + 3 * SWEEP
azims = np.linspace(-55, 305, nf, endpoint=False)

fi = 0
for axis, label in phases:
    steps = FULL if axis is None else SWEEP
    for k in range(steps):
        if axis is None:
            cut, frac = None, 0.0
        else:
            frac = np.sin(np.pi * k / (steps - 1))    # 0 -> 1 -> 0
            cut = hi - 0.62 * frac * (hi - lo)
        fig = plt.figure(figsize=(6.4, 6.0), facecolor=BG)
        ax3 = fig.add_subplot(111, projection="3d", facecolor=BG)
        for key, _c, _a in CLASSES:
            tri, cen, cols = meshes[key]
            keep = slice(None) if cut is None else cen[:, axis] <= cut
            ax3.add_collection3d(Poly3DCollection(tri[keep],
                                 facecolors=cols[keep], edgecolor="none"))
        cp = cells if cut is None else cells[cells[:, axis] <= cut]
        ax3.scatter(cp[:, 0], cp[:, 1], cp[:, 2], s=5, c="#f0c93c",
                    alpha=0.9, depthshade=False)
        ax3.set_xlim(lo, hi); ax3.set_ylim(lo, hi); ax3.set_zlim(lo, hi)
        ax3.set_box_aspect((1, 1, 1))
        ax3.view_init(elev=18, azim=float(azims[fi]))
        ax3.set_axis_off()
        if label:
            fig.text(0.05, 0.92, label, color="#9aa0ae", fontsize=14,
                     family="monospace")
        plt.subplots_adjust(left=-0.18, right=1.18, top=1.18, bottom=-0.18)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, facecolor=BG)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        fi += 1
        print(f"frame {fi}/{nf}", flush=True)

frames = [f.quantize(colors=128, method=Image.MEDIANCUT) for f in frames]
frames[0].save(OUT, save_all=True, append_images=frames[1:], loop=0,
               duration=110, optimize=True)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB, {len(frames)} frames)")
