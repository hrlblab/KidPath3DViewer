"""Reconstruct any 22-184 trail with the final (v3) recipe:
local-ECC rigid registration + trajectory smoothing + SDF z-interp 3um->1um.
Usage: python reconstruct3d_ball.py <glom-number>
Writes recon3d/meshes_g{N}.json + qc_registration_g{N}_v3.jpg
"""
import json
import sys

import cv2
import numpy as np
from skimage.measure import marching_cubes

from reconstruct3d_g1 import MPP, OUTDIR, ZUM, b64
from reconstruct3d_g1_v3 import VXY, build_small_stacks, field, sdf_upsample_z

G = int(sys.argv[1])
stacks, cells, (z0, z1) = build_small_stacks(G)
print(f"g{G}: sections {z0}-{z1}")

qc = np.full(stacks["glom"].shape[1:] + (3,), 255, np.uint8)
nz = stacks["glom"].shape[0]
for z in range(nz):
    col = tuple(int(x) for x in cv2.applyColorMap(
        np.array([[int(255 * z / (nz - 1))]], np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0])
    cnts, _ = cv2.findContours(stacks["glom"][z], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(qc, cnts, -1, col, 1)
cv2.imwrite(str(OUTDIR / f"qc_registration_g{G}_v3.jpg"),
            cv2.resize(qc, (800, 800), interpolation=cv2.INTER_NEAREST))

up = {c: sdf_upsample_z(stacks[c], 3) for c in stacks}
flds = {c: field(up[c], (1.0, 1.0, 1.0)) for c in up}
out = {"meta": dict(mouse="22-184", glom=f"g{G}", sections=[z0, z1], z_um=ZUM,
                    voxel_xy_um=VXY, version="v3-sdf-interp-1um"), "meshes": {}}
for c, f in flds.items():
    if f.max() < 0.5:
        continue
    verts, faces, normals, _ = marching_cubes(f, 0.5, spacing=(ZUM / 3, VXY, VXY))
    out["meshes"][c] = dict(v=b64(verts[:, [2, 1, 0]].astype(np.float32)),
                            n=b64((-normals[:, [2, 1, 0]]).astype(np.float32)),
                            f=b64(faces.astype(np.uint32)), nv=len(verts), nf=len(faces))
    print(c, len(verts), "verts")
out["cells"] = dict(p=b64(cells[:, [1, 2, 0]]), r=b64(cells[:, 3]), n=len(cells))
print("cells:", len(cells))
json.dump(out, open(OUTDIR / f"meshes_g{G}.json", "w"))
print("->", OUTDIR / f"meshes_g{G}.json")
