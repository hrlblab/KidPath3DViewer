"""3D reconstruction of 22-184 glom-1 from the SAM-refined patch GeoJSONs.

Pipeline (confirmed with user: z = 3 um/section, pilot = g1):
  1. per section, take the glom/tuft polygon CONTAINING the marker (+ vessels/
     mesangial/cell instances whose centroid lies inside it),
  2. register consecutive sections: centroid init + ECC euclidean refine on the
     raw grayscale patches, chain the transforms into the first section's frame,
  3. rasterise per-class masks into a common 1600^2 canvas, fill missing
     sections by signed-distance interpolation,
  4. downsample to ~1 um/px, gaussian-smooth, marching cubes per class
     (spacing z=3um), cells exported as centroid spheres,
  5. write meshes_g1.json (base64 float32/uint32) + registration QC image.
"""
from __future__ import annotations

import base64
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import openslide
from scipy import ndimage
from skimage.measure import marching_cubes

MPP = 0.24305
ZUM = 3.0
S = 1200
CANVAS = 1600
OFF = (CANVAS - S) // 2
DS = 4                       # xy downsample -> 0.972 um/px
GJ = Path("/data3/zhuy39/GLomSeg/new_data_segmentation/_review/raw_patches_22-184/_detail_geojson_sam")
NEW = Path("/data3/zhuy39/GLomSeg/new data/22-184")
OUTDIR = Path("/data3/zhuy39/GLomSeg/new_data_segmentation/_review/recon3d")
OUTDIR.mkdir(exist_ok=True)


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def load_sections(g=1):
    rows = [r for r in csv.DictReader(open(
        "/data3/zhuy39/GLomSeg/new_data_segmentation/_review/detail_seg_sam_manifest.csv"))
        if r["mouse"] == "22-184" and r["glom"] == str(g)]
    secs = {}
    for r in rows:
        stem = f"g{g}__{Path(r['slide']).stem}__s{int(r['snn']):02d}"
        gj = json.load(open(GJ / f"{stem}.geojson"))
        mx, my = gj["properties"]["marker_xy"]
        ox, oy = gj["properties"]["patch_origin"]
        host, ha = None, 0
        feats = []
        for f in gj["features"]:
            cls = f["properties"]["classification"]["name"]
            poly = np.array(f["geometry"]["coordinates"][0], float)
            feats.append((cls, poly))
            if cls in ("glom", "tuft"):
                if cv2.pointPolygonTest(poly.astype(np.float32), (float(mx), float(my)), False) >= 0:
                    a = poly_area(poly)
                    if a > ha:
                        host, ha = poly, a
        if host is None:
            continue
        inner = {"vessels": [], "mesangial": [], "cell": []}
        hostf = host.astype(np.float32)
        for cls, poly in feats:
            if cls not in inner:
                continue
            c = poly.mean(0)
            if cv2.pointPolygonTest(hostf, (float(c[0]), float(c[1])), False) >= 0:
                inner[cls].append(poly)
        secs[int(r["section"])] = dict(host=host, inner=inner, origin=(ox, oy),
                                       slide=r["slide"], marker=(mx, my))
    return secs


def gray_patch(slides, info):
    sl = info["slide"]
    if sl not in slides:
        slides[sl] = openslide.OpenSlide(str(NEW / sl))
    ox, oy = info["origin"]
    img = np.array(slides[sl].read_region((ox, oy), 0, (S, S)).convert("L"))
    return cv2.GaussianBlur(img, (0, 0), 2)


def ecc_pair(prev_g, cur_g, init_dx, init_dy):
    """Euclidean warp mapping cur -> prev patch coords (at full res)."""
    sc = 2
    a = cv2.resize(prev_g, (S // sc, S // sc))
    b = cv2.resize(cur_g, (S // sc, S // sc))
    W = np.array([[1, 0, init_dx / sc], [0, 1, init_dy / sc]], np.float32)
    try:
        _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_EUCLIDEAN,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5))
    except cv2.error:
        pass
    W = W.copy()
    W[:, 2] *= sc
    return W


def compose(A, B):
    """2x3 affines: result applies B then A."""
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2]


def rasterise(polys, origin, warp):
    m = np.zeros((S, S), np.uint8)
    for p in polys:
        q = (p - np.array(origin)).astype(np.int32)
        cv2.fillPoly(m, [q], 1)
    return cv2.warpAffine(m, warp, (CANVAS, CANVAS), flags=cv2.INTER_NEAREST)


def sdf(m):
    inside = ndimage.distance_transform_edt(m)
    outside = ndimage.distance_transform_edt(1 - m)
    return inside - outside


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def main():
    secs = load_sections()
    order = sorted(secs)
    print("sections with g1 ball:", order)
    slides = {}

    # chain registration
    warps = {}      # section -> 2x3 mapping patch coords -> canvas frame
    base = np.array([[1, 0, OFF], [0, 1, OFF]], np.float32)
    prev_sec = None
    for s in order:
        info = secs[s]
        if prev_sec is None:
            warps[s] = base
        else:
            pinfo = secs[prev_sec]
            # ECC wants warp cur->prev; init from host centroids (patch-local)
            pc = pinfo["host"].mean(0) - np.array(pinfo["origin"])
            cc = info["host"].mean(0) - np.array(info["origin"])
            init = (cc - pc)
            W = ecc_pair(gray_patch(slides, pinfo), gray_patch(slides, info),
                         init[0], init[1])
            warps[s] = compose(warps[prev_sec], W)
        prev_sec = s

    # rasterise into z-stack
    z0, z1 = order[0], order[-1]
    nz = z1 - z0 + 1
    classes = ["glom", "vessels", "mesangial"]
    vols = {c: np.zeros((nz, CANVAS, CANVAS), np.uint8) for c in classes}
    cells = []   # (z_um, x_um, y_um, r_um) in canvas frame
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

    # fill missing z by sdf interpolation (glom/vessels/mesangial)
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

    # registration QC: host outlines over sections
    qc = np.full((CANVAS, CANVAS, 3), 255, np.uint8)
    for i, s in enumerate(order):
        col = tuple(int(x) for x in cv2.applyColorMap(
            np.array([[int(255 * i / max(len(order) - 1, 1))]], np.uint8),
            cv2.COLORMAP_VIRIDIS)[0, 0])
        cnts, _ = cv2.findContours(vols["glom"][s - z0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(qc, cnts, -1, col, 2)
    cv2.imwrite(str(OUTDIR / "qc_registration_g1.jpg"), qc)

    # volumes -> meshes
    out = {"meta": dict(mouse="22-184", glom="g1", sections=[z0, z1],
                        z_um=ZUM, voxel_xy_um=MPP * DS), "meshes": {}}
    for c in classes:
        small = np.stack([cv2.resize(sl.astype(np.float32), (CANVAS // DS, CANVAS // DS),
                                     interpolation=cv2.INTER_AREA) for sl in vols[c]])
        if small.max() == 0:
            continue
        pad = np.zeros((small.shape[0] + 2, small.shape[1], small.shape[2]), np.float32)
        pad[1:-1] = small
        sm = ndimage.gaussian_filter(pad, (0.9, 1.3, 1.3))
        if sm.max() < 0.5:
            continue
        verts, faces, normals, _ = marching_cubes(sm, 0.5, spacing=(ZUM, MPP * DS, MPP * DS))
        verts = verts[:, [2, 1, 0]]        # -> (x, y, z) um
        normals = normals[:, [2, 1, 0]]
        out["meshes"][c] = dict(v=b64(verts.astype(np.float32)),
                                n=b64((-normals).astype(np.float32)),
                                f=b64(faces.astype(np.uint32)),
                                nv=len(verts), nf=len(faces))
        print(c, len(verts), "verts", len(faces), "faces")
    cells = np.array(cells, np.float32)
    out["cells"] = dict(p=b64(cells[:, [1, 2, 0]]), r=b64(cells[:, 3]), n=len(cells))
    print("cells:", len(cells))
    json.dump(out, open(OUTDIR / "meshes_g1.json", "w"))
    print("size:", (OUTDIR / "meshes_g1.json").stat().st_size // 1024, "KB ->", OUTDIR)


if __name__ == "__main__":
    main()
