# Glom3D Viewer

Interactive 3D reconstruction viewer for glomeruli segmented on serial kidney
sections. Pure static HTML + [three.js](https://threejs.org) — no install, no
server: open the page, rotate/zoom the models, toggle structures
(capsule / capillary loops / mesangium / nuclei), slide a Z cut plane, or drag
& drop your own `meshes.json` to view it. Two ways to see the real histology
in 3D: **H&E slice planes** (registered, ball-masked patches at their true z
positions, densified to 1 µm by optical-flow morphing), **real-stain meshes**
(per-vertex colours sampled from the nearest registered section), and **RGB
volume rendering** (raycast through a near-isotropic 1 µm RGBA voxel volume).
Flow-interpolated frames and voxels between real sections are synthesized for
rendering only — compute quantitative statistics on the original sections.

**Live demo**: enable GitHub Pages on this repo (Settings → Pages → Deploy from
branch → `main` / root), then open `https://<user>.github.io/<repo>/`.

Bundled demo data: two glomeruli of mouse `22-184`, reconstructed from 24 and
30 serial H&E sections (3 µm), segmented with YOLO11-L + SAM2.1 and
registered/interpolated with the pipeline below.

## Use your own data

The viewer loads any JSON with this schema (all binary fields are
base64-encoded little-endian arrays):

```jsonc
{
  "meta":   { "z_um": 3.0, "sections": [11, 36], "...": "free-form" },
  "meshes": {
    "glom":      { "v": "<u16 xyz triplets>", "lo": [x,y,z], "scale": [sx,sy,sz],
                   "f": "<u32 triangle indices>" },
    "vessels":   { "...": "same; 'v' may instead be plain float32 without lo/scale" },
    "mesangial": { }
  },
  "cells":  { "p": "<f32 xyz um>", "r": "<f32 radius um>", "n": 123 },
  "slices": { "dir": "tex_g1", "x0": 0, "y0": 0, "w": 180, "h": 175,
              "items": [ { "f": "sec11.webp", "z": 1.0 } ] }
}
```

Coordinates are micrometres. `meshes.glom` is required; other classes,
`cells` and `slices` are optional. Normals (`"n"`) are computed in the browser
when absent; per-vertex colours (`"c"`, u8 RGB) enable the real-stain mode.
`slices` textures are RGBA images served relative to the JSON (works for
bundled data; drag & drop loads meshes only).

## Reconstruction pipeline (`pipeline/`)

From per-section instance segmentations (GeoJSON polygons in slide
coordinates, one file per section) to `meshes.json`:

1. `reconstruct3d_g1.py` — data loading, rasterisation, SDF helpers (library).
2. `reconstruct3d_g1_v2.py` — ball-centred ECC rigid registration
   (translation + rotation) + Savitzky-Golay trajectory smoothing.
3. `registration_v4.py` — the registration used by the final exports: v2 as
   initialisation, then a bad-pair repair pass (pairs whose masked NCC is low
   get a translation grid search applied to the downstream chain) and a gated
   groupwise refinement (corrections only accepted when they provably raise
   the masked NCC).
4. `reconstruct3d_g1_v3.py` / `reconstruct3d_ball.py` — final recipe:
   registration → signed-distance-field z-interpolation (3 µm → 1 µm, removes
   stair-step artefacts without fattening thin structures) → marching cubes →
   quantised JSON export.

```
python pipeline/reconstruct3d_ball.py <glom-number>
```

The scripts carry absolute paths from our environment — edit the constants at
the top (slide folder, GeoJSON folder, pixel size `MPP`, section thickness
`ZUM`) for your data. Dependencies: `numpy scipy scikit-image opencv-python
openslide-python`.

Note: the SDF z-interpolation generates geometric transitions between
sections for rendering; quantitative volume/surface statistics should be
computed on the original sections.

## 中文说明

交互式肾小球三维重建查看器:纯静态网页,打开即用。鼠标拖动旋转、滚轮缩放、
右键平移;可逐类显隐(外膜/血管袢/系膜/细胞核)、调外膜透明度、Z 向剖切;
把你自己生成的 `meshes.json` 拖进页面即可加载(格式见上)。
重建管线:以球为中心的 ECC 刚体配准 → 轨迹平滑 → SDF 层间插值(3µm→1µm)
→ marching cubes。管线脚本在 `pipeline/`,需按自己的数据改文件头部的路径与
像素尺寸/切片厚度常量。
