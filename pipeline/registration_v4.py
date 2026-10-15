"""v4 registration: v2 (ball-centred local ECC chain + trajectory smoothing)
as initialisation, then GROUPWISE refinement — every section is re-registered
(euclidean) against the weighted mean of its +-2 registered neighbours, inside
the dilated ball mask, iterated to convergence.  This fixes what v2 leaves:
residual rotation jitter (savgol only smooths translation), accumulated chain
drift, and the pairs where chained ECC failed and fell back to translation.

Exports: compute_warps(secs, order, slides) -> {section: 2x3 warp}
Prints consecutive-pair NCC (inside the ball) before/after as the metric.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import savgol_filter

from reconstruct3d_g1 import DS, OFF, compose, gray_patch, rasterise
from reconstruct3d_g1_v2 import local_ecc
from reconstruct3d_g1_v3 import GRID


def _ncc(a, b, m):
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _metric(regs, masks):
    vals = []
    for i in range(len(regs) - 1):
        m = (masks[i] > 0) & (masks[i + 1] > 0)
        if m.sum() > 100:
            vals.append(_ncc(regs[i], regs[i + 1], m))
    return np.mean(vals), np.min(vals)


def compute_warps(secs, order, slides, iters=3, verbose=True):
    # ---- v2 initialisation: chained local ECC + savgol smoothing ----
    grays = {s: gray_patch(slides, secs[s]) for s in order}
    warps = {}
    base = np.array([[1, 0, OFF], [0, 1, OFF]], np.float32)
    for i, s in enumerate(order):
        if i == 0:
            warps[s] = base
            continue
        prev = order[i - 1]
        p_c = secs[prev]["host"].mean(0) - np.array(secs[prev]["origin"])
        c_c = secs[s]["host"].mean(0) - np.array(secs[s]["origin"])
        W, ok = local_ecc(grays[prev], grays[s], p_c, c_c)
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

    # ---- groupwise refinement at GRID resolution ----
    SC = np.array([[1 / DS, 0, 0], [0, 1 / DS, 0]], np.float32)
    SCi = np.array([[DS, 0, 0], [0, DS, 0]], np.float32)

    def render(s):
        g = cv2.warpAffine(grays[s], compose(SC, warps[s]), (GRID, GRID),
                           flags=cv2.INTER_LINEAR, borderValue=245).astype(np.float32)
        m = rasterise([secs[s]["host"]], secs[s]["origin"], warps[s]).astype(np.float32)
        m = (cv2.resize(m, (GRID, GRID), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
        return g, m

    # adjacent 3um sections decorrelate at 1um texture scale (different cell
    # layers), so estimate on heavily blurred images (coarse structures:
    # lumina, capsule) and only ACCEPT a correction that provably improves the
    # masked NCC and stays small (<=10px, <=3deg).  Guaranteed >= v2 quality.
    import math as _math

    def blur(g):
        return cv2.GaussianBlur(g, (0, 0), 3)

    regs, masks = map(list, zip(*[render(s) for s in order]))

    def pair_ncc(rb, i):
        m = (masks[i] > 0) & (masks[i + 1] > 0)
        return _ncc(rb[i], rb[i + 1], m) if m.sum() > 100 else 1.0

    if verbose:
        mu, mn = _metric([blur(r) for r in regs], masks)
        print(f"v4 init : NCC mean {mu:.4f}  min {mn:.4f}")

    # ---- bad-pair repair: the chained-ECC fallback pairs can be tens of px
    # off; grid-search a translation that maximises the masked NCC against the
    # previous section and apply it to the whole downstream chain. ----
    rb = [blur(r) for r in regs]
    for i in range(1, len(order)):
        n0 = pair_ncc(rb, i - 1)
        if n0 >= 0.15:
            continue
        best, bdx, bdy = n0, 0, 0
        m_prev = masks[i - 1] > 0
        for step, rng, cx, cy in [(3, 30, 0, 0), (1, 3, None, None)]:
            if cx is None:
                cx, cy = bdx, bdy
            for dy in range(cy - rng, cy + rng + 1, step):
                for dx in range(cx - rng, cx + rng + 1, step):
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    g = cv2.warpAffine(rb[i], M, (GRID, GRID), borderValue=245)
                    mm = cv2.warpAffine(masks[i], M, (GRID, GRID)) > 0
                    sel = m_prev & mm
                    if sel.sum() < 100:
                        continue
                    v = _ncc(g, rb[i - 1], sel)
                    if v > best:
                        best, bdx, bdy = v, dx, dy
        if best < n0 + 0.05 or (bdx == 0 and bdy == 0):
            continue
        if verbose:
            print(f"  repair pair {order[i - 1]}->{order[i]}: "
                  f"NCC {n0:.3f} -> {best:.3f}  shift ({bdx},{bdy})px")
        T = np.array([[1, 0, bdx * DS], [0, 1, bdy * DS]], np.float32)
        for j in range(i, len(order)):
            warps[order[j]] = compose(T, warps[order[j]])
            regs[j], masks[j] = render(order[j])
            rb[j] = blur(regs[j])
    if verbose:
        mu, mn = _metric(rb, masks)
        print(f"v4 rep  : NCC mean {mu:.4f}  min {mn:.4f}")

    K = np.ones((15, 15), np.uint8)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    for it in range(iters):
        acc, shifts = 0, []
        rb = [blur(r) for r in regs]
        for i, s in enumerate(order):
            nb = [j for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(order)]
            wn = np.array([1.0 if abs(j - i) == 1 else 0.5 for j in nb], np.float32)
            tmpl = (sum(w * rb[j] for w, j in zip(wn, nb)) / wn.sum()).astype(np.float32)
            mask = cv2.dilate(masks[i], K)
            W = np.eye(2, 3, dtype=np.float32)
            try:
                _, W = cv2.findTransformECC(tmpl, rb[i], W, cv2.MOTION_EUCLIDEAN,
                                            crit, mask, 5)
            except cv2.error:
                continue
            sh = float(np.hypot(W[0, 2], W[1, 2]))
            rot = abs(_math.degrees(_math.atan2(W[1, 0], W[0, 0])))
            if sh > 10 or rot > 3:
                continue
            m = mask > 0
            warped = cv2.warpAffine(rb[i], W, (GRID, GRID), borderValue=245)
            if _ncc(warped, tmpl, m) <= _ncc(rb[i], tmpl, m) + 1e-4:
                continue
            warps[s] = compose(SCi, compose(W, compose(SC, warps[s])))
            regs[i], masks[i] = render(s)
            rb[i] = blur(regs[i])
            acc += 1
            shifts.append(sh)
        if verbose:
            mu, mn = _metric([blur(r) for r in regs], masks)
            print(f"v4 iter{it + 1}: NCC mean {mu:.4f}  min {mn:.4f}  "
                  f"accepted {acc}/{len(order)} mean shift "
                  f"{np.mean(shifts) if shifts else 0:.2f}px")
        if not acc:
            break
    return warps
