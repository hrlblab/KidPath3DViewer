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

    # ---- pairwise repair + polish, each correction applied to the whole
    # downstream chain so inter-pair alignment is preserved. ----
    rb = [blur(r) for r in regs]

    def apply_downstream(i, Wg):
        """Wg: correction in GRID coords for section i and everything after."""
        T = compose(SCi, compose(Wg, SC))
        for j in range(i, len(order)):
            warps[order[j]] = compose(T, warps[order[j]])
            regs[j], masks[j] = render(order[j])
            rb[j] = blur(regs[j])

    # pass A: translation grid search on EVERY pair, gated on NCC gain.
    # NCC is evaluated on a FIXED domain (the previous section's ball
    # interior): a shrinking mask-intersection domain rewards sliding the
    # ball away until only a high-gradient sliver overlaps.
    for i in range(1, len(order)):
        dom = masks[i - 1] > 0
        n0 = _ncc(rb[i], rb[i - 1], dom)
        best, bdx, bdy = n0, 0, 0

        def ncc_at(dx, dy):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            g = cv2.warpAffine(rb[i], M, (GRID, GRID), borderValue=245)
            return _ncc(g, rb[i - 1], dom)

        rng = 24 if n0 < 0.15 else 12
        for dy in range(-rng, rng + 1, 2):
            for dx in range(-rng, rng + 1, 2):
                v = ncc_at(dx, dy)
                if v > best:
                    best, bdx, bdy = v, dx, dy
        # hill-climb +-2 from the coarse optimum (handles optima at the edge)
        for _ in range(15):
            better = False
            for dy in range(bdy - 2, bdy + 3):
                for dx in range(bdx - 2, bdx + 3):
                    if dx == bdx and dy == bdy:
                        continue
                    v = ncc_at(dx, dy)
                    if v > best:
                        best, bdx, bdy, better = v, dx, dy, True
            if not better:
                break
        gate = 0.05 if n0 < 0.15 else 0.03
        if best < n0 + gate or (bdx == 0 and bdy == 0):
            continue
        if verbose:
            print(f"  repair pair {order[i - 1]}->{order[i]}: "
                  f"NCC {n0:.3f} -> {best:.3f}  shift ({bdx},{bdy})px")
        apply_downstream(i, np.float32([[1, 0, bdx], [0, 1, bdy]]))
    if verbose:
        mu, mn = _metric(rb, masks)
        print(f"v4 rep  : NCC mean {mu:.4f}  min {mn:.4f}")

    # pass B: per-pair euclidean ECC polish (adds the rotation the grid search
    # cannot; gated on NCC gain and small motion)
    K = np.ones((15, 15), np.uint8)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    for i in range(1, len(order)):
        dom = masks[i - 1] > 0
        n0 = _ncc(rb[i], rb[i - 1], dom)
        W = np.eye(2, 3, dtype=np.float32)
        try:
            _, W = cv2.findTransformECC(rb[i - 1], rb[i], W, cv2.MOTION_EUCLIDEAN,
                                        crit, cv2.dilate(masks[i], K), 5)
        except cv2.error:
            continue
        sh = float(np.hypot(W[0, 2], W[1, 2]))
        rot = abs(_math.degrees(_math.atan2(W[1, 0], W[0, 0])))
        if sh > 8 or rot > 4:
            continue
        warped = cv2.warpAffine(rb[i], W, (GRID, GRID), borderValue=245)
        n1 = _ncc(warped, rb[i - 1], dom)
        if n1 <= n0 + 0.01:
            continue
        if verbose:
            print(f"  polish pair {order[i - 1]}->{order[i]}: "
                  f"NCC {n0:.3f} -> {n1:.3f}  shift {sh:.1f}px rot {rot:.2f}deg")
        apply_downstream(i, W)
    if verbose:
        mu, mn = _metric(rb, masks)
        print(f"v4 pw   : NCC mean {mu:.4f}  min {mn:.4f}")
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

    # ---- de-drift: the pairwise corrections accumulate into a low-frequency
    # drift/tilt of the whole stack that serial sections cannot determine
    # (no global reference per slide).  Subtract the smoothed ball-centroid
    # trajectory: the ball stays compact and upright, while the pairwise
    # (high-frequency) alignment is preserved — adjacent sections move by the
    # smooth path's per-step difference only (<2px).
    cents = []
    for i in range(len(order)):
        ys, xs = np.nonzero(masks[i])
        cents.append((xs.mean(), ys.mean()))
    cents = np.array(cents)
    # linear trend only: removing a straight line perturbs every pair by the
    # same (total drift / n) — the minimal per-pair disturbance; a smoothed
    # path would locally undo the repaired pairs.
    t = np.arange(len(order), dtype=float)
    A = np.column_stack([t, np.ones_like(t)])
    smp = A @ np.linalg.lstsq(A, cents, rcond=None)[0]
    smp -= smp.mean(0)
    for i, s in enumerate(order):
        T = np.float32([[1, 0, -smp[i, 0]], [0, 1, -smp[i, 1]]])
        warps[s] = compose(SCi, compose(T, compose(SC, warps[s])))
        regs[i], masks[i] = render(s)
    if verbose:
        rb = [blur(r) for r in regs]
        mu, mn = _metric(rb, masks)
        print(f"v4 dedrift: removed up to {np.abs(smp).max():.1f}px drift; "
              f"NCC mean {mu:.4f}  min {mn:.4f}")
        vals = sorted((pair_ncc(rb, i), order[i], order[i + 1])
                      for i in range(len(order) - 1))
        print("v4 worst pairs:", ", ".join(f"{a}->{b}:{v:.3f}" for v, a, b in vals[:4]))
    return warps
