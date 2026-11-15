"""Shape decision for SORTING (cube / cylinder / triangle / None = unknown). Separate from perception.shape_and_axes,
which only chooses pinch axes and stays as it was. Classical, CPU only, about a millisecond per object.

Why not the pinch-axis shape: it takes the hull of the top points INSIDE the eroded colour mask. Under other light the
colour mask covers one face only (a washed-out top is not "red" any more) and the hull of a part says nothing.

What this does instead (see exp/shape/REPORT.md for the numbers):
  1. `top_face`: the object's top face from the DEPTH image alone: object-high pixels connected to the colour blob
     (heights measured from the counter around the object), then the pixels within `band` of that region's top
     height. Light and colour do not matter.
  2. `fit`: the top face is laid on a 1.5 mm grid of the top plane in metric counter coordinates (each top pixel with
     its own depth, so the oblique view is undone). Every other pixel is either SEEN to be lower (counter, the
     object's own walls) or NOT SEEN (a hand above it, no depth, outside the image). The true-size footprints of the
     three shapes (45 mm square, 45 mm circle, triangle of 45 mm altitude; every yaw, +-12 mm shift) are slid over
     the grid; a shape's error = top cells it does not cover + cells it covers that are seen not to be top.
     Hidden cells cost nothing, so an object half under a hand is judged on the part that shows.
  3. `decide`: the best shape wins when its error is small, the second best is clearly worse and enough of the
     footprint was seen; otherwise None ("unknown": hidden, lying on its side, two objects touching, too close).
`ShapeMemory` keeps the label per object over the looks of an episode (tracked by position, votes weighted by
confidence), so an object that is dragged, tipped or half hidden later keeps the label of its clean looks.
"""
from __future__ import annotations

import threading

import numpy as np

SHAPES = ("cube", "cylinder", "triangle")
DEFAULTS = {
    "band": 0.005,            # m below the top height that still counts as top face
    "min_h": 0.35,            # object-high = above this fraction of the object height
    "min_top_px": 150,        # fewer top pixels: no decision
    "ridge": True,            # name a lying prism by its ridge
    "min_ridge_px": 40,       # fewer top pixels: not even a ridge
    "ridge_len": (0.6, 1.35), # strip length, in object sizes (the prism is 1.0 long)
    "ridge_width": 0.36,      # strip width at most (a lying cylinder's top band is 0.6 wide, a prism's ridge 0.13-0.25)
    "ridge_fill": 0.55,       # share of the strip's rectangle that is top
    "ridge_sides": 0.7,       # share of seen-lower cells in the bands along both long sides
    "ridge_conf": 0.4,
    "floor_pct": 35.0,        # counter height near the object = this percentile of the near-counter heights in the patch
    "floor_max": 0.03,        # m: "near the counter" = within this of scene.yaml's counter height
    "above": 0.012,           # m above the nominal object height that still belongs to an object
    "cell": 0.0015,           # m, grid of the top plane
    "reach": 8,               # cells the footprint may slide from the centre of the seen top face (+-12 mm)
    "yaw_step_deg": 5.0,
    "edge_px": 1,             # half-high readings this close (px) to the top face count as "not seen": smeared depth edges
    "edge_h": 0.25,           # ... "half-high" = above this share of the top height
    "blob_reach_px": 6,       # a top face this close (px) to the colour blob belongs to it
    "max_off": 0.8,           # fitted footprint centre to the detection's position, in sizes; beyond = somebody else's face
    "min_seen": 0.4,          # share of the best footprint that lies in seen cells (the rest is behind a hand)
    "max_err": 0.10,          # error of the best shape, in size^2 (a cube mistaken for a cylinder is 0.21)
    "min_gap": 0.30,          # error gap to every other shape, as a share of that pair's full-view gap (FULL_GAP)
    "full_gap": 0.8,          # gap that counts as full confidence
}


def _cfg(cfg):
    c = dict(DEFAULTS)
    if cfg:
        c.update({k: v for k, v in cfg.items() if k in DEFAULTS})
    return c


def top_face(roi, K, T, counter_z, height, cfg=None):
    """Top-face mask of the object in the patch, from depth. Returns (mask uint8, P pelvis points HxWx3, info) or
    (None, None, info)."""
    import cv2
    c = _cfg(cfg)
    depth = roi["depth"]
    if depth is None or depth.size == 0:
        return None, None, {"why": "no_depth"}
    d = cv2.medianBlur(np.ascontiguousarray(depth, np.uint16), 5).astype(np.float32)
    valid = (d >= 150) & (d <= 2000)
    if valid.mean() < 0.3:
        return None, None, {"why": "no_depth"}
    h_, w_ = d.shape
    uu, vv = np.meshgrid(np.arange(w_, dtype=np.float32) + roi["u0"], np.arange(h_, dtype=np.float32) + roi["v0"])
    z = d * 0.001
    x = (uu - K[0, 2]) / K[0, 0] * z
    y = (vv - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], -1) @ np.asarray(T[:3, :3], np.float32).T + np.asarray(T[:3, 3], np.float32)
    hgt = P[..., 2] - counter_z
    bl = (roi["blob"] > 0) & valid
    if bl.sum() < 20:
        return None, None, {"why": "no_overlap"}
    # heights are taken from the counter AROUND the object, not from scene.yaml: a head camera that is 1-2 deg or
    # 1-2 cm off lifts or sinks the whole counter by 1-2 cm (then the mat itself reads "object-high")
    near = valid & (np.abs(hgt) < c["floor_max"])
    floor = float(np.percentile(hgt[near], c["floor_pct"])) if near.sum() > 200 else 0.0
    hgt = hgt - floor
    # the coloured thing itself is taller than an object can be (other blocks than scene.yaml says, or two stacked):
    # the footprints below are for the nominal size only, so say nothing
    if float(np.percentile(hgt[bl], 95)) > height + c["above"]:
        return None, None, {"why": "tall"}
    # object-high and not higher than an object can be: anything clearly above that next to it is a hand. (Not the
    # colour blob's own height: under strong light the top is washed out and the blob is one side wall only.)
    obj = (valid & (hgt > c["min_h"] * height) & (hgt < height + c["above"])).astype(np.uint8)
    if roi.get("x_max") is not None:
        obj[P[..., 0] > roi["x_max"]] = 0          # the boxes' walls
    n, lab = cv2.connectedComponents(obj, connectivity=4)
    if n < 2:
        return None, None, {"why": "no_object"}
    cnt = np.bincount(lab[bl], minlength=n)
    cnt[0] = 0
    if cnt.max() < 20:
        return None, None, {"why": "no_overlap"}
    comp = lab == int(cnt.argmax())
    h_top = float(np.percentile(hgt[comp], 90))
    top = (comp & (hgt > h_top - c["band"])).astype(np.uint8)
    top = cv2.morphologyEx(top, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(top, connectivity=8)
    if n < 2:
        return None, None, {"why": "no_top"}
    # several top faces in one object-high region = objects that touch: take the one at the colour blob (the blob may
    # be a side wall only, so look a few pixels around it), the largest only when the blob touches none
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if n > 2:
        r = int(c["blob_reach_px"])
        at = np.bincount(lab[cv2.dilate(bl.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0], minlength=n)
        at[0] = 0
        if at.max() > 0:
            k = int(at.argmax())
    top = (lab == k).astype(np.uint8)
    top = cv2.morphologyEx(top, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # unknown = no depth, or something nearer to the camera than the top face (a hand above it): what lies behind
    # such a pixel cannot be judged. Everything else that is not the top face is known to be lower (counter, walls).
    unknown = (~valid) | (hgt > h_top + c["above"])
    return top, P, {"h_top": h_top, "floor": floor, "n_top": int(top.sum()), "n_obj": int(comp.sum()), "unknown": unknown,
                    "z_top": float(np.median(P[..., 2][top > 0]))}


_TEMPLATES = {}


def templates(size, cell, step_deg=5.0):
    """Footprint masks of the three shapes on a `cell` grid, every `step_deg` of yaw: {shape: (N, n, n) float32}."""
    import cv2
    key = (round(size, 5), round(cell, 5), step_deg)
    if key in _TEMPLATES:
        return _TEMPLATES[key]
    s = size / cell                                   # object size in cells
    n = int(np.ceil(4.0 * s / 3.0)) + 3               # the triangle's circumcircle is 4/3 * altitude wide
    n += 1 - n % 2
    c = (n - 1) / 2.0
    out = {}
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * s / 2
    side = 2.0 * s / np.sqrt(3.0)                     # equilateral triangle with altitude s, centroid at the origin
    tr = np.array([[-side / 2, -s / 3], [side / 2, -s / 3], [0.0, 2 * s / 3]])
    for name, poly, period in (("cube", sq, 90.0), ("triangle", tr, 120.0)):
        ms = []
        for a in np.arange(0.0, period, step_deg):
            ca, sa = np.cos(np.radians(a)), np.sin(np.radians(a))
            p = poly @ np.array([[ca, sa], [-sa, ca]]) + c
            m = np.zeros((n, n), np.uint8)
            cv2.fillConvexPoly(m, np.round(p * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
            ms.append(m.astype(np.float32))
        out[name] = np.stack(ms)
    m = np.zeros((n, n), np.uint8)
    cv2.circle(m, (int(round(c * 16)), int(round(c * 16))), int(round(s / 2 * 16)), 1, -1, lineType=cv2.LINE_8, shift=4)
    out["cylinder"] = m.astype(np.float32)[None]
    _TEMPLATES[key] = out
    return out


def footprint(roi, K, T, top, P, info, cell=0.0015, reach=8):
    """The top face on a metric grid of the top plane: +1 = top face, -1 = seen and not top face, 0 = not seen (hidden
    by a hand, outside the image, no depth). Top pixels land at their own measured xy; every other seen pixel at the
    point where its ray crosses the top plane (a wall or counter pixel always crosses it outside the footprint)."""
    R, o = np.asarray(T[:3, :3], float), np.asarray(T[:3, 3], float)
    tp = top > 0
    xy_top = P[..., :2][tp].astype(float)
    cx, cy = np.median(xy_top, axis=0)
    h_, w_ = top.shape
    uu, vv = np.meshgrid(np.arange(w_) + roi["u0"], np.arange(h_) + roi["v0"])
    ray = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1) @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = (info["z_top"] - o[2]) / ray[..., 2]
    px, py = o[0] + sc * ray[..., 0], o[1] + sc * ray[..., 1]
    return cx, cy, px, py, xy_top


def fit(roi, K, T, top, P, info, size, cfg):
    """Slide the three footprints (all yaws, +-reach cells) over the grid; error of a shape = top-face cells it does
    not cover + cells it covers that are seen NOT to be top face, in units of size^2."""
    import cv2
    c = cfg
    cell, reach = float(c["cell"]), int(c["reach"])
    tpl = templates(size, cell, float(c["yaw_step_deg"]))
    n = tpl["cylinder"].shape[-1]
    G = n + 2 * reach
    cx, cy, px, py, xy_top = footprint(roi, K, T, top, P, info, cell)
    W = np.zeros((G, G), np.float32)
    lower = (top == 0) & (~info["unknown"]) & np.isfinite(px)
    k = int(c["edge_px"])
    if k > 0:
        # a pixel right at the top face's outline that reads neither top nor counter is a smeared depth edge (or the
        # first mm of a side wall): it does not say where the face ends, so it counts as "not seen"
        hgt = P[..., 2] - (info["z_top"] - info["h_top"])
        near = cv2.dilate((top > 0).astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
        lower &= ~(near & (hgt > float(c["edge_h"]) * info["h_top"]))
    gx = np.round((px[lower] - cx) / cell).astype(np.int64) + G // 2
    gy = np.round((py[lower] - cy) / cell).astype(np.int64) + G // 2
    ok = (gx >= 0) & (gx < G) & (gy >= 0) & (gy < G)
    W[gy[ok], gx[ok]] = -1.0
    gx = np.round((xy_top[:, 0] - cx) / cell).astype(np.int64) + G // 2
    gy = np.round((xy_top[:, 1] - cy) / cell).astype(np.int64) + G // 2
    ok = (gx >= 0) & (gx < G) & (gy >= 0) & (gy < G)
    W[gy[ok], gx[ok]] = 1.0
    n_top = float((W > 0).sum())
    known = (W != 0).astype(np.float32)
    s2 = (size / cell) ** 2
    err, seen, pose = {}, {}, {}
    for name in SHAPES:
        best = None
        for i, M in enumerate(tpl[name]):
            r = cv2.matchTemplate(W, M, cv2.TM_CCORR)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if best is None or mx > best[0]:
                best = (mx, i, loc)
        mx, i, (lx, ly) = best
        M = tpl[name][i]
        err[name] = float(n_top - mx) / s2
        seen[name] = float((known[ly:ly + n, lx:lx + n] * M).sum() / M.sum())
        pose[name] = (i, lx - reach, ly - reach)
    off = None
    if roi.get("xy") is not None:      # how far the fitted footprints sit from where the detection says the object is
        off = {k: float(np.hypot(cx + pose[k][1] * cell - roi["xy"][0], cy + pose[k][2] * cell - roi["xy"][1])) for k in pose}
    return err, seen, n_top / s2, pose, ridge(W, size, cell, c), off


def ridge(W, size, cell, c):
    """A prism lying on one of its rectangular faces shows no top face, only its ridge: a narrow straight strip about
    one object long, with lower ground SEEN on both long sides (a sliver of a top face that peeps out from under a
    hand has the hand, i.e. unseen cells, on one side). Returns {"w", "l", "sides"} in units of size, or None."""
    import cv2
    ys, xs = np.nonzero(W > 0)
    if len(xs) < 12:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (cx, cy), (a, b), ang = cv2.minAreaRect(pts)
    if a < b:
        a, b, ang = b, a, ang + 90.0
    s = size / cell
    l, w = (a + 1.0) / s, (b + 1.0) / s
    if not (c["ridge_len"][0] <= l <= c["ridge_len"][1] and w <= c["ridge_width"]):
        return None
    if len(xs) < c["ridge_fill"] * (a + 1.0) * (b + 1.0):
        return None
    u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])      # along the strip
    n = np.array([-u[1], u[0]])
    off, wid = b / 2 + 0.13 * s, 0.13 * s                                  # a band 6 mm wide, 6 mm off the strip's edge
    sides = []
    for sg in (1.0, -1.0):
        ctr = np.array([cx, cy]) + sg * off * n
        q = np.array([ctr + 0.4 * a * u + wid / 2 * n, ctr + 0.4 * a * u - wid / 2 * n,
                      ctr - 0.4 * a * u - wid / 2 * n, ctr - 0.4 * a * u + wid / 2 * n])
        m = np.zeros(W.shape, np.uint8)
        cv2.fillConvexPoly(m, np.round(q).astype(np.int32), 1)
        k = m > 0
        sides.append(float((W[k] < 0).mean()) if k.sum() >= 6 else 0.0)
    if min(sides) < c["ridge_sides"]:
        return None
    return {"w": round(float(w), 2), "l": round(float(l), 2), "sides": round(min(sides), 2)}


def features(roi, K, T, counter_z, size, height, cfg=None):
    """Measurements of the top face: dict; without a usable top face it holds only the reason ("why")."""
    c = _cfg(cfg)
    top, P, info = top_face(roi, K, T, counter_z, height, c)
    if top is None:
        return info
    if info["n_top"] < c["min_ridge_px"]:
        return {"why": "small", "n_top": info["n_top"]}
    err, seen, area, pose, rdg, off = fit(roi, K, T, top, P, info, size, c)
    f = {"h_top": round(info["h_top"], 4), "floor": round(info["floor"], 4), "n_top": info["n_top"], "area": round(area, 3),
         "err": {k: round(v, 3) for k, v in err.items()}, "seen": {k: round(v, 2) for k, v in seen.items()}}
    if off:
        f["off"] = {k: round(v / size, 2) for k, v in off.items()}
    if rdg:
        f["ridge"] = rdg
    return f


# error gap to the other shapes when the whole top face shows (medians of the sim tuning set, in size^2): the gap
# of a look is judged against it, so "a circle that is a little better than a triangle" counts for less than "a
# circle that is a little better than a square" (a half-hidden triangle fits a circle with the rest behind the hand)
FULL_GAP = {"cube": {"cylinder": 0.153, "triangle": 0.364}, "cylinder": {"cube": 0.160, "triangle": 0.253},
            "triangle": {"cube": 0.358, "cylinder": 0.252}}


def decide(f, cfg=None):
    """features -> (shape or None, confidence 0..1, info)."""
    c = _cfg(cfg)
    if "err" not in f:
        return None, 0.0, f
    err = f["err"]
    best = min(err, key=err.get)
    gap = min((err[o] - err[best]) / FULL_GAP[best][o] for o in err if o != best)
    f = {**f, "gap": round(float(gap), 2)}
    why = None
    if f.get("off") and f["off"][best] > c["max_off"]:
        # the face that was measured is not where the detection is: a neighbour's (objects that touch, a stray blob)
        return None, 0.0, {**f, "why": "elsewhere"}
    if f.get("ridge") and c["ridge"]:
        # a prism on its side: the only one of the three shapes that shows a sharp ridge
        return "triangle", float(c["ridge_conf"]), {**f, "lying": True}
    if f["n_top"] < c["min_top_px"]:
        why = "small"
    elif f["seen"][best] < c["min_seen"]:
        why = "hidden"
    elif err[best] > c["max_err"]:
        why = "far"
    elif gap < c["min_gap"]:
        why = "close"
    if why:
        return None, 0.0, {**f, "why": why}
    conf = float(np.clip((gap - c["min_gap"]) / (c["full_gap"] - c["min_gap"]), 0.2, 1.0) * min(f["seen"][best] / 0.9, 1.0))
    return best, conf, f


def classify(roi, K, T, counter_z, size, height, cfg=None):
    return decide(features(roi, K, T, counter_z, size, height, cfg), cfg)


MEM_DEFAULTS = {
    "track_radius": 0.06,     # m: a detection of the same colour this near a remembered object is that object
    "same_spot": 0.02,        # m: this near it is that object whatever colour it reads
    "moved": 0.015,           # m from where it was first seen = it has been pushed: later looks count half
    "min_votes": 0.15,        # summed confidence the leading shape needs (one clean look gives 0.2-1.0)
    "min_share": 0.7,         # ... and its share of all votes
    "max_missed": 2,          # looks without a detection before the object is forgotten (picked or gone)
}


class ShapeMemory:
    """Sorting shape per object over one episode. update(dets) matches the detections of a look to the remembered
    objects by position, adds the look's decision (weight = its confidence) and writes the remembered label back to
    every detection: det.sort_label (cube | cylinder | triangle | None), det.sort_label_conf, det.sort_votes."""

    def __init__(self, cfg=None):
        self.c = dict(MEM_DEFAULTS)
        self.c.update({k: v for k, v in (cfg or {}).items() if k in MEM_DEFAULTS})
        self.tracks = []
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.tracks = []

    def label(self, votes):
        tot = float(sum(votes.values()))
        if tot <= 0.0:
            return None, 0.0
        best = max(votes, key=votes.get)
        share = votes[best] / tot
        if votes[best] < self.c["min_votes"] or share < self.c["min_share"]:
            return None, 0.0
        return best, float(share * min(votes[best], 1.0))

    def update(self, dets, look=True):
        """look=False (a glance while the arms move): votes are added, nothing is forgotten."""
        c = self.c
        with self._lock:
            pairs = []
            for i, d in enumerate(dets):
                xy = np.asarray(d.top[:2], float)
                for j, t in enumerate(self.tracks):
                    e = float(np.linalg.norm(xy - t["xy"]))
                    if e < c["same_spot"] or (e < c["track_radius"] and t["color"] == d.color):
                        pairs.append((e, i, j))
            pairs.sort()
            di, tj = {}, set()
            for e, i, j in pairs:
                if i in di or j in tj:
                    continue
                di[i] = j
                tj.add(j)
            for i, d in enumerate(dets):
                xy = np.asarray(d.top[:2], float).copy()
                if i in di:
                    t = self.tracks[di[i]]
                else:
                    t = {"xy": xy, "xy0": xy, "color": d.color, "votes": {}, "missed": 0, "n": 0}
                    self.tracks.append(t)
                t["xy"], t["color"], t["missed"] = xy, d.color, 0
                t["n"] += 1
                s = getattr(d, "sort_shape", None)
                if s in SHAPES:
                    w = max(float(getattr(d, "sort_conf", 0.0) or 0.0), 0.05)
                    if np.linalg.norm(xy - t["xy0"]) > c["moved"]:
                        w *= 0.5
                    t["votes"][s] = t["votes"].get(s, 0.0) + w
                d.sort_label, d.sort_label_conf = self.label(t["votes"])
                d.sort_votes = {k: round(v, 2) for k, v in t["votes"].items()}
                t["seen"] = True
            for t in self.tracks:
                if look and not t.pop("seen", False):
                    t["missed"] += 1
                t.pop("seen", None)
            if look:
                self.tracks = [t for t in self.tracks if t["missed"] <= c["max_missed"]]
        return dets
