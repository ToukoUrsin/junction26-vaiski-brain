"""Colour perception that survives the real camera.

The baseline's fixed HSV boxes (red H 0-8, yellow H 18-35) were tuned on the sim. On the real head camera (data drop
1/2, venue light) the "red" objects read as orange (H 12-14), yellow as H 40-46 with S ~90, blue/purple H ~109, and
the orange bins H 4-6. Fixed boxes miss two of the three real colours.

Instead every saturated pixel is classified by its nearest hue PROTOTYPE (circular distance), and the prototypes
adapt per frame: a 1-D k-means on the hues of the saturated pixels that survive the geometric bin/counter masks,
seeded at the nominal prototypes. Bins are removed geometrically (known positions + calibration + depth height), not
by colour, so orange objects next to orange bins still work.
"""
from __future__ import annotations

import numpy as np

from aaltonomous.baselines.classical.perception import Perception as BasePerception

# nominal prototypes (OpenCV hue 0-180): sim + real both land nearest the right one
PROTO = {"red": 6.0, "yellow": 34.0, "blue": 114.0}
MAX_DIST = {"red": 16.0, "yellow": 24.0, "blue": 32.0}
# A pixel also counts as a colour when it is this close to the colour's FIXED prototype (whatever the per-frame
# prototype drifted to). Red reaches further on its magenta side only (H >= 164): under one sim lighting a "red"
# cylinder is H 169-174, 16-17 from the prototype, and with the red prototype pulled towards orange by box-edge
# pixels (H 11-16) it was never detected in the whole episode (sim seed 406, three runs of three). Nothing else in
# the scene lives at H 150-170 (blue ends at 146), and on the orange side the nearest-prototype rule already hands
# H > ~20 to yellow, so this adds no confusion.
BASE_DIST = {"red": 22.0, "yellow": 24.0, "blue": 32.0}


def hue_dist(h, p):
    d = np.abs(np.asarray(h, float) - p) % 180.0
    return np.minimum(d, 180.0 - d)


def shape_and_axes(xy):
    """Top-face points (pelvis xy, m) -> (shape, pinch axes, info). The convex hull is compared with its minimum
    enclosing rectangle, circle and triangle; the best-filled one names the shape (ideal fill 1.0; a square fills
    its enclosing triangle 0.5, a circle its rectangle 0.79, a triangle its rectangle 0.5).
    Pinch axes (unit pelvis-xy line directions) = where a two-sided pinch holds:
      cube: the two face normals; triangle: the three altitudes (flat face <-> opposite apex, through the centre
      of mass); cylinder: None (any)."""
    import cv2
    if len(xy) < 25:
        return None, None, None
    p = (np.asarray(xy, float) * 1000.0).astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(p)
    A = float(cv2.contourArea(hull))
    if A < 200.0:          # < 2 cm^2 of top face: too little to judge
        return None, None, None
    (cx, cy), (w, h), ang = cv2.minAreaRect(p)
    _, r = cv2.minEnclosingCircle(p)
    At, tri = cv2.minEnclosingTriangle(hull)
    fill = {"cube": A / max(w * h, 1e-6), "cylinder": A / max(np.pi * r * r, 1e-6), "triangle": A / max(At, 1e-6)}
    shape = max(fill, key=fill.get)
    info = {k: round(float(v), 2) for k, v in fill.items()}
    if shape == "cube" and max(w, h) > 2.0 * max(min(w, h), 1e-6) and max(w, h) > 30.0:
        # a long thin top strip = a triangle (or cylinder) lying on its side: only its ridge is near the top. The
        # flat parallel faces are its two ends: pinch along the strip, never across the sloped/round sides
        shape = "lying"
        a = np.radians(ang) + (0.0 if w >= h else np.pi / 2)
        axes = [np.array([np.cos(a), np.sin(a)])]
    elif shape == "cube":
        a = np.radians(ang)
        axes = [np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])]
    elif shape == "triangle":
        V = tri.reshape(3, 2).astype(float)
        c = V.mean(0)
        axes = [(c - vi) / np.linalg.norm(c - vi) for vi in V]
    else:
        axes = None
    return shape, axes, info


class Perception(BasePerception):
    def __init__(self, cfg, scene, calibration):
        super().__init__(cfg, scene, calibration)
        v = self.cfg.get("adaptive", {}) or {}
        self.s_min = float(v.get("s_min", 60))
        self.v_min = float(v.get("v_min", 70))
        # the black Dex3 hands / arm covers read dark blue (V < 80): blue needs a brighter pixel than the others
        self.v_min_c = {"red": self.v_min, "yellow": self.v_min, "blue": float(v.get("v_min_blue", 100))}
        self.adapt = bool(v.get("enabled", True))
        self.proto = dict(PROTO)
        self._depth = None
        self.self_chains = None          # pelvis-frame polylines of the robot's own arms (set by the policy)
        self.sort_cfg = cfg.get("sort_shape", {}) or {}      # shape decision for SORTING (shape.py); pinch axes untouched
        self.sort_on = False             # set by the policy: only an instruction that sorts by shape needs it
        self.keep_roi = False            # analysis only: attach the raw image patch to every detection
        self._rgb = self._masks = self._keep = self._arm = self._keep_geo = None
        self.self_scale = float(v.get("self_scale", 1.0))   # inflate the FK tubes (calibration slack)
        self.s_pale = float(self.cfg.get("pale_s_min", 0) or 0)
        self.v_pale = float(self.cfg.get("pale_v_min", 140))
        self.pale_min_strict = int(self.cfg.get("pale_min_strict_px", 40))
        self.bins_how = "colour"

    def self_mask(self, shape):
        """255 = free, 0 = the robot's own arm/hand (FK polylines projected and drawn thick)."""
        import cv2
        keep = np.full(shape, 255, np.uint8)
        if not self.self_chains:
            return keep
        for ch, rad in self.self_chains:
            uvz = self.cal.project(self.cam, np.asarray(ch, float))
            for a, b in zip(uvz[:-1], uvz[1:]):
                if a[2] <= 0.05 or b[2] <= 0.05:
                    continue
                z = 0.5 * (a[2] + b[2])
                r = int(np.clip(rad * self.self_scale * self.cal.cams[self.cam].K[0, 0] / z, 3, 120))
                pa, pb = tuple(int(x) for x in np.round(a[:2])), tuple(int(x) for x in np.round(b[:2]))
                if max(abs(pa[0]), abs(pa[1]), abs(pb[0]), abs(pb[1])) > 5000:
                    continue
                cv2.line(keep, pa, pb, 0, thickness=2 * r)
        return keep

    def _keep_mask(self, depth):
        keep = self._keep = super()._keep_mask(depth)
        self._keep_geo = keep.copy()        # before the arm mask: object-high over the mat, bins removed
        sm = self.self_mask(keep.shape)
        if depth is not None and (sm == 0).any():
            # inside the arm tubes keep only pixels whose 3D point is at object height: an object under/next to a
            # hand survives, the hand/wrist itself (higher, or no depth) goes
            vv, uu = np.nonzero(sm == 0)
            d = depth[vv, uu].astype(float)
            lo, hi = self.cfg["depth_range_mm"]
            ok = (d >= lo) & (d <= hi)
            if ok.any():
                P = self.cal.pixel_to_pelvis(self.cam, uu[ok], vv[ok], d[ok])
                h = (P[:, 2] - self.scene.counter_z) / self.height
                obj = (h >= 0.25) & (h <= float(self.cfg.get("adaptive", {}).get("self_keep_h", 1.2)))
                sm[vv[ok][obj], uu[ok][obj]] = 255
        self._arm = sm                     # 0 = the robot's own arm in front (after the object-height carve-out)
        self._keep = keep & sm
        return self._keep

    def counter_plane_fix(self, rgb, depth, max_tilt_deg=4.0, max_dz=0.03):
        """(C, tilt_deg, dz) where C (4x4, pelvis frame) puts the counter surface seen in `depth` back on the level
        plane z = counter_z, or None when there is no reliable fit.
        A head extrinsic error of 1-2 deg tilts the whole scene: at 0.5-0.7 m that is 1-2.5 cm of height, i.e. the
        counter itself rises to "object height" (or objects sink into it) and every height test in the pipeline
        (counter removal, bin walls, the self mask's keep band) goes wrong; under tinted light the mat then became
        "blue" objects. The counter is the biggest flat thing in view, so its plane gives pitch, roll and height of
        the camera; the box row (locate_bins) then gives x / y. Rotation is about the centroid of the fitted points,
        so the xy of what stands there barely moves (a real robot that leans a little gets the same heights the
        planner assumes, nothing else changes)."""
        if depth is None:
            return None
        sc = self.scene
        d = depth[::2, ::2].astype(float)
        lo, hi = self.cfg["depth_range_mm"]
        ok = (d >= lo) & (d <= hi) & (np.asarray(rgb)[::2, ::2].max(-1) >= 40)       # (not the black hands)
        vv, uu = np.nonzero(ok)
        if len(vv) < 3000:
            return None
        P = self.cal.pixel_to_pelvis(self.cam, 2 * uu, 2 * vv, d[vv, uu])
        b = sc.cfg["bins"]
        near = min(sc.mat_to_pelvis(np.r_[np.asarray(c, float), 0.0])[0] for c in b["centers"].values())
        near -= float(b["outer_size"][1]) / 2
        P = P[(P[:, 0] > 0.18) & (P[:, 0] < near - 0.04) & (np.abs(P[:, 1]) < 0.32)
              & (np.abs(P[:, 2] - sc.counter_z) < 0.05)]
        if len(P) < 3000:
            return None
        # robust plane z = a x + b y + c: everything that is not counter (objects, fingers) lies ABOVE it
        a_, b_, c_ = 0.0, 0.0, float(np.percentile(P[:, 2], 30))
        for up, dn in ((0.02, 0.04), (0.008, 0.015), (0.004, 0.008), (0.003, 0.006), (0.003, 0.006)):
            r = P[:, 2] - (a_ * P[:, 0] + b_ * P[:, 1] + c_)
            sel = (r < up) & (r > -dn)
            if np.count_nonzero(sel) < 2000:
                return None
            a_, b_, c_ = np.linalg.lstsq(np.c_[P[sel, :2], np.ones(np.count_nonzero(sel))], P[sel, 2], rcond=None)[0]
        n = np.array([-a_, -b_, 1.0])
        n /= np.linalg.norm(n)
        tilt = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
        xc, yc = P[sel, 0].mean(), P[sel, 1].mean()
        pc = np.array([xc, yc, a_ * xc + b_ * yc + c_])
        dz = float(pc[2] - sc.counter_z)
        if tilt > np.radians(max_tilt_deg) or abs(dz) > max_dz:
            return None
        ax = np.cross(n, [0.0, 0.0, 1.0])
        s_ = np.linalg.norm(ax)
        R = np.eye(3)
        if s_ > 1e-9:
            k = ax / s_
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + np.sin(tilt) * K + (1 - np.cos(tilt)) * K @ K
        C = np.eye(4)
        C[:3, :3] = R
        C[:3, 3] = np.array([xc, yc, sc.counter_z]) - R @ pc
        return C, float(np.degrees(tilt)), dz

    def locate_bins(self, rgb, depth, max_shift=0.15):
        """The three bins from head RGB-D: {name: pelvis (x, y)} or None. First by colour (orange walls); when that
        finds nothing (washed-out or tinted light, repainted boxes: sim variant with 2x light the walls read S 60,
        H 25) by shape alone: anything wall-high standing in the box row."""
        self.bins_how = "colour"
        r = self._locate_bins(rgb, depth, max_shift, True)
        if r is None and bool((self.cfg.get("adaptive", {}) or {}).get("bins_by_shape", True)):
            r = self._locate_bins(rgb, depth, max_shift, False)
            self.bins_how = "shape"
        return r

    def _locate_bins(self, rgb, depth, max_shift, by_colour):
        """Measure the three bins from head RGB-D: orange-ish wall points above the counter, split into three
        clusters along y, each centre = middle of its robust x/y extent. Returns {name: pelvis (x, y)} or None.
        The real box row need not be where scene.yaml says (drop 2 shows it further away); objects come from the
        camera anyway, so measuring the boxes with the same camera keeps object and bin consistent."""
        import cv2
        if depth is None:
            return None
        sc, b = self.scene, self.scene.cfg["bins"]
        d = depth[::2, ::2].astype(float)
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb[::2, ::2]), cv2.COLOR_RGB2HSV)
        H, S, V = hsv[..., 0].astype(float), hsv[..., 1], hsv[..., 2]
        lo, hi = self.cfg["depth_range_mm"]
        nominal = {n: sc.mat_to_pelvis(np.r_[np.asarray(c, float), 0.0])[:2] for n, c in b["centers"].items()}
        half = float(b["outer_size"][0]) / 2
        near = min(c[0] for c in nominal.values()) - half
        if by_colour:
            m = ((d >= lo) & (d <= hi) & (S >= 70) & (V >= 60) & ((H <= 22) | (H >= 172))).astype(np.uint8)
        else:
            # any colour: not black (hands), 2 cm above the counter up to just over the rim, in the box row's area
            m = np.zeros(d.shape, np.uint8)
            v0, u0 = np.nonzero((d >= lo) & (d <= hi) & (V >= 40))
            if len(v0) < 400:
                return None
            P0 = self.cal.pixel_to_pelvis(self.cam, 2 * u0, 2 * v0, d[v0, u0])
            h0 = P0[:, 2] - sc.counter_z
            # (only the walls pass, not the box floors: thin rings, so a smaller blob limit, and objects are kept out
            # by x instead: nothing on the mat stands within 5 cm of the box row)
            k0 = (h0 > 0.02) & (h0 < float(b.get("wall_height", 0.06)) + 0.03) & (P0[:, 0] > near - 0.05) \
                & (np.abs(P0[:, 1]) < 0.7)
            m[v0[k0], u0[k0]] = 1
        # boxes are big blobs; a (possibly orange) object is < ~900 px at this half resolution
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        big = np.zeros(n, bool)
        big[1:] = st[1:, cv2.CC_STAT_AREA] >= (int(self.cfg.get("adaptive", {}).get("bin_min_px", 1500))
                                              if by_colour else 300)
        vv, uu = np.nonzero(big[lab])        # (no self mask: the hands are black, and they hide the near wall's edge)
        if len(vv) < 400:
            return None
        P = self.cal.pixel_to_pelvis(self.cam, 2 * uu, 2 * vv, d[vv, uu])
        h = P[:, 2] - sc.counter_z
        k = (h > 0.012) & (h < float(b.get("wall_height", 0.06)) + 0.12) & (P[:, 0] > near - 0.15) & (np.abs(P[:, 1]) < 0.7)
        P, h = P[k], h[k]
        if len(P) < 300:
            return None
        # template: three W-wide boxes, pitch = the nominal spacing; slide along y, keep the best-covered offset
        ny = {n: c[1] for n, c in nominal.items()}
        W = 2 * half
        offs = np.arange(-max_shift, max_shift + 1e-9, 0.005)
        # by shape only the walls are seen (rings): the template's 5 cm gaps would also fit inside a box, so slide
        # it over the near walls alone (three 21 cm segments)
        Py = P[:, 1] if by_colour else P[P[:, 0] < np.percentile(P[:, 0], 2) + 0.04, 1]
        score = [sum(np.count_nonzero(np.abs(Py - (ny[n] + o)) < half) for n in ny) for o in offs]
        o = offs[int(np.argmax(score))]
        out = {}
        # wall points above any upright object's top (an orange/red object in front of a box would otherwise pull
        # its near wall towards the robot: sim seed 505 moved the right bin 13 cm and every object became "in bin")
        h_wall = self.height + 0.008
        for name, c in nominal.items():
            yc = c[1] + o
            band = np.abs(P[:, 1] - yc) < half - 0.02
            sel = band & (h > h_wall)
            if np.count_nonzero(sel) < 50:
                sel = band & (h > 0.02)
            if np.count_nonzero(sel) < 50:
                return None
            x_near = np.percentile(P[sel, 0], 2)          # the near wall's outer face is always in view
            cc = np.array([x_near + W / 2, yc])
            if np.linalg.norm(cc - c) > max_shift:
                return None
            out[name] = cc
        return out

    def counter_plane(self, rgb, depth, z_nominal):
        """Head depth -> the bare counter in front of the box row as a plane (pelvis frame):
        z = z + sx * (x - x0) + sy * y. Returns {z, x0, tilt_x_deg, tilt_y_deg, n, frac, rms} or None.
        Only uncoloured pixels outside the robot's own arms vote, inside a +-6 cm slab around the nominal height;
        the counter is the biggest flat thing in that slab (mode of z), refined by a trimmed least-squares plane."""
        import cv2
        if depth is None:
            return None
        sc, b = self.scene, self.scene.cfg["bins"]
        d = depth[::2, ::2].astype(float)
        lo, hi = self.cfg["depth_range_mm"]
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb[::2, ::2]), cv2.COLOR_RGB2HSV)
        sm = self.self_mask(depth.shape)[::2, ::2]
        m = (d >= lo) & (d <= hi) & (sm > 0) & ~((hsv[..., 1] >= self.s_min) & (hsv[..., 2] >= self.v_min))
        vv, uu = np.nonzero(m)
        if len(vv) < 2000:
            return None
        P = self.cal.pixel_to_pelvis(self.cam, 2 * uu, 2 * vv, d[vv, uu])
        near = min(sc.mat_to_pelvis(np.r_[np.asarray(c, float), 0.0])[0] for c in b["centers"].values())
        near -= float(b["outer_size"][1]) / 2
        x_lo = float(sc.origin[0]) + 0.02
        slab = 0.06
        k = (P[:, 0] > x_lo) & (P[:, 0] < near - 0.03) & (np.abs(P[:, 1]) < 0.35) & (np.abs(P[:, 2] - z_nominal) < slab)
        P = P[k]
        if len(P) < 1500:
            return None
        hist, edges = np.histogram(P[:, 2], bins=np.arange(z_nominal - slab, z_nominal + slab + 1e-6, 0.002))
        zm = edges[int(np.argmax(np.convolve(hist, np.ones(3), "same")))] + 0.001
        x0 = 0.5 * (x_lo + near)
        sel = np.abs(P[:, 2] - zm) < 0.008
        for tol in (0.008, 0.005, 0.004):
            Q = P[sel]
            if len(Q) < 1000:
                return None
            A = np.c_[np.ones(len(Q)), Q[:, 0] - x0, Q[:, 1]]
            coef = np.linalg.lstsq(A, Q[:, 2], rcond=None)[0]
            res = P[:, 2] - (coef[0] + coef[1] * (P[:, 0] - x0) + coef[2] * P[:, 1])
            sel = np.abs(res) < tol
        n = int(sel.sum())
        if n < 1000:
            return None
        # distance of the camera centre from the plane: unlike z at a point, it does not change when the camera is
        # merely ROTATED wrong (a pitch error tilts the plane about the camera), so it separates the two errors
        o = self.cal._T(self.cam)[:3, 3]
        nrm = float(np.sqrt(1.0 + coef[1] ** 2 + coef[2] ** 2))
        dist = float((o[2] - coef[0] - coef[1] * (o[0] - x0) - coef[2] * o[1]) / nrm)
        return {"z": float(coef[0]), "x0": float(x0), "sx": float(coef[1]), "sy": float(coef[2]),
                "tilt_x_deg": float(np.degrees(np.arctan(coef[1]))), "tilt_y_deg": float(np.degrees(np.arctan(coef[2]))),
                "tilt_deg": float(np.degrees(np.arccos(1.0 / nrm))), "dh": float(o[2] - z_nominal - dist),
                "n": n, "frac": float(sel.mean()), "rms": float(np.sqrt(np.mean(res[sel] ** 2)))}

    def _reject(self, det):
        why = super()._reject(det)
        if why:
            return why
        # Orange bins look like the "red" (orange on the real robot) objects: anything at or past the bin row's near
        # edge is a bin (or an object already in one), whatever its colour.
        b = self.scene.cfg["bins"]
        near = min(self.scene.mat_to_pelvis(np.r_[np.asarray(c, float), 0.0])[0] for c in b["centers"].values())
        near -= float(b["outer_size"][1]) / 2
        if det.top[0] > near - float(self.cfg.get("adaptive", {}).get("bin_row_margin", 0.015)):
            return "bin_row"
        return None

    def detect(self, rgb, depth, debug=False):
        self._depth = depth
        out = super().detect(rgb, depth, debug)
        dets = out[0] if debug else out
        # one object, two colours (an orange-red cube under tinted light was "red" on one face and "yellow" on
        # another: two picks planned at one spot, one of them into the wrong box): the bigger mask names it
        dets.sort(key=lambda o: -o.n_px)
        keep = []
        for o in dets:
            if not any(k.color != o.color and np.linalg.norm(k.top[:2] - o.top[:2]) < 0.03 for k in keep):
                keep.append(o)
        dets[:] = keep
        if depth is not None and bool((self.cfg.get("adaptive", {}) or {}).get("shape_objects", True)):
            try:
                dets.extend(self._objects_by_shape(np.asarray(rgb), np.asarray(depth), dets))
            except Exception:      # the fallback must never cost the colour detections
                pass
        return out

    def _objects_by_shape(self, rgb, depth, dets):
        """Objects the colour masks missed: blobs of object height standing on the mat (depth only), named by the hue
        of their most saturated pixels. Strong light washes the TOP faces out to white (sim, light x2: S 5-30, under
        s_min) while the side faces keep some colour; dull objects under dim light are the same case. Blobs within
        4.5 cm of a colour detection are that object. Whatever has no clear hue stays unknown and is left alone."""
        import cv2
        c, sc = self.cfg, self.scene
        dlo, dhi = c["depth_range_mm"]
        d = depth.astype(float)
        valid = (d >= dlo) & (d <= dhi) & (self._keep_mask(depth) > 0)      # bins, counter and own arms are out
        vv, uu = np.nonzero(valid)
        if not len(vv):
            return []
        P = self.cal.pixel_to_pelvis(self.cam, uu, vv, d[vv, uu])
        h = (P[:, 2] - sc.counter_z) / self.height
        ok = (h >= 0.3) & (h <= 1.3)
        for o in dets:             # what a colour detection already explains (so a touching neighbour stands alone)
            ok &= np.hypot(P[:, 0] - o.top[0], P[:, 1] - o.top[1]) > 0.032
        g = np.zeros(depth.shape, np.uint8)
        g[vv[ok], uu[ok]] = 255
        g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
        hsv = None
        out = []
        for i in range(1, n):
            area = int(st[i, cv2.CC_STAT_AREA])
            if not 500 <= area <= 14000:
                continue
            v, u = np.nonzero(lab == i)
            Pb = self.cal.pixel_to_pelvis(self.cam, u, v, d[v, u])
            ext = np.percentile(Pb[:, :2], 98, 0) - np.percentile(Pb[:, :2], 2, 0)
            if ext.max() > 0.10:                       # more than one object wide: not a single object
                continue
            det = self._from_depth("?", u, v, d[v, u], area)
            if det is None or any(np.linalg.norm(det.top[:2] - o.top[:2]) < 0.045 for o in list(dets) + out):
                continue
            if hsv is None:
                hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
            col = self._blob_colour(hsv[v, u])
            if col is None:
                continue
            det.color = col
            det.method = "shape"
            if self._reject(det):
                continue
            out.append(det)
        return out

    def _blob_colour(self, px):
        """Colour name of a blob from its HSV pixels (N, 3): circular mean hue of its most saturated 40 %, if they
        agree and sit near a prototype; else None."""
        S, V = px[:, 1].astype(float), px[:, 2]
        sel = (S >= 30) & (V >= 40)
        if np.count_nonzero(sel) < max(80, 0.06 * len(px)):
            return None
        hs, ss = px[sel, 0].astype(float), S[sel]
        hs = hs[ss >= np.percentile(ss, 60)]
        a = np.radians(hs * 2.0)
        sn, cs = np.sin(a).mean(), np.cos(a).mean()
        if np.hypot(sn, cs) < 0.6:
            return None
        m = (np.degrees(np.arctan2(sn, cs)) / 2.0) % 180.0
        # the blob IS an object (geometry says so), so the question is only which of the three colours: nearest
        # prototype (this frame's or the nominal one), if it is near enough and clearly nearer than the next
        # (a wrong box costs points, an object left on the counter costs none)
        dist = {k: min(float(hue_dist(m, p)), float(hue_dist(m, PROTO[k]))) for k, p in self.proto.items()}
        k, k2 = sorted(dist, key=dist.get)[:2]
        return k if dist[k] <= 32.0 and dist[k2] - dist[k] >= 10.0 else None

    def _from_depth(self, color, u, v, d, area):
        det = super()._from_depth(color, u, v, d, area)
        if det is None:
            return None
        try:
            P = self.cal.pixel_to_pelvis(self.cam, u, v, d)
            z_top = np.percentile(P[:, 2], self.cfg["top_percentile"])
            top = P[P[:, 2] >= z_top - max(self.cfg["top_band"], 0.008)]
            det.shape, det.pinch_axes, det.shape_info = shape_and_axes(top[:, :2])
        except Exception:
            det.shape, det.pinch_axes, det.shape_info = None, None, None
        # shape for SORTING: separate from the pinch-axis shape above (which stays exactly as it was)
        det.sort_shape, det.sort_conf, det.sort_info = None, 0.0, None
        if self.sort_on and self.sort_cfg.get("enabled", True):
            try:
                roi = self._roi(color, u, v)
                if roi is not None:
                    from . import shape as _shape
                    roi["xy"] = (float(det.top[0]), float(det.top[1]))     # where the detection says the object is
                    cam = self.cal.cams[self.cam]
                    det.sort_shape, det.sort_conf, det.sort_info = _shape.classify(
                        roi, cam.K, self.cal._T(self.cam), self.scene.counter_z, self.size, self.height, self.sort_cfg)
                    if self.keep_roi:
                        det.roi = roi
            except Exception as e:
                det.sort_info = {"error": repr(e)}
        return det

    def _roi(self, color, u, v):
        """Image patch around one blob for shape.classify: dict(rgb, depth, blob, keep, u0, v0). `blob` = the
        connected component of the colour mask (after the geometric keep mask) that holds most of the detection's
        core pixels; `keep` = the geometric keep mask (object-high over the mat, bins removed; arms NOT removed: the arm
        tubes are wide and would bite into a top face that reads a few mm too high). The patch
        is padded by about one object width: the colour mask may cover only one face (washed-out top)."""
        import cv2
        if self._masks is None or self._rgb is None or self._depth is None or color not in self._masks:
            return None
        H, W = self._depth.shape[:2]
        pad = int(self.sort_cfg.get("roi_pad_px", 45))
        u0, u1 = max(int(u.min()) - pad, 0), min(int(u.max()) + pad + 1, W)
        v0, v1 = max(int(v.min()) - pad, 0), min(int(v.max()) + pad + 1, H)
        m = self._masks[color][v0:v1, u0:u1]
        keep = self._keep[v0:v1, u0:u1] if self._keep is not None else np.full(m.shape, 255, np.uint8)
        geo = self._keep_geo[v0:v1, u0:u1] if self._keep_geo is not None else keep
        n, lab = cv2.connectedComponents(((m > 0) & (keep > 0)).astype(np.uint8), connectivity=8)
        if n < 2:
            return None
        cnt = np.bincount(lab[v - v0, u - u0], minlength=n)
        cnt[0] = 0
        blob = (lab == int(cnt.argmax())).astype(np.uint8)
        arm = self._arm[v0:v1, u0:u1] if getattr(self, "_arm", None) is not None else np.full(m.shape, 255, np.uint8)
        # the box row's near edge (pelvis x): an object standing against a box must not grow into its wall
        b = self.scene.cfg["bins"]
        x_max = min(self.scene.mat_to_pelvis(np.r_[np.asarray(c, float), 0.0])[0] for c in b["centers"].values())
        x_max -= float(b["outer_size"][1]) / 2 + 0.004
        return {"rgb": self._rgb[v0:v1, u0:u1], "depth": self._depth[v0:v1, u0:u1], "blob": blob, "keep": geo,
                "arm": arm, "u0": u0, "v0": v0, "W": W, "H": H, "x_max": float(x_max)}


    def masks(self, rgb):
        import cv2
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        H = hsv[..., 0].astype(float)
        sat = (hsv[..., 1] >= self.s_min) & (hsv[..., 2] >= self.v_min)
        # geometric keep mask (bins, counter, arms out) decides which pixels may tune the prototypes
        keep = self._keep_mask(self._depth) > 0
        proto = dict(PROTO)
        if self.adapt:
            bright = hsv[..., 2] >= self.v_min_c["blue"]
            hs = H[sat & keep & bright]
            if len(hs) > 200:
                for _ in range(4):
                    d = np.stack([hue_dist(hs, proto[c]) for c in proto], 1)
                    lab = d.argmin(1)
                    for i, c in enumerate(proto):
                        sel = hs[(lab == i) & (d[:, i] < MAX_DIST[c])]
                        if len(sel) > 80:
                            # circular mean, then clamp the drift so a big orange bin edge can't capture "red"
                            a = np.radians(sel * 2.0)
                            m = (np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2.0) % 180.0
                            drift = hue_dist(m, PROTO[c])
                            if drift <= MAX_DIST[c] * 0.75:
                                proto[c] = float(m)
        self.proto = proto
        d = np.stack([hue_dist(H, proto[c]) for c in proto], -1)
        lab = d.argmin(-1)
        near = np.stack([(d[..., i] < MAX_DIST[c]) | (hue_dist(H, PROTO[c]) < BASE_DIST[c])
                         for i, c in enumerate(proto)], -1)
        out = {}
        k = int(self.cfg["open_px"])
        # Washed-out top faces: under a bright lamp a lit top face drops to S ~50 while the side faces stay
        # saturated (sim seed 412: cylinder top S 51-54, yellow cube top S 52). Then only the side face was masked,
        # the "top" was its upper edge: 15-19 mm short towards the robot, and a cylinder read as a triangle.
        # Pale pixels may only EXTEND a blob of normally saturated pixels of the same colour, and only where the
        # depth puts them at object height above the counter (never the white counter, a bin or an arm).
        pale = None
        if self.s_pale > 0 and self._depth is not None:
            lo, hi = self.cfg["depth_range_mm"]
            valid = (self._depth >= lo) & (self._depth <= hi)
            if valid.mean() >= float(self.cfg.get("min_valid_frac", 0.2)):
                pale = (hsv[..., 1] >= self.s_pale) & (hsv[..., 2] >= self.v_pale) & valid & keep & ~sat
        for i, c in enumerate(proto):
            m = sat & (lab == i) & near[..., i] & (hsv[..., 2] >= self.v_min_c[c])
            if pale is not None and m.any():
                ext = pale & (lab == i) & near[..., i]
                if ext.any():
                    n, cc = cv2.connectedComponents((m | ext).astype(np.uint8), connectivity=4)
                    ok = np.bincount(cc[m], minlength=n) >= self.pale_min_strict
                    ok[0] = False
                    m = m | (ext & ok[cc])
            m = m.astype(np.uint8) * 255
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            out[c] = m
        self._rgb, self._masks = rgb, out
        return out
