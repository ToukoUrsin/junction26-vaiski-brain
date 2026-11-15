"""VaiskiPolicy: the baseline's state machine with our perception and the episode's colour->bin mapping."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.policy import ClassicalPolicy

from .perception import Perception
from .shape import ShapeMemory


def _fast_least_squares():
    """The baseline IK (scipy TRF) polishes to 1e-8/1e-10: ~3x slower planning for sub-0.1 mm gains against a 6 mm
    tolerance. Loosen it (planning time is robot time on the real robot)."""
    import scipy.optimize as so
    if getattr(so.least_squares, "_vaiski", False):
        return
    orig = so.least_squares

    def ls(*a, **k):
        k["xtol"], k["ftol"], k["gtol"] = 1e-5, 1e-6, 1e-6
        return orig(*a, **k)
    ls._vaiski = True
    so.least_squares = ls


_fast_least_squares()

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"
DEBUG_DIR = Path(os.environ["VAISKI_DEBUG_DIR"]) if os.environ.get("VAISKI_DEBUG_DIR") else None
SELF_LINKS = ("shoulder_roll_link", "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link", "hand_palm_link")
# self-mask tube radii (m): forearm + wrist motors are fat, the Dex3 fingers thin
R_ARM, R_HAND, R_FINGER = 0.05, 0.045, 0.022


class VaiskiPolicy(ClassicalPolicy):
    @staticmethod
    def _bench(n=400):
        """CPU seconds for a fixed workload shaped like the IK inner loop (small cross / norm / svd); best of 3."""
        rng = np.random.default_rng(0)
        A, v, w = rng.standard_normal((6, 7)), rng.standard_normal(3), rng.standard_normal(3)
        best = 1e9
        for _ in range(3):
            c0 = time.thread_time()
            for _ in range(n):
                np.cross(v, w)
                np.linalg.norm(v)
                np.linalg.svd(A, full_matrices=False)
            best = min(best, time.thread_time() - c0)
        return best

    def __init__(self, config=None, overrides=None, **kw):
        if os.environ.get("VAISKI_OVERRIDES"):      # experiments: JSON {dotted.key: value}
            import json
            overrides = {**(overrides or {}), **json.loads(os.environ["VAISKI_OVERRIDES"])}
        cfg = load_config(config or (CONFIG if CONFIG.exists() else None), overrides,
                          profile=os.environ.get("AALTO_PROFILE"))
        super().__init__(config=cfg, **kw)
        self.perception = Perception(self.cfg, self.scene, self.cal)
        self._bins_nominal = {k: list(v) for k, v in self.scene.cfg["bins"]["centers"].items()}
        # shape for SORTING, remembered per object over the looks of an episode (shape.py)
        self.shape_mem = ShapeMemory(self.cfg.get("sort_shape", {}) or {})
        self._bin_meas = []
        self._counter_nominal = [float(v) for v in self.scene.cfg["counter"]["mat_origin_in_pelvis"]]
        self._counter_meas, self._counter_tries = [], 0
        self._plane_dz = 0.03       # largest camera height error the per-look plane fit corrects (m)
        self._bin_med, self._cam_mode, self._cam_cur = None, None, None
        self._streak, self._holds, self._flips = [], 0, 0
        self._T0 = np.array(self.cal.cams[self.perception.cam].T_pelvis_optical, float).copy()
        self._Tcorr = self._T0.copy()
        m = self.kin.model
        self._self_frames = {s: [m.getFrameId(f"{s}_{l}") for l in SELF_LINKS if m.existFrame(f"{s}_{l}")]
                             for s in ("left", "right")}
        try:        # how fast this computer is, for the plan search budget (CPU seconds of a fixed workload)
            self._speed = float(np.clip(self._bench() / float(self.cfg["policy"].get("bench_ref_s", 0.025)), 1.0, 4.0))
        except Exception:
            self._speed = 1.0
        self.log(f"[vaiski] computer speed: plan budgets x{self._speed:.2f}")
        self._orig_side_hints = self.planner._plan_side_hints
        self.planner._plan_side_hints = self._plan_side_aligned
        # The strong finger-direction hint of the aligned pinch matters at the object only. Kept through the carry it
        # made the IK at the box fail ("IK failed on ['lower']"), and the plan fell back to the default fingers,
        # which hold a triangle 1 time in 5 (27 % of triangle plans; 48 % at x = 0.32 m).
        self._w_hint0 = self.kin.w_hint
        self._orig_cart, self._orig_jcarry = self.planner._cartesian, self.planner._joint_carry
        self.planner._cartesian = lambda side, p0, p1, q0, name, plan: self._weak_hint(
            name, self._orig_cart, side, p0, p1, q0, name, plan)
        self.planner._joint_carry = lambda side, q_from, p_from, p_to, z_min, name, plan: self._weak_hint(
            name, self._orig_jcarry, side, q_from, p_from, p_to, z_min, name, plan)
        # drop slots: the baseline releases every object at the same point of a bin, so a second one lands ON the
        # first and rests above the rim (scored "counter"): sim 5 of 18 misses
        self._orig_drop_xy, self._orig_plan_side = self.planner.drop_xy, self.planner._plan_side
        self._slot_off = np.zeros(2)
        self._bin_fill = {}
        self._side_dead, self._carry_fails, self._no_direct, self._reach = {}, {}, [], {}
        self._Tplane = None
        self.planner.drop_xy = lambda name: self._orig_drop_xy(name) + self._slot_off
        # horizontal keep-out around the boxes below rim height (planner default 1 cm). Objects whose centre is
        # nearer than ~6.5 cm to a box wall have no pick with 1 cm (the straight spare finger ends up beside the wall).
        rim_margin = float((self.cfg.get("adaptive", {}) or {}).get("rim_margin", 0.01))
        _bin_clearance = self.planner.bin_clearance
        self.planner.bin_clearance = lambda side, a, h, margin=rim_margin: _bin_clearance(side, a, h, margin=margin)
        self.planner._plan_side = self._plan_side_slots

    # ------------------------------------------------------------------ shape-aligned pinch
    def aligned_hints(self, det, side):
        """Finger directions (pelvis xy, unit) along the object's pinch axes that this arm can comfortably take,
        best first. Left arm: fingers between forward (0 deg) and inward (-90 deg); the right arm mirrors."""
        axes = getattr(det, "pinch_axes", None)
        if not axes:
            return []
        g = self.cfg.get("adaptive", {}) or {}
        centre = np.radians(float(g.get("align_centre_deg", -30.0)))
        lo, hi = np.radians(g.get("align_range_deg", [-115.0, 25.0]))
        n_max = int(g.get("align_max_hints", 2))
        if getattr(det, "shape", None) == "lying":
            # A prism on its side holds ONLY when pinched on its two flat ends (default fingers: 0 of 11 in sim), and
            # the organisers' real frames show every prism lying like that. So both directions along its ridge are
            # offered whatever the angle (the comfortable ones still come first): the IK decides what the wrist can do
            # (planning bench: right box 21 -> 29 of 33 layouts, most of them at 0 deg finger error).
            lo, hi = np.radians(g.get("lying_range_deg", [-181.0, 181.0]))
            n_max = int(g.get("lying_max_hints", 4))
        if side == "right":
            centre, lo, hi = -centre, -hi, -lo
        out = []
        # triangle axes point apex -> flat face: fingers on the face, thumb on the apex (see _plan_side_aligned)
        signs = (1.0,) if getattr(det, "shape", None) == "triangle" and g.get("triangle_fingers", "face") == "face" \
            else (1.0, -1.0)
        for a in axes:
            for s in signs:
                d = s * np.asarray(a, float)
                ang = np.arctan2(d[1], d[0])
                if lo <= ang <= hi:
                    out.append((abs(ang - centre), d))
        out.sort(key=lambda t: t[0])
        return [d for _, d in out[:n_max]]

    def _plan_side_aligned(self, det, side, bin_name, arm_q, hand_q, **kw):
        """The baseline's finger directions are fixed (forward, then inward): a triangle pinched across two sloped
        faces, or a cube across its diagonal, slips. Try the object's pinch axes first (strong hint), then the
        baseline directions."""
        ad = self.cfg.get("adaptive", {}) or {}
        mode = getattr(self, "_align_mode", None)      # "only" / "skip": see direct() in _look_two_pass
        hints = self.aligned_hints(det, side) if ad.get("align", True) and mode != "skip" else []
        if not hints and mode == "only":
            from aaltonomous.baselines.classical.planner import Plan
            return Plan(False, side=side, color=det.color, reason=f"no pinch along a flat face for the {side} arm")
        if hints:
            g = self.cfg["grasp"]
            saved_h, saved_w, saved_o = g.get("finger_hints"), self.kin.w_hint, self.kin.offset[side].copy()
            saved_hf = g["height_frac"]
            try:
                mir = np.array([1.0, -1.0, 1.0]) if side == "right" else np.ones(3)
                g["finger_hints"] = [list(np.r_[h, 0.0] * mir) for h in hints]
                self.kin.w_hint = float(ad.get("align_weight", 1.5))
                tp = ad.get("triangle_point_in_palm", [0.091, -0.084, 0.0])
                if getattr(det, "shape", None) == "lying":
                    g["height_frac"] = float(ad.get("lying_height_frac", saved_hf))
                if getattr(det, "shape", None) == "triangle" and tp is not None:
                    # Dex3 three-point grasp: index + middle (57 mm apart) on the 52 mm flat face, thumb on the apex,
                    # all three in line with the centre of mass. The baseline's point (15 mm sideways, thumb vs ONE
                    # finger) twists a triangle out of the hand. Centroid 15 mm behind the face, 30 mm from the apex.
                    self.kin.offset[side] = np.asarray(tp, float) * mir
                    # lower pinch: less tipping torque on the tall prism (sim triangles 11/11 vs 7/11 at 0.5)
                    g["height_frac"] = float(ad.get("triangle_height_frac", saved_hf))
                p = self._orig_side_hints(det, side, bin_name, arm_q, hand_q, **kw)
            finally:
                g["finger_hints"], self.kin.w_hint = saved_h, saved_w
                self.kin.offset[side] = saved_o
                g["height_frac"] = saved_hf
            if p.ok:
                p.aligned = self._finger_error(p, side, hints)
                return p
            if self.planner._late() or mode == "only":
                return p
        p = self._orig_side_hints(det, side, bin_name, arm_q, hand_q, **kw)
        if p.ok or self.planner._late() or not ad.get("rim_sideways", True) \
                or "a bin rim during" not in (p.reason or ""):
            return p
        # An object close to the box row: fingers pointing forward reach into the keep-out around the rims ("hand
        # -0.6 cm above a bin rim during descend", every sim object at x >= 0.33 m in front of a box). With the
        # fingers sideways nothing of the hand is between object and box. The weak default hint does not turn the
        # hand that far; a strong one does (plans up to x = 0.345 m instead of 0.33). Last resort: only after
        # every other finger direction of this arm failed on a rim.
        # Closer still (x 0.35-0.37 m) the hand's spare finger (the pinch is thumb vs ONE finger, the other one sits
        # 42 mm beside the object) is what reaches the box: pinch with the other finger, and the spare one is on the
        # robot's side of the object.
        g = self.cfg["grasp"]
        saved_h, saved_w, saved_o = g.get("finger_hints"), self.kin.w_hint, self.kin.offset[side].copy()
        try:
            g["finger_hints"] = [[0.0, -1.0, 0.0]]      # LEFT arm convention, the right arm mirrors
            self.kin.w_hint = float(ad.get("align_weight", 1.5))
            for other in ((False, True) if ad.get("rim_other_finger", True) else (False,)):
                self.kin.offset[side] = saved_o * ([1.0, 1.0, -1.0] if other else 1.0)
                p2 = self._orig_side_hints(det, side, bin_name, arm_q, hand_q, **kw)
                if p2.ok:
                    p2.sideways = "other finger" if other else "same finger"
                    return p2
                if self.planner._late() or "a bin rim during" not in (p2.reason or ""):
                    break
        finally:
            g["finger_hints"], self.kin.w_hint = saved_h, saved_w
            self.kin.offset[side] = saved_o
        return p

    def _weak_hint(self, name, fn, *a):
        ad = self.cfg.get("adaptive", {}) or {}
        if name not in ("carry", "lower", "retreat") or self.kin.w_hint <= self._w_hint0 \
                or not ad.get("align_grasp_only", True):
            return fn(*a)
        saved, self.kin.w_hint = self.kin.w_hint, self._w_hint0
        try:
            return fn(*a)
        finally:
            self.kin.w_hint = saved

    def drop_slots(self, bin_name):
        """Release offsets (pelvis xy) from the bin's near drop point, in fill order: beside, then further in."""
        ad = self.cfg.get("adaptive", {}) or {}
        by = self.planner.bin_xy(bin_name)[1]
        sl = ad.get("drop_slots_side" if abs(by) > 0.1 else "drop_slots_center")
        if not sl:
            return [np.zeros(2)]
        out = []
        for dx, dy in sl:
            if abs(by) > 0.1:            # side bin: dy > 0 = away from the centre bin
                out.append(np.array([dx, dy * np.sign(by)], float))
            else:                        # centre bin: alternate left/right of the middle
                out.append(np.array([dx, dy], float))
                if dy:
                    out.append(np.array([dx, -dy], float))
        return out

    # failure reasons that say "this arm cannot bring the object to that bin / spot", whatever the finger direction
    _CARRY_FAIL = ("carry: no joint-space path",)
    # ... and reasons that concern the pick itself: another drop slot cannot help
    _PICK_FAIL = ("IK grasp", "IK pregrasp", "fingers hit the counter", "grasp point", "during descend", "during close",
                  "during lift", "during to_pregrasp", "['descend']", "['lift']", "IK jump on descend", "IK jump on lift")

    def _plan_side_slots(self, det, side, bin_name, arm_q, hand_q, place_xy=None):
        """Drop slots + planning-time guards. Planning time is robot time (and sim time): one object that no arm can
        carry to its bin used to cost ~20 s of IK per look (5 finger directions x 4 s, each failing on the same
        carry), look after look (sim seeds 400, 607, 621, 623: 7-19 s lost before every pick)."""
        import time
        from aaltonomous.baselines.classical.planner import Plan
        key = (tuple(np.round(det.top[:2], 4)), side, bin_name,
               None if place_xy is None else tuple(np.round(place_xy, 3)))
        dead = self._side_dead.get(key)
        if dead is None and self.planner._late():
            dead = "planning budget spent"
        if dead is not None:
            return Plan(False, side=side, color=det.color, reason=dead,
                        target_bin=bin_name if place_xy is None else "counter")
        if place_xy is not None or bin_name is None or bin_name == "counter":
            self._slot_off = np.zeros(2)
            p = self._orig_plan_side(det, side, bin_name, arm_q, hand_q, place_xy=place_xy)
        else:
            if not self._bin_reach(side, bin_name):
                return Plan(False, side=side, color=det.color, target_bin=bin_name,
                            reason="carry: no joint-space path above the bin rim (box out of this arm's reach)")
            slots = self.drop_slots(bin_name)
            n = self._bin_fill.get(bin_name, 0) % len(slots)
            p = None
            # wanted slot, then the still empty ones, and only then back onto slot 0 (a second object released on the
            # first one stays on top of it, above the wall, and scores as "on the counter": sim seed 400, twice)
            for k in dict.fromkeys([n] + list(range(n + 1, len(slots))) + [0]):
                self._slot_off = slots[k]
                try:
                    p = self._orig_plan_side(det, side, bin_name, arm_q, hand_q, place_xy=None)
                finally:
                    self._slot_off = np.zeros(2)
                if p.ok:
                    p.slot = k
                    return p
                if self.planner._late():
                    break
                if any(w in (p.reason or "") for w in self._PICK_FAIL):
                    break                      # the pick itself failed: another slot would fail the same way
        if not p.ok and not self.planner.late_hit and any(w in (p.reason or "") for w in self._CARRY_FAIL):
            # a second finger direction is still tried (it changes the wrist pose at the grasp: 4 of 221 feasible
            # cells of the planning bench need it), the third to fifth are not
            self._carry_fails[key] = self._carry_fails.get(key, 0) + 1
            if self._carry_fails[key] >= 2:
                self._side_dead[key] = p.reason
        return p

    def _bin_reach(self, side, bin_name):
        """Can this arm get above that box at all? The arm of the other side cannot (its IK misses the nearest drop
        point of a side box by 5-14 cm), and finding that out the slow way, a failed carry for every finger direction
        and drop slot, cost 10-80 s of planning per object: wall time on the robot. One IK per arm and box, kept
        until the measured box moves 1 cm. Anything within `reach_slack` of reachable gets the full search as before."""
        ad = self.cfg.get("adaptive", {}) or {}
        if not ad.get("reach_check", True):
            return True
        try:
            bxy = np.asarray(self._orig_drop_xy(bin_name), float)
            was = self._reach.get((side, bin_name))
            if was is not None and np.linalg.norm(was[0] - bxy) < 0.01:
                return was[1]
            g, sc = self.cfg["grasp"], self.scene
            z = sc.counter_z + max(g["carry_height"], float(sc.cfg["bins"]["wall_height"]) + self.planner.height + 0.03)
            saved_w, self.kin.w_hint = self.kin.w_hint, self._w_hint0
            try:
                _, ok, err, _ = self.kin.solve(side, np.r_[bxy, z], self.home.copy(), top_down=True,
                                               extra_seeds=[self.home[self.kin.sl(side)]])
            finally:
                self.kin.w_hint = saved_w
            reach = bool(ok or err < float(ad.get("reach_slack", 0.02)))
            self._reach[(side, bin_name)] = (bxy, reach)
            if not reach:
                self.log(f"[vaiski] the {side} arm cannot reach the {bin_name} box (misses by {100 * err:.0f} cm): "
                         f"not planned with it")
            return reach
        except Exception as e:
            self.log(f"[vaiski] reach check failed: {e!r}")
            return True

    def _finger_error(self, plan, side, hints):
        """Angle (deg) between the planned finger direction at the grasp and the nearest wanted pinch direction."""
        try:
            ph = {n: (s_, e) for n, s_, e in plan.phases}
            k = ph["close"][0] if "close" in ph else len(plan.arm) // 2
            self.kin._fk(plan.arm[k])
            f = self.kin.data.oMf[self.kin.palm[side]].rotation[:, 0][:2]
            f = f / (np.linalg.norm(f) + 1e-9)
            return float(min(np.degrees(np.arccos(np.clip(abs(f @ h), -1, 1))) for h in hints))
        except Exception:
            return None

    def reset(self, info=None):
        # real-hardware robustness state (see _clean_obs / _after_look)
        self._good = {}             # last usable measured arm_q / hand_q
        self._hand_meas = True      # False while hand_q in the current obs is a substitute, not a measurement
        self._bad_obs = 0
        self._n_retry = 0           # looks that failed (blank frame, exception)
        self._none_streak = 0       # consecutive looks with nothing to sort
        self._seen_any = False      # an object was detected at least once this episode
        self._n_exc = 0
        self._blind = 0
        self._plan_exc = []         # where the planner raised (transient vs. persistent)
        self._time_note = False
        self._sag_meas = []         # joint compliance measured at rest, one value per look
        self._hard = []             # where a direct plan failed or ran out of time: these objects go last
        self._plan_cpu = []         # (found, CPU s) of every direct plan search this episode
        if hasattr(self, "planner"):
            self.planner.sag_c = 0.0
            self.planner.deadline, self.planner.cancel = None, False
        if hasattr(self, "_bins_nominal") and not isinstance(info, str):
            self.scene.cfg["bins"]["centers"] = {k: list(v) for k, v in self._bins_nominal.items()}
            self.scene.cfg["counter"]["mat_origin_in_pelvis"] = list(self._counter_nominal)
            self.cal.cams[self.perception.cam].T_pelvis_optical = self._T0.copy()
            self._bin_meas = []
            self._counter_meas, self._counter_tries = [], 0
            self._plane_dz = 0.03
            self._Tcorr = self._T0.copy()
            self._bin_fill = {}
            self._bin_med, self._cam_mode, self._cam_cur = None, None, None
            self._streak, self._holds, self._flips = [], 0, 0
            self._side_dead, self._carry_fails, self._no_direct, self._reach = {}, {}, [], {}
            self._Tplane = None
        # set only when the instruction sorts by shape ("cubes left", "red cubes right, the rest center")
        self.parse = info.get("parse") if isinstance(info, dict) else None
        if hasattr(self, "shape_mem") and not isinstance(info, str):
            self.shape_mem.reset()
        # the sorting shape costs ~20 ms a look and is new code: it runs only under an instruction that sorts by shape
        self.perception.sort_on = bool((getattr(self.parse, "by_shape", False)
                                        and self.cfg.get("language", {}).get("shape_rules", True))
                                       or DEBUG_DIR)
        return super().reset(info)

    def sort_shape(self, d):
        """Shape of one detection for SORTING: cube | cylinder | triangle | None (unknown). The label remembered for
        that object over this episode's looks (shape.ShapeMemory); `sort_shape.enabled: false` = the old pinch-axis
        shape. Below `min_conf` it is unknown: a wrong box costs a point, an object left on the counter costs none."""
        sc = self.cfg.get("sort_shape", {}) or {}
        if not sc.get("enabled", True):
            return getattr(d, "shape", None)
        s = getattr(d, "sort_label", None)
        if s is None or float(getattr(d, "sort_label_conf", 0.0) or 0.0) < float(sc.get("min_conf", 0.15)):
            return None
        return s

    def _bin_for(self, d):
        """Bin for one detection; None = leave it on the counter (the instruction says so, names no bin for it,
        or its bin is uncertain). Shape rules need the detected shape: an unknown shape is sorted only when every
        shape it could be goes to the same bin."""
        p = getattr(self, "parse", None)
        if p is not None and self.cfg.get("language", {}).get("shape_rules", True):
            return p.target(d.color, self.sort_shape(d))
        return self.mapping.get(d.color)

    def _Tb(self):
        """Head extrinsics the box measurement starts from: factory, or with the tilt correction of _update_counter."""
        Tp = getattr(self, "_Tplane", None)
        return (self._Tcorr if Tp is None else Tp).copy()

    def _update_counter(self, obs):
        """First look of an episode: fit the counter plane in the head depth image (factory extrinsics).
        scene.yaml's counter height is a (default) -- 0.80 m table minus an ASSUMED 0.79 m pelvis height, robot in a
        gantry -- and the head extrinsics come from the URDF, not from this robot. v8 had two cliffs here (mock world):
        counter 2-3 cm off -> every detection rejected as too low/high -> `done` after 1 s; camera pitched 5 deg ->
        box row 4 cm off -> read as 'boxes moved' -> every grasp 4 cm off. Two independent numbers come out of the plane:
          * its TILT = camera rotation error (the counter is level). Above counter_tilt_min (2.5 deg; sim DR is 1 deg)
            the head extrinsics are rotated about the camera centre until the plane is level; the box row then
            fine-tunes xy as before.
          * the camera's DISTANCE from the plane (unchanged by a rotation error) = counter height. More than
            counter_fix_min (1.5 cm; sim DR moves the camera 1 cm) off: the COUNTER is elsewhere -> the scene moves
            (perception gates, grasp, carry and drop heights follow), by at most counter_fix_up / counter_fix_down
            (down is clamped harder: lowering the grasp on a wrong measurement pushes fingertips into the table).
        Inside both dead bands nothing changes."""
        ad = self.cfg.get("adaptive", {}) or {}
        if not ad.get("counter_fix", True) or self._counter_meas or self._counter_tries >= 5:
            return
        self._counter_tries += 1
        cam = self.perception.cam
        cur = self.cal.cams[cam].T_pelvis_optical
        self.cal.cams[cam].T_pelvis_optical = self._T0
        try:
            r = self.perception.counter_plane(np.asarray(obs["head_rgb"]), np.asarray(obs["head_depth"]),
                                              self._counter_nominal[2])
        finally:
            self.cal.cams[cam].T_pelvis_optical = cur
        if r is None:
            return
        txt = (f"counter plane: camera-to-counter distance {1000 * r['dh']:+.0f} mm vs scene.yaml (+ = counter "
               f"higher), tilt {r['tilt_deg']:.1f} deg (x {r['tilt_x_deg']:+.1f}, y {r['tilt_y_deg']:+.1f}), "
               f"{r['n']} px, {100 * r['frac']:.0f} % inliers, rms {1000 * r['rms']:.1f} mm")
        if r["frac"] < 0.6 or r["rms"] > 0.004:
            self.log(f"[vaiski] {txt}: poor fit, ignored")
            return
        self._counter_meas.append(r)
        if float(ad.get("counter_tilt_min", 2.5)) < r["tilt_deg"] <= float(ad.get("counter_tilt_max", 12.0)):
            n = np.array([-r["sx"], -r["sy"], 1.0])
            n /= np.linalg.norm(n)
            ax = np.cross(n, [0.0, 0.0, 1.0])
            s_, c_ = float(np.linalg.norm(ax)), float(n[2])
            k = ax / s_
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + s_ * K + (1.0 - c_) * (K @ K)          # Rodrigues: turns the seen normal onto +z
            T = self._T0.copy()
            T[:3, :3] = R @ T[:3, :3]                                # about the camera centre (translation kept)
            self._Tcorr = T
            self.cal.cams[cam].T_pelvis_optical = T.copy()
            txt += f" -> head extrinsics rotated by {r['tilt_deg']:.1f} deg"
        dh = r["dh"]
        if float(ad.get("counter_fix_min", 0.015)) < abs(dh) <= float(ad.get("counter_fix_reject", 0.05)):
            # Camera or counter? One depth image cannot tell a camera that sits 2 cm low from a counter that stands
            # 2 cm high, and the wrong reading puts every grasp 2 cm off (sim, camera jitter doubled, seed 422: read as
            # "counter +24 mm", 1 of 4 sorted). What differs is where each can happen: the simulator moves only the
            # camera, never the counter; on the real stand the camera is bolted to the robot and the unknown is how
            # high the pelvis hangs over the table. The arm tells the two apart: the real runner's PD (kp 80, no
            # gravity feed-forward) is 4x softer than the simulator's (12.5 vs 3.1 mrad/Nm, measured at this look).
            # Soft arm, or no measurement: the counter moved. Stiff arm: the camera is off, the counter stays, and
            # the plane fit of every look (_update_bins) corrects the camera's height instead.
            c = float(np.median(self._sag_meas[-5:])) if self._sag_meas else None
            if c is None or c >= float(ad.get("counter_fix_soft", 0.006)):
                shift = float(np.clip(dh, -float(ad.get("counter_fix_down", 0.015)), float(ad.get("counter_fix_up", 0.045))))
                o = list(self._counter_nominal)
                o[2] += shift
                self.scene.cfg["counter"]["mat_origin_in_pelvis"] = o
                txt += f" -> counter height {1000 * shift:+.0f} mm"
            else:
                self._plane_dz = float(ad.get("counter_fix_reject", 0.05))
                txt += (f" -> stiff arm ({1e3 * c:.1f} mrad/Nm): read as the head camera {1000 * abs(dh):.0f} mm "
                        f"{'low' if dh > 0 else 'high'}, counter height kept")
        self.log(f"[vaiski] {txt}")

    def _update_sag(self, obs):
        """Gravity sag of the arms, from the MEASURED joints at rest (every look: the arms have held one command
        for settle_s). A PD joint without gravity feed-forward settles tau_gravity / kp below its command; the
        real runner sends kp 80 and tau_ff 0 (organisers: "the real arm sags ~0.05 rad"), the sim arm is ~4x stiffer.
        One number is estimated, the joint compliance c = (command - measured) / tau_gravity [rad/Nm], pooled over
        shoulders + elbows of both arms. Sim: 0.0031 (max 0.0039 in 158 looks); real PD: 1/80 = 0.0125.
        Below `sag_on` nothing changes. Above `sag_full` the planner commands every waypoint (c - sag_ref) *
        tau_gravity higher, so the hand ends where it ends in the sim the grasp was tuned in; in between the
        correction fades in. Clamped per joint (planner.sag_max)."""
        ad = self.cfg.get("adaptive", {})
        if not ad.get("sag_fix", True) or obs.get("_arm_sub") or self.arm_cmd is None:
            return
        q, cmd = np.asarray(obs["arm_q"], float), np.asarray(self.arm_cmd, float)
        dq = np.asarray(obs.get("arm_dq", np.zeros(14)), float)
        if dq.shape != (14,) or not np.all(np.isfinite(dq)):
            dq = np.zeros(14)
        j = self.planner.SAG_JOINTS
        g, e = self.kin.gravity(q)[j], (cmd - q)[j]
        loaded = np.abs(g) > float(ad.get("sag_min_torque", 1.0))
        ok = loaded & (np.abs(dq[j]) < 0.15)        # joints at rest
        if ok.sum() < 2:                            # a noisy velocity signal must not switch the measurement off:
            ok = loaded                             # the median over the last looks absorbs a moving joint
        if ok.sum() < 2 or float(g[ok] @ g[ok]) < 4.0:
            return
        self._sag_meas.append(float(e[ok] @ g[ok] / (g[ok] @ g[ok])))
        c = float(np.median(self._sag_meas[-5:]))
        on, full = float(ad.get("sag_on", 0.0045)), float(ad.get("sag_full", 0.007))
        ref, cmax = float(ad.get("sag_ref", 0.0032)), float(ad.get("sag_cmax", 0.02))
        w = float(np.clip((c - on) / max(full - on, 1e-9), 0.0, 1.0))   # fades in: no jump at the threshold
        new = w * (min(c, cmax) - ref)
        if len(self._sag_meas) == 1 or abs(new - self.planner.sag_c) > 1e-3:
            self.log(f"[vaiski] arm sag: joint compliance {1e3 * c:.1f} mrad/Nm at rest (sim 3.1, real PD 12.5) -> "
                     + (f"compensating {1e3 * new:.1f} mrad/Nm" if new else "no compensation"))
        self.planner.sag_c, self.planner.sag_max = new, float(ad.get("sag_max", 0.07))

    def _update_bins(self, obs):
        """First looks of an episode: measure the box row with the head camera (perception.locate_bins, always with
        the factory extrinsics T0) and compare it with scene.yaml.
          * small common shift (<= cam_fix_max, boxes agree with each other): the CAMERA is off (head extrinsic
            error; the sim's DR jitters it by ~1 cm / 1 deg) -> shift the head extrinsics so the boxes land where they
            are known to be; every detection (and mask) moves with it.
          * large shift (> cam_fix_max, <= 15 cm): the BOXES are elsewhere -> move the bins to the measurement."""
        if len(self._bin_meas) >= 3:
            return
        cam = self.perception.cam
        ad = self.cfg.get("adaptive", {}) or {}
        rgb, depth = np.asarray(obs["head_rgb"]), np.asarray(obs["head_depth"])
        cur = self.cal.cams[cam].T_pelvis_optical
        base = self._Tcorr.copy()       # factory extrinsics, levelled by _update_counter when the tilt was large
        self.cal.cams[cam].T_pelvis_optical = base
        Tp = base
        try:
            # pitch / roll / height of the camera from the counter plane, then x / y from the box row
            fix = self.perception.counter_plane_fix(rgb, depth, max_dz=self._plane_dz) if ad.get("plane_fix", True) else None
            if fix is not None:
                Tp = fix[0] @ base
                self.cal.cams[cam].T_pelvis_optical = Tp
                if fix[1] > 0.3 or abs(fix[2]) > 0.003:
                    self.log(f"[vaiski] counter plane seen tilted {fix[1]:.1f} deg, {100 * fix[2]:+.1f} cm high "
                             f"-> corrected extrinsics")
            r = self.perception.locate_bins(rgb, depth)
        finally:
            self.cal.cams[cam].T_pelvis_optical = cur
        self._Tplane = Tp.copy()
        if r is None:
            if not self._bin_meas:      # no box measurement yet: at least the heights are right
                self.cal.cams[cam].T_pelvis_optical = Tp.copy()
            self.log("[vaiski] boxes not found in this look")
            return
        sc = self.scene
        # the boxes are a fixed row: a look whose boxes disagree with each other is a bad measurement, not news
        nom0 = {n: sc.mat_to_pelvis(np.r_[np.asarray(c, float), 0.0])[:2] for n, c in self._bins_nominal.items()}
        d0 = np.array([r[n] - nom0[n] for n in r])
        if self._bin_meas and float(np.max(np.linalg.norm(d0 - np.median(d0, 0), axis=1))) > 0.04:
            self.log(f"[vaiski] bin look ignored (boxes disagree by "
                     f"{100 * float(np.max(np.linalg.norm(d0 - np.median(d0, 0), axis=1))):.1f} cm)")
            return
        self._bin_meas.append(r)
        self._bin_med = {n: np.median([m[n] for m in self._bin_meas], 0) for n in r}
        self._apply_bins()

    def _apply_bins(self, quiet=False):
        """Use the measured box row (self._bin_med) under the current hypothesis: 'camera' (head extrinsics off: shift
        them so the boxes land where they are known to be) or 'boxes' (the boxes stand elsewhere: move the bins).
        self._cam_mode None = decide by the size of the shift (cam_fix_max)."""
        sc, cam, med = self.scene, self.perception.cam, self._bin_med
        nom = {n: sc.mat_to_pelvis(np.r_[np.asarray(c, float), 0.0])[:2] for n, c in self._bins_nominal.items()}
        d = np.array([med[n] - nom[n] for n in med])
        s = np.median(d, 0)
        spread = float(np.max(np.linalg.norm(d - s, axis=1)))
        self._bin_shift, self._bin_spread = s, spread
        ad = self.cfg.get("adaptive", {}) or {}
        cam_max = float(ad.get("cam_fix_max", 0.03))
        txt = ", ".join(f"{n} ({c[0]:+.3f},{c[1]:+.3f})" for n, c in med.items())
        if getattr(self.perception, "bins_how", "colour") != "colour":
            txt += " (by shape)"
        mode = self._cam_mode or ("camera" if np.linalg.norm(s) <= cam_max and spread <= 0.02 else "boxes")
        # A stiff arm is the simulator's (see _update_counter), and the simulator moves only the camera, never the
        # boxes: there the row is read as "camera off" up to 5 cm, whatever the size rule or the failed grasps say.
        # Measured before this rule, 80 episodes: 4 switched to 'boxes' after two slipped grasps and 3 started as
        # 'boxes' at a 3.0-4.3 cm shift; every one was a camera error, and after it nearly every grasp came up empty.
        if (self._stiff_arm() and np.linalg.norm(s) <= float(ad.get("cam_flip_shift", [0.015, 0.05])[1])
                and spread <= 0.02):
            mode = "camera"
        self._cam_cur = mode
        if mode == "camera":
            # The camera error has a turn too (spec: +-1 deg). Seen from above it turns the whole picture about the
            # camera: the shift measured at the box row (0.5 m out) is then wrong by turn x 0.2 m for the objects
            # (0.3 m out): 3.5 mm sideways per degree, the largest part of the remaining sideways error in sim
            # (16 episodes: row turn vs true camera yaw r = 0.99). The row's turn shows as the left box seen
            # further away than the right one. Undo turn and shift together, about the middle of the row.
            yaw = self._row_yaw(med, nom) if ad.get("cam_fix_yaw", True) else 0.0
            c = np.mean([nom[n] for n in med], 0)
            Rz = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])     # rotation by -yaw
            A = np.eye(4)
            A[:2, :2] = Rz
            A[:2, 3] = c - Rz @ (c + s)
            self.cal.cams[cam].T_pelvis_optical = A @ self._Tb()
            sc.cfg["bins"]["centers"] = {k: list(v) for k, v in self._bins_nominal.items()}
            if not quiet:
                self.log(f"[vaiski] bins seen at {txt}: head camera off by ({100 * s[0]:+.1f},{100 * s[1]:+.1f}) cm, "
                         f"{np.degrees(yaw):+.1f} deg -> corrected extrinsics")
        else:
            self.cal.cams[cam].T_pelvis_optical = self._Tb()
            sc.cfg["bins"]["centers"] = {n: [float(v) for v in sc.pelvis_to_mat(np.r_[med[n], sc.counter_z])[:2]]
                                         for n in med}
            if not quiet:
                self.log(f"[vaiski] bins seen at {txt} (shift {100 * np.linalg.norm(s):.1f} cm, spread "
                         f"{100 * spread:.1f} cm): boxes moved -> using the measured bins")

    def _row_yaw(self, med, nom):
        """Turn (rad, about the vertical) of the measured box row against the nominal one; 0 unless all three boxes
        were seen and the turn is small enough to be a camera error (cam_fix_yaw_max)."""
        if len(med) < 3:
            return 0.0
        P = np.array([med[n] for n in med], float)
        N = np.array([nom[n] for n in med], float)
        P, N = P - P.mean(0), N - N.mean(0)
        yaw = float(np.arctan2(np.sum(N[:, 0] * P[:, 1] - N[:, 1] * P[:, 0]), np.sum(N * P)))
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        res = float(np.max(np.linalg.norm(N @ R.T - P, axis=1)))      # a row that is not rigid is not a camera turn
        lim = np.radians(float((self.cfg.get("adaptive", {}) or {}).get("cam_fix_yaw_max_deg", 3.0)))
        return yaw if abs(yaw) <= lim and res <= 0.006 else 0.0

    def _stiff_arm(self):
        """True when the arm measures as stiff as the simulator's (median of the last looks under counter_fix_soft).
        False without a measurement, and on the real robot, whose arm is 4x softer."""
        ad = self.cfg.get("adaptive", {}) or {}
        return bool(self._sag_meas) and float(np.median(self._sag_meas[-5:])) < float(ad.get("counter_fix_soft", 0.006))

    def _grasp_result(self, held):
        """One call per pick attempt once its outcome is known (held to the release, or empty / lost).
        Camera-or-boxes self-correction: the box row seen 1.5-5 cm from where scene.yaml puts it can mean "head
        camera off" (then every detection must be shifted) or "boxes moved" (then none). The first guess goes by the
        size of the shift (cam_fix_max); a wrong guess puts EVERY grasp 1-3 cm off (sim seed 605: shift 3.0 cm read
        as "boxes moved", 12 misses of 12). Two empty grasps in a row on two different objects and NO hold yet under
        the current guess: switch to the other guess (at most twice per episode). One hold proves the guess: a
        looser rule (<= 1 hold) flipped a correct guess after two ordinary misses (sim seed 412)."""
        ex = getattr(self, "_exec", None)
        g = ex["plan"].waypoints.get("grasp") if ex else None
        if held:
            self._streak, self._holds = [], self._holds + 1
            return
        if g is not None:
            self._streak.append(np.asarray(g[:2], float))
        ad = self.cfg.get("adaptive", {}) or {}
        n = int(ad.get("cam_flip_after", 2))
        if n <= 0 or self._bin_med is None or self._flips >= 2 or self._holds > 0 or len(self._streak) < n:
            return
        if self._stiff_arm():
            return                      # the simulator: the boxes never move, the guess stays (see _apply_bins)
        last = self._streak[-n:]
        if min(np.linalg.norm(a - b) for i, a in enumerate(last) for b in last[i + 1:]) < float(ad.get("cam_flip_apart", 0.05)):
            return                      # the same object failing again says nothing about the camera
        lo, hi = ad.get("cam_flip_shift", [0.015, 0.05])
        if not (lo <= float(np.linalg.norm(self._bin_shift)) <= hi) or self._bin_spread > 0.02:
            return
        self._cam_mode = "boxes" if self._cam_cur == "camera" else "camera"
        self._flips += 1
        self._streak, self._holds = [], 0
        self.log(f"[vaiski] {n} empty grasps in a row on different objects with the box row "
                 f"{100 * np.linalg.norm(self._bin_shift):.1f} cm off: switching the guess to '{self._cam_mode}' "
                 f"({'head camera off: shift every detection' if self._cam_mode == 'camera' else 'boxes moved: detections as seen'})")
        self._apply_bins()
        # the tried spots were wrong places: let the objects be tried again
        self.attempts, self.failed = [], []

    def body_chains(self, arm_q, hand_q):
        """[(pelvis-frame polyline, tube radius m)] covering both arms and hands, for self-masking."""
        chains = []
        for s in ("left", "right"):
            hp = self.kin.hand_points(s, arm_q, hand_q)   # runs FK: palm, grasp, 3 finger links, 3 tips
            links = np.vstack([self.kin.data.oMf[f].translation for f in self._self_frames[s]])
            chains.append((links, R_ARM))
            chains.append((np.vstack([hp[0], hp[1]]), R_HAND))
            nf = (len(hp) - 2) // 2
            for i in range(nf):
                chains.append((np.vstack([hp[0], hp[2 + i], hp[2 + nf + i]]), R_FINGER))
        return chains

    def _frame(self, obs):
        """(rgb, depth, None) for a usable head frame, else (None, None, why). The runner hands out an all-zero image
        when a camera has no frame yet; v8 read that as 'counter empty' and ended the episode after 1 s."""
        rgb = obs.get("head_rgb") if hasattr(obs, "get") else None
        try:
            rgb = np.asarray(rgb)
        except Exception:
            rgb = None
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 60 or rgb.shape[1] < 80:
            return None, None, "no head image"
        if rgb.dtype != np.uint8:
            rgb = np.clip(np.nan_to_num(rgb.astype(float)), 0, 255).astype(np.uint8)
        g = rgb[::8, ::8]
        if int(g.max()) < 40 or float(g.std()) < 4.0:
            return None, None, f"blank head image (max {int(g.max())}, std {float(g.std()):.1f})"
        c = self.cal.cams[self.perception.cam]
        hw = (int(c.height), int(c.width))
        try:
            if rgb.shape[:2] != hw:      # the calibration is for this size only
                import cv2
                rgb = cv2.resize(rgb, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
            d = np.asarray(obs.get("head_depth"))
            if d.ndim == 3:
                d = d[..., 0]
            if d.ndim != 2 or d.size == 0:
                raise ValueError("no depth")
            if d.dtype.kind == "f":
                d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                if 0.0 < float(d.max()) < 20.0:     # metres instead of the documented uint16 millimetres
                    d = d * 1000.0
            if d.shape != rgb.shape[:2]:
                import cv2
                d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        except Exception:
            d = np.zeros(rgb.shape[:2], np.uint16)     # no depth = the plane fallback of the perception
        return np.ascontiguousarray(rgb), d, None

    def _look(self, obs):
        rgb, depth, why = self._frame(obs)
        if rgb is None:
            return ("retry", why, 0)
        obs = dict(obs)
        obs["head_rgb"], obs["head_depth"] = rgb, depth
        try:
            aq = obs.get("arm_q", self.arm_cmd) if hasattr(obs, "get") else self.arm_cmd
            hq = obs.get("hand_q", self.hand_cmd) if hasattr(obs, "get") else self.hand_cmd
            self.perception.self_chains = self.body_chains(np.asarray(aq, float), np.asarray(hq, float))
        except Exception as e:  # never block a look on the self mask
            self.log(f"[vaiski] self mask failed: {e!r}")
            self.perception.self_chains = None
        try:
            self._update_sag(obs)       # before the counter: how soft the arm is decides how a height error is read
        except Exception as e:
            self.planner.sag_c = 0.0
            self.log(f"[vaiski] arm sag measurement failed: {e!r}")
        try:
            self._update_counter(obs)
        except Exception as e:
            self.log(f"[vaiski] counter measurement failed: {e!r}")
        try:
            self._update_bins(obs)
        except Exception as e:
            self.log(f"[vaiski] bin measurement failed: {e!r}")
        out = self._look_two_pass(obs)
        if DEBUG_DIR:
            try:
                self._dump(obs, out)
            except Exception as e:
                self.log(f"[vaiski] debug dump failed: {e!r}")
        return out

    def _look_two_pass(self, obs):
        """Baseline look, but direct picks of ALL candidates are tried before any (slow, ~30 s of IK) relocation,
        and both searches have a budget in CPU seconds (see below)."""
        self.planner.deadline = None
        dets = self.perception.detect(obs["head_rgb"], obs["head_depth"])
        if getattr(self.perception, "sort_on", False):
            try:
                self.shape_mem.update(dets)
                self.log("[vaiski] shapes: " + "; ".join(
                    f"{d.color} ({d.top[0]:.3f},{d.top[1]:+.3f}) look={getattr(d, 'sort_shape', None)}"
                    f"/{getattr(d, 'sort_conf', 0.0):.2f} label={self.sort_shape(d)}/{getattr(d, 'sort_label_conf', 0.0):.2f}"
                    f" why={(getattr(d, 'sort_info', None) or {}).get('why')} -> {self._bin_for(d)}" for d in dets))
            except Exception as e:      # never block a look on the sorting shape
                self.log(f"[vaiski] shape memory failed: {e!r}")
        self.last_detections = dets
        pc = self.cfg["policy"]
        R = pc["same_object_radius"]
        cands = []
        for d in dets:
            xy = d.top[:2]
            if any(np.linalg.norm(xy - f) < R for f in self.failed):
                continue
            if sum(c for p, c in self.attempts if np.linalg.norm(xy - p) < R) >= pc["max_attempts"]:
                continue
            if self._bin_for(d) is None:
                continue
            cands.append(d)

        # an object that keeps failing creeps (each miss nudges it), escaping the same_object_radius count, and as
        # the nearest-to-its-bin candidate it was picked every time: one tipped triangle ate a whole episode (sim
        # seed 100). Fewest tries nearby first, so the easy objects go before the hard one gets more chances.
        R2 = float(pc.get("retry_radius", 0.08))

        def tries(d):
            return sum(c for p, c in self.attempts if np.linalg.norm(d.top[:2] - p) < R2)
        cands = [d for d in cands if tries(d) < int(pc.get("max_tries_area", 6))]

        # Thinking time. In sim a look that takes 30 s costs 7 s of the episode (sim clock 0.22x); on the robot it
        # costs 30 s of the 120. v8 sim logs: 830 looks, median 1.0-1.5 s, 43 over 3 s, 9 over 10 s, max 49.5 s; an
        # object without a direct plan was searched in full (30-90 s when both arms fail late) again at every look,
        # before the objects that can be sorted. So: (1) an object whose direct plan failed goes behind the others
        # at later looks; (2) while other objects are still untried, one object gets `plan_budget_s` of search
        # (CPU seconds of this thread, so a busy machine does not shrink it); one that runs out is put aside and
        # gets the full search when nothing else can be planned. Nothing is given up earlier than before.
        def hard(d):
            return any(np.linalg.norm(d.top[:2] - q) < R for q in self._hard)

        def key(d):
            return (hard(d), tries(d), np.linalg.norm(d.top[:2] - self.planner.bin_xy(self._bin_for(d))))
        cands.sort(key=key)
        budget_p = float(pc.get("plan_budget_s", 12.0)) * getattr(self, "_speed", 1.0)   # slower computer: longer

        def direct(d, others, budget):
            """(plan, ran out of time); plan None = the planner raised."""
            pl = self.planner
            c0 = pl.clock()
            pl.late_hit, pl.deadline = False, (c0 + budget if budget else None)
            # A prism (upright or lying) holds only when pinched along its flat faces: default fingers held 1 of 23
            # in sim, the aligned pinch 57 of 71. So the other arm's aligned pinch comes before this arm's default
            # fingers: first both arms aligned only, then both arms with the default directions.
            two = getattr(d, "shape", None) in ("triangle", "lying") and bool(getattr(d, "pinch_axes", None)) \
                and (self.cfg.get("adaptive", {}) or {}).get("aligned_first", True)
            try:
                self._align_mode = "only" if two else None
                if two:     # its own short budget: a cross-body carry that cannot work takes 10 s+ to rule out
                    end = c0 + float(pc.get("aligned_budget_s", 6.0)) * getattr(self, "_speed", 1.0)
                    pl.deadline = end if pl.deadline is None else min(pl.deadline, end)
                p = pl.plan(d, self.arm_cmd, self.hand_cmd, bin_name=self._bin_for(d), obstacles=others, relocate=False)
                if two and not p.ok:
                    pl.late_hit, pl.deadline = False, (c0 + budget if budget else None)
                    self._align_mode = "skip"
                    p = pl.plan(d, self.arm_cmd, self.hand_cmd, bin_name=self._bin_for(d), obstacles=others,
                                relocate=False)
            except Exception as e:      # one object the planner chokes on must not end the episode
                self._plan_exc.append(d.top[:2])
                again = sum(np.linalg.norm(d.top[:2] - q) < R for q in self._plan_exc)
                self.log(f"[vaiski] planner failed on {d.color} at {d.top[:2].round(3)}: {e!r} (#{again} here)")
                if again >= 3:          # not a transient: stop offering this object
                    self.failed.append(d.top[:2])
                return None, False
            finally:
                pl.deadline, self._align_mode = None, None
            self.plans.append(p)
            c = pl.clock() - c0
            self._plan_cpu.append((bool(p.ok), c))
            if p.ok:
                a = getattr(p, "aligned", None)
                self.log(f"[vaiski] {d.color} {getattr(d, 'shape', None)} {getattr(d, 'shape_info', None)}: "
                         + (f"pinch axis aligned, finger error {a:.0f} deg" if a is not None else
                            f"fingers sideways, {p.sideways} (close to a box)" if getattr(p, "sideways", False)
                            else "default fingers")
                         + f" (planned in {c:.1f} s)")
            return p, (not p.ok and pl.late_hit)

        need_reloc, late = [], []
        self._side_dead, self._carry_fails = {}, {}
        R3 = float(pc.get("replan_radius", 0.012))
        for i, d in enumerate(cands):
            others = [o.top[:2] for o in dets if o is not d]
            bn = self._bin_for(d)
            # an object that had no direct plan (full search) and has not moved since has none now: straight to the
            # relocation pass (the direct plan does not depend on the other objects, only the relocation does)
            known = next((w for xy, b, w in self._no_direct if b == bn and np.linalg.norm(d.top[:2] - xy) < R3), None)
            if known is not None:
                need_reloc.append((d, others, known))
                continue
            # the budget only buys time for objects still to come: the last one, and one that already failed
            # before (then everything easier was tried first in this look), get the full search straight away
            p, timed = direct(d, others, budget_p if (i < len(cands) - 1 and not hard(d)) else None)
            if p is None:
                continue
            if p.ok:
                return ("plan", p, len(dets))
            if not hard(d):
                self._hard.append(np.array(d.top[:2], float))
            if timed:
                self.log(f"[vaiski] {d.color} at {d.top[:2].round(3)}: no plan within {budget_p:.0f} s, "
                         f"trying the other objects first")
                late.append((d, others))
            else:
                self._no_direct.append((d.top[:2].copy(), bn, p.reason))
                self.log(f"[vaiski] no direct plan for {d.color} at ({d.top[0]:+.3f},{d.top[1]:+.3f}): {p.reason}")
                need_reloc.append((d, others, p.reason))
        for d, others in late:          # nothing else can be planned: the full search these did not get yet
            p, _ = direct(d, others, None)
            if p is None:
                continue
            if p.ok:
                return ("plan", p, len(dets))
            self._no_direct.append((d.top[:2].copy(), self._bin_for(d), p.reason))
            self.log(f"[vaiski] no direct plan for {d.color} at ({d.top[0]:+.3f},{d.top[1]:+.3f}): {p.reason}")
            need_reloc.append((d, others, p.reason))
        t0 = self.planner.clock()
        budget = float(pc.get("relocate_budget_s", 12.0)) * getattr(self, "_speed", 1.0)
        for d, others, why in need_reloc:
            if self.planner.clock() - t0 > budget:
                self.log(f"[vaiski] relocation budget spent, giving up on {d.color} at {d.top[:2].round(3)}")
                self.failed.append(d.top[:2])
                continue
            try:    # v8 checked the budget only between objects, and one relocation search takes 13 s+ of CPU:
                    # on the robot that is wall time of the 120 s. The deadline stops the search itself.
                self.planner.deadline = t0 + budget
                p = self.planner.plan_relocation(d, self._bin_for(d), self.arm_cmd, self.hand_cmd, others)
            except Exception as e:
                self._plan_exc.append(d.top[:2])
                again = sum(np.linalg.norm(d.top[:2] - q) < R for q in self._plan_exc)
                self.log(f"[vaiski] relocation planner failed on {d.color} at {d.top[:2].round(3)}: {e!r} (#{again} here)")
                if again >= 3:
                    self.failed.append(d.top[:2])
                continue
            finally:
                self.planner.deadline = None     # direct plans are never cut short
            self.plans.append(p)
            if p.ok:
                return ("plan", p, len(dets))
            self.failed.append(d.top[:2])
            self.log(f"[classical] skip {d}: {why}; relocate: {p.reason} ({self.planner.clock() - t0:.1f} s)")
        return ("none", None, len(dets))

    # ------------------------------------------------------------------ empty-grasp abort
    def _start(self, plan, t):
        self._exec = {"plan": plan, "n": len(plan.arm), "checked": False}
        return super()._start(plan, t)

    def _check_grasp(self, obs):
        """After the close: if the fingers track the closed command almost exactly, nothing is between them
        (sim: empty < 0.005 rad once settled, holding 0.02-0.07 rad). Then skip the rest: (finish the lift,) open,
        straight home. The threshold errs on the safe side: a hand that never tracks that well (real Dex3 offsets)
        never aborts.
        The check runs at EVERY act() from the end of the close until the release: at the end of the close the
        fingers are often still settling (sim: 19 of 72 "holding" verdicts were an empty hand whose fingers reached
        the command 0.3 s later, each costing a full carry), and an object can also slip out during the lift."""
        ex = getattr(self, "_exec", None)
        if not ex or ex["checked"] or self.queue_a is None or "hand_q" not in obs:
            return
        plan = ex["plan"]
        ph = {n: (s_, e) for n, s_, e in plan.phases}
        if "close" not in ph or "lift" not in ph or "open" not in ph or plan.target_bin == "counter":
            ex["checked"] = True
            return
        done = ex["n"] - len(self.queue_a)
        if done < ph["close"][1]:
            return
        ad = self.cfg.get("adaptive", {}) or {}
        was = ex.get("verdict")
        if was is None:
            # this plan (most likely) fills its drop slot: the next object for that bin goes to the next slot
            self._bin_fill[plan.target_bin] = self._bin_fill.get(plan.target_bin, 0) + 1
        recheck = bool(ad.get("grasp_recheck", True))
        if done >= ph["open"][0] - int(ad.get("grasp_recheck_margin", 12)) or (was == "holding" and not recheck) \
                or (was is None and done >= ph["lift"][1] - 2 and not recheck):
            ex["checked"] = True          # about to release anyway (or single-check mode): the verdict stands
            held = was != "empty"
            if not held:
                self._bin_fill[plan.target_bin] -= 1
            self._grasp_result(held)
            return
        sl = slice(0, 7) if plan.side == "left" else slice(7, 14)
        hq = np.asarray(obs["hand_q"], float)[sl]
        hc = plan.hand[ph["close"][1] - 1][sl]
        err = float(np.abs(hq - hc)[3:].max())
        thr = float(ad.get("empty_grasp_rad", 0.010))
        if not self._hand_meas or not np.isfinite(err):
            if was is None:
                ex["verdict"] = "holding"     # cannot tell: carry on with the plan
                self.log("[vaiski] grasp check: no usable finger measurement -> keep the plan")
            return
        if err >= thr:
            ex["low"] = 0
            if was is None:
                ex["verdict"] = "holding"
                self.log(f"[vaiski] grasp check: fingers blocked by {err:.3f} rad -> holding {plan.color}")
            return
        ex["low"] = ex.get("low", 0) + 1
        if was == "holding" and ex["low"] < int(ad.get("grasp_recheck_n", 3)):
            return                         # a hold is called lost only on 3 readings in a row (0.1 s): encoder noise
        ex["verdict"] = "empty"
        rest = max(0, ph["lift"][1] - done)
        a_l, h_l = self.queue_a[:rest], self.queue_h[:rest]
        a0 = a_l[-1] if len(a_l) else self.arm_cmd
        h0 = h_l[-1] if len(h_l) else self.hand_cmd
        a_h, h_h = self.planner.go_to(a0, h0, self.home.copy(), self.post["open"].copy())
        A, Hh = np.vstack([a_l, a_h]) if len(a_l) else a_h, np.vstack([h_l, h_h]) if len(h_l) else h_h
        side = plan.side
        for k in range(0, len(A), 3):
            if self.planner.clearance(side, A[k], Hh[k]) < self.cfg["motion"]["counter_clearance"] - 2e-3 \
                    or self.planner.bin_clearance(side, A[k], Hh[k]) < 0.003:
                if was != "empty":
                    self.log(f"[vaiski] grasp check: empty ({err:.3f} rad) but the short way home is not clear: keep plan")
                if not recheck:
                    ex["checked"] = True
                    self._bin_fill[plan.target_bin] -= 1
                    self._grasp_result(False)
                return                     # (recheck: try again from further along the carry)
        ex["checked"] = True
        saved = (len(self.queue_a) - len(A)) / 30.0
        if was == "holding":
            self.log(f"[vaiski] grasp check: {plan.color} lost after the close ({err:.3f} rad) -> abort, straight home "
                     f"(saves {saved:.1f} s)")
        else:
            self.log(f"[vaiski] grasp check: empty hand ({err:.3f} rad) -> abort {plan.color}, straight home "
                     f"(saves {saved:.1f} s)")
        self.queue_a, self.queue_h = A, Hh
        self._bin_fill[plan.target_bin] -= 1
        self._grasp_result(False)

    # ------------------------------------------------------------------ never freeze, throw or stop early
    def _clean_obs(self, obs):
        """A dropped DDS sample must not reach the planner as NaN, a spike or a missing key: unusable joint vectors
        are replaced by the last good measurement (else the last command). None = no joint state at all yet."""
        if not isinstance(obs, dict):
            return obs
        o = dict(obs)
        sp = self.spec
        self._hand_meas = True
        for key, lo, hi, cmd in (("arm_q", sp.arm_lower, sp.arm_upper, self.arm_cmd),
                                 ("hand_q", sp.hand_lower, sp.hand_upper, self.hand_cmd)):
            try:
                q = np.asarray(obs.get(key), float)
                ok = q.shape == lo.shape and bool(np.all(np.isfinite(q))) and bool(np.all((q > lo - 0.6) & (q < hi + 0.6)))
            except Exception:
                ok = False
            if ok:
                self._good[key] = q.copy()
            else:
                self._bad_obs += 1
                if key == "hand_q":
                    self._hand_meas = False
                else:
                    o["_arm_sub"] = True   # not a measurement: do not estimate the arm sag from it
                q = self._good.get(key, cmd)
                if self._bad_obs in (1, 10, 100, 1000):
                    self.log(f"[vaiski] unusable {key} in the observation (#{self._bad_obs}): using the last good value")
                if q is None:
                    return None
            o[key] = q
        return o

    def _wait_then_look(self, wait_s):
        n = max(1, int(round(wait_s * 30.0)))
        self._enqueue(np.repeat(self.arm_cmd[None], n, 0), np.repeat(self.hand_cmd[None], n, 0))
        self.state = "look"
        return self._pop()

    def _after_look(self, res, t):
        """The baseline answers a failed look, an empty look and 'not enough time' with done=True, which ends the
        episode for good. On the real robot each of those can be a transient (camera not streaming yet, a dropped
        frame, a hand in front of the lens, an unknown time limit), so: look again, and say done only after
        `done_confirm_looks` consecutive healthy looks with nothing to sort, in an episode where objects were seen."""
        kind = res[0]
        pc = self.cfg["policy"]
        relook = float(pc.get("relook_s", 0.5))
        if kind in ("error", "retry"):
            self._n_retry += 1
            if self._n_retry <= 3 or self._n_retry % 20 == 0:
                self._note(t, f"look failed ({res[1]}): holding and looking again (#{self._n_retry})")
            return self._wait_then_look(relook)
        if kind == "none":
            self._none_streak += 1
            self._seen_any = self._seen_any or res[2] > 0
            if not self._seen_any or self._none_streak < int(pc.get("done_confirm_looks", 3)):
                if self._none_streak in (1, 2) or self._none_streak % 20 == 0:
                    self._note(t, f"nothing to sort in this look ({res[2]} detections, #{self._none_streak}): looking again")
                return self._wait_then_look(relook)
            return super()._after_look(res, t)
        self._none_streak, self._seen_any = 0, True
        plan = res[1]
        if t + plan.duration > self.time_limit:
            if t > self.time_limit + 1.0:   # still running: the real limit is longer than we were told
                self._note(t, f"episode runs past the assumed {self.time_limit:.0f} s limit: carrying on")
                self.time_limit = float("inf")
            else:                           # wait at home; no `done` (the finals' time limit is not announced)
                if not self._time_note:
                    self._note(t, f"not enough time for {plan.color} ({plan.duration:.0f} s): waiting at home")
                    self._time_note = True
                self.state = "idle_end"
                return self._hold()
        return super()._after_look(res, t)

    def act(self, obs):
        """Never raise (an error reply for 1 s ends the episode) and never answer with targets built on garbage."""
        try:
            clean = self._clean_obs(obs)
            if clean is None:               # no joint state yet: do not move blind (the safety filter holds)
                self._blind += 1
                if self._blind in (1, 30):
                    self.log("[vaiski] no usable joint state in the first observation: not commanding anything")
                nan = np.full((1, 14), np.nan)
                return {"arm_q_target": nan, "hand_q_target": nan.copy(), "done": False}
            if self.state == "idle_end":
                t = float(clean.get("t", 0.0)) if isinstance(clean, dict) else 0.0
                if not t > self.time_limit + 1.0:
                    return self._hold()
                self._note(t, f"episode runs past the assumed {self.time_limit:.0f} s limit: carrying on")
                self.time_limit = float("inf")
                self.state = "settle"
            return self._act(clean)
        except Exception as e:
            self._n_exc += 1
            if self._n_exc <= 5 or self._n_exc % 50 == 0:
                self.log(f"[vaiski] act() failed in state {self.state} (#{self._n_exc}): {e!r}")
            if self.arm_cmd is None:
                nan = np.full((1, 14), np.nan)
                return {"arm_q_target": nan, "hand_q_target": nan.copy(), "done": False}
            if self.queue_a is not None and len(self.queue_a):
                try:                        # a validated plan is in flight: keep executing it
                    return self._pop()
                except Exception:
                    self.queue_a = self.queue_h = None
            self._job = None
            if self.state not in ("init", "home", "await_home") or self._n_exc > 5:
                self.state = "settle"       # look again (from wherever the arms are)
            return self._hold()

    def _act(self, obs):
        if isinstance(obs, dict) and self.state == "executing":
            try:
                self._check_grasp(obs)
            except Exception as e:
                self.log(f"[vaiski] grasp check failed: {e!r}")
        if DEBUG_DIR and isinstance(obs, dict) and "head_rgb" in obs and self.state == "executing":
            self._n_act = getattr(self, "_n_act", 0) + 1
            ex = getattr(self, "_exec", None)
            wk = f"{ex['plan'].side}_wrist_rgb" if ex else None
            if os.environ.get("VAISKI_DEBUG_WRIST") and wk and wk in obs:
                try:
                    import cv2
                    n = getattr(self, "_n_dump", 0)
                    done = ex["n"] - (len(self.queue_a) if self.queue_a is not None else 0)
                    ph = next((nm for nm, s_, e in ex["plan"].phases if s_ <= done < e), "?")
                    im = cv2.resize(np.asarray(obs[wk])[..., ::-1], (320, 240))
                    cv2.putText(im, f"{ph} {done}", (5, 15), 0, 0.5, (255, 255, 255), 1)
                    cv2.imwrite(str(DEBUG_DIR / f"wrist_{n:04d}_{self._n_act:05d}.jpg"), im)
                except Exception:
                    pass
            if self._n_act % 3 == 0:   # ~1 frame/s of the execution (act() is called once per 10-tick chunk)
                try:
                    import cv2
                    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                    n = getattr(self, "_n_dump", 0)
                    cv2.imwrite(str(DEBUG_DIR / f"exec_{n:04d}_{self._n_act:05d}.jpg"),
                                cv2.resize(np.asarray(obs["head_rgb"])[..., ::-1], (320, 240)))
                except Exception:
                    pass
        return super().act(obs)

    def _dump(self, obs, out):
        import cv2
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        self._n_dump = getattr(self, "_n_dump", 0) + 1
        rgb = np.asarray(obs["head_rgb"])
        vis = rgb[..., ::-1].copy()
        sm = self.perception.self_mask(rgb.shape[:2])
        vis[sm == 0] = (vis[sm == 0] * 0.6).astype(np.uint8)
        col = {"red": (0, 0, 255), "yellow": (0, 255, 255), "blue": (255, 0, 0)}
        for d in self.last_detections or []:
            u, v = int(d.uv[0]), int(d.uv[1])
            cv2.circle(vis, (u, v), 6, col.get(d.color, (255, 255, 255)), 2)
            cv2.putText(vis, f"{d.color[0]} {d.top[0]:.2f},{d.top[1]:.2f}", (u + 8, v), 0, 0.4, (255, 255, 255), 1)
        plan = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        g = (getattr(plan, "waypoints", None) or {}).get("grasp")
        if g is not None:
            p = self.cal.project("head", np.asarray(g, float)[None, :3])[0]
            cv2.drawMarker(vis, (int(p[0]), int(p[1])), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.imwrite(str(DEBUG_DIR / f"look_{self._n_dump:04d}.jpg"), vis)
        np.savez_compressed(DEBUG_DIR / f"look_{self._n_dump:04d}.npz", rgb=rgb, depth=np.asarray(obs["head_depth"]),
                            arm_q=np.asarray(obs.get("arm_q", [])), hand_q=np.asarray(obs.get("hand_q", [])))
