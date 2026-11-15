"""Mock world + our server policy in-process (like mock_eval.py) with FAULT INJECTION on the observations.

    python aaltonomous/vaiski/checks/fault_eval.py --fault black_first --seeds 0 1 2 [--task sort_language]

Per seed: score / max, clipped steps, end reason, time `done` was sent, exceptions raised by act(), slowest act().
A fault is fine when: no exception, 0 clips, no early `done` with objects left, and the score matches fault=none.
"""
import argparse
import logging
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=".")
ap.add_argument("--fault", default="none")
ap.add_argument("--task", default="sort_language")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--counter_dz", type=float, default=0.0, help="real counter this much higher than scene.yaml (m)")
ap.add_argument("--cam_pitch_deg", type=float, default=0.0, help="true camera pitched by this vs calibration.yaml")
ap.add_argument("--cam_dx", type=float, default=0.0)
ap.add_argument("--sag_kp", type=float, default=0.0, help="arm PD stiffness without gravity feed-forward: shoulder/"
                "elbow kp (Nm/rad), wrists kp/2; the real runner has 80, the sim arm behaves like ~320; 0 = rigid")
ap.add_argument("--attach_radius", type=float, default=0.04, help="mock grasp: object sticks within this distance")
ap.add_argument("--depth_noise_mm", type=float, default=1.0, help="mock depth noise sigma (default 1 mm)")
ap.add_argument("--hole_frac", type=float, default=0.02, help="mock depth: fraction of invalid pixels")
ap.add_argument("--verbose", action="store_true")
ap.add_argument("--threaded", action="store_true", help="server code path: look in a thread, env paced in real time")
a = ap.parse_args()
sys.path.insert(0, a.repo)
from aaltonomous.baselines.classical.mock_world import make_mock_env, Renderer  # noqa: E402
from aaltonomous.vaiski.policy_server import VaiskiServerPolicy  # noqa: E402

logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")


def mutate(obs, k, t, st, rng, fault):
    """Return the observation the policy sees."""
    o = dict(obs)
    black = lambda: (np.zeros_like(obs["head_rgb"]), np.zeros_like(obs["head_depth"]))  # noqa: E731
    if fault == "black_first" and t < 3.0:
        o["head_rgb"], o["head_depth"] = black()
    elif fault == "black_mid" and 15.0 <= t < 21.0:
        o["head_rgb"], o["head_depth"] = black()
    elif fault == "black_random" and rng.random() < 0.3:
        o["head_rgb"], o["head_depth"] = black()
    elif fault == "no_depth":
        o["head_depth"] = np.zeros_like(obs["head_depth"])
    elif fault == "depth_first_missing" and t < 3.0:
        o["head_depth"] = np.zeros_like(obs["head_depth"])
    elif fault == "depth_holes":
        d = obs["head_depth"].copy()
        d[rng.random(d.shape) < 0.5] = 0
        d[:, 200:260] = 0
        o["head_depth"] = d
    elif fault == "depth_garbage":
        d = obs["head_depth"].copy()
        m = rng.random(d.shape) < 0.1
        d[m] = rng.integers(0, 65535, int(m.sum()))
        o["head_depth"] = d
    elif fault == "nan_joints" and rng.random() < 0.3:
        q = obs["arm_q"].astype(float).copy()
        q[rng.integers(0, 14)] = np.nan
        o["arm_q"], o["hand_q"] = q, np.full(14, np.nan)
    elif fault == "nan_first" and k < 5:
        o["arm_q"], o["hand_q"] = np.full(14, np.nan), np.full(14, np.nan)
    elif fault == "spike_joints" and rng.random() < 0.3:
        q = obs["arm_q"].astype(float).copy()
        q[rng.integers(0, 14)] += 1e3
        o["arm_q"], o["hand_q"] = q, np.zeros(14)
    elif fault == "frozen" and t >= 12.0:
        if "frame" not in st:
            st["frame"] = (obs["head_rgb"].copy(), obs["head_depth"].copy())
        o["head_rgb"], o["head_depth"] = st["frame"]
    elif fault == "t_garbage":
        o["t"] = 1e9
    elif fault == "t_nan":
        o["t"] = float("nan")
    elif fault == "missing_keys" and rng.random() < 0.3:
        for key in ("head_depth", "left_wrist_rgb", "right_wrist_rgb"):
            o.pop(key, None)
    elif fault == "no_image_keys" and t < 3.0:
        for key in ("head_rgb", "head_depth"):
            o.pop(key, None)
    return o


tot = mx = 0
for s in a.seeds:
    env = make_mock_env(seed=s, realtime=a.threaded)
    env.task = a.task
    pol = VaiskiServerPolicy()
    pol.pol.threaded = a.threaded
    pol.pol.cfg.setdefault("adaptive", {})["empty_grasp_rad"] = 0.0   # mock fingers track exactly
    obs = env.reset()
    if a.threaded:   # the real runner sends holds while the policy computes; this in-process loop cannot, so
        env.filter.cfg.stop_after_s = 1e9   # the watchdog stop is switched off and slow act() calls are counted instead
    if a.counter_dz or a.cam_pitch_deg or a.cam_dx:       # the WORLD differs from what the policy believes
        sc = env.backend.scene
        if a.counter_dz:
            mo = list(sc.cfg["counter"]["mat_origin_in_pelvis"])
            mo[2] += a.counter_dz
            sc.cfg["counter"]["mat_origin_in_pelvis"] = mo
            for ob in env.backend.objects:
                ob["pos"] = ob["pos"] + np.array([0, 0, a.counter_dz])
        cal = env.backend.renderer.cal
        if a.cam_pitch_deg or a.cam_dx:
            import copy
            cal = copy.deepcopy(cal)
            T = np.array(cal.cams["head"].T_pelvis_optical, float)
            th = np.radians(a.cam_pitch_deg)
            Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])  # about pelvis y
            T[:3, :3] = Ry @ T[:3, :3]
            T[0, 3] += a.cam_dx
            cal.cams["head"].T_pelvis_optical = T
        env.backend.renderer = Renderer(sc, cal)
        env.backend._cache = None
        obs = env._obs()
    if a.fault == "look_error":
        orig = pol.pol.perception.detect
        cnt = {"n": 0}

        def bad(*x, **kw):
            cnt["n"] += 1
            if cnt["n"] <= 3 or cnt["n"] % 4 == 0:
                raise RuntimeError("injected perception failure")
            return orig(*x, **kw)
        pol.pol.perception.detect = bad
    if a.fault == "ik_error":
        origp = pol.pol.planner.plan
        cnt2 = {"n": 0}

        def badp(*x, **kw):
            cnt2["n"] += 1
            if cnt2["n"] in (1, 2, 5):
                raise RuntimeError("injected planner failure")
            return origp(*x, **kw)
        pol.pol.planner.plan = badp
    env.backend.attach_radius = a.attach_radius
    env.backend.depth_noise_mm, env.backend.hole_frac = a.depth_noise_mm, a.hole_frac
    env.backend._cache = None
    sag = {"clear": [], "close": []}
    if a.sag_kp:      # the arm settles tau_gravity / kp below its command, like a PD joint without feed-forward
        import pinocchio as pin
        b, kin, pl = env.backend, pol.pol.kin, pol.pol.planner
        gd = kin.model.createData()
        kvec = np.array([a.sag_kp] * 4 + [a.sag_kp / 2] * 3) .repeat(1)
        kvec = np.concatenate([kvec, kvec])
        orig_adv, prev_f = b.advance, {"left": 0.0, "right": 0.0}

        def adv(dt, b=b, kin=kin, pl=pl, gd=gd, kvec=kvec, orig_adv=orig_adv, prev_f=prev_f, sag=sag):
            cmd = b.arm_cmd.copy()
            qq = kin.q0.copy()
            qq[kin.arm_iq] = b.arm_q
            g = np.array(pin.computeGeneralizedGravity(kin.model, gd, qq))[kin.arm_iv]
            b.arm_cmd = cmd - g / kvec
            orig_adv(dt)
            b.arm_cmd = cmd
            for side in ("left", "right"):
                f = b._closure(side)
                if f > 0.05:      # hand not open: it is working at the counter or carrying
                    sag["clear"].append(pl.clearance(side, b.arm_q, b.hand_q))
                if prev_f[side] <= 0.6 < f:   # the moment the mock decides whether the object sticks
                    gp, _ = kin.grasp_pose(side, b.arm_q)
                    free = [o["pos"] for i, o in enumerate(b.objects)]
                    o = min(free, key=lambda p: np.linalg.norm(p - gp))
                    sag["close"].append(gp - o)
                prev_f[side] = f
        b.advance = adv
    pol.reset(env.instruction, task=a.task, seed=s)
    rng = np.random.default_rng(100 + s)
    st, k, exc, slow, done_t, vanished = {}, 0, [], 0.0, None, False
    t0 = time.time()
    while True:
        seen = mutate(obs, k, float(obs["t"]), st, rng, a.fault)
        if a.fault in ("vanish_held", "vanish_before") and not vanished:
            b = env.backend
            held = [h for h in b.held.values() if h is not None]
            if a.fault == "vanish_held" and held:
                i = held[0][0]
                b.held = {"left": None, "right": None}
                b.objects[i]["pos"] = np.array([2.0, 2.0, -1.0])
                vanished = True
            if a.fault == "vanish_before" and pol.pol.state == "executing":
                b.objects[0]["pos"] = np.array([2.0, 2.0, -1.0])
                vanished = True
        tc, cc = time.perf_counter(), time.thread_time()
        try:
            act = pol.act(seen)
        except Exception as e:  # a crash of the policy = an error reply = watchdog stop on the robot
            exc.append(repr(e)[:120])
            act = {"arm_q_target": np.full((1, 14), np.nan), "hand_q_target": np.full((1, 14), np.nan), "done": False}
            if len(exc) > 30:
                break
        slow = max(slow, time.perf_counter() - tc)
        if time.perf_counter() - tc > 0.2:
            st["think_cpu"] = st.get("think_cpu", 0.0) + time.thread_time() - cc
            st["think_max"] = max(st.get("think_max", 0.0), time.thread_time() - cc)
        st["late"] = st.get("late", 0) + (time.perf_counter() - tc > 0.2)
        k += 1
        obs, info = env.step(act)
        if act.get("done") and done_t is None:
            done_t = float(obs["t"])
        if info.get("episode_over"):
            break
    outc = env.backend.outcomes(env.mapping)
    sc_ = env.scene.score(a.task, outc)
    best = len(outc) * (2 if a.task == "sort_language" else 1)
    tot += sc_
    mx += best
    left = sum(o != "correct" for o in outc)
    print(f"fault={a.fault} seed {s}: score {sc_}/{best} left={left} clipped={info.get('clipped_steps')} "
          f"end={info.get('end_reason')} t_end={obs['t']:.0f}s done_at={done_t} exceptions={len(exc)} "
          + (f"SAG kp={a.sag_kp:.0f}: lowest hand point {1000 * min(sag['clear']):+.0f} mm above the counter, grasp "
             f"point - object centre at closing (mm) mean {np.round(1000 * np.mean(sag['close'], 0))} n={len(sag['close'])} "
             if a.sag_kp and sag["close"] else "") +
          f"{exc[:1]} slowest_act={1000 * slow:.0f}ms acts>0.2s={st.get('late', 0)} wall={time.time() - t0:.0f}s "
          f"THINK cpu_total={st.get('think_cpu', 0.0):.1f}s cpu_max={st.get('think_max', 0.0):.1f}s "
          f"plans_ok={[round(c, 1) for ok, c in getattr(pol.pol, '_plan_cpu', []) if ok]} "
          f"plans_fail={[round(c, 1) for ok, c in getattr(pol.pol, '_plan_cpu', []) if not ok]}", flush=True)
print(f"fault={a.fault} TOTAL {tot}/{mx}")
