"""Rendered mock world, our server policy in-process. python aaltonomous/vaiski/checks/mock_eval.py --task sort_language --seeds 0 1 2"""
import argparse, sys, time
sys.path.insert(0, ".")
import numpy as np
from aaltonomous.baselines.classical.mock_world import make_mock_env
from aaltonomous.vaiski.policy_server import VaiskiServerPolicy
ap = argparse.ArgumentParser(); ap.add_argument("--task", default="sort_language"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
a = ap.parse_args()
import logging; logging.basicConfig(level=logging.INFO, format="%(message)s")
tot = 0
for s in a.seeds:
    env = make_mock_env(seed=s); env.task = a.task
    pol = VaiskiServerPolicy(); pol.pol.threaded = False
    pol.pol.cfg.setdefault("adaptive", {})["empty_grasp_rad"] = 0.0  # mock fingers track exactly: no contact signal
    obs = env.reset()
    pol.reset(env.instruction, task=a.task, seed=s)
    t0 = time.time(); clips = 0
    while True:
        act = pol.act(obs)
        obs, info = env.step(act)
        if act.get("done") or info.get("episode_over"):
            break
    outc = env.backend.outcomes(env.mapping)
    sc = env.scene.score(a.task, outc); tot += sc
    print(f"seed {s} t={obs['t']:.0f}s clipped={info.get('clipped_steps')} end={info.get('end_reason')} mapping_ok={pol.pol.mapping==env.mapping} {outc} score={sc} wall={time.time()-t0:.0f}s")
print("total", tot)
