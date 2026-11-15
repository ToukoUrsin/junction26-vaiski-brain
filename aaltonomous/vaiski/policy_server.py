"""Väiski brain: the submission policy server (port 8000).

    python -m aaltonomous.vaiski.policy_server

= classical pick-and-place core (baseline planner, IK, safety pre-check) +
  vaiski.language (offline instruction -> colour->bin mapping, any phrasing) +
  vaiski.perception (hue prototypes that adapt to the real camera, geometric bin removal).
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from aaltonomous.deploy.policy_server import Policy, serve
from aaltonomous.interface.tasks import FIXED_MAPPING

from .brain import VaiskiPolicy
from .language import executed, parse

log = logging.getLogger("vaiski")


class VaiskiServerPolicy(Policy):
    name = "vaiski"

    def __init__(self, time_limit=None, **kw):
        tl = float(time_limit or os.environ.get("AALTO_TIME_LIMIT", 120.0))
        self.pol = VaiskiPolicy(threaded=True, require_confirm=False, time_limit=tl, log=lambda m: log.info(m), **kw)
        self.action_horizon = int(self.pol.cfg["motion"]["chunk"])

    def reset(self, instruction, task=None, seed=None):
        job, t0 = self.pol._job, time.monotonic()
        self.pol.planner.cancel = True      # a look still planning for the last episode stops at its next check
        try:
            while job is not None and job["result"] is None and time.monotonic() - t0 < 30.0:
                time.sleep(0.02)
        finally:
            self.pol.planner.cancel = False
        if task == "sort_fixed" and not instruction:
            mapping = dict(FIXED_MAPPING)
            p = None
        else:
            p = parse(instruction or "")
            # what is safe to act on: a colour whose bin is uncertain stays on the counter (wrong box -1, counter 0)
            mapping = executed(p, task)
        by_shape = p if (p is not None and p.by_shape and task != "sort_fixed") else None
        self.pol.reset({"instruction": instruction, "mapping": mapping, "task": task, "parse": by_shape})
        log.info("reset task=%s seed=%s instruction=%r -> %s", task, seed, instruction, p if p else mapping)

    def act(self, obs):
        a = self.pol.act(obs)
        return {"arm_q_target": np.asarray(a["arm_q_target"], np.float64),
                "hand_q_target": np.asarray(a["hand_q_target"], np.float64), "done": bool(a.get("done", False))}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    pol = VaiskiServerPolicy()
    log.info("vaiski loaded in %.1f s", time.monotonic() - t0)
    serve(pol, port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
