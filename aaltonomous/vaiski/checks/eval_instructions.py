"""Score a parser on the stress set (9-cell ground truth) and on the colour-only regression sets.

    python aaltonomous/vaiski/checks/eval_instructions.py . aaltonomous/vaiski/checks/data/unseen1.jsonl [--show CLASS] [--n 15] [--calib] [--only-wrong]
    python aaltonomous/vaiski/checks/eval_instructions.py . --reg FILE.jsonl ...     (colour mapping exact match, like score_parser.py)

REPO = directory that contains aaltonomous/ (the parser under test). Per class: exact = all 9 object types right;
harmful = at least one type would be put in a WRONG box (-1 each); points = expected points per object under the
language-rung scoring (+2 right box, -1 wrong box, 0 left on the counter), next to the maximum a perfect parser gets.
"""
import argparse
import collections
import json
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("stress", nargs="?")
ap.add_argument("--reg", nargs="*", default=[])
ap.add_argument("--show")
ap.add_argument("--n", type=int, default=15)
ap.add_argument("--calib", action="store_true")
ap.add_argument("--only-wrong", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.repo)
from aaltonomous.vaiski.language import parse  # noqa: E402

COLORS, SHAPES = ("red", "yellow", "blue"), ("cube", "cylinder", "triangle")


def table(p):
    if hasattr(p, "table"):
        return {k: (v or "none") for k, v in p.table().items()}
    return {f"{c}/{s}": (p.mapping.get(c) or "none") for c in COLORS for s in SHAPES}


for f in a.reg:
    rows = [json.loads(l) for l in open(f)]
    bad = [r for r in rows if parse(r["text"]).mapping != r["mapping"]]
    print(f"{f}: {len(rows) - len(bad)}/{len(rows)} correct")
    if a.show == "reg":
        for r in bad[:a.n]:
            p = parse(r["text"])
            print("-", r["text"], "\n   want", r["mapping"], "\n   got ", p)

if a.stress:
    rows = [json.loads(l) for l in open(a.stress)]
    st = collections.defaultdict(lambda: collections.Counter())
    calib = collections.defaultdict(lambda: collections.Counter())
    shown = 0
    t0 = time.perf_counter()
    worst = 0.0
    for r in rows:
        t1 = time.perf_counter()
        p = parse(r["text"])
        worst = max(worst, time.perf_counter() - t1)
        got, want = table(p), r["truth"]
        wrong = [k for k in got if got[k] != "none" and got[k] != want[k]]
        missed = [k for k in got if got[k] == "none" and want[k] != "none"]
        right = [k for k in got if got[k] != "none" and got[k] == want[k]]
        for cls in (r["cls"], "TOTAL"):
            s = st[cls]
            s["n"] += 1
            s["exact"] += got == want
            s["harmful"] += bool(wrong)
            s["pts"] += 2 * len(right) - len(wrong)
            s["max"] += 2 * sum(v != "none" for v in want.values())
            s["wrong_cells"] += len(wrong)
            s["placed_cells"] += len(wrong) + len(right)
        if a.calib and hasattr(p, "certainty"):
            for key, (tag, pr) in p.certainty.items():
                if isinstance(key, tuple) or key == "default":
                    cells = [f"{key[0]}/{key[1]}"] if isinstance(key, tuple) else []
                elif key in COLORS:
                    cells = [f"{key}/{s}" for s in SHAPES]
                else:
                    cells = [f"{c}/{key}" for c in COLORS]
                raw = getattr(p, "raw", {}).get(key, None) if not p.by_shape else p.rules.get(key)
                for cell in cells[:1]:
                    val = raw or "none"
                    calib[tag]["n"] += 1
                    calib[tag]["ok"] += val == want[cell]
                    calib[tag]["placed"] += val != "none"
                    calib[tag]["placed_ok"] += val != "none" and val == want[cell]
            conf_b = "conf>=0.9" if p.confidence >= 0.9 else ("conf 0.5-0.9" if p.confidence >= 0.5 else "conf<0.5")
            calib[conf_b]["n"] += 1
            calib[conf_b]["ok"] += got == want
            calib[conf_b]["placed"] += 1
            calib[conf_b]["placed_ok"] += not wrong
        if a.show and (a.show == r["cls"] or a.show == "all") and got != want and shown < a.n and (wrong or not a.only_wrong):
            shown += 1
            print(f"- [{r['cls']}] {r['text']}\n   spec {r['spec']}\n   want {want}\n   got  {got}\n   {p}")
    dt = time.perf_counter() - t0
    print(f"{'class':14s}    n  exact  harmful  points/max   (wrong cells / placed cells)")
    for cls in sorted(st, key=lambda c: (c == "TOTAL", c)):
        s = st[cls]
        print(f"{cls:14s} {s['n']:4d} {100 * s['exact'] / s['n']:5.1f}% {100 * s['harmful'] / s['n']:6.1f}%  "
              f"{s['pts'] / 9 / s['n']:5.2f}/{s['max'] / 9 / s['n']:4.2f}   {s['wrong_cells']}/{s['placed_cells']}")
    print(f"parse time: mean {1000 * dt / len(rows):.2f} ms, worst {1000 * worst:.1f} ms")
    if a.calib:
        print("calibration (per key: tag -> share of best guesses that are right; per instruction: confidence bucket -> no wrong cell)")
        for tag in sorted(calib):
            c = calib[tag]
            print(f"  {tag:14s} n={c['n']:5d}  right {100 * c['ok'] / max(1, c['n']):5.1f}%   "
                  f"when a bin is named: {c['placed_ok']}/{c['placed']} = {100 * c['placed_ok'] / max(1, c['placed']):5.1f}%")
