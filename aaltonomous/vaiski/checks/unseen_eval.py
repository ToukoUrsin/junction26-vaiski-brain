"""evaluate.py for sort_language with UNSEEN instruction sentences: python unseen_eval.py <evaluate args>.
Each episode keeps the organisers' own colour -> box mapping and layout for its seed (their make_instruction still
consumes the rng); only the sentence is swapped for one from our never-read set (checks/data/unseen[123].jsonl)
that asks for exactly that sorting. The organisers' scorer gets the unchanged mapping."""
import json
import os
import random
import sys

import aaltonomous.interface.env as E
from aaltonomous.interface import tasks

_orig = tasks.make_instruction
COLORS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")
POOL = {}
for line in open(os.environ.get("UNSEEN", os.path.expanduser("~/unseen_all.jsonl")  # = cat checks/data/unseen[123].jsonl)):
    r = json.loads(line)
    t = r["truth"]
    m = {}
    for c in COLORS:
        b = {t.get(f"{c}/{s}") for s in SHAPES}
        if len(b) == 1:
            m[c] = b.pop()
    if len(m) == 3 and set(m.values()) == {"left", "center", "right"}:
        POOL.setdefault(tuple(sorted(m.items())), []).append(r)
_rng = random.Random(int(os.environ.get("UNSEEN_SEED", "2026")))
for k in sorted(POOL):
    _rng.shuffle(POOL[k])
print("unseen pool:", {" ".join(f"{c[0]}>{b[0]}" for c, b in k): len(v) for k, v in sorted(POOL.items())}, file=sys.stderr, flush=True)
USED = []


def make_instruction(task, rng):
    text, m = _orig(task, rng)
    if task != "sort_language":
        return text, m
    r = POOL[tuple(sorted(m.items()))].pop()
    USED.append(r)
    print(f"UNSEEN [{r.get('cls')}/{r.get('lang')}] {r['text']!r} -> {m}", file=sys.stderr, flush=True)
    return r["text"], m


E.make_instruction = make_instruction
from aaltonomous.sim import evaluate  # noqa: E402

if __name__ == "__main__":
    evaluate.main(sys.argv[1:])
