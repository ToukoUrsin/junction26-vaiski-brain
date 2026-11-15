"""evaluate.py with a SHAPE instruction for sort_language (helper "shape"): python shape_eval.py <evaluate args>.
SHAPE_INSTR=shape (default) | mixed. The layout of a seed stays what sort_language gives for it (the original
make_instruction still consumes the rng); the organisers' own scorer gets the matching shape -> bin mapping."""
import os
import sys

import aaltonomous.interface.env as E
from aaltonomous.interface import tasks

_orig = tasks.make_instruction
KIND = os.environ.get("SHAPE_INSTR", "shape")


def make_instruction(task, rng):
    text, m = _orig(task, rng)
    if task != "sort_language":
        return text, m
    if KIND == "mixed":
        return ("Red cubes go in the left bin, blue cylinders in the right bin, and everything else in the center bin.",
                {"red/cube": "left", "blue/cylinder": "right", "default": "center"})
    return ("Cubes go left, cylinders in the middle, prisms on the right.",
            {"cube": "left", "cylinder": "center", "triangle": "right"})


E.make_instruction = make_instruction
from aaltonomous.sim import evaluate  # noqa: E402

evaluate.main(sys.argv[1:])
