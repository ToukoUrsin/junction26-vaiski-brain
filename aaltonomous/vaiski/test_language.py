"""python -m pytest aaltonomous/vaiski/test_language.py -q"""
import itertools

import numpy as np

from aaltonomous.interface.tasks import COLORS, BINS, TEMPLATES, FIXED_INSTRUCTION, make_instruction
from aaltonomous.vaiski.language import parse

R, Y, B = "red", "yellow", "blue"


def test_fixed():
    assert parse(FIXED_INSTRUCTION).mapping == {R: "left", Y: "center", B: "right"}


def test_all_training_templates_all_permutations():
    for t in TEMPLATES:
        for cs in itertools.permutations(COLORS):
            for bs in itertools.permutations(BINS):
                w = {f"c{i}": c for i, c in enumerate(cs)}
                w.update({f"C{i}": c.capitalize() for i, c in enumerate(cs)})
                w.update({f"b{i}": b for i, b in enumerate(bs)})
                text = t.format(**w)
                assert parse(text).mapping == dict(zip(cs, bs)), text


def test_make_instruction_seeds():
    for s in range(300):
        text, m = make_instruction("sort_language", np.random.default_rng(s))
        assert parse(text).mapping == m, text


CASES = [
    ("Sort the objects: red into right, blue into left, yellow into center.", {R: "right", B: "left", Y: "center"}),
    ("Blue things go in the middle box, red in the rightmost one and yellow in the leftmost.",
     {B: "center", R: "right", Y: "left"}),
    ("Left bin: yellow. Middle bin: blue. Right bin: red.", {Y: "left", B: "center", R: "right"}),
    ("Could you please put all the red pieces into the box on the right, the blue ones into the box on the left, "
     "and the yellow ones in the middle?", {R: "right", B: "left", Y: "center"}),
    ("Red and yellow are swapped compared to the usual sorting.", {R: "center", Y: "left", B: "right"}),
    ("Swap red and blue.", {R: "right", Y: "center", B: "left"}),
    ("Do the usual sorting but the other way round.", {R: "right", Y: "center", B: "left"}),
    ("The red objects belong in the left container, blue in the right one.", {R: "left", B: "right", Y: "center"}),
    ("Red: right. Yellow: left. Blue: middle.", {R: "right", Y: "left", B: "center"}),
    ("Put blue on the left, yellow on the right, and red in the middle.", {B: "left", Y: "right", R: "center"}),
    ("Red, yellow and blue go to the right, left and center bins respectively.", {R: "right", Y: "left", B: "center"}),
    ("Yellow objects should not go to the center; put them on the left. Red goes to the center and blue right.",
     {Y: "left", R: "center", B: "right"}),
    ("Crimson items to the centre crate, golden ones to the right crate, navy ones to the left crate.",
     {R: "center", Y: "right", B: "left"}),
    ("The middle box takes the red blocks; the left box the blue blocks; the right box the yellow blocks.",
     {R: "center", B: "left", Y: "right"}),
    ("I want the yellow cubes and cylinders on the right side, the red stuff left, and blue center.",
     {Y: "right", R: "left", B: "center"}),
    ("Everything blue into the left bin. Everything yellow into the right bin. Red stays for the middle bin.",
     {B: "left", Y: "right", R: "center"}),
    ("Laita punaiset oikeaan laatikkoon, siniset vasempaan ja keltaiset keskelle.", {R: "right", B: "left", Y: "center"}),
    ("Into the rightmost bin go the blue objects, into the leftmost the red, and into the center bin the yellow.",
     {B: "right", R: "left", Y: "center"}),
    ("Red to the right bin, blue to the left bin.", {R: "right", B: "left", Y: "center"}),
    ("Please sort by colour: blue in the first box, red in the second box, yellow in the third box.",
     {B: "left", R: "center", Y: "right"}),
]


def test_paraphrases():
    bad = []
    for text, want in CASES:
        got = parse(text).mapping
        if got != want:
            bad.append((text, got, want))
    assert not bad, "\n".join(f"{t}\n   got {g}\n  want {w}" for t, g, w in bad)


# ---------------------------------------------------------------------------------------------------------------
# Unseen-instruction classes (helper "language", 14.11.2026). What the robot EXECUTES is parse(text).table():
# 9 object types -> bin | None (None = leave on the counter). `.mapping` stays the per-colour best guess.
L, C, RT = "left", "center", "right"


def _cells(want):
    """{colour: bin} or {colour/shape or colour or shape or 'default': bin} -> the 9-cell table."""
    out = {}
    for c in COLORS:
        for s in ("cube", "cylinder", "triangle"):
            for k in (f"{c}/{s}", c, s, "default"):
                if k in want:
                    out[f"{c}/{s}"] = want[k]
                    break
            else:
                out[f"{c}/{s}"] = None
    return out


EXECUTED = [
    # negation
    ("Red must never go left or center. Yellow is not for the right bin and not for the center bin. "
     "Blue takes the bin that is neither left nor right.", {R: RT, Y: L, B: C}),
    ("Don't put red in the middle or on the right. Blue never goes left or right. Yellow avoids the left and the center.",
     {R: L, B: C, Y: RT}),
    ("The center bin, not either end bin, receives red objects; the left bin, rather than the right or center, "
     "takes yellow objects; blue objects go to the right bin and nowhere else.", {R: C, Y: L, B: RT}),
    # exceptions
    ("Everything except blue goes in the left bin; blue goes right.", {R: L, Y: L, B: RT}),
    ("All but the reds go right. Reds go to the center.", {R: C, Y: RT, B: RT}),
    ("Apart from yellow, which goes to the center bin, put everything in the right bin.", {R: RT, Y: C, B: RT}),
    # swaps relative to the default rule
    ("Same as the usual rule, but red and blue trade places.", {R: RT, Y: C, B: L}),
    ("Use the default sorting, except swap yellow and blue.", {R: L, Y: RT, B: C}),
    ("Do the standard rule mirrored.", {R: RT, Y: C, B: L}),
    # ordinals and relative positions
    ("Counting from the right: blue in the first bin, red in the second, yellow in the third.", {B: RT, R: C, Y: L}),
    ("Yellow goes in the box furthest to the right, red in the one between the others, blue in the leftmost.",
     {Y: RT, R: C, B: L}),
    ("Blue into the leftmost container, red into the container just to its right, and yellow into the last one.",
     {B: L, R: C, Y: RT}),
    ("Red goes in the middle bin. Yellow goes in the bin to the left of the bin that gets red. Blue goes in the bin "
     "to the right of the red one.", {R: C, Y: L, B: RT}),
    ("From left to right: yellow, blue, red.", {Y: L, B: C, R: RT}),
    # synonyms, shades, typos
    ("Crimson pieces go to the right-hand box, navy ones to the left-hand box, lemon ones in the middle.",
     {R: RT, B: L, Y: C}),
    ("rde objcts in teh rigth bin, yelow in the midle, bleu on the lfet", {R: RT, Y: C, B: L}),
    # chatty filler, asides, long
    ("Right, so, morning everyone. The coffee machine is broken again. Anyway: all the red ones (cubes, cylinders, "
     "whatever) belong in the center bin, right? Blue ones you drop on the left hand side and yellow on the right. "
     "Use whichever hand you like.", {R: C, B: L, Y: RT}),
    ("Yellow ones, doesn't matter if it's a cube or a cylinder or a prism, go into the center bin. Red goes left "
     "and blue goes right.", {Y: C, R: L, B: RT}),
    # corrections inside the instruction
    ("Red left, no wait, right. Blue left. Yellow in the middle.", {R: RT, B: L, Y: C}),
    ("Send red left, yellow right and blue center, no, wait: center, left and right respectively.",
     {R: C, Y: L, B: RT}),
    ("Blue goes in the center bin, actually no, into the right bin; yellow in the left; red in the center.",
     {B: RT, Y: L, R: C}),
    # other languages
    ("Laita punaiset oikeaan laatikkoon, keltaiset keskimmäiseen ja siniset vasempaan.", {R: RT, Y: C, B: L}),
    ("Rote Objekte nach links, gelbe nach rechts, blaue in die Mitte.", {R: L, Y: RT, B: C}),
    ("Les rouges à droite, les jaunes au milieu, les bleus à gauche.", {R: RT, Y: C, B: L}),
    ("Lägg de röda till vänster, de gula till höger och de blå i mitten.", {R: L, Y: RT, B: C}),
    # two colours into one box
    ("Red and yellow both go in the left bin; blue goes in the right bin.", {R: L, Y: L, B: RT}),
    ("Center gets yellow and blue, left gets red.", {Y: C, B: C, R: L}),
    ("Red objects belong in the left bin; yellow objects too; blue objects belong in the right bin.",
     {R: L, Y: L, B: RT}),
    ("Keep the center bin empty: reds and yellows to the left bin, blues to the right.", {R: L, Y: L, B: RT}),
    ("Put everything in the center bin.", {R: C, Y: C, B: C}),
    # leave on the counter
    ("Red goes left and blue goes right. Leave the yellow ones on the counter.", {R: L, B: RT, Y: None}),
    ("Only sort the red objects: they go in the right bin. Do not touch anything else.", {R: RT, Y: None, B: None}),
    ("Yellow in the center, red on the right; ignore blue.", {Y: C, R: RT, B: None}),
    # a colour that is never named is NOT sorted by elimination (a distractor stays on the counter)
    ("Put the red objects in the right bin and the blue objects in the left bin.", {R: RT, B: L, Y: None}),
    # ... but "the remaining bin" names it
    ("Red goes right, blue goes left, and yellow takes whichever bin is left.", {R: RT, B: L, Y: C}),
    # sorting by shape, colour + shape, catch-all
    ("Sort by shape, not colour: cubes left, cylinders in the middle, triangular prisms right.",
     {"cube": L, "cylinder": C, "triangle": RT}),
    ("Red cubes go in the left bin. Everything else goes in the right bin.", {"red/cube": L, "default": RT}),
    ("Red goes left, yellow goes center and blue goes right, except blue cubes, which go in the center bin.",
     {R: L, Y: C, B: RT, "blue/cube": C}),
    ("Red left, yellow center, blue right. The sole exception is blue triangular prisms, which must remain on the "
     "counter.", {R: L, Y: C, B: RT, "blue/triangle": None}),
    ("Anything that's a blue or yellow cube goes right; everything else lands in the center bin.",
     {"blue/cube": RT, "yellow/cube": RT, "default": C}),
    # second round (patterns from a second fresh batch)
    ("Everything except red and yellow objects goes to the left bin. Red objects go to the right bin, whereas yellow "
     "objects go to the center bin.", {B: L, R: RT, Y: C}),
    ("Yellow to the first bin, red to the second, blue to the last.", {Y: L, R: C, B: RT}),
    ("Can you find the last bin? Yellow things go there. Red things belong in the bin just before it, and blue things "
     "go in the first bin.", {Y: RT, R: C, B: L}),
    ("- Colours in order: red, yellow, blue\n- Bins they go to, respectively: right, center, left", {R: RT, Y: C, B: L}),
    ("- Colour order: red, yellow, blue.\n- Matching bin order: left, right, center. Sorry, I meant left, center, right.",
     {R: L, Y: C, B: RT}),
    ("Destination(red) = left bin; destination(yellow) = right bin; destination(blue) = center bin.", {R: L, Y: RT, B: C}),
    ("The left bin, that's where red goes, never the right. Yellow goes in the center. Blue, not the middle one, the "
     "bin on the right.", {R: L, Y: C, B: RT}),
    ("Red in the left bin, yellow in the center, blue in the right. Sorry, I mixed that up: yellow belongs in the right "
     "bin and blue in the center bin. Red stays left.", {R: L, Y: RT, B: C}),
    ("Keep red and yellow objects together in the right bin, and place blue objects in the left bin.", {R: RT, Y: RT, B: L}),
    ("Left yellow, center blue, right red. Do not mix. Do not swap left and right.", {Y: L, B: C, R: RT}),
    ("Please place red objects in the center bin, left yellow objects on the counter, and put blue objects in the right "
     "bin.", {R: C, Y: None, B: RT}),
    ("Please leave all red and blue objects on the counter, and place every yellow object in the center bin.",
     {R: None, B: None, Y: C}),
    ("The left bin takes yellow, the center bin takes red, the right bin takes blue. Wait, no, that's not right at all; "
     "the center bin takes blue and the right bin takes red. The left bin still takes yellow.", {Y: L, B: C, R: RT}),
    ("Pls sort like this, from lfet to right in order: yellow, blue, red.", {Y: L, B: C, R: RT}),
    ("Yellow objects to the left bin, red objects to the center bin, blue objects join the reds.", {Y: L, R: C, B: C}),
    ("Please keep red from left and right, yellow from center and right, and blue from left and center.",
     {R: C, Y: L, B: RT}),
    ("Do not put red in center or right, yellow in left or right, or blue in left or center.", {R: L, Y: C, B: RT}),
    ("Blue in the last bin on the right, yellow two bins left of that, red, um, the leftover one.", {B: RT, Y: L, R: C}),
    ("Sort the usual way, only mirrored, so that the colours headed for the left and right bins trade places while "
     "yellow stays in the center.", {R: RT, Y: C, B: L}),
    ("Mirror the usual sorting: flip it left to right. Whatever goes to one side normally now goes to the opposite side.",
     {R: RT, Y: C, B: L}),
    ("Grab every red piece and drop it into the bin on your right then do the same thing with the yellow pieces except "
     "those go in the middle bin.", {R: RT, Y: C, B: None}),
    ("Do not hesitate to put red in the right bin, yellow in the left bin and blue in the center one.", {R: RT, Y: L, B: C}),
    ("Please don't forget to put the red ones in the center bin, yellow in the left and blue in the right.",
     {R: C, Y: L, B: RT}),
    ("Red pieces, on the other hand, go straight into the center bin. Yellow goes left and blue goes right.",
     {R: C, Y: L, B: RT}),
    ("The bin at the far left is taking all yellow objects. Then the bin just right of it is for the blue ones, and the "
     "last bin, the one at the far right, is for red objects.", {Y: L, B: C, R: RT}),
    ("Route the red objects to the right bin and the yellow objects to the left bin, but let all blue objects remain on "
     "the counter.", {R: RT, Y: L, B: None}),
    ("Unless an object is a red triangular prism, place it in the left bin; place red triangular prisms in the right "
     "bin.", {"red/triangle": RT, "default": L}),
    ("Red into the center bin, yellow into the left bin, blue into the right bin. The sole exception is the blue "
     "cylinder, which shall not be binned and must remain on the counter.", {R: C, Y: L, B: RT, "blue/cylinder": None}),
    ("Cube to left bin, tube to right bin, triangle to center bin, any colour.", {"cube": L, "cylinder": RT, "triangle": C}),
    ("Drop every cube, cylinder and prism into the center bin, whatever its shape, except the yellow ones. Those yellow "
     "pieces go over to the left bin instead.", {R: C, B: C, Y: L}),
    # third round (patterns from a third fresh batch)
    ("Blue, red and yellow things go to the right, center and left bins, in exactly that order.", {B: RT, R: C, Y: L}),
    ("Left, center and right bin: put there yellow, blue and red objects, in same order.", {Y: L, B: C, R: RT}),
    ("Going from your left to your right, the bins get blue, then red, then yellow objects.", {B: L, R: C, Y: RT}),
    ("Left bin for yellow and blue, red in the center, hurry!", {Y: L, B: L, R: C}),
    ("Put yellow in the left bin, red center, and blue left, sorry, I meant right?", {Y: L, R: C, B: RT}),
    ("Can you send yellow to the center and both red and blue right? Quick!", {Y: C, R: RT, B: RT}),
    ("Put red in the left bin, yellow in the center, blue on the right, right?", {R: L, Y: C, B: RT}),
    ("Rule of the round: every object, whatever its colour, goes into the right bin.", {R: RT, Y: RT, B: RT}),
    ("Yellow pieces go left. Blue pieces skip the left and the center altogether and go straight into the right bin. "
     "Red pieces go in the center.", {Y: L, B: RT, R: C}),
    ("Mapping: red → center bin; yellow → left bin; blue → none (leave on counter).", {R: C, Y: L, B: None}),
    ("Do the default sorting, only with the left and right bins switching roles.", {R: RT, Y: C, B: L}),
    ("Put blue cubes, cylinders, and triangular prisms in the left bin, no, correction: blue objects go in the center "
     "bin. Yellow objects belong in the left bin, while red objects belong in the right bin.", {B: C, Y: L, R: RT}),
    ("Se ti capita un oggetto rosso, due righe: il suo posto è il contenitore di sinistra; se è giallo, portalo in "
     "quello di mezzo; se è blu, in quello di destra.", {R: L, Y: C, B: RT}),
    ("If it is red, put it in the leftmost bin. If it is blue, put it in the bin one step to the right of the one "
     "holding red. Yellow goes in the rightmost bin.", {R: L, B: C, Y: RT}),
    ("Count the bins off from the left: one, two, three. Blue goes in one. Red goes in two. Yellow goes in three.",
     {B: L, R: C, Y: RT}),
    ("If it's red, left bin; if it's yellow, right bin; if it's blue, center bin. Unless it's a yellow cube, in which "
     "case it goes to the left bin.", {R: L, Y: RT, B: C, "yellow/cube": L}),
    ("All cubes go into the right bin, prisms into the center bin and cylinders into the left bin, except for blue "
     "prisms, which I'd like in the left bin instead.", {"cube": RT, "triangle": C, "cylinder": L, "blue/triangle": L}),
    ("Cubes in the center bin, cylinders in the left bin, prisms in the right bin. Red and blue cubes go to the right "
     "bin instead.", {"cube": C, "cylinder": L, "triangle": RT, "red/cube": RT, "blue/cube": RT}),
    ("Move the red ones from the left side to the right bin, yellow center, blue left.", {R: RT, Y: C, B: L}),
]


def test_executed_rules_per_class():
    bad = []
    for text, want in EXECUTED:
        got = parse(text).table()
        if got != _cells(want):
            bad.append((text, {k: v for k, v in got.items() if v != _cells(want)[k]}))
    assert not bad, "\n".join(f"{t}\n   differs: {g}" for t, g in bad)


def test_abstain_and_best_guess():
    p = parse("Put the red objects in the right bin and the blue objects in the left bin.")
    assert p.mapping == {R: RT, B: L, Y: C}                 # legacy best guess (elimination) stays available
    assert p.target(Y) is None and p.target(R) == RT        # ... but the never-named colour is not executed
    assert p.certainty[Y][1] < 0.4 <= p.certainty[R][1]
    p = parse("Red goes left, no, right, no, left... blue center.")
    assert p.target(B) == C


def test_target_with_shapes():
    p = parse("Cubes go left. Cylinders and triangular prisms go right.")
    assert p.by_shape and p.target(R, "cube") == L and p.target(B, "triangle") == RT
    assert p.target(Y, "lying") == RT                       # lying = cylinder or prism: both go right
    assert p.target(Y, None) is None                        # unknown shape: not sorted when the shape matters
    p = parse("Red left, yellow center, blue right.")
    assert not p.by_shape and p.target(R, None) == L and p.target(B, "lying") == RT


def test_executed_mapping_for_the_brain():
    from aaltonomous.vaiski.language import executed, FIXED
    assert executed(parse("blue left, red right"), "sort_language") == {R: RT, B: L, Y: None}
    assert executed(parse("blue left, red right"), "sort_fixed") == FIXED           # fixed rung: not fully certain -> the rule
    assert executed(parse(FIXED_INSTRUCTION), "sort_fixed") == FIXED
    assert executed(parse(""), "sort_language") == FIXED                            # no rule words at all: default rule
    assert executed(parse("Cubes left, cylinders center, prisms right."), "sort_language") == {R: None, Y: None, B: None}


def test_speed_and_determinism():
    import time
    texts = [t for t, _ in EXECUTED] + [t for t, _ in CASES]
    parse(texts[0])
    t0 = time.perf_counter()
    first = [repr(parse(t)) for t in texts]
    dt = (time.perf_counter() - t0) / len(texts)
    assert dt < 0.02, dt
    assert first == [repr(parse(t)) for t in texts]
