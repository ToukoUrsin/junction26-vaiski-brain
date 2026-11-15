# A brain for Väiski

Our entry for Aalto's Aaltonomous Humanoid challenge at Junction 2026. Väiski, a Unitree G1 with two Dex3 hands,
sorts three to five blocks into three boxes in 120 seconds.

Nothing in it is trained. It looks, works out where the blocks and the boxes are, plans one pick, checks the plan
against the safety filter, moves, and looks again. When a step fails the log says which one and why.

## What is ours and what is the organisers'

This repository is the organisers' starter code (https://github.com/zupermiez/aaltonomous-humanoid, MIT) with our
work on top.

| Path | Whose | What |
|---|---|---|
| `aaltonomous/vaiski/` | ours | The brain: about 3,900 lines of Python without the tests |
| `aaltonomous/vaiski/checks/` | ours | The checks behind the numbers below, with their data |
| `aaltonomous/docker/vaiski.Dockerfile` | ours | The image we hand in (a copy of the organisers' baseline Dockerfile with our files) |
| `aaltonomous/sim/dr/hard_*.yaml` | ours | Harder simulator settings we test on: all randomness doubled, bright, dark |
| `aaltonomous/baselines/classical/planner.py`, `kinematics.py` | organisers', changed by us | See below |
| everything else | organisers' | Simulator, safety filter, scene, runner, baseline |

The arm planner and the pinch (thumb against one finger) are the organisers'. We changed three things in their
planner, each off unless our brain turns it on:

- `kinematics.gravity()` and `planner.sag_delta()`: command the arm a little higher when it is measured to hang
  under its own weight.
- `planner._transit()`: a free move may go over a raised midpoint when the straight path dips towards the counter.
- `planner.deadline` / `cancel`: a plan search stops when its time is spent or a new episode starts.

## The files

| File | What it does |
|---|---|
| `brain.py` | The loop: look, pick the next block, plan, dry-run the plan through the safety filter, move, check the hand, drop, look again. Also measures the camera, the counter, the boxes and how soft the arm is. |
| `perception.py` | Colours that follow the light, the counter plane and the box row from depth, the robot's own arms masked out. |
| `shape.py` | Cube, cylinder or prism from the outline of a block's top, and which sides are flat enough to pinch. |
| `language.py` | Instruction to box, without a language model. For colours there are 27 ways to fill three boxes; it crosses out every one the sentence rules out. |
| `policy_server.py` | The websocket server on port 8000. Looking and planning run in a thread, so it always answers in time. |
| `config.yaml` | Every threshold, with the reason next to it. |

## What we measured

In the organisers' simulator with the Docker image, blocks that ended in the right box.

| Run | Ours | Organisers' baseline, unchanged |
|---|---|---|
| Fixed task, 16 episodes | 66 of 66 | 36 of 66 |
| With an instruction in the organisers' example sentences, 16 episodes | 63 of 65 | it ignores the instruction |
| With a sentence we had not read before the run (`checks/unseen_eval.py`), 32 episodes | 123 of 129 | |
| Rule by shape (`checks/shape_eval.py`: "cubes left, cylinders in the middle, prisms right"), 8 episodes | 32 of 33 | |
| Prisms lying on their side (`checks/patch_lying.py` changes how the simulator spawns them), 8 episodes | 33 of 34 | |
| Everything the simulator varies, varied twice as much (`sim/dr/hard_double.yaml`), fixed task, 16 episodes | 64 of 66 | |
| The same with an instruction, 16 episodes | 61 of 65 | |

Seeds: fixed task 400 to 415, example sentences 420 to 427 and 620 to 627, unseen sentences 820 to 851, shape rule 420 to 427,
lying prisms 500 to 507. The two doubled lines use the seeds of the first two lines.

No block ended in a wrong box in these 112 episodes. By the text test below, about 3 in 100 unseen sentences would cause one. 16 of 458 blocks never reached a box. 10 it never lifted: the hand closed on air or the block slipped out. 8 of those were prisms near the middle of the counter. It repeated the same grip until it gave up after eight or nine tries or time ran out. Changing the grip after a miss is the fix we ran out of time for. 5 more were dropped on the way and ended out of sight or reach. The last one it could not reach at all.

The organisers' gate: 0 clipped steps and 68 to 91 Hz where 27 is required. The container answers 1.1 s after it starts, on a
CPU, without a network.

Instructions, parser only: 1,788 sentences a language model wrote to trip it up, each confirmed by a second model. We
did not read them while building the parser: we fixed it three times on a separate set and looked only at the totals of
these. 94.7 %
come out fully right, 3.3 % would put a block in a wrong box, the rest would leave a block on the counter.

What we don't know:

- It has not touched a real block. The organisers' real recordings have no depth image and no labels, so on real data
  we could only count how often each colour is reported: red and yellow in about 3 of 4 of 1,752 frames, blue in 9 of
  10.
- The sagging arm. The organisers' code says the real arm sags under its weight (`interface/check_safety_real.py`),
  which we work out to about 2 cm at the hand. The brain measures the gap between commanded and measured joints at
  every look and corrects for it. The simulator's arm is four times stiffer, so this has only run in our mock world.
- Camera, counter or boxes. When the depth image puts the counter or the box row more than 1.5 cm out of place, either
  the camera moved or the thing itself did. The simulator only ever moves the camera, so with a stiff arm (the
  simulator's) the brain corrects the camera. With a sagging arm it assumes the real robot: it moves the counter, and
  for the boxes it goes by the size of the shift and switches after two empty grasps. Those branches have never run
  in the simulator.
- Time. Half the episodes finish within 59 seconds. 5 of 112 ran out of the 120 seconds, and the real robot may be slower, so expect a block left behind in some real episodes.

## Run it

```bash
# the robot model, once (the organisers' setup script does the same)
git clone https://github.com/unitreerobotics/xr_teleoperate third_party/xr_teleoperate

docker build -f aaltonomous/docker/vaiski.Dockerfile -t vaiski .
docker run --rm -p 8000:8000 vaiski

# the organisers' pre-check: startup time and command rate
python -m aaltonomous.deploy.replay --synthetic 100 --container vaiski --gpus none

# scored episodes in Isaac (a cloud RTX machine with the sim running, see QUICKSTART.md)
python -m aaltonomous.sim.evaluate --container vaiski --team vaiski --task sort_language --trials 8
```

## Check it yourself, without a GPU

```bash
pip install numpy scipy opencv-python-headless pyyaml msgpack websockets pin pytest

python -m pytest -q aaltonomous/vaiski                      # the parser's unit tests

# three episodes in the organisers' mock world (rendered images, no physics)
python aaltonomous/vaiski/checks/mock_eval.py --task sort_language

# the same with broken inputs: black frames, no depth, a look that fails, a plan that fails
python aaltonomous/vaiski/checks/fault_eval.py --fault black_first
# a soft arm (stiffness 80, no gravity compensation) and a counter 3 cm higher than the scene file says
python aaltonomous/vaiski/checks/fault_eval.py --sag_kp 80 --counter_dz 0.03

# the 1,788 unseen instructions (three files; 20 kinds of sentence, some in Finnish and other languages)
cat aaltonomous/vaiski/checks/data/unseen[123].jsonl > /tmp/unseen.jsonl
python aaltonomous/vaiski/checks/eval_instructions.py . /tmp/unseen.jsonl
```

With the simulator running, `checks/unseen_eval.py` is the organisers' `sim/evaluate.py` with one change: each
episode keeps their colour-to-box mapping and layout, and the sentence is swapped for an unseen one that asks for
exactly that sorting.

```bash
UNSEEN=/tmp/unseen.jsonl python aaltonomous/vaiski/checks/unseen_eval.py --container vaiski --team vaiski --task sort_language --trials 8
```
