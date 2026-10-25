# agilex: AgileX PiPER + Pika gripper + wrist D405

`adapter.py` wraps the robot-proven see-and-grab code in [dual-piper-teleop](https://github.com/ruosili/dual-piper-teleop)
(`src/dualpiper/grab/`, its `docs/grab.md`): the arm, gripper and camera backend, the depth geometry and the grasp. Calibration and
every grasp number stay in that repo's configs (`configs/grab.yaml`, `configs/llm-real.yaml`, the camera calibration), with their
provenance there; none is set here.

| Interface | Implementation |
| --- | --- |
| `frame` | D405 colour + depth, joint angles at capture |
| `detect` | none: the core's Gemma boxes (only without motion; with motion `survey` finds the cans) |
| `survey` | `Grab.survey`: reads labels from 3 low poses, votes, measures from the view pose; returns each can with its best crop |
| `listen` / `say` | `arecord` 16 kHz mono (USB speakerphone) / `spd-say -w` |
| `locate` | `Grab.look` (depth points in the box -> can axis, top, diameter in the base frame); refuses too wide, out of reach, crowded |
| `pick` | `Grab._pick`: pre-grasp, correction from close up, straight approach with a contact stop, close, jaws vs measured diameter, 6 cm lift |
| `holding` | gripper width against the measured diameter |
| `lift` / `offer` | already lifted by `pick`; held where it is (no hand-over pose measured yet) |
| `release` / `home` | open / back out and return to the view pose; `shutdown` folds to rest unless a stop or failure left it holding |

## Run

Needs dual-piper-teleop checked out (default `~/work/MoissanClub/dual-piper-teleop`, or `--opt repo=...` / `DUALPIPER_REPO`), its
venv (numpy, OpenCV, piper_sdk, pyrealsense2), the arm's CAN up (`can_left`), and its Gemma server on port 8090
(`scripts/serve-gemma.sh` there; 8080 is its Quest console).

```bash
DPT=~/work/MoissanClub/dual-piper-teleop
$DPT/.venv/bin/python -B run.py -e agilex --server http://127.0.0.1:8090/v1 --typed                 # dry: nothing moves
$DPT/.venv/bin/python -B run.py -e agilex --server http://127.0.0.1:8090/v1 --opt motion=true      # owner + E-stop
```

`motion=true` moves to the view pose at start (announced) and asks Enter before each motion stage unless `--opt yes=true`.
Without motion the arm stays where it is and `pick` refuses before moving.

## Status (2026-10-24)

- `check.py`: PASS (with an OpenCV Python; system Python on Linux has no `sips` fallback).
- **Delivered 4 of 5 orders on the arm with motion** (shared loop, `yes=true`, cans in one row ~35-40 cm out): typed "mango" ->
  mango; typed "give me a fanta" -> Coca-Cola (wrong, before the brand fix below), then -> Fanta; spoken "pepsi" (PowerConf mic)
  -> Pepsi, Gemma heard the order in 372 ms. 75-86 s per order: survey ~22 s, pick + lift ~22 s.
- **The wrist camera has to move to see**: from one pose it sees can tops ("silver chilly"), no flavour. `survey()` runs
  `Grab.survey`: three low reading poses (`read_q_deg`), a majority vote per can, positions from the view pose.
- **The guest's word must be in the description**: the vote names the flavour ("orange"), and "give me a fanta" chose
  "coca-cola" (Fanta is a Coca-Cola brand): 3 of 4 phrasings, `none` for the 4th, with either choose prompt. The core now describes each can's largest
  reading-pose crop and keeps the voted name (`Gemma.describe_named`): "orange fanta" -> 7/7 orders right on the saved crops, 1/1 on
  the arm.
- "Out of reach" refusals earlier in the day were right: back-row cans measured 46-48 cm out, the workspace ends at 45 cm.
- Open: a can read from a reading pose but not boxed from the view pose is left out of the choice (mango at y = -27 cm, 3 runs;
  Pepsi once); repeated words in a name ("orange fanta orange"); "take it" released on the first clip, heard as "Thank you."
  (`thank` is a let-go word); image left/right vs robot left/right for position words is unchecked.
