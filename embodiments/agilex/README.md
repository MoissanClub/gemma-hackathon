# agilex: AgileX PiPER + Pika gripper + wrist D405

`adapter.py` wraps the robot-proven see-and-grab code in [dual-piper-teleop](https://github.com/ruosili/dual-piper-teleop)
(`src/dualpiper/grab/`, its `docs/grab.md`): the arm, gripper and camera backend, the depth geometry and the grasp. Calibration and
every grasp number stay in that repo's configs (`configs/grab.yaml`, `configs/llm-real.yaml`, the camera calibration), with their
provenance there; none is set here.

| Interface | Implementation |
| --- | --- |
| `frame` | D405 colour + depth, joint angles at capture |
| `detect` | none: the core's Gemma boxes |
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
- Dry run of the shared loop on the arm, folded at rest, motion off, typed orders (n=2): LOOK boxed 4 cans and described them;
  "the white peach please" -> `none` (no peach among the 4 read) -> the honest negative; "the watermelon one" -> can 2 -> LOCATE
  measured it and refused it as out of reach. From the folded pose the D405 sees part of each can (the same pose measured a 57 mm
  can as 35 mm in dual-piper-teleop plan-only runs the same day); `motion=true` looks from the view pose, where dual-piper-teleop's
  own grab read and measured it right (57 mm, grasped 1/1).
- Not yet run: the shared loop with motion; the spoken order and "take it" on this mic; image left/right vs robot left/right for
  position words (unchecked for the wrist camera).
- The core's prompts describe cans by colour first ("silver chi watermelon"); every can here is silver, so flavour does the work.
  dual-piper-teleop's flavour read (2x crop, flavour-only question) is its measured variant for these cans; not used here yet.
