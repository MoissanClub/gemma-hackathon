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
- **Motion orders, afternoon** (shared loop, `yes=true`, cans in one row ~35-40 cm out): typed "mango" ->
  mango; typed "give me a fanta" -> Coca-Cola (wrong, before the brand fix below), then -> Fanta; spoken "pepsi" (PowerConf mic)
  -> Pepsi, Gemma heard the order in 372 ms. 75-86 s per order: survey ~22 s, pick + lift ~22 s.
- **The wrist camera has to move to see**: from one pose it sees can tops ("silver chilly"), no flavour. `survey()` runs
  `Grab.survey`: three low reading poses (`read_q_deg`), a majority vote per can, positions from the view pose.
- **The guest's word must be in the description**: the vote names the flavour ("orange"), and "give me a fanta" chose
  "coca-cola" (Fanta is a Coca-Cola brand): 3 of 4 phrasings, `none` for the 4th, with either choose prompt. The core now describes each can's largest
  reading-pose crop and keeps the voted name (`Gemma.describe_named`): "orange fanta" -> 7/7 orders right on the saved crops, 1/1 on
  the arm.
- "Out of reach" refusals earlier in the day were right: back-row cans measured 46-48 cm out, the workspace ends at 45 cm.
- **Tally, 12 motion orders today**: right can 6 (typed mango, typed Fanta x2, voice Pepsi x2, typed watermelon); wrong can 4
  (Fanta -> Coca-Cola before the brand fix; voice: Pepsi -> watermelon, Pepsi -> apple, Fanta -> peach); refused 1 (Fanta at
  45.3 cm, past the workspace); unconfirmed 1 (heard "Coca-Cola", delivered Coca-Cola).
- Evening (after b3e8b34): voice Pepsi delivered on the second try (the first clip was room talk: audio and transcript chose
  differently, so it asked again); typed Fanta and watermelon delivered. Voice in the venue failed 3 times: a silent clip chose
  watermelon, "hapsy" chose apple, room talk with "3" chose can 3 of 6 (both reads agreed on the number).
- Fixed: a can seen only from the reading poses (closer than ~30 cm: the view pose cannot see it) is offered with its
  reading-pose position, left/right from the measured y; LOCATE uses the survey's measurement (a second Gemma boxing missed the
  Pepsi); an order needs a non-empty transcript and the same can from the clip and from its transcript; a can whose reads tie
  keeps every tied name.
- Open: numbered labels ("number 3" with more than 3 cans) let a spoken number pick a can; cans closer than a few cm merge in the
  survey (a box over two cans; the Fanta vanished into the watermelon once); cans past 45 cm are refused (the workspace edge);
  "take it" also fires on "take care" and "thank you"; a quiet room or typed orders for the demo.
