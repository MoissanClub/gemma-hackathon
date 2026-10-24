# g1: Unitree G1 (Junda)

Left BrainCo hand (the right thumb motor is faulty), D435i head camera, C922 webcam as mic and third camera, Jetson Orin NX 16 GB
("PC2") running Gemma 4 E2B in llama.cpp. The robot stays clipped into its stand, FSM 500, no walking.

| Path | What |
| --- | --- |
| `brain/` | **brain v2**, the robot-proven program: one process, the whole order loop. `brain/README.md` is the full description; `HANDOFF.md`, `V2_STRATEGY.md`, `DESIGN_REVIEW.md`, `V2_REVIEW.md` are its design and review history. |
| `brain/media/` | the three office files `brain/mock_robot.py` needs for the desk check |
| `adapter.py` | the shared-harness adapter: every method mapped to the brain v2 code that implements it, **not wired yet** |
| `scripts/` | the spike scripts of Oct 18-23 (arm, hand, mic, camera, LiteRT probes, `teach_reach.py` hand-guiding). They import from `/home/unitree/harness/bench` and `/home/unitree/g1-fetch` on PC2. |
| `notes/` | `ROBOT_RULES.md` (hard rules), `INCIDENTS.md`, `HACKATHON_PLAN.md` (the demo plan and what is verified), the Orin Sunday, office-grab and v2-grasp session notes |

## Status (2026-10-24)

- 2026-10-23: brain v2 grasped a 66 mm can on its own 3 of 3 (`--model-grasp`, can about 40 cm ahead, 14-15 cm left, desk at about
  88 cm, table and edge from depth); pre-grasp within 2-4 mm every run. `notes/2026-10-23-v2-grasp.md`.
- Open: the way home from the raised offer pose (`plan_retreat` holds), the let-go words (misheard), the 4 mock-hand failures in
  `brain/check.py`, identification under the venue's lighting.

## Desk check (Mac, no robot)

```bash
cd embodiments/g1/brain && python -B check.py     # numpy + OpenCV; ~2 min; 80 of 84 on 2026-10-24 (servo, contact, fwd39, fwd54 known)
```

## On the robot

`brain/README.md` "Start on PC2". `brain.py` imports `core/client.py` from this repo (`brain/gemma.py`), changes directory to
`~/g1-fetch` itself and uses its robot, detector and arm streamer. PC2 today runs the `harness` repo's copy
(`~/harness/hackathon/brain`); to run from this repo instead, clone it on PC2 and use this folder in place of that path (the serve
script writes its launch line to `~/harness/bench/server-launch.txt`).

## Porting brain v2 onto the shared loop

`adapter.py` lists, per interface method, the `Brain` method that already does it. brain v2's LOOK / LISTEN / UNDERSTAND / CONFIRM
are the same Gemma calls and sentences as `core/loop.py`, so the port wraps the physical stages only. Do it on the Mac against
`brain.py --mock --mock-gemma` first and keep `brain/check.py` at its result.
