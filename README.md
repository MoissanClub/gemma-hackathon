# Cross-embodiment Gemma harness

**Claim: one harness, minimal adaptation per robot.** A small on-device Gemma (4 E2B, llama.cpp) does the perception and the
decisions: it describes what it sees, hears the guest's order in any language, picks the object they meant, checks the hand, and
hears "take it". Each robot plugs in through one small interface (`core/embodiment.py`): frames, the object's position, and a
handful of skills (pick, lift, offer, release, home). The task loop, the prompts, the logging and the robot API are shared.

The design comes from what worked on a Unitree G1 (left hand, Jetson Orin NX): **Gemma answers one direct perception question per
request, after the image; code does geometry, motion and safety.** Brain v2 on the G1 grasped a can on its own 3 of 3 on
2026-10-23. What we measured and why the split is drawn here: [LEARNINGS.md](LEARNINGS.md).

```
               ┌────────────── core/ (shared, stdlib only) ──────────────┐
  guest audio ─►  gemma.py   describe / choose / in-hand / take-it / box  │
  camera frame ─► loop.py    LOOK > LISTEN > UNDERSTAND > CONFIRM >       │
                             LOCATE > PICK > VERIFY > LIFT > HANDOVER     │
                  robot_api.py  look() pick() give() place() say() listen()│
               └──────────────────────┬──────────────────────────────────┘
                                      │ core/embodiment.py (one interface)
          ┌───────────────┬───────────┴────────┬─────────────────┐
     embodiments/mock  embodiments/g1    embodiments/agilex   embodiments/<yours>
     (recorded G1      (brain v2: URDF   (skeleton: a
      senses, no hw)    reach, servo...)  teammate's port)
```

## Status

| Embodiment | Shared loop (`run.py`) | Robot-proven program | Owner |
| --- | --- | --- | --- |
| `mock` | runs end to end, fake or real Gemma | - | reference |
| `g1` | adapter mapped, **not wired** | `embodiments/g1/brain/brain.py` (brain v2) | Junda |
| `agilex` | delivered 4 of 5 orders on the arm (typed and spoken), survey from 3 reading poses | dual-piper-teleop `grab` (wrapped) | teammate |

## Quick start (Mac)

```bash
python3 -B check.py                                   # offline checks, no model, ~3 s
python3 -B run.py --embodiment mock --fake-gemma      # the whole loop with scripted answers
serve/run_llama.sh                                    # Gemma 4 E2B on 127.0.0.1:8080 (first run downloads ~3.6 GB)
python3 -B run.py --embodiment mock                   # the loop with real Gemma: a real G1 head frame and two real spoken orders
python3 -B run.py --embodiment mock --typed           # type the order instead
python3 -B run.py --embodiment mock --task tasks/give_leftmost.py   # a task program on the shared robot API
```

On a Jetson: `serve/serve-orin.sh` (paths inside are PC2's: llama.cpp built with CUDA in `~/llama.cpp`, models in `~/models/gemma4`).
Python 3.9+ and nothing else for `core/`; OpenCV is used when installed. Logs go to `runs/<session>/` (git-ignored).

## Add an embodiment

1. `cp -r embodiments/mock embodiments/<name>` (or start from `embodiments/agilex/adapter.py`), add the name to
   `embodiments/__init__.py` `KNOWN`.
2. Implement `frame`, `locate`, `pick`, `release`, `home`; add `detect`, `listen`, `say`, `holding`, `lift`, `offer` when the robot
   has them. Missing senses change the loop, not the prompts: no mic -> typed orders, no detector -> Gemma boxes, no grip sensor ->
   Gemma checks the hand.
3. `python3 -B run.py -e <name> --fake-gemma`, then with the server. Keep robot-specific code, calibration, scripts and notes in
   your folder (see `embodiments/g1/`). Do not change `core/` prompts for one robot: override `Embodiment.prompts` and bench it.

## Layout

| Path | What |
| --- | --- |
| `core/` | `gemma.py` (the verified Gemma calls), `embodiment.py` (the interface), `loop.py` (the task loop + JSONL log), `robot_api.py` (the codegen API on any robot), `image.py`, `client.py` (timed llama.cpp client), `fake.py` (scripted answers) |
| `embodiments/` | one folder per robot: `adapter.py` + its own code, calibration, scripts, notes |
| `tasks/` | task programs for `robot_api` (`def task(robot):`), the same file on every robot |
| `bench/` | latency/accuracy benchmark and the codegen probe; images are generated, not committed (`bench/README.md`) |
| `serve/` | llama.cpp launch scripts (Mac, Orin) and the model fetch script |
| `docs/` | `LATENCY_RESULTS.md`, Orin model sizes, LiteRT GPU, `DEV_WORKFLOW.md`, session templates |
| `check.py`, `run.py` | offline checks; one entry point for every robot |

Source history: the `harness` repo (branch `hackathon`, commit `1f13fc6`); experiment logs, frames and the research corpus
stayed there to keep this repo small.
