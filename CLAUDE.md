# Cross-embodiment Gemma harness: rules for agents and teammates

Read `README.md` (layout, quick start) and `LEARNINGS.md` (what is measured) first. Hackathon: Google Gemma Hardware Hackathon,
AGI House, 2026-10-24, Gemma-based and entirely on-device.

## Boundaries

- `core/` is shared by every robot: stdlib only (OpenCV optional), Python 3.9+, terse existing style. A change there must keep
  `python3 -B check.py` passing and must not break another embodiment. Robot-specific code never goes in `core/`.
- `embodiments/<name>/` belongs to that robot's owner: adapter, drivers, calibration, scripts, notes. Do not edit another team's
  folder without asking them.
- The Gemma prompts in `core/gemma.py` are measured variants (G1 and the Mac). A different prompt is a new, unmeasured variant:
  override it per embodiment (`Embodiment.prompts`) and bench it before relying on it.
- `embodiments/g1/brain/` is the robot-proven G1 program (brain v2). Its gate is `python -B check.py` in that folder (numpy + OpenCV;
  80 of 84 on 2026-10-24, the 4 failures are the known mock-hand cases after `palm_to_axis_m` 0.035). Keep that result or better.
  Its rules: `embodiments/g1/notes/ROBOT_RULES.md`; process: `docs/DEV_WORKFLOW.md`.

## Model and server

- Gemma 4 E2B QAT (`unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL` + `mmproj-BF16`), llama.cpp server on `127.0.0.1:8080`, alias
  `gemma4-e2b`, thinking off, greedy. Mac: `serve/run_llama.sh`; Orin: `serve/serve-orin.sh`; bench restarts: `bench/serve.sh`
  (it stops only the port-8080 process; any other llama instance on another port is someone else's).
- One inference job at a time: benchmarks are serial and GPU-exclusive.
- Never time repeated identical images (llama.cpp reuses the image KV). Mock-run latencies are not benchmarks.

## Robots

- A robot moves only with its owner present and an emergency stop in hand. Announce every motion before sending it.
- No hard-coded distances in motion code: measure at runtime or keep the number in a calib file with provenance.
- Do not tune small deltas on the robot: reproduce the fault offline first. No code edits on the robot's computer: deploy from a
  committed tree.
- A stop holds the robot where it is; the way home runs only on the operator's confirmation.

## Repo hygiene (the venue Wi-Fi is slow)

- Do not commit images, videos, model files, bench image sets or run logs (`.gitignore` covers `runs/`, `bench/images*/`,
  `*.gguf`, media files outside `embodiments/*/media/`). Keep `media/` folders to the few small files a mock or check needs.
- Downloads the project needs (models, wheels, sources) are pre-approved; list what was fetched in the run notes.
- Report accuracy with n and per-class counts; Mac numbers are not Orin numbers.
