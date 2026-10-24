# agilex: AgileX arm (teammate)

Skeleton in `adapter.py`; the interface is `core/embodiment.py`, a complete working example is `embodiments/mock/adapter.py`.

## Steps

1. Camera only: implement `frame()` (return `Frame(bgr_array)` or `Frame(jpeg_bytes)`), keep the mock's fake `locate`/`pick`, run
   `python3 run.py -e agilex` against the server: Gemma should describe the objects on your table. Nothing moves yet.
2. `locate()`: pixel box -> base-frame point. With a depth camera: median depth in the box, deproject, camera-to-base transform,
   then the object axis = front surface + half its diameter. Without depth: a measured table plane + camera calibration. Put every
   measured number in a `calib.json` here with provenance (value, unit, frame, method, date, n).
3. `pick()` / `release()` / `home()` with the arm SDK (PiPER: `piper_sdk` over CAN). Plan the whole path before moving, cap the step
   per control tick, stop on a tracking-error rise. `holding()` from the gripper width or effort.
4. `lift()` / `offer()`, then optional `listen()` (USB mic, 16 kHz mono WAV) and `say()` (any local TTS).

Options reach `make()` from the CLI: `python3 run.py -e agilex --opt can_port=can0 --opt camera=0`.

Notes, calibration and scripts for this arm go in this folder. Report what you measure with n (see `LEARNINGS.md`).
