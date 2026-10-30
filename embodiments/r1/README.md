# r1: Stefan's LeKiwi-style robot, wrist Orbbec perception

R1 (`r1.tail190602.ts.net`, Jetson Orin Nano Super 8 GB, Ubuntu 24.04, JetPack R39) is Stefan's robot: SO-101 arm, omni base, IMU,
lidar. His `seraphos` stack runs there as `stefan` user services (`r1-mux` servos and base, `r1-imu`, `r1-arm-pathd`, `r1-endpoint`
teleop, `r1-legate`, `r1-wristcam`). This folder runs the AgileX recipe (Gemma boxes the object, the wrist depth camera measures it)
from the `rosie` account. **Perception only**: nothing here commands a motor, and `pick` refuses.

| Interface | Implementation |
| --- | --- |
| `frame` | wrist Orbbec Gemini 305: 848x530 colour (YUYV) + depth aligned to colour in software, intrinsics in `meta["K"]` |
| `detect` | none: the core's Gemma boxes |
| `locate` | floor plane fitted to a ring of depth around the box; the box points more than max(`min_height`, 3 sigma of the ring) above it, if at least 5% of the box: centroid, height, width. Otherwise `flat` (measured where the box meets the floor). Wrist camera optical frame (x right, y down, z forward) |
| `pick` | refuses: no arm interface, no hand-eye calibration yet |

## Cameras on R1

| Device | What | Notes |
| --- | --- | --- |
| Orbbec Gemini 305 (`/dev/video0-9`) | on the wrist: the jaws in the bottom of the image | colour `/dev/video6` via V4L2; depth only through the SDK |
| RealSense D435 (`/dev/video10-15`) | low on the base, forward, slight roll | 640x480 depth valid 80% (Oct 29, dark room) |
| `USB2.0_CAM1` (`/dev/video16`) | low on the base, forward | served by `r1-wristcam` despite its name |

## Setup (once, as rosie; done 2026-10-29)

`~/r1-gemma/`: `.venv` (numpy, opencv-python-headless, pyrealsense2 2.59.1, pyorbbecsdk2 2.1.2), `models/` (Gemma 4 E2B QAT
`UD-Q4_K_XL` + `mmproj-BF16` from unsloth), `llama.cpp/` (CUDA build for sm_87), `harness/` (this repo, from `git archive` of a
commit). The Orbbec SDK needs its V4L2 backend (R1 has no Orbbec udev rule, so libusb cannot open it) and must not reboot a shared
camera:

```bash
X=~/r1-gemma/.venv/lib/python3.12/site-packages/pyorbbecsdk/OrbbecSDKConfig.xml
sed -e 's|<LinuxUVCBackend>Auto<|<LinuxUVCBackend>V4L2<|' \
    -e 's|<LinuxUVCAutoRebootOnFault>true<|<LinuxUVCAutoRebootOnFault>false<|g' $X > ~/r1-gemma/ob-config.xml
```

## Run

Stefan's `r1-wristcam` holds the Orbbec's depth and metadata nodes (`fuser`: its pid on `/dev/video0`, `1`, `7`). Pause it, and
start it again afterwards (needs rosie's sudo password, so from a terminal):

```bash
ssh -t rosie@r1 'sudo -u stefan XDG_RUNTIME_DIR=/run/user/2002 systemctl --user stop r1-wristcam'    # ... start r1-wristcam
```

Deploy a committed tree, serve Gemma (`scripts/serve-r1.sh`, port 8080; `serve-r1.sh stop`), probe:

```bash
git archive --format=tar HEAD | ssh rosie@r1 'mkdir -p ~/r1-gemma/harness && find ~/r1-gemma/harness -mindepth 1 -maxdepth 1 ! -name runs -exec rm -rf {} + && tar -x -C ~/r1-gemma/harness'   # keeps runs/
ssh rosie@r1 'cd ~/r1-gemma/harness && ../.venv/bin/python -B check.py'
ssh rosie@r1 '~/r1-gemma/harness/embodiments/r1/scripts/serve-r1.sh'
ssh rosie@r1 'cd ~/r1-gemma/harness && ../.venv/bin/python -B embodiments/r1/scripts/look.py --ob-config ~/r1-gemma/ob-config.xml "the crumpled paper"'
ssh -t rosie@r1 'cd ~/r1-gemma/harness && ../.venv/bin/python -B run.py -e r1 --typed --opt ob_config=~/r1-gemma/ob-config.xml'
```

## Status (2026-10-29)

- `check.py` PASS on R1 (its venv has OpenCV).
- `locate` on one saved frame, hand-drawn boxes: a crumpled tissue at z 0.337 m, 3.7 cm tall, 3.7 cm wide; the same with a box
  20 px looser (1.2 mm apart); three bare-floor boxes at 0.31, 0.38 and 0.55 m all `flat`. The cut is 2.2 cm at 0.35 m (floor depth
  noise sigma ~7 mm): lower objects come back `flat`. n = 1 frame: not a measurement of accuracy.
- Live, `look.py "the white tissue" --n 3` (Gemma 4 E2B on R1's Orin Nano, llama.cpp 3c809f4, CUDA sm_87): a box on the tissue in
  3/3 fresh frames, 2271 ms for the first call then 1049 and 1041 ms; located at z 0.336-0.340 m, x 3.7-3.8 cm right, 3.9-4.1 cm
  tall. n = 3 frames of one scene. R1 memory with the server up: 6.3 of 7.4 GB used.
- Arm (read only so far): `/dev/ttyACM0` at 1 Mbaud, STS3215 ids 1-6 (pan, lift, elbow, wrist flex, wrist roll, gripper) and 7-9
  (wheels); servo EEPROM uncalibrated (homing offset 0, limits 0-4095): Stefan's calibration (`~/.seraphos/calibration/
  so101_r1_main.yaml`, his hand sweep 2026-10-19) holds the zero, the joint limits and the rest pose, and his
  `seraphos/arm/so101_kinematics.py` the FK/IK. The bus is free only while his `r1-mux` is stopped.
- Next: base-frame targets (his hand-eye calibration); joint state at capture and a hand-eye calibration (base-frame targets); then the arm,
  through Stefan's interface and with him present.
