#!/usr/bin/env python3
"""MOVES THE BASE. A short straight drive of R1's omni base on the Feetech bus (wheels 7-9 only; the arm's ids 1-6 are never addressed),
from Stefan's base_geometry.yaml (wheel radius, speed unit, layout: MEASURED there; the drive directions are not, so the first move is
short and measured against the wrist camera before a longer one).

  drive.py --dist 0.03 [--speed 0.04] [--sign 1]     # forward 3 cm at 4 cm/s; --sign -1 if the measurement says it went backward

Velocity mode is set on the wheels for the move (not persistent: the EEPROM lock stays on, a power cycle restores mode 0); at the end,
on Ctrl-C or on any error, every wheel's speed is set to 0 and its torque switched off. A process killed outright (SIGKILL) cannot do
that and the wheels keep turning: keep a hand on the power switch."""
import argparse, math, os, sys, time
import scservo_sdk as s
import yaml

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dist", type=float, required=True, help="metres, forward positive; at most 0.15")
ap.add_argument("--speed", type=float, default=0.04, help="m/s, at most 0.08")
ap.add_argument("--sign", type=int, default=1, choices=(1, -1), help="-1 flips forward/backward (the unmeasured drive direction)")
ap.add_argument("--geometry", default="~/stefan-calib/home/stefan/workspace/seraphos/deploy/robots/r1/base_geometry.yaml")
ap.add_argument("--port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6057345-if00")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()
if abs(a.dist) > 0.15 or not 0 < a.speed <= 0.08: sys.exit("refused: --dist within 0.15 m, --speed within 0.08 m/s")

g = yaml.safe_load(open(os.path.expanduser(a.geometry)))["omni"]
ids, angles, signs = g["servo_ids"], g["wheel_angles_deg"], g["signs"]
assert ids == [7, 8, 9], ids
m_per_unit = 49.5 * 2 * math.pi / 4096 * g["wheel_radius_m"]      # base_geometry: 1 speed unit = ~49.5 steps/s
v = math.copysign(a.speed, a.dist) * a.sign
raw = [int(round(sg * v * math.cos(math.radians(th)) / m_per_unit)) for sg, th in zip(signs, angles)]   # body forward = +x
secs = abs(a.dist) / a.speed
print("drive %+.3f m at %.3f m/s: %.2f s, wheel speeds (raw units) %s for ids %s (%.4f m/s per unit)" % (a.dist, a.speed, secs, raw, ids, m_per_unit), flush=True)
if max(abs(r) for r in raw) > g["max_wheel_raw"]: sys.exit("refused: over max_wheel_raw")
if not a.yes and input("  owner at the robot, power switch in reach, path clear? type 'go': ").strip().lower() != "go": sys.exit("not moving")

ph, pk = s.PortHandler(a.port), s.PacketHandler(0)
if not ph.openPort() or not ph.setBaudRate(1000000): sys.exit("cannot open " + a.port)
sm = lambda x: (abs(x) | 0x8000) if x < 0 else x                 # STS sign-magnitude, bit 15
def speeds(vals):
    for i, x in zip(ids, vals): pk.write2ByteTxRx(ph, i, 46, sm(x))
def stop():
    for _ in range(3):
        speeds([0, 0, 0])
        for i in ids: pk.write1ByteTxRx(ph, i, 40, 0)
pos0 = [pk.read2ByteTxRx(ph, i, 56)[0] for i in ids]
try:
    for i in ids:
        pk.write1ByteTxRx(ph, i, 40, 0); pk.write2ByteTxRx(ph, i, 46, 0); pk.write1ByteTxRx(ph, i, 33, 1)   # velocity mode, speed 0
        pk.write1ByteTxRx(ph, i, 40, 1)
    t0 = time.monotonic(); ramp = min(0.3, secs / 3)
    while (t := time.monotonic() - t0) < secs:
        f = min(1.0, t / ramp, (secs - t) / ramp + 0.2)              # short ramp in and out
        speeds([int(round(r * f)) for r in raw]); time.sleep(0.05)
finally:
    stop()
    pos1 = [pk.read2ByteTxRx(ph, i, 56)[0] for i in ids]
    modes = [pk.read1ByteTxRx(ph, i, 33)[0] for i in ids]
    ph.closePort()
    print("stopped, torque off. wheel positions %s -> %s (counts), modes %s" % (pos0, pos1, modes), flush=True)
