#!/usr/bin/env python3
"""Gemma-guided grasp on R1: Gemma boxes the object in the wrist Orbbec, depth + Stefan's FK and hand-eye put it in the arm base frame,
and Stefan's pick sequence (seraphos.arm.pick_seq.Picker: hover, depth servo, descent to his floor protection, close, judge, lift,
rest) runs on the bus directly (embodiments/r1/arm.py). His r1-mux must be stopped (it owns the port).

  grasp.py dry   "the white tissue"    # arm limp, read-only: the target, the hover and descent IK, his path checks. Nothing moves.
  grasp.py hover "the white tissue"    # MOVES: to the servo pose over the object, looks twice, back to rest, torque off
  grasp.py diag  "the white tissue"    # MOVES: hover, servo, descend, look twice at the bottom, NO close, back to rest, torque off
  grasp.py pick  "the white tissue"    # MOVES: the whole pick, then rest, torque off
Every look's frame is saved with its boxes and the kinematic TCP drawn in (look*.jpg, depth beside it).

The arm moves only with the owner present and the power switch in reach. Ctrl-C at any point: back to rest, then torque off.
Every event goes to runs/<timestamp>-r1-grasp/."""
import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("mode", choices=("dry", "hover", "diag", "pick"))
ap.add_argument("what")
ap.add_argument("--seraphos", default="~/stefan-calib/home/stefan/workspace/seraphos", help="Stefan's seraphos tree (read-only copy)")
ap.add_argument("--calibration", default="~/stefan-calib/home/stefan/.seraphos/calibration/so101_r1_main.yaml")
ap.add_argument("--ob-config", default="~/r1-gemma/ob-config.xml")
ap.add_argument("--server", default="http://127.0.0.1:8080/v1")
ap.add_argument("--speed", type=float, default=0.8, help="joint speed cap, rad/s (his ceiling 1.5)")
ap.add_argument("--yes", action="store_true", help="do not ask before the first motion")
ap.add_argument("--look", default=None, metavar="X,Y", help="find the object from the hover pose over (X, Y) m in the arm base frame (MOVES), "
                "not from where the arm is: rest sees only the floor past ~30 cm, the arm reaches ~36 cm")
a = ap.parse_args()

from embodiments.r1 import arm as A
root = A.load_seraphos(a.seraphos)
import numpy as np, yaml
from core.gemma import Gemma
from embodiments.r1.adapter import R1
from seraphos.arm.handeye import load_extrinsics
from seraphos.arm.pick import path_clear, descent_waypoints
from seraphos.arm.pick_seq import PickConfig, Picker
from seraphos.arm.so101_kinematics import fk

out = os.path.join(ROOT, "runs", time.strftime("%Y%m%dT%H%M%S") + "-r1-grasp"); os.makedirs(out)
logf = open(os.path.join(out, "events.jsonl"), "a")
def log(msg, **kw):
    print(msg, flush=True); logf.write(json.dumps(dict(kw, msg=msg, t=time.time()), default=float) + "\n"); logf.flush()
deg = lambda q: [round(math.degrees(v), 1) for v in q]

cal_path, prof_path, ext_path = A.robot_files(root)
cal_path = os.path.expanduser(a.calibration) if a.calibration and os.path.exists(os.path.expanduser(a.calibration)) else cal_path
ext = load_extrinsics(ext_path)
prof = yaml.safe_load(open(prof_path))["sink"]
tip_contact_z = prof["tcp_z_min_m"] - 0.020          # arm_profile: tcp_z_min_m = the measured jaw-tip floor contact plus 20 mm
log("calibration %s, extrinsics %s (floor %.4f m below the base), jaw-tip floor contact at TCP z %.4f" % (cal_path, ext_path, ext.floor_below_base_m, tip_contact_z))

gem = Gemma(a.server, on_record=lambda r: logf.write(json.dumps(dict(r, ev="gemma"), default=str) + "\n"))
cam = R1(ob_config=a.ob_config); cam.start()
arm = A.R1Arm(root, speed_rad_s=a.speed, calibration=cal_path)
log("d2c: R diag %s, t %s mm" % (np.round(np.diag(cam.d2c[0]), 4).tolist(), np.round(cam.d2c[1] * 1000, 2).tolist()))
try:
    s = arm.state()
    if s.fault: raise SystemExit("arm: " + s.fault)
    log("arm now (deg) %s, rest (deg) %s, torque %s" % (deg(s.q), deg(arm.rest_q), "on" if s.powered else "off"), q=s.q)
    # the target from where the arm is now (it must be still: limp at rest, or held)
    go = json.load(open(os.path.join(ROOT, "embodiments/r1/grasp-offset.json")))
    log("grasp centre in the TCP frame %s m (grasp-offset.json: %s)" % (go["grasp_centre_tcp_m"], go["provenance"][:60]))
    perc = A.GemmaPerception(cam, gem, arm, ext, a.what, [0.3, 0.0], tip_contact_z, gate_m=10.0, log=log, save_dir=out,
                             grasp_offset=go["grasp_centre_tcp_m"])
    cfg = PickConfig(arm.limits[:4], ext, arm.rest_q[:5], approach_stop_m=0.020)
    pk = Picker(arm, perc, cfg)
    if a.look:                                                 # look from above the reachable floor, not from rest (rest sees only past ~30 cm)
        lx, ly = map(float, a.look.split(","))
        q_look = pk._hover_q(lx, ly)
        log("LOOK POSE over (%.2f, %.2f) (deg) %s, TCP %s, path rest->look clear: %s" % (lx, ly, deg(q_look), np.round(fk(q_look)[:3, 3], 3).tolist(),
            path_clear(s.q[:5], q_look, ext, tcp_min_m=0.03)))
        if a.mode == "dry": log("DRY: the look needs the arm to move; nothing was sent"); raise SystemExit(0)
        log("MOTION: torque on where it is, staged move to the look pose, gripper open; if nothing is seen: back to rest, torque off")
        if not a.yes and input("  owner at the robot, power switch in reach? type 'go' to move: ").strip().lower() != "go":
            raise SystemExit("not moving")
        log("enable: " + arm.enable(True))
        try:
            pk._staged(arm.state().q[:5], q_look, gripper=cfg.gripper_open_rad); pk._gripper_stopped(); arm.wait(0.5)
            p = perc.object_base() or perc.object_base()
        except BaseException as e:
            log("abort at the look pose: %s: %s" % (type(e).__name__, e)); p = None
            if not isinstance(e, Exception): pk._rest(); log("disable: " + arm.enable(False)); raise
        if p is None:
            pk._rest(); log("disable: " + arm.enable(False))
            raise SystemExit("Gemma + depth found no '%s' from the look pose" % a.what)
    else:
        p = perc.object_base()
        if p is None: raise SystemExit("Gemma + depth found no '%s' from this pose" % a.what)
    perc.gate = 0.08
    log("TARGET %s in the arm base frame: x %.3f y %.3f z %.3f m (floor z %.3f; +-1.5 cm per his hand-eye)" % (a.what, p[0], p[1], p[2], -ext.floor_below_base_m), target=p)
    obj, aim = list(perc.target), list(perc.target)
    for _ in range(4):                                         # aim the TCP so that the grasp centre, not the fixed jaw, is over the object
        probe, q_h = list(aim), None                           # the offset at a reachable pose near the aim (it barely changes with the pose)
        for _k in range(12):
            try: q_h = pk._hover_q(probe[0], probe[1]); break
            except RuntimeError: probe = [probe[0] * 0.97, probe[1] * 0.97]
        if q_h is None: raise SystemExit("no reachable hover near (%.3f, %.3f)" % tuple(aim))
        T = fk(q_h); dgc = perc.grasp_centre(T) - T[:3, 3]
        aim = [obj[0] - float(dgc[0]), obj[1] - float(dgc[1])]
    log("grasp centre minus TCP in the base frame (%.3f, %.3f) m: TCP aim x %.3f y %.3f for the object at (%.3f, %.3f)" % (dgc[0], dgc[1], aim[0], aim[1], obj[0], obj[1]))
    p = [aim[0], aim[1], p[2]]
    try: q_h = pk._hover_q(p[0], p[1])
    except RuntimeError as e:
        if arm.powered: pk._rest(); log("disable: " + arm.enable(False))
        raise SystemExit("%s: move the object about %.0f cm closer to the robot" % (e, max(1, 100 * (math.hypot(*aim) - 0.355))))
    log("TCP aim x %.3f y %.3f so the grasp centre is over the object (%.3f, %.3f)" % (p[0], p[1], perc.target[0], perc.target[1]))
    log("hover/servo pose (deg) %s, TCP %s" % (deg(q_h), np.round(fk(q_h)[:3, 3], 3).tolist()), q_hover=q_h)
    log("path rest->hover clear: %s; now->hover clear: %s" % (path_clear(arm.rest_q[:5], q_h, ext, tcp_min_m=0.03), path_clear(s.q[:5], q_h, ext, tcp_min_m=0.03)))
    z0 = -ext.floor_below_base_m + cfg.servo_m; z1 = tip_contact_z + cfg.approach_stop_m
    wps = descent_waypoints(p[0], p[1], z0, z1, 0.02, cfg.tilt_deg, cfg.limits, cfg.rest_q[4], q_h[:4])
    log("descent %.3f -> %.3f m: %d waypoints, last (deg) %s" % (z0, z1, len(wps), deg(wps[-1].q)))
    log("lift to TCP z %.3f, then rest" % (-ext.floor_below_base_m + cfg.hover_m + cfg.lift_m))
    if arm.clamps: log("clamped: %s" % arm.clamps)
    if a.mode == "dry":
        log("DRY: nothing was sent to the arm"); raise SystemExit(0)

    plan = {"hover": "torque on where it is, staged move to the servo pose over the %s, gripper open, look twice, back to rest, torque off",
            "diag": "torque on where it is, staged move over the %s, gripper open, up to 3 servo corrections, descend in steps to the floor "
                    "protection or contact, look twice there (no close), up, back to rest, torque off",
            "pick": "torque on where it is, staged move over the %s, empty close and open, up to 3 servo corrections, descend in steps to the "
                    "floor protection or contact, close, judge, lift 10 cm, judge again, back to rest, torque off"}[a.mode] % a.what
    log("MOTION PLAN (%s, speed cap %.1f rad/s): %s" % (a.mode, a.speed, plan))
    if not arm.powered:
        if not a.yes and input("  owner at the robot, power switch in reach? type 'go' to move: ").strip().lower() != "go":
            raise SystemExit("not moving")
        log("enable: " + arm.enable(True))
    try:
        if a.mode == "hover":
            pk._staged(arm.state().q[:5], q_h, gripper=cfg.gripper_open_rad)
            for i in range(2):
                d = perc.object_minus_tip()
                log("at hover %d: object minus tip %s cm" % (i, None if d is None else [round(d.dx * 100, 1), round(d.dy * 100, 1)]))
            pk._rest()
        elif a.mode == "diag":                                    # live_pick.py's diag, with our perception
            x, y = p[0], p[1]
            pk._staged(arm.state().q[:5], q_h, gripper=cfg.gripper_open_rad); pk._gripper_stopped()
            for i in range(cfg.servo_iterations):
                d = perc.object_minus_tip()
                log("servo %d: object minus tip %s cm" % (i, None if d is None else [round(d.dx * 100, 1), round(d.dy * 100, 1)]))
                if d is None or math.hypot(d.dx, d.dy) < cfg.servo_tol_m: break
                x, y = x + d.dx, y + d.dy; q_h = pk._hover_q(x, y); pk._go(q_h, cfg.gripper_open_rad); arm.wait(0.5)
            log("descent ended: %s" % pk._descend(x, y, q_h))
            T = fk(arm.state().q[:5]); gc = perc.grasp_centre(T)
            log("at the bottom: TCP %s, grasp centre %s (TCP aimed x %.3f y %.3f)" % (np.round(T[:3, 3], 3).tolist(), np.round(gc, 3).tolist(), x, y))
            for i in range(2):
                q_b, f_b, found = perc.look()
                log("bottom look %d: %s" % (i, ["(%.3f, %.3f, %.3f) minus grasp centre (%.1f, %.1f) cm" % (pp[0], pp[1], pp[2], (pp[0] - gc[0]) * 100,
                                                (pp[1] - gc[1]) * 100) for _, pp, _ in found]))
            pk._go(pk._hover_q(x, y), cfg.gripper_open_rad); pk._rest()
        else:
            rep = pk.run(p[0], p[1])
            for e in rep.events: log("  pick: %s" % json.dumps(e, default=float)[:240], event=e)
            log("OUTCOME %s after %d attempt(s)" % (rep.outcome, rep.attempts))
    except BaseException as e:
        log("abort: %s: %s" % (type(e).__name__, e))
        try: pk._rest()
        except Exception as e2: log("rest failed: %s" % e2)
        raise
    finally:
        s = arm.state()
        log("end pose (deg) %s; clamps %s" % (deg(s.q), arm.clamps))
        log("disable: " + arm.enable(False))
finally:
    if arm.powered:                                            # an error between two motions: rest first, a released arm up high would drop
        try: pk._rest(); log("rested after an error; disable: " + arm.enable(False))
        except Exception as e: log("rest failed: %s" % e)
    arm.close(); cam.shutdown(); logf.close()
    print("log", out)
