"""R1's SO-101 straight on its Feetech bus, for Stefan's pick sequence (seraphos.arm.pick_seq.Picker), with Gemma as its eyes.

Stefan's seraphos tree is used read-only from a path (`load_seraphos`): his driver (FeetechStsDriver: present pose -> goal before
torque on), kinematics, hand-eye and pick sequence, and his calibration files (zero_raw, q_lim, rest pose, camera extrinsics). Nothing
of his is copied into this repo. What is ours:

  R1Arm          his ArmClient protocol on the bus directly (his r1-mux must be stopped: it owns the port). Servo ids 1-6 only; the
                 wheels (7-9) are never addressed. Every target is CLAMPED to q_lim before it is sent (his command rule), speed capped.
  GemmaPerception  his Perception protocol: Gemma boxes the object in a fresh wrist frame, the Orbbec depth locates it (adapter.R1),
                 his FK and hand-eye map it to the arm base frame. The jaw height is kinematic (TCP height against the measured jaw-tip
                 floor contact, arm_profile.yaml), so the descent stops at his floor protection, not on a depth guess.
"""
import math, os, sys, time, types

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def load_seraphos(root):
    """Put Stefan's seraphos checkout (read-only copy) on the path. seraphos.sensor.floor_objects is not in the copy: pick.py uses it
    only in detect_in_base (never called here) and in annotations (never evaluated), so an empty module stands in for it."""
    root = os.path.expanduser(root)
    if not os.path.isdir(os.path.join(root, "seraphos", "arm")): raise SystemExit("no seraphos/arm under %s" % root)
    sys.path.insert(0, root)
    if "seraphos.sensor.floor_objects" not in sys.modules:
        import seraphos
        sensor = types.ModuleType("seraphos.sensor"); fo = types.ModuleType("seraphos.sensor.floor_objects")
        sensor.floor_objects = fo; sys.modules["seraphos.sensor"] = sensor; sys.modules["seraphos.sensor.floor_objects"] = fo
    return root


class SerialPort:
    """The two-method transport his driver wants, on pyserial, opened exclusively."""
    def __init__(self, path, baud=1000000):
        import serial
        self.s = serial.Serial(path, baud, timeout=0, exclusive=True)

    def write(self, data): self.s.write(data)

    def read(self, n, timeout):
        end, out = time.monotonic() + timeout, b""
        while len(out) < n:
            out += self.s.read(n - len(out))
            if out or time.monotonic() >= end: break
            time.sleep(0.0005)
        return out

    def close(self): self.s.close()


def robot_files(root):
    """(calibration, profile, extrinsics) paths: his ~/.seraphos calibration where the copy has it, else the deploy file."""
    d = os.path.join(root, "deploy", "robots", "r1")
    return os.path.join(d, "arm_calibration.yaml"), os.path.join(d, "arm_profile.yaml"), os.path.join(d, "camera_extrinsics.yaml")


class R1Arm:
    """seraphos.arm.pick.ArmClient on the bus. q in radians, JOINTS order."""
    def __init__(self, root, port="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6057345-if00", speed_rad_s=0.8, acc_rad_s2=8.0,
                 calibration=None):
        import json, yaml
        from seraphos.arm.drivers import JointSpec
        from seraphos.arm.drivers.feetech_sts import FeetechStsDriver
        cal_path = calibration or robot_files(root)[0]
        cal = yaml.safe_load(open(cal_path))["joints"]
        model = {j["name"]: j for j in json.load(open(os.path.join(root, "seraphos/arm/models/so101/model.json")))["joints"]}
        self.specs = [JointSpec(n, model[n]["motor"], model[n]["bus_id"], None, model[n].get("direction", 1), cal[n]["zero_raw"]) for n in JOINTS]
        assert [s.bus_id for s in self.specs] == [1, 2, 3, 4, 5, 6], "arm ids must be 1-6 (7-9 are the wheels)"
        self.limits = [list(map(float, cal[n]["q_lim"])) for n in JOINTS]
        rpc = 2 * math.pi / 4096
        self.rest_q = [s.direction * (cal[n]["rest_pose_raw"] - s.zero_raw) * rpc for s, n in zip(self.specs, JOINTS)]
        self.cal_path, self.speed, self.acc = cal_path, speed_rad_s, acc_rad_s2
        self.port = SerialPort(port)
        self.drv = FeetechStsDriver(self.port, self.specs, default_speed=int(round(speed_rad_s / rpc)), default_acc=max(1, int(round(acc_rad_s2 / rpc / 100))))
        self.powered, self.goal, self.clamps, self.hold, self.gripper_fault = False, None, [], False, None
        self.grip_limit = 400                                    # per mille of the STS3215's rated torque; see enable()

    def clamp(self, q):
        out = []
        for i, (v, (lo, hi)) in enumerate(zip(q, self.limits)):
            c = min(hi, max(lo, v))
            if abs(c - v) > 1e-6: self.clamps.append((JOINTS[i], v, c))
            out.append(c)
        return out

    def state(self):
        from seraphos.arm.pick import ArmState
        st = self.drv.read(poll=True)
        stale = [JOINTS[i] for i, s in enumerate(st) if not s.fresh]
        # the gripper's own fault (an overload from squeezing) is logged, not a stop: it must not keep the arm from going to rest
        if st[5].fault: self.gripper_fault = self.drv.decode_fault(st[5].fault)
        fault = ("no answer from " + ", ".join(stale)) if stale else next(("%s fault %s" % (JOINTS[i], self.drv.decode_fault(s.fault))
                                                                          for i, s in enumerate(st[:5]) if s.fault), None)
        return ArmState([s.position for s in st], [s.load or 0.0 for s in st], None, self.powered, fault,
                        any(abs(s.velocity) > 0.02 for s in st))

    def command(self, q):
        from seraphos.arm.drivers import JointCommand
        if not self.powered: raise RuntimeError("command with torque off")
        base = self.goal or self.state().q
        self.goal = self.clamp([b if v is None else float(v) for b, v in zip(base, q)])
        self.drv.send([JointCommand(position=v, velocity=self.speed) for v in self.goal])

    def wait(self, seconds): time.sleep(max(0.0, seconds))

    def enable(self, on):
        if on:
            from seraphos.arm.drivers.feetech_sts import protocol as P
            # the gripper's torque limit (RAM, 0-1000 per mille): squeezing a ball at full stall load cut the servo power on
            # 2026-10-29 (every servo silent, the arm dropped). Written before torque on, every time (a power cycle restores 1000).
            self.port.write(P.sync_write(48, {6: bytes([self.grip_limit & 0xFF, self.grip_limit >> 8])})); time.sleep(0.01)
            st = self.drv.enable(); self.powered = True; self.goal = [s.position for s in st]
            return "torque on (gripper torque limit %d per mille), holding %s" % (self.grip_limit, [round(math.degrees(v)) for v in self.goal])
        self.drv.disable(); self.powered = False; self.goal = None
        return "torque off"

    def close(self):
        try:
            if self.powered and not self.hold: self.drv.disable()    # a dry run never wrote to a servo; a HELD arm keeps its torque
        finally: self.port.close()


def r1_picker(Picker):
    """Stefan's Picker judging the grasp by the gripper's LOAD where it would use the current (judge_grasp: position margin OR
    current rise). This bus reads no current, and a soft object squeezes flat: on 2026-10-29 a held tissue stopped the gripper only
    1.8 deg above an empty close (his margin is 6 deg) while its load read 0.272 against 0.104-0.112 empty, and the image after the
    lift showed it in the jaws: judged a miss, released by the retry. The load threshold is cfg.hold_current_a (grasp-offset.json)."""
    class R1Picker(Picker):
        def _close_and_measure(self, label):
            self.c.command([None] * 5 + [self.cfg.gripper_closed_rad])
            s = self._gripper_stopped()
            ld = abs(s.load[5])
            # hold just past where it stopped, not at fully closed: squeezing a ball at full load tripped the gripper's overload (2026-10-29)
            self.c.command([None] * 5 + [max(self.cfg.gripper_closed_rad, s.q[5] - math.radians(3.0))])
            self.report.log(label, gripper=s.q[5], gripper_load=s.load[5], judged_by="load in place of current")
            return s.q[5], ld

        def _attempt(self, x, y):
            ok = Picker._attempt(self, x, y)
            if not ok:                                           # the lift re-check reads the load straight from the state: redo it
                s = self.c.state(); e = self._empty
                ok = (s.q[5] - e[0] > self.cfg.hold_margin_rad or abs(s.load[5]) - (e[1] or 0.0) > self.cfg.hold_current_a) and \
                     any(ev.get("what") == "grasp judged at the floor" and ev.get("holding") for ev in self.report.events[-3:])
                self.report.log("grasp re-checked by load after the lift", holding=ok, gripper_load=s.load[5], empty_load=e[1])
            return ok
    return R1Picker


class GemmaPerception:
    """seraphos.arm.pick_seq.Perception from Gemma boxes + Orbbec depth. `target` is the current estimate of the object in the arm base
    frame; a box is accepted only within `gate_m` of it (a box on the jaws or on another object is ignored)."""
    def __init__(self, cam, gem, arm, ext, what, target, tip_contact_z, gate_m=0.08, log=print, save_dir=None, grasp_offset=(0.0, 0.0, 0.0)):
        self.cam, self.gem, self.arm, self.ext, self.what, self.log = cam, gem, arm, ext, what, log
        self.grasp_offset = list(grasp_offset)               # the middle of the open jaws in the TCP frame (grasp-offset.json): the TCP is the fixed jaw tip
        self.target, self.tip_contact_z, self.gate, self.save_dir = list(target), tip_contact_z, gate_m, save_dir
        self.seen, self.n = [], 0                                # (time, q, box, xyz_base) of every accepted look

    def save(self, f, found, label=""):
        """The look's frame with every box, its base-frame point, and the TCP projected into the image (where the kinematics put
        the grasp centre): the offset between the two is what a miss is made of."""
        if not self.save_dir: return
        import cv2, numpy as np
        from seraphos.arm.handeye import base_to_camera
        from seraphos.arm.so101_kinematics import fk
        img, q = f.image.copy(), self.arm.state().q[:5]; tcp = fk(q)
        for b, p, _ in found:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            cv2.putText(img, "%.3f %.3f %.3f" % tuple(p), (int(b[0]), max(14, int(b[1]) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        fx, fy, cx, cy = f.meta["K"]
        for pt, name, col in ((tcp[:3, 3], "TCP (fixed jaw)", (0, 0, 255)), (self.grasp_centre(tcp), "grasp centre", (255, 0, 255))):
            c = base_to_camera(tcp, pt, self.ext)
            if c[2] > 0.01:
                u, v = int(round(fx * c[0] / c[2] + cx)), int(round(fy * c[1] / c[2] + cy))
                cv2.drawMarker(img, (u, v), col, cv2.MARKER_CROSS, 24, 2)
                cv2.putText(img, name, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.imwrite(os.path.join(self.save_dir, "look%02d%s.jpg" % (self.n, label)), img)
        np.save(os.path.join(self.save_dir, "look%02d%s_depth.npy" % (self.n, label)), f.depth.astype(np.float16))
        self.n += 1

    def look(self):
        """Fresh frame at the arm's present pose -> [(box, xyz_base)] for every Gemma box that depth can locate."""
        import numpy as np
        from core.embodiment import Refused
        from seraphos.arm.handeye import camera_to_base
        from seraphos.arm.so101_kinematics import fk
        q = self.arm.state().q[:5]; f = self.cam.frame(); tcp = fk(q)
        boxes, _ = self.gem.boxes(f.image, what=self.what)
        out = []
        for b in boxes:
            try: tg = self.cam.locate(f, b, self.what)
            except Refused as e: self.log("  box %s refused: %s" % ([round(v) for v in b], e)); continue
            p = camera_to_base(tcp, np.array(self.cam.to_depth_frame(tg.xyz)), self.ext)
            out.append((b, [float(v) for v in p], tg))
        self.save(f, out)
        return q, f, out

    def object_base(self):
        q, f, found = self.look()
        near = [(math.hypot(p[0] - self.target[0], p[1] - self.target[1]), b, p) for b, p, _ in found]
        near = [n for n in near if n[0] <= self.gate]
        self.log("  look: %d box(es), %s" % (len(found), ["(%.3f, %.3f, %.3f)" % tuple(p) for _, p, _ in found]))
        if not near: return None
        _, b, p = min(near)
        self.target = p[:2]; self.seen.append((time.time(), q, b, p)); return p

    # --- seraphos.arm.pick_seq.Perception
    def object_minus_tip(self):
        from seraphos.arm.pick_seq import Offset
        from seraphos.arm.so101_kinematics import fk
        p = self.object_base()
        if p is None: return None
        c = self.grasp_centre(fk(self.arm.state().q[:5]))
        return Offset(p[0] - float(c[0]), p[1] - float(c[1]))

    def grasp_centre(self, tcp):
        import numpy as np
        return (tcp @ np.append(np.asarray(self.grasp_offset, float), 1.0))[:3]

    def jaw_height_above_floor(self):
        from seraphos.arm.so101_kinematics import fk
        return float(fk(self.arm.state().q[:5])[2, 3]) - self.tip_contact_z

    def object_present(self):
        return self.object_base() is not None
