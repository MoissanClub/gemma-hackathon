"""AgileX arm (e.g. PiPER, 6-DoF + parallel gripper) on the shared Embodiment interface. SKELETON: a teammate fills this in.

Start from embodiments/mock/adapter.py (a complete, working adapter) and replace one fake at a time; run
`python3 run.py --embodiment agilex --fake-gemma` after each step, then without --fake-gemma against the llama-server.
What the core needs, in the order the loop calls it:

  frame()     a fresh colour frame of the table (wrist or fixed camera). Numpy BGR (OpenCV) or JPEG bytes.
  detect()    optional: return None and Gemma finds the objects (core.gemma.Gemma.boxes); a detector is faster and steadier.
  listen()    optional: return None and orders are typed (or plug a USB mic: 16 kHz mono WAV bytes; Gemma hears any language).
  locate()    box -> Target in the arm's base frame (metres, x ahead, y left, z up). Needs depth or a known table plane +
              camera calibration. Refuse in words when out of reach / at the image border. Keep every measured number in a
              calib file with provenance (see embodiments/g1/brain/calib.json); never hard-code a table distance in motion code.
  pick()      approach from above or the side, close, raise a few cm. The gripper width / effort tells holding().
  lift(), offer(), release(), home()   as in core/embodiment.py.

Lessons from the G1 that transfer (LEARNINGS.md): measure the setup at runtime (table plane and edge from depth); solve the whole
path before moving; cap every per-step joint change; stop on a tracking-error rise (contact); a stop holds, it never retreats on
its own; let vision feedback absorb calibration error instead of tuning deltas on the robot."""
from core.embodiment import Embodiment, Frame, Refused, SkillFailed, Target


class AgileX(Embodiment):
    name = "agilex"
    capabilities = dict(Embodiment.capabilities, depth=False, mic=False, speaker=False, detector=False, grip_sensor=True)

    def __init__(self, can_port="can0", camera=0, **options):
        self.can_port, self.camera, self.options = can_port, camera, options
        # e.g. from piper_sdk import C_PiperInterface; self.arm = C_PiperInterface(can_port); self.arm.ConnectPort()

    def start(self):
        raise NotImplementedError("enable the arm, open the gripper, move to home")

    def frame(self):
        raise NotImplementedError("grab a fresh camera frame -> Frame(image)")

    def locate(self, frame, box, desc=None):
        raise NotImplementedError("pixel box -> Target(xyz in the base frame); raise Refused('...') when unreachable")

    def pick(self, target):
        raise NotImplementedError("pre-grasp above/beside, approach, close; raise SkillFailed on an empty close")

    def holding(self):
        return None     # gripper width/effort -> True/False; None makes the loop ask Gemma "is it in the hand?"

    def release(self):
        raise NotImplementedError("open the gripper")

    def home(self):
        raise NotImplementedError("move to the safe home pose")

    def shutdown(self):
        pass            # home, disable


def make(**options): return AgileX(**options)
