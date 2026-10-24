"""Unitree G1 (left BrainCo hand, D435i head camera, C922 mic/third camera, Jetson Orin NX PC2) on the shared Embodiment interface.

STATUS: not wired. The robot-proven G1 program today is `brain/brain.py` (brain v2: 3 of 3 autonomous grasps on 2026-10-23,
`notes/2026-10-23-v2-grasp.md`). Its LOOK / LISTEN / UNDERSTAND / CONFIRM / LOCATE match core/loop.py line for line (same Gemma
calls, same sentences); what differs is everything physical, which is exactly what this adapter wraps. Each method below names the
brain v2 code that already implements it, so the port is moving code, not writing it.

Port plan (test it on the Mac first: brain.py --mock --mock-gemma installs fake g1_fetch modules, see brain/mock_robot.py):
  1. make(): sys.argv = ["brain.py"] + brain_args; import brain (it parses args and imports g1_fetch at import); b = brain.Brain().
  2. Fill the methods from the table; keep brain.Stopped / the hold-and-`y` way home as the G1's home() semantics.
  3. `python -B brain/check.py` must stay at its current result (80 of 84 on 2026-10-24, the 4 known mock-hand failures).
Never edit code on PC2; follow notes/ROBOT_RULES.md and docs/DEV_WORKFLOW.md."""
from core.embodiment import Embodiment

NOT_WIRED = "G1 adapter not wired yet: run embodiments/g1/brain/brain.py (see embodiments/g1/README.md)"


class G1(Embodiment):
    name = "g1"
    capabilities = dict(Embodiment.capabilities, depth=True, mic=True, speaker=True, detector=True, grip_sensor=True, third_camera=True)
    listen_seconds = 4.0

    def __init__(self, brain):
        self.b = brain

    def start(self):         raise NotImplementedError(NOT_WIRED)   # Brain.__init__ + arms_enable(); preflight.py first
    def shutdown(self):      raise NotImplementedError(NOT_WIRED)   # Brain.shutdown(): home, release the arm, arm service back on
    def stop(self):          raise NotImplementedError(NOT_WIRED)   # Brain.stop_key("a"): freeze at the next 20 ms target, hold
    def frame(self):         raise NotImplementedError(NOT_WIRED)   # Brain.head_frame() -> Frame(f.color, depth=f) (D435i, pelvis frame in meta)
    def detect(self, frame): raise NotImplementedError(NOT_WIRED)   # Brain.detect(frame.depth): YOLOE boxes
    def listen(self, s):     raise NotImplementedError(NOT_WIRED)   # Brain.mic.record(s, path): C922 arecord, 16 kHz mono
    def say(self, text):     raise NotImplementedError(NOT_WIRED)   # Brain.say(text, wait): Unitree TtsMaker, chest speaker
    def third_frame(self):   raise NotImplementedError(NOT_WIRED)   # Brain.cam3.shot(): the C922 view of the hand

    def locate(self, frame, box, desc=None):
        # Brain.locate() after its match: can_point (axis = front depth + classified radius), border rule, ahead_clear (0.30 m rule),
        # table() (level plane from depth), plan() (geom.plan_reach feasibility search); refusal sentences -> Refused
        raise NotImplementedError(NOT_WIRED)

    def position_words(self, boxes, width):
        return Embodiment.position_words(self, boxes, width)        # brain v2 also adds the size class ("slim") from depth

    def pick(self, target):  raise NotImplementedError(NOT_WIRED)   # Brain.pick(tg): pre-grasp, servo, approach, close, stall-pattern fix
    def holding(self):       raise NotImplementedError(NOT_WIRED)   # Brain.verify(): hold signature from the finger stall
    def lift(self):          raise NotImplementedError(NOT_WIRED)   # Brain.lift(tg): 12 cm with correction
    def offer(self):         raise NotImplementedError(NOT_WIRED)   # Brain.offer_pose() + the offer move in Brain.handover()
    def release(self):       raise NotImplementedError(NOT_WIRED)   # Brain.handover()'s release: staged hand open
    def home(self):          raise NotImplementedError(NOT_WIRED)   # Brain.retreat(): checked way home, run only on `y`; no path -> Refused
    def extra(self):         raise NotImplementedError(NOT_WIRED)   # calib sha16, git sha, deviations (calib.py)


def make(**options):
    raise NotImplementedError(NOT_WIRED)
