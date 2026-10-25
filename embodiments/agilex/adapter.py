"""AgileX PiPER (6-DoF) with a Pika gripper and the Pika's wrist D405, on the shared Embodiment interface.

A thin wrapper over the robot-proven see-and-grab code in dual-piper-teleop (`src/dualpiper/grab/`, docs/grab.md there): the arm,
gripper and camera backend (`dualpiper.llm.real.PiperRobot`), the depth geometry (`Grab.look`: box -> can axis, top and diameter
from the D405, in the arm's base frame) and the grasp (`Grab._pick`: pre-grasp, measured correction from close up, straight
approach with a contact stop, close, jaws judged against the measured diameter, 6 cm lift). Every number (view pose, workspace,
camera calibration, gripper widths, grasp offsets) stays in that repo's configs with its provenance; none is set here.

Run with dual-piper-teleop's venv (numpy, OpenCV, piper_sdk, pyrealsense2) and its Gemma server (port 8090; 8080 is its Quest
console there):

  DPT=~/work/MoissanClub/dual-piper-teleop
  $DPT/.venv/bin/python -B run.py -e agilex --server http://127.0.0.1:8090/v1                  # dry: look, listen, choose, locate
  $DPT/.venv/bin/python -B run.py -e agilex --server http://127.0.0.1:8090/v1 --opt motion=true  # the arm moves (owner + E-stop)

Options (--opt key=value): motion (false), yes (false: Enter before each motion stage), repo (dual-piper-teleop's path),
config (configs/grab.yaml there), mic (ALSA device; default: the system's).

Without motion the arm stays where it is (usually folded at rest, a poor view) and pick() refuses before moving.
Coordinates: Target.xyz is the arm's base frame, metres, x ahead, y left, z up. Image left/right as robot left/right has not been
checked for this wrist camera; position words may be mirrored."""
import os, subprocess, sys, tempfile, time
from core.embodiment import Embodiment, Frame, Refused, SkillFailed, Target

REPO = os.path.expanduser(os.environ.get("DUALPIPER_REPO", "~/work/MoissanClub/dual-piper-teleop"))
REFUSALS = {"too wide": "The %s one is too wide for my gripper.", "unreachable": "The %s one is out of my reach.",
           "blocked": "The %s one is too close to its neighbours for my fingers. Please give it a few centimetres on each side."}


class AgileX(Embodiment):
    name = "agilex"
    capabilities = dict(Embodiment.capabilities, depth=True, mic=True, speaker=True, detector=False, grip_sensor=True)
    listen_seconds = 5.0

    def __init__(self, motion=False, yes=False, repo=REPO, config="configs/grab.yaml", mic=None, **options):
        self.motion, self.yes, self.repo, self.config, self.mic = motion, yes, os.path.expanduser(repo), config, mic
        self.r = self.grab = self.held = None
        self.clean = True               # False after a stop or a failed skill: shutdown then holds instead of parking

    # --- lifecycle --------------------------------------------------------------------------------------------------------
    def start(self):
        sys.path.insert(0, os.path.join(self.repo, "src"))
        from dualpiper.grab.__main__ import load_config
        from dualpiper.grab.gemma import Gemma as DptGemma
        from dualpiper.grab.skill import Grab
        from dualpiper.llm.real import PiperRobot
        self.cfg = load_config(self.config)
        self.r = PiperRobot(self.cfg["llm_config"], motion=self.motion)
        confirm = (lambda p: True) if self.yes else (lambda p: input("  %s [Enter = yes, n = stop] " % p).strip().lower() != "n")
        # Grab's own Gemma serves only its close looks (boxes when a close-up has no round lid); the core's Gemma does the rest
        self.grab = Grab(self.r, DptGemma(self.cfg["gemma_url"]), self.cfg, say=self.say, confirm=confirm,
                         out_dir=os.path.join(self.repo, "sessions", "grab"))
        self.grab.trial = {"id": "agilex-" + time.strftime("%Y%m%d-%H%M%S"), "params": dict(self.cfg)}
        if self.r.T_cam is None:
            raise RuntimeError("camera not calibrated: python -m dualpiper.llm.calibrate in %s" % self.repo)
        if self.motion:
            self.say("Moving to my view pose.")
            self._check(self.r.go_view(), "the view pose")
            self.r.grip.set(self.r.grip.open_mm)

    def shutdown(self):
        if self.r is None: return
        try:
            if self.motion and self.clean and self.held is None:
                self.say("Folding my arm."); print("  " + self.r.park(), flush=True)
            elif self.motion:
                print("  holding where it stopped (a stop or a failed skill): the operator decides the way home", flush=True)
        finally:
            self.r.close()

    def stop(self):
        self.clean = False

    # --- senses -----------------------------------------------------------------------------------------------------------
    def frame(self):
        colour, depth = self.r.cam.capture()
        return Frame(colour, depth=depth, meta={"q": self.r.q_meas().tolist()}, t=time.time())

    def listen(self, seconds):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            cmd = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(max(1, int(round(seconds))))]
            subprocess.run(cmd + (["-D", self.mic] if self.mic else []) + [f.name], check=True)
            with open(f.name, "rb") as g: return g.read()

    def say(self, text):
        print("  [say] " + text, flush=True)
        try: subprocess.run(["spd-say", "-w", text], check=False)      # -w: returns once spoken, so the mic does not hear it
        except FileNotFoundError: pass

    # --- geometry ---------------------------------------------------------------------------------------------------------
    def locate(self, frame, box, desc=None):
        """The can in `box`: Grab.look on a fresh frame (its own boxes, with depth and the arm pose at capture), the box that
        overlaps the core's most, then the grasp checked for size, reach and crowding before anything moves."""
        from dualpiper.grab.scene import grasp_points
        from core.loop import iou
        what = desc or "that"
        dets = [d for d in self.grab.look("locate", read=False) if d["can"] is not None]
        if not dets: raise Refused("I cannot measure the %s one. Please move it a little toward the middle of the table." % what)
        best = max(dets, key=lambda d: iou(d["box"], box))
        if iou(best["box"], box) < 0.3:
            raise Refused("I lost the %s one when I measured it. Please ask again." % what)
        can, c = best["can"], self.cfg
        style = self.grab.style(can)
        if style == "too wide": raise Refused(REFUSALS["too wide"] % what)
        g, pre = grasp_points(can, c["below_top_cm"] / 100, c["pre_cm"] / 100, self.r.ws["z"][0] / 100.0)
        if not (self.grab.reachable(g) and self.grab.reachable(pre)): raise Refused(REFUSALS["unreachable"] % what)
        if style == "enclose" and self.grab.blocked(can): raise Refused(REFUSALS["blocked"] % what)
        return Target([float(v) for v in can.axis] + [float(can.top_z)], box=box, desc=desc,
                      data={"can": can, "style": style, "diameter_mm": round(can.diameter * 1000)})

    # --- skills -----------------------------------------------------------------------------------------------------------
    def pick(self, target):
        """Grab._pick: approach, correct from close up, grasp, judge the jaws, lift. Refused before moving, SkillFailed after."""
        if not self.motion: raise Refused("My motion is switched off, so I will not pick it up.")
        from dualpiper.grab.skill import RESULT_SAY, Stop
        from dualpiper.llm.real import Abort
        can, rec = target.data["can"], {}
        self.grab.r.grip.set(self.grab.r.grip.open_mm)
        try: result = self.grab._pick(can, target.desc or self.cfg["label"], rec)
        except (Stop, Abort) as e:
            self.clean = False; raise SkillFailed("I stopped: %s." % e)
        self.grab.trial.setdefault("picks", []).append(dict(rec, result=result))
        if result in REFUSALS: raise Refused(REFUSALS[result] % (target.desc or "that"))
        if result != "grasped":
            self.grab._recover()                                         # open, back out, view pose
            raise SkillFailed(RESULT_SAY.get(result, result))
        self.held = (can, rec["style"])

    def holding(self):
        if self.held is None: return False
        can, style = self.held
        return self.grab.held_now(self.r.grip.width_mm(), can, style)

    def lift(self):
        pass                            # Grab._pick already lifted it lift_cm and re-checked the jaws

    def offer(self):
        pass                            # held where it was lifted, in the camera's view; no hand-over pose measured yet

    def release(self):
        self.r.grip.set(self.r.grip.open_mm); self.held = None

    def home(self):
        """Back out along the gripper axis and return to the view pose (ready for the next look)."""
        if not self.motion: return
        if self.held is not None: raise Refused("I am still holding it, so I will stay here. Please take it from my gripper.")
        self.say("Going back to my view pose.")
        self.grab._recover()
        err = self.grab.trial.pop("recover_error", None)
        if err: self.clean = False; raise Refused("I could not get back to my view pose: %s" % err)

    def extra(self):
        try: sha = subprocess.run(["git", "-C", self.repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        except FileNotFoundError: sha = None
        return {"dualpiper_sha": sha, "motion": self.motion, "config": self.config}

    def _check(self, msg, label):
        print("  " + msg, flush=True)
        if msg.startswith(("Refused", "ABORTED")):
            self.clean = False; raise Refused("I could not reach %s." % label)


def make(**options): return AgileX(**options)
