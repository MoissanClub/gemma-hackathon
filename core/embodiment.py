"""The one interface a robot implements to run the shared harness (core/loop.py, core/robot_api.py).

The split, learned on the G1 (LEARNINGS.md): Gemma answers direct perception questions (what is this, which one did the guest
mean, is it in the hand); the embodiment owns everything physical (frames, depth, geometry, motion, safety, its own calibration).
The core never sees a joint, a frame transform or a gripper value, so a new robot needs only this file's methods.

Coordinates: `Target.xyz` is in the embodiment's own base frame, metres, x ahead, y left, z up (the G1's pelvis frame). The core
only passes it back to the same embodiment. Image left/right must be robot left/right (head camera facing forward), or override
`position_words`."""


class Refused(Exception):
    """The embodiment will not do it, and says why in words the guest can act on ("Please move it closer to me.")."""


class SkillFailed(Exception):
    """A skill ran and did not achieve its goal (a missed grasp, a lost object). The embodiment is already in a safe state."""


class Frame:
    """One camera frame. `image` is what Gemma sees: a numpy BGR array or JPEG bytes (core/image.py). `depth`, `meta` and anything
    else belong to the embodiment (intrinsics, the camera-to-base transform, joint state at capture) and are only handed back to it."""
    def __init__(self, image, depth=None, meta=None, t=None):
        self.image, self.depth, self.meta, self.t = image, depth, meta or {}, t


class Target:
    """A located object: base-frame point, the box it came from, and the embodiment's own data (diameter, table plane, plan)."""
    def __init__(self, xyz, box=None, desc=None, data=None):
        self.xyz, self.box, self.desc, self.data = xyz, box, desc, data or {}

    def __repr__(self): return "Target(%s, %r)" % (None if self.xyz is None else [round(v, 3) for v in self.xyz], self.desc)


class Embodiment:
    """Subclass and fill in. Required: frame, locate, pick, release, home. Everything else has a usable default."""
    name = "abstract"
    # What the loop may count on. A missing sense changes the loop, never the prompts:
    # mic=False -> typed orders; detector=False -> Gemma boxes; grip_sensor=False -> Gemma checks the hand in a frame.
    capabilities = {"camera": True, "depth": False, "mic": False, "speaker": False, "detector": False, "grip_sensor": False,
                    "third_camera": False, "arms": 1, "mobile": False}
    prompts = {}            # overrides of core.gemma.PROMPTS (a different object or role); each override is an unmeasured variant
    listen_seconds = 4.0    # the G1's order clip

    # --- lifecycle --------------------------------------------------------------------------------------------------------
    def start(self):
        """Connect, enable, move to a known safe pose. Called once."""

    def shutdown(self):
        """Leave the robot safe (arm home and released, motors off). Called once, also after an exception."""

    def stop(self):
        """Freeze motion now (operator stop). May be called from another thread."""

    # --- senses -----------------------------------------------------------------------------------------------------------
    def frame(self):
        """A fresh Frame from the camera that sees the workspace. Never a cached frame: llama.cpp reuses the KV of a repeated image."""
        raise NotImplementedError

    def detect(self, frame):
        """Object boxes [(x1, y1, x2, y2) pixels] in the frame, or None to let Gemma find them (core.gemma.Gemma.boxes)."""
        return None

    def listen(self, seconds):
        """A mono 16 kHz WAV clip as bytes, or None without a microphone (the loop then reads a typed order)."""
        return None

    def say(self, text):
        """Speak (TTS) or print. Should return once the sentence is out, so the robot does not hear itself."""
        print("  [say] " + text, flush=True)

    def third_frame(self):
        """A frame that shows the hand from outside (the G1's C922), or None: the in-hand check then uses frame()."""
        return None

    # --- geometry ---------------------------------------------------------------------------------------------------------
    def locate(self, frame, box, desc=None):
        """The object in `box` as a Target in the base frame, ready for pick(). Raise Refused with a sentence when it is out of
        reach, cut by the image border, too close to another object, or depth is missing. All geometry and safety live here."""
        raise NotImplementedError

    def position_words(self, boxes, width):
        """Words that tell the objects apart by position, robot's view. The G1 adds size ("slim") from depth."""
        n = len(boxes)
        if n == 1: return [""]
        if n <= 3:
            order = sorted(range(n), key=lambda i: (boxes[i][0] + boxes[i][2]) / 2)      # left to right in the image
            names = ["on the left", "on the right"] if n == 2 else ["on the left", "in the middle", "on the right"]
            out = [""] * n
            for rank, i in enumerate(order): out[i] = names[rank]
            return out
        return ["number %d" % (i + 1) for i in range(n)]

    # --- skills -----------------------------------------------------------------------------------------------------------
    def pick(self, target):
        """Grasp the target and leave it held just above where it was. Raise Refused before moving, SkillFailed after."""
        raise NotImplementedError

    def holding(self):
        """True / False from the gripper's own sensing (finger stall, gripper width, current), or None when unknown."""
        return None

    def lift(self):
        """Raise the held object clear of the table."""

    def offer(self):
        """Hold the object out to the guest, inside the camera's view if possible."""

    def release(self):
        """Open the gripper (the guest has the object)."""
        raise NotImplementedError

    def home(self):
        """Back to the safe pose by a checked path. Raise Refused if no safe path exists; the operator takes over."""
        raise NotImplementedError

    def place(self, spot):
        """Put the held object down at a named spot ("table", "tray"). Optional."""
        raise Refused("I cannot put things down yet.")

    def extra(self):
        """Embodiment facts for the log (calibration sha, firmware, git sha)."""
        return {}
