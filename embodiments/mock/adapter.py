"""Mock embodiment: no hardware, stdlib only. The reference for writing a new adapter and the way to test the core on a laptop.

Senses are real recordings from the G1 office day (2026-10-19): `media/scene_two_cans.jpg` (the D435i head frame: a purple La Croix
on the left, a Chi sparkling lemon can in the middle) and two spoken orders from the C922 mic (`voice_order.wav` asks for the purple
one in English, `voice_order_lang2.wav` for the white/silver one in another language). Geometry and motion are faked.
Mock latencies are not benchmarks: the same image repeats, and llama.cpp reuses its KV (CLAUDE.md)."""
import os, subprocess, tempfile, time
from core.embodiment import Embodiment, Frame, Refused, SkillFailed, Target

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "media")


class Mock(Embodiment):
    name = "mock"
    capabilities = dict(Embodiment.capabilities, mic=True, speaker=False, depth=False, detector=False, grip_sensor=True)

    def __init__(self, scene="scene_two_cans.jpg", clips=("voice_order.wav", "voice_order_lang2.wav"), typed=False, miss=False, delay=0.2):
        self.scene, self.clips, self.miss, self.delay, self.i, self.held = os.path.join(MEDIA, scene), list(clips), miss, delay, 0, False
        if typed: self.capabilities = dict(self.capabilities, mic=False)

    def frame(self):
        with open(self.scene, "rb") as f: return Frame(f.read(), t=time.time())

    def listen(self, seconds):
        if seconds <= 2.5: return self.take_it()                          # the handover clip: the guest says "take it"
        path = os.path.join(MEDIA, self.clips[self.i % len(self.clips)]); self.i += 1
        print("  [mic] playing %s" % os.path.basename(path), flush=True)
        with open(path, "rb") as f: return f.read()

    def take_it(self):
        """A synthetic "Okay, I have it, thanks." (macOS `say`), cached in the system temp folder; None elsewhere (the loop times out)."""
        wav = os.path.join(tempfile.gettempdir(), "gemma-hackathon-take-it.wav")
        if not os.path.exists(wav):
            aiff = wav[:-4] + ".aiff"
            try:
                subprocess.run(["say", "-o", aiff, "Okay, I have it, thanks."], check=True, capture_output=True)
                subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, wav], check=True, capture_output=True)
            except (FileNotFoundError, subprocess.CalledProcessError): return None
        with open(wav, "rb") as f: return f.read()

    def locate(self, frame, box, desc=None):
        from core.image import size
        W, H = size(frame.image); x1, y1, x2, y2 = box
        if min(x1, y1, W - 1 - x2, H - 1 - y2) <= 2:                       # the G1's rule: a box cut by the border has the wrong size and axis
            raise Refused("The %s one is at the edge of my view. Please move it a little toward my centre." % (desc or "that"))
        cx = (x1 + x2) / 2 / W
        return Target([0.40, round((0.5 - cx) * 0.5, 3), 0.05], box=box, desc=desc, data={"faked": True})   # x ahead, y left, z up

    def pick(self, target):
        print("  [mock] reach %s, close" % (target.xyz,), flush=True); time.sleep(self.delay)
        if self.miss: raise SkillFailed("The gripper closed on nothing.")
        self.held = True

    def holding(self): return self.held
    def lift(self): print("  [mock] lift", flush=True); time.sleep(self.delay)
    def offer(self): print("  [mock] offer", flush=True); time.sleep(self.delay)
    def release(self): print("  [mock] open", flush=True); self.held = False
    def home(self): print("  [mock] home", flush=True); time.sleep(self.delay)
    def place(self, spot): print("  [mock] place on the %s" % spot, flush=True); self.held = False
    def extra(self): return {"scene": os.path.basename(self.scene)}


def make(**options): return Mock(**options)
