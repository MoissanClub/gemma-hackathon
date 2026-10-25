"""The small robot API from bench/codegen_probe.py, implemented on any Embodiment, so one task program runs on every robot.

On the Orin (docs/2026-10-22-orin-model-sizes.md) E4B wrote correct `def task(robot):` code for 8 of 9 unseen one-step tasks against
this API (E2B 2 of 9), so the same API is the surface for hand-written tasks today and model-written ones next. Keep API (below)
identical to codegen_probe.py's text when changing either. Coordinates as in that API: x centimetres to the robot's LEFT, y ahead."""
from . import image
from .embodiment import Refused, SkillFailed
from .gemma import audio_part, colour_in

API = """The robot API (all you may use, besides Python built-ins):
- robot.look() -> list of dicts, one per can still on the table: {"id": int, "colour": str, "label": str, "x": float, "y": float}
  x is centimetres to the robot's left (negative = right of the robot), y is centimetres ahead.
- robot.pick(can_id) -> bool. Picks that can up. The robot holds at most one can; pick fails while holding one.
- robot.give() -> bool. Hands the held can to the guest.
- robot.place(spot) -> bool. Puts the held can down at spot, "tray" or "table".
- robot.say(text) -> None. Speaks to the guest.
- robot.listen() -> str. Waits for the guest to speak and returns what they said, in English.
- robot.wave() -> None. Waves at the guest."""


class RobotAPI:
    def __init__(self, embodiment, gem, log=None):
        self.e, self.g, self.log, self.seen, self.held = embodiment, gem, log or (lambda **r: None), {}, False

    def look(self):
        sc = self.e.survey()
        if sc is not None: frame, named = sc[0], [(b, self.g.describe_named(nm, c)) for b, nm, c in sc[1]]
        else:
            frame = self.e.frame(); boxes = self.e.detect(frame)
            if boxes is None: boxes = self.g.boxes(frame.image)[0]
            named = [(b, self.g.describe_crop(image.crop(frame.image, b))[0]) for b in boxes]
        out, self.seen = [], {}
        for i, (b, desc) in enumerate(named):
            try: tg = self.e.locate(frame, b, desc)
            except Refused: tg = None
            self.seen[i] = (frame, b, desc, tg)
            x, y = (None, None) if tg is None or tg.xyz is None else (round(tg.xyz[1] * 100, 1), round(tg.xyz[0] * 100, 1))
            out.append({"id": i, "colour": colour_in(desc) or "", "label": desc, "x": x, "y": y})
        self.log(ev="api_look", cans=out); return out

    def pick(self, can_id):
        if self.held or can_id not in self.seen or self.seen[can_id][3] is None: return False
        try: self.e.pick(self.seen[can_id][3])
        except (Refused, SkillFailed) as e: self.e.say(str(e)); return False
        self.held = self.e.holding() is not False; return self.held

    def give(self):
        if not self.held: return False
        self.e.lift(); self.e.offer(); self.e.release(); self.e.home(); self.held = False; return True

    def place(self, spot):
        if not self.held: return False
        try: self.e.place(spot)
        except Refused as e: self.e.say(str(e)); return False
        self.held = False; return True

    def say(self, text): self.e.say(str(text))

    def listen(self):
        wav = self.e.listen(self.e.listen_seconds)
        if not wav: return input("  guest> ")
        msgs = [{"role": "user", "content": [{"type": "text", "text": self.g.p["take_q"]}, audio_part(wav)]}]
        return self.g.ask("transcribe", self.g.p["take_q"], msgs, 32, "LISTEN", media="audio")[0]

    def wave(self):
        getattr(self.e, "wave", lambda: self.e.say("Hello!"))()
