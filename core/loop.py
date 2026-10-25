"""The shared task loop: hand the guest the object they ask for, in any language, on any Embodiment (core/embodiment.py).

IDLE > LOOK > LISTEN > UNDERSTAND > CONFIRM > LOCATE > PICK > VERIFY > LIFT > HANDOVER > HOME

The sequence and every spoken sentence are the G1 brain v2's (embodiments/g1/brain/brain.py look/listen/confirm/locate); the
physical stages are one call each into the embodiment. Gemma decides only what it perceives (descriptions, which one the guest
meant, is it in the hand, did the guest say take it); code sequences. Every event is logged with a provenance tag:
"Gemma decision", "embodiment skill" or "scripted"."""
import json, os, time
from . import image
from .embodiment import Refused, SkillFailed

ASK, LISTENING = "Tell me which one you'd like.", "Listening."


class Log:
    """runs/<session>/order_NN.jsonl (one event per line) and orders.jsonl (one summary per order)."""
    def __init__(self, root="runs", session=None):
        self.dir = os.path.join(root, session or time.strftime("%Y%m%dT%H%M%S")); os.makedirs(self.dir, exist_ok=True)
        self.f, self.n = None, 0

    def begin(self, n):
        self.n = n; self.f = open(os.path.join(self.dir, "order_%02d.jsonl" % n), "a")

    def event(self, **rec):
        rec.setdefault("t", time.time())
        if self.f: self.f.write(json.dumps(rec, default=str) + "\n"); self.f.flush()
        if rec.get("ev") == "gemma":
            print("  gemma %-11s %4d ms  %r -> %r" % (rec["kind"], rec["ms"], rec["question"][:48], rec["answer"][:40]), flush=True)

    def save(self, name, data):
        with open(os.path.join(self.dir, "o%02d_%s" % (self.n, name)), "wb") as f: f.write(data)

    def end(self, summary):
        with open(os.path.join(self.dir, "orders.jsonl"), "a") as f: f.write(json.dumps(summary, default=str) + "\n")
        if self.f: self.f.close(); self.f = None


def typed_input(prompt):
    try: return input(prompt)
    except EOFError: return ""          # stdin closed (piped orders): no order / release now


def iou(a, b):
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


class Loop:
    def __init__(self, robot, gem, log, typed=None, retries=3, take_wait=20.0):
        self.r, self.g, self.log, self.retries, self.take_wait = robot, gem, log, retries, take_wait
        self.caps = dict(robot.capabilities)
        self.typed = typed if typed is not None else typed_input
        self.state, self.o = "IDLE", {}

    def stage(self, s, note=""):
        self.state = s; self.o.setdefault("stages", []).append([s, round(time.time() - self.t0, 2)])
        self.log.event(ev="stage", stage=s, note=note); print("[%s] %s" % (s, note), flush=True)

    def say(self, text):
        self.log.event(ev="say", stage=self.state, text=text, tag="scripted"); self.r.say(text)

    # --- perception ---------------------------------------------------------------------------------------------------------
    def boxes(self, frame):
        b = self.r.detect(frame)
        if b is not None: return [tuple(map(float, x)) for x in b], "embodiment detector"
        return self.g.boxes(frame.image, stage=self.state)[0], "Gemma"

    def look(self):
        """LOOK: fresh frame, boxes, one short Gemma description per crop, plus words for position (and size, if the robot adds it)."""
        self.stage("LOOK"); sc = self.r.survey(); self.surveyed = None
        if sc is not None:
            frame, named = sc; boxes, src = [n[0] for n in named], "embodiment survey"
            descs = [self.g.describe_named(nm, c, stage="LOOK", media="object %d" % i) for i, (_, nm, c) in enumerate(named)]
            self.surveyed = (frame, boxes)
        else:
            frame = self.r.frame(); boxes, src = self.boxes(frame)
            descs = [self.g.describe_crop(image.crop(frame.image, b), stage="LOOK", media="object %d" % i)[0] for i, b in enumerate(boxes)]
        pos = self.r.position_words(boxes, image.size(frame.image)[0])
        scene = [{"i": i, "box": [round(v) for v in b], "desc": d, "words": ", ".join(w for w in (d, p) if w)} for i, (b, d, p) in enumerate(zip(boxes, descs, pos))]
        self.log.event(ev="scene", stage="LOOK", boxes_by=src, objects=scene, tag="Gemma decision")
        self.log.save("look.jpg", image.to_jpeg(frame.image, None)); self.o["scene"] = [s["words"] for s in scene]
        print("  scene: " + ("; ".join("%d %s" % (s["i"] + 1, s["words"]) for s in scene) or "nothing"), flush=True)
        return scene

    def listen(self, scene):
        """LISTEN + UNDERSTAND (brain v2's rules): `none` once -> the honest negative and another listen; twice -> the ambiguity question."""
        words, short, nones = [s["words"] for s in scene], [s["desc"] for s in scene], 0
        for attempt in range(self.retries):
            self.stage("LISTEN", "attempt %d" % (attempt + 1))
            if attempt == 0: self.say(ASK)
            elif nones >= 2 and len(short) >= 2: self.say("The %s one, or the %s one?" % (short[0], short[1]))
            elif nones == 1: self.say("I do not see that. I see %s." % " and ".join(short))
            else: self.say("Please say it again.")
            wav = None
            if self.caps.get("mic"):
                self.say(LISTENING); wav = self.r.listen(self.r.listen_seconds)
                if wav: self.log.save("order_%d.wav" % (attempt + 1), wav); self.log.event(ev="record", stage="LISTEN", bytes=len(wav), tag="embodiment skill")
            self.stage("UNDERSTAND")
            if wav:
                said = self.g.transcribe(wav)[0].strip()
                if not said: self.o.setdefault("heard", []).append(""); continue      # no speech: never choose from silence
                idx, ans, _ = self.g.hear_choice(wav, words)
                if self.g.choose_text(said, words)[0] != idx:
                    # the clip and its transcript must pick the same one (AgileX 2026-10-24, 9 clips: the 5 that agreed were all
                    # right; 'hapsy' -> apple and silence -> watermelon disagreed): ask again rather than guess
                    self.o.setdefault("heard", []).append("%s / %r" % (ans, said)); continue
            else:
                text = self.typed("  order> ").strip()
                if not text: continue
                idx, ans, _ = self.g.choose_text(text, words)
            self.o.setdefault("heard", []).append(ans)
            if idx is not None: return idx
            if "none" in ans or ans.strip() == "0": nones += 1
        return None

    def relocate(self, want):
        """LOCATE: a fresh frame, the chosen object matched to the new boxes (IoU, else by description), then the embodiment's geometry.
        After an embodiment survey: its own frame and box, not a second detection (2026-10-24, AgileX: a fresh Gemma boxing missed
        the Pepsi the survey had found, and the top-down crops no longer said 'pepsi')."""
        self.stage("LOCATE", want["desc"])
        if self.surveyed:
            frame, boxes = self.surveyed; tg = self.r.locate(frame, boxes[want["i"]], want["desc"])
            self.log.event(ev="target", stage="LOCATE", desc=want["desc"], xyz=tg.xyz, box=want["box"], tag="embodiment skill")
            print("  target %r" % (tg,), flush=True); return tg
        frame = self.r.frame(); boxes, _ = self.boxes(frame)
        if not boxes: raise Refused("I lost sight of it.")
        best = max(boxes, key=lambda b: iou(b, want["box"]))
        if iou(best, want["box"]) < 0.2:
            descs = [self.g.describe_crop(image.crop(frame.image, b), stage="LOCATE")[0] for b in boxes]
            same = [b for b, d in zip(boxes, descs) if d == want["desc"] or (set(d.split()) & set(want["desc"].split())) - {"can", "soda"}]
            if not same: raise Refused("I cannot find the %s one any more." % want["desc"])
            best = same[0]
        tg = self.r.locate(frame, best, want["desc"])
        self.log.event(ev="target", stage="LOCATE", desc=want["desc"], xyz=tg.xyz, box=[round(v) for v in best], tag="embodiment skill")
        print("  target %r" % (tg,), flush=True); return tg

    def in_hand(self):
        """VERIFY: the gripper's own sensing if it has any, else Gemma on a frame that shows the hand."""
        h = self.r.holding()
        if h is not None: self.log.event(ev="verify", stage="VERIFY", held=h, by="grip sensor", tag="embodiment skill"); return h
        img = (self.r.third_frame() or self.r.frame()).image
        h, ans, _ = self.g.yesno(img, self.g.p["in_hand_q"], stage="VERIFY")
        self.log.event(ev="verify", stage="VERIFY", held=h, by="Gemma", answer=ans, tag="Gemma decision"); return bool(h)

    def wait_take(self):
        """HANDOVER: release on a let-go phrase (Gemma transcript), else on Enter (typed), else after take_wait seconds."""
        t_end = time.time() + self.take_wait
        while time.time() < t_end:
            if self.caps.get("mic"):
                wav = self.r.listen(2.0)
                if wav:
                    ok, ans, _ = self.g.heard_take_it(wav)
                    if ok: return "heard: " + ans
                else: time.sleep(0.5)
            else:
                self.typed("  press Enter when the guest holds it> "); return "key"
        return "timeout"

    # --- one order ----------------------------------------------------------------------------------------------------------
    def order(self, n):
        self.t0, self.o = time.time(), {"n": n, "embodiment": self.r.name, "extra": self.r.extra()}; self.log.begin(n); result = "?"
        try:
            scene = self.look()
            if not scene: self.say("I do not see any %ss." % self.g.p["object"]); result = "nothing_seen"; return result
            idx = self.listen(scene)
            if idx is None: self.say("Sorry, I did not get that."); result = "not_understood"; return result
            want = scene[idx]; self.o["choice"] = want["words"]
            self.stage("CONFIRM", want["desc"]); self.say("The %s one. Coming up." % want["desc"])
            tg = self.relocate(want)
            self.stage("PICK"); self.r.pick(tg)
            self.stage("VERIFY")
            if not self.in_hand(): self.say("I missed it."); self.r.home(); result = "miss"; return result
            self.stage("LIFT"); self.r.lift()
            self.stage("HANDOVER"); self.r.offer(); self.say("Here you go. Say take it when you have it.")
            how = self.wait_take(); self.log.event(ev="release", stage="HANDOVER", how=how, tag="scripted")
            self.r.release(); self.say("Enjoy.")
            self.stage("HOME"); self.r.home(); result = "delivered"; return result
        except Refused as e:
            self.say(str(e)); result = "refused"; self.o["reason"] = str(e); return result
        except SkillFailed as e:
            self.say("I could not do that. " + str(e)); result = "skill_failed"; self.o["reason"] = str(e)
            try: self.r.home()
            except Refused as e2: self.say(str(e2)); result = "skill_failed_holding"
            return result
        finally:
            self.o.update(result=result, seconds=round(time.time() - self.t0, 2), gemma_calls=self.g.n)
            self.log.event(ev="result", **self.o); self.log.end(self.o)
            print("order %d: %s in %.1f s" % (n, result, self.o["seconds"]), flush=True); self.state = "IDLE"
