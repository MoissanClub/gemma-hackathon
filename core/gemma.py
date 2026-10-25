"""Gemma 4 E2B calls shared by every embodiment: llama-server at 127.0.0.1:8080 (serve/), thinking off, greedy, one cached prefix
per call type. The payloads are the ones verified on the G1 (embodiments/g1/brain/gemma.py, Oct 18-23); what a robot sees through
them does not depend on the robot. Every call returns a record (question, answer, ms, provenance tag, server timings) for the log.

Rules learned the hard way (LEARNINGS.md): one direct perception question per request, AFTER the image; few output tokens; the
answer carries the percept and code does geometry and sequencing. E2B cannot compose a plan from a percept without thinking."""
import base64, re, time
from . import image
from .client import timed_chat

TAG = "Gemma decision"
BASE = {"model": "gemma4-e2b", "cache_prompt": True, "temperature": 0.0, "top_k": 1, "chat_template_kwargs": {"enable_thinking": False}}
COLORS = ["red", "green", "blue", "white", "purple", "black", "silver", "yellow", "orange", "pink", "grey", "gray"]
# words that mean "let go" in a transcript; the yes/no judgement was 9/14 on synthetic clips, transcribe + keywords 14/14 (Mac, 2026-10-19)
TAKE_WORDS = ("take", "took", "got it", "have it", "let go", "thank", "give", "mine", "gracias", "dame", "lo tengo", "suelta", "merci",
              "donne", "danke", "gib", "谢谢", "給我", "给我", "拿", "放手")
# Every prompt an embodiment may override (Gemma(prompts={...})): a different object or a different host role. The defaults are the
# G1's verified wording; a changed prompt is a new, unmeasured variant: bench it before relying on it.
PROMPTS = {
    "hear_system": ("You are a drinks host robot. The guest names the colour of the can they want. The guest may speak any language. "
                    "Answer with the English colour word only, or none if no colour was named."),
    "hear_q": "Which colour did the guest ask for?",
    "take_q": "Transcribe this audio in English.",
    "vision_system": "You are a robot's vision module. Image left/right are robot left/right.",
    "crop_q": "What is the main colour of this can? Answer with one colour word.",
    "in_hand_q": "Is the robot holding the soda can in its hand, off the table?",
    "human_hand_q": "Is a human hand touching the soda can that the robot is holding? The robot's own white hand does not count.",
    "describe_system": "You label drink cans. Reply with at most four words: the colour, then the brand or flavour if readable, and any fruit pictured. No sentences.",
    "describe_q": "Describe this drink can in at most four words: colour, brand or flavour, and any fruit pictured.",
    "choose_system": ("You are a drinks host robot. The cans on the table are numbered: {cans}. The guest says which can they want, in any "
                      "language; they may name its colour, brand, flavour, picture, size or position. Answer with the number of that can only, "
                      "or none if no listed can matches."),
    "choose_q": "Which can does the guest want?",
    "object": "drink can",
}
STOP_WORDS = {"a", "an", "the", "of", "can", "this", "is", "it", "unknown", "brand", "colour", "color", "no", "not", "none", "fruit",
              "pictured", "visible", "drink", "soda"}


def image_part(img, size=image.SIZE):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image.to_jpeg(img, size)).decode()}}


def audio_part(wav_bytes):
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"}}


def colour_in(text):
    """First known colour word in a Gemma answer, or None."""
    return next((w for w in re.findall(r"[a-z]+", (text or "").lower()) if w in COLORS), None)


def _box_nums(text, W, H):
    nums = [int(n) for n in re.findall(r"\d+", text)][:4]
    if len(nums) != 4: return None
    y1, x1, y2, x2 = nums
    return (x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H)


class Gemma:
    def __init__(self, base_url="http://127.0.0.1:8080/v1", on_record=None, chat=timed_chat, prompts=None, timeout=20.0):
        self.base_url, self.on_record, self.chat, self.n, self.timeout = base_url, on_record or (lambda r: None), chat, 0, timeout
        self.p = dict(PROMPTS, **(prompts or {}))

    def ask(self, kind, question, messages, max_tokens, stage=None, media=None, extra=None, tag=TAG):
        payload = dict(BASE, max_tokens=max_tokens, messages=messages); payload.update(extra or {})
        t0 = time.perf_counter(); err = None
        try: r = self.chat(self.base_url, payload, timeout=self.timeout)       # bounded: a hung server must not hold an arm out for minutes
        except Exception as e: r, err = {"content": "", "timings": {}, "total_s": time.perf_counter() - t0}, "%s: %s" % (type(e).__name__, e)
        ans = (r.get("content") or "").strip(); tm = r.get("timings") or {}
        rec = {"ev": "gemma", "t": time.time(), "stage": stage, "kind": kind, "question": question, "answer": ans,
               "ms": int(round(r["total_s"] * 1000)), "tag": tag, "media": media,
               "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"), "predicted_n": tm.get("predicted_n"),
               "predicted_ms": tm.get("predicted_ms"), "cache_n": tm.get("cache_n")}
        if err: rec["error"] = err
        self.n += 1; self.on_record(rec)
        return ans.lower(), rec

    # --- verified on the G1 (Oct 18-23) ------------------------------------------------------------------------------------
    def describe_crop(self, crop, stage="LOOK", media=None):
        """One detected object's crop -> a short description ("purple la croix", "silver chi peach"), lowercase, at most 4 words."""
        msgs = [{"role": "system", "content": self.p["describe_system"]},
                {"role": "user", "content": [image_part(crop, size=None), {"type": "text", "text": self.p["describe_q"]}]}]
        ans, rec = self.ask("describe", self.p["describe_q"], msgs, 10, stage, media=media)
        words = [w for w in re.findall(r"[a-z0-9]+", ans) if w not in STOP_WORDS]
        return " ".join(words[:4]) or "unknown " + self.p["object"], rec

    def describe_named(self, name, crop, stage="LOOK", media=None):
        """An embodiment's own label plus the description of its best crop: the guest's word must be in the description
        (LEARNINGS.md), and a flavour vote drops the brand ('orange' for a Fanta whose crop reads 'orange fanta')."""
        if crop is None: return name or "unknown " + self.p["object"]
        d = self.describe_crop(crop, stage=stage, media=media)[0]
        if d.startswith("unknown"): return name or d
        return d if set(re.findall(r"[a-z0-9]+", name.lower())) <= set(d.split()) else "%s %s" % (d, name)

    def hear_choice(self, wav_bytes, descriptions, stage="UNDERSTAND"):
        """Spoken order (any language) + the numbered descriptions in the system prompt -> (index or None, raw answer, record).
        Known failure: an order-free clip ("Hm.") once answered 1 (G1, 2026-10-23): confirm before acting."""
        cans = "; ".join("%d: %s" % (i + 1, d) for i, d in enumerate(descriptions))
        msgs = [{"role": "system", "content": self.p["choose_system"].format(cans=cans)},
                {"role": "user", "content": [{"type": "text", "text": self.p["choose_q"]}, audio_part(wav_bytes)]}]
        ans, rec = self.ask("hear_choice", self.p["choose_q"], msgs, 4, stage, media="audio")
        return self._index(ans, descriptions), ans, rec

    def hear_colour(self, wav_bytes, stage="UNDERSTAND"):
        """Spoken order (any language) -> English colour word or None."""
        msgs = [{"role": "system", "content": self.p["hear_system"]},
                {"role": "user", "content": [{"type": "text", "text": self.p["hear_q"]}, audio_part(wav_bytes)]}]
        ans, rec = self.ask("hear_colour", self.p["hear_q"], msgs, 6, stage, media="audio")
        return colour_in(ans), ans, rec

    def crop_colour(self, crop, stage="LOCATE", media=None):
        """One crop -> one colour word (no system prompt, no resize)."""
        msgs = [{"role": "user", "content": [image_part(crop, size=None), {"type": "text", "text": self.p["crop_q"]}]}]
        return self.ask("crop_colour", self.p["crop_q"], msgs, 4, stage, media=media)

    def yesno(self, img, question, stage, media=None, system=None, tag=TAG):
        """Image + question after it -> (True / False / None, raw answer, record)."""
        q = question + " Answer with one word: yes or no."
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": [image_part(img), {"type": "text", "text": q}]}]
        ans, rec = self.ask("yesno", question, msgs, 3, stage, media=media, tag=tag)
        return (True if ans.startswith("yes") else False if ans.startswith("no") else None), ans, rec

    def transcribe(self, wav_bytes, stage="UNDERSTAND"):
        """A clip -> (English transcript, record). Empty for a clip with no speech: the order clip that was only the tail of
        "Listening." came back '' while hear_choice still answered 2 (AgileX, 2026-10-24; 5 clips with speech, all non-empty)."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": self.p["take_q"]}, audio_part(wav_bytes)]}]
        return self.ask("transcribe", self.p["take_q"], msgs, 16, stage, media="audio")

    def heard_take_it(self, wav_bytes, stage="HANDOVER"):
        """Short clip while offering the object -> (let-go phrase heard, transcript, record). Misheard on the G1 2026-10-23
        ("A cake.", "Packet"): keep a key or a timeout as the fallback release."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": self.p["take_q"]}, audio_part(wav_bytes)]}]
        ans, rec = self.ask("take_it", self.p["take_q"], msgs, 16, stage, media="audio")
        return any(w in ans for w in TAKE_WORDS), ans, rec

    def box(self, img, what, stage="LOCATE", media=None):
        """Gemma box_2d of `what` in the frame's pixels, or None. 100% on 78 fridge views (docs/LATENCY_RESULTS.md)."""
        q = "Detect " + what + ". Output only JSON: [{\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": \"can\"}] with coordinates normalized to 0-1000."
        msgs = [{"role": "user", "content": [image_part(img), {"type": "text", "text": q}]},
                {"role": "assistant", "content": "```json\n[\n  {\"box_2d\": ["}]
        ans, rec = self.ask("box", "Detect " + what, msgs, 24, stage, media=media, extra={"stop": ["]"]})
        W, H = image.size(img)
        return _box_nums(ans[ans.rindex("[") + 1:] if "[" in ans else ans, W, H), rec

    def warmup(self, img=None, stage="PREFLIGHT"):
        """One request that fills the cached prefix of the vision path (or a text turn without an image)."""
        if img is not None: return self.yesno(img, "Is anything visible in the image?", stage)[1:]
        msgs = [{"role": "system", "content": self.p["vision_system"]}, {"role": "user", "content": "Say ready. Answer with one word."}]
        return self.ask("warmup", "Say ready.", msgs, 3, stage)

    # --- added for the shared harness: measured on the Mac only, not on a robot --------------------------------------------
    def boxes(self, img, what=None, stage="LOOK", media=None, max_n=6):
        """Every `what` in the frame -> list of pixel boxes, for an embodiment without its own detector (the G1 uses YOLOE)."""
        what = what or "every " + self.p["object"]
        q = "Detect " + what + ". Output only JSON: [{\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": \"can\"}, ...] with coordinates normalized to 0-1000."
        msgs = [{"role": "user", "content": [image_part(img), {"type": "text", "text": q}]},
                {"role": "assistant", "content": "```json\n[\n  {\"box_2d\": ["}]
        ans, rec = self.ask("boxes", "Detect " + what, msgs, 24 * max_n, stage, media=media, extra={"stop": ["```"]})
        W, H = image.size(img); out = []
        for chunk in re.split(r"box_2d", "[" + ans):
            b = _box_nums(chunk, W, H)
            if b and b[2] > b[0] and b[3] > b[1]: out.append(b)
        return out[:max_n], rec

    def choose_text(self, text, descriptions, stage="UNDERSTAND"):
        """A typed (or externally transcribed) order against the numbered descriptions: hear_choice without audio."""
        cans = "; ".join("%d: %s" % (i + 1, d) for i, d in enumerate(descriptions))
        msgs = [{"role": "system", "content": self.p["choose_system"].format(cans=cans)},
                {"role": "user", "content": self.p["choose_q"] + " The guest says: " + text}]
        ans, rec = self.ask("choose_text", self.p["choose_q"], msgs, 4, stage, media="text")
        return self._index(ans, descriptions), ans, rec

    def ask_image(self, img, question, max_tokens=8, stage=None, system=None, media=None):
        """Any direct perception question, asked after the image (the rule that keeps accuracy). Short answers only."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": [image_part(img), {"type": "text", "text": question}]}]
        return self.ask("ask_image", question, msgs, max_tokens, stage, media=media)

    @staticmethod
    def _index(ans, descriptions):
        m = re.search(r"\d+", ans); idx = int(m.group()) - 1 if m else None
        return idx if idx is not None and 0 <= idx < len(descriptions) else None
