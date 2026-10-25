#!/usr/bin/env python3
"""Pointing benchmark on Where2Place (embodiments/agilex/notes/2026-10-25-benchmark-plan.md, step 1).

Each of the 100 questions asks for points in a free-space region; a point inside the region's mask is a hit. Score per question:
the fraction of its points inside the mask (RoboPoint's metric, arXiv 2406.10721), 0 when nothing parses. No model judges.

  python3 bench/pointing.py --model gemma --server http://127.0.0.1:8090/v1      # llama.cpp, one request at a time
  ~/.venvs/molmo2/bin/python bench/pointing.py --model molmo2-4b                 # Transformers, 4-bit, own venv
  python3 bench/pointing.py --model gemini-er                                     # API key in ~/.config/gemini/api-key
  python3 bench/pointing.py --model gemini-er --resume runs/bench/pointing-<stamp>-gemini-er.jsonl   # continue tomorrow
  python3 bench/pointing.py --report runs/bench/pointing-<stamp>-gemma.jsonl     # re-score a run

Data: bench/images-where2place/ (git-ignored; huggingface.co/datasets/wentao-yuan/where2place). Results: runs/bench/ (git-ignored).
Needs OpenCV for the masks. Points in [0, 1] are normalized (x, y), as the question asks; larger values are Gemma's native
0-1000 [y, x] and flagged as scale "1000".
"""
import argparse, json, re, statistics, sys, time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from client import image_part, timed_chat  # noqa: E402

DATA = ROOT / "bench" / "images-where2place"
NUM = r"-?\d+(?:\.\d+)?"
PAIR = re.compile(r"\(\s*(%s)\s*,\s*(%s)\s*\)|\[\s*(%s)\s*,\s*(%s)\s*\]" % (NUM, NUM, NUM, NUM))


def questions():
    return [json.loads(line) for line in (DATA / "point_questions.jsonl").read_text().splitlines() if line.strip()]


def mask(name):
    stem = Path(name).stem
    for ext in (".png", ".jpg"):                      # 31 masks ship as both: the png is clean 0/255, the jpg has compression noise
        p = DATA / "masks" / (stem + ext)
        if p.exists():
            return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127
    raise FileNotFoundError(stem)


def parse(text):
    """Every (x, y) or [x, y] pair -> (points normalized to [0, 1], scale used: 'unit', '1000' or None)."""
    pts = [tuple(float(v) for v in m.groups() if v is not None) for m in PAIR.finditer(text)]
    if not pts: return [], None
    if max(max(p) for p in pts) <= 1.0: return pts, "unit"
    # 0-1000 is Gemma's own format ({"point": [y, x]}, as its box_2d is [ymin, xmin, ...]): read as [y, x]. Drawn on images 00 and
    # 02 (2026-10-25), [y, x] put the points on the named mugs and in the drawer; [x, y] put them on unrelated objects.
    return [(x / 1000.0, y / 1000.0) for y, x in pts], "1000"


def score(points, m):
    if not points: return 0.0
    H, W = m.shape
    hits = [0 <= x < 1 and 0 <= y < 1 and bool(m[int(y * H), int(x * W)]) for x, y in points]
    return sum(hits) / len(hits)


FORMAT = " Your answer should be formatted"      # the dataset's format sentence; identical in all 100 questions


def instruction(q):
    """The question without the dataset's format sentence, for a model asked in its own point format: Molmo2-4B mixed the
    requested normalized tuples with its 0-1000 scale ('(0.23, 739)', 3 of 5 smoke-test answers, 2026-10-25)."""
    return q["text"].split(FORMAT)[0]


def ask_gemma(server, q, text=None):
    payload = {"model": "gemma4-e2b", "cache_prompt": True, "temperature": 0.0, "top_k": 1, "max_tokens": 200,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": [image_part(DATA / "images" / q["image"]), {"type": "text", "text": text or q["text"]}]}]}
    r = timed_chat(server, payload)
    return r["content"], r["total_s"], (r.get("timings") or {})


MOLMO = {}
# the Molmo2 model card's regexes, except that an answer cut by max_new_tokens (no closing quote; it lists 20+ points) still counts
MOLMO_COORDS = re.compile(r'<(?:points|tracks)[^>]*? coords="([0-9\t:;, .]+)')
MOLMO_FRAME = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
MOLMO_POINT = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def parse_molmo(text):
    """Molmo2's <points coords="1 1 523 413 2 ..."> (image, then id x y triples; x first, scaled by 1000) -> normalized (x, y)."""
    pts = [(float(m.group(2)) / 1000, float(m.group(3)) / 1000)
           for c in MOLMO_COORDS.finditer(text) for f in MOLMO_FRAME.finditer(c.group(1)) for m in MOLMO_POINT.finditer(f.group(2))]
    return (pts, "molmo") if pts else parse(text)


def ask_molmo(server, q, model_id="allenai/Molmo2-4B"):
    """Molmo2 through Transformers, its language model 4-bit (bitsandbytes nf4, bf16 compute) to fit an 8 GB GPU. Stop the Gemma
    server first: Molmo2 rebuilds its 744 MB embedding table every step, and the two do not fit in 8 GB together (3070 Laptop).
    Runs in its own venv (~/.venvs/molmo2: torch, transformers==4.57.1, bitsandbytes, molmo_utils, opencv). `server` unused."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    if not MOLMO:
        # the language model in 4-bit; the vision backbone and the head stay bf16 (4-bit vision layer norms fail: "LayerNormKernelImpl
        # not implemented for 'Byte'")
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                   llm_int8_skip_modules=["vision_backbone", "lm_head"])
        MOLMO["proc"] = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        MOLMO["model"] = AutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True, quantization_config=quant,
                                                                     dtype=torch.bfloat16, device_map="cuda")
    proc, model = MOLMO["proc"], MOLMO["model"]
    torch.cuda.synchronize(); t0 = time.perf_counter()                           # from the image file, like Gemma's request
    msgs = [{"role": "user", "content": [dict(type="image", image=Image.open(DATA / "images" / q["image"]).convert("RGB")),
                                         dict(type="text", text=instruction(q))]}]                 # question after the image
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    torch.cuda.synchronize(); s = time.perf_counter() - t0
    new = out[0, inputs["input_ids"].size(1):]
    return proc.tokenizer.decode(new, skip_special_tokens=True), s, {"prompt_n": int(inputs["input_ids"].size(1)), "predicted_n": int(len(new))}


GEMINI_KEY = Path.home() / ".config" / "gemini" / "api-key"     # never printed or logged
GEMINI_FORMAT = (' The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. '
                 'The points are in [y, x] format normalized to 0-1000.')                          # Google's ER pointing prompt


def ask_gemini(server, q, model="gemini-robotics-er-2-preview"):
    """Gemini Robotics-ER over the REST API (stdlib), free tier: one request at a time; a 429 waits for the server's retry delay
    and tries again, so no question is skipped. Asked in Google's own point format ([y, x] on 0-1000), read by parse()."""
    import base64, urllib.error, urllib.request
    body = json.dumps({"contents": [{"parts": [
        {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode((DATA / "images" / q["image"]).read_bytes()).decode()}},
        {"text": instruction(q) + GEMINI_FORMAT}]}]}).encode()
    url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model
    for attempt in range(8):
        req = urllib.request.Request(url, body, {"Content-Type": "application/json", "x-goog-api-key": GEMINI_KEY.read_text().strip()})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=180) as r: d = json.load(r)
        except urllib.error.HTTPError as e:
            err = e.read().decode()[:400]
            if e.code not in (429, 500, 503): raise RuntimeError("Gemini HTTP %d: %s" % (e.code, err))
            m = re.search(r'"retryDelay":\s*"(\d+)', err); wait = int(m.group(1)) + 1 if m else 15 * (attempt + 1)
            print("  Gemini %d: waiting %d s" % (e.code, wait), flush=True); time.sleep(wait); continue
        s = time.perf_counter() - t0
        text = "".join(p.get("text", "") for c in d.get("candidates", [])[:1] for p in c.get("content", {}).get("parts", []))
        u = d.get("usageMetadata", {})
        return text, s, {"prompt_n": u.get("promptTokenCount"), "predicted_n": u.get("candidatesTokenCount")}
    raise RuntimeError("Gemini: still rate-limited after 8 tries")


MODELS = {"gemma": ask_gemma, "molmo2-4b": ask_molmo, "gemini-er": ask_gemini,
          "gemma-instruction": lambda server, q: ask_gemma(server, q, instruction(q))}   # same prompt as Molmo2
PARSERS = {"molmo2-4b": parse_molmo}


def report(rows):
    def line(name, rs):
        s = [r["score"] for r in rs]
        return "%-8s n=%3d  mean %.3f  any-hit %3d  no points %2d" % (name, len(rs), statistics.mean(s), sum(x > 0 for x in s),
                                                                     sum(r["n_points"] == 0 for r in rs))
    print(line("all", rows))
    for cat in sorted({r["category"] for r in rows}): print(line(cat, [r for r in rows if r["category"] == cat]))
    lat = sorted(r["s"] for r in rows)
    print("latency s: p50 %.2f  p90 %.2f   format: %s" % (lat[len(lat) // 2], lat[int(len(lat) * 0.9)],
          {k: sum(r["scale"] == k for r in rows) for k in sorted({r["scale"] for r in rows}, key=str)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--server", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--report")
    ap.add_argument("--resume", help="a results file to continue: skips its questions, appends the rest (free tier: 20 a day)")
    a = ap.parse_args()
    if a.report:
        report([json.loads(l) for l in open(a.report)]); return
    if not a.model: ap.error("--model or --report")
    out = Path(a.resume) if a.resume else ROOT / "runs" / "bench" / ("pointing-%s-%s.jsonl" % (time.strftime("%Y%m%dT%H%M%S"), a.model))
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in out.open()] if a.resume else []
    done = {r["id"] for r in rows}
    with out.open("a" if a.resume else "w") as f:
        for q in [q for q in questions()[:a.limit] if q["question_id"] not in done]:   # serial: one job at a time; every image differs
            text, s, tm = MODELS[a.model](a.server, q)
            pts, scale = PARSERS.get(a.model, parse)(text)
            row = {"id": q["question_id"], "image": q["image"], "category": q["category"], "answer": text, "n_points": len(pts),
                   "scale": scale, "score": round(score(pts, mask(q["image"])), 4), "s": round(s, 3),
                   "prompt_n": tm.get("prompt_n"), "predicted_n": tm.get("predicted_n")}
            rows.append(row); f.write(json.dumps(row) + "\n"); f.flush()
            print("%3d %-6s %.2f  %4.2fs  %s" % (row["id"], row["category"], row["score"], s, text[:70].replace("\n", " ")), flush=True)
    report(rows); print("saved", out)


if __name__ == "__main__":
    main()
