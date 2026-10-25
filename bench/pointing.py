#!/usr/bin/env python3
"""Pointing benchmark on Where2Place (embodiments/agilex/notes/2026-10-25-benchmark-plan.md, step 1).

Each of the 100 questions asks for points in a free-space region; a point inside the region's mask is a hit. Score per question:
the fraction of its points inside the mask (RoboPoint's metric, arXiv 2406.10721), 0 when nothing parses. No model judges.

  python3 bench/pointing.py --model gemma --server http://127.0.0.1:8090/v1      # llama.cpp, one request at a time
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


def ask_gemma(server, q):
    payload = {"model": "gemma4-e2b", "cache_prompt": True, "temperature": 0.0, "top_k": 1, "max_tokens": 200,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": [image_part(DATA / "images" / q["image"]), {"type": "text", "text": q["text"]}]}]}
    r = timed_chat(server, payload)
    return r["content"], r["total_s"], (r.get("timings") or {})


MODELS = {"gemma": ask_gemma}


def report(rows):
    def line(name, rs):
        s = [r["score"] for r in rs]
        return "%-8s n=%3d  mean %.3f  any-hit %3d  no points %2d" % (name, len(rs), statistics.mean(s), sum(x > 0 for x in s),
                                                                     sum(r["n_points"] == 0 for r in rs))
    print(line("all", rows))
    for cat in sorted({r["category"] for r in rows}): print(line(cat, [r for r in rows if r["category"] == cat]))
    lat = sorted(r["s"] for r in rows)
    print("latency s: p50 %.2f  p90 %.2f   scale: %s" % (lat[len(lat) // 2], lat[int(len(lat) * 0.9)],
          {k: sum(r["scale"] == k for r in rows) for k in ("unit", "1000", None)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--server", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.report:
        report([json.loads(l) for l in open(a.report)]); return
    if not a.model: ap.error("--model or --report")
    out = ROOT / "runs" / "bench" / ("pointing-%s-%s.jsonl" % (time.strftime("%Y%m%dT%H%M%S"), a.model))
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("w") as f:
        for q in questions()[:a.limit]:                  # serial: one inference job at a time; every image is different
            text, s, tm = MODELS[a.model](a.server, q)
            pts, scale = parse(text)
            row = {"id": q["question_id"], "image": q["image"], "category": q["category"], "answer": text, "n_points": len(pts),
                   "scale": scale, "score": round(score(pts, mask(q["image"])), 4), "s": round(s, 3),
                   "prompt_n": tm.get("prompt_n"), "predicted_n": tm.get("predicted_n")}
            rows.append(row); f.write(json.dumps(row) + "\n"); f.flush()
            print("%3d %-6s %.2f  %4.2fs  %s" % (row["id"], row["category"], row["score"], s, text[:70].replace("\n", " ")), flush=True)
    report(rows); print("saved", out)


if __name__ == "__main__":
    main()
