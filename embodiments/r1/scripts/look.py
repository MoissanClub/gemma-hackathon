#!/usr/bin/env python3
"""Perception probe on R1, nothing moves: one wrist frame, Gemma boxes `what`, each box located (embodiments/r1/adapter.py), an
annotated image and a JSON line per object under runs/<timestamp>-r1-look/.

  .venv/bin/python -B embodiments/r1/scripts/look.py --ob-config ~/r1-gemma/ob-config.xml "the crumpled paper" [--n 3]"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, ROOT)
import cv2
from core.embodiment import Refused
from core.gemma import Gemma
from embodiments.r1.adapter import R1

ap = argparse.ArgumentParser()
ap.add_argument("what", nargs="?", default=None, help="what to box (default: the core's 'every drink can')")
ap.add_argument("--ob-config", required=True)
ap.add_argument("--server", default="http://127.0.0.1:8080/v1")
ap.add_argument("--n", type=int, default=1, help="frames (fresh each time: never time a repeated image)")
a = ap.parse_args()
out = os.path.join(ROOT, "runs", time.strftime("%Y%m%dT%H%M%S") + "-r1-look"); os.makedirs(out)
recs = open(os.path.join(out, "look.jsonl"), "a")
gem = Gemma(a.server, on_record=lambda r: recs.write(json.dumps(r, default=str) + "\n"))
r = R1(ob_config=a.ob_config); r.start()
try:
    for i in range(a.n):
        f = r.frame(); t = time.time()
        boxes, rec = gem.boxes(f.image, what=a.what)
        ms = int((time.time() - t) * 1000); img = f.image.copy()
        print("frame %d: %d box(es), Gemma %d ms, answer %r" % (i, len(boxes), ms, rec.get("answer", "")[:80]), flush=True)
        for j, b in enumerate(boxes):
            try:
                tg = r.locate(f, b, a.what); row = {"frame": i, "box": [round(v) for v in b], "xyz": tg.xyz, **tg.data}; col = (0, 255, 0)
                label = "z %.2f m h %.1f cm" % (tg.xyz[2], tg.data["height_m"] * 100)
            except Refused as e:
                row = {"frame": i, "box": [round(v) for v in b], "refused": str(e)}; col, label = (0, 0, 255), "refused"
            print("  ", json.dumps(row), flush=True); recs.write(json.dumps(dict(row, ev="locate")) + "\n")
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
            cv2.putText(img, label, (int(b[0]), max(15, int(b[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.imwrite(os.path.join(out, "frame%02d.jpg" % i), img)
finally:
    r.shutdown(); recs.close()
print("log", out)
