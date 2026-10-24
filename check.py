#!/usr/bin/env python3
"""Offline checks for the shared harness (no model, no robot, a few seconds): `python3 -B check.py`. Run before every push.
Each embodiment adds its own gate (the G1's: embodiments/g1/brain/check.py, needs numpy + OpenCV, ~2 min)."""
import io, os, py_compile, sys, tempfile, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from core import fake, image
from core.embodiment import Refused
from core.gemma import Gemma
from core.loop import Log, Loop
from core.robot_api import RobotAPI
import embodiments

ok = True


def case(name, cond, note=""):
    global ok; ok &= bool(cond); print("%-34s %s  %s" % (name, "ok  " if cond else "FAIL", note), flush=True)


def run(robot, typed=(), orders=1):
    feed = iter(typed); out = io.StringIO(); log = Log(tempfile.mkdtemp(prefix="gh-check-"))
    gem = Gemma(chat=fake.chat, on_record=lambda r: log.event(**r))
    loop = Loop(robot, gem, log, typed=lambda p: next(feed, ""), take_wait=1.0)
    with contextlib.redirect_stdout(out): res = [loop.order(n) for n in range(1, orders + 1)]
    return res, out.getvalue()


for root, _, files in os.walk(HERE):
    if ".git" in root or "/g1/" in root + "/": continue           # the G1 brain needs numpy/OpenCV: its own check.py compiles it
    for f in files:
        if f.endswith(".py"): py_compile.compile(os.path.join(root, f), doraise=True)
case("compile (core, mock, agilex, bench)", True)
jpg = open(os.path.join(HERE, "embodiments/mock/media/scene_two_cans.jpg"), "rb").read()
case("jpeg_size", image.jpeg_size(jpg) == (640, 480), str(image.jpeg_size(jpg)))
try:
    c = image.crop(jpg, (20, 140, 140, 300)); case("crop (OpenCV or sips)", image.jpeg_size(c) == (140, 180), str(image.jpeg_size(c)))
    r = image.to_jpeg(jpg); case("resize to 480x336", image.jpeg_size(r) == (480, 336), str(image.jpeg_size(r)))
except Exception as e:
    case("crop/resize", False, "%s (needs OpenCV or macOS sips)" % e)
res, out = run(embodiments.load("mock"))
case("mock order, recorded clip", res == ["delivered"], res[0])
res, out = run(embodiments.load("mock", typed=True), typed=["a coke please", "the silver one"])
case("typed: honest negative, then pick", res == ["delivered"] and "I do not see that. I see" in out and "silver chi lemon one" in out, res[0])
res, out = run(embodiments.load("mock", typed=True), typed=["", "", ""])
case("typed: no order", res == ["not_understood"], res[0])
res, out = run(embodiments.load("mock", miss=True))
case("miss -> skill_failed, home", res == ["skill_failed"] and "[mock] home" in out, res[0])
m = embodiments.load("mock")
try:
    m.locate(m.frame(), (0, 100, 50, 200), "purple"); case("border refusal", False)
except Refused as e:
    case("border refusal", "edge of my view" in str(e), str(e)[:40])
with contextlib.redirect_stdout(io.StringIO()) as out:
    api = RobotAPI(embodiments.load("mock"), Gemma(chat=fake.chat)); cans = api.look(); got = api.pick(cans[0]["id"]) and api.give()
case("robot_api look/pick/give", len(cans) == 2 and got and cans[0]["x"] is not None, str([(c["label"], c["x"]) for c in cans]))
print("CHECK: %s" % ("PASS" if ok else "FAIL")); sys.exit(0 if ok else 1)
