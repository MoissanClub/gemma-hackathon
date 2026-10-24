#!/usr/bin/env python3
"""One entry point for every robot: the shared fetch loop (core/loop.py) or a task program (core/robot_api.py) on an embodiment.

  python3 run.py --embodiment mock --fake-gemma          # no model, no robot: wiring test
  python3 run.py --embodiment mock                       # real Gemma (serve/run_llama.sh on the Mac, serve/serve-orin.sh on a Jetson)
  python3 run.py --embodiment mock --typed --orders 2    # type the orders instead of the recorded clips
  python3 run.py --embodiment mock --task tasks/give_leftmost.py
  python3 run.py --embodiment agilex --opt can_port=can0 # options go to embodiments/<name>/adapter.py make()

The G1's robot-proven program is embodiments/g1/brain/brain.py (its own CLI); see embodiments/g1/README.md."""
import argparse, os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import embodiments
from core.gemma import Gemma
from core.loop import Log, Loop
from core.robot_api import RobotAPI


def opt(s):
    k, _, v = s.partition("=")
    for cast in (int, float):
        try: return k, cast(v)
        except ValueError: pass
    return k, {"true": True, "false": False}.get(v.lower(), v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embodiment", "-e", default="mock", help="one of %s" % ", ".join(embodiments.KNOWN))
    ap.add_argument("--opt", type=opt, action="append", default=[], metavar="KEY=VALUE", help="adapter option (repeatable)")
    ap.add_argument("--server", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--fake-gemma", action="store_true", help="scripted answers instead of llama-server (core/fake.py)")
    ap.add_argument("--typed", action="store_true", help="type orders even if the robot has a microphone")
    ap.add_argument("--orders", type=int, default=1)
    ap.add_argument("--task", help="a Python file defining task(robot) against core/robot_api.py's API")
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    a = ap.parse_args()
    opts = dict(a.opt)
    if a.typed and a.embodiment == "mock": opts["typed"] = True
    robot = embodiments.load(a.embodiment, **opts)
    if a.typed: robot.capabilities = dict(robot.capabilities, mic=False)
    log = Log(a.runs)
    kw = {}
    if a.fake_gemma:
        from core import fake; kw["chat"] = fake.chat
    gem = Gemma(a.server, on_record=lambda r: log.event(**r), prompts=robot.prompts, **kw)
    print("embodiment %s, capabilities %s, log %s" % (robot.name, {k: v for k, v in robot.capabilities.items() if v}, log.dir), flush=True)
    robot.start()
    try:
        if a.task:
            log.begin(1); api = RobotAPI(robot, gem, log.event)
            runpy.run_path(a.task)["task"](api); log.end({"task": a.task, "gemma_calls": gem.n})
        else:
            loop = Loop(robot, gem, log)
            for n in range(1, a.orders + 1): loop.order(n)
    except KeyboardInterrupt:
        print("stop", flush=True); robot.stop()
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
