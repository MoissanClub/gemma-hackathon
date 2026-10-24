"""Hand the guest the leftmost object. The same file runs on every embodiment (core/robot_api.py)."""


def task(robot):
    cans = [c for c in robot.look() if c["x"] is not None]
    if not cans:
        robot.say("I do not see anything I can reach.")
        return
    can = max(cans, key=lambda c: c["x"])          # x is centimetres to the robot's left
    robot.say("The %s one." % can["label"])
    if robot.pick(can["id"]):
        robot.give()
