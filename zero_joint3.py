"""Set the zero of joint 3 at its folded stop, pressed by hand, for best accuracy.

Arm does this by itself at connect whenever the arm rests folded with a stale
zero, to within the half degree of play at the fold. Press the forearm
against its stop while this runs to remove that play as well.

    python zero_joint3.py

Keep pressing until the script says done, about five seconds.
"""

from __future__ import annotations

import time

import numpy as np

from piper_llm.arm import FOLD_J3_DEG, zero_joint3_at_fold
from piper_llm.config import ArmConfig


def main() -> None:
    from piper_control import piper_interface

    robot = piper_interface.PiperInterface(can_port=ArmConfig().can_port)
    time.sleep(0.5)
    q = np.degrees(robot.get_joint_positions())
    print(f"reported joints: {np.round(q, 1).tolist()}")
    if not (abs(q[1]) < 3 and FOLD_J3_DEG[0] < q[2] < FOLD_J3_DEG[1]):
        raise SystemExit("the arm is not resting folded with a stale zero: nothing to do, or fold it first")
    print("press the forearm against its fold stop ...")
    before, after, flag = zero_joint3_at_fold(robot, press_seconds=2.0)
    print(f"joint 3: {before:+.2f} -> {after:+.2f} deg, arm flag {flag}")
    if flag != 1 or abs(after) > 1.0:
        raise SystemExit("zero NOT set; press firmly and run again")
    print("done: joint 3 zero is at the fold until the next power-up")


if __name__ == "__main__":
    main()
