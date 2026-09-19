"""Check 6: the grasp signal. No model, no camera.

The runs confirm a grasp from the gripper alone: Arm.holding() is True when the
fingers were told to close and stopped part way, and False when they closed on
nothing. This hand-feeds the object to test both answers.

    python grasp_check.py

The arm homes with the gripper open and stays powered. Hold the cube between the
fingers when asked; keep hands clear of the fingers themselves.
"""

from __future__ import annotations

from piper_llm.arm import Arm
from piper_llm.config import ArmConfig, SafetyLimits
from piper_llm.kinematics import Kinematics
from piper_llm.safety import Governor
from piper_llm.skills import Runner


def report(arm: Arm, label: str) -> None:
    print(f"  {label}: opening {arm.gripper():.2f}, holding {arm.holding()}, jammed {arm.jammed()}")


def main() -> None:
    arm = Arm(ArmConfig(speed_percent=10))
    run = Runner(arm, Kinematics(), Governor(SafetyLimits()))
    try:
        print("homing, gripper opens")
        arm.home()
        report(arm, "open, empty")
        input("hold the cube between the fingers, then press Enter to close on it")
        run.hold(3.0, 0.0)
        report(arm, "closed on the cube")
        ok_held = arm.holding()
        input("let go of the cube; press Enter to open")
        run.hold(2.0, 1.0)
        report(arm, "opened")
        input("take the cube away; press Enter to close on nothing")
        run.hold(3.0, 0.0)
        report(arm, "closed on nothing")
        ok_empty = not arm.holding()
        run.hold(2.0, 1.0)
        print("check 6:", "PASSED" if ok_held and ok_empty else
              f"FAILED (held detected: {ok_held}, empty detected: {ok_empty})")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
