"""PiPER arm control through piper_control, with limits applied in code.

piper_control wraps AgileX piper_sdk and handles the enable sequence and the
cases where the arm stops responding. Written against piper_control 1.5:
the robot object is `piper_interface.PiperInterface`, motion speed is a mode
setting, and the enable and disable sequences live in `piper_init`.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt

from piper_llm.config import GRIPPER_MAX_M, JOINT_HIGH, JOINT_LOW, ArmConfig, joint_offsets

Vec = npt.NDArray[np.float64]
STALE_J3_DEG = 3.0  # a zeroed joint 3 never reads above this
# Per-joint sense of the arm relative to the MuJoCo model. All six agree:
# a flip of joint 6 tried on 2026-09-19 mirrored every grasp angle (a plan of
# +50 deg came out at -32 deg, along the handle). Kept as the one place to set.
JOINT_SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
FOLD_J3_DEG = (6.0, 10.0)  # what a stale joint 3 reads when the arm rests folded


def zero_joint3_at_fold(robot: Any, press_seconds: float = 0.0) -> tuple[float, float, int | None]:
    """Set joint 3's zero where it is, with the piper_sdk demo's sequence.

    The arm forgets this zero at every power-up and then reports the fold
    about 8 degrees off, which shifts every target and leaves the forearm
    dropping when power is cut. Whole arm enabled, motor 3 alone disabled
    for press_seconds, zero set, motor 3 re-enabled, all motors off again.
    Returns (before, after, flag): joint 3 in degrees and the arm's success flag.
    """
    import math

    from piper_control import piper_init

    piper_init.enable_arm(robot)
    robot.piper.MotionCtrl_2(0x01, 0x01, 10, 0x00)
    time.sleep(0.3)
    robot.piper.DisableArm(3)
    time.sleep(0.5 + press_seconds)
    before = math.degrees(robot.get_joint_positions()[2])
    robot.piper.JointConfig(3, 0xAE)
    time.sleep(0.5)
    resp = robot.piper.GetRespInstruction()
    flag = getattr(getattr(resp, "instruction_response", resp), "is_set_zero_successfully", None)
    after = math.degrees(robot.get_joint_positions()[2])
    robot.piper.EnableArm(3)
    time.sleep(0.3)
    robot.piper.MotionCtrl_2(0x01, 0x01, 10, 0x00)
    time.sleep(0.2)
    robot.command_joint_positions(robot.get_joint_positions())
    time.sleep(0.5)
    piper_init.disable_arm(robot)
    time.sleep(1.0)
    return before, after, flag


class Arm:
    """Joint-space control of one PiPER arm."""

    def __init__(self, cfg: ArmConfig = ArmConfig()) -> None:
        from piper_control import piper_connect, piper_init, piper_interface

        self.cfg = cfg
        self._init = piper_init
        self._stop_code = piper_interface.EmergencyStop.STOP
        piper_connect.activate()  # needs sudo rights on the CAN interface
        self.robot = piper_interface.PiperInterface(can_port=cfg.can_port)
        time.sleep(0.3)
        q = np.degrees(self.robot.get_joint_positions())
        if q[2] > STALE_J3_DEG:  # the arm forgot the fold zero (it does at every power-up)
            if abs(q[1]) < 3 and FOLD_J3_DEG[0] < q[2] < FOLD_J3_DEG[1]:
                before, after, flag = zero_joint3_at_fold(self.robot)
                if flag != 1 or abs(after) > 1.0:
                    raise RuntimeError(f"joint 3 zeroing failed (flag {flag}, reads {after:+.1f} deg)")
                print(f"joint 3 zero set at the fold: {before:+.1f} -> {after:+.1f} deg")
            else:
                raise RuntimeError(
                    f"joint 3 reports {q[2]:+.1f} deg, beyond its zero, and the arm is not resting "
                    "folded: fold it, then reconnect or run python zero_joint3.py")
        # reset_arm cuts motor power for under a second, then enables torque and
        # motion at full speed. The arm must be resting when this runs.
        piper_init.reset_arm(self.robot)
        piper_init.reset_gripper(self.robot)
        # Apply the configured speed before the first command.
        self.robot.set_arm_mode(speed=cfg.speed_percent)
        # Joint angles in and out of this class are in the model's convention:
        # calibration.json holds the corrections to the arm's reported zeros.
        self.offsets = np.asarray(joint_offsets(), dtype=np.float64)
        self._last_gripper_cmd: float | None = None

    # --- state ---------------------------------------------------------------
    def joints(self) -> Vec:
        """Six joint angles in radians, in the model's convention."""
        return np.asarray(self.robot.get_joint_positions(), dtype=np.float64) * JOINT_SIGN + self.offsets

    def gripper(self) -> float:
        """Opening, 0 closed .. 1 open."""
        width = float(self.robot.get_gripper_state()[0])  # metres
        return float(np.clip(width / GRIPPER_MAX_M, 0.0, 1.0))

    def holding(self) -> bool:
        """True when the fingers were told to close but stopped part-way.

        This is proprioception, not vision, and it is the most reliable grasp
        check on this arm.
        """
        cmd = self._last_gripper_cmd
        return cmd is not None and cmd < 0.2 and 0.15 < self.gripper() < 0.85

    def closed(self) -> bool:
        """True when the fingers were last told to close, however far they got:
        a wide part stops them past half the stroke."""
        cmd = self._last_gripper_cmd
        return cmd is not None and cmd < 0.2

    def jammed(self) -> bool:
        """Fingers commanded shut but stopped wide: a corner is in the way.

        The opening reads about 5 mm high under load (drive play), so a 29 mm
        cube held flat reads 34 mm (0.49) and its corner about 46 mm (0.66):
        the line sits between them at 0.6.
        """
        cmd = self._last_gripper_cmd
        return cmd is not None and cmd < 0.2 and self.gripper() > 0.6 and not self.holding()

    # --- commands ------------------------------------------------------------
    def command_joints(self, q: Vec, gripper: float, clamp: bool = True) -> None:
        """Send one joint target, clamped to the joint limits.

        clamp=False is for the rest pose only: the fold sits on the mechanical
        stops, inside the margin the limits keep.
        """
        q = np.asarray(q, dtype=np.float64)
        if clamp:
            q = np.clip(q, JOINT_LOW, JOINT_HIGH)
        self.robot.command_joint_positions(((q - self.offsets) * JOINT_SIGN).tolist())
        grip = float(np.clip(gripper, 0.0, 1.0))
        self.robot.command_gripper(position=grip * GRIPPER_MAX_M,
                                   effort=self.cfg.gripper_effort_nm)
        self._last_gripper_cmd = grip

    def move_to(self, q: Vec, gripper: float, seconds: float = 4.0, clamp: bool = True) -> None:
        """Interpolated move, for homing, parking and resting."""
        start = self.joints()
        q = np.asarray(q, dtype=np.float64)
        steps = max(1, int(seconds * self.cfg.control_hz))
        for i in range(1, steps + 1):
            self.command_joints(start + (q - start) * i / steps, gripper, clamp)
            time.sleep(1.0 / self.cfg.control_hz)

    def home(self) -> None:
        self.move_to(np.asarray(self.cfg.home_joints), 1.0)

    def park(self) -> None:
        """Move aside so the camera sees the table. Not a pose to cut power in."""
        self.move_to(np.asarray(self.cfg.park_joints), 1.0)

    def rest_joints(self) -> Vec:
        """The folded rest in the model's convention: the arm's own zeros on
        joints 1..3 (reported zero is model `offset`), wrist from the config."""
        return np.concatenate([self.offsets[:3], np.asarray(self.cfg.rest_wrist)])

    def rest(self) -> None:
        """Fold onto the mechanical stops with the gripper closed, then cut power."""
        self.move_to(self.rest_joints(), 0.0, clamp=False)
        self.settle()
        self.disable()

    def settle(self, timeout: float = 3.0) -> None:
        """Wait until the joints stop moving: the firmware lags the last commands."""
        end = time.time() + timeout
        last = self.joints()
        while time.time() < end:
            time.sleep(0.3)
            now = self.joints()
            if np.degrees(np.abs(now - last).max()) < 0.2:
                return
            last = now

    # --- stopping ------------------------------------------------------------
    def stop(self) -> None:
        """Hold the current pose."""
        self.command_joints(self.joints(), self.gripper())

    def emergency_stop(self) -> None:
        """Firmware quick stop.

        Sends EmergencyStop.STOP (0x01). Never send RESUME (0x02) through
        MotionCtrl_1: the arm drops.
        """
        self.robot.set_emergency_stop(self._stop_code)

    def disable(self) -> None:
        """Cut motor power where the arm is, without moving it first.

        Only for an arm that is resting or supported: an extended arm drops.
        """
        self._init.disable_arm(self.robot)
        self._init.disable_gripper(self.robot)

    def close(self) -> None:
        """Fold onto the stops and disable. Disabling while extended drops the arm."""
        try:
            self.rest()
        finally:
            if self.robot.is_arm_enabled():
                self.disable()

    def status_error(self) -> str | None:
        """Arm fault text, if the firmware reports one."""
        status = getattr(self.robot.get_arm_status(), "arm_status", None)
        err = getattr(status, "err_code", 0)
        return f"arm error code {err}" if err else None

    def __enter__(self) -> "Arm":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
