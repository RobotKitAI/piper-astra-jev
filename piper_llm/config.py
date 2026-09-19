"""Limits and fixed numbers for the real arm. Review these before every run."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

CALIB = Path(__file__).resolve().parent.parent / "calibration.json"

# piper_sdk joint limits (rad), inset by 0.05 for margin. Joint 6 is held to
# +-100 deg (datasheet), not +-120: commanding beyond it has dropped the arm into
# its protective damp in practice.
JOINT_LOW = (-2.5679, 0.05, -2.647, -1.695, -1.17, -1.695)
JOINT_HIGH = (2.5679, 3.09, -0.05, 1.695, 1.17, 1.695)
GRIPPER_MAX_M = 0.07  # full stroke


def joint_offsets(path: Path = CALIB) -> tuple[float, ...]:
    """Corrections added to the reported joint angles (rad), from calibration.json.

    The arm reports joint 3 a few degrees off the model's zero; calibrate.py
    estimates the correction together with the camera transform. Zeros until
    a calibration exists.
    """
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return (0.0,) * 6
    return tuple(float(v) for v in data.get("joint_offsets", (0.0,) * 6))


@dataclass(frozen=True)
class ArmConfig:
    """Arm settings. Start slow."""

    can_port: str = "can0"
    control_hz: float = 15.0
    speed_percent: int = 10  # firmware speed; raise only after a clean run
    gripper_effort_nm: float = 1.0
    # Tool pointing straight down at (0.22, 0, 0.12): the skill runner demands a
    # vertical tool from its first step, so home must already be one.
    home_joints: tuple[float, ...] = (0.0, 1.40, -0.93, 0.0, 1.18, 0.0)
    park_joints: tuple[float, ...] = (-2.2, 0.9, -0.6, 0.0, 0.3, 0.0)  # clears the camera view
    # Folded rest: joints 1..3 go to the arm's own zeros (the mechanical stops),
    # the wrist to these angles. Power is cut only there.
    rest_wrist: tuple[float, float, float] = (0.0, 0.39, 0.0)


@dataclass(frozen=True)
class SafetyLimits:
    """Hard limits enforced in code, below the model."""

    table_z: float = 0.005  # no tool point below this
    workspace_low: tuple[float, float, float] = (0.10, -0.35, 0.005)
    workspace_high: tuple[float, float, float] = (0.50, 0.35, 0.40)
    max_step_m: float = 0.010  # per control step at full speed
    slow_step_m: float = 0.004  # per control step near an object
    caution_m: float = 0.06  # distance that triggers the slow step
    max_joint_step_rad: float = math.radians(8.0)  # catches IK branch flips (20+); mid-move tilt corrections reach 5
    first_action_max_m: float = 0.08  # first command must be near the measured pose
    max_deviation_m: float = 0.03  # command vs measurement before a protective stop
    min_confidence: float = 0.10  # Jev confidence below this is not acted on


@dataclass(frozen=True)
class SceneConfig:
    """Object names for the perception prompts, and grasp heights."""

    target: str = "red cube"
    # Plain "tray": colour words mislead Grounding DINO on the brown crate in
    # the lab ("blue tray" boxes the whole table), and it is the only tray.
    place_target: str = "tray"
    # Largest plausible detection box (px at 1280x720): the cube is ~80 px, the
    # crate ~700 px; anything larger is the table or the arm.
    target_max_px: int = 200
    place_max_px: int = 900
    target_max_z: float = 0.06  # the target sits on the table: a sighting higher up is on the gripper
    # Grasp planning on a mask (SAM 3): solid material stands at least `solid_rise`
    # above the table; the grasp goes to the thickest part between these widths.
    solid_rise: float = 0.010
    grasp_width_min: float = 0.015
    grasp_width_max: float = 0.045
    pad_depth: float = 0.025  # the tool point goes this far below the part's crest: the tips 4 cm under its top
    # When the target is a part (a handle), phrases for the whole object, tried in
    # order: the part's axis runs from the object's centre out to the part, and
    # the whole object tells how far it can hang below the grasp once lifted.
    target_whole: tuple[str, ...] = ()
    place_rim_z: float = 0.10  # the tray's rim when depth cannot measure it
    # The lab's tray is a 10 cm deep crate: carry the cube above its rim (the
    # cube hangs about grasp_z below the tool point) and release it 3 cm inside.
    approach_z: float = 0.16
    grasp_z: float = 0.025  # pads 2.5 cm up on the 29 mm cube, tips 1 cm clear of the table
    place_z: float = 0.07


# An object taken by its handle (runs 4 and 5: a trowel).
# Grounding DINO boxes the whole object, SAM 3 segments the handle; the box is
# larger than a cube's and a raised handle is taller than the cube filter allows.
HANDLE_SCENE = dict(target="handle", target_whole=("tool", "object"), target_max_px=600,
                    target_max_z=0.12, grasp_z=0.02)  # pads low on a round grip


# The trowel with its geometry (run 6): SAM 3 masks the whole tool for the pose, and scores
# "tool" higher than "trowel". The grasp comes from the STL, so the camera-only limits do not apply.
TROWEL_SCENE = dict(target="tool", target_max_px=700, target_max_z=0.12, grasp_z=0.02)


# The chrome pipe fitting (run 7): SAM 3 names it best as "metal object".
FITTING_SCENE = dict(target="metal object", target_max_px=250, target_max_z=0.06, grasp_z=0.02)
