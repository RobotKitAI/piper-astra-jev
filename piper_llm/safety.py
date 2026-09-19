"""Safety layer under the model's choice.

The model picks any skill it likes. These checks run underneath and can refuse a
skill, slow a motion or stop it. Every refusal returns a reason, which the caller
feeds back to the model as an observation.

This reduces risk. It does not make the arm safe. Keep the e-stop in reach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from piper_llm.config import SafetyLimits

Vec = npt.NDArray[np.float64]


@dataclass
class SafetyEvent:
    kind: str  # precondition | workspace | deviation | confidence | speed | fault
    skill: str
    detail: str


@dataclass
class Governor:
    """Checks a chosen skill and the motion it produces."""

    limits: SafetyLimits = field(default_factory=SafetyLimits)
    events: list[SafetyEvent] = field(default_factory=list)
    _first_action: bool = True

    def reset(self) -> None:
        self._first_action = True

    def check_confidence(self, skill: str, confidence: float | None) -> tuple[bool, str]:
        """An uncertain decision is not acted on; look again instead."""
        if confidence is None or confidence >= self.limits.min_confidence:
            return True, ""
        detail = f"confidence {confidence:.2f} below {self.limits.min_confidence}"
        self.events.append(SafetyEvent("confidence", skill, detail))
        return False, detail

    def check_precondition(self, skill: str, state: dict[str, Any]) -> tuple[bool, str]:
        """Refuse a skill whose preconditions do not hold."""
        held = bool(state.get("object_held"))
        placed = bool(state.get("object_placed"))
        closed = bool(state.get("gripper_closed"))
        target = state.get("target_xy")
        z = float(state.get("tool_z", 0.0))
        over_target = bool(state.get("over_target"))
        over_place = bool(state.get("over_place_target"))
        rules = {
            "approach_target": target is not None and not held and not placed,
            "descend_to_target": target is not None and over_target and not held and not closed
                                 and not placed,
            "close_gripper": not closed and not placed and (
                bool(state["at_grasp_height"]) if "at_grasp_height" in state else z < 0.05),
            "rotate_grasp": bool(state.get("fingers_jammed")),
            "lift": closed or held,
            "move_over_place": held,
            "lower_to_place": held and over_place,
            "open_gripper": closed,
            "move_over_spanner": held,
            "align_over_spanner": held and over_place,
            "slide_down": held and bool(state.get("aligned")),
            "let_go": held and bool(state.get("on_spanner")),  # only well onto the spanner
            "retreat": True,
            "done": bool(state.get("task_complete")),
        }
        if rules.get(skill, False):
            return True, ""
        detail = f"preconditions for {skill} do not hold"
        self.events.append(SafetyEvent("precondition", skill, detail))
        return False, detail

    def check_target(self, skill: str, point: Vec) -> tuple[bool, str]:
        """Refuse a target outside the workspace box or below the table."""
        lo = np.asarray(self.limits.workspace_low)
        hi = np.asarray(self.limits.workspace_high)
        if point[2] < self.limits.table_z:
            detail = f"target {point[2] * 1000:.0f} mm is below the table limit"
            self.events.append(SafetyEvent("workspace", skill, detail))
            return False, detail
        if np.any(point < lo - 1e-6) or np.any(point > hi + 1e-6):
            detail = f"target {np.round(point, 3).tolist()} is outside the workspace box"
            self.events.append(SafetyEvent("workspace", skill, detail))
            return False, detail
        return True, ""

    def check_first_action(self, skill: str, commanded: Vec, measured: Vec) -> tuple[bool, str]:
        """The first command of a run must be near the measured pose."""
        if not self._first_action:
            return True, ""
        self._first_action = False
        gap = float(np.linalg.norm(np.asarray(commanded)[:3] - np.asarray(measured)[:3]))
        if gap <= self.limits.first_action_max_m:
            return True, ""
        detail = f"first command is {gap * 100:.1f} cm from the measured pose"
        self.events.append(SafetyEvent("fault", skill, detail))
        return False, detail

    def step_size(self, skill: str, clearance_m: float) -> float:
        """Smaller steps near an object or the table."""
        if clearance_m >= self.limits.caution_m:
            return self.limits.max_step_m
        self.events.append(SafetyEvent(
            "speed", skill, f"slowed: {clearance_m * 100:.1f} cm clearance"))
        return self.limits.slow_step_m

    def check_deviation(self, skill: str, commanded: Vec, measured: Vec) -> tuple[bool, str]:
        """A command the arm cannot follow means something is in the way."""
        gap = float(np.linalg.norm(np.asarray(commanded)[:3] - np.asarray(measured)[:3]))
        if gap <= self.limits.max_deviation_m:
            return True, ""
        detail = f"arm is {gap * 100:.1f} cm behind the command: stopping"
        self.events.append(SafetyEvent("deviation", skill, detail))
        return False, detail

    def check_joint_step(self, skill: str, q_cmd: Vec, q_now: Vec) -> tuple[bool, str]:
        """Refuse a joint jump larger than the per-step limit."""
        deltas = np.abs(np.asarray(q_cmd) - np.asarray(q_now))
        jump = float(np.max(deltas))
        if jump <= self.limits.max_joint_step_rad:
            return True, ""
        detail = (f"joint {int(np.argmax(deltas)) + 1} jump {math.degrees(jump):.1f} deg over the limit "
                  f"(steps deg {np.round(np.degrees(deltas), 1).tolist()})")
        self.events.append(SafetyEvent("fault", skill, detail))
        return False, detail

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.events:
            out[e.kind] = out.get(e.kind, 0) + 1
        return out
