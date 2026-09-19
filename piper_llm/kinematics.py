"""PiPER kinematics from the MuJoCo Menagerie model.

The same model gives forward kinematics for the tool point and inverse
kinematics for a target, so the skill layer can work in metres rather than
joint angles.

Check the joint zero and sign conventions against the real arm before the first
torque-on run: read the joint angles in a known pose and compare.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]
GRASP_OFFSET_M = 0.12  # link6 origin to between the finger pads (tips at 0.135)
TIP_OFFSET_M = 0.015  # from the tool point to the closed finger tips, along the tool axis
R_DOWN = np.diag([-1.0, 1.0, -1.0])  # tool pointing straight down
# PiPER model from MuJoCo Menagerie (MIT), vendored so the repo is self-contained.
MJCF = Path(__file__).resolve().parent / "assets" / "agilex_piper" / "piper.xml"


def find_mjcf(explicit: str | None = None) -> Path:
    for candidate in (explicit, os.environ.get("PIPER_MJCF"), MJCF):
        if candidate and Path(candidate).expanduser().is_file():
            return Path(candidate).expanduser().resolve()
    raise FileNotFoundError(
        f"PiPER MJCF not found at {MJCF}. Set PIPER_MJCF to another piper.xml.")


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def tool_rotation(target_xyz: Vec, yaw: float, tilt: float = 0.0, about_fingers: bool = False) -> np.ndarray:
    """Tool pointing down with the given yaw, tilted by `tilt` (rad) about the
    wrist's pitch axis: positive tilt points the tips away from the base.

    With `about_fingers`, the tilt turns the tool about the line the fingers
    close along instead, pointing the tips towards the yaw direction. The
    fingers stay level, so a part held upright between them stays upright
    however far the tool leans, and the grip can face any pair of flats."""
    if about_fingers:
        return _rz(yaw) @ _ry(-tilt) @ R_DOWN
    phi = math.atan2(float(target_xyz[1]), float(target_xyz[0]))
    return _rz(phi) @ _ry(-tilt) @ _rz(-phi) @ _rz(yaw) @ R_DOWN


class Kinematics:
    """Tool-point forward and inverse kinematics."""

    def __init__(self, mjcf: str | None = None) -> None:
        import mujoco

        self._mj = mujoco
        spec = mujoco.MjSpec.from_file(str(find_mjcf(mjcf)))
        for key in list(spec.keys):
            spec.delete(key)
        spec.body("link6").add_site(name="tool", pos=[0.0, 0.0, GRASP_OFFSET_M],
                                    size=[0.005, 0, 0], group=4)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("tool").id
        self.qadr = np.array([self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 7)])
        self.dofadr = np.array([self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 7)])

    def tool_pose(self, q: Vec) -> Vec:
        """[x, y, z, yaw] of the tool point in the base frame."""
        self.data.qpos[self.qadr] = np.asarray(q, dtype=np.float64)
        self._mj.mj_kinematics(self.model, self.data)
        pos = self.data.site_xpos[self.site].copy()
        rot = self.data.site_xmat[self.site].reshape(3, 3)
        a = rot @ R_DOWN
        return np.array([pos[0], pos[1], pos[2], math.atan2(a[1, 0], a[0, 0])])

    def tool_point(self, q: Vec, along: float = 0.0) -> Vec:
        """Base-frame position of a point on the tool axis, `along` metres past
        the tool point. The closed finger tips are at TIP_OFFSET_M."""
        self.data.qpos[self.qadr] = np.asarray(q, dtype=np.float64)
        self._mj.mj_kinematics(self.model, self.data)
        rot = self.data.site_xmat[self.site].reshape(3, 3)
        return np.asarray(self.data.site_xpos[self.site] + rot @ np.array([0.0, 0.0, along]))

    def ik(self, target_xyz: Vec, yaw: float, seed: Vec, iters: int = 80,
           tilt: float = 0.0, about_fingers: bool = False) -> tuple[Vec, float]:
        """Joint angles for a tool target. Returns (q, position error in m).

        `tilt` leans the tool away from the base about the wrist's pitch axis,
        for targets the wrist cannot reach with a vertical tool.
        """
        mj = self._mj
        target = np.asarray(target_xyz, dtype=np.float64)
        rot = tool_rotation(target, yaw, tilt, about_fingers)
        q = np.asarray(seed, dtype=np.float64).copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        eye = np.eye(6)
        err = float("inf")
        for _ in range(iters):
            self.data.qpos[self.qadr] = q
            mj.mj_kinematics(self.model, self.data)
            mj.mj_comPos(self.model, self.data)
            pos = self.data.site_xpos[self.site]
            cur = self.data.site_xmat[self.site].reshape(3, 3)
            e_pos = target - pos
            err = float(np.linalg.norm(e_pos))
            r = rot @ cur.T
            angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
            axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
            e_rot = axis / (2 * math.sin(angle)) * angle if angle > 1e-6 else 0.5 * axis
            if err < 5e-4 and angle < 5e-3:
                break
            mj.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
            jp, jr = jacp[:, self.dofadr], jacr[:, self.dofadr]
            pinv = jp.T @ np.linalg.inv(jp @ jp.T + 1e-4 * np.eye(3))
            dq = pinv @ (e_pos * min(1.0, 0.05 / max(err, 1e-9)))
            null = eye - pinv @ jp
            jrn = jr @ null
            dq2 = jrn.T @ np.linalg.solve(jrn @ jrn.T + 5e-2 * np.eye(3),
                                          e_rot * min(1.0, 0.1 / max(angle, 1e-9)) - jr @ dq)
            dq = dq + null @ dq2
            scale = float(np.max(np.abs(dq)))
            if scale > 0.15:
                dq *= 0.15 / scale
            q = q + dq
        return q, err
