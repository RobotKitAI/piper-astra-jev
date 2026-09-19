"""The tasks the run notebooks share: what the decider sees and what each
chosen skill does. `PickPlace` serves runs 1 to 7, `SlideOver` run 8.

The decider only picks skill names. `PickPlace` turns camera frames into the
state it reads and a chosen skill into a tool target, and it remembers what the
camera loses: the grasp planned while the target was in clear view (the fingers
hide it at grasp height), the tray while the arm is over it, whether the object
has been let go, and where it was picked.

Heights follow the object. When it plans the grasp, it measures how far the
object can reach below the grasp point: the farthest visible point of the whole
object from it, a bound on how deep anything can hang once lifted. It carries
the object that much above the tray's rim, and lowers it until that depth ends
just above the tray's floor. A carry the arm cannot reach is reported in
`problem` before anything is lifted, and so are two grasps that left the
gripper empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import mujoco
import numpy as np
import numpy.typing as npt

from piper_llm.config import ArmConfig, JOINT_HIGH, JOINT_LOW, SafetyLimits, SceneConfig
from piper_llm.kinematics import tool_rotation
from piper_llm.skills import plan_grasp, whole_detection

Vec = npt.NDArray[np.float64]


def mask_points(mask: Any, depth: Any, frame: Any) -> Vec:
    """Base-frame points of a mask's pixels. A pixel without depth is taken
    where its ray meets the table: right for a flat part lying on it, and
    farther than the truth for a raised one, so distances err long."""
    cam = frame.cam
    ys, xs = np.nonzero(np.asarray(mask))
    z = depth[ys, xs].astype(np.float64)
    ok = (z > 0.05) & (z < 3.0)
    pts = np.empty((xs.size, 3))
    if ok.any():
        c = np.stack([(xs[ok] - cam.cx) * z[ok] / cam.fx, (ys[ok] - cam.cy) * z[ok] / cam.fy, z[ok]], axis=1)
        pts[ok] = c @ frame.rot.T + frame.pos
    if (~ok).any():
        d = np.stack([(xs[~ok] - cam.cx) / cam.fx, (ys[~ok] - cam.cy) / cam.fy,
                      np.ones(int((~ok).sum()))], axis=1) @ frame.rot.T
        pts[~ok] = frame.pos + d * (-frame.pos[2] / d[:, 2])[:, None]
    return pts


def detection_points(det: Any, depth: Any, frame: Any, above: float | None = None,
                     below: float | None = None) -> Vec:
    """Base-frame points of a detection. A mask gives all its points, flat
    parts on the table included, up to `below`. A box also holds table and
    whatever stands behind the object, so only its points standing between
    `above` and `below` count."""
    if det.mask is not None:
        pts = mask_points(det.mask, depth, frame)
        return pts if below is None else pts[pts[:, 2] < below]
    h, w = depth.shape
    x0, y0, x1, y1 = (int(round(v)) for v in det.box)
    box = np.zeros((h, w), np.uint8)
    box[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 1
    pts = mask_points(box, depth, frame)
    keep = np.ones(len(pts), bool)
    if above is not None:
        keep &= pts[:, 2] > above
    if below is not None:
        keep &= pts[:, 2] < below
    return pts[keep]


def box_reach(det: Any, frame: Any, point: Vec) -> float:
    """How far an object known only by its box can reach from a point: the
    farthest of the box's corners where their rays meet the table. The box
    holds table and background too, so its pixels cannot say more."""
    cam = frame.cam
    x0, y0, x1, y1 = det.box
    px = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float64)
    d = np.stack([(px[:, 0] - cam.cx) / cam.fx, (px[:, 1] - cam.cy) / cam.fy, np.ones(4)], axis=1) @ frame.rot.T
    corners = frame.pos + d * (-frame.pos[2] / d[:, 2])[:, None]
    return float(np.linalg.norm(corners - point, axis=1).max())


@dataclass
class PickPlace:
    scene: SceneConfig
    detector: Any
    frame: Any
    runner: Any = None  # for the reach check before lifting
    clearance: float = 0.03  # the held object's lowest point passes this far above the rim
    release_gap: float = 0.02  # and ends this far above the tray's floor when let go

    geometry: Any = None  # an ObjectGeometry (the part's STL): pose from the silhouette, grasp and reach from it
    pose: np.ndarray | None = None  # the part's located pose (4x4, base frame)
    fit: float = 0.0  # how well the geometry's outline covers the mask (IoU)
    grasp: Vec | None = None  # tool point of the planned grasp
    grasp_yaw: float = 0.0
    hang: float = 0.03  # how far the object can reach below the grasp point
    tray: Vec | None = None
    rim: float | None = None
    floor: float = 0.0
    released: bool = False
    pick: Vec | None = None  # where the gripper closed, to put the object back
    missed: int = 0  # grasps that left the gripper empty: it closed on nothing, or the part slipped out
    plan_problem: str = ""  # why the plan cannot be carried out; each reach check replaces its own

    @property
    def problem(self) -> str:
        """Why the run must stop: a plan the arm cannot carry out, or grasps that keep missing."""
        if self.missed >= 2:
            return f"the gripper held nothing after {self.missed} grasps"
        return self.plan_problem

    @property
    def carry_z(self) -> float:
        rim = self.scene.place_rim_z if self.rim is None else self.rim
        return max(self.scene.approach_z, rim + self.hang + self.clearance)

    @property
    def release_z(self) -> float:
        return max(self.scene.place_z, self.floor + self.hang + self.release_gap)

    # --- what the decider sees --------------------------------------------------
    def observe(self, rgb: Any, depth: Any, pose: Vec, arm: Any) -> dict[str, Any]:
        held, closed = arm.holding(), arm.closed()
        if not (held or closed or self.released):
            self._plan(rgb, depth, pose)
        self._see_tray(rgb, depth, pose)
        g = None if self.released else self.grasp
        tz = float(pose[2])
        clear = tz > self.scene.approach_z - 0.015
        return {
            "tool_xyz": np.round(pose[:3], 3).tolist(),
            "target_xy": None if g is None else np.round(g[:2], 3).tolist(),
            "place_target_xy": None if self.tray is None else np.round(self.tray[:2], 3).tolist(),
            "over_target": bool(g is not None and np.linalg.norm(pose[:2] - g[:2]) < 0.03),
            "over_place_target": bool(self.tray is not None and np.linalg.norm(pose[:2] - self.tray[:2]) < 0.05),
            "tool_z": round(tz, 3),
            "at_grasp_height": bool(g is not None and abs(tz - g[2]) < 0.01),
            "at_carry_height": bool(tz > self.carry_z - 0.015),
            "at_release_height": bool(tz < self.release_z + 0.015),
            "at_clear_height": bool(clear),
            "gripper_closed": bool(closed),
            "object_held": bool(held),
            "object_placed": bool(self.released),
            "fingers_jammed": bool(arm.jammed()),
            "task_complete": bool(self.released and not held and clear),
        }

    def _plan(self, rgb: Any, depth: Any, pose: Vec) -> None:
        if self.grasp is not None and np.linalg.norm(pose[:2] - self.grasp[:2]) < 0.08:
            return  # the gripper is over the target and in the picture: keep the plan made in clear view
        d = self.detector.best(rgb, self.scene.target, max_px=self.scene.target_max_px)
        if d is not None and self.geometry is not None:
            self._plan_on_geometry(d)
            return
        plan = None if d is None else plan_grasp(d, rgb, depth, self.frame, self.scene, self.detector)
        if plan is None or plan[0][2] > self.scene.target_max_z:
            return  # nothing, or higher than the target can be (on the gripper): keep the last clear plan
        self.grasp, self.grasp_yaw = plan
        if d.mask is None:  # a box detector already boxes the whole object; no other box is tighter
            self.hang = max(0.02, box_reach(d, self.frame, self.grasp))
            self._check_reach()
            return
        whole = (whole_detection(self.detector, rgb, d, self.scene) if self.scene.target_whole else None) or d
        if whole.mask is None:
            self.hang = max(0.02, box_reach(whole, self.frame, self.grasp))
        else:
            pts = detection_points(whole, depth, self.frame, below=self.scene.target_max_z)
            if len(pts):
                self.hang = max(0.02, float(np.percentile(np.linalg.norm(pts - self.grasp, axis=1), 99.5)))
        self._check_reach()

    def _plan_on_geometry(self, d: Any) -> None:
        """Pose from the silhouette, then grasp and reach from the geometry: no depth."""
        if d.mask is None:
            self.plan_problem = "the geometry needs a mask to be fitted: use SAM 3"
            return
        T, fit, _ = self.geometry.locate(d.mask, self.frame)
        if fit < 0.6:
            return  # the geometry does not match this sighting: keep the last plan
        g = self.geometry.grasp(T, lowest_tool=self.scene.grasp_z)
        if g is None:
            self.plan_problem = "no grasp on the geometry fits the gripper"
            return
        self.pose, self.fit = T, fit
        self.grasp, self.grasp_yaw = g[0], g[1]
        self.hang = max(0.02, self.geometry.reach(T, self.grasp))
        self._check_reach()

    def _see_tray(self, rgb: Any, depth: Any, pose: Vec) -> None:
        if self.tray is not None and np.linalg.norm(pose[:2] - self.tray[:2]) < 0.20:
            return  # the arm is over the tray and hides it: keep the last clear sighting
        t = self.detector.best(rgb, self.scene.place_target, max_px=self.scene.place_max_px)
        p = None if t is None else self.frame.to_base(*t.centre, depth)
        if p is None:
            return
        self.tray = np.asarray(p, dtype=np.float64)
        h = detection_points(t, depth, self.frame)[:, 2]
        if h.size > 50:
            self.rim = float(np.percentile(h, 98))
            self.floor = float(np.clip(np.percentile(h, 5), 0.0, 0.03))
        if self.grasp is not None:
            self._check_reach()

    def _check_reach(self) -> None:
        z = self.carry_z
        top = SafetyLimits().workspace_high[2]
        self.plan_problem = ""
        if z > top + 1e-6:
            self.plan_problem = (f"the object can reach {100 * self.hang:.0f} cm below the grasp: clearing the "
                            f"{100 * (self.rim or self.scene.place_rim_z):.0f} cm rim needs a {100 * z:.0f} cm "
                            f"carry, above the {100 * top:.0f} cm the arm may use")
            return
        if self.runner is None:
            return
        seed = np.asarray(ArmConfig().home_joints, dtype=np.float64)
        spots = [("over the grasp", self.grasp[:2], z)]
        if self.tray is not None:
            spots += [("over the tray", self.tray[:2], z), ("at the release", self.tray[:2], self.release_z)]
        kept = self.runner._lean  # a check, not a move: leave the runner as it was
        try:
            for name, xy, zz in spots:
                self.runner._lean = 0.0
                spot = np.array([xy[0], xy[1], zz])
                q, err = self.runner.solve(spot, self.runner.reachable_yaw(spot, self.grasp_yaw, seed), seed)
                if err > 0.003 or np.any(q < JOINT_LOW) or np.any(q > JOINT_HIGH):
                    self.plan_problem = f"the arm cannot reach {100 * zz:.0f} cm {name}"
                    return
        finally:
            self.runner._lean = kept

    # --- where each skill goes -----------------------------------------------------
    def target(self, skill: str, pose: Vec) -> tuple[Vec | None, float, float]:
        yaw, z = self.grasp_yaw, float(pose[2])
        here = np.asarray(pose[:3], dtype=np.float64)
        if skill == "approach_target":
            return (None if self.grasp is None else np.array([*self.grasp[:2], self.scene.approach_z])), yaw, 1.0
        if skill == "descend_to_target":
            return (None if self.grasp is None else self.grasp.copy()), yaw, 1.0
        if skill == "close_gripper":
            return here.copy(), yaw, 0.0
        if skill == "rotate_grasp":
            self.grasp_yaw = (yaw + math.pi / 4 + math.pi / 2) % math.pi - math.pi / 2
            return here.copy(), self.grasp_yaw, 1.0
        if skill == "lift":
            return np.array([here[0], here[1], self.carry_z]), yaw, 0.0
        if skill in ("move_over_place", "lower_to_place"):
            if self.tray is None:
                return None, yaw, 0.0
            zz = self.carry_z if skill == "move_over_place" else self.release_z
            return np.array([self.tray[0], self.tray[1], zz]), yaw, 0.0
        if skill == "open_gripper":
            return here.copy(), yaw, 1.0
        if skill == "retreat":
            return np.array([here[0], here[1], max(self.scene.approach_z, z)]), yaw, 1.0
        return None, yaw, 1.0

    def after(self, skill: str, reached: bool, state: dict[str, Any], pose: Vec) -> None:
        if skill == "close_gripper":
            self.pick = np.asarray(pose[:3], dtype=np.float64).copy()
        if skill == "open_gripper" and not state["object_held"]:
            self.missed += 1  # nothing to let go: the grasp missed, or the part slipped out on the way
        elif skill == "open_gripper" and reached and state["over_place_target"]:
            self.released = True

    # --- before folding --------------------------------------------------------------
    def finish(self, runner: Any, arm: Any) -> None:
        """Leave the gripper empty and clear of the scene: a run that ends with
        the object held places it in the tray when the tray is known and the
        carry is possible, otherwise puts it back where it was picked."""
        runner.gov.reset()
        yaw = self.grasp_yaw
        pose = runner.kin.tool_pose(arm.joints())
        if arm.holding():
            runner.move_tool_to("lift", np.array([pose[0], pose[1], self.carry_z]), yaw, 0.0)
            if self.tray is not None and not self.problem:
                runner.move_tool_to("move_over_place", np.array([*self.tray[:2], self.carry_z]), yaw, 0.0)
                runner.move_tool_to("lower_to_place", np.array([*self.tray[:2], self.release_z]), yaw, 0.0)
            elif self.pick is not None:
                runner.move_tool_to("move_back", np.array([*self.pick[:2], self.carry_z]), yaw, 0.0)
                runner.move_tool_to("put_back", self.pick.copy(), yaw, 0.0)
            runner.hold(2.0, 1.0)
            pose = runner.kin.tool_pose(arm.joints())
        z = max(self.scene.approach_z, float(pose[2]))
        runner.move_tool_to("retreat", np.array([pose[0], pose[1], z]), yaw, arm.gripper())
        runner.move_tool_to("clear", np.array([0.25, 0.0, z]), yaw, arm.gripper())  # fold from near home


# --- run 8: the pipe fitting slid down over a standing box spanner ------------------------------------

_FAST, _SLOW, _CREEP = 0.03, 0.012, 0.005  # m/s: transfers, the last few cm to a part, down the spanner
_BOX = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _turn(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _heading(yaw: float) -> Vec:
    """Which way the fingertips lean at this yaw, seen from above: away from the arm."""
    return np.array([math.cos(yaw), math.sin(yaw)])


@dataclass
class SlideOver:
    """Run 8: take the standing pipe fitting and slide it down over a standing box spanner.

    Both parts are chrome, so both are placed by their geometry: every SAM 3 mask of
    a metal part is fitted with both STLs, and the better fit names it. The wrist
    cannot point the gripper straight down this high, so the gripper leans `lean`
    about the line between its fingers the whole run: the fingers stay level, and
    the fitting held upright between them stays upright. The gripper takes the
    fitting `shift` towards the arm, so the fitting sits at the fingers' outer end
    and can go further down before the gripper meets the spanner's top.

    Over the spanner, the camera measures where the held fitting really is against
    the spanner's axis, and the arm moves by the difference until the two agree
    within `aligned_m`: the hand-eye error up there is a few millimetres, and the
    bore leaves about 2 mm a side. The fitting then goes down slowly while the depth
    camera watches the gap between the gripper and the spanner's top. Let go well
    onto the spanner, it slides down to the table.
    """

    detector: Any
    frame: Any
    runner: Any
    cam: Any  # fresh frames for the looks and the depth watch
    fitting: Any  # ObjectGeometry of the fitting
    spanner: Any  # ObjectGeometry of the spanner, loaded with min_prob=0 to keep its standing poses
    rec: Any = None  # a Recorder: notes, and a picture of each look
    phrase: str = "metal object"  # what SAM 3 is asked for: it finds both parts
    lean: float = math.radians(45.0)  # about the line between the fingers, the whole run
    grip_z: float = 0.023  # tool point above the fitting's bottom where the fingers take it: as low as they clear the table
    shift: float = 0.010  # the gripper holds the fitting this far towards the arm
    up_gap: float = 0.080  # over the fitting the gripper comes and goes this far above the spanner's top
    carry_gap: float = 0.030  # the fitting's bottom above the spanner's top while carried
    hover_gap: float = 0.020  # ... as it arrives over the spanner
    look_gap: float = 0.008  # ... while the camera measures where it is
    aligned_m: float = 0.0008  # slide down only with the fitting's axis this close to the spanner's
    floor_z: float = 0.010  # the fitting's bottom goes no lower than this
    gap_stop: float = 0.010  # stop when the depth camera sees the gripper this close above the spanner's top
    onto_m: float = 0.020  # let go only with the fitting's bottom this far below the spanner's top
    table_clear: float = 0.0015  # the fingers' lowest corner at least this far above the table where they take it
    max_looks: int = 8

    problem: str = ""  # why the run must stop; finish() then leaves the gripper empty
    fitting_pose: np.ndarray | None = None
    spanner_pose: np.ndarray | None = None
    top: float = 0.0  # the spanner's top above the table
    height: float = 0.0  # the fitting's height
    grasp: Vec | None = None  # tool point where the fingers take the fitting
    yaw_pick: float = 0.0
    yaw_place: float = 0.0
    aim: Vec | None = None  # the gripper's xy over the spanner, moved by each look
    aligned: bool = False
    error: float | None = None  # the last look: the fitting's axis from the spanner's (m)
    looks: int = 0
    bottom: float | None = None  # how low the fitting's bottom went down the spanner
    on_spanner: bool = False
    released: bool = False
    _spanner_px: np.ndarray | None = None
    _close_readings: int = 0
    _moved: bool = False

    def __post_init__(self) -> None:
        self.runner.fixed_lean = self.lean

    @property
    def up_z(self) -> float:
        return self.top + self.up_gap

    @property
    def carry_z(self) -> float:
        return self.top + self.carry_gap + self.grip_z

    @property
    def look_z(self) -> float:
        return self.top + self.look_gap + self.grip_z

    # --- what the decider sees --------------------------------------------------
    def observe(self, rgb: Any, depth: Any, pose: Vec, arm: Any) -> dict[str, Any]:
        held, closed = arm.holding(), arm.closed()
        if not (held or closed or self.released) and (
                self.grasp is None or np.linalg.norm(pose[:2] - self.grasp[:2]) > 0.08):
            self._plan(rgb)  # while the arm is away from the fitting and the view is clear
        g = None if self.released else self.grasp
        tz = float(pose[2])
        up = self.grasp is not None and tz > self.carry_z - 0.015
        return {
            "tool_xyz": np.round(pose[:3], 3).tolist(),
            "target_xy": None if g is None else np.round(g[:2], 3).tolist(),
            "place_target_xy": None if self.spanner_pose is None else np.round(self.spanner_pose[:2, 3], 3).tolist(),
            "over_target": bool(g is not None and np.linalg.norm(pose[:2] - g[:2]) < 0.02),
            "over_place_target": bool(self.aim is not None and np.linalg.norm(pose[:2] - self.aim) < 0.02),
            "tool_z": round(tz, 3),
            "at_grasp_height": bool(g is not None and abs(tz - g[2]) < 0.01),
            "at_carry_height": bool(up),
            "gripper_closed": bool(closed),
            "object_held": bool(held),
            "aligned": self.aligned,
            "alignment_error_mm": None if self.error is None else round(1000 * self.error, 1),
            "spanner_top_mm": round(1000 * self.top),
            "fitting_bottom_mm": round(1000 * (tz - self.grip_z)) if held else None,
            "on_spanner": self.on_spanner,
            "object_placed": self.released,
            "at_clear_height": bool(up),
            "task_complete": bool(self.released and not held and up),
        }

    def _plan(self, rgb: Any) -> None:
        """Both parts from their geometry, then the grasp and the path, all checked before anything moves."""
        found: dict[str, tuple[np.ndarray, float]] = {}
        for d in self.detector.detect(rgb, [self.phrase]):
            if d.mask is None:
                continue
            fits = {"spanner": self.spanner.locate(d.mask, self.frame)[:2],
                    "fitting": self.fitting.locate(d.mask, self.frame)[:2]}
            name = max(fits, key=lambda k: fits[k][1])
            if name not in found or fits[name][1] > found[name][1]:
                found[name] = fits[name]
        if set(found) != {"spanner", "fitting"}:
            self.problem = f"the camera must see both parts; it found {', '.join(sorted(found)) or 'neither'}"
            return
        (Ts, fit_s), (Tf, fit_f) = found["spanner"], found["fitting"]
        top = float((self.spanner.mesh.vertices @ Ts[:3, :3].T + Ts[:3, 3])[:, 2].max())
        fz = (self.fitting.mesh.vertices @ Tf[:3, :3].T + Tf[:3, 3])[:, 2]
        if min(fit_s, fit_f) < 0.85:
            self.problem = f"the outlines do not match the geometry well enough (spanner {fit_s:.2f}, fitting {fit_f:.2f})"
        elif abs(Ts[2, 2]) < 0.9 or abs(Tf[2, 2]) < 0.9:
            self.problem = "both parts must stand upright on the table"
        elif not 0.13 < top < 0.15:
            self.problem = f"the spanner's top is {1000 * top:.0f} mm up, not about 141 mm: is it standing?"
        elif np.linalg.norm(Tf[:2, 3] - Ts[:2, 3]) < 0.08:
            self.problem = "the fitting is on or beside the spanner: stand it at least 8 cm away"
        if self.problem:
            return
        self.fitting_pose, self.spanner_pose, self.top, self.height = Tf, Ts, top, float(fz.max() - fz.min())
        # the fingers across a pair of flats, turned so the tips lean as nearly straight out from the base as the flats allow
        corner = math.atan2(Tf[1, 0], Tf[0, 0])  # the fitting's x axis points at a hex corner
        out = math.atan2(Tf[1, 3], Tf[0, 3])
        self.yaw_pick = min((_wrap(corner + math.radians(30 + 60 * k) - math.pi / 2) for k in range(6)),
                            key=lambda y: abs(_wrap(y - out)))
        self.yaw_place = math.atan2(Ts[1, 3], Ts[0, 3])
        self.grasp = np.array([*(Tf[:2, 3] - self.shift * _heading(self.yaw_pick)), self.grip_z])
        self.aim = Ts[:2, 3] - self.shift * _heading(self.yaw_place)
        h, w = rgb.shape[:2]
        self._spanner_px = self.spanner.outline(Ts, self.frame, (0, 0, w, h)).astype(bool)
        self._check_path()

    def _check_path(self) -> None:
        """Every waypoint inside the workspace and reachable with the lean held to within a degree,
        and the fingers clear of the table where they take the fitting."""
        kin, gov = self.runner.kin, self.runner.gov
        g, a = self.grasp[:2], self.aim
        q = np.asarray(self.runner.arm.cfg.home_joints, dtype=np.float64)
        for name, xy, z, yaw in (("above the fitting", g, self.up_z, self.yaw_pick),
                                 ("at the fitting", g, self.grip_z, self.yaw_pick),
                                 ("lifted", g, self.carry_z, self.yaw_pick),
                                 ("over the spanner", a, self.carry_z, self.yaw_place),
                                 ("at the look", a, self.look_z, self.yaw_place),
                                 ("down the spanner", a, self.floor_z + self.grip_z, self.yaw_place)):
            p = np.array([xy[0], xy[1], z])
            ok, why = gov.check_target("plan", p)
            if not ok:
                self.problem = f"{name}: {why}"
                return
            q, err = kin.ik(p, yaw, q, iters=400, tilt=self.lean, about_fingers=True)
            kin.tool_pose(q)
            cur = kin.data.site_xmat[kin.site].reshape(3, 3)
            off = math.degrees(math.acos(float(np.clip((np.trace(tool_rotation(p, yaw, self.lean, True) @ cur.T) - 1) / 2,
                                                       -1.0, 1.0))))
            if err > 0.002 or off > 1.0 or np.any(q < JOINT_LOW) or np.any(q > JOINT_HIGH):
                self.problem = f"the arm cannot reach {name} with the gripper leaning {math.degrees(self.lean):.0f} deg"
                return
            if name == "at the fitting":
                low = min(min(c[:, 2].min() for c in self._finger_boxes(q, grip)) for grip in (1.0, 0.56))
                if low < self.table_clear:
                    self.problem = (f"the leaning fingertips would come within {1000 * low:.1f} mm of the table: "
                                    "take the fitting higher or lean less")
                    return

    def _finger_boxes(self, q: Vec, grip: float) -> list[np.ndarray]:
        """The corners of the fingers' collision boxes (the Menagerie model), the arm at q, the gripper this open."""
        kin = self.runner.kin
        m, d = kin.model, kin.data
        d.qpos[kin.qadr] = q
        d.qpos[m.joint("joint7").qposadr[0]], d.qpos[m.joint("joint8").qposadr[0]] = grip * 0.035, -grip * 0.035
        mujoco.mj_kinematics(m, d)
        return [d.geom_xpos[g] + _BOX * m.geom_size[g] @ d.geom_xmat[g].reshape(3, 3).T
                for g in range(m.ngeom) if m.geom_group[g] == 3 and m.body(m.geom_bodyid[g]).name in ("link7", "link8")]

    # --- what each skill does ------------------------------------------------------
    def act(self, skill: str) -> tuple[bool, str]:
        """Carry out a chosen skill. Returns (reached, reason); the reason is empty when it finished."""
        arm = self.runner.arm
        here = self.runner.kin.tool_pose(arm.joints())[:3]
        g = None if self.grasp is None else self.grasp[:2]
        if skill in ("approach_target", "descend_to_target", "lift", "move_over_spanner", "align_over_spanner",
                     "slide_down") and g is None:
            return False, "no plan yet"
        if skill == "approach_target":  # through open space: joint by joint, high enough to pass over the spanner
            return self._glide(skill, g, self.up_z, self.yaw_pick, 1.0, _FAST, joint_space=True)
        if skill == "descend_to_target":  # quickly to 3 cm above the grasp, then slowly
            ok, why = self._glide("lower", g, self.grip_z + 0.03, self.yaw_pick, 1.0, _FAST)
            return self._glide(skill, g, self.grip_z, self.yaw_pick, 1.0, _SLOW) if ok else (ok, why)
        if skill == "close_gripper":
            self._note(skill)
            self.runner.hold(1.5, 0.0)
            if not arm.holding():
                self.problem = "the fingers closed on nothing"
            return True, ""
        if skill == "lift":
            return self._glide(skill, here[:2], self.carry_z, self.yaw_pick, 0.0, _FAST)
        if skill == "move_over_spanner":  # across at the carry height, then slowly down to just above its top
            ok, why = self._glide(skill, self.aim, self.carry_z, self.yaw_place, 0.0, _FAST)
            return self._glide("hover", self.aim, self.top + self.hover_gap + self.grip_z, self.yaw_place, 0.0,
                               0.010) if ok else (ok, why)
        if skill == "align_over_spanner":
            return self._align()
        if skill == "slide_down":
            return self._slide()
        if skill == "let_go":
            self._note(skill)
            self.runner.hold(1.5, 1.0)
            self.released = True
            return True, ""
        if skill == "retreat":  # straight up: near the spanner, the only safe way out
            near = self.aim is not None and np.linalg.norm(here[:2] - self.aim) < 0.05
            return self._glide(skill, here[:2], max(self.carry_z, float(here[2])), self.yaw_place if near else self.yaw_pick,
                               0.0 if arm.holding() else 1.0, 0.015)
        return False, f"{skill} is not a skill of this task"

    def _glide(self, name: str, xy: Vec, z: float, yaw: float, grip: float, speed: float,
               may_stop: bool = False, **kw: Any) -> tuple[bool, str]:
        """A smooth move. One that stops (something in the way, no path) ends the run, rather than
        let the same move be chosen and tried again; only the slide down the spanner may stop."""
        self._note(name)
        self._moved = True
        ok, why = self.runner.glide(name, np.array([xy[0], xy[1], z], dtype=np.float64), yaw, grip, speed=speed,
                                    joint_speed_deg=10.0, **kw)
        if not ok and not may_stop:
            self.problem = f"{name} stopped: {why}"
        return ok, why

    def _align(self) -> tuple[bool, str]:
        """Down to just above the spanner's top, then look and correct, up to four times."""
        ok, why = self._glide("look", self.aim, self.look_z, self.yaw_place, 0.0, 0.006)
        if not ok:
            return ok, why
        for _ in range(4):
            err = self._look(self.looks)
            self.looks += 1
            if err is None:
                return False, self.problem
            self.error = float(np.linalg.norm(err))
            if self.error <= self.aligned_m:
                self.aligned = True
                return True, ""
            if self.error > 0.012:
                self.problem = f"the fitting is {1000 * self.error:.1f} mm off the spanner's axis: too far to trust"
                return False, self.problem
            if self.looks >= self.max_looks:
                self.problem = f"not lined up after {self.looks} looks: still {1000 * self.error:.1f} mm off"
                return False, self.problem
            self.aim = self.aim + err
            ok, why = self._glide("correct", self.aim, self.look_z, self.yaw_place, 0.0, 0.004)
            if not ok:
                return ok, why
        return True, ""

    def _look(self, i: int) -> Vec | None:
        """Where the held fitting really is against the spanner's axis, both from the same camera, so the
        calibration error mostly cancels. Returns the move that puts it on the axis."""
        runner, kin, arm = self.runner, self.runner.kin, self.runner.arm
        runner.hold(0.8, 0.0)  # settle, and let the camera catch up
        rgb, _ = self.cam.frames()
        q = arm.joints()
        tool = kin.tool_pose(q)[:3]
        Tf = self.fitting_pose
        T0 = np.eye(4)  # the fitting as the arm has it: turned with the gripper, its axis `shift` beyond the tool point
        T0[:3, :3] = _turn(self.yaw_place - self.yaw_pick) @ Tf[:3, :3]
        T0[:3, 3] = [*(tool[:2] + self.shift * _heading(self.yaw_place)), tool[2] - self.grip_z + Tf[2, 3]]
        h, w = rgb.shape[:2]
        fingers = np.zeros((h, w), np.uint8)  # what hides it: the fingers in front, and the spanner behind
        for c in self._finger_boxes(q, arm.gripper()):
            px = np.array([self.frame.to_pixel(p) for p in c], np.int32)
            cv2.fillConvexPoly(fingers, cv2.convexHull(px), 1)
        hidden = cv2.dilate(fingers, np.ones((7, 7), np.uint8)).astype(bool) | self._spanner_px
        expected = self.fitting.outline(T0, self.frame, (0, 0, w, h)).astype(bool)
        masks = [d.mask.astype(bool) for d in self.detector.detect(rgb, [self.phrase]) if d.mask is not None]
        if not masks:
            self.problem = "the camera sees no metal part over the spanner"
            return None
        free = ~hidden
        seen = max(masks, key=lambda mk: (mk & expected & free).sum() / max(1, ((mk | expected) & free).sum()))
        seen &= cv2.dilate(expected.astype(np.uint8), np.ones((75, 75), np.uint8)).astype(bool)  # within about 14 mm
        T, iou = self.fitting.locate_held(seen.astype(np.uint8), self.frame, T0, hidden)
        err = self.spanner_pose[:2, 3] - T[:2, 3]
        off = float(np.linalg.norm(T[:2, 3] - T0[:2, 3]))
        if self.rec is not None:
            view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            view[hidden] = (0.5 * view[hidden]).astype(np.uint8)
            fitted = self.fitting.outline(T, self.frame, (0, 0, w, h)).astype(bool)
            for mk, col in ((seen, (255, 120, 0)), (expected, (0, 200, 0)), (fitted, (0, 255, 255))):
                cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(view, cs, -1, col, 1)
            for pt, col in (([*self.spanner_pose[:2, 3], self.top], (0, 0, 255)), ([*T[:2, 3], T0[2, 3]], (0, 255, 255)),
                            ([*tool[:2], T0[2, 3]], (0, 200, 0))):
                u, v = self.frame.to_pixel(np.array(pt))
                cv2.drawMarker(view, (int(round(u)), int(round(v))), col, cv2.MARKER_CROSS, 10, 1)
            cv2.imwrite(str(self.rec.dir / f"look_{i}.jpg"), view)
        self._say(f"look {i}: outline match {iou:.2f}; the fitting is {1000 * off:.1f} mm from where the arm has it and "
                  f"{1000 * np.linalg.norm(err):.1f} mm off the spanner's axis ({1000 * err[0]:+.1f} x, {1000 * err[1]:+.1f} y)")
        if iou < 0.35 or off > 0.012:
            self.problem = "the camera cannot see the held fitting well enough to line it up"
            return None
        return err

    def _slide(self) -> tuple[bool, str]:
        """Down the spanner at 5 mm/s until something is close (the depth watch), something is in
        the way (the arm falls 3 mm behind), or the fitting's bottom reaches `floor_z`."""
        self._close_readings = 0
        ok, why = self._glide("slide_down", self.aim, self.floor_z + self.grip_z, self.yaw_place, 0.0, _CREEP,
                              may_stop=True, max_dev=0.003, should_stop=self._watch)
        self.bottom = float(self.runner.kin.tool_pose(self.runner.arm.joints())[2]) - self.grip_z
        self.on_spanner = self.bottom < self.top - self.onto_m
        down = (f"{1000 * (self.top - self.bottom):.0f} mm down the spanner" if self.bottom < self.top
                else f"still {1000 * (self.bottom - self.top):.0f} mm above the spanner's top")
        self._say(f"the fitting's bottom went down to {1000 * self.bottom:.0f} mm above the table, {down}")
        if not self.on_spanner:
            self.problem = "the fitting stopped at the spanner's top: it goes back where it was picked"
        return ok, why

    def _watch(self) -> str:
        """Stop the slide before the gripper touches the spanner's top: once the top is out of the fitting,
        two depth readings in a row closer than `gap_stop`."""
        bottom = float(self.runner.kin.tool_pose(self.runner.arm.joints())[2]) - self.grip_z
        if bottom > self.top - self.height - 0.005:
            return ""
        gap = self._gap()
        self._close_readings = self._close_readings + 1 if gap < self.gap_stop else 0
        return f"the camera sees the gripper {1000 * gap:.0f} mm above the spanner's top" if self._close_readings >= 2 else ""

    def _gap(self) -> float:
        """How far above the spanner's top the lowest thing over it is, from the depth camera: points
        within 12 mm of its axis, 3 to 50 mm above its top."""
        _, z = self.cam.frames()
        cam, sxy = self.frame.cam, self.spanner_pose[:2, 3]
        su, sv = (int(v) for v in self.frame.to_pixel(np.array([*sxy, self.top])))
        y0, y1, x0, x1 = max(0, sv - 120), min(z.shape[0], sv + 40), max(0, su - 60), min(z.shape[1], su + 60)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        zz = z[y0:y1, x0:x1]
        ok = zz > 0.1
        pc = np.stack([(xs[ok] - cam.cx) / cam.fx * zz[ok], (ys[ok] - cam.cy) / cam.fy * zz[ok], zz[ok]], axis=1)
        pb = pc @ self.frame.rot.T + self.frame.pos
        near = ((np.linalg.norm(pb[:, :2] - sxy, axis=1) <= 0.012) & (pb[:, 2] > self.top + 0.003)
                & (pb[:, 2] < self.top + 0.050))
        return float(np.percentile(pb[near, 2], 5) - self.top) if near.sum() >= 20 else float("inf")

    def _note(self, text: str) -> None:
        if self.rec is not None:
            self.rec.note(text)

    def _say(self, text: str) -> None:
        print("   " + text)
        self._note(text)

    # --- before folding --------------------------------------------------------------
    def finish(self, runner: Any = None, arm: Any = None) -> None:
        """Leave the gripper empty and the arm clear: straight up first, off the spanner or away from
        its top, and a fitting still held goes back where it was picked."""
        runner = self.runner
        arm, kin = runner.arm, runner.kin
        runner.gov.reset()
        if not self._moved:
            return  # the arm never left home
        held = arm.holding()
        here = kin.tool_pose(arm.joints())[:3]
        yaw = runner._last[1] if runner._last is not None else self.yaw_pick
        near = self.aim is not None and float(np.linalg.norm(here[:2] - self.aim)) < 0.06
        up = self.carry_z if (near or held) else self.up_z
        if here[2] < up - 0.005:
            runner.glide("up", np.array([here[0], here[1], up]), yaw, 0.0 if held else arm.gripper(),
                         speed=_CREEP if near and held else (0.01 if near else _FAST))
        if held and self.grasp is not None:  # never fold holding it
            g = self.grasp[:2]
            runner.glide("carry back", np.array([*g, self.carry_z]), self.yaw_pick, 0.0, speed=_FAST)
            runner.glide("lower", np.array([*g, self.grip_z + 0.03]), self.yaw_pick, 0.0, speed=_FAST)
            if runner.glide("set down", np.array([*g, self.grip_z + 0.001]), self.yaw_pick, 0.0, speed=_SLOW)[0]:
                runner.hold(1.0, 1.0)
                runner.glide("clear", np.array([*g, self.up_z]), self.yaw_pick, 1.0, speed=_FAST)
        runner.glide("clear", np.array([0.25, 0.0, 0.19]), 0.0, 0.0 if arm.holding() else arm.gripper(),
                     joint_space=True, joint_speed_deg=10.0)
        if arm.holding():
            print("   still holding the fitting: it could not be set down")
