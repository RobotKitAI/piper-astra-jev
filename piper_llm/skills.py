"""Skills the model chooses from, and the loop that runs them.

Each skill is a bounded motion written in code. The model only picks which one
runs next; it never sends joint angles.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
import numpy.typing as npt

from piper_llm.arm import Arm
from piper_llm.config import JOINT_HIGH, JOINT_LOW, SceneConfig
from piper_llm.kinematics import Kinematics
from piper_llm.safety import Governor

Vec = npt.NDArray[np.float64]

PICK_PLACE_SKILLS: dict[str, str] = {
    "approach_target": "Move the open gripper above the target object. "
                       "Only when over_target is false and object_placed is false.",
    "descend_to_target": "Lower the open gripper onto the target, ready to grasp. Only when "
                         "over_target is true, at_grasp_height is false, gripper_closed is false "
                         "and object_placed is false.",
    "close_gripper": "Close the gripper to grasp what is between the fingers. Only when "
                     "over_target is true, at_grasp_height is true, gripper_closed is false and "
                     "object_placed is false.",
    "rotate_grasp": "Open and turn the gripper 45 degrees to retry a grasp. "
                    "Only when fingers_jammed is true.",
    "lift": "Raise the held object to the carry height. Only when object_held is true, "
            "at_carry_height is false and over_place_target is false.",
    "move_over_place": "Carry the held object above the place target. Only when object_held "
                       "is true, at_carry_height is true and over_place_target is false.",
    "lower_to_place": "Lower the held object into the place target. Only when object_held is "
                      "true, over_place_target is true and at_release_height is false.",
    "open_gripper": "Open the gripper to release the object. Only when object_held is true, "
                    "over_place_target is true and at_release_height is true; or when "
                    "gripper_closed is true and object_held is false, after a missed grasp.",
    "retreat": "Raise the open gripper clear of the scene. "
               "Only when object_placed is true and at_clear_height is false.",
    "done": "The task is complete: stop. Only when task_complete is true.",
}

# Run 8: the pipe fitting slid down over a standing box spanner (task.SlideOver).
SLIDE_OVER_SKILLS: dict[str, str] = {
    "approach_target": "Move the open gripper high above the fitting. Only when over_target is false, "
                       "object_held is false and object_placed is false.",
    "descend_to_target": "Lower the open gripper around the fitting, ready to grasp. Only when over_target "
                         "is true, at_grasp_height is false, gripper_closed is false and object_placed is false.",
    "close_gripper": "Close the gripper on the fitting. Only when over_target is true, at_grasp_height is "
                     "true, gripper_closed is false and object_placed is false.",
    "lift": "Raise the held fitting to the carry height, above the spanner's top. Only when object_held is "
            "true, at_carry_height is false and over_place_target is false.",
    "move_over_spanner": "Carry the held fitting over the spanner and lower it to just above the spanner's "
                         "top. Only when object_held is true, at_carry_height is true and over_place_target "
                         "is false.",
    "align_over_spanner": "Measure with the camera where the held fitting is against the spanner's axis and "
                          "move it onto the axis. Only when object_held is true, over_place_target is true "
                          "and aligned is false.",
    "slide_down": "Lower the fitting slowly down over the spanner, until the gripper nears the spanner's "
                  "top. Only when object_held is true, aligned is true and on_spanner is false.",
    "let_go": "Open the gripper so the fitting slides down the spanner to the table. Only when "
              "object_held is true and on_spanner is true.",
    "retreat": "Raise the open gripper clear of the spanner. Only when object_placed is true and "
               "at_clear_height is false.",
    "done": "The task is complete: stop. Only when task_complete is true.",
}

APPROACH_SKILLS: dict[str, str] = {
    "look": "Park the arm aside and take a fresh picture of the part.",
    "approach_hole": "Move the tool tip 4 cm above the selected bolt hole.",
    "align": "Hold position and re-measure the hole before descending.",
    "descend_to_hole": "Lower the tool tip to 5 mm above the hole mouth.",
    "next_hole": "Select the next bolt hole.",
    "retreat": "Raise the tool clear of the part.",
    "done": "All selected holes were approached: stop.",
}


def grasp_from_mask(det: Any, depth: Any, frame: Any, scene: SceneConfig,
                    whole: Any = None) -> tuple[Vec, float, float] | None:
    """Plan a grasp on a segmented object: (top point xyz, yaw, width in m).

    The mask says where the object is, the depth says what is solid: pixels
    standing at least `solid_rise` above the table. A ring's hole and the
    table seen through it fall away. The grasp goes to the thickest solid
    part whose local width fits between grasp_width_min and grasp_width_max,
    and the yaw follows the part's axis, so the fingers close across it. With
    `whole` (a detection of the whole object, mask included) that axis runs
    from the object's centre out to the grasp point; without it, from the
    shape of the part. None when nothing solid of a graspable width is found.
    """
    mask = np.asarray(det.mask, dtype=np.uint8)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    z = depth[ys, xs].astype(np.float64)
    valid = (z > 0.05) & (z < 3.0)
    cam = frame.cam
    pts = np.stack([(xs - cam.cx) * z / cam.fx, (ys - cam.cy) * z / cam.fy, z], axis=1)
    height = (pts @ frame.rot.T + frame.pos)[:, 2]
    solid = np.zeros_like(mask)
    keep = valid & (height > scene.solid_rise)
    solid[ys[keep], xs[keep]] = 1
    if solid.sum() < 50:
        return None
    m_per_px = float(np.median(z[keep])) / cam.fx
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    width = 2.0 * dist * m_per_px
    cand = (width >= scene.grasp_width_min) & (width <= scene.grasp_width_max)
    if not cand.any():
        return None
    iy, ix = np.unravel_index(np.argmax(np.where(cand, dist, 0.0)), dist.shape)
    # axis of the part: the connected run of graspable-width pixels around the
    # grasp point (the whole tube of a handle, not a small disc on it)
    part = cand.astype(np.uint8)
    n_lab, labels = cv2.connectedComponents(part)
    py, px = np.nonzero(labels == labels[iy, ix])
    local = np.c_[px, py].astype(np.float64)
    local -= local.mean(0)
    axis_px = np.linalg.svd(local, full_matrices=False)[2][0]
    z0 = float(np.median(z[keep]))
    a = np.array([(ix - cam.cx) * z0 / cam.fx, (iy - cam.cy) * z0 / cam.fy, z0])
    b = np.array([(ix + 20 * axis_px[0] - cam.cx) * z0 / cam.fx,
                  (iy + 20 * axis_px[1] - cam.cy) * z0 / cam.fy, z0])
    d = frame.rot @ (b - a)
    yaw = math.atan2(d[1], d[0])
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2  # a parallel gripper: fold to +-90 deg
    top = frame.to_base(float(ix), float(iy), depth)
    if top is None:
        return None
    # the part's crest, not one pixel's reading: a thin bar is missed otherwise
    crest = float(np.percentile(height[keep][labels[ys[keep], xs[keep]] == labels[iy, ix]], 90))
    top = np.array([top[0], top[1], max(float(top[2]), crest)])
    if whole is not None and getattr(whole, "mask", None) is not None:
        wy, wx = np.nonzero(np.asarray(whole.mask, dtype=np.uint8))
        wz = depth[wy, wx].astype(np.float64)
        ok = (wz > 0.05) & (wz < 3.0)
        if ok.sum() > 50:
            wp = np.stack([(wx - cam.cx) * wz / cam.fx, (wy - cam.cy) * wz / cam.fy, wz], axis=1)[ok]
            centre = (wp @ frame.rot.T + frame.pos)[:, :2].mean(0)
            v = np.asarray(top)[:2] - centre
            if np.linalg.norm(v) > 0.02:
                yaw = (math.atan2(v[1], v[0]) + math.pi / 2) % math.pi - math.pi / 2
    return np.asarray(top), float(yaw), float(width[iy, ix])


def grasp_search(det: Any, depth: Any, frame: Any, scene: SceneConfig, cell: float = 0.002,
                 max_open: float = 0.060, pad: float = 0.012, margin: float = 0.004,
                 tip_below: float = 0.015, contact_min: float = 0.012
                 ) -> tuple[Vec, float, float] | None:
    """Search a top-down grasp on a height map: (tool point xyz, yaw, width).

    The depth around the part becomes a height map in the base frame. A
    candidate is a point on the part, a finger direction and a tool height.
    The finger tips sit `tip_below` under the tool point; a candidate counts
    when the material between the fingers above the tips is a graspable width
    and reaches at least `contact_min` above the tips and belongs to the part's
    own mask, and both finger landing
    zones beside it are lower than the tips. For each point and direction the
    deepest such height is taken, so a bar over a hole is held across the bar
    and a grasp along it fails because the fingers would land on the loop.
    The best candidate has the deepest free landing zones and the highest
    material. None when no line qualifies.
    """
    mask = np.asarray(det.mask, dtype=np.uint8)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    cam = frame.cam
    x0, x1 = max(0, xs.min() - 160), min(mask.shape[1], xs.max() + 160)
    y0, y1 = max(0, ys.min() - 160), min(mask.shape[0], ys.max() + 160)
    gy, gx = np.mgrid[y0:y1, x0:x1]
    z = depth[y0:y1, x0:x1].astype(np.float64)
    ok = (z > 0.05) & (z < 3.0)
    pts = np.stack([(gx - cam.cx) * z / cam.fx, (gy - cam.cy) * z / cam.fy, z], axis=-1)[ok]
    base = pts @ frame.rot.T + frame.pos
    part_px = mask[y0:y1, x0:x1][ok].astype(bool)
    bx0, by0 = base[:, 0].min(), base[:, 1].min()
    nx = int((base[:, 0].max() - bx0) / cell) + 2
    ny = int((base[:, 1].max() - by0) / cell) + 2
    ix = ((base[:, 0] - bx0) / cell).astype(int)
    iy = ((base[:, 1] - by0) / cell).astype(int)
    hmap = np.full((ny, nx), -np.inf)
    np.maximum.at(hmap, (iy, ix), base[:, 2])
    hmap[np.isneginf(hmap)] = np.inf  # unseen cells count as blocked
    part = np.zeros((ny, nx), bool)
    part[iy[part_px], ix[part_px]] = True
    part = cv2.morphologyEx(part.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8)).astype(bool)  # depth speckle leaves holes
    near_part = cv2.dilate(part.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)  # 1 cm smear band
    # a raised part hides the table behind it from the camera: unseen cells
    # within 2 cm of the part are its shadow on the table, not an obstacle
    shadow = cv2.dilate(part.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool) & np.isinf(hmap)
    hmap[shadow] = 0.0
    solid = part & (hmap > scene.solid_rise) & np.isfinite(hmap)
    cy, cx = np.nonzero(solid)
    if cx.size == 0:
        return None
    sel = np.arange(cx.size)[::3]
    cy, cx = cy[sel], cx[sel]
    # the part's centre from the image mask, which is sharp, placed at the top
    # face's height: depth smears the far edge of an object along the line of
    # sight, so a centroid of the projected cells would sit off the object
    mx, my = float(xs.mean()), float(ys.mean())
    z_top = float(np.nanmedian(np.where(mask[ys, xs] > 0, depth[ys, xs], np.nan)))
    ctr = frame.rot @ np.array([(mx - cam.cx) * z_top / cam.fx, (my - cam.cy) * z_top / cam.fy, z_top]) + frame.pos
    pcx, pcy = (ctr[0] - bx0) / cell, (ctr[1] - by0) / cell
    # the silhouette's axes in the base frame: fingers square to the object
    # hold better than fingers on its corners
    foot = frame.rot @ np.stack([(xs - cam.cx) * z_top / cam.fx, (ys - cam.cy) * z_top / cam.fy,
                                 np.full(xs.size, z_top)], axis=0) + frame.pos[:, None]
    axis_deg = cv2.minAreaRect((foot[:2].T * 1000).astype(np.float32))[2]
    reach = int(max_open / cell)
    steps = np.arange(1, reach + 1)
    lateral = np.array([-0.008, 0.0, 0.008]) / cell
    r = int(0.01 / cell)
    best = None
    for deg in range(0, 180, 10):
        th = math.radians(deg)
        c, s_ = math.cos(th), math.sin(th)
        for k in range(cx.size):
            px, py = cx[k], cy[k]
            win_p = part[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1]
            win_h = hmap[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1]
            crest = float(np.max(np.where(win_p & np.isfinite(win_h), win_h, -np.inf)))
            if not np.isfinite(crest):
                continue
            # the material between the fingers is the part's own mask along the
            # line (sharp from the image), read outward until two cells of gap
            lines, ext = [], []
            for sign in (1, -1):
                xx = np.clip((px + sign * steps * c).round().astype(int), 0, nx - 1)
                yy = np.clip((py + sign * steps * s_).round().astype(int), 0, ny - 1)
                lines.append((sign, hmap[yy, xx]))
                gaps = np.nonzero(~part[yy, xx])[0]
                e = reach
                for g in gaps:
                    if g + 1 < steps.size and not part[yy[g + 1], xx[g + 1]]:
                        e = g
                        break
                ext.append(e + 1)
            width = (ext[0] + ext[1]) * cell
            if width < scene.grasp_width_min or width > max_open - 2 * margin:
                continue
            # tool heights from deep to shallow; the first feasible one wins
            z_lo = max(scene.grasp_z, crest - scene.pad_depth)
            z_hi = crest + tip_below - contact_min
            for z_tool in np.arange(z_lo, z_hi + 1e-9, 0.005):
                tips = z_tool - tip_below
                clear = np.inf
                # what stands between the fingers must be the part itself, not a
                # neighbour touching it (a crate wall beside a handle). The depth
                # image smears the part's own sides a centimetre outside its mask,
                # never above its crest: that is tolerated within the smear band.
                foreign = 0
                for sign, e in ((1, ext[0]), (-1, ext[1])):
                    xx = np.clip((px + sign * steps[:e] * c).round().astype(int), 0, nx - 1)
                    yy = np.clip((py + sign * steps[:e] * s_).round().astype(int), 0, ny - 1)
                    h_line = hmap[yy, xx]
                    outside = ~part[yy, xx]
                    foreign += int(np.sum(outside & (h_line > crest + margin)))
                    foreign += int(np.sum(outside & ~near_part[yy, xx] & (h_line > tips + margin)))
                if foreign * cell > 0.004:
                    continue
                for (sign, h), e in zip(lines, ext):
                    land = np.arange(e + int(margin / cell), e + int((margin + pad) / cell) + 1)
                    for lat in lateral:
                        xx = np.clip((px + sign * land * c - lat * s_).round().astype(int), 0, nx - 1)
                        yy = np.clip((py + sign * land * s_ + lat * c).round().astype(int), 0, ny - 1)
                        hz = hmap[yy, xx].copy()
                        # the part's own smear beside it (within 1 cm, no higher
                        # than its crest) is not an obstacle for the finger
                        smear = near_part[yy, xx] & (hz <= crest + margin)
                        hz[smear] = np.minimum(hz[smear], tips - margin)
                        clear = min(clear, float(np.min(tips - hz)))
                if clear < margin:
                    continue
                # prefer: deep clearance, high material, the point centred between the
                # two material edges, the narrowest way across, a deep pad height
                # prefer: near the part's centre of mass, centred between the two
                # material edges, the narrowest line there, deep clearance, high
                # material, a deep pad height
                centred = abs(ext[0] - ext[1]) * cell
                off_centre = math.hypot(px - pcx, py - pcy) * cell
                off_axis = min(abs((deg - axis_deg + 45) % 90 - 45), 45.0)  # deg from a silhouette axis
                score = (min(clear, 0.02) + 0.5 * crest - 5.0 * off_centre - 2.0 * centred
                         - 3.0 * width - 0.002 * off_axis - 0.5 * (z_tool - z_lo))
                if best is None or score > best[0]:
                    best = (score, px, py, th, width, z_tool)
                break
    if best is None:
        return None
    _, px, py, th, width, z_tool = best
    x, y = bx0 + (px + 0.5) * cell, by0 + (py + 0.5) * cell
    yaw = (th - math.pi / 2 + math.pi / 2) % math.pi - math.pi / 2  # fingers close along Rz(yaw) y
    return np.array([x, y, z_tool]), float(yaw), float(width)


def whole_detection(detector: Any, rgb: Any, part: Any, scene: SceneConfig) -> Any:
    """The best detection of the whole object the part belongs to, from the
    scene's target_whole phrases: its box must hold at least 80% of the part's
    box. A neighbour that merely overlaps the part (a tray beside a handle)
    does not qualify."""
    phrases = (scene.target_whole,) if isinstance(scene.target_whole, str) else tuple(scene.target_whole)
    px0, py0, px1, py1 = part.box
    part_area = max(1.0, (px1 - px0) * (py1 - py0))
    best = None
    for phrase in phrases:
        d = detector.best(rgb, phrase, max_px=900)
        if d is None:
            continue
        x0, y0, x1, y1 = d.box
        inside = max(0.0, min(x1, px1) - max(x0, px0)) * max(0.0, min(y1, py1) - max(y0, py0))
        if inside >= 0.8 * part_area and (best is None or d.score > best.score):
            best = d
    return best


def plan_grasp(det: Any, rgb: Any, depth: Any, frame: Any, scene: SceneConfig,
               detector: Any) -> tuple[Vec, float] | None:
    """Where to close the gripper on one detection of the target: (tool point,
    yaw). A mask gets the grasp search on mask and depth, then the thickest-part
    planner; a box gets its centre with the fingers along the base y axis. None
    when the sighting is not the target (a sliver seen on the gripper)."""
    if det.mask is not None:
        grasp = grasp_search(det, depth, frame, scene)
        if grasp is None:
            g = grasp_from_mask(det, depth, frame, scene, whole_detection(detector, rgb, det, scene))
            if g is not None:
                grasp = (np.array([g[0][0], g[0][1], max(scene.grasp_z, g[0][2] - scene.pad_depth)]),
                         g[1], g[2])
        if grasp is not None:
            return np.asarray(grasp[0], dtype=np.float64), float(grasp[1])
    at = frame.to_base(det.centre[0], det.centre[1], depth)
    if at is None or at[2] > scene.target_max_z:
        return None
    return np.array([at[0], at[1], max(scene.grasp_z, float(at[2]) - scene.pad_depth)]), 0.0


@dataclass
class Runner:
    """Executes one skill at a time, with the governor underneath."""

    arm: Arm
    kin: Kinematics
    gov: Governor
    scene: SceneConfig = field(default_factory=SceneConfig)
    on_step: Callable[[dict[str, Any]], None] | None = None
    reach_tol_m: float = 0.005  # the real arm settles a few mm off a target
    stall_steps: int = 15  # no movement over this many steps means it has settled
    leash_m: float = 0.015  # the command waits when the arm is further behind than this
    max_tilt_rad: float = math.radians(30.0)  # tool lean allowed when the wrist cannot go vertical
    max_reach_tilt_rad: float = math.radians(60.0)  # lean allowed to reach a high carry
    yaw_step_rad: float = math.radians(5.0)  # the tool turns at most this much per step
    fixed_lean: float | None = None  # hold this lean (rad) about the fingers' line instead of the smallest that reaches
    _last: tuple[Vec, float] | None = None  # (joints, yaw) last commanded, for the next move
    _lean: float = 0.0  # lean of the last reach solution, tried first next time

    def move_tool_to(self, skill: str, target_xyz: Vec, yaw: float, gripper: float,
                     clearance: Callable[[Vec], float] | None = None,
                     max_steps: int = 600) -> tuple[bool, str]:
        """Move the tool point to a base-frame target in small steps.

        The command runs ahead of the arm as a smooth trajectory: each step is
        solved from the previous command, not from the measured joints, which
        lag and tilt mid-move. The command only advances while the arm is
        within `leash_m` of it. Returns (reached, reason); the reason is empty
        when the motion finished.
        """
        target = np.asarray(target_xyz, dtype=np.float64)
        ok, why = self.gov.check_target(skill, target)
        if not ok:
            return False, why
        g_cmd = self.arm.gripper()
        q_cmd = self.arm.joints()
        start = self.kin.tool_pose(q_cmd)
        cmd = start[:3]
        yaw_cmd = float(start[3])  # the tool turns gradually toward the requested yaw
        if self._last is not None and np.degrees(np.abs(q_cmd - self._last[0]).max()) < 2.0:
            yaw_cmd = self._last[1]  # a leaned tool reads its yaw off: keep the commanded one
        axis = self.kin.data.site_xmat[self.kin.site].reshape(3, 3)[:, 2]
        self._lean = float(math.acos(float(np.clip(-axis[2], -1.0, 1.0))))  # the lean the arm has now
        yaw = self.reachable_yaw(target, yaw, q_cmd, prefer=yaw_cmd)
        recent: list[Vec] = []
        for _ in range(max_steps):
            q_now = self.arm.joints()
            pose = self.kin.tool_pose(q_now)
            gap = float(np.linalg.norm(target - pose[:3]))
            turn = (yaw - yaw_cmd + math.pi) % (2 * math.pi) - math.pi
            behind = float(np.degrees(np.abs(q_now - q_cmd).max()))  # the wrist lags a turn at low speed
            if gap <= self.reach_tol_m and abs(g_cmd - gripper) < 1e-3 and abs(turn) < 1e-6 and behind < 2.0:
                return True, ""
            recent.append(q_now.copy())
            commanded = bool(np.allclose(cmd, target) and abs(turn) < 1e-6)  # nothing left to command
            if commanded and len(recent) > self.stall_steps and np.degrees(
                    np.abs(recent[-1] - recent[-1 - self.stall_steps]).max()) < 0.2:
                return False, f"arm settled {1000 * gap:.0f} mm and {behind:.0f} deg short of the target"
            step = self.gov.step_size(
                skill, clearance(pose[:3]) if clearance else max(0.0, pose[2]))
            if float(np.linalg.norm(cmd - pose[:3])) < self.leash_m:
                remaining = float(np.linalg.norm(target - cmd))
                cmd = target if remaining <= step else cmd + (target - cmd) * step / remaining
                yaw_cmd += float(np.clip(turn, -self.yaw_step_rad, self.yaw_step_rad))
            q_new, err = self.solve(cmd, yaw_cmd, q_cmd)
            if err > 0.003:
                return False, f"no IK solution for {np.round(cmd, 3).tolist()}"
            over = np.where((q_new < JOINT_LOW) | (q_new > JOINT_HIGH))[0]
            if over.size:  # the arm would clamp it and go somewhere else
                j = int(over[0])
                return False, (f"joint {j + 1} would need {np.degrees(q_new[j]):.0f} deg at "
                               f"{np.round(cmd, 3).tolist()}, beyond its limit")
            ok, why = self.gov.check_joint_step(skill, q_new, q_cmd)
            if not ok:
                return False, why
            ok, why = self.gov.check_first_action(skill, cmd, pose[:3])
            if not ok:
                return False, why
            q_cmd = q_new
            self._last = (q_cmd, yaw_cmd)
            g_cmd = float(np.clip(gripper, g_cmd - 0.1, g_cmd + 0.1))
            self.arm.command_joints(q_cmd, g_cmd)
            time.sleep(1.0 / self.arm.cfg.control_hz)
            fault = self.arm.status_error()
            if fault:
                self.arm.stop()
                return False, fault
            ok, why = self.gov.check_deviation(skill, cmd, self.kin.tool_pose(
                self.arm.joints())[:3])
            if not ok:
                self.arm.stop()
                return False, why
            if self.on_step:
                self.on_step({"skill": skill, "target": np.round(target, 4).tolist()})
        return False, f"motion did not finish within {max_steps} steps"

    def glide(self, skill: str, target_xyz: Vec, yaw: float, gripper: float, speed: float = 0.015,
              max_dev: float = 0.015, rate_hz: float = 50.0, settle_s: float = 3.0,
              joint_space: bool = False, joint_speed_deg: float = 8.0,
              should_stop: Callable[[], str] | None = None) -> tuple[bool, str]:
        """Move the tool point in a straight line to the target, smoothly and slowly.

        The whole path is solved before the arm moves: IK every millimetre,
        with the yaw and the lean (`fixed_lean`, or vertical) eased along it.
        With `joint_space`, for moves through open space, the joints are
        blended straight from where they are to the target's solution instead:
        the tool's path is then a curve, and only the target has to be inside
        the joint limits (the home pose sits a hair past joint 5's).
        It is then streamed at `rate_hz` with minimum-jerk timing at about
        `speed` m/s, so the arm starts, cruises and stops without jolts. A
        tool point more than `max_dev` behind its command means something
        is in the way: the arm stops there. `should_stop`, checked ten times a
        second, can stop it too, by returning a reason (a camera watching a
        gap close, say). Returns (reached, reason)."""
        target = np.asarray(target_xyz, dtype=np.float64)
        ok, why = self.gov.check_target(skill, target)
        if not ok:
            return False, why
        q_now = self.arm.joints()
        near_last = self._last is not None and np.degrees(np.abs(q_now - self._last[0]).max()) < 2.0
        q0 = self._last[0].copy() if near_last else q_now
        pose0 = self.kin.tool_pose(q0)
        p0 = pose0[:3].copy()
        axis = self.kin.data.site_xmat[self.kin.site].reshape(3, 3)[:, 2]
        lean0 = float(math.acos(float(np.clip(-axis[2], -1.0, 1.0))))
        lean1 = self.fixed_lean if self.fixed_lean is not None else 0.0
        yaw0 = self._last[1] if near_last else float(pose0[3])
        turn = (yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
        dist = float(np.linalg.norm(target - p0))
        if joint_space:
            q1, err = self.kin.ik(target, yaw, q0, iters=400, tilt=lean1, about_fingers=self.fixed_lean is not None)
            if err > 0.002 or np.any(q1 < JOINT_LOW) or np.any(q1 > JOINT_HIGH):
                return False, f"no pose for {np.round(target, 3).tolist()}"
            n = max(2, int(math.ceil(float(np.degrees(np.abs(q1 - q0).max())) / 0.25)))
            path = q0 + (q1 - q0) * np.linspace(0.0, 1.0, n + 1)[:, None]
            joint_s = float(np.degrees(np.abs(q1 - q0).max())) / joint_speed_deg  # the average; the peak is 1.9 times it
            dist = max(dist, joint_s * speed)  # the timing below takes the slower of the two
            n = -1  # the path is ready
        else:
            n = max(2, int(math.ceil(max(dist / 0.001, abs(turn) / math.radians(0.5), abs(lean1 - lean0) / math.radians(0.25)))))
            path = [q0]
        q = q0
        for i in range(1, n + 1):
            f = i / n
            q, err = self.kin.ik(p0 + (target - p0) * f, yaw0 + turn * f, q, iters=200,
                                 tilt=lean0 + (lean1 - lean0) * f, about_fingers=self.fixed_lean is not None)
            if err > 0.002 or np.any(q < JOINT_LOW) or np.any(q > JOINT_HIGH):
                return False, f"no smooth path: {np.round(p0 + (target - p0) * f, 3).tolist()} is out of reach"
            if np.degrees(np.abs(q - path[-1]).max()) > 2.0:
                return False, f"no smooth path: the joints jump near {np.round(p0 + (target - p0) * f, 3).tolist()}"
            path.append(q)
        path = np.array(path)
        duration = max(1.0, dist / max(speed, 1e-4) * 1.875, abs(turn) / math.radians(10.0) * 1.875)  # 1.875: peak of minimum jerk
        t0 = time.perf_counter()
        tick = 0
        while True:
            tau = min(1.0, (time.perf_counter() - t0) / duration)
            s = tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)
            x = s * (len(path) - 1)
            i = min(int(x), len(path) - 2)
            q_cmd = path[i] + (path[i + 1] - path[i]) * (x - i)
            self.arm.command_joints(q_cmd, gripper)
            self._last = (q_cmd, yaw0 + turn * s)
            if tick % 5 == 0:
                fault = self.arm.status_error()
                if fault:
                    self.arm.stop()
                    return False, fault
                lag = float(np.linalg.norm(self.kin.tool_pose(self.arm.joints())[:3] - self.kin.tool_pose(q_cmd)[:3]))
                if lag > max_dev:
                    self.arm.command_joints(self.arm.joints(), gripper)  # stay where it is
                    self._last = None
                    return False, f"the arm fell {1000 * lag:.0f} mm behind: something is in the way"
                reason = should_stop() if should_stop else ""
                if reason:
                    self.arm.command_joints(self.arm.joints(), gripper)
                    self._last = None
                    return False, reason
                if self.on_step:
                    self.on_step({"skill": skill, "target": np.round(target, 4).tolist()})
            if tau >= 1.0:
                break
            tick += 1
            time.sleep(max(0.0, t0 + tick / rate_hz - time.perf_counter()))
        self._lean = lean1
        end = time.time() + settle_s
        while time.time() < end:  # let it settle on the last command
            if np.degrees(np.abs(self.arm.joints() - path[-1]).max()) < 0.5:
                break
            self.arm.command_joints(path[-1], gripper)
            time.sleep(1.0 / rate_hz)
        return True, ""

    def reachable_yaw(self, target: Vec, yaw: float, seed: Vec, prefer: float | None = None) -> float:
        """A parallel gripper turned half a turn is the same grasp: of `yaw` and
        `yaw` + 180 degrees, the one the arm can hold at `target` with its wrist
        inside the joint limits, the smaller turn from `prefer` first. Not with
        `fixed_lean`: that lean points the tips along the yaw, so half a turn
        would lean them the other way."""
        if self.fixed_lean is not None:
            return float((yaw + math.pi) % (2 * math.pi) - math.pi)
        kept = self._lean
        best: tuple[float, float] | None = None
        try:
            for cand in (yaw, yaw + math.pi):
                q, err = self.solve(np.asarray(target, dtype=np.float64), cand, seed)
                if err > 0.003 or np.any(q < JOINT_LOW) or np.any(q > JOINT_HIGH):
                    continue
                turn = 0.0 if prefer is None else abs((cand - prefer + math.pi) % (2 * math.pi) - math.pi)
                margin = min(q[5] - JOINT_LOW[5], JOINT_HIGH[5] - q[5])
                cost = turn + (0.0 if margin > math.radians(15.0) else math.pi / 2)  # near the limit, a path can cross it
                if best is None or cost < best[0]:
                    best = (cost, float((cand + math.pi) % (2 * math.pi) - math.pi))
        finally:
            self._lean = kept
        return yaw if best is None else best[1]

    def solve(self, target: Vec, yaw: float, seed: Vec) -> tuple[Vec, float]:
        """IK with the tool vertical, or leaned away from the base just enough
        for joint 5 to stay inside its limit, up to max_tilt_rad; or, with
        `fixed_lean` set, at that lean whatever the target. A target out
        of reach that way (a high carry) gets more lean, up to
        max_reach_tilt_rad: a hanging object does not mind how the gripper is
        angled, only where the fingers are. The lean changes by a degree or two
        per waypoint, so the wrist never jumps."""
        def fits(q: Vec, err: float) -> bool:
            return err <= 0.003 and bool(np.all(q >= JOINT_LOW) and np.all(q <= JOINT_HIGH))

        if self.fixed_lean is not None:  # a set lean, reached a degree or two per waypoint
            two = math.radians(2.0)
            self._lean += float(np.clip(self.fixed_lean - self._lean, -two, two))
            return self.kin.ik(target, yaw, seed, iters=200, tilt=self._lean, about_fingers=True)
        q, err = self.kin.ik(target, yaw, seed)
        tilt = 0.0
        excess = q[4] - JOINT_HIGH[4]
        if excess > 0:
            tilt = min(excess + math.radians(1.0), self.max_tilt_rad)
            q, err = self.kin.ik(target, yaw, seed, tilt=tilt)
        one = math.radians(1.0)
        if fits(q, err) and tilt >= self._lean - 2 * one:
            self._lean = tilt
            return q, err
        start = max(self._lean, tilt)
        for lean in [start - 2 * one, start - one, start] + [start + i * one for i in range(1, 61)]:
            if lean <= 0.0 or lean > self.max_reach_tilt_rad + 1e-9:
                continue
            q2, e2 = self.kin.ik(target, yaw, seed, iters=200, tilt=float(lean))
            if fits(q2, e2):
                self._lean = float(lean)
                return q2, e2
        return q, err

    def hold(self, seconds: float, gripper: float) -> None:
        """Hold the pose with a gripper command, e.g. while closing."""
        q = self.arm.joints()
        end = time.time() + seconds
        g = self.arm.gripper()
        while time.time() < end:
            g = float(np.clip(gripper, g - 0.1, g + 0.1))
            self.arm.command_joints(q, g)
            time.sleep(1.0 / self.arm.cfg.control_hz)
