"""Hand-eye calibration: camera frame to robot base frame.

Runs with the motors off; the arm is moved by hand and nothing is commanded.

1. Put six or more small markers on the table, spread over the camera's view,
   a few of them on a block if one is handy. Keep the arm out of the view and
   press Enter. One frame is taken and shown: click every marker centre, then
   press Enter in the window (u undoes the last click, Esc aborts).
2. The window stays open with the numbered markers, the current one in green.
   For each marker, in click order, move the arm by hand so the closed finger
   tips touch the marker centre, hold still and press Enter in the window
   (s skips a marker). The joints are read and the tip position comes from
   the model.
3. The fit finds the camera transform and, with five or more markers, the zero
   correction of joint 3, which the arm reports a few degrees off the model.
   Markers with a residual over 8 mm are shown for re-touching until the
   maximum is under 10 mm or you press Esc to accept. calibration.json is
   written; camera.Frame and Arm read it.

    python calibrate.py

Point pairs and residuals are also written to out/calibration_points.json.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from piper_llm.camera import RealSense, save_calibration
from piper_llm.config import ArmConfig
from piper_llm.kinematics import TIP_OFFSET_M, Kinematics

OUT = Path(__file__).resolve().parent / "out"
WINDOW = "calibration"
GOOD_M = 0.010  # accept when every residual is under this
RETOUCH_M = 0.008  # ask to re-touch markers above this


def fit_transform(cam_points: np.ndarray, base_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform from camera points to base points (Kabsch)."""
    c_mean, b_mean = cam_points.mean(0), base_points.mean(0)
    h = (cam_points - c_mean).T @ (base_points - b_mean)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, b_mean - rot @ c_mean


def fit(records: list[dict], kin: Kinematics) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform plus joint 3 zero correction (grid search, fresh each call).

    Returns (joint_offsets, rotation, translation, residual per record) and
    updates each record's base point and residual.
    """
    cam = np.array([r["cam"] for r in records])
    joints = np.array([r["joints"] for r in records])
    degrees = np.arange(-12.0, 12.01, 0.1) if len(records) >= 5 else np.array([0.0])
    best: tuple | None = None
    for deg in degrees:
        off = np.zeros(6)
        off[2] = math.radians(deg)
        base = np.array([kin.tool_point(q + off, TIP_OFFSET_M) for q in joints])
        rot, pos = fit_transform(cam, base)
        res = np.linalg.norm(cam @ rot.T + pos - base, axis=1)
        rms = float(np.sqrt(np.mean(res ** 2)))
        if best is None or rms < best[0]:
            best = (rms, off, rot, pos, res, base)
    _, off, rot, pos, res, base = best
    for rec, b, r in zip(records, base, res):
        rec["base"] = b.tolist()
        rec["residual_m"] = float(r)
    return off, rot, pos, res


def draw(bgr: np.ndarray, clicks: list[tuple[int, int]], current: int = -1,
         done: set[int] = frozenset(), labels: dict[int, str] = {}, banner: str = "") -> np.ndarray:
    """Numbered markers: green for the current one, grey once done, red otherwise."""
    view = bgr.copy()
    for i, (x, y) in enumerate(clicks):
        colour = (0, 220, 0) if i == current else (160, 160, 160) if i in done else (0, 0, 255)
        cv2.circle(view, (x, y), 14 if i == current else 8, colour, 3 if i == current else 2)
        cv2.putText(view, f"{i + 1}{labels.get(i, '')}", (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    if banner:
        cv2.rectangle(view, (0, 0), (view.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(view, banner, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return view


def click_markers(bgr: np.ndarray) -> list[tuple[int, int]]:
    """Show the frame and collect marker pixels by mouse click."""
    clicks: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        view = draw(bgr, clicks, banner="click every marker centre, then Enter (u undo, Esc abort)")
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):  # Enter
            break
        if key == ord("u") and clicks:
            clicks.pop()
        if key == 27:  # Esc
            clicks.clear()
            break
    cv2.imwrite(str(OUT / "calibration_markers.jpg"), view)
    return clicks


def wait_touch(bgr: np.ndarray, clicks: list[tuple[int, int]], current: int, done: set[int],
               labels: dict[int, str] = {}, what: str = "touch") -> str:
    """Keep the window up with the current marker highlighted until Enter, s or Esc."""
    banner = (f"{what} marker {current + 1} with the closed finger tips, hold still, "
              "Enter (s skip, Esc stop)")
    while True:
        cv2.imshow(WINDOW, draw(bgr, clicks, current, done, labels, banner))
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):
            return "ok"
        if key == ord("s"):
            return "skip"
        if key == 27:
            return "abort"


def read_joints(robot, seconds: float = 0.5) -> np.ndarray:
    """Average the joint read-out over a short hold; warn if the arm moved."""
    samples = []
    end = time.time() + seconds
    while time.time() < end:
        samples.append(robot.get_joint_positions())
        time.sleep(0.05)
    q = np.array(samples, dtype=np.float64)
    spread = np.degrees(q.max(0) - q.min(0)).max()
    if spread > 0.5:
        print(f"  arm moved {spread:.1f} deg during the read; hold still and try again")
        return read_joints(robot, seconds)
    return q.mean(0)


def report(off: np.ndarray, pos: np.ndarray, res: np.ndarray) -> None:
    print("residual: median %.1f mm, max %.1f mm; joint 3 zero correction %+.1f deg"
          % (1000 * np.median(res), 1000 * res.max(), math.degrees(off[2])))
    print(f"camera sits at base xyz {np.round(pos, 3).tolist()} m")


def main() -> None:
    from piper_control import piper_interface

    OUT.mkdir(exist_ok=True)
    robot = piper_interface.PiperInterface(can_port=ArmConfig().can_port)
    time.sleep(0.5)
    if robot.is_arm_enabled():
        raise SystemExit("The motors are enabled. Switch them off before moving the arm by hand.")
    if math.degrees(robot.get_joint_positions()[2]) > 3.0:
        raise SystemExit("joint 3 reads beyond its zero: the fold zero is stale. Run python zero_joint3.py first.")
    kin = Kinematics()
    cam = RealSense()
    try:
        input("Keep the arm out of the camera view, then press Enter to take the frame")
        rgb, depth = cam.frames()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        clicks = click_markers(bgr)
        if len(clicks) < 3:
            raise SystemExit("Need at least three markers; six or more is better.")

        records: dict[int, dict] = {}
        done: set[int] = set()
        for i, (px, py) in enumerate(clicks):
            patch = depth[max(0, py - 2):py + 3, max(0, px - 2):px + 3]
            valid = patch[(patch > 0.05) & (patch < 3.0)]
            if valid.size == 0:
                print(f"marker {i + 1}: no depth at ({px}, {py}), skipped")
                continue
            z = float(np.median(valid))
            print(f"marker {i + 1}/{len(clicks)} at pixel ({px}, {py}), {z:.3f} m from the camera")
            answer = wait_touch(bgr, clicks, i, done)
            if answer == "abort":
                raise SystemExit("aborted")
            if answer == "skip":
                print("  skipped")
                continue
            q = read_joints(robot)
            records[i] = {"marker": i + 1, "pixel": [px, py],
                          "cam": [(px - cam.cx) * z / cam.fx, (py - cam.cy) * z / cam.fy, z],
                          "joints": q.tolist()}
            print(f"  tips at base xyz {np.round(kin.tool_point(q, TIP_OFFSET_M), 3).tolist()} m")
            done.add(i)
        if len(records) < 3:
            raise SystemExit("Fewer than three markers touched.")

        while True:
            order = sorted(records)
            recs = [records[i] for i in order]
            off, rot, pos, res = fit(recs, kin)
            report(off, pos, res)
            labels = {i: f": {1000 * r:.0f}mm" for i, r in zip(order, res)}
            bad = [i for i, r in sorted(zip(order, res), key=lambda t: -t[1]) if r > RETOUCH_M]
            if res.max() <= GOOD_M or not bad:
                break
            print(f"re-touch markers {[i + 1 for i in bad]} (Esc in the window accepts as is)")
            stop = False
            for i in bad:
                answer = wait_touch(bgr, clicks, i, done - {i}, labels, what="re-touch")
                if answer == "abort":
                    stop = True
                    break
                if answer == "ok":
                    records[i]["joints"] = read_joints(robot).tolist()
            if stop:
                break

        if res.max() > GOOD_M:
            print("WARNING: over 10 mm. Targets may miss; consider re-running with more markers.")
        save_calibration(rot, pos, off)
        (OUT / "calibration_points.json").write_text(json.dumps(recs, indent=1))
        print("wrote calibration.json and out/calibration_points.json")
    finally:
        cv2.destroyAllWindows()
        cam.close()


if __name__ == "__main__":
    main()
