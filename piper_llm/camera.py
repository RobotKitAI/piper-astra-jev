"""RealSense colour and depth, and pixel -> world using a hand-eye calibration.

The camera is fixed and looks at the table. `calibrate.py` writes the transform
from camera frame to robot base frame into `calibration.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from piper_llm.config import CALIB

Vec = npt.NDArray[np.float64]


class RealSense:
    """Aligned colour and depth frames from a RealSense camera."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
        self.fx, self.fy, self.cx, self.cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        for _ in range(10):  # let auto-exposure settle
            self.frames()

    def frames(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
        """One colour frame (RGB) and one depth frame (metres)."""
        fs = self.align.process(self.pipeline.wait_for_frames())
        colour = np.asanyarray(fs.get_color_frame().get_data())
        depth = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * self.scale
        return colour, depth

    def close(self) -> None:
        self.pipeline.stop()


class Frame:
    """Camera-to-base transform plus pixel back-projection."""

    def __init__(self, camera: RealSense, path: Path = CALIB) -> None:
        data = json.loads(Path(path).read_text())
        self.rot = np.array(data["rotation"], dtype=np.float64).reshape(3, 3)
        self.pos = np.array(data["translation"], dtype=np.float64)
        self.cam = camera

    def to_base(self, px: float, py: float, depth: npt.NDArray[np.float32],
                ring: int = 0) -> Vec | None:
        """Back-project a pixel into robot base coordinates.

        With `ring` set, depth is read from an annulus around the pixel instead of
        its centre: looking into a bolt hole returns the bottom of the hole, while
        the ring sits on the face around it.
        """
        xi, yi = int(round(px)), int(round(py))
        h, w = depth.shape
        if not (0 <= xi < w and 0 <= yi < h):
            return None
        if ring > 0:
            ys, xs = np.ogrid[-ring - 2:ring + 3, -ring - 2:ring + 3]
            mask = (xs ** 2 + ys ** 2 >= ring ** 2) & (xs ** 2 + ys ** 2 <= (ring + 2) ** 2)
            patch = depth[max(0, yi - ring - 2):yi + ring + 3,
                          max(0, xi - ring - 2):xi + ring + 3]
            m = mask[:patch.shape[0], :patch.shape[1]]
            valid = patch[m & (patch > 0.05) & (patch < 3.0)]
        else:
            patch = depth[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3]
            valid = patch[(patch > 0.05) & (patch < 3.0)]
        if valid.size == 0:
            return None
        z = float(np.median(valid))
        point_cam = np.array([(px - self.cam.cx) * z / self.cam.fx,
                              (py - self.cam.cy) * z / self.cam.fy, z])
        return np.asarray(self.rot @ point_cam + self.pos)

    def to_pixel(self, point_base: Vec) -> tuple[float, float]:
        """Project a base-frame point back into the image, for overlays."""
        p = self.rot.T @ (np.asarray(point_base) - self.pos)
        return (float(p[0] * self.cam.fx / p[2] + self.cam.cx),
                float(p[1] * self.cam.fy / p[2] + self.cam.cy))


def save_calibration(rotation: Any, translation: Any, joint_offsets: Any = (0.0,) * 6,
                     path: Path = CALIB) -> None:
    """Write the camera-to-base transform and the joint zero corrections."""
    Path(path).write_text(json.dumps({
        "rotation": np.asarray(rotation).reshape(3, 3).tolist(),
        "translation": np.asarray(translation).reshape(3).tolist(),
        "joint_offsets": np.asarray(joint_offsets, dtype=float).reshape(6).tolist(),
    }, indent=1))
