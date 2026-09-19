"""Record a run: colour and depth video, raw depth, joints and events, one clock.

    rec = Recorder(cam, arm, "run2")             # after RealSense() and Arm()
    ...
    rec.note("3 lift 0.98")                       # stamps a line on the timeline and the video
    ...
    rec.close()                                   # before cam.close()

The recorder's thread is the only reader of the camera; `cam.frames()` then
returns the latest captured pair, so the run loop needs no change. Written to
a new folder under `root` named by the start time and the run name,
e.g. out/recordings/20260918_234807_run2:

    colour.mp4      1280x720 at `fps`, with the last note overlaid
    depth.mp4       depth colour-mapped, 0 to 1 m
    depth_raw/      16-bit PNG in millimetres, every `raw_depth_every` frames
    frames.csv      frame index, time
    joints.csv      time, six joint angles (rad), gripper opening, holding
    events.csv      time, note
"""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class Recorder:
    def __init__(self, cam: Any, arm: Any | None, name: str = "run",
                 root: str | Path = "out/recordings", fps: float = 15.0,
                 raw_depth_every: int = 5) -> None:
        self.cam, self.arm, self.fps = cam, arm, fps
        self.dir = Path(root) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + name)
        (self.dir / "depth_raw").mkdir(parents=True)
        self.raw_every = max(1, int(raw_depth_every))
        self.t0 = time.time()
        self._note = ""
        self._latest: tuple[np.ndarray, np.ndarray] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        rgb, depth = cam.frames()
        h, w = depth.shape
        self._colour = cv2.VideoWriter(str(self.dir / "colour.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self._depth = cv2.VideoWriter(str(self.dir / "depth.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self._frames = open(self.dir / "frames.csv", "w", newline="")
        self._joints = open(self.dir / "joints.csv", "w", newline="")
        self._events = open(self.dir / "events.csv", "w", newline="")
        self._frames_w, self._joints_w, self._events_w = (csv.writer(f) for f in (self._frames, self._joints, self._events))
        self._frames_w.writerow(["frame", "t"])
        self._joints_w.writerow(["t", "j1", "j2", "j3", "j4", "j5", "j6", "gripper", "holding"])
        self._events_w.writerow(["t", "note"])
        self._latest = (rgb, depth)
        self._real_frames = cam.frames
        cam.frames = self.latest_frames  # the run loop reads what the thread captured
        self._thread = threading.Thread(target=self._loop, name="recorder", daemon=True)
        self._thread.start()

    def latest_frames(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            rgb, depth = self._latest
            return rgb.copy(), depth.copy()

    def note(self, text: str) -> None:
        """Stamp a note on the timeline and on the video until the next one."""
        self._note = text
        with self._lock:
            self._events_w.writerow([round(time.time() - self.t0, 3), text])
            self._events.flush()

    def _loop(self) -> None:
        period = 1.0 / self.fps
        idx = 0
        while not self._stop.is_set():
            t_frame = time.time()
            try:
                rgb, depth = self._real_frames()
            except Exception:
                continue
            t = time.time() - self.t0
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if self._note:
                cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 36), (0, 0, 0), -1)
                cv2.putText(bgr, self._note, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            d8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)  # 0..1 m
            d8[depth <= 0] = 0
            row: list[Any] = [round(t, 3)]
            if self.arm is not None:
                try:
                    row += [round(float(v), 4) for v in self.arm.joints()] + [round(self.arm.gripper(), 3), self.arm.holding()]
                except Exception:
                    row += [""] * 8
            with self._lock:
                self._latest = (rgb, depth)
                self._colour.write(bgr)
                self._depth.write(cv2.applyColorMap(d8, cv2.COLORMAP_JET))
                if idx % self.raw_every == 0:
                    cv2.imwrite(str(self.dir / "depth_raw" / f"{idx:06d}.png"), (depth * 1000.0).astype(np.uint16))
                self._frames_w.writerow([idx, round(t, 3)])
                if self.arm is not None:
                    self._joints_w.writerow(row)
            idx += 1
            time.sleep(max(0.0, period - (time.time() - t_frame)))

    def close(self) -> None:
        """Stop the thread, finish the files, give the camera back."""
        self._stop.set()
        self._thread.join(timeout=3.0)
        self.cam.frames = self._real_frames
        for w in (self._colour, self._depth):
            w.release()
        for f in (self._frames, self._joints, self._events):
            f.flush()
            f.close()
        print("recording:", self.dir)
