"""A rigid object's geometry for picking: its pose on the table from the camera's
silhouette, and a grasp and a reach computed on the geometry instead of on depth.

Depth fails on shiny parts: chrome shows the camera the ceiling or nothing at
all. The outline SAM 3 draws is still clean, and an object resting on the table
has only three unknowns for each way it can rest: x, y and its turn about the
vertical. The pose is found by rendering the geometry's outline from the camera
in each resting pose, shifting and turning it until it covers the mask best. The
grasp, and how far the object reaches below it once lifted, then come from the
geometry (an STL of the part).

Lifted, an object turns about the line between the finger pads until its centre
of mass hangs below that line. The grasp therefore puts that line as close to the
centre of mass as the part allows. The centre of mass is the geometry's centre of
volume unless `meta.json` gives one: a trowel's steel blade outweighs its wooden
handle, so it balances far from the middle of its volume.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


@dataclass
class ObjectGeometry:
    mesh: Any  # trimesh.Trimesh, metres, the object's own frame
    rests: list[np.ndarray]  # 4x4: object frame -> resting on z = 0, centre of mass over x = y = 0
    points: np.ndarray  # surface samples in the object frame, for outlines and contacts
    normals: np.ndarray  # the outward normal of the face each sample lies on
    com: np.ndarray  # centre of mass in the object frame
    name: str = ""

    @classmethod
    def load(cls, folder: str | Path, min_prob: float = 0.05, samples: int = 6000) -> "ObjectGeometry":
        """The STL in `folder` and the ways it can rest on a table (trimesh's
        stable poses, those likelier than min_prob), with the centre of mass
        from `meta.json` when it gives one."""
        import trimesh

        folder = Path(folder)
        mesh = trimesh.load(next(folder.glob("*.stl")))
        meta = folder / "meta.json"
        given = json.loads(meta.read_text()).get("centre_of_mass_m") if meta.exists() else None
        com = np.asarray(mesh.center_mass if given is None else given, dtype=np.float64)
        transforms, probs = trimesh.poses.compute_stable_poses(mesh, center_mass=com, n_samples=1)
        rests = []
        for T, p in zip(transforms, probs):
            if p < min_prob:
                continue
            c = trimesh.transform_points([com], T)[0]
            T = T.copy()
            T[0, 3] -= c[0]
            T[1, 3] -= c[1]
            rests.append(T)
        points, faces = trimesh.sample.sample_surface(mesh, samples, seed=0)
        return cls(mesh, rests, np.asarray(points, dtype=np.float64), np.asarray(mesh.face_normals[faces], dtype=np.float64),
                   com, folder.name)

    # --- where it is ---------------------------------------------------------------
    def placed(self, rest: int, x: float, y: float, yaw: float, base_z: float = 0.0) -> np.ndarray:
        """4x4 pose in the base frame: resting pose `rest`, turned by yaw, at (x, y),
        on a surface `base_z` above the table."""
        T = _rz(yaw) @ self.rests[rest]
        T[0, 3] += x
        T[1, 3] += y
        T[2, 3] += base_z
        return T

    def outline(self, T: np.ndarray, frame: Any, box: tuple[int, int, int, int]) -> np.ndarray:
        """Filled outline of the posed geometry in the image window `box`
        (x0, y0, x1, y1), as a 0/1 mask the size of the window."""
        cam = frame.cam
        p = self.points @ T[:3, :3].T + T[:3, 3]
        c = (p - frame.pos) @ frame.rot  # base -> camera: rot.T @ (p - pos)
        u = np.round(c[:, 0] * cam.fx / c[:, 2] + cam.cx).astype(int) - box[0]
        v = np.round(c[:, 1] * cam.fy / c[:, 2] + cam.cy).astype(int) - box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        img = np.zeros((h, w), np.uint8)
        img[v[ok], u[ok]] = 1
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
        flood = img.copy()
        cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
        return img | (1 - flood)  # holes filled: SAM 3's masks cover a bore seen from above

    def locate(self, mask: np.ndarray, frame: Any, base_z: float = 0.0) -> tuple[np.ndarray, float, int]:
        """Pose that best covers `mask` (a full-image 0/1 mask), resting on a
        surface `base_z` above the table (a stand, say): (T, IoU, rest)."""
        cam = frame.cam
        ys, xs = np.nonzero(mask)
        pad = 40
        box = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(mask.shape[1], xs.max() + pad), min(mask.shape[0], ys.max() + pad))
        seen = mask[box[1]:box[3], box[0]:box[2]].astype(bool)

        def ray_at(u: float, v: float, z: float) -> np.ndarray:
            d = frame.rot @ np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
            return frame.pos + d * ((z - frame.pos[2]) / d[2])

        def iou(T: np.ndarray) -> tuple[float, np.ndarray]:
            m = self.outline(T, frame, box).astype(bool)
            return float((m & seen).sum() / max(1, (m | seen).sum())), m

        best = (-1.0, None, 0)
        for r in range(len(self.rests)):
            zc = float(self.rests[r][2, 3] + (self.rests[r][:3, :3] @ self.com)[2]) + base_z
            for yaw in np.radians(np.arange(0.0, 360.0, 6.0)):
                x, y = ray_at(xs.mean(), ys.mean(), zc)[:2]
                for _ in range(2):  # move the outline's centre onto the mask's centre
                    s, m = iou(self.placed(r, x, y, yaw, base_z))
                    if not m.any():
                        break
                    my, mx = np.nonzero(m)
                    shift = ray_at(xs.mean(), ys.mean(), zc) - ray_at(mx.mean() + box[0], my.mean() + box[1], zc)
                    x, y = x + shift[0], y + shift[1]
                s, _ = iou(self.placed(r, x, y, yaw, base_z))
                if s > best[0]:
                    best = (s, (r, x, y, yaw), r)
        s, (r, x, y, yaw), _ = best
        for step_yaw, step_xy in ((math.radians(2.0), 0.002), (math.radians(0.5), 0.0005)):
            improved = True
            while improved:
                improved = False
                for dr in ((step_yaw, 0, 0), (-step_yaw, 0, 0), (0, step_xy, 0), (0, -step_xy, 0), (0, 0, step_xy), (0, 0, -step_xy)):
                    cand = (yaw + dr[0], x + dr[1], y + dr[2])
                    sc, _ = iou(self.placed(r, cand[1], cand[2], cand[0], base_z))
                    if sc > s + 1e-6:
                        s, (yaw, x, y), improved = sc, cand, True
        return self.placed(r, x, y, yaw, base_z), s, r

    def locate_held(self, mask: np.ndarray, frame: Any, T0: np.ndarray, hidden: np.ndarray | None = None,
                    radius: float = 0.012, step: float = 0.002) -> tuple[np.ndarray, float]:
        """Where a part held by the gripper really is. Its orientation and height
        are known (T0, from the grasp and the arm); only x and y are searched,
        within `radius` of T0, for the outline that covers `mask` (a full-image
        0/1 mask) best. Pixels in `hidden` (fingers in front of the part, a part
        behind it) count for neither side. Returns (T, IoU)."""
        cam = frame.cam
        seen_full = mask.astype(bool)
        keep_full = np.ones_like(seen_full) if hidden is None else ~hidden.astype(bool)
        # a window around where the outline can be, the search radius included
        c = (self.points @ T0[:3, :3].T + T0[:3, 3] - frame.pos) @ frame.rot
        u, v = c[:, 0] * cam.fx / c[:, 2] + cam.cx, c[:, 1] * cam.fy / c[:, 2] + cam.cy
        pad = int(radius * cam.fx / float(np.median(c[:, 2]))) + 20
        h, w = mask.shape
        box = (max(0, int(u.min()) - pad), max(0, int(v.min()) - pad), min(w, int(u.max()) + pad), min(h, int(v.max()) + pad))
        seen = seen_full[box[1]:box[3], box[0]:box[2]]
        keep = keep_full[box[1]:box[3], box[0]:box[2]]

        def score(dx: float, dy: float) -> float:
            T = T0.copy()
            T[0, 3] += dx
            T[1, 3] += dy
            m = self.outline(T, frame, box).astype(bool)
            return float((m & seen & keep).sum() / max(1, ((m | seen) & keep).sum()))

        grid = np.arange(-radius, radius + 1e-9, step)
        best = max(((score(dx, dy), dx, dy) for dx in grid for dy in grid), key=lambda r: r[0])
        for fine in (step / 2, step / 4):
            improved = True
            while improved:
                improved = False
                for ddx, ddy in ((fine, 0), (-fine, 0), (0, fine), (0, -fine)):
                    sc = score(best[1] + ddx, best[2] + ddy)
                    if sc > best[0] + 1e-6:
                        best, improved = (sc, best[1] + ddx, best[2] + ddy), True
        T = T0.copy()
        T[0, 3] += best[1]
        T[1, 3] += best[2]
        return T, best[0]

    # --- how to hold it ----------------------------------------------------------------
    def grasp(self, T: np.ndarray, open_m: float = 0.070, tip_below: float = 0.015, tip_clear: float = 0.005,
              pad_half_width: float = 0.010, pad_above: float = 0.012, lowest_tool: float = 0.020,
              finger: float = 0.005, finger_half_width: float = 0.012, finger_above: float = 0.045,
              step: float = 0.002, touch: float = 0.0005, square: float = 0.9, lever_step: float = 0.003,
              margin_min: float = 0.006
              ) -> tuple[Vec, float, float] | None:
        """A top-down grasp on the posed geometry: (tool point, yaw, width).

        For each finger direction, place along the part and tool height, the
        closing fingers touch the outermost material within their width. That
        has to lie on the pads, between the tips and the top of the pads: not
        higher up the fingers, and not where the part still widens below the
        tips. Where the pads touch, the surface must face them squarely (the
        median normal within about 25 degrees of the closing direction, cos >=
        `square`): on a slope or an edge they would slip. The two touched
        places must face each other across the part, or the squeeze would turn
        it. The tips stay above the table, and the open fingers come down
        beside the part, not on it.

        Lifted, the part turns about the line through the touched places until
        its centre of mass hangs below it. So the grasp whose line passes
        closest to the centre of mass wins, in steps of `lever_step` (a few
        millimetres of lever is far below what the pads' friction holds). Then
        a height that keeps the touched places at least `margin_min` inside the
        pads, then the largest contact (flat faces, a rod along the pads), the
        narrowest, the best centred and the smallest lever."""
        P = np.vstack([self.mesh.vertices, self.points]) @ T[:3, :3].T + T[:3, 3]
        com = T[:3, :3] @ self.com + T[:3, 3]
        rel, pz = P[:, :2] - com[:2], P[:, 2]
        heights = np.arange(max(lowest_tool, tip_clear + tip_below), float(pz.max()) - 0.004, step)
        w = int(round(pad_half_width / step))

        def spread(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            LO, HI = lo.copy(), hi.copy()  # over the w bins either side
            for k in range(1, w + 1):
                LO[:-k], LO[k:] = np.minimum(LO[:-k], lo[k:]), np.minimum(LO[k:], lo[:-k])
                HI[:-k], HI[k:] = np.maximum(HI[:-k], hi[k:]), np.maximum(HI[k:], hi[:-k])
            return LO, HI

        # 1. every finger direction, place and height where the pads can close on the part (coarse, on bins)
        rows = []  # (lever estimate, deg, z, pad centre, middle along the closing line, width)
        for deg in np.arange(0.0, 180.0, 2.0):
            th = math.radians(deg)
            along = rel @ np.array([-math.sin(th), math.cos(th)])
            b = np.round(rel @ np.array([math.cos(th), math.sin(th)]) / step).astype(int)
            b0 = int(b.min())
            b -= b0
            nb = int(b.max()) + 1
            for z in heights:
                ext = []
                for zlo, zhi in ((z - tip_below, z + pad_above), (z + pad_above, z + finger_above),
                                 (z - tip_below - 0.003, z - tip_below)):  # pads, fingers above them, just below the tips
                    sel = (pz >= zlo) & (pz < zhi)
                    lo, hi = np.full(nb, np.inf), np.full(nb, -np.inf)
                    np.minimum.at(lo, b[sel], along[sel])
                    np.maximum.at(hi, b[sel], along[sel])
                    ext.append((lo, hi, *spread(lo, hi)))
                (lo, hi, LO, HI), (_, _, LOu, HIu), (_, _, LOb, HIb) = ext
                width = HI - LO
                ok = (np.isfinite(width) & (width >= 0.015) & (width <= open_m - 0.008)
                      & (LO <= LOu + touch) & (HI >= HIu - touch) & (LOb >= LO - touch) & (HIb <= HI + touch))
                if not ok.any():
                    continue
                n, s_sum = np.zeros(nb), np.zeros(nb)  # where the pads touch, for a first lever estimate
                for k in range(-w, w + 1):
                    src = slice(max(0, k), nb + min(0, k))
                    dst = slice(max(0, -k), nb + min(0, -k))
                    t = (lo[src] <= LO[dst] + touch).astype(float) + (hi[src] >= HI[dst] - touch)
                    n[dst] += t
                    s_sum[dst] += t * np.arange(nb)[src]
                for j in np.nonzero(ok & (n > 0))[0]:
                    rows.append((abs(s_sum[j] / n[j] + b0) * step, float(deg), float(z), (j + b0) * step,
                                 float((LO[j] + HI[j]) / 2), float(width[j])))
        if not rows:
            return None

        # 2. score the promising ones on the surface samples, and take the best the fingers can come down for
        S, SN = self.points @ T[:3, :3].T + T[:3, 3], self.normals @ T[:3, :3].T
        srel, sz = S[:, :2] - com[:2], S[:, 2]
        rows.sort(key=lambda r: r[0])
        done = 0
        for limit in (2 * lever_step, 6 * lever_step, np.inf):
            batch = [r for r in rows[done:] if r[0] <= limit]
            done += len(batch)
            scored = []
            per_deg: dict[float, list] = {}
            for r in batch:
                per_deg.setdefault(r[1], []).append(r)
            for deg, group in per_deg.items():
                th = math.radians(deg)
                c, a = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
                sa, sn, sside = srel @ c, SN[:, :2] @ c, srel @ a
                for _, _, z, s, mid, width in group:
                    pad = (np.abs(sside - s) <= pad_half_width) & (sz >= z - tip_below) & (sz <= z + pad_above)
                    near, far = pad & (sa <= mid - width / 2 + touch), pad & (sa >= mid + width / 2 - touch)
                    if near.sum() < 3 or far.sum() < 3 or min(-np.median(sn[near]), np.median(sn[far])) < square:
                        continue  # the pads would touch an edge or a slope
                    lat_near, lat_far = float(sside[near].mean()), float(sside[far].mean())
                    if abs(lat_near - lat_far) > 0.3 * width:
                        continue  # the touched places are not opposite each other: the squeeze would turn the part
                    touched = near | far
                    lever = abs(lat_near + lat_far) / 2
                    cz = float(sz[touched].mean())
                    margin = min(cz - (z - tip_below), z + pad_above - cz)  # how far inside the pads the touched places are
                    scored.append((int(lever / lever_step), margin < margin_min, -int(touched.sum() / 5), round(width, 3),
                                   -margin, lever, deg, z, s, mid))
            for *_, deg, z, s, mid in sorted(scored):
                th = math.radians(deg)
                c, a = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
                out = np.abs(rel @ c - mid) - open_m / 2
                if ((out >= 0) & (out <= finger) & (np.abs(rel @ a - s) <= finger_half_width) & (pz >= z - tip_below)).any():
                    continue  # an open finger would come down on the part
                point = np.array([*(com[:2] + c * mid + a * s), z])
                width = float(next(r[5] for r in batch if r[1] == deg and r[2] == z and r[3] == s))
                return point, float((th + math.pi / 2) % math.pi - math.pi / 2), width
        return None

    def lever(self, T: np.ndarray, point: Vec, yaw: float, pad_half_width: float = 0.010, tip_below: float = 0.015,
              pad_above: float = 0.012, touch: float = 0.0005) -> float:
        """How far (m) the centre of mass lies from the line between the places
        the pads touch, for a grasp at `point` closing along `yaw`: the arm that
        tilts the part once it is lifted."""
        S = self.points @ T[:3, :3].T + T[:3, 3]
        com = T[:3, :3] @ self.com + T[:3, 3]
        c, a = np.array([-math.sin(yaw), math.cos(yaw)]), np.array([math.cos(yaw), math.sin(yaw)])
        rel = S[:, :2] - np.asarray(point)[:2]
        pad = (np.abs(rel @ a) <= pad_half_width) & (S[:, 2] >= point[2] - tip_below) & (S[:, 2] <= point[2] + pad_above)
        if not pad.any():
            return float("nan")
        along = rel @ c
        near = pad & (along <= along[pad].min() + touch)
        far = pad & (along >= along[pad].max() - touch)
        mid = (float((rel[near] @ a).mean()) + float((rel[far] @ a).mean())) / 2
        return abs(mid - float((com[:2] - np.asarray(point)[:2]) @ a))

    def reach(self, T: np.ndarray, point: Vec) -> float:
        """How far the posed geometry reaches from `point`: a bound on how deep it
        can hang below a grasp there."""
        V = self.mesh.vertices @ T[:3, :3].T + T[:3, 3]
        return float(np.linalg.norm(V - np.asarray(point), axis=1).max())
