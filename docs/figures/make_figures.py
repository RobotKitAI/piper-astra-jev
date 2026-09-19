"""Figures for docs/runs-4-to-8.md, drawn from the recordings of runs 4 to 8.

The recordings are not in the repository. Runs 3 and 4 are zips on the GitHub
release "recordings"; unzip them into out/recordings. piper_llm/record.py writes
your own runs to the same place.
Run from the repository root, with the recordings in out/recordings:

    python docs/figures/make_figures.py      # every run's figures
    python docs/figures/make_figures.py 8    # only run 8's

Each overlay uses a run's first recorded frame (the scene before the arm moves),
the joints the arm reported when it closed the gripper, and the perception and
geometry the run used. Nothing here moves the arm.
"""
from __future__ import annotations

import csv
import math
import re
import sys
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")  # the Hugging Face token for SAM 3's gated weights, as in the notebooks
sys.path.insert(0, str(ROOT))
from piper_llm.camera import Frame  # noqa: E402
from piper_llm.config import FITTING_SCENE, HANDLE_SCENE, TROWEL_SCENE, SceneConfig  # noqa: E402
from piper_llm.geometry import ObjectGeometry  # noqa: E402
from piper_llm.kinematics import Kinematics  # noqa: E402
from piper_llm.perception import Detector  # noqa: E402
from piper_llm.skills import plan_grasp  # noqa: E402

OUT = Path(__file__).resolve().parent
RUNS = {4: "20260919_023612_run4", 5: "20260919_022642_run5", 6: "20260919_042020_run6", 7: "20260919_025834_run7",
        8: "20260919_073724_run8"}
CAMERA = types.SimpleNamespace(fx=912.102, fy=912.117, cx=638.714, cy=376.352)  # D435i colour, 1280 x 720
OPEN_M = 0.070  # the gripper's full opening; Arm.gripper() reads 0..1 of it
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
GREEN, RED, BLUE, YELLOW, MAGENTA, WHITE = (60, 200, 60), (230, 50, 50), (60, 120, 255), (255, 210, 0), (230, 60, 230), (255, 255, 255)

kin = Kinematics()
frame = Frame(CAMERA, ROOT / "calibration.json")


class Recording:
    """One run's folder: colour video, raw depth every few frames, joints and decisions."""

    def __init__(self, run: int) -> None:
        self.dir = ROOT / "out" / "recordings" / RUNS[run]
        self.events: dict[str, float] = {}
        for r in csv.DictReader(open(self.dir / "events.csv")):
            parts = r["note"].split()
            if len(parts) >= 2 and parts[0].isdigit():
                self.events.setdefault(parts[1], float(r["t"]))
        self.frames = [(int(r["frame"]), float(r["t"])) for r in csv.DictReader(open(self.dir / "frames.csv"))]
        self.joints = list(csv.DictReader(open(self.dir / "joints.csv")))
        self.video = cv2.VideoCapture(str(self.dir / "colour.mp4"))

    def index(self, t: float) -> int:
        return min(self.frames, key=lambda f: abs(f[1] - t))[0]

    def time(self, index: int) -> float:
        return dict(self.frames)[index]

    def rgb(self, index: int) -> np.ndarray:
        self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = self.video.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def depth(self, index: int) -> np.ndarray:
        return cv2.imread(str(self.dir / "depth_raw" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    def raw_indices(self) -> list[int]:
        return sorted(int(p.stem) for p in (self.dir / "depth_raw").glob("*.png"))

    def row(self, t: float) -> dict:
        return min(self.joints, key=lambda r: abs(float(r["t"]) - t))

    def closed_pose(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Tool point, tool rotation and yaw when the gripper was told to close, and its reading once closed."""
        r = self.row(self.events["close_gripper"] - 0.1)
        q = np.array([float(r[f"j{i}"]) for i in range(1, 7)])
        yaw = float(kin.tool_pose(q)[3])
        pos, rot = kin.data.site_xpos[kin.site].copy(), kin.data.site_xmat[kin.site].reshape(3, 3).copy()
        return pos, rot, yaw, float(self.row(self.events["lift"] - 0.2)["gripper"])


# --- drawing -----------------------------------------------------------------------
def px(p: np.ndarray) -> tuple[int, int]:
    u, v = frame.to_pixel(p)
    return int(round(u)), int(round(v))


def pads(img: np.ndarray, pos: np.ndarray, rot: np.ndarray, reading: float, colour: tuple) -> None:
    """The two finger pads where they closed: 24 mm wide, 5 mm thick, at the pads' mid height."""
    half = reading * OPEN_M / 2
    layer = img.copy()
    for side in (-1, 1):
        corners = [pos + rot @ np.array([x, side * y, 0.0015]) for x, y in ((-0.012, half), (0.012, half), (0.012, half + 0.005), (-0.012, half + 0.005))]
        poly = np.array([px(c) for c in corners], np.int32)
        cv2.fillPoly(layer, [poly], colour)
        cv2.polylines(img, [poly], True, colour, 2, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.55, img, 0.45, 0, dst=img)


def tint(img: np.ndarray, mask: np.ndarray, colour: tuple, alpha: float = 0.35) -> None:
    layer = img.copy()
    layer[mask.astype(bool)] = colour
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)


def contour(img: np.ndarray, mask: np.ndarray, colour: tuple, width: int = 2, offset: tuple = (0, 0)) -> None:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, colour, width, cv2.LINE_AA, offset=offset)


def cross(img: np.ndarray, uv: tuple, colour: tuple, size: int = 12, width: int = 2) -> None:
    cv2.drawMarker(img, (int(uv[0]), int(uv[1])), colour, cv2.MARKER_TILTED_CROSS, size, width, cv2.LINE_AA)


def crop(img: np.ndarray, centre: tuple, w: int, h: int, scale: float) -> np.ndarray:
    x0 = int(np.clip(centre[0] - w // 2, 0, img.shape[1] - w))
    y0 = int(np.clip(centre[1] - h // 2, 0, img.shape[0] - h))
    return cv2.resize(img[y0:y0 + h, x0:x0 + w], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def caption(img: np.ndarray, lines: list[tuple[str, tuple]], size: int = 20) -> np.ndarray:
    """A dark band on top with one coloured line of text per item."""
    font = ImageFont.truetype(FONT, size)
    band = 10 + len(lines) * (size + 8)
    pil = Image.fromarray(np.vstack([np.full((band, img.shape[1], 3), 25, np.uint8), img]))
    d = ImageDraw.Draw(pil)
    for i, (text, colour) in enumerate(lines):
        d.text((12, 6 + i * (size + 8)), text, font=font, fill=colour)
    return np.asarray(pil)


def save(img: np.ndarray, name: str) -> None:
    Image.fromarray(img).save(OUT / name, quality=85, optimize=True)
    print("wrote", name, img.shape[1], "x", img.shape[0])


def contact_centre(geo: ObjectGeometry, T: np.ndarray, point: np.ndarray, yaw: float) -> np.ndarray:
    """Base-frame middle of the places the pads touch (the samples lever() uses)."""
    S = geo.points @ T[:3, :3].T + T[:3, 3]
    c, a = np.array([-math.sin(yaw), math.cos(yaw)]), np.array([math.cos(yaw), math.sin(yaw)])
    rel = S[:, :2] - point[:2]
    pad = (np.abs(rel @ a) <= 0.010) & (S[:, 2] >= point[2] - 0.015) & (S[:, 2] <= point[2] + 0.012)
    along = rel @ c
    near, far = pad & (along <= along[pad].min() + 0.0005), pad & (along >= along[pad].max() - 0.0005)
    return (S[near].mean(0) + S[far].mean(0)) / 2


# --- the figures -----------------------------------------------------------------------
def sequence(run: int, rec: Recording, title: str) -> None:
    ev = rec.events
    moments = [("gripper closed", ev["lift"] - 0.3), ("lifting", ev["lift"] + 1.2),
               ("at the release height", ev["open_gripper"] - 0.3), ("after the run", rec.frames[-1][1] - 0.3)]
    tiles = []
    for name, t in moments:
        img = cv2.resize(rec.rgb(rec.index(t)), (640, 360), interpolation=cv2.INTER_AREA)
        reading = float(rec.row(t)["gripper"])
        text = f"{name}, t {t:.1f} s" + (f", gripper {reading:.2f}" if name != "after the run" else "")
        tiles.append(caption(img, [(text, WHITE)], 18))
    save(caption(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [(title, WHITE)], 22), f"run{run}_sequence.jpg")


def run4(det_dino: Detector) -> None:
    rec = Recording(4)
    rgb, scene = rec.rgb(0), SceneConfig(**HANDLE_SCENE)
    d = det_dino.best(rgb, scene.target, max_px=scene.target_max_px)
    pos, rot, yaw, reading = rec.closed_pose()
    img = rgb.copy()
    x0, y0, x1, y1 = map(int, d.box)
    cv2.rectangle(img, (x0, y0), (x1, y1), RED, 2, cv2.LINE_AA)
    cross(img, d.centre, RED, 16, 3)
    pads(img, pos, rot, reading, GREEN)
    out = crop(img, ((x0 + x1) // 2, (y0 + y1) // 2), 560, 380, 1.6)
    save(caption(out, [("Run 4, Grounding DINO asked for 'handle': a box (red) and its centre (red cross)", RED),
                       ("green: where the fingers closed, on the neck between blade and handle", GREEN)]), "run4_plan.jpg")
    sequence(4, rec, "Run 4: the trowel taken at its box's centre")


def run5(det_sam: Detector) -> tuple:
    rec = Recording(5)
    rgb, scene = rec.rgb(0), SceneConfig(**HANDLE_SCENE)
    d = det_sam.best(rgb, scene.target, max_px=scene.target_max_px)
    pos, rot, yaw, reading = rec.closed_pose()
    img = rgb.copy()
    tint(img, d.mask, BLUE, 0.4)
    contour(img, d.mask, BLUE)
    pads(img, pos, rot, reading, GREEN)
    ys, xs = np.nonzero(d.mask)
    out = crop(img, (int(xs.mean()), int(ys.mean()) + 40), 560, 380, 1.6)
    save(caption(out, [("Run 5, SAM 3 asked for 'handle': a mask of the handle (blue)", BLUE),
                       ("green: where the fingers closed, across the handle, found on the mask and depth", GREEN)]), "run5_plan.jpg")
    sequence(5, rec, "Run 5: the trowel taken by its handle, found on the mask")
    return rec, pos, yaw


def run6(det_sam: Detector, trowel: ObjectGeometry, run5_grip: tuple) -> None:
    rec = Recording(6)
    rgb, scene = rec.rgb(0), SceneConfig(**TROWEL_SCENE)
    d = det_sam.best(rgb, scene.target, max_px=scene.target_max_px)
    T, fit, _ = trowel.locate(d.mask, frame)
    pos, rot, yaw, reading = rec.closed_pose()
    com = T[:3, :3] @ trowel.com + T[:3, 3]
    img = rgb.copy()
    tint(img, d.mask, BLUE, 0.3)
    ys, xs = np.nonzero(d.mask)
    box = (max(0, xs.min() - 40), max(0, ys.min() - 40), min(1280, xs.max() + 40), min(720, ys.max() + 40))
    contour(img, trowel.outline(T, frame, box), YELLOW, 2, offset=box[:2])
    pads(img, pos, rot, reading, GREEN)
    cross(img, px(com), MAGENTA, 18, 3)
    out = crop(img, px(com), 560, 380, 1.6)
    lever6 = trowel.lever(T, pos, yaw)
    save(caption(out, [(f"Run 6, the trowel's STL fitted to SAM 3's mask (yellow outline, match {fit:.2f})", YELLOW),
                       ("magenta: the centre of mass, from the steel and beech parts", MAGENTA),
                       (f"green: where the fingers closed, across the ferrule, {1000 * lever6:.1f} mm from the centre of mass", GREEN)]),
         "run6_plan.jpg")
    sequence(6, rec, "Run 6: the trowel taken where it balances, carried level")

    # the side view: both grips on the trowel, in the trowel's own frame
    rec5, pos5, yaw5 = run5_grip
    d5 = det_sam.best(rec5.rgb(0), "tool", max_px=700)
    T5, _, _ = trowel.locate(d5.mask, frame)
    lever5 = trowel.lever(T5, pos5, yaw5)
    heel = trowel.mesh.bounds[1][0] - 0.140  # the heel edge: the blade is 140 mm long
    floor = trowel.mesh.bounds[0][2]
    def side(p_part: np.ndarray) -> tuple[float, float]:  # mm from the heel edge, mm above the table
        return 1000 * (p_part[0] - heel), 1000 * (p_part[2] - floor)
    def to_part(T: np.ndarray, p: np.ndarray) -> np.ndarray:
        inv = np.linalg.inv(T)
        return inv[:3, :3] @ p + inv[:3, 3]
    s, x_min, z_top = 4.0, -115.0, 95.0  # pixels per mm, left edge, top
    W, H = int((150 - x_min) * s), int((z_top + 25) * s)
    img = np.full((H, W, 3), 250, np.uint8)
    to_px = lambda x, z: (int((x - x_min) * s), int((z_top - z) * s))
    cv2.line(img, to_px(x_min, 0), to_px(150, 0), (170, 170, 170), 2)
    for f in trowel.mesh.vertices[trowel.mesh.faces]:
        cv2.fillConvexPoly(img, np.array([to_px(*side(v)) for v in f], np.int32), (125, 125, 125), cv2.LINE_AA)
    grip_x = {}
    for label_, colour, T_, pos_, yaw_, lever_ in (("run 5", RED, T5, pos5, yaw5, lever5), ("run 6", GREEN, T, pos, yaw, lever6)):
        gx, gz = side(to_part(T_, pos_))
        layer = img.copy()
        cv2.rectangle(layer, to_px(gx - 12, gz - 15), to_px(gx + 12, gz + 12), colour, -1)
        cv2.addWeighted(layer, 0.45, img, 0.55, 0, dst=img)
        cv2.rectangle(img, to_px(gx - 12, gz - 15), to_px(gx + 12, gz + 12), colour, 2)
        cx_, cz_ = side(to_part(T_, contact_centre(trowel, T_, pos_, yaw_)))
        cv2.line(img, to_px(cx_, gz - 22), to_px(cx_, gz + 20), colour, 2, cv2.LINE_AA)
        grip_x[label_] = cx_
    cmx, cmz = side(trowel.com)
    y_arrow = cmz - 12  # the lever of run 5, drawn under the handle
    cv2.arrowedLine(img, to_px(cmx, y_arrow), to_px(grip_x["run 5"], y_arrow), RED, 2, cv2.LINE_AA, tipLength=0.04)
    cv2.arrowedLine(img, to_px(grip_x["run 5"], y_arrow), to_px(cmx, y_arrow), RED, 2, cv2.LINE_AA, tipLength=0.04)
    cv2.line(img, to_px(cmx, cmz), to_px(cmx, y_arrow - 3), MAGENTA, 1, cv2.LINE_AA)
    cv2.circle(img, to_px(cmx, cmz), 9, MAGENTA, -1, cv2.LINE_AA)
    pil = Image.fromarray(img)
    dr, f = ImageDraw.Draw(pil), ImageFont.truetype(FONT, 22)
    dr.text(to_px(-108, 88), "handle (beech)", font=f, fill=(90, 90, 90))
    dr.text(to_px(60, 16), "blade (steel)", font=f, fill=(90, 90, 90))
    dr.text(to_px(cmx + 24, cmz - 4), "centre of mass", font=f, fill=MAGENTA)
    dr.text(to_px((cmx + grip_x["run 5"]) / 2 - 16, y_arrow - 2), f"{1000 * lever5:.0f} mm", font=f, fill=RED)
    img = caption(np.asarray(pil), [("The trowel from the side, from its STL, with the two grips where the fingers closed", WHITE),
                                   (f"run 5, red: on the handle, the grip line {1000 * lever5:.0f} mm behind the centre of mass: lifted, it tilts blade down", RED),
                                   (f"run 6, green: across the ferrule, {1000 * lever6:.1f} mm from it: lifted, it stays level", GREEN)])
    save(img, "trowel_side.jpg")


def run7(det_sam: Detector, fitting: ObjectGeometry) -> None:
    rec = Recording(7)
    rgb, depth, scene = rec.rgb(0), rec.depth(0), SceneConfig(**FITTING_SCENE)
    d = det_sam.best(rgb, scene.target, max_px=scene.target_max_px)
    T, fit, _ = fitting.locate(d.mask, frame)
    pos, rot, yaw, reading = rec.closed_pose()
    centre = px(T[:3, 3])

    # colour and measured height side by side
    ys_, xs_ = np.mgrid[0:720, 0:1280]
    zc = depth
    cam_pts = np.stack([(xs_ - CAMERA.cx) / CAMERA.fx * zc, (ys_ - CAMERA.cy) / CAMERA.fy * zc, zc], -1)
    height = (cam_pts @ frame.rot.T + frame.pos)[..., 2]
    inside = d.mask.astype(bool)
    missing = float((zc[inside] <= 0).mean())
    h_in = 1000 * height[inside & (zc > 0)]
    above = float((h_in > 33.0).mean())  # nothing on the part is higher than its 31.5 mm top
    c_img = crop(rgb, centre, 200, 150, 2.4)
    hc = crop(np.where(zc > 0, height, np.nan).astype(np.float32), centre, 200, 150, 2.4)
    hn = np.clip(np.nan_to_num(hc, nan=0.0) / 0.040, 0, 1)
    h_img = cv2.cvtColor(cv2.applyColorMap((255 * hn).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    h_img[~np.isfinite(hc)] = 0
    cs, _ = cv2.findContours(crop(inside.astype(np.uint8), centre, 200, 150, 2.4), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(h_img, cs, -1, WHITE, 2, cv2.LINE_AA)
    both = np.hstack([c_img, np.full((c_img.shape[0], 8, 3), 255, np.uint8), h_img])
    save(caption(both, [("Run 7: the chrome pipe fitting in colour, and the height the depth camera measures", WHITE),
                        ("height scale: blue at the table, red at 40 mm; black: no depth; white: SAM 3's outline", WHITE),
                        (f"inside the outline, {100 * missing:.0f}% of pixels have no depth and {100 * (1 - missing) * above:.0f}% read higher "
                         f"than the part's 31.5 mm top", WHITE)], 18), "run7_depth.jpg")
    print(f"run 7 depth: {100 * missing:.0f}% missing inside the mask, {100 * (1 - missing) * above:.1f}% of the mask above 33 mm")

    # camera-only plans from the still frames before the arm comes, against the geometry's grasp
    img = rgb.copy()
    ys, xs = np.nonzero(d.mask)
    box = (max(0, xs.min() - 40), max(0, ys.min() - 40), min(1280, xs.max() + 40), min(720, ys.max() + 40))
    contour(img, d.mask, BLUE, 2)
    contour(img, fitting.outline(T, frame, box), YELLOW, 2, offset=box[:2])
    plans = []
    for i in [k for k in rec.raw_indices() if rec.time(k) < rec.events["approach_target"] - 0.5]:
        f_rgb, f_depth = rec.rgb(i), rec.depth(i)
        di = det_sam.best(f_rgb, scene.target, max_px=scene.target_max_px)
        p = None if di is None else plan_grasp(di, f_rgb, f_depth, frame, scene, det_sam)
        if p is not None and p[0][2] <= scene.target_max_z:
            plans.append(p[0])
    for p in plans:
        cross(img, px(p), RED, 12, 2)
    pads(img, pos, rot, reading, GREEN)
    spread = max(np.linalg.norm(p[:2] - q[:2]) for p in plans for q in plans) if len(plans) > 1 else 0.0
    off = 1000 * np.linalg.norm(pos[:2] - T[:2, 3])
    save(caption(crop(img, centre, 340, 220, 2.6),
                 [(f"Run 7: SAM 3's mask (blue); the fitting's STL fitted to it (yellow, match {fit:.2f})", YELLOW),
                  (f"red: camera-only grasp points, one per still frame ({len(plans)}), up to {1000 * spread:.0f} mm apart", RED),
                  (f"green: where the fingers closed from the geometry, across the flats, {off:.1f} mm off the axis", GREEN)]),
         "run7_plan.jpg")
    sequence(7, rec, f"Run 7: the chrome fitting gripped across its flats (gripper {reading:.2f} = 34.7 mm flats plus play)")


def top_shift(gray: np.ndarray, template: np.ndarray, uv: tuple[int, int]) -> tuple[tuple[int, int], float]:
    """Where the spanner's top hex face is in a frame, in pixels from where it stood in the first frame."""
    u, v = uv
    res = cv2.matchTemplate(gray[v - 50:v + 50, u - 50:u + 50], template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return (loc[0] - 32, loc[1] - 32), float(score)


def run8(det_sam: Detector, fitting: ObjectGeometry, spanner: ObjectGeometry) -> None:
    rec = Recording(8)
    rgb = rec.rgb(0)
    found: dict[str, tuple] = {}
    for d in det_sam.detect(rgb, ["metal object"]):  # each mask is named by the geometry that fits it better, as in the run
        if d.mask is None:
            continue
        fits = {"fitting": fitting.locate(d.mask, frame), "spanner": spanner.locate(d.mask, frame)}
        name = max(fits, key=lambda k: fits[k][1])
        if name not in found or fits[name][1] > found[name][1][1]:
            found[name] = (d.mask.astype(bool), fits[name])
    (mf, (Tf, fit_f, _)), (ms, (Ts, fit_s, _)) = found["fitting"], found["spanner"]
    top = float((spanner.mesh.vertices @ Ts[:3, :3].T + Ts[:3, 3])[:, 2].max())
    top_uv = px(np.array([*Ts[:2, 3], top]))
    pos, rot, yaw, reading = rec.closed_pose()
    grip_z = 0.023  # the tool point above the fitting's bottom where the fingers took it (task.SlideOver)
    ys, xs = np.nonzero(mf | ms)
    region = ((xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2 - 20)
    w, h = min(1280, int(xs.max() - xs.min()) + 260), min(720, int(ys.max() - ys.min()) + 220)

    # the plan: both parts from their STLs, and where the fingers closed
    img = rgb.copy()
    for mask, T, geo, colour in ((mf, Tf, fitting, YELLOW), (ms, Ts, spanner, BLUE)):
        tint(img, mask, colour, 0.25)
        contour(img, geo.outline(T, frame, (0, 0, 1280, 720)), colour, 2)
    pads(img, pos, rot, reading, GREEN)
    cross(img, top_uv, RED, 16, 3)
    save(caption(crop(img, region, w, h, 1000 / w),
                 [(f"Run 8: each SAM 3 mask fitted with both STLs: the fitting (yellow, match {fit_f:.2f}), "
                   f"the spanner (blue, {fit_s:.2f})", WHITE),
                  ("green: where the fingers closed, across the flats, leaning 45°, 10 mm towards the arm", GREEN),
                  (f"red: the spanner's axis at its top, {1000 * top:.0f} mm up; the bore leaves about 2 mm a side", RED)], 18),
         "run8_plan.jpg")

    # the looks, as the run saved them
    notes = [r["note"] for r in csv.DictReader(open(rec.dir / "events.csv")) if r["note"].startswith("look ")]
    tiles = []
    for i, note in enumerate(notes):
        m = re.search(r"outline match ([\d.]+).* and ([\d.]+) mm off the spanner's axis", note)
        look = cv2.cvtColor(cv2.imread(str(rec.dir / f"look_{i}.jpg")), cv2.COLOR_BGR2RGB)
        then = "move by that" if i < len(notes) - 1 else "slide down"
        tiles.append(caption(crop(look, (top_uv[0], top_uv[1] - 40), 300, 240, 1.6),
                             [(f"look {i}: {m[2]} mm off (match {m[1]}): {then}", WHITE)], 16))
    save(caption(np.hstack(tiles), [("Run 8: the held fitting 8 mm above the spanner's top, as the camera measured it", WHITE),
                                    ("blue: SAM 3's mask; green: where the arm has it; yellow: the STL fitted to the mask", WHITE),
                                    ("red cross: the spanner's axis at its top; yellow cross: the fitting's centre, 24 mm", WHITE),
                                    ("higher, so lined up they sit one above the other; dimmed: left out of the fit", WHITE)], 18),
         "run8_look.jpg")

    # the steps
    ev = rec.events
    t_slide, t_go = ev["slide_down"], ev["let_go"]
    moments = [("gripped", ev["lift"] - 0.3), ("lined up", t_slide + 0.3), ("half way down", (t_slide + t_go) / 2),
               ("at the floor", t_go - 0.3), ("let go", t_go + 1.6), ("after the run", rec.frames[-1][1] - 0.3)]
    tiles = []
    for name, t in moments:
        tile = cv2.resize(crop(rec.rgb(rec.index(t)), region, w, h, 1.0), (480, int(480 * h / w)), interpolation=cv2.INTER_AREA)
        r = rec.row(t)
        bottom = 1000 * (float(kin.tool_pose(np.array([float(r[f"j{k}"]) for k in range(1, 7)]))[2]) - grip_z)
        text = f"{name}, t {t:.0f} s" + (f", the fitting's bottom {bottom:.0f} mm up" if r["holding"] == "True" else "")
        text = text.replace(", the fitting's bottom 1 mm up", ", on the table")  # gripped: still standing on it
        tiles.append(caption(tile, [(text, WHITE)], 15))
    save(caption(np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]),
                 [("Run 8: Jev's steps, from the grasp to the fitting on the table around the spanner", WHITE)], 20),
         "run8_sequence.jpg")

    # the spanner's top: where it stood, where the slide pushed it most, where it ended
    tpl = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[top_uv[1] - 18:top_uv[1] + 18, top_uv[0] - 18:top_uv[0] + 18]
    most, t_most = (0, 0), t_slide
    rec.video.set(cv2.CAP_PROP_POS_FRAMES, rec.index(t_slide))
    for i in range(rec.index(t_slide), rec.index(t_go) + 1):
        ok, bgr = rec.video.read()
        if not ok:
            break
        s_, score = top_shift(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), tpl, top_uv)
        if score > 0.85 and math.hypot(*s_) > math.hypot(*most):  # only where the top is in clear view
            most, t_most = s_, rec.time(i)
    t_end = rec.frames[-1][1] - 0.3
    end, _ = top_shift(cv2.cvtColor(rec.rgb(rec.index(t_end)), cv2.COLOR_RGB2GRAY), tpl, top_uv)
    mm = 1000 * float(((np.array([*Ts[:2, 3], top]) - frame.pos) @ frame.rot)[2]) / CAMERA.fx  # per pixel at the top
    r = rec.row(t_most)
    b_most = 1000 * (float(kin.tool_pose(np.array([float(r[f"j{k}"]) for k in range(1, 7)]))[2]) - grip_z)
    print(f"run 8: the spanner's top moved at most {mm * math.hypot(*most):.1f} mm (t {t_most:.1f} s, the fitting's bottom "
          f"{b_most:.0f} mm up) and ended {mm * math.hypot(*end):.1f} mm from where it stood")
    tiles = []
    for name, t, s_ in (("before the run", 1.0, (0, 0)), (f"most pushed, t {t_most:.0f} s", t_most, most), ("after the run", t_end, end)):
        img = rec.rgb(rec.index(t)).copy()
        cross(img, top_uv, RED, 14, 2)
        cross(img, (top_uv[0] + s_[0], top_uv[1] + s_[1]), YELLOW, 14, 2)
        tiles.append(caption(crop(img, top_uv, 170, 190, 2.0), [(f"{name}: {mm * math.hypot(*s_):.1f} mm off", WHITE)], 16))
    save(caption(np.hstack(tiles), [("Run 8: the spanner's top; red: where it stood before the run, yellow: where it is", WHITE)], 18),
         "run8_spanner.jpg")


if __name__ == "__main__":
    only = {int(a) for a in sys.argv[1:]} or set(RUNS)
    sam = Detector(backend="sam3")
    fitting = ObjectGeometry.load(ROOT / "piper_llm" / "assets" / "objects" / "pipe-fitting")
    if 4 in only:
        run4(Detector(backend="dino"))
    if only & {5, 6}:
        grip5 = run5(sam)  # run 6's side view compares its grip with run 5's
    if 6 in only:
        run6(sam, ObjectGeometry.load(ROOT / "piper_llm" / "assets" / "objects" / "trowel"), grip5)
    if 7 in only:
        run7(sam, fitting)
    if 8 in only:
        run8(sam, fitting, ObjectGeometry.load(ROOT / "piper_llm" / "assets" / "objects" / "box-spanner", min_prob=0.0))
