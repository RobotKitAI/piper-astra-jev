# piper-astra-jev

Eight demo runs on a real AgileX PiPER arm, driven by large models.

| Run | Decider | Perception | Scene |
|---|---|---|---|
| 1 | GPT-6 Astra (OpenAI) | Astra sees the camera images directly | cube into a tray |
| 2 | Jev (TypeSafe, via OpenRouter) | Grounding DINO, local | cube into a tray |
| 3 | Jev | SAM 3, local | cube into a tray |
| 4 | Jev | Grounding DINO, local | an object by its handle, into the tray |
| 5 | Jev | SAM 3, local | same, with a grasp planned on the mask |
| 6 | Jev | SAM 3 and the object's geometry (STL) | the trowel of runs 4 and 5, held where it balances |
| 7 | Jev | SAM 3 and the part's geometry (STL) | a chrome pipe fitting into the tray |
| 8 | Jev | SAM 3 and both parts' geometry (STL) | the fitting of run 7 slid down over a standing box spanner |

Runs 1 to 3 use the same scene: run 1 changes the decider, run 3 changes the perception. Runs 4
and 5 change the object to one taken by its handle, where a box and a mask lead to
different grasps; see "How a grasp works". Runs 6 and 7 add the object's geometry, its STL: run 6
holds the trowel of runs 4 and 5 where it balances, so it is carried level, and run 7 grips a chrome
part the depth camera cannot measure. Run 8 places two chrome parts by their geometry and fits one over
the other, with about 2 mm to spare on each side. [`docs/runs-4-to-8.md`](docs/runs-4-to-8.md) explains
runs 4 to 8 with pictures from their recordings.

These are single runs, not a benchmark: each notebook was run once for the
recording it describes. Timings and outcomes will vary between runs.

## ⚠️ Safety

**The arm can move in ways you do not expect.** A model may choose a wrong
skill, and perception may report an object in the wrong place. The software
limits in `piper_llm/safety.py` reduce risk but do not remove it.

Before every run:

- Keep the physical e-stop in reach. Use it first, ask later.
- Keep hands and cables outside the arm's reach while it is enabled.
- Clear the table of anything you cannot afford to have knocked over.
- Start with `speed_percent=10` and raise it only after a clean run.
- Run the dry checks in each notebook before enabling torque.
- Never leave the arm enabled and unattended.

The gripper closes with real force. Do not put fingers between the jaws.

## Hardware

- AgileX PiPER arm on a CAN interface (`can0`)
- Intel RealSense depth camera looking at the table
- Workstation with an RTX 4090 (or similar) for the local perception models
- Physical e-stop

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The PiPER MuJoCo model used for kinematics comes from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (MIT) and is
vendored in `piper_llm/assets/agilex_piper`, so nothing else needs cloning.

Bring up the CAN bus:

```bash
sudo ip link set can0 up type can bitrate 1000000
```

## Keys

No keys are stored in this repository. Copy `.env.example` to `.env` and fill it
in locally:

```
OPENROUTER_API_KEY=   # run 2 and 3 (Jev)
OPENAI_API_KEY=       # run 1 (Astra)
HF_TOKEN=             # SAM 3 is a gated model
```

`.env` is git-ignored.

## Run

```bash
jupyter lab
```

Then open, in order:

1. `notebooks/01_astra_pick_place.ipynb`
2. `notebooks/02_jev_dino_pick_place.ipynb`
3. `notebooks/03_jev_sam_pick_place.ipynb`
4. `notebooks/04_jev_dino_handle.ipynb`
5. `notebooks/05_jev_sam_handle.ipynb`
6. `notebooks/06_jev_sam_trowel_geometry.ipynb`
7. `notebooks/07_jev_sam_fitting_geometry.ipynb`
8. `notebooks/08_jev_sam_fitting_over_spanner.ipynb`

Each notebook has the same shape: check hardware, calibrate the camera, run
without torque, then run for real.

## How a grasp works: look, reach, touch

The fixed camera sees the whole scene only while the arm is out of the way. As
the gripper comes down over the object, the fingers hide it. The run loop,
`piper_llm/task.py`, therefore works the way a person does:

1. **Look while the view is clear.** The grasp is planned while the target is in
   clear view, and kept while the fingers are in the picture. The tray is
   remembered while the arm is over it.
2. **Reach and trust the memory.** A sighting above the table height
   (`SceneConfig.target_max_z`) or larger than the object can be
   (`target_max_px`) is not the target: it is a sliver seen on the gripper.
3. **Confirm by touch.** Whether the grasp worked comes from the gripper, not
   the image: `Arm.holding()` is true when the fingers stopped part way. The
   lift is only allowed on that signal.

**Where to grasp.** A box detector gives only a box, so the grasp goes to its
centre: right for a cube, wrong for a ring or a tool, where the centre may be
a hole or a flat blade. SAM 3 gives a mask; fused with depth it tells where
solid material is and how wide, and the grasp goes to a line across the part
with room for both fingers beside it (`skills.grasp_search`). Runs 4 and 5 show
the difference on an object taken by its handle.

**How high to carry.** An object lifted by one end swings until its centre of
mass hangs below the fingers: a trowel held mid-handle hangs blade down, about
18 cm below the gripper. Before the grasp, the task measures how far the whole
object reaches from the grasp point, which bounds how deep anything can hang,
and carries it that much above the tray's rim, measured from depth, with 3 cm
to spare. It lowers it until that depth ends 2 cm above the tray's floor. When
a carry is higher than the arm can go, the gripper leans outward, which a
hanging object does not mind; when even that is not enough, the run stops
before lifting and says why.

**When depth fails: the part's geometry.** Chrome shows the depth camera the
ceiling or nothing. With the part's STL, `piper_llm/geometry.py` finds its pose
from the silhouette alone: it renders the geometry's outline from the camera in
each way the part can rest on the table, and shifts and turns it until it covers
SAM 3's mask. Resting on the table fixes the height, so no depth is needed. The
grasp (across a pair of flat faces where the part has them) and the reach then
come from the geometry. Run 7 uses it on a chrome pipe fitting.

**Where it balances: the geometry's centre of mass.** Lifted, an object turns
about the line between the pads until its centre of mass hangs below that line.
With the geometry, the grasp search puts that line as close to the centre of
mass as the fingers can grip. It tries every finger direction, height and place
along the part, keeps those where the pads, not the fingertips or the fingers
above them, meet the outermost material, and ranks them by how far the centre
of mass lies from the points the pads touch. A trowel's steel blade outweighs
its beech handle, so it balances under the ferrule, 42 mm from the middle of its
volume; `meta.json` gives that centre of mass. Run 5's camera-only grasp on the
handle was 56 mm from it, and the trowel tilted blade down; run 6 grips straight
across the ferrule, a few millimetres from it, with the pads centred on the ferrule's widest line.

**Two parts, 2 mm to spare.** Run 8 slides the fitting of run 7 down over a box spanner standing on end:
the fitting's bore is 25.5 mm, the spanner's hex 21.3 mm across its corners. Both are chrome, so both are
placed by their STLs (`task.SlideOver`). The wrist cannot point the gripper straight down 20 cm up, so the
gripper leans 45 degrees about the line between its fingers the whole run: the fingers stay level, and the
fitting held upright between them stays upright. That lean needs the fitting 50 to 60 cm from the base, so
the run widens the workspace box to 58 cm. The gripper holds the fitting 10 mm towards the arm, at the
fingers' outer end, so it goes further down before the gripper reaches the spanner's top. The hand-eye error
up there is a few millimetres, more than the gap. So over the spanner the camera measures the held fitting
against the spanner's axis, with the fingers masked out; both come from the same camera, so its
calibration error mostly cancels. The arm moves by the difference until the two agree within 0.8 mm. The
fitting then goes down at 5 mm/s while the depth camera watches the gap between the gripper and the
spanner's top. The arm lets go well onto the spanner, and the fitting slides down to the table.

The limit is the same as for a person reaching with closed eyes: if the object
moves while hidden, the plan is wrong, the grasp closes on nothing, and the
decider has to open and look again. A wrist camera would remove that blind
moment.

## Arm driver

The arm is driven through [piper_control](https://github.com/Reimagine-Robotics/piper_control),
an MIT-licensed wrapper around AgileX `piper_sdk` that handles the enable
sequence and the cases where the arm stops responding.

## What was tested in simulation

These runs come from simulation experiments on the same skill layer. Measured
there: Jev decides in about 350 ms, Grounding DINO locates objects in about
70 ms on a 4090, SAM 3 locates bolt holes to about 0.3 mm when it finds them,
and its recall is roughly 60%. Expect worse on real hardware.

## License

Apache License 2.0, see [`LICENSE`](LICENSE). The PiPER MuJoCo model in
`piper_llm/assets/agilex_piper` is from MuJoCo Menagerie and keeps its own MIT
license, in that folder.
