# Pre-run checks

Run these once per session, in order. Stop at the first failure.

## 1. CAN bus

```bash
sudo ip link set can0 up type can bitrate 1000000
candump can0 | head        # traffic means the arm is powered and connected
```

## 1b. Joint 3 zero, after every power-up of the arm

The arm forgets the zero of joint 3 when powered off and then reports the
fold about 8 degrees off, which shifts every target and makes the forearm drop
when power is cut. `Arm` re-zeros it by itself at connect when the arm rests
folded. For the last half degree of play, press the forearm against its stop
and run `python zero_joint3.py` instead. `calibrate.py` needs it done first.

## 2. Arm, read-only

```python
from piper_llm.arm import Arm
arm = Arm()
print(arm.joints(), arm.gripper(), arm.status_error())
```

Compare the joint angles against the arm's actual pose. If a sign or zero
differs from the MuJoCo model, fix it before any motion: every target in these
runs is computed from that model. Found on this arm: joint 3 reports about 8
degrees off its zero (step 1b); the six joints turn in the same sense as the
model (`Arm.JOINT_SIGN`, all +1). A wrong joint 6 sign only shows when a grasp
needs the fingers turned: they end along a handle instead of across it.

## 3. Camera

```python
from piper_llm.camera import RealSense
cam = RealSense(); rgb, depth = cam.frames()
```

Depth should be dense over the table. Shiny metal returns holes in the depth
image; add matte tape or move the light if the part area is empty.

## 4. Calibration

```bash
python calibrate.py
```

Motors off, arm moved by hand. Put six or more small markers on the table,
click each one in the camera frame, then touch each with the closed finger
tips when prompted. The fit also estimates the joint 3 zero correction, which
Arm applies from calibration.json. Residual should be under 5 mm. Above 10 mm,
targets will miss.

## 5. Motion, low speed

With `speed_percent=10`, run `arm.home()` and watch. The arm should move
smoothly, without jerks at the start.

## 6. Grasp check

```bash
python grasp_check.py
```

Hand the cube to the open gripper when asked, then take it away. `arm.holding()`
must be True with the cube between the fingers and False after closing on
nothing. The runs use this, not vision, to confirm a grasp.

Observed 2026-09-18 with the 29 mm cube: the opening reads 34 mm (0.49 of the
70 mm stroke) while holding it, and 0.0 closed on nothing. The reading is about
5 mm high under load, from play in the gripper drive once the pads press on the
object; it is exact when the fingers are free, so the zero is fine and there is
nothing to calibrate. Thresholds in `Arm` allow for it: held is an opening of
0.15 to 0.85, jammed is above 0.6 (a corner of this cube reads about 0.66).
Check these two numbers again for a different object.
