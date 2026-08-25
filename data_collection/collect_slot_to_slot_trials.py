"""Generate the trolley test set by cycling a rack between deck slots.

Picks from each of the six slots and places into another with a random offset,
producing evaluation data in the same layout as the training collectors but
from poses the models were never trained on.
"""

import panda_py
import numpy as np
from panda_py import libfranka
from scipy.spatial.transform import Rotation as R
import random
# --- Connection settings -----------------------------------------------
# Set these in your shell or a .env file; see .env.example at the repo root.
import os as _os
PANDA_HOSTNAME = _os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
# -----------------------------------------------------------------------

# ---------------- CONFIG ----------------
hostname = PANDA_HOSTNAME
panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)
gripper.move(0.07, 0.2)

speed_pick  = 0.04
speed_place = 0.05
stiffness = np.array([600, 600, 600, 600, 250, 150, 50])

# Base positions (x, y, z) for 6 slots
SLOTS = {
    "slot1": [0.58, 0.264, 0.311],
    "slot2": [0.577, 0.132, 0.311],
    "slot3": [0.577, 0.000, 0.311],
    "slot4": [0.667, 0.264, 0.311],
    "slot5": [0.671, 0.132, 0.311],
    "slot6": [0.667, 0.000, 0.311],
}

# keep angles in DEGREES (your format)
def euler_to_R_deg(roll_deg, pitch_deg, yaw_deg):
    return R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix

def move_to_pose_deg(x, y, z, euler_deg, speed):
    pose = panda.get_pose
    pose[:3, 3] = [x, y, z]
    pose[:3, :3] = euler_to_R_deg(*euler_deg)
    panda.move_to_pose(pose, speed_factor=speed, stiffness=stiffness)

# ---------------- MAIN ----------------
slot_name = "slot5"  # choose one: "slot1".."slot6"


base = np.array(SLOTS[slot_name], dtype=float)  # [x, y, z]

# random offsets
dx     = 0.005#random.uniform(-0.000, 0.008)  # pickup: x-perturbation only
dxx    = 0*random.uniform(-0.012, 0.012)  # place: x-perturbation
dyy    = 0*random.uniform(-0.012, 0.012)  # place: y-perturbation
dtheta = 0*random.uniform(-3.0, 3.0)      # place: yaw (deg)

print(f"{slot_name}  |  dx={dx:.6f}, dxx={dxx:.6f}, dyy={dyy:.6f}, dtheta={dtheta:.5f}°")

# your angle format in DEGREES
euler_pick  = [-180, 0.0, -1.4]
euler_place = [-180, 0.0, -1.4 + dtheta]

# ---------- PICK at (base + dx, same y/z) ----------
pos_pick_above = base.copy; pos_pick_above[0] += dx
move_to_pose_deg(*pos_pick_above, euler_pick, speed_pick)

pos_pick_down = pos_pick_above.copy; pos_pick_down[2] -= 0.08
move_to_pose_deg(*pos_pick_down, euler_pick, speed_pick)

# close gripper to grasp
gripper.grasp(width=0.05, speed=0.02, force=20, epsilon_inner=0.04, epsilon_outer=0.04)

# lift
pos_lift = pos_pick_down.copy; pos_lift[2] += 0.08
move_to_pose_deg(*pos_lift, euler_pick, speed_pick)

# ---------- PLACE at SAME SLOT with (dxx, dyy, dtheta) ----------
pos_place = base.copy
pos_place[0] += dxx
pos_place[1] += dyy
move_to_pose_deg(*pos_place, euler_place, speed_place)

