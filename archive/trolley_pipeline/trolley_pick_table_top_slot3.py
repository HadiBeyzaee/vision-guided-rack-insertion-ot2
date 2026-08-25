import os

# --- Connection settings (override in your shell or a .env file) -------
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
# -----------------------------------------------------------------------
import panda_py
import numpy as np
import logging
import panda_py.libfranka
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from panda_py import libfranka
import time
import random

    
# Configure logging
logging.basicConfig(level=logging.INFO)

# Connect to the Panda robot
hostname = PANDA_HOSTNAME
panda = panda_py.Panda(hostname)

# Move the robot to the desired Cartesian coordinates and orientation
speed_factor = 0.05

# panda.move_to_joint_position([1.6245797046670103, -0.11402809741964566, -0.08876788540685361, -1.4589830255597422, -0.07862584180588267, 1.432335729651981, 0.7393831875382199], speed_factor=0.05)

# panda.move_to_joint_position([1.405592972866556, 0.035182558052371894, 0.030527212032514416, -1.8813557672333296, -0.002476126438648679, 1.9326529455714756, 0.6681306372541519], speed_factor=0.05)


# Function to convert Euler angles to a rotation matrix
def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix()

# if rack_color == 'blue' or 'white':
# position = [0.0006, 0.636, 0.14]
# else:
#panda.move_to_joint_position([1.0400525315351652, 0.30537512338418743, 0.6040118927311647, -1.686294190741374, -0.18297076455455014, 1.9470353934388271, 0.9026697534755894], speed_factor=0.05)

panda.move_to_joint_position([1.0325034557810564, 0.44021967446595084, 0.2586991294894302, -1.4646487560067245, -0.10496250588374732, 1.937397437284309, 0.5141607142578062], speed_factor=0.05)

position = [0.225, 0.655, 0.138]
#position = [0.0006, 0.636, 0.138]
euler_angles = [-180, -2,90]  # roll, pitch, yaw
# Get the rotation matrix from Euler angles
rotation_matrix = euler_to_rotation_matrix(*euler_angles)
# Get the current end-effector pose
pose = panda.get_pose()
# Set the desired Cartesian coordinates
pose[0, 3] = position[0]  # x-coordinate
pose[1, 3] = position[1]  # y-coordinate
pose[2, 3] = position[2]  # z-coordinate
# Set the desired orientation from the rotation matrix
pose[0, 0:3] = rotation_matrix[0, 0:3]
pose[1, 0:3] = rotation_matrix[1, 0:3]
pose[2, 0:3] = rotation_matrix[2, 0:3]
stiffness = 1.5*np.array([600, 600, 600, 600, 250, 150, 50])
gripper = libfranka.Gripper(hostname)
panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)

gripper.grasp(0.05, 0.02, 15, 0.04, 0.04)


# gripper.move(0.07, 0.2)

# # Above the rack before grasping
# q1 = [1.6522268642208024, 0.3189756105308374, -0.25968031432791683, -1.9625914591906364, 0.10745561182498932, 2.2812393447396713, 0.5488670073408219]
# panda.move_to_joint_position(q1, speed_factor=speed_factor)

# # Moving downward for grasping
# position = [0.10, 0.60, 0.130]
# euler_angles = [-180, 0.0,90]  # roll, pitch, yaw

# rotation_matrix = euler_to_rotation_matrix(*euler_angles)
# pose[0, 3] = position[0]  # x-coordinate
# pose[1, 3] = position[1]  # y-coordinate
# pose[2, 3] = position[2]  # z-coordinate
# pose[0, 0:3] = rotation_matrix[0, 0:3]
# pose[1, 0:3] = rotation_matrix[1, 0:3]
# pose[2, 0:3] = rotation_matrix[2, 0:3]
# panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)

# gripper = libfranka.Gripper(hostname)

# # Grasp the rack
# gripper.grasp(0.05, 0.02, 15, 0.04, 0.04)

# # Move upward
# current_position = panda.get_position()
# new_position = current_position.copy()
# new_position[2] += 0.07
# panda.move_to_pose(new_position, panda.get_orientation(),  speed_factor=speed_factor, stiffness=stiffness)


# # Move towards the opentron

# q2 = [1.401717395086952, -0.1582481132057126, -0.2477848635284524, -2.0508495509884037, -0.025922474119465994, 1.9282702525986593, 0.32094156090997983]

# q3 = [1.1545740926349684, -0.347583938460601, -0.46481066047163194, -2.3278333051283444, -0.19853817572196414, 2.0591041994624666, 1.6150751256715847]

# q4 = [0.6040359361212861, -0.37890707754599845, -0.24579776216112706, -2.4519961366291256, -0.07341820777891514, 2.113998993570137, 1.2283624733653384]

# q5 =  [0.6095732961711157, -0.14951755506094894, -0.30901473061200774, -2.2111904554869004, -0.059756699750820826, 2.0878790935410394, 1.1706965011813575]

# # q6 =  [1.1296618449646127, 0.21143403378320794, -0.9676127707870397, -1.8773410562716029, 0.19422849208116527, 2.0108957974645825, 0.9085344118335181]

# waypoints = [ q2, q3, q4, q5]


# # Move through them in order
# for i, q in enumerate(waypoints, start=1):
# print(f"Moving to waypoint {i}...")
# panda.move_to_joint_position(q, speed_factor=speed_factor)
# time.sleep(0.1)   # small pause between moves


# # Move to the desired slot for insertion

# SLOTS = {
# "slot1": [0.577, 0.264, 0.313],
# "slot2": [0.577, 0.132, 0.313],
# "slot3": [0.577, 0.000, 0.313],
# "slot4": [0.667, 0.264, 0.313],
# "slot5": [0.667, 0.132, 0.313],
# "slot6": [0.667, 0.000, 0.313],
# }

# # keep angles in DEGREES (your format)
# def euler_to_R_deg(roll_deg, pitch_deg, yaw_deg):
# return R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()

# def move_to_pose_deg(x, y, z, euler_deg, speed):
# pose = panda.get_pose()
# pose[:3, 3] = [x, y, z]
# pose[:3, :3] = euler_to_R_deg(*euler_deg)
# panda.move_to_pose(pose, speed_factor=speed, stiffness=stiffness)

# # ---------------- MAIN ----------------
# slot_name = "slot3" 

# base = np.array(SLOTS[slot_name], dtype=float)  # [x, y, z]

# # your angle format in DEGREES
# euler_pick  = [-180, 0.0, -1.4]

# # ---------- PICK at (base + dx, same y/z) ----------
# pos_pick_above = base.copy()
# speed_factor = 0.03
# move_to_pose_deg(*pos_pick_above, euler_pick, speed_factor)
