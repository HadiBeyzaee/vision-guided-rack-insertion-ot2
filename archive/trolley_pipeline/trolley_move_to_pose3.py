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

# Configure logging
logging.basicConfig(level=logging.INFO)

# Connect to the Panda robot
hostname = PANDA_HOSTNAME
panda = panda_py.Panda(hostname)

# Move the robot to the desired Cartesian coordinates and orientation
speed_factor = 0.03


def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix()

position = [0.577, -0.002, 0.313]

euler_angles = [-180, 0,-1.4]  # roll, pitch, yaw
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


gripper.move(0.07, 0.2)

panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)

position = [0.577, -0.002, 0.235]
#position = [0.695, 0.13, 0.174]
euler_angles = [-180, 0.0,-1.4]  # roll, pitch, yaw
rotation_matrix = euler_to_rotation_matrix(*euler_angles)
pose[0, 3] = position[0]  # x-coordinate
pose[1, 3] = position[1]  # y-coordinate
pose[2, 3] = position[2]  # z-coordinate
pose[0, 0:3] = rotation_matrix[0, 0:3]
pose[1, 0:3] = rotation_matrix[1, 0:3]
pose[2, 0:3] = rotation_matrix[2, 0:3]
panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)

gripper = libfranka.Gripper(hostname)

gripper.grasp(0.05, 0.02, 15, 0.04, 0.04)

current_position = panda.get_position()
new_position = current_position.copy()
new_position[2] += 0.08
panda.move_to_pose(new_position, panda.get_orientation(),  speed_factor=speed_factor, stiffness=stiffness)

speed_factor = 0.03
current_position = panda.get_position()
new_position = current_position.copy()
new_position[2] -= 0.077
panda.move_to_pose(new_position, panda.get_orientation(),  speed_factor=speed_factor, stiffness=stiffness)

