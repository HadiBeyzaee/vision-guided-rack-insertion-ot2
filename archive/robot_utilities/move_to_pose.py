import os

# --- Archived variant. Connection settings and paths parameterised. ----
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
SLACK_BOT_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "")
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
gripper = libfranka.Gripper(hostname)
#gripper.move(0.05, 0.1)  # Open gripper

# q =  [0.8140336688716595, -0.1765760058478305, -0.4343708291743767,
# -2.213754286749321, -0.07460837950308843, 2.044259855005476,
# 1.1454152624488427]
# speed_factor = 0.02


# panda.move_to_joint_position(q, speed_factor=speed_factor)

# Return to home
def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix

#position = [0.267, 0.5, 0.173]
# position = [0.27, 0.5, 0.144]

#position = [0.125, 0.598, 0.13]

#position = [-0.026, 0.598, 0.13]

#position = [-0.172, 0.6, 0.13]
# position = [0.16, 0.4, 0.2]

#euler_angles = [179.0, -1.50,89.5]

# position = [-0.173, 0.648, 0.195]

position = [-0.017, 0.55, 0.15]

euler_angles = [180.0, 0,90]

rotation_matrix = euler_to_rotation_matrix(*euler_angles)

pose = panda.get_pose
pose[0, 3] = position[0]
pose[1, 3] = position[1]
pose[2, 3] = position[2]
pose[0, 0:3] = rotation_matrix[0, 0:3]
pose[1, 0:3] = rotation_matrix[1, 0:3]
pose[2, 0:3] = rotation_matrix[2, 0:3]

speed_factor = 0.01
panda.move_to_pose(pose, speed_factor=speed_factor)

#
#
#gripper.grasp(0.05, 0.01, 0.5, 0.025, 0.025)
#gripper.grasp(0.056, 0.01, 1.0, 0.025, 0.025)
# q =  [-0.08089184208966164, 0.07199296008276616, 0.26400353095938256, -1.9076700537890998,
# -0.019922973766305193, 1.9903514703644647, 0.9750838425217403]
# # move inside opentron
# panda.move_to_joint_position(q, speed_factor=0.03)



# q =  [0.39743855652080295, -0.24100679653778406, -0.06517830838263042,
# -2.341520213612752, -0.01295525201904286, 2.0915939679906757,
# 1.1066858238437107]
# speed_factor = 0.05
# # move inside opentron
# panda.move_to_joint_position(q, speed_factor=speed_factor)
