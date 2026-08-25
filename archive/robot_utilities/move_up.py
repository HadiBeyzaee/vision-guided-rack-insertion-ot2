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

# Function to convert Euler angles to a rotation matrix
def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix

speed_factor = 0.01
#stiffness = np.array([600, 600, 600, 600, 250, 150, 50])

# q =  [0.37192881217326546, -0.030705216997402734, -0.2716641324818431, -2.0949646771315544, -0.034720237692213815, 2.0847736915482415, 0.9078737968661719]

# panda.move_to_joint_position(q, speed_factor=0.03)

current_position = panda.get_position
new_position = current_position.copy
new_position[2] -= 0.125
panda.move_to_pose(new_position, panda.get_orientation,  speed_factor=speed_factor)
