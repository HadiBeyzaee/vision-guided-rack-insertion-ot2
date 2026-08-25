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
import random

# Configure logging
logging.basicConfig(level=logging.INFO)

# Connect to the Panda robot
hostname = PANDA_HOSTNAME
panda = panda_py.Panda(hostname)

# Move the robot to the desired Cartesian coordinates and orientation
speed_factor = 0.05

stiffness = 1.5*np.array([600, 600, 600, 600, 250, 150, 50])

gripper = libfranka.Gripper(hostname)


gripper.move(0.08, 0.2)

# Above the rack before grasping
q1 =  [-0.576500771990991, 0.1712960520801733, 0.9723060879186809, -1.9065820081065459, -0.08194686988327238, 2.013161342991723, 1.104528565941066]
q2 = [-0.04793107532252345, -0.1976025456190109, 0.5310279979468071, -2.2286992006788684, 0.11628678689400353, 2.0290724452866447, 1.2375890034680517]

panda.move_to_joint_position(q1, speed_factor=speed_factor)
panda.move_to_joint_position(q2, speed_factor=speed_factor)
