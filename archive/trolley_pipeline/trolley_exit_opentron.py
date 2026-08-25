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

stiffness = 1.5*np.array([600, 600, 600, 600, 250, 150, 50])

gripper = libfranka.Gripper(hostname)


gripper.move(0.08, 0.2)

# Above the rack before grasping
q1 =  [0.6095732961711157, -0.14951755506094894, -0.30901473061200774, -2.2111904554869004, -0.059756699750820826, 2.0878790935410394, 1.1706965011813575]
q2 = [0.6040359361212861, -0.37890707754599845, -0.24579776216112706, -2.4519961366291256, -0.07341820777891514, 2.113998993570137, 1.2283624733653384]

panda.move_to_joint_position(q1, speed_factor=speed_factor)
panda.move_to_joint_position(q2, speed_factor=speed_factor)
