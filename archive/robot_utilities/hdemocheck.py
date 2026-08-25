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
import time

ROBOT_IP  = PANDA_HOSTNAME
DESK_USER = "liverpool_uni"
DESK_PASS = "liverpool1881"    # change to your actual Desk password

# -- Step 1: activate FCI via Desk -------------------------
print("Activating FCI via Desk...")
desk = panda_py.Desk(ROBOT_IP, DESK_USER, DESK_PASS)
desk.unlock
desk.activate_fci
print("FCI active.")

# -- Step 2: connect robot ---------------------------------
robot = panda_py.Panda(ROBOT_IP)
robot.recover
robot.set_default_behavior
print(f"Mode: {robot.get_state.robot_mode}")

# -- Step 3: teaching mode ---------------------------------
robot.teaching_mode(active=True)
robot.enable_logging(buffer_size=100000)
print(f"Mode after teaching_mode: {robot.get_state.robot_mode}")
print("Try moving robot by hand now...")

input("Press Enter to stop\n")

robot.teaching_mode(active=False)
log    = robot.get_log
joints = np.array(log["q"])
print(f"Recorded {len(joints)} steps")
print(f"Movement: {np.round(np.abs(joints[-1]-joints[0]), 3)}")
np.save("test_demo.npy", joints)
