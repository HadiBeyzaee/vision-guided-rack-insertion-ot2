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
from panda_py import libfranka
from scipy.spatial.transform import Rotation as R

hostname = PANDA_HOSTNAME

panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)


print('----------------------')
print(panda.get_pose)
pose = panda.get_pose
print('----------------------')
print(panda.get_position)
print('----------------------')
print(panda.get_orientation)
orientation_quaternion = panda.get_orientation

state = panda.get_state
print("q:", state.q)     # 7 joint positions (rad)
#print(state)

# Convert the quaternion to Euler angles (roll, pitch, yaw)
r = R.from_quat(orientation_quaternion)
euler_angles = r.as_euler('xyz', degrees=True)

# Print the Euler angles
print(f"Current end-effector Euler angles: Roll: {euler_angles[0]}, Pitch: {euler_angles[1]}, Yaw: {euler_angles[2]}")
