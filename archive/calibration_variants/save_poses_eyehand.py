import os

# --- Connection settings (override in your shell or a .env file) -------
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
# -----------------------------------------------------------------------
# save_ee_pose.py
import panda_py
import numpy as np
import csv
from datetime import datetime

hostname = PANDA_HOSTNAME
panda = panda_py.Panda(hostname)

def save_pose_csv(filename="ee_poses_spring.csv"):
    pose = panda.get_pose()  # 4x4 numpy array
    print(pose)
    pose_flat = pose.flatten()
    timestamp = datetime.now().isoformat()
    row = [timestamp] + pose_flat.tolist()
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"[] EE pose saved at {timestamp}")

if __name__ == "__main__":
    save_pose_csv()
