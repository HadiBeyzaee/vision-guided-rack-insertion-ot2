import os

# --- Connection settings (override in your shell or a .env file) -------
PANDA_HOSTNAME = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST = os.environ.get("INFERENCE_HOST", "127.0.0.1")
# -----------------------------------------------------------------------
"""
Save Panda robot end-effector pose to a CSV file.
Press Enter to record a pose, or type 'q' to quit.
"""

import panda_py
import numpy as np
import csv
import os
from datetime import datetime

# =========================================================
# Configuration
# =========================================================

HOSTNAME = PANDA_HOSTNAME
OUTPUT_CSV = "ee_pose_log.csv"

# Ensure CSV file exists with header if new
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp"] + [f"T[{i}]" for i in range(16)]
        writer.writerow(header)

# =========================================================
# Functions
# =========================================================

def save_pose():
    """Capture current end-effector pose and append to CSV."""
    pose_matrix = panda.get_pose()  # 4x4 homogeneous transform
    pose_flat = pose_matrix.flatten().tolist()
    timestamp = datetime.now().isoformat()

    with open(OUTPUT_CSV, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp] + pose_flat)

    print(f"Pose saved at {timestamp}")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    try:
        panda = panda_py.Panda(HOSTNAME)
        print(f"Connected to Panda robot at {HOSTNAME}")
        print("Press Enter to record pose, or 'q' then Enter to quit.")

        while True:
            user_input = input("> ")
            if user_input.lower() == "q":
                print("Exiting.")
                break

            save_pose()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Program finished.")
