import os

# --- Archived variant. Connection settings and paths parameterised. ----
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
SLACK_BOT_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "")
# -----------------------------------------------------------------------
import sys
import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm

# argv: 16 numbers for T (row-major) + 7 numbers for q_start
data = np.array(list(map(float, sys.argv[1:])))
T_flat = data[:16]
q_start = data[16:]

T_mat = T_flat.reshape(4, 4)
T_goal = sm.SE3(T_mat)

try:
    robot = rtb.models.ETS.Panda
except Exception:
    robot = rtb.models.DH.Panda

def solve_ik(T, q0):
    sol = robot.ikine_LM(T, q0=q0, ilimit=300, slimit=150, tol=1e-6)
    if sol.success:
        return sol.q
    sol = robot.ikine_LM(T, q0=q0, mask=[1, 1, 1, 0, 0, 1],
                         ilimit=300, slimit=150, tol=1e-6)
    if sol.success:
        return sol.q
    return None

q_target = solve_ik(T_goal, q_start)

if q_target is None:
    print("IK_FAIL")
    sys.exit(1)

print(" ".join(map(str, q_target)))
