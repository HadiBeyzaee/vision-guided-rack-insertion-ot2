"""IK - give it a cartesian pose, get the joint angles back (panda_py.ik).
Handy for checking a pose is reachable before committing to it.
"""

import numpy as np
import panda_py
from panda_py import libfranka
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3

# --- Connection settings -----------------------------------------------
# Set these in your shell or a .env file; see .env.example at the repo root.
import os as _os
hostname = _os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
# -----------------------------------------------------------------------
panda = panda_py.Panda(hostname)

# --- Get current state ---
state = panda.get_state
q_current = np.array(state.q)
print("q_current:", np.round(q_current, 3))

# --- Current pose ---
TCLAM = np.array([[ 1.68607634e-01,  9.85683248e-01, -1.20711383e-16, 6.55407745e-02],
 [ 9.85683248e-01,-1.68607634e-01,  2.06484799e-17,  5.95914594e-01],
 [ 0.00000000e+00, -1.22464680e-16, -1.00000000e+00,  1.30000000e-01],
 [ 0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

print("TCLAM:\n", np.round(TCLAM, 5))

# --- IK function wrapper ---
def ik_solutions(T):
    sols = []
    base_rpy =  [-179.92,   4.48, -98.25]

    # candidate RPY flips (generate multiple orientations that are equivalent)
    candidates = [
        base_rpy,
        [base_rpy[0] + 180, base_rpy[1], base_rpy[2]],
        [base_rpy[0], base_rpy[1] + 180, base_rpy[2]],
        [base_rpy[0], base_rpy[1], base_rpy[2] + 180],
    ]

    for rpy in candidates:
        pose = SE3(T[:3,3]) * SE3.RPY(np.radians(rpy), order='zyx')
        q = panda_py.ik(pose.A)
        if q is not None and not np.isnan(q).any:
            sols.append(q)
    return sols

# --- Collect all candidate IK solutions ---
solutions = ik_solutions(TCLAM)

if not solutions:
    print("No IK solutions found.")
else:
    print(f"Found {len(solutions)} candidate IK solutions.")

    # Pick the one closest to current q
    solutions = np.array(solutions)
    diffs = np.linalg.norm(solutions - q_current, axis=1)
    best_idx = np.argmin(diffs)
    q_best = solutions[best_idx]

    print("q_best (closest IK):", np.round(q_best, 3))
    print("Difference (rad):", np.round(q_best - q_current, 4))
    print("Difference (deg):", np.round(np.degrees(q_best - q_current), 2))

    # Optional: move robot there
    # panda.move_to_joint_position(q_best, speed_factor=0.05)
