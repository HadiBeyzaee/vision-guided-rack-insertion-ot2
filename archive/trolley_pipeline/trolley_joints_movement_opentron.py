import os

# --- Connection settings (override in your shell or a .env file) -------
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
# -----------------------------------------------------------------------
import time
import logging
import panda_py
from panda_py import libfranka

logging.basicConfig(level=logging.INFO)

# ---------------- Robot ----------------
HOSTNAME = PANDA_HOSTNAME
panda = panda_py.Panda(HOSTNAME)
SPEED = 0.05
MIDPOINT_PAUSE_S = 0.8


MIDPOINT_1 = [1.401717395086952, -0.1582481132057126, -0.2477848635284524, -2.0508495509884037, -0.025922474119465994, 1.9282702525986593, 0.32094156090997983]

MIDPOINT_2 = [1.1545740926349684, -0.347583938460601, -0.46481066047163194, -2.3278333051283444, -0.19853817572196414, 2.0591041994624666, 1.6150751256715847]

MIDPOINT_3 = [0.6040359361212861, -0.37890707754599845, -0.24579776216112706, -2.4519961366291256, -0.07341820777891514, 2.113998993570137, 1.2283624733653384]

MIDPOINT_4 =  [0.6095732961711157, -0.14951755506094894, -0.30901473061200774, -2.2111904554869004, -0.059756699750820826, 2.0878790935410394, 1.1706965011813575]

# ---------------- Midpoints (approach path) ----------------

MIDPOINTS = [MIDPOINT_1, MIDPOINT_2, MIDPOINT_3, MIDPOINT_4]
POSE_SLOT_1 =  [0.48963496475189544, 0.26933636016823664, -0.06863498265387719, -1.7433215872321275, 0.020747175650081346, 2.0250543769730456, 1.2241563333301657]
POSE_SLOT_2 = [0.3205217592860453, 0.14412777348684203, -0.10104022437961484, -1.9221173760598163, 0.018403565767730303, 2.067896595060743, 1.0217988612709774]
POSE_SLOT_3 =  [-0.19833941673395947, 0.09932508212955372, 0.20187099244537599, -1.977616179565765, -0.021922961064642605, 2.086272546745287, 0.8228694997686479]
POSE_SLOT_4 =  [0.2728261362697688, 0.5584861971037983, 0.14063541551197803, -1.306950127936246, -0.07928370873651076, 1.8721278858714665, 1.2240289919844853]
POSE_SLOT_5 = [-0.13511812622505323, 0.4464207150249187, 0.4063089615618867, -1.5218498452337164, -0.1819293333891888, 1.9405499319354302, 1.1089574566513638]
POSE_SLOT_6 = [-0.2184935975325735, 0.3853007004878614, 0.2577071130944971, -1.588209605869444, -0.10373740682982607, 1.970007414870792, 0.8694439157803718]


SLOT_POSES = {
    "slot1": POSE_SLOT_1, "1": POSE_SLOT_1,
    "slot2": POSE_SLOT_2, "2": POSE_SLOT_2,
    "slot3": POSE_SLOT_3, "3": POSE_SLOT_3,
    "slot4": POSE_SLOT_4, "4": POSE_SLOT_4,
    "slot5": POSE_SLOT_5, "5": POSE_SLOT_5,
    "slot6": POSE_SLOT_6, "6": POSE_SLOT_6,
}

def normalize_slot_key(name: str) -> str:
    """Accept 'slot 3', 'Slot3', '3', etc., and return normalized dict key."""
    return name.strip().lower().replace(" ", "")

def move_through_midpoints(panda, mids, speed=SPEED, pause_s=MIDPOINT_PAUSE_S):
    for i, q in enumerate(mids, 1):
        print(f" Moving via midpoint {i} ...")
        panda.move_to_joint_position(q, speed_factor=speed)
        time.sleep(pause_s)

# ======== CHOOSE YOUR TARGET SLOT HERE ========
slot_name = "slot5"   
# ==============================================

try:
    # 1) Approach path: MIDPOINT_1 -> MIDPOINT_2 -> MIDPOINT_3 -> MIDPOINT_4
    move_through_midpoints(panda, MIDPOINTS)

    # 2) Branch to selected slot
    key = normalize_slot_key(slot_name)
    if key not in SLOT_POSES:
        raise ValueError(f"Unknown slot '{slot_name}'. Use slot1..slot6 or 1..6.")

    target_q = SLOT_POSES[key]
    print(f"Moving to {slot_name} ...")
    panda.move_to_joint_position(target_q, speed_factor=SPEED)

    print("Done.")

except KeyboardInterrupt:
    print("Interrupted by user.")
except Exception as e:
    print(f"Error: {e}")
