import os

# --- Connection settings (override in your shell or a .env file) -------
PANDA_HOSTNAME = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST = os.environ.get("INFERENCE_HOST", "127.0.0.1")
# -----------------------------------------------------------------------
import time
import logging
import numpy as np
import panda_py
from panda_py import libfranka

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MoveToSlot")


#====================================================
# Robot Configuration
#====================================================
HOSTNAME = PANDA_HOSTNAME
SPEED_FACTOR = 0.05
MIDPOINT_PAUSE = 0.8
STIFFNESS = np.array([600, 600, 600, 600, 250, 150, 50])

# Intermediate midpoints (approach path)
MIDPOINTS = [
    [1.1545740926349684, -0.347583938460601, -0.46481066047163194,
      -2.3278333051283444, -0.19853817572196414, 2.0591041994624666, 1.6150751256715847],

    [0.6040359361212861, -0.37890707754599845, -0.24579776216112706,
      -2.4519961366291256, -0.07341820777891514, 2.113998993570137, 1.2283624733653384],

    [0.6095732961711157, -0.14951755506094894, -0.30901473061200774,
      -2.2111904554869004, -0.059756699750820826, 2.0878790935410394, 1.1706965011813575],
]

# Final joint configurations for slot positions
SLOT_POSES = {
    "slot1": [0.48963496475189544, 0.26933636016823664, -0.06863498265387719,
            -1.7433215872321275, 0.020747175650081346, 2.0250543769730456, 1.2241563333301657],

    "slot2": [0.3205217592860453, 0.14412777348684203, -0.10104022437961484,
            -1.9221173760598163, 0.018403565767730303, 2.067896595060743, 1.0217988612709774],

    "slot3": [-0.19833941673395947, 0.09932508212955372, 0.20187099244537599,
            -1.977616179565765, -0.021922961064642605, 2.086272546745287, 0.8228694997686479],

    "slot4": [0.2728261362697688, 0.5584861971037983, 0.14063541551197803,
            -1.306950127936246, -0.07928370873651076, 1.8721278858714665, 1.2240289919844853],

    "slot5": [-0.13511812622505323, 0.4464207150249187, 0.4063089615618867,
            -1.5218498452337164, -0.1819293333891888, 1.9405499319354302, 1.1089574566513638],

    "slot6": [-0.2184935975325735, 0.3853007004878614, 0.2577071130944971,
            -1.588209605869444, -0.10373740682982607, 1.970007414870792, 0.8694439157803718]
,
}

# Also accept numeric input: "1", "2", ...
for i in range(1, 7):
    SLOT_POSES[str(i)] = SLOT_POSES[f"slot{i}"]


#====================================================
# Helper Functions
#====================================================
def normalize_slot_key(name: str) -> str:
    """Normalize input like 'Slot 3' → 'slot3'"""
    return name.strip.lower.replace(" ", "")


def move_via_midpoints(robot: panda_py.Panda):
    """Move robot through predefined approach trajectory midpoints."""
    for idx, q in enumerate(MIDPOINTS, 1):
        logger.info(f"Moving to midpoint {idx} ...")
        robot.move_to_joint_position(q, speed_factor=SPEED_FACTOR)
        time.sleep(MIDPOINT_PAUSE)


#====================================================
# Main Function
#====================================================
def move_to_slot(slot_name: str) -> None:
    """
    Move robot to predefined “slot1–slot6”.
    Supports input: 'slot3', 'Slot 3', '3'
    """
    try:
        key = normalize_slot_key(slot_name)
        if key not in SLOT_POSES:
            raise ValueError(f"Unknown slot '{slot_name}'. Expected 1–6 or slot1–slot6.")

        panda = panda_py.Panda(HOSTNAME)

        # Lift slightly before starting approach
        current_pos = panda.get_position
        new_pos = current_pos.copy
        new_pos[2] += 0.1
        panda.move_to_pose(new_pos, panda.get_orientation,
                           speed_factor=SPEED_FACTOR, stiffness=STIFFNESS)

        move_via_midpoints(panda)

        target_q = SLOT_POSES[key]
        logger.info(f"Moving to {slot_name} ...")
        panda.move_to_joint_position(target_q, speed_factor=SPEED_FACTOR)

        logger.info("Slot movement complete.")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    except Exception as exc:
        logger.error(f"Error during movement: {exc}")
