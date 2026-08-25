import os

# --- Connection settings (override in your shell or a .env file) -------
PANDA_HOSTNAME = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST = os.environ.get("INFERENCE_HOST", "127.0.0.1")
# -----------------------------------------------------------------------
import logging
import numpy as np
import panda_py
from panda_py import libfranka
from scipy.spatial.transform import Rotation as R


def pick_rack(rack_color="white"):
    """
    Move Panda robot to pick a rack of specified color.
    Supported values: 'blue', 'transparent', 'white', 'black'
    """

    logging.basicConfig(level=logging.INFO)

    hostname = PANDA_HOSTNAME
    panda = panda_py.Panda(hostname)

    speed = 0.06

    # Stage 1: Move above rack approach area
    panda.move_to_joint_position([1.6302882202703461, -0.3869230633451228, -0.22301009356938106,
                                -2.131014741860183, -0.11868175496657687, 1.784126625114017,
                                2.216848552551014], speed_factor=0.06)

    panda.move_to_joint_position([1.8509402765056544, 0.2452749267760773, -0.4539188752024905,
                                -1.8670797690843277, 0.11360158467510313, 2.12655499422588,
                                2.1396939033725197], speed_factor=0.06)


    # Stage 2: Move to the correct color rack area
    rack_positions = {
        "blue": (
            [1.518974692561906, 0.38147112919573195, 0.20305326368715199,
             -1.7846989695063813, -0.08960567940998324, 2.165450320296817,
             2.567260927603477],
            [-0.083, 0.637, 0.142]
        ),
        "transparent": (
            [1.2965813023719173, 0.37163063789086165, 0.25750702109671475,
             -1.804711330185138, -0.11138214733365903, 2.1706292444155637,
             2.4086684522743713],
            [0.028, 0.637, 0.142]
        ),
        "white": (
            [1.7954133189937524, 0.388119922115886, -0.48966008185164095,
             -1.776560034336966, 0.16377048800365923, 2.154817649576399,
             2.007632598753564],
            [0.142, 0.637, 0.142]
        ),
        "black": (
            [1.6948112014887626, 0.5282052480547051, -0.5996896277228965,
             -1.6223245750226414, 0.295944362165455, 2.107331738524967,
             1.844226034093665],
            [0.256, 0.637, 0.144]
        )
    }

    if rack_color not in rack_positions:
        raise ValueError(f"Unsupported rack color: '{rack_color}'. "
                         f"Use: {list(rack_positions.keys)}")

    joint_pose, target_pos = rack_positions[rack_color]
    panda.move_to_joint_position(joint_pose, speed_factor=speed)

    # Stage 3: Orientation and grasp pose
    euler_angles = [-180, 0.0, 0.0]  # roll, pitch, yaw
    rotation = R.from_euler("xyz", euler_angles, degrees=True).as_matrix

    pose = panda.get_pose
    pose[:3, :3] = rotation
    pose[:3, 3] = target_pos

    stiffness = np.array([600, 600, 600, 600, 250, 150, 50])
    panda.move_to_pose(pose, speed_factor=speed, stiffness=stiffness)

    # Grasp the rack
    gripper = libfranka.Gripper(hostname)
    gripper.grasp(width=0.05, speed=0.02, force=15,
                  epsilon_inner=0.04, epsilon_outer=0.04)

    logging.info(f"Rack pick completed: {rack_color}")


if __name__ == "__main__":
    pick_rack("white")
