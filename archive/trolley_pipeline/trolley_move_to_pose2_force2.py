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
import panda_py
import numpy as np
import logging
import panda_py.libfranka
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from panda_py import libfranka
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)

# Connect to the Panda robot
hostname = PANDA_HOSTNAME
panda = panda_py.Panda(hostname)


gripper = libfranka.Gripper(hostname)


gripper.move(0.06, 0.2)

# Move the robot to the desired Cartesian coordinates and orientation
speed_factor = 0.05

q_saved = [[0.3205217592860453, 0.14412777348684203, -0.10104022437961484, -1.9221173760598163, 0.018403565767730303, 2.067896595060743, 1.0217988612709774]]


panda.move_to_joint_position(q_saved, speed_factor=speed_factor)

# Function to convert Euler angles to a rotation matrix
def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix()

position = [0.58, 0.132, 0.313]

euler_angles = [-180, 0,-1.4]  # roll, pitch, yaw
# Get the rotation matrix from Euler angles
rotation_matrix = euler_to_rotation_matrix(*euler_angles)

# Get the current end-effector pose
pose = panda.get_pose()

# Set the desired Cartesian coordinates
pose[0, 3] = position[0]  # x-coordinate
pose[1, 3] = position[1]  # y-coordinate
pose[2, 3] = position[2]  # z-coordinate

# Set the desired orientation from the rotation matrix
pose[0, 0:3] = rotation_matrix[0, 0:3]
pose[1, 0:3] = rotation_matrix[1, 0:3]
pose[2, 0:3] = rotation_matrix[2, 0:3]


stiffness = 1.5*np.array([600, 600, 600, 600, 250, 150, 50])


position = [0.58, 0.132, 0.235]
#position = [0.695, 0.13, 0.174]
euler_angles = [-180, 0.0,-1.4]  # roll, pitch, yaw
rotation_matrix = euler_to_rotation_matrix(*euler_angles)
pose[0, 3] = position[0]  # x-coordinate
pose[1, 3] = position[1]  # y-coordinate
pose[2, 3] = position[2]  # z-coordinate
pose[0, 0:3] = rotation_matrix[0, 0:3]
pose[1, 0:3] = rotation_matrix[1, 0:3]
pose[2, 0:3] = rotation_matrix[2, 0:3]
panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)

gripper = libfranka.Gripper(hostname)

   
gripper.grasp(0.05, 0.02, 50, 0.04, 0.04)

current_position = panda.get_position()
new_position = current_position.copy()
new_position[2] += 0.07
panda.move_to_pose(new_position, panda.get_orientation(),  speed_factor=speed_factor, stiffness=stiffness)

class ForceTorqueListener(Node):
    def __init__(self):
        super().__init__('force_torque_listener')
        self.subscription = self.create_subscription(
            WrenchStamped,
            '/robotiq_force_torque_sensor_broadcaster/wrench',
            self.listener_callback,
            10)
        self.force_x = []
        self.force_y = []
        self.force_z = []
        self.torque_x = []
        self.torque_y = []
        self.torque_z = []
        self.time_stamps = []
        self.recording = False  # To control recording

    def listener_callback(self, msg):
        if self.recording:  # Only record data while recording is True
            current_time = time.time() - self.start_time
            self.time_stamps.append(current_time)
            self.force_x.append(msg.wrench.force.x)
            self.force_y.append(msg.wrench.force.y)
            self.force_z.append(msg.wrench.force.z)
            self.torque_x.append(msg.wrench.torque.x)
            self.torque_y.append(msg.wrench.torque.y)
            self.torque_z.append(msg.wrench.torque.z)

    def start_recording(self):
        self.recording = True
        self.start_time = time.time()

    def stop_recording(self):
        self.recording = False

    def plot_data(self):
        # Plotting force data
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.time_stamps, self.force_x, label='Force X')
        plt.plot(self.time_stamps, self.force_y, label='Force Y')
        plt.plot(self.time_stamps, self.force_z, label='Force Z')
        plt.title('Force vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend()

        # Plotting torque data
        plt.subplot(2, 1, 2)
        plt.plot(self.time_stamps, self.torque_x, label='Torque X')
        plt.plot(self.time_stamps, self.torque_y, label='Torque Y')
        plt.plot(self.time_stamps, self.torque_z, label='Torque Z')
        plt.title('Torque vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()

        plt.tight_layout()
        plt.savefig('force_new.png')
        plt.show()

import threading, time, numpy as np, rclpy

def move_robot_down(stop_flag, motion_done):
    """Moves robot down incrementally, can be stopped mid-way."""
    speed_factor = 0.03
    step_size = 0.01
    num_steps = 8
    print(f"[INFO] Moving down in {num_steps} small steps...")
    start_time = time.time()
    try:
        for i in range(num_steps):
            if stop_flag.is_set():
                print(f"[INFO] Movement stopped early at step {i}/{num_steps}")
                break
            current_position = panda.get_position()
            new_position = current_position.copy()
            new_position[2] -= step_size
            panda.move_to_pose(
                new_position,
                panda.get_orientation(),
                speed_factor=speed_factor,
                stiffness=stiffness,
            )
            time.sleep(0.02)
        print(f"[INFO] Downward motion complete ({time.time() - start_time:.2f}s).")
    finally:
        motion_done.set()   # < -  tell the monitor we’re done (success or early stop)


def monitor_force(listener, stop_flag, motion_done, threshold_N=10.0):
    """Continuously monitors Fz; stops on threshold OR when motion finishes."""
    start_time = time.time()
    while (not stop_flag.is_set()) and (not motion_done.is_set()):
        if listener.force_z:
            Fz = listener.force_z[-1]
            elapsed = time.time() - start_time
            if abs(Fz) > threshold_N:
                print(f"[ALERT] Fz exceeded {Fz:.2f} N at {elapsed:.2f}s")
                stop_flag.set()
                break
        time.sleep(0.01)


def move_robot_with_recording(listener):
    """Combines motion + recording + continuous force monitor."""
    listener.start_recording()
    time.sleep(0.2)  # small warm-up so the subscriber starts filling

    stop_flag = threading.Event()
    motion_done = threading.Event()

    # Start movement and monitoring in parallel
    move_thread = threading.Thread(target=move_robot_down, args=(stop_flag, motion_done))
    monitor_thread = threading.Thread(target=monitor_force, args=(listener, stop_flag, motion_done))

    move_thread.start()
    monitor_thread.start()

    # Spin ROS listener while threads are running
    while move_thread.is_alive() or monitor_thread.is_alive():
        rclpy.spin_once(listener, timeout_sec=0.05)

    move_thread.join()
    monitor_thread.join()

    listener.stop_recording()

    if stop_flag.is_set():
        print("[INFO] Moving up for safety...")
        current_position = panda.get_position()
        new_position = current_position.copy()
        new_position[2] += 0.03
        panda.move_to_pose(new_position, panda.get_orientation(), speed_factor=0.05)
        print("[INFO] Safety recovery complete.")
    else:
        print("[INFO] Normal motion completed.")

# --- Main ---
rclpy.init()
listener = ForceTorqueListener()

move_robot_with_recording(listener)

listener.destroy_node()
rclpy.shutdown()       # <-- shutdown ROS first
listener.plot_data()   # <-- then call plot_data()
