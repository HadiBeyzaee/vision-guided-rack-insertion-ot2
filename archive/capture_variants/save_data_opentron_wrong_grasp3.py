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
import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from panda_py import libfranka
from sensor_msgs.msg import Image
import os
import random
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Connect to the Panda robot
hostname = PANDA_HOSTNAME
panda = panda_py.Panda(hostname)

# Function to convert Euler angles to a rotation matrix
def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix()


    
# Setup ROS2 node for camera images
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.image1 = None

        self.depth_image1 = None

        self.new_image1 = False

        self.new_depth1 = False


        self.subscription1 = self.create_subscription(
            Image,
            '/camera1/camera1/color/image_raw',
            self.image_callback1,
            10)

        
        # Depth image subscriptions
        self.depth_subscription1 = self.create_subscription(
            Image,
            '/camera1/camera1/depth/image_rect_raw',
            self.depth_callback1,
            10)


    def image_callback1(self, msg):
        try:
            self.image1 = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.new_image1 = True
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')


    def depth_callback1(self, msg):
        try:
            self.depth_image1 = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            self.new_depth1 = True
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')


    def reset_flags(self):
        self.new_image1 = False

        self.new_depth1 = False

# Initialize ROS2
rclpy.init(args=None)
image_subscriber = ImageSubscriber()

executor = rclpy.executors.SingleThreadedExecutor()
executor.add_node(image_subscriber)

# Base directory for saving images
base_dir = 'opentron_wrong_grasp_base2'
os.makedirs(base_dir, exist_ok=True)

error_data_path = os.path.join(base_dir, 'error_data.txt')

def save_error_data(error_data_path, error_data):
    with open(error_data_path, 'a') as f:
        f.write(" ".join(map(str, error_data)) + "\n")   


# Function to convert Euler angles to a rotation matrix
def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix()



dx = 0*random.uniform(0.01, 0.0)

#position = [0.591 + dx, -0.003, 0.253]

# position = [0.591 + dx, 0.13, 0.253]

position = [0.591 + dx, 0.262, 0.253]

position = [0.681 + dx, 0.13, 0.253]


#position = [0.681 + dx, -0.003, 0.253]

#position = [0.681 + dx, 0.262, 0.253]

euler_angles = [-180, 0,-1.4]  # roll, pitch, yaw
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

# Move the robot to the desired Cartesian coordinates and orientation
speed_factor = 0.05
stiffness = np.array([600, 600, 600, 600, 250, 150, 50])

gripper = libfranka.Gripper(hostname)


gripper.move(0.058, 0.2)

panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)

#position = [0.591 + dx, -0.003, 0.172]

# position = [0.591 + dx, 0.13, 0.172]

position = [0.591 + dx, 0.262, 0.172]

position = [0.681 + dx, 0.13, 0.172]

#position = [0.681 + dx, -0.003, 0.172]

#position = [0.681 + dx, 0.262, 0.172]

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

gripper.grasp(0.05, 0.02, 10, 0.04, 0.04)


current_position = panda.get_position()
new_position = current_position.copy()
new_position[2] += 0.08
panda.move_to_pose(new_position, panda.get_orientation(),  speed_factor=speed_factor, stiffness=stiffness)


#----------------------------------------- above slot 2 ---------------------------------------------------------------

dxx = 0*random.uniform(-0.011, 0.011)
dyy = 0*random.uniform(-0.011, 0.011)

print('dx: ', dx)
print('dxx: ', dxx)
print('dyy: ', dyy)

#position = [0.591 + dxx , -0.003 + dyy , 0.25]

# position = [0.591 + dxx , 0.13 + dyy , 0.25]

position = [0.59 + dxx , 0.262 + dyy , 0.25]

position = [0.681 + dxx , 0.13 + dyy , 0.25]

#position = [0.681 + dxx , -0.003 + dyy , 0.25]

#position = [0.681 + dxx , 0.262 + dyy , 0.25]

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

# Move the robot to the desired Cartesian coordinates and orientation
speed_factor = 0.05
stiffness = np.array([600, 600, 600, 600, 250, 150, 50])

gripper = libfranka.Gripper(hostname)


panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)

