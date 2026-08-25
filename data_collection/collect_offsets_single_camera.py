"""As collect_offsets_dual_camera.py, but recording camera 2 only.

Used once camera 1 stopped contributing to model accuracy - halves the storage
and removes the two-camera synchronisation wait from the capture loop.
"""

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
# --- Connection settings -----------------------------------------------
# Set these in your shell or a .env file; see .env.example at the repo root.
import os as _os
PANDA_HOSTNAME = _os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
# -----------------------------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO)

# Connect to the Panda robot
hostname = PANDA_HOSTNAME
panda = panda_py.Panda(hostname)

# Function to convert Euler angles to a rotation matrix
def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix

# Setup ROS2 node for camera images
class ImageSubscriber(Node):
    def __init__(self):
        super.__init__('image_subscriber')
        self.bridge = CvBridge
        self.image1 = None

        self.depth_image1 = None

        self.new_image1 = False

        self.new_depth1 = False


        self.subscription1 = self.create_subscription(
            Image,
            '/camera2/camera2/color/image_raw',
            self.image_callback1,
            10)


        # Depth image subscriptions
        self.depth_subscription1 = self.create_subscription(
            Image,
            '/camera2/camera2/depth/image_rect_raw',
            self.depth_callback1,
            10)


    def image_callback1(self, msg):
        try:
            self.image1 = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.new_image1 = True
        except CvBridgeError as e:
            self.get_logger.error(f'CvBridge Error: {e}')


    def depth_callback1(self, msg):
        try:
            self.depth_image1 = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            self.new_depth1 = True
        except CvBridgeError as e:
            self.get_logger.error(f'CvBridge Error: {e}')


    def reset_flags(self):
        self.new_image1 = False

        self.new_depth1 = False


def save_images(color_image1, depth_image1, index, base_dir):
    # Directory paths
    color_dir1 = os.path.join(base_dir, 'color_images', 'camera2')

    gray_dir1 = os.path.join(base_dir, 'grayscale_depth_images', 'camera2')

    color_mapped_dir1 = os.path.join(base_dir, 'color_mapped_depth_images', 'camera2')


    # Ensure directories exist
    os.makedirs(color_dir1, exist_ok=True)

    os.makedirs(gray_dir1, exist_ok=True)

    os.makedirs(color_mapped_dir1, exist_ok=True)


    # Save color images
    color_image1_filename = os.path.join(color_dir1, f'{index}.png')

    cv2.imwrite(color_image1_filename, color_image1)


    # Process and save grayscale depth images
    def process_and_save_depth_images(depth_image, gray_dir, color_mapped_dir, index):
        # Clipping depth values to a relevant range dynamically based on percentiles
        lower_percentile = 5
        upper_percentile = 95
        min_depth = np.percentile(depth_image, lower_percentile)
        max_depth = np.percentile(depth_image, upper_percentile)
        depth_image_clipped = np.clip(depth_image, min_depth, max_depth)

        # Compute alpha dynamically based on the clipped depth range
        alpha = 255.0 / (max_depth - min_depth)
        cv_image_8u = cv2.convertScaleAbs(depth_image_clipped - min_depth, alpha=alpha)

        grayscale_filename = os.path.join(gray_dir, f'{index}.png')
        cv2.imwrite(grayscale_filename, cv_image_8u)

        # Apply histogram equalization to enhance contrast
        if cv_image_8u.max > 0:
            cv_image_8u = cv2.equalizeHist(cv_image_8u)
        # Apply a color map to the grayscale image
        cv_image_color = cv2.applyColorMap(cv_image_8u, cv2.COLORMAP_JET)

        # Overlay the edges on the colorized image
        #cv_image_combined = cv2.addWeighted(cv_image_color, 0.8, edges_bgr, 0.2, 0)

        # Save the color-mapped depth image with edges
        color_mapped_filename = os.path.join(color_mapped_dir, f'{index}.png')
        cv2.imwrite(color_mapped_filename, cv_image_color)

    # Process and save depth images for both cameras
    process_and_save_depth_images(depth_image1, gray_dir1, color_mapped_dir1, index)




# Initialize ROS2
rclpy.init(args=None)
image_subscriber = ImageSubscriber

executor = rclpy.executors.SingleThreadedExecutor
executor.add_node(image_subscriber)

# Base directory for saving images
base_dir = 'presentation_images'
os.makedirs(base_dir, exist_ok=True)

error_data_path = os.path.join(base_dir, 'error_data.txt')

def save_error_data(error_data_path, error_data):
    with open(error_data_path, 'a') as f:
        f.write(" ".join(map(str, error_data)) + "\n")

# Desired position and orientation
position = [0.58, 0.128, 0.313]

euler_angles = [-180, 0, -1.4]  # roll, pitch, yaw
rotation_matrix = euler_to_rotation_matrix(*euler_angles)

# Get the current end-effector pose
pose = panda.get_pose

# Set the initial desired Cartesian coordinates and orientation
pose[0, 3] = position[0]
pose[1, 3] = position[1]
pose[2, 3] = position[2]
pose[0, 0:3] = rotation_matrix[0, 0:3]
pose[1, 0:3] = rotation_matrix[1, 0:3]
pose[2, 0:3] = rotation_matrix[2, 0:3]

# Move the robot to the initial position
speed_factor = 0.03

gripper = libfranka.Gripper(hostname)

panda.move_to_pose(pose, speed_factor=speed_factor)

# Loop to apply random offsets, capture images, and save data
for i in range(100):
    # Generate random offsets between -0.05 and 0.05
    dx = random.uniform(-0.012, 0.012)  # -0.05 to 0.05 meters
    dy = random.uniform(-0.012, 0.012)  # -0.05 to 0.05 meters
    #dz = random.uniform(-0.01, 0.01)
    dtheta = random.uniform(-5.0, 5.0)    # -5 to 5 degrees

    error_data = [dx, dy,  dtheta]

    pose[0, 3] = position[0] + dx
    pose[1, 3] = position[1] + dy
    pose[2, 3] = position[2]

    # Apply offset to the roll angle
    #current_euler = R.from_matrix(pose[0:3, 0:3]).as_euler('xyz', degrees=True)
    #print(current_euler[2])
    euler_angles[2] =  -1.4+dtheta
    euler_angles[1] = 0
    euler_angles[0] = -180

    print(euler_angles)

    new_rotation_matrix = euler_to_rotation_matrix(*euler_angles)
    pose[0, 0:3] = new_rotation_matrix[0, 0:3]
    pose[1, 0:3] = new_rotation_matrix[1, 0:3]
    pose[2, 0:3] = new_rotation_matrix[2, 0:3]

    try:
        def count_row(filename):
            with open(filename, 'r') as file:
                row_count = 0
                for line in file:
                    if line.strip:
                        row_count += 1
            return row_count

        stiffness = 2*np.array([700, 700, 700, 700, 350, 250, 100])
        panda.move_to_pose(pose, speed_factor=speed_factor, stiffness = stiffness)

        # Reset image flags and spin ROS2 executor to process image callbacks
        image_subscriber.reset_flags
        while not (image_subscriber.new_image1 and image_subscriber.new_depth1):
            rclpy.spin_once(image_subscriber, timeout_sec=0.1)

        if image_subscriber.image1 is not None and image_subscriber.depth_image1 is not None:

            save_error_data(error_data_path, error_data)
            num_of_rows = count_row(error_data_path)
            number_of_image = int(num_of_rows)

            # Save images using the dedicated function
            save_images(image_subscriber.image1, image_subscriber.depth_image1, number_of_image, base_dir)

    except Exception as e:
        logging.error(f'Error during iteration {i}: {e}')

# Shutdown ROS2
image_subscriber.destroy_node
rclpy.shutdown
