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

for trial in range(50):
    print(f"\nTrial {trial + 1}/100")
        
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

    def save_images2(color_image1, depth_image1, index, base_dir):
        # Directory paths
        color_dir1 = os.path.join(base_dir, 'color_images', 'camera1_align')

        color_mapped_dir1 = os.path.join(base_dir, 'color_mapped_depth_images', 'camera1_align')

        # Ensure directories exist
        os.makedirs(color_dir1, exist_ok=True)
        os.makedirs(color_mapped_dir1, exist_ok=True)

        # Save color images
        color_image1_filename = os.path.join(color_dir1, f'{index}.png')
        cv2.imwrite(color_image1_filename, color_image1)
        
        # Process and save grayscale depth images
        def process_and_save_depth_images(depth_image, color_mapped_dir, index):
            # Clipping depth values to a relevant range dynamically based on percentiles
            lower_percentile = 5
            upper_percentile = 95
            min_depth = np.percentile(depth_image, lower_percentile)
            max_depth = np.percentile(depth_image, upper_percentile)
            depth_image_clipped = np.clip(depth_image, min_depth, max_depth)

            # Compute alpha dynamically based on the clipped depth range
            alpha = 255.0 / (max_depth - min_depth)
            cv_image_8u = cv2.convertScaleAbs(depth_image_clipped - min_depth, alpha=alpha)

            # Apply histogram equalization to enhance contrast
            if cv_image_8u.max() > 0:
                cv_image_8u = cv2.equalizeHist(cv_image_8u)
            # Apply a color map to the grayscale image
            cv_image_color = cv2.applyColorMap(cv_image_8u, cv2.COLORMAP_JET)

            # Save the color-mapped depth image with edges
            color_mapped_filename = os.path.join(color_mapped_dir, f'{index}.png')
            cv2.imwrite(color_mapped_filename, cv_image_color)

        # Process and save depth images for both cameras
        process_and_save_depth_images(depth_image1,  color_mapped_dir1, index)

    def save_images(color_image1, depth_image1, index, base_dir):
        # Directory paths
        color_dir1 = os.path.join(base_dir, 'color_images', 'camera1')

        color_mapped_dir1 = os.path.join(base_dir, 'color_mapped_depth_images', 'camera1')

        # Ensure directories exist
        os.makedirs(color_dir1, exist_ok=True)
        os.makedirs(color_mapped_dir1, exist_ok=True)

        # Save color images
        color_image1_filename = os.path.join(color_dir1, f'{index}.png')
        cv2.imwrite(color_image1_filename, color_image1)
        
        # Process and save grayscale depth images
        def process_and_save_depth_images(depth_image, color_mapped_dir, index):
            # Clipping depth values to a relevant range dynamically based on percentiles
            lower_percentile = 5
            upper_percentile = 95
            min_depth = np.percentile(depth_image, lower_percentile)
            max_depth = np.percentile(depth_image, upper_percentile)
            depth_image_clipped = np.clip(depth_image, min_depth, max_depth)

            # Compute alpha dynamically based on the clipped depth range
            alpha = 255.0 / (max_depth - min_depth)
            cv_image_8u = cv2.convertScaleAbs(depth_image_clipped - min_depth, alpha=alpha)

            # Apply histogram equalization to enhance contrast
            if cv_image_8u.max() > 0:
                cv_image_8u = cv2.equalizeHist(cv_image_8u)
            # Apply a color map to the grayscale image
            cv_image_color = cv2.applyColorMap(cv_image_8u, cv2.COLORMAP_JET)

            # Save the color-mapped depth image with edges
            color_mapped_filename = os.path.join(color_mapped_dir, f'{index}.png')
            cv2.imwrite(color_mapped_filename, cv_image_color)

        # Process and save depth images for both cameras
        process_and_save_depth_images(depth_image1,  color_mapped_dir1, index)

    # Initialize ROS2
    rclpy.init(args=None)
    image_subscriber = ImageSubscriber()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(image_subscriber)

    # Base directory for saving images
    base_dir = 'slot6_grasp_base_offset1'
    os.makedirs(base_dir, exist_ok=True)

    error_data_path = os.path.join(base_dir, 'error_data.txt')

    def save_error_data(error_data_path, error_data):
        with open(error_data_path, 'a') as f:
            f.write(" ".join(map(str, error_data)) + "\n")   


    # Function to convert Euler angles to a rotation matrix
    def euler_to_rotation_matrix(roll, pitch, yaw):
        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        return r.as_matrix()


    # if random.choice([True, False]):
    # dx = random.uniform(0.01, 0.01)
    # else:
    # dx = random.uniform(-0.01, -0.01)


    # dx_values_uniform = np.linspace(-0.01, 0.01, 50)  # 50 uniform dx values

    # if trial < 50:
    # dx = dx_values_uniform[trial]
    # else:
    # dx = random.uniform(-0.01, 0.01)  # random dx for later trials


    dx = random.uniform(-0.01, 0.01)

    print(f"\nTrial {trial + 1}/100 | dx = {dx:.5f}")

    position = [0.681 + dx, 0.129, 0.253]
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


    gripper.move(0.06, 0.2)

    panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)

    position = [0.681 + dx, 0.129, 0.17]
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

    gripper.grasp(0.05, 0.02, 400, 0.04, 0.04)


    current_position = panda.get_position()
    new_position = current_position.copy()
    new_position[2] += 0.08
    panda.move_to_pose(new_position, panda.get_orientation(),  speed_factor=speed_factor, stiffness=stiffness)


    #----------------------------------------- above slot 2 ---------------------------------------------------------------

    position = [0.681 , 0.129 , 0.25]
    #position = [0.68 , -0.003 , 0.25]
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


    time.sleep(1.0)
    try:
        print("Starting automatic capture of 20 image pairs...")

        for i in range(150):

            #position = [0.72 , 0.1285, 0.25]
            dxx = random.uniform(-0.01, 0.01)
            dyy = random.uniform(-0.01, 0.01)
            if i > 130:
                dxx = random.uniform(-0.002, 0.002)
                dyy = random.uniform(-0.002, 0.002)
            error_data = [dx, dxx, dyy]  # or whatever you define for dx

            position = [0.681 + dxx, 0.129 + dyy, 0.25]

            #position = [0.68 + dxx, -0.003 + dyy, 0.25]
            euler_angles = [-180, 0,-1.4]  # roll, pitch, yaw
            # Get the rotation matrix from Euler angles
            rotation_matrix = euler_to_rotation_matrix(*euler_angles)

            pose = panda.get_pose()

            # Set the desired Cartesian coordinates
            pose[0, 3] = position[0]  # x-coordinate
            pose[1, 3] = position[1]  # y-coordinate
            pose[2, 3] = position[2]  # z-coordinate
            # Set the desired orientation from the rotation matrix
            pose[0, 0:3] = rotation_matrix[0, 0:3]
            pose[1, 0:3] = rotation_matrix[1, 0:3]
            pose[2, 0:3] = rotation_matrix[2, 0:3]

            speed_factor = 0.05
            panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)
            time.sleep(0.55)
            # Reset flags and wait for both images
            image_subscriber.reset_flags()
            while not (image_subscriber.new_image1 and image_subscriber.new_depth1):
                rclpy.spin_once(image_subscriber, timeout_sec=0.1)

            if image_subscriber.image1 is not None and image_subscriber.depth_image1 is not None:
                # Save the error data
                save_error_data(error_data_path, error_data)

                # Count saved rows to index the image name
                def count_row(filename):
                    with open(filename, 'r') as file:
                        return sum(1 for line in file if line.strip())

                number_of_image = count_row(error_data_path)

                # Save images
                save_images(image_subscriber.image1, image_subscriber.depth_image1, number_of_image, base_dir)

                print(f"Saved sample #{number_of_image}")

            else:
                print(f"Image #{i+1} not ready. Skipping...")

        for i in range(30):
            position = [0.681 + dx, 0.129, 0.25]

            #position = [0.68 + dx, -0.003, 0.25]
            euler_angles = [-180, 0,-1.4]  # roll, pitch, yaw
            # Get the rotation matrix from Euler angles
            rotation_matrix = euler_to_rotation_matrix(*euler_angles)

            pose = panda.get_pose()

            # Set the desired Cartesian coordinates
            pose[0, 3] = position[0]  # x-coordinate
            pose[1, 3] = position[1]  # y-coordinate
            pose[2, 3] = position[2]  # z-coordinate
            # Set the desired orientation from the rotation matrix
            pose[0, 0:3] = rotation_matrix[0, 0:3]
            pose[1, 0:3] = rotation_matrix[1, 0:3]
            pose[2, 0:3] = rotation_matrix[2, 0:3]

            speed_factor = 0.05
            panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)
            time.sleep(0.4)
            # Reset flags and wait for both images
            image_subscriber.reset_flags()
            while not (image_subscriber.new_image1 and image_subscriber.new_depth1):
                rclpy.spin_once(image_subscriber, timeout_sec=0.1)

            if image_subscriber.image1 is not None and image_subscriber.depth_image1 is not None:

                # Save images
                number_of_image = f"{'_' if dx < 0 else ''}{str(dx)[3:]}_{trial}_{i}"


                save_images2(image_subscriber.image1, image_subscriber.depth_image1,  number_of_image, base_dir)

                print(f"Saved sample #{number_of_image}")

            else:
                print(f"Image #{i+1} not ready. Skipping...")
    except KeyboardInterrupt:
        print("\nCapture stopped by user.")

    finally:
        image_subscriber.destroy_node()
        rclpy.shutdown()


    position = [0.681 + dx, 0.128, 0.25]
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


    panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)

    # position = [0.681 + dx, 0.13, 0.173]
    # euler_angles = [-180, 0.0,-1.4]  # roll, pitch, yaw
    # rotation_matrix = euler_to_rotation_matrix(*euler_angles)
    # pose[0, 3] = position[0]  # x-coordinate
    # pose[1, 3] = position[1]  # y-coordinate
    # pose[2, 3] = position[2]  # z-coordinate
    # pose[0, 0:3] = rotation_matrix[0, 0:3]
    # pose[1, 0:3] = rotation_matrix[1, 0:3]
    # pose[2, 0:3] = rotation_matrix[2, 0:3]
    # panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)


    speed_factor = 0.03
    current_position = panda.get_position()
    new_position = current_position.copy()
    new_position[2] -= 0.09
    panda.move_to_pose(new_position, panda.get_orientation(),  speed_factor=speed_factor, stiffness=stiffness)


    gripper = libfranka.Gripper(hostname)

    gripper.move(0.058, 0.2)
