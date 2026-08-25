import os
import requests
import numpy as np
import logging
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
import threading
import panda_py
from panda_py import libfranka
from scipy.spatial.transform import Rotation as R
from datetime import datetime



logging.basicConfig(level=logging.INFO)

hostname = PANDA_HOSTNAME
panda = panda_py.Panda(hostname)

FLASK_SERVER_URL = f"http://{INFERENCE_HOST}:4001/predict"

dx_step = 0.0012
dy_step = 0.0011
dtheta_step = 0.1
dz_drop = 0.09
speed_factor = 0.05

initial_roll = -180
initial_pitch = 0

def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix()


dx_step = 0.0012
dy_step = 0.0011
dtheta_step = 0.1
dz_drop = 0.081
speed_factor = 0.05

initial_roll = -180
initial_pitch = 0

TOP_MARGIN = 240
BOTTOM_MARGIN = 320
LEFT_MARGIN = 500
RIGHT_MARGIN = 200


# TOP_MARGIN = 220
# BOTTOM_MARGIN = 250

# LEFT_MARGIN = 450
# RIGHT_MARGIN = 150

# LEFT_MARGIN = 525
# RIGHT_MARGIN = 240

# LEFT_MARGIN = 530
# RIGHT_MARGIN = 240

def crop_image_cv2(image, top=0, bottom=0, left=0, right=0):
    height, width = image.shape[:2]
    return image[top:height - bottom, left:width - right]


import albumentations as A
from albumentations.pytorch import ToTensorV2

# Albumentations augmentation pipeline
augmentations = A.Compose([
    # A.Resize(height=100, width=250),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
    A.RandomGamma(gamma_limit=(80, 120), p=0.8),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    A.ToGray(p=0.2),
    A.Solarize(threshold=192, p=0.2),
    A.InvertImg(p=0.2),
])

# ---- Sobel edge detection augmentation ----
def sobel_edges(gray_img, ksize=3, alpha=1.5):
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.convertScaleAbs(sobel_norm, alpha=alpha, beta=0)


class DualSubscriptionNode(Node):
    def __init__(self):
        super().__init__('dual_sub_node')
        self.bridge = CvBridge()

        # --- State ---
        self.last_prediction_time = time.time()
        self.final_lowering_done = False
        self.final_timer = None
        self.no_move_count = 0
        self.last_no_move_stamp = None

        # For overlay text
        self.latest_prediction_text = "Waiting for prediction..."
        self.prediction_lock = threading.Lock()

        # --- Callback Groups ---
        self.video_cb_group = ReentrantCallbackGroup()
        self.pred_cb_group = MutuallyExclusiveCallbackGroup()

        # --- Subscriptions ---
        self.video_sub = self.create_subscription(
            Image,
            '/camera2/camera2/color/image_raw',
            self.video_callback,
            10,
            callback_group=self.video_cb_group
        )

        self.pred_sub = self.create_subscription(
            Image,
            '/camera2/camera2/color/image_raw',
            self.prediction_callback,
            10,
            callback_group=self.pred_cb_group
        )

        self.get_logger().info("DualSubscriptionNode initialized (Multi-Thread, no video recording).")

    def video_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            resized = cv2.resize(cv_image, (640, 480))

            # Draw overlay text (just for debugging / visualization)
            with self.prediction_lock:
                display_text = self.latest_prediction_text

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Action: {display_text}"
            cv2.putText(resized, text, (50, resized.shape[0] - 60),
                        font, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

            # No saving to disk anymore
        except CvBridgeError as e:
            self.get_logger().error(f"Video callback CvBridge Error: {e}")

    def cleanup_and_exit(self):
        self.get_logger().info("Cleaning up (no video writer to release).")
        self.destroy_node()
        rclpy.shutdown()


    def prediction_callback(self, msg):
        if self.final_lowering_done:
            return

        now = time.time()
        if now - self.last_prediction_time < 1.0:
            return

        self.last_prediction_time = now
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            cropped = crop_image_cv2(cv_image, TOP_MARGIN, BOTTOM_MARGIN, LEFT_MARGIN, RIGHT_MARGIN)

            # Convert BGR -> RGB for Albumentations
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

            # Apply Albumentations
            augmented = augmentations(image=cropped_rgb)["image"]

            # Convert back to BGR for saving
            augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

            # Crop image before saving
            #cropped = crop_image_cv2(cv_image, TOP_MARGIN, BOTTOM_MARGIN, LEFT_MARGIN, RIGHT_MARGIN)

            temp_path = "/tmp/camera2_image.jpg"
            #cv2.imwrite(temp_path, augmented)
            cv2.imwrite(temp_path, cropped)

            # # Apply Sobel edge augmentation instead of Albumentations
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            gray_cropped = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
            sobel_img = sobel_edges(gray_cropped, ksize=3, alpha=1.5)


            # temp_path = "/tmp/camera1_image.jpg"
            # cv2.imwrite(temp_path, sobel_img)

            self.get_logger().info(f"Sending image for prediction  -  timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

            prediction = self.get_movement_prediction(temp_path)
            with self.prediction_lock:
                self.latest_prediction_text = prediction

            self.get_logger().info(f"Prediction: {prediction}")

            if prediction.strip() == "No Move, No Move, No Rotate":
                if hasattr(msg, "header") and msg.header.stamp == self.last_no_move_stamp:
                    self.get_logger().info("⏸Duplicate frame  -  not incrementing streak.")
                    return

                self.no_move_count += 1
                self.last_no_move_stamp = msg.header.stamp if hasattr(msg, "header") else None
                self.get_logger().info(f"'No Move' streak: {self.no_move_count}/3")

                if self.no_move_count >= 3:
                    self.get_logger().info("Triple 'No Move' confirmed  -  lowering Z.")
                    self.no_move_count = 0
                    self.last_no_move_stamp = None
                    self.lower_z_and_schedule_timer()
                else:
                    # Small jitter to refresh the frame
                    # pose = panda.get_pose()
                    # position = [pose[0, 3] , pose[1, 3] , 0.25]
                    # pose[0, 3], pose[1, 3], pose[2, 3] = position
                    # panda.move_to_pose(pose, speed_factor=0.05)
                    self.get_logger().info("Small move applied to ensure a new frame.")

                return

            else:
                # any non-no-move breaks the streak
                if self.no_move_count > 0:
                    self.get_logger().info("↩Prediction broke the streak  -  resetting.")
                self.no_move_count = 0
                self.last_no_move_stamp = None
                self.move_robot(prediction)


        except CvBridgeError as e:
            self.get_logger().error(f"Pred callback CvBridge Error: {e}")


    def get_movement_prediction(self, image_path):
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            try:
                response = requests.post(FLASK_SERVER_URL, files=files, timeout=10)
                if response.status_code == 200:
                    return response.json().get("movement", "Unknown")
                else:
                    self.get_logger().warning(f"Server error: {response.status_code} | {response.text}")
                    return "Unknown"
            except requests.exceptions.RequestException as e:
                self.get_logger().error(f"Request failed: {e}")
                return "Unknown"

    def move_robot(self, movement_prediction):
        pose = panda.get_pose()
        position = [pose[0, 3], pose[1, 3], 0.31]
        yaw = R.from_matrix(pose[0:3, 0:3]).as_euler('xyz', degrees=True)[2]
        current_euler = [initial_roll, initial_pitch, yaw]

        try:
            dx_label, dy_label, dtheta_label = [x.strip() for x in movement_prediction.split(",")]
        except ValueError:
            self.get_logger().warning(f"Invalid format: {movement_prediction}")
            return
        
        if dx_label == "Move Up":
            position[0] -= dx_step
        elif dx_label == "Move Down":
            position[0] += dx_step

        if dy_label == "Move Left":
            position[1] += dy_step
        elif dy_label == "Move Right":
            position[1] -= dy_step

        if dtheta_label == "Rotate Counterclockwise":
            current_euler[2] += dtheta_step
        elif dtheta_label == "Rotate Clockwise":
            current_euler[2] -= dtheta_step

        print('euler after: ', current_euler)
        print(' position: ', panda.get_position())

        rotation_matrix = euler_to_rotation_matrix(*current_euler)
        pose[0, 0:3] = rotation_matrix[0, 0:3]
        pose[1, 0:3] = rotation_matrix[1, 0:3]
        pose[2, 0:3] = rotation_matrix[2, 0:3]
        pose[0:3, 0:3] = rotation_matrix
        pose[0, 3], pose[1, 3], pose[2, 3] = position

        self.get_logger().info(f"Moving: {dx_label}, {dy_label}, {dtheta_label}")
        stiffness = 2*np.array([600, 600, 600, 600, 250, 150, 50])
        panda.move_to_pose(pose, speed_factor=speed_factor,stiffness=stiffness)
        time.sleep(0.5)

    def lower_z_and_schedule_timer(self):

        speed_factor = 0.03
        pose = panda.get_pose()
        pose[2, 3] -= dz_drop
        pose[1, 3] -= 0.002
        # pose[0, 3] += 0.003

        self.get_logger().info("Lowering Z now...")
        panda.move_to_pose(pose, speed_factor=speed_factor)

        gripper = libfranka.Gripper(hostname)
        gripper.move(0.06, 0.2)

        current_position = panda.get_position()
        new_position = current_position.copy()
        new_position[2] += 0.08
        panda.move_to_pose(new_position, panda.get_orientation(),  speed_factor=0.08)


        self.final_lowering_done = True
        self.final_timer = self.create_timer(2.0, self.on_final_timer)
        self.get_logger().info("⏲2s timer => shutting down after it fires.")

    def on_final_timer(self):
        self.get_logger().info("⏰ Final timer fired. Shutting down now.")
        self.destroy_timer(self.final_timer)
        self.cleanup_and_exit()

    def cleanup_and_exit(self):
        self.get_logger().info("Releasing video writer...")
        #self.video_writer.release()
        #elf.get_logger().info(f"Video saved at: {self.video_path}")
        self.destroy_node()
        rclpy.shutdown()

################################################################################
# Main                                        #
################################################################################

def main():
    rclpy.init(args=None)
    node = DualSubscriptionNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    node.get_logger().info("Starting multi-threaded session with action overlay on video...")
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
        node.cleanup_and_exit()

if __name__ == "__main__":
    main()
