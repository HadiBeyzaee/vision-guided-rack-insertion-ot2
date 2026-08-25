"""Insertion driven by a VLM queried once per axis.

Instead of one label describing the whole correction, this version asks the
server separately for dx, dy and dtheta and applies translation and rotation as
separate moves. Slower per iteration but easier to debug when one axis is
misbehaving, because you can see which query is wrong.
"""

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
# --- Connection settings -----------------------------------------------
# Set these in your shell or a .env file; see .env.example at the repo root.
import os as _os
PANDA_HOSTNAME = _os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST = _os.environ.get("INFERENCE_HOST", "127.0.0.1")
# -----------------------------------------------------------------------

################################################################################
# Global Config                                  #
################################################################################

logging.basicConfig(level=logging.INFO)

hostname = PANDA_HOSTNAME
panda = panda_py.Panda(hostname)

TRANSLATION_SERVER = f"http://{INFERENCE_HOST}:5001/predict_translation"
ROTATION_SERVER = f"http://{INFERENCE_HOST}:5002/predict_rotation"

# Movement step sizes
dx_step = 0.0015
dy_step = 0.0015
dtheta_step = 0.2
dz_drop = 0.02
speed_factor = 0.05

initial_roll = -180
initial_pitch = 0

################################################################################
# Utilities                                      #
################################################################################

def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix()

# Cropping margins
TOP_MARGIN = 84
BOTTOM_MARGIN = 300
LEFT_MARGIN = 242
RIGHT_MARGIN = 30

def crop_image_cv2(image, top=0, bottom=0, left=0, right=0):
    height, width = image.shape[:2]
    return image[top:height - bottom, left:width - right]

class DualPredictionNode(Node):
    def __init__(self):
        super().__init__('dual_prediction_node')
        self.bridge = CvBridge()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = f"videos/session_{timestamp}.avi"
        self.video_writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (1280, 720))

        self.latest_text = "Waiting..."
        self.prediction_lock = threading.Lock()
        self.final_lowering_done = False

        self.rotation_satisfied = False
        self.translation_satisfied = False

        self.video_cb_group = ReentrantCallbackGroup()
        self.pred_cb_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            Image,
            '/camera2/camera2/color/image_raw',
            self.video_callback,
            10,
            callback_group=self.video_cb_group
        )
        self.create_subscription(
            Image,
            '/camera2/camera2/color/image_raw',
            self.prediction_callback,
            10,
            callback_group=self.pred_cb_group
        )

    def video_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            resized = cv2.resize(img, (1280, 720))

            with self.prediction_lock:
                text = f"Action: {self.latest_text}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(resized, text, (50, 650), font, 1.0, (0, 0, 255), 3)
            self.video_writer.write(resized)
        except CvBridgeError as e:
            self.get_logger().error(f"Video Error: {e}")

    def prediction_callback(self, msg):
        if self.final_lowering_done:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cropped = crop_image_cv2(img, TOP_MARGIN, BOTTOM_MARGIN, LEFT_MARGIN, RIGHT_MARGIN)
            temp_path = "/tmp/cam1.jpg"
            cv2.imwrite(temp_path, cropped)

            if not self.rotation_satisfied:
                dtheta_label = self.query_server(ROTATION_SERVER, temp_path, label_key="rotation")
                with self.prediction_lock:
                    self.latest_text = f"rotation: {dtheta_label}"

                if dtheta_label == "No Rotate":
                    self.rotation_satisfied = True
                    pose = panda.get_pose()
                    self.last_yaw_after_rotation = R.from_matrix(pose[0:3, 0:3]).as_euler('xyz', degrees=True)[2]
                else:
                    self.move_rotation(dtheta_label)
                    return

            if not self.translation_satisfied:
                dx_dy = self.query_server(TRANSLATION_SERVER, temp_path, label_key="movement")
                dx_label, dy_label = [x.strip() for x in dx_dy.split(",")]

                with self.prediction_lock:
                    self.latest_text = f"Translation: {dx_dy}"

                if dx_label == "No Move" and dy_label == "No Move":
                    self.translation_satisfied = True
                else:
                    self.move_translation(dx_label, dy_label)
                    return

            if self.rotation_satisfied and self.translation_satisfied:
                self.get_logger().info("All movements are 'No Move'. Lowering Z.")
                self.lower_z_and_exit()

        except CvBridgeError as e:
            self.get_logger().error(f"Prediction Callback Error: {e}")

    def query_server(self, url, image_path, label_key="movement"):
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            try:
                response = requests.post(url, files=files, timeout=10)
                if response.status_code == 200:
                    return response.json().get(label_key, "Unknown")
                else:
                    return "Unknown"
            except requests.RequestException:
                return "Unknown"

    def move_translation(self, dx_label, dy_label):
        pose = panda.get_pose()
        x, y = pose[0, 3], pose[1, 3]

        if dx_label == "Move Left":
            x -= dx_step
        elif dx_label == "Move Right":
            x += dx_step

        if dy_label == "Move Down":
            y -= dy_step
        elif dy_label == "Move Up":
            y += dy_step

        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = 0.185

        # apply last yaw angle from rotation phase
        rotation_matrix = euler_to_rotation_matrix(initial_roll, initial_pitch, self.last_yaw_after_rotation)
        pose[0:3, 0:3] = rotation_matrix

        self.get_logger().info(f"Translation: {dx_label}, {dy_label}")
        panda.move_to_pose(pose, speed_factor=speed_factor)

    def move_rotation(self, dtheta_label):
        pose = panda.get_pose()
        yaw = R.from_matrix(pose[0:3, 0:3]).as_euler('xyz', degrees=True)[2]
        euler = [initial_roll, initial_pitch, yaw]

        if dtheta_label == "Rotate Clockwise":
            euler[2] -= dtheta_step
        elif dtheta_label == "Rotate Counterclockwise":
            euler[2] += dtheta_step

        self.last_yaw_after_rotation = euler[2]

        rotation_matrix = euler_to_rotation_matrix(*euler)
        pose[0:3, 0:3] = rotation_matrix
        pose[2, 3] = 0.185

        self.get_logger().info(f"rotation: {dtheta_label}")
        panda.move_to_pose(pose, speed_factor=speed_factor)

    def lower_z_and_exit(self):
        pose = panda.get_pose()
        pose[2, 3] -= dz_drop
        self.get_logger().info("Lowering Z now...")
        panda.move_to_pose(pose, speed_factor=speed_factor)

        self.final_lowering_done = True
        self.get_logger().info(f"Session complete. Video saved at {self.video_path}")
        self.destroy_node()
        rclpy.shutdown()

def main():
    rclpy.init()
    node = DualPredictionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
