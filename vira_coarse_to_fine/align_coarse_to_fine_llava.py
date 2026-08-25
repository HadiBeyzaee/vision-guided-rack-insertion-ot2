"""Insertion driven by a vision-language model, with adaptive step size.

The most developed version of the VLM control line. Two ROS 2 callbacks share
the camera topic: one records an annotated session video, the other crops the
frame once a second and queries two servers -

  URL_DIRECTION   -> which way to move ("Move Left", "Rotate CW", "No Move", ...)
  URL_COARSE_FINE -> whether each axis should take a coarse or fine step

so the arm approaches quickly and settles precisely. On convergence it switches
to a third endpoint for the final confirmation, then drops z to insert.

Requires: ROS 2 with both cameras publishing, the LLaVA servers reachable.
Run:      python3 align_coarse_to_fine_llava.py
Safety:   moves a real robot immediately on execution.
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

FLASK_SERVER_URL = f"http://{INFERENCE_HOST}:5091/predict"
FLASK_SERVER_URL2 = f"http://{INFERENCE_HOST}:5012/predict"
FLASK_SERVER_URL3 = f"http://{INFERENCE_HOST}:5000/predict"

# dx_step = 0.001
# dy_step = 0.001
# dtheta_step = 0.1
dz_drop = 0.075
speed_factor = 0.02

initial_roll = -180
initial_pitch = 0

def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix()


TOP_MARGIN = 440
BOTTOM_MARGIN = 80

LEFT_MARGIN = 380
RIGHT_MARGIN = 230

TOP_MARGIN2 = 450
BOTTOM_MARGIN2 = 140

LEFT_MARGIN2 = 460
RIGHT_MARGIN2 = 310

TOP_MARGIN3 = 450
BOTTOM_MARGIN3 = 140

LEFT_MARGIN3 = 460
RIGHT_MARGIN3 = 310

def crop_image_cv2(image, top=0, bottom=0, left=0, right=0):
    height, width = image.shape[:2]
    return image[top:height - bottom, left:width - right]

class DualSubscriptionNode(Node):
    def __init__(self):
        super().__init__('dual_sub_node')
        self.bridge = CvBridge()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = f"videos/camera1_session_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 30.0, (1280, 720))
        if not self.video_writer.isOpened():
            raise RuntimeError("Failed to open video writer (XVID / AVI).")

        self.get_logger().info(f"Video Writer opened: {self.video_path}")

        self.last_prediction_time = time.time()
        self.final_lowering_done = False
        self.final_timer = None

        self.latest_prediction_text = "Waiting for prediction..."
        self.prediction_lock = threading.Lock()

        self.video_cb_group = ReentrantCallbackGroup()
        self.pred_cb_group = MutuallyExclusiveCallbackGroup()

        self.video_sub = self.create_subscription(
            Image,
            '/camera1/camera1/color/image_raw',
            self.video_callback,
            10,
            callback_group=self.video_cb_group
        )

        self.pred_sub = self.create_subscription(
            Image,
            '/camera1/camera1/color/image_raw',
            self.prediction_callback,
            10,
            callback_group=self.pred_cb_group
        )

        self.get_logger().info("DualSubscriptionNode initialized (Multi-Thread).")

        self.dx_step = 0.007
        self.dy_step = 0.007
        self.dtheta_step = 1.0
        self.use_final_url = False  # Start with URL1


        self.skip_amount_check = False

    def video_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            resized = cv2.resize(cv_image, (1280, 720))

            with self.prediction_lock:
                display_text = self.latest_prediction_text

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 3
            text = f"Action: {display_text}"
            color = (0, 0, 255)
            bg_color = (255, 255, 255)

            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size
            x, y = 50, resized.shape[0] - 60

            cv2.rectangle(resized, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), bg_color, -1)

            cv2.putText(resized, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

            # Determine if step sizes are coarse or fine
            dx_mode = "coarse" if self.dx_step > 0.001 else "fine"
            dy_mode = "coarse" if self.dy_step > 0.001 else "fine"
            dtheta_mode = "coarse" if self.dtheta_step > 0.1 else "fine"

            step_text = f"dx: {dx_mode}, dy: {dy_mode}, dt: {dtheta_mode}"
            step_y = y + 50  # Below action text
            step_text_size, _ = cv2.getTextSize(step_text, font, font_scale * 0.9, thickness)
            step_width, step_height = step_text_size

            cv2.rectangle(resized, (x - 10, step_y - step_height - 10),
                        (x + step_width + 10, step_y + 10), bg_color, -1)
            cv2.putText(resized, step_text, (x, step_y), font, font_scale * 0.9, color, thickness, cv2.LINE_AA)

            self.video_writer.write(resized)
        except CvBridgeError as e:
            self.get_logger().error(f"Video callback CvBridge Error: {e}")

    def get_movement_prediction(self, image_path):
        url = FLASK_SERVER_URL3 if self.use_final_url else FLASK_SERVER_URL
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            try:
                response = requests.post(url, files=files, timeout=20)
                if response.status_code == 200:
                    return response.json().get("movement")
                else:
                    self.get_logger().warning(f"Server error from {url}: {response.status_code} | {response.text}")
                    return "Unknown"
            except requests.exceptions.RequestException as e:
                self.get_logger().error(f"Request to {url} failed: {e}")
                return "Unknown"


    def get_movement_amount_flags(self, image_path):
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            try:
                response = requests.post(FLASK_SERVER_URL2, files=files, timeout=20)
                if response.status_code == 200:
                    amount_string = response.json().get("movement_cf", "fine fine fine")
                    flags = amount_string.lower().strip().split()
                    print('**************',flags)
                    return flags[0].strip(',') == "coarse", flags[1].strip(',') == "coarse", flags[2].strip(',') == "coarse"

                else:
                    self.get_logger().warning(f"Amount server error: {response.status_code} | {response.text}")
                    return False, False, False
            except requests.exceptions.RequestException as e:
                self.get_logger().error(f"Amount request failed: {e}")
                return False, False, False

    def prediction_callback(self, msg):
        if self.final_lowering_done:
            return

        now = time.time()
        if now - self.last_prediction_time < 1.0:
            return

        self.last_prediction_time = now
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            #cropped = crop_image_cv2(cv_image, TOP_MARGIN, BOTTOM_MARGIN, LEFT_MARGIN, RIGHT_MARGIN)
            # if self.use_final_url:
            # cropped = crop_image_cv2(cv_image, TOP_MARGIN2, BOTTOM_MARGIN2, LEFT_MARGIN2, RIGHT_MARGIN2)
            # else:
            # cropped = crop_image_cv2(cv_image, TOP_MARGIN, BOTTOM_MARGIN, LEFT_MARGIN, RIGHT_MARGIN)
            if not self.skip_amount_check and not self.use_final_url:
                # First movement prediction uses crop1
                cropped = crop_image_cv2(cv_image, TOP_MARGIN, BOTTOM_MARGIN, LEFT_MARGIN, RIGHT_MARGIN)
            elif not self.use_final_url:
                # While checking coarse/fine flags, use crop2
                cropped = crop_image_cv2(cv_image, TOP_MARGIN2, BOTTOM_MARGIN2, LEFT_MARGIN2, RIGHT_MARGIN2)
            else:
                # Final prediction when all are fine, use crop3
                cropped = crop_image_cv2(cv_image, TOP_MARGIN3, BOTTOM_MARGIN3, LEFT_MARGIN3, RIGHT_MARGIN3)

            temp_path = "/tmp/camera1_image.jpg"
            cv2.imwrite(temp_path, cropped)

            prediction = self.get_movement_prediction(temp_path)

            if not self.skip_amount_check:
                dx_big, dy_big, dtheta_big = self.get_movement_amount_flags(temp_path)
                if not dx_big and not dy_big and not dtheta_big:
                    self.skip_amount_check = True
                    self.use_final_url = True  # Switch to URL3 for final fine-tuning
                    self.get_logger().info("All amounts are 'fine'. Switching to FLASK_SERVER_URL3 for final prediction.")

                self.dx_step = 0.007 if dx_big else 0.0009
                self.dy_step = 0.007 if dy_big else 0.0009
                self.dtheta_step = 0.8 if dtheta_big else 0.1
                self.get_logger().info(f"Step sizes set -> dx: {self.dx_step}, dy: {self.dy_step}, dθ: {self.dtheta_step}")

            with self.prediction_lock:
                self.latest_prediction_text = prediction

            self.get_logger().info(f"Prediction: {prediction}")

            if prediction.strip() == "No Move, No Move, No Rotate":
                self.get_logger().info("Stop condition met. Lowering Z + final shutdown soon...")
                self.lower_z_and_schedule_timer()
            else:
                self.move_robot(prediction)

        except CvBridgeError as e:
            self.get_logger().error(f"Pred callback CvBridge Error: {e}")

    def move_robot(self, movement_prediction):
        pose = panda.get_pose()
        position = [pose[0, 3], pose[1, 3], 0.253]
        yaw = R.from_matrix(pose[0:3, 0:3]).as_euler('xyz', degrees=True)[2]
        current_euler = [initial_roll, initial_pitch, yaw]
        print('current_euler before: ', current_euler)

        try:
            dx_label, dy_label, dtheta_label = [x.strip() for x in movement_prediction.split(",")]
        except ValueError:
            self.get_logger().warning(f"Invalid format: {movement_prediction}")
            return

        if dx_label == "Move Up":
            position[0] += self.dx_step
        elif dx_label == "Move Down":
            position[0] -= self.dx_step

        if dy_label == "Move Left":
            position[1] += self.dy_step
        elif dy_label == "Move Right":
            position[1] -= self.dy_step

        if dtheta_label == "Rotate Counterclockwise":
            current_euler[2] += self.dtheta_step
        elif dtheta_label == "Rotate Clockwise":
            current_euler[2] -= self.dtheta_step

        print('current_euler after: ', current_euler)
        print(' position: ', panda.get_position())

        rotation_matrix = euler_to_rotation_matrix(*current_euler)
        pose[0:3, 0:3] = rotation_matrix
        pose[0, 3], pose[1, 3], pose[2, 3] = position

        self.get_logger().info(f"Moving: {dx_label}, {dy_label}, {dtheta_label}")
        stiffness = 2 * np.array([600, 600, 600, 600, 250, 150, 50])
        panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)

    def lower_z_and_schedule_timer(self):
        pose = panda.get_pose()
        pose[2, 3] -= dz_drop
        self.get_logger().info("Lowering Z now...")
        panda.move_to_pose(pose, speed_factor=speed_factor)

        self.final_lowering_done = True
        self.final_timer = self.create_timer(2.0, self.on_final_timer)
        self.get_logger().info("2s timer => shutting down after it fires.")

    def on_final_timer(self):
        self.get_logger().info("Final timer fired. Shutting down now.")
        self.destroy_timer(self.final_timer)
        self.cleanup_and_exit()

    def cleanup_and_exit(self):
        self.get_logger().info("Releasing video writer...")
        self.video_writer.release()
        self.get_logger().info(f"Video saved at: {self.video_path}")
        self.destroy_node()
        rclpy.shutdown()

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
