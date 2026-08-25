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

FLASK_SERVER_URL = f"http://{INFERENCE_HOST}:5002/predict"

dx_step = 0.0007
dy_step = 0.0007
dtheta_step = 0.1
dz_drop = 0.075
speed_factor = 0.05

initial_roll = -180
initial_pitch = 0

def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix()

TOP_MARGIN = 170
BOTTOM_MARGIN = 410

LEFT_MARGIN = 460
RIGHT_MARGIN = 310

def crop_image_cv2(image, top=0, bottom=0, left=0, right=0):
    height, width = image.shape[:2]
    return image[top:height - bottom, left:width - right]


class DualSubscriptionNode(Node):
    def __init__(self):
        super().__init__('dual_sub_node')
        self.bridge = CvBridge()

        # --- Video Writer Setup (.avi + XVID) ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = fos.path.join(BASE_DIR, "run_output")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 30.0, (1280, 720))
        if not self.video_writer.isOpened():
            raise RuntimeError("Failed to open video writer (XVID / AVI).")

        self.get_logger().info(f"Video Writer opened: {self.video_path}")

        # --- State ---
        self.last_prediction_time = time.time()
        self.final_lowering_done = False
        self.final_timer = None

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

        self.get_logger().info("DualSubscriptionNode initialized (Multi-Thread).")

    def video_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            resized = cv2.resize(cv_image, (1280, 720))

            # Draw overlay text from latest prediction
            with self.prediction_lock:
                display_text = self.latest_prediction_text

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 3
            text = f"Action: {display_text}"
            color = (0, 0, 255)  # Red text
            bg_color = (255, 255, 255)  # White background

            # Position text in the bottom-left corner
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size
            x, y = 50, resized.shape[0] - 60  # bottom-left corner

            # Draw background rectangle
            cv2.rectangle(
                resized,
                (x - 10, y - text_height - 10),
                (x + text_width + 10, y + 10),
                bg_color,
                thickness=-1  # filled rectangle
            )

            # Draw text over it
            cv2.putText(
                resized,
                text,
                org=(x, y),
                fontFace=font,
                fontScale=font_scale,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA
            )


            self.video_writer.write(resized)
        except CvBridgeError as e:
            self.get_logger().error(f"Video callback CvBridge Error: {e}")

    def prediction_callback(self, msg):
        if self.final_lowering_done:
            return

        now = time.time()
        if now - self.last_prediction_time < 1.0:
            return

        self.last_prediction_time = now
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Crop image before saving
            cropped = crop_image_cv2(cv_image, TOP_MARGIN, BOTTOM_MARGIN, LEFT_MARGIN, RIGHT_MARGIN)

            temp_path = "/tmp/camera2_image.jpg"
            cv2.imwrite(temp_path, cropped)

            prediction = self.get_movement_prediction(temp_path)
            with self.prediction_lock:
                self.latest_prediction_text = prediction

            self.get_logger().info(f"Prediction: {prediction}")

            if prediction.strip() in [
                "No Move, No Move, No Rotate",
            ]:
                self.get_logger().info("Stop condition met. Lowering Z + final shutdown soon...")
                self.lower_z_and_schedule_timer()
            else:
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
            position[0] += dx_step
        elif dx_label == "Move Down":
            position[0] -= dx_step

        if dy_label == "Move Left":
            position[1] += dy_step
        elif dy_label == "Move Right":
            position[1] -= dy_step

        if dtheta_label == "Rotate Counterclockwise":
            current_euler[2] += dtheta_step
        elif dtheta_label == "Rotate Clockwise":
            current_euler[2] -= dtheta_step
        
        print('current_euler after: ', current_euler)
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

    def lower_z_and_schedule_timer(self):

        speed_factor = 0.01
        pose = panda.get_pose()
        pose[2, 3] -= dz_drop
        self.get_logger().info("Lowering Z now...")
        panda.move_to_pose(pose, speed_factor=speed_factor)

        self.final_lowering_done = True
        self.final_timer = self.create_timer(2.0, self.on_final_timer)
        self.get_logger().info("⏲2s timer => shutting down after it fires.")

    def on_final_timer(self):
        self.get_logger().info("⏰ Final timer fired. Shutting down now.")
        self.destroy_timer(self.final_timer)
        self.cleanup_and_exit()

    def cleanup_and_exit(self):
        self.get_logger().info("Releasing video writer...")
        self.video_writer.release()
        self.get_logger().info(f"Video saved at: {self.video_path}")
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
