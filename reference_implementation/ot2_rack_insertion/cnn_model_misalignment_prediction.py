import os
import requests
import numpy as np
import logging
import cv2
import time
import threading
import pyrealsense2 as rs
import panda_py
from panda_py import libfranka
from scipy.spatial.transform import Rotation as R
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# Robot setup
hostname = PANDA_HOSTNAME
panda = panda_py.Panda(hostname)

# Server URL
FLASK_SERVER_URL = f"http://{INFERENCE_HOST}:4001/predict"

# Movement parameters
dx_step = 0.001
dy_step = 0.001
dtheta_step = 0.1
dz_drop = 0.08
speed_factor = 0.05

# Orientation constants
initial_roll = -180
initial_pitch = 0

# Cropping margins
TOP_MARGIN = 240
BOTTOM_MARGIN = 320
LEFT_MARGIN = 500
RIGHT_MARGIN = 200

# Initial pose adjustment
pose = panda.get_pose()
pose[2, 3] = 0.30
panda.move_to_pose(pose, speed_factor=speed_factor)


def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix()


def crop_image_cv2(image, top=0, bottom=0, left=0, right=0):
    height, width = image.shape[:2]
    return image[top:height - bottom, left:width - right]


class RealSenseNode:
    def __init__(self):
        # State
        self.last_prediction_time = time.time()
        self.final_lowering_done = False
        self.no_move_count = 0
        self.latest_prediction_text = "Waiting for prediction..."
        self.prediction_lock = threading.Lock()

        # RealSense setup
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)

        for _ in range(30):
            self.pipeline.wait_for_frames()

        logging.info("RealSense initialized at 1280x720")

        # Video recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = "path/to/videos"
        os.makedirs(video_dir, exist_ok=True)
        self.video_path = os.path.join(video_dir, f"prediction_{timestamp}.avi")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_out = cv2.VideoWriter(self.video_path, fourcc, 30.0, (1280, 720))
        if not self.video_out.isOpened():
            raise RuntimeError("Failed to open video writer")

        self.stop_event = threading.Event()
        self.rec_thread = threading.Thread(target=self.record_loop, daemon=True)
        self.rec_thread.start()
        logging.info(f"Recording started -> {self.video_path}")

    def record_loop(self):
        align = rs.align(rs.stream.color)
        while not self.stop_event.is_set():
            frames = self.pipeline.wait_for_frames()
            aligned = align.process(frames)
            cf = aligned.get_color_frame()
            if cf:
                frame = np.asanyarray(cf.get_data())
                self.video_out.write(frame)

    def prediction_callback(self, cv_image):
        if self.final_lowering_done:
            return

        now = time.time()
        if now - self.last_prediction_time < 1.0:
            return

        self.last_prediction_time = now

        cropped = crop_image_cv2(cv_image, TOP_MARGIN, BOTTOM_MARGIN, LEFT_MARGIN, RIGHT_MARGIN)

        temp_path = "/tmp/camera_image.jpg"
        cv2.imwrite(temp_path, cropped)

        prediction = self.get_movement_prediction(temp_path)

        with self.prediction_lock:
            self.latest_prediction_text = prediction

        logging.info(f"Prediction: {prediction}")

        if prediction.strip() == "No Move, No Move, No Rotate":
            self.no_move_count += 1
            logging.info(f"'No Move' count: {self.no_move_count}")
            if self.no_move_count >= 3:
                logging.info("Final lowering triggered")
                self.no_move_count = 0
                self.lower_z_and_schedule_timer()
        else:
            self.no_move_count = 0
            self.move_robot(prediction)

    def get_movement_prediction(self, image_path):
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            try:
                response = requests.post(FLASK_SERVER_URL, files=files, timeout=10)
                if response.status_code == 200:
                    return response.json().get("movement", "Unknown")
                else:
                    logging.warning(f"Server error: {response.status_code}")
                    return "Unknown"
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                return "Unknown"

    def move_robot(self, movement_prediction):
        panda = panda_py.Panda(hostname)
        pose = panda.get_pose()
        position = [pose[0, 3], pose[1, 3], 0.30]
        yaw = R.from_matrix(pose[0:3, 0:3]).as_euler('xyz', degrees=True)[2]
        current_euler = [initial_roll, initial_pitch, yaw]

        try:
            dx_label, dy_label, dtheta_label = [x.strip() for x in movement_prediction.split(",")]
        except ValueError:
            logging.warning("Invalid prediction format")
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

        rotation_matrix = euler_to_rotation_matrix(*current_euler)
        pose[0:3, 0:3] = rotation_matrix
        pose[0, 3], pose[1, 3], pose[2, 3] = position

        stiffness = 2 * np.array([600, 600, 600, 600, 250, 150, 50])
        panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)
        time.sleep(0.5)

    def lower_z_and_schedule_timer(self):
        panda = panda_py.Panda(hostname)
        speed_factor_local = 0.03
        pose = panda.get_pose()
        pose[2, 3] -= dz_drop
        panda.move_to_pose(pose, speed_factor=speed_factor_local)

        gripper = libfranka.Gripper(hostname)
        gripper.move(0.06, 0.2)

        current_position = panda.get_position()
        new_position = current_position.copy()
        new_position[2] += 0.08
        panda.move_to_pose(new_position, panda.get_orientation(), speed_factor=0.08)

        self.final_lowering_done = True
        logging.info("Final lowering complete")



def start_robot_prediction():
    """Call this function to run the full robot control + camera loop."""
    node = RealSenseNode()
    logging.info("Starting RealSense loop...")

    try:
        while not node.final_lowering_done:
            frames = node.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                cv_image = np.asanyarray(color_frame.get_data())
                node.prediction_callback(cv_image)

    except KeyboardInterrupt:
        logging.info("Stopped by user")

    finally:
        node.stop_event.set()
        node.rec_thread.join(timeout=2)
        node.video_out.release()
        node.pipeline.stop()
        logging.info(f"Video saved -> {node.video_path}")
        logging.info("RealSense pipeline stopped")



if __name__ == "__main__":
    start_robot_prediction()
