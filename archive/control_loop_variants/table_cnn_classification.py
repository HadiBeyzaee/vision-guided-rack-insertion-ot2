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

hostname = PANDA_HOSTNAME
panda = panda_py.Panda(hostname)

FLASK_SERVER_URL = f"http://{INFERENCE_HOST}:4000/predict"

dx_step = 0.0015
dy_step = 0.0015
dtheta_step = 0.1
dz_drop = 0.081
speed_factor = 0.05

initial_roll = -179
initial_pitch = 0

TOP_MARGIN = 340
BOTTOM_MARGIN = 250
LEFT_MARGIN = 460
RIGHT_MARGIN = 170

def euler_to_rotation_matrix(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_matrix()

def crop_image_cv2(image, top=0, bottom=0, left=0, right=0):
    height, width = image.shape[:2]
    return image[top:height - bottom, left:width - right]


class RealSenseNode:
    def __init__(self):
        # --- State ---
        self.last_prediction_time = time.time()
        self.final_lowering_done = False
        self.no_move_count = 0
        self.last_no_move_stamp = None

        # For overlay text
        self.latest_prediction_text = "Waiting for prediction..."
        self.prediction_lock = threading.Lock()

        # RealSense setup
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # Warm-up
        for _ in range(30):
            self.pipeline.wait_for_frames()

        logging.info("RealSenseNode initialized with streaming at 1280x720.")

    def prediction_callback(self, cv_image):
        if self.final_lowering_done:
            return

        now = time.time()
        if now - self.last_prediction_time < 1.0:
            return

        self.last_prediction_time = now

        cropped = crop_image_cv2(cv_image, TOP_MARGIN, BOTTOM_MARGIN, LEFT_MARGIN, RIGHT_MARGIN)

        temp_path = "/tmp/camera2_image.jpg"
        cv2.imwrite(temp_path, cropped)

        logging.info("Sending image for prediction…")

        prediction = self.get_movement_prediction(temp_path)
        with self.prediction_lock:
            self.latest_prediction_text = prediction

        logging.info(f"Prediction: {prediction}")

        if prediction.strip() == "No Move, No Move, No Rotate":
            self.no_move_count += 1
            logging.info(f"'No Move' streak: {self.no_move_count}/2")

            if self.no_move_count >= 2:
                logging.info("Triple 'No Move' confirmed  -  lowering Z.")
                self.no_move_count = 0
                self.lower_z_and_schedule_timer()
            return
        else:
            if self.no_move_count > 0:
                logging.info("↩Prediction broke the streak  -  resetting.")
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
                    logging.warning(f"Server error: {response.status_code} | {response.text}")
                    return "Unknown"
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                return "Unknown"

    def move_robot(self, movement_prediction):
        pose = panda.get_pose()
        position = [pose[0, 3], pose[1, 3], 0.22]
        yaw = R.from_matrix(pose[0:3, 0:3]).as_euler('xyz', degrees=True)[2]
        current_euler = [initial_roll, initial_pitch, yaw]

        try:
            dx_label, dy_label, dtheta_label = [x.strip() for x in movement_prediction.split(",")]
        except ValueError:
            logging.warning(f"Invalid format: {movement_prediction}")
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

        rotation_matrix = euler_to_rotation_matrix(*current_euler)
        pose[0:3, 0:3] = rotation_matrix
        pose[0, 3], pose[1, 3], pose[2, 3] = position

        logging.info(f"Moving: {dx_label}, {dy_label}, {dtheta_label}")
        stiffness = 2*np.array([600, 600, 600, 600, 250, 150, 50])
        panda.move_to_pose(pose, speed_factor=speed_factor, stiffness=stiffness)
        time.sleep(0.5)

    def lower_z_and_schedule_timer(self):
        speed_factor_local = 0.03
        pose = panda.get_pose()
        pose[2, 3] -= dz_drop

        logging.info("Lowering Z now…")
        panda.move_to_pose(pose, speed_factor=speed_factor_local)

        gripper = libfranka.Gripper(hostname)
        gripper.move(0.08, 0.2)

        current_position = panda.get_position()
        new_position = current_position.copy()
        new_position[2] += 0.08
        panda.move_to_pose(new_position, panda.get_orientation(), speed_factor=0.08)

        self.final_lowering_done = True
        logging.info("⏲Final lowering done  -  shutting down soon.")


def main():
    node = RealSenseNode()
    logging.info("Starting RealSense streaming loop…")
    try:
        while True:
            if node.final_lowering_done:
                break  # Exit loop once lowering is complete

            frames = node.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            cv_image = np.asanyarray(color_frame.get_data())
            node.prediction_callback(cv_image)

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        node.pipeline.stop()
        logging.info("RealSense pipeline stopped.")


if __name__ == "__main__":
    main()