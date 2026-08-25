import os
import cv2
import threading
from datetime import datetime
import pyrealsense2 as rs
import numpy as np


class RealSenseRecorder:
    """Handles RealSense color video recording to a file."""

    def __init__(self, serial_number="147322070835", save_dir="path/to/videos"):
        self.serial_number = serial_number
        self.save_dir = save_dir

        self.pipeline = None
        self.video_out = None
        self.rec_thread = None
        self.stop_event = None
        self.video_path = None

    def start(self, prefix="session"):
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        ctx = rs.context()
        config.enable_device(self.serial_number)
        self.pipeline.start(config)

        # Create output directory and file path
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(self.save_dir, f"{prefix}_{timestamp}.avi")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_out = cv2.VideoWriter(self.video_path, fourcc, 30.0, (1280, 720))

        if not self.video_out.isOpened():
            raise RuntimeError("Could not open video writer")

        print(f"Recording started → {self.video_path}")

        # Start recording thread
        self.stop_event = threading.Event()
        align = rs.align(rs.stream.color)

        def record_loop():
            while not self.stop_event.is_set():
                frames = self.pipeline.wait_for_frames()
                aligned = align.process(frames)
                cf = aligned.get_color_frame()
                if cf:
                    frame = np.asanyarray(cf.get_data())
                    self.video_out.write(frame)

        self.rec_thread = threading.Thread(target=record_loop, daemon=True)
        self.rec_thread.start()

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
        if self.rec_thread:
            self.rec_thread.join(timeout=2)

        if self.video_out:
            self.video_out.release()
            print(f"Video saved → {self.video_path}")

        if self.pipeline:
            self.pipeline.stop()

        print("RealSense pipeline stopped")
