import os

# --- Connection settings (override in your shell or a .env file) -------
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
# -----------------------------------------------------------------------
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from datetime import datetime
import argparse
class VideoRecorder(Node):
    def __init__(self, prefix="session"):
        super().__init__('video_recorder')
        self.bridge = CvBridge()
        self.writer = None   # initialize later

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = fos.path.join(BASE_DIR, "run_output")

        # Subscribe to camera
        self.sub = self.create_subscription(
            Image,
            '/camera2/camera2/color/image_raw',
            self.callback,
            10
        )
        self.get_logger().info(f"Will record to {self.video_path}")

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Initialize VideoWriter once we know frame size
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(self.video_path, fourcc, 30.0, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError("Could not open video writer")
            self.get_logger().info(f"Frame size = {w}x{h}, writer initialized")

        self.writer.write(frame)

    def cleanup(self):
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info(f"Saved video -> {self.video_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="session")
    args = parser.parse_args()

    rclpy.init()
    node = VideoRecorder(prefix=args.prefix)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
