"""Forward frames from a ROS 2 camera topic to the inference server.

A ROS node that POSTs each frame to the prediction endpoint and logs the reply.
Useful for testing a newly deployed model against the live camera without
running a control loop, so nothing moves while you check the predictions.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import requests  # For sending images to Flask server
# --- Connection settings -----------------------------------------------
# Set these in your shell or a .env file; see .env.example at the repo root.
import os as _os
INFERENCE_HOST = _os.environ.get("INFERENCE_HOST", "127.0.0.1")
# -----------------------------------------------------------------------

# Flask Server Address
FLASK_SERVER_URL = f"http://{INFERENCE_HOST}:5000/predict"

class Camera1ImageSender(Node):
    def __init__(self):
        super.__init__('camera1_image_sender')
        self.bridge = CvBridge
        self.latest_image = None

        # Subscribe to Camera 1 Topic
        self.subscription = self.create_subscription(
            Image,
            '/camera1/camera1/color/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        """ Capture one image from the camera and send it to Flask, then exit """
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.get_logger.info("Received image from Camera 1")

            # Save Image Temporarily
            temp_image_path = "/tmp/camera1_image.jpg"
            cv2.imwrite(temp_image_path, self.latest_image)

            # Send image using requests
            with open(temp_image_path, "rb") as image_file:
                files = {"image": image_file}
                try:
                    response = requests.post(FLASK_SERVER_URL, files=files, timeout=10)
                    if response.status_code == 200:
                        print(f"Sent Image | Response: {response.json}")
                    else:
                        print(f"Server Error: {response.status_code} | {response.text}")
                except requests.exceptions.RequestException as e:
                    print(f"Failed to send image: {e}")

            # Stop node after sending the image
            self.get_logger.info("Image sent successfully. Exiting...")
            self.destroy_node  # Destroy the ROS node
            rclpy.shutdown  # Ensure ROS2 is completely shut down

        except CvBridgeError as e:
            self.get_logger.error(f'CvBridge Error: {e}')
            self.destroy_node
            rclpy.shutdown

# Initialize ROS2
rclpy.init(args=None)
image_sender = Camera1ImageSender

try:
    print("Waiting for a single image from Camera 1...")
    rclpy.spin(image_sender)  # Runs until the node is destroyed
except KeyboardInterrupt:
    print("Stopping...")
    image_sender.destroy_node
    rclpy.shutdown


#curl -X POST http://<INFERENCE_HOST>:5000/predict -F "image=@test_lower5/color_images/camera1/1.png"
