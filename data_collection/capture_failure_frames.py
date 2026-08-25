"""Save camera-2 frames on demand into the failure folder.

The quick manual counterpart to collect_grasp_and_place_offsets.py: run it beside
a live experiment and capture the moments something actually goes wrong,
rather than staging the failure.
"""

import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# ---------------------- Image Subscriber ----------------------
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.image = None
        self.subscription = self.create_subscription(
            Image,
            '/camera2/camera2/color/image_raw',
            self.image_callback,
            10)
        self.get_logger().info("Subscribed to /camera2/camera2/color/image_raw")

    def image_callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")


# ---------------------- Main ----------------------
def main():
    rclpy.init()
    node = ImageSubscriber()

    save_dir = "data/fail_image"
    os.makedirs(save_dir, exist_ok=True)

    print("[INFO] Press ENTER to save the current image, or 'q' + ENTER to quit.")

    img_counter = len([f for f in os.listdir(save_dir) if f.endswith(".png")]) + 1

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.image is not None:
                cv2.imshow("Live RGB Feed", node.image)

            key = cv2.waitKey(100)  # check for key input
            if key == 13:  # Enter key
                if node.image is not None:
                    img_path = os.path.join(save_dir, f"{img_counter}.png")
                    cv2.imwrite(img_path, node.image)
                    print(f"[INFO] Saved image -> {img_path}")
                    img_counter += 1
                else:
                    print("[WARN] No image received yet.")
            elif key == ord('q') or key == 27:  # 'q' or Esc to quit
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
