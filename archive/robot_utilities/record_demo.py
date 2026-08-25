import os

# --- Archived variant. Connection settings and paths parameterised. ----
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
SLACK_BOT_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "")
# -----------------------------------------------------------------------
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt

# Specific RealSense camera serial
CAMERA_SERIAL = "148522074814"

WIDTH = 640
HEIGHT = 480
FPS = 30

pipeline = rs.pipeline
config = rs.config

# Select specific RealSense camera
config.enable_device(CAMERA_SERIAL)

# Use RGB directly because matplotlib expects RGB, not BGR
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)

running = {"value": True}


def on_key(event):
    if event.key == "q":
        running["value"] = False


try:
    pipeline.start(config)
    print(f"Started RealSense camera with serial: {CAMERA_SERIAL}")
    print("Press q inside the matplotlib window to quit.")

    plt.ion
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initial blank image
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image_display = ax.imshow(blank)

    ax.set_title(f"RealSense RGB Preview - Serial: {CAMERA_SERIAL}\nPress q to quit")
    ax.axis("off")

    while running["value"]:
        frames = pipeline.wait_for_frames
        color_frame = frames.get_color_frame

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data)

        image_display.set_data(frame)
        fig.canvas.draw_idle
        plt.pause(0.001)

except KeyboardInterrupt:
    print("Stopped by Ctrl+C.")

finally:
    pipeline.stop
    plt.close("all")
    print("Camera stopped.")
