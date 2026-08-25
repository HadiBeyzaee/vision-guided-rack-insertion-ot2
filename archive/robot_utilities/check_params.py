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

pipeline = rs.pipeline
config = rs.config
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

profile = pipeline.start(config)

device = profile.get_device
color_sensor = device.first_color_sensor

print("\n===== CAMERA SENSOR OPTIONS =====")
for opt in color_sensor.get_supported_options:
    try:
        value = color_sensor.get_option(opt)
        print(f"{opt.name}: {value}")
    except:
        pass

print("\n===== STREAM INTRINSICS =====")
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile
intr = color_stream.get_intrinsics

print(f"Width: {intr.width}")
print(f"Height: {intr.height}")
print(f"FX: {intr.fx}")
print(f"FY: {intr.fy}")
print(f"PPX: {intr.ppx}")
print(f"PPY: {intr.ppy}")
print(f"Distortion Model: {intr.model}")
print(f"Distortion Coeffs: {intr.coeffs}")

print("\n===== CAMERA INFO =====")
print("Name:", device.get_info(rs.camera_info.name))
print("Serial:", device.get_info(rs.camera_info.serial_number))
print("Firmware:", device.get_info(rs.camera_info.firmware_version))

pipeline.stop
