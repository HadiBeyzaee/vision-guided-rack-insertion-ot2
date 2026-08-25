import pyrealsense2 as rs
import json

# Start pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Get intrinsics from color stream
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()

# Build cam_K (3x3 matrix flattened row-major)
cam_K = [
    intrinsics.fx, 0.0, intrinsics.ppx,
    0.0, intrinsics.fy, intrinsics.ppy,
    0.0, 0.0, 1.0
]

# Get depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Print
params = {
    "cam_K": cam_K,
    "depth_scale": depth_scale
}
print(json.dumps(params, indent=2))

# Cleanup
pipeline.stop()
