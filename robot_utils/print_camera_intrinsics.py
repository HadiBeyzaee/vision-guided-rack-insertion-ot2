"""Print the RealSense colour stream intrinsics.

Reports width, height, fx, fy, cx, cy, the distortion model and its
coefficients. These are the numbers that go into config/camera1_cam.json for
the SAM-6D pose server, and into any pixel-to-metre conversion.

Re-run after changing resolution - the intrinsics differ per stream profile.
"""

import pyrealsense2 as rs

pipeline = rs.pipeline
config = rs.config
pipeline.start(config)
profile = pipeline.get_active_profile
intr = profile.get_stream(rs.stream.color).as_video_stream_profile.get_intrinsics

print("Width:", intr.width)
print("Height:", intr.height)
print("fx:", intr.fx)
print("fy:", intr.fy)
print("ppx (cx):", intr.ppx)
print("ppy (cy):", intr.ppy)
print("distortion model:", intr.model)
print("coeffs:", intr.coeffs)
