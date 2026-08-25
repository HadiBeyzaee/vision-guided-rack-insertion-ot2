import os

# --- Robot-PC settings (override in your shell or a .env file) ---------
# These scripts ran on the robot PC, where the account was "panda", while
# perception ran on a separate GPU box. The paths below were absolute on
# that machine; they are variables here so the split works anywhere.
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")

# Interpreter with the ROS 2 / robot stack installed
ROS_PYTHON  = os.environ.get("ROS_PYTHON", "python3")
# Directory holding ik_solver_rtb.py and camera1_cam.json
SAM6D_DIR   = os.environ.get("SAM6D_DIR", os.path.dirname(os.path.abspath(__file__)))
IK_SOLVER   = os.path.join(SAM6D_DIR, "ik_solver_rtb.py")
CAMERA_JSON = os.environ.get("CAMERA_JSON", os.path.join(SAM6D_DIR, "camera1_cam.json"))
# -----------------------------------------------------------------------
#!/usr/bin/env python3
"""
record_demo.py  —  Human demonstration recorder
Run on robot PC in isaac_py310 conda env.

Usage:
  python record_demo.py              # records to demos/demo_001/
  python record_demo.py --name grasp # records to demos/grasp/

Controls:
  Enter  — stop recording
  Ctrl+C — cancel without saving
"""

import argparse, os, time
import numpy as np
import panda_py
from panda_py import controllers

# -- optional camera — comment out if no RealSense connected --
try:
    import pyrealsense2 as rs
    import cv2
    import threading
    CAMERA = True
except ImportError:
    CAMERA = False
    print("pyrealsense2 not found — recording joints only")

# ==============================================================================
# CONFIG
# ==============================================================================
ROBOT_IP  = PANDA_HOSTNAME
SAVE_DIR  = os.path.join(BASE_DIR, "demos")
os.makedirs(SAVE_DIR, exist_ok=True)


# ==============================================================================
# CAMERA THREAD
# ==============================================================================
def start_camera():
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    for _ in range(10): pipeline.wait_for_frames()
    print("Camera ready.")
    return pipeline, align

frames_rgb   = []
frames_depth = []
frame_times  = []
stop_flag    = [False]

def capture_loop(pipeline, align):
    while not stop_flag[0]:
        f   = pipeline.wait_for_frames()
        a   = align.process(f)
        bgr = np.asanyarray(a.get_color_frame().get_data())
        d   = np.asanyarray(a.get_depth_frame().get_data())
        frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        frames_depth.append(d.copy())
        frame_times.append(time.time())


# ==============================================================================
# RECORD
# ==============================================================================
def record(name: str):
    save_path = os.path.join(SAVE_DIR, name)
    os.makedirs(save_path, exist_ok=True)

    # -- connect robot --------------------------------------
    print(f"\nConnecting to robot at {ROBOT_IP}...")
    robot = panda_py.Panda(ROBOT_IP)
    robot.recover()
    robot.set_default_behavior()
    print(f"Connected.  Current joints: {np.round(robot.q, 3)}")

    # -- optional: start camera ----------------------------
    pipeline = align = cam_thread = None
    if CAMERA:
        pipeline, align = start_camera()
        cam_thread = threading.Thread(
            target=capture_loop, args=(pipeline, align), daemon=True)

    # -- enable teaching mode ------------------------------
    print("\n" + "="*50)
    print(f"  Demo: {name}")
    print("="*50)
    print("\nRobot going into TEACHING MODE — it will feel soft.")
    print("Physically grab the arm and move it through your task.")
    print("Press Enter when done.\n")

    robot.teaching_mode(active=True)
    robot.enable_logging(buffer_size=100000)   # ~100 seconds at 1kHz

    t_start = time.time()

    # start camera after robot is in teaching mode
    if cam_thread:
        stop_flag[0] = False
        cam_thread.start()

    try:
        input(">>> Move robot now.  Press Enter to stop <<<\n")
    except KeyboardInterrupt:
        print("\nCancelled — not saving.")
        robot.teaching_mode(active=False)
        if pipeline: pipeline.stop()
        return

    duration = time.time() - t_start

    # -- stop ----------------------------------------
    robot.teaching_mode(active=False)
    stop_flag[0] = True

    # -- get robot log -------------------------------------
    log      = robot.get_log()
    q_array  = np.array(log["q"])       # (N, 7) joint positions
    dq_array = np.array(log["dq"])      # (N, 7) joint velocities
    tau_array= np.array(log["tau_J"])   # (N, 7) joint torques

    print(f"\nRecorded:")
    print(f"  Duration:  {duration:.1f}s")
    print(f"  Joints:    {len(q_array)} steps  "
          f"({len(q_array)/duration:.0f} Hz)")
    if CAMERA and frames_rgb:
        print(f"  Images:    {len(frames_rgb)} frames  "
              f"({len(frames_rgb)/duration:.0f} Hz)")

    # -- save ----------------------------------------
    np.save(os.path.join(save_path, "joints.npy"),     q_array)
    np.save(os.path.join(save_path, "velocities.npy"), dq_array)
    np.save(os.path.join(save_path, "torques.npy"),    tau_array)

    if CAMERA and frames_rgb:
        if cam_thread: cam_thread.join(timeout=2.0)
        pipeline.stop()
        np.save(os.path.join(save_path, "frames_rgb.npy"),
                np.array(frames_rgb, dtype=np.uint8))
        np.save(os.path.join(save_path, "frames_depth.npy"),
                np.array(frames_depth, dtype=np.uint16))
        np.save(os.path.join(save_path, "frame_times.npy"),
                np.array(frame_times))

    # metadata
    import json
    meta = {
        "name":         name,
        "duration_s":   round(duration, 2),
        "n_joint_steps":len(q_array),
        "joint_hz":     round(len(q_array)/duration, 1),
        "n_frames":     len(frames_rgb) if CAMERA else 0,
        "camera_hz":    round(len(frames_rgb)/duration,1) if CAMERA else 0,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(save_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to: {save_path}/")
    print(f"  joints.npy         {q_array.shape}")
    if CAMERA and frames_rgb:
        print(f"  frames_rgb.npy     ({len(frames_rgb)}, 480, 640, 3)")
        print(f"  frames_depth.npy   ({len(frames_depth)}, 480, 640)")
    print(f"  meta.json")
    print(f"\nDone.\n")


# ==============================================================================
# REPLAY  (optional — verify your recording looks right)
# ==============================================================================
def replay(name: str, speed: float = 1.0):
    save_path = os.path.join(SAVE_DIR, name)
    joints    = np.load(os.path.join(save_path, "joints.npy"))

    print(f"\nReplaying '{name}'  ({len(joints)} steps)")
    print("Robot will move — make sure area is clear.")
    input("Press Enter to start replay, Ctrl+C to cancel\n")

    robot = panda_py.Panda(ROBOT_IP)
    robot.recover()
    robot.set_default_behavior()

    # move to start position first
    print("Moving to start position...")
    robot.move_to_joint_position(joints[0], speed_factor=0.1)
    input("At start position. Press Enter to replay full trajectory\n")

    ctrl = controllers.JointPosition()
    robot.start_controller(ctrl)

    freq = 30.0 * speed   # replay at 30Hz (recorded at ~1kHz, downsample)
    step = max(1, int(len(joints) / (freq * (len(joints) / 1000.0))))

    with robot.create_context(frequency=freq) as ctx:
        i = 0
        while ctx.ok() and i < len(joints):
            ctrl.set_control(joints[i])
            i += step
            print(f"  {i}/{len(joints)}", end="\r")

    robot.stop_controller()
    print("\nReplay done.")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',   default=None,
                        help='demo name, e.g. grasp_bottle')
    parser.add_argument('--replay', action='store_true',
                        help='replay a saved demo instead of recording')
    parser.add_argument('--speed',  type=float, default=1.0,
                        help='replay speed factor (default 1.0)')
    args = parser.parse_args()

    # auto-name if not given
    if args.name is None:
        existing = [d for d in os.listdir(SAVE_DIR)
                    if os.path.isdir(os.path.join(SAVE_DIR, d))]
        idx      = len(existing) + 1
        name     = f"demo_{idx:03d}"
    else:
        name = args.name

    if args.replay:
        replay(name, args.speed)
    else:
        record(name)


if __name__ == "__main__":
    main()
