import os

# --- Archived variant. Connection settings and paths parameterised. ----
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
SLACK_BOT_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "")
# -----------------------------------------------------------------------
#!/usr/bin/env python3
"""
rack_client_robot.py  -  Capture rack image -> analyse -> pick uncapped vials.

Set the config block below, then run:
    python rack_client_robot.py
"""

import socket, struct, pickle, json, os, io
from datetime import datetime
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
import panda_py
from panda_py import libfranka
import roboticstoolbox as rtb
import spatialmath as sm

# ==============================================================================
# NETWORK
# ==============================================================================
SERVER_IP   = '100.68.52.97'   # <- GPU PC IP (same as OT2_IP in robot server)
SERVER_PORT = 5000
RESULTS_DIR = 'results'          # folder created on the robot PC

# ==============================================================================
# RACK CONFIG  <- set before each run
# ==============================================================================
AUTO_DETECT_RACK = False
RACK_TYPE        = 8        # 6, 8, or 18

# Position and heading of the rack centre in the robot base frame.
# x, y, yaw change each time the rack is placed on the table.
RACK_X_M     = -0.0175    # metres
RACK_Y_M     = 0.55    # metres
RACK_YAW_DEG = 90.0     # degrees (0 = rack x-axis aligned with robot x-axis)

# ==============================================================================
# ROBOT CONFIG  <- set before each run
# ==============================================================================
HOSTNAME = PANDA_HOSTNAME

Z_GRASP_M = 0.14    # height of vial top in robot base frame (metres)
Z_ABOVE_M = 0.2    # clearance height above slot before/after grasp (metres)

GRASP_WIDTH_M = 0.02   # gripper finger separation at grasp (metres)
GRASP_SPEED   = 0.02   # m/s
GRASP_FORCE_N = 20     # N
GRASP_EPS     = 0.04   # width tolerance (metres)


# ==============================================================================
# GEOMETRY
# ==============================================================================
def build_T_base_rack(x_m, y_m, yaw_deg):
    """4x4 pose of rack centre in robot base frame (rack is flat: roll=pitch=0)."""
    rot = R.from_euler('z', yaw_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[0, 3] = x_m
    T[1, 3] = y_m
    return T


def slot_xy_in_base(slot, T_base_rack):
    """
    x_m / y_m in the server result are the slot's offset from the rack centre
    along the rack's own axes (metres).  Transform to robot base frame.
    """
    p_rack = np.array([slot['x_m'], slot['y_m'], 0.0, 1.0])
    p_base = T_base_rack @ p_rack
    return float(p_base[0]), float(p_base[1])


def gripper_down_rot(T_base_rack):
    """Rotation: gripper pointing straight down, yaw matching rack heading."""
    yaw = R.from_matrix(T_base_rack[:3, :3]).as_euler('xyz', degrees=True)[2]
    return R.from_euler('xyz', [-180.0, 0.0, yaw], degrees=True).as_matrix()


# ==============================================================================
# IK  (disabled for dry-run - re-enable on robot PC)
# ==============================================================================
def solve_ik(T_goal, q0):
    try:
        robot = rtb.models.ETS.Panda()
    except Exception:
        robot = rtb.models.DH.Panda()
    sol = robot.ikine_LM(T_goal, q0=q0, ilimit=300, slimit=150, tol=1e-6)
    if sol.success:
        return sol.q
    sol = robot.ikine_LM(T_goal, q0=q0, mask=[1, 1, 1, 0, 0, 1],
                         ilimit=300, slimit=150, tol=1e-6)
    return sol.q if sol.success else None


# ==============================================================================
# NETWORK HELPERS
# ==============================================================================
def send_frame(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack('!I', len(data)))
    sock.sendall(data)

def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        pkt = sock.recv(n - len(buf))
        if not pkt:
            return None
        buf.extend(pkt)
    return bytes(buf)

def recv_json(sock):
    hdr = _recvall(sock, 4)
    if hdr is None:
        return None
    data = _recvall(sock, struct.unpack('!I', hdr)[0])
    return json.loads(data.decode('utf-8')) if data else None


# ==============================================================================
# CAPTURE + ANALYSE
# ==============================================================================
def _open_realsense():
    """Start RealSense pipeline, resetting hardware once if the camera is stuck."""
    import time as _time
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline = rs.pipeline()
    try:
        pipeline.start(cfg)
        return pipeline
    except RuntimeError as e:
        print(f"  Camera open failed ({e})")
        print("  Attempting hardware reset - make sure the robot server is not holding the camera.")
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                dev.hardware_reset()
        except Exception:
            pass
        _time.sleep(3)
        pipeline = rs.pipeline()
        pipeline.start(cfg)   # raises if still broken
        return pipeline


def capture_and_analyse():
    """Capture a single RealSense frame and send to server for analysis."""
    pipeline = _open_realsense()
    align = rs.align(rs.stream.color)

    print("Warming up camera...")
    for _ in range(30):
        pipeline.wait_for_frames()

    print("Capturing frame...")
    rgb_np = depth_np = None
    try:
        for _ in range(5):   # flush auto-exposure
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if cf and df:
                bgr      = np.asanyarray(cf.get_data())
                depth_mm = np.asanyarray(df.get_data())
                rgb_np   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                depth_np = depth_mm.copy()
    finally:
        pipeline.stop()

    if rgb_np is None:
        print("Failed to capture frame.")
        return None, None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(RESULTS_DIR, f"raw_capture_{ts}.png")
    cv2.imwrite(raw_path, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
    print(f"  Raw capture saved -> {raw_path}")

    print("  Connecting to server and sending...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(120)
    sock.connect((SERVER_IP, SERVER_PORT))
    try:
        send_frame(sock, {"rack_type": RACK_TYPE, "auto_detect": AUTO_DETECT_RACK})
        send_frame(sock, rgb_np)
        send_frame(sock, depth_np)
        result    = recv_json(sock)
        img_bytes = _recvall(sock, struct.unpack('!I', _recvall(sock, 4))[0])
        img_bytes = pickle.loads(img_bytes) if img_bytes else b""
    finally:
        sock.close()
    return result, img_bytes


# ==============================================================================
# PICK ONE VIAL  (commented out - uncomment on robot PC)
# ==============================================================================
def pick_vial(panda, gripper, slot, T_base_rack):
    x_b, y_b = slot_xy_in_base(slot, T_base_rack)
    rot = gripper_down_rot(T_base_rack)

    T_above = np.eye(4); T_above[:3, :3] = rot; T_above[:3, 3] = [x_b, y_b, Z_ABOVE_M]
    T_grasp = np.eye(4); T_grasp[:3, :3] = rot; T_grasp[:3, 3] = [x_b, y_b, Z_GRASP_M]

    # 1. Open gripper wide before approaching
    print(f"  [{slot['id']}] opening gripper")
    gripper.move(0.04, 0.1)

    # 2. Move above the slot
    print(f"  [{slot['id']}] above  ({x_b:.3f}, {y_b:.3f}, {Z_ABOVE_M:.3f}) m")
    q0      = np.array(panda.get_state().q, dtype=float)
    q_above = solve_ik(sm.SE3(T_above), q0)
    if q_above is None:
        raise RuntimeError(f"IK failed (above) for {slot['id']}")
    panda.move_to_joint_position(q_above, speed_factor=0.05)

    # 3. Lower to grasp height (IK - avoids joint limits from Cartesian control)
    print(f"  [{slot['id']}] lowering to z={Z_GRASP_M:.3f} m")
    q_grasp = solve_ik(sm.SE3(T_grasp), q_above)
    if q_grasp is None:
        raise RuntimeError(f"IK failed (grasp) for {slot['id']}")
    panda.move_to_joint_position(q_grasp, speed_factor=0.02)

    # 4. Close gripper to grasp vial
    print(f"  [{slot['id']}] grasping")
    gripper.grasp(GRASP_WIDTH_M, GRASP_SPEED, GRASP_FORCE_N, GRASP_EPS, GRASP_EPS)

    # 5. Lift back up (IK - same reason)
    print(f"  [{slot['id']}] lifting")
    q_lift = solve_ik(sm.SE3(T_above), q_grasp)
    if q_lift is None:
        raise RuntimeError(f"IK failed (lift) for {slot['id']}")
    panda.move_to_joint_position(q_lift, speed_factor=0.02)


# ==============================================================================
# DRY-RUN: print grasp poses in terminal
# ==============================================================================
def print_grasp_pose(slot, T_base_rack):
    x_b, y_b = slot_xy_in_base(slot, T_base_rack)
    rot = gripper_down_rot(T_base_rack)
    roll, pitch, yaw = R.from_matrix(rot).as_euler('xyz', degrees=True)

    print(f"\n  Slot {slot['id']}  (state: {slot['state']})")
    print(f"    Above  : x={x_b:.4f}  y={y_b:.4f}  z={Z_ABOVE_M:.4f} m")
    print(f"    Grasp  : x={x_b:.4f}  y={y_b:.4f}  z={Z_GRASP_M:.4f} m")
    print(f"    RPY    : roll={roll:.1f}°  pitch={pitch:.1f}°  yaw={yaw:.1f}°")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    T_base_rack = build_T_base_rack(RACK_X_M, RACK_Y_M, RACK_YAW_DEG)

    print("=" * 60)
    print(f"  Rack   : {RACK_TYPE}-well  ({RACK_X_M} m, {RACK_Y_M} m)  yaw={RACK_YAW_DEG}°")
    print(f"  Server : {SERVER_IP}:{SERVER_PORT}")
    print("=" * 60)

    # -- robot init (uncomment on robot PC) ------------------------------------
    panda   = panda_py.Panda(HOSTNAME)
    gripper = libfranka.Gripper(HOSTNAME)
    panda.set_default_behavior()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    while True:
        cmd = input("\nPress Enter to capture and analyse  (q + Enter to quit): ").strip().lower()
        if cmd == 'q':
            break

        result, img_bytes = capture_and_analyse()

        if result is None:
            print("Capture failed.")
            continue

        # -- Save results ----------------------------------------
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(RESULTS_DIR, f"result_{ts}.json")
        png_path  = os.path.join(RESULTS_DIR, f"result_{ts}.png")

        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved JSON -> {json_path}")

        if img_bytes:
            with open(png_path, 'wb') as f:
                f.write(img_bytes)
            print(f"  Saved image -> {png_path}")
            try:
                img_arr = mpimg.imread(io.BytesIO(img_bytes), format='png')
                fig, ax = plt.subplots(figsize=(14, 7))
                fig.patch.set_facecolor('#0d0d0d')
                ax.imshow(img_arr); ax.axis('off')
                plt.tight_layout(pad=0.3)
                fig.canvas.mpl_connect(
                    'key_press_event',
                    lambda e: plt.close('all') if e.key in ('q', 'escape') else None)
                plt.show(block=False)
                plt.pause(0.1)
            except Exception as _e:
                print(f"  (Could not display image: {_e})")

        if not result or not result.get('success'):
            print(f"Analysis failed: {result.get('error') if result else 'no response'}")
            continue

        summary = result['summary']
        print(f"\nRack {result['rack_type']}-well  |  "
              f"occupied={summary['occupied']}  uncapped={summary['uncapped']}")

        uncapped = [s for s in result['slots']
                    if s['state'] == 'uncapped']

        if not uncapped:
            print("No uncapped vials found.")
        else:
            print(f"\nUncapped vials: {[s['id'] for s in uncapped]}")
            print("\n--- Grasp poses in robot base frame ---")
            for slot in uncapped:
                print_grasp_pose(slot, T_base_rack)
                pick_vial(panda, gripper, slot, T_base_rack)  # uncomment on robot PC

        print("\n--- Enter to capture next rack  |  q to quit ---")

    print("\nDone.")


if __name__ == "__main__":
    main()
