import os

# --- Connection settings (override in your shell or a .env file) -------
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
# -----------------------------------------------------------------------
import requests
import pyrealsense2 as rs
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import panda_py
from panda_py import libfranka
import roboticstoolbox as rtb
import spatialmath as sm
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3

# === CONFIG ===
server_url = f"http://{INFERENCE_HOST}:6000/pose"
rgb_path = "/tmp/rgb.png"
depth_path = "/tmp/depth.png"
camera_path = os.path.join(BASE_DIR, "run_output")
hostname = PANDA_HOSTNAME

panda = panda_py.Panda(PANDA_HOSTNAME)
panda.set_default_behavior()
# === RealSense Setup ===

panda.move_to_joint_position([1.623086131806959, 0.03188909898855184, -0.08891237463792574, -1.2761273268482132, -0.06930268824100493, 1.39296535815133, 0.7088873520887489], speed_factor=0.05)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

serial_number = "147322070835"  # Replace with your target camera's serial number
ctx = rs.context()

# Look for the device with the correct serial number
found = False
for d in ctx.devices:
    if d.get_info(rs.camera_info.serial_number) == serial_number:
        config.enable_device(serial_number)
        found = True
        break

if not found:
    raise RuntimeError(f"RealSense device with serial number {serial_number} not found.")

pipeline.start(config)

# Warm-up
for _ in range(30):
    pipeline.wait_for_frames()

try:
    # === Capture Frame ===
    frames = pipeline.wait_for_frames()

    align_to = rs.stream.color
    align = rs.align(align_to)
    aligned_frames = align.process(frames)

    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # color_frame = frames.get_color_frame()
    # depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        print("Frame capture failed.")
    else:
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        cv2.imwrite(rgb_path, color_image)
        cv2.imwrite(depth_path, depth_image)
        print(f"Saved RGB to {rgb_path}, Depth to {depth_path}")

        # === Send to Server ===
        print("Sending to pose estimation server...")
        files = {
            'rgb': open(rgb_path, 'rb'),
            'depth': open(depth_path, 'rb'),
            'camera': open(camera_path, 'rb')
        }
        # Add rack name here
        data = {
            'rack': 'white'   # choose: "black", "blue", "white", "transparent"
        }

        
        response = requests.post(server_url, files=files, data=data)
        response_json = response.json()

        if "T_obj_cam" in response_json:
            T_obj_cam = np.array(response_json["T_obj_cam"])


            def rebuild_pose_with_clamped_yaw(T, yaw_range=(-90, 90)):
                # Extract translation
                t = T[:3, 3]

                # Extract and fix rotation
                R_original = T[:3, :3]

                # If Z-axis points into the camera (e.g. z < 0), we flip 180° around X
                z_axis = R_original[:, 2]
                if z_axis[2] < 0:  # object is upside down in camera frame
                    R_flip = R.from_euler('x', 180, degrees=True).as_matrix()
                    R_original = R_original @ R_flip

                # Extract Euler angles
                r = R.from_matrix(R_original)
                roll, pitch, yaw = r.as_euler('xyz', degrees=True)

                # Normalize yaw to [-180, 180]
                yaw = ((yaw + 180) % 360) - 180

                # Clamp yaw to range
                if yaw < yaw_range[0]:
                    yaw += 180
                elif yaw > yaw_range[1]:
                    yaw -= 180

                # Rebuild rotation with corrected roll/pitch/yaw
                R_fixed = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()

                # Build new pose
                T_new = np.eye(4)
                T_new[:3, :3] = R_fixed
                T_new[:3, 3] = t

                return T_new, yaw

            # === Apply clamped yaw pose reconstruction ===
            T_obj_cam_clamped, yaw_deg = rebuild_pose_with_clamped_yaw(T_obj_cam, yaw_range=(-90, 90))

            print("New pose with clamped yaw:\n", T_obj_cam_clamped)
            print(f"Yaw angle (deg): {yaw_deg:.2f}")


            # T_cam_ee = np.array([[-0.02804 ,-0.99743, -0.06598 , 0.07064],
            # [ 0.99961 ,-0.02795, -0.00237, -0.03389],
            # [ 0.00052, -0.06602 , 0.99782, -0.02701],
            # [ 0.   ,    0.   ,    0.    ,   1.     ]])

            T_cam_ee = np.array([[ 0.01684, -0.99885,  0.04483 ,-0.072],
                                [ 0.99984 , 0.01655, -0.00688, -0.035],
                                [ 0.00613 , 0.04494 , 0.99897, -0.00064],
                                [ 0.     ,  0.   ,    0.   ,    1.     ]])          


            T_ee_obj = T_cam_ee @ T_obj_cam_clamped

            # === Step 5: Move robot ===
            panda = panda_py.Panda(PANDA_HOSTNAME)
            pose = panda.get_pose()

            t = np.array([pose[0, 3], pose[1, 3], pose[2, 3]])  # in meters
            R_base_ee = pose[:3, :3]

            T_base_ee = np.eye(4)
            T_base_ee[:3, :3] =  pose[:3, :3] 
            T_base_ee[:3, 3] = t

            T_base_obj = T_base_ee @ T_ee_obj 



            rpy = R.from_matrix(T_base_obj[:3, :3] ).as_euler('xyz', degrees=True)

            print("Object orientation in Base frame:")
            print(f"Roll  (X): {rpy[0]:.2f}°")
            print(f"Pitch (Y): {rpy[1]:.2f}°")
            print(f"Yaw   (Z): {rpy[2]:.2f}°")

            print('T_base_obj: ', T_base_obj)


            rpy_base = R.from_matrix(T_base_obj[:3, :3]).as_euler('xyz', degrees=True)

            roll = -180
            pitch = 0
            yaw = rpy_base[2]

            # # Normalize to [-180, 180]
            # yaw = (yaw + 180) % 360 - 180

            # # Clamp yaw to [-90, 90]
            # if yaw > 90:
            # yaw -= 180
            # elif yaw < -90:
            # yaw += 180

            # Normalize to [-180, 180]
            yaw = (yaw + 180) % 360 - 180

            if yaw < 0:
                yaw = 180 + yaw

            print("Final RPY (deg):", [roll, pitch, yaw])

            # Convert to rotation matrix
            rpy_clamped_rad = np.radians([roll, pitch, yaw])
            rotmat = R.from_euler('xyz', rpy_clamped_rad).as_matrix()

            # Reconstruct full 4x4 transformation matrix
            T_clamped = np.eye(4)
            T_clamped[:3, :3] = rotmat
            T_clamped[:3, 3] = T_base_obj[:3, 3]  # use original position

            T_clamped[2, 3] = 0.23

            print('T_clamped: ', T_clamped)


            Rmat = T_clamped[:3,:3].copy()
            U,_,Vt = np.linalg.svd(Rmat)
            Rfix = U @ Vt
            if np.linalg.det(Rfix) < 0:
                U[:,-1]*=-1
                Rfix = U @ Vt
            T_goal = sm.SE3.Rt(Rfix, T_clamped[:3,3])

            # ---------- RTB Panda model (for IK/FK) ----------
            try:
                robot = rtb.models.ETS.Panda()
            except Exception:
                robot = rtb.models.DH.Panda()

            # If your TCLAM is a TOOL pose (not flange), set flange->tool here:
            # robot.tool = sm.SE3(...)  # <- IMPORTANT if you have a custom EE

            # ---------- Connect to robot & read current joints ----------
            panda = panda_py.Panda(hostname)
            q_start = np.array(panda.get_state().q, float)
            print("q_start:", np.round(q_start, 6))

            # ---------- Solve IK to get q_target (pure IK route) ----------
            def solve_ik(T, q0):
                # full-pose first
                sol = robot.ikine_LM(T, q0=q0, ilimit=300, slimit=150, tol=1e-6)
                if sol.success: return sol.q
                # yaw-only fallback
                sol = robot.ikine_LM(T, q0=q0, mask=[1,1,1, 0,0,1], ilimit=300, slimit=150, tol=1e-6)
                if sol.success: return sol.q
                # try 180° wrist flips (tool frame)
                for rx,ry,rz in [(180,0,0),(0,180,0),(0,0,180)]:
                    Rflip = R.from_euler('xyz', [rx,ry,rz], degrees=True).as_matrix()
                    Tt = sm.SE3(T.A.copy()); Tt.A[:3,:3] = T.A[:3,:3] @ Rflip
                    sol = robot.ikine_LM(Tt, q0=q0, ilimit=300, slimit=150, tol=1e-6)
                    if sol.success: return sol.q
                return None

            q_target = solve_ik(T_goal, q_start)
            if q_target is None:
                raise RuntimeError("IK failed for T_goal (check tool frame or pose).")
            print("q_target (IK):", np.round(q_target, 6))

            # ---------- Pure joint move test ----------
            try:
                panda.move_to_joint_position(q_target, speed_factor=0.02)  # slow & safe
                print("Joint move completed.")
            except Exception as e:
                print("Joint move aborted:", e)


            current_position = panda.get_position()
            new_position = current_position.copy()
            new_position[2] = 0.128
            panda.move_to_pose(new_position, panda.get_orientation(),  speed_factor=0.02)

            gripper = libfranka.Gripper(hostname)

            gripper.grasp(0.05, 0.02, 20, 0.04, 0.04)


        else:
            print("Pose not found in server response:", response_json)

except Exception as e:
    print("Request or control failed:", str(e))

finally:
    pipeline.stop()
    print("RealSense pipeline stopped.")