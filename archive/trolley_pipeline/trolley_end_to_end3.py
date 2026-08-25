# controller.py

import subprocess
# from trolley_pose_estimation_rack2 import run_pose_estimation
from trolley_pose_estimation_rack_grasp import run_pose_estimation
from trolley_joints_movement_opentron2 import move_to_slot
import time

tasks = [
    ("blue", "slot1"),
    ("transparent", "slot3"),
    ("white", "slot5"),
    ("black", "slot6"),

]

def main():

    overall_start = time.time()  
    for rack, slot in tasks:
        print(f"\nStarting task: Rack={rack}, Slot={slot}")
        task_start = time.time()  # ⏱start task timer
        # 1) Detect rack pose
        run_pose_estimation(rack_color=rack)

        # 2) Move robot to slot
        move_to_slot(slot)

        # 3) Run CNN correction script (blocking call)
        subprocess.run(["python3", "trolley_test_cnn_classification_dxy2.py"], check=True)

        # 4) Run exit script (blocking call)
        subprocess.run(["python3", "trolley_exit_opentron.py"], check=True)

        print(f"Finished task: Rack={rack}, Slot={slot}")
        task_end = time.time()
        task_duration = task_end - task_start
        print(f"Finished task: Rack={rack}, Slot={slot} | ⏱Duration: {task_duration:.2f} sec")

    overall_end = time.time()
    overall_duration = overall_end - overall_start
    print(f"\nAll tasks completed in {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes).")

if __name__ == "__main__":
    main()
