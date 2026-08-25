import time
import subprocess
import signal
from trolley_pose_estimation_rack_grasp2 import run_pose_estimation
from trolley_joints_movement_opentron2 import move_to_slot
from trolley_record_realsense2 import RealSenseRecorder   # <-- use streaming version

tasks = [
    ("transparent", "slot6"),
    ("blue", "slot2"),
    ("white", "slot4"),
    ("black", "slot3"),
]


def main():
    overall_start = time.time()
    recorder = RealSenseRecorder()   # one recorder object reused

    for rack, slot in tasks:
        print(f"\nStarting task: Rack={rack}, Slot={slot}")
        task_start = time.time()

        # 1) Detect rack pose (already records inside)
        run_pose_estimation(rack_color=rack)

        # 2) Move robot to slot (record around it)
        recorder.start(prefix=f"move_slot_{slot}")
        move_to_slot(slot)
        recorder.stop()

        # 3) Run CNN correction script (already records inside)
        subprocess.run(["python3", "trolley_test_cnn_classification_dxy3.py"], check=True)

        # 4) Exit slot (record around it)
        recorder.start(prefix=f"exit_{slot}")
        subprocess.run(["python3", "trolley_exit_opentron.py"], check=True)
        recorder.stop()

        print(f"Finished task: Rack={rack}, Slot={slot}")
        task_end = time.time()
        print(f"⏱Duration: {task_end - task_start:.2f} sec")

    overall_end = time.time()
    print(f"\nAll tasks completed in {(overall_end - overall_start):.2f} sec "
          f"({(overall_end - overall_start)/60:.2f} min).")

if __name__ == "__main__":
    main()
