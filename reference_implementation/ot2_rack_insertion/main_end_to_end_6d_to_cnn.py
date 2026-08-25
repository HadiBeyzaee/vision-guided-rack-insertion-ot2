import time
from pose_6d_estimation_ot2_racks import run_pose_estimation
from move_rack_above_target_slot import move_to_slot
from video_recorder import RealSenseRecorder
from cnn_model_misalignment_prediction import start_robot_prediction
from exit_opentron_after_alignment import exit_opentron

TASKS = [
    ("black", "slot6"),
    ("blue", "slot2"),
    ("white", "slot4"),
    ("black", "slot3"),
]


def main():
    overall_start = time.time()
    recorder = RealSenseRecorder()  # Single recorder instance reused

    for rack_color, slot_id in TASKS:
        print(f"\nStarting task: Rack={rack_color}, Slot={slot_id}")
        task_start = time.time()

        # 1) Pose estimation (this step records inside itself)
        run_pose_estimation(rack_color=rack_color)

        # 2) Move robot toward target slot with recording
        recorder.start(prefix=f"move_to_{slot_id}")
        move_to_slot(slot_id)
        recorder.stop()

        # 3) CNN alignment
        start_robot_prediction()

        # 4) Exit and retract robot with recording
        recorder.start(prefix=f"exit_{slot_id}")
        exit_opentron()
        recorder.stop()

        duration = time.time() - task_start
        print(f"Completed task: Rack={rack_color}, Slot={slot_id} in {duration:.2f} sec")

    total_time = time.time() - overall_start
    print(f"\nAll tasks completed in {total_time:.2f} sec ({total_time/60:.2f} min)")


if __name__ == "__main__":
    main()
