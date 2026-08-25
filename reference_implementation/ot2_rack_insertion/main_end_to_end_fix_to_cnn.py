import time
from pick_rack_fixed_position import pick_rack
from move_rack_above_target_slot import move_to_slot
from video_recorder import RealSenseRecorder
from cnn_model_misalignment_prediction import start_robot_prediction
from exit_opentron_after_alignment import exit_opentron


TASKS = [
    ("blue", "slot2"),
    ("transparent", "slot6"),
    ("white", "slot4"),
    ("black", "slot3"),
]


def main():
    overall_start = time.time()
    recorder = RealSenseRecorder()  # Reused recorder instance

    for index, (rack_color, slot_id) in enumerate(TASKS, start=1):
        print(f"\nStarting Task {index}: Rack={rack_color}, Slot={slot_id}")
        task_start = time.time()

        # 1) Pick rack
        recorder.start(prefix=f"task{index}_pickRack")
        pick_rack(rack_color)
        recorder.stop()

        # 2) Move to target slot
        recorder.start(prefix=f"task{index}_moveToSlot")
        move_to_slot(slot_id)
        recorder.stop()

        # 3) CNN alignment
        start_robot_prediction()

        # 4) Exit motion / release
        recorder.start(prefix=f"task{index}_exit")
        exit_opentron()
        recorder.stop()

        duration = time.time() - task_start
        print(f"Completed Task {index}: Rack={rack_color}, Slot={slot_id} in {duration:.2f} sec")

    total_time = time.time() - overall_start
    print(f"\nAll tasks completed in {total_time:.2f} sec ({total_time/60:.2f} min)")


if __name__ == "__main__":
    main()
