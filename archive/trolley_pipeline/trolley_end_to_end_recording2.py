import time
import subprocess
from trolley_pick_table_top_slot4 import pick_Rack
from trolley_joints_movement_opentron2 import move_to_slot
from trolley_record_realsense2 import RealSenseRecorder   # <-- streaming recorder class

tasks = [
    ("blue", "slot2"),
    ("transparent", "slot6"),
    ("white", "slot4"),
    ("black", "slot3"),
]

def main():
    overall_start = time.time()
    recorder = RealSenseRecorder()   # one recorder object reused

    for i, (rack, slot) in enumerate(tasks, start=1):
        print(f"\nStarting task {i}: Rack={rack}, Slot={slot}")
        task_start = time.time()

        # 1) Pick rack (recorded)
        recorder.start(prefix=f"task{i}_pickRack")
        pick_Rack(rack)
        recorder.stop()

        # 2) Move to slot (recorded)
        recorder.start(prefix=f"task{i}_moveToSlot")
        move_to_slot(slot)
        recorder.stop()

        # 3) CNN correction (already records inside, don’t wrap)
        subprocess.run(["python3", "trolley_test_cnn_classification_dxy3.py"], check=True)

        # 4) Exit (recorded)
        recorder.start(prefix=f"task{i}_exit")
        subprocess.run(["python3", "trolley_exit_opentron.py"], check=True)
        recorder.stop()

        print(f"Finished task {i}: Rack={rack}, Slot={slot} | ⏱{time.time()-task_start:.2f} sec")

    overall_end = time.time()
    print(f"\nAll tasks completed in {(overall_end - overall_start):.2f} seconds "
          f"({(overall_end - overall_start)/60:.2f} min).")

if __name__ == "__main__":
    main()
