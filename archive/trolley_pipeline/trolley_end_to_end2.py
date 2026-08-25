import subprocess
from trolley_pick_table_top_slot4 import pick_Rack
from trolley_joints_movement_opentron2 import move_to_slot
import time
import os

tasks = [
    ("blue", "slot1"),
    ("transparent", "slot2"),
    ("white", "slot3"),
    ("black", "slot5"),
]

def record_step(prefix, func, *args):
    # Start recorder in background
    record_proc = subprocess.Popen(["python3", "trolley_record_realsense.py", "--prefix", prefix])
    time.sleep(2)  # small delay so recording starts

    try:
        func(*args)
    finally:
        record_proc.terminate()
        record_proc.wait()
        print(f"Finished recording {prefix}")

def main():
    overall_start = time.time()
    for i, (rack, slot) in enumerate(tasks, start=1):
        print(f"\nStarting task {i}: Rack={rack}, Slot={slot}")
        task_start = time.time()

        record_step(f"task{i}_pickRack", pick_Rack, rack)
        record_step(f"task{i}_moveToSlot", move_to_slot, slot)
        record_step(f"task{i}_cnn", subprocess.run, ["python3", "trolley_test_cnn_classification_dxy2.py"],)
        record_step(f"task{i}_exit", subprocess.run, ["python3", "trolley_exit_opentron.py"],)

        print(f"Finished task {i}: Rack={rack}, Slot={slot} | ⏱{time.time()-task_start:.2f} sec")

    print(f"\nAll tasks completed in {time.time()-overall_start:.2f} seconds")

if __name__ == "__main__":
    main()
