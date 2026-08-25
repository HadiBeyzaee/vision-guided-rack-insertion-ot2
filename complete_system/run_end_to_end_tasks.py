"""Orchestrator: run a list of (rack colour, deck slot) tasks back to back.

For each task in TASKS:
  1. localise_and_grasp_sam6d.run_pose_estimation() - find and grasp the rack
  2. transfer_to_deck_slot.move_to_slot()               - carry it to the slot
  3. align_and_insert_cnn.py                  - close the loop and insert
  4. retreat_from_deck.py                           - back off cleanly

Steps 3 and 4 run as subprocesses so a crash in one task does not take the
whole run down. Times each task and the full sequence.

Run:    python3 run_end_to_end_tasks.py
Safety: moves a real robot immediately on execution.
"""

# controller.py

import subprocess
# from trolley_pose_estimation_rack2 import run_pose_estimation
from localise_and_grasp_sam6d import run_pose_estimation
from transfer_to_deck_slot import move_to_slot
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
        task_start = time.time()  # start task timer
        # 1) Detect rack pose
        run_pose_estimation(rack_color=rack)

        # 2) Move robot to slot
        move_to_slot(slot)

        # 3) Run CNN correction script (blocking call)
        subprocess.run(["python3", "align_and_insert_cnn.py"], check=True)

        # 4) Run exit script (blocking call)
        subprocess.run(["python3", "retreat_from_deck.py"], check=True)

        print(f"Finished task: Rack={rack}, Slot={slot}")
        task_end = time.time()
        task_duration = task_end - task_start
        print(f"Finished task: Rack={rack}, Slot={slot} | Duration: {task_duration:.2f} sec")

    overall_end = time.time()
    overall_duration = overall_end - overall_start
    print(f"\nAll tasks completed in {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes).")

if __name__ == "__main__":
    main()
