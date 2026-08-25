import os

# --- Archived variant. Connection settings and paths parameterised. ----
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
SLACK_BOT_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "")
# -----------------------------------------------------------------------
"""
ot2_controller.py
-----------------
Runs on the OT-2 (or your PC for testing).

Flow for 4 rounds:
    OT-2 -> robot : "prepare_racks"  round=1   (initial load)
    robot -> OT-2 : "racks_ready"              (robot finished)
    OT-2 waits 10 seconds                     (experiment running)

    OT-2 -> robot : "prepare_racks"  round=2   (swap)
    robot -> OT-2 : "racks_ready"
    OT-2 waits 10 seconds

    ... rounds 3 and 4 same ...

    OT-2 -> robot : "cleanup_racks"            (dispose all, no new racks)
    robot -> OT-2 : "cleanup_complete"
    Done.

Run in terminal 2 (after robot_server.py is already running in terminal 1):
    python ot2_controller.py
"""

import time
import threading
import requests
from flask import Flask, request, jsonify

# =============================================================================
# CONFIG
# =============================================================================

ROBOT_IP     = "127.0.0.1"   # change to robot PC IP on the real system
ROBOT_PORT   = 5001

OT2_HOST     = "0.0.0.0"
OT2_PORT     = 5000

TOTAL_ROUNDS        = 4
EXPERIMENT_DURATION = 10      # seconds to wait between rounds (simulates experiment)
ROBOT_TIMEOUT       = 1800    # max seconds to wait for robot to finish one round

# =============================================================================
# CALLBACK SERVER  (receives messages FROM the robot)
# =============================================================================

app                  = Flask(__name__)
_racks_ready_event   = threading.Event()
_cleanup_done_event  = threading.Event()
_robot_status        = {}


@app.route("/racks_ready", methods=["POST"])
def racks_ready():
    data = request.json or {}
    _robot_status.update(data)
    print(f"\n  Robot says: racks ready  "
          f"(round={data.get('round')}  status={data.get('status')})")
    _racks_ready_event.set()
    return jsonify({"status": "ok"})


@app.route("/cleanup_complete", methods=["POST"])
def cleanup_complete():
    data = request.json or {}
    _robot_status.update(data)
    print(f"\n  Robot says: cleanup complete  (status={data.get('status')})")
    _cleanup_done_event.set()
    return jsonify({"status": "ok"})


def _start_callback_server():
    """Run the Flask callback server in a background thread."""
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)   # silence Flask request logs
    app.run(host=OT2_HOST, port=OT2_PORT, debug=False, use_reloader=False)

# =============================================================================
# SEND HELPERS
# =============================================================================

def send_to_robot(endpoint: str, payload: dict) -> bool:
    """POST a message to the robot server. Returns True if accepted."""
    try:
        r = requests.post(
            f"http://{ROBOT_IP}:{ROBOT_PORT}/{endpoint}",
            json=payload,
            timeout=15,
        )
        if r.status_code == 200:
            print(f"  Sent /{endpoint} -> robot accepted")
            return True
        else:
            print(f"   Robot rejected /{endpoint}: {r.text}")
            return False
    except Exception as e:
        print(f"  Could not reach robot server: {e}")
        return False


def wait_for_robot(event: threading.Event, description: str) -> str:
    """Block until the robot sends its callback. Returns status string."""
    print(f"  Waiting for robot to finish ({description})...")
    triggered = event.wait(timeout=ROBOT_TIMEOUT)
    event.clear()
    if not triggered:
        print(f"  Timeout - robot did not respond within {ROBOT_TIMEOUT}s")
        return "timeout"
    return _robot_status.get("status", "unknown")


def check_robot_alive() -> bool:
    try:
        r = requests.get(f"http://{ROBOT_IP}:{ROBOT_PORT}/health", timeout=5)
        busy = r.json().get("busy", False)
        print(f"  Robot reachable - busy={busy}")
        return True
    except Exception as e:
        print(f"  Robot not reachable: {e}")
        return False

# =============================================================================
# EXPERIMENT PLACEHOLDER
# =============================================================================

def run_experiment(round_num: int):
    """
    Replace this with your real OT-2 protocol call.
    For now it just counts down EXPERIMENT_DURATION seconds.
    """
    print(f"\n  Running experiment (round {round_num})...")
    for remaining in range(EXPERIMENT_DURATION, 0, -1):
        print(f"     {remaining}s remaining...", end="\r")
        time.sleep(1)
    print(f"  Experiment {round_num} done                    ")

# =============================================================================
# MAIN LOOP
# =============================================================================

def run():
    print("\n" + "=" * 55)
    print(f"  OT-2 CONTROLLER  -  {TOTAL_ROUNDS} rounds")
    print("=" * 55)

    # Start callback server so robot can reach us
    threading.Thread(target=_start_callback_server, daemon=True).start()
    time.sleep(0.5)
    print(f"Callback server listening on port {OT2_PORT}\n")

    # Check robot is up
    print("Checking robot server...")
    if not check_robot_alive():
        print("Robot not reachable. Is robot_server.py running?")
        return

    # -- Main experiment loop ----------------------------------------
    for round_num in range(1, TOTAL_ROUNDS + 1):
        print(f"\n{'='*55}")
        print(f"  ROUND {round_num} of {TOTAL_ROUNDS}")
        print(f"{'='*55}")

        # Tell robot to prepare racks for this round
        print(f"\nOT-2 -> Robot: prepare racks (round {round_num})")
        ok = send_to_robot("prepare_racks", {
            "round":        round_num,
            "total_rounds": TOTAL_ROUNDS,
        })
        if not ok:
            print("Aborting - could not send to robot.")
            return

        # Wait for robot to finish
        status = wait_for_robot(_racks_ready_event, f"round {round_num} rack prep")
        print(f"  Robot finished with status: {status}")

        # Run the experiment
        run_experiment(round_num)

    # -- After last experiment: tell robot to clean up -----------------
    print(f"\n{'='*55}")
    print(f"  ALL EXPERIMENTS DONE - sending cleanup request")
    print(f"{'='*55}")

    print(f"\nOT-2 -> Robot: cleanup racks (dispose all, no new racks)")
    ok = send_to_robot("cleanup_racks", {"total_rounds": TOTAL_ROUNDS})
    if ok:
        status = wait_for_robot(_cleanup_done_event, "final disposal")
        print(f"  Cleanup finished with status: {status}")
    else:
        print("  Could not send cleanup request - dispose manually.")

    print("\n" + "=" * 55)
    print("  SERIES COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    run()
