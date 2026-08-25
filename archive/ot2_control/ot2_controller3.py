import os

# --- Archived variant. Connection settings and paths parameterised. ----
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
SLACK_BOT_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "")
# -----------------------------------------------------------------------

import requests
import time
import os
from slack_sdk import WebClient

# sudo ip addr add 169.254.100.1/16 dev enxa0cec870c472
# ================= CONFIGURATION =================
ROBOT_IP = "169.254.91.54"
HEADERS = {"Opentrons-Version": "3"}

CHANNEL_ID = SLACK_CHANNEL_ID
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", SLACK_BOT_TOKEN)

slack_client = WebClient(token=SLACK_BOT_TOKEN)


# ================= SLACK FUNCTIONS =================

def send_slack(message):
    """Send message to Slack"""
    try:
        slack_client.chat_postMessage(channel=CHANNEL_ID, text=message)
        print(f"Slack: {message}")
    except Exception as e:
        print(f"Slack error: {e}")


def wait_for_panda_done():
    """Wait for Panda robot to say 'DONE!' in Slack"""
    print("Waiting for Panda robot to complete...")

    history = slack_client.conversations_history(channel=CHANNEL_ID, limit=1)
    if history["messages"]:
        last_ts = float(history["messages"][0]["ts"])
    else:
        last_ts = time.time()

    while True:
        time.sleep(2)
        try:
            history = slack_client.conversations_history(channel=CHANNEL_ID, limit=5)
            for msg in history["messages"]:
                msg_ts = float(msg["ts"])
                text = msg.get("text", "")

                if msg_ts > last_ts:
                    # ONLY look for the EXACT official message from Panda
                    # Must contain BOTH "DONE" and "ready to continue"
                    if "DONE" in text and "ready to continue" in text:
                        print("Panda robot finished!")
                        return True
        except Exception as e:
            print(f"Polling error: {e}")
            time.sleep(2)


def request_rack_replacement(slot_num, rack_type):
    """Send rack replacement request to Slack and wait for Panda"""

    send_slack(
        f" *OT-2*:  *RACK REPLACEMENT REQUEST*\n"
        f"------------------------------------\n"
        f"• *Slot*: {slot_num}\n"
        f"• *Rack Type*: {rack_type}\n"
        f"• *Reason*: Task complete, need fresh rack\n"
        f"------------------------------------\n"
        f" *@Panda Robot* - Please replace Slot {slot_num}!"
    )

    # Wait for Panda to finish
    wait_for_panda_done()

    send_slack(f" *OT-2*: Slot {slot_num} replacement confirmed! Resuming...")


# ================= OT-2 FUNCTIONS =================

def check_connection():
    """Check if OT-2 is reachable"""
    print("Checking OT-2 connection...")
    for attempt in range(10):
        try:
            response = requests.get(
                f"http://{ROBOT_IP}:31950/health",
                headers=HEADERS,
                timeout=5
            )
            if response.status_code == 200:
                print("OT-2 is online!")
                return True
        except:
            print(f"  Attempt {attempt+1}/10 - Waiting...")
            time.sleep(3)

    print("Cannot reach OT-2")
    return False


def pause_for_panda():
    """Pause OT-2 operations to let Panda work (no homing movement)"""
    print("OT-2 paused for Panda access...")
    print("OT-2 ready for Panda")


def run_protocol(protocol_code):
    """Upload and run protocol on OT-2"""

    if not check_connection():
        return False

    with open("/tmp/protocol.py", "w") as f:
        f.write(protocol_code)

    # Upload
    print("Uploading protocol...")
    try:
        with open("/tmp/protocol.py", "rb") as f:
            response = requests.post(
                f"http://{ROBOT_IP}:31950/protocols",
                headers=HEADERS,
                files=[("files", ("protocol.py", f, "text/plain"))],
                timeout=30
            )

        data = response.json()
        if "data" not in data:
            print(f"Upload failed: {data}")
            return False

        protocol_id = data["data"]["id"]
        print(f"Protocol ID: {protocol_id}")
    except Exception as e:
        print(f"Upload error: {e}")
        return False

    time.sleep(3)

    # Create run
    print("Creating run...")
    try:
        response = requests.post(
            f"http://{ROBOT_IP}:31950/runs",
            headers={**HEADERS, "Content-Type": "application/json"},
            json={"data": {"protocolId": protocol_id}},
            timeout=30
        )

        data = response.json()
        if "data" not in data:
            print(f"Create run failed: {data}")
            return False

        run_id = data["data"]["id"]
        print(f"Run ID: {run_id}")
    except Exception as e:
        print(f"Create run error: {e}")
        return False

    # Start
    print("Starting run...")
    requests.post(
        f"http://{ROBOT_IP}:31950/runs/{run_id}/actions",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"data": {"actionType": "play"}},
        timeout=30
    )

    # Monitor
    print("Running...")
    while True:
        try:
            response = requests.get(
                f"http://{ROBOT_IP}:31950/runs/{run_id}",
                headers=HEADERS,
                timeout=10
            )
            status = response.json()["data"]["status"]
            print(f"  Status: {status}")

            if status in ["succeeded", "failed", "stopped"]:
                break
        except Exception as e:
            print(f"  Monitor error: {e}")

        time.sleep(2)

    return status == "succeeded"


# ================= PROTOCOLS =================

# TASK 1: Pick tip from Slot 1 (A1), go to Slot 2 (Abgene), dispense, drop tip
TASK_1_PROTOCOL = '''
from opentrons import protocol_api

metadata = {
    "protocolName": "Task 1 - Tip to Abgene",
    "author": "Panda",
    "apiLevel": "2.15"
}

def run(protocol: protocol_api.ProtocolContext):
    # Slot 1: Tip rack
    tip_rack = protocol.load_labware("opentrons_96_filtertiprack_200ul", "1")

    # Slot 2: Abgene storage plate
    abgene = protocol.load_labware("nest_96_wellplate_2ml_deep", "2")

    # Load pipette
    pipette = protocol.load_instrument("p300_single_gen2", "left")

    # Pick up tip from A1
    pipette.pick_up_tip(tip_rack["A1"])

    # Move to Abgene plate A1
    pipette.move_to(abgene["A1"].top())

    # Dispense into Abgene
    pipette.dispense(100, abgene["A1"])

    # Drop tip in trash
    pipette.drop_tip()
'''

# TASK 2: Pick tip from Slot 1 (A1 - new rack), go to Slot 2 (Abgene), dispense, drop tip
TASK_2_PROTOCOL = '''
from opentrons import protocol_api

metadata = {
    "protocolName": "Task 2 - Tip to Abgene (new rack)",
    "author": "Panda",
    "apiLevel": "2.15"
}

def run(protocol: protocol_api.ProtocolContext):
    # Slot 1: NEW Tip rack (replaced by Panda)
    tip_rack = protocol.load_labware("opentrons_96_filtertiprack_200ul", "1")

    # Slot 2: Abgene storage plate
    abgene = protocol.load_labware("nest_96_wellplate_2ml_deep", "2")

    # Load pipette
    pipette = protocol.load_instrument("p300_single_gen2", "left")

    # Pick up tip from A1
    pipette.pick_up_tip(tip_rack["A1"])

    # Move to Abgene plate A1
    pipette.move_to(abgene["A1"].top())

    # Dispense into Abgene
    pipette.dispense(100, abgene["A1"])

    # Drop tip in trash
    pipette.drop_tip()
'''


# ================= MAIN WORKFLOW =================

def main():
    print("=" * 60)
    print("OT-2 + PANDA FULL INTEGRATION WORKFLOW")
    print("=" * 60)

    send_slack(
        " *OT-2*:  *STARTING FULL WORKFLOW*\n"
        "------------------------------------\n"
        "Task 1: Tip (Slot 1) -> Abgene (Slot 2)\n"
        "Task 2: Tip (Slot 1) -> Abgene (Slot 2)\n"
        "Rack replacements: Slot 1, Slot 2\n"
        "------------------------------------"
    )

    # =============================================
    # TASK 1: Pick tip from Slot 1, dispense to Slot 2
    # =============================================
    send_slack(" *OT-2*: *TASK 1* - Picking tip from Slot 1, dispensing to Slot 2 (Abgene)...")

    success = run_protocol(TASK_1_PROTOCOL)

    if not success:
        send_slack(" *OT-2*: Task 1 FAILED!")
        return

    send_slack(" *OT-2*: Task 1 complete!")

    # =============================================
    # REPLACEMENT 1: Replace Slot 1 tip rack
    # =============================================
    send_slack(" *OT-2*: Pausing for Panda to replace Slot 1...")
    pause_for_panda()

    request_rack_replacement(
        slot_num=1,
        rack_type="Tip Rack 200µL"
    )

    # =============================================
    # TASK 2: Pick tip from NEW Slot 1 rack, dispense to Slot 2
    # =============================================
    send_slack(" *OT-2*: *TASK 2* - Picking tip from Slot 1 (new rack), dispensing to Slot 2 (Abgene)...")

    success = run_protocol(TASK_2_PROTOCOL)

    if not success:
        send_slack(" *OT-2*: Task 2 FAILED!")
        return

    send_slack(" *OT-2*: Task 2 complete!")

    # =============================================
    # REPLACEMENT 2: Replace Slot 2 Abgene plate
    # =============================================
    send_slack(" *OT-2*: Pausing for Panda to replace Slot 2...")
    pause_for_panda()

    request_rack_replacement(
        slot_num=2,
        rack_type="Abgene Storage Plate (deep well)"
    )

    # =============================================
    # DONE!
    # =============================================
    send_slack(
        " *OT-2*: *FULL WORKFLOW COMPLETE!*\n"
        "------------------------------------\n"
        "• Task 1 (Tip -> Abgene): \n"
        "• Slot 1 replacement: \n"
        "• Task 2 (Tip -> Abgene): \n"
        "• Slot 2 replacement: \n"
        "------------------------------------"
    )

    print("\n" + "=" * 60)
    print("FULL WORKFLOW COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()


# curl -X POST -H "Opentrons-Version: 2" -H "Content-Type: application/json" -d '{"on": true}' http://169.254.91.54:31950/robot/lights
#sudo ip addr add 169.254.100.1/16 dev enxa0cec870c472


#sudo ip link set enx00e096691135 up

#sudo ip addr add 169.254.100.1/16 dev enx00e096691135

#curl -X POST -H "Opentrons-Version: 2" -H "Content-Type: application/json" -d '{"on": false}' http://169.254.227.210:31950/robot/lights

