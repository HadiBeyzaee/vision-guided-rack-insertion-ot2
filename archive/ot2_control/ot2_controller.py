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
    """Wait for Panda robot to say 'done' in Slack"""
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
                text = msg.get("text", "").lower()

                # Check for completion message from Panda
                if msg_ts > last_ts:
                    if "done" in text or "complete" in text or "success" in text or "replaced" in text:
                        print("Panda robot finished!")
                        return True
        except Exception as e:
            print(f"Polling error: {e}")
            time.sleep(2)


# ================= OT-2 FUNCTIONS =================

def home_ot2():
    """Home the OT-2 robot"""
    print("Homing OT-2...")
    requests.post(
        f"http://{ROBOT_IP}:31950/robot/home",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"target": "robot"}
    )
    time.sleep(3)


def run_protocol(protocol_code):
    """Upload and run protocol on OT-2"""

    # Save to temp file
    with open("/tmp/protocol.py", "w") as f:
        f.write(protocol_code)

    # Upload
    print("Uploading protocol...")
    with open("/tmp/protocol.py", "rb") as f:
        response = requests.post(
            f"http://{ROBOT_IP}:31950/protocols",
            headers=HEADERS,
            files=[("files", ("protocol.py", f, "text/plain"))]
        )

    data = response.json()
    if "data" not in data:
        print(f"Upload failed: {data}")
        return False

    protocol_id = data["data"]["id"]
    print(f"Protocol ID: {protocol_id}")

    time.sleep(2)

    # Create run
    print("Creating run...")
    response = requests.post(
        f"http://{ROBOT_IP}:31950/runs",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"data": {"protocolId": protocol_id}}
    )

    data = response.json()
    if "data" not in data:
        print(f"Create run failed: {data}")
        return False

    run_id = data["data"]["id"]
    print(f"Run ID: {run_id}")

    # Start
    print("Starting run...")
    requests.post(
        f"http://{ROBOT_IP}:31950/runs/{run_id}/actions",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"data": {"actionType": "play"}}
    )

    # Monitor
    print("Running...")
    while True:
        response = requests.get(f"http://{ROBOT_IP}:31950/runs/{run_id}", headers=HEADERS)
        status = response.json()["data"]["status"]
        print(f"  Status: {status}")

        if status in ["succeeded", "failed", "stopped"]:
            break
        time.sleep(2)

    return status == "succeeded"


# ================= PROTOCOL =================

PICK_AND_DISPENSE_PROTOCOL = '''
from opentrons import protocol_api

metadata = {
    "protocolName": "Pick Tip and Dispense to Reservoir",
    "author": "Panda",
    "apiLevel": "2.15"
}

def run(protocol: protocol_api.ProtocolContext):
    # Slot 1: Tip rack (200uL filter)
    tip_rack = protocol.load_labware("opentrons_96_filtertiprack_200ul", "1")

    # Slot 2: Reservoir (for dispensing)
    dest_plate = protocol.load_labware("nest_96_wellplate_2ml_deep", "2")

    # Load pipette
    pipette = protocol.load_instrument("p300_single_gen2", "left")

    # Step 1: Pick up tip from A1
    protocol.comment("Picking up tip from Slot 1, A1...")
    pipette.pick_up_tip(tip_rack["A1"])
    protocol.delay(seconds=1)

    # Step 2: Move to Abgene plate
    protocol.comment("Moving to Slot 2 (Abgene), A1...")
    pipette.move_to(dest_plate["A1"].top())
    protocol.delay(seconds=1)

    # Step 4: Dispense into Abgene plate
    protocol.comment("Dispensing 100uL into Abgene A1...")
    pipette.dispense(100, dest_plate["A1"])
    protocol.delay(seconds=1)

    # Step 4: Drop tip in trash
    protocol.comment("Dropping tip in trash...")
    pipette.drop_tip()

    protocol.comment("OT-2 task complete!")
'''


# ================= MAIN =================

def main():
    print("=" * 60)
    print("OT-2 + PANDA INTEGRATION TEST")
    print("=" * 60)

    send_slack(" *OT-2*: Starting integrated workflow!")

    # ===== STEP 1: Do the pipetting task =====
    send_slack(" *OT-2*: Picking tip from Slot 1 (A11) and moving to reservoir...")

    success = run_protocol(PICK_AND_DISPENSE_PROTOCOL)

    if success:
        send_slack(" *OT-2*: Pipetting task complete!")
    else:
        send_slack(" *OT-2*: Pipetting task failed!")
        return

    # ===== STEP 2: Home OT-2 (move out of the way) =====
    send_slack(" *OT-2*: Moving to home position (clearing space for Panda)...")
    home_ot2()

    # ===== STEP 3: Request rack replacement =====
    send_slack(
        " *OT-2*: *RACK REPLACEMENT REQUEST*\n"
        "------------------------------------\n"
        "• *Slot*: 1\n"
        "• *Rack Type*: Tip Rack 200µL\n"
        "• *Reason*: Task complete, need fresh rack\n"
        "------------------------------------\n"
        " *@Panda Robot* - Please replace Slot 1!"
    )

    # # ===== STEP 4: Wait for Panda to finish =====
    # wait_for_panda_done()

    # send_slack(" *OT-2*: Received confirmation from Panda. Rack replaced!")

    # # ===== STEP 5: Done =====
    # send_slack(
    # " *OT-2*: *WORKFLOW COMPLETE!*\n"
    # "• Pipetting: \n"
    # "• Rack replacement: \n"
    # "• Ready for next task!"
    # )

    # print("\n" + "=" * 60)
    # print("WORKFLOW COMPLETE!")
    # print("=" * 60)


if __name__ == "__main__":
    main()
