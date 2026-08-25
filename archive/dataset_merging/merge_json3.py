import os

# --- Archived variant. Paths parameterised; otherwise unmodified. ------
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
LLAVA_REPO     = os.environ.get("LLAVA_REPO", "/opt/LLaVA")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
RUN_LLAVA      = os.path.join(LLAVA_REPO, "llava/eval/run_llava.py")
# -----------------------------------------------------------------------
import json


file1_path = os.path.join(BASE_DIR, "external_wrong1/augmented_crop1_wrong.json")
file2_path = os.path.join(BASE_DIR, "external_wrong2/augmented_crop1_wrong.json")
file3_path = os.path.join(BASE_DIR, "external_wrong3/augmented_crop1_wrong.json")

file4_path = os.path.join(BASE_DIR, "opentron_station12/camera1_crop2_wrong_.json")

file5_path = os.path.join(BASE_DIR, "opentron_station21/camera1_crop2_wrong_.json")
file6_path = os.path.join(BASE_DIR, "opentron_station22/camera1_crop2_wrong_.json")
file7_path = os.path.join(BASE_DIR, "opentron_station31/camera1_crop2_wrong_.json")
file8_path = os.path.join(BASE_DIR, "opentron_station32/camera1_crop2_wrong_.json")

output_path = os.path.join(BASE_DIR, "augmented_crop1_wrong_wrong_merged.json")

with open(file1_path, "r") as f1:
    data1 = json.load(f1)

with open(file2_path, "r") as f2:
    data2 = json.load(f2)

with open(file3_path, "r") as f3:
    data3 = json.load(f3)

with open(file4_path, "r") as f4:
    data4 = json.load(f4)

with open(file5_path, "r") as f5:
    data5 = json.load(f5)

with open(file6_path, "r") as f6:
    data6 = json.load(f6)

with open(file7_path, "r") as f7:
    data7 = json.load(f7)

with open(file8_path, "r") as f8:
    data8 = json.load(f8)


merged_data = data1 + data2 + data3 #+  data4 + data5 +data6 + data7 + data8

# Save the merged JSON
with open(output_path, "w") as output_file:
    json.dump(merged_data, output_file, indent=4)

print(f"\nMerged JSON saved at: {output_path}")
