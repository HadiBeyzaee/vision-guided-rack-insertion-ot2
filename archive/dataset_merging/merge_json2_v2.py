import os

# --- Archived variant. Paths parameterised; otherwise unmodified. ------
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
LLAVA_REPO     = os.environ.get("LLAVA_REPO", "/opt/LLaVA")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
RUN_LLAVA      = os.path.join(LLAVA_REPO, "llava/eval/run_llava.py")
# -----------------------------------------------------------------------
import json

# # # # Define file paths
file1_path = os.path.join(BASE_DIR, "opentron_station31/camera2_crop2_coarse_fine2.json")
file2_path = os.path.join(BASE_DIR, "opentron_station32/camera2_crop2_coarse_fine2.json")

file3_path = os.path.join(BASE_DIR, "opentron_station22/camera2_crop2_coarse_fine2.json")
file4_path = os.path.join(BASE_DIR, "opentron_station21/camera2_crop2_coarse_fine2.json")

file5_path = os.path.join(BASE_DIR, "opentron_station11/camera2_crop2_coarse_fine2.json")
file6_path = os.path.join(BASE_DIR, "opentron_station12/camera2_crop2_coarse_fine2.json")
file7_path = os.path.join(BASE_DIR, "opentron_station13/camera2_crop2_coarse_fine2.json")
file8_path = os.path.join(BASE_DIR, "opentron_station14/camera2_crop2_coarse_fine2.json")

file9_path = os.path.join(BASE_DIR, "opentron_station23/camera2_crop2_coarse_fine2.json")
file10_path = os.path.join(BASE_DIR, "opentron_station15/camera2_crop2_coarse_fine2.json")
file11_path = os.path.join(BASE_DIR, "opentron_station16/camera2_crop2_coarse_fine2.json")

output_path = os.path.join(BASE_DIR, "camera2_crop2_coarse_fine2_merged.json")

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

with open(file9_path, "r") as f9:
    data9 = json.load(f9)

with open(file10_path, "r") as f10:
    data10 = json.load(f10)

with open(file11_path, "r") as f11:
    data11 = json.load(f11)

merged_data = data1 + data2 + data3 +  data4 + data5 +data6 + data7 + data8 + data9 + data10 + data11

# Save the merged JSON
with open(output_path, "w") as output_file:
    json.dump(merged_data, output_file, indent=4)

print(f"\nMerged JSON saved at: {output_path}")
