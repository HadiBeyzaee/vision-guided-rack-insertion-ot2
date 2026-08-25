import os

# --- Archived variant. Paths parameterised; otherwise unmodified. ------
BASE_DIR       = os.environ.get("BASE_DIR", "/data/project")
LLAVA_REPO     = os.environ.get("LLAVA_REPO", "/opt/LLaVA")
LLAVA_BASE     = os.environ.get("LLAVA_BASE", "/data/llava/llava-v1.5-7b")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
RUN_LLAVA      = os.path.join(LLAVA_REPO, "llava/eval/run_llava.py")
# -----------------------------------------------------------------------
import json


file1_path = os.path.join(BASE_DIR, "camera2_dxyt1/camera2_crop2_align.json")
file2_path = os.path.join(BASE_DIR, "camera2_dxyt1/camera2_crop2.json")

file3_path = os.path.join(BASE_DIR, "camera2_dxyt2/camera2_crop2_align.json")
file4_path = os.path.join(BASE_DIR, "camera2_dxyt2/camera2_crop2.json")

file5_path = os.path.join(BASE_DIR, "camera2_dxyt3/camera2_crop2_align.json")
file6_path = os.path.join(BASE_DIR, "camera2_dxyt3/camera2_crop2.json")

#file7_path = os.path.join(BASE_DIR, "camera2_dxyt4/camera2_crop2_align.json")
file8_path = os.path.join(BASE_DIR, "camera2_dxyt4/camera2_crop2.json")

file9_path = os.path.join(BASE_DIR, "opentron_wrong_grasp_base5/augmented_camera1_crop1_align.json")
file10_path = os.path.join(BASE_DIR, "opentron_wrong_grasp_base5/augmented_camera1_crop1.json")

# file11_path = os.path.join(BASE_DIR, "opentron_wrong_grasp_base1/augmented_camera1_crop1_align.json")
# file12_path = os.path.join(BASE_DIR, "opentron_wrong_grasp_base1/augmented_camera1_crop1.json")

# file13_path = os.path.join(BASE_DIR, "opentron_wrong_grasp_base2/augmented_camera1_crop1_align.json")
# file14_path = os.path.join(BASE_DIR, "opentron_wrong_grasp_base2/augmented_camera1_crop1.json")

# file15_path = os.path.join(BASE_DIR, "opentron_wrong_grasp_base3/augmented_camera1_crop1_align.json")
# file16_path = os.path.join(BASE_DIR, "opentron_wrong_grasp_base3/augmented_camera1_crop1.json")

output_path = os.path.join(BASE_DIR, "camera2_crop2_dxyt_1234.json")

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

# with open(file7_path, "r") as f7:
# data7 = json.load(f7)

with open(file8_path, "r") as f8:
    data8 = json.load(f8)

with open(file9_path, "r") as f9:
    data9 = json.load(f9)

with open(file10_path, "r") as f10:
    data10 = json.load(f10)

# with open(file11_path, "r") as f11:
# data11 = json.load(f11)

# with open(file12_path, "r") as f12:
# data12 = json.load(f12)

# with open(file13_path, "r") as f13:
# data13 = json.load(f13)

# with open(file14_path, "r") as f14:
# data14 = json.load(f14)

# with open(file15_path, "r") as f15:
# data15 = json.load(f15)

# with open(file16_path, "r") as f16:
# data16 = json.load(f16)


merged_data = data1 + data2 + data3 +  data4+ data5 + data6 +  data8 #+ data9 +  data10 #+  data11 + data12 +  data13 +  data14 + data15 +  data16

# Save the merged JSON
with open(output_path, "w") as output_file:
    json.dump(merged_data, output_file, indent=4)

print(f"\nMerged JSON saved at: {output_path}")
