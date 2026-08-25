"""Apply the nominal-yaw correction to a collected error_data.txt.

The collectors command yaw as `-1.4 + dtheta` (the taught nominal yaw of the
pre-insertion pose), but write only `dtheta` to error_data.txt. Where the label
must express absolute yaw error rather than the commanded increment, this adds
the -1.4 deg offset back in.

Check which convention your dataset needs before running - applying this twice
silently biases every rotation label.
"""

import os

BASE_DIR = os.environ.get("BASE_DIR", "/data/project")

input_path  = os.path.join(BASE_DIR, "error_data.txt")
output_path = os.path.join(BASE_DIR, "corrected_error_data.txt")

correction_angle = -1.4  # degrees, the taught nominal yaw

with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    for line in infile:
        dx, dy, dtheta = map(float, line.strip.split)
        corrected_dtheta = dtheta + correction_angle
        outfile.write(f"{dx:.16f} {dy:.16f} {corrected_dtheta:.16f}\n")

print("Wrote:", output_path)
