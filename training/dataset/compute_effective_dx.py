"""Collapse the 4-column CASE label file to the 3-column form the builders read.

The single-stage dataset (the single-stage study) has two independent sources of x error:
a grasp offset applied before the gripper closes, and a placement offset applied
after transfer above the slot. The collector therefore writes FOUR columns:

    dx1  dx2  dy  dtheta

where dx1 is the grasp offset (constant within a trial) and dx2 is the
post-grasp placement offset. Only their combination is observable in the image,
so this reduces them to the single effective displacement of the effective-displacement step and emits
the three columns every dataset builder expects:

    (dx1 - dx2)  dy  dtheta

SIGN CONVENTION - CHECK BEFORE USE. One convention writes this as
`dx = dx2 - dx1`; this script computes `dx1 - dx2`, the negation. Which one is
correct depends on which column the collector wrote first, and the archived
label files cannot settle it on their own. Getting it backwards trains a model
that drives every x correction the wrong way. Verify against a known-offset
capture before trusting a rebuilt dataset.
"""

import os

BASE_DIR = os.environ.get("BASE_DIR", "/data/project")

input_path  = os.path.join(BASE_DIR, "error_data.txt")
output_path = os.path.join(BASE_DIR, "error_data_effective_dx.txt")


def process_file(input_path, output_path):
    n = 0
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            dx1, dx2, dy, dtheta = map(float, line.strip.split)
            f_out.write(f"{dx1 - dx2} {dy} {dtheta}\n")
            n += 1
    print(f"Wrote {n} rows -> {output_path}")


if __name__ == "__main__":
    process_file(input_path, output_path)
