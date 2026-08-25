"""Concatenate per-slot dataset JSON files into one training set.

The datasets were collected one folder per OT-2 slot and camera configuration
(`opentron_station11` ... `opentron_station33`), each producing its own JSON.
Training uses the concatenation of those, which is why every dataset name in
the archive ends in `_merged`.

This replaces the archived `merge_json.py`, `merge_json2.py` and
`merge_json3.py`, which were the same list concatenation with different
hard-coded file lists (13, 11 and 8 inputs respectively).

Usage:
    python3 merge_datasets.py out.json in1.json in2.json [in3.json ...]
    python3 merge_datasets.py out.json 'stations/*/camera1_crop2.json'

Reports the per-file and total sample counts so a silently short input is
visible. Analysis only; does not touch the robot.
"""

import glob
import json
import sys


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1

    output_path = argv[1]

    inputs = []
    for pattern in argv[2:]:
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"no match for: {pattern}")
            return 1
        inputs.extend(matched)

    merged = []
    for path in inputs:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"expected a JSON list in {path}, got {type(data).__name__}")
            return 1
        print(f"{len(data):>7}  {path}")
        merged.extend(data)

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=4)

    print(f"{len(merged):>7}  TOTAL -> {output_path}  ({len(inputs)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
