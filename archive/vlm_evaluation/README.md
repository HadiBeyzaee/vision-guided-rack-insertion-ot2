# Offline VLM scoring

Each shells out to `run_llava.py` per image, rebuilds ground truth from `error_data.txt`, and reports accuracy - overall and per class. They differ in the adapter scored, the prompt sent, and the label set.

All of them measure **classification** accuracy on held-out images. None measures insertion success; the reported success rates were counted by hand from the physical trials.

## Files (13)

- `test_amount_offset.py`
- `test_coarse_fine2.py`
- `test_llava.py`
- `test_multi_two_questions2.py`
- `test_nine.py`
- `test_twenty_seven_dtheta.py`
- `test_twenty_seven_dxy.py`
- `test_twenty_seven_together2.py`
- `test_twenty_seven_together2_v2.py`
- `test_twenty_seven_together3.py`
- `test_twenty_seven_together3_v2.py`
- `test_twenty_seven_together_merged.py`
- `test_twenty_seven_together_merged_v2.py`

Archived variants: paths parameterised, otherwise unmodified and not re-tested.
See [`../README.md`](../README.md) for why this folder exists.
