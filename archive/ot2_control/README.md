# Opentrons OT-2 control

Driving the liquid handler itself, as opposed to the arm that services it.
This work treats the OT-2 as the target instrument, so these are the scripts
that put it into a known state between insertion trials and confirm the deck
matches what the robot expects.

Included because the insertion task only means anything in the context of a
protocol that then runs.

## Files (4)

- `ot2_controller.py`
- `ot2_controller2.py`
- `ot2_controller3.py`
- `ot2_controller_test.py`

Archived: connection settings and paths parameterised, otherwise unmodified
and not re-tested. See [`../README.md`](../README.md).
