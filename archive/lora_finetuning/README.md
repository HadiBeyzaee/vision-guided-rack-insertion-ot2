# LoRA launch configurations

DeepSpeed command builders for fine-tuning LLaVA-1.5-7B. They differ in rank and alpha, epoch count, batch size and which dataset JSON they point at. One is labelled SHORT-PROMPT-HIGH-ALPHA, which is the pairing it was testing.

The configuration behind the reported models is `../../training/models/finetune_llava.py`.

## Files (11)

- `train_twenty_seven_dtheta.py`
- `train_twenty_seven_dxy.py`
- `train_twenty_seven_together.py`
- `train_twenty_seven_together2.py`
- `train_twenty_seven_together2_v2.py`
- `train_twenty_seven_together3.py`
- `train_twenty_seven_together4.py`
- `train_twenty_seven_together5.py`
- `train_twenty_seven_together6.py`
- `train_twenty_seven_together7.py`
- `train_twenty_seven_together_v2.py`

Archived variants: paths parameterised, otherwise unmodified and not re-tested.
See [`../README.md`](../README.md) for why this folder exists.
