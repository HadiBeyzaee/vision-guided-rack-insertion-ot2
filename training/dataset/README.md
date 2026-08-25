# Dataset assembly

The 10,000-image datasets were not collected in one run. Each OT-2 slot and
camera configuration was collected into its own folder - `opentron_station11`
through `opentron_station33` in the archive - converted to JSON separately, and
only then concatenated. That is why every dataset name in the archive ends in
`_merged`.

This folder holds the steps that sit either side of the per-folder builders in
[`../`](../) and [`../vira/`](../vira_adapters/).

## The no-correction subset

The chapter's stopping behaviour is taught by images that need **no** correction
 -  1,500 samples in the ViRA set, 2,000 in the single-stage set. Those were not
produced by commanding a small offset. They were captured with the rack already
seated correctly, and then given a label file of zeros:

```
renumber_images.py       aligned captures -> 1.png, 2.png, ... N.png
make_aligned_labels.py   writes N rows of "0.0 0.0 0.0"
```

The result flows into the normal builders, which classify `(0, 0, 0)` as
`No Move, No Move, No Rotate`. Note that these samples are described as
including *small nonzero* offsets up to ±0.5 mm and ±0.1°; the label file this
script writes is exactly zero, and the real residual is whatever the physical
seating left behind.

## Merging

```bash
python3 merge_datasets.py merged.json 'stations/*/camera1_crop2_coarse_fine.json'
```

`merge_datasets.py` replaces the archived `merge_json.py`, `merge_json2.py` and
`merge_json3.py` - all three were the same list concatenation with different
hard-coded file lists. It prints per-file counts so a short input is visible
rather than silently halving your dataset.

Merge only files built with the **same prompt and the same label set**. Mixing a
16-label coarse dataset into a 27-label fine one produces a JSON that trains
without complaint and a model that cannot stop.

## Files

| File | What it does |
| --- | --- |
| `renumber_images.py` | Copy a folder of captures to `1.png ... N.png` in sorted order. |
| `make_aligned_labels.py` | Write one `0.0 0.0 0.0` row per image, for the no-correction subset. |
| `merge_datasets.py` | Concatenate N dataset JSON files into one, with counts. |
