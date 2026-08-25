# Caveats

Things that will cost you time if you do not know them. Most were found by
reading the code rather than running it.

## Faults that were repaired

1. **`training/dataset/rename_images.py` could not run.** Its four path
   constants were commented out, so it raised `NameError` on the first
   reference. Restored as constants derived from `BASE_DIR`.

2. **The tidied servers did not speak the clients' protocol.** They returned
   `{"predicted_label": ...}` while every robot client reads
   `response.json().get("movement")`. A control loop run against them would have
   received `"Unknown"` on every frame and never moved. Both now return
   `movement`, with `predicted_label` kept as an alias.

3. **Port mismatch.** The CNN server defaulted to 4003; its client posts to
   4001. Default is now 4001, overridable with `PORT`.

4. **Hard-coded absolute paths and a lab IP** throughout. All read from
   environment variables now.

5. **A live Slack bot token sat in the source** as a default argument. Removed;
   credentials come from the environment only.

## Two mapping schemas, silently incompatible

`rename_images.py` writes `{original_filename, new_filename, row_index}`.
Every builder in `training/vira_adapters/` reads
`{original_name, new_name, row_number}`.

Feeding one to the other raises a `KeyError` deep inside the builder that names
nothing useful. Both renamers are present;
`rename_images_rowmapping.py` is the one the adapter builders need.

## Three CNN head shapes, none interchangeable

```
1024 -> 27           training/models/train_cnn_27class.py
1024 -> 512  -> 27   train_cnn_27class_as_deployed.py, and the deployed weights
1024 -> 1024 -> 27   archive/inference_servers/server_cnn5.py
```

`serving/serve_cnn_27class.py` reads the head geometry and the backbone out of
the checkpoint rather than assuming either, so any of the three loads. Loading
the wrong one by hand gives a wall of unexpected-key errors that says nothing
about the cause.

The archive also holds two backbones at the same 27-class output: a 177 MB
VGG-19 and a 47 MB ResNet-18. Size alone tells them apart.

## Label thresholds vary between builders

The translation dead band is 0.0004 m in some builders, 0.0005 m in others and
0.0008 m in the per-axis ones. Yaw is 0.1 deg or 0.2 deg. One archived variant
uses `abs(dx) < 0.0`, a dead band of zero, so nothing is ever labelled
`No Move`.

Which threshold a merged dataset was built with cannot be recovered from the
JSON. Do not merge datasets built with different ones.

The `Coarse`/`Fine` boundary is separate again at 5 mm and 1.0 deg. It decides
when the coarse-to-fine loop drops its step size; it is not the `No Move` dead
band.

## The label file has four columns, not three

The single-stage dataset has two independent sources of x error: a grasp offset
applied before the gripper closes, and a placement offset applied after
transfer. The collector writes both:

```
dx1  dx2  dy  dtheta
```

`training/dataset/compute_effective_dx.py` collapses them before the builders
run. **Check the sign before trusting a rebuilt dataset** - the two orderings
differ only in sign, and backwards means every x correction drives the wrong
way.

## Dataset assembly

Collection ran one folder per deck slot and configuration, converted to JSON
separately, then concatenated; hence the `_merged` suffix throughout. The
no-correction subset was captured with the rack already seated and given a
label file of literal zeros, so its true residual is whatever the physical
seating left and is not recorded.

## Latent bugs left in place

These affect training behaviour. They are described rather than patched,
because changing them changes the numbers the models were trained under.

- **`train_cnn_27class.py` trains without augmentation.** `random_split`
  returns `Subset` objects sharing one dataset, so rebinding
  `val_ds.dataset.transform` rebinds it for training too, replacing the
  augmentation pipeline before the first epoch.
- **`train_cnn_coarse_fine.py` has the mirror problem**: it never rebinds, so
  augmentation is applied to validation and test as well.
- **Dead code in the single-stage loop.** `align_and_insert_cnn.py` builds an
  Albumentations pipeline and a Sobel edge image every frame, then sends the
  plain crop and discards both. It costs latency per frame and makes
  `albumentations` a dependency nothing needs.
- **Duplicated configuration.** The same file defines `dx_step`, `dy_step`,
  `dtheta_step` and `dz_drop` twice. The second block wins, so the effective
  drop is 0.081 m, not the 0.09 m in the first.

## Scripts that move the robot on import

Not guarded by `if __name__ == "__main__"`, so importing them is enough:

- `complete_system/retreat_from_deck.py`
- `data_collection/collect_offsets_dual_camera.py`
- `data_collection/collect_offsets_single_camera.py`
- `data_collection/collect_slot_to_slot_trials.py`
- `robot_utils/ik_joint_angles_from_pose.py` (connects and reads state)

The end-to-end runner calls the ones it needs as subprocesses, which is why
this never caused trouble in use.

## Missing pieces

- **No scoring script for physical trials.** Insertion success was counted by
  hand from recorded video. `training/models/evaluate_cnn.py` and
  `evaluate_llava.py` report classification accuracy on held-out images, which
  is a different quantity.
- **Latency was observed, not instrumented.** The `time.time()` calls in the
  control loops are rate limiters. Roughly 6-7 s per VLM query against under a
  second for the CNN, measured by watching rather than logging.
- **The dual-prompt adapter has been deleted**, so
  `vira_coarse_to_fine/servers/serve_dual_prompt.py` cannot run as-is. Its
  dataset builder and evaluator remain, so it can be retrained.
