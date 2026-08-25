# Archive inventory - every OT-2 script found

This repository is a curated subset. This file is the full picture, so that
anything left out is *visible* rather than lost.

## How this was produced

All Python files under the OT-2 and VLM/CNN working directories on the local
machine and both external drives were collected, deduplicated by content hash,
and filtered to those that actually mention the OT-2, a deck slot, a rack, a
correction label, LLaVA, VGG/ResNet, `panda_py` or SAM-6D.

```
2177  Python files found across all sources
1064  unique by content hash
 826  content-relevant to this line of work
 257  specific to the OT-2 (mention opentron / slot / deck)   <- listed below
 569  the earlier rack-holder rig, LLaVA_NEW and train_new_data
      (these belong to the a separate repository, not here)
```

The 257 below are the set. This repository carries a curated
selection of them: one representative per distinct function, chosen by reading
the contents rather than the filenames. Most of the remainder are the same
script with different hard-coded paths, thresholds, crop margins or checkpoint
names - the working history of tuning each stage.

**If you want any specific variant pulled in, it is listed here with its full
path.**

## Scope note

This is a survey of what exists on the author's machine, **not** a list of what
belongs in this repository. The filter was "does this script mention the OT-2",
which also catches the geometry-based line - edge and slot-line detection,
spring-corner matching, the two-stage geometric correction. Those files appear
in the listing below but are deliberately **not** in this repository; they
belong to `geometry-based-rack-alignment-ot2`.

Entries under `github_ot2_recovery`, and anything named `*edge*`, `*spring*`,
`recovery_*`, `analyse_fyn*` or `step1_detect_marker_*`, belong to that separate project.

The paths are the author's own drive layout, recorded so that any variant can be
located again.

## Why most variants were not carried over

The archive contains long families of near-identical files: `server_together_opentron`
through `...7`, `llava_twenty_seven_together` in eight forms, `train_cnn` through
`train_cnn3`. They differ in a checkpoint name, a crop rectangle, or a prompt
string. Committing all of them would bury the versions that produced the
reported results. The selection rule was: keep the variant the deployed system
actually called, plus any variant that represents a *different idea* rather
than a different number.

## Counts by source

| Source | Unique OT-2 scripts |
| --- | --- |
| `/media/hadi/New Volume1/opentron` | 126 |
| `/media/hadi/Windows1/opentron_codes` | 73 |
| `/media/hadi/New Volume1/github_ot2_recovery` | 28 |
| `/media/hadi/Windows1/opentron` | 23 |
| `/home/hadi/OT2_Misalignment_Correction` | 7 |

## Counts by detected function

Tags come from content inspection, not filenames.

| Function | Count |
| --- | --- |
| other | 69 |
| augment | 35 |
| server | 27 |
| dataset-build | 25 |
| eval | 19 |
| robot | 16 |
| lora-ft | 15 |
| crop | 11 |
| train augment eval | 11 |
| train augment | 8 |
| merge | 5 |
| server robot sam6d slack | 5 |
| slack | 4 |
| robot sam6d slack | 3 |
| robot slack | 3 |
| crop augment | 1 |

## Full listing

Sorted by function, then name.

| Function | File | Location |
| --- | --- | --- |
| augment | `augmentation_llava.py` | `/media/hadi/New Volume1/opentron` |
| augment | `augmentation_llava2.py` | `/media/hadi/New Volume1/opentron` |
| augment | `augmentation_opentron.py` | `/media/hadi/Windows1/opentron` |
| augment | `augmentation_opentron2.py` | `/media/hadi/Windows1/opentron` |
| augment | `edge_detection.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `edge_detection3.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `edge_detection4.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `edge_detection5.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `edge_detection6.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `edge_detection7.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `edge_detection7_all.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `edge_detection8.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `function_angle_of_slot_line.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `green_detection_final.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `green_detection_final2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `green_detection_final3.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom/line_spring_codes` |
| augment | `green_detection_final3.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `green_detection_final4.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `green_detection_final_only4image.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom/line_spring_codes` |
| augment | `green_detection_final_only4image.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `line_edge_detection.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `spring_matching5.py` | `/media/hadi/New Volume1/opentron` |
| augment | `test_new_green_founder2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `yolo_spring_box_heatmap.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom/line_spring_codes` |
| augment | `yolo_spring_box_heatmap.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `yolo_spring_box_heatmap2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `yolo_spring_box_heatmap2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom/line_spring_codes` |
| augment | `yolo_spring_box_heatmap3.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `yolo_spring_box_heatmap4_all.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `yolo_spring_box_heatmap4_subprocess.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `yolo_spring_box_heatmap4_subprocess2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `yolo_spring_box_heatmap4_subprocess3_all.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `yolo_spring_box_heatmap4_subprocess3_all2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| augment | `yolo_spring_box_heatmap4_well.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| crop | `save_cut.py` | `/media/hadi/Windows1/opentron` |
| crop | `save_cut.py` | `/media/hadi/Windows1/opentron_codes` |
| crop | `save_cut2.py` | `/media/hadi/Windows1/opentron` |
| crop | `save_cut_wrong.py` | `/media/hadi/New Volume1/opentron/codes_opentron_newvolume` |
| crop | `save_cut_wrong.py` | `/media/hadi/New Volume1/opentron` |
| crop | `save_cut_wrong.py` | `/media/hadi/Windows1/opentron_codes` |
| crop | `save_cut_wrong.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom/line_spring_codes` |
| crop | `save_cut_wrong.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| crop | `save_cut_wrong2.py` | `/media/hadi/New Volume1/opentron` |
| crop | `save_cut_wrong3.py` | `/media/hadi/New Volume1/opentron` |
| crop | `save_cut_wrong3.py` | `/media/hadi/New Volume1/opentron/codes_opentron_newvolume` |
| crop augment | `save_blur_cut.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `classification_dxyt.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `cnn_json.py` | `/media/hadi/Windows1/opentron` |
| dataset-build | `cnn_json_to_json.py` | `/media/hadi/New Volume1/opentron` |
| dataset-build | `cnn_json_to_json.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_compare.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_dtheta.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_dtheta_aug.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_dxy.py` | `/media/hadi/Windows1/opentron` |
| dataset-build | `llava_dxy_aug.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_second_offset_big_yes_no.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_second_offset_big_yes_no2.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_to_qwen.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_twenty_seven_together.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_twenty_seven_together2.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_twenty_seven_together_aug.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_twenty_seven_together_aug_multi.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_twenty_seven_together_aug_multi2.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_twenty_seven_together_aug_wrong.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_twenty_seven_together_multi.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_twenty_seven_together_multi2.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_twenty_seven_together_nine.py` | `/media/hadi/Windows1/opentron_codes` |
| dataset-build | `llava_wrong copy.py` | `/media/hadi/New Volume1/opentron` |
| dataset-build | `llava_wrong.py` | `/media/hadi/New Volume1/opentron` |
| dataset-build | `llava_wrong2.py` | `/media/hadi/New Volume1/opentron` |
| dataset-build | `llava_wrong4.py` | `/media/hadi/New Volume1/opentron` |
| eval | `new_pose_using_rectangle3.py` | `/home/hadi/OT2_Misalignment_Correction` |
| eval | `test_amount_offset.py` | `/media/hadi/Windows1/opentron_codes` |
| eval | `test_cnn_new.py` | `/media/hadi/Windows1/opentron_codes` |
| eval | `test_cnn_new2.py` | `/media/hadi/Windows1/opentron` |
| eval | `test_cnn_new3.py` | `/media/hadi/Windows1/opentron_codes` |
| eval | `test_cnn_new3.py` | `/media/hadi/Windows1/opentron` |
| eval | `test_cnn_new_coarse_fine.py` | `/media/hadi/Windows1/opentron_codes` |
| eval | `test_llava.py` | `/media/hadi/New Volume1/opentron` |
| eval | `test_nine.py` | `/media/hadi/Windows1/opentron_codes` |
| eval | `test_twenty_seven_dtheta.py` | `/media/hadi/Windows1/opentron_codes` |
| eval | `test_twenty_seven_dxy.py` | `/media/hadi/Windows1/opentron_codes` |
| eval | `test_twenty_seven_together.py` | `/media/hadi/Windows1/opentron_codes` |
| eval | `test_twenty_seven_together2.py` | `/media/hadi/Windows1/opentron` |
| eval | `test_twenty_seven_together2.py` | `/media/hadi/Windows1/opentron_codes` |
| eval | `test_twenty_seven_together3.py` | `/media/hadi/Windows1/opentron` |
| eval | `test_twenty_seven_together3.py` | `/media/hadi/Windows1/opentron_codes` |
| eval | `test_twenty_seven_together_merged.py` | `/media/hadi/Windows1/opentron` |
| eval | `test_twenty_seven_together_merged.py` | `/media/hadi/Windows1/opentron_codes` |
| eval | `train_cnn_coarse_fine2.py` | `/media/hadi/Windows1/opentron_codes` |
| lora-ft | `train_offset_amount.py` | `/media/hadi/Windows1/opentron_codes` |
| lora-ft | `train_offset_amount2.py` | `/media/hadi/Windows1/opentron_codes` |
| lora-ft | `train_offset_amount3.py` | `/media/hadi/Windows1/opentron_codes` |
| lora-ft | `train_offset_amount4.py` | `/media/hadi/Windows1/opentron_codes` |
| lora-ft | `train_twenty_seven_dtheta.py` | `/media/hadi/Windows1/opentron_codes` |
| lora-ft | `train_twenty_seven_dxy.py` | `/media/hadi/Windows1/opentron` |
| lora-ft | `train_twenty_seven_together.py` | `/media/hadi/Windows1/opentron` |
| lora-ft | `train_twenty_seven_together.py` | `/media/hadi/New Volume1/opentron` |
| lora-ft | `train_twenty_seven_together2.py` | `/media/hadi/Windows1/opentron` |
| lora-ft | `train_twenty_seven_together2.py` | `/media/hadi/New Volume1/opentron` |
| lora-ft | `train_twenty_seven_together3.py` | `/media/hadi/Windows1/opentron` |
| lora-ft | `train_twenty_seven_together4.py` | `/media/hadi/Windows1/opentron` |
| lora-ft | `train_twenty_seven_together5.py` | `/media/hadi/Windows1/opentron` |
| lora-ft | `train_twenty_seven_together6.py` | `/media/hadi/Windows1/opentron` |
| lora-ft | `train_twenty_seven_together7.py` | `/media/hadi/Windows1/opentron` |
| merge | `merge_json.py` | `/media/hadi/Windows1/opentron_codes` |
| merge | `merge_json.py` | `/media/hadi/New Volume1/opentron` |
| merge | `merge_json2.py` | `/media/hadi/Windows1/opentron_codes` |
| merge | `merge_json2.py` | `/media/hadi/New Volume1/opentron` |
| merge | `merge_json3.py` | `/media/hadi/Windows1/opentron_codes` |
| other | `aling_rename.py` | `/media/hadi/New Volume1/opentron` |
| other | `analyse_fyn.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `analyse_fyn2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `analyse_fyn3.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `augmentation_llava3.py` | `/media/hadi/New Volume1/opentron` |
| other | `background_sam3_test.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `centre_to_centre.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `compare_centres.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `correct_error.py` | `/media/hadi/Windows1/opentron_codes` |
| other | `detect_spring_to_grasp.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `detect_spring_to_grasp2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `edge_detection2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `error_data_align.py` | `/media/hadi/New Volume1/opentron` |
| other | `extract_templates.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `extract_templates2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `function_slot_corner.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `lengh_find.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `llava_merge.py` | `/media/hadi/Windows1/opentron_codes` |
| other | `new_pose_using_rectangle.py` | `/home/hadi/OT2_Misalignment_Correction` |
| other | `new_pose_using_rectangle2.py` | `/home/hadi/OT2_Misalignment_Correction` |
| other | `old_frame_on_rectangle_rounded2.py` | `/home/hadi/OT2_Misalignment_Correction` |
| other | `rack_finder.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `rack_finder2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `rename_images.py` | `/media/hadi/Windows1/opentron_codes` |
| other | `rename_unique.py` | `/media/hadi/Windows1/opentron` |
| other | `rename_unique.py` | `/media/hadi/New Volume1/opentron` |
| other | `rgb_gray.py` | `/media/hadi/New Volume1/opentron` |
| other | `sam3_1x4_pipeline.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `sam3_background.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `sam3_batch_remove_background.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `save_correction_directions.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `single_spring_finder.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `spring_corner_horizental_vline.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `spring_corner_line.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `spring_finder.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `spring_finder2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `spring_finder3.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `spring_matching.py` | `/media/hadi/New Volume1/opentron` |
| other | `spring_matching2.py` | `/media/hadi/New Volume1/opentron` |
| other | `spring_matching3.py` | `/media/hadi/New Volume1/opentron` |
| other | `spring_matching4.py` | `/media/hadi/New Volume1/opentron` |
| other | `temp_ext.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `test_analyse.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `test_analyse2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `test_analyse3.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `test_coarse_fine.py` | `/media/hadi/Windows1/opentron_codes` |
| other | `test_coarse_fine2.py` | `/media/hadi/Windows1/opentron_codes` |
| other | `test_fyn_orb.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `test_fyn_slot_corner.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `test_fyn_slot_corner2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `test_fyn_slot_corner3.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `test_fyn_slot_corner_yolo.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `test_fyn_slot_corner_yolo.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom/line_spring_codes` |
| other | `test_multi_two_questions.py` | `/media/hadi/Windows1/opentron_codes` |
| other | `test_multi_two_questions2.py` | `/media/hadi/Windows1/opentron_codes` |
| other | `test_regression.py` | `/media/hadi/Windows1/opentron_codes` |
| other | `test_speck_yolo.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `test_speck_yolo2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `test_spring_detection.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `update_txt.py` | `/media/hadi/New Volume1/opentron` |
| other | `video_corner_spring.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `video_corner_spring2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `video_corner_spring_dxyt.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `video_corner_spring_dxyt2.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| other | `yolo_spring_upper_lower.py` | `/media/hadi/New Volume1/opentron/rack_slot_red_bottom` |
| robot | `main_detect_marker_replace_rack.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot | `main_detect_marker_replace_rack2.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot | `main_marker_edge_correction.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot | `main_marker_edge_correction2.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot | `main_marker_edge_correction3.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot | `move_inside_ot2_above_target_slot.py` | `/home/hadi/OT2_Misalignment_Correction/error_recovery_6D_grasp` |
| robot | `move_inside_ot2_above_target_slot.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot | `move_to_pose.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot | `recovery_main.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot | `recovery_main.py` | `/home/hadi/OT2_Misalignment_Correction/error_recovery_fixed_grasp` |
| robot | `recovery_main2.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot | `recovery_main3.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot | `recovery_main_6D.py` | `/home/hadi/OT2_Misalignment_Correction/error_recovery_6D_grasp` |
| robot | `recovery_main_6d.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot | `recovery_main_6d_2.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot | `step1_detect_marker_grasp_rack.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot sam6d slack | `main_marker_end_to_end_sam6d.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot sam6d slack | `main_marker_end_to_end_sam6d_all.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot sam6d slack | `test_rack_swap.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot slack | `step1_detect_marker_grasp_rack2.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot slack | `step1_detect_marker_grasp_using_rectangle.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| robot slack | `step1_detect_marker_grasp_using_rectangle_new.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| server | `server_amount.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `serve_cnn_27class.py` | `/media/hadi/New Volume1/opentron` |
| server | `server_cnn1.py` | `/media/hadi/Windows1/opentron` |
| server | `server_cnn2.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_cnn2.py` | `/media/hadi/New Volume1/opentron` |
| server | `server_cnn2.py` | `/media/hadi/New Volume1/opentron/codes_opentron_newvolume` |
| server | `server_cnn3.py` | `/media/hadi/New Volume1/opentron` |
| server | `server_cnn4.py` | `/media/hadi/New Volume1/opentron` |
| server | `server_cnn5.py` | `/media/hadi/New Volume1/opentron` |
| server | `server_cnn_cf.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_cnn_single.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_cnn_single2.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_cnn_works_new_camera.py` | `/media/hadi/New Volume1/opentron` |
| server | `serve_llava_27class.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_rotation_opentron.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_together_opentron.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_together_opentron2.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_together_opentron3.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_together_opentron4.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_together_opentron5.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_together_opentron6.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_together_opentron7.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_together_opentron_coarse_fine.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_together_opentron_coarse_fine2.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_together_opentron_multi.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_transition_opentron.py` | `/media/hadi/Windows1/opentron_codes` |
| server | `server_vlm.py` | `/media/hadi/New Volume1/opentron` |
| server robot sam6d slack | `test_rack_swap_final.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| server robot sam6d slack | `test_rack_swap_final2.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| server robot sam6d slack | `test_rack_swap_final3.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| server robot sam6d slack | `test_rack_swap_final_new.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| server robot sam6d slack | `test_rack_swap_final_new2.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| slack | `ot2_controller.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| slack | `ot2_controller2.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| slack | `ot2_controller3.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| slack | `recovery_rack_edge_detection_using_sam3.py` | `/media/hadi/New Volume1/github_ot2_recovery` |
| train augment | `train_cnn_regression.py` | `/media/hadi/New Volume1/opentron` |
| train augment | `train_cnn_regression2.py` | `/media/hadi/New Volume1/opentron` |
| train augment | `train_cnn_regression3.py` | `/media/hadi/New Volume1/opentron` |
| train augment | `train_cnn_regression4.py` | `/media/hadi/New Volume1/opentron` |
| train augment | `train_regression.py` | `/media/hadi/Windows1/opentron_codes` |
| train augment | `train_regression.py` | `/media/hadi/Windows1/opentron` |
| train augment | `train_regression1.py` | `/media/hadi/Windows1/opentron_codes` |
| train augment | `train_regression2.py` | `/media/hadi/Windows1/opentron_codes` |
| train augment eval | `train_cnn_27class.py` | `/media/hadi/Windows1/opentron` |
| train augment eval | `train_cnn2.py` | `/media/hadi/Windows1/opentron_codes` |
| train augment eval | `train_cnn3.py` | `/media/hadi/Windows1/opentron_codes` |
| train augment eval | `train_cnn_classification.py` | `/media/hadi/New Volume1/opentron` |
| train augment eval | `train_cnn_classification2.py` | `/media/hadi/New Volume1/opentron` |
| train augment eval | `train_cnn_classification3.py` | `/media/hadi/New Volume1/opentron` |
| train augment eval | `train_cnn_classification4.py` | `/media/hadi/New Volume1/opentron` |
| train augment eval | `train_cnn_classification5.py` | `/media/hadi/New Volume1/opentron` |
| train augment eval | `train_cnn_classification6.py` | `/media/hadi/New Volume1/opentron` |
| train augment eval | `train_cnn_classification8.py` | `/media/hadi/New Volume1/opentron` |
| train augment eval | `train_cnn_coarse_fine.py` | `/media/hadi/Windows1/opentron_codes` |
