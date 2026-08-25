# Control loop variants run on the robot

Closed-loop alignment scripts as used during testing. Each is a different combination of crop rectangle, camera and server
set. The crop and port together identify which model a script was driving.

| File | Crop (t,b,l,r) | Ports | Camera |
| --- | --- | --- | --- |
| `llava_test_opentron_full_two_stage.py` | 440,80,370,230 | 5091, 5011, 5015 | camera1 |
| `llava_test_opentron_full_two_stage2.py` | 100,410,380,230 | 5092, 5012, 5003 | camera2 |
| `llava_test_opentron_full2.py` | 170,410,460,310 | 5002 | camera2 |
| `llava_test_opentron_full3.py` | 450,140,460,310 | 5003 | camera1 |
| `llava_test_opentron_full4.py` | 360,150,445,305 | 5004 | camera2 |
| `llava_test_opentron_full5.py` | 360,150,445,305 | 5005 | camera1 |
| `llava_test_opentron_cnn.py` | 410,100,480,280 | 4002 | camera1 |
| `test_cnn_classification_dxy.py` | 80,450,520,255 | 4010 | camera1 |
| `test_cnn_classification_dxy2.py` | 240,320,500,200 | 4001 | camera2 |
| `test_cnn_classification_dxy3.py` | 220,250,450,150 | 4002 | camera2 |
| `table_cnn_classification.py` | 340,250,460,170 | 4000 | - |

The maintained versions of the two configurations that were kept are
`../../vira_coarse_to_fine/align_coarse_to_fine_llava.py` (440,80,380,230 with
5091/5012/5000) and `../../complete_system/align_and_insert_cnn.py`
(240,320,500,200 with 4001).

**These move a real robot.** Paths and connection settings are
parameterised; the code is otherwise as it ran.
