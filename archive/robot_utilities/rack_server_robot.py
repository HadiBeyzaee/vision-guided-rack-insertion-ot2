import os

# --- Archived variant. Connection settings and paths parameterised. ----
PANDA_HOSTNAME   = os.environ.get("PANDA_HOSTNAME", "192.168.0.1")
INFERENCE_HOST   = os.environ.get("INFERENCE_HOST", "127.0.0.1")
REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
BASE_DIR         = os.environ.get("BASE_DIR", "/data/project")
SLACK_BOT_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "")
# -----------------------------------------------------------------------
#!/usr/bin/env python3
"""
Rack Analysis Server  (GPU PC)
==============================
Listens for RGB+Depth frames from robot PC, runs SAM3 analysis,
returns JSON result. Also shows result figure and blocks until closed.

Auto-detects rack type (6 / 8 / 18) via Gemini on every incoming frame,
OR uses manually configured RACK_TYPE if AUTO_DETECT_RACK = False.
"""

import socket
import struct
import pickle
import json
import sys
import os
import io
import tempfile
import subprocess
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # non-interactive backend - display happens on client
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.environ.get("SAM3_DIR", os.path.join(BASE_DIR, "sam3")))
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ==============================================================================
# NETWORK CONFIG
# ==============================================================================
HOST = '0.0.0.0'
PORT = 5000

# ==============================================================================
# RACK TYPE SERVER DEFAULTS  (overridden per-request by rack_client4.py)
# ==============================================================================
AUTO_DETECT_RACK = False   # fallback if client sends no config
RACK_TYPE        = 8       # fallback if client sends no config

# ==============================================================================
# CAMERA / GEOMETRY CONFIG
# ==============================================================================
CAP_DETECTION_USE_CROP = True
CAP_CROP_UPSCALE       = 5.0

FX = 604.3773803710938;  FY = 604.48193359375
CX = 322.6944274902344;  CY = 247.81814575195312
# Height of vial top above rack surface [m] - same for every well
Z_VIAL_TOP_M = 0.03

# ==============================================================================
# GEMINI AUTO-DETECTION CONFIG  (only used when AUTO_DETECT_RACK = True)
# ==============================================================================
GEMINI_PYTHON = os.environ.get("GEMINI_PYTHON", "python3")

# ==============================================================================
# SLOT TABLES
# ==============================================================================
_GRIPPER_IDS_18 = {'c32', 'd42'}
_COL_NAMES_18   = ['a', 'b', 'c', 'd', 'e', 'f']
_COL_X_CM_18    = [1.5, 5.0, 8.5, 12.0, 15.5, 19.0]
_ROW_Y_CM_18    = [1.3, 3.9, 6.5]

SLOTS_18 = []
for _ci, (_c, _cx) in enumerate(zip(_COL_NAMES_18, _COL_X_CM_18)):
    for _ri, _ry in enumerate(_ROW_Y_CM_18):
        _sid = f"{_c}{_ci+1}{_ri+1}"
        SLOTS_18.append({'id': _sid, 'x_cm': _cx, 'y_cm': _ry,
                         'col_idx': _ci, 'is_gripper': _sid in _GRIPPER_IDS_18})

SLOTS_6 = [
    {'id': 'a11', 'x_cm':  2.0, 'y_cm': 2.0, 'col_idx': 0, 'is_gripper': False},
    {'id': 'a12', 'x_cm':  2.0, 'y_cm': 5.8, 'col_idx': 0, 'is_gripper': False},
    {'id': 'b21', 'x_cm':  5.9, 'y_cm': 3.8, 'col_idx': 1, 'is_gripper': False},
    {'id': 'g1',  'x_cm': 10.0, 'y_cm': 2.0, 'col_idx': 2, 'is_gripper': True},
    {'id': 'g2',  'x_cm': 10.0, 'y_cm': 5.8, 'col_idx': 2, 'is_gripper': True},
    {'id': 'd41', 'x_cm': 14.3, 'y_cm': 3.8, 'col_idx': 3, 'is_gripper': False},
    {'id': 'e51', 'x_cm': 18.0, 'y_cm': 2.0, 'col_idx': 4, 'is_gripper': False},
    {'id': 'e52', 'x_cm': 18.0, 'y_cm': 5.8, 'col_idx': 4, 'is_gripper': False},
]

SLOTS_8 = [
    {'id': 'a11', 'x_cm':  2.0, 'y_cm': 2.0, 'col_idx': 0, 'is_gripper': False},
    {'id': 'a12', 'x_cm':  2.0, 'y_cm': 5.8, 'col_idx': 0, 'is_gripper': False},
    {'id': 'b21', 'x_cm':  5.9, 'y_cm': 2.0, 'col_idx': 1, 'is_gripper': False},
    {'id': 'b22', 'x_cm':  5.9, 'y_cm': 5.8, 'col_idx': 1, 'is_gripper': False},
    {'id': 'g1',  'x_cm': 10.0, 'y_cm': 2.0, 'col_idx': 2, 'is_gripper': True},
    {'id': 'g2',  'x_cm': 10.0, 'y_cm': 5.8, 'col_idx': 2, 'is_gripper': True},
    {'id': 'd41', 'x_cm': 14.3, 'y_cm': 2.0, 'col_idx': 3, 'is_gripper': False},
    {'id': 'd42', 'x_cm': 14.3, 'y_cm': 5.8, 'col_idx': 3, 'is_gripper': False},
    {'id': 'e51', 'x_cm': 18.0, 'y_cm': 2.0, 'col_idx': 4, 'is_gripper': False},
    {'id': 'e52', 'x_cm': 18.0, 'y_cm': 5.8, 'col_idx': 4, 'is_gripper': False},
]

# ==============================================================================
# DYNAMIC RACK CONFIGURATION
# ==============================================================================
def _configure_for_rack_type(rack_type):
    """Set all module-level globals based on the detected rack type."""
    global RACK_TYPE, SLOTS, RACK_W_CM, RACK_H_CM, WELL_R_CM
    global COL_RGB, VIAL_SLOTS, N_VIAL_SLOTS, N_GRIPPER

    assert rack_type in (6, 8, 18), f"Invalid rack type: {rack_type}"
    RACK_TYPE = rack_type

    if RACK_TYPE == 18:
        SLOTS = SLOTS_18
        RACK_W_CM = 20.5
        RACK_H_CM = 7.8
        WELL_R_CM = 1.2
        COL_RGB = {0: (80,140,255), 1: (60,210,80),  2: (255,200,40),
                   3: (255,80,80),  4: (180,80,255), 5: (40,210,200)}
    else:
        SLOTS = SLOTS_6 if RACK_TYPE == 6 else SLOTS_8
        RACK_W_CM = 20.5
        RACK_H_CM = 7.84
        WELL_R_CM = 1.25
        COL_RGB = {0: (80,140,255), 1: (60,210,80), 2: (160,160,160),
                   3: (255,160,40), 4: (180,80,255)}

    VIAL_SLOTS   = [s for s in SLOTS if not s['is_gripper']]
    N_VIAL_SLOTS = len(VIAL_SLOTS)
    N_GRIPPER    = len(SLOTS) - N_VIAL_SLOTS

# Initialize with manual setting or default
_configure_for_rack_type(RACK_TYPE)

VIAL_MATCH_RADIUS_CM = WELL_R_CM * 2.0
CAP_MATCH_FACTOR     = 1.8
SAM3_HOLE_MATCH_PX   = 80

EMPTY, VIAL_CAPPED, VIAL_UNCAPPED, GRIPPER = 'empty', 'capped', 'uncapped', 'gripper'

# ==============================================================================
# GEMINI RACK-TYPE DETECTION  (runs in gemini-robotics env via subprocess)
# ==============================================================================
def _detect_rack_type_gemini(image_path):
    """
    Runs Gemini inside the gemini-robotics conda env via subprocess.
    Returns 6, 8, or 18 (your internal RACK_TYPE values).
    """
    script = f'''import os, json, sys
from google import genai
from google.genai import types

MODEL_NAME = "gemini-robotics-er-1.6-preview"

PROMPT = """You are looking at a photo of a small laboratory rack. Identify the rack type by examining the RACK BODY STRUCTURE, not the vials.

All racks have 2 square gripper slots in the middle column.

- Type 18: Long rack, 6 columns x 3 rows.
- Type 10: Compact rack. 5 columns x 2 rows. All 4 non-gripper columns have positions at BOTH top and bottom heights. Clean 2x4 grid pattern.
- Type 8: Compact rack. 5 columns. Outermost left and right columns have positions at top AND bottom. Inner columns (next to gripper) have position ONLY at middle height. Offset pattern: 2, 1, gripper, 1, 2.

Rules:
1. Look at the plastic body, not glass vials.
2. Infer the pattern from visible holes AND vial positions.
3. A column with a vial at top and nothing at bottom still has 2 positions (one occupied, one empty).

Return ONLY JSON:
{{"rack_type": 10, "reasoning": "brief"}}
If uncertain, default to 10."""

image_path = {repr(image_path)}
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key or not os.path.exists(image_path):
    print(json.dumps({{"rack_type": 18, "reasoning": "missing key or image"}}))
    sys.exit(0)

client = genai.Client(api_key=api_key)
with open(image_path, "rb") as f:
    image_bytes = f.read()

try:
    r = client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/png"), PROMPT],
        config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
    )
    raw = r.text.strip()
    result = json.loads(raw)
    result["rack_type"] = int(result.get("rack_type", 18))
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"rack_type": 18, "reasoning": str(e)}}))
'''
    try:
        proc = subprocess.run(
            [GEMINI_PYTHON, "-"],
            input=script,
            capture_output=True,
            text=True,
            timeout=45,
            env=os.environ,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            lines = [l for l in proc.stdout.strip().splitlines() if l.strip().startswith("{")]
            if lines:
                result = json.loads(lines[-1])
                gemini_type = int(result.get("rack_type", 18))
                reasoning = result.get("reasoning", "")

                # Map Gemini's total-position numbering to your internal RACK_TYPE
                # Gemini: 8 total -> your code calls it 6 (6 vial wells + 2 grippers)
                # Gemini: 10 total -> your code calls it 8 (8 vial wells + 2 grippers)
                # Gemini: 18 total -> your code calls it 18 (16 vial wells + 2 grippers)
                GEMINI_TO_CODE = {8: 6, 10: 8, 18: 18}
                rack_type = GEMINI_TO_CODE.get(gemini_type, 18)

                print(f"  [Gemini] Raw detection: {gemini_type} total positions ({reasoning})")
                print(f"  [Gemini] Mapped to RACK_TYPE: {rack_type}")
                return rack_type
        print(f"  [Gemini] stderr: {proc.stderr.strip()}")
    except Exception as e:
        print(f"  [Gemini] Detection failed: {e}")
    return 18

# ==============================================================================
# ALL ANALYSIS FUNCTIONS
# ==============================================================================
def sam3_find(processor, image_pil, prompt):
    state = processor.set_image(image_pil)
    out   = processor.set_text_prompt(state=state, prompt=prompt)
    n     = out["masks"].shape[0]
    masks = out["masks"].cpu().numpy() if n > 0 else np.array([])
    boxes = out["boxes"].cpu().numpy() if n > 0 else np.array([])
    return n, masks, boxes

def mask_to_binary(mask_np, H, W):
    m2d = mask_np[0] if mask_np.ndim == 3 else mask_np
    b   = (m2d > 0.5).astype(np.uint8) * 255
    if b.shape != (H, W):
        b = cv2.resize(b, (W, H), interpolation=cv2.INTER_NEAREST)
    return b

def box_centre(box):
    x1, y1, x2, y2 = [int(v) for v in box]
    return (x1 + x2) // 2, (y1 + y2) // 2

def _font(size):
    try:    return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except: return ImageFont.load_default()

def find_rack_mask(processor, full_pil):
    W, H = full_pil.size
    pad  = 10
    best_mask, best_bbox, best_area = None, None, 0
    for prompt in ["black holder", "white holder", "rack", "tray"]:
        n, masks, _ = sam3_find(processor, full_pil, prompt)
        if n > 0:
            ri     = int(np.argmax([(m[0] if m.ndim==3 else m).sum() for m in masks]))
            binary = mask_to_binary(masks[ri], H, W)
            area   = int(binary.sum())
            if area > best_area:
                best_area = area
                best_mask = binary
                ys, xs = np.where(binary > 0)
                best_bbox = (max(0,int(xs.min())-pad), max(0,int(ys.min())-pad),
                             min(W,int(xs.max())+pad), min(H,int(ys.max())+pad))
    if best_mask is not None:
        return best_mask, best_bbox
    return None, None

def fit_plane_and_project(rack_mask, depth_img):
    ys, xs  = np.where(rack_mask > 0)
    depths  = depth_img[ys, xs].astype(np.float64)
    valid   = depths > 0
    ys, xs, depths = ys[valid], xs[valid], depths[valid] / 1000.0
    X = (xs - CX) * depths / FX
    Y = (ys - CY) * depths / FY
    Z = depths
    pts3d = np.stack([X, Y, Z], axis=1)
    if len(pts3d) < 20:
        return None, None, None, pts3d, [None]*len(SLOTS)
    centroid = pts3d.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts3d - centroid, full_matrices=False)
    x_ax   = Vt[0].copy()
    normal = Vt[2].copy()
    if normal[2] < 0: normal = -normal
    y_ax = np.cross(normal, x_ax); y_ax /= np.linalg.norm(y_ax)
    x_ax = np.cross(y_ax, normal); x_ax /= np.linalg.norm(x_ax)
    centred = pts3d - centroid
    px = centred @ x_ax;  py = centred @ y_ax
    if x_ax[0] < 0: x_ax = -x_ax; px = -px
    if y_ax[1] < 0: y_ax = -y_ax; py = -py
    origin = centroid + px.min()*x_ax + py.min()*y_ax
    def project(x_cm, y_cm):
        pt = origin + (x_cm/100.0)*x_ax + (y_cm/100.0)*y_ax
        Xp, Yp, Zp = pt
        if Zp <= 0: return None
        return (FX*Xp/Zp + CX, FY*Yp/Zp + CY)
    init_px = [project(s['x_cm'], s['y_cm']) for s in SLOTS]
    return origin, x_ax, y_ax, pts3d, init_px

def detect_wells_hough(rgb_np, bbox, approx_r_px):
    rx1, ry1, rx2, ry2 = bbox
    crop    = rgb_np[ry1:ry2, rx1:rx2]
    gray    = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    inv     = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inv, (5, 5), 0)
    r_min   = max(5,  int(approx_r_px * 0.55))
    r_max   = min(80, int(approx_r_px * 1.45))
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=int(approx_r_px*1.5),
                               param1=60, param2=28,
                               minRadius=r_min, maxRadius=r_max)
    results = []
    if circles is not None:
        for (cx, cy, r) in circles[0]:
            results.append((cx+rx1, cy+ry1, r))
    return results

def detect_wells_sam3(processor, full_pil, bbox, approx_r_px=20.0):
    min_r_px = approx_r_px / WELL_R_CM * 1.0
    rx1, ry1, rx2, ry2 = bbox
    crop_pil = full_pil.crop((rx1, ry1, rx2, ry2))
    for prompt in ["circle holes", "well hole", "circular hole",
                   "well opening", "round hole", "hole", "well"]:
        n, masks, boxes = sam3_find(processor, crop_pil, prompt)
        if n > 0:
            raw, triples = [], []
            for i in range(n):
                x1, y1, x2, y2 = boxes[i]
                cx = (x1+x2)/2.0+rx1;  cy = (y1+y2)/2.0+ry1
                r  = ((x2-x1)+(y2-y1))/4.0
                if r < min_r_px: continue
                raw.append((float(cx), float(cy)))
                triples.append((float(cx), float(cy), float(r)))
            if raw:
                return raw, triples, prompt
    return [], [], None

def get_combined_wells(hough_wells, sam3_triples, approx_r_px):
    all_wells  = hough_wells + sam3_triples
    merged     = []
    merge_dist = approx_r_px * 0.75
    for cx, cy, r in all_wells:
        is_dup = False
        for i, (mcx, mcy, mr) in enumerate(merged):
            if np.sqrt((cx-mcx)**2+(cy-mcy)**2) < merge_dist:
                merged[i] = ((cx+mcx)/2,(cy+mcy)/2,(r+mr)/2)
                is_dup = True; break
        if not is_dup:
            merged.append((cx, cy, r))
    return merged

def refine_with_circles(detected, init_px):
    if len(detected) < 3:
        return None, [], list(init_px)
    det_xy = np.array([(c[0], c[1]) for c in detected])
    ini_xy = np.array([(p[0], p[1]) if p else (-9999,-9999) for p in init_px])
    INF  = 1e9
    cost = np.full((len(ini_xy), len(det_xy)), INF)
    for si in range(len(ini_xy)):
        if ini_xy[si,0] < -100: continue
        for di in range(len(det_xy)):
            d = np.linalg.norm(ini_xy[si] - det_xy[di])
            if d < 120: cost[si,di] = d
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = [(si,di) for si,di in zip(row_ind,col_ind) if cost[si,di] < INF]
    if len(matches) < 3:
        return None, matches, list(init_px)
    src = np.array([ini_xy[si] for si,di in matches], dtype=np.float32)
    dst = np.array([det_xy[di] for si,di in matches], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                        ransacReprojThreshold=15.0)
    if M is None: M, _ = cv2.estimateAffine2D(src, dst)
    if M is None: return None, matches, list(init_px)
    pts   = np.array([(p[0],p[1]) if p else (0,0) for p in init_px], dtype=np.float32)
    pts_h = np.hstack([pts, np.ones((len(pts),1))])
    out   = (M @ pts_h.T).T
    refined = [(float(out[i,0]),float(out[i,1])) if init_px[i] else None
               for i in range(len(init_px))]
    return M, matches, refined

def override_with_sam3_holes(sam3_raw, refined_px):
    if not sam3_raw:
        return list(refined_px), {}
    sam3_xy = np.array(sam3_raw, dtype=np.float64)
    slot_xy = np.array([(p[0],p[1]) if p else (-9999,-9999) for p in refined_px],
                        dtype=np.float64)
    INF  = 1e9
    cost = np.full((len(sam3_xy), len(SLOTS)), INF)
    for hi in range(len(sam3_xy)):
        for si in range(len(SLOTS)):
            if slot_xy[si,0] < -100: continue
            d = np.linalg.norm(sam3_xy[hi] - slot_xy[si])
            if d < SAM3_HOLE_MATCH_PX: cost[hi,si] = d
    row_ind, col_ind = linear_sum_assignment(cost)
    final_px, sam3_matched = list(refined_px), {}
    for hi, si in zip(row_ind, col_ind):
        if cost[hi,si] >= INF: continue
        final_px[si]     = (float(sam3_xy[hi,0]), float(sam3_xy[hi,1]))
        sam3_matched[hi] = si
    return final_px, sam3_matched

def find_vials(processor, full_pil, bbox):
    rx1, ry1, rx2, ry2 = bbox
    crop_pil = full_pil.crop((rx1, ry1, rx2, ry2))
    cW, cH   = crop_pil.size
    for prompt in ["vial", "glass vial", "bottle", "tube"]:
        n, masks, boxes = sam3_find(processor, crop_pil, prompt)
        if n > 0:
            centres, masks_out = [], []
            for vi in range(n):
                vcx, vcy = box_centre(boxes[vi])
                centres.append((vcx+rx1, vcy+ry1))
                masks_out.append(masks[vi])
            return n, masks_out, centres, (cW, cH)
    return 0, [], [], (rx2-rx1, ry2-ry1)

def assign_vials_to_slots(vial_centres_full, final_px):
    n_vials = len(vial_centres_full)
    dists_col = []
    if RACK_TYPE == 18:
        n_rows = len(_ROW_Y_CM_18)
        for ri in range(n_rows):
            for ci in range(len(_COL_NAMES_18) - 1):
                si_a = ci * n_rows + ri
                si_b = (ci+1) * n_rows + ri
                pa, pb = final_px[si_a], final_px[si_b]
                if pa and pb:
                    d = np.sqrt((pb[0]-pa[0])**2 + (pb[1]-pa[1])**2)
                    if 5 < d < 300:
                        dists_col.append(d / (_COL_X_CM_18[ci+1] - _COL_X_CM_18[ci]))
    else:
        for i in range(len(SLOTS) - 1):
            if SLOTS[i]['y_cm'] != SLOTS[i+1]['y_cm']: continue
            pa, pb = final_px[i], final_px[i+1]
            if pa is None or pb is None: continue
            dx_cm = abs(SLOTS[i+1]['x_cm'] - SLOTS[i]['x_cm'])
            if dx_cm == 0: continue
            d = np.sqrt((pb[0]-pa[0])**2 + (pb[1]-pa[1])**2)
            if 5 < d < 300:
                dists_col.append(d / dx_cm)
    px_per_cm       = float(np.median(dists_col)) if dists_col else 15.0
    match_radius_px = VIAL_MATCH_RADIUS_CM * px_per_cm
    occupancy = {}
    for s in SLOTS:
        occupancy[s['id']] = GRIPPER if s['is_gripper'] else EMPTY
    if n_vials == 0:
        return occupancy, {}, px_per_cm
    vial_xy = np.array([(cx,cy) for cx,cy in vial_centres_full], dtype=np.float64)
    active_indices = [i for i, s in enumerate(SLOTS) if not s['is_gripper']]
    slot_xy_active = np.array(
        [(final_px[i][0], final_px[i][1]) if final_px[i] else (-9999,-9999)
         for i in active_indices], dtype=np.float64)
    INF  = 1e9
    cost = np.full((n_vials, len(active_indices)), INF)
    for vi in range(n_vials):
        for aj, si in enumerate(active_indices):
            if slot_xy_active[aj,0] < -100: continue
            d = np.linalg.norm(vial_xy[vi] - slot_xy_active[aj])
            if d < match_radius_px:
                cost[vi, aj] = d
    row_ind, col_ind = linear_sum_assignment(cost)
    vi_to_slot, assigned_slots = {}, set()
    for vi, aj in zip(row_ind, col_ind):
        if cost[vi, aj] < INF:
            slot_id = SLOTS[active_indices[aj]]['id']
            vi_to_slot[vi] = slot_id
            assigned_slots.add(slot_id)
    for vi in range(n_vials):
        if vi in vi_to_slot: continue
        best_si, best_d = None, float('inf')
        for aj, si in enumerate(active_indices):
            sid = SLOTS[si]['id']
            if sid in assigned_slots: continue
            if slot_xy_active[aj,0] < -100: continue
            d = np.linalg.norm(vial_xy[vi] - slot_xy_active[aj])
            if d < best_d:
                best_d, best_si = d, aj
        if best_si is not None and best_d < match_radius_px * 2.0:
            slot_id = SLOTS[active_indices[best_si]]['id']
            vi_to_slot[vi] = slot_id
            assigned_slots.add(slot_id)
    if RACK_TYPE == 18:
        unmatched = [vi for vi in range(n_vials) if vi not in vi_to_slot]
        if unmatched:
            n_rows = len(_ROW_Y_CM_18);  n_cols = len(_COL_NAMES_18)
            col_xs = []
            for ci in range(n_cols):
                pts = [slot_xy_active[aj] for aj, si in enumerate(active_indices)
                       if SLOTS[si]['col_idx'] == ci and slot_xy_active[aj,0] > -100]
                col_xs.append(float(np.mean([p[0] for p in pts])) if pts else None)
            x_dividers = [-np.inf]
            for ci in range(n_cols - 1):
                if col_xs[ci] and col_xs[ci+1]:
                    x_dividers.append((col_xs[ci]+col_xs[ci+1])/2.0)
                else:
                    x_dividers.append(x_dividers[-1] + match_radius_px*2)
            x_dividers.append(np.inf)
            col_buckets = {ci: [] for ci in range(n_cols)}
            for vi in unmatched:
                cx, cy = vial_xy[vi]
                for ci in range(n_cols):
                    if x_dividers[ci] <= cx < x_dividers[ci+1]:
                        col_buckets[ci].append((vi, cy)); break
            for ci, bucket in col_buckets.items():
                if not bucket: continue
                bucket.sort(key=lambda x: x[1])
                for rank, (vi, _) in enumerate(bucket):
                    if rank >= n_rows: continue
                    si_global = ci * n_rows + rank
                    slot_id   = SLOTS[si_global]['id']
                    if SLOTS[si_global]['is_gripper'] or slot_id in assigned_slots: continue
                    vi_to_slot[vi] = slot_id
                    assigned_slots.add(slot_id)
    for vi, slot_id in vi_to_slot.items():
        occupancy[slot_id] = VIAL_UNCAPPED
    return occupancy, vi_to_slot, px_per_cm

def find_caps(processor, full_pil, bbox):
    rx1, ry1, rx2, ry2 = bbox
    W_full, H_full = full_pil.size
    if CAP_DETECTION_USE_CROP:
        crop      = full_pil.crop(bbox)
        S         = CAP_CROP_UPSCALE
        new_w     = int(crop.width  * S)
        new_h     = int(crop.height * S)
        input_pil = crop.resize((new_w, new_h), Image.LANCZOS)
        ox, oy    = rx1, ry1
    else:
        input_pil  = full_pil;  S = 1.0;  ox, oy = 0, 0
    if RACK_TYPE == 18:
        prompts = ["an aluminum lid", "an aluminum cap",
                   "a circular metal lid", "circular aluminum cap"]
    else:
        prompts = ["a vial cap", "plastic cap", "glass vial cap"]
    for prompt in prompts:
        n, masks, boxes = sam3_find(processor, input_pil, prompt)
        if n > 0:
            centres, masks_out = [], []
            crop_w, crop_h = rx2 - rx1, ry2 - ry1
            for ci in range(n):
                ccx, ccy = box_centre(boxes[ci])
                centres.append((ccx/S + ox, ccy/S + oy))
                if CAP_DETECTION_USE_CROP:
                    # mask is in upscaled-crop space -> resize to crop -> place in full image
                    cm = mask_to_binary(masks[ci], new_h, new_w)
                    cm = cv2.resize(cm, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                    full_mask = np.zeros((H_full, W_full), dtype=np.uint8)
                    full_mask[ry1:ry2, rx1:rx2] = cm
                    masks_out.append(full_mask[np.newaxis])
                else:
                    masks_out.append(masks[ci])
            return n, masks_out, centres, (W_full, H_full)
    return 0, [], [], (W_full, H_full)

def assign_caps_to_vials(cap_centres_full, vial_centres_full,
                         vi_to_slot, occupancy, px_per_cm):
    cap_to_vi = {}
    if not cap_centres_full or not vial_centres_full:
        return occupancy, cap_to_vi
    cap_r_px = CAP_MATCH_FACTOR * WELL_R_CM * px_per_cm
    cap_xy   = np.array(cap_centres_full,  dtype=np.float64)
    vial_xy  = np.array(vial_centres_full, dtype=np.float64)
    for ci, cpt in enumerate(cap_xy):
        dists      = np.linalg.norm(vial_xy - cpt, axis=1)
        nearest_vi = int(np.argmin(dists))
        if dists[nearest_vi] < cap_r_px:
            slot_id = vi_to_slot.get(nearest_vi)
            cap_to_vi[ci] = nearest_vi
            if slot_id and occupancy.get(slot_id) == VIAL_UNCAPPED:
                occupancy[slot_id] = VIAL_CAPPED
    return occupancy, cap_to_vi

# ==============================================================================
# DRAWING HELPERS
# ==============================================================================
def estimate_r_px(px_list):
    valid = [p for p in px_list if p is not None]
    if len(valid) < 2: return 16
    dists = []
    for k in range(len(valid) - 1):
        d = np.sqrt((valid[k][0]-valid[k+1][0])**2 + (valid[k][1]-valid[k+1][1])**2)
        if 5 < d < 300: dists.append(d)
    if not dists: return 16
    if RACK_TYPE == 18:
        col_step_cm = _COL_X_CM_18[1] - _COL_X_CM_18[0]
    else:
        ref = [2.0, 5.9, 10.0, 14.3, 18.0]
        col_step_cm = (ref[-1] - ref[0]) / (len(ref) - 1)
    return max(8, int(np.median(dists) * WELL_R_CM / col_step_cm))

def draw_template_overlay(base_np, rack_mask, px_list, large=True, mask_bg=True):
    arr  = base_np.copy()
    if mask_bg: arr[rack_mask == 0] = 0
    r_px = estimate_r_px(px_list)
    f_id = _font(max(11, int(r_px*0.55)));  f_sm = _font(max(9, int(r_px*0.38)))
    pil  = Image.fromarray(arr).convert("RGBA")
    draw = ImageDraw.Draw(pil, 'RGBA')
    for i, slot in enumerate(SLOTS):
        pt = px_list[i]
        if pt is None: continue
        u, v = int(round(pt[0])), int(round(pt[1]))
        if slot['is_gripper']:
            draw.rectangle((u-r_px,v-r_px,u+r_px,v+r_px),
                           fill=(60,60,60,120), outline=(180,180,180,200), width=2)
            draw.line((u-r_px,v-r_px,u+r_px,v+r_px), fill=(180,180,180,180), width=2)
            draw.line((u+r_px,v-r_px,u-r_px,v+r_px), fill=(180,180,180,180), width=2)
            draw.text((u,v), slot['id'], font=f_sm, fill=(220,220,220,200), anchor='mm')
        else:
            rgb = COL_RGB[slot['col_idx']]
            if large:
                draw.ellipse((u-r_px,v-r_px,u+r_px,v+r_px),
                             fill=rgb+(40,), outline=rgb+(240,), width=3)
                draw.ellipse((u-4,v-4,u+4,v+4), fill=(0,255,255,220))
                draw.text((u,v), slot['id'], font=f_id, fill=(255,230,0,230), anchor='mm')
            else:
                draw.ellipse((u-r_px,v-r_px,u+r_px,v+r_px),
                             fill=rgb+(35,), outline=rgb+(180,), width=2)
                cs = 6
                draw.line((u-cs,v,u+cs,v), fill=(0,255,255,255), width=2)
                draw.line((u,v-cs,u,v+cs), fill=(0,255,255,255), width=2)
                draw.text((u,v-r_px-2), slot['id'], font=f_sm,
                          fill=(255,230,0,200), anchor='mb')
    return np.array(pil.convert("RGB")), r_px

def draw_vials_panel(base_np, rack_mask, bbox, vial_masks, vial_centres_full,
                     vi_to_slot, crop_size):
    rx1, ry1, _, _ = bbox
    cW, cH = crop_size
    arr = base_np.copy()
    arr[rack_mask == 0] = 0
    for vi, vmask in enumerate(vial_masks):
        vb      = mask_to_binary(vmask, cH, cW)
        full_vb = np.zeros(base_np.shape[:2], dtype=np.uint8)
        full_vb[ry1:ry1+cH, rx1:rx1+cW] = vb
        colour  = (60,220,60) if vi in vi_to_slot else (220,80,80)
        m = full_vb > 0
        arr[m] = (arr[m]*0.4 + np.array(colour)*0.6).clip(0,255).astype(np.uint8)
    pil  = Image.fromarray(arr).convert("RGBA")
    draw = ImageDraw.Draw(pil, 'RGBA')
    f_id = _font(14)
    for vi, (cx, cy) in enumerate(vial_centres_full):
        slot_id = vi_to_slot.get(vi, '?')
        col = (0,255,255,255) if vi in vi_to_slot else (255,80,80,255)
        u, v = int(cx), int(cy)
        draw.ellipse((u-6,v-6,u+6,v+6), fill=col)
        draw.text((u,v-14), slot_id, font=f_id, fill=col, anchor='mb')
    return np.array(pil.convert("RGB"))

def draw_caps_panel(base_np, rack_mask, cap_masks, cap_centres_full, cap_to_vi):
    H_full, W_full = base_np.shape[:2]
    arr = base_np.copy()
    arr[rack_mask == 0] = 0
    for ci, cmask in enumerate(cap_masks):
        cb = mask_to_binary(cmask, H_full, W_full)
        colour = (0,220,220) if ci in cap_to_vi else (255,140,0)
        m = cb > 0
        arr[m] = (arr[m]*0.35 + np.array(colour)*0.65).clip(0,255).astype(np.uint8)
    pil  = Image.fromarray(arr).convert("RGBA")
    draw = ImageDraw.Draw(pil, 'RGBA')
    f_id = _font(13)
    for ci, (cx, cy) in enumerate(cap_centres_full):
        vi    = cap_to_vi.get(ci)
        label = f"->{vi}" if vi is not None else "?"
        col   = (0,255,255,255) if ci in cap_to_vi else (255,165,0,255)
        u, v  = int(cx), int(cy)
        draw.ellipse((u-5,v-5,u+5,v+5), fill=col)
        draw.text((u,v-13), label, font=f_id, fill=col, anchor='mb')
    return np.array(pil.convert("RGB"))

def draw_final_summary(base_np, rack_mask, final_px, occupancy, sam3_matched=None):
    arr  = base_np.copy()
    arr[rack_mask == 0] = 0
    r_px = estimate_r_px(final_px)
    f_id = _font(max(12, int(r_px*0.60)));  f_sm = _font(max(9, int(r_px*0.40)))
    pil  = Image.fromarray(arr).convert("RGBA")
    draw = ImageDraw.Draw(pil, 'RGBA')
    pinned = set(sam3_matched.values()) if sam3_matched else set()
    for i, slot in enumerate(SLOTS):
        pt    = final_px[i]
        if pt is None: continue
        u, v  = int(round(pt[0])), int(round(pt[1]))
        state = occupancy.get(slot['id'], EMPTY)
        lw    = 4 if i in pinned else 2
        if slot['is_gripper']:
            sq = r_px
            draw.rectangle((u-sq,v-sq,u+sq,v+sq),
                           fill=(50,50,50,160), outline=(150,150,150,220), width=2)
            draw.line((u-sq,v-sq,u+sq,v+sq), fill=(150,150,150,200), width=2)
            draw.line((u+sq,v-sq,u-sq,v+sq), fill=(150,150,150,200), width=2)
            draw.text((u,v), "GRIP", font=f_sm, fill=(200,200,200,230), anchor='mm')
            continue
        if state == VIAL_CAPPED:
            draw.ellipse((u-r_px,v-r_px,u+r_px,v+r_px),
                         fill=(0,200,80,200), outline=(0,255,100,255), width=lw)
            draw.text((u,v), slot['id'], font=f_id, fill=(255,255,255,240), anchor='mm')
        elif state == VIAL_UNCAPPED:
            draw.ellipse((u-r_px,v-r_px,u+r_px,v+r_px),
                         fill=(240,200,0,200), outline=(255,230,0,255), width=lw)
            xo = int(r_px*0.55)
            draw.line((u+xo-5,v-xo+5,u+xo+3,v-xo-3), fill=(255,60,60,255), width=2)
            draw.line((u+xo-5,v-xo-3,u+xo+3,v-xo+5), fill=(255,60,60,255), width=2)
            draw.text((u,v), slot['id'], font=f_id, fill=(40,40,40,240), anchor='mm')
        else:
            draw.ellipse((u-r_px,v-r_px,u+r_px,v+r_px),
                         fill=(0,0,0,0), outline=(255,60,60,220), width=lw)
            d = int(r_px*0.45)
            draw.line((u-d,v-d,u+d,v+d), fill=(255,60,60,200), width=2)
            draw.line((u+d,v-d,u-d,v+d), fill=(255,60,60,200), width=2)
            draw.text((u,v), slot['id'], font=f_sm, fill=(255,100,100,200), anchor='mm')
        if i in pinned:
            draw.ellipse((u-4,v-4,u+4,v+4), fill=(255,220,0,240))
    return np.array(pil.convert("RGB")), r_px

# ==============================================================================
# NETWORK HELPERS
# ==============================================================================
def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

def recv_frame(sock):
    size_data = recvall(sock, 4)
    if size_data is None:
        return None
    size = struct.unpack('!I', size_data)[0]
    data = recvall(sock, size)
    if data is None:
        return None
    return pickle.loads(data)

def send_json(sock, obj):
    data = json.dumps(obj).encode('utf-8')
    sock.sendall(struct.pack('!I', len(data)))
    sock.sendall(data)

# ==============================================================================
# ANALYSIS + GUI POPUP
# ==============================================================================
def analyze_and_show(processor, rgb_pil, depth_img, rack_type=None, auto_detect=None):
    """
    Runs full pipeline. Shows result figure (blocks until closed).
    Returns JSON dict with per-slot status.

    rack_type  - well count (6, 8, 18). Ignored when auto_detect=True.
    auto_detect - True = use Gemini VLM; False = use rack_type directly.
    Falls back to server-level AUTO_DETECT_RACK / RACK_TYPE if not given.
    """
    if auto_detect is None:
        auto_detect = AUTO_DETECT_RACK
    if rack_type is None:
        rack_type = RACK_TYPE

    rgb_np = np.array(rgb_pil)

    # -- Auto-detect rack type via Gemini (if enabled) -------------------------
    if auto_detect:
        rack_type = 18
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                rgb_pil.save(tmp_path, "PNG")
            rack_type = _detect_rack_type_gemini(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        print(f"  [Manual] Using RACK_TYPE from client: {rack_type}")

    # -- Reconfigure globals for this request ----------------------------------
    _configure_for_rack_type(rack_type)

    # -- Run pipeline ----------------------------------------
    rack_mask, bbox = find_rack_mask(processor, rgb_pil)
    if rack_mask is None:
        # Show error popup
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        ax.text(0.5, 0.5, "RACK NOT FOUND\n\nCould not detect rack in image.",
                color='red', fontsize=20, ha='center', va='center', fontweight='bold')
        ax.axis('off')
        plt.close('all')
        return {"success": False, "error": "rack not found"}, None

    rx1, ry1, rx2, ry2 = bbox

    def _slots_inside(px_list):
        inside = sum(1 for p in px_list if p and rx1 <= p[0] <= rx2 and ry1 <= p[1] <= ry2)
        return inside / max(len(px_list), 1) >= 0.5

    def _project_from_rect(mask_binary):
        """
        Fit a minimum-area rectangle to all nonzero mask pixels via their
        convex hull.  Works even when the SAM3 mask has large holes - the
        outer boundary of the detected plastic frame still spans the rack.
        """
        ys_m, xs_m = np.where(mask_binary > 0)
        if len(xs_m) < 5:
            return None
        all_pts = np.column_stack([xs_m, ys_m]).astype(np.float32)
        hull = cv2.convexHull(all_pts)
        rect = cv2.minAreaRect(hull)
        center, (w, h), angle = rect
        # Ensure the long side maps to RACK_W_CM (rack is wider than tall)
        if w < h:
            w, h = h, w
            angle += 90.0
        if w < 20 or h < 5:   # sanity: mask must be non-trivial
            return None
        px_per_cm_x = w / RACK_W_CM
        px_per_cm_y = h / RACK_H_CM
        cx, cy = center
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        result = []
        for s in SLOTS:
            dx_cm = s['x_cm'] - RACK_W_CM / 2.0
            dy_cm = s['y_cm'] - RACK_H_CM / 2.0
            dx_px = dx_cm * px_per_cm_x
            dy_px = dy_cm * px_per_cm_y
            result.append((cx + dx_px * cos_a - dy_px * sin_a,
                           cy + dx_px * sin_a + dy_px * cos_a))
        return result

    # Always run depth fitting for pts3d (used for z-distance display)
    origin, x_ax, y_ax, pts3d, plane_px = fit_plane_and_project(rack_mask, depth_img)

    # Primary: fit min-area rectangle to mask - robust against sparse/holey SAM3 masks
    rect_px = _project_from_rect(rack_mask)
    if rect_px and _slots_inside(rect_px):
        init_px = rect_px
    elif origin is not None and _slots_inside(plane_px):
        init_px = plane_px
    else:
        # Last resort: proportional bbox (assumes bbox ≈ rack size)
        cW, cH = rx2-rx1, ry2-ry1
        init_px = [(rx1 + s['x_cm']/RACK_W_CM*cW,
                    ry1 + s['y_cm']/RACK_H_CM*cH) for s in SLOTS]

    if RACK_TYPE == 18:
        pa = init_px[0]; pb = init_px[len(_ROW_Y_CM_18)]
        if pa and pb:
            d_col = np.sqrt((pb[0]-pa[0])**2 + (pb[1]-pa[1])**2)
            approx_r = d_col / (_COL_X_CM_18[1] - _COL_X_CM_18[0]) * WELL_R_CM
        else:
            approx_r = 20.0
    else:
        pa, pb = init_px[0], init_px[-1]
        if pa and pb:
            dx_cm = abs(SLOTS[-1]['x_cm'] - SLOTS[0]['x_cm'])
            d_span = np.sqrt((pb[0]-pa[0])**2 + (pb[1]-pa[1])**2)
            approx_r = d_span / dx_cm * WELL_R_CM if dx_cm > 0 else 20.0
        else:
            approx_r = 20.0

    hough_wells = detect_wells_hough(rgb_np, bbox, approx_r)

    if RACK_TYPE == 18:
        n_s, _, boxes_s = sam3_find(processor, rgb_pil.crop((rx1, ry1, rx2, ry2)), "circle holes")
        sam3_triples = []
        if n_s > 0:
            for box in boxes_s:
                x1, y1, x2, y2 = box
                cx = (x1+x2)/2.0+rx1; cy = (y1+y2)/2.0+ry1
                r  = ((x2-x1)+(y2-y1))/4.0
                sam3_triples.append((float(cx), float(cy), float(r)))
        sam3_raw = [(t[0], t[1]) for t in sam3_triples]
        sam3_prompt = "circle holes"
    else:
        sam3_raw, sam3_triples, sam3_prompt = detect_wells_sam3(processor, rgb_pil, bbox, approx_r)

    combined = get_combined_wells(hough_wells, sam3_triples, approx_r)
    M, matches, refined_px = refine_with_circles(combined, init_px)

    if RACK_TYPE == 18:
        if not _slots_inside(refined_px):
            refined_px = init_px
        final_px = refined_px
        sam3_matched = {}
    else:
        if not _slots_inside(refined_px):
            refined_px = init_px
        final_px, sam3_matched = override_with_sam3_holes(sam3_raw, refined_px)

    n_vials, vial_masks, vial_centres_full, crop_size = find_vials(processor, rgb_pil, bbox)
    occupancy, vi_to_slot, px_per_cm = assign_vials_to_slots(vial_centres_full, final_px)
    n_caps, cap_masks, cap_centres_full, _ = find_caps(processor, rgb_pil, bbox)
    occupancy, cap_to_vi = assign_caps_to_vials(
        cap_centres_full, vial_centres_full, vi_to_slot, occupancy, px_per_cm)

    # -- Build JSON result ----------------------------------------
    n_capped   = sum(1 for s in VIAL_SLOTS if occupancy.get(s['id']) == VIAL_CAPPED)
    n_uncapped = sum(1 for s in VIAL_SLOTS if occupancy.get(s['id']) == VIAL_UNCAPPED)
    n_empty    = sum(1 for s in VIAL_SLOTS if occupancy.get(s['id']) == EMPTY)
    n_occupied = n_capped + n_uncapped

    slots_result = []
    for s in SLOTS:
        sid = s['id']
        state = occupancy.get(sid, EMPTY)
        pt = final_px[SLOTS.index(s)]

        # Auto-convert corner-based definition -> rack-center-relative [meters]
        # dx_m = offset along rack long axis (x_cm, columns a->e)
        # dy_m = offset along rack short axis (y_cm, rows)
        # With RACK_YAW_DEG=90° the long axis lies along robot-Y and the
        # short axis lies along robot-X, so we send them swapped so that
        # the client's build_T_base_rack rotation produces correct robot-frame coords.
        dx_m = (s['x_cm'] - RACK_W_CM / 2.0) / 100.0
        dy_m = (s['y_cm'] - RACK_H_CM / 2.0) / 100.0
        dz_m = 0.0 if s['is_gripper'] else Z_VIAL_TOP_M

        slots_result.append({
            "id": sid,
            "state": state,
            "is_gripper": s['is_gripper'],
            "pixel_x": round(pt[0], 2) if pt else None,
            "pixel_y": round(pt[1], 2) if pt else None,
            "x_m": round(dy_m, 4),   # short-axis offset -> robot X (via 90° rotation)
            "y_m": round(dx_m, 4),   # long-axis offset  -> robot Y (via 90° rotation)
            "z_m": round(dz_m, 4),
        })

    result = {
        "rack_type": RACK_TYPE,
        "success": True,
        "summary": {
            "total_slots": len(SLOTS),
            "vial_slots": N_VIAL_SLOTS,
            "gripper_slots": N_GRIPPER,
            "occupied": n_occupied,
            "capped": n_capped,
            "uncapped": n_uncapped,
            "empty": n_empty,
        },
        "slots": slots_result,
        "metadata": {
            "n_vials_detected": n_vials,
            "n_caps_detected": n_caps,
            "n_caps_matched": len(cap_to_vi),
            "sam3_pinned": len(sam3_matched) if RACK_TYPE != 18 else 0,
            "rack_source": "gemini" if AUTO_DETECT_RACK else "manual",
        }
    }

    # -- Build and show figure ----------------------------------------
    conts, _ = cv2.findContours(rack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    p_rgb = rgb_np.copy()
    cv2.drawContours(p_rgb, conts, -1, (0,255,255), 2)
    p_mask = np.full_like(rgb_np, 255)
    p_mask[rack_mask > 0] = rgb_np[rack_mask > 0]
    p_vials = draw_vials_panel(rgb_np, rack_mask, bbox,
                               vial_masks, vial_centres_full, vi_to_slot, crop_size)
    p_tmpl, r_tmpl = draw_template_overlay(rgb_np, rack_mask, final_px, large=True, mask_bg=True)
    p_caps = draw_caps_panel(rgb_np, rack_mask, cap_masks, cap_centres_full, cap_to_vi)
    p_final, _ = draw_final_summary(rgb_np, rack_mask, final_px, occupancy,
                                    sam3_matched if RACK_TYPE != 18 else None)

    legend_elements = [
        mpatches.Patch(facecolor='#00c850', edgecolor='#00ff64', label=f'Capped   ({n_capped})'),
        mpatches.Patch(facecolor='#f0c800', edgecolor='#ffe600', label=f'Uncapped ({n_uncapped})'),
        mpatches.Patch(facecolor='none',    edgecolor='#ff3c3c', label=f'Empty    ({n_empty})'),
        mpatches.Patch(facecolor='#323232', edgecolor='#969696', label=f'Gripper  ({N_GRIPPER}, excl.)'),
    ]

    z_med = float(np.median(pts3d[:,2])) if pts3d is not None and len(pts3d) > 0 else 0
    mode_label = "gemini auto-detect" if AUTO_DETECT_RACK else f"manual RACK_TYPE={RACK_TYPE}"

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor('#0d0d0d')
    for ax in axes.flat:
        ax.axis('off'); ax.set_facecolor('#0d0d0d')

    axes[0,0].imshow(p_rgb)
    axes[0,0].set_title("(a) Original RGB\ncyan = rack outline", color='white', fontsize=9, fontweight='bold')
    axes[0,1].imshow(p_mask)
    axes[0,1].set_title(f"(b) SAM3 rack mask\n{(rack_mask>0).sum()} px", color='cyan', fontsize=9, fontweight='bold')
    axes[0,2].imshow(p_vials)
    axes[0,2].set_title(f"(c) SAM3 vials  ({n_vials})\ngreen=matched  red=unmatched", color='#ffaa44', fontsize=9, fontweight='bold')
    axes[1,0].imshow(p_tmpl)
    axes[1,0].set_title(f"(e) Hough+SAM3 affine template\nwell ⌀={WELL_R_CM*2:.1f}cm  matched {len(matches)}/{len(SLOTS)}  (=gripper)", color='yellow', fontsize=9, fontweight='bold')
    axes[1,1].imshow(p_caps)
    axes[1,1].set_title(f"(g) SAM3 caps detected  ({n_caps})\ncyan=matched  orange=unmatched", color='#44ddff', fontsize=9, fontweight='bold')
    axes[1,2].imshow(p_final)
    axes[1,2].legend(handles=legend_elements, loc='lower center', ncol=2, framealpha=0.25, fontsize=8,
                     labelcolor='white', facecolor='#1a1a1a', edgecolor='#444')
    axes[1,2].set_title(f"(h) FINAL RESULT  ({N_VIAL_SLOTS} countable slots)\ncapped {n_capped}  |  uncapped {n_uncapped}  |  empty {n_empty}", color='lime', fontsize=10, fontweight='bold')

    fig.suptitle(
        f"{RACK_TYPE}-well rack  [{mode_label}]  |  ~{z_med*100:.0f}cm  |  "
        f"vials {n_occupied}/{N_VIAL_SLOTS}  (capped {n_capped} · uncapped {n_uncapped})  |  "
        f"empty {n_empty}  |  SAM3-pinned {len(sam3_matched)}/{len(SLOTS)}",
        color='white', fontsize=10, fontweight='bold', y=1.005)

    plt.tight_layout(pad=0.5)

    # Save figure to bytes before showing (show blocks until user closes)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img_bytes = buf.read()

    plt.close('all')

    return result, img_bytes

# ==============================================================================
# SERVER MAIN
# ==============================================================================
def handle_client(conn, addr, processor):
    print(f"\n[Client connected] {addr}")
    try:
        # -- 1. Config dict from client ----------------------------------------
        cfg = recv_frame(conn)
        if cfg is None or not isinstance(cfg, dict):
            send_json(conn, {"success": False, "error": "failed to receive config"})
            return
        client_rack_type  = cfg.get("rack_type",   RACK_TYPE)
        client_auto_detect = cfg.get("auto_detect", AUTO_DETECT_RACK)
        print(f"  Config: rack_type={client_rack_type}  auto_detect={client_auto_detect}")

        # -- 2. RGB frame ----------------------------------------
        rgb_data = recv_frame(conn)
        if rgb_data is None:
            print("  Failed to receive RGB")
            send_json(conn, {"success": False, "error": "failed to receive RGB"})
            return
        rgb_np = rgb_data
        print(f"  Received RGB: {rgb_np.shape}")

        # -- 3. Depth frame ----------------------------------------
        depth_data = recv_frame(conn)
        if depth_data is None:
            print("  Failed to receive depth")
            send_json(conn, {"success": False, "error": "failed to receive depth"})
            return
        depth_img = depth_data
        print(f"  Received Depth: {depth_img.shape}")

        rgb_pil = Image.fromarray(rgb_np).convert("RGB")

        print("  Running analysis...")
        result, img_bytes = analyze_and_show(processor, rgb_pil, depth_img,
                                             rack_type=client_rack_type,
                                             auto_detect=client_auto_detect)

        if result.get('success'):
            print(f"  Result summary: {result['summary']}")
        else:
            print(f"  Analysis failed: {result.get('error')}")

        send_json(conn, result)
        print(f"  Sent JSON result to client")

        # Send figure image bytes (None -> send empty bytes so client always recvs)
        img_payload = img_bytes if img_bytes is not None else b""
        data = pickle.dumps(img_payload, protocol=pickle.HIGHEST_PROTOCOL)
        conn.sendall(struct.pack('!I', len(data)))
        conn.sendall(data)
        print(f"  Sent figure image ({len(img_payload)//1024} KB) to client")

    except Exception as e:
        print(f"  Error handling client: {e}")
        import traceback
        traceback.print_exc()
        try:
            send_json(conn, {"success": False, "error": str(e)})
        except:
            pass
    finally:
        conn.close()
        print(f"[Client disconnected] {addr}")


def main():
    print("=" * 60)
    print("  RACK ANALYSIS SERVER")
    if AUTO_DETECT_RACK:
        print(f"  Rack type: auto-detect via Gemini")
    else:
        print(f"  Rack type: MANUAL = {RACK_TYPE}-well  "
              f"({N_VIAL_SLOTS} vial slots + {N_GRIPPER} gripper slots)")
        print(f"  (Set AUTO_DETECT_RACK = True to switch to Gemini detection)")
    print(f"  Listening on {HOST}:{PORT}")
    print("=" * 60)

    print("\nLoading SAM3 model (this may take a moment)...")
    processor = Sam3Processor(build_sam3_image_model())
    print("SAM3 ready.")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)

    print(f"\nServer running. Waiting for clients...\n")

    try:
        while True:
            conn, addr = server.accept()
            handle_client(conn, addr, processor)
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
