#!/usr/bin/env python3
import json
from pathlib import Path
import os
import re
from tkinter import TRUE
import numpy as np
from scipy.special import erf
from PIL import Image
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('obj_dir', type=str)
args = parser.parse_args()

# --- CONFIG ---

OBJ_DIR = Path(args.obj_dir)
SHADOW_ART_FOLDER = 'shadow_art'
LINE_DRAWING_FOLDER = 'line_drawing'
PROCESSED_FOLDER = 'line_drawing_processed'
PROMPT_FILE = 'prompt.txt'
CLIP_SCORES_FILE = 'clip_scores.json'
IMAGE_REWARD_SCORES_FILE = 'imagereward_scores.json'
INCLUDE_STROKE_DESCRIPTIONS = True
CREATE_PROMPT_FILE = True
SHOW_K = 4  # default number of entries shown

ADD_HPS_SCORES = True
HPS_SCORES_FILE = 'hps_scores.json'

OUTPUT_HTML = OBJ_DIR / 'index.html'
OPTIMIZE_SCENE_PARAMS = False
OPTIMIZE_SCENE_PARAMS_FOLDER = 'scene_params_optimized'

ADD_VQA_QUESTIONS = True
VQA_SCORES_STROKE_FILE = 'vqa_score_stroke_composition.json'

ADD_MANUAL_LABELS = False
MANUAL_LABELS_FILE = 'labels.txt'

ADD_SHADOW_ART_SCORES = False

def fmt_score(val, decimal_places=3):
    # show 4 decimal places if val is between 0 and 1, otherwise show 2 decimal places
    return f"{val:.{decimal_places}f}"

lines = [
    "<!DOCTYPE html>",
    "<html><head><meta charset='utf-8'><title>Shadow Art Viewer</title>",
    "<style>",
    "  body { font-family: sans-serif; padding: 20px; }",
    "  .item { margin-bottom: 40px; }",
    "  .category-header { margin-bottom: 5px; font-weight: bold; }",
    "  .num-input { width: 60px; margin: 0 5px; }",
    "  .apply-button {",
    "    display: inline-block;",
    "    padding: 6px 12px;",
    "    background-color: #007BFF;",
    "    color: #fff;",
    "    border: 1px solid #0056b3;",
    "    border-radius: 2px;",
    "    cursor: pointer;",
    "    font-size: 0.7em;",
    "    margin-left: 5px;",
    "  }",
    "  .apply-button:hover { background-color: #0056b3; }",
    "  .shadow-item { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 20px; }",
    "  .shadow-item img { max-width: 150px; border: 1px solid #ddd; padding: 4px; }",
    "  .score-text { font-size: 0.85em; margin-top: 4px; text-align: center; }",
    "  .prompt-text { font-size: 0.95em; line-height: 1.4; max-width: 300px; }",
    "</style>",
    "</head><body>",
    "<h1>Shadow Art Gallery</h1>",
]

# (re)build prompt file if needed
prompt_path = OBJ_DIR / PROMPT_FILE
if CREATE_PROMPT_FILE:
    with open(prompt_path, 'w') as f:
        if not (OBJ_DIR / 'prompt').is_dir():
            print(f"❌ Missing prompt directory for {OBJ_DIR.name}")
            exit()
        for file in os.listdir(OBJ_DIR / 'prompt'):
            plines = open(OBJ_DIR / 'prompt' / file).readlines()
            content = plines[0].strip().split('\t')[-1] + '\t' + plines[-1].strip()
            f.write(file.replace('.txt','.png') + '\t' + content + '\n')

shadow_dir = OBJ_DIR / SHADOW_ART_FOLDER
line_dir   = OBJ_DIR / LINE_DRAWING_FOLDER
proc_dir   = OBJ_DIR / PROCESSED_FOLDER
scene_params_dir = OBJ_DIR / OPTIMIZE_SCENE_PARAMS_FOLDER
clip_path  = OBJ_DIR / CLIP_SCORES_FILE
image_reward_path = OBJ_DIR / IMAGE_REWARD_SCORES_FILE
hps_path = OBJ_DIR / HPS_SCORES_FILE
vqa_stroke_path = OBJ_DIR / VQA_SCORES_STROKE_FILE
manual_labels_path = OBJ_DIR / MANUAL_LABELS_FILE
if not (shadow_dir.is_dir() and line_dir.is_dir() and proc_dir.is_dir()
        and clip_path.is_file()
        and image_reward_path.is_file()
        and (hps_path.is_file() or not ADD_HPS_SCORES)
        and (vqa_stroke_path.is_file() or not ADD_VQA_QUESTIONS)
        and (manual_labels_path.is_file() or not ADD_MANUAL_LABELS)
        ):
    # find which is missing
    missing = []
    if not shadow_dir.is_dir(): missing.append('shadow art')
    if not line_dir.is_dir(): missing.append('line')
    if not proc_dir.is_dir(): missing.append('processed')
    if not clip_path.is_file(): missing.append('clip')
    if not image_reward_path.is_file(): missing.append('image_reward')
    if not hps_path.is_file() and ADD_HPS_SCORES: missing.append('hps scores')
    if not scene_params_dir.is_dir() and OPTIMIZE_SCENE_PARAMS: missing.append('scene params')
    if not vqa_stroke_path.is_file() and ADD_VQA_QUESTIONS: missing.append('vqa score stroke')
    if not manual_labels_path.is_file() and ADD_MANUAL_LABELS: missing.append('manual labels')
    print(f"❌ Missing files: {', '.join(missing)}")
    

clip_scores = json.loads(clip_path.read_text(encoding='utf-8'))
image_reward_scores = json.loads(image_reward_path.read_text(encoding='utf-8'))
hps_scores = json.loads(hps_path.read_text(encoding='utf-8')) if ADD_HPS_SCORES else None
vqa_scores_stroke = json.loads(vqa_stroke_path.read_text(encoding='utf-8')) if ADD_VQA_QUESTIONS else None
manual_labels = {}
if ADD_MANUAL_LABELS:
    with open(manual_labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            manual_labels[parts[0]] = parts[1]
entries = []
ld_list = []

for row in prompt_path.open(encoding='utf-8'):
    parts = row.strip().split('\t')
    fn, prompt = parts[0], '\n'.join(parts[1:])
    s, l, p = shadow_dir/fn, line_dir/fn, proc_dir/fn
    if not (l.exists() and p.exists()): 
        print(f"❌ Missing files for {OBJ_DIR.name} — {fn}")
        exit()
    if not s.exists():
        # find all paths that start with fn.split('.')[0]
        s_paths = [ss for ss in shadow_dir.iterdir() if ss.name.startswith(fn[:-4])]
        if len(s_paths) == 0:
            print(f"❌ Missing shadow for {OBJ_DIR.name} — {fn}")
            exit()
        s_paths.sort(key=lambda x: int(x.name[:-4].split('_')[-1]))
        s = s_paths[0]

    ld = clip_scores.get(fn,{}).get('line_drawing_score')
    pld = clip_scores.get(fn,{}).get('line_drawing_processed_score')
    ldIR = image_reward_scores.get(fn,{}).get('line_drawing_score')
    pldIR = image_reward_scores.get(fn,{}).get('line_drawing_processed_score')
    if ADD_SHADOW_ART_SCORES:
        sa_clip = clip_scores.get(fn,{}).get('shadow_art_score')
        sa_ir = image_reward_scores.get(fn,{}).get('shadow_art_score')
        sa_hps = hps_scores.get(fn,{}).get('shadow_art_score')
    if ADD_HPS_SCORES:
        ldhps = hps_scores.get(fn,{}).get('line_drawing_score')
        pldhps = hps_scores.get(fn,{}).get('line_drawing_processed_score')
    if ADD_VQA_QUESTIONS:
        vqa_score_stroke = vqa_scores_stroke.get(fn)['yes']
    if ADD_MANUAL_LABELS:
        manual_label = manual_labels.get(fn)

    if ld is None or pld is None or ldIR is None or pldIR is None: 
        print(f"❌ Missing scores for {OBJ_DIR.name} — {fn}")
        exit()

    clip_good = ld and pld and ld>pld
    ir_good = ldIR and pldIR and ldIR>pldIR
    ld_list.extend([ld, pld])
    azi = float(fn.split('azi')[1][:6])
    ele = float(fn.split('ele')[1][:6])

    entry = {
        'fn': fn, 'prompt': prompt, 'azi': azi, 'ele': ele,
        'rel_s': s.relative_to(OBJ_DIR), 'rel_l': l.relative_to(OBJ_DIR), 'rel_p': p.relative_to(OBJ_DIR),
        'ld': ld, 'pld': pld, 
        'ldIR': ldIR, 'pldIR': pldIR,
        'clip_good': clip_good,
        'clip_color': 'green' if clip_good else 'red',
        'ir_color': 'green' if ir_good else 'red',
        'ir_good': ir_good,
        'vqa_score_stroke': vqa_score_stroke if ADD_VQA_QUESTIONS else None,
        'manual_label': manual_label if ADD_MANUAL_LABELS else None,
    }

    if ADD_SHADOW_ART_SCORES:
        entry['sa_clip'] = sa_clip
        entry['sa_ir'] = sa_ir
        entry['sa_hps'] = sa_hps

    if ADD_HPS_SCORES:  
        entry['ldhps'] = ldhps
        entry['pldhps'] = pldhps
        hps_good = ldhps and pldhps and ldhps>pldhps
        entry['hps_color'] = 'green' if hps_good else 'red'
        entry['hps_good'] = hps_good
    else:
        entry['hps_good'] = True

    if OPTIMIZE_SCENE_PARAMS:
        scene_params_path = scene_params_dir / fn.replace('.png', '.json')
        if scene_params_path.exists():
            scene_params = json.loads(scene_params_path.read_text(encoding='utf-8'))
            light_azi = float(scene_params['light_azimuth'])
            light_ele = float(scene_params['light_elevation'])
            entry['light_azi'] = light_azi
            entry['light_ele'] = light_ele

    entries.append(entry)

# normalize scores
ld_mean = np.mean(ld_list)
ld_std = np.std(ld_list)
for e in entries:
    def cdf(x):
        return 1/2 * (1 + erf(x/np.sqrt(2)))
    e['clip_score'] = e['ld'] ** 2 / e['pld'] ** 2
    e['ir_score'] = cdf(e['ldIR']) ** 2 - cdf(e['pldIR']) ** 2
    if ADD_HPS_SCORES:
        e['hps_score'] = e['ldhps'] ** 2 - e['pldhps'] ** 2
    else:
        e['hps_score'] = 1
    if ADD_VQA_QUESTIONS:
        e['vqa_color_stroke'] = 'green' if e['vqa_score_stroke'] > 0.5 else 'red'
    if ADD_MANUAL_LABELS:
        e['manual_label_color'] = 'purple' if e['manual_label'] == 'extraordinary' \
            else 'green' if e['manual_label'] == 'good' \
            else 'orange' if e['manual_label'] == 'normal' \
            else 'red' if e['manual_label'] == 'bad' \
            else '#800000' # use hex color for dark red

entries.sort(key=lambda e: (
                            ADD_VQA_QUESTIONS and (e['vqa_score_stroke'] < 0.5),
                            not e['hps_good'] or not e['ir_good'],
                            -e['clip_score'] * e['ir_score'] * e['hps_score']
                            ))

raw_id = f"{OBJ_DIR.name}"
item_id = re.sub(r'[^0-9A-Za-z_-]+','-', raw_id)

# Item container
lines.append(f"<div class='item' id='{item_id}'>")
lines.append("  <div class='category-header'>")
lines.append(f"    {OBJ_DIR.name}")
lines.append("  </div>")
lines.append("  <div style='margin:5px 0;'>")
lines.append(f"    Show <input type='number' min='0' max='{len(entries)}' value='{SHOW_K}'"
                f" class='num-input' data-target='{item_id}'>")
lines.append(f"    <button class='apply-button' data-target='{item_id}'>Apply</button>")
lines.append(f" of {len(entries)} entries")
lines.append("  </div>")

# Entries
for i, e in enumerate(entries):
    hidden = i >= SHOW_K
    style = " style='display:none;'" if hidden else ""
    lines.append(f"  <div class='shadow-item'{style}>")
    lines.append(f"    <div>")
    lines.append(f"      <img src='{e['rel_s'].as_posix()}' alt='{e['fn']}'>")
    lines.append(f"      <div class='score-text'> azi: {e['azi']:06.2f}, ele: {e['ele']:06.2f}</div>")
    if 'rot' in e['fn']:
        rots = [float(x[:6]) for x in e['fn'].split('rot')[1:]]
        lines.append(f"      <div class='score-text'> rot: {', '.join([f'{r:06.2f}' for r in rots])}</div>")
    if OPTIMIZE_SCENE_PARAMS:
        lines.append(f"      <div class='score-text'> azi: {e['light_azi']:06.2f}, ele: {e['light_ele']:06.2f}</div>")
    if ADD_MANUAL_LABELS:
        lines.append(f"      <div class='score-text' style='color:{e['manual_label_color']}'>{e['manual_label']}</div>")
    lines.append("    </div>")
    lines.append(f"   <div>")
    lines.append(f"      <img src='{e['rel_l'].as_posix()}' alt='{e['fn']}'>")
    lines.append(f"      <div class='score-text' style='color:{e['clip_color']}'>CLIP: {fmt_score(e['ld'])}</div>")
    lines.append(f"      <div class='score-text' style='color:{e['ir_color']}'>Image Reward: {fmt_score(e['ldIR'])}</div>")
    if ADD_HPS_SCORES:
        lines.append(f"      <div class='score-text' style='color:{e['hps_color']}'>HPS: {fmt_score(e['ldhps'])}</div>")
    if ADD_VQA_QUESTIONS:
        lines.append(f"      <div class='score-text' style='color:{e['vqa_color_stroke']}'>VQA stroke: {fmt_score(e['vqa_score_stroke'])}</div>")
    lines.append("    </div>")
    lines.append("    <div>")
    lines.append(f"      <img src='{e['rel_p'].as_posix()}' alt='{e['fn']}'>")
    lines.append(f"      <div class='score-text' style='color:{e['clip_color']}'>CLIP: {fmt_score(e['pld'])}</div>")
    lines.append(f"      <div class='score-text' style='color:{e['ir_color']}'>Image Reward: {fmt_score(e['pldIR'])}</div>")
    if ADD_HPS_SCORES:
        lines.append(f"      <div class='score-text' style='color:{e['hps_color']}'>HPS: {fmt_score(e['pldhps'])}</div>")
    lines.append("    </div>")
    stroke_desc, line_drawing_desc = e['prompt'].split('\n')
    lines.append(f"    <div class='prompt-text'>{line_drawing_desc.strip()}</div>")
    if INCLUDE_STROKE_DESCRIPTIONS:
        lines.append(f"    <div class='prompt-text'>{stroke_desc.strip()}</div>")
    lines.append("  </div>")
lines.append("</div>")

# Interactive script
lines.append("<script>")
lines.append("document.addEventListener('DOMContentLoaded', () => {")
lines.append("  document.querySelectorAll('.apply-button').forEach(btn => {")
lines.append("    btn.addEventListener('click', () => {")
lines.append("      const targetId = btn.dataset.target;")
lines.append("      const container = document.getElementById(targetId);")
lines.append("      const input = container.querySelector('.num-input[data-target=\"' + targetId + '\"]');")
lines.append("      const k = parseInt(input.value) || 0;")
lines.append("      const items = container.querySelectorAll('.shadow-item');")
lines.append("      items.forEach((item, idx) => {")
lines.append("        item.style.display = idx < k ? 'flex' : 'none';")
lines.append("      });")
lines.append("    });")
lines.append("  });")
lines.append("  const globalBtn = document.getElementById('apply-all-button');")
lines.append("  const globalInput = document.getElementById('global-num-input');")
lines.append("  globalBtn.addEventListener('click', () => {")
lines.append("    const kAll = parseInt(globalInput.value) || 0;")
lines.append("    document.querySelectorAll('.item').forEach(container => {")
lines.append("      // update the per-item input")
lines.append("      const input = container.querySelector('.num-input');")
lines.append("      input.value = kAll;")
lines.append("      // apply to all its shadow-items")
lines.append("      container.querySelectorAll('.shadow-item').forEach((item, idx) => {")
lines.append("        item.style.display = idx < kAll ? 'flex' : 'none';")
lines.append("      });")
lines.append("    });")
lines.append("  });")
lines.append("});")
lines.append("</script>")

lines.append("</body></html>")

# Write out the HTML
OUTPUT_HTML.write_text("\n".join(lines), encoding='utf-8')
print(f"✔️ Generated {OUTPUT_HTML}")
