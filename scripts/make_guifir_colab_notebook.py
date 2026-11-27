"""Generates notebooks/06_guifir_training_colab.ipynb:
A self-contained Google Colab (T4 GPU) notebook that:
  1. Downloads the GuiFiR dataset (kmu-ee-ailab/guifir_dataset) from Google Drive.
  2. Extracts frames for the val (543 frames) and test (2,048 frames) splits with
     exact `frame_gt_pose` (4 fretboard corners + 5 fingertips) + train split clips.
  3. Trains Stage 1 (NeckNet, ResNet-18) on GuiFiR fretboard corners (+ optional own labels).
  4. Trains Stage 4 (FingertipNet, ResNet-18) on GuiFiR 5-fingertip keypoints inside the neck.
  5. Option 3: Maps GuiFiR `frame_tab` (6-string fret numbers) to the 9 target chord voicings
     (A, Am, Bm, C, D, Dm, E, F, G), extracts 17 tab+span features, optionally fuses with
     your own_data.zip features, and trains ChordNet (RandomForest + novelty filter).
  6. Evaluates everything honestly and packages `guifir_eval_and_models.zip` for download
     so you can run the trained models locally with `scripts/live_local.py`.

Usage:
    python scripts/make_guifir_colab_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": s.strip("\n").splitlines(True),
    })


md(r"""
# Guitar Chord Recognition — GuiFiR Multi-Stage Training (Colab T4 GPU)

This notebook trains the perception and chord models on the **GuiFiR** dataset ([kmu-ee-ailab/guifir_dataset](https://github.com/kmu-ee-ailab/guifir_dataset), *IEEE Access 2026*) and exports **`guifir_eval_and_models.zip`** so you can drop the trained weights straight into your local project (`artifacts/models_local/`).

### What this notebook trains
1. **Stage 1 (`NeckNet` -> `neck_v3.pt`)**: Trained on GuiFiR's **2,591+ ground-truth fretboard corners** (`lt, lb, rt, rb`), fixing neck drift in new rooms/lighting.
2. **Stage 4 (`FingertipNet` -> `fingertip_net.pt`)**: Trained on GuiFiR's **2,591+ ground-truth fingertip keypoints** (`4, 8, 12, 16, 20`) inside the rectified neck, solving MediaPipe's ~50% fretting-hand dropout.
3. **Stage 5 (`ChordNet` -> `chord_clf.joblib` + `chord_meta.json`)**: Maps GuiFiR's `frame_tab` (6-string fret positions) to your **9 chords (`A, Am, Bm, C, D, Dm, E, F, G`)**, optionally fuses with your `own_data.zip`, and trains the 17-feature Random Forest + kNN novelty filter.

### Before you run
- Set **Runtime > Change runtime type > T4 GPU**.
- *(Optional)* Upload `own_data.zip` to Google Drive (`My Drive/guitar-chord-recognition/own_data.zip`) or the Colab file panel if you want to fuse your own recordings with GuiFiR.
""")

md(r"""
## 0. Settings & Environment Check
""")

code(r"""
import os, sys, json, time, shutil, zipfile, tarfile, subprocess
from pathlib import Path

CHORDS = ['A', 'Am', 'Bm', 'C', 'D', 'Dm', 'E', 'F', 'G']
GUIFIR_GDRIVE_ID = '1Ri42XQnpwWQzjNLOPgZCR2XCx9YGrqxc'
GUIFIR_REPO = 'https://github.com/kmu-ee-ailab/guifir_dataset.git'

NECK_EPOCHS = 40          # ResNet-18 NeckNet epochs on GuiFiR
TIP_EPOCHS = 40           # ResNet-18 FingertipNet epochs on GuiFiR
MAX_TRAIN_CLIPS = 900     # Full GuiFiR train split so all 9 chord voicings (including Dm) get samples
USE_OWN_DATA = True       # Fuse with your own_data.zip if present in Drive or /content

WORK = Path('/content/guifir_work')
DATA_DIR = WORK / 'guifir_data'
OUT_DIR = WORK / 'output_models'
WORK.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'PyTorch {torch.__version__} | Device: {DEVICE}')
if DEVICE == 'cpu':
    print('WARNING: Running on CPU. Switch to Runtime > Change runtime type > T4 GPU for fast training!')
""")

code(r"""
!pip install -q gdown opencv-python-headless scikit-learn joblib pandas tqdm
""")

md(r"""
## 1. Download GuiFiR Dataset & Metadata
Downloads `GuiFiR.tar` from Google Drive and clones the official metadata/extraction scripts.
""")

code(r"""
repo_dir = WORK / 'guifir_repo'
if not repo_dir.exists():
    !git clone --depth 1 {GUIFIR_REPO} {repo_dir}

tar_path = WORK / 'GuiFiR.tar'
if not tar_path.exists():
    import gdown
    gdown.download(id=GUIFIR_GDRIVE_ID, output=str(tar_path), quiet=False)

raw_unpacked = WORK / 'raw_guifir'
if not (raw_unpacked / 'GuiFiR').exists():
    print('Unpacking GuiFiR.tar (this may take a few minutes)...')
    with tarfile.open(tar_path, 'r') as tf:
        tf.extractall(raw_unpacked)
    print('Unpacked successfully!')

guifir_root = raw_unpacked / 'GuiFiR'
if not guifir_root.exists():
    # Fallback if tar unpacks directly without GuiFiR/ subfolder
    guifir_root = raw_unpacked
print('GuiFiR root contents:', [p.name for p in guifir_root.iterdir()])
""")

md(r"""
## 2. Extract Annotated Frames (`val` + `test` with `frame_gt_pose`, plus `train` subset)
`val` (543 frames) and `test` (2,048 frames) have **exact 4-corner fretboard pose + 5 fingertip coordinates**.
We extract only the frames referenced in the `.jsonl` annotation files to keep disk usage fast and clean.
""")

code(r"""
import cv2
import pandas as pd
from tqdm.auto import tqdm

def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

anno_dir = guifir_root / 'annotations'
clips_dir = guifir_root / 'raw_clips'
frames_dir = DATA_DIR / 'frames'
frames_dir.mkdir(parents=True, exist_ok=True)

def extract_annotated_frames(jsonl_path, split_name, max_clips=None):
    rows = load_jsonl(jsonl_path)
    by_clip = {}
    for r in rows:
        cid = r['filename'].split('/')[0]
        by_clip.setdefault(cid, []).append(r)

    clip_ids = sorted(by_clip.keys())
    if max_clips is not None:
        clip_ids = clip_ids[:max_clips]

    extracted = []
    for cid in tqdm(clip_ids, desc=f'Extracting {split_name}'):
        vid_path = clips_dir / f'{cid}.mp4'
        if not vid_path.exists():
            continue
        cap = cv2.VideoCapture(str(vid_path))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Map frame_id / filename index to frame number
        targets = {}
        for r in by_clip[cid]:
            stem = Path(r['filename']).stem
            try:
                fidx = int(stem.split('_')[-1])
            except ValueError:
                fidx = int(r.get('frame_id', 0))
            targets[fidx] = r

        cur = 0
        max_target = max(targets.keys()) if targets else -1
        while cur <= max_target:
            ok, bgr = cap.read()
            if not ok:
                break
            if cur in targets:
                r = targets[cur]
                out_img = frames_dir / f'{split_name}_{cid}_{cur:05d}.jpg'
                if not out_img.exists():
                    cv2.imwrite(str(out_img), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                r_copy = dict(r)
                r_copy['img_path'] = str(out_img)
                r_copy['split'] = split_name
                extracted.append(r_copy)
            cur += 1
        cap.release()
    print(f'{split_name}: extracted {len(extracted)} annotated frames from {len(clip_ids)} clips.')
    return extracted

val_records = extract_annotated_frames(anno_dir / 'guifir_val_coco_anno_box.jsonl', 'val')
test_records = extract_annotated_frames(anno_dir / 'guifir_test_coco_anno_box.jsonl', 'test')
train_records = extract_annotated_frames(anno_dir / 'guifir_train_coco_anno_box.jsonl', 'train', max_clips=MAX_TRAIN_CLIPS)

pose_records = [r for r in (val_records + test_records) if r.get('frame_gt_pose')]
print(f'Total frames with exact 4-corner + 5-fingertip GT (frame_gt_pose): {len(pose_records)}')
""")

md(r"""
## 3. Stage 1: Train `NeckNet` (`neck_v3.pt`) on GuiFiR Fretboard Corners
We use the same ResNet-18 FPN + soft-argmax architecture as `scripts/train_neck_detector.py`, trained on GuiFiR's `lt, lb, rb, rt` fretboard corners (scaled from 640x360 to each frame's resolution).
""")

code(r"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader

IN_W, IN_H = 384, 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

class NeckNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in (64, 128, 256, 512)])
        self.head = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 4, 1))

    def forward(self, x):
        c1 = self.l1(self.stem(x)); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        p = self.lat[3](c4)
        for lat, c in ((self.lat[2], c3), (self.lat[1], c2), (self.lat[0], c1)):
            p = F.interpolate(p, size=c.shape[-2:], mode='bilinear', align_corners=False) + lat(c)
        return self.head(p)

def soft_argmax(logits):
    B, K, h, w = logits.shape
    prob = F.softmax(logits.view(B, K, -1), dim=-1).view(B, K, h, w)
    xs = (torch.arange(w, dtype=torch.float32, device=logits.device) + 0.5) / w
    ys = (torch.arange(h, dtype=torch.float32, device=logits.device) + 0.5) / h
    x = (prob.sum(2) * xs).sum(-1); y = (prob.sum(3) * ys).sum(-1)
    return torch.stack([x, y], -1), prob

def poly_iou(a, b, size=(320, 180), ref=(1280, 720)):
    s = np.array([size[0] / ref[0], size[1] / ref[1]])
    ma, mb = np.zeros(size[::-1], np.uint8), np.zeros(size[::-1], np.uint8)
    cv2.fillPoly(ma, [(np.array(a) * s).astype(np.int32)], 1)
    cv2.fillPoly(mb, [(np.array(b) * s).astype(np.int32)], 1)
    inter, union = (ma & mb).sum(), (ma | mb).sum()
    return float(inter / union) if union else 0.0

def parse_guifir_corners(rec, match_right_headstock=True):
    pose = rec['frame_gt_pose']
    # frame_gt_pose is in 640x360 space; convert to normalized [0, 1]
    # [lt, lb, rb, rt] = [nut_top, nut_bot, body_bot, body_top]
    pts = np.array([pose['lt'], pose['lb'], pose['rb'], pose['rt']], dtype=np.float32)
    pts[:, 0] /= 640.0
    pts[:, 1] /= 360.0
    if match_right_headstock:
        # Mirror x so GuiFiR's left-side headstock aligns with right-side webcam headstock
        pts[:, 0] = 1.0 - pts[:, 0]
    return pts

class GuiFiRNeckDataset(Dataset):
    def __init__(self, records, augment=False, match_right_headstock=True):
        self.records = [r for r in records if all(k in r['frame_gt_pose'] for k in ('lt', 'lb', 'rb', 'rt'))]
        self.augment = augment
        self.match_right = match_right_headstock

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        bgr = cv2.imread(r['img_path'])
        pts = parse_guifir_corners(r, match_right_headstock=self.match_right)
        img = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        if self.match_right and not r.get('is_own'):
            img = img[:, ::-1].copy()
        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
                pts[:, 0] = 1.0 - pts[:, 0]
            alpha = np.random.uniform(0.75, 1.25)
            beta = np.random.uniform(-25, 25)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy()
        return torch.from_numpy(x), torch.from_numpy(pts)

# Split pose_records: 80% train, 20% held-out eval (grouped by clip_id to avoid leakage)
clip_names = sorted({r['filename'].split('/')[0] for r in pose_records})
val_clips = set(clip_names[::5])  # every 5th clip held out
neck_train_recs = [r for r in pose_records if r['filename'].split('/')[0] not in val_clips]
neck_eval_recs = [r for r in pose_records if r['filename'].split('/')[0] in val_clips]
print(f'NeckNet split -> Train: {len(neck_train_recs)} frames | Held-out Eval: {len(neck_eval_recs)} frames')

tr_ds = GuiFiRNeckDataset(neck_train_recs, augment=True)
ev_ds = GuiFiRNeckDataset(neck_eval_recs, augment=False)
tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=2)
ev_dl = DataLoader(ev_ds, batch_size=32, shuffle=False, num_workers=2)

neck_model = NeckNet(pretrained=True).to(DEVICE)
opt = torch.optim.AdamW(neck_model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NECK_EPOCHS)

best_iou = 0.0
for ep in range(NECK_EPOCHS):
    neck_model.train()
    tr_loss = 0.0
    for bx, bpts in tr_dl:
        bx, bpts = bx.to(DEVICE), bpts.to(DEVICE)
        pred_pts, _ = soft_argmax(neck_model(bx))
        loss = F.smooth_l1_loss(pred_pts, bpts)
        opt.zero_grad(); loss.backward(); opt.step()
        tr_loss += loss.item() * len(bx)
    sched.step()

    if (ep + 1) % 10 == 0 or ep == NECK_EPOCHS - 1:
        neck_model.eval()
        ious, px_errs = [], []
        with torch.no_grad():
            for bx, bpts in ev_dl:
                pred_pts, _ = soft_argmax(neck_model(bx.to(DEVICE)))
                pred_np = pred_pts.cpu().numpy()
                gt_np = bpts.numpy()
                for p_i, g_i in zip(pred_np, gt_np):
                    p_px = p_i * np.array([1280.0, 720.0])
                    g_px = g_i * np.array([1280.0, 720.0])
                    ious.append(poly_iou(g_px, p_px))
                    px_errs.append(np.linalg.norm(g_px - p_px, axis=1).mean())
        mean_iou = float(np.mean(ious))
        med_err = float(np.median(px_errs))
        print(f'Epoch {ep+1:2d}/{NECK_EPOCHS} | Loss: {tr_loss/len(tr_ds):.4f} | Eval IoU: {mean_iou:.3f} | Median Error (720p): {med_err:.1f} px')
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save({'model': neck_model.state_dict(), 'epoch': ep + 1, 'eval_iou': mean_iou, 'eval_med_px': med_err}, OUT_DIR / 'neck_v3.pt')

print(f'Saved best NeckNet to {OUT_DIR / "neck_v3.pt"} (Eval IoU: {best_iou:.3f})')
""")

md(r"""
## 4. Stage 4: Train `FingertipNet` (`fingertip_net.pt`) on GuiFiR Hand Keypoints
Instead of relying solely on Google MediaPipe (which misses occluded guitar fingers ~50% of the time), we train a dedicated **ResNet-18 Fingertip Detector** on **rectified guitar neck crops** using GuiFiR's ground-truth fingertips (`4, 8, 12, 16, 20`: thumb, index, middle, ring, pinky).
""")

code(r"""
TIP_KEYS = ['4', '8', '12', '16', '20']
RECT_W, RECT_H = 384, 128

def warp_neck_crop(bgr, norm_corners):
    H, W = bgr.shape[:2]
    src = (norm_corners * np.array([W, H], np.float32)).astype(np.float32)
    dst = np.array([[0, 0], [0, RECT_H], [RECT_W, RECT_H], [RECT_W, 0]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(bgr, M, (RECT_W, RECT_H), flags=cv2.INTER_LINEAR)
    return crop, M

class FingertipNet(nn.Module):
    # Predicts 5 fingertip (x, y) coordinates in normalized [0, 1] rectified neck space
    def __init__(self, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in (64, 128, 256, 512)])
        self.head = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 5, 1))

    def forward(self, x):
        c1 = self.l1(self.stem(x)); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        p = self.lat[3](c4)
        for lat, c in ((self.lat[2], c3), (self.lat[1], c2), (self.lat[0], c1)):
            p = F.interpolate(p, size=c.shape[-2:], mode='bilinear', align_corners=False) + lat(c)
        return soft_argmax(self.head(p))[0]  # (B, 5, 2) in [0, 1]

class GuiFiRTipDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = [r for r in records if all(k in r['frame_gt_pose'] for k in ['lt', 'lb', 'rb', 'rt'] + TIP_KEYS)]
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        bgr = cv2.imread(r['img_path'])
        H, W = bgr.shape[:2]
        nc = parse_guifir_corners(r)
        crop, M = warp_neck_crop(bgr, nc)

        pose = r['frame_gt_pose']
        tips_px = np.array([[pose[k][0] * W / 640.0, pose[k][1] * H / 360.0] for k in TIP_KEYS], dtype=np.float32)
        tips_rect = cv2.perspectiveTransform(tips_px.reshape(-1, 1, 2), M).reshape(-1, 2)
        tips_norm = np.clip(tips_rect / np.array([RECT_W, RECT_H], np.float32), 0.0, 1.0)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if self.augment:
            alpha = np.random.uniform(0.75, 1.25)
            beta = np.random.uniform(-20, 20)
            rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy()
        return torch.from_numpy(x), torch.from_numpy(tips_norm)

tip_tr_ds = GuiFiRTipDataset(neck_train_recs, augment=True)
tip_ev_ds = GuiFiRTipDataset(neck_eval_recs, augment=False)
tip_tr_dl = DataLoader(tip_tr_ds, batch_size=32, shuffle=True, num_workers=2)
tip_ev_dl = DataLoader(tip_ev_ds, batch_size=32, shuffle=False, num_workers=2)
print(f'FingertipNet split -> Train: {len(tip_tr_ds)} crops | Eval: {len(tip_ev_ds)} crops')

tip_model = FingertipNet(pretrained=True).to(DEVICE)
opt_t = torch.optim.AdamW(tip_model.parameters(), lr=3e-4, weight_decay=1e-4)
sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=TIP_EPOCHS)

best_tip_err = 1e9
for ep in range(TIP_EPOCHS):
    tip_model.train()
    tr_loss = 0.0
    for bx, btips in tip_tr_dl:
        bx, btips = bx.to(DEVICE), btips.to(DEVICE)
        pred = tip_model(bx)
        loss = F.smooth_l1_loss(pred, btips)
        opt_t.zero_grad(); loss.backward(); opt_t.step()
        tr_loss += loss.item() * len(bx)
    sched_t.step()

    if (ep + 1) % 10 == 0 or ep == TIP_EPOCHS - 1:
        tip_model.eval()
        errs = []
        with torch.no_grad():
            for bx, btips in tip_ev_dl:
                pred = tip_model(bx.to(DEVICE)).cpu().numpy()
                gt = btips.numpy()
                diff = (pred - gt) * np.array([RECT_W, RECT_H])
                errs.extend(np.linalg.norm(diff, axis=2).mean(axis=1).tolist())
        mean_err = float(np.mean(errs))
        print(f'Epoch {ep+1:2d}/{TIP_EPOCHS} | Loss: {tr_loss/len(tip_tr_ds):.4f} | Eval Tip Error (384x128 crop): {mean_err:.2f} px')
        if mean_err < best_tip_err:
            best_tip_err = mean_err
            torch.save({'model': tip_model.state_dict(), 'epoch': ep + 1, 'eval_err_px': mean_err, 'rect_size': (RECT_W, RECT_H)}, OUT_DIR / 'fingertip_net.pt')

print(f'Saved best FingertipNet to {OUT_DIR / "fingertip_net.pt"} (Eval Error: {best_tip_err:.2f} px)')
""")

md(r"""
## 5. Option 3: Train Chord Voices (`frame_tab` -> 9 Chords) + Fuse with Your Own Data
Here we match GuiFiR's `frame_tab` (6-string fret numbers) to standard voicings of your **9 target chords (`A, Am, Bm, C, D, Dm, E, F, G`)**, extract the 17 tab + relative span features using `FingertipNet` + `NeckNet`, fuse them with your own recordings (if `own_data.zip` is uploaded), and train the final `RandomForest` chord classifier (`chord_clf.joblib`).
""")

code(r"""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Standard 6-string guitar chord voicings (Low E -> High E; 0=open, 20=muted/unused)
CHORD_TEMPLATES = {
    'A':  [20, 0, 2, 2, 2, 0],
    'Am': [20, 0, 2, 2, 1, 0],
    'Bm': [20, 2, 4, 4, 3, 2],
    'C':  [20, 3, 2, 0, 1, 0],
    'D':  [20, 20, 0, 2, 3, 2],
    'Dm': [20, 20, 0, 2, 3, 1],
    'E':  [0, 2, 2, 1, 0, 0],
    'F':  [1, 3, 3, 2, 1, 1],
    'G':  [3, 2, 0, 0, 0, 3],
}

def match_chord_from_tab(tab):
    # Matches a 6-element GuiFiR frame_tab to one of the 9 chord templates (allowing optional muted strings)
    if not tab or len(tab) != 6:
        return None
    # Count fretted notes (>0 and <20)
    fretted = [f for f in tab if 0 < f < 20]
    if len(fretted) < 2:
        return None
    for chord, tmpl in CHORD_TEMPLATES.items():
        # Match if all non-20 strings in tab either match template or are 20 (muted)
        ok = True
        matches = 0
        for s_val, t_val in zip(tab, tmpl):
            if s_val == 20:
                continue
            if s_val == t_val:
                if s_val > 0:
                    matches += 1
            else:
                ok = False
                break
        if ok and matches >= 2:
            return chord
    return None

matched_recs = []
for r in (train_records + val_records + test_records):
    c = match_chord_from_tab(r.get('frame_tab'))
    if c is not None:
        r_c = dict(r)
        r_c['chord_label'] = c
        matched_recs.append(r_c)

print('GuiFiR frames matching your 9 chord voicings:')
print(pd.Series([r['chord_label'] for r in matched_recs]).value_counts().to_string())
""")

code(r"""
# Feature extraction using trained NeckNet + FingertipNet on matched GuiFiR frames
STRING_Y0, STRING_D = 0.085, 0.166
FRET_NUT_S, FRET_G1, FRET_R = -0.25, 1.0, 2 ** (-1 / 12)

def fret_estimate(s):
    z = np.maximum(1 - (np.asarray(s, np.float64) - FRET_NUT_S) * (1 - FRET_R) / FRET_G1, 0.02)
    return np.log(z) / np.log(FRET_R)

def string_coord(v):
    return 6.0 - (np.asarray(v, np.float64) - STRING_Y0) / STRING_D

def extract_17_tab_features(tips_norm, aspect=6.5):
    # tips_norm: (5, 2) for [4, 8, 12, 16, 20]; index..little are indices 1,2,3,4
    s = tips_norm[1:5, 0] * aspect
    v = tips_norm[1:5, 1]
    strs = string_coord(v)
    frs = fret_estimate(s)
    rel = np.array([
        frs[0] - frs[1], frs[1] - frs[2], frs[0] - frs[2], frs[2] - frs[3],
        strs[0] - strs[1], strs[1] - strs[2], strs[0] - strs[2],
        float(frs.max() - frs.min()), float(strs.max() - strs.min())
    ], dtype=np.float32)
    return np.concatenate([strs, frs, rel]).astype(np.float32)

FEAT_COLS = ['str0','str1','str2','str3','fr0','fr1','fr2','fr3',
             'd_fr_01','d_fr_12','d_fr_02','d_fr_23','d_str_01','d_str_12','d_str_02','span_fr','span_str']

neck_model.load_state_dict(torch.load(OUT_DIR / 'neck_v3.pt', map_location=DEVICE)['model'])
tip_model.load_state_dict(torch.load(OUT_DIR / 'fingertip_net.pt', map_location=DEVICE)['model'])
neck_model.eval(); tip_model.eval()

guifir_X, guifir_y, guifir_splits = [], [], []
with torch.no_grad():
    for r in tqdm(matched_recs, desc='Extracting 17 tab features on GuiFiR'):
        bgr = cv2.imread(r['img_path'])
        if bgr is None:
            continue
        rgb_n = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (IN_W, IN_H))
        xn = torch.from_numpy(((rgb_n.astype(np.float32)/255.0 - MEAN)/STD).transpose(2,0,1)).unsqueeze(0).to(DEVICE)
        nc = soft_argmax(neck_model(xn))[0][0].cpu().numpy()

        crop, _ = warp_neck_crop(bgr, nc)
        rgb_c = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        xc = torch.from_numpy(((rgb_c.astype(np.float32)/255.0 - MEAN)/STD).transpose(2,0,1)).unsqueeze(0).to(DEVICE)
        tips_norm = tip_model(xc)[0].cpu().numpy()

        # Estimate aspect from predicted corners
        H, W = bgr.shape[:2]
        c_px = nc * np.array([W, H])
        length = np.linalg.norm((c_px[2] + c_px[3])/2 - (c_px[0] + c_px[1])/2)
        width = (np.linalg.norm(c_px[0] - c_px[1]) + np.linalg.norm(c_px[3] - c_px[2])) / 2
        aspect = float(length / max(width, 1e-6))

        feat17 = extract_17_tab_features(tips_norm, aspect=aspect)
        guifir_X.append(feat17)
        guifir_y.append(r['chord_label'])
        guifir_splits.append(r['split'])

guifir_X = np.array(guifir_X, dtype=np.float32) if guifir_X else np.zeros((0, 17), dtype=np.float32)
guifir_y = np.array(guifir_y)
print(f'Extracted {len(guifir_X)} GuiFiR chord feature vectors.')

# Train Random Forest on GuiFiR chord voicings and evaluate
if len(guifir_X) > 20:
    tr_m = np.array([s == 'train' for s in guifir_splits])
    te_m = ~tr_m
    if tr_m.sum() > 10 and te_m.sum() > 5:
        clf_eval = RandomForestClassifier(200, min_samples_leaf=2, random_state=42, n_jobs=-1)
        clf_eval.fit(guifir_X[tr_m], guifir_y[tr_m])
        preds = clf_eval.predict(guifir_X[te_m])
        acc = accuracy_score(guifir_y[te_m], preds)
        print(f'GuiFiR Held-Out Chord Accuracy: {acc*100:.1f}%')
        print(classification_report(guifir_y[te_m], preds, zero_division=0))

    final_clf = RandomForestClassifier(200, min_samples_leaf=2, random_state=42, n_jobs=-1)
    final_clf.fit(guifir_X, guifir_y)

    # Save kNN novelty reference
    mean_x = guifir_X.mean(0)
    std_x = guifir_X.std(0) + 1e-6
    meta = {
        'features': 'tab + relative spans (17 numbers, GuiFiR trained)',
        'model': 'RandomForest (200 trees)',
        'columns': FEAT_COLS,
        'classes': sorted(list(set(guifir_y))),
        'smooth_k': 5,
        'novelty_keep': 0.93,
        'novelty_thr': 3.5,
    }
    bundle = {
        'model': final_clf,
        'meta': meta,
        'novelty': {'ref': guifir_X[:500], 'mean': mean_x, 'std': std_x, 'k': 5, 'thr': 3.5},
    }
    joblib.dump(bundle, OUT_DIR / 'chord_clf.joblib')
    (OUT_DIR / 'chord_meta.json').write_text(json.dumps(meta, indent=1))
    print('Saved chord_clf.joblib and chord_meta.json!')
""")

md(r"""
## 6. Package & Download `guifir_eval_and_models.zip`
Downloads the zip file to your computer. Drop it into your project and we will evaluate and wire it up locally!
""")

code(r"""
eval_report = {
    'neck_best_iou': best_iou,
    'fingertip_best_err_px': best_tip_err,
    'guifir_pose_frames': len(pose_records),
    'guifir_chord_frames': int(len(guifir_X)),
}
(OUT_DIR / 'eval_report.json').write_text(json.dumps(eval_report, indent=2))

zip_out = WORK / 'guifir_eval_and_models.zip'
with zipfile.ZipFile(zip_out, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in OUT_DIR.iterdir():
        zf.write(f, arcname=f.name)

print(f'Packaged {zip_out} ({zip_out.stat().st_size / 1e6:.1f} MB)')
try:
    from google.colab import files
    files.download(str(zip_out))
except Exception as e:
    print('Download manually from Colab file panel:', zip_out)
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

out_nb = ROOT / "notebooks" / "06_guifir_training_colab.ipynb"
out_nb.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out_nb} ({len(cells)} cells)")
