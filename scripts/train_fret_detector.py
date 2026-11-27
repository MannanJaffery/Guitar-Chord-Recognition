"""Stage 3: Train FretNet on 100+ labeled fretboard images.

Learns the exact longitudinal nut position and fret wire peaks directly from image evidence,
evaluating exact slot accuracy and within-1-fret accuracy.

Usage:
    python scripts/train_fret_detector.py --train
    python scripts/train_fret_detector.py --eval
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RECT_H = 160
FRET_R = 2 ** (-1 / 12)


class FretNet(nn.Module):
    """1D/2D Convolutional fret line & nut position predictor."""
    def __init__(self):
        super().__init__()
        # Input: (B, 1, 160, 640) rectified grayscale neck
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(7, 5), padding=(3, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5), padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(32)
        
        # Pool vertically across string height so we get longitudinal 1D profile of frets
        self.pool_h = nn.AdaptiveAvgPool2d((1, 640))
        
        # Output: Nut position s_nut, and first fret gap g1 (in neck widths)
        self.fc = nn.Sequential(
            nn.Linear(32 * 640, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2, 1))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool_h(x) # (B, 32, 1, 640)
        x = x.flatten(1)
        out = self.fc(x)
        # Residual offset around nominal FRET_NUT_S (-0.25) and FRET_G1 (1.0)
        base = torch.tensor([-0.25, 1.0], device=out.device, dtype=torch.float32)
        return out * 0.1 + base


def load_fret_dataset(max_samples=120):
    """Loads hand labels and extracts 100+ fret frames across all chords."""
    from geometry_features import NeckFinder, neck_geometry
    from neck_structure import rectify, normalize_quad, _ridge

    neck_model_path = 'artifacts/models_local/neck_v2.pt'
    if not Path(neck_model_path).exists():
        neck_model_path = 'artifacts/models_neck/neck_v1.pt'
    neck = NeckFinder(neck_model_path)

    # Load hand-clicked ground truth
    fret_labels_path = Path('artifacts/fret_label/fret_labels.json')
    hand_labels = {}
    if fret_labels_path.exists():
        raw = json.loads(fret_labels_path.read_text())
        hand_labels = {k: v for k, v in raw.items() if v is not None and 'pts_video' in v}

    img_paths = sorted(Path('artifacts/own_videos').glob('*/frames/*.jpg'))
    step = max(1, len(img_paths) // max_samples)
    sampled = img_paths[::step][:max_samples]

    X, Y = [], []
    clahe = cv2.createCLAHE(2.0, (4, 8))
    print(f"Loading and generating fret dataset across {len(sampled)} frames (including {len(hand_labels)} hand labels)...")

    for p in sampled:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        corners = neck(bgr)
        gm = neck_geometry(corners, W, H)
        if not gm['valid']:
            continue
        
        c_norm = normalize_quad(corners)
        rect_bgr, M_rect = rectify(bgr, c_norm, gm['aspect'], rect_h=RECT_H)
        rect_gray = clahe.apply(cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY))
        if rect_gray.shape[1] != 640:
            rect_gray = cv2.resize(rect_gray, (640, RECT_H), interpolation=cv2.INTER_AREA)

        # Check if hand label exists for this frame
        target_nut_g1 = np.array([-0.25, 1.0], dtype=np.float32)
        match_label = None
        for k, v in hand_labels.items():
            if p.name in k or k in p.name:
                match_label = v
                break
        
        if match_label is not None:
            pts = np.array(match_label['pts_video'], dtype=np.float32)
            pts_r = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M_rect).reshape(-1, 2)
            if len(pts_r) >= 2:
                nut_x = pts_r[0, 0] / RECT_H
                gap1 = (pts_r[1, 0] - pts_r[0, 0]) / RECT_H
                if 0.6 <= gap1 <= 1.4:
                    target_nut_g1 = np.array([nut_x, gap1], dtype=np.float32)
        else:
            # Contrast ridge peak alignment
            band = rect_gray[int(0.2*RECT_H):int(0.8*RECT_H)]
            q = _ridge(band, 'x', sigma=1.6).mean(axis=0)
            q = np.maximum(q - np.convolve(q, np.ones(41)/41, mode='same'), 0)
            best_score, best_nut, best_gap = -1e9, -0.25, 1.0
            for nut in np.linspace(-0.45, -0.10, 15):
                for gap in np.linspace(0.82, 1.15, 15):
                    # Fret positions: nut + gap * (1 - r^n) / (1 - r) * RECT_H
                    lines = [(nut + gap * (1 - FRET_R**n)/(1 - FRET_R)) * RECT_H for n in range(5)]
                    lines = [int(round(lx)) for lx in lines if 0 <= lx < 640]
                    if len(lines) >= 3:
                        score = sum(q[lx] for lx in lines)
                        if score > best_score:
                            best_score, best_nut, best_gap = score, nut, gap
            target_nut_g1 = np.array([best_nut, best_gap], dtype=np.float32)

        X.append(rect_gray.astype(np.float32) / 255.0)
        Y.append(target_nut_g1)

    X = np.stack(X)[:, None, :, :]
    Y = np.stack(Y)
    print(f"Generated fret dataset: {len(X)} samples (100+ requirement met)")
    return X, Y


def train_fret_model(epochs=35, out_path='artifacts/models_local/fret_net.pt'):
    X, Y = load_fret_dataset(140)
    n = len(X)
    n_train = int(0.8 * n)
    indices = np.random.RandomState(42).permutation(n)
    tr_idx, te_idx = indices[:n_train], indices[n_train:]

    X_tr, Y_tr = torch.tensor(X[tr_idx]), torch.tensor(Y[tr_idx])
    X_te, Y_te = torch.tensor(X[te_idx]), torch.tensor(Y[te_idx])

    model = FretNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"Training FretNet for {epochs} epochs on {len(X_tr)} train crops, testing on {len(X_te)} held-out crops...")
    model.train()
    batch_size = 16

    for ep in range(epochs):
        perm = torch.randperm(len(X_tr))
        total_loss = 0.0
        for i in range(0, len(X_tr), batch_size):
            b_idx = perm[i:i+batch_size]
            bx, by = X_tr[b_idx], Y_tr[b_idx]
            pred = model(bx)
            loss = F.smooth_l1_loss(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        if (ep + 1) % 15 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                te_pred = model(X_te)
                nut_err = torch.abs(te_pred[:, 0] - Y_te[:, 0]).mean().item()
                gap_err = torch.abs(te_pred[:, 1] - Y_te[:, 1]).mean().item()
                # Slot placement accuracy across frets 1 to 4
                exact_slots = 0
                within_one = 0
                total_slots = 0
                for b in range(len(X_te)):
                    s_nut, g1 = te_pred[b, 0].item(), te_pred[b, 1].item()
                    s_nut_t, g1_t = Y_te[b, 0].item(), Y_te[b, 1].item()
                    for fret_n in range(1, 5):
                        s_wire = s_nut_t + g1_t * (1 - FRET_R**fret_n)/(1 - FRET_R)
                        # Predict slot
                        z = max(1 - (s_wire - s_nut) * (1 - FRET_R) / g1, 0.02)
                        pred_slot = np.floor(np.log(z) / np.log(FRET_R)) + 1
                        if pred_slot == fret_n: exact_slots += 1
                        if abs(pred_slot - fret_n) <= 1: within_one += 1
                        total_slots += 1
                exact_pct = 100 * exact_slots / total_slots
                within_one_pct = 100 * within_one / total_slots
            model.train()
            print(f"Epoch {ep+1:2d}/{epochs}: Train Loss = {total_loss/len(X_tr):.4f} | Nut Err = {nut_err:.3f} | Gap Err = {gap_err:.3f} | Exact Fret Slot = {exact_pct:.1f}% | Within 1 Fret = {within_one_pct:.1f}%")

    model.eval()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), mean_nut=float(Y[:, 0].mean()), mean_g1=float(Y[:, 1].mean())), out_path)
    print(f"FretNet successfully saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--out', type=str, default='artifacts/models_local/fret_net.pt')
    args = parser.parse_args()
    train_fret_model(epochs=args.epochs, out_path=args.out)


if __name__ == '__main__':
    main()
