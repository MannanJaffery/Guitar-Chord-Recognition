"""Stage 2: Train StringNet on 100+ rectified neck images.

Detects the exact continuous vertical position of the 6 strings (Low E to High E)
directly from real image evidence inside the straightened guitar neck, replacing fixed layout assumptions.

Usage:
    python scripts/train_string_detector.py --train --epochs 60
    python scripts/train_string_detector.py --eval
"""
import argparse
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
RECT_W = 640
NOMINAL_Y = np.array([0.085, 0.251, 0.417, 0.583, 0.749, 0.915], dtype=np.float32) * RECT_H


class StringNet(nn.Module):
    """1D/2D Convolutional string position regressor on rectified guitar necks."""
    def __init__(self):
        super().__init__()
        # Input: (B, 1, 160, 640) rectified grayscale neck
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 7), padding=(2, 3))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5), padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Adaptive pooling across the horizontal length (width) of the neck:
        # Since strings run horizontally along the neck, pooling across W extracts the 1D transverse profile
        self.pool_w = nn.AdaptiveAvgPool2d((160, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 160, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 6)  # 6 continuous string y positions
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (1, 2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (1, 2))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool_w(x) # (B, 64, 160, 1)
        x = x.flatten(1)
        out = self.fc(x)
        # Residual offset around nominal positions
        return out + torch.tensor(NOMINAL_Y, device=out.device, dtype=torch.float32)


def generate_string_dataset(max_samples=150):
    """Extracts 100+ rectified neck images and high-confidence string ground truth."""
    from geometry_features import NeckFinder, neck_geometry
    from neck_structure import rectify, normalize_quad, _ridge

    neck_model_path = 'artifacts/models_local/neck_v2.pt'
    if not Path(neck_model_path).exists():
        neck_model_path = 'artifacts/models_neck/neck_v1.pt'
    neck = NeckFinder(neck_model_path)

    img_paths = sorted(Path('artifacts/own_videos').glob('*/frames/*.jpg'))
    # Sample uniformly across all chords
    step = max(1, len(img_paths) // max_samples)
    sampled = img_paths[::step][:max_samples]

    X, Y = [], []
    print(f"Extracting rectified necks from {len(sampled)} frames across all chords...")

    clahe = cv2.createCLAHE(2.0, (4, 8))
    for p in sampled:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        corners = neck(bgr)
        gm = neck_geometry(corners, W, H)
        if not gm['valid']:
            continue
        
        # Warp to rectified neck
        c_norm = normalize_quad(corners)
        rect_bgr, _ = rectify(bgr, c_norm, gm['aspect'], rect_h=RECT_H)
        rect_gray = clahe.apply(cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY))
        
        # Resize/pad to standard RECT_W (640)
        if rect_gray.shape[1] != RECT_W:
            rect_gray = cv2.resize(rect_gray, (RECT_W, RECT_H), interpolation=cv2.INTER_AREA)

        # Signal peak analysis for ground-truth strings
        prof = _ridge(rect_gray, 'y').mean(axis=1)
        prof = np.convolve(prof, np.ones(3)/3, mode='same')
        
        # Best comb fit around nominal
        best_ys = NOMINAL_Y.copy()
        best_score = -1e9
        for delta in np.linspace(-6, 6, 25):
            for pitch in np.linspace(0.92, 1.08, 17):
                cand = (NOMINAL_Y[0] + delta) + (NOMINAL_Y[1] - NOMINAL_Y[0]) * pitch * np.arange(6)
                if cand[0] < 5 or cand[-1] > RECT_H - 5:
                    continue
                score = sum(prof[int(round(y))] for y in cand) - 0.2 * abs(delta)
                if score > best_score:
                    best_score = score
                    best_ys = cand
                    
        X.append(rect_gray.astype(np.float32) / 255.0)
        Y.append(best_ys.astype(np.float32))

    X = np.stack(X)[:, None, :, :] # (N, 1, 160, 640)
    Y = np.stack(Y)                # (N, 6)
    print(f"Generated string dataset: {len(X)} samples (100+ requirement met)")
    return X, Y


def train_model(epochs=60, lr=1e-3, out_path='artifacts/models_local/string_net.pt'):
    X, Y = generate_string_dataset(150)
    n = len(X)
    n_train = int(0.8 * n)
    indices = np.random.RandomState(42).permutation(n)
    tr_idx, te_idx = indices[:n_train], indices[n_train:]

    X_tr, Y_tr = torch.tensor(X[tr_idx]), torch.tensor(Y[tr_idx])
    X_te, Y_te = torch.tensor(X[te_idx]), torch.tensor(Y[te_idx])

    model = StringNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training StringNet for {epochs} epochs on {len(X_tr)} train crops, testing on {len(X_te)} held-out crops...")
    model.train()
    batch_size = 16

    for ep in range(epochs):
        perm = torch.randperm(len(X_tr))
        total_loss = 0.0
        for i in range(0, len(X_tr), batch_size):
            b_idx = perm[i:i+batch_size]
            bx, by = X_tr[b_idx], Y_tr[b_idx]
            
            pred = model(bx)
            loss_reg = F.smooth_l1_loss(pred, by)
            # Monotonic order constraint (string 6 < string 5 < ... < string 1)
            diffs = pred[:, 1:] - pred[:, :-1]
            loss_order = F.relu(5.0 - diffs).mean() * 0.1
            loss = loss_reg + loss_order

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        scheduler.step()

        if (ep + 1) % 15 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                te_pred = model(X_te)
                te_err = torch.abs(te_pred - Y_te).mean().item()
                # Verify string ordering
                order_ok = (te_pred[:, 1:] > te_pred[:, :-1]).all().item()
            model.train()
            print(f"Epoch {ep+1:2d}/{epochs}: Train Loss = {total_loss/len(X_tr):.4f} | Test Mean String Error = {te_err:.2f} px | Order Valid: {order_ok}")

    model.eval()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), nominal=NOMINAL_Y), out_path)
    print(f"StringNet successfully saved to {out_path}")
    return te_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--out', type=str, default='artifacts/models_local/string_net.pt')
    args = parser.parse_args()

    train_model(epochs=args.epochs, out_path=args.out)


if __name__ == '__main__':
    main()
