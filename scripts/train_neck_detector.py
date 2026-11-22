"""Train a small neck-corner detector from the labels made with scripts/label_neck.py.

Model: a ResNet-18 (pretrained on everyday photos) with a small upsampling head that outputs one heatmap per corner;
the corner position is read out with a differentiable soft-argmax. Four corners = the fretboard quadrilateral.

Usage (needs PyTorch, torchvision, opencv, numpy):
    python train_neck_detector.py --frames <dir> --labels <neck_labels.json> --out <model.pt> --test-take take3

Frames are named like C_take1_000.jpg; --test-take holds out every frame of that take for an honest test.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IN_W, IN_H = 384, 224                       # network input size (frames are resized to this)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


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
        return self.head(p)                                   # (B, 4, H/4, W/4) heatmap logits


def soft_argmax(logits):
    # logits (B,4,h,w) -> normalised coordinates (B,4,2) in [0,1], plus the probability maps
    B, K, h, w = logits.shape
    prob = F.softmax(logits.view(B, K, -1), dim=-1).view(B, K, h, w)
    xs = (torch.arange(w, dtype=torch.float32, device=logits.device) + 0.5) / w
    ys = (torch.arange(h, dtype=torch.float32, device=logits.device) + 0.5) / h
    x = (prob.sum(2) * xs).sum(-1); y = (prob.sum(3) * ys).sum(-1)
    return torch.stack([x, y], -1), prob


def gaussian_target(pts_norm, h, w, sigma=1.5):
    B, K, _ = pts_norm.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij')
    t = torch.zeros(B, K, h, w)
    for b in range(B):
        for k in range(K):
            cx, cy = pts_norm[b, k, 0] * w - 0.5, pts_norm[b, k, 1] * h - 0.5
            g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
            t[b, k] = g / g.sum().clamp_min(1e-8)
    return t


def prep(img_bgr):
    x = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy())


def augment(img, pts, rng):
    H, W = img.shape[:2]
    ang, sc = rng.uniform(-12, 12), rng.uniform(0.8, 1.2)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc)
    M[0, 2] += rng.uniform(-0.1, 0.1) * W; M[1, 2] += rng.uniform(-0.1, 0.1) * H
    out = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    p = np.hstack([pts, np.ones((4, 1))]) @ M.T
    if rng.random() < 0.5:                                   # mirror: corner meaning (nut-side/body-side, upper/lower) is unchanged
        out = out[:, ::-1].copy(); p[:, 0] = W - p[:, 0]
    out = out.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 2.0))
    return np.clip(out, 0, 255).astype(np.uint8), p


def poly_iou(a, b, size=(320, 180), ref=(1280, 720)):
    s = np.array([size[0] / ref[0], size[1] / ref[1]])
    ma, mb = np.zeros(size[::-1], np.uint8), np.zeros(size[::-1], np.uint8)
    cv2.fillPoly(ma, [(np.array(a) * s).astype(np.int32)], 1); cv2.fillPoly(mb, [(np.array(b) * s).astype(np.int32)], 1)
    inter, union = (ma & mb).sum(), (ma | mb).sum()
    return float(inter / union) if union else 0.0


@torch.no_grad()
def predict(model, img_bgr):
    model.eval()
    H, W = img_bgr.shape[:2]
    dev = next(model.parameters()).device                    # works on CPU and on a GPU
    co, prob = soft_argmax(model(prep(img_bgr)[None].to(dev)))
    conf = prob.flatten(2).max(-1).values[0]                 # how peaked each corner heatmap is
    return (co[0].cpu().numpy() * np.array([W, H])), conf.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True); ap.add_argument('--labels', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--test-take', default='take3'); ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--threads', type=int, default=8); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); random.seed(a.seed)
    labels = {k: v for k, v in json.loads(Path(a.labels).read_text(encoding='utf-8')).items() if v}
    data = {k: (cv2.imread(str(Path(a.frames) / k)), np.array(v, np.float64)) for k, v in labels.items()}
    test = sorted(k for k in data if f'_{a.test_take}_' in k); train = sorted(k for k in data if k not in test)
    print(f'{len(train)} training frames, {len(test)} held-out frames ({a.test_take})', flush=True)
    model = NeckNet()
    opt = torch.optim.AdamW([{'params': [p for n, p in model.named_parameters() if n.startswith(('stem', 'l1', 'l2', 'l3', 'l4'))], 'lr': 3e-4},
                             {'params': [p for n, p in model.named_parameters() if not n.startswith(('stem', 'l1', 'l2', 'l3', 'l4'))], 'lr': 1e-3}], weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train) / 8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-4, 1e-3], total_steps=a.epochs * steps_per_epoch)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); order = list(train); random.shuffle(order); tot = 0.0
        for i in range(0, len(order), 8):
            batch = order[i:i + 8]
            xs, ps = [], []
            for k in batch:
                img, p = augment(data[k][0], data[k][1], rng)
                H, W = img.shape[:2]
                xs.append(prep(img)); ps.append(torch.tensor(np.clip(p / np.array([W, H]), 0.0, 1.0), dtype=torch.float32))
            x, tgt = torch.stack(xs), torch.stack(ps)
            logits = model(x)
            co, prob = soft_argmax(logits)
            heat = gaussian_target(tgt, logits.shape[2], logits.shape[3])
            kl = (heat * (torch.log(heat.clamp_min(1e-8)) - torch.log(prob.clamp_min(1e-8)))).sum((-1, -2)).mean()
            loss = F.smooth_l1_loss(co, tgt, beta=0.02) * 20 + 0.1 * kl
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(batch)
        if ep % 10 == 0 or ep == 1:
            msg = f'epoch {ep:3d}  loss {tot / len(train):.4f}  ({time.time() - t0:.0f}s)'
            if test:
                errs = [np.linalg.norm(predict(model, data[k][0])[0] - data[k][1], axis=1).mean() for k in test]
                msg += f'  | held-out mean corner error {np.mean(errs):.1f} px'
            print(msg, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'in_size': (IN_W, IN_H), 'test_take': a.test_take, 'train': train, 'test': test}, a.out)
    if test:
        res = []
        for k in test:
            pr, conf = predict(model, data[k][0]); res.append((k, np.linalg.norm(pr - data[k][1], axis=1).mean(), poly_iou(pr, data[k][1])))
        ious = np.array([r[2] for r in res]); errs = np.array([r[1] for r in res])
        print(f'\nHELD-OUT {a.test_take}: {len(test)} frames | mean corner error {errs.mean():.1f} px (frame is 1280x720) | median {np.median(errs):.1f}'
              f' | polygon IoU mean {ious.mean():.2f}, median {np.median(ious):.2f} | IoU>0.5: {100 * (ious > 0.5).mean():.0f}% | IoU>0.7: {100 * (ious > 0.7).mean():.0f}%')
    print('saved', a.out)


if __name__ == '__main__':
    main()
