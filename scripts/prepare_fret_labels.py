"""Pick frames to label for fret numbering and crop each one tightly around the neck (so the frets are big enough to click).

Usage:
    python scripts/prepare_fret_labels.py [--per-chord 7] [--out artifacts/fret_label] [--models artifacts/models_local]
Writes <out>/frames/<Chord>_<take>_<frame>.jpg and <out>/frames/meta.json (where each crop came from, to map clicks back to the video frame).
Frames are chosen from those where the neck finder found a neck AND a fretting hand was found (the frames the classifier really uses),
spread evenly over each chord's takes and time. Run scripts/label_frets.py on the output folder afterwards.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_features import NeckFinder, HandFinder, neck_geometry, pick_fretting_hand    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC_W = 640                 # the neck finder runs on frames shrunk to this width (as in the live demo)
MARGIN = 0.30                # crop margin around the neck box, as a fraction of its size
OUT_W = 1400                 # crops are scaled up (at most 3x) toward this width


def crop_box(c, W, H, margin=MARGIN):
    x0, y0, x1, y1 = c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()
    mx, my = max((x1 - x0) * margin, 30), max((y1 - y0) * margin, 60)
    return int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(W, x1 + mx)), int(min(H, y1 + my))


def candidates(video, neck, hand, step_s=0.5):
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 30; step = max(1, int(round(fps * step_s))); i = -1; out = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        i += 1
        if i % step: continue
        H, W = bgr.shape[:2]; k = PROC_W / W if W > PROC_W else 1.0
        small = cv2.resize(bgr, None, fx=k, fy=k) if k < 1 else bgr
        c = neck(small); g = neck_geometry(c, small.shape[1], small.shape[0])
        if not g['valid']: continue
        hands = hand(small)
        if hands and pick_fretting_hand(hands, c) is not None: out.append((i, bgr, c / k))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--per-chord', type=int, default=7); ap.add_argument('--out', default=str(ROOT / 'artifacts' / 'fret_label'))
    ap.add_argument('--models', default=str(ROOT / 'artifacts' / 'models_local')); ap.add_argument('--videos', default=str(ROOT / 'artifacts' / 'own_videos'))
    a = ap.parse_args()
    d = Path(a.models); neck, hand = NeckFinder(d / 'neck_v2.pt', 'cpu'), HandFinder(d / 'hand_landmarker.task')
    out = Path(a.out) / 'frames'; out.mkdir(parents=True, exist_ok=True); meta = {}
    for chord_dir in sorted(p for p in Path(a.videos).iterdir() if p.is_dir()):
        pool = []
        for v in sorted(chord_dir.glob('take*.mp4')):
            pool += [(v, i, bgr, c) for i, bgr, c in candidates(v, neck, hand)]
        if not pool: print(chord_dir.name, ': no usable frames'); continue
        idx = np.unique(np.linspace(0, len(pool) - 1, min(a.per_chord, len(pool))).round().astype(int))
        for j in idx:
            v, i, bgr, c = pool[j]; H, W = bgr.shape[:2]
            x0, y0, x1, y1 = crop_box(c, W, H); crop = bgr[y0:y1, x0:x1]; sc = min(3.0, OUT_W / crop.shape[1])
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            name = f'{chord_dir.name}_{v.stem}_{i:04d}.jpg'; cv2.imwrite(str(out / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            meta[name] = dict(video=str(v.relative_to(Path(a.videos))), frame=int(i), x0=x0, y0=y0, scale=sc, orig_w=W, orig_h=H)
        print(f'{chord_dir.name}: {len(pool)} usable frames, wrote {len(idx)}')
    (out / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(f'{len(meta)} frames in {out}')


if __name__ == '__main__':
    main()
