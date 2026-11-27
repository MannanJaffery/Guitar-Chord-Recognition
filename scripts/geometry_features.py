"""From video frames to finger-position numbers in the NECK's own coordinates (no room, no lighting, no photo style).

Steps for every frame:
  1. the learned neck finder gives the 4 corners of the fretboard (nut-side upper/lower, body-side lower/upper)
  2. MediaPipe gives the 21 hand points; the fretting hand is the one lying on the neck
  3. a homography (perspective transform) maps the neck quadrilateral onto a rectangle, so every hand point becomes
       s = position ALONG the neck, measured in neck-widths from the visible nut-side end
       v = position ACROSS the neck, 0 = upper edge, 1 = lower edge (so ~string position)
  4. features: v of all 21 points (absolute) and s of all 21 points relative to the fingertips' average (hand-centric),
     because the visible nut-side end moves with the camera framing, but the hand shape on the neck does not.

Usage:  python geometry_features.py --videos <folder with <Chord>/take1.mp4 ...> --neck-model <neck_v1.pt> --out features.csv
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch                                                   # noqa: E402
from train_neck_detector import NeckNet, predict               # noqa: E402

TIPS = [4, 8, 12, 16, 20]
FRET_TIPS = [8, 12, 16, 20]
UNIT = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], np.float32)   # corners 1,2,3,4 -> nut-upper, nut-lower, body-lower, body-upper


class NeckFinder:
    def __init__(self, path, device='cpu'):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.model = NeckNet(pretrained=False); self.model.load_state_dict(ck['model']); self.model.to(device).eval()

    def __call__(self, bgr):
        return predict(self.model, bgr)[0]                      # 4x2 corners in pixels


def neck_geometry(corners, W, H):
    """Returns dict(valid, reason, M (homography to unit rectangle), aspect, width_px) for the 4 predicted corners."""
    c = np.asarray(corners, np.float32)
    area = abs(cv2.contourArea(c)) / (W * H)
    top_mid, bot_mid = (c[0] + c[1]) / 2, (c[2] + c[3]) / 2
    length = float(np.linalg.norm(bot_mid - top_mid))
    width = float((np.linalg.norm(c[0] - c[1]) + np.linalg.norm(c[3] - c[2])) / 2)
    aspect = length / max(width, 1e-6)
    reason = None
    if not cv2.isContourConvex(c.reshape(-1, 1, 2)): reason = 'not convex'
    elif not (0.012 <= area <= 0.45): reason = f'area {area:.3f}'
    elif not (2.0 <= aspect <= 16): reason = f'aspect {aspect:.1f}'
    M = cv2.getPerspectiveTransform(c, UNIT) if reason is None else None
    return dict(valid=reason is None, reason=reason, M=M, aspect=aspect, width_px=width, area=area)


class HandFinder:
    def __init__(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(model_path)), running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3))

    def __call__(self, bgr):
        H, W = bgr.shape[:2]
        res = self.lm.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        return [np.array([[l.x * W, l.y * H] for l in h], np.float32) for h in res.hand_landmarks]


def pick_fretting_hand(hands, corners, max_along=0.85):
    """The fretting hand is the hand NEAREST THE NUT that touches the neck. (Picking "the most points inside the neck" also picked the
    strumming hand over the sound hole: in 35% of our frames, and in over half the frames of 8 of the 23 takes.)
    A hand whose fingertips are past `max_along` of the neck length (the body end) is the strumming hand: it is rejected.
    Returns (21 hand points, share of points inside the neck) or None when there is no fretting hand."""
    if not hands: return None
    c = np.asarray(corners, np.float32); ctr = c.mean(0); grown = (ctr + (c - ctr) * 1.35).astype(np.float32).reshape(-1, 1, 2)
    M = cv2.getPerspectiveTransform(c, UNIT)
    cands = []
    for h in hands:
        inside = float(np.mean([cv2.pointPolygonTest(grown, (float(x), float(y)), False) >= 0 for x, y in h]))
        along = float(cv2.perspectiveTransform(h[FRET_TIPS].reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)[:, 0].mean())    # 0 = nut, 1 = body end
        if inside > 0 and along <= max_along: cands.append((along, inside, h))
    if not cands: return None
    along, inside, h = min(cands, key=lambda t: t[0])
    return h, inside


def neck_coords(lm_px, M, aspect):
    uv = cv2.perspectiveTransform(lm_px.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
    return uv[:, 0] * aspect, uv[:, 1]                          # s in neck-widths along the neck, v across (0..1)


FRET_NUT_S = -0.25        # where the real nut sits along the neck outline, in neck widths (fitted on the user's hand-labelled frames)
FRET_G1 = 1.0             # gap between the nut and fret 1, in neck widths (same fit). Per fingertip: exact fret slot 68%, within one slot 95%.
OPEN_MAX_FRET = 3.5       # mean fingertip fret coordinate; the training chords reach at most 4.15 (99% of frames are below 3.1), beyond 3.5 the classifier has never seen a hand
MIN_ASPECT = 6.0          # an outline shorter than this (in neck widths) is a truncated neck: the fret estimate is not trusted
STRING_Y0, STRING_D = 0.085, 0.166   # string 6 (low E, the UPPER edge in the user's videos) sits at this fraction of the neck width, then every D (typical layout)
FRET_R = 2 ** (-1 / 12)


def fret_estimate(s):
    """Fret number (0 = nut) at distance s (neck widths) along the neck outline, from the 2^(-1/12) fret spacing. Accepts a number or an array."""
    r = 2 ** (-1 / 12); z = np.maximum(1 - (np.asarray(s, np.float64) - FRET_NUT_S) * (1 - r) / FRET_G1, 0.02)
    out = np.log(z) / np.log(r)
    return float(out) if out.ndim == 0 else out


def fret_line_s(n):
    """Position of fret line n (0 = nut) along the neck outline, in neck widths."""
    return FRET_NUT_S + FRET_G1 * (1 - FRET_R ** np.asarray(n, np.float64)) / (1 - FRET_R)


def fret_slot(s):
    """Fret number a fingertip is pressing: 1 = between the nut and fret 1, 2 = between fret 1 and 2, ... 0 = behind the nut."""
    return np.maximum(np.floor(fret_estimate(s)) + 1, 0).astype(int)


def string_coord(v):
    """Continuous string number from the across-neck position: 6.0 = low E string (upper edge), 1.0 = high E string. Round for 'the string'."""
    return 6.0 - (np.asarray(v, np.float64) - STRING_Y0) / STRING_D


def tab_features(s, v):
    """The explicit 'tab' of the fretting hand: 4 string numbers, 4 fret numbers, plus relative spans (17 numbers)."""
    idx = FRET_TIPS
    strs = string_coord(v[idx])
    frs = fret_estimate(s[idx])
    d_fr_01 = frs[0] - frs[1]
    d_fr_12 = frs[1] - frs[2]
    d_fr_02 = frs[0] - frs[2]
    d_fr_23 = frs[2] - frs[3]
    d_str_01 = strs[0] - strs[1]
    d_str_12 = strs[1] - strs[2]
    d_str_02 = strs[0] - strs[2]
    span_fr = float(np.max(frs) - np.min(frs))
    span_str = float(np.max(strs) - np.min(strs))
    rel_spans = np.array([d_fr_01, d_fr_12, d_fr_02, d_fr_23, d_str_01, d_str_12, d_str_02, span_fr, span_str], dtype=np.float64)
    return np.concatenate([strs, frs, rel_spans])


def feature_vector(s, v):
    """64 numbers per frame: 21 across-neck, 21 along-neck relative, 5 fret numbers, and 17 enriched tab numbers."""
    s_rel = s - s[FRET_TIPS].mean()
    frets = fret_estimate(s[TIPS])
    tab = tab_features(s, v)
    tab_names = [f'str{i}' for i in range(4)] + [f'fr{i}' for i in range(4)] + [
        'd_fr_01', 'd_fr_12', 'd_fr_02', 'd_fr_23', 'd_str_01', 'd_str_12', 'd_str_02', 'span_fr', 'span_str'
    ]
    names = [f'v{i}' for i in range(21)] + [f's{i}' for i in range(21)] + [f'g{i}' for i in range(5)] + tab_names
    return np.concatenate([v, s_rel, frets, tab]), names


def frame_features(bgr, neck, hand):
    H, W = bgr.shape[:2]
    corners = neck(bgr); g = neck_geometry(corners, W, H)
    out = dict(neck_valid=g['valid'], neck_reason=g['reason'], hand_found=False, inside=np.nan, feat=None, corners=corners)
    hands = hand(bgr); out['hand_found'] = bool(hands)
    if not g['valid'] or not hands: return out
    pick = pick_fretting_hand(hands, corners)
    if pick is None: return out                                                  # only the strumming hand (or a stray hand) was seen
    lm, inside = pick; out['inside'] = inside
    s, v = neck_coords(lm, g['M'], g['aspect']); out['feat'], _ = feature_vector(s, v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', required=True); ap.add_argument('--neck-model', required=True); ap.add_argument('--hand-model', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--step', type=float, default=0.5, help='seconds between sampled frames')
    a = ap.parse_args()
    import pandas as pd
    neck, hand = NeckFinder(a.neck_model), HandFinder(a.hand_model)
    names = feature_vector(np.zeros(21), np.zeros(21))[1]
    rows = []
    for vid in sorted(Path(a.videos).glob('*/take*.mp4')):
        chord, take = vid.parent.name, vid.stem
        (vid.parent / 'frames').mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(vid)); fps = cap.get(cv2.CAP_PROP_FPS) or 17.0; step = max(1, int(round(fps * a.step))); i = k = 0
        while True:
            ok, bgr = cap.read()
            if not ok: break
            if i % step == 0:
                fid = f'{chord}_{take}_{k:03d}'
                cv2.imwrite(str(vid.parent / 'frames' / f'{fid}.jpg'), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                r = frame_features(bgr, neck, hand)
                row = dict(id=fid, chord=chord, take=take, frame=k, neck_valid=r['neck_valid'], neck_reason=r['neck_reason'], hand_found=r['hand_found'], inside=r['inside'])
                row.update({n: (np.nan if r['feat'] is None else float(x)) for n, x in zip(names, r['feat'] if r['feat'] is not None else [np.nan] * 55)})
                row.update({f'c{j}{ax}': float(r['corners'][j][q]) for j in range(4) for q, ax in enumerate('xy')})
                rows.append(row); k += 1
            i += 1
        print(f'{chord}/{take}: {k} frames', flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print('saved', a.out, len(rows), 'rows')


if __name__ == '__main__':
    main()
