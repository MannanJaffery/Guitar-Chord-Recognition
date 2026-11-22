"""The real-time chord pipeline, shared by scripts/live_local.py and the Colab notebook (which embeds these functions verbatim).

Per frame: neck corners -> smooth them over time -> MediaPipe hand -> hand points in neck coordinates -> classifier probabilities
-> average over the last few frames -> answer, or "not sure" below the threshold.
"""
import collections

import cv2
import numpy as np

from geometry_features import neck_geometry, pick_fretting_hand, neck_coords, feature_vector
from neck_structure import normalize_quad, rectify, find_strings, find_frets

TIP_IDS = (4, 8, 12, 16, 20)


class Pipeline:
    """Feed it frames with step(); it keeps the smoothing state between frames."""

    def __init__(self, neck, hand, clf, cols, smooth_k=5, threshold=0.0):
        self.neck, self.hand, self.clf, self.cols = neck, hand, clf, list(cols)    # cols = indices into the 42 features the classifier uses
        self.smooth_k, self.threshold = smooth_k, threshold
        self.reset()

    def reset(self):
        self.ema = None; self.miss = 0; self.nohand = 0
        self.hist = collections.deque(maxlen=self.smooth_k)

    def step(self, bgr):
        H, W = bgr.shape[:2]
        c = self.neck(bgr); g = neck_geometry(c, W, H)
        if g['valid']:
            self.ema = c if self.ema is None else 0.6 * self.ema + 0.4 * c; self.miss = 0      # smooth the 4 corners: the neck finds itself frame after frame
        else:
            self.miss += 1
            if self.miss > 15: self.ema = None
        lm = None
        if self.ema is not None:
            gm = neck_geometry(self.ema, W, H); hands = self.hand(bgr)
            if gm['valid'] and hands:
                lm, _ = pick_fretting_hand(hands, self.ema)
                s, v = neck_coords(lm, gm['M'], gm['aspect']); feat, _ = feature_vector(s, v)
                self.hist.append(self.clf.predict_proba(feat[self.cols][None])[0]); self.nohand = 0
            else:
                self.nohand += 1
                if self.nohand > 8: self.hist.clear()                       # hand gone for a while: forget the old answer
        probs = np.mean(self.hist, axis=0) if self.hist else None
        answer = best = None; conf = 0.0
        if probs is not None:
            top = int(np.argmax(probs)); conf = float(probs[top]); best = self.clf.classes_[top]
            answer = best if conf >= self.threshold else None                  # None = "not sure"
        return dict(corners=self.ema, lm=lm, probs=probs, classes=list(self.clf.classes_), answer=answer, best=best, conf=conf)


def structure_segments(bgr, corners):
    # strings and frets found inside the straightened neck, mapped back onto the picture
    H, W = bgr.shape[:2]; c = normalize_quad(corners); g = neck_geometry(c, W, H)
    if not g['valid']: return [], []
    rect, M = rectify(bgr, c, g['aspect']); ys, sup, found = find_strings(rect); frets, f0, fs, _ = find_frets(rect, ys)
    Mi = np.linalg.inv(M); w = rect.shape[1]
    def back(p): return cv2.perspectiveTransform(np.array(p, np.float32).reshape(-1, 1, 2), Mi).reshape(-1, 2).astype(int)
    return [back([[0, y], [w, y]]) for y in ys], [back([[x, ys[0] - 4], [x, ys[-1] + 4]]) for x in frets]


def draw_overlay(img, out, fps=None, structure=True):
    s = max(img.shape[:2]) / 720.0
    if out['corners'] is not None:
        cv2.polylines(img, [out['corners'].astype(np.int32)], True, (0, 165, 255), max(2, int(3 * s)), cv2.LINE_AA)
        if structure:
            try:
                strs, frs = structure_segments(img, out['corners'])
                for a in strs: cv2.line(img, tuple(a[0]), tuple(a[1]), (255, 255, 0), 1, cv2.LINE_AA)
                for a in frs: cv2.line(img, tuple(a[0]), tuple(a[1]), (0, 255, 0), 2, cv2.LINE_AA)
            except Exception:
                pass
    if out['lm'] is not None:
        for i, (x, y) in enumerate(out['lm']):
            tip = i in TIP_IDS
            cv2.circle(img, (int(x), int(y)), 6 if tip else 3, (0, 255, 255) if tip else (200, 200, 200), -1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], int(58 * s)), (0, 0, 0), -1)
    if out['best'] is None: txt, col = 'no hand / no neck', (150, 150, 150)
    elif out['answer'] is None: txt, col = f"not sure  (best guess {out['best']} {out['conf']:.2f})", (200, 200, 200)
    else: txt, col = f"{out['answer']}   {out['conf']:.2f}", (0, 255, 255)
    cv2.putText(img, txt, (int(10 * s), int(42 * s)), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * s, col, max(2, int(2.5 * s)), cv2.LINE_AA)
    if fps: cv2.putText(img, f'{fps:.1f} fps', (img.shape[1] - int(130 * s), img.shape[0] - int(12 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (255, 255, 255), 2, cv2.LINE_AA)
    if out['probs'] is not None:                                          # the three most likely chords as small bars
        order = np.argsort(-out['probs'])[:3]
        for r, k in enumerate(order):
            y0 = img.shape[0] - int((20 + 26 * (2 - r)) * s)
            cv2.rectangle(img, (int(10 * s), y0), (int((10 + 160 * float(out['probs'][k])) * s), y0 + int(16 * s)), (0, 200, 255) if r == 0 else (120, 120, 120), -1)
            cv2.putText(img, f"{out['classes'][k]} {100 * float(out['probs'][k]):.0f}%", (int(180 * s), y0 + int(14 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (255, 255, 255), 1, cv2.LINE_AA)
    return img
