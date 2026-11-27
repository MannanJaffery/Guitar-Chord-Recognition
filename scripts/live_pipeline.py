"""The real-time chord pipeline, shared by scripts/live_local.py and the Colab notebook (which embeds these functions verbatim).

Per frame, in this order:
  1. the neck finder gives the fretboard corners (smoothed over time)         -> the neck, and from it the 6 strings and the numbered fret lines
  2. MediaPipe finds the fretting hand; its fingertips are mapped into neck coordinates
  3. each fretting fingertip gets a STRING number (1-6) and a FRET number    -> the "tab" of the hand
  4. the classifier decides the chord from those string + fret numbers only  -> average over the last few frames
  5. refuse: "out of open position" (fingers beyond the trained frets), "unfamiliar hand shape", or "not sure"
"""
import collections

import cv2
import numpy as np

from geometry_features import (neck_geometry, pick_fretting_hand, neck_coords, feature_vector, fret_estimate, fret_slot, fret_line_s, string_coord,
                               FRET_TIPS, OPEN_MAX_FRET, MIN_ASPECT, STRING_Y0, STRING_D, FRET_NUT_S)

TIP_IDS = (4, 8, 12, 16, 20)


def knn_distance(X, ref, mean, std, k=5):
    """How unfamiliar is a hand shape? Mean distance (in standardised units) from each row of X to its k nearest TRAINING frames."""
    A = (np.asarray(ref, np.float64) - mean) / std; B = (np.atleast_2d(np.asarray(X, np.float64)) - mean) / std
    D = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(2)); D.sort(1)
    return D[:, :k].mean(1)


class LandmarkTracker:
    """Tracks the 21 fretting hand landmarks across frames with velocity decay.
    Bridges 1-3 dropped frames when MediaPipe momentarily loses hand tracking due to occlusion."""
    def __init__(self, max_drop=3):
        self.last_lm = None
        self.vel = None
        self.drop_count = 0
        self.max_drop = max_drop

    def update(self, lm):
        if lm is not None:
            if self.last_lm is not None:
                self.vel = 0.6 * (lm - self.last_lm) + 0.4 * (self.vel if self.vel is not None else 0.0)
            self.last_lm = lm.copy()
            self.drop_count = 0
            return lm, False
        elif self.last_lm is not None and self.drop_count < self.max_drop:
            self.drop_count += 1
            if self.vel is not None:
                self.last_lm = self.last_lm + 0.5 * self.vel
            return self.last_lm.copy(), True
        else:
            self.last_lm = None
            self.vel = None
            return None, False

    def reset(self):
        self.last_lm = None
        self.vel = None
        self.drop_count = 0


class Pipeline:
    """Feed it frames with step(); it keeps the smoothing state between frames."""

    def __init__(self, neck, hand, clf, cols, smooth_k=5, threshold=0.0, novelty=None):
        names = feature_vector(np.zeros(21), np.zeros(21))[1]
        name_to_idx = {name: i for i, name in enumerate(names)}
        self.cols = [name_to_idx[c] if c in name_to_idx else (int(c[1:]) if str(c).startswith('f') else int(c)) for c in cols]
        self.neck, self.hand, self.clf = neck, hand, clf
        self.smooth_k, self.threshold, self.novelty = smooth_k, threshold, novelty     # novelty = dict(ref, mean, std, k, thr) or None
        self.tracker = LandmarkTracker(max_drop=3)
        try: self.clf.set_params(n_jobs=1)                                           # one frame at a time: a thread pool per frame is far slower than one thread
        except Exception: pass
        self.reset()

    def reset(self):
        self.ema = None; self.miss = 0; self.nohand = 0; self.trusted = False
        self.tracker.reset()
        self.hist = collections.deque(maxlen=self.smooth_k); self.fhist = collections.deque(maxlen=self.smooth_k); self.nhist = collections.deque(maxlen=self.smooth_k)

    def step(self, bgr):
        H, W = bgr.shape[:2]
        c = self.neck(bgr); g = neck_geometry(c, W, H)
        if g['valid']:
            self.ema = c if self.ema is None else 0.6 * self.ema + 0.4 * c; self.miss = 0      # smooth the 4 corners: the neck finds itself frame after frame
        else:
            self.miss += 1
            if self.miss > 15: self.ema = None; self.tracker.reset()
        lm = None; geom = None; tips = None
        if self.ema is not None:
            gm = neck_geometry(self.ema, W, H)
            if gm['valid']:
                hands = self.hand(bgr)
                pick = pick_fretting_hand(hands, self.ema) if hands else None
                raw_lm = pick[0] if pick is not None else None
                lm, extrapolated = self.tracker.update(raw_lm)
            else:
                self.tracker.reset()
            if lm is not None and gm['valid']:
                s, v = neck_coords(lm, gm['M'], gm['aspect']); feat, _ = feature_vector(s, v)
                geom = dict(M=gm['M'], aspect=gm['aspect'])
                tips = [dict(xy=lm[i], string=int(np.clip(np.round(string_coord(v[i])), 1, 6)), slot=int(fret_slot(s[i]))) for i in FRET_TIPS]     # the tab of the hand
                feat_sub = feat[self.cols]
                self.hist.append(self.clf.predict_proba(feat_sub[None])[0]); self.nohand = 0
                if self.novelty is not None: nv = self.novelty; self.nhist.append(float(knn_distance(feat_sub, nv['ref'], nv['mean'], nv['std'], nv['k'])[0]))
                self.fhist.append(fret_estimate(float(np.mean(s[list(FRET_TIPS)])))); self.trusted = gm['aspect'] >= MIN_ASPECT
            else:
                self.nohand += 1
                if self.nohand > 8: self.hist.clear(); self.fhist.clear(); self.nhist.clear()   # hand gone for a while: forget the old answer
        if geom is None and self.ema is not None:
            gm2 = neck_geometry(self.ema, W, H)
            if gm2['valid']: geom = dict(M=gm2['M'], aspect=gm2['aspect'])          # the strings and frets are drawn even without a hand
        probs = np.mean(self.hist, axis=0) if self.hist else None
        answer = best = None; conf = 0.0
        if probs is not None:
            top = int(np.argmax(probs)); conf = float(probs[top]); best = self.clf.classes_[top]
            answer = best if conf >= self.threshold else None                  # None = "not sure"
        fret = float(np.median(self.fhist)) if self.fhist else None          # about which fret the fretting fingertips are at (+-1)
        out_of_pos = bool(fret is not None and self.trusted and fret > OPEN_MAX_FRET)   # the classifier never saw a hand this far up the neck
        unfamiliar = bool(self.novelty is not None and self.nhist and float(np.median(self.nhist)) > self.novelty['thr'])    # the hand shape is unlike every trained chord
        if out_of_pos or unfamiliar: answer = None
        return dict(corners=self.ema, lm=lm, probs=probs, classes=list(self.clf.classes_), answer=answer, best=best, conf=conf, fret=fret, out_of_pos=out_of_pos, unfamiliar=unfamiliar, geom=geom, tips=tips)


def _to_image(geom, pts_sv):
    """neck coordinates (s along the neck in neck widths, v across 0..1) -> picture pixels"""
    p = np.array([[s / geom['aspect'], v] for s, v in pts_sv], np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.linalg.inv(geom['M'])).reshape(-1, 2)


def button_rect(img):
    s = max(img.shape[:2]) / 720.0
    return (img.shape[1] - int(212 * s), int(66 * s), img.shape[1] - int(8 * s), int(100 * s))


def draw_structure(img, geom, tips=None):
    """Draw the 6 strings (numbered 6..1) and the numbered fret lines, computed from the neck geometry: the same geometry that gives each fingertip its string and fret."""
    s = max(img.shape[:2]) / 720.0; A = geom['aspect']; lo = max(0.0, FRET_NUT_S); hi = 0.97 * A
    for k in range(1, 7):                                                 # strings
        v = STRING_Y0 + (6 - k) * STRING_D; a, b = _to_image(geom, [(lo, v), (hi, v)]).astype(int)
        cv2.line(img, tuple(a), tuple(b), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(k), (int(b[0]) + 3, int(b[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (255, 255, 0), 1, cv2.LINE_AA)
    for n in range(0, 13):                                                # frets 0 (nut) to 12
        x = float(fret_line_s(n))
        if x < 0: continue
        if x > hi: break
        a, b = _to_image(geom, [(x, 0.0), (x, 1.0)]).astype(int)
        cv2.line(img, tuple(a), tuple(b), (0, 255, 0) if n else (0, 0, 255), 2 if n else 3, cv2.LINE_AA)
        cv2.putText(img, 'nut' if n == 0 else str(n), (int(a[0]) - 6, int(a[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 255, 0), 1, cv2.LINE_AA)
    if tips:
        for t_ in tips:
            x, y = int(t_['xy'][0]), int(t_['xy'][1])
            cv2.putText(img, f"S{t_['string']} F{t_['slot']}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (0, 255, 255), 2, cv2.LINE_AA)


def draw_overlay(img, out, fps=None, structure=True, button=False):
    s = max(img.shape[:2]) / 720.0
    if out['corners'] is not None:
        cv2.polylines(img, [out['corners'].astype(np.int32)], True, (0, 165, 255), max(2, int(3 * s)), cv2.LINE_AA)
        if structure and out.get('geom') is not None:
            try: draw_structure(img, out['geom'], out.get('tips'))
            except Exception: pass
    if out['lm'] is not None:
        for i, (x, y) in enumerate(out['lm']):
            tip = i in TIP_IDS
            cv2.circle(img, (int(x), int(y)), 6 if tip else 3, (0, 255, 255) if tip else (200, 200, 200), -1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], int(58 * s)), (0, 0, 0), -1)
    if out['best'] is None: txt, col = 'no hand / no neck', (150, 150, 150)
    elif out.get('out_of_pos'): txt, col = f"out of open position (fret ~{out['fret']:.0f})", (0, 140, 255)
    elif out.get('unfamiliar'): txt, col = 'unfamiliar hand shape - no chord', (0, 140, 255)
    elif out['answer'] is None: txt, col = f"not sure  (best guess {out['best']} {out['conf']:.2f})", (200, 200, 200)
    else: txt, col = f"{out['answer']}   {out['conf']:.2f}", (0, 255, 255)
    cv2.putText(img, txt, (int(10 * s), int(42 * s)), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * s, col, max(2, int(2.5 * s)), cv2.LINE_AA)
    if out.get('fret') is not None: cv2.putText(img, f"fret ~{out['fret']:.1f}", (img.shape[1] - int(190 * s), int(46 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (0, 255, 0), 2, cv2.LINE_AA)
    if fps: cv2.putText(img, f'{fps:.1f} fps', (img.shape[1] - int(130 * s), img.shape[0] - int(12 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (255, 255, 255), 2, cv2.LINE_AA)
    if out['probs'] is not None:                                          # the three most likely chords as small bars
        order = np.argsort(-out['probs'])[:3]
        for r, k in enumerate(order):
            y0 = img.shape[0] - int((20 + 26 * (2 - r)) * s)
            cv2.rectangle(img, (int(10 * s), y0), (int((10 + 160 * float(out['probs'][k])) * s), y0 + int(16 * s)), (0, 200, 255) if r == 0 else (120, 120, 120), -1)
            cv2.putText(img, f"{out['classes'][k]} {100 * float(out['probs'][k]):.0f}%", (int(180 * s), y0 + int(14 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (255, 255, 255), 1, cv2.LINE_AA)
    if button:                                                            # clickable toggle (the mouse callback in live_local.py uses button_rect)
        x0, y0, x1, y1 = button_rect(img); cv2.rectangle(img, (x0, y0), (x1, y1), (0, 140, 0) if structure else (70, 70, 70), -1); cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)
        cv2.putText(img, 'strings+frets: ' + ('ON' if structure else 'OFF'), (x0 + int(8 * s), y1 - int(10 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (255, 255, 255), 1, cv2.LINE_AA)
    return img
