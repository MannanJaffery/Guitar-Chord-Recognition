"""Find the 6 strings and the fret lines inside the straightened neck.

The neck quadrilateral (from the learned neck finder) is warped to a flat rectangle:
    x = along the neck (0 = nut side, increasing toward the body), y = across the neck (0 = upper edge, RECT_H = lower edge).
In that clean view strings are near-horizontal lines and frets near-vertical lines, so simple 1-D signal analysis is enough.
"""
import cv2
import numpy as np
from scipy.signal import find_peaks

RECT_H = 160                                   # rectified neck height in pixels (the neck width)
NOMINAL_Y0, NOMINAL_D = 0.085, 0.166           # typical string layout as a fraction of the neck width: first string and spacing


FRET_RATIO = 2 ** (-1 / 12)                    # each fret gap is this much narrower than the previous one
FIRST_GAP = 0.84                               # gap between the nut and fret 1, in neck widths (typical acoustic)


def normalize_quad(c):
    """Re-cut the neck quadrilateral so its two ends are perpendicular to the neck axis (removes the shear that arbitrary end
    positions, for example where the neck runs off the picture, would cause). Uses only the two long edges."""
    c = np.asarray(c, np.float64)
    nut_mid, body_mid = (c[0] + c[1]) / 2, (c[2] + c[3]) / 2
    axis = body_mid - nut_mid
    if np.linalg.norm(axis) < 1e-6: return c.astype(np.float32)
    axis /= np.linalg.norm(axis); n = np.array([-axis[1], axis[0]])

    def hit(m, a, b):                                                # where the line m + t*n crosses the long edge a-b
        e = b - a; A = np.array([n, -e]).T
        if abs(np.linalg.det(A)) < 1e-9: return None
        return m + np.linalg.solve(A, a - m)[0] * n
    out = []
    for m in (nut_mid, body_mid):
        up, low = hit(m, c[0], c[3]), hit(m, c[1], c[2])
        if up is None or low is None: return c.astype(np.float32)
        out.append((up, low))
    return np.array([out[0][0], out[0][1], out[1][1], out[1][0]], np.float32)


def rectify(bgr, corners, aspect, rect_h=RECT_H):
    w = int(np.clip(aspect * rect_h, 320, 1800))
    dst = np.float32([[0, 0], [0, rect_h], [w, rect_h], [w, 0]])
    M = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    return cv2.warpPerspective(bgr, M, (w, rect_h), flags=cv2.INTER_LINEAR), M


def _ridge(gray, axis, sigma=1.4):
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    d = cv2.Sobel(g, cv2.CV_32F, 2 if axis == 'x' else 0, 2 if axis == 'y' else 0, ksize=3)
    r = np.abs(d)
    return np.minimum(r, np.percentile(r, 97))             # clip so one huge edge (finger, shadow) cannot dominate


def find_strings(rect_bgr):
    """Returns (y positions of the 6 strings in the rectified neck, quality 0..1, found?)."""
    gray = cv2.createCLAHE(2.0, (4, 8)).apply(cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY))
    H = gray.shape[0]
    prof = _ridge(gray, 'y').mean(axis=1)
    prof = np.convolve(prof, np.ones(3) / 3, mode='same'); prof = np.maximum(prof - np.convolve(prof, np.ones(41) / 41, mode='same'), 0)
    pd_ = cv2.dilate(prof.reshape(-1, 1).astype(np.float32), np.ones((5, 1), np.uint8)).ravel()
    best, nominal = None, np.array([(NOMINAL_Y0 + i * NOMINAL_D) * H for i in range(6)])
    for d in np.arange(0.12 * H, 0.21 * H, 0.5):
        for y0 in np.arange(0.02 * H, 0.22 * H, 1.0):
            ys = y0 + d * np.arange(6)
            if ys[-1] > 0.98 * H: continue
            v = np.sort(pd_[np.clip(ys.round().astype(int), 0, H - 1)])
            score = v[1:].sum() * np.sqrt(v[1] / (v[-1] + 1e-9)) - 0.15 * np.abs(ys - nominal).mean() * prof.max()    # prefer the usual layout
            if best is None or score > best[0]: best = (score, ys)
    ys = best[1]
    support = float(np.mean([prof[int(np.clip(round(y), 0, H - 1))] for y in ys]) / (prof.mean() + 1e-9))
    found = support > 1.3
    return (ys if found else nominal), support, found


def find_frets(rect_bgr, ys):
    """Fit the real fret pattern (gaps shrinking by 2^(-1/12) per fret) to the vertical line evidence.
    Returns (x positions of the fret lines, absolute number of the FIRST line (0 = the nut), fit score, raw evidence curve)."""
    gray = cv2.createCLAHE(2.0, (4, 8)).apply(cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY))
    H, W = gray.shape
    band = slice(int(max(0, ys[0] - 4)), int(min(H, ys[-1] + 5)))
    q = _ridge(gray[band], 'x', sigma=1.6).mean(axis=0)
    q = np.maximum(q - np.convolve(q, np.ones(61) / 61, mode='same'), 0)
    if q.max() <= 0: return np.array([]), np.nan, 0.0, q
    qd = cv2.dilate(q.reshape(1, -1).astype(np.float32), np.ones((1, 7), np.uint8)).ravel()
    base = 0.8 * float(qd.mean())
    best = None
    for g0 in np.arange(0.28 * H, 1.1 * H, 2.0):                     # gap at the first visible fret, in pixels
        offs, o, g = [], 0.0, g0                                     # offsets of successive frets from the first one
        while o < W and g > 6 and len(offs) < 26:                    # stop when the gaps become too small to see (the sum converges!)
            offs.append(o); o += g; g *= FRET_RATIO
        offs = np.array(offs)
        A = np.arange(0.0, g0, 3.0)[:, None] + offs[None, :]         # every candidate start position at once
        ok = A < W
        vals = np.where(ok, qd[np.clip(A.astype(int), 0, W - 1)] - base, 0.0)
        score = vals.sum(axis=1); n = ok.sum(axis=1)
        score = np.where(n >= 3, score, -np.inf)
        i = int(np.argmax(score))
        if np.isfinite(score[i]) and (best is None or score[i] > best[0]):
            best = (float(score[i]), A[i][ok[i]], g0)
    if best is None: return np.array([]), np.nan, 0.0, q
    score, xs, g0 = best
    first_no = float(np.log(g0 / (FIRST_GAP * H)) / np.log(FRET_RATIO))   # which fret the first line is (0 = nut) from the gap size
    return xs, first_no, score / (q.mean() * len(xs) + 1e-9), q


def fret_spacing_per_width(frets, rect_h=RECT_H):
    """Median distance between neighbouring detected frets, measured in neck widths (shrinks toward the body: an absolute-position hint)."""
    if len(frets) < 3: return np.nan
    return float(np.median(np.diff(frets)) / rect_h)


def string_coord(y, ys):
    """Continuous string index (0 = first string, 5 = sixth) for a y position in the rectified neck."""
    return float((y - ys[0]) / (np.mean(np.diff(ys)) + 1e-9))


def fret_coord(x, frets, first_no):
    """Fractional fret number of x (e.g. 2.4 = 40% of the way from fret 2 to fret 3)."""
    if len(frets) < 2 or not np.isfinite(first_no): return np.nan
    k = int(np.clip(np.searchsorted(frets, x) - 1, 0, len(frets) - 2))
    return first_no + k + float((x - frets[k]) / (frets[k + 1] - frets[k] + 1e-9))
