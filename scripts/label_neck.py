"""Click-to-label tool: mark the 4 corners of the guitar neck (fretboard) on each frame. Everything stays on your computer.

Usage:
    python scripts/label_neck.py <frames_folder> [--out neck_labels.json] [--every 1]
    python scripts/label_neck.py --selftest

Click the 4 corners IN THIS ORDER (always the same order, whichever way the guitar points):
    1  nut-side end of the visible fretboard, the corner that is UPPER in the picture
    2  nut-side end of the visible fretboard, the corner that is LOWER in the picture
    3  body-side end of the visible fretboard, the LOWER corner
    4  body-side end of the visible fretboard, the UPPER corner
"Nut-side" = toward the headstock. If the neck runs out of the picture, click where it meets the picture edge.
The fretboard is the wooden strip with the frets (not the back of the neck, not the body).

Keys:  ENTER/SPACE = save and go to the next picture | u = undo last click | s = skip (no neck visible: saved as "no neck")
       b = go back one picture | q or ESC = quit (everything already saved is kept; run again to continue)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_W, MAX_H = 1400, 800            # display size limit (labels are stored in ORIGINAL pixel coordinates)
CORNER_TEXT = ['1: nut-side, UPPER corner', '2: nut-side, LOWER corner', '3: body-side, LOWER corner', '4: body-side, UPPER corner']


def display_scale(w, h, max_w=MAX_W, max_h=MAX_H):
    return min(1.0, max_w / w, max_h / h)


def to_original(pt_display, scale):
    return [round(pt_display[0] / scale, 1), round(pt_display[1] / scale, 1)]


def load_labels(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else {}


def save_labels(path, labels):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(labels, indent=1), encoding='utf-8')
    tmp.replace(path)                # atomic: a crash never leaves a half-written file


def is_valid_quad(pts):
    # 4 points, non-degenerate, and the polygon must not be self-crossing (a 'bow tie' means the order was wrong)
    p = np.array(pts, float)
    if p.shape != (4, 2):
        return False
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    signs = [np.sign(cross(p[i], p[(i + 1) % 4], p[(i + 2) % 4])) for i in range(4)]
    return abs(sum(signs)) == 4 and cv2.contourArea(p.astype(np.float32)) > 100


def draw(img, pts, name, idx, n, labeled):
    out = img.copy()
    for i, p in enumerate(pts):
        cv2.circle(out, (int(p[0]), int(p[1])), 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (int(p[0]) + 8, int(p[1]) - 8), 0, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    if len(pts) >= 2:
        cv2.polylines(out, [np.array(pts, np.int32)], len(pts) == 4, (0, 165, 255), 2, cv2.LINE_AA)
    nxt = CORNER_TEXT[len(pts)] if len(pts) < 4 else 'ENTER = save | u = undo'
    bar = f'{name}  ({idx + 1}/{n}, {labeled} labelled)   NEXT CLICK -> {nxt}   [s skip, b back, q quit]'
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, bar, (8, 21), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run(folder, out_path, every):
    files = [f for f in sorted(Path(folder).iterdir()) if f.suffix.lower() in EXT][::every]
    if not files:
        sys.exit(f'no images found in {folder}')
    labels = load_labels(out_path)
    todo = [i for i, f in enumerate(files) if f.name not in labels]
    print(f'{len(files)} pictures, {len(files) - len(todo)} already labelled, {len(todo)} to do. Saving to {out_path}')
    state = {'pts': []}
    win = 'label neck corners'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state['pts']) < 4:
            state['pts'].append((x, y))
    cv2.setMouseCallback(win, on_click)

    pos = 0
    while pos < len(todo):
        i = todo[pos]; f = files[i]
        img = cv2.imread(str(f))
        sc = display_scale(img.shape[1], img.shape[0])
        small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else img
        state['pts'] = []
        while True:
            cv2.imshow(win, draw(small, state['pts'], f.name, i, len(files), len(labels)))
            k = cv2.waitKey(30) & 0xFF
            if k in (ord('q'), 27):
                save_labels(out_path, labels); print(f'saved {len(labels)} labels. bye'); return
            if k == ord('u') and state['pts']:
                state['pts'].pop()
            elif k == ord('s'):
                labels[f.name] = None; save_labels(out_path, labels); pos += 1; break
            elif k == ord('b') and pos > 0:
                pos -= 1; labels.pop(files[todo[pos]].name, None); break
            elif k in (13, 32) and len(state['pts']) == 4:
                orig = [to_original(p, sc) for p in state['pts']]
                if not is_valid_quad(orig):
                    print('  the 4 points cross each other or are too small: check the click ORDER (u to undo)'); continue
                labels[f.name] = orig; save_labels(out_path, labels); pos += 1; break
    save_labels(out_path, labels)
    print(f'all done: {len(labels)} labels saved to {out_path}')
    cv2.destroyAllWindows()


def selftest():
    assert display_scale(1280, 720) == 1.0 and abs(display_scale(2800, 1600) - 0.5) < 1e-9
    assert to_original((100, 50), 0.5) == [200.0, 100.0]
    good = [(10, 10), (10, 60), (300, 70), (300, 5)]                  # nut-upper, nut-lower, body-lower, body-upper
    bow = [(10, 10), (300, 70), (10, 60), (300, 5)]                   # clicked in the wrong order
    assert is_valid_quad(good) and not is_valid_quad(bow) and not is_valid_quad([(0, 0), (1, 1), (2, 2)])
    import tempfile
    p = Path(tempfile.mkdtemp()) / 'l.json'; save_labels(p, {'a.jpg': [[1, 2]] * 4, 'b.jpg': None})
    assert load_labels(p) == {'a.jpg': [[1, 2]] * 4, 'b.jpg': None}
    print('selftest OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', nargs='?'); ap.add_argument('--out', default='neck_labels.json'); ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest() if a.selftest else run(a.folder, a.out, a.every) if a.folder else ap.print_help()
