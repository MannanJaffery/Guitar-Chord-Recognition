"""Click-to-label tool: mark the nut and the fret lines on pictures of the neck (made by scripts/prepare_fret_labels.py). Stays on your computer.

Usage:
    python scripts/label_frets.py <frames_folder> [--out fret_labels.json]
    python scripts/label_frets.py --selftest

For each picture, click along the MIDDLE of the neck, starting at the NUT and moving toward the body:
    1st click = the nut (where the fretboard meets the headstock; a thin pale bar). 2nd click = fret 1 (the first metal line), 3rd = fret 2, ...
    Click every fret line you can see, in order, without skipping any. If a finger hides a line, click where it would be.
    Stop when the lines get too small or blurry to see.
If the nut is NOT visible (the neck is cut off at the headstock side): press a number key first = the fret number your FIRST click will be
(for example press 2, then click fret 2, fret 3, ...). Default is 0 = the nut.

Keys:  ENTER/SPACE = save and next | u = undo last click | 0-9 = number of the first click (default 0 = nut) | s = skip (cannot tell: saved as skipped)
       b = back one picture | q or ESC = quit (everything saved is kept; run again to continue)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

EXT = {'.jpg', '.jpeg', '.png'}
MAX_W, MAX_H = 1500, 850


def to_video_frame(pts, m):
    """Crop picture coordinates (as clicked at display scale already divided out) -> coordinates in the original video frame."""
    return [[round(x / m['scale'] + m['x0'], 1), round(y / m['scale'] + m['y0'], 1)] for x, y in pts]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else {}


def save(path, labels):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(labels, indent=1), encoding='utf-8'); tmp.replace(path)


def draw(img, pts, first, name, idx, n, done):
    out = img.copy()
    for k, (x, y) in enumerate(pts):
        no = first + k; col = (0, 0, 255) if no == 0 else (0, 255, 255)
        cv2.line(out, (int(x), int(y) - 14), (int(x), int(y) + 14), col, 1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 4, col, -1, cv2.LINE_AA)
        cv2.putText(out, 'nut' if no == 0 else str(no), (int(x) - 8, int(y) - 20), 0, 0.6, col, 2, cv2.LINE_AA)
    nxt = first + len(pts); nxt_txt = 'the NUT' if nxt == 0 else f'FRET {nxt}'
    bar = f'{name} ({idx + 1}/{n}, {done} done)  NEXT CLICK = {nxt_txt}   [ENTER save, u undo, 0-9 first number, s skip, b back, q quit]'
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1); cv2.putText(out, bar, (8, 21), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run(folder, out_path):
    folder = Path(folder); files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in EXT]
    meta = load(folder / 'meta.json'); labels = load(out_path)
    todo = [i for i, f in enumerate(files) if f.name not in labels]
    if not files: sys.exit(f'no pictures in {folder}')
    print(f'{len(files)} pictures, {len(files) - len(todo)} done, {len(todo)} to do. Saving to {out_path}')
    st = {'pts': [], 'first': 0}; win = 'label nut and frets'; cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: st['pts'].append((x, y))
    cv2.setMouseCallback(win, on_click)
    pos = 0
    while pos < len(todo):
        i = todo[pos]; f = files[i]; img = cv2.imread(str(f)); sc = min(1.0, MAX_W / img.shape[1], MAX_H / img.shape[0])
        small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else img
        st['pts'] = []; st['first'] = 0
        while True:
            cv2.imshow(win, draw(small, st['pts'], st['first'], f.name, i, len(files), len(labels)))
            k = cv2.waitKey(30) & 0xFF
            if k in (ord('q'), 27): save(out_path, labels); print(f'saved {len(labels)} labels. bye'); return
            if k == ord('u') and st['pts']: st['pts'].pop()
            elif ord('0') <= k <= ord('9') and not st['pts']: st['first'] = k - ord('0')
            elif k == ord('s'): labels[f.name] = None; save(out_path, labels); pos += 1; break
            elif k == ord('b') and pos > 0: pos -= 1; labels.pop(files[todo[pos]].name, None); break
            elif k in (13, 32):
                if len(st['pts']) < 4: print('  click at least 4 lines (nut/frets) first, or press s to skip'); continue
                xs = [p[0] for p in st['pts']]
                if xs != sorted(xs) and xs != sorted(xs, reverse=True):
                    print('  the clicks must go in ONE direction (nut toward body): u to undo'); continue
                crop_pts = [(x / sc, y / sc) for x, y in st['pts']]               # back to the crop picture's pixels
                rec = dict(first=st['first'], pts_crop=[[round(x, 1), round(y, 1)] for x, y in crop_pts])
                if f.name in meta: rec['pts_video'] = to_video_frame(crop_pts, meta[f.name]); rec.update(meta[f.name])
                labels[f.name] = rec; save(out_path, labels); pos += 1; break
    save(out_path, labels); print(f'all done: {len(labels)} labels in {out_path}'); cv2.destroyAllWindows()


def selftest():
    m = dict(scale=2.0, x0=100, y0=50)
    assert to_video_frame([(200, 100)], m) == [[200.0, 100.0]]
    import tempfile
    p = Path(tempfile.mkdtemp()) / 'l.json'; save(p, {'a.jpg': None}); assert load(p) == {'a.jpg': None}
    print('selftest OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', nargs='?'); ap.add_argument('--out', default=None); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest: selftest()
    elif a.folder: run(a.folder, a.out or str(Path(a.folder).parent / 'fret_labels.json'))
    else: ap.print_help()
