"""Live chord recognition on your own laptop: webcam (or a video file) -> window with neck, hand and chord. Runs on the CPU or a CUDA GPU.

Usage:
    python scripts/live_local.py                          webcam 0 with the models in artifacts/models_local
    python scripts/live_local.py --camera 1               another camera
    python scripts/live_local.py --video some.mp4         a video file instead of a camera
    python scripts/live_local.py --record out.mp4         also save what you see (for a backup demo video)
    python scripts/live_local.py --video x.mp4 --no-window --record out.mp4 --max-seconds 10     headless (for testing)

Keys:  q or ESC = quit | s = strings/frets overlay on/off (slower) | r = reset the smoothing | h = hide/show the help line
Models: copy `neck_v2.pt`, `chord_clf.joblib`, `chord_meta.json` and `hand_landmarker.task` into one folder (the notebook saves the first three in
WORK/models/, and downloads the hand model there too).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_features import NeckFinder, HandFinder                    # noqa: E402
from live_pipeline import Pipeline, draw_overlay, button_rect                         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_pipeline(models_dir):
    d = Path(models_dir)
    need = ['neck_v2.pt', 'chord_clf.joblib', 'chord_meta.json', 'hand_landmarker.task']
    missing = [n for n in need if not (d / n).exists()]
    if missing:
        sys.exit(f'missing in {d}: {missing}\nCopy them from the notebook\'s WORK/models/ folder (see the top of this script).')
    meta = json.loads((d / 'chord_meta.json').read_text(encoding='utf-8'))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    neck, hand = NeckFinder(d / 'neck_v2.pt', dev), HandFinder(d / 'hand_landmarker.task')
    bundle = joblib.load(d / 'chord_clf.joblib'); clf = bundle['model']; novelty = bundle.get('novelty')
    cols = meta['columns']
    thr = meta.get('threshold', 0.0)
    print(f'device: {dev} | chords: {meta["classes"]} | features: {meta["features"]} | model: {meta["model"]} | "not sure" below {thr:.2f} | average of {meta["smooth_k"]} frames')
    return Pipeline(neck, hand, clf, cols, meta['smooth_k'], thr, novelty), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default=str(ROOT / 'artifacts' / 'models_local')); ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--video', default=None); ap.add_argument('--record', default=None)
    ap.add_argument('--proc-width', type=int, default=640, help='frames are shrunk to this width before processing (speed)')
    ap.add_argument('--hide-structure', action='store_true', help='start with the strings/frets overlay off (click the button or press s to show it)')
    ap.add_argument('--no-window', action='store_true'); ap.add_argument('--max-seconds', type=float, default=None)
    a = ap.parse_args()
    pipe, meta = load_pipeline(a.models)
    if a.video:
        cap = cv2.VideoCapture(a.video)
    else:
        cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(a.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit('could not open ' + (a.video or f'camera {a.camera}') + ' (is another program using the camera? try --camera 1)')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    structure, show_help, fps, last, n, t0, writer, answers = not a.hide_structure, True, 0.0, time.time(), 0, time.time(), None, {}
    state = {'structure': structure, 'size': (640, 360)}

    def on_click(event, x, y, flags, param):                                    # the on-screen button toggles strings + frets
        if event == cv2.EVENT_LBUTTONDOWN:
            x0, y0, x1, y1 = button_rect(np.zeros((state['size'][1], state['size'][0], 3), np.uint8))
            if x0 <= x <= x1 and y0 <= y <= y1: state['structure'] = not state['structure']
    if not a.no_window: cv2.namedWindow('chord recognizer'); cv2.setMouseCallback('chord recognizer', on_click)
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if bgr.shape[1] > a.proc_width:
            bgr = cv2.resize(bgr, (a.proc_width, int(bgr.shape[0] * a.proc_width / bgr.shape[1])), interpolation=cv2.INTER_AREA)
        out = pipe.step(bgr); n += 1
        now = time.time(); fps = 0.85 * fps + 0.15 / max(now - last, 1e-3) if n > 1 else 0.0; last = now
        state['size'] = (bgr.shape[1], bgr.shape[0]); structure = state['structure']
        img = draw_overlay(bgr.copy(), out, fps, structure, button=not a.no_window)
        if out['answer']: answers[out['answer']] = answers.get(out['answer'], 0) + 1
        if show_help and not a.no_window:
            cv2.putText(img, 'q quit | s or click button: strings/frets | r reset | h hide help', (10, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if a.record:
            if writer is None: writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*'mp4v'), src_fps if a.video else max(5.0, min(fps or 8.0, 15.0)), (img.shape[1], img.shape[0]))
            writer.write(img)
        if not a.no_window:
            cv2.imshow('chord recognizer', img); k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27): break
            elif k == ord('s'): state['structure'] = not state['structure']
            elif k == ord('r'): pipe.reset()
            elif k == ord('h'): show_help = not show_help
        if a.max_seconds and time.time() - t0 > a.max_seconds: break
    cap.release()
    if writer is not None: writer.release()
    if not a.no_window: cv2.destroyAllWindows()
    top = sorted(answers.items(), key=lambda kv: -kv[1])[:3]
    print(f'{n} frames in {time.time() - t0:.0f}s ({n / max(time.time() - t0, 1e-3):.1f} frames/s) | most frequent answers: {top if top else "none (always not sure / no hand)"}')
    if a.record: print('saved', a.record)


if __name__ == '__main__':
    main()
