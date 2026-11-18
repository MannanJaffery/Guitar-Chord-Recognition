"""Generates notebooks/03_fingertip_fret_mapping.ipynb (Phase 3: map MediaPipe fingertips onto the Phase 2 fretboard).

Usage: python scripts/make_mapping_notebook.py

- Cells whose first line is '# colab-only' need Google Colab. Everything else runs anywhere (the notebook was tested
  by executing it locally with the colab-only cells skipped).
- The Phase 2 detector is NOT re-implemented here: its function cells are copied verbatim out of
  notebooks/02_fretboard_detection.ipynb at generation time, so Phase 2 stays untouched and cannot drift.
- If configs/phase3_fingertip_labels.json exists it is embedded in the notebook as the manually checked test set.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


# ---------------------------------------------------------------------------------------------------------------
# Pull the Phase 2 detector out of notebook 02 (verbatim, demo tails cut off)
# ---------------------------------------------------------------------------------------------------------------
def phase2_detector_source():
    nb = json.loads((ROOT / "notebooks" / "02_fretboard_detection.ipynb").read_text(encoding="utf-8"))
    srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    # (marker that identifies the cell, text at which the demo tail starts or None)
    wanted = [("CFG = dict(", None), ("def ridge_response", "\nDEMO_INDEX"), ("def seg_lengths", "\nsegs = hough_segments(pre"),
              ("def neck_matrix", "\ncands, (phis, comb_scores)"), ("def string_y", "\n# Demo on the best"),
              ("def merge_intervals", "\n# Demo\n"), ("def _to_original", "\nres_demo =")]
    parts = []
    for marker, cut in wanted:
        hit = [s for s in srcs if marker in s]
        assert hit, f"Phase 2 cell with {marker!r} not found in notebook 02"
        s = hit[0]
        if cut:
            assert cut in s, f"cut marker {cut!r} not found"
            s = s[:s.index(cut)]
        # Phase 2's output folder lines are replaced: Phase 3 writes only into its own folder
        s = s.replace("OUT_DIR = Path(PROJECT_DIR) / 'outputs' / 'fretboard'\nOUT_DIR.mkdir(parents=True, exist_ok=True)\n", "")
        parts.append(s.rstrip() + "\n")
    body = "\n\n".join(parts)
    digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:10]
    return body, digest


LABELS_FILE = ROOT / "configs" / "phase3_fingertip_labels.json"
labels_literal = LABELS_FILE.read_text(encoding="utf-8") if LABELS_FILE.exists() else "{}"
json.loads(labels_literal)  # must be valid JSON

# ---------------------------------------------------------------------------------------------------------------
md(r"""
# Notebook 03: Mapping fingertips onto the fretboard (Phase 3)

**Goal:** for every fingertip MediaPipe finds (Phase 1), say **which guitar string and which fret region it is closest to**, using the fretboard geometry from Phase 2, and say **how sure we are**. When we are not sure we **refuse to answer** instead of guessing.

**Not in this notebook:** chord recognition, any new neural network, or any claim that a finger is *pressing* a string. Everything here is about **2-D proximity in the photo**, which is not the same as physical contact (see section 10).

### What Phase 1 and Phase 2 actually give us (inspected, not assumed)
| | Phase 1 (`01_hand_landmarks_colab.ipynb`) | Phase 2 (`02_fretboard_detection.ipynb`) |
|---|---|---|
| Saves files? | **No.** It only draws and prints. No JSON, no image IDs. | Yes: one JSON per image (+ overlays) in the Colab session, downloaded as `fretboard_results.zip`. Not stored in this repo. |
| Coordinates | Prints fingertip pixels **rounded to whole numbers** and only the 5 fingertips (not all 21 landmarks), with no image size recorded. | **Original-image pixels** with the image's width and height recorded. |
| Identity | None. | `image.name` = `<folder>_<file>`, e.g. `Am_train_00281`. |
| Needed here | Fingertip positions, per image, with an ID | Strings, frets, nut, fretboard polygon, `status`, `confidence` |

**Consequence:** Phase 1's output is not enough to build on, and the existing Phase 2 results are not on disk. We are told not to modify the Phase 1 and 2 notebooks, so this notebook **reproduces both** with exactly their settings and saves its own exports:
- Phase 1 is reproduced with the same pretrained MediaPipe model and the same settings (image mode, up to 2 hands, confidence 0.3), but we keep **all 21 landmarks as unrounded pixels** and the image ID/size.
- Phase 2 is reproduced by **copying its function cells out of notebook 02 at generation time** (section 4), or you can load your own saved `fretboard_results.zip` instead. Either way it is written to a **separate folder** (`outputs/phase3/`), never over existing results.

### The idea in one paragraph
Phase 2 gives us lines: 6 strings, some fret lines, maybe a nut. A fingertip is a point. For a point and a set of lines we can compute the **signed distance** to every line (negative on one side, positive on the other). Counting how many lines the point has passed tells us **which two neighbouring strings it sits between** and **which two neighbouring frets**. Because we use the *actual* detected lines (not a rotated, idealised grid), strings that fan out because of perspective are handled automatically. The ratio of the two distances tells us **how close to a boundary** the point is, which is our ambiguity measure.

### Cells
0. Setup, 1. Config and load images, 2. Phase 1 reproduction (MediaPipe), 3. Phase 2 geometry, 4. Detector code from notebook 02, 5. Geometry model, 6. Fingertip mapping + confidence, 7. Run on all images + save JSON, 8. Visualization, 9. Evaluation (+ manually checked test set), 10. Limitations and Phase 4 hand-off
""")

md(r"""
## 0. Setup
**What:** creates the working folders and makes sure MediaPipe is installed. No Google Drive needed: you upload `guitar_chords_sample.zip` once (section 1) and download a results zip at the end.
**Output layout** (all new, nothing from Phases 1-2 is overwritten):
```
outputs/phase3/hand_landmarks/<image_id>.json      Phase 1 export (all 21 landmarks, float pixels)
outputs/phase3/fretboard_recomputed/<image_id>.json  Phase 2 geometry reproduced here (if not loaded from your zip)
outputs/phase3/mapping/<image_id>.json             the Phase 3 result: one record per fingertip
outputs/phase3/overlays/ , review/                  pictures
outputs/phase3/summary.json
```
""")
code(r"""
# colab-only
import os
PROJECT_DIR = '/content/Guitar-Chord-Recognition'      # temporary folder on the Colab machine
for sub in ['data/sample_images']:
    os.makedirs(f'{PROJECT_DIR}/{sub}', exist_ok=True)
print('Project folder:', PROJECT_DIR)
""")
code(r"""
import os, sys, subprocess, importlib
try:
    import mediapipe as mp
except ImportError:                                           # Colab usually needs this once per session
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'mediapipe'], check=True)
    importlib.invalidate_caches()
    import mediapipe as mp
print('mediapipe', mp.__version__)
""")

md(r"""
## 1. Config and load the original images
**What:** every setting that changes behaviour is in two dictionaries so you can tune them in one place, then the original images are loaded exactly like in notebook 02 so that the **image IDs are identical** (`<folder>_<file>`).
**Why IDs matter:** the only way to join Phase 1 landmarks, Phase 2 geometry and Phase 3 results (and later Phase 4) is by a stable ID plus the image's original width and height. Every coordinate in this notebook is in **original-image pixels**, never in a resized copy.

`MAPCFG` explained:
- `string_margin_min` / `fret_margin_min`: how clearly a fingertip must sit on one string / inside one fret region before we accept it (explained in section 6).
- `outside_frac`: a point farther than this fraction of a string gap beyond the first or last string is "off the neck".
- `accept_conf`: minimum overall confidence for an accepted mapping.
- `allow_low_geometry`: Phase 2 marks many results `low` confidence. By default a mapping on low-confidence geometry is only **tentative**, never accepted.
""")
code(r"""
import json, glob, random, zipfile, urllib.request
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAPCFG = dict(
    string_margin_min=0.30,   # 0 = exactly halfway between two strings, 1 = exactly on one string
    fret_margin_min=0.12,     # fraction of the fret-region width that the tip must stay away from a fret line
    outside_frac=0.6,         # beyond the outer strings by more than this x a string gap = not on the neck
    near_neck_frac=0.5,       # when choosing the fretting hand, tips this close (x string gap) to the neck count as near
    accept_conf=0.35,         # minimum overall confidence for 'accepted'
    geometry_quality={'high': 1.0, 'low': 0.6},   # how much we trust Phase 2's own confidence label
    allow_low_geometry=False, # False: low-confidence Phase 2 geometry gives 'tentative' results only
    fret_ratio=2 ** (-1 / 12),# real fret spacing shrinks by this factor per fret (12-tone equal temperament)
    wide_region_ratio=1.8,    # a fret region this many times wider than its neighbours hides a missed fret line
    narrow_region_ratio=0.45, # ... and this many times narrower suggests a spurious line
)

IMAGE_SOURCE = 'zip'       # 'zip' | 'upload'
MAX_IMAGES = 60            # how many images to process; None = all 280 (Phase 2 takes ~1.7 s per image on a CPU)
SEED = 7
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

ZIP_PATH = f'{PROJECT_DIR}/data/guitar_chords_sample.zip'
EXTRACT_DIR = Path(PROJECT_DIR) / 'data' / 'sample_images'
PH3 = Path(PROJECT_DIR) / 'outputs' / 'phase3'
for sub in ['hand_landmarks', 'fretboard_recomputed', 'mapping', 'overlays', 'review']:
    (PH3 / sub).mkdir(parents=True, exist_ok=True)

IMAGES = []   # list of (image_id, bgr_image)

if IMAGE_SOURCE == 'zip':
    if not os.path.exists(ZIP_PATH):                      # skip the upload if the zip is already on the machine
        from google.colab import files
        print('Choose guitar_chords_sample.zip in the file picker below...')
        uploaded = files.upload()
        zips = [n for n in uploaded if n.lower().endswith('.zip')]
        assert zips, 'Please choose guitar_chords_sample.zip'
        Path(ZIP_PATH).write_bytes(uploaded[zips[0]])
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_DIR)
    for f in sorted(EXTRACT_DIR.rglob('*')):
        if f.suffix.lower() in IMG_EXT and '__MACOSX' not in f.parts and not f.name.startswith('.'):
            bgr = cv2.imread(str(f))                       # BGR array, shape (height, width, 3), ORIGINAL size
            if bgr is not None:
                parent = f.parent.name if f.parent != EXTRACT_DIR else ''
                IMAGES.append((f'{parent}_{f.stem}' if parent else f.stem, bgr))
elif IMAGE_SOURCE == 'upload':
    from google.colab import files
    for fname, data in files.upload().items():
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            IMAGES.append((Path(fname).stem, bgr))

random.Random(SEED).shuffle(IMAGES)                         # same seed as notebook 02
if MAX_IMAGES:
    IMAGES = IMAGES[:MAX_IMAGES]
assert IMAGES, 'No images were loaded.'
assert len({i for i, _ in IMAGES}) == len(IMAGES), 'image IDs must be unique'
IMG = dict(IMAGES)                                          # id -> original BGR image
print(f'{len(IMAGES)} images loaded; sizes:', sorted({(b.shape[1], b.shape[0]) for _, b in IMAGES}))
""")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 2. Phase 1 reproduction: MediaPipe fingertips, exported properly
**What:** MediaPipe's **Hand Landmarker** finds up to 2 hands and returns **21 landmarks per hand** (wrist, knuckles, joints, fingertips). The fingertips are landmarks 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky).
**How it works (short version):** a small pretrained neural network first finds the **palm** in the image, then a second network **regresses** the 21 landmark positions inside the cropped hand. Nothing is trained here; we only run it.
**Coordinates:** MediaPipe returns `x` and `y` as fractions of the image width and height (0 to 1). We multiply by the **original** width and height to get pixels. Phase 1 rounded these to integers; we keep the decimals because a 1-pixel error matters when strings are 10-20 pixels apart.
**`z`:** relative depth with the wrist as zero. Useful for hand shape, not for deciding contact with a string.
**Handedness:** MediaPipe's `Left`/`Right` label assumes a mirrored (selfie) image, so for these photos it may be the opposite of the real hand. We therefore do **not** use it; we pick the fretting hand by where its fingertips are (section 6).
**Same settings as Phase 1:** image mode, `num_hands=2`, all confidences 0.3 (the dataset is heavily augmented, so a low threshold is needed).
In the dataset about **43% of images get no hand at all** at this threshold; those are reported as `no_hand`, not guessed.
""")
code(r"""
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
MODEL_PATH = f'{PROJECT_DIR}/hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)       # ~8 MB pretrained model from Google

FINGERTIPS = {4: 'Thumb', 8: 'Index', 12: 'Middle', 16: 'Ring', 20: 'Pinky'}
FINGER_COLORS = {'Thumb': (0, 165, 255), 'Index': (0, 255, 0), 'Middle': (255, 128, 0), 'Ring': (255, 0, 255), 'Pinky': (0, 255, 255)}   # BGR, as in Phase 1
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
               (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

_opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE, num_hands=2,
    min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3)
LANDMARKER = vision.HandLandmarker.create_from_options(_opts)


def detect_hands(bgr, image_id):
    # Returns the Phase 1 export for one image. All coordinates are ORIGINAL-image pixels (floats).
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)               # MediaPipe wants RGB, OpenCV gives BGR
    res = LANDMARKER.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    hands = []
    for i, lms in enumerate(res.hand_landmarks):
        cat = res.handedness[i][0]
        pts = [[round(lm.x * w, 2), round(lm.y * h, 2), round(lm.z, 4)] for lm in lms]   # normalized -> pixels
        tips = {name: dict(id=idx, x_px=pts[idx][0], y_px=pts[idx][1], z=pts[idx][2]) for idx, name in FINGERTIPS.items()}
        hands.append(dict(handedness_as_model_sees_it=cat.category_name, score=round(float(cat.score), 3),
                          landmarks_px=pts, tips=tips))
    return dict(schema_version=1, image=dict(id=image_id, width=w, height=h), hands=hands)


HANDS = {}
for image_id, bgr in IMAGES:
    HANDS[image_id] = detect_hands(bgr, image_id)
    (PH3 / 'hand_landmarks' / f'{image_id}.json').write_text(json.dumps(HANDS[image_id], indent=1), encoding='utf-8')

n_hands = pd.Series([len(v['hands']) for v in HANDS.values()]).value_counts().sort_index()
print('hands found per image:', n_hands.to_dict(), '| images with no hand:', int((n_hands.get(0, 0))), 'of', len(HANDS))
print(f'saved {len(HANDS)} Phase 1 exports to', PH3 / 'hand_landmarks')
""")

md(r"""
## 3. Phase 2 geometry: load your results or reproduce them
**What:** a function `get_phase2(image_id, bgr)` returns the Phase 2 result dictionary for one image. Choose the source with `PHASE2_SOURCE`:
- `'recompute'` (default): runs the Phase 2 detector (section 4) on each original image. Needs nothing but the images. The results are written to `outputs/phase3/fretboard_recomputed/`.
- `'zip'`: pick the `fretboard_results.zip` you downloaded from notebook 02. Results are read by ID; any image missing from the zip is recomputed.

**Coordinate check (requirement 4):** whatever the source, we compare the width and height recorded inside the Phase 2 JSON with the real size of the original image, and the Phase 1 export's size too. A mismatch (for example Phase 2 run on a resized copy) is reported loudly instead of silently producing shifted coordinates.
""")
code(r"""
PHASE2_SOURCE = 'recompute'      # 'recompute' | 'zip'
P2_LOADED = {}                   # image_id -> Phase 2 result loaded from your zip

if PHASE2_SOURCE == 'zip':
    from google.colab import files
    print('Choose fretboard_results.zip (downloaded from notebook 02)...')
    up = files.upload()
    zname = [n for n in up if n.lower().endswith('.zip')][0]
    ldir = PH3 / 'phase2_loaded'; ldir.mkdir(exist_ok=True)
    zp = ldir / 'fretboard_results.zip'; zp.write_bytes(up[zname])
    with zipfile.ZipFile(zp) as z:
        z.extractall(ldir)
    for jf in ldir.rglob('*.json'):
        if jf.name == 'summary.json':
            continue
        d = json.loads(jf.read_text(encoding='utf-8'))
        if 'image' in d and 'strings' in d:
            P2_LOADED[d['image']['name']] = d
    print(f'loaded {len(P2_LOADED)} Phase 2 results from your zip')
""")

md(r"""
## 4. The Phase 2 detector (copied unchanged from notebook 02)
**What:** this single cell is the function code of `02_fretboard_detection.ipynb` (config, ridge filter, Hough segments, neck-direction comb search, string fitting, fret and nut finding, `detect_fretboard`), copied **verbatim by the generator script**, demo cells removed. Notebook 02 itself is untouched.
**Why copy instead of import:** Colab notebooks cannot import each other, and re-writing the detector here would let the two versions drift apart. The fingerprint in the first comment line changes whenever the Phase 2 code changes, so you can see if this copy is stale (re-run `python scripts/make_mapping_notebook.py` to refresh it).
**What it returns:** a dictionary with `status` (`ok`/`partial`/`failed`), `confidence` (`high`/`low`/`none`), `strings` (6 lines), `frets` (lines), `nut` and `fretboard.polygon`, all in original pixels. Strings are numbered 1-6 by their position in the neck-aligned view, **not** by guitar tuning.
""")
_p2src, _p2digest = phase2_detector_source()
code(f"# Phase 2 detector: copied from notebooks/02_fretboard_detection.ipynb (fingerprint {_p2digest}). Do not edit here.\n" + _p2src)
code(r"""
def get_phase2(image_id, bgr):
    # Phase 2 result for one image: from your zip if available, otherwise recomputed (and saved to our own folder).
    if image_id in P2_LOADED:
        return P2_LOADED[image_id], 'zip'
    res = detect_fretboard(bgr, image_id)
    (PH3 / 'fretboard_recomputed' / f'{image_id}.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    return res, 'recompute'


def check_coordinates(image_id, bgr, hands, p2):
    # Requirement 4: all three sources must describe the same pixel grid. Returns a list of problems (empty = fine).
    h, w = bgr.shape[:2]
    problems = []
    if (hands['image']['width'], hands['image']['height']) != (w, h):
        problems.append('Phase 1 export size differs from the image')
    if (p2['image']['width'], p2['image']['height']) != (w, h):
        problems.append(f"Phase 2 size {p2['image']['width']}x{p2['image']['height']} differs from the image {w}x{h}")
    if p2.get('image', {}).get('name') != image_id:
        problems.append('Phase 2 image name differs from the image ID')
    if p2.get('fretboard'):
        poly = np.array(p2['fretboard']['polygon'])
        if poly.min() < -2 * max(w, h) or poly.max() > 3 * max(w, h):
            problems.append('Phase 2 polygon is wildly outside the image (wrong coordinate frame?)')
    return problems


PHASE2, P2_SRC, COORD_PROBLEMS = {}, {}, {}
for image_id, bgr in IMAGES:
    PHASE2[image_id], P2_SRC[image_id] = get_phase2(image_id, bgr)
    COORD_PROBLEMS[image_id] = check_coordinates(image_id, bgr, HANDS[image_id], PHASE2[image_id])
bad = {k: v for k, v in COORD_PROBLEMS.items() if v}
print('coordinate check:', 'all', len(IMAGES), 'images consistent' if not bad else f'{len(bad)} PROBLEMS')
for k, v in list(bad.items())[:10]:
    print('  ', k, v)
print('Phase 2 source:', pd.Series(P2_SRC).value_counts().to_dict(),
      '| status:', pd.Series([r['status'] for r in PHASE2.values()]).value_counts().to_dict(),
      '| confidence:', pd.Series([r['confidence'] for r in PHASE2.values()]).value_counts().to_dict())
""")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 5. The geometry model: lines, signed distance, and handling perspective
**Technique 1: signed point-to-line distance.** A line through `p1` and `p2` has a unit **normal** `n` (a vector at right angles to it). For a point `P`, `(P - p1) · n` is the perpendicular distance from `P` to the line, **positive on one side and negative on the other**. The sign tells us which side of the line the point is on.

**Technique 2: bracketing.** Sort the strings from 1 to 6. Count how many of them the point is "past" (positive distance). If it is past 3 strings, it lies between string 3 and string 4. The same trick with the fret lines gives the fret region. This is the same idea as finding which cell of a grid a point is in, but the grid lines are the **real detected lines**, so they can fan out or tilt: **perspective and non-parallel strings need no special case**, because we never rotate everything into a perfect grid.

**Technique 3: ambiguity as a distance ratio.** If `d1` and `d2` are the distances to the nearest and second-nearest line, then `margin = (d2 - d1) / (d2 + d1)` is 1 when the point is exactly on the nearest line and 0 when it is exactly halfway. A small margin means "could be either", so we refuse to choose. This ratio idea is the same as the ratio test used to reject ambiguous feature matches.

**Technique 4: fret spacing prior (to repair missed frets).** Frets are closer together toward the body: each gap is about `2^(-1/12)` (0.944) times the previous one. If a detected gap is much bigger than this predicts, a fret line was probably hidden by a finger. We insert **virtual** boundaries there (clearly marked), because otherwise a single missed fret would shift every fret number after it. Virtual boundaries never raise confidence.

**Which end is the nut?** If Phase 2 reports a nut we measure from it. Otherwise the end with the wider fret gaps is the headstock end (Phase 2 uses the same rule). If neither is reliable the fret *number* is unknown and we only report the **slot** (1st, 2nd, ... region counted from the first detected line).
**`fret_slot` vs `fret_number`:** `fret_slot` is always "which region, counted from the headstock-side line we found". `fret_number` equals the slot only when a nut with `medium` confidence anchors it; otherwise it is `None`, because in a close-up the "nut" may really be fret 1 or 3.
""")
code(r"""
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _mid(line):
    return (line['p1'] + line['p2']) / 2


def signed_dist(line, P, ref_dir):
    # Perpendicular distance from P to the (infinite) line through p1,p2, positive on the side ref_dir points to.
    d = line['p2'] - line['p1']
    n = _unit([-d[1], d[0]])                      # a normal vector (rotate the direction by 90 degrees)
    if n @ ref_dir < 0:
        n = -n
    return float((P - line['p1']) @ n)


def build_neck_model(p2):
    # Turn a Phase 2 result into the objects the mapping needs. Returns None if there are no usable strings.
    if p2.get('status') == 'failed' or len(p2.get('strings', [])) != 6 or not p2.get('fretboard'):
        return None
    S = [dict(p1=np.array(s['p1'], float), p2=np.array(s['p2'], float)) for s in sorted(p2['strings'], key=lambda s: s['index'])]
    cen = [_mid(s) for s in S]
    c_dir = _unit(cen[-1] - cen[0])                                  # across the neck: string 1 -> string 6
    spacing = float(np.mean(np.diff([(c - cen[0]) @ c_dir for c in cen])))
    if spacing <= 1:
        return None
    a_rough = _unit(np.mean([s['p2'] - s['p1'] for s in S], axis=0))  # along the neck (left end -> right end of Phase 2's view)

    lines = [dict(p1=np.array(f['p1'], float), p2=np.array(f['p2'], float), kind='fret', virtual=False) for f in p2.get('frets', [])]
    nut_conf = None
    if p2.get('nut'):
        lines.append(dict(p1=np.array(p2['nut']['p1'], float), p2=np.array(p2['nut']['p2'], float), kind='nut', virtual=False,
                          conf=p2['nut'].get('confidence')))
    model = dict(strings=S, c_dir=c_dir, spacing=spacing, polygon=np.array(p2['fretboard']['polygon'], float),
                 boundaries=[], b_dir=None, orientation='none', nut_conf=None, n_virtual=0)
    if len(lines) < 2:
        return model                                                  # strings only: we can name the string but not the fret

    lines.sort(key=lambda l: _mid(l) @ a_rough)                       # left to right along the neck
    # Decide which end is the headstock end. Prefer the nut; otherwise the end with wider gaps.
    nut_idx = [i for i, l in enumerate(lines) if l['kind'] == 'nut']
    pos = [_mid(l) @ a_rough for l in lines]
    gaps = np.diff(pos)
    orientation = 'unknown'
    if nut_idx and nut_idx[0] in (0, len(lines) - 1):
        if nut_idx[0] == len(lines) - 1:
            lines = lines[::-1]
        orientation = 'nut'
        model['nut_conf'] = lines[0].get('conf')
    elif len(lines) >= 3:
        if gaps[0] < gaps[-1]:
            lines = lines[::-1]
        orientation = 'gap_trend'
    # Repair missed frets with the 12-tone spacing rule (gaps should shrink by ~0.944 per fret).
    ordered, unit_gap = [lines[0]], None
    r12 = MAPCFG['fret_ratio']
    for j in range(1, len(lines)):
        g = float(np.linalg.norm(_mid(lines[j]) - _mid(lines[j - 1])))
        if j == 1 and orientation != 'unknown' and len(lines) >= 3:
            # The FIRST gap has no earlier gap to compare with, so compare it with the NEXT one. If a gap holds mm+1 real
            # frets, its width relative to the next region is (1 + r + ... + r^mm) / r^(mm+1): 1.06, 2.2, 3.4, ...
            g2 = float(np.linalg.norm(_mid(lines[2]) - _mid(lines[1])))
            R = g / max(g2, 1e-6)
            expect = [sum(r12 ** i for i in range(mm + 1)) / r12 ** (mm + 1) for mm in range(5)]
            mm = int(np.argmin([abs(np.log(max(R, 1e-6) / e)) for e in expect]))
            if mm >= 1 and R > 1.5:
                tot = sum(r12 ** i for i in range(mm + 1))
                for t in np.cumsum([r12 ** i for i in range(mm)]) / tot:     # missed frets sit at the 12-tone proportions
                    ordered.append(dict(p1=lines[0]['p1'] * (1 - t) + lines[1]['p1'] * t,
                                        p2=lines[0]['p2'] * (1 - t) + lines[1]['p2'] * t, kind='fret', virtual=True))
                    model['n_virtual'] += 1
                ordered.append(lines[1])
                unit_gap = g * r12 ** mm / tot
                continue
        m = 1
        if unit_gap is not None and orientation != 'unknown':
            expected = unit_gap * MAPCFG['fret_ratio']
            m = int(np.clip(round(g / expected), 1, 3))
            if m < 2 or g < 1.5 * expected:
                m = 1
        for i in range(1, m):                                         # virtual boundaries, evenly spread inside the gap
            t = i / m
            ordered.append(dict(p1=lines[j - 1]['p1'] * (1 - t) + lines[j]['p1'] * t,
                                p2=lines[j - 1]['p2'] * (1 - t) + lines[j]['p2'] * t, kind='fret', virtual=True))
            model['n_virtual'] += 1
        ordered.append(lines[j])
        unit_gap = g / m
    model.update(boundaries=ordered, orientation=orientation,
                 b_dir=_unit(_mid(ordered[-1]) - _mid(ordered[0])))
    # Irregular spacing check: a real fret region is only ~6% narrower than the one before it. A region much wider than
    # its neighbours hides a missed fret (or two); a much narrower one means a spurious line. We cannot name the fret there.
    widths = [float(np.linalg.norm(_mid(ordered[k]) - _mid(ordered[k - 1]))) for k in range(1, len(ordered))]   # slot k = widths[k-1]
    suspect = {}
    for k in range(1, len(ordered)):
        if ordered[k]['virtual'] or ordered[k - 1]['virtual']:
            continue
        nb = [widths[j - 1] for j in (k - 1, k + 1) if 1 <= j <= len(widths)]
        if not nb:
            continue
        ratio = widths[k - 1] / max(float(np.mean(nb)), 1e-6)
        if ratio > MAPCFG['wide_region_ratio']:
            suspect[k] = f'region {ratio:.1f}x wider than its neighbours: a fret line was probably missed'
        elif ratio < MAPCFG['narrow_region_ratio']:
            suspect[k] = f'region only {ratio:.2f}x as wide as its neighbours: a spurious fret line is likely'
    model['region_suspect'] = suspect
    return model


def locate_string(model, P):
    # Which string is P closest to, and how clearly? Uses the real (possibly fanned-out) string lines.
    S, c_dir = model['strings'], model['c_dir']
    sd = [signed_dist(s, P, c_dir) for s in S]                       # positive = on the string-6 side of that string
    crossing = any(sd[i] <= sd[i + 1] for i in range(5))              # strings must stay in order at this point
    m = sum(d > 0 for d in sd)                                        # number of strings P is past
    if m == 0:                                                        # above string 1
        d1, gap = -sd[0], sd[0] - sd[1]
        near, d2 = 1, d1 + max(gap, 1e-6)
        outside = d1 > MAPCFG['outside_frac'] * max(gap, 1e-6)
        pos = 1 - d1 / max(gap, 1e-6)
    elif m == 6:                                                      # below string 6
        d1, gap = sd[5], sd[4] - sd[5]
        near, d2 = 6, d1 + max(gap, 1e-6)
        outside = d1 > MAPCFG['outside_frac'] * max(gap, 1e-6)
        pos = 6 + d1 / max(gap, 1e-6)
    else:                                                             # between string m and string m+1
        da, db = sd[m - 1], -sd[m]
        near = m if da <= db else m + 1
        d1, d2, gap = min(da, db), max(da, db), da + db
        outside = False
        pos = m + da / max(da + db, 1e-6)
    margin = (d2 - d1) / max(d2 + d1, 1e-6)
    return dict(index=int(near), position=float(pos), margin=float(margin), dist_px=float(d1), outside=bool(outside),
                crossing=bool(crossing), gap_px=float(gap))


def locate_fret(model, P):
    # Which fret region is P in (between which two boundaries), and how far from the nearer boundary?
    B, b_dir = model['boundaries'], model['b_dir']
    sd = [signed_dist(b, P, b_dir) for b in B]                       # positive = on the body side of that line
    k = sum(d > 0 for d in sd)                                        # number of boundaries P is past
    if k == 0:
        return dict(region='before_first_line', slot=None, margin=0.0, virtual_edge=False)
    if k == len(B):
        return dict(region='after_last_line', slot=None, margin=0.0, virtual_edge=False)
    dl, dr = sd[k - 1], -sd[k]                                        # distances to the boundary behind and ahead of P
    width = max(dl + dr, 1e-6)
    return dict(region='inside', slot=int(k), margin=float(min(dl, dr) / width), frac=float(dl / width),
                virtual_edge=bool(B[k - 1]['virtual'] or B[k]['virtual']), suspect=model['region_suspect'].get(int(k)))


def snap_to_string(model, P, index):
    # Foot of the perpendicular from P onto string `index`: the "mapped position" we draw.
    s = model['strings'][index - 1]
    d = _unit(s['p2'] - s['p1'])
    return (s['p1'] + ((P - s['p1']) @ d) * d)
""")

md(r"""
## 6. Mapping one fingertip, with confidence and refusal
**Choosing the fretting hand.** A photo can contain two hands (strumming hand and fretting hand). We pick the hand whose index, middle, ring and pinky tips are most often **inside or just beside the detected fretboard polygon** (`cv2.pointPolygonTest` returns a point's signed distance to a polygon, positive inside). The other hand is ignored and counted.

**Per-fingertip decision.** For each fingertip we compute the string result and the fret result (section 5) and then assign **one status**:

| status | meaning |
|---|---|
| `mapped` | clearly on one string **and** clearly inside one fret region |
| `string_only` | clearly on one string, but Phase 2 found too few fret lines to name the fret |
| `ambiguous_string` / `ambiguous_fret` / `ambiguous_both` | too close to the boundary between two strings / two frets, **or** the fret region has irregular width (a fret line was probably missed or invented), so we **refuse** |
| `outside_fretboard` | beyond the outer strings, or behind a **confirmed** nut (headstock side) |
| `beyond_detected_frets` | before the first or past the last detected fret line when the nut is not confirmed: more fretboard may lie there, but we do not know the region width, so we cannot name the fret |
| `no_geometry` | Phase 2 failed, so there is nothing to map onto |

**Confidence** (0 to 1) is a product of independent evidence, so any weak link lowers it:
`confidence = geometry_quality x hand_score x sqrt(string_clarity x fret_clarity)`
- `geometry_quality`: 1.0 if Phase 2 said `high`, 0.6 if `low` (`MAPCFG`).
- `hand_score`: MediaPipe's own handedness score for that hand.
- `string_clarity` = the string margin from section 5; `fret_clarity` = `2 x` the fret margin (so a point exactly mid-region scores 1). Virtual (inferred) fret boundaries halve the fret clarity.

**Decision** (what Phase 4 should use):
- `accepted`: status is `mapped`/`string_only`, Phase 2 geometry is `high`, confidence is at least `accept_conf`, and it is not the thumb.
- `tentative`: the mapping is geometrically valid but rests on `low`-confidence Phase 2 geometry (or is below `accept_conf`). Shown, but not trusted.
- `rejected`: everything else, with a `reason`.
The **thumb** is never accepted: it normally rests behind the neck, so a thumb tip near a string says nothing about fretting.

**Important:** every record carries `proximity_only: true`. The fingertip landmark is where the fingertip *appears* in a 2-D photo. The finger sits above the strings, the camera is rarely square-on, and the landmark is the tip, not the pad that touches the string. So a high-confidence record means *"this point is clearly closest to string s in fret region f in the picture"*, **not** *"the finger presses string s"*.
""")
code(r"""
def select_fretting_hand(hands, model):
    # Pick the hand most likely to be the fretting hand. Returns (index or None, number of other hands).
    if not hands['hands']:
        return None, 0
    best, best_key = 0, None
    for i, hd in enumerate(hands['hands']):
        near = 0
        if model is not None:
            poly = model['polygon'].astype(np.float32)
            for name in ('Index', 'Middle', 'Ring', 'Pinky'):
                t = hd['tips'][name]
                # signed distance to the polygon: > 0 inside, < 0 outside (in pixels)
                dist = cv2.pointPolygonTest(poly, (float(t['x_px']), float(t['y_px'])), True)
                near += dist >= -MAPCFG['near_neck_frac'] * model['spacing']
        key = (near, hd['score'])
        if best_key is None or key > best_key:
            best, best_key = i, key
    return best, len(hands['hands']) - 1


def map_fingertip(model, p2, finger, tip, hand_score):
    P = np.array([tip['x_px'], tip['y_px']], float)
    r = dict(finger=finger, landmark_id=tip['id'], x_px=tip['x_px'], y_px=tip['y_px'], proximity_only=True,
             status='no_geometry', decision='rejected', reason='', string_index=None, string_position=None,
             string_margin=None, fret_slot=None, fret_number=None, fret_margin=None, confidence=0.0, snap_xy=None)
    if model is None:
        r['reason'] = f"Phase 2 found no usable strings ({p2.get('failure_reason')})"
        return r
    s = locate_string(model, P)
    r.update(string_index=s['index'], string_position=round(s['position'], 2), string_margin=round(s['margin'], 2))
    if s['crossing']:
        r.update(status='ambiguous_string', reason='detected strings cross at this point; geometry unreliable here')
        return r
    if s['outside']:
        r.update(status='outside_fretboard', reason='beyond the outer string', string_index=None)
        return r
    f = locate_fret(model, P) if model['boundaries'] else None
    if f is not None and f['region'] == 'before_first_line':
        if model['nut_conf'] == 'medium':                              # a believable nut: beyond it is the headstock
            r.update(status='outside_fretboard', reason='behind the nut (headstock side)')
        else:                                                          # nut not confirmed: more fretboard may lie beyond
            r.update(status='beyond_detected_frets', reason='before the first detected line and the nut is not confirmed; region unknown')
        return r
    if f is not None and f['region'] == 'after_last_line':
        r.update(status='beyond_detected_frets', reason='past the last detected fret line; region width unknown')
        return r
    s_ok = s['margin'] >= MAPCFG['string_margin_min']
    if f is None:
        status = 'string_only' if s_ok else 'ambiguous_string'
        f_clear = 1.0
    else:
        f_ok = f['margin'] >= MAPCFG['fret_margin_min'] and not f['suspect']
        status = 'mapped' if (s_ok and f_ok) else 'ambiguous_both' if (not s_ok and not f_ok) else 'ambiguous_string' if not s_ok else 'ambiguous_fret'
        f_clear = float(np.clip(2 * f['margin'], 0, 1)) * (0.5 if f['virtual_edge'] else 1.0)
        r.update(fret_slot=f['slot'], fret_margin=round(f['margin'], 2),
                 fret_number=f['slot'] if model['nut_conf'] == 'medium' else None)
    gq = MAPCFG['geometry_quality'].get(p2.get('confidence'), 0.3)
    conf = gq * hand_score * float(np.sqrt(np.clip(s['margin'], 0, 1) * f_clear))
    r.update(status=status, confidence=round(float(conf), 3), snap_xy=[round(float(v), 1) for v in snap_to_string(model, P, s['index'])])
    if status.startswith('ambiguous'):
        r['reason'] = 'too close to the boundary between two ' + ('strings' if status == 'ambiguous_string' else 'fret regions' if status == 'ambiguous_fret' else 'strings and fret lines')
        if f is not None and f['suspect'] and status in ('ambiguous_fret', 'ambiguous_both'):
            r['reason'] = f"irregular fret spacing here ({f['suspect']})"
    elif finger == 'Thumb':
        r['reason'] = 'thumb usually rests behind the neck'
    elif p2.get('confidence') != 'high' and not MAPCFG['allow_low_geometry']:
        r.update(decision='tentative', reason='Phase 2 geometry is low confidence')
    elif conf < MAPCFG['accept_conf']:
        r.update(decision='tentative', reason=f"confidence {conf:.2f} below {MAPCFG['accept_conf']}")
    else:
        r.update(decision='accepted', reason='')
    return r


def map_image(image_id, bgr):
    hands, p2 = HANDS[image_id], PHASE2[image_id]
    model = build_neck_model(p2)
    h_idx, n_other = select_fretting_hand(hands, model)
    out = dict(schema_version=1, image=dict(id=image_id, width=bgr.shape[1], height=bgr.shape[0]),
               phase1=dict(n_hands=len(hands['hands']), fretting_hand_index=h_idx, n_other_hands=n_other),
               phase2=dict(source=P2_SRC[image_id], status=p2['status'], confidence=p2['confidence'], failure_reason=p2.get('failure_reason'),
                           neck_angle_deg=p2.get('neck_angle_deg')),
               geometry=dict(usable=model is not None, n_fret_boundaries=0 if not model else len(model['boundaries']),
                             n_virtual_boundaries=0 if not model else model['n_virtual'],
                             orientation='none' if not model else model['orientation'],
                             nut_confidence=None if not model else model['nut_conf'],
                             string_index_convention='1..6 in Phase 2 order (one side of the neck to the other); NOT guitar tuning',
                             fret_slot_convention='1 = region after the first detected headstock-side line; fret_number only when a medium-confidence nut anchors it'),
               coordinate_problems=COORD_PROBLEMS[image_id], fingertips=[], category='')
    if h_idx is None:
        out['category'] = 'no_hand'
        return out, model
    hd = hands['hands'][h_idx]
    for name in FINGERTIPS.values():
        r = map_fingertip(model, p2, name, hd['tips'][name], hd['score'])
        r['hand_score'] = hd['score']
        out['fingertips'].append(r)
    dec = [r['decision'] for r in out['fingertips']]
    stat = [r['status'] for r in out['fingertips']]
    if model is None:
        out['category'] = 'geometry_failed'
    elif 'accepted' in dec:
        out['category'] = 'success'
    elif 'tentative' in dec:
        out['category'] = 'tentative_low_geometry'
    elif any(s.startswith('ambiguous') for s in stat):
        out['category'] = 'ambiguous'
    else:
        out['category'] = 'outside_or_rejected'
    return out, model
""")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 7. Run everything and save one JSON per image
**What:** maps every loaded image, writes `outputs/phase3/mapping/<image_id>.json` (the Phase 4 input) and a `summary.json`, then prints a table.
**Image categories:** `success` (at least one accepted fingertip), `tentative_low_geometry`, `ambiguous`, `outside_or_rejected`, `geometry_failed` (Phase 2 found no strings), `no_hand` (MediaPipe found no hand).
""")
code(r"""
RESULTS, MODELS = {}, {}
for image_id, bgr in IMAGES:
    RESULTS[image_id], MODELS[image_id] = map_image(image_id, bgr)
    (PH3 / 'mapping' / f'{image_id}.json').write_text(json.dumps(RESULTS[image_id], indent=1), encoding='utf-8')

rows = []
for image_id, r in RESULTS.items():
    for t in r['fingertips']:
        rows.append(dict(image=image_id, finger=t['finger'], status=t['status'], decision=t['decision'], string=t['string_index'],
                         slot=t['fret_slot'], conf=t['confidence'], s_margin=t['string_margin'], f_margin=t['fret_margin'], reason=t['reason']))
TIPS = pd.DataFrame(rows)
CATS = pd.Series({k: v['category'] for k, v in RESULTS.items()})
summary = dict(n_images=len(RESULTS), image_categories=CATS.value_counts().to_dict(),
               fingertip_decisions=TIPS['decision'].value_counts().to_dict() if len(TIPS) else {},
               fingertip_status=TIPS['status'].value_counts().to_dict() if len(TIPS) else {},
               phase2_confidence=pd.Series([v['phase2']['confidence'] for v in RESULTS.values()]).value_counts().to_dict(),
               coordinate_problems=int(sum(bool(v['coordinate_problems']) for v in RESULTS.values())),
               mapcfg={k: v for k, v in MAPCFG.items()})
(PH3 / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'mapcfg'}, indent=1))
TIPS.head(12)
""")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 8. Visualization
**What:** for any image `draw_mapping` draws, on the **original** photo, everything the task asks for:
- the **hand skeleton** (thin white) and all 21 landmarks (grey dots),
- the five **fingertips** highlighted in Phase 1's colours,
- the Phase 2 **fretboard region** (orange), **six strings** (cyan), **frets** (green) and **nut** (magenta),
- the **mapped position**: a small cross on the nearest string directly "below" the fingertip, joined to the tip by a thin line,
- a **label** next to each tip: `I s3 f2 .61` = Index, string 3, fret slot 2, confidence 0.61.

**Marker meaning:** filled circle = accepted; ring with a dashed look (yellow) = tentative; hollow red ring = ambiguous (`?` in the label); hollow grey ring with an `x` = outside the neck or no geometry.
Labels are pushed apart vertically so close fingertips stay readable (as in Phase 1).
`render(image_id, zoom=True)` crops to the neck and fingertips and enlarges, because strings are only a few pixels apart in the full photo.
""")
code(r"""
INITIAL = {'Thumb': 'T', 'Index': 'I', 'Middle': 'M', 'Ring': 'R', 'Pinky': 'P'}


def put_label(img, text, org, scale, color):
    th = max(1, int(round(1.5 * scale)))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), th + 2, cv2.LINE_AA)   # dark outline
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, color, th, cv2.LINE_AA)


def tip_label(t):
    base = INITIAL[t['finger']]
    if t['status'] in ('mapped', 'string_only'):
        txt = f"{base} s{t['string_index']}" + (f" f{t['fret_slot']}" if t['fret_slot'] else '') + f" {t['confidence']:.2f}"
    elif t['status'].startswith('ambiguous'):
        txt = f"{base} ?"
    elif t['status'] == 'beyond_detected_frets':
        txt = f"{base} >fret"
    else:
        txt = f"{base} x"
    return txt


def draw_mapping(bgr, image_id, show_skeleton=True):
    res, hands, p2 = RESULTS[image_id], HANDS[image_id], PHASE2[image_id]
    out = draw_result(bgr, p2, alpha=0.18)                         # Phase 2 drawing function (region, strings, frets, nut)
    s = max(out.shape[:2]) / 640.0
    if res['phase1']['fretting_hand_index'] is None:
        return out
    hd = hands['hands'][res['phase1']['fretting_hand_index']]
    pts = [(int(round(p[0])), int(round(p[1]))) for p in hd['landmarks_px']]
    if show_skeleton:
        for a, b in CONNECTIONS:
            cv2.line(out, pts[a], pts[b], (235, 235, 235), max(1, int(round(1.2 * s))), cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, p, max(1, int(round(2 * s))), (110, 110, 110), -1, cv2.LINE_AA)
    placed = []
    for t in sorted(res['fingertips'], key=lambda t: t['y_px']):
        c = FINGER_COLORS[t['finger']]
        tip = (int(round(t['x_px'])), int(round(t['y_px'])))
        rad = max(3, int(round(6 * s)))
        if t['decision'] == 'accepted':
            cv2.circle(out, tip, rad, c, -1, cv2.LINE_AA); cv2.circle(out, tip, rad + 1, (255, 255, 255), max(1, int(round(s))), cv2.LINE_AA)
        elif t['decision'] == 'tentative':
            cv2.circle(out, tip, rad, c, -1, cv2.LINE_AA); cv2.circle(out, tip, rad + 2, (0, 255, 255), max(1, int(round(1.5 * s))), cv2.LINE_AA)
        elif t['status'].startswith('ambiguous'):
            cv2.circle(out, tip, rad, (0, 0, 255), max(1, int(round(1.5 * s))), cv2.LINE_AA)
        else:
            cv2.circle(out, tip, rad, (150, 150, 150), max(1, int(round(1.5 * s))), cv2.LINE_AA)
            cv2.line(out, (tip[0] - rad, tip[1] - rad), (tip[0] + rad, tip[1] + rad), (150, 150, 150), 1, cv2.LINE_AA)
        if t['snap_xy'] is not None:
            sp = (int(round(t['snap_xy'][0])), int(round(t['snap_xy'][1])))
            cv2.line(out, tip, sp, c, 1, cv2.LINE_AA)
            cv2.drawMarker(out, sp, c, cv2.MARKER_CROSS, max(6, int(round(10 * s))), max(1, int(round(1.5 * s))), cv2.LINE_AA)
        lx, ly = tip[0] + int(8 * s), tip[1] - int(6 * s)
        for py in placed:                                          # push labels apart vertically
            if abs(ly - py) < 14 * s:
                ly = py + int(14 * s)
        placed.append(ly)
        put_label(out, tip_label(t), (lx, ly), s, c)
    return out


def crop_to_neck(img, image_id, pad=0.15, min_size=240, max_scale=3.0):
    res, p2 = RESULTS[image_id], PHASE2[image_id]
    pts = []
    if p2.get('fretboard'):
        pts += [tuple(p) for p in p2['fretboard']['polygon']]
    pts += [(t['x_px'], t['y_px']) for t in res['fingertips']]
    if not pts:
        return img
    pts = np.array(pts, float)
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    px, py = pad * (x1 - x0 + 40), pad * (y1 - y0 + 40)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))]
    if x1 - x0 < 20 or y1 - y0 < 20:
        return img
    crop = img[y0:y1, x0:x1]
    sc = min(max_scale, max(1.0, min_size / min(crop.shape[:2])))
    return cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC) if sc > 1.01 else crop


def render(image_id, zoom=True):
    # original, and the annotated picture (cropped to the neck if zoom=True), as a 2-panel figure
    bgr = IMG[image_id]
    res = RESULTS[image_id]
    ann = draw_mapping(bgr, image_id)
    shown = crop_to_neck(ann, image_id) if zoom else ann
    p2 = res['phase2']
    title = (f"{image_id} | Phase 2: {p2['status']}/{p2['confidence']} | hands: {res['phase1']['n_hands']} | "
             f"category: {res['category']}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={'width_ratios': [1, 1.25]})
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); axes[0].set_title('original', fontsize=9); axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)); axes[1].set_title('mapping (zoomed to the neck)' if zoom else 'mapping', fontsize=9); axes[1].axis('off')
    fig.suptitle(title, fontsize=10)
    for t in res['fingertips']:
        extra = f"  [{t['reason']}]" if t['reason'] else ''
        print(f"  {t['finger']:6s} {t['status']:18s} {t['decision']:9s} string={t['string_index']} slot={t['fret_slot']} conf={t['confidence']:.2f}{extra}")
    plt.tight_layout(); plt.show()


def gallery(ids, title, cols=3, size=4.6):
    ids = list(ids)
    print(f'=== {title} ({len(ids)} images) ===')
    if not ids:
        return
    ims, ts = [], []
    for i in ids:
        ims.append(crop_to_neck(draw_mapping(IMG[i], i), i)); ts.append(f"{i} | {RESULTS[i]['category']}")
    show_images(ims, ts, cols=cols, size=size)


# Two detailed examples: the best-supported success and one failure, to read the picture carefully
_succ = [k for k, v in RESULTS.items() if v['category'] == 'success']
if _succ:
    print('--- an example success ---'); render(_succ[0])
_fail = [k for k, v in RESULTS.items() if v['category'] in ('geometry_failed', 'ambiguous', 'outside_or_rejected')]
if _fail:
    print('--- an example that was refused ---'); render(_fail[0])
""")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 9. Evaluation
### 9a. Where does every image end up?
We never have a ground truth for free, so first we look at **what the system did** across the processed images: how many were mapped, refused, or impossible (no hand, no geometry), and the galleries below let you look at each group. **At least 24 images are shown** (up to 6 per category).

### 9b. Confidence summary
How confidence is distributed for the fingertips that reached a mapping, and how it differs by decision.

### 9c. Synthetic ground truth (exact, verifiable accuracy of the mapping maths)
For real photos we do not know the true string and fret of a finger. For **synthetic** necks we do: we build a neck with known strings and fret positions (real 12-tone fret spacing, widening toward the body), push it through a random **homography** (a perspective transform: rotation, scale and the strings fanning out), place fingertips at known (string, fret) spots, add realistic errors, and check what the mapping says.
Scenarios add, in turn: landmark noise (MediaPipe is not pixel-exact), a tip offset from the string (the camera/pad effect), and fret lines the detector missed. **What this does and does not prove:** it verifies the geometry and refusal logic when Phase 2 is right; it says nothing about how often Phase 2 or MediaPipe themselves are wrong on real photos.

### 9d. A manually checked test set on real photos (to measure real accuracy)
A picture can look plausible and still be wrong, so we measure against labels. **Label meaning:** for each fingertip of the fretting hand, the **string** and **fret slot** (numbered exactly as the overlay shows them) that the finger is *actually on*, judged by looking at the photo; `null` means "not on the strings of this neck / not resting on a fret region / cannot tell". Labels are only used for images whose Phase 2 geometry is visually correct (`geometry_ok`), so this measures **mapping accuracy given correct geometry**, separately from how good Phase 2 is.

**Honesty note:** no real-photo labels ship with this notebook. From a 2-D photo it is hard to tell *which string a finger really presses*, and labels that were not made by a person looking carefully would give a false sense of accuracy. Instead the notebook gives you the tools:
1. `review_sheet([...ids...])` draws **numbered** strings and fret slots on each photo (saved to `outputs/phase3/review/`).
2. `write_label_template(ids)` writes `labels_template.json` with empty fields. Fill in `geometry_ok` (is the drawn geometry right?), and per finger the `string` and `fret_slot` you see (or leave `null`), and set `human_verified: true`.
3. Save it as `configs/phase3_fingertip_labels.json` and re-run `python scripts/make_mapping_notebook.py` to embed it, or paste it into `LABELS` below. The evaluation cell then reports accuracy, coverage and false accepts.
**Metrics reported:** coverage (how often we dared to answer), string accuracy and fret-slot accuracy **among answered**, false-accept rate on fingers that are not actually on a fret region, and accuracy by confidence bin (do higher confidences really mean higher accuracy?).
""")
code(r"""
# 9a. galleries, at least 24 images: up to 6 per category
order = ['success', 'tentative_low_geometry', 'ambiguous', 'outside_or_rejected', 'geometry_failed', 'no_hand']
print(CATS.value_counts().reindex(order).fillna(0).astype(int).to_string())
shown_ids = []
for cat in order:
    ids = [k for k, v in RESULTS.items() if v['category'] == cat][:6]
    shown_ids += ids
    if cat == 'no_hand':
        print(f'=== no_hand ({int((CATS == "no_hand").sum())} images in total) ===  MediaPipe found no hand, nothing can be mapped')
        if ids:
            show_images([IMG[i] for i in ids], [i for i in ids], cols=3, size=3)
    else:
        gallery(ids, cat)
print('images displayed:', len(shown_ids))
""")
code(r"""
# 9b. confidence summary
mapped = TIPS[TIPS['status'].isin(['mapped', 'string_only']) | TIPS['status'].str.startswith('ambiguous')]
print('Fingertip records:', len(TIPS), '| reached a geometric mapping attempt:', len(mapped))
print(TIPS.groupby(['decision', 'status']).size().to_string())
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for d, c in [('accepted', 'tab:green'), ('tentative', 'tab:orange'), ('rejected', 'tab:red')]:
    v = TIPS[(TIPS['decision'] == d) & (TIPS['conf'] > 0)]['conf']
    if len(v):
        axes[0].hist(v, bins=np.linspace(0, 1, 21), alpha=0.6, label=f'{d} (n={len(v)})', color=c)
axes[0].set_title('confidence by decision'); axes[0].set_xlabel('confidence'); axes[0].legend()
TIPS['status'].value_counts().plot.barh(ax=axes[1], color='gray'); axes[1].set_title('fingertip status')
plt.tight_layout(); plt.show()
acc = TIPS[TIPS['decision'] == 'accepted']
print('accepted fingertips:', len(acc), '| images with at least one accepted:', int((CATS == 'success').sum()), 'of', len(CATS))
if len(acc):
    print(acc['conf'].describe()[['min', '50%', 'max']].round(2).to_dict())
""")
code(r"""
# 9c. synthetic ground truth: necks whose true geometry is known, so mapping accuracy can be measured EXACTLY
def make_synthetic_case(rng, drop_frets=0):
    N_FR, L = 12, 650.0
    fu = lambda n: L * (1 - 2 ** (-n / 12))                    # distance of fret n from the nut (real 12-tone rule, mm-like units)
    width = lambda u: 36.0 + 12.0 * u / 400.0                   # distance between the outer strings; the neck widens toward the body
    sv = lambda i, u: (i - 2.5) * width(u) / 5                  # across-neck position of string i (0..5) at distance u
    u_max = fu(N_FR)
    scale, th = rng.uniform(1.4, 2.2), rng.uniform(0, 2 * np.pi)   # random size and ANY rotation (nut can be on either side)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    src = np.float32([[0, -30], [u_max, -30], [u_max, 30], [0, 30]])
    jit = rng.uniform(-1, 1, (4, 2)) * np.float32([0.05 * u_max, 7.0])            # corner jitter = perspective (strings fan out)
    dst = ((src + jit) * scale) @ Rm.T
    dst = (dst - dst.mean(0) + 400).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)                    # the homography: neck plane -> photo
    to_img = lambda pts: cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)
    u0 = 4.0
    strings = [dict(index=i + 1, p1=to_img([[u0, sv(i, u0)]])[0].tolist(), p2=to_img([[u_max, sv(i, u_max)]])[0].tolist()) for i in range(6)]
    keep = [n for n in range(1, N_FR + 1)]
    for n in (rng.choice(np.arange(1, N_FR + 1), size=drop_frets, replace=False) if drop_frets else []):   # any fret, even the first
        keep.remove(int(n))                                      # frets the detector 'missed'
    frets = [dict(index=k + 1, p1=to_img([[fu(n), -width(fu(n)) / 2 - 4]])[0].tolist(), p2=to_img([[fu(n), width(fu(n)) / 2 + 4]])[0].tolist())
             for k, n in enumerate(keep)]
    nut = dict(p1=to_img([[0, -width(0) / 2 - 4]])[0].tolist(), p2=to_img([[0, width(0) / 2 + 4]])[0].tolist(), confidence='medium')
    poly = to_img([[0, -width(0) / 2 - 3], [u_max, -width(u_max) / 2 - 3], [u_max, width(u_max) / 2 + 3], [0, width(0) / 2 + 3]]).tolist()
    p2 = dict(status='ok', confidence='high', failure_reason=None, strings=strings, frets=frets, nut=nut, fretboard=dict(polygon=poly))
    return p2, dict(fu=fu, sv=sv, width=width, to_img=to_img, N_FR=N_FR)


def synthetic_eval(n_necks=40, tips_per_neck=8, noise_px=1.0, parallax=0.0, drop_frets=0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_necks):
        p2, g = make_synthetic_case(rng, drop_frets=drop_frets)
        model = build_neck_model(p2)
        for _ in range(tips_per_neck):
            s, n = int(rng.integers(0, 6)), int(rng.integers(1, g['N_FR'] + 1))
            u = g['fu'](n - 1) + rng.uniform(0.4, 0.8) * (g['fu'](n) - g['fu'](n - 1))      # finger sits in fret region n, behind the wire
            v = g['sv'](s, u) + rng.uniform(-parallax, parallax) * g['width'](u) / 5          # camera/pad offset from the string
            P = g['to_img']([[u, v]])[0] + rng.normal(0, noise_px, 2)                         # landmark error
            r = map_fingertip(model, p2, 'Index', dict(id=8, x_px=float(P[0]), y_px=float(P[1])), 0.9)
            rows.append(dict(decision=r['decision'], status=r['status'], gt_string=s + 1, gt_fret=n, pred_string=r['string_index'],
                             pred_fret=r['fret_number'], conf=r['confidence']))
    return pd.DataFrame(rows)


configs = [('clean (1 px landmark noise)', dict(noise_px=1.0)),
           ('3 px landmark noise', dict(noise_px=3.0)),
           ('3 px noise + tip offset up to 0.25 string gap', dict(noise_px=3.0, parallax=0.25)),
           ('... + 1 missed fret line', dict(noise_px=3.0, parallax=0.25, drop_frets=1)),
           ('... + 2 missed fret lines', dict(noise_px=3.0, parallax=0.25, drop_frets=2))]
table = []
for name, kw in configs:
    d = synthetic_eval(seed=11, **kw)
    acc = d[d['decision'] == 'accepted']
    table.append({'scenario': name, 'fingertips': len(d), 'accepted': len(acc), 'coverage': round(len(acc) / len(d), 2),
                  'string correct (accepted)': round((acc['pred_string'] == acc['gt_string']).mean(), 3) if len(acc) else None,
                  'fret correct (accepted)': round((acc['pred_fret'] == acc['gt_fret']).mean(), 3) if len(acc) else None,
                  'wrong accepts': int(((acc['pred_string'] != acc['gt_string']) | (acc['pred_fret'] != acc['gt_fret'])).sum()),
                  'refused as ambiguous': int(d['status'].str.startswith('ambiguous').sum())})
SYNTH = pd.DataFrame(table)
print('Exact ground truth (known geometry, random rotation + perspective); the fingertip is placed at a known string and fret.')
print(SYNTH.to_string(index=False))
""")

code(r"""
# 9d. manually checked test set
LABELS = json.loads('''""" + labels_literal.replace("\\", "\\\\").replace("'''", "'\\''") + r"""''')


def review_sheet(ids, save=True, show=True):
    # Numbered review pictures: strings 1-6 are numbered at one end, fret SLOTS are numbered inside each region,
    # fingertips are lettered. Use these to write or check labels (configs/phase3_fingertip_labels.json).
    out_imgs, titles = [], []
    for i in ids:
        bgr, res, p2, model = IMG[i], RESULTS[i], PHASE2[i], MODELS[i]
        img = draw_mapping(bgr, i, show_skeleton=False)
        s = max(img.shape[:2]) / 640.0
        if model is not None:
            for k, st in enumerate(model['strings'], 1):
                p = tuple(int(round(v)) for v in st['p1'])
                put_label(img, str(k), (p[0] - int(12 * s), p[1] + int(4 * s)), 1.3 * s, (255, 255, 0))
            B = model['boundaries']
            for k in range(1, len(B)):
                c = (_mid(B[k - 1]) + _mid(B[k])) / 2
                put_label(img, f'f{k}', (int(c[0]) - int(8 * s), int(c[1]) - int(model['spacing'] * 3.2)), 1.2 * s, (0, 255, 0))
        out = crop_to_neck(img, i, pad=0.25, min_size=420, max_scale=4.0)
        out_imgs.append(out); titles.append(f"{i}  (cyan = string number, green fK = fret slot K)")
        if save:
            cv2.imwrite(str(PH3 / 'review' / f'{i}.jpg'), out)
    if show:
        show_images(out_imgs, titles, cols=2, size=7)


def write_label_template(ids, path=None):
    path = Path(path or PH3 / 'review' / 'labels_template.json')
    tmpl = {i: dict(geometry_ok=None, human_verified=False, annotator='',
                    fingers={f: dict(string=None, fret_slot=None) for f in ['Index', 'Middle', 'Ring', 'Pinky']}) for i in ids}
    path.write_text(json.dumps(tmpl, indent=1), encoding='utf-8')
    return path


def evaluate_labels(labels, results):
    rows = []
    for image_id, lab in labels.items():
        if image_id not in results or not lab.get('geometry_ok', False):
            continue
        res = results[image_id]
        by_finger = {t['finger']: t for t in res['fingertips']}
        for finger, gt in lab['fingers'].items():
            t = by_finger.get(finger)
            on = gt.get('string') is not None
            rows.append(dict(image=image_id, finger=finger, gt_string=gt.get('string'), gt_slot=gt.get('fret_slot'), on_fret=on,
                             decision=t['decision'] if t else 'no_hand', pred_string=t['string_index'] if t else None,
                             pred_slot=t['fret_slot'] if t else None, conf=t['confidence'] if t else 0.0,
                             verified=lab.get('human_verified', False)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pred = df['decision'].isin(['accepted', 'tentative'])
    df['answered'] = pred
    df['string_ok'] = pred & (df['pred_string'] == df['gt_string'])
    df['slot_ok'] = pred & (df['pred_slot'] == df['gt_slot'])
    df['both_ok'] = df['string_ok'] & df['slot_ok']
    df['string_off_by_1'] = pred & df['gt_string'].notna() & ((df['pred_string'] - df['gt_string']).abs() <= 1)
    return df


_cand = [k for k, v in RESULTS.items() if v['category'] in ('success', 'tentative_low_geometry', 'ambiguous')]
print('label template for', len(_cand), 'candidate images written to', write_label_template(_cand))
EVAL = evaluate_labels(LABELS, RESULTS)
if EVAL.empty:
    print('No manually checked labels are embedded yet (LABELS is empty or none of its images were processed in this run).')
    print('Create them with review_sheet([...ids...]) and the template above (see the text in section 9d).')
else:
    n_img = EVAL['image'].nunique()
    print(f"labeled fingertips: {len(EVAL)} on {n_img} images | human-verified: {int(EVAL['verified'].sum())} of {len(EVAL)} (the rest need your check)")
    on = EVAL[EVAL['on_fret']]
    off = EVAL[~EVAL['on_fret']]
    for name, sub in [('accepted only', on[on['decision'] == 'accepted']), ('accepted + tentative', on[on['answered']])]:
        if len(sub):
            print(f"  [{name}] n={len(sub)}  string acc={sub['string_ok'].mean():.2f}  within 1 string={sub['string_off_by_1'].mean():.2f}  "
                  f"fret-slot acc={sub['slot_ok'].mean():.2f}  string+slot acc={sub['both_ok'].mean():.2f}")
    print(f"  coverage on fingers that are on a fret region: accepted {np.mean(on['decision'] == 'accepted'):.2f}, "
          f"accepted+tentative {on['answered'].mean():.2f}, abstained {1 - on['answered'].mean():.2f}")
    if len(off):
        print(f"  false accepts (finger NOT on a fret region but we answered 'accepted'): {int((off['decision'] == 'accepted').sum())} of {len(off)}")
    ans = on[on['answered']].copy()
    if len(ans) >= 4:
        ans['bin'] = pd.cut(ans['conf'], [0, 0.3, 0.5, 1.0])
        print('  accuracy by confidence bin (string+slot):')
        print(ans.groupby('bin', observed=True)['both_ok'].agg(['mean', 'count']).round(2).to_string())
    wrong = EVAL[EVAL['answered'] & EVAL['on_fret'] & ~EVAL['both_ok']]
    print('  incorrect mappings:', len(wrong))
    if len(wrong):
        print(wrong[['image', 'finger', 'gt_string', 'gt_slot', 'pred_string', 'pred_slot', 'decision', 'conf']].to_string(index=False))
        gallery(list(dict.fromkeys(wrong['image']))[:6], 'incorrect mappings (labels vs prediction in the table above)')
EVAL.head(10) if not EVAL.empty else None
""")

md(r"""
## 10. Limitations, and the hand-off to Phase 4
**What this notebook cannot tell you**
| Limit | Why |
|---|---|
| **Proximity is not contact** | The photo is 2-D. A fingertip drawn on top of a string may be hovering above it, touching a neighbouring string, or damping it. The landmark is the *tip*, while the string is pressed with the *pad*. Camera angle shifts the apparent position by up to a string gap. Every record is therefore flagged `proximity_only`. |
| **String numbers are not tuning** | Strings are numbered 1-6 in Phase 2's order. Which end is the high E depends on handedness and camera side and is not known from the picture. |
| **Fret numbers need a nut** | Without a medium-confidence nut only the **slot** is reported. A single missed fret line shifts all later slots; we repair obvious gaps with virtual boundaries but cannot repair everything. |
| **Garbage in, garbage out** | If Phase 2 locked onto wood grain or a sign, the mapping is confidently wrong. That is why low-confidence geometry gives only `tentative` and why Phase 2's confidence is part of the product. |
| **MediaPipe misses many hands** | About 43% of the dataset gets no hand at the Phase 1 threshold, and fingertips are often hidden when fretting; hidden fingertips are *guessed* by the model. |
| **Heavily augmented images** | Colour inversion, rotation and black borders hurt both the hand model and the line detector. Test on your own clean footage. |
| **Thumb** | Never accepted: it normally sits behind the neck. |

**Phase 4 hand-off:** read `outputs/phase3/mapping/<image_id>.json`. Use only fingertips with `decision == "accepted"`. Each has `string_index`, `fret_slot` (and `fret_number` when anchored by a nut), `confidence`, and the image ID that matches the Phase 1 and Phase 2 exports. `phase2.confidence`, `geometry.orientation` and `geometry.n_virtual_boundaries` let a later model discount shaky geometry. Do not feed `tentative` or `rejected` rows into chord logic as facts.
""")
code(r"""
# colab-only
import shutil
from google.colab import files
zip_out = shutil.make_archive('/content/phase3_results', 'zip', PH3)
print('Created', zip_out)
files.download(zip_out)
""")

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "03_fingertip_fret_mapping.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "| Phase 2 detector fingerprint", _p2digest, "| labels embedded:", len(json.loads(labels_literal)))
