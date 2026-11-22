"""Generates notebooks/04a_extract_features.ipynb (Phase 4, step a: run Phases 1-3 over the whole dataset).

Usage: python scripts/make_features_notebook.py

- Cells whose first line is '# colab-only' need Google Colab. Everything else runs anywhere (tested locally with the
  colab-only cells skipped and DATA_SOURCE='zip').
- Phase 1-3 code is NOT re-written here: it is copied verbatim out of notebooks/03_fingertip_fret_mapping.ipynb
  (which itself copies Phase 2 from notebook 02), so the three phases cannot drift apart.
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
# Copy code cells out of notebook 03
# ---------------------------------------------------------------------------------------------------------------
NB3 = json.loads((ROOT / "notebooks" / "03_fingertip_fret_mapping.ipynb").read_text(encoding="utf-8"))
SRC3 = ["".join(c["source"]) for c in NB3["cells"] if c["cell_type"] == "code"]


def cell3(marker):
    hit = [s for s in SRC3 if marker in s]
    assert hit, f"notebook 03 has no code cell containing {marker!r}"
    return hit[0]


def cut(s, start=None, end=None):
    if start:
        assert start in s, start
        s = s[s.index(start):]
    if end:
        assert end in s, end
        s = s[:s.index(end)]
    return s.rstrip() + "\n"


# Phase 1 + 3 settings and helpers
cfg_src = cut(cell3("MAPCFG = dict("), end="\nIMAGE_SOURCE")                       # imports + MAPCFG
mp_head = cut(cell3("from mediapipe.tasks import python as mp_python"), end="\n_opts = vision")   # model + constants
mp_detect = cut(cell3("def detect_hands"), start="def detect_hands", end="\n\nHANDS = {}")
coord_src = cut(cell3("def check_coordinates"), start="def check_coordinates", end="\n\nPHASE2, P2_SRC")
p2_src = cell3("# Phase 2 detector: copied from")
geom_src = cell3("def _unit(v):")
map_src = cell3("def select_fretting_hand")
DIGEST = hashlib.sha1((cfg_src + mp_head + mp_detect + coord_src + p2_src + geom_src + map_src).encode("utf-8")).hexdigest()[:10]

# ---------------------------------------------------------------------------------------------------------------
md(r"""
# Notebook 04a: Run Phases 1-3 over the whole dataset and build the feature table (Phase 4, step a)

**Goal:** for every dataset image, run **MediaPipe (Phase 1) -> fretboard detection (Phase 2) -> fingertip-to-string/fret mapping (Phase 3)** and save the result. At the end we have, for each image, a row of **numbers describing where the fingers are on the neck**. Notebook 04b trains a chord classifier on those numbers (and on the images themselves).

**Not in this notebook:** training or any chord prediction. This notebook only *extracts* information.

### Why a separate notebook?
Phase 2 costs about **1.7 seconds per image** on a CPU. The dataset has **3,151 images**, so a full run is roughly **1.5-2.5 hours**. We do it once, save one small JSON per image, and reuse it for every experiment afterwards. That is why this notebook is built to be:
- **Resumable:** every finished image is written to its own file. If Colab disconnects, run the extraction cell again and it skips what is already done.
- **Smoke-testable:** `SMOKE = True` (the default) processes only ~20 images per chord first, so you can see the whole pipeline work in a few minutes before committing to the long run.
- **Parallel:** two worker processes (Colab gives 2 CPU cores) so the long run takes about half the time.

### What the feature table contains
One row per image. Per finger (index, middle, ring, pinky, thumb): the fingertip's **string position** (a number from 1 to 6, continuous, not rounded), its **position along the neck** (in fret-region units), a confidence, and whether it was accepted by Phase 3. Plus **relative** features (differences between fingers), because a chord is a *shape*, not an absolute position. Images where no hand or no neck was found get empty (`NaN`) cells instead of invented numbers; the model in 04b is built to handle missing values.

### Important facts to keep in mind
- MediaPipe finds **no hand in about 43%** of these images, so many rows will have no finger features. That is expected and is reported below.
- Every coordinate is in **original-image pixels**, and the image **ID** is `<Chord>_<split>_<row>` (for example `Am_train_00281`), the same IDs used in notebooks 01-03.

### Cells
0. Setup, 1. Config and data, 2. Copied Phase 1-3 code, 3. Extraction (resumable, parallel), 4. Feature table, 5. Report, 6. Download
""")

md(r"""
## 0. Setup
**What:** creates the folders and installs the two libraries this notebook needs (`mediapipe`, `datasets`). Everything is written under `outputs/phase4/`. No Google Drive.
""")
code(r"""
# colab-only
import os
PROJECT_DIR = '/content/Guitar-Chord-Recognition'      # temporary folder on the Colab machine
os.makedirs(f'{PROJECT_DIR}/data', exist_ok=True)
print('Project folder:', PROJECT_DIR)
""")
code(r"""
import os, sys, subprocess, importlib
def _ensure(pkg, mod=None):
    try:
        importlib.import_module(mod or pkg)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', pkg], check=True)
        importlib.invalidate_caches()
_ensure('mediapipe'); _ensure('datasets')
import mediapipe as mp
print('mediapipe', mp.__version__)
""")

md(r"""
## 1. Config and the list of images to process
**What:** one place for the run settings, then the list `ITEMS` of images to process.
- `DATA_SOURCE = 'hf'` (default): download the dataset `j-asra-j/guitar-chords` from Hugging Face inside Colab (no upload needed). `'zip'` uses the small `guitar_chords_sample.zip` (280 images) instead, for quick tests.
- `SMOKE = True`: only `SMOKE_PER_CLASS` images per chord (fixed random seed). **Set `SMOKE = False` for the full 3,151-image run.**
- `N_WORKERS = 2`: processes running in parallel. `TIME_BUDGET_MIN`: stop after this many minutes (and just run the cell again later to continue), useful if you want to do the long run in several sittings.
- IDs are `<Chord>_<split>_<row:05d>`, where `row` is the image's position in its split. The sample zip uses the same IDs, so results line up with notebooks 01-03.
""")
code(r"""
import json, glob, random, zipfile, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd

DATA_SOURCE = 'hf'          # 'hf' | 'zip'
SMOKE = True                # True: ~20 images per chord first; False: all 3,151 images
SMOKE_PER_CLASS = 20
N_WORKERS = 2               # parallel processes (Colab has 2 CPU cores); 1 = run in this process
TIME_BUDGET_MIN = None      # e.g. 60 = stop after an hour, re-run the extraction cell to continue
CHECKPOINT_EVERY = 500      # every this-many NEW images: save a partial zip and (on Colab) download it to your computer
CHECKPOINT_DOWNLOAD = True  # False = only save the zip on the Colab disk (lost if Colab resets)
RESUME_FROM_UPLOAD = False  # True = pick a partial p4_features*.zip from an earlier run so finished images are not redone
SEED = 7

P4 = Path(PROJECT_DIR) / 'outputs' / 'phase4'
FEAT_DIR = P4 / 'features'
FEAT_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = f'{PROJECT_DIR}/data/guitar_chords_sample.zip'

if RESUME_FROM_UPLOAD:                                       # restore finished images from an earlier (partial) run
    from google.colab import files
    print('Choose a partial p4_features*.zip from an earlier run ...')
    restored = 0
    for fname, data in files.upload().items():
        if fname.lower().endswith('.zip'):
            zp = P4 / fname; zp.write_bytes(data)
            with zipfile.ZipFile(zp) as z:
                for m in z.namelist():
                    if m.startswith('features/') and m.endswith('.json'):
                        (FEAT_DIR / Path(m).name).write_bytes(z.read(m)); restored += 1
    print(f'restored {restored} finished images; the extraction below will skip them')

DS = {}                      # split -> HF dataset (only for DATA_SOURCE='hf')
ITEMS = []                   # one dict per image: id, chord, split, row (+ zip_name)

if DATA_SOURCE == 'hf':
    from datasets import load_dataset
    R = 'hf://datasets/j-asra-j/guitar-chords@refs%2Fconvert%2Fparquet/default/'
    _ds = load_dataset('parquet', data_files={'train': [R + 'train/0000.parquet', R + 'train/0001.parquet'],
                                               'validation': R + 'validation/0000.parquet', 'test': R + 'test/0000.parquet'})
    NAMES = _ds['train'].features['label'].names
    for split in ('train', 'validation', 'test'):
        DS[split] = _ds[split]
        for row, lab in enumerate(_ds[split]['label']):
            ITEMS.append(dict(id=f'{NAMES[lab]}_{split}_{row:05d}', chord=NAMES[lab], split=split, row=row))
else:
    if not os.path.exists(ZIP_PATH):
        from google.colab import files
        print('Choose guitar_chords_sample.zip ...')
        up = files.upload()
        Path(ZIP_PATH).write_bytes(up[[n for n in up if n.lower().endswith('.zip')][0]])
    with zipfile.ZipFile(ZIP_PATH) as z:
        for zn in sorted(n for n in z.namelist() if n.lower().endswith('.jpg') and '__MACOSX' not in n):
            chord, stem = zn.split('/')[-2], Path(zn).stem            # 'Am', 'train_00281'
            split, row = stem.rsplit('_', 1)
            ITEMS.append(dict(id=f'{chord}_{stem}', chord=chord, split=split, row=int(row), zip_name=zn))
    NAMES = sorted({it['chord'] for it in ITEMS})

assert len({it['id'] for it in ITEMS}) == len(ITEMS), 'image IDs must be unique'
if SMOKE:                                                         # a few images per chord, same ones every run
    rng = random.Random(SEED)
    keep = []
    for ch in NAMES:
        pool = [it for it in ITEMS if it['chord'] == ch]
        keep += rng.sample(pool, min(SMOKE_PER_CLASS, len(pool)))
    ITEMS = sorted(keep, key=lambda it: it['id'])
done = sum((FEAT_DIR / f"{it['id']}.json").exists() for it in ITEMS)
print(f"{len(ITEMS)} images selected ({'SMOKE' if SMOKE else 'FULL'} run, source={DATA_SOURCE}); already extracted: {done}")
print(pd.Series([it['chord'] for it in ITEMS]).value_counts().sort_index().to_dict())
""")

md(r"""
## 2. Phases 1-3 code (copied unchanged from notebook 03)
**What:** three cells of function code, copied **verbatim by the generator script** from `03_fingertip_fret_mapping.ipynb`, which itself copies the Phase 2 detector from notebook 02. Notebooks 01-03 are not modified.
**Why copy instead of import:** Colab notebooks cannot import each other, and re-writing the code would let the versions drift. The fingerprint in the first comment line changes whenever the underlying code changes (re-run `python scripts/make_features_notebook.py` to refresh).

**What the code does, in order (all explained in notebooks 01-03):**
1. `detect_hands`: MediaPipe finds up to 2 hands and 21 landmarks each (pixels in the *original* image, not rounded).
2. `detect_fretboard`: ridge filter + angle comb search + per-string fitting + Hough frets and nut.
3. `map_image`: for the fretting hand's fingertips, signed distance to the detected string and fret lines, ambiguity margins, 12-tone missed-fret repair, and confidence. Each fingertip ends as `accepted`, `tentative` or `rejected`, and is always marked `proximity_only` (nearness in a 2-D photo is not proof of pressing a string).
""")
code(f"# Phase 1/3 settings and imports: copied from notebooks/03_fingertip_fret_mapping.ipynb (fingerprint {DIGEST}). Do not edit here.\n"
     + cfg_src + "\n" + mp_head + "\n" + mp_detect + "\n\n" + coord_src)
code(f"# Phase 2 detector: copied from notebook 03 / notebook 02 (fingerprint {DIGEST}). Do not edit here.\n" + p2_src)
code(f"# Phase 3 geometry and mapping: copied from notebook 03 (fingerprint {DIGEST}). Do not edit here.\n" + geom_src + "\n\n" + map_src)

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 3. Extraction (resumable and parallel)
**What:** `process_item` handles one image end to end and writes `outputs/phase4/features/<id>.json` containing the Phase 3 result, the full Phase 2 geometry, and the chord label.
**Technique: multiprocessing.** `multiprocessing.Pool` starts 2 worker processes; each one loads its **own** MediaPipe model (these objects must not be shared between processes) and then takes images from a queue. `fork` is used so the workers inherit all the functions defined above.
**Resumable:** a file is written only after the whole image is processed (written to a temporary name, then renamed), so a crash or disconnect never leaves a half-written file, and re-running skips finished images.
**Protection against Colab resets (this really happened once):** Colab can recycle the machine and erase its disk with no warning, losing everything. So every `CHECKPOINT_EVERY` (500) new images the notebook saves a partial zip and **downloads it to your computer** (your browser may ask once to allow multiple downloads). If the machine is reset, set `RESUME_FROM_UPLOAD = True` in section 1, run it, and pick the newest partial zip: the finished images are restored and only the rest is processed.
**Live progress:** a status line every few images: images done, speed, time left, and how the results are splitting into categories (`success`, `no_hand`, ...).
""")
code(r"""
import multiprocessing as mproc
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

LANDMARKER = None


def make_landmarker():
    # Exactly the Phase 1 settings: image mode, up to 2 hands, all confidences 0.3.
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3)
    return vision.HandLandmarker.create_from_options(opts)


def _init_worker():
    global LANDMARKER
    LANDMARKER = make_landmarker()            # one MediaPipe model per process


def load_bgr(item):
    # The ORIGINAL image as a BGR array. Nothing is resized here, so all coordinates stay in original pixels.
    if DATA_SOURCE == 'hf':
        pil = DS[item['split']][item['row']]['image'].convert('RGB')
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    with zipfile.ZipFile(ZIP_PATH) as z:
        return cv2.imdecode(np.frombuffer(z.read(item['zip_name']), np.uint8), cv2.IMREAD_COLOR)


def process_item(item):
    out = FEAT_DIR / f"{item['id']}.json"
    if out.exists():
        return item['id'], 'cached', 0.0
    t0 = time.time()
    bgr = load_bgr(item)
    hands = detect_hands(bgr, item['id'])                                       # Phase 1
    p2 = detect_fretboard(bgr, item['id'])                                      # Phase 2
    problems = check_coordinates(item['id'], bgr, hands, p2)                    # same pixel grid everywhere?
    res, _ = map_image(item['id'], bgr, hands, p2, 'recompute', problems)       # Phase 3
    res.update(chord=item['chord'], split=item['split'], row=item['row'], phase2_geometry=p2,
               n_hands_raw=[dict(score=h['score'], handedness=h['handedness_as_model_sees_it']) for h in hands['hands']])
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(res), encoding='utf-8')
    tmp.replace(out)                                                            # atomic: no half-written files
    return item['id'], res['category'], time.time() - t0


def save_checkpoint():
    # A partial results zip. Colab can reset the machine without warning and erases its disk, so we also download it.
    n_done = len(list(FEAT_DIR.glob('*.json')))
    zp = shutil.make_archive(str(Path(PROJECT_DIR) / f'p4_features_partial_{n_done:04d}'), 'zip', P4)
    print(f'  checkpoint: {n_done} images -> {zp} ({os.path.getsize(zp) / 1e6:.1f} MB)', flush=True)
    if CHECKPOINT_DOWNLOAD:
        try:
            from google.colab import files
            files.download(zp)
        except Exception as e:                                  # not on Colab / no browser attached: the zip is still on disk
            print('  (browser download not started:', type(e).__name__, ')')


def run_extraction(items, n_workers=N_WORKERS, budget_min=TIME_BUDGET_MIN, every=10, checkpoint_every=CHECKPOINT_EVERY):
    todo = [it for it in items if not (FEAT_DIR / f"{it['id']}.json").exists()]
    print(f'{len(items) - len(todo)} already done, {len(todo)} to do, {n_workers} worker(s)')
    if not todo:
        return
    t_start, counts, n = time.time(), {}, 0
    if n_workers > 1:
        pool = mproc.get_context('fork').Pool(n_workers, initializer=_init_worker)
        stream = pool.imap_unordered(process_item, todo, chunksize=2)
    else:
        _init_worker(); pool = None
        stream = map(process_item, todo)
    try:
        for iid, cat, dt in stream:
            n += 1; counts[cat] = counts.get(cat, 0) + 1
            if n % every == 0 or n == len(todo):
                el = time.time() - t_start
                print(f'[{n}/{len(todo)}] {el / 60:.1f} min elapsed, {el / n:.2f} s/image, ~{(len(todo) - n) * el / n / 60:.1f} min left | {counts}', flush=True)
            if checkpoint_every and n % checkpoint_every == 0 and n < len(todo):
                save_checkpoint()
            if budget_min and (time.time() - t_start) / 60 > budget_min:
                print(f'time budget of {budget_min} min reached: re-run this cell to continue'); break
    finally:
        if pool is not None:
            pool.terminate(); pool.join()
    print('finished this run:', counts)


run_extraction(ITEMS)
print('JSON files now on disk:', len(list(FEAT_DIR.glob('*.json'))))
""")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 4. Build the feature table
**What:** read every JSON and flatten it into one row per image (`features.csv`).

**Columns, in plain words:**
- Identity: `id`, `chord` (the label), `split` (the dataset's own split), `row`.
- Availability flags: `hand_found`, `geom_usable` (6 strings found), `geom_conf` (`high`/`low`/`none` from Phase 2) and the number `geom_conf_num` (1.0 / 0.6 / 0), `n_accepted` fingertips.
- Per finger `F` in Index, Middle, Ring, Pinky, Thumb: `F_sp` (string position 1-6, continuous), `F_nc` (position along the neck in fret-region units), `F_conf` (Phase 3 confidence), `F_dec` (2 = accepted, 1 = tentative, 0 = rejected), `F_in` (1 if the tip is on the neck area).
- **Relative features:** differences between every pair of the four fretting fingers (`d_sp_Index_Middle`, `d_nc_...`), and each finger minus the lowest finger on the neck (`F_sp_rel`, `F_nc_rel`). A chord is a *shape*, so these should matter more than absolute positions.
- Crop box for the image models: `crop_x0..y1` plus `crop_kind` (`neck+hand`, `neck`, or `full` when nothing was found). Box = fretboard polygon plus hand box, padded 12% and clipped to the image.

**Missing values:** when a finger is not on the neck, or no hand/neck was found, the cells are left **empty (NaN)**. Nothing is invented. Gradient-boosted trees (notebook 04b) handle NaN natively.
""")
code(r"""
FINGERS = ['Index', 'Middle', 'Ring', 'Pinky', 'Thumb']
FRET4 = ['Index', 'Middle', 'Ring', 'Pinky']
DEC = {'accepted': 2, 'tentative': 1, 'rejected': 0}
IN_NECK = {'mapped', 'string_only', 'ambiguous_string', 'ambiguous_fret', 'ambiguous_both'}
nan = float('nan')


def crop_box(res):
    # Box (in original pixels) around the fretboard and the hand, padded and clipped to the image.
    W, H = res['image']['width'], res['image']['height']
    pts = []
    poly = (res.get('phase2_geometry') or {}).get('fretboard')
    kinds = []
    if poly:
        pts += [tuple(p) for p in poly['polygon']]; kinds.append('neck')
    bb = res['phase1'].get('hand_bbox')
    if bb:
        pts += [(bb[0], bb[1]), (bb[2], bb[3])]; kinds.append('hand')
    if not pts:
        return 0, 0, W, H, 'full'
    a = np.array(pts, float)
    x0, y0, x1, y1 = a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()
    px, py = 0.12 * (x1 - x0), 0.12 * (y1 - y0)
    box = [int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(W, x1 + px)), int(min(H, y1 + py))]
    if box[2] - box[0] < 32 or box[3] - box[1] < 32:
        return 0, 0, W, H, 'full'
    return (*box, '+'.join(kinds))


def feature_row(res):
    p2, g = res['phase2'], res['geometry']
    tips = {t['finger']: t for t in res['fingertips']}
    row = dict(id=res['image']['id'], chord=res['chord'], split=res['split'], row=res['row'],
               width=res['image']['width'], height=res['image']['height'],
               hand_found=int(res['phase1']['fretting_hand_index'] is not None), n_hands=res['phase1']['n_hands'],
               geom_usable=int(g['usable']), geom_status=p2['status'], geom_conf=p2['confidence'],
               geom_conf_num={'high': 1.0, 'low': 0.6}.get(p2['confidence'], 0.0), category=res['category'],
               n_accepted=sum(t['decision'] == 'accepted' for t in res['fingertips']),
               nut_conf=g.get('nut_confidence') or 'none', n_boundaries=g['n_fret_boundaries'], n_virtual=g['n_virtual_boundaries'],
               hand_score=tips['Index']['hand_score'] if tips else nan)
    for f in FINGERS:
        t = tips.get(f)
        on_neck = t is not None and t['status'] in IN_NECK
        row[f'{f}_in'] = int(on_neck)
        row[f'{f}_sp'] = t['string_position'] if (t and (on_neck or t['status'] == 'beyond_detected_frets')) else nan
        row[f'{f}_nc'] = t['neck_coord'] if (on_neck and t['neck_coord'] is not None) else nan
        row[f'{f}_conf'] = t['confidence'] if t else nan
        row[f'{f}_dec'] = DEC[t['decision']] if t else nan
    for a_i, a in enumerate(FRET4):                      # differences between the fretting fingers: chords are shapes
        for b in FRET4[a_i + 1:]:
            row[f'd_sp_{a}_{b}'] = row[f'{a}_sp'] - row[f'{b}_sp']
            row[f'd_nc_{a}_{b}'] = row[f'{a}_nc'] - row[f'{b}_nc']
    sp = [row[f'{f}_sp'] for f in FRET4]; nc = [row[f'{f}_nc'] for f in FRET4]
    sp_min = np.nanmin(sp) if not np.all(np.isnan(sp)) else nan
    nc_min = np.nanmin(nc) if not np.all(np.isnan(nc)) else nan
    for f in FRET4:
        row[f'{f}_sp_rel'] = row[f'{f}_sp'] - sp_min
        row[f'{f}_nc_rel'] = row[f'{f}_nc'] - nc_min
    row['n_in_neck'] = sum(row[f'{f}_in'] for f in FRET4)
    row['features_usable'] = int(row['hand_found'] and row['n_in_neck'] >= 2)
    x0, y0, x1, y1, kind = crop_box(res)
    row.update(crop_x0=x0, crop_y0=y0, crop_x1=x1, crop_y1=y1, crop_kind=kind)
    return row


rows = []
for it in ITEMS:
    f = FEAT_DIR / f"{it['id']}.json"
    if f.exists():
        rows.append(feature_row(json.loads(f.read_text(encoding='utf-8'))))
FEATS = pd.DataFrame(rows)
FEATS.to_csv(P4 / 'features.csv', index=False)
print(FEATS.shape, '->', P4 / 'features.csv')
FEATS[['id', 'chord', 'hand_found', 'geom_conf', 'n_accepted', 'Index_sp', 'Index_nc', 'features_usable', 'crop_kind']].head(8)
""")

md(r"""
## 5. Report: how much of the dataset do we actually get features for?
This is the honest picture that decides how much the geometry features can help in 04b. Read it before trusting the next notebook's results.
- **hand found:** MediaPipe returned a hand.
- **geometry high:** Phase 2 reported `high` confidence.
- **>=1 accepted:** at least one fingertip was `accepted` by Phase 3.
- **features usable:** a hand was found and at least 2 of the 4 fretting fingers are on the neck area, so the relative features can be computed.
""")
code(r"""
import matplotlib.pyplot as plt

def pct(s): return round(100 * float(np.mean(s)), 1)

by_chord = FEATS.groupby('chord').apply(lambda d: pd.Series({
    'images': len(d), 'hand found %': pct(d['hand_found']), 'geometry high %': pct(d['geom_conf'] == 'high'),
    '>=1 accepted %': pct(d['n_accepted'] > 0), 'features usable %': pct(d['features_usable'])}), include_groups=False)
total = pd.Series({'images': len(FEATS), 'hand found %': pct(FEATS['hand_found']), 'geometry high %': pct(FEATS['geom_conf'] == 'high'),
                   '>=1 accepted %': pct(FEATS['n_accepted'] > 0), 'features usable %': pct(FEATS['features_usable'])}, name='ALL')
REPORT = pd.concat([by_chord, total.to_frame().T])
print(REPORT.to_string())
print('\ncategories:', FEATS['category'].value_counts().to_dict())
print('crop kinds:', FEATS['crop_kind'].value_counts().to_dict())
print('coordinate problems (should be 0):', int(sum(bool(json.loads((FEAT_DIR / f"{i}.json").read_text())['coordinate_problems']) for i in FEATS['id'])))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
by_chord[['hand found %', 'geometry high %', 'features usable %']].plot.bar(ax=ax[0]); ax[0].set_title('availability by chord'); ax[0].set_ylabel('% of images')
FEATS['category'].value_counts().plot.barh(ax=ax[1], color='gray'); ax[1].set_title('Phase 3 image categories')
plt.tight_layout(); plt.show()
""")
code(r"""
# A look at the extraction: six images with the Phase 1-3 result drawn on the ORIGINAL photo
def quick_overlay(item):
    res = json.loads((FEAT_DIR / f"{item['id']}.json").read_text(encoding='utf-8'))
    bgr = load_bgr(item)
    out = draw_result(bgr, res['phase2_geometry'], alpha=0.18)            # Phase 2 drawing (region, strings, frets, nut)
    s = max(out.shape[:2]) / 640.0
    for t in res['fingertips']:
        c = {'accepted': (0, 200, 0), 'tentative': (0, 255, 255)}.get(t['decision'], (160, 160, 160))
        p = (int(round(t['x_px'])), int(round(t['y_px'])))
        cv2.circle(out, p, max(3, int(6 * s)), c, -1, cv2.LINE_AA); cv2.circle(out, p, max(3, int(6 * s)) + 1, (0, 0, 0), 1, cv2.LINE_AA)
    x0, y0, x1, y1, _ = crop_box(res)
    return out[y0:y1, x0:x1], f"{item['id']} | {res['category']}"

pick = [it for it in ITEMS if (FEAT_DIR / f"{it['id']}.json").exists()]
cats = {it['id']: json.loads((FEAT_DIR / f"{it['id']}.json").read_text(encoding='utf-8'))['category'] for it in pick}
show = []
for c in ('success', 'tentative_low_geometry', 'ambiguous', 'outside_or_rejected', 'geometry_failed', 'no_hand'):
    show += [it for it in pick if cats[it['id']] == c][:1]
show = (show + pick)[:6]
ims, ts = zip(*[quick_overlay(it) for it in show])
show_images(list(ims), list(ts), cols=3, size=4.6)
""")

md(r"""
## 6. Save and download
**What:** packs `features.csv` and all per-image JSON files into `p4_features.zip` with a small `manifest.json` (settings, counts, library versions). Notebook 04b reads this zip: if you run 04b **in the same Colab session** it finds the files on disk and you do not need to upload anything; otherwise upload this zip there.
**Reminder:** Colab's disk is erased when the session ends, so download the zip before closing, especially after the long full run.
""")
code(r"""
manifest = dict(n_images=len(FEATS), smoke=SMOKE, data_source=DATA_SOURCE, schema_version=2, n_workers=N_WORKERS,
                counts=FEATS['category'].value_counts().to_dict(), mediapipe=mp.__version__,
                columns=list(FEATS.columns), created=time.strftime('%Y-%m-%d %H:%M:%S'))
(P4 / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
zip_out = shutil.make_archive(str(Path(PROJECT_DIR) / 'p4_features'), 'zip', P4)
print('Created', zip_out, f'({os.path.getsize(zip_out) / 1e6:.1f} MB)')
""")
code(r"""
# colab-only
from google.colab import files
files.download(f'{PROJECT_DIR}/p4_features.zip')
""")

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "04a_extract_features.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "| copied-code fingerprint", DIGEST)
