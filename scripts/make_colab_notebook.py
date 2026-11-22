"""Generates notebooks/05_chord_recognizer_colab.ipynb: ONE self-contained notebook that does everything (Colab or Kaggle).

Usage:  python scripts/make_colab_notebook.py

The library code (neck finder, hand features, straightening, strings/frets) is copied VERBATIM out of
scripts/train_neck_detector.py, scripts/geometry_features.py and scripts/neck_structure.py at generation time, so the notebook
can never drift from the scripts that were tested locally. Cells whose first line is '# colab-only' need a browser (file picker,
camera, downloads); everything else also runs locally (used for the dry-run test).
"""
import ast
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


def grab(path, names):
    """Source text (decorators included) of the named top-level functions / classes / assignments of a script, in file order."""
    src = Path(path).read_text(encoding="utf-8"); lines = src.splitlines(); tree = ast.parse(src)
    found, out = set(), []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nm = {node.name}
        elif isinstance(node, ast.Assign):
            nm = {n.id for t in node.targets for n in (t.elts if isinstance(t, ast.Tuple) else [t]) if isinstance(n, ast.Name)}
        else:
            continue
        if nm & set(names):
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            out.append("\n".join(lines[start - 1:node.end_lineno])); found |= nm
    missing = set(names) - found
    assert not missing, f"{path}: not found {missing}"
    return "\n\n".join(out)


S = ROOT / "scripts"
LIB_NECK = grab(S / "train_neck_detector.py", ["IN_W", "IN_H", "MEAN", "STD", "NeckNet", "soft_argmax", "gaussian_target", "prep", "augment", "poly_iou", "predict"])
LIB_GEO = grab(S / "geometry_features.py", ["TIPS", "FRET_TIPS", "UNIT", "NeckFinder", "neck_geometry", "HandFinder", "pick_fretting_hand", "neck_coords", "feature_vector", "frame_features"])
LIB_STRUCT = grab(S / "neck_structure.py", ["RECT_H", "NOMINAL_Y0", "NOMINAL_D", "FRET_RATIO", "FIRST_GAP", "normalize_quad", "rectify", "_ridge", "find_strings", "find_frets"])
LIB_LIVE = grab(S / "live_pipeline.py", ["TIP_IDS", "Pipeline", "structure_segments", "draw_overlay"])
LIVE_BLOCK = (
    "# ---- real-time pipeline (copied from scripts/live_pipeline.py; the same code runs in scripts/live_local.py)\n"
    + LIB_LIVE + "\n\n\n"
    "def new_pipeline():\n"
    "    return Pipeline(NECK, HAND, CLF, COLS_M, META['smooth_k'], META['threshold'])\n\n\n")
DIGEST = hashlib.sha1((LIB_NECK + LIB_GEO + LIB_STRUCT + LIB_LIVE).encode()).hexdigest()[:10]

# =====================================================================================================================
md(r"""
# Guitar chord recognizer: ONE notebook, everything inside (Colab or Kaggle)

**What it does, from your own videos to a live demo:** it learns to find the guitar neck, turns your fretting hand into numbers *relative to the neck* (so the room and the lighting stop mattering), learns which numbers belong to which chord, tests itself honestly, and then shows the result on a video or on your webcam.

**Chords:** A, Am, Bm, C, D, Dm, E, F, G (9 chords). The code and the data format support all 14; the notebook simply uses the chords it finds.

### How a frame becomes a chord
1. **Neck finder** (a small neural network trained from your clicked corners) outputs the 4 corners of the fretboard.
2. The neck is **straightened** into a flat rectangle, so a guitar at any angle or height looks the same.
3. **MediaPipe** finds the 21 points of your fretting hand.
4. Each hand point gets two numbers in the straightened neck: **how far across** (which strings) and **how far along** (which frets, relative to the hand). That is the finger-position description of the frame.
5. A small **classifier** turns those numbers into a probability per chord. Averaging the last few frames and saying **"not sure"** when it is not confident make it steadier.

### Saved so you never retrain
Everything this notebook produces is written to **Google Drive** (`My Drive/guitar-chord-recognition/`): the cut frames, the per-video features, the neck-finder checkpoints and final model, the chord classifier, and the reports. Run it again later and every finished stage is **loaded, not recomputed**. If Drive cannot be mounted, the notebook falls back to temporary disk and gives you a **Bundle** (zip) cell to download and a **Restore** cell to upload next time.

### What you need
- `own_data.zip` (made by `scripts/package_for_colab.py`) in `My Drive/guitar-chord-recognition/` (or the Colab file panel, or the file picker below).
- **Runtime > Change runtime type > T4 GPU** (it also runs on CPU, slower).
- Internet on (downloads the MediaPipe hand model and the ResNet-18 starting weights).

### Honest expectations (written before running)
About 50% per frame and 60-65% when averaging 5 frames on the 9 chords in the earlier local test (chance is 11%). D, F, C and G are the strongest; A, Am and E are weak. The report at the end prints the numbers measured by this notebook, **only on takes the model did not train on**. A live Colab webcam runs at a few frames per second, not 30.
""")

md(r"""
## 0. Settings
One cell holds every switch. The defaults are what we tested. `FORCE_*` re-does a stage even if its saved result exists. `RESUME` continues an interrupted training from its last checkpoint.
""")
code(r"""
import os, sys, json, time, glob, shutil, zipfile, hashlib, random, subprocess, importlib, collections, math
from pathlib import Path

CHORDS = ['A', 'Am', 'Bm', 'C', 'D', 'Dm', 'E', 'F', 'G']   # the chords of your recordings
FRAME_STEP_S = 0.5                  # one frame every this-many seconds is cut from each video
NECK_EPOCHS = int(os.environ.get('CHORD_NECK_EPOCHS', 120))   # neck-finder training length
NECK_CKPT_EVERY = 10                # save a resumable checkpoint every this-many epochs
NECK_HOLDOUT_TAKE = 'take3'         # labelled frames of this take are NEVER trained on (honest neck test); None = train on all labels
NECK_WARM_START = False             # False = the tested recipe (pre-trained ResNet-18, 120 epochs). True = start from the neck_v1.pt in own_data.zip (untested: a short run can come out WORSE)
RESUME = True                       # continue interrupted training from the last checkpoint
FORCE_FRAMES = FORCE_NECK = FORCE_FEATURES = FORCE_CLASSIFIER = False
USE_PUBLIC_DATA = False             # optional experiment: add the public dataset's photos of the SAME chords (see section 7)
SMOOTH_K = 5                        # the live answer averages the last K frames
ABSTAIN_TARGET_ACC = 0.80           # the "not sure" threshold is the loosest one that reaches this accuracy in cross-validation
OVERLAY_STRUCTURE = True            # draw the 6 strings and the frets on the demo video
SEED = 0
CODE_VERSION = 'v1'                 # bump to invalidate every cached stage
random.seed(SEED)
print('settings loaded | chords:', CHORDS)
""")

md(r"""
## 1. Platform and storage
Detects Colab / Kaggle, tries to mount Google Drive, and falls back to temporary disk with a zip **Bundle** if Drive is not available. `WORK` is the folder where everything is saved; all later cells use it.
""")
code(r"""
def detect_platform():
    if os.environ.get('CHORD_PLATFORM'): return os.environ['CHORD_PLATFORM']
    if os.path.exists('/kaggle/working'): return 'kaggle'
    try:
        import google.colab  # noqa
        return 'colab'
    except ImportError:
        return 'local'
PLATFORM = detect_platform()


def choose_work():
    if os.environ.get('CHORD_WORK'):
        return Path(os.environ['CHORD_WORK']), 'folder from CHORD_WORK (testing)', True
    if PLATFORM == 'colab':
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            p = Path('/content/drive/MyDrive/guitar-chord-recognition'); p.mkdir(parents=True, exist_ok=True)
            t = p / '.write_test'; t.write_text('ok'); t.unlink()
            return p, 'Google Drive (saved between sessions)', True
        except Exception as e:
            print('!! Google Drive is not available (' + type(e).__name__ + '). Using temporary disk.')
            print('!! Download the Bundle (section 11) before closing, and Restore it next time.')
            return Path('/content/work'), 'Colab temporary disk (use Bundle / Restore)', False
    if PLATFORM == 'kaggle':
        return Path('/kaggle/working/work'), 'Kaggle working folder (kept only as notebook output)', False
    return Path('work_local'), 'local folder', True


WORK, WORK_KIND, PERSISTENT = choose_work()
for sub in ['data', 'cache/frames', 'cache/features', 'checkpoints/neck', 'models', 'runs']:
    (WORK / sub).mkdir(parents=True, exist_ok=True)
print(f'platform: {PLATFORM} | saving to: {WORK}  [{WORK_KIND}]')

# ---- stage fingerprints: a saved result is reused only if its settings and inputs are unchanged
STATE_PATH = WORK / 'state.json'
def _state(): return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
def fingerprint(*parts): return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]
def stage_is_done(name, fp, outputs=(), force=False):
    return (not force) and _state().get(name) == fp and all(Path(o).exists() for o in outputs)
def mark_done(name, fp):
    s = _state(); s[name] = fp; tmp = STATE_PATH.with_suffix('.tmp'); tmp.write_text(json.dumps(s, indent=1)); tmp.replace(STATE_PATH)
def file_sig(p):
    # content-based (size + a hash of the first and last 4 MB), NOT the modification time: a file restored from a zip bundle
    # gets a new timestamp, and that must not make the notebook think its inputs changed
    p = Path(p)
    if not p.exists(): return 'none'
    h = hashlib.md5(); size = p.stat().st_size
    with open(p, 'rb') as f:
        h.update(f.read(4_000_000)); f.seek(max(0, size - 4_000_000)); h.update(f.read(4_000_000))
    return f'{size}-{h.hexdigest()[:10]}'
print('stage bookkeeping ready (state file:', STATE_PATH.name + ')')
""")

md(r"""
## 2. Libraries, GPU and downloads
Installs the two packages Colab does not ship (`mediapipe`, `scikit-learn`), checks for a GPU, and downloads the MediaPipe hand model once into `WORK/models/`.
""")
code(r"""
def ensure(pkg, module=None):
    try:
        importlib.import_module(module or pkg)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', pkg], check=True); importlib.invalidate_caches()
ensure('mediapipe'); ensure('scikit-learn', 'sklearn')

import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F, torchvision
import matplotlib; matplotlib.use(matplotlib.get_backend() if PLATFORM != 'local' else 'Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import urllib.request

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch', torch.__version__, '| device:', DEV, '|', torch.cuda.get_device_name(0) if DEV == 'cuda' else 'no GPU (everything still works, only slower)')
HAND_MODEL = WORK / 'models' / 'hand_landmarker.task'
if not HAND_MODEL.exists():
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', HAND_MODEL)
print('hand model:', HAND_MODEL.name, f'{HAND_MODEL.stat().st_size / 1e6:.1f} MB')
""")

md(rf"""
## 3. The code library (copied unchanged from the tested scripts; fingerprint `{DIGEST}`)
These three cells are the neck finder network, the hand-to-neck features, and the straightening / string / fret finders. They are generated from `scripts/train_neck_detector.py`, `scripts/geometry_features.py` and `scripts/neck_structure.py`, so this notebook runs exactly the code that was tested on your laptop. **Do not edit them here.**
""")
code("# ---- neck finder: network, soft-argmax, augmentation, prediction (from scripts/train_neck_detector.py)\n" + LIB_NECK)
code("# ---- straightening, strings, frets (from scripts/neck_structure.py)\n" + LIB_STRUCT)
code("# ---- hand finder, neck sanity check, hand points in neck coordinates (from scripts/geometry_features.py)\n" + LIB_GEO)

md(r"""
## 4. Your data
Finds `own_data.zip` (a copy already in `WORK/data`, your Drive, the Kaggle inputs, or the file picker as a last resort), unpacks it once, and prints how many takes each chord has.
""")
code(r"""
DATA = WORK / 'data'

def find_data_zip():
    cands = [os.environ.get('CHORD_DATA_ZIP', ''), DATA / 'own_data.zip', WORK / 'own_data.zip', '/content/own_data.zip',
             '/content/drive/MyDrive/own_data.zip', '/content/drive/MyDrive/guitar-chord-recognition/own_data.zip']
    cands += glob.glob('/kaggle/input/*/own_data.zip')
    for c in cands:
        if c and Path(c).exists(): return Path(c)
    return None

marker = DATA / '.unpacked'
zp = find_data_zip()
if zp is not None and (not marker.exists() or marker.read_text() != file_sig(zp)):
    print('unpacking', zp, f'({zp.stat().st_size / 1e6:.0f} MB) ...')
    with zipfile.ZipFile(zp) as z: z.extractall(DATA)
    marker.write_text(file_sig(zp))
elif zp is None and not (DATA / 'own_videos').exists():
    if PLATFORM == 'colab':
        from google.colab import files
        print('own_data.zip not found. Choose it in the picker (large files upload faster if you put it in Drive first) ...')
        up = files.upload(); name = [n for n in up if n.endswith('.zip')][0]
        (DATA / 'own_data.zip').write_bytes(up[name])
        with zipfile.ZipFile(DATA / 'own_data.zip') as z: z.extractall(DATA)
        marker.write_text(file_sig(DATA / 'own_data.zip'))
    else:
        raise SystemExit('own_data.zip not found. Put it in ' + str(DATA))

VIDEOS = sorted((DATA / 'own_videos').glob('*/take*.mp4'))
assert VIDEOS, 'no videos found inside own_data.zip'
tbl = pd.Series([v.parent.name for v in VIDEOS]).value_counts().sort_index()
print(f'{len(VIDEOS)} videos, takes per chord:', tbl.to_dict())
missing = [c for c in CHORDS if c not in tbl.index]
if missing: print('!! chords with no video (they will not be recognised):', missing)
NECK_LABELS_FILE = DATA / 'neck_labels.json'
NECK_V1 = DATA / 'neck_v1.pt'
print('neck labels:', NECK_LABELS_FILE.exists(), '| warm-start neck model:', NECK_V1.exists())
""")

md(r"""
## 5. Frames
Cuts one frame every `FRAME_STEP_S` seconds from every video into `WORK/cache/frames/<Chord>/`. A video that is already cut is skipped. (The frame names match the names in your neck labels, e.g. `C_take1_000.jpg`.)
""")
code(r"""
FRAMES = WORK / 'cache' / 'frames'

def cut_frames(vid):
    chord, take = vid.parent.name, vid.stem
    out = FRAMES / chord; out.mkdir(parents=True, exist_ok=True)
    done = out / f'{take}.done'
    if done.exists() and not FORCE_FRAMES: return sorted(out.glob(f'{chord}_{take}_*.jpg'))
    cap = cv2.VideoCapture(str(vid)); fps = cap.get(cv2.CAP_PROP_FPS) or 17.0
    step = max(1, int(round(fps * FRAME_STEP_S))); i = k = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if i % step == 0:
            cv2.imwrite(str(out / f'{chord}_{take}_{k:03d}.jpg'), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]); k += 1
        i += 1
    done.write_text(str(k)); return sorted(out.glob(f'{chord}_{take}_*.jpg'))

t0 = time.time(); n_frames = 0
for v in VIDEOS:
    n_frames += len(cut_frames(v))
print(f'{n_frames} frames ready in {time.time() - t0:.0f}s ({FRAMES})')
""")

md(r"""
## 6. The neck finder (train once, saved, resumable)
Trains the small network that finds the four fretboard corners, from the frames you labelled with the clicking tool. By default it starts from a network pre-trained on everyday photos and trains for 120 epochs (the recipe that was tested: held-out corner error about 18 px, overlap about 0.86). `NECK_WARM_START = True` starts from the `neck_v1.pt` in your upload instead, but that is untested and a short run can come out worse.
- **Honest test:** the labelled frames of `NECK_HOLDOUT_TAKE` are kept out of training and used only to measure how close the predicted corners are to yours (in pixels, and as an overlap score, IoU).
- **Checkpoints:** every `NECK_CKPT_EVERY` epochs the full state (model, optimizer, schedule, random generators, epoch) is saved to `WORK/checkpoints/neck/last.pt`. If Colab disconnects, run the cell again and it **resumes at that epoch**.
- **Saved model:** `WORK/models/neck_v2.pt`. On later runs this stage is skipped.
""")
code(r"""
NECK_MODEL = WORK / 'models' / 'neck_v2.pt'
CKPT = WORK / 'checkpoints' / 'neck' / 'last.pt'
labels_all = {k: v for k, v in json.loads(NECK_LABELS_FILE.read_text()).items() if v} if NECK_LABELS_FILE.exists() else {}
neck_fp = fingerprint('neck', CODE_VERSION, NECK_EPOCHS, NECK_HOLDOUT_TAKE, NECK_WARM_START and NECK_V1.exists(), file_sig(NECK_LABELS_FILE), file_sig(NECK_V1))


def neck_frame_path(name):
    return FRAMES / name.split('_')[0] / name

def _atomic_save(obj, path):
    tmp = Path(str(path) + '.tmp'); torch.save(obj, tmp); tmp.replace(path)


def train_neck():
    keys = [k for k in labels_all if neck_frame_path(k).exists()]
    test = sorted(k for k in keys if NECK_HOLDOUT_TAKE and f'_{NECK_HOLDOUT_TAKE}_' in k)
    train = sorted(k for k in keys if k not in test)
    assert train, 'no labelled frames found: check neck_labels.json and the frames of those chords'
    data = {k: (cv2.imread(str(neck_frame_path(k))), np.array(labels_all[k], np.float64)) for k in keys}
    print(f'{len(train)} training frames, {len(test)} held-out frames ({NECK_HOLDOUT_TAKE})')
    warm = NECK_WARM_START and NECK_V1.exists()
    model = NeckNet(pretrained=not warm)
    if warm: model.load_state_dict(torch.load(NECK_V1, map_location='cpu', weights_only=False)['model']); print('warm start from neck_v1.pt')
    model.to(DEV)
    bb = ('stem', 'l1', 'l2', 'l3', 'l4')
    opt = torch.optim.AdamW([{'params': [p for n, p in model.named_parameters() if n.startswith(bb)], 'lr': 3e-4},
                             {'params': [p for n, p in model.named_parameters() if not n.startswith(bb)], 'lr': 1e-3}], weight_decay=1e-4)
    steps = math.ceil(len(train) / 8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-4, 1e-3], total_steps=NECK_EPOCHS * steps)
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED); start = 1
    if RESUME and CKPT.exists() and not FORCE_NECK:
        ck = torch.load(CKPT, map_location='cpu', weights_only=False)
        if ck.get('fp') == neck_fp:
            model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
            rng.bit_generator.state = ck['rng']; random.setstate(ck['py_rng']); torch.set_rng_state(ck['torch_rng'])
            start = ck['epoch'] + 1; print(f'RESUMING from the checkpoint at epoch {ck["epoch"]} of {NECK_EPOCHS}')
        else:
            print('a checkpoint exists but its settings differ: training from the start')
    t0 = time.time()
    for ep in range(start, NECK_EPOCHS + 1):
        model.train(); order = list(train); random.shuffle(order); tot = 0.0
        for i in range(0, len(order), 8):
            xs, ps = [], []
            for k in order[i:i + 8]:
                img, p = augment(data[k][0], data[k][1], rng); H, W = img.shape[:2]
                xs.append(prep(img)); ps.append(torch.tensor(np.clip(p / np.array([W, H]), 0.0, 1.0), dtype=torch.float32))
            x, tgt = torch.stack(xs).to(DEV), torch.stack(ps)
            logits = model(x); co, prob = soft_argmax(logits)
            heat = gaussian_target(tgt, logits.shape[2], logits.shape[3]).to(DEV)
            kl = (heat * (torch.log(heat.clamp_min(1e-8)) - torch.log(prob.clamp_min(1e-8)))).sum((-1, -2)).mean()
            loss = F.smooth_l1_loss(co, tgt.to(DEV), beta=0.02) * 20 + 0.1 * kl
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(order[i:i + 8])
        if ep % NECK_CKPT_EVERY == 0 or ep == NECK_EPOCHS or ep == start:
            msg = f'epoch {ep:3d}/{NECK_EPOCHS}  loss {tot / len(train):.4f}  ({time.time() - t0:.0f}s)'
            if test:
                errs = [np.linalg.norm(predict(model, data[k][0])[0] - data[k][1], axis=1).mean() for k in test]
                msg += f'  | held-out corner error {np.mean(errs):.1f} px'
            print(msg, flush=True)
        if ep % NECK_CKPT_EVERY == 0 and ep < NECK_EPOCHS:
            _atomic_save({'fp': neck_fp, 'epoch': ep, 'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                          'rng': rng.bit_generator.state, 'py_rng': random.getstate(), 'torch_rng': torch.get_rng_state()}, CKPT)
    model.cpu()
    _atomic_save({'model': model.state_dict(), 'in_size': (IN_W, IN_H), 'test_take': NECK_HOLDOUT_TAKE, 'train': train, 'test': test}, NECK_MODEL)
    if CKPT.exists(): CKPT.unlink()
    if test:
        res = [(np.linalg.norm(predict(model.to(DEV), data[k][0])[0] - data[k][1], axis=1).mean(), poly_iou(predict(model, data[k][0])[0], data[k][1])) for k in test]
        e, io = np.array([r[0] for r in res]), np.array([r[1] for r in res])
        print(f'\nNECK FINDER on {len(test)} frames it never trained on: median corner error {np.median(e):.1f} px, mean {e.mean():.1f} px (frame is about 1280x720) | overlap IoU mean {io.mean():.2f}, > 0.7 in {100 * (io > 0.7).mean():.0f}% of frames')
    mark_done('neck', neck_fp)


if stage_is_done('neck', neck_fp, [NECK_MODEL], FORCE_NECK):
    print('neck finder: loaded from', NECK_MODEL, '(already trained, skipping training)')
else:
    train_neck()
NECK = NeckFinder(NECK_MODEL, DEV)
HAND = HandFinder(HAND_MODEL)
print('neck finder and hand finder ready on', DEV)
""")

md(r"""
## 7. Finger positions for every frame
For each video, every frame goes through: neck corners -> sanity check (a real neck is long, thin and convex) -> straighten -> MediaPipe hand -> 21 hand points in neck coordinates -> 42 numbers (21 across-neck positions and 21 along-neck positions relative to the fingertips' average). Results are saved **per video** in `WORK/cache/features/`, so an interrupted run only redoes the unfinished videos.
""")
code(r"""
FEAT = WORK / 'cache' / 'features'
feat_fp = fingerprint('features', CODE_VERSION, file_sig(NECK_MODEL), FRAME_STEP_S)


def video_features(vid):
    chord, take = vid.parent.name, vid.stem
    path = FEAT / f'{chord}_{take}.npz'
    if path.exists() and not FORCE_FEATURES:
        z = np.load(path, allow_pickle=True)
        if str(z['fp']) == feat_fp: return z
    frames = sorted((FRAMES / chord).glob(f'{chord}_{take}_*.jpg')); n = len(frames)
    ids = []; feat = np.full((n, 42), np.nan); nv = np.zeros(n, bool); hf = np.zeros(n, bool); ins = np.full(n, np.nan); cor = np.zeros((n, 4, 2))
    for i, f in enumerate(frames):
        r = frame_features(cv2.imread(str(f)), NECK, HAND)
        ids.append(f.stem); nv[i] = r['neck_valid']; hf[i] = r['hand_found']; ins[i] = r['inside']; cor[i] = r['corners']
        if r['feat'] is not None: feat[i] = r['feat']
    np.savez_compressed(path, ids=np.array(ids), feat=feat, neck_valid=nv, hand_found=hf, inside=ins, corners=cor, fp=feat_fp)
    return np.load(path, allow_pickle=True)


t0 = time.time(); rows = []
for v in VIDEOS:
    z = video_features(v)
    for i, fid in enumerate(z['ids']):
        rows.append(dict(id=str(fid), chord=v.parent.name, take=v.stem, frame=i, usable=bool(np.isfinite(z['feat'][i]).all()),
                         neck_valid=bool(z['neck_valid'][i]), hand_found=bool(z['hand_found'][i]),
                         **{f'f{j}': float(z['feat'][i][j]) for j in range(42)}))
ALL = pd.DataFrame(rows)
mark_done('features', feat_fp)
print(f'{len(ALL)} frames, {int(ALL.usable.sum())} usable ({100 * ALL.usable.mean():.0f}%) in {time.time() - t0:.0f}s')
print(ALL.groupby('chord').agg(takes=('take', 'nunique'), frames=('id', 'size'), neck_ok=('neck_valid', 'mean'), hand=('hand_found', 'mean'),
                               usable=('usable', 'sum')).round(2).to_string())
""")

md(r"""
## 8. The chord classifier (choose by honest testing, then save)
**How the choice is made:** every combination of *feature set* (across-neck only, fingertips only, across + along), *model* (random forest, logistic regression, nearest neighbours) and *jitter augmentation* (on or off) is tested with **leave-one-take-out cross-validation**: train on the other takes, test on the take that was held out, repeat for each take. Nothing is ever tested on frames it trained on. The best combination is retrained on all takes and saved.
- **Frame averaging:** the answer averages the probabilities of the last `SMOOTH_K` frames of a sequence.
- **"Not sure" threshold:** the loosest confidence threshold at which the answered frames reach `ABSTAIN_TARGET_ACC` in cross-validation (at least 30% of frames answered).
""")
code(r"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

TIP_IDX = [4, 8, 12, 16, 20]
FEATURE_SETS = {'across-neck, all 21 points': list(range(21)), 'across-neck, 5 fingertips': TIP_IDX,
                'across + along, 5 fingertips': TIP_IDX + [21 + i for i in TIP_IDX], 'across + along, all 21 points': list(range(42))}
MODELS = {'random forest': lambda: RandomForestClassifier(300, random_state=SEED, min_samples_leaf=2, n_jobs=-1),
          'logistic regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)),
          'nearest neighbours': lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5))}
U = ALL[ALL.usable & ALL.chord.isin(CHORDS)].reset_index(drop=True)
CLASSES = sorted(U.chord.unique())
FCOLS = [f'f{j}' for j in range(42)]
print(f'{len(U)} usable frames, {len(CLASSES)} chords, chance = {100 / len(CLASSES):.0f}%')


def augment_rows(X, y, times=3, sigma=0.03, seed=SEED):
    r = np.random.default_rng(seed); return np.vstack([X] + [X + r.normal(0, sigma, X.shape) for _ in range(times)]), np.concatenate([y] * (times + 1))


def fit_model(mname, X, y, jitter):
    if jitter: X, y = augment_rows(X, y)
    return MODELS[mname]().fit(X, y)


def smooth_accuracy(P, y, groups, k):
    ok = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for j in range(len(idx)):
            ok.append(CLASSES[int(P[idx[max(0, j - k + 1): j + 1]].mean(0).argmax())] == y[idx[j]])
    return float(np.mean(ok))


def cross_validate(fset, mname, jitter, extra=None):
    cols = [FCOLS[i] for i in FEATURE_SETS[fset]]
    P = np.zeros((len(U), len(CLASSES)))
    for t in sorted(U['take'].unique()):
        tr, te = U['take'] != t, U['take'] == t
        if not te.any(): continue
        Xtr, ytr = U.loc[tr, cols].values, U.loc[tr, 'chord'].values
        if extra is not None: Xtr, ytr = np.vstack([Xtr, extra[cols].values]), np.concatenate([ytr, extra['chord'].values])
        m = fit_model(mname, Xtr, ytr, jitter); pr = m.predict_proba(U.loc[te, cols].values)
        P[np.where(te)[0][:, None], [CLASSES.index(c) for c in m.classes_]] = pr
    groups = (U.chord + '_' + U['take']).values; y = U.chord.values
    acc1 = float(np.mean(np.array(CLASSES)[P.argmax(1)] == y))
    return P, acc1, smooth_accuracy(P, y, groups, SMOOTH_K), groups


clf_fp = fingerprint('clf', CODE_VERSION, feat_fp, SMOOTH_K, ABSTAIN_TARGET_ACC, USE_PUBLIC_DATA, len(U))
CLF_PATH, META_PATH = WORK / 'models' / 'chord_clf.joblib', WORK / 'models' / 'chord_meta.json'
if stage_is_done('classifier', clf_fp, [CLF_PATH, META_PATH], FORCE_CLASSIFIER):
    BUNDLE = joblib.load(CLF_PATH); META = json.loads(META_PATH.read_text())
    print('chord classifier: loaded from', CLF_PATH.name, '| chosen:', META['features'], '+', META['model'], '| jitter', META['jitter'])
else:
    results = []; t0 = time.time()
    for fset in FEATURE_SETS:
        for mname in MODELS:
            for jitter in (False, True):
                P, a1, ak, groups = cross_validate(fset, mname, jitter)
                results.append(dict(features=fset, model=mname, jitter=jitter, per_frame=a1, averaged=ak, score=(a1 + ak) / 2, P=P))
    R = pd.DataFrame([{k: v for k, v in r.items() if k != 'P'} for r in results]).sort_values('score', ascending=False)
    print(f'cross-validation of {len(results)} combinations in {time.time() - t0:.0f}s (leave-one-take-out; accuracy of {len(CLASSES)} chords, chance {100 / len(CLASSES):.0f}%):')
    print(R.head(8).assign(per_frame=lambda d: (100 * d.per_frame).round(1), averaged=lambda d: (100 * d.averaged).round(1), score=lambda d: (100 * d.score).round(1)).to_string(index=False))
    best = max(results, key=lambda r: r['score'])
    P, y = best['P'], U.chord.values
    # "not sure" threshold from the cross-validated, averaged probabilities
    groups = (U.chord + '_' + U['take']).values; Pa = np.zeros_like(P)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for j in range(len(idx)): Pa[idx[j]] = P[idx[max(0, j - SMOOTH_K + 1): j + 1]].mean(0)
    thr, cov_at = 0.0, 1.0
    for th in np.arange(0.0, 0.96, 0.02):
        keep = Pa.max(1) >= th
        if keep.mean() >= 0.30 and np.mean(np.array(CLASSES)[Pa.argmax(1)][keep] == y[keep]) >= ABSTAIN_TARGET_ACC: thr, cov_at = float(th), float(keep.mean()); break
    cols = [FCOLS[i] for i in FEATURE_SETS[best['features']]]
    final = fit_model(best['model'], U[cols].values, U.chord.values, best['jitter'])
    META = dict(features=best['features'], model=best['model'], jitter=bool(best['jitter']), columns=cols, classes=CLASSES, smooth_k=SMOOTH_K,
                threshold=thr, coverage_at_threshold=cov_at, cv_per_frame=best['per_frame'], cv_averaged=best['averaged'], target_acc=ABSTAIN_TARGET_ACC)
    BUNDLE = dict(model=final, meta=META)
    joblib.dump(BUNDLE, CLF_PATH); META_PATH.write_text(json.dumps(META, indent=1)); mark_done('classifier', clf_fp)
    OOF = dict(P=P, Pa=Pa, y=y, groups=groups, take=U['take'].values)
    print(f"\nCHOSEN: {best['features']} + {best['model']} + jitter {best['jitter']} | not-sure threshold {thr:.2f} (answers {100 * cov_at:.0f}% of frames at >= {100 * ABSTAIN_TARGET_ACC:.0f}% accuracy)" if thr > 0 else
          f"\nCHOSEN: {best['features']} + {best['model']} + jitter {best['jitter']} | the {100 * ABSTAIN_TARGET_ACC:.0f}% target is not reachable, so no 'not sure' threshold is applied")
    print('saved:', CLF_PATH.name, META_PATH.name)
""")

md(r"""
## 9. Optional experiment: more training data from the public dataset (same chords only)
Switch `USE_PUBLIC_DATA = True` in section 0. The notebook downloads the public chord photos (inside Colab), keeps **only the chords you recorded**, runs them through the same neck-finder -> hand -> features pipeline, and compares two cross-validations on **your** held-out takes: *your videos only* versus *your videos + the public photos*. The photos are kept **only if they improve the score by at least one point**. Many public photos are rotated or colour-inverted, so the neck and hand are found in fewer of them.
""")
code(r"""
PUB_FILE = FEAT / 'public_same_chords.npz'
PUBLIC = None
if USE_PUBLIC_DATA:
    ensure('datasets')
    if PUB_FILE.exists() and str(np.load(PUB_FILE, allow_pickle=True)['fp']) == feat_fp:
        z = np.load(PUB_FILE, allow_pickle=True); PUBLIC = pd.DataFrame(z['feat'], columns=FCOLS); PUBLIC['chord'] = z['chord']
    else:
        from datasets import load_dataset
        R_URL = 'hf://datasets/j-asra-j/guitar-chords@refs%2Fconvert%2Fparquet/default/'
        ds = load_dataset('parquet', data_files={'train': [R_URL + 'train/0000.parquet', R_URL + 'train/0001.parquet'], 'validation': R_URL + 'validation/0000.parquet', 'test': R_URL + 'test/0000.parquet'})
        names = ds['train'].features['label'].names; feats, chs = [], []; t0 = time.time()
        for sp in ('train', 'validation', 'test'):
            for row in ds[sp]:
                ch = names[row['label']]
                if ch not in CHORDS: continue
                r = frame_features(cv2.cvtColor(np.array(row['image'].convert('RGB')), cv2.COLOR_RGB2BGR), NECK, HAND)
                if r['feat'] is not None and np.isfinite(r['feat']).all(): feats.append(r['feat']); chs.append(ch)
        np.savez_compressed(PUB_FILE, feat=np.array(feats), chord=np.array(chs), fp=feat_fp)
        PUBLIC = pd.DataFrame(np.array(feats), columns=FCOLS); PUBLIC['chord'] = chs
        print(f'public photos: {len(PUBLIC)} usable ({time.time() - t0:.0f}s)')
    base_cfg = (META['features'], META['model'], META['jitter'])
    _, a_own, k_own, _ = cross_validate(*base_cfg); _, a_pub, k_pub, _ = cross_validate(*base_cfg, extra=PUBLIC)
    gain = (a_pub + k_pub) / 2 - (a_own + k_own) / 2
    print(f'your videos only      : per frame {100 * a_own:.1f}%  averaged {100 * k_own:.1f}%')
    print(f'+ public photos ({len(PUBLIC)}) : per frame {100 * a_pub:.1f}%  averaged {100 * k_pub:.1f}%   (change {100 * gain:+.1f} points)')
    if gain >= 0.01 and not META.get('uses_public_data'):
        cols_p = META['columns']
        Xp = np.vstack([U[cols_p].values, PUBLIC[cols_p].values]); yp = np.concatenate([U.chord.values, PUBLIC.chord.values])
        BUNDLE = dict(model=fit_model(META['model'], Xp, yp, META['jitter']), meta=META)
        META.update(uses_public_data=True, cv_per_frame=a_pub, cv_averaged=k_pub, public_photos=int(len(PUBLIC)))
        joblib.dump(BUNDLE, CLF_PATH); META_PATH.write_text(json.dumps(META, indent=1))
        CLF = BUNDLE['model']
        print('-> the public photos HELP: the saved classifier was retrained on your videos + the public photos (same chords only).')
    elif gain >= 0.01:
        print('-> the public photos help; the saved classifier already includes them.')
    else:
        print('-> the public photos do NOT help, so they are not used.')
else:
    print('skipped (USE_PUBLIC_DATA is False)')
""")

md(r"""
## 10. Honest report
Everything below is measured **only on takes the model did not train on** (leave-one-take-out). It prints the accuracy per frame and with frame averaging, the recall of each chord, which chords get mixed up, and saves everything to `WORK/runs/<time>/`.
""")
code(r"""
from sklearn.metrics import confusion_matrix
cols = META['columns']; best_cfg = (META['features'], META['model'], META['jitter'])
P, a1, ak, groups = cross_validate(*best_cfg, extra=PUBLIC if (META.get('uses_public_data') and PUBLIC is not None) else None)
y = U.chord.values; pred = np.array(CLASSES)[P.argmax(1)]
cm = confusion_matrix(y, pred, labels=CLASSES)
recall = {c: cm[i, i] / max(cm[i].sum(), 1) for i, c in enumerate(CLASSES)}
frames_per = U.chord.value_counts().to_dict(); takes_per = U.groupby('chord')['take'].nunique().to_dict()
usable_per = ALL.groupby('chord').usable.mean().round(2).to_dict()
mix = sorted(((int(cm[i, j]), CLASSES[i], CLASSES[j]) for i in range(len(CLASSES)) for j in range(len(CLASSES)) if i != j), reverse=True)[:6]
print(f"{len(CLASSES)} chords | chance {100 / len(CLASSES):.0f}% | features: {META['features']} | model: {META['model']}")
print(f'accuracy on held-out takes: per frame {100 * a1:.1f}% | averaged over the last {SMOOTH_K} frames {100 * ak:.1f}%')
print(pd.DataFrame({'recall %': {c: round(100 * r) for c, r in recall.items()}, 'takes': takes_per, 'usable frames': frames_per, 'frames usable %': {c: round(100 * usable_per.get(c, 0)) for c in CLASSES}}).T.to_string())
print('most common mix-ups (count, true -> predicted):', [(n, f'{a}->{b}') for n, a, b in mix])
weak = [c for c, r in recall.items() if r < 0.4]
print('chords the system is NOT reliable on:', weak if weak else 'none below 40% recall')
print('reference "before" (earlier experiment): the whole-image CNN trained on the public photos scored 2.4% on these videos')
fig, ax = plt.subplots(figsize=(6.5, 5.5)); ax.imshow(cm, cmap='Blues'); ax.set_xticks(range(len(CLASSES)), CLASSES); ax.set_yticks(range(len(CLASSES)), CLASSES)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        if cm[i, j]: ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8, color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title('held-out takes: confusion matrix'); plt.tight_layout()
RUN = WORK / 'runs' / time.strftime('%Y%m%d_%H%M%S'); RUN.mkdir(parents=True, exist_ok=True)
fig.savefig(RUN / 'confusion.png', dpi=130); plt.show() if PLATFORM != 'local' else plt.close(fig)
pd.DataFrame({'recall': recall, 'takes': takes_per, 'usable_frames': frames_per}).to_csv(RUN / 'per_chord.csv')
(RUN / 'metrics.json').write_text(json.dumps(dict(chords=CLASSES, per_frame=a1, averaged=ak, smooth_k=SMOOTH_K, recall=recall, meta=META, mixups=[[n, a, b] for n, a, b in mix],
                                                   created=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1))
print('saved report to', RUN)
""")

md(r"""
## 11. Demo A: annotated video (works on Colab and Kaggle)
Runs the whole system on a video and writes a new video with the neck outline, the 6 strings and the frets (found inside the straightened neck), the hand points, and the chord with its confidence, or **"not sure"**. The neck finder runs on **every frame** and its corners are smoothed over time, so there is **no re-lock button**: if the neck moves, the outline follows it. The rendered video plays at the video's own frame rate, which is the smooth look for a presentation. On Colab you can also upload your own test video in the next cell.
""")
code(r"""
CLASSES_M = META['classes']; CLF = BUNDLE['model']; COLS_M = [FCOLS.index(c) for c in META['columns']]


""" + LIVE_BLOCK + r"""def annotate_video(src, dst, max_seconds=None, structure=OVERLAY_STRUCTURE):
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 17.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = str(dst) + '.tmp.mp4'; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    pipe = new_pipeline(); n = 0; t0 = time.time(); answers = []
    while True:
        ok, bgr = cap.read()
        if not ok or (max_seconds and n / fps > max_seconds): break
        out = pipe.step(bgr); vw.write(draw_overlay(bgr, out, None, structure)); n += 1
        if out['answer']: answers.append(out['answer'])
    vw.release()
    if shutil.which('ffmpeg'):                                   # H.264 plays in browsers; mp4v often does not
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', str(dst)]); os.remove(tmp)
    else: os.replace(tmp, dst)
    top = collections.Counter(answers).most_common(1)
    print(f'{n} frames in {time.time() - t0:.0f}s ({n / (time.time() - t0):.1f} frames/s of processing) -> {dst} | most frequent answer: {top[0] if top else "none"}')
    return dst


DEMO_DIR = RUN; demo_src = next((v for v in VIDEOS if v.parent.name == CLASSES_M[0]), VIDEOS[0])
DEMO_OUT = annotate_video(demo_src, DEMO_DIR / f'demo_{demo_src.parent.name}_{demo_src.stem}.mp4', max_seconds=10)
print('self-test video written. (It uses a video the model trained on, so it only checks that the pipeline runs end to end.)')
""")
code(r"""
# colab-only
# Upload your own test video and get an annotated copy (for the presentation).
from google.colab import files
from IPython.display import Video, display
up = files.upload()
for name, data in up.items():
    p = WORK / 'data' / name; p.write_bytes(data)
    out = annotate_video(p, RUN / ('annotated_' + Path(name).stem + '.mp4'))
    display(Video(str(out), embed=True, width=720)); files.download(str(out))
""")

md(r"""
## 12. Demo B: live webcam in Colab
The neck finder runs on **every frame** and smooths the corners, so the neck finds itself and follows the guitar (no re-lock button). Then MediaPipe, the classifier, and an average over the last frames. Click **Allow** when the browser asks for the camera, show your fretting hand and the neck, and hold each chord for a second or two. **Expect a few frames per second:** every frame travels browser -> Colab -> browser, and that bridge, not the models, is the limit (a T4 handles the models in tens of milliseconds). For a smooth 30 fps look use the annotated video of section 11. The first line printed below is the measured speed of the models alone.
""")
code(r"""
# how fast are the models on this machine (without the browser bridge)?
_probe = cv2.imread(str(next((FRAMES / CLASSES_M[0]).glob('*.jpg')))); _probe = cv2.resize(_probe, (640, 360)); _p = new_pipeline(); _p.step(_probe)
t0 = time.time()
for _ in range(10): _p.step(_probe)
ms = (time.time() - t0) / 10 * 1000
print(f'models alone: {ms:.0f} ms per frame -> up to about {1000 / ms:.0f} frames per second; the Colab webcam bridge will make the live view slower than that')
""")
code(r"""
# colab-only
import base64
from google.colab import output
from IPython.display import display, Javascript, JSON

LIVE = new_pipeline(); _t = {'last': time.time(), 'fps': 0.0, 'n': 0}

def process_frame(data_url):
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(data_url.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
    bgr = cv2.resize(bgr, (640, 360)) if bgr.shape[1] > 640 else bgr
    out = LIVE.step(bgr); now = time.time(); _t['n'] += 1
    _t['fps'] = 0.85 * _t['fps'] + 0.15 / max(now - _t['last'], 1e-3) if _t['n'] > 1 else 0.0; _t['last'] = now
    img = draw_overlay(bgr, out, _t['fps'], structure=False)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    text = 'no hand / no neck' if out['best'] is None else (f"{out['answer']} ({out['conf']:.2f})" if out['answer'] else 'not sure')
    return JSON({'img': 'data:image/jpeg;base64,' + base64.b64encode(buf).decode(), 'text': text + f'   |   {_t["fps"]:.1f} fps'})

output.register_callback('notebook.chord_frame', process_frame)
JS = '''
async function run() {
  const box = document.createElement("div"); const stop = document.createElement("button"); stop.textContent = "Stop camera"; stop.style.cssText = "font-size:16px;padding:6px 14px";
  const status = document.createElement("div"); status.style.cssText = "font:18px sans-serif;margin:8px 0"; const img = document.createElement("img"); img.style.maxWidth = "100%";
  box.append(stop, status, img); document.querySelector("#output-area").appendChild(box); let halt = false; stop.onclick = () => { halt = true; };
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 360}});
    const video = document.createElement("video"); video.srcObject = stream; video.muted = true; await video.play();
    const canvas = document.createElement("canvas"); canvas.width = video.videoWidth; canvas.height = video.videoHeight; const ctx = canvas.getContext("2d");
    while (!halt) {
      ctx.drawImage(video, 0, 0);
      const res = await google.colab.kernel.invokeFunction("notebook.chord_frame", [canvas.toDataURL("image/jpeg", 0.6)], {});
      const d = res.data["application/json"]; img.src = d.img; status.textContent = d.text;
    }
    stream.getTracks().forEach(t => t.stop()); status.textContent = "Camera stopped.";
  } catch (e) { status.textContent = "Camera error: " + e; }
}
run();
'''
display(Javascript(JS))
""")

md(r"""
## 13. Bundle / restore (the no-Drive fallback) and a summary of what is saved
If Drive was not available, download the **Bundle** before the session ends (models, checkpoints, per-video features, reports; not the big videos or frames) and upload it with **Restore** at the start of the next session, before section 6. With Drive, the files are already saved and you can skip this.
""")
code(r"""
def make_bundle():
    out = WORK / 'bundle.zip'
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for sub in ['models', 'checkpoints', 'cache/features', 'runs']:
            for f in (WORK / sub).rglob('*'):
                if f.is_file() and f.name != 'hand_landmarker.task': z.write(f, f.relative_to(WORK))
        z.write(STATE_PATH, 'state.json') if STATE_PATH.exists() else None
    return out

print('what is saved in', WORK)
for sub in ['models', 'checkpoints/neck', 'cache/features', 'runs']:
    fs = [f for f in (WORK / sub).rglob('*') if f.is_file()]
    print(f'  {sub:18s} {len(fs):3d} files  {sum(f.stat().st_size for f in fs) / 1e6:7.1f} MB')
print('persistent between sessions:', PERSISTENT)
""")
code(r"""
# colab-only
# BUNDLE: download everything needed to skip training next time (use this if Google Drive was not available)
from google.colab import files
b = make_bundle(); print('bundle:', b, f'{b.stat().st_size / 1e6:.0f} MB'); files.download(str(b))
""")
code(r"""
# colab-only
# RESTORE: upload a bundle.zip from an earlier session (run this BEFORE section 6 in a new session)
from google.colab import files
up = files.upload()
for name, data in up.items():
    if name.endswith('.zip'):
        (WORK / 'restore.zip').write_bytes(data)
        with zipfile.ZipFile(WORK / 'restore.zip') as z: z.extractall(WORK)
        print('restored', name, '-> models, checkpoints, features and reports are back')
""")

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "05_chord_recognizer_colab.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "| library fingerprint", DIGEST, "|", sum(1 for c in cells if c["cell_type"] == "code"), "code cells")
