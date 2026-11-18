"""Generates notebooks/04c_evaluate_and_demo.ipynb (Phase 4, step c: honest evaluation, own-photo test, live demo).

Usage: python scripts/make_eval_notebook.py   (run make_features_notebook.py and make_train_notebook.py first)

Code is copied verbatim from the earlier notebooks, never re-written:
  Phase 1-3 functions and the feature-row code  <- notebooks/04a_extract_features.ipynb
  metrics / fusion / coverage helpers          <- notebooks/04b_train_models.ipynb
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


def load_cells(name):
    nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


SRC4A, SRC4B = load_cells("04a_extract_features.ipynb"), load_cells("04b_train_models.ipynb")


def pick(srcs, marker):
    hit = [s for s in srcs if marker in s]
    assert hit, f"no cell containing {marker!r}"
    return hit[0]


def cut(s, start=None, end=None):
    if start:
        assert start in s, start
        s = s[s.index(start):]
    if end:
        assert end in s, end
        s = s[:s.index(end)]
    return s.rstrip() + "\n"


copied_cfg = pick(SRC4A, "Phase 1/3 settings and imports: copied from")       # settings + MediaPipe helpers + detect_hands
copied_p2 = pick(SRC4A, "# Phase 2 detector: copied from")
copied_p3 = pick(SRC4A, "# Phase 3 geometry and mapping: copied from")
feature_code = cut(pick(SRC4A, "def crop_box(res):"), start="FINGERS = ['Index'", end="\n\nrows = []")
helpers_code = pick(SRC4B, "def group_by_similarity")
DIGEST = hashlib.sha1((copied_cfg + copied_p2 + copied_p3 + feature_code + helpers_code).encode("utf-8")).hexdigest()[:10]

# ---------------------------------------------------------------------------------------------------------------
md(r"""
# Notebook 04c: Honest evaluation, your own photos, and the live demo (Phase 4, step c)

**Goal:** look at what the models from 04b can *really* do, and let you try them yourself.
1. **Report** the group-split test results for every model, with the confusion matrix, per-chord scores, and the coverage-vs-accuracy trade-off.
2. **Error gallery:** wrong predictions drawn with the Phase 1-3 overlay, so each mistake can be traced to a cause (a missed hand, a wrong neck, a genuinely ambiguous chord).
3. **Your own photos** (optional): the number that tells you whether this works for *your* guitar and camera. Models never saw these photos.
4. **Live demo** with your webcam: chord prediction on the live video, with smoothing and a "not sure" answer.

**Not in this notebook:** more training or any accuracy tuning (that is the next phase).

### How a photo becomes a chord (the whole pipeline in one picture)
```
photo --> Phase 1 MediaPipe --> fingertips ----+
      \-> Phase 2 OpenCV    --> neck lines     +--> Phase 3 --> fingertip (string, fret) numbers --> Model B (trees) -+
                                                                                                                    +--> blend --> chord, or "not sure"
photo --> crop around neck + hand (from the same boxes) --> Model A2 (CNN) ----------------------------------------+
```
The blend trusts Model B only as much as Phase 3 was confident, and uses the CNN alone when no fingertip features exist.

### Needs
`p4_models.zip` from notebook 04b (found automatically if 04b ran in this same Colab session, otherwise a file picker asks for it). A GPU runtime is recommended for the CNNs, but inference also works on CPU.
""")

md(r"""
## 0. Setup
**What:** installs `mediapipe` and `datasets` if missing, and prepares folders. No Google Drive.
""")
code(r"""
# colab-only
import os
PROJECT_DIR = '/content/Guitar-Chord-Recognition'
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
import torch
print('mediapipe', mp.__version__, '| torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU is fine for inference)')
""")

md(r"""
## 1. Load the trained models and the Phase 1-3 code
**What:** loads the artifacts from 04b (`A1_group.pt`, `A2_crop.pt`, `model_b.joblib`, `fusion.json`, `predictions.npz`), then the same Phase 1-3 function code used in 04a (copied unchanged: notebook 03's mapping, which copies notebook 02's detector).
**Why the same code:** the model was trained on features produced by exactly this code. Using a different version at test time would silently shift the numbers.
**Note:** if 04b ran as a *rehearsal* (small smoke data), `fusion.json` says so and everything below is a pipeline check, not a result.
""")
code(r"""
import json, time, shutil, zipfile, collections, base64, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

P4 = Path(PROJECT_DIR) / 'outputs' / 'phase4'
MODELS = P4 / 'models'
if not (MODELS / 'fusion.json').exists():
    from google.colab import files
    print('Choose p4_models.zip (made by notebook 04b) ...')
    up = files.upload()
    MODELS.mkdir(parents=True, exist_ok=True)
    zp = Path(PROJECT_DIR) / 'p4_models.zip'
    zp.write_bytes(up[[n for n in up if n.lower().endswith('.zip')][0]])
    with zipfile.ZipFile(zp) as z:
        z.extractall(MODELS)
SET = json.loads((MODELS / 'fusion.json').read_text(encoding='utf-8'))
NAMES, BASE, ALPHA, THRESH, B_COLS = SET['classes'], SET['base_cnn'], SET['alpha'], SET['abstain_threshold'], SET['b_columns']
print('chords:', NAMES)
print(f"blend CNN: {BASE} | alpha {ALPHA} | abstain below {THRESH} | trained as {'REHEARSAL (numbers are not meaningful)' if SET['rehearsal'] else 'FULL run'}")
""")
code(copied_cfg.replace("# Phase 1/3 settings and imports: copied from", "# Phase 1/3 settings and imports: copied from", 1))
code(copied_p2)
code(copied_p3)
code(feature_code)
code(helpers_code)

md(r"""
## 2. Model loading and one-image prediction
**What:** `predict_image(bgr)` is the **whole system** for one photo: Phase 1 (hand), Phase 2 (neck), Phase 3 (finger positions), feature row, crop, the CNN probabilities, Model B probabilities, the blend, and the final answer.
- `chord` is the most likely chord, or `None` (shown as "not sure") when the blended top probability is below the abstention threshold chosen in 04b.
- `used` says which parts contributed: `cnn only` (no usable fingertip features) or `cnn + features`.
**Technique: softmax probabilities.** Each model ends with 14 scores that are turned into probabilities summing to 1. The highest one is the guess and its value is the confidence. A model can be confidently wrong, which is why we also measure coverage vs accuracy instead of trusting the confidence blindly.
""")
code(r"""
import torch.nn as nn
import joblib
from torchvision import models, transforms as T
from PIL import Image

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
S = SET['img_size']
eval_tf = T.Compose([T.Resize((S, S)), T.ToTensor(), T.Normalize(SET['mean'], SET['std'])])


def make_landmarker():
    # Exactly the Phase 1 settings: image mode, up to 2 hands, all confidences 0.3 (`detect_hands` uses the global LANDMARKER).
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH), running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3))
LANDMARKER = make_landmarker()


def load_cnn(path):
    ck = torch.load(path, map_location=dev)
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(NAMES))
    m.load_state_dict(ck['model'])
    return m.to(dev).eval(), ck['mode']


CNN = {}
for key, fname in (('A1', 'A1_group.pt'), ('A2', 'A2_crop.pt')):
    if (MODELS / fname).exists():
        CNN[key] = load_cnn(MODELS / fname)
print('loaded CNNs:', {k: v[1] for k, v in CNN.items()})
MODEL_B = joblib.load(MODELS / 'model_b.joblib') if (MODELS / 'model_b.joblib').exists() else None
print('Model B (feature trees):', 'loaded' if MODEL_B else 'not available (it was not trained in 04b)')


@torch.no_grad()
def cnn_probs(key, pil):
    model, mode = CNN[key]
    x = eval_tf(pil).unsqueeze(0).to(dev)
    return torch.softmax(model(x).float(), 1)[0].cpu().numpy()


def proba_full(model, X):
    p = model.predict_proba(X); out = np.zeros((len(X), len(NAMES))); out[:, model.classes_] = p
    return out


def feat_conf_row(row):
    # confidence in the fingertip features: Phase 2 geometry confidence x mean Phase 3 confidence of the fingertips on the neck
    if not row['features_usable']:
        return np.nan
    c = [row[f'{f}_conf'] for f in ('Index', 'Middle', 'Ring', 'Pinky') if row[f'{f}_in'] == 1 and not np.isnan(row[f'{f}_conf'])]
    return row['geom_conf_num'] * float(np.mean(c)) if c else np.nan


def run_phases(bgr, image_id='photo', p2=None):
    # Phases 1-3 for one photo (pass p2 to reuse an already detected fretboard)
    hands = detect_hands(bgr, image_id)
    p2 = p2 if p2 is not None else detect_fretboard(bgr, image_id)
    res, _ = map_image(image_id, bgr, hands, p2, 'recompute', check_coordinates(image_id, bgr, hands, p2))
    res.update(chord='?', split='own', row=0, phase2_geometry=p2)
    return res, feature_row(res), p2


def predict_from(bgr, row):
    # models only (Phases 1-3 already done): CNN on the right input, Model B on the feature row, blend, abstain
    pil_full = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    pil_crop = pil_full.crop((int(row['crop_x0']), int(row['crop_y0']), int(row['crop_x1']), int(row['crop_y1']))) if row['crop_kind'] != 'full' else pil_full
    probs = {k: cnn_probs(k, pil_crop if CNN[k][1] == 'crop' else pil_full) for k in CNN}
    p_cnn = probs[BASE if BASE in probs else next(iter(probs))][None, :]
    p_feat = np.full((1, len(NAMES)), np.nan)
    fc = feat_conf_row(row)
    if MODEL_B is not None and not np.isnan(fc):
        p_feat = proba_full(MODEL_B['trees'], np.array([[row[c] for c in B_COLS]], float))
    fused = fuse(p_cnn, p_feat, np.array([fc]), ALPHA)[0]
    top = int(fused.argmax())
    used = 'cnn + features' if (not np.isnan(fc) and not np.isnan(p_feat).all() and ALPHA > 0) else 'cnn only'
    return dict(chord=NAMES[top] if fused[top] >= THRESH else None, best=NAMES[top], conf=float(fused[top]), probs=fused,
                p_cnn=p_cnn[0], p_feat=None if np.isnan(p_feat).all() else p_feat[0], used=used, feat_conf=fc)


def predict_image(bgr, image_id='photo'):
    res, row, p2 = run_phases(bgr, image_id)
    out = predict_from(bgr, row)
    out.update(res=res, row=row, p2=p2)
    return out
print('prediction code ready')
""")

md(r"""
## 3. Report on the group-split test set
**What:** the saved probabilities from 04b for the **test images** (a held-out set whose near-duplicate groups never appeared in training), turned into the metrics below. *Top-1* is plain accuracy; *top-3* is "right chord among the 3 best guesses"; *root* counts `Am` for `A` as right; *major/minor* only checks that. *Balanced accuracy* and *macro-F1* weigh every chord equally.
**Subset "with features":** images where Phase 3 produced usable fingertip numbers. This is where Model B and the blend can help at all; everywhere else the system is the CNN alone.
**Coverage vs accuracy:** answering only when confident. The row at the chosen abstention threshold is what the system would do in use.
""")
code(r"""
P = np.load(MODELS / 'predictions.npz', allow_pickle=True)
yt = P['y_test']; test_ids = list(P['test_ids'])
fused, a1, a2, pb = P['fused_test'], P['a1_test'], P['a2_test'], P['b_test']
has_feat = ~np.isnan(pb).all(axis=1)
print(f'test images: {len(yt)} | with usable fingertip features: {has_feat.sum()} ({has_feat.mean():.0%})')

rows = []
for name, prob in (('A1 whole-image CNN', a1), ('A2 crop CNN', a2), ('FUSION (what the system answers)', fused)):
    for subset, sel in (('all test images', np.ones(len(yt), bool)), ('test images WITH features', has_feat), ('test images WITHOUT features', ~has_feat)):
        if sel.any():
            rows.append(dict(model=name, subset=subset, **{k: round(v, 3) for k, v in metrics(yt[sel], prob[sel], NAMES).items()}))
REPORT = pd.DataFrame(rows); print(REPORT.to_string(index=False))

pred = fused.argmax(1); conf = fused.max(1)
cls_rows = []
for c, n in enumerate(NAMES):
    t = yt == c
    if t.any():
        tp = ((pred == c) & t).sum(); fp = ((pred == c) & ~t).sum()
        cls_rows.append(dict(chord=n, images=int(t.sum()), recall=round(tp / t.sum(), 3), precision=round(tp / (tp + fp), 3) if (tp + fp) else np.nan))
print('\nper chord (fusion):'); print(pd.DataFrame(cls_rows).to_string(index=False))
conf_pairs = collections.Counter((NAMES[a], NAMES[b]) for a, b in zip(yt, pred) if a != b)
print('\nmost common mistakes (true -> predicted):', [(f'{a}->{b}', n) for (a, b), n in conf_pairs.most_common(8)])

cm = np.zeros((len(NAMES), len(NAMES)), int)
for a, b in zip(yt, pred): cm[a, b] += 1
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
ax[0].imshow(cm, cmap='Blues'); ax[0].set_xticks(range(len(NAMES)), NAMES, rotation=90); ax[0].set_yticks(range(len(NAMES)), NAMES)
ax[0].set_xlabel('predicted'); ax[0].set_ylabel('true'); ax[0].set_title('confusion matrix (test, fusion)')
for i in range(len(NAMES)):
    for j in range(len(NAMES)):
        if cm[i, j]: ax[0].text(j, i, cm[i, j], ha='center', va='center', fontsize=7, color='white' if cm[i, j] > cm.max() / 2 else 'black')
cv = coverage_curve(fused, yt, np.round(np.linspace(0, 0.95, 20), 2))
ax[1].plot(cv['coverage'], cv['accuracy'], 'o-'); ax[1].set_xlabel('coverage (share of images answered)'); ax[1].set_ylabel('accuracy on answered'); ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=.3)
ax[1].set_title('answer fewer, be right more often')
plt.tight_layout(); plt.show()
k = conf >= THRESH
print(f'\nAt the chosen abstention threshold {THRESH}: the system answers {k.mean():.1%} of test images, with accuracy {(pred[k] == yt[k]).mean():.3f} on those it answers.')
if SET['rehearsal']:
    print('REHEARSAL models: do not read anything into these numbers.')
""")

md(r"""
## 4. Error gallery: why was it wrong?
**What:** test images the system got **wrong**, re-processed through Phases 1-3 and drawn on the original photo. Orange = fretboard region, cyan = strings, green = frets, magenta = nut; fingertip dots are green (accepted by Phase 3), yellow (tentative) or grey (rejected). The title shows `true -> predicted (confidence)` and whether the blend used fingertip features.
**How to read a mistake:**
- *No hand dots at all:* MediaPipe missed the hand, so the CNN had to guess alone.
- *Orange region on the wrong thing:* Phase 2 found the wrong neck.
- *Dots look right and the prediction is a neighbour chord (Am for A):* a genuinely hard major/minor call.
It re-runs Phases 1-3 for each shown image (about 2 seconds each), and loads the pictures from the same Hugging Face dataset.
""")
code(r"""
from datasets import load_dataset
R = 'hf://datasets/j-asra-j/guitar-chords@refs%2Fconvert%2Fparquet/default/'
_ds = load_dataset('parquet', data_files={'train': [R + 'train/0000.parquet', R + 'train/0001.parquet'],
                                           'validation': R + 'validation/0000.parquet', 'test': R + 'test/0000.parquet'})
HF = {k: _ds[k] for k in ('train', 'validation', 'test')}


def bgr_by_id(iid):
    chord, split, row = iid.rsplit('_', 2)[0], iid.rsplit('_', 2)[1], int(iid.rsplit('_', 1)[1])
    return cv2.cvtColor(np.array(HF[split][row]['image'].convert('RGB')), cv2.COLOR_RGB2BGR)


def overlay(bgr, out):
    img = draw_result(bgr, out['p2'], alpha=0.18)
    s = max(img.shape[:2]) / 640.0
    for t in out['res']['fingertips']:
        c = {'accepted': (0, 200, 0), 'tentative': (0, 255, 255)}.get(t['decision'], (160, 160, 160))
        p = (int(round(t['x_px'])), int(round(t['y_px'])))
        cv2.circle(img, p, max(3, int(6 * s)), c, -1, cv2.LINE_AA); cv2.circle(img, p, max(3, int(6 * s)) + 1, (0, 0, 0), 1, cv2.LINE_AA)
    r = out['row']
    return img[int(r['crop_y0']):int(r['crop_y1']), int(r['crop_x0']):int(r['crop_x1'])]


wrong = [i for i in range(len(yt)) if pred[i] != yt[i]]
rng = random.Random(0); rng.shuffle(wrong)
print(f'{len(wrong)} wrong test images of {len(yt)}; showing up to 9')
ims, ts = [], []
for i in wrong[:9]:
    iid = test_ids[i]; bgr = bgr_by_id(iid)
    out = predict_image(bgr, iid)
    ims.append(overlay(bgr, out))
    ts.append(f"{iid}\ntrue {NAMES[yt[i]]} -> {out['best']} ({out['conf']:.2f}) | {out['used']}" + ('' if out['chord'] else ' | NOT SURE'))
if ims:
    show_images(ims, ts, cols=3, size=4.6)
""")

md(r"""
## 5. Your own photos (the number that matters)
**What:** a zip of your photos, one folder per chord using the exact chord names (`Am/`, `C/`, `Gm/` ...). Each photo goes through the full system. The models **never saw these photos**, so this score tells you whether the system works on *your* guitar, lighting and camera angle.
**How to take them (this matters):** the same camera and angle you will use in the demo; the chord fully formed; both the fretting hand and the neck in view; vary the lighting a little; 10-20 photos per chord is a good start; take some right after re-fretting so they are not near-identical.
**Set up:** zip the chord folders into `own_test.zip` and either put it at `PROJECT_DIR/own_test.zip` or set `ASK_FOR_OWN_PHOTOS = True` to get a file picker. If there is no zip, the cell prints this and skips.
**Reading the result:** a big gap between section 3 and this section is the honest measure of how much the dataset's look differs from your setup. That gap is what the next phase (fine-tuning with some of your photos) is for.
""")
code(r"""
ASK_FOR_OWN_PHOTOS = False
OWN_ZIP = Path(PROJECT_DIR) / 'own_test.zip'
if not OWN_ZIP.exists() and ASK_FOR_OWN_PHOTOS:
    from google.colab import files
    up = files.upload(); OWN_ZIP.write_bytes(up[[n for n in up if n.lower().endswith('.zip')][0]])

OWN = []
if OWN_ZIP.exists():
    d = Path(PROJECT_DIR) / 'own_test'; shutil.rmtree(d, ignore_errors=True)
    with zipfile.ZipFile(OWN_ZIP) as z: z.extractall(d)
    for f in sorted(d.rglob('*')):
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'} and '__MACOSX' not in f.parts and f.parent.name in NAMES:
            bgr = cv2.imread(str(f))
            if bgr is not None:
                OWN.append(dict(path=f, chord=f.parent.name, bgr=bgr))
    print(f'{len(OWN)} own photos found; chords: {pd.Series([o["chord"] for o in OWN]).value_counts().sort_index().to_dict()}')
    unknown = {f.parent.name for f in d.rglob('*') if f.is_file() and f.parent.name not in NAMES and f.parent != d}
    if unknown: print('ignored folders (not a known chord name):', sorted(unknown))
else:
    print('No own_test.zip yet, so this section is skipped. See the text above for how to take the photos.')

if OWN:
    t0 = time.time(); rows, outs = [], []
    for o in OWN:
        out = predict_image(o['bgr'], o['path'].stem); outs.append(out)
        rows.append(dict(photo=o['path'].name, true=o['chord'], pred=out['best'], conf=round(out['conf'], 3), answered=out['chord'] is not None,
                         used=out['used'], hand_found=int(out['row']['hand_found']), geometry=out['row']['geom_conf'],
                         cnn_only_pred=NAMES[int(out['p_cnn'].argmax())]))
    OWN_DF = pd.DataFrame(rows)
    y = np.array([NAMES.index(c) for c in OWN_DF['true']]); prob = np.vstack([o['probs'] for o in outs]); pc = np.vstack([o['p_cnn'] for o in outs])
    print(f'processed {len(OWN)} photos in {time.time() - t0:.0f}s')
    print('FUSION on your photos :', {k: round(v, 3) for k, v in metrics(y, prob, NAMES).items()})
    print('CNN alone on your photos:', {k: round(v, 3) for k, v in metrics(y, pc, NAMES).items()})
    ans = OWN_DF['answered'].values
    print(f"answered {ans.mean():.0%} of photos; accuracy on answered {(OWN_DF['pred'][ans] == OWN_DF['true'][ans]).mean():.3f}" if ans.any() else 'abstained on every photo')
    print('hand found in', f"{OWN_DF['hand_found'].mean():.0%}", 'of your photos (fingertip features exist only for those)')
    print(OWN_DF.groupby('true').apply(lambda g: pd.Series({'photos': len(g), 'correct': int((g['pred'] == g['true']).sum())}), include_groups=False).T.to_string())
    bad = [i for i, o in enumerate(OWN) if OWN_DF.loc[i, 'pred'] != OWN_DF.loc[i, 'true']][:9]
    if bad:
        show_images([overlay(OWN[i]['bgr'], outs[i]) for i in bad],
                    [f"{OWN[i]['path'].name}\ntrue {OWN_DF.loc[i, 'true']} -> {OWN_DF.loc[i, 'pred']} ({OWN_DF.loc[i, 'conf']})" for i in bad], cols=3, size=4.6)
""")

md(r"""
## 6. Live demo with your webcam
**What:** the whole system on live video.
1. **Lock the fretboard.** Phase 2 takes about 1.7 s per frame, far too slow to run on every frame, and the guitar hardly moves. So the first frame (or when you press **Re-lock fretboard**) detects the neck once and every later frame **reuses** it. If you move the guitar, press Re-lock.
2. **Every frame:** MediaPipe in **video mode** (which tracks the hand between frames), then Phase 3 with the locked neck, the feature row, the crop, the CNN, Model B, and the blend.
3. **Smoothing:** single frames are noisy. The last `SMOOTH_FRAMES` probability vectors are averaged, then the abstention threshold decides whether to show a chord or **not sure**.
**Honest expectation:** the video goes browser to Colab and back every frame, so expect a few frames per second, not smooth video. Hold the chord steady for a second or two before reading the answer.
**Test without a webcam:** `simulate_stream(images)` pushes still images through the exact same code, so the pipeline can be checked without a camera.
""")
code(r"""
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _vision

SMOOTH_FRAMES = 8


class VideoLandmarker:
    # Lets the unchanged `detect_hands` (which calls LANDMARKER.detect) run in MediaPipe's VIDEO mode, which tracks hands over time.
    def __init__(self):
        self.lm = _vision.HandLandmarker.create_from_options(_vision.HandLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=MODEL_PATH), running_mode=_vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3))
        self.t0, self.last = time.time(), -1
    def detect(self, mp_img):
        ts = max(int((time.time() - self.t0) * 1000), self.last + 1)        # timestamps must strictly increase
        self.last = ts
        return self.lm.detect_for_video(mp_img, ts)


LIVE = dict(p2=None, hist=collections.deque(maxlen=SMOOTH_FRAMES), n=0, t_last=time.time(), fps=0.0)
IMAGE_LANDMARKER = LANDMARKER            # still photos always use the image-mode landmarker
VIDEO_LANDMARKER = VideoLandmarker()     # the live demo uses the tracking (video-mode) one


def live_reset():
    LIVE.update(p2=None, n=0); LIVE['hist'].clear()


def live_step(bgr, relock=False):
    # One frame through the whole system. Returns the annotated frame (BGR), the label and a short info text.
    global LANDMARKER
    LANDMARKER = VIDEO_LANDMARKER                                   # swapped in only for the live frame ...
    try:
        LIVE['n'] += 1
        if relock or LIVE['p2'] is None or (LIVE['p2']['status'] == 'failed' and LIVE['n'] % 20 == 0):
            LIVE['p2'] = detect_fretboard(bgr, 'live'); LIVE['hist'].clear()      # slow (about 2 s) but done only when needed
        res, row, _ = run_phases(bgr, 'live', p2=LIVE['p2'])
    finally:
        LANDMARKER = IMAGE_LANDMARKER                               # ... and restored, so still photos never inherit tracking state
    out = predict_from(bgr, row)
    LIVE['hist'].append(out['probs'])
    avg = np.mean(LIVE['hist'], axis=0); top = int(avg.argmax()); conf = float(avg[top])
    now = time.time(); LIVE['fps'] = 0.8 * LIVE['fps'] + 0.2 / max(now - LIVE['t_last'], 1e-3) if LIVE['n'] > 1 else 0.0; LIVE['t_last'] = now
    img = draw_result(bgr, LIVE['p2'], alpha=0.15)
    s = max(img.shape[:2]) / 640.0
    for t in res['fingertips']:
        c = {'accepted': (0, 200, 0), 'tentative': (0, 255, 255)}.get(t['decision'], (160, 160, 160))
        cv2.circle(img, (int(t['x_px']), int(t['y_px'])), max(3, int(6 * s)), c, -1, cv2.LINE_AA)
    label = f"{NAMES[top]}  {conf:.2f}" if conf >= THRESH else 'not sure'
    cv2.rectangle(img, (0, 0), (img.shape[1], int(46 * s)), (0, 0, 0), -1)
    cv2.putText(img, label, (int(8 * s), int(34 * s)), cv2.FONT_HERSHEY_SIMPLEX, 1.1 * s, (0, 255, 255) if conf >= THRESH else (200, 200, 200), max(2, int(2 * s)), cv2.LINE_AA)
    info = f"{out['used']} | hand {'yes' if row['hand_found'] else 'no'} | neck {row['geom_conf']} | {LIVE['fps']:.1f} fps"
    return img, label, info


def simulate_stream(frames, relock_first=True, show=True):
    # Push still images (BGR arrays) through the live code, as if they were consecutive video frames.
    live_reset(); outs = []
    for k, f in enumerate(frames):
        img, label, info = live_step(f, relock=(relock_first and k == 0)); outs.append((img, label, info))
    if show:
        show_images([o[0] for o in outs[:6]], [f'frame {k + 1}: {o[1]}\n{o[2]}' for k, o in enumerate(outs[:6])], cols=3, size=4.4)
    return outs
print('live code ready (webcam cell below; or call simulate_stream([...images...]))')
""")
code(r"""
# Live webcam. Click Allow when the browser asks for the camera. Hold the chord, press 'Re-lock fretboard' if the guitar moves.
from google.colab import output
from IPython.display import display, Javascript, JSON


def process_frame(data_url, relock):
    jpg = base64.b64decode(data_url.split(',')[1])
    bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    img, label, info = live_step(bgr, bool(relock))
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return JSON({'img': 'data:image/jpeg;base64,' + base64.b64encode(buf).decode(), 'label': label, 'info': info})

output.register_callback('notebook.chord_frame', process_frame)
live_reset()

JS = '''
async function run() {
  const box = document.createElement("div");
  const stopBtn = document.createElement("button"); stopBtn.textContent = "Stop camera"; stopBtn.style.cssText = "font-size:16px;padding:6px 14px;margin-right:8px";
  const lockBtn = document.createElement("button"); lockBtn.textContent = "Re-lock fretboard"; lockBtn.style.cssText = "font-size:16px;padding:6px 14px";
  const status = document.createElement("div"); status.style.cssText = "font:16px sans-serif;margin:8px 0";
  const img = document.createElement("img"); img.style.maxWidth = "100%";
  box.append(stopBtn, lockBtn, status, img);
  document.querySelector("#output-area").appendChild(box);
  let stop = false, relock = true;
  stopBtn.onclick = () => { stop = true; }; lockBtn.onclick = () => { relock = true; };
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}});
    const video = document.createElement("video"); video.srcObject = stream; video.muted = true; await video.play();
    const canvas = document.createElement("canvas"); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    status.textContent = "Locking the fretboard on the first frame (takes a couple of seconds)...";
    while (!stop) {
      ctx.drawImage(video, 0, 0);
      const r = relock; relock = false;
      const res = await google.colab.kernel.invokeFunction("notebook.chord_frame", [canvas.toDataURL("image/jpeg", 0.7), r], {});
      const d = res.data["application/json"];
      img.src = d.img; status.textContent = d.label + "   (" + d.info + ")";
    }
    stream.getTracks().forEach(t => t.stop()); status.textContent = "Camera stopped.";
  } catch (e) { status.textContent = "Camera error: " + e; }
}
run();
'''
display(Javascript(JS))
""")
code(r"""
# Check the live pipeline WITHOUT a webcam: feed a few test photos of one chord as if they were video frames
sample_ids = [test_ids[i] for i in range(len(test_ids)) if yt[i] == yt[0]][:4]
frames = [bgr_by_id(i) for i in sample_ids]
print('true chord:', NAMES[yt[0]], '| frames:', sample_ids)
_ = simulate_stream(frames)
""")

md(r"""
## 7. Summary, honest limits, and what comes next
**Save:** the evaluation numbers go into `p4_eval.json` (downloaded as a zip) so they can be compared after the next phase.

**What these results do and do not say**
- The **group-split test** is the cleanest number available from the dataset, but the dataset itself is augmented and visually narrow. Treat it as an upper bound.
- **Your own photos** are the real measure. A big drop there is expected before any fine-tuning.
- Fingertip features exist only for the images where MediaPipe found a hand *and* Phase 3 placed fingers on the neck. Everywhere else the system is the CNN alone, and the report says how large each part is.
- A confidence value is not a guarantee. Use the coverage-vs-accuracy curve, not a single number, to decide when to trust an answer.

**Next phase (fine-tuning), in the order I would try**
1. Add a portion of **your own photos** to training (keeping a separate set for testing).
2. Look at which chords get confused (usually major/minor pairs) and collect more of those.
3. Improve the weakest link the error gallery points to: missed hands, wrong necks, or near-identical chord shapes.
4. Only then tune the CNN (longer training, stronger augmentation, a larger backbone).
""")
code(r"""
summary = dict(rehearsal=SET['rehearsal'], base_cnn=BASE, alpha=ALPHA, abstain_threshold=THRESH,
               test_report=json.loads(REPORT.to_json(orient='records')),
               own_photos=None if not OWN else dict(n=len(OWN),
                    fusion={k: round(v, 4) for k, v in metrics(y, prob, NAMES).items()}, cnn_only={k: round(v, 4) for k, v in metrics(y, pc, NAMES).items()}))
EVAL_DIR = P4 / 'eval'; EVAL_DIR.mkdir(exist_ok=True)
(EVAL_DIR / 'p4_eval.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
REPORT.to_csv(EVAL_DIR / 'test_report.csv', index=False)
zip_out = shutil.make_archive(str(Path(PROJECT_DIR) / 'p4_eval'), 'zip', EVAL_DIR)
print('saved', zip_out)
""")
code(r"""
# colab-only
from google.colab import files
files.download(f'{PROJECT_DIR}/p4_eval.zip')
""")

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "04c_evaluate_and_demo.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "| copied-code fingerprint", DIGEST)
