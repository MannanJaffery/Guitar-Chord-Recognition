"""Generates notebooks/04b_train_models.ipynb (Phase 4, step b: group split, CNN models, feature model, fusion).

Usage: python scripts/make_train_notebook.py

- Cells whose first line is '# colab-only' need Google Colab.
- The training loop follows the existing baseline in scripts/make_notebook.py (EfficientNet-B0, class-weighted loss,
  label smoothing, OneCycleLR, no horizontal flip) but drops Google Drive and uses our own split and crops.
- Helper cell 'helpers' is numpy/pandas only, so it can be unit-tested without a GPU or PyTorch.
"""
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
md(r"""
# Notebook 04b: Train the chord models (Phase 4, step b)

**Goal:** train and compare three ways of recognising a chord, then combine them:
- **Model A1 (baseline CNN):** looks at the **whole photo**.
- **Model A2 (crop CNN):** looks at a **crop around the neck and hand**, found by your Phase 1 and 2 work (notebook 04a saved the crop boxes).
- **Model B (feature model):** looks only at **numbers**: where each fingertip sits on the strings and frets (Phase 3). No pixels.
- **Fusion:** blends the CNN and Model B, trusting Model B more when Phase 3 was confident, and **abstains** ("not sure") when the blended confidence is low.

**Not in this notebook:** your own-photo test and the live webcam demo (04c), or any tuning to squeeze out accuracy (a later phase).

### Why the split matters more than the model
Your dataset contains **augmented near-copies** of the same photos (rotated, recoloured). The exact-duplicate check in the dataset report found nothing, because the copies are not byte-identical. If near-copies end up on both sides of a train/test split, the model is "tested" on pictures it already saw, and the score is **fake-high**. So this notebook:
1. finds near-duplicate **groups** (with image embeddings),
2. splits **by group**, so a group is entirely in train or entirely in test,
3. reports **both** the official-split score (optimistic) and the group-split score (honest), so you can see how big the difference is.

### Another trap this notebook guards against
Notebook 04a showed that images where MediaPipe finds a hand are **not a random sample**: Cm, Fm and Gm (a visually different source) nearly always have a hand, while most major chords do not. So "a hand was found" is itself a hint about the chord, and a model that sees "feature present/missing" could cheat. Model B therefore (a) uses **only geometry numbers**, never availability flags, (b) trains **only on images where the features exist**, and (c) is always compared with the CNN **on the same images**.

### Honest expectation (written before training, so we cannot move the goalposts)
Group-split test accuracy around 75-92%; much lower on your own photos before any fine-tuning. The fusion may help only a little, because only a part of the images have usable fingertip features. The notebook reports the numbers either way.

### Cells
0. Setup, 1. Config and data, 2. Helper functions, 3. Near-duplicate groups and the split, 4. CNN code, 5. Model A1 (+ official-split baseline), 6. Model A2, 7. Model B, 8. Fusion and abstention, 9. Report, save, download
""")

md(r"""
## 0. Setup
**What:** checks for a GPU (the CNNs need one: **Runtime > Change runtime type > T4 GPU**) and installs `datasets` if needed. scikit-learn, PyTorch and torchvision are already on Colab. No Google Drive: models are saved to the Colab disk and downloaded as a zip at the end.
""")
code(r"""
# colab-only
import os
PROJECT_DIR = '/content/Guitar-Chord-Recognition'
os.makedirs(f'{PROJECT_DIR}/data', exist_ok=True)
print('Project folder:', PROJECT_DIR)
""")
code(r"""
import os, sys, subprocess, importlib, json, time, random, zipfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure(pkg, mod=None):
    try:
        importlib.import_module(mod or pkg)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', pkg], check=True)
        importlib.invalidate_caches()
_ensure('datasets')

import torch
print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (CNN training will be very slow)')
""")

md(r"""
## 1. Config, features from 04a, and the images
**What:** loads `features.csv` made by notebook 04a (if you ran 04a in this same session it is already on disk; otherwise a file picker asks for `p4_features.zip`), and loads the dataset images from Hugging Face.
**The universe of this notebook = the images listed in `features.csv`.** All models are trained and tested on exactly these images, so the comparison is fair. If 04a ran in `SMOKE` mode (only ~280 images) this notebook runs as a **rehearsal** with fewer epochs, and its numbers mean nothing about the real accuracy; it only proves the pipeline works.
""")
code(r"""
CFG = dict(img_size=224, batch_size=32, lr_head=1e-3, lr_backbone=1e-4, weight_decay=1e-4, label_smoothing=0.1, seed=42,
           epochs_full=15, epochs_rehearsal=3,      # rehearsal = tiny run to check the pipeline
           group_tau=0.88,                          # similarity at/above which two images count as near-copies (checked in section 3)
           group_max_frac=0.12,                     # a chain of look-alikes bigger than this share of all images is broken up
           official_baseline=True,                  # also train A1 on the dataset's official split, to expose the leak
           abstain_target_acc=0.90)                 # abstention threshold is the loosest one reaching this accuracy on validation
P4 = Path(PROJECT_DIR) / 'outputs' / 'phase4'
MODELS = P4 / 'models'; MODELS.mkdir(parents=True, exist_ok=True)
FEAT_CSV = P4 / 'features.csv'

if not FEAT_CSV.exists():                                     # not in this session: ask for the zip from notebook 04a
    from google.colab import files
    print('Choose p4_features.zip (made by notebook 04a) ...')
    up = files.upload()
    zp = Path(PROJECT_DIR) / 'p4_features.zip'
    zp.write_bytes(up[[n for n in up if n.lower().endswith('.zip')][0]])
    with zipfile.ZipFile(zp) as z:
        z.extractall(P4)

DF = pd.read_csv(FEAT_CSV).set_index('id', drop=False)
NAMES = sorted(DF['chord'].unique())
CIDX = {c: i for i, c in enumerate(NAMES)}
DF['y'] = DF['chord'].map(CIDX)
REHEARSAL = len(DF) < 1500
EPOCHS = CFG['epochs_rehearsal'] if REHEARSAL else CFG['epochs_full']
random.seed(CFG['seed']); np.random.seed(CFG['seed']); torch.manual_seed(CFG['seed'])
print(f"{len(DF)} images, {len(NAMES)} chords: {NAMES}")
print('MODE:', 'REHEARSAL (few images, numbers are NOT meaningful)' if REHEARSAL else 'FULL', '| epochs:', EPOCHS)
print('images per official split:', DF['split'].value_counts().to_dict())
print('feature rows with usable fingertip features:', int(DF['features_usable'].sum()), 'of', len(DF))

from datasets import load_dataset
R = 'hf://datasets/j-asra-j/guitar-chords@refs%2Fconvert%2Fparquet/default/'
_ds = load_dataset('parquet', data_files={'train': [R + 'train/0000.parquet', R + 'train/0001.parquet'],
                                           'validation': R + 'validation/0000.parquet', 'test': R + 'test/0000.parquet'})
HF = {k: _ds[k] for k in ('train', 'validation', 'test')}


def get_pil(iid):
    # the ORIGINAL image for an ID (same ID scheme as notebooks 01-04a)
    r = DF.loc[iid]
    return HF[r['split']][int(r['row'])]['image'].convert('RGB')


def get_crop_pil(iid):
    # crop around the fretboard and hand (boxes computed in 04a, original pixels); whole image if nothing was found
    r = DF.loc[iid]
    im = get_pil(iid)
    if r['crop_kind'] != 'full':
        im = im.crop((int(r['crop_x0']), int(r['crop_y0']), int(r['crop_x1']), int(r['crop_y1'])))
    return im
""")

md(r"""
## 2. Helper functions (numpy only)
**What:** the small, testable pieces of logic, kept free of PyTorch so they can be checked on their own.

- **Metrics.** *Accuracy* (top-1), *top-3* (right chord among the 3 best guesses), *balanced accuracy* (average of per-chord recall, so large classes do not hide small ones), *macro-F1* (same idea, combining precision and recall), **root-note accuracy** (guessing `Am` for `A` still gets the root right), and **major/minor accuracy**. With 14 look-alike chords, root and major/minor accuracy tell you *how* the model is wrong.
- **Near-duplicate grouping.** Two images are linked if their embedding similarity is at least `tau`. **Connected components** (union-find) turn links into groups. Linking can chain: A~B, B~C, C~D makes one giant group (think of consecutive video frames), so any group larger than `group_max_frac` of the data is re-split with a stricter `tau`, and if that cannot separate it (a continuous drift of look-alikes) it is broken with **leader clustering** (take one image and only its direct near-neighbours). The price: neighbouring frames may land in different groups, so a little leakage can remain; the leak rate printed below tells you how much.
- **Fusion.** `p = (1 - w) * p_cnn + w * p_features`, where `w = alpha x feature_confidence` (zero when features are missing, capped at 0.9).
""")
code(r"""
def root_of(name): return name[:-1] if name.endswith('m') else name
def is_minor(name): return name.endswith('m')


def metrics(y, prob, names):
    # y: true class indices, prob: [N, C] probabilities. Returns a dict of the metrics described above.
    y = np.asarray(y); pred = prob.argmax(1)
    top3 = np.mean([yy in np.argsort(-p)[:3] for yy, p in zip(y, prob)]) if len(y) else np.nan
    roots = np.array([root_of(n) for n in names]); minor = np.array([is_minor(n) for n in names])
    cls = np.unique(y)
    f1 = []
    for c in cls:
        tp, fp, fn = ((pred == c) & (y == c)).sum(), ((pred == c) & (y != c)).sum(), ((pred != c) & (y == c)).sum()
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return dict(n=int(len(y)), acc=float((pred == y).mean()) if len(y) else np.nan, top3=float(top3),
                bal_acc=float(np.mean([(pred[y == c] == c).mean() for c in cls])) if len(cls) else np.nan,
                macro_f1=float(np.mean(f1)) if f1 else np.nan,
                root_acc=float((roots[pred] == roots[y]).mean()) if len(y) else np.nan,
                mm_acc=float((minor[pred] == minor[y]).mean()) if len(y) else np.nan)


def _components(n, ii, jj):
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def _leader_clusters(sub, t):
    # Chain-free grouping: repeatedly take the image with the most unassigned near-neighbours and group it WITH those
    # neighbours only (neighbours of neighbours are not pulled in), so a slow drift of look-alikes cannot become one blob.
    m = len(sub); free = np.ones(m, bool); lab = -np.ones(m, int); adj = sub >= t; k = 0
    while free.any():
        deg = (adj & free[None, :]).sum(1) * free
        seed = int(deg.argmax()) if deg.max() > 0 else int(np.where(free)[0][0])
        grp = free & (adj[seed] | (np.arange(m) == seed))
        lab[grp] = k; free &= ~grp; k += 1
    return lab


def group_by_similarity(sim, tau, max_frac=0.12, step=0.02, max_tau=0.995):
    # Connected components of the graph "similarity >= tau". An oversized group is re-split with a stricter tau; if tau
    # cannot be tightened any further (a continuous chain of look-alikes), it is broken with chain-free leader clustering.
    # NOTE: breaking a chain can put neighbouring frames in different groups, so a little leakage can remain.
    n = len(sim); groups = -np.ones(n, int); nxt = 0
    stack = [(np.arange(n), tau)]
    while stack:
        idx, t = stack.pop()
        sub = sim[np.ix_(idx, idx)]
        ii, jj = np.where(np.triu(sub >= t, 1))
        lab = _components(len(idx), ii, jj)
        for c in np.unique(lab):
            members = idx[lab == c]
            if len(members) <= max_frac * n:
                groups[members] = nxt; nxt += 1
            elif t + step <= max_tau:
                stack.append((members, t + step))                 # still too big: tighten the threshold and look again
            else:
                sub2 = sim[np.ix_(members, members)]
                lab2 = _leader_clusters(sub2, t)
                for c2 in np.unique(lab2):
                    groups[members[lab2 == c2]] = nxt; nxt += 1
    return groups


def fuse(p_cnn, p_feat, feat_conf, alpha, w_max=0.9):
    # Blend the two probability tables. Where features are missing (feat_conf is NaN) the CNN is used alone.
    fc = np.nan_to_num(np.asarray(feat_conf, float), nan=0.0)
    w = np.clip(alpha * fc, 0.0, w_max)[:, None]
    pf = np.where(np.isnan(p_feat), 0.0, p_feat)
    # the blend applies only where there is BOTH a feature confidence and a Model B probability row; elsewhere the CNN stands alone
    has = (~np.isnan(np.asarray(feat_conf, float)) & ~np.isnan(p_feat).all(axis=1))[:, None]
    return np.where(has, (1 - w) * p_cnn + w * pf, p_cnn)


def coverage_curve(prob, y, thresholds):
    # For each confidence threshold: share of images answered, and accuracy on the answered ones.
    conf, pred = prob.max(1), prob.argmax(1)
    out = []
    for t in thresholds:
        keep = conf >= t
        out.append((t, keep.mean(), (pred[keep] == y[keep]).mean() if keep.any() else np.nan))
    return pd.DataFrame(out, columns=['threshold', 'coverage', 'accuracy'])
print('helpers ready')
""")

md(r"""
## 3. Near-duplicate groups and the split
**Technique: image embeddings.** A pretrained CNN (EfficientNet-B0, ImageNet) turns each photo into a list of 1,280 numbers (an **embedding**) describing what it looks like. Two photos of the same scene, even rotated or re-lit, get similar embeddings. **Cosine similarity** (1 = same direction) measures it. We embed each image in **grayscale and also colour-inverted**, and take the larger similarity, because your dataset contains colour-inverted copies that would otherwise look unrelated.

**Choosing `group_tau`:** too low and unrelated photos merge into huge groups; too high and near-copies are missed. The cells below show (1) the distribution of each image's best-match similarity, (2) example pairs in several similarity bands so **you can look at them and judge**, and (3) the **leak rate**: the share of the official validation/test images that have a near-copy in the official train set. A high leak rate confirms why the official score is optimistic.

**The split:** `StratifiedGroupKFold` makes 7 folds in which every group stays whole and every chord is spread as evenly as possible. Fold 0 is **test** (about 14%), fold 1 is **validation** (about 14%), the other five are **train**. (Cm, Fm and Gm, which have no official validation/test images, get proper ones here.)
""")
code(r"""
import torch.nn as nn
from torchvision import models, transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import ImageOps

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
S = CFG['img_size']


class PilDS(Dataset):
    def __init__(self, ids, fn): self.ids, self.fn = list(ids), fn
    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.fn(self.ids[i])


def embed_all(ids, invert=False, bs=64):
    net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    net.classifier = nn.Identity()                                     # keep the 1280-number feature vector
    net = net.to(dev).eval()
    tf = T.Compose([T.Resize((S, S)), T.ToTensor(), T.Normalize(MEAN, STD)])
    def prep(iid):
        im = ImageOps.autocontrast(get_pil(iid).convert('L'))          # grayscale + contrast stretch
        if invert:
            im = ImageOps.invert(im)
        return tf(im.convert('RGB'))
    out = []
    with torch.no_grad():
        for x in DataLoader(PilDS(ids, prep), batch_size=bs, num_workers=2):
            out.append(net(x.to(dev)).float().cpu())
    E = torch.cat(out).numpy()
    return E / np.linalg.norm(E, axis=1, keepdims=True)


t0 = time.time()
IDS = list(DF.index)
E1, E2 = embed_all(IDS), embed_all(IDS, invert=True)
SIM = np.maximum(E1 @ E1.T, E1 @ E2.T)
SIM = np.maximum(SIM, SIM.T)
np.fill_diagonal(SIM, -1)                                              # ignore self-matches
print(f'embeddings + similarity for {len(IDS)} images in {time.time() - t0:.0f}s')
nn_sim = SIM.max(1)
print('best-match similarity per image: median %.3f | share >= 0.95: %.2f | >= 0.90: %.2f | >= %.2f (chosen tau): %.2f'
      % (np.median(nn_sim), (nn_sim >= 0.95).mean(), (nn_sim >= 0.90).mean(), CFG['group_tau'], (nn_sim >= CFG['group_tau']).mean()))
plt.figure(figsize=(6, 2.6)); plt.hist(nn_sim, bins=50); plt.axvline(CFG['group_tau'], color='r', ls='--')
plt.title('each image: similarity to its closest other image (red = tau)'); plt.tight_layout(); plt.show()
""")
code(r"""
# Look at pairs in several similarity bands, so tau can be judged by eye (left = image, right = its closest match)
def show_pairs(lo, hi, k=3, seed=0):
    rng = np.random.default_rng(seed)
    cand = [i for i in range(len(IDS)) if lo <= nn_sim[i] < hi]
    if not cand:
        print(f'  band {lo:.2f}-{hi:.2f}: no pairs'); return
    pick = rng.choice(cand, size=min(k, len(cand)), replace=False)
    fig, axes = plt.subplots(1, 2 * len(pick), figsize=(3 * 2 * len(pick), 3))
    for n, i in enumerate(pick):
        j = int(SIM[i].argmax())
        for m, idx in enumerate((i, j)):
            ax = axes[2 * n + m]; ax.imshow(get_pil(IDS[idx]).resize((200, 200))); ax.axis('off')
            ax.set_title(f'{IDS[idx]}\n' + (f'sim {nn_sim[i]:.2f}' if m == 0 else ''), fontsize=7)
    fig.suptitle(f'band {lo:.2f} - {hi:.2f}', fontsize=10); plt.tight_layout(); plt.show()

for lo, hi in [(0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85), (0.70, 0.80)]:
    show_pairs(lo, hi)
""")
code(r"""
TAU = CFG['group_tau']
GROUP = group_by_similarity(SIM, TAU, max_frac=CFG['group_max_frac'])
DF['group'] = GROUP
gs = pd.Series(GROUP).value_counts()
print(f'tau={TAU}: {len(gs)} groups | largest group: {gs.iloc[0]} images ({gs.iloc[0] / len(DF):.1%}) | singletons: {(gs == 1).sum()} '
      f'| images in groups of 2+: {(DF.groupby("group")["y"].transform("size") > 1).mean():.1%}')

# Leak rate of the OFFICIAL split: how many official val/test images have a near-copy in official train?
pos = {iid: k for k, iid in enumerate(IDS)}
is_tr = (DF['split'] == 'train').values
for sp in ('validation', 'test'):
    m = (DF['split'] == sp).values
    if m.any() and is_tr.any():
        print(f'official {sp}: {m.sum()} images, {(SIM[m][:, is_tr].max(1) >= TAU).mean():.1%} have a near-copy (similarity >= {TAU}) in official train')

from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=CFG['seed'])
folds = [f[1] for f in sgkf.split(np.zeros(len(DF)), DF['y'].values, DF['group'].values)]
DF['gsplit'] = 'train'
DF.iloc[folds[0], DF.columns.get_loc('gsplit')] = 'test'
DF.iloc[folds[1], DF.columns.get_loc('gsplit')] = 'val'
G = {k: list(DF.index[DF['gsplit'] == k]) for k in ('train', 'val', 'test')}
for k, v in G.items():
    print(f'group split {k}: {len(v)} images, chords present: {DF.loc[v, "chord"].nunique()} of {len(NAMES)}')
assert not (set(DF.loc[G['train'], 'group']) & set(DF.loc[G['test'], 'group'])), 'a group crosses train and test!'
assert not (set(DF.loc[G['train'], 'group']) & set(DF.loc[G['val'], 'group'])), 'a group crosses train and val!'
print('OK: no group crosses train/val/test')
tr_m = (DF['gsplit'] == 'train').values
for k in ('val', 'test'):                                      # residual leakage of OUR split (compare with the official split above)
    m = (DF['gsplit'] == k).values
    print(f'group split {k}: {(SIM[m][:, tr_m].max(1) >= TAU).mean():.1%} of images still have a near-copy (similarity >= {TAU}) in train'
          + ('  <- should be near 0%' if k == 'test' else ''))
print(pd.crosstab(DF['chord'], DF['gsplit']).T.to_string())
miss = [(k, c) for k in ('val', 'test') for c in NAMES if not ((DF['gsplit'] == k) & (DF['chord'] == c)).any()]
if miss:
    print('WARNING: chords with no images in a split (their scores cannot be measured):', miss)
""")

md(r"""
## 4. The CNN code (shared by A1 and A2)
**What:** one training function used for both CNN models. It follows the earlier baseline notebook: **EfficientNet-B0 pretrained on ImageNet** with its last layer replaced by a 14-way chord layer (*transfer learning*), **class-weighted loss** (rare chords count more, because Cm/Fm/Gm are 48% of the training photos), **label smoothing** (stops over-confident guesses), **AdamW** with a smaller learning rate for the pretrained body than for the new head, a **one-cycle** schedule, and **mixed precision** on the GPU. **No horizontal flip**: a flipped right-hand chord looks like a left-hand chord.
**Mode `full`** feeds the whole photo, **mode `crop`** feeds the neck-and-hand crop from 04a.
**Selecting the best epoch:** by validation **macro-F1**, not plain accuracy, so a model that ignores the small classes is not rewarded.
**Live view:** after every epoch the loss and validation curves are redrawn.
""")
code(r"""
from IPython.display import clear_output

train_tf = T.Compose([T.Resize((S, S)), T.RandomResizedCrop(S, scale=(0.75, 1.0), ratio=(0.9, 1.1)), T.RandomRotation(10),
                      T.ColorJitter(0.3, 0.3, 0.2, 0.02), T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize((S, S)), T.ToTensor(), T.Normalize(MEAN, STD)])
LABEL = DF['y'].to_dict()


class ChordDS(Dataset):
    def __init__(self, ids, mode, tf): self.ids, self.mode, self.tf = list(ids), mode, tf
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        iid = self.ids[i]
        im = get_crop_pil(iid) if self.mode == 'crop' else get_pil(iid)
        return self.tf(im), LABEL[iid]


def loader(ids, mode, train):
    return DataLoader(ChordDS(ids, mode, train_tf if train else eval_tf), batch_size=CFG['batch_size'], shuffle=train,
                      num_workers=2, pin_memory=(dev == 'cuda'), drop_last=False)


def new_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(NAMES))
    return m.to(dev)


@torch.no_grad()
def predict_probs(model, ids, mode):
    model.eval(); out = []
    for x, _ in loader(ids, mode, False):
        with torch.autocast(dev, enabled=(dev == 'cuda')):
            out.append(torch.softmax(model(x.to(dev)).float(), 1).cpu())
    return torch.cat(out).numpy()


def plot_history(hist, title):
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3))
    ax[0].plot(h['epoch'], h['train_loss'], 'o-'); ax[0].set_title(f'{title}: training loss'); ax[0].set_xlabel('epoch')
    ax[1].plot(h['epoch'], h['val_acc'], 'o-', label='val accuracy'); ax[1].plot(h['epoch'], h['val_f1'], 's-', label='val macro-F1')
    ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].set_title('validation'); ax[1].set_xlabel('epoch')
    plt.tight_layout(); plt.show()


def train_cnn(name, mode, tr_ids, va_ids, epochs):
    tr_dl = loader(tr_ids, mode, True)
    model = new_model()
    cnt = np.bincount([LABEL[i] for i in tr_ids], minlength=len(NAMES)).astype(float)
    w = torch.tensor(np.where(cnt > 0, cnt.sum() / (len(NAMES) * np.maximum(cnt, 1)), 0.0), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=CFG['label_smoothing'])
    opt = torch.optim.AdamW([{'params': model.features.parameters(), 'lr': CFG['lr_backbone']},
                             {'params': model.classifier.parameters(), 'lr': CFG['lr_head']}], weight_decay=CFG['weight_decay'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[CFG['lr_backbone'], CFG['lr_head']], total_steps=epochs * len(tr_dl))
    scaler = torch.amp.GradScaler(dev, enabled=(dev == 'cuda'))
    y_va = np.array([LABEL[i] for i in va_ids])
    hist, best, best_state = [], -1, None
    for ep in range(1, epochs + 1):
        model.train(); tot, t0 = 0.0, time.time()
        for x, y in tr_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, enabled=(dev == 'cuda')):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x)
        m = metrics(y_va, predict_probs(model, va_ids, mode), NAMES)
        hist.append(dict(epoch=ep, train_loss=tot / len(tr_ids), val_acc=m['acc'], val_f1=m['macro_f1'], secs=time.time() - t0))
        if m['macro_f1'] > best:
            best = m['macro_f1']; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        clear_output(wait=True)
        plot_history(hist, name)
        print(pd.DataFrame(hist).round(3).tail(5).to_string(index=False))
    model.load_state_dict(best_state)
    torch.save({'model': best_state, 'classes': NAMES, 'img_size': S, 'mean': MEAN, 'std': STD, 'mode': mode}, MODELS / f'{name}.pt')
    print(f'{name}: best validation macro-F1 {best:.3f}  (saved to {MODELS / (name + ".pt")})')
    return model, hist
print('CNN code ready; device:', dev)
""")

md(r"""
## 5. Model A1: the baseline CNN on the whole photo
Trained on the **group split** (honest). If `official_baseline` is on, the same network is also trained on the **dataset's official split** (with the usual fix that moves 15% of Cm/Fm/Gm training images into validation). Comparing the two shows how much of an "official" score is just near-copies. Watch the curves update each epoch.
""")
code(r"""
RES = {}                                   # name -> dict(prob_val, prob_test, ids ...)
A1, h = train_cnn('A1_group', 'full', G['train'], G['val'], EPOCHS)
RES['A1'] = dict(val=predict_probs(A1, G['val'], 'full'), test=predict_probs(A1, G['test'], 'full'))
yv, yt = DF.loc[G['val'], 'y'].values, DF.loc[G['test'], 'y'].values
print('A1 (group split) VAL ', {k: round(v, 3) for k, v in metrics(yv, RES['A1']['val'], NAMES).items()})
print('A1 (group split) TEST', {k: round(v, 3) for k, v in metrics(yt, RES['A1']['test'], NAMES).items()})
""")
code(r"""
# Official-split baseline: what the dataset's own split would have told us (expected to look better than it deserves)
OFFICIAL = None
if CFG['official_baseline']:
    rng = np.random.default_rng(CFG['seed'])
    tr = list(DF.index[DF['split'] == 'train']); va = list(DF.index[DF['split'] == 'validation']); te = list(DF.index[DF['split'] == 'test'])
    have_val = set(DF.loc[va, 'chord']) if va else set()
    for c in NAMES:                                            # classes with no official val: borrow 15% of their train images
        if c not in have_val:
            pool = [i for i in tr if DF.loc[i, 'chord'] == c]
            move = list(rng.choice(pool, size=max(1, int(0.15 * len(pool))), replace=False)) if pool else []
            moved = set(move); va += move; tr = [i for i in tr if i not in moved]
    if len(te) and len(va) and len(tr):
        A1o, _ = train_cnn('A1_official', 'full', tr, va, EPOCHS)
        pv, pt = predict_probs(A1o, va, 'full'), predict_probs(A1o, te, 'full')
        OFFICIAL = dict(val=metrics(DF.loc[va, 'y'].values, pv, NAMES), test=metrics(DF.loc[te, 'y'].values, pt, NAMES))
        print('A1 (OFFICIAL split) VAL ', {k: round(v, 3) for k, v in OFFICIAL['val'].items()})
        print('A1 (OFFICIAL split) TEST', {k: round(v, 3) for k, v in OFFICIAL['test'].items()})
        print('-> compare with the group-split numbers above: the gap is the optimism caused by near-copies')
    else:
        print('official split too small in this run; skipped')
""")

md(r"""
## 6. Model A2: the CNN on the neck-and-hand crop
Same network and settings as A1, but the input is the **crop around the fretboard and hand** (boxes from 04a, built from the Phase 2 polygon and the Phase 1 hand box). Cropping removes most of the background, which is where a whole-image model tends to take shortcuts. Images where neither a neck nor a hand was found keep the whole photo.
""")
code(r"""
print('crop kinds in this run:', DF['crop_kind'].value_counts().to_dict())
A2, h = train_cnn('A2_crop', 'crop', G['train'], G['val'], EPOCHS)
RES['A2'] = dict(val=predict_probs(A2, G['val'], 'crop'), test=predict_probs(A2, G['test'], 'crop'))
print('A2 (group split) VAL ', {k: round(v, 3) for k, v in metrics(yv, RES['A2']['val'], NAMES).items()})
print('A2 (group split) TEST', {k: round(v, 3) for k, v in metrics(yt, RES['A2']['test'], NAMES).items()})
""")

md(r"""
## 7. Model B: the feature model (numbers only)
**Technique: gradient-boosted decision trees** (`HistGradientBoostingClassifier`). Many small decision trees, each correcting the previous ones' mistakes. It works on a table of numbers, trains in seconds on a CPU, and **accepts missing values directly** (a missing fingertip simply takes a learned "default" branch).
**Inputs (and only these):** for every finger, its string position, its position along the neck, both relative to the lowest finger, and the differences between each pair of fretting fingers. **No availability flags, no confidence columns**, to avoid the "hand found means Cm/Fm/Gm" shortcut described at the top.
**Training rows:** only images where `features_usable == 1` (a hand and at least 2 of the 4 fretting fingers on the neck). Chords that never occur in those rows cannot be predicted by Model B at all; the table below shows which.
**Compared fairly:** we always evaluate Model B **and the CNNs on exactly the same usable test images**. A **logistic regression** (a simple linear model, with missing values filled by the median) is trained too, as a sanity check on whether the trees add anything.
""")
code(r"""
import re
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

B_COLS = [c for c in DF.columns if re.match(r'^(Index|Middle|Ring|Pinky|Thumb)_(sp|nc|sp_rel|nc_rel)$', c) or c.startswith('d_sp_') or c.startswith('d_nc_')]
print(len(B_COLS), 'feature columns, e.g.', B_COLS[:6])
use = DF['features_usable'] == 1
tr_b = [i for i in G['train'] if use[i]]; va_b = [i for i in G['val'] if use[i]]; te_b = [i for i in G['test'] if use[i]]
print(f'usable images -> train {len(tr_b)}, val {len(va_b)}, test {len(te_b)}')
print('chords present in Model B training rows:', DF.loc[tr_b, 'chord'].value_counts().to_dict())
absent = [c for c in NAMES if c not in set(DF.loc[tr_b, 'chord'])]
print('chords Model B can NEVER predict (no usable training rows):', absent)

MODEL_B = LOGREG = None
PB = {k: np.full((len(G[k]), len(NAMES)), np.nan) for k in ('val', 'test')}       # NaN = features unavailable for that image
if len(tr_b) >= 20 and DF.loc[tr_b, 'y'].nunique() >= 2:
    Xtr, ytr = DF.loc[tr_b, B_COLS].values, DF.loc[tr_b, 'y'].values
    cnt = np.bincount(ytr, minlength=len(NAMES)); sw = (len(ytr) / (np.count_nonzero(cnt) * np.maximum(cnt, 1)))[ytr]    # balance classes
    MODEL_B = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4, l2_regularization=1.0, random_state=CFG['seed'])
    MODEL_B.fit(Xtr, ytr, sample_weight=sw)
    LOGREG = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
    LOGREG.fit(Xtr, ytr)

    def proba_full(model, X):
        p = model.predict_proba(X); out = np.zeros((len(X), len(NAMES))); out[:, model.classes_] = p
        return out
    for k, ids_all, ids_use in (('val', G['val'], va_b), ('test', G['test'], te_b)):
        pos_in = {iid: j for j, iid in enumerate(ids_all)}
        if ids_use:
            pr = proba_full(MODEL_B, DF.loc[ids_use, B_COLS].values)
            for iid, row in zip(ids_use, pr):
                PB[k][pos_in[iid]] = row
    rows = []
    for nm, ids_use, ids_all, k in (('val', va_b, G['val'], 'val'), ('test', te_b, G['test'], 'test')):
        if not ids_use: continue
        pos_in = {iid: j for j, iid in enumerate(ids_all)}; sel = [pos_in[i] for i in ids_use]; yy = DF.loc[ids_use, 'y'].values
        X = DF.loc[ids_use, B_COLS].values
        for label, prob in (('Model B (trees)', proba_full(MODEL_B, X)), ('logistic regression', proba_full(LOGREG, X)),
                            ('CNN A1 (whole image)', RES['A1'][k][sel]), ('CNN A2 (crop)', RES['A2'][k][sel])):
            m = metrics(yy, prob, NAMES); rows.append(dict(subset=f'{nm} (usable images only)', model=label, **{a: round(b, 3) for a, b in m.items()}))
    print(pd.DataFrame(rows).to_string(index=False))
    print('\nRead this carefully: on these images the CNNs and Model B are compared on the SAME pictures. '
          'If Model B only matches the CNN here, remember it exists for only a part of the dataset.')
else:
    print('Not enough usable feature rows to train Model B in this run (expected in a small rehearsal). Fusion will use the CNN alone.')
""")

md(r"""
## 8. Fusion and abstention
**Fusion:** `p = (1 - w) * p_cnn + w * p_features` with `w = alpha x feature_confidence`. The feature confidence for an image is the **Phase 2 geometry confidence** (1.0 `high`, 0.6 `low`) times the **mean Phase 3 confidence** of its fingertips on the neck. If an image has no usable features, **the CNN is used alone**. `alpha` is the only thing tuned, on the **validation** set (never the test set). The CNN used in the blend is whichever of A1/A2 has the better validation macro-F1.

**Abstention:** the system answers "not sure" when its top probability is below a threshold. The threshold is the **loosest one that still reaches the target accuracy on validation** (`abstain_target_acc`). The **coverage-vs-accuracy curve** shows the trade-off: answer fewer images, be right more often.
""")
code(r"""
def feat_conf(ids):
    d = DF.loc[ids]
    confs = np.vstack([d[f'{f}_conf'].where(d[f'{f}_in'] == 1).values for f in ('Index', 'Middle', 'Ring', 'Pinky')]).T
    mc = np.nanmean(np.where(np.isnan(confs), np.nan, confs), axis=1) if confs.size else np.array([])
    fc = d['geom_conf_num'].values * mc
    return np.where(d['features_usable'].values == 1, fc, np.nan)           # NaN = no usable features -> CNN alone

f1v = metrics(yv, RES['A1']['val'], NAMES)['macro_f1']; f2v = metrics(yv, RES['A2']['val'], NAMES)['macro_f1']
BASE = 'A2' if f2v >= f1v else 'A1'
print(f'validation macro-F1: A1 {f1v:.3f}, A2 {f2v:.3f} -> the blend uses {BASE}')
FC = {k: feat_conf(G[k]) for k in ('val', 'test')}
print('share of images with feature confidence (val/test):', {k: round(float(np.mean(~np.isnan(v))), 3) for k, v in FC.items()})

ALPHAS = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
rows = []
for a in ALPHAS:
    pv = fuse(RES[BASE]['val'], PB['val'], FC['val'], a)
    rows.append(dict(alpha=a, **{k: round(v, 3) for k, v in metrics(yv, pv, NAMES).items() if k in ('acc', 'macro_f1', 'bal_acc')}))
tune = pd.DataFrame(rows); print(tune.to_string(index=False))
ALPHA = float(tune.sort_values(['macro_f1', 'acc', 'alpha'], ascending=[False, False, True]).iloc[0]['alpha'])   # ties -> smaller alpha
print('chosen alpha (validation):', ALPHA, '(0 means the feature model did not help on validation)')

FUSED = {k: fuse(RES[BASE][k], PB[k], FC[k], ALPHA) for k in ('val', 'test')}
ths = np.round(np.linspace(0.0, 0.95, 20), 2)
cv_val = coverage_curve(FUSED['val'], yv, ths)
ok = cv_val[(cv_val['accuracy'] >= CFG['abstain_target_acc']) & (cv_val['coverage'] > 0.05)]
THRESH = float(ok['threshold'].min()) if len(ok) else 0.0
print(f"abstention threshold: {THRESH} " + ('(reaches the target on validation)' if len(ok) else '(target accuracy NOT reachable on validation; no abstention applied)'))
cv_test = coverage_curve(FUSED['test'], yt, ths)
fig, ax = plt.subplots(figsize=(5.5, 3.4)); ax.plot(cv_test['coverage'], cv_test['accuracy'], 'o-', label='test'); ax.plot(cv_val['coverage'], cv_val['accuracy'], 's--', label='validation', alpha=.6)
ax.set_xlabel('coverage (share of images answered)'); ax.set_ylabel('accuracy on answered images'); ax.set_ylim(0, 1.02); ax.legend(); ax.grid(alpha=.3); ax.set_title('answer fewer, be right more often')
plt.tight_layout(); plt.show()
kept = FUSED['test'].max(1) >= THRESH
print(f"TEST at threshold {THRESH}: answers {kept.mean():.1%} of images, accuracy on answered {(FUSED['test'].argmax(1)[kept] == yt[kept]).mean():.3f}")
""")

md(r"""
## 9. Report, save, download
**The summary table** puts every model side by side on the **group-split test set** (the honest one), and on the **subset of test images where Phase 3 features exist** (where Model B and fusion can matter). It also lists the official-split baseline score, so the cost of near-copies is visible.
**Saved artifacts** (zip for notebook 04c): the three models, the fusion settings, the group assignment of every image, and the validation/test probabilities. Colab erases its disk when the session ends, so download the zip.
""")
code(r"""
rows = []
usable_t = (DF.loc[G['test'], 'features_usable'].values == 1)
def add(name, prob, note=''):
    m = metrics(yt, prob, NAMES); rows.append(dict(model=name, subset='all test images', **{k: round(v, 3) for k, v in m.items()}))
    if usable_t.any():
        m2 = metrics(yt[usable_t], prob[usable_t], NAMES); rows.append(dict(model=name, subset='test images WITH features', **{k: round(v, 3) for k, v in m2.items()}))
add('A1 whole-image CNN', RES['A1']['test']); add('A2 crop CNN', RES['A2']['test']); add(f'FUSION ({BASE} + Model B, alpha={ALPHA})', FUSED['test'])
SUMMARY = pd.DataFrame(rows)
print('=== GROUP-SPLIT TEST (honest) ==='); print(SUMMARY.to_string(index=False))
if OFFICIAL:
    print('\n=== OFFICIAL-SPLIT BASELINE (optimistic): A1 ===')
    print('test:', {k: round(v, 3) for k, v in OFFICIAL['test'].items()})
if REHEARSAL:
    print('\nREHEARSAL RUN: tiny data and epochs. These numbers only prove the pipeline works.')

from sklearn.metrics import confusion_matrix
best_prob = FUSED['test']; pred = best_prob.argmax(1)
cm = confusion_matrix(yt, pred, labels=range(len(NAMES)))
fig, ax = plt.subplots(figsize=(7, 6)); ax.imshow(cm, cmap='Blues'); ax.set_xticks(range(len(NAMES)), NAMES, rotation=90); ax.set_yticks(range(len(NAMES)), NAMES)
ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title('fusion: group-split test confusion matrix')
for i in range(len(NAMES)):
    for j in range(len(NAMES)):
        if cm[i, j]: ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7, color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.tight_layout(); plt.show()
""")
code(r"""
import joblib
settings = dict(classes=NAMES, base_cnn=BASE, alpha=ALPHA, abstain_threshold=THRESH, abstain_target_acc=CFG['abstain_target_acc'],
                b_columns=B_COLS, group_tau=TAU, img_size=S, mean=MEAN, std=STD, rehearsal=bool(REHEARSAL), epochs=EPOCHS, seed=CFG['seed'])
(MODELS / 'fusion.json').write_text(json.dumps(settings, indent=1), encoding='utf-8')
if MODEL_B is not None:
    joblib.dump(dict(trees=MODEL_B, logreg=LOGREG), MODELS / 'model_b.joblib')
DF[['chord', 'split', 'group', 'gsplit']].to_csv(MODELS / 'groups.csv')
np.savez_compressed(MODELS / 'predictions.npz', val_ids=np.array(G['val']), test_ids=np.array(G['test']), y_val=yv, y_test=yt,
                    a1_val=RES['A1']['val'], a1_test=RES['A1']['test'], a2_val=RES['A2']['val'], a2_test=RES['A2']['test'],
                    b_val=PB['val'], b_test=PB['test'], fused_val=FUSED['val'], fused_test=FUSED['test'])
SUMMARY.to_csv(MODELS / 'summary.csv', index=False)
zip_out = shutil.make_archive(str(Path(PROJECT_DIR) / 'p4_models'), 'zip', MODELS)
print('Created', zip_out, f'({os.path.getsize(zip_out) / 1e6:.1f} MB)')
""")
code(r"""
# colab-only
from google.colab import files
files.download(f'{PROJECT_DIR}/p4_models.zip')
""")

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "04b_train_models.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
