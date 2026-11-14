"""Generates notebooks/train_colab.ipynb (kept as code so the notebook diffs cleanly).

Usage: python scripts/make_notebook.py
"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# Guitar Chord Recognition: Stage 1 (visual chord classifier)
Fine-tunes a pretrained EfficientNet-B0 on `j-asra-j/guitar-chords` (14 classes: A-G major/minor).
**Runtime > Change runtime type > T4 GPU** before running.
""")
code("""
import torch, os
assert torch.cuda.is_available(), "Enable a GPU: Runtime > Change runtime type > T4 GPU"
print(torch.cuda.get_device_name(0))
!pip -q install datasets
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/guitar-chord-recognition'
os.makedirs(OUT, exist_ok=True)
""")
md("## 1. Config")
code("""
CFG = dict(
    repo='j-asra-j/guitar-chords',
    img_size=224, batch_size=32, epochs=15,
    lr_head=1e-3, lr_backbone=1e-4, weight_decay=1e-4,
    label_smoothing=0.1, seed=42,
    extra_val_frac=0.15,          # carved from train for classes with no val images
    own_test_dir='/content/drive/MyDrive/guitar-chord-recognition/own_test',  # optional: own_test/<Chord>/*.jpg
)
""")
md("## 2. Load data (Parquet branch; the raw repo has filenames that break on some OSes)")
code("""
import numpy as np, random
from datasets import load_dataset
R = f"hf://datasets/{CFG['repo']}@refs%2Fconvert%2Fparquet/default/"
ds = load_dataset('parquet', data_files={
    'train': [R+'train/0000.parquet', R+'train/0001.parquet'],
    'validation': R+'validation/0000.parquet',
    'test': R+'test/0000.parquet'})
names = ds['train'].features['label'].names
print(names, {k: len(v) for k, v in ds.items()})
random.seed(CFG['seed']); np.random.seed(CFG['seed']); torch.manual_seed(CFG['seed'])
""")
md("""
## 3. Fix the splits
Cm, Fm and Gm have **no** validation/test images in the official split. We move a random slice of their train images into validation so they can be monitored.
**Caveat:** the data contains augmented near-copies, so these three classes' scores will be optimistic. Use your own photos (section 8) for a real test.
""")
code("""
from collections import Counter
val_counts = Counter(ds['validation']['label'])
missing = [i for i in range(len(names)) if val_counts.get(i, 0) == 0]
print('classes missing from val:', [names[i] for i in missing])

labels = np.array(ds['train']['label'])
move = []
for c in missing:
    idx = np.where(labels == c)[0]
    move += list(np.random.choice(idx, int(len(idx) * CFG['extra_val_frac']), replace=False))
move = set(int(i) for i in move)
keep = [i for i in range(len(labels)) if i not in move]

from datasets import concatenate_datasets
train_ds = ds['train'].select(keep)
val_ds = concatenate_datasets([ds['validation'], ds['train'].select(sorted(move))])
test_ds = ds['test']
print(len(train_ds), len(val_ds), len(test_ds))
""")
md("## 4. Transforms and loaders\nMild augmentation only (data is already heavily augmented). **No horizontal flip**: it turns a right-hand chord shape into a left-hand one.")
code("""
from torchvision import transforms as T
from torch.utils.data import DataLoader
S = CFG['img_size']
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = T.Compose([T.Resize((S, S)), T.RandomResizedCrop(S, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
                      T.RandomRotation(10), T.ColorJitter(0.3, 0.3, 0.2, 0.02), T.ToTensor(), T.Normalize(mean, std)])
eval_tf = T.Compose([T.Resize((S, S)), T.ToTensor(), T.Normalize(mean, std)])

def make(tf):
    def f(b): return {'x': [tf(im.convert('RGB')) for im in b['image']], 'y': b['label']}
    return f
def collate(b): return torch.stack([i['x'] for i in b]), torch.tensor([i['y'] for i in b])
def loader(d, tf, shuffle):
    return DataLoader(d.with_transform(make(tf)), batch_size=CFG['batch_size'], shuffle=shuffle,
                      num_workers=2, collate_fn=collate, pin_memory=True)
train_dl, val_dl, test_dl = loader(train_ds, train_tf, True), loader(val_ds, eval_tf, False), loader(test_ds, eval_tf, False)
""")
md("## 5. Model\nEfficientNet-B0 (ImageNet pretrained), new 14-way head, class-weighted loss for imbalance.")
code("""
import torch.nn as nn
from torchvision import models
dev = 'cuda'
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(names))
model = model.to(dev)

cnt = np.bincount(np.array(train_ds['label']), minlength=len(names))
w = torch.tensor(cnt.sum() / (len(names) * cnt), dtype=torch.float32, device=dev)
crit = nn.CrossEntropyLoss(weight=w, label_smoothing=CFG['label_smoothing'])
opt = torch.optim.AdamW([
    {'params': model.features.parameters(), 'lr': CFG['lr_backbone']},
    {'params': model.classifier.parameters(), 'lr': CFG['lr_head']}], weight_decay=CFG['weight_decay'])
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[CFG['lr_backbone'], CFG['lr_head']],
                                            total_steps=CFG['epochs'] * len(train_dl))
scaler = torch.amp.GradScaler('cuda')
""")
md("## 6. Train")
code("""
def evaluate(dl):
    model.eval(); P, Y = [], []
    with torch.no_grad(), torch.autocast('cuda'):
        for x, y in dl:
            P.append(model(x.to(dev)).argmax(1).cpu()); Y.append(y)
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    return (P == Y).mean(), P, Y

best = 0
for ep in range(CFG['epochs']):
    model.train(); tot = 0
    for x, y in train_dl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * len(x)
    acc, _, _ = evaluate(val_dl)
    print(f"epoch {ep+1:02d}  train_loss {tot/len(train_ds):.3f}  val_acc {acc:.3f}")
    if acc > best:
        best = acc
        torch.save({'model': model.state_dict(), 'classes': names, 'img_size': S, 'mean': mean, 'std': std},
                   f'{OUT}/best.pt')
print('best val acc', best)
""")
md("## 7. Evaluate (per-class, not just overall)")
code("""
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
ck = torch.load(f'{OUT}/best.pt'); model.load_state_dict(ck['model'])
for nm, dl in [('val', val_dl), ('test', test_dl)]:
    acc, P, Y = evaluate(dl)
    print(f'=== {nm}: acc {acc:.3f}')
    print(classification_report(Y, P, labels=range(len(names)), target_names=names, zero_division=0))
cm = confusion_matrix(Y, P, labels=range(len(names)))
fig, ax = plt.subplots(figsize=(8, 7)); ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(names)), names, rotation=90); ax.set_yticks(range(len(names)), names)
ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title('Test confusion matrix'); plt.show()
""")
md("""
## 8. Honest test on your own photos
Put photos in Drive at `guitar-chord-recognition/own_test/<Chord>/` (e.g. `own_test/Am/img1.jpg`), using the same names as the classes. Take them with your own guitar, from the angle your final camera will use. 20+ per chord is a good start. **This score is the one that tells you whether the model works.**
""")
code("""
import glob
from PIL import Image
if os.path.isdir(CFG['own_test_dir']):
    P, Y = [], []
    model.eval()
    for c, nm in enumerate(names):
        for f in glob.glob(f"{CFG['own_test_dir']}/{nm}/*"):
            x = eval_tf(Image.open(f).convert('RGB')).unsqueeze(0).to(dev)
            with torch.no_grad(): P.append(model(x).argmax(1).item()); Y.append(c)
    P, Y = np.array(P), np.array(Y)
    print(f'own photos: {len(Y)}  accuracy {(P==Y).mean():.3f}')
    print(classification_report(Y, P, labels=range(len(names)), target_names=names, zero_division=0))
else:
    print('No own_test folder yet. Add one (see above).')
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "train_colab.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
