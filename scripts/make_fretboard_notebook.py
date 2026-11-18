"""Generates notebooks/02_fretboard_detection.ipynb (Phase 2: classical OpenCV fretboard detection).

Usage: python scripts/make_fretboard_notebook.py

Cells whose first line is '# colab-only' need Google Colab. Everything else runs anywhere, so the notebook
can also be executed end to end locally for testing.
"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


# --------------------------------------------------------------------------------------------------
md(r"""
# Notebook 02: Guitar fretboard detection with classical OpenCV

**Goal:** from a photo of someone playing guitar, find the **fretboard geometry**: the neck region, the **6 strings**, the **fret lines** and the **nut**.
**Not in this notebook:** chord recognition, fingertip-to-fret mapping, or any trained / pretrained detection model. Only classical computer vision.

### The idea in one paragraph
Strings are long, thin, evenly spaced, nearly parallel lines. Frets are shorter lines roughly perpendicular to the strings. So:
1. build a **ridge map** (responds to thin lines, which is what strings are) and an **edge map** (good for the thicker fret lines),
2. **search every angle** for the direction where 6 evenly spaced lines line up in the ridge map (a "comb" search); that direction is the neck,
3. **rotate our coordinates** so the neck is horizontal (this is how we handle any guitar angle),
4. fit **each of the 6 strings** individually (so perspective, where strings fan out, is handled),
5. find the **frets** as almost-vertical Hough segments crossing the strings, and a **nut** candidate,
6. test a few candidate directions, keep the best, map everything back onto the original photo and **save it as JSON** for the next notebook.

### Honest warning
Classical methods are fragile. They work on clear, well-lit photos where strings and frets show as crisp straight edges. They struggle with hands covering the frets, strong perspective, motion blur, low contrast and cluttered backgrounds. This notebook measures its own failures (section 9) and explains where a learned model would be needed (section 10).

### Cells
0. Setup, 1. Config and helpers, 2. Load images, 3. Preprocessing + ridge map, 4. Line segments (for frets), 5. Neck direction (comb search), 6. Strings, 7. Frets and nut, 8. Full pipeline + visualization, 9. Evaluation and saving, 10. Limitations
""")

md(r"""
## 0. Setup (Google Colab)
**What:** creates the working folders inside the Colab session. No Google Drive needed: you upload a **zip file of images** in section 2, and download a zip of the results at the end (section 9).
**Why:** Colab's disk is temporary and is wiped when the session ends, so anything you want to keep (the saved JSON results) must be downloaded before you close it.
""")
code(r"""
# colab-only
import os
PROJECT_DIR = '/content/Guitar-Chord-Recognition'      # temporary folder on the Colab machine
for sub in ['data/sample_images', 'outputs/fretboard']:
    os.makedirs(f'{PROJECT_DIR}/{sub}', exist_ok=True)
print('Project folder:', PROJECT_DIR)
""")

md(r"""
## 1. Config and small helper functions
**What:** imports, tunable settings in one place (`CFG`), and a `show_images` helper to display OpenCV images.
**Why:** one place to tweak parameters; and OpenCV stores colors as **BGR** while matplotlib expects **RGB**, so the helper converts for display.
""")
code(r"""
import os, json, glob, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CFG = dict(
    max_side=800,            # resize so the longest image side is at most this many pixels (speed + consistent scale)
    canny_sigma=0.33,        # Canny thresholds are set from the image's median brightness (used for the fret lines)
    hough_threshold=40,      # votes a line needs in the Hough transform (higher = fewer, surer lines)
    hough_min_len_frac=0.08, # shortest accepted segment, as a fraction of the longest image side
    hough_max_gap=8,         # gaps (px) in an edge line that are bridged
    ridge_sigmas=(1.2, 2.0), # line widths (px) the ridge filter looks for; thin strings, slightly thicker fret wires
    angle_step=2.0,         # the neck-direction search tries every this-many degrees
    n_angle_candidates=4,    # how many different neck directions are fully tested; the best-scoring one wins
    string_gap_min_frac=0.015,  # smallest allowed spacing between neighbouring strings (x longest image side)
    string_gap_max_frac=0.09,   # largest allowed spacing between neighbouring strings
    string_min_extent_frac=0.2, # strings must run along at least this fraction of the image (rejects small noisy patches)
    string_min_density=0.5,     # at least this fraction of each string's length must show real line evidence
    fret_max_tilt=30,        # frets must be within this many degrees of perpendicular to the strings
    fret_min_cover=0.5,      # a fret must span at least this fraction of the neck width
)

# Where outputs go. PROJECT_DIR is defined in cell 0.
OUT_DIR = Path(PROJECT_DIR) / 'outputs' / 'fretboard'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def show_images(images, titles, cols=3, size=4):
    # Display a list of images in a grid. Grayscale images (2D) are shown in gray.
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
    axes = np.atleast_1d(axes).ravel()
    for ax, im, t in zip(axes, images, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))   # OpenCV is BGR, matplotlib is RGB
        ax.set_title(t, fontsize=9); ax.axis('off')
    for ax in axes[len(images):]:
        ax.axis('off')
    plt.tight_layout(); plt.show()
""")

md(r"""
## 2. Load images
**What:** builds the list `IMAGES` of `(name, image)` pairs. Pick one source with `IMAGE_SOURCE`:
- `'zip'` (default): a file picker opens, you choose **`guitar_chords_sample.zip`**, and the notebook unzips it and loads every image inside. The zip can contain subfolders (the sample zip has one folder per chord). You can also zip your own photos and use them the same way.
- `'upload'`: pick individual image files instead of a zip.
- `'dataset'`: downloads photos straight from Hugging Face (needs internet from Colab, no upload).

**Why:** every later step needs images in memory. **Key function:** `cv2.imread` reads an image file into a NumPy array of shape (height, width, 3) with BGR colors (`cv2.imdecode` does the same from raw bytes).
`MAX_IMAGES` limits how many are used (`None` = all). Images are shuffled with a fixed seed so every run uses the same ones.
""")
code(r"""
import zipfile, sys, subprocess

IMAGE_SOURCE = 'zip'       # 'zip' | 'upload' | 'dataset'
MAX_IMAGES = None          # e.g. 40 for a quick run; None = use every image
SEED = 7
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

ZIP_PATH = f'{PROJECT_DIR}/data/guitar_chords_sample.zip'
EXTRACT_DIR = Path(PROJECT_DIR) / 'data' / 'sample_images'

IMAGES = []   # list of (name, bgr_image)

if IMAGE_SOURCE == 'zip':
    if not os.path.exists(ZIP_PATH):                      # skip the upload if the zip is already on the machine
        from google.colab import files
        print('Choose guitar_chords_sample.zip in the file picker below...')
        uploaded = files.upload()
        zips = [n for n in uploaded if n.lower().endswith('.zip')]
        if zips:
            Path(ZIP_PATH).write_bytes(uploaded[zips[0]])
        else:                                              # no zip chosen: use any plain images that were picked instead
            print('No .zip chosen, so using the uploaded files as individual images.')
            for fname, data in uploaded.items():
                bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    IMAGES.append((Path(fname).stem, bgr))
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(EXTRACT_DIR)                      # unzip into data/sample_images/
        for f in sorted(EXTRACT_DIR.rglob('*')):
            if f.suffix.lower() in IMG_EXT and '__MACOSX' not in f.parts and not f.name.startswith('.'):
                bgr = cv2.imread(str(f))                   # returns None if the file is not a readable image
                if bgr is not None:
                    parent = f.parent.name if f.parent != EXTRACT_DIR else ''
                    IMAGES.append((f'{parent}_{f.stem}' if parent else f.stem, bgr))

elif IMAGE_SOURCE == 'upload':
    from google.colab import files
    for fname, data in files.upload().items():
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            IMAGES.append((Path(fname).stem, bgr))

elif IMAGE_SOURCE == 'dataset':
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'], check=True)
    from datasets import load_dataset
    R = 'hf://datasets/j-asra-j/guitar-chords@refs%2Fconvert%2Fparquet/default/'
    ds = load_dataset('parquet', data_files={'validation': R + 'validation/0000.parquet', 'test': R + 'test/0000.parquet'})
    names_cls = ds['validation'].features['label'].names
    for sp in ('validation', 'test'):
        for i in range(len(ds[sp])):
            row = ds[sp][i]
            bgr = cv2.cvtColor(np.array(row['image'].convert('RGB')), cv2.COLOR_RGB2BGR)
            IMAGES.append((f"{names_cls[row['label']]}_{sp[:3]}{i:03d}", bgr))

random.Random(SEED).shuffle(IMAGES)
if MAX_IMAGES:
    IMAGES = IMAGES[:MAX_IMAGES]
assert IMAGES, 'No images were loaded. Re-run this cell and pick guitar_chords_sample.zip (or image files).'
print(f'{len(IMAGES)} images loaded')
show_images([im for _, im in IMAGES[:8]], [n for n, _ in IMAGES[:8]], cols=4, size=3)
""")

# --------------------------------------------------------------------------------------------------
md(r"""
## 3. Preprocessing: resize, grayscale, blur, edges, and a ridge map
Each step makes the next one more reliable:

| Step | OpenCV function | Why |
|---|---|---|
| Resize | `cv2.resize` | Huge photos are slow, and our pixel thresholds assume a ~800 px image. `INTER_AREA` is the best interpolation for shrinking. |
| Grayscale | `cv2.cvtColor(..., COLOR_BGR2GRAY)` | Edges are changes in brightness; color is not needed and triples the work. |
| Gaussian blur | `cv2.GaussianBlur` | Smooths pixel noise and wood grain so Canny does not report hundreds of tiny fake edges. |
| Canny edges | `cv2.Canny(img, low, high)` | Keeps only strong brightness changes, as 1-pixel-thin lines. A pixel above `high` is an edge; between `low` and `high` it is an edge only if connected to a strong one. We pick the thresholds from the image's **median brightness** so it adapts to dark and bright photos. **Used for the frets** (section 7). |
| Border mask | threshold + `morphologyEx` + `erode` | Your dataset has black borders from rotation augmentation. The border/photo boundary is a perfect straight edge that would fool the line finder, so we erase edges near pure-black areas. |
| **Ridge map** | `cv2.createCLAHE`, `GaussianBlur`, `Sobel` | **Used for the strings** (section 5-6). A guitar string is only 1-2 pixels wide, so Canny sees it as a broken *pair* of edges and most of it is lost. A *ridge filter* instead responds to a thin bright **or** dark line as a single line. We build it from the second derivatives (the Hessian) of the image: across a thin line the brightness curves sharply, along it hardly at all. CLAHE first evens out the contrast, so dark strings on dark wood still show up. |
""")
code(r"""
def ridge_response(gray, valid):
    # Highlights thin bright or dark lines (strings, frets) as one response each, scaled to 0..1.
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray).astype(np.float32) / 255.0   # local contrast boost
    best = np.zeros_like(g)
    for s in CFG['ridge_sigmas']:                                    # two line widths; keep the stronger response
        gs = cv2.GaussianBlur(g, (0, 0), s)
        dxx = cv2.Sobel(gs, cv2.CV_32F, 2, 0, ksize=3)               # second derivatives = how sharply brightness bends
        dyy = cv2.Sobel(gs, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(gs, cv2.CV_32F, 1, 1, ksize=3)
        t = np.sqrt(((dxx - dyy) / 2) ** 2 + dxy ** 2)
        l1 = (dxx + dyy) / 2 + t                                     # the two eigenvalues of the Hessian (l1 >= l2)
        l2 = (dxx + dyy) / 2 - t
        bright = np.maximum(np.where(np.abs(l1) < 0.5 * np.abs(l2), -l2, 0), 0)   # thin BRIGHT line: one strongly negative curvature
        dark = np.maximum(np.where(np.abs(l2) < 0.5 * np.abs(l1), l1, 0), 0)      # thin DARK line: one strongly positive curvature
        best = np.maximum(best, (bright + dark) * s ** 2)            # s**2 makes the two scales comparable
    best *= (valid > 0)
    ref = np.percentile(best[valid > 0], 97) if (valid > 0).any() else 1.0
    return np.clip(best / max(ref, 1e-6), 0, 1).astype(np.float32)   # clipping stops one huge edge from dominating


def preprocess(bgr, max_side=800):
    h0, w0 = bgr.shape[:2]
    scale = min(1.0, max_side / max(h0, w0))                       # never enlarge, only shrink
    img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)                        # 5x5 kernel, sigma chosen automatically

    med = float(np.median(blur))
    lo = int(max(20, (1 - CFG['canny_sigma']) * med))
    hi = int(max(60, (1 + CFG['canny_sigma']) * med))
    edges_raw = cv2.Canny(blur, lo, hi)

    # Mask out the black padding that rotation-augmented images have, and the edge where it meets the photo
    valid = (gray > 8).astype(np.uint8)
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    valid = cv2.erode(valid, np.ones((13, 13), np.uint8))
    edges = cv2.bitwise_and(edges_raw, edges_raw, mask=valid)

    ridge = ridge_response(gray, valid)
    return dict(img=img, gray=gray, blur=blur, edges_raw=edges_raw, edges=edges, ridge=ridge, scale=scale, canny=(lo, hi))


DEMO_INDEX = 0          # which loaded image the step-by-step cells use; change it and re-run from here
demo_name, demo_bgr = IMAGES[DEMO_INDEX]
pre = preprocess(demo_bgr, CFG['max_side'])
print(demo_name, '| original size', demo_bgr.shape[1], 'x', demo_bgr.shape[0], '| scale', round(pre['scale'], 3),
      '| Canny thresholds', pre['canny'])
show_images([pre['img'], pre['gray'], pre['blur'], pre['edges_raw'], pre['edges'], (pre['ridge'] * 255).astype(np.uint8)],
            ['1. resized original', '2. grayscale', '3. Gaussian blur', '4. Canny edges (raw)', '5. edges after border mask',
             '6. ridge map (thin lines; used for strings)'],
            cols=3, size=4.2)
""")

md(r"""
## 4. Line segments with the Hough transform (used for the frets)
**What:** `cv2.HoughLinesP` finds straight line **segments** in the edge image.
**Why:** frets are short, fairly thick straight edges, which Canny + Hough handles well. (Strings are *not* found this way: they are too thin, see sections 5-6.)
**How it works (simple version):** every edge pixel "votes" for all the straight lines that could pass through it. Lines that collect many votes (`threshold`) are real. The `P` means *probabilistic*: it returns segments with start and end points `(x1, y1, x2, y2)` instead of infinite lines, which is what we need.
**Key parameters:** `minLineLength` drops short specks; `maxLineGap` joins a line broken by a finger or glare.
Expect many segments that are **not** frets (hand edges, background); the next steps filter them by direction and spacing.
""")
code(r"""
def seg_lengths(segs):
    return np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])

def seg_angles(segs):
    # direction of each segment in degrees, in [0, 180). 0 = pointing right, 90 = pointing down.
    return np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180.0

def signed_angle(deg):
    # wraps angles to [-90, 90) so "almost horizontal" is near 0 whichever way the line points
    return (deg + 90.0) % 180.0 - 90.0


def hough_segments(edges):
    h, w = edges.shape
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=CFG['hough_threshold'],
                            minLineLength=int(CFG['hough_min_len_frac'] * max(h, w)),
                            maxLineGap=CFG['hough_max_gap'])
    if lines is None:
        return np.zeros((0, 4))
    return lines.reshape(-1, 4).astype(float)     # shape differs between OpenCV versions; this handles both


segs = hough_segments(pre['edges'])
vis = pre['img'].copy()
for x1, y1, x2, y2 in segs.astype(int):
    cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
print(len(segs), 'line segments found')
show_images([pre['img'], vis], ['resized original', 'all Hough segments (red)'], cols=2, size=6)
""")
md(r"""
## 5. Which way does the neck point? (a comb search over angles)
**What:** for every direction from 0 to 180 degrees (every `angle_step`), we rotate the **ridge map** so that direction is horizontal and add up each row. A row that lies along a string collects a lot of ridge evidence, so each string shows up as a **peak** in that row profile. Then we slide a **comb** of 6 equally spaced teeth along the profile and ask: how well do 6 evenly spaced peaks line up with it? The direction with the best comb score is the neck direction.
**Why this instead of a histogram of line segments:** the old approach histogrammed Hough segment angles and trusted the tallest peak. Strings are thin, so very few of their segments survived, while table grain, wall edges and image borders produced plenty. The tallest peak was often not the neck. Here the evidence comes straight from the ridge map, and "6 evenly spaced parallel lines" is part of the test itself.
**Details that matter:**
- Before summing, each row is **averaged along its length** (`cv2.blur`) and only the *coherent* part is kept. Long unbroken lines survive; speckled texture does not.
- The comb can ignore one weak tooth, so one invisible or occluded string does not break the match.
- The spacing between teeth is limited (`string_gap_min_frac`, `string_gap_max_frac`). Without a lower limit, dense texture fakes a perfect comb at 4-5 px spacing.
- We keep the **top few different directions** (not just the winner) and run the full detector on each in section 8, then keep the best, because the winning comb is not always the neck (table edges can win).
**Key functions:** `cv2.warpAffine` (rotates the map), `np.sort`/`np.sum` (comb scoring). Rotation uses a canvas big enough to hold the whole rotated image (its diagonal), so no corner of the photo is cut off.
""")
code(r"""
def neck_matrix(angle, w, h):
    # Rotation that makes lines pointing at `angle` degrees horizontal. The output canvas is the image diagonal wide,
    # so nothing is cropped when the photo is rotated. Returns the 2x3 matrix and the canvas size.
    D = int(np.ceil(np.hypot(w, h)))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += D / 2 - w / 2
    M[1, 2] += D / 2 - h / 2
    return M, D


def to_neck_frame(segs, M):
    # Apply the same rotation to segment endpoints.
    pts = segs.reshape(-1, 2) @ M[:, :2].T + M[:, 2]
    return pts.reshape(-1, 4)


def comb_search(profile, d_min, d_max, n=6, tol=2):
    # profile: ridge evidence per row. Find the best place and spacing d for n equally spaced teeth.
    p = np.convolve(profile, np.ones(3) / 3, mode='same')
    base = np.convolve(p, np.ones(41) / 41, mode='same')                  # slowly varying background level
    p = np.maximum(p - base, 0).astype(np.float32)                         # keep only peaks standing above it
    pd_ = cv2.dilate(p.reshape(-1, 1), np.ones((2 * tol + 1, 1), np.uint8)).ravel()   # each tooth may be +-tol rows off
    best = (0.0, 0, 0)                                                     # (score, first row, spacing)
    for d in range(d_min, d_max + 1):
        offs = np.arange(n) * d
        starts = np.arange(0, len(pd_) - offs[-1])
        if len(starts) == 0:
            break
        v = np.sort(pd_[starts[:, None] + offs[None, :]], axis=1)          # evidence under each tooth, weakest first
        # drop the weakest tooth (a string may be hidden) but reward teeth of similar strength
        score = v[:, 1:].sum(1) * np.sqrt(v[:, 1] / (v[:, -1] + 1e-9))
        i = int(np.argmax(score))
        if score[i] > best[0]:
            best = (float(score[i]), int(starts[i]), d)
    return best


def neck_candidates(ridge, n=None):
    # Returns the best few distinct neck directions as dicts, and (angles, scores) for plotting.
    n = n or CFG['n_angle_candidates']
    h, w = ridge.shape
    big = max(h, w)
    d_min, d_max = max(8, int(CFG['string_gap_min_frac'] * big)), int(CFG['string_gap_max_frac'] * big)
    phis = np.arange(0, 180, CFG['angle_step'])
    scores, info = np.zeros(len(phis)), []
    for i, phi in enumerate(phis):
        M, D = neck_matrix(phi, w, h)
        rot = cv2.warpAffine(ridge, M, (D, D))                             # ridge map with direction phi made horizontal
        coherent = np.maximum(cv2.blur(rot, (41, 1)) - 0.35, 0)            # keep only long unbroken lines
        sc, y_start, d = comb_search(coherent.sum(axis=1), d_min, d_max)
        scores[i] = sc
        info.append((y_start, d))
    picked = []
    for i in np.argsort(scores)[::-1]:                                     # best first; skip directions too close to a pick
        if scores[i] <= 0:
            break
        if all(min(abs(phis[i] - phis[j]), 180 - abs(phis[i] - phis[j])) >= 10 for j in picked):
            picked.append(int(i))
        if len(picked) == n:
            break
    cands = [dict(angle=float(phis[i]), comb_score=float(scores[i]), y_start=info[i][0], d=info[i][1]) for i in picked]
    return cands, (phis, scores)


cands, (phis, comb_scores) = neck_candidates(pre['ridge'])
print('candidate neck directions (degrees, comb score, string spacing px):',
      [(c['angle'], round(c['comb_score']), c['d']) for c in cands])

h, w = pre['img'].shape[:2]
plt.figure(figsize=(7, 2.6))
plt.plot(phis, comb_scores); plt.title('comb score per direction (red = candidate neck directions)')
plt.xlabel('direction (degrees)')
for c in cands:
    plt.axvline(c['angle'], color='r', ls='--')
plt.tight_layout(); plt.show()
rotated_views, rotated_titles = [], []
for c in cands[:2]:
    M, D = neck_matrix(c['angle'], w, h)
    rotated_views.append(cv2.warpAffine(pre['img'], M, (D, D)))
    rotated_titles.append(f"view rotated by {c['angle']:.0f} deg (neck should look horizontal)")
show_images(rotated_views, rotated_titles, cols=2, size=5)
""")

md(r"""
## 6. Find the six strings
The comb gave us a rough position for each string in the neck-aligned view, but it assumed all strings are parallel. In a real photo the camera looks at the neck at an angle, so the strings **fan out** (they get closer together toward the headstock). We fix this per string:

1. **Slope:** for each string, try small tilts (up to 12 degrees) and keep the one that collects the most ridge evidence along the line.
2. **Smooth the slopes:** the six slopes should change smoothly from the first string to the last (a fan), so we fit a straight line through them and keep each string's slope within a couple of degrees of that fit. This stops two strings from "stealing" the same ridge and crossing each other, which is what the independent fits did on strongly angled necks.
3. **Least squares:** refit each string to the ridge pixels close to it (`np.polyfit`, weighted by ridge strength), three times with a narrowing search band.
4. **Extent:** walk along each string and mark where there really is ridge evidence (gaps up to 30 px from a finger are bridged). The neck is where **at least 4 of the 6** strings have evidence. `density` is the fraction of that stretch that is actually supported, which later feeds the confidence.
5. **Sanity check:** the strings must not cross inside the neck, and the spacing irregularity (coefficient of variation of the gaps) must be small.

**This step can still be wrong.** Dark strings on dark wood, a thumb across the strings, or a table with strong grain in the picture can all look like six lines. Section 8 cross-checks the result against the fret lines, and marks doubtful results `low` confidence.
""")
code(r"""
def string_y(st, x):
    # y position of a string at horizontal position x (strings are lines y = y0 + m*(x - xc))
    return st['y0'] + st['m'] * (x - st['xc'])


def _runs(mask):
    # (start, end) of each run of True values in a 1-D boolean array
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1))


def find_strings(ridge, cand, n_strings=6):
    # Refine the comb's rough guess into 6 fitted string lines in the neck-aligned view. Returns a dict, or None.
    h, w = ridge.shape
    M, D = neck_matrix(cand['angle'], w, h)
    rot = cv2.warpAffine(ridge, M, (D, D))
    ys, xs = np.nonzero(rot > 0.25)                                        # ridge pixels worth fitting to
    wt = rot[ys, xs].astype(float); ys = ys.astype(float); xs = xs.astype(float)
    d = cand['d']
    y_init = [cand['y_start'] + i * d for i in range(n_strings)]
    near = np.zeros(len(ys), bool)
    for yi in y_init:
        near |= np.abs(ys - yi) <= 0.35 * d
    if near.sum() < 50:
        return None
    xc = float(np.average(xs[near], weights=wt[near]))                     # shared pivot column in the middle of the evidence

    # 1. per-string slope search (perspective makes strings fan out)
    slopes = []
    for yi in y_init:
        scores = [(wt[np.abs(ys - (yi + t * (xs - xc))) <= 2.0].sum(), t) for t in np.tan(np.radians(np.arange(-12, 12.1, 1.0)))]
        slopes.append(max(scores)[1])
    # 2. slopes should vary smoothly across the 6 strings: fit a line through them and stay close to it
    idx = np.arange(n_strings)
    a, b = np.polyfit(idx, slopes, 1)
    model = a * idx + b
    lines = []
    for i, yi in enumerate(y_init):
        y0, m = yi, float(model[i])
        # 3. weighted least squares on nearby ridge pixels, with a narrowing band
        for hw in (0.30 * d, 0.22 * d, 0.16 * d):
            sel = np.abs(ys - (y0 + m * (xs - xc))) <= hw
            if sel.sum() < 25:
                break
            A = np.polyfit(xs[sel] - xc, ys[sel], 1, w=wt[sel])
            m = float(np.clip(A[0], model[i] - 0.04, model[i] + 0.04))     # at most ~2.3 degrees away from the fan model
            y0 = float(A[1])
        lines.append((y0, m))

    # 4. where along the neck is there real evidence? (a string may be hidden by fingers over part of its length)
    X = np.arange(D)
    support, count = [], np.zeros(D)
    for y0, m in lines:
        yy = np.rint(y0 + m * (X - xc)).astype(int)
        ok = (yy >= 1) & (yy < D - 1)
        v = np.zeros(D)
        for dy in (-1, 0, 1):
            v[ok] = np.maximum(v[ok], rot[yy[ok] + dy, X[ok]])
        s = cv2.morphologyEx((v > 0.2).astype(np.uint8).reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, 31), np.uint8)).ravel()
        support.append(s); count += s
    good = cv2.morphologyEx((count >= 4).astype(np.uint8).reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, 41), np.uint8)).ravel()
    runs = _runs(good > 0)
    if not runs:
        return None
    xl, xr = max(runs, key=lambda r: count[r[0]:r[1] + 1].sum())
    strings = []
    for (y0, m), s in zip(lines, support):
        dens = float(s[xl:xr + 1].mean())
        strings.append(dict(xc=xc, y0=y0, m=m, x1=float(xl), x2=float(xr), length=float((xr - xl) * dens), density=dens))

    # 5. sanity checks: strings must stay in order (no crossing) over the whole stretch, with fairly even spacing
    gaps_l = np.diff([string_y(s, xl) for s in strings]); gaps_r = np.diff([string_y(s, xr) for s in strings])
    if gaps_l.min() < 2 or gaps_r.min() < 2:
        return None
    gaps = np.diff([s['y0'] for s in strings])
    cv = float(gaps.std() / gaps.mean())
    return dict(M=M, strings=strings, cv=cv, xl=float(xl), xr=float(xr), extent=float(xr - xl),
                density=float(np.mean([s['density'] for s in strings])))


# Demo on the best candidate direction
fit_demo = find_strings(pre['ridge'], cands[0]) if cands else None
if fit_demo:
    M_demo = fit_demo['M']; D_demo = int(np.ceil(np.hypot(w, h)))
    view = cv2.warpAffine(pre['img'], M_demo, (D_demo, D_demo))
    for st in fit_demo['strings']:
        cv2.line(view, (int(st['x1']), int(string_y(st, st['x1']))), (int(st['x2']), int(string_y(st, st['x2']))), (255, 255, 0), 2)
    print(f"6 strings fitted | neck stretch {fit_demo['extent']:.0f} px | evidence density {fit_demo['density']:.2f}"
          f" | spacing irregularity {fit_demo['cv']:.2f} (lower is better)")
    show_images([view], ['neck-aligned view: cyan = fitted strings'], cols=1, size=7)
else:
    strings_demo = []
    print('No valid set of 6 strings for this image and direction. This is a real failure; see section 9.')
""")

md(r"""
## 7. Find the frets and the nut
**Frets:** in the neck frame they are almost-vertical segments. For each, we measure where it crosses the middle of the string band and how much of the neck width it covers. Segments from the same fret (a fret wire has two edges) are merged. A candidate counts as a fret only if it spans at least half the neck width.

**Nut:** the nut is where the fretboard starts (at the headstock end). Frets get **closer together toward the body**, so the end of the fret sequence with the *wider* gaps is the headstock side. We call the extreme fret on that side the **nut candidate**.
This is the weakest step. A photo can show the nut, or it can start at fret 1 or later. We cannot tell the difference from line geometry alone, so the nut is labeled with a **confidence**: `medium` if the strings visibly continue past it (headstock in view), otherwise `low`.

Also note: strings are numbered 1 to 6 **by position in the neck frame**, not by guitar tuning. We cannot tell the high E from the low E here.
""")
code(r"""
def merge_intervals(iv):
    # total length covered by a set of (start, end) intervals, counting overlaps once
    iv = sorted(iv); total, cur_s, cur_e = 0.0, None, None
    for a, b in iv:
        if cur_e is None or a > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def find_frets(sr, strings, spacing):
    # strings: the 6 chosen strings. spacing: mean distance between neighboring strings.
    xc = strings[0]['xc']
    y_top = strings[0]['y0'] - spacing / 2          # neck band edges at x = xc
    y_bot = strings[-1]['y0'] + spacing / 2
    band = y_bot - y_top
    y_mid = (y_top + y_bot) / 2

    ang = np.abs(signed_angle(seg_angles(sr)))
    s = sr[ang >= 90 - CFG['fret_max_tilt']].copy()                       # keep almost-vertical segments
    if len(s) == 0:
        return [], dict(y_top=y_top, y_bot=y_bot, y_mid=y_mid)
    swap = s[:, 1] > s[:, 3]                                              # make every segment point downward
    s[swap] = s[swap][:, [2, 3, 0, 1]]

    items = []
    for x1, y1, x2, y2 in s:
        if y2 - y1 < 1e-6:
            continue
        a, b = max(y1, y_top), min(y2, y_bot)                              # part of the segment inside the neck band
        if b - a < 0.25 * band:
            continue
        k = (x2 - x1) / (y2 - y1)                                          # horizontal shift per unit of y
        items.append((x1 + k * (y_mid - y1), k, a, b, y2 - y1))            # (x at band middle, k, y-range, full length)
    if not items:
        return [], dict(y_top=y_top, y_bot=y_bot, y_mid=y_mid)

    items.sort(key=lambda t: t[0])
    tol = max(3.0, 0.8 * spacing)
    groups, cur = [], [items[0]]
    for it in items[1:]:
        if it[0] - cur[-1][0] <= tol:
            cur.append(it)
        else:
            groups.append(cur); cur = [it]
    groups.append(cur)

    frets = []
    for g in groups:
        cover = merge_intervals([(a, b) for _, _, a, b, _ in g]) / band
        if cover < CFG['fret_min_cover']:
            continue
        wgt = np.array([b - a for _, _, a, b, _ in g])
        frets.append(dict(xm=float(np.average([t[0] for t in g], weights=wgt)),
                          k=float(np.average([t[1] for t in g], weights=wgt)), cover=float(cover),
                          span=float(max(t[4] for t in g))))             # longest piece seen = real neck width
    return frets, dict(y_top=y_top, y_bot=y_bot, y_mid=y_mid)


def pick_nut(frets, strings):
    # Returns (nut or None, remaining frets). Needs 3+ lines so we can compare the gaps at both ends.
    if len(frets) < 3:
        return None, frets
    xs = [f['xm'] for f in frets]
    first_gap, last_gap = xs[1] - xs[0], xs[-1] - xs[-2]
    if first_gap >= last_gap:                                  # wider gaps on the left -> headstock on the left
        nut, rest, gap = frets[0], frets[1:], first_gap
        continues = min(st['x1'] for st in strings) <= nut['xm'] - 0.5 * gap
    else:
        nut, rest, gap = frets[-1], frets[:-1], last_gap
        continues = max(st['x2'] for st in strings) >= nut['xm'] + 0.5 * gap
    nut = dict(nut, confidence='medium' if continues else 'low')
    return nut, rest


# Demo
if fit_demo:
    strings_demo = fit_demo['strings']
    sr = to_neck_frame(segs, M_demo)                                       # Hough segments in the neck-aligned view
    spacing_demo = float(np.mean(np.diff([c['y0'] for c in strings_demo])))
    frets_demo, band_demo = find_frets(sr, strings_demo, spacing_demo)
    nut_demo, frets_rest = pick_nut(frets_demo, strings_demo)
    view2 = view.copy()
    def fret_line(f, band):
        return ((f['xm'] + f['k'] * (band['y_top'] - band['y_mid']), band['y_top']),
                (f['xm'] + f['k'] * (band['y_bot'] - band['y_mid']), band['y_bot']))
    for f in frets_rest:
        p1, p2 = fret_line(f, band_demo)
        cv2.line(view2, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 2)
    if nut_demo:
        p1, p2 = fret_line(nut_demo, band_demo)
        cv2.line(view2, tuple(map(int, p1)), tuple(map(int, p2)), (255, 0, 255), 3)
    print(f'{len(frets_rest)} frets found; nut candidate:', nut_demo['confidence'] + ' confidence' if nut_demo else 'none')
    show_images([view2], ['neck-aligned view: green = frets, magenta = nut candidate'], cols=1, size=7)
else:
    print('No strings found for this demo image, so frets are not attempted. This is a real failure; see section 9.')
""")

# --------------------------------------------------------------------------------------------------
md(r"""
## 8. The full pipeline and the visualization
**What:** `detect_fretboard(bgr, name)` runs every step above for each candidate neck angle, scores each one (comb strength, boosted by the number of fret lines crossing the strings and by how continuous the string evidence is), keeps the best, and converts the coordinates **back to the original image's pixels** (undo the rotation with `cv2.invertAffineTransform`, then undo the resize).
It returns a plain dictionary (the **result format**, saved as JSON in section 9):

| Key | Meaning |
|---|---|
| `image` | name, width, height of the original photo |
| `status` | `ok` (6 strings and 2+ fret lines), `partial` (strings but not enough frets) or `failed` |
| `confidence` | `high` / `low` / `none`. **Use this, not `status` alone.** `high` needs 4+ fret lines, a neck spanning 40%+ of the image, strings covering a sensible share of the neck width (checked against the fret lines) and continuous string evidence (`string_density` 0.6+); false detections on background texture usually come out `low` |
| `failure_reason` | why, when not `ok` |
| `neck_angle_deg` | direction of the strings in the image |
| `fretboard.polygon` | 4 corners of the fretboard region, original pixels |
| `strings` / `frets` / `nut` | each a line with endpoints `p1`, `p2` in original pixels |

**Colors:** orange = fretboard region, cyan = strings, green = frets, magenta = nut.
""")
code(r"""
def _to_original(pts_rot, M, scale):
    inv = cv2.invertAffineTransform(M)
    p = np.asarray(pts_rot, float).reshape(-1, 2) @ inv[:, :2].T + inv[:, 2]
    return (p / scale).round(1).tolist()


def _try_angle(pre, segs, cand):
    # Run strings -> frets -> nut for ONE candidate neck direction and give the result a score for ranking.
    h, w = pre['img'].shape[:2]
    big = max(h, w)
    out = dict(angle=cand['angle'], M=None, strings=[], cv=None, n_frets=0, score=0.0, density=0.0, comb_score=cand['comb_score'])
    fit = find_strings(pre['ridge'], cand)
    if fit is None:
        out.update(status='failed', reason='no 6 evenly spaced, non-crossing strings found along this direction')
        return out
    out.update(M=fit['M'], strings=fit['strings'], cv=fit['cv'], density=fit['density'], xl=fit['xl'], xr=fit['xr'])
    if fit['extent'] < CFG['string_min_extent_frac'] * big or fit['density'] < CFG['string_min_density']:
        out.update(status='failed', reason=f"strings too short ({fit['extent'] / big:.0%} of the image) or too patchy (density {fit['density']:.2f})")
        return out

    strings = fit['strings']
    spacing = float(np.mean(np.diff([c['y0'] for c in strings])))
    sr = to_neck_frame(segs, fit['M'])
    frets, band = find_frets(sr, strings, spacing)
    nut, rest = pick_nut(frets, strings)
    n_lines = len(rest) + (1 if nut else 0)
    # Cross-check: the strings should span about as much width as the fret lines do. If the strings cover much less,
    # we probably grabbed the wrong lines (for example a patch of wood grain).
    all_lines = rest + ([nut] if nut else [])
    ratio = (band['y_bot'] - band['y_top']) / float(np.median([f['span'] for f in all_lines])) if all_lines else None
    # Ranking score: the comb strength, boosted when fret lines cross the strings (a real neck has them, wood grain does not)
    # and by how continuous the string evidence is.
    score = cand['comb_score'] * (1 + 0.3 * min(n_lines, 6)) * fit['density']
    out.update(spacing=spacing, band=band, frets=rest, nut=nut, n_frets=n_lines, band_ratio=ratio, score=score)

    if n_lines >= 2:
        out.update(status='ok', reason=None)
    else:
        out.update(status='partial', reason=f'6 strings found but only {n_lines} fret line(s)')
    return out


def detect_fretboard(bgr, name='image'):
    pre = preprocess(bgr, CFG['max_side'])
    segs = hough_segments(pre['edges'])
    H0, W0 = bgr.shape[:2]
    res = dict(schema_version=1, image=dict(name=name, width=W0, height=H0), status='failed', success=False,
               confidence='none', failure_reason=None, neck_angle_deg=None,
               fretboard=None, strings=[], frets=[], nut=None, diagnostics=dict(n_segments=int(len(segs))))
    cands, _ = neck_candidates(pre['ridge'])
    if not cands:
        res['failure_reason'] = 'no line pattern found (the ridge map is empty)'
        return res

    tries = [_try_angle(pre, segs, c) for c in cands]
    best = max(tries, key=lambda t: t['score'])
    if best['score'] <= 0:                              # nothing valid at all: report the most informative failure
        best = tries[0]

    res['status'], res['success'] = best['status'], best['status'] == 'ok'
    res['failure_reason'], res['neck_angle_deg'] = best['reason'], round(float(best['angle']), 1)
    ratio = best.get('band_ratio')
    res['diagnostics'].update(n_angle_candidates=len(cands), comb_score=round(best['comb_score'], 1),
                              string_density=round(best['density'], 2),
                              string_band_vs_fret_ratio=None if ratio is None else round(float(ratio), 2),
                              string_spacing_cv=None if best['cv'] is None else round(best['cv'], 3),
                              n_fret_lines=best['n_frets'])
    if best['status'] == 'failed':
        return res

    M, scale, strings, sp = best['M'], pre['scale'], best['strings'], best['spacing']
    # Neck extent along the strings comes from where the strings have evidence. Fret lines can extend it a little
    # (frets are sometimes visible where the strings are hidden), but a stray fret far away is ignored.
    xl, xr, basis = best['xl'], best['xr'], 'strings'
    margin = 0.15 * (xr - xl)
    near_x = [f['xm'] for f in best['frets'] + ([best['nut']] if best['nut'] else []) if xl - margin <= f['xm'] <= xr + margin]
    if near_x and (min(near_x) < xl or max(near_x) > xr):
        xl, xr, basis = min(xl, min(near_x)), max(xr, max(near_x)), 'strings+frets'

    top_y = lambda x: string_y(strings[0], x) - sp / 2
    bot_y = lambda x: string_y(strings[-1], x) + sp / 2
    band = best.get('band')

    def fret_ends(f):
        yt, yb = top_y(f['xm']), bot_y(f['xm'])
        return [(f['xm'] + f['k'] * (yt - band['y_mid']), yt), (f['xm'] + f['k'] * (yb - band['y_mid']), yb)]

    res['strings'] = [dict(index=i + 1, p1=_to_original([(xl, string_y(st, xl))], M, scale)[0],
                           p2=_to_original([(xr, string_y(st, xr))], M, scale)[0]) for i, st in enumerate(strings)]
    kept = [f for f in best['frets'] if xl - 1 <= f['xm'] <= xr + 1]
    res['frets'] = [dict(index=i + 1, p1=_to_original([fret_ends(f)[0]], M, scale)[0],
                         p2=_to_original([fret_ends(f)[1]], M, scale)[0]) for i, f in enumerate(kept)]
    if best['nut']:
        e = fret_ends(best['nut'])
        res['nut'] = dict(p1=_to_original([e[0]], M, scale)[0], p2=_to_original([e[1]], M, scale)[0],
                          confidence=best['nut']['confidence'])
    corners = [(xl, top_y(xl)), (xr, top_y(xr)), (xr, bot_y(xr)), (xl, bot_y(xl))]
    res['fretboard'] = dict(polygon=_to_original(corners, M, scale), basis=basis)

    # Confidence: 'ok' only means "geometry was produced". A result is trusted ('high') only if several fret lines
    # were found, the neck spans a large part of the image, the strings cover a sensible share of the neck width,
    # AND the strings show continuous line evidence. Small patches of background texture fail this.
    poly = np.array(res['fretboard']['polygon'])
    length_frac = float(np.hypot(*(poly[1] - poly[0]))) / max(W0, H0)
    res['diagnostics']['neck_length_frac'] = round(length_frac, 2)
    ratio_ok = ratio is not None and ratio >= 0.6              # strings must cover a sensible share of the neck width
    if best['status'] == 'ok' and best['n_frets'] >= 4 and length_frac >= 0.4 and ratio_ok and best['density'] >= 0.6:
        res['confidence'] = 'high'
    else:
        res['confidence'] = 'low'
    return res


COL = dict(region=(0, 165, 255), string=(255, 255, 0), fret=(0, 255, 0), nut=(255, 0, 255))   # BGR colors


def draw_result(bgr, res, alpha=0.22):
    # Draw a result dictionary (works on one loaded from JSON too) on top of the original image.
    out = bgr.copy()
    t = max(1, int(round(max(out.shape[:2]) / 500)))
    P = lambda p: (int(round(p[0])), int(round(p[1])))
    if res.get('fretboard'):
        poly = np.array(res['fretboard']['polygon'], np.int32)
        ov = out.copy(); cv2.fillPoly(ov, [poly], COL['region'])
        out = cv2.addWeighted(ov, alpha, out, 1 - alpha, 0)
        cv2.polylines(out, [poly], True, COL['region'], t + 1, cv2.LINE_AA)
    for s in res.get('strings', []):
        cv2.line(out, P(s['p1']), P(s['p2']), COL['string'], t, cv2.LINE_AA)
    for f in res.get('frets', []):
        cv2.line(out, P(f['p1']), P(f['p2']), COL['fret'], t + 1, cv2.LINE_AA)
    if res.get('nut'):
        cv2.line(out, P(res['nut']['p1']), P(res['nut']['p2']), COL['nut'], t + 2, cv2.LINE_AA)
    return out


res_demo = detect_fretboard(demo_bgr, demo_name)
print('status:', res_demo['status'], '|', res_demo['failure_reason'])
print({k: v for k, v in res_demo.items() if k in ('neck_angle_deg', 'diagnostics')})
show_images([demo_bgr, draw_result(demo_bgr, res_demo)],
            ['original', f"result: {res_demo['status']}  (orange=region, cyan=strings, green=frets, magenta=nut)"],
            cols=2, size=6)
""")

# --------------------------------------------------------------------------------------------------
md(r"""
## 9. Evaluation on all images, and saving the results
**What:** run the detector on every loaded image, save one JSON file per image plus an overlay picture and a `summary.json` into `outputs/fretboard/` (inside the Colab session), then summarize. The last cell of this notebook packs those into a zip and **downloads it to your computer**: do this before closing Colab, because the session's disk is deleted.

**How to read the numbers:** we have no hand-labeled ground truth, so `ok` means *the detector produced a 6-string, 2+ fret result*, **not that it is correct**. In testing, some `ok` results had locked onto a patch of background texture while the real neck was right next to it. That is why we also record a `confidence`. Always look at the pictures: the galleries below let you check the high-confidence results, inspect the doubtful `ok` ones, and see what goes wrong in failures.
""")
code(r"""
def save_result(res, path):
    Path(path).write_text(json.dumps(res, indent=2), encoding='utf-8')

def load_result(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


(OUT_DIR / 'overlays').mkdir(exist_ok=True)
all_results = []
for name, bgr in IMAGES:
    res = detect_fretboard(bgr, name)
    save_result(res, OUT_DIR / f'{name}.json')
    cv2.imwrite(str(OUT_DIR / 'overlays' / f'{name}.jpg'), draw_result(bgr, res))
    all_results.append((name, bgr, res))

df = pd.DataFrame([dict(image=n, status=r['status'], confidence=r['confidence'], strings=len(r['strings']), frets=len(r['frets']),
                        nut=(r['nut'] or {}).get('confidence', '-'), reason=r['failure_reason'] or '')
                   for n, _, r in all_results])
summary = dict(n_images=len(df), **df['status'].value_counts().to_dict(), high_confidence=int((df['confidence'] == 'high').sum()))
(OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('Summary:', summary)
print(f"Saved {len(df)} JSON files + overlays to {OUT_DIR}")
df
""")
code(r"""
def gallery(items, title, limit=6):
    items = items[:limit]
    if not items:
        print(f'{title}: none'); return
    show_images([draw_result(b, r) for _, b, r in items],
                [f"{n} | {r['status']} | conf: {r['confidence']}" + (f" | {r['failure_reason']}" if r['failure_reason'] else '') for n, _, r in items],
                cols=3, size=4.3)

print('=== HIGH-confidence detections (these should be right; check them) ===')
gallery([x for x in all_results if x[2]['confidence'] == 'high'], 'high', limit=6)
print('=== status OK but LOW confidence (some are right, some are false detections; look closely) ===')
gallery([x for x in all_results if x[2]['status'] == 'ok' and x[2]['confidence'] != 'high'], 'ok-low', limit=6)
print('=== Failed / partial detections ===')
gallery([x for x in all_results if x[2]['status'] != 'ok'], 'failed', limit=6)
""")
code(r"""
# Proof that a future notebook can reuse the saved results: reload one JSON and redraw from the file alone.
first_json = sorted(OUT_DIR.glob('*.json'))[0]
if first_json.name != 'summary.json':
    loaded = load_result(first_json)
    src = dict(IMAGES)[loaded['image']['name']]
    print('Loaded', first_json.name, '| status:', loaded['status'], '| keys:', list(loaded.keys()))
    show_images([draw_result(src, loaded)], ['redrawn purely from the saved JSON'], cols=1, size=6)
""")

md(r"""
### Download the results
Packs `outputs/fretboard/` (all JSON files, overlay pictures, `summary.json`) into one zip and downloads it. Keep it: the next notebook will load these JSON files.
""")
code(r"""
# colab-only
import shutil
from google.colab import files
zip_out = shutil.make_archive('/content/fretboard_results', 'zip', OUT_DIR)
print('Created', zip_out)
files.download(zip_out)
""")

md(r"""
## 10. Limitations, and when to move to a stronger method
Look at your failure gallery with these in mind:

| What goes wrong | Why classical CV fails | What would fix it |
|---|---|---|
| **Fingers covering the frets** (very common when playing chords!) | Fret edges are hidden, so fewer than 2 fret lines survive | A learned model that "imagines" the hidden structure; or use the *visible* frets plus the known geometric spacing of real frets |
| **Strings not all visible or too thin/low contrast** (dark strings on dark wood, heavy image augmentation) | The ridge map is weak, so a table edge or wood grain with strong contrast can outscore the real strings and the detector locks onto it. One hidden string is tolerated, several are not | Learned segmentation of the neck, or more tolerant fitting (RANSAC with a fret-spacing prior) |
| **Strong perspective** (camera looking along the neck) | Strings fan out. We allow a smooth fan (per-string slopes, section 6) but not the full perspective of looking almost straight down the neck, and fret lines are still searched in a straight band | Vanishing-point geometry, or a homography estimated from a keypoint model |
| **Busy backgrounds, wood grain, patterned shirts** | Lots of straight edges that are not strings or frets | Segmenting the guitar first (a segmentation / detection model) |
| **Close-ups (neck fills the frame, no headstock)** | The nut may simply not be visible; our nut guess is only a guess | Use the headstock/tuners as a landmark with a keypoint detector |
| **Glare, blur, low light** | Canny edges become broken or noisy | Better exposure; or learned features |
| **False detections that look valid** (a small patch of textured wall, a shirt pattern) | Any 6 roughly parallel lines with 2 crossing lines pass the geometry test, even if they are not a neck | The `confidence` field catches many of these (short neck, few frets), but not all. A learned neck detector, or a check that the region contains a guitar-like texture, is the real fix |
| **Dataset images with heavy augmentation** (rotation, color inversion, black borders) | Colors flipped, borders add false edges (we mask borders, but not everything) | Test on your own clean photos from the camera angle you will actually use |
| **String numbering and nut identity** | Geometry alone cannot say which string is high E or whether line 1 is the nut or fret 1 | Needs a model that sees the headstock, or a user setting such as "left/right handed, camera on the left" |

**Practical recommendation:** use the `success` / `status` flag downstream. When detection fails, skip that frame instead of guessing. In a webcam setting you can also detect on many frames and keep the most confident result, because the fretboard barely moves.

**When to use a more advanced approach:** if `ok` rates stay low on **your own** camera footage, the next step is a small learned model (for example segmentation of the fretboard or keypoints at the nut and frets), trained on a few hundred of your own labeled frames. That is deliberately outside this milestone.
""")

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_fretboard_detection.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
