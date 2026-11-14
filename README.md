# Guitar Chord Recognition

Recognize guitar chords from video. **Stage 1 (this repo, current): visual chord classifier** from the fretting hand and fretboard. Stage 2 (planned): audio model and fusion.

## Dataset
[j-asra-j/guitar-chords](https://huggingface.co/datasets/j-asra-j/guitar-chords): 3,151 images, 14 classes (A, Am, B, Bm, C, Cm, D, Dm, E, Em, F, Fm, G, Gm). It is **classification-only** (image + label, no boxes). License is not stated upstream; check before any redistribution.

Known issues (see `outputs/dataset_report/` after running the inspection):
- **Cm, Fm, Gm have no validation/test images** in the official split. The notebook carves a slice from train.
- Many images are augmented near-copies of the same photo, so validation accuracy is optimistic.
- Cm/Fm/Gm appear to come from a single video each (same person and background), so the model may learn the background. **Collect your own photos** for a real test.
- Class imbalance: 100 to 551 train images per class (handled with a weighted loss).
- The upstream repo has filenames containing `:` that Windows cannot store, so we load the Hugging Face Parquet branch (`src/gcr/data.py`).

## Layout
```
configs/dataset.yaml        dataset settings
src/gcr/data.py             dataset loading
scripts/inspect_dataset.py  download + quality report -> outputs/dataset_report/
scripts/make_notebook.py    regenerates notebooks/train_colab.ipynb
notebooks/train_colab.ipynb training + evaluation (run on Colab)
data/, outputs/             git-ignored
```

## Setup (local)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/inspect_dataset.py
```

## Train (Colab)
1. Upload `notebooks/train_colab.ipynb` to Google Colab (File > Upload notebook), or open it from GitHub once pushed.
2. Runtime > Change runtime type > **T4 GPU**.
3. Run all cells. The best checkpoint is saved to Drive at `MyDrive/guitar-chord-recognition/best.pt`.
4. (Recommended) Add your own test photos to `MyDrive/guitar-chord-recognition/own_test/<Chord>/` and run the last cell.

## Model
EfficientNet-B0 (ImageNet pretrained), 224x224 input, AdamW with OneCycle schedule, label smoothing 0.1, class-weighted cross-entropy, mild augmentation, no horizontal flip.

## Roadmap
- [x] Dataset inspection and report
- [x] Colab training notebook (classifier)
- [ ] Train, then evaluate on own photos
- [ ] Hand/fretboard cropping (MediaPipe) to cut background reliance
- [ ] Real-time webcam inference with temporal smoothing
- [ ] Audio chord model and fusion
