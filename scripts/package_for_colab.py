"""Pack everything the Colab notebook needs into ONE zip: your chord videos + the neck labels (+ the warm-start neck model).

Usage:  python scripts/package_for_colab.py [--videos artifacts/own_videos] [--out artifacts/colab_upload/own_data.zip] [--no-model]

Zip layout (what the notebook expects):
    own_videos/<Chord>/take1.mp4, take2.mp4, ...
    neck_labels.json          frame name -> 4 neck corners (made with scripts/label_neck.py)
    neck_v1.pt                optional warm start for the neck finder (skip with --no-model)
Upload the zip to Google Drive (My Drive/guitar-chord-recognition/own_data.zip) or to the Colab file panel, then run the notebook.
"""
import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', default=str(ROOT / 'artifacts' / 'own_videos'))
    ap.add_argument('--labels', default=None, help='default: merge every <Chord>/neck_labels.json found under --videos')
    ap.add_argument('--model', default=str(ROOT / 'artifacts' / 'models_neck' / 'neck_v1.pt'))
    ap.add_argument('--no-model', action='store_true')
    ap.add_argument('--out', default=str(ROOT / 'artifacts' / 'colab_upload' / 'own_data.zip'))
    a = ap.parse_args()
    vids = sorted(Path(a.videos).glob('*/take*.mp4'))
    if not vids:
        raise SystemExit(f'no videos found under {a.videos}')
    labels = {}
    for f in ([Path(a.labels)] if a.labels else sorted(Path(a.videos).glob('*/neck_labels.json'))):
        labels.update(json.loads(f.read_text(encoding='utf-8')))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as z:           # mp4 is already compressed
        for v in vids:
            z.write(v, f'own_videos/{v.parent.name}/{v.name}')
        z.writestr('neck_labels.json', json.dumps(labels, indent=1))
        if not a.no_model and Path(a.model).exists():
            z.write(a.model, 'neck_v1.pt')
    chords = sorted({v.parent.name for v in vids})
    print(f'{len(vids)} videos, chords {chords}, {sum(1 for v in labels.values() if v)} labelled frames')
    print(f'wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)')
    (out.parent / 'README.txt').write_text(
        'Upload own_data.zip to Google Drive: My Drive/guitar-chord-recognition/own_data.zip (or to the Colab file panel).\n'
        'Then upload notebooks/05_chord_recognizer_colab.ipynb to Colab, set the runtime to GPU (T4) and run it from the top.\n', encoding='utf-8')


if __name__ == '__main__':
    main()
