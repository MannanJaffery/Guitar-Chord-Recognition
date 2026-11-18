"""Build a small zip of dataset images to upload to Colab (instead of Google Drive).

Needs data/images/ first:  python scripts/export_images.py
Usage:  python scripts/make_sample_zip.py [--per-class 20] [--max-side 800] [--out data/guitar_chords_sample.zip]

Zip layout:  <Chord>/<split>_<number>.jpg   (e.g. Am/train_00123.jpg)
"""
import argparse
import random
import sys
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--max-side", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "guitar_chords_sample.zip"))
    args = ap.parse_args()

    src = ROOT / "data" / "images"
    if not src.exists():
        sys.exit("data/images not found. Run: python scripts/export_images.py")

    rng = random.Random(args.seed)
    chords = sorted({p.name for split in src.iterdir() for p in split.iterdir()})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:       # jpgs are already compressed
        for chord in chords:
            files = sorted(p for split in src.iterdir() for p in (split / chord).glob("*.jpg"))
            for p in rng.sample(files, min(args.per_class, len(files))):
                im = Image.open(p).convert("RGB")
                im.thumbnail((args.max_side, args.max_side))        # keeps aspect ratio, only shrinks
                buf = BytesIO()
                im.save(buf, "JPEG", quality=90)
                z.writestr(f"{chord}/{p.parent.parent.name}_{p.stem}.jpg", buf.getvalue())
                n += 1
    print(f"{n} images from {len(chords)} chords -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
