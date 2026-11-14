"""Export every dataset image to data/images/<split>/<chord>/<n>.jpg so you can browse them.

Usage: python scripts/export_images.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from gcr.data import ROOT, load_config, load_guitar_chords  # noqa: E402


def main() -> None:
    cfg = load_config()
    ds = load_guitar_chords(cfg["hf_repo"], cache_dir=ROOT / cfg["raw_dir"])
    names = ds["train"].features["label"].names
    out = ROOT / "data" / "images"
    for split, d in ds.items():
        for i, row in enumerate(tqdm(d, desc=split)):
            folder = out / split / names[row["label"]]
            folder.mkdir(parents=True, exist_ok=True)
            row["image"].convert("RGB").save(folder / f"{i:05d}.jpg", quality=92)
    print("Images in", out)


if __name__ == "__main__":
    main()
