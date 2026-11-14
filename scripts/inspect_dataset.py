"""Download the dataset and produce a quality report.

Usage:  python scripts/inspect_dataset.py
Output: outputs/dataset_report/{summary.json, class_counts.png, samples_<split>.png, duplicates.txt}
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gcr.data import ROOT, load_config, load_guitar_chords  # noqa: E402


def main() -> None:
    cfg = load_config()
    out = ROOT / cfg["report_dir"]
    out.mkdir(parents=True, exist_ok=True)
    ds = load_guitar_chords(cfg["hf_repo"], cache_dir=ROOT / cfg["raw_dir"])
    names = ds["train"].features["label"].names
    rng = random.Random(cfg["seed"])

    summary: dict = {"classes": names, "splits": {}}
    hashes: dict[str, list[tuple[str, int]]] = {}

    for split, d in ds.items():
        counts = Counter(d["label"])
        sizes = Counter()
        for i, row in enumerate(d):
            img = row["image"]
            sizes[f"{img.width}x{img.height}"] += 1
            h = hashlib.md5(img.tobytes()).hexdigest()
            hashes.setdefault(h, []).append((split, i))
        summary["splits"][split] = {
            "n": len(d),
            "per_class": {names[k]: counts.get(k, 0) for k in range(len(names))},
            "top_sizes": sizes.most_common(5),
        }
        print(f"{split}: {len(d)} images, sizes {sizes.most_common(3)}")

    # Exact duplicates, especially across splits (data leakage)
    dups = [v for v in hashes.values() if len(v) > 1]
    leaks = [v for v in dups if len({s for s, _ in v}) > 1]
    summary["exact_duplicate_groups"] = len(dups)
    summary["cross_split_leaks"] = len(leaks)
    (out / "duplicates.txt").write_text("\n".join(map(str, dups)), encoding="utf-8")
    print(f"exact duplicate groups: {len(dups)} (cross-split leaks: {len(leaks)})")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Class balance plot
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.27
    for j, split in enumerate(ds):
        ax.bar([k + j * width for k in range(len(names))],
               [summary["splits"][split]["per_class"][n] for n in names], width, label=split)
    ax.set_xticks([k + width for k in range(len(names))], names)
    ax.set_ylabel("images")
    ax.set_title("Images per chord class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_counts.png", dpi=120)
    plt.close(fig)

    # Sample grid: n examples per class from train
    n = cfg["preview_samples_per_class"]
    train = ds["train"]
    by_class: dict[int, list[int]] = {k: [] for k in range(len(names))}
    for i, lab in enumerate(train["label"]):
        by_class[lab].append(i)
    fig, axes = plt.subplots(len(names), n, figsize=(n * 2.2, len(names) * 2.2))
    for k in range(len(names)):
        picks = rng.sample(by_class[k], min(n, len(by_class[k])))
        for c in range(n):
            ax = axes[k][c]
            ax.axis("off")
            if c < len(picks):
                ax.imshow(train[picks[c]]["image"])
            if c == 0:
                ax.set_title(names[k], loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "samples_train.png", dpi=70)
    plt.close(fig)
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
