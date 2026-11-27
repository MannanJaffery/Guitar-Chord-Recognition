"""Unpack guifir_eval_and_models.zip from Colab, print the evaluation report,
back up existing artifacts/models_local/, and install the new models.

Usage:
    python scripts/install_guifir_models.py --zip artifacts/external/guifir_eval_and_models.zip
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "artifacts" / "models_local"
BACKUP_DIR = ROOT / "artifacts" / "models_local_prev_guifir"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=str(ROOT / "artifacts" / "external" / "guifir_eval_and_models.zip"))
    args = ap.parse_args()

    zpath = Path(args.zip)
    if not zpath.exists():
        raise SystemExit(f"Zip not found: {zpath}\nPlace guifir_eval_and_models.zip there or pass --zip <path>.")

    if MODELS_DIR.exists() and not BACKUP_DIR.exists():
        shutil.copytree(MODELS_DIR, BACKUP_DIR)
        print(f"Backed up previous models to {BACKUP_DIR}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(MODELS_DIR)
        print("Extracted:", zf.namelist())

    # If neck_v3.pt is present, also copy/link as neck_v2.pt if live_local.py expects it
    v3 = MODELS_DIR / "neck_v3.pt"
    if v3.exists():
        shutil.copy2(v3, MODELS_DIR / "neck_v2.pt")
        print("Updated neck_v2.pt from neck_v3.pt")

    eval_path = MODELS_DIR / "eval_report.json"
    if eval_path.exists():
        print("\n--- GuiFiR Colab Evaluation Report ---")
        print(json.dumps(json.loads(eval_path.read_text()), indent=2))


if __name__ == "__main__":
    main()
