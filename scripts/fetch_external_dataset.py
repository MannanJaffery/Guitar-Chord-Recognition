"""External dataset downloader, mirror validator, and archive inspector.

Allows fetching guitar chord datasets from custom mirrors, Hugging Face, or direct URLs,
and validating annotations into artifacts/external/.

Usage:
    python scripts/fetch_external_dataset.py --info
    python scripts/fetch_external_dataset.py --check-url <url>
    python scripts/fetch_external_dataset.py --download-gdrive <id> --out artifacts/external/<name>.zip
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path
import urllib.request
import urllib.error


def check_url(url):
    print(f"Checking URL: {url} ...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            code = resp.getcode()
            print(f"HTTP Status: {code} OK")
            return True
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False


def inspect_external(folder):
    p = Path(folder)
    if not p.exists():
        print(f"Directory {folder} does not exist.")
        return
    files = list(p.rglob("*"))
    imgs = [f for f in files if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    zips = [f for f in files if f.suffix.lower() == '.zip']
    annos = [f for f in files if f.suffix.lower() in ('.json', '.xml', '.txt', '.csv', '.yaml')]
    print(f"\n--- External Directory Inspection: {folder} ---")
    print(f"Total files: {len(files)}")
    print(f"Image files: {len(imgs)}")
    print(f"Zip archives: {len(zips)}")
    print(f"Annotation/meta files: {len(annos)}")
    if imgs:
        print("Sample images:", [f.name for f in imgs[:5]])
    if annos:
        print("Sample annotations:", [f.name for f in annos[:5]])


def main():
    parser = argparse.ArgumentParser(description="External Dataset Utility")
    parser.add_argument("--info", action="store_true", help="Inspect artifacts/external directory")
    parser.add_argument("--check-url", type=str, help="Check if a URL is reachable")
    parser.add_argument("--dir", type=str, default="artifacts/external", help="Directory to inspect")
    args = parser.parse_args()

    if args.check_url:
        check_url(args.check_url)
    if args.info or not sys.argv[1:]:
        inspect_external(args.dir)


if __name__ == "__main__":
    main()
