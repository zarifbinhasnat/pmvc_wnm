"""
download_data.py — fetch the b-and-b-80k Banglish emotion dataset from Kaggle.

Requires Kaggle auth once (either ~/.kaggle/kaggle.json, or the
KAGGLE_USERNAME / KAGGLE_KEY environment variables). Get the token from
kaggle.com -> Settings -> "Create New Token".

Usage:
    pip install kagglehub
    python download_data.py

Prints the folder kagglehub downloaded into and every CSV it contains, so
you can pass one straight to run_local.py.
"""
import glob
import os
import sys


def download(slug="deepz99/b-and-b-80k"):
    import kagglehub
    path = kagglehub.dataset_download(slug)
    print("Path to dataset files:", path)

    csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csvs:
        print("No CSV found in the download — check the folder above.")
        return path, csvs

    print("\nCSV files found:")
    for c in csvs:
        print(f"  {c}")
    print(f"\nRun the model with:\n  python run_local.py \"{csvs[0]}\"")
    return path, csvs


if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "deepz99/b-and-b-80k"
    download(slug)
