#!/usr/bin/env python3
"""Pre-download VSI-Bench into Hugging Face caches.

Run this on a machine with network access before offline evaluation jobs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import DownloadMode, load_dataset
from huggingface_hub import snapshot_download


def resolve_paths() -> dict[str, Path]:
    hf_home = Path(os.environ.get("HF_HOME", "/home/leihan/links/scratch/thinking-in-space/.cache/huggingface")).expanduser()
    hub_cache = Path(
        os.environ.get("HUGGINGFACE_HUB_CACHE", os.environ.get("HF_HUB_CACHE", str(hf_home / "hub")))
    ).expanduser()
    datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE", str(hf_home / "datasets"))).expanduser()
    modules_cache = Path(os.environ.get("HF_MODULES_CACHE", str(hf_home / "modules"))).expanduser()
    return {
        "HF_HOME": hf_home,
        "HUGGINGFACE_HUB_CACHE": hub_cache,
        "HF_DATASETS_CACHE": datasets_cache,
        "HF_MODULES_CACHE": modules_cache,
    }


def ensure_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def predownload(dataset_id: str, split: str, token: str | None, force_redownload: bool) -> None:
    paths = resolve_paths()
    ensure_dirs(paths)

    # Keep all HF clients aligned on the same cache roots.
    os.environ["HF_HOME"] = str(paths["HF_HOME"])
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(paths["HUGGINGFACE_HUB_CACHE"])
    os.environ["HF_HUB_CACHE"] = str(paths["HUGGINGFACE_HUB_CACHE"])
    os.environ["HF_DATASETS_CACHE"] = str(paths["HF_DATASETS_CACHE"])
    os.environ["HF_MODULES_CACHE"] = str(paths["HF_MODULES_CACHE"])

    print("Resolved cache paths:")
    for key, value in paths.items():
        print(f"  {key}={value}")

    print("\nStep 1/2: Download dataset repository snapshot to Hub cache...")
    snapshot_dir = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        cache_dir=str(paths["HUGGINGFACE_HUB_CACHE"]),
        token=token,
        force_download=force_redownload,
    )
    print(f"  Snapshot: {snapshot_dir}")

    print("\nStep 2/2: Materialize dataset with datasets.load_dataset...")
    ds = load_dataset(
        path=dataset_id,
        split=split,
        cache_dir=str(paths["HF_DATASETS_CACHE"]),
        token=token if token else False,
        download_mode=DownloadMode.FORCE_REDOWNLOAD if force_redownload else DownloadMode.REUSE_CACHE_IF_EXISTS,
    )
    ds.save_to_disk(paths["HF_DATASETS_CACHE"] / dataset_id.replace("/", "_"))
    print(f"  Loaded split '{split}' with {len(ds)} rows")

    print("\nPre-download complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-download VSI-Bench into HF caches")
    parser.add_argument("--dataset-id", default="nyu-visionx/VSI-Bench", help="Hugging Face dataset id")
    parser.add_argument("--split", default="test", help="Split to materialize")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Optional HF token")
    parser.add_argument("--force-redownload", action="store_true", help="Force cache refresh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predownload(
        dataset_id=args.dataset_id,
        split=args.split,
        token=args.token,
        force_redownload=args.force_redownload,
    )


if __name__ == "__main__":
    main()
