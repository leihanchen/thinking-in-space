#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys
import zipfile
from glob import glob

import h5py
import numpy as np
import torch
from PIL import Image


def log(message):
    print(message, flush=True)


def warn(message):
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_image(path):
    with Image.open(path) as img:
        return np.array(img)


def read_pincam(path):
    with open(path, "r", encoding="utf-8") as handle:
        line = handle.read().strip()
    values = np.fromstring(line, sep=" ", dtype=np.float32)
    if values.shape[0] != 6:
        raise ValueError(f"expected 6 values in {path}")
    return values


def collect_files(folder, suffix):
    return sorted(
        path for path in glob(os.path.join(folder, f"*{suffix}")) if os.path.isfile(path)
    )


def find_frames_dir(root, scene):
    candidates = [
        os.path.join(root, scene, f"{scene}_frames"),
        os.path.join(root, f"{scene}_frames"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    for dirpath, _dirnames, _filenames in os.walk(root):
        if os.path.basename(dirpath) == f"{scene}_frames":
            return dirpath
    return None


def verify_images(unzipped_dir, packed_dir, label):
    ok = True
    files = collect_files(unzipped_dir, ".png")
    if not files:
        warn(f"{label}: no png files found in {unzipped_dir}")
        return False

    json_path = os.path.join(packed_dir, "images.json")
    h5_path = os.path.join(packed_dir, "images.h5")
    if not os.path.isfile(json_path) or not os.path.isfile(h5_path):
        warn(f"{label}: missing packed outputs in {packed_dir}")
        return False

    try:
        meta = load_json(json_path)
    except Exception as exc:
        warn(f"{label}: failed to read {json_path}: {exc}")
        return False

    mapping = meta.get("files", {})
    dataset_name = meta.get("dataset", "images")

    try:
        with h5py.File(h5_path, "r") as handle:
            if dataset_name not in handle:
                warn(f"{label}: dataset {dataset_name} missing in {h5_path}")
                return False
            dataset = handle[dataset_name]
            for path in files:
                name = os.path.basename(path)
                if name not in mapping:
                    warn(f"{label}: missing mapping for {name}")
                    ok = False
                    continue
                idx = mapping[name]
                if not isinstance(idx, int):
                    warn(f"{label}: non-integer index for {name}")
                    ok = False
                    continue
                if idx < 0 or idx >= dataset.shape[0]:
                    warn(f"{label}: index out of range for {name}")
                    ok = False
                    continue
                try:
                    arr = read_image(path)
                    stored = dataset[idx]
                except Exception as exc:
                    warn(f"{label}: failed to read {name} at index {idx}: {exc}")
                    ok = False
                    continue
                if arr.shape != stored.shape or arr.dtype != stored.dtype:
                    warn(f"{label}: shape/dtype mismatch for {name}")
                    ok = False
                    continue
                if not np.array_equal(arr, stored):
                    warn(f"{label}: data mismatch for {name}")
                    ok = False
    except Exception as exc:
        warn(f"{label}: failed to open {h5_path}: {exc}")
        return False

    return ok


def verify_intrinsics(unzipped_dir, packed_dir, label):
    ok = True
    files = collect_files(unzipped_dir, ".pincam")
    if not files:
        warn(f"{label}: no pincam files found in {unzipped_dir}")
        return False

    json_path = os.path.join(packed_dir, "intrinsics.json")
    pt_path = os.path.join(packed_dir, "intrinsics.pt")
    if not os.path.isfile(json_path) or not os.path.isfile(pt_path):
        warn(f"{label}: missing packed outputs in {packed_dir}")
        return False

    try:
        meta = load_json(json_path)
    except Exception as exc:
        warn(f"{label}: failed to read {json_path}: {exc}")
        return False

    mapping = meta.get("files", {})

    try:
        tensor = torch.load(pt_path, map_location="cpu")
    except Exception as exc:
        warn(f"{label}: failed to load {pt_path}: {exc}")
        return False

    if not torch.is_tensor(tensor):
        warn(f"{label}: intrinsics file did not contain a tensor")
        return False
    if tensor.ndim != 2 or tensor.shape[1] != 6:
        warn(f"{label}: intrinsics tensor shape mismatch")
        return False

    for path in files:
        name = os.path.basename(path)
        if name not in mapping:
            warn(f"{label}: missing mapping for {name}")
            ok = False
            continue
        idx = mapping[name]
        if not isinstance(idx, int):
            warn(f"{label}: non-integer index for {name}")
            ok = False
            continue
        if idx < 0 or idx >= tensor.shape[0]:
            warn(f"{label}: index out of range for {name}")
            ok = False
            continue
        try:
            values = read_pincam(path)
        except Exception as exc:
            warn(f"{label}: failed to read {name}: {exc}")
            ok = False
            continue
        stored = tensor[idx].cpu().numpy()
        if not np.allclose(values, stored, rtol=1e-6, atol=1e-6):
            warn(f"{label}: data mismatch for {name}")
            ok = False

    return ok


def safe_rmtree(path, expected_basename):
    if os.path.basename(path) != expected_basename:
        raise ValueError(f"refusing to remove unexpected path {path}")
    shutil.rmtree(path)


def unzip_zip(zip_file, dest_dir):
    with zipfile.ZipFile(zip_file, "r") as handle:
        handle.extractall(dest_dir)


def process_zip(zip_file, dataset_root, unzipped_dir, args):
    if not os.path.isfile(zip_file):
        warn(f"zip file not found: {zip_file}")
        return False

    scene = os.path.splitext(os.path.basename(zip_file))[0]
    zip_dir = os.path.dirname(zip_file)
    if not dataset_root:
        dataset_root = zip_dir
    if not unzipped_dir:
        unzipped_dir = os.path.join(zip_dir, f"{scene}_zip")

    created = False
    existed = os.path.isdir(unzipped_dir)

    if args.force_unzip and existed and not args.skip_unzip:
        try:
            safe_rmtree(unzipped_dir, f"{scene}_zip")
        except ValueError as exc:
            warn(f"{scene}: {exc}")
            return False
        existed = False

    try:
        if args.skip_unzip:
            if not os.path.isdir(unzipped_dir):
                warn(f"{scene}: unzipped dir missing at {unzipped_dir}")
                return False
        else:
            if not existed:
                os.makedirs(unzipped_dir, exist_ok=True)
                created = True
                try:
                    unzip_zip(zip_file, unzipped_dir)
                except Exception as exc:
                    warn(f"{scene}: unzip failed: {exc}")
                    return False

        frames_dir = find_frames_dir(unzipped_dir, scene)
        if not frames_dir:
            warn(f"{scene}: could not find {scene}_frames in {unzipped_dir}")
            return False

        packed_frames_dir = os.path.join(dataset_root, scene, f"{scene}_frames")
        if not os.path.isdir(packed_frames_dir):
            warn(f"{scene}: packed frames dir missing at {packed_frames_dir}")
            return False

        ok = True
        if "depth" in args.modalities:
            unzipped_depth = os.path.join(frames_dir, "lowres_depth")
            packed_depth = os.path.join(packed_frames_dir, "lowres_depth")
            if not os.path.isdir(unzipped_depth):
                warn(f"{scene} depth: missing unzipped folder at {unzipped_depth}")
                ok = False
            elif not os.path.isdir(packed_depth):
                warn(f"{scene} depth: missing packed folder at {packed_depth}")
                ok = False
            else:
                ok = verify_images(unzipped_depth, packed_depth, f"{scene} depth") and ok

        if "wide" in args.modalities:
            unzipped_wide = os.path.join(frames_dir, "lowres_wide")
            packed_wide = os.path.join(packed_frames_dir, "lowres_wide")
            if not os.path.isdir(unzipped_wide):
                warn(f"{scene} wide: missing unzipped folder at {unzipped_wide}")
                ok = False
            elif not os.path.isdir(packed_wide):
                warn(f"{scene} wide: missing packed folder at {packed_wide}")
                ok = False
            else:
                ok = verify_images(unzipped_wide, packed_wide, f"{scene} wide") and ok

        if "intrinsics" in args.modalities:
            unzipped_intr = os.path.join(frames_dir, "lowres_wide_intrinsics")
            packed_intr = os.path.join(packed_frames_dir, "lowres_wide_intrinsics")
            if not os.path.isdir(unzipped_intr):
                warn(f"{scene} intrinsics: missing unzipped folder at {unzipped_intr}")
                ok = False
            elif not os.path.isdir(packed_intr):
                warn(f"{scene} intrinsics: missing packed folder at {packed_intr}")
                ok = False
            else:
                ok = (
                    verify_intrinsics(unzipped_intr, packed_intr, f"{scene} intrinsics")
                    and ok
                )

        if ok:
            log(f"{scene}: OK")

        return ok
    finally:
        if created and not args.keep_unzipped:
            shutil.rmtree(unzipped_dir, ignore_errors=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Verify packed ARKitScenes files against zipped originals."
    )
    parser.add_argument("--zip-file", required=True, help="Path to X.zip.")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Root containing packed scene folders (default: zip directory).",
    )
    parser.add_argument(
        "--unzipped-dir",
        default=None,
        help="Where to unzip (default: <zipdir>/<scene>_zip).",
    )
    parser.add_argument(
        "--modalities",
        nargs="*",
        choices=["depth", "wide", "intrinsics"],
        default=["depth", "wide", "intrinsics"],
        help="Which modalities to verify.",
    )
    parser.add_argument(
        "--skip-unzip",
        action="store_true",
        help="Do not unzip; verify existing unzipped folder.",
    )
    parser.add_argument(
        "--force-unzip",
        action="store_true",
        help="Delete existing <scene>_zip before unzipping.",
    )
    parser.add_argument(
        "--keep-unzipped",
        action="store_true",
        help="Keep the unzipped folder after verification.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    ok = process_zip(args.zip_file, args.dataset_root, args.unzipped_dir, args)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
