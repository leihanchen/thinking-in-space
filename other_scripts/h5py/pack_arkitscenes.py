#!/usr/bin/env python3

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import h5py
import numpy as np
import torch
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


DEFAULT_DATASET = "/scratch/indrisch/ARKitScenes_data/3dod/Validation/"


def log(message):
    print(message, flush=True)


def list_scenes(dataset_root, scenes):
    if scenes:
        return scenes
    results = []
    for name in sorted(os.listdir(dataset_root)):
        path = os.path.join(dataset_root, name)
        if os.path.isdir(path) and name.isdigit() and len(name) == 8:
            results.append(name)
    return results


def collect_files(folder, suffix):
    return sorted(
        path for path in glob(os.path.join(folder, f"*{suffix}")) if os.path.isfile(path)
    )


def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return len(collect_files(folder, ".png"))


def read_image(path):
    with Image.open(path) as img:
        return np.array(img)


def image_dimensions(path):
    with Image.open(path) as img:
        width, height = img.size
        bands = img.getbands()
    if len(bands) == 1:
        return (height, width)
    return (height, width, len(bands))


def write_json_atomic(data, path):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def sample_indices(count):
    if count <= 0:
        return []
    indices = {0}
    if count > 1:
        indices.add(count - 1)
    if count > 2:
        indices.add(count // 2)
    return sorted(indices)


def check_image_dimensions(files, expected_shape, progress):
    for path in progress(files, desc="    checking", unit="img"):
        actual = image_dimensions(path)
        if actual != expected_shape:
            log(
                f"  - dimension mismatch for {path}: expected {expected_shape}, got {actual}"
            )
            return False
    return True


def verify_h5(files, h5_path, dataset_name):
    try:
        with h5py.File(h5_path, "r") as handle:
            if dataset_name not in handle:
                log(f"  - missing dataset {dataset_name} in {h5_path}")
                return False
            dataset = handle[dataset_name]
            if dataset.shape[0] != len(files):
                log("  - count mismatch")
                return False
            for idx in sample_indices(len(files)):
                arr = read_image(files[idx])
                stored = dataset[idx]
                if arr.shape != stored.shape or arr.dtype != stored.dtype:
                    log(f"  - shape/dtype mismatch at index {idx}")
                    return False
                if not np.array_equal(arr, stored):
                    log(f"  - data mismatch at index {idx}")
                    return False
    except Exception as exc:
        log(f"  - verify error: {exc}")
        return False
    return True


def verify_intrinsics(files, pt_path):
    try:
        tensor = torch.load(pt_path, map_location="cpu")
        if not torch.is_tensor(tensor):
            log("  - intrinsics file did not contain a tensor")
            return False
        if tensor.shape[0] != len(files) or tensor.shape[1] != 6:
            log("  - intrinsics shape mismatch")
            return False
        for idx in sample_indices(len(files)):
            with open(files[idx], "r", encoding="utf-8") as handle:
                line = handle.read().strip()
            vals = np.fromstring(line, sep=" ", dtype=np.float32)
            if vals.shape[0] != 6:
                return False
            if not np.allclose(vals, tensor[idx].cpu().numpy(), rtol=1e-6, atol=1e-6):
                log(f"  - intrinsics mismatch at index {idx}")
                return False
    except Exception as exc:
        log(f"  - verify error: {exc}")
        return False
    return True


def delete_files(files, progress, dry_run):
    if dry_run:
        log("  - dry-run: would delete source files")
        return
    for path in progress(files, desc="    deleting", unit="file"):
        os.remove(path)


def pack_images(folder, dataset_name, delete, verify, force, dry_run, progress):
    files = collect_files(folder, ".png")
    if not files:
        log(f"  - no png files in {folder}")
        return True

    h5_path = os.path.join(folder, "images.h5")
    json_path = os.path.join(folder, "images.json")

    if not force and (os.path.exists(h5_path) or os.path.exists(json_path)):
        log(f"  - output exists, skipping {folder}")
        return True

    if dry_run:
        log(f"  - dry-run: would pack {len(files)} images in {folder}")
        return True

    tmp_h5 = f"{h5_path}.tmp"
    remove_if_exists(tmp_h5)

    first = read_image(files[0])
    shape = first.shape
    dtype = first.dtype
    mapping = {os.path.basename(files[0]): 0}

    if not check_image_dimensions(files, shape, progress):
        return False

    try:
        with h5py.File(tmp_h5, "w") as handle:
            chunk_shape = (1,) + shape
            dataset = handle.create_dataset(
                dataset_name,
                shape=(len(files),) + shape,
                dtype=dtype,
                chunks=chunk_shape,
            )
            dataset[0] = first
            for idx, path in enumerate(
                progress(files[1:], desc="    writing", unit="img"), start=1
            ):
                arr = read_image(path)
                if arr.shape != shape or arr.dtype != dtype:
                    raise ValueError(f"shape/dtype mismatch for {path}")
                dataset[idx] = arr
                mapping[os.path.basename(path)] = idx
    except Exception:
        remove_if_exists(tmp_h5)
        raise

    os.replace(tmp_h5, h5_path)

    meta = {
        "dataset": dataset_name,
        "dtype": str(dtype),
        "shape": list(shape),
        "files": mapping,
    }
    write_json_atomic(meta, json_path)

    if verify and not verify_h5(files, h5_path, dataset_name):
        log("  - verification failed")
        return False

    if delete:
        delete_files(files, progress, dry_run=False)

    return True


def pack_intrinsics(folder, delete, verify, force, dry_run, progress):
    files = collect_files(folder, ".pincam")
    if not files:
        log(f"  - no pincam files in {folder}")
        return True

    pt_path = os.path.join(folder, "intrinsics.pt")
    json_path = os.path.join(folder, "intrinsics.json")

    if not force and (os.path.exists(pt_path) or os.path.exists(json_path)):
        log(f"  - output exists, skipping {folder}")
        return True

    if dry_run:
        log(f"  - dry-run: would pack {len(files)} intrinsics in {folder}")
        return True

    data = np.zeros((len(files), 6), dtype=np.float32)
    mapping = {}

    for idx, path in enumerate(progress(files, desc="    reading", unit="file")):
        with open(path, "r", encoding="utf-8") as handle:
            line = handle.read().strip()
        vals = np.fromstring(line, sep=" ", dtype=np.float32)
        if vals.shape[0] != 6:
            raise ValueError(f"expected 6 values in {path}")
        data[idx] = vals
        mapping[os.path.basename(path)] = idx

    tensor = torch.from_numpy(data)
    tmp_pt = f"{pt_path}.tmp"
    remove_if_exists(tmp_pt)
    torch.save(tensor, tmp_pt)
    os.replace(tmp_pt, pt_path)

    meta = {
        "dtype": "float32",
        "shape": [len(files), 6],
        "files": mapping,
    }
    write_json_atomic(meta, json_path)

    if verify and not verify_intrinsics(files, pt_path):
        log("  - verification failed")
        return False

    if delete:
        delete_files(files, progress, dry_run=False)

    return True


def process_scene(scene, dataset_root, modalities, args, progress):
    scene_dir = os.path.join(dataset_root, scene)
    frame_dir = os.path.join(scene_dir, f"{scene}_frames")
    if not os.path.isdir(frame_dir):
        log(f"Skipping {scene}: missing frames dir")
        return True

    depth_dir = os.path.join(frame_dir, "lowres_depth")
    wide_dir = os.path.join(frame_dir, "lowres_wide")
    if count_images(depth_dir) + count_images(wide_dir) == 0:
        log(f"Skipping {scene}: no images found")
        return True

    ok = True

    if "depth" in modalities:
        if os.path.isdir(depth_dir):
            log(f"{scene} depth")
            ok = pack_images(
                depth_dir,
                "images",
                args.delete,
                args.verify,
                args.force,
                args.dry_run,
                progress,
            ) and ok
        else:
            log(f"{scene} depth missing folder")

    if "wide" in modalities:
        if os.path.isdir(wide_dir):
            log(f"{scene} wide")
            ok = pack_images(
                wide_dir,
                "images",
                args.delete,
                args.verify,
                args.force,
                args.dry_run,
                progress,
            ) and ok
        else:
            log(f"{scene} wide missing folder")

    if "intrinsics" in modalities:
        intr_dir = os.path.join(frame_dir, "lowres_wide_intrinsics")
        if os.path.isdir(intr_dir):
            log(f"{scene} intrinsics")
            ok = pack_intrinsics(
                intr_dir,
                args.delete,
                args.verify,
                args.force,
                args.dry_run,
                progress,
            ) and ok
        else:
            log(f"{scene} intrinsics missing folder")

    return ok


def process_scene_parallel(scene, dataset_root, modalities, args):
    progress = lambda x, **_kwargs: x
    try:
        ok = process_scene(scene, dataset_root, modalities, args, progress)
        return scene, ok, None
    except Exception as exc:
        return scene, False, str(exc)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pack ARKitScenes lowres images and intrinsics into HDF5/PT."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset root containing scene folders.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        help="Scene IDs to process (default: all scenes).",
    )
    parser.add_argument(
        "--modalities",
        nargs="*",
        choices=["depth", "wide", "intrinsics"],
        default=["depth", "wide", "intrinsics"],
        help="Which modalities to pack.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without writing outputs.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify outputs with a small spot-check.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete source files after successful verification.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue processing other scenes on errors.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes to use (default: CPU count).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.delete:
        args.verify = True

    if args.dry_run and args.delete:
        log("dry-run: delete disabled")
        args.delete = False

    use_progress = not args.no_progress

    if not os.path.isdir(args.dataset):
        log(f"Dataset root not found: {args.dataset}")
        return 1

    scenes = list_scenes(args.dataset, args.scenes)
    if not scenes:
        log("No scenes found")
        return 1

    if args.workers < 1:
        log("Workers must be >= 1")
        return 1

    workers = min(args.workers, len(scenes))
    if workers > 1 and use_progress:
        log("Disabling progress bars for parallel execution")
        use_progress = False

    progress = tqdm if use_progress else (lambda x, **_kwargs: x)

    overall_ok = True
    if workers == 1:
        for scene in scenes:
            try:
                ok = process_scene(scene, args.dataset, args.modalities, args, progress)
                overall_ok = overall_ok and ok
                if not ok and not args.keep_going:
                    return 1
            except Exception as exc:
                log(f"Error in scene {scene}: {exc}")
                overall_ok = False
                if not args.keep_going:
                    return 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_scene_parallel, scene, args.dataset, args.modalities, args
                ): scene
                for scene in scenes
            }
            for future in as_completed(futures):
                scene = futures[future]
                try:
                    scene_name, ok, error = future.result()
                except Exception as exc:
                    log(f"Error in scene {scene}: {exc}")
                    overall_ok = False
                    if not args.keep_going:
                        executor.shutdown(cancel_futures=True)
                        return 1
                    continue

                if error:
                    log(f"Error in scene {scene_name}: {error}")
                    overall_ok = False
                elif not ok:
                    overall_ok = False

                if not ok and not args.keep_going:
                    executor.shutdown(cancel_futures=True)
                    return 1

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
