Our dataset is in the folder `$DATASET = "/scratch/indrisch/ARKitScenes_data/3dod/Training/"`.

Each maxdepth 1 subdir of `$DATASET` contains a folder `$SCENE`, where `$SCENE` is named according to an 8-digit numerical string, such as 41126964.

Each `$SCENE` has a subdir named `${SCENE}_frames`. 

Each `${SCENE}_frames` contains three subdirs: `lowres_depth`, `lowres_wide`, `lowres_wide_intrinsics`:
* `lowres_depth` contains a bunch of image files, named `${SCENE}_X.Y.png`, where `X` and `Y` are numerical strings.
* `lowres_wide` contains a bunch of image files, named `${SCENE}_X.Y.png`, where `X` and `Y` are numerical strings.
* `lowres_wide_intrinsics` contains a bunch of text files, named `${SCENE}_X.Y.pincam`, where `X` and `Y` are numerical strings. `${SCENE}_X.Y.pincam` contains a single line with 6 numbers (first two are integers, the rest are floating-point) separated by a space. 

Your task is to reduce the number of files on this filesystem by the following:
* The folder `lowres_depth` should contain all of the image files packed into an HDF5 file, and may have a JSON for mapping as needed. Delete the image files when done.
* The folder `lowres_wide` should contain all of the image files packed into an HDF5 file, and may have a JSON for mapping as needed. Delete the image files when done.
* The folder `lowres_wide_intrinsics` should contain all of the pincam files packed into a pytorch tensor, and may have a JSON for mapping as needed. Delete the pincam files when done.


You may create any bash or python files in folder "/scratch/indrisch/thinking-in-space/other_scripts/h5py" to make this happen. Note that you may need to create a virtual environment.

One way this could be implemented is to have a JSON file mapping the filename to where it is in the h5py; for example, the key could be a filename and the value could be something like the appropriate slice in the h5py/tensor to get it out.

## Implementation

Script:
* other_scripts/h5py/pack_arkitscenes.py

Outputs (per scene):
* lowres_depth/images.h5 + lowres_depth/images.json
* lowres_wide/images.h5 + lowres_wide/images.json
* lowres_wide_intrinsics/intrinsics.pt + lowres_wide_intrinsics/intrinsics.json

Notes:
* Outputs are skipped if they already exist unless --force is used.
* Verification spot-checks a few indices before deletion.
* Images are expected to have a uniform size within a scene.

Usage examples:
```bash
python other_scripts/h5py/pack_arkitscenes.py --scenes 41126964 --verify
python other_scripts/h5py/pack_arkitscenes.py --scenes 41126964 --verify --delete
python other_scripts/h5py/pack_arkitscenes.py --modalities intrinsics --scenes 41126964
```