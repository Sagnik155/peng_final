# Training — Fragment 2-class semantic segmentation (Baseline Phase 2: Fragments)

Dataset: `Dataset457_PENGWIN_frag`
Labels: `0=background, 1=fragment`

---

## Prerequisites

- `python preprocessing/fragments/create_fragment_nnUNet_dataset.py` has been run, producing
  ```
  [YOUR_OUTPUT_PATH_HERE]
  ├── imagesTr/      case_<pid>_frag_<anat>_<frag_id>_0000.nii.gz  (CT image)
  |                  case_<pid>_frag_<anat>_<frag_id>_0001.nii.gz  (fragment click)
  |                  case_<pid>_frag_<anat>_<frag_id>_0002.nii.gz  (background click)
  |
  ├── labelsTr/      case_<pid>_frag_<anat>_<frag_id>.nii.gz       (values 0..1)
  ```
  Here, `<anat>` is either `Left`, `Right`, `Femur`, or `Sacrum` and refers to which anatomy the fragment in the foreground label `1` belongs to.


- The conda env created by `install.sh` is activated:
  ```bash
  conda activate pengwin_nnunet
  ```
- The three nnUNetv2 path variables are exported. Adjust the paths to your machine:
  ```bash
  export nnUNet_raw=/path/to/nnUNet_raw
  export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
  export nnUNet_results=/path/to/nnUNet_results
  ```

---

## Step 1 — Copy all needed dataset files into `nnUNet_raw`

```bash
mkdir "${nnUNet_raw}/Dataset457_PENGWIN_frag"
cp -r [YOUR_OUTPUT_PATH]/ "${nnUNet_raw}/Dataset457_PENGWIN_frag"
cp ./dataset.json "${nnUNet_raw}/Dataset457_PENGWIN_frag"
```

nnUNetv2 identifies datasets by the integer ID embedded in the folder name (`Dataset<ID>_<Name>`); here we use `457`. A direct copy is recommended so that any internal changes made by nnUNet do not affect the original files under `[YOUR_OUTPUT_PATH]/`.

## Step 2 — Plan & preprocess
Remove the `--verify_dataset_integrity` if you want to save time and are sure your files are not corrupted.

```bash
nnUNetv2_plan_and_preprocess -d 457 --verify_dataset_integrity -c 3d_fullres
```

This writes `nnUNetPlans.json` and preprocessed tensors for the `3d_fullres` configuration into `${nnUNet_preprocessed}/Dataset457_PENGWIN_frag/`.

## Step 3 — Train

Single fold (fold 0) - what we provide with our baseline:

```bash
nnUNetv2_train 457 3d_fullres 0 --npz
```

Alternatively, full 5-fold cross-validation:

```bash
for f in 0 1 2 3 4; do
    nnUNetv2_train 457 3d_fullres ${f} --npz
done
```

Other common variants:

| Goal | Command |
|------|---------|
| Lower memory footprint | `nnUNetv2_train 457 3d_lowres 0 --npz` |
| 2D model               | `nnUNetv2_train 457 2d 0 --npz` |
| Shorter (250 epochs)   | `nnUNetv2_train 457 3d_fullres 0 -tr nnUNetTrainer_250epochs --npz` |
| Resume training        | append `--c` |

Checkpoints live under `${nnUNet_results}/Dataset457_PENGWIN_frag/`.

## Step 4 — Inference (example with model trained on fold 0)

```bash
nnUNetv2_predict \
    -i <input_dir> \
    -o <output_dir> \
    -d 457 \
    -c 3d_fullres \
    -f 0 
```

