# Training — Anatomical 5-class semantic segmentation (Baseline Phase 1: Anatomy)

Dataset: `Dataset456_PENGWIN`
Labels: `0=background, 1=sacrum, 2=leftHip, 3=rightHip, 4=femur`

---

## Prerequisites

- `python preprocessing/anatomy/convert_to_nnunet_format.py` has been run, producing
  ```
  [YOUR_OUTPUT_PATH_HERE]
  ├── imagesTr/      <pid>_0000.nii.gz
  ├── labelsTr/      <pid>.nii.gz       (values 0..200)
  ```

- `python preprocessing/anatomy/add_pengwin_heatmaps_anatomy.py` has been run, producing
  ```
  [YOUR_OUTPUT_PATH_HERE]
  ├── imagesTr/      <pid>_0001.nii.gz (sacrum clicks)
                     <pid>_0002.nii.gz (left hip clicks)
                     <pid>_0003.nii.gz (right hip clicks)
                     <pid>_0004.nii.gz (femur clicks)
  ```
- `python preprocessing/anatomy/remap_pengwin_labels.py` has been run, re-mapping all labels from 0-200 to 0-4 (just anatomy).
  ```
  [YOUR_OUTPUT_PATH_HERE]
  ├── labelsTr/      <pid>.nii.gz       (values 0..4)
  ```

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
mkdir "${nnUNet_raw}/Dataset456_PENGWIN"
cp -r [YOUR_OUTPUT_PATH]/ "${nnUNet_raw}/Dataset456_PENGWIN"
cp ./dataset.json "${nnUNet_raw}/Dataset456_PENGWIN"
```

nnUNetv2 identifies datasets by the integer ID embedded in the folder name (`Dataset<ID>_<Name>`); here we use `456`. A direct copy is recommended so that any internal changes made by nnUNet do not affect the original files under `[YOUR_OUTPUT_PATH]/`.

## Step 2 — Plan & preprocess
Remove the `--verify_dataset_integrity` if you want to save time and are sure your files are not corrupted.

```bash
nnUNetv2_plan_and_preprocess -d 456 --verify_dataset_integrity -c 3d_fullres
```

This writes `nnUNetPlans.json` and preprocessed tensors for the `3d_fullres` configuration into `${nnUNet_preprocessed}/Dataset456_PENGWIN/`.

## Step 3 — Train

Single fold (fold 0) - what we provide with our baseline:

```bash
nnUNetv2_train 456 3d_fullres 0 --npz
```

Alternatively, full 5-fold cross-validation:

```bash
for f in 0 1 2 3 4; do
    nnUNetv2_train 456 3d_fullres ${f} --npz
done
```

Other common variants:

| Goal | Command |
|------|---------|
| Lower memory footprint | `nnUNetv2_train 456 3d_lowres 0 --npz` |
| 2D model               | `nnUNetv2_train 456 2d 0 --npz` |
| Shorter (250 epochs)   | `nnUNetv2_train 456 3d_fullres 0 -tr nnUNetTrainer_250epochs --npz` |
| Resume training        | append `--c` |

Checkpoints live under `${nnUNet_results}/Dataset456_PENGWIN/`.

## Step 4 — Inference (example with model trained on fold 0)

```bash
nnUNetv2_predict \
    -i <input_dir> \
    -o <output_dir> \
    -d 456 \
    -c 3d_fullres \
    -f 0 
```
