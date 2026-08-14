# Preprocessing scripts for the baseline model (Phase 1: Anatomy)

---
- `convert_to_nnunet_format.py`
    -   
Convert the dataset from the original PENGWIN format stored on [Zenodo](https://zenodo.org/records/19732767) to the nnUNet format with `imagesTr` and `labelsTr`.
## Usage 
```bash
   python convert_to_nnunet_format.py --input_dir [YOUR_PATH_TO_PENGWIN_FROM_ZENODO] --output_dir [YOUR_OUTPUT_PATH]
```
```bash
   options:
    --input_dir INPUT_DIR
                            Root directory containing PENGWIN parts
    --output_dir OUTPUT_DIR
                            Output nnU-Net dataset directory
```
The scripts expects the input to be stored in this file structure:
  ```bash
  [YOUR_PATH]
  ├── PENGWIN26_task1_2_train_part1
  ├── PENGWIN26_task1_2_train_part2
  ├── PENGWIN26_task1_2_train_part3
  ├── PENGWIN26_task1_2_train_part4
  └── PENGWIN26_task2_train_clicks
  ```
and will copy these files to a new folder structure as:
```
[YOUR_OUTPUT_PATH_HERE]
├── imagesTr/      <pid>_0000.nii.gz
├── labelsTr/      <pid>.nii.gz       
```


---
- `add_pengwin_heatmaps_anatomy.py`
    -   
Add Gaussian heatmaps to PENGWIN nnU-Net dataset using the pre-simulated clicks from the JSON files. This will generate 4 heatmaps for each CT image - one heatmaps for each label (`0=background, 1=sacrum, 2=leftHip, 3=rightHip, 4=femur`). It will choose randomly from the four click simulation strategies for each heatmap (`uniformly_sampled`, `euclidean_distance_transform`, `center_of_mass`, or `boundary_internal_margin`).
## Usage 
```bash
   python add_pengwin_heatmaps_anatomy.py 
            --input_images [YOUR_NNUNET_DATA_PATH]/imagesTr \
            --clicks_root [YOUR_PATH_TO_PENGWIN_FROM_ZENODO]/PENGWIN26_task2_train_clicks \
            --output_images [YOUR_NNUNET_DATA_PATH]/imagesTr
```
```bash
   options:
    --input_images INPUT_IMAGES
                            Path to imagesTr (nnU-Net format)
    --clicks_root CLICKS_ROOT
                            Path to PENGWIN26_task2_train_clicks
    --output_images OUTPUT_IMAGES
                            Output imagesTr with heatmaps

```
---
- `remap_pengwin_labels.py`
    -   
Remap PENGWIN labels to 0–4 classes (`0=background, 1=sacrum, 2=leftHip, 3=rightHip, 4=femur`)
## Usage 
Note: This will save the original labels with class labels `0-200` in `[YOUR_PATH]/labelsTr_unmapped` and overwrite the `labelsTr` directory with the remapped labels.
```bash
   python remap_pengwin_labels.py --input_labels [YOUR_PATH]/labelsTr --output_labels [YOUR_PATH]/labelsTr
```
```bash
options:
  --input_labels INPUT_LABELS
                        Path to original labelsTr directory 
  --output_labels OUTPUT_LABELS
                        Path to remapped labelsTr directory 
```

