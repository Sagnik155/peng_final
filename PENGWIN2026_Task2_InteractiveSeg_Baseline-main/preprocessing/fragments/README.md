# Preprocessing scripts for the baseline model (Phase 2: Fragments)

---
- `create_fragment_nnUNet_dataset.py`
    -   
Creates a new dataset in the nnUNet format with foreground and background heatmaps for fragments. The output folder structure will look like this:
  ```
  [YOUR_FRAG_DATASET_PATH]
  ├── imagesTr/      case_<pid>_frag_<anat>_<frag_id>_0000.nii.gz  (CT image)
  |                  case_<pid>_frag_<anat>_<frag_id>_0001.nii.gz  (fragment click)
  |                  case_<pid>_frag_<anat>_<frag_id>_0002.nii.gz  (background click)
  |
  ├── labelsTr/      case_<pid>_frag_<anat>_<frag_id>.nii.gz       (values 0..1)
  ```

## What This Script Does
The script iterates over all cases (`<pid>`) and generates multiple new cases per original case.

For each anatomical structure with fractures (i.e., anatomies containing more than one fragment), the script creates a new case for each fragment (`<frag_id>`) where:
- The target fragment `<frag_id>` is labeled as **foreground**
- Everything else is labeled as **background**
- The corresponding fragment click generates a foreground heatmap (`_0001.mha`)
- All other clicks serve as background heatmaps (`_0002.mha`)

## Usage 
```bash
   python create_fragment_nnUNet_dataset.py 
            --images [YOUR_NNUNET_DATA_PATH]/imagesTr \
            --labels [YOUR_NNUNET_DATA_PATH]/labelsTr_unmapped \
            --clicks_root [YOUR_PATH_TO_PENGWIN_FROM_ZENODO]/PENGWIN26_task2_train_clicks \
            --out_images [YOUR_FRAG_DATASET_PATH]/imagesTr \
            --out_labels [YOUR_FRAG_DATASET_PATH]/labelsTr \

```
```bash
   options:
      --images IMAGES
      --labels LABELS
      --clicks_root CLICKS_ROOT
      --out_images OUT_IMAGES
      --out_labels OUT_LABELS
```
