
# Inference

The inference is done in two phases:

- Phase 1: Anatomy prediction (sacrum, hips, femur) with the baseline trained on the 456 dataset

- Phase 2: Fragment prediction with the baseline trained on the 457 dataset

---

## Inference pipeline

```
Phase 1:
                ┌─────────────────────────────┐                      filter_left_right.py
       CT   ──► │Anatomical model (Dataset456)│ ──► 5-class anatomy ──────────────────► (improved) 5-class anatomy
 + Heatmaps for │   0=bg, 1=sacrum, 2=leftHip,│      mask per voxel                                 mask per voxel
 each anatomy   │   3=rightHip, 4=femur       │
(1 + 4 channels)└─────────────────────────────┘

-------------------------------------------------------------------------------------------------------------------

Phase 2:
                ┌─────────────────────────────┐                      
       CT   ──► │                             │ ──► 2-class fragment ─────────────────┐ 
 + Heatmap for  │  Fragment model (Dataset457)│       mask per voxel                  │
 one fragment + │       0=bg, 1=fragment      │                                       │
 Heatmap for    │                             │                                       │
 background     └─────────────────────────────┘                                       │
(1 + 2 channels)                                                                      │
          │                                                                           │ 
          └───────────────────────────────────────────────────────────────────────────┘
                       repeat with another forward pass for each fragment
                              num. forward passes = num. fragments
                    skip anatomies with only one component (copy them from Phase 1) 
                                                  │
                                                  │
                                                  │ for each fragment prediction
                                                  ▼
                                    ┌────────────────────────────┐
                                    │ Mask the 2-class prediction│
                                    │ with the anatomy prediction│
                                    │ from Phase 1, i.e., zero   │
                                    │ out voxels outside anatomy │
                                    └─────────────┬──────────────┘
                                                  ▼
                                    ┌────────────────────────────┐
                                    │Copy prediction for unbroken│ 
                                    │     bones from Phase 1     │      
                                    └─────────────┬──────────────┘
                                                  ▼
                                    ┌────────────────────────────┐
                                    │  Map all prediction IDs    │ ──► challenge submission
                                    │  into PENGWIN ranges       │      (sacrum 1-50, leftHip 51-100,
                                    │                            │       rightHip 101-150, femur 151-200)
                                    └────────────────────────────┘
```

`frac_to_instance.py` covers the boxed step. The final ID-range offset stage is dataset-specific and is left to the user.

---

## Inference in Phase 1
Convert the json file to heatmaps and rename the image to end on ```_0000.mha``` using this command:

```bash
python convert_anatomy_to_nnunet_input.py 
    --json input/perpelvic-fragment-clicks.json \
    --image input/peripelvic-fracture-ct/test_case.mha \
    --output temp/nnunet_input/ 
```

This results in the following **input:**
```
├── temp/nnunet_input/          <pid>_0000.mha (CT)
                                <pid>_0001.mha (sacrum clicks)
                                <pid>_0002.mha (left hip clicks)
                                <pid>_0003.mha (right hip clicks)
                                <pid>_0004.mha (femur clicks)
```

Resample to expected nnUNet spacing (we do this as nnUNet is quite slow by default):
```bash 
python resample_input.py 
    --input_dir temp/nnunet_input/ \
    --output_dir temp/nnunet_input/resampled_anatomy 
```
Then we predict with the model:
```bash
# 1) Anatomical predictions (disabled tta to fit the 10 minute time limit)
nnUNetv2_predict \
    -i  temp/nnunet_input/resampled_anatomy \
    -o  temp/predictions/anatomical \
    -d  456 -c 3d_fullres -f 0 --disable_tta
```

Resample prediction back to orignal resolution:
```bash 
python resample_input.py 
    --input_dir temp/predictions/anatomical/ \
    --output_dir temp/predictions/anatomical/ \
    --load_spacing temp/nnunet_input/resampled_anatomy/original_spacing.npy
```

We also do a very simple post-processing of these predictions to make sure that the left hip is on the left and the right hip is on the right. This script computes the centroid of the sacrum and flips all connected components with centroids left of the sacrum to the leftHip class, and all components with centroid right of the sacrum to the rightHip class. The femur and sacrum predictions remain unchanged.
```bash
python filter_left_right.py \
    --input  temp/predictions/anatomical/ 
```


**Output**: A directory of anatomy predictions saved by `nnUNetv2_predict`. Each `.mha` is an integer volume with values:

| value | meaning |
|-------|---------|
| 0     | background |
| 1     | sacrum     |
| 2     | leftHip    |
| 3     | rightHip   |
| 4     | femur      | 



## Inference in Phase 2

We first create the nnUNet inputs for the Phase 2 baseline model:

```bash
python convert_fragments_to_nnunet_input.py 
    --json input/perpelvic-fragment-clicks.json \
    --image input/peripelvic-fracture-ct/test_case.mha
    --output_base temp/nnunet_input/fragments/
```
This results into multiple cases you need to process with the nnUNet fragment model saved in `sacrum`, `left_hip`, `right_hip`, and `femur` subdirectories. If a directory for a certain anatomy does not exist, it means it has only one (or 0) fragments and we will just copy the prediction from the anatomy model.
```
temp/nnunet_input/fragments/
└── sacrum
    ├── 1
    │   ├── <pid>_0000.mha
    │   ├── <pid>_0001.mha
    │   └── <pid>_0002.mha
    ├── 2
    │   ├── <pid>_0000.mha
    │   ├── <pid>_0001.mha
    │   └── <pid>_0002.mha
    └── 3
        ├── <pid>_0000.mha
        ├── <pid>_0001.mha
        └── <pid>_0002.mha
└── left_hip
│   ├── 51
│   │   ├── <pid>_0000.mha
│   │   ├── <pid>_0001.mha
│   │   └── <pid>_0002.mha
│   └── 52
│       ├── <pid>_0000.mha
│       ├── <pid>_0001.mha
│       └── <pid>_0002.mha
...

```

Let us then move all these files to a single directory so nnUNet can see them all and predict without loading the model for each fragment:
```bash
bash create_all_cases_dir.sh
```
Then we resample the `mha` files to make the predictions faster with nnUNet:
```bash
python resample_input.py 
        --input_dir temp/nnunet_input/fragments/all_data/ \
        --output_dir temp/nnunet_input/fragments/all_data/resampled 
        --target fragments
```

Now, all we need to do now is predict for each fragment:
```bash
# 2) Fragment predictions
nnUNetv2_predict \
    -i  temp/nnunet_input/fragments/all_data/resampled \
    -o  temp/predictions/fragments/all_data \
    -d  457 -c 3d_fullres -f 0 --disable_tta
```
And resample back to the original resolution:
```bash
python resample_input.py 
    --input_dir temp/predictions/fragments/all_data/ \
    --output_dir temp/predictions/fragments/all_data/ \
    --load_spacing temp/nnunet_input/fragments/all_data/resampled/original_spacing.npy
```

And move back all data to the original file structure:
```bash
bash move_all_data_to_separate_folders.sh
```



Then, we postprocess these predictions to only keep the connected component that is clicked by the heatmap:
```bash
bash keep_connected_components.sh
```


Then, for anatomies with only a single fragment, we simply copy their predictions into the same interface:
```bash
python copy_single_fragments.py \
    --anatomy_pred temp/predictions/anatomical/ \
    --fragment_input temp/nnunet_input/fragments/ \
    --fragment_output temp/predictions/fragments/ \
    --pid <pid>
```

We merge all of these predictions into one final `.mha` file containing values between `0-200` in the PENGWIN format. We use this script:

```bash
python merge_fragment_predictions.py \
    --input temp/predictions/fragments/ \
    --output output/pelvic-fracture-segmentation/final_prediction.mha \
    --pid <pid>
```

As a final step, we fill up all `background` voxels with predicted anatomy based on the fragment from that anatomy that has the largest volume. 
```bash
python expand_fragments_to_anatomy.py \
    --anat_pred temp/predictions/anatomical/<pid>.mha \
    --frag_pred output/pelvic-fracture-segmentation/final_prediction.mha \
    --output output/pelvic-fracture-segmentation/final_prediction.mha 
```