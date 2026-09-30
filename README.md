# Peripelvic Fracture Segmentation with Interactive 3D U-Net

This repository contains an interactive segmentation pipeline for pelvic and femur fracture fragments in CT scans. It was developed for the PENGWIN (Peripelvic-Femur Fracture Segmentation) challenge.

The main idea is fairly simple: a user provides a small number of clicks, and the model uses those clicks to guide segmentation of the corresponding fracture fragments. The pipeline works with 3D `.mha` volumes and produces an instance segmentation mask in the same format.

## How it works

The prediction pipeline is split into a few stages:

1. **Read the input**
   - CT volumes are read from `.mha` files.
   - Click annotations are read from JSON files.
   - The click parser extracts the coordinates and the router determines whether the input should be handled as pelvic or femur data.

2. **Run click-guided inference**
   - A 3D region of interest is created around each click.
   - Each ROI is passed through a lightweight 3D U-Net.
   - The network takes two channels as input: the CT data and a click-point indicator.
   - It predicts two channels, representing the fragment core and its boundary.

3. **Merge ROI predictions**
   - Predictions from overlapping ROIs are compared and merged using an IoU threshold of `0.5`.
   - This helps avoid duplicate instances when nearby clicks produce overlapping predictions.

4. **Resolve conflicts and build instances**
   - Conflicting boundary regions are resolved before the predictions are combined into the final mask.
   - The baseline anatomy masks can also be used at this stage to provide additional anatomical context.
   - Instance labels are stored as `uint8` values in the range `0–200`.

5. **Write the result**
   - The final instance mask is saved as an `.mha` file with the original image metadata preserved.

The overall flow looks like this:

```text
CT volume + click annotations
            |
      Parse and route clicks
            |
       Create 3D ROIs
            |
   Click-guided 3D U-Net
    (core + boundary masks)
            |
      Merge overlapping ROIs
            |
      Resolve conflicts
            |
       Pack instance labels
            |
       Output segmentation
```

## Click strategies

The project supports several ways of generating or selecting clicks:

- `uniformly_sampled`: distributes clicks across a fragment
- `center_of_mass`: places a click near the fragment centroid
- `boundary_internal_margin`: samples near boundaries and internal margins
- `euclidean_distance_transform`: uses a distance transform to select click locations

## Model

The segmentation model is a compact 3D U-Net implemented in PyTorch. It uses 3D convolutions, skip connections, and GroupNorm. The default model uses 16 base filters.

- **Input:** 2 channels (CT image and click indicator)
- **Output:** 2 channels (fragment core and boundary)
- **Framework:** PyTorch

## Repository layout

```text
peng_final/
├── method/
│   ├── predict.py                    # Main prediction pipeline
│   ├── config.py                     # Configuration and paths
│   ├── fragment_seg/
│   │   ├── unet3d.py                 # 3D U-Net definition
│   │   ├── train.py                  # Training script
│   │   └── losses.py                 # Loss functions
│   ├── parallel_infer/
│   │   └── batch_runner.py           # Batch ROI inference
│   ├── fusion/
│   │   ├── merge_overlap.py          # Merge overlapping predictions
│   │   └── resolve_conflicts.py      # Resolve prediction conflicts
│   ├── baseline_adapter/
│   │   └── run_baseline.py           # Use baseline anatomy masks
│   ├── shared/
│   │   ├── click_parser.py           # Parse click annotations
│   │   ├── io_utils.py               # MHA input/output helpers
│   │   ├── pelvic_femur_router.py    # Pelvic/femur routing
│   │   └── label_conventions.py      # Label definitions
│   └── evaluation/
│       ├── metrics.py                # Fragment-level metrics
│       └── pengwin_metrics.py        # Surface-based metrics
├── predict.py                         # Container entry point
├── Dockerfile
├── build.bat
└── requirements.txt
```

## Requirements

The main dependencies are:

- Python
- PyTorch (2.0 or newer)
- SimpleITK (2.3 or newer)
- NumPy (1.24 or newer)
- SciPy (1.11 or newer)
- torchvision
- tqdm

Install them with:

```bash
pip install -r requirements.txt
```

## Running the pipeline

### Locally

```bash
python method/predict.py
```

### With Docker

```bash
docker build -t peripelvic-fracture-seg .
docker run -v /input:/input -v /output:/output peripelvic-fracture-seg
```

The expected directories are:

```text
/input/images/peripelvic-fracture-ct/
/output/images/peripelvic-fracture-ct-segmentation/output.mha
```

The input directory should contain the CT image and its click annotations. The output file is an instance segmentation mask, where each non-zero value identifies a fracture fragment.

## Evaluation

The evaluation code includes the following measures:

- **Fragment Dice:** overlap between an individual predicted fragment and its reference mask
- **Merge errors:** cases where multiple reference fragments are combined into one prediction
- **Split errors:** cases where one reference fragment is divided into multiple predictions
- **IoU:** intersection over union of the predicted and reference regions
- **HD95:** the 95th-percentile surface distance in millimetres
- **ASSD:** average symmetric surface distance in millimetres

The fragment-level metrics are implemented in `method/evaluation/metrics.py`; the surface-based metrics are in `method/evaluation/pengwin_metrics.py`.

## Notes

This project is intended for research and challenge evaluation. Results depend on the trained model weights, the click annotations, and the input volume metadata. The repository provides the processing code, but it does not replace clinical review or validation on an independent dataset.
