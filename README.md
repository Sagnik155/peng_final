# Peripelvic Fracture Segmentation using Interactive Click-Guided 3D UNet

## Project Overview

This project implements an interactive medical image segmentation system designed for identifying and segmenting peripelvic bone fractures in CT scans. The system uses a lightweight 3D U-Net neural network combined with a multi-stage post-processing pipeline to accurately segment individual fracture fragments based on minimal user interaction through click-based annotations.

The solution addresses the PENGWIN (Peripelvic-Femur Fracture Segmentation) challenge, where the goal is to segment distinct bone fracture fragments in pelvic CT images using sparse interactive input.

---

## System Architecture

### 1. **Input Processing**

The pipeline accepts the following inputs:

- **CT Images**: Medical images in `.mha` (MetaHeader) format containing 3D volumetric CT scan data
- **Click Annotations**: JSON files containing user-provided 2D click coordinates that indicate regions of interest (ROIs) within the CT volume
- **Multiple Click Strategies**: The system supports various click sampling strategies including:
  - `uniformly_sampled` — evenly distributed clicks across fracture fragments
  - `center_of_mass` — clicks at the centroid of each fragment
  - `boundary_internal_margin` — clicks at region boundaries and internal margins
  - `euclidean_distance_transform` — clicks based on distance transforms

### 2. **Core Processing Pipeline**

The system follows a sequential architecture:

#### **Stage 1: Click Parsing & Routing**
- **Component**: `click_parser.py`, `pelvic_femur_router.py`
- **Processing**:
  - Extracts 2D click coordinates from JSON annotations
  - Analyzes CT image metadata (voxel spacing, physical dimensions)
  - Routes images to the appropriate segmentation model based on anatomical classification (pelvic vs. femur)

#### **Stage 2: Region of Interest (ROI) Generation & Inference**
- **Component**: `parallel_infer/batch_runner.py`, `fragment_seg/unet3d.py`
- **Processing**:
  - Generates 3D ROI patches around each click location
  - Passes ROI patches through a lightweight 3D U-Net (16 base filters)
  - Model produces dual-head outputs: **Core Mask** (fragment interior) and **Boundary Mask** (fracture edges)
  - Network trained with combined loss functions for robust segmentation
  - Batch processing for computational efficiency

**Model Architecture**: 
- **Input**: 2 channels (CT image + click point indicator)
- **Output**: 2 channels (core + boundary segmentation masks)
- **Design**: Lightweight encoder-decoder with skip connections, GroupNorm normalization, and 3D convolutions

#### **Stage 3: Overlapping Prediction Merging**
- **Component**: `fusion/merge_overlap.py`
- **Processing**:
  - Combines predictions from multiple ROIs with IoU-based merging (threshold: 0.5)
  - Resolves overlapping regions to produce a unified segmentation
  - Prevents duplicate detections from adjacent ROI patches

#### **Stage 4: Conflict Resolution & Instance Packing**
- **Component**: `fusion/resolve_conflicts.py`, `baseline_adapter/run_baseline.py`
- **Processing**:
  - Resolves boundary conflicts using anatomical guidance
  - Integrates baseline anatomy segmentation masks for anatomical correctness
  - Packs final instance IDs into the range 0-200 (uint8)
  - Ensures anatomical validity of the final segmentation

#### **Stage 5: Output Generation**
- **Component**: `shared/io_utils.py`
- **Output**:
  - Instance segmentation mask (each pixel labeled with fragment ID)
  - Saved in `.mha` format preserving original image metadata
  - Output directory: `/output/images/peripelvic-fracture-ct-segmentation/output.mha`

---

## Input → Transformation → Output Flow

```
CT Image (.mha)  +  Click Annotations (.json)
        ↓
    Click Parsing
    (Extract coordinates & routes)
        ↓
ROI Generation (3D patches around clicks)
        ↓
3D U-Net Inference (Fragment Segmentation)
    ↓ (Core + Boundary masks)
Overlap Merging (IoU-based fusion)
        ↓
Conflict Resolution (with anatomical guidance)
        ↓
Instance Packing (ID range 0-200)
        ↓
Output Instance Segmentation (.mha)
```

---

## Technical Stack

- **Deep Learning Framework**: PyTorch (≥2.0.0)
- **Medical Image I/O**: SimpleITK (≥2.3.0)
- **Numerical Computing**: NumPy (≥1.24.0), SciPy (≥1.11.0)
- **Model Architecture**: 3D U-Net with GroupNorm & skip connections
- **Deployment**: Docker containerization for portability

---

## Evaluation Criteria

The project uses multiple evaluation metrics to assess segmentation performance:

### **1. Fragment-Level Dice Similarity Coefficient**
- Measures overlap between predicted and ground truth fragment masks
- Range: 0 to 1 (1 = perfect segmentation)
- Computed as: `2 × Intersection / (Pred + GT)`
- **Purpose**: Evaluates spatial accuracy of individual fragments

### **2. Merge Errors**
- Counts instances where multiple ground truth fragments are incorrectly merged into a single prediction
- **Purpose**: Penalizes over-segmentation and false merging
- Target: Minimize merge errors

### **3. Split Errors**
- Counts instances where a single ground truth fragment is incorrectly split into multiple predictions
- **Purpose**: Penalizes under-segmentation and false splitting
- Target: Minimize split errors

### **4. Intersection over Union (IoU)**
- Measures region overlap: `Intersection / Union`
- Provides complementary view to Dice coefficient
- **Purpose**: Evaluates boundary accuracy

### **5. 95th Percentile Hausdorff Distance (HD95)**
- Measures maximum boundary distance between predictions and ground truth
- Units: millimeters (accounting for voxel spacing)
- **Purpose**: Evaluates boundary precision at 95th percentile
- Lower values indicate better boundary alignment

### **6. Average Symmetric Surface Distance (ASSD)**
- Mean distance between prediction and ground truth surfaces
- Units: millimeters
- **Purpose**: Provides robust boundary evaluation metric
- Lower values indicate better boundary accuracy

---

## Project Structure

```
peng_final/
├── method/
│   ├── predict.py                    # Main prediction pipeline
│   ├── config.py                     # Configuration & paths
│   ├── fragment_seg/
│   │   ├── unet3d.py                # 3D U-Net model definition
│   │   ├── train.py                 # Model training script
│   │   └── losses.py                # Loss functions
│   ├── parallel_infer/
│   │   └── batch_runner.py          # Batch ROI inference
│   ├── fusion/
│   │   ├── merge_overlap.py         # Overlapping prediction merging
│   │   └── resolve_conflicts.py     # Boundary conflict resolution
│   ├── baseline_adapter/
│   │   └── run_baseline.py          # Anatomical mask integration
│   ├── shared/
│   │   ├── click_parser.py          # Click annotation parsing
│   │   ├── io_utils.py              # MHA file I/O operations
│   │   ├── pelvic_femur_router.py  # Anatomical routing logic
│   │   └── label_conventions.py    # Labeling standards
│   └── evaluation/
│       ├── metrics.py               # Fragment-level metrics (Dice, Merge/Split errors)
│       └── pengwin_metrics.py       # Surface-based metrics (IoU, HD95, ASSD)
├── predict.py                        # Docker entry point
├── Dockerfile                        # Containerization
├── build.bat                         # Windows build script
└── requirements.txt                  # Python dependencies
```

---

## Key Features

✓ **Interactive Segmentation**: Minimal user effort through sparse click annotations  
✓ **Fragment-Level Accuracy**: Individual fracture identification with Dice scoring  
✓ **Error Quantification**: Explicit merge and split error tracking  
✓ **Surface-Based Evaluation**: HD95 and ASSD for boundary precision  
✓ **Anatomical Guidance**: Integration with baseline anatomy segmentation  
✓ **Scalability**: Batch processing and GPU acceleration support  
✓ **Containerized**: Docker support for reproducible deployment  

---

## Dependencies

All required packages are listed in `requirements.txt`:
- `torch` — Deep learning framework
- `torchvision` — Imaging utilities
- `numpy` — Numerical computing
- `SimpleITK` — Medical image I/O
- `scipy` — Scientific computing
- `tqdm` — Progress tracking

Install with:
```bash
pip install -r requirements.txt
```

---

## Usage

### Local Execution
```bash
python method/predict.py
```

### Docker Execution
```bash
docker build -t peripelvic-fracture-seg .
docker run -v /input:/input -v /output:/output peripelvic-fracture-seg
```

The pipeline expects:
- **Input**: `/input/images/peripelvic-fracture-ct/` (CT images and click annotations)
- **Output**: `/output/images/peripelvic-fracture-ct-segmentation/output.mha`

---

## Summary

This project implements a complete end-to-end medical image segmentation pipeline that transforms minimal user input (CT images + sparse clicks) into high-quality instance segmentation maps of bone fracture fragments. The multi-stage architecture combines deep learning inference with post-processing fusion and anatomical constraints, evaluated through comprehensive metrics that assess both spatial accuracy and boundary precision.
