# Pelvic Fragment Click Generator

Generate intelligent point-of-interest clicks from 3D pelvis fragment labels for training interactive segmentation models.

## Overview

This script processes 3D MHA label files containing pelvic bone fragments (Sacrum, Left Hip, Right Hip, Femur) and generates four types of click points:

-  Center of Mass - Geometric or EDT-based center points
-  Uniformly Sampled - Random interior points (4 per fragment)
-  Euclidean Distance Transform - Maximum EDT interior points  
-  Boundary Internal Margin - Points along interior margins (4 per fragment)

##  Quick Start
```
python generate_clicks.py -i /path/to/label.mha --json_output /path/to/output
```

##  Arguments


| Argument | Required | Description |
|----------|----------|-------------|
| -i, --input_label | Yes | Path to the input MHA label file |
| --json_output | Yes | Output directory for JSON click files |
| --debug_output | No | Output directory for visualization heatmaps (MHA format) |

##  Output Structure

```
json_output/
├── center_of_mass/
│   └── {subject_id}/
│       └── peripelvic-fragment-clicks.json
├── uniformly_sampled/
│   └── {subject_id}/
│       └── peripelvic-fragment-clicks.json
├── boundary_internal_margin/
│   └── {subject_id}/
│       └── peripelvic-fragment-clicks.json
└── euclidean_distance_transform/
    └── {subject_id}/
        └── peripelvic-fragment-clicks.json
```

##  JSON Output Format
```
{
  "name": "Center of Mass Points of Interest",
  "type": "Multiple Points",
  "version": {
    "major": 1,
    "minor": 0
  },
  "points": [
    {
      "name": "Sacrum Center of Mass Point 1",
      "point": [123, 45, 67]
    },
    {
      "name": "Left Hip Euclidean Distance Transform Point 2",
      "point": [89, 101, 112]
    }
  ]
}
```
##  Label Mapping

| Label Range | Anatomical Region |
|-------------|-------------------|
| 1 - 50 | Sacrum |
| 51 - 100 | Left Hip |
| 101 - 150 | Right Hip |
| 151 - 200 | Femur |

##  Dependencies

```pip install SimpleITK cc3d numpy cupy cucim scipy```

Note: CUDA-capable GPU recommended for optimal performance. Falls back to CPU if GPU memory is insufficient.

##  Debug Mode

Generate heatmap visualizations for verification:

```
python generate_clicks.py -i label.mha --json_output ./json --debug_output ./debug
```

This creates MHA files with Gaussian heatmaps around each click point.

##  Example

# Process a single subject
```
python generate_clicks.py \
  -i datasets/pelvis/subject_001/labels/fragments.mha \
  --json_output ./clicks/subject_001
```
# With debug visualizations
```
python generate_clicks.py \
  -i datasets/pelvis/subject_001/labels/fragments.mha \
  --json_output ./clicks/subject_001 \
  --debug_output ./debug/subject_001
```
##  Click Generation Logic

| Type | Method |
|------|--------|
| Uniformly Sampled  | Random interior points |
| Euclidean Distance Transform | Point farthest from boundary |
| Center of Mass  | Geometric centroid (fallback to EDT if outside) |
| Boundary Internal Margin  | Points on inverted EDT ridge |



