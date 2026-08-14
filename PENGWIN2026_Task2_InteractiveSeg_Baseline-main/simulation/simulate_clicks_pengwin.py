import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_label", required=True, help="Path to the mha label")
parser.add_argument("--debug_output", required=False, help="Output path for click visualization for debugging")
parser.add_argument("--json_output", required=True, help="Output path for JSON containing all clicks")
args = parser.parse_args()


import SimpleITK as sitk

import cc3d
import numpy as np
import cupy as cp
from cucim.core.operations import morphology
import json

def load_image_file_as_array(*, location):
    """Loads an image using SimpleITK and returns the image array along with its metadata."""
    input_file = location
    image = sitk.ReadImage(input_file)

    # Extract metadata
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()

    # Convert image to a numpy array
    array = sitk.GetArrayFromImage(image)

    return array, origin, spacing, direction


def write_array_as_image_file(*, location, array, origin, spacing, direction):
    """Writes a numpy array as an image file while preserving origin, spacing, and direction."""
    image = sitk.GetImageFromArray(array)

    # Set metadata
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)

    sitk.WriteImage(image, location, useCompression=True)

# For click visualization 
def generate_gaussian_heatmap(coords, shape, sigma=2.0):
    from scipy.ndimage import gaussian_filter
    """
    Generate a 3D Gaussian heatmap for given coordinates.

    Args:
        coords (list): List of [x, y, z] coordinates.
        shape (tuple): Shape of the output volume.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: 3D volume with Gaussian heatmaps at the specified coordinates.
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    for coord in coords:
        if 0 <= coord[0] < shape[0] and 0 <= coord[1] < shape[1] and 0 <= coord[2] < shape[2]:
            heatmap[tuple(coord)] = 1.0

    heatmap = gaussian_filter(heatmap, sigma=sigma)    
    return heatmap

def save_click_heatmaps(clicks, ref_shape, debug_output, input_label, origin, spacing, direction, suffix=''):
    fragment_coords = clicks['points']
    fragment_coords = [point['point'] for point in fragment_coords]
    fragment_heatmap = generate_gaussian_heatmap(fragment_coords, ref_shape, 3)
    write_array_as_image_file(
        location=os.path.join(debug_output, f'{input_label.split("/")[-1].split(".mha")[0]}_{label}_{suffix}_heatmaps.mha'),
        array=fragment_heatmap,
        origin=origin,
        spacing=spacing,
        direction=direction,
    )

def map_label(label):
    if label <= 50:
        return 'Sacrum'
    elif label <= 100:
        return 'Left Hip'
    elif label <= 150:
        return 'Right Hip'
    elif label <= 200:
        return 'Femur'


SEED = 42
np.random.seed(SEED)
cp.random.seed(SEED)

label_im, origin, spacing, direction = load_image_file_as_array(location=args.input_label)
ref_shape = label_im.shape
labels = np.unique(label_im)
clicks_center = {"name": "Center of Mass Points of Interest", "type": "Multiple Points", "points":[], "version": {"major": 1, "minor": 0}}
clicks_uniform = {"name": "Uniformly Sampled Points of Interest", "type": "Multiple Points", "points":[], "version": {"major": 1, "minor": 0}}
clicks_interior_margin = {"name": "Boundary Internal Margin Points of Interest", "type": "Multiple Points", "points":[], "version": {"major": 1, "minor": 0}}
clicks_euclidean ={"name": "Euclidean Distance Transform Points of Interest", "type": "Multiple Points", "points":[], "version": {"major": 1, "minor": 0}}


if np.sum(label_im) == 0:
    print("[WARNING] GT is empty, no fragments found!")
    exit()
else: 
    ##### Fragment Clicks #####
    unique_labels = labels[1:] # Skip background label 0

    # Sample center clicks for 10 random components
    for ind, label in enumerate(unique_labels, start=1):    
        labeled_mask = label_im == label
        labeled_mask = cp.array(labeled_mask)
        try:
            # Attempt to compute EDT using GPU
            edt = morphology.distance_transform_edt(labeled_mask)
        except MemoryError:
            from scipy import ndimage

            print("Out of memory on GPU! Applying CPU-based EDT. This might be a bit slower...")
            edt = ndimage.morphology.distance_transform_edt(labeled_mask.get())
            edt = cp.array(edt)
            labeled_mask = cp.array(labeled_mask)

        # Click Simulation 1
        # Sample a random click
        indices = cp.argwhere(labeled_mask == 1)
        random_indices = indices[cp.random.permutation(indices.shape[0])[:1]]
        random_indices_host = random_indices.get()
        for click in random_indices_host:
            assert label_im[int(click[0]), int(click[1]), int(click[2])] == label   
            clicks_uniform['points'].append({'name': f'{map_label(label)} Uniformly Sampled Point {ind}', 'point': [int(click[0]), int(click[1]), int(click[2])]})


        # Click simulation 2
        # Compute Euclidean Distance Transform Maximum 
        center_edt = cp.unravel_index(cp.argmax(edt), edt.shape)
        assert label_im[int(center_edt[0]), int(center_edt[1]), int(center_edt[2])] == label
        clicks_euclidean['points'].append({'name': f'{map_label(label)} Euclidean Distance Transform Point {ind}', 'point': [int(center_edt[0]), int(center_edt[1]), int(center_edt[2])]})



        # Click simulation 3
        # Compute Geometric Center
        stats = cc3d.statistics(labeled_mask.get())
        center_geometric = [int(el) for el in stats['centroids'][1]]
        if not label_im[int(center_geometric[0]), int(center_geometric[1]), int(center_geometric[2])] == label:
            print("Geometric center is outside the label due to concavity! Using EDT instead!")
            clicks_center['points'].append({'name': f'{map_label(label)} Center of Mass Point {ind}', 'point': [int(center_edt[0]), int(center_edt[1]), int(center_edt[2])]})
        else:
            clicks_center['points'].append({'name': f'{map_label(label)} Center of Mass Point {ind}', 'point': [int(center_geometric[0]), int(center_geometric[1]), int(center_geometric[2])]})

        
        # Click simulation 4
        # Interior Margin Points - Sample one boundary click
        edt_inverted = (cp.max(edt) - edt) * (edt > 0) 
        boundary_elements = (edt_inverted == cp.max(edt_inverted)) * (labeled_mask > 0)
        indices = cp.array(cp.nonzero(boundary_elements)).T.get()  # Shape: (num_true, ndim)
        for i in range(1):
            boundary_click = indices[np.random.choice(indices.shape[0])]
            clicks_interior_margin['points'].append({'name': f'{map_label(label)} Boundary Internal Margin Point {ind}', 'point': [int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])]})

            assert label_im[int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])] == label

if args.debug_output is not None:
    os.makedirs(args.debug_output, exist_ok = True)
    save_click_heatmaps(clicks_center, ref_shape, args.debug_output, args.input_label, origin, spacing, direction, suffix='center')
    save_click_heatmaps(clicks_euclidean, ref_shape, args.debug_output, args.input_label, origin, spacing, direction, suffix='edt')
    save_click_heatmaps(clicks_uniform, ref_shape, args.debug_output, args.input_label, origin, spacing, direction, suffix='uniform')
    save_click_heatmaps(clicks_interior_margin, ref_shape, args.debug_output, args.input_label, origin, spacing, direction, suffix='interior')


os.makedirs(args.json_output, exist_ok=True)
os.makedirs(os.path.join(args.json_output, 'center_of_mass', args.input_label.split('/')[-2]), exist_ok=True)
os.makedirs(os.path.join(args.json_output, 'uniformly_sampled', args.input_label.split('/')[-2]), exist_ok=True)
os.makedirs(os.path.join(args.json_output, 'boundary_internal_margin', args.input_label.split('/')[-2]), exist_ok=True)
os.makedirs(os.path.join(args.json_output, 'euclidean_distance_transform', args.input_label.split('/')[-2]), exist_ok=True)




# Add indent=2 or indent=4 for pretty formatting
with open(os.path.join(args.json_output, 'center_of_mass', args.input_label.split('/')[-2], 'peripelvic-fragment-clicks.json'), 'w') as f:
    json.dump(clicks_center, f, indent=2)  # or indent=4 for more spacing

with open(os.path.join(args.json_output, 'euclidean_distance_transform', args.input_label.split('/')[-2], 'peripelvic-fragment-clicks.json'), 'w') as f:
    json.dump(clicks_euclidean, f, indent=2)

with open(os.path.join(args.json_output, 'boundary_internal_margin', args.input_label.split('/')[-2], 'peripelvic-fragment-clicks.json'), 'w') as f:
    json.dump(clicks_interior_margin, f, indent=2)

with open(os.path.join(args.json_output, 'uniformly_sampled', args.input_label.split('/')[-2], 'peripelvic-fragment-clicks.json'), 'w') as f:
    json.dump(clicks_uniform, f, indent=2)

print("Done with", args.input_label)