#!/bin/bash

# Define source and destination directories
SOURCE_DIR="temp/nnunet_input/fragments"
DEST_DIR="temp/nnunet_input/fragments/all_data"

# Create destination directory
mkdir -p "$DEST_DIR"

# Loop through anatomy directories
for anatomy in left_hip sacrum; do
    # Loop through fragment directories
    for fragment in "$SOURCE_DIR/$anatomy"/*; do
        # Get just the fragment name (e.g., 51, 52, 1, 2, 3)
        fragment_name=$(basename "$fragment")
        
        # Loop through .mha files
        for file in "$fragment"/*.mha; do
            # Get just the filename
            filename=$(basename "$file")
            
            # Create new filename
            new_filename="${anatomy}_${fragment_name}_${filename}"
            
            # Copy the file
            mv "$file" "$DEST_DIR/$new_filename"
        done
    done
done

# List all files in the new directory
echo -e "\n=== Files in $DEST_DIR ===\n"
ls -1 "$DEST_DIR"
