#!/bin/bash

# Loop through both directory pairs
for source_dir in "temp/predictions/fragments/all_data" "temp/nnunet_input/fragments/all_data"; do
    # Determine destination base (remove /all_data from path)
    dest_base="${source_dir%/all_data}"
    
    echo "=========================================="
    echo "Processing: $source_dir"
    echo "=========================================="
    
    # Loop through all .mha files in all_data
    for file in "$source_dir"/*.mha; do
        # Skip if no files
        [ -f "$file" ] || continue
        
        filename=$(basename "$file")
        basename="${filename%.mha}"
        
        # Split by underscore
        IFS='_' read -ra fields <<< "$basename"
        
        # Find fragment (numeric field)
        fragment=""
        anatomy_parts=()
        for i in "${!fields[@]}"; do
            if [[ "${fields[$i]}" =~ ^[0-9]+$ ]]; then
                fragment="${fields[$i]}"
                anatomy_parts=("${fields[@]:0:$i}")
                break
            fi
        done
        
        # Reconstruct anatomy
        anatomy=$(IFS='_'; echo "${anatomy_parts[*]}")
        
        # Get original filename
        for i in "${!fields[@]}"; do
            if [[ "${fields[$i]}" == "$fragment" ]]; then
                remaining_fields=("${fields[@]:$i+1}")
                original_filename=$(IFS='_'; echo "${remaining_fields[*]}").mha
                break
            fi
        done
        
        # Create destination and move
        dest_dir="$dest_base/$anatomy/$fragment"
        mkdir -p "$dest_dir"
        mv "$file" "$dest_dir/$original_filename"
        
        echo "  $filename -> $anatomy/$fragment/$original_filename"
    done
    
    # Remove empty directory
    rmdir "$source_dir" 2>/dev/null
    echo ""
done

# Show results
echo -e "=== Restored directory structure ===\n"
for base in "temp/predictions/fragments" "temp/nnunet_input/fragments"; do
    if [ -d "$base" ]; then
        echo "📁 $base:"
        tree "$base" -L 2 2>/dev/null || find "$base" -type f -name "*.mha" | head -10
        echo ""
    fi
done

echo "✅ Complete!"