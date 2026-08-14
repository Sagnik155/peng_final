for pred in temp/predictions/fragments/*/*/*.mha; do 
    pid=$(basename $(dirname $pred))
    anatomy=$(basename $(dirname $(dirname $pred)))
    heatmap=$(ls temp/nnunet_input/fragments/$anatomy/$pid/*_0001.mha 2>/dev/null)
    if [ -f "$heatmap" ]; then
        python keep_clicked_fragment.py --input_pred "$pred" --input_heatmap "$heatmap" --output "$pred" --quiet
    else
        echo "Warning: No heatmap found for $anatomy/$pid"
    fi
done
