#!/bin/bash

# Create output directory
mkdir -p output

echo "Starting batch extraction..."
echo "============================="

# Loop through all files in the input directory
for file in input/*; do
  if [ -f "$file" ]; then
    filename=$(basename -- "$file")
    name="${filename%.*}"
    
    echo "Processing: $file"
    
    # Run the extractor — saves JSON + Markdown table
    ./venv/bin/python3 -m statement_extractor extract "$file" \
      --json    "output/${name}.json" \
      --markdown "output/${name}.md" \
      --debug \
      --debug-dir "output/debug_${name}"
      
    echo "Done: $name"
    echo "-----------------------------"
  fi
done

echo "All files processed successfully!"
