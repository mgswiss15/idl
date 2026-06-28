#!/bin/bash

# Target directory defaults to current directory if not provided
TARGET_DIR="${1:-.}"

# Check if the provided directory actually exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

echo "Scanning for .tex files in: $TARGET_DIR"
echo "----------------------------------------"

# Use 'find' to look for all .tex files recursively
find "$TARGET_DIR" -type f -name "*.tex" | while read -r file; do
    echo "Processing: $file"
    tmp_file=$(mktemp)

    # Use awk to process lines while tracking the previous line's content
    awk '
    BEGIN { prev_line = "" }
    
    /\\resizebox.*\\input.*\\/ {
        # Check if the previous line already contains \tikzsetnextfilename
        if (prev_line ~ /\\tikzsetnextfilename/) {
            # Already correct, do nothing
        } else {
            # Extract the filename between the last slash / and the .tex extension
            match($0, /\/[a-zA-Z0-9_-]+\.tex/)
            if (RLENGTH > 0) {
                filename = substr($0, RSTART + 1, RLENGTH - 5)
                
                # Match the indentation of the current line
                match($0, /^[ \t]*/)
                indent = substr($0, RSTART, RLENGTH)
                
                # Print the missing tikzsetnextfilename line
                print indent "\\tikzsetnextfilename{" filename "}"
            }
        }
    }
    
    # Always print the current line and update the tracking variable
    { 
        print $0
        prev_line = $0 
    }
    ' "$file" > "$tmp_file"

    # Overwrite original file
    mv "$tmp_file" "$file"
done

echo "----------------------------------------"
echo "Done! All files in '$TARGET_DIR' have been processed safely."