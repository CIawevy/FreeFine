#!/bin/bash

# Define the two directories to compare
DIR1="/work/nvme/bcgq/yimingg8/geobench/coarse_img"
DIR2="/work/nvme/bcgq/yimingg8/geodiffuser_output_2D"

# Check if directories exist
if [ ! -d "$DIR1" ]; then
    echo "Error: Directory $DIR1 does not exist."
    exit 1
fi

if [ ! -d "$DIR2" ]; then
    echo "Error: Directory $DIR2 does not exist."
    exit 1
fi

echo "Comparing subdirectory names in '$DIR1' and '$DIR2'..."
echo

# Create temporary files to store sorted subdirectory names
TMP_FILE1=$(mktemp)
TMP_FILE2=$(mktemp)

# List subdirectories (names only, without path), sort them, and store in temp files
# Using find -mindepth 1 -maxdepth 1 -type d -printf "%f\n" to get only the directory names
find "$DIR1" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort > "$TMP_FILE1"
find "$DIR2" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort > "$TMP_FILE2"

# Use comm to find differences
# comm -1 prints lines unique to file2
# comm -2 prints lines unique to file1
# comm -3 prints lines common to both (suppressed here)

echo "Subdirectories unique to '$DIR1':"
comm -23 "$TMP_FILE1" "$TMP_FILE2"
echo

echo "Subdirectories unique to '$DIR2':"
comm -13 "$TMP_FILE1" "$TMP_FILE2"
echo

# If you want to see common subdirectories, you can use:
# echo "Common subdirectories:"
# comm -12 "$TMP_FILE1" "$TMP_FILE2"
# echo

# Clean up temporary files
rm "$TMP_FILE1" "$TMP_FILE2"

echo "Comparison finished."
