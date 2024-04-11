import os
import re
import shutil

# Replace this with the path to the directory containing your txt files
directory_path = r'D:\Chang_files\workspace\Qupath_proj\hemit_unet\unet'

# Regex pattern to match the image block numbers in the filename
pattern = re.compile(r'\[\d+,\d+\]')

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith("_real_B.tif Detections.txt") or filename.endswith("_fake_B.tif Detections.txt"):
        # Extract the image block identifier using the regex pattern
        match = pattern.search(filename)
        if match:
            image_block = match.group()

            # Create a subdirectory for the image block if it doesn't exist
            subdirectory = os.path.join(directory_path, image_block)
            if not os.path.exists(subdirectory):
                os.makedirs(subdirectory)

            # Copy the file into its image block subdirectory
            original_file_path = os.path.join(directory_path, filename)
            new_file_path = os.path.join(subdirectory, filename)
            shutil.copy(original_file_path, new_file_path)

print("Files have been copied into subfolders based on image blocks.")
