import cv2
import os
from tifffile import imread, imwrite
from skimage.filters import threshold_otsu
'''
# Path to your images
input_folder = r'F:\2023_4_11_data_organization\1024_patches\DAPI\HE-1\[7591,38493]'
output_folder = r'F:\2023_4_11_data_organization\1024_patches\norm_test'

# Parameters for CLAHE
clip_limit = 3.2
tile_grid_size = (64, 64)

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# For each directory, subdirectory, and file in the root directory
for dirpath, dirnames, filenames in os.walk(input_folder):
    for filename in filenames:
        if filename.endswith(".tif"):
            input_path = os.path.join(dirpath, filename)
            rel_dir = os.path.relpath(dirpath, input_folder)
            output_dir = os.path.join(output_folder, rel_dir)

            # Make sure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, filename)

            # Load image
            img = imread(input_path)

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            img_clahe = clahe.apply(img)

            # Write image
            imwrite(output_path, img_clahe)

print("Processing done!")
'''

import cv2
import os
from tifffile import imread, imwrite

base_root = r'F:\2023_4_11_data_organization\224_patches'
channel_dirs = ['DAPI_unnorm']
output_dir = r'F:\2023_4_11_data_organization\224_patches\DAPI'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Parameters for CLAHE
clip_limit = 2
tile_grid_size = (32, 32)

# Go through each slide subfolder
for slide in ['HE-1', 'HE-2', 'HE-3', 'HE-4', 'HE-5', 'HE-6']:
    # We'll assume that the 'dapi' directory will always contain all files
    dapi_dir = os.path.join(base_root, 'DAPI_unnorm', slide)

    for root, subdir, filenames in os.walk(dapi_dir):
        for filename in filenames:
            if filename.endswith('.tif'):
                input_path = os.path.join(root, filename)
                rel_path = os.path.relpath(root, dapi_dir)  # Relative path of the input file to the base directory

                output_folder = os.path.join(output_dir, slide, rel_path)  # Append relative path to output folder

                # Create output subdirectory if it doesn't exist
                os.makedirs(output_folder, exist_ok=True)

                output_path = os.path.join(output_folder, filename)

                # Load image
                img = imread(input_path)

                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                img_clahe = clahe.apply(img)

                # Write image
                imwrite(output_path, img_clahe)

print("Processing done!")

