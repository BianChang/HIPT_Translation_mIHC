import cv2
import os
from tifffile import imread, imwrite
from skimage.filters import threshold_otsu

# Path to your images
input_folder = r'F:\2023_4_11_data_organization\1024_patches\DAPI\HE-1\[7591,38493]'
output_folder = r'F:\2023_4_11_data_organization\1024_patches\norm_test'

# Parameters for CLAHE
clip_limit = 3.2
tile_grid_size = (64, 64)

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# For each file in the directory
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load image
        img = imread(input_path)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_clahe = clahe.apply(img)

        # Write image
        imwrite(output_path, img)
        imwrite(output_path.replace('.tif', '_clahe.tif'), img_clahe)

print("Processing done!")
