import os
import cv2
from tifffile import imread, imwrite


def crop_images_in_directory(input_dir, output_dir, patch_size, stride):
    """
    Crop images in a directory into patches and save them into output directory.

    Args:
    input_dir (str): The input directory containing images to crop.
    output_dir (str): The output directory to save patches.
    patch_size (int): The size of each patch.
    stride (int): The stride between adjacent patches.
    """

    # Loop over all subdirectories in the input directory
    for subdir, _, files in os.walk(input_dir):
        # Create a corresponding subdirectory in the output directory
        rel_path = os.path.relpath(subdir, input_dir)
        out_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(out_subdir, exist_ok=True)

        # Loop over all images in the current subdirectory
        for filename in files:
            if filename.endswith(".tif") or filename.endswith(".jpg"):
                # Read the image
                img_path = os.path.join(subdir, filename)
                # for HE images
                # img = cv2.imread(img_path)
                # for marker images in tif format
                img = imread(img_path)

                # Determine the number of patches to crop
                # for 3 channel HE images
                # height, width, channels = img.shape
                # for markers
                height, width = img.shape
                num_rows = (height - patch_size) // stride + 1
                num_cols = (width - patch_size) // stride + 1

                # Crop patches from the image
                for i in range(num_rows):
                    for j in range(num_cols):
                        x = i * stride
                        y = j * stride
                        # for HE images
                        # patch = img[x:x + patch_size, y:y + patch_size, :]
                        # for markers
                        patch = img[x:x + patch_size, y:y + patch_size]
                        # Determine output filename and directory
                        patch_dir = os.path.join(out_subdir, filename.split('.')[0])
                        os.makedirs(patch_dir, exist_ok=True)
                        patch_filename = f"{filename.split('.')[0]}_patch_{i}_{j}.tif"
                        patch_path = os.path.join(patch_dir, patch_filename)

                        # Save the patch to disk
                        # cv2.imwrite(patch_path, patch)
                        imwrite(patch_path, patch)


input_dir = r'F:\2023_4_11_data_organization\PANCK_block\grey'
output_dir = r'F:\2023_4_11_data_organization\1024_patches\PANCK'
patch_size = 1024
stride = 512
crop_images_in_directory(input_dir, output_dir, patch_size, stride)
