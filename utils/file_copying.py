import os
import shutil
from tqdm import tqdm
import concurrent.futures
import cv2
import glob
import re
import numpy as np


def copy_images(src_dir, dest_dir):
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Prepare to gather all image files
    image_files = []

    for subdir, dirs, files in os.walk(src_dir):
        for file in files:
            # Check if the file is an image (You can add more image extensions here)
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith(
                    '.tif') or file.endswith('.tiff'):
                image_files.append(os.path.join(subdir, file))

    # Copy files with a progress bar
    for file in tqdm(image_files, desc="Copying images", unit="file"):
        shutil.copy(file, dest_dir)


def copy_matching_files(dir_A, dir_B, dir_C):
    """
    For each file in dir_A, if a file with the same name exists in dir_B, it makes
    a copy of the file from dir_B and saves it in dir_C.
    This function will also search all subdirectories in dir_B.
    """
    os.makedirs(dir_C, exist_ok=True)

    # This will include all files in dir_B and its subdirectories
    files_in_B = set()
    for dirpath, dirnames, filenames in os.walk(dir_B):
        for file in filenames:
            files_in_B.add(file)

    for dirpath, dirnames, filenames in os.walk(dir_A):
        for filename in filenames:
            if filename in files_in_B:
                matching_file_path = [os.path.join(dirpath, file) for dirpath, _, files in os.walk(dir_B) for file in files if file == filename][0]
                shutil.copy(matching_file_path, dir_C)
            else:
                print(filename)


def count_files_in_directory(directory):
    return sum([len(files) for r, d, files in os.walk(directory)])


def copy_file(file, dest_dir):
    shutil.copy(file, dest_dir)

def copy_images_faster(src_dir, dest_dir):
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Prepare to gather all image files
    image_files = []

    for subdir, dirs, files in os.walk(src_dir):
        for file in files:
            # Check if the file is an image (You can add more image extensions here)
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.tif') or file.endswith('.tiff'):
                image_files.append(os.path.join(subdir, file))

    # Copy files with a progress bar
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for file in tqdm(image_files, desc="Copying images", unit="file"):
            executor.submit(copy_file, file, dest_dir)


def save_images_as_grayscale(input_dir, output_dir):
    """
    Finds images with 'depth-0', 'depth-1', or 'depth-2' in the filename, converts these to grayscale,
    and saves them to a new path.

    Args:
    input_dir (str): The directory to search for images.
    output_dir (str): The directory where grayscale images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    pattern = re.compile(r'.*depth-([0-2])(?!\d)')  # Regex to match the required filenames

    for img_file in glob.glob(os.path.join(input_dir, '*')):
        if pattern.match(img_file):
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                filename = os.path.basename(img_file)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, img)
                print(f"Grayscale image saved: {output_path}")


def concatenate_images_by_depth(input_dir, output_dir):
    """
    Concatenates images with similar names but differing by 'depth-[0-2]' into a single image with 3 channels.
    The output filenames replace only the 'depth-[0-2]' part with 'depth_computation', keeping the rest unchanged.

    Args:
    input_dir (str): Directory containing grayscale images to concatenate.
    output_dir (str): Directory to save concatenated images.
    """
    os.makedirs(output_dir, exist_ok=True)
    images = {}
    # Pattern to identify the depth part and capture the prefix and suffix for reconstruction
    pattern = re.compile(r'(.*z_depth-)([0-2])(.*)')

    for img_file in os.listdir(input_dir):
        match = pattern.match(img_file)
        if match:
            prefix, depth, suffix = match.groups()
            # Unique identifier without the depth number, keeping prefix and suffix for reconstruction
            identifier = prefix[:-1] + suffix  # Removes the trailing '-' from 'z_depth-'
            depth = int(depth)
            if identifier not in images:
                images[identifier] = [None, None, None]
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images[identifier][depth] = img

    for identifier, depths in images.items():
        if all(depth is not None for depth in depths):
            concatenated_img = cv2.merge(depths)
            # Construct the output filename by replacing 'z_depth-[0-2]' with 'depth_computation'
            output_filename = identifier.replace('z_depth', 'depth_computation')
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, concatenated_img)
            print(f"Concatenated image saved: {output_path}")

def main():

    # Use the functions

    #src_dir = r'F:\2023_4_11_data_organization\1024_patches\PD-L1'
    #dest_dir = r'F:\2023_4_11_data_organization\1024_patches\train\PD-L1'
    #copy_images(src_dir, dest_dir)
    # copy_images_faster(src_dir, dest_dir)


    #dir_A = r'F:\2023_4_11_data_organization\224_patches\224_small_dataset\test\input'
    #dir_B = r'F:\2023_4_11_data_organization\224_patches\merged_channels'
    #dir_C = r'F:\2023_4_11_data_organization\224_patches\224_small_dataset\test\label'
    #copy_matching_files(dir_A, dir_B, dir_C)

    # Usage
    #directory = r"F:\2023_4_11_data_organization\224_patches\val\input"
    #print(f"There are {count_files_in_directory(directory)} files in {directory}")

    # Example usage
    input_dir = r'D:\Chang_files\workspace\data\InSillico\Finkbeiner\test\greyscale_depths0-2\yusha_0_1'
    output_dir = r'D:\Chang_files\workspace\data\InSillico\Finkbeiner\test\3-channel_merged\yusha_0_1'
    concatenate_images_by_depth(input_dir, output_dir)
    # save_images_as_grayscale(input_dir, output_dir)

# Run the main function
if __name__ == "__main__":
    main()
