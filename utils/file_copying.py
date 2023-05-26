import os
import shutil
from tqdm import tqdm


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


def main():
    # Use the functions
    src_dir = r'F:\2023_4_11_data_organization\224_patches\HE'
    dest_dir = r'F:\2023_4_11_data_organization\224_patches\merged\HE'
    copy_images(src_dir, dest_dir)
    '''
    dir_A = '/path/to/A/directory'
    dir_B = '/path/to/B/directory'
    dir_C = '/path/to/C/directory'
    copy_matching_files(dir_A, dir_B, dir_C)
    '''
# Run the main function
if __name__ == "__main__":
    main()
