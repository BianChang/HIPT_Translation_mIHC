import os
import shutil

# Define the directories
dir_A = r'F:\2023_4_11_data_organization\512_patches\train\input'
dir_B = r'F:\2023_4_11_data_organization\512_patches\all_slide\merged'
dir_C = r'F:\2023_4_11_data_organization\512_patches\train\label'

# Get list of file names in directory A (not including the path)
files_in_A = set(os.listdir(dir_A))

# Go through all files in directory B
for filename in os.listdir(dir_B):
    # If a file with the same name is in directory A, copy it to directory C
    if filename in files_in_A:
        shutil.copy(os.path.join(dir_B, filename), dir_C)
