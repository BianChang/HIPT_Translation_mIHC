import os
import shutil
'''
This code will search through every file in dir_B, if there is a file with the same name exists in dir_A, it will make 
a copy of the file in dir_B and save it in dir_C
'''
# Define the directories
dir_A = r'F:\2023_4_11_data_organization\512_patches\train\input'
dir_B = r'F:\2023_4_11_data_organization\512_patches\all_slide\merged'
dir_C = r'F:\2023_4_11_data_organization\512_patches\train\label'

# Get list of file names in directory B (not including the path)
files_in_B = set(os.listdir(dir_B))

# Go through all files in directory A
for filename in os.listdir(dir_A):
    # If a file with the same name is in directory B, copy it to directory C
    if filename in files_in_B:
        shutil.copy(os.path.join(dir_B, filename), dir_C)
    else:
        print(filename)

