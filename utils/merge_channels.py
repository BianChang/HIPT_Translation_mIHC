import os
import cv2
import numpy as np
from tifffile import imread, imwrite
'''
flag = r'HE-1\[11760,39505]'
base_root = r'F:\2023_4_11_data_organization\512_patches\all_slide'
DAPI_DIR = os.path.join(base_root, 'dapi', flag)
CD3_DIR = os.path.join(base_root, 'cd3', flag)
CD20_DIR = os.path.join(base_root, 'cd20', flag)
PANCK_DIR = os.path.join(base_root, 'panck', flag)
'''
'''
DAPI_DIR = os.path.join(r'F:\2023_4_11_data_organization\512_patches\all_slide\dapi')
CD3_DIR = os.path.join(r'F:\2023_4_11_data_organization\512_patches\all_slide\cd3')
CD20_DIR = os.path.join(r'F:\2023_4_11_data_organization\512_patches\all_slide\cd20')
PANCK_DIR = os.path.join(r'F:\2023_4_11_data_organization\512_patches\all_slide\panck')

output_dir = r'F:\2023_4_11_data_organization\512_patches\all_slide\merged'

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)



for filename in os.listdir(DAPI_DIR):
    # print(filename)
    srcsuffix = '.tif'
    nosurffixname = os.path.splitext(filename)[0]
    DAPI = cv2.imread(os.path.join(DAPI_DIR, filename), cv2.IMREAD_GRAYSCALE)
    CD3 = cv2.imread(os.path.join(CD3_DIR, filename), cv2.IMREAD_GRAYSCALE)
    CD20 = cv2.imread(os.path.join(CD20_DIR, filename), cv2.IMREAD_GRAYSCALE)
    PANCK = cv2.imread(os.path.join(PANCK_DIR, filename), cv2.IMREAD_GRAYSCALE)

    MIHC_patch = np.zeros((DAPI.shape[0], DAPI.shape[1], 4), dtype=np.uint8)
    MIHC_patch[:, :, 0] = DAPI
    MIHC_patch[:, :, 1] = CD3
    MIHC_patch[:, :, 2] = CD20
    MIHC_patch[:, :, 3] = PANCK

    imwrite(os.path.join(output_dir, nosurffixname + srcsuffix), MIHC_patch)
'''

base_root = r'F:\2023_4_11_data_organization\224_patches'
channel_dirs = ['DAPI', 'CD3', 'CD20', 'PANCK']
output_dir = r'F:\2023_4_11_data_organization\224_patches\merged_channels'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Go through each slide subfolder
for slide in ['HE-1', 'HE-2', 'HE-3', 'HE-4', 'HE-5', 'HE-6']:
    # Create output slide directory
    os.makedirs(os.path.join(output_dir, slide), exist_ok=True)

    # We'll assume that the 'dapi' directory will always contain all files
    dapi_dir = os.path.join(base_root, 'DAPI', slide)

    for root, _, filenames in os.walk(dapi_dir):
        for filename in filenames:
            if filename.endswith('.tif'):
                MIHC_patch = np.zeros((*imread(os.path.join(root, filename)).shape, 4), dtype=np.uint8)

                # Go through each channel and add it to the patch
                for i, dir in enumerate(channel_dirs):
                    # Replace base path by dir in order to navigate in each channel directory
                    current_channel_root = root.replace('DAPI', dir)
                    img = imread(os.path.join(current_channel_root, filename))
                    MIHC_patch[:, :, i] = img

                # Save the combined image
                imwrite(os.path.join(output_dir, slide, filename), MIHC_patch)


