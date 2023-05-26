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

import os
import cv2
import numpy as np
from tifffile import imread, imwrite

base_root = r'F:\2023_4_11_data_organization\224_patches'
output_root = r'F:\2023_4_11_data_organization\224_patches\merged\merged_channels'

# The channels and their respective directories
channels = {
    "dapi": os.path.join(base_root, 'dapi'),
    "cd3": os.path.join(base_root, 'cd3'),
    "cd20": os.path.join(base_root, 'cd20'),
    "panck": os.path.join(base_root, 'panck'),
}

# The subdirectories (slides)
slides = [f"HE-{i}" for i in range(1, 7)]

for slide in slides:
    # Create output directory for each slide
    output_dir = os.path.join(output_root, slide)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each slide
    for root, dirs, files in os.walk(os.path.join(channels["dapi"], slide)):
        for filename in files:
            if filename.endswith('.tif'):  # Check for the image file extension
                srcsuffix = '.tif'
                nosurffixname = os.path.splitext(filename)[0]

                MIHC_patch = np.zeros((224, 224, 4), dtype=np.uint8)
                for i, (channel, dir) in enumerate(channels.items()):
                    # replace root path by dir in order to navigate in each channel directory
                    img_path = os.path.join(dir, slide, os.path.relpath(root, os.path.join(dir, slide)), filename)
                    img = imread(img_path)
                    MIHC_patch[:, :, i] = img

                imwrite(os.path.join(output_dir, nosurffixname + srcsuffix), MIHC_patch)

