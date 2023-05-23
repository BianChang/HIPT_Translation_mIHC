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
