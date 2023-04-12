import os
import cv2
import numpy as np

DAPI_DIR = r'D:\work\project1\workspace\2023_4_11_data_organization\Patches\all_slide\DAPI'
CD3_DIR = r'D:\work\project1\workspace\2023_4_11_data_organization\Patches\all_slide\CD3'
CD20_DIR = r'D:\work\project1\workspace\2023_4_11_data_organization\Patches\all_slide\CD20'
PANCK_DIR = r'D:\work\project1\workspace\2023_4_11_data_organization\Patches\all_slide\PANCK'

output_dir = r'D:\work\project1\workspace\2023_4_11_data_organization\Patches\MERGED'

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)



for filename in os.listdir(DAPI_DIR):
    print(filename)
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
    cv2.imwrite(os.path.join(output_dir, nosurffixname + srcsuffix), MIHC_patch)
