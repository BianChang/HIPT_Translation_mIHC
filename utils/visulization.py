import os
import cv2
import numpy as np
from tifffile import imread, imwrite


def visualize_4channel_tif(input_dir, output_dir):
    """
    Visualize 4-channel tif images where each channel is shown in a specific color.

    Args:
        input_dir (str): The input directory containing 4-channel tif images.
        output_dir (str): The output directory to save the visualizations.
    """

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.tif'):
            continue

        # Read the 4-channel tif image
        img_path = os.path.join(input_dir, filename)
        img = imread(img_path)

        # Split channels
        dapi = img[:, :, 0]
        cd3 = img[:, :, 1]
        cd20 = img[:, :, 2]
        panck = img[:, :, 3]

        # Create a 4-color visualization where each channel is shown in a specific color
        visualization = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        visualization[:, :, 0] = dapi  # Blue channel
        visualization[:, :, 1] = cd3 + cd20  # Green channel
        visualization[:, :, 2] = panck +cd20  # Red channel

        # Save the visualization
        vis_path = os.path.join(output_dir, filename[:-4] + '_vis.jpg')
        cv2.imwrite(vis_path, visualization)


def normalize_4channel_tif(input_dir, output_dir):
    """
    Normalize 4-channel tif images so all channels have the same distribution.

    Args:
        input_dir (str): The input directory containing 4-channel tif images.
        output_dir (str): The output directory to save the normalized images.
    """

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.tif'):
            continue

        # Read the 4-channel tif image
        img_path = os.path.join(input_dir, filename)
        img = imread(img_path)

        # Normalize each channel individually
        for i in range(img.shape[-1]):
            img[..., i] = cv2.normalize(img[..., i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Save the normalized image
        norm_img_path = os.path.join(output_dir, filename[:-4] + '_normalized.tif')
        cv2.imwrite(norm_img_path, img)


def main():
    input_dir = r'F:\2023_4_11_data_organization\224_patches\merged\train\label'
    output_dir = r'F:\2023_4_11_data_organization\224_patches\merged\vis\train'
    visualize_4channel_tif(input_dir, output_dir)
    # normalize_4channel_tif(input_dir, output_dir)


if __name__ == '__main__':
    main()
