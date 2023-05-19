import os
import cv2
import numpy as np


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
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Convert the image values back to [0, 1] from [-1, 1]
        # img = (img + 1.0) / 2.0

        # Convert the image values back to [0, 255] from [0, 1]
        # img = (img * 255).astype(np.uint8)

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


def main():
    input_dir = r'F:\2023_4_11_data_organization\224_patches\merged\test\label'
    output_dir = r'F:\2023_4_11_data_organization\224_patches\merged\vis\test'
    visualize_4channel_tif(input_dir, output_dir)


if __name__ == '__main__':
    main()
