import os
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_otsu


def preprocess_channel(channel):
    #threshold = threshold_otsu(channel)
    #print(threshold)
    channel_preprocessed = channel.copy()
    channel_preprocessed[channel_preprocessed < 25] = 0
    return channel_preprocessed

def remove_channel(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    file_list = os.listdir(input_folder)

    # Process each file in the input folder
    for i, file_name in enumerate(file_list):
        # Check if the file is a TIFF image
        if file_name.endswith('.tif') or file_name.endswith('.tiff'):
            # Construct the input and output file paths
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Read the image using tifffile
            image = tiff.imread(input_path)

            # Remove the third channel and keep channels 1, 2, and 4
            image = image[:, :, [3, 1, 0]]

            # Preprocess channel 2 using Otsu's method
            channel_2 = image[:, :, 1]
            channel_2_preprocessed = preprocess_channel(channel_2)

            # Update the image with the preprocessed channel 2
            image[:, :, 1] = channel_2_preprocessed

            # Save the modified image
            tiff.imsave(output_path, image)

            print(f"Processed: {file_name}")

            # Plot the 1st, 2nd, and 4th channels of the first 5 files
            if i < 5:
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(image[:, :, 0], cmap='gray')
                plt.title('panck')
                plt.subplot(1, 3, 2)
                plt.imshow(image[:, :, 1], cmap='gray')
                plt.title('cd3')
                plt.subplot(1, 3, 3)
                plt.imshow(image[:, :, 2], cmap='gray')
                plt.title('dapi')
                plt.tight_layout()
                plt.show()

# Specify the input and output folder paths
input_folder = r'F:\2023_4_11_data_organization\1024_patches\train\label'
output_folder = r'F:\2023_4_11_data_organization\1024_patches\train\label_3'

# Call the function to remove the third channel, preprocess channel 2, and plot channels 1, 2, and 4
remove_channel(input_folder, output_folder)
