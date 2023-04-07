import os
import cv2


def crop_images_in_directory(input_dir, output_dir, patch_size, stride):
    """
    Crop images in a directory into patches and save them into output directory.

    Args:
    input_dir (str): The input directory containing images to crop.
    output_dir (str): The output directory to save patches.
    patch_size (int): The size of each patch.
    stride (int): The stride between adjacent patches.
    """

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Read the image
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            # Determine the number of patches to crop
            height, width, channels = img.shape
            num_rows = (height - patch_size) // stride + 1
            num_cols = (width - patch_size) // stride + 1

            # Crop patches from the image
            for i in range(num_rows):
                for j in range(num_cols):
                    x = i * stride
                    y = j * stride
                    patch = img[x:x + patch_size, y:y + patch_size, :]

                    # Determine output filename and directory
                    patch_dir = os.path.join(output_dir, filename.split('.')[0])
                    os.makedirs(patch_dir, exist_ok=True)
                    patch_filename = f"{filename.split('.')[0]}_patch_{i}_{j}.jpg"
                    patch_path = os.path.join(patch_dir, patch_filename)

                    # Save the patch to disk
                    cv2.imwrite(patch_path, patch)


input_dir = r'D:\Chang_files\workspace\data\MIHC\PANCK_reg/HE-6'
output_dir = r'D:\Chang_files\workspace\data\mihc_patches/PANCK/HE-6'
patch_size = 1024
stride = 512
crop_images_in_directory(input_dir, output_dir, patch_size, stride)
