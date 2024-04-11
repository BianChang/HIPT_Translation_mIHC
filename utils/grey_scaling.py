from PIL import Image
import os

# Specify the root directory containing the subfolders
root_dir = r'F:\2023_4_11_data_organization\PD-L1_block'

# Loop over the subfolders in the root directory
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)

    # Create a new subdirectory to save the grayscale images
    gray_dir = os.path.join(root_dir, 'grey', subdir)
    os.makedirs(gray_dir, exist_ok=True)

    # Loop over the RGB images in the current subdirectory
    for filename in os.listdir(subdir_path):
        if filename.endswith('.tif') or filename.endswith('.jpg'):
            # Extract the number from the original filename
            number = filename.split('[')[-1].split(']')[0]

            # Open the RGB image and convert it to grayscale
            img_path = os.path.join(subdir_path, filename)
            img = Image.open(img_path).convert('L')

            # Save the grayscale image in the new directory with the number as the filename
            gray_filename = f'[{number}].tif'
            gray_path = os.path.join(gray_dir, gray_filename)
            img.save(gray_path)

            print(f'Saved {gray_path}')
