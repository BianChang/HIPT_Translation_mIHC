import numpy as np
import os
from PIL import Image

# Define the paths to your dataset and label directories
data_dir = r'F:\2023_4_11_data_organization\224_patches\merged\train\input'
label_dir = r'F:\2023_4_11_data_organization\224_patches\merged\train\label'

# Initialize the accumulator variables for mean and variance
mean = np.zeros(3)
variance = np.zeros(3)

# Iterate over the images in the dataset directory
for filename in os.listdir(data_dir):
    #print(filename)
    # Load the image using PIL and convert to numpy array
    img = np.array(Image.open(os.path.join(data_dir, filename)).convert('RGB'))

    # Accumulate the pixel values for mean and variance calculation
    mean += np.mean(img, axis=(0,1))
    variance += np.var(img, axis=(0,1))

# Divide the accumulator variables by the number of images to get the mean and variance
mean /= len(os.listdir(data_dir))
variance /= len(os.listdir(data_dir))

# Calculate the standard deviation from the variance
std_dev = np.sqrt(variance)

# Print the mean and standard deviation
print("RGB Mean: ", mean)
print("RGB Standard Deviation: ", std_dev)

# Repeat the above process for the label dataset
# Initialize the accumulator variables for mean and variance
mean = np.zeros(4)
variance = np.zeros(4)

# Iterate over the images in the label directory
for filename in os.listdir(label_dir):
    # Load the label image using PIL and convert to numpy array
    #print(filename)
    label = np.array(Image.open(os.path.join(label_dir, filename)))

    # Accumulate the pixel values for mean and variance calculation
    mean += np.mean(label, axis=(0,1))
    variance += np.var(label, axis=(0,1))

# Divide the accumulator variables by the number of images to get the mean and variance
mean /= len(os.listdir(label_dir))
variance /= len(os.listdir(label_dir))

# Calculate the standard deviation from the variance
std_dev = np.sqrt(variance)

# Print the mean and standard deviation
print("Label Mean: ", mean)
print("Label Standard Deviation: ", std_dev)
