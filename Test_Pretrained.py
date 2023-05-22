import torch
from SwinVisionTranformer import SwinTransformer
'''
import timm

# Load the pre-trained model
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)

# Print the names and structure of the layers
for name, param in model.named_parameters():
    print(name)
'''
'''
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
from huggingface_hub import hf_hub_download


model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")

for name, param in model.named_parameters():
    print(name)
'''


#swin_t = SwinTransformer()

# match the layers correctly here
'''
for name, param in swin_t.named_parameters():
    print(name)
    if name in pretrained_model.state_dict():
        print(name, ' exist')
        param.data = pretrained_model.state_dict()[name].data
    else:
        print(name, ' does not exist')
'''

# Freeze the encoder's weights
# for param in swin_t.parameters():
    # param.requires_grad = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
from dataset.ImageToImageDataset import ImageToImageDataset

def check_data(loader):
    for images, labels in loader:
        # get the first channel of labels
        first_channel_labels = labels[:, 0, :, :]
        print( "Labels Shape:", first_channel_labels.shape)
        print("DAPI contain NaNs: ", torch.isnan(first_channel_labels).any().item())
        print("DAPI contain Infs: ", torch.isinf(first_channel_labels).any().item())

        '''
        print("Images contain NaNs: ", torch.isnan(images).any().item())
        print("Images contain Infs: ", torch.isinf(images).any().item())
        print("Labels contain NaNs: ", torch.isnan(labels).any().item())
        print("Labels contain Infs: ", torch.isinf(labels).any().item())
        '''

def main():
    mean_data = [0.5, 0.5, 0.5]
    std_data = [0.5, 0.5, 0.5]
    mean_label = [0.5, 0.5, 0.5, 0.5]
    std_label = [0.5, 0.5, 0.5, 0.5]

    input_transform = Compose([
        ToTensor(),
        Normalize(mean=mean_data, std=std_data)
    ])
    label_transform = Compose([
        ToTensor(),
        Normalize(mean=mean_label, std=std_label)
    ])

    # Create instances of the ImageToImageDataset for the training, validation, and test sets
    train_dataset = ImageToImageDataset(r'F:\2023_4_11_data_organization\224_patches\merged\small_dataset\train',
                                        input_transform=input_transform, label_transform=label_transform)
    val_dataset = ImageToImageDataset(r'F:\2023_4_11_data_organization\224_patches\merged\small_dataset\val',
                                      input_transform=input_transform, label_transform=label_transform)
    test_dataset = ImageToImageDataset(r'F:\2023_4_11_data_organization\224_patches\merged\small_dataset\test',
                                       input_transform=input_transform, label_transform=label_transform)

    # Create instances of the DataLoader for the training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    print("Checking train loader...")
    check_data(train_loader)
    print("Checking val loader...")
    check_data(val_loader)
    print("Checking test loader...")
    check_data(test_loader)

if __name__ == '__main__':
    main()
