import torch
from SwinVisionTranformer import SwinTransformer

import timm

# Load the pre-trained model
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)

# Print the names and structure of the layers
for name, param in model.named_parameters():
    print(name)

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