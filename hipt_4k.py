### Dependencies
# Base Dependencies
import os
import pickle
import sys

# LinAlg / Stats / Plotting Dependencies
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

# Torch Dependencies
import torch
import torch.multiprocessing
import torchvision
from torchvision import transforms
from einops import rearrange, repeat
torch.multiprocessing.set_sharing_strategy('file_system')

class HIPT_4K(nn.Module):
    def __init__(self, new_vit, new_vit4k):
        super().__init__()
        self.new_vit = new_vit
        self.new_vit4k = new_vit4k

    def prepare_img_tensor(self, img: torch.Tensor, patch_size=256):
        """
        Helper function that takes a non-square image tensor, and takes a center crop s.t. the width / height
        are divisible by 256.

        (Note: "_256" for w / h is should technically be renamed as "_ps", but may not be easier to read.
        Until I need to make HIPT with patch_sizes != 256, keeping the naming convention as-is.)

        Args:
            - img (torch.Tensor): [1 x C x W' x H'] image tensor.
            - patch_size (int): Desired patch size to evenly subdivide the image.

        Return:
            - img_new (torch.Tensor): [1 x C x W x H] image tensor, where W and H are divisble by patch_size.
            - w_256 (int): # of [256 x 256] patches of img_new's width (e.g. - W/256)
            - h_256 (int): # of [256 x 256] patches of img_new's height (e.g. - H/256)
        """
        make_divisble = lambda l, patch_size: (l - (l % patch_size))
        b, c, w, h = img.shape
        load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
        w_256, h_256 = w // patch_size, h // patch_size
        img_new = transforms.CenterCrop(load_size)(img)
        return img_new, w_256, h_256

    def forward(self, x):
        batch_256, w_256, h_256 = self.prepare_img_tensor(x)
        batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)
        batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')

        features_output_new_vit = []
        for mini_bs in range(0, batch_256.shape[0], 256):
            minibatch_256 = batch_256[mini_bs:mini_bs + 256]
            features_output_new_vit.append(self.new_vit(minibatch_256))

        features_output_new_vit = torch.cat(features_output_new_vit, dim=0)
        features_output_new_vit = features_output_new_vit.reshape(w_256, h_256, 3, 256, 256).transpose(0, 1).transpose(
            0, 2).unsqueeze(dim=0)

        output_new_vit4k = self.new_vit4k(features_output_new_vit)
        return output_new_vit4k