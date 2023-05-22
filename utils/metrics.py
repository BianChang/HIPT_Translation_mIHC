import torch
from torchmetrics import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
import numpy as np


def calculate_ssim_per_channel(input_tensor, target_tensor):
    # Check that the input and target tensors have the same number of channels
    assert input_tensor.size(1) == target_tensor.size(
        1), "Input and target tensors must have the same number of channels"

    # Create an instance of StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure(range=2.0, reduction='none')

    # Calculate SSIM for each channel
    ssim_scores = []

    for channel in range(input_tensor.size(1)):
        input_channel = input_tensor[:, channel, :, :].unsqueeze(1)
        target_channel = target_tensor[:, channel, :, :].unsqueeze(1)

        # If the standard deviation of the input channel is zero, add a tiny value to the first pixel
        if input_channel.std() == 0:
            input_channel[0, 0, 0, 0] += 1e-8
        if target_channel.std() == 0:
            target_channel[0, 0, 0, 0] += 1e-8

        ssim_channel = ssim(input_channel, target_channel)
        ssim_channel = ssim_channel.mean()
        ssim_scores.append(ssim_channel.item())  # Convert to Python float right away

    return ssim_scores  # Returns a list of Python floats


def calculate_pearson_corr(input_tensor, target_tensor):
    pearson_corr_scores = []
    psnr_scores = []

    for channel in range(input_tensor.size(1)):
        input_channel = input_tensor[:, channel, :, :].detach().cpu().numpy()
        target_channel = target_tensor[:, channel, :, :].detach().cpu().numpy()

        # If the standard deviation of the input channel is zero, add a tiny value to the first pixel
        if input_channel.std() == 0:
            input_channel[0, 0, 0, 0] += 1e-8
        if target_channel.std() == 0:
            target_channel[0, 0, 0, 0] += 1e-8

        # Flatten the arrays
        input_channel_flat = input_channel.flatten()
        target_channel_flat = target_channel.flatten()

        # Calculate Pearson correlation coefficient
        corr_coef_matrix = np.corrcoef(input_channel_flat, target_channel_flat)

        # np.corrcoef returns a 2D array, the correlation coefficient is at index [0,1] or [1,0]
        corr_coef = corr_coef_matrix[0, 1]

        pearson_corr_scores.append(corr_coef)

        # Calculate psnr
        mse_loss = F.mse_loss(input_channel, target_channel)
        if mse_loss < 1.0e-10:
            psnr = torch.tensor(100.)
        else:
            # if the image is -1 to 1
            max_val = 2.0
            psnr = 20 * torch.log10(max_val / torch.sqrt(mse_loss))

        pearson_corr_scores.append(corr_coef.item())
        psnr_scores.append(psnr.item())

    return pearson_corr_scores, psnr_scores




