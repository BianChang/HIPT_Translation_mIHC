import torch
from torchmetrics import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
import torch.nn as nn



def calculate_ssim_per_channel(input_tensor, target_tensor):
    # Check that the input and target tensors have the same number of channels
    assert input_tensor.size(1) == target_tensor.size(
        1), "Input and target tensors must have the same number of channels"

    # Create an instance of StructuralSimilarityIndexMeasure
    # ssim = StructuralSimilarityIndexMeasure(range=1.0, reduction='none')

    # Calculate SSIM for each channel
    ssim_scores = []

    for channel in range(input_tensor.size(1)):
        input_channel = input_tensor[:, channel, :, :].squeeze()
        target_channel = target_tensor[:, channel, :, :].squeeze()

        input_channel = (input_channel + 1.0) / 2.0
        target_channel = (target_channel + 1.0) / 2.0

        input_channel  = (input_channel .cpu().detach().numpy() * 255).astype(np.uint8)
        target_channel = (target_channel.cpu().detach().numpy() * 255).astype(np.uint8)

        # If the standard deviation of the input channel is zero, add a tiny value to the first pixel
        if input_channel.std() == 0:
            input_channel[0][0] += 1
        if target_channel.std() == 0:
            target_channel[0][0] += 1

        # if batch size is not 1, transpose channels
        if input_channel.ndim != 2:
            input_channel = np.transpose(input_channel, (1, 2, 0))
            target_channel = np.transpose(target_channel, (1, 2, 0))

        ssim_channel = ssim(input_channel, target_channel, data_range=255, multichannel=True)
        # ssim_channel = ssim_channel.mean()
        ssim_scores.append(ssim_channel)

    return ssim_scores  # Returns a list of Python floats


def calculate_pearson_corr(input_tensor, target_tensor):
    pearson_corr_scores = []
    psnr_scores = []

    for channel in range(input_tensor.size(1)):
        input_channel = input_tensor[:, channel, :, :].squeeze()
        target_channel = target_tensor[:, channel, :, :].squeeze()

        input_channel_tensor = input_tensor[:, channel, :, :].unsqueeze(1)
        target_channel_tensor = target_tensor[:, channel, :, :].unsqueeze(1)

        input_channel = (input_channel + 1.0) / 2.0
        target_channel = (target_channel + 1.0) / 2.0

        input_channel = (input_channel.cpu().detach().numpy() * 255).astype(np.uint8)
        target_channel = (target_channel.cpu().detach().numpy() * 255).astype(np.uint8)

        # If the standard deviation of the input channel is zero, add a tiny value to the first pixel
        if input_channel.std() == 0:
            input_channel[0][0] += 1
        if target_channel.std() == 0:
            target_channel[0][0] += 1

        # Flatten the arrays
        input_channel_flat = input_channel.flatten()
        target_channel_flat = target_channel.flatten()

        # Calculate Pearson correlation coefficient
        corr_coef_matrix = np.corrcoef(input_channel_flat, target_channel_flat)

        # np.corrcoef returns a 2D array, the correlation coefficient is at index [0,1] or [1,0]
        corr_coef = corr_coef_matrix[0, 1]
        pearson_corr_scores.append(corr_coef)

        # Calculate psnr
        '''
        mse_loss = F.mse_loss(input_channel_tensor, target_channel_tensor)
        if mse_loss < 1.0e-10:
            psnr = torch.tensor(100.)
        else:
            # if the image is -1 to 1
            max_val = 2.0
            psnr = 20 * torch.log10(max_val / torch.sqrt(mse_loss))
            
        psnr_scores.append(psnr.item())
        '''
        psnr = peak_signal_noise_ratio(input_channel_tensor, target_channel_tensor)

        psnr_scores.append(psnr)

    return pearson_corr_scores, psnr_scores


class ChannelWeightedFocalLoss(nn.Module):
    def __init__(self, weights, gamma=2.0):
        super(ChannelWeightedFocalLoss, self).__init__()
        self.weights = weights
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Normalize the inputs and targets to [0, 1]
        inputs = (inputs + 1) / 2
        targets = (targets + 1) / 2

        # Compute the standard L1Loss
        L1_loss = F.l1_loss(inputs, targets, reduction='none')
        # Convert the L1Loss to 'pseudo probabilities', in [0,1]
        pt = torch.exp(-L1_loss)

        # Compute the Focal Loss
        weights = self.weights.view(1, -1, 1, 1)
        F_loss = weights * (1-pt)**self.gamma * L1_loss

        return F_loss.mean()




