import torch
from torchmetrics import StructuralSimilarityIndexMeasure
import torch.nn.functional as F


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
        ssim_channel = ssim(input_channel, target_channel)
        ssim_channel = ssim_channel.mean()
        ssim_scores.append(ssim_channel.item())  # Convert to Python float right away

    return ssim_scores  # Returns a list of Python floats


def calculate_pearson_corr(input_tensor, target_tensor):
    # Calculate mean squared error (MSE) and mean absolute error (MAE) between the output and target tensors
    mse_loss = F.mse_loss(input_tensor, target_tensor)
    mae_loss = F.l1_loss(input_tensor, target_tensor)

    # Calculate Pearson correlation coefficient
    corr_coef = 1.0 - (2.0 * mae_loss) / (mse_loss + torch.mean(target_tensor ** 2))

    # Calculate psnr
    max_val = 1.0
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse_loss))

    return corr_coef, psnr




