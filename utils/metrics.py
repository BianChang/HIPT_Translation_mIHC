import torch
from torchmetrics import StructuralSimilarityIndexMeasure


def calculate_ssim_per_channel(input_tensor, target_tensor):
    input_tensor = input_tensor.cpu()
    target_tensor = target_tensor.cpu()
    # Check that the input and target tensors have the same number of channels
    assert input_tensor.size(1) == target_tensor.size(
        1), "Input and target tensors must have the same number of channels"

    # Create an instance of StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure(range=1.0, reduction='none')

    # Calculate SSIM for each channel
    ssim_scores = []

    for channel in range(input_tensor.size(1)):
        input_channel = input_tensor[:, channel, :, :].unsqueeze(1)
        target_channel = target_tensor[:, channel, :, :].unsqueeze(1)
        ssim_channel = ssim(input_channel, target_channel)
        ssim_channel = ssim_channel.mean().unsqueeze(0)
        ssim_scores.append(ssim_channel)

    ssim_scores = torch.cat(ssim_scores, dim=0)

    # Convert tensor to tuple
    ssim_scores = ssim_scores.squeeze().tolist()

    return ssim_scores
