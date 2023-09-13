import math
import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


# Define a custom loss function for Peak Signal-to-Noise Ratio (PSNR)
class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()

    def forward(self, prediction, target):
        psnr = PeakSignalNoiseRatio()
        psnr_value = psnr(prediction, target)
        
        # Since PSNR is typically used as a quality metric, we negate it to use it as a loss
        return -psnr_value

# Define a custom loss function for SSIM that maps [0, 1] to (inf, -inf)
class TangentSSIMLoss(nn.Module):
    def __init__(self):
        super(TangentSSIMLoss, self).__init__()

    def forward(self, prediction, target):
        # Calculate Structural Similarity Index (SSIM)
        ssim = StructuralSimilarityIndexMeasure(data_range=1)
        ssim_value = ssim(prediction, target)

        # Calculate a function which maps [0,1] to (inf, -inf)
        loss = -torch.tan(math.pi * (ssim_value - 0.5))

        return loss

# Define a custom loss function for Peak Signal-to-Noise Ratio (PSNR)
class SSIM_PSNRLoss(nn.Module):
    def __init__(self):
        super(SSIM_PSNRLoss, self).__init__()

    def forward(self, prediction, target):
        psnr = PeakSignalNoiseRatio()
        psnr_value = psnr(prediction, target)
        
        # Calculate Structural Similarity Index (SSIM)
        ssim = StructuralSimilarityIndexMeasure(data_range=1)
        ssim_value = ssim(prediction, target)

        # Calculate a function which maps [0,1] to (inf, -inf)
        ssim_loss = torch.tan(math.pi / 2 * (1 - ssim_value))

        loss = -psnr_value + ssim_loss*40
        
        # Since PSNR is typically used as a quality metric, we negate it to use it as a loss
        return loss
    
class NormalizeNorm1Loss(nn.Module):
    def __init__(self):
        super(NormalizeNorm1Loss, self).__init__()

    def forward(self, prediction, target):
        max = torch.maximum(prediction, target) + 1e-3 * torch.ones_like(prediction)
        norm1 = torch.absolute(prediction - target)
        loss = torch.mean(torch.mul(norm1, max.pow(-1)))

        return torch.log10(loss)*10