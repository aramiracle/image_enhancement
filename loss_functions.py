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
        loss = torch.tan(math.pi / 2 * (1 - ssim_value))

        return loss

# Define a custom loss function for Peak Signal-to-Noise Ratio (PSNR)
class LNL1Loss(nn.Module):
    def __init__(self):
        super(LNL1Loss, self).__init__()

    def forward(self, prediction, target):
        max = torch.maximum(prediction, target) + 1e-3 * torch.ones_like(prediction)
        norm1 = torch.absolute(prediction - target)
        normalize_norm1 = torch.mean(torch.mul(norm1, max.pow(-1)))
        lnl1_value = -10*torch.log10(normalize_norm1)

        loss = -lnl1_value
        
        # Since PSNR is typically used as a quality metric, we negate it to use it as a loss
        return loss
    
class PSNR_SSIM_LNL1Loss(nn.Module):
    def __init__(self, weight_1, weight_2, weight_3):
        super(PSNR_SSIM_LNL1Loss, self).__init__()
        self.w1 = weight_1
        self.w2 = weight_2
        self.w3 = weight_3

    def forward(self, prediction, target):
        
        psnr = PeakSignalNoiseRatio()
        psnr_value = psnr(prediction, target)
        psnr_loss = -psnr_value

        # Calculate Structural Similarity Index (SSIM)
        ssim = StructuralSimilarityIndexMeasure(data_range=1)
        ssim_value = ssim(prediction, target)

        # Calculate a function which maps [0,1] to (inf, -inf)
        ssim_loss = torch.tan(math.pi / 2 * (1 - ssim_value))
        
        max = torch.maximum(prediction, target) + 1e-3 * torch.ones_like(prediction)
        norm1 = torch.absolute(prediction - target)
        normalize_norm1 = torch.mean(torch.mul(norm1, max.pow(-1)))
        lnl1_value = -10*torch.log10(normalize_norm1)
        lnl1_loss = -lnl1_value
        
        loss = (self.w1 * psnr_loss) + (self.w2 * ssim_loss) + (self.w3 * lnl1_loss)
        # Since PSNR is typically used as a quality metric, we negate it to use it as a loss
        return loss
    