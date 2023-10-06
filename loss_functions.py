import math
import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# Custom loss function for Peak Signal-to-Noise Ratio (PSNR)
class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()

    def forward(self, prediction, target):
        # Calculate PSNR using the PeakSignalNoiseRatio metric
        psnr = PeakSignalNoiseRatio()
        psnr_value = psnr(prediction, target)
        
        # Negate PSNR to use it as a loss (lower PSNR is worse)
        return -psnr_value

# Custom loss function for Structural Similarity Index (SSIM) with tangent mapping
class TangentSSIMLoss(nn.Module):
    def __init__(self):
        super(TangentSSIMLoss, self).__init__()

    def forward(self, prediction, target):
        # Calculate SSIM using the StructuralSimilarityIndexMeasure metric
        ssim = StructuralSimilarityIndexMeasure(data_range=1)
        ssim_value = ssim(prediction, target)

        # Map SSIM to (-inf, 0] using a tangent function
        loss = torch.tan(math.pi / 2 * (1 - (1 + ssim_value)/2))
        return loss

# Custom loss function for Logarithmic Normalized L1 (LNL1) loss
class LNL1Loss(nn.Module):
    def __init__(self):
        super(LNL1Loss, self).__init__()

    def forward(self, prediction, target):
        # Calculate element-wise absolute difference between prediction and target
        norm1 = torch.abs(prediction - target)

        # Calculate a scaling factor to normalize the loss
        max = torch.maximum(prediction, target) + 1e-3 * torch.ones_like(prediction)
        normalize_norm1 = torch.mean(torch.mul(norm1, max.pow(-1)))

        # Calculate LNL1 value and negate it (lower LNL1 is worse)
        lnl1_value = -20 * torch.log10(normalize_norm1)
        loss = -lnl1_value
        return loss

# Custom combined loss function using PSNR, SSIM, and LNL1 with specified weights
class PSNR_SSIM_LNL1Loss(nn.Module):
    def __init__(self, weight_1, weight_2, weight_3):
        super(PSNR_SSIM_LNL1Loss, self).__init__()
        self.w1 = weight_1  # Weight for PSNR loss
        self.w2 = weight_2  # Weight for SSIM loss
        self.w3 = weight_3  # Weight for LNL1 loss

    def forward(self, prediction, target):
        # Calculate PSNR and negate it (lower PSNR is worse)
        psnr = PeakSignalNoiseRatio()
        psnr_value = psnr(prediction, target)
        psnr_loss = -psnr_value

        # Calculate SSIM and map it using the tangent function
        ssim = StructuralSimilarityIndexMeasure(data_range=1)
        ssim_value = ssim(prediction, target)
        ssim_loss = torch.tan(math.pi / 2 * (1 - (1 + ssim_value) / 2))

        # Calculate LNL1 and negate it (lower LNL1 is worse)
        max = torch.maximum(prediction, target) + 1e-3 * torch.ones_like(prediction)
        norm1 = torch.abs(prediction - target)
        normalize_norm1 = torch.mean(torch.mul(norm1, max.pow(-1)))
        lnl1_value = -20 * torch.log10(normalize_norm1)
        lnl1_loss = -lnl1_value
        
        # Combine the three loss components with specified weights
        loss = (self.w1 * psnr_loss) + (self.w2 * ssim_loss) + (self.w3 * lnl1_loss)
        return loss
