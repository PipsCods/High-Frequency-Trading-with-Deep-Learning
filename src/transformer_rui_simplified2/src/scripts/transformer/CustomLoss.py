import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma

    """def forward(self, output, target):
        #Shape: [B, S] for both
        breakpoint()
        mse = nn.MSELoss()(output, target)

        # Directional loss: soft classification using log loss
        directional_loss = torch.log1p(torch.exp(-output * target)).mean()

        combined_loss = self.alpha * mse + (1 - self.alpha) * directional_loss

        return combined_loss"""

    def forward(self, output, target):
        mean_variance_obj = -(output * target).mean(dim=1) + 0.5 * self.gamma * ((output * target).pow(2)).mean(dim=1)
        return mean_variance_obj.mean()

