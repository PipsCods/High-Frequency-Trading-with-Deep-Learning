import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target):
        mse = ((output - target) ** 2).mean()
        sign_loss = (torch.sign(output) != torch.sign(target)).float().mean()

        # Ensure both components are positive
        mse = torch.clamp(mse, min=0.0)
        sign_loss = torch.clamp(sign_loss, min=0.0, max=1.0)

        combined_loss = self.alpha * mse + (1 - self.alpha) * sign_loss

        return torch.clamp(combined_loss, min=1e-6)
