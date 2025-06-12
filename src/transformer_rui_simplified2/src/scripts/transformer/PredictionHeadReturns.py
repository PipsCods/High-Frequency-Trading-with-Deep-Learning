import torch.nn as nn
import torch.nn.init as init

class PredictionHeadReturns(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.head = nn.Linear(d_model, output_dim)

        # He (Kaiming) initialization for GELU
        init.kaiming_uniform_(self.head.weight, nonlinearity='relu')
        self.head.bias.data.fill_(0.01)  # small positive bias to encourage non-zero output

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, S, d_model]

        Returns:
            Tensor of shape [B, S] or [B, S, output_dim]
        """
        out = self.head(x)  # [B, S, output_dim]
        if out.shape[-1] == 1:
            out = out.squeeze(-1)  # [B, S]
        return out
