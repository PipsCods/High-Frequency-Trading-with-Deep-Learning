import torch.nn as nn
import torch.nn.init as init

class PredictionHead(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.head = nn.Linear(d_model, output_dim)

        # Xavier initialization
        init.xavier_uniform_(self.head.weight)
        self.head.bias.data.fill_(0.01)  # small positive bias to encourage non-zero output

    def forward(self, x):
        """
        x: [B, T, S, d_model]
        Returns:
            [B, S] or [B, S, 1] depending on output_dim
        """
        last_step = x[:, -1, :, :]  # [B, S, d_model]
        out = self.head(last_step)  # [B, S, output_dim]
        if out.shape[-1] == 1:
            out = out.squeeze(-1)   # [B, S]
        return out
