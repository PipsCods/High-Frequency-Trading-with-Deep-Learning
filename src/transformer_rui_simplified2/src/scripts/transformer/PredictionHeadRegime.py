import torch.nn as nn
import torch.nn.init as init

class PredictionHeadRegime(nn.Module):
    def __init__(self, d_model, output_dim=1, hidden_dim=32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.head(x)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out