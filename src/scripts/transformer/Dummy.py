import torch
import torch.nn as nn

class DummyPredictor(nn.Module):
    def __init__(self, constant_value=0.02):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(constant_value), requires_grad=False)

    def forward(self, x):
        # x shape: [B, T, S, D] — we want to output shape [B, S]
        B, T, S, D = x.shape
        return self.constant.expand(B, S)