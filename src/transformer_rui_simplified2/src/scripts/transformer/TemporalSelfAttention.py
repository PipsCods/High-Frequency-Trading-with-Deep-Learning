import torch.nn as nn

class TemporalSelfAttention(nn.Module):
    """
    Applies multi-head self-attention across time dimension (T) for each stock.
    Input: [B, T, S, d_model]
    Output: [B, T, S, d_model]
    """
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        """
        x: tensor of shape [B, T, S, d_model]
        """
        B, T, S, D = x.shape

        # Reshape to apply attention over time per stock
        x = x.permute(0, 2, 1, 3).reshape(B * S, T, D)  # [B*S, T, D]

        # Apply attention over the time dimension
        out, _ = self.attention(x, x, x)  # [B*S, T, D]

        # Reshape back to [B, T, S, D]
        out = out.reshape(B, S, T, D).permute(0, 2, 1, 3)  # [B, T, S, D]

        return out