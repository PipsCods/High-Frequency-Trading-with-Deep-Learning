import torch.nn as nn

class CrossSectionalSelfAttention(nn.Module):
    """
    Applies multi-head self-attention across stock dimension (S) for each time step.
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

        # Reshape to apply attention over stocks at each time step
        x = x.reshape(B * T, S, D)  # [B*T, S, D]

        # Apply attention over the stock dimension
        out, _ = self.attention(x, x, x)  # [B*T, S, D]

        # Reshape back to [B, T, S, D]
        out = out.reshape(B, T, S, D)

        return out
