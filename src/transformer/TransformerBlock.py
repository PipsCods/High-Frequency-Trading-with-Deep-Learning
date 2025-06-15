import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, expansion_factor, dropout):
        super().__init__()

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Position-wise feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, expansion_factor * d_model),
            nn.GELU(),  # GELU is great for smooth gradients in continuous inputs
            nn.Linear(expansion_factor * d_model, d_model)
        )

        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the transformer block.

        Args:
            x (Tensor): Shape [B, T, d_model]

        Returns:
            Tensor: Output shape [B, T, d_model]
        """
        # Multi-head self-attention with residual connection and layer norm
        attn_out, _ = self.attention(x, x, x)  # Self-attention: [B, T, d_model]
        x = self.norm1(x + self.dropout1(attn_out))  # Residual + norm

        # Feedforward with residual connection and layer norm
        ff_out = self.feed_forward(x)  # [B, T, d_model]
        x = self.norm2(x + self.dropout2(ff_out))  # Residual + norm

        return x
