import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, expansion_factor, dropout):
        super(TransformerBlock, self).__init__()

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

        # LayerNorms - it allows for stabilizing the learning and outputs
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward network - non-linear transformations per time step
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, expansion_factor * d_model),
            nn.GELU(), #activation function GELU --> better for time-series, smoother gradients, robust on noise, better-performance on continuous values
            nn.Linear(expansion_factor * d_model, d_model)
        )

        # Dropouts - as seen in the class, it prevents overfitting by killing some neurons
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        # Self-attention + residual + norm
        attn_out, _ = self.attention(x, x, x)            # shape: [B, T, d_model]
        x = self.norm1(x + self.dropout1(attn_out))      # residual connection

        # Feedforward + residual + norm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))        # residual connection

        return x  # shape: [B, T, d_model]
