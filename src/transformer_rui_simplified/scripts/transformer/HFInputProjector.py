import torch
from torch import nn

class HFInputProjector(nn.Module):
    """
    Projects combined categorical embeddings and continuous features
    into a shared d_model space, per stock and time step.
    """
    def __init__(self, embedding_dim, continuous_dim, d_model):
        super().__init__()
        self.input_proj = nn.Linear(embedding_dim + continuous_dim, d_model)
        self.norm = nn.LayerNorm(d_model)  # Optional but recommended for stability

    def forward(self, embedding_out, continuous_features):
        """
        Args:
            embedding_out:      [B, T, S, E]
            continuous_features: [B, T, S, C]

        Returns:
            projected_input: [B, T, S, d_model]
        """
        if embedding_out.shape[:-1] != continuous_features.shape[:-1]:
            raise ValueError(f"Shape mismatch: {embedding_out.shape} vs {continuous_features.shape}")

        x = torch.cat([embedding_out, continuous_features], dim=-1)  # [B, T, S, E+C]
        x = self.input_proj(x)                                       # [B, T, S, d_model]
        return self.norm(x)
