import torch
from torch import nn

class HFInputProjector(nn.Module):
    """
    Projects combined categorical embeddings and continuous features
    into a shared d_model space, per stock and time step.
    """
    def __init__(self, embedding_dim, continuous_dim, d_model):
        """
        Args:
            embedding_dim: output dimension of HFEmbedding per stock [E]
            continuous_dim: number of continuous features per stock per time step [C]
            d_model: dimension expected by Transformer [D]
        """
        super(HFInputProjector, self).__init__()
        total_input_dim = embedding_dim + continuous_dim
        self.input_proj = nn.Linear(total_input_dim, d_model)

    def forward(self, embedding_out, continuous_features):
        """
        Args:
            embedding_out:      [B, T, S, E]
            continuous_features: [B, T, S, C]

        Returns:
            projected_input: [B, T, S, d_model]
        """
        x = torch.cat([embedding_out, continuous_features], dim=-1)  # [B, T, S, E+C]
        return self.input_proj(x)  # [B, T, S, d_model]
