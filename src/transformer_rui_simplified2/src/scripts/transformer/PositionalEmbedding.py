from torch import nn
import torch

class PositionalEmbedding(nn.Module):
    """We choose a max_len, which is the maximum expected sequence length (e.g., the number of lags or time steps used as input).
    It can be set conservatively higher to accommodate longer sequences.

    We choose d_model, the dimensionality of the Transformer input vectors — this is the output size of the HFInputProjector,
    which combines and projects embeddings and continuous features.

    We then add the positional embeddings to the projected input vectors. This step encodes time-step information so the
    Transformer can understand the order of events in the sequence."""

    def __init__(self, max_len, d_model):
        """
        Args:
            max_len: maximum possible sequence length (e.g., 100)
            d_model: dimension of model input vectors
        """
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: [B, T, S, d_model] — model input avec dimension stock

        Returns:
            x + positional embedding: [B, T, S, d_model]
        """

        B, T, S, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).unsqueeze(-1).expand(B, T, S)  # [B, T, S]
        pos_embed = self.pos_embedding(positions)  # [B, T, S, d_model]
        return x + pos_embed
