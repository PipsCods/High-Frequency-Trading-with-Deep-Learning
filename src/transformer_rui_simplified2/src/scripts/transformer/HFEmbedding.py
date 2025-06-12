import torch
import torch.nn as nn

class HFEmbedding(nn.Module):
    """
    Uniform embedding of all categorical features.
    """
    def __init__(self, vocab_sizes, embed_dims):
        """
        Args:
            vocab_sizes: dict {feature_name: vocab_size}
            embed_dims: dict {feature_name: embed_dim}
        """
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embed_dims[name])
            for name, vocab_size in vocab_sizes.items()
        })

        self.feature_order = list(vocab_sizes.keys())

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, S, num_cat_feats]
               Assumes that features are in the same order as self.feature_order

        Returns:
            Tensor of shape [B, T, S, total_embedding_dim]
        """
        embeds = []
        for i, name in enumerate(self.feature_order):
            embeds.append(self.embeddings[name](x[..., i].long()))
        return torch.cat(embeds, dim=-1)
