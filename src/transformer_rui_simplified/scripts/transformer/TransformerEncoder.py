import torch
import torch.nn as nn
from transformer.HFEmbedding import HFEmbedding
from transformer.HFInputProjector import HFInputProjector
from transformer.PositionalEmbedding import PositionalEmbedding
from transformer.TransformerBlock import TransformerBlock

class TransformerEncoder(nn.Module):
    def __init__(self,
                 vocab_sizes,
                 embed_dims,
                 num_cont_features,
                 d_model,
                 seq_len,
                 num_layers,
                 expansion_factor,
                 n_heads,
                 dropout: float,
                 debug: bool = False):
        super().__init__()

        self.embedding = HFEmbedding(vocab_sizes=vocab_sizes, embed_dims=embed_dims)
        total_embedding_dim = sum(embed_dims.values())
        self.projector = HFInputProjector(total_embedding_dim, num_cont_features, d_model)
        self.pos_encoder = PositionalEmbedding(max_len=seq_len, d_model=d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, expansion_factor, dropout)
            for _ in range(num_layers)
        ])
        self.debug = debug

    def forward(self, cat_inputs, cont_inputs):
        """
        Args:
            cat_inputs:  [B, T, S, num_cat_features]
            cont_inputs: [B, T, S, num_cont_features]
        Returns:
            [B, T, S, d_model]
        """

        emb = self.embedding(cat_inputs)     # [B, T, S, E]
        x = self.projector(emb, cont_inputs) # [B, T, S, d_model]
        x = self.pos_encoder(x)              # [B, T, S, d_model]

        for block in self.blocks:
            x = block(x)

        if self.debug:
            print("  [TransformerEncoder] output shape:", x.shape)

        return x

