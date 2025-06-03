import torch
import torch.nn as nn
from transformer.HFEmbedding import HFEmbedding
from transformer.HFInputProjector import HFInputProjector
from transformer.PositionalEmbedding import PositionalEmbedding
from transformer.TransformerBlock import TransformerBlock

class TransformerEncoder(nn.Module):
    def __init__(self,
                 basic_embed_dims, # embed dimensions for the basic features (symbol, day, dayname)
                 embed_dims, # embed dimensions for the continuous features
                 vocab_sizes_basic, # how many unique values are there in the basic features
                 vocab_sizes, # how many unique values are there in the continuous features
                 num_cont_features, # number of features (continuous ones)
                 d_model, # size of the vector where we will project our embedding
                 seq_len, # what is the max sequence (minimum number of lags)
                 num_layers,# number of layers
                 expansion_factor, # this allows to boost the reasonning of the model in the FeedForward sublayer process
                 n_heads,# number of attention heads
                 dropout,
                 ):

        """
        Embed_dims should contain the name of the different categorical features and what is the embedding dimension for each one.
        If the value is low, then the model goes faster but will capture less complex relationships.
        If the value is high, then the model goes faster but will capture more complex relationships. However, there is
        a higher risk of overfitting and asks a lot of memory.
        """

        super(TransformerEncoder, self).__init__()

        # Embedding for categorical + cyclical time features
        self.embedding = HFEmbedding(
            basic_embed_dims=basic_embed_dims,
            vocab_sizes_basic=vocab_sizes_basic,
            embed_dims=embed_dims,
            vocab_sizes=vocab_sizes
        )

        # Project embeddings before cont_features + continuous features into d_model
        total_input_dim_before_cont = sum(basic_embed_dims.values()) + sum(embed_dims.values()) + 4 # here the number 4 comes from the fact that we are positioning the hours and min
                                                                           # according to a cycle (cos,sin) for both
        self.projector = HFInputProjector(total_input_dim_before_cont, num_cont_features, d_model)

        # Add positional encoding
        self.pos_encoder = PositionalEmbedding(seq_len, d_model)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, expansion_factor, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, cat_inputs, cont_inputs):
        emb = self.embedding(cat_inputs)  # [B, T, S, emb_dim]
        x = self.projector(emb, cont_inputs)  # [B, T, S, d_model]
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        return x
