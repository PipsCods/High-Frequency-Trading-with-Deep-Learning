import torch
import torch.nn as nn

class HFEmbedding(nn.Module):
    """
    HFEmbedding with dynamic handling of both basic and additional categorical variables,
    and cyclical time encoding.
    """
    def __init__(self,
                 basic_embed_dims,
                 vocab_sizes_basic,
                 embed_dims,
                 vocab_sizes):
        """
        Args:
            basic_embed_dims: dict with keys 'symbol', 'day', 'day_name'
            vocab_sizes_basic: dict with vocabulary sizes for basic features
            embed_dims: dict with keys for additional categorical features
            vocab_sizes: dict with vocabulary sizes for additional categorical features
        """
        super(HFEmbedding, self).__init__()

        # Basic categorical embeddings
        self.symbol_embed = nn.Embedding(vocab_sizes_basic['symbol_vocab_size'], basic_embed_dims['symbol'])
        self.day_embed = nn.Embedding(vocab_sizes_basic['day_vocab_size'], basic_embed_dims['day'])
        self.dayname_embed = nn.Embedding(vocab_sizes_basic['day_name_vocab_size'], basic_embed_dims['day_name'])

        # Additional categorical embeddings
        self.additional_embeds = nn.ModuleDict({
            feat: nn.Embedding(vocab_sizes[f"{feat}_vocab_size"], embed_dims[feat])
            for feat in embed_dims
        })

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor [B, T, S, num_cat_feats]

        Returns:
            Tensor [B, T, S, total_embedding_dim]
        """
        # Basic embedding
        symbol = inputs[..., 0].long()
        day = inputs[..., 1].long()
        day_name = inputs[..., 2].long()
        hour = inputs[..., 3].long()
        minute = inputs[..., 4].long()

        s = self.symbol_embed(symbol)      # [B, T, S, embed_dim_symbol]
        d = self.day_embed(day)            # [B, T, S, embed_dim_day]
        dn = self.dayname_embed(day_name)  # [B, T, S, embed_dim_day_name]

        # Cyclical time encoding for hour and minute
        hour_sin = torch.sin(2 * torch.pi * hour / 24)
        hour_cos = torch.cos(2 * torch.pi * hour / 24)
        minute_sin = torch.sin(2 * torch.pi * minute / 60)
        minute_cos = torch.cos(2 * torch.pi * minute / 60)
        time_vec = torch.stack([hour_sin, hour_cos, minute_sin, minute_cos], dim=-1)  # [B, T, S, 4]

        # Additional categorical embeddings starting at index 5
        additional_cats = []
        for i, feat in enumerate(self.additional_embeds.keys()):
            feat_tensor = inputs[..., 5 + i].long()
            additional_cats.append(self.additional_embeds[feat](feat_tensor))  # [B, T, S, embed_dim_feat]

        # Concatenate all embeddings and time encoding vector
        out = torch.cat([s, d, dn, time_vec] + additional_cats, dim=-1)  # [B, T, S, total_embed_dim]

        return out