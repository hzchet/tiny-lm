import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len: int = 512):
        """
        Inputs
            embedding_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        
        pe = torch.zeros(1, max_len, embedding_dim)
        
        numerator = torch.arange(max_len, dtype=torch.float).view(-1, 1)
        reciprocal_denominator = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )
        grid = numerator * reciprocal_denominator
        
        pe[:, :, 0::2] = torch.sin(grid)
        pe[:, :, 1::2] = torch.cos(grid)
        
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = self.pe[:, :x.size(1), :] + x
        return x
    

class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        num_encoder_layers: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_len: int = 512,
        **kwargs
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=3)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim, 
            n_heads, 
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.head = nn.Linear(embedding_dim, num_embeddings)

        self.init_weights()
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.head.bias.data.zero_()
        self.head.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, input_ids, padding_mask):
        """
        Args:
            input_ids: tokenized texts, torch.Tensor (B, L)
            padding_mask: padding mask, torch.Tensor (B, L)
        """
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(x.device)
        output = self.encoder(x, causal_mask, is_causal=True)
        logits = self.head(output)
        return logits
