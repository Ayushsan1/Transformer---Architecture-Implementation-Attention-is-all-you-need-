# Positional Encoding
import math

import torch

from config import d_model as configured_d_model
from config import max_length_seq as configured_max_length_seq


class PositionalEncoding(torch.nn.Module):
    def __init__(
        self,
        d_model: int = configured_d_model,
        max_length_seq: int = configured_max_length_seq,
    ):
        super().__init__()
        self.d_model = d_model

        # Create a matrix of shape (max_len, d_model) to hold the positional encodings
        pe = torch.zeros(max_length_seq, d_model)
        position = torch.arange(0, max_length_seq, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add a batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to have shape (batch_size, seq_len, embedding_dim)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len] 
