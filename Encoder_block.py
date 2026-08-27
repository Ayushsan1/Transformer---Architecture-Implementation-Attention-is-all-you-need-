import torch
import torch.nn as nn

from MLP import FeedForwardNetwork
from AddNorm import AddNormalization
from positional_encoding import PositionalEncoding
from Multiheadattention import MultiHeadAttention
from input_embedding import InputEmbedding

from config import (
    d_model as configured_d_model,
    num_heads as configured_num_heads,
    max_length_seq as configured_max_length_seq,
    d_ff as configured_d_ff,
    num_layers as configured_num_layers,
)


class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
    ):
        super().__init__()

        self.MHA = MultiHeadAttention(
            d_model,
            num_heads,
        )

        self.AddNorm1 = AddNormalization(
            d_model
        )

        self.MLP = FeedForwardNetwork(
            d_model,
            d_ff,
        )

        self.AddNorm2 = AddNormalization(
            d_model
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        attention_output = self.MHA(
            x,
            x,
            x,
            mask,
        )

        x = self.AddNorm1(
            x,
            attention_output,
        )

        ffn_output = self.MLP(x)

        x = self.AddNorm2(
            x,
            ffn_output,
        )

        return x


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int = configured_d_model,
        num_heads: int = configured_num_heads,
        max_length_seq: int = configured_max_length_seq,
        d_ff: int = configured_d_ff,
        num_layers: int = configured_num_layers,
    ):
        super().__init__()

        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
        )

        self.positional_encoding = PositionalEncoding(
            d_model,
            max_length_seq,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    num_heads,
                    d_ff,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = self.embedding(x)

        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(
                x,
                mask,
            )

        return x