import torch
import torch.nn as nn

import self_attention

from config import (
    d_model as configured_d_model,
    num_heads as configured_num_heads,
)


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int = configured_d_model,
        num_heads: int = configured_num_heads,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(
            d_model,
            d_model,
        )

        self.W_K = nn.Linear(
            d_model,
            d_model,
        )

        self.W_V = nn.Linear(
            d_model,
            d_model,
        )

        self.W_O = nn.Linear(
            d_model,
            d_model,
        )

        self.attention = (
            self_attention.ScaledDotProductAttention(
                d_model,
                num_heads,
            )
        )

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        batch_size = Q.size(0)

        Q_proj = (
            self.W_Q(Q)
            .view(
                batch_size,
                -1,
                self.num_heads,
                self.d_k,
            )
            .transpose(1, 2)
        )

        K_proj = (
            self.W_K(K)
            .view(
                batch_size,
                -1,
                self.num_heads,
                self.d_k,
            )
            .transpose(1, 2)
        )

        V_proj = (
            self.W_V(V)
            .view(
                batch_size,
                -1,
                self.num_heads,
                self.d_k,
            )
            .transpose(1, 2)
        )

        attention_output, _ = self.attention(
            Q_proj,
            K_proj,
            V_proj,
            mask,
        )

        attention_output = (
            attention_output
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                -1,
                self.d_model,
            )
        )

        output = self.W_O(
            attention_output
        )

        return output