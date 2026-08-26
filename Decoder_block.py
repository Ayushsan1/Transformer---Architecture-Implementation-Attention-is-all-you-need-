import torch
from torch import nn
from MLP import FeedForwardNetwork
from CrossAttention import CrossAttention
from positional_encoding import PositionalEncoding
from MaskMHA import MaskedMultiHeadAttention
from AddNorm import AddNormalization
from input_embedding import InputEmbedding
from config import num_heads, num_layers, max_length_seq, src_vocab_size, tgt_vocab_size, d_ff, d_model, dropout


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int = d_model,
        num_heads: int = num_heads,
        d_ff: int = d_ff):
        super().__init__()

        self.masked_self_attention = MaskedMultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNormalization(d_model)

        self.cross_attention = CrossAttention(d_model, num_heads)
        self.add_norm2 = AddNormalization(d_model)

        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.add_norm3 = AddNormalization(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        masked_attention_output = self.masked_self_attention(
            x, x, x, target_mask
        )
        x = self.add_norm1(x, masked_attention_output)

        cross_attention_output, _ = self.cross_attention(
            x, encoder_output, memory_mask
        )
        x = self.add_norm2(x, cross_attention_output)
        feed_forward_output = self.feed_forward(x)
        return self.add_norm3(x, feed_forward_output)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = tgt_vocab_size,
        d_model: int = d_model,
        num_heads: int = num_heads,
        max_length_seq: int = max_length_seq,
        d_ff: int = d_ff,
        num_layers: int = num_layers,
    ):
        super().__init__()

        self.embedding = InputEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length_seq)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff)
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
        return torch.tril(
            torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        target_tokens: torch.Tensor,
        encoder_output: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(target_tokens)
        x = self.positional_encoding(x)

        if target_mask is None:
            target_mask = self.causal_mask(x.size(1), x.device)

        for layer in self.layers:
            x = layer(x, encoder_output, target_mask, memory_mask)

        return x
