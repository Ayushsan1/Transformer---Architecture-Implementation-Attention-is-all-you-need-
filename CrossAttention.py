import torch
from torch import nn

from config import d_model as configured_d_model
from config import num_heads as configured_num_heads
from self_attention import ScaledDotProductAttention


class CrossAttention(nn.Module):
	def __init__(
		self,
		d_model: int = configured_d_model,
		num_heads: int = configured_num_heads,
	):
		super().__init__()

		if d_model % num_heads != 0:
			raise ValueError("d_model must be divisible by num_heads")

		self.d_model = d_model
		self.num_heads = num_heads
		self.d_k = d_model // num_heads

		# Queries come from the decoder; keys and values come from the encoder.
		self.W_Q = nn.Linear(d_model, d_model)
		self.W_K = nn.Linear(d_model, d_model)
		self.W_V = nn.Linear(d_model, d_model)
		self.W_O = nn.Linear(d_model, d_model)
		self.attention = ScaledDotProductAttention(d_model, num_heads)

	def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, sequence_length, _ = x.shape
		return x.view(
			batch_size, sequence_length, self.num_heads, self.d_k
		).transpose(1, 2)

	def forward(
		self,
		decoder_states: torch.Tensor,
		encoder_output: torch.Tensor,
		mask: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Attend from decoder states to the encoder output.

		decoder_states: (batch_size, target_length, d_model)
		encoder_output: (batch_size, source_length, d_model)
		mask: optional mask broadcastable to (batch, heads, target_length, source_length)
		"""
		query = self._split_heads(self.W_Q(decoder_states))
		key = self._split_heads(self.W_K(encoder_output))
		value = self._split_heads(self.W_V(encoder_output))

		attention_output, attention_weights = self.attention(query, key, value, mask)
		attention_output = attention_output.transpose(1, 2).contiguous()
		attention_output = attention_output.view(
			decoder_states.size(0), decoder_states.size(1), self.d_model
		)

		return self.W_O(attention_output), attention_weights



