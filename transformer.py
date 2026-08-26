import torch
import tiktoken
from torch import nn

from Decoder_block import TransformerDecoder
from Encoder_block import TransformerEncoder


class WordTokenizer:
	# Create the GPT-2 tokenizer and assign IDs for special tokens.
	def __init__(self, model_name: str = "gpt2"):
		self.tokenizer = tiktoken.get_encoding(model_name)
		self.base_vocab_size = self.tokenizer.n_vocab
		self.pad_id = self.base_vocab_size
		self.bos_id = self.base_vocab_size + 1
		self.eos_id = self.base_vocab_size + 2

	# Return the total number of normal and special tokens.
	def get_vocab_size(self) -> int:
		return self.base_vocab_size + 3

	# Return the ID used for padding.
	def get_pad_id(self) -> int:
		return self.pad_id

	# Return the ID that marks the beginning of a sentence.
	def get_bos_id(self) -> int:
		return self.bos_id

	# Return the ID that marks the end of a sentence.
	def get_eos_id(self) -> int:
		return self.eos_id

	# Convert text into GPT-2 token IDs and optionally add sentence markers.
	def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
		token_ids = self.tokenizer.encode(text)
		if add_special_tokens:
			return [self.bos_id] + token_ids + [self.eos_id]
		return token_ids

	# Convert token IDs back into readable text while removing special tokens.
	def decode(self, token_ids: list[int]) -> str:
		actual_token_ids = []
		for token_id in token_ids:
			if token_id == self.pad_id or token_id == self.bos_id or token_id == self.eos_id:
				continue
			actual_token_ids.append(token_id)

		return self.tokenizer.decode(actual_token_ids)


class Transformer(nn.Module):
	"""Complete encoder-decoder Transformer with a vocabulary projection."""

	# Create the encoder, decoder, and final vocabulary projection layer.
	def __init__(
		self,
		source_vocab_size: int,
		target_vocab_size: int,
		d_model: int = 64,
		num_heads: int = 4,
		d_ff: int = 128,
		num_layers: int = 2,
		max_length_seq: int = 32
	):
		super().__init__()
		self.encoder = TransformerEncoder(
			vocab_size=source_vocab_size,
			d_model=d_model,
			num_heads=num_heads,
			max_length_seq=max_length_seq,
			d_ff=d_ff,
			num_layers=num_layers,
		)
		self.decoder = TransformerDecoder(
			vocab_size=target_vocab_size,
			d_model=d_model,
			num_heads=num_heads,
			max_length_seq=max_length_seq,
			d_ff=d_ff,
			num_layers=num_layers,
		)

		# Convert each decoder hidden state into a score for every target token.
		self.linear = nn.Linear(d_model, target_vocab_size)

	# Run source and target tokens through the full Transformer and return logits.
	def forward(
		self,
		source_tokens: torch.Tensor,
		target_input_tokens: torch.Tensor,
	) -> torch.Tensor:
		encoder_output = self.encoder(source_tokens)
		decoder_output = self.decoder(target_input_tokens, encoder_output)
		logits = self.linear(decoder_output)
		return logits

	# Convert the model's logits into probabilities using softmax.
	def probabilities(
		self,
		source_tokens: torch.Tensor,
		target_input_tokens: torch.Tensor,
	) -> torch.Tensor:
		"""Return softmax probabilities for each target position."""
		logits = self(source_tokens, target_input_tokens)
		probabilities = torch.softmax(logits, dim=-1)
		return probabilities


# Convert source and target text into tensors for teacher-forcing training.
def make_training_tensors(
	source_text: str,
	target_text: str,
	target_word: str,
	source_tokenizer: WordTokenizer,
	target_tokenizer: WordTokenizer,
	device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	full_target_text = target_text + " " + target_word
	source_tokens = torch.tensor(
		[source_tokenizer.encode(source_text)], dtype=torch.long, device=device
	)
	target_tokens = torch.tensor(
		[target_tokenizer.encode(full_target_text)], dtype=torch.long, device=device
	)

	# The decoder receives <bos> ... and learns to predict ... <eos>.
	target_input_tokens = target_tokens[:, :-1]
	target_labels = target_tokens[:, 1:]
	return source_tokens, target_input_tokens, target_labels


# Generate a translation one token at a time by choosing the highest score.
def greedy_translate(
	model: Transformer,
	source_text: str,
	source_tokenizer: WordTokenizer,
	target_tokenizer: WordTokenizer,
	device: torch.device,
	max_new_tokens: int = 12,
) -> str:
	model.eval()
	source_tokens = torch.tensor(
		[source_tokenizer.encode(source_text)], dtype=torch.long, device=device
	)
	generated = torch.tensor(
		[[target_tokenizer.get_bos_id()]], dtype=torch.long, device=device
	)

	with torch.no_grad():
		for _ in range(max_new_tokens):
			logits = model(source_tokens, generated)
			next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
			generated = torch.cat((generated, next_token), dim=1)
			if next_token.item() == target_tokenizer.get_eos_id():
				break

	return target_tokenizer.decode(generated.squeeze(0).tolist())
