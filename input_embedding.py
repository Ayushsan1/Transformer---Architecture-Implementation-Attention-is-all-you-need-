import math
import tiktoken
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from config import d_model as configured_d_model, src_vocab_size


class Tokenizer:

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        # Use the variable passed to the constructor
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.vocab_size = self.tokenizer.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)


class InputEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int = src_vocab_size,
        d_model: int = configured_d_model,
    ):
        super().__init__()
        # vocab_size defines the number of rows in the lookup matrix
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len) -> returns (batch_size, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model) #WHYYYY???


# --- Example Test Run ---
if __name__ == "__main__":
    # 1. Tokenize
    tok = Tokenizer("gpt2")
    text = "Attention is all you need"
    token_ids = tok.encode(text)
    print("Token IDs:", token_ids)

    # 2. Create configured-vocabulary IDs for the embedding example
    input_tensor = torch.randint(
        0,
        src_vocab_size,
        (1, len(token_ids)),
        dtype=torch.long,
    )

    # 3. Pass through Input Embedding
    embed_layer = InputEmbedding()
    embeddings = embed_layer(input_tensor)

    print("Embedding Output Shape:", embeddings.shape)
    # Output Shape: torch.Size([1, 5, 512])

