import torch
import tiktoken
from torch import nn

from Decoder_block import TransformerDecoder
from Encoder_block import TransformerEncoder

from config import (
    d_model,
    num_heads,
    d_ff,
    num_layers,
    max_length_seq,
)


class WordTokenizer:
    """
    Wrapper around the GPT-2 tiktoken tokenizer.

    Note:
    Despite the class name, this is actually a
    subword/BPE tokenizer rather than a word tokenizer.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
    ):
        self.model_name = model_name

        self.tokenizer = (
            tiktoken.get_encoding(
                model_name
            )
        )

        self.base_vocab_size = (
            self.tokenizer.n_vocab
        )

        self.pad_id = (
            self.base_vocab_size
        )

        self.bos_id = (
            self.base_vocab_size + 1
        )

        self.eos_id = (
            self.base_vocab_size + 2
        )

    def get_vocab_size(self) -> int:
        return (
            self.base_vocab_size + 3
        )

    def get_pad_id(self) -> int:
        return self.pad_id

    def get_bos_id(self) -> int:
        return self.bos_id

    def get_eos_id(self) -> int:
        return self.eos_id

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:

        token_ids = (
            self.tokenizer.encode(text)
        )

        if add_special_tokens:

            return (
                [self.bos_id]
                + token_ids
                + [self.eos_id]
            )

        return token_ids

    def decode(
        self,
        token_ids: list[int],
    ) -> str:

        actual_token_ids = []

        for token_id in token_ids:

            if token_id in (
                self.pad_id,
                self.bos_id,
                self.eos_id,
            ):
                continue

            actual_token_ids.append(
                token_id
            )

        return self.tokenizer.decode(
            actual_token_ids
        )


class Transformer(nn.Module):
    """
    Complete encoder-decoder Transformer.
    """

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        d_model: int = d_model,
        num_heads: int = num_heads,
        d_ff: int = d_ff,
        num_layers: int = num_layers,
        max_length_seq: int = max_length_seq,
    ):
        super().__init__()

        self.source_pad_id = None
        self.target_pad_id = None

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

        self.linear = nn.Linear(
            d_model,
            target_vocab_size,
        )

        self.model_config = {
            "d_model": d_model,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "max_length_seq": max_length_seq,
        }

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_input_tokens: torch.Tensor,
        source_pad_id: int | None = None,
        target_pad_id: int | None = None,
    ) -> torch.Tensor:

        source_mask = None

        if source_pad_id is not None:

            source_mask = (
                source_tokens != source_pad_id
            ).unsqueeze(1).unsqueeze(2)

        encoder_output = self.encoder(
            source_tokens,
            source_mask,
        )

        decoder_output = self.decoder(
            target_input_tokens,
            encoder_output,
            target_pad_id=target_pad_id,
            source_pad_mask=source_mask,
        )

        logits = self.linear(
            decoder_output
        )

        return logits

    def probabilities(
        self,
        source_tokens: torch.Tensor,
        target_input_tokens: torch.Tensor,
        source_pad_id: int | None = None,
        target_pad_id: int | None = None,
    ) -> torch.Tensor:

        logits = self(
            source_tokens,
            target_input_tokens,
            source_pad_id=source_pad_id,
            target_pad_id=target_pad_id,
        )

        return torch.softmax(
            logits,
            dim=-1,
        )


def greedy_translate(
    model: Transformer,
    source_text: str,
    source_tokenizer: WordTokenizer,
    target_tokenizer: WordTokenizer,
    device: torch.device,
    max_new_tokens: int = 20,
) -> str:

    model.eval()

    source_tokens = torch.tensor(
        [
            source_tokenizer.encode(
                source_text
            )
        ],
        dtype=torch.long,
        device=device,
    )

    generated = torch.tensor(
        [
            [
                target_tokenizer.get_bos_id()
            ]
        ],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            logits = model(
                source_tokens,
                generated,
                source_pad_id=source_tokenizer.get_pad_id(),
                target_pad_id=target_tokenizer.get_pad_id(),
            )

            last_token_logits = (
                logits[:, -1, :]
            )

            next_token = (
                last_token_logits
                .argmax(
                    dim=-1,
                    keepdim=True,
                )
            )

            generated = torch.cat(
                (
                    generated,
                    next_token,
                ),
                dim=1,
            )

            if (
                next_token.item()
                == target_tokenizer.get_eos_id()
            ):
                break

    return target_tokenizer.decode(
        generated.squeeze(0).tolist()
    )