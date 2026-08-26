import torch
from torch import nn

from transformer import Transformer, WordTokenizer, make_training_tensors


SOURCE_TEXT = "i like apples"
TARGET_TEXT = "j aime les"
TARGET_WORD = "pommes"
CHECKPOINT_PATH = "transformer_model.pt"


def train(epochs: int = 100, learning_rate: float = 0.003) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)

    source_tokenizer = WordTokenizer()
    target_tokenizer = WordTokenizer()
    source_tokens, target_input_tokens, target_labels = make_training_tensors(
        SOURCE_TEXT,
        TARGET_TEXT,
        TARGET_WORD,
        source_tokenizer,
        target_tokenizer,
        device,
    )

    model = Transformer(
        source_vocab_size=source_tokenizer.get_vocab_size(),
        target_vocab_size=target_tokenizer.get_vocab_size(),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss(ignore_index=target_tokenizer.get_pad_id())

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(source_tokens, target_input_tokens)
        loss = loss_function(
            logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:>3} | loss {loss.item():.4f}")

    torch.save(
        {"model_state": model.state_dict(), "tokenizer_name": "gpt2"},
        CHECKPOINT_PATH,
    )
    print(f"saved trained model to {CHECKPOINT_PATH}")
    print(f"training device: {device}")


if __name__ == "__main__":
    train()