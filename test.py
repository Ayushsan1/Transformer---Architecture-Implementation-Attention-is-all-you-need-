import torch
from torch import nn

from train import CHECKPOINT_PATH, SOURCE_TEXT, TARGET_TEXT, TARGET_WORD
from transformer import Transformer, WordTokenizer, greedy_translate


def test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_tokenizer = WordTokenizer()
    target_tokenizer = WordTokenizer()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model = Transformer(
        source_vocab_size=source_tokenizer.get_vocab_size(),
        target_vocab_size=target_tokenizer.get_vocab_size(),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    source_tokens = torch.tensor(
        [source_tokenizer.encode(SOURCE_TEXT)], dtype=torch.long, device=device
    )
    full_target = target_tokenizer.encode(TARGET_TEXT + " " + TARGET_WORD)
    target_labels = torch.tensor([full_target[1:]], dtype=torch.long, device=device)

    with torch.no_grad():
        loss_function = nn.CrossEntropyLoss(
            ignore_index=target_tokenizer.get_pad_id()
        )
        target_input_tokens = torch.tensor(
            [full_target[:-1]], dtype=torch.long, device=device
        )
        logits = model(source_tokens, target_input_tokens)
        loss = loss_function(
            logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1)
        )

    print(f"test loss: {loss.item():.4f}")
    print(f"full translation: {greedy_translate(model, SOURCE_TEXT, source_tokenizer, target_tokenizer, device)}")


if __name__ == "__main__":
    test()