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
    target_input = torch.tensor(
        [full_target[:-1]], dtype=torch.long, device=device
    )
    labels = torch.tensor([full_target[1:]], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(source_tokens, target_input)
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        # Give the decoder the French prefix and predict the target word.
        french_prefix = torch.tensor(
            [target_tokenizer.encode(TARGET_TEXT, add_special_tokens=False)],
            dtype=torch.long,
            device=device,
        )
        prefix_with_bos = torch.cat(
            (
                 torch.tensor([[target_tokenizer.get_bos_id()]], device=device),
                french_prefix,
            ),
            dim=1,
        )
        probabilities = model.probabilities(source_tokens, prefix_with_bos)
        predicted_id = probabilities[:, -1, :].argmax(dim=-1).item()

    print(f"test loss: {loss.item():.4f}")
    print(f"French prefix: {TARGET_TEXT}")
    print(f"target word: {TARGET_WORD}")
    print(f"predicted final word: {target_tokenizer.decode([predicted_id])}")
    print(f"full translation: {greedy_translate(model, SOURCE_TEXT, source_tokenizer, target_tokenizer, device)}")


if __name__ == "__main__":
    test()