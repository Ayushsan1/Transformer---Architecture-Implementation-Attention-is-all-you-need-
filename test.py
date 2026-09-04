from pathlib import Path
import torch
from dataset import (
    load_translation_data,
    split_dataset,
)
from log_utils import setup_logger
from transformer import (
    Transformer,
    WordTokenizer,
    greedy_translate,
)

DATASET_PATH = (
    Path(__file__).parent
    / "translation_dataset.json"
)

CHECKPOINT_PATH = (
    Path(__file__).parent
    / "transformer_model_no_tf.pt"
)

SEED = 7

NUM_TEST_EXAMPLES_TO_SHOW = 10

def test():
    logger = setup_logger("test")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info("=" * 60)
    logger.info("TRANSFORMER AUTOREGRESSIVE TESTING")
    logger.info("=" * 60)
    logger.info("Device: %s", device)

    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    source_tokenizer = WordTokenizer()
    target_tokenizer = WordTokenizer()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            f"Run train.py first."
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model_config = checkpoint.get(
        "model_config",
        {},
    )

    model = Transformer(
        source_vocab_size=source_tokenizer.get_vocab_size(),
        target_vocab_size=target_tokenizer.get_vocab_size(),
        **model_config,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data = load_translation_data(DATASET_PATH)
    _, _, test_data = split_dataset(
        data,
        train_ratio=0.8,
        validation_ratio=0.1,
        seed=SEED,
    )

    logger.info("Test examples: %s", len(test_data))
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SET PREDICTIONS")
    logger.info("=" * 60)

    for index, item in enumerate(test_data[:NUM_TEST_EXAMPLES_TO_SHOW], start=1):
        source_text = item["source"]
        target_text = item["target"]

        prediction = greedy_translate(
            model=model,
            source_text=source_text,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            device=device,
            max_new_tokens=20,
        )

        logger.info("")
        logger.info("Example %s", index)
        logger.info("Source    : %s", source_text)
        logger.info("Expected  : %s", target_text)
        logger.info("Predicted : %s", prediction)

    logger.info("")
    logger.info("CUSTOM TRANSLATION")

    custom_source = input(
        "Enter English sentence "
        "(press Enter to skip): "
    ).strip()

    if custom_source:
        prediction = greedy_translate(
            model=model,
            source_text=custom_source,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            device=device,
            max_new_tokens=20,
        )

        logger.info("")
        logger.info("English   : %s", custom_source)
        logger.info("French    : %s", prediction)


if __name__ == "__main__":
    test()