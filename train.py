import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformer import Transformer, WordTokenizer
from dataset import (
    TranslationDataset,
    load_translation_data,
    split_dataset,
    create_collate_fn,
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

BATCH_SIZE = 8

EPOCHS = 100

LEARNING_RATE = 0.001

MAX_GENERATION_LENGTH = 20


torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



def autoregressive_training_step(
    model,
    source_tokens,
    target_tokens,
    source_tokenizer,
    target_tokenizer,
    loss_function,
):

    batch_size = source_tokens.size(0)

    # Start every sentence with BOS.
    generated = torch.full(
        (
            batch_size,
            1,
        ),
        target_tokenizer.get_bos_id(),
        dtype=torch.long,
        device=source_tokens.device,
    )

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=source_tokens.device,
    )

    losses = []

    for position in range(
        1,
        target_tokens.size(1),
    ):

        target_labels = (
            target_tokens[:, position]
        )

        valid_positions = (
            target_labels
            != target_tokenizer.get_pad_id()
        )

        if not valid_positions.any():
            break

        logits = model(
            source_tokens,
            generated,
            source_pad_id=(
                source_tokenizer.get_pad_id()
            ),
            target_pad_id=(
                target_tokenizer.get_pad_id()
            ),
        )

        next_token_logits = (
            logits[:, -1, :]
        )

        step_loss = loss_function(
            next_token_logits,
            target_labels,
        )

        losses.append(
            step_loss
        )

        predicted_token = (
            next_token_logits
            .argmax(
                dim=-1,
                keepdim=True,
            )
        )

        eos_tensor = torch.full_like(
            predicted_token,
            target_tokenizer.get_eos_id(),
        )

        predicted_token = torch.where(
            finished.unsqueeze(1),
            eos_tensor,
            predicted_token,
        )

        generated = torch.cat(
            (
                generated,
                predicted_token,
            ),
            dim=1,
        )

        finished = (
            finished
            | (
                predicted_token.squeeze(1)
                == target_tokenizer.get_eos_id()
            )
        )

        if finished.all():
            break

    if not losses:
        raise RuntimeError(
            "No valid target tokens were found."
        )

    loss = torch.stack(
        losses
    ).mean()

    return loss


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    data_loader,
    source_tokenizer,
    target_tokenizer,
    loss_function,
    device,
):
    model.eval()

    total_loss = 0.0
    batches = 0

    for batch in data_loader:

        source_tokens = (
            batch["source_tokens"]
            .to(device)
        )

        target_tokens = (
            batch["target_tokens"]
            .to(device)
        )

        loss = autoregressive_training_step(
            model=model,
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            loss_function=loss_function,
        )

        total_loss += loss.item()
        batches += 1

    if batches == 0:
        return float("inf")

    return total_loss / batches


# ============================================================
# Main Training Function
# ============================================================

def train(
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("TRANSFORMER AUTOREGRESSIVE TRAINING")
    print("=" * 60)

    print(
        f"Training device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        f"Dataset: {DATASET_PATH}"
    )

    print(
        f"Teacher forcing: DISABLED"
    )

    # --------------------------------------------------------
    # Tokenizers
    # --------------------------------------------------------

    source_tokenizer = WordTokenizer()
    target_tokenizer = WordTokenizer()

    print(
        f"Source vocabulary: "
        f"{source_tokenizer.get_vocab_size()}"
    )

    print(
        f"Target vocabulary: "
        f"{target_tokenizer.get_vocab_size()}"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data = load_translation_data(
        DATASET_PATH
    )

    print(
        f"Total sentence pairs: {len(data)}"
    )

    train_data, validation_data, test_data = (
        split_dataset(
            data,
            train_ratio=0.8,
            validation_ratio=0.1,
            seed=SEED,
        )
    )

    print(
        f"Training examples: {len(train_data)}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_data)}"
    )

    print(
        f"Test examples: {len(test_data)}"
    )

    # --------------------------------------------------------
    # Dataset objects
    # --------------------------------------------------------

    train_dataset = TranslationDataset(
        train_data
    )

    validation_dataset = TranslationDataset(
        validation_data
    )

    test_dataset = TranslationDataset(
        test_data
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    collate_fn = create_collate_fn(
        source_tokenizer,
        target_tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = Transformer(
        source_vocab_size=(
            source_tokenizer.get_vocab_size()
        ),
        target_vocab_size=(
            target_tokenizer.get_vocab_size()
        ),
    ).to(device)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
#LR scheduler that reduces the learning rate when a metric has stopped improving. 
#In this case, we monitor the validation loss and reduce the learning rate if it doesn't improve for 10 epochs. This can help the model converge better during training.
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=10,
    )

    loss_function = nn.CrossEntropyLoss(
        ignore_index=(
            target_tokenizer.get_pad_id()
        )
    )


    best_validation_loss = float("inf")

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        total_train_loss = 0.0
        train_batches = 0

        for batch in train_loader:

            source_tokens = (
                batch["source_tokens"]
                .to(device, non_blocking=True)
            )

            target_tokens = (
                batch["target_tokens"]
                .to(device, non_blocking=True)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = autoregressive_training_step(
                model=model,
                source_tokens=source_tokens,
                target_tokens=target_tokens,
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
                loss_function=loss_function,
            )

            loss.backward()

            # Prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        train_loss = (
            total_train_loss
            / max(train_batches, 1)
        )

        validation_loss = evaluate(
            model=model,
            data_loader=validation_loader,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            loss_function=loss_function,
            device=device,
        )

        lr_scheduler.step(validation_loss)

        if (
            epoch == 1
            or epoch % 5 == 0
        ):

            print(
                f"Epoch {epoch:>3}/{epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"validation loss: {validation_loss:.4f}"
            )
#Save the best model

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            torch.save(
                {
                    "model_state": (
                        model.state_dict()
                    ),
                    "optimizer_state": (
                        optimizer.state_dict()
                    ),
                    "lr_scheduler_state": (
                        lr_scheduler.state_dict()
                    ),
                    "epoch": epoch,
                    "model_config": (
                        model.model_config
                    ),
                    "tokenizer_name": "gpt2",
                    "source_pad_id": (
                        source_tokenizer.get_pad_id()
                    ),
                    "target_pad_id": (
                        target_tokenizer.get_pad_id()
                    ),
                    "best_validation_loss": (
                        best_validation_loss
                    ),
                },
                CHECKPOINT_PATH,
            )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Checkpoint saved to:\n"
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"Training device: {device}"
    )


if __name__ == "__main__":
    train()