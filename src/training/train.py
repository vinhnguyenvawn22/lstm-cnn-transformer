
import math
import time
from pathlib import Path

import pandas as pd
import torch

from tqdm.auto import tqdm


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device
):
    model.train()

    total_loss = 0.0
    total_tokens = 0

    progress = tqdm(train_loader, desc="Training", leave=False)

    for X, Y in progress:

        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()

        logits = model(X)

        vocab_size = logits.size(-1)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            Y.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        n_tokens = Y.numel()

        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return avg_loss, ppl


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for X, Y in tqdm(
        data_loader,
        desc="Evaluating",
        leave=False
    ):
        X = X.to(device)
        Y = Y.to(device)

        logits = model(X)

        vocab_size = logits.size(-1)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            Y.reshape(-1)
        )

        n_tokens = Y.numel()

        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return avg_loss, ppl


def fit(
    model,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    device,
    model_name,
    experiment_name,
    context_length,
    batch_size,
    root,
    epochs=50,
    patience=5
):
    # Thư mục lưu từng thí nghiệm
    checkpoint_dir = (
        Path(root) / "checkpoints" /
        model_name / experiment_name
    )

    results_dir = (
        Path(root) / "results" /
        model_name / experiment_name
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_valid_loss = float("inf")
    patience_counter = 0
    history = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    for epoch in range(epochs):

        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_ppl = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        valid_loss, valid_ppl = evaluate(
            model,
            valid_loader,
            criterion,
            device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train PPL: {train_ppl:.2f}"
        )

        print(
            f"Valid Loss: {valid_loss:.4f} | "
            f"Valid PPL: {valid_ppl:.2f}"
        )

        history.append({
            "epoch": epoch + 1,
            "context_length": context_length,
            "batch_size": batch_size,
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "valid_loss": valid_loss,
            "valid_ppl": valid_ppl
        })

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                checkpoint_dir / "best_model.pt"
            )

            print("Đã lưu best model!")

        else:

            patience_counter += 1

            print(
                f"Không cải thiện: "
                f"{patience_counter}/{patience}"
            )

        pd.DataFrame(history).to_csv(
            results_dir / "history.csv",
            index=False
        )

        if patience_counter >= patience:
            print("Early Stopping!")
            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    training_time = time.time() - start_time

    print(
        f"\nTraining time: "
        f"{training_time / 60:.2f} phút"
    )

    return {
        "best_valid_loss": best_valid_loss,
        "best_valid_ppl": math.exp(best_valid_loss),
        "epochs_trained": len(history),
        "training_time_seconds": training_time,
        "checkpoint_path": str(
            checkpoint_dir / "best_model.pt"
        ),
        "results_dir": str(results_dir),
        "history": pd.DataFrame(history)
    }
