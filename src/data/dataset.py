import json
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# Đường dẫn thư mục gốc dự án
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"


# ============================
# 1. LANGUAGE MODELING DATASET
# ============================

class LanguageModelingDataset(Dataset):

    def __init__(self, token_ids, context_length=64):

        self.tokens = torch.as_tensor(
            token_ids,
            dtype=torch.long
        )

        self.context_length = context_length

    def __len__(self):

        return max(
            0,
            (len(self.tokens) - 1)
            // self.context_length
        )

    def __getitem__(self, idx):

        start = idx * self.context_length
        end = start + self.context_length

        # Input
        x = self.tokens[start:end]

        # Target dịch sang phải 1 token
        y = self.tokens[start + 1:end + 1]

        return x, y


# ============================
# 2. LOAD PROCESSED DATA
# ============================

def load_processed_data():

    train_tokens = torch.load(
        DATA_DIR / "train.pt",
        weights_only=True
    )

    valid_tokens = torch.load(
        DATA_DIR / "valid.pt",
        weights_only=True
    )

    test_tokens = torch.load(
        DATA_DIR / "test.pt",
        weights_only=True
    )

    with open(
        DATA_DIR / "word2idx.json",
        "r",
        encoding="utf-8"
    ) as f:
        word2idx = json.load(f)

    with open(
        DATA_DIR / "config.json",
        "r",
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    return (
        train_tokens,
        valid_tokens,
        test_tokens,
        word2idx,
        config
    )


# ============================
# 3. CREATE DATALOADERS
# ============================

def create_dataloaders(
    context_length=None,
    batch_size=None
):
    (
        train_tokens,
        valid_tokens,
        test_tokens,
        word2idx,
        config
    ) = load_processed_data()

    if context_length is None:
        context_length = config["context_length"]

    if batch_size is None:
        batch_size = config["batch_size"]

    if context_length <= 0 or batch_size <= 0:
        raise ValueError("context_length và batch_size phải > 0")

    train_dataset = LanguageModelingDataset(
        train_tokens,
        context_length
    )

    valid_dataset = LanguageModelingDataset(
        valid_tokens,
        context_length
    )

    test_dataset = LanguageModelingDataset(
        test_tokens,
        context_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    vocab_size = len(word2idx)

    return (
        train_loader,
        valid_loader,
        test_loader,
        vocab_size
    )


# ============================
# 4. TEST DATASET
# ============================

if __name__ == "__main__":

    (
        train_loader,
        valid_loader,
        test_loader,
        vocab_size
    ) = create_dataloaders()

    X, Y = next(iter(train_loader))

    print("Vocabulary size:", vocab_size)

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(valid_loader))
    print("Test batches:", len(test_loader))

    print("Input shape:", X.shape)
    print("Target shape:", Y.shape)

    print("Input:", X[0][:10])
    print("Target:", Y[0][:10])

    assert torch.equal(X[:, 1:], Y[:, :-1])

    print("Dataset kiểm tra thành công!")