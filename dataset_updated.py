"""Shared next-token language-modeling Dataset/DataLoader for word and BPE."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"


class LanguageModelingDataset(Dataset):
    def __init__(self, token_ids, context_length=64):
        if context_length <= 0:
            raise ValueError("context_length phải > 0")
        # Giữ dữ liệu ở định dạng gốc (int32/memmap) thay vì nhân đôi RAM.
        self.tokens = token_ids
        self.context_length = context_length

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.context_length)

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length
        # Chỉ đổi sang int64 với mẫu đang lấy, phù hợp với nn.Embedding.
        x = torch.as_tensor(np.array(self.tokens[start:end], dtype=np.int64, copy=True))
        y = torch.as_tensor(np.array(self.tokens[start + 1:end + 1], dtype=np.int64, copy=True))
        return x, y


def load_processed_data(tokenizer_type="word"):
    if tokenizer_type not in ("word", "bpe"):
        raise ValueError("tokenizer_type phải là 'word' hoặc 'bpe'")

    data_dir = DATA_DIR / "bpe" if tokenizer_type == "bpe" else DATA_DIR
    with open(data_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    def load_split(stem):
        npy_path = data_dir / f"{stem}.npy"
        pt_path = data_dir / f"{stem}.pt"
        if npy_path.exists():
            return np.load(npy_path, mmap_mode="r", allow_pickle=False)
        if tokenizer_type == "word" and pt_path.exists():
            # Legacy word-level .pt; nếu lưu bằng list Python rất lớn,
            # torch.load có thể tốn RAM. Nên chuyển sang .npy.
            return torch.load(pt_path, map_location="cpu", weights_only=True)
        raise FileNotFoundError(f"Không tìm thấy {npy_path} (hoặc file .pt word-level)")

    train_tokens = load_split("train")
    valid_tokens = load_split("valid")
    test_tokens = load_split("test")

    if tokenizer_type == "word":
        with open(data_dir / "word2idx.json", encoding="utf-8") as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
    else:
        vocab = None  # BPE dùng tokenizer.json, không dùng word2idx.json.
        vocab_size = int(config["vocab_size"])

    return train_tokens, valid_tokens, test_tokens, vocab, config


def create_dataloaders(context_length=None, batch_size=None, tokenizer_type="word"):
    train_tokens, valid_tokens, test_tokens, vocab, config = load_processed_data(tokenizer_type)
    context_length = config["context_length"] if context_length is None else context_length
    batch_size = config["batch_size"] if batch_size is None else batch_size
    if context_length <= 0 or batch_size <= 0:
        raise ValueError("context_length và batch_size phải > 0")

    datasets = [
        LanguageModelingDataset(tokens, context_length)
        for tokens in (train_tokens, valid_tokens, test_tokens)
    ]
    if any(len(ds) == 0 for ds in datasets):
        raise ValueError("Một tập dữ liệu quá ngắn so với context_length")

    loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=(i == 0), num_workers=0)
        for i, ds in enumerate(datasets)
    ]
    vocab_size = len(vocab) if tokenizer_type == "word" else int(config["vocab_size"])
    return *loaders, vocab_size


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", choices=["word", "bpe"], default="word")
    args = parser.parse_args()
    train_loader, valid_loader, test_loader, vocab_size = create_dataloaders(
        tokenizer_type=args.tokenizer
    )
    X, Y = next(iter(train_loader))
    print("Tokenizer:", args.tokenizer)
    print("Vocabulary size:", vocab_size)
    print("Batches:", len(train_loader), len(valid_loader), len(test_loader))
    print("Input shape:", X.shape, "Target shape:", Y.shape)
    assert torch.equal(X[:, 1:], Y[:, :-1])
    print("Dataset kiểm tra thành công!")
