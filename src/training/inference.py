
import re
import torch


def tokenize_text(text):
    # Phải giống tokenizer trong preprocessing
    return re.findall(
        r"\w+(?:['’]\w+)*|[^\w\s]",
        text,
        flags=re.UNICODE
    )


def encode_text(text, word2idx):
    tokens = tokenize_text(text)
    unk_id = word2idx["<unk>"]

    return [
        word2idx.get(token, unk_id)
        for token in tokens
    ]


@torch.no_grad()
def predict_next_token(
    model,
    text,
    word2idx,
    idx2word,
    device,
    context_length=64,
    top_k=5
):
    model.eval()

    ids = encode_text(text, word2idx)

    if not ids:
        raise ValueError("Vui lòng nhập ít nhất một token.")

    x = torch.tensor(
        [ids[-context_length:]],
        dtype=torch.long,
        device=device
    )

    logits = model(x)

    # Điểm dự đoán tại vị trí cuối
    next_logits = logits[0, -1, :]

    probs = torch.softmax(next_logits, dim=-1)

    # Không sinh PAD hoặc UNK
    for token in ["<pad>", "<unk>"]:
        if token in word2idx:
            probs[word2idx[token]] = 0

    probs = probs / probs.sum()

    values, indices = torch.topk(
        probs,
        k=min(top_k, probs.numel())
    )

    predictions = []

    for probability, token_id in zip(
        values.tolist(),
        indices.tolist()
    ):
        predictions.append({
            "token": idx2word[token_id],
            "probability": probability
        })

    return predictions


@torch.no_grad()
def generate_text(
    model,
    prompt,
    word2idx,
    idx2word,
    device,
    context_length=64,
    max_new_tokens=30,
    temperature=1.0,
    top_k=20
):
    model.eval()

    if temperature <= 0:
        raise ValueError("temperature phải lớn hơn 0.")

    generated_ids = encode_text(prompt, word2idx)

    if not generated_ids:
        raise ValueError("Prompt không được để trống.")

    for _ in range(max_new_tokens):

        x = torch.tensor(
            [generated_ids[-context_length:]],
            dtype=torch.long,
            device=device
        )

        logits = model(x)
        next_logits = logits[0, -1, :] / temperature

        # Không sinh token padding / unknown
        for token in ["<pad>", "<unk>"]:
            if token in word2idx:
                next_logits[word2idx[token]] = -float("inf")

        k = min(top_k, next_logits.numel())

        if k > 0:
            values, indices = torch.topk(next_logits, k)
            probs = torch.softmax(values, dim=-1)

            sampled = torch.multinomial(probs, 1)
            next_id = indices[sampled].item()
        else:
            next_id = torch.argmax(next_logits).item()

        if next_id == word2idx["<eos>"]:
            break

        generated_ids.append(next_id)

    tokens = [
        idx2word[idx]
        for idx in generated_ids
        if idx != word2idx["<eos>"]
    ]

    return " ".join(tokens)
