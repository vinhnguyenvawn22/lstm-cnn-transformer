
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================
# 1. TOKEN EMBEDDING
# ======================================

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_size
        )

    def forward(self, x):
        return self.embedding(x)


# ======================================
# 2. POSITIONAL ENCODING
# ======================================

class PositionalEncoding(nn.Module):

    def __init__(self, hidden_size, max_len=4096):
        super().__init__()

        pe = torch.zeros(max_len, hidden_size)

        position = torch.arange(
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0, hidden_size, 2,
                dtype=torch.float32
            ) * (-math.log(10000.0) / hidden_size)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term[:hidden_size // 2]
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
            persistent=False
        )

    def forward(self, x):

        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            raise ValueError(
                "Sequence length vượt quá max_len"
            )

        return x + self.pe[:, :seq_len, :]


# ======================================
# 3. CAUSAL MASK
# ======================================

def create_causal_mask(seq_len, device):

    # True nghĩa là vị trí bị chặn
    mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            device=device,
            dtype=torch.bool
        ),
        diagonal=1
    )

    return mask


# ======================================
# 4. MULTI-HEAD SELF-ATTENTION
# ======================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        dropout=0.2
    ):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.scale = self.head_dim ** -0.5

        # Tự xây dựng phép chiếu Q, K, V
        self.q_proj = nn.Linear(
            hidden_size, hidden_size
        )

        self.k_proj = nn.Linear(
            hidden_size, hidden_size
        )

        self.v_proj = nn.Linear(
            hidden_size, hidden_size
        )

        self.o_proj = nn.Linear(
            hidden_size, hidden_size
        )

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        B, T, D = x.shape

        # Tạo Query, Key, Value
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, D] -> [B, heads, T, head_dim]
        q = q.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        ) * self.scale

        # Chặn token tương lai
        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0),
                float("-inf")
            )

        # Attention weights
        attention = F.softmax(
            scores,
            dim=-1
        )

        attention = self.attn_dropout(attention)

        # Nhân với Value
        output = torch.matmul(
            attention,
            v
        )

        # Ghép các head
        output = output.transpose(
            1, 2
        ).contiguous().reshape(B, T, D)

        return self.o_proj(output)


# ======================================
# 5. FEED FORWARD NETWORK
# ======================================

class FeedForward(nn.Module):

    def __init__(
        self,
        hidden_size,
        ffn_dim,
        dropout=0.2
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            hidden_size,
            ffn_dim
        )

        self.fc2 = nn.Linear(
            ffn_dim,
            hidden_size
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.fc1(x)

        x = F.gelu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ======================================
# 6. TRANSFORMER BLOCK
# ======================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        ffn_dim,
        dropout=0.2
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            hidden_size,
            num_heads,
            dropout
        )

        self.ffn = FeedForward(
            hidden_size,
            ffn_dim,
            dropout
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):

        # Attention + Residual Connection
        h = self.norm1(x)

        x = x + self.dropout1(
            self.attention(h, mask)
        )

        # Feed Forward + Residual Connection
        h = self.norm2(x)

        x = x + self.dropout2(
            self.ffn(h)
        )

        return x


# ======================================
# 7. TRANSFORMER LANGUAGE MODEL
# ======================================

class TransformerLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        num_heads=4,
        num_layers=2,
        ffn_dim=1024,
        dropout=0.2,
        max_seq_len=256
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        self.position = PositionalEncoding(
            embedding_dim,
            max_len=max_seq_len
        )

        self.dropout = nn.Dropout(dropout)

        # Xếp chồng các Transformer Block
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=embedding_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        B, T = x.shape

        if T > self.max_seq_len:
            raise ValueError(
                f"Context {T} vượt quá "
                f"max_seq_len={self.max_seq_len}"
            )

        # Token Embedding
        x = self.embedding(x)

        # Positional Encoding
        x = self.position(x)

        x = self.dropout(x)

        # Causal Mask
        mask = create_causal_mask(
            T,
            x.device
        )

        # Transformer Blocks
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        # Dự đoán token tiếp theo
        logits = self.fc(x)

        return logits
