
import torch
import torch.nn as nn


# =====================================
# 1. TOKEN EMBEDDING
# =====================================

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

    def forward(self, x):
        return self.embedding(x)


# =====================================
# 2. CUSTOM LSTM CELL
# =====================================

class LSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input -> 4 gates
        self.W_x = nn.Linear(
            input_dim,
            4 * hidden_dim,
            bias=True
        )

        # Hidden state -> 4 gates
        self.W_h = nn.Linear(
            hidden_dim,
            4 * hidden_dim,
            bias=False
        )

    def forward(self, x_t, h_prev, c_prev):

        # Tính đồng thời giá trị cho 4 cổng
        gates = (
            self.W_x(x_t)
            + self.W_h(h_prev)
        )

        # Chia thành 4 phần
        f, i, g, o = gates.chunk(4, dim=-1)

        # Forget Gate
        f = torch.sigmoid(f)

        # Input Gate
        i = torch.sigmoid(i)

        # Candidate Cell
        g = torch.tanh(g)

        # Output Gate
        o = torch.sigmoid(o)

        # Cập nhật Cell State
        c_t = (
            f * c_prev
            + i * g
        )

        # Cập nhật Hidden State
        h_t = (
            o * torch.tanh(c_t)
        )

        return h_t, c_t


# =====================================
# 3. CUSTOM LSTM LAYER
# =====================================

class LSTMLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.cell = LSTMCell(
            input_dim,
            hidden_dim
        )

    def forward(self, x):

        B, T, D = x.shape

        h_t = x.new_zeros(
            B, self.hidden_dim
        )

        c_t = x.new_zeros(
            B, self.hidden_dim
        )

        outputs = []

        # Xử lý tuần tự từng token
        for t in range(T):

            x_t = x[:, t, :]

            h_t, c_t = self.cell(
                x_t,
                h_t,
                c_t
            )

            outputs.append(h_t)

        # [B, T, hidden_dim]
        output = torch.stack(
            outputs,
            dim=1
        )

        return output, (h_t, c_t)


# =====================================
# 4. STACKED LSTM
# =====================================

class StackedLSTM(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):

            layer_input_dim = (
                input_dim
                if layer_idx == 0
                else hidden_dim
            )

            self.layers.append(
                LSTMLayer(
                    layer_input_dim,
                    hidden_dim
                )
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        for idx, layer in enumerate(self.layers):

            x, (h_t, c_t) = layer(x)

            # Dropout giữa các LSTM layer
            if idx < len(self.layers) - 1:
                x = self.dropout(x)

        return x


# =====================================
# 5. LSTM LANGUAGE MODEL
# =====================================

class LSTMLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        # Embedding
        self.embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        # LSTM tự xây dựng
        self.lstm = StackedLSTM(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Dropout trước output
        self.dropout = nn.Dropout(dropout)

        # Dự đoán token tiếp theo
        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):

        # [B, T] -> [B, T, embedding_dim]
        x = self.embedding(x)

        # [B, T, embedding_dim]
        # -> [B, T, hidden_dim]
        x = self.lstm(x)

        x = self.dropout(x)

        # [B, T, hidden_dim]
        # -> [B, T, vocab_size]
        logits = self.fc(x)

        return logits
