
import torch
import torch.nn as nn
import torch.nn.functional as F



# 1. TOKEN EMBEDDING

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, x):
        return self.embedding(x)



# 2. CUSTOM CAUSAL CONVOLUTION 1D


class CausalConv1d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1
    ):
        super().__init__()

        if kernel_size < 1 or dilation < 1:
            raise ValueError(
                "kernel_size và dilation phải >= 1"
            )

        self.kernel_size = kernel_size
        self.dilation = dilation

        # Trọng số của convolution
        self.weight = nn.Parameter(
            torch.empty(
                kernel_size,
                in_channels,
                out_channels
            )
        )

        # Bias của convolution
        self.bias = nn.Parameter(
            torch.zeros(out_channels)
        )

        # Khởi tạo trọng số
        for k in range(kernel_size):
            nn.init.kaiming_uniform_(
                self.weight[k],
                nonlinearity="relu"
            )

    def forward(self, x):

        # x: [B, T, in_channels]
        B, T, C = x.shape

        # Padding chỉ ở bên trái
        left_pad = (
            self.kernel_size - 1
        ) * self.dilation

        # F.pad trên chiều sequence
        x = F.pad(
            x.transpose(1, 2),
            (left_pad, 0)
        ).transpose(1, 2)

        output = x.new_zeros(
            B,
            T,
            self.bias.numel()
        )

        # Tự thực hiện convolution
        for k in range(self.kernel_size):

            start = k * self.dilation

            x_slice = x[
                :,
                start:start + T,
                :
            ]

            # [B, T, C] @ [C, out_channels]
            output = output + (
                x_slice @ self.weight[k]
            )

        output = output + self.bias

        return output



# 3. CNN BLOCK


class CNNBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_dim,
        kernel_size=3,
        dilation=1,
        dropout=0.2
    ):
        super().__init__()

        self.conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x



# 4. STACKED CNN


class StackedCNN(nn.Module):

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        kernel_size=3,
        dilations=(1, 2, 4),
        dropout=0.2
    ):
        super().__init__()

        layers = []

        in_channels = embedding_dim

        for dilation in dilations:

            layers.append(
                CNNBlock(
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

            in_channels = hidden_dim

        self.layers = nn.ModuleList(layers)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x



# 5. CNN LANGUAGE MODEL


class CNNLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=256,
        kernel_size=3,
        dropout=0.2,
        dilations=(1, 2, 4)
    ):
        super().__init__()

        # Embedding
        self.embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        # Các lớp Causal Convolution
        self.cnn = StackedCNN(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout
        )

        # Dự đoán token tiếp theo
        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

        self.receptive_field = (
            1 + (kernel_size - 1) * sum(dilations)
        )

    def forward(self, x):

        # [B, T] -> [B, T, embedding_dim]
        x = self.embedding(x)

        # [B, T, embedding_dim]
        # -> [B, T, hidden_dim]
        x = self.cnn(x)

        # [B, T, hidden_dim]
        # -> [B, T, vocab_size]
        logits = self.fc(x)

        return logits
