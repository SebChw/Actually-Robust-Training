import torch
import torch.nn as nn
from torch.nn import functional as F


class M5(nn.Module):
    # https://arxiv.org/pdf/1610.00087.pdf
    def __init__(self, sampling_rate=16000, n_classes=36, n_channels=32):
        super().__init__()

        first_conv_size = sampling_rate // 100  # 10ms
        first_stride = first_conv_size // 10

        N_CONV_LAYERS = 4

        channels_dim = [n_channels * (2**i) for i in range(N_CONV_LAYERS)]

        self.first_block = self.get_1d_block(
            1, channels_dim[0], first_conv_size, first_stride
        )
        self.conv_blocks = nn.ModuleList(
            [
                self.get_1d_block(channels_dim[i - 1], channels_dim[i], 3)
                for i in range(1, N_CONV_LAYERS)
            ]
        )

        self.classification_head = nn.Linear(channels_dim[-1], n_classes)

    def get_1d_block(self, in_channels, out_channels, kernel_size, stride=1):
        return nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(4),  # equivalent to 2x2 for images
        )

    def forward(self, X):
        X = self.first_block(X)
        for block in self.conv_blocks:
            X = block(X)

        X = F.avg_pool1d(X, X.shape[-1])[:, :, 0]  # global avg pool

        return self.classification_head(X)


if __name__ == "__main__":
    model = M5()
    X = model(torch.randn((1, 1, 16000)))
    print(X.shape)
