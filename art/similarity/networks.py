from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import Conv2dNormActivation


class EfficientWordNet(nn.Module):
    def __init__(self):
        super().__init__()

        N_BLOCKS_FROM_EFFNET = 5
        features_blocks = models.efficientnet_b0().features
        self.efnet_part = nn.Sequential(
            *[features_blocks[i] for i in range(N_BLOCKS_FROM_EFFNET)]
        )

        # Our input is one channel spectrogram that's why we change first conv layer
        norm_layer = None
        firstconv_output_channels = 32
        self.efnet_part[0] = Conv2dNormActivation(
            1,
            firstconv_output_channels,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=nn.SiLU,
        )

        # They claim that input is (1,98,64) spectrogram (98 filterbanks and 64 time steps)
        # Output from 4th block of efficient net yields (80, 7, 4) Quite small image, but they use more convolutions
        self.our_part = nn.Sequential(
            nn.Conv2d(80, 32, 3, padding="same"),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(
                2
            ),  # They're not using ReLU in original implementation. This is actually okay with just max Pooling
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.Flatten(),  # Output at this stage is 384
            nn.Linear(384, 128),
        )

    def forward(self, X):
        X = self.efnet_part(X)
        X = self.our_part(X)
        return F.normalize(X)  # We force this embeddings to lie in a unit hypersphere
