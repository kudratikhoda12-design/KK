"""Small CNN for 96x96 histopathology patch binary classification."""
import torch
import torch.nn as nn


class HistoCNN(nn.Module):
    """
    A compact VGG-style CNN trained from scratch (no pretrained weights).
    Four conv blocks (32->64->128->256 channels) + global average pool + FC head.
    `self.features` ends with the last conv block, which Grad-CAM hooks into.
    """

    def __init__(self, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256, pool=False),  # keep spatial size for Grad-CAM
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(c_in, c_out, pool=True):
        layers = [
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.features(x)
        x = self.gap(feat)
        return self.classifier(x)
