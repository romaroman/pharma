from typing import Any

import torch
from torch import nn


class ResNet18Hash(nn.Module):

    def __init__(self, pretrained_model: nn.Module, output_size: int):
        super(ResNet18Hash, self).__init__()

        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])

        self.fc1 = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)
