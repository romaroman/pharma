import torch
from torch import nn
from typing import List

import torchvision


class HashEncoder(nn.Module):

    def __init__(self, base_model: str, descriptor_lengths: List[int]):
        super(HashEncoder, self).__init__()

        pretrained_model = torchvision.models.__dict__[base_model](pretrained=True)

        self.base_model: str = base_model
        self.descriptor_lengths: List[int] = descriptor_lengths
        self.features: nn.Module = nn.Sequential(*list(pretrained_model.children())[:-1])

        self.fully_connected_layers = nn.ModuleList(
            [nn.Linear(pretrained_model.fc.in_features, length) for length in self.descriptor_lengths]
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return [fully_connected(x) for fully_connected in self.fully_connected_layers]
