import torch
from torch import nn
from typing import List, Dict

import torchvision


class HashEncoder(nn.Module):

    def __init__(self, base_model: str, output_sizes: List[int]):
        super(HashEncoder, self).__init__()

        pretrained_model = torchvision.models.__dict__[base_model](pretrained=True)
        self.base_model: str = base_model
        self.features: nn.Module = nn.Sequential(*list(pretrained_model.children())[:-1])

        self.fcs: Dict[int, nn.Linear] = dict()
        for output_size in output_sizes:
            self.fcs[output_size] = nn.Linear(pretrained_model.fc.in_features, output_size)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return dict([(size, fc(x)) for size, fc in self.fcs.items()])