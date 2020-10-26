import torch
from torch import nn


class HashEncoder(nn.Module):

    def __init__(self, pretrained_model: nn.Module, output_size: int):
        super(HashEncoder, self).__init__()

        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.fc = nn.Linear(pretrained_model.fc.in_features, output_size)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
