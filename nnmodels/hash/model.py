import torch
from torch import nn


class ResNet18Hash(nn.Module):

    def __init__(self, pretrained_model):
        super(ResNet18Hash, self).__init__()

        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])

        self.fc1 = nn.Linear(2048, 256)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 256)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        x1 = self.fc1(x)
        # x2 = self.fc2(x1)
        # x3 = self.fc3(x2)

        return x1  # , x2, x3
