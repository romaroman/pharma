from enum import auto

from utils import CustomEnum


class Model(CustomEnum):
    Triplet = auto(),
    Resnet50 = auto(),
    Siamese = auto(),
    SimCLR = auto(),
