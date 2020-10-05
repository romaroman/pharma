from enums import auto

import utils


class Model(utils.CustomEnum):
    Triplet = auto(),
    Resnet50 = auto(),
    Siamese = auto(),
    SimCLR = auto(),
