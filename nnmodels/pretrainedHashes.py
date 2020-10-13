from redis import Redis

from nearpy import Engine
from nearpy.storage import RedisStorage

import torch
from torch import nn
import torchvision

from nnmodels.datasets import PharmaPackDataset, get_train_validation_data_loaders
from nnmodels import config
import utils


class HashingNN(nn.Module):

    def __init__(self, pretrained_model):
        super(HashingNN, self).__init__()

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


if __name__ == '__main__':
    resnet50 = torchvision.models.resnet50(pretrained=True, num_classes=1000)
    model = HashingNN(resnet50)
    cuda = torch.cuda.is_available()

    redis_db = Redis(host='localhost', port=6379, db=0)
    redis_db.flushall()
    engine = Engine(256, storage=RedisStorage(redis_db))

    train_loader, test_loader = get_train_validation_data_loaders(PharmaPackDataset())

    img_index = 0
    with torch.no_grad():
        model.eval()
        val_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == 3:
                break
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            v256 = model(*data) # v1024, v512,
            for vector, tc in zip(v256.numpy(), target.numpy()):
                engine.store_vector(v=vector, data=f"{str(tc)}_{utils.zfill_n(img_index, 15)}")
                img_index += 1
