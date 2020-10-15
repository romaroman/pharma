import torch
from torch import nn
import torchvision

from nnmodels.resnet import cresnet
from nnmodels import config
from nnmodels.datasets import PharmaPackDatasetTriplet, get_train_validation_data_loaders

from nnmodels.triplet import trainer
from nnmodels.triplet.losses import TripletLoss
from nnmodels.triplet.models import EmbeddingNet, TripletNet


if __name__ == '__main__':
    cuda = torch.cuda.is_available()

    dataset = PharmaPackDatasetTriplet()
    train_loader, test_loader = get_train_validation_data_loaders(dataset)

    margin = 1.
    embedding_net = EmbeddingNet()
    resnet18 = cresnet.resnet18(pretrained=False, num_classes=512)

    model = TripletNet(resnet18)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    trainer.fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, config.epochs, cuda)
    # model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    # trainer.test_epoch(test_loader, model, loss_fn, cuda, list())
