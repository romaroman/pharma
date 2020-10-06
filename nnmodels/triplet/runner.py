import torch

from nnmodels import config
from nnmodels.datasets import PharmaPackDatasetTriplet, get_train_validation_data_loaders

from nnmodels.triplet.trainer import fit
from nnmodels.triplet.losses import TripletLoss
from nnmodels.triplet.models import EmbeddingNet, TripletNet


if __name__ == '__main__':
    cuda = torch.cuda.is_available()

    dataset = PharmaPackDatasetTriplet()
    train_loader, test_loader = get_train_validation_data_loaders(dataset)

    margin = 1.
    embedding_net = EmbeddingNet()
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, config.epochs, cuda)
