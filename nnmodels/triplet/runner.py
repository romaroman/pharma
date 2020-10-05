import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim

from trainer import fit
from models import EmbeddingNet, ClassificationNet, SiameseNet, TripletNet
from metrics import AccumulatedAccuracyMetric, AverageNonzeroTripletsMetric
from losses import ContrastiveLoss, OnlineContrastiveLoss, TripletLoss, OnlineTripletLoss
from datasets import TripletMNIST, BalancedBatchSampler, SiameseMNIST
# from .helpers import AllPositivePairSelector, HardNegativePairSelector, AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


if __name__ == '__main__':
    cuda = torch.cuda.is_available()

    mean, std = 0.1307, 0.3081

    train_dataset = MNIST('../data/MNIST', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((mean,), (std,))
                          ]))
    test_dataset = MNIST('../data/MNIST', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((mean,), (std,))
                         ]))
    n_classes = 10

    mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

    batch_size = 256
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    # embedding_net = EmbeddingNet()
    # model = ClassificationNet(embedding_net, n_classes=n_classes)
    # if cuda:
    #     model.cuda()
    # loss_fn = torch.nn.NLLLoss()
    # lr = 1e-2
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    # n_epochs = 20
    # log_interval = 50
    #
    # fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
    #     metrics=[AccumulatedAccuracyMetric()])
    #
    # train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
    # plot_embeddings(train_embeddings_baseline, train_labels_baseline)
    # val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
    # plot_embeddings(val_embeddings_baseline, val_labels_baseline)
    #
    # siamese_train_dataset = SiameseMNIST(train_dataset)  # Returns pairs of images and target same/different
    # siamese_test_dataset = SiameseMNIST(test_dataset)
    # batch_size = 128
    # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    # siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True,
    #                                                    **kwargs)
    # siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False,
    #                                                   **kwargs)
    #
    # margin = 1.
    # embedding_net = EmbeddingNet()
    # model = SiameseNet(embedding_net)
    # if cuda:
    #     model.cuda()
    # loss_fn = ContrastiveLoss(margin)
    # lr = 1e-3
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    # n_epochs = 20
    # log_interval = 100
    #
    # fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    #
    # train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
    # plot_embeddings(train_embeddings_cl, train_labels_cl)
    # val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
    # plot_embeddings(val_embeddings_cl, val_labels_cl)

    triplet_train_dataset = TripletMNIST(train_dataset)
    triplet_test_dataset = TripletMNIST(test_dataset)
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True,
                                                       **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False,
                                                      **kwargs)

    margin = 1.
    embedding_net = EmbeddingNet()
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 100

    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings_tl, train_labels_tl)
    val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
    plot_embeddings(val_embeddings_tl, val_labels_tl)

    # train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=10, n_samples=25)
    # test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=10, n_samples=25)
    #
    # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    # online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    # online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)
    #
    # margin = 1.
    # embedding_net = EmbeddingNet()
    # model = embedding_net
    # if cuda:
    #     model.cuda()
    # loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
    # lr = 1e-3
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    # n_epochs = 20
    # log_interval = 50
    #
    # fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    #
    # train_embeddings_ocl, train_labels_ocl = extract_embeddings(train_loader, model)
    # plot_embeddings(train_embeddings_ocl, train_labels_ocl)
    # val_embeddings_ocl, val_labels_ocl = extract_embeddings(test_loader, model)
    # plot_embeddings(val_embeddings_ocl, val_labels_ocl)
    #
    # train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=10, n_samples=25)
    # test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=10, n_samples=25)
    #
    # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    # online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    # online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)
    #
    # margin = 1.
    # embedding_net = EmbeddingNet()
    # model = embedding_net
    # if cuda:
    #     model.cuda()
    # loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
    # lr = 1e-3
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    # n_epochs = 20
    # log_interval = 50
    #
    # fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
    #     metrics=[AverageNonzeroTripletsMetric()])
    #
    # train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model)
    # plot_embeddings(train_embeddings_otl, train_labels_otl)
    # val_embeddings_otl, val_labels_otl = extract_embeddings(test_loader, model)
    # plot_embeddings(val_embeddings_otl, val_labels_otl)
    #
    # display_emb_online, display_emb, display_label_online, display_label = val_embeddings_otl, val_embeddings_tl, val_labels_otl, val_labels_tl
    # x_lim = (np.min(display_emb_online[:, 0]), np.max(display_emb_online[:, 0]))
    # y_lim = (np.min(display_emb_online[:, 1]), np.max(display_emb_online[:, 1]))
    # plot_embeddings(display_emb, display_label, x_lim, y_lim)
    # plot_embeddings(display_emb_online, display_label_online, x_lim, y_lim)
    #
    # x_lim = (np.min(train_embeddings_ocl[:, 0]), np.max(train_embeddings_ocl[:, 0]))
    # y_lim = (np.min(train_embeddings_ocl[:, 1]), np.max(train_embeddings_ocl[:, 1]))
    # plot_embeddings(train_embeddings_cl, train_labels_cl, x_lim, y_lim)
    # plot_embeddings(train_embeddings_ocl, train_labels_ocl, x_lim, y_lim)