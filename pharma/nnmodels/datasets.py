import random
from pathlib import Path
from typing import Tuple, List, NoReturn

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import datasets, transforms

from pharma.nnmodels.transforms import get_pipeline_transform


def get_train_validation_data_loaders(
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        test_size: float) -> Tuple[DataLoader, DataLoader]:
    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    split = int(np.floor(test_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = sampler.SubsetRandomSampler(train_idx)
    valid_sampler = sampler.SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, drop_last=True, shuffle=False
    )
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers, drop_last=True, shuffle=False
    )

    return train_loader, test_loader


class PharmaPackDatasetTriplet(Dataset):

    def __init__(self, path: Path) -> NoReturn:
        self.dataset = datasets.ImageFolder(str(path.resolve()))

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List[int]]:
        image_positive, label1 = self.dataset[index]
        negative_label = random.choice(list({cl for cl in self.dataset.classes} - {self.dataset.classes[label1]}))
        negative_index = np.random.choice(np.where(np.asarray(self.dataset.targets) == self.dataset.class_to_idx[negative_label])[0])
        image_negative = self.dataset[negative_index][0]

        # for i in range(5**2):
        #
        #     triplet = np.vstack([
        #         transforms.ToTensor()(image_positive).permute(1, 2, 0).numpy(),
        #         get_pipeline_transform()(image_positive).permute(1, 2, 0).numpy(),
        #         transforms.ToTensor()(image_negative).permute(1, 2, 0).numpy()
        #     ])
        #     # utils.display(triplet)
        #     images.append((triplet * 255).astype(np.uint8))
        #
        # combined = utils.combine_images(images)
        # utils.display(combined)

        return (
                   transforms.ToTensor()(image_positive),
                   get_pipeline_transform()(image_positive),
                   transforms.ToTensor()(image_negative)
               ), []

    def __len__(self) -> int:
        return len(self.dataset)


class PharmaPackDataset(Dataset):

    def __init__(self, path: Path) -> NoReturn:
        self.dataset = datasets.ImageFolder(str(path))
        self.actual_classes = {v: k for k, v in self.dataset.class_to_idx.items()}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        image, label = self.dataset[index]

        return transforms.ToTensor()(image), self.dataset.imgs[index][0]

    def __len__(self) -> int:
        return len(self.dataset)

    def get_actual_class_by_idx(self, index: int) -> str:
        return self.actual_classes[index]
