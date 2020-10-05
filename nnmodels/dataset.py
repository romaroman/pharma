from pathlib import Path
from PIL import Image
import numpy as np
import cv2 as cv

import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import datasets, transforms

from nnmodels import config
import utils


class GaussianBlur(object):
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class CloudDataset(Dataset):
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True):
        super().__init__()

        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch

    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):
        files = {'red': r_file,
                 'green': g_dir / r_file.name.replace('red', 'green'),
                 'blue': b_dir / r_file.name.replace('red', 'blue'),
                 'nir': nir_dir / r_file.name.replace('red', 'nir'),
                 'gt': gt_dir / r_file.name.replace('red', 'gt')}

        return files

    def __len__(self):
        return len(self.files)

    def open_as_array(self, idx, invert=False, include_nir=False):

        raw_rgb = np.stack([
            np.array(Image.open(self.files[idx]['red'])),
            np.array(Image.open(self.files[idx]['green'])),
            np.array(Image.open(self.files[idx]['blue'])),
        ], axis=2)

        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)

        if invert:
            raw_rgb = raw_rgb.transpose((2, 0, 1))

        # normalize
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)

    def __getitem__(self, idx):

        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_nir=True), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)

        return x, y

    def open_as_pil(self, idx):

        arr = 256 * self.open_as_array(idx)

        return Image.fromarray(arr.astype(np.uint8), 'RGB')

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())

        return s


class Dataset(object):

    def __init__(self):
        self.batch_size = config.batch_size
        self.num_workers = config.dataset_num_workers
        self.valid_size = config.dataset_valid_size
        self.s = config.dataset_s
        self.input_shape = config.dataset_input_shape

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()
        train_dataset = datasets.ImageFolder('/data/500gb/NN', transform=SimCLRDataTransformSingle(data_augment))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=self.input_shape[0]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
            transforms.ToTensor()
        ])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = sampler.SubsetRandomSampler(train_idx)
        valid_sampler = sampler.SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


class SimCLRDataTransformSingle(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return self.transform(sample)
