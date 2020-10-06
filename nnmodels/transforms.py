import cv2 as cv
import numpy as np
from torchvision import transforms

from nnmodels import config


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


def get_pipeline_transform() -> transforms.Compose:
    s = config.dataset_s * 0.8

    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=config.dataset_input_shape[0]),
        transforms.RandomApply([transforms.RandomAffine(scale=(0.5, 2.0), degrees=(15, 345))], p=0.2),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(s, s, s, 1-s)], p=0.3),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * config.dataset_input_shape[0])),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.05, 0.2)),
    ])

    return data_transforms
