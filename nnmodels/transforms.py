import cv2 as cv
import numpy as np
from torchvision import transforms


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
    data_transforms = transforms.Compose([
        transforms.RandomApply(
            [transforms.RandomResizedCrop(size=256, scale=(0.75, 1.25), ratio=(0.75, 1.25))],
            p=0.25
        ),
        # transforms.RandomApply([transforms.RandomAffine(degrees=(0, 0), scale=(0.75, 1.25))], p=0.2),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.3),
        transforms.RandomGrayscale(p=0.25),
        GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.2, scale=(0.05, 0.01)),
    ])

    return data_transforms
