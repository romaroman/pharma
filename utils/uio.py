import datetime
import os
import logging
import warnings
from pathlib import Path

from typing import NoReturn, List, Union, Any

import cv2 as cv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rc('image', cmap='Greys_r')


class Formatter(object):
    def __init__(self):
        self.types = {}
        self.htchar = '\t'
        self.lfchar = '\n'
        self.indent = 0
        self.set_formatter(object, self.__class__.format_object)
        self.set_formatter(dict, self.__class__.format_dict)
        self.set_formatter(list, self.__class__.format_list)
        self.set_formatter(tuple, self.__class__.format_tuple)

    def set_formatter(self, obj, callback):
        self.types[obj] = callback

    def __call__(self, value, **args):
        for key in args:
            setattr(self, key, args[key])
        formatter = self.types[type(value) if type(value) in self.types else object]
        return formatter(self, value, self.indent)

    def format_object(self, value, indent):
        return repr(value)

    def format_dict(self, value, indent):
        items = [
            self.lfchar + self.htchar * (indent + 1) + repr(key) + ': ' +
            (self.types[type(value[key]) if type(value[key]) in self.types else object])(self, value[key], indent + 1)
            for key in value
        ]
        return '{%s}' % (','.join(items) + self.lfchar + self.htchar * indent)

    def format_list(self, value, indent):
        items = [
            self.lfchar + self.htchar * (indent + 1) + (self.types[type(item) if type(item) in self.types else object])(self, item, indent + 1)
            for item in value
        ]
        return '[%s]' % (','.join(items) + self.lfchar + self.htchar * indent)

    def format_tuple(self, value, indent):
        items = [
            self.lfchar + self.htchar * (indent + 1) + (self.types[type(item) if type(item) in self.types else object])(self, item, indent + 1)
            for item in value
        ]
        return '(%s)' % (','.join(items) + self.lfchar + self.htchar * indent)


pretty = Formatter()


def display(image: np.ndarray) -> NoReturn:
    image_display = np.copy(image)

    if len(image.shape) == 3:
        h, w, _ = image.shape
        image_display = cv.cvtColor(image_display, cv.COLOR_BGR2RGB)
    elif len(image.shape) == 2:
        h, w = image.shape

    fig = plt.figure(figsize=(max(int(w / 100), 3), max(int(h / 100), 3)), frameon=False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.tight_layout()
    fig.patch.set_visible(False)

    ax = fig.add_subplot(111)
    plt.axis('off')
    ax.imshow(image_display, interpolation="nearest")
    ax.autoscale(False)
    plt.show()


def show_image_as_window(image: np.ndarray, title: str = "") -> NoReturn:
    default_width = 720

    ratio = default_width / image.shape[0]
    h = int(image.shape[1] * ratio)

    image = cv.resize(image, (h, default_width), interpolation=cv.INTER_NEAREST)

    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyWindow(title)


def setup_logger(name: str, level: int):
    log = logging.Logger(name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    log.addHandler(ch)


def combine_images(images: List[np.ndarray]) -> Union[np.ndarray, None]:
    shape = images[0].shape
    for image in images[1:]:
        if image.shape != shape:
            return None

    amount = len(images)
    side = int(np.sqrt(amount))

    if not side == np.sqrt(amount):
        side = side + 1

    image_result = None
    for i in range(0, side):
        images_selected = images[i * side: (i + 1) * side]

        while len(images_selected) < side:
            images_selected.append(np.zeros_like(images[0], dtype=np.uint8))

        image_row = np.hstack(images_selected).astype(np.uint8)

        if image_result is None:
            image_result = image_row
        else:
            image_result = np.vstack([image_result, image_row])

    return image_result


def init_path(path_str: str) -> Path:
    path = Path(path_str)

    if not path.exists():
        os.makedirs(str(path.resolve()))
    return path.absolute()


def suppress_warnings() -> NoReturn:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice.')


def get_str_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
