import datetime
import os
import logging
import warnings
from pathlib import Path

from typing import NoReturn, List, Union, Tuple

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
pretty_print = lambda obj: print(pretty(obj))


def display(image: np.ndarray, figsize: Union[Tuple[int, int], None] = None) -> NoReturn:
    image_display = np.copy(image)

    if len(image.shape) == 3:
        h, w, _ = image.shape
        image_display = cv.cvtColor(image_display, cv.COLOR_BGR2RGB)
    elif len(image.shape) == 2:
        h, w = image.shape

    if not figsize:
        figsize = (max(int(w / 100), 3), max(int(h / 100), 3))

    fig = plt.figure(figsize=figsize, frameon=False)
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


def setup_logger(name: str, level: int, filename: str):
    format_str = '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s'

    log = logging.Logger(name)
    formatter = logging.Formatter(format_str)
    logging.basicConfig(level=level, format=format_str)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    fh = logging.FileHandler(filename)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    log.addHandler(fh)


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


def suppress_warnings() -> NoReturn:
    warnings.filterwarnings('ignore')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        warnings.filterwarnings('ignore', r'Creating an ndarray from ragged nested sequences')
        warnings.filterwarnings('ignore', r'invalid value encountered in double_scalars')
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        warnings.filterwarnings('ignore', r'SIFT_create DEPRECATED')


def get_str_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")


def create_parent_dirs(path: Path) -> NoReturn:
    os.makedirs(str(path.parent.resolve()), exist_ok=True)


def add_text(image: np.ndarray, text: str, scale: int = 2) -> np.ndarray:
    return cv.putText(
        img=np.copy(image), text=text, org=(25, 25),
        fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=scale, color=(0, 255, 0), thickness=2
    )
