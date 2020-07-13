import matplotlib as mpl
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import glob
import logging

from typing import List


mpl.rc('image', cmap='Greys_r')


def display(image):
    image_disp = np.copy(image)
    if len(image.shape) == 3:
        h, w, _ = image.shape
        image_disp = cv.cvtColor(image_disp, cv.COLOR_BGR2RGB)
    elif len(image.shape) == 2:
        h, w = image.shape

    fig = plt.figure(figsize=(max(int(w / 100), 3), max(int(h / 100), 3)), frameon=False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.tight_layout()
    fig.patch.set_visible(False)

    ax = fig.add_subplot(111)
    plt.axis('off')
    ax.imshow(image_disp, interpolation="nearest")
    ax.autoscale(False)
    plt.show()


def show_image_as_window(image: np.ndarray, title: str = ""):
    default_width = 720
    ratio = default_width / image.shape[0]
    h = int(image.shape[1] * ratio)

    #  image = cv.resize(image, (h, default_width), interpolation=cv.INTER_NEAREST)

    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyWindow(title)


def get_logger(name: str, level: int = logging.INFO):
    log = logging.Logger(name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    return log
