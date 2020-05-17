import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
from typing import List
import matplotlib as mpl


mpl.rc('image', cmap='Greys_r')


def show_image_as_plot(image):
    figsize_h = int(image.shape[0] / 100)
    figsize_w = int(image.shape[1] / 100)
    fig = plt.figure(figsize=(figsize_w, figsize_h), frameon=False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.tight_layout()
    fig.patch.set_visible(False)

    ax = fig.add_subplot(111)
    plt.axis('off')
    ax.imshow(image, interpolation="nearest")
    ax.autoscale(False)
    plt.show()


def show_image_as_window(image: np.ndarray, title: str = ""):
    default_width = 720
    ratio = default_width / image.shape[0]
    h = int(image.shape[1] * ratio)

    image = cv.resize(image, (h, default_width), interpolation=cv.INTER_NEAREST)

    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyWindow(title)


def load_images(src_folder: str) -> List[str]:
    return glob.glob(src_folder + "\\*.png")
