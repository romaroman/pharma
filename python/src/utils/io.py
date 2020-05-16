import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
from typing import List
import matplotlib as mpl


mpl.rc('image', cmap='Greys_r')


def show_image_as_plot(image, figsize=12):
    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.add_subplot(111)
    ax.imshow(image)
    plt.show()


def show_image_as_windows(image: np.ndarray, title: str = ""):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyWindow(title)


def load_images(src_folder: str) -> List[str]:
    return glob.glob(src_folder + "\\*.png")
