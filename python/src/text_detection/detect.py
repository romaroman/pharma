import cv2 as cv
import numpy as np

import utils


from typing import List, Tuple

import glob
import os
import shutil
import pathlib
import skimage
import skimage.measure

from scipy.sparse.csgraph import connected_components

KERNEL_SIZE = 5


def detect_text_regions(image_gray: np.ndarray) -> List[Tuple[float, float]]:

    def is_prop_valid(prop: skimage.measure._regionprops._RegionProperties) -> bool:
        if prop.minor_axis_length > 20:
            if prop.solidity > 0.33:
                if prop.area > 150:
                    return True

        return False

    global KERNEL_SIZE
    image_filtered = utils.std_filter(image_gray, KERNEL_SIZE)

    _, image_bw = cv.threshold(image_filtered, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    image_dilated = cv.dilate(image_bw, np.ones((KERNEL_SIZE, KERNEL_SIZE)))

    image_cleared = utils.clear_borders(image_dilated)

    contours, _ = cv.findContours(image_cleared, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    image_filled = np.zeros_like(image_dilated)
    cv.drawContours(image_filled, contours, -1, 255, -1)

    n_l, image_labeled = cv.connectedComponents(image_filled)

    props = skimage.measure.regionprops(image_labeled)

    valid_props = list(filter(is_prop_valid, props))

    scale = 0.25
    line_length = int(200 * scale)
    new_size = (int(image_filled.shape[0] * scale), int(image_filled.shape[1] * scale))
    image_resized = cv.resize(image_filled, new_size, interpolation=cv.INTER_NEAREST)

    line_rotation_angles = [i for i in range(-90, 90, 15)]

    counts = []
    for line_rotation_angle in line_rotation_angles:
        structural_element = utils.morph_line(line_length, line_rotation_angle)
        image_linearly_morphed = cv.dilate(image_resized, structural_element)
        counts.append(cv.countNonZero(image_linearly_morphed))
        # utils.show_image_as_plot(image_linearly_morphed)

    k = np.asarray(counts).argmin()

    structural_element = utils.morph_line(int(line_length / 2), line_rotation_angles[k])
    image_linearly_morphed = cv.dilate(image_resized, structural_element)

    _, image_labeled = cv.connectedComponents(image_linearly_morphed)

    for k in range(image_labeled.max()):
        image_blob = (image_labeled == k).astype(np.uint8) * 255
        contours, _ = cv.findContours(image_blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        top_left, bot_right, angle = cv.minAreaRect(contours[0])
        area = cv.contourArea(contours[0])

        poly = np.asarray([top_left, [top_left[0], bot_right[1]], bot_right, [top_left[1], bot_right[0]]], dtype=np.int32)
        mask = np.zeros_like(image_gray)
        cv.fillPoly(mask, [poly], 255)






def load_images(src_folder: str) -> List[str]:
    return glob.glob(src_folder + "\\*.png")


def detect():
    base_folder = "D:\\pharmapack\\"
    database = "PharmaPack_R_I_S1"
    src_folder = base_folder + database + "\\cropped\\"

    # images_path = load_images(src_folder)
    # for image_path in images_path:

    image_path = src_folder + "PharmaPack_R_I_S1_Ph1_P0016_D01_S001_C2_R1.png"

    image = cv.imread(image_path)
    image_gray = utils.prepare_image_gray(image)

    detect_text_regions(image_gray)


if __name__ == '__main__':
    detect()
