import cv2 as cv
import numpy as np
import skimage
import skimage.measure

import glob
import time
import os
import shutil
import pathlib

from typing import List, Tuple


import utils


display = utils.show_image_as_plot
start_time = time.time()


def extract_edges(image_gray: np.ndarray) -> np.ndarray:
    sobel_x = cv.Sobel(image_gray, cv.CV_64F, 1, 0)  # Find x and y gradients
    sobel_y = cv.Sobel(image_gray, cv.CV_64F, 0, 1)

    # Find magnitude and angle
    magnitude = np.sqrt(sobel_x ** 2.0 + sobel_y ** 2.0)
    angle = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)

    nbins = 7
    ind = np.ceil((angle + 180) / (360 / (nbins - 1))) + 1

    magnitude = magnitude / magnitude.max()

    threshold = 0.075
    binary_mask = magnitude > threshold

    edges = binary_mask * ind / nbins

    return edges


def preprocess_image(image_gray: np.ndarray) -> np.ndarray:
    kernel_size = 5

    image_filtered = utils.std_filter(image_gray, kernel_size)

    image_edges = extract_edges(image_filtered)

    _, image_bw = cv.threshold(image_filtered, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    image_dilated = cv.dilate(image_bw, np.ones((kernel_size, kernel_size)))

    image_cleared = utils.clear_borders(image_dilated)

    contours, _ = cv.findContours(image_cleared, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    image_filled = np.zeros_like(image_dilated)
    cv.drawContours(image_filled, contours, -1, 255, -1)

    return image_filled





def filter_non_text_blobs(image_preprocessed: np.ndarray) -> np.ndarray:

    def is_prop_valid(prop: skimage.measure._regionprops._RegionProperties) -> bool:
        if prop.minor_axis_length > 20:
            if prop.solidity > 0.33:
                if prop.area > 150:
                    return True

        return False

    _, image_labeled = cv.connectedComponents(image_preprocessed)
    props = skimage.measure.regionprops(image_labeled, coordinates='rc')

    valid_props = list(filter(is_prop_valid, props))
    valid_labels = np.asarray([prop.label for prop in valid_props])

    image_filtered = ((np.isin(image_labeled, valid_labels) > 0) * 255).astype(np.uint8)

    return image_filtered


def apply_line_morphology(image_resized: np.ndarray, scale: float) -> np.ndarray:
    line_length = int(200 * scale)

    line_rotation_angles = [i for i in range(-90, 90, 15)]

    pixel_amounts = []
    for line_rotation_angle in line_rotation_angles:
        strel_line_rotated = utils.morph_line(line_length, line_rotation_angle)
        image_linearly_morphed = cv.dilate(image_resized, strel_line_rotated)

        pixel_amounts.append(cv.countNonZero(image_linearly_morphed))

    k = np.asarray(pixel_amounts).argmin()

    strel_line_rotated = utils.morph_line(int(line_length / 2), line_rotation_angles[k])
    image_linearly_morphed = cv.dilate(image_resized, strel_line_rotated)

    return image_linearly_morphed


def detect_text_regions(image_gray: np.array) -> np.ndarray:

    image_preprocessed = preprocess_image(image_gray)
    print("PREPROCESS --- %s seconds ---" % (time.time() - start_time))

    image_filtered = filter_non_text_blobs(image_preprocessed)
    print("FILTER BLOBS --- %s seconds ---" % (time.time() - start_time))

    scale = 0.25
    new_size = (int(image_filtered.shape[1] * scale), int(image_filtered.shape[0] * scale))
    image_resized = cv.resize(image_filtered, new_size, interpolation=cv.INTER_NEAREST)

    image_linearly_morphed = apply_line_morphology(image_resized, scale)
    print("APPLY MORPHOLOGY --- %s seconds ---" % (time.time() - start_time))

    _, image_labeled = cv.connectedComponents(image_linearly_morphed)

    image_final_mask = np.zeros(image_resized.shape, dtype=np.uint8)
    coordinates = list()
    for k in range(1, image_labeled.max()):

        image_blob = (image_labeled == k).astype(np.uint8) * 255
        contours, _ = cv.findContours(image_blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        points = np.int0(cv.boxPoints(cv.minAreaRect(contours[0])))

        coordinates.append(np.transpose(points).ravel())
        cv.drawContours(image_final_mask, [points], -1, 255, -1)

    print("FIND MASK --- %s seconds ---" % (time.time() - start_time))
    return np.asarray(coordinates, np.int0)


def detect():
    base_folder = "D:\\pharmapack\\"
    database = "PharmaPack_R_I_S1"
    src_folder = base_folder + database + "\\cropped\\"
    dst_folder = base_folder + database + "\\text_regions_py\\"

    # images_path = load_images(src_folder)
    # for image_path in images_path:

    image_path = src_folder + "PharmaPack_R_I_S1_Ph1_P0016_D01_S001_C2_R1.png"

    image = cv.imread(image_path)
    image_gray = utils.prepare_image_gray(image)

    coordinates = detect_text_regions(image_gray)
    np.savetxt("here.csv", coordinates, delimiter=",", fmt='%i')


if __name__ == '__main__':
    detect()
