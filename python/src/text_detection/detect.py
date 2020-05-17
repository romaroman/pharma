import cv2 as cv
import numpy as np
import skimage
import skimage.measure

from copy import deepcopy
import glob
import time
import os
import shutil
import pathlib

from typing import List, Tuple
from enum import Enum, auto

import utils


display = utils.show_image_as_plot
start_time = time.time()


class PreprocessMethod(Enum):

    def __str__(self):
        return str(self.value)

    GeneralMorphology = auto(),
    SimpleEdgeDetection = auto(),
    ComplexEdgeDetection = auto()


def extract_edges(image_gray: np.ndarray, method: int = 1) -> np.ndarray:
    nbins = 7

    sobel_x = cv.Sobel(image_gray, cv.CV_64F, 1, 0)  # Find x and y gradients
    sobel_y = cv.Sobel(image_gray, cv.CV_64F, 0, 1)

    # Find magnitude and angle
    magnitude = np.sqrt(sobel_x ** 2.0 + sobel_y ** 2.0)
    angle = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)

    _, image_bw = cv.threshold(image_gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)

    image_bw = image_bw / 255

    magnitude = magnitude * image_bw
    angle = angle * image_bw

    ind = (np.ceil((angle + 180) / (360 / (nbins - 1))) + 1).astype(np.uint8)

    magnitude = magnitude / magnitude.max()

    threshold = 0.075
    binary_mask = (magnitude > threshold).astype(np.uint8) * 255

    edges = np.zeros_like(binary_mask)
    if method == 1:
        edges = binary_mask * ind / nbins
    elif method == 2:
        new_ind = np.zeros_like(ind)

        for j in range(1, nbins + 1):
            A = deepcopy(ind)
            A[A != j] = 0
            A[A != 0] = 1
            A = A * binary_mask

            A = cv.dilate(A, np.ones((12, 12)))

            contours, _ = cv.findContours(A, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            valid_contours = list(filter(lambda c: cv.contourArea(c) > 300, contours))

            bw = np.zeros_like(A)
            cv.drawContours(bw, valid_contours, -1, 1, -1)

            A = deepcopy(ind)
            A[A != j] = 0

            new_ind = new_ind + bw * A

        edges = binary_mask * new_ind / nbins
        edges[edges != 0] = 255
        edges = cv.dilate(edges, np.ones((5, 5)))

    return edges.astype(np.uint8)


def preprocess_image(image_gray: np.ndarray) -> np.ndarray:
    kernel_size = 5

    _, image_bw = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    image_dilated = cv.dilate(image_bw, np.ones((kernel_size, kernel_size)))

    image_cleared = utils.clear_borders(image_dilated)

    image_filled = utils.fill_holes(image_cleared)

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

    image_filtered = (np.isin(image_labeled, valid_labels) > 0).astype(np.uint8) * 255

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


def detect_text_regions(image_path: str, preprocess_method: PreprocessMethod) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    image = cv.imread(image_path)
    image_gray = utils.prepare_image_gray(image)

    kernel_size = 5

    image_std = utils.std_filter(image_gray, kernel_size)

    image_preprocessed = np.zeros_like(image_std)
    if preprocess_method == PreprocessMethod.GeneralMorphology:
        image_preprocessed = preprocess_image(image_std)
        print("GENERAL MORPHOLOGY --- %s seconds ---" % (time.time() - start_time))
    elif preprocess_method == PreprocessMethod.SimpleEdgeDetection:
        image_preprocessed = extract_edges(image_std, 1)
        print("SIMPLE EDGES EXTRACTION --- %s seconds ---" % (time.time() - start_time))
    elif preprocess_method == PreprocessMethod.ComplexEdgeDetection:
        image_preprocessed = extract_edges(image_std, 2)
        print("COMPLEX EDGES EXTRACTION --- %s seconds ---" % (time.time() - start_time))

    image_clear_borders = utils.clear_borders(image_preprocessed)
    image_filled = utils.fill_holes(image_clear_borders)

    image_filtered = filter_non_text_blobs(image_filled)
    print("FILTER BLOBS --- %s seconds ---" % (time.time() - start_time))


    scale = 0.25
    new_size = (int(image_filtered.shape[1] * scale), int(image_filtered.shape[0] * scale))
    image_resized = cv.resize(image_filtered, new_size, interpolation=cv.INTER_NEAREST)

    image_linearly_morphed = apply_line_morphology(image_resized, scale)
    print("APPLY LINEAR MORPHOLOGY --- %s seconds ---" % (time.time() - start_time))

    _, image_labeled = cv.connectedComponents(image_linearly_morphed)

    image_visualization = cv.resize(image, new_size, interpolation=cv.INTER_NEAREST)
    image_final_mask = np.zeros(image_resized.shape, dtype=np.uint8)

    coordinates = list()
    coordinates_ravel = list()

    for k in range(1, image_labeled.max() + 1):

        image_blob = (image_labeled == k).astype(np.uint8) * 255
        contours, _ = cv.findContours(image_blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        points = np.int0(cv.boxPoints(cv.minAreaRect(contours[0])))

        coordinates.append(points)
        coordinates_ravel.append(np.transpose(points).ravel())

        cv.drawContours(image_visualization, [points], -1, (255, 0, 0), 2)
        cv.drawContours(image_final_mask, [points], -1, 255, -1)

    print("FIND COORDINATES AND MASK --- %s seconds ---" % (time.time() - start_time))
    return image_visualization, image_final_mask, np.asarray(coordinates, np.int0), np.asarray(coordinates_ravel, np.int0)


def detect():
    base_folder = "D:\\pharmapack\\"
    database = "Enrollment"
    src_folder = base_folder + database + "\\cropped\\"
    reg_dst_folder = base_folder + database + "\\text_regions_py\\{}.csv"
    vis_dst_folder = base_folder + database + "\\visualization_py\\{}.png"
    mask_dst_folder = base_folder + database + "\\mask_regions_py\\{}.png"

    images_path = utils.load_images(src_folder)
    for image_path in images_path:

        filename = pathlib.Path(image_path).stem

        # if filename not in ["PFP_Ph1_P0001_D01_S001_C4_az100_side1"]:
        #     continue

        image_visualization, image_mask, _, coordinates_ravel = detect_text_regions(image_path, PreprocessMethod.ComplexEdgeDetection)

        image_mask = np.stack((image_mask,) * 3, axis=-1)
        image_to_display = np.vstack([image_mask, image_visualization])
        display(image_to_display)

        s = 1
        # np.savetxt(reg_dst_folder.format(filename), coordinates_ravel, delimiter=",", fmt='%i')
        # cv.imwrite(vis_dst_folder.format(filename), image_visualization)
        # cv.imwrite(mask_dst_folder.format(filename), image_mask)


if __name__ == '__main__':
    detect()
