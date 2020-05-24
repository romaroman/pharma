import cv2.cv2 as cv
import numpy as np

from copy import deepcopy
import pathlib
import os
import random
import skimage.measure

from typing import List, Tuple, NoReturn
from enum import Enum, auto

from file_info import get_file_info, FileInfoEnrollment, FileInfoRecognition
import utils


_rrect = List[Tuple[Tuple[float, float], Tuple[float, float], float]]
logger = utils.get_logger('detect')

display = utils.show_image_as_plot


class DetectTextRegion:

    class PreprocessMethod(Enum):

        def __str__(self):
            return str(self.value)

        BasicMorphology = auto(),
        SimpleEdgeDetection = auto(),
        ComplexEdgeDetection = auto()

    KERNEL_SIZE_STD = 5

    def __init__(self, image_path: str, preprocess_method: PreprocessMethod):
        self.filename = pathlib.Path(image_path).stem
        self.preprocess_method = preprocess_method

        self.image_orig = cv.imread(image_path)

        self.image_gray = cv.cvtColor(self.image_orig, cv.COLOR_BGR2GRAY)
        self.image_std_filtered = utils.std_filter(self.image_gray, self.KERNEL_SIZE_STD)

        self.image_visualization = np.zeros((self.image_gray.shape[0] * 3, self.image_gray.shape[1] * 3))

        self.image_preprocessed = np.zeros_like(self.image_gray)

        self.image_filled = np.zeros_like(self.image_gray)
        self.image_cleared_borders = np.zeros_like(self.image_gray)
        self.image_filtered = np.zeros_like(self.image_gray)
        self.image_linearly_morphed = np.zeros_like(self.image_gray)
        self.image_labeled = np.zeros_like(self.image_gray)

        self.image_text_masks = np.zeros_like(self.image_gray)
        self.image_text_regions = deepcopy(self.image_gray)

        self.rrects = list()
        self.coordinates = list()
        self.coordinates_ravel = list()

    def detect_text_regions(self) -> NoReturn:
        self.preprocess()

        self.image_cleared_borders = utils.clear_borders(self.image_preprocessed)

        self.image_filled = utils.fill_holes(self.image_cleared_borders)

        self._filter_non_text_blobs()

        self._apply_line_morphology(scale=1)

        self._find_text_regions()

        self._draw_text_regions_and_mask()

        self._create_visualization()

        logger.info(f'SUCCESS {self.filename}, found {len(self.rrects)} regions')

    def write_results(self, base_folder: str, database: str) -> NoReturn:
        text_coord_dst_folder = base_folder + database + f"/python/text_coords"
        text_regions_dst_folder = base_folder + database + f"/python/text_regions"
        text_masks_dst_folder = base_folder + database + f"/python/text_masks"
        visualization_dst_folder = base_folder + database + f"/python/visualizations"

        os.makedirs(text_coord_dst_folder, exist_ok=True)
        os.makedirs(text_regions_dst_folder, exist_ok=True)
        os.makedirs(text_masks_dst_folder, exist_ok=True)
        os.makedirs(visualization_dst_folder, exist_ok=True)

        text_coord_dst_path = text_coord_dst_folder + f"/{self.filename}.csv"
        text_regions_dst_path = text_regions_dst_folder + f"/{self.filename}.png"
        text_masks_dst_path = text_masks_dst_folder + f"/{self.filename}.png"
        visualization_dst_path = visualization_dst_folder + f"/{self.filename}.png"

        np.savetxt(text_coord_dst_path, self.coordinates_ravel, delimiter=",", fmt='%i')
        cv.imwrite(text_regions_dst_path, self.image_text_regions)
        cv.imwrite(text_masks_dst_path, self.image_text_masks)
        cv.imwrite(visualization_dst_path, self.image_visualization)

    def preprocess(self):
        if self.preprocess_method == self.PreprocessMethod.BasicMorphology:
            self._apply_basic_morphology()
        elif self.preprocess_method == self.PreprocessMethod.SimpleEdgeDetection:
            self._extract_edges(1)
        elif self.preprocess_method == self.PreprocessMethod.ComplexEdgeDetection:
            self._extract_edges(2)

    def _apply_basic_morphology(self) -> NoReturn:
        _, image_bw = cv.threshold(self.image_std_filtered, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        image_dilated = cv.dilate(image_bw, kernel=np.ones((self.KERNEL_SIZE_STD, self.KERNEL_SIZE_STD)))

        image_cleared = utils.clear_borders(image_dilated)

        self.image_preprocessed = utils.fill_holes(image_cleared)

    def _extract_edges(self, method: int = 1) -> NoReturn:

        magnitude, angle = utils.find_magnitude_and_angle(self.image_std_filtered)

        nbins = 7
        image_ind = (np.ceil((angle + 180) / (360 / (nbins - 1))) + 1).astype(np.uint8)

        threshold = 0.075
        image_binary_mask = (magnitude > threshold).astype(np.uint8) * 255

        if method == 1:
            self.image_preprocessed = image_binary_mask * image_ind / nbins
        elif method == 2:
            image_new_ind = utils.filter_long_edges(image_ind, image_binary_mask, nbins)

            self.image_preprocessed = image_binary_mask * image_new_ind / nbins
            self.image_preprocessed[self.image_preprocessed != 0] = 255
            self.image_preprocessed = cv.dilate(self.image_preprocessed, kernel=np.ones((10, 10))).astype(np.uint8)

    def _filter_non_text_blobs(self) -> NoReturn:

        def is_prop_valid(prop: skimage.measure._regionprops._RegionProperties) -> bool:
            if prop.minor_axis_length > 20:
                if prop.solidity > 0.33:
                    if prop.area > 500:
                        return True

            return False

        _, image_labeled = cv.connectedComponents(self.image_filled)
        props = skimage.measure.regionprops(image_labeled, coordinates='rc')

        valid_props = list(filter(is_prop_valid, props))
        valid_labels = np.asarray([prop.label for prop in valid_props])

        self.image_filtered = (np.isin(image_labeled, valid_labels) > 0).astype(np.uint8) * 255

    def _apply_line_morphology(self, scale: float) -> NoReturn:
        line_length = int(150 * scale)

        line_rotation_angles = [i for i in range(-90, 90, 5)]

        pixel_amounts = []
        for line_rotation_angle in line_rotation_angles:
            strel_line_rotated = utils.morph_line(line_length, line_rotation_angle)
            self.image_linearly_morphed = cv.dilate(self.image_filtered, strel_line_rotated)

            pixel_amounts.append(cv.countNonZero(self.image_linearly_morphed))

        k = np.asarray(pixel_amounts).argmin()

        strel_line_rotated = utils.morph_line(int(line_length / 2), line_rotation_angles[k])
        self.image_linearly_morphed = cv.dilate(self.image_filtered, strel_line_rotated)

    def _find_text_regions(self) -> NoReturn:
        self.image_linearly_morphed = cv.morphologyEx(self.image_linearly_morphed, cv.MORPH_OPEN, kernel=np.ones((5, 5)))

        _, self.image_labeled = cv.connectedComponents(self.image_linearly_morphed)

        for k in range(1, self.image_labeled.max() + 1):
            image_blob = (self.image_labeled == k).astype(np.uint8) * 255
            contours, _ = cv.findContours(image_blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            rrect = cv.minAreaRect(contours[0])
            points = np.int0(cv.boxPoints(rrect))

            self.rrects.append(rrect)
            self.coordinates.append(points)
            self.coordinates_ravel.append(np.transpose(points).ravel())

    def _draw_text_regions_and_mask(self, color: int = 255) -> NoReturn:
        for coord in self.coordinates:
            cv.drawContours(self.image_text_regions, [coord], -1, color, 2)
            cv.drawContours(self.image_text_masks, [coord], -1, color, -1)

    def _create_visualization(self) -> NoReturn:
        first_row = np.hstack([self.image_gray, self.image_preprocessed, self.image_cleared_borders])
        second_row = np.hstack([self.image_filled, self.image_filtered, self.image_linearly_morphed])
        third_row = np.hstack([self.image_labeled * int(255 / self.image_labeled.max()), self.image_text_masks, self.image_text_regions])

        self.image_visualization = np.vstack([first_row, second_row, third_row])


def main():
    base_folder = "/fls/pharmapack/"
    database = "Enrollment"
    src_folder = base_folder + database + "/cropped/"

    images_path = utils.load_images(src_folder)
    # rng = 123
    # random.Random(rng).shuffle(images_path)

    for image_path in images_path:
        file_info = get_file_info(image_path, database)

        if file_info.angle != 360:
            continue

        detect = DetectTextRegion(image_path, DetectTextRegion.PreprocessMethod.ComplexEdgeDetection)

        detect.detect_text_regions()
        detect.write_results(base_folder, database)


if __name__ == '__main__':
    main()
