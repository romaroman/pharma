import cv2.cv2 as cv
import numpy as np

from copy import deepcopy
import pathlib
import os
import random

from scipy import ndimage
import skimage.measure
from skimage.feature import peak_local_max
from skimage.morphology import watershed

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

    def extract_edges(self, method: int = 1) -> NoReturn:
        nbins = 10

        sobel_x = cv.Sobel(self.image_std_filtered, cv.CV_64F, 1, 0)  # Find x and y gradients
        sobel_y = cv.Sobel(self.image_std_filtered, cv.CV_64F, 0, 1)

        magnitude = np.sqrt(sobel_x ** 2.0 + sobel_y ** 2.0)
        angle = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)

        _, image_bw = cv.threshold(self.image_std_filtered, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)

        image_bw = image_bw / 255

        magnitude = magnitude * image_bw
        angle = angle * image_bw

        ind = (np.ceil((angle + 180) / (360 / (nbins - 1))) + 1).astype(np.uint8)

        magnitude = magnitude / magnitude.max()

        threshold = 0.075
        binary_mask = (magnitude > threshold).astype(np.uint8) * 255

        if method == 1:
            self.image_preprocessed = binary_mask * ind / nbins
        elif method == 2:
            new_ind = np.zeros_like(ind)

            for j in range(1, nbins + 1):
                A = deepcopy(ind)
                A[A != j] = 0
                A[A != 0] = 1
                A = A * binary_mask

                A = cv.dilate(A, kernel=np.ones((14, 14)))

                contours, _ = cv.findContours(A, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                valid_contours = list(filter(lambda c: cv.contourArea(c) > 300, contours))

                bw = np.zeros_like(A)
                cv.drawContours(bw, valid_contours, -1, 1, -1)

                A = deepcopy(ind)
                A[A != j] = 0

                new_ind = new_ind + bw * A

            self.image_preprocessed = binary_mask * new_ind / nbins
            self.image_preprocessed[self.image_preprocessed != 0] = 255
            self.image_preprocessed = cv.dilate(self.image_preprocessed, kernel=np.ones((10, 10))).astype(np.uint8)

    def apply_basic_morphology(self) -> NoReturn:
        _, image_bw = cv.threshold(self.image_std_filtered, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        image_dilated = cv.dilate(image_bw, kernel=np.ones((self.KERNEL_SIZE_STD, self.KERNEL_SIZE_STD)))

        image_cleared = utils.clear_borders(image_dilated)

        self.image_preprocessed = utils.fill_holes(image_cleared)

    def filter_non_text_blobs(self) -> NoReturn:

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

    def apply_line_morphology(self, scale: float) -> NoReturn:
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

    def apply_watershed(self, image_morphed: np.ndarray) -> NoReturn:
        D = ndimage.distance_transform_edt(image_morphed)
        localMax = peak_local_max(D, indices=False, min_distance=50, labels=image_morphed)

        markers, _ = ndimage.label(localMax, structure=np.ones((3, 3)))
        labels = watershed(-D, markers, mask=image_morphed)

        display(labels * int(255 / labels.max()))
        return

    def find_text_regions(self) -> NoReturn:
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

    def draw_text_regions_and_mask(self, color: int = 255) -> NoReturn:
        for coord in self.coordinates:
            cv.drawContours(self.image_text_regions, [coord], -1, color, 2)
            cv.drawContours(self.image_text_masks, [coord], -1, color, -1)

    def create_visualization(self) -> NoReturn:
        first_row = np.hstack([self.image_gray, self.image_preprocessed, self.image_filled])
        second_row = np.hstack([self.image_cleared_borders, self.image_filtered, self.image_linearly_morphed])
        third_row = np.hstack([self.image_labeled * int(255 / self.image_labeled.max()), self.image_text_masks, self.image_text_regions])

        self.image_visualization = np.vstack([first_row, second_row, third_row])

    def detect_text_regions(self) -> NoReturn:
        self.preprocess()

        self.image_cleared_borders = utils.clear_borders(self.image_preprocessed)

        self.image_filled = utils.fill_holes(self.image_cleared_borders)

        self.filter_non_text_blobs()

        self.apply_line_morphology(scale=1)

        self.find_text_regions()

        self.draw_text_regions_and_mask()

        self.create_visualization()

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
            self.apply_basic_morphology()
        elif self.preprocess_method == self.PreprocessMethod.SimpleEdgeDetection:
            self.extract_edges(1)
        elif self.preprocess_method == self.PreprocessMethod.ComplexEdgeDetection:
            self.extract_edges(2)


def main():
    base_folder = "/fls/pharmapack/"
    database = "Enrollment"
    src_folder = base_folder + database + "/cropped/"

    images_path = utils.load_images(src_folder)
    # rng = 123
    # random.Random(rng).shuffle(images_path)

    for image_path in images_path[:20]:
        detect = DetectTextRegion(image_path, DetectTextRegion.PreprocessMethod.ComplexEdgeDetection)
        file_info = get_file_info(detect.filename, database)

        # if detect.filename not in ["PFP_Ph1_P0001_D01_S001_C4_az100_side1"]:
        #     continue

        detect.detect_text_regions()
        detect.write_results(base_folder, database)


if __name__ == '__main__':
    main()
