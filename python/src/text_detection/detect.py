import cv2 as cv
import numpy as np
import skimage
import skimage.measure

from copy import deepcopy
import time
import pathlib
import random

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage


from typing import List, Tuple, NoReturn
from enum import Enum, auto

import utils

_rrect = List[Tuple[Tuple[float, float], Tuple[float, float], float]]
logger = utils.get_logger('detect')

display = utils.show_image_as_plot


class PreprocessMethod(Enum):

    def __str__(self):
        return str(self.value)

    GeneralMorphology = auto(),
    SimpleEdgeDetection = auto(),
    ComplexEdgeDetection = auto()


class DetectTextRegion:

    KERNEL_SIZE_STD = 5

    def __init__(self, image_path: str):
        self.filename = pathlib.Path(image_path).stem

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

                A = cv.dilate(A, np.ones((14, 14)))

                contours, _ = cv.findContours(A, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                valid_contours = list(filter(lambda c: cv.contourArea(c) > 300, contours))

                bw = np.zeros_like(A)
                cv.drawContours(bw, valid_contours, -1, 1, -1)

                A = deepcopy(ind)
                A[A != j] = 0

                new_ind = new_ind + bw * A

            self.image_preprocessed = binary_mask * new_ind / nbins
            self.image_preprocessed[self.image_preprocessed != 0] = 255
            self.image_preprocessed = cv.dilate(self.image_preprocessed, np.ones((10, 10))).astype(np.uint8)

    def preprocess_image(self) -> NoReturn:
        _, image_bw = cv.threshold(self.image_std_filtered, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        image_dilated = cv.dilate(image_bw, np.ones((self.KERNEL_SIZE_STD, self.KERNEL_SIZE_STD)))

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

    def find_text_areas(self) -> NoReturn:
        self.image_cleared_borders = utils.clear_borders(self.image_preprocessed)
        self.image_filled = utils.fill_holes(self.image_cleared_borders)

        self.image_filled = cv.erode(self.image_filled, np.ones((3, 3)))

        self.filter_non_text_blobs()

        # scale = 0.25
        # new_size = (int(image_filtered.shape[1] * scale), int(image_filtered.shape[0] * scale))
        # old_size = image_filtered.shape[1], image_filtered.shape[0]
        # image_resized = cv.resize(image_filtered, new_size, interpolation=cv.INTER_NEAREST)

        self.apply_line_morphology(1)

        self.image_linearly_morphed = cv.morphologyEx(self.image_linearly_morphed, cv.MORPH_OPEN, np.ones((5, 5)))

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

    # TODO: implement
    def combine_intersections(self) -> NoReturn:
        dist_threshold = 20
        for idx1 in range(0, len(self.rrects)):
            for idx2 in range(idx1 + 1, len(self.rrects)):
                if not np.equal(self.rrects[idx1], self.rrects[idx2]).all():
                    distance = utils.calc_rrects_distance(self.coordinates[idx1], self.coordinates[idx2])
                    if distance < dist_threshold:
                        print(distance)
                        # image_visualization_rrect = deepcopy(image_visualization)
                        # cv.drawContours(image_visualization_rrect, [cv.boxPoints(rrects[idx1]).astype(np.int0)], -1, (255, 255, 0), 2)
                        # cv.drawContours(image_visualization_rrect, [cv.boxPoints(rrects[idx2]).astype(np.int0)], -1, (255, 255, 0), 2)
                        # display(image_visualization_rrect)

    def create_visualization(self) -> NoReturn:
        first_row = np.hstack([self.image_gray, self.image_preprocessed, self.image_filled])
        second_row = np.hstack([self.image_cleared_borders, self.image_filtered, self.image_linearly_morphed])
        third_row = np.hstack([self.image_labeled * int(255 / self.image_labeled.max()), self.image_text_masks, self.image_text_regions])
        self.image_visualization = np.vstack([first_row, second_row, third_row])

    def detect_text_regions(self, preprocess_method: PreprocessMethod) -> NoReturn:
        try:
            # start_time = time.time()
            # print("LOAD AND FILTER --- %s seconds ---" % (time.time() - start_time))
            if preprocess_method == PreprocessMethod.GeneralMorphology:
                self.preprocess_image()
                # print("GENERAL MORPHOLOGY --- %s seconds ---" % (time.time() - start_time))
            elif preprocess_method == PreprocessMethod.SimpleEdgeDetection:
                self.extract_edges(1)
                # print("SIMPLE EDGES EXTRACTION --- %s seconds ---" % (time.time() - start_time))
            elif preprocess_method == PreprocessMethod.ComplexEdgeDetection:
                self.extract_edges(2)
                # print("COMPLEX EDGES EXTRACTION --- %s seconds ---" % (time.time() - start_time))

            self.find_text_areas()
            # print("FIND COORDINATES AND MASK --- %s seconds ---" % (time.time() - start_time))

            self.draw_text_regions_and_mask()

            self.create_visualization()

            logger.info(f'SUCCESS {self.filename}, found {len(self.rrects)} regions')
        except Exception as e:
            logger.warn(f'WARNING {self.filename}, occured error: {e}')

    def write_results(self, base_folder: str, database: str) -> NoReturn:
        text_coord_dst_folder = base_folder + database + f"\\python\\text_coords\\{self.filename}.csv"
        text_regions_dst_folder = base_folder + database + f"\\python\\text_regions\\{self.filename}.png"
        text_masks_dst_folder = base_folder + database + f"\\python\\text_masks\\{self.filename}.png"
        visualization_dst_folder = base_folder + database + f"\\python\\visualizations\\{self.filename}.png"

        np.savetxt(text_coord_dst_folder, self.coordinates_ravel, delimiter=",", fmt='%i')
        cv.imwrite(text_regions_dst_folder, self.image_text_regions)
        cv.imwrite(text_masks_dst_folder, self.image_text_masks)
        cv.imwrite(visualization_dst_folder, self.image_visualization)


def main():
    base_folder = "D:\\pharmapack\\"
    database = "Enrollment"
    src_folder = base_folder + database + "\\cropped\\"

    images_path = utils.load_images(src_folder)
    rng = 123
    random.Random(rng).shuffle(images_path)

    for image_path in images_path:
        detect = DetectTextRegion(image_path)

        # if detect.filename not in ["PFP_Ph1_P0001_D01_S001_C4_az100_side1"]:
        #     continue

        detect.detect_text_regions(PreprocessMethod.ComplexEdgeDetection)
        detect.write_results(base_folder, database)


if __name__ == '__main__':
    main()