import os
from typing import NoReturn
import numpy as np
import cv2 as cv
import glob
import random

import utils
from text_detection.detect import DetectTextRegion
from text_detection.args import parser, Options
from text_detection.file_info import get_file_info


logger = utils.get_logger(__name__)


class Run:

    def __init__(self, options: Options):
        self.options = options
        self.logger = utils.get_logger(__name__, options.log_level)
        self.image_paths = self._load_images()

    def process(self):
        random.shuffle(self.image_paths)
        for image_path in self.image_paths:
            file_info = get_file_info(image_path, self.options.database)

            image_orig = cv.imread(image_path)
            flags = DetectTextRegion.Flags(self.options.visualize, self.options.time_profiling)

            detect = DetectTextRegion(image_orig, flags)
            detect.detect_text_regions()

            if self.options.write:
                self.write_result(detect, file_info.filename)

    def _load_images(self):
        return glob.glob(self.options.base_folder + self.options.database + "/cropped/*.png")

    def write_result(self, detect: DetectTextRegion, filename: str) -> NoReturn:
        common_folder = self.options.base_folder + self.options.database

        text_coord_dst_folder = f"{common_folder}/python/text_coords"
        text_regions_dst_folder = f"{common_folder}/python/text_regions"
        text_masks_dst_folder = f"{common_folder}/python/text_masks"
        visualization_dst_folder = f"{common_folder}/python/visualizations"

        os.makedirs(text_coord_dst_folder, exist_ok=True)
        os.makedirs(text_regions_dst_folder, exist_ok=True)
        os.makedirs(text_masks_dst_folder, exist_ok=True)
        os.makedirs(visualization_dst_folder, exist_ok=True)

        text_coord_dst_path = f"{text_coord_dst_folder}/{filename}.csv"
        text_regions_dst_path = f"{text_regions_dst_folder}/{filename}.png"
        text_masks_dst_path = f"{text_masks_dst_folder}/{filename}.png"
        visualization_dst_path = f"{visualization_dst_folder}/{filename}.png"

        # np.savetxt(text_coord_dst_path, detect.coordinates_ravel, delimiter=",", fmt='%i')
        cv.imwrite(text_regions_dst_path, detect.image_word_regions)
        # cv.imwrite(text_masks_dst_path, detect.image_text_masks)
        cv.imwrite(visualization_dst_path, detect.image_visualization)


def main():
    options = Options(parser)
    runner = Run(options)
    runner.process()


if __name__ == '__main__':
    main()
