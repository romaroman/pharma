import re
import sys
import logging
import traceback

import pandas as pd
import numpy as np
import cv2 as cv
from numpy.distutils.command.config import config

from common import config
from common.enums import Descriptor

from finegrained import Detector, Serializer, Matcher
# from segmentation.segmenter import Segmenter
from segmentation.annotation import Annotation
from segmentation.loader import Loader
import utils


if __name__ == '__main__':
    utils.setup_logger(
        __name__,
        config.general.log_level,
        str(config.general.dir_output / f'{__name__}.txt')
    )
    logger = logging.getLogger(__name__)
    utils.suppress_warnings()

    annotations = dict()

    for file in (config.general.dir_source / 'Annotations').glob('*xml'):
        annotations[file.stem[9:17]] = Annotation(file)

    masks = (config.general.dir_source / 'MasksUnaligned').glob('*.png')
    df = pd.read_csv('/data/500gb/pharmapack/candidates_filtered_desc.csv', index_col=None)
    samples_valid = df.sample_actual.to_list()


    results_all = []
    for i, mask_file in enumerate(masks, start=1):
        algorithm, filename = mask_file.name.split(':')
        filename_formatted = filename[4:-10]

        try:
            row = df[df.sample_actual == filename_formatted]
            if len(row.index) == 0:
                logger.info(f"{utils.zfill_n(i)} | Skip {mask_file.name}")
                continue

            image_ver_mask = cv.imread(str(mask_file.resolve()), 0)
            image_ver = cv.imread(str(config.general.dir_source / 'Enrollment' / 'cropped' / filename))

            contours, _ = cv.findContours(image_ver_mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            image_ver_mask_contours = np.zeros_like(image_ver)
            cv.drawContours(image_ver_mask_contours, contours, -1, (255, 255, 0), 3)

            image_ver_draw = cv.add(image_ver, image_ver_mask_contours)
            keypoints_ver = Serializer.load_detections_from_file(mask_file.stem, False)[Descriptor.AKAZE]

            for candidate in list(set(row.to_numpy()[0].tolist()[2:])):
                candidate = str(candidate)
                keypoints_ref = Serializer.load_detections_from_redis(f'*{candidate[:-5]}*', True)[0][Descriptor.AKAZE]
                anno = Annotation.load_by_pattern(f"*{candidate[:-5]}*")
                image_ref = anno.image_ref
                image_ref_mask = anno.create_mask(color=(0, 255, 255))

                image_ref_draw = cv.add(image_ref, image_ref_mask)

                res, img_vis = Matcher.match(
                    keypoints_ver, keypoints_ref,
                    image_ver_draw, image_ref_draw
                )
                cv.imwrite(f'/data/500gb/pharmapack/Output/finematcing/{filename_formatted}_{candidate}.png', img_vis)
                results_all.append([filename_formatted, algorithm, candidate] + res)
        except:
            traceback.print_exc()
            results_all.append([filename_formatted, algorithm, traceback.format_exc(limit=10)])

        logger.info(f"{utils.zfill_n(i)} | Finematched {mask_file.name}")

    pd.DataFrame(results_all).to_csv('res.csv', index=False)
    sys.exit(0)
