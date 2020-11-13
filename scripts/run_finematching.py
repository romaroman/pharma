import re
import sys
import logging

import pandas as pd
import cv2 as cv
from numpy.distutils.command.config import config

from common import config
from common.enums import Descriptor

from finegrained import Extractor, Serializer, Matcher
from finegrained.matcher import Serializer
from segmentation.segmenter import Segmenter
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

    loader = Loader(config.general.dir_source / 'Enrollment' / 'cropped')
    df = pd.read_csv('/fls/pharmapack/candidates_filtered.csv', index_col=None)
    samples_valid = df.sample_actual.to_list()

    for i, file in enumerate(loader.get_files(), start=1):
        row = df[df.sample_actual == file.filename[4:-6]]
        if len(row.index) == 0:
            i -= 1
            continue

        image = cv.imread(str(file.path.resolve()))

        segmenter = Segmenter(image)
        segmenter.segment(config.segmentation.algorithms)

        keypoints_verification = Extractor.extract_descriptor(image, Descriptor.AKAZE)
        for candidate in row.to_numpy()[0].tolist()[2:]:
            keypoints_reference = Serializer.load_from_redis_by_pattern(
                f'*{candidate[:-5]}*', as_reference=True
            )[Descriptor.AKAZE]
            image_ref = cv.imread(str(list((config.general.dir_source / 'Annotations').glob(f"*{candidate[:-5]}*.png"))[0].resolve()))
            Matcher.match_flann_candidate(
                keypoints_verification, keypoints_reference,
                image, image_ref
            )


        logger.info(f"{utils.zfill_n(i)} | Finematched {file.filename}")

    sys.exit(0)
