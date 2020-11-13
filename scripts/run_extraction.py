import sys
import logging

import cv2 as cv

from common import config
from finegrained.extractor import Extractor
from finegrained.serializer import Serializer
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

    loader = Loader(config.general.dir_source / 'Annotations')
    extractor = Extractor(config.finegrained.descriptors_used)

    for i, file in enumerate(loader.get_files(), start=1):
        image = cv.imread(str(file.path.resolve()))
        keypoints = extractor.extract(image)
        Serializer.save_to_redis(keypoints, file.filename)
        logger.info(f"{utils.zfill_n(i)} | Extracted {file.filename}")

    sys.exit(0)
