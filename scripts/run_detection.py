import sys
import logging


from pharma.common import config
from pharma.common.enums import Descriptor

from pharma.finegrained.detector import Detector
from pharma.finegrained.serializer import Serializer
from pharma.segmentation.loader import Loader
from pharma.segmentation.annotation import Annotation

import pyutils as pu


if __name__ == '__main__':
    pu.setup_logger(
        __name__,
        config.general.log_level,
        str(config.general.dir_output / f'{__name__}.txt')
    )
    logger = logging.getLogger(__name__)
    pu.suppress_warnings()

    loader = Loader(config.general.dir_source / 'Annotations')

    for i, file in enumerate(loader.get_files(), start=1):
        anno = Annotation.load_by_pattern(file.filename)
        image_mask = anno.create_mask(color=(255, 255, 255))[:, :, 0]
        detections = Detector.detect_descriptors(anno.image_ref, config.finegrained.descriptors_used, image_mask)
        Serializer.save_detections_to_redis(detections, ":".join(["REF", file.filename]), True)
        # Serializer.save_to_file(descriptors, file.filename, True)

        # from_redis = Serializer.load_from_redis_by_key(file.filename, True)
        # from_file = Serializer.load_from_file(file.filename, True)

        logger.info(f"{pu.zfill_n(i)} | Extracted {file.filename}")

    sys.exit(0)
