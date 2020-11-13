import sys
import logging

from common import config
import segmentation
import utils


if __name__ == '__main__':
    logger = logging.getLogger('segmentation')

    segmentation.writer.prepare_output_folder()

    utils.setup_logger(
        'text_detector',
        config.general.log_level,
        str(config.general.dir_output / 'log.txt')
    )
    utils.suppress_warnings()

    logger.info(f"Currently used configuration:\n{config.segmentation.confuse.dump()}")
    segmentation.Runner.process()

    sys.exit(0)
