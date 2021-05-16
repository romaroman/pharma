import sys
import logging

from pharma.common import config
from pharma.segmentation import writer, Runner
import pyutils as pu


if __name__ == '__main__':
    logger = logging.getLogger('segmentation')

    writer.prepare_output_folder()

    pu.setup_logger(
        'segmentation',
        config.general.log_level,
        str(config.general.dir_output / 'log.txt')
    )
    pu.suppress_warnings()

    logger.info(f"Currently used configuration:\n{config.segmentation.confuse.parent.dump()}")
    Runner.process()

    sys.exit(0)
