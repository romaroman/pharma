import sys
import logging
from typing import NoReturn

from textdetector import config, Writer, Runner
import utils


logger = logging.getLogger('textdetector')


def main() -> int:
    Writer.prepare_output_folder()

    utils.setup_logger('text_detector', config.out_log_level, str(config.dir_output / 'log.txt'))
    utils.suppress_warnings()

    logger.info(f"Currently used configuration:\n{config.confuse.dump()}")

    runner = Runner()
    runner.process()

    return sys.exit(0)


if __name__ == '__main__':
    main()
