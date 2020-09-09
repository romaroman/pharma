import sys
import logging
from typing import NoReturn

from textdetector import config, Writer, Runner
import utils


logger = logging.getLogger('textdetector')


def setup() -> NoReturn:
    config.validate()
    Writer.prepare_output_folder()

    utils.setup_logger('text_detector', config.logging_level, str(config.dst_folder / 'log.txt'))
    utils.suppress_warnings()

    logger.info(f"Currently used configuration:\n{utils.pretty(config.to_dict())}")


def main() -> int:
    setup()

    runner = Runner()
    runner.process()

    return sys.exit(0)


if __name__ == '__main__':
    main()
