import sys
import logging

import textdetector
import visualization
import utils


logger = logging.getLogger('textdetector')


def main() -> int:
    textdetector.writer.prepare_output_folder()

    utils.setup_logger(
        'text_detector',
        textdetector.config.out_log_level,
        str(textdetector.config.dir_output / 'log.txt')
    )
    utils.suppress_warnings()

    logger.info(f"Currently used configuration:\n{textdetector.config.confuse.dump()}")
    textdetector.Runner.process()

    visualization.plot_all_classes(textdetector.config.dir_output / f"session_pd_{textdetector.config.timestamp}.csv")

    return sys.exit(0)


if __name__ == '__main__':
    main()
