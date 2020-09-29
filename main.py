import sys
import logging

import nnmodels
import textdetector
import visualization
import utils


logger = logging.getLogger('textdetector')


def run_textdetection() -> int:
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


def run_simclr() -> int:
    dataset = nnmodels.simclr.Wrapper()
    simclr = nnmodels.simclr.SimCLR(dataset)
    simclr.train()
    return sys.exit(0)


if __name__ == '__main__':
    run_simclr()
    # run_textdetection()
