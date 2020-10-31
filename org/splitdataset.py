import os
from pathlib import Path
from argparse import ArgumentParser
from textdetector.fileinfo import FileInfo


parser = ArgumentParser()
parser.add_argument('dir_complete', type=str)


def split_insert_search(dir_complete: Path):
    for file in dir_complete.glob('**/**/*.png'):
        file_info = FileInfo.get_file_info_by_path(file)
        if file_info.angle == 360:
            dst_folder = 'Insert'
        else:
            dst_folder = 'Search'

        dst_file = Path(str(file.resolve()).replace('Complete', dst_folder))
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        os.link(file, dst_file)


if __name__ == '__main__':
    args = parser.parse_args()
    split_insert_search(Path(args.dir_complete))
