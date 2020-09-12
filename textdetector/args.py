import argparse

from textdetector.enums import DetectionAlgorithm, ResultMethod, AlignmentMethod


parser = argparse.ArgumentParser(description='TextDetector')

# Required arguments
parser.add_argument('database', type=str, metavar='DATABASE', help='Enrollment | Recognition(1-3)')
parser.add_argument('src_folder', type=str, metavar='SRC_DIR', help='Path to source folder')
parser.add_argument('dst_folder', type=str, metavar='DST_DIR', help='Path to destination folder')

# Image loading arguments
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--seed', type=bool, default=True)
parser.add_argument('--split_on_chunks', type=bool, default=False)
parser.add_argument('--percentage', type=int, default=100)

# Console output arguments
parser.add_argument('--logging_level', type=str, default='DEBUG')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--profile', type=bool, default=False)

# Writing arguments
parser.add_argument('--write', type=bool, default=True)
parser.add_argument('--visualize', type=bool, default=True)
parser.add_argument('--clear_output', type=bool, default=True)

# General processing arguments
parser.add_argument('--multithreading', type=bool, default=True)
parser.add_argument('--scale_factor', type=float, default=1)
parser.add_argument('--algorithms', type=str, default=DetectionAlgorithm.to_string_list())
parser.add_argument('--approx_method', type=str, default=ResultMethod.Hull.name)
parser.add_argument('--alignment_method', type=str, default=AlignmentMethod.Reference.name)

# Additional processing stages args
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--extract_reference', type=bool, default=True)
