from argparse import ArgumentParser


parser = ArgumentParser(description='Detect text regions using only image processing operations')
parser.add_argument('base_folder', type=str, help='an integer for the accumulator', default="D:/pharmapack/")
parser.add_argument('database', type=str, help='Enrollment / PharmaPack_R_I_S[1-3]', default="Enrollment")
parser.add_argument('iterations', type=int, help='1 / 2 / 3', default="2")
parser.add_argument('--write', dest="write", type=bool, help='true or false', default="True")
parser.add_argument('--log_level', dest="log_level", type=int, help='10 / 20 / 30 / 40 / 50', default=0)
parser.add_argument('--is_debug', dest="is_debug", type=bool, help='true or false', default=True)
parser.add_argument('--visualize', dest="visualize", type=bool, help='true or false', default=True)


class Options:
    def __init__(self, parser: ArgumentParser):
        args = parser.parse_args()

        self.base_folder = args.base_folder
        self.database = args.database
        self.iterations = args.iterations
        self.write = args.write
        self.log_level = args.log_level
        self.is_debug = args.is_debug
        self.visualize = args.visualize
