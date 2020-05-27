import argparse


parser = argparse.ArgumentParser(description='Detect text regions using only image processing operations')
parser.add_argument('base_folder', type=str, help='an integer for the accumulator', default="D:/pharmapack/")
parser.add_argument('database', type=str, help='Enrollment / PharmaPack_R_I_S[1-3]', default="Enrollment")
parser.add_argument('--log_level', type=int, help='10 / 20 / 30 / 40 / 50')
parser.add_argument('--is_debug', type=bool, help='true or false')
