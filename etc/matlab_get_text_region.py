import time
import h5py
import glob
import shutil
import pathlib
import timeit


start_time = time.time()


import matlab.engine

eng = matlab.engine.start_matlab()
eng.addpath('C:\\Users\\r.chaban\\Projects\\pharmapack-recognition\\matlab\\Text_extraction_no_quality_controle', '-end')
eng.addpath('C:\\Users\\r.chaban\\Projects\\pharmapack-recognition\\matlab\\Text_extraction_no_quality_controle\\libs', '-end')

print("INIT MATLAB --- %s seconds ---" % (time.time() - start_time))


def resolve_path(path_str: str) -> str:
    return str(pathlib.Path(path_str).resolve())


database = 'PharmaPack_R_I_S1'

source = 'D:\\pharmapack\\' + database + '\\cropped\\'
destination = 'D:\\pharmapack\\' + database + '\\text_regions\\no_quality_control\\'

for number, source_image in enumerate(glob.glob(source + "*.png"), start=1):
    print("INIT LOOP --- %s seconds ---" % (time.time() - start_time))
    visu_img_dst, masks_mat_dst, coords_mat_dst = eng.detect_text_regions_single_fun(source_image, number, nargout=3)
    print("FINISHED MATLAB SCRIPT --- %s seconds ---" % (time.time() - start_time))

    masks_path = resolve_path(masks_mat_dst)
    coords_path = resolve_path(coords_mat_dst)

    coords_file = h5py.File(coords_path)

    coords_ds = coords_file['TextRegionsCoordinates']

    number_of_detected_areas = coords_ds.shape[0]

    coords_pairs = []
    for i in range(number_of_detected_areas):
        coords_array = coords_file[coords_ds[i][0]][()]
        coords_pairs.append(list(zip(coords_array[0], coords_array[1])))


