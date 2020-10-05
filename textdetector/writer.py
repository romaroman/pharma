import os
import json
import shutil
from pathlib import Path
from typing import Union, Dict, List, NoReturn, Any

import cv2 as cv
import numpy as np
import pandas as pd

import config
import utils

from detector import Detector
from referencer import Referencer


def write_entity(
        entity: Union[List[np.ndarray], np.ndarray, int, Dict[str, Dict[str, Union[int, float]]]],
        folder_suffix: str,
        filename: str,
        extension: str
) -> NoReturn:
    dst_folder = config.dir_output / folder_suffix
    os.makedirs(str(dst_folder.resolve()), exist_ok=True)
    dst_path = str(dst_folder / Path(filename + f".{extension}"))

    if extension == "png":
        cv.imwrite(dst_path, entity)

    elif extension == "csv":
        np.savetxt(dst_path, entity, delimiter=",", fmt='%i')

    elif extension == "json":
        write_json(entity, dst_path)


def write_image_region(image: np.ndarray, folder_suffix: str, filename: str, order: str) -> NoReturn:
    dst_folder = config.dir_output / folder_suffix / filename
    os.makedirs(str(dst_folder.resolve()), exist_ok=True)
    dst_path = str(dst_folder / f"{order}.png")

    cv.imwrite(dst_path, image)


def write_json(data: Any, path: str) -> NoReturn:
    with open(path, 'w+') as file:
        json.dump(data, file, indent=2, sort_keys=True)


def save_detection_results(detection: Detector, filename: str) -> NoReturn:
    for algorithm, result in detection.results.items():
        for method in result.masks.keys():
            common_part = f"{algorithm}/{method}"
            write_entity(result.masks[method], f"{common_part}/masks", filename, "png")

            for index, region in enumerate(result.regions[method], start=1):
                write_image_region(
                    region.crop_image(detection.image_not_scaled),
                    f"{common_part}/parts", filename, utils.zfill_n(index)
                )


def save_reference_results(referencer: Referencer, filename: str) -> NoReturn:
    for label, image in referencer.results.items():
        write_image_region(image, f"REF/parts", filename, label)


def prepare_output_folder() -> NoReturn:
    if config.out_clear_output_dir:
        shutil.rmtree(config.dir_output, ignore_errors=True)
        os.makedirs(str(config.dir_output.resolve()), exist_ok=True)


def update_session_with_pd(results: List[Dict[str, Any]]) -> NoReturn:
    df_file = f"session_pd_{config.timestamp}.csv"

    if os.path.exists(str(config.dir_output / df_file)):
        df = pd.read_csv(config.dir_output / df_file, index_col=False)
    else:
        df = pd.DataFrame()

    for result in results:
        dict_combined = dict()

        for key_general, dict_result in result.items():
            dict_result_general = {}
            for key_short, value in dict_result.items():
                dict_result_general[f"{key_general}_{key_short}"] = value

            dict_combined.update(dict_result_general)

        df = df.append(pd.Series(dict_combined), ignore_index=True)

    df.to_csv(config.dir_output / df_file, index=False)


def write_nn_inputs(detection: Detector, unique_identifier: str) -> NoReturn:
    for algorithm, result in detection.results.items():

        parent_folder = config.dir_output / 'NN' / algorithm / unique_identifier

        if parent_folder.exists():
            start = int([file for file in parent_folder.glob('*')][-1].stem)
        else:
            start = 1
            utils.create_dirs(parent_folder)

        for index, region in enumerate(result.get_default_regions(), start):
            cv.imwrite(str(parent_folder / f"{utils.zfill_n(index)}.png"), region.as_nn_input(detection.image_not_scaled))
