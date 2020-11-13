import json
import shutil
from pathlib import Path
from typing import Union, Dict, List, NoReturn, Any

import cv2 as cv
import numpy as np

from common import config

from segmentation.segmenter import Segmenter
from segmentation.fileinfo import FileInfo
from segmentation.referencer import Referencer

import utils


def write_entity(
        entity: Union[List[np.ndarray], np.ndarray, int, Dict[str, Dict[str, Union[int, float]]]],
        folder_suffix: str,
        filename: str,
        extension: str
) -> NoReturn:
    dst_folder = config.general.dir_output / folder_suffix
    dst_folder.mkdir(parents=True, exist_ok=True)

    dst_path = str(dst_folder / Path(filename + f".{extension}"))

    if extension == "png":
        cv.imwrite(dst_path, entity)

    elif extension == "csv":
        np.savetxt(dst_path, entity, delimiter=',', fmt='%i')

    elif extension == "json":
        write_json(entity, dst_path)


def write_image_region(image: np.ndarray, folder_suffix: str, filename: str, order: str) -> NoReturn:
    dst_folder = config.general.dir_output / folder_suffix / filename
    dst_folder.mkdir(parents=True, exist_ok=True)
    dst_path = dst_folder / f"{order}.png"

    cv.imwrite(str(dst_path), image)


def write_json(data: Any, path: str) -> NoReturn:
    with open(path, 'w+') as file:
        json.dump(data, file, indent=2, sort_keys=True)


def save_segmentation_results(segmenter: Segmenter, fileinfo: FileInfo) -> NoReturn:
    if not config.general.det_write:
        return

    for algorithm, result in segmenter.results.items():
        for method in result.masks.keys():

            common_part = f"{algorithm.blob()}/{method}"
            if 'mask' in config.general.det_write:
                write_entity(result.masks[method], f"{common_part}/masks", fileinfo.filename, "png")

            if 'regions' in config.general.det_write:
                for index, region in enumerate(result.regions[method], start=1):
                    write_image_region(
                        region.crop_image(segmenter.image_not_scaled),
                        f"{common_part}/regions", fileinfo.filename, utils.zfill_n(index)
                    )

        if 'nn' in config.segmentation.write:
            parent_folder = config.general.dir_output / "NN" / algorithm.blob() / fileinfo.get_unique_identifier()

            parent_folder.mkdir(parents=True, exist_ok=True)

            for index, region in enumerate(result.get_default_regions(), start=1):
                cv.imwrite(
                    str(parent_folder / f"{fileinfo.filename}_{utils.zfill_n(index)}.png"),
                    region.as_nn_input(segmenter.image_not_scaled)
                )

        if 'verref' in config.segmentation.write:
            segmenter.save_results(config.general.dir_source / "VerificationReferences" / fileinfo.filename)


def save_reference_results(referencer: Referencer, filename: str) -> NoReturn:
    if not config.segmentation.extract_reference_write or not config.segmentation.extract_reference:
        return

    for label, image in referencer.results.items():
        write_image_region(image, f"REF/regions", filename, label)


def prepare_output_folder() -> NoReturn:
    if config.general.clear_dir_output:
        shutil.rmtree(config.general.dir_output, ignore_errors=True)
        config.general.dir_output.mkdir(parents=True, exist_ok=True)
