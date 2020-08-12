import os
import shutil
from pathlib import Path
from typing import Union, Dict, List, NoReturn

import cv2 as cv
import numpy as np
import pandas as pd

import textdetector.config as config
from textdetector.detector import Detector


class Writer:

    def __init__(self):
        self._df_result: pd.DataFrame = pd.DataFrame()
        self._dicts_result: List[Dict[str, Union[int, float]]] = []

        self._output_folder = config.root_folder / "output_python"

        if config.clear_output:
            shutil.rmtree(self._output_folder, ignore_errors=True)

    def update_dataframe(self):
        dict_combined = {}
        for dict_result in self._dicts_result:
            dict_combined.update(dict_result)
        self._dicts_result = []

        self._df_result = self._df_result.append(pd.Series(dict_combined), ignore_index=True)

    def add_dict_result(self, dict_result: Dict[str, Union[int, float]]) -> NoReturn:
        self._dicts_result.append(dict_result)

    def save_dataframe(self):
        self._df_result.to_csv(self._output_folder / "result.csv")

    def save_single_detection(self, detection: Detector, filename: str) -> NoReturn:

        def write_entity(entity: Union[List[np.ndarray], np.ndarray], folder_suffix: str, extension: str) -> NoReturn:
            dst_folder = self._output_folder / folder_suffix
            os.makedirs(str(dst_folder.resolve()), exist_ok=True)
            dst_path = str(dst_folder / Path(filename + f".{extension}"))

            if extension == "png":
                cv.imwrite(dst_path, entity)
            elif extension == "csv":
                np.savetxt(dst_path, entity, delimiter=",", fmt='%i')

        def write_image_region(image: np.ndarray, folder_suffix: str, index: int) -> NoReturn:
            dst_folder = self._output_folder / folder_suffix / filename
            os.makedirs(str(dst_folder.resolve()), exist_ok=True)
            dst_path = str(dst_folder / f"{str(index).zfill(4)}.png")

            cv.imwrite(dst_path, image)

        write_entity(detection.get_text_coordinates(), "coords_text", "csv")
        write_entity(detection.get_words_coordinates(), "coords_word", "csv")

        write_entity(detection.image_word_masks, "masks_word_regions", "png")
        write_entity(detection.image_text_masks, "masks_text_regions", "png")

        for index, text_region in enumerate(detection.word_regions, start=1):
            write_image_region(text_region.image_orig, "parts_word_regions", index)

        for index, text_region in enumerate(detection.text_regions, start=1):
            write_image_region(text_region.image_orig, "parts_text_regions", index)

        if config.visualize:
            write_entity(detection.image_visualization, "visualizations", "png")
