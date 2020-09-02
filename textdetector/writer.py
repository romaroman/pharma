import os
import datetime
import json
import shutil
from pathlib import Path
from typing import Union, Dict, List, NoReturn, Any

import cv2 as cv
import numpy as np
import pandas as pd

import textdetector.config as config
from textdetector.detector import Detector

import utils
from textdetector.referencer import Referencer


class Writer:

    def __init__(self):
        self._df_result: pd.DataFrame = pd.DataFrame()

        self._dicts_result: Dict[str, Dict[str, Union[int, float]]] = dict()
        self._failed_files: Dict[str, str] = dict()

        self._output_folder = config.root_folder / "output_python" / utils.get_str_timestamp()

        if config.clear_output:
            shutil.rmtree(self._output_folder, ignore_errors=True)
        os.makedirs(self._output_folder, exist_ok=True)

        self._save_config()

    def update_dataframe(self):
        dict_combined = dict()

        for key, dict_result in self._dicts_result.items():
            dict_combined.update(dict_result)

        self._clear_current_results()
        self._df_result = self._df_result.append(pd.Series(dict_combined), ignore_index=True)

        if len(self._df_result.index) % 10:
            self.save_dataframe()

    def add_dict_result(
        self, blob: str,
        dict_result: Union[Dict[str, Union[int, float]], Dict[str, Dict[int, Dict[str, Union[int, float]]]]]
    ) -> NoReturn:
        self._dicts_result[blob] = dict_result

    def add_failed_file(self, file: str, error: str) -> NoReturn:
        self._failed_files[file] = error

    def save_dataframe(self):
        self._df_result.to_csv(self._output_folder / "result.csv", index=False)

    def _clear_current_results(self):
        self._dicts_result.clear()

    def _write_entity(
            self,
            entity: Union[List[np.ndarray], np.ndarray, int, Dict[str, Dict[str, Union[int, float]]]],
            folder_suffix: str,
            filename: str,
            extension: str
    ) -> NoReturn:
        dst_folder = self._output_folder / folder_suffix
        os.makedirs(str(dst_folder.resolve()), exist_ok=True)
        dst_path = str(dst_folder / Path(filename + f".{extension}"))

        if extension == "png":
            cv.imwrite(dst_path, entity)

        elif extension == "csv":
            np.savetxt(dst_path, entity, delimiter=",", fmt='%i')

        elif extension == "json":
            self._write_json(entity, dst_path)

    def _write_image_region(self, image: np.ndarray, folder_suffix: str, filename: str, order: str) -> NoReturn:
        dst_folder = self._output_folder / folder_suffix / filename
        os.makedirs(str(dst_folder.resolve()), exist_ok=True)
        dst_path = str(dst_folder / f"{order}.png")

        cv.imwrite(dst_path, image)

    @classmethod
    def _write_json(cls, data: Any, path: str) -> NoReturn:
        with open(path, 'w+') as file:
            json.dump(data, file, indent=2, sort_keys=True)

    def save_all_results(self, detection: Detector, filename: str) -> NoReturn:

        for algorithm, result in detection.results.items():
            image_mask, regions = result

            self._write_entity(detection.get_coordinates_from_regions(regions), f"{algorithm}/coords", filename, "csv")
            self._write_entity(image_mask, f"{algorithm}/masks", filename, "png")

            for index, region in enumerate(regions, start=1):
                self._write_image_region(region.image_orig, f"{algorithm}/parts", filename, str(index).zfill(4))

        self._write_entity(self._dicts_result, "jsons", filename, "json")

        if config.visualize:
            self._write_entity(detection.create_visualization(), "visualizations", filename, "png")

    def save_reference_results(self, referencer: Referencer, filename: str) -> NoReturn:
        self._write_entity(referencer.get_coordinates(), f"reference/coords", filename, "csv")

        for label, result in referencer.dict_results.items():
            image_region, _ = result
            self._write_image_region(image_region, f"reference/parts", filename, label)


    def _save_config(self):
        self._write_json(config.to_dict(), str(self._output_folder / "config.json"))
