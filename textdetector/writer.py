import os
import json
import shutil
from pathlib import Path
from typing import Union, Dict, List, NoReturn, Any

import cv2 as cv
import numpy as np
import pandas as pd

import textdetector.config as config
from textdetector.detector import Detector
from textdetector.referencer import Referencer


class Writer:

    def __init__(self):
        self._dicts_result: Dict[str, Dict[str, Union[int, float]]] = dict()

    def add_dict_result(
        self, blob: str,
        dict_result: Union[Dict[str, Union[int, float]], Dict[str, Dict[int, Dict[str, Union[int, float]]]]]
    ) -> NoReturn:
        self._dicts_result[blob] = dict_result

    def clear_current_results(self) -> NoReturn:
        self._dicts_result.clear()

    def get_current_results(self) -> Dict[str, Dict[str, Union[int, float]]]:
        return self._dicts_result

    @classmethod
    def write_entity(
            cls,
            entity: Union[List[np.ndarray], np.ndarray, int, Dict[str, Dict[str, Union[int, float]]]],
            folder_suffix: str,
            filename: str,
            extension: str
    ) -> NoReturn:
        dst_folder = config.dst_folder / folder_suffix
        os.makedirs(str(dst_folder.resolve()), exist_ok=True)
        dst_path = str(dst_folder / Path(filename + f".{extension}"))

        if extension == "png":
            cv.imwrite(dst_path, entity)

        elif extension == "csv":
            np.savetxt(dst_path, entity, delimiter=",", fmt='%i')

        elif extension == "json":
            cls.write_json(entity, dst_path)

    @staticmethod
    def write_image_region(image: np.ndarray, folder_suffix: str, filename: str, order: str) -> NoReturn:
        dst_folder = config.dst_folder / folder_suffix / filename
        os.makedirs(str(dst_folder.resolve()), exist_ok=True)
        dst_path = str(dst_folder / f"{order}.png")

        cv.imwrite(dst_path, image)

    @classmethod
    def write_json(cls, data: Any, path: str) -> NoReturn:
        with open(path, 'w+') as file:
            json.dump(data, file, indent=2, sort_keys=True)

    def save_all_results(self, detection: Detector, filename: str) -> NoReturn:

        for algorithm, result in detection.results.items():
            image_mask, regions = result

            self.write_entity(detection.get_coordinates_from_regions(regions), f"{algorithm}/coords", filename, "csv")
            self.write_entity(image_mask, f"{algorithm}/masks", filename, "png")

            for index, region in enumerate(regions, start=1):
                self.write_image_region(region.image_orig, f"{algorithm}/parts", filename, str(index).zfill(4))

        self.write_entity(self._dicts_result, "jsons", filename, "json")

        if config.visualize:
            self.write_entity(detection.create_visualization(), "visualizations", filename, "png")

    def save_reference_results(self, referencer: Referencer, filename: str) -> NoReturn:
        self.write_entity(referencer.get_coordinates(), f"reference/coords", filename, "csv")

        for label, result in referencer._results.items():
            image_region, _ = result
            self.write_image_region(image_region, f"reference/parts", filename, label)

    @classmethod
    def prepare_output_folder(cls) -> NoReturn:
        if config.clear_output:
            shutil.rmtree(config.dst_folder, ignore_errors=True)
            os.makedirs(config.dst_folder, exist_ok=True)
        cls.write_json(config.to_dict(), str(config.dst_folder / "config.json"))

    @staticmethod
    def update_session(results: List[Dict[str, Dict[str, Union[int, float]]]]) -> NoReturn:
        df_file = "session.csv"

        if os.path.exists(config.dst_folder / df_file):
            df = pd.read_csv(config.dst_folder / df_file, index_col=False)
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

        df.to_csv(config.dst_folder / df_file, index=False)
