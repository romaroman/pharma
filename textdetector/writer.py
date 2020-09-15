import os
import json
import shutil
from pathlib import Path
from typing import Union, Dict, List, NoReturn, Any

import cv2 as cv
import numpy as np
import pandas as pd

from textdetector import config
from textdetector.detector import Detector
from textdetector.referencer import Referencer


class Writer:

    def __init__(self):
        self.results: Dict[str, Dict[str, Union[int, float]]] = dict()

    def add_dict_result(
        self, blob: str,
        dict_result: Union[Dict[str, Union[int, float]], Dict[str, Dict[int, Dict[str, Union[int, float]]]]]
    ) -> NoReturn:
        self.results[blob] = dict_result

    def clear_current_results(self) -> NoReturn:
        self.results.clear()

    def get_current_results(self) -> Dict[str, Dict[str, Union[int, float]]]:
        return self.results

    @classmethod
    def write_entity(
            cls,
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
            cls.write_json(entity, dst_path)

    @staticmethod
    def write_image_region(image: np.ndarray, folder_suffix: str, filename: str, order: str) -> NoReturn:
        dst_folder = config.dir_output / folder_suffix / filename
        os.makedirs(str(dst_folder.resolve()), exist_ok=True)
        dst_path = str(dst_folder / f"{order}.png")

        cv.imwrite(dst_path, image)

    @classmethod
    def write_json(cls, data: Any, path: str) -> NoReturn:
        with open(path, 'w+') as file:
            json.dump(data, file, indent=2, sort_keys=True)

    def save_all_results(self, detection: Detector, filename: str) -> NoReturn:

        for algorithm, result in detection.results.items():
            for method in result.masks.keys():
                common_part = f"{algorithm}/{method}"
                # self.write_entity(result.get_coordinates_from_regions(method), f"{common_part}/coords", filename, "csv")
                self.write_entity(result.masks[method], f"{common_part}/masks", filename, "png")

                for index, region in enumerate(result.regions[method], start=1):
                    self.write_image_region(
                        region.crop_image(detection.image_not_scaled),
                        f"{common_part}/parts", filename, str(index).zfill(4)
                    )

        self.write_entity(self.results, "JSON", filename, "json")

        if config.wr_visualization:
            self.write_entity(detection.get_visualization(), "visualizations", filename, "png")

    def save_reference_results(self, referencer: Referencer, filename: str) -> NoReturn:
        self.write_entity(referencer.get_coordinates(), f"REF/coords", filename, "csv")

        for label, result in referencer.results.items():
            image_region, _ = result
            self.write_image_region(image_region, f"REF/parts", filename, label)

    @classmethod
    def prepare_output_folder(cls) -> NoReturn:
        if config.wr_clean_before:
            shutil.rmtree(config.dir_output, ignore_errors=True)
            os.makedirs(str(config.dir_output.resolve()), exist_ok=True)

    @staticmethod
    def update_session_with_pd(results: List[Dict[str, Dict[str, Union[int, float]]]]) -> NoReturn:
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
