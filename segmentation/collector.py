from typing import NoReturn, Dict, Any, List

import pandas as pd

from common import config


class Collector:

    def __init__(self) -> NoReturn:
        self.storage: pd.DataFrame = pd.DataFrame()

    def add_result(self, result: Dict[str, Any]) -> NoReturn:
        dict_combined = dict()

        for key_general, dict_result in result.items():
            dict_result_general = dict()
            for key_short, value in dict_result.items():
                dict_result_general[f"{key_general}_{key_short}"] = value

            dict_combined.update(dict_result_general)

        self.storage = self.storage.append(pd.Series(dict_combined), ignore_index=True)

    def add_results(self, results: List[Dict[str, Any]]) -> NoReturn:
        for result in results:
            self.add_result(result)

    def dump(self) -> NoReturn:
        df_file = f"session_{config.general.timestamp}.csv"
        dst_path = config.general.dir_output / df_file

        if dst_path.exists():
            pd.concat([pd.read_csv(dst_path, index_col=None), self.storage]).to_csv(dst_path, index=False)
        else:
            self.storage.to_csv(dst_path, index=False)

        self.storage = self.storage[0:0]
