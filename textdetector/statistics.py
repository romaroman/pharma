import glob

import pandas as pd

from textdetector.annotation import Annotation


if __name__ == '__main__':
    SRC_FOLDER = "D:/pharmapack/Enrollment/annotations/"

    df_statistics = pd.DataFrame({
        'filename': pd.Series([], dtype='float'),
        'type': pd.Series([], dtype='int'),
        'label': pd.Series([], dtype='str'),
        'height': pd.Series([], dtype='str'),
        'width': pd.Series([], dtype='float'),
        'area': pd.Series([], dtype='float'),
        'ratio': pd.Series([], dtype='object'),
    })
    annotations = []

    for file in glob.glob(SRC_FOLDER + "/*.xml"):
        annotation = Annotation(file)
        annotations.append(annotation)

        annotation.add_statistics(df_statistics)

    df_statistics.to_csv('stats.csv')
