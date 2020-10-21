import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


logger_name = 'nnmodels | hash'

parser = argparse.ArgumentParser()
parser.add_argument('src', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    df = pd.read_csv(args.src, index_col=None)

    df.rename(
        columns={
            'phone_actual': 'class_actual1', 'class_actual': 'phone_actual1',
            'phone_predicted': 'class_predicted1', 'class_predicted': 'phone_predicted1',
        },
        inplace=True
    )
    df.rename(
        columns={
            'phone_actual1': 'phone_actual', 'class_actual1': 'class_actual',
            'class_predicted1': 'class_predicted', 'phone_predicted1': 'phone_predicted',
        },
        inplace=True
    )

    df_u = df.set_index(['class_actual', 'distinct_actual', 'sample_actual', 'size_actual', 'phone_actual', 'angle_actual'])

    for ind in df_u.index.unique():
        try:
            df_s = df_u.loc[ind]
            df_s = df_s.reset_index()
            df_s = df_s[df_s.score <= 0.2]

            phone_actual = str(df_s.phone_actual.iloc[0])
            class_actual = str(df_s.class_actual.iloc[0]).zfill(4)
            distinct_actual = str(df_s.distinct_actual.iloc[0]).zfill(2)
            sample_actual = str(df_s.sample_actual.iloc[0]).zfill(3)
            size_actual = str(df_s.size_actual.iloc[0])
            angle_actual = str(df_s.angle_actual.iloc[0]).zfill(3)

            filename = f"PFP_Ph{phone_actual}_P{class_actual}_D{distinct_actual}_S{sample_actual}_C{size_actual}_az{angle_actual}_side1"

            df_s['class_distinct_predicted'] = df_s['class_predicted'].astype(str) + "_" + df_s['distinct_predicted'].astype(
                str)
            df_s = df_s[
                ['class_actual', 'distinct_actual', 'phone_actual', 'angle_actual',
                 'class_distinct_predicted', 'index_actual', 'score']
            ]

            df_piv = df_s.pivot_table(
                index=[
                    'class_actual', 'distinct_actual', 'phone_actual', 'angle_actual', 'class_distinct_predicted',
                ],
                fill_value=np.nan,
                aggfunc=lambda x: [s for s in x]
            ).reset_index()

            scores_ravel = df_piv.score.to_numpy()
            indexes_ravel = df_piv.index_actual.to_numpy()
            scores = []
            max_len = max([len(s) for s in scores_ravel])
            for i, (index, score) in enumerate(zip(indexes_ravel, scores_ravel)):
                scores_insert = np.zeros(max_len, dtype=np.float32)
                scores_insert[:len(score)] = sorted(filter(lambda x: x < 0.2, score), reverse=True)
                scores.append(scores_insert)
            scores = np.asarray(scores).T

            predicted = df_piv.class_distinct_predicted.to_list()

            plt.figure(figsize=(10, 10))
            indx = list(range(len(predicted)))

            for index, score in enumerate(scores):
                if index == 0:
                    print(len(indx), len(scores[index]))
                    plt.bar(x=indx, height=scores[index], width=0.4)
                else:
                    plt.bar(x=indx, height=scores[index], bottom=scores[index - 1], width=0.4)

            plt.xticks(indx, predicted, rotation=90)
            plt.title(filename)
            plt.savefig(f"/ndata/chaban/pharmapack/Barplots/{filename}.png")

            plt.cla()
            plt.clf()
            plt.close()
        except:
            pass