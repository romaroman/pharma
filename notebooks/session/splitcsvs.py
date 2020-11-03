import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool


# class_num = 998
# total = len(csvs) * class_num
# pbar = tqdm(np.arange(total), total=total, desc='Splitting')


def split_csv(csv_path):
    df_orig = pd.read_csv(csv_path, index_col=None)
    df_orig = df_orig.drop(['1','2','3','5','6','7'], axis=1)
    df_orig['4'] = df_orig['4'].str.replace('S0010', 'S010')
    df_orig['8'] = df_orig['8'].str.replace('S0010', 'S010')

    df = pd.DataFrame()
    df['score'] = df_orig['0'].astype(np.float)
    df['index'] = df_orig['4'].str[4:31]
    df['id_actual'] = df_orig['4'].str[8:17]
    df['index_actual'] = df_orig['4'].str[38:].astype(np.int)
    df['id_predicted'] = df_orig['8'].str[8:17]
    
    parent_dir = (Path('separate') / csv_path.stem)
    parent_dir.mkdir(parents=True, exist_ok=True)
    dfs = [x for _, x in df.groupby('index')]

    pbar = tqdm(np.arange(1000), total=1000, desc=f'Splitting {csv_path.stem}')
    for df_r in dfs:
        index_single = df_r['index'].iloc[0]
        id_actual_single = df_r['id_actual'].iloc[0]
        df_r = df_r.drop(['index', 'id_actual'], axis=1)
        df_r.to_csv(parent_dir / f'{index_single}_{id_actual_single}.csv', index=False)
        pbar.update()
    pbar.close()
#         print(f"{csv_path.stem}\t{index_single}\t{id_actual_single}")
        
if __name__ == '__main__':
    csvs = list(Path('pipeline_results').glob('*.csv'))
    pool = Pool(processes=len(csvs))
    pool.map(split_csv, csvs)