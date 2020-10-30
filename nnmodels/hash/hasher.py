import logging
import argparse
import itertools
import pickle

from tqdm import tqdm
from redis import Redis
from pathlib import Path
from typing import List, Union

import pandas as pd
import numpy as np

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections, RandomBinaryProjectionTree, LSHash
from nearpy.distances import EuclideanDistance
from nearpy.storage import RedisStorage

from nnmodels.hash.helpers import get_image_uuid
from libs.lopq import LOPQModel, LOPQSearcherLMDB
from textdetector.fileinfo import FileInfo

import utils


logger_name = 'nnmodels | hasher'
utils.setup_logger(logger_name, logging.INFO)
logger = logging.getLogger(logger_name)

parser = argparse.ArgumentParser()

parser.add_argument('dir_complete', type=str)
parser.add_argument('--flushdb', type=bool, default=False)
parser.add_argument('--db_insert', type=int, default=8)
parser.add_argument('--db_complete', type=int, default=6)

base_models = ['resnet18']#, 'resnet50', 'resnet101']
vector_sizes = [256]#, 512, 1024]
hash_lengths = [64, 96, 128, 160, 256]


def dump_hashes(hashes: List[LSHash], folder):
    folder.mkdir(parents=True, exist_ok=True)

    for hash in filter(lambda l: type(l) is RandomBinaryProjectionTree, hashes):
        with open(folder / f"{hash.hash_name}.pkl", "wb") as f:
            pickle.dump(hash.get_config(), f)


def load_hashes(folder, minimum_result_size: Union[None, int] = None) -> List[LSHash]:
    hashes = []

    for file in folder.glob('*.pkl'):
        with open(file, "rb") as f:
            data = pickle.load(f)
            hash = RandomBinaryProjectionTree(data['hash_name'], data['projection_count'], data['minimum_result_size'])
            hash.apply_config(data)

            if minimum_result_size:
                hash.minimum_result_size = minimum_result_size

            hashes.append(hash)

    return hashes


def get_vector_by_file(db_complete: Redis, file: Path) -> np.ndarray:
    return pickle.loads(db_complete.get(get_image_uuid(base_model, vector_size, file)))['vector']


if __name__ == '__main__':
    args = parser.parse_args()
    db_insert = Redis(host='localhost', port=6379, db=args.db_insert)
    db_complete = Redis(host='localhost', port=6379, db=args.db_complete)

    if args.flushdb:
        answer = input(f"Do you really want to clear {args.db} database [yes/no]?   ")
        if answer.startswith('yes'):
            logger.info(f"Cleared {args.db} database")
            db_insert.flushdb()
        else:
            logger.info("Didn't clear storage")

    dir_complete = Path(args.dir_complete)
    dirs_alg = list(dir_complete.glob('*'))

    for index, (dir_alg, base_model, vector_size) in \
            enumerate(itertools.product(*[dirs_alg, base_models, vector_sizes]), start=0):

        generate_hash_names = lambda prefix: ["+".join([prefix, base_model, str(vector_size), dir_alg.stem, str(hash_length)]) for hash_length in hash_lengths]

        hash_names = generate_hash_names("rbp")
        rbt_hashes = [
            RandomBinaryProjections(hash_name=hash_name, projection_count=hash_length)
            for hash_name, hash_length in zip(hash_names, hash_lengths)
        ]

        hash_names = generate_hash_names("rbpt")
        rbpt_hashes = [
            RandomBinaryProjectionTree(hash_name=hash_name, minimum_result_size=1, projection_count=hash_length)
            for hash_name, hash_length in zip(hash_names, hash_lengths)
        ]
        engine = Engine(
            dim=vector_size,
            distance=EuclideanDistance(),
            lshashes=rbpt_hashes + rbt_hashes,
            storage=RedisStorage(db_insert)
        )

        files_to_hash, files_to_search = list(), list()
        for file in dir_alg.glob('**/*.png'):
            fi = FileInfo.get_file_info_by_path(file)

            if fi.angle == 360:
                files_to_hash.append(file)
            else:
                files_to_search.append(file)

        logger.info(f"Start hashing model={base_model} alg={dir_alg.stem} vs={vector_size}")

        total = len(files_to_hash)
        pbar = tqdm(np.arange(total), desc='Hashing', total=total)

        vectors = []
        for file in files_to_hash:
            uuid = get_image_uuid(base_model, vector_size, file)
            data = pickle.loads(db_complete.get(uuid))
            engine.store_vector(data['vector'], uuid)
            vectors.append(data['vector'])
            pbar.update()

        pbar.close()
        m = LOPQModel(V=8, M=8, subquantizer_clusters=4)
        m.fit(np.asarray(vectors), n_init=1)
        dump_hashes(engine.lshashes, dir_complete.parent / 'TreeHashes')

        min_neighbours = list(range(1, 15, 3))
        total = len(files_to_search) * len(min_neighbours)
        pbar = tqdm(np.arange(total), desc='Searching', total=total)

        results = []
        for min_neighbour in min_neighbours:
            rbpt_hashes_updated = load_hashes(dir_complete.parent / 'TreeHashes', min_neighbour)

            engine_search = Engine(
                dim=vector_size,
                distance=engine.distance,
                lshashes=rbt_hashes + rbpt_hashes_updated,
                storage=engine.storage
            )

            for file in files_to_search:
                uuid = get_image_uuid(base_model, vector_size, file)
                data = pickle.loads(db_complete.get(uuid))
                neighbours = engine_search.neighbours(data['vector'])
                results.append([min_neighbour, file, [n[1:] for n in neighbours]])

                pbar.update()

        pbar.close()

        pd.DataFrame(results)
