import argparse
import logging
import pickle
import itertools

from tqdm import tqdm
from redis import Redis
from pathlib import Path
from typing import List, Union, NoReturn, Tuple

import pandas as pd
import numpy as np

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections, RandomBinaryProjectionTree, LSHash
from nearpy.distances import EuclideanDistance
from nearpy.storage import RedisStorage

from nnmodels.hash.helpers import encode_image_to_uuid, decode_image_from_uuid
from libs.lopq import LOPQModel, LOPQSearcherLMDB
from textdetector.fileinfo import FileInfo

import utils


parser = argparse.ArgumentParser()

parser.add_argument('dir_complete', type=str)
parser.add_argument('--flushdb', type=bool, default=False)
parser.add_argument('--db_insert', type=int, default=8)
parser.add_argument('--db_complete', type=int, default=6)

logger_name = 'nnmodels | hasher'
utils.setup_logger(logger_name, logging.INFO)
logger = logging.getLogger(logger_name)


def dump_engine_tree_hashes(hashes: List[LSHash], folder):
    folder.mkdir(parents=True, exist_ok=True)

    tree_hashes = list(filter(lambda l: type(l) is RandomBinaryProjectionTree, hashes))
    for tree_hash in tree_hashes:
        with open(folder / f"{tree_hash.hash_name}.pkl", "wb") as f:
            pickle.dump(tree_hash.get_config(), f)

    # logger.info(f"Dumped {len(tree_hashes)} tree hashes")


def load_engine_tree_hashes(folder, minimum_result_size: Union[None, int] = None) -> List[RandomBinaryProjectionTree]:
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


def init_nearpy_engine(
        descriptor_length: int,
        neighbours_amount: int,
        db_insert: Redis
) -> Engine:
    def generate_hash_names(prefix: str) -> List[str]:
        return [
            "+".join([
                prefix, base_model, str(descriptor_length), dir_alg.stem, str(hash_length)
            ]) for hash_length in hash_lengths
        ]

    rbt_hashes = [
        RandomBinaryProjections(hash_name=hash_name, projection_count=hash_length)
        for hash_name, hash_length in zip(generate_hash_names("rbp"), hash_lengths)
    ]
    rbpt_hashes = [
        RandomBinaryProjectionTree(hash_name=hash_name, minimum_result_size=neighbours_amount, projection_count=hash_length)
        for hash_name, hash_length in zip(generate_hash_names("rbpt"), hash_lengths)
    ]

    return Engine(
        dim=descriptor_length,
        distance=EuclideanDistance(),
        lshashes=rbpt_hashes + rbt_hashes,
        storage=RedisStorage(db_insert)
    )


def init_nearpy_search_engine(nearpy_engine_insert: Engine, neighbours_amount: int, dir_alg: Path, descriptor_length: int):
    tree_hashes_updated = load_engine_tree_hashes(dir_alg.parent.parent / 'TreeHashes', neighbours_amount)
    bin_hashes = [lshash for lshash in nearpy_engine_insert.lshashes if type(lshash) is RandomBinaryProjections]

    return Engine(
        dim=descriptor_length,
        distance=nearpy_engine_insert.distance,
        lshashes=bin_hashes + tree_hashes_updated,
        storage=nearpy_engine_insert.storage
    )


def load_descriptors(
        dir_alg: Path, base_model: str, descriptor_length: int, db_complete: Redis
) -> Tuple[List[np.ndarray], List[str], List[np.ndarray], List[str]]:
    descriptors_to_hash, uuids_to_hash, descriptors_to_search, uuids_to_search = list(), list(), list(), list()

    for file in dir_alg.glob('**/*.png'):

        fi = FileInfo.get_file_info_by_path(file)
        uuid = encode_image_to_uuid(base_model, descriptor_length, file)
        descriptor_bytes = db_complete.get(uuid)

        if descriptor_bytes:
            descriptor = pickle.loads(descriptor_bytes)
            if fi.angle == 360:
                uuids_to_hash.append(uuid)
                descriptors_to_hash.append(descriptor)
            else:
                uuids_to_search.append(uuid)
                descriptors_to_search.append(descriptor)

    logger.info(f"Loaded {len(descriptors_to_hash)} to hash and {len(descriptors_to_search)} to search")
    return descriptors_to_hash, uuids_to_hash, descriptors_to_search, uuids_to_search


def hash_nearpy(
        engine: Engine,
        descriptors_to_hash: List[np.ndarray],
        uuids_to_hash: List[str],
        dir_alg: Path,
        base_model: str,
        descriptor_length: int
) -> NoReturn:
    logger.info(f"Start hashing alg={dir_alg.stem} model={base_model} dlen={descriptor_length}")

    total = len(descriptors_to_hash)
    pbar = tqdm(np.arange(total), desc='Hashing', total=total)
    for descriptor, uuid in zip(descriptors_to_hash, uuids_to_hash):
        engine.store_vector(descriptor, uuid)
        pbar.update()

    pbar.close()
    dump_engine_tree_hashes(engine.lshashes, dir_alg.parent.parent / 'TreeHashes')


def search_nearpy(
        nearpy_engine: Engine,
        descriptors_to_search: List[np.ndarray],
        uuids_to_search: List[str],
        dir_alg: Path,
        descriptor_length: int
) -> pd.DataFrame:
    results = []
    
    # for neighbours_amount in range(1, 12, 2):
        
    # engine_search = init_nearpy_search_engine(nearpy_engine_hash, neighbours_amount, dir_alg, descriptor_length)

    total = len(descriptors_to_search)
    pbar = tqdm(np.arange(total), total=total)
    logger.info(f"Start searching alg={dir_alg.stem} model={base_model} dlen={descriptor_length}")

    for descriptor_actual, uuid_actual in zip(descriptors_to_search, uuids_to_search):
        parts_actual = list(decode_image_from_uuid(uuid_actual))

        neighbours = nearpy_engine.neighbours(descriptor_actual)
        for neighbour in neighbours:
            _, uuid_predicted, distance = neighbour
            parts_predicted = list(decode_image_from_uuid(uuid_predicted))
            result = [neighbours_amount, distance] + parts_actual + parts_predicted
            results.append(result)

        pbar.update()
    pbar.close()

    # return pd.DataFrame(results)


# def insert_lopq():
#     lopq_model = LOPQModel()
#     lopq_model.fit(np.asarray(descriptors_to_hash), n_init=1)
#     searcher = LOPQSearcherLMDB(lopq_model, "/data/500gb/pharmapack/LMDB/default.lmdb")
#     pass
#
#
# def search_lopq():
#     pass

def process_single_subset(
        dir_alg: Path, base_model: str, descriptor_length: int, neighbours_amount: int, db_insert: Redis, db_complete: Redis
) -> NoReturn:

    descriptors_to_hash, uuids_to_hash, descriptors_to_search, uuids_to_search = load_descriptors(
        dir_alg, base_model, descriptor_length, db_complete
    )
    nearpy_engine = init_nearpy_engine(descriptor_length, neighbours_amount, db_insert)

    hash_nearpy(nearpy_engine, descriptors_to_hash, uuids_to_hash, dir_alg, base_model, descriptor_length)

    df = search_nearpy(nearpy_engine, descriptors_to_search, uuids_to_search, dir_alg, descriptor_length)

    df_uuid = "+".join([dir_alg.stem, base_model, str(descriptor_length), str(neighbours_amount)])
    df_path = Path(f"pipeline_results/{df_uuid}.csv")
    df_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(df_path, index=False)


base_models = ['resnet18', 'resnet50', 'resnet101']
descriptor_lengths = [256, 512, 1024]
hash_lengths = [64, 96, 128, 160, 256]
neighbours_amounts = [1, 3, 5, 7]


if __name__ == '__main__':
    args = parser.parse_args()
    db_insert = Redis(host='localhost', port=6379, db=args.db_insert)
    db_complete = Redis(host='localhost', port=6379, db=args.db_complete)

    if args.flushdb:
        answer = input(f"Do you really want to clear {args.db} database [yes/no]?   ")
        if answer.startswith('yes'):
            db_insert.flushdb()
            logger.info(f"Cleared {args.db} database")
        else:
            logger.info("Didn't clear storage")

    for subset in itertools.product(*[list(Path(args.dir_complete).glob('*')), base_models, descriptor_lengths, neighbours_amounts]):
        dir_alg, base_model, descriptor_length, neighbours_amount = subset
        if dir_alg.stem == 'MSER':
            continue
        logger.info(f"Start working on alg={dir_alg.stem} model={base_model} dlen={descriptor_length} nbs={neighbours_amount}")
        process_single_subset(dir_alg, base_model, descriptor_length, neighbours_amount, db_insert, db_complete)
