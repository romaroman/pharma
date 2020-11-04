import argparse
import logging
import pickle

from itertools import repeat
from functools import partial
from multiprocessing import Pool
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
from nearpy.filters import NearestFilter

from libs.lopq import LOPQModel, LOPQSearcher

import utils


parser = argparse.ArgumentParser()

parser.add_argument('--flushdb', type=bool, default=False)
parser.add_argument('--db_insert', type=int, default=8)
parser.add_argument('--db_complete', type=int, default=6)
parser.add_argument('--neighbours_amount', type=int, default=5)
parser.add_argument('--base_model', type=str, default='resnet18')
parser.add_argument('--descriptor_length', type=int, default=256)
parser.add_argument('--alg', type=str, default='MI1')

parser.add_argument('--V', type=int, default=16)
parser.add_argument('--M', type=int, default=16)
parser.add_argument('--clusters', type=int, default=1024)



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
        base_model: str,
        alg: str,
        db_insert: Redis
) -> Engine:
    def generate_hash_names(prefix: str) -> List[str]:
        return [
            "+".join([
                prefix, base_model, str(descriptor_length), alg, str(hash_length)
            ]) for hash_length in hash_lengths
        ]

    rbt_hashes = [
        RandomBinaryProjections(hash_name=hash_name, projection_count=hash_length)
        for hash_name, hash_length in zip(generate_hash_names("rbp"), hash_lengths)
    ]
    # rbpt_hashes = [
    #     RandomBinaryProjectionTree(hash_name=hash_name, minimum_result_size=neighbours_amount, projection_count=hash_length)
    #     for hash_name, hash_length in zip(generate_hash_names("rbpt"), hash_lengths)
    # ]

    return Engine(
        dim=descriptor_length,
        distance=EuclideanDistance(),
        vector_filters=NearestFilter(neighbours_amount),
        lshashes=rbt_hashes,
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
        alg: str, base_model: str, descriptor_length: int, db_complete: Redis
) -> Tuple[List[np.ndarray], List[str], List[np.ndarray], List[str]]:
    descriptors_to_hash, uuids_to_hash, descriptors_to_search, uuids_to_search = list(), list(), list(), list()

    keys = db_complete.keys(f"{base_model}+{descriptor_length}+{alg}*")

    total = len(keys)
    pbar = tqdm(np.arange(total), total=total, desc='Loading')

    for key in keys:
        descriptor_bytes = db_complete.get(key)

        if descriptor_bytes:
            descriptor = pickle.loads(descriptor_bytes)
            key_str = key.decode("utf-8")
            if key_str.find("az360") != -1:
                uuids_to_hash.append(key_str)
                descriptors_to_hash.append(descriptor)
            else:
                uuids_to_search.append(key_str)
                descriptors_to_search.append(descriptor)

        pbar.update()
    pbar.close()

    logger.info(f"Loaded {len(descriptors_to_hash)} to hash and {len(descriptors_to_search)} to search")
    return descriptors_to_hash, uuids_to_hash, descriptors_to_search, uuids_to_search


def load_descriptors_l2(
        alg: str, base_model: str, descriptor_length: int, db_complete: Redis
) -> Tuple[List[np.ndarray], List[str], List[np.ndarray], List[str]]:
    descriptors_based, uuids_based, descriptors_to_search, uuids_to_search = list(), list(), list(), list()

    keys_o = db_complete.keys(f"{base_model}+{descriptor_length}+{alg}*Ph1*")
    keys = []
    for key in keys_o:
        key_s = key.decode('utf-8')
        if key_s.find('0842_D01_S001') != -1 or\
                key_s.find('0235_D01_S001') != -1 or\
                key_s.find('0457_D01_S001') != -1 or\
                key_s.find('0131_D01_S001') != -1 or\
                key_s.find('0715_D01_S001') != -1 or\
                key_s.find('0174_D01_S001') != -1 or\
                key_s.find('0701_D01_S001') != -1 or\
                key_s.find('0475_D01_S001') != -1 or\
                key_s.find('0102_D01_S001') != -1:
            keys.append(key)

    total = len(keys)
    pbar = tqdm(np.arange(total), total=total, desc='Loading')
    for key in keys:
        descriptor_bytes = db_complete.get(key)

        if descriptor_bytes:
            descriptor = pickle.loads(descriptor_bytes)
            key_str = key.decode("utf-8")
            if key_str.find("az0") != -1:
                uuids_based.append(key_str)
                descriptors_based.append(descriptor)
            elif key_str.find("az3") != -1:
                uuids_to_search.append(key_str)
                descriptors_to_search.append(descriptor)

        pbar.update()
    pbar.close()
    logger.info(f"Loaded {len(descriptors_based)} based and {len(descriptors_to_search)} to search")

    return descriptors_based, uuids_based, descriptors_to_search, uuids_to_search


def hash_nearpy(
        engine: Engine,
        descriptors_to_hash: List[np.ndarray],
        uuids_to_hash: List[str],
) -> NoReturn:

    total = len(descriptors_to_hash)
    pbar = tqdm(np.arange(total), desc='Hashing', total=total)
    for descriptor, uuid in zip(descriptors_to_hash, uuids_to_hash):
        engine.store_vector(descriptor, uuid)
        pbar.update()

    pbar.close()
    # dump_engine_tree_hashes(engine.lshashes, dir_alg.parent.parent / 'TreeHashes')


def search_nearpy(
        nearpy_engine: Engine,
        descriptors_to_search: List[np.ndarray],
        uuids_to_search: List[str],
) -> pd.DataFrame:
    results = []

    total = len(descriptors_to_search)
    pbar = tqdm(np.arange(total), desc='Searching', total=total)

    for descriptor_actual, uuid_actual in zip(descriptors_to_search, uuids_to_search):
        filename_actual = (uuid_actual).split('+')[-1]

        neighbours = nearpy_engine.neighbours(descriptor_actual)
        for neighbour in neighbours:
            _, uuid_predicted, distance = neighbour
            filename_predicted = (uuid_predicted).split('+')[-1]
            results.append([distance, filename_actual, filename_predicted])

        pbar.update()
    pbar.close()

    return pd.DataFrame(results)


def search_lopq(
    descriptors_train_lopq: List[np.ndarray],
    uuids_train_lopq: List[str],
    descriptors_search_lopq: List[np.ndarray],
    uuids_search_lopq: List[str],
) -> NoReturn:
    train_array = np.asarray(descriptors_train_lopq)

    lopq_model = LOPQModel(V=args.V, M=args.M, subquantizer_clusters=args.clusters)
    lopq_model.fit(train_array, n_init=1)

    lopq_searcher = LOPQSearcher(model=lopq_model)
    lopq_searcher.add_data(train_array, ids=uuids_train_lopq)

    results = list()
    total = len(descriptors_search_lopq)
    pbar = tqdm(np.arange(total), desc='Searching', total=total)
    
    for descriptor_search, uuid_search in zip(descriptors_search_lopq, uuids_search_lopq):
        neighbours, _ = lopq_searcher.search(descriptor_search, quota=args.neighbours_amount, with_dists=True)
        neighbours = list(neighbours)
        if neighbours:
            for neighbour in neighbours:
                results.append([uuid_search, neighbour.id, neighbour.dist])
        else:
            results.append([uuid_search, np.nan, np.nan])

        pbar.update()
    pbar.close()

    return pd.DataFrame(results)


def calc_l2(based_tuple, search_tuple):
    descriptor_based, uuid_based = based_tuple
    descriptor_to_search, filename_search = search_tuple
    filename_based = (uuid_based).split('+')[-1]
    # distance = np.linalg.norm(descriptor_to_search['vector'] - descriptor_based['vector'])
    distance = np.linalg.norm(descriptor_to_search - descriptor_based)
    return [distance, filename_search, filename_based]


def search_l2(
    descriptors_based: List[np.ndarray],
    uuids_based: List[str],
    descriptors_to_search: List[np.ndarray],
    uuids_to_search: List[str],
):
    results = []

    total = len(descriptors_to_search)
    pbar = tqdm(np.arange(total), desc='Searching L2', total=total)

    for descriptor_to_search, uuid_to_search in zip(descriptors_to_search, uuids_to_search):
        filename_search = (uuid_to_search).split('+')[-1]

        pool = Pool(processes=46)

        res = pool.map(
            partial(calc_l2, search_tuple=(descriptor_to_search, filename_search)),
            zip(descriptors_based, uuids_based)
        )
        pool.close()
        results.extend(res)


        pbar.update()
    pbar.close()

    return pd.DataFrame(results)


def process_single_subset(
        alg: str,
        base_model: str,
        descriptor_length: int,
        db_insert: Redis,
        db_complete: Redis
) -> NoReturn:
    logger.info(f"Start working on alg={alg} model={base_model} dlen={descriptor_length}")

    descriptors_train_lopq, uuids_train_lopq, descriptors_search_lopq, uuids_search_lopq = load_descriptors(
        alg, base_model, descriptor_length, db_complete
    )

    df = search_lopq(descriptors_train_lopq, uuids_train_lopq, descriptors_search_lopq, uuids_search_lopq)

    df_uuid = "+".join(["lopq", alg, base_model, str(descriptor_length)])
    df_path = Path(f"lopq/{df_uuid}.csv")
    df_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(df_path, index=False)

    # descriptors_based_l2, uuids_based_l2, descriptors_to_search_l2, uuids_to_search_l2 = load_descriptors_l2(
    #     dir_alg, base_model, descriptor_length, db_complete
    # )
    # df = search_l2(descriptors_based_l2, uuids_based_l2, descriptors_to_search_l2, uuids_to_search_l2)
    # df_uuid = "+".join([dir_alg.stem, base_model, str(descriptor_length)])
    # df_path = Path(f"l2_results/{df_uuid}.csv")
    # df_path.parent.mkdir(parents=True, exist_ok=True)
    #
    # df.to_csv(df_path, index=False)

    # for neighbours_amount in neighbours_amounts:
    #     nearpy_engine = init_nearpy_engine(descriptor_length, neighbours_amount, base_model, dir_alg.stem, db_insert)
    #
    #     hash_nearpy(nearpy_engine, descriptors_to_hash, uuids_to_hash)
    #
    #     df = search_nearpy(nearpy_engine, descriptors_to_search, uuids_to_search)
    #
    #     df_uuid = "+".join([dir_alg.stem, base_model, str(descriptor_length), str(neighbours_amount)])
    #     df_path = Path(f"pipeline_results/{df_uuid}.csv")
    #     df_path.parent.mkdir(parents=True, exist_ok=True)
    #
    #     df.to_csv(df_path, index=False)


base_models = ['resnet18', 'resnet50', 'resnet101']
descriptor_lengths = [256, 512, 1024]
hash_lengths = [64, 96, 128, 160, 256]


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

    # for subset in itertools.product(*[list(Path(args.dir_complete).glob('*')), base_models, descriptor_lengths]):
    #     dir_alg, base_model, descriptor_length = subset
    #     if dir_alg.stem == 'MSER':
    #         continue
    #     process_single_subset(dir_alg, base_model, descriptor_length, db_insert, db_complete)

    process_single_subset(args.alg, args.base_model, args.descriptor_length, db_insert, db_complete)
