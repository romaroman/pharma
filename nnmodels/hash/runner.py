import os
import logging
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
from redis import Redis

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjectionTree
from nearpy.filters import NearestFilter
from nearpy.distances import EuclideanDistance
from nearpy.storage import RedisStorage

import torch
from torch.utils.data import DataLoader, sampler
import torchvision

from nnmodels.hash import HashEncoder
from nnmodels.datasets import PharmaPackDataset

from textdetector.fileinfo import FileInfo
import utils


logger_name = 'nnmodels | hash'
utils.setup_logger(logger_name, logging.INFO, 'hash.log')
logger = logging.getLogger(logger_name)

parser = argparse.ArgumentParser()
parser.add_argument('src_insert', type=str)
parser.add_argument('src_search', type=str)

parser.add_argument('--base_model', type=str, default="resnet18")
parser.add_argument('--vector_dimension', type=int, default=256)
parser.add_argument('--hash_length', type=int, default=24)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--flush_all', type=bool, default=False)
parser.add_argument('--neighbours', type=int, default=1)
parser.add_argument('--db', type=int, default=0)


def unzip(data, target):
    target = target if len(target) > 0 else None
    if not type(data) in (tuple, list):
        data = (data,)
    if cuda:
        data = tuple(d.cuda() for d in data)
        if target is not None:
            target = target.cuda()
    return data, target


def unzip_d(data):
    if not type(data) in (tuple, list):
        data = (data,)
    if cuda:
        data = tuple(d.cuda() for d in data)
    return data


def insert(loader, nearpy_engine, hash_model):

    with torch.no_grad():
        model.eval()

        bar = tqdm(np.arange(len(loader)), desc='Inserting...', total=len(loader))

        for batch_idx, (data, filenames) in enumerate(loader, start=1):
            data = unzip_d(data)

            tensor_vectors = hash_model(*data)

            for vector, filename in zip(tensor_vectors.cpu().numpy(), filenames):
                nearpy_engine.store_vector(v=vector, data=filename)

            bar.update()

            # logger.info(f'Inserted {batch_idx * args.batch_size} vectors')


def search(loader, nearpy_engine, hash_model):
    results = []

    with torch.no_grad():
        hash_model.eval()
        bar = tqdm(np.arange(len(loader)), desc='Searching...', total=len(loader))

        for batch_idx, (data, filenames) in enumerate(loader, start=1):
            data = unzip_d(data)

            tensor_vectors = hash_model(*data)

            for vector, filename in zip(tensor_vectors.cpu().numpy(), filenames):
                neighbours = nearpy_engine.neighbours(vector)
                # if len(neighbours) == 0:
                #     results.append([filename, None, None])
                # else:
                for neighbour in neighbours:
                    vector_neighbour, predicted, score = neighbour
                    results.append([filename, predicted, score])

            bar.update()

            # logger.info(f'Found {batch_idx * args.batch_size} neighbours')

    return pd.DataFrame(results)


def get_loader(path, batch_size):
    dataset = PharmaPackDataset(path)
    logger.info(f"Image amount is {len(dataset)}")

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler.SubsetRandomSampler(np.arange(len(dataset))),
        num_workers=0,
        drop_last=True,
        shuffle=False
    )


if __name__ == '__main__':
    args = parser.parse_args()

    base_model = torchvision.models.__dict__[args.base_model](pretrained=True)
    model = torch.nn.DataParallel(HashEncoder(base_model, args.vector_dimension))
    cuda = torch.cuda.is_available()
    model.to('cuda') if cuda else model.to('cpu')

    redis_db = Redis(host='localhost', port=6379, db=args.db)
    if args.flush_all:
        answer = input(f"Do you really want to clear {args.db} database [yes/no]?   ")
        if answer.startswith('yes'):
            logger.info(f"Cleared {args.db} database")
            redis_db.flushall()
        else:
            logger.info("Didn't clear storage")

    engine = Engine(
        dim=args.vector_dimension,
        distance=EuclideanDistance(),
        lshashes=[
            RandomBinaryProjectionTree(
                hash_name='rbp_tree',
                minimum_result_size=args.neighbours,
                projection_count=args.hash_length
            ),
            # RandomBinaryProjections(hash_name='rbp', projection_count=args.hash_length)
        ],
        vector_filters=[NearestFilter(24)],
        storage=RedisStorage(redis_db)
    )


    if args.src_insert:
        logger.info("Start inserting")
        loader = get_loader(args.src_insert, args.batch_size)
        insert(loader, engine, model)
        logger.info(f"Finished inserting and inserted {redis_db.dbsize()}")

    if args.src_search:
        logger.info("Start searching")
        loader = get_loader(args.src_search, args.batch_size)
        df = search(loader, engine, model)
        df_res = pd.DataFrame()

        descriptors = ['class', 'phone', 'distinct', 'sample', 'size', 'angle', 'side']

        df_res['filename_actual'] = df[0].astype(str)
        df_res['filename_predicted'] = df[1].astype(str)

        df_res[[f'{desc}_actual' for desc in descriptors]] = df.apply(
            result_type='expand', func=lambda row: FileInfo.get_file_info_by_path(row[0]).to_list(), axis=1
        )
        df_res[[f'{desc}_predicted' for desc in descriptors]] = df.apply(
            result_type='expand', func=lambda row: FileInfo.get_file_info_by_path(row[1]).to_list(), axis=1
        )
        df_res['index_actual'] = df[0].str.split('_').str[-1].str[:4].astype(np.int64)
        df_res['index_predicted'] = df[1].str.split('_').str[-1].str[:4].astype(np.int64)
        df_res['score'] = df[2].astype(np.float)

        dst_folder = "hashmatching"

        os.makedirs(dst_folder, exist_ok=True)
        dst_path = f"{dst_folder}/{args.base_model}_vec{args.vector_dimension}_hash{args.hash_length}.csv"

        if os.path.exists(dst_path):
            os.remove(dst_path)

        df_res.to_csv(dst_path, index=False)
        logger.info(f"Finished searching and written {dst_path}")
