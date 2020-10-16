import os
import logging
import argparse

import numpy as np
import pandas as pd
from redis import Redis

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.storage import RedisStorage

import torch
from torch.utils.data import DataLoader, sampler
import torchvision

from nnmodels.hash import HashEncoder
from nnmodels.datasets import PharmaPackDataset
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


def unzip(data, target):
    target = target if len(target) > 0 else None
    if not type(data) in (tuple, list):
        data = (data,)
    if cuda:
        data = tuple(d.cuda() for d in data)
        if target is not None:
            target = target.cuda()
    return data, target


def insert(loader, nearpy_engine, hash_model):
    img_index = 0

    with torch.no_grad():
        model.eval()

        for batch_idx, (data, target) in enumerate(loader, start=1):
            data, target = unzip(data, target)

            tensor_vectors = hash_model(*data)
            for vector, actual in zip(tensor_vectors.cpu().numpy(), target.cpu().numpy()):
                nearpy_engine.store_vector(v=vector, data=f"{str(actual)}_{utils.zfill_n(img_index, 12)}")
                img_index += 1

            logger.info(f'Inserted {batch_idx * args.batch_size} vectors')


def search(loader, nearpy_engine, hash_model):
    results = []
    img_index = 0

    with torch.no_grad():
        hash_model.eval()

        for batch_idx, (data, target) in enumerate(loader, start=1):
            data, target = unzip(data, target)

            tensor_vectors = hash_model(*data)

            for vector, actual in zip(tensor_vectors.cpu().numpy(), target.cpu().numpy()):
                neighbours = nearpy_engine.neighbours(vector)
                for neighbour in neighbours:
                    vector_neighbour, predicted, score = neighbour
                    results.append([f"{str(actual)}_{utils.zfill_n(img_index, 12)}", predicted, score])
                img_index += 1

            logger.info(f'Found {batch_idx * args.batch_size} neighbours')

    return pd.DataFrame(results)


def get_loader(path, batch_size):
    dataset = PharmaPackDataset(path)
    logger.info(f"Image amount is {len(dataset)}")

    srs_sampler = sampler.SubsetRandomSampler(list(range(len(dataset))))

    return DataLoader(dataset, batch_size=batch_size, sampler=srs_sampler, num_workers=0, drop_last=True, shuffle=False)


if __name__ == '__main__':
    args = parser.parse_args()

    base_model = torchvision.models.__dict__[args.base_model](pretrained=True)
    model = HashEncoder(base_model, args.vector_dimension)
    cuda = torch.cuda.is_available()
    model.to('cuda') if cuda else model.to('cpu')

    redis_db = Redis(host='localhost', port=6379, db=0)
    if args.flush_all:
        answer = input("Do you really want to clear db [yes/no]?   ")
        if answer.startswith('yes'):
            logger.info("Cleared Redis db")
            redis_db.flushall()
        else:
            logger.info("Didn't clear storage")

    hashes = RandomBinaryProjections('default', projection_count=args.hash_length)
    engine = Engine(args.vector_dimension, lshashes=[hashes], storage=RedisStorage(redis_db))

    if args.src_insert:
        loader = get_loader(args.src_insert, args.batch_size)
        insert(loader, engine, model)
        logger.info(f"Finished inserting and inserted {redis_db.dbsize()}")

    if args.src_search:
        loader = get_loader(args.src_search, args.batch_size)
        df = search(loader, engine, model)
        df_res = pd.DataFrame()

        df_res['actual'] = df[df.columns[0]]# .astype(np.int64)
        df_res['predicted'] = df[df.columns[1]]# .str.slice(0, 3).str.replace('_', '').astype(np.int64)
        df_res['score'] = df[df.columns[2]].astype(np.float)
        # df_res['match'] = df.actual == df.predicted

        dst_folder = "hashmatching"

        os.makedirs(dst_folder, exist_ok=True)
        dst_path = f"{dst_folder}/{args.base_model}_vec{args.vector_dimension}_hash{args.hash_length}.csv"
        if os.path.exists(dst_path):
            os.remove(dst_path)

        df_res.to_csv(dst_path, index=False)
        logger.info(f"Finished searching and written {dst_path}")
