import logging
import argparse
import hashlib
import pickle

import numpy as np

from pathlib import Path
from tqdm import tqdm
from redis import Redis

import torch
from torch.utils.data import DataLoader, sampler

from nnmodels.hash import HashEncoder
from nnmodels.datasets import PharmaPackDataset

import utils


logger_name = 'nnmodels | hash'
utils.setup_logger(logger_name, logging.INFO, 'hash.log')
logger = logging.getLogger(logger_name)

parser = argparse.ArgumentParser()

parser.add_argument('src_insert', type=str)
parser.add_argument('--flushdb', type=bool, default=False)
parser.add_argument('--db', type=int, default=1)

models = ['resnet18', 'resnet50', 'resnet108']


def unzip_d(data):
    if not type(data) in (tuple, list):
        data = (data,)
    if cuda:
        data = tuple(d.cuda() for d in data)
    return data


def get_unique_identifier(base_model: str, vector_size: int, filepath: Path) -> str:
    identifier = "_".join(
        [
            base_model,
            str(vector_size),
            filepath.parent.parent.stem,
            filepath.stem
        ]
    )
    # id = int(hashlib.md5(identifier.encode('utf-8')).hexdigest(), 16)

    return identifier


def insert(loader, hash_model: HashEncoder, db: Redis):

    with torch.no_grad():
        model.eval()

        bar = tqdm(np.arange(len(loader)), desc='Inserting\t', total=len(loader))

        for batch_idx, (data, filepaths) in enumerate(loader, start=1):
            data = unzip_d(data)

            result = hash_model(*data)

            # for vector_size, tensor in result.items():
            for vector, filepath in zip(result.cpu().numpy(), filepaths):
                p_data = pickle.dumps({
                    'vector': vector,
                    'path': filepath,
                    'model': hash_model.base_model,
                })

                id = get_unique_identifier(hash_model.base_model, 256, Path(filepath))
                db.append(id, p_data)

            bar.update()
        bar.close()


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
    redis_db = Redis(host='localhost', port=6379, db=1)

    if args.flushdb:
        answer = input(f"Do you really want to clear {args.db} database [yes/no]?   ")
        if answer.startswith('yes'):
            logger.info(f"Cleared {args.db} database")
            redis_db.flushdb()
        else:
            logger.info("Didn't clear storage")

    logger.info("Start inserting")

    for index, base_model in enumerate(models, start=0):
        model = HashEncoder(base_model, [256, 512, 1024])

        loader = get_loader(args.src_insert, batch_size=4)
        batch_size = dict(resnet18=1024, resnet50=512, resnet108=256).get(base_model, 512)

        cuda = torch.cuda.is_available()
        model = model #  if cuda else model
        model.to('cuda') if cuda else model.to('cpu')

        insert(loader, model, redis_db)
        logger.info(f"Finished inserting and inserted {redis_db.dbsize()}")
