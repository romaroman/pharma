import logging
import argparse
import pickle

import numpy as np

from pathlib import Path
from tqdm import tqdm
from redis import Redis

import torch
from torch.utils.data import DataLoader, sampler

from nnmodels.hash import HashEncoder
from nnmodels.datasets import PharmaPackDataset
from nnmodels.hash.helpers import get_image_uuid

import utils


logger_name = 'nnmodels | inserter'
utils.setup_logger(logger_name, logging.INFO, 'hash.log')
logger = logging.getLogger(logger_name)

parser = argparse.ArgumentParser()

parser.add_argument('dir_complete', type=str)
parser.add_argument('--flushdb', type=bool, default=False)
parser.add_argument('--db', type=int, default=6)

base_models = ['resnet18']#, 'resnet50', 'resnet101']
vectors = [256]#, 512, 1024]

def unzip_d(data):
    if not type(data) in (tuple, list):
        data = (data,)
    if cuda:
        data = tuple(d.cuda() for d in data)
    return data


def insert(loader, hash_model: HashEncoder, db: Redis):
    parallel_model = torch.nn.DataParallel(hash_model)

    with torch.no_grad():
        model.eval()

        bar = tqdm(np.arange(len(loader)), desc='Inserting', total=len(loader))

        for batch_idx, (data, filepaths) in enumerate(loader, start=1):
            data = unzip_d(data)

            result = parallel_model(*data)

            for vector_size, tensor in zip(hash_model.output_sizes, result):
                for vector, filepath in zip(tensor.cpu().numpy(), filepaths):
                    p_data = pickle.dumps({
                        'vector': vector,
                        'path': filepath,
                        'model': hash_model.base_model,
                    })
                    uuid = get_image_uuid(hash_model.base_model, vector_size, Path(filepath))
                    db.append(uuid, p_data)

            bar.update()
        bar.close()


if __name__ == '__main__':
    args = parser.parse_args()
    redis_db = Redis(host='localhost', port=6379, db=args.db)

    if args.flushdb:
        answer = input(f"Do you really want to flush db #{args.db}  [yes/no]?   ")
        if answer.startswith('yes'):
            logger.info(f"Flushed db #{args.db} ")
            redis_db.flushdb()
        else:
            logger.info(f"Didn't flush db #{args.db}")

    for dir_alg in Path(args.dir_complete).glob('*'):
        for base_model in base_models:
            # batch_size = dict(resnet18=1024, resnet50=1024, resnet101=1024).get(base_model, 512)
            batch_size = 4

            dataset = PharmaPackDataset(dir_alg)
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler.SubsetRandomSampler(np.arange(len(dataset))),
                num_workers=0,
                drop_last=False,
                shuffle=False
            )

            model = HashEncoder(base_model, vectors)
            cuda = torch.cuda.is_available()
            model.to('cuda') if cuda else model.to('cpu')

            logger.info(f"Start inserting model={base_model} alg={dir_alg.stem} ba={len(loader)} bs={batch_size}")
            insert(loader, model, redis_db)
            logger.info(f"Finished inserting and inserted {redis_db.dbsize()}")
