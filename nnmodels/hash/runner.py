import os
import logging
import argparse

import pandas as pd
from redis import Redis

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.storage import RedisStorage

import torch
import torchvision

from nnmodels.hash import ResNet18Hash
from nnmodels.datasets import PharmaPackDataset, get_train_validation_data_loaders
import utils


logger_name = 'nnmodels | hash'
utils.setup_logger(logger_name, logging.INFO, 'hash.log')
logger = logging.getLogger(logger_name)

parser = argparse.ArgumentParser()
parser.add_argument('src', type=str)
parser.add_argument('dst', type=str)
parser.add_argument('insert', type=bool)
parser.add_argument('search', type=bool)

parser.add_argument('--vector_dimension', type=int, default=256)
parser.add_argument('--hash_length', type=int, default=24)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--test_size', type=float, default=0.1)
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
                nearpy_engine.store_vector(v=vector, data=f"{str(actual)}_{utils.zfill_n(img_index, 9)}")
                img_index += 1

            logger.info(f'Inserted {batch_idx * args.batch_size} vectors')


def search(loader, nearpy_engine, hash_model):
    results = []

    with torch.no_grad():
        hash_model.eval()

        for batch_idx, (data, target) in enumerate(loader):
            data, target = unzip(data, target)

            tensor_vectors = hash_model(*data)

            for vector, actual in zip(tensor_vectors.cpu().numpy(), target.cpu().numpy()):
                neighbours = nearpy_engine.neighbours(vector)
                for neighbour in neighbours:
                    vector_neighbour, predicted, score = neighbour
                    results.append([actual, predicted, score])

            logger.info(f'Found {batch_idx * args.batch_size} neighbours')

    return pd.DataFrame(results)


if __name__ == '__main__':
    args = parser.parse_args()

    resnet = torchvision.models.resnet18(pretrained=True, num_classes=1000)
    model = ResNet18Hash(resnet, args.vector_dimension)
    cuda = torch.cuda.is_available()
    model.to('cuda') if cuda else model.to('cpu')

    redis_db = Redis(host='localhost', port=6379, db=0)
    if args.flush_all:
        answer = input('Do you really want to clear db [yes/no]?   ')
        if answer.startswith('yes'):
            logger.info("Cleared Redis db")
            redis_db.flushall()
        else:
            logger.info("Didn't clear storage")

    hashes = RandomBinaryProjections('default', projection_count=args.hash_length)
    engine = Engine(args.vector_dimension, lshashes=[hashes],  storage=RedisStorage(redis_db))

    dataset = PharmaPackDataset(args.src)
    logger.info(f"Image amount is {len(dataset)}")
    train_loader, test_loader = get_train_validation_data_loaders(dataset, args.batch_size, 0, args.test_size)

    if args.insert:
        insert(train_loader, engine, model)
        logger.info("Finished inserting")

    if args.search:
        df_result = search(test_loader, engine, model)

        if os.path.exists(args.dst):
            os.remove(args.dst)

        df_result.to_csv(args.dst, index=False)
        logger.info("Finished searching")

