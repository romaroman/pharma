import os
import argparse
import itertools
from lopq.search import LOPQSearcherLMDB
from lopq.model import LOPQModel

parser = argparse.ArgumentParser()
parser.add_argument('--src_insert', type=str)
parser.add_argument('--src_search', type=str)
args = parser.parse_args()


models = ['resnet18', 'resnet50', 'resnet108']
vectors = [256, 512, 1024]
hashes = [64, 128, 256]
# neighbours = [1, 5, 9, 13]

for i, (model, vector, hash) in enumerate(itertools.product(*[models, vectors, hashes])):

    batch_size = {
        'resnet18': 5000,
        'resnet50': 3000,
        'resnet108': 1000
    }.get(model, 512)


    command = f"python nnmodels/hash/runner.py --base_model {model} --vector_dimension {vector}" \
              f" --hash_length {hash} --batch_size {batch_size} --db {i}"

    if args.src_insert:
        command += f" --src_insert {args.src_insert}"

    if args.src_search:
        command += f" --src_insert {args.src_search}"

    print(command)
    exit_code = os.system(command)
    if exit_code != 0:
        break
