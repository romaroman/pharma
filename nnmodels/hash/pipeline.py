import os
import argparse
import itertools


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
        'resnet18': 2500,
        'resnet50': 1500,
        'resnet108': 600
    }.get(model, 512)


    command = f"python nnmodels/hash/runner.py --src_insert {args.src_insert} --base_model {model} --vector_dimension {vector}" \
              f" --hash_length {hash} --batch_size {batch_size} --db {i}"

    print(command)
    # exit_code = os.system(command)
    # if exit_code != 0:
        # break
#