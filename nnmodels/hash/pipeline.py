import os
import argparse
import itertools


parser = argparse.ArgumentParser()
parser.add_argument('src_insert', type=str)
parser.add_argument('src_search', type=str)
args = parser.parse_args()


neighbours = [1, 5, 9, 13]
models = ['resnet18', 'resnet50', 'resnet108']
vectors = [256, 512, 1024]
hashes = [64, 128, 256]
experiments = itertools.product(*[neighbours, models, vectors, hashes])

for neighbour, model, vector, hash in experiments:

    batch_size = {
        'resnet18': 2500,
        'resnet50': 1500,
        'resnet108': 600
    }.get(model, 512)

    db = f"{model}_v{vector}_h{hash}"

    command = f"python runner.py {args.src_insert} {args.src_search} --base_model {model} --vector_dimension {vector}" \
              f" --hash_length {hash} --batch_size {batch_size} --neighbours {neighbour} --db {db} --flush_all False"

    print(command)
    os.system(command)
