import os
import argparse
import pickle

import flexs
from tqdm import tqdm
from itertools import product
import torch.multiprocessing as mp


def get_fitness(args):
    oracle, xs = args
    x_to_r = dict()
    for x in xs:
        r = oracle.get_fitness([x])
        x_to_r[x] = r.item()
    return x_to_r

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=1)
    
    args = parser.parse_args()
    
    problem = flexs.landscapes.rna.registry()[f'L14_RNA{args.task}']
    oracle = flexs.landscapes.RNABinding(**problem['params'])
    
    alphabet = ["A", "U", "C", "G"]
    length = 14
    
    xs = list(product(alphabet, repeat=length))
    xs = ["".join(x) for x in xs]
    print(xs[0])
    
    # batch_size = 100
    batch_size = 10000
    # start = 0
    start = 161700000
    
    x_to_r = dict()
    # mp.set_start_method('spawn')
    with tqdm(total=len(xs)-start) as pbar:
        while start < len(xs):
            number_of_worker = 30
            multi_worker = mp.Pool(number_of_worker)
            multi_worker_result = multi_worker.map(get_fitness, [(oracle, xs[start + i * batch_size: start + (i+1) * batch_size]) for i in range(number_of_worker)])
            for result in multi_worker_result:
                x_to_r.update(result)
            multi_worker.close()
            multi_worker.join()
            pbar.update(min(len(xs) - start, number_of_worker*batch_size))
            start += number_of_worker*batch_size
            
        with open(f"reward_L14_RNA{args.task}_part2.pkl", "wb") as f:
            pickle.dump(x_to_r, f)
