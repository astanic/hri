import numpy as np
import SharedArray as sa
import os
import argparse

import torch

# to load the data:
# python sharedArrayRAM.py --dataset springs_32_4 --cmd load
# to delete the data:
# python sharedArrayRAM.py --dataset springs_32_4 --cmd delete

ROOT_DATA_DIR = os.path.join(os.environ["HOME"], "data")

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='springs_32_0.08_4',
                    help='Dataset directory.')
parser.add_argument('--cmd', type=str, default='load',
                    help='load/delete')

args = parser.parse_args()

dataset_path = os.path.join(ROOT_DATA_DIR, args.dataset)

def load_delete_split(split):
    if args.cmd == 'load':
        path = os.path.join(dataset_path, split + '.pt')
        dataset = torch.load(path)

    def load_delete(label):

        sa_name = args.dataset + split + '_' + label
        if args.cmd == 'load':
            data = dataset[label]

            sa_data = sa.create("shm://" + sa_name,
                                np.shape(data), dtype=data.dtype)

            print('Transferring %s to shared memory ' % (sa_name))
            sa_data[:] = data
        elif args.cmd == 'delete':
            sa.delete(sa_name)

            print('Deleted %s from the shared memory' % sa_name)
        else:
            raise NotImplementedError

    load_delete('images')
    load_delete('loc')
    load_delete('vel')
    load_delete('edges')
    load_delete('full_adj')

load_delete_split(split='train')
load_delete_split(split='valid')
load_delete_split(split='test')


if args.cmd == 'load':
    print("Datasets loaded into shared np arrays. To delete them from the memory "
          "run: python sharedArrayRAM.py --dataset {} --cmd delete".
          format(args.dataset))
