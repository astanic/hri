# Built off of NRI's code: https://github.com/ethanfetaya/NRI
import os

from synthetic_sim import ChargedParticlesSim, SpringSim
from synthetic_sim import HspringsV1TreeRandomDrop, HspringsV2TreeRandomClusters, HspringsV3TreeRandomAllLevels
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hri.utils import render_state
import time
import numpy as np
import argparse
import multiprocessing

from functools import partial

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data',
                    help='Where to save the data.')
parser.add_argument('--simulation', type=str, default='springs',
                    help='What simulation to generate.')
parser.add_argument('--structure-config', type=str, default='00',
                    help='Graph structure config type - init pos, forces etc.')
parser.add_argument('--render', type=bool, default=True,
                    help='Whether to render the trajectories.')
parser.add_argument('--figsize', type=int, default=32,
                    help='Height and width of the rendered image.')
parser.add_argument('--normalize-box-size', type=bool, default=True,
                    help='Normalize trajectories to [-1,1] interval.')
parser.add_argument('--box-size', type=int, default=2,
                    help='Distance of the env walls from the center.')
parser.add_argument('--ball-radius', type=float, default=0.04,
                    help='Render ball radius.')
parser.add_argument('--different-ball-radius', action='store_true', default=False,
                    help='Different ball radius across hierarchy levels.')
parser.add_argument('--render-all-levels', action='store_true', default=False,
                    help='Render balls from all hierarchy levels.')
parser.add_argument('--num-train', type=int, default=50000,
                    help='Number of training trajectories to simulate.')
parser.add_argument('--num-valid', type=int, default=10000,
                    help='Number of validation trajectories to simulate.')
parser.add_argument('--num-test', type=int, default=10000,
                    help='Number of test trajectories to simulate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=5000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--interaction-strength', type=float, default=.1,
                    help='Interaction strength between the balls.')
parser.add_argument('--n-children', type=str, default='4-4',
                    help='Number of children each node has.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--pool-size', type=int, default=None, help='Pool size.')

args = parser.parse_args()

if args.simulation == 'springs':
    sim = SpringSim(
        noise_var=0.0, n_balls=args.n_balls,
        box_size=args.box_size,
        interaction_strength=args.interaction_strength,
        structure_config=args.structure_config)
    dataset_name = 'springs'
elif args.simulation == 'HspringsV1':
    sim = HspringsV1TreeRandomDrop(
        n_children=args.n_children,
        box_size=args.box_size,
        interaction_strength=args.interaction_strength,
        structure_config=args.structure_config)
    dataset_name = 'HspringsV1'
elif args.simulation == 'HspringsV2':
    sim = HspringsV2TreeRandomClusters(
        n_children=args.n_children,
        box_size=args.box_size,
        interaction_strength=args.interaction_strength,
        structure_config=args.structure_config)
    dataset_name = 'HspringsV2'
elif args.simulation == 'HspringsV3':
    sim = HspringsV3TreeRandomAllLevels(
        n_children=args.n_children,
        box_size=args.box_size,
        interaction_strength=args.interaction_strength,
        structure_config=args.structure_config)
    dataset_name = 'HspringsV3'
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(
        noise_var=0.0, n_balls=args.n_balls,
        box_size=args.box_size,
        interaction_strength=args.interaction_strength)
    dataset_name = 'charged'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

# Hierarchical dataset specifics
if 'Hsprings' in args.simulation:
    if args.different_ball_radius:
        dataset_name += "DBR"
if 'Hsprings' in args.simulation:
    sep = "_%d"
    for nc in sim.nc:
        dataset_name += sep % nc  # children
        sep = "-%d"
else:
    sep = "_%d"
    dataset_name += sep % sim.n_balls

# Use sim.n_balls b/c hierarchical dataset calculates it on the fly
dataset_name += "_%d" % sim.n_balls

dataset_name += "_%s" % args.structure_config
dataset_name += "_%d" % args.figsize
dataset_name += "_%.2f" % args.ball_radius

print("Dataset name:", dataset_name)


def sample_trajectory_mp(index, length, sample_freq):
    """Wraps sample_trajectory for multi-processing."""
    np.random.seed(args.seed + index)
    return sim.sample_trajectory(T=length, sample_freq=sample_freq)


def generate_trajectories(num_sims, length, sample_freq):
    """Generates a dataset of num_sims simulations in parallel."""

    # Init argument of sample_trajectory function and pool.
    partial_sample_trajectory_mp = partial(
        sample_trajectory_mp, length=length, sample_freq=sample_freq)
    pool = multiprocessing.Pool(args.pool_size)

    # Perform simulations.
    t = time.time()
    loc_all, vel_all, edges_all = zip(*pool.map(
        partial_sample_trajectory_mp, range(num_sims)))

    print("Simulation time: {}".format(time.time() - t))

    return np.stack(loc_all), np.stack(vel_all), np.stack(edges_all)


def write_to_dir(data_dir, data_dict, prefix):
    """Writes data dictionary with the tensors to output_dir. """

    # Ensure data_dir exists, else make it.
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = prefix + '.pt'
    data_path = os.path.join(data_dir, filename)
    print('Saving data to', data_path)

    t = time.time()
    torch.save(data_dict, data_path, pickle_protocol=4)
    print("Write time: {}".format(time.time() - t))


# Init save dir
if not os.path.exists(args.path):
    os.makedirs(args.path)

# Set-up dict with dataset generation specifics
data_dict = dict()
if args.num_train > 0:
    data_dict["train"] = {"size": args.num_train, "length": args.length}
if args.num_valid > 0:
    data_dict["valid"] = {"size": args.num_valid, "length": args.length}
if args.num_test > 0:
    data_dict["test"] = {"size": args.num_test, "length": args.length_test}

# Generate datasets
for split, config in data_dict.items():
    # Simulate Trajectories
    print("Simulating {} {} trajectories".format(config["size"], split))
    loc, vel, edges = generate_trajectories(
        config["size"], config["length"], args.sample_freq)

    loc = loc.astype(np.float32)
    vel = vel.astype(np.float32)
    edges = edges.astype(np.float32)

    if args.normalize_box_size:
        loc_max = np.abs(loc).max()
        loc /= loc_max
        vel /= loc_max

    full_adj, _, _ = sim.generate_full_adj()
    full_adj.astype(np.float32)
    n_children = None
    if 'Hsprings' in args.simulation:
        n_children = args.n_children

    data = {
        'loc': loc,
        'vel': vel,
        'edges': edges,
    }

    if args.render:
        print("Starting rendering")

        # Init argument of sample_trajectory function and pool.
        partial_render_state = partial(render_state,
                                       n_children=n_children,
                                       figsize=(args.figsize, args.figsize),
                                       ball_radius=args.ball_radius,
                                       fixed_ball_radius=not args.different_ball_radius,
                                       render_only_last_level=not args.render_all_levels)
        pool = multiprocessing.Pool(args.pool_size)

        # Perform simulations.
        t = time.time()
        data['images'] = np.array(pool.map(partial_render_state, loc))
        print("Rendering time: {}".format(time.time() - t))

    data['full_adj'] = full_adj

    # Save
    data_dir = os.path.join(args.path, dataset_name)
    write_to_dir(data_dir, data, prefix=split)


