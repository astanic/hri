import os
import numpy as np
import SharedArray

import torch
from torch.utils.data import Dataset, DataLoader

from utils import create_hierarchy_nodes_list, create_HRN_MP_masks, create_last_level_MP_mask

from sacred import Ingredient

ROOT_DATA_DIR = "data/"

d = Ingredient("sdata")


@d.config
def config():
    batch_size = 32  # Number of samples per batch.

    name = 'springs_4_4_32_0.08'  # Dataset name.

    n_children = name.split('_')[1]
    n_atoms = int(name.split('_')[2])
    structure_config = name.split('_')[3]
    fig_size = int(name.split('_')[4])
    ball_radius = float(name.split('_')[5])

    use_shared_memory = True
    num_workers = 4
    pin_memory = True

    debug = True
    bw = False

    n_edge_types = 2  # The number of edge types to infer.
    use_self_edges = False

    seq_len = 50


class HRIDataset(Dataset):
    """ HRI project video dataset. """

    @d.capture
    def __init__(self, name, split, meta_data, use_shared_memory,
                 use_self_edges, n_children, n_atoms, debug, bw, seq_len):

        self.meta_data = meta_data
        self.use_self_edges = use_self_edges
        self.name = name
        self.n_children = n_children
        self.n_atoms = n_atoms

        if use_shared_memory:
            # Data is already in SharedArray in the RAM
            sa_name = name + split
            self.images = SharedArray.attach("shm://" + sa_name + '_images')
            loc = SharedArray.attach("shm://" + sa_name + '_loc')
            vel = SharedArray.attach("shm://" + sa_name + '_vel')
            self.edges = SharedArray.attach("shm://" + sa_name + '_edges')
            self.full_adj = SharedArray.attach("shm://" + sa_name + '_full_adj').copy()
        else:
            dataset_path = os.path.join(ROOT_DATA_DIR, name)
            # Load the data from the disk to the RAM
            dataset = torch.load(os.path.join(dataset_path, split + '.pt'))
            self.images = dataset['images']
            loc = dataset['loc']
            vel = dataset['vel']
            self.edges = dataset['edges']
            # full_adj is the adjacency matrix of the max possible graph
            # graphs of the samples in the dataset are generated by starting
            # from this full_adj matrix, and then dropping certain edges
            self.full_adj = dataset["full_adj"].copy()

        self.n_seq = self.images.shape[0]
        if debug:
            self.n_seq = self.n_seq // 10
            self.images = self.images[:self.n_seq]
            loc = loc[:self.n_seq]
            vel = vel[:self.n_seq]
            self.edges = self.edges[:self.n_seq]

        if seq_len < 50:
            self.images = self.images[:, :seq_len]
            loc = loc[:, :seq_len]
            vel = vel[:, :seq_len]

        if bw:
            self.images = np.amax(self.images, axis=-1, keepdims=True)

        self.hierarchy_nodes_list = create_hierarchy_nodes_list(
            self.name, self.n_children, self.n_atoms)

        self.preprocess_data(split, loc, vel)

    @d.capture
    def preprocess_data(self, split, loc, vel, n_edge_types, n_atoms):
        """Load data from numpy arrays."""
        # [num_samples, num_timesteps, num_dims, n_atoms]

        if split == "train":
            self.meta_data = {
                "loc_max": loc.max(), "loc_min": loc.min(),
                "vel_max": vel.max(), "vel_min": vel.min()
            }

        # Normalize to [-1, 1]
        loc_max, loc_min = self.meta_data["loc_max"], self.meta_data["loc_min"]
        vel_max, vel_min = self.meta_data["vel_max"], self.meta_data["vel_min"]

        loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1
        vel = (vel - vel_min) * 2 / (vel_max - vel_min) - 1

        # Reshape to: [num_sims, n_atoms, num_timesteps, num_dims]
        loc = np.transpose(loc, [0, 3, 1, 2])
        vel = np.transpose(vel, [0, 3, 1, 2])
        self.traj = np.concatenate([loc, vel], axis=3)
        self.edges = np.reshape(self.edges, [-1, n_atoms ** 2])
        self.edges = np.array((self.edges + 1) / 2, dtype=np.int64)

        if not self.use_self_edges:
            # Exclude self edges
            off_diag_idx = np.ravel_multi_index(
                np.where(np.ones((n_atoms, n_atoms)) - np.eye(n_atoms)),
                [n_atoms, n_atoms])
            self.edges = self.edges[:, off_diag_idx]

        # Note this overwrites meta_data on purpose
        self.edges_sparse_mask = np.expand_dims(self.full_adj, axis=0)
        self.edges_sparse_mask = np.reshape(self.edges_sparse_mask,
                                            [-1, n_atoms ** 2])
        if not self.use_self_edges:
            self.edges_sparse_mask = self.edges_sparse_mask[:, off_diag_idx]
        self.edges_sparse_mask = torch.Tensor(self.edges_sparse_mask).cuda()
        # prepare to align with dynamic graph inference
        self.edges_sparse_mask = self.edges_sparse_mask.unsqueeze(0)

        self.meta_data["n_atoms"] = n_atoms
        self.meta_data["n_dims"] = self.traj.shape[-1]
        self.meta_data["seq_len"] = self.images.shape[1]
        self.meta_data["img_size"] = self.images.shape[2:]
        self.meta_data["use_self_edges"] = self.use_self_edges
        self.meta_data["full_adj"] = self.full_adj.copy()
        self.meta_data["edges_sparse_mask"] = self.edges_sparse_mask
        self.meta_data["n_edge_types"] = n_edge_types
        self.meta_data["hierarchy_nodes_list"] = self.hierarchy_nodes_list
        self.meta_data["n_children"] = self.n_children

        # Create MP operations' masks L2A, WS, A2D
        self.meta_data["mp_l2a_adjs"], self.meta_data["mp_ws_adj"], \
        self.meta_data["mp_a2d_adjs"] = create_HRN_MP_masks(
            self.full_adj, self.hierarchy_nodes_list)

        # Create "last level" MP mask - used when MP only at last level
        self.meta_data["mp_last_level_mask"] = create_last_level_MP_mask(
            self.full_adj, self.hierarchy_nodes_list)

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = self.images[idx]
        traj = self.traj[idx]
        edges = self.edges[idx]

        sample = {'imgs': imgs,
                  'traj': traj,
                  'edges': edges}

        return sample


@d.capture
def load_data(batch_size, num_workers, pin_memory):

    train_data = HRIDataset(split='train', meta_data=None)
    valid_data = HRIDataset(split='valid', meta_data=train_data.meta_data)
    test_data = HRIDataset(split='test', meta_data=train_data.meta_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, train_data.meta_data