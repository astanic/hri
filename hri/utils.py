import os
import shutil
import subprocess
import multiprocessing
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator
from PIL import Image
import imageio

###############################################################################
###############################################################################
# logs

def add_sacred_log(key, value, _run):
    """
    Adds logs to the Sacred run info dict. Creates new dicts along the (dotted)
    key path if not available, and appends the value to a list at the final
    destination.

    :param key: (dotted) path.to.log.location.
    :param value: (scalar) value to append at log location
    :param _run: _run dictionary of the current experiment

    """
    if 'logs' not in _run.info:
        _run.info['logs'] = {}
    logs = _run.info['logs']
    split_path = key.split('.')
    current = logs
    for p in split_path[:-1]:
        if p not in current:
            current[p] = {}
        current = current[p]

    final_key = split_path[-1]
    if final_key not in current:
        current[final_key] = []
    entries = current[final_key]
    entries.append(value)


def copy_git_src_files_to_logdir(logdir):
    srcdir = os.path.join(logdir, 'src')
    os.makedirs(srcdir)
    src_git_files = subprocess.run(['git', 'ls-files'], stdout=subprocess.PIPE)
    src_git_files = src_git_files.stdout.decode('utf-8').split('\n')
    for file in src_git_files:
        if file.endswith(".py"):
            # handle nested dirs
            if '/' in file:
                sp = file.split('/')
                nested = os.path.join(*sp[:-1])
                nested = os.path.join(srcdir, nested)
                if not os.path.exists(nested):
                    os.makedirs(nested)
            else:
                nested = srcdir
            shutil.copy2(file, nested)
            print('Copied file {} to {}'.format(file, nested))


###############################################################################
###############################################################################
#  Model logs, Training loop


def setup_experiment_dir(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    log_file_stdout = os.path.join(exp_dir, 'std.out')
    log_file_stderr = os.path.join(exp_dir, 'std.err')

    print('***** Experiment directory: ', exp_dir, '*****')
    print('***** stdout log file:      ', log_file_stdout, '*****')
    print('***** stderr log file:      ', log_file_stderr, '*****')

    return log_file_stdout, log_file_stderr


def print_model_size(model):
    print("Model params::")
    total_params = 0
    for name, param in model.named_parameters():
        print("\t", name, np.prod(param.size()))
        total_params += np.prod(param.size())
    print('Total number of params:', total_params)


###############################################################################
###############################################################################
#  Graph methods
#  source: https://github.com/ethanfetaya/NRI/

def encode_onehot(labels, n_atoms):
    classes = set(list(range(n_atoms)))
    classes_dict = {c: np.identity(len(classes))[i, :]
                    for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()

###############################################################################
###############################################################################
#  Hierarchical Graph methods

# Adjacency matrices utils
def create_CNN_HRN_full_adj(only_last_level_ws=True):
    """
    Generates the full adjacency matrix a la HspringV2_4-4_21
    :return: adjacency matrix
    """
    n_atoms = 21
    nc = [4, 4]
    nl = 2
    nn = [1, 4, 16]

    full_adj = np.zeros(shape=(n_atoms, n_atoms))
    row_idx = 0  # start filling adjacency mat from root node
    col_idx = 1  # skip the root node and start from 2nd node
    for l in range(nl):
        for n in range(nn[l]):
            full_adj[row_idx, col_idx:col_idx + nc[l]] = 1
            # Fill symmetric the lower triangular (undirected graph)
            full_adj[col_idx:col_idx + nc[l], row_idx] = 1

            if l == nl - 1 or not only_last_level_ws:
                # create full cluster adjacency matrix
                full_adj[col_idx:col_idx + nc[l], col_idx:col_idx + nc[l]] = \
                    np.ones((nc[l], nc[l])) - np.eye(nc[l])

            # Increase counters after filling connections for a parent node
            col_idx += nc[l]
            row_idx += 1

    return full_adj


def create_hierarchy_nodes_list(dataset_name, n_children, n_atoms):
    # Note, this is a duplicate code from data/synthetic_sim.py classes,
    # due to difficulty in loading multidimensional np.array of
    # varying size from SharedArray memory
    if 'Hsprings' in dataset_name:
        nc = list(map(int, n_children.split('-')))
        nl = len(nc)
        # number of nodes in each level of the graph that have children
        i = 0
        hns = [[i]]  # root node
        i += 1
        for l in range(1, nl + 1):
            hns.append(list(range(i, i+np.prod(nc[:l]))))
            i += np.prod(nc[:l])
    else:
        # not hierarchical dataset - all nodes inside 1 level
        hns = [list(range(n_atoms))]
    return hns


def create_HRN_MP_masks(full_adj, hns):
    mp_zeros = full_adj * 0

    # L2A - leafs to ancestors
    mp_l2a_adjs = []
    for l in range(len(hns)-1, 0, -1):
        x = mp_zeros.copy()
        x[np.ix_(hns[l], hns[l-1])] = 1
        mp_l2a_adjs.append(x * full_adj)

    # WS - within siblings
    mp_ws_adj = mp_zeros.copy()
    for l in range(len(hns)):
        x = mp_zeros.copy()
        x[np.ix_(hns[l], hns[l])] = 1
        mp_ws_adj += x * full_adj  # this phase has only 1 stage

    # A2D - ancestors to descendants
    mp_a2d_adjs = []
    for l in range(1, len(hns)):
        x = mp_zeros.copy()
        x[np.ix_(hns[l-1], hns[l])] = 1
        mp_a2d_adjs.append(x * full_adj)
    return mp_l2a_adjs, mp_ws_adj, mp_a2d_adjs


def create_last_level_MP_mask(full_adj, hns):
        mp_mask = full_adj * 0
        mp_mask[np.ix_(hns[-1], hns[-1])] = 1
        return mp_mask


###############################################################################
###############################################################################
#  Losses. Initial implementation source: https://github.com/ethanfetaya/NRI/


def nll_gaussian(preds, target, variance, avg_up_to_dim=-1,
                 last_level_nodes=None, add_const=False):
    nll = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        nll += const

    nll_partial = None
    if last_level_nodes is not None:
        if last_level_nodes[-1] < nll.size(1):
            # if we haven't clipped the trajectories already
            nll_partial = nll[:, last_level_nodes]
        else:
            nll_partial = nll
        nll_partial = nll_partial.sum() / np.prod(nll_partial.shape[:avg_up_to_dim])

    nll = nll.sum() / np.prod(nll.shape[:avg_up_to_dim])
    return nll, nll_partial


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(preds, num_atoms, n_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(n_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def complete_sparse_adj(meta_data, sparse_adj, inf_adj):
    full_adj = meta_data['model_full_adj']

    sp_idcs = np.where(sparse_adj)
    fu_idcs = np.where(full_adj)

    dummy_edges = torch.tensor([1, 0], dtype=inf_adj.dtype,
                               device=inf_adj.device)
    dummy_edges = dummy_edges.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    dummy_edges = dummy_edges.repeat(inf_adj.size(0), inf_adj.size(1), 1, 1)

    res = []
    j = 0
    for i in range(len(fu_idcs[0])):
        if fu_idcs[0][i] == sp_idcs[0][j] and fu_idcs[1][i] == sp_idcs[1][j]:
            # use sparse tensor
            res.append(inf_adj[:, :, j:j+1])
            j += 1
        else:
            res.append(dummy_edges)

    res = torch.cat(res, dim=2)
    return res


def edge_accuracy(preds, target, sparse_mask):
    _, preds = preds.max(-1)
    # align with dynamic graph inference
    target = target.unsqueeze(1).expand(list(preds.shape))
    correct = preds.float().data.eq(target.float().data.view_as(preds))

    tshp = target.shape
    edge_acc = np.float(correct.cpu().sum()) / np.prod(tshp)

    # Compute sparse mask acc (in case of hierarchical dataset)
    edge_acc_sparse = edge_acc
    if sparse_mask is not None:
        # sparse_mask = sparse_mask.repeat(1, preds.size(1), 1)
        edge_acc_sparse = (correct * sparse_mask).cpu().sum() / \
                          (np.prod(tshp[:-1]) * sparse_mask.cpu().sum())

    return edge_acc, edge_acc_sparse


def graph_accuracy(target, preds):
    # align with dynamic graph inference
    target = target.unsqueeze(1).expand(list(preds.shape))
    correct = (preds == target).cpu().numpy()
    correct = np.mean(np.prod(correct, -1))
    return correct


###############################################################################
###############################################################################
# NRI gumbel softmax


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/
    327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/
    3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/
    327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/
    3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/
    327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor

    based on
    https://github.com/ericjang/gumbel-softmax/blob/
    3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.data + y_soft
    else:
        y = y_soft
    return y


def gumbel_sigmoid(logits, tau=1, hard=False, eps=1e-10):
    uniform = logits.new_empty([2]+list(logits.shape)).uniform_(0,1)
    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    res = torch.sigmoid((logits + noise) / tau)
    if hard:
        res = ((res > 0.5).type_as(res) - res).detach() + res
    return res


###############################################################################
###############################################################################
# plot methods

def plot_batch_graphs(edges_gt, edges_inferred, plot_dir, full_adj):
    """
    Draws ground truth and inferred graphs.
    First it creates the full adjacency matrices, because the inputs
    are flattened elements of the off-diagonal adjacency matrices.
    :param edges_gt: (batch_size, num_atoms * (num_atoms - 1)) int32
    :param edges_inferred: (batch_size, num_atoms * (num_atoms - 1)) int32
    :param plot_dir: string, where to save all plots
    :return:
    """
    def plot_graphs(edges_gt, edges_inferred, i, filename):
        # Increase def figsize from (6.4, 4.8)
        plt.figure(figsize=(60, 9.6))

        # when dynamic graph inference, plot only the last step
        a_gt = offdiag_adj2full_adj(edges_gt[i])

        plt.subplot(1, 11, 1)
        plt.gca().set_title('gt')
        G = nx.from_numpy_matrix(np.array(a_gt))
        if edges_gt[0].shape[-1] == edges_inferred[i, 0].shape[-1]:
            pos = nx.spring_layout(G)
        else:
            # visual case
            G = nx.from_numpy_matrix(np.array(full_adj))
            pos = nx.spring_layout(G)

        nx.draw(G, pos=None, with_labels=True)

        for j in range(min(10, edges_inferred.shape[1])):
            a_inf = offdiag_adj2full_adj(edges_inferred[i, j])

            plt.subplot(1, 11, j+2)
            plt.gca().set_title('inferred')
            G = nx.from_numpy_matrix(np.array(a_inf))
            if pos is None:
                pos = nx.spring_layout(G)
            nx.draw(G, pos=pos, with_labels=True)

        plt.savefig(filename)
        plt.clf()
        plt.close()

    def offdiag_adj2full_adj(edges_v):
        from math import sqrt
        def perfect_sqrt(n):
            if n % n ** 0.5 == 0:
                return int(sqrt(n))
            else:
                return perfect_sqrt(n+1)

        n_atoms = perfect_sqrt(edges_v.shape[-1])

        edges_adj = edges_v
        assert len(edges_adj.shape) == 1

        if edges_adj.shape[0] < n_atoms ** 2:
            # Add self-connections to the adjacency matrix
            diag_idx = np.ravel_multi_index(np.where(np.eye(n_atoms)),
                                            [n_atoms, n_atoms])
            diag_zeros = np.zeros(len(diag_idx), np.int32)

            for i in range(len(diag_idx) - 1):
                edges_adj = np.insert(edges_adj, diag_idx[i], diag_zeros[i])
            edges_adj = np.append(edges_adj, [0])

        return np.reshape(edges_adj, [n_atoms, n_atoms])

    n_seq = np.minimum(len(edges_gt), 10)
    for i in range(n_seq):

        plot_graphs(edges_gt, edges_inferred, i,
                    os.path.join(plot_dir, 'graph_gt_inf_{:03}.png'.format(i)))


def overlay_imgs(imgs, alp, eps, overlay_n):
    ret = np.zeros_like(imgs[:,:1])
    n_seq = np.minimum(imgs.shape[1], overlay_n)
    for i in range(n_seq):
        ret = alp * ret * (imgs[:,i:i+1]<eps) + imgs[:,i:i+1]
    return ret

def concat_pred_imgs_and_save(timgs, pimgs, pobjsimgs, save_dir,
                              rollout_seq_len=10,
                              draw_objs=True, overlay_rollouts=False,
                              overlay_alpha=1, overlay_eps=1e-1,
                              overlay_n=10):
    timgs = timgs.transpose(-1, -3)
    timgs = timgs.cpu().numpy()

    pimgs = pimgs.transpose(-1, -3)
    pimgs = pimgs.clamp(0, 1)
    pimgs = pimgs.cpu().detach().numpy()
    # white space borders
    pw = ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0))

    if draw_objs:
        # calculate image difference and concat with target
        dimgs = np.absolute(pimgs - timgs)

        # overlay the target with the predicted
        oimgs = pimgs + timgs
        oimgs = oimgs.clip(0, 1)
        # ignore the sum
        oimgs = np.zeros_like(oimgs)

        timgs = np.pad(timgs, pad_width=pw, constant_values=1)
        pimgs = np.pad(pimgs, pad_width=pw, constant_values=1)
        dimgs = np.pad(dimgs, pad_width=pw, constant_values=1)
        oimgs = np.pad(oimgs, pad_width=pw, constant_values=1)

        tdopimgs = np.concatenate((timgs, pimgs, dimgs, oimgs), axis=-3)

        if pobjsimgs is not None:
            pobjsimgs = pobjsimgs.transpose(-1, -3)
            pobjsimgs = pobjsimgs.clamp(0, 1)
            pobjsimgs = pobjsimgs.cpu().detach()
        else:
            pobjsimgs = tdopimgs

        imgs = np.concatenate((tdopimgs, pobjsimgs), axis=-2)
        write_imgs(imgs, save_dir, write_seq_len=rollout_seq_len)
    else:
        timgs = np.pad(timgs, pad_width=pw, constant_values=1)
        pimgs = np.pad(pimgs, pad_width=pw, constant_values=1)
        if overlay_rollouts:
            timgs = overlay_imgs(timgs, overlay_alpha, overlay_eps, overlay_n)
            pimgs = overlay_imgs(pimgs, overlay_alpha, overlay_eps, overlay_n)
        imgs = np.concatenate((timgs, pimgs), axis=-3)
        write_imgs(imgs, save_dir, one_img_for_seq=True,
                   write_seq_len=rollout_seq_len)


def write_imgs(imgs, save_dir, one_img_for_seq=False, write_seq_len=10):
    # concatenate along image col dimension
    imgs = np.copy(imgs)
    if imgs.shape[-1] == 1:
        # Grayscale case
        imgs = np.repeat(imgs, 3, axis=-1)
    imgs *= 255
    imgs = imgs.astype(np.uint8)
    write_img_batch_to_dir(imgs, save_dir, one_img_for_seq=one_img_for_seq,
                           write_seq_len=write_seq_len)


def write_img_batch_to_dir(imgs, save_dir, max_n_seq=3, one_img_for_seq=False,
                           write_seq_len=10):
    # RGB2BGR for plot
    imgs = np.flip(imgs, -1)
    name_pattern = os.path.join(save_dir, "batch_%d_f_%d.png")
    name_pattern_seq = os.path.join(save_dir, "batch_%d.png")
    video_name_pattern = os.path.join(save_dir, "video_%d.gif")
    n_seq = np.minimum(imgs.shape[0], max_n_seq)
    for i in range(n_seq):
        # write images
        if not one_img_for_seq:
            for j in range(imgs.shape[1]):
                im = Image.fromarray(imgs[i,j])
                im.save(name_pattern % (i, j))
        else:
            im = imgs[i]
            im = np.transpose(im, (1, 0, 2, 3))
            im = im[:, -write_seq_len:, :, :]
            shp = im.shape
            im = im.reshape((shp[0], shp[1] * shp[2], shp[3]))
            im = Image.fromarray(im)
            im.save(name_pattern_seq % i)

        # write video
        video_filename = video_name_pattern % i
        imageio.mimsave(video_filename, imgs[i])


###############################################################################
###############################################################################
# render methods


def render_state(locations, figsize=(32, 32), ball_radius=0.08,
                 box_size=1., n_children=None, fixed_ball_radius=False,
                 render_only_last_level=False, edges=None, fixed_sides=0, curtain=False):
    """Renders the state of the environment from its locations (T, 2, K)."""

    images = []

    BG_COLOR = 'black'
    BALL_COLORS = ['black', 'blue', '#00ff00', 'red',
                   'white', 'cyan', 'magenta', 'yellow']
    BALL_COLORS.remove(BG_COLOR)

    n_total = locations.shape[-1]
    # handle the hierarchy
    nl = 0
    nn = [locations.shape[-1]]
    if n_children is not None:
        if '-' in n_children:
            nc = list(map(int, n_children.split('-')))
            nl = len(nc)
            # number of nodes in each level of the graph that have children
            nn = [1]  # root node
            for l in range(1, nl + 1):
                nn.append(nn[-1] * nc[l - 1])

    sides = []
    for l in range(nl + 1):
        for n in range(nn[l]):
            sample = np.random.sample()
            if fixed_sides < 9:  # 9 means sample a random shape
                sides.append(fixed_sides)
            elif sample < 1/3:
                sides.append(0)
            elif sample < 2/3:
                sides.append(3)
            else:
                sides.append(4)

    ch, cw = 12, 12
    xc = np.random.randint(5, 32-cw-5)
    yc = np.random.randint(5, 32-ch-5)

    for i in range(locations.shape[0]):
        loc = locations[i]
        fig = Figure(figsize=(figsize[0] / 100, figsize[1] / 100))
        fig.patch.set_facecolor(BG_COLOR)

        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())

        # for p in range(locations.shape[-1]):
        p = 0
        br = ball_radius
        for l in range(nl + 1):
            for n in range(nn[l]):

                x_pos = (loc[0, p] + box_size) / (2 * box_size)
                y_pos = (loc[1, p] + box_size) / (2 * box_size)

                if nl == 0:
                    cc = BALL_COLORS[p % len(BALL_COLORS)]
                else:
                    cc = BALL_COLORS[p % len(BALL_COLORS)]
                    if not render_only_last_level and nl > 0:
                        if l == 0:
                            cc = 'gold'
                        elif l == 1:
                            cc = 'silver'

                def get_polygon_coords(xc, yc, edge_len=0.10, sides=3):
                    L = edge_len
                    if sides == 3:
                        L *= 2
                    N = sides
                    R = L / (2 * np.sin(np.pi / N))

                    xy = np.ndarray((N, 2), dtype=float)
                    for i in range(N):
                        xy[i, 0] = xc + R * np.cos(np.pi / N * (1 + 2 * i))
                        xy[i, 1] = yc + R * np.sin(np.pi / N * (1 + 2 * i))
                    return xy

                if not (l < nl and render_only_last_level):
                    if sides[p] == 0:
                        particle = plt.Circle((x_pos, y_pos), br, color=cc, clip_on=False)
                    else:
                        xy = get_polygon_coords(x_pos, y_pos, edge_len=br*2, sides=sides[p])
                        particle = plt.Polygon(xy, color=cc, fill=True, clip_on=False)
                    ax.add_artist(particle)

                if edges is not None and not render_only_last_level and p < n_total-1:
                    edg = edges[i]
                    for pp in range(p, n_total-1):
                        if edg[p * (n_total - 1) + pp]:
                            x1 = (loc[0, p] + box_size) / (2 * box_size)
                            x2 = (loc[0, pp+1] + box_size) / (2 * box_size)
                            y1 = (loc[1, p] + box_size) / (2 * box_size)
                            y2 = (loc[1, pp+1] + box_size) / (2 * box_size)
                            llll = mlines.Line2D([x1, x2], [y1, y2],
                                                 color='white',
                                                 linewidth=1.4,
                                                 linestyle=':'
                                                 # '-', '--', '-.', ':', ''
                                                 )
                            ax.add_artist(llll)

                p += 1

            if not fixed_ball_radius:
                br /= 2

        ax.axis('off')

        # Draw image
        canvas.draw()

        # Convert to numpy array
        flat_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = flat_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if curtain:
            im = image.copy()
            im[yc:yc+ch, xc:xc+cw, :] = 0
            image = im

        images.append(image)
        plt.close(fig)

    return np.array(images)


def render_batch(loc_batch, n_children, edges=None,
                 render_only_last_level=False):
    # partial_render_state = partial(render_state,
    #                                ball_radius=0.04)
    # pool = multiprocessing.Pool(4)
    #
    # images = np.array(render_state(loc_batch,
    #                                ball_radius=0.04,
    #                                edges=edges))
    # pool.close()

    images = []
    for i in range(loc_batch.shape[0]):
        edg=None
        if edges is not None:
            edg = edges[i]
        images.append(render_state(loc_batch[i],
                                   ball_radius=0.04,
                                   n_children=n_children,
                                   figsize=(128, 128),
                                   edges=edg,
                                   fixed_ball_radius=True,
                                   render_only_last_level=render_only_last_level))

    images = np.array(images)
    return images


def render_traj(pred_loc_vel, n_children, edges=None, render_only_last_level=False):
    """
    Renders images from predicted trajectories and computes MSE loss to GT.
    :param pred_loc_vel: np.array (bs, n_balls, seq_len - 1, 4)
                         contains both pred loc and vel (last dimension)
    :param target_images: np array
    :return:
    """
    # Take loc (x,y) are the first 2 elements and transpose
    pred_loc = pred_loc_vel[:, :, :, :2]
    pred_loc = np.transpose(pred_loc, (0, 2, 3, 1))
    pred_images = render_batch(pred_loc, n_children, edges,
                               render_only_last_level=render_only_last_level)
    pred_images = pred_images.astype(np.float32)
    pred_images /= 255

    # drop the first target image (no prediction)
    return pred_images


def render_traj_and_save(traj_target, traj_pred, save_dir, overlay_dir,
                         n_children, rollout_seq_len=10,
                         last_level_nodes=None, tedges=None, pedges=None,
                         render_only_last_level_nodes=False,
                         overlay_alpha=1, overlay_eps=1e-1, overlay_n=10):

    if last_level_nodes is not None and render_only_last_level_nodes:
        traj_target = traj_target[:, last_level_nodes]
        traj_pred = traj_pred[:, last_level_nodes]

    tedges = tedges.unsqueeze(1).expand(list(pedges.shape))
    timgs = render_traj(traj_target.cpu().detach(), n_children,
                        render_only_last_level=True)
    timgs_lat = render_traj(traj_target.cpu().detach(), n_children, tedges,
                            render_only_last_level=False)
    pimgs = render_traj(traj_pred.cpu().detach(), n_children, pedges,
                        render_only_last_level=False)

    pw = ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0))
    timgs = np.pad(timgs, pad_width=pw, constant_values=1)
    timgs_lat = np.pad(timgs_lat, pad_width=pw, constant_values=1)
    pimgs = np.pad(pimgs, pad_width=pw, constant_values=1)
    imgs = np.concatenate((timgs, timgs_lat, pimgs), axis=-3)
    write_imgs(imgs, save_dir, one_img_for_seq=True, write_seq_len=rollout_seq_len)

    # render the overlay imgs
    timgs = render_traj(traj_target.cpu().detach(), n_children, render_only_last_level=False)
    pimgs = render_traj(traj_pred.cpu().detach(), n_children, render_only_last_level=False)
    timgs = overlay_imgs(timgs, overlay_alpha, overlay_eps, overlay_n)
    pimgs = overlay_imgs(pimgs, overlay_alpha, overlay_eps, overlay_n)
    imgs = np.concatenate((timgs, pimgs), axis=-3)
    write_imgs(imgs, overlay_dir, one_img_for_seq=True, write_seq_len=rollout_seq_len)


def AOverB(color_a, a_a, color_b, a_b):
    """
    Returns a composite image by computing A OVER B
    :param color_a: Tensor of pixel values for A
    :param a_a: Tensor of alpha values for A
    :param color_b: Tensor of pixel values for B
    :param a_b: Tensor of alpha values for B
    :return: Composite image, given by new pixel values and alpha
    """
    alpha_o = a_a + a_b * (1. - a_a)
    color_o = (color_a * a_a + color_b * a_b * (1. - a_a)) / alpha_o
    return color_o, alpha_o


###############################################################################
###############################################################################
###  pretrained model methods

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def select_params_to_update(model):
    total_params = 0
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name, np.prod(param.size()))
            total_params += np.prod(param.size())
    print('Total number of params:', total_params)
    return params_to_update
