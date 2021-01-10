import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sacred import Ingredient

from utils import encode_onehot, gumbel_softmax, complete_sparse_adj
from utils import nll_gaussian, kl_categorical_uniform, edge_accuracy, graph_accuracy
from utils import create_CNN_HRN_full_adj, create_hierarchy_nodes_list
from utils import create_HRN_MP_masks, create_last_level_MP_mask

from modules_encoder import I2TEncConv, I2TEncVGG, I2TEncDCGAN
from modules_encoder import T2HEncConv, T2HEncVGG, T2HEncDCGAN

from modules_obj_vae import ObjectSlotVAE
from modules_relational import NRIMLPEncoder
from modules_pred import NRIMLPDecoder, LSTMBaseline, LSTMDecoder
from modules_decoder import SlotConvDecoder, ParConvDecoder
from modules_decoder import SlotVGGDecoder, ParVGGDecoder
from modules_decoder import SlotDCGANDecoder, ParDCGANDecoder
from modules_decoder import ParSBDecoder, SlotSBDecoder


m = Ingredient("smodel")


@m.config
def config():
    increase_pred_steps_every = 0
    train_pred_steps = 10  # Num steps to predict before using GT labels
    valid_pred_steps = 5
    test_pred_steps = 10

    n_slots = None
    n_slot_dims = 16
    decode_masked_slots = False

    # Latent space config
    temp = 0.5  # Temperature for Gumbel softmax.

    # Output config
    output_variance = 5e-5  # Output variance.

    # Architecture defining config
    l_alpha_profile = 3  # controls behavior of l_alpha_trj and l_alpha_img
                         # 1: (0,1)
                         # 2: (1,0)
                         # 3: (1,1)
                         # see also update_loss_coefficients function

    l_alpha_trj = 1.0  # Loss coeff. for trajectory MSE
    l_alpha_edg = 1.0  # Loss coeff. for edge KL
    l_alpha_img = 1.0  # Loss coeff. for image MSE
    l_alpha_obj = 1.0  # Loss coeff. for object KL

    l_beta_trj = 1.0  # Loss coeff. multiplier for trajectory MSE

    l_delta_sf = 0.0  # Loss coeff. for slow features enforcing

    par_to_opt='all'  # 'all' or 'dyn'
    opt_dynamics_start = 0

    enc_i2t = {
        # common
        "type": "",  # options: "", i2t_t2h
        "i2t_type": "conv",  # options: conv, slot_con
        "t2h_type": "",  # options: "", conv
        "load_path": "",  # a path from which to load pretrained module
        "load_path_h": "",  # a path from which to load pretrained t2h module
        "downscale_to_1x1": False,
        "output_channels": (48, 48, 48, 48, 48),
        "kernel_shape": (8, 8, 8, 2, 2),
        "stride": (2, 2, 2, 2, 2),
        "normalization_module": "batch",
        "non_linearity": "relu",
        "n_stack": 1,
        "append_xy_mesh": True,
        "out_map": 'nonlinear', # none, nonl, mlp
        "out_map_shared": True,
        "out_map_non_linearity": "relu",
        "out_map_bn": False,
        # VIN config
        "n_dims": 64,  # VIN uses 64, but for NRI we need 4 (x, y, vx, vy)
    }

    obj_vae = {
        "enabled": False,
        "net": 'linear',  # linear, mlp
        "net_non_linearity": "none",
        "net_bn": False,
        "anneal_obj_kl": False,
    }

    enc_t2r = {
        "type": "",  # options: nri
        "load_path": "",  # a path from which to load pretrained module
        "detach_i2t_grad": False,
        # NRI encoder to get latent edges
        "n_hidden": 64,  # Number of hidden units.
        "factor": True,
        "dropout_rate": 0.0,
        "dynamic": True,  # if True, infer edges at every time step
        "horizon": 10,  # number of time steps to infer edges for current step
        "mp_HRN": False,
        "mp_HRN_shared": True,
    }

    dyn_t2t = {
        "type": "",  # options: nri, rnem, LSTM, RNN
        "load_path": "",  # a path from which to load pretrained module
        # common
        "n_hidden": 64,  # Number of hidden units.
        # NRI
        "dropout_rate": 0.0,
        "skip_first_edge_type": True,
        # which adjacency matrix to use for MP?
        "adj_mat": "t2r_nri",  # t2r_nri, t2r_nri_H, gt, full, full_H
        #   t2r_nri   - use inferred adj matrix via enc_t2r.type="nri"
        #   t2r_nri_H - use inferred adj matrix via enc_t2r.type="nri",
        #               but start from the (sparse) full_adj mat in meta_data
        #               check dataset.py for details on full_adj
        #   gt        - use ground truth adj matrix
        #   full      - MP between ALL pairs of nodes
        #   full_H    - do MP sparsely (based on meta_data full_adj)
        "last_level": False,  # MP between ALL pairs of nodes, but ONLY in
                              # last level - intended for "t2r_nri" or "full"
        "mp_HRN": False,
        "mp_HRN_shared": True,
        "mp_input": "hidden",  # hidden, cell or both
        # RNEM
        "use_attention": True,
        # LSTM, RNN
        "use_in_mlp": False,
        "independent_objs": False,
        "input_delta": False,
        "predict_delta": True,
        "rollout_zeros": False,
        "rollout_zeros_in_train": False,
        "n_layers": 1,  # Number of layers for LSTMBaseline from NRI
    }

    dec_t2i = {
        "type": "",  # options: slot_conv, par_conv,
        "load_path": "",  # a path from which to load pretrained module
        "in_channels":  (64, 64, 64, 64),
        "kernel_shape": (4, 4, 4, 4),
        "conv_stride":  (2, 2, 2, 2),
        "append_object_idcs": True,
        "append_xy_mesh": True,
        "use_output_sigmoid": False,
        "compositing": "sum",
        "compositing_detach": False,
        "compositing_eps": 0.1,
        # ParConvDecoder
        "broadcast_input": False,  # Broadcast input vector - akin to SBDecoder
        "use_skip": False,
        "skip_code": "YYYYY",  # YYYYY, NNNNY, NNNYY, NNYYY, etc
        "enc_skip_repeat_curr": True,
        "in_map": 'nonlinear', # none, nonlinear, mlp NOTE: DEFAULT SLOT: nonlinear, DEFAULT PAR: none
        "in_map_non_linearity": "relu",
        "in_map_bn": False,
    }

    pred_i2i = {
        "type": "",  # rnn, lstm, convlstm
        "load_path": "",  # a path from which to load pretrained module
        # RNN, LSTM n_hidden
        "n_hidden": 2048,  # input size is 32*32*3=3072
        # ConvLSTM settings
        "filter_size": 4,
        "n_features": 24,
        "n_layers": 3,
    }


class Model(nn.Module):

    @m.capture
    def __init__(self, meta_data, temp,
                 train_pred_steps, valid_pred_steps, test_pred_steps,
                 opt_dynamics_start, par_to_opt, increase_pred_steps_every,
                 n_slots, n_slot_dims, decode_masked_slots,
                 enc_i2t, enc_t2r, dyn_t2t, dec_t2i, pred_i2i, obj_vae,
                 l_alpha_profile,
                 l_alpha_trj, l_alpha_edg, l_alpha_img, l_alpha_obj,
                 l_beta_trj, l_delta_sf):
        super(Model, self).__init__()

        self.decode_masked_slots = decode_masked_slots
        self.first_batch = False

        n_atoms = meta_data['n_atoms']
        n_edge_types = meta_data['n_edge_types']
        n_dims = meta_data['n_dims']
        img_size = meta_data['img_size']

        self.n_edge_types = n_edge_types
        self.seq_len = meta_data['seq_len']
        self.n_atoms = n_atoms
        self.n_slots = n_atoms if n_slots is None else n_slots
        self.enc_i2t_type = enc_i2t["type"]
        if enc_i2t["type"] == "cnn" or enc_i2t["type"] == "block_cnn" or \
                enc_i2t["type"] == "cnn_t2h" or enc_i2t["type"] == "block_cnn_t2h" or \
                enc_i2t["type"] == "i2t_t2h":
            self.n_slots = 21
            # Overwrite adj mat meta_data
            meta_data["full_adj"] = create_CNN_HRN_full_adj()
            meta_data["hierarchy_nodes_list"] = \
                create_hierarchy_nodes_list("Hsprings", "4-4", 21)
            # Overwrite MP operations' masks L2A, WS, A2D
            fa = meta_data["full_adj"].copy()
            hnl = meta_data["hierarchy_nodes_list"]
            meta_data["mp_l2a_adjs"], meta_data["mp_ws_adj"], \
            meta_data["mp_a2d_adjs"] = create_HRN_MP_masks(fa, hnl)
            # Overwrite "last level" MP mask - used when MP only at last level
            meta_data["mp_last_level_mask"] = create_last_level_MP_mask(fa, hnl)

        self.n_slot_dims = n_dims if n_slot_dims is None else n_slot_dims
        self.n_slots_eq_n_atoms = (self.n_slots == n_atoms)
        self.slot_dims_are_valid = (self.n_slot_dims == n_dims)
        self.hnl = meta_data["hierarchy_nodes_list"]

        self.opt_dynamics_start = opt_dynamics_start
        self.par_to_opt = par_to_opt
        self.increase_pred_steps_every = increase_pred_steps_every
        self.train_pred_steps_args = train_pred_steps
        self.train_pred_steps_curr = train_pred_steps
        self.valid_pred_steps = valid_pred_steps
        self.test_pred_steps = test_pred_steps
        self.temp = temp
        self.use_2_t2t_dec = not dyn_t2t["skip_first_edge_type"]

        # Generate off-diagonal interaction graph
        self.dyn_t2t_sparse_edge_inference = False
        self.mp_full_adj = meta_data['full_adj']

        if dyn_t2t["adj_mat"] == "t2r_nri_H" or dyn_t2t["adj_mat"] == "full_H":
            self.mp_adj = meta_data['full_adj']  # note: this is sparse
            self.dyn_t2t_sparse_edge_inference = True
        elif dyn_t2t["adj_mat"] == "t2r_nri" or \
                dyn_t2t["adj_mat"] == "full" or dyn_t2t["adj_mat"] == "gt":
            # this will also be "dyn_t2t["adj_mat"] == "full"" case
            self.mp_adj = self.create_model_full_adj(self.n_slots, meta_data)
        else:
            raise NotImplementedError('dyn_t2t adj_mat needs to be some from '
                                      '["t2r_nri", "t2r_nri_H", '
                                      '"gt", "full", "full_H"]')

        meta_data['model_full_adj'] = self.create_model_full_adj(self.n_slots,
                                                                 meta_data)

        if dyn_t2t["last_level"]:
            self.mp_adj *= meta_data['mp_last_level_mask']
            self.dyn_t2t_sparse_edge_inference = True

        self.rel_rec_full, self.rel_send_full, self.mp_adj_idcs = \
            self.adj_2_onehot_rel(self.mp_adj)

        self.mp_adj = self.mp_adj
        self.last_level_loss = dyn_t2t["last_level"]

        if enc_t2r["mp_HRN"]:
            self.dyn_t2t_sparse_edge_inference = True

        # Create HRN MP relations: receivers, senders
        # for 3 MP phases: L2A, WS, A2D
        # Moreover, for each phase create index which will map
        # (inferred) edges from complete adj mat to edges of this stage:
        # These are used in the t2t NRI decoder for HRN-style MP
        # mp_l2a_2_mp_adj_idcs - list of L2A edge idcs in latent_edge_samples
        # mp_ws_2_mp_adj_idcs - list of WS edge idcs in latent_edge_samples
        # mp_a2d_2_mp_adj_idcs - list of A2D edge idcs in latent_edge_samples
        self.rel_rec_l2a, self.rel_send_l2a = [], []
        self.mp_l2a_2_mp_adj_idcs = []
        for adj in meta_data["mp_l2a_adjs"]:
            # Has n_level (in hierarchy) stages
            rec, send, idcs = self.adj_2_onehot_rel(adj)
            self.rel_rec_l2a.append(rec)
            self.rel_send_l2a.append(send)
            self.mp_l2a_2_mp_adj_idcs.append(
                self.map_adj_idcs(idcs, self.mp_adj_idcs))

        self.rel_rec_ws, self.rel_send_ws, self.mp_ws_adj_idcs = \
            self.adj_2_onehot_rel(meta_data["mp_ws_adj"])
        self.mp_ws_2_mp_adj_idcs = self.map_adj_idcs(self.mp_ws_adj_idcs,
                                                     self.mp_adj_idcs)

        self.rel_rec_a2d, self.rel_send_a2d = [], []
        self.mp_a2d_2_mp_adj_idcs = []
        for adj in meta_data["mp_a2d_adjs"]:
            # Has n_level (in hierarchy) stages
            rec, send, idcs = self.adj_2_onehot_rel(adj)
            self.rel_rec_a2d.append(rec)
            self.rel_send_a2d.append(send)
            self.mp_a2d_2_mp_adj_idcs.append(
                self.map_adj_idcs(idcs, self.mp_adj_idcs))

        # Image Encoder
        if enc_i2t["type"] == "i2t_t2h":
            if enc_i2t["i2t_type"] == 'conv' or \
                    enc_i2t["i2t_type"] == 'slot_conv':
                self.enc_i2t = I2TEncConv(self.n_slot_dims, img_size, enc_i2t)
            elif enc_i2t["i2t_type"] == 'vgg':
                self.enc_i2t = I2TEncVGG(self.n_slot_dims, img_size, enc_i2t)
            elif enc_i2t["i2t_type"] == 'dcgan':
                self.enc_i2t = I2TEncDCGAN(self.n_slot_dims, img_size, enc_i2t)
            else:
                raise NotImplementedError
            if enc_i2t["t2h_type"] == 'conv':
                self.enc_t2h = T2HEncConv(self.n_slot_dims, img_size, enc_i2t)
            elif enc_i2t["t2h_type"] == 'vgg':
                self.enc_t2h = T2HEncVGG(self.n_slot_dims, img_size, enc_i2t)
            elif enc_i2t["t2h_type"] == 'dcgan':
                self.enc_t2h = T2HEncDCGAN(self.n_slot_dims, img_size, enc_i2t)
            elif enc_i2t["t2h_type"] == "":
                pass
            else:
                raise NotImplementedError
        elif enc_i2t["type"] == "":
            pass
        else:
            raise NotImplementedError('i2t encoder type unknown')
        if (n_slots is not None or n_slot_dims is not None) \
                and enc_i2t["type"] == "":
            raise NotImplementedError('Custom number of slots (dims) allowed '
                                      'only for i2t encoder')

        if obj_vae["enabled"]:
            self.obj_vae = ObjectSlotVAE(self.n_slot_dims, obj_vae)
        self.l_alpha_loss_profile = l_alpha_profile
        self.l_alpha_trj_args = l_alpha_trj
        self.l_alpha_edg_args = l_alpha_edg
        self.l_alpha_img_args = l_alpha_img
        self.l_alpha_obj_args = l_alpha_obj
        self.anneal_obj_kl = obj_vae['anneal_obj_kl']
        self.l_alpha_trj_curr = torch.tensor(l_alpha_trj, dtype=torch.float32).cuda()
        self.l_alpha_edg_curr = torch.tensor(l_alpha_edg, dtype=torch.float32).cuda()
        self.l_alpha_img_curr = torch.tensor(l_alpha_img, dtype=torch.float32).cuda()
        self.l_alpha_obj_curr = torch.tensor(l_alpha_obj, dtype=torch.float32).cuda()
        self.zero = torch.tensor(0, dtype=torch.float32).cuda()
        self.one = torch.tensor(1, dtype=torch.float32).cuda()
        self.l_beta_trj = l_beta_trj
        self.l_delta_sf = l_delta_sf

        # Message passing encoder: trajectories (objects) to relations
        self.enc_t2r_detach_i2t_grad = enc_t2r["detach_i2t_grad"]
        if enc_t2r["type"] == "nri":
            self.enc_t2r = NRIMLPEncoder(self.seq_len, self.n_slot_dims,
                                         n_edge_types, enc_t2r,
                                         self.rel_rec_full, self.rel_send_full,
                                         self.rel_rec_l2a, self.rel_send_l2a,
                                         self.rel_rec_ws, self.rel_send_ws,
                                         self.rel_rec_a2d, self.rel_send_a2d)
        elif enc_t2r["type"] == "":
            pass
        else:
            raise NotImplementedError('t2r encoder needs to be some from '
                                      '["nri"]')

        # Dynamics: message passing or lstm/rnn
        self.dyn_t2t_type = dyn_t2t["type"]
        self.dyn_t2t_adj_mat = dyn_t2t["adj_mat"]
        self.dyn_t2t_last_level = dyn_t2t["last_level"]
        n_slots_bl = self.n_slots
        if self.dyn_t2t_last_level:
            n_slots_bl = len(self.hnl[-1])
        if dyn_t2t["type"] == 'nri':
            self.dyn_t2t = NRIMLPDecoder(self.n_slot_dims, n_edge_types, dyn_t2t,
                                         self.rel_rec_full, self.rel_send_full,
                                         self.rel_rec_l2a, self.rel_send_l2a,
                                         self.mp_l2a_2_mp_adj_idcs,
                                         self.rel_rec_ws, self.rel_send_ws,
                                         self.mp_ws_2_mp_adj_idcs,
                                         self.rel_rec_a2d, self.rel_send_a2d,
                                         self.mp_a2d_2_mp_adj_idcs)

        elif dyn_t2t["type"] == 'nri_lstm':
            self.dyn_t2t = LSTMDecoder(self.n_slot_dims, n_edge_types, dyn_t2t,
                                       self.rel_rec_full, self.rel_send_full,
                                       self.rel_rec_l2a, self.rel_send_l2a,
                                       self.mp_l2a_2_mp_adj_idcs,
                                       self.rel_rec_ws, self.rel_send_ws,
                                       self.mp_ws_2_mp_adj_idcs,
                                       self.rel_rec_a2d, self.rel_send_a2d,
                                       self.mp_a2d_2_mp_adj_idcs)
        elif dyn_t2t["type"] == 'lstm':
            self.dyn_t2t = LSTMBaseline(n_slots_bl, self.n_slot_dims, dyn_t2t)
        elif dyn_t2t["type"] == "":
            pass
        else:
            raise NotImplementedError('t2t dyn pred needs to be some from '
                                      '["nri", "rnem", "rnn", "lstm"]')

        # t2i decoder
        self.use_skip = dec_t2i['use_skip']
        self.enc_skip_repeat_curr = dec_t2i['enc_skip_repeat_curr']
        skip_n_channels = self.get_skip_n_channels()
        if dec_t2i["type"] == "slot_conv":
            self.dec_t2i = SlotConvDecoder(self.n_slot_dims, img_size, dec_t2i, skip_n_channels)
        elif dec_t2i["type"] == "par_conv":
            self.dec_t2i = ParConvDecoder(self.n_slot_dims, img_size, dec_t2i, skip_n_channels)
        elif dec_t2i["type"] == "slot_vgg":
            self.dec_t2i = SlotVGGDecoder(self.n_slot_dims, img_size, dec_t2i, skip_n_channels)
        elif dec_t2i["type"] == "par_vgg":
            self.dec_t2i = ParVGGDecoder(self.n_slot_dims, img_size, dec_t2i, skip_n_channels)
        elif dec_t2i["type"] == "slot_dcgan":
            self.dec_t2i = SlotDCGANDecoder(self.n_slot_dims, img_size, dec_t2i, skip_n_channels)
        elif dec_t2i["type"] == "par_dcgan":
            self.dec_t2i = ParDCGANDecoder(self.n_slot_dims, img_size, dec_t2i, skip_n_channels)
        elif dec_t2i["type"] == "par_sb":
            self.dec_t2i = ParSBDecoder(self.n_slot_dims, img_size, dec_t2i, skip_n_channels)
        elif dec_t2i["type"] == "slot_sb":
            self.dec_t2i = SlotSBDecoder(self.n_slot_dims, img_size, dec_t2i, skip_n_channels)
        elif dec_t2i["type"] == "":
            pass
        else:
            raise NotImplementedError('t2i decoder type needs to be some from '
                                      '["sb", "slot_conv", "par_conv"]')

        # Image to Image (video) predictor
        if pred_i2i["type"] == 'lstm':
            self.predictor_i2i = LSTMBaseline(img_size, pred_i2i)
        elif pred_i2i["type"] == "":
            pass
        else:
            raise NotImplementedError('Prediction module type needs to be '
                                      'some from ["RNN", "LSTM", "ConvLSTM"]')

        self.load_pretrained_modules(enc_i2t, enc_t2r, dyn_t2t, dec_t2i, pred_i2i)

    def adj_2_onehot_rel(self, adj):
        # idx: 1 for receivers, 0 for senders
        adj_idcs = np.where(adj)
        rel_rec = np.array(encode_onehot(adj_idcs[1], self.n_slots),
                           dtype=np.float32)
        rel_send = np.array(encode_onehot(adj_idcs[0], self.n_slots),
                            dtype=np.float32)
        rel_rec = torch.FloatTensor(rel_rec).cuda()
        rel_send = torch.FloatTensor(rel_send).cuda()
        return rel_rec, rel_send, adj_idcs

    def map_adj_idcs(self, csi, fai):
        # Example:
        # csi - custom_adj_idcs:
        # [3 4 5 6]
        # [1 1 2 2]
        # fai - full_adj_idcs:
        # [0 0 1 1 1 2 2 2 3 3 4 4 5 5 6 6]
        # [1 2 0 3 4 0 5 6 1 4 1 3 2 6 2 5]
        # idx_map:
        # [8, 10, 12, 14]
        idx_map = []
        for i in range(len(csi[0])):
            for j in range(len(fai[0])):
                if csi[0][i] == fai[0][j] and csi[1][i] == fai[1][j]:
                    idx_map.append(j)
                    break
        return idx_map

    def create_model_full_adj(self, n_slots, meta_data):
        full_adj = np.ones([n_slots, n_slots])
        if not meta_data['use_self_edges']:
            full_adj -= np.eye(n_slots)
        return full_adj

    def get_skip_n_channels(self):
        skip_n_channels=None
        if hasattr(self, 'enc_i2t'):
            skip_n_channels = self.enc_i2t.skip_n_channels
            if not self.use_skip:
                skip_n_channels = [s * 0 for s in skip_n_channels]
        elif self.use_skip:
            raise NotImplementedError('cannot use skip connecitons w/o enc_i2t')
        return skip_n_channels

    def encode_i2t(self, inputs):
        traj_emb_pred = None
        enc_skip = None
        kl_obj = None
        if hasattr(self, 'enc_i2t'):
            if hasattr(self, 'enc_t2h'):
                x16, x16ord, enc_skip = self.enc_i2t(inputs['imgs'])
                traj_emb_pred = self.enc_t2h(inputs['imgs'], x16, x16ord)
            else:
                # traj_emb_pred, _, enc_skip = self.enc_i2t(inputs['imgs'])
                x16, x16ord, enc_skip = self.enc_i2t(inputs['imgs'])
                ishp = list(inputs['imgs'].size())
                traj_emb_pred = x16ord.view(
                    ishp[0], ishp[1], x16ord.size(-2), x16ord.size(-1))
                # For NRI we need [num_sims, n_atoms, num_timesteps, num_dims]
                traj_emb_pred = traj_emb_pred.transpose(1, 2).contiguous()

            kl_obj = None
            if self.l_alpha_obj_curr > 0:
                if hasattr(self, 'obj_vae'):
                    traj_emb_pred, kl_obj = self.obj_vae(traj_emb_pred)
        return traj_emb_pred, enc_skip, kl_obj

    def encode_t2r(self, inputs, traj_emb_pred, n_epoch):
        latent_edge_samples, latent_edge_probs, latent_edge_logits = \
            None, None, None
        if self.opt_dynamics(n_epoch) and hasattr(self, 'enc_t2r'):
            if traj_emb_pred is None:
                nri_input = inputs['traj']
            else:
                nri_input = traj_emb_pred
                if self.enc_t2r_detach_i2t_grad:
                    nri_input = nri_input.detach()

            latent_edge_logits = self.enc_t2r(nri_input)

            latent_edge_samples = gumbel_softmax(latent_edge_logits,
                                                 self.temp,
                                                 hard=not self.training)

            latent_edge_probs = F.softmax(latent_edge_logits, dim=-1)

            # [B * T, n_atoms * (n_atoms - 1), 256] - dynamic graph inference
            # [B * 1, n_atoms * (n_atoms - 1), 256] - static graph inference
            batch_size = nri_input.size(0)
            shp = latent_edge_logits.shape
            shp = [batch_size, -1] + list(shp[1:])
            latent_edge_logits = latent_edge_logits.view(shp)
            latent_edge_samples = latent_edge_samples.view(shp)
            latent_edge_probs = latent_edge_probs.view(shp)

        return latent_edge_samples, latent_edge_probs, latent_edge_logits

    def create_t2t_adj_mat(self, latent_edges, gt_edges):
        # for details check smodel.dyn_t2t['adj_mat']
        mp_dec_adj = latent_edges
        if self.dyn_t2t_adj_mat == "gt" or self.dyn_t2t_adj_mat == "full" or \
                self.dyn_t2t_adj_mat == "full_H":
            if self.dyn_t2t_adj_mat == "gt":
                edges = gt_edges
            else:
                # full or full_H cases - just create as many 1's as necessary
                edges = torch.ones((gt_edges.size(0), int(self.mp_adj.sum())),
                                   dtype=gt_edges.dtype,
                                   device=gt_edges.device)
            # create one hot edges from the GT inputs
            shp = list(edges.shape) + [self.n_edge_types]
            mp_dec_adj = torch.zeros(shp,
                                     dtype=edges.dtype,
                                     device=edges.device)
            edges = edges.unsqueeze(-1)
            mp_dec_adj.scatter_(-1, edges, 1)
            # unsqueeze to match dynamic graph case
            mp_dec_adj = mp_dec_adj.unsqueeze(1)
        return mp_dec_adj

    def dynamics_t2t(self, inputs, traj_emb_pred, latent_edges, n_epoch,
                     pred_steps, is_test):
        # Predict trajectory dynamics: either by NRI, VIN or baseline RNN/LSTM
        # take the (x, y, v_x, v_y) from the dataset iterator
        mp_dec_input = inputs['traj']

        # overwrite GT trajectories if encoder is i2t
        traj_pred, traj_target = None, None
        if hasattr(self, 'enc_i2t') and traj_emb_pred is not None:
            mp_dec_input = traj_emb_pred
            traj_pred = mp_dec_input[:, :, 1:, :]
            traj_target = mp_dec_input[:, :, 1:, :]

        bis = None
        bis_test = None
        self.enc_skip_pred_steps = 1
        self.enc_skip_test_pred_steps = 1

        # NRI message passing w/ latent graph inference
        test_traj_pred, test_traj_target = None, None
        if self.opt_dynamics(n_epoch):
            if hasattr(self, 'dyn_t2t'):
                self.enc_skip_pred_steps = pred_steps
                self.enc_skip_test_pred_steps = self.test_pred_steps

                # Determine adj mat use for MP
                mp_dec_adj = self.create_t2t_adj_mat(latent_edges,
                                                     inputs['edges'])

                if self.dyn_t2t_type == 'nri':
                    traj_pred = self.dyn_t2t(mp_dec_input, mp_dec_adj,
                                             pred_steps)

                    if is_test:
                        mp_dec_input_test = mp_dec_input[:, :, -self.test_pred_steps-1:, :]
                        # dynamic graph case
                        if mp_dec_adj.size(1) > mp_dec_input_test.size(2):
                            mp_dec_adj = mp_dec_adj[:, -self.test_pred_steps-1:]
                        test_traj_pred = self.dyn_t2t(mp_dec_input_test, mp_dec_adj,
                                                      self.test_pred_steps)
                        test_traj_pred = test_traj_pred[:, :, -self.test_pred_steps:, :]
                elif self.dyn_t2t_type == 'nri_rnn' or \
                        self.dyn_t2t_type == 'nri_lstm':
                    bis = mp_dec_input.size(2) - pred_steps
                    traj_pred = self.dyn_t2t(mp_dec_input, mp_dec_adj,
                                             pred_steps,
                                             burn_in=True, burn_in_steps=bis)
                    if is_test:
                        bis_test = mp_dec_input.size(2) - self.test_pred_steps
                        test_traj_pred = self.dyn_t2t(mp_dec_input, mp_dec_adj,
                                                      self.test_pred_steps,
                                                      burn_in=True,
                                                      burn_in_steps=bis_test)
                        test_traj_pred = test_traj_pred[:, :, -self.test_pred_steps:, :]
                elif self.dyn_t2t_type == 'last_step_baseline':
                    traj_pred = self.dyn_t2t(mp_dec_input, self.seq_len - 1)
                    if is_test:
                        test_traj_pred = self.dyn_t2t(mp_dec_input,
                                                      self.test_pred_steps)
                else:
                    # rnem, vin, rnn, lstm
                    bis = mp_dec_input.size(2) - pred_steps

                    if self.dyn_t2t_last_level:
                        # predict only last level nodes
                        mp_dec_input = mp_dec_input[:, self.hnl[-1]]

                    traj_pred = self.dyn_t2t(mp_dec_input, pred_steps,
                                             burn_in=True, burn_in_steps=bis)
                    if is_test:
                        bis_test = mp_dec_input.size(2) - self.test_pred_steps
                        test_traj_pred = self.dyn_t2t(mp_dec_input,
                                                      self.test_pred_steps,
                                                      burn_in=True,
                                                      burn_in_steps=bis_test)
                        test_traj_pred = test_traj_pred[:, :, -self.test_pred_steps:, :]
                if is_test:
                    # -pred_steps because we won't be substracting in loss comp
                    test_traj_target = mp_dec_input[:, :, -self.test_pred_steps:, :]
            traj_target = mp_dec_input[:, :, 1:, :]

        self.enc_skip_bis = bis
        self.enc_skip_bis_test = bis_test

        return traj_pred, traj_target, test_traj_pred, test_traj_target

    def enc_skip_repeat_curr_frame(self, enc_skip, start_idx, pred_steps, bis):

        if not self.use_skip:
            return enc_skip

        if pred_steps == 1 or not self.enc_skip_repeat_curr:
            enc_skip = [e[:, start_idx:] for e in enc_skip]
            return enc_skip

        ret = []
        if bis:
            # recurrent case, nri_rnn, rnn, lstm ...
            for e in enc_skip:
                c = e.clone()
                for b in range(bis, c.shape[1]):
                    c[:, b] = c[:, bis]
                ret.append(c)
        else:
            # nri mlp
            for e in enc_skip:
                curr_frame = 0
                c = e.clone()
                for b in range(c.shape[1]):
                    if b % pred_steps == 0:
                        curr_frame = b
                    else:
                        c[:, b] = c[:, curr_frame]
                ret.append(c)

        ret = [e[:, start_idx:] for e in ret]
        return ret

    def decode_t2i(self, traj_pred, test_traj_pred, enc_skip):
        # Decode predicted trajectories to images
        imgs_pred, imgs_pred_objs = None, None
        test_imgs_pred, test_imgs_pred_objs = None, None
        if hasattr(self, 'dec_t2i'):
            dms = self.decode_masked_slots and self.first_batch
            # take only last level nodes
            # IF exist due to LSTM & RNN predicting only last level nodes
            if traj_pred.size(1) > len(self.hnl[-1]):
                traj_pred = traj_pred[:, self.hnl[-1]]
            enc_skip_input = self.enc_skip_repeat_curr_frame(
                enc_skip, 1, self.enc_skip_pred_steps, self.enc_skip_bis)
            imgs_pred, imgs_pred_objs = self.dec_t2i(traj_pred, enc_skip_input, dms)

            if test_traj_pred is not None:
                # take only last level nodes
                # IF exist due to LSTM & RNN predicting only last level nodes
                if test_traj_pred.size(1) > len(self.hnl[-1]):
                    test_traj_pred = test_traj_pred[:, self.hnl[-1]]
                    enc_skip_input = self.enc_skip_repeat_curr_frame(
                    enc_skip, -self.test_pred_steps,
                    self.enc_skip_test_pred_steps, self.enc_skip_bis_test)
                test_imgs_pred, test_imgs_pred_objs = \
                    self.dec_t2i(test_traj_pred, enc_skip_input, dms)

        return imgs_pred, imgs_pred_objs, test_imgs_pred, test_imgs_pred_objs

    def predict_i2i(self, inputs, pred_steps):
        imgs = inputs['imgs']
        imgs_pred = self.pred_i2i(imgs, pred_steps)
        return imgs_pred

    def get_dynamics_params(self):
        params = []
        if hasattr(self, 'enc_t2h'):
            params += list(self.enc_t2h.parameters())
        if hasattr(self, 'enc_t2r'):
            params += list(self.enc_t2r.parameters())
        if hasattr(self, 'dyn_t2t'):
            params += list(self.dyn_t2t.parameters())
        return params

    def get_image_enc_dec_params(self):
        params = []
        if hasattr(self, 'enc_i2t'):
            params += list(self.enc_i2t.parameters())
        if hasattr(self, 'obj_vae'):
            params += list(self.obj_vae.parameters())
        if hasattr(self, 'dec_t2i'):
            params += list(self.dec_t2i.parameters())
        if hasattr(self, 'pred_i2i'):
            params += list(self.pred_i2i.parameters())
        return params

    def opt_dynamics(self, n_epoch):
        return n_epoch >= self.opt_dynamics_start

    def params_to_optimize(self, n_epoch):
        res = self.par_to_opt
        if not self.opt_dynamics(n_epoch) and self.par_to_opt == 'all':
            res = 'img'
        return res

    def update_loss_coefficients(self, n_epoch):
        if self.l_alpha_loss_profile == 1:
            self.l_alpha_trj_curr = self.zero
            self.l_alpha_img_curr = self.one
        elif self.l_alpha_loss_profile == 2:
            self.l_alpha_trj_curr = self.one
            self.l_alpha_img_curr = self.zero
        elif self.l_alpha_loss_profile == 3:
            self.l_alpha_trj_curr = self.one
            self.l_alpha_img_curr = self.one

        if self.anneal_obj_kl:
            max_epoch = 50
            if self.opt_dynamics_start > 0:
                max_epoch = self.opt_dynamics_start
            self.l_alpha_obj_curr = self.l_alpha_obj_args * \
                torch.tensor(np.minimum(1, n_epoch / max_epoch),
                             dtype=torch.float32).cuda()

        if self.increase_pred_steps_every > 0 and n_epoch >= self.opt_dynamics_start:
            curr_pred_step = (n_epoch - self.opt_dynamics_start) // self.increase_pred_steps_every + 2
            self.train_pred_steps_curr = np.minimum(
                self.train_pred_steps_args, curr_pred_step)

    def reset_best_valid_loss(self, n_epoch):
        return n_epoch == self.opt_dynamics_start

    def save(self, exp_dir):
        # all params
        save_file = os.path.join(exp_dir, 'model_all.pt')
        torch.save(self.state_dict(), save_file)

        # dyn params
        if hasattr(self, 'enc_t2h'):
            save_file = os.path.join(exp_dir, 'model_enc_t2h.pt')
            torch.save(self.enc_t2h.state_dict(), save_file)
        if hasattr(self, 'enc_t2r'):
            save_file = os.path.join(exp_dir, 'model_enc_t2r.pt')
            torch.save(self.enc_t2r.state_dict(), save_file)
        if hasattr(self, 'dyn_t2t'):
            save_file = os.path.join(exp_dir, 'model_dyn_t2t.pt')
            torch.save(self.dyn_t2t.state_dict(), save_file)

        # img params
        if hasattr(self, 'enc_i2t'):
            save_file = os.path.join(exp_dir, 'model_enc_i2t.pt')
            torch.save(self.enc_i2t.state_dict(), save_file)
        if hasattr(self, 'obj_vae'):
            save_file = os.path.join(exp_dir, 'model_obj_vae.pt')
            torch.save(self.obj_vae.state_dict(), save_file)
        if hasattr(self, 'dec_t2i'):
            save_file = os.path.join(exp_dir, 'model_dec_t2i.pt')
            torch.save(self.dec_t2i.state_dict(), save_file)
        if hasattr(self, 'pred_i2i'):
            save_file = os.path.join(exp_dir, 'model_pred_i2i.pt')
            torch.save(self.pred_i2i.state_dict(), save_file)

    def load_pretrained_modules(self, enc_i2t, enc_t2r, dyn_t2t, dec_t2i,
                                pred_i2i):
        # dyn params
        if hasattr(self, 'enc_t2h'):
            if enc_i2t['load_path_h'] != '':
                module_file = os.path.join(enc_i2t['load_path_h'],
                                           'model_enc_t2h.pt')
                self.enc_t2h.load_state_dict(torch.load(module_file))
                info = 'Loaded pretrained module: {}'.format(module_file)
                print(info)
        if hasattr(self, 'enc_t2r'):
            if enc_t2r['load_path'] != '':
                module_file = os.path.join(enc_t2r['load_path'],
                                           'model_enc_t2r.pt')
                self.enc_t2r.load_state_dict(torch.load(module_file))
                info = 'Loaded pretrained module: {}'.format(module_file)
                print(info)
        if hasattr(self, 'dyn_t2t'):
            if dyn_t2t['load_path'] != '':
                module_file = os.path.join(dyn_t2t['load_path'],
                                           'model_dyn_t2t.pt')
                self.dyn_t2t.load_state_dict(torch.load(module_file))
                info = 'Loaded pretrained module: {}'.format(module_file)
                print(info)

        # img params
        if hasattr(self, 'enc_i2t'):
            if enc_i2t['load_path'] != '':
                module_file = os.path.join(enc_i2t['load_path'],
                                           'model_enc_i2t.pt')
                self.enc_i2t.load_state_dict(torch.load(module_file))
                info = 'Loaded pretrained module: {}'.format(module_file)
                print(info)
        if hasattr(self, 'obj_vae'):
            if enc_i2t['load_path'] != '':
                module_file = os.path.join(enc_i2t['load_path'],
                                           'model_obj_vae.pt')
                self.obj_vae.load_state_dict(torch.load(module_file))
                info = 'Loaded pretrained module: {}'.format(module_file)
                print(info)
        if hasattr(self, 'dec_t2i'):
            if dec_t2i['load_path'] != '':
                module_file = os.path.join(dec_t2i['load_path'],
                                           'model_dec_t2i.pt')
                pretrained_dict = torch.load(module_file)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                   (k not in ["x_grid", "y_grid"])}
                self.dec_t2i.load_state_dict(pretrained_dict, strict=False)
                info = 'Loaded pretrained module: {}'.format(module_file)
                print(info)
        if hasattr(self, 'pred_i2i'):
            if pred_i2i['load_path'] != '':
                module_file = os.path.join(pred_i2i['load_path'],
                                           'model_pred_i2i.pt')
                self.pred_i2i.load_state_dict(torch.load(module_file))
                info = 'Loaded pretrained module: {}'.format(module_file)
                print(info)

    def forward(self, inputs, n_epoch, is_test=False):
        # This is the key to getting nice latent graph inference
        if self.training:
            pred_steps = self.train_pred_steps_curr
        else:
            pred_steps = self.valid_pred_steps

        traj_emb_pred, enc_skip, kl_obj = self.encode_i2t(inputs)

        latent_edge_samples, latent_edge_probs, latent_edge_logits = \
            self.encode_t2r(inputs, traj_emb_pred, n_epoch)

        traj_pred, traj_target, test_traj_pred, test_traj_target = \
            self.dynamics_t2t(inputs, traj_emb_pred, latent_edge_samples,
                              n_epoch, pred_steps, is_test=is_test)

        imgs_pred, imgs_pred_objs, test_imgs_pred, test_imgs_pred_objs = \
            self.decode_t2i(traj_pred, test_traj_pred, enc_skip)

        test_imgs_target = None
        if test_imgs_pred is not None:
            # -pred_steps because we won't be substracting in loss comp
            test_imgs_target = inputs['imgs'][:, -self.test_pred_steps:]

        if hasattr(self, 'pred_i2i'):
            imgs_pred = self.predict_i2i(inputs, pred_steps)

        outputs = {
            'traj_pred': traj_pred,
            'traj_target': traj_target,
            'kl_obj': kl_obj,
            'latent_edge_logits': latent_edge_logits,
            'latent_edge_probs': latent_edge_probs,
            'latent_edge_samples': latent_edge_samples,
            'imgs_pred': imgs_pred,
            'imgs_pred_objs': imgs_pred_objs,
            'test_traj_pred': test_traj_pred,
            'test_traj_target': test_traj_target,
            'test_imgs_pred': test_imgs_pred,
            'test_imgs_target': test_imgs_target,
            'test_imgs_pred_objs': test_imgs_pred_objs,
        }

        return outputs


@m.capture
def compute_loss(i, o, meta_data, model,
                 output_variance, test_pred_steps):
    # Compute trajectory loss
    traj_nll, traj_nll_partial = nll_gaussian(
        o['traj_pred'], o['traj_target'], output_variance, avg_up_to_dim=-2,
        last_level_nodes=meta_data["hierarchy_nodes_list"][-1])
    if model.last_level_loss:
        traj_nll = traj_nll_partial
    else:
        traj_nll_partial = traj_nll_partial.detach()
    traj_mse = F.mse_loss(o['traj_pred'], o['traj_target'])
    if o['test_traj_pred'] is not None:
        traj_pred_mse = ((o['test_traj_pred'] -
                          o['test_traj_target']) ** 2).mean(dim=[0, 1, -1])
    else:
        traj_pred_mse = torch.tensor([0] * test_pred_steps)

    # Compute slow-features MSE loss
    sf_traj_nll, _ = nll_gaussian(o['traj_pred'][:, :, 1:, :],
                                  o['traj_pred'][:, :, :-1, :],
                                  output_variance, avg_up_to_dim=-2)

    # Compute latent graph loss, if inferred
    if o['latent_edge_probs'] is not None:
        # Case where we encode latent graph (NRI)
        # Note: we assume that the prior is uniform
        kl_edge = kl_categorical_uniform(o['latent_edge_probs'],
                                              meta_data['n_atoms'],
                                              meta_data['n_edge_types'])
    else:
        kl_edge = torch.tensor(0)

    # Compute visual mse loss
    if o['imgs_pred'] is not None:
        imgs_target = i['imgs'][:, 1:, :, :, :]
        imgs_nll, _ = nll_gaussian(o['imgs_pred'], imgs_target,
                                   output_variance, avg_up_to_dim=-3)
        imgs_mse = F.mse_loss(o['imgs_pred'], imgs_target)
        if o['test_imgs_pred'] is not None:
            imgs_pred_mse = ((o['test_imgs_pred'] - o['test_imgs_target']) ** 2
                             ).mean(dim=[0, -3, -2, -1])
            pred_len = o['test_imgs_pred'].shape[1]
            imgs_pred_nll, _ = nll_gaussian(o['test_imgs_pred'],
                                            imgs_target[:, -pred_len:],
                                            output_variance, avg_up_to_dim=-3)
        else:
            pred_len = model.valid_pred_steps
            pred = o['imgs_pred'][:, -pred_len:]
            target = imgs_target[:, -pred_len:]
            imgs_pred_mse = ((pred - target) ** 2).mean(dim=[0, -3, -2, -1])
            imgs_pred_nll, _ = nll_gaussian(pred, target,
                                            output_variance, avg_up_to_dim=-3)
    else:
        imgs_nll = torch.tensor(0)
        imgs_mse = torch.tensor(0)
        imgs_pred_mse = torch.tensor([0] * test_pred_steps)
        imgs_pred_nll = torch.tensor(0)

    # Object VAE loss
    kl_obj = o['kl_obj']
    if kl_obj is None:
        kl_obj = torch.tensor(0)

    ltraj = model.l_alpha_trj_curr * traj_nll * model.l_beta_trj
    lkle = model.l_alpha_edg_curr * kl_edge
    limg = model.l_alpha_img_curr * imgs_nll
    lklo = model.l_alpha_obj_curr * kl_obj
    lsf = model.l_delta_sf * sf_traj_nll
    # Generate optimization loss
    loss = ltraj + lkle + limg + lklo + lsf

    # Compute misc: latent edge and graph accuracy
    edge_acc, edge_acc_sparse, graph_acc = 0, 0, 0
    if o['latent_edge_samples'] is not None:
        # Case where n_slots is equal to n_atoms in input graph (traj)
        # If we infer sparse graph, complete with dummy labels
        if model.dyn_t2t_sparse_edge_inference:
            o['latent_edge_samples'] = complete_sparse_adj(
                meta_data, model.mp_adj, o['latent_edge_samples'])

        if o['latent_edge_samples'].shape[-2] == i['edges'].shape[-1]:

            # Case where we encode latent graph (NRI)
            # Calculate latent edge and graph accuracy
            edge_acc, edge_acc_sparse = edge_accuracy(
                o['latent_edge_samples'], i['edges'],
                meta_data['edges_sparse_mask'])

            _, o['latent_edge_samples'] = o['latent_edge_samples'].max(-1)
            # In case we have only edge/no-edge types, we don't know which one of
            # the last two dims is going to represent "edge" latent variable
            if edge_acc < 0.5 and model.use_2_t2t_dec:
                edge_acc = 1 - edge_acc
                edge_acc_sparse = 1 - edge_acc_sparse
                o['latent_edge_samples'] = 1 - o['latent_edge_samples']
                graph_acc = graph_accuracy(i['edges'], o['latent_edge_samples'])
            else:
                graph_acc = graph_accuracy(i['edges'], o['latent_edge_samples'])
        else:
            _, o['latent_edge_samples'] = o['latent_edge_samples'].max(-1)


    loss_report = {
        "loss": loss.item(),
        # Trajectory
        "traj_nll": traj_nll.item(),
        "traj_nll_partial": traj_nll_partial.item(),
        "sf_traj_nll": sf_traj_nll.item(),
        "traj_mse": traj_mse.item(),
        "traj_pred_mse": traj_pred_mse.cpu().detach().numpy(),
        # KL losses
        "kl_edge": kl_edge.item(),
        "kl_obj": kl_obj.item(),
        # Latent Graph
        "edge_acc": edge_acc,
        "edge_acc_sparse": edge_acc_sparse,
        "graph_acc": graph_acc,
        # Visual
        "imgs_nll": imgs_nll.item(),
        "imgs_mse": imgs_mse.item(),
        "imgs_pred_mse": imgs_pred_mse.cpu().detach().numpy(),
        "imgs_pred_nll": imgs_pred_nll.item(),
    }

    return loss, loss_report

