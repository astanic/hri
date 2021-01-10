import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import gumbel_softmax

class NRIMLPDecoder(nn.Module):
    """MLP decoder module."""
    # Initially based on: https://github.com/ethanfetaya/NRI/

    def __init__(self, n_in_node, n_edge_types, dyn_t2t,
                 rel_rec_full, rel_send_full,
                 rel_rec_l2a, rel_send_l2a, mp_l2a_2_mp_adj_idcs,
                 rel_rec_ws, rel_send_ws, mp_ws_2_mp_adj_idcs,
                 rel_rec_a2d, rel_send_a2d, mp_a2d_2_mp_adj_idcs):
        super(NRIMLPDecoder, self).__init__()
        msg_hid = dyn_t2t["n_hidden"]
        msg_out = dyn_t2t["n_hidden"]
        n_hid = dyn_t2t["n_hidden"]
        do_prob = dyn_t2t["dropout_rate"]

        self.mp_HRN = dyn_t2t["mp_HRN"]
        self.mp_HRN_shared = dyn_t2t["mp_HRN_shared"]
        self.predict_delta = dyn_t2t["predict_delta"]

        n_et = n_edge_types
        if self.mp_HRN:
            # self.mlp_node_embed = MLP(n_in_node, msg_hid, msg_hid)
            self.mlp_node_embed = nn.Linear(n_in_node, msg_hid)

            if self.mp_HRN_shared:
                edge_fc1 = nn.ModuleList(
                    [nn.Linear(2 * msg_hid, msg_hid) for _ in range(n_et)])
                edge_fc2 = nn.ModuleList(
                    [nn.Linear(msg_hid, msg_out) for _ in range(n_et)])
                self.l2a_edge_fc1 = edge_fc1
                self.l2a_edge_fc2 = edge_fc2
                self.ws_edge_fc1 = edge_fc1
                self.ws_edge_fc2 = edge_fc2
                self.a2d_edge_fc1 = edge_fc1
                self.a2d_edge_fc2 = edge_fc2
            else:
                self.l2a_edge_fc1 = nn.ModuleList(
                    [nn.Linear(2 * msg_hid, msg_hid) for _ in range(n_et)])
                self.l2a_edge_fc2 = nn.ModuleList(
                    [nn.Linear(msg_hid, msg_out) for _ in range(n_et)])
                self.ws_edge_fc1 = nn.ModuleList(
                    [nn.Linear(2 * msg_hid, msg_hid) for _ in range(n_et)])
                self.ws_edge_fc2 = nn.ModuleList(
                    [nn.Linear(msg_hid, msg_out) for _ in range(n_et)])
                self.a2d_edge_fc1 = nn.ModuleList(
                    [nn.Linear(2 * msg_hid, msg_hid) for _ in range(n_et)])
                self.a2d_edge_fc2 = nn.ModuleList(
                    [nn.Linear(msg_hid, msg_out) for _ in range(n_et)])

        else:
            self.full_edge_fc1 = nn.ModuleList(
                [nn.Linear(2 * n_in_node, msg_hid) for _ in range(n_et)])
            self.full_edge_fc2 = nn.ModuleList(
                [nn.Linear(msg_hid, msg_out) for _ in range(n_et)])

        self.msg_out_shape = msg_out

        if dyn_t2t["skip_first_edge_type"]:
            self.mp_decoders_start_idx = 1
        else:
            self.mp_decoders_start_idx = 0

        self.rr_full, self.rs_full = rel_rec_full, rel_send_full
        self.rr_l2a, self.rs_l2a = rel_rec_l2a, rel_send_l2a
        self.rr_ws, self.rs_ws = rel_rec_ws, rel_send_ws
        self.rr_a2d, self.rs_a2d = rel_rec_a2d, rel_send_a2d
        self.mp_l2a_2_mp_adj_idcs = mp_l2a_2_mp_adj_idcs
        self.mp_ws_2_mp_adj_idcs = mp_ws_2_mp_adj_idcs
        self.mp_a2d_2_mp_adj_idcs = mp_a2d_2_mp_adj_idcs

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)
        self.dropout = nn.Dropout(do_prob)

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=-1)
        return edges

    def mp_round(self, single_timestep_inputs, single_timestep_rel_type,
                 rel_rec, rel_send, sp_mp_2_mp_adj_idcs, edge_fc1, edge_fc2):
        # sp_mp_2_mp_adj_idcs: sparse MP (e.g. L2A) adj mapping to full MP
        pre_msg = self.node2edge(single_timestep_inputs, rel_rec, rel_send)

        all_msgs = torch.zeros((pre_msg.size(0), pre_msg.size(1),
                                pre_msg.size(2), self.msg_out_shape),
                               device=single_timestep_inputs.device)

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(self.mp_decoders_start_idx, len(edge_fc2)):
            msg = F.relu(edge_fc1[i](pre_msg), inplace=True)
            msg = self.dropout(msg)
            msg = F.relu(edge_fc2[i](msg), inplace=True)
            inferred_edges = single_timestep_rel_type[:, :, :, i:i + 1]
            if sp_mp_2_mp_adj_idcs is not None:
                inferred_edges = inferred_edges[:, :, sp_mp_2_mp_adj_idcs]
            msg = msg * inferred_edges
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = self.edge2node(all_msgs, rel_rec, rel_send)
        return agg_msgs

    def single_step_forward(self, single_timestep_inputs,
                            single_timestep_rel_type):
        # single_timestep_inputs shape [B, T, K, num_dims]
        # single_timestep_rel_type shape [B, T, K*(K-1), n_edge_types]

        if not self.mp_HRN:
            x = self.mp_round(single_timestep_inputs, single_timestep_rel_type,
                              self.rr_full, self.rs_full, None,
                              self.full_edge_fc1, self.full_edge_fc2)
        else:
            # L2A
            x = self.mlp_node_embed(single_timestep_inputs)
            for i in range(len(self.rr_l2a)):
                x += self.mp_round(x, single_timestep_rel_type,
                                   self.rr_l2a[i], self.rs_l2a[i],
                                   self.mp_l2a_2_mp_adj_idcs[i],
                                   self.l2a_edge_fc1, self.l2a_edge_fc2)
            # WS
            x += self.mp_round(x, single_timestep_rel_type,
                               self.rr_ws, self.rs_ws,
                               self.mp_ws_2_mp_adj_idcs,
                               self.ws_edge_fc1, self.ws_edge_fc2)
            # A2D
            for i in range(len(self.rr_a2d)):
                x += self.mp_round(x, single_timestep_rel_type,
                                   self.rr_a2d[i], self.rs_a2d[i],
                                   self.mp_a2d_2_mp_adj_idcs[i],
                                   self.a2d_edge_fc1, self.a2d_edge_fc2)

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, x], dim=-1)

        # Output MLP
        pred_delta = self.dropout(F.relu(self.out_fc1(aug_inputs), inplace=True))
        pred_delta = self.dropout(F.relu(self.out_fc2(pred_delta), inplace=True))
        pred_delta = self.out_fc3(pred_delta)

        # Add predicted position/velocity difference
        if self.predict_delta:
            x_tp1 = single_timestep_inputs + pred_delta
        else:
            x_tp1 = pred_delta
        # single_timestep_inputs += pred_delta
        return x_tp1

    def forward(self, inputs, rel_type, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.
        inputs = inputs.transpose(1, 2).contiguous()

        if rel_type.size(1) == 1:
            # static graph case
            rt_shp = rel_type.size
            sizes = [rt_shp(0), inputs.size(1), rt_shp(-2), rt_shp(-1)]
            rel_type = rel_type.expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]

        # Run n prediction steps
        for step in range(0, pred_steps):
            curr_rel_type = rel_type[:, step::pred_steps, :, :]
            if curr_rel_type.size(1) < last_pred.size(1):
                # last step - pad with dummy zeros
                # we don't care about that step - in the last line it's removed
                n_pad = last_pred.size(1) - curr_rel_type.size(1)
                curr_rel_type = F.pad(curr_rel_type, [0, 0] * 2 + [n_pad, 0])
            last_pred = self.single_step_forward(last_pred, curr_rel_type)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = torch.zeros(sizes, device=inputs.device)

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        # Remove the last step prediction
        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class LSTMDecoder(nn.Module):
    """Hierarchical MP LSTM decoder for trajectory prediction."""

    def __init__(self, n_in_node, n_edge_types, dyn_t2t,
                 rel_rec_full, rel_send_full,
                 rel_rec_l2a, rel_send_l2a, mp_l2a_2_mp_adj_idcs,
                 rel_rec_ws, rel_send_ws, mp_ws_2_mp_adj_idcs,
                 rel_rec_a2d, rel_send_a2d, mp_a2d_2_mp_adj_idcs):
        super(LSTMDecoder, self).__init__()
        n_hid = dyn_t2t["n_hidden"]
        do_prob = dyn_t2t["dropout_rate"]

        self.mp_HRN = dyn_t2t["mp_HRN"]
        self.mp_HRN_shared = dyn_t2t["mp_HRN_shared"]
        self.predict_delta = dyn_t2t["predict_delta"]
        self.input_delta = dyn_t2t["input_delta"]
        self.rollout_zeros = dyn_t2t["rollout_zeros"]
        self.rollout_zeros_in_train = dyn_t2t["rollout_zeros_in_train"]
        self.mp_input = dyn_t2t["mp_input"]

        n_et = n_edge_types
        if self.mp_HRN:
            if self.mp_HRN_shared:
                edge_fc1 = nn.ModuleList(
                    [nn.Linear(2 * n_hid, n_hid) for _ in range(n_et)])
                edge_fc2 = nn.ModuleList(
                    [nn.Linear(n_hid, n_hid) for _ in range(n_et)])
                self.l2a_edge_fc1 = edge_fc1
                self.l2a_edge_fc2 = edge_fc2
                self.ws_edge_fc1 = edge_fc1
                self.ws_edge_fc2 = edge_fc2
                self.a2d_edge_fc1 = edge_fc1
                self.a2d_edge_fc2 = edge_fc2
            else:
                self.l2a_edge_fc1 = nn.ModuleList(
                    [nn.Linear(2 * n_hid, n_hid) for _ in range(n_et)])
                self.l2a_edge_fc2 = nn.ModuleList(
                    [nn.Linear(n_hid, n_hid) for _ in range(n_et)])
                self.ws_edge_fc1 = nn.ModuleList(
                    [nn.Linear(2 * n_hid, n_hid) for _ in range(n_et)])
                self.ws_edge_fc2 = nn.ModuleList(
                    [nn.Linear(n_hid, n_hid) for _ in range(n_et)])
                self.a2d_edge_fc1 = nn.ModuleList(
                    [nn.Linear(2 * n_hid, n_hid) for _ in range(n_et)])
                self.a2d_edge_fc2 = nn.ModuleList(
                    [nn.Linear(n_hid, n_hid) for _ in range(n_et)])

        else:
            self.full_edge_fc1 = nn.ModuleList(
                [nn.Linear(2 * n_hid, n_hid) for _ in range(n_et)])
            self.full_edge_fc2 = nn.ModuleList(
                [nn.Linear(n_hid, n_hid) for _ in range(n_et)])
        self.msg_out_shape = n_hid
        if dyn_t2t["skip_first_edge_type"]:
            self.mp_dec_start_idx = 1
            self.mp_dec_norm = float(n_et) - 1.
        else:
            self.mp_dec_start_idx = 0
            self.mp_dec_norm = float(n_et)

        self.rr_full, self.rs_full = rel_rec_full, rel_send_full
        self.rr_l2a, self.rs_l2a = rel_rec_l2a, rel_send_l2a
        self.rr_ws, self.rs_ws = rel_rec_ws, rel_send_ws
        self.rr_a2d, self.rs_a2d = rel_rec_a2d, rel_send_a2d
        self.mp_l2a_2_mp_adj_idcs = mp_l2a_2_mp_adj_idcs
        self.mp_ws_2_mp_adj_idcs = mp_ws_2_mp_adj_idcs
        self.mp_a2d_2_mp_adj_idcs = mp_a2d_2_mp_adj_idcs

        n_layers = dyn_t2t['n_layers']
        assert (n_layers == 1)  # only this variant is supported
        self.rnn = nn.LSTM(n_in_node, n_hid, n_layers)
        self.fc_out_1 = nn.Linear(n_hid, n_hid)
        self.fc_out_2 = nn.Linear(n_hid, n_in_node)

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def upscale_rel(self, sp_rel, shp, map_idx):
        res = torch.zeros(shp, dtype=sp_rel.dtype, device=sp_rel.device)
        for i in range(len(map_idx)):
            res[map_idx[i]] = sp_rel[i]
        return res

    def upscale_rel_list(self, sp_rels, shp, map_idcs):
        return [self.upscale_rel(rr, shp , map_idx)
                for (rr, map_idx) in zip(sp_rels, map_idcs)]

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=-1)
        return edges

    def mp_round(self, hidden, rel_type, rel_rec, rel_send,
                 sp_mp_2_mp_adj_idcs, edge_fc1, edge_fc2):
        # sp_mp_2_mp_adj_idcs: sparse MP (e.g. L2A) adj mapping to full MP
        pre_msg = self.node2edge(hidden, rel_rec, rel_send)

        all_msgs = torch.zeros((pre_msg.size(0), pre_msg.size(1),
                                self.msg_out_shape),
                               device=hidden.device)

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        for i in range(self.mp_dec_start_idx, len(edge_fc2)):
            msg = torch.tanh(edge_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = torch.tanh(edge_fc2[i](msg))
            inferred_edges = rel_type[:, :, i:i + 1]
            if sp_mp_2_mp_adj_idcs is not None:
                inferred_edges = inferred_edges[:, sp_mp_2_mp_adj_idcs]
            msg = msg * inferred_edges
            all_msgs += msg / self.mp_dec_norm

        # Aggregate all msgs to receiver
        agg_msgs = self.edge2node(all_msgs, rel_rec, rel_send)
        return agg_msgs

    def mp_phase(self, mp_input, rel_type):

        if not self.mp_HRN:
            x = self.mp_round(mp_input, rel_type,
                              self.rr_full, self.rs_full, None,
                              self.full_edge_fc1, self.full_edge_fc2)
        else:
            # L2A
            x = mp_input
            for i in range(len(self.rr_l2a)):
                res = self.mp_round(x, rel_type,
                                   self.rr_l2a[i], self.rs_l2a[i],
                                   self.mp_l2a_2_mp_adj_idcs[i],
                                   self.l2a_edge_fc1, self.l2a_edge_fc2)
                x = x + res

            # WS
            res = self.mp_round(x, rel_type,
                               self.rr_ws, self.rs_ws,
                               self.mp_ws_2_mp_adj_idcs,
                               self.ws_edge_fc1, self.ws_edge_fc2)
            x = x + res

            # A2D
            for i in range(len(self.rr_a2d)):
                res = self.mp_round(x, rel_type,
                                   self.rr_a2d[i], self.rs_a2d[i],
                                   self.mp_a2d_2_mp_adj_idcs[i],
                                   self.a2d_edge_fc1, self.a2d_edge_fc2)
                x = x + res

        return x

    def single_step_forward(self, inputs, hidden, rel_type):

        if self.mp_input == 'hidden':
            hidden[0][0] = self.mp_phase(hidden[0][0], rel_type)
        if self.mp_input == 'cell':
            hidden[1][0] = self.mp_phase(hidden[1][0], rel_type)
        if self.mp_input == 'both':
            hidden[0][0] = self.mp_phase(hidden[0][0], rel_type)
            hidden[1][0] = self.mp_phase(hidden[1][0], rel_type)

        # LSTM step
        preds = inputs.unsqueeze(0)
        # Flatten and preds
        h_shp = hidden[0].shape
        hidden = (hidden[0].flatten(-3, -2), hidden[1].flatten(-3, -2))
        p_shp = preds.shape
        preds = preds.flatten(-3, -2)
        preds, hidden = self.rnn(preds, hidden)
        preds = preds[0, :, :]
        # Unflatten state
        hidden = (hidden[0].view(h_shp), hidden[1].view(h_shp))

        # Output MLP
        preds = F.relu(self.fc_out_1(preds), inplace=True)
        # [num_sims, n_atoms*n_out] or [num_sims, n_atoms*n_out]
        preds = self.fc_out_2(preds)

        # Unflatten preds
        preds = preds.view(p_shp[1:])
        # Predict position/velocity difference
        if self.predict_delta:
            preds = preds + inputs
        return preds, hidden

    def forward(self, data, rel_type,
                pred_steps=1,
                burn_in=False, burn_in_steps=1, dynamic_graph=False,
                encoder=None, temp=None):

        # inputs shape [B, T, K, num_dims]
        inputs = data.transpose(1, 2).contiguous()
        time_steps = inputs.size(1)

        # rel_type [B, 1 or T, K*(K-1), n_edge_types]
        if rel_type.size(1) == 1:
            # static graph case
            rt_shp = rel_type.size
            sizes = [rt_shp(0), inputs.size(1), rt_shp(-2), rt_shp(-1)]
            rel_type = rel_type.expand(sizes)

        zrs = torch.zeros((1, inputs.size(0), inputs.size(2), self.msg_out_shape),
                          device=inputs.device)
        hidden = (zrs, zrs)
        pred_all = []
        first_step = True
        rollout_zeros = self.rollout_zeros and \
                        (self.rollout_zeros_in_train or not self.training)
        for step in range(0, inputs.size(1) - 1):

            if burn_in:
                if step <= burn_in_steps or self.input_delta:
                    ins = inputs[:, step, :, :]
                    if self.input_delta and not first_step:
                        ins = ins - pred_all[step - 1]
                else:
                    ins = pred_all[step - 1]
            else:
                assert (pred_steps <= time_steps)
                # Use ground truth trajectory inputs vs. last prediction
                if not step % pred_steps or self.input_delta:
                    ins = inputs[:, step, :, :]
                    if self.input_delta:
                        if not first_step and not rollout_zeros:
                            ins = ins - pred_all[step - 1]
                        elif rollout_zeros:
                            ins = torch.zeros_like(ins, device=ins.device)
                else:
                    ins = pred_all[step - 1]

            if dynamic_graph and step >= burn_in_steps:
                # Note assumes burn_in_steps == args.timesteps
                logits = encoder(
                    data[:, :, step-burn_in_steps:step, :].contiguous(),
                    self.rr_full, self.rs_full
                )
                curr_rel_type = gumbel_softmax(logits, tau=temp, hard=True)
            else:
                curr_rel_type = rel_type[:, step, :, :]

            pred, hidden = self.single_step_forward(ins, hidden, curr_rel_type)
            pred_all.append(pred)
            first_step = False

        preds = torch.stack(pred_all, dim=1)

        return preds.transpose(1, 2).contiguous()

class LSTMBaseline(nn.Module):
    """LSTM model for joint trajectory prediction."""
    def __init__(self, n_atoms, n_dims, dyn_t2t):

        n_in = n_dims
        n_hid = dyn_t2t['n_hidden']
        n_out = n_dims
        n_layers = dyn_t2t['n_layers']
        self.independent_objs = dyn_t2t["independent_objs"]
        self.predict_delta = dyn_t2t["predict_delta"]
        self.input_delta = dyn_t2t["input_delta"]
        self.rollout_zeros = dyn_t2t["rollout_zeros"]
        self.rollout_zeros_in_train = dyn_t2t["rollout_zeros_in_train"]
        if self.independent_objs:
            K = 1
        else:
            K = n_atoms

        super(LSTMBaseline, self).__init__()

        n_in_lstm = n_in
        n_in_lstm *= K
        self.rnn = nn.LSTM(n_in_lstm, n_hid, n_layers)
        self.fc2_1 = nn.Linear(n_hid, n_hid)
        self.fc2_2 = nn.Linear(n_hid, K * n_out)

        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = dyn_t2t['dropout_rate']

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def step(self, ins, hidden=None):
        # Input shape: [num_sims, n_atoms, n_in]
        in_shp = ins.size()
        x = ins.contiguous()

        x_shp = x.size()
        if self.independent_objs:
            x = x.view(-1, x_shp[-1])
            # [num_sims*n_atoms, n_hid]
        else:
            x = x.view(x_shp[0], -1)
            # [num_sims, n_atoms*n_hid]

        x = x.unsqueeze(0)
        x, hidden = self.rnn(x, hidden)
        x = x[0, :, :]

        x = F.relu(self.fc2_1(x), inplace=True)
        x = self.fc2_2(x)
        # [num_sims, n_atoms*n_out] or [num_sims, n_atoms*n_out]

        x = x.view(in_shp[0], in_shp[1], -1)
        # [num_sims, n_atoms, n_out]

        # Predict position/velocity difference
        if self.predict_delta:
            x = x + ins

        return x, hidden

    def forward(self, inputs, prediction_steps, burn_in=False, burn_in_steps=1):

        # Input shape: [num_sims, num_things, num_timesteps, n_in]

        outputs = []
        hidden = None
        first_step = True
        rollout_zeros = self.rollout_zeros and \
                        (self.rollout_zeros_in_train or not self.training)
        for step in range(0, inputs.size(2) - 1):

            if burn_in:
                if step <= burn_in_steps or self.input_delta:
                    ins = inputs[:, :, step, :]
                    if self.input_delta and not first_step:
                        ins = ins - outputs[step - 1]
                else:
                    ins = outputs[step - 1]
            else:
                # Use ground truth trajectory input vs. last prediction
                if not step % prediction_steps or self.input_delta:
                    ins = inputs[:, :, step, :]
                    if self.input_delta:
                        if not first_step and not rollout_zeros:
                            ins = ins - outputs[step - 1]
                        elif rollout_zeros:
                            ins = torch.zeros_like(ins, device=ins.device)
                else:
                    ins = outputs[step - 1]

            output, hidden = self.step(ins, hidden)

            # Predict position/velocity difference
            outputs.append(output)

        outputs = torch.stack(outputs, dim=2)

        return outputs
