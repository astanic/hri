import torch
import torch.nn as nn
import torch.nn.functional as F

from modules_common import MLP


class NRIMLPEncoder(nn.Module):
    """MLP encoder module."""

    def __init__(self, seq_len, n_dims , n_edge_types, enc_t2r,
                 rel_rec_full, rel_send_full, rel_rec_l2a, rel_send_l2a,
                 rel_rec_ws, rel_send_ws, rel_rec_a2d, rel_send_a2d):
        super(NRIMLPEncoder, self).__init__()
        n_hid = enc_t2r["n_hidden"]
        n_out = n_edge_types
        do_prob = enc_t2r["dropout_rate"]
        self.dynamic = enc_t2r["dynamic"]
        self.horizon = enc_t2r["horizon"]
        self.mp_HRN = enc_t2r["mp_HRN"]
        self.mp_HRN_shared = enc_t2r["mp_HRN_shared"]
        self.seq_len = seq_len

        self.rr_full, self.rs_full = rel_rec_full, rel_send_full
        self.rr_l2a, self.rs_l2a = rel_rec_l2a, rel_send_l2a
        self.rr_ws, self.rs_ws = rel_rec_ws, rel_send_ws
        self.rr_a2d, self.rs_a2d = rel_rec_a2d, rel_send_a2d

        if self.dynamic:
            n_in = self.horizon * n_dims
        else:
            n_in = self.seq_len * n_dims

        self.mlp_node_embed = MLP(n_in, n_hid, n_hid, do_prob)

        if self.mp_HRN:
            if self.mp_HRN_shared:
                mlp_n2e = MLP(n_hid * 2, n_hid, n_hid, do_prob)
                mlp_e2n = MLP(n_hid, n_hid, n_hid, do_prob)
                self.l2a_mlp_n2e = mlp_n2e
                self.l2a_mlp_e2n = mlp_e2n
                self.ws_mlp_n2e = mlp_n2e
                self.ws_mlp_e2n = mlp_e2n
                self.a2d_mlp_n2e = mlp_n2e
                self.a2d_mlp_e2n = mlp_e2n
            else:
                self.l2a_mlp_n2e = MLP(n_hid * 2, n_hid, n_hid, do_prob)
                self.l2a_mlp_e2n = MLP(n_hid, n_hid, n_hid, do_prob)
                self.ws_mlp_n2e = MLP(n_hid * 2, n_hid, n_hid, do_prob)
                self.ws_mlp_e2n = MLP(n_hid, n_hid, n_hid, do_prob)
                self.a2d_mlp_n2e = MLP(n_hid * 2, n_hid, n_hid, do_prob)
                self.a2d_mlp_e2n = MLP(n_hid, n_hid, n_hid, do_prob)

        else:
            self.full_mlp_n2e = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            self.full_mlp_e2n = MLP(n_hid, n_hid, n_hid, do_prob)

        if self.mp_HRN:
            self.full_mlp_out = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        else:
            # there will be a skip connection
            self.full_mlp_out = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def stack_consecutive_tensors(self, input, n_stack=10, dim=2):
        """
        Stacks consecutive n_stack consecutive tensors along an axis.
        :param input: (128, 5, 49, 4) (B, n_atoms, T, n_dims)
        :param n_stack: int how many tensors to have in a stack
        :param dim: int along which to stack the tensors
        :return: image stack:(128, 5, 49, n_stack, 4) for axis=3
        """
        # pad n_stack zeros at the end of the "-4th" dimension
        if dim < 0:
            pos = -dim + 1
        else:
            pos = len(input.shape) - dim - 1
        input = F.pad(input, [0, 0] * pos + [n_stack, 0])
        # split tensors and tile
        input = input.unsqueeze(dim)
        n_tile = [1 for _ in range(input.ndim)]
        n_tile[dim] = n_stack
        input = input.repeat(n_tile)
        slices = []
        for i in range(n_stack):
            s = torch.index_select(input, dim, index=torch.tensor(i).cuda())
            s = torch.roll(s, shifts=-i, dims=dim)
            slices.append(s)
        stack = torch.cat(slices, dim=dim)
        # remove the dummy padded parts
        stack = stack[:, :, :, :-n_stack, :]
        return stack

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def mp_round(self, x, rel_rec, rel_send, mlp1, mlp2):
        x = self.node2edge(x, rel_rec, rel_send)
        x = mlp1(x)
        x = self.edge2node(x, rel_rec, rel_send)
        x = mlp2(x)
        return x


    def forward(self, inputs):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        if self.dynamic:
            # [128, 5, 49, 4]
            inputs = self.stack_consecutive_tensors(inputs, self.horizon)
            # [128, 5, 10, 49, 4]
            inputs = inputs.transpose(2, 3).transpose(1, 2).contiguous()
            # [128, 49, 5, 10, 4]
            inputs = inputs.view([-1] + list(inputs.shape[2:]))
            # [128*49, 5, 10, 4]

        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        # Initial node embeddings
        x = self.mlp_node_embed(x)  # 2-layer ELU net per node

        if not self.mp_HRN:
            x = self.node2edge(x, self.rr_full, self.rs_full)
            x = self.full_mlp_n2e(x)
            x_skip = x
            x = self.edge2node(x, self.rr_full, self.rs_full)
            x = self.full_mlp_e2n(x)

            x = self.node2edge(x, self.rr_full, self.rs_full)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.full_mlp_out(x)
        else:
            # L2A
            for i in range(len(self.rr_l2a)):
                x += self.mp_round(x, self.rr_l2a[i], self.rs_l2a[i],
                                   self.l2a_mlp_n2e, self.l2a_mlp_e2n)
            # WS
            x += self.mp_round(x, self.rr_ws, self.rs_ws,
                               self.ws_mlp_n2e, self.ws_mlp_e2n)
            # A2D
            for i in range(len(self.rr_a2d)):
                x += self.mp_round(x, self.rr_a2d[i], self.rs_a2d[i],
                                   self.a2d_mlp_n2e, self.a2d_mlp_e2n)

            x = self.node2edge(x, self.rr_full, self.rs_full)
            x = self.full_mlp_out(x)

        x = self.fc_out(x)
        return x
