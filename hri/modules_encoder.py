import torch
import torch.nn as nn
import torch.nn.functional as F

from modules_common import NonLinear, MLP, ConvBNBlock
from modules_common import vgg_layer, dcgan_conv, _init_layer

###############################################################################
###############################################################################
# Image modules


class BaseEncoder(nn.Module):
    def __init__(self, n_dims, img_size, enc_i2t):
        super(BaseEncoder, self).__init__()

        self.is_enc_slot_conv = (enc_i2t["i2t_type"] == 'slot_conv')

        self.nf_vgg = enc_i2t["nf_vgg"]
        self.nf_dcgan = enc_i2t["nf_dcgan"]
        oc = enc_i2t["output_channels"]
        ks = enc_i2t["kernel_shape"]
        st = enc_i2t["stride"]
        nm = enc_i2t["normalization_module"]
        nl = enc_i2t["non_linearity"]
        self.n_stack = enc_i2t["n_stack"]
        self.append_xy = enc_i2t["append_xy_mesh"]
        self.out_map = enc_i2t["out_map"]
        self.out_map_shared = enc_i2t["out_map_shared"]
        self.out_map_nonl = enc_i2t["out_map_non_linearity"]
        self.out_map_bn = enc_i2t["out_map_bn"]
        self.use_out_map = (self.out_map != 'none')
        self.img_size = img_size

        self.n_dims = n_dims  # 4-dim vector (x, y, v_x, v_y)

        self.oc = [c for c in oc]
        # Calculate n output channels; if some ks == 0, we use only max pool
        # => num output channels does not change, so oc[i+1] = oc[i]
        assert len(ks) == len(oc)
        for i in range(len(self.oc) - 1):
            if ks[i + 1] == 0:
                self.oc[i + 1] = self.oc[i]

        n_out = self.n_dims
        if not self.use_out_map:
            self.oc[2] = n_out
            self.oc[3] = n_out
            self.oc[4] = n_out

        self.ic = self.oc[:-1]
        ic0 = img_size[-1] * self.n_stack
        ic0 += 2 if self.append_xy else 0
        self.ic = [ic0] + self.ic

        self.skip_n_channels = self.oc[:-2] if img_size == 32 else self.oc[:-1]

        self.create_xy_mesh(img_size[:2])
        self.create_layers(self.ic, ks, nm, nl, st, n_out)

    def create_xy_mesh(self, xy_mesh_size):
        xs, ys = xy_mesh_size
        x = torch.linspace(-1, 1, xs).cuda()
        y = torch.linspace(-1, 1, ys).cuda()
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as a constant, with extra dims for N and C
        with torch.no_grad():
            self.x_grid = x_grid.view((1, 1) + x_grid.shape)
            self.y_grid = y_grid.view((1, 1) + y_grid.shape)

    def append_xy_mesh(self, input):
        bs = input.size()[0]
        input = torch.cat((self.x_grid.expand(bs, -1, -1, -1),
                           self.y_grid.expand(bs, -1, -1, -1), input), dim=-3)
        return input

    def order_lowest_level(self, x16):
        # order the last level of the 21-node (1+4+16) adjacency matrix
        x = []
        for i in range(2):
            for j in range(2):
                xi = x16[:, :, i * 2:i * 2 + 2, j * 2:j * 2 + 2]
                xi = xi.contiguous().view(xi.size(0), xi.size(1), -1)
                x.append(xi)

        x = torch.cat(x, dim=-1)
        return x

    def stack_consecutive_frames(self, images, n_stack=3, dim=5):
        """
        Stacks consecutive n_stack consecutive frames along axis.
        :param images: (32, 49, 3, 64, 64)
        :param dim: int along which to stack the frames
        :return: image stack:(32, 49, 3, n_stack, 64, 64) for axis=5
        """
        # append n_stack frames at the end of the "dim" axis
        if dim < 0:
            pos = -dim - 1
        else:
            pos = len(images.shape) - dim - 1
        images = F.pad(images, [0, 0] * pos + [n_stack, 0])
        # split the frames and tile
        images = images.unsqueeze(dim)
        n_tile = [1 for _ in range(images.ndim)]
        n_tile[dim] = n_stack
        images = images.repeat(n_tile)
        slices = []
        for i in range(n_stack):
            s = torch.index_select(images, dim, index=torch.tensor(i).cuda())
            # s = s.unsqueeze(dim)
            s = torch.roll(s, shifts=-i, dims=dim)
            slices.append(s)
        stack = torch.cat(slices, dim=dim)
        # remove the dummy padded parts
        stack = stack[:, :-n_stack, :, :, :, :]
        return stack

    def forward(self, x):
        return NotImplementedError


class I2TEncConv(BaseEncoder):
    def __init__(self, n_dims, img_size, enc_i2t):
        super(I2TEncConv, self).__init__(n_dims, img_size, enc_i2t)

    def create_conv_layers(self, ic, ks, nm, nl, st):
        # Add extra conv layer (after c0) in case input is size 64
        self.c64 = None
        if self.img_size[0] == 64:
            self.c64 = ConvBNBlock(ic[0], self.oc[0], ks[0], [2, 2], 'SAME', nm, nl)
            ic[0] = self.oc[0]
        elif self.img_size[0] != 32:
            raise NotImplementedError('CNN expects 32 or 64 input size, got',
                                      self.img_size[0])
        if self.is_enc_slot_conv:
            # ks[0] can be 8, 10, 12 or 14
            self.c0 = ConvBNBlock(ic[0], self.oc[0], ks[0], [1, 1], 'SAME', nm,
                                  nl, stride=8)
            self.c1 = None
            self.c2 = None
        else:
            self.c0 = ConvBNBlock(ic[0], self.oc[0], ks[0], [st[0], st[0]], 'SAME', nm, nl)
            # 16 x 16
            self.c1 = ConvBNBlock(ic[1], self.oc[1], ks[1], [st[1], st[1]], 'SAME', nm, nl)
            # 8 x 8
            self.c2 = ConvBNBlock(ic[2], self.oc[2], ks[2], [st[2], st[2]], 'SAME', nm, nl)
            # 4 x 4

    def create_layers(self, ic, ks, nm, nl, st, n_out):
        self.create_conv_layers(ic, ks, nm, nl, st)
        equal_oc = (self.oc[2] == self.oc[3]) and (self.oc[2] == self.oc[4])

        if self.out_map_shared and equal_oc:
            if self.out_map == 'nonlinear':
                self.fc2 = NonLinear(self.oc[2], n_out, self.out_map_nonl)
            elif self.out_map == 'mlp':
                self.fc2 = MLP(self.oc[2], self.oc[2], n_out,
                               out_nl=self.out_map_nonl, out_bn=self.out_map_bn)
        else:
            if self.out_map == 'nonlinear':
                self.fc2 = NonLinear(self.oc[2], n_out, self.out_map_nonl)
            elif self.out_map == 'mlp':
                self.fc2 = MLP(self.oc[2], self.oc[2], n_out,
                               self.out_map_nonl, out_bn=self.out_map_bn)

    def stack_frames_and_append_xy_mesh(self, images):
        # Input images shape: [num_sims, num_timesteps, 64, 64, 3]
        # stack multiple frames along the channel dimension
        if self.n_stack > 1:
            stack = self.stack_consecutive_frames(images, self.n_stack, dim=-4)
            ss = list(stack.size())
            images = stack.view(ss[:2] + [ss[2] * ss[3]] + ss[4:])

        ishp = list(images.size())
        # merge batch and time dimensions
        x = images.contiguous().view([-1] + ishp[2:])
        if self.append_xy:
            x = self.append_xy_mesh(x)
        return x, ishp

    def forward(self, images):

        x, ishp = self.stack_frames_and_append_xy_mesh(images)

        enc_skip = []
        if self.c64 is not None:
            # case where input image size is 64
            x = self.c64(x)
            enc_skip.append(x.view(ishp[0], ishp[1], x.size(-3), x.size(-2), x.size(-1)))

        x = self.c0(x)
        # Here you get 4x4 representations for Slot Conv Enc
        enc_skip.append(x.view(ishp[0], ishp[1], x.size(-3), x.size(-2), x.size(-1)))

        if not self.is_enc_slot_conv:
            x = self.c1(x)
            enc_skip.append(x.view(ishp[0], ishp[1], x.size(-3), x.size(-2), x.size(-1)))

            x = self.c2(x)
            # Here you get 4x4 representations for Standard Conv Enc
            enc_skip.append(x.view(ishp[0], ishp[1], x.size(-3), x.size(-2), x.size(-1)))

        x16 = x
        x16ord = self.order_lowest_level(x16)
        x16ord = x16ord.transpose(-2, -1)
        if self.use_out_map:
            x16ord = self.fc2(x16ord)

        return x16, x16ord, enc_skip


class I2TEncVGG(I2TEncConv):
    def __init__(self, n_dims, img_size, enc_i2t):
        super(I2TEncVGG, self).__init__(n_dims, img_size, enc_i2t)
        nf = self.nf_vgg
        self.skip_n_channels = [nf, nf*2, nf*4, self.oc[2]]
        if self.img_size[0] == 64:
            self.skip_n_channels = [self.ic[0]] + self.skip_n_channels

    def create_conv_layers(self, ic, ks, nm, nl, st):
        # 32 x 32 / 64 x 64
        self.c64 = None
        nf = self.nf_vgg
        if self.img_size[0] == 64:
            self.c64 = nn.Sequential(
                vgg_layer(self.ic[0], self.ic[0]),
                vgg_layer(self.ic[0], self.ic[0]),
            )
        self.c0 = nn.Sequential(
            vgg_layer(self.ic[0], nf),
            vgg_layer(nf, nf),
        )
        self.c1 = nn.Sequential(
            vgg_layer(nf, nf*2),
            vgg_layer(nf*2, nf*2),
            vgg_layer(nf*2, nf*2),
        )
        #  / # 4 x 4 - depending on the input image resolution
        self.c2 = nn.Sequential(
            vgg_layer(nf*2, nf*4),
            vgg_layer(nf*4, nf*4),
            vgg_layer(nf*4, nf*4),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(nf*4, self.oc[2], 3, 1, 1),
            nn.BatchNorm2d(self.oc[2]),
        )

        for module in self.c3:
            _init_layer(module)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, images):
        x, ishp = self.stack_frames_and_append_xy_mesh(images)

        enc_skip = []
        if self.c64 is not None:
            # case where input image size is 64
            x = self.c64(x)
            enc_skip.append(x.view(ishp[0], ishp[1], x.size(-3), x.size(-2), x.size(-1)))
            x = self.mp(x)  # 64 -> 32

        x = self.c0(x)  # 32
        enc_skip.append(x.view(ishp[0], ishp[1], x.size(-3), x.size(-2), x.size(-1)))

        x = self.mp(x)  # 32 -> 16
        x = self.c1(x)  # 16
        enc_skip.append(x.view(ishp[0], ishp[1], x.size(-3), x.size(-2), x.size(-1)))

        x = self.mp(x)  # 16 -> 8
        x = self.c2(x)  # 8
        enc_skip.append(x.view(ishp[0], ishp[1], x.size(-3), x.size(-2), x.size(-1)))

        x = self.mp(x)  # 8 -> 4
        x = self.c3(x)  # 4
        enc_skip.append(x.view(ishp[0], ishp[1], x.size(-3), x.size(-2), x.size(-1)))

        x16 = x
        x16ord = self.order_lowest_level(x16)
        x16ord = x16ord.transpose(-2, -1)
        if self.use_out_map:
            x16ord = self.fc2(x16ord)

        return x16, x16ord, enc_skip


class I2TEncDCGAN(I2TEncConv):
    def __init__(self, n_dims, img_size, enc_i2t):
        super(I2TEncDCGAN, self).__init__(n_dims, img_size, enc_i2t)
        nf = self.nf_dcgan
        self.skip_n_channels = [nf*2, nf*4, self.oc[2]]
        if self.img_size[0] == 64:
            self.skip_n_channels = [self.ic[0]] + self.skip_n_channels

    def create_conv_layers(self, ic, ks, nm, nl, st):
        # 32 x 32 / 64 x 64
        self.c64 = None
        nf = self.nf_dcgan
        if self.img_size[0] == 64:
            self.c64 = dcgan_conv(self.ic[0], self.ic[0])
        # 16 x 16
        self.c0 = dcgan_conv(ic[0], nf * 2)
        # 8 x 8
        self.c1 = dcgan_conv(nf * 2, nf * 4)
        #  / # 4 x 4 - depending on the input image resolution
        self.c2 = dcgan_conv(nf * 4, self.oc[2])


###############################################################################
###############################################################################
# T2H modules


class T2HEncConv(BaseEncoder):
    def __init__(self, n_dims, img_size, enc_i2t):
        super(T2HEncConv, self).__init__(n_dims, img_size, enc_i2t)

    def create_conv_layers(self, ic, ks, nm, nl, st):
        self.c3 = ConvBNBlock(ic[3], self.oc[3], ks[3], [st[3], st[3]], 'SAME', nm, nl)
        self.c4 = ConvBNBlock(ic[4], self.oc[4], ks[4], [st[4], st[4]], 'SAME', nm, nl)

    def create_layers(self, ic, ks, nm, nl, st, n_out):
        self.create_conv_layers(ic, ks, nm, nl, st)

        equal_oc = (self.oc[2] == self.oc[3]) and (self.oc[2] == self.oc[4])

        if self.out_map_shared and equal_oc:
            if self.out_map == 'nonlinear':
                self.fc3 = NonLinear(self.oc[2], n_out, self.out_map_nonl)
                self.fc4 = self.fc3
            elif self.out_map == 'mlp':
                self.fc3 = MLP(self.oc[2], self.oc[2], n_out,
                               out_nl=self.out_map_nonl, out_bn=self.out_map_bn)
                self.fc4 = self.fc3
        else:
            if self.out_map == 'nonlinear':
                self.fc3 = NonLinear(self.oc[3], n_out, self.out_map_nonl)
                self.fc4 = NonLinear(self.oc[4], n_out, self.out_map_nonl)
            elif self.out_map == 'mlp':
                self.fc3 = MLP(self.oc[3], self.oc[3], n_out,
                               self.out_map_nonl, out_bn=self.out_map_bn)
                self.fc4 = MLP(self.oc[4], self.oc[4], n_out,
                               self.out_map_nonl, out_bn=self.out_map_bn)

    def forward(self, images, x16, x16ord):
        ishp = list(images.size())
        x4 = self.c3(x16)
        x1 = self.c4(x4)

        x4 = x4.view(x4.size(0), x4.size(1), -1)
        x1 = x1.view(x1.size(0), x1.size(1), -1)

        x4 = x4.transpose(-2, -1)
        x1 = x1.transpose(-2, -1)

        if self.use_out_map:
            x4 = self.fc3(x4)
            x1 = self.fc4(x1)

        x = torch.cat([x1, x4, x16ord], dim=-2)
        x = x.view(ishp[0], ishp[1], x.size(-2), x.size(-1))
        # For NRI we need [num_sims, n_atoms, num_timesteps, num_dims]
        x = x.transpose(1, 2).contiguous()
        return x


class T2HEncVGG(T2HEncConv):
    def __init__(self, n_dims, img_size, enc_i2t):
        super(T2HEncVGG, self).__init__(n_dims, img_size, enc_i2t)

    def create_conv_layers(self, ic, ks, nm, nl, st):
        self.c3 = nn.Sequential(
            vgg_layer(ic[3], ic[3]),
            vgg_layer(ic[3], self.oc[3]),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            )
        self.c4 = nn.Sequential(
            vgg_layer(ic[4], ic[4]),
            vgg_layer(ic[4], self.oc[4]),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )


class T2HEncDCGAN(T2HEncConv):
    def __init__(self, n_dims, img_size, enc_i2t):
        super(T2HEncDCGAN, self).__init__(n_dims, img_size, enc_i2t)

    def create_conv_layers(self, ic, ks, nm, nl, st):
        self.c3 = dcgan_conv(ic[3], self.oc[3])
        self.c4 = dcgan_conv(ic[4], self.oc[4])
