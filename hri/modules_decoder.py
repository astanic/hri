import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AOverB

from modules_common import NonLinear, MLP, Conv2dTF
from modules_common import vgg_layer, dcgan_upconv, _init_layer


###############################################################################
###############################################################################
###### Image Decoder Modules

class BaseDecoder(nn.Module):
    """
    Takes INDIVIDUAL trajectories as input, decodes each into an image,
    and sums them up for the final image
    """
    def __init__(self, n_dims, img_size, dec_t2i, skip_n_channels):
        super(BaseDecoder, self).__init__()

        self.use_output_sigmoid = dec_t2i["use_output_sigmoid"]

        self.compositing = dec_t2i["compositing"]
        self.compositing_detach = dec_t2i["compositing_detach"]
        self.compositing_eps = dec_t2i["compositing_eps"]

        self.in_map_type = dec_t2i["in_map"]
        self.in_map_nonl = dec_t2i["in_map_non_linearity"]
        self.in_map_bn = dec_t2i["in_map_bn"]
        self.use_in_map = any(self.in_map_type in s for s in ['nonlinear', 'mlp'])

        self.img_size = img_size
        self.broadcast_input = dec_t2i["broadcast_input"]
        self.use_skip = dec_t2i["use_skip"]
        self.skip_code = dec_t2i["skip_code"]
        self.skip_n_channels = skip_n_channels
        for i in range(-1, -len(self.skip_n_channels)-1, -1):
            if not self.skip_code[i] == "Y":
                self.skip_n_channels[i] = 0

        self.do_append_object_idcs = dec_t2i["append_object_idcs"]
        self.n_dims = n_dims
        self.in_features = n_dims + 1 if self.do_append_object_idcs else n_dims

        self.nf_vgg = dec_t2i["nf_vgg"]
        self.nf_dcgan = dec_t2i["nf_dcgan"]
        self.nf_sb = dec_t2i["nf_sb"]
        self.in_channels = dec_t2i["in_channels"].copy()
        self.in_channels[0] = self.in_features
        self.ks = dec_t2i["kernel_shape"].copy()
        self.conv_stride = dec_t2i["conv_stride"].copy()
        assert len(self.in_channels) == len(self.ks)
        assert len(self.in_channels) == len(self.conv_stride)

        if img_size[0] == 64:
            self.in_channels.append(self.in_channels[-1])
            self.ks.append(self.ks[-1])
            self.conv_stride.append(self.conv_stride[-1])

        self.is_dcgan = False  # needed not to break the pretrained models

        self.calculate_input_output_n_channels()
        self.create_layers()

    def calculate_input_output_n_channels(self):
        raise NotImplementedError

    def init_weights(self):
        # Convs
        for module in self.dec_convs:
            n = module.kernel_size[0] * module.in_channels
            module.weight.data.normal_(0, np.sqrt(2. / n))
            module.bias.data.fill_(0.1)
        n = self.last_conv.kernel_size[0] * self.last_conv.out_channels
        self.last_conv.weight.data.normal_(0, np.sqrt(2. / n))
        self.last_conv.bias.data.fill_(0.1)

    def create_layers(self):
        self.in_map = None
        if self.in_map_type == 'nonlinear':
            self.in_map = NonLinear(
                self.in_features, self.input_map_n_out, self.in_map_nonl)
        elif self.in_map_type == 'mlp':
            self.in_map = MLP(
                self.in_features, self.in_features, self.input_map_n_out,
                out_nl=self.in_map_nonl, out_bn=self.in_map_bn)

        self.create_conv_layers()

    def create_conv_layers(self):
        dec_convs = []
        for i in range(len(self.in_channels) - 1):
            if self.conv_stride[i] == 2:
                dec_convs.append(
                    nn.ConvTranspose2d(in_channels=self.in_channels[i] + self.skip_n_channels[-i-1],
                                      out_channels=self.in_channels[i + 1],
                                      kernel_size=4,
                                      stride=self.conv_stride[i],
                                      padding=1)
                )
            elif self.conv_stride[i] == 1:
                dec_convs.append(nn.Upsample(scale_factor=2,
                                             mode='bilinear',
                                             align_corners=True))
                dec_convs.append(Conv2dTF(self.in_channels[i] + self.skip_n_channels[-i-1],
                                          self.in_channels[i+1],
                                          self.ks[i],
                                          [1, 1],
                                          padding="SAME"))
            else:
                raise ValueError('Conv stride needs to be in 1 or 2. '
                                 '{} not supported.'.format(self.conv_stride[i]))
        self.dec_convs = nn.ModuleList(dec_convs)

        out_channels = self.img_size[-1]
        if self.conv_stride[-1] == 2:
            self.last_conv = nn.ConvTranspose2d(
                in_channels=self.in_channels[-1] + self.skip_n_channels[0],
                out_channels=out_channels,
                kernel_size=4,
                stride=self.conv_stride[-1],
                padding=1)
            scale_factor = 1
        elif self.conv_stride[-1] == 1:
            self.last_conv = Conv2dTF(self.in_channels[-1] +  + self.skip_n_channels[0],
                                      out_channels,
                                      self.ks[-1],
                                      [1, 1],
                                      padding="SAME")
            scale_factor = 2
        else:
            raise ValueError('Conv stride needs to be in 1 or 2. '
                             '{} not supported.'.format(self.conv_stride[-1]))
        self.last_upsample = None
        if scale_factor > 1:
            self.last_upsample = nn.Upsample(scale_factor=scale_factor,
                                             mode='bilinear',
                                             align_corners=True)

    def append_object_idcs(self, x):
        # append a label to each object - help the slot decoder to learn colors
        if self.do_append_object_idcs:
            idcs_array = torch.arange(1, x.size(1) + 1,
                                      dtype=torch.float32, device=x.device)

            idcs_array = idcs_array / idcs_array.size(0)
            idcs_array = idcs_array.view(1, x.size(1), 1, 1)
            idcs_array = idcs_array.expand([x.size(0), -1, x.size(2), -1])
            x = torch.cat((x, idcs_array), dim=-1)
        return x

    def broadcast_op(self, x, size):
        x = x.view(x.shape + (1, 1))
        # Tile across to match (e.g. image size)
        # Shape: NxDx64x64
        x = x.expand(-1, -1, size, size)
        return x

    def prepare_input_for_conv(self, x):
        raise NotImplementedError

    def post_process_output_and_objs(self, x, x_in_shp):
        raise NotImplementedError

    def forward(self, x, enc_skip, decode_masked_slots):
        """
        :param x: trajectories (num_sims, num_atoms, num_timesteps, num_dims)
        :return:
        """
        x_in_shp = list(x.size())

        x = self.append_object_idcs(x)
        x = self.prepare_input_for_conv(x)

        skip_idx = -1
        for module in self.dec_convs:
            if self.use_skip and self.skip_code[skip_idx] == "Y":
                es = enc_skip[skip_idx]
                es = es.contiguous().view([-1] + list(es.shape[2:]))
                x = torch.cat([x, es], -3)
                skip_idx -= 1
            if self.is_dcgan:
                x = module(x)
            else:
                x = F.relu(module(x), inplace=True)

        if self.use_skip and self.skip_code[skip_idx] == "Y":
            es = enc_skip[skip_idx]
            es = es.contiguous().view([-1] + list(es.shape[2:]))
            x = torch.cat([x, es], -3)
        if self.last_upsample:
            x = self.last_upsample(x)
        x = self.last_conv(x)
        if self.use_output_sigmoid:
            x = torch.sigmoid(x)

        x, x_objs = self.post_process_output_and_objs(x, x_in_shp)

        return x, x_objs


class SlotConvDecoder(BaseDecoder):
    """
    Takes INDIVIDUAL trajectories as input, decodes each into an image,
    and sums them up for the final image
    """
    def __init__(self, n_dims, img_size, dec_t2i, skip_n_channels):
        super(SlotConvDecoder, self).__init__(n_dims, img_size, dec_t2i, skip_n_channels)

    def calculate_input_output_n_channels(self):
        self.dec_start_res = 2
        self.input_map_n_out = self.in_channels[0] * (self.dec_start_res ** 2)
        if self.use_in_map:
            factor = self.dec_start_res ** 2 if self.broadcast_input else 1
            self.in_channels[0] *= factor

    def prepare_input_for_conv(self, x):
        # flatten the B, K, T dimensions
        x = x.view(-1, x.size(-1))
        bkt = x.shape[0]
        if self.in_map:
            x = self.in_map(x)
        if self.broadcast_input:
            x = self.broadcast_op(x, self.dec_start_res)
        else:
            assert (x.shape[-1] % self.dec_start_res) % self.dec_start_res == 0
            x = x.view(bkt, -1, self.dec_start_res, self.dec_start_res)

        return x

    def post_process_output_and_objs(self, x, x_in_shp):
        # now split the images into individual objects and sum them up
        x_objs = x.view(x_in_shp[0], x_in_shp[1], x_in_shp[2],
                        x.size(-3), x.size(-2), x.size(-1))
        if self.compositing == 'sum':
            # do the sum-compositing along dim=1 into the final decoded image
            x = torch.sum(x_objs, dim=1).squeeze(1)
        elif self.compositing == 'implicit_alpha':
            decoded_x = torch.zeros_like(x_objs[:, 0, ...])
            decoded_a = torch.ones_like(x_objs[:, 0, :, :1, ...])
            for i in range(x_objs.size(1)):
                x_i = x_objs[:, i, ...]
                a_i = (x_i > self.compositing_eps).float()
                a_i, _ = torch.max(a_i, dim=2, keepdim=True)
                if self.compositing_detach:
                    a_i = a_i.detach()

                decoded_x, decoded_a = AOverB(x_i, a_i, decoded_x, decoded_a)
            x = decoded_x * decoded_a
        elif self.compositing == 'alpha':
            x_c, x_a = AOverB(
                x_objs[:, 0, :, :3, :, :], x_objs[:, 0, :, -1:, :, :],
                x_objs[:, 1, :, :3, :, :], x_objs[:, 1, :, -1:, :, :])
            for i in range(2, x_objs.size(1)):
                x_c, x_a = AOverB(x_c,
                                  x_a,
                                  x_objs[:, i, :, :3, :, :],
                                  x_objs[:, i, :, -1:, :, :])
            # remove alpha channel
            x = x_c
            x_objs = x_objs[:, :, :, :-1, :, :]
        else:
            raise NotImplementedError(
                'Compositing must be either "sum" or "alpha"]')

        # create objs img for the final plot
        # pad with white space for borders
        p2d = (1, 1, 1, 1)  # pad last dim by (1, 1) and 2nd to last by (1, 1)
        x_objs = F.pad(x_objs, p2d, 'constant', 1)

        x_objs = x_objs.transpose(1,2).transpose(2,3).transpose(3,4)
        x_objs = x_objs.unsqueeze(4)
        xos = list(x_objs.shape)
        xos[4] = 4
        xos[5] = -1
        x_objs = x_objs.view(xos)
        x_objs = x_objs.transpose(-2, -3)
        x_objs = x_objs.transpose(-3, -4)
        x_objs = x_objs.flatten(-2, -1).flatten(-3, -2)
        return x, x_objs


class ParConvDecoder(BaseDecoder):
    """
    Takes 16 (object) slots and decodes them all in parallel.
    """
    def __init__(self, n_dims, img_size, dec_t2i, skip_n_channels):
        super(ParConvDecoder, self).__init__(n_dims, img_size, dec_t2i, skip_n_channels)

    def calculate_input_output_n_channels(self):
        # Take just first 3(4) layer configs - up-convolve to 32(64) img size
        # Here we start from 4x4 tensor and do 3(4) upsampling operations
        self.dec_start_res = 4
        self.in_channels = self.in_channels[:-1]
        self.ks = self.ks[-1]
        self.conv_stride = self.conv_stride[:-1]
        self.input_map_n_out = self.in_channels[0]
        factor = self.dec_start_res ** 2 if self.broadcast_input else 1
        self.in_channels[0] *= factor

    def prepare_input_for_conv(self, x):
        # flatten the B, K, T dimensions
        xshp = list(x.size())
        x = x.view(-1, x.size(-1))
        if self.in_map:
            x = self.in_map(x)
        x = x.view(xshp[:-1] + [x.shape[-1]])

        x = x.transpose(1, 2)
        x = x.transpose(2, 3)
        if self.broadcast_input:
            x = x.contiguous()
            x = x.view(-1, x.size(-2) * x.size(-1))
            x = self.broadcast_op(x, self.dec_start_res)
        else:
            assert x.size(-1) == 16
            x = x.view(x.size(0), x.size(1), x.size(2),
                       self.dec_start_res, self.dec_start_res)
            x = x.contiguous().view(-1, x.size(2),
                                    self.dec_start_res, self.dec_start_res)
        return x

    def post_process_output_and_objs(self, x, x_in_shp):
        x = x.view(x_in_shp[0], x_in_shp[2], x.size(-3), x.size(-2), x.size(-1))
        return x, None


class SlotVGGDecoder(SlotConvDecoder):
    def __init__(self, n_dims, img_size, dec_t2i, skip_n_channels):
        super(SlotVGGDecoder, self).__init__(n_dims, img_size, dec_t2i, skip_n_channels)

    def calculate_input_output_n_channels(self):
        self.dec_start_res = 4
        self.input_map_n_out = self.in_channels[0] * (self.dec_start_res ** 2)
        if self.use_in_map:
            factor = self.dec_start_res ** 2 if self.broadcast_input else 1
            self.in_channels[0] *= factor

    def create_conv_layers(self):
        dec_convs = []

        nf = self.nf_vgg
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels[0] + self.skip_n_channels[-1], nf*4, 3, 1, 1),
                nn.BatchNorm2d(nf * 4),
                nn.LeakyReLU(0.2, inplace=True)
                )

        dec_convs.append(
            nn.Sequential(
                vgg_layer(nf*4 + self.skip_n_channels[-2], nf*4),
                vgg_layer(nf*4, nf*4),
                vgg_layer(nf*4, nf*2),
            )
        )  # 8 x 8

        dec_convs.append(
            nn.Sequential(
                vgg_layer(nf*2 + self.skip_n_channels[-3], nf*2),
                vgg_layer(nf*2, nf*2),
                vgg_layer(nf*2, nf*2),
            )
        )  # 16 x 16

        if self.img_size[0] == 64:
            dec_convs.append(
                nn.Sequential(
                    vgg_layer(nf*2 + self.skip_n_channels[-4], nf*2),
                    vgg_layer(nf*2, nf),
                )
            )  # 32 x 32
            self.dec_convs = nn.ModuleList(dec_convs)

            out_channels = self.img_size[-1]
            self.last_conv = nn.Sequential(
                vgg_layer(nf + self.skip_n_channels[0], nf),
                nn.ConvTranspose2d(nf, out_channels, 3, 1, 1),
            )  # 64 x 64
        elif self.img_size[0] == 32:
            self.dec_convs = nn.ModuleList(dec_convs)

            out_channels = self.img_size[-1]
            self.last_conv = nn.Sequential(
                vgg_layer(nf*2 + self.skip_n_channels[0], nf),
                nn.ConvTranspose2d(nf, out_channels, 3, 1, 1),
            )  # 32 x 32
        else:
            raise NotImplementedError
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

        for module in self.upc1:
            _init_layer(module)
        for module in self.last_conv:
            _init_layer(module)

    def forward(self, x, enc_skip, decode_masked_slots):
        """
        :param x: trajectories (num_sims, num_atoms, num_timesteps, num_dims)
        :return:
        """
        x_in_shp = list(x.size())

        x = self.append_object_idcs(x)
        x = self.prepare_input_for_conv(x)

        skip_idx = -1
        if self.use_skip and self.skip_code[skip_idx] == "Y":
            es = enc_skip[skip_idx]
            es = es.contiguous().view([-1] + list(es.shape[2:]))
            x = torch.cat([x, es], -3)
            skip_idx -= 1
        x = self.upc1(x)

        for module in self.dec_convs:
            x = self.up(x)
            if self.use_skip and self.skip_code[skip_idx] == "Y":
                es = enc_skip[skip_idx]
                es = es.contiguous().view([-1] + list(es.shape[2:]))
                x = torch.cat([x, es], -3)
                skip_idx -= 1
            x = module(x)

        x = self.up(x)
        if self.use_skip and self.skip_code[skip_idx] == "Y":
            es = enc_skip[skip_idx]
            es = es.contiguous().view([-1] + list(es.shape[2:]))
            x = torch.cat([x, es], -3)
        x = self.last_conv(x) # 32
        if self.use_output_sigmoid:
            x = torch.sigmoid(x)

        x, x_objs = self.post_process_output_and_objs(x, x_in_shp)

        return x, x_objs


class ParVGGDecoder(ParConvDecoder):
    def __init__(self, n_dims, img_size, dec_t2i, skip_n_channels):
        super(ParVGGDecoder, self).__init__(n_dims, img_size, dec_t2i, skip_n_channels)

    def create_conv_layers(self):
        dec_convs = []

        nf = self.nf_vgg
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels[0] + self.skip_n_channels[-1], nf*4, 3, 1, 1),
                nn.BatchNorm2d(nf * 4),
                nn.LeakyReLU(0.2, inplace=True)
                )

        dec_convs.append(
            nn.Sequential(
                vgg_layer(nf*4 + self.skip_n_channels[-2], nf*4),
                vgg_layer(nf*4, nf*4),
                vgg_layer(nf*4, nf*2),
            )
        )  # 8 x 8

        dec_convs.append(
            nn.Sequential(
                vgg_layer(nf*2 + self.skip_n_channels[-3], nf*2),
                vgg_layer(nf*2, nf*2),
                vgg_layer(nf*2, nf*2),
            )
        )  # 16 x 16

        if self.img_size[0] == 64:
            dec_convs.append(
                nn.Sequential(
                    vgg_layer(nf*2 + self.skip_n_channels[-4], nf*2),
                    vgg_layer(nf*2, nf),
                )
            )  # 32 x 32
            self.dec_convs = nn.ModuleList(dec_convs)

            out_channels = self.img_size[-1]
            self.last_conv = nn.Sequential(
                vgg_layer(nf + self.skip_n_channels[0], nf),
                nn.ConvTranspose2d(nf, out_channels, 3, 1, 1),
            )  # 64 x 64
        elif self.img_size[0] == 32:
            self.dec_convs = nn.ModuleList(dec_convs)

            out_channels = self.img_size[-1]
            self.last_conv = nn.Sequential(
                vgg_layer(nf*2 + self.skip_n_channels[0], nf),
                nn.ConvTranspose2d(nf, out_channels, 3, 1, 1),
            )  # 32 x 32
        else:
            raise NotImplementedError
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

        for module in self.upc1:
            _init_layer(module)
        for module in self.last_conv:
            _init_layer(module)

    def forward(self, x, enc_skip, decode_masked_slots):
        """
        :param x: trajectories (num_sims, num_atoms, num_timesteps, num_dims)
        :return:
        """
        x_in_shp = list(x.size())

        x = self.append_object_idcs(x)
        x = self.prepare_input_for_conv(x)

        skip_idx = -1
        if self.use_skip and self.skip_code[skip_idx] == "Y":
            es = enc_skip[skip_idx]
            es = es.contiguous().view([-1] + list(es.shape[2:]))
            x = torch.cat([x, es], -3)
            skip_idx -= 1
        x = self.upc1(x)

        for module in self.dec_convs:
            x = self.up(x)
            if self.use_skip and self.skip_code[skip_idx] == "Y":
                es = enc_skip[skip_idx]
                es = es.contiguous().view([-1] + list(es.shape[2:]))
                x = torch.cat([x, es], -3)
                skip_idx -= 1
            x = module(x)

        x = self.up(x)
        if self.use_skip and self.skip_code[skip_idx] == "Y":
            es = enc_skip[skip_idx]
            es = es.contiguous().view([-1] + list(es.shape[2:]))
            x = torch.cat([x, es], -3)
        x = self.last_conv(x) # 32
        if self.use_output_sigmoid:
            x = torch.sigmoid(x)

        x, x_objs = self.post_process_output_and_objs(x, x_in_shp)

        return x, x_objs


class ParDCGANDecoder(ParConvDecoder):
    def __init__(self, n_dims, img_size, dec_t2i, skip_n_channels):
        super(ParDCGANDecoder, self).__init__(n_dims, img_size, dec_t2i, skip_n_channels)
        self.is_dcgan = True

    def create_conv_layers(self):
        dec_convs = []

        nf = self.nf_dcgan
        dec_convs.append(
            dcgan_upconv(self.in_channels[0] + self.skip_n_channels[-1], nf * 2))
        dec_convs.append(
            dcgan_upconv(nf * 2 + self.skip_n_channels[-2], nf))

        if self.img_size[0] == 64:
            dec_convs.append(
                dcgan_upconv(nf + self.skip_n_channels[-3], nf))
            self.dec_convs = nn.ModuleList(dec_convs)
        elif self.img_size[0] == 32:
            self.dec_convs = nn.ModuleList(dec_convs)
        else:
            raise NotImplementedError
        nc = self.img_size[-1]
        self.last_upsample = None
        self.last_conv = nn.Sequential(
            nn.ConvTranspose2d(nf + self.skip_n_channels[0], nc, 4, 2, 1))
        for module in self.last_conv:
            _init_layer(module)


class SlotDCGANDecoder(SlotConvDecoder):
    def __init__(self, n_dims, img_size, dec_t2i, skip_n_channels):
        super(SlotDCGANDecoder, self).__init__(n_dims, img_size, dec_t2i, skip_n_channels)
        self.is_dcgan = True

    def create_conv_layers(self):
        dec_convs = []

        nf = self.nf_dcgan
        dec_convs.append(
            dcgan_upconv(self.in_channels[0] + self.skip_n_channels[-1], nf * 4))
        dec_convs.append(
            dcgan_upconv(nf * 4 + self.skip_n_channels[-2], nf * 2))
        dec_convs.append(
            dcgan_upconv(nf * 2 + self.skip_n_channels[-3], nf))

        if self.img_size[0] == 64:
            dec_convs.append(
                dcgan_upconv(nf + self.skip_n_channels[-4], nf))
            self.dec_convs = nn.ModuleList(dec_convs)
        elif self.img_size[0] == 32:
            self.dec_convs = nn.ModuleList(dec_convs)
        else:
            raise NotImplementedError
        nc = self.img_size[-1]
        self.last_upsample = None
        self.last_conv = nn.Sequential(
            nn.ConvTranspose2d(nf + self.skip_n_channels[0], nc, 4, 2, 1))
        for module in self.last_conv:
            _init_layer(module)


class ParSBDecoder(ParConvDecoder):
    def __init__(self, n_dims, img_size, dec_t2i, skip_n_channels):
        super(ParSBDecoder, self).__init__(n_dims, img_size, dec_t2i, skip_n_channels)
        xs, ys = self.img_size[:2]
        x = torch.linspace(-1, 1, xs)
        y = torch.linspace(-1, 1, ys)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as a constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

    def calculate_input_output_n_channels(self):
        # Take just first 3(4) layer configs - up-convolve to 32(64) img size
        # Here we start from 4x4 tensor and do 3(4) upsampling operations
        self.dec_start_res = self.img_size[0]
        self.in_channels = self.in_channels[:-1]
        self.ks = self.ks[-1]
        self.conv_stride = self.conv_stride[:-1]
        self.input_map_n_out = self.in_channels[0]

    def prepare_input_for_conv(self, x):
        # flatten the B, K, T dimensions
        xshp = list(x.size())
        x = x.view(-1, x.size(-1))
        if self.in_map:
            x = self.in_map(x)
        x = x.view(xshp[:-1] + [x.shape[-1]])

        x = x.transpose(1, 2)
        # broadcast_input:
        x = x.contiguous()
        x = x.view(-1, x.size(-2)* x.size(-1))
        x = self.broadcast_op(x, self.dec_start_res)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x32x32/64x64
        batch_size = x.shape[0]
        x = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                       self.y_grid.expand(batch_size, -1, -1, -1), x), dim=1)
        return x

    def create_conv_layers(self):
        dec_convs = []
        nf = self.nf_sb
        dec_convs.append(
            nn.Conv2d(in_channels=self.in_channels[0] * 16 + 2, out_channels=nf,
                      kernel_size=3, padding=1))
        dec_convs.append(nn.Conv2d(in_channels=nf, out_channels=nf,
                                   kernel_size=3, padding=1))
        self.dec_convs = nn.ModuleList(dec_convs)
        self.last_upsample = None
        self.last_conv = nn.Conv2d(in_channels=nf,
                                   out_channels=self.img_size[-1],
                                   kernel_size=3, padding=1)


class SlotSBDecoder(SlotConvDecoder):
    def __init__(self, n_dims, img_size, dec_t2i, skip_n_channels):
        super(SlotSBDecoder, self).__init__(n_dims, img_size, dec_t2i, skip_n_channels)
        xs, ys = self.img_size[:2]
        x = torch.linspace(-1, 1, xs)
        y = torch.linspace(-1, 1, ys)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as a constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

    def calculate_input_output_n_channels(self):
        # Take just first 3(4) layer configs - up-convolve to 32(64) img size
        # Here we start from 4x4 tensor and do 3(4) upsampling operations
        self.dec_start_res = self.img_size[0]
        self.in_channels = self.in_channels[:-1]
        self.ks = self.ks[-1]
        self.conv_stride = self.conv_stride[:-1]
        self.input_map_n_out = self.in_channels[0]

    def prepare_input_for_conv(self, x):
        # flatten the B, K, T dimensions
        x = x.view(-1, x.size(-1))
        if self.in_map:
            x = self.in_map(x)
        x = self.broadcast_op(x, self.dec_start_res)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x32x32/64x64
        batch_size = x.shape[0]
        x = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                       self.y_grid.expand(batch_size, -1, -1, -1), x), dim=1)
        return x

    def create_conv_layers(self):
        dec_convs = []
        nf = self.nf_sb
        dec_convs.append(
            nn.Conv2d(in_channels=self.in_channels[0] + 2, out_channels=nf,
                      kernel_size=3, padding=1))
        dec_convs.append(nn.Conv2d(in_channels=nf, out_channels=nf,
                                   kernel_size=3, padding=1))
        self.dec_convs = nn.ModuleList(dec_convs)
        self.last_upsample = None
        self.last_conv = nn.Conv2d(in_channels=nf,
                                   out_channels=self.img_size[-1],
                                   kernel_size=3, padding=1)
