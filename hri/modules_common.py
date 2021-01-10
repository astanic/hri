import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class LeakyRelu(nn.LeakyReLU):
    def __init__(self, inplace, *args, **kwargs):
        super().__init__(negative_slope=0.2, inplace=inplace)


ACTIVATION_FUNCTIONS = {
    'none': Identity,
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'leaky_relu': LeakyRelu,
}


NORMALIZATION_MODULE = {
    'none': Identity,
    'layer': nn.LayerNorm,
    'instance': nn.InstanceNorm2d,
    'batch': nn.BatchNorm2d,
}


class NonLinear(nn.Module):
    def __init__(self, n_in, n_out, non_linearity):
        super(NonLinear, self).__init__()
        self.layer = nn.Linear(n_in, n_out)
        nn.init.xavier_normal_(self.layer.weight.data)
        self.layer.bias.data.fill_(0.1)
        self.non_linearity = ACTIVATION_FUNCTIONS[non_linearity]()

    def forward(self, x):
        return self.non_linearity(self.layer(x))


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""
    # source: https://github.com/ethanfetaya/NRI/

    def __init__(self, n_in, n_hid, n_out, do_prob=0.,
                 out_nl='elu', out_bn=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout = nn.Dropout(do_prob)

        self.out_nl = ACTIVATION_FUNCTIONS[out_nl]()
        self.out_bn = out_bn

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

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.out_nl(x)
        if self.out_bn:
            x = self.batch_norm(x)
        return x


###############################################################################
###############################################################################
## Image modules


class Conv2dTF(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    source: https://github.com/mlperf/inference/blob/master/others/edge/
                    object_detection/ssd_mobilenet/pytorch/utils.py#L40
    """

    def __init__(self, *args, **kwargs):
        self.padding = kwargs.get("padding", "SAME")
        kwargs["padding"] = 0
        super(Conv2dTF, self).__init__(*args, **kwargs)

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)
        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])
        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class ConvBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool, padding,
                 nm='none', non_linearity='none', stride=1):
        super(ConvBNBlock, self).__init__()

        self.kernel_size = kernel_size

        if self.kernel_size > 0:
            self.conv = Conv2dTF(in_channels, out_channels, kernel_size,
                                 [stride, stride], padding=padding)
            self.nm = NORMALIZATION_MODULE[nm](out_channels)
            self.non_linearity = ACTIVATION_FUNCTIONS[non_linearity](inplace=True)
            # init weights
            self.init_weights(nm)
        self.pool = nn.MaxPool2d(pool)


    def init_weights(self, nm):
        # Conv
        n = self.conv.kernel_size[0] * self.conv.out_channels
        self.conv.weight.data.normal_(0, np.sqrt(2. / n))
        self.conv.bias.data.fill_(0.1)
        # BN
        if nm == 'batch' or nm == 'layer':
            self.nm.weight.data.fill_(1)
            self.nm.bias.data.zero_()

    def forward(self, inputs):
        # Input shape: [num_sims * num_timesteps, H, W, C]
        x = inputs
        if self.kernel_size > 0:
            x = self.conv(x)
            x = self.nm(x)
            x = self.non_linearity(x)
        x = self.pool(x)
        return x


def _init_layer(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
