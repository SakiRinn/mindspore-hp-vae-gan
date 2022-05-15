from __future__ import absolute_import, division, print_function
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import Normal, Zero
import mindspore.nn.probability.distribution as msd
import copy
import sys
sys.path.append("..")
import numpy as np
import utils

matmul = ops.MatMul()
exp = ops.Exp()
log = ops.Log()
tanh = ops.Tanh()
sigmoid = ops.Sigmoid()
bernoulli = msd.Bernoulli(0.5)
uniform = msd.Uniform(0, 1)


# Unused.
# def conv_weights_init_ones(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv2d') == 0 or classname.find('Conv2d') == 0:
#         m.weight.data.fill_(1 / np.prod(m.kernel_size))
#         m.bias.data.fill_(0)


# Unused.
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv2d') != -1 or classname.find('Conv3d') != -1:
#         m.weight.data = Tensor(shape=m.weight.data.shape, init=Normal(0.0, 0.02))
#     elif classname.find('Norm') != -1:
#         m.weight.data = Tensor(shape=m.weight.data.shape, init=Normal(1.0, 0.02))
#         m.bias.data = Tensor(shape=m.weight.data.shape, init=Zero())


def get_activation(act):
    activations = {
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "prelu": nn.PReLU(),
        "selu": ops.SeLU() 
    }
    return activations[act]


def reparameterize(mu, logvar, training):
    if training:
        std = exp(logvar * 0.5)
        eps = Tensor(shape=std.shape, init=Normal())
        return matmul(eps, std) + (mu)
    else:
        return Tensor(shape=mu.shape, init=Normal())


def reparameterize_bern(x, training):
    if training:
        eps = uniform.prob(Tensor(shape=x.shape, init=Zero()))
        return log(x + 1e-20) - log(-log(eps + 1e-20) + 1e-20)
    else: 
        return bernoulli.prob(Tensor(shape=x.shape, init=Zero()))


class ConvBlock2D(nn.SequentialCell):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu'):
        super(ConvBlock2D, self).__init__()
        self.append(nn.Conv2d(in_channel, out_channel, kernel_size=ker_size,
                              stride=stride, padding=padding, weight_init=Normal(0.0, 0.02),
                              pad_mode='pad', has_bias=True))
        if bn:
            self.append(nn.BatchNorm2d(out_channel))
        if act is not None:
            self.append(get_activation(act))


class ConvBlock2DSN(nn.SequentialCell):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu'):
        super(ConvBlock2DSN, self).__init__()
        if bn:
            self.append(utils.SpectualNormConv2d(in_channel, out_channel, 
                                                 kernel_size=ker_size,
                                                 stride=stride, padding=padding,
                                                 pad_mode='pad', has_bias=True))
            self.append(nn.Conv2d(in_channel, out_channel, kernel_size=ker_size,
                                  stride=stride, padding=padding, weight_init=Normal(0.0, 0.02),
                                  pad_mode='pad', has_bias=True))
        else:
            paddings = ((padding, padding), (padding, padding), 
                        (padding, padding), (padding, padding))
            self.append(nn.Pad(paddings=paddings, mode='REFLECT'))  # FIXME: reflect pad
            self.append(nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, 
                                  stride=stride,weight_init=Normal(0.0, 0.02),
                                  pad_mode='valid', has_bias=True))
        if act is not None:
            self.append(get_activation(act))


class FeatureExtractor(nn.SequentialCell):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, num_blocks=2, return_linear=False):
        super(FeatureExtractor, self).__init__()
        self.append(ConvBlock2DSN(in_channel, out_channel, ker_size, padding, stride))
        for i in range(num_blocks - 1):
            self.append(ConvBlock2DSN(out_channel, out_channel, ker_size, padding, stride))
        if return_linear:
            self.append(ConvBlock2DSN(out_channel, out_channel, ker_size, 
                                      padding, stride, bn=False, act=None))
        else:
            self.append(ConvBlock2DSN(out_channel, out_channel, ker_size, padding, stride))


class Encode2DVAE(nn.Cell):
    def __init__(self, opt, out_dim=None, num_blocks=2):
        super(Encode2DVAE, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            assert type(out_dim) is int
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size,
                                         opt.ker_size // 2, 1, num_blocks=num_blocks)
        self.mu = ConvBlock2D(opt.nfc, output_dim, opt.ker_size, 
                              opt.ker_size // 2, 1, bn=False, act=None)
        self.logvar = ConvBlock2D(opt.nfc, output_dim, opt.ker_size, 
                                  opt.ker_size // 2, 1, bn=False, act=None)

    def forward(self, x):
        features = self.features(x)
        mu = self.mu(features)
        logvar = self.logvar(features)

        return mu, logvar


class Encode2DVAE_nb(nn.Cell):
    def __init__(self, opt, out_dim=None, num_blocks=2):
        super(Encode2DVAE_nb, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            assert type(out_dim) is int
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size, 
                                         opt.ker_size // 2, 1, num_blocks=num_blocks)
        self.mu = nn.SequentialCell()
        self.mu.append(ConvBlock2D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, 
                                   bn=False, act=None))
        
        self.logvar = nn.SequentialCell()
        self.logvar.append(ConvBlock2D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, 
                                       bn=False, act=None))
        # self.logvar.append(nn.AdaptiveAvgPool2D(1))
        
        self.bern = ConvBlock2D(opt.nfc, 1, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)

    def forward(self, x):
        reduce_mean = ops.ReduceMean(keep_dims=True)
        
        features = self.features(x)
        bern = sigmoid(self.bern(features))
        features = bern * features
        mu = reduce_mean(self.mu(features))
        logvar = reduce_mean(self.logvar(features))

        return mu, logvar, bern


class Encode3DVAE1x1(nn.Cell):
    def __init__(self, opt, out_dim=None):
        super(Encode3DVAE1x1, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            assert type(out_dim) is int
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, 1, 0, 1, num_blocks=2)
        self.mu = ConvBlock2D(opt.nfc, output_dim, 1, 0, 1, bn=False, act=None)
        self.logvar = ConvBlock2D(opt.nfc, output_dim, 1, 0, 1, bn=False, act=None)

    def forward(self, x):
        features = self.features(x)
        mu = self.mu(features)
        logvar = self.logvar(features)

        return mu, logvar


class WDiscriminator2D(nn.Cell):
    def __init__(self, opt):
        super(WDiscriminator2D, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.head = ConvBlock2DSN(opt.nc_im, N, opt.ker_size, 
                                  opt.ker_size // 2, stride=1, bn=True, act='lrelu')
        self.body = nn.SequentialCell()
        for i in range(opt.num_layer):
            block = ConvBlock2DSN(N, N, opt.ker_size, 
                                  opt.ker_size // 2, stride=1, bn=True, act='lrelu')
            self.body.append(block)
        self.tail = nn.Conv2d(N, 1, kernel_size=opt.ker_size, padding=1, stride=1, 
                              weight_init=Normal(0.0, 0.02), 
                              pad_mode='pad', has_bias=True)

    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out


class GeneratorHPVAEGAN(nn.Cell):
    def __init__(self, opt):
        super(GeneratorHPVAEGAN, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.N = N

        self.encode = Encode2DVAE(opt, out_dim=opt.latent_dim, num_blocks=opt.enc_blocks)
        self.decoder = nn.SequentialCell()

        # Normal Decoder
        self.decoder.append(ConvBlock2D(opt.latent_dim, N, opt.ker_size, opt.padd_size, stride=1))
        for i in range(opt.num_layer):
            block = ConvBlock2D(N, N, opt.ker_size, opt.padd_size, stride=1)
            self.decoder.append(block)
        self.decoder.append(nn.Conv2d(N, opt.nc_im, opt.ker_size, 
                                      stride=1, padding=opt.ker_size // 2, weight_init=Normal(0.0, 0.02),
                                      pad_mode='pad', has_bias=True))

        # 1x1 Decoder
        # self.decoder.append(ConvBlock2D(opt.latent_dim, N, 1, 0, stride=1))
        # for i in range(opt.num_layer):
        #     block = ConvBlock2D(N, N, 1, 0, stride=1)
        #     self.decoder.append(block)
        # self.decoder.append(nn.Conv2d(N, opt.nc_im, 1, 1, 0))

        self.body = nn.CellList([])

    def init_next_stage(self):
        if len(self.body) == 0:
            _first_stage = nn.SequentialCell()
            _first_stage.append(ConvBlock2D(self.opt.nc_im, self.N, 
                                            self.opt.ker_size, self.opt.padd_size, stride=1))
            for i in range(self.opt.num_layer):
                block = ConvBlock2D(self.N, self.N, self.opt.ker_size, self.opt.padd_size, stride=1)
                _first_stage.append(block)
            _first_stage.append(nn.Conv2d(self.N, self.opt.nc_im, self.opt.ker_size, 
                                          stride=1, padding=self.opt.ker_size // 2, weight_init=Normal(0.0, 0.02),
                                          pad_mode='pad', has_bias=True))
            self.body.append(_first_stage)
        else:
            self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, video, noise_amp, noise_init=None, sample_init=None, mode='rand'):
        if sample_init is not None:
            assert len(self.body) > sample_init[0], "Strating index must be lower than # of body blocks"

        if noise_init is None:
            mu, logvar = self.encode(video)
            z_vae = reparameterize(mu, logvar, self.training)
        else:
            z_vae = noise_init

        vae_out = tanh(self.decoder(z_vae))

        if sample_init is not None:
            x_prev_out = self.refinement_layers(sample_init[0], sample_init[1], noise_amp, mode)
        else:
            x_prev_out = self.refinement_layers(0, vae_out, noise_amp, mode)

        if noise_init is None:
            return x_prev_out, vae_out, (mu, logvar)
        else:
            return x_prev_out, vae_out

    def refinement_layers(self, start_idx, x_prev_out, noise_amp, mode):
        for idx, block in enumerate(self.body[start_idx:], start_idx):
            if self.opt.vae_levels == idx + 1 and not self.opt.train_all:
                x_prev_out = ops.stop_gradient(x_prev_out)

            # Upscale
            x_prev_out_up = utils.upscale_2d(x_prev_out, idx + 1, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if mode == 'rand':
                noise = utils.generate_noise(ref=x_prev_out_up)
                x_prev = block(x_prev_out_up + noise * noise_amp[idx + 1])
            else:
                x_prev = block(x_prev_out_up)

            x_prev_out = tanh(x_prev + x_prev_out_up)

        return x_prev_out


class GeneratorVAE_nb(nn.Cell):
    def __init__(self, opt):
        super(GeneratorVAE_nb, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.N = N

        self.encode = Encode2DVAE_nb(opt, out_dim=opt.latent_dim, num_blocks=opt.enc_blocks)
        self.decoder = nn.SequentialCell()

        # Normal Decoder
        self.decoder.append(ConvBlock2D(opt.latent_dim, N, opt.ker_size, opt.padd_size, stride=1))
        for i in range(opt.num_layer):
            block = ConvBlock2D(N, N, opt.ker_size, opt.padd_size, stride=1)
            self.decoder.append(block)
        self.decoder.append(nn.Conv2d(N, opt.nc_im, opt.ker_size,
                                      stride=1, padding=opt.ker_size // 2,
                                      weight_init=Normal(0.0, 0.02),
                                      pad_mode='pad', has_bias=True))

        self.body = nn.CellList([])

    def init_next_stage(self):
        if len(self.body) == 0:
            _first_stage = nn.SequentialCell()
            _first_stage.append(ConvBlock2D(self.opt.nc_im, self.N, 
                                            self.opt.ker_size, self.opt.padd_size, stride=1))
            for i in range(self.opt.num_layer):
                block = ConvBlock2D(self.N, self.N, self.opt.ker_size, self.opt.padd_size, stride=1)
                _first_stage.append(block)
            _first_stage.append(nn.Conv2d(self.N, self.opt.nc_im, self.opt.ker_size, 
                                          stride=1, padding=self.opt.ker_size // 2,
                                          weight_init=Normal(0.0, 0.02),
                                          pad_mode='pad', has_bias=True))
            self.body.append(_first_stage)
        else:
            self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, video, noise_amp, 
                noise_init_norm=None, noise_init_bern=None, sample_init=None, mode='rand'):
        if sample_init is not None:
            assert len(self.body) > sample_init[0], "Strating index must be lower than # of body blocks"

        if noise_init_norm is None:
            mu, logvar, bern = self.encode(video)
            z_vae_norm = reparameterize(mu, logvar, self.training)
            z_vae_bern = reparameterize_bern(bern, self.training)
        else:
            z_vae_norm = noise_init_norm
            z_vae_bern = noise_init_bern

        vae_out = tanh(self.decoder(z_vae_norm * z_vae_bern))

        if sample_init is not None:
            x_prev_out = self.refinement_layers(sample_init[0], sample_init[1], noise_amp, mode)
        else:
            x_prev_out = self.refinement_layers(0, vae_out, noise_amp, mode)

        if noise_init_norm is None:
            return x_prev_out, vae_out, (mu, logvar, bern)
        else:
            return x_prev_out, vae_out

    def refinement_layers(self, start_idx, x_prev_out, noise_amp, mode):
        for idx, block in enumerate(self.body[start_idx:], start_idx):
            if self.opt.vae_levels == idx + 1:
                x_prev_out = ops.stop_gradient(x_prev_out)

            # Upscale
            x_prev_out_up = utils.upscale_2d(x_prev_out, idx + 1, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if mode == 'rand':
                noise = utils.generate_noise(ref=x_prev_out_up)
                x_prev = block(x_prev_out_up + noise * noise_amp[idx + 1])
            else:
                x_prev = block(x_prev_out_up)

            x_prev_out = tanh(x_prev + x_prev_out_up)

        return x_prev_out


if __name__ == '__main__':
    class Opt:
        def __init__(self):
            self.nfc = 64
            self.nc_im = 3
            self.ker_size = 3
            self.num_layer = 5
            self.latent_dim = 128
            self.enc_blocks = 2
            self.padd_size = 1

    opt = Opt()
    model = GeneratorVAE_nb(opt)
    model.init_next_stage()
    print(model)