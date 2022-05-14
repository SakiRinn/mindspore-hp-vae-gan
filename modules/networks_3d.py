from __future__ import absolute_import, division, print_function
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import Normal, Zero
import mindspore.nn.probability.distribution as msd
from mindspore import nn
import copy
import numpy as np
import sys
sys.path.append("..")
import utils

matmul = ops.MatMul()
exp = ops.Exp()
log = ops.Log()
tanh = ops.Tanh()
sigmoid = ops.Sigmoid()
adaptive_avg_pool_2d = ops.AdaptiveAvgPool2D(1)
bernoulli = msd.Bernoulli(0.5)
uniform = msd.Uniform(0, 1)


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


# Basic blocks

class ConvBlock3D(nn.SequentialCell):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu'):
        super(ConvBlock3D, self).__init__()
        self.append(nn.Conv3d(in_channel, out_channel, kernel_size=ker_size,
                              stride=stride, padding=padding, weight_init=Normal(0.0, 0.02),
                              pad_mode='pad', has_bias=True))
        if bn:
            self.append(nn.BatchNorm3d(out_channel))
        if act is not None:
            self.append(get_activation(act))


class ConvBlock3DSN(nn.SequentialCell):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu'):
        super(ConvBlock3DSN, self).__init__()
        if bn:
            self.append(utils.SpectualNormConv2d(in_channel, out_channel, 
                                                 kernel_size=ker_size,
                                                 stride=stride, padding=padding,
                                                 pad_mode='pad', has_bias=True))
            self.append(nn.Conv3d(in_channel, out_channel, kernel_size=ker_size,
                                  stride=stride, padding=padding, weight_init=Normal(0.0, 0.02),
                                  pad_mode='pad', has_bias=True))
        else:
            # paddings = ((padding, padding), (padding, padding), 
            #             (padding, padding), (padding, padding),
            #             (padding, padding), (padding, padding),
            #             (padding, padding), (padding, padding),
            #             (padding, padding), (padding, padding))
            # self.append(nn.Pad(paddings=paddings, mode='REFLECT'))  # FIXME: reflect pad
            self.append(nn.Conv3d(in_channel, out_channel, kernel_size=ker_size, 
                                  stride=stride, weight_init=Normal(0.0, 0.02),
                                  pad_mode='valid', has_bias=False))
        if act is not None:
            self.append(get_activation(act))


class FeatureExtractor(nn.SequentialCell):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, 
                 num_blocks=2, return_linear=False):
        super(FeatureExtractor, self).__init__()
        self.append(ConvBlock3DSN(in_channel, out_channel, ker_size, padding, stride)),
        for i in range(num_blocks - 1):
            self.append(ConvBlock3DSN(out_channel, out_channel, ker_size, padding, stride))
        if return_linear:
            self.append(ConvBlock3DSN(out_channel, out_channel, ker_size, padding, stride, 
                                      bn=False, act=None))
        else:
            self.append(ConvBlock3DSN(out_channel, out_channel, ker_size, padding, stride))


class Encode3DVAE(nn.Cell):
    def __init__(self, opt, out_dim=None, num_blocks=2):
        super(Encode3DVAE, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            assert type(out_dim) is int
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size, opt.ker_size // 2, 1, num_blocks=num_blocks)
        self.mu = ConvBlock3D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)
        self.logvar = ConvBlock3D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)

    def forward(self, x):
        features = self.features(x)
        mu = self.mu(features)
        logvar = self.logvar(features)

        return mu, logvar


class Encode3DVAE_nb(nn.Cell):
    def __init__(self, opt, out_dim=None, num_blocks=2):
        super(Encode3DVAE_nb, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            assert type(out_dim) is int
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size, opt.ker_size // 2, 1, num_blocks=num_blocks)
        
        self.mu = nn.SequentialCell()
        self.mu.append(ConvBlock3D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, 
                                   bn=False, act=None))
        # self.mu.append(ops.ReduceMean(keep_dims=True))    # FIXME: 没有nn
        
        self.logvar = nn.SequentialCell()
        self.logvar.append(ConvBlock3D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1,
                                       bn=False, act=None))
        # self.logvar.append(ops.ReduceMean(keep_dims=True))
        
        self.bern = ConvBlock3D(opt.nfc, 1, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)

    def forward(self, x):
        features = self.features(x)
        bern = sigmoid(self.bern(features))
        features = bern * features
        mu = self.mu(features)
        logvar = self.logvar(features)

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
        self.mu = ConvBlock3D(opt.nfc, output_dim, 1, 0, 1, bn=False, act=None)
        self.logvar = ConvBlock3D(opt.nfc, output_dim, 1, 0, 1, bn=False, act=None)

    def forward(self, x):
        features = self.features(x)
        mu = self.mu(features)
        logvar = self.logvar(features)

        return mu, logvar


class WDiscriminator3D(nn.Cell):
    def __init__(self, opt):
        super(WDiscriminator3D, self).__init__()

        self.opt = opt
        N = int(opt.nfc)

        self.head = ConvBlock3DSN(opt.nc_im, N, opt.ker_size, opt.ker_size // 2, stride=1, 
                                  bn=True, act='lrelu')
        self.body = nn.SequentialCell()
        for i in range(opt.num_layer):
            block = ConvBlock3DSN(N, N, opt.ker_size, opt.ker_size // 2, stride=1, bn=True, act='lrelu')
            self.body.append(block)
        self.tail = nn.Conv3d(N, 1, kernel_size=opt.ker_size, padding=1, stride=1,
                              weight_init=Normal(0.0, 0.02), pad_mode='pad', has_bias=True)

    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out


class WDiscriminatorBaselines(nn.Cell):
    def __init__(self, opt):
        super(WDiscriminatorBaselines, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.p3d = (self.opt.num_layer + 2, self.opt.num_layer + 2,
                    self.opt.num_layer + 2, self.opt.num_layer + 2,
                    self.opt.num_layer + 2, self.opt.num_layer + 2)

        self.head = ConvBlock3D(opt.nc_im, N, opt.ker_size, opt.padd_size, stride=1, bn=False, act='lrelu')
        self.body = nn.SequentialCell()
        for i in range(opt.num_layer):
            block = ConvBlock3D(N, N, opt.ker_size, opt.padd_size, stride=1, bn=True, act='lrelu')
            self.body.append(block)

        self.tail = nn.Conv3d(N, 1, kernel_size=opt.ker_size, padding=opt.padd_size, stride=1,
                              weight_init=Normal(0.0, 0.02), pad_mode='pad', has_bias=True)

    def forward(self, x):
        pad_op = ops.Pad(self.p3d)
        x = pad_op(x)

        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out


class GeneratorCSG(nn.Cell):
    def __init__(self, opt):
        super(GeneratorCSG, self).__init__()

        self.opt = opt
        N = int(opt.nfc)

        self.p3d_once = (1, 1,
                         1, 1,
                         1, 1)
        self.p3d = (self.opt.num_layer + 0, self.opt.num_layer + 0,
                    self.opt.num_layer + 0, self.opt.num_layer + 0,
                    self.opt.num_layer + 0, self.opt.num_layer + 0)

        self.head = ConvBlock3D(opt.nc_im, N, opt.ker_size, padding=0, stride=1)

        self.body = nn.CellList([])
        _first_stage = nn.SequentialCell()
        for i in range(opt.num_layer):
            block = ConvBlock3D(N, N, opt.ker_size, padding=0, stride=1)
            _first_stage.append(block)
        self.body.append(_first_stage)

        self.tail = nn.SequentialCell()
        self.tail.append(nn.Conv3d(N, opt.nc_im, kernel_size=opt.ker_size, padding=0, stride=1,
                                   weight_init=Normal(0.0, 0.02), pad_mode='pad', has_bias=True))
        self.tail.append(nn.Tanh())


    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise_init, noise_amp, mode='rand'):
        pad_op1 = ops.Pad(self.p3d_once)
        x = self.head(pad_op1(noise_init))

        pad_op2 = ops.Pad(self.p3d)
        x_prev_out = self.body[0](pad_op2(x))

        for idx, block in enumerate(self.body[1:], 1):
            # Upscale
            x_prev_out_up = utils.upscale(x_prev_out, idx, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if mode == 'rand':
                x_prev_out_up_2 = utils.interpolate_3D(x_prev_out, size=[
                    x_prev_out_up.shape[-3] + (self.opt.num_layer + 0) * 2,
                    x_prev_out_up.shape[-2] + (self.opt.num_layer + 0) * 2,
                    x_prev_out_up.shape[-1] + (self.opt.num_layer + 0) * 2
                ])
                noise = utils.generate_noise(ref=x_prev_out_up_2)
                x_prev = block(x_prev_out_up_2 + noise * noise_amp[idx])
            else:
                x_prev = block(pad_op2(x_prev_out_up))
            x_prev_out = x_prev + x_prev_out_up

        out = self.tail(pad_op1(x_prev_out))
        return out


class GeneratorSG(nn.Cell):
    def __init__(self, opt):
        super(GeneratorSG, self).__init__()

        self.opt = opt
        N = int(opt.nfc)

        self.p3d = (self.opt.num_layer + 2, self.opt.num_layer + 2,
                    self.opt.num_layer + 2, self.opt.num_layer + 2,
                    self.opt.num_layer + 2, self.opt.num_layer + 2)

        self.body = nn.CellList([])

        _first_stage = nn.SequentialCell()
        _first_stage.append(ConvBlock3D(opt.nc_im, N, opt.ker_size, padding=0, stride=1))
        for i in range(opt.num_layer):
            block = ConvBlock3D(N, N, opt.ker_size, padding=0, stride=1)
            _first_stage.append(block)
        _first_stage.append(nn.Conv3d(N, opt.nc_im, kernel_size=opt.ker_size, padding=0, stride=1, 
                                      weight_init=Normal(0.0, 0.02)))
        self.body.append(_first_stage)


    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise_init, noise_amp, mode='rand'):
        pad_op = ops.Pad(self.p3d)
        x_prev_out = self.body[0](pad_op(noise_init))

        for idx, block in enumerate(self.body[1:], 1):
            x_prev_out = tanh(x_prev_out)

            # Upscale
            x_prev_out_up = utils.upscale(x_prev_out, idx, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if mode == 'rand':
                x_prev_out_up_2 = utils.interpolate_3D(x_prev_out, size=[
                    x_prev_out_up.shape[-3] + (self.opt.num_layer + 2) * 2,
                    x_prev_out_up.shape[-2] + (self.opt.num_layer + 2) * 2,
                    x_prev_out_up.shape[-1] + (self.opt.num_layer + 2) * 2
                ])
                noise = utils.generate_noise(ref=x_prev_out_up_2)
                x_prev = block(x_prev_out_up_2 + noise * noise_amp[idx])
            else:
                x_prev = block(pad_op(x_prev_out_up))
            x_prev_out = x_prev + x_prev_out_up

        out = tanh(x_prev_out)
        return out


class GeneratorHPVAEGAN(nn.Cell):
    def __init__(self, opt):
        super(GeneratorHPVAEGAN, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.N = N

        self.encode = Encode3DVAE(opt, out_dim=opt.latent_dim, num_blocks=opt.enc_blocks)
        self.decoder = nn.SequentialCell()

        # Normal Decoder
        self.decoder.append(ConvBlock3D(opt.latent_dim, N, opt.ker_size, opt.padd_size, stride=1))
        for i in range(opt.num_layer):
            block = ConvBlock3D(N, N, opt.ker_size, opt.padd_size, stride=1)
            self.decoder.append(block)
        self.decoder.append(nn.Conv3d(N, opt.nc_im, opt.ker_size, 
                                      stride=1, padding=opt.ker_size // 2, 
                                      weight_init=Normal(0.0, 0.02), pad_mode='pad', has_bias=True))

        # 1x1 Decoder
        # self.decoder.append(ConvBlock3D(opt.latent_dim, N, 1, 0, stride=1))
        # for i in range(opt.num_layer):
        #     block = ConvBlock3D(N, N, 1, 0, stride=1)
        #     self.decoder.append(block)
        # self.decoder.append(nn.Conv3d(N, opt.nc_im, 1, 1, 0))

        self.body = nn.CellList([])

    def init_next_stage(self):
        if len(self.body) == 0:
            _first_stage = nn.SequentialCell()
            _first_stage.append(ConvBlock3D(self.opt.nc_im, self.N, 
                                            self.opt.ker_size, self.opt.padd_size, stride=1))
            for i in range(self.opt.num_layer):
                block = ConvBlock3D(self.N, self.N, self.opt.ker_size, self.opt.padd_size, stride=1)
                _first_stage.append(block)
            _first_stage.append(nn.Conv3d(self.N, self.opt.nc_im, self.opt.ker_size, 
                                          stride=1, padding=self.opt.ker_size // 2, 
                                          weight_init=Normal(0.0, 0.02), pad_mode='pad', has_bias=True))
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

        vae_out = ops.tanh(self.decoder(z_vae))

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
            x_prev_out_up = utils.upscale(x_prev_out, idx + 1, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if mode == 'rand' and self.opt.vae_levels <= idx + 1:
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

        self.encode = Encode3DVAE_nb(opt, out_dim=opt.latent_dim, num_blocks=opt.enc_blocks)
        self.decoder = nn.SequentialCell()

        # Normal Decoder
        self.decoder.append(ConvBlock3D(opt.latent_dim, N, opt.ker_size, opt.padd_size, stride=1))
        for i in range(opt.num_layer):
            block = ConvBlock3D(N, N, opt.ker_size, opt.padd_size, stride=1)
            self.decoder.append(block)
        self.decoder.append(nn.Conv3d(N, opt.nc_im, opt.ker_size, 
                                      stride=1, padding=opt.ker_size // 2, 
                                      weight_init=Normal(0.0, 0.02), pad_mode='pad', has_bias=True))

        self.body = nn.CellList([])

    def init_next_stage(self):
        if len(self.body) == 0:
            _first_stage = nn.SequentialCell()
            _first_stage.append(ConvBlock3D(self.opt.nc_im, self.N, 
                                            self.opt.ker_size, self.opt.padd_size, stride=1))
            for i in range(self.opt.num_layer):
                block = ConvBlock3D(self.N, self.N, 
                                    self.opt.ker_size, self.opt.padd_size, stride=1)
                _first_stage.append(block)
            _first_stage.append(nn.Conv3d(self.N, self.opt.nc_im, self.opt.ker_size, 
                                          stride=1, padding=self.opt.ker_size // 2, 
                                          weight_init=Normal(0.0, 0.02), pad_mode='pad', has_bias=True))
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
            x_prev_out_up = utils.upscale(x_prev_out, idx + 1, self.opt)

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
            # Model
            self.nfc = 64
            self.nc_im = 3
            self.ker_size = 3
            self.num_layer = 5
            self.latent_dim = 128
            self.enc_blocks = 2
            self.padd_size = 1
            # Dataset
            self.image_path = '../data/imgs/air_balloons.jpg'
            self.hflip = False
            self.img_size = 256
    
    opt = Opt()
    model = GeneratorVAE_nb(opt)
    model.init_next_stage()
    print(model)