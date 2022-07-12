from __future__ import absolute_import, division, print_function

import copy
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.common.initializer import Normal, Zero

import sys
sys.path.insert(0, '..')
sys.path.append('../datasets')
import datasets
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
        # "selu": ops.SeLU()
    }
    return activations[act]


def reparameterize(mu, logvar, training):
    if training:
        std = exp(logvar * 0.5)
        eps = Tensor(shape=std.shape, init=Normal(), dtype=mstype.float32)
        return matmul(eps, std) + (mu)
    else:
        return Tensor(shape=mu.shape, init=Normal(), dtype=mstype.float32)


def reparameterize_bern(x, training):
    if training:
        eps = uniform.prob(Tensor(shape=x.shape, init=Zero(), dtype=mstype.float32))
        return log(x + 1e-20) - log(-log(eps + 1e-20) + 1e-20)
    else:
        return bernoulli.prob(Tensor(shape=x.shape, init=Zero(), dtype=mstype.float32))


# Basic blocks

class ConvBlock3D(nn.SequentialCell):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu'):
        super(ConvBlock3D, self).__init__()
        self.append(nn.Conv3d(in_channel, out_channel, kernel_size=ker_size,
                              stride=stride, padding=padding, weight_init=Normal(0.02, 0.0),
                              pad_mode='pad', has_bias=True))
        if bn:
            self.append(nn.BatchNorm3d(out_channel, gamma_init=Normal(0.02, 1.0)))
        if act is not None:
            self.append(get_activation(act))


class ConvBlock3DSN(nn.SequentialCell):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu'):
        super(ConvBlock3DSN, self).__init__()
        if bn:
            self.append(utils.SpectualNormConv3d(in_channel, out_channel, kernel_size=ker_size,
                                                 stride=stride, padding=padding, weight_init=Normal(0.02, 0.0),
                                                 pad_mode='pad', has_bias=True))
        else:
            paddings = ((0, 0), (0, 0),
                        (padding, padding), (padding, padding),
                        (padding, padding), (padding, padding))
            self.append(nn.Pad(paddings=paddings, mode='REFLECT'))
            self.append(nn.Conv3d(in_channel, out_channel, kernel_size=ker_size,
                                  stride=stride, weight_init=Normal(0.02, 0.0),
                                  pad_mode='valid', has_bias=False))
        if act is not None:
            self.append(get_activation(act))


class FeatureExtractor(nn.SequentialCell):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride,
                 num_blocks=2, return_linear=False):
        super(FeatureExtractor, self).__init__()
        self.append(ConvBlock3DSN(in_channel, out_channel, ker_size, padding, stride)),
        for _ in range(num_blocks - 1):
            self.append(ConvBlock3DSN(out_channel, out_channel, ker_size, padding, stride))
        if return_linear:
            self.append(ConvBlock3DSN(out_channel, out_channel, ker_size, padding, stride, bn=False, act=None))
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

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size, opt.ker_size // 2, 1,
                                         num_blocks=num_blocks)
        self.mu = ConvBlock3D(opt.nfc, output_dim, opt.ker_size,
                              opt.ker_size // 2, 1, bn=False, act=None)
        self.logvar = ConvBlock3D(opt.nfc, output_dim, opt.ker_size,
                                  opt.ker_size // 2, 1, bn=False, act=None)

    def construct(self, x):
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

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size,
                                         opt.ker_size // 2, 1, num_blocks=num_blocks)
        self.mu = ConvBlock3D(opt.nfc, output_dim, opt.ker_size,
                              opt.ker_size // 2, 1, bn=False, act=None)
        self.logvar = ConvBlock3D(opt.nfc, output_dim, opt.ker_size,
                                  opt.ker_size // 2, 1, bn=False, act=None)

        self.bern = ConvBlock3D(opt.nfc, 1, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)

    def construct(self, x):
        reduce_mean = ops.ReduceMean(keep_dims=True)

        features = self.features(x)
        bern = sigmoid(self.bern(features))
        features = bern * features
        mu = reduce_mean(self.mu(features), 2)     # nn.AdaptiveAvgPool3D(1)
        mu = reduce_mean(mu, 3)
        mu = reduce_mean(mu, 4)
        logvar = reduce_mean(self.logvar(features), 2)     # nn.AdaptiveAvgPool3D(1)
        logvar = reduce_mean(logvar, 3)
        logvar = reduce_mean(logvar, 4)

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

    def construct(self, x):
        features = self.features(x)
        mu = self.mu(features)
        logvar = self.logvar(features)

        return mu, logvar


class WDiscriminator3D(nn.Cell):
    def __init__(self, opt):
        super(WDiscriminator3D, self).__init__()

        self.opt = opt
        N = int(opt.nfc)

        self.head = ConvBlock3DSN(opt.nc_im, N, opt.ker_size, opt.ker_size // 2,
                                  stride=1, bn=True, act='lrelu')

        body = []
        for _ in range(opt.num_layer):
            body.append(ConvBlock3DSN(N, N, opt.ker_size, opt.ker_size // 2,
                                      stride=1, bn=True, act='lrelu'))
        self.body = nn.SequentialCell(body)

        self.tail = nn.Conv3d(N, 1, kernel_size=opt.ker_size, padding=1, stride=1,
                              weight_init=Normal(0.02, 0.0), pad_mode='pad', has_bias=True)

    def construct(self, x):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out


class WDiscriminatorBaselines(nn.Cell):
    def __init__(self, opt):
        super(WDiscriminatorBaselines, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.p3d = ((0, 0), (0, 0),
                    (self.opt.num_layer + 2, self.opt.num_layer + 2),
                    (self.opt.num_layer + 2, self.opt.num_layer + 2),
                    (self.opt.num_layer + 2, self.opt.num_layer + 2))

        self.head = ConvBlock3D(opt.nc_im, N, opt.ker_size, opt.padd_size,
                                stride=1, bn=False, act='lrelu')

        body = []
        for _ in range(opt.num_layer):
            body.append(ConvBlock3D(N, N, opt.ker_size, opt.padd_size,
                                    stride=1, bn=True, act='lrelu'))
        self.body = nn.SequentialCell(body)

        self.tail = nn.Conv3d(N, 1, kernel_size=opt.ker_size, padding=opt.padd_size, stride=1,
                              weight_init=Normal(0.02, 0.0), pad_mode='pad', has_bias=True)

    def construct(self, x):
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

        self.p3d_once = ((0, 0), (0, 0),
                         (1, 1), (1, 1), (1, 1))
        self.p3d = ((0, 0), (0, 0),
                    (self.opt.num_layer + 0, self.opt.num_layer + 0),
                    (self.opt.num_layer + 0, self.opt.num_layer + 0),
                    (self.opt.num_layer + 0, self.opt.num_layer + 0))

        self.head = ConvBlock3D(opt.nc_im, N, opt.ker_size, padding=0, stride=1)

        self.body = nn.CellList([])
        _first_stage = nn.SequentialCell()
        for _ in range(opt.num_layer):
            block = ConvBlock3D(N, N, opt.ker_size, padding=0, stride=1)
            _first_stage.append(block)
        self.body.append(_first_stage)

        tail = []
        tail.append(nn.Conv3d(N, opt.nc_im, kernel_size=opt.ker_size, padding=0, stride=1,
                              weight_init=Normal(0.02, 0.0), pad_mode='pad', has_bias=True))
        tail.append(nn.Tanh())
        self.tail = nn.SequentialCell(tail)


    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def construct(self, noise_init, noise_amp, mode='rand'):
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

        self.p3d = ((0, 0), (0, 0),
                    (self.opt.num_layer + 2, self.opt.num_layer + 2),
                    (self.opt.num_layer + 2, self.opt.num_layer + 2),
                    (self.opt.num_layer + 2, self.opt.num_layer + 2))

        self.body = nn.CellList([])

        _first_stage = nn.SequentialCell()
        _first_stage.append(ConvBlock3D(opt.nc_im, N, opt.ker_size, padding=0, stride=1))
        for _ in range(opt.num_layer):
            block = ConvBlock3D(N, N, opt.ker_size, padding=0, stride=1)
            _first_stage.append(block)
        _first_stage.append(nn.Conv3d(N, opt.nc_im, kernel_size=opt.ker_size,
                                      padding=0, stride=1, weight_init=Normal(0.02, 0.0)))
        self.body.append(_first_stage)


    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def construct(self, noise_init, noise_amp, mode='rand'):
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

        # Normal Decoder
        decoder = []
        decoder.append(ConvBlock3D(opt.latent_dim, N, opt.ker_size, opt.padd_size, stride=1))
        for _ in range(opt.num_layer):
            decoder.append(ConvBlock3D(N, N, opt.ker_size, opt.padd_size, stride=1))
        decoder.append(nn.Conv3d(N, opt.nc_im, opt.ker_size, stride=1, padding=opt.ker_size // 2,
                                 weight_init=Normal(0.02, 0.0), pad_mode='pad', has_bias=True))
        self.decoder = nn.SequentialCell(decoder)

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
            for _ in range(self.opt.num_layer):
                block = ConvBlock3D(self.N, self.N, self.opt.ker_size, self.opt.padd_size, stride=1)
                _first_stage.append(block)
            _first_stage.append(nn.Conv3d(self.N, self.opt.nc_im, self.opt.ker_size,
                                          stride=1, padding=self.opt.ker_size // 2,
                                          weight_init=Normal(0.02, 0.0), pad_mode='pad', has_bias=True))
            self.body.append(_first_stage)
        else:
            self.body.append(copy.deepcopy(self.body[-1]))

    def construct(self, video, noise_amp, noise_init=None, sample_init=None, randMode=False):
        if sample_init is not None:
            if len(self.body) <= sample_init[0]:
                exit(1)

        if noise_init is None:
            mu, logvar = self.encode(video)
            z_vae = reparameterize(mu, logvar, self.training)
        else:
            z_vae = noise_init

        vae_out = ops.tanh(self.decoder(z_vae))

        if sample_init is not None:
            x_prev_out = self.refinement_layers(sample_init[0], sample_init[1], noise_amp, randMode)
        else:
            x_prev_out = self.refinement_layers(0, vae_out, noise_amp, randMode)

        if noise_init is None:
            return x_prev_out, vae_out, (mu, logvar)
        else:
            return x_prev_out, vae_out

    def refinement_layers(self, start_idx, x_prev_out, noise_amp, randMode=False):
        for idx, block in enumerate(self.body[start_idx:], start_idx):
            if self.opt.vae_levels == idx + 1 and not self.opt.train_all:
                x_prev_out = ops.stop_gradient(x_prev_out)

            # Upscale
            x_prev_out_up = utils.upscale(x_prev_out, idx + 1, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if randMode and self.opt.vae_levels <= idx + 1:
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

        # Normal Decoder
        decoder = []
        decoder.append(ConvBlock3D(opt.latent_dim, N, opt.ker_size, opt.padd_size, stride=1))
        for _ in range(opt.num_layer):
            block = ConvBlock3D(N, N, opt.ker_size, opt.padd_size, stride=1)
            decoder.append(block)
        decoder.append(nn.Conv3d(N, opt.nc_im, opt.ker_size, stride=1, padding=opt.ker_size // 2,
                                 weight_init=Normal(0.02, 0.0), pad_mode='pad', has_bias=True))
        self.decoder = nn.SequentialCell(decoder)

        self.body = nn.CellList([])

    def init_next_stage(self):
        if len(self.body) == 0:
            _first_stage = nn.SequentialCell()
            _first_stage.append(ConvBlock3D(self.opt.nc_im, self.N,
                                            self.opt.ker_size, self.opt.padd_size, stride=1))
            for _ in range(self.opt.num_layer):
                block = ConvBlock3D(self.N, self.N,
                                    self.opt.ker_size, self.opt.padd_size, stride=1)
                _first_stage.append(block)
            _first_stage.append(nn.Conv3d(self.N, self.opt.nc_im, self.opt.ker_size,
                                          stride=1, padding=self.opt.ker_size // 2,
                                          weight_init=Normal(0.02, 0.0), pad_mode='pad', has_bias=True))
            self.body.append(_first_stage)
        else:
            self.body.append(copy.deepcopy(self.body[-1]))

    def construct(self, video, noise_amp,
                  noise_init_norm=None, noise_init_bern=None, sample_init=None, randMode=False):
        if sample_init is not None:
            if len(self.body) <= sample_init[0]:
                exit(1)

        if noise_init_norm is None:
            mu, logvar, bern = self.encode(video)
            z_vae_norm = reparameterize(mu, logvar, self.training)
            z_vae_bern = reparameterize_bern(bern, self.training)
        else:
            z_vae_norm = noise_init_norm
            z_vae_bern = noise_init_bern

        vae_out = tanh(self.decoder(z_vae_norm * z_vae_bern))

        if sample_init is not None:
            x_prev_out = self.refinement_layers(sample_init[0], sample_init[1], noise_amp, randMode)
        else:
            x_prev_out = self.refinement_layers(0, vae_out, noise_amp, randMode)

        if noise_init_norm is None:
            return x_prev_out, vae_out, (mu, logvar, bern)
        else:
            return x_prev_out, vae_out

    def refinement_layers(self, start_idx, x_prev_out, noise_amp, randMode=False):
        for idx, block in enumerate(self.body[start_idx:], start_idx):
            if self.opt.vae_levels == idx + 1:
                x_prev_out = ops.stop_gradient(x_prev_out)

            # Upscale
            x_prev_out_up = utils.upscale(x_prev_out, idx + 1, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if randMode:
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
            self.image_path = '../data/imgs/air_balloons.jpg'
            self.video_path = '../data/vids/air_balloons.mp4'
            self.hflip = True
            self.img_size = 256
            self.data_rep = 1000
            self.scale_factor = 0.75
            self.stop_scale = 9
            self.stop_scale_time = 9
            self.scale_idx = 0
            self.vae_levels = 3
            self.sampling_rates = [4, 3, 2, 1]
            import cv2
            import numpy as np
            capture = cv2.VideoCapture(self.video_path)
            self.org_fps = capture.get(cv2.CAP_PROP_FPS)
            self.fps_lcm = np.lcm.reduce(self.sampling_rates)
            self.Noise_Amps = [1, 1, 1]

    opt = Opt()
    dataset = datasets.SingleImageDataset(opt)
    sn = ConvBlock3DSN(opt.nc_im, int(opt.nfc), opt.ker_size, opt.ker_size // 2,
                       stride=1, bn=True, act='lrelu')
    model = GeneratorCSG(opt)
    model.init_next_stage()
    from mindspore.common.initializer import One
    x = Tensor(shape=(64, 3, 3, 3, 3), init=One(), dtype=mstype.float32)
    print(model(x, opt.Noise_Amps))
