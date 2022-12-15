from __future__ import absolute_import, division, print_function
import numpy as np
import copy

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import constexpr
from mindspore.common.initializer import Normal
from mindspore import context

from .. import utils
from .. import tools


def get_activation(act):
    activations = {
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "prelu": nn.PReLU(),
        "selu": ops.SeLU()
    }
    return activations[act]


@constexpr(reuse_result=False)
def reparam(std_shape):
    return Tensor(np.random.normal(size=std_shape).astype('float32'))

@constexpr(reuse_result=False)
def reparam_pred(mu_shape):
    return Tensor(np.random.normal(size=mu_shape).astype('float32'))

@constexpr(reuse_result=False)
def reparam_bern(bern_shape):
    return Tensor(np.random.uniform(0, 1, size=bern_shape).astype('float32'))

@constexpr(reuse_result=False)
def reparam_pred_bern(bern_shape):
    return Tensor(np.random.binomial(1, 0.5, size=bern_shape).astype('float32'))


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
            self.append(tools.SpectualNormConv3d(in_channel, out_channel, kernel_size=ker_size,
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
        self.append(ConvBlock3DSN(in_channel, out_channel, ker_size, padding, stride))
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
            if not isinstance(out_dim, int):
                exit(1)
            output_dim = out_dim

        self._features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size, opt.ker_size // 2, 1,
                                         num_blocks=num_blocks)
        self._mu = ConvBlock3D(opt.nfc, output_dim, opt.ker_size,
                              opt.ker_size // 2, 1, bn=False, act=None)
        self._logvar = ConvBlock3D(opt.nfc, output_dim, opt.ker_size,
                                  opt.ker_size // 2, 1, bn=False, act=None)

    def construct(self, x):
        features = self._features(x)
        mu = self._mu(features)
        logvar = self._logvar(features)

        return mu, logvar


class Encode3DVAE_nb(nn.Cell):
    def __init__(self, opt, out_dim=None, num_blocks=2):
        super(Encode3DVAE_nb, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            if not isinstance(out_dim, int):
                exit(1)
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size, opt.ker_size // 2, 1, num_blocks=num_blocks)
        self._mu = ConvBlock3D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)
        self._logvar = ConvBlock3D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)
        self._bern = ConvBlock3D(opt.nfc, 1, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)

    def construct(self, x):
        reduce_mean = ops.ReduceMean(keep_dims=True)

        features = self.features(x)
        bern = ops.Sigmoid()(self._bern(features))
        features = bern * features
        mu = reduce_mean(self._mu(features), 2)     # nn.AdaptiveAvgPool3D(1)
        mu = reduce_mean(mu, 3)
        mu = reduce_mean(mu, 4)
        logvar = reduce_mean(self._logvar(features), 2)     # nn.AdaptiveAvgPool3D(1)
        logvar = reduce_mean(logvar, 3)
        logvar = reduce_mean(logvar, 4)

        return mu, logvar, bern


class Encode3DVAE1x1(nn.Cell):
    def __init__(self, opt, out_dim=None):
        super(Encode3DVAE1x1, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            if not isinstance(out_dim, int):
                exit(1)
            output_dim = out_dim

        self._features = FeatureExtractor(opt.nc_im, opt.nfc, 1, 0, 1, num_blocks=2)
        self._mu = ConvBlock3D(opt.nfc, output_dim, 1, 0, 1, bn=False, act=None)
        self._logvar = ConvBlock3D(opt.nfc, output_dim, 1, 0, 1, bn=False, act=None)

    def construct(self, x):
        features = self._features(x)
        mu = self._mu(features)
        logvar = self._logvar(features)

        return mu, logvar


class WDiscriminator3D(nn.Cell):
    def __init__(self, opt):
        super(WDiscriminator3D, self).__init__()

        N = int(opt.nfc)

        self.head = ConvBlock3DSN(opt.nc_im, N, opt.ker_size,
                                  opt.ker_size // 2, stride=1, bn=True, act='lrelu')

        # FIXME: Name BUG.
        self.body = nn.SequentialCell([ConvBlock3DSN(N, N, opt.ker_size,
                                       opt.ker_size // 2, stride=1, bn=True, act='lrelu')])
        for _ in range(opt.num_layer - 1):
            self.body.append(ConvBlock3DSN(N, N, opt.ker_size,
                                           opt.ker_size // 2, stride=1, bn=True, act='lrelu'))

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

        N = int(opt.nfc)
        self.p3d = ((0, 0), (0, 0),
                    (opt.num_layer + 2, opt.num_layer + 2),
                    (opt.num_layer + 2, opt.num_layer + 2),
                    (opt.num_layer + 2, opt.num_layer + 2))

        self.head = ConvBlock3D(opt.nc_im, N, opt.ker_size,
                                opt.padd_size, stride=1, bn=False, act='lrelu')

        # FIXME: Name BUG.
        self.body = nn.SequentialCell([ConvBlock3DSN(N, N, opt.ker_size,
                                       opt.ker_size // 2, stride=1, bn=True, act='lrelu')])
        for _ in range(opt.num_layer - 1):
            self.body.append(ConvBlock3DSN(N, N, opt.ker_size,
                                           opt.ker_size // 2, stride=1, bn=True, act='lrelu'))

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

        self.num_layer = opt.num_layer
        self.scale_factor = opt.scale_factor
        self.stop_scale = opt.stop_scale
        self.img_size = opt.img_size
        self.stop_scale_time = opt.stop_scale_time
        self.sampling_rates = opt.sampling_rates
        self.org_fps = opt.org_fps
        self.fps_lcm = opt.fps_lcm
        self.ar = opt.ar

        N = int(opt.nfc)

        self.p3d_once = ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1))
        self.p3d = ((0, 0), (0, 0),
                    (opt.num_layer + 0, opt.num_layer + 0),
                    (opt.num_layer + 0, opt.num_layer + 0),
                    (opt.num_layer + 0, opt.num_layer + 0))

        self.head = ConvBlock3D(opt.nc_im, N, opt.ker_size, padding=0, stride=1)

        _first_stage = nn.SequentialCell([ConvBlock3D(N, N, opt.ker_size, padding=0, stride=1)])
        for _ in range(opt.num_layer):
            _first_stage.append(ConvBlock3D(N, N, opt.ker_size, padding=0, stride=1))
        self.body = nn.CellList([_first_stage])

        self.tail = nn.SequentialCell([
            nn.Conv3d(N, opt.nc_im, kernel_size=opt.ker_size, padding=0, stride=1,
                      weight_init=Normal(0.02, 0.0), pad_mode='pad', has_bias=True),
            nn.Tanh()
        ])

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def construct(self, noise_init, noise_amp, isRandom=False):
        x = self.head(ops.Pad(self.p3d_once)(noise_init))
        x_prev_out = self.body[0](ops.Pad(self.p3d)(x))

        for idx, block in enumerate(self.body[1:], 1):
            # Upscale
            x_prev_out_up = utils.upscale(x_prev_out, idx, self.scale_factor, self.stop_scale, self.img_size,
                                          self.stop_scale_time, self.sampling_rates, self.org_fps, self.fps_lcm, self.ar)
            # Whether add noise
            if isRandom:
                # Yes - in random mode
                x_prev_out_up_2 = utils.interpolate_3D(x_prev_out, size=[
                    x_prev_out_up.shape[-3] + (self.num_layer + 0) * 2,
                    x_prev_out_up.shape[-2] + (self.num_layer + 0) * 2,
                    x_prev_out_up.shape[-1] + (self.num_layer + 0) * 2
                ])
                noise = utils.generate_noise_ref(x_prev_out_up_2.shape)
                x_prev = block(x_prev_out_up_2 + noise * noise_amp[idx])
            else:
                # No - in reconstruction mode
                x_prev = block(ops.Pad(self.p3d)(x_prev_out_up))
            x_prev_out = x_prev + x_prev_out_up

        out = self.tail(ops.Pad(self.p3d_once)(x_prev_out))
        return out


class GeneratorSG(nn.Cell):
    def __init__(self, opt):
        super(GeneratorSG, self).__init__()

        self.num_layer = opt.num_layer
        self.scale_factor = opt.scale_factor
        self.stop_scale = opt.stop_scale
        self.img_size = opt.img_size
        self.stop_scale_time = opt.stop_scale_time
        self.sampling_rates = opt.sampling_rates
        self.org_fps = opt.org_fps
        self.fps_lcm = opt.fps_lcm
        self.ar = opt.ar

        N = int(opt.nfc)

        self.p3d = ((0, 0), (0, 0),
                    (self.opt.num_layer + 2, self.opt.num_layer + 2),
                    (self.opt.num_layer + 2, self.opt.num_layer + 2),
                    (self.opt.num_layer + 2, self.opt.num_layer + 2))

        _first_stage = nn.SequentialCell([ConvBlock3D(opt.nc_im, N, opt.ker_size, padding=0, stride=1)])
        for _ in range(opt.num_layer):
            _first_stage.append(ConvBlock3D(N, N, opt.ker_size, padding=0, stride=1))
        _first_stage.append(nn.Conv3d(N, opt.nc_im, kernel_size=opt.ker_size,
                                      padding=0, stride=1, weight_init=Normal(0.02, 0.0)))
        self.body = nn.CellList([_first_stage])

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def construct(self, noise_init, noise_amp, isRandom=False):
        pad_op = ops.Pad(self.p3d)
        x_prev_out = self.body[0](pad_op(noise_init))

        for idx, block in enumerate(self.body[1:], 1):
            x_prev_out = ops.Tanh()(x_prev_out)

            # Upscale
            x_prev_out_up = utils.upscale(x_prev_out, idx, self.scale_factor, self.stop_scale, self.img_size,
                                          self.stop_scale_time, self.sampling_rates, self.org_fps, self.fps_lcm, self.ar)

            # Whether add noise
            if isRandom:
                # Yes - in random mode
                x_prev_out_up_2 = utils.interpolate_3D(x_prev_out, size=[
                    x_prev_out_up.shape[-3] + (self.opt.num_layer + 2) * 2,
                    x_prev_out_up.shape[-2] + (self.opt.num_layer + 2) * 2,
                    x_prev_out_up.shape[-1] + (self.opt.num_layer + 2) * 2
                ])
                noise = utils.generate_noise_ref(x_prev_out_up_2.shape)
                x_prev = block(x_prev_out_up_2 + noise * noise_amp[idx])
            else:
                # No - in reconstruction mode
                x_prev = block(pad_op(x_prev_out_up))

        out = ops.Tanh()(x_prev + x_prev_out_up)
        return out


class GeneratorHPVAEGAN(nn.Cell):
    def __init__(self, opt, is_training=False):
        super(GeneratorHPVAEGAN, self).__init__()

        self.opt = opt
        self.scale_factor = opt.scale_factor
        self.stop_scale = opt.stop_scale
        self.img_size = opt.img_size
        self.stop_scale_time = opt.stop_scale_time
        self.sampling_rates = opt.sampling_rates
        self.org_fps = opt.org_fps
        self.fps_lcm = opt.fps_lcm
        self.ar = opt.ar
        self.vae_levels = opt.vae_levels
        self.train_all = opt.train_all

        N = int(opt.nfc)
        self.N = N
        self.is_training = is_training

        self.encode = Encode3DVAE(opt, out_dim=opt.latent_dim, num_blocks=opt.enc_blocks)

        # Normal Decoder
        decoder = nn.SequentialCell([ConvBlock3D(opt.latent_dim, N, opt.ker_size, opt.padd_size, stride=1)])
        for _ in range(opt.num_layer):
            decoder.append(ConvBlock3D(N, N, opt.ker_size, opt.padd_size, stride=1))
        decoder.append(nn.Conv3d(N, opt.nc_im, opt.ker_size, stride=1, padding=opt.ker_size // 2,
                                 weight_init=Normal(0.02, 0.0), pad_mode='pad', has_bias=True))
        self.decoder = decoder

        # 1x1 Decoder
        # self.decoder.append(ConvBlock3D(opt.latent_dim, N, 1, 0, stride=1))
        # for i in range(opt.num_layer):
        #     block = ConvBlock3D(N, N, 1, 0, stride=1)
        #     self.decoder.append(block)
        # self.decoder.append(nn.Conv3d(N, opt.nc_im, 1, 1, 0))

        self.body = nn.CellList([])

    def init_next_stage(self):
        if len(self.body) == 0:
            _first_stage = nn.SequentialCell([ConvBlock3D(self.opt.nc_im, self.N, self.opt.ker_size,
                                                          self.opt.padd_size, stride=1)])
            for _ in range(self.opt.num_layer):
                _first_stage.append(ConvBlock3D(self.N, self.N, self.opt.ker_size, self.opt.padd_size, stride=1))
            _first_stage.append(nn.Conv3d(self.N, self.opt.nc_im, self.opt.ker_size,
                                          stride=1, padding=self.opt.ker_size // 2,
                                          weight_init=Normal(0.02, 0.0), pad_mode='pad', has_bias=True))
            self.body = nn.CellList([_first_stage])    # FIXME: Init BUG.
        else:
            self.body.append(copy.deepcopy(self.body[-1]))

    def construct(self, video, noise_amp, noise_init=None, sample_init=None, isRandom=False):
        if sample_init is not None:
            if len(self.body) <= sample_init[0]:
                exit(1)

        mu, logvar = None, None
        if noise_init is None:
            mu, logvar = self.encode(video)
            if self.is_training:
                std = ops.Exp()(logvar * 0.5)
                eps = reparam(std.shape)
                z_vae = ops.Mul()(eps, std) + mu
            else:
                z_vae = reparam_pred(mu.shape)
        else:
            z_vae = noise_init

        vae_out = ops.Tanh()(self.decoder(z_vae))

        if sample_init is None:
            x_prev_out = self.refinement_layers(0, vae_out, noise_amp, isRandom)
        else:
            x_prev_out = self.refinement_layers(sample_init[0], sample_init[1], noise_amp, isRandom)

        if noise_init is None:
            return x_prev_out, vae_out, mu, logvar
        return x_prev_out, vae_out

    def refinement_layers(self, start_idx, x_prev_out, noise_amp, isRandom=False):
        x_prev_out_up = 0
        for idx, block in enumerate(self.body[start_idx:], start_idx):
            if self.vae_levels == idx + 1 and not self.train_all:
                x_prev_out = ops.stop_gradient(x_prev_out)
            # Upscale
            x_prev_out_up = utils.upscale(x_prev_out, idx + 1, self.scale_factor, self.stop_scale, self.img_size,
                                          self.stop_scale_time, self.sampling_rates, self.org_fps, self.fps_lcm, self.ar)
            # Whether add noise
            if isRandom and self.opt.vae_levels <= idx + 1:
                # Yes - in random mode
                noise = utils.generate_noise_ref(x_prev_out_up.shape)
                x_prev = block(x_prev_out_up + noise * noise_amp[idx + 1])
            else:
                # No - in reconstruction mode
                x_prev = block(x_prev_out_up)
            x_prev_out = ops.Tanh()(x_prev + x_prev_out_up)
        return x_prev_out


class GeneratorVAE_nb(nn.Cell):
    def __init__(self, opt, is_training=False):
        super(GeneratorVAE_nb, self).__init__()

        self.opt = opt
        self.scale_factor = opt.scale_factor
        self.stop_scale = opt.stop_scale
        self.img_size = opt.img_size
        self.stop_scale_time = opt.stop_scale_time
        self.sampling_rates = opt.sampling_rates
        self.org_fps = opt.org_fps
        self.fps_lcm = opt.fps_lcm
        self.ar = opt.ar
        self.vae_levels = opt.vae_levels

        N = int(opt.nfc)
        self.N = N
        self.is_training = is_training

        self.encode = Encode3DVAE_nb(opt, out_dim=opt.latent_dim, num_blocks=opt.enc_blocks)

        # Normal Decoder
        decoder = nn.SequentialCell([ConvBlock3D(opt.latent_dim, N, opt.ker_size, opt.padd_size, stride=1)])
        for _ in range(opt.num_layer):
            decoder.append(ConvBlock3D(N, N, opt.ker_size, opt.padd_size, stride=1))
        decoder.append(nn.Conv3d(N, opt.nc_im, opt.ker_size, stride=1, padding=opt.ker_size // 2,
                                 weight_init=Normal(0.02, 0.0), pad_mode='pad', has_bias=True))
        self.decoder = decoder

        self.body = nn.CellList([])

    def init_next_stage(self):
        if len(self.body) == 0:
            _first_stage = nn.SequentialCell([ConvBlock3D(self.opt.nc_im, self.N, self.opt.ker_size,
                                                          self.opt.padd_size, stride=1)])
            for _ in range(self.opt.num_layer):
                _first_stage.append(ConvBlock3D(self.N, self.N, self.opt.ker_size, self.opt.padd_size, stride=1))
            _first_stage.append(nn.Conv3d(self.N, self.opt.nc_im, self.opt.ker_size,
                                          stride=1, padding=self.opt.ker_size // 2,
                                          weight_init=Normal(0.02, 0.0), pad_mode='pad', has_bias=True))
            self.body = nn.CellList([_first_stage])    # FIXME: Init BUG.
        else:
            self.body.append(copy.deepcopy(self.body[-1]))

    def construct(self, video, noise_amp,
                  noise_init_norm=None, noise_init_bern=None, sample_init=None, randMode=False):
        if sample_init is not None:
            if len(self.body) <= sample_init[0]:
                exit(1)

        mu, logvar, bern = None, None, None
        if noise_init_norm is None:
            mu, logvar, bern = self.encode(video)
            if self.is_training:
                # Norm
                std = ops.Exp()(logvar * 0.5)
                eps = reparam(std.shape)
                z_vae_norm = ops.Mul()(eps, std) + mu
                # Bern
                log = ops.Log()
                eps = reparam_bern(bern.shape)
                z_vae_bern = log(bern + 1e-20) - log(-log(eps + 1e-20) + 1e-20)
            else:
                z_vae_norm = reparam_pred(mu.shape)
                z_vae_bern = reparam_pred_bern(bern.shape)
        else:
            z_vae_norm = noise_init_norm
            z_vae_bern = noise_init_bern

        vae_out = ops.Tanh()(self.decoder(z_vae_norm * z_vae_bern))

        if sample_init is None:
            x_prev_out = self.refinement_layers(0, vae_out, noise_amp, randMode)
        else:
            x_prev_out = self.refinement_layers(sample_init[0], sample_init[1], noise_amp, randMode)

        if noise_init_norm is None:
            return x_prev_out, vae_out, mu, logvar, bern
        return x_prev_out, vae_out

    def refinement_layers(self, start_idx, x_prev_out, noise_amp, randMode=False):
        x_prev_out_up = 0
        for idx, block in enumerate(self.body[start_idx:], start_idx):
            if self.vae_levels == idx + 1:
                x_prev_out = ops.stop_gradient(x_prev_out)
            # Upscale
            x_prev_out_up = utils.upscale(x_prev_out, idx + 1, self.scale_factor, self.stop_scale, self.img_size,
                                          self.stop_scale_time, self.sampling_rates, self.org_fps, self.fps_lcm, self.ar)
            # Whether add noise
            if randMode:
                # Yes - in random mode
                noise = utils.generate_noise_ref(x_prev_out_up.shape)
                x_prev = block(x_prev_out_up + noise * noise_amp[idx + 1])
            else:
                # No - in reconstruction mode
                x_prev = block(x_prev_out_up)
            x_prev_out = ops.Tanh()(x_prev + x_prev_out_up)
        return x_prev_out


if __name__ == '__main__':
    import cv2
    import numpy as np

    context.set_context(device_id=6, mode=1)

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
            self.scale_factor = 0.75
            self.stop_scale = 9
            self.stop_scale_time = 9
            self.scale_idx = 0
            self.vae_levels = 3
            self.sampling_rates = [4, 3, 2, 1]
            capture = cv2.VideoCapture(self.video_path)
            self.org_fps = capture.get(cv2.CAP_PROP_FPS)
            self.fps_lcm = np.lcm.reduce(self.sampling_rates)
            self.Noise_Amps = [1, 1, 1]
            self.ar = 1
            self.train_all = True

    opt = Opt()
    model = GeneratorHPVAEGAN(opt)
    for i in range(1):
        model.init_next_stage()
    from mindspore.common.initializer import One
    x = Tensor(shape=(8, 3, 4, 2, 2), init=One(), dtype=mstype.float32)
    y = model(x, opt.Noise_Amps)
    print(y[0].shape, y[1].shape)
