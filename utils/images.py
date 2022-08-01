import math
from mindspore import Tensor
import mindspore.ops as ops
from mindspore.common.initializer import Zero
from mindspore import dtype as mstype
import mindspore.nn.probability.distribution as msd

import numpy as np  # TODO: 替代三线性插值，待移除
import torch
import torch.nn.functional as F

import mindspore

__all__ = ['interpolate', 'interpolate_3D', 'adjust_scales2image', 'generate_noise', 'get_scales_by_index',
           'get_fps_td_by_index', 'get_fps_by_index', 'upscale', 'upscale_2d']
transpose = ops.Transpose()
bernoulli = msd.Bernoulli(0.5)
normal = msd.Normal(0, 1)
uniform = msd.Uniform(0, 1)
uniform_int = ops.UniformInt()


def interpolate(input, size=None):
    resize_bilinear = ops.ResizeBilinear(size, align_corners=True)    # TODO: align_corners

    if input.dim() == 5:
        b, c, t, h0, w0 = input.shape
        img = transpose(input, (0, 2, 1, 3, 4)).reshape(input.shape[0] + input.shape[1],
                                                        *input.shape[2:])  # (B+T)CHW
        scaled = resize_bilinear(img)
        _, _, h1, w1 = scaled.shape
        scaled = transpose(scaled.reshape(b, t, c, h1, w1), (0, 2, 1, 3, 4))
    else:
        scaled = resize_bilinear(input)

    return scaled


def interpolate_3D(input, size=None):
    if input.dim() != 5:
        exit(1)
    # resize_bilinear = ops.ResizeTrilinear(size, align_corners=True)
    # scaled = resize_bilinear(input)

    input = input.asnumpy()
    input = torch.Tensor(input)
    scaled = F.interpolate(input, size=size, align_corners=True).to(np.float32)
    scaled = Tensor(scaled)

    return scaled


def adjust_scales2image(size, opt):
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / size, 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, size]) / size, opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / size, 1)
    opt.scale_factor = math.pow(opt.min_size / size, 1 / opt.stop_scale)
    scale2stop = math.ceil(math.log(min([opt.max_size, size]) / size, opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop


def generate_noise(ref=None, size=None, type='normal', emb_size=None):
    # Initiate noise without batch size
    if ref is not None:
        noise = Tensor(shape=ref.shape, init=Zero(), dtype=mstype.float32)
    elif size is not None:
        noise = Tensor(shape=size, init=Zero(), dtype=mstype.float32)
    else:
        # raise Exception("ref or size must be applied")
        exit(1)

    if type == 'normal':
        return normal.prob(Tensor(shape=noise.shape, init=Zero(), dtype=mstype.float32))
    elif type == 'benoulli':
        return bernoulli.prob(Tensor(shape=noise.shape, init=Zero(), dtype=mstype.float32))

    if type == 'int':
        if emb_size is None or size is None:
            exit(1)
        return uniform_int(size, 0, emb_size)

    return uniform.prob(Tensor(shape=noise.shape, init=Zero(), dtype=mstype.float32))  # Default == Uniform


def get_scales_by_index(index, scale_factor, stop_scale, img_size):
    scale = math.pow(scale_factor, stop_scale - index)
    s_size = math.ceil(scale * img_size)

    return s_size


def get_fps_by_index(index, opt):
    # Linear fps interpolation by divisors
    fps_index = int((index / opt.stop_scale_time) * (len(opt.sampling_rates) - 1))

    return opt.org_fps / opt.sampling_rates[fps_index], fps_index


def get_fps_td_by_index(index, opt):
    fps, fps_index = get_fps_by_index(index, opt)

    every = opt.sampling_rates[fps_index]
    time_depth = opt.fps_lcm // every + 1

    return fps, time_depth, fps_index


def upscale(video, index, opt):
    if index <= 0:
        exit(1)

    next_shape = get_scales_by_index(index, opt.scale_factor, opt.stop_scale, opt.img_size)
    next_fps, next_td, _ = get_fps_td_by_index(index, opt)
    next_shape = [next_td, int(next_shape * opt.ar), next_shape]

    # Video interpolation
    vid_up = interpolate_3D(video, size=next_shape)

    return vid_up


def upscale_2d(image, index, opt):
    if index <= 0:
        exit(1)

    next_shape = get_scales_by_index(index, opt.scale_factor, opt.stop_scale, opt.img_size)
    next_shape = [int(next_shape * opt.ar), next_shape]

    # Video interpolation
    img_up = interpolate(image, size=next_shape)

    return img_up
