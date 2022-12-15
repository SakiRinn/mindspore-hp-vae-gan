import math
import numpy as np

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import constexpr
import mindspore.ops as ops
from mindspore.common.initializer import Zero

from ..tools import UpsampleTrilinear3D

__all__ = ['interpolate', 'interpolate_3D', 'adjust_scales2image',
           'generate_noise_size', 'generate_noise_ref','get_scales_by_index',
           'get_fps_td_by_index', 'get_fps_by_index', 'upscale', 'upscale_2d']


@constexpr(reuse_result=False)
def generate_noise_size(size=None, type='normal', emb_size=None):
    noise = Tensor(shape=size, init=Zero(), dtype=mstype.float32)
    if type == 'normal':
        return Tensor(np.random.normal(size=noise.shape).astype('float32'))
    elif type == 'benoulli':
        return Tensor(np.random.binomial(1, 0.5, size=noise.shape).astype('float32'))
    elif type == 'int':
        if emb_size is None or size is None:
            exit(1)
        return ops.UniformInt()(size, 0, emb_size)
    return Tensor(np.random.uniform(0, 1, size=noise.shape).astype('float32'))

@constexpr(reuse_result=False)
def generate_noise_ref(ref_shape, type='normal'):
    noise = Tensor(shape=ref_shape, init=Zero(), dtype=mstype.float32)
    if type == 'normal':
        return Tensor(np.random.normal(size=noise.shape).astype('float32'))
    elif type == 'benoulli':
        return Tensor(np.random.binomial(1, 0.5, size=noise.shape).astype('float32'))
    return Tensor(np.random.uniform(0, 1, size=noise.shape).astype('float32'))


def interpolate(input, size=None):
    resize_bilinear = ops.ResizeBilinear(size, align_corners=True)
    if input.ndim == 5:
        b, c, t, h0, w0 = input.shape
        img = input.transpose(0, 2, 1, 3, 4).reshape(input.shape[0] + input.shape[1], *input.shape[2:])  # (B+T)CHW
        scaled = resize_bilinear(img)
        _, _, h1, w1 = scaled.shape
        scaled = scaled.reshape(b, t, c, h1, w1).transpose(0, 2, 1, 3, 4)
    else:
        scaled = resize_bilinear(input)

    return scaled


def interpolate_3D(input, size):
    if input.ndim != 5:
        exit(1)
    size = tuple([int(v) for v in size])
    resize_trilinear = UpsampleTrilinear3D(size, align_corners=True)
    scaled = resize_trilinear(input)

    return scaled


def adjust_scales2image(size, opt):
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / size, 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, size]) / size, opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / size, 1)
    opt.scale_factor = math.pow(opt.min_size / size, 1 / opt.stop_scale)
    scale2stop = math.ceil(math.log(min([opt.max_size, size]) / size, opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop


def get_scales_by_index(index, scale_factor, stop_scale, img_size):
    scale = math.pow(scale_factor, stop_scale - index) + 1e-6
    s_size = math.ceil(scale * img_size)
    return s_size


def get_fps_by_index(index, stop_scale_time, sampling_rates, org_fps):
    # Linear fps interpolation by divisors
    fps_index = int((index / stop_scale_time) * (len(sampling_rates) - 1))

    return org_fps / sampling_rates[fps_index], fps_index


def get_fps_td_by_index(index, stop_scale_time, sampling_rates, org_fps, fps_lcm):
    fps, fps_index = get_fps_by_index(index, stop_scale_time, sampling_rates, org_fps)

    every = sampling_rates[fps_index]
    time_depth = fps_lcm // every + 1

    return fps, time_depth, fps_index


def upscale(video, index, scale_factor, stop_scale, img_size,
            stop_scale_time, sampling_rates, org_fps, fps_lcm, ar):
    if index <= 0:
        exit(1)
    next_shape = get_scales_by_index(index, scale_factor, stop_scale, img_size)
    next_fps, next_td, _ = get_fps_td_by_index(index, stop_scale_time, sampling_rates, org_fps, fps_lcm)
    next_shape = [next_td, int(next_shape * ar), next_shape]

    # Video interpolation
    vid_up = interpolate_3D(video, size=next_shape)

    return vid_up


def upscale_2d(image, index, scale_factor, stop_scale, img_size, ar):
    if index <= 0:
        exit(1)
    next_shape = get_scales_by_index(index, scale_factor, stop_scale, img_size)
    next_shape = [int(next_shape * ar), next_shape]

    # Video interpolation
    img_up = interpolate(image, size=next_shape)

    return img_up


if __name__ == '__main__':
    import mindspore.context as context
    context.set_context(mode=1, device_id=5)
    image = Tensor(np.random.binomial(1, 0.5, size=(2, 3, 38, 51)).astype('float32'))
    print(get_scales_by_index(3, 0.7937005259840998, 9, 256))
