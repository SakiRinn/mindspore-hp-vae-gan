import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.ops import functional as F


def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class ClippedAdam(nn.Adam):
    def __init__(self, opt, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False,
                 use_nesterov=False, weight_decay=0.0, loss_scale=1.0):
        super(ClippedAdam, self).__init__(params, learning_rate, beta1, beta2,
                                          eps, use_locking, use_nesterov, weight_decay, loss_scale)
        self._opt = opt

    def construct(self, gradients):
        gradients = _clip_grad(1, self._opt.grad_clip, gradients)
        return super(ClippedAdam, self).construct(gradients)
