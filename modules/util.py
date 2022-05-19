import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import Normal, One
from mindspore import dtype as mstype

concat = ops.Concat()


class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x)


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA=1):
    alpha = Tensor(shape=(1, 1), init=Normal(), dtype=mstype.float32)
    alpha = alpha.expand_as(Tensor(shape=(real_data.size()), 
                                   init=One(), dtype=mstype.float32))

    interpolates = (alpha * real_data + ((1 - alpha) * fake_data))
    # disc_interpolates = netD(interpolates)
    # gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
    #                                grad_outputs=torch.ones(disc_interpolates.size()).to(device),
    #                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = GradNetWrtX(netD)(interpolates)
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
