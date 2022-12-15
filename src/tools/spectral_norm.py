import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.layer.conv import _check_input_3d, _check_input_5dims

stdnormal = ops.StandardNormal(seed=43)
l2normalize = ops.L2Normalize(epsilon=1e-12)


class SpectualNormConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 data_format='NCHW',
                 power_iterations=1):
        super(SpectualNormConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_bias,
            weight_init,
            bias_init,
            data_format)
        self.power_iterations = power_iterations
        height = self.weight.shape[0]
        width = self.weight.view(height, -1).shape[1]
        self.weight_u = mindspore.Parameter(l2normalize(stdnormal((height,1))), requires_grad=False)
        self.weight_v = mindspore.Parameter(l2normalize(stdnormal((width,1))), requires_grad=False)
    
    def construct(self, x):
        height = self.weight.shape[0]
        for _ in range(self.power_iterations):
            self.weight_v = l2normalize(ops.tensor_dot(self.weight.view(height, -1).T, self.weight_u, axes=1))
            self.weight_u = l2normalize(ops.tensor_dot(self.weight.view(height, -1), self.weight_v, axes=1))
        sigma = ops.tensor_dot(self.weight_u.T, self.weight.view(height, -1), axes=1)
        sigma = ops.tensor_dot(sigma, self.weight_v, axes=1)
        weight = self.weight / sigma
        output = self.conv2d(x, weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output


class SpectualNormConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 data_format='NCHW',
                 power_iterations=1):
        super(SpectualNormConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_bias,
            weight_init,
            bias_init)
        self.power_iterations = power_iterations
        height = self.weight.shape[0]
        width = self.weight.view(height, -1).shape[1]
        self.weight_u = mindspore.Parameter(l2normalize(stdnormal((height,1))), requires_grad=False)
        self.weight_v = mindspore.Parameter(l2normalize(stdnormal((width,1))), requires_grad=False)
    
    def construct(self, x):
        # x_shape = self.shape(x)
        # _check_input_3d(x_shape)
        x = self.expand_dims(x, 2)
        height = self.weight.shape[0]
        for _ in range(self.power_iterations):
            self.weight_v = l2normalize(ops.tensor_dot(self.weight.view(height, -1).T, self.weight_u, axes=1))
            self.weight_u = l2normalize(ops.tensor_dot(self.weight.view(height, -1), self.weight_v, axes=1))
        sigma = ops.tensor_dot(self.weight_u.T, self.weight.view(height, -1), axes=1)
        sigma = ops.tensor_dot(sigma, self.weight_v, axes=1)
        weight = self.weight / sigma
        output = self.conv2d(x, weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        output = self.squeeze(output)
        return output
    
    
class SpectualNormConv3d(nn.Conv3d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 data_format='NCHW',
                 power_iterations=1):
        super(SpectualNormConv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_bias,
            weight_init,
            bias_init)
        self.power_iterations = power_iterations
        height = self.weight.shape[0]
        width = self.weight.view(height, -1).shape[1]
        self.weight_u = mindspore.Parameter(l2normalize(stdnormal((height,1))), requires_grad=False)
        self.weight_v = mindspore.Parameter(l2normalize(stdnormal((width,1))), requires_grad=False)
    
    def construct(self, x):
        # x_shape = self.shape(x)
        # _check_input_5dims(x_shape)
        height = self.weight.shape[0]
        for _ in range(self.power_iterations):
            self.weight_v = l2normalize(ops.tensor_dot(self.weight.view(height, -1).T, self.weight_u, axes=1))
            self.weight_u = l2normalize(ops.tensor_dot(self.weight.view(height, -1), self.weight_v, axes=1))
        sigma = ops.tensor_dot(self.weight_u.T, self.weight.view(height, -1), axes=1)
        sigma = ops.tensor_dot(sigma, self.weight_v, axes=1)
        weight = self.weight / sigma
        output = self.conv3d(x, weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output
    

if __name__=="__main__":
    from mindspore import context
    import numpy as np
    
    class withLoss(nn.Cell):
        def __init__(self, model):
            super(withLoss, self).__init__(auto_prefix=False)
            self.model = model

        def construct(self, x, y):
            y_hat = self.model(x)
            loss = mindspore.ops.absolute(y_hat-y).mean()
            return loss
    
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=7)
    mindspore.set_seed(0)
    np.random.seed(0)
    model = SpectualNormConv1d(in_channels=1, out_channels=2, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
    x = mindspore.Tensor(np.random.randn(2,1,5), dtype=mindspore.float32)
    y = mindspore.Tensor(np.random.randn(2,2,5), dtype=mindspore.float32)
    opt = nn.optim.SGD(model.trainable_params(), learning_rate=0.01)
    print(model.trainable_params())
    lmodel = withLoss(model)
    TrainOneStepCell = nn.TrainOneStepCell(lmodel, opt)
    TrainOneStepCell.set_train()
    for i in range(2):
        print(mindspore.Tensor(model.weight))
        print('='*20)
        loss = TrainOneStepCell(x, y)
        print(loss)
        print(mindspore.Tensor(model.weight))
        print('='*20)

