from collections import OrderedDict
import yaml
import sys
sys.path.append(".")

import torch
import mindspore
from mindspore import Tensor, Parameter


def p2m_image(state):
    new_state = OrderedDict()
    for key, value in state.items():

        if 'conv' in key and 'layer' not in key:
            key = key[-12:]

        if 'bn' in key and 'weight' in key and 'layer' not in key:
            key = key[-10:-6] + 'gamma'
        if 'bn' in key and 'bias' in key and 'layer' not in key:
            key = key[-8:-4] + 'beta'

        if 'conv' in key and 'layer' in key:
            key = key[-21:]
        if 'bn' in key and 'weight' in key and 'layer' in key:
            key = key[-19:-6] + 'gamma'
        if 'bn' in key and 'bias' in key and 'layer' in key:
            key = key[-17:-4] + 'beta'

        if 'downsample.0' in key and 'weight' in key:
            key = key[-28:]
        if 'downsample.1' in key and 'weight' in key:
            key = key[-28:-6] + 'gamma'
        if 'downsample.1' in key and 'bias' in key:
            key = key[-26:-4] + 'beta'

        if 'fc' in key and 'weight' in key:
            key = key[-9:]
        if 'fc' in key and 'bias' in key:
            key = key[-7:]

        if 'in' in key and 'bina' not in key and 'run' not in key:
            key = key[-10:]

        if 'out2' in key:
            key = 'out2.weight'
        if 'out3' in key:
            key = 'out3.weight'
        if 'out4' in key:
            key = 'out4.weight'
        if 'out5' in key:
            key = 'out5.weight'

        if ('binarize.0' in key or 'binarize.3' in key or 'binarize.6' in key) and ('weight' in key):
            key = key[-17:]
        if 'binarize.3.bias' in key or 'binarize.6.bias' in key:
            key = key[-15:]
        if 'binarize.1.weight' in key or 'binarize.4.weight' in key:
            key = key[-17:-6] + 'gamma'
        if 'binarize.1.bias' in key or 'binarize.4.bias' in key:
            key = key[-15:-4] + 'beta'

        if ('thresh.0' in key or 'thresh.3' in key or 'thresh.6' in key) and ('weight' in key):
            key = key[-15:]
        if 'thresh.3.bias' in key or 'thresh.6.bias' in key:
            key = key[-13:]
        if 'thresh.1.weight' in key or 'thresh.4.weight' in key:
            key = key[-15:-6] + 'gamma'
        if 'thresh.1.bias' in key or 'thresh.4.bias' in key:
            key = key[-13:-4] + 'beta'

        if 'bn' in key and 'running_mean' in key and 'layer' not in key:
            key = 'bn1.moving_mean'
        if 'bn' in key and 'running_var' in key and 'layer' not in key:
            key = 'bn1.moving_variance'

        if 'bn' in key and 'running_mean' in key and 'layer' in key:
            key = key[-25:-12] + 'moving_mean'
        if 'bn' in key and 'running_var' in key and 'layer' in key:
            key = key[-24:-11] + 'moving_variance'

        if 'downsample' in key and 'running_mean' in key:
            key = key[-34:-12] + 'moving_mean'
        if 'downsample' in key and 'running_var' in key:
            key = key[-33:-11] + 'moving_variance'

        if 'binarize' in key and 'running_mean' in key:
            key = key[-23:-12] + 'moving_mean'
        if 'binarize' in key and 'running_var' in key:
            key = key[-22:-11] + 'moving_variance'

        if 'thresh' in key and 'running_mean' in key:
            key = key[-21:-12] + 'moving_mean'
        if 'thresh' in key and 'running_var' in key:
            key = key[-20:-11] + 'moving_variance'

        new_state[key] = Parameter(Tensor(value.numpy()).astype(mindspore.float32))
    return new_state


if __name__ == "__main__":
    from mindspore import context
    from modules.networks_2d import GeneratorHPVAEGAN, WDiscriminator2D

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=7)

    ## load pth
    # param_dic = torch.load("tests/pre-trained-model-synthtext-resnet18", map_location=torch.device('cpu'))
    # new_dic = mod_Liao(param_dic=param_dic)

    # Pred = Predict(DBnet(isTrain=False), new_dic)
    # Pred.show(img_path='data/train_images/img_3.jpg')
    # print("完成")stream = open('config.yaml', 'r', encoding='utf-8')

    # load ckpt
    param_dic = torch.load("tests/1epoch", map_location=torch.device('cpu'))
    new_dic = p2m_image(param_dic)


    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    net = GeneratorHPVAEGAN(config)
    mindspore.load_param_into_net(net, new_dic)
    mindspore.save_checkpoint(net, 'checkpoints/pthTOckpt/1epoch.ckpt')