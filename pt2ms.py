from collections import OrderedDict
import re
import json

import torch
import mindspore
from mindspore import Tensor, Parameter


def p2m_WDiscriminator_2d(netD_pth) -> OrderedDict:
    netD_pth = netD_pth['state_dict']
    new_state = OrderedDict()
    for key, value in netD_pth.items():

        if 'body.' in key and int(re.search(r"block(\d+?)\.", key).group(1)) != 0:
            key = key.replace("body.", "body.0.", 1)

        if re.search(r"block(\d+?)\.", key) is not None:
            prefix = re.search(r"block(\d+?)\.", key).group(0)
            num = int(re.search(r"block(\d+?)\.", key).group(1))
            key = key.replace(prefix, f"{num}.", 1)

        if "conv." in key:
            key = key.replace("conv.", "0.", 1)
            if "weight_orig" in key:
                key = key.replace("weight_orig", "weight", 1)

        new_state[key] = Parameter(Tensor(value.numpy()).astype(mindspore.float32))
    return new_state


def p2m_HPVAEGAN_2d(netG_pth) -> OrderedDict:
    netG_pth = netG_pth['state_dict']
    new_state = OrderedDict()
    for key, value in netG_pth.items():

        ## Encode
        if "encode." in key:

            # prefix
            if re.search(r"features\.conv_block_(\d+?)\.", key) is not None:
                prefix = re.search(r"features.conv_block_(\d+?)\.", key).group(0)
                num = int(re.search(r"features.conv_block_(\d+?)\.", key).group(1))
                key = key.replace(prefix, f"_features.{num}.")
            elif "mu" in key:
                key = key.replace("mu.", "_mu.", 1)
            elif "logvar" in key:
                key = key.replace("logvar.", "_logvar.", 1)

            # module
            if "conv." in key:
                key = key.replace("conv.", "0.", 1)
                if "weight_orig" in key:
                    key = key.replace("weight_orig", "weight", 1)

        ## Decoder & Body
        if 'decoder.' in key or 'body.' in key:

            if 'body.' in key and int(re.search(r"body\.(\d+?)\.", key).group(1)) != 0:
                key = key.replace("body.", "body.0.0.", 1)

            # prefix
            if "head." in key:
                key = key.replace("head.", "0.", 1)
            elif re.search(r"block(\d+?)\.", key) is not None:
                prefix = re.search(r"block(\d+?)\.", key).group(0)
                num = int(re.search(r"block(\d+?)\.", key).group(1))
                key = key.replace(prefix, f"{num + 1}.", 1)
            elif "tail." in key:
                key = key.replace("tail.", "6.", 1)

            # module
            if "conv." in key:
                key = key.replace("conv.", "0.", 1)
            elif "norm." in key:
                key = key.replace("norm.", "1.", 1)
                if "weight" in key:
                    key = key.replace("weight", "gamma", 1)
                elif "bias" in key:
                    key = key.replace("bias", "beta", 1)
                elif "running_mean" in key:
                    key = key.replace("running_mean", "moving_mean", 1)
                elif "running_var" in key:
                    key = key.replace("running_var", "moving_variance", 1)
                elif 'num_batches_tracked' in key:
                    continue

        if 'weight_u' in key or 'weight_v' in key:
            value = value.unsqueeze(-1)
        new_state[key] = Parameter(Tensor(value.numpy()).astype(mindspore.float32))
    return new_state


def m2m_HPVAEGAN_2d(netG_pth) -> OrderedDict:
    new_state = OrderedDict()
    for key, value in netG_pth.items():

        ## Body
        if 'encode.' not in key and 'decoder.' in key:
            num = int(re.search(r"^(\d+?)\.", key).group(1))
            if num != 0:
                key = key.replace(f"{num}.", f"0.0.{num}.", 1)
            key = 'body.' + key

        new_state[key] = Parameter(Tensor(value.numpy()).astype(mindspore.float32))
    return new_state


def load_intermediate(netG_pth) -> OrderedDict:
    new_state = OrderedDict(noise_amps=netG_pth['noise_amps'], scale_idx=netG_pth['scale'])
    return new_state


if __name__ == '__main__':
    import modules.networks_2d as networks_2d
    from glob import glob
    import argparse
    import os
    import ast
    from mindspore import context

    context.set_context(mode=0, device_id=5)

    pth = torch.load("./netG_9.pth", map_location=torch.device('cpu'))
    inter = load_intermediate(pth)
    ckpt = p2m_HPVAEGAN_2d(pth)

    ## Opt
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', default='run/air_balloons/myimagetest/experiment_1', help="Experiment directory")
    parser.add_argument('--generator', type=str, default='GeneratorHPVAEGAN', help='generator model')
    parser.add_argument('--discriminator', type=str, default='WDiscriminator2D', help='discriminator model')
    opt = parser.parse_args()

    # exceptions = ['niter', 'data_rep', 'batch_size', 'netG']
    all_dirs = glob(opt.exp_dir)

    for idx, exp_dir in enumerate(all_dirs):
        opt.experiment_dir = exp_dir
        keys = vars(opt).keys()
        with open(os.path.join(exp_dir, 'args.txt'), 'r') as f:
            for line in f.readlines():
                log_arg = line.replace(' ', '').replace('\n', '').split(':')
                assert len(log_arg) == 2
                # if log_arg[0] in exceptions:
                #     continue
                try:
                    setattr(opt, log_arg[0], ast.literal_eval(log_arg[1]))
                except Exception:
                    setattr(opt, log_arg[0], log_arg[1])

    ## Current networks
    assert hasattr(networks_2d, opt.generator)
    netG = getattr(networks_2d, opt.generator)(opt)

    ## Load
    for _ in range(inter['scale_idx']):
        netG.init_next_stage()
    mindspore.load_param_into_net(netG, ckpt)
    mindspore.save_checkpoint(netG, 'test.ckpt')