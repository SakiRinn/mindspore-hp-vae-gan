from collections import OrderedDict
import re
import yaml
import sys
sys.path.append(".")

import mindspore
from mindspore import Tensor, Parameter


def p2m_WDiscriminator_2d(netD_pth) -> OrderedDict:
    netD_pth = netD_pth['state_dict']
    new_state = OrderedDict()
    for key, value in netD_pth.items():

        if re.search(r"block\d+.", key) is not None:
            prefix = re.search(r"block\d+.", key).group()
            num = int(re.search(r"\d+", prefix).group())
            key = key.replace(prefix, f"{num}.")

        if "conv." in key:
            key = key.replace("conv.", "0.")
            if "weight_orig" in key:
                key = key.replace("weight_orig", "weight")

        new_state[key] = Parameter(Tensor(value.numpy()).astype(mindspore.float32))
    return new_state


def p2m_HPVAEGAN_2d(netG_pth) -> OrderedDict:
    netG_pth = netG_pth['state_dict']
    new_state = OrderedDict()
    for key, value in netG_pth.items():

        ## Encode
        if "encode." in key:
            # prefix
            if re.search(r"features.conv_block_\d+.", key) is not None:
                prefix = re.search(r"features.conv_block_\d+.", key).group()
                num = int(re.search(r"\d+", prefix).group())
                key = key.replace(prefix, f"_features.{num}.")
            elif "mu" in key:
                key = key.replace("mu.", "_mu.")
            elif "logvar" in key:
                key = key.replace("logvar.", "_logvar.")
            # module
            if "conv." in key:
                key = key.replace("conv.", "0.")
                if "weight_orig" in key:
                    key = key.replace("weight_orig", "weight")

        ## Decoder & Body
        if 'decoder.' in key or re.search(r"body.\d+.", key) is not None:
            # prefix
            if "head." in key:
                key = key.replace("head.", "0.")
            elif re.search(r"block\d+.", key) is not None:
                prefix = re.search(r"block\d+.", key).group()
                num = int(re.search(r"\d+", prefix).group())
                key = key.replace(prefix, f"{num + 1}.")
            if "tail." in key:
                key = key.replace("tail.", f"6.")
            # module
            if "conv." in key:
                key = key.replace("conv.", "0.")
            elif "norm." in key:
                key = key.replace("norm.", "1.")
                if "weight." in key:
                    key = key.replace("weight.", "gamma.")
                elif "bias." in key:
                    key = key.replace("bias.", "beta.")

        new_state[key] = Parameter(Tensor(value.numpy()).astype(mindspore.float32))
    return new_state


def p2m_HPVAEGAN_3d(netG_pth):
    ...


def load_scale(netG_pth) -> int:
    return netG_pth['scale']


def load_Noise_Amps(Noise_Amps_pth) -> list:
    return Noise_Amps_pth['data']


if __name__ == "__main__":
    ...