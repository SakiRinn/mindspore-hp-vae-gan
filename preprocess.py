# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" postprocess """
import src.utils as utils
import argparse
import mindspore.context as context
from mindspore import Tensor, float32
import os
import ast
import numpy as np


def pre_process(opt):
    if not hasattr(opt, 'Z_init_size'):
        initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        initial_size = [int(initial_size * opt.ar), initial_size]
        opt.Z_init_size = [opt.batch_size, opt.latent_dim, *initial_size]

    noise_init = utils.generate_noise_size(opt.Z_init_size)
    noise_init = utils.generate_noise_ref(noise_init.shape)
    noise_amps = opt.saver.load_json('intermediate.json', path=opt.exp_dir)['noise_amps'][:opt.scale_idx + 1]
    noise_amps = Tensor(noise_amps, dtype=float32)

    if not os.path.exists(os.path.join(opt.experiment_dir, 'noise_init')):
        os.mkdir(os.path.join(opt.experiment_dir, 'noise_init'))
    if not os.path.exists(os.path.join(opt.experiment_dir, 'noise_amps')):
        os.mkdir(os.path.join(opt.experiment_dir, 'noise_amps'))
    noise_init.asnumpy().tofile(os.path.join(opt.experiment_dir, 'noise_init', 'noise_init.bin'))
    noise_amps.asnumpy().tofile(os.path.join(opt.experiment_dir, 'noise_amps', 'noise_amps.bin'))

    return noise_init, noise_amps


if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, required=True, help="Experiment directory")
    parser.add_argument('--device-id', default=0, type=int, help='Device ID')
    parser.add_argument('--scale-idx', type=int, default=-1, help='current scale idx (=len of body)')
    opt = parser.parse_args()

    context.set_context(mode=0, device_id=opt.device_id)

    includes = ['scale_factor', 'stop_scale', 'img_size', 'ar', 'latent_dim']
    opt.batch_size = 1
    opt.experiment_dir = os.path.join(opt.exp_dir, 'infer')
    if not os.path.exists(opt.experiment_dir):
        os.mkdir(opt.experiment_dir)
    opt.saver = utils.DataSaver(opt)
    os.rmdir(os.path.join(opt.experiment_dir, 'eval'))

    keys = vars(opt).keys()
    with open(os.path.join(opt.exp_dir, 'args.txt'), 'r') as f:
        for line in f.readlines():
            log_arg = line.replace(' ', '').replace('\n', '').split(':')
            assert len(log_arg) == 2
            if log_arg[0] not in includes:
                continue
            try:
                setattr(opt, log_arg[0], ast.literal_eval(log_arg[1]))
            except Exception:
                setattr(opt, log_arg[0], log_arg[1])

    # Init
    if opt.scale_idx == -1:
        opt.scale_idx = opt.saver.load_json('intermediate.json', opt.exp_dir)['scale_idx']

    pre_process(opt)