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
""" preprocess """
import src.utils as utils
import argparse
import mindspore.context as context
import os
import ast
import src.utils as utils
from src.sinFID import calculate_SIFID


def post_process(opt):
    utils.generate_images(opt)

    # SIFID
    real_dir = os.path.join(*opt.dataset.image_path.split('/')[:-1]) if opt.dataset.image_path[0] != '/' \
                else '/' + os.path.join(*opt.dataset.image_path.split('/')[:-1])
    fake_dir = os.path.join(opt.saver.eval_dir, opt.save_path)
    sifid = calculate_SIFID(real_dir, fake_dir)
    print(f'SVFID: {sifid}')


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
