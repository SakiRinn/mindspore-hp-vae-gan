import os
import argparse
import ast
import json

import torch
import mindspore
import mindspore.context as context

import src.tools.pt2ms as pt2ms
import src.utils as utils
from src.modules import networks_2d
from preprocess import pre_process


if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, required=True, help="Experiment directory")
    parser.add_argument('--device-id', default=0, type=int, help='Device ID')
    parser.add_argument("--format", type=str, default='MINDIR', help="MINDIR or AIR")
    parser.add_argument('--netG', type=str, default='netG.ckpt', help="path to netG (to continue training)")
    parser.add_argument('--scale-idx', type=int, default=-1, help='current scale idx (=len of body)')
    opt = parser.parse_args()

    context.set_context(mode=0, device_id=opt.device_id)

    exceptions = ['niter', 'data_rep', 'batch_size', 'netG', 'scale_idx']
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
            if log_arg[0] in exceptions:
                continue
            try:
                setattr(opt, log_arg[0], ast.literal_eval(log_arg[1]))
            except Exception:
                setattr(opt, log_arg[0], log_arg[1])

    opt.netG = os.path.join(opt.exp_dir, opt.netG)
    if not os.path.exists(opt.netG):
        print('Skipping {}, file not exists!'.format(opt.netG))
        exit(1)

    # Load
    if not os.path.isfile(opt.netG):
        raise RuntimeError("=> no <G> checkpoint found at '{}'".format(opt.netG))
    if opt.netG.endswith('.pth'):
        checkpoint = torch.load(opt.netG, map_location=torch.device('cpu'))
        intermediate = pt2ms.load_intermediate(checkpoint)
        with open(os.path.join(opt.exp_dir, 'intermediate.json'), 'w') as f:
            json.dump(intermediate, f, indent=4)
        checkpoint = pt2ms.p2m_HPVAEGAN_2d(checkpoint)
    elif opt.netG.endswith('.ckpt'):
        checkpoint = mindspore.load_checkpoint(opt.netG)
        checkpoint = pt2ms.m2m_HPVAEGAN_2d(checkpoint)

    # Init
    if opt.scale_idx == -1:
        opt.scale_idx = opt.saver.load_json('intermediate.json', opt.exp_dir)['scale_idx']
    save_dir = os.path.join(opt.experiment_dir, opt.netG.split('/')[-1].split('.')[0])

    # Current networks
    assert hasattr(networks_2d, opt.generator)
    netG = getattr(networks_2d, opt.generator)(opt)
    for _ in range(opt.scale_idx):
        netG.init_next_stage()
    mindspore.load_param_into_net(netG, checkpoint)

    ## EXPORT
    noise_init, noise_amps = pre_process(opt)
    mindspore.export(netG, noise_init, noise_amps, noise_init,
                     file_name=save_dir, file_format=opt.format)