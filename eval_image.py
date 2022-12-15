import os
import argparse
from glob import glob
import ast
import colorama
import numpy as np
import logging
import json

import torch
import mindspore
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset

import src.utils as utils
from src.utils import progress_bar, logger
from src.sinFID import calculate_SIFID
import src.tools.pt2ms as pt2ms
from src.modules import networks_2d
from src.datasets.image import SingleImageDataset


def eval(opt, netG):
    # Re-generate dataset frames
    if not hasattr(opt, 'Z_init_size'):
        initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        initial_size = [int(initial_size * opt.ar), initial_size]
        opt.Z_init_size = [opt.batch_size, opt.latent_dim, *initial_size]
    G_curr = netG

    progressbar_args = {
        "iterable": range(opt.niter),
        "desc": "Training scale [{}/{}]".format(opt.scale_idx + 1, opt.stop_scale + 1),
        "train": True,
        "offset": 0,
        "logging_on_update": False,
        "logging_on_close": True,
        "postfix": True
    }
    epoch_iterator = progress_bar.create_progressbar(**progressbar_args)

    random_samples = []
    for iteration in epoch_iterator:

        noise_init = utils.generate_noise_size(opt.Z_init_size)

        # Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
        ))

        fake_var = []
        fake_vae_var = []
        for _ in range(opt.num_samples):
            noise_init = utils.generate_noise_ref(noise_init.shape)
            fake, fake_vae = G_curr(noise_init, opt.Noise_Amps, noise_init=noise_init, isRandom=True)
            fake_var.append(fake)
            fake_vae_var.append(fake_vae)
        fake_var = ops.Concat(0)(fake_var)
        fake_vae_var = ops.Concat(0)(fake_vae_var)

        # Tensorboard
        # opt.summary.visualize_image(opt, iteration, real, 'Real')
        # opt.summary.visualize_image(opt, iteration, fake_var, 'Fake var')
        # opt.summary.visualize_image(opt, iteration, fake_vae_var, 'Fake VAE var')

        random_samples.append(fake_var)

    random_samples = ops.Concat(0)(random_samples)
    with open(os.path.join(opt.saver.eval_dir, 'random_samples.npy'), 'wb') as f:
        np.save(f, random_samples.asnumpy())
    epoch_iterator.close()

    return random_samples


if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-id', default=0, type=int, help='Device ID')

    parser.add_argument('--exp-dir', type=str, required=True, help="Experiment directory")
    parser.add_argument('--netG', type=str, default='netG.ckpt', help="path to netG (to continue training)")
    parser.add_argument('--save-path', type=str, default='images', help="New directory for outputs")

    parser.add_argument('--num-samples', type=int, default=10, help='number of samples to generate')
    parser.add_argument('--niter', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--data-rep', type=int, default=1, help='data repetition')
    parser.add_argument('--scale-idx', type=int, default=-1, help='current scale idx (=len of body)')
    parser.add_argument('--max-samples', type=int, default=4, help="Maximum number of samples")

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()

    clear = colorama.Style.RESET_ALL
    blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
    green = colorama.Fore.GREEN + colorama.Style.BRIGHT
    magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT

    context.set_context(mode=1, device_id=opt.device_id)

    exceptions = ['niter', 'data_rep', 'batch_size', 'netG', 'scale_idx']
    all_dirs = glob(opt.exp_dir)

    progressbar_args = {
        "iterable": all_dirs,
        "desc": "Experiments",
        "train": True,
        "offset": 0,
        "logging_on_update": False,
        "logging_on_close": True,
        "postfix": True
    }
    exp_iterator = progress_bar.create_progressbar(**progressbar_args)
    logger.configure_logging(os.path.abspath(os.path.join(opt.exp_dir, 'logbook.txt')))

    for idx, exp_dir in enumerate(exp_iterator):
        opt.experiment_dir = exp_dir
        keys = vars(opt).keys()
        with open(os.path.join(exp_dir, 'args.txt'), 'r') as f:
            for line in f.readlines():
                log_arg = line.replace(' ', '').replace('\n', '').split(':')
                assert len(log_arg) == 2
                if log_arg[0] in exceptions:
                    continue
                try:
                    setattr(opt, log_arg[0], ast.literal_eval(log_arg[1]))
                except Exception:
                    setattr(opt, log_arg[0], log_arg[1])

        opt.netG = os.path.join(exp_dir, opt.netG)
        if not os.path.exists(opt.netG):
            logging.info('Skipping {}, file not exists!'.format(opt.netG))
            continue

        ## Define & Initialize
        # Saver
        opt.saver = utils.DataSaver(opt)

        # Tensorboard Summary
        # opt.summary = utils.TensorboardSummary(opt.saver.eval_dir)

        # Adjust scales
        utils.adjust_scales2image(opt.img_size, opt)

        # Dataset
        opt.dataset = SingleImageDataset(opt)
        data_loader = GeneratorDataset(opt.dataset, ['data', 'zero-scale data'], shuffle=True)
        opt.data_loader = data_loader.batch(opt.batch_size)

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
            opt.scale_idx = opt.saver.load_json('intermediate.json')['scale_idx']
        opt.Noise_Amps = opt.saver.load_json('intermediate.json')['noise_amps'][:opt.scale_idx + 1]

        ## Current networks
        assert hasattr(networks_2d, opt.generator)
        netG = getattr(networks_2d, opt.generator)(opt)
        for _ in range(opt.scale_idx):
            netG.init_next_stage()
        mindspore.load_param_into_net(netG, checkpoint)

        ## Eval
        random_samples = eval(opt, netG)
        opt.experiments = sorted(glob(opt.exp_dir))
        utils.generate_images(opt)

        # SIFID
        real_dir = os.path.join(*opt.dataset.image_path.split('/')[:-1]) if opt.dataset.image_path[0] != '/' \
                   else '/' + os.path.join(*opt.dataset.image_path.split('/')[:-1])
        fake_dir = os.path.join(opt.saver.eval_dir, opt.save_path)
        sifid = calculate_SIFID(real_dir, fake_dir)
        logging.info(f'SVFID: {sifid}')
        print(f'SVFID: {sifid}')