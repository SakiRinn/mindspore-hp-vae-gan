import argparse
import os
from glob import glob
import ast
import pickle as pkl
import logging
import colorama

import mindspore
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset

import sys
sys.path.insert(0, '.')
import utils
from utils import logger, tools
from modules import networks_2d
from datasets.image import SingleImageDataset

clear = colorama.Style.RESET_ALL
blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
green = colorama.Fore.GREEN + colorama.Style.BRIGHT
magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT


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
    epoch_iterator = tools.create_progressbar(**progressbar_args)

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
    with open(os.path.join(opt.saver.eval_dir, 'random_samples.pkl'), 'wb') as f:
        pkl.dump(random_samples, f)
    epoch_iterator.close()


if __name__ == '__main__':
    context.set_context(mode=0, device_id=5)

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', required=True, help="Experiment directory")
    parser.add_argument('--num-samples', type=int, default=10, help='number of samples to generate')
    parser.add_argument('--netG', default='netG.ckpt', help="path to netG (to continue training)")
    parser.add_argument('--niter', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--data-rep', type=int, default=1, help='data repetition')

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()

    exceptions = ['niter', 'data_rep', 'batch_size', 'netG']
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
    exp_iterator = tools.create_progressbar(**progressbar_args)

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

        # Logger
        logger.configure_logging(os.path.abspath(os.path.join(opt.experiment_dir, 'logbook.txt')))

        # Adjust scales
        utils.adjust_scales2image(opt.img_size, opt)

        # Initial parameters
        opt.scale_idx = 0
        opt.nfc_prev = 0
        opt.Noise_Amps = []

        # Dataset
        dataset = SingleImageDataset(opt)
        data_loader = GeneratorDataset(dataset, ['data', 'zero-scale data'], shuffle=True)
        data_loader = data_loader.batch(opt.batch_size)
        opt.data_loader = data_loader

        ## Current networks
        assert hasattr(networks_2d, opt.generator)
        netG = getattr(networks_2d, opt.generator)(opt)

        # Init
        opt.Noise_Amps = opt.saver.load_json('intermediate.json')['noise_amps']
        opt.scale_idx = opt.saver.load_json('intermediate.json')['scale_idx']
        opt.resumed_idx = opt.saver.load_json('intermediate.json')['scale_idx']
        opt.resume_dir = '/'.join(opt.netG.split('/')[:-1])

        # Load
        if not os.path.isfile(opt.netG):
            raise RuntimeError("=> no <G> checkpoint found at '{}'".format(opt.netG))
        checkpoint = mindspore.load_checkpoint(opt.netG)
        for _ in range(opt.scale_idx):
            netG.init_next_stage()
        mindspore.load_param_into_net(netG, checkpoint)

        ## Eval
        eval(opt, netG)
