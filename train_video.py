import argparse
import utils
import random
import os
import colorama
import logging

from utils import logger, tools
from modules import networks_3d
from modules.losses import DWithLoss, GWithLoss
from modules.optimizers import ClippedAdam
from datasets import SingleVideoDataset

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset


def train(opt, netG):
    ############
    ### INIT ###
    ############

    ## Re-generate dataset frames
    fps, td, fps_index = utils.get_fps_td_by_index(opt.scale_idx, opt)
    opt.fps = fps
    opt.td = td
    opt.fps_index = fps_index

    ## Log
    with logger.LoggingBlock("Updating dataset", emph=True):
        logging.info("{}FPS :{} {}{}".format(green, clear, opt.fps, clear))
        logging.info("{}Time-Depth :{} {}{}".format(green, clear, opt.td, clear))
        logging.info("{}Sampling-Ratio :{} {}{}".format(green, clear, opt.sampling_rates[opt.fps_index], clear))
        opt.dataset.generate_frames(opt.scale_idx)

    ## Noise
    if not hasattr(opt, 'Z_init_size'):
        initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        initial_size = [int(initial_size * opt.ar), initial_size]
        opt.Z_init_size = [opt.batch_size, opt.latent_dim, opt.td, *initial_size]


    ## Discriminator
    if opt.vae_levels < opt.scale_idx + 1:
        # Current discriminator
        D_curr = getattr(networks_3d, opt.discriminator)(opt)

        # load parameters for discriminator
        if (opt.netG != '') and (opt.resumed_idx == opt.scale_idx):
            D_curr.load_param_into_net(
                mindspore.load_checkpoint('{}/netD_{}.ckpt'\
                                          .format(opt.resume_dir, opt.scale_idx - 1))['state_dict']
            )
        elif opt.vae_levels < opt.scale_idx:
            D_curr.load_param_into_net(
                mindspore.load_checkpoint('{}/netD_{}.ckpt'\
                                          .format(opt.saver.experiment_dir, opt.scale_idx - 1))['state_dict']
            )

        # Current optimizer for discriminator
        optimizerD = nn.Adam(D_curr.get_parameters(), opt.lr_d, beta1=opt.beta1, beta2=0.999)


    ## Generator
    parameter_list = []

    if not opt.train_all:
        # (1) train all
        if opt.vae_levels < opt.scale_idx + 1:
            train_depth = min(opt.train_depth, len(netG.body) - opt.vae_levels + 1)
            parameter_list += [
                {"params": block.get_parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-train_depth:])]
        else:
            parameter_list += [{"params": netG.encode.get_parameters(),
                                "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder.get_parameters(),
                                "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}
            ]
            parameter_list += [
                {"params": block.get_parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-opt.train_depth:])
            ]
    else:
        # (2) NOT train all
        if len(netG.body) < opt.train_depth:
            parameter_list += [{"params": netG.encode.get_parameters(),
                                "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder.get_parameters(),
                                "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}
            ]
            parameter_list += [
                {"params": block.get_parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body) - 1 - idx))}
                for idx, block in enumerate(netG.body)
            ]
        else:
            parameter_list += [
                {"params": block.get_parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-opt.train_depth:])
            ]

    # Current generator
    G_curr = netG
    # Current optimizer for generator
    optimizerG = ClippedAdam(opt, parameter_list, opt.lr_g, beta1=opt.beta1, beta2=0.999)


    ## Train-one-step cell
    D_loss = DWithLoss(opt, D_curr, G_curr)
    G_loss = GWithLoss(opt, D_curr, G_curr)
    D_train = nn.TrainOneStepCell(D_loss, optimizerD)
    G_train = nn.TrainOneStepCell(G_loss, optimizerG)


    ## Progress bar
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
    iterator = iter(opt.data_loader)


    #############
    ### TRAIN ###
    #############
    for iteration in epoch_iterator:
        ## Initialize
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(opt.data_loader)
            data = next(iterator)

        if opt.scale_idx > 0:
            real, real_zero = data
        else:
            real, _ = data
            real_zero = real

        noise_init = utils.generate_noise(size=opt.Z_init_size)


        ## Calculate noise_amp (First iteration)
        if iteration == 0:
            if opt.const_amp:
                opt.Noise_Amps.append(1)
            else:
                if opt.scale_idx == 0:
                    opt.noise_amp = 1
                    opt.Noise_Amps.append(opt.noise_amp)
                else:
                    opt.Noise_Amps.append(0)
                    z_reconstruction, _, _ = G_curr(real_zero, opt.Noise_Amps, mode="rec")
                    RMSE = nn.RMSELoss()(real, z_reconstruction)
                    RMSE = ops.stop_gradient(RMSE)

                    opt.noise_amp = opt.noise_amp_init * RMSE.item() / opt.batch_size
                    opt.Noise_Amps[-1] = opt.noise_amp


        ## Update parameters
        generated, generated_vae, (mu, logvar) = G_curr(real_zero, opt.Noise_Amps, randMode=False)
        if opt.vae_levels >= opt.scale_idx + 1:
            # (1) Update VAE network
            G_loss.VAEMode(True)
            G_train(real, real_zero, fake, generated, generated_vae, mu, logvar)
        else:
            # (2) Update distriminator: maximize D(x) + D(G(z))
            fake, _ = G_curr(noise_init, opt.Noise_Amps, noise_init=noise_init, randMode=True)
            D_train(real, fake)

            # (3) Update generator: maximize D(G(z)) (After grad clipping)
            G_loss.VAEMode(False)
            G_train(real, real_zero, fake, generated, generated_vae, mu, logvar)


        ## Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
        ))


        ## Virsualize with tensorboard
        # if opt.visualize:
        #     # Tensorboard
        #     opt.summary.add_scalar('Video/Scale {}/noise_amp'.format(opt.scale_idx), opt.noise_amp, iteration)
        #     if opt.vae_levels >= opt.scale_idx + 1:
        #         opt.summary.add_scalar('Video/Scale {}/KLD'.format(opt.scale_idx), kl_loss.item(), iteration)
        #     else:
        #         opt.summary.add_scalar('Video/Scale {}/rec loss'.format(opt.scale_idx), rec_loss.item(), iteration)
        #     opt.summary.add_scalar('Video/Scale {}/noise_amp'.format(opt.scale_idx), opt.noise_amp, iteration)
        #     if opt.vae_levels < opt.scale_idx + 1:
        #         opt.summary.add_scalar('Video/Scale {}/errG'.format(opt.scale_idx), errG.item(), iteration)
        #         opt.summary.add_scalar('Video/Scale {}/errD_fake'.format(opt.scale_idx), errD_fake.item(), iteration)
        #         opt.summary.add_scalar('Video/Scale {}/errD_real'.format(opt.scale_idx), errD_real.item(), iteration)
        #     else:
        #         opt.summary.add_scalar('Video/Scale {}/Rec VAE'.format(opt.scale_idx), rec_vae_loss.item(), iteration)

        #     if iteration % opt.print_interval == 0:
        #         with torch.no_grad():
        #             fake_var = []
        #             fake_vae_var = []
        #             for _ in range(3):
        #                 noise_init = utils.generate_noise(ref=noise_init)
        #                 fake, fake_vae = G_curr(noise_init, opt.Noise_Amps, noise_init=noise_init, mode="rand")
        #                 fake_var.append(fake)
        #                 fake_vae_var.append(fake_vae)
        #             fake_var = torch.cat(fake_var, dim=0)
        #             fake_vae_var = torch.cat(fake_vae_var, dim=0)

        #         opt.summary.visualize_video(opt, iteration, real, 'Real')
        #         opt.summary.visualize_video(opt, iteration, generated, 'Generated')
        #         opt.summary.visualize_video(opt, iteration, generated_vae, 'Generated VAE')
        #         opt.summary.visualize_video(opt, iteration, fake_var, 'Fake var')
        #         opt.summary.visualize_video(opt, iteration, fake_vae_var, 'Fake VAE var')

    epoch_iterator.close()


    ## Save data
    opt.saver.save_checkpoint({'data': opt.Noise_Amps}, 'Noise_Amps.ckpt')
    opt.saver.save_checkpoint({
        'scale': opt.scale_idx,
        'parameters_dict': netG.parameters_dict(),
        'optimizer': optimizerG.parameters_dict(),
        'noise_amps': opt.Noise_Amps,
    }, 'netG.ckpt')
    if opt.vae_levels < opt.scale_idx + 1:
        opt.saver.save_checkpoint({
            'scale': opt.scale_idx,
            'parameters_dict': D_curr.module.parameters_dict()
                          if opt.device != 'CPU' else D_curr.parameters_dict(),
            'optimizer': optimizerD.parameters_dict(),
        }, 'netD_{}.ckpt'.format(opt.scale_idx))


if __name__ == '__main__':
    ## Parser
    parser = argparse.ArgumentParser()

    # Load, input, save configurations
    parser.add_argument('--netG', default='', help='path to netG (to continue training)')
    parser.add_argument('--netD', default='', help='path to netD (to continue training)')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    # Networks hyper parameters
    parser.add_argument('--nc-im', type=int, default=3, help='# channels')
    parser.add_argument('--nfc', type=int, default=64, help='model basic # channels')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dim size')
    parser.add_argument('--vae-levels', type=int, default=3, help='# VAE levels')
    parser.add_argument('--enc-blocks', type=int, default=2, help='# encoder blocks')
    parser.add_argument('--ker-size', type=int, default=3, help='kernel size')
    parser.add_argument('--num-layer', type=int, default=5, help='number of layers')
    parser.add_argument('--stride', default=1, help='stride')
    parser.add_argument('--padd-size', type=int, default=1, help='net pad size')
    parser.add_argument('--generator', type=str, default='GeneratorHPVAEGAN', help='generator model')
    parser.add_argument('--discriminator', type=str, default='WDiscriminator3D', help='discriminator model')

    # Pyramid parameters
    parser.add_argument('--scale-factor', type=float, default=0.75, help='pyramid scale factor')
    parser.add_argument('--noise_amp', type=float, default=0.1, help='addative noise cont weight')
    parser.add_argument('--min-size', type=int, default=32, help='image minimal size at the coarser scale')
    parser.add_argument('--max-size', type=int, default=256, help='image minimal size at the coarser scale')

    # Optimization hyper parameters
    parser.add_argument('--niter', type=int, default=50000, help='number of iterations to train per scale')
    parser.add_argument('--lr-g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr-d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lambda-grad', type=float, default=0.1, help='gradient penelty weight')
    parser.add_argument('--rec-weight', type=float, default=10., help='reconstruction loss weight')
    parser.add_argument('--kl-weight', type=float, default=1., help='reconstruction loss weight')
    parser.add_argument('--disc-loss-weight', type=float, default=1.0, help='discriminator weight')
    parser.add_argument('--lr-scale', type=float, default=0.2, help='scaling of learning rate for lower stages')
    parser.add_argument('--train-depth', type=int, default=1, help='how many layers are trained if growing')
    parser.add_argument('--grad-clip', type=float, default=5, help='gradient clip')
    parser.add_argument('--const-amp', action='store_true', default=False, help='constant noise amplitude')
    parser.add_argument('--train-all', action='store_true', default=False, help='train all levels w.r.t. train-depth')

    # Dataset
    parser.add_argument('--video-path', required=True, help='video path')
    parser.add_argument('--start-frame', default=0, type=int, help='start frame number')
    parser.add_argument('--max-frames', default=13, type=int, help='# frames to save')
    parser.add_argument('--hflip', action='store_true', default=False, help='horizontal flip')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--sampling-rates', type=int, nargs='+', default=[4, 3, 2, 1], help='sampling rates')
    parser.add_argument('--stop-scale-time', type=int, default=-1)
    parser.add_argument('--data-rep', type=int, default=1000, help='data repetition')

    # Main arguments
    parser.add_argument('--checkname', type=str, default='DEBUG', help='check name')
    parser.add_argument('--mode', default='train', help='task to be done')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--print-interval', type=int, default=100, help='print interva')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize using tensorboard')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda')

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()

    assert opt.vae_levels > 0
    assert opt.disc_loss_weight > 0


    ## Color
    clear = colorama.Style.RESET_ALL
    blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
    green = colorama.Fore.GREEN + colorama.Style.BRIGHT
    magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT


    ## Define & Initialize
    # Saver
    opt.saver = utils.VideoSaver(opt)

    # Tensorboard Summary
    # opt.summary = utils.TensorboardSummary(opt.saver.experiment_dir)
    logger.configure_logging(os.path.abspath(os.path.join(opt.saver.experiment_dir, 'logbook.txt')))

    # Device
    device = mindspore.get_context('device_target')
    opt.device = device

    # Config
    opt.noise_amp_init = opt.noise_amp
    opt.scale_factor_init = opt.scale_factor

    # Adjust scales
    utils.adjust_scales2image(opt.img_size, opt)

    # Manual seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    logging.info("Random Seed: {}".format(opt.manualSeed))
    random.seed(opt.manualSeed)
    mindspore.set_seed(opt.manualSeed)

    # Reconstruction loss
    opt.rec_loss = nn.RMSELoss()

    # Initial parameters
    opt.scale_idx = 0
    opt.nfc_prev = 0
    opt.Noise_Amps = []

    # Dataset
    dataset_generator = SingleVideoDataset(opt)
    dataset = GeneratorDataset(dataset_generator, ['data', 'zero-scale data'],shuffle=True)
    dataset = dataset.batch(opt.batch_size)
    dataset = dataset.shuffle(4)
    data_loader = dataset.create_dict_iterator()

    if opt.stop_scale_time == -1:
        opt.stop_scale_time = opt.stop_scale

    opt.dataset = dataset
    opt.data_loader = data_loader


    ## Load
    with open(os.path.join(opt.saver.experiment_dir, 'args.txt'), 'w') as args_file:
        for argument, value in sorted(vars(opt).items()):
            if type(value) in (str, int, float, tuple, list, bool):
                args_file.write('{}: {}\n'.format(argument, value))

    with logger.LoggingBlock("Commandline Arguments", emph=True):
        for argument, value in sorted(vars(opt).items()):
            if type(value) in (str, int, float, tuple, list):
                logging.info('{}: {}'.format(argument, value))

    with logger.LoggingBlock("Experiment Summary", emph=True):
        video_file_name, checkname, experiment = opt.saver.experiment_dir.split('/')[-3:]
        logging.info("{}Video file :{} {}{}".format(magenta, clear, video_file_name, clear))
        logging.info("{}Checkname  :{} {}{}".format(magenta, clear, checkname, clear))
        logging.info("{}Experiment :{} {}{}".format(magenta, clear, experiment, clear))

        with logger.LoggingBlock("Commandline Summary", emph=True):
            logging.info("{}Start frame    :{} {}{}".format(blue, clear, opt.start_frame, clear))
            logging.info("{}Max frames     :{} {}{}".format(blue, clear, opt.max_frames, clear))
            logging.info("{}Generator      :{} {}{}".format(blue, clear, opt.generator, clear))
            logging.info("{}Iterations     :{} {}{}".format(blue, clear, opt.niter, clear))
            logging.info("{}Rec. Weight    :{} {}{}".format(blue, clear, opt.rec_weight, clear))
            logging.info("{}Sampling rates :{} {}{}".format(blue, clear, opt.sampling_rates, clear))


    ## Current networks
    assert hasattr(networks_3d, opt.generator)
    netG = getattr(networks_3d, opt.generator)(opt)

    if opt.netG != '':
        if not os.path.isfile(opt.netG):
            raise RuntimeError("=> no <G> checkpoint found at '{}'".format(opt.netG))
        checkpoint = mindspore.load_checkpoint(opt.netG)
        opt.scale_idx = checkpoint['scale']
        opt.resumed_idx = checkpoint['scale']
        opt.resume_dir = '/'.join(opt.netG.split('/')[:-1])
        for _ in range(opt.scale_idx):
            netG.init_next_stage()
        netG.load_param_into_net(checkpoint['state_dict'])
        # Noise Amp
        opt.Noise_Amps = mindspore.load_checkpoint(os.path.join(opt.resume_dir, 'Noise_Amps.ckpt'))['data']
    else:
        opt.resumed_idx = -1


    ## Train
    while opt.scale_idx < opt.stop_scale + 1:
        if (opt.scale_idx > 0) and (opt.resumed_idx != opt.scale_idx):
            netG.init_next_stage()
        train(opt, netG)

        # Increase scale
        opt.scale_idx += 1
