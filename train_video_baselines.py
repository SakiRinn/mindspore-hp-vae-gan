import os
import argparse
import random
import logging
import colorama

import mindspore
import mindspore.nn as nn
from mindspore import context
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset

import src.utils as utils
from src.modules import networks_3d
from src.datasets import SingleVideoDataset
from src.modules.losses import DWithLoss, GWithLoss
from src.utils import logger, progress_bar
import src.tools.pt2ms as pt2ms


def train(opt, netG):
    ############
    ### INIT ###
    ############

    ## Re-generate dataset frames
    fps, td, fps_index = utils.get_fps_td_by_index(opt.scale_idx, opt.stop_scale_time,
                                                   opt.sampling_rates, opt.org_fps, opt.fps_lcm)
    opt.fps = fps
    opt.td = td
    opt.fps_index = fps_index

    with logger.LoggingBlock("Updating dataset", emph=True):
        logging.info("{}FPS :{} {}{}".format(green, clear, opt.fps, clear))
        logging.info("{}Time-Depth :{} {}{}".format(green, clear, opt.td, clear))
        logging.info("{}Sampling-Ratio :{} {}{}".format(green, clear, opt.sampling_rates[opt.fps_index], clear))
        opt.dataset.generate_frames(opt.scale_idx)


    ## Noise
    if not hasattr(opt, 'Z_init'):
        initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        initial_size = [int(initial_size * opt.ar), initial_size]
        opt.Z_init = utils.generate_noise_size([1, 3, opt.td, *initial_size])


    ## Current Networks
    D_curr = getattr(networks_3d, opt.discriminator)(opt)
    G_curr = netG

    if opt.scale_idx > 0:
        # Current discriminator
        checkpoint = mindspore.load_checkpoint(f'{opt.resume_dir}/netD_{opt.scale_idx - 1}.ckpt')
        mindspore.load_param_into_net(D_curr, checkpoint)

    # Optimizer
    optimizerD = nn.Adam(D_curr.trainable_params(), opt.lr_d, beta1=opt.beta1, beta2=0.999)
    # With-loss cell
    D_loss = DWithLoss(opt, D_curr, G_curr)
    # Train-one-step cell
    D_train = nn.TrainOneStepCell(D_loss, optimizerD)
    D_train.set_train()

    ## Generator
    # remove gradients from stages that are not trained
    for block in netG.body[:-opt.train_depth]:
        for param in block.parameters_dict().keys():
            param.requires_grad = False

    # set different learning rate for lower stages
    parameter_list = [
        {"params": block.trainable_params(),
         "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
        for idx, block in enumerate(netG.body[-opt.train_depth:])]

    # add parameters of head and tail to training
    if hasattr(netG, 'head'):
        if opt.scale_idx - opt.train_depth < 0:
            parameter_list += [{"params": netG.head.trainable_params(),
                                "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
    if hasattr(netG, 'tail'):
        parameter_list += [{"params": netG.tail.trainable_params(),
                            "lr": opt.lr_g}]
    # Optimizer
    optimizerG = nn.Adam(parameter_list, opt.lr_g, beta1=opt.beta1, beta2=0.999)
    # With-loss cell
    G_loss = GWithLoss(opt, D_curr, G_curr)
    # Train-one-step cell
    G_train = nn.TrainOneStepCell(G_loss, optimizerG)
    G_train.set_train()


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
    epoch_iterator = progress_bar.create_progressbar(**progressbar_args)


    #############
    ### TRAIN ###
    #############
    iterator = opt.data_loader.create_tuple_iterator()
    # idx = 0
    for iteration in epoch_iterator:
        try:
            data = next(iterator)
        except StopIteration:
            iterator = opt.data_loader.create_tuple_iterator()
            data = next(iterator)

        if opt.scale_idx > 0:
            real, real_zero = data
        else:
            real, _ = data
            real_zero = real.copy()

        noise_init = utils.generate_noise_ref(opt.Z_init.shape)


        ## Calculate noise_amp (First iteration)
        if iteration == 0:
            if opt.scale_idx == 0:
                opt.noise_amp = 1
                opt.Noise_Amps.append(opt.noise_amp)
            else:
                opt.Noise_Amps.append(0)
                return_list = G_curr(real_zero, opt.Noise_Amps, isRandom=False)
                z_reconstruction = return_list[0]
                RMSE = nn.RMSELoss()(real, z_reconstruction)

                opt.noise_amp = opt.noise_amp_init * RMSE / opt.batch_size
                opt.Noise_Amps[-1] = opt.noise_amp.asnumpy().item()


        ## Train
        # (1) Update distriminator: maximize D(x) + D(G(z))
        curD_loss = D_train(real, noise_init, opt.Noise_Amps)
        # (2) Update generator: maximize D(G(z)) (After grad clipping)
        curG_loss = G_train(real, real_zero, noise_init, opt.Noise_Amps, False)


        ## Verbose
        # Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
        ))

        # Print
        if (iteration + 1) % opt.print_interval == 0:
            if opt.vae_levels >= opt.scale_idx + 1:
                logging.debug('[Scale {}/Iter {}] Noise amp: {}, Gloss: {}'.format(
                    opt.scale_idx + 1, iteration + 1, opt.noise_amp, curG_loss))
            else:
                logging.debug('[Scale {}/Iter {}] Noise amp: {}, Gloss: {}, Dloss: {}'.format(
                    opt.scale_idx + 1, iteration + 1, opt.noise_amp, curG_loss, curD_loss))

        # Visualize
        if opt.visualize and (iteration + 1) % opt.image_interval == 0:
            # Real
            opt.saver.save_image(real, f'real_{iteration+1}.jpg')
            # Generated
            return_list = G_curr(real_zero, opt.Noise_Amps, isRandom=False)
            generated = return_list[0] * 255
            generated_vae = return_list[1] * 255
            opt.saver.save_image(generated, f'generated_{iteration+1}.jpg')
            opt.saver.save_image(generated_vae, f'generated_vae_{iteration+1}.jpg')
            # Fake
            fake_var = []
            fake_vae_var = []
            for _ in range(3):
                noise_init = utils.generate_noise_ref(noise_init.shape)
                noise_init = ops.stop_gradient(noise_init)
                return_list = G_curr(noise_init, opt.Noise_Amps, noise_init=noise_init, isRandom=True)
                fake_var.append(return_list[0])
                fake_vae_var.append(return_list[1])
            fake_var = ops.Concat()(fake_var) * 255
            fake_vae_var = ops.Concat()(fake_vae_var) * 255
            opt.saver.save_image(fake_var, f'fake_var_{iteration}.jpg')
            opt.saver.save_image(fake_vae_var, f'fake_vae_var{iteration}.jpg')

    epoch_iterator.close()


    ## Save data
    opt.saver.save_json({'noise_amps': opt.Noise_Amps, 'scale_idx': opt.scale_idx}, 'intermediate.json')
    opt.saver.save_checkpoint(G_curr, f'netG_{opt.scale_idx}.ckpt')
    if opt.vae_levels < opt.scale_idx + 1:
        opt.saver.save_checkpoint(D_curr, f'netD_{opt.scale_idx}.ckpt')


if __name__ == '__main__':
    ## Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-id', default=0, type=int, help='Device ID')

    # Load, input, save configurations
    parser.add_argument('--netG', default='', help='path to netG (to continue training)')
    parser.add_argument('--netD', default='', help='path to netD (to continue training)')
    parser.add_argument('--intermediate', default='', help='path to intermediate file')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    # Networks hyper parameters
    parser.add_argument('--nc-z', type=int, default=3, help='noise # channels')
    parser.add_argument('--nc-im', type=int, help='image # channels', default=3)
    parser.add_argument('--nfc', type=int, default=64, help='model basic # channels')
    parser.add_argument('--ker-size', type=int, default=3, help='kernel size')
    parser.add_argument('--num-layer', type=int, default=5, help='number of layers')
    parser.add_argument('--stride', default=1, help='stride')
    parser.add_argument('--padd-size', type=int, default=1, help='net pad size')
    parser.add_argument('--generator', type=str, help='Generator model', default='GeneratorCSG')
    parser.add_argument('--discriminator', type=str, help='Discriminator model', default='WDiscriminator3D')

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
    parser.add_argument('--disc-loss-weight', type=float, default=1.0, help='3D disc weight')
    parser.add_argument('--lambda-grad', type=float, default=0.1, help='gradient penelty weight')
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=10.)
    parser.add_argument('--lr-scale', type=float, default=0.2, help='scaling of learning rate for lower stages')
    parser.add_argument('--train-depth', type=int, default=1, help='how many layers are trained if growing')

    # Dataset
    parser.add_argument('--video-path', required=True, help='video path')
    parser.add_argument('--start-frame', default=0, type=int, help='start frame number')
    parser.add_argument('--max-frames', default=1000, type=int, help='# frames to save')
    parser.add_argument('--hflip', action='store_true', default=False, help='horizontal flip')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--sampling-rates', type=int, nargs='+', default=[4, 3, 2, 1], help='sampling rates')
    parser.add_argument('--stop-scale-time', type=int, default=-1)
    parser.add_argument('--data-rep', type=int, default=1, help='data repetition')

    # Main arguments
    parser.add_argument('--checkname', type=str, default='DEBUG', help='check name')
    parser.add_argument('--mode', default='train', help='task to be done')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--print-interval', type=int, default=10, help='print interval')
    parser.add_argument('--image-interval', type=int, default=100, help='image interval')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize the image')

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()

    context.set_context(mode=1, device_id=opt.device_id)

    assert opt.disc_loss_weight > 0


    ## Color
    clear = colorama.Style.RESET_ALL
    blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
    green = colorama.Fore.GREEN + colorama.Style.BRIGHT
    magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT

    ## Define & Initialize
    # Saver
    opt.saver = utils.DataSaver(opt)

    # Tensorboard Summary
    # opt.summary = utils.TensorboardSummary(opt.saver.experiment_dir)
    logger.configure_logging(os.path.abspath(os.path.join(opt.saver.experiment_dir, 'logbook.txt')))

    # Device
    # device = mindspore.get_context('device_target')
    # opt.device = device

    # Config
    opt.noise_amp_init = opt.noise_amp
    opt.scale_factor_init = opt.scale_factor

    # Adjust scales
    utils.adjust_scales2image(opt.img_size, opt)

    # Manual seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    logging.info(f"Random Seed: {opt.manualSeed}")
    random.seed(opt.manualSeed)
    mindspore.set_seed(opt.manualSeed)

    # Reconstruction loss
    opt.rec_loss = nn.MSELoss()

    # Initial parameters
    opt.scale_idx = 0
    opt.nfc_prev = 0
    opt.Noise_Amps = []

    # Dataset
    dataset = SingleVideoDataset(opt)
    data_loader = GeneratorDataset(dataset, ['data', 'zero-scale data'], shuffle=True, num_parallel_workers=4)
    data_loader = data_loader.batch(opt.batch_size, num_parallel_workers=4)
    data_loader = data_loader.shuffle(4)

    if opt.stop_scale_time == -1:
        opt.stop_scale_time = opt.stop_scale

    opt.dataset = dataset
    opt.data_loader = data_loader


    ## Load
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
            logging.info("{}Alpha          :{} {}{}".format(blue, clear, opt.alpha, clear))
            logging.info("{}Sampling rates :{} {}{}".format(blue, clear, opt.sampling_rates, clear))


    ## Current networks
    assert hasattr(networks_3d, opt.generator)
    netG = getattr(networks_3d, opt.generator)(opt).to(opt.device)

    if opt.netG != '':
        opt.intermediate = os.path.join(*opt.intermediate.split('/')[:-1]) if opt.intermediate[0] != '/' \
                           else '/' + os.path.join(*opt.intermediate.split('/')[:-1])
        if opt.intermediate == '':
            raise FileNotFoundError("intermediate file DOESN'T be empty.")
        # Init
        opt.Noise_Amps = opt.saver.load_json('intermediate.json', path=opt.intermediate)['noise_amps']
        opt.scale_idx = opt.saver.load_json('intermediate.json', path=opt.intermediate)['scale_idx']
        opt.resumed_idx = opt.saver.load_json('intermediate.json', path=opt.intermediate)['scale_idx']
        opt.resume_dir = os.path.join(*opt.netG.split('/')[:-1]) if opt.netG[0] != '/' \
                         else '/' + os.path.join(*opt.netG.split('/')[:-1])
        # Load
        if not os.path.isfile(opt.netG):
            raise RuntimeError(f"=> no <G> checkpoint found at '{opt.netG}'")
        checkpoint = mindspore.load_checkpoint(opt.netG)
        checkpoint = pt2ms.m2m_HPVAEGAN_3d(checkpoint)
        for _ in range(opt.scale_idx):
            netG.init_next_stage()
        mindspore.load_param_into_net(netG, checkpoint)
    else:
        opt.resumed_idx = -1


    ## Train
    while opt.scale_idx < opt.stop_scale + 1:
        if (opt.scale_idx > 0) and (opt.resumed_idx != opt.scale_idx):
            netG.init_next_stage()
        train(opt, netG)

        # Increase scale
        opt.scale_idx += 1
