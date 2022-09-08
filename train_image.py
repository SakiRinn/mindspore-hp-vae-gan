import argparse
import utils
import random
import os
import colorama
import logging

from utils import logger, tools
from modules import networks_2d
from modules.losses import DWithLoss, GWithLoss
from modules.optimizers import ClippedAdam
from datasets import SingleImageDataset

from mindspore import context, Tensor
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset


def train(opt, netG):
    ############
    ### INIT ###
    ############

    ## Current Networks
    D_curr = getattr(networks_2d, opt.discriminator)(opt)
    G_curr = netG

    if opt.vae_levels < opt.scale_idx + 1:
        # Load parameters for discriminator
        if (opt.netG != '') and (opt.resumed_idx == opt.scale_idx):
            mindspore.load_checkpoint(f'{opt.resume_dir}/netD_{opt.scale_idx - 1}.ckpt', D_curr)
        elif opt.vae_levels < opt.scale_idx:
            mindspore.load_checkpoint(f'{opt.saver.experiment_dir}/netD_{opt.scale_idx - 1}.ckpt', D_curr)

        # Optimizer
        optimizerD = nn.Adam(D_curr.trainable_params(), opt.lr_d, beta1=opt.beta1, beta2=0.999)
        # With-loss cell
        D_loss = DWithLoss(opt, D_curr, G_curr)
        # Train-one-step cell
        D_train = nn.TrainOneStepCell(D_loss, optimizerD)


    ## Generator
    parameter_list = []

    if not opt.train_all:
        # (1) NOT train all
        if opt.vae_levels < opt.scale_idx + 1:
            train_depth = min(opt.train_depth, len(netG.body) - opt.vae_levels + 1)
            parameter_list += [
                {"params": block.trainable_params(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-train_depth:])
            ]
        else:
            parameter_list += [{"params": netG.encode.trainable_params(),
                                "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder.trainable_params(),
                                "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}
            ]
            parameter_list += [
                {"params": block.trainable_params(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-opt.train_depth:])
            ]
    else:
        # (2) train all
        if len(netG.body) < opt.train_depth:
            parameter_list += [{"params": netG.encode.trainable_params(),
                                "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder.trainable_params(),
                                "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}
            ]
            parameter_list += [
                {"params": block.trainable_params(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body) - 1 - idx))}
                for idx, block in enumerate(netG.body)
            ]
        else:
            parameter_list += [
                {"params": block.trainable_params(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-opt.train_depth:])
            ]

    # Optimizer
    optimizerG = ClippedAdam(opt, parameter_list, opt.lr_g, beta1=opt.beta1, beta2=0.999)
    # With-loss cell
    G_loss = GWithLoss(opt, D_curr, G_curr)
    # Train-one-step cell
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



    #############
    ### TRAIN ###
    #############
    total_loss = 0
    iterator = iter(opt.data_loader)

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

        initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        initial_size = [int(initial_size * opt.ar), initial_size]
        opt.Z_init_size = [opt.batch_size, opt.latent_dim, *initial_size]

        noise_init = utils.generate_noise_size(opt.Z_init_size)


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
                    z_reconstruction, _, _, _ = G_curr(real_zero, opt.Noise_Amps, isRandom=False)
                    RMSE = nn.RMSELoss()(real, z_reconstruction)
                    RMSE = ops.stop_gradient(RMSE)

                    opt.noise_amp = opt.noise_amp_init * RMSE / opt.batch_size
                    opt.Noise_Amps[-1] = opt.noise_amp.asnumpy().item()


        ## Update parameters
        fake, generated, generated_vae = None, None, None
        if opt.vae_levels >= opt.scale_idx + 1:
            # (1) Update VAE network
            curG_loss = G_train(real, real_zero, 0, opt.Noise_Amps, True)
        else:
            # (2) Update distriminator: maximize D(x) + D(G(z))
            curD_loss = D_train(real, noise_init, opt.Noise_Amps)
            fake = D_loss.fake
            # (3) Update generator: maximize D(G(z)) (After grad clipping)
            curG_loss = G_train(real, real_zero, fake, opt.Noise_Amps, False)
            generated = G_loss.generated
            generated_vae = G_loss.generated_vae
        total_loss += curG_loss


        ## Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
        ))


        ## Log
        if (iteration + 1) % opt.print_interval == 0:
            if opt.vae_levels >= opt.scale_idx + 1:
                logging.info(f'[Scale {opt.scale_idx + 1}/Iter {iteration}] Noise amp: {opt.noise_amp}, Gloss: {curG_loss}')
            else:
                logging.info(f'[Scale {opt.scale_idx + 1}/Iter {iteration}] Noise amp: {opt.noise_amp}, Gloss: {curG_loss}, Dloss: {curD_loss}')

            if opt.visualize:
                opt.saver.save_images(real, 'real.jpg')

                if opt.vae_levels < opt.scale_idx + 1:
                    opt.saver.save_images(generated, f'generated_{iteration}.jpg')
                    opt.saver.save_images(generated_vae, f'generated_vae_{iteration}.jpg')

                if iteration % opt.image_interval == 0:
                    fake_var = []
                    fake_vae_var = []
                    for _ in range(3):
                        noise_init = utils.generate_noise_ref(noise_init)
                        noise_init = ops.stop_gradient(noise_init)
                        fake, fake_vae = G_curr(noise_init, opt.Noise_Amps, noise_init=noise_init, isRandom=True)
                        fake_var.append(fake)
                        fake_vae_var.append(fake_vae)
                    fake_var = ops.Concat()(fake_var)
                    fake_vae_var = ops.Concat()(fake_vae_var)
                    opt.saver.save_images(fake_var, f'fake_var_{iteration}.jpg')
                    opt.saver.save_images(fake_vae_var, f'fake_vae_var{iteration}.jpg')


        ## Virsualize with Tensorboard
        # if opt.visualize:
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
        #         # with torch.no_grad():
        #         fake_var = []
        #         fake_vae_var = []
        #         for _ in range(3):
        #             noise_init = utils.generate_noise(ref=noise_init)
        #             noise_init = ops.stop_gradient(noise_init)
        #             fake, fake_vae = G_curr(noise_init, opt.Noise_Amps, noise_init=noise_init, mode="rand")
        #             fake_var.append(fake)
        #             fake_vae_var.append(fake_vae)
        #         fake_var = cat(fake_var)
        #         fake_vae_var = cat(fake_vae_var)

        #         opt.summary.visualize_image(opt, iteration, real, 'Real')
        #         opt.summary.visualize_image(opt, iteration, generated, 'Generated')
        #         opt.summary.visualize_image(opt, iteration, generated_vae, 'Generated VAE')
        #         opt.summary.visualize_image(opt, iteration, fake_var, 'Fake var')
        #         opt.summary.visualize_image(opt, iteration, fake_vae_var, 'Fake VAE var')

    epoch_iterator.close()


    ## Save data
    opt.saver.save_json({'noise_amps': opt.Noise_Amps, 'scale_idx': opt.scale_idx}, 'config.json')
    opt.saver.save_checkpoint(G_curr, 'netG.ckpt')
    if opt.vae_levels < opt.scale_idx + 1:
        opt.saver.save_checkpoint(D_curr, f'netD_{opt.scale_idx}.ckpt')


if __name__ == '__main__':
    context.set_context(mode=1, device_id=4)

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
    parser.add_argument('--discriminator', type=str, default='WDiscriminator2D', help='discriminator model')

    # Pyramid parameters
    parser.add_argument('--scale-factor', type=float, default=0.75, help='pyramid scale factor')
    parser.add_argument('--noise_amp', type=float, default=0.1, help='addative noise cont weight')
    parser.add_argument('--min-size', type=int, default=32, help='image minimal size at the coarser scale')
    parser.add_argument('--max-size', type=int, default=256, help='image minimal size at the coarser scale')

    # Optimization hyper parameters
    parser.add_argument('--niter', type=int, default=5000, help='number of iterations to train per scale')
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
    parser.add_argument('--image-path', required=True, help="image path")
    parser.add_argument('--hflip', action='store_true', default=False, help='horizontal flip')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--stop-scale-time', type=int, default=-1)
    parser.add_argument('--data-rep', type=int, default=1000, help='data repetition')

    # Main arguments
    parser.add_argument('--checkname', type=str, default='debug', help='check name')
    parser.add_argument('--mode', default='train', help='task to be done')
    parser.add_argument('--print-interval', type=int, default=10, help='print interval')
    parser.add_argument('--image-interval', type=int, default=100, help='image interval')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--visualize', action='store_true', default=True, help='visualize using tensorboard')
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda')

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()

    assert opt.vae_levels > 0
    assert opt.disc_loss_weight > 0

    if opt.data_rep < opt.batch_size:
        opt.data_rep = opt.batch_size


    ## Color
    clear = colorama.Style.RESET_ALL
    blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
    green = colorama.Fore.GREEN + colorama.Style.BRIGHT
    magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT


    ## Define & Initialize
    # Saver
    opt.saver = utils.ImageSaver(opt)

    # Tensorboard Summary
    # opt.summary = utils.TensorboardSummary(opt.saver.experiment_dir)
    logger.configure_logging(os.path.abspath(os.path.join(opt.saver.experiment_dir, 'logbook.txt')))

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
    opt.rec_loss = nn.RMSELoss()

    # Initial parameters
    opt.scale_idx = 0
    opt.nfc_prev = 0
    opt.Noise_Amps = []

    # Dataset
    dataset = SingleImageDataset(opt)
    data_loader = GeneratorDataset(dataset, ['data', 'zero-scale data'], shuffle=True)
    data_loader = data_loader.batch(opt.batch_size)
    data_loader = data_loader.shuffle(4)

    if opt.stop_scale_time == -1:
        opt.stop_scale_time = opt.stop_scale

    opt.dataset = dataset
    opt.data_loader = data_loader


    ## Logging
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
        logging.info("{}Checkname  :{} {}{}".format(magenta, clear, checkname, clear))
        logging.info("{}Experiment :{} {}{}".format(magenta, clear, experiment, clear))

        with logger.LoggingBlock("Commandline Summary", emph=True):
            logging.info("{}Generator      :{} {}{}".format(blue, clear, opt.generator, clear))
            logging.info("{}Iterations     :{} {}{}".format(blue, clear, opt.niter, clear))
            logging.info("{}Rec. Weight    :{} {}{}".format(blue, clear, opt.rec_weight, clear))


    ## Current networks
    assert hasattr(networks_2d, opt.generator)
    netG = getattr(networks_2d, opt.generator)(opt)

    if opt.netG != '':
        if not os.path.isfile(opt.netG):
            raise RuntimeError(f"=> no <G> checkpoint found at '{opt.netG}'")
        checkpoint = mindspore.load_checkpoint(opt.netG)
        opt.scale_idx = opt.saver.load_json('config.json')['scale']
        opt.resumed_idx = opt.saver.load_json('config.json')['scale']
        opt.resume_dir = '/'.join(opt.netG.split('/')[:-1])
        for _ in range(opt.scale_idx):
            netG.init_next_stage()
        netG.load_checkpoint(opt.netG)
        # Noise Amp
        opt.Noise_Amps = opt.saver.load_json('config.json')['noise_amps']
    else:
        opt.resumed_idx = -1


    ## Train
    while opt.scale_idx < opt.stop_scale + 1:
        if (opt.scale_idx > 0) and (opt.resumed_idx != opt.scale_idx):
            netG.init_next_stage()
        train(opt, netG)

        # Increase scale
        opt.scale_idx += 1
