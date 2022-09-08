import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from utils import calc_gradient_penalty
from .networks_2d import GeneratorHPVAEGAN, WDiscriminator2D


def kl_criterion(mu, logvar):
    KLD = -0.5 * (1 + logvar - ops.Pow()(mu, 2) - ops.Exp()(logvar))
    return KLD.mean()


def kl_bern_criterion(x):
    log = ops.Log()
    matmul = ops.MatMul()
    KLD = matmul(x, log(x + 1e-20) - log(0.5)) + matmul(1 - x, log(1 - x + 1e-20) - log(1 - 0.5))
    return KLD.mean()


class DWithLoss(nn.Cell):
    def __init__(self, opt, netD, netG):
        super(DWithLoss, self).__init__(auto_prefix=False)
        self._opt = opt

        self._netD = netD
        self._netG = netG

        self.fake = 0
        self.gradient_penalty = 0

    def construct(self, real, noise_init, noist_amps):
        fake, _ = self._netG(noise_init, noist_amps, noise_init=noise_init, isRandom=True)
        fake = ops.stop_gradient(fake)
        self.fake = fake

        # Train with real
        output_real = self.backbone_network(real)
        errD_real = -output_real.mean()

        # Train with fake
        output_fake = self.backbone_network(fake)
        errD_fake = output_fake.mean()

        # Gradient penalty
        # gradient_penalty = calc_gradient_penalty(self.backbone_network, real, fake, self._opt.lambda_grad)

        # Total error for Discriminator
        errD_total = errD_real + errD_fake + self.gradient_penalty
        return errD_total

    @property
    def backbone_network(self):
        return self._netD


class GWithLoss(nn.Cell):
    def __init__(self, opt, netD, netG):
        super(GWithLoss, self).__init__(auto_prefix=False)
        self._netD = netD
        self._netG = netG

        self.rec_loss = opt.rec_loss
        self.rec_weight = opt.rec_weight
        self.kl_weight = opt.kl_weight
        self.disc_loss_weight = opt.disc_loss_weight

    def construct(self, real, real_zero, fake, noise_amps, isVAE=False):
        fake = ops.stop_gradient(fake)
        generated, generated_vae, mu, logvar = self.backbone_network(real_zero, noise_amps, isRandom=False)
        # TODO: 前两个参数+元组, 目前无bern
        if isVAE:
            ## (1) VAE loss
            rec_vae_loss = self.rec_loss(generated, real) + self.rec_loss(generated_vae, real_zero)
            kl_loss = kl_criterion(mu, logvar)
            vae_loss = self.rec_weight * rec_vae_loss + self.kl_weight * kl_loss

            return vae_loss
        else:
            ## (2) Generator loss
            rec_loss = self.rec_loss(generated, real)
            errG_total = self.rec_weight * rec_loss

            # Train with Discriminator(fake)
            output = self._netD(fake)
            errG = -output.mean() * self.disc_loss_weight
            errG_total += errG

            return errG_total

    @property
    def backbone_network(self):
        return self._netG


if __name__ == '__main__':
    class Opt:
        def __init__(self):
            self.nfc = 64
            self.nc_im = 3
            self.ker_size = 3
            self.num_layer = 5
            self.latent_dim = 128
            self.enc_blocks = 2
            self.padd_size = 1
            self.image_path = '../data/imgs/air_balloons.jpg'
            self.hflip = True
            self.img_size = 256
            self.scale_factor = 0.75
            self.stop_scale = 9
            self.scale_idx = 0
            self.vae_levels = 3
            self.Noise_Amps = [1]
            self.rec_loss = nn.RMSELoss()
            self.rec_weight = 10.0

    opt = Opt()
    netD = WDiscriminator2D(opt)
    netG = GeneratorHPVAEGAN(opt)
    Gloss = GWithLoss(opt, netD, netG)

    normal = ops.StandardNormal()
    real_zero = normal((1, 3, 24, 33))
    real = normal((1, 3, 24, 33))
    noise_init = normal((2, 128, 24, 33))

    generated, generated_vae, (mu, logvar) = netG(real_zero, opt.Noise_Amps, randMode=False)
    fake, _ = netG(noise_init, opt.Noise_Amps, noise_init=noise_init, randMode=True)
    result = Gloss(real, real_zero, fake, generated, generated_vae, mu, logvar)
    print(result)