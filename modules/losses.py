import mindspore.ops as ops
import mindspore.nn as nn
from modules import calc_gradient_penalty

log = ops.Log()
matmul = ops.MatMul()
total_loss = 0


def kl_criterion(mu, logvar):
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return KLD.mean()


def kl_bern_criterion(x):
    KLD = matmul(x, log(x + 1e-20) - log(0.5)) + matmul(1 - x, log(1 - x + 1e-20) - log(1 - 0.5))
    return KLD.mean()


class DWithLoss(nn.Cell):
    def __init__(self, opt, netD, netG):
        super(DWithLoss, self).__init__(auto_prefix=False)
        self._netD = netD
        self._netG = netG
        self._opt = opt

    def construct(self, real, fake):
        # Train with real
        output = self._netD(real)
        errD_real = -output.mean()

        # Train with fake
        output = self._netD(ops.stop_gradient(fake))
        errD_fake = output.mean()

        # Total error for Discriminator
        gradient_penalty = calc_gradient_penalty(self._netD, real, fake, self._opt.lambda_grad)
        errD_total = errD_real + errD_fake + gradient_penalty
        return errD_total

    @property
    def backbone_network(self):
        return self._netD


class GWithLoss(nn.Cell):
    def __init__(self, opt, netG):
        super(GWithLoss, self).__init__(auto_prefix=False)
        self._netG = netG
        self._opt = opt
        self.total_loss = 0
        self.isVAE = False

    def VAEMode(self, flag):
        self.isVAE = flag

    def construct(self, real, real_zero, fake, generated, generated_vae, mu, logvar):
        if self.isVAE:
            ## (1) VAE loss
            rec_vae_loss = self._opt.rec_loss(generated, real) + self._opt.rec_loss(generated_vae, real_zero)
            kl_loss = kl_criterion(mu, logvar)
            vae_loss = self._opt.rec_weight * rec_vae_loss + self._opt.kl_weight * kl_loss

            self.total_loss += vae_loss
        else:
            ## (2) G loss
            rec_loss = self._opt.rec_loss(generated, real)
            errG_total = self._opt.rec_weight * rec_loss

            # Train with Discriminator(fake)
            output = self._netD(fake)
            errG = -output.mean() * self._opt.disc_loss_weight
            errG_total += errG

            self.total_loss += errG_total

        return self.total_loss

    @property
    def backbone_network(self):
        return self._netG