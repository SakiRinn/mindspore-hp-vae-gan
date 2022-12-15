import mindspore.nn as nn
import mindspore.ops as ops


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

        self._netD = netD
        self._netG = netG

        self.lambda_grad = opt.lambda_grad
        self.alpha = ops.UniformReal()((1, 1))

    def construct(self, real, noise_init, noise_amps):
        # Fake
        return_list = self._netG(noise_init, noise_amps, noise_init=noise_init, isRandom=True)
        fake = ops.stop_gradient(return_list[0])

        # Train with real
        output_real = self.backbone_network(real)
        errD_real = -output_real.mean()

        # Train with fake
        output_fake = self.backbone_network(fake)
        errD_fake = output_fake.mean()

        # Gradient penalty
        gradient_penalty = self.calc_gradient_penalty(real, fake, self.lambda_grad)

        # Total error for Discriminator
        errD_total = errD_real + errD_fake + gradient_penalty
        return errD_total

    def calc_gradient_penalty(self, real, fake, LAMBDA=1):
        alpha = ops.BroadcastTo(real.shape)(self.alpha)
        interpolates = alpha * real + ((1 - alpha) * fake)
        gradients = ops.GradOperation()(self.backbone_network)(interpolates)
        gradient_penalty = ((ops.LpNorm(1, 2)(gradients) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

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

    def construct(self, real, real_zero, noise_init, noise_amps, isVAE=False):
        total_loss = 0
        # Forward
        return_list = self.backbone_network(real_zero, noise_amps, isRandom=False)
        generated = return_list[0]
        generated_vae = return_list[1]
        mu = return_list[2]
        logvar = return_list[3]

        if isVAE:
            ## (1) VAE loss
            rec_vae_loss = self.rec_loss(generated, real) + self.rec_loss(generated_vae, real_zero)
            kl_loss = kl_criterion(mu, logvar)
            vae_loss = self.rec_weight * rec_vae_loss + self.kl_weight * kl_loss

            total_loss += vae_loss
        else:
            ## (2) Generator loss
            errG_total = 0
            rec_loss = self.rec_loss(generated, real)
            errG_total = self.rec_weight * rec_loss

            # Fake
            return_list = self.backbone_network(noise_init, noise_amps, noise_init=noise_init, isRandom=True)
            fake = ops.stop_gradient(return_list[0])

            # Train with Discriminator
            output = self._netD(fake)
            errG = -output.mean() * self.disc_loss_weight
            errG_total += errG

            total_loss += errG_total

        return total_loss

    @property
    def backbone_network(self):
        return self._netG


if __name__ == '__main__':
    from mindspore import context
    context.set_context(mode=0, device_id=5)
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
            self.rec_loss = nn.MSELoss()
            self.rec_weight = 10.0

    opt = Opt()
    print(ops.UniformReal()((1, 1)))