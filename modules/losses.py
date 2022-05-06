import mindspore.ops as ops

__all__ = ['kl_criterion', 'kl_bern_criterion']
log = ops.Log()
matmul = ops.MatMul()


def kl_criterion(mu, logvar):
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return KLD.mean()


def kl_bern_criterion(x):
    KLD = matmul(x, log(x + 1e-20) - log(0.5)) + matmul(1 - x, log(1 - x + 1e-20) - log(1 - 0.5))
    return KLD.mean()