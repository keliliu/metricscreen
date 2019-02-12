import numpy as np


def _logit(p):

    ptrunc = np.clip(p, a_min=1e-8, a_max=1 - 1e-8)
    return np.log(ptrunc/(1-ptrunc))


def _expit(eta):
    return 1/(1+np.exp(-eta))