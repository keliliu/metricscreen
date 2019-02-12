import numpy as np


# given vector of class probabilities computes the within and between class weights
def pairwise_concordance_prob(class_p):
    """
    From 1-D array of class probabilities, P(Y_i = 1), computes the probability that any
    pair of individuals will be from the same class (i.e. both 0 or both 1)
    :param class_p: nx1 1-D array of class probabilities.
    :return: n*(n-1)/2 1-D array containing probabilities for each pair of individuals
    to be in the same class
    """
    nn = class_p.size

    w_within = np.empty(shape=((nn * (nn - 1)) // 2,))

    pairs = [(i, j) for i in range(nn) for j in np.arange(i + 1, nn, 1)]
    ctr = 0

    for pair in pairs:
        w_within[ctr] = class_p[pair[0]] * class_p[pair[1]] + (1 - class_p[pair[0]]) * (1 - class_p[pair[1]])
        ctr += 1

    return w_within


# deprecated version of forming within group differences
def form_diff_within_slow(x):
    nn = x.shape[0]
    pp = x.shape[1]
    D = np.empty(shape=(((nn * (nn - 1)) // 2), pp))

    pairs = [(i, j) for i in range(nn) for j in np.arange(i + 1, nn, 1)]
    ctr = 0
    for pair in pairs:
        D[ctr, :] = np.power(x[pair[0], :] - x[pair[1], :], 2)
        ctr += 1

    return D


# deprecated version of computing pairwise concordance probability
def pairwise_concordance_prob_slow(x):
    w_within = scipy.spatial.distance.pdist(x.reshape((-1, 1)),\
                                            metric=lambda p, q: p * q + (1 - p) * (1 - q))

    return w_within


def form_diff_btw(x0,x1):
    n0 = x0.shape[0]
    n1 = x1.shape[0]
    pp = x0.shape[1]
    D = np.empty(shape = (n0*n1, pp))
   
    pairs = [(i,j) for i in range(n0) for j in range(n1)]
    ctr = 0
    for pair in pairs:
        D[ctr,:] = np.power(x0[pair[0],:] - x1[pair[1],:], 2)
        ctr += 1

    return D