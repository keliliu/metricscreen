import numpy as np


def perm_false_positive(screener, X, y, grid_min_scale=0.01, grid_range=20, grid_size=15, num_perm=5, **kwargs):

    n, p = X.shape

    # form penalty grid
    grid_min = grid_min_scale*np.sqrt(np.log(p)/np.min([np.sum(y), np.sum(1-y)]))
    grid_max = grid_range*grid_min
    grid = np.exp(np.linspace(np.log(grid_min), np.log(grid_max), num=grid_size))

    num_chosen = np.zeros((grid_size, num_perm))

    for jj in range(num_perm):

        yperm = np.random.permutation(y)

        for ii in range(grid_size):
            screener.reset()
            screener.train(X=X, class_p=yperm, lam=grid[ii], **kwargs)
            num_chosen[ii,jj] = len(screener.get_vars())

    return grid, num_chosen


def find_penalty(grid, fpr, fpr_cutoff=0.025):
    if np.min(np.average(fpr, axis=1)) >= fpr_cutoff:
        return np.max(grid)
    else:
        return np.min(grid[np.average(fpr, axis=1) < fpr_cutoff])