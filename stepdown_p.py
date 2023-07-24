import numpy as np
from tqdm import trange

def stepdown_p(stat, stat_boot):
    """
    Calculate corrected p values

    stat - array shape N - "true" values of the test statistic
    stat_boot - array shape M x N - the values of the test statistic
        obtained from M bootstraps (or permutations, or ...)
    """
    M, N = stat_boot.shape
    if not N == len(stat):
        raise ValueError('length of stat must match number of variables'
                ' in stat_boot')
    # order the test hypotheses with decreasing significance 
    order = np.argsort(stat)[::-1]
    stat = stat[order]
    stat_boot = stat_boot[:,order]
    # initialize results array
    p = [(np.sum(np.max(stat_boot[:,i:], 1) >= stat[i]) + 1)/float(M + 1)
            for i in trange(N)]
    # enforce monotonicity
    p = np.maximum.accumulate(p)
    # revert the original order of the hypothesis
    return p[np.argsort(order)]
