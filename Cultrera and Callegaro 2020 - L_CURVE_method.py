import numpy as np

def l_curve_corner_search(lam_min, lam_max, f, tol=1E-5, args=[], kwargs={},):
    """
    Calculate optimal regularization parameter according to:

    A. Cultrera, L. Callegaro
    A simple algorithm to find the L-curve corner in the regularisation of
    ill-posed inverse problems.
    IOPSciNotes. 1, 025004 (2020).
    """
    gs = (1 + np.sqrt(5))/2
    x_min = np.log10(lam_min)
    x_max = np.log10(lam_max)
    x = [x_min,
         (x_max + gs * x_min) / (1 + gs)]
    x.extend([
         x_min + (x_max - x[1]),
         x_max])
    lam = 10**np.r_[x]
    P1, P2, P3, P4 = [f(lam_now, *args, **kwargs) for lam_now in lam]
    while (lam[3] - lam[0])/lam[3] > tol:
        C2 = menger(P1, P2, P3)
        C3 = menger(P2, P3, P4)
        while C3 <= 0:
            x[3] = x[2]; lam[3] = lam[2]; P4 = P3
            x[2] = x[1]; lam[2] = lam[1]; P3 = P2
            x[1] = (x[3] + gs * x[0]) / (1 + gs); lam[1] = 10**x[1]
            P2 = f(lam[1], *args, **kwargs)
            C3 = menger(P2, P3, P4)
        if C2 > C3:
            store_lam = lam[1]
            x[3] = x[2]; lam[3] = lam[2]; P4 = P3
            x[2] = x[1]; lam[2] = lam[1]; P3 = P2
            x[1] = (x[3] + gs * x[0]) / (1 + gs); lam[1] = 10**x[1]
            P2 = f(lam[1], *args, **kwargs)
        else:
            store_lam = lam[2]
            x[0] = x[1]; lam[0] = lam[1]; P1 = P2
            x[1] = x[2]; lam[1] = lam[2]; P2 = P3
            x[2] = x[0] + (x[3] - x[1]); lam[2] = 10**x[2]
            P3 = f(lam[2], *args, **kwargs)
    return store_lam


