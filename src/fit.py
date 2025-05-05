import numpy as np
import lmfit
from models import broken_powerlaw_1break, broken_powerlaw_2break


def fit_broken_powerlaw(bin_centers, e3j, e3j_err, n_breaks=1):
    if n_breaks == 1:
        model = lmfit.Model(broken_powerlaw_1break)
        params = model.make_params(
            A=1e24, m1=-2.7, m2=-4.6, logE_break=19.83)
        params['A'].set(min=1e20, max=1e30)
        params['m1'].set(min=-10, max=+10)
        params['m2'].set(min=-10, max=+10)
        params['logE_break'].set(min=19.5, max=20.0)

    elif n_breaks == 2:
        model = lmfit.Model(broken_powerlaw_2break)
        params = model.make_params(
            A1=1e24, m1=-2.6, m2=-2.83, m3=-4.6, logE1=19.15, logE2=19.83)
        params['A1'].set(min=1e20, max=1e30)
        params['m1'].set(min=-10, max=+10)
        params['m2'].set(min=-10, max=+10)
        params['m3'].set(min=-10, max=-1)
        params['logE1'].set(min=18.8, max=19.3)
        params['logE2'].set(min=19.5, max=20.0)

    else:
        raise ValueError("Only n_breaks = 1 or 2 are supported.")

    sigma = np.asarray(e3j_err)
    result = model.fit(e3j, params, x=bin_centers, weights=1/sigma)
    print(result.fit_report())
    return result
