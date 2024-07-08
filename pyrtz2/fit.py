import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score as r2


# Functions
def powerlaw(x, a, b):
    return a * (x ** b)


def poly(x, a, b, c):
    return a * (x ** b) + c * x


def hertzian(ind, diameter, e_star):
    return (4 / 3) * e_star * ((0.5 * diameter) ** 0.5) * (ind ** 1.5)


def exponential(t, y_0, tau, y_f):
    return (y_0 - y_f) * np.exp(-t * tau) + y_f


def biexponential(t, y_0, c, tau1, tau2, y_f):
    return (y_0 - y_f) * (c * np.exp(-t * tau1) + (1 - c) * np.exp(-t * tau2)) + y_f


def poroelastic(t, y_0, c, tau1, tau2, y_f):
    return (y_0 - y_f) * (c * np.exp(-np.sqrt(t) * tau1) + (1 - c) * np.exp(-t * tau2)) + y_f


# Fitting
def lin_fit(x, y):
    popt = np.polyfit(x, y, 1)
    y_pred = np.poly1d(popt)(x)
    r2_score = r2(y, y_pred)
    return popt, r2_score, y_pred


def calculate_residuals(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)


def powerlaw_fit(x, y):
    x_min = min(x)
    y_min = min(y)
    x = (x - x_min)
    x_max = max(x)
    x = x / x_max
    y = (y - y_min)
    y_max = max(y)
    y = y / y_max
    bounds = ([0, 0], [np.inf, np.inf])
    popt, _ = curve_fit(powerlaw, x, y, bounds=bounds)
    y_pred = powerlaw(x, *popt)
    r2_score = r2(y, y_pred)
    return popt, r2_score, y_pred * y_max + y_min


def poly_fit(x, y):
    x_min = min(x)
    y_min = min(y)
    x = (x - x_min)
    x_max = max(x)
    x = x / x_max
    y = (y - y_min)
    y_max = max(y)
    y = y / y_max
    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    p0 = [1, 1, 1]
    popt, _ = curve_fit(poly, x, y, bounds=bounds, p0=p0, jac='3-point')
    y_pred = poly(x, *popt)
    r2_score = r2(y, y_pred)
    return popt, r2_score, y_pred * y_max + y_min


def hertzian_fit(x, y, diameter):
    def wrapper(ind, e_star):
        return hertzian(ind, diameter, e_star)

    popt, _ = curve_fit(wrapper, x, y)
    y_pred = hertzian(x, diameter, *popt)
    r2_score = r2(y, y_pred)
    return popt, r2_score, y_pred


def exponential_fit(x, y, bound=False):
    x -= x[0]
    if bound:
        def bnd_wrapper(t, tau):
            return exponential(t, y[0], tau, y[-1])

        bounds = ([0], [np.inf])
        popt, _ = curve_fit(bnd_wrapper, x, y, bounds=bounds)
        args = [y[0], *popt, y[-1]]
    else:
        def ubnd_wrapper(t, tau, y_f):
            return exponential(t, y[0], tau, y_f)

        bounds = ([0, -np.inf], [np.inf, np.inf])
        popt, _ = curve_fit(ubnd_wrapper, x, y, bounds=bounds)
        args = [y[0], *popt]

    y_pred = exponential(x, *args)
    r2_score = r2(y, y_pred)
    return popt, r2_score, y_pred


def biexponential_fit(x, y, bound=False):
    x -= x[0]
    if bound:
        def bnd_wrapper(t, c, tau1, tau2):
            return biexponential(t, y[0], c, tau1, tau2, y[-1])

        bounds = ([0, 0, 0], [1, np.inf, np.inf])
        p0 = [0.4, 1, 0.1]
        popt, _ = curve_fit(bnd_wrapper, x, y,
                            bounds=bounds, p0=p0, jac='3-point')
        # args = [y[0], *popt, y[-1]]

        # y_pred = biexponential(x, *args)
        # r2_score = r2(y, y_pred)

    else:
        def ubnd_wrapper(t, c, tau1, tau2, y_f):
            return biexponential(t, y[0], c, tau1, tau2, y_f)

        results = {}

        p0 = [0.4, 1, 0.1, y[-1]]
        bounds = ([0, 0, 0, -np.inf], [1, np.inf, np.inf, np.inf])

        popt_with_p0, _ = curve_fit(
            ubnd_wrapper, x, y, bounds=bounds, p0=p0, method='trf', jac='3-point')
        y_pred_with_p0 = biexponential(x, y[0], *popt_with_p0)
        r2_with_p0 = r2(y, y_pred_with_p0)
        results['with_p0'] = (popt_with_p0, r2_with_p0, y_pred_with_p0)

        popt_without_p0, _ = curve_fit(
            ubnd_wrapper, x, y, bounds=bounds, method='trf', jac='3-point')
        y_pred_without_p0 = biexponential(x, y[0], *popt_without_p0)
        r2_without_p0 = r2(y, y_pred_without_p0)
        results['without_p0'] = (
            popt_without_p0, r2_without_p0, y_pred_without_p0)

        if results['with_p0'][1] >= results['without_p0'][1]:
            best_fit = results['with_p0']
        else:
            best_fit = results['without_p0']

        popt, r2_score, y_pred = best_fit
        '''
        # y0 = y[0]
        y_f = y[-1]
        # e_threshold = y0 - 0.63 * (y0 - y_f)
        # e_time = x[np.where(y < e_threshold)[0][0]]
        # tau1_guess = e_time
        # tau2_guess = 0.1 * tau1_guess

        p0 = [0.4, 1, 0.1, y_f]
        bounds = ([0, 0, 0, -np.inf], [1, np.inf, np.inf, np.inf])
        
        popt, _ = curve_fit(ubnd_wrapper, x, y,
                            bounds=bounds,
                            p0=p0,
                            method='trf',
                            jac='3-point'
                            )
        args = [y[0], *popt]
        '''

    # y_pred = biexponential(x, *args)
    # r2_score = r2(y, y_pred)
    return popt, r2_score, y_pred


def poroelastic_fit(x, y, bound=False):
    x -= x[0]
    if bound:
        def bnd_wrapper(t, c, tau1, tau2):
            return poroelastic(t, y[0], c, tau1, tau2, y[-1])

        bounds = ([0, 0, 0], [1, np.inf, np.inf])
        p0 = [0.4, 1, 0.1]
        popt, _ = curve_fit(bnd_wrapper, x, y,
                            bounds=bounds, p0=p0, jac='3-point')
        args = [y[0], *popt, y[-1]]
    else:
        def ubnd_wrapper(t, c, tau1, tau2, y_f):
            return poroelastic(t, y[0], c, tau1, tau2, y_f)

        p0 = [0.4, 1, 0.1, 0]
        bounds = ([0, 0, 0, -np.inf], [1, np.inf, np.inf, np.inf])
        popt, _ = curve_fit(ubnd_wrapper, x, y,
                            bounds=bounds, p0=p0, jac='3-point')
        args = [y[0], *popt]

    y_pred = poroelastic(x, *args)
    r2_score = r2(y, y_pred)
    return popt, r2_score, y_pred


if __name__ == '__main__':
    pass
