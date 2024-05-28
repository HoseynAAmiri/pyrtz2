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
    return (y_0 - y_f) * np.exp(-(t-t[0]) * tau) + y_f


def biexponential(t, y_0, c, tau1, tau2, y_f):
    return (y_0 - y_f) * (c * np.exp(-(t-t[0]) * tau1) + (1 - c) * np.exp(-(t-t[0]) * tau2)) + y_f


def poroelastic(t, y_0, c, tau1, tau2, y_f):
    return (y_0 - y_f) * (c * np.exp(-np.sqrt(t-t[0]) * tau1) + (1 - c) * np.exp(-(t-t[0]) * tau2)) + y_f


# Fitting
def eval_fit(x, y, fun, y_max=1, y_min=0, bounds=None, p0=None, jac=None, popt=None, norm=False):
    if norm:
        x_min = min(x)
        y_min = min(y)
        x = (x - x_min)
        x_max = max(x)
        x = x / x_max
        y = (y - y_min)
        y_max = max(y)
        y = y / y_max
    
    if popt is None:
        if bounds:
            if p0 and jac:
                popt, _ = curve_fit(fun, x, y, bounds=bounds, p0=p0, jac=jac)
            else:
                popt, _ = curve_fit(fun, x, y, bounds=bounds)
        else:
            popt, _ = curve_fit(fun, x, y)
    
    y_pred = fun(x, *popt)
    r2_score = r2(y, y_pred)
    return popt, r2_score, y_pred * y_max + y_min

def eval_fit_bounds(x, y, fun, y_max=1, y_min=0, bounds=None, p0=None, jac=None, popt=None, norm=False, bound=None):
    
    if bound:
        def bnd_wrapper(t, *params):
            return fun(t, y[0], *params, y[-1])
        fun = wrapper
    elif bound is False: # Need this to skip bound=None case 
        def ubnd_wrapper(t, *params, y_f):
            return exponential(t, y[0], *params, y_f)
        fun = wrapper

    if norm:
        x_min = min(x)
        y_min = min(y)
        x = (x - x_min)
        x_max = max(x)
        x = x / x_max
        y = (y - y_min)
        y_max = max(y)
        y = y / y_max
    
    if popt is None:
        if bounds:
            if p0 and jac:
                popt, _ = curve_fit(fun, x, y, bounds=bounds, p0=p0, jac=jac)
            else:
                popt, _ = curve_fit(fun, x, y, bounds=bounds)
        else:
            popt, _ = curve_fit(fun, x, y)
    
    y_pred = fun(x, *popt)
    r2_score = r2(y, y_pred)
    return popt, r2_score, y_pred * y_max + y_min

def lin_fit(x, y):
    popt = np.polyfit(x, y, 1)
    return eval_fit(x, y, np.poly1d(popt), popt=[])

def powerlaw_fit(x, y):
    bounds = ([0, 0], [np.inf, np.inf])
    return eval_fit(x, y, powerlaw, bounds=bounds,norm=True)

def poly_fit(x, y):
    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    p0 = [1, 1, 1]
    return eval_fit(x, y, poly, bounds=bounds, p0=p0, jac='3-point', norm=True)

def hertzian_fit(x, y, diameter):
    def wrapper(ind, e_star):
        return hertzian(ind, diameter, e_star)
    
    return eval_fit(x, y, wrapper)

def exponential_fit(x, y, bound=False):
    if bound:
        def bnd_wrapper(t, tau):
            return exponential(t, y[0], tau, y[-1])

        bounds = ([0], [np.inf])
        return eval_fit(x, y, bnd_wrapper, bounds=bounds)
    else:
        def ubnd_wrapper(t, tau, y_f):
            return exponential(t, y[0], tau, y_f)

        bounds = ([0, -np.inf], [np.inf, np.inf])
        return eval_fit(x, y, ubnd_wrapper, bounds=bound)

def biexponential_fit(x, y, bound=False):
    if bound:
        def bnd_wrapper(t, c, tau1, tau2):
            return biexponential(t, y[0], c, tau1, tau2, y[-1])

        bounds = ([0, 0, 0], [1, np.inf, np.inf])
        p0 = [0.4, 1, 0.1]
        return eval_fit(x, y, bnd_wrapper, bounds=bounds, p0=p0, jac='3-point')       
    else:
        def ubnd_wrapper(t, c, tau1, tau2, y_f):
            return biexponential(t, y[0], c, tau1, tau2, y_f)

        p0 = [0.4, 1, 0.1, 0]
        bounds = ([0, 0, 0, -np.inf], [1, np.inf, np.inf, np.inf])
        return eval_fit(x, y, ubnd_wrapper, bounds=bounds, p0=p0, jac='3-point')

def poroelastic_fit(x, y, bound=False):
    if bound:
        def bnd_wrapper(t, c, tau1, tau2):
            return poroelastic(t, y[0], c, tau1, tau2, y[-1])

        bounds = ([0, 0, 0], [1, np.inf, np.inf])
        p0 = [0.4, 1, 0.1]
        return eval_fit(x, y, bnd_wrapper, bounds=bounds, p0=p0, jac='3-point')
    else:
        def ubnd_wrapper(t, c, tau1, tau2, y_f):
            return poroelastic(t, y[0], c, tau1, tau2, y_f)

        p0 = [0.4, 1, 0.1, 0]
        bounds = ([0, 0, 0, -np.inf], [1, np.inf, np.inf, np.inf])
        return eval_fit(x, y, ubnd_wrapper, bounds=bounds, p0=p0, jac='3-point')

if __name__ == '__main__':
    pass
