import numpy as np
import numpy.typing as npt
from typing import Callable
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score as r2
import functools


# Functions
def lin(x: npt.NDArray, a: float, b: float) -> npt.NDArray:
    return x * a + b


def powerlaw(x: npt.NDArray, a: float, b: float) -> npt.NDArray:
    return a * (x ** b)


def poly(x: npt.NDArray, a: float, b: float, c: float) -> npt.NDArray:
    return a * (x ** b) + c * x


def hertzian(ind: npt.NDArray, diameter: float, e_star: float) -> npt.NDArray:
    return (4 / 3) * e_star * ((0.5 * diameter) ** 0.5) * (ind ** 1.5)


def exponential(t: npt.NDArray, tau: float, y_f: float, y_0: float) -> npt.NDArray:
    return (y_0 - y_f) * np.exp(-(t - t[0]) * tau) + y_f


def biexponential(t: npt.NDArray, c: float, tau1: float, tau2: float, y_f: float, y_0: float) -> npt.NDArray:
    return (y_0 - y_f) * (c * np.exp(-(t - t[0]) * tau1) + (1 - c) * np.exp(-(t - t[0]) * tau2)) + y_f


def poroelastic(t: npt.NDArray, c: float, tau1: float, tau2: float, y_f: float, y_0: float) -> npt.NDArray:
    return (y_0 - y_f) * (c * np.exp(-np.sqrt(t - t[0]) * tau1) + (1 - c) * np.exp(-(t - t[0]) * tau2)) + y_f


def fit(x: npt.NDArray, y: npt.NDArray, fun: Callable, y_max: float = 1.0, y_min: float = 0,
        bounds: tuple[list[int], list[float]] | None = None, p0: list[float] | None = None, jac: str | None = None,
        popt: npt.NDArray[float] | None = None, norm: bool = False, bound: bool | None = None):
    # Wrapping the function if bound or unbound
    if bound:
        func = functools.partial(fun, y_0=y[0], y_f=y[-1])
    elif bound is False:  # Need this to skip bound=None case
        func = functools.partial(fun, y_0=y[0])
    else:
        func = fun

    # Normalizing the input between 0 and 1
    if norm:
        x_min = min(x)
        y_min = min(y)
        x = (x - x_min)
        x_max = max(x)
        x = x / x_max
        y = (y - y_min)
        y_max = max(y)
        y = y / y_max

    # Calculating fit parameters if not provided
    if popt is None:
        if bounds:
            if p0 and jac:
                popt, _ = curve_fit(func, x, y, bounds=bounds, p0=p0, jac=jac)
            else:
                popt, _ = curve_fit(func, x, y, bounds=bounds)
        else:
            popt, _ = curve_fit(func, x, y)

    # Evaluating fit
    y_pred = func(x, *popt)
    r2_score = r2(y, y_pred)
    return popt, r2_score, (y_pred * y_max + y_min)


def lin_fit(x: npt.NDArray, y: npt.NDArray):
    def wrapper(t, *popt):
        return np.poly1d([*popt])(t)

    return fit(x, y, wrapper, popt=np.polyfit(x, y, 1))


def powerlaw_fit(x: npt.NDArray, y: npt.NDArray):
    return fit(x, y, powerlaw, bounds=([0, 0], [np.inf, np.inf]), norm=True)


def poly_fit(x: npt.NDArray, y: npt.NDArray):
    return fit(x, y, poly, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), p0=[1, 1, 1], jac='3-point', norm=True)


def hertzian_fit(x: npt.NDArray, y: npt.NDArray, diameter: float):
    def wrapper(ind, e_star):
        return hertzian(ind, diameter, e_star)

    return fit(x, y, wrapper)


def exponential_fit(x: npt.NDArray, y: npt.NDArray, bound: bool = False):
    return fit(x, y, exponential, bounds=([0], [np.inf]) if bound else ([0, -np.inf], [np.inf, np.inf]), bound=bound)


def biexponential_fit(x: npt.NDArray, y: npt.NDArray, bound: bool = False):
    return fit(x, y, biexponential, bounds=([0, 0, 0], [1, np.inf, np.inf]) if bound else ([0, 0, 0, -np.inf],
                                                                                           [1, np.inf, np.inf, np.inf]),
               p0=[0.4, 1, 0.1] if bound else [0.4, 1, 0.1, 0], jac='3-point', bound=bound)


def poroelastic_fit(x: npt.NDArray, y: npt.NDArray, bound: bool = False):
    return fit(x, y, poroelastic, bounds=([0, 0, 0], [1, np.inf, np.inf]) if bound else ([0, 0, 0, -np.inf],
                                                                                         [1, np.inf, np.inf, np.inf]),
               p0=[0.4, 1, 0.1] if bound else [0.4, 1, 0.1, 0], jac='3-point', bound=bound)


if __name__ == '__main__':
    pass
