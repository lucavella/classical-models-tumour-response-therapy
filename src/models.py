import numpy as np
import math
import scipy.integrate as scint



# y0 = V(t=0) (or y0 = [K(t=0), V(t=0)] for dynamic CC)
def solve_ivp(de, t, *y0):
    t_range = (np.min(t), np.max(t))
    pred = scint.solve_ivp(
        de,
        t_range,
        y0,
        t_eval=t,
        rtol=1e-5,
        atol=1e-8,
        method='LSODA'
    ).y

    if np.any(pred):
        return pred[0]
    else:
        return [math.nan] * len(t)

def solve_odeint(de, t, *y0):
    return scint.odeint(
        de, 
        y0, 
        t, 
        tfirst=True
    )[:, 0]

solution = solve_odeint


# Introduction to mathematical oncology (table 2.1)
class Exponential:
    def predict(t, V0, a, b):
        t = np.array(t)
        return solution(
            lambda t, V: \
                (a - b) * V,
            t,
            V0
        )

    params = 3
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # a
        (0, np.inf), # b
    ]

class Logistic:
    def predict(t, V0, g, K):
        t = np.array(t)
        return solution(
            lambda t, V: \
                g * V * (1 - V / K),
            t,
            V0
        )

    params = 3
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # g
        (0, np.inf), # K
    ]

class Gompertz:
    def predict(t, V0, a, b):
        t = np.array(t)
        return solution(
            lambda t, V: \
                V * (b - a * np.log(V)),
            t,
            V0
        )

    params = 3
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # a
        (-np.inf, np.inf), # b
    ]

class GeneralGompertz:
    def predict(t, V0, a, b, l):
        t = np.array(t)
        return solution(
            lambda t, V: \
                V ** l * (b - a * np.log(V)),
            t,
            V0
        )

    params = 4
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # a
        (-np.inf, np.inf), # b
        (2 / 3, 1),  # l
    ]

class ClassicBertalanffy:
    def predict(t, V0, a, b):
        t = np.array(t)
        return solution(
            lambda t, V: \
                a * V ** (2 / 3) - b * V,
            t,
            V0
        )

    params = 3
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # a
        (0, np.inf), # b
    ]

class GeneralBertalanffy:
    def predict(t, V0, a, b, l):
        t = np.array(t)
        return solution(
            lambda t, V: \
                a * V ** l - b * V,
            t,
            V0
        )

    params = 4
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # a
        (0, np.inf), # b
        (2 / 3, 1),  # l
    ]


# Extra models
# https://doi.org/10.1371/journal.pcbi.1003800 (p. 3)

class ExponentialLinear:
    def predict(t, V0, a, b):
        t = np.array(t)
        u = 1 / a * np.log(b / (a * V0))
        return solution(
            lambda t, V: \
                a * V if t <= u else b,
            t,
            V0
        )

    params = 3
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # a
        (0, np.inf), # b
    ]

class GeneralLogistic:
    def predict(t, V0, a, v, K):
        t = np.array(t)
        return solution(
            lambda t, V: \
                a * V * (1 - (V / K) ** v),
            t,
            V0
        )

    params = 4
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # a
        (2 / 3, 1),  # v
        (0, np.inf), # K
    ]

class DynamicCarryingCapacity:
    def predict(t, V0, K0, a, b):
        t = np.array(t)        
        def dcc_system(t, y):
            # y is a [K, V] vector
            K, V = y
            dK_dt = b * V ** (2 / 3)
            dV_dt = a * V * np.log(K / V)

            return [dK_dt, dV_dt]

        return solution(
            dcc_system,
            t,
            V0,
            K0
        )

    params = 4
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # K0
        (0, np.inf), # a
        (0, np.inf), # b
    ]