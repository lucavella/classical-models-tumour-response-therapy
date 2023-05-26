import numpy as np
import scipy.integrate as scint



# y0 = V(t=0) (or y0 = [K(t=0), V(t=0)] for dynamic CC)
def solutionDE(de):
    return lambda t, y0: \
        scint.solve_ivp(de, (t[0], t[-1]), [y0], dense_output=True, vectorized=True).sol(t)


# Introduction to mathematical oncology (table 2.1)
class Exponential:
    def predict(t, V0, a, b):
        t = np.array(t)
        # return solutionDE(
        #     lambda t, V: \
        #         (a - b) * V
        # )(t, V0)[0]
        return V0 * np.exp((a - b) * t)

    params = 3
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # a
        (0, np.inf), # b
    ]

class LogisticVerhulst:
    def predict(t, V0, g, K):
        t = np.array(t)
        # return solutionDE(
        #     lambda t, V: \
        #         g * V * (1 - V / K)
        # )(t, V0)[0]
        return (V0 * K * np.exp(g * t)) / (K - V0 * (1 - np.exp(g * t)))

    params = 3
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # a
        (0, np.inf), # b
    ]

class Gompertz:
    def predict(t, V0, a, b):
        t = np.array(t)
        # return solutionDE(
        #     lambda t, V: \
        #         V * (b - a * np.log(V))
        # )(t, V0)[0]
        return np.exp(b / a + (np.log(V0) - b / a) * np.exp(-a * t))

    params = 3
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # a
        (0, np.inf), # b
    ]

class GeneralGompertz:
    def predict(t, V0, a, b, l):
        t = np.array(t)
        return solutionDE(
            lambda t, V: \
                V ** l * (b - a * np.log(V))
        )(t, V0)[0]

    params = 4
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # a
        (0, np.inf), # b
        (2 / 3, 1),  # l
    ]

class ClassicBertalanffy:
    def predict(t, V0, a, b):
        t = np.array(t)
        # return solutionDE(
        #     lambda t, V: \
        #         a * V ** (2 / 3) - b * V
        # )(t, V0)[0]
        return (a / b + (V0 ** (1 / 3) - a / b) * np.exp(-1 / 3 * b * t)) ** 3

    params = 3
    bounds = [
        (0, np.inf), # V0
        (0, np.inf), # a
        (0, np.inf), # b
    ]

class GeneralBertalanffy:
    def predict(t, V0, a, b, l):
        t = np.array(t)
        # return solutionDE(
        #     lambda t, V: \
        #         a * V ** l - b * V
        # )(t, V0)[0]
        return (a / b + (V0 ** (1 - l) - a / b) * np.exp(-b * (1 - l) * t)) ** (1 / (1 - l))

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
    def predict(t, V0, a, b, u):
        t = np.array(t)
        return solutionDE(
            lambda t, V: \
                a * V if t <= u else b
        )(t, V0)[0]

    params = 4

class GeneralLogistic:
    def predict(t, V0, a, b, K):
        t = np.array(t)
        return solutionDE(
            lambda t, V: \
                a * V * (1 - (V / K) ** b)
        )(t, V0)[0]

    params = 4

class DynamicCarryingCapacity:
    def predict(t, V0, a, b):
        t = np.array(t)

        def dKdt(V):
            return b * V ** (2 / 3)
        def dVdt(K, V):
            return a * V * np.log(K / V)

        return solutionDE(
            # y is a [K, V] vector
            lambda t, y : \
                [dKdt(y[1]), dVdt(y[0], y[1])]
        )(t, V0)[1]

    params = 3