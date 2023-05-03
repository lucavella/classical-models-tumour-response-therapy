import numpy as np
import scipy.integrate as scint



# y0 = V(t=0) (or y0 = [K(t=0), V(t=0)] for dynamic CC)
def solutionDE(de):
    return lambda t, y0: \
        scint.odeint(de, y0, t)


# Introduction to mathematical oncology (table 2.1)
class Exponential:
    def predict(t, V0, a, b):
        return solutionDE(
            lambda V, t: \
                (a - b) * V
        )(t, V0)

class LogisticVerhulst:
    def predict(t, V0, a, b):
        return solutionDE(
            lambda V, t: \
                a * V - b * V ** 2
        )(t, V0)

class Gompertz:
    def predict(t, V0, a, b):
        return solutionDE(
            lambda V, t: \
                V * (b - a * np.log(V))
        )(t, V0)

class GeneralGompertz:
    def predict(t, V0, a, b, l):
        return solutionDE(
            lambda V, t: \
                V ** l * (b - a * np.log(V))
        )(t, V0)

class ClassicBertalanffy:
    def predict(t, V0, a, b):
        return solutionDE(
            lambda V, t: \
                a * V ** (2 / 3) - b * V
        )(t, V0)

class GeneralBertalanffy:
    def predict(t, V0, a, b, l):
    return solutionDE(
        lambda V, t: \
            a * V ** l - b * V
    )(t, V0)


# Extra models
# https://doi.org/10.1371/journal.pcbi.1003800 (p. 3)

class ExponentialLinear:
    def predict(t, V0, a, b, u):
        return solutionDE(
            lambda V, t: \
                a * V if t <= u else b
        )(t, V0)

class GeneralLogistic:
    def predict(t, V0, a, v, K):
        return solutionDE(
            lambda V, t: \
                gamm * V * (1 - (V / K) ** v)
        )(t, V0)

class dynamicCarryingCapacity:
    def predict(t, V0, a, b):
        def dKdt(V):
            return b * V ** (2 / 3)
        def dVdt(K, V):
            return a * V * np.log(K / V)

        return solutionDE(
            # y is a [K, V] vector
            lambda y, t : \
                [dKdt(y[1]), dVdt(y[0], y[1])]
        )(t, V0)