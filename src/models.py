import numpy as np
import scipy.integrate as scint

# y0 = V0 (or y0 = [K0, V0] for dynamic CC)
def solutionDE(de):
    return lambda y0, t: \
        scint.odeint(de, y0, t)


def exponentialDE(alph, beta):
    return solutionDE(
        lambda V, t: \
            (alph - beta) * V
    )

def logisticDE(gamm, K):
    return solutionDE(
        lambda V, t: \
            gamm * V * (1 - V / K)
    )

def gompertzDE(gamm, delt):
    return solutionDE(
        lambda V, t: \
            V * (delt - gamm * np.log(V))
    )

def generalGompertzDE(gamm, delt, lamb):
    return solutionDE(
        lambda V, t: \
            V ** lamb * (delt - gamm * np.log(V))
    )

def classicBertalanffyDE(alph, beta):
    return solutionDE(
        lambda V, t: \
            alph * V ** (2 / 3) - beta * V
    )

def generalBertalanffyDE(alph, beta, lamb):
    return solutionDE(
        lambda V, t: \
            alph * V ** lamb - beta * V
    )


# Extra models

# https://doi.org/10.1371/journal.pcbi.1003800 (p. 3)
def exponentialLinearDE(a, b, tau):
    return solutionDE(
        lambda V, t: \
            a * V if t <= tau else b
    )

def generalLogisticDE(a, v, K):
    return solutionDE(
        lambda V, t: \
            gamm * V * (1 - (V / K) ** v)
    )

def dynamicCarryingCapacityDE(a, b):
    def dKdt(V):
        return b * V ** (2 / 3)
    def dVdt(K, V):
        return a * V * np.log(K / V)

    return solutionDE(
        # y is a [K, V] vector
        lambda y, t : \
            [dKdt(y[1]), dVdt(y[0], y[1])]
    )

