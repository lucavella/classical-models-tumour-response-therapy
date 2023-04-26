import numpy as np
import scipy.integrate as scint


def solutionDE(de):
    return lambda t, V0: \
        scint.odeint(de, V0, t)


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