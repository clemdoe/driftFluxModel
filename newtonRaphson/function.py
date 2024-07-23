import numpy as np
from iapws import IAPWS97

def frict(u, P, h, parameters, i, D):
    density = parameters[:,0][i]
    mu = getmu(P, h)
    Re = getReynolds(density, u, D, mu)
    A=0.186
    B=-0.2
    C=0.0
    return A*Re**B+C

def tpmultiplier(parameters,i):
    epsilon = parameters[:,3][i]
    return 1 + 3*epsilon

def Quality(parameters,i):
    epsilon = parameters[:,3][i]
    rho_g = parameters[:,2][i]
    rho_l = parameters[:,1][i]
    x_th = (epsilon * rho_g) / (epsilon * rho_g + (1 - epsilon) * rho_l)
    return x_th

def VoidFraction(parameters,i):
    x_th = parameters[:,6][i]
    rho_g = parameters[:,2][i]
    rho_l = parameters[:,1][i]
    return (x_th * rho_l) / (x_th * rho_l + (1 - x_th) * rho_g)

getReynolds = lambda density, u, D, mu: density*u*D/mu
getmu = lambda P, h: IAPWS97(P=P*(10**(-6)), h=h*(10**(-3))).Liquid.mu
getTemperature = lambda P, h: IAPWS97(P=P*(10**(-6)), h=h*(10**(-3))).T
getV_gj = lambda rho_l, rho_g, D_h, g: 0.188 * np.sqrt(((rho_l - rho_g) * g * D_h ) / rho_g )
getC0 = lambda rho_g, rho_l: 1.2 - 0.2*np.sqrt(rho_g / rho_l)
getHfg = lambda T: IAPWS97(T = T, x = 1).h - IAPWS97(T = T, x = 0).h


def getDensity(P, h, epsilon):
    T = getTemperature(P, h)
    Psat = IAPWS97(T = T, x = 1).P
    rho_g = IAPWS97(P = Psat, T = T).Vapor.rho
    if rho_g == None:
        rho_g = 1
    #rho_g = 12 + (18/325) * (T - 273.15)
    rho_l = 715 - (0.2 * (T - 273.15))

    #print(f'In getDensity: rho_g: {rho_g}, rho_l: {rho_l}, epsilon: {epsilon}, P: {P}, T: {T}')
    rho = rho_l * (1 - epsilon) + rho_g * epsilon
    return rho_g, rho_l, rho
