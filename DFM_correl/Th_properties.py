import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt

P2Pcorel = 'MNmodel'
frfaccorel = 'base'

#QUALITY
def getQuality(H, P): #H in J/kg, P in Pa
    hl, hg = getPhasesEnthalpy(P)
    if H < hl:
        return 0
    elif H > hg:
        return 1
    elif H*0.001 <= hg and H >= hl:
        return (H - hl)/(hg - hl)

def getPhasesEnthalpy(P): #P in Pa
    water = IAPWS97(P = P * 10**(-6), x=0)
    steam = IAPWS97(P = P * 10**(-6), x=1)
    return water.h * 1000, steam.h * 1000 #Pa

#GEOMETRIC PARAMETERS
def getPhi2Phi(epsilon, rho, rho_l, rho_g, x_th, P):
    if P2Pcorel == 'base': #Validated
        phi2phi = 1 + 3*epsilon
    elif P2Pcorel == 'HEM1': #Validated
        phi2phi = (rho/rho_l)*((rho_l/rho_g)*x_th + +1)
    elif P2Pcorel == 'HEM2': #Validated
        m = IAPWS97(P = P*(10**(-6)), x = 0).mu / IAPWS97(P = P*(10**(-6)), x = 1).mu
        phi2phi = (rho/rho_l)*((m-1)*x_th + 1)*((rho_l/rho_g)*x_th + +1)**(0.25)
    elif P2Pcorel == 'MNmodel': #Validated
        phi2phi = (1.2 * (rho_l/rho_g -1)*x_th**(0.824) + 1)*(rho/rho_l)
    return phi2phi

def getFrictionFactor(rho, U, D_h, P):
    if frfaccorel == 'base': #Validated
        return 0.0001
    Re = getReynoldsNumber(rho, U, D_h, P)
    print(f'Re: {Re}')
    if frfaccorel == 'blasius': #Not Validated
        print(f'Friction factor: {0.186 * Re**(-0.2)}')
        return 0.186 * Re**(-0.2)
    if frfaccorel == 'Churchill': #Not Validated
        Ra = 0.4 * (10**(-6)) #Roughness
        R = Ra / D_h
        frict=8*(((8.0/Re)**12)+((2.475*np.log(((7/Re)**0.9)+0.27*R))**16+(37530/Re)**16)**(-1.5))**(1/12)   
        return frict

def getAreas(A, Phi2Phi, D_h, K_loss, DV, Dz, f):
    #print(f'Inside getAreas: A: {A}, Phi2Phi: {Phi2Phi}, D_h: {D_h}, K_loss: {K_loss}, DV: {DV}, Dz: {Dz}, f: {f}')
    A_chap = A +  (Phi2Phi/2) * ((f / D_h) + (K_loss / Dz)) * DV
    #print(f'Second membre: {(Phi2Phi/2) * ((f / D_h) + (K_loss / Dz)) * DV}')
    return A_chap


#VOID FRACTION
def getVoidFraction(x_th, rho_l, rho_g): #x_th in m3/m3, rho_l in kg/m3, rho_g in kg/m3
    if x_th == 0:
        return 0.0001
    elif x_th == 1:
        return 0.99
    else:
        return (x_th * rho_l)/(x_th * rho_l + (1 - x_th) * rho_g)
    
def getVgj(rho_g, rho_l, g, D_h):
    if rho_g == 0:
        return 0
    if rho_l == 0:
        return 0
    return 0.188 * np.sqrt(((rho_l - rho_g) * g * D_h ) / rho_g )

def getC0(rho_g, rho_l):
    if rho_g == 0:
        return 0
    if rho_l == 0:
        return 0
    return 1.2 - 0.2*np.sqrt(rho_g / rho_l)

def getVgj_prime(rho_g, rho_l, g, D_h, U):
    C0 = getC0(rho_g, rho_l)
    Vgj = getVgj(rho_g, rho_l, g, D_h)
    Vgj_prime = Vgj + (C0 - 1) * U
    return Vgj_prime

#THEMODYNAMIC PROPERTIES
def getDensity(P, x): #P in Pa, x in m3/m3
    vapor = IAPWS97(P = P*(10**(-6)), x = 1)
    liquid = IAPWS97(P = P*(10**(-6)), x = 0)
    rho_g = vapor.rho
    rho_l = liquid.rho

    rho = rho_l * (1 - x) + rho_g * x
    return rho_l, rho_g, rho


def getHfg(P):#P in Pa
    h_fg = IAPWS97(P = P*(10**(-6)), x = 1).h - IAPWS97(P = P*(10**(-6)), x = 0).h
    return h_fg*1000

def getReynoldsNumber(rho, U, D_h, P):
    m = IAPWS97(P = P*(10**(-6)), x = 0).mu
    return rho * abs(U) * D_h / m


#SOLVER
def splitVar(VAR):
    nCells = len(VAR)//3
    U = VAR[0:nCells]
    P = VAR[nCells:2*nCells]
    H = VAR[2*nCells:]
    return U, P, H

def mergeVar(U, P, H):
    return np.concatenate((U, P, H))