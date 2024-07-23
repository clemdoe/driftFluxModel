import numpy as np
from iapws import IAPWS97

##Creation of function
#FUnction to calculate the temperature of the mixture
def getTemperature(P, H, x):
    T = IAPWS97(P = P, h = H, x = x).T
    return T

#Function to calculate the density of the mixture
def getDensity(epsilon, P, T):
    """ state = IAPWS97(P = P, x = epsilon, T = T)
    rho_g = state.Vapor.rho
    rho_l = state.Liquid.rho
    if rho_g == None:
        rho_g = 0 """
    
    #rho_g = 12 + (18/325) * (T - 273.15)
    #rho_l = 715 - (0.2 * (T - 273.15))

    vapor = IAPWS97(P = P, x = 1)
    liquid = IAPWS97(P = P, x = 0)
    rho_g = 1/vapor.v
    rho_l = 1/liquid.v

    #print(f'In getDensity: rho_g: {rho_g}, rho_l: {rho_l}, epsilon: {epsilon}, P: {P}, T: {T}')
    rho = rho_l * (1 - epsilon) + rho_g * epsilon
    return rho_g, rho_l, rho

#Function to calculate the areas of the mixture
def getAreas(A, Phi2Phi, D_h, K_loss, DV, Dz, f):
    #print(f'Inside getAreas: A: {A}, Phi2Phi: {Phi2Phi}, D_h: {D_h}, K_loss: {K_loss}, DV: {DV}, Dz: {Dz}, f: {f}')
    A_chap = A +  (Phi2Phi/2) * ((f / D_h) + (K_loss / Dz)) * DV
    #print(f'Second membre: {(Phi2Phi/2) * ((f / D_h) + (K_loss / Dz)) * DV}')
    return A_chap


def getVelocities(U, epsilon, rho_l, rho, V_gj_prime):
    u_l = U + (rho_l/rho) * (epsilon/ (1 - epsilon)) * V_gj_prime
    u_g = U + (rho_l/rho) * V_gj_prime
    return u_l, u_g

def getThermodynamicQuality(P, H, U, epsilon, rho_l, rho_g, rho, V_gj_prime, A):
    """ u_l, u_g = getVelocities(U, epsilon, rho_l, rho, V_gj_prime)
    m_g_ = rho_g * A * u_g * epsilon
    m_l_ = rho_l * A * u_l * (1 - epsilon)
    x_th = m_g_ / (m_g_ + m_l_)
    if rho_g == 0:
        return 0
    if rho_l == 0:
        return 1 """
    hlsat  = getHlsat(P)
    hgsat = getHgsat(P)
    #print(f'P: {P}, H: {H}, hlsat: {hlsat}, hgsat: {hgsat}')
    if H < hlsat:
        return 0.0001
    elif H > hgsat:
        return 0.9999
    else:
        x_th = (H - hlsat) / (hgsat - hlsat)
        
    #x_th = (epsilon * rho_g) / (epsilon * rho_g + (1 - epsilon) * rho_l)

    return x_th

#Function to calculate the drift velocity of the mixture
def getDriftVelocity(rho_g, rho_l, g, D_h):
    if rho_g == 0:
        return 0
    if rho_l == 0:
        return 0
    return 0.188 * np.sqrt(((rho_l - rho_g) * g * D_h ) / rho_g )

#Function to calculate the constant C0 called the distribution parameter
def getC0(rho_g, rho_l):
    if rho_g == 0:
        return 0
    if rho_l == 0:
        return 0
    return 1.2 - 0.2*np.sqrt(rho_g / rho_l)

#Function to calculate the void fraction of the mixture
def getVoidFraction(x_th, C0, Vgj_prime, rho, U, rho_g, rho_l):
    """ if x_th == 0:
        return 0
    epsilon = x_th / ((C0 * (x_th + (rho_g/rho_l) * (1 - x_th))) + (rho_g * Vgj_prime / rho * U)) """
    if x_th == 0:
        return 0
    if x_th == 1:
        return 0.999
    epsilon = (x_th * rho_l) / ( x_th * rho_l + (1 - x_th) * rho_g)
    #print(f'Inside getVoidFraction: rho_g: {rho_g}, rho_l: {rho_l}, U: {U}, H: {H}, P: {P}, U_old: {U_old}, rho: {rho}, D_h: {D_h}, g: {g}, C0: {C0}, x_th: {x_th}, Vgj: {Vgj}, Vgj_prime: {Vgj_prime}, epsilon: {epsilon}')
    return epsilon

def getHfg(P):
    h_fg = IAPWS97(P = P, x = 1).h - IAPWS97(P = P, x = 0).h
    return h_fg/1000

def getMuLiquid(P, epsilon):
    return 0.0001
    #return IAPWS97(P = P, x = epsilon).Liquid.mu

def getMuGas(P, epsilon):
    return 0.0001
    #return IAPWS97(P = P, x = epsilon).Vapor.mu

def getReynoldsNumber(rho, U, D_h, mu):
    return rho * U * D_h / mu

def getPhi2Phi(epsilon):
    return 1+3*epsilon

def getFrictionFactor(Re, rugo, D_h):
    return 0.0001 #0.3164 * Re**(-0.25) + 0.0018 * (rugo / D_h)**1.155

def get_parameters(P,H,rho_l,rho_g,rho,epsilon, D_h, g, U, areaMatrix):
    rho_old = rho
    rho_l_old = rho_l
    rho_g_old = rho_g

    Vgj = getDriftVelocity(rho_g, rho_l, g, D_h)
    C0 = getC0(rho_g, rho_l)
    Vgj_prime = Vgj + (C0 - 1) * U
    
    x_th_new = getThermodynamicQuality(P, H, U, epsilon, rho_l, rho_g, rho, Vgj_prime, areaMatrix) #depend de H uniquement
    rho_g_old, rho_l_old, rho_old = getDensity(epsilon, P, 1) #depend de P pour rhol rhog, rho repend de epsilon
    epsilon_new = getVoidFraction(x_th_new, C0, Vgj_prime, rho_old, U, rho_g_old, rho_l_old) #depend de rhol rhog uniquement, eux même dependent de seulement P
    rho_g_new, rho_l_new, rho_new = getDensity(epsilon_new, P, 1) #depend de P pour rhol rhog, rho repend de epsilon

    Vgj = getDriftVelocity(rho_g_new, rho_l_new, g, D_h)
    C0 = getC0(rho_g_new, rho_l_new)
    Vgj_prime = Vgj + (C0 - 1) * U
    h_fg = getHfg(P)
    #print(f'rho_g_new: {rho_g_new}, rho_l_new: {rho_l_new}, rho_new: {rho_new}, epsilon_new: {epsilon_new}, x_th_new: {x_th_new}, C0: {C0}, Vgj: {Vgj}, h_fg: {h_fg}')
    return rho_g_new, rho_l_new, rho_new, epsilon_new, x_th_new, C0, Vgj, h_fg

def getHlsat(P):
    return IAPWS97(P = P, x = 0).h

def getHgsat(P):
    return IAPWS97(P = P, x = 1).h

#Function to calculate the hydraulic diameter
def getD_h(L,l,geoType,cladRadius):
    if geoType == "square":
        return ((l*L)-(np.pi*cladRadius**2))/(cladRadius*2*np.pi)
    if geoType =="cylinder":
        print('Cylinder geometry not implemented yet')
        return
    
#Function to merge the 3 component of the matrix U,P,H
def createVar(U,P,H):
    var = np.zeros(3*len(U))
    for i in range(len(var)):
        if i < len(U):
            var[i] = U[i]
        elif i < 2*len(U):
            var[i] = P[i-len(U)]
        else:
            var[i] = H[i-2*len(U)]
    return var

#Function to split the 3 component of the matrix U,P,H
def splitVar(var):
    U = var[:len(var)//3]
    P = var[len(var)//3:2*len(var)//3]
    H = var[2*len(var)//3:]
    return list(U),list(P),list(H)

def sousRelaxation(Field, oldField, beta):
    n = len(Field)
    newField = np.zeros(n)
    for i in range(n):
        newField[i] = beta * Field[i] + (1 - beta) * oldField[i]
    return newField