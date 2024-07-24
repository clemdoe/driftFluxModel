import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt
from Th_properties import *

def setInitialFields(nCells, u_inlet, P_outlet, h_inlet, epsilonTarget, A, D_h, K_loss, DV, Dz, epsInnerIteration, maxInnerIteration): #u_inlet in m/s, P_outlet in Pa, h_inlet in J/kg, epsilonTarget in m3/m3, A in m2, D_h in m, K_loss in m, DV in m, Dz in m
    U = np.ones(nCells)*u_inlet
    P = np.ones(nCells)*P_outlet
    H = np.ones(nCells)*h_inlet
    epsilon = np.array([i*epsilonTarget/nCells for i in range(nCells)])
    print(f'epsilon inside setInitialFields: {epsilon}')

    rho = np.ones(nCells)
    rho_g = np.ones(nCells)
    rho_l = np.ones(nCells)
    areaMatrix = np.ones(nCells)
    areaMatrix_1 = np.ones(nCells)
    areaMatrix_2 = np.ones(nCells)
    Dhfg = np.ones(nCells)
    x_th = np.ones(nCells)
    f = np.ones(nCells)
    Vgj = np.ones(nCells)
    Vgj_prime = np.ones(nCells)
    C0 = np.ones(nCells)

    for i in range(nCells):
        rho_l[i], rho_g[i], rho[i] = getDensity(P[i], epsilon[i]) 
        Dhfg[i] = getHfg(P[i])
        f[i] = getFrictionFactor(rho[i], U[i], D_h, P[i]) 
        areaMatrix_1[i] = getAreas(A, getPhi2Phi(epsilon[i], rho[i], rho_l[i], rho_g[i], x_th[i], P[i]), D_h, K_loss, DV, Dz, f[i])
        areaMatrix_2[i] = getAreas(A, -getPhi2Phi(epsilon[i], rho[i], rho_l[i], rho_g[i], x_th[i], P[i]), D_h, K_loss, DV, Dz, f[i])
        areaMatrix[i] = A
        Vgj[i] = getVgj(rho_g[i], rho_l[i], 9.81, D_h)
        C0[i] = getC0(rho_g[i], rho_l[i])
        Vgj_prime[i] = getVgj_prime(rho_g[i], rho_l[i], 9.81, D_h, U[i])
 
    return U, P, H, epsilon, rho, rho_g, rho_l, areaMatrix, areaMatrix_1, areaMatrix_2, Dhfg, x_th, f, Vgj, Vgj_prime, C0

def updateFields(U, P, H, epsilon, rho, rho_g, rho_l, areaMatrix, areaMatrix_1, areaMatrix_2, Dhfg, x_th, f, nCells, u_inlet, P_outlet, h_inlet, A, D_h, K_loss, DV, Dz, Vgj, Vgj_prime, C0, epsInnerIteration, maxInnerIteration):
    epsilonUpdated = np.ones(nCells)
    rhoUpdated = np.ones(nCells)
    rho_gUpdated = np.ones(nCells)
    rho_lUpdated = np.ones(nCells)
    areaMatrixUpdated = np.ones(nCells)
    areaMatrix_1Updated = np.ones(nCells)
    areaMatrix_2Updated = np.ones(nCells)
    DhfgUpdated = np.ones(nCells)
    x_thUpdated = np.ones(nCells)
    fUpdated = np.ones(nCells)
    VgjUpdated = np.ones(nCells)
    Vgj_primeUpdated = np.ones(nCells)
    C0Updated = np.ones(nCells)

    for i in range(nCells):
        rho_lUpdated[i], rho_gUpdated[i], rhoUpdated[i], epsilonUpdated[i], x_thUpdated[i] = resolveParameters(U[i], P[i], H[i], epsilon[i], x_th[i], rho_l[i], rho_g[i], rho[i], areaMatrix[i], epsInnerIteration, maxInnerIteration)
        DhfgUpdated[i] = getHfg(P[i])
        fUpdated[i] = getFrictionFactor(rho[i], U[i], D_h, P[i])
        areaMatrix_1Updated[i] = getAreas(A, getPhi2Phi(epsilonUpdated[i], rhoUpdated[i], rho_lUpdated[i], rho_gUpdated[i], x_thUpdated[i], P[i]), D_h, K_loss, DV, Dz, fUpdated[i])
        areaMatrix_2Updated[i] = getAreas(A, -getPhi2Phi(epsilonUpdated[i], rhoUpdated[i], rho_lUpdated[i], rho_gUpdated[i], x_thUpdated[i], P[i]), D_h, K_loss, DV, Dz, fUpdated[i])
        areaMatrixUpdated[i] = A
        VgjUpdated[i] = getVgj(rho_gUpdated[i], rho_lUpdated[i], 9.81, D_h)
        C0Updated[i] = getC0(rho_gUpdated[i], rho_lUpdated[i])
        Vgj_primeUpdated[i] = getVgj_prime(rho_gUpdated[i], rho_lUpdated[i], 9.81, D_h, U[i])

    return epsilonUpdated, rhoUpdated, rho_gUpdated, rho_lUpdated, VgjUpdated, Vgj_primeUpdated, areaMatrixUpdated, areaMatrix_1Updated, areaMatrix_2Updated, DhfgUpdated, C0Updated, x_thUpdated, fUpdated

def resolveParameters(U, P ,h, epsilon, x_th, rho_l, rho_g, rho, areaMatrix, epsInnerIteration, maxInnerIteration): #U in m/s, P in Pa, h in J/kg, epsilon in m3/m3, rho_l in kg/m3, rho_g in kg/m3, rho in kg/m3, areaMatrix in m2
    print(f'P: {P}')
    x_th_new = getQuality(h, P)
    rho_l_new, rho_g_new, rho_new = getDensity(P, epsilon)
    epsilon_new = getVoidFraction(x_th_new, rho_l_new, rho_g_new)
    rho_l_new, rho_g_new, rho_new = getDensity(P, epsilon_new)
    print(f'Inside resolveParameters: rho_l_new: {rho_l_new}, rho_g_new: {rho_g_new}, rho_new: {rho_new}, epsilon_new: {epsilon_new}, x_th_new: {x_th_new}')
    return rho_l_new, rho_g_new, rho_new, epsilon_new, x_th_new


