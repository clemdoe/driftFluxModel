# This file contains the class FVM and the class plotting to solve a differential equation and plot the results
# Used to calculate the drift velocity of the mixture in the fuel assembly
# Authors : Clément Huet

import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt
from Th_properties import *
from Th_resolution import *

#class to solve a differential equation. In this project it can solve differention equation of enthalpy, pressure, velocity
#The parameters are:
#ai, bi, ci, di: the parameters of the differential equation
#A00, A01, Am0, Am1, D0, Dm1: the boundary conditions
#N_vol: the number of volumes
#H: the height of the fuel rod
class FVM:
    def __init__(self, A00, A01, Am0, Am1, D0, Dm1, N_vol, H):
        self.A00 = A00
        self.A01 = A01
        self.Am0 = Am0
        self.Am1 = Am1
        self.D0 = D0
        self.Dm1 = Dm1
        self.N_vol = N_vol
        self.H = H
        self.dz = H / N_vol
        self.z = np.linspace(0, H, self.N_vol)
        self.A, self.D = np.eye(self.N_vol), np.zeros(self.N_vol)

    #function to set the matrix A and D
    def set_ADi(self, i, ci, ai, bi, di):
        self.A[i, i-1:i+2] = [ci, ai, bi]
        self.D[i] = di
        return
    
    #function to set the boundary conditions
    def set_CL(self, A0, Am1, D0, Dm1):
        self.A[0], self.A[-1] = A0, Am1
        self.D[0], self.D[-1] = D0, Dm1
        return
    
    #function to solve the system of equations
    def resoudre_h(self):
        return np.linalg.solve(self.A, self.D)
    
    #function to set the transient parameters
    """ def set_transitoire(self, t_tot, Tini, dt):
        self.t_tot, self.dt = t_tot, dt           
        self.N_temps = round(self.t_tot / self.dt) # pas de temps (timesteps), il faut etre un nombre entier
        self.T = np.zeros((self.N_temps, self.N_vol)) # tableau 2D de temperature. 
        self.T[0] = Tini # Tini est une liste de temperature initiale """
        
    #function to fill the matrix A and D
    # def AD_filling(self):
    #     for i in range(1, self.N_vol-1):
    #         self.set_ADi(i, ci = self.ci,
    #         ai = self.ai,
    #         bi = self.bi,
    #         di = self.di )
    
    #function to set the boundary conditions
    def boundaryFilling(self):
        A0, Am1 = np.zeros(self.N_vol), np.zeros(self.N_vol)
        A0[:2] = [self.A00, self.A01]
        Am1[-2:] = [self.Am0, self.Am1]
        D0 = self.D0
        Dm1 = self.Dm1
        self.set_CL(A0, Am1, D0, Dm1)

    def fillingOutside(self, i, j, ci, ai, bi):
        self.A[i, j-1:j+2] = [ci, ai, bi]
        return 
    
    def fillingOutsideBoundary(self, i, j, ai, bi):
        self.A[i, j:j+2] = [ai, bi]
        return 
    
    #function to solve the differential equation of enthalppie
    def differential(self):
        self.boundaryFilling()
        self.h = self.resoudre_h()
        return self.h
    
    #function to calculate the temperature of the surface of the fuel using the fluid parameters, the distribution of enthalpie and the heat flux
    def verticalResolution(self):
        self.differential()

def resolveMVF(U_old, P_old, H_old, epsilon_old, rho_l_old, rho_g_old, rho_old, areaMatrix, areaMatrix_old_1, areaMatrix_old_2, Dhfg, V_gj_old, Vgj_prime, C0, x_th_old, nCells, U_inlet, P_outlet, h_inlet, DV, g, height, q__):
    VAR_old = mergeVar(U_old,P_old,H_old)
    rho_old = mergeVar(rho_old, rho_old, rho_old)
    rho_g_old = mergeVar(rho_g_old, rho_g_old, rho_g_old)
    rho_l_old = mergeVar(rho_l_old, rho_l_old, rho_l_old)
    epsilon_old = mergeVar(epsilon_old, epsilon_old, epsilon_old)
    areaMatrix = mergeVar(areaMatrix, areaMatrix, areaMatrix)
    areaMatrix_old_1 = mergeVar(areaMatrix_old_1, areaMatrix_old_1, areaMatrix_old_1)
    areaMatrix_old_2 = mergeVar(areaMatrix_old_2, areaMatrix_old_2, areaMatrix_old_2)
    V_gj_old = mergeVar(V_gj_old, V_gj_old, V_gj_old)
    Vgj_prime = mergeVar(Vgj_prime, Vgj_prime, Vgj_prime)
    Dhfg = mergeVar(Dhfg, Dhfg, Dhfg)
    C0 = mergeVar(C0, C0, C0)
    x_th_old = mergeVar(x_th_old, x_th_old, x_th_old)

    i = -1
    DI = (1/2) * (VAR_old[i-nCells]*areaMatrix[i] - VAR_old[i-1-nCells]*areaMatrix[i-1]) * ((VAR_old[i-2*nCells]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (VAR_old[i-1-2*nCells]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
    DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*areaMatrix[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*areaMatrix[i-1]/rho_old[i-1])
    DM1 = q__ * DV + DI + 0.1*DI2
    VAR_VFM_Class = FVM(A00 = 1, A01 = 0, Am0 = - rho_old[-2] * VAR_old[nCells-2] * areaMatrix[-2], Am1 = rho_old[-1] * VAR_old[nCells-1] * areaMatrix[-1], D0 = U_inlet, Dm1 = DM1, N_vol = 3*nCells, H = height)
    VAR_VFM_Class.boundaryFilling()
    for i in range(1, VAR_VFM_Class.N_vol-1):
        #Inside the velocity submatrix
        if i < nCells-1:
            VAR_VFM_Class.set_ADi(i, ci = - rho_old[i-1]*areaMatrix[i-1],
            ai = rho_old[i]*areaMatrix[i],
            bi = 0,
            di =  0)
        elif i == nCells-1:
            VAR_VFM_Class.set_ADi(i, 
            ci = - rho_old[i-1]*areaMatrix[i-1],
            ai = rho_old[i]*areaMatrix[i],
            bi = 0,
            di =  0)

        #Inside the pressure submatrix
        elif i == nCells:
            DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * areaMatrix[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * areaMatrix[i] )/ ((1 - epsilon_old[i])*rho_old[i]) )     
            VAR_VFM_Class.set_ADi(nCells, 
            ci = 0,
            ai = - areaMatrix[i],
            bi = areaMatrix[i+1],
            di = - ((rho_old[i+1]- rho_old[i])* g * DV / 2) + DI)
        
            VAR_VFM_Class.fillingOutsideBoundary(i, i-nCells,
            ai = - rho_old[i]*VAR_old[i-nCells]*areaMatrix_old_2[i],
            bi = rho_old[i+1]*VAR_old[i-nCells]*areaMatrix_old_1[i+1])

        elif i > nCells and i < 2*nCells-1:
            DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * areaMatrix[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * areaMatrix[i] )/ ((1 - epsilon_old[i])*rho_old[i]) )     
            VAR_VFM_Class.set_ADi(i, ci = 0,
            ai = - areaMatrix_old_2[i],
            bi = areaMatrix_old_1[i+1],
            di = - ((rho_old[i+1]- rho_old[i])* g * DV / 2) + DI)
        
            VAR_VFM_Class.fillingOutsideBoundary(i, i-nCells,
            ai = - rho_old[i]*VAR_old[i-nCells]*areaMatrix[i],
            bi = rho_old[i+1]*VAR_old[i+1-nCells]*areaMatrix[i+1])

        elif i == 2*nCells -1:
            VAR_VFM_Class.set_ADi(i, 
            ci = 0,
            ai = 1,
            bi = 0,
            di =  P_outlet)

            VAR_VFM_Class.fillingOutsideBoundary(2*nCells -1, 2*nCells -1 - nCells,
            ai = 0,
            bi = 0)

        #Inside the enthalpy submatrix
        elif i == 2*nCells:
            VAR_VFM_Class.set_ADi(2*nCells, 
            ci = 0,
            ai = 1,
            bi = 0,
            di =  h_inlet)

        elif i > 2*nCells and i < 3*nCells:
            DI = (1/2) * (VAR_old[i-nCells]*areaMatrix[i] - VAR_old[i-1-nCells]*areaMatrix[i-1]) * ((VAR_old[i-2*nCells]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (VAR_old[i-1-2*nCells]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
            DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*areaMatrix[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*areaMatrix[i-1]/rho_old[i-1])
            VAR_VFM_Class.set_ADi(i, ci =  - rho_old[i-1] * VAR_old[i-1-2*nCells] * areaMatrix[i-1],
            ai = rho_old[i] * VAR_old[i-2*nCells] * areaMatrix[i],
            bi = 0,
            di =  q__ * DV + DI + DI2)

    print(f'VAR_VFM_Class.A: {VAR_VFM_Class.A}, VAR_VFM_Class.D: {VAR_VFM_Class.D}')
    VAR = VAR_VFM_Class.resoudre_h()
    U, P, H = splitVar(VAR)

    return U, P, H


def sousRelaxation(Field, oldField, beta):
    n = len(Field)
    newField = np.zeros(n)
    for i in range(n):
        newField[i] = beta * Field[i] + (1 - beta) * oldField[i]
    return newField

