# This file contains the implementation of the drift flux model for the THM prototype
# Authors : Clement Huet
# Date : 2021-06-01

import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt
from FVM import FVM

class DFMclass():
    def __init__(self, canal_type, nCells, hInlet, uInlet, pOutlet, height, fuelRadius, cladRadius, cote,  numericalMethod, frfaccorel, P2P2corel, voidFractionCorrel):
        
        self.nCells = nCells
        self.uInlet = uInlet
        self.pOutlet = pOutlet
        self.hInlet = hInlet

        #Geometry parameters
        self.height = height #m
        self.fuelRadius = fuelRadius #External radius of the fuel m
        self.cladRadius = cladRadius #External radius of the clad m
        self.cote = cote
        self.wall_dist = cote
        self.canalType = canal_type

        if self.canalType == 'square':
            self.flowArea = self.cote ** 2
        elif self.canalType == 'cylindrical':
            self.waterGap = self.cote #Gap between the clad and the water m
            self.waterRadius =  self.cladRadius + self.waterGap #External radius of the water m
            self.flowArea = np.pi * self.waterRadius ** 2

        self.flowArea = self.cote ** 2
        self.DV = (self.height/self.nCells) * self.flowArea #Volume of the control volume m3
        print(f'flowArea: {self.flowArea}, self.cladRadius: {cladRadius}')
        self.D_h_fake = 4 * self.flowArea / (np.pi*self.cladRadius) #Hydraulic diameter m2
        self.D_h = 0.0078395462
        self.Dz = self.height/self.nCells #Height of the control volume m
        self.z_mesh = np.linspace(0, self.height, self.nCells)
        self.epsilonTarget = 0.18
        self.K_loss = 0

        print(f"D_h: {self.D_h}, D_h_fake: {self.D_h_fake}, DV: {self.DV}, Dz: {self.Dz}")

        self.epsInnerIteration = 1e-3
        self.maxInnerIteration = 1000
        self.sousRelaxFactor = 0.12
        self.epsOuterIteration = 1e-3
        self.maxOuterIteration = 1000

        #Universal constant
        self.g = 9.81 #m/s^2
        self.R = 8.314 #J/(mol*K)

        #residuals
        self.EPSresiduals = []
        self.rhoResiduals = []
        self.rhoGResiduals = []
        self.rhoLResiduals = []
        self.xThResiduals = []
        self.Iteration = []
        self.I = []

        #user choice
        self.frfaccorel = frfaccorel
        self.P2Pcorel = P2P2corel
        self.numericalMethod = numericalMethod
        self.voidFractionCorrel = voidFractionCorrel
        self.voidFractionEquation = 'base'

        self.hlSat = []
        self.hgSat = []
    
    def set_Fission_Power(self, Q):
        self.q__ = []
        for i in range(len(Q)):
            self.q__.append((np.pi * self.fuelRadius**2 * Q[i]) / self.flowArea) #W/m3

    def get_Fission_Power(self):
        """
        function to retrieve a given source term from the axial profile used to model fission power distribution in the fuel rod
        """
        return self.q__
        
    def setInitialFields(self): #crée les fields et remplis la premiere colonne
        self.U = [np.ones(self.nCells)*self.uInlet]
        self.P = [np.ones(self.nCells)*self.pOutlet]
        self.H = [np.ones(self.nCells)*self.hInlet]
        self.voidFraction = [np.array([i*self.epsilonTarget/self.nCells for i in range(self.nCells)])]

        updateVariables = statesVariables(self.U[-1], self.P[-1], self.H[-1], self.voidFraction[-1], self.D_h, self.flowArea, self.DV, self.voidFractionCorrel, self.frfaccorel, self.P2Pcorel)
        updateVariables.createFields()
            
        self.xTh = [np.ones(self.nCells)]
        self.rhoL= [updateVariables.rholTEMP]
        self.rhoG = [updateVariables.rhogTEMP]
        self.rho = [updateVariables.rhoTEMP]
        self.Dhfg = [updateVariables.DhfgTEMP]
        self.f = [updateVariables.fTEMP]
        self.areaMatrix_1 = [updateVariables.areaMatrix_1TEMP]
        self.areaMatrix_2 = [updateVariables.areaMatrix_2TEMP]
        self.areaMatrix = updateVariables.areaMatrixTEMP
        self.Vgj = [updateVariables.VgjTEMP]
        self.C0 =[updateVariables.C0TEMP]
        self.VgjPrime = [updateVariables.VgjPrimeTEMP]

    def resolveMVF(self):
            
        U_old = self.U[-1]
        P_old = self.P[-1]
        H_old = self.H[-1]
        epsilon_old = self.voidFraction[-1]
        rho_old = self.rho[-1]
        rho_g_old = self.rhoG[-1]
        rho_l_old = self.rhoL[-1]
        areaMatrix = self.areaMatrix
        areaMatrix_old_1 = self.areaMatrix_1[-1]
        areaMatrix_old_2 = self.areaMatrix_2[-1]
        Dhfg = self.Dhfg[-1]
        x_th_old = self.xTh[-1]
        f = self.f[-1]
        V_gj_old = self.Vgj[-1]
        Vgj_prime = self.VgjPrime[-1]
        C0 = self.C0[-1]

        VAR_old = self.mergeVar(U_old,P_old,H_old)
        rho_old = self.mergeVar(rho_old, rho_old, rho_old)
        rho_g_old = self.mergeVar(rho_g_old, rho_g_old, rho_g_old)
        rho_l_old = self.mergeVar(rho_l_old, rho_l_old, rho_l_old)
        epsilon_old = self.mergeVar(epsilon_old, epsilon_old, epsilon_old)
        areaMatrix = self.mergeVar(areaMatrix, areaMatrix, areaMatrix)
        areaMatrix_old_1 = self.mergeVar(areaMatrix_old_1, areaMatrix_old_1, areaMatrix_old_1)
        areaMatrix_old_2 = self.mergeVar(areaMatrix_old_2, areaMatrix_old_2, areaMatrix_old_2)
        V_gj_old = self.mergeVar(V_gj_old, V_gj_old, V_gj_old)
        Vgj_prime = self.mergeVar(Vgj_prime, Vgj_prime, Vgj_prime)
        Dhfg = self.mergeVar(Dhfg, Dhfg, Dhfg)
        C0 = self.mergeVar(C0, C0, C0)
        x_th_old = self.mergeVar(x_th_old, x_th_old, x_th_old)
        
        i = -1
        DI = (1/2) * (VAR_old[i-self.nCells]*areaMatrix[i] - VAR_old[i-1-self.nCells]*areaMatrix[i-1]) * ((VAR_old[i-2*self.nCells]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (VAR_old[i-1-2*self.nCells]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
        DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*areaMatrix[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*areaMatrix[i-1]/rho_old[i-1])
        DM1 = self.q__[i] * self.DV + DI + DI2
        VAR_VFM_Class = FVM(A00 = 1, A01 = 0, Am0 = - rho_old[-2] * VAR_old[self.nCells-2] * areaMatrix[-2], Am1 = rho_old[-1] * VAR_old[self.nCells-1] * areaMatrix[-1], D0 = self.uInlet, Dm1 = DM1, N_vol = 3*self.nCells, H = self.height)
        VAR_VFM_Class.boundaryFilling()
        for i in range(1, VAR_VFM_Class.N_vol-1):
            #Inside the velocity submatrix
            if i < self.nCells-1:
                VAR_VFM_Class.set_ADi(i, ci = - rho_old[i-1]*areaMatrix[i-1],
                ai = rho_old[i]*areaMatrix[i],
                bi = 0,
                di =  0)
            elif i == self.nCells-1:
                VAR_VFM_Class.set_ADi(i, 
                ci = - rho_old[i-1]*areaMatrix[i-1],
                ai = rho_old[i]*areaMatrix[i],
                bi = 0,
                di =  0)

            #Inside the pressure submatrix
            elif i == self.nCells:
                DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * areaMatrix[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * areaMatrix[i] )/ ((1 - epsilon_old[i])*rho_old[i]) )     
                VAR_VFM_Class.set_ADi(self.nCells, 
                ci = 0,
                ai = - areaMatrix[i],
                bi = areaMatrix[i+1],
                di = - ((rho_old[i+1]- rho_old[i])* self.g * self.DV / 2) + DI)
            
                VAR_VFM_Class.fillingOutsideBoundary(i, i-self.nCells,
                ai = - rho_old[i]*VAR_old[i-self.nCells]*areaMatrix_old_2[i],
                bi = rho_old[i+1]*VAR_old[i-self.nCells+1]*areaMatrix_old_1[i+1])

            elif i > self.nCells and i < 2*self.nCells-1:
                DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * areaMatrix[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * areaMatrix[i] )/ ((1 - epsilon_old[i])*rho_old[i]) )     
                VAR_VFM_Class.set_ADi(i, ci = 0,
                ai = - areaMatrix_old_2[i],
                bi = areaMatrix_old_1[i+1],
                di = - ((rho_old[i+1]- rho_old[i])* self.g * self.DV / 2) + DI)
            
                VAR_VFM_Class.fillingOutsideBoundary(i, i-self.nCells,
                ai = - rho_old[i]*VAR_old[i-self.nCells]*areaMatrix[i],
                bi = rho_old[i+1]*VAR_old[i+1-self.nCells]*areaMatrix[i+1])

            elif i == 2*self.nCells - 1:
                VAR_VFM_Class.set_ADi(i, 
                ci = 0,
                ai = 1,
                bi = 0,
                di =  self.pOutlet)

                VAR_VFM_Class.fillingOutsideBoundary(2*self.nCells -1, 2*self.nCells -1 - self.nCells,
                ai = 0,
                bi = 0)

            #Inside the enthalpy submatrix
            elif i == 2*self.nCells:
                VAR_VFM_Class.set_ADi(2*self.nCells, 
                ci = 0,
                ai = 1,
                bi = 0,
                di =  self.hInlet)

            elif i > 2*self.nCells and i < 3*self.nCells:
                DI = (1/2) * (VAR_old[i-self.nCells]*areaMatrix[i] - VAR_old[i-1-self.nCells]*areaMatrix[i-1]) * ((VAR_old[i-2*self.nCells]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (VAR_old[i-1-2*self.nCells]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
                DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*areaMatrix[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*areaMatrix[i-1]/rho_old[i-1])
                VAR_VFM_Class.set_ADi(i, ci =  - rho_old[i-1] * VAR_old[i-1-2*self.nCells] * areaMatrix[i-1],
                ai = rho_old[i] * VAR_old[i-2*self.nCells] * areaMatrix[i],
                bi = 0,
                di =  self.q__[i%self.nCells] * self.DV + DI + DI2)

        VAR = VAR_VFM_Class.resoudre_h()
        U, P, H = self.splitVar(VAR)

        return U, P, H

    def calculateResiduals(self):#change les residus
        self.EPSresiduals.append(np.linalg.norm(self.voidFraction[-1] - self.voidFraction[-2]))
        self.rhoResiduals.append(np.linalg.norm(self.rho[-1] - self.rho[-2]))
        self.rhoGResiduals.append(np.linalg.norm(self.rhoG[-1] - self.rhoG[-2]))
        self.rhoLResiduals.append(np.linalg.norm(self.rhoL[-1] - self.rhoL[-2]))
        self.xThResiduals.append(np.linalg.norm(self.xTh[-1] - self.xTh[-2]))

    def testConvergence(self, k):#change rien et return un boolean
        print(f'Convergence test: EPS: {self.EPSresiduals[-1]}, rho: {self.rhoResiduals[-1]}, rhoG: {self.rhoGResiduals[-1]}, rhoL: {self.rhoLResiduals[-1]}, xTh: {self.xThResiduals[-1]}')
        if self.EPSresiduals[-1] < 1e-3 and self.rhoGResiduals[-1] < 1e-3 and self.rhoLResiduals[-1] < 1e-3 and self.xThResiduals[-1] < 1e-3:
            print(f'Convergence reached at iteration number: {k}')
            return True
        else:
            return False

    def resolveDFM(self):

        self.setInitialFields()
 
        for k in range(self.maxOuterIteration):
            if self.numericalMethod == 'FVM':
                Utemp, Ptemp, Htemp = self.resolveMVF()
                self.U.append(Utemp)
                self.P.append(Ptemp)
                self.H.append(Htemp)
            
            updateVariables = statesVariables(self.U[-1], self.P[-1], self.H[-1], self.voidFraction[-1], self.D_h, self.flowArea, self.DV, self.voidFractionCorrel, self.frfaccorel, self.P2Pcorel)
            updateVariables.updateFields()

            self.xTh.append(updateVariables.xThTEMP)
            self.rhoL.append(updateVariables.rholTEMP)
            self.rhoG.append(updateVariables.rhogTEMP)
            self.rho.append(updateVariables.rhoTEMP)
            self.voidFraction.append(updateVariables.voidFractionTEMP)
            self.Dhfg.append(updateVariables.DhfgTEMP)
            self.f.append(updateVariables.fTEMP)
            self.areaMatrix_1.append(updateVariables.areaMatrix_1TEMP)
            self.areaMatrix_2.append(updateVariables.areaMatrix_2TEMP)
            self.Vgj.append(updateVariables.VgjTEMP)
            self.C0.append(updateVariables.C0TEMP)
            self.VgjPrime.append(updateVariables.VgjPrimeTEMP)

            self.sousRelaxation()
            self.calculateResiduals()
            self.I.append(k)
            convergence = self.testConvergence(k)

            if convergence == True:
                break

            elif k == self.maxOuterIteration - 1:
                print('Convergence not reached')
                break
        
        self.T_water = np.zeros(self.nCells)
        for i in range(self.nCells):
            print(f'self.P : {self.P[-1][i]*10**-6}, self.H = {self.H[-1][i]*10**-3}')
            self.T_water[i] = IAPWS97(P=self.P[-1][i]*10**-6, h=self.H[-1][i]*10**-3).T

    def compute_T_surf(self):
        self.Pfin = self.P[-1]
        self.h_z = self.H[-1]
        self.T_surf = np.zeros(self.nCells)
        self.Hc = np.zeros(self.nCells)
        for i in range(self.nCells):
            print(f'At axial slice = {i}, Pfin = {self.Pfin[i]}, h_z = {self.h_z[i]}')
            Pr_number = IAPWS97(P=self.Pfin[i]*10**-6, h=self.h_z[i]*10**-3).Liquid.Prandt
            Re_number = self.getReynoldsNumber(i)
            k_fluid = IAPWS97(P=self.Pfin[i]*10**-6, h=self.h_z[i]*10**-3).Liquid.k
            print(f"At axial slice = {i}, computed Reynold # = {Re_number}, computed Prandt # = {Pr_number}, k_fluid = {k_fluid}")
            self.Hc[i] = (0.023)*(Pr_number)**0.4*(Re_number)**0.8*k_fluid/self.D_h
            print(f'self.Hc[i]: {self.Hc[i]}, \n self.q__[i]: {self.q__[i]} ,\n 2*np.pi*self.cladRadius: {2*np.pi*self.cladRadius}')
            self.T_surf[i] = ((self.q__[i]*self.flowArea)/(2*np.pi*self.cladRadius)/self.Hc[i]+self.T_water[i])
    
        return self.T_surf

    def sousRelaxation(self):

        for i in range(self.nCells):
            self.voidFraction[-1][i] = self.voidFraction[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.voidFraction[-2][i]
            self.rho[-1][i] = self.rho[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.rho[-2][i]
            self.rhoG[-1][i] = self.rhoG[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.rhoG[-2][i]
            self.rhoL[-1][i] = self.rhoL[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.rhoL[-2][i]
            self.xTh[-1][i] = self.xTh[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.xTh[-2][i]
            self.Vgj[-1][i] = self.Vgj[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.Vgj[-2][i]
            self.C0[-1][i] = self.C0[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.C0[-2][i]
            self.VgjPrime[-1][i] = self.VgjPrime[-1][i] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.VgjPrime[-2][i]
    
    def mergeVar(self, U, P, H): #créer une liste a partir de 3 liste
        VAR = np.concatenate((U, P, H))
        return VAR
    
    def splitVar(self, VAR): #créer 3 liste a partir d'une liste
        U = VAR[:self.nCells]
        P = VAR[self.nCells:2*self.nCells]
        H = VAR[2*self.nCells:]
        return U, P, H
    
    def createBoundaryEnthalpy(self):
        for i in range(self.nCells):
            self.hlSat.append(self.getPhasesEnthalpy(i)[0])
            self.hgSat.append(self.getPhasesEnthalpy(i)[1]) 
 
    def getReynoldsNumber(self, i):
        return (self.U[-1][i] * self.D_h * self.rho[-1][i]) / IAPWS97(P=self.P[-1][i]*10**-6, x=0).Liquid.mu
     

class statesVariables():
    def __init__(self, U, P, H, voidFraction, D_h, flowarea, DV, voidFractionCorrel, frfaccorel, P2Pcorel):
        
        self.nCells = len(U)
        self.U = U
        self.P = P
        self.H = H
        self.voidFraction = voidFraction
        self.voidFractionCorrel = voidFractionCorrel
        self.frfaccorel = frfaccorel
        self.P2Pcorel = P2Pcorel
        self.g = 9.81
        self.D_h = D_h
        self.flowArea = flowarea
        self.K_loss = 0
        self.Dz = 1
        self.DV = DV

    def createFields(self):

        self.areaMatrixTEMP = np.ones(self.nCells)
        self.rholTEMP, self.rhogTEMP, self.rhoTEMP, self.voidFractionTEMP, self.DhfgTEMP, self.fTEMP, self.areaMatrix_1TEMP, self.areaMatrix_2TEMP, self.areaMatrix_2TEMP, self.VgjTEMP, self.C0TEMP, self.VgjPrimeTEMP = np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells)
        self.voidFractionTEMP = self.voidFraction
        self.xThTEMP = np.ones(self.nCells)
        for i in range(self.nCells):
            self.xThTEMP[i] = self.getQuality(i)
            self.areaMatrixTEMP[i] = self.flowArea
            self.rholTEMP[i], self.rhogTEMP[i], self.rhoTEMP[i] = self.getDensity(i)
            self.C0TEMP[i] = self.getC0(i)
            self.VgjTEMP[i] = self.getVgj(i)
            self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
            self.DhfgTEMP[i] = self.getHfg(i)
            self.fTEMP[i] = self.getFrictionFactor(i)
            self.areaMatrix_1TEMP[i], self.areaMatrix_2TEMP[i] = self.getAreas(i)

    def updateFields(self):

        self.xThTEMP = np.ones(self.nCells)
        for i in range(self.nCells):
            self.xThTEMP[i] = self.getQuality(i)
        
        if self.voidFractionCorrel == 'modBestion':
            self.modBestion()

        elif self.voidFractionCorrel == 'HEM1':
            self.HEM1()

        elif self.voidFractionCorrel == 'GEramp':
            self.GEramp()

        elif self.voidFractionCorrel == 'EPRIvoidModel':
            self.EPRIvoidModel()

    
    def modBestion(self):
        print('in modBestion')
        self.rholTEMP, self.rhogTEMP, self.rhoTEMP, self.voidFractionTEMP, self.DhfgTEMP, self.fTEMP, self.areaMatrix_1TEMP, self.areaMatrix_2TEMP, self.areaMatrix_2TEMP, self.VgjTEMP, self.C0TEMP, self.VgjPrimeTEMP = np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells)
        self.voidFractionOld = self.voidFraction
        for i in range(self.nCells):
            self.rholTEMP[i], self.rhogTEMP[i], self.rhoTEMP[i] = self.getDensity(i)
            self.C0TEMP[i] = self.getC0(i)
            self.VgjTEMP[i] = self.getVgj(i)
            self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
            self.DhfgTEMP[i] = self.getHfg(i)
            voidFractionNew = self.getVoidFraction(i)
            self.voidFractionTEMP[i] = voidFractionNew
            self.rhoTEMP[i] = self.getDensity(i)[2]
            self.voidFractionTEMP[i] = voidFractionNew
            self.rhoTEMP[i] = self.getDensity(i)[2]
            self.fTEMP[i] = self.getFrictionFactor(i)
            self.areaMatrix_1TEMP[i], self.areaMatrix_2TEMP[i] = self.getAreas(i)
    
    def HEM1(self):
        self.rholTEMP, self.rhogTEMP, self.rhoTEMP, self.voidFractionTEMP, self.DhfgTEMP, self.fTEMP, self.areaMatrix_1TEMP, self.areaMatrix_2TEMP, self.areaMatrix_2TEMP, self.VgjTEMP, self.C0TEMP, self.VgjPrimeTEMP = np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells)
        self.voidFractionOld = self.voidFraction
        for i in range(self.nCells):
            self.rholTEMP[i], self.rhogTEMP[i], self.rhoTEMP[i] = self.getDensity(i)
            self.C0TEMP[i] = self.getC0(i)
            self.VgjTEMP[i] = self.getVgj(i)
            self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
            self.DhfgTEMP[i] = self.getHfg(i)
            voidFractionNew = self.getVoidFraction(i)
            self.voidFractionTEMP[i] = voidFractionNew
            self.rhoTEMP[i] = self.getDensity(i)[2]
            self.voidFractionTEMP[i] = voidFractionNew
            self.rhoTEMP[i] = self.getDensity(i)[2]
            self.fTEMP[i] = self.getFrictionFactor(i)
            self.areaMatrix_1TEMP[i], self.areaMatrix_2TEMP[i] = self.getAreas(i)

    def GEramp(self):
        print('in GE Ramp')
        self.rholTEMP, self.rhogTEMP, self.rhoTEMP, self.voidFractionTEMP, self.DhfgTEMP, self.fTEMP, self.areaMatrix_1TEMP, self.areaMatrix_2TEMP, self.areaMatrix_2TEMP, self.VgjTEMP, self.C0TEMP, self.VgjPrimeTEMP = np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells)
        self.voidFractionOld = self.voidFraction
        for i in range(self.nCells):
            self.rholTEMP[i], self.rhogTEMP[i], self.rhoTEMP[i] = self.getDensity(i)
            self.C0TEMP[i] = self.getC0(i)
            self.VgjTEMP[i] = self.getVgj(i)
            self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
            self.DhfgTEMP[i] = self.getHfg(i)
            for j in range(1000):
                voidFractionNew = self.getVoidFraction(i)
                if np.linalg.norm(voidFractionNew - self.voidFractionTEMP[i]) < 1e-3:
                    self.voidFractionTEMP[i] = voidFractionNew
                    self.rhoTEMP[i] = self.getDensity(i)[2]
                    self.C0TEMP[i] = self.getC0(i)
                    self.VgjTEMP[i] = self.getVgj(i)
                    self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
                    break
                elif j == 999:
                    print('Convergence in update fields not reached')
                    break
                else:
                    self.voidFractionTEMP[i] = voidFractionNew
                    self.rhoTEMP[i] = self.getDensity(i)[2]
                    self.C0TEMP[i] = self.getC0(i)
                    self.VgjTEMP[i] = self.getVgj(i)
                    self.VgjPrimeTEMP[i] = self.getVgj_prime(i)

            self.fTEMP[i] = self.getFrictionFactor(i)
            self.areaMatrix_1TEMP[i], self.areaMatrix_2TEMP[i] = self.getAreas(i)

    def EPRIvoidModel(self):
        self.rholTEMP, self.rhogTEMP, self.rhoTEMP, self.voidFractionTEMP, self.DhfgTEMP, self.fTEMP, self.areaMatrix_1TEMP, self.areaMatrix_2TEMP, self.areaMatrix_2TEMP, self.VgjTEMP, self.C0TEMP, self.VgjPrimeTEMP = np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells), np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells),np.ones(self.nCells)
        self.voidFractionOld = self.voidFraction
        for i in range(self.nCells):
            self.rholTEMP[i], self.rhogTEMP[i], self.rhoTEMP[i] = self.getDensity(i)
            self.C0TEMP[i] = self.getC0(i)
            self.VgjTEMP[i] = self.getVgj(i)
            self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
            self.DhfgTEMP[i] = self.getHfg(i)
            for j in range(1000):
                voidFractionNew = self.getVoidFraction(i)
                if np.linalg.norm(voidFractionNew - self.voidFractionTEMP[i]) < 1e-3:
                    self.voidFractionTEMP[i] = voidFractionNew
                    self.rhoTEMP[i] = self.getDensity(i)[2]
                    self.C0TEMP[i] = self.getC0(i)
                    self.VgjTEMP[i] = self.getVgj(i)
                    self.VgjPrimeTEMP[i] = self.getVgj_prime(i)
                    break
                elif j == 999:
                    print('Convergence in update fields not reached')
                    break
                else:
                    self.voidFractionTEMP[i] = voidFractionNew
                    self.rhoTEMP[i] = self.getDensity(i)[2]
                    self.C0TEMP[i] = self.getC0(i)
                    self.VgjTEMP[i] = self.getVgj(i)
                    self.VgjPrimeTEMP[i] = self.getVgj_prime(i)

            self.fTEMP[i] = self.getFrictionFactor(i)
            self.areaMatrix_1TEMP[i], self.areaMatrix_2TEMP[i] = self.getAreas(i)

    def getDensity(self, i):
        vapor = IAPWS97(P = self.P[i]*(10**(-6)), x = 1)
        liquid = IAPWS97(P = self.P[i]*(10**(-6)), x = 0)
        rho_g = vapor.rho
        rho_l = liquid.rho
        rho = rho_l * (1 - self.voidFractionTEMP[i]) + rho_g * self.voidFractionTEMP[i]
        return rho_l, rho_g, rho
    
    def getQuality(self, i):
        hl, hg = self.getPhasesEnthalpy(i)
        H = self.H[i]
        if H*0.001 < hl:
            return 0
        elif H*0.001 > hg:
            return 1
        elif H*0.001 <= hg and H*0.001 >= hl:
            return (H*0.001 - hl)/(hg - hl)
    
    def getVoidFraction(self, i):
        correl = 'paths'
        if correl == 'simple':
            x_th = self.xThTEMP[i]
            rho_l = self.rholTEMP[i]
            rho_g = self.rhogTEMP[i]
            if x_th == 0:
                return 0.0001
            elif x_th == 1:
                return 0.99
            else:
                return (x_th * rho_l)/(x_th * rho_l + (1 - x_th) * rho_g)
        elif correl == 'paths':
            x_th = self.xThTEMP[i]
            rho_l = self.rholTEMP[i]
            rho_g = self.rhogTEMP[i]
            u = self.U[i]
            V_gj = self.VgjTEMP[i]
            C0 = self.C0TEMP[i]
            if x_th == 0:
                return 0.0001
            elif x_th == 1:
                return 0.99
            else:
                return x_th / (C0 * (x_th + (rho_g / rho_l) * (1 - x_th)) + (rho_g * V_gj) / (rho_l * u))
    
    def getVgj(self, i):
        if self.voidFractionCorrel == 'GEramp':
            if self.rhogTEMP[i] == 0:
                return 0
            if self.rholTEMP[i] == 0:
                return 0
            
            sigma = IAPWS97(P = self.P[i]*(10**(-6)), x = 0).sigma
            if sigma == 0:
                return 0
            
            Vgj0 = ((self.g * sigma * (self.rholTEMP[i] - self.rhogTEMP[i]) / self.rholTEMP[i]**2)**0.25)

            if self.voidFractionTEMP[i] <= 0.65:
                return 2.9 * Vgj0
            elif self.voidFractionTEMP[i] > 0.65:
                return (2.9/0.35)*(1-self.voidFractionTEMP[i]) * Vgj0
        
        if self.voidFractionCorrel == 'modBestion':
            if self.rhogTEMP[i] == 0:
                return 0
            if self.rholTEMP[i] == 0:
                return 0
            return 0.188 * np.sqrt(((self.rholTEMP[i] - self.rhogTEMP[i]) * self.g * self.D_h ) / self.rhogTEMP[i] )
        
        if self.voidFractionCorrel == 'EPRIvoidModel':
            if self.rhogTEMP[i] == 0:
                return 0
            if self.rholTEMP[i] == 0:
                return 0
            sigma = IAPWS97(P = self.P[i]*(10**(-6)), x = 0).sigma
            Vgj = (np.sqrt(2)*(self.g * sigma * (self.rholTEMP[i] - self.rhogTEMP[i]) / self.rholTEMP[i]**2)**0.25) * (1 + self.voidFractionTEMP[i])**(3/2)
            return Vgj
        
        if self.voidFractionCorrel == 'HEM1':
            return 0
            
            
    
    def getC0(self, i):
        if self.voidFractionCorrel == 'GEramp':
            rho_g = self.rhogTEMP[i]
            rho_l = self.rholTEMP[i]
            if rho_g == 0:
                return 0
            if rho_l == 0:
                return 0
            if self.voidFractionTEMP[i] <= 0.65:
                return 1.1
            elif self.voidFractionTEMP[i] > 0.65:
                return 1 + (0.1/0.35)*(1-self.voidFractionTEMP[i])
        
        if self.voidFractionCorrel == 'modBestion':
            rho_g = self.rhogTEMP[i]
            rho_l = self.rholTEMP[i]
            if rho_g == 0:
                return 0
            if rho_l == 0:
                return 0
            return 1.2 - 0.2*np.sqrt(rho_g / rho_l)
        
        if self.voidFractionCorrel == 'EPRIvoidModel':
            rho_g = self.rhogTEMP[i]
            rho_l = self.rholTEMP[i]
            Pc = 22060000
            P = self.P[i]
            Re = self.getReynoldsNumber(i)
            C1 = (4 * Pc**2)/(P*(Pc - P))
            k1 = min(0.8, 1/(1 + np.exp(-Re /60000)))
            k0 = k1 + (1-k1) * (rho_g / rho_l)**2
            r = (1+1.57*(rho_g/rho_l))/(1-k1)
            C0 = (((k0 + (1 - k0) * (self.voidFractionTEMP[i]**r))**(-1)) * ((1 - np.exp(-C1 * self.voidFractionTEMP[i]))/(1 - np.exp(-C1))))
            return C0

        if self.voidFractionCorrel == 'HEM1':
            return 1
            
        
    def getVgj_prime(self, i):
        U = self.U[i]
        C0 = self.C0TEMP[i]
        Vgj = self.VgjTEMP[i]
        Vgj_prime = Vgj + (C0 - 1) * U
        return Vgj_prime
    
    def getHfg(self, i):
        vapor = IAPWS97(P = self.P[i]*(10**(-6)), x = 1)
        liquid = IAPWS97(P = self.P[i]*(10**(-6)), x = 0)
        return (vapor.h - liquid.h)
    
    def getFrictionFactor(self, i):
        U = self.U[i]
        P = self.P[i]

        if self.frfaccorel == 'base': #Validated
            return 0.000033
        Re = self.getReynoldsNumber(i)
        if self.frfaccorel == 'blasius': #Not Validated
            return 0.186 * Re**(-0.2)
        if self.frfaccorel == 'Churchill': #Not Validated
            Ra = 0.4 * (10**(-6)) #Roughness
            R = Ra / self.D_h
            frict=8*(((8.0/Re)**12)+((2.475*np.log(((7/Re)**0.9)+0.27*R))**16+(37530/Re)**16)**(-1.5))**(1/12)   
            return frict
        
    def getPhi2Phi(self, i):
        x_th = self.xThTEMP[i]
        rho_l = self.rholTEMP[i]
        rho_g = self.rhogTEMP[i]
        rho = self.rhoTEMP[i]
        P = self.P[i]
        epsilon = self.voidFractionTEMP[i]

        if self.P2Pcorel == 'base': #Validated
            phi2phi = 1 + 3*epsilon
        elif self.P2Pcorel == 'HEM1': #Validated
            phi2phi = (rho/rho_l)*((rho_l/rho_g)*x_th + +1)
        elif self.P2Pcorel == 'HEM2': #Validated
            m = IAPWS97(P = P*(10**(-6)), x = 0).mu / IAPWS97(P = P*(10**(-6)), x = 1).mu
            phi2phi = (rho/rho_l)*((m-1)*x_th + 1)*((rho_l/rho_g)*x_th + +1)**(0.25)
        elif self.P2Pcorel == 'MNmodel': #Validated
            phi2phi = (1.2 * (rho_l/rho_g -1)*x_th**(0.824) + 1)*(rho/rho_l)
        return phi2phi
    
    def getAreas(self, i):
        A_chap_pos = self.flowArea +  (self.getPhi2Phi(i)/2) * ((self.fTEMP[i] / self.D_h) + (self.K_loss / self.Dz)) * self.DV
        A_chap_neg = self.flowArea - (self.getPhi2Phi(i)/2) * ((self.fTEMP[i] / self.D_h) + (self.K_loss / self.Dz)) * self.DV
        return A_chap_pos, A_chap_neg

    def getPhasesEnthalpy(self, i):
        P = self.P[i]
        vapor = IAPWS97(P = P*(10**(-6)), x = 1)
        liquid = IAPWS97(P = P*(10**(-6)), x = 0)
        return liquid.h, vapor.h
    
    def getReynoldsNumber(self, i):
        U = self.U[i]
        rho = self.rhoTEMP[i]
        P = self.P[i]
        alpha = self.voidFractionTEMP[i]
        print(f'At axial slice = {i}, P = {P}')
        ml = IAPWS97(P = P*(10**(-6)), x = 0).mu
        mv = IAPWS97(P = P*(10**(-6)), x = 1).mu
        m = (mv * ml) / ( ml * (1 - alpha) + mv * alpha )
        
        print(f'At axial slice = {i}, computed Reynold # = {rho * abs(U) * self.D_h / m}')
        return rho * abs(U) * self.D_h / m

