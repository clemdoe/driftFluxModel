import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt
from FVM import FVM

class DFMclass():
    def __init__(self, nCells, uInlet, pOutlet, hInlet, height, fuelRadius, cladRadius, cote,  numericalMethod, frfaccorel, P2P2corel, voidFractionCorrel, Qmax, Qtype):
        
        self.nCells = nCells
        self.uInlet = uInlet
        self.pOutlet = pOutlet
        self.hInlet = hInlet

        #Geometry parameters
        self.height = height #m
        self.fuelRadius = fuelRadius #External radius of the fuel m
        self.cladRadius = cladRadius #External radius of the clad m
        self.cote = cote
        self.canalType = 'square'

        if self.canalType == 'square':
            self.flowArea = self.cote ** 2
        elif self.canalType == 'circular':
            self.waterGap = self.cote #Gap between the clad and the water m
            self.waterRadius =  self.cladRadius + self.waterGap #External radius of the water m
            self.flowArea = np.pi * self.waterRadius ** 2

        self.DV = (self.height/self.nCells) * self.flowArea #Volume of the control volume m3
        self.D_h = self.flowArea / (np.pi*self.cladRadius) #Hydraulic diameter m2
        self.Dz = self.height/self.nCells #Height of the control volume m
        self.zList = np.linspace(0, self.height, self.nCells)
        self.epsilonTarget = 0.18
        self.K_loss = 0

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

        #Heating parameters
        self.startheating = 0
        self.endheating = self.height
        self.Q = []
        for z in self.zList:
            if z >= self.startheating and z <= self.endheating:
                
                if Qtype == 'constant':
                    self.Q.append(Qmax)
                elif Qtype == 'sinusoidal':
                    self.Q.append(Qmax*np.sin(z*np.pi/self.height)) #volumetric heat generation rate W/m3
            else:
                self.Q.append(0)
        self.q__ = []

        for i in range(len(self.Q)):
            #self.q__.append(10.910/self.DV)
            self.q__.append((np.pi * self.fuelRadius**2 * self.Q[i]) / self.flowArea) #W/m3
        

        self.hlSat = []
        self.hgSat = []
     

    def setInitialFields(self): #crée les fields et remplis la premiere colonne
        self.U = [np.ones(self.nCells)*self.uInlet]
        self.P = [np.ones(self.nCells)*self.pOutlet]
        self.H = [np.ones(self.nCells)*self.hInlet]
        self.voidFraction = [np.array([i*self.epsilonTarget/self.nCells for i in range(self.nCells)])]

        self.rho = [np.ones(self.nCells)]
        self.rhoG = [np.ones(self.nCells)]
        self.rhoL = [np.ones(self.nCells)]
        self.areaMatrix = np.ones(self.nCells)
        self.areaMatrix_1 = [np.ones(self.nCells)]
        self.areaMatrix_2 = [np.ones(self.nCells)]
        self.Dhfg = [np.ones(self.nCells)]
        self.xTh = [np.ones(self.nCells)]
        self.f = [np.ones(self.nCells)]
        self.Vgj = [np.ones(self.nCells)]
        self.VgjPrime = [np.ones(self.nCells)]
        self.C0 = [np.ones(self.nCells)]
        self.xTh = [np.ones(self.nCells)]

        for i in range(self.nCells):
            self.rhoL[-1][i], self.rhoG[-1][i], self.rho[-1][i] = self.getDensity(i) 
        for i in range(self.nCells):
            self.Dhfg[0][i] = self.getHfg(i)
        for i in range(self.nCells):
            self.f[0][i] = self.getFrictionFactor(i) 
        for i in range(self.nCells):
            self.areaMatrix_1[0][i], self.areaMatrix_2[0][i] = self.getAreas(i)
            self.areaMatrix[i] = self.flowArea
        for i in range(self.nCells):
            self.Vgj[0][i] = self.getVgj(i) 
            self.C0[0][i] = self.getC0(i)
        for i in range(self.nCells):
            self.VgjPrime[0][i] = self.getVgj_prime(i)

        
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
    
    def updateFields(self):

        self.xTh.append(np.ones(self.nCells))
        for i in range(self.nCells):
            self.xTh[-1][i] = self.getQuality(i)
        
        if self.voidFractionCorrel == 'modBestion':
            self.modBestion()

        elif self.voidFractionCorrel == 'HEM1':
            self.HEM1()

        elif self.voidFractionCorrel == 'GEramp':
            self.GEramp()

        elif self.voidFractionCorrel == 'EPRIvoidModel':
            self.EPRIvoidModel()

    def EPRIvoidModel(self):
        pass

    def GEramp(self):

        self.rhoL.append(np.ones(self.nCells))
        self.rhoG.append(np.ones(self.nCells))
        self.rho.append(np.ones(self.nCells))

        for i in range(self.nCells):
            rho_l_new, rho_g_new, rho_new = self.getDensity(i)
            self.rhoL[-1][i] = rho_l_new
            self.rhoG[-1][i] = rho_g_new

        self.voidFraction.append(np.ones(self.nCells))
        for i in range(self.nCells):
            self.voidFraction[-1][i] = self.getVoidFraction(i)

        for i in range(self.nCells):
            rho_l_new, rho_g_new, rho_new = self.getDensity(i)
            self.rho[-1][i] = rho_new

        self.Dhfg.append(np.ones(self.nCells))
        for i in range(self.nCells):
            self.Dhfg[-1][i] = self.getHfg(i)

        self.f.append(np.ones(self.nCells))
        for i in range(self.nCells):
            self.f[-1][i] = self.getFrictionFactor(i)

        self.areaMatrix_1.append(np.ones(self.nCells))
        self.areaMatrix_2.append(np.ones(self.nCells))
        for i in range(self.nCells):
            self.areaMatrix_1[-1][i] = self.getAreas(i)[0]
            self.areaMatrix_2[-1][i] = self.getAreas(i)[1]

        self.Vgj.append(np.ones(self.nCells))
        self.C0.append(np.ones(self.nCells))
        for i in range(self.nCells):
            self.Vgj[-1][i] = self.getVgj(i)
            self.C0[-1][i] = self.getC0(i)

        self.VgjPrime.append(np.ones(self.nCells))
        for i in range(self.nCells):
            self.VgjPrime[-1][i] = self.getVgj_prime(i)

    def HEM1(self):
        pass

    def modBestion(self):

        for k in range(1000):
            self.rhoL.append(np.ones(self.nCells))
            self.rhoG.append(np.ones(self.nCells))
            self.rho.append(np.ones(self.nCells))
            for i in range(self.nCells):
                rho_l_new, rho_g_new, rho_new = self.getDensity(i)
                self.rhoL[-1][i] = rho_l_new
                self.rhoG[-1][i] = rho_g_new

            self.voidFraction.append(np.ones(self.nCells))
            for i in range(self.nCells):
                self.voidFraction[-1][i] = self.getVoidFraction(i)

            for i in range(self.nCells):
                rho_l_new, rho_g_new, rho_new = self.getDensity(i)
                self.rho[-1][i] = rho_new

            self.Dhfg.append(np.ones(self.nCells))
            for i in range(self.nCells):
                self.Dhfg[-1][i] = self.getHfg(i)

            self.f.append(np.ones(self.nCells))
            for i in range(self.nCells):
                self.f[-1][i] = self.getFrictionFactor(i)

            self.areaMatrix_1.append(np.ones(self.nCells))
            self.areaMatrix_2.append(np.ones(self.nCells))
            for i in range(self.nCells):
                self.areaMatrix_1[-1][i] = self.getAreas(i)[0]
                self.areaMatrix_2[-1][i] = self.getAreas(i)[1]

            self.Vgj.append(np.ones(self.nCells))
            self.C0.append(np.ones(self.nCells))
            for i in range(self.nCells):
                self.Vgj[-1][i] = self.getVgj(i)
                self.C0[-1][i] = self.getC0(i)
                
            self.VgjPrime.append(np.ones(self.nCells))
            for i in range(self.nCells):
                self.VgjPrime[-1][i] = self.getVgj_prime(i)
 
            if np.linalg.norm(self.voidFraction[-2] - self.voidFraction[-1]) < 1e-3:
                break

            elif k == 999:
                print('Convergence in update fields not reached')
                break

        for j in range(1,k):
            np.delete(self.rhoL, -(k+1))
            np.delete(self.rhoG, -(k+1))
            np.delete(self.rho, -(k+1))
            np.delete(self.voidFraction, -(k+1))
            np.delete(self.Dhfg, -(k+1))
            np.delete(self.f, -(k+1))
            np.delete(self.areaMatrix_1, -(k+1))
            np.delete(self.areaMatrix_2, -(k+1))
            np.delete(self.Vgj, -(k+1))
            np.delete(self.C0, -(k+1))
            np.delete(self.VgjPrime, -(k+1))


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
            
            self.updateFields()
            self.sousRelaxation()
            self.calculateResiduals()
            self.I.append(k)
            convergence = self.testConvergence(k)

            if convergence == True:
                break

            elif k == self.maxOuterIteration - 1:
                print('Convergence not reached')
                break

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

    def getAreas(self, i): #phi2phi in m3/m3, f in m
        A_chap_pos = self.flowArea +  (self.getPhi2Phi(i)/2) * ((self.f[-1][i] / self.D_h) + (self.K_loss / self.Dz)) * self.DV
        A_chap_neg = self.flowArea - (self.getPhi2Phi(i)/2) * ((self.f[-1][i] / self.D_h) + (self.K_loss / self.Dz)) * self.DV
        return A_chap_pos, A_chap_neg
    
    def getQuality(self, i):
        H = self.H[-1][i]
        hl, hg = self.getPhasesEnthalpy(i)
        if H*0.001 < hl:
            return 0
        elif H*0.001 > hg:
            return 1
        elif H*0.001 <= hg and H*0.001 >= hl:
            return (H*0.001 - hl)/(hg - hl)
        
    def getPhasesEnthalpy(self, i):
        P = self.P[-1][i]
        vapor = IAPWS97(P = P*(10**(-6)), x = 1)
        liquid = IAPWS97(P = P*(10**(-6)), x = 0)
        return liquid.h, vapor.h
    
    def getDensity(self, i): #P in Pa, x in m3/m3
        vapor = IAPWS97(P = self.P[-1][i]*(10**(-6)), x = 1)
        liquid = IAPWS97(P = self.P[-1][i]*(10**(-6)), x = 0)
        rho_g = vapor.rho
        rho_l = liquid.rho

        rho = rho_l * (1 - self.voidFraction[-1][i]) + rho_g * self.voidFraction[-1][i]
        return rho_l, rho_g, rho
    
    def getVgj(self, i):
        rho_g = self.rhoG[-1][i]
        rho_l = self.rhoL[-1][i]
        if self.voidFractionCorrel == 'modBestion':
            if rho_g == 0:
                return 0
            if rho_l == 0:
                return 0
            return 0.188 * np.sqrt(((rho_l - rho_g) * self.g * self.D_h ) / rho_g )
        
        elif self.voidFractionCorrel == 'GEramp':
            sigma = IAPWS97(P = self.P[-1][i]*(10**(-6)), x = 0).sigma
            Vgj0 = ((self.g * sigma * (rho_l - rho_g) / (rho_l**2))**(1/4))*(1-self.voidFraction[-1][i])**(3/2)
            if self.voidFraction[-1][i] <= 0.65:
                return 2.9 * Vgj0
            else:
                return 2.9 * ( 1 - self.voidFraction[-1][i] ) * Vgj0 / 0.35
    
    def getVgj_prime(self, i):
        if self.voidFractionCorrel == 'modBestion':
            rho_g = self.rhoG[-1][i]
            rho_l = self.rhoL[-1][i]
            U = self.U[-1][i]

            C0 = self.C0[-1][i]
            Vgj = self.Vgj[-1][i]
            Vgj_prime = Vgj + (C0 - 1) * U
            return Vgj_prime
        
        elif self.voidFractionCorrel == 'GEramp':
            if self.voidFraction[-1][i] <= 0.65:
                return 1.1
            else:
                return 1 + 0.1* (1 - self.voidFraction[-1][i]) / 0.35
    
    def getC0(self, i):
        rho_g = self.rhoG[-1][i]
        rho_l = self.rhoL[-1][i]

        if rho_g == 0:
            return 0
        if rho_l == 0:
            return 0
        return 1.2 - 0.2*np.sqrt(rho_g / rho_l)
    
    def getHfg(self, i):#P in Pa

        vapor = IAPWS97(P = self.P[-1][i]*(10**(-6)), x = 1)
        liquid = IAPWS97(P = self.P[-1][i]*(10**(-6)), x = 0)
        return (vapor.h - liquid.h)
    
    def getFrictionFactor(self, i):
        rho = self.rho[-1][i]
        U = self.U[-1][i]
        P = self.P[-1][i]

        if self.frfaccorel == 'base': #Validated
            return 0.000033
        Re = self.getReynoldsNumber(i)
        if self.frfaccorel == 'blasius': #Not Validated
            return 0.00003
            return 0.186 * Re**(-0.2)
        if self.frfaccorel == 'Churchill': #Not Validated
            Ra = 0.4 * (10**(-6)) #Roughness
            R = Ra / self.D_h
            frict=8*(((8.0/Re)**12)+((2.475*np.log(((7/Re)**0.9)+0.27*R))**16+(37530/Re)**16)**(-1.5))**(1/12)   
            return frict
        
    def getPhi2Phi(self, i):
        x_th = self.xTh[-1][i]
        rho_l = self.rhoL[-1][i]
        rho_g = self.rhoG[-1][i]
        rho = self.rho[-1][i]
        P = self.P[-1][i]
        epsilon = self.voidFraction[-1][i]

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
    
    def getVoidFraction(self, i): #x_th in m3/m3, rho_l in kg/m3, rho_g in kg/m3
        x_th = self.xTh[-1][i]
        rho_l = self.rhoL[-1][i]
        rho_g = self.rhoG[-1][i]

        if x_th == 0:
            return 0.0001
        elif x_th == 1:
            return 0.99
        else:
            return (x_th * rho_l)/(x_th * rho_l + (1 - x_th) * rho_g)
    
    def getReynoldsNumber(self, i):
        U = self.U[-1][i]
        rho = self.rho[-1][i]
        P = self.P[-1][i]

        m = IAPWS97(P = P*(10**(-6)), x = 0).mu
        return rho * abs(U) * self.D_h / m
    
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