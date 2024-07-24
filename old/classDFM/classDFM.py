import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt
from Th_plotting import plotter

class DFMclass():
    def __init__(self, nCells, uInlet, pOutlet, hInlet):
        
        self.nCells = nCells
        self.uInlet = uInlet
        self.pOutlet = pOutlet
        self.hInlet = hInlet

        #Geometry parameters
        self.height = 1.655 #m
        self.fuelRadius = 5.6*10**(-3) #External radius of the fuel m
        self.cladRadius = 6.52*10**(-3) #External radius of the clad m
        self.waterGap = 0.5*10**(-3) #Gap between the clad and the water m
        self.waterRadius =  self.cladRadius + self.waterGap #External radius of the water m

        self.cote = 0.0157
        self.Poro = 0.5655077285
        self.flowArea = self.cote ** 2 * self.Poro #m2
        self.DV = (self.height/self.nCells) * self.flowArea #Volume of the control volume m3
        self.D_h = self.flowArea / (np.pi*self.cladRadius) #Hydraulic diameter m2
        self.D_h = 0.0078395462 #Hydraulic diameter m2
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
        self.frfaccorel = 'base'
        self.P2Pcorel = 'base'
        self.numericalMethod = 'FVM'

        #Heating parameters
        self.Q = 500000000 #volumetric heat generation rate W/m3
        self.q__fluid = np.pi * self.fuelRadius**2 * self.Q #linear heat generation rate W/m
        self.q__ = self.q__fluid / self.flowArea #W/m3

        self.hlSat = []
        self.hgSat = []
     

    def setInitialFields(self): #ne change rien, return uniquement les premier champs
        U_initial = np.ones(self.nCells)*self.uInlet
        P_initial = np.ones(self.nCells)*self.pOutlet
        H_initial = np.ones(self.nCells)*self.hInlet
        epsilon_initial = np.array([i*self.epsilonTarget/self.nCells for i in range(self.nCells)])

        rho_initial = np.ones(self.nCells)
        rho_g_initial = np.ones(self.nCells)
        rho_l_initial = np.ones(self.nCells)
        areaMatrix = np.ones(self.nCells)
        areaMatrix_1_initial = np.ones(self.nCells)
        areaMatrix_2_initial = np.ones(self.nCells)
        Dhfg_initial = np.ones(self.nCells)
        x_th_initial = np.ones(self.nCells)
        f_initial = np.ones(self.nCells)
        Vgj_initial = np.ones(self.nCells)
        Vgj_prime_initial = np.ones(self.nCells)
        C0_initial = np.ones(self.nCells)

        

        for i in range(self.nCells):
            rho_l_initial[i], rho_g_initial[i], rho_initial[i] = self.getDensity(P_initial[i], epsilon_initial[i]) 
            Dhfg_initial[i] = self.getHfg(P_initial[i])
            f_initial[i] = self.getFrictionFactor(rho_initial[i], U_initial[i], P_initial[i]) 
            areaMatrix_1_initial[i] = self.getAreas(self.getPhi2Phi(epsilon_initial[i], rho_initial[i], rho_l_initial[i], rho_g_initial[i], x_th_initial[i], P_initial[i]), f_initial[i])
            areaMatrix_2_initial[i] = self.getAreas( -self.getPhi2Phi(epsilon_initial[i], rho_initial[i], rho_l_initial[i], rho_g_initial[i], x_th_initial[i], P_initial[i]), f_initial[i])
            areaMatrix[i] = self.flowArea
            Vgj_initial[i] = self.getVgj(rho_g_initial[i], rho_l_initial[i])
            C0_initial[i] = self.getC0(rho_g_initial[i], rho_l_initial[i])
            Vgj_prime_initial[i] = self.getVgj_prime(rho_g_initial[i], rho_l_initial[i], U_initial[i])
    
        return U_initial, P_initial, H_initial, epsilon_initial, rho_initial, rho_g_initial, rho_l_initial, areaMatrix, areaMatrix_1_initial, areaMatrix_2_initial, Dhfg_initial, x_th_initial, f_initial, Vgj_initial, Vgj_prime_initial, C0_initial
    
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
        DM1 = self.q__ * self.DV + DI + DI2
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
                bi = rho_old[i+1]*VAR_old[i-self.nCells]*areaMatrix_old_1[i+1])

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
                di =  self.q__ * self.DV + DI + DI2)

        #print(f'VAR_VFM_Class.A: {VAR_VFM_Class.A}, VAR_VFM_Class.D: {VAR_VFM_Class.D}')
        VAR = VAR_VFM_Class.resoudre_h()
        U, P, H = self.splitVar(VAR)

        return U, P, H
    
    def updateFields(self):

        epsilonUpdated = np.ones(self.nCells)
        rhoUpdated = np.ones(self.nCells)
        rho_gUpdated = np.ones(self.nCells)
        rho_lUpdated = np.ones(self.nCells)
        areaMatrix_1Updated = np.ones(self.nCells)
        areaMatrix_2Updated = np.ones(self.nCells)
        DhfgUpdated = np.ones(self.nCells)
        x_thUpdated = np.ones(self.nCells)
        fUpdated = np.ones(self.nCells)
        VgjUpdated = np.ones(self.nCells)
        Vgj_primeUpdated = np.ones(self.nCells)
        C0Updated = np.ones(self.nCells)

        for i in range(self.nCells):
            rho_lUpdated[i], rho_gUpdated[i], rhoUpdated[i], epsilonUpdated[i], x_thUpdated[i] = self.resolveParameters(i)
            DhfgUpdated[i] = self.getHfg(self.P[-1][i])
            fUpdated[i] = self.getFrictionFactor(rhoUpdated[i], self.U[-1][i], self.P[-1][i])
            areaMatrix_1Updated[i] = self.getAreas( self.getPhi2Phi(epsilonUpdated[i], rhoUpdated[i], rho_lUpdated[i], rho_gUpdated[i], x_thUpdated[i], self.P[-1][i]), fUpdated[i])
            areaMatrix_2Updated[i] = self.getAreas( -self.getPhi2Phi(epsilonUpdated[i], rhoUpdated[i], rho_lUpdated[i], rho_gUpdated[i], x_thUpdated[i], self.P[-1][i]), fUpdated[i])
            VgjUpdated[i] = self.getVgj(rho_gUpdated[i], rho_lUpdated[i])
            C0Updated[i] = self.getC0(rho_gUpdated[i], rho_lUpdated[i])
            Vgj_primeUpdated[i] = self.getVgj_prime(rho_gUpdated[i], rho_lUpdated[i], self.U[-1][i])

        print(f'x_thUpdated: {x_thUpdated}')

        self.voidFraction.append(epsilonUpdated)
        self.rho.append(rhoUpdated)
        self.rhoG.append(rho_gUpdated)
        self.rhoL.append(rho_lUpdated)
        self.areaMatrix_1.append(areaMatrix_1Updated)
        self.areaMatrix_2.append(areaMatrix_2Updated)
        self.Dhfg.append(DhfgUpdated)
        self.xTh.append(x_thUpdated)
        self.f.append(fUpdated)
        self.Vgj.append(VgjUpdated)
        self.VgjPrime.append(Vgj_primeUpdated)
        self.C0.append(C0Updated)

    def resolveParameters(self, i):
        #U, P ,h, epsilon, x_th, rho_l, rho_g, rho, areaMatrix, epsInnerIteration, maxInnerIteration) #U in m/s, P in Pa, h in J/kg, epsilon in m3/m3, rho_l in kg/m3, rho_g in kg/m3, rho in kg/m3, areaMatrix in m2
        x_th_new = self.getQuality(self.H[-1][i], self.P[-1][i])
        print(f'x_th_new: {x_th_new}')
        rho_l_new, rho_g_new, rho_new = self.getDensity(self.P[-1][i], self.voidFraction[-1][i])
        epsilon_new = self.getVoidFraction(x_th_new, rho_l_new, rho_g_new)
        rho_l_new, rho_g_new, rho_new = self.getDensity(self.P[-1][i], epsilon_new)
        
        return rho_l_new, rho_g_new, rho_new, epsilon_new, x_th_new
    
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

        Utemp, Ptemp, Htemp, epsilonTemp, rhoTemp, rhoGTemp, rhoLTemp, areaMatrixTemp, areaMatrix_1_temp, areaMatrix_2_temp, DhfgTemp, xThTemp, fTemp, VgjTemp, VgjPrimeTemp, C0Temp = self.setInitialFields()
        self.U, self.P, self.H, self.voidFraction, self.rho, self.rhoG, self.rhoL, self.areaMatrix, self.areaMatrix_1, self.areaMatrix_2, self.Dhfg, self.xTh, self.f, self.Vgj, self.VgjPrime, self.C0 = [Utemp], [Ptemp], [Htemp], [epsilonTemp], [rhoTemp], [rhoGTemp], [rhoLTemp], areaMatrixTemp, [areaMatrix_1_temp], [areaMatrix_2_temp], [DhfgTemp], [xThTemp], [fTemp], [VgjTemp], [VgjPrimeTemp], [C0Temp]

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

        self.voidFraction[-1] = self.voidFraction[-1] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.voidFraction[-2]
        self.rho[-1] = self.rho[-1] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.rho[-2]
        self.rhoG[-1] = self.rhoG[-1] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.rhoG[-2]
        self.rhoL[-1] = self.rhoL[-1] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.rhoL[-2]
        self.xTh[-1] = self.xTh[-1] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.xTh[-2]
        self.Vgj[-1] = self.Vgj[-1] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.Vgj[-2]
        self.C0[-1] = self.C0[-1] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.C0[-2]
        self.VgjPrime[-1] = self.VgjPrime[-1] * self.sousRelaxFactor + (1-self.sousRelaxFactor)*self.VgjPrime[-2]

    def getAreas(self, phi2phi, f): #phi2phi in m3/m3, f in m
        A_chap = self.flowArea +  (phi2phi/2) * ((f / self.D_h) + (self.K_loss / self.Dz)) * self.DV
        return A_chap
    
    def getQuality(self, H, P):
        hl, hg = self.getPhasesEnthalpy(P)
        print(f'hl: {hl}, hg: {hg}, H: {H}')
        if H*0.001 < hl:
            return 0
        elif H*0.001 > hg:
            return 1
        elif H*0.001 <= hg and H*0.001 >= hl:
            return (H*0.001 - hl)/(hg - hl)
        
    def getPhasesEnthalpy(self, P):
        vapor = IAPWS97(P = P*(10**(-6)), x = 1)
        liquid = IAPWS97(P = P*(10**(-6)), x = 0)
        return liquid.h, vapor.h
    
    def getDensity(self, P, x): #P in Pa, x in m3/m3
        vapor = IAPWS97(P = P*(10**(-6)), x = 1)
        liquid = IAPWS97(P = P*(10**(-6)), x = 0)
        rho_g = vapor.rho
        rho_l = liquid.rho

        rho = rho_l * (1 - x) + rho_g * x
        return rho_l, rho_g, rho
    
    def getVgj(self, rho_g, rho_l):
        if rho_g == 0:
            return 0
        if rho_l == 0:
            return 0
        return 0.188 * np.sqrt(((rho_l - rho_g) * self.g * self.D_h ) / rho_g )
    
    def getVgj_prime(self, rho_g, rho_l, U):
        C0 = self.getC0(rho_g, rho_l)
        Vgj = self.getVgj(rho_g, rho_l)
        Vgj_prime = Vgj + (C0 - 1) * U
        return Vgj_prime
    
    def getC0(self, rho_g, rho_l):
        if rho_g == 0:
            return 0
        if rho_l == 0:
            return 0
        return 1.2 - 0.2*np.sqrt(rho_g / rho_l)
    
    def getHfg(self, P):#P in Pa
        vapor = IAPWS97(P = P*(10**(-6)), x = 1)
        liquid = IAPWS97(P = P*(10**(-6)), x = 0)
        return (vapor.h - liquid.h)
    
    def getFrictionFactor(self, rho, U, P):
        if self.frfaccorel == 'base': #Validated
            return 0.0001
        Re = self.getReynoldsNumber(rho, U, P)
        if self.frfaccorel == 'blasius': #Not Validated
            return 0.186 * Re**(-0.2)
        if self.frfaccorel == 'Churchill': #Not Validated
            Ra = 0.4 * (10**(-6)) #Roughness
            R = Ra / self.D_h
            frict=8*(((8.0/Re)**12)+((2.475*np.log(((7/Re)**0.9)+0.27*R))**16+(37530/Re)**16)**(-1.5))**(1/12)   
            return frict
        
    def getPhi2Phi(self, epsilon, rho, rho_l, rho_g, x_th, P):
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
    
    def getVoidFraction(self, x_th, rho_l, rho_g): #x_th in m3/m3, rho_l in kg/m3, rho_g in kg/m3
        if x_th == 0:
            return 0.0001
        elif x_th == 1:
            return 0.99
        else:
            return (x_th * rho_l)/(x_th * rho_l + (1 - x_th) * rho_g)
    
    def getReynoldsNumber(self, rho, U, P):
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
            self.hlSat.append(self.getPhasesEnthalpy(self.P[-1][i])[0])
            self.hgSat.append(self.getPhasesEnthalpy(self.P[-1][i])[1])

    


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



#Boundary conditions
T_inlet = 602.75 #K
P_inlet = 14.9 * (10**6) #Pa
h_inlet = IAPWS97(T = T_inlet, P = P_inlet * 10**(-6)).h * 1000 #J/kg
u_inlet = 4.68292412 #m/s
P_outlet =  14739394.95 #Pa
nCells = 10

DFM1 = DFMclass(nCells, u_inlet, P_outlet, h_inlet)
DFM1.resolveDFM()
DFM1.createBoundaryEnthalpy()
h = DFM1.H[-1]/1000
eps = DFM1.voidFraction[-1]
z = DFM1.zList

T = []
for i in range(nCells):
    T.append(IAPWS97(P = DFM1.P[-1][i] * 10**(-6), h = h[i], x = 0).T)

#print(f'U: {DFM1.U}, P: {DFM1.P}, H: {DFM1.H}, epsilon: {DFM1.voidFraction} rho: {DFM1.rho}, rhoG: {DFM1.rhoG}, rhoL: {DFM1.rhoL}, xTh: {DFM1.xTh}, f: {DFM1.f}, Vgj: {DFM1.Vgj}, Vgj_prime: {DFM1.VgjPrime}, C0: {DFM1.C0}')
print(f'T: {T} \n U: {list(DFM1.U[-1])}, \n P: {list(DFM1.P[-1])}, \n H: {list(DFM1.H[-1])}, \n epsilon: {list(DFM1.voidFraction[-1])}, \n rho: {list(DFM1.rho[-1])}, \n rhoG: {list(DFM1.rhoG[-1])}, \n rhoL: {list(DFM1.rhoL[-1])}, \n xTh: {list(DFM1.xTh[-1])}, \n f: {list(DFM1.f[-1])}, \n Vgj: {list(DFM1.Vgj[-1])}, \n Vgj_prime: {list(DFM1.VgjPrime[-1])}, \n C0: {list(DFM1.C0[-1])}')
      
print(f'z: {list(z)}') 

fig1 = plt.figure()
plt.plot(z, h)
plt.plot(z,  DFM1.hlSat, label="hlSat")
plt.plot(z, DFM1.hgSat, label="hgSat")
plt.xlabel('Height (m)')
plt.ylabel('Enthalpy (J/kg)')
plt.title('Enthalpy distribution')
plt.legend()

fig2 = plt.figure()
plt.plot(DFM1.I,DFM1.rhoGResiduals, label="Residuals rhoG")
plt.plot(DFM1.I, DFM1.rhoLResiduals, label="Residuals rhoL")
plt.xlabel('Iteration')
plt.ylabel('Residuals')
plt.legend()

fig3 = plt.figure()
plt.plot(DFM1.I, DFM1.rhoResiduals)
plt.xlabel('Iteration')
plt.ylabel('rho Residuals')
plt.legend()

fig4 = plt.figure()
plt.plot(DFM1.I, DFM1.EPSresiduals)
plt.xlabel('Iteration')
plt.ylabel('Epsilon Residuals')
plt.legend()

fig5 = plt.figure()
plt.plot(DFM1.I, DFM1.xThResiduals)
plt.xlabel('Iteration')
plt.ylabel('xTh Residuals')
plt.legend()

fig6 = plt.figure()
plt.plot(z, eps)
plt.xlabel('Height (m)')
plt.ylabel('Void fraction')
plt.title('Void fraction distribution')
plt.legend()

fig7 = plt.figure()
plt.plot(z, DFM1.rho[-1])
plt.xlabel('Height (m)')
plt.ylabel('Density (kg/m3)')
plt.title('Density distribution')
plt.legend()

fig8 = plt.figure()
plt.plot(z, DFM1.rhoG[-1])
plt.xlabel('Height (m)')
plt.ylabel('Density gas (kg/m3)')
plt.title('Density gas distribution')
plt.legend()

fig9 = plt.figure()
plt.plot(z, DFM1.rhoL[-1])
plt.xlabel('Height (m)')
plt.ylabel('Density liquid (kg/m3)')
plt.title('Density liquid distribution')
plt.legend()

fig10 = plt.figure()
plt.plot(z, DFM1.U[-1])
plt.xlabel('Height (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity distribution')
plt.legend()

fig11 = plt.figure()
plt.plot(z, DFM1.P[-1])
plt.xlabel('Height (m)')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure distribution')
plt.legend()



plt.show()