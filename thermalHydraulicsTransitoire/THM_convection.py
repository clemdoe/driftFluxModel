# This file contains the implementation of the drift flux model for the THM prototype
# This class models the dynamic and steady-state behavior of a two-phase flow system in different geometries (square and cylindrical channels). 
# It discretizes the channel geometry and sets up the necessary fields for fluid flow, pressure, enthalpy, and void fraction in a thermal-hydraulic model. 
# It also includes methods for transient and steady-state simulations using various numerical techniques. The class supports setting up fission power, initializing 
# the flow fields, creating systems of equations for both steady and transient analysis, and solving them using finite volume method (FVM). Visualization tools 
# are provided to track residuals and convergence during iterative solving.

# Authors : Clement Huet
# Date : 2021-06-01
# Python3 class part of THM_prototype
# uses : - Drift flux model for two-phase flow
#        - Finite volume method for discretization of the conservation equations
#        - IAPWS97 for water properties
#        - THM_linalg for numerical resolution, it include a newton simple iteration method, a Gauss Siedel method, a BiCGStab method and a BiCG method
#        - THM_waterProp for water properties, the calculation of the void fraction, the calculation of the friction factor and the two-phase mltp depend on correlations
#        - THM_plotting for plotting

import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt
from THM_linalg import FVM
from THM_linalg import numericalResolution
from THM_waterProp import statesVariables

class DFMclass():
    def __init__(self, canal_type, nCells, tInlet, qFlow, pOutlet, height, fuelRadius, cladRadius, cote,  numericalMethod, frfaccorel, P2P2corel, voidFractionCorrel, dt = 0, t_tot = 0):
        
        """
        Attributes:
        - nCells: Number of discretized cells.
        - pOutlet (Pa), tInlet (K): Inlet velocity, outlet pressure, and inlet enthalpy.
        - qFlow: Mass flow rate of the fluid (kg/s).
        - height (m), fuelRadius (m), cladRadius (m): Geometry of the channel (length, fuel, and clad radii).
        - cote: Channel width or distance, depending on the geometry.
        - canalType: Geometry type of the channel, either 'square' or 'cylindrical'.
        - numericalMethod: Chosen method for numerical resolution (e.g., Gauss-Seidel, FVM, BiGStab).
        - voidFractionCorrel, frfaccorel, P2Pcorel: Correlations used for void fraction and other flow properties.
        - dt, t_tot: Time-step and total simulation time for transient analysis.
        """

        """
        Methods:
        - __init__(...): Initializes the class with geometric, inlet, outlet, and other user-specified parameters. It also sets up physical constants and mesh properties.
        - set_Fission_Power(Q): Sets the fission power source term for the system.
        - get_Fission_Power(): Returns the source term (fission power) distribution along the channel.
        - setInitialFields(): Initializes the primary variables (velocity, pressure, enthalpy, void fraction) for steady-state simulation and updates flow properties.
        - createSystem(): Constructs the system of equations for solving the steady-state two-phase flow problem using the finite volume method.
        - createSystemTransient(): Sets up the system of equations for transient simulation.
        - calculateResiduals(): Calculates the residuals for velocity, pressure, and void fraction, and monitors the convergence.
        - testConvergence(k): Checks if the solution has converged based on residuals after iteration k.
        - residualsVisu(): Updates and visualizes residuals during the iterative solving process.
        - resolveDFM(): Main function that orchestrates the simulation by calling initializations, solving the system, and managing iterations and convergence criteria.
        - plotResults(): Plots the results of the simulation, including velocity, pressure, enthalpy, and void fraction profiles.
        - setInitialFieldsTransient(): Initializes the fields for transient simulation.
        - compute_T_surf(): Computes the surface temperature on the clad on the enthalpy profile.
        - sousRelaxation(): Implements under-relaxation for the solution update.
        - mergeVAR(): Merges the variables for the system of equations.
        - splitVAR(): Splits the variables for the system of equations.
        - createBoundaryEnthalpy(): Sets the boundary saturation lines for enthalpy.
        """ 

        self.nCells = nCells
        self.pOutlet = pOutlet
        self.tInlet = tInlet

        #calculate temporary hInlet
        pressureDrop = 186737 #Pa/m
        falsePInlet = pOutlet - height * pressureDrop
        self.hInlet = IAPWS97(T = self.tInlet, P = falsePInlet * 10**(-6)).h*1000 #J/kg
        print(f'hInlet: {self.hInlet}')

        #Geometry parameters
        self.height = height #m
        self.fuelRadius = fuelRadius #External radius of the fuel m
        self.cladRadius = cladRadius #External radius of the clad m
        self.cote = cote
        self.wall_dist = cote
        self.canalType = canal_type

        if self.canalType == 'square':
            self.flowArea = self.cote ** 2 - np.pi * self.cladRadius ** 2
        elif self.canalType == 'cylindrical':
            self.waterGap = self.cote -  self.cladRadius#Gap between the clad and the water m
            self.waterRadius =  self.cote #External radius of the water m
            self.flowArea = np.pi * self.waterRadius ** 2 - np.pi * self.cladRadius ** 2

        #calculate temporary uInlet
        self.qFlow = qFlow #kg/s
        self.rhoInlet = IAPWS97(T = self.tInlet, P = falsePInlet*10**(-6)).rho #kg/m3
        self.uInlet = self.qFlow / (self.flowArea * self.rhoInlet) #m/s
        print(f'uInlet: {self.uInlet}')

        self.DV = (self.height/self.nCells) * self.flowArea #Volume of the control volume m3
        if self.height == 2.155:
            ###########GENFOAM CALCULATION
            self.D_h = 7.83954616*10**(-3)
        else:
            self.D_h = 4 * self.flowArea / (self.cote**2 + np.pi * self.cladRadius**2) #Hydraulic diameter m2   
        
        self.Dz = self.height/self.nCells #Height of the control volume m
        self.z_mesh = np.linspace(0, self.height, self.nCells)
        self.epsilonTarget = 0.18
        self.K_loss = 0
        self.dx = self.height / self.nCells

        self.epsInnerIteration = 1e-3
        self.maxInnerIteration = 1000
        self.sousRelaxFactor = 0.2
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
        self.UResiduals = []
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

        #Transient parameters
        self.dt = dt
        if dt != 0:
            self.t_tot = t_tot
            self.timeList = np.arange(0, self.t_tot, self.dt)
            self.timeCount = 0

    
    def set_Fission_Power(self, Q):
        self.q__ = []
        for i in range(len(Q)):
            #self.q__.append(Q[i])
            self.q__.append((np.pi * self.fuelRadius**2 * Q[i]) / self.flowArea) #W/m3

    def get_Fission_Power(self):
        """
        function to retrieve a given source term from the axial profile used to model fission power distribution in the fuel rod
        """
        return self.q__
        
    def setInitialFields(self): #crée les fields et remplis la premiere colonne
        
        if self.dt == 0:
        
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

        else:
            if self.timeCount == 0:
                self.U = [self.velocityList[self.timeCount]]
                self.P = [self.pressureList[self.timeCount]]
                self.H = [self.enthalpyList[self.timeCount]]
                self.voidFraction = [self.voidFractionList[self.timeCount]]

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

                self.xThList[self.timeCount] = self.xTh[-1]
                self.rhoList[self.timeCount] = self.rho[-1]
                self.rhoGList[self.timeCount] = self.rhoG[-1]
                self.rhoLList[self.timeCount] = self.rhoL[-1]
                self.DhfgList[self.timeCount] = self.Dhfg[-1]
                self.fList[self.timeCount] = self.f[-1]
                self.areaMatrix_1List[self.timeCount] = self.areaMatrix_1[-1]
                self.areaMatrix_2List[self.timeCount] = self.areaMatrix_2[-1]
                self.areaMatrixList[self.timeCount] = self.areaMatrix
                self.VgjList[self.timeCount] = self.Vgj[-1]
                self.C0List[self.timeCount] = self.C0[-1]
                self.VgjPrimeList[self.timeCount] = self.VgjPrime[-1]
            
            if self.timeCount != 0:
                print(self.timeCount)
                self.U = [self.velocityList[self.timeCount]]
                self.P = [self.pressureList[self.timeCount]]
                self.H = [self.enthalpyList[self.timeCount]]
                self.voidFraction = [self.voidFractionList[self.timeCount]]
                self.xTh = [self.xThList[self.timeCount]]
                self.rhoL = [self.rhoLList[self.timeCount]]
                self.rhoG = [self.rhoGList[self.timeCount]]
                self.rho = [self.rhoList[self.timeCount]]
                self.Dhfg = [self.DhfgList[self.timeCount]]
                self.f = [self.fList[self.timeCount]]
                self.areaMatrix_1 = [self.areaMatrix_1List[self.timeCount]]
                self.areaMatrix_2 = [self.areaMatrix_2List[self.timeCount]]
                self.areaMatrix = self.areaMatrixList[self.timeCount]
                self.Vgj = [self.VgjList[self.timeCount]]
                self.C0 = [self.C0List[self.timeCount]]
                self.VgjPrime = [self.VgjPrimeList[self.timeCount]]

    
    def createSystemVelocityPressure(self):
        
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

        VAR_old = self.mergeVar(U_old,P_old)
        rho_old = self.mergeVar(rho_old, rho_old)
        rho_g_old = self.mergeVar(rho_g_old, rho_g_old)
        rho_l_old = self.mergeVar(rho_l_old, rho_l_old)
        epsilon_old = self.mergeVar(epsilon_old, epsilon_old)
        areaMatrix = self.mergeVar(areaMatrix, areaMatrix)
        areaMatrix_old_1 = self.mergeVar(areaMatrix_old_1, areaMatrix_old_1)
        areaMatrix_old_2 = self.mergeVar(areaMatrix_old_2, areaMatrix_old_2)
        V_gj_old = self.mergeVar(V_gj_old, V_gj_old)
        Vgj_prime = self.mergeVar(Vgj_prime, Vgj_prime)
        Dhfg = self.mergeVar(Dhfg, Dhfg)
        C0 = self.mergeVar(C0, C0)
        x_th_old = self.mergeVar(x_th_old, x_th_old)
        
        VAR_VFM_Class = FVM(A00 = 1, A01 = 0, Am0 = 0, Am1 = 1, D0 = self.uInlet, Dm1 = self.pOutlet, N_vol = 2*self.nCells, H = self.height)
        VAR_VFM_Class.boundaryFilling()
        for i in range(1, 2*self.nCells-1):
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
                ai = - areaMatrix[i],
                bi = areaMatrix[i+1],
                di = - ((rho_old[i+1]- rho_old[i])* self.g * self.DV / 2) + DI)
            
                VAR_VFM_Class.fillingOutsideBoundary(i, i-self.nCells,
                ai = - rho_old[i]*VAR_old[i-self.nCells]*areaMatrix_old_2[i],
                bi = rho_old[i+1]*VAR_old[i+1-self.nCells]*areaMatrix_old_1[i+1])

        self.FVM = VAR_VFM_Class

    def createSystemEnthalpy(self):

        U_old = self.U[-1]
        P_old = self.P[-1]
        H_old = self.H[-1]
        epsilon_old = self.voidFraction[-1]
        rho_old = self.rho[-1]
        rho_g_old = self.rhoG[-1]
        rho_l_old = self.rhoL[-1]
        areaMatrix = self.areaMatrix
        Dhfg = self.Dhfg[-1]
        x_th_old = self.xTh[-1]
        f = self.f[-1]
        V_gj_old = self.Vgj[-1]
        Vgj_prime = self.VgjPrime[-1]
        C0 = self.C0[-1]


        i = -1
        DI = (1/2) * (P_old[i]*areaMatrix[i] - P_old[i-1]*areaMatrix[i-1]) * ((U_old[i]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (U_old[i-1]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
        DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*areaMatrix[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*areaMatrix[i-1]/rho_old[i-1])
        DM1 = self.q__[i] * self.DV + DI + DI2
        VAR_VFM_Class = FVM(A00 = 1, A01 = 0, Am0 = - rho_old[-2] * U_old[-2] * areaMatrix[-2], Am1 = rho_old[-1] * U_old[-1] * areaMatrix[-1], D0 = self.hInlet, Dm1 = DM1, N_vol = self.nCells, H = self.height)
        VAR_VFM_Class.boundaryFilling()
        for i in range(1,self.nCells -1):
            #Inside the enthalpy submatrix
            DI = (1/2) * (P_old[i]*areaMatrix[i] - P_old[i-1]*areaMatrix[i-1]) * ((U_old[i]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (U_old[i-1]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
            DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*areaMatrix[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*areaMatrix[i-1]/rho_old[i-1])
            VAR_VFM_Class.set_ADi(i, ci =  - rho_old[i-1] * U_old[i-1] * areaMatrix[i-1],
                ai = rho_old[i] * U_old[i] * areaMatrix[i],
                bi = 0,
                di =  self.q__[i] * self.DV + DI + DI2)
        
        self.FVM = VAR_VFM_Class

    def createSystemTransient(self):
            
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
                di = ( self.rhoList[self.timeCount][i%self.nCells] *areaMatrix[i] - rho_old[i] *areaMatrix[i] ) * (self.dx / self.dt))
            elif i == self.nCells-1:
                VAR_VFM_Class.set_ADi(i, 
                ci = - rho_old[i-1]*areaMatrix[i-1],
                ai = rho_old[i]*areaMatrix[i],
                bi = 0,
                di =  ( self.rhoList[self.timeCount][i%self.nCells] * areaMatrix[i] - rho_old[i] *areaMatrix[i] ) * (self.dx / self.dt))

            #Inside the pressure submatrix
            elif i == self.nCells:
                DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * areaMatrix[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * areaMatrix[i] )/ ((1 - epsilon_old[i])*rho_old[i]) )     
                VAR_VFM_Class.set_ADi(self.nCells,
                ci = 0,
                ai = - areaMatrix[i],
                bi = areaMatrix[i+1],
                di = - ((rho_old[i+1]- rho_old[i])* self.g * self.DV / 2) + DI + (self.rhoList[self.timeCount][i%self.nCells] * areaMatrix[i] * self.velocityList[self.timeCount][i%self.nCells] * (self.dx / self.dt)))
            
                VAR_VFM_Class.fillingOutsideBoundary(i, i-self.nCells,
                ai = - rho_old[i]*VAR_old[i-self.nCells]*areaMatrix_old_2[i] + rho_old[i]*areaMatrix[i]*(self.dx/self.dt),
                bi = rho_old[i+1]*VAR_old[i-self.nCells+1]*areaMatrix_old_1[i+1])

            elif i > self.nCells and i < 2*self.nCells-1:
                DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * areaMatrix[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * areaMatrix[i] )/ ((1 - epsilon_old[i])*rho_old[i]) )     
                VAR_VFM_Class.set_ADi(i, ci = 0,
                ai = - areaMatrix[i],
                bi = areaMatrix[i+1],
                di = - ((rho_old[i+1]- rho_old[i])* self.g * self.DV / 2) + DI + (self.rhoList[self.timeCount][i%self.nCells] * areaMatrix[i] * self.velocityList[self.timeCount][i%self.nCells] * (self.dx / self.dt)))
            
                VAR_VFM_Class.fillingOutsideBoundary(i, i-self.nCells,
                ai = - rho_old[i]*VAR_old[i-self.nCells]*areaMatrix_old_2[i] + rho_old[i]*areaMatrix[i]*(self.dx/self.dt),
                bi = rho_old[i+1]*VAR_old[i+1-self.nCells]*areaMatrix_old_1[i+1])

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
                VAR_VFM_Class.set_ADi(i, ci =  - rho_old[i-1] * VAR_old[i-1-2*self.nCells] * areaMatrix[i-1] + rho_old[i]*areaMatrix[i]*self.dx/self.dt,
                ai = rho_old[i] * VAR_old[i-2*self.nCells] * areaMatrix[i],
                bi = 0,
                di =  self.q__[i%self.nCells] * self.DV + DI + DI2 + (self.rhoList[self.timeCount][i%self.nCells]* self.enthalpyList[self.timeCount][i%self.nCells] *areaMatrix[i] * self.dx / self.dt))

        self.FVM = VAR_VFM_Class


    def calculateResiduals(self):#change les residus
        self.EPSresiduals.append(np.linalg.norm(self.voidFraction[-1] - self.voidFraction[-2]))
        self.rhoResiduals.append(np.linalg.norm((self.rho[-1] - self.rho[-2])/self.rho[-1]))
        #self.UResiduals.append(np.linalg.norm((self.U[-1] - self.U[-2])/self.U[-1]))
        #self.rhoGResiduals.append(np.linalg.norm(self.rhoG[-1] - self.rhoG[-2]))
        #self.rhoLResiduals.append(np.linalg.norm(self.rhoL[-1] - self.rhoL[-2]))
        self.xThResiduals.append(np.linalg.norm(self.xTh[-1] - self.xTh[-2]))

    def testConvergence(self, k):#change rien et return un boolean
        print(f'Convergence test: EPS: {self.EPSresiduals[-1]}, rho: {self.rhoResiduals[-1]}, xTh: {self.xThResiduals[-1]}')
        #print(f'Convergence test: rho: {self.rhoResiduals[-1]}, U: {self.UResiduals[-1]}')
        if self.EPSresiduals[-1] < 1e-3 and self.xThResiduals[-1] < 1e-3 and self.rhoResiduals[-1] < 1e-3:
        #if self.rhoResiduals[-1] < 1e-2 and self.UResiduals[-1] < 1e-2:
            print(f'Convergence reached at iteration number: {k}')
            return True
        else:
            return False

    def setInitialFieldsTransient(self): #crée les fields et remplis la premiere colonne
        
        self.velocityList = np.zeros((len(self.timeList), self.nCells))
        self.pressureList = np.zeros((len(self.timeList), self.nCells))
        self.enthalpyList = np.zeros((len(self.timeList), self.nCells))
        self.voidFractionList = np.zeros((len(self.timeList), self.nCells))
        self.rhoList = np.zeros((len(self.timeList), self.nCells))
        self.rhoGList = np.zeros((len(self.timeList), self.nCells))
        self.rhoLList = np.zeros((len(self.timeList), self.nCells))
        self.rhoList = np.zeros((len(self.timeList), self.nCells))
        self.rhoGList = np.zeros((len(self.timeList), self.nCells))
        self.rhoLList = np.zeros((len(self.timeList), self.nCells))
        self.xThList = np.zeros((len(self.timeList), self.nCells))
        self.DhfgList = np.zeros((len(self.timeList), self.nCells))
        self.fList = np.zeros((len(self.timeList), self.nCells))
        self.areaMatrix_1List = np.zeros((len(self.timeList), self.nCells))
        self.areaMatrix_2List = np.zeros((len(self.timeList), self.nCells))
        self.areaMatrixList = np.zeros((len(self.timeList), self.nCells))
        self.VgjList = np.zeros((len(self.timeList), self.nCells))
        self.C0List = np.zeros((len(self.timeList), self.nCells))
        self.VgjPrimeList = np.zeros((len(self.timeList), self.nCells))

        self.velocityList[:,0] = self.uInlet
        self.pressureList[:,-1] = self.pOutlet
        self.enthalpyList[:,0] = self.hInlet
        self.velocityList[0,:] = self.uInlet
        self.pressureList[0,:] = self.pOutlet
        self.enthalpyList[0,:] = self.hInlet

    def residualsVisu(self):
        # Mise à jour des données de la ligne
        self.line.set_xdata(self.I)
        self.line.set_ydata(self.rhoResiduals)

        # Ajuste les limites des axes si nécessaire
        self.ax.relim()         # Recalcule les limites des données
        self.ax.autoscale_view()  # Réajuste la vue automatiquement

        # Dessine les modifications
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def updateInlet(self):
        #Update uInlet
        print(f'T: {self.tInlet}, P: {self.P[-1][0]*10**(-6)}')
        self.rhoInlet = IAPWS97(T = self.tInlet, P = self.P[-1][0]*10**(-6)).rho #kg/m3
        print(f'New inlet density: {self.rhoInlet}, self.qFlow: {self.qFlow}, self.flowArea: {self.flowArea}')
        self.uInlet = self.qFlow / (self.flowArea * self.rhoInlet) #m/s
        print(f'New inlet velocity: {self.uInlet}')
        #Update hInlet
        self.hInlet = IAPWS97(T = self.tInlet, P = self.P[-1][0]*10**(-6)).h*1000 #J/kg

    
    def resolveDFM(self):

        if self.dt == 0:

            self.setInitialFields()
            # Active le mode interactif
            plt.ion()
            # Crée la figure et l'axe
            self.fig, self.ax = plt.subplots()
            # Initialisation de la ligne qui sera mise à jour
            self.line, = self.ax.plot(self.I, self.rhoResiduals, 'r-', marker='o')  # 'r-' pour une ligne rouge avec des marqueurs

    
            for k in range(self.maxOuterIteration):
                
                self.createSystemVelocityPressure()
                resolveSystem = numericalResolution(self.FVM,self.mergeVar(self.U[-1], self.P[-1]), self.epsInnerIteration, self.maxInnerIteration, self.numericalMethod)
                print(resolveSystem.x)
                Utemp, Ptemp = self.splitVar(resolveSystem.x)
                
                self.U.append(Utemp)
                self.P.append(Ptemp)

                self.updateInlet()
                
                self.createSystemEnthalpy()
                resolveSystem = numericalResolution(self.FVM, self.H[-1], self.epsInnerIteration, self.maxInnerIteration, self.numericalMethod)
                
                Htemp = resolveSystem.x

                self.H.append(Htemp)

                print(Utemp, Ptemp, Htemp)
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
                self.residualsVisu()

                convergence = self.testConvergence(k)

                self.updateInlet()

                if convergence == True:
                    break

                elif k == self.maxOuterIteration - 1:
                    print('Convergence not reached')
                    break

            plt.ioff()
            plt.show()

        elif self.dt != 0:

            self.setInitialFieldsTransient()
            # Active le mode interactif
            plt.ion()
            # Crée la figure et l'axe
            self.fig, self.ax = plt.subplots()
            # Initialisation de la ligne qui sera mise à jour
            self.line, = self.ax.plot(self.I, self.rhoResiduals, 'r-', marker='o')  # 'r-' pour une ligne rouge avec des marqueurs
            
            for t in range(0, len(self.timeList)-1):

                self.setInitialFields()

                for k in range(self.maxOuterIteration):
                    
                    self.createSystemTransient()
                    resolveSystem = numericalResolution(self.FVM,self.mergeVar(self.U[-1], self.P[-1], self.H[-1]), self.epsInnerIteration, self.maxInnerIteration, self.numericalMethod)
                    
                    Utemp, Ptemp, Htemp = self.splitVar(resolveSystem.x)
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
                    if self.I == []:
                        self.I.append(0)
                    else:
                        self.I.append(1+self.I[-1])
                    self.residualsVisu()

                    convergence = self.testConvergence(k)

                    self.updateInlet()

                    if convergence == True:
                        self.timeCount += 1
                        self.velocityList[self.timeCount] = self.U[-1]
                        self.pressureList[self.timeCount] = self.P[-1]
                        self.enthalpyList[self.timeCount] = self.H[-1]
                        self.voidFractionList[self.timeCount] = self.voidFraction[-1]
                        self.xThList[self.timeCount] = self.xTh[-1]
                        self.rhoList[self.timeCount] = self.rho[-1]
                        self.rhoGList[self.timeCount] = self.rhoG[-1]
                        self.rhoLList[self.timeCount] = self.rhoL[-1]
                        self.DhfgList[self.timeCount] = self.Dhfg[-1]
                        self.fList[self.timeCount] = self.f[-1]
                        self.areaMatrix_1List[self.timeCount] = self.areaMatrix_1[-1]
                        self.areaMatrix_2List[self.timeCount] = self.areaMatrix_2[-1]
                        self.areaMatrixList[self.timeCount] = self.areaMatrix
                        self.VgjList[self.timeCount] = self.Vgj[-1]
                        self.C0List[self.timeCount] = self.C0[-1]
                        self.VgjPrimeList[self.timeCount] = self.VgjPrime[-1]
                        print(f'Convergence reached at iteration number: {k} for time step: {t}')
                        break

                    elif k == self.maxOuterIteration - 1:
                        raise ValueError('Convergence not reached')
                    
        
            plt.ioff()
            plt.show()
        
        
        
        self.T_water = np.zeros(self.nCells)
        for i in range(self.nCells):
            self.T_water[i] = IAPWS97(P=self.P[-1][i]*10**-6, h=self.H[-1][i]*10**-3).T

    def compute_T_surf(self):
        self.Pfin = self.P[-1]
        self.h_z = self.H[-1]
        self.T_surf = np.zeros(self.nCells)
        self.Hc = np.zeros(self.nCells)
        for i in range(self.nCells):
            #print(f'At axial slice = {i}, Pfin = {self.Pfin[i]}, h_z = {self.h_z[i]}')
            Pr_number = IAPWS97(P=self.Pfin[i]*10**-6, h=self.h_z[i]*10**-3).Liquid.Prandt
            Re_number = self.getReynoldsNumber(i)
            k_fluid = IAPWS97(P=self.Pfin[i]*10**-6, h=self.h_z[i]*10**-3).Liquid.k
            #print(f"At axial slice = {i}, computed Reynold # = {Re_number}, computed Prandt # = {Pr_number}, k_fluid = {k_fluid}")
            self.Hc[i] = (0.023)*(Pr_number)**0.4*(Re_number)**0.8*k_fluid/self.D_h
            #print(f'self.Hc[i]: {self.Hc[i]}, \n self.q__[i]: {self.q__[i]} ,\n 2*np.pi*self.cladRadius: {2*np.pi*self.cladRadius}')
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
    
    def mergeVar(self, U, P): #créer une liste a partir de 3 liste
        VAR = np.concatenate((U, P))
        return VAR
    
    def splitVar(self, VAR): #créer 3 liste a partir d'une liste
        U = VAR[:self.nCells]
        P = VAR[self.nCells:]
        return U, P
    
    def createBoundaryEnthalpy(self):
        for i in range(self.nCells):
            self.hlSat.append(self.getPhasesEnthalpy(i)[0])
            self.hgSat.append(self.getPhasesEnthalpy(i)[1]) 
 
    def getReynoldsNumber(self, i):
        return (self.U[-1][i] * self.D_h * self.rho[-1][i]) / IAPWS97(P=self.P[-1][i]*10**-6, x=0).Liquid.mu
     
