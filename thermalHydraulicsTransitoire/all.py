"""
This program simulates radial heat conduction in a nuclear fuel pin using the finite difference method (FDM).
The code models the temperature distribution across the fuel pin, considering the fuel, gap, and cladding regions.
technical documentation : "Revisiting the simplified thermo-hydraulics module THM: in DONJON5 code" - A. Hébert, March 2018
document available at http://merlin.polymtl.ca/downloads/thm.pdf
Author : R. Guasch, C. Huet

Key features:
- Discretization of the radial domain into finite difference mesh elements.
- Calculation of thermal conductivity for each region (fuel, gap, and cladding).
- Solver for the temperature distribution using a tridiagonal matrix approach.
- Boundary condition application and computation of center and effective temperatures.
- Optional extension to visualize the entire system, including the coolant channel.

This program follows standard finite difference approaches to solve the heat conduction equation in cylindrical coordinates, with assumptions of constant properties per region (fuel, gap, cladding), and fixed boundary conditions for outer surface temperature.
"""

import numpy as np

class FDM_HeatConductionInFuelPin:

    """
    Parameters:
    - `r_fuel` : Radius of the fuel pin (m).
    - `I_f`    : Number of mesh elements in the fuel.
    - `gap_r`  : Radius of the gap between fuel and cladding (m).
    - `clad_r` : Radius of the cladding (m).
    - `I_c`    : Number of mesh elements in the cladding.
    - `Qfiss`  : Fission power density in the fuel (W/m^3).
    - `kf`     : Thermal conductivity of the fuel (W/m/K).
    - `kc`     : Thermal conductivity of the cladding (W/m/K).
    - `Hgap`   : Heat transfer coefficient through the gap (W/m^2/K).
    - `z`      : Height corresponding to axial discretization (for potential coupling with axial convection models).
    - `T_surf` : Outer cladding surface temperature, obtained from convection models.

    Methods:
    - `compute_Area_meshes`: Calculates mesh boundaries and areas for each region.
    - `compute_radii`: Converts areas into radial positions for mesh centers and boundaries.
    - `solve_T_in_pin`: Solves the system of equations for the temperature distribution.
    - `compute_T_center`: Computes the fuel center temperature using established formulas.
    - `compute_T_eff`: Computes the effective fuel temperature using simplified correlations.
    - `initialize_ks`: Initializes the thermal conductivity values for each mesh region.
    - `set_ADi_cond`: Sets the tri-diagonal matrix entries for internal nodes.
    - `set_CL_cond`: Applies boundary conditions at the first and last nodes.
    - `extend_to_canal_visu`: Extends the mesh and temperature distribution for visualization, including coolant.
    - `initialise_plotting_mesh`: Prepares the radial mesh for plotting the temperature profile.
    """

    def __init__(self, r_fuel, I_f, gap_r, clad_r, I_c, Qfiss, kf, kc, Hgap, z, T_surf):
        # Physical prameters
        self.r_f = r_fuel # fuel pin radius in meters
        self.I_f = I_f # number of mesh elements in the fuel
        self.gap_r = gap_r # gap radius in meters, used to determine mesh elements for constant surface discretization
        self.clad_r = clad_r # clad radius in meters, used to determine mesh elements for constant surface discretization

        self.I_c = I_c # number of mesh elements in clad
        self.Qfiss = Qfiss # Fission power density in W/m^3
        self.Hg = Hgap # Heat transfer coefficient through gap W/m^2/K
        self.z = z # corresponding height in m corresponding to axial discretization used in FVM_ConvectionInCanal class
        self.T_surf = T_surf # Boundary condition outer clad surface temperature computed from FVM_ConvectionInCanal class 
        self.kf = kf
        self.kc = kc
        # compute relevant quantities to initialise object
        self.N_node = I_f + I_c +2
        self.A = np.eye(self.N_node)
        self.D = np.zeros(self.N_node)
        self.compute_Area_meshes()
        self.compute_radii()
        self.initialize_ks()

        self.initialise_plotting_mesh("m")
        self.physical_regions_bounds = [0, self.r_f, self.gap_r, self.clad_r]
        
        
    def initialize_ks(self):
        # this array is probaby not needed here as one might assume that kf and kc
        # are constant in fuel/clad. I wanted to keep and option to let them vary according to temperature at node as conductive properties might differ when high temperature gradients are present.
        self.k = np.zeros(self.N_node-1) # associate a k to the center of each mesh element
        for i in range(len(self.k)):
            if i <self.I_f:
                self.k[i]=self.kf
            elif i == self.I_f:
                self.k[i]=0 # in gap!
            elif i<self.N_node:
                self.k[i]=self.kc
        return


    def compute_Area_meshes(self):
        """
        building necessary meshes for solving the heat conduction equation on the constant area discretization 
        """
        self.A_mesh_bounds = []
        self.A_mesh_centers = []
        A_f = self.r_f**2/2
        A_gf =self.gap_r**2/2
        A_cgf = self.clad_r**2/2
        self.deltaA_f = A_f / self.I_f # base assumption is that delta_A is constant in each region --> delta A fuel = constant in fuel, delta A clad = constant in clad and 1 delta A gap.
        for i in range(self.I_f+1):
            self.A_mesh_bounds.append(i*self.deltaA_f)
        for i in range(self.I_f):
            self.A_mesh_centers.append(i*self.deltaA_f+self.deltaA_f/2)
    
        self.deltaA_g = A_gf-A_f
        self.A_mesh_bounds.append(self.A_mesh_bounds[-1]+self.deltaA_g)
        self.A_mesh_centers.append(self.A_mesh_centers[-1]+self.deltaA_f/2+self.deltaA_g/2) # last center in fuel + half of the fuel area step to get to the last fuel bound + half of the gap area step to get to the center of the gap
        self.deltaA_c = (A_cgf-A_gf)/self.I_c
        for i in range(self.I_c):
            self.A_mesh_bounds.append(self.A_mesh_bounds[-1]+self.deltaA_c)
        for i in range(self.I_c):
            if i==0:
                self.A_mesh_centers.append(self.A_mesh_centers[-1]+self.deltaA_c/2+self.deltaA_g/2)
            else:
                self.A_mesh_centers.append(self.A_mesh_centers[-1]+self.deltaA_c)
        self.A_mesh_centers = np.array(self.A_mesh_centers)
        self.A_mesh_bounds = np.array(self.A_mesh_bounds)
        self.A_calculation_mesh = np.zeros(self.N_node)
        for i in range(self.N_node):
            if i < self.I_f:
                self.A_calculation_mesh[i] = i*self.deltaA_f + self.deltaA_f/2
            elif i == self.I_f:
                self.A_calculation_mesh[i] = A_f
            elif i == self.I_f+1:
                self.A_calculation_mesh[i] = A_f + self.deltaA_g
            elif i > self.I_f+1:
                self.A_calculation_mesh[i] = A_gf + self.deltaA_c/2 + (i-(self.I_f+2))*self.deltaA_c

        return
    
    def compute_radii(self):
        self.radii_at_centers = np.sqrt(self.A_mesh_centers*2)
        self.radii_at_bounds = np.sqrt(self.A_mesh_bounds*2)
        return
    
    def get_Di_half(self,i):
        if i > self.I_f+1:
            i=i-1
            Di_half = 4*self.A_mesh_bounds[i+1]/((self.deltaA_c/self.k[i])+(self.deltaA_c/self.k[i+1]))
        else:
            Di_half = 4*self.A_mesh_bounds[i+1]/((self.deltaA_f/self.k[i])+(self.deltaA_f/self.k[i+1]))
        return Di_half
    
    def get_Ei_fuel(self):
        Ei_half = 4*self.A_mesh_bounds[self.I_f]*self.k[self.I_f-1]/self.deltaA_f
        return Ei_half
    
    def get_Ei_gap(self):
        Ei_half = 4*self.A_mesh_bounds[self.I_f+1]*self.k[self.I_f+1]/self.deltaA_c
        return Ei_half
    
    def get_Ei_clad(self):
        Ei_half = 4*self.A_mesh_bounds[-1]*self.k[-1]/self.deltaA_c
        return Ei_half
    
    def get_Fi_gap(self):
        Fi_half = 4*self.A_mesh_bounds[self.I_f+1]*self.k[self.I_f+1]/self.deltaA_c
        return Fi_half
    
    def get_Gi(self):
        return self.Hg*self.radii_at_centers[self.I_f]
    
    def set_ADi_cond(self, i, ci, ai, bi, di):
        # create lines for the tri-diagonal entries in
        self.A[i, i-1:i+2] = [ci, ai, bi]
        self.D[i] = di
        return
    
    def set_CL_cond(self, A0, Am1, D0, Dm1):
        # I_and_half is the index for the I+1/2 element of th mesh which corresponds to the last point in the fuel.
        # conditions aux limites
        # A0 = A[0], Am1 = A[-1], A moins 1, 
        # D0 = D[0], Dm1 = D[-1], D moins 1.
        self.A[0], self.A[-1] = A0, Am1
        self.D[0], self.D[-1] = D0, Dm1
        return
    
    def solve_T_in_pin(self):
        for row in self.A:
            line = "[  "
            for elem in row:
                line+=f"{elem:.3f}   "
            line += "  ]\n"
            #print(line)
        self.T_distrib = np.linalg.solve(self.A, self.D)
        self.compute_T_center()
        T_distrib_with_center = np.zeros(self.N_node+1)
        T_distrib_with_center[0] = self.T_center
        for i in range(1,self.N_node+1):
            T_distrib_with_center[i] = self.T_distrib[i-1]
        self.T_distrib = T_distrib_with_center

    def compute_T_center(self):
        # using equation (13) of "Revisiting the simplified thermo-hydraulics module THM: in DONJON5 code" - A. Hébert, March 2018 to compute T_(3/2).
        T_3_2 = (self.deltaA_f*self.k[0]*self.T_distrib[0]+self.deltaA_f*self.k[1]*self.T_distrib[1])/(self.deltaA_f*self.k[0]+self.deltaA_f*self.k[1])
        # using equation (46) of "Revisiting the simplified thermo-hydraulics module THM: in DONJON5 code" - A. Hébert, March 2018 to compute T_center.
        self.T_center = 2*self.T_distrib[0] - T_3_2
        return
    def compute_T_eff(self):
        # using equation (45) of "Revisiting the simplified thermo-hydraulics module THM: in DONJON5 code" - A. Hébert, March 2018 to compute T_eff
        # This corresponds to the simplified correlation, the so-called Rowlands formula :
        if len(self.T_distrib) == self.N_node:
            self.T_eff = (5/9)*self.T_distrib[self.I_f] + (4/9)*self.T_center
        elif len(self.T_distrib) == self.N_node+1:
            self.T_eff = (5/9)*self.T_distrib[self.I_f+1] + (4/9)*self.T_center
        return
    
    def initialise_plotting_mesh(self, unit):
        """
        unit = "m" or "mm"
        this only affects the visualization/plotting units. However all input quantities have to be in MKS units to be consistent with the solver. 
        """
        # building plot mesh in radial units taking the central temperature into account.
        self.plot_mesh = np.zeros(self.N_node+1)
        self.plot_mesh[0] = 0
        self.plotting_units = unit
        for i in range(1,self.N_node+1):
            if unit == "m":
                self.plot_mesh[i] = np.sqrt(2*self.A_calculation_mesh[i-1])
            elif unit == "mm":
                self.plot_mesh[i] = np.sqrt(2*self.A_calculation_mesh[i-1])*1e3
    
    def extend_to_canal_visu(self, rw, Tw):
        A_w = rw**2/2
        deltA_w = A_w - self.A_mesh_bounds[-1] 
        w_center = np.sqrt(2*(self.A_mesh_bounds[-1] + deltA_w/2)) 
        self.plot_mesh = np.append(self.plot_mesh,[w_center])
        self.physical_regions_bounds.append(rw)        
        self.T_distrib = np.append(self.T_distrib, [Tw])
        self.radii_at_bounds = np.append(self.radii_at_bounds, [rw])


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

        #user choice
        self.frfaccorel = frfaccorel
        self.P2Pcorel = P2P2corel
        self.numericalMethod = numericalMethod
        self.voidFractionCorrel = voidFractionCorrel
        self.voidFractionEquation = 'base'

        self.nCells = nCells
        self.pOutlet = pOutlet
        self.tInlet = tInlet

        #calculate temporary hInlet
        pressureDrop = 186737 #Pa/m
        falsePInlet = pOutlet - height * pressureDrop
        self.hInlet = IAPWS97(T = self.tInlet, P = falsePInlet * 10**(-6)).h*1000 #J/kg
        #print(f'hInlet: {self.hInlet}')

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
        #print(f'uInlet: {self.uInlet}')

        self.DV = (self.height/self.nCells) * self.flowArea #Volume of the control volume m3
        if self.height == 2.155:
            ###########GENFOAM CALCULATION
            self.D_h = 7.83954616*10**(-3)
        else:
            if self.canalType == 'square':
                self.D_h = 4 * self.flowArea / (2*self.cote + 2*np.pi * self.cladRadius)
            elif self.canalType == 'cylindrical':
                self.D_h = 4 * self.flowArea / (np.pi * self.waterRadius*2 + np.pi * self.cladRadius*2)
        
        self.Dz = self.height/self.nCells #Height of the control volume m
        self.z_mesh = np.linspace(0, self.height, self.nCells)
        self.epsilonTarget = 0.18
        self.K_loss = 0
        self.dx = self.height / self.nCells

        self.epsInnerIteration = 1e-3
        self.maxInnerIteration = 1000
        if self.numericalMethod == 'BiCGStab':
            self.sousRelaxFactor = 0.8
        else:
            self.sousRelaxFactor = 1
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


        self.hlSat = []
        self.hgSat = []

        #Transient parameters
        self.dt = dt
        if dt != 0:
            self.t_tot = t_tot
            self.timeList = np.arange(0, self.t_tot, self.dt)
            self.timeCount = 0

    
    def set_Fission_Power(self, Q):
        print(f'Fission power fluid')
        self.q__ = []
        for i in range(len(Q)):
            #self.q__.append(Q[i])
            self.q__.append((np.pi * self.fuelRadius**2 * Q[i]) / self.flowArea) #W/m3
            
            #print((np.pi * self.fuelRadius**2 * Q[i]) / self.flowArea)

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

            updateVariables = statesVariables(self.U[-1], self.P[-1], self.H[-1], self.voidFraction[-1], self.D_h, self.flowArea, self.DV, self.voidFractionCorrel, self.frfaccorel, self.P2Pcorel, self.Dz)
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

                updateVariables = statesVariables(self.U[-1], self.P[-1], self.H[-1], self.voidFraction[-1], self.D_h, self.flowArea, self.DV, self.voidFractionCorrel, self.frfaccorel, self.P2Pcorel, self.Dz)
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
        print(f'Convergence test, residuals: epsilon: {self.EPSresiduals[-1]}, rho: {self.rhoResiduals[-1]}, xTh: {self.xThResiduals[-1]}')
        #print(f'Convergence test: rho: {self.rhoResiduals[-1]}, U: {self.UResiduals[-1]}')
        if self.EPSresiduals[-1] < 1e-3 and self.xThResiduals[-1] < 1e-3 and self.rhoResiduals[-1] < 1e-3:
        #if self.rhoResiduals[-1] < 1e-2 and self.UResiduals[-1] < 1e-2:
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
        self.rhoInlet = IAPWS97(T = self.tInlet, P = self.P[-1][0]*10**(-6)).rho #kg/m3
        #print(f'New inlet density: {self.rhoInlet}, self.qFlow: {self.qFlow}, self.flowArea: {self.flowArea}')
        self.uInlet = self.qFlow / (self.flowArea * self.rhoInlet) #m/s
        #print(f'New inlet velocity: {self.uInlet}')
        #Update hInlet
        self.hInlet = IAPWS97(T = self.tInlet, P = self.P[-1][0]*10**(-6)).h*1000 #J/kg

    
    def resolveDFM(self):

        if self.dt == 0:

            self.setInitialFields()
            # Active le mode interactif
            #plt.ion()
            # Crée la figure et l'axe
            #self.fig, self.ax = plt.subplots()
            # Initialisation de la ligne qui sera mise à jour
            #self.line, = self.ax.plot(self.I, self.rhoResiduals, 'r-', marker='o')  # 'r-' pour une ligne rouge avec des marqueurs

    
            for k in range(self.maxOuterIteration):
                
                self.createSystemVelocityPressure()
                resolveSystem = numericalResolution(self.FVM,self.mergeVar(self.U[-1], self.P[-1]), self.epsInnerIteration, self.maxInnerIteration, self.numericalMethod)
                Utemp, Ptemp = self.splitVar(resolveSystem.x)
                
                self.U.append(Utemp)
                self.P.append(Ptemp)

                self.updateInlet()
                
                self.createSystemEnthalpy()
                resolveSystem = numericalResolution(self.FVM, self.H[-1], self.epsInnerIteration, self.maxInnerIteration, self.numericalMethod)
                
                Htemp = resolveSystem.x

                self.H.append(Htemp)

                updateVariables = statesVariables(self.U[-1], self.P[-1], self.H[-1], self.voidFraction[-1], self.D_h, self.flowArea, self.DV, self.voidFractionCorrel, self.frfaccorel, self.P2Pcorel, self.Dz)
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
                #self.residualsVisu()

                convergence = self.testConvergence(k)

                self.updateInlet()

                if convergence == True:
                    print(f'Convergence reached at iteration number: {k}')
                    break

                elif k == self.maxOuterIteration - 1:
                    raise ValueError('Convergence not reached in the resolution of the drift flux model, not enough iterations. k = ', k)


            #print(f'U: {self.U[-1]}, P: {self.P[-1]}, H: {self.H[-1]}')

            #plt.ioff()
            #plt.show()

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

                    updateVariables = statesVariables(self.U[-1], self.P[-1], self.H[-1], self.voidFraction[-1], self.D_h, self.flowArea, self.DV, self.voidFractionCorrel, self.frfaccorel, self.P2Pcorel, self.Dz)
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
     
# THM_DONJON_parser class
# Used to post treat THM_DONJON data and compare it to THM_prototype results
# Authors : Clement Huet, Raphael Guasch
import numpy as np
class THM_DONJON_parser:
    def __init__(self, path, t0, time_step, t_end, nz, h_mesh):
        self.path = path
        self.data = None
        self.lines = self.load_data()
        
        # Axial mesh information
        self.nz = nz
        print(f"self.nz is {self.nz}")
        self.h_mesh = h_mesh
        self.dz = self.h_mesh/self.nz
        

        # Time mesh information
        self.begin_time = t0
        self.dt = time_step
        self.end_time = t_end
        if self.dt == 0:
            self.mumber_time_steps = 1
        else:
            self.mumber_time_steps = int((self.end_time-self.begin_time) / self.dt)
   
        self.list_tables = self.parse_lines()
        self.global_table = self.create_global()
        self.TCOMB, self.TSURF, self.DCOOL, self.TCOOL, self.PCOOL, self.HCOOL, self.QFUEL, self.QCOOL, self.VOID, self.QUAL, self.SLIP, self.FLOW_REGIME = self.create_values()

    def load_data(self):
        with open(self.path, 'r') as data_file:
            lines=data_file.readlines()
        return lines


    def parse_lines(self):
        #Parse the lines from the input file to create a list of arrays
        #Each sub-array contains the time at the 0th index and the values (list) of fields (values) at the 1st index
        t=0
        i=0
        table_time_val=[]
        while i<len(self.lines):
            if " ________________________________________________________________________________________________________________________________________________________________" in self.lines[i]:
                #print(f"Itération trouvée a la ligne i = {i}")
                #print(f'Le pas de temps vaut t = {t} s')
                table_time_val.append([t,self.lines[i+4:i+self.nz+4]])
                t+=1
                i+=self.nz
            else:
                i+=1
        return table_time_val


    def parse_table(self, table_t):
        # takes the index of the list created by parse_lines as input
        # returns the associated time step and processed array
        t = table_t[0]
        table_temp = table_t[1]
        tableau=[]
        print(len(table_temp))
        for i in range(len(table_temp)):
            print(i,table_temp[i])
            table_temp[i] = table_temp[i].strip()
            table_temp[i] = table_temp[i].replace(" ", "")
            table_temp[i] = table_temp[i].replace("|", ",")
            table_temp[i] = table_temp[i].split(',')
            table_temp[i] = table_temp[i][2:-1]
            print(i,table_temp[i])
            for j in range(len(table_temp[i])):
                table_temp[i][j]=float(table_temp[i][j])
            tableau.append(table_temp[i])
            print(i,table_temp[i])
        return t, tableau
    

    def create_global(self):
        # create the global array of results which concatenates all the arrays for
        # each time step : data needed to create the values with the create_values function
        big_table = []
        for table in self.list_tables:
            t,tableau = self.parse_table(table)
            big_table.append(tableau)

        return big_table


    def create_values(self):
        # create arrays with the values of the fields for each time step
        # Lines are the time steps, columns are the vertical mesh elements
        # creates the list of time steps and the list of vertical steps
        if self.dt != 0:
            T=np.arange(0,self.end_time,self.dt)
        else:
            T=np.array([1])
        list_z=np.arange(0,self.h_mesh,self.dz)
        time = len(self.global_table)
        TCOMB, TSURF, DCOOL, TCOOL, PCOOL, HCOOL, QFUEL, QCOOL, VOID, QUAL, SLIP, FLOW_REGIME=np.zeros((time, self.nz)),np.zeros((time, self.nz)),np.zeros((time, self.nz)),np.zeros((time, self.nz)),np.zeros((time, self.nz)),np.zeros((time, self.nz)),np.zeros((time, self.nz)),np.zeros((time, self.nz)),np.zeros((time, self.nz)),np.zeros((time, self.nz)),np.zeros((time, self.nz)),np.zeros((time, self.nz))
        for i in range(time):
            for j in range(self.nz):
                TCOMB[i][j]=self.global_table[i][j][0]
                TSURF[i][j]=self.global_table[i][j][1]
                DCOOL[i][j]=self.global_table[i][j][2]
                TCOOL[i][j]=self.global_table[i][j][3]
                PCOOL[i][j]=self.global_table[i][j][4]
                HCOOL[i][j]=self.global_table[i][j][5]
                QFUEL[i][j]=self.global_table[i][j][6]
                QCOOL[i][j]=self.global_table[i][j][7]
                VOID[i][j]=self.global_table[i][j][8]
                QUAL[i][j]=self.global_table[i][j][9]
                SLIP[i][j]=self.global_table[i][j][10]
                FLOW_REGIME=self.global_table[i][j][11]
        return TCOMB, TSURF, DCOOL, TCOOL, PCOOL, HCOOL, QFUEL, QCOOL, VOID, QUAL, SLIP, FLOW_REGIME





    def recup_val_tcst(val, t):
        t=float(t)
        T=list(np.arange(0,end_time,dt))
        list_z=list(np.arange(0,h_mesh,dz))
        return list_z, val[T.index(t)]

    def recup_val_xcst(val, x):
        x=float(x)
        list_z=list(np.arange(0,h_mesh,dz))
        T=list(np.arange(0,end_time,dt))
        return T,val[:,list_z.index(x)]
    

#This class implements several numerical methods for solving linear systems of the form Ax = b. 
#It supports different solvers and preconditioning techniques for improving convergence, and it is designed to handle large systems.
# The file is used in the BWR/driftFluxModel/thermalHydraulicsTransitoire/THM_convection.py file
#Authors: Clément Huet
#Date: 2024-09-16
#uses : - newton method
#       - Biconjhugate gradient stabilized method
#       - Gauss Siedel method
#       - Biconjugate gradient method
#       - matrix inversion method
#       - preconditionner method ILU and SPAI
#       - scipy.sparse library 


import numpy as np
from iapws import IAPWS97


class numericalResolution():

    """"
    Attributes:
    - `A`: Matrix A of the system Ax = b.
    - `b`: Right-hand side vector of the system Ax = b.
    - `x0`: Initial guess for the solution vector.
    - `tol`: Tolerance for convergence criteria.
    - `maxIter`: Maximum number of iterations allowed for iterative solvers.
    - `numericalMethod`: String indicating the chosen numerical method ('FVM', 'BiCStab', 'GaussSiedel', 'BiCG').

    Methods:
    - `resolve`: Chooses the appropriate solver based on the `numericalMethod` attribute.
    - `resolveInversion`: Directly solves the system using matrix inversion with preconditioning (LU factorization).
    - `resolveGaussSiedel`: Implements the Gauss-Seidel iterative solver with preconditioning.
    - `resolveBiConjugateGradient`: Implements the BiConjugate Gradient method (BiCG) with preconditioning.
    - `resolveBiCGStab`: Implements the BiCGStab method, an improved version of BiCG with stabilization for better convergence.
    - `preconditionner`: Constructs the preconditioner matrix using either ILU (Incomplete LU) or SPAI (Sparse Approximate Inverse) methods.
    - `scalBIC`: Computes the scalar product of two vectors using the conjugate transpose of the first vector.
    """

    """
    
    Numerical Methods:
    1. `FVM`: Solves the system using matrix inversion with preconditioning.
    2. `GaussSiedel`: Applies the Gauss-Seidel iterative solver.
    3. `BiCG`: Uses the BiConjugate Gradient method for solving non-symmetric or indefinite matrices.
    4. `BiCStab`: Applies the BiCGStab (BiConjugate Gradient Stabilized) method to ensure faster and more stable convergence.

    Preconditioning:
    - `preconditionner`: A function to improve convergence by applying either ILU or SPAI techniques to approximate the inverse of matrix A.
    - ILU: Incomplete LU factorization (used by default).
    - SPAI: Sparse Approximate Inverse (optional for special cases).

    Usage:
    - Create an instance of `numericalResolution` by providing the system's matrix `A`, vector `b`, initial guess `x0`, tolerance `tol`, maximum iterations `maxIter`, and the desired numerical method.
    - Call the `resolve` method to solve the system using the chosen solver.
    - Preconditioning is automatically applied to improve convergence for iterative methods.
    """

    def __init__(self, obj, x0, tol, maxIter, numericalMethod):
        self.A = obj.A
        self.b = obj.D
        self.x0 = x0
        self.tol = tol
        self.maxIter = maxIter
        self.numericalMethod = numericalMethod
        self.n = self.A.shape[0]

        self.resolve()

    def resolve(self):
        #print(f'Inside resolve, Numerical method: {self.numericalMethod}')

        if self.numericalMethod == 'FVM':
            self.x = self.resolveInversion()
        elif self.numericalMethod == 'BiCGStab':
            self.x = self.resolveBiCGStab()
        elif self.numericalMethod == 'GaussSiedel':
            self.x = self.resolveGaussSiedel()
        elif self.numericalMethod == 'BiCG':
            self.x = self.resolveBiConjugateGradient()
        else:
            raise ValueError('Numerical method not recognized')

    def resolveInversion(self):
        print(f'Inside resolveInversion')
        L, U = self.preconditionner(self.A)
        M = np.dot(L,U)
        VAR = np.linalg.solve(self.A,self.b)
        return VAR

    def resolveGaussSiedel(self):
        L, U = self.preconditionner(self.A)
        M = L @ U
        MStar = np.linalg.inv(M)

        A = np.dot(MStar,self.A)
        b = np.dot(MStar, self.b)

        x = self.x0
        n = len(x)
        err0 = 0
        Ax = np.zeros(n)
        for i in range(n):
            Ax[i] = np.dot(A[i, :], x)

        for i in range(n):
            err = b[i] - Ax[i]
            err0 += err**2
        err0 = np.sqrt(err0)

        for m in range(1,1000):
            esum = 0
            for i in range(n):
                x_old = x[i]
                sum = 0
                for j in range(n):
                    if j != i:
                        sum += A[i, j] * x[j]
                x[i] = (b[i] - sum) / A[i, i]
                esum += (x[i] - x_old)**2
            
            erout = np.sqrt(esum)
            if np.sqrt(erout) <= self.tol:
                break

        return x
    
    def preconditionner(self, A):
        m = 50 
        ILU = True
        SPAI = False

        if SPAI == True:
            """Perform m step of the SPAI iteration."""
            from scipy.sparse import identity
            from scipy.sparse import diags
            from scipy.sparse.linalg import onenormest
            from scipy.sparse import csr_array

            A = csr_array(A)
            ident = identity(n, format='csr')
            alpha = 2 / onenormest(A @ A.T)
            M = alpha * A
                
            for index in range(m):
                C = A @ M
                G = ident - C
                AG = A @ G
                trace = (G.T @ AG).diagonal().sum()
                alpha = trace / np.linalg.norm(AG.data)**2
                M = M + alpha * G
            #print(f'M inside preconditionner ILU: {M.todense()}')
            return M.todense()
        
        if ILU == True:
            #Initialize L and U as copies of A
            L = np.eye(self.n)
            U = np.copy(A)

            #Perform ILU factorization
            for i in range(1,self.n):
                for k in range(i):
                    if U[k,k] != 0:
                        L[i,k] = U[i,k] / U[k,k]
                        U[i,k:] = U[i,k:] - L[i,k] * U[k,k:]
            #print(f'M inside preconditionner SPAI: {np.dot(L,U)}')

            return L,U

    def resolveBiConjugateGradient(self):

        L,U = self.preconditionner(self.A)
        M = L @ U
        #print(f'M: {M}, \n A: {self.FVM.A}, \n M-1 * A: {np.dot(np.linalg.inv(M), self.FVM.A)}')
        self.condNUMBERB = np.linalg.cond(self.A)
        self.condNUMBER = np.linalg.cond(np.dot(np.linalg.inv(M), self.A))
        #print(f'CondNumber : self.condNUMBER: {self.condNUMBER}')
        #print(f'CondNumberOld : self.condNUMBERB: {self.condNUMBERB}')
        
        MStar = np.linalg.inv(M)
        AStar = np.transpose(self.A)
        x0 = self.x0
        r0 = self.b - np.dot(self.A,x0)
        r0Star = np.transpose(self.b) - np.dot(np.transpose(x0),AStar)
        p0 = np.dot(MStar,r0)
        p0Star = np.dot(r0Star, MStar)
        x0Star = np.transpose(x0)
        for k in range(100000):
            alpha = np.dot(np.dot(r0Star,MStar), r0) / np.dot(p0Star,np.dot(self.A, p0))
            alphaBar = np.conjugate(alpha)
            x = x0 + alpha * p0
            xStar  = x0Star + alphaBar * p0Star
            r = r0 - alpha * np.dot(self.A, p0)
            rStar = r0Star - alphaBar * np.dot(p0Star, AStar)
            if np.linalg.norm(r) < self.tol:
                break
            if k == 99999:
                raise ValueError('BiConjugateGradient did not converge')
            beta = np.dot(np.dot(rStar, MStar),r) / np.dot(np.dot(r0Star, MStar), r0)
            betaBar = np.conjugate(beta)
            p = np.dot(MStar,r) + beta * p0
            pStar = np.dot(rStar, MStar) + betaBar * p0Star

            r0 = r
            r0Star = rStar
            x0 = x
            x0Star = xStar
            p0 = p
            p0Star = pStar

        return x

    def scalBIC(self, a, b):
        return np.dot(np.transpose(a), b)
    
    def resolveBiCGStab(self):

        K1, K2 = self.preconditionner(self.A)
        K1Star = np.linalg.inv(self.A)
        K2Star = K1Star.copy()
        x0 = self.x0
        r0 = self.b - np.dot(self.A,self.x0)
        r0Star = r0
        rho0 = self.scalBIC(r0Star, r0)
        p0 = r0

        for k in range(100000):
            y = K2Star @ K1Star @ p0
            v = self.A @ y
            alpha = rho0 / self.scalBIC(r0Star, v)
            h = x0 + alpha * y
            s = r0 - alpha * v
            r = self.A @ h - self.b
            if np.linalg.norm(r) < self.tol:
                x = h
                break
            z = K2Star @ K1Star @ s
            t = self.A @ z
            omega = self.scalBIC(K1Star @ t, K1Star @ s) / self.scalBIC(K1Star @ t, K1Star @ t)
            x = h + omega * z
            r = s - omega * t
            res = self.A @ x - self.b
            if np.linalg.norm(res) < self.tol:
                break
            rho = self.scalBIC(r0Star, r)
            beta = (rho / rho0) * (alpha / omega)
            p = r + beta * (p0 - omega * v)
            
            if k == 99999:
                raise ValueError('BiCGStab did not converge')
            rho0 = rho
            r0 = r
            x0 = x
            p0 = p

        #print(f'end of resolveBiCGStab with k = {k}')
        return x
 
    


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


#Used to run the THM prototype class and compare the results with a reference THM_DONJON case.
#Authors : Clement Huet, Raphael Guasch


from THM_conduction import FDM_HeatConductionInFuelPin as FDM_Fuel
from THM_convection import DFMclass
from THM_DONJON_parser import THM_DONJON_parser
import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook

class Version5_THM_prototype:
    def __init__(self, case_name, canal_type,
                 canal_radius, fuel_radius, gap_radius, clad_radius, fuel_rod_length, tInlet, pOutlet, qFlow, Qfiss,
                 k_fuel, H_gap, k_clad, I_z, I_f, I_c, plot_at_z, solveConduction,
                 dt, t_tot, frfaccorel = 'base', P2Pcorel = 'base', voidFractionCorrel = 'GEramp', numericalMethod= 'FVM'):
        """
        Main constructor for THM case, first set of parameters correspond to canal properties, second set to fuel/gap/clad properties
        The structure followed is : 
        In FVM_ConvectionInCanal class : use a finite volume method to solve heat convection in the canal, then use the Dittus-Boelter correlation to obtain the convective heat transfer coef 
        between the water and the fuel rod's outer surface. This allows to solve for the temperature at this outer surface. 
        Then in the FDM_HeatConductionInFuelPin class, solve for the heat conduction using MCFD method. Compute temperature at the center of the fuel rod.
        Options to plot results can be activated giving an array of z values at which the results should be plotted.
        """
        self.name = case_name
        # time atributes to prepare for transient simulations
        self.t0 = 0
        self.dt = dt
        self.t_end = t_tot

        # canal attributes

        self.r_w = canal_radius # outer canal radius (m) if type is cylindrical, if type = square rw is the radius of inscribed circle in the square canal, ie half the square's side.
        self.canal_type = canal_type # cylindrical or square, used to determine the cross sectional flow area in the canal and the hydraulic diameter
        self.Lf = fuel_rod_length # fuel rod length in m
        

        self.tInlet = tInlet
        self.qFlow = qFlow #  mass flux in kg/s, assumed to be constant along the axial profile.
        self.I_z = I_z # number of mesh elements on axial mesh
        self.rhoInlet = 1000
        self.pOutlet =  pOutlet #Pa
        self.uInlet = self.qFlow / self.rhoInlet #m/s

        self.Qfiss = Qfiss # amplitude of sine variation, or constant value if Qfiss_variation_type = "constant"

        self.r_f = fuel_radius # fuel pin radius in meters
        self.gap_r = gap_radius # gap radius in meters, used to determine mesh elements for constant surface discretization
        self.clad_r = clad_radius # clad radius in meters, used to determine mesh elements for constant surface discretization
        self.k_fuel = k_fuel # thermal conductivity coefficient in fuel W/m/K
        self.H_gap = H_gap # Heat transfer coefficient through gap W/m^2/K
        self.k_clad = k_clad # thermal conductivity coefficient in clad W/m/K
        self.I_f = I_f # number of mesh elements in the fuel
        self.I_c = I_c # number of mesh elements in clad

        self.frfaccorel = frfaccorel # friction factor correlation
        self.P2Pcorel = P2Pcorel # pressure drop correlation
        self.voidFractionCorrel = voidFractionCorrel # void fraction correlation
        self.numericalMethod = numericalMethod # numerical method used to solve the convection problem in the canal
        #Poro = 0.5655077285

        self.plot_results = plot_at_z
        self.solveConduction = solveConduction

        print(f"$$$---------- THM: prototype, case treated : {self.name}.")
        if self.dt == 0 :
            self.transient = False
            print("$$$---------- THM: prototype, steady state case.")

        # Prepare and solve 1D heat convection along the z direction in the canal.
        print("$$---------- Calling DFM class.")
        print(f"Setting up heat convection solution along the axial dimension. zmax = {self.Lf} m with {self.I_z} axial elements.")
        # Create an object of the class DFMclass
        print(f'self.I_z: {self.I_z}')
        print(f'self.qFlow: {self.qFlow}')
        print(f'self.pOutlet: {self.pOutlet}')
        print(f'self.Lf: {self.Lf}')
        print(f'self.r_f: {self.r_f}')
        print(f'self.clad_r: {self.clad_r}')
        print(f'self.r_w: {self.r_w}')
        print(f'self.Dz: {self.Lf/self.I_z}')
        print(f'self.dt: {self.dt}')
        print(f'Courant number: {self.uInlet*self.dt/(self.Lf/self.I_z)}')
        print(f"Numerical Method {numericalMethod}")
        self.convection_sol = DFMclass(self.canal_type, self.I_z, self.tInlet, self.qFlow, self.pOutlet, self.Lf, self.r_f, self.clad_r, self.r_w, self.numericalMethod, self.frfaccorel, self.P2Pcorel, self.voidFractionCorrel, dt = self.dt, t_tot = self.t_end)
        print(f'Hydraulic diameter: {self.convection_sol.D_h}')
        print(f'Velocity at inlet: {self.uInlet} m/s')
        # Set the fission power in the fuel rod
        self.convection_sol.set_Fission_Power(self.Qfiss)
        # Resolve the DFM
        self.convection_sol.resolveDFM()
        print(f'Pressure: {self.convection_sol.P[-1]} Pa')
        print(f'Enthalpy: {self.convection_sol.H[-1]} J/kg')
        print(f'Void fraction: {self.convection_sol.voidFraction[-1]}')
        print(f'Density: {self.convection_sol.rho[-1]} kg/m^3')
        print(f'DV: {self.convection_sol.DV}')
        if self.solveConduction:
            Tsurf = self.convection_sol.compute_T_surf()
            print(f'Temperature at the surface: {Tsurf} K')
            print(f'Temperature of water: {self.convection_sol.T_water} K')

        if self.solveConduction:
            # Prepare and solve 1D radial heat conduction in the fuel rod, given a Clad surface temperature as a bondary condition 
            self.SetupAndSolve_Conduction_at_all_z() # creates a list of Temperature distributions in the fuel rod given a surface temperature computed by solving the conection problem
            self.get_TFuel_rowlands() # compute and store in the T_eff_fuel attribute the effective fuel temperature given by the Rowlands formula
            self.get_Tfuel_surface() # store in the T_fuel_surface attribute the fuel surface temperature computed

            # extend to Twater : adding a mesh point corresponding to the middle of the canal in the plotting array, add rw to the bounds array and add Twater to the results array
            for index_z in range(len(self.convection_sol.z_mesh)):
                self.T_distributions_axial[index_z].extend_to_canal_visu(rw = self.convection_sol.wall_dist, Tw = self.convection_sol.T_water[index_z])
                
            if self.plot_results:
                for z_val in self.plot_results:
                    self.plot_Temperature_at_z(z_val)
    
    def set_transitoire(self, t_tot, Tini, dt):
        self.t_tot, self.dt = t_tot, dt           
        self.N_temps = round(self.t_tot / self.dt) # pas de temps (timesteps), il faut etre un nombre entier
        self.T = np.zeros((self.N_temps+1, self.N_vol)) # tableau 2D de temperature. 
        self.T[0] = Tini # Tini est une liste
        return


    def SetupAndSolve_Conduction_at_all_z(self, transient = False):
        self.T_distributions_axial = []
        for axial_plane_nb in range(self.convection_sol.nCells):
            z = self.convection_sol.z_mesh[axial_plane_nb]
            T_surf = self.convection_sol.T_surf[axial_plane_nb]
            Qfiss = self.convection_sol.get_Fission_Power()[axial_plane_nb]
            self.T_distributions_axial.append(self.run_Conduction_In_Fuel_at_z(z,Qfiss,T_surf, transient))

        return
    
    def run_Conduction_In_Fuel_at_z(self,z,Qfiss_z,T_surf_z, transient = False):
        print(f"$$---------- Setting up FDM_HeatConductionInFuelPin class for z = {z} m, Qfiss(z) = {Qfiss_z} W/m^3 and T_surf(z) = {T_surf_z} K")
        heat_conduction = FDM_Fuel(self.r_f, self.I_f, self.gap_r, self.clad_r, self.I_c, Qfiss_z, self.k_fuel, self.k_clad, self.H_gap, z, T_surf_z)
        if transient:
            print(f"Error: transient case not implemented yet")
        else:
            for i in range(1,heat_conduction.N_node-1):

                if i<heat_conduction.I_f-1: # setting Aij and Di values for nodes inside the fuel 
                    heat_conduction.set_ADi_cond(i, 
                                        -heat_conduction.get_Di_half(i-1), 
                                        heat_conduction.get_Di_half(i-1)+heat_conduction.get_Di_half(i), 
                                        -heat_conduction.get_Di_half(i), 
                                        heat_conduction.deltaA_f*heat_conduction.Qfiss)
                elif i==heat_conduction.I_f-1: # setting Aij and Di values for last fuel element
                    heat_conduction.set_ADi_cond(i,
                                    -heat_conduction.get_Di_half(i-1),
                                    (heat_conduction.get_Di_half(i-1)+heat_conduction.get_Ei_fuel()),
                                    -heat_conduction.get_Ei_fuel(),
                                    heat_conduction.deltaA_f*heat_conduction.Qfiss)
                elif i==heat_conduction.I_f: # setting Aij and Di values first fuel / gap interface
                    heat_conduction.set_ADi_cond(i, 
                                        -heat_conduction.get_Ei_fuel(), 
                                        heat_conduction.get_Ei_fuel()+heat_conduction.get_Gi(), 
                                        -heat_conduction.get_Gi(), 
                                        0)
                elif i == heat_conduction.I_f+1: # setting Aij and Di values second gap / clad interface
                    heat_conduction.set_ADi_cond(i, 
                                        -heat_conduction.get_Gi(), 
                                        heat_conduction.get_Fi_gap()+heat_conduction.get_Gi(), 
                                        -heat_conduction.get_Fi_gap(), 
                                        0)
                elif i == heat_conduction.I_f+2: # Treating the first clad element interface with the gap.
                    heat_conduction.set_ADi_cond(i,
                                                -heat_conduction.get_Ei_gap(),
                                                (heat_conduction.get_Di_half(i)+heat_conduction.get_Ei_gap()),
                                                -heat_conduction.get_Di_half(i),
                                                0)
                elif i>heat_conduction.I_f+2 : # setting Aij and Di for all elements in the clad, apart from the last one
                    heat_conduction.set_ADi_cond(i, 
                                        -heat_conduction.get_Di_half(i-1), 
                                        heat_conduction.get_Di_half(i-1)+heat_conduction.get_Di_half(i), 
                                        -heat_conduction.get_Di_half(i), 
                                        0)
            A0,Am1 = np.zeros(heat_conduction.N_node), np.zeros(heat_conduction.N_node) 
            A0[:2] = [heat_conduction.get_Di_half(0), -heat_conduction.get_Di_half(0)]
            Am1[-2:] = [-heat_conduction.get_Di_half(heat_conduction.N_node-2), heat_conduction.get_Di_half(heat_conduction.N_node-2)+heat_conduction.get_Ei_clad()]
            D0 = heat_conduction.deltaA_f*heat_conduction.Qfiss
            Dm1 = heat_conduction.get_Ei_clad()*heat_conduction.T_surf
            heat_conduction.set_CL_cond(A0, Am1, D0, Dm1)
            print(f"$---------- Solving for T(r) using the Finite Difference Method, at z = {z}.")
            heat_conduction.solve_T_in_pin()
        return heat_conduction
    
    def get_TFuel_rowlands(self):
        self.T_eff_fuel = np.zeros(self.convection_sol.nCells)
        for i in range(len(self.T_distributions_axial)):
            self.T_distributions_axial[i].compute_T_eff()
            T_eff_z = self.T_distributions_axial[i].T_eff
            self.T_eff_fuel[i] = T_eff_z
        return
    
    def get_Tfuel_surface(self):
        self.T_fuel_surface = np.zeros(self.convection_sol.nCells)
        for i in range(len(self.T_distributions_axial)):
            if len(self.T_distributions_axial[i].T_distrib) == self.T_distributions_axial[i].N_node:
                T_surf_fuel_z = self.T_distributions_axial[i].T_distrib[self.I_f]
            else:
                T_surf_fuel_z = self.T_distributions_axial[i].T_distrib[self.I_f+1]
            self.T_fuel_surface[i] = T_surf_fuel_z
        return

    def plot_Temperature_at_z(self, z_val):
        print(f"$$---------- Plotting Temperature distribution in rod + canal z = {z_val} m")

        print(f"z_val is {z_val}")
        print(f"z_mesh is {self.convection_sol.z_mesh}")
        if z_val in self.convection_sol.z_mesh:
            plane_index = int(np.where(self.convection_sol.z_mesh==z_val)[0][0])
            Temperature_distrib_to_plot = self.T_distributions_axial[plane_index].T_distrib
            plotting_mesh = self.T_distributions_axial[plane_index].plot_mesh
            radii_at_bounds = self.T_distributions_axial[plane_index].radii_at_bounds
            physical_regions_bounds = self.T_distributions_axial[plane_index].physical_regions_bounds
            plotting_units = self.T_distributions_axial[plane_index].plotting_units
            Tsurf = self.convection_sol.T_surf[plane_index]
            Twater = self.convection_sol.T_water[plane_index]
            Tcenter = self.T_distributions_axial[plane_index].T_center
        else: # Interpolate between nearest z values to obtain Temperature distribution at a given z.
            second_plane_index = np.where(self.convection_sol.z_mesh>z_val)[0][0]
            first_plane_index = second_plane_index-1
            plane_index = (first_plane_index+second_plane_index)/2
            print(f"plane index used is {plane_index}")
            plotting_mesh = self.T_distributions_axial[first_plane_index].plot_mesh
            radii_at_bounds = self.T_distributions_axial[first_plane_index].radii_at_bounds
            physical_regions_bounds = self.T_distributions_axial[first_plane_index].physical_regions_bounds
            Temperature_distrib_to_plot = self.T_distributions_axial[first_plane_index].T_distrib+(z_val-self.convection_sol.z_mesh[first_plane_index])*(self.T_distributions_axial[second_plane_index].T_distrib-self.T_distributions_axial[first_plane_index].T_distrib)/(self.convection_sol.z_mesh[second_plane_index]-self.convection_sol.z_mesh[first_plane_index])
            plotting_units = self.T_distributions_axial[first_plane_index].plotting_units
            Tsurf = self.convection_sol.T_surf[first_plane_index] + (z_val-self.convection_sol.z_mesh[first_plane_index])*(self.convection_sol.T_surf[second_plane_index]-self.convection_sol.T_surf[first_plane_index])/(self.convection_sol.z_mesh[second_plane_index]-self.convection_sol.z_mesh[first_plane_index])
            Twater = self.convection_sol.T_water[first_plane_index] + (z_val-self.convection_sol.z_mesh[first_plane_index])*(self.convection_sol.T_water[second_plane_index]-self.convection_sol.T_water[first_plane_index])/(self.convection_sol.z_mesh[second_plane_index]-self.convection_sol.z_mesh[first_plane_index])
            Tcenter = self.T_distributions_axial[first_plane_index].T_center + (z_val-self.convection_sol.z_mesh[first_plane_index])*(self.T_distributions_axial[second_plane_index].T_center-self.T_distributions_axial[first_plane_index].T_center)/(self.convection_sol.z_mesh[second_plane_index]-self.convection_sol.z_mesh[first_plane_index])

        
        
        if (isinstance(plane_index, int)):
            plane_index_print = plane_index
        else:
            plane_index_print = str(plane_index).split(".")[0]+str(plane_index).split(".")[1]
        print(f"at z = {z_val}, temp distrib is = {Temperature_distrib_to_plot}")
        print(f'z_axis is {plotting_mesh}')
        colors = ["lime", "bisque", "chocolate", "royalblue"]
        labels = ["Fuel", "Gap", "Clad", "Water"]
        fig_filled, axs = plt.subplots()
        for i in range(len(physical_regions_bounds)-1):
            axs.fill_between(x=radii_at_bounds, y1=(Tcenter+50)*np.ones(len(radii_at_bounds)), y2=(Twater-50)*np.ones(len(radii_at_bounds)),where=(radii_at_bounds>=physical_regions_bounds[i])&(radii_at_bounds<=physical_regions_bounds[i+1]), color = colors[i], label = labels[i])
        axs.scatter(plotting_mesh, Temperature_distrib_to_plot, marker = "D", color="black",s=10, label="Radial temperature distribution in Fuel rod.")
        axs.legend(loc = "best")
        axs.grid()
        axs.set_xlabel(f"Radial position in {plotting_units}")
        axs.set_ylabel(f"Temperature in K")
        axs.set_title(f"Temperature distribution in fuel rod at z = {z_val}, {self.name}")
        fig_filled.savefig(f"{self.name}_Figure_plane{plane_index_print}_colors")
        plt.show()

    def plotColorMap(self):
        print("$$---------- Plotting colormap of temperature distribution in fuel rod.")
        fig, ax = plt.subplots()
        T = []
        for i in range(len(self.T_distributions_axial)):
            T.append(self.T_distributions_axial[i].T_distrib)
        R = self.T_distributions_axial[0].plot_mesh
        Z = self.convection_sol.z_mesh
        print(f'T: {T}')
        print(f'R: {R}')
        print(f'Z: {Z}')
        plt.xlabel('Rayon (mm)')
        plt.ylabel('Hauteur (m)')
        plt.title('Temperature (K) en fonction du rayon et de la hauteur')
        plt.pcolormesh(R, Z, T, cmap = 'plasma') 
        plt.colorbar()
        plt.show()


    def compare_with_THM_DONJON(self, THM_DONJON_path, visu_params):
        """
        Function used to compare the results obtained in the current instance with a reference DONJON/THM.
        """
        self.reference_case = THM_DONJON_parser(THM_DONJON_path, self.t0, self.dt, self.t_end, self.convection_sol.nCells, self.Lf)
        self.visu_TFuel = visu_params[0]
        self.visu_TFuelSurface = visu_params[1]
        self.visu_TWater = visu_params[2]
        self.visu_deltaTFuel = visu_params[3]
        self.visu_deltaTFuelSurface = visu_params[4]
        self.visu_deltaTWater = visu_params[5]
        print(f"$$---------- Comparing results with DONJON/THM: results from {THM_DONJON_path}.")
        #print(f"The fuel efective teperature from reference results is : {self.reference_case.TCOMB}")
        # Analyzing relaive errors on the fuel effective temperature, fuel surface temperature and water temperature
        if self.dt == 0:
            TEFF_FUEL = np.flip(self.reference_case.TCOMB[0])
            TSURF_FUEL = np.flip(self.reference_case.TSURF[0])
            TCOOL = np.flip(self.reference_case.TCOOL[0])
            # plotting the relative errors on properties in steady state case
            if self.visu_TFuel:
                print("$ --- Visualizing TFUEL")
                fig, ax = plt.subplots(dpi=200)
                ax.plot(self.convection_sol.z_mesh, self.T_eff_fuel, label="THM prototype")
                ax.plot(self.convection_sol.z_mesh, TEFF_FUEL, label="DONJON/THM")
                ax.set_xlabel("Axial position in m")
                ax.set_ylabel("Efective fuel temperature in K")
                ax.set_title("Effective fuel temperature comparison")        
                ax.legend(loc="best")
                fig.savefig(f"{self.name}_Teff_fuel_comparison_THM_DONJON")

            
            if self.visu_deltaTFuel:
                print("$ --- Visualizing error on TFUEL")
                fig, ax = plt.subplots(dpi=200)
                error_Tfuel = []
                for i in range(len(self.convection_sol.z_mesh)):
                    error_Tfuel.append((self.T_eff_fuel[i]-TEFF_FUEL[i])*100/TEFF_FUEL[i])
                ax.plot(self.convection_sol.z_mesh, error_Tfuel, label="THM prototype", marker="x", linestyle="--", markersize=5)
                ax.set_xlabel("Axial position in m")
                ax.set_ylabel("Relative error on effective fuel temperature (%)")
                ax.set_title("Error on effective fuel temperature (Prototype vs DONJON)")        
                ax.legend(loc="best")
                ax.grid()
                fig.savefig(f"{self.name}_error_Teff_fuel_THM_DONJON")
            
            
            if self.visu_TFuelSurface:
                print("$ --- Visualizing TSURF")
                fig, ax = plt.subplots(dpi=200)
                ax.plot(self.convection_sol.z_mesh, self.T_fuel_surface, label="THM prototype")
                ax.plot(self.convection_sol.z_mesh, TSURF_FUEL, label="DONJON/THM")
                ax.set_xlabel("Axial position in m")
                ax.set_ylabel("Fuel surface temperature in K")
                ax.set_title("Fuel surface temperature comparison")        
                ax.legend(loc="best")
                fig.savefig(f"{self.name}_Tsurf_comparison_THM_DONJON")
            
            if self.visu_deltaTFuelSurface:
                print("$ --- Visualizing error on TSURF")
                fig, ax = plt.subplots(dpi=200)
                error_Tsurf = []
                for i in range(len(self.convection_sol.z_mesh)):
                    error_Tsurf.append((self.T_fuel_surface[i]-TSURF_FUEL[i])*100/TSURF_FUEL[i])
                ax.plot(self.convection_sol.z_mesh, error_Tsurf, label="THM prototype", marker="x", linestyle="--", markersize=5)
                ax.set_xlabel("Axial position in m")
                ax.set_ylabel("Relative error on fuel surface temperature (%)")
                ax.set_title("Error on fuel surface temperature (Prototype vs DONJON)")        
                ax.legend(loc="best")
                fig.savefig(f"{self.name}_error_Tsurf_THM_DONJON")

            if self.visu_TWater:
                print("$ --- Visualizing TCOOL")
                fig, ax = plt.subplots(dpi=200)
                ax.plot(self.convection_sol.z_mesh, self.convection_sol.T_water, label="THM prototype")
                ax.plot(self.convection_sol.z_mesh, TCOOL, label="DONJON/THM")
                ax.set_xlabel("Axial position in m")
                ax.set_ylabel("Coolant temperature in K")
                ax.set_title("Coolant temperature comparison")        
                ax.legend(loc="best")
                fig.savefig(f"{self.name}_Twater_comparison_THM_DONJON")
            
            if self.visu_deltaTWater:
                print("$ --- Visualizing error on TCOOL")
                fig, ax = plt.subplots(dpi=200)
                error_Twater = []
                for i in range(len(self.convection_sol.z_mesh)):
                    error_Twater.append((self.convection_sol.T_water[i]-TCOOL[i])*100/TCOOL[i])
                ax.plot(self.convection_sol.z_mesh, error_Twater, label="THM prototype", marker="x", linestyle="--", markersize=5)
                ax.set_xlabel("Axial position in m")
                ax.set_ylabel("Relative error on coolant temperature (%)")
                ax.set_title("Error on coolant temperature (Prototype vs DONJON)")        
                ax.legend(loc="best")
                fig.savefig(f"{self.name}_error_Twater_THM_DONJON")

            plt.show()
        #print('TCOMB',TCOMB)
        return


    def get_TH_parameters(self):
        if self.solveConduction:
            print(f'Tfuel : {self.T_eff_fuel} K')
            print(f'Twater : {self.convection_sol.T_water} K')
            print(f'Water void fraction: {self.convection_sol.voidFraction[-1]} K')
            print(f'Water density: {self.convection_sol.rho[-1]} kg/m^3')
            
            return self.T_eff_fuel, self.convection_sol.T_water, self.convection_sol.rho[-1]
        
        else: 
            print(f'Twater : {self.convection_sol.T_water} K')
            print(f'Water void fraction: {self.convection_sol.voidFraction[-1]} K')
            print(f'Water density: {self.convection_sol.rho[-1]} kg/m^3')
            
            return [0], self.convection_sol.T_water, self.convection_sol.voidFraction[-1], self.convection_sol.rho[-1]

    def plotThermohydraulicParameters(self, visuParam):
        
        if visuParam[0]:
            fig1, ax1 = plt.subplots()
            if self.solveConduction:
                ax1.plot(self.convection_sol.z_mesh, self.T_eff_fuel, label="Fuel temperature")
            ax1.plot(self.convection_sol.z_mesh, self.convection_sol.T_water, label="Coolant temperature")
            ax1.set_xlabel("Axial position in m")
            ax1.set_ylabel("Temperature in K")
            ax1.set_title("Temperature distribution pincell")
            ax1.legend(loc="best")

        if visuParam[1]:
            fig2, ax2 = plt.subplots()
            ax2.plot(self.convection_sol.z_mesh, self.convection_sol.voidFraction[-1], label="Void fraction")
            ax2.set_xlabel("Axial position in m")
            ax2.set_ylabel("Void fraction")
            ax2.set_title("Void fraction distribution in coolant canal")
            ax2.legend(loc="best")

        if visuParam[2]:
            fig3, ax3 = plt.subplots()
            ax3.plot(self.convection_sol.z_mesh, self.convection_sol.rho[-1], label="Density")
            ax3.set_xlabel("Axial position in m")
            ax3.set_ylabel("Density in kg/m^3")
            ax3.set_title("Density distribution in coolant canal")
            ax3.legend(loc="best")

        if visuParam[3]:
            fig4, ax4 = plt.subplots() 
            ax4.plot(self.convection_sol.z_mesh, self.convection_sol.P[-1], label="Pressure")
            ax4.set_xlabel("Axial position in m")
            ax4.set_ylabel("Pressure in Pa")
            ax4.set_title("Pressure distribution in coolant canal")

        if visuParam[4]:
            fig5, ax5 = plt.subplots()
            ax5.plot(self.convection_sol.z_mesh, self.convection_sol.U[-1], label="Enthalpy")
            ax5.set_xlabel("Axial position in m")
            ax5.set_ylabel("Velocity in m/s")
            ax5.set_title("Velocity distribution in coolant canal")

        plt.show()
        return
    

class plotting:
    def __init__(self, caseList):
        self.caseList = caseList

    def writeResults(self, filename):
    # Create a new Excel workbook
        temperature_cases, void_fraction_cases, density_cases, pressure, velocity, enthalpy_cases = [self.caseList[0].convection_sol.z_mesh],[self.caseList[0].convection_sol.z_mesh],[self.caseList[0].convection_sol.z_mesh],[self.caseList[0].convection_sol.z_mesh],[self.caseList[0].convection_sol.z_mesh],[self.caseList[0].convection_sol.z_mesh]
        for case in self.caseList:
            temperature_cases.append(list(case.convection_sol.T_water))
            void_fraction_cases.append(list(case.convection_sol.voidFraction[-1]))
            density_cases.append(list(case.convection_sol.rho[-1]))
            pressure.append(list(case.convection_sol.P[-1]))
            velocity.append(list(case.convection_sol.U[-1]))
            enthalpy_cases.append(list(case.convection_sol.H[-1]))
            parameters = [['waterRadius', 'fuelRadius','gapRadius','cladRadius','height','pOutlet', 'tInlet', 'u_inlet', 'qFlow', 'Iz1', 'qFiss', 'Dz', 'frfaccorel', 'P2Pcorel', 'voidFractionCorrel', 'numericalMethod']]
            for case in self.caseList:
                parameters.append([case.r_w, case.r_f, case.gap_r, case.clad_r, case.Lf, case.pOutlet, case.tInlet, case.uInlet, case.qFlow, case.I_z, case.Qfiss, case.convection_sol.DV, case.frfaccorel, case.P2Pcorel, case.voidFractionCorrel, case.numericalMethod])

        print(len(temperature_cases))

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            #Write parameters for each case to a sheet
            parameters_df = pd.DataFrame(parameters).T
            parameters_df.columns = ['Parameter'] + [f"Case {i}" for i in range(len(self.caseList))]
            parameters_df.to_excel(writer, sheet_name='Parameters', index=False)

            # Write temperature data to a sheet
            temperature_df = pd.DataFrame(temperature_cases).T
            temperature_df.columns = ['Axial position'] + [f"Temperature (K) Case {i}" for i in range(1,len(temperature_cases))]
            temperature_df.to_excel(writer, sheet_name='Temperature', index=False)

            # Write void fraction data to a sheet
            void_fraction_df = pd.DataFrame(void_fraction_cases).T
            void_fraction_df.columns =  ['Axial position']+[f"Void Fraction Case {i}" for i in range(1,len(void_fraction_cases))]
            void_fraction_df.to_excel(writer, sheet_name='Void Fraction', index=False)

            # Write density data to a sheet
            density_df = pd.DataFrame(density_cases).T
            density_df.columns =  ['Axial position']+[f"Density (kg/m3) Case {i}" for i in range(1,len(density_cases))]
            density_df.to_excel(writer, sheet_name='Density', index=False)

            # Write pressure data to a sheet
            pressure_df = pd.DataFrame(pressure).T
            pressure_df.columns = ['Axial position']+[f"Pressure (Pa) Case {i}" for i in range(1,len(pressure))]
            pressure_df.to_excel(writer, sheet_name='Pressure', index=False)

            # Write velocity data to a sheet
            velocity_df = pd.DataFrame(velocity).T
            velocity_df.columns =  ['Axial position']+[f"Velocity (m/s) Case {i}" for i in range(1,len(velocity))]
            velocity_df.to_excel(writer, sheet_name='Velocity', index=False)

            # Write enthalpy data to a sheet
            enthalpy_df = pd.DataFrame(enthalpy_cases).T
            enthalpy_df.columns = ['Axial position']+[f"Enthalpy (kg/m3) Case {i+1}" for i in range(1,len(enthalpy_cases))]
            enthalpy_df.to_excel(writer, sheet_name='Enthalpy', index=False)
            
    def compute_error(self, GenFoamPathCase, compParam, genFoamVolumeFraction):

        # Read the Excel file
        df = pd.read_excel(GenFoamPathCase)

        # Create empty lists for each column
        columns = df.columns.tolist()
        data = [[] for _ in columns]

        # Iterate over each row and append values to the corresponding list
        for index, row in df.iterrows():
            for i, col in enumerate(columns):
                data[i].append(row[col])

        for i in range(len(data[7])):
            data[7][i] = (1/(1-genFoamVolumeFraction)) * data[7][i]

        genfoamCASE = [data[0], data[3], data[7], data[3], data[1], data[5]]
        print(genfoamCASE[2])

        Tw_error, voidFraction_error, pressure_error, velocity_error, = [], [], [], []
        
        for i in range(len(self.caseList)):
            Tw_error.append([])
            voidFraction_error.append([])
            pressure_error.append([])
            velocity_error.append([])
            for j in range(len(self.caseList[i].convection_sol.z_mesh)):
                jGF = int(j*len(genfoamCASE[0])/len(self.caseList[i].convection_sol.z_mesh))
                Tw_error[i].append(100*abs(self.caseList[i].convection_sol.T_water[j] - genfoamCASE[1][jGF])/genfoamCASE[1][jGF])
                voidFraction_error[i].append(abs(self.caseList[i].convection_sol.voidFraction[-1][j] - genfoamCASE[2][jGF]))
                pressure_error[i].append(100*abs(self.caseList[i].convection_sol.P[-1][j] - genfoamCASE[4][jGF])/genfoamCASE[4][jGF])
                velocity_error[i].append(100*abs(self.caseList[i].convection_sol.U[-1][j] - genfoamCASE[5][jGF])/genfoamCASE[5][jGF])

        if compParam == 'numericalMethod':
            fig1, ax1 = plt.subplots()
            for i in range(len(self.caseList)):
                ax1.plot(self.caseList[0].convection_sol.z_mesh, Tw_error[i], label=self.caseList[i].numericalMethod)
            ax1.set_xlabel("Axial position in m")
            ax1.set_ylabel("Error (%)")
            ax1.set_title("Error in temperature distribution")
            ax1.legend(loc="best")

            fig2, ax2 = plt.subplots()
            for i in range(len(self.caseList)):
                ax2.plot(self.caseList[0].convection_sol.z_mesh, voidFraction_error[i], label=self.caseList[i].numericalMethod)
            ax2.set_xlabel("Axial position in m")
            ax2.set_ylabel("Error (%)")
            ax2.set_title("Error in void fraction distribution")
            ax2.legend(loc="best")

            fig3, ax3 = plt.subplots()
            for i in range(len(self.caseList)):
                ax3.plot(self.caseList[0].convection_sol.z_mesh, pressure_error[i], label=self.caseList[i].numericalMethod)
            ax3.set_xlabel("Axial position in m")
            ax3.set_ylabel("Error (%)")
            ax3.set_title("Error in pressure distribution")
            ax3.legend(loc="best")

            fig4, ax4 = plt.subplots()
            for i in range(len(self.caseList)):
                ax4.plot(self.caseList[0].convection_sol.z_mesh, velocity_error[i], label=self.caseList[i].numericalMethod)
            ax4.set_xlabel("Axial position in m")
            ax4.set_ylabel("Error (%)")
            ax4.set_title("Error in velocity distribution")
            ax4.legend(loc="best")

            plt.show()

        elif compParam == 'voidFractionCorrel':
            fig1, ax1 = plt.subplots()
            for i in range(len(self.caseList)):
                ax1.plot(self.caseList[0].convection_sol.z_mesh, Tw_error[i], label=self.caseList[i].voidFractionCorrel)
            ax1.set_xlabel("Axial position in m")
            ax1.set_ylabel("Error (%)")
            ax1.set_title("Error in temperature distribution")
            ax1.legend(loc="best")

            fig2, ax2 = plt.subplots()
            for i in range(len(self.caseList)):
                ax2.plot(self.caseList[0].convection_sol.z_mesh, voidFraction_error[i], label=self.caseList[i].voidFractionCorrel)
            ax2.set_xlabel("Axial position in m")
            ax2.set_ylabel("Error (%)")
            ax2.set_title("Error in void fraction distribution")
            ax2.legend(loc="best")

            fig3, ax3 = plt.subplots()
            for i in range(len(self.caseList)):
                ax3.plot(self.caseList[0].convection_sol.z_mesh, pressure_error[i], label=self.caseList[i].voidFractionCorrel)
            ax3.set_xlabel("Axial position in m")
            ax3.set_ylabel("Error (%)")
            ax3.set_title("Error in pressure distribution")
            ax3.legend(loc="best")

            fig4, ax4 = plt.subplots()
            for i in range(len(self.caseList)):
                ax4.plot(self.caseList[0].convection_sol.z_mesh, velocity_error[i], label=self.caseList[i].voidFractionCorrel)
            ax4.set_xlabel("Axial position in m")
            ax4.set_ylabel("Error (%)")
            ax4.set_title("Error in velocity distribution")
            ax4.legend(loc="best")

            plt.show()

        elif compParam == 'frfaccorel':
            fig1, ax1 = plt.subplots()
            for i in range(len(self.caseList)):
                ax1.plot(self.caseList[0].convection_sol.z_mesh, Tw_error[i], label=self.caseList[i].frfaccorel)
            ax1.set_xlabel("Axial position in m")
            ax1.set_ylabel("Error (%)")
            ax1.set_title("Error in temperature distribution")
            ax1.legend(loc="best")

            fig2, ax2 = plt.subplots()
            for i in range(len(self.caseList)):
                ax2.plot(self.caseList[0].convection_sol.z_mesh, voidFraction_error[i], label=self.caseList[i].frfaccorel)
            ax2.set_xlabel("Axial position in m")
            ax2.set_ylabel("Error (%)")
            ax2.set_title("Error in void fraction distribution")
            ax2.legend(loc="best")

            fig3, ax3 = plt.subplots()
            for i in range(len(self.caseList)):
                ax3.plot(self.caseList[0].convection_sol.z_mesh, pressure_error[i], label=self.caseList[i].frfaccorel)
            ax3.set_xlabel("Axial position in m")
            ax3.set_ylabel("Error (%)")
            ax3.set_title("Error in pressure distribution")
            ax3.legend(loc="best")

            fig4, ax4 = plt.subplots()
            for i in range(len(self.caseList)):
                ax4.plot(self.caseList[0].convection_sol.z_mesh, velocity_error[i], label=self.caseList[i].frfaccorel)
            ax4.set_xlabel("Axial position in m")
            ax4.set_ylabel("Error (%)")
            ax4.set_title("Error in velocity distribution")
            ax4.legend(loc="best")

            plt.show()

        elif compParam == 'P2Pcorrel':
            fig1, ax1 = plt.subplots()
            for i in range(len(self.caseList)):
                ax1.plot(self.caseList[0].convection_sol.z_mesh, Tw_error[i], label=self.caseList[i].P2Pcorel)
            ax1.set_xlabel("Axial position in m")
            ax1.set_ylabel("Error (%)")
            ax1.set_title("Error in temperature distribution")
            ax1.legend(loc="best")

            fig2, ax2 = plt.subplots()
            for i in range(len(self.caseList)):
                ax2.plot(self.caseList[0].convection_sol.z_mesh, voidFraction_error[i], label=self.caseList[i].P2Pcorel)
            ax2.set_xlabel("Axial position in m")
            ax2.set_ylabel("Error (%)")
            ax2.set_title("Error in void fraction distribution")
            ax2.legend(loc="best")

            fig3, ax3 = plt.subplots()
            for i in range(len(self.caseList)):
                ax3.plot(self.caseList[0].convection_sol.z_mesh, pressure_error[i], label=self.caseList[i].P2Pcorel)
            ax3.set_xlabel("Axial position in m")
            ax3.set_ylabel("Error (%)")
            ax3.set_title("Error in pressure distribution")
            ax3.legend(loc="best")

            fig4, ax4 = plt.subplots()
            for i in range(len(self.caseList)):
                ax4.plot(self.caseList[0].convection_sol.z_mesh, velocity_error[i], label=self.caseList[i].P2Pcorel)
            ax4.set_xlabel("Axial position in m")
            ax4.set_ylabel("Error (%)")
            ax4.set_title("Error in velocity distribution")
            ax4.legend(loc="best")

            plt.show()

        
        for i in range(len(self.caseList)):
            print(f'Case {i}')
            voidFraction_error[i] = self.cleanList(voidFraction_error[i])
            Tw_error[i] = self.cleanList(Tw_error[i])
            pressure_error[i] = self.cleanList(pressure_error[i])
            velocity_error[i] = self.cleanList(velocity_error[i])
            print(f'mean voidFraction: {voidFraction_error[i]}')
            print(f'Void Fraction error moyenne: {np.mean(voidFraction_error[i])}, erreur max: {np.max(voidFraction_error[i])}')
            print(f'Temperature error moyenne: {np.mean(Tw_error[i])}, erreur max: {np.max(Tw_error[i])}')
            print(f'Pressure error moyenne: {np.mean(pressure_error[i])}, erreur max: {np.max(pressure_error[i])}')
            print(f'Velocity error moyenne: {np.mean(velocity_error[i])}, erreur max: {np.max(velocity_error[i])}')

    def cleanList(self, data):
        # Convert to numpy array if it's not already
        data = np.array(data)
        # Filtrer les NaN et les valeurs infinies
        cleaned_data = data[np.isfinite(data)]
        return cleaned_data


    def plotComparison(self, compParam, visuParam):
        if compParam == 'voidFractionCorrel':
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].voidFractionCorrel)
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title("Temperature distribution in pincell")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].voidFractionCorrel)
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title("Void fraction distribution in coolant canal")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].voidFractionCorrel)
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title("Density distribution in coolant canal")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].voidFractionCorrel)
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title("Pressure distribution in coolant canal")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].voidFractionCorrel)
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title("Velocity distribution in coolant canal")
                ax5.legend(loc="best")

            fig6, ax6 = plt.subplots()
            for i in range(len(self.caseList)):
                ax6.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.get_Fission_Power(), label="Fission power")
            ax6.set_xlabel("Axial position in m")
            ax6.set_ylabel("Fission power in W/m^3")
            ax6.set_title("Fission power distribution in fuel rod")

            
            fig7, ax7 = plt.subplots()
            for i in range(len(self.caseList)):
                ax7.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.H[-1], label=self.caseList[i].voidFractionCorrel)
            ax7.set_xlabel("Axial position in m")
            ax7.set_ylabel("Enthalpy in K")
            ax7.set_title("Enthalpy distribution in pincell")
            ax7.legend(loc="best")

            plt.show()
    
        elif compParam == 'frfaccorel':
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].frfaccorel)
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title("Temperature distribution in pincell")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].frfaccorel)
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title("Void fraction distribution in coolant canal")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].frfaccorel)
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title("Density distribution in coolant canal")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].frfaccorel)
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title("Pressure distribution in coolant canal")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].frfaccorel)
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title("Velocity distribution in coolant canal")
                ax5.legend(loc="best")

            fig6, ax6 = plt.subplots()
            for i in range(len(self.caseList)):
                ax6.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.get_Fission_Power(), label="Fission power")
            ax6.set_xlabel("Axial position in m")
            ax6.set_ylabel("Fission power in W/m^3")
            ax6.set_title("Fission power distribution in fuel rod")

            
            fig7, ax7 = plt.subplots()
            for i in range(len(self.caseList)):
                ax7.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.H[-1], label=self.caseList[i].frfaccorel)
            ax7.set_xlabel("Axial position in m")
            ax7.set_ylabel("Enthalpy in K")
            ax7.set_title("Enthalpy distribution in pincell")
            ax7.legend(loc="best")

            plt.show()
    
        elif compParam == 'P2Pcorrel':
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].P2Pcorel)
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title("Temperature distribution in pincell")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].P2Pcorel)
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title("Void fraction distribution in coolant canal")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].P2Pcorel)
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title("Density distribution in coolant canal")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].P2Pcorel)
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title("Pressure distribution in coolant canal")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].P2Pcorel)
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title("Velocity distribution in coolant canal")
                ax5.legend(loc="best")

            fig6, ax6 = plt.subplots()
            for i in range(len(self.caseList)):
                ax6.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.get_Fission_Power(), label="Fission power")
            ax6.set_xlabel("Axial position in m")
            ax6.set_ylabel("Fission power in W/m^3")
            ax6.set_title("Fission power distribution in fuel rod")

            
            fig7, ax7 = plt.subplots()
            for i in range(len(self.caseList)):
                ax7.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.H[-1], label=self.caseList[i].P2Pcorel)
            ax7.set_xlabel("Axial position in m")
            ax7.set_ylabel("Enthalpy in K")
            ax7.set_title("Enthalpy distribution in pincell")
            ax7.legend(loc="best")

            plt.show()

        elif compParam == 'numericalMethod':
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    print(f'Twater: { self.caseList[i].convection_sol.T_water}')
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].numericalMethod)
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title("Temperature distribution in pincell")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].numericalMethod)
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title("Void fraction distribution in coolant canal")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].numericalMethod)
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title("Density distribution in coolant canal")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].numericalMethod)
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title("Pressure distribution in coolant canal")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].numericalMethod)
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title("Velocity distribution in coolant canal")
                ax5.legend(loc="best")

            if visuParam[5]:
                fig6, ax6 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax6.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.get_Fission_Power(), label="Fission power")
                ax6.set_xlabel("Axial position in m")
                ax6.set_ylabel("Fission power in W/m^3")
                ax6.set_title("Fission power distribution in fuel rod")

            if visuParam[6]:
                fig7, ax7 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax7.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.H[-1], label=self.caseList[i].numericalMethod)
                ax7.set_xlabel("Axial position in m")
                ax7.set_ylabel("Enthalpy in K")
                ax7.set_title("Enthalpy distribution in pincell")
                ax7.legend(loc="best")

            plt.show()

    
    def GenFoamComp(self, GenFoamPathCase, compParam, visuParam, genFoamVolumeFraction):

        # Read the Excel file
        df = pd.read_excel(GenFoamPathCase)

        # Create empty lists for each column
        columns = df.columns.tolist()
        data = [[] for _ in columns]

        # Iterate over each row and append values to the corresponding list
        for index, row in df.iterrows():
            for i, col in enumerate(columns):
                data[i].append(row[col])

        for i in range(len(data[7])):
            data[7][i] = (1/(1-genFoamVolumeFraction)) * data[7][i]

        genfoamCASE = [data[0], data[3], data[7], data[3], data[1], data[5]]
        if compParam == 'voidFractionCorrel':
            title = f"Methode numérique: {self.caseList[0].numericalMethod}, \n Correlation multiplicateur biphasique: {self.caseList[0].convection_sol.P2Pcorel}, \n Correlation facteur de friction: {self.caseList[0].convection_sol.frfaccorel}"
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].voidFractionCorrel)
                ax1.plot(genfoamCASE[0], genfoamCASE[1], label="GenFoam")
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title(title)
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].voidFractionCorrel)
                ax2.plot(genfoamCASE[0], genfoamCASE[2], label="GenFoam")
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title(title)
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].voidFractionCorrel)
                #ax3.plot(genfoamCASE[0], genfoamCASE[3], label="GenFoam")
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title(title)
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].voidFractionCorrel)
                ax4.plot(genfoamCASE[0], genfoamCASE[4], label="GenFoam")
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title(title)
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].voidFractionCorrel)
                ax5.plot(genfoamCASE[0], genfoamCASE[5], label="GenFoam")
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title(title)
                ax5.legend(loc="best")

            plt.show()

        if compParam == 'frfaccorel':
            title = f"Methode numérique: {self.caseList[0].numericalMethod}, \n Correlation multiplicateur biphasique: {self.caseList[0].convection_sol.P2Pcorel}, \n Correlation void fraction: {self.caseList[0].convection_sol.voidFractionCorrel}"
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].frfaccorel)
                ax1.plot(genfoamCASE[0], genfoamCASE[1], label="GenFoam")
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title(f"{title}")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].frfaccorel)
                ax2.plot(genfoamCASE[0], genfoamCASE[2], label="GenFoam")
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title(f"{title}")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].frfaccorel)
                #ax3.plot(genfoamCASE[0], genfoamCASE[3], label="GenFoam")
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title(f"{title}")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].frfaccorel)
                ax4.plot(genfoamCASE[0], genfoamCASE[4], label="GenFoam")
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title(f"{title}")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].frfaccorel)
                ax5.plot(genfoamCASE[0], genfoamCASE[5], label="GenFoam")
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title(f"{title}")
                ax5.legend(loc="best")

            plt.show()

        if compParam == 'numericalMethod':
            title = f"Correlation fracteur de friction: {self.caseList[0].convection_sol.frfaccorel}, \n Correlation multiplicateur biphasique: {self.caseList[0].convection_sol.P2Pcorel}, \n Correlation void fraction: {self.caseList[0].convection_sol.voidFractionCorrel}"
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].numericalMethod)
                ax1.plot(genfoamCASE[0], genfoamCASE[1], label="GenFoam")
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title(f"{title}")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].numericalMethod)
                ax2.plot(genfoamCASE[0], genfoamCASE[2], label="GenFoam")
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title(f"{title}")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].numericalMethod)
                #ax3.plot(genfoamCASE[0], genfoamCASE[3], label="GenFoam")
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title(f"{title}")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].numericalMethod)
                ax4.plot(genfoamCASE[0], genfoamCASE[4], label="GenFoam")
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title(f"{title}")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].numericalMethod)
                ax5.plot(genfoamCASE[0], genfoamCASE[5], label="GenFoam")
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title(f"{title}")
                ax5.legend(loc="best")

            plt.show()

        if compParam == 'P2Pcorrel':
            title = f"Correlation fracteur de friction: {self.caseList[0].convection_sol.frfaccorel}, \n Méthode numérique: {self.caseList[0].convection_sol.numericalMethod}, \n Correlation void fraction: {self.caseList[0].convection_sol.voidFractionCorrel}"
            if visuParam[0]:
                fig1, ax1 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax1.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.T_water, label=self.caseList[i].P2Pcorel)
                ax1.plot(genfoamCASE[0], genfoamCASE[1], label="GenFoam")
                ax1.set_xlabel("Axial position in m")
                ax1.set_ylabel("Temperature in K")
                ax1.set_title(f"{title}")
                ax1.legend(loc="best")

            if visuParam[1]:
                fig2, ax2 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax2.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.voidFraction[-1], label=self.caseList[i].P2Pcorel)
                ax2.plot(genfoamCASE[0], genfoamCASE[2], label="GenFoam")
                ax2.set_xlabel("Axial position in m")
                ax2.set_ylabel("Void fraction")
                ax2.set_title(f"{title}")
                ax2.legend(loc="best")

            if visuParam[2]:
                fig3, ax3 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax3.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.rho[-1], label=self.caseList[i].P2Pcorel)
                #ax3.plot(genfoamCASE[0], genfoamCASE[3], label="GenFoam")
                ax3.set_xlabel("Axial position in m")
                ax3.set_ylabel("Density in kg/m^3")
                ax3.set_title(f"{title}")
                ax3.legend(loc="best")

            if visuParam[3]:
                fig4, ax4 = plt.subplots() 
                for i in range(len(self.caseList)):
                    ax4.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.P[-1], label=self.caseList[i].P2Pcorel)
                ax4.plot(genfoamCASE[0], genfoamCASE[4], label="GenFoam")
                ax4.set_xlabel("Axial position in m")
                ax4.set_ylabel("Pressure in Pa")
                ax4.set_title(f"{title}")
                ax4.legend(loc="best")

            if visuParam[4]:
                fig5, ax5 = plt.subplots()
                for i in range(len(self.caseList)):
                    ax5.plot(self.caseList[i].convection_sol.z_mesh, self.caseList[i].convection_sol.U[-1], label=self.caseList[i].P2Pcorel)
                ax5.plot(genfoamCASE[0], genfoamCASE[5], label="GenFoam")
                ax5.set_xlabel("Axial position in m")
                ax5.set_ylabel("Velocity in m/s")
                ax5.set_title(f"{title}")
                ax5.legend(loc="best")

            plt.show()


# This file contains the classes and functions to create fields and give the values of the variables in the drift flux model reoslution file
# This class models the thermohydraulic behavior of a fluid flow in a system divided into multiple cells. Each cell is characterized by properties such as velocity (U), pressure (P), enthalpy (H), void fraction, and various geometric parameters. 
#The class employs different correlations to compute key thermophysical properties, and performs iterative updates to simulate two-phase flow dynamics.
# The file is used in the BWR/driftFluxModel/thermalHydraulicsTransitoire/THM_convection.py file
#Authors: Clément Huet
#Date: 2024-09-16

import numpy as np
from iapws import IAPWS97


class statesVariables():

    """
    Attributes:
    - nCells (int): Number of cells in the system.
    - U (array): Velocity values for each cell.
    - P (array): Pressure values for each cell.
    - H (array): Enthalpy values for each cell.
    - voidFraction (array): Initial void fraction values for each cell.
    - voidFractionCorrel (str): Correlation model used for void fraction calculations (e.g., 'modBestion', 'HEM1').
    - frfaccorel (str): Correlation model used for friction factor calculations (e.g., 'base', 'blasius', 'Churchill').
    - P2Pcorel (str): Correlation model used for two-phase pressure multiplier (e.g., 'base', 'HEM1').
    - D_h (float): Hydraulic diameter of the system.
    - flowArea (float): Flow area of the system.
    - DV (float): Differential volume between cells.
    - g (float): Gravitational acceleration (9.81 m/s²).
    - K_loss (float): Loss coefficient in the system.
    - Dz (float): Distance between cells.

    Methods:
    - createFields(): Initializes temporary arrays for cell-specific properties such as densities, void fractions, friction factors, and geometric areas. Uses specific methods to populate these arrays.
    - updateFields(): Updates key properties (e.g., void fraction, densities) based on the selected void fraction correlation model. Calls corresponding methods for each model.
    - modBestion(): Updates void fraction and related properties using the "modBestion" correlation with iterative convergence.
    - HEM1(): Updates void fraction and related properties using the "HEM1" correlation with iterative convergence.
    - GEramp(): Updates void fraction and related properties using the "GEramp" correlation with iterative convergence.
    - EPRIvoidModel(): Updates void fraction and related properties using the "EPRIvoidModel" correlation with iterative convergence.
    - getDensity(i): Retrieves the liquid and vapor densities for a given cell using IAPWS97 thermodynamic models.
    - getQuality(i): Calculates the quality (phase fraction) based on enthalpy values.
    - getVoidFraction(i): Computes void fraction using different correlations, depending on the selected model.
    - getVgj(i): Calculates the drift velocity for a given cell based on the selected void fraction correlation.
    - getC0(i): Computes the slip ratio for a given cell based on the selected correlation model.
    - getFrictionFactor(i): Calculates the friction factor based on Reynolds number and the selected friction factor correlation.
    - getPhi2Phi(i): Computes the two-phase pressure multiplier for a given cell.
    - getAreas(i): Calculates the positive and negative flow areas for a given cell.
    - getPhasesEnthalpy(i): Retrieves the enthalpy values for the liquid and vapor phases at a given pressure.
    - getReynoldsNumber(i): Computes the Reynolds number for flow in a given cell.
    """

    def __init__(self, U, P, H, voidFraction, D_h, flowarea, DV, voidFractionCorrel, frfaccorel, P2Pcorel, Dz):
        
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
        self.K_loss = 0#0.17
        self.Dz = Dz
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
        else:
            raise ValueError('Invalid void fraction correlation model')

    
    def modBestion(self):
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
                    raise ValueError('Convergence in update fields not reached')
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
                    raise ValueError('Convergence in update fields not reached')
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
        Re = self.getReynoldsNumber(i)

        if self.frfaccorel == 'base': #Validated
            return 1
        
        elif self.frfaccorel == 'blasius': #Validated
            return 0.316 * Re**(-0.25)
        elif self.frfaccorel == 'Churchill': #Validated
            Ra = 0.4 * (10**(-6)) #Roughness
            R = Ra / self.D_h
            frict=8*(((8.0/Re)**12)+((2.475*np.log(((7/Re)**0.9)+0.27*R))**16+(37530/Re)**16)**(-1.5))**(1/12)
            return frict
        elif self.frfaccorel == 'Churchill_notOK':
            Re = self.getReynoldsNumberLiquid(i)
            B = (37530/Re)**16
            A = (2.475*np.log(1/(((7/Re)**0.9)+0.27*(0.4/self.D_h))))**16
            f  = 8*(((8/Re)**12) + (1/(A+B)**1.5))**(1/12)
            return f
        else:
            raise ValueError('Invalid friction factor correlation model')

        
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
            #print(f'Phi2phi : {phi2phi}')
        else:
            raise ValueError('Invalid two-phase pressure multiplier correlation model')
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
        ml = IAPWS97(P = P*(10**(-6)), x = 0).mu
        mv = IAPWS97(P = P*(10**(-6)), x = 1).mu
        m = (mv * ml) / ( ml * (1 - alpha) + mv * alpha )
        
        return rho * abs(U) * self.D_h / m
    
    def getReynoldsNumberLiquid(self, i):
        Ul = self.getUl(i)
        rho = self.rholTEMP[i]
        P = self.P[i]
        m = IAPWS97(P = P*(10**(-6)), x = 0).mu
        return rho * abs(Ul) * self.D_h / m
    
    def getUl(self, i):
        return self.U[i] + (self.voidFractionTEMP[i] / ( 1- self.voidFractionTEMP[i])) * (self.rholTEMP[i] / self.rhoTEMP[i]) * self.VgjPrimeTEMP[i]
    
    def getUg(self, i):
        return self.U[i] + (self.rholTEMP[i] / self.rhoTEMP[i]) * self.VgjPrimeTEMP[i]

