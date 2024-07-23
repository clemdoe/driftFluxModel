import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt
from FVM import FVM
from classDFM import *

class runDFM():
    def __init__(self, timeSelection, tMax, timeStep, nCells, u_inlet, P_outlet, h_inlet, height, fuelRadius, cladRadius, waterGap):
        self.timeSelection = timeSelection
        self.tMax = tMax
        self.timeStep = timeStep
        self.time = np.arange(0, tMax, timeStep)
        self.dz = height / nCells

        self.nCells = nCells
        self.u_inlet = u_inlet
        self.P_outlet = P_outlet
        self.h_inlet = h_inlet
        self.height = height
        self.fuelRadius = fuelRadius
        self.cladRadius = cladRadius
        self.waterGap = waterGap

        self.rhoG = []
        self.rhoL = []
        self.rho = []
        self.f = []
        self.Vgj = []
        self.VgjPrime = []
        self.C0 = []
        self.U = []
        self.P = []
        self.h = []
        self.voidFraction = []
        self.xTh = []

        self.nombreCourant = (self.timeStep / self.dz)*self.u_inlet
    
    def testCourant(self):
        if self.nombreCourant > 1:
            print('Le nombre de Courant est supérieur à 1, la simulation ne sera pas stable')
        else:
            print('Le nombre de Courant est inférieur à 1, la simulation sera stable')

    def runTransient(self):
        rho_init = IAPWS97(h = self.h_inlet/1000, P = self.P_outlet/1000000).rho
        previousDensity = [rho_init for i in range(self.nCells)]
        previousVelocity = [self.u_inlet for i in range(self.nCells)]
        previousPressure = [self.P_outlet for i in range(self.nCells)]
        previousEnthalpy = [self.h_inlet for i in range(self.nCells)]
        for t in range(len(self.time)):
            self.testCourant()
            timeStep = self.timeStep
            DFM1 = DFMclassTransient(self.nCells, self.u_inlet, self.P_outlet, self.h_inlet, self.height, self.fuelRadius, self.cladRadius, self.waterGap, 'FVM', 'base', 'base', 'GEramp', 150000000, 'sinusoidal', previousDensity, previousVelocity, previousPressure, previousEnthalpy, timeStep)
            DFM1.resolveDFM()

            print(f'Convergence atteinte pour timestep {t}')

            previousDensity = DFM1.rho[-1]
            previousVelocity = DFM1.U[-1]
            previousPressure = DFM1.P[-1]
            previousEnthalpy = DFM1.H[-1]
            print(f'U: {DFM1.U[-1]}, P: {DFM1.P[-1]}, H: {DFM1.H[-1]}, epsilon: {DFM1.voidFraction}, rho: {DFM1.rho}, rhoG: {DFM1.rhoG}, rhoL: {DFM1.rhoL}, xTh: {DFM1.xTh}, f: {DFM1.f}, Vgj: {DFM1.Vgj}, Vgj_prime: {DFM1.VgjPrime}, C0: {DFM1.C0}')
            self.U.append(DFM1.U[-1])
            self.P.append(DFM1.P[-1])
            self.h.append(DFM1.H[-1]/1000)
            self.voidFraction.append(DFM1.voidFraction[-1])
            self.xTh.append(DFM1.xTh[-1])
            self.rho.append(DFM1.rho[-1])
            self.rhoG.append(DFM1.rhoG[-1])
            self.rhoL.append(DFM1.rhoL[-1])
            self.f.append(DFM1.f[-1])
            self.Vgj.append(DFM1.Vgj[-1])
            self.VgjPrime.append(DFM1.VgjPrime[-1])
            self.C0.append(DFM1.C0[-1])

            self.nombreCourant = (self.timeStep / self.dz)*np.mean(DFM1.U[-1])

        self.z = DFM1.zList


    def runSteadyState(self):

        DFM1 = DFMclassSteadyS(self.nCells, self.u_inlet, self.P_outlet, self.h_inlet, self.height, self.fuelRadius, self.cladRadius, self.waterGap, 'FVM', 'base', 'base', 'GEramp', 1500000000, 'sinusoidal')
        DFM1.resolveDFM()
        self.U = DFM1.U[-1]
        self.P = DFM1.P[-1]
        self.h = h = DFM1.H[-1]/1000
        self.voidFraction = DFM1.voidFraction[-1]
        self.xTh = DFM1.xTh[-1]
        self.rho = DFM1.rho[-1]
        self.rhoG = DFM1.rhoG[-1]
        self.rhoL = DFM1.rhoL[-1]
        self.f = DFM1.f[-1]
        self.Vgj = DFM1.Vgj[-1]
        self.VgjPrime = DFM1.VgjPrime[-1]
        self.C0 = DFM1.C0[-1]
        self.z = DFM1.zList

    def main(self):
        if self.timeSelection == 'transient':
            self.runTransient()
        
        elif self.timeSelection == 'steadyState':
            self.runSteadyState()


#Boundary conditions
T_inlet = 602.75 #K
P_inlet = 14.9 * (10**6) #Pa
h_inlet = IAPWS97(T = T_inlet, P = P_inlet * 10**(-6)).h * 1000 #J/kg
u_inlet = 4.68292412 #m/s
P_outlet =  14739394.95 #Pa
nCells = 30

#Geometry parameters
height = 1.655 #m
fuelRadius = 5.6*10**(-3) #External radius of the fuel m
cladRadius = 6.52*10**(-3) #External radius of the clad m
waterGap = 0.5*10**(-3) #Gap between the clad and the water m

transientDFM = runDFM('steadyState', 10, 0.01, 10, u_inlet, P_outlet, h_inlet, height, fuelRadius, cladRadius, waterGap)
transientDFM.main()


fig1 = plt.figure()
plt.plot(transientDFM.z, transientDFM.h, label="Q = 500 MW")
plt.xlabel('z [m]')
plt.ylabel('h [kJ/kg]')

fig2 = plt.figure()
plt.plot(transientDFM.z, transientDFM.U, label="Q = 500 MW")
plt.xlabel('z [m]')
plt.ylabel('U [m/s]')

fig3 = plt.figure()
plt.plot(transientDFM.z, transientDFM.P, label="Q = 500 MW")
plt.xlabel('z [m]')
plt.ylabel('P [Pa]')

fig4 = plt.figure()
plt.plot(transientDFM.z, transientDFM.voidFraction, label="Q = 500 MW")
plt.xlabel('z [m]')
plt.ylabel('Void fraction')
plt.show()